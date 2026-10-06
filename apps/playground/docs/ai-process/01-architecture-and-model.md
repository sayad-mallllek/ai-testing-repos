# 01 — What runs where, and what the model computes

## Start with the smallest useful definition

An agent here is a loop. Python asks a language model what to do. The model either produces an answer or requests an allowed tool. Python executes that tool, adds its result to the conversation, and asks the model again.

The model does not have a PostgreSQL connection. It cannot call a Python function directly. It receives descriptions of tools and produces structured data naming a function and its arguments. Our application decides whether that request is valid and performs the actual operation.

```mermaid
sequenceDiagram
    participant U as User / HTTP client
    participant A as FastAPI + Python agent
    participant M as OpenAI model
    participant D as PostgreSQL
    participant W as DuckDuckGo
    U->>A: POST message with conversation UUID
    A->>D: Lock conversation; load recent turns
    A->>M: Instructions + history + message + tool schemas
    M-->>A: Tool calls as JSON arguments
    A->>D: Read products / order / knowledge chunks
    A->>W: Search if requested
    D-->>A: Rows / retrieved text
    W-->>A: Snippets + URLs
    A->>M: Append tool results and request continuation
    M-->>A: Final answer
    A->>D: Save complete turn; commit
    A-->>U: Answer + transcript + tool events
```

Not every turn uses both PostgreSQL tools and web search. A greeting may finish with one model request. A shopping question may need two or more requests. A tool round can contain multiple calls; we execute them sequentially.

## Why these responsibilities are separate

`agent_routes.py` owns HTTP and the database transaction. `service.py` owns message ordering and the loop. `tools.py` owns validated execution. `llm.py` owns the provider-specific API call. `rag.py` owns retrieval math.

This separation gives you specific questions to ask when debugging:

- Was the HTTP input accepted? Look at the request schema and route.
- Did the model request the right tool? Look at the assistant's `tool_calls`.
- Were the arguments valid? Look at the executor and tool event.
- Did retrieval find useful text? Call `/knowledge/search` directly.
- Did history persist? Look at the database transaction and saved turns.

There is no agent framework making these decisions for you. The complete loop is short enough to read in one sitting.

## An HTTP request is bytes before it is Python

Consider this request body:

```json
{"message":"Find boots below $150"}
```

Your HTTP client encodes those characters into bytes, usually UTF-8. The server receives the request through Uvicorn and the ASGI interface. FastAPI decodes the JSON and constructs `ChatRequest`:

```python
class ChatRequest(InputModel):
    message: str = Field(min_length=1, max_length=8000)
```

The base input model strips outer whitespace and rejects unknown fields. A blank message fails validation with HTTP 422. The body does not allow callers to replace the system prompt, choose arbitrary tools, or supply their own tool-result messages.

FastAPI's dependencies supply an async database session, settings, and a model adapter. These are Python objects. Dependency injection means the route asks for these objects by declaring parameters rather than manually constructing each one inside the route.

## What `await` means

An asynchronous Python function can pause at `await` while waiting for I/O, letting the event loop run other requests. Waiting on PostgreSQL or a model HTTP response is mostly waiting on another process or machine. Keeping that waiting asynchronous allows the server to stay responsive.

`await` does not automatically make CPU work faster. Our small retrieval calculation runs synchronously in Python because the corpus is tiny. DuckDuckGo's library is synchronous, so it runs in a worker thread:

```python
asyncio.to_thread(duckduckgo_search, query, limit, timeout)
```

The thread performs the blocking search; the event loop can continue handling requests. We deliberately avoid concurrent operations on the same `AsyncSession`. One session is a mutable transaction context, not a connection pool you can safely share among unrelated tasks.

## What goes into a model request

The beginning of `run_agent` creates ordinary dictionaries:

```python
messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]
turn_start = len(messages)
messages.append({"role": "user", "content": prompt})
tools = [spec.as_schema() for spec in TOOL_SPECS]
```

`*history` expands a list of earlier messages into this list. `turn_start` remembers where the new turn begins so we can save only the new messages later. `tools` contains names, descriptions, and argument schemas. It does not contain Python function objects or database credentials.

The adapter sends them using the SDK:

```python
result = await self.client.chat.completions.create(
    model=self.settings.openai_model,
    messages=messages,
    tools=tools,
    tool_choice="auto" if allow_tools else "none",
    max_completion_tokens=self.settings.agent_max_output_tokens,
)
```

The SDK serializes this request to JSON and sends it over HTTPS. The API key authenticates the request in an HTTP header. Our database URL stays in the Python process; it is not part of the model request. However, text returned by tools is sent to the model on the next request, so store and conversation content do leave the process.

The example uses the [Chat Completions request interface](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create). We isolate that interface in one file so a later provider change need not rewrite the database or tools.

## Under the hood of an LLM

The following is a simplified conceptual account of a typical autoregressive transformer. It explains the computation you are relying on; it does not claim knowledge of the selected model's private architecture or training details.

### 1. Tokenization

A tokenizer maps text to token IDs. A token can represent a word, part of a word, punctuation, or another sequence of characters. Message roles and tool metadata also have representation overhead. Tokens are not the same as English words, bytes, or characters.

Conceptually:

```text
"waterproof boots" -> [some integer token IDs]
```

Those example words do not have stable IDs across all models. Our app never needs to invent the IDs; the model service handles tokenization. This also explains why the history character limit is only a rough size guard, not exact token accounting.

### 2. Embedding lookup

Each token ID selects a learned vector of numbers. Position information helps the network distinguish the same token in different locations. The vectors are inputs to the model's layers. Their dimensions are learned features, not neat human labels such as “price” or “boot.”

### 3. Attention and feed-forward layers

At a high level, attention lets a token's representation combine information from earlier tokens. A common mathematical picture is:

```text
Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d)) V
```

`Q`, `K`, and `V` are learned transformations of token representations. The dot products measure compatibility. Softmax turns scores into normalized weights. Those weights combine values. Autoregressive masking prevents a position from using future output tokens during generation. Feed-forward layers and other transformations refine the representation.

This is matrix arithmetic. The model is not secretly executing `SELECT * FROM products` while thinking about inventory. It can only use information present in the supplied context or encoded statistically in its learned weights.

### 4. Next-token prediction

The network produces scores for possible next tokens. These scores are converted into a probability distribution, and decoding selects a token. The token is appended and the process repeats until the response ends or reaches a limit.

A model can produce normal prose or a structured function-call response. Tool calling relies on the model being trained to produce the expected structured form from the tool specifications and conversation. Generating that form still does not execute a function.

### 5. Why plausible claims can be wrong

Producing likely text is different from proving a statement against current inventory. The model's weights are not this store's live database. A convincingly written price can still be invented. Database and RAG tools provide evidence from the application's current data.

Instructions asking the model to cite evidence improve behavior, but do not prove every sentence follows from that evidence. The transcript lets you inspect this difference. Tests of our Python protocol establish execution behavior; evaluation of answer correctness is a separate job.

## No training happens during a chat

Sending history or retrieved passages changes the input to a model request. It does not update the model's weights. Saving a turn writes PostgreSQL rows; it does not train the model either.

The application can appear to remember because it resends saved information on later requests. The application can appear to know a new policy because it retrieves that policy and supplies its text. These are contextual behaviors, which are central to both sessions and RAG.

Next: [database and seed](02-database-and-seed.md).
