# 03 — From a function description to an executed tool

## There are two contracts

A tool has a **model-facing contract** and a **Python execution contract**.

The model-facing contract describes the name, purpose, and argument shape. The Python execution contract validates the actual arguments and maps an allowed name to an implementation. Advertising a tool does not, by itself, validate or execute it.

This app exposes exactly these tools:

| Tool | Data source | Output |
| --- | --- | --- |
| `search_products` | Product rows | Bounded list with prices and stock |
| `get_product` | One SKU | A product or a not-found error |
| `get_order` | Order and line-item rows | Status, items, and calculated merchandise total |
| `retrieve_store_knowledge` | Knowledge chunks | Relevant passages, scores, and citation IDs |
| `search_web` | DuckDuckGo | Bounded titles, snippets, and source URLs |

Database tools only read. Document ingestion is a separate HTTP endpoint controlled by the caller; it is not an agent tool.

## Step 1: define the arguments

For a product lookup:

```python
class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

class ProductLookup(ToolArguments):
    sku: str = Field(min_length=1, max_length=40)
```

`extra="forbid"` rejects unexpected keys. `strict=True` avoids silently turning a string into an integer in budget fields. `str_strip_whitespace=True` makes an all-whitespace SKU or search query fail its minimum length check.

For product search, omitted values can take defaults. `limit` defaults to 5 and is restricted to 1–10. `max_price_cents` is an integer or null. `category` is an exact category string, such as `footwear`, `packs`, `shelter`, `sleep`, `clothing`, or `accessories`.

## Step 2: advertise the tool

The schema comes from the same Pydantic model used for runtime validation:

```python
class ToolSpec:
    # Dataclass fields and decorator are shown in the source companion.
    def as_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arguments.model_json_schema(),
            },
        }
```

A simplified product lookup schema looks like:

```json
{
  "type": "function",
  "function": {
    "name": "get_product",
    "description": "Read one store product by exact SKU.",
    "parameters": {
      "type": "object",
      "properties": {"sku":{"type":"string","minLength":1,"maxLength":40}},
      "required": ["sku"],
      "additionalProperties": false
    }
  }
}
```

JSON Schema describes data; it is not executable code. The descriptions tell the model when each tool is relevant. Good descriptions also prevent source confusion: web search cannot know the store's current stock, and a generic gear guide cannot know an order's status.

Pydantic strict validation and provider **strict structured output** are different ideas. This app performs strict local validation but does not set the API's `strict: true` function-schema flag. Optional/default argument fields remain optional for this teaching example. Always validate locally, even if you later opt into provider schema enforcement.

## Step 3: the model requests a function

An assistant reply can have no prose and still be useful:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_boot_1",
      "type": "function",
      "function": {
        "name": "get_product",
        "arguments": "{\"sku\":\"BOOT-001\"}"
      }
    }
  ]
}
```

Notice that `arguments` is a **string containing JSON**, not a Python dictionary yet. The outer response is structured data; the argument string is another serialized payload that must be decoded and validated.

The call ID identifies this specific request, not the tool type. Two product lookups need two call IDs even though both use `get_product`. The ID will connect each result to its request. This message/result structure follows the official [function-calling protocol](https://developers.openai.com/api/docs/guides/function-calling).

## Step 4: validate before execution

The executor's key lines are:

```python
spec = TOOL_BY_NAME.get(call.name)
args = spec.arguments.model_validate_json(call.arguments)
arguments = args.model_dump()
result = await self.dispatch(call.name, args)
```

These lines occur inside branches that handle unknown names and `ValidationError`. The actual file shows the full error handling.

`TOOL_BY_NAME` is an allowlist. A requested name such as `execute_sql`, `delete_products`, or `run_shell` is not present, so it never reaches dispatch. Validating the JSON checks types, lengths, bounds, required fields, and extra keys. A model-requested `limit=1000` cannot trigger an unbounded read.

The executor does not use `eval`, dynamic imports from a model name, or generated SQL. A tool name is matched to fixed application code. This is the meaningful enforcement boundary. The system prompt describes expected behavior; the allowlist constrains what can actually execute.

Recoverable failures become tool-result data:

```json
{"error":"invalid_arguments","details":["...validation details..."]}
```

The model can make a corrected request or explain a limitation. Database transaction failures are handled differently: they abort the whole turn, because continuing on a failed SQL transaction would be misleading.

## Step 5: run ordinary Python and SQL

The exact product lookup is:

```python
async def get_product(session: AsyncSession, args: ProductLookup) -> dict:
    product = await session.scalar(select(Product).where(Product.sku == args.sku))
    return product_data(product) if product else {"error": "product_not_found"}
```

Product search combines predicates. Each search keyword must appear in at least one of the name, description, or SKU fields. The fields are combined with OR for each keyword; successive `.where(...)` calls combine the keyword requirements with AND.

```python
for keyword in args.query.split():
    statement = statement.where(
        or_(
            Product.name.icontains(keyword, autoescape=True),
            Product.description.icontains(keyword, autoescape=True),
            Product.sku.icontains(keyword, autoescape=True),
        )
    )
```

`autoescape=True` treats `%` and `_` in user/model values as literal characters rather than wildcard instructions. Parameter binding handles SQL values separately from SQL structure. These mechanisms solve different problems and both are useful here.

The search is lexical. “Not waterproof” contains the keyword “waterproof”; the tool returns the description so the model can distinguish it. If you need guaranteed attribute filtering, model `waterproof` as a dedicated boolean field and expose a boolean filter. Natural language matching alone does not enforce that property.

## Step 6: return the result to the model

The agent adds a result message:

```python
def tool_result_message(event: dict) -> dict:
    return {
        "role": "tool",
        "tool_call_id": event["tool_call_id"],
        "content": json.dumps(event["result"], ensure_ascii=False),
    }
```

The content is JSON text. The duration and trace metadata are saved for learning, but only the result is inserted into the tool message. The assistant's original call must appear before its matching tool message. For multiple requested calls, the agent supplies one result for each ID before asking the model to continue.

The transcript is then:

```text
system instructions
user asks for a product
assistant requests get_product, ID call_boot_1
tool returns the product, ID call_boot_1
assistant writes the answer
```

The second model request contains the product evidence. The first request did not magically access it.

## The bounded loop

`run_agent` iterates at most `agent_max_tool_rounds + 1` model requests. The extra request is reserved for a final answer with tools disabled. A separate cap limits total executed tool calls:

```python
allow_tools = (
    round_number < settings.agent_max_tool_rounds
    and executed_calls < settings.agent_max_tool_calls
)
reply = await model.complete(messages, tools, allow_tools=allow_tools)
```

A round is a model response that may request tools. A call is one function execution. One round might request three calls. With defaults, the agent allows up to six tool-request rounds and twelve total executions, plus a final answer-only request if needed.

If a response contains more calls than remain in the budget, excess calls receive `tool_budget_exhausted` results. They are not executed, but their result messages preserve the protocol. When tools are disabled, another tool request is a provider error rather than an opportunity to exceed the budget.

We execute calls in sequence. This makes the trace predictable and avoids sharing one `AsyncSession` across concurrent operations. A future implementation can parallelize independent web calls or database work with separate sessions, but that would require explicit transaction and ordering choices.

## DuckDuckGo's thread and failure path

The blocking search call is:

```python
results = DDGS(timeout=timeout).text(
    query, max_results=limit, backend="duckduckgo", safesearch="moderate"
)
```

The backend is pinned to `duckduckgo`, rather than DDGS's automatic choice among search services. The API integration is based on the [DDGS documentation](https://github.com/deedy5/ddgs). We return at most five results, truncate their fields, and keep HTTP(S) source URLs. We do not fetch those pages or treat snippets as verified full articles.

`asyncio.wait_for` bounds the wait on the worker thread. Cancelling the await does not forcefully kill the thread; DDGS's own timeout helps bound its network work. A timeout or DDGS failure returns `search_unavailable` as tool data. Search failure does not turn into made-up results.

Next: [sessions and transactions](04-sessions-and-transactions.md).
