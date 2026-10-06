# 04 — Persistent conversations without pretending the model remembers

## Two different kinds of session

A `ChatSession` identifies a conversation. It persists in PostgreSQL and can survive process restarts. An SQLAlchemy `AsyncSession` is a short-lived Python unit of work used to talk to the database. These concepts happen to share a word, but they have different lifetimes and jobs.

When you create a conversation, the app inserts a `ChatSession` row with a UUID, a title, and a timestamp. Posting a message with that UUID connects the new turn to that conversation.

The title is metadata. It is not automatically included in the model prompt. Naming a session “Vegetarian gear recommendations” does not impose a model instruction; send the preference in a message if you want the model to use it.

## What one turn stores

A `ChatTurn` stores one successfully completed user request and all its model/tool exchanges:

```python
turn = ChatTurn(
    session_id=session_id,
    position=(last_position or 0) + 1,
    prompt=payload.message,
    answer=result.answer,
    messages=result.messages,
    tool_events=result.tool_events,
    usage=result.usage,
)
```

The `messages` array contains the user message, any assistant function calls, each matching tool result, and the final assistant answer. `tool_events` also records validated arguments and timings in a shape convenient to inspect.

JSONB stores nested arrays and objects without requiring one SQL table per tool-result field. The fixed relational fields still make it easy to filter turns by session and sequence. This intentionally duplicates some content between the transcript and inspection trace; it favors clear learning over minimum storage.

The `position` integer is more reliable for conversation order than comparing timestamps. Two timestamps can be close or equal. A unique `(session_id, position)` constraint prevents duplicate sequence positions.

## The model receives memory as input

The chat route loads recent turns for just this conversation:

```python
select(ChatTurn)
.where(ChatTurn.session_id == session_id)
.order_by(ChatTurn.position.desc())
.limit(settings.agent_history_turns)
```

The database returns newest turns first. The route reverses them before building history so the model sees chronological order. Turns from another session are never included in this query.

Then `run_agent` prepends the current system instructions, includes selected historical messages, and appends the new user message. Each Chat Completions request supplies the required context explicitly. The model does not receive our session UUID as a magic memory handle.

Closing and reopening the server does not erase PostgreSQL history. However, a fact outside the selected history window is no longer available to the model unless a new message or tool result supplies it.

## Why history is bounded

LLM inputs have finite context limits, and larger inputs increase latency and token usage. Saving every turn does not mean resending every turn forever.

Defaults are eight previous turns and at most 60,000 serialized history characters. The code is:

```python
def bounded_history(turn_messages: list[list[dict]], settings: AgentSettings) -> list[dict]:
    selected = []
    remaining = settings.agent_history_chars
    turns = turn_messages[-settings.agent_history_turns :] if settings.agent_history_turns else []
    for messages in reversed(turns):
        size = len(json.dumps(messages, ensure_ascii=False))
        if size > remaining:
            break
        selected.append(messages)
        remaining -= size
    return [message for messages in reversed(selected) for message in messages]
```

Read this from the inside outward:

1. Consider only the allowed number of newest turns.
2. Visit those turns from newest to oldest.
3. Count each full serialized transcript's characters.
4. Keep it only if the whole turn fits.
5. Stop at the first oversized turn, preserving a contiguous suffix of conversation history.
6. Reverse the selected turns back into chronological order and flatten their message arrays.

If even the newest turn is too large, none of the previous turns are included. Setting `AGENT_HISTORY_TURNS=0` disables history while still saving new turns.

This is not token-exact budgeting. System instructions, tool schemas, the new user message, and current-turn tool results are outside the history character cap. The output token cap also applies to each model request, not the combined output of a whole turn. A larger application should calculate token budgets for the entire request and reserve output space explicitly.

## Why complete turns matter

Imagine cutting a transcript at an arbitrary message count:

```text
assistant: requests tool, call ID X
-------------------- cut here --------------------
tool: result for call ID X
assistant: answer
```

If you send the bottom section alone, the tool result refers to a call the model request does not contain. It is an orphaned result. By retaining or dropping entire turns, we preserve the assistant-call/tool-result relationship.

Budget-exhaustion system messages are excluded from the saved turn. They applied to one invocation and should not tell the model that tools are exhausted during the next invocation. The current main system prompt is supplied afresh each time.

Stored tool evidence can become stale. The system prompt asks the model to re-query current prices, stock, and policies when needed. That instruction improves behavior but does not force a re-query mechanically. If freshness becomes a hard requirement, change the orchestration to inject fresh results or reject unsupported claims.

## Transaction lifecycle

The chat route encloses the turn in both a deadline and a database transaction:

```python
async with asyncio.timeout(settings.agent_timeout_seconds), session.begin():
    # Lock, load history, run agent, insert turn, flush, refresh.
```

Entering `session.begin()` establishes the unit of work. Exiting normally commits. Exiting because an exception occurred rolls back. The Python context manager is why many error branches do not need to repeat `await session.rollback()`.

A `flush` sends pending writes to PostgreSQL within the transaction. It is not a commit. The route refreshes the new turn while the transaction is active so generated IDs and timestamps are loaded before FastAPI builds the response.

The database becomes durable when the transaction commits. A model answer sitting only in Python memory is not yet a saved chat turn.

## What happens on failure

If the model times out after a product lookup, no partial `ChatTurn` is committed. The HTTP response explains the failure, and a retry can start again at the next valid position. If the provider fails, the same rollback rule applies. Read-only tools make this simple: there is no order placement to reverse.

A database rollback cannot undo the external model request or any tokens already used. This is not a distributed transaction spanning PostgreSQL and OpenAI. Likewise, if the server commits but the response is lost on the network, a blind retry can create a second turn. We do not implement idempotency keys for HTTP retries in this basic version.

An exception from DuckDuckGo is usually different: it becomes a tool error, and the model can complete an honest answer explaining that web evidence is unavailable. If the model completes, that successful turn, including its failed tool event, is saved.

## Concurrent messages and row locks

Two requests can arrive for the same conversation before either finishes. Without coordination, both could load the same history and choose the same next position. Even if a unique constraint catches the second write, the requests would have reasoned over conflicting histories.

We lock the conversation row before reading its turns:

```python
select(ChatSession)
.where(ChatSession.id == session_id)
.with_for_update(nowait=True)
```

Conceptually PostgreSQL executes:

```sql
SELECT ...
FROM app_playground.chat_sessions
WHERE id = :session_id
FOR UPDATE NOWAIT;
```

`FOR UPDATE` acquires a row lock until the transaction ends. `NOWAIT` asks PostgreSQL to report a conflict immediately instead of waiting behind another turn. PostgreSQL's lock-unavailable SQLSTATE is `55P03`; the route maps that case to HTTP 409.

Different session rows can be locked independently, so separate conversations can run concurrently. This works across multiple API processes because the lock lives in PostgreSQL, unlike a single-process Python dictionary of locks.

The conversation lock does not lock product rows or turn the whole database into a fixed snapshot. Under ordinary PostgreSQL READ COMMITTED behavior, later statements can see other transactions' newly committed store changes. The lock's purpose is conversation ordering.

## Why holding the transaction is a tradeoff

The implementation keeps the transaction and its connection open while waiting for model and tool calls. That makes ordering and rollback easy to follow. It also occupies a database connection for the duration of a turn and can limit throughput if many conversations run at once.

The whole-turn timeout bounds this work. At larger scale, use a durable “turn running” state, jobs, lease expiry, and a shorter final commit, as described in chapter 08. Those mechanisms add failure states and recovery logic, so they are intentionally beyond this basic implementation.

Next: [RAG from the math up](05-rag-from-the-math-up.md).
