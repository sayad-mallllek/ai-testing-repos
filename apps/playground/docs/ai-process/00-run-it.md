# 00 — Run the agent before studying it

## What you need

Use Python 3.12 or later, uv, and the existing PostgreSQL Compose service. Run the commands from the **repository root**, not from this documentation folder:

```bash
make setup
make db-up
make migrate APP=playground
make seed
make dev APP=playground PORT=8000
```

`make setup` installs the workspace and creates `.env` only if it does not already exist. `make migrate` applies the database schema. `make seed` inserts fictional store records and chunks the guides. `make dev` starts the HTTP server. Keep that terminal running and use another terminal for the requests below.

Open <http://localhost:8000/docs> for Swagger UI. You can issue all the following requests there without writing a frontend.

### Enable the model

Add or set these values in your local `.env`:

```dotenv
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-4.1-mini
```

Do not commit the real key. `.env` is already ignored by Git. The default model is configurable; changing it requires a model that supports the Chat Completions function-call interface used in `llm.py`. The app does not silently switch providers.

If `.env` already existed, `make setup` does not append the new agent settings. Add the key yourself; the other settings have defaults. Restart the API after changing `.env`: the settings function is cached in the running process. A ChatGPT subscription is not an API key.

The seed, store inspection, knowledge ingestion, and knowledge search all work without a model key. Only the chat endpoint constructs a live model client.

## Inspect the store and retrieval first

```bash
curl 'http://localhost:8000/store/products?category=footwear&max_price_cents=15000'
curl http://localhost:8000/store/products/BOOT-001
curl http://localhost:8000/store/products/SHOE-001

curl -X POST http://localhost:8000/knowledge/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"returns boots worn outdoors","limit":3}'
```

`15000` means $150.00. Listing products defaults to in-stock items, while an exact SKU lookup can show an item with zero stock. `SHOE-001` is deliberately out of stock.

A knowledge search returns an object shaped like this; scores and ordering depend on the query and current documents:

```json
{
  "query": "returns boots worn outdoors",
  "method": "tf-idf-cosine",
  "matches": [
    {
      "citation": "store:returns#0",
      "title": "Returns and exchanges",
      "source": "store://returns",
      "content": "...the actual stored policy chunk...",
      "score": 0.248408
    }
  ]
}
```

This request uses the same retrieval function as the agent's RAG tool. It lets you inspect retrieval independently of the model's decision to call a tool.

## Start a conversation

```bash
curl -X POST http://localhost:8000/sessions \
  -H 'Content-Type: application/json' \
  -d '{"title":"First hiking trip"}'
```

The response contains a UUID `id`. Substitute that value for `YOUR_SESSION_ID` below:

```bash
curl -X POST http://localhost:8000/sessions/YOUR_SESSION_ID/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":"Find waterproof hiking boots below $150. Explain the fit advice and return policy, with sources."}'
```

The expected behavior is a product lookup and knowledge retrieval followed by a grounded answer. The exact wording and tool sequence are model decisions, so they can vary. A product description containing “not waterproof” can match the word “waterproof”; the model must read the result rather than treating a keyword match as proof of the property.

Try a follow-up in the same session:

```bash
curl -X POST http://localhost:8000/sessions/YOUR_SESSION_ID/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":"Can I return them after wearing them outdoors?"}'
```

Now the model receives recent complete turns from that conversation. It can connect “them” to the boots and formulate a standalone policy search.

Try the order and web tools:

```bash
curl -X POST http://localhost:8000/sessions/YOUR_SESSION_ID/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the status and total of order TRAIL-1001?"}'

curl -X POST http://localhost:8000/sessions/YOUR_SESSION_ID/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":"Search DuckDuckGo for recent public advice about maintaining waterproof hiking boots. Include source URLs."}'
```

Orders are fictional. `TRAIL-1001` is shipped and its merchandise total is $189.97. DuckDuckGo can be unavailable or rate limited; the tool returns an error to the agent so it can explain the missing evidence.

## Inspect what happened

```bash
curl http://localhost:8000/sessions
curl http://localhost:8000/sessions/YOUR_SESSION_ID
curl 'http://localhost:8000/sessions/YOUR_SESSION_ID/messages?limit=20&offset=0'
```

Each turn includes:

- `prompt` and `answer`: the user-facing conversation.
- `messages`: the complete user/assistant/tool message sequence for that turn.
- `tool_events`: names, validated arguments, results, call IDs, and measured tool durations.
- `usage`: summed model token usage across the turn's model requests.
- `position`: the conversation's sequence number, starting at 1.

This is an execution trace. It does not expose hidden model reasoning. It shows the actions requested and the evidence actually returned.

## Add your own knowledge

```bash
curl -X POST http://localhost:8000/knowledge/documents \
  -H 'Content-Type: application/json' \
  -d '{"slug":"water-bottle-care","title":"Water bottle care","source":"manual","content":"Wash reusable bottles with mild soap and warm water after each trip. Let the bottle dry with the cap removed. Follow the manufacturer instructions before using a dishwasher."}'

curl -X POST http://localhost:8000/knowledge/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"water bottle washing care"}'
```

Posting the same slug again replaces that document's content and chunks in a transaction. No model call or model retraining is needed. `source` is a label for provenance; ingestion does not fetch a URL supplied there. The documents endpoint accepts text, not PDFs or arbitrary file uploads.

Editing a seeded `.md` file and rerunning the seed preserves the existing database version. To apply an intentional change, use this ingestion endpoint or call `ingest_document` directly. This behavior prevents a seed rerun from overwriting edits.

## Endpoint reference

| Method and path | Purpose |
| --- | --- |
| `POST /sessions` | Create a conversation |
| `GET /sessions` | List conversations with pagination |
| `GET /sessions/{id}` | Read a conversation's metadata |
| `POST /sessions/{id}/messages` | Run and save one agent turn |
| `GET /sessions/{id}/messages` | Read turns in chronological order with pagination |
| `GET /store/products` | Inspect product search/filter results |
| `GET /store/products/{sku}` | Inspect a product by SKU |
| `GET /knowledge/documents` | List source documents with pagination |
| `POST /knowledge/documents` | Insert or replace a document and its chunks |
| `POST /knowledge/search` | Run retrieval without an LLM |
| `GET /health` | Check the HTTP process |
| `GET /health/ready` | Check PostgreSQL and representative migrated tables |
| `POST /runs`, `GET /runs`, `GET /runs/{id}` | Original echo experiment endpoints |

This is a local learning API with no user authentication. A conversation UUID identifies data; it is not an authorization mechanism. Use fictional information while learning. Chapter 08 explains the concrete changes needed for multiple authenticated users.

Next: [architecture and the model](01-architecture-and-model.md).
