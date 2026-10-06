# Learn the store agent, from request to saved answer

This is a working, deliberately small agent for **Trailhead Outfitters**, a fictional outdoor gear store. It has an OpenAI chat model, five Python tools, DuckDuckGo search, PostgreSQL conversations, and retrieval augmented generation (RAG) over store policies and gear guides.

Start with the runnable examples. Then follow the chapters in order. Each chapter starts with the plain-language idea and goes down into the implementation. You do not need to understand every type annotation before running the app.

| Read | You will understand |
| --- | --- |
| [00 — Run it](00-run-it.md) | Setup, HTTP requests, inspecting tools, and adding a knowledge document |
| [01 — The architecture and model](01-architecture-and-model.md) | What an agent is, what an LLM actually computes, and where Python takes control |
| [02 — The database and seed](02-database-and-seed.md) | Tables, UUIDs, cents, SQLAlchemy, migrations, constraints, and repeatable seed data |
| [03 — Tool calling](03-tool-calling.md) | Tool schemas, generated arguments, validation, execution, and the agent loop |
| [04 — Sessions and transactions](04-sessions-and-transactions.md) | Persistent memory, context limits, message ordering, locks, commits, and rollbacks |
| [05 — RAG from the math up](05-rag-from-the-math-up.md) | Ingestion, overlapping chunks, TF-IDF, cosine similarity, context, and citations |
| [06 — Trace a complete request](06-trace-a-request.md) | Every handoff in a shopping question and a follow-up question |
| [07 — Testing and troubleshooting](07-testing-and-troubleshooting.md) | What the tests prove, offline model doubles, network limitations, and debugging |
| [08 — Extend it deliberately](08-extend-it.md) | Adding tools, semantic embeddings, scaling sessions, and choosing the next improvement |
| [09 — Source code companion](09-source-code.md) | Full copies of the implementation files, grouped by responsibility |

## The code you are learning

Paths below are relative to `apps/playground/`:

| File | Responsibility |
| --- | --- |
| `src/ai_playground/main.py` | Construct FastAPI and register routes |
| `src/ai_playground/agent_routes.py` | Validate HTTP requests and coordinate transactions |
| `src/ai_playground/schemas.py` | Define the HTTP input/output shapes |
| `src/ai_playground/config.py` | Read the model key and bounded agent settings |
| `src/ai_playground/llm.py` | Translate between our Python reply objects and the OpenAI SDK |
| `src/ai_playground/service.py` | Run the explicit model → tools → model loop |
| `src/ai_playground/tools.py` | Advertise, validate, and execute the five allowed tools |
| `src/ai_playground/rag.py` | Split documents and retrieve relevant chunks |
| `src/ai_playground/models.py` | Define the database tables |
| `src/ai_playground/database.py` | Create request-scoped async database sessions |
| `src/ai_playground/seed.py` | Insert fictional products, orders, and knowledge |
| `src/ai_playground/knowledge/*.md` | Editable source documents used by the seed |
| `migrations/versions/0002_add_store_knowledge_and_chat_sessions.py` | Create the new tables and indexes |

The source files are authoritative. The full code companion is a snapshot of this implementation; regenerate it or update it when changing code.

## Five ideas to keep separate

- **LLM:** computes a response from the messages supplied to it.
- **Agent:** the Python loop that lets the LLM ask for tools and then continues with their results.
- **Tool:** an ordinary function that Python executes after checking the model's request.
- **Session:** a database record identifying a conversation and its saved turns.
- **RAG:** retrieve relevant text, give that text to the model, and generate an answer grounded in it.

This app uses **lexical TF-IDF RAG**, not neural embeddings. That makes the retrieval implementation small, inspectable, and runnable without an embedding key. The chat model still requires `OPENAI_API_KEY`. We discuss the strengths and limitations of both retrieval approaches in chapters 05 and 08.

The existing `/runs` endpoint still echoes a prompt. It is useful for the original scaffold's database smoke test. Use `/sessions/{id}/messages` for the actual agent.

## Where the external interfaces came from

The tool protocol follows the official [OpenAI function-calling guide](https://developers.openai.com/api/docs/guides/function-calling). The model adapter uses the [Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create); the API also offers Responses, but the explicit message array here makes the conversation protocol easy to see. DuckDuckGo integration follows the [DDGS project documentation](https://github.com/deedy5/ddgs). Async database usage follows [SQLAlchemy's asyncio documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html).

These links describe external interfaces. The design decisions, math walkthroughs, and annotated examples below explain this repository's implementation.
