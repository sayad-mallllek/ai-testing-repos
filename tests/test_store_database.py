"""Opt-in real-PostgreSQL checks. Every test uses a disposable, unique schema."""

import asyncio
import os
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from ai_playground.agent_routes import get_model
from ai_playground.config import AgentSettings, get_agent_settings
from ai_playground.database import get_session
from ai_playground.llm import AgentProviderError, ModelReply, ToolCall
from ai_playground.main import app
from ai_playground.models import (
    SCHEMA,
    Base,
    ChatTurn,
    KnowledgeChunk,
    KnowledgeDocument,
    OrderItem,
    Product,
    StoreOrder,
)
from ai_playground.rag import ingest_document, retrieve_knowledge
from ai_playground.seed import seed_store
from ai_playground.tools import (
    OrderLookup,
    ProductLookup,
    ProductSearch,
    get_order,
    get_product,
    search_products,
)
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.schema import CreateSchema, DropSchema
from test_agent import ScriptedModel

pytestmark = pytest.mark.skipif(
    not os.getenv("AI_LAB_TEST_DATABASE_URL"),
    reason="Set AI_LAB_TEST_DATABASE_URL to run PostgreSQL integration tests",
)


@pytest_asyncio.fixture
async def store_sessions():
    schema = f"test_trailhead_{uuid4().hex[:12]}"
    engine = create_async_engine(
        os.environ["AI_LAB_TEST_DATABASE_URL"],
        execution_options={"schema_translate_map": {SCHEMA: schema}},
    )
    try:
        async with engine.begin() as connection:
            await connection.execute(CreateSchema(schema))
            await connection.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory.begin() as session:
            await seed_store(session)
        yield factory
    finally:
        async with engine.begin() as connection:
            await connection.execute(DropSchema(schema, cascade=True, if_exists=True))
        await engine.dispose()


@pytest_asyncio.fixture
async def client(store_sessions):
    async def override_session():
        async with store_sessions() as session:
            yield session

    app.dependency_overrides[get_session] = override_session
    app.dependency_overrides[get_agent_settings] = lambda: AgentSettings(_env_file=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http:
        yield http
    app.dependency_overrides.clear()


async def test_seed_is_repeatable_and_preserves_edits(store_sessions):
    async with store_sessions.begin() as session:
        product = await session.scalar(select(Product).where(Product.sku == "BOOT-001"))
        product.stock = 3
        await session.flush()
        await seed_store(session)
    async with store_sessions() as session:
        for model, count in [
            (Product, 12),
            (StoreOrder, 3),
            (OrderItem, 6),
            (KnowledgeDocument, 6),
            (KnowledgeChunk, 11),
        ]:
            assert await session.scalar(select(func.count()).select_from(model)) == count
        assert (await get_product(session, ProductLookup(sku="BOOT-001")))["stock"] == 3


async def test_store_filters_stock_budget_and_literal_sql_characters(store_sessions):
    async with store_sessions() as session:
        result = await search_products(
            session, ProductSearch(category="footwear", max_price_cents=9000)
        )
        assert [item["sku"] for item in result["products"]] == ["BOOT-002"]
        assert (await search_products(session, ProductSearch(query="%")))["products"] == []
        assert (await search_products(session, ProductSearch(query="' OR 1=1 --")))[
            "products"
        ] == []
        assert (await get_product(session, ProductLookup(sku="SHOE-001")))["stock"] == 0
        order = await get_order(session, OrderLookup(reference="TRAIL-1001"))
        assert order["status"] == "shipped"
        assert order["total_cents"] == 18997
        assert (await get_order(session, OrderLookup(reference="missing")))[
            "error"
        ] == "order_not_found"


async def test_documents_can_be_replaced_without_duplicate_or_stale_chunks(store_sessions):
    async with store_sessions.begin() as session:
        result = await retrieve_knowledge(session, "returns unused gear", 3)
        assert result["matches"][0]["citation"].startswith("store:returns#")
        document = await ingest_document(
            session, slug="test-guide", title="Zebra guide", content="zebra " * 200, source="manual"
        )
        document_id = document.id
    async with store_sessions.begin() as session:
        replacement = await ingest_document(
            session,
            slug="test-guide",
            title="Zebra guide",
            content="zebra stripes",
            source="manual",
        )
        assert replacement.id == document_id
        chunks = (
            await session.scalars(
                select(KnowledgeChunk).where(KnowledgeChunk.document_id == document_id)
            )
        ).all()
        assert len(chunks) == 1
        assert chunks[0].content == "zebra stripes"


async def test_api_persists_transcript_and_loads_history_only_for_its_session(client):
    model = ScriptedModel(
        [
            ModelReply(tool_calls=[ToolCall("call_boot", "get_product", '{"sku":"BOOT-001"}')]),
            ModelReply(content="Ridge boots cost $129.99."),
            ModelReply(content="You asked about Ridge boots."),
            ModelReply(content="This conversation is new."),
        ]
    )
    app.dependency_overrides[get_model] = lambda: model
    first = (await client.post("/sessions", json={"title": "Boot shopping"})).json()
    second = (await client.post("/sessions", json={})).json()
    response = await client.post(
        f"/sessions/{first['id']}/messages", json={"message": "Price of BOOT-001?"}
    )
    assert response.status_code == 201, response.text
    turn = response.json()
    assert turn["position"] == 1
    assert turn["tool_events"][0]["result"]["price_cents"] == 12999
    assert (
        await client.post(f"/sessions/{first['id']}/messages", json={"message": "What did I ask?"})
    ).status_code == 201
    assert (
        await client.post(f"/sessions/{second['id']}/messages", json={"message": "Hello"})
    ).status_code == 201
    first_history = (await client.get(f"/sessions/{first['id']}/messages")).json()
    assert [item["position"] for item in first_history] == [1, 2]
    assert first_history[0] == turn
    assert any(m.get("content") == "Price of BOOT-001?" for m in model.requests[2]["messages"])
    assert not any(m.get("content") == "Price of BOOT-001?" for m in model.requests[3]["messages"])


async def test_rag_tool_returns_real_citations_to_model(client):
    model = ScriptedModel(
        [
            ModelReply(
                tool_calls=[
                    ToolCall(
                        "call_policy", "retrieve_store_knowledge", '{"query":"returns unused gear"}'
                    )
                ]
            ),
            ModelReply(content="Unused gear has a 30 day window [store:returns#0]."),
        ]
    )
    app.dependency_overrides[get_model] = lambda: model
    conversation = (await client.post("/sessions", json={})).json()
    response = await client.post(
        f"/sessions/{conversation['id']}/messages", json={"message": "What are your returns?"}
    )
    assert response.status_code == 201
    matches = response.json()["tool_events"][0]["result"]["matches"]
    assert matches[0]["citation"] == "store:returns#0"
    assert "30 days" in matches[0]["content"]


async def test_failed_model_does_not_save_a_partial_turn_and_can_be_retried(client, store_sessions):
    class FailingModel:
        async def complete(self, *args, **kwargs):
            raise AgentProviderError("failure")

    app.dependency_overrides[get_model] = lambda: FailingModel()
    conversation = (await client.post("/sessions", json={})).json()
    path = f"/sessions/{conversation['id']}/messages"
    assert (await client.post(path, json={"message": "hello"})).status_code == 502
    async with store_sessions() as session:
        assert await session.scalar(select(func.count()).select_from(ChatTurn)) == 0
    app.dependency_overrides[get_model] = lambda: ScriptedModel(
        [ModelReply(content="Retry worked")]
    )
    response = await client.post(path, json={"message": "hello"})
    assert response.status_code == 201
    assert response.json()["position"] == 1


async def test_concurrent_messages_in_one_session_get_409_without_corrupting_history(client):
    entered, release = asyncio.Event(), asyncio.Event()

    class PausingModel:
        async def complete(self, *args, **kwargs):
            entered.set()
            await release.wait()
            return ModelReply(content="Saved once")

    app.dependency_overrides[get_model] = lambda: PausingModel()
    conversation = (await client.post("/sessions", json={})).json()
    path = f"/sessions/{conversation['id']}/messages"
    task = asyncio.create_task(client.post(path, json={"message": "first"}))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        conflict = await client.post(path, json={"message": "second"})
        assert conflict.status_code == 409, conflict.text
    finally:
        release.set()
        response = await task
    assert response.status_code == 201
    assert len((await client.get(path)).json()) == 1


async def test_turn_timeout_rolls_back_and_releases_the_session_lock(client):
    class SlowModel:
        async def complete(self, *args, **kwargs):
            await asyncio.sleep(2)

    app.dependency_overrides[get_agent_settings] = lambda: AgentSettings(
        _env_file=None, agent_timeout_seconds=0.05
    )
    app.dependency_overrides[get_model] = lambda: SlowModel()
    conversation = (await client.post("/sessions", json={})).json()
    path = f"/sessions/{conversation['id']}/messages"
    assert (await client.post(path, json={"message": "hello"})).status_code == 504
    assert (await client.get(path)).json() == []
    app.dependency_overrides[get_model] = lambda: ScriptedModel([ModelReply(content="ready")])
    assert (await client.post(path, json={"message": "hello"})).status_code == 201


async def test_http_validation_and_manual_rag_work_without_a_model(client):
    assert (await client.post("/sessions", json={"title": "   "})).status_code == 422
    assert (await client.post("/knowledge/search", json={"query": " "})).status_code == 422
    assert (await client.get("/store/products?limit=0")).status_code == 422
    assert (await client.get("/store/products/SHOE-001")).json()["stock"] == 0
    assert (await client.get("/store/products/missing")).status_code == 404
    unknown = str(uuid4())
    assert (await client.get(f"/sessions/{unknown}/messages")).status_code == 404
    search = await client.post("/knowledge/search", json={"query": "shipping delivery"})
    assert search.status_code == 200
    assert search.json()["matches"]
    for content in ("zebra information", "zebra updated"):
        response = await client.post(
            "/knowledge/documents",
            json={"slug": "test-policy", "title": "Test policy", "content": content},
        )
        assert response.status_code == 200
    assert (await client.get("/knowledge/documents")).status_code == 200
    conversation = (await client.post("/sessions", json={})).json()
    assert UUID(conversation["id"])
    assert (await client.get(f"/sessions/{conversation['id']}")).status_code == 200
    assert (await client.get("/sessions")).status_code == 200
