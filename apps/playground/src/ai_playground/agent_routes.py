"""HTTP endpoints for durable conversations, store inspection and knowledge ingestion."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from openai import OpenAIError
from sqlalchemy import func, select
from sqlalchemy.exc import DBAPIError, IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ai_playground.config import AgentSettings, get_agent_settings
from ai_playground.database import get_session
from ai_playground.llm import AgentConfigurationError, AgentProviderError, OpenAIChatModel
from ai_playground.models import ChatSession, ChatTurn, KnowledgeDocument
from ai_playground.rag import ingest_document, retrieve_knowledge
from ai_playground.schemas import (
    ChatRequest,
    DocumentCreate,
    DocumentRead,
    KnowledgeQuery,
    SessionCreate,
    SessionRead,
    TurnRead,
)
from ai_playground.service import bounded_history, run_agent
from ai_playground.tools import (
    ProductLookup,
    ProductSearch,
    ToolExecutor,
    get_product,
    search_products,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["store agent"])
SessionDependency = Annotated[AsyncSession, Depends(get_session)]
SettingsDependency = Annotated[AgentSettings, Depends(get_agent_settings)]


async def get_model(settings: SettingsDependency) -> AsyncIterator[OpenAIChatModel]:
    try:
        model = OpenAIChatModel(settings)
    except AgentConfigurationError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    try:
        yield model
    finally:
        await model.close()


ModelDependency = Annotated[OpenAIChatModel, Depends(get_model)]


@router.post("/sessions", response_model=SessionRead, status_code=201)
async def create_session(payload: SessionCreate, session: SessionDependency):
    conversation = ChatSession(title=payload.title)
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


@router.get("/sessions", response_model=list[SessionRead])
async def list_sessions(
    session: SessionDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    conversations = await session.scalars(
        select(ChatSession)
        .order_by(ChatSession.created_at.desc(), ChatSession.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return conversations.all()


@router.get("/sessions/{session_id}", response_model=SessionRead)
async def read_session(session_id: UUID, session: SessionDependency):
    conversation = await session.get(ChatSession, session_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return conversation


@router.get("/sessions/{session_id}/messages", response_model=list[TurnRead])
async def list_turns(
    session_id: UUID,
    session: SessionDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    if await session.get(ChatSession, session_id) is None:
        raise HTTPException(status_code=404, detail="Session not found")
    turns = await session.scalars(
        select(ChatTurn)
        .where(ChatTurn.session_id == session_id)
        .order_by(ChatTurn.position)
        .limit(limit)
        .offset(offset)
    )
    return turns.all()


@router.post("/sessions/{session_id}/messages", response_model=TurnRead, status_code=201)
async def chat(
    session_id: UUID,
    payload: ChatRequest,
    session: SessionDependency,
    settings: SettingsDependency,
    model: ModelDependency,
):
    try:
        async with asyncio.timeout(settings.agent_timeout_seconds), session.begin():
            # A DB lock works across processes. NOWAIT returns 409 for an overlapping turn.
            conversation = await session.scalar(
                select(ChatSession).where(ChatSession.id == session_id).with_for_update(nowait=True)
            )
            if conversation is None:
                raise HTTPException(status_code=404, detail="Session not found")
            previous = (
                (
                    await session.scalars(
                        select(ChatTurn)
                        .where(ChatTurn.session_id == session_id)
                        .order_by(ChatTurn.position.desc())
                        .limit(settings.agent_history_turns)
                    )
                ).all()
                if settings.agent_history_turns
                else []
            )
            history = bounded_history([turn.messages for turn in reversed(previous)], settings)
            last_position = await session.scalar(
                select(func.max(ChatTurn.position)).where(ChatTurn.session_id == session_id)
            )
            result = await run_agent(
                payload.message, history, model, ToolExecutor(session, settings), settings
            )
            turn = ChatTurn(
                session_id=session_id,
                position=(last_position or 0) + 1,
                prompt=payload.message,
                answer=result.answer,
                messages=result.messages,
                tool_events=result.tool_events,
                usage=result.usage,
            )
            session.add(turn)
            await session.flush()
            # Load server defaults before returning an ORM object from async code.
            await session.refresh(turn)
        return turn
    except TimeoutError as error:
        raise HTTPException(
            status_code=504, detail="Agent turn timed out; no turn was saved."
        ) from error
    except (OpenAIError, AgentProviderError) as error:
        logger.warning("Model request failed", exc_info=True)
        raise HTTPException(
            status_code=502, detail="Model request failed; no turn was saved."
        ) from error
    except DBAPIError as error:
        if getattr(error.orig, "sqlstate", None) == "55P03":
            raise HTTPException(
                status_code=409, detail="This session already has a running turn."
            ) from error
        logger.exception("Agent database request failed")
        raise HTTPException(
            status_code=503, detail="Database request failed; no turn was saved."
        ) from error
    except SQLAlchemyError as error:
        logger.exception("Agent database request failed")
        raise HTTPException(
            status_code=503, detail="Database request failed; no turn was saved."
        ) from error


@router.get("/store/products")
async def list_products(
    session: SessionDependency,
    query: Annotated[str, Query(max_length=200)] = "",
    category: Annotated[str | None, Query(max_length=80)] = None,
    max_price_cents: Annotated[int | None, Query(ge=0, le=10_000_000)] = None,
    in_stock_only: bool = True,
    limit: Annotated[int, Query(ge=1, le=10)] = 5,
):
    return await search_products(
        session,
        ProductSearch(
            query=query,
            category=category,
            max_price_cents=max_price_cents,
            in_stock_only=in_stock_only,
            limit=limit,
        ),
    )


@router.get("/store/products/{sku}")
async def read_product(
    sku: Annotated[str, Path(min_length=1, max_length=40)], session: SessionDependency
):
    result = await get_product(session, ProductLookup(sku=sku))
    if "error" in result:
        raise HTTPException(status_code=404, detail="Product not found")
    return result


@router.get("/knowledge/documents", response_model=list[DocumentRead])
async def list_documents(
    session: SessionDependency,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    documents = await session.scalars(
        select(KnowledgeDocument).order_by(KnowledgeDocument.slug).limit(limit).offset(offset)
    )
    return documents.all()


@router.post("/knowledge/documents", response_model=DocumentRead)
async def upsert_document(payload: DocumentCreate, session: SessionDependency):
    try:
        async with session.begin():
            document = await ingest_document(session, **payload.model_dump())
        return document
    except IntegrityError as error:
        raise HTTPException(
            status_code=409, detail="Document changed concurrently; retry."
        ) from error


@router.post("/knowledge/search")
async def search_knowledge(payload: KnowledgeQuery, session: SessionDependency):
    return await retrieve_knowledge(session, payload.query, payload.limit)
