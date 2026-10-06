from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from ai_playground.agent_routes import router as agent_router
from ai_playground.database import engine
from ai_playground.models import ChatTurn, KnowledgeChunk, Product, Run
from ai_playground.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        await engine.dispose()


app = FastAPI(title="Trailhead store agent playground", version="0.2.0", lifespan=lifespan)
app.include_router(router)
app.include_router(agent_router)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "app": "playground"}


@app.get("/health/ready", tags=["health"])
async def ready():
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
            for model in (Run, Product, KnowledgeChunk, ChatTurn):
                await connection.execute(select(model.id).limit(1))
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503, detail="Database unavailable or migrations missing"
        ) from error
    return {"status": "ready", "app": "playground"}
