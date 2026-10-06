from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from __MODULE_NAME__.database import engine
from __MODULE_NAME__.models import Run
from __MODULE_NAME__.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        await engine.dispose()


app = FastAPI(title="__APP_NAME__", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok", "app": "__APP_NAME__"}


@app.get("/health/ready", tags=["health"])
async def ready():
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
            await connection.execute(select(Run.id).limit(1))
    except SQLAlchemyError as error:
        raise HTTPException(
            status_code=503, detail="Database unavailable or migrations missing"
        ) from error
    return {"status": "ready", "app": "__APP_NAME__"}
