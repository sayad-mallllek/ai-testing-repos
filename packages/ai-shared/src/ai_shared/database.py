from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from ai_shared.config import get_settings


def create_engine() -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=settings.sql_echo,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )
