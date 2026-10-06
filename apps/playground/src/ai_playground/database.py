from ai_shared.database import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker

engine = create_engine()
Session = async_sessionmaker(engine, expire_on_commit=False)


async def get_session():
    async with Session() as session:
        yield session
