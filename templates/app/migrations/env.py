import asyncio
from importlib import import_module

from ai_shared.config import get_settings
from ai_shared.database import create_engine
from alembic import context
from sqlalchemy.schema import CreateSchema

# Load this app's models without coupling shared migration code to a package name.
models = import_module("__MODULE_NAME__.models")
SCHEMA = models.SCHEMA
Base = models.Base


def include_name(name, type_, parent_names):
    # Autogenerate must never suggest dropping another experiment's tables.
    if type_ == "schema":
        return name == SCHEMA
    return True


def configure(connection=None):
    context.configure(
        connection=connection,
        url=get_settings().database_url if connection is None else None,
        target_metadata=Base.metadata,
        include_schemas=True,
        include_name=include_name,
        version_table_schema=SCHEMA,
        compare_type=True,
        literal_binds=connection is None,
    )


def migrate(connection):
    connection.execute(CreateSchema(SCHEMA, if_not_exists=True))
    connection.commit()
    configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_online():
    engine = create_engine()
    try:
        async with engine.connect() as connection:
            await connection.run_sync(migrate)
    finally:
        await engine.dispose()


if context.is_offline_mode():
    configure()
    with context.begin_transaction():
        context.execute(CreateSchema(SCHEMA, if_not_exists=True))
        context.run_migrations()
else:
    asyncio.run(run_online())
