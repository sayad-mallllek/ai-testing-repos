"""Opt-in PostgreSQL integration test with two generated, disposable app schemas."""

import importlib
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from ai_shared.config import get_settings
from ai_shared.database import create_engine
from httpx import ASGITransport, AsyncClient
from sqlalchemy.schema import DropSchema
from test_generator import manage


@pytest.mark.skipif(
    not os.getenv("AI_LAB_TEST_DATABASE_URL"),
    reason="Set AI_LAB_TEST_DATABASE_URL to run PostgreSQL integration tests",
)
async def test_generated_apps_have_independent_data_and_migrations(tmp_path, monkeypatch):
    database_url = os.environ["AI_LAB_TEST_DATABASE_URL"]
    monkeypatch.setenv("DATABASE_URL", database_url)
    get_settings.cache_clear()
    token = uuid4().hex[:10]
    names = [f"integration-{token}-a", f"integration-{token}-b"]
    paths = [manage.create_app(name, root=tmp_path) for name in names]
    sources = [str(path / "src") for path in paths]
    for source in sources:
        monkeypatch.syspath_prepend(source)
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([*sources, *sys.path])}
    modules = []
    cleanup_engine = create_engine()

    def alembic(path: Path, *args: str):
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "-c", str(path / "alembic.ini"), *args],
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    try:
        for name, path in zip(names, paths, strict=True):
            alembic(path, "upgrade", "head")
            modules.append(importlib.import_module(f"ai_{name.replace('-', '_')}.main"))
        # Both migration histories stay clean after the second app creates its tables.
        for path in paths:
            alembic(path, "check")
        # Autogenerate a real change while another app's schema is present.
        models_path = paths[0] / "src" / f"ai_{names[0].replace('-', '_')}" / "models.py"
        with models_path.open("a") as source:
            source.write("\n    notes: Mapped[str | None] = mapped_column(Text, nullable=True)\n")
        alembic(paths[0], "revision", "--autogenerate", "-m", "add notes")
        alembic(paths[0], "upgrade", "head")
        for path in paths:
            alembic(path, "check")
        async with (
            AsyncClient(transport=ASGITransport(app=modules[0].app), base_url="http://a") as a,
            AsyncClient(transport=ASGITransport(app=modules[1].app), base_url="http://b") as b,
        ):
            assert (await a.get("/health/ready")).status_code == 200
            assert (await b.get("/health/ready")).status_code == 200
            assert (await a.post("/runs", json={"prompt": ""})).status_code == 422
            assert (await a.get("/runs?limit=0")).status_code == 422
            saved = await a.post("/runs", json={"prompt": "hello", "parameters": {"k": 3}})
            assert saved.status_code == 201
            run = saved.json()
            assert run["output"] == {"text": "hello", "provider": "local-echo"}
            assert run["parameters"] == {"k": 3}
            assert (await a.get(f"/runs/{run['id']}")).json() == run
            assert (await a.get("/runs")).json() == [run]
            assert (await b.get("/runs")).json() == []
            assert (await b.get(f"/runs/{run['id']}")).status_code == 404
            alembic(paths[0], "downgrade", "base")
            assert (await a.get("/health/ready")).status_code == 503
            assert (await b.get("/health/ready")).status_code == 200
            alembic(paths[0], "upgrade", "head")
            assert (await a.get("/health/ready")).status_code == 200
    finally:
        for module in modules:
            await module.engine.dispose()
        async with cleanup_engine.begin() as connection:
            for name in names:
                await connection.execute(
                    DropSchema("app_" + name.replace("-", "_"), cascade=True, if_exists=True)
                )
        await cleanup_engine.dispose()
        get_settings.cache_clear()
