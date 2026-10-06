import importlib.util
import py_compile
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("manage", ROOT / "scripts" / "manage.py")
manage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manage)


def test_new_app_is_a_complete_workspace_package(tmp_path):
    app = manage.create_app("rag-demo", root=tmp_path)
    metadata = tomllib.loads((app / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "ai-app-rag-demo"
    assert metadata["tool"]["uv"]["sources"]["ai-shared"] == {"workspace": True}
    assert (app / "src" / "ai_rag_demo" / "main.py").is_file()
    assert 'SCHEMA = "app_rag_demo"' in (app / "src" / "ai_rag_demo" / "models.py").read_text()
    for path in app.rglob("*"):
        if path.is_file():
            assert "__APP_NAME__" not in path.read_text()
            assert "__MODULE_NAME__" not in path.read_text()
            assert "__SCHEMA_NAME__" not in path.read_text()
    for path in app.rglob("*.py"):
        py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("name", ["", "../escape", "BadName", "foo_bar", "a--b", "x" * 49])
def test_invalid_names_cannot_write_files(tmp_path, name):
    with pytest.raises(ValueError):
        manage.create_app(name, root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_existing_app_is_never_overwritten(tmp_path):
    app = manage.create_app("rag-demo", root=tmp_path)
    marker = app / "README.md"
    marker.write_text("user changes")
    with pytest.raises(ValueError, match="already exists"):
        manage.create_app("rag-demo", root=tmp_path)
    assert marker.read_text() == "user changes"
