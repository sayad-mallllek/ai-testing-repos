"""Workspace commands; scaffolding uses only the Python standard library."""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "templates" / "app"
NAME_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


def validate_name(name: str) -> str:
    if not NAME_PATTERN.fullmatch(name) or len(name) > 48:
        raise ValueError("Use a lowercase app name, e.g. rag-demo (maximum 48 characters).")
    return name


def create_app(name: str, root: Path = ROOT) -> Path:
    validate_name(name)
    destination = root / "apps" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f"App already exists: {name}")
    replacements = {
        "__APP_NAME__": name,
        "__MODULE_NAME__": "ai_" + name.replace("-", "_"),
        "__SCHEMA_NAME__": "app_" + name.replace("-", "_"),
    }
    # Render outside the workspace glob so uv never sees a partially created app.
    with tempfile.TemporaryDirectory(prefix=".new-app-", dir=root) as temporary:
        staged = Path(temporary) / name
        for source in sorted(TEMPLATE.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            relative = source.relative_to(TEMPLATE).as_posix()
            content = source.read_text()
            for placeholder, value in replacements.items():
                relative = relative.replace(placeholder, value)
                content = content.replace(placeholder, value)
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
        staged.rename(destination)
    return destination


def run_uv(app: str, command: list[str]) -> int:
    validate_name(app)
    if not (ROOT / "apps" / app / "pyproject.toml").is_file():
        raise ValueError(f"Unknown app: {app}. Create it with make new-app NAME={app}")
    if shutil.which("uv") is None:
        raise ValueError(
            "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
        )
    return subprocess.call(["uv", "run", "--package", f"ai-app-{app}", *command], cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and run independent FastAPI experiments.")
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="Scaffold an app: make new-app NAME=rag-demo")
    new.add_argument("name")
    commands.add_parser("init-env", help="Create .env without overwriting existing settings")
    commands.add_parser("list", help="List apps")
    for name in ("dev", "migrate", "revision", "check"):
        command = commands.add_parser(name)
        command.add_argument("app")
        if name == "dev":
            command.add_argument("--port", type=int, default=8000)
            command.add_argument("--host", default="127.0.0.1")
        if name == "revision":
            command.add_argument("--message", required=True)
    args = parser.parse_args()
    try:
        if args.command == "new":
            create_app(args.name)
            print(f"Created apps/{args.name}")
            print(f"Next: make migrate APP={args.name}")
            print(f"      make dev APP={args.name} PORT=8001")
        elif args.command == "init-env":
            try:
                with (ROOT / ".env").open("x") as target:
                    target.write((ROOT / ".env.example").read_text())
                os.chmod(ROOT / ".env", 0o600)
                print("Created .env")
            except FileExistsError:
                print("Using existing .env")
        elif args.command == "list":
            for path in sorted((ROOT / "apps").glob("*/pyproject.toml")):
                print(path.parent.name)
        elif args.command == "dev":
            if not 1 <= args.port <= 65535:
                raise ValueError("Port must be between 1 and 65535.")
            module = "ai_" + args.app.replace("-", "_")
            return run_uv(
                args.app,
                [
                    "uvicorn",
                    f"{module}.main:app",
                    "--reload",
                    "--reload-dir",
                    str(ROOT / "apps" / args.app / "src"),
                    "--reload-dir",
                    str(ROOT / "packages" / "ai-shared" / "src"),
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                ],
            )
        else:
            if args.command == "revision" and not args.message.strip():
                raise ValueError("Provide a migration message: MESSAGE='add documents'")
            alembic = ["alembic", "-c", str(ROOT / "apps" / args.app / "alembic.ini")]
            if args.command == "migrate":
                alembic += ["upgrade", "head"]
            elif args.command == "revision":
                alembic += ["revision", "--autogenerate", "-m", args.message]
            else:
                alembic += ["check"]
            return run_uv(args.app, alembic)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
