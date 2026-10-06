# AI testing monorepo

Independent FastAPI apps for trying chatbots, RAG, agents, classification, and other
AI ideas. Apps share PostgreSQL and infrastructure code, while each app owns its
models, routes, experiment logic, database schema, and migrations.

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/),
Docker with Compose, and Make. uv uses Python 3.13 by default and can download it.
Run commands from this directory:

```bash
make setup
make db-up
make migrate
make dev
```

The included `playground` app serves http://localhost:8000/docs. Try an experiment:

```bash
curl -X POST http://localhost:8000/runs \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello AI lab","parameters":{"temperature":0.7}}'

curl http://localhost:8000/runs
```

The starter echoes the prompt locally and saves its input, parameters, output, and
timestamp in PostgreSQL. It makes no model API calls. Replace the app's
`service.py` with your own async model call or pipeline and add provider settings
when needed.

## Create another app

```bash
make new-app NAME=rag-demo
make migrate APP=rag-demo
make dev APP=rag-demo PORT=8001
```

Use lowercase names separated by hyphens. The generator refuses to overwrite an
existing app and automatically renders its Python package, schema, and migration
names. `make new-app` also installs the new workspace package and updates `uv.lock`.
Scaffolding alone, without installing dependencies: `python3 scripts/manage.py new rag-demo`.

Run multiple apps in separate terminals with different ports:

```bash
make dev APP=playground PORT=8000
make dev APP=rag-demo PORT=8001
make dev APP=classifier PORT=8002  # after creating and migrating classifier
```

`make list` lists apps. Add dependencies to a specific app:

```bash
uv add --package ai-app-rag-demo <package-name>
```

All apps use a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
with one virtual environment and lockfile. Use separate repositories or environments
if experiments require conflicting versions of the same dependency.

## Structure

```text
apps/
  playground/
    src/ai_playground/    # API, models, service, configuration of the app
    migrations/          # independent Alembic history
    alembic.ini
    pyproject.toml
packages/
  ai-shared/src/ai_shared/ # common settings and async database engine factory
templates/app/            # editable starter for future apps
scripts/manage.py         # generator and development commands
tests/                    # generator and opt-in PostgreSQL integration tests
compose.yaml              # PostgreSQL 18 with persistent storage
pyproject.toml            # uv workspace and development tools
uv.lock
```

## Database and migrations

PostgreSQL listens on `127.0.0.1:55432` by default, avoiding the usual local 5432
port. `.env` supplies Compose credentials and the application `DATABASE_URL`.
If you change a port or credentials, update both. PostgreSQL initialization
credentials apply when its volume is first created.

Schemas such as `app_playground` and `app_rag_demo` keep app tables separate.
Each schema also has its own `alembic_version` table. This is data organization;
apps use the same development database role and can access other schemas.
Migrations filter schema reflection so autogenerate only examines the current app.

After changing an app's `models.py`:

```bash
make revision APP=rag-demo MESSAGE="add documents"
# Review the generated migration, then:
make migrate APP=rag-demo
make check APP=rag-demo
```

`GET /health` checks the API. `GET /health/ready` also checks PostgreSQL and the
app's migrated table. `make db-down` stops PostgreSQL and keeps its data volume.

## Verification

```bash
make lint
make test
```

The database test is opt-in. With PostgreSQL running, supply a reachable SQLAlchemy
URL for a development database role that can create schemas:

```bash
AI_LAB_TEST_DATABASE_URL=postgresql+psycopg://ai_lab:ai_lab_dev@localhost:55432/ai_lab \
  uv run --all-packages pytest
```

It generates two temporary apps with unique schemas, exercises CRUD and schema
isolation, checks migration consistency, tests downgrade/upgrade, and removes its
schemas afterward. It does not use the playground's tables.

For future apps, edit `templates/app/`. Changes there affect newly generated apps;
update existing apps separately or put reusable infrastructure in `ai-shared`.
