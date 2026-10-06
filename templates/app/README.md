# __APP_NAME__

Run these commands from the monorepo root:

```bash
make migrate APP=__APP_NAME__
make dev APP=__APP_NAME__ PORT=8001
```

Open http://localhost:8001/docs to try the API. Replace
`src/__MODULE_NAME__/service.py` with your AI experiment. The initial service echoes
the prompt locally so you can verify the API and storage without a paid provider.

This app owns PostgreSQL schema `__SCHEMA_NAME__` and its own Alembic migration
history. After changing `models.py`, generate and review a migration:

```bash
make revision APP=__APP_NAME__ MESSAGE="add documents"
make migrate APP=__APP_NAME__
make check APP=__APP_NAME__
```

Add app-specific dependencies with `uv add --package ai-app-__APP_NAME__ <package>`.
