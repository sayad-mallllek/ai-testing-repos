# playground

Run these commands from the monorepo root:

```bash
make migrate APP=playground
make dev APP=playground PORT=8001
```

Open http://localhost:8001/docs to try the API. Replace
`src/ai_playground/service.py` with your AI experiment. The initial service echoes
the prompt locally so you can verify the API and storage without a paid provider.

This app owns PostgreSQL schema `app_playground` and its own Alembic migration
history. After changing `models.py`, generate and review a migration:

```bash
make revision APP=playground MESSAGE="add documents"
make migrate APP=playground
make check APP=playground
```

Add app-specific dependencies with `uv add --package ai-app-playground <package>`.
