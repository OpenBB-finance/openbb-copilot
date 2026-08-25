# AGENTS.md

This file is the entry point for coding agents working in this repository.

Keep this document short. Put setup, environment, and command details in
[README.md](README.md). Put system structure and code navigation in
[ARCHITECTURE.md](ARCHITECTURE.md).

`AGENTS.md` should be a map, not an encyclopedia.

## Documentation Map

- [README.md](README.md): development setup, model configuration, runtime options,
  and validation commands.
- [ARCHITECTURE.md](ARCHITECTURE.md): current architecture, request flow, service
  boundaries, storage, and extension points.
- [openbb_ada/main.py](openbb_ada/main.py): FastAPI routes, middleware, SSE
  streaming surface.
- [openbb_ada/dependencies.py](openbb_ada/dependencies.py): request-scoped
  dependency graph and service composition.
- [openbb_ada/copilot.py](openbb_ada/copilot.py): top-level Ada orchestration.
- [openbb_ada/services/](openbb_ada/services): focused services for documents,
  SQL, web search, MCP, prompt enhancement, citations, and metadata generation.
- [openbb_ada/templates/](openbb_ada/templates): Jinja prompt templates.
- [tests/](tests): route, service, model, and utility coverage.

## Repo-Specific Rules

- Read the relevant code and tests before changing behavior.
- Reuse existing services and helpers before adding new abstractions.
- Prefer surgical edits over broad refactors unless the task requires them.
- Verify imports and types against the actual module before using them.
- Use `.venv/bin/...` for Python-based commands in this repo.
- For local development, prefer the IDE debugger configurations documented in
  [Running the Service](README.md#running-the-service). Do not assume
  `docker compose` is the default path.
- When running tests, set `ENVIRONMENT=TEST` and use a valid `.env`; see
  [Running Tests](README.md#running-tests).
- Validate the smallest relevant surface for the change and report what you ran.

## Where To Look

- Request handling and streaming: [openbb_ada/main.py](openbb_ada/main.py)
- Query orchestration: [openbb_ada/copilot.py](openbb_ada/copilot.py)
- Context loading and SQL-backed artifact querying:
  [openbb_ada/services/context.py](openbb_ada/services/context.py) and
  [openbb_ada/services/sql_agent.py](openbb_ada/services/sql_agent.py)
- Document ingestion and RAG:
  [openbb_ada/services/document.py](openbb_ada/services/document.py) and
  [openbb_ada/services/document_agent.py](openbb_ada/services/document_agent.py)
- MCP and tool-call flows:
  [openbb_ada/services/client_function_call.py](openbb_ada/services/client_function_call.py),
  [openbb_ada/services/native_function_call.py](openbb_ada/services/native_function_call.py),
  and [openbb_ada/services/mcp_data.py](openbb_ada/services/mcp_data.py)
- Shared request and response models: [openbb_ada/models.py](openbb_ada/models.py)

## Validation Commands

Use the command definitions in:

- [Linting and Formatting](README.md#linting-and-formatting)
- [Running Tests](README.md#running-tests)

## Documentation Maintenance

- Keep `AGENTS.md` concise and link outward instead of duplicating detail.
- Update [README.md](README.md) when setup, commands, env vars, or operator
  workflows change.
- Update [ARCHITECTURE.md](ARCHITECTURE.md) when routes, request flow, service
  boundaries, or storage patterns change.
- If two docs start repeating each other, keep the fuller explanation in the
  source document and replace the duplicate with a link.
