# Architecture

This document is the architecture source of truth for `openbb-ada`.

Keep setup, environment, and command details in [README.md](README.md). Keep
agent workflow rules in [AGENTS.md](AGENTS.md). Do not duplicate those sections
here unless a change cannot be understood without them.

## Current System Snapshot

OpenBB Ada is a FastAPI backend that streams AI responses for OpenBB Workspace.
The current codebase is broader than the older "Copilot + document/sql/web
search" summary. The live system now includes:

- SSE query orchestration
- request-scoped dependency wiring
- structured and unstructured context loading
- document and image ingestion
- SQL querying over structured data
- client-side and native tool-call execution
- MCP tool support and MCP artifact normalization
- URL retrieval and web-search flows
- prompt enhancement
- widget, chat, and dashboard title generation
- optional SQL and Python code generation
- skill catalog and selected-skill support in query requests

## Code Map

| Area | Primary file(s) | Responsibility |
| --- | --- | --- |
| HTTP surface | [openbb_ada/main.py](openbb_ada/main.py) | Routes, middleware, auth gate, SSE responses |
| Dependency graph | [openbb_ada/dependencies.py](openbb_ada/dependencies.py) | Request-scoped service construction |
| Main orchestration | [openbb_ada/copilot.py](openbb_ada/copilot.py) | Chat assembly, tool orchestration, streamed response flow |
| Shared models | [openbb_ada/models.py](openbb_ada/models.py) | Pydantic models layered on top of `openbb-ai` |
| Services | [openbb_ada/services/](openbb_ada/services) | Focused runtime capabilities |
| Prompt templates | [openbb_ada/templates/](openbb_ada/templates) | Jinja prompt rendering |
| Storage helpers | [openbb_ada/vector_db.py](openbb_ada/vector_db.py) | FAISS-backed vector storage and embeddings |
| Utilities | [openbb_ada/utils/](openbb_ada/utils) | Stream helpers, normalization, auth, AI config, chart/data helpers |
| Tests | [tests/](tests) | Route, service, model, and utility coverage |

## HTTP Endpoints

The FastAPI app lives in [openbb_ada/main.py](openbb_ada/main.py).

| Route | Purpose |
| --- | --- |
| `GET /agents.json` | Advertises the Ada agent and feature flags to clients |
| `POST /v1/query` | Main streaming query endpoint |
| `POST /v1/enhance_prompt` | Prompt enhancement helper |
| `POST /v1/generate/widget_info` | Widget title/description generation from data |
| `POST /v1/generate/widget_info/file` | Widget metadata generation from uploaded files |
| `POST /v1/generate/skill_info` | Skill slug/description/content generation from a conversation |
| `POST /v1/generate/chat/title` | Chat title generation |
| `POST /v1/generate/dashboard/title` | Dashboard title generation |
| `POST /v1/generate/code` | SQL or Python code suggestion generation |
| `GET /status`, `GET /health`, `GET /healthz` | Health checks |

## Request Flow

The main query flow is:

1. [openbb_ada/main.py](openbb_ada/main.py) applies request middleware for trace
   IDs, structured request logging, CORS, and optional auth.
2. `POST /v1/query` parses an [AdaQueryRequest](openbb_ada/models.py) through
   [openbb_ada/dependencies.py](openbb_ada/dependencies.py), which builds the
   request-scoped service graph.
3. The route chooses a dedicated
   [WebSearchLlmService](openbb_ada/services/web_search_llm.py) only when forced
   web search is enabled. Otherwise it calls
   [CopilotService.query](openbb_ada/copilot.py).
4. [CopilotService](openbb_ada/copilot.py) loads explicit context, rebuilds chat
   history, registers frontend-supplied MCP tools, and coordinates native tool
   calls, client-boundary tool calls, and downstream services.
5. Events are streamed back as SSE payloads via `EventSourceResponse`, including
   status updates, message chunks, citations, function-call boundaries, and
   artifacts.

## Service Boundaries

### Core Orchestrator

- [openbb_ada/copilot.py](openbb_ada/copilot.py): the main coordinator for query
  handling. It combines context loading, prompt construction, tool execution,
  citation handling, and streamed output.

### Request Composition

- [openbb_ada/dependencies.py](openbb_ada/dependencies.py): builds most services
  per request so they can share request-specific state such as user ID, API keys,
  widgets, workspace options, skills, and trace IDs.

### Data and Context

- [openbb_ada/services/context.py](openbb_ada/services/context.py): turns explicit
  context artifacts into structured or unstructured runtime context and bridges
  SQL-backed tables back into the prompt/tool flow.
- [openbb_ada/services/copilot_data.py](openbb_ada/services/copilot_data.py):
  handles widget-backed data-source discovery, extra widget retrieval, and
  related request shaping.
- [openbb_ada/services/citation.py](openbb_ada/services/citation.py): centralizes
  citation registration and formatting.
- [openbb_ada/services/url_retrieval.py](openbb_ada/services/url_retrieval.py):
  pulls URL content into the broader query flow when enabled.

### Document and File Handling

- [openbb_ada/services/document.py](openbb_ada/services/document.py): the entry
  point for loaded documents. It classifies files, delegates structured content
  to the SQL path, delegates unstructured content to the document agent, and
  optionally caches vector DB payloads in Redis.
- [openbb_ada/services/document_agent.py](openbb_ada/services/document_agent.py):
  performs document and image analysis and loads unstructured content into
  [VectorDb](openbb_ada/vector_db.py).
- [openbb_ada/services/user_file.py](openbb_ada/services/user_file.py): handles
  user file retrieval for the current request.

### Structured Data and SQL

- [openbb_ada/services/sql_agent.py](openbb_ada/services/sql_agent.py): owns the
  SQL query loop and a request-scoped SQLite database by default. It can also
  operate with an injected engine.
- [openbb_ada/utils/nested_data_normalizer.py](openbb_ada/utils/nested_data_normalizer.py):
  normalizes nested structures into parent/child tables before insertion.
- [openbb_ada/utils/child_table_schema_builder.py](openbb_ada/utils/child_table_schema_builder.py):
  builds child-table schemas for normalized nested data.

### Tool Calls and MCP

- [openbb_ada/services/client_function_call.py](openbb_ada/services/client_function_call.py):
  manages client-boundary function calls and flattens frontend-supplied MCP tools
  into LLM-callable functions.
- [openbb_ada/services/native_function_call.py](openbb_ada/services/native_function_call.py):
  owns backend-executed tools such as planning, prompt enhancement, data queries,
  document queries, web search, and code-suggestion helpers.
- [openbb_ada/services/mcp_data.py](openbb_ada/services/mcp_data.py): converts MCP
  responses into artifacts and, when possible, SQL-queryable structured context.

### Prompts and Generated Text

- [openbb_ada/services/template.py](openbb_ada/services/template.py): renders the
  Jinja templates in [openbb_ada/templates/](openbb_ada/templates).
- [openbb_ada/services/prompt_enhancement.py](openbb_ada/services/prompt_enhancement.py):
  rewrites vague user prompts into clearer requests.
- [openbb_ada/services/chat_title_generation.py](openbb_ada/services/chat_title_generation.py),
  [openbb_ada/services/dashboard_title_generation.py](openbb_ada/services/dashboard_title_generation.py),
  [openbb_ada/services/widget_metadata_generation.py](openbb_ada/services/widget_metadata_generation.py),
  and [openbb_ada/services/skill_generation.py](openbb_ada/services/skill_generation.py):
  support auxiliary title, metadata, and skill generation endpoints.
- [openbb_ada/services/sql_query_generation.py](openbb_ada/services/sql_query_generation.py)
  and [openbb_ada/services/python_code_generation.py](openbb_ada/services/python_code_generation.py):
  provide optional code-suggestion flows used by `POST /v1/generate/code`.

### Web Search

- [openbb_ada/services/web_search_llm.py](openbb_ada/services/web_search_llm.py):
  runs the dedicated web-search path using the OpenAI Responses API with
  `web_search_preview`, streaming both text deltas and collected citations.

## Storage and State

### Request-Scoped Runtime State

Most services are created per request in
[openbb_ada/dependencies.py](openbb_ada/dependencies.py). This keeps user API
keys, workspace context, selected skills, document state, and trace metadata out
of global mutable state.

### Structured Data

[openbb_ada/services/sql_agent.py](openbb_ada/services/sql_agent.py) uses an
in-memory SQLite database by default. Structured files, explicit structured
context, and some MCP outputs are inserted as SQL tables so the agent can query
them later in the same request flow.

### Unstructured Data

[openbb_ada/vector_db.py](openbb_ada/vector_db.py) wraps FAISS-based vector
storage and embedding providers. Unstructured documents are loaded through
[openbb_ada/services/document_agent.py](openbb_ada/services/document_agent.py).

### Redis Caching

[openbb_ada/services/document.py](openbb_ada/services/document.py) can persist
serialized vector DB payloads in Redis so repeated document loads can reuse
cached embeddings when document caching is enabled.

## Models and Protocol Boundaries

- [openbb_ada/models.py](openbb_ada/models.py) extends `openbb-ai` request and
  response models rather than redefining the full protocol locally.
- SSE payload types such as message chunks, status updates, and some citation
  structures come from `openbb-ai`.
- The query request model now includes workspace options, skill catalog entries,
  selected skills, widgets, context, URLs, and tool definitions.

## Templates and Prompting

Prompt templates live in [openbb_ada/templates/](openbb_ada/templates) and are
rendered by [openbb_ada/services/template.py](openbb_ada/services/template.py).
Important templates include:

- [openbb_ada/templates/copilot_system_prompt_template.jinja](openbb_ada/templates/copilot_system_prompt_template.jinja)
- [openbb_ada/templates/copilot_document_agent_system_prompt_template.jinja](openbb_ada/templates/copilot_document_agent_system_prompt_template.jinja)
- [openbb_ada/templates/copilot_sql_agent_system_prompt_template.jinja](openbb_ada/templates/copilot_sql_agent_system_prompt_template.jinja)
- [openbb_ada/templates/prompt_enhancement_template.jinja](openbb_ada/templates/prompt_enhancement_template.jinja)

## Operational References

For commands and environment details, use:

- [Model Configuration](README.md#model-configuration)
- [Running the Service](README.md#running-the-service)
- [Linting and Formatting](README.md#linting-and-formatting)
- [Running Tests](README.md#running-tests)

## Keeping This Doc Current

When architecture changes, update this file if any of the following changed:

- public routes in [openbb_ada/main.py](openbb_ada/main.py)
- request composition in [openbb_ada/dependencies.py](openbb_ada/dependencies.py)
- major service ownership or delegation boundaries in
  [openbb_ada/services/](openbb_ada/services)
- storage strategy for SQL, vector data, or Redis caching
- protocol boundaries in [openbb_ada/models.py](openbb_ada/models.py)

When a change is only operational, update [README.md](README.md) instead.
