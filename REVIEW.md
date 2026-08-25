# Code Review Guidelines

This document outlines what to look for when reviewing pull requests for the OpenBB Ada project.

> **Product + Architecture Context:** OpenBB Ada is the AI copilot backend for the OpenBB Workspace platform. It provides streaming LLM responses via SSE, processes documents for RAG, executes SQL queries on user data, and orchestrates tool calls. The backend uses FastAPI with dependency injection, Pydantic for validation, and async patterns throughout. Start every review with that architecture in mind.

## Review Philosophy

- **Be constructive, not critical** - Suggest improvements, don't just point out flaws
- **Assume good intent** - The author made decisions for reasons; ask before assuming they're wrong
- **Focus on what matters** - Don't nitpick formatting if ruff handles it
- **Review the code, not the person** - "This function could be simplified" not "You wrote this wrong"
- **DON'T BE A YES MAN** - We need to make sure the code is good and follows the guidelines

---

## Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Classes | PascalCase + descriptive suffix | `CopilotService`, `SqlAgentError` |
| Functions/methods | snake_case | `query_widgets`, `get_sql_tables_info` |
| Private attributes | Single underscore prefix | `self._template_service`, `_get_model()` |
| Constants | UPPER_SNAKE_CASE | `MAX_CALLS`, `AUTH_ENABLED` |
| Files | snake_case | `copilot_data.py`, `sql_agent.py` |
| Internal modules | Underscore prefix | `_logging.py` (utility/internal modules) |

---

## Review Checklist

### 1. Tests

> **This is the most important section.** Tests are not optional.

- [ ] **New code has tests** - All new functions, services, and endpoints must have tests
- [ ] **Tests are meaningful** - Not just coverage padding; they test actual behavior
- [ ] **Tests cover edge cases** - Empty inputs, error states, boundary conditions
- [ ] **Tests follow patterns** - Consistent with existing test structure in `tests/`
- [ ] **Async tests use proper fixtures** - `@pytest.mark.asyncio` and async fixtures
- [ ] **Stateful tests are marked** - `@pytest.mark.stateful` for non-parallelizable tests
- [ ] **Integration tests are marked** - `@pytest.mark.integration` for external service tests
- [ ] **No skipped tests** - `@pytest.mark.skip` should not be committed without justification
- [ ] **Tests pass locally** - Author should confirm tests pass before requesting review

**Red flags:**
```python
# Bad: No assertions
def test_service_query():
    service = MyService()
    service.query("test")


# Good: Meaningful assertion
def test_service_query_returns_expected_result():
    service = MyService()
    result = service.query("test")
    assert result.answer == "expected"
```

```python
# Bad: Testing implementation, not behavior
def test_internal_state():
    service = MyService()
    assert service._internal_cache == {}


# Good: Testing observable behavior
async def test_query_caches_result():
    service = MyService()
    result1 = await service.query("test")
    result2 = await service.query("test")
    assert result1 == result2
```

---

### 2. Code Quality

#### Readability

- [ ] **Code is self-documenting** - Variable/function names explain what they do
- [ ] **No unnecessary comments** - Comments explain "why", not "what"
- [ ] **Functions are focused** - Each function does one thing
- [ ] **No magic numbers/strings** - Use constants from `constants.py`

```python
# Bad
if response.status_code == 429:
    await asyncio.sleep(60)

# Good
from openbb_ada.constants import RATE_LIMIT_RETRY_SECONDS

if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
    await asyncio.sleep(RATE_LIMIT_RETRY_SECONDS)
```

#### Complexity

- [ ] **No over-engineering** - Solve the problem at hand, not hypothetical future problems
- [ ] **No premature abstraction** - Three similar lines are often better than a premature helper
- [ ] **Appropriate error handling** - Handle errors at system boundaries, trust internal code

```python
# Bad: Over-engineered factory pattern for simple use case
class ServiceFactory:
    _registry: dict[str, type[BaseService]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(service_cls):
            cls._registry[name] = service_cls
            return service_cls

        return decorator


# Good: Simple function that does what's needed
def get_document_service(request: AdaQueryRequest) -> DocumentService:
    return DocumentService(documents=request.documents)
```

#### DRY vs WET

- [ ] **Appropriate duplication** - Some duplication is OK; bad abstractions are worse
- [ ] **Shared logic is truly shared** - If extracting, ensure it's used in 3+ places

#### Import Organization

Imports must follow this order (ruff's `I` rule enforces this):
1. Standard library
2. Third-party packages
3. Local imports

```python
# Standard library
import asyncio
import json
from typing import Any, AsyncGenerator
from uuid import UUID

# Third-party
from fastapi import Depends, Request
from pydantic import BaseModel, Field
from magentic import StreamedStr

# Local
from ..constants import MAX_CALLS
from ..errors import SqlAgentError
from .template import TemplateService
```

#### Docstring Guidelines

- [ ] **Public methods have docstrings** - Required for service methods, endpoints, complex functions
- [ ] **Private methods optional** - Methods prefixed with `_` don't require docstrings
- [ ] **Use Google-style format** - With Parameters and Returns sections for complex methods

```python
# Good: Public method with docstring
def insert_table_with_all_infos(
    self,
    df: pd.DataFrame,
    table_name: str,
    description: str | None = None,
) -> list[SqlTableInfo]:
    """Insert a DataFrame into SQLite with automatic nested data normalization.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to insert.
    table_name : str
        Name of the table to create.
    description : str | None
        Optional description of the table.

    Returns
    -------
    list[SqlTableInfo]
        List of created table metadata.
    """
```

```python
# OK: Private method without docstring (self-explanatory name)
def _is_sqlite_internal_table(self, table_name: str) -> bool:
    return table_name.startswith("sqlite_")
```

#### Simplicity & Minimalism

- [ ] **Early returns** - Exit functions early to reduce nesting
- [ ] **Composition over inheritance** - Services use DI, not class hierarchies
- [ ] **Single-purpose methods** - Each method does one thing well
- [ ] **Explicit over clever** - Readable code beats compact code

```python
# Bad: Deeply nested
def process_data(data):
    if data is not None:
        if data.is_valid:
            if data.has_content:
                return transform(data)
    return None


# Good: Early returns
def process_data(data):
    if data is None:
        return None
    if not data.is_valid:
        return None
    if not data.has_content:
        return None
    return transform(data)
```

```python
# Bad: Inheritance hierarchy
class BaseProcessor:
    def process(self): ...


class DataProcessor(BaseProcessor):
    def process(self): ...


# Good: Composition via dependency injection
class DataProcessor:
    def __init__(self, validator: Validator, transformer: Transformer):
        self._validator = validator
        self._transformer = transformer
```

---

### 3. Python Type Hints

- [ ] **All function parameters typed** - Every parameter has a type hint
- [ ] **All return types typed** - Every function specifies its return type
- [ ] **Use `Annotated[]` for FastAPI** - Dependencies use `Annotated[Type, Depends(...)]`
- [ ] **No `Any` types** - If unavoidable, add comment explaining why
- [ ] **Pydantic models for data structures** - Don't use plain dicts for complex data
- [ ] **Use Python 3.10+ union syntax** - Use `str | None` not `Optional[str]`

```python
# Bad: Missing types
def process_query(query, context):
    return query + context


# Good: Fully typed
def process_query(query: str, context: list[str]) -> str:
    return query + " ".join(context)
```

```python
# Bad: Using dict for structured data
def get_user_info() -> dict:
    return {"name": "John", "role": "admin"}


# Good: Using Pydantic model
class UserInfo(BaseModel):
    name: str
    role: str


def get_user_info() -> UserInfo:
    return UserInfo(name="John", role="admin")
```

```python
# Bad: Old-style Depends
async def endpoint(service=Depends(get_service)): ...


# Good: Annotated style
async def endpoint(service: Annotated[MyService, Depends(get_service)]) -> Response: ...
```

```python
# Bad: Using Optional (old style)
from typing import Optional


def get_user(user_id: Optional[str] = None) -> Optional[dict]: ...


# Good: Python 3.10+ union syntax
def get_user(user_id: str | None = None) -> dict | None: ...
```

---

### 4. FastAPI Patterns

#### Dependency Injection

- [ ] **Services injected via `Depends()`** - No direct instantiation in endpoints
- [ ] **Dependencies in `dependencies.py`** - Factory functions centralized
- [ ] **Proper dependency lifecycle** - Resources cleaned up appropriately

```python
# Bad: Direct instantiation
@app.post("/query")
async def query_endpoint(request: Request):
    service = CopilotService()  # Wrong!
    return await service.query()


# Good: Dependency injection
@app.post("/query")
async def query_endpoint(
    service: Annotated[CopilotService, Depends(get_copilot_service)],
) -> EventSourceResponse:
    return EventSourceResponse(service.query())
```

#### Endpoints

- [ ] **Return types specified** - Endpoints have explicit return type annotations
- [ ] **Proper HTTP status codes** - Use appropriate codes (201 for created, 404 for not found)
- [ ] **Request validation via Pydantic** - Input validated through request models

#### Streaming

- [ ] **SSE uses async generators** - `AsyncGenerator[SSE, None]` for streaming
- [ ] **Proper event types** - Use `MessageChunkSSE`, `FunctionCallSSE`, etc.
- [ ] **Generator instrumentation** - Use `@instrument_async_generator` for tracing

```python
# Good: Properly typed async generator
async def query(self) -> AsyncGenerator[SSE, None]:
    yield StatusUpdateSSE(message="Processing...")
    async for chunk in self._llm_stream():
        yield MessageChunkSSE(content=chunk)
```

---

### 5. Service Layer

- [ ] **Services in `services/` directory** - Not scattered across codebase
- [ ] **Services injectable via `dependencies.py`** - Factory functions provided
- [ ] **Use `LoggingService`** - Not raw `logging.getLogger()`
- [ ] **Trace ID included in logs** - All logs include request trace_id
- [ ] **Custom exceptions from `errors.py`** - Domain-specific error types
- [ ] **Use `%` formatting in logs** - Not f-strings (for lazy evaluation)
- [ ] **Use `@logfire.instrument()`** - For tracing critical methods

#### Service Architecture Pattern

All services follow this standard structure:

```python
class MyService:
    """Service description."""

    def __init__(
        self,
        template_service: TemplateService,
        logging_service: LoggingService,
    ):
        # 1. Private dependencies (underscore prefix)
        self._template_service = template_service
        self._logging_service = logging_service

    # 2. Public API methods
    @logfire.instrument("MyService.query")
    async def query(self, request: QueryRequest) -> QueryResult:
        """Public method with docstring."""
        self._logging_service.info("Processing query: %s", request.id)
        return await self._process_request(request)

    # 3. Private implementation methods
    async def _process_request(self, request: QueryRequest) -> QueryResult:
        # Implementation details
        ...
```

#### Logging Patterns

```python
# Bad: f-string in logger (evaluated even if not logged)
logger.info(f"Processing user {user_id} with data {expensive_serialize(data)}")

# Good: % formatting (lazy evaluation)
self._logging_service.info("Processing user %s with data %s", user_id, data)
```

```python
# Bad: Raw logging without context
import logging

logger = logging.getLogger(__name__)


def my_function():
    logger.info("Processing request")


# Good: LoggingService with trace_id
class MyService:
    def __init__(self, logging_service: LoggingService):
        self._logging_service = logging_service

    def my_function(self):
        self._logging_service.info("Processing request")
```

```python
# Good: Instrument critical methods for tracing
@logfire.instrument("CopilotDataService.query_widgets")
async def query_widgets(
    self, requests: list[QueryWidgetRequest]
) -> QueryWidgetsResult: ...
```

#### Custom Exceptions

```python
# Bad: Generic exception
if not valid:
    raise Exception("Invalid input")

# Good: Custom exception from errors.py
from openbb_ada.errors import StructuredContextError

if not valid:
    raise StructuredContextError("Context data structure is invalid")
```

---

### 6. API & Data Handling

- [ ] **Pydantic validators for complex validation** - Use `@model_validator`
- [ ] **Error states handled** - What happens when external API fails?
- [ ] **Retries with backoff** - Use `retry_on_exception` decorator for transient failures
- [ ] **Async httpx for HTTP calls** - Not `requests` library
- [ ] **Explicit timeouts** - All HTTP calls have timeout specified

```python
# Bad: No timeout, using requests
import requests

response = requests.get(url)

# Good: Async with timeout
async with httpx.AsyncClient() as client:
    response = await client.get(url, timeout=15)
```

```python
# Good: Retry with backoff
@retry_on_exception(max_retries=3, exceptions=(httpx.TimeoutException,))
async def fetch_data(url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, timeout=15)
        return response.json()
```

---

### 7. Security

- [ ] **No secrets in code** - API keys, tokens in environment variables only
- [ ] **No secrets in logs** - Never log credentials, tokens, or sensitive data
- [ ] **Auth middleware for protected routes** - Verify JWT tokens
- [ ] **Input validation via Pydantic** - All user input validated
- [ ] **SQL injection prevention** - Use parameterized queries, never string formatting

```python
# Bad: Hardcoded secret
API_KEY = "sk-1234567890abcdef"

# Good: From environment
from openbb_ada.constants import OPENBB_AGENT_OPENAI_API_KEY
```

```python
# Bad: SQL injection vulnerability
query = f"SELECT * FROM users WHERE name = '{user_input}'"

# Good: Parameterized query
query = "SELECT * FROM users WHERE name = :name"
result = connection.execute(query, {"name": user_input})
```

---

### 8. Configuration

- [ ] **All settings in `constants.py`** - Loaded from environment variables
- [ ] **Feature flags documented** - Purpose and default value clear
- [ ] **No hardcoded URLs/values** - Everything configurable
- [ ] **Graceful degradation** - Features disabled if config missing

```python
# Bad: Hardcoded configuration
BASE_URL = "https://api.openbb.co"
MAX_RETRIES = 3

# Good: From constants.py
from openbb_ada.constants import (
    OPENBB_API_BASE_URL,
    MAX_RETRIES,
)
```

---

### 9. LLM Integration

- [ ] **Prompts in `templates/` directory** - Not hardcoded strings
- [ ] **Use `TemplateService`** - Render Jinja2 templates properly
- [ ] **Token limits respected** - Check token count before API calls
- [ ] **Function schemas valid** - Tool definitions match expected format
- [ ] **Streaming properly handled** - Async generator pattern for LLM responses

```python
# Bad: Hardcoded prompt
prompt = "You are a helpful assistant. Answer the following question: " + query

# Good: Template-based
prompt = self._template_service.render_template(
    "copilot_system_prompt_template.jinja", {"query": query, "context": context}
)
```

---

### 10. File Organization

- [ ] **Correct location** - File is in appropriate directory per project structure
- [ ] **Test file exists** - Corresponding test file in `tests/`
- [ ] **Services in `services/`** - All service classes
- [ ] **Templates in `templates/`** - All Jinja2 templates
- [ ] **Models in `models.py`** - All Pydantic models
- [ ] **No circular imports** - Check for import cycles

---

### System Impact

- [ ] **Consider downstream effects** - Shared services, API contracts, feature flags
- [ ] **Look for likely regressions** - Auth, streaming, error handling, caching
- [ ] **Confirm observability** - Logging at key points with trace_id
- [ ] Ask "what else could this impact?" and note any follow-ups needed

---

## PR Description Checklist

The PR description should include:

- [ ] **Summary** - What does this PR do?
- [ ] **Why** - What problem does it solve?
- [ ] **How to test** - Steps to verify the change works
- [ ] **Breaking changes** - If any, how to migrate
- [ ] **Environment variables** - If new config added, document it

---

## When to Request Changes vs Approve

### Request Changes

- Tests are missing or inadequate
- Security vulnerability present
- Breaking change without migration path
- Code doesn't work as described
- Major performance issue (blocking I/O in async, N+1 queries)
- Type hints missing on public API

### Approve with Comments

- Minor style preferences
- Suggestions for future improvements
- Questions about design decisions (non-blocking)
- Nitpicks that don't affect functionality

### Approve

- Code works as described
- Tests are adequate
- Follows project conventions
- No security concerns

---

## Review Response Etiquette

### As a Reviewer

- Differentiate between blocking issues and suggestions
- Prefix non-blocking comments with "nit:" or "suggestion:"
- Explain the "why" behind requested changes
- Offer solutions, not just problems
- Respond promptly to author's questions

### As an Author

- Don't take feedback personally
- Explain your reasoning if you disagree
- Ask for clarification if feedback is unclear
- Address all comments before re-requesting review
- Thank reviewers for their time

---

## Common Review Comments (Copy-Paste)

### Missing Tests
```
This new function/service needs tests. Please add tests covering:
- Happy path
- Error cases
- Edge cases (empty input, invalid data, etc.)

See existing tests in `tests/` for patterns.
```

### Missing Type Hints
```
This function is missing type hints. Please add:
- Parameter types
- Return type

Example:
def my_function(param: str, count: int) -> list[str]:
```

### Over-Engineering
```
This seems more complex than necessary for the current requirements.
Could we simplify to just handle the current use case? We can always
extend later if needed (YAGNI principle).
```

### Missing Error Handling
```
What happens if this external call fails? Please add error handling
with appropriate logging and user-friendly error messages.
```

### Hardcoded Value
```
This value should be in `constants.py` and loaded from environment
variables to allow configuration without code changes.
```

### Missing Logging
```
This operation should have logging for observability. Please add
logging using `LoggingService` with appropriate level (info/warning/error).
```

### Blocking I/O in Async
```
This appears to be a blocking call inside an async function.
Please use the async equivalent (e.g., `httpx` instead of `requests`,
`aiofiles` instead of `open`).
```
