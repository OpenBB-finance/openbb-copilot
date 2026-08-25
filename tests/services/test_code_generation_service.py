from __future__ import annotations

from typing import AsyncGenerator
from unittest.mock import Mock

import pytest

from openbb_ada.errors import CodeGenerationError
from openbb_ada.models import (
    CodeGenerationRequest,
    PythonCodeGenerationResult,
    SqlQueryGenerationResult,
)
from openbb_ada.services.snowflake_cortex_analyst import SnowflakeCortexAnalystService


class StubSqlGenerationService:
    def __init__(self, sql: str = "SELECT 1", should_error: bool = False):
        self.called = False
        self._sql = sql
        self._should_error = should_error

    async def generate_query(
        self,
        user_request: str,
        widget_uuid: str,
        sql_widget_dict,
        generate_query_only: bool = True,
    ) -> AsyncGenerator[SqlQueryGenerationResult, None]:
        self.called = True
        if self._should_error:
            return
        yield SqlQueryGenerationResult(
            sql_query=self._sql,
            widget_uuid=widget_uuid,
            widget_id=sql_widget_dict.widget_id or "widget",
            widget_origin=sql_widget_dict.widget_origin or "OpenBB Workspace",
            artifacts=[],
        )


class StubPythonGenerationService:
    def __init__(self, code: str = "print('hello')"):
        self.called = False
        self._code = code

    async def generate_code(
        self,
        user_request: str,
        widget_uuid: str,
        python_widget,
        generate_code_only: bool = True,
        sql_widgets=None,
    ) -> AsyncGenerator[PythonCodeGenerationResult, None]:
        self.called = True
        yield PythonCodeGenerationResult(
            python_code=self._code,
            widget_uuid=widget_uuid,
            widget_id=python_widget.widget_id,
            widget_origin=python_widget.widget_origin,
            artifacts=[],
        )


class StubCortexAnalystService:
    def __init__(self, sql: str = "SELECT 2", should_error: bool = False):
        self.called = False
        self._sql = sql
        self._should_error = should_error

    async def resolve_semantic_view(self, sql_schema=None) -> str | None:
        return None

    async def generate_sql(
        self,
        request: CodeGenerationRequest,
        resolved_semantic_view: str | None = None,
    ) -> tuple[str, list[str]]:
        self.called = True
        if self._should_error:
            raise CodeGenerationError(status_code=502, message="cortex unavailable")
        return self._sql, ["cortex warning"]


@pytest.mark.asyncio
async def test_generate_sql_uses_llm_when_no_semantic():
    sql_service = StubSqlGenerationService(sql="SELECT * FROM users")
    cortex_service = StubCortexAnalystService()
    service = SnowflakeCortexAnalystService(
        logging_service=Mock(),
        sql_query_generation_service=sql_service,
        python_code_generation_service=None,
        cortex_analyst_client=cortex_service,
    )
    request = CodeGenerationRequest(
        widget_uuid="widget-1",
        user_prompt="get all users",
        language="sql",
    )

    response = await service.generate_code(request)

    assert response.generated_code == "SELECT * FROM users"
    assert response.generation_source == "llm"
    assert sql_service.called is True
    assert cortex_service.called is False


@pytest.mark.asyncio
async def test_generate_sql_uses_cortex_when_semantic_available():
    sql_service = StubSqlGenerationService(sql="SELECT * FROM should_not_run")
    cortex_service = StubCortexAnalystService(sql="SELECT * FROM cortex_result")
    service = SnowflakeCortexAnalystService(
        logging_service=Mock(),
        sql_query_generation_service=sql_service,
        python_code_generation_service=None,
        cortex_analyst_client=cortex_service,
    )
    request = CodeGenerationRequest(
        widget_uuid="widget-1",
        user_prompt="get total revenue",
        language="sql",
        semantic_view="MY_DB.MY_SCHEMA.MY_SEMANTIC_VIEW",
    )

    response = await service.generate_code(request)

    assert response.generated_code == "SELECT * FROM cortex_result"
    assert response.generation_source == "cortex_analyst"
    assert cortex_service.called is True
    assert sql_service.called is False


@pytest.mark.asyncio
async def test_generate_sql_uses_cortex_when_semantic_models_are_available():
    sql_service = StubSqlGenerationService(sql="SELECT * FROM should_not_run")
    cortex_service = StubCortexAnalystService(sql="SELECT * FROM cortex_result")
    service = SnowflakeCortexAnalystService(
        logging_service=Mock(),
        sql_query_generation_service=sql_service,
        python_code_generation_service=None,
        cortex_analyst_client=cortex_service,
    )
    request = CodeGenerationRequest(
        widget_uuid="widget-1",
        user_prompt="compare revenue and cost",
        language="sql",
        semantic_models=[
            {"semantic_view": "MY_DB.MY_SCHEMA.REVENUE_VIEW"},
            {"semantic_view": "MY_DB.MY_SCHEMA.COST_VIEW"},
        ],
    )

    response = await service.generate_code(request)

    assert response.generated_code == "SELECT * FROM cortex_result"
    assert response.generation_source == "cortex_analyst"
    assert cortex_service.called is True
    assert sql_service.called is False


@pytest.mark.asyncio
async def test_generate_sql_falls_back_to_llm_when_cortex_fails():
    sql_service = StubSqlGenerationService(sql="SELECT * FROM llm_fallback")
    cortex_service = StubCortexAnalystService(should_error=True)
    service = SnowflakeCortexAnalystService(
        logging_service=Mock(),
        sql_query_generation_service=sql_service,
        python_code_generation_service=None,
        cortex_analyst_client=cortex_service,
    )
    request = CodeGenerationRequest(
        widget_uuid="widget-1",
        user_prompt="get revenue",
        language="sql",
        semantic_view="MY_DB.MY_SCHEMA.MY_SEMANTIC_VIEW",
    )

    response = await service.generate_code(request)

    assert response.generated_code == "SELECT * FROM llm_fallback"
    assert response.generation_source == "llm"
    assert cortex_service.called is True
    assert sql_service.called is True


@pytest.mark.asyncio
async def test_generate_python_uses_python_service():
    python_service = StubPythonGenerationService(code="df = session.table('users')")
    service = SnowflakeCortexAnalystService(
        logging_service=Mock(),
        sql_query_generation_service=None,
        python_code_generation_service=python_service,
        cortex_analyst_client=None,
    )
    request = CodeGenerationRequest(
        widget_uuid="widget-python",
        user_prompt="load users",
        language="python",
    )

    response = await service.generate_code(request)

    assert response.generated_code == "df = session.table('users')"
    assert response.generation_source == "llm"
    assert python_service.called is True
