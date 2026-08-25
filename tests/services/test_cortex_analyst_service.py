"""Tests for CortexAnalystClient request payload and response handling."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from openbb_ada.errors import CodeGenerationError
from openbb_ada.models import CodeGenerationRequest, SemanticModelReference
from openbb_ada.services.snowflake_cortex_analyst import CortexAnalystClient

# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def test_build_payload_uses_semantic_view():
    request = CodeGenerationRequest(
        widget_uuid="widget-1",
        user_prompt="show revenue",
        language="sql",
        semantic_view="DB.SCHEMA.REVENUE_VIEW",
    )

    payload = CortexAnalystClient._build_payload(request)

    assert payload["semantic_view"] == "DB.SCHEMA.REVENUE_VIEW"
    assert "semantic_models" not in payload


def test_build_payload_uses_semantic_models_when_present():
    request = CodeGenerationRequest(
        widget_uuid="widget-1",
        user_prompt="show revenue",
        language="sql",
        semantic_models=[
            SemanticModelReference(semantic_view="DB.SCHEMA.REVENUE_VIEW"),
            SemanticModelReference(semantic_view="DB.SCHEMA.COST_VIEW"),
        ],
    )

    payload = CortexAnalystClient._build_payload(request)

    assert payload["semantic_models"] == [
        {"semantic_view": "DB.SCHEMA.REVENUE_VIEW"},
        {"semantic_view": "DB.SCHEMA.COST_VIEW"},
    ]
    assert "semantic_view" not in payload


def test_build_payload_no_semantic_config():
    request = CodeGenerationRequest(
        widget_uuid="widget-1",
        user_prompt="show revenue",
        language="sql",
    )

    payload = CortexAnalystClient._build_payload(request)

    assert "semantic_view" not in payload
    assert "semantic_models" not in payload
    assert "semantic_model" not in payload
    assert "semantic_model_file" not in payload
    assert payload["messages"][0]["content"][0]["text"] == "show revenue"


# ---------------------------------------------------------------------------
# Response parsing via _call_cortex_analyst
# ---------------------------------------------------------------------------


def _make_client() -> CortexAnalystClient:
    return CortexAnalystClient(logging_service=Mock())


def _make_request(**kwargs) -> CodeGenerationRequest:
    defaults = {"widget_uuid": "w-1", "user_prompt": "get revenue", "language": "sql"}
    defaults.update(kwargs)
    return CodeGenerationRequest(**defaults)


def _mock_response(status_code: int = 200, json_data=None, text: str = ""):
    resp = Mock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    if json_data is not None:
        resp.json = Mock(return_value=json_data)
    else:
        resp.json = Mock(side_effect=ValueError("no json"))
    return resp


@pytest.mark.asyncio
async def test_parse_valid_sql_block():
    client = _make_client()
    response_data = {
        "message": {
            "content": [{"type": "sql", "statement": "SELECT SUM(revenue) FROM sales"}]
        }
    }
    mock_resp = _mock_response(json_data=response_data)
    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.post = AsyncMock(return_value=mock_resp)

    with (
        patch.object(client, "_get_auth_headers", return_value={}),
        patch(
            "openbb_ada.services.snowflake_cortex_analyst.get_snowflake_host",
            return_value="test.snowflakecomputing.com",
        ),
        patch("httpx.AsyncClient", return_value=mock_http),
    ):
        sql, warnings, error, suggestions = await client._call_cortex_analyst(
            _make_request(), "get revenue"
        )

    assert sql == "SELECT SUM(revenue) FROM sales"
    assert warnings == []
    assert error is None


@pytest.mark.asyncio
async def test_parse_sql_block_with_warnings():
    client = _make_client()
    response_data = {
        "message": {"content": [{"type": "sql", "statement": "SELECT 1"}]},
        "warnings": [
            {"message": "Column 'x' is ambiguous"},
            {"message": "Using default time range"},
        ],
    }
    mock_resp = _mock_response(json_data=response_data)
    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.post = AsyncMock(return_value=mock_resp)

    with (
        patch.object(client, "_get_auth_headers", return_value={}),
        patch(
            "openbb_ada.services.snowflake_cortex_analyst.get_snowflake_host",
            return_value="test.snowflakecomputing.com",
        ),
        patch("httpx.AsyncClient", return_value=mock_http),
    ):
        sql, warnings, error, suggestions = await client._call_cortex_analyst(
            _make_request(), "get revenue"
        )

    assert sql == "SELECT 1"
    assert warnings == ["Column 'x' is ambiguous", "Using default time range"]


@pytest.mark.asyncio
async def test_parse_no_sql_returns_feedback():
    client = _make_client()
    response_data = {
        "message": {
            "content": [
                {"type": "text", "text": "I cannot generate SQL for this request."},
                {
                    "type": "suggestions",
                    "suggestions": ["Try asking about revenue", "Specify a date range"],
                },
            ]
        }
    }
    mock_resp = _mock_response(json_data=response_data)
    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.post = AsyncMock(return_value=mock_resp)

    with (
        patch.object(client, "_get_auth_headers", return_value={}),
        patch(
            "openbb_ada.services.snowflake_cortex_analyst.get_snowflake_host",
            return_value="test.snowflakecomputing.com",
        ),
        patch("httpx.AsyncClient", return_value=mock_http),
    ):
        sql, warnings, error, suggestions = await client._call_cortex_analyst(
            _make_request(), "get revenue"
        )

    assert sql is None
    assert error == "I cannot generate SQL for this request."
    assert suggestions == ["Try asking about revenue", "Specify a date range"]


@pytest.mark.asyncio
async def test_parse_sql_block_with_none_statement_is_skipped():
    """Block with type=sql but statement=None must not return 'None'."""
    client = _make_client()
    response_data = {
        "message": {
            "content": [
                {"type": "sql", "statement": None},
                {"type": "text", "text": "Could not resolve table."},
            ]
        }
    }
    mock_resp = _mock_response(json_data=response_data)
    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.post = AsyncMock(return_value=mock_resp)

    with (
        patch.object(client, "_get_auth_headers", return_value={}),
        patch(
            "openbb_ada.services.snowflake_cortex_analyst.get_snowflake_host",
            return_value="test.snowflakecomputing.com",
        ),
        patch("httpx.AsyncClient", return_value=mock_http),
    ):
        sql, warnings, error, suggestions = await client._call_cortex_analyst(
            _make_request(), "get revenue"
        )

    assert sql is None
    assert error == "Could not resolve table."


# ---------------------------------------------------------------------------
# HTTP error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_error_raises_code_generation_error():
    client = _make_client()
    mock_resp = _mock_response(status_code=500, text="Internal Server Error")
    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.post = AsyncMock(return_value=mock_resp)

    with (
        patch.object(client, "_get_auth_headers", return_value={}),
        patch(
            "openbb_ada.services.snowflake_cortex_analyst.get_snowflake_host",
            return_value="test.snowflakecomputing.com",
        ),
        patch("httpx.AsyncClient", return_value=mock_http),
    ):
        with pytest.raises(CodeGenerationError) as exc_info:
            await client._call_cortex_analyst(_make_request(), "get revenue")

    assert exc_info.value.status_code == 500
    assert "Internal Server Error" in exc_info.value.message


@pytest.mark.asyncio
async def test_malformed_json_raises_code_generation_error():
    client = _make_client()
    mock_resp = Mock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json = Mock(side_effect=json.JSONDecodeError("Expecting value", "doc", 0))
    mock_resp.text = "not json"
    mock_http = AsyncMock()
    mock_http.__aenter__ = AsyncMock(return_value=mock_http)
    mock_http.__aexit__ = AsyncMock(return_value=False)
    mock_http.post = AsyncMock(return_value=mock_resp)

    with (
        patch.object(client, "_get_auth_headers", return_value={}),
        patch(
            "openbb_ada.services.snowflake_cortex_analyst.get_snowflake_host",
            return_value="test.snowflakecomputing.com",
        ),
        patch("httpx.AsyncClient", return_value=mock_http),
    ):
        with pytest.raises(CodeGenerationError) as exc_info:
            await client._call_cortex_analyst(_make_request(), "get revenue")

    assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------


def test_get_auth_headers_uses_oauth_in_spcs():
    client = _make_client()

    with (
        patch(
            "openbb_ada.services.snowflake_cortex_analyst.get_login_token",
            return_value="token-123",
        ),
        patch(
            "openbb_ada.services.snowflake_cortex_analyst.is_running_in_spcs",
            return_value=True,
        ),
    ):
        headers = client._get_auth_headers()

    assert headers == {
        "Authorization": "Bearer token-123",
        "X-Snowflake-Authorization-Token-Type": "OAUTH",
    }


def test_get_auth_headers_uses_pat_locally():
    client = _make_client()

    with (
        patch(
            "openbb_ada.services.snowflake_cortex_analyst.get_login_token",
            return_value="token-123",
        ),
        patch(
            "openbb_ada.services.snowflake_cortex_analyst.is_running_in_spcs",
            return_value=False,
        ),
    ):
        headers = client._get_auth_headers()

    assert headers == {
        "Authorization": "Bearer token-123",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    }


# ---------------------------------------------------------------------------
# Retry / reformulation via generate_sql
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_sql_retries_on_feedback():
    """generate_sql reformulates the prompt and retries when Cortex returns feedback."""
    client = _make_client()
    call_count = 0

    async def mock_call(request, user_prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return None, [], "Unknown column 'foo'", ["Try 'revenue'"]
        return "SELECT revenue FROM sales", ["warning1"], None, []

    with patch.object(client, "_call_cortex_analyst", side_effect=mock_call):
        sql, warnings = await client.generate_sql(_make_request(semantic_view="DB.S.V"))

    assert sql == "SELECT revenue FROM sales"
    assert warnings == ["warning1"]
    assert call_count == 2


@pytest.mark.asyncio
async def test_generate_sql_raises_after_max_retries():
    """generate_sql raises CodeGenerationError after exhausting retries."""
    client = _make_client()

    async def always_fail(request, user_prompt):
        return None, [], "Cannot resolve", []

    with patch.object(client, "_call_cortex_analyst", side_effect=always_fail):
        with pytest.raises(CodeGenerationError) as exc_info:
            await client.generate_sql(_make_request(semantic_view="DB.S.V"))

    assert exc_info.value.status_code == 502
    assert "Cannot resolve" in exc_info.value.message


# ---------------------------------------------------------------------------
# Reformulate prompt
# ---------------------------------------------------------------------------


def test_reformulate_prompt_includes_suggestions():
    result = CortexAnalystClient._reformulate_prompt(
        original_prompt="show revenue",
        cortex_feedback="Column not found",
        suggestions=["Try total_revenue", "Use sum(amount)"],
    )

    assert "show revenue" in result
    assert "Column not found" in result
    assert "- Try total_revenue" in result
    assert "- Use sum(amount)" in result


def test_reformulate_prompt_without_suggestions():
    result = CortexAnalystClient._reformulate_prompt(
        original_prompt="show revenue",
        cortex_feedback="Column not found",
        suggestions=[],
    )

    assert "show revenue" in result
    assert "Column not found" in result
    assert "Suggested reformulations" not in result
