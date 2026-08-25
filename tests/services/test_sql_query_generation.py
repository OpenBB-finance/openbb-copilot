"""Tests for SQL Query Generation Service."""

import json
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from openbb_ai.models import (
    StatusUpdateSSE,
    StatusUpdateSSEData,
    Widget,
    WidgetCollection,
    WidgetParam,
)

from openbb_ada.models import (
    SqlQueryFunctionCallResult,
    SqlQueryGenerationResult,
    SqlWidgetDict,
)
from openbb_ada.services.sql_query_generation import SqlGenerationResponse

# Test UUIDs
CME_WIDGET_UUID = "08d47a2f-bd35-4f53-a0e6-a45b4c7252f0"
REGULAR_WIDGET_UUID = "a1b2c3d4-1fbc-472a-852a-cc96adff0701"

# CME CBT_SETTLEMENTS schema as JSON (Snowflake format)
CME_SETTLEMENTS_SCHEMA = {
    "database": "CME_DATA",
    "schema": "PUBLIC",
    "tableName": "CBT_SETTLEMENTS",
    "kind": "TABLE",
    "columns": [
        {"name": "BizDt", "type": "VARCHAR", "null": "Y", "comment": None},
        {"name": "Sym", "type": "VARCHAR", "null": "Y", "comment": None},
        {"name": "ID", "type": "VARCHAR", "null": "Y", "comment": None},
        {"name": "StrkPx", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "SecTyp", "type": "VARCHAR", "null": "Y", "comment": None},
        {"name": "MMY", "type": "NUMBER", "null": "Y", "comment": None},
        {"name": "MatDt", "type": "VARCHAR", "null": "Y", "comment": None},
        {"name": "PutCall", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "Exch", "type": "VARCHAR", "null": "Y", "comment": None},
        {"name": "Desc", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "LastTrdDt", "type": "VARCHAR", "null": "Y", "comment": None},
        {"name": "BidPrice", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "OpeningPrice", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "SettlePrice", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "SettleDelta", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "HighLimit", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "LowLimit", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "DHighPrice", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "DLowPrice", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "HighBid", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "LowBid", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "PrevDayVol", "type": "NUMBER", "null": "Y", "comment": None},
        {"name": "PrevDayOI", "type": "NUMBER", "null": "Y", "comment": None},
        {"name": "FixingPrice", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "UndlyExch", "type": "VARCHAR", "null": "Y", "comment": None},
        {"name": "UndlyID", "type": "VARCHAR", "null": "Y", "comment": None},
        {"name": "UndlySecTyp", "type": "VARCHAR", "null": "Y", "comment": None},
        {"name": "UndlyMMY", "type": "FLOAT", "null": "Y", "comment": None},
        {"name": "BankBusDay", "type": "FLOAT", "null": "Y", "comment": None},
    ],
    "column_count": 29,
}

DEFAULT_SQL_QUERY = "SELECT * FROM CME_DATA.PUBLIC.CBT_SETTLEMENTS LIMIT 100"


@pytest.fixture
def cme_widget_with_sql_metadata() -> Widget:
    """Widget with SQL metadata as it comes from the frontend."""
    return Widget(
        uuid=UUID(CME_WIDGET_UUID),
        name="CBT_SETTLEMENTS",
        origin="SNOW backend",
        widget_id="cme_data_public_cbt_settlements",
        description="Query data from CME_DATA.PUBLIC.CBT_SETTLEMENTS (table)",
        params=[
            WidgetParam(
                name="query",
                type="text",
                description="SQL query for CME_DATA.PUBLIC.CBT_SETTLEMENTS",
                default_value=DEFAULT_SQL_QUERY,
                current_value=DEFAULT_SQL_QUERY,
            )
        ],
        metadata={
            "schema": json.dumps(CME_SETTLEMENTS_SCHEMA),
        },
    )


@pytest.fixture
def extracted_sql_widgets(cme_widget_with_sql_metadata: Widget) -> list[SqlWidgetDict]:
    """SQL widgets as extracted by _get_sql_enabled_widgets."""
    widget = cme_widget_with_sql_metadata
    return [
        SqlWidgetDict(
            widget_uuid=str(widget.uuid),
            widget_id=widget.widget_id,
            widget_name=widget.name,
            widget_origin=widget.origin,
            sql_schema=json.loads(widget.metadata["schema"]),
            current_sql=widget.params[0].current_value,
        ),
    ]


@pytest.mark.asyncio
async def test_llm_generate_sql_query_function(
    test_copilot_service,
    extracted_sql_widgets,
):
    """Test the llm_generate_sql_query native function with CME widget."""
    expected_sql = (
        "SELECT Sym, SettlePrice FROM CBT_SETTLEMENTS "
        "ORDER BY SettlePrice DESC LIMIT 10"
    )

    async def mock_generate_query(*args, **kwargs):
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message="SQL query generated",
                details=[{"SQL": expected_sql}],
            )
        )
        yield SqlQueryGenerationResult(
            sql_query=expected_sql,
            widget_uuid=CME_WIDGET_UUID,
            widget_id="cme_data_public_cbt_settlements",
            widget_origin="SNOW backend",
            artifacts=[],
        )

    native_service = test_copilot_service._native_function_call_service
    native_service._sql_query_generation_service.generate_query = mock_generate_query

    events = []
    async for event in native_service.llm_generate_sql_query(
        user_request="Show me top 10 symbols by settle price",
        widget_uuid=CME_WIDGET_UUID,
        summary="Generating SQL query",
        sql_widgets=extracted_sql_widgets,
        generate_query_only=True,
    ):
        events.append(event)

    # 3 events: initial summary status, SQL generated status, and result
    assert len(events) == 3
    assert isinstance(events[0], StatusUpdateSSE)
    assert events[0].data.message == "Generating SQL query"
    assert isinstance(events[1], StatusUpdateSSE)
    assert isinstance(events[2], SqlQueryGenerationResult)
    assert events[2].sql_query == expected_sql
    assert events[2].widget_uuid == CME_WIDGET_UUID
    assert events[2].widget_id == "cme_data_public_cbt_settlements"
    assert events[2].widget_origin == "SNOW backend"


def test_get_sql_enabled_widgets_extracts_metadata(
    test_copilot_service,
    cme_widget_with_sql_metadata,
):
    """Test _get_sql_enabled_widgets extracts SQL metadata from widgets."""
    widget_collection = WidgetCollection(
        primary=[cme_widget_with_sql_metadata],
        secondary=[],
        extra=[],
    )

    result = test_copilot_service._get_sql_enabled_widgets(widget_collection)

    assert len(result) == 1
    assert result[0].widget_uuid == CME_WIDGET_UUID
    assert result[0].widget_id == "cme_data_public_cbt_settlements"
    assert result[0].widget_name == "CBT_SETTLEMENTS"
    assert result[0].widget_origin == "SNOW backend"
    assert result[0].sql_schema["tableName"] == "CBT_SETTLEMENTS"
    assert result[0].current_sql == DEFAULT_SQL_QUERY


def test_get_sql_enabled_widgets_ignores_widgets_without_schema(
    test_copilot_service,
):
    """Test that widgets without metadata.schema are ignored."""
    widget_without_sql = Widget(
        uuid=UUID(REGULAR_WIDGET_UUID),
        name="Regular Widget",
        origin="some-origin",
        widget_id="regular",
        description="A regular widget without SQL",
        params=[],
        metadata={},
    )

    widget_collection = WidgetCollection(
        primary=[widget_without_sql],
        secondary=[],
        extra=[],
    )

    result = test_copilot_service._get_sql_enabled_widgets(widget_collection)

    assert len(result) == 0


@pytest.mark.asyncio
async def test_generate_query_with_generate_query_only_true(
    test_sql_query_generation_service,
    cme_widget_with_sql_metadata,
):
    """When generate_query_only=True, artifacts should NOT be in StatusUpdateSSE."""
    widget_dict = SqlWidgetDict(
        widget_id=cme_widget_with_sql_metadata.widget_id,
        widget_uuid=str(cme_widget_with_sql_metadata.uuid),
        widget_name=cme_widget_with_sql_metadata.name,
        widget_origin=cme_widget_with_sql_metadata.origin,
        sql_schema=json.loads(cme_widget_with_sql_metadata.metadata["schema"]),
        current_sql=None,
    )

    with patch("openbb_ada.services.sql_query_generation.prompt") as mock_prompt:
        mock_prompt.return_value = lambda func: AsyncMock(
            return_value=SqlGenerationResponse(
                success=True,
                sql_query="SELECT * FROM CME_DATA.PUBLIC.CBT_SETTLEMENTS LIMIT 10",
            )
        )

        events = []
        async for event in test_sql_query_generation_service.generate_query(
            user_request="Show me all records",
            widget_uuid=str(cme_widget_with_sql_metadata.uuid),
            sql_widget_dict=widget_dict,
            generate_query_only=True,
        ):
            events.append(event)

    assert len(events) == 2
    assert isinstance(events[0], StatusUpdateSSE)
    assert events[0].data.message == "SQL query generated"
    assert events[0].data.artifacts == []

    assert isinstance(events[1], SqlQueryGenerationResult)
    assert (
        events[1].sql_query == "SELECT * FROM CME_DATA.PUBLIC.CBT_SETTLEMENTS LIMIT 10"
    )
    assert len(events[1].artifacts) == 1
    assert events[1].artifacts[0].data_format.parse_as == "snowflake_query"
    assert (
        events[1].artifacts[0].content
        == "```sql\nSELECT * FROM CME_DATA.PUBLIC.CBT_SETTLEMENTS LIMIT 10\n```"
    )


@pytest.mark.asyncio
async def test_generate_query_with_generate_query_only_false(
    test_sql_query_generation_service,
    cme_widget_with_sql_metadata,
):
    """When generate_query_only=False, artifacts SHOULD be in StatusUpdateSSE."""
    widget_dict = SqlWidgetDict(
        widget_id=cme_widget_with_sql_metadata.widget_id,
        widget_uuid=str(cme_widget_with_sql_metadata.uuid),
        widget_name=cme_widget_with_sql_metadata.name,
        widget_origin=cme_widget_with_sql_metadata.origin,
        sql_schema=json.loads(cme_widget_with_sql_metadata.metadata["schema"]),
        current_sql=None,
    )

    with patch("openbb_ada.services.sql_query_generation.prompt") as mock_prompt:
        mock_prompt.return_value = lambda func: AsyncMock(
            return_value=SqlGenerationResponse(
                success=True,
                sql_query="SELECT * FROM CME_DATA.PUBLIC.CBT_SETTLEMENTS LIMIT 10",
            )
        )

        events = []
        async for event in test_sql_query_generation_service.generate_query(
            user_request="Show me all records",
            widget_uuid=str(cme_widget_with_sql_metadata.uuid),
            sql_widget_dict=widget_dict,
            generate_query_only=False,
        ):
            events.append(event)

    assert len(events) == 2
    assert isinstance(events[0], StatusUpdateSSE)
    assert events[0].data.message == "SQL query generated"
    assert len(events[0].data.artifacts) == 1
    assert events[0].data.artifacts[0].type == "snowflake_query"

    assert isinstance(events[1], SqlQueryFunctionCallResult)
    assert (
        events[1].sql_query == "SELECT * FROM CME_DATA.PUBLIC.CBT_SETTLEMENTS LIMIT 10"
    )
    assert len(events[1].artifacts) == 1
    assert events[1].artifacts[0].data_format.parse_as == "snowflake_query"
    assert (
        events[1].artifacts[0].content
        == "```sql\nSELECT * FROM CME_DATA.PUBLIC.CBT_SETTLEMENTS LIMIT 10\n```"
    )
