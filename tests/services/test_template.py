from types import SimpleNamespace
from uuid import UUID, uuid4

import pandas as pd
import pytest
from openbb_ai.models import (
    Citation,
    DonutChartParameters,
    LineChartParameters,
    LlmClientMessage,
    PieChartParameters,
    RawObjectDataFormat,
    ScatterChartParameters,
    SourceInfo,
    Widget,
    WidgetCollection,
    WidgetParam,
)

from openbb_ada.models import (
    AvailableSemanticView,
    CopilotArtifact,
    Document,
    SkillCatalogEntry,
    SkillPayload,
    SqlTableInfo,
    SqlWidgetContext,
    StructuredContext,
    UnstructuredContext,
    WebContext,
)
from openbb_ada.services import TemplateService
from openbb_ada.utils.utils import get_current_datetime
from tests.conftest import MockUUIDs


def test_template_service_render_copilot_system_prompt_template_empty():
    template_service = TemplateService()

    actual_result = template_service.render_copilot_system_prompt()
    assert "No widgets found in explicit context" in actual_result
    assert "No widgets found on the dashboard" in actual_result
    assert "Global widget search is not enabled" in actual_result
    assert "## User Files" not in actual_result


def test_template_service_render_copilot_system_prompt_workflow_guidance():
    """Validate model-facing workflow guidance in the rendered system prompt."""
    template_service = TemplateService()

    actual_result = template_service.render_copilot_system_prompt()

    # Multi-query widget flows should map to one artifact per tool call.
    assert "When you call a widget more than once" in actual_result
    assert "one artifact per call" in actual_result
    # File/image questions should route to file-query tooling, not ad-hoc reasoning.
    assert (
        "When handling queries related to uploaded files or images, use the "
        "_llm_query_uploaded_files tool"
    ) in actual_result
    assert "DO NOT DUPLICATE RETRIEVAL PATHS" in actual_result
    assert "EXTRA WIDGETS ARE FALLBACK ONLY" in actual_result


def test_template_service_render_copilot_system_prompt_semantic_views_override(
    monkeypatch,
):
    monkeypatch.setattr("openbb_ada.constants.SNOWFLAKE_NATIVE_APP", True)
    template_service = TemplateService(semantic_views=["DB.OLD.DEFAULT_VIEW"])

    actual_result = template_service.render_copilot_system_prompt(
        semantic_views=["DB.SCHEMA.REVENUE_VIEW"]
    )

    assert "DB.SCHEMA.REVENUE_VIEW" in actual_result
    assert "DB.OLD.DEFAULT_VIEW" not in actual_result


def test_template_service_render_copilot_system_prompt_available_semantic_views(
    monkeypatch,
):
    monkeypatch.setattr("openbb_ada.constants.SNOWFLAKE_NATIVE_APP", True)
    template_service = TemplateService(
        available_semantic_views=[
            AvailableSemanticView(
                fqn="DB.SCHEMA.REVENUE_VIEW",
                database="DB",
                schema_name="SCHEMA",
                view_name="REVENUE_VIEW",
                base_table="REVENUE",
                comment="Curated revenue metrics",
            )
        ]
    )

    actual_result = template_service.render_copilot_system_prompt()

    assert "Available Semantic Views" in actual_result
    assert "DB.SCHEMA.REVENUE_VIEW" in actual_result
    assert "Base Table: REVENUE" in actual_result
    assert "semantic_view` argument" in actual_result
    assert "query the underlying base table(s)" in actual_result


def test_template_service_hides_snowflake_semantic_prompt_outside_native_mode(
    monkeypatch,
):
    monkeypatch.setattr("openbb_ada.constants.SNOWFLAKE_NATIVE_APP", False)
    template_service = TemplateService(
        semantic_views=["DB.SCHEMA.REVENUE_VIEW"],
        available_semantic_views=[
            AvailableSemanticView(
                fqn="DB.SCHEMA.REVENUE_VIEW",
                database="DB",
                schema_name="SCHEMA",
                view_name="REVENUE_VIEW",
                base_table="REVENUE",
            )
        ],
    )

    actual_result = template_service.render_copilot_system_prompt()

    assert "Selected Semantic Views" not in actual_result
    assert "Available Semantic Views" not in actual_result
    assert "llm_generate_sql_query_snowflake" not in actual_result


def test_template_service_render_sql_query_generation_prompt_allows_incomplete_schema():
    template_service = TemplateService()

    actual_result = template_service.render_sql_query_generation_prompt(
        sql_schema={},
        current_sql="SELECT * FROM DATA",
        data_sample=[{"date": "2026-05-14", "close": 10.5}],
    )

    assert "schema context is empty or incomplete" in actual_result
    assert "instead of rejecting solely because they are absent" in actual_result
    assert '"close": 10.5' in actual_result
    assert "local `DATA` table" not in actual_result
    assert "local table named `DATA`" not in actual_result


def test_template_service_render_sql_query_generation_prompt_keeps_strict_schema_rule():
    template_service = TemplateService()

    actual_result = template_service.render_sql_query_generation_prompt(
        sql_schema={"users": {"columns": [{"name": "id"}]}},
    )

    assert "When database schema is provided" in actual_result
    assert "ONLY generate queries for tables and columns" in actual_result


def test_template_service_render_copilot_system_prompt_template_with_context_widgets_files(  # noqa: E501
    mock_uuids: type[MockUUIDs],
):
    template_service = TemplateService()

    actual_result = template_service.render_copilot_system_prompt(
        unstructured_context=[
            UnstructuredContext(
                content="context a",
                source_info=SourceInfo(
                    type="widget",
                    uuid=UUID(mock_uuids.ID1.value),
                    name="widget_a",
                    description="desc widget_a",
                    metadata={"ticker": "TICKER_A"},
                ),
            ),
        ],
        widget_collection=WidgetCollection(
            primary=[
                Widget(
                    uuid=UUID(mock_uuids.ID1.value),
                    origin="origin_1",
                    widget_id="widget_a",
                    name="widget_a",
                    description="desc widget_a",
                    params=[
                        WidgetParam(
                            name="ticker",
                            type="string",
                            description="ticker symbol",
                            current_value="TICKER_A",
                        ),
                    ],
                    metadata={},
                )
            ],
            secondary=[],
            extra=[],
        ),
        documents=[
            Document(
                content=b"mock file content",
                file_uuid=uuid4(),
                filename="file1.txt",
                extension="txt",
                source_info=SourceInfo(
                    type="widget",
                    uuid=UUID(mock_uuids.ID1.value),
                    name="a test file about life",
                    description="a description of the test file",
                ),
            )
        ],
    )

    assert "# Available widgets" in actual_result
    assert "## Current Session Context" in actual_result
    assert "## Loaded User Files" in actual_result


def test_template_service_render_copilot_system_prompt_template_with_widgets(
    mock_uuids: type[MockUUIDs],
):
    template_service = TemplateService()

    test_widget_collection = WidgetCollection(
        primary=[
            Widget(
                uuid=UUID(mock_uuids.ID1.value),
                origin="origin_1",
                widget_id="widget_a",
                name="widget_a",
                description="desc widget_a",
                params=[
                    WidgetParam(
                        name="ticker",
                        type="string",
                        description="ticker symbol",
                        current_value="TICKER_A",
                    ),
                    WidgetParam(
                        name="date",
                        type="string",
                        description="date",
                        current_value="2024-01-01",
                    ),
                ],
                metadata={},
            ),
            Widget(
                uuid=UUID(mock_uuids.ID2.value),
                origin="origin_2",
                widget_id="widget_b",
                name="widget_b",
                description="desc widget_b",
                params=[
                    WidgetParam(
                        name="ticker",
                        type="string",
                        description="ticker symbol",
                        current_value="TICKER_B",
                    ),
                    WidgetParam(
                        name="start_date",
                        type="string",
                        description="start date",
                        current_value="2024-01-02",
                    ),
                ],
                metadata={},
            ),
            Widget(
                uuid=UUID(mock_uuids.ID3.value),
                origin="origin_3",
                widget_id="widget_c",
                name="widget_c",
                description="desc widget_c",
                params=[
                    WidgetParam(
                        name="ticker",
                        type="string",
                        multi_select=True,
                        description="ticker symbol",
                        current_value="TICKER_C",
                    ),
                    WidgetParam(
                        name="end_date",
                        type="string",
                        description="end date",
                        current_value="2024-01-03",
                    ),
                ],
                metadata={},
            ),
        ],
        secondary=[],
        extra=[],
    )

    actual_result = template_service.render_copilot_system_prompt(
        widget_collection=test_widget_collection
    )

    assert "## Available widgets" in actual_result
    assert "llm_get_widget_input_state" in actual_result

    assert (
        """\
uuid: 08d47a2f-bd35-4f53-a0e6-a45b4c7252f0
name: widget_a
description: desc widget_a
origin: origin_1
input arguments:
  ticker ([string]) = TICKER_A
     ticker symbol
     get_options: False
  date ([string]) = 2024-01-01
     date
     get_options: False
---
uuid: a1b2c3d4-1fbc-472a-852a-cc96adff0701
name: widget_b
description: desc widget_b
origin: origin_2
input arguments:
  ticker ([string]) = TICKER_B
     ticker symbol
     get_options: False
  start_date ([string]) = 2024-01-02
     start date
     get_options: False
---
uuid: f47ac10b-58cc-4372-a567-0e02b2c3d479
name: widget_c
description: desc widget_c
origin: origin_3
input arguments:
  ticker (list[string]) = TICKER_C
     ticker symbol
     get_options: False
  end_date ([string]) = 2024-01-03
     end date
     get_options: False
---
"""
        in actual_result
    )

    assert "No widgets are available" not in actual_result


def test_template_service_render_copilot_prompt_with_unapplied_param_draft(
    mock_uuids: type[MockUUIDs],
):
    template_service = TemplateService()

    actual_result = template_service.render_copilot_system_prompt(
        widget_collection=WidgetCollection(
            primary=[
                Widget(
                    uuid=UUID(mock_uuids.ID1.value),
                    origin="origin_sql",
                    widget_id="ssrm_sql_widget",
                    name="SQL Widget",
                    description="SQL query editor",
                    params=[
                        WidgetParam(
                            name="query",
                            type="string",
                            description="SQL query",
                            current_value="SELECT * FROM orders LIMIT 10",
                            executed_value="SELECT * FROM orders LIMIT 5",
                        ),
                    ],
                    metadata={},
                )
            ],
            secondary=[],
            extra=[],
        )
    )

    assert "query ([string]) = SELECT * FROM orders LIMIT 10" in actual_result
    assert "last_executed_value: SELECT * FROM orders LIMIT 5" in actual_result
    assert "note: current_value is an unapplied editor draft." in actual_result
    assert (
        "If the user asks for the current prompt/query/input arguments" in actual_result
    )


def test_template_service_render_copilot_system_prompt_template_no_widgets():
    template_service = TemplateService()

    test_widget_collection = WidgetCollection(primary=[], secondary=[], extra=[])

    actual_result = template_service.render_copilot_system_prompt(
        widget_collection=test_widget_collection
    )

    assert "No widgets found in explicit context" in actual_result
    assert "No widgets found on the dashboard" in actual_result
    assert "Global widget search is not enabled" in actual_result


def test_template_service_render_prompt_enhancement_prompt_stays_domain_agnostic():
    template_service = TemplateService()

    actual_result = template_service.render_prompt_enhancement_prompt(
        messages=[LlmClientMessage(role="human", content="Compare these datasets")],
    )

    assert "You are a prompt enhancement expert." in actual_result
    assert "specializing in financial and investment queries" not in actual_result
    assert "Add domain context carefully" in actual_result


def test_template_service_render_prompt_enhancement_prompt_prioritizes_active_tab():
    workspace_state = SimpleNamespace(
        current_page_context="dashboard",
        current_dashboard_info=SimpleNamespace(
            name="Amex Workspace",
            current_tab_id="active-tab",
            tabs=[
                SimpleNamespace(
                    tab_id="active-tab",
                    widgets=[SimpleNamespace(name="Spend Breakdown")],
                ),
                SimpleNamespace(
                    tab_id="other-tab",
                    widgets=[SimpleNamespace(name="Other Widget")],
                ),
            ],
        ),
    )
    template_service = TemplateService(workspace_state=workspace_state)

    actual_result = template_service.render_prompt_enhancement_prompt(
        messages=[
            LlmClientMessage(role="human", content="Summarize what matters here")
        ],
        widgets=SimpleNamespace(
            primary=[],
            secondary=[
                SimpleNamespace(
                    name="Spend Breakdown",
                    description="Transaction data by category",
                    params=[],
                ),
                SimpleNamespace(
                    name="Other Widget",
                    description="Historical context",
                    params=[],
                ),
            ],
        ),
    )

    assert "Current tab widgets: Spend Breakdown" in actual_result
    expected_guidance = (
        "Prefer current tab widgets over other dashboard widgets unless the user "
        "asks for something broader."
    )
    assert expected_guidance in actual_result


def test_template_service_render_copilot_system_prompt_template_with_unstructured_contextual_widgets(  # noqa: E501
    mock_uuids: type[MockUUIDs],
):
    template_service = TemplateService()

    test_context = [
        UnstructuredContext(
            content="context a",
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID1.value),
                name="widget_a",
                description="desc widget_a",
                metadata={"ticker": "TICKER_A"},
            ),
        ),
        UnstructuredContext(
            content="context b",
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID2.value),
                name="widget_b",
                description="desc widget_b",
                metadata={"ticker": "TICKER_B"},
            ),
        ),
    ]

    actual_result = template_service.render_copilot_system_prompt(
        unstructured_context=test_context
    )

    assert "## Current Session Context" in actual_result

    assert mock_uuids.ID1.value in actual_result
    assert "widget_a" in actual_result
    assert "desc widget_a" in actual_result
    assert "context a" in actual_result
    assert "TICKER_A" in actual_result

    assert mock_uuids.ID2.value in actual_result
    assert "widget_b" in actual_result
    assert "desc widget_b" in actual_result
    assert "context b" in actual_result
    assert "TICKER_B" in actual_result


def test_template_service_render_copilot_system_prompt_template_with_unstructured_contextual_widgets_with_date_metadata(  # noqa: E501
    mock_uuids: type[MockUUIDs],
):
    template_service = TemplateService()

    test_context = [
        UnstructuredContext(
            content="context a",
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID1.value),
                name="widget_a",
                description="desc widget_a",
                metadata={"ticker": "TICKER_A", "date": "2024-01-01"},
            ),
        ),
        UnstructuredContext(
            content="context b",
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID2.value),
                name="widget_b",
                description="desc widget_b",
                metadata={"ticker": "TICKER_B", "start_date": "2024-01-02"},
            ),
        ),
        UnstructuredContext(
            content="context c",
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID3.value),
                name="widget_c",
                description="desc widget_c",
                metadata={"ticker": "TICKER_C", "end_date": "2024-01-03"},
            ),
        ),
    ]

    actual_result = template_service.render_copilot_system_prompt(
        unstructured_context=test_context
    )

    assert "## Current Session Context" in actual_result

    assert mock_uuids.ID1.value in actual_result
    assert "widget_a" in actual_result
    assert "desc widget_a" in actual_result
    assert "context a" in actual_result
    assert "TICKER_A" in actual_result

    assert mock_uuids.ID2.value in actual_result
    assert "widget_b" in actual_result
    assert "desc widget_b" in actual_result
    assert "context b" in actual_result
    assert "TICKER_B" in actual_result

    assert mock_uuids.ID3.value in actual_result
    assert "widget_c" in actual_result
    assert "desc widget_c" in actual_result
    assert "context c" in actual_result
    assert "TICKER_C" in actual_result

    assert "earlier than the start_date" in actual_result
    assert "later than the end_date" in actual_result
    assert "doesn't match the date" in actual_result


def test_template_service_render_copilot_system_prompt_template_with_structured_context(
    mock_uuids: type[MockUUIDs],
):  # noqa: E501
    template_service = TemplateService()

    test_structured_context = [
        StructuredContext(
            content="{some data}",
            sql_table_info=SqlTableInfo(
                table_name="stock_prices_aapl",
                sql_schema="the table schema for aapl",
                unique_column_values={},
            ),
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID1.value),
                name="stock price widget",
                description="contains historical stock prices of a ticker.",
                metadata={"ticker": "AAPL"},
            ),
            data_format=RawObjectDataFormat(parse_as="table"),
        ),
        StructuredContext(
            content="{some data}",
            sql_table_info=SqlTableInfo(
                table_name="stock_prices_msft",
                sql_schema="the table schema for msft",
                unique_column_values={},
            ),
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID2.value),
                name="stock price widget",
                description="contains historical stock prices of a ticker.",
                metadata={"ticker": "MSFT"},
            ),
            data_format=RawObjectDataFormat(parse_as="table"),
        ),
    ]

    actual_result = template_service.render_copilot_system_prompt(
        structured_context=test_structured_context
    )
    assert "## Available Data Sources" in actual_result

    assert "contains historical stock prices of a ticker." in actual_result
    assert "AAPL" in actual_result
    assert "stock_prices_aapl" in actual_result

    assert "contains historical stock prices of a ticker." in actual_result
    assert "MSFT" in actual_result
    assert "stock_prices_msft" in actual_result


def test_template_service_render_copilot_system_prompt_template_with_structured_context_with_date_metadata(  # noqa: E501
    mock_uuids: type[MockUUIDs],
):
    template_service = TemplateService()

    test_structured_context = [
        StructuredContext(
            content="{some data}",
            sql_table_info=SqlTableInfo(
                table_name="stock_prices_aapl",
                sql_schema="the table schema for aapl",
                unique_column_values={},
            ),
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID1.value),
                name="stock price widget",
                description="contains historical stock prices of a ticker.",
                metadata={"ticker": "AAPL", "date": "2024-01-01"},
            ),
            data_format=RawObjectDataFormat(parse_as="table"),
        ),
        StructuredContext(
            content="{some data}",
            sql_table_info=SqlTableInfo(
                table_name="stock_prices_msft",
                sql_schema="the table schema for msft",
                unique_column_values={},
            ),
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID2.value),
                name="stock price widget",
                description="contains historical stock prices of a ticker.",
                metadata={"ticker": "MSFT", "start_date": "2024-01-02"},
            ),
            data_format=RawObjectDataFormat(parse_as="table"),
        ),
        StructuredContext(
            content="{some data}",
            sql_table_info=SqlTableInfo(
                table_name="stock_prices_goog",
                sql_schema="the table schema for goog",
                unique_column_values={},
            ),
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID3.value),
                name="stock price widget",
                description="contains historical stock prices of a ticker.",
                metadata={"ticker": "GOOG", "end_date": "2024-01-03"},
            ),
            data_format=RawObjectDataFormat(parse_as="table"),
        ),
    ]

    actual_result = template_service.render_copilot_system_prompt(
        structured_context=test_structured_context
    )
    assert "## Available Data Sources" in actual_result

    assert "contains historical stock prices of a ticker." in actual_result
    assert "AAPL" in actual_result
    assert "stock_prices_aapl" in actual_result

    assert "contains historical stock prices of a ticker." in actual_result
    assert "MSFT" in actual_result
    assert "stock_prices_msft" in actual_result

    assert "contains historical stock prices of a ticker." in actual_result
    assert "GOOG" in actual_result
    assert "stock_prices_goog" in actual_result

    assert "earlier than the start_date" in actual_result
    assert "later than the end_date" in actual_result
    assert "doesn't match the date" in actual_result


def test_template_service_render_copilot_system_prompt_template_with_structured_and_unstructured_context(  # noqa: E501
    mock_uuids: type[MockUUIDs],
):
    template_service = TemplateService()

    test_structured_context = [
        StructuredContext(
            content="{some data}",
            sql_table_info=SqlTableInfo(
                table_name="stock_prices_aapl",
                sql_schema="the table schema for aapl",
                unique_column_values={},
            ),
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID1.value),
                name="stock price widget",
                description="contains historical stock prices of a ticker.",
                metadata={"ticker": "AAPL"},
            ),
            data_format=RawObjectDataFormat(parse_as="table"),
        )
    ]

    test_unstructured_context = [
        UnstructuredContext(
            content="<the earnings transcript>",
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID2.value),
                name="earnings transcript aapl",
                description="Contains the latest earnings transcript for AAPL",
                metadata={"ticker": "AAPL"},
            ),
        )
    ]

    actual_result = template_service.render_copilot_system_prompt(
        structured_context=test_structured_context,
        unstructured_context=test_unstructured_context,
    )
    assert "## Available Data Sources" in actual_result

    assert "contains historical stock prices of a ticker." in actual_result
    assert "AAPL" in actual_result
    assert "stock_prices_aapl" in actual_result


def test_template_service_render_copilot_system_prompt_template_with_web_pages():
    template_service = TemplateService()

    test_web_contexts = [
        WebContext(
            url="https://wikipedia.org",
            content="Wikipedia is a free online encyclopedia",
            citation=Citation(
                source_info=SourceInfo(
                    type="web",
                    name="https://wikipedia.org",
                )
            ),
        ),
        WebContext(
            url="https://openbb.co",
            content="OpenBB is an open-source library for the financial analysis community",  # noqa: E501
            citation=Citation(
                source_info=SourceInfo(
                    type="web",
                    name="https://openbb.co",
                )
            ),
        ),
    ]

    actual_result = template_service.render_copilot_system_prompt(
        web_pages=test_web_contexts
    )

    assert "### Web Search Results" in actual_result
    assert (
        """\
url: https://wikipedia.org
content: Wikipedia is a free online encyclopedia
"""
        in actual_result
    )

    assert (
        """\
url: https://openbb.co
content: OpenBB is an open-source library for the financial analysis community
"""
        in actual_result
    )


def test_template_service_render_copilot_system_prompt_template_with_no_web_pages():
    template_service = TemplateService()

    test_web_contexts = None
    actual_result = template_service.render_copilot_system_prompt(
        web_pages=test_web_contexts
    )

    assert "## Retrieved from the Web" not in actual_result


def test_template_service_render_copilot_system_prompt_template_with_documents(
    mock_uuids: type[MockUUIDs],
):
    template_service = TemplateService()
    test_documents = [
        Document(
            content=b"mock file content",
            filename="test_file_1.txt",
            extension="txt",
            file_uuid=uuid4(),
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID1.value),
                name="a test file about life",
                description="a description of the test file",
            ),
        ),
        Document(
            content=b"mock file content 2",
            filename="test_file_2.pdf",
            extension="pdf",
            file_uuid=uuid4(),
            source_info=SourceInfo(
                type="widget",
                uuid=UUID(mock_uuids.ID2.value),
                name="a test document about love",
                description="a description of the test document",
            ),
        ),
    ]

    actual_result = template_service.render_copilot_system_prompt(
        documents=test_documents
    )

    assert "## Loaded User Files" in actual_result

    assert mock_uuids.ID1.value in actual_result
    assert "test_file_1.txt" in actual_result
    assert "a test file about life" in actual_result

    assert mock_uuids.ID2.value in actual_result
    assert "test_file_2.pdf" in actual_result
    assert "a test document about love" in actual_result


def test_template_service_render_copilot_system_prompt_template_with_no_documents():
    template_service = TemplateService()

    test_documents = None

    actual_result = template_service.render_copilot_system_prompt(
        documents=test_documents
    )

    assert "## User Files" not in actual_result


def test_template_service_render_copilot_sql_agent_system_prompt():
    template_service = TemplateService()

    test_sql_tables_info = [
        SqlTableInfo(
            table_name="stock_prices_aapl",
            sql_schema="the table schema for aapl",
            unique_column_values={},
        ),
        SqlTableInfo(
            table_name="stock_prices_msft",
            sql_schema="the table schema for msft",
            unique_column_values={"positions": ["CEO", "CFO"]},
        ),
    ]

    actual_result = template_service.render_copilot_sql_agent_system_prompt(
        sql_tables_info=test_sql_tables_info,
    )

    assert "the table schema for aapl" in actual_result
    assert "the table schema for msft" in actual_result
    assert "positions" in actual_result
    assert "CEO" in actual_result
    assert "CFO" in actual_result

    assert "SQLite" in actual_result


def test_template_service_render_copilot_native_function_call_result_unstructured():
    template_service = TemplateService()
    test_artifact = CopilotArtifact(
        content="some text",
        source_info=SourceInfo(
            uuid=uuid4(),
            type="direct retrieval",
            name="test_file.txt",
            description="File accessed.",
        ),
        data_format=RawObjectDataFormat(parse_as="text"),
    )

    actual_result = template_service.render_copilot_native_function_call_result(
        answer="some answer",
        artifact=test_artifact,
    )

    assert "Result / Guidance: some answer" in actual_result
    assert f"Artifact UUID: `{test_artifact.source_info.uuid}`" in actual_result
    assert f"Artifact Name: `{test_artifact.source_info.name}`" in actual_result
    assert "Artifact Sample" in actual_result
    assert "some text" in actual_result
    assert (
        f"You can insert this artifact in your final response by writing <|start_artifact_id|>{test_artifact.source_info.name}<|end_artifact_id|>"  # noqa: E501
        in actual_result
    )
    assert (
        "IMPORTANT: This is only a sample. You MUST retrieve the full artifact if you need to take further actions with this data."  # noqa: E501
        in actual_result
    )


def test_template_service_render_copilot_native_function_call_result_structured():
    template_service = TemplateService()
    test_artifact = CopilotArtifact(
        content=pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).to_json(
            orient="records"
        ),
        source_info=SourceInfo(
            type="widget",
            name="test_file.csv",
            description="File accessed.",
        ),
        data_format=RawObjectDataFormat(type="table"),
    )

    actual_result = template_service.render_copilot_native_function_call_result(
        answer="some answer",
        table_preview="some table preview",
        artifact=test_artifact,
    )
    assert "Result / Guidance: some answer" in actual_result
    assert f"Artifact UUID: `{test_artifact.source_info.uuid}`" in actual_result
    assert f"Artifact Name: `{test_artifact.source_info.name}`" in actual_result
    assert "Artifact Sample" in actual_result
    assert "some table preview" in actual_result
    assert (
        f"You can insert this artifact in your final response by writing <|start_artifact_id|>{test_artifact.source_info.name}<|end_artifact_id|>"  # noqa: E501
        in actual_result
    )
    assert (
        "IMPORTANT: This is only a sample. You MUST retrieve the full artifact if you need to take further actions with this data."  # noqa: E501
        in actual_result
    )


@pytest.mark.parametrize(
    "chart_params,expected_outputs",
    [
        (
            LineChartParameters(
                chartType="line",
                xKey="x-variable",
                yKey=["y-variable"],
            ),
            [
                "A `line` chart",
                "showing the `x-variable` variable on the X-axis",
                "and the `y-variable` variable on the Y-axis",
            ],
        ),
        (
            LineChartParameters(
                chartType="line",
                xKey="date",
                yKey=["price", "volume"],
            ),
            [
                "A `line` chart",
                "showing the `date` variable on the X-axis",
                "and the `price`, `volume` variables on the Y-axis",
            ],
        ),
        (
            PieChartParameters(
                chartType="pie", angleKey="value", calloutLabelKey="category"
            ),
            [
                "A `pie` chart",
                "displaying `value` values",
                "with `category` labels",
            ],
        ),
        (
            ScatterChartParameters(
                chartType="scatter",
                xKey="x-scatter",
                yKey=["y-scatter"],
            ),
            [
                "A `scatter` chart",
                "showing the `x-scatter` variable on the X-axis",
                "and the `y-scatter` variable on the Y-axis",
            ],
        ),
        (
            DonutChartParameters(
                chartType="donut", angleKey="percentage", calloutLabelKey="segment"
            ),
            [
                "A `donut` chart",
                "displaying `percentage` values",
                "with `segment` labels",
            ],
        ),
    ],
    ids=["single_line", "multi_line", "pie", "scatter", "donut"],
)
def test_template_service_render_copilot_native_function_call_result_charts(
    chart_params, expected_outputs
):
    template_service = TemplateService()
    test_artifact = CopilotArtifact(
        content="some text",
        source_info=SourceInfo(
            type="artifact",
            uuid=uuid4(),
            name="test_chartifact",
            description="test chartifact",
        ),
        data_format=RawObjectDataFormat(
            parse_as="chart",
            chart_params=chart_params,
        ),
    )

    actual_result = template_service.render_copilot_native_function_call_result(
        answer="This is an answer",
        artifact=test_artifact,
    )

    # Common assertions for all chart types
    assert "Result / Guidance: This is an answer" in actual_result
    assert f"Artifact Name: `{test_artifact.source_info.name}`" in actual_result
    assert "Artifact Sample" in actual_result

    # Check specific chart outputs
    for expected in expected_outputs:
        assert expected in actual_result


@pytest.mark.parametrize(
    "input_str,expected_output",
    [
        ("hello", "hello"),
        ("{hello}", "{{hello}}"),
        ("hello {world}", "hello {{world}}"),
        ("{{hello}}", "{{hello}}"),
        ("hello {{world}}", "hello {{world}}"),
        ("{hello} {{world}}", "{{hello}} {{world}}"),
        ("{{hello}} {world}", "{{hello}} {{world}}"),
        ("{a} {b} {c}", "{{a}} {{b}} {{c}}"),
        ("{{a}} {b} {{c}}", "{{a}} {{b}} {{c}}"),
        ("", ""),
        ("{}", "{{}}"),
        ("{{}", "{{}}"),
        ("{{}}", "{{}}"),
        (123, "123"),
        (None, "None"),
        ("{value:.2f}", "{{value:.2f}}"),
        ("Hello {name:>10}!", "Hello {{name:>10}}!"),
        ("{ {nested} }", "{{ {{nested}} }}"),
        ("The value is {value}", "The value is {{value}}"),
        ("Price: ${price:.2f}", "Price: ${{price:.2f}}"),
        ("From: {start}, To: {end}", "From: {{start}}, To: {{end}}"),
    ],
    ids=[
        "plain_text",
        "single_brace",
        "single_brace_with_text",
        "double_brace",
        "double_brace_with_text",
        "mixed_single_double",
        "mixed_double_single",
        "multiple_single",
        "multiple_mixed",
        "empty_string",
        "empty_braces",
        "unmatched_double",
        "matched_double",
        "number_input",
        "none_input",
        "format_spec",
        "alignment_spec",
        "nested_braces",
        "example_simple",
        "example_price",
        "example_range",
    ],
)
def test_filter_escape_string(input_str, expected_output):
    result = TemplateService._filter_escape_string(input_str)
    assert result == expected_output


def test_render_copilot_docs_create_sql_table_name_prompt():
    template_service = TemplateService()
    actual_result = template_service.render_copilot_docs_create_sql_table_name_prompt(
        "test_file.csv"
    )
    assert "test_file" in actual_result


@pytest.mark.parametrize(
    "query,expected_outputs",
    [
        # Basic query
        ("Show me the data", ["Show me the data"]),
        # Query with braces - should be passed through as-is
        (
            "Query with {braces}",
            ["Query with {braces}"],  # Braces should not be escaped in user prompts
        ),
        # Empty query
        (
            "",
            [],  # Test empty query
        ),
    ],
    ids=["basic_query", "query_with_braces", "empty_query"],
)
def test_template_service_render_copilot_user_prompt(query, expected_outputs):
    template_service = TemplateService()
    result = template_service.render_copilot_user_prompt(query)

    for expected in expected_outputs:
        assert expected in result


def test_template_service_render_system_prompt_with_timezone():
    template_service = TemplateService(
        # doesn't observe daylight savings, so won't change on us!
        current_datetime=get_current_datetime(timezone="America/Phoenix")
    )

    actual_result = template_service.render_copilot_system_prompt()
    assert "MST" in actual_result


def test_template_service_render_copilot_snowflake_query_result(mock_uuids):
    """Test rendering of snowflake_query artifact type."""
    template_service = TemplateService()

    test_artifact = CopilotArtifact(
        content="SELECT * FROM users WHERE id > 100",
        source_info=SourceInfo(
            uuid=UUID(mock_uuids.ID1.value),
            type="artifact",
            name="user_query_artifact",
            description="Query for users with ID > 100",
            metadata={
                "parse_as": "snowflake_query",
                "query_data_source": {
                    "origin": "SNOW backend",
                    "id": "snowflake_users_table",
                    "widget_uuid": mock_uuids.ID2.value,
                },
            },
            citable=False,
        ),
        data_format=RawObjectDataFormat(
            parse_as="snowflake_query",
            query_data_source={
                "origin": "SNOW backend",
                "id": "snowflake_users_table",
                "widget_uuid": mock_uuids.ID2.value,
            },
        ),
    )

    actual_result = template_service.render_copilot_snowflake_query_result(
        sql_query="SELECT * FROM users WHERE id > 100",
        artifact=test_artifact,
        generate_query_only=True,
    )

    # Verify template output includes key elements
    assert f"Artifact Name: `{test_artifact.source_info.name}`" in actual_result
    assert "SELECT * FROM users WHERE id > 100" in actual_result
    assert "SNOW backend" in actual_result
    assert "snowflake_users_table" in actual_result
    assert "reuse this SQL exactly as generated in the artifact" in actual_result
    assert (
        "<|start_artifact_id|>user_query_artifact<|end_artifact_id|>" in actual_result
    )


def test_template_service_render_copilot_snowflake_query_result_for_follow_up_action(
    mock_uuids,
):
    template_service = TemplateService()

    test_artifact = CopilotArtifact(
        content="SELECT * FROM users LIMIT 5",
        source_info=SourceInfo(
            uuid=UUID(mock_uuids.ID1.value),
            type="artifact",
            name="user_query_artifact",
            description="Top 5 rows query",
            metadata={
                "parse_as": "snowflake_query",
                "query_data_source": {
                    "origin": "SNOW backend",
                    "id": "snowflake_users_table",
                    "widget_uuid": mock_uuids.ID2.value,
                },
            },
            citable=False,
        ),
        data_format=RawObjectDataFormat(
            parse_as="snowflake_query",
            query_data_source={
                "origin": "SNOW backend",
                "id": "snowflake_users_table",
                "widget_uuid": mock_uuids.ID2.value,
            },
        ),
    )

    actual_result = template_service.render_copilot_snowflake_query_result(
        sql_query="SELECT * FROM users LIMIT 5",
        artifact=test_artifact,
        generate_query_only=False,
    )

    assert "Do NOT stop after only returning the artifact" in actual_result
    assert "llm_generate_widget_in_dashboard" in actual_result
    assert "llm_update_widget_in_dashboard" in actual_result
    assert "another/separate dashboard widget" in actual_result
    assert 'Use `widget_type="table"` for table output' in actual_result
    assert 'or `widget_type="chart"` with the required `chart_params`' in actual_result
    assert "Only call `llm_update_widget_in_dashboard`" in actual_result
    assert "use the SQL from this artifact exactly as generated" in actual_result
    assert "always respond with this artifact reference" not in actual_result
    assert "GENERATED but NOT automatically executed" not in actual_result


def test_system_prompt_prefers_new_widget_for_additive_sql_dashboard_intent():
    template_service = TemplateService(workspace_options={"generative-ui": True})

    actual_result = template_service.render_copilot_system_prompt(
        sql_widgets=[
            SqlWidgetContext(
                widget_uuid="widget-uuid-1",
                widget_id="fred_public_dsge",
                widget_name="DSGE Table",
                widget_origin="SNOW backend",
                widget_description="DSGE data",
                sql_schema={
                    "columns": [
                        {"name": "date", "type": "DATE"},
                        {"name": "value", "type": "FLOAT"},
                    ]
                },
                sql_schema_sanitized=(
                    "CREATE TABLE fred_public_dsge (date DATE, value FLOAT)"
                ),
                current_sql="SELECT * FROM fred_public_dsge",
            )
        ]
    )

    assert "Treat dashboard intent as **additive by default**" in actual_result
    assert "DASHBOARD CREATION RULE" in actual_result
    assert "Call `llm_search_widgets` first" in actual_result
    assert "concise keyword query" in actual_result
    assert "broader adjacent terms implied by the request" in actual_result
    assert "Never create a dashboard from unrelated widgets" in actual_result
    assert (
        "Never call `llm_create_app` after an empty or irrelevant search result"
        in actual_result
    )
    assert "APP LAYOUT RULE" in actual_result
    assert "Only set `state.params` keys" in actual_result
    assert "widgets with params must set explicit `state.params`" in actual_result
    assert "current/default params do not fit" in actual_result
    assert "Do not choose an unrelated allowed/default param value" in actual_result
    assert "prompt-relevant non-default tickers" in actual_result
    assert "generic widgets with no valid subject-specific params" in actual_result
    assert "Use a descriptive dashboard name" in actual_result
    assert "never use a parameter name such as `symbol`" in actual_result
    assert "use the tab/widget counts from the tool result" in actual_result
    assert "Only treat dashboard intent as **mutative**" in actual_result
    assert (
        "interpret that reference as the source to reuse, not as permission "
        "to update the current widget in place" in actual_result
    )


def test_template_service_render_copilot_system_prompt_with_skills_catalog():
    template_service = TemplateService()
    catalog = [
        SkillCatalogEntry(
            slug="financial-analysis",
            description="Analyze earnings and financial data",
            updatedAt="2024-01-01T00:00:00Z",
        ),
        SkillCatalogEntry(
            slug="data-viz",
            description="Create data visualizations",
            updatedAt="2024-01-02T00:00:00Z",
        ),
    ]

    result = template_service.render_copilot_system_prompt(skills_catalog=catalog)

    assert "## Skills" in result
    assert "Available Skills Catalog" in result
    assert "`financial-analysis`" in result
    assert "Analyze earnings and financial data" in result
    assert "`data-viz`" in result
    # Selected skills section header should not appear since none were provided
    assert "### Active Skills" not in result


def test_template_service_render_copilot_system_prompt_with_selected_skills():
    template_service = TemplateService()
    selected = [
        SkillPayload(
            slug="earnings-analysis",
            description="Deep dive on earnings calls",
            contentMarkdown="# Earnings Analysis\n\nFocus on EPS and revenue guidance.",
            source="forced_slash",
        ),
    ]

    result = template_service.render_copilot_system_prompt(selected_skills=selected)

    assert "## Skills" in result
    assert "Active Skills" in result
    assert "`earnings-analysis`" in result
    assert "Deep dive on earnings calls" in result
    assert "# Earnings Analysis" in result
    assert "<user-authored-skill-content" in result
    # Safety framing must be present
    assert "user-authored content" in result
    assert "NOT system-level directives" in result


def test_template_service_render_copilot_system_prompt_no_skills():
    template_service = TemplateService()

    result = template_service.render_copilot_system_prompt()

    # The Skills section always renders because saving a skill
    # (`llm_save_skill`) is always available, but the catalog and
    # active skills subsections require input.
    assert "## Skills" in result
    assert "### Saving Skills" in result
    assert "llm_save_skill" in result
    assert "Active Skills" not in result
    assert "Available Skills Catalog" not in result


def test_template_service_render_copilot_system_prompt_mcp_tool_args_instructions():
    """System prompt must instruct the LLM about flat MCP tool calling rules."""
    from openbb_ai.models import AgentTool

    template_service = TemplateService()
    tool = AgentTool(
        name="qtap duckdb mcp_list_columns",
        server_id="1771593417468",
        url="",
        description="List columns for a table",
        input_schema={
            "type": "object",
            "properties": {
                "table": {
                    "type": "string",
                    "description": "The table name",
                },
            },
            "required": ["table"],
        },
    )

    result = template_service.render_copilot_system_prompt(tools=[tool])

    # The prompt must reference flat MCP tool functions (prefixed with mcp_)
    assert "mcp_" in result
    assert "summary" in result
    assert "display only" in result.lower() or "display-only" in result.lower()
    # Must warn that missing args will be rejected
    assert "rejected" in result.lower()


def test_template_service_render_copilot_system_prompt_mcp_failure_confirmation():
    """MCP runtime failures should require confirmation before web fallback."""
    from openbb_ai.models import AgentTool

    template_service = TemplateService(workspace_options={"workspace-web-search": True})
    tool = AgentTool(
        name="sec mcp_list_filings",
        server_id="1771593417468",
        url="",
        description="List company filings",
        input_schema={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The ticker symbol",
                },
            },
            "required": ["ticker"],
        },
    )

    result = template_service.render_copilot_system_prompt(tools=[tool])

    assert "MCP FAILURE OVERRIDE" in result
    assert "Do not treat MCP tool failures as ordinary fallback signals" in result
    assert "Wait for confirmation before retrying" in result
    assert "suggested fix, next step, or URL" in result
    assert "surface that solution explicitly" in result
    assert "do not automatically retry" in result
    assert "include any remediation guidance from the error text" in result
    assert "ask for confirmation before taking that next action" in result
