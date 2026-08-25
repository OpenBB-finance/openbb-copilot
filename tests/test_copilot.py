import asyncio
import json
from unittest.mock import Mock
from uuid import UUID

import pytest
from magentic import (
    AssistantMessage,
    AsyncStreamedResponse,
    AsyncStreamedStr,
    FunctionCall,
    FunctionResultMessage,
    UserMessage,
)
from magentic.chat_model.base import aparse_stream
from openbb_ai.models import (
    AgentTool,
    BarChartParameters,
    Citation,
    DataContent,
    FunctionCallSSE,
    FunctionCallSSEData,
    LlmClientFunctionCall,
    LlmClientFunctionCallResultMessage,
    LlmClientMessage,
    MessageArtifactSSE,
    MessageChunkSSE,
    RoleEnum,
    SingleDataContent,
    StatusUpdateSSE,
    StatusUpdateSSEData,
    Widget,
    WidgetCollection,
    WidgetParam,
)

from openbb_ada.errors import ToolLimitExceededError
from openbb_ada.models import (
    CopilotArtifact,
    RawObjectDataFormat,
    SourceInfo,
    SqlQueryFunctionCallResult,
    UnstructuredContext,
)
from openbb_ada.services import CitationService, ContextService, LoggingService


async def _async_text_stream(*chunks: str):
    for chunk in chunks:
        yield chunk


def test_parse_prompt_suggestions_removes_hidden_block(test_copilot_service):
    clean_text, suggestions = test_copilot_service._parse_prompt_suggestions(
        "Answer\n\n<suggestions>"
        "<suggestion>Show revenue growth</suggestion>"
        "<suggestion>Compare margin trends</suggestion>"
        "</suggestions>"
    )

    assert clean_text == "Answer"
    assert suggestions == ["Show revenue growth", "Compare margin trends"]


@pytest.mark.asyncio
async def test_stream_filter_emits_prompt_suggestions_without_leaking_tags(
    test_copilot_service,
):
    test_copilot_service._workspace_options = {"prompt-suggestions": True}
    test_copilot_service._citation_service = CitationService()

    events = [
        event
        async for event in test_copilot_service._handle_copilot_stream(
            _async_text_stream(
                "Answer",
                "\n<suggestions>",
                "<suggestion>Show revenue growth</suggestion>",
                "<suggestion>Compare margin trends</suggestion>",
                "</suggestions>",
            )
        )
    ]
    message = "".join(
        event.data.delta for event in events if isinstance(event, MessageChunkSSE)
    )

    assert message == "Answer\n"
    assert "<suggestions>" not in message

    final_events = [
        event async for event in test_copilot_service._finalize_copilot_stream([])
    ]

    assert len(final_events) == 1
    assert final_events[0].event == "copilotPromptSuggestions"
    assert final_events[0].data.suggestions == [
        "Show revenue growth",
        "Compare margin trends",
    ]


@pytest.mark.asyncio
async def test_finalize_stream_omits_empty_prompt_suggestions(test_copilot_service):
    test_copilot_service._workspace_options = {"prompt-suggestions": True}
    test_copilot_service._pending_prompt_suggestions = []
    test_copilot_service._citation_service = CitationService()

    final_events = [
        event async for event in test_copilot_service._finalize_copilot_stream([])
    ]

    assert final_events == []


@pytest.mark.parametrize(
    ("model", "token_count", "expected"),
    [
        ("gpt-5.5", 840_000, True),
        ("gpt-5.5", 840_001, False),
        ("gpt-5.4-mini", 320_000, True),
        ("gpt-5.4-mini", 320_001, False),
        ("unknown-model", 160_000, True),
        ("unknown-model", 160_001, False),
    ],
)
def test_is_within_context_limit_uses_model_specific_limits(
    test_copilot_service,
    monkeypatch,
    model,
    token_count,
    expected,
):
    monkeypatch.setattr(test_copilot_service, "_get_model", lambda: Mock(model=model))
    monkeypatch.setattr(
        test_copilot_service,
        "_count_tokens",
        lambda messages: token_count,
    )

    assert test_copilot_service._is_within_context_limit(messages=[]) is expected


def test_build_llm_functions_skips_prompt_enhancement_after_current_turn_result(
    test_copilot_service,
):
    test_copilot_service._prompt_enhancement_service = Mock()

    functions = test_copilot_service._build_llm_functions(
        messages=[
            UserMessage("summarize this data"),
            AssistantMessage(
                "Enhanced query: summarize the current dataset by key metrics\n\n"
                "I will now proceed with this clarified interpretation."
            ),
        ],
        documents=None,
        tools=None,
        widget_collection=None,
        sql_widgets=None,
        python_widgets=None,
    )

    function_names = [func.__name__ for func in functions if hasattr(func, "__name__")]
    assert "llm_enhance_prompt" not in function_names


def test_build_llm_functions_skips_planning_after_tool_result(
    test_copilot_service,
):
    test_copilot_service._context_service.structured_context = None
    test_copilot_service._context_service.unstructured_context = None
    think_call = FunctionCall(
        test_copilot_service._native_function_call_service._llm_think,
        plan="Query the file, then create a table.",
        summary="Planning",
    )

    functions = test_copilot_service._build_llm_functions(
        messages=[
            UserMessage("Create a table with insights from this PDF"),
            AssistantMessage(think_call),
            FunctionResultMessage(
                content="Query the file, then create a table.",
                function_call=think_call,
            ),
        ],
        documents=None,
        tools=None,
        widget_collection=None,
        sql_widgets=None,
        python_widgets=None,
    )

    function_names = [func.__name__ for func in functions if hasattr(func, "__name__")]
    assert "_llm_think" not in function_names
    assert "llm_complete" in function_names


@pytest.mark.asyncio
async def test_build_llm_functions_uses_preserved_app_widget_collection(
    test_copilot_service,
):
    test_copilot_service._context_service.structured_context = None
    test_copilot_service._context_service.unstructured_context = None

    note_widget = Widget(
        uuid=UUID("11111111-1111-1111-1111-111111111111"),
        origin="Research Backend",
        widget_id="research/notes",
        name="Research Notes",
        description="Documentation notes",
        params=[],
        metadata={},
    )
    data_widget = Widget(
        uuid=UUID("22222222-2222-2222-2222-222222222222"),
        origin="Research Backend",
        widget_id="research_dataset",
        name="Research Dataset",
        description="Research data widget",
        params=[],
        metadata={"schema": {"tableName": "RESEARCH_DATASET"}},
    )

    functions = test_copilot_service._build_llm_functions(
        messages=[UserMessage("Create a research dashboard")],
        documents=None,
        tools=None,
        widget_collection=WidgetCollection(
            primary=[],
            secondary=[],
            extra=[note_widget],
        ),
        sql_widgets=None,
        python_widgets=None,
        app_widget_collection=WidgetCollection(
            primary=[],
            secondary=[],
            extra=[note_widget, data_widget],
        ),
    )

    function_names = [func.__name__ for func in functions if hasattr(func, "__name__")]
    assert "llm_create_app" in function_names

    search_widgets = next(
        func for func in functions if func.__name__ == "llm_search_widgets"
    )
    events = []
    async for event in search_widgets(query="research"):
        events.append(event)

    assert isinstance(events[0], StatusUpdateSSE)
    payload = json.loads(events[-1])
    assert {match["widget_id"] for match in payload["matches"]} == {
        "research/notes",
        "research_dataset",
    }


def test_build_llm_functions_keeps_planning_after_prompt_enhancement(
    test_copilot_service,
):
    test_copilot_service._context_service.structured_context = None
    test_copilot_service._context_service.unstructured_context = None
    enhance_call = FunctionCall(
        test_copilot_service._native_function_call_service.llm_enhance_prompt,
        reasoning="The request is broad and needs a clearer objective.",
        summary="Clarifying request",
    )

    functions = test_copilot_service._build_llm_functions(
        messages=[
            UserMessage("analyze this"),
            AssistantMessage(enhance_call),
            FunctionResultMessage(
                content=(
                    "Enhanced query: summarize the uploaded file and identify "
                    "the main policy conclusions\n\n"
                    "I will now proceed with this clarified interpretation."
                ),
                function_call=enhance_call,
            ),
        ],
        documents=None,
        tools=None,
        widget_collection=None,
        sql_widgets=None,
        python_widgets=None,
    )

    function_names = [func.__name__ for func in functions if hasattr(func, "__name__")]
    assert "llm_enhance_prompt" not in function_names
    assert "_llm_think" in function_names
    assert "llm_complete" in function_names


def _make_agent_tool(name: str) -> AgentTool:
    return AgentTool(
        name=name,
        server_id="server-1",
        url="",
        endpoint=None,
        description="Test MCP tool",
        input_schema={"type": "object", "properties": {}, "required": []},
        auth_token=None,
    )


def test_llm_query_extra_widgets_extra_state_error(test_copilot_service, monkeypatch):
    """Test that _llm_query_extra_widgets handles extra_state parameter correctly.

    Verifies TypeError is raised when extra_state is passed multiple times.
    """
    # Setup
    mock_data_service = Mock()
    monkeypatch.setattr(
        test_copilot_service, "_copilot_data_service", mock_data_service
    )

    search_queries = [{"description": "test", "query": "test query"}]
    extra_state = {"test": "value"}

    # Create duplicate parameter scenario
    with pytest.raises(TypeError) as exc_info:
        # Pass extra_state both as parameter and in kwargs
        test_copilot_service._client_function_call_service.llm_query_extra_widgets(
            search_queries=search_queries,
            extra_state=extra_state,
            **{"extra_state": extra_state},  # Duplicate parameter
        )

    # Verify the expected error matches production error
    assert "got multiple values for keyword argument 'extra_state'" in str(
        exc_info.value
    )
    mock_data_service.query_extra_widgets.assert_not_called()


def test_llm_query_extra_widgets_parameter_order_regression(test_copilot_service):
    """Test parameter order doesn't leak summary into extra_state.

    Regression test for bug where incorrect parameter order caused the summary
    parameter to leak into the extra_state parameter when using wrapped_partial.

    The bug occurred when:
    1. Function is registered with wrapped_partial(func, extra_state=None)
    2. LLM calls with positional args that misalign parameters due to wrong order
    3. Summary parameter gets passed as extra_state instead of summary

    This test verifies that the parameter order is correct and prevents the bug.
    """
    import inspect
    from functools import partial

    from openbb_ada.models import DataSourceSearchQuery

    # Test data that simulates LLM function call
    search_queries = [
        DataSourceSearchQuery(description="test desc", query="test query")
    ]
    summary_text = "Test summary from LLM"

    # Get the function signature to verify parameter order
    sig = inspect.signature(
        test_copilot_service._client_function_call_service.llm_query_extra_widgets
    )
    param_names = list(sig.parameters.keys())

    # Verify the parameter order is correct (this is the fix)
    # The order should be:
    # 1. search_queries,
    # 2. summary,
    # 3. extra_state (self is implicit for bound method)
    expected_order = ["search_queries", "summary", "extra_state"]
    assert param_names == expected_order, (
        f"Parameter order regression: expected {expected_order}, got {param_names}"
    )

    # Test that wrapped_partial works correctly with the fixed parameter order
    # This simulates how the function is registered in _get_llm_functions
    partial_func = partial(
        test_copilot_service._client_function_call_service.llm_query_extra_widgets,
        extra_state=None,
    )

    # This should work without raising TypeError about multiple values
    # The key test: calling with positional args should map correctly
    try:
        # This call should work because:
        # - partial_func has extra_state=None bound
        # - search_queries goes to position 0 (after self)
        # - summary_text goes to position 1 (summary parameter)
        # - extra_state is already bound to None
        result = partial_func(search_queries, summary_text)
        assert result is not None

    except TypeError as e:
        if "got multiple values for argument" in str(e):
            pytest.fail(f"Parameter order regression detected: {e}")
        raise

    # Additional verification: test that wrong parameter order would cause the bug
    # Create a mock function with wrong parameter order to verify our fix
    def wrong_order_func(search_queries, extra_state=None, summary="default"):
        return f"called with {search_queries}, {extra_state}, {summary}"

    wrong_partial = partial(wrong_order_func, extra_state=None)

    # This would cause the bug: summary_text would be passed as extra_state
    with pytest.raises(TypeError) as exc_info:
        wrong_partial(search_queries, summary_text)  # type: ignore

    # Verify this is the exact error we fixed
    assert "got multiple values for argument 'extra_state'" in str(exc_info.value)


def test_get_sql_enabled_widgets_with_include_extra_false(
    test_copilot_service, mock_uuids
):
    """When include_extra=False, only primary+secondary widgets returned."""
    primary_widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        widget_id="primary_sql",
        name="Primary SQL Widget",
        description="Primary SQL widget",
        origin="test",
        metadata={"schema": {"tableName": "primary_table"}},
        params=[],
    )

    secondary_widget = Widget(
        uuid=UUID(mock_uuids.ID2.value),
        widget_id="secondary_sql",
        name="Secondary SQL Widget",
        description="Secondary SQL widget",
        origin="test",
        metadata={"schema": {"tableName": "secondary_table"}},
        params=[],
    )

    extra_widget = Widget(
        uuid=UUID(mock_uuids.ID3.value),
        widget_id="extra_sql",
        name="Extra SQL Widget",
        description="Extra SQL widget",
        origin="test",
        metadata={"schema": {"tableName": "extra_table"}},
        params=[],
    )

    collection = WidgetCollection(
        primary=[primary_widget],
        secondary=[secondary_widget],
        extra=[extra_widget],
    )

    result = test_copilot_service._get_sql_enabled_widgets(
        collection, include_extra=False
    )

    # Should only have 2 widgets (primary + secondary)
    assert len(result) == 2
    widget_ids = [w.widget_id for w in result]
    assert "primary_sql" in widget_ids
    assert "secondary_sql" in widget_ids
    assert "extra_sql" not in widget_ids  # Critical assertion


def test_get_sql_enabled_widgets_with_include_extra_true(
    test_copilot_service, mock_uuids
):
    """When include_extra=True, all widgets (primary+secondary+extra) returned."""
    primary_widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        widget_id="primary_sql",
        name="Primary SQL Widget",
        description="Primary SQL widget",
        origin="test",
        metadata={"schema": {"tableName": "primary_table"}},
        params=[],
    )

    secondary_widget = Widget(
        uuid=UUID(mock_uuids.ID2.value),
        widget_id="secondary_sql",
        name="Secondary SQL Widget",
        description="Secondary SQL widget",
        origin="test",
        metadata={"schema": {"tableName": "secondary_table"}},
        params=[],
    )

    extra_widget = Widget(
        uuid=UUID(mock_uuids.ID3.value),
        widget_id="extra_sql",
        name="Extra SQL Widget",
        description="Extra SQL widget",
        origin="test",
        metadata={"schema": {"tableName": "extra_table"}},
        params=[],
    )

    collection = WidgetCollection(
        primary=[primary_widget],
        secondary=[secondary_widget],
        extra=[extra_widget],
    )

    result = test_copilot_service._get_sql_enabled_widgets(
        collection, include_extra=True
    )

    # Should have all 3 widgets
    assert len(result) == 3
    widget_ids = [w.widget_id for w in result]
    assert "primary_sql" in widget_ids
    assert "secondary_sql" in widget_ids
    assert "extra_sql" in widget_ids  # Critical assertion


def test_get_sql_enabled_widgets_priority_ordering(test_copilot_service, mock_uuids):
    """Widgets should be returned in priority order: primary, secondary, extra."""
    primary_widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        widget_id="primary_sql",
        name="Primary SQL Widget",
        description="Primary SQL widget",
        origin="test",
        metadata={"schema": {"tableName": "primary_table"}},
        params=[],
    )

    secondary_widget = Widget(
        uuid=UUID(mock_uuids.ID2.value),
        widget_id="secondary_sql",
        name="Secondary SQL Widget",
        description="Secondary SQL widget",
        origin="test",
        metadata={"schema": {"tableName": "secondary_table"}},
        params=[],
    )

    extra_widget = Widget(
        uuid=UUID(mock_uuids.ID3.value),
        widget_id="extra_sql",
        name="Extra SQL Widget",
        description="Extra SQL widget",
        origin="test",
        metadata={"schema": {"tableName": "extra_table"}},
        params=[],
    )

    collection = WidgetCollection(
        primary=[primary_widget],
        secondary=[secondary_widget],
        extra=[extra_widget],
    )

    result = test_copilot_service._get_sql_enabled_widgets(
        collection, include_extra=True
    )

    # Verify order
    assert result[0].widget_name == "Primary SQL Widget"
    assert result[1].widget_name == "Secondary SQL Widget"
    assert result[2].widget_name == "Extra SQL Widget"


# --- Python Widget Tests ---


def _create_python_param(name: str = "prompt", current_value: str | None = None):
    """Create a WidgetParam with language="python"."""
    return WidgetParam(
        name=name,
        type="text",
        description="Python code",
        language="python",
        current_value=current_value,
        default_value="# Default code",
    )


def _create_non_python_param(name: str = "ticker"):
    """Create a WidgetParam without language="python"."""
    return WidgetParam(
        name=name,
        type="string",
        description="Ticker symbol",
        current_value="AAPL",
        default_value="MSFT",
    )


def test_get_code_enabled_widgets_with_include_extra_false(
    test_copilot_service, mock_uuids
):
    """When include_extra=False, only primary+secondary Python widgets returned."""
    primary_widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        widget_id="primary_python",
        name="Primary Python Widget",
        description="Primary Python widget",
        origin="test",
        metadata={},
        params=[_create_python_param()],
    )

    secondary_widget = Widget(
        uuid=UUID(mock_uuids.ID2.value),
        widget_id="secondary_python",
        name="Secondary Python Widget",
        description="Secondary Python widget",
        origin="test",
        metadata={},
        params=[_create_python_param()],
    )

    extra_widget = Widget(
        uuid=UUID(mock_uuids.ID3.value),
        widget_id="extra_python",
        name="Extra Python Widget",
        description="Extra Python widget",
        origin="test",
        metadata={},
        params=[_create_python_param()],
    )

    collection = WidgetCollection(
        primary=[primary_widget],
        secondary=[secondary_widget],
        extra=[extra_widget],
    )

    result = test_copilot_service._get_code_enabled_widgets(
        collection, include_extra=False
    )

    # Should only have 2 widgets (primary + secondary)
    assert len(result) == 2
    widget_ids = [w.widget_id for w in result]
    assert "primary_python" in widget_ids
    assert "secondary_python" in widget_ids
    assert "extra_python" not in widget_ids  # Critical assertion


def test_get_code_enabled_widgets_with_include_extra_true(
    test_copilot_service, mock_uuids
):
    """When include_extra=True, all Python widgets returned."""
    primary_widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        widget_id="primary_python",
        name="Primary Python Widget",
        description="Primary Python widget",
        origin="test",
        metadata={},
        params=[_create_python_param()],
    )

    secondary_widget = Widget(
        uuid=UUID(mock_uuids.ID2.value),
        widget_id="secondary_python",
        name="Secondary Python Widget",
        description="Secondary Python widget",
        origin="test",
        metadata={},
        params=[_create_python_param()],
    )

    extra_widget = Widget(
        uuid=UUID(mock_uuids.ID3.value),
        widget_id="extra_python",
        name="Extra Python Widget",
        description="Extra Python widget",
        origin="test",
        metadata={},
        params=[_create_python_param()],
    )

    collection = WidgetCollection(
        primary=[primary_widget],
        secondary=[secondary_widget],
        extra=[extra_widget],
    )

    result = test_copilot_service._get_code_enabled_widgets(
        collection, include_extra=True
    )

    # Should have all 3 widgets
    assert len(result) == 3
    widget_ids = [w.widget_id for w in result]
    assert "primary_python" in widget_ids
    assert "secondary_python" in widget_ids
    assert "extra_python" in widget_ids  # Critical assertion


def test_get_code_enabled_widgets_priority_ordering(test_copilot_service, mock_uuids):
    """Python widgets returned in priority order: primary, secondary, extra."""
    primary_widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        widget_id="primary_python",
        name="Primary Python Widget",
        description="Primary Python widget",
        origin="test",
        metadata={},
        params=[_create_python_param()],
    )

    secondary_widget = Widget(
        uuid=UUID(mock_uuids.ID2.value),
        widget_id="secondary_python",
        name="Secondary Python Widget",
        description="Secondary Python widget",
        origin="test",
        metadata={},
        params=[_create_python_param()],
    )

    extra_widget = Widget(
        uuid=UUID(mock_uuids.ID3.value),
        widget_id="extra_python",
        name="Extra Python Widget",
        description="Extra Python widget",
        origin="test",
        metadata={},
        params=[_create_python_param()],
    )

    collection = WidgetCollection(
        primary=[primary_widget],
        secondary=[secondary_widget],
        extra=[extra_widget],
    )

    result = test_copilot_service._get_code_enabled_widgets(
        collection, include_extra=True
    )

    # Verify order
    assert result[0].widget_name == "Primary Python Widget"
    assert result[1].widget_name == "Secondary Python Widget"
    assert result[2].widget_name == "Extra Python Widget"


def test_get_code_enabled_widgets_detects_language_python(
    test_copilot_service, mock_uuids
):
    """Only widgets with params having language='python' should be detected."""
    python_widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        widget_id="python_widget",
        name="Python Widget",
        description="Widget with Python param",
        origin="test",
        metadata={},
        params=[_create_python_param()],
    )

    non_python_widget = Widget(
        uuid=UUID(mock_uuids.ID2.value),
        widget_id="regular_widget",
        name="Regular Widget",
        description="Widget without Python param",
        origin="test",
        metadata={},
        params=[_create_non_python_param()],
    )

    mixed_widget = Widget(
        uuid=UUID(mock_uuids.ID3.value),
        widget_id="mixed_widget",
        name="Mixed Widget",
        description="Widget with both Python and non-Python params",
        origin="test",
        metadata={},
        params=[_create_python_param(), _create_non_python_param()],
    )

    collection = WidgetCollection(
        primary=[python_widget, non_python_widget, mixed_widget],
        secondary=[],
        extra=[],
    )

    result = test_copilot_service._get_code_enabled_widgets(
        collection, include_extra=False
    )

    # Should only detect python_widget and mixed_widget (both have Python params)
    assert len(result) == 2
    widget_ids = [w.widget_id for w in result]
    assert "python_widget" in widget_ids
    assert "mixed_widget" in widget_ids
    assert "regular_widget" not in widget_ids  # No Python param


def test_get_code_enabled_widgets_extracts_current_code(
    test_copilot_service, mock_uuids
):
    """Current code should be extracted from the prompt param."""
    current_code = "result = session.sql('SELECT * FROM table').to_pandas()"
    python_widget = Widget(
        uuid=UUID(mock_uuids.ID1.value),
        widget_id="python_widget",
        name="Python Widget",
        description="Widget with current code",
        origin="test",
        metadata={},
        params=[_create_python_param(name="prompt", current_value=current_code)],
    )

    collection = WidgetCollection(
        primary=[python_widget],
        secondary=[],
        extra=[],
    )

    result = test_copilot_service._get_code_enabled_widgets(
        collection, include_extra=False
    )

    assert len(result) == 1
    assert result[0].current_code == current_code


def test_handle_function_call_result_error_includes_latest_query_input(
    test_copilot_service,
):
    client_error = Mock()
    client_error.error_type = "unexpected_error"
    client_error.content = "Error executing the function call."

    data_source_request = Mock()
    data_source_request.widget_uuid = "3b4ef8fe-d508-45d8-af85-98480e4197a0"
    data_source_request.origin = "SNOW backend"
    data_source_request.id = (
        "snowflake_public_data_paid_public_data_sec_corporate_report_attributes"
    )
    data_source_request.input_args = {
        "query": "SELECT * FROM MY_TABLE LIMIT 10",
    }

    result = test_copilot_service._handle_function_call_result_error(
        client_function_call_error=client_error,
        data_source_request=data_source_request,
    )

    assert "latest_input_text: SELECT * FROM MY_TABLE LIMIT 10" in result
    assert "current state sent by the UI" in result
    assert "input_args={'query': 'SELECT * FROM MY_TABLE LIMIT 10'}" in result


@pytest.mark.asyncio
async def test_widget_data_citation_details_include_input_args(
    test_copilot_service,
):
    widget_uuid = "3c0a1a9c-5324-42b0-98b5-5e8d307b7d49"
    data_source = Mock()
    data_source.origin = "Portfolio Risk"
    data_source.name = "Portfolio Exposure by Country"
    data_source.id = "portfolio_countries_custom_obb"
    data_source.description = "Get portfolio's exposure by country."
    data_source.widget = Mock()
    data_source.widget.metadata = {}

    test_copilot_service._citation_service = CitationService()
    test_copilot_service._context_service = Mock()
    test_copilot_service._document_service.get_unavailable_documents_by_data_source = (
        Mock(return_value=[])
    )
    test_copilot_service._copilot_data_service.get_data_source_from_map_or_db = Mock(
        return_value=data_source
    )

    message = LlmClientFunctionCallResultMessage(
        function="get_widget_data",
        input_arguments={
            "data_sources": [
                {
                    "widget_uuid": widget_uuid,
                    "origin": "Portfolio Risk",
                    "id": "portfolio_countries_custom_obb",
                    "input_args": {"portfolio": "Client 2"},
                }
            ]
        },
        data=[
            DataContent(
                items=[
                    SingleDataContent(
                        content='[{"Country":"United States","Weight":67.61}]',
                        citable=True,
                    )
                ]
            )
        ],
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=FunctionCall(_dummy_function),
        is_last_message=True,
    ):
        events.append(event)

    citations = test_copilot_service._citation_service.citations
    assert len(citations) == 1
    assert citations[0].details == [
        {
            "Origin": "Portfolio Risk",
            "Data source": "Portfolio Exposure by Country",
            "Portfolio": "Client 2",
        }
    ]


def test_get_previously_queried_file_widget_uuids_scopes_to_current_human_turn(
    test_copilot_service,
):
    old_widget_uuid = UUID("11111111-1111-1111-1111-111111111111")
    current_widget_uuid = UUID("22222222-2222-2222-2222-222222222222")

    messages = [
        LlmClientMessage(role=RoleEnum.human, content="first question"),
        LlmClientFunctionCallResultMessage(
            function="get_widget_data",
            input_arguments={
                "data_sources": [
                    {
                        "widget_uuid": str(old_widget_uuid),
                        "origin": "OpenBB Hub",
                        "id": f"file-{old_widget_uuid}",
                        "input_args": {},
                    }
                ]
            },
            data=[],
        ),
        LlmClientMessage(role=RoleEnum.human, content="second question"),
        LlmClientFunctionCallResultMessage(
            function="get_widget_data",
            input_arguments={
                "data_sources": [
                    {
                        "widget_uuid": str(current_widget_uuid),
                        "origin": "OpenBB Hub",
                        "id": f"file-{current_widget_uuid}",
                        "input_args": {},
                    }
                ]
            },
            data=[],
        ),
    ]

    result = test_copilot_service._get_previously_queried_file_widget_uuids(messages)

    assert result == {current_widget_uuid}


def test_get_repeated_file_queries_only_blocks_file_widgets(
    test_copilot_service,
):
    file_widget_uuid = UUID("33333333-3333-3333-3333-333333333333")
    non_file_widget_uuid = UUID("44444444-4444-4444-4444-444444444444")

    def _get_data_source(widget_uuid: UUID):
        if widget_uuid == file_widget_uuid:
            return Mock(id=f"file-{widget_uuid}")
        if widget_uuid == non_file_widget_uuid:
            return Mock(id="news-widget")
        return None

    test_copilot_service._copilot_data_service.get_data_source_from_map_or_db = Mock(
        side_effect=_get_data_source
    )
    function_call = FunctionCall(
        test_copilot_service._client_function_call_service.llm_query_widgets,
        summary="Querying widgets",
        widget_queries=[
            {
                "widget_uuid": str(file_widget_uuid),
                "query": "Summarize the file widget.",
                "use_current_inputs": True,
            },
            {
                "widget_uuid": str(non_file_widget_uuid),
                "query": "Summarize the news widget.",
                "use_current_inputs": True,
            },
        ],
    )

    repeated_queries = test_copilot_service._get_repeated_file_queries(
        function_call=function_call,
        previously_queried_file_widget_uuids={
            file_widget_uuid,
            non_file_widget_uuid,
        },
    )

    assert len(repeated_queries) == 1
    assert repeated_queries[0].widget_uuid == file_widget_uuid


def test_build_repeated_file_widget_retry_messages_guides_llm_to_file_query_tool(
    test_copilot_service,
):
    from openbb_ada.models import WidgetQueryRequest

    file_widget_uuid = UUID("55555555-5555-5555-5555-555555555555")
    function_call = FunctionCall(
        test_copilot_service._client_function_call_service.llm_query_widgets,
        summary="Querying widgets",
        widget_queries=[
            {
                "widget_uuid": str(file_widget_uuid),
                "query": "Summarize the file widget.",
                "use_current_inputs": True,
            }
        ],
    )

    messages = test_copilot_service._build_repeated_file_widget_retry_messages(
        function_call=function_call,
        repeated_widget_queries=[
            WidgetQueryRequest(
                widget_uuid=file_widget_uuid,
                query="Summarize the file widget.",
                use_current_inputs=True,
            )
        ],
    )

    assert len(messages) == 1
    assert isinstance(messages[0], AssistantMessage)
    assert (
        "The following tool call was blocked and was not executed"
        in messages[0].content
    )
    assert "Do NOT call `llm_query_widgets` again" in messages[0].content
    assert "`_llm_query_uploaded_files` next" in messages[0].content


def _make_mcp_tool_result_message(
    tool_name: str = "qtap duckdb mcp_list_tables",
    server_id: str = "1771593417468",
    content: str = '{"success": true, "tables": [{"name": "dim_funds"}]}',
) -> LlmClientFunctionCallResultMessage:
    """Create a minimal MCP tool result message for CopilotService unit tests."""
    return LlmClientFunctionCallResultMessage(
        function="execute_agent_tool",
        input_arguments={
            "server_id": server_id,
            "tool_name": tool_name,
            "parameters": {},
        },
        data=[
            DataContent(
                items=[SingleDataContent(content=content)],
            )
        ],
        extra_state={
            "copilot_function_call_arguments": {
                "server_id": server_id,
                "tool_name": tool_name,
                "tool_args": {},
                "summary": "Listing tables",
            }
        },
    )


def _dummy_function(**kwargs):
    """Dummy callable used to instantiate a magentic FunctionCall in tests."""
    return kwargs


async def _empty_async_gen():
    """Return no MCP events while still being an async generator."""
    if False:  # pragma: no cover
        yield None


@pytest.mark.asyncio
async def test_handle_mcp_result_includes_continuation_instruction_when_last_message(
    test_copilot_service,
):
    """Interpretation: last MCP results should steer the model to keep tool-calling.

    This protects against the regression where the model narrates "next steps"
    and stops after an MCP result instead of issuing the next tool call.
    """
    message = _make_mcp_tool_result_message()
    function_call = FunctionCall(_dummy_function)

    mcp_processor = Mock(side_effect=lambda **kwargs: _empty_async_gen())
    test_copilot_service._mcp_data_service.process_mcp_tool_response = mcp_processor
    test_copilot_service._mcp_data_service._is_mcp_error = Mock(return_value=True)
    test_copilot_service._mcp_data_service._get_mcp_error_content = Mock(
        return_value="missing required parameter: symbol"
    )
    test_copilot_service._mcp_data_service._is_mcp_error = Mock(return_value=True)
    test_copilot_service._mcp_data_service._get_mcp_error_content = Mock(
        return_value="missing required parameter: symbol"
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=function_call,
        is_last_message=True,
    ):
        events.append(event)

    result_messages = [event for event in events if isinstance(event, list)]
    assert len(result_messages) == 1
    assert len(result_messages[0]) == 1
    function_result_message = result_messages[0][0]
    assert isinstance(function_result_message, FunctionResultMessage)

    content = function_result_message.content
    assert "Result from qtap duckdb mcp_list_tables" in content
    assert "dim_funds" in content
    assert "Continue executing your plan" in content
    assert "next relevant tool now" in content
    assert "`_llm_complete`" in content

    mcp_processor.assert_called_once()


@pytest.mark.asyncio
async def test_handle_mcp_result_no_continuation_instruction_when_not_last_message(
    test_copilot_service,
):
    """Interpretation: replayed MCP history should not inject execution directives."""
    message = _make_mcp_tool_result_message()
    function_call = FunctionCall(_dummy_function)

    mcp_processor = Mock(side_effect=lambda **kwargs: _empty_async_gen())
    test_copilot_service._mcp_data_service.process_mcp_tool_response = mcp_processor
    test_copilot_service._mcp_data_service._is_mcp_error = Mock(return_value=True)
    test_copilot_service._mcp_data_service._get_mcp_error_content = Mock(
        return_value=None
    )
    test_copilot_service._mcp_data_service._is_mcp_error = Mock(return_value=True)
    test_copilot_service._mcp_data_service._get_mcp_error_content = Mock(
        return_value=None
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=function_call,
        is_last_message=False,
    ):
        events.append(event)

    result_messages = [event for event in events if isinstance(event, list)]
    assert len(result_messages) == 1
    assert len(result_messages[0]) == 1
    function_result_message = result_messages[0][0]
    assert isinstance(function_result_message, FunctionResultMessage)

    content = function_result_message.content
    assert "Result from qtap duckdb mcp_list_tables" in content
    assert "dim_funds" in content
    assert "Continue executing your plan" not in content
    assert "next relevant tool now" not in content
    assert "`_llm_complete`" not in content

    mcp_processor.assert_not_called()


@pytest.mark.asyncio
async def test_handle_mcp_result_error_includes_self_correction_instruction(
    test_copilot_service,
):
    """MCP errors should ask the model to retry once with corrected arguments."""
    mcp_error_content = json.dumps(
        {
            "isError": True,
            "content": [
                {"type": "text", "text": "missing required parameter: symbol"},
            ],
        }
    )
    message = _make_mcp_tool_result_message_with_content(mcp_error_content)
    function_call = FunctionCall(_dummy_function)

    mcp_processor = Mock(side_effect=lambda **kwargs: _empty_async_gen())
    test_copilot_service._mcp_data_service.process_mcp_tool_response = mcp_processor
    test_copilot_service._mcp_data_service._is_mcp_error = Mock(return_value=True)
    test_copilot_service._mcp_data_service._get_mcp_error_content = Mock(
        return_value="missing required parameter: symbol"
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=function_call,
        is_last_message=True,
    ):
        events.append(event)

    result_messages = [event for event in events if isinstance(event, list)]
    assert len(result_messages) == 1
    assert len(result_messages[0]) == 1
    function_result_message = result_messages[0][0]
    assert isinstance(function_result_message, FunctionResultMessage)

    content = function_result_message.content
    assert "MCP tool call failed." in content
    assert "missing required parameter: symbol" in content
    assert "unexpected runtime/service error" in content
    assert "do not call fallback tools" in content
    assert "suggested fix, next step, or URL" in content
    assert "include that solution explicitly" in content
    assert "ask whether the user wants you to retry" in content
    assert "retry the tool exactly once with corrected arguments" in content
    assert "Do not reuse the same arguments." in content

    mcp_processor.assert_called_once()


@pytest.mark.asyncio
async def test_handle_mcp_result_unexpected_error_requires_confirmation(
    test_copilot_service,
):
    """Unexpected MCP errors should be surfaced before retry or web fallback."""
    error_text = (
        'Error executing MCP tool "SEC MCP_list_filings": '
        "MCP error -32000: SEC MCP tool call quota reached for this "
        "billing period (100 calls/month). Go to https://example.com/usage "
        "to upgrade your plan and continue using SEC MCP."
    )
    mcp_error_content = json.dumps(
        {
            "error_type": "unexpected_error",
            "content": error_text,
        }
    )
    message = _make_mcp_tool_result_message_with_content(mcp_error_content)
    function_call = FunctionCall(_dummy_function)

    mcp_processor = Mock(side_effect=lambda **kwargs: _empty_async_gen())
    test_copilot_service._mcp_data_service.process_mcp_tool_response = mcp_processor
    test_copilot_service._mcp_data_service._is_mcp_error = Mock(return_value=True)
    test_copilot_service._mcp_data_service._get_mcp_error_content = Mock(
        return_value=error_text
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=function_call,
        is_last_message=True,
    ):
        events.append(event)

    result_messages = [event for event in events if isinstance(event, list)]
    function_result_message = result_messages[0][0]

    content = function_result_message.content
    assert "MCP tool call failed." in content
    assert "quota reached" in content
    assert "https://example.com/usage" in content
    assert "do not retry immediately" in content
    assert "do not call fallback tools" in content
    assert "suggested fix, next step, or URL" in content
    assert "include that solution explicitly" in content
    assert "ask whether the user wants you to retry" in content
    assert "After asking, call `_llm_complete`" in content

    mcp_processor.assert_called_once()


@pytest.mark.asyncio
async def test_handle_mcp_result_error_without_details_avoids_blind_retry(
    test_copilot_service,
):
    """MCP errors without details should not ask for guessed retries."""
    mcp_error_content = json.dumps({"isError": True})
    message = _make_mcp_tool_result_message_with_content(mcp_error_content)
    function_call = FunctionCall(_dummy_function)

    mcp_processor = Mock(side_effect=lambda **kwargs: _empty_async_gen())
    test_copilot_service._mcp_data_service.process_mcp_tool_response = mcp_processor
    test_copilot_service._mcp_data_service._is_mcp_error = Mock(return_value=True)
    test_copilot_service._mcp_data_service._get_mcp_error_content = Mock(
        return_value=None
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=function_call,
        is_last_message=True,
    ):
        events.append(event)

    result_messages = [event for event in events if isinstance(event, list)]
    assert len(result_messages) == 1
    assert len(result_messages[0]) == 1
    function_result_message = result_messages[0][0]
    assert isinstance(function_result_message, FunctionResultMessage)

    content = function_result_message.content
    assert "MCP tool call failed." in content
    assert "Error: Unknown MCP error" in content
    assert "No actionable error details were returned." in content
    assert "Do not blindly retry with guessed arguments." in content
    assert "Do not call fallback tools" in content
    assert "whether they want you to retry" in content
    assert "retry the tool exactly once with corrected arguments" not in content

    mcp_processor.assert_called_once()


def _make_mcp_tool_result_message_with_content(
    content: str,
    tool_name: str = "qtap duckdb mcp_list_tables",
    server_id: str = "1771593417468",
) -> LlmClientFunctionCallResultMessage:
    """Create an MCP tool result message with arbitrary raw content."""
    return LlmClientFunctionCallResultMessage(
        function="execute_agent_tool",
        input_arguments={
            "server_id": server_id,
            "tool_name": tool_name,
            "parameters": {},
        },
        data=[
            DataContent(
                items=[SingleDataContent(content=content)],
            )
        ],
        extra_state={
            "copilot_function_call_arguments": {
                "server_id": server_id,
                "tool_name": tool_name,
                "tool_args": {},
                "summary": "Listing tables",
            }
        },
    )


@pytest.mark.asyncio
async def test_handle_mcp_result_plain_text_content_does_not_raise(
    test_copilot_service,
):
    """MCP tool results that return plain text (not JSON) must not raise NameError.

    Regression test for the bug where the `except (json.JSONDecodeError, TypeError)`
    block in `_handle_function_call_result_message` references `parsed_content` — a
    variable that is only assigned inside the preceding `try` block. When
    `json.loads(item.content)` fails on the FIRST item processed, `parsed_content`
    is undefined, causing an unhandled NameError that propagates up and causes the
    MCP call to appear stuck.

    Real-world trigger: DuckDB MCP tools like `mcp_list_tables` can return plain
    text or non-JSON formatted responses (e.g. "table1\\ntable2\\ntable3").
    """
    # Plain text content — json.loads will raise JSONDecodeError on the first item.
    # Before the fix this triggers: NameError: name 'parsed_content' is not defined
    plain_text_content = "table1\ntable2\ntable3"
    message = _make_mcp_tool_result_message_with_content(plain_text_content)
    function_call = FunctionCall(_dummy_function)

    mcp_processor = Mock(side_effect=lambda **kwargs: _empty_async_gen())
    test_copilot_service._mcp_data_service.process_mcp_tool_response = mcp_processor

    # This should NOT raise — before the fix it raises NameError
    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=function_call,
        is_last_message=True,
    ):
        events.append(event)

    # The plain-text content must appear in the result message passed to the LLM
    result_messages = [e for e in events if isinstance(e, list)]
    assert len(result_messages) == 1
    function_result = result_messages[0][0]
    assert isinstance(function_result, FunctionResultMessage)
    assert plain_text_content in function_result.content


@pytest.mark.asyncio
async def test_handle_mcp_result_stale_parsed_content_does_not_corrupt_second_item(
    test_copilot_service,
):
    """Two-item MCP response where second item is non-JSON must not use stale data.

    Another variant of the parsed_content stale-variable bug: if the FIRST item
    parses successfully (setting parsed_content = <valid data>) and the SECOND item
    fails json.loads, the except block would reference the first item's parsed_content
    and silently return wrong data to the MCP artifact pipeline.
    """
    from openbb_ai.models import DataContent, SingleDataContent

    good_json = '{"tables": ["dim_funds", "fact_returns"]}'
    bad_text = "not valid json at all"
    message = LlmClientFunctionCallResultMessage(
        function="execute_agent_tool",
        input_arguments={
            "server_id": "1771593417468",
            "tool_name": "qtap duckdb mcp_mixed",
            "parameters": {},
        },
        data=[
            DataContent(
                items=[
                    SingleDataContent(content=good_json),
                    SingleDataContent(content=bad_text),
                ],
            )
        ],
        extra_state={
            "copilot_function_call_arguments": {
                "server_id": "1771593417468",
                "tool_name": "qtap duckdb mcp_mixed",
                "tool_args": {},
                "summary": "Mixed result",
            }
        },
    )
    function_call = FunctionCall(_dummy_function)

    mcp_processor = Mock(side_effect=lambda **kwargs: _empty_async_gen())
    test_copilot_service._mcp_data_service.process_mcp_tool_response = mcp_processor

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=function_call,
        is_last_message=True,
    ):
        events.append(event)

    result_messages = [e for e in events if isinstance(e, list)]
    assert len(result_messages) == 1
    function_result = result_messages[0][0]
    assert isinstance(function_result, FunctionResultMessage)
    # Both items' content must be present in the LLM message
    assert good_json in function_result.content
    assert bad_text in function_result.content
    # Artifact pipeline must use the last item's parsed content.
    mcp_processor.assert_called_once()
    call_kwargs = mcp_processor.call_args.kwargs
    assert call_kwargs["raw_content"] == bad_text


# --- normalize_jsonish unit tests ---


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ('{"success": true, "tables": ["t1"]}', {"success": True, "tables": ["t1"]}),
        (
            json.dumps(['{"success": true, "tables": ["dim_funds"]}']),
            {"success": True, "tables": ["dim_funds"]},
        ),
        ('[{"success": true}]', {"success": True}),
        ('["not json at all"]', "not json at all"),
        ("table1\ntable2\ntable3", "table1\ntable2\ntable3"),
        ("[1, 2, 3]", [1, 2, 3]),
        ('[[["hello"]]]', "hello"),
        (json.dumps('{"key": "value"}'), {"key": "value"}),
        ("[]", []),
        ({"already": "parsed"}, {"already": "parsed"}),
        (42, 42),
        (None, None),
    ],
)
def test_normalize_jsonish(raw_value, expected):
    """Unit tests for the normalize_jsonish helper used in MCP response parsing."""
    from openbb_ada.services.mcp_data import normalize_jsonish

    assert normalize_jsonish(raw_value) == expected


# --- Integration tests for MCP parsing with normalize_jsonish ---


@pytest.mark.asyncio
async def test_handle_mcp_result_unwraps_single_element_list_with_json_string(
    test_copilot_service,
):
    """MCP content like '["{\\\"success\\\": true}"]' should be unwrapped for artifacts.

    Some MCP servers return results wrapped in a single-element list where the
    element is itself a JSON-encoded string. The normalize_jsonish helper should
    recursively unwrap this before passing raw_content to the artifact pipeline.
    """
    inner_json = '{"success": true, "tables": ["dim_funds"]}'
    wrapped_content = json.dumps([inner_json])  # '["{...}"]'

    message = _make_mcp_tool_result_message_with_content(wrapped_content)
    function_call = FunctionCall(_dummy_function)

    mcp_processor = Mock(side_effect=lambda **kwargs: _empty_async_gen())
    test_copilot_service._mcp_data_service.process_mcp_tool_response = mcp_processor

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=function_call,
        is_last_message=True,
    ):
        events.append(event)

    # The processor should receive the fully unwrapped dict
    mcp_processor.assert_called_once()
    call_kwargs = mcp_processor.call_args.kwargs
    assert call_kwargs["raw_content"] == {"success": True, "tables": ["dim_funds"]}


def _make_streamed_response_function_call() -> FunctionCall:
    def get_widget_data(data_sources: list[dict]) -> str:
        return "mock result"

    return FunctionCall(get_widget_data, data_sources=[{"id": "risk_metrics"}])


async def _mixed_streamed_response():
    async def _text_tokens():
        for token in ["Now let me ", "query the ", "strategies:"]:
            yield token

    yield AsyncStreamedStr(_text_tokens())
    yield _make_streamed_response_function_call()


async def _text_only_streamed_response():
    async def _text_tokens():
        for token in ["Here is ", "the final ", "answer."]:
            yield token

    yield AsyncStreamedStr(_text_tokens())


async def _function_only_streamed_response():
    yield _make_streamed_response_function_call()


@pytest.mark.asyncio
async def test_aparse_stream_drops_function_call_without_async_streamed_response():
    result = await aparse_stream(
        _mixed_streamed_response(),
        [AsyncStreamedStr, FunctionCall],
    )

    assert isinstance(result, AsyncStreamedStr)
    assert await result.to_string() == "Now let me query the strategies:"


@pytest.mark.asyncio
async def test_aparse_stream_preserves_mixed_response_with_async_streamed_response():
    result = await aparse_stream(
        _mixed_streamed_response(),
        [AsyncStreamedStr, FunctionCall, AsyncStreamedResponse],
    )

    assert isinstance(result, AsyncStreamedResponse)

    items = [item async for item in result]

    assert len(items) == 2
    assert isinstance(items[0], AsyncStreamedStr)
    assert await items[0].to_string() == "Now let me query the strategies:"
    assert isinstance(items[1], FunctionCall)
    assert items[1].function.__name__ == "get_widget_data"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stream_factory", "expected_item_type"),
    [
        (_text_only_streamed_response, AsyncStreamedStr),
        (_function_only_streamed_response, FunctionCall),
    ],
)
async def test_aparse_stream_wraps_single_item_responses_with_async_streamed_response(
    stream_factory,
    expected_item_type,
):
    result = await aparse_stream(
        stream_factory(),
        [AsyncStreamedStr, FunctionCall, AsyncStreamedResponse],
    )

    assert isinstance(result, AsyncStreamedResponse)

    items = [item async for item in result]

    assert len(items) == 1
    assert isinstance(items[0], expected_item_type)


@pytest.mark.asyncio
async def test_compose_chain_uses_async_streamed_response_output_type(
    test_copilot_service,
    monkeypatch,
):
    captured_return_type = None

    def fake_chatprompt(*args, **kwargs):
        del args, kwargs

        def decorator(func):
            nonlocal captured_return_type
            captured_return_type = func.__annotations__["return"]

            async def fake_chain():
                return None

            return fake_chain

        return decorator

    monkeypatch.setattr("openbb_ada.copilot.chatprompt", fake_chatprompt)
    monkeypatch.setattr(test_copilot_service, "_log_token_breakdown", lambda **_: None)
    monkeypatch.setattr(
        test_copilot_service, "_log_request_diagnostics", lambda **_: None
    )
    monkeypatch.setattr(
        test_copilot_service,
        "_is_within_context_limit",
        lambda messages: True,
    )
    monkeypatch.setattr(test_copilot_service, "_get_model", lambda: None)
    test_copilot_service._context_service.structured_context = None
    test_copilot_service._context_service.unstructured_context = None

    await test_copilot_service._compose_chain(
        chat_messages=[UserMessage("test prompt")],
        documents=None,
        web_pages=None,
        tools=None,
        original_messages=None,
        original_context=None,
        widget_collection=None,
        sql_widgets=None,
        python_widgets=None,
    )

    assert captured_return_type is AsyncStreamedResponse


@pytest.mark.asyncio
async def test_compose_chain_escapes_json_in_latest_user_message(
    test_copilot_service,
    monkeypatch,
):
    formatted_messages = []

    def fake_chatprompt(*messages, **kwargs):
        del kwargs

        def decorator(func):
            del func

            async def fake_chain():
                formatted_messages.extend(message.format() for message in messages)

            return fake_chain

        return decorator

    monkeypatch.setattr("openbb_ada.copilot.chatprompt", fake_chatprompt)
    monkeypatch.setattr(test_copilot_service, "_log_token_breakdown", lambda **_: None)
    monkeypatch.setattr(
        test_copilot_service, "_log_request_diagnostics", lambda **_: None
    )
    monkeypatch.setattr(
        test_copilot_service,
        "_is_within_context_limit",
        lambda messages: True,
    )
    monkeypatch.setattr(test_copilot_service, "_get_model", lambda: None)
    test_copilot_service._context_service.structured_context = None
    test_copilot_service._context_service.unstructured_context = None

    query = '{"uuid": "abc", "artifact_ref": "artifact-123"}'
    chain = await test_copilot_service._compose_chain(
        chat_messages=[UserMessage(query)],
        documents=None,
        web_pages=None,
        tools=None,
        original_messages=[
            LlmClientMessage(role=RoleEnum.human, content=query),
        ],
        original_context=None,
        widget_collection=None,
        sql_widgets=None,
        python_widgets=None,
    )

    await chain()

    user_message = next(
        message for message in formatted_messages if isinstance(message, UserMessage)
    )
    assert query in user_message.content


@pytest.mark.asyncio
async def test_compose_chain_raises_when_tool_count_exceeds_openai_limit(
    test_copilot_service,
    monkeypatch,
):
    monkeypatch.setattr(test_copilot_service, "_log_token_breakdown", lambda **_: None)
    monkeypatch.setattr(
        test_copilot_service, "_set_prompt_enhancement_context", lambda **_: None
    )
    monkeypatch.setattr(
        test_copilot_service,
        "_build_llm_functions",
        lambda **_: [Mock() for _ in range(129)],
    )
    test_copilot_service._flat_mcp_functions = [Mock() for _ in range(124)]
    test_copilot_service._context_service.structured_context = None
    test_copilot_service._context_service.unstructured_context = None

    with pytest.raises(ToolLimitExceededError) as exc_info:
        await test_copilot_service._compose_chain(
            chat_messages=[UserMessage("test prompt")],
            documents=None,
            web_pages=None,
            tools=None,
            original_messages=None,
            original_context=None,
            widget_collection=None,
            sql_widgets=None,
            python_widgets=None,
        )

    assert exc_info.value.total_tool_count == 129
    assert exc_info.value.mcp_tool_count == 124
    assert exc_info.value.threshold == 128


def test_merge_client_function_calls_combines_query_lists(test_copilot_service):
    first_call = FunctionCall(
        test_copilot_service._client_function_call_service.llm_query_extra_widgets,
        search_queries=[{"description": "news", "query": "AAPL news"}],
        summary="Searching widgets",
        extra_state=None,
    )
    second_call = FunctionCall(
        test_copilot_service._client_function_call_service.llm_query_extra_widgets,
        search_queries=[{"description": "price", "query": "MSFT price"}],
        summary="Searching widgets",
        extra_state=None,
    )

    merged_call = test_copilot_service._merge_client_function_calls(
        [first_call, second_call]
    )

    assert merged_call.function.__name__ == "llm_query_extra_widgets"
    assert merged_call.arguments["search_queries"] == [
        {"description": "news", "query": "AAPL news"},
        {"description": "price", "query": "MSFT price"},
    ]


def test_restore_deferred_function_calls_uses_function_registry(test_copilot_service):
    def fake_tool(query: str, summary: str = "Querying data"):
        return query, summary

    deferred_specs = [
        {
            "function_name": "fake_tool",
            "arguments": {"query": "SELECT 1", "summary": "Querying test data"},
        }
    ]

    restored_calls = test_copilot_service._restore_deferred_function_calls(
        deferred_specs=deferred_specs,
        function_registry={"fake_tool": fake_tool},
    )

    assert len(restored_calls) == 1
    assert restored_calls[0].function is fake_tool
    assert restored_calls[0].arguments == {
        "query": "SELECT 1",
        "summary": "Querying test data",
    }


def test_augment_client_function_call_event_serializes_intermediate_artifacts(
    test_copilot_service,
    test_template_service,
    test_sql_agent_service,
):
    context_service = ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )
    artifact_rows = [
        {"week": "Week 1", "amount": 25000},
        {"week": "Week 2", "amount": 30000},
    ]
    artifact = CopilotArtifact(
        content=artifact_rows,
        source_info=SourceInfo(
            type="artifact",
            uuid=UUID("00000000-0000-0000-0000-000000000001"),
            name="table_artifact_sales",
            description="Sales table artifact",
            citable=False,
        ),
        data_format=RawObjectDataFormat(parse_as="table"),
    )
    context_service.load_context(elements=[artifact.to_parsed_context()])

    test_copilot_service._context_service = context_service
    test_copilot_service._citation_service.citations = []

    client_event = FunctionCallSSE(
        data=FunctionCallSSEData(
            function="get_extra_widget_data",
            input_arguments={"data_sources": []},
            extra_state={},
        )
    )

    augmented_event = test_copilot_service._augment_client_function_call_event(
        client_fc_event=client_event,
        chat_messages=[],
        original_chat_messages_count=0,
    )

    assert augmented_event.data.extra_state is not None
    serialized_artifact = augmented_event.data.extra_state["intermediate_artifacts"][
        "table_artifact_sales"
    ]
    assert serialized_artifact["source_info"]["name"] == "table_artifact_sales"
    assert serialized_artifact["data_format"]["parse_as"] == "table"
    assert json.loads(serialized_artifact["content"]) == [
        {"index": 0, "week": "Week 1", "amount": 25000},
        {"index": 1, "week": "Week 2", "amount": 30000},
    ]


def test_augment_add_generative_widget_attaches_intermediate_artifacts_to_input_args(
    test_copilot_service,
    test_template_service,
    test_sql_agent_service,
):
    context_service = ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )
    artifact = CopilotArtifact(
        content=[
            {"country": "United States", "weight": 67.61},
            {"country": "Ireland", "weight": 9.017},
        ],
        source_info=SourceInfo(
            type="artifact",
            uuid=UUID("00000000-0000-0000-0000-000000000002"),
            name="table_artifact_country",
            description="Portfolio exposure by country",
            citable=False,
        ),
        data_format=RawObjectDataFormat(parse_as="table"),
    )
    context_service.load_context(elements=[artifact.to_parsed_context()])

    test_copilot_service._context_service = context_service
    test_copilot_service._citation_service.citations = []

    augmented_event = test_copilot_service._augment_client_function_call_event(
        client_fc_event=FunctionCallSSE(
            data=FunctionCallSSEData(
                function="add_generative_widget",
                input_arguments={
                    "widget_type": "note",
                    "data": (
                        "See table "
                        "<|start_artifact_id|>table_artifact_country"
                        "<|end_artifact_id|>"
                    ),
                    "name": "Portfolio Note",
                },
                extra_state={},
            )
        ),
        chat_messages=[],
        original_chat_messages_count=0,
    )

    input_artifact = augmented_event.data.input_arguments["artifacts"][0]
    assert input_artifact["source_info"]["name"] == "table_artifact_country"
    assert input_artifact["data_format"]["parse_as"] == "table"
    assert json.loads(input_artifact["content"]) == [
        {"index": 0, "country": "United States", "weight": 67.61},
        {"index": 1, "country": "Ireland", "weight": 9.017},
    ]
    assert (
        augmented_event.data.extra_state["intermediate_artifacts"][
            "table_artifact_country"
        ]
        == input_artifact
    )


def test_augment_add_generative_widget_keeps_inline_citation_id(
    test_copilot_service,
):
    stale_citation_id = UUID("00000000-0000-0000-0000-000000000001")
    inline_citation_id = UUID("00000000-0000-0000-0000-000000000002")
    source_info = SourceInfo(
        type="widget",
        uuid=UUID("00000000-0000-0000-0000-000000000003"),
        origin="Portfolio Risk",
        widget_id="portfolio_countries_custom_obb",
        name="Portfolio Exposure by Country",
        description="Get portfolio's exposure by country.",
        metadata={
            "input_args": {"portfolio": "Client 1"},
            "widget_uuid": "c32994c3-7d7c-4b93-950c-f7796d3c7070",
        },
        citable=True,
    )
    details = [
        {
            "Origin": "Portfolio Risk",
            "Data source": "Portfolio Exposure by Country",
            "Portfolio": "Client 1",
        }
    ]

    test_copilot_service._citation_service = CitationService()
    test_copilot_service._citation_service.add_citation(
        Citation(
            id=stale_citation_id,
            source_info=source_info,
            details=details,
        )
    )
    test_copilot_service._citation_service.add_citation(
        Citation(
            id=inline_citation_id,
            source_info=source_info,
            details=details,
        )
    )
    test_copilot_service._context_service.dump_roundtrip_artifacts = Mock(
        return_value={}
    )

    augmented_event = test_copilot_service._augment_client_function_call_event(
        client_fc_event=FunctionCallSSE(
            data=FunctionCallSSEData(
                function="add_generative_widget",
                input_arguments={
                    "widget_type": "note",
                    "data": (
                        "Portfolio exposure by country "
                        f"<|start_citation_id|>{inline_citation_id}"
                        "<|end_citation_id|>"
                    ),
                    "name": "Portfolio Note",
                },
                extra_state={},
            )
        ),
        chat_messages=[],
        original_chat_messages_count=0,
    )

    input_citations = augmented_event.data.input_arguments["citations"]
    input_citation_ids = [citation["id"] for citation in input_citations]
    assert input_citation_ids[0] == str(inline_citation_id)
    assert str(stale_citation_id) not in input_citation_ids
    assert augmented_event.data.extra_state["intermediate_citations"][0]["id"] == str(
        inline_citation_id
    )


def test_intermediate_artifacts_survive_openbb_ai_sse_and_message_models(
    test_copilot_service,
    test_template_service,
    test_sql_agent_service,
):
    query_data_source = {
        "origin": "SNOW backend",
        "id": "fred_public_dsge",
        "widget_uuid": "3a2b6d0f-79d1-4209-8bea-63c1bb8f48b9",
    }
    context_service = ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )
    query_artifact = CopilotArtifact(
        content="```sql\nSELECT * FROM FRED.PUBLIC.DSGE_SV\n```",
        source_info=SourceInfo(
            type="artifact",
            uuid=UUID("00000000-0000-0000-0000-000000000099"),
            name="query_artifact_dsge",
            description="Snowflake SQL query generated by AI copilot",
            metadata={
                "parse_as": "snowflake_query",
                "query_data_source": query_data_source,
            },
            citable=False,
        ),
        data_format=RawObjectDataFormat(
            parse_as="snowflake_query",
            query_data_source=query_data_source,
        ),
    )
    context_service.load_context(elements=[query_artifact.to_parsed_context()])

    test_copilot_service._context_service = context_service
    test_copilot_service._citation_service.citations = []

    augmented_event = test_copilot_service._augment_client_function_call_event(
        client_fc_event=FunctionCallSSE(
            data=FunctionCallSSEData(
                function="get_extra_widget_data",
                input_arguments={"data_sources": []},
                extra_state={},
            )
        ),
        chat_messages=[],
        original_chat_messages_count=0,
    )

    sse_payload = augmented_event.model_dump()
    function_call_data = json.loads(sse_payload["data"])
    tool_message = LlmClientFunctionCallResultMessage(
        function=function_call_data["function"],
        input_arguments=function_call_data["input_arguments"],
        data=[],
        extra_state=function_call_data["extra_state"],
    )

    restored_context_service = ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )
    restored_context_service.restore_roundtrip_artifacts(
        tool_message.extra_state["intermediate_artifacts"]
    )
    restored_context = restored_context_service.get_context_by_name(
        "query_artifact_dsge"
    )

    assert isinstance(restored_context, UnstructuredContext)
    assert restored_context.content == "```sql\nSELECT * FROM FRED.PUBLIC.DSGE_SV\n```"
    assert restored_context.source_info.metadata["parse_as"] == "snowflake_query"
    assert restored_context.source_info.metadata["query_data_source"] == (
        query_data_source
    )


@pytest.mark.asyncio
async def test_replay_sql_query_function_call_result_emits_get_widget_data(
    test_copilot_service,
):
    widget_uuid = "3a2b6d0f-79d1-4209-8bea-63c1bb8f48b9"
    sql_query = "SELECT * FROM FRED.PUBLIC.DSGE LIMIT 7"
    query_data_source = {
        "origin": "SNOW backend",
        "id": "fred_public_dsge",
        "widget_uuid": widget_uuid,
    }
    query_artifact = CopilotArtifact(
        content=f"```sql\n{sql_query}\n```",
        source_info=SourceInfo(
            type="artifact",
            uuid=UUID("00000000-0000-0000-0000-000000000199"),
            name="query_artifact_dsge",
            description="Snowflake SQL query generated by AI copilot",
            metadata={
                "parse_as": "snowflake_query",
                "query_data_source": query_data_source,
            },
            citable=False,
        ),
        data_format=RawObjectDataFormat(
            parse_as="snowflake_query",
            query_data_source=query_data_source,
        ),
    )
    sql_result = SqlQueryFunctionCallResult(
        sql_query=sql_query,
        widget_uuid=widget_uuid,
        widget_id="fred_public_dsge",
        widget_origin="SNOW backend",
        artifacts=[query_artifact],
    )

    context_service = Mock()
    context_service.dump_roundtrip_artifacts.return_value = {}
    test_copilot_service._context_service = context_service
    test_copilot_service._citation_service.citations = []

    (
        outbound_events,
        should_exit,
    ) = await test_copilot_service._replay_native_function_call_events(
        native_events=[sql_result],
        chat_messages=[],
        original_chat_messages_count=0,
        deferred_citations=[],
    )

    assert should_exit is True
    assert len(outbound_events) == 1
    function_event = outbound_events[0]
    assert isinstance(function_event, FunctionCallSSE)
    assert function_event.data.function == "get_widget_data"
    data_source = function_event.data.input_arguments["data_sources"][0]
    assert data_source["widget_uuid"] == widget_uuid
    assert data_source["origin"] == "SNOW backend"
    assert data_source["id"] == "fred_public_dsge"
    assert data_source["input_args"]["query"] == sql_query
    assert function_event.data.extra_state["sql_query"] == sql_query
    assert (
        function_event.data.extra_state["copilot_function_call_arguments"][
            "widget_queries"
        ][0]["use_current_inputs"]
        is False
    )
    context_service.load_context.assert_called_once()


def test_build_sql_artifact_dashboard_event_bridges_snowflake_query_artifact(
    test_copilot_service,
    test_template_service,
    test_sql_agent_service,
):
    context_service = ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )
    query_data_source = {
        "origin": "SNOW backend",
        "id": "fred_public_dsge",
        "widget_uuid": "3a2b6d0f-79d1-4209-8bea-63c1bb8f48b9",
    }
    query_artifact = CopilotArtifact(
        content="```sql\nSELECT * FROM FRED.PUBLIC.DSGE_SV\n```",
        source_info=SourceInfo(
            type="artifact",
            uuid=UUID("00000000-0000-0000-0000-000000000099"),
            name="query_artifact_dsge",
            description="Snowflake SQL query generated by AI copilot",
            metadata={
                "parse_as": "snowflake_query",
                "query_data_source": query_data_source,
            },
            citable=False,
        ),
        data_format=RawObjectDataFormat(
            parse_as="snowflake_query",
            query_data_source=query_data_source,
        ),
    )
    context_service.load_context(elements=[query_artifact.to_parsed_context()])

    test_copilot_service._context_service = context_service
    test_copilot_service._citation_service.citations = []
    client_function_call_service = test_copilot_service._client_function_call_service
    generate_widget_in_dashboard = (
        client_function_call_service.llm_generate_widget_in_dashboard
    )

    function_call = FunctionCall(
        generate_widget_in_dashboard,
        widget_type="table",
        artifact_id="query_artifact_dsge",
        inner_tab="Summary",
        summary="Adding DSGE table to dashboard",
    )

    bridged_event = test_copilot_service._build_sql_artifact_dashboard_event(
        function_call
    )

    assert bridged_event is not None
    assert bridged_event.data.function == "add_widget_to_dashboard"
    data_source = bridged_event.data.input_arguments["data_sources"][0]
    assert data_source["widget_uuid"] == query_data_source["widget_uuid"]
    assert data_source["origin"] == query_data_source["origin"]
    assert data_source["id"] == query_data_source["id"]
    assert data_source["input_args"]["query"] == "SELECT * FROM FRED.PUBLIC.DSGE_SV"
    assert data_source["input_args"]["inner_tab"] == "Summary"
    assert bridged_event.data.extra_state["sql_query"] == (
        "SELECT * FROM FRED.PUBLIC.DSGE_SV"
    )
    assert bridged_event.data.extra_state["copilot_function_call_arguments"] == {
        "summary": "Adding DSGE table to dashboard",
        "search_queries": [
            {
                "description": "fred_public_dsge",
                "query": "SELECT * FROM FRED.PUBLIC.DSGE_SV",
                "inner_tab": "Summary",
            }
        ],
    }


def test_build_sql_artifact_dashboard_event_unwraps_json_encoded_sql_artifact(
    test_copilot_service,
    test_template_service,
    test_sql_agent_service,
):
    context_service = ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )
    query_data_source = {
        "origin": "SNOW backend",
        "id": "fred_public_dsge",
        "widget_uuid": "3a2b6d0f-79d1-4209-8bea-63c1bb8f48b9",
    }
    query_artifact = CopilotArtifact(
        content=json.dumps("```sql\nSELECT * FROM FRED.PUBLIC.DSGE_SV\n```"),
        source_info=SourceInfo(
            type="artifact",
            uuid=UUID("00000000-0000-0000-0000-000000000099"),
            name="query_artifact_dsge",
            description="Snowflake SQL query generated by AI copilot",
            metadata={
                "parse_as": "snowflake_query",
                "query_data_source": query_data_source,
            },
            citable=False,
        ),
        data_format=RawObjectDataFormat(
            parse_as="snowflake_query",
            query_data_source=query_data_source,
        ),
    )
    context_service.load_context(elements=[query_artifact.to_parsed_context()])

    test_copilot_service._context_service = context_service
    test_copilot_service._citation_service.citations = []
    client_function_call_service = test_copilot_service._client_function_call_service
    generate_widget_in_dashboard = (
        client_function_call_service.llm_generate_widget_in_dashboard
    )

    function_call = FunctionCall(
        generate_widget_in_dashboard,
        widget_type="table",
        artifact_id="query_artifact_dsge",
        summary="Adding DSGE table to dashboard",
    )

    bridged_event = test_copilot_service._build_sql_artifact_dashboard_event(
        function_call
    )

    assert bridged_event is not None
    data_source = bridged_event.data.input_arguments["data_sources"][0]
    assert data_source["input_args"]["query"] == "SELECT * FROM FRED.PUBLIC.DSGE_SV"


def test_build_sql_artifact_dashboard_event_preserves_chart_widget_intent(
    test_copilot_service,
    test_template_service,
    test_sql_agent_service,
):
    context_service = ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )
    query_data_source = {
        "origin": "SNOW backend",
        "id": "fred_public_dsge",
        "widget_uuid": "3a2b6d0f-79d1-4209-8bea-63c1bb8f48b9",
    }
    query_artifact = CopilotArtifact(
        content="```sql\nSELECT * FROM FRED.PUBLIC.DSGE_SV\n```",
        source_info=SourceInfo(
            type="artifact",
            uuid=UUID("00000000-0000-0000-0000-000000000099"),
            name="query_artifact_dsge",
            description="Snowflake SQL query generated by AI copilot",
            metadata={
                "parse_as": "snowflake_query",
                "query_data_source": query_data_source,
            },
            citable=False,
        ),
        data_format=RawObjectDataFormat(
            parse_as="snowflake_query",
            query_data_source=query_data_source,
        ),
    )
    context_service.load_context(elements=[query_artifact.to_parsed_context()])

    test_copilot_service._context_service = context_service
    test_copilot_service._citation_service.citations = []
    client_function_call_service = test_copilot_service._client_function_call_service
    generate_widget_in_dashboard = (
        client_function_call_service.llm_generate_widget_in_dashboard
    )

    function_call = FunctionCall(
        generate_widget_in_dashboard,
        widget_type="chart",
        artifact_id="query_artifact_dsge",
        name="DSGE Macro Model Chart",
        description="Line chart of DSGE metrics",
        chart_params={
            "chartType": "line",
            "xKey": "date",
            "yKey": ["ffr"],
        },
        summary="Adding DSGE chart to dashboard",
    )

    bridged_event = test_copilot_service._build_sql_artifact_dashboard_event(
        function_call
    )

    assert bridged_event is not None
    assert bridged_event.data.function == "add_widget_to_dashboard"
    data_source = bridged_event.data.input_arguments["data_sources"][0]
    assert data_source["widget_uuid"] == query_data_source["widget_uuid"]
    assert data_source["origin"] == query_data_source["origin"]
    assert data_source["id"] == query_data_source["id"]
    assert data_source["input_args"]["query"] == "SELECT * FROM FRED.PUBLIC.DSGE_SV"
    assert bridged_event.data.extra_state["copilot_function_call_arguments"] == {
        "widget_type": "chart",
        "name": "DSGE Macro Model Chart",
        "description": "Line chart of DSGE metrics",
        "chart_params": {
            "chartType": "line",
            "xKey": "date",
            "yKey": ["ffr"],
        },
        "summary": "Adding DSGE chart to dashboard",
        "search_queries": [
            {
                "description": "fred_public_dsge",
                "query": "SELECT * FROM FRED.PUBLIC.DSGE_SV",
            }
        ],
    }


def _prepare_query_test_state(test_copilot_service) -> None:
    test_copilot_service._context_service.structured_context = []
    test_copilot_service._context_service.unstructured_context = []
    test_copilot_service._context_service.dump_roundtrip_artifacts.return_value = {}
    test_copilot_service._copilot_data_service.get_widget_collection.return_value = None
    test_copilot_service._citation_service.citations = []
    test_copilot_service._citation_service.clear = Mock()


def _make_structured_data_function_call(
    test_copilot_service,
    *,
    query: str,
    summary: str,
) -> FunctionCall:
    return FunctionCall(
        test_copilot_service._native_function_call_service.llm_query_structured_data,
        query=query,
        summary=summary,
    )


def _make_extra_widget_boundary_messages(
    deferred_function_calls: list[dict[str, object]] | None = None,
) -> tuple[LlmClientMessage, LlmClientFunctionCallResultMessage]:
    return (
        LlmClientMessage(
            role=RoleEnum.ai,
            content=LlmClientFunctionCall(
                function="get_extra_widget_data",
                input_arguments={"data_sources": []},
            ),
        ),
        LlmClientFunctionCallResultMessage(
            function="get_extra_widget_data",
            input_arguments={"data_sources": []},
            data=[],
            extra_state={
                "copilot_function_call_arguments": {
                    "search_queries": [
                        {
                            "description": "chart",
                            "query": "AAPL price chart",
                        }
                    ],
                    "summary": "Searching widgets",
                },
                "deferred_function_calls": deferred_function_calls or [],
            },
        ),
    )


@pytest.mark.asyncio
async def test_query_batches_backend_calls_and_defers_tail_after_client_boundary(
    test_copilot_service,
    monkeypatch,
):
    _prepare_query_test_state(test_copilot_service)

    backend_call_order: list[str] = []

    backend_call_1 = _make_structured_data_function_call(
        test_copilot_service,
        query="table for strategy A",
        summary="Querying strategy A",
    )
    client_call = FunctionCall(
        test_copilot_service._client_function_call_service.llm_query_extra_widgets,
        search_queries=[{"description": "chart", "query": "AAPL price chart"}],
        summary="Searching widgets",
        extra_state=None,
    )
    backend_call_2 = _make_structured_data_function_call(
        test_copilot_service,
        query="table for strategy B",
        summary="Querying strategy B",
    )

    async def streamed_response_items():
        yield backend_call_1
        yield client_call
        yield backend_call_2

    async def fake_run_agent_step():
        return await aparse_stream(
            streamed_response_items(),
            [AsyncStreamedStr, FunctionCall, AsyncStreamedResponse],
        )

    async def fake_compose_chain(**kwargs):
        del kwargs
        return fake_run_agent_step

    async def fake_native_handle(function_call):
        backend_call_order.append(function_call.arguments["query"])
        yield [
            AssistantMessage(function_call),
            FunctionResultMessage(
                content=f"Result for {function_call.arguments['query']}",
                function_call=function_call,
            ),
        ]

    async def fake_client_handle(function_call):
        yield FunctionCallSSE(
            data=FunctionCallSSEData(
                function="get_extra_widget_data",
                input_arguments={"data_sources": []},
                extra_state={
                    "copilot_function_call_arguments": function_call.arguments,
                },
            )
        )

    monkeypatch.setattr(test_copilot_service, "_compose_chain", fake_compose_chain)
    monkeypatch.setattr(
        test_copilot_service._native_function_call_service,
        "handle_function_calls",
        fake_native_handle,
    )
    monkeypatch.setattr(
        test_copilot_service._client_function_call_service,
        "handle_function_calls",
        fake_client_handle,
    )

    events = []
    async for event in test_copilot_service.query(
        messages=[LlmClientMessage(role=RoleEnum.human, content="compare strategies")]
    ):
        events.append(event)

    function_call_events = [
        event for event in events if isinstance(event, FunctionCallSSE)
    ]

    assert backend_call_order == ["table for strategy A"]
    assert len(function_call_events) == 1
    assert function_call_events[0].data.extra_state["deferred_function_calls"] == [
        {
            "function_name": "llm_query_structured_data",
            "arguments": {
                "query": "table for strategy B",
                "summary": "Querying strategy B",
            },
        }
    ]


@pytest.mark.asyncio
async def test_query_parallel_native_calls_stream_status_updates_before_completion(
    test_copilot_service,
    monkeypatch,
):
    _prepare_query_test_state(test_copilot_service)

    slow_query_released = asyncio.Event()
    compose_call_count = 0

    slow_call = _make_structured_data_function_call(
        test_copilot_service,
        query="slow query",
        summary="Querying slow data",
    )
    fast_call = _make_structured_data_function_call(
        test_copilot_service,
        query="fast query",
        summary="Querying fast data",
    )

    async def streamed_response_items():
        yield slow_call
        yield fast_call

    async def fake_compose_chain(**kwargs):
        del kwargs

        async def run_agent_step():
            nonlocal compose_call_count
            compose_call_count += 1
            if compose_call_count == 1:
                return await aparse_stream(
                    streamed_response_items(),
                    [AsyncStreamedStr, FunctionCall, AsyncStreamedResponse],
                )

        return run_agent_step

    async def fake_native_handle(function_call):
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=function_call.arguments["summary"],
            )
        )
        if function_call.arguments["query"] == "slow query":
            await slow_query_released.wait()
        else:
            await asyncio.sleep(0.01)
        yield [
            AssistantMessage(function_call),
            FunctionResultMessage(
                content=f"Result for {function_call.arguments['query']}",
                function_call=function_call,
            ),
        ]

    monkeypatch.setattr(test_copilot_service, "_compose_chain", fake_compose_chain)
    monkeypatch.setattr(
        test_copilot_service._native_function_call_service,
        "handle_function_calls",
        fake_native_handle,
    )

    query_stream = test_copilot_service.query(
        messages=[LlmClientMessage(role=RoleEnum.human, content="compare datasets")]
    )

    first_event = await asyncio.wait_for(query_stream.__anext__(), timeout=0.5)
    assert isinstance(first_event, StatusUpdateSSE)
    assert first_event.data.message in {"Querying slow data", "Querying fast data"}

    slow_query_released.set()
    remaining_events = [event async for event in query_stream]
    status_messages = [
        event.data.message
        for event in [first_event, *remaining_events]
        if isinstance(event, StatusUpdateSSE)
    ]
    assert "Querying slow data" in status_messages
    assert "Querying fast data" in status_messages


@pytest.mark.asyncio
async def test_query_returns_artifact_when_model_completes_without_visible_reply(
    test_copilot_service,
    test_template_service,
    test_sql_agent_service,
    monkeypatch,
):
    _prepare_query_test_state(test_copilot_service)
    context_service = ContextService(
        template_service=test_template_service,
        sql_agent_service=test_sql_agent_service,
        logging_service=LoggingService(),
    )
    test_copilot_service._context_service = context_service

    compose_call_count = 0
    chart_call = _make_structured_data_function_call(
        test_copilot_service,
        query="create revenue chart",
        summary="Querying revenue data",
    )
    chart_artifact = CopilotArtifact(
        content=[
            {"year": 2020, "angola_revenue_meur": 370.7},
            {"year": 2021, "angola_revenue_meur": 401.2},
        ],
        source_info=SourceInfo(
            type="artifact",
            uuid=UUID("00000000-0000-0000-0000-000000000001"),
            name="chart_artifact_test",
            description="Revenue chart",
            citable=False,
        ),
        data_format=RawObjectDataFormat(
            parse_as="chart",
            chart_params=BarChartParameters(
                chartType="bar",
                xKey="year",
                yKey=["angola_revenue_meur"],
            ),
        ),
    )
    context_service.load_context(elements=[chart_artifact.to_parsed_context()])

    async def fake_compose_chain(**kwargs):
        del kwargs

        async def run_agent_step():
            nonlocal compose_call_count
            compose_call_count += 1
            if compose_call_count == 1:
                return chart_call
            return FunctionCall(
                test_copilot_service._native_function_call_service.llm_complete
            )

        return run_agent_step

    async def fake_native_handle(function_call):
        template_service = test_copilot_service._template_service
        result_content = template_service.render_copilot_native_function_call_result(
            answer="Created the revenue chart.",
            artifact=chart_artifact,
        )
        yield [
            AssistantMessage(function_call),
            FunctionResultMessage(
                content=result_content,
                function_call=function_call,
            ),
        ]
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message="Artifact generated",
                details=[],
                artifacts=[chart_artifact.to_client_artifact()],
            )
        )

    monkeypatch.setattr(test_copilot_service, "_compose_chain", fake_compose_chain)
    monkeypatch.setattr(
        test_copilot_service._native_function_call_service,
        "handle_function_calls",
        fake_native_handle,
    )

    events = []
    async for event in test_copilot_service.query(
        messages=[LlmClientMessage(role=RoleEnum.human, content="create chart")]
    ):
        events.append(event)

    message_text = "".join(
        event.data.delta for event in events if isinstance(event, MessageChunkSSE)
    )
    message_artifacts = [
        event for event in events if isinstance(event, MessageArtifactSSE)
    ]

    assert "Created the requested artifact." in message_text
    assert len(message_artifacts) == 1
    assert message_artifacts[0].data.uuid == chart_artifact.source_info.uuid


@pytest.mark.asyncio
async def test_query_passes_semantic_view_to_prompt_without_emitting_status(
    test_copilot_service,
    monkeypatch,
):
    _prepare_query_test_state(test_copilot_service)
    test_copilot_service._context_service.structured_context = None
    test_copilot_service._native_function_call_service._semantic_views = [
        "DB.SCHEMA.EXPLICIT_VIEW"
    ]
    captured_semantic_views: list[str] | None = None

    async def fake_compose_chain(**kwargs):
        nonlocal captured_semantic_views
        captured_semantic_views = kwargs.get("semantic_views_for_prompt")

        async def run_agent_step():
            return FunctionCall(
                test_copilot_service._native_function_call_service.llm_complete
            )

        return run_agent_step

    monkeypatch.setattr(test_copilot_service, "_compose_chain", fake_compose_chain)

    events = []
    async for event in test_copilot_service.query(
        messages=[LlmClientMessage(role=RoleEnum.human, content="show revenue")]
    ):
        events.append(event)

    semantic_status_events = [
        event
        for event in events
        if isinstance(event, StatusUpdateSSE)
        and event.data.message == "Using semantic view context"
    ]

    assert semantic_status_events == []
    assert captured_semantic_views == ["DB.SCHEMA.EXPLICIT_VIEW"]


@pytest.mark.asyncio
async def test_query_parallel_native_calls_keep_successes_when_one_fails(
    test_copilot_service,
    monkeypatch,
):
    _prepare_query_test_state(test_copilot_service)

    compose_call_count = 0
    captured_second_turn_messages = None

    failing_call = _make_structured_data_function_call(
        test_copilot_service,
        query="failing query",
        summary="Querying failing data",
    )
    successful_call = _make_structured_data_function_call(
        test_copilot_service,
        query="successful query",
        summary="Querying successful data",
    )

    async def streamed_response_items():
        yield failing_call
        yield successful_call

    async def fake_compose_chain(**kwargs):
        nonlocal compose_call_count, captured_second_turn_messages
        compose_call_count += 1
        if compose_call_count == 2:
            captured_second_turn_messages = kwargs["chat_messages"]

        async def run_agent_step():
            if compose_call_count == 1:
                return await aparse_stream(
                    streamed_response_items(),
                    [AsyncStreamedStr, FunctionCall, AsyncStreamedResponse],
                )
            return FunctionCall(
                test_copilot_service._native_function_call_service.llm_complete
            )

        return run_agent_step

    async def fake_native_handle(function_call):
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=function_call.arguments["summary"],
            )
        )
        if function_call.arguments["query"] == "failing query":
            raise RuntimeError("simulated parallel failure")
        yield [
            AssistantMessage(function_call),
            FunctionResultMessage(
                content=f"Result for {function_call.arguments['query']}",
                function_call=function_call,
            ),
        ]

    monkeypatch.setattr(test_copilot_service, "_compose_chain", fake_compose_chain)
    monkeypatch.setattr(
        test_copilot_service._native_function_call_service,
        "handle_function_calls",
        fake_native_handle,
    )

    events = []
    async for event in test_copilot_service.query(
        messages=[LlmClientMessage(role=RoleEnum.human, content="compare datasets")]
    ):
        events.append(event)

    status_messages = [
        event.data.message for event in events if isinstance(event, StatusUpdateSSE)
    ]
    assert "Querying failing data" in status_messages
    assert "Querying successful data" in status_messages

    assert captured_second_turn_messages is not None
    function_results = [
        message
        for message in captured_second_turn_messages
        if isinstance(message, FunctionResultMessage)
    ]
    assert any(
        message.content == "Result for successful query" for message in function_results
    )
    assert any(
        (
            "Tool call `llm_query_structured_data` failed with RuntimeError: "
            "simulated parallel failure"
        )
        == message.content
        for message in function_results
    )


@pytest.mark.asyncio
async def test_query_warns_and_closes_stream_before_openai_when_tool_limit_exceeded(
    test_copilot_service,
    monkeypatch,
):
    _prepare_query_test_state(test_copilot_service)
    test_copilot_service._context_service.structured_context = Mock()

    async def fail_if_called(**kwargs):
        del kwargs
        pytest.fail("_compose_chain should not run when tool limit preflight fails")

    monkeypatch.setattr(test_copilot_service, "_compose_chain", fail_if_called)

    tools = [_make_agent_tool(f"mcp_tool_{idx}") for idx in range(124)]

    events = []
    async for event in test_copilot_service.query(
        messages=[LlmClientMessage(role=RoleEnum.human, content="compare datasets")],
        tools=tools,
    ):
        events.append(event)

    assert len(events) == 1
    assert isinstance(events[0], StatusUpdateSSE)
    assert events[0].data.eventType == "WARNING"
    assert "128" in events[0].data.message
    assert "129 tools total" in events[0].data.details[0]["Detail"]
    assert "124 MCP tools" in events[0].data.details[0]["Detail"]


@pytest.mark.asyncio
async def test_query_resumed_request_emits_status_before_deferred_native_call_finishes(
    test_copilot_service,
    monkeypatch,
):
    _prepare_query_test_state(test_copilot_service)
    test_copilot_service._context_service.structured_context = Mock()

    deferred_native_call_released = asyncio.Event()

    (
        resumed_function_call_message,
        resumed_function_result_message,
    ) = _make_extra_widget_boundary_messages(
        deferred_function_calls=[
            {
                "function_name": "llm_query_structured_data",
                "arguments": {
                    "query": "resumed structured query",
                    "summary": "Querying resumed data",
                },
            }
        ]
    )

    async def fake_native_handle(function_call):
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=function_call.arguments["summary"],
            )
        )
        await deferred_native_call_released.wait()
        yield [
            AssistantMessage(function_call),
            FunctionResultMessage(
                content="Deferred native result",
                function_call=function_call,
            ),
        ]

    monkeypatch.setattr(
        test_copilot_service._native_function_call_service,
        "handle_function_calls",
        fake_native_handle,
    )

    query_stream = test_copilot_service.query(
        messages=[
            resumed_function_call_message,
            resumed_function_result_message,
        ]
    )

    first_event = await asyncio.wait_for(query_stream.__anext__(), timeout=0.1)
    assert isinstance(first_event, StatusUpdateSSE)
    assert first_event.data.message == "Querying resumed data"

    deferred_native_call_released.set()
    await query_stream.aclose()


@pytest.mark.asyncio
async def test_handle_mcp_result_unwraps_single_element_list_with_dict(
    test_copilot_service,
):
    """MCP content like '[{"success": true}]' should be unwrapped to the inner dict."""
    wrapped_content = '[{"success": true, "tables": ["dim_funds"]}]'

    message = _make_mcp_tool_result_message_with_content(wrapped_content)
    function_call = FunctionCall(_dummy_function)

    mcp_processor = Mock(side_effect=lambda **kwargs: _empty_async_gen())
    test_copilot_service._mcp_data_service.process_mcp_tool_response = mcp_processor

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=function_call,
        is_last_message=True,
    ):
        events.append(event)

    mcp_processor.assert_called_once()
    call_kwargs = mcp_processor.call_args.kwargs
    assert call_kwargs["raw_content"] == {"success": True, "tables": ["dim_funds"]}


@pytest.mark.asyncio
async def test_handle_mcp_result_plain_json_object_passed_directly(
    test_copilot_service,
):
    """MCP content that is a direct JSON object should be passed through as-is."""
    json_content = '{"success": true, "tables": ["dim_funds"]}'

    message = _make_mcp_tool_result_message_with_content(json_content)
    function_call = FunctionCall(_dummy_function)

    mcp_processor = Mock(side_effect=lambda **kwargs: _empty_async_gen())
    test_copilot_service._mcp_data_service.process_mcp_tool_response = mcp_processor

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=function_call,
        is_last_message=True,
    ):
        events.append(event)

    mcp_processor.assert_called_once()
    call_kwargs = mcp_processor.call_args.kwargs
    assert call_kwargs["raw_content"] == {"success": True, "tables": ["dim_funds"]}
