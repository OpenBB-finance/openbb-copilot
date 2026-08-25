"""
Tests for param options result parsing in ClientFunctionCallService.

Covers `_parse_param_options_results` and the branches that consume it in
`_get_partial_query_widget_requests_with_options_from_extra_state_for_index`
and
`_get_partial_query_extra_widget_requests_with_options_from_extra_state_for_index`,
including the best-effort skip of param options sub-queries that failed on the
client (returned as ClientFunctionCallError) or came back malformed.
"""

import json
from typing import Any
from unittest.mock import Mock
from uuid import uuid4

import pytest
from openbb_ai.models import (
    ClientCommandResult,
    ClientFunctionCallError,
    DataContent,
    LlmClientFunctionCallResultMessage,
    SingleDataContent,
)

from openbb_ada.models import (
    DataSourceSearchQuery,
    QueryExtraWidgetsRequest,
    QueryWidgetRequest,
    WidgetQueryRequest,
)
from openbb_ada.services import ClientFunctionCallService, LoggingService


@pytest.fixture
def client_function_call_service() -> ClientFunctionCallService:
    """Create a ClientFunctionCallService instance for testing."""
    return ClientFunctionCallService(
        copilot_data_service=Mock(),
        logging_service=LoggingService(),
    )


def make_param_options_data_item(
    param: str = "symbol",
    options: list[dict[str, str]] | None = None,
) -> DataContent:
    """Build a well-formed param options result data item."""
    if options is None:
        options = [{"label": "Gold in USD", "value": "gold-in-usd"}]
    return DataContent(
        items=[
            SingleDataContent(
                content=json.dumps(
                    {"param_options": [{"param": param, "options": options}]}
                )
            )
        ]
    )


def make_client_error() -> ClientFunctionCallError:
    return ClientFunctionCallError(
        error_type="unexpected_error",
        content="Failed to fetch param options.",
    )


def make_intermediate_tool_call_result(
    param_queries: list[dict[str, Any]],
    data: list[Any],
) -> LlmClientFunctionCallResultMessage:
    return LlmClientFunctionCallResultMessage(
        function="get_params_options",
        input_arguments={"param_options_queries": param_queries},
        data=data,
    )


class TestParseParamOptionsResults:
    """Tests for _parse_param_options_results."""

    def test_empty_input_returns_empty_list(
        self, client_function_call_service: ClientFunctionCallService
    ):
        assert client_function_call_service._parse_param_options_results([]) == []

    def test_parses_successful_results_in_order(
        self, client_function_call_service: ClientFunctionCallService
    ):
        data_items = [
            make_param_options_data_item(param="symbol"),
            make_param_options_data_item(param="country"),
        ]
        results = client_function_call_service._parse_param_options_results(data_items)
        assert [r["param"] for r in results] == ["symbol", "country"]

    def test_client_error_yields_none_and_keeps_index_alignment(
        self, client_function_call_service: ClientFunctionCallService
    ):
        data_items = [
            make_client_error(),
            make_param_options_data_item(param="country"),
        ]
        results = client_function_call_service._parse_param_options_results(data_items)
        assert len(results) == 2
        assert results[0] is None
        assert results[1]["param"] == "country"

    def test_invalid_json_yields_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        data_items = [
            DataContent(items=[SingleDataContent(content="not valid json")]),
        ]
        results = client_function_call_service._parse_param_options_results(data_items)
        assert results == [None]

    def test_missing_param_options_key_yields_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        data_items = [
            DataContent(items=[SingleDataContent(content=json.dumps({"other": []}))]),
        ]
        results = client_function_call_service._parse_param_options_results(data_items)
        assert results == [None]

    def test_empty_param_options_list_yields_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        data_items = [
            DataContent(
                items=[SingleDataContent(content=json.dumps({"param_options": []}))]
            ),
        ]
        results = client_function_call_service._parse_param_options_results(data_items)
        assert results == [None]

    def test_empty_items_list_yields_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        data_items = [DataContent(items=[])]
        results = client_function_call_service._parse_param_options_results(data_items)
        assert results == [None]

    def test_data_item_without_items_attribute_yields_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        data_items = [ClientCommandResult(status="success")]
        results = client_function_call_service._parse_param_options_results(data_items)
        assert results == [None]

    def test_non_dict_json_content_yields_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        data_items = [
            DataContent(items=[SingleDataContent(content=json.dumps(["a", "b"]))]),
        ]
        results = client_function_call_service._parse_param_options_results(data_items)
        assert results == [None]


class TestGetPartialQueryWidgetRequestsWithOptions:
    """Tests for the widget-query branch consuming parsed param options."""

    def _call(
        self,
        service: ClientFunctionCallService,
        extra_state: dict[str, Any],
        widget_queries: list[WidgetQueryRequest] | None = None,
        widget_query_index: int = 0,
    ) -> QueryWidgetRequest | None:
        if widget_queries is None:
            widget_queries = [
                WidgetQueryRequest(
                    widget_uuid=uuid4(),
                    query="What is the price of gold in USD?",
                )
            ]
        return service._get_partial_query_widget_requests_with_options_from_extra_state_for_index(  # noqa: E501
            widget_queries=widget_queries,
            widget_query_index=widget_query_index,
            extra_state=extra_state,
        )

    def test_returns_none_without_intermediate_tool_call_result(
        self, client_function_call_service: ClientFunctionCallService
    ):
        assert self._call(client_function_call_service, extra_state={}) is None

    def test_returns_none_for_other_function(
        self, client_function_call_service: ClientFunctionCallService
    ):
        extra_state = {
            "intermediate_tool_call_result": LlmClientFunctionCallResultMessage(
                function="get_widget_data",
                data=[make_param_options_data_item()],
            ),
            "param_options_widget_query_mapping": [{"widget_query_index": 0}],
        }
        assert self._call(client_function_call_service, extra_state) is None

    def test_happy_path_builds_request_with_options(
        self, client_function_call_service: ClientFunctionCallService
    ):
        widget_queries = [
            WidgetQueryRequest(
                widget_uuid=uuid4(),
                query="What is the price of gold in USD?",
            )
        ]
        extra_state = {
            "intermediate_tool_call_result": make_intermediate_tool_call_result(
                param_queries=[{"origin": "test_origin", "id": "price_feeds"}],
                data=[make_param_options_data_item(param="symbol")],
            ),
            "param_options_widget_query_mapping": [
                {"widget_query_index": 0, "partial_input_args": {"interval": "1d"}},
            ],
        }
        result = self._call(
            client_function_call_service, extra_state, widget_queries=widget_queries
        )
        assert isinstance(result, QueryWidgetRequest)
        assert result.widget_uuid == widget_queries[0].widget_uuid
        assert result.widget_query == widget_queries[0].query
        assert result.partial_input_args == {"interval": "1d"}
        assert len(result.extra_param_options) == 1
        assert result.extra_param_options[0].widget_origin == "test_origin"
        assert result.extra_param_options[0].widget_id == "price_feeds"
        assert result.extra_param_options[0].param_name == "symbol"
        assert result.extra_param_options[0].options[0].value == "gold-in-usd"

    def test_failed_sub_query_is_skipped_and_successful_ones_kept(
        self, client_function_call_service: ClientFunctionCallService
    ):
        extra_state = {
            "intermediate_tool_call_result": make_intermediate_tool_call_result(
                param_queries=[
                    {"origin": "test_origin", "id": "price_feeds"},
                    {"origin": "test_origin", "id": "price_feeds"},
                ],
                data=[
                    make_client_error(),
                    make_param_options_data_item(param="country"),
                ],
            ),
            "param_options_widget_query_mapping": [
                {"widget_query_index": 0},
                {"widget_query_index": 0},
            ],
        }
        result = self._call(client_function_call_service, extra_state)
        assert result is not None
        assert len(result.extra_param_options) == 1
        assert result.extra_param_options[0].param_name == "country"

    def test_all_sub_queries_failed_returns_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        extra_state = {
            "intermediate_tool_call_result": make_intermediate_tool_call_result(
                param_queries=[{"origin": "test_origin", "id": "price_feeds"}],
                data=[make_client_error()],
            ),
            "param_options_widget_query_mapping": [{"widget_query_index": 0}],
        }
        assert self._call(client_function_call_service, extra_state) is None

    def test_mapping_index_beyond_results_is_skipped(
        self, client_function_call_service: ClientFunctionCallService
    ):
        extra_state = {
            "intermediate_tool_call_result": make_intermediate_tool_call_result(
                param_queries=[{"origin": "test_origin", "id": "price_feeds"}],
                data=[make_param_options_data_item(param="symbol")],
            ),
            "param_options_widget_query_mapping": [
                {"widget_query_index": 0},
                # No matching param query/result for this second mapping entry.
                {"widget_query_index": 0},
            ],
        }
        result = self._call(client_function_call_service, extra_state)
        assert result is not None
        assert len(result.extra_param_options) == 1
        assert result.extra_param_options[0].param_name == "symbol"

    def test_mapping_for_other_widget_query_index_returns_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        extra_state = {
            "intermediate_tool_call_result": make_intermediate_tool_call_result(
                param_queries=[{"origin": "test_origin", "id": "price_feeds"}],
                data=[make_param_options_data_item(param="symbol")],
            ),
            "param_options_widget_query_mapping": [{"widget_query_index": 1}],
        }
        assert self._call(client_function_call_service, extra_state) is None


class TestGetPartialQueryExtraWidgetRequestsWithOptions:
    """Tests for the extra-widgets branch consuming parsed param options."""

    def _call(
        self,
        service: ClientFunctionCallService,
        extra_state: dict[str, Any],
        widget_query_index: int = 0,
    ) -> QueryExtraWidgetsRequest | None:
        search_queries = [
            DataSourceSearchQuery(
                description="A widget with price feeds.",
                query="What is the price of gold in USD?",
                user_context="original user question",
            )
        ]
        return service._get_partial_query_extra_widget_requests_with_options_from_extra_state_for_index(  # noqa: E501
            search_queries=search_queries,
            widget_query_index=widget_query_index,
            extra_state=extra_state,
        )

    def test_happy_path_builds_request_with_options(
        self, client_function_call_service: ClientFunctionCallService
    ):
        extra_state = {
            "intermediate_tool_call_result": make_intermediate_tool_call_result(
                param_queries=[{"origin": "test_origin", "id": "price_feeds"}],
                data=[make_param_options_data_item(param="symbol")],
            ),
            "param_options_widget_query_mapping": [
                {"widget_query_index": 0, "partial_input_args": {"interval": "1d"}},
            ],
        }
        result = self._call(client_function_call_service, extra_state)
        assert isinstance(result, QueryExtraWidgetsRequest)
        assert result.data_source_description == "A widget with price feeds."
        assert result.widget_query == "What is the price of gold in USD?"
        assert result.user_context == "original user question"
        assert result.partial_input_args == {"interval": "1d"}
        assert len(result.extra_param_options) == 1
        assert result.extra_param_options[0].param_name == "symbol"

    def test_failed_sub_query_is_skipped_and_successful_ones_kept(
        self, client_function_call_service: ClientFunctionCallService
    ):
        extra_state = {
            "intermediate_tool_call_result": make_intermediate_tool_call_result(
                param_queries=[
                    {"origin": "test_origin", "id": "price_feeds"},
                    {"origin": "test_origin", "id": "price_feeds"},
                ],
                data=[
                    make_client_error(),
                    make_param_options_data_item(param="country"),
                ],
            ),
            "param_options_widget_query_mapping": [
                {"widget_query_index": 0},
                {"widget_query_index": 0},
            ],
        }
        result = self._call(client_function_call_service, extra_state)
        assert result is not None
        assert len(result.extra_param_options) == 1
        assert result.extra_param_options[0].param_name == "country"

    def test_all_sub_queries_failed_returns_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        extra_state = {
            "intermediate_tool_call_result": make_intermediate_tool_call_result(
                param_queries=[{"origin": "test_origin", "id": "price_feeds"}],
                data=[make_client_error()],
            ),
            "param_options_widget_query_mapping": [{"widget_query_index": 0}],
        }
        assert self._call(client_function_call_service, extra_state) is None
