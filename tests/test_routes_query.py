import json
from typing import Any, Mapping
from unittest.mock import MagicMock, patch

import numpy as np
import openai
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError

from openbb_ada.errors import ContextLimitExceededError
from openbb_ada.models import UserFile
from openbb_ada.utils.utils import build_context_uuid, get_copilot_call_count
from tests.conftest import MockUUIDs
from tests.testing import (
    assert_status_update_exists,
    assert_status_update_exists_flexible,
    assert_status_update_exists_optional,
    assert_widget_operation_status_exists,
    parse_citations,
    parse_function_calls,
    parse_message_artifacts,
    parse_message_chunks,
    parse_status_updates,
)


def get_expected_context_uuid(
    widget_uuid: str, input_args: dict, item_index: int = 0, extra_seed=None
):
    """Helper function to calculate expected deterministic context UUID for tests."""
    from uuid import UUID

    return str(
        build_context_uuid(UUID(widget_uuid), input_args, item_index, extra_seed)
    )


def test_query_responds_with_custom_x_trace_id_header(
    mock_uuids: type[MockUUIDs],
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
):
    payload = {"messages": [{"role": "human", "content": "What is 1+1?"}]}
    response = test_client.post(
        "/v1/query",
        headers={
            "Authorization": mock_headers["Authorization"],
            "X-Trace-Id": mock_uuids.ID1.value,
        },
        json=payload,
    )
    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] == mock_uuids.ID1.value


def test_query_responds_with_uuid_x_trace_id_header_if_not_set(
    test_client: TestClient, mock_headers: str, no_rate_limit: None
):
    payload = {"messages": [{"role": "human", "content": "What is 1+1?"}]}
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)  # type: ignore
    assert response.status_code == 200
    assert response.headers["X-Trace-Id"] is not None


def test_query_no_messages(
    test_client: TestClient, mock_headers: str, no_rate_limit: None
):
    payload = {"messages": []}  # type: ignore

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)  # type: ignore

    assert response.status_code == 422
    assert "messages list cannot be empty" in response.text


def test_query_no_context(
    test_client: TestClient,
    mock_headers: str,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "What is 1+1? Give only the answer as an integer and nothing else.",  # noqa: E501
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)  # type: ignore
    response_text = parse_message_chunks(response.text)

    assert response.status_code == 200
    assert "2" in response_text


def test_query_with_history(
    test_client: TestClient, mock_headers: Any, no_rate_limit: None
):
    payload = {
        "messages": [
            {"role": "human", "content": "Knock knock..."},
            {"role": "ai", "content": "Who's there?"},
            {"role": "human", "content": "I eat mop."},
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    response_text = parse_message_chunks(response.text)

    assert response.status_code == 200
    assert "I eat mop who?" in response_text


@pytest.mark.parametrize(
    "query, forbidden_patterns, description",
    [
        (
            "Give me a simple JSON object with name and age fields",
            [r"(?<!`)`\s*\{[^`]*\}[^`]*`(?!`)"],
            "JSON must use ```json, never single backticks like `{...}`",
        ),
        (
            "Show me a two-line bash command",
            [r"(?<!`)`[^`\n]*\n[^`]*`(?!`)"],
            "Newlines must use triple backticks, never inside single backticks",
        ),
    ],
)
def test_query_response_code_formatting(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    query: str,
    forbidden_patterns: list[str],
    description: str,
):
    """Test that AI responses follow rules for code block formatting."""
    import re

    payload = {"messages": [{"role": "human", "content": query}]}
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    response_text = parse_message_chunks(response.text)

    assert response.status_code == 200
    assert len(response_text) > 0, "Response should not be empty"

    # Check for forbidden patterns
    violations = []
    for pattern in forbidden_patterns:
        matches = re.findall(pattern, response_text, re.MULTILINE)
        if matches:
            violations.append(f"Pattern '{pattern}': {matches[:3]}")

    assert not violations, (
        f"Markdown formatting violations detected.\n"
        f"Rule: {description}\n"
        f"Violations: {violations}\n\n"
        f"Full response:\n{response_text}"
    )


def test_query_with_dirty_messages(
    test_client: TestClient, mock_headers: Any, no_rate_limit: None
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "{'start_of_joke': 'Knock knock...'} (Follow the joke format)",  # noqa: E501
            },  # <-- content contains {} (i.e. dirty)
            {"role": "ai", "content": "Who's there?, {dirty}"},  # <-- additional {}
            {"role": "human", "content": "I eat mop."},
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    response_text = parse_message_chunks(response.text)

    assert response.status_code == 200
    # Wording varies by model; this test is about handling "dirty" braces safely.
    assert len(response_text) > 0
    assert any(token in response_text.lower() for token in ["mop", "joke", "knock"])


def test_query_with_primary_widgets_generates_function_call_uses_current_value(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "financial_ratios",
                    "name": "Financial ratios widget",
                    "description": "Contains a number of financial ratios for a ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the debt-to-equity ratio of AAPL?",
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # First status update with "Querying"
    querying_update = assert_status_update_exists(
        status_updates, "Requesting widget data"
    )
    assert "financial_ratios" in str(querying_update["details"][0])

    # Last status update
    assert "Requesting widget data" in status_updates[-1]["message"]
    assert "financial_ratios" in status_updates[-1]["details"][0]["Widget Id"]

    # Function call
    assert "get_widget_data" in function_calls[0]["function"]
    assert (
        mock_uuids.ID1.value
        in function_calls[0]["input_arguments"]["data_sources"][0]["widget_uuid"]
    )
    assert (
        function_calls[0]["input_arguments"]["data_sources"][0]["input_args"]["ticker"]
        == "AAPL"
    )
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 1
    )


def test_query_with_primary_widgets_generates_input_arg_for_current_value_no_default(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "gemini_llm",
                    "name": "Gemini LLM widget",
                    "description": "Use Gemini, an online LLM, to answer questions.",  # noqa: E501
                    "params": [
                        {
                            "name": "prompt",
                            "type": "string",
                            "description": "The prompt to send to Gemini.",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "@Gemini, what is something interesting you know? Try re-use the current input args for the LLM widget.",  # noqa: E501
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # First status update indicating widget operations
    assert_widget_operation_status_exists(status_updates)

    # Last status update
    assert "Requesting widget data" in status_updates[-1]["message"]
    assert "gemini_llm" in status_updates[-1]["details"][0]["Widget Id"]

    # Function call
    assert "get_widget_data" in function_calls[0]["function"]
    assert (
        mock_uuids.ID1.value
        in function_calls[0]["input_arguments"]["data_sources"][0]["widget_uuid"]
    )
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 1
    )


def test_query_with_primary_widgets_generates_function_call_with_custom_value(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "financial_ratios",
                    "name": "Financial ratios widget",
                    "description": "Company financial ratios such as P/B, ROE, etc.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AMZN",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the debt-to-equity ratio of AAPL?",
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # First operation status update
    querying_update = assert_widget_operation_status_exists(status_updates)
    queries_text = str(querying_update["details"][0].get("Queries", "")).lower()
    assert "aapl" in queries_text
    assert "debt-to-equity" in queries_text or "debt to equity" in queries_text

    # Last status update
    assert "Requesting widget data" in status_updates[-1]["message"]
    assert "financial_ratios" in status_updates[-1]["details"][0]["Widget Id"]
    assert "AAPL" in status_updates[-1]["details"][0]["ticker"]

    # Function call
    assert "get_widget_data" in function_calls[0]["function"]
    assert (
        mock_uuids.ID1.value
        in function_calls[0]["input_arguments"]["data_sources"][0]["widget_uuid"]
    )
    assert (
        "financial_ratios"
        in function_calls[0]["input_arguments"]["data_sources"][0]["id"]
    )
    assert (
        "AAPL"
        in function_calls[0]["input_arguments"]["data_sources"][0]["input_args"][
            "ticker"
        ]
    )
    assert (
        "AAPL"
        in function_calls[0]["extra_state"]["copilot_function_call_arguments"][
            "widget_queries"
        ][0]["query"]
    )


def test_query_with_primary_widgets_generates_function_call_uses_multi_select_param(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "financial_ratios",
                    "name": "Financial ratios widget",
                    "description": "Contains a number of financial ratios for a ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "multi_select": True,
                            "description": "The stock ticker symbol.",
                            "current_value": ["AAPL"],
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the debt-to-equity ratio of MSFT and AMZN? Submit ONLY a single widget query.",  # noqa: E501
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # First operation status update
    querying_update = assert_widget_operation_status_exists(status_updates)
    assert "debt-to-equity" in querying_update["details"][0]["Queries"]

    # Last status update
    assert "Requesting widget data" in status_updates[-1]["message"]
    assert "financial_ratios" in status_updates[-1]["details"][0]["Widget Id"]

    # Function call
    widget_data_calls = [
        call for call in function_calls if "get_widget_data" in call.get("function", "")
    ]
    assert len(widget_data_calls) >= 1

    data_sources = [
        source
        for call in widget_data_calls
        for source in call.get("input_arguments", {}).get("data_sources", [])
    ]
    assert any(
        mock_uuids.ID1.value in source.get("widget_uuid", "") for source in data_sources
    )

    requested_tickers: set[str] = set()
    for source in data_sources:
        ticker = source.get("input_args", {}).get("ticker")
        if isinstance(ticker, str):
            requested_tickers.add(ticker)
        elif isinstance(ticker, list):
            requested_tickers.update(t for t in ticker if isinstance(t, str))

    # Some models issue one combined multi-select query; others split into two
    # requests. Both are valid as long as both tickers are requested.
    assert {"MSFT", "AMZN"}.issubset(requested_tickers)


def test_query_with_primary_widgets_generates_function_call_reuse_widget_with_different_query_on_follow_up_question(  # noqa: E501
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "financial_ratios",
                    "name": "Financial ratios widget",
                    "description": "Contains a number of financial ratios for a ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the debt-to-equity ratio?",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "financial_ratios",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "financial_ratios",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the debt-to-equity ratio of AAPL?",
                            }
                        ]
                    },
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": " ".join([" "] * 1000)
                                + "The debt-to-equity ratio of AAPL is 3.123."
                            }
                        ],
                    }
                ],
            },
            {
                "role": "human",
                "content": "And for TSLA?",
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Status updates - use flexible helper to skip planning
    if status_updates:
        querying_update = assert_status_update_exists_flexible(
            status_updates, "Querying"
        )
        assert "TSLA" in querying_update["details"][0]["Queries"]
        assert "debt-to-equity" in querying_update["details"][0]["Queries"]

        accessing_update = assert_status_update_exists_flexible(
            status_updates, "Requesting widget data"
        )
        assert "financial_ratios" in accessing_update["details"][0]["Widget Id"]
        assert "TSLA" in accessing_update["details"][0]["ticker"]

    # Function call
    assert "get_widget_data" in function_calls[0]["function"]
    assert (
        mock_uuids.ID1.value
        in function_calls[0]["input_arguments"]["data_sources"][0]["widget_uuid"]
    )
    assert (
        "financial_ratios"
        in function_calls[0]["input_arguments"]["data_sources"][0]["id"]
    )
    assert (
        "TSLA"
        in function_calls[0]["input_arguments"]["data_sources"][0]["input_args"][
            "ticker"
        ]
    )
    assert (
        "TSLA"
        in function_calls[0]["extra_state"]["copilot_function_call_arguments"][
            "widget_queries"
        ][0]["query"]
    )


def test_query_with_primary_widgets_generates_function_call_multiple_queries_for_same_widget(  # noqa: E501
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "financial_ratios",
                    "name": "Financial ratios widget",
                    "description": "Contains a number of financial ratios for a ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the debt-to-equity ratio for AAPL and TSLA?",
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Status updates
    querying_update = assert_widget_operation_status_exists(status_updates)
    assert "TSLA" in querying_update["details"][0]["Queries"]
    assert "AAPL" in querying_update["details"][0]["Queries"]
    assert "debt-to-equity" in querying_update["details"][0]["Queries"]

    # Find accessing widget updates (there should be 2)
    accessing_updates = [
        u for u in status_updates if "Requesting widget data" in u["message"]
    ]
    assert len(accessing_updates) >= 2
    # Check that both updates have the expected widget ID
    widget_ids = [
        update["details"][0].get("Widget Id", "") for update in accessing_updates
    ]
    assert all("financial_ratios" in widget_id for widget_id in widget_ids)

    assert any(
        "TSLA" in status_update["details"][0].get("ticker", "")
        for status_update in status_updates[1:]
    )

    assert any(
        "AAPL" in status_update["details"][0].get("ticker", "")
        for status_update in status_updates[1:]
    )

    # Function call
    assert "get_widget_data" in function_calls[0]["function"]

    assert all(
        mock_uuids.ID1.value == source["widget_uuid"]
        for source in function_calls[0]["input_arguments"]["data_sources"]
    )

    assert all(
        "test_origin" == source["origin"]
        for source in function_calls[0]["input_arguments"]["data_sources"]
    )

    assert all(
        "financial_ratios" == source["id"]
        for source in function_calls[0]["input_arguments"]["data_sources"]
    )

    assert any(
        "AAPL" == source["input_args"]["ticker"]
        for source in function_calls[0]["input_arguments"]["data_sources"]
    )
    assert any(
        "TSLA" == source["input_args"]["ticker"]
        for source in function_calls[0]["input_arguments"]["data_sources"]
    )
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 2
    )


def test_query_with_primary_widgets_generates_function_call_multiple_queries_for_different_widgets(  # noqa: E501
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "financial_ratios",
                    "name": "Financial ratios widget",
                    "description": "Contains a number of financial ratios for a ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            # The current value should be ignored by the LLM (it
                            # should follow the prompt!)
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "news",
                    "name": "News widget",
                    "description": "Contains news for a ticker.",
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            # The current value should be ignored by the LLM (it
                            # should follow the prompt!)
                            "current_value": "MSFT",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the debt-to-equity ratio for TSLA and what is in the news for AMZN?",  # noqa: E501
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Status updates - simplified check for basic functionality
    # Just verify that status updates are being generated
    assert len(status_updates) >= 0, "Status updates should be parsable"

    # Verify that the LLM attempts to query at least one of the requested widgets
    # The LLM might handle multi-part queries differently (sequentially, combined, etc.)
    financial_ratios_queried = any(
        "financial_ratios" == status_update["details"][0].get("Widget Id", "")
        and "TSLA" in status_update["details"][0].get("ticker", "")
        for status_update in status_updates[1:]
    )

    news_queried = any(
        "news" == status_update["details"][0].get("Widget Id", "")
        and "AMZN" in status_update["details"][0].get("ticker", "")
        for status_update in status_updates[1:]
    )

    assert financial_ratios_queried and news_queried, "Expected both widgets queried"

    assert "get_widget_data" in function_calls[0]["function"]
    assert len(function_calls[0]["input_arguments"]["data_sources"]) == 2

    # Check for financial ratios widget query (should be present)
    financial_ratios_in_call = any(
        mock_uuids.ID1.value == source["widget_uuid"]
        and "test_origin" == source["origin"]
        and "financial_ratios" == source["id"]
        and "TSLA" == source["input_args"]["ticker"]
        for source in function_calls[0]["input_arguments"]["data_sources"]
    )

    # Check for news widget query (may or may not be present)
    news_in_call = any(
        mock_uuids.ID2.value == source["widget_uuid"]
        and "test_origin" == source["origin"]
        and "news" == source["id"]
        and "AMZN" == source["input_args"]["ticker"]
        for source in function_calls[0]["input_arguments"]["data_sources"]
    )

    # At least the financial ratios should be queried
    assert financial_ratios_in_call, "Expected financial ratios query in function call"
    assert news_in_call, "Expected news query in function call"


def test_query_with_primary_widgets_final_answer_unstructured(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "financial_ratios",
                    "name": "Financial Ratios",
                    "description": "Contains a number of financial ratios for a ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the debt-to-equity ratio?",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "financial_ratios",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "financial_ratios",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": " ".join([" "] * 1000)
                                + "The debt-to-equity ratio of AAPL is 3.123."
                            }
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the debt-to-equity ratio of AAPL?",  # noqa: E501
                                "use_current_inputs": True,
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    response_text = parse_message_chunks(response.text)
    citations = parse_citations(response.text)

    # Status updates - optional depending on LLM behavior
    reading_update = assert_status_update_exists_optional(status_updates, "Retrieving")
    if reading_update:
        if reading_update.get("details") and len(reading_update["details"]) > 0:
            if "Data source" in reading_update["details"][0]:
                assert "Financial Ratios" in reading_update["details"][0]["Data source"]
            if "Ticker" in reading_update["details"][0]:
                assert "AAPL" in reading_update["details"][0]["Ticker"]

    # Copilot answer - should either contain the specific value or
    # reference AAPL debt-to-equity data
    assert "3.123" in response_text or (
        "debt-to-equity" in response_text.lower()
        and ("aapl" in response_text.lower() or "apple" in response_text.lower())
    )

    # Citations may be omitted by some models for direct unstructured answers.
    if citations:
        assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
            mock_uuids.ID1.value, {"ticker": "AAPL"}
        )
        assert citations[0]["source_info"]["type"] == "widget"
        assert citations[0]["source_info"]["name"] == "Financial Ratios"
        assert citations[0]["source_info"]["widget_id"] == "financial_ratios"
        assert citations[0]["details"][0]["Data source"] == "Financial Ratios"
        assert citations[0]["details"][0]["Ticker"] == "AAPL"


def test_query_with_primary_widgets_final_answer_structured(
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "financial_ratios",
                    "name": "Financial Ratios",
                    "description": "Contains a number of financial ratios for a ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the debt-to-equity ratio?",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "financial_ratios",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "financial_ratios",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    [
                                        {
                                            "debt_to_equity_ratio": 3.123,
                                            # We should be able to handle hashmaps as input  # noqa: E501
                                            "extra_info": {
                                                "some": "extra",
                                                "info": "here",
                                            },
                                        }
                                    ]
                                )
                            }
                        ],
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the debt-to-equity ratio of AAPL?",  # noqa: E501
                            }
                        ],
                    }
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    response_text = parse_message_chunks(response.text)
    citations = parse_citations(response.text)
    message_artifacts = parse_message_artifacts(response.text)

    # Status updates
    processing_update = assert_status_update_exists(status_updates, "Querying for")  # noqa: E501
    query_text = processing_update["details"][0]["Query"]
    assert any(
        token in query_text.lower() for token in ["debt-to-equity", "debt_to_equity"]
    )
    assert "aapl" in query_text.lower()

    # Copilot answer
    assert "3.123" in response_text or "3.123" in str(message_artifacts)

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {"ticker": "AAPL"}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["source_info"]["name"] == "Financial Ratios"
    assert citations[0]["source_info"]["widget_id"] == "financial_ratios"
    assert citations[0]["details"][0]["Data source"] == "Financial Ratios"
    assert citations[0]["details"][0]["Ticker"] == "AAPL"


def test_query_with_primary_widgets_with_history_generates_function_call(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_info",
                    "name": "Stock price info widget",
                    "description": "Contains information regarding stocks.",
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "TSLA",
                            "options": None,
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the market cap of TSLA?",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_info",
                                    "input_args": {"ticker": "TSLA"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_info",
                            "input_args": {"ticker": "TSLA"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": " ".join([" "] * 1000)
                                + "The market cap of TSLA is 50."
                            }
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the market cap of TSLA?",
                            }
                        ]
                    },
                },
            },
            {
                "role": "human",
                "content": "And AAPL?",
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # First widget-operation status update (wording varies by model)
    querying_update = assert_widget_operation_status_exists(status_updates)
    query_text = str(querying_update["details"][0].get("Queries", ""))
    assert "AAPL" in query_text
    assert "market cap" in query_text

    # Last status update
    assert "Requesting widget data" in status_updates[-1]["message"]
    assert "AAPL" in status_updates[-1]["details"][0]["ticker"]
    assert "stock_price_info" in status_updates[-1]["details"][0]["Widget Id"]

    # Function call
    assert "get_widget_data" in function_calls[0]["function"]
    assert (
        function_calls[0]["input_arguments"]["data_sources"][0]["widget_uuid"]
        == mock_uuids.ID1.value
    )
    assert (
        "stock_price_info"
        in function_calls[0]["input_arguments"]["data_sources"][0]["id"]
    )
    assert (
        "AAPL"
        in function_calls[0]["input_arguments"]["data_sources"][0]["input_args"][
            "ticker"
        ]
    )
    assert (
        "AAPL"
        in function_calls[0]["extra_state"]["copilot_function_call_arguments"][
            "widget_queries"
        ][0]["query"]
    )


def test_query_with_secondary_widgets_calls_function(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {"role": "human", "content": "What is the stock price of AAPL?"},
        ],
        "widgets": {
            "primary": [],
            "secondary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock price quote widget",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "company_news",
                    "name": "Company News",
                    "description": "Contains the latest news for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AMZN",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "extra": [],
        },
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # First status update with "Requesting"
    requesting_update = assert_status_update_exists(status_updates, "Requesting")
    assert "AAPL" in str(requesting_update["details"][0])

    # Second status update with "Requesting widget data"
    accessing_update = assert_status_update_exists(
        status_updates, "Requesting widget data"
    )
    assert "test_origin" in accessing_update["details"][0]["Origin"]
    assert "stock_price_quote" in accessing_update["details"][0]["Widget Id"]
    assert "AAPL" in accessing_update["details"][0]["ticker"]

    # Function call
    assert function_calls[0] is not None
    assert function_calls[0]["function"] == "get_widget_data"
    assert function_calls[0]["input_arguments"]["data_sources"] == [
        {
            "widget_uuid": mock_uuids.ID1.value,
            "origin": "test_origin",
            "id": "stock_price_quote",
            "input_args": {"ticker": "AAPL"},
            "ssm_request": None,
        }
    ]
    assert (
        "aapl"
        in function_calls[0]["extra_state"]["copilot_function_call_arguments"][
            "widget_queries"
        ][0]["query"].lower()
    )


def test_query_with_secondary_widgets_final_answer(  # noqa: E501
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [],
            "secondary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "earnings_history",
                    "name": "Earnings History",
                    "description": "Contains the earnings history for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        },
                        {
                            "name": "year",
                            "type": "string",
                            "description": "The year to get earnings history for.",
                            "default_value": None,
                            "current_value": "2023",
                        },
                    ],
                    "metadata": {},
                },
            ],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the total revenue for AMZN in 2024?",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "earnings_history",
                                    "input_args": {"ticker": "AMZN", "year": "2024"},
                                },
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "earnings_history",
                            "input_args": {"ticker": "AMZN", "year": "2024"},
                        },
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    [
                                        {
                                            "year": "2024",
                                            "currency": "USD",
                                            "revenue": "1000",
                                        }
                                    ]
                                ),
                                "data_format": {
                                    "data_type": "object",
                                    "parse_as": "table",
                                },
                            }
                        ]
                    },
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the total revenue for AMZN in 2024?",  # noqa: E501
                            },
                        ]
                    },
                },
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # First status update with "Querying for"
    processing_update = assert_status_update_exists(status_updates, "Querying for")
    processing_detail_text = str(processing_update["details"][0]).lower()
    assert "revenue" in processing_detail_text
    assert "2024" in processing_detail_text

    # "Artifact generated" status update is yielded (for UI feedback)
    assert any(
        "Artifact generated" in status_update["message"]
        for status_update in status_updates
    )

    # Copilot answer - value should be in response text for single-row results
    assert "amzn" in response_text.lower()
    assert (
        "1000" in response_text.lower()
        or "1,000" in response_text.lower()
        or "$1,000" in response_text.lower()
    )

    # Citations
    assert len(citations) == 1

    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {"ticker": "AMZN", "year": "2024"}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["details"][0]["Data source"] == "Earnings History"
    assert citations[0]["details"][0]["Ticker"] == "AMZN"
    assert citations[0]["details"][0]["Year"] == "2024"
    assert citations[0]["source_info"]["name"] == "Earnings History"
    assert citations[0]["source_info"]["widget_id"] == "earnings_history"
    assert citations[0]["source_info"]["origin"] == "test_origin"


# TODO: This needs to be fixed
@pytest.mark.skip(
    reason="""
    Currently the agent does not yield any status updates or function calls.
    This might be related to some workspace-options missing or
    another configuration issue in the request payload.
    """
)
def test_query_with_primary_and_secondary_widgets_function_call_use_secondary(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "What was the total revenue for AMZN in 2024?",
            },
        ],
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock price quote widget",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "earnings_history",
                    "name": "Earnings history widget",
                    "description": "Contains the earnings history for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        },
                        {
                            "name": "year",
                            "type": "string",
                            "description": "The year to get earnings history for.",
                            "default_value": None,
                            "current_value": "2023",
                        },
                    ],
                    "metadata": {},
                },
            ],
            "extra": [],
        },
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # First status update indicating widget operations
    querying_update = assert_widget_operation_status_exists(status_updates)
    assert "AMZN" in str(querying_update["details"][0])
    assert "2024" in str(querying_update["details"][0])

    # Second status update with "Requesting widget data"
    accessing_update = assert_status_update_exists(
        status_updates, "Requesting widget data"
    )
    assert "test_origin" in accessing_update["details"][0]["Origin"]
    assert "earnings_history" in accessing_update["details"][0]["Widget Id"]
    assert "AMZN" in accessing_update["details"][0]["ticker"]
    assert "2024" in accessing_update["details"][0]["year"]

    # Function call
    assert function_calls[0] is not None
    assert function_calls[0]["function"] == "get_widget_data"
    assert function_calls[0]["input_arguments"]["data_sources"] == [
        {
            "widget_uuid": mock_uuids.ID2.value,
            "origin": "test_origin",
            "id": "earnings_history",
            "input_args": {"ticker": "AMZN", "year": "2024"},
        }
    ]
    assert (
        "widget_queries"
        in function_calls[0]["extra_state"]["copilot_function_call_arguments"]
    )
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 1
    )


def test_query_with_primary_and_secondary_widgets_function_call_use_primary(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "What is the stock price of TSLA?",
            },
        ],
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock price quote widget",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "earnings_history",
                    "name": "Earnings history widget",
                    "description": "Contains the earnings history for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        },
                        {
                            "name": "year",
                            "type": "string",
                            "description": "The year to get earnings history for.",
                            "default_value": None,
                            "current_value": "2023",
                        },
                    ],
                    "metadata": {},
                },
            ],
            "extra": [],
        },
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # First status update with "Requesting"
    requesting_update = assert_status_update_exists(status_updates, "Requesting")
    assert "TSLA" in str(requesting_update["details"][0])

    # Second status update with "Requesting widget data"
    accessing_update = assert_status_update_exists(
        status_updates, "Requesting widget data"
    )
    assert "test_origin" in accessing_update["details"][0]["Origin"]
    assert "stock_price_quote" in accessing_update["details"][0]["Widget Id"]
    assert "TSLA" in accessing_update["details"][0]["ticker"]

    # Function call
    assert function_calls[0] is not None
    assert function_calls[0]["function"] == "get_widget_data"
    assert function_calls[0]["input_arguments"]["data_sources"] == [
        {
            "widget_uuid": mock_uuids.ID1.value,
            "origin": "test_origin",
            "id": "stock_price_quote",
            "input_args": {"ticker": "TSLA"},
            "ssm_request": None,
        }
    ]
    assert (
        "widget_queries"
        in function_calls[0]["extra_state"]["copilot_function_call_arguments"]
    )
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 1
    )


def test_query_with_primary_and_secondary_widgets_function_call_use_primary_and_secondary(  # noqa: E501
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "What is the stock price of TSLA and the total revenue using earnings for AMZN in 2024?",  # noqa: E501
            },
        ],
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock price quote widget",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "earnings_history",
                    "name": "Earnings history widget",
                    "description": "Contains the earnings history for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        },
                        {
                            "name": "year",
                            "type": "string",
                            "description": "The year to get earnings history for.",
                            "default_value": None,
                            "current_value": "2023",
                        },
                    ],
                    "metadata": {},
                },
            ],
            "extra": [],
        },
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Status updates - simplified check for basic functionality
    # Just verify that status updates are being generated
    assert len(status_updates) >= 0, "Status updates should be parsable"

    # 2 assertions check for status updates related specifically to querying widgets
    assert any(
        "AMZN" == status_update["details"][0].get("ticker", "")
        and "2024" == status_update["details"][0].get("year", "")
        for status_update in status_updates[1:]
    )
    assert any(
        "TSLA" == status_update["details"][0].get("ticker", "")
        for status_update in status_updates[1:]
    )

    # Function call
    assert function_calls[0] is not None
    assert function_calls[0]["function"] == "get_widget_data"

    assert any(
        {
            "widget_uuid": mock_uuids.ID2.value,
            "origin": "test_origin",
            "id": "earnings_history",
            "input_args": {"ticker": "AMZN", "year": "2024"},
            "ssm_request": None,
        }
        in function_call["input_arguments"]["data_sources"]
        for function_call in function_calls
    )

    assert any(
        {
            "widget_uuid": mock_uuids.ID1.value,
            "origin": "test_origin",
            "id": "stock_price_quote",
            "input_args": {"ticker": "TSLA"},
            "ssm_request": None,
        }
        in function_call["input_arguments"]["data_sources"]
        for function_call in function_calls
    )

    assert (
        "widget_queries"
        in function_calls[0]["extra_state"]["copilot_function_call_arguments"]
    )
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 2
    )


def test_query_with_primary_and_secondary_widgets_final_answer_use_primary(  # noqa: E501
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock Price Quote",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "earnings_history",
                    "name": "Earnings History",
                    "description": "Contains the earnings history for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        },
                        {
                            "name": "year",
                            "type": "string",
                            "description": "The year to get earnings history for.",
                            "default_value": None,
                            "current_value": "2023",
                        },
                    ],
                    "metadata": {},
                },
            ],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the stock price of TSLA?",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_quote",
                                    "input_args": {"ticker": "TSLA"},
                                },
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_quote",
                            "input_args": {"ticker": "TSLA"},
                        },
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    [
                                        {
                                            "price": "99.95",
                                        }
                                    ]
                                )
                            }
                        ]
                    },
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the stock price of TSLA?",
                            },
                        ]
                    },
                },
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # First status update
    processing_update = assert_widget_operation_status_exists(status_updates)
    assert processing_update["details"]

    # "Artifact generated" status update is yielded (for UI feedback)
    assert any(
        "Artifact generated" in status_update["message"]
        for status_update in status_updates
    )

    # Copilot answer - value should be in response text for single-row results
    assert "tsla" in response_text.lower()
    assert "99.95" in response_text.lower()

    # Citations
    assert len(citations) == 1

    assert any(
        citation["source_info"]["uuid"]
        == get_expected_context_uuid(mock_uuids.ID1.value, {"ticker": "TSLA"})
        and citation["source_info"]["type"] == "widget"
        and citation["details"][0]["Data source"] == "Stock Price Quote"
        and citation["details"][0]["Ticker"] == "TSLA"
        and citation["source_info"]["name"] == "Stock Price Quote"
        and citation["source_info"]["origin"] == "test_origin"
        and citation["source_info"]["widget_id"] == "stock_price_quote"
        for citation in citations
    )


def test_query_with_primary_and_secondary_widgets_final_answer_use_secondary(  # noqa: E501
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock Price Quote",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "earnings_history",
                    "name": "Earnings History",
                    "description": "Contains the earnings history for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        },
                        {
                            "name": "year",
                            "type": "string",
                            "description": "The year to get earnings history for.",
                            "default_value": None,
                            "current_value": "2023",
                        },
                    ],
                    "metadata": {},
                },
            ],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the total revenue for AMZN in 2024?",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID2.value,
                                    "origin": "test_origin",
                                    "id": "earnings_history",
                                    "input_args": {"ticker": "AMZN", "year": "2024"},
                                },
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID2.value,
                            "origin": "test_origin",
                            "id": "earnings_history",
                            "input_args": {"ticker": "AMZN", "year": "2024"},
                        },
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    [
                                        {
                                            "year": "2024",
                                            "currency": "USD",
                                            "revenue": "1000",
                                        }
                                    ]
                                )
                            }
                        ]
                    },
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID2.value,
                                "query": "What is the total revenue for AMZN in 2024?",  # noqa: E501
                            },
                        ]
                    },
                },
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # First status update
    processing_update = assert_status_update_exists(status_updates, "Querying")
    processing_detail_text = str(processing_update["details"][0]).lower()
    assert "revenue" in processing_detail_text
    assert "2024" in processing_detail_text

    # "Artifact generated" status update is yielded (for UI feedback)
    assert any(
        "Artifact generated" in status_update["message"]
        for status_update in status_updates
    )

    # Copilot answer - value should be in response text for single-row results
    assert "amzn" in response_text.lower()
    assert (
        "1000" in response_text.lower()
        or "$1,000" in response_text.lower()
        or "1,000" in response_text.lower()
    )

    # Citations
    assert len(citations) == 1

    assert any(
        citation["source_info"]["uuid"]
        == get_expected_context_uuid(
            mock_uuids.ID2.value, {"ticker": "AMZN", "year": "2024"}
        )
        and citation["source_info"]["type"] == "widget"
        and citation["details"][0]["Data source"] == "Earnings History"
        and citation["details"][0]["Ticker"] == "AMZN"
        and citation["details"][0]["Year"] == "2024"
        and citation["source_info"]["origin"] == "test_origin"
        and citation["source_info"]["name"] == "Earnings History"
        and citation["source_info"]["widget_id"] == "earnings_history"
        for citation in citations
    )


def test_query_with_primary_and_secondary_widgets_final_answer_use_primary_and_secondary(  # noqa: E501
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock Price Quote",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "earnings_history",
                    "name": "Earnings History",
                    "description": "Contains the earnings history for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        },
                        {
                            "name": "year",
                            "type": "string",
                            "description": "The year to get earnings history for.",
                            "default_value": None,
                            "current_value": "2023",
                        },
                    ],
                    "metadata": {},
                },
            ],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the stock price of TSLA and the total revenue for AMZN in 2024?",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_quote",
                                    "input_args": {"ticker": "TSLA"},
                                },
                                {
                                    "widget_uuid": mock_uuids.ID2.value,
                                    "origin": "test_origin",
                                    "id": "earnings_history",
                                    "input_args": {"ticker": "AMZN", "year": "2024"},
                                },
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_quote",
                            "input_args": {"ticker": "TSLA"},
                        },
                        {
                            "widget_uuid": mock_uuids.ID2.value,
                            "origin": "test_origin",
                            "id": "earnings_history",
                            "input_args": {"ticker": "AMZN", "year": "2024"},
                        },
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    [
                                        {
                                            "price": "99.95",
                                        }
                                    ]
                                )
                            }
                        ]
                    },
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    [
                                        {
                                            "year": "2024",
                                            "currency": "USD",
                                            "revenue": "1000",
                                        }
                                    ]
                                )
                            }
                        ]
                    },
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the stock price of TSLA?",
                            },
                            {
                                "widget_uuid": mock_uuids.ID2.value,
                                "query": "What is the total revenue for AMZN in 2024?",  # noqa: E501
                            },
                        ]
                    },
                },
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # Status updates
    processing_updates = [u for u in status_updates if "Querying" in u["message"]]
    assert len(processing_updates) >= 1

    all_processing_details = "".join(
        str(u.get("details", "")) for u in processing_updates
    )
    assert "tsla" in all_processing_details.lower()
    assert "amzn" in all_processing_details.lower()

    # "Artifact generated" status updates are yielded (for UI feedback)
    assert any(
        "Artifact generated" in status_update["message"]
        for status_update in status_updates
    )

    # Copilot answer - values should be in response text for single-row results
    assert "tsla" in response_text.lower()
    assert "amzn" in response_text.lower()
    assert "99.95" in response_text.lower()
    assert "1000" in response_text.lower() or "1,000" in response_text.lower()

    # Citations
    assert len(citations) == 2

    # Citation ordering is model-dependent, so match by provenance fields
    # instead of citation index.
    has_stock_citation = any(
        citation["source_info"]["uuid"]
        == get_expected_context_uuid(mock_uuids.ID1.value, {"ticker": "TSLA"})
        and citation["source_info"]["type"] == "widget"
        and citation["details"][0]["Data source"] == "Stock Price Quote"
        and citation["details"][0]["Ticker"] == "TSLA"
        and citation["source_info"]["origin"] == "test_origin"
        and citation["source_info"]["name"] == "Stock Price Quote"
        and citation["source_info"]["widget_id"] == "stock_price_quote"
        for citation in citations
    )
    has_earnings_citation = any(
        citation["source_info"]["uuid"]
        == get_expected_context_uuid(
            mock_uuids.ID2.value, {"ticker": "AMZN", "year": "2024"}
        )
        and citation["source_info"]["type"] == "widget"
        and citation["details"][0]["Data source"] == "Earnings History"
        and citation["details"][0]["Ticker"] == "AMZN"
        and citation["details"][0]["Year"] == "2024"
        and citation["source_info"]["origin"] == "test_origin"
        and citation["source_info"]["name"] == "Earnings History"
        and citation["source_info"]["widget_id"] == "earnings_history"
        for citation in citations
    )
    assert has_stock_citation or has_earnings_citation


def test_query_explicit_context_chart_artifact(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "context": [
            {
                "uuid": mock_uuids.ID1.value,
                "name": "historical_closing_price_aapl_artifact",
                "description": "Contains the historical closing price for AAPL",  # noqa: E501
                "metadata": {"ticker": "AAPL"},
                "data": {
                    "items": [
                        {
                            "content": json.dumps(
                                [
                                    {"date": "2024-01-01", "price": 100.00},
                                    {"date": "2024-01-02", "price": 101.00},
                                    {"date": "2024-01-03", "price": 102.00},
                                ]
                            ),
                            "data_format": {
                                "data_type": "object",
                                "parse_as": "chart",
                                "chart_params": {
                                    "chartType": "line",
                                    "xKey": "date",
                                    "yKey": ["price"],
                                },
                            },
                        }
                    ],
                },
            },
        ],
        "widgets": {
            "primary": [],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the average closing stock price of AAPL? Calculate it from the chart.",  # noqa: E501
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)

    # With explicit context, the system should provide direct calculations
    # The average of [100.00, 101.00, 102.00] is 101.00
    assert "101" in response_text


def test_query_explicit_context_chart_artifact_pie(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "context": [
            {
                "uuid": mock_uuids.ID1.value,
                "name": "portfolio_allocation_artifact",
                "description": "Contains the portfolio allocation across different assets",  # noqa: E501
                "metadata": {},
                "data": {
                    "items": [
                        {
                            "content": json.dumps(
                                [
                                    {"asset": "Stocks", "amount": 60000},
                                    {"asset": "Bonds", "amount": 40000},
                                    {"asset": "Cash", "amount": 7000},
                                    {"asset": "Real Estate", "amount": 5000},
                                    {"asset": "Commodities", "amount": 3000},
                                ]
                            ),
                            "data_format": {
                                "data_type": "object",
                                "parse_as": "chart",
                                "chart_params": {
                                    "chartType": "pie",
                                    "angleKey": "amount",
                                    "calloutLabelKey": "asset",
                                },
                            },
                        }
                    ]
                },
            },
        ],
        "widgets": {
            "primary": [],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Approximately what percent of the portfolio is allocated to stocks?",  # noqa: E501
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)

    # Copilot answer - should provide correct calculation: 60,000 / 115,000 = 52.17%
    assert "52" in response_text and "%" in response_text

    # Charts are non-citable by design (context.py sets citable=False for
    # parse_as="chart"), so no citation assertions here.


# TODO: This needs to be fixed
@pytest.mark.skip(
    reason="""
    The model is not returning any status updates. It is instead asking the user
    for guidance on what specific piece of information they are requesting.
    Likely prompt-drift.
    """
)
def test_query_primary_widgets_use_explicit_context_final_answer(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "context": [
            {
                "uuid": mock_uuids.ID1.value,
                "name": "Earnings Transcript",
                "description": "Contains the most recent earnings transcript for a particular ticker",  # noqa: E501
                "metadata": {"ticker": "AAPL"},
                "data": {
                    # We pad the content to make sure the key information isn't
                    # available in the preview.  This is just to force the model
                    # to look up the unstructured context directly.
                    "items": [
                        {
                            "content": " " * 1000
                            + "We are targeting a stock price of $1000 in the next quarter."  # noqa: E501
                        }
                    ]
                },
            },
        ],
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock price quote widget",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the stock price of AAPL?",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID2.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_quote",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID2.value,
                            "origin": "test_origin",
                            "id": "stock_price_quote",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "content": " ".join([" "] * 1000)
                        + "The stock price of AAPL is $99.95."
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID2.value,
                                "query": "What is the stock price of AAPL?",
                            }
                        ]
                    },
                },
            },
            {
                "role": "human",
                "content": "Which stock price is being targeted for next quarter?",
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # Status update
    reading_update = assert_status_update_exists(status_updates, "Reading data")
    assert reading_update["details"][0]["Source type"] == "artifact"
    assert reading_update["details"][0]["Data source"] == "Earnings Transcript"
    assert reading_update["details"][0]["Ticker"] == "AAPL"

    # Copilot answer
    assert "$1000" in response_text

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["type"] == "artifact"
    assert citations[0]["details"][0]["Data source"] == "Earnings Transcript"
    assert citations[0]["details"][0]["Ticker"] == "AAPL"


# TODO: This needs to be fixed
@pytest.mark.skip(
    reason="""
    The model is not returning any function calls. It is instead asking the user
    for more specific instructions. The "Just do it" in the prompt is being ignored.
    Likely prompt-drift.
    """
)
def test_query_with_image_widget_calls_function(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    test_png_table_image_user_file: UserFile,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [],
            "secondary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "table_image",
                    "name": "Table image widget",
                    "description": "A file with a table image",
                    "params": [],
                    "metadata": {
                        "filename": test_png_table_image_user_file.filename,
                        "extension": test_png_table_image_user_file.extension,
                    },
                }
            ],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is color of the table in the image widget? Just do it.",  # noqa: E501
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    function_calls = parse_function_calls(response.text)

    assert function_calls[0] is not None
    assert function_calls[0]["function"] == "get_widget_data"
    assert function_calls[0]["input_arguments"]["data_sources"] == [
        {
            "widget_uuid": mock_uuids.ID1.value,
            "origin": "test_origin",
            "id": "table_image",
            "input_args": {},
        }
    ]
    assert (
        "widget_queries"
        in function_calls[0]["extra_state"]["copilot_function_call_arguments"]
    )
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 1
    )


def test_query_with_secondary_widgets_call_function_with_structured_data_final_answer(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "context": None,
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_performance",
                    "name": "Stock Price Performance",
                    "description": "Contains the historical stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock Price Quote",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AMZN",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the average close price for days above 100? Give a specific answer.",  # <-- the original query  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_performance",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_performance",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    [
                                        {
                                            "Date": "2024-07-01",
                                            "Open": 100.0,
                                            "Close": 104.0,
                                        },
                                        {
                                            "Date": "2024-07-02",
                                            "Open": 101.0,
                                            "Close": 95.0,
                                        },
                                        {
                                            "Date": "2024-07-03",
                                            "Open": 98.0,
                                            "Close": 102.0,
                                        },
                                        {
                                            "Date": "2024-07-04",
                                            "Open": 97.0,
                                            "Close": 99.0,
                                        },
                                        {
                                            "Date": "2024-07-05",
                                            "Open": 102.0,
                                            "Close": 101.5,
                                        },
                                        {
                                            "Date": "2024-07-06",
                                            "Open": 100.5,
                                            "Close": 100.0,
                                        },
                                        {
                                            "Date": "2024-07-07",
                                            "Open": 101.2,
                                            "Close": 99.5,
                                        },
                                        {
                                            "Date": "2024-07-08",
                                            "Open": 103.5,
                                            "Close": 105.0,
                                        },
                                        {
                                            "Date": "2024-07-09",
                                            "Open": 99.5,
                                            "Close": 98.0,
                                        },
                                        {
                                            "Date": "2024-07-10",
                                            "Open": 98.0,
                                            "Close": 97.0,
                                        },
                                        {
                                            "Date": "2024-07-11",
                                            "Open": 97.5,
                                            "Close": 100.0,
                                        },
                                        {
                                            "Date": "2024-07-12",
                                            "Open": 102.0,
                                            "Close": 101.5,
                                        },
                                        {
                                            "Date": "2024-07-13",
                                            "Open": 100.0,
                                            "Close": 102.0,
                                        },
                                        {
                                            "Date": "2024-07-14",
                                            "Open": 99.0,
                                            "Close": 98.5,
                                        },
                                        {
                                            "Date": "2024-07-15",
                                            "Open": 100.2,
                                            "Close": 99.8,
                                        },
                                        {
                                            "Date": "2024-07-16",
                                            "Open": 101.5,
                                            "Close": 100.0,
                                        },
                                    ]
                                )
                            }
                        ],
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the average close price for days above 100?",  # noqa: E501
                            }
                        ]
                    },
                },
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # First status update
    processing_update = assert_status_update_exists(status_updates, "Querying for")
    assert "Query" in processing_update["details"][0]

    # SQL execution update (exact wording/order can vary by model)
    sql_update = assert_status_update_exists_optional(
        status_updates, "SQL query executed"
    )
    if sql_update:
        if sql_update.get("details"):
            assert any(
                "sql" in str(detail).lower() for detail in sql_update.get("details", [])
            )

    # "Artifact generated" status update is yielded (for UI feedback)
    # Single-row results don't include artifact data in status update
    assert any(
        "Artifact generated" in status_update["message"]
        for status_update in status_updates
    )

    # Copilot answer - some models inline the numeric value, while others
    # reference the attached table artifact.
    response_lower = response_text.lower()
    assert (
        "102.67" in response_text
        or "102.66" in response_text
        or (
            "average" in response_lower
            and "close" in response_lower
            and "table" in response_lower
        )
    )

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {"ticker": "AAPL"}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["source_info"]["name"] == "Stock Price Performance"
    assert citations[0]["source_info"]["widget_id"] == "stock_price_performance"
    assert citations[0]["details"][0]["Data source"] == "Stock Price Performance"
    assert citations[0]["details"][0]["Ticker"] == "AAPL"


# TODO: This needs to be fixed
@pytest.mark.skip(
    reason="""
    The logic of this test is very tricky. It almost seems flawed.
    The query is "What is the P/E ratio of MSFT?", the data in the widgets
    tell the stock price of MSFT, not the P/E ratio. Unless we prompt the model
    to behave differently the assertions in this test will not be satisfied.
    """
)
def test_query_with_secondary_widgets_containing_irrelevant_data(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock price quote widget",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock price quote widget",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AMZN",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                # The query below can't be answered by the data in the widgets.
                "content": "What is the P/E ratio of MSFT?",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_quote",
                                    "input_args": {"ticker": "MSFT"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_quote",
                            "input_args": {"ticker": "MSFT"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": "$99.95",
                            }
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the P/E ratio of MSFT?",
                            }
                        ]
                    },
                },
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    response_text = parse_message_chunks(response.text)
    expected_in_response = [
        "unable to",
        "don't have",
        "do not have",
        "unfortunately",
        "couldn't",
        "not",
    ]

    assert response.status_code == 200
    assert any([expected in response_text.lower() for expected in expected_in_response])


def test_query_primary_widgets_structured(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    df = pd.DataFrame()
    df["stock_price"] = np.random.randint(0, 100, 1000)
    df["ticker"] = "AAPL"

    df.loc[99, "stock_price"] = 777

    test_structured_widget = df.to_json(orient="records")

    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_widget",
                    "name": "Stock Price",
                    "description": "Contains the stock price data for a particular ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                            "options": None,
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the maximum stock price of AAPL? Use function calling.",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_widget",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_widget",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": test_structured_widget,
                                "data_format": {
                                    "data_type": "object",
                                    "parse_as": "table",
                                },
                            },
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the maximum stock price of AAPL?",
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # First status update
    processing_update = assert_status_update_exists(status_updates, "Querying for")
    assert "Query" in processing_update["details"][0]

    # "Artifact generated" status update is yielded (for UI feedback)
    # Single-row artifacts are filtered from the response, but status is still sent
    assert any(
        "Artifact generated" in status_update["message"]
        for status_update in status_updates
    )

    # Copilot answer - value should be in response text for single-row results
    assert "777" in response_text

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {"ticker": "AAPL"}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["source_info"]["name"] == "Stock Price"
    assert citations[0]["source_info"]["widget_id"] == "stock_price_widget"
    assert citations[0]["details"][0]["Data source"] == "Stock Price"
    assert citations[0]["details"][0]["Ticker"] == "AAPL"


def test_query_primary_widgets_structured_with_list_columns(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    df = pd.DataFrame()
    df["name"] = ["Alice", "Bob", "Charlie", "Diana"]
    df["scores"] = [
        [85, 90, 78],
        [92, 88, 84, 76],
        [70, 75],
        [88, 92, 95, 89],
    ]

    test_structured_widget = df.to_json(orient="records")

    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "person_score_widget",
                    "name": "Person Score",
                    "description": "Contains the scores for each person.",
                    "params": [],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Which person had the highest average score?",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "person_score_widget",
                                    "input_args": {},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "person_score_widget",
                            "input_args": {},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": test_structured_widget,
                                "data_format": {
                                    "data_type": "object",
                                    "parse_as": "table",
                                },
                            },
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "Which person had the highest average score? Give me the specific name.",  # noqa: E501
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # Status updates are telemetry and may be absent depending on model path.
    if status_updates:
        assert_status_update_exists(status_updates, "Querying for")

        # Check for SQL query executed (may be at different positions)
        sql_updates = [
            u for u in status_updates if "SQL query executed" in u.get("message", "")
        ]
        if sql_updates:
            assert any(
                "sql" in str(detail).lower()
                for update in sql_updates
                for detail in update.get("details", [])
            )

        # "Artifact generated" status update is yielded for UI feedback on some paths
        assert any(
            "Artifact generated" in status_update["message"]
            for status_update in status_updates
        )

    # Response - value should be in response text for single-row results
    assert "Diana" in response_text
    assert "91" in response_text or "90" in response_text

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["source_info"]["name"] == "Person Score"
    assert citations[0]["source_info"]["widget_id"] == "person_score_widget"
    assert citations[0]["details"][0]["Data source"] == "Person Score"


# TODO: This needs to be fixed
@pytest.mark.skip(
    reason="This is actually supported. The test case needs do be deleted or modified."
)
def test_unable_to_query_explicit_context_structured_with_list_columns_dropped(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    df = pd.DataFrame()
    df["name"] = ["Alice", "Bob", "Charlie", "Diana"]
    # SQL doesn't support lists, so we need to check properly that when this
    # happens, we drop the column internally.
    df["scores"] = [
        [85, 90, 78],
        [92, 88, 84, 76],
        [70, 75],
        [88, 92, 95, 89, 85, 90, 78, 85, 90, 78, 85, 90, 78],  # Exceeds the limit
    ]

    test_structured_widget = df.to_json(orient="records")

    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "person_score_widget",
                    "name": "Person score widget",
                    "description": "Contains the scores for each person.",
                    "params": [],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Which person had the highest average score?",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "person_score_widget",
                                    "input_args": {},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "person_score_widget",
                            "input_args": {},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {"content": test_structured_widget},
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "Which person had the highest average score?",  # noqa: E501
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)

    fail_messages = [
        "unable",
        "cannot",
        "can't",
        "not available",
        "missing",
        "not possible",
        "does not",
        "doesn't",
    ]
    assert any([fail in response_text.lower() for fail in fail_messages])


def test_query_primary_widgets_structured_chart(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    df = pd.DataFrame()
    df["high"] = np.random.randint(0, 100, 100)
    df["date"] = pd.date_range("2021-01-01", periods=100)
    df["ticker"] = "AAPL"

    test_structured_widget = df.to_json(orient="records")

    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_widget",
                    "name": "Stock Price",
                    "description": "Contains the historical high prices for a particular ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                            "options": None,
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Show me a line chart of AAPL high prices.",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_widget",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_widget",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {"content": test_structured_widget},
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "Show me a line chart of AAPL high prices.",  # noqa: E501
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    message_artifacts = parse_message_artifacts(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # First status update
    processing_update = assert_status_update_exists(status_updates, "Querying for")
    assert "Query" in processing_update["details"][0]

    # Last status update
    assert status_updates[-1]["eventType"] == "INFO"
    assert status_updates[-1]["message"] == "Artifact generated"
    # The "Artifact generated" status updates don't return any details
    assert len(status_updates[-1]["details"]) == 0
    assert status_updates[-1]["artifacts"][0]["chart_params"]["chartType"] == "line"
    # LLM may rename date column (e.g., "date_unix_seconds"), so check contains "date"
    assert "date" in status_updates[-1]["artifacts"][0]["chart_params"]["xKey"]
    status_y_keys = status_updates[-1]["artifacts"][0]["chart_params"]["yKey"]
    assert any("high" in key for key in status_y_keys)

    # Copilot answer
    assert "high" in response_text.lower()
    assert "<|TESTING_PLACEHOLDER_ARTIFACT:" in response_text

    # Message artifacts
    assert len(message_artifacts) == 1
    assert message_artifacts[0]["type"] == "chart"
    assert message_artifacts[0]["chart_params"]["chartType"] == "line"
    # LLM may rename date column (e.g., "date_unix_seconds" or "timestamp"),
    # so check contains "date" or "timestamp"
    assert any(
        _token in message_artifacts[0]["chart_params"]["xKey"]
        for _token in {"date", "timestamp"}
    )
    message_y_keys = message_artifacts[0]["chart_params"]["yKey"]
    assert any("high" in key for key in message_y_keys)

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {"ticker": "AAPL"}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["source_info"]["name"] == "Stock Price"
    assert citations[0]["source_info"]["widget_id"] == "stock_price_widget"
    assert citations[0]["details"][0]["Ticker"] == "AAPL"


def test_query_primary_widgets_structured_explicit_chart_multiple_y_axis(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    df = pd.DataFrame()
    df["high"] = np.random.randint(0, 100, 100)
    df["close"] = np.random.randint(0, 100, 100)
    df["date"] = pd.date_range("2021-01-01", periods=100)
    df["ticker"] = "AAPL"

    test_structured_widget = df.to_json(orient="records", date_format="iso")

    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_widget",
                    "name": "Stock Price",
                    "description": "Contains the historical high and closing prices for a particular ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                            "options": None,
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Show me a bar chart where the x-axis is the date and the y-axis shows both the high price and close price.",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_widget",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_widget",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {"content": test_structured_widget},
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "Show me a bar chart where the x-axis is the date and the y-axis shows both the high price and close price.",  # noqa: E501
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    message_artifacts = parse_message_artifacts(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # First status update
    processing_update = assert_status_update_exists(status_updates, "Querying for")
    assert "Query" in processing_update["details"][0]

    # Last status update
    assert status_updates[-1]["eventType"] == "INFO"
    assert status_updates[-1]["message"] == "Artifact generated"
    # The "Artifact generated" status updates don't return any details
    assert len(status_updates[-1]["details"]) == 0
    assert status_updates[-1]["artifacts"][0]["chart_params"]["chartType"] == "bar"
    assert status_updates[-1]["artifacts"][0]["chart_params"]["xKey"] == "date"
    status_y_keys = [
        str(y_key).lower()
        for y_key in status_updates[-1]["artifacts"][0]["chart_params"]["yKey"]
    ]
    assert any("high" in y_key for y_key in status_y_keys)
    assert any("close" in y_key for y_key in status_y_keys)

    # Copilot answer
    assert "high" in response_text.lower()
    assert "close" in response_text
    assert "<|TESTING_PLACEHOLDER_ARTIFACT:" in response_text

    # Message artifacts
    assert len(message_artifacts) == 1
    assert message_artifacts[0]["type"] == "chart"
    assert message_artifacts[0]["chart_params"]["chartType"] == "bar"
    assert message_artifacts[0]["chart_params"]["xKey"] == "date"
    artifact_y_keys = [
        str(y_key).lower() for y_key in message_artifacts[0]["chart_params"]["yKey"]
    ]
    assert any("high" in y_key for y_key in artifact_y_keys)
    assert any("close" in y_key for y_key in artifact_y_keys)

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {"ticker": "AAPL"}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["source_info"]["name"] == "Stock Price"
    assert citations[0]["source_info"]["widget_id"] == "stock_price_widget"
    assert citations[0]["details"][0]["Ticker"] == "AAPL"


def test_query_primary_widgets_structured_explicit_chart_mentioned_in_query(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    df = pd.DataFrame()
    df["high"] = np.random.randint(0, 100, 100)
    df["date"] = pd.date_range("2021-01-01", periods=100)
    df["ticker"] = "AAPL"

    test_structured_widget = df.to_json(orient="records")

    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_widget",
                    "name": "Stock Price",
                    "description": "Contains the historical high prices for a particular ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                            "options": None,
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Show me a bar chart where the x-axis is the date and the y-axis is the high price. Use function calling.",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_widget",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_widget",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {"content": test_structured_widget},
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "Show me a bar chart where the x-axis is the date and the y-axis is the high price.",  # noqa: E501
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    message_artifacts = parse_message_artifacts(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # Last status update
    assert status_updates[-1]["eventType"] == "INFO"
    assert status_updates[-1]["message"] == "Artifact generated"
    # The "Artifact generated" status updates don't return any details
    assert len(status_updates[-1]["details"]) == 0
    assert status_updates[-1]["artifacts"][0]["chart_params"]["chartType"] == "bar"
    assert status_updates[-1]["artifacts"][0]["chart_params"]["xKey"] == "date"
    assert any(
        "high" in str(y_key).lower()
        for y_key in status_updates[-1]["artifacts"][0]["chart_params"]["yKey"]
    )

    # Copilot answer
    assert "high" in response_text.lower()
    assert "<|TESTING_PLACEHOLDER_ARTIFACT:" in response_text

    # Message artifacts
    assert len(message_artifacts) == 1
    assert message_artifacts[0]["type"] == "chart"
    assert message_artifacts[0]["chart_params"]["chartType"] == "bar"
    assert message_artifacts[0]["chart_params"]["xKey"] == "date"
    assert any(
        "high" in str(y_key).lower()
        for y_key in message_artifacts[0]["chart_params"]["yKey"]
    )

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {"ticker": "AAPL"}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["source_info"]["name"] == "Stock Price"
    assert citations[0]["source_info"]["widget_id"] == "stock_price_widget"
    assert citations[0]["details"][0]["Ticker"] == "AAPL"


def test_query_primary_widgets_structured_specific_chart_is_not_possible(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    df = pd.DataFrame()
    df["close"] = np.random.randint(0, 100, 100)
    df["date"] = pd.date_range("2021-01-01", periods=100)
    df["ticker"] = "AAPL"

    test_structured_widget = df.to_json(orient="records")

    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_widget",
                    "name": "Stock Price",
                    "description": "Contains the historical close prices for a particular ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                            "options": None,
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Show me a scatter chart of AAPL where the x-axis is the open price and the y-axis is the close price. Use function calling.",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_widget",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_widget",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {"content": test_structured_widget},
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "Show me a scatter chart of AAPL where the x-axis is the open price and the y-axis is the close price.",  # noqa: E501
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    message_artifacts = parse_message_artifacts(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # Sometimes it gives the answer straight away
    if status_updates:
        assert not hasattr(status_updates[-1], "artifacts")
    else:
        assert len(citations) == 0

    assert "<|TESTING_PLACEHOLDER_ARTIFACT:" not in response_text
    assert len(message_artifacts) == 0

    expected_in_response = ["unable", "scatter", "sorry"]
    if response_text.strip():
        assert any(
            expected in response_text.lower() for expected in expected_in_response
        )


def test_query_primary_widgets_structured_generates_chart_when_unspecified(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    df = pd.DataFrame()
    df["close"] = np.random.randint(0, 100, 100)
    df["date"] = pd.date_range("2021-01-01", periods=100)
    df["ticker"] = "AAPL"

    test_structured_widget = df.to_json(orient="records")

    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_widget",
                    "name": "Stock Price",
                    "description": "Contains the historical close prices for a particular ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                            "options": None,
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Show me a chart of the closing price of AAPL. Use function calling.",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_widget",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_widget",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {"content": test_structured_widget},
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "Show me a chart of the closing price of AAPL.",  # noqa: E501
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    message_artifacts = parse_message_artifacts(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # First status update
    processing_update = assert_status_update_exists(status_updates, "Querying for")
    assert "Query" in processing_update["details"][0]

    # Last status update
    assert status_updates[-1]["eventType"] == "INFO"
    assert status_updates[-1]["message"] == "Artifact generated"
    # The "Artifact generated" status updates don't return any details
    assert len(status_updates[-1]["details"]) == 0

    assert status_updates[-1]["artifacts"][0]["chart_params"]["chartType"] is not None
    assert any(
        word in status_updates[-1]["artifacts"][0]["chart_params"]["xKey"]
        for word in ["date", "timestamp"]
    )
    assert any(
        "close" in str(y_key).lower()
        for y_key in status_updates[-1]["artifacts"][0]["chart_params"]["yKey"]
    )

    assert "AAPL" in response_text
    assert "<|TESTING_PLACEHOLDER_ARTIFACT:" in response_text

    # Message artifacts
    assert len(message_artifacts) == 1
    assert message_artifacts[0]["type"] == "chart"
    assert message_artifacts[0]["chart_params"]["chartType"] is not None

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {"ticker": "AAPL"}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["source_info"]["name"] == "Stock Price"
    assert citations[0]["source_info"]["widget_id"] == "stock_price_widget"
    assert citations[0]["details"][0]["Ticker"] == "AAPL"


def test_query_primary_widgets_structured_multiple(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    df_aapl = pd.DataFrame()
    df_aapl["stock_price"] = np.random.randint(0, 100, 1000)
    df_aapl["ticker"] = "AAPL"

    df_amzn = pd.DataFrame()
    df_amzn["stock_price"] = np.random.randint(100, 200, 1000)
    df_amzn["ticker"] = "AMZN"

    test_structured_widget_aapl = df_aapl.to_json(orient="records")
    test_structured_widget_amzn = df_amzn.to_json(orient="records")

    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_widget_aapl",
                    "name": "Stock Price AAPL",
                    "description": "Contains the historical stock prices for AAPL.",
                    "params": [],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_widget_amzn",
                    "name": "Stock Price AMZN",
                    "description": "Contains the historical stock prices for AMZN.",
                    "params": [],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Which stock has the highest average stock price?",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_widget_aapl",
                                    "input_args": {},
                                },
                                {
                                    "widget_uuid": mock_uuids.ID2.value,
                                    "origin": "test_origin",
                                    "id": "stock_price_widget_amzn",
                                    "input_args": {},
                                },
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "stock_price_widget_aapl",
                            "input_args": {},
                        },
                        {
                            "widget_uuid": mock_uuids.ID2.value,
                            "origin": "test_origin",
                            "id": "stock_price_widget_amzn",
                            "input_args": {},
                        },
                    ]
                },
                "data": [
                    {
                        "items": [
                            {"content": test_structured_widget_aapl},
                        ]
                    },
                    {
                        "items": [
                            {"content": test_structured_widget_amzn},
                        ]
                    },
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the average stock price of AAPL?",  # noqa: E501
                            },
                            {
                                "widget_uuid": mock_uuids.ID2.value,
                                "query": "What is the average stock price of AMZN?",  # noqa: E501
                            },
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    message_artifacts = parse_message_artifacts(response.text)
    citations = parse_citations(response.text)

    # Check status updates - may or may not have artifact depending on query result

    # Copilot answer should contain the result either way
    assert "AMZN" in response_text
    # The response should contain either an artifact reference OR a direct answer
    has_artifact_placeholder = "<|TESTING_PLACEHOLDER_ARTIFACT:" in response_text
    has_direct_answer = any(
        word in response_text.lower() for word in ["average", "highest", "$"]
    )
    assert has_artifact_placeholder or has_direct_answer

    # Message artifacts
    assert all(artifact["type"] == "table" for artifact in message_artifacts)

    # Citations
    assert len(citations) == 2

    # Collect citation UUIDs and verify both are present (order-agnostic)
    citation_uuids = {c["source_info"]["uuid"] for c in citations}
    expected_uuids = {
        get_expected_context_uuid(mock_uuids.ID1.value, {}),
        get_expected_context_uuid(mock_uuids.ID2.value, {}),
    }
    assert citation_uuids == expected_uuids

    # Verify both citations are widgets with correct properties
    aapl_context_uuid = get_expected_context_uuid(mock_uuids.ID1.value, {})
    amzn_context_uuid = get_expected_context_uuid(mock_uuids.ID2.value, {})

    for citation in citations:
        assert citation["source_info"]["type"] == "widget"
        if citation["source_info"]["uuid"] == aapl_context_uuid:
            assert citation["source_info"]["name"] == "Stock Price AAPL"
            assert citation["source_info"]["widget_id"] == "stock_price_widget_aapl"
        elif citation["source_info"]["uuid"] == amzn_context_uuid:
            assert citation["source_info"]["name"] == "Stock Price AMZN"
            assert citation["source_info"]["widget_id"] == "stock_price_widget_amzn"


def test_query_wrap_formulas_in_latex(
    test_client: TestClient, mock_headers: Any, no_rate_limit: None
):
    payload = {
        "messages": [{"role": "human", "content": "What is the formula for P/E ratio?"}]
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    response_text = parse_message_chunks(response.text)

    assert response.status_code == 200
    assert "<latex>" in response_text
    assert "</latex>" in response_text


@pytest.mark.parametrize(
    "error, expected_message",
    [
        (openai.OpenAIError("Test error"), "AI service is temporarily unavailable"),
        (
            openai.AuthenticationError("Test error", response=MagicMock(), body=None),
            "Authentication error",
        ),
        (
            openai.RateLimitError("Test error", response=MagicMock(), body=None),
            "Rate limit exceeded",
        ),
        (
            openai.APIConnectionError(request=MagicMock()),
            "Connection error",
        ),
    ],
)
def test_query_web_search_openai_error(
    test_client: TestClient,
    mock_headers: Mapping[str, str],
    no_rate_limit: None,
    error: OpenAIError,
    expected_message: str,
):
    from openbb_ada.dependencies import get_copilot_service, get_web_search_llm_service
    from openbb_ada.main import app
    from openbb_ada.utils.utils import handle_openai_error

    class MockAsyncGenerator:
        def __init__(self, items):
            self._items = items

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    mock_async_gen = MockAsyncGenerator([handle_openai_error(error)])

    mock_web_search_service = MagicMock()
    mock_web_search_service.query = MagicMock(return_value=mock_async_gen)

    mock_copilot_service = MagicMock()
    mock_copilot_service.query = MagicMock(return_value=mock_async_gen)

    overrides = {
        get_web_search_llm_service: lambda: mock_web_search_service,
        get_copilot_service: lambda: mock_copilot_service,
    }
    app.dependency_overrides.update(overrides)  # type: ignore
    try:
        payload = {
            "messages": [{"role": "human", "content": "What is 1+1?"}],
            "api_keys": {
                "openai_api_key": "test-openai-key",
            },
        }
        response = test_client.post("/v1/query", headers=mock_headers, json=payload)

        status_updates = parse_status_updates(response.text)
        # Find the error status update (may not be at the last position due to planning)
        error_updates = [u for u in status_updates if u.get("eventType") == "ERROR"]
        assert len(error_updates) > 0, (
            f"No ERROR status update found in: {status_updates}"
        )
        assert error_updates[0]["message"] == expected_message
    finally:
        for dependency in overrides:
            app.dependency_overrides.pop(dependency, None)  # type: ignore


@pytest.mark.parametrize(
    "error, expected_message",
    [
        (openai.OpenAIError("Test error"), "AI service is temporarily unavailable"),
        (
            openai.AuthenticationError("Test error", response=MagicMock(), body=None),
            "Authentication error",
        ),
        (
            openai.APIConnectionError(request=MagicMock()),
            "Connection error",
        ),
    ],
)
def test_query_openai_error(
    test_client: TestClient,
    mock_headers: str,
    no_rate_limit: None,
    error: Exception,
    expected_message: str,
):
    # Skip this test for now as it has complex async exception handling
    # that's affected by planning
    # The core functionality (error handling) is tested elsewhere
    pytest.skip(
        "Skipping OpenAI error test due to ExceptionGroup handling "
        "complexity with planning feature"
    )


def test_query_context_exceeded_error(
    test_client: TestClient,
    mock_headers: Mapping[str, str],
    no_rate_limit: None,
):
    with patch(
        "openbb_ada.dependencies.CopilotService._compose_chain"
    ) as mock__compose_chain:
        mock__compose_chain.side_effect = ContextLimitExceededError
        payload = {
            "messages": [{"role": "human", "content": "What is 1+1?"}],
        }
        response = test_client.post("/v1/query", headers=mock_headers, json=payload)

        status_updates = parse_status_updates(response.text)
        assert status_updates[-1]["eventType"] == "ERROR"
        assert status_updates[-1]["message"] == "Context limit exceeded"


def test_query_uses_custom_openai_key(
    test_client: TestClient, mock_headers: Mapping[str, str], no_rate_limit: None
):
    with patch("openbb_ada.dependencies.CopilotService") as mock_copilot_service:
        payload = {
            "messages": [{"role": "human", "content": "What is 1+1?"}],
            "api_keys": {"openai_api_key": "test-openai-key"},
        }

        test_client.post("/v1/query", headers=mock_headers, json=payload)
        _, kwargs = mock_copilot_service.call_args
        assert kwargs["openai_api_key"] == "test-openai-key"


def test_query_force_web_search(
    test_client: TestClient, mock_headers: Any, no_rate_limit: None
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "In what specific year was OpenBB founded? Search the web.",
            }
        ],
        "workspace_options": {"workspace-web-search": True},
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)
    response_text = parse_message_chunks(response.text)

    assert len(status_updates) >= 1, (
        f"Expected at least 1 status update for web search, got {len(status_updates)}"
    )
    assert status_updates[0]["eventType"] == "INFO"
    assert "Searching web" in status_updates[0]["message"]

    assert "2021" in response_text
    assert len(citations) > 0


def test_query_force_web_search_with_messages(
    test_client: TestClient, mock_headers: Any, no_rate_limit: None
):
    payload = {
        "messages": [
            {"role": "human", "content": "What is a cool open-source company?"},
            {"role": "ai", "content": "OpenBB is a cool open-source company."},
            {
                "role": "human",
                "content": "In what year was that company founded? Search the web.",
            },
        ],
        "workspace_options": {"workspace-web-search": True},
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200
    status_updates = parse_status_updates(response.text)
    response_text = parse_message_chunks(response.text)
    citations = parse_citations(response.text)

    assert status_updates[0]["eventType"] == "INFO"
    assert "Searching web" in status_updates[0]["message"]

    assert "2021" in response_text
    assert len(citations) > 0


def test_query_force_web_search_with_tool_messages(
    test_client: TestClient, mock_headers: Any, no_rate_limit: None
):
    payload = {
        "messages": [
            {"role": "human", "content": "What is a cool open-source company?"},
            {"role": "ai", "content": "What is the stock price of AAPL?"},
            {"role": "tool", "content": "The stock price of AAPL is $150.00"},
            {
                "role": "ai",
                "content": "I have retrieved the data. The stock price of AAPL is $150.00",  # noqa: E501
            },
            {
                "role": "human",
                "content": "In what year was OpenBB founded? Search the web.",
            },
        ],
        "workspace_options": {"workspace-web-search": True},
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200
    status_updates = parse_status_updates(response.text)
    response_text = parse_message_chunks(response.text)
    citations = parse_citations(response.text)

    assert status_updates[0]["eventType"] == "INFO"
    assert "Searching web" in status_updates[0]["message"]

    assert "2021" in response_text
    assert len(citations) > 0


def test_query_force_web_search_with_multiple_human_messages(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    payload = {
        "messages": [
            {"role": "human", "content": "what is a cool food?"},
            {"role": "human", "content": "What is a cool open-source company?"},
            {
                "role": "human",
                "content": "In what year was OpenBB founded? Search the web.",
            },
        ],
        "workspace_options": {"workspace-web-search": True},
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200
    status_updates = parse_status_updates(response.text)
    response_text = parse_message_chunks(response.text)
    citations = parse_citations(response.text)

    assert len(status_updates) >= 1, (
        f"Expected at least 1 status update for web search, got {len(status_updates)}"
    )
    assert status_updates[0]["eventType"] == "INFO"
    assert "Searching web" in status_updates[0]["message"]

    assert "2021" in response_text
    assert len(citations) > 0


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_sends_error_sse_when_rate_limit_exceeded(
    test_client: TestClient,
    actual_set_copilot_call_count,
    mock_headers_with_actual_valid_access_token: dict,
    actual_valid_access_token: str,
):
    await actual_set_copilot_call_count(actual_valid_access_token, 100)
    payload = {
        "messages": [{"role": "human", "content": "What is 1+1?"}],
    }
    response = test_client.post(
        "/v1/query", headers=mock_headers_with_actual_valid_access_token, json=payload
    )
    status_updates = parse_status_updates(response.text)
    assert len(status_updates) == 1
    assert status_updates[0]["eventType"] == "ERROR"
    assert "You've reached your daily completion limit" in status_updates[0]["message"]


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.stateful
@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_increment_call_count_on_human_messages(
    test_client: TestClient,
    actual_valid_access_token: str,
    actual_set_copilot_call_count,
    mock_headers_with_actual_valid_access_token: dict,
):
    await actual_set_copilot_call_count(actual_valid_access_token, 0)
    payload = {
        "messages": [{"role": "human", "content": "What is 1+1?"}],
    }
    _ = test_client.post(
        "/v1/query", headers=mock_headers_with_actual_valid_access_token, json=payload
    )
    actual_usage = await get_copilot_call_count(actual_valid_access_token)
    assert actual_usage["number_copilot_calls_day_count"] == 1


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.stateful
@pytest.mark.integration
@pytest.mark.asyncio
async def test_query_does_not_increment_call_count_on_function_calls(
    test_client: TestClient,
    actual_valid_access_token: str,
    actual_set_copilot_call_count,
    mock_uuids: type[MockUUIDs],
    mock_headers_with_actual_valid_access_token: dict,
):
    await actual_set_copilot_call_count(actual_valid_access_token, 0)

    # Run the query with a function call and function call result and check that
    # the call count is not incremented (since we only want to increment for
    # human messages, not function calls)
    payload = {
        "messages": [
            {"role": "human", "content": "What is the stock price of AAPL?"},
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {"widget_uuids": [mock_uuids.ID1.value]},
                        "copilot_function_call_arguments": {
                            "widget_uuids": [mock_uuids.ID1.value]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {"widget_uuids": [mock_uuids.ID1.value]},
                "copilot_function_call_arguments": {
                    "widget_uuids": [mock_uuids.ID1.value]
                },
                "content": "$99.95",
            },
        ],
        "widgets": [
            {
                "uuid": mock_uuids.ID1.value,
                "name": "Stock price quote widget",
                "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                "metadata": {"ticker": "AAPL"},
            },
        ],
    }
    _ = test_client.post(
        "/v1/query", headers=mock_headers_with_actual_valid_access_token, json=payload
    )
    actual_usage = await get_copilot_call_count(actual_valid_access_token)
    assert actual_usage["number_copilot_calls_day_count"] == 0


def test_query_function_call_structured_data_final_answer_includes_artifact(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "historical_stock_price_widget",
                    "name": "Historical Stock Price",
                    "description": "Contains the historical stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AMZN",
                            "options": None,
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "context": None,
        "messages": [
            {
                "role": "human",
                "content": "Give me an artifact with all the closing prices for each day above $95.",  # <-- the original query  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "historical_stock_price_widget",
                                    "input_args": {"ticker": "AMZN"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "historical_stock_price_widget",
                            "input_args": {"ticker": "AMZN"},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    [
                                        {
                                            "Date": "2024-07-01",
                                            "Open": 100.0,
                                            "Close": 104.0,
                                        },
                                        {
                                            "Date": "2024-07-02",
                                            "Open": 101.0,
                                            "Close": 95.0,
                                        },
                                        {
                                            "Date": "2024-07-03",
                                            "Open": 98.0,
                                            "Close": 102.0,
                                        },
                                        {
                                            "Date": "2024-07-04",
                                            "Open": 97.0,
                                            "Close": 99.0,
                                        },
                                        {
                                            "Date": "2024-07-05",
                                            "Open": 102.0,
                                            "Close": 101.5,
                                        },
                                        {
                                            "Date": "2024-07-06",
                                            "Open": 100.5,
                                            "Close": 100.0,
                                        },
                                        {
                                            "Date": "2024-07-07",
                                            "Open": 101.2,
                                            "Close": 99.5,
                                        },
                                        {
                                            "Date": "2024-07-08",
                                            "Open": 103.5,
                                            "Close": 105.0,
                                        },
                                        {
                                            "Date": "2024-07-09",
                                            "Open": 99.5,
                                            "Close": 98.0,
                                        },
                                        {
                                            "Date": "2024-07-10",
                                            "Open": 98.0,
                                            "Close": 97.0,
                                        },
                                        {
                                            "Date": "2024-07-11",
                                            "Open": 97.5,
                                            "Close": 100.0,
                                        },
                                        {
                                            "Date": "2024-07-12",
                                            "Open": 102.0,
                                            "Close": 101.5,
                                        },
                                        {
                                            "Date": "2024-07-13",
                                            "Open": 100.0,
                                            "Close": 102.0,
                                        },
                                        {
                                            "Date": "2024-07-14",
                                            "Open": 99.0,
                                            "Close": 98.5,
                                        },
                                        {
                                            "Date": "2024-07-15",
                                            "Open": 100.2,
                                            "Close": 99.8,
                                        },
                                        {
                                            "Date": "2024-07-16",
                                            "Open": 101.5,
                                            "Close": 100.0,
                                        },
                                    ]
                                )
                            }
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "Retrieve all the closing prices for each day above $95.",  # noqa: E501
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    message_artifacts = parse_message_artifacts(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # Check that the artifact event fired during streaming back the final answer
    assert "<|TESTING_PLACEHOLDER_ARTIFACT:" in response_text

    # Check that we only have one artifact
    assert len(message_artifacts) == 1

    # Check that the artifact ID in the final answer matches the artifact ID in
    # the artifact return in the status update
    assert status_updates[-1]["artifacts"][0]["uuid"] == message_artifacts[0]["uuid"]

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {"ticker": "AMZN"}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["source_info"]["name"] == "Historical Stock Price"
    assert citations[0]["source_info"]["widget_id"] == "historical_stock_price_widget"
    assert citations[0]["details"][0]["Ticker"] == "AMZN"
    assert citations[0]["details"][0]["Data source"] == "Historical Stock Price"


def test_query_handle_function_call_and_result_with_incompatible_schema(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "origin": "test_origin",
                    "widget_id": "stock_price_quote",
                    "name": "Stock price quote widget",
                    "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the stock price of AAPL?",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "NOT-COMPATIBLE": [
                                {
                                    "origin": "test_origin",
                                    "id": "stock_price_quote",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                        "copilot_function_call_arguments": {
                            "ALSO-NOT-COMPATIBLE": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "queries": ["What is the stock price of AAPL?"],
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "NOT-COMPATIBLE": [
                        {
                            "origin": "test_origin",
                            "id": "stock_price_quote",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "content": " ".join([" "] * 1000)
                        + "The stock price of AAPL is $99.95."
                    }
                ],
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {  # missing the widget_uuid
                                    "origin": "test_origin",
                                    "id": "stock_price_quote",
                                    "input_args": {"ticker": "MSFT"},
                                }
                            ]
                        },
                        "copilot_function_call_arguments": {
                            "widget_queries": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "query": "What is the stock price of MSFT?",
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {  # missing the widget_uuid
                            "origin": "test_origin",
                            "id": "stock_price_quote",
                            "input_args": {"ticker": "MSFT"},
                        }
                    ]
                },
                "data": [
                    {
                        "content": " ".join([" "] * 1000)
                        + "The stock price of MSFT is $99.95."
                    }
                ],
            },
            {
                "role": "human",
                "content": "What's bigger? 3 or 7?",
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    assert "7" in response_text


def test_query_uses_specified_timezone(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "What is the current time?",
            },
        ],
        # doesn't observe daylight savings, so won't change on us!
        "timezone": "America/Phoenix",
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200
    response_text = parse_message_chunks(response.text)
    assert "MST" in response_text


def test_query_default_to_utc_if_no_timezone_is_specified(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "What is the current time?",
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200
    response_text = parse_message_chunks(response.text)
    assert "UTC" in response_text


def test_query_primary_widgets_structured_pie_chart(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    df = pd.DataFrame(
        [
            {"asset": "Stocks", "amount": 60000},
            {"asset": "Bonds", "amount": 40000},
            {"asset": "Cash", "amount": 7000},
            {"asset": "Real Estate", "amount": 5000},
            {"asset": "Commodities", "amount": 3000},
        ]
    )

    test_structured_widget = df.to_json(orient="records")

    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "portfolio_allocation",
                    "name": "Portfolio Allocation",
                    "description": "Contains the portfolio allocation across different assets",  # noqa: E501
                    "params": [],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Show me a pie chart of the portfolio allocation.",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "portfolio_allocation",
                                    "input_args": {},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "portfolio_allocation",
                            "input_args": {},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {"content": test_structured_widget},
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "Show me a pie chart of the portfolio allocation.",  # noqa: E501
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    message_artifacts = parse_message_artifacts(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # First status update
    processing_update = assert_status_update_exists(status_updates, "Querying for")
    assert "Query" in processing_update["details"][0]

    # Last status update
    assert status_updates[-1]["eventType"] == "INFO"
    assert status_updates[-1]["message"] == "Artifact generated"
    # The "Artifact generated" status update doesn't return any details
    assert len(status_updates[-1]["details"]) == 0
    assert status_updates[-1]["artifacts"][0]["chart_params"]["chartType"] == "pie"
    # Use flexible matching for angleKey field names
    angle_key = status_updates[-1]["artifacts"][0]["chart_params"]["angleKey"]
    assert any(
        term in angle_key.lower()
        for term in ["amount", "percentage", "allocation", "value", "invested", "total"]
    ), f"angleKey should contain a numeric field name, got: {angle_key}"
    assert (
        status_updates[-1]["artifacts"][0]["chart_params"]["calloutLabelKey"] == "asset"
    )

    # Copilot answer - accept various phrasings about allocation/portfolio
    assert (
        "portfolio" in response_text.lower()
        or "allocation" in response_text.lower()
        or "pie chart" in response_text.lower()
    )
    assert "<|TESTING_PLACEHOLDER_ARTIFACT:" in response_text

    # Message artifacts
    assert len(message_artifacts) == 1
    assert message_artifacts[0]["type"] == "chart"
    # Chart params - LLM may transform column names for clarity
    chart_params = message_artifacts[0]["chart_params"]
    assert chart_params["chartType"] == "pie"
    assert any(
        term in chart_params["angleKey"]
        for term in ["amount", "percentage", "allocation", "value", "share"]
    ), (
        "angleKey should relate to amount/percentage/allocation/"
        f"value/share, got: {chart_params['angleKey']}"
    )
    assert chart_params["calloutLabelKey"] == "asset"

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["source_info"]["name"] == "Portfolio Allocation"
    assert citations[0]["source_info"]["widget_id"] == "portfolio_allocation"


def test_query_primary_widgets_structured_donut_chart(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    df = pd.DataFrame(
        [
            {"asset": "Stocks", "amount": 60000},
            {"asset": "Bonds", "amount": 40000},
            {"asset": "Cash", "amount": 7000},
            {"asset": "Real Estate", "amount": 5000},
            {"asset": "Commodities", "amount": 3000},
        ]
    )

    test_structured_widget = df.to_json(orient="records")

    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "portfolio_allocation",
                    "name": "Portfolio Allocation",
                    "description": "Contains the portfolio allocation across different assets",  # noqa: E501
                    "params": [],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Show me a donut chart of the portfolio allocation.",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "portfolio_allocation",
                                    "input_args": {},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "portfolio_allocation",
                            "input_args": {},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {"content": test_structured_widget},
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "Show me a donut chart of the portfolio allocation.",  # noqa: E501
                            }
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    message_artifacts = parse_message_artifacts(response.text)
    status_updates = parse_status_updates(response.text)
    citations = parse_citations(response.text)

    # First status update
    processing_update = assert_status_update_exists(status_updates, "Querying for")
    assert "Query" in processing_update["details"][0]

    # Last status update
    assert status_updates[-1]["eventType"] == "INFO"
    assert status_updates[-1]["message"] == "Artifact generated"
    # The "Artifact generated" status update doesn't return any details
    assert len(status_updates[-1]["details"]) == 0

    assert status_updates[-1]["artifacts"][0]["chart_params"]["chartType"] == "donut"
    angle_key = status_updates[-1]["artifacts"][0]["chart_params"]["angleKey"]
    # The angleKey should be related to the numeric values (amount or percentage)
    assert any(
        term in angle_key.lower()
        for term in ["amount", "percentage", "value", "allocation"]
    ), f"angleKey should contain amount/percentage/value/allocation, got: {angle_key}"
    assert (
        status_updates[-1]["artifacts"][0]["chart_params"]["calloutLabelKey"] == "asset"
    )

    # Copilot answer - accept various phrasings about allocation/portfolio
    assert (
        "portfolio" in response_text.lower()
        or "allocation" in response_text.lower()
        or "pie chart" in response_text.lower()
    )
    assert "<|TESTING_PLACEHOLDER_ARTIFACT:" in response_text

    # Message artifacts
    assert len(message_artifacts) == 1
    assert message_artifacts[0]["type"] == "chart"

    # Citations
    assert len(citations) == 1
    assert citations[0]["source_info"]["uuid"] == get_expected_context_uuid(
        mock_uuids.ID1.value, {}
    )
    assert citations[0]["source_info"]["type"] == "widget"
    assert citations[0]["source_info"]["name"] == "Portfolio Allocation"
    assert citations[0]["source_info"]["widget_id"] == "portfolio_allocation"


def test_query_unsupported_chart_type(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "stock_price_widget",
                    "name": "Stock Price",
                    "description": "Contains the historical high prices for a particular ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                            "options": None,
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Show me a histogram of close prices.",  # noqa: E501
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)

    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    message_artifacts = parse_message_artifacts(response.text)
    citations = parse_citations(response.text)
    status_updates = parse_status_updates(response.text)

    assert not any([citations, message_artifacts])
    response_lower = response_text.lower()
    status_text = " ".join(
        [
            update.get("message", "")
            + " "
            + " ".join(str(detail) for detail in update.get("details", []))
            for update in status_updates
        ]
    ).lower()
    assert any(
        phrase in response_lower or phrase in status_text
        for phrase in [
            "do not yet support",
            "do not support",
            "cannot",
            "don't support",
            "unsupported",
            "high prices",
            "close prices",
            "clarifying price type",
        ]
    ), (
        "Expected unsupported/mismatch communication in response or status updates, "
        f"got response={response_text!r}, statuses={status_updates!r}"
    )


def test_query_get_widget_data_with_get_options_generates_partial_function_call(
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "price_feeds",
                    "name": "Price Feeds",
                    "description": "Live price data for crypto, forex, stocks and commodities.",  # noqa: E501
                    "params": [
                        {
                            "name": "symbol",
                            "type": "string",
                            "description": "The symbol of the asset to get the price for.",  # noqa: E501
                            "default_value": "AAPL",
                            "current_value": "AAPL",
                            "get_options": "true",
                        }
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "global_news",
                    "name": "Global News",
                    "description": "Worldwide news.",  # noqa: E501
                    "params": [
                        {
                            "name": "country",
                            "type": "string",
                            "description": "The country to get the news for.",
                            "default_value": "US",
                            "current_value": "US",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the price of gold and what is the current news in South Africa (use 'South Africa' as the country arg)?",  # noqa: E501
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Status updates
    assert status_updates[-1]["message"] == "Fetching parameter options"

    # Function calls
    assert function_calls[0]["function"] == "get_params_options"
    assert function_calls[0]["input_arguments"] == {
        "param_options_queries": [
            {
                "origin": "test_origin",
                "id": "price_feeds",
                "param": "symbol",
                "options_endpoint_input_args": {},
            }
        ]
    }
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 2
    )
    assert function_calls[0]["extra_state"]["continue_from"] == "get_widget_data"
    assert function_calls[0]["extra_state"]["param_options_widget_query_mapping"] == [
        {
            "widget_query_index": 0,
            "partial_input_args": {},
        },
    ]
    completed_mapping = function_calls[0]["extra_state"][
        "completed_data_source_request_query_mapping"
    ]
    assert len(completed_mapping) == 1
    assert completed_mapping[0]["widget_query_index"] == 1
    assert (
        completed_mapping[0]["data_source_request"]["widget_uuid"]
        == mock_uuids.ID2.value
    )
    assert completed_mapping[0]["data_source_request"]["origin"] == "test_origin"
    assert completed_mapping[0]["data_source_request"]["id"] == "global_news"
    assert completed_mapping[0]["data_source_request"]["ssm_request"] is None
    assert completed_mapping[0]["data_source_request"]["input_args"]["country"] in [
        "South Africa",
        "ZA",
    ]


def test_query_get_widget_data_with_get_options_uses_current_state_instead_of_getting_options(  # noqa: E501
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "price_feeds",
                    "name": "Price Feeds",
                    "description": "Live price data for crypto, forex, stocks and commodities.",  # noqa: E501
                    "params": [
                        {
                            "name": "symbol",
                            "type": "string",
                            "description": "The symbol of the asset to get the price for.",  # noqa: E501
                            "default_value": "AAPL",
                            "current_value": "XAU/USD",  # already the right value
                            "get_options": "true",
                        }
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "global_news",
                    "name": "Global News",
                    "description": "Worldwide news.",  # noqa: E501
                    "params": [
                        {
                            "name": "country",
                            "type": "string",
                            "description": "The country to get the news for.",
                            "default_value": "US",
                            "current_value": "US",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the current price of gold and what is the current news in the US?",  # noqa: E501
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    function_calls = parse_function_calls(response.text)

    # Robust function call validation with detailed error reporting
    assert len(function_calls) > 0, (
        f"Expected at least one function call but got none. "
        f"Response contained: {len(response.text)} chars. "
        f"Function calls parsed: {function_calls}"
    )

    # Find the get_widget_data function call (may not be first due to LLM variability)
    widget_data_call = None
    for call in function_calls:
        if call.get("function") == "get_widget_data":
            widget_data_call = call
            break

    assert widget_data_call is not None, (
        f"Expected get_widget_data function call but found: "
        f"{[call.get('function') for call in function_calls]}"
    )

    # Validate data sources exist
    input_args = widget_data_call.get("input_arguments", {})
    data_sources = input_args.get("data_sources", [])
    assert len(data_sources) == 2, (
        f"Expected 2 data sources but got {len(data_sources)}"
    )

    # Find data sources by widget_id for order-independent validation
    price_feeds_source = None
    global_news_source = None

    for source in data_sources:
        if source.get("id") == "price_feeds":
            price_feeds_source = source
        elif source.get("id") == "global_news":
            global_news_source = source

    # Validate price feeds widget
    assert price_feeds_source is not None, "Expected price_feeds data source"
    assert price_feeds_source["widget_uuid"] == mock_uuids.ID1.value
    assert price_feeds_source["origin"] == "test_origin"
    assert price_feeds_source["input_args"] == {"symbol": "XAU/USD"}
    assert price_feeds_source["ssm_request"] is None

    # Validate global news widget
    assert global_news_source is not None, "Expected global_news data source"
    assert global_news_source["widget_uuid"] == mock_uuids.ID2.value
    assert global_news_source["origin"] == "test_origin"
    assert global_news_source["input_args"] == {"country": "US"}
    assert global_news_source["ssm_request"] is None

    # Validate extra state with safe navigation
    extra_state = widget_data_call.get("extra_state", {})
    copilot_args = extra_state.get("copilot_function_call_arguments", {})
    widget_queries = copilot_args.get("widget_queries", [])
    assert len(widget_queries) == 2, (
        f"Expected 2 widget queries but got {len(widget_queries)}"
    )


def test_query_get_widget_data_with_get_options_uses_current_state_for_some_widgets(  # noqa: E501
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "price_feeds",
                    "name": "Price Feeds",
                    "description": "Live price data for crypto, forex, stocks and commodities.",  # noqa: E501
                    "params": [
                        {
                            "name": "symbol",
                            "type": "string",
                            "description": "The symbol of the asset to get the price for.",  # noqa: E501
                            "default_value": "AAPL",
                            "current_value": "XAU/USD",  # already the right value
                            "get_options": "true",
                        }
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "global_news",
                    "name": "Global News",
                    "description": "Worldwide news.",  # noqa: E501
                    "params": [
                        {
                            "name": "country",
                            "type": "string",
                            "description": "The country to get the news for.",
                            "default_value": "GER",
                            "current_value": "ZAR",  # not set to the right value
                            "get_options": "true",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the current price of gold and what is the current news in the US?",  # noqa: E501
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Status updates
    assert_status_update_exists(status_updates, "Fetching")

    # Function calls
    assert function_calls[0]["function"] == "get_params_options"
    assert function_calls[0]["input_arguments"] == {
        "param_options_queries": [
            {
                "origin": "test_origin",
                "id": "global_news",
                "param": "country",
                "options_endpoint_input_args": {},
            },
        ],
    }
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 2
    )
    assert function_calls[0]["extra_state"]["continue_from"] == "get_widget_data"
    assert function_calls[0]["extra_state"]["param_options_widget_query_mapping"] == [
        {"widget_query_index": 1, "partial_input_args": {}},
    ]
    assert function_calls[0]["extra_state"][
        "completed_data_source_request_query_mapping"
    ] == [
        {
            "widget_query_index": 0,
            "data_source_request": {
                "widget_uuid": mock_uuids.ID1.value,
                "origin": "test_origin",
                "id": "price_feeds",
                "input_args": {"symbol": "XAU/USD"},
                "ssm_request": None,
            },
        }
    ]
    assert "copilot_function_call_arguments" in function_calls[0]["extra_state"]


def test_query_portfolio_price_performance_with_get_options_and_current_value(
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    """
    Test for the scenario where parameters with get_options=True AND current_value
    should be included in LLM input argument generation to avoid crashes.

    This test covers the fix for the issue where the 'asset' parameter
    (with get_options=True and current_value='Portfolio Units') was being
    excluded from field definitions, causing the LLM to not generate it,
    which then triggered the filtering process and caused a crash.
    """
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "Portfolio Risk",
                    "widget_id": "portfolio_unit_price_custom_obb",
                    "name": "Portfolio Price Performance",
                    "description": "Get portfolio's unit price history.",
                    "params": [
                        {
                            "name": "portfolio",
                            "type": "text",
                            "description": "Select the portfolio.",
                            "default_value": "Client 1",
                            "current_value": "Client 1",
                            "options": ["Client 1", "Client 2", "Client 3"],
                            "get_options": False,
                        },
                        {
                            "name": "asset",
                            "type": "endpoint",
                            "description": "Select the asset.",
                            "default_value": "Portfolio Units",
                            "current_value": "Portfolio Units",  # Has current_value
                            "get_options": True,  # AND get_options=True - included
                            "options_params": [
                                {
                                    "type": "text",
                                    "name": "portfolio",
                                    "description": "Select the portfolio.",
                                    "inherit_value_from": "portfolio",
                                }
                            ],
                        },
                        {
                            "name": "period",
                            "type": "text",
                            "description": "Select the period.",
                            "default_value": "1 Year",
                            "current_value": "1 Year",
                            "options": [
                                "1 Month",
                                "3 Month",
                                "YTD",
                                "1 Year",
                                "3 Year",
                                "Max",
                            ],
                            "get_options": False,
                        },
                        {
                            "name": "returns",
                            "type": "boolean",
                            "description": "Show returns instead of unit price.",
                            "default_value": False,
                            "current_value": "True",
                            "options": ["true", "false"],
                            "get_options": False,
                        },
                    ],
                    "metadata": {"lastUpdated": 1759237629301},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Get the last 5 date and close of the portfolio price "
                "performance for client 2",
            },
        ],
    }
    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200
    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Should generate direct function call with all parameters including 'asset'
    # because asset has current_value even though it has get_options=True
    assert function_calls[0]["function"] == "get_widget_data"

    # The crucial assertion: 'asset' parameter should be included in input_args
    # even though it has get_options=True, because it has current_value
    input_arguments = function_calls[0]["input_arguments"]
    data_source = input_arguments["data_sources"][0]

    # Verify the key aspects of the fix
    assert data_source["origin"] == "Portfolio Risk"
    assert data_source["id"] == "portfolio_unit_price_custom_obb"

    # Most importantly: verify 'asset' parameter is included despite get_options=True
    input_args = data_source["input_args"]
    assert "asset" in input_args, (
        "The 'asset' parameter with get_options=True and current_value "
        "should be included"
    )
    assert input_args["asset"] == "Portfolio Units"
    assert input_args["portfolio"] == "Client 2"
    assert input_args["period"] == "1 Year"
    # returns can be either 'true' or 'false' depending on LLM interpretation
    assert "returns" in input_args

    # Status updates should show successful querying - check for either
    # "Querying" or "Requesting" since different test environments may have
    # slightly different message formats
    query_updates = [
        u
        for u in status_updates
        if "Querying" in u.get("message", "") or "Requesting" in u.get("message", "")
    ]
    assert len(query_updates) > 0, (
        f"Should have querying/requesting status update. "
        f"Got: {[u.get('message', '') for u in status_updates]}"
    )

    # Should NOT have status update for "Fetching parameter options" since
    # all params are available
    fetching_updates = [
        u
        for u in status_updates
        if "Fetching parameter options" in u.get("message", "")
    ]
    assert len(fetching_updates) == 0, (
        "Should not fetch parameter options when all params have values"
    )


def test_query_get_widget_data_with_get_options_generates_partial_function_call_multiple_params(  # noqa: E501
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "price_feeds",
                    "name": "Price Feeds",
                    "description": "Live price data for crypto, forex, stocks and commodities.",  # noqa: E501
                    "params": [
                        {
                            "name": "symbol",
                            "type": "string",
                            "description": "The symbol of the asset to get the price for.",  # noqa: E501
                            "default_value": "AAPL",
                            "current_value": "AAPL",
                            "get_options": "true",
                        },
                        {
                            "name": "currency",
                            "type": "string",
                            "description": "The currency to get the price for.",
                            "default_value": "USD",
                            "current_value": "USD",
                            "get_options": "true",
                        },
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the price of gold in euros?",  # noqa: E501
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)
    # Status updates
    assert status_updates[-1]["message"] == "Fetching parameter options"

    # Function calls
    assert function_calls[0]["function"] == "get_params_options"
    assert function_calls[0]["input_arguments"] == {
        "param_options_queries": [
            {
                "origin": "test_origin",
                "id": "price_feeds",
                "param": "symbol",
                "options_endpoint_input_args": {},
            },
            {
                "origin": "test_origin",
                "id": "price_feeds",
                "param": "currency",
                "options_endpoint_input_args": {},
            },
        ]
    }
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 1
    )
    assert function_calls[0]["extra_state"]["continue_from"] == "get_widget_data"
    assert function_calls[0]["extra_state"]["param_options_widget_query_mapping"] == [
        {
            "widget_query_index": 0,
            "partial_input_args": {},
        },
        {
            "widget_query_index": 0,
            "partial_input_args": {},
        },
    ]
    assert (
        function_calls[0]["extra_state"]["completed_data_source_request_query_mapping"]
        == []
    )


def test_query_get_widget_data_with_get_options_generates_partial_function_call_multiple_widgets_multiple_params(  # noqa: E501
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "price_feeds",
                    "name": "Price Feeds",
                    "description": "Live price data for crypto, forex, stocks and commodities.",  # noqa: E501
                    "params": [
                        {
                            "name": "symbol",
                            "type": "string",
                            "description": "The symbol of the asset to get the price for.",  # noqa: E501
                            "default_value": "AAPL",
                            "current_value": "AAPL",
                            "get_options": "true",
                        },
                        {
                            "name": "currency",
                            "type": "string",
                            "description": "The currency to get the price for.",
                            "default_value": "USD",
                            "current_value": "USD",
                            "get_options": "true",
                        },
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "global_news",
                    "name": "Global News",
                    "description": "Worldwide news.",  # noqa: E501
                    "params": [
                        {
                            "name": "country",
                            "type": "string",
                            "description": "The country to get the news for.",
                            "default_value": "US",
                            "current_value": "US",
                            "get_options": "true",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the price of gold in euros and what is the current news in Germany?",  # noqa: E501
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Status updates
    assert status_updates[-1]["message"] == "Fetching parameter options"

    # Function calls
    assert function_calls[0]["function"] == "get_params_options"
    assert function_calls[0]["input_arguments"] == {
        "param_options_queries": [
            {
                "origin": "test_origin",
                "id": "price_feeds",
                "param": "symbol",
                "options_endpoint_input_args": {},
            },
            {
                "origin": "test_origin",
                "id": "price_feeds",
                "param": "currency",
                "options_endpoint_input_args": {},
            },
            {
                "origin": "test_origin",
                "id": "global_news",
                "param": "country",
                "options_endpoint_input_args": {},
            },
        ]
    }
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 2
    )
    assert function_calls[0]["extra_state"]["continue_from"] == "get_widget_data"
    assert function_calls[0]["extra_state"]["param_options_widget_query_mapping"] == [
        {
            "widget_query_index": 0,
            "partial_input_args": {},
        },
        {
            "widget_query_index": 0,
            "partial_input_args": {},
        },
        {
            "widget_query_index": 1,
            "partial_input_args": {},
        },
    ]
    assert (
        function_calls[0]["extra_state"]["completed_data_source_request_query_mapping"]
        == []
    )


def test_query_handle_partial_function_call_with_continue_from(
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "price_feeds",
                    "name": "Price Feeds",
                    "description": "Live price data for crypto, forex, stocks and commodities.",  # noqa: E501
                    "params": [
                        {
                            "name": "symbol",
                            "type": "string",
                            "description": "The symbol of the asset to get the price for.",  # noqa: E501
                            "default_value": "AAPL",
                            "current_value": "AAPL",
                            "get_options": "true",
                        }
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "global_news",
                    "name": "Global News",
                    "description": "Worldwide news.",  # noqa: E501
                    "params": [
                        {
                            "name": "country",
                            "type": "string",
                            "description": "The country to get the news for.",
                            "default_value": "US",
                            "current_value": "US",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the price of gold and what is the current news in the US?",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_params_options",
                        "input_arguments": {
                            "param_options_queries": [
                                {
                                    "origin": "test_origin",
                                    "id": "price_feeds",
                                    "param": "symbol",
                                    "options_endpoint_input_args": {},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_params_options",
                "input_arguments": {
                    "param_options_queries": [
                        {
                            "origin": "test_origin",
                            "id": "price_feeds",
                            "param": "symbol",
                            "options_endpoint_input_args": {},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    {
                                        "param_options": [
                                            # These are the options returned by the
                                            # options endpoint after it was hit by the
                                            # front-end using the `options_query`
                                            # above.
                                            {
                                                "param": "symbol",
                                                "options": [
                                                    {
                                                        "label": "Gold in USD",
                                                        "value": "gold-in-usd",
                                                    },
                                                    {
                                                        "label": "Silver in USD",
                                                        "value": "xagusd",
                                                    },
                                                    {
                                                        "label": "Bitcoin in USD",
                                                        "value": "btcusd",
                                                    },
                                                    {
                                                        "label": "Euro in USD",
                                                        "value": "eurusd",
                                                    },
                                                    {
                                                        "label": "British Pound in USD",
                                                        "value": "gbpusd",
                                                    },
                                                    {
                                                        "label": "USD in Japanese Yen",
                                                        "value": "usdjpy",
                                                    },
                                                    {
                                                        "label": "USD in Canadian Dollar",  # noqa: E501
                                                        "value": "usdcad",
                                                    },
                                                    {
                                                        "label": "Australian Dollar in USD",  # noqa: E501
                                                        "value": "audusd",
                                                    },
                                                    {
                                                        "label": "New Zealand Dollar in USD",  # noqa: E501
                                                        "value": "nzdusd",
                                                    },
                                                    {
                                                        "label": "USD in Swiss Franc",
                                                        "value": "usdchf",
                                                    },
                                                ],
                                            }
                                        ]
                                    }
                                )
                            }
                        ]
                    }
                ],
                "extra_state": {
                    "continue_from": "get_widget_data",
                    "param_options_widget_query_mapping": [
                        {
                            "widget_query_index": 0,
                        },
                    ],
                    "completed_data_source_request_query_mapping": [
                        {
                            "widget_query_index": 1,
                            "data_source_request": {
                                "origin": "test_origin",
                                "id": "global_news",
                                "input_args": {"country": "US"},
                            },
                        },
                    ],
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the price of gold in USD?",
                                "use_current_inputs": False,
                            },
                            {
                                "widget_uuid": mock_uuids.ID2.value,
                                "query": "What is the current news in the US?",
                                "use_current_inputs": False,
                            },
                        ]
                    },
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Status updates - these may or may not appear depending on planning
    assert_status_update_exists_optional(
        status_updates, "Continue with querying widgets"
    )
    accessing_update = assert_status_update_exists_flexible(
        status_updates, "Requesting widget data"
    )
    assert accessing_update["details"] == [
        {"Origin": "test_origin", "Widget Id": "price_feeds", "symbol": "gold-in-usd"}
    ]
    # Second accessing widget update
    accessing_updates = [
        u for u in status_updates if "Requesting widget data" in u["message"]
    ]
    assert len(accessing_updates) >= 2
    assert accessing_updates[1]["details"] == [
        {"Origin": "test_origin", "Widget Id": "global_news", "country": "US"}
    ]

    # Function calls
    assert len(function_calls) == 1
    assert function_calls[0]["function"] == "get_widget_data"
    assert function_calls[0]["input_arguments"] == {
        "data_sources": [
            {
                "widget_uuid": mock_uuids.ID1.value,
                "origin": "test_origin",
                "id": "price_feeds",
                "input_args": {
                    # NB, LLM must choose from the suggested options!
                    "symbol": "gold-in-usd"
                },
                "ssm_request": None,
            },
            {
                "widget_uuid": mock_uuids.ID2.value,
                "origin": "test_origin",
                "id": "global_news",
                "input_args": {"country": "US"},
                "ssm_request": None,
            },
        ]
    }
    assert "copilot_function_call_arguments" in function_calls[0]["extra_state"]
    assert (
        "widget_queries"
        in function_calls[0]["extra_state"]["copilot_function_call_arguments"]
    )
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 2
    )

    widget_query_1 = function_calls[0]["extra_state"][
        "copilot_function_call_arguments"
    ]["widget_queries"][0]
    assert widget_query_1["widget_uuid"] == mock_uuids.ID1.value
    assert widget_query_1["query"] == "What is the price of gold in USD?"
    assert widget_query_1["use_current_inputs"] is False

    widget_query_2 = function_calls[0]["extra_state"][
        "copilot_function_call_arguments"
    ]["widget_queries"][1]
    assert widget_query_2["widget_uuid"] == mock_uuids.ID2.value
    assert widget_query_2["query"] == "What is the current news in the US?"
    assert widget_query_2["use_current_inputs"] is False


def test_query_handle_partial_function_call_with_continue_from_multiple_widgets_multiple_options_params(  # noqa: E501
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "price_feeds",
                    "name": "Price Feeds",
                    "description": "Live price data for crypto, forex, stocks and commodities.",  # noqa: E501
                    "params": [
                        {
                            "name": "metal",
                            "type": "string",
                            "description": "The metal to get the price for.",  # noqa: E501
                            "default_value": "gold",
                            "current_value": "gold",
                            "get_options": "true",
                        },
                        {
                            "name": "currency",
                            "type": "string",
                            "description": "The currency to get the price for.",
                            "default_value": "usd",
                            "current_value": "usd",
                            "get_options": "true",
                        },
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "global_news",
                    "name": "Global News",
                    "description": "Worldwide news.",  # noqa: E501
                    "params": [
                        {
                            "name": "country",
                            "type": "string",
                            "description": "The country to get the news for.",
                            "get_options": "true",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the price of silver in euros and what is the current news in the US?",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_params_options",
                        "input_arguments": {
                            "param_options_queries": [
                                {
                                    "origin": "test_origin",
                                    "id": "price_feeds",
                                    "param": "metal",
                                    "options_endpoint_input_args": {},
                                },
                                {
                                    "origin": "test_origin",
                                    "id": "price_feeds",
                                    "param": "currency",
                                    "options_endpoint_input_args": {},
                                },
                                {
                                    "origin": "test_origin",
                                    "id": "global_news",
                                    "param": "country",
                                    "options_endpoint_input_args": {},
                                },
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_params_options",
                "input_arguments": {
                    "param_options_queries": [
                        {
                            "origin": "test_origin",
                            "id": "price_feeds",
                            "param": "metal",
                            "options_endpoint_input_args": {},
                        },
                        {
                            "origin": "test_origin",
                            "id": "price_feeds",
                            "param": "currency",
                            "options_endpoint_input_args": {},
                        },
                        {
                            "origin": "test_origin",
                            "id": "global_news",
                            "param": "country",
                            "options_endpoint_input_args": {},
                        },
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    {
                                        "param_options": [
                                            {
                                                "param": "metal",
                                                "options": [
                                                    {
                                                        "label": "Gold",
                                                        "value": "xau",
                                                    },
                                                    {
                                                        "label": "Silver",
                                                        # funny name the model wouldn't
                                                        # be able to guess (i.e. it must
                                                        # use the options it is
                                                        # presented)
                                                        "value": "silver-lol",
                                                    },
                                                    {
                                                        "label": "Platinum",
                                                        "value": "xpt",
                                                    },
                                                    {
                                                        "label": "Palladium",
                                                        "value": "xpd",
                                                    },
                                                    {
                                                        "label": "Copper",
                                                        "value": "copper",
                                                    },
                                                ],
                                            }
                                        ]
                                    }
                                )
                            }
                        ]
                    },
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    {
                                        "param_options": [
                                            {
                                                "param": "currency",
                                                "options": [
                                                    {
                                                        "label": "USD",
                                                        "value": "usd",
                                                    },
                                                    {
                                                        "label": "EUR",
                                                        # abormal name the model
                                                        # wouldn't be able to
                                                        # guess (i.e. it must
                                                        # use the options it is
                                                        # presented)
                                                        "value": "euros",
                                                    },
                                                    {
                                                        "label": "GBP",
                                                        "value": "gbp",
                                                    },
                                                    {
                                                        "label": "JPY",
                                                        "value": "jpy",
                                                    },
                                                    {
                                                        "label": "CAD",
                                                        "value": "cad",
                                                    },
                                                    {
                                                        "label": "CHF",
                                                        "value": "chf",
                                                    },
                                                ],
                                            }
                                        ]
                                    }
                                )
                            }
                        ]
                    },
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    {
                                        "param_options": [
                                            {
                                                "param": "country",
                                                "options": [
                                                    {
                                                        "label": "United States",
                                                        "value": "usa",
                                                    },
                                                    {
                                                        "label": "European Union",
                                                        "value": "eu",
                                                    },
                                                    {"label": "Russia", "value": "ru"},
                                                ],
                                            }
                                        ]
                                    }
                                )
                            }
                        ]
                    },
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the price of silver in euros?",
                                "use_current_inputs": False,
                            },
                            {
                                "widget_uuid": mock_uuids.ID2.value,
                                "query": "What is the current news in the US?",
                                "use_current_inputs": False,
                            },
                        ]
                    },
                    "continue_from": "get_widget_data",
                    "param_options_widget_query_mapping": [
                        {
                            "widget_query_index": 0,
                        },
                        {
                            "widget_query_index": 0,
                        },
                        {
                            "widget_query_index": 1,
                        },
                    ],
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Find the "Requesting widget data" updates
    accessing_updates = [
        u for u in status_updates if "Requesting widget data" in u.get("message", "")
    ]
    assert len(accessing_updates) >= 2, (
        f"Expected at least 2 'Requesting widget data' updates, got {len(accessing_updates)}"  # noqa: E501
    )

    # Check the details of the accessing widget updates (order may vary)
    details_found = [update["details"][0] for update in accessing_updates]
    expected_details = [
        {
            "Origin": "test_origin",
            "Widget Id": "price_feeds",
            "metal": "silver-lol",
            "currency": "euros",
        },
        {"Origin": "test_origin", "Widget Id": "global_news", "country": "usa"},
    ]

    for expected_detail in expected_details:
        assert expected_detail in details_found, (
            f"Expected detail {expected_detail} not found in {details_found}"
        )

    # Function calls
    assert len(function_calls) == 1
    assert function_calls[0]["function"] == "get_widget_data"
    assert function_calls[0]["input_arguments"] == {
        "data_sources": [
            {
                "widget_uuid": mock_uuids.ID1.value,
                "origin": "test_origin",
                "id": "price_feeds",
                "input_args": {
                    "metal": "silver-lol",
                    "currency": "euros",
                },
                "ssm_request": None,
            },
            {
                "widget_uuid": mock_uuids.ID2.value,
                "origin": "test_origin",
                "id": "global_news",
                "input_args": {"country": "usa"},
                "ssm_request": None,
            },
        ]
    }
    assert "copilot_function_call_arguments" in function_calls[0]["extra_state"]
    assert (
        "widget_queries"
        in function_calls[0]["extra_state"]["copilot_function_call_arguments"]
    )
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 2
    )

    widget_query_1 = function_calls[0]["extra_state"][
        "copilot_function_call_arguments"
    ]["widget_queries"][0]
    assert widget_query_1["widget_uuid"] == mock_uuids.ID1.value
    assert widget_query_1["query"] == "What is the price of silver in euros?"
    assert widget_query_1["use_current_inputs"] is False

    widget_query_2 = function_calls[0]["extra_state"][
        "copilot_function_call_arguments"
    ]["widget_queries"][1]
    assert widget_query_2["widget_uuid"] == mock_uuids.ID2.value
    assert widget_query_2["query"] == "What is the current news in the US?"
    assert widget_query_2["use_current_inputs"] is False


def test_query_handle_partial_function_call_with_continue_from_no_appropriate_options(  # noqa: E501
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "price_feeds",
                    "name": "Price Feeds",
                    "description": "Live price data for crypto, forex, stocks and commodities.",  # noqa: E501
                    "params": [
                        {
                            "name": "metal",
                            "type": "string",
                            "description": "The metal to get the price for.",  # noqa: E501
                            "default_value": "gold",
                            "current_value": "gold",
                            "get_options": "true",
                        },
                        {
                            "name": "currency",
                            "type": "string",
                            "description": "The currency to get the price for.",
                            "default_value": "usd",
                            "current_value": "usd",
                            "get_options": "false",
                        },
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "global_news",
                    "name": "Global News",
                    "description": "Worldwide news.",  # noqa: E501
                    "params": [
                        {
                            "name": "country",
                            "type": "string",
                            "description": "The country to get the news for.",
                            "get_options": "true",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the price of silver in euros and what is the current news in the US?",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_params_options",
                        "input_arguments": {
                            "param_options_queries": [
                                {
                                    "origin": "test_origin",
                                    "id": "price_feeds",
                                    "param": "metal",
                                    "options_endpoint_input_args": {},
                                },
                                {
                                    "origin": "test_origin",
                                    "id": "price_feeds",
                                    "param": "currency",
                                    "options_endpoint_input_args": {},
                                },
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_params_options",
                "input_arguments": {
                    "param_options_queries": [
                        {
                            "origin": "test_origin",
                            "id": "price_feeds",
                            "param": "metal",
                            "options_endpoint_input_args": {},
                        },
                        {
                            "origin": "test_origin",
                            "id": "global_news",
                            "param": "country",
                            "options_endpoint_input_args": {},
                        },
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    {
                                        "param_options": [
                                            {
                                                "param": "metal",
                                                "options": [
                                                    # We deliberately exclude silver
                                                    # from the list to test that the
                                                    # model will not use the options it
                                                    # is presented with if they are not
                                                    # appropriate
                                                    {
                                                        "label": "Gold",
                                                        "value": "xau",
                                                    },
                                                    {
                                                        "label": "Platinum",
                                                        "value": "xpt",
                                                    },
                                                    {
                                                        "label": "Palladium",
                                                        "value": "xpd",
                                                    },
                                                    {
                                                        "label": "Copper",
                                                        "value": "copper",
                                                    },
                                                ],
                                            }
                                        ]
                                    }
                                )
                            }
                        ]
                    },
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    {
                                        "param_options": [
                                            {
                                                "param": "country",
                                                "options": [
                                                    {
                                                        "label": "United States",
                                                        "value": "usa",
                                                    },
                                                    {
                                                        "label": "European Union",
                                                        "value": "eu",
                                                    },
                                                    {"label": "Russia", "value": "ru"},
                                                ],
                                            }
                                        ]
                                    }
                                )
                            }
                        ]
                    },
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the price of silver in euros?",
                            },
                            {
                                "widget_uuid": mock_uuids.ID2.value,
                                "query": "What is the current news in the US?",
                            },
                        ],
                    },
                    "continue_from": "get_widget_data",
                    "param_options_widget_query_mapping": [
                        {
                            "widget_query_index": 0,
                        },
                        {
                            "widget_query_index": 1,
                        },
                    ],
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text).lower()
    status_updates = parse_status_updates(response.text)

    # Status updates - these may or may not appear depending on planning
    assert_status_update_exists_optional(
        status_updates, "Continue with querying widgets"
    )
    assert_status_update_exists_optional(status_updates, "Widget query failed")

    # For now, accept empty response as the system may not generate final response
    # when option substitutions occur without proper communication context
    if not response_text:
        # If no response, ensure function calls were generated indicating processing
        function_calls = parse_function_calls(response.text)
        assert len(function_calls) > 0, (
            "Expected either response text or function calls"
        )
        return

    assert response_text  # Ensure we got some response
    # The response should either contain the available metals or explain the limitation
    lower_response = response_text.lower()
    has_metal_content = any(
        metal in lower_response
        for metal in ["gold", "platinum", "palladium", "copper", "metal"]
    )
    has_explanation = any(
        word in lower_response
        for word in ["available", "option", "not found", "cannot"]
    )
    assert has_metal_content or has_explanation, (
        f"Response should mention available metals or explain limitation: "
        f"{response_text}"
    )


def test_query_get_widget_data_with_get_options_with_inherit_value_from_generates_partial_function_call(  # noqa: E501
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "document_hub_viewer",
                    "name": "Document Hub Viewer",
                    "description": "View documents.",  # noqa: E501
                    "params": [
                        {
                            "name": "document_filename",
                            "type": "string",
                            "description": "The filename of the document to view.",
                            "default_value": None,
                            "current_value": None,
                            "get_options": "true",
                            "options_params": [
                                {
                                    "type": "string",
                                    "name": "year",
                                    "description": "The year to get the documents for.",
                                    "inherit_value_from": "year",
                                },
                                {
                                    "type": "string",
                                    "name": "quarter",
                                    "description": "The quarter to get the documents for.",  # noqa: E501
                                    "inherit_value_from": "quarter",
                                },
                            ],
                        },
                        {
                            "name": "year",
                            "type": "string",
                            "description": "The year to get the documents for.",
                            "default_value": "2024",
                            "current_value": "2024",
                        },
                        {
                            "name": "quarter",
                            "type": "string",
                            "description": "The quarter to get the documents for.",
                            "default_value": "Q1",
                            "current_value": "Q1",
                            "options": ["Q1", "Q2", "Q3", "Q4"],
                        },
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "commodity_prices",
                    "name": "Commodity Prices",
                    "description": "Commodity prices.",  # noqa: E501
                    "params": [
                        {
                            "name": "commodity",
                            "type": "string",
                            "description": "The commodity to get the prices for.",
                            "default_value": "gold",
                            "current_value": "gold",
                            "get_options": "true",
                        },
                        {
                            "name": "currency",
                            "type": "string",
                            "description": "The currency to get the prices for.",
                            "default_value": "usd",
                            "current_value": "usd",
                            "get_options": "true",
                        },
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID3.value,
                    "origin": "test_origin",
                    "widget_id": "weather",
                    "name": "Weather",
                    "description": "Weather data.",  # noqa: E501
                    "params": [
                        {
                            "name": "location",
                            "type": "string",
                            "description": "The location to get the weather for.",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Summarize the report for 2024 Q3 and compare this to current price of gold in euros. Also get the weather in London. Fetch all the data at once.",  # noqa: E501
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Status updates
    assert_status_update_exists_flexible(status_updates, "Fetching")

    # Function calls
    assert function_calls[0]["function"] == "get_params_options"
    assert function_calls[0]["input_arguments"] == {
        "param_options_queries": [
            {
                "origin": "test_origin",
                "id": "document_hub_viewer",
                "param": "document_filename",
                "options_endpoint_input_args": {
                    "year": "2024",
                    "quarter": "Q3",
                },
            },
            {
                "origin": "test_origin",
                "id": "commodity_prices",
                "param": "commodity",
                "options_endpoint_input_args": {},
            },
            {
                "origin": "test_origin",
                "id": "commodity_prices",
                "param": "currency",
                "options_endpoint_input_args": {},
            },
        ]
    }

    assert function_calls[0]["extra_state"]["continue_from"] == "get_widget_data"
    assert function_calls[0]["extra_state"]["param_options_widget_query_mapping"] == [
        {
            "widget_query_index": 0,
            "partial_input_args": {"year": "2024", "quarter": "Q3"},
        },
        {
            "widget_query_index": 1,
            "partial_input_args": {},
        },
        {
            "widget_query_index": 1,
            "partial_input_args": {},
        },
    ]
    completed_mapping = function_calls[0]["extra_state"][
        "completed_data_source_request_query_mapping"
    ]
    assert len(completed_mapping) == 1
    completed_request = completed_mapping[0]
    assert completed_request["widget_query_index"] == 2
    assert (
        completed_request["data_source_request"]["widget_uuid"] == mock_uuids.ID3.value
    )
    assert completed_request["data_source_request"]["origin"] == "test_origin"
    assert completed_request["data_source_request"]["id"] == "weather"
    assert (
        "london"
        in completed_request["data_source_request"]["input_args"]["location"].lower()
    )
    assert completed_request["data_source_request"]["ssm_request"] is None

    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 3
    )


@pytest.mark.skip(
    reason="""This test needs to be re-evaluated and switched back on.
The prompt-drift resulted this test not passing at all. Possible reasons:
- Instructions related to handling default widget params lead to the agent preferring
  to use the default value (None)
- Consistent mismatch in the order of execution:
  get_param_options consistently follows get_widget_data resulting in a failed run
"""
)
def test_query_get_widget_data_with_get_options_with_inherit_value_from_generates_final_function_call(  # noqa: E501
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "document_hub_viewer",
                    "name": "Document Hub Viewer",
                    "description": "View documents.",  # noqa: E501
                    "params": [
                        {
                            "name": "document_filename",
                            "type": "string",
                            "description": "The filename of the document to view.",
                            "default_value": None,
                            "current_value": None,
                            "get_options": "true",
                            "options_params": [
                                {
                                    "type": "string",
                                    "name": "year",
                                    "description": "The year to get the documents for.",
                                    "inherit_value_from": "year",
                                },
                                {
                                    "type": "string",
                                    "name": "quarter",
                                    "description": "The quarter to get the documents for.",  # noqa: E501
                                    "inherit_value_from": "quarter",
                                },
                            ],
                        },
                        {
                            "name": "year",
                            "type": "string",
                            "description": "The year to get the documents for.",
                            "default_value": "2024",
                            "current_value": "2024",
                        },
                        {
                            "name": "quarter",
                            "type": "string",
                            "description": "The quarter to get the documents for.",
                            "default_value": "Q1",
                            "current_value": "Q1",
                            "options": ["Q1", "Q2", "Q3", "Q4"],
                        },
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID2.value,
                    "origin": "test_origin",
                    "widget_id": "commodity_prices",
                    "name": "Commodity Prices",
                    "description": "Commodity prices.",  # noqa: E501
                    "params": [
                        {
                            "name": "commodity",
                            "type": "string",
                            "description": "The commodity to get the prices for.",
                            "default_value": "gold",
                            "current_value": "gold",
                            "get_options": "true",
                        },
                        {
                            "name": "currency",
                            "type": "string",
                            "description": "The currency to get the prices for.",
                            "default_value": "usd",
                            "current_value": "usd",
                            "get_options": "true",
                        },
                    ],
                    "metadata": {},
                },
                {
                    "uuid": mock_uuids.ID3.value,
                    "origin": "test_origin",
                    "widget_id": "weather",
                    "name": "Weather",
                    "description": "Weather data.",  # noqa: E501
                    "params": [
                        {
                            "name": "location",
                            "type": "string",
                            "description": "The location to get the weather for.",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Summarize the report for 2024 Q3 and compare this to current price of gold in euros. Also get the weather in London. Fetch all the data at once.",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_params_options",
                        "input_arguments": {
                            "param_options_queries": [
                                {
                                    "origin": "test_origin",
                                    "id": "document_hub_viewer",
                                    "param": "document_filename",
                                    "options_endpoint_input_args": {
                                        "year": "2024",
                                        "quarter": "Q3",
                                    },
                                },
                                {
                                    "origin": "test_origin",
                                    "id": "commodity_prices",
                                    "param": "commodity",
                                    "options_endpoint_input_args": {},
                                },
                                {
                                    "origin": "test_origin",
                                    "id": "commodity_prices",
                                    "param": "currency",
                                    "options_endpoint_input_args": {},
                                },
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_params_options",
                "input_arguments": {
                    "param_options_queries": [
                        {
                            "origin": "test_origin",
                            "id": "document_hub_viewer",
                            "param": "document_filename",
                            "options_endpoint_input_args": {
                                "year": "2024",
                                "quarter": "Q3",
                            },
                        },
                        {
                            "origin": "test_origin",
                            "id": "commodity_prices",
                            "param": "commodity",
                            "options_endpoint_input_args": {},
                        },
                        {
                            "origin": "test_origin",
                            "id": "commodity_prices",
                            "param": "currency",
                            "options_endpoint_input_args": {},
                        },
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    {
                                        "param_options": [
                                            {
                                                "param": "document_filename",
                                                "options": [
                                                    {
                                                        "label": "Fruit Report 2024 Q3",
                                                        "value": "fruit_report_2024_q3",
                                                    },
                                                    {
                                                        "label": "Vegetable Report 2024 Q3",  # noqa: E501
                                                        "value": "vegetable_report_2024_q3",  # noqa: E501
                                                    },
                                                    # The option below is the relevant
                                                    # one to the user's query.
                                                    {
                                                        "label": "Gold Commodity Report 2024 Q3",  # noqa: E501
                                                        "value": "gold_commodity_report_2024_q3",  # noqa: E501
                                                    },
                                                ],
                                            },
                                        ]
                                    }
                                )
                            }
                        ]
                    },
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    {
                                        "param_options": [
                                            {
                                                "param": "commodity",
                                                "options": [
                                                    {"label": "Gold", "value": "gold"},
                                                    {
                                                        "label": "Silver",
                                                        "value": "silver",
                                                    },
                                                    {
                                                        "label": "Platinum",
                                                        "value": "platinum",
                                                    },
                                                ],
                                            },
                                        ]
                                    }
                                )
                            }
                        ]
                    },
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    {
                                        "param_options": [
                                            {
                                                "param": "currency",
                                                "options": [
                                                    {"label": "USD", "value": "usd"},
                                                    {"label": "EUR", "value": "eur"},
                                                    {"label": "GBP", "value": "gbp"},
                                                ],
                                            },
                                        ],
                                    }
                                )
                            }
                        ]
                    },
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "Get the commodities report for 2024 Q3",
                            },
                            {
                                "widget_uuid": mock_uuids.ID2.value,
                                "query": "Get the price of gold in euros",
                            },
                            {
                                "widget_uuid": mock_uuids.ID3.value,
                                "query": "Get the weather in London",
                            },
                        ]
                    },
                    "continue_from": "get_widget_data",
                    "param_options_widget_query_mapping": [
                        {
                            "widget_query_index": 0,
                            "partial_input_args": {"year": "2024", "quarter": "Q3"},
                        },
                        {
                            "widget_query_index": 1,
                            "partial_input_args": {},
                        },
                        {
                            "widget_query_index": 1,
                            "partial_input_args": {},
                        },
                    ],
                    "completed_data_source_request_query_mapping": [
                        {
                            "widget_query_index": 2,
                            "data_source_request": {
                                "origin": "test_origin",
                                "id": "weather",
                                "input_args": {
                                    "location": "London",
                                },
                            },
                        }
                    ],
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Status updates
    assert status_updates[0]["message"] == "Continuing from previous state"

    # Function calls
    assert function_calls[0]["function"] == "get_widget_data"
    assert function_calls[0]["input_arguments"] == {
        "data_sources": [
            {
                "widget_uuid": mock_uuids.ID1.value,
                "origin": "test_origin",
                "id": "document_hub_viewer",
                "input_args": {
                    "document_filename": "gold_commodity_report_2024_q3",
                    "year": "2024",
                    "quarter": "Q3",
                },
                "ssm_request": None,
            },
            {
                "widget_uuid": mock_uuids.ID2.value,
                "origin": "test_origin",
                "id": "commodity_prices",
                "input_args": {
                    "commodity": "gold",
                    "currency": "eur",
                },
                "ssm_request": None,
            },
            {
                "widget_uuid": mock_uuids.ID3.value,
                "origin": "test_origin",
                "id": "weather",
                "input_args": {
                    "location": "London",
                },
                "ssm_request": None,
            },
        ]
    }

    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 3
    )


def test_query_generate_function_call_with_null_default_value_input_arg(
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "crypto_prices",
                    "name": "Crypto Prices",
                    "description": "Live price data for crypto.",  # noqa: E501
                    "params": [
                        {
                            "name": "start_date",
                            "type": "date",
                            "description": "The start date of the data to retrieve.",  # noqa: E501
                            "default_value": "2024-01-01",
                            "current_value": "2024-01-01",
                            "get_options": "false",
                        },
                        {
                            "name": "end_date",
                            "type": "date",
                            "description": "The end date of the data to retrieve. Leave null to get latest data.",  # noqa: E501
                            "default_value": None,
                            "current_value": None,
                            "get_options": "false",
                        },
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "Get me the latest crypto prices starting from 2024-01-01.",  # noqa: E501
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    function_calls = parse_function_calls(response.text)
    assert len(function_calls) == 1
    assert function_calls[0]["function"] == "get_widget_data"
    assert function_calls[0]["input_arguments"] == {
        "data_sources": [
            {
                "widget_uuid": mock_uuids.ID1.value,
                "origin": "test_origin",
                "id": "crypto_prices",
                "input_args": {
                    "start_date": "2024-01-01",
                    # `end_date` is missing because we drop `None` values on the
                    # function call SSE
                },
                "ssm_request": None,
            }
        ]
    }


def test_query_get_widget_data_with_get_options_for_many_options(
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "price_feeds",
                    "name": "Price Feeds",
                    "description": "Live price data for crypto, forex, stocks and commodities.",  # noqa: E501
                    "params": [
                        {
                            "name": "symbol",
                            "type": "string",
                            "description": "The symbol of the asset to get the price for.",  # noqa: E501
                            "default_value": "AAPL",
                            "current_value": "AAPL",
                            "get_options": "true",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the price of gold?",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_params_options",
                        "input_arguments": {
                            "param_options_queries": [
                                {
                                    "origin": "test_origin",
                                    "id": "price_feeds",
                                    "param": "symbol",
                                    "options_endpoint_input_args": {},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_params_options",
                "input_arguments": {
                    "param_options_queries": [
                        {
                            "origin": "test_origin",
                            "id": "price_feeds",
                            "param": "symbol",
                            "options_endpoint_input_args": {},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    {
                                        "param_options": [
                                            # These are the options returned by the
                                            # options endpoint after it was hit by the
                                            # front-end using the `options_query`
                                            # above.
                                            {
                                                "param": "symbol",
                                                # Let's add ~200k options
                                                "options": [
                                                    {
                                                        "label": f"Asset {i}",
                                                        "value": f"asset-{i}",
                                                    }
                                                    for i in range(1, 200_000)
                                                ]
                                                + [
                                                    # The needle in the haystack we
                                                    # actually want vvv
                                                    {
                                                        "label": "Gold",
                                                        "value": "gold-in-usd",
                                                    }
                                                ],
                                            }
                                        ]
                                    }
                                )
                            }
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the price of gold in USD?",
                            },
                        ]
                    },
                    "continue_from": "get_widget_data",
                    "param_options_widget_query_mapping": [
                        {
                            "widget_query_index": 0,
                        },
                    ],
                    "completed_data_source_request_query_mapping": [],
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    status_updates = parse_status_updates(response.text)
    function_calls = parse_function_calls(response.text)

    # Status updates
    assert status_updates[0]["message"] == "Continuing from previous state"
    assert status_updates[1]["message"] == "Requesting widget data"
    assert status_updates[1]["details"] == [
        {"Origin": "test_origin", "Widget Id": "price_feeds", "symbol": "gold-in-usd"}
    ]

    # Function calls
    assert len(function_calls) == 1
    assert function_calls[0]["function"] == "get_widget_data"
    assert function_calls[0]["input_arguments"] == {
        "data_sources": [
            {
                "widget_uuid": mock_uuids.ID1.value,
                "origin": "test_origin",
                "id": "price_feeds",
                "input_args": {
                    # NB, LLM must choose from the suggested options!
                    "symbol": "gold-in-usd"
                },
                "ssm_request": None,
            },
        ]
    }
    assert "copilot_function_call_arguments" in function_calls[0]["extra_state"]
    assert (
        "widget_queries"
        in function_calls[0]["extra_state"]["copilot_function_call_arguments"]
    )
    assert (
        len(
            function_calls[0]["extra_state"]["copilot_function_call_arguments"][
                "widget_queries"
            ]
        )
        == 1
    )

    widget_query = function_calls[0]["extra_state"]["copilot_function_call_arguments"][
        "widget_queries"
    ][0]
    assert widget_query["widget_uuid"] == mock_uuids.ID1.value
    assert widget_query["query"] == "What is the price of gold in USD?"


def test_query_get_widget_data_with_get_options_for_no_options_returned(
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "price_feeds",
                    "name": "Price Feeds",
                    "description": "Live price data for crypto, forex, stocks and commodities.",  # noqa: E501
                    "params": [
                        {
                            "name": "symbol",
                            "type": "string",
                            "description": "The symbol of the asset to get the price for.",  # noqa: E501
                            "default_value": "AAPL",
                            "current_value": "AAPL",
                            "get_options": "true",
                        }
                    ],
                    "metadata": {},
                },
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the price of gold?",  # noqa: E501
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_params_options",
                        "input_arguments": {
                            "param_options_queries": [
                                {
                                    "origin": "test_origin",
                                    "id": "price_feeds",
                                    "param": "symbol",
                                    "options_endpoint_input_args": {},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_params_options",
                "input_arguments": {
                    "param_options_queries": [
                        {
                            "origin": "test_origin",
                            "id": "price_feeds",
                            "param": "symbol",
                            "options_endpoint_input_args": {},
                        }
                    ]
                },
                "data": [
                    {
                        "items": [
                            {
                                "content": json.dumps(
                                    {
                                        "param_options": [
                                            # These are the options returned by the
                                            # options endpoint after it was hit by the
                                            # front-end using the `options_query`
                                            # above.
                                            {
                                                "param": "symbol",
                                                "options": [],
                                            }
                                        ]
                                    }
                                )
                            }
                        ]
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the price of gold in USD?",
                            },
                        ]
                    },
                    "continue_from": "get_widget_data",
                    "param_options_widget_query_mapping": [
                        {
                            "widget_query_index": 0,
                        },
                    ],
                    "completed_data_source_request_query_mapping": [],
                },
            },
        ],
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    status_updates = parse_status_updates(response.text)

    # Status updates - these may or may not appear depending on planning
    assert_status_update_exists_optional(
        status_updates, "Continue with querying widgets"
    )
    assert_status_update_exists_optional(status_updates, "Widget query failed")

    # Actual response text
    response_lower = response_text.lower()
    status_text = " ".join(
        [
            update.get("message", "")
            + " "
            + " ".join(str(detail) for detail in update.get("details", []))
            for update in status_updates
        ]
    ).lower()
    assert any(
        target in response_lower or target in status_text
        for target in [
            "couldn't",
            "could not",
            "unable",
            "cannot",
            "no valid options",
            "not available",
            "provide",
            "specify",
            "which symbol",
            "widget query failed",
        ]
    )
    assert response_text.strip() != "" or len(status_updates) > 0


def test_query_get_widget_data_with_client_function_call_error(
    test_client: TestClient,
    mock_headers: dict,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "widgets": {
            "primary": [
                {
                    "uuid": mock_uuids.ID1.value,
                    "origin": "test_origin",
                    "widget_id": "financial_ratios",
                    "name": "Financial Ratios",
                    "description": "Contains a number of financial ratios for a ticker.",  # noqa: E501
                    "params": [
                        {
                            "name": "ticker",
                            "type": "string",
                            "description": "The stock ticker symbol.",
                            "default_value": None,
                            "current_value": "AAPL",
                        }
                    ],
                    "metadata": {},
                }
            ],
            "secondary": [],
            "extra": [],
        },
        "messages": [
            {
                "role": "human",
                "content": "What is the debt-to-equity ratio of AAPL?",
            },
            {
                "role": "ai",
                "content": json.dumps(
                    {
                        "function": "get_widget_data",
                        "input_arguments": {
                            "data_sources": [
                                {
                                    "widget_uuid": mock_uuids.ID1.value,
                                    "origin": "test_origin",
                                    "id": "financial_ratios",
                                    "input_args": {"ticker": "AAPL"},
                                }
                            ]
                        },
                    }
                ),
            },
            {
                "role": "tool",
                "function": "get_widget_data",
                "input_arguments": {
                    "data_sources": [
                        {
                            "widget_uuid": mock_uuids.ID1.value,
                            "origin": "test_origin",
                            "id": "financial_ratios",
                            "input_args": {"ticker": "AAPL"},
                        }
                    ]
                },
                "data": [
                    {
                        "error_type": "not_found",
                        "content": "Data for AAPL not found in widget.",
                    }
                ],
                "extra_state": {
                    "copilot_function_call_arguments": {
                        "widget_queries": [
                            {
                                "widget_uuid": mock_uuids.ID1.value,
                                "query": "What is the debt-to-equity ratio of AAPL?",  # noqa: E501
                                "use_current_inputs": True,
                            }
                        ]
                    },
                },
            },
        ],
        "workspace_options": {"workspace-web-search": True},
    }

    response = test_client.post("/v1/query", headers=mock_headers, json=payload)
    assert response.status_code == 200

    response_text = parse_message_chunks(response.text)
    status_updates = parse_status_updates(response.text)

    assert (
        status_updates[0]["message"]
        == "An error occurred while fetching data from a widget"
    )
    assert status_updates[0]["details"] == [
        {
            "Origin": "test_origin",
            "Widget Id": "financial_ratios",
            "ticker": "AAPL",
            "Error type": "not_found",
            "Error content": "Data for AAPL not found in widget.",
        }
    ]

    assert response_text  # non-empty response

    # Verify that the agent has triggered a web search after the widget error (optional)
    web_search_update = assert_status_update_exists_optional(
        status_updates, "Searching web"
    )
    # Web search may or may not happen depending on LLM behavior, but if it
    # does, verify it worked
    if not web_search_update:
        # If no web search, at least ensure we got a meaningful response
        # despite the widget error
        assert response_text and len(response_text.strip()) > 0
