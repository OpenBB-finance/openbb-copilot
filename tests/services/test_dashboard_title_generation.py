from typing import Any

from fastapi.testclient import TestClient


def test_service_generate_dashboard_title_multiple_widgets(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
):
    payload = {
        "widgets": [
            {
                "origin": "test_origin_1",
                "widget_id": "stock_price_quote",
                "name": "Stock price quote widget",
                "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                "params": [
                    {
                        "name": "ticker",
                        "type": "string",
                        "description": "The stock ticker symbol.",
                        "current_value": "AAPL",
                    }
                ],
                "metadata": {},
            },
            {
                "origin": "test_origin_2",
                "widget_id": "company_overview",
                "name": "Company overview widget",
                "description": "Contains the overview of a particular company",
                "params": [
                    {
                        "name": "ticker",
                        "type": "string",
                        "description": "The stock ticker symbol.",
                        "current_value": "AAPL",
                    }
                ],
                "metadata": {},
            },
            {
                "origin": "test_origin_3",
                "widget_id": "stock_price_news",
                "name": "Stock price news widget",
                "description": "Contains the latest news for a particular stock",
                "params": [
                    {
                        "name": "ticker",
                        "type": "string",
                        "description": "The stock ticker symbol.",
                        "current_value": "AAPL",
                    }
                ],
                "metadata": {},
            },
            {
                "origin": "test_origin_4",
                "widget_id": "asset_allocation",
                "name": "Asset allocation widget",
                "description": "Contains the asset allocation for a particular stock",
                "params": [
                    {
                        "name": "ticker",
                        "type": "string",
                        "description": "The stock ticker symbol.",
                        "current_value": "AAPL",
                    }
                ],
                "metadata": {},
            },
        ],
    }

    response = test_client.post(
        "/v1/generate/dashboard/title",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response = response.json()
    # More flexible keywords to account for LLM variation
    expected_in_title = [
        "aapl",
        "allocation",
        "analysis",
        "analytics",
        "apple",
        "asset",
        "company",
        "dashboard",
        "data",
        "equity",
        "financial",
        "hub",
        "insights",
        "intelligence",
        "investment",
        "market",
        "metrics",
        "monitor",
        "news",
        "outlook",
        "overview",
        "performance",
        "portfolio",
        "price",
        "quote",
        "report",
        "review",
        "snapshot",
        "stock",
        "summary",
        "tracker",
        "view",
        "watch",
    ]
    # First check that we got a valid response
    assert isinstance(response, str) and len(response.strip()) > 0, (
        f"Expected non-empty string response, got: {response}"
    )
    # Check if response contains expected keywords (more flexible)
    response_lower = response.lower()
    matching_words = [word for word in expected_in_title if word in response_lower]
    assert len(matching_words) > 0, (
        f"Expected title to contain at least one of {expected_in_title}, "
        f"but got: '{response}'"
    )


def test_service_generate_dashboard_title_multiple_similar_widgets(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
):
    payload = {
        "widgets": [
            {
                "origin": "test_origin_1",
                "widget_id": "stock_price_quote",
                "name": "Stock price quote widget",
                "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                "params": [
                    {
                        "name": "ticker",
                        "type": "string",
                        "description": "The stock ticker symbol.",
                        "current_value": "AAPL",
                    }
                ],
                "metadata": {},
            },
            {
                "origin": "test_origin_2",
                "widget_id": "stock_price_quote",
                "name": "Stock price quote widget",
                "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                "params": [
                    {
                        "name": "ticker",
                        "type": "string",
                        "description": "The stock ticker symbol.",
                        "current_value": "AMZN",
                    }
                ],
                "metadata": {},
            },
        ],
    }

    response = test_client.post(
        "/v1/generate/dashboard/title",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response = response.json()
    expected_in_title = ["stock", "price", "overview", "quote"]
    assert any(word in response.lower() for word in expected_in_title)


def test_service_generate_dashboard_title_multiple_dissimilar_widgets(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
):
    payload = {
        "widgets": [
            {
                "origin": "test_origin_1",
                "widget_id": "stock_price_quote",
                "name": "Stock price quote widget",
                "description": "Contains the current stock price for a particular ticker",  # noqa: E501
                "params": [
                    {
                        "name": "ticker",
                        "type": "string",
                        "description": "The stock ticker symbol.",
                        "current_value": "AAPL",
                    }
                ],
                "metadata": {},
            },
            {
                "origin": "test_origin_2",
                "widget_id": "foreign_exchange_rate",
                "name": "Foreign exchange rate widget",
                "description": "Contains the current foreign exchange rate for a particular currency",  # noqa: E501
                "params": [
                    {
                        "name": "ticker",
                        "type": "string",
                        "description": "The currency pair symbol.",
                        "current_value": "USD-EUR",
                    }
                ],
                "metadata": {},
            },
            {
                "origin": "test_origin_3",
                "widget_id": "cpi_inflation_rate",
                "name": "CPI inflation rate widget",
                "description": "Contains the current CPI inflation rate",
                "params": [],
                "metadata": {},
            },
            {
                "origin": "test_origin_4",
                "widget_id": "global_news",
                "name": "Global news widget",
                "description": "Contains the latest global news",
                "params": [],
                "metadata": {},
            },
        ],
    }

    response = test_client.post(
        "/v1/generate/dashboard/title",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response = response.json()
    expected_in_title = ["market", "overview", "macroeconomic"]
    assert any(word in response.lower() for word in expected_in_title)
