from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import MockUUIDs


def test_service_generate_chat_title_avoid_picking_typo_and_question_in_title(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "who is jeff bezzos?",
            },
            {
                "role": "ai",
                "content": """Jeff Bezos is an American entrepreneur, media proprietor, investor, and commercial astronaut.
                He is best known as the founder of Amazon, the world's largest online retailer. Bezos founded Amazon in 1994,
                initially as an online bookstore, but it quickly expanded to a wide variety of other products and services,
                including cloud computing, artificial intelligence, and digital streaming.

                Bezos served as the CEO of Amazon from its inception until July 2021, when he stepped down to become the executive
                chairman of Amazon's board. Under his leadership, Amazon grew to become one of the most valuable companies in the world.

                In addition to his work with Amazon, Bezos founded Blue Origin, a private aerospace manufacturer and sub-orbital spaceflight
                services company, in 2000. He also purchased The Washington Post in 2013.

                Bezos is one of the wealthiest individuals in the world, with his net worth largely derived from his holdings in Amazon.""",  # noqa: E501
            },
        ]
    }

    response = test_client.post(
        "/v1/generate/chat/title",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response = response.json()

    assert "jeff bezos" in response.lower()
    assert "who" not in response.lower()


def test_service_generate_chat_title_about_narrow_topic(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "In one sentence, describe the P/E ratio.",
            },
            {
                "role": "ai",
                "content": """
                The P/E ratio, or price-to-earnings ratio, is a measure of a company's current share price relative to its per-share earnings,
                used by investors to evaluate the value and potential growth of a company.
                """,  # noqa: E501
            },
        ]
    }

    response = test_client.post(
        "/v1/generate/chat/title",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response = response.json()

    expected_terms = ["p/e", "valuation", "price-to-earnings", "price to earnings"]
    assert any(term in response.lower() for term in expected_terms)


def test_service_generate_chat_title_about_broader_topic(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "In one sentence, describe the P/E ratio.",
            },
            {
                "role": "ai",
                "content": """
                The P/E ratio, or price-to-earnings ratio, is a measure of a company's current share price relative to its per-share earnings,
                used by investors to evaluate the value and potential growth of a company.
                """,  # noqa: E501
            },
            {
                "role": "human",
                "content": "what are other ratios to know",
            },
            {
                "role": "ai",
                "content": """
                Other important financial ratios to know include:

                Debt-to-Equity Ratio: Measures a company's financial leverage by comparing its total liabilities to its shareholder equity.
                Current Ratio: Assesses a company's ability to pay short-term obligations with its current assets.
                Return on Equity (ROE): Indicates how efficiently a company is using its equity to generate profit.
                Gross Margin Ratio: Shows the percentage of revenue that exceeds the cost of goods sold.
                Quick Ratio: Evaluates a company's ability to meet short-term liabilities with its most liquid assets.
                Price-to-Book (P/B) Ratio: Compares a company's market value to its book value.
                Earnings Per Share (EPS): Represents the portion of a company's profit allocated to each outstanding share of common stock.
                These ratios provide insights into various aspects of a company's financial health and operational efficiency.
                """,  # noqa: E501
            },
        ]
    }

    response = test_client.post(
        "/v1/generate/chat/title",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response = response.json()

    assert "ratios" in response.lower()


def test_service_generate_chat_title_about_different_field(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "how to say hello in portuguese?",
            },
            {
                "role": "ai",
                "content": "Hello in Portuguese is Ola",  # noqa: E501
            },
        ]
    }

    response = test_client.post(
        "/v1/generate/chat/title",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response = response.json()

    assert (
        "greeting" in response.lower()
        or "greetings" in response.lower()
        or "translation" in response.lower()
    )


def test_service_generate_chat_title_multiple_messages(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "what is 10 times 10",
            },
            {
                "role": "ai",
                "content": "10 times 10 is 100.",  # noqa: E501
            },
            {
                "role": "human",
                "content": "and what if you subtract 33",
            },
            {
                "role": "ai",
                "content": "If you subtract 33 from 100, the result is 67.",  # noqa: E501
            },
        ]
    }

    response = test_client.post(
        "/v1/generate/chat/title",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response = response.json()

    assert (
        "math" in response.lower()
        or "calculations" in response.lower()
        or "arithmetic" in response.lower()
    )


def test_service_generate_chat_title_get_title_from_copilot_answer(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": "who is the owner of tesla, spacex and twitter?",
            },
            {
                "role": "ai",
                "content": (
                    "Elon Musk is the owner of Tesla, SpaceX, and Twitter. He is the CEO and largest shareholder of Tesla, the founder and CEO of SpaceX, "  # noqa: E501
                    "and he acquired Twitter in October 2022."
                ),
            },
        ]
    }

    response = test_client.post(
        "/v1/generate/chat/title",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response = response.json()

    assert "elon" in response.lower()


def test_service_generate_chat_title_long_text(
    test_client: TestClient,
    mock_headers: Any,
    no_rate_limit: None,
    mock_uuids: type[MockUUIDs],
):
    payload = {
        "messages": [
            {
                "role": "human",
                "content": """Good Afternoon, and welcome to the Apple Q2 Fiscal Year 2024 Earnings Conference Call. My name is Suhasini Chandramouli, Director of Investor Relations.
                Today's call is being recorded. Speaking first today is Apple's CEO, Tim Cook, and he'll be followed by CFO, Luca Maestri. After that, we'll open the call to questions from analysts.
                Please note that some of the information you'll hear during our discussion today will consist of forward-looking statements, including, without limitation, those regarding revenue, gross margin,
                operating expenses, other income and expense, taxes, capital allocation and future business outlook, including the potential impact of macroeconomic conditions on the company's business and results of operations.
                These statements involve risks and uncertainties that may cause actual results or trends to differ materially from our forecast. For more information, please refer to the risk factors discussed in Apple's most recently filed Annual Report on Form 10-K and the Form 8-K filed with the SEC today, along with the associated press release.
                Apple assumes no obligation to update any forward-looking statements, which speak only as of the date they are made. I'd now like to turn the call over to Tim for introductory remarks.""",  # noqa: E501
            },
            {
                "role": "ai",
                "content": """
                    Good afternoon, Suhasini. Thank you for the introduction.

                    As per your request, I will summarize the key points from Apple's Q2 Fiscal Year 2024 Earnings Conference Call:

                    Introduction and Forward-Looking Statements:

                    The call includes forward-looking statements regarding various financial metrics and business outlook.
                    These statements are subject to risks and uncertainties, which could cause actual results to differ materially from forecasts.
                    For detailed risk factors, refer to Apple's most recent Annual Report on Form 10-K and the Form 8-K filed with the SEC.

                    Speakers:

                    Tim Cook, CEO of Apple, will provide introductory remarks.
                    Luca Maestri, CFO of Apple, will follow with additional details.
                    The call will then be opened to questions from analysts.

                    Key Topics:

                    Revenue
                    Gross margin
                    Operating expenses
                    Other income and expenses
                    Taxes
                    Capital allocation
                    Future business outlook, including the impact of macroeconomic conditions.

                    Obligations:

                    Apple assumes no obligation to update any forward-looking statements made during the call.
                    For more detailed information, please refer to the specific filings mentioned (Form 10-K and Form 8-K) and the associated press release.
                    """,  # noqa: E501
            },
        ]
    }

    response = test_client.post(
        "/v1/generate/chat/title",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response = response.json()

    assert "apple" in response.lower()
    assert "earnings" in response.lower()
