from pathlib import Path

from fastapi.testclient import TestClient

from openbb_ada.dependencies import (
    get_editor_content_generation_service,
    get_sql_suggestion_service,
)
from openbb_ada.main import app
from openbb_ada.models import CodeGenerationResponse, UserFile


def test_generate_widget_title_and_description_unstructured(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    test_widget_data = """
    The earnings transcript for Apple (AAPL) for Q1 Fiscal Year 2023, reported
    on February 2, 2023, highlighted several key points regarding the company's
    performance and future outlook. Apple reported a revenue of $117.2 billion
    for the December quarter, marking a 5% decrease year-over-year. This decline
    was attributed to three main factors: foreign exchange headwinds,
    COVID-19-related challenges impacting the supply of iPhone 14 Pro and iPhone
    14 Pro Max, and a challenging macroeconomic environment. Despite these
    challenges, Apple achieved all-time revenue records in several markets,
    including Canada, Indonesia, Mexico, Spain, Turkey, and Vietnam, and
    quarterly records in Brazil and India.

    The company's product revenue was $96.4 billion, down 8% from the previous
    year, while the services segment set an all-time revenue record of $20.8
    billion, up 6% year-over-year. Apple's installed base of active devices grew
    to over 2 billion, doubling in size from 7 years ago. This growth was
    attributed to strong customer satisfaction, loyalty, and a high number of
    new customers to Apple products.

    Looking ahead to the March quarter, Apple expects its year-over-year revenue
    performance to be similar to the December quarter, with an acceleration in
    underlying business performance. Foreign exchange is anticipated to continue
    being a headwind, with a negative impact of 5 percentage points. The company
    also expects revenue growth in services, despite macroeconomic challenges in
    areas such as digital advertising and mobile gaming. For iPhone, revenue
    performance is expected to accelerate relative to the December quarter,
    while Mac and iPad revenues are projected to decline double digits
    year-over-year due to challenging comparisons and macroeconomic headwinds.

    Apple's gross margin guidance for the March quarter is between 43.5% and
    44.5%, with operating expenses expected to be between $13.7 billion and
    $13.9 billion. The company remains committed to its goal of becoming net
    cash neutral over time and declared a cash dividend of $0.23 per share for
    the February quarter.

    In summary, Apple faced several challenges in the December quarter,
    including foreign exchange headwinds and supply constraints, but achieved
    significant milestones, such as reaching an installed base of over 2 billion
    active devices. The company remains optimistic about its future performance,
    despite ongoing macroeconomic uncertainties
"""
    payload = {
        "widget_generation_request": {
            "name": "text_artifact_12345",
            "description": "The summary of the earnings transcript for Apple (AAPL) for Q1 Fiscal Year 2023 is provided below.",  # noqa: E501
            "widget_data": test_widget_data,
        }
    }
    response = test_client.post(
        "/v1/generate/widget_info",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response_json = response.json()

    assert "title" in response_json
    assert "description" in response_json
    assert "earnings" in response_json["title"].lower()
    assert (
        "aapl" in response_json["title"].lower()
        or "apple" in response_json["title"].lower()
    )
    assert "q1" in response_json["title"].lower()
    assert "2023" in response_json["title"].lower()
    assert "apple" in response_json["description"].lower()
    assert "2023" in response_json["description"].lower()


def test_generate_widget_title_and_description_structured(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    test_widget_data = '[{"date":"2024-12-03 00:00:00.000000","close":242.65},{"date":"2024-12-02 00:00:00.000000","close":239.59},{"date":"2024-11-29 00:00:00.000000","close":237.33},{"date":"2024-11-27 00:00:00.000000","close":234.93},{"date":"2024-11-26 00:00:00.000000","close":235.06},{"date":"2024-11-25 00:00:00.000000","close":232.87},{"date":"2024-11-22 00:00:00.000000","close":229.87},{"date":"2024-11-21 00:00:00.000000","close":228.52},{"date":"2024-11-20 00:00:00.000000","close":229},{"date":"2024-11-19 00:00:00.000000","close":228.28},{"date":"2024-11-18 00:00:00.000000","close":228.02},{"date":"2024-11-15 00:00:00.000000","close":225},{"date":"2024-11-14 00:00:00.000000","close":228.22},{"date":"2024-11-13 00:00:00.000000","close":225.12},{"date":"2024-11-12 00:00:00.000000","close":224.23}]'  # noqa: E501
    payload = {
        "widget_generation_request": {
            "name": "table_artifact_12345",
            "description": "The table of stock prices for Apple (AAPL) is provided below.",  # noqa: E501
            "widget_data": test_widget_data,
        }
    }

    response = test_client.post(
        "/v1/generate/widget_info",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response_json = response.json()

    assert "title" in response_json
    assert "description" in response_json
    assert any(
        term in response_json["title"].lower() for term in ["stock", "price", "closing"]
    )
    assert (
        "aapl" in response_json["title"].lower()
        or "apple" in response_json["title"].lower()
    )
    assert "prices" in response_json["description"].lower()


def test_generate_widget_title_and_description_dirty_input(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    test_widget_data = "To assess the company's liquidity position, we can calculate the current ratio and quick ratio using the data from the most recent financial statements.\\n\\n### Current Ratio\\nThe current ratio is calculated as:\\n<latex>\\n\\\\text{Current Ratio} = \\\\frac{\\\\text{Total Current Assets}}{\\\\text{Total Current Liabilities}}\\n</latex>\\n\\nFor the fiscal year ending September 30, 2023:\\n- **Total Current Assets**: \\\\$143,566,000,000\\n- **Total Current Liabilities**: \\\\$145,308,000,000\\n\\n\\\\[\\n\\\\text{Current Ratio} = \\\\frac{143,566,000,000}{145,308,000,000} \\\\approx 0.99\\n\\\\]\\n\\n### Quick Ratio\\nThe quick ratio is calculated as:\\n<latex>\\n\\\\text{Quick Ratio} = \\\\frac{\\\\text{Total Current Assets} - \\\\text{Inventory}}{\\\\text{Total Current Liabilities}}\\n</latex>\\n\\nFor the fiscal year ending September 30, 2023:\\n- **Total Current Assets**: \\\\$143,566,000,000\\n- **Inventory**: \\\\$6,331,000,000\\n- **Total Current Liabilities**: \\\\$145,308,000,000\\n\\n\\\\[\\n\\\\text{Quick Ratio} = \\\\frac{143,566,000,000 - 6,331,000,000}{145,308,000,000} \\\\approx 0.95\\n\\\\]\\n\\n### Analysis\\n- **Current Ratio**: A current ratio of 0.99 indicates that the company has almost enough current assets to cover its current liabilities, but it is slightly below the ideal ratio of 1.0 or higher, which suggests a potential concern in meeting short-term obligations.\\n- **Quick Ratio**: A quick ratio of 0.95, which excludes inventory, is also below 1.0. This indicates that the company might face challenges in covering its short-term liabilities without relying on the sale of inventory.\\n\\n### Conclusion\\nBoth the current and quick ratios are slightly below 1.0, which could be a concern regarding the company's short-term solvency. It suggests that the company may not have sufficient liquid assets to cover its short-term liabilities comfortably. This warrants closer monitoring of the company's liquidity management and cash flow strategies.\\n\\n<citation>widget:109bcabb-c9f1-4d10-8696-d4871af5395c</citation>"  # noqa: E501
    payload = {
        "widget_generation_request": {
            "name": "Liquidity Analysis",
            "widget_data": test_widget_data,
        }
    }

    response = test_client.post(
        "/v1/generate/widget_info",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 200

    response_json = response.json()

    assert "title" in response_json
    assert "description" in response_json
    assert "liquidity" in response_json["title"].lower()
    assert "ratio" in response_json["description"].lower()


def test_generate_widget_title_and_description_file_pdf(
    test_client: TestClient,
    test_pdf_openbb_story_user_file: UserFile,
    mock_headers: dict,
    no_rate_limit: None,
):
    test_file_path = Path(__file__).parent / "test_data" / "openbb_story.pdf"
    test_file = [("file", ("openbb_story.pdf", open(test_file_path, "rb")))]

    response = test_client.post(
        "/v1/generate/widget_info/file",
        headers=mock_headers,
        files=test_file,
    )

    assert response.status_code == 200

    response_json = response.json()
    assert "title" in response_json
    assert "description" in response_json
    assert any(
        [
            e in response_json["title"].lower()
            for e in ["openbb", "announcement", "launch"]
        ]
    )
    assert any(
        [
            e in response_json["description"].lower()
            for e in ["software", "investment", "research", "open-source"]
        ]
    )


def test_generate_widget_title_and_description_file_csv(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    test_file_path = Path(__file__).parent / "test_data" / "tsla_historical.csv"
    test_file = [("file", ("tsla_historical.csv", open(test_file_path, "rb")))]

    response = test_client.post(
        "/v1/generate/widget_info/file",
        headers=mock_headers,
        files=test_file,
    )

    assert response.status_code == 200
    response_json = response.json()

    assert "title" in response_json
    assert "description" in response_json
    assert any([e in response_json["title"].lower() for e in ["tesla", "tsla"]])
    assert any(
        [e in response_json["description"].lower() for e in ["tesla", "tsla", "stock"]]
    )
    assert any(
        [
            e in response_json["title"].lower()
            for e in ["stock", "data", "tsla", "historical"]
        ]
    )
    assert any(
        [
            e in response_json["description"].lower()
            for e in ["open", "high", "low", "close", "historical", "stock", "tesla"]
        ]
    )


def test_generate_widget_title_and_description_file_xlsx(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    test_file_path = Path(__file__).parent / "test_data" / "management_team_comp.xlsx"
    test_file = [("file", ("management_team_comp.xlsx", open(test_file_path, "rb")))]

    response = test_client.post(
        "/v1/generate/widget_info/file",
        headers=mock_headers,
        files=test_file,
    )

    assert response.status_code == 200

    response_json = response.json()
    expected_in_title = "compensation"
    expected_in_description = "compensation"
    expected_at_least_one_in_title = [
        "compensation",
        "executive",
        "management",
        "team",
    ]
    expected_at_least_one_in_description = [
        "compensation",
        "executive",
        "management",
        "names",
        "summary",
        "titles",
        "USD",
    ]

    assert "title" in response_json
    assert "description" in response_json
    assert expected_in_title in response_json.get("title", "").lower()
    assert expected_in_description in response_json.get("description", "").lower()
    assert any(
        [
            e in response_json.get("title", "").lower()
            for e in expected_at_least_one_in_title
        ]
    )
    assert any(
        [
            e in response_json.get("description", "").lower()
            for e in expected_at_least_one_in_description
        ]
    )


def test_generate_widget_title_and_description_file_txt(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    test_file_path = Path(__file__).parent / "test_data" / "amzn_data.txt"
    test_file = [("file", ("amzn_data.txt", open(test_file_path, "rb")))]

    response = test_client.post(
        "/v1/generate/widget_info/file",
        headers=mock_headers,
        files=test_file,
    )

    assert response.status_code == 200

    response_json = response.json()
    expected_at_least_one_in_title = ["stock", "price", "data"]
    expected_at_least_one_in_description = [
        "current",
        "stock",
        "price",
        "amazon",
    ]

    assert "title" in response_json
    assert "description" in response_json
    assert (
        "amazon" in response_json["title"].lower()
        or "amzn" in response_json["title"].lower()
    )
    assert any(
        [e in response_json["title"].lower() for e in expected_at_least_one_in_title]
    )
    assert any(
        [
            e in response_json["description"].lower()
            for e in expected_at_least_one_in_description
        ]
    )


def test_generate_widget_title_and_description_file_png(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    test_file_path = Path(__file__).parent / "test_data" / "table.png"
    test_file = [("file", ("table.png", open(test_file_path, "rb")))]

    response = test_client.post(
        "/v1/generate/widget_info/file",
        headers=mock_headers,
        files=test_file,
    )

    assert response.status_code == 200
    response_json = response.json()

    expected_at_least_one_in_title = ["revenue", "financial", "quarterly", "summary"]
    expected_at_least_one_in_description = [
        "financial",
        "data",
        "revenue",
        "company",
        "business",
    ]

    assert "title" in response_json
    assert "description" in response_json
    assert any(
        [e in response_json["title"].lower() for e in expected_at_least_one_in_title]
    )
    assert any(
        [
            e in response_json["description"].lower()
            for e in expected_at_least_one_in_description
        ]
    )
    assert "2023" in response_json["description"].lower()
    assert "2024" in response_json["description"].lower()


def test_generate_widget_title_and_description_file_jpg(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    test_file_path = Path(__file__).parent / "test_data" / "table.jpg"
    test_file = [("file", ("table.jpg", open(test_file_path, "rb")))]

    response = test_client.post(
        "/v1/generate/widget_info/file",
        headers=mock_headers,
        files=test_file,
    )

    assert response.status_code == 200

    response_json = response.json()
    expected_at_least_one_in_title = ["revenue", "financial", "quarterly", "summary"]
    expected_at_least_one_in_description = [
        "financial",
        "data",
        "revenue",
        "company",
        "business",
    ]

    assert "title" in response_json
    assert "description" in response_json
    assert any(
        [e in response_json["title"].lower() for e in expected_at_least_one_in_title]
    )
    assert any(
        [
            e in response_json["description"].lower()
            for e in expected_at_least_one_in_description
        ]
    )
    assert "2023" in response_json["description"].lower()
    assert "2024" in response_json["description"].lower()


def test_generate_widget_title_and_description_file_jpeg(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    test_file_path = Path(__file__).parent / "test_data" / "table.jpeg"
    test_file = [("file", ("table.jpeg", open(test_file_path, "rb")))]

    response = test_client.post(
        "/v1/generate/widget_info/file",
        headers=mock_headers,
        files=test_file,
    )

    assert response.status_code == 200

    response_json = response.json()
    expected_at_least_one_in_title = ["revenue", "financial", "quarterly", "summary"]
    expected_at_least_one_in_description = [
        "financial",
        "data",
        "revenue",
        "company",
        "business",
    ]

    assert "title" in response_json
    assert "description" in response_json
    assert any(
        [e in response_json["title"].lower() for e in expected_at_least_one_in_title]
    )
    assert any(
        [
            e in response_json["description"].lower()
            for e in expected_at_least_one_in_description
        ]
    )
    assert "2023" in response_json["description"].lower()
    assert "2024" in response_json["description"].lower()


def test_generate_code_route(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    class StubCodeGenerationService:
        async def generate_sql_suggestion(self, **kwargs) -> CodeGenerationResponse:
            return CodeGenerationResponse(
                generated_code="SELECT * FROM users",
            )

    app.dependency_overrides[get_sql_suggestion_service] = lambda: (
        StubCodeGenerationService()
    )
    try:
        response = test_client.post(
            "/v1/generate/code",
            headers=mock_headers,
            json={
                "widget_uuid": "widget-1",
                "user_prompt": "get all users",
                "language": "sql",
            },
        )
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["generated_code"] == "SELECT * FROM users"
    finally:
        app.dependency_overrides.pop(get_sql_suggestion_service, None)


def test_generate_code_route_handles_text_language(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    class StubEditorContentGenerationService:
        async def generate_content_suggestion(self, **kwargs) -> CodeGenerationResponse:
            return CodeGenerationResponse(
                generated_code="bye",
                generation_source="llm",
            )

    app.dependency_overrides[get_editor_content_generation_service] = (
        StubEditorContentGenerationService
    )

    try:
        response = test_client.post(
            "/v1/generate/code",
            headers=mock_headers,
            json={
                "widget_uuid": "widget-1",
                "user_prompt": "update to bye",
                "current_code": "Hello World",
                "language": "text",
            },
        )
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["generated_code"] == "bye"
        assert response_json["generation_source"] == "llm"
    finally:
        app.dependency_overrides.pop(get_editor_content_generation_service, None)


def test_generate_code_route_handles_python_language(
    test_client: TestClient, mock_headers: dict, no_rate_limit: None
):
    class StubEditorContentGenerationService:
        async def generate_content_suggestion(self, **kwargs) -> CodeGenerationResponse:
            return CodeGenerationResponse(
                generated_code="print('bye')",
                generation_source="llm",
            )

    app.dependency_overrides[get_editor_content_generation_service] = (
        StubEditorContentGenerationService
    )

    try:
        response = test_client.post(
            "/v1/generate/code",
            headers=mock_headers,
            json={
                "widget_uuid": "widget-1",
                "user_prompt": "update to bye",
                "current_code": "print('hi')",
                "language": "python",
            },
        )
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["generated_code"] == "print('bye')"
        assert response_json["generation_source"] == "llm"
    finally:
        app.dependency_overrides.pop(get_editor_content_generation_service, None)
