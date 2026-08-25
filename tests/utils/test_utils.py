import json
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pandas as pd
import pytest
from fastapi import HTTPException, UploadFile

from openbb_ada.errors import RetryExceededError
from openbb_ada.models import WidgetFileDetails
from openbb_ada.utils.utils import (
    extract_text_from_html,
    find_longest_common_substring_indices,
    flatten_and_format_dict,
    get_current_datetime,
    get_file_details,
    handle_duplicate_columns_names,
    rate_limit,
    retry_on_exception,
)


@pytest.mark.asyncio
async def test_get_file_details_pdf(monkeypatch):
    """Test get_file_details with PDF file."""
    # Mock UploadFile
    mock_file = Mock(spec=UploadFile)
    mock_file.content_type = "application/pdf"
    mock_file.filename = "test.pdf"
    mock_file.read = AsyncMock(return_value=b"fake pdf content")
    mock_file.seek = AsyncMock()

    # Mock Pdf class
    mock_pdf_instance = Mock()
    mock_pdf_instance.get_text.return_value = ["Page 1 content", "Page 2 content"]
    monkeypatch.setattr(
        "openbb_ada.utils.utils.Pdf", Mock(return_value=mock_pdf_instance)
    )

    result = await get_file_details(mock_file)

    # Assertions
    assert isinstance(result, WidgetFileDetails)
    assert result.filename == "test.pdf"
    assert result.widget_data == "Page 1 content\nPage 2 content"
    assert result.columns is None
    assert result.index is None


@pytest.mark.asyncio
async def test_get_file_details_csv(monkeypatch):
    """Test get_file_details with CSV file."""
    # Mock UploadFile
    mock_file = Mock(spec=UploadFile)
    mock_file.content_type = "text/csv"
    mock_file.filename = "test.csv"
    mock_file.read = AsyncMock(return_value=b"name,age\nJohn,25\nJane,30")
    mock_file.seek = AsyncMock()

    # Mock pandas read_csv
    mock_df = pd.DataFrame({"name": ["John", "Jane"], "age": [25, 30]})
    monkeypatch.setattr(
        "openbb_ada.utils.utils.pd.read_csv", Mock(return_value=mock_df)
    )

    result = await get_file_details(mock_file)

    # Assertions
    assert isinstance(result, WidgetFileDetails)
    assert result.filename == "test.csv"
    assert result.columns == "name, age"
    assert "John" in result.widget_data
    assert "Jane" in result.widget_data
    assert result.index is not None


@pytest.mark.asyncio
async def test_get_file_details_excel(monkeypatch):
    """Test get_file_details with Excel file."""
    # Mock UploadFile
    mock_file = Mock(spec=UploadFile)
    mock_file.content_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    mock_file.filename = "test.xlsx"
    mock_file.read = AsyncMock(return_value=b"fake excel content")
    mock_file.seek = AsyncMock()

    # Mock pandas read_excel
    mock_df = pd.DataFrame({"name": ["John", "Jane"], "age": [25, 30]})
    monkeypatch.setattr(
        "openbb_ada.utils.utils.pd.read_excel", Mock(return_value=mock_df)
    )

    result = await get_file_details(mock_file)

    # Assertions
    assert isinstance(result, WidgetFileDetails)
    assert result.filename == "test.xlsx"
    assert result.columns == "name, age"
    assert "John" in result.widget_data
    assert "Jane" in result.widget_data
    assert result.index is not None


@pytest.mark.asyncio
async def test_get_file_details_text():
    """Test get_file_details with text file."""
    # Mock UploadFile
    mock_file = Mock(spec=UploadFile)
    mock_file.content_type = "text/plain"
    mock_file.filename = "test.txt"
    mock_file.read = AsyncMock(return_value=b"Hello, World!")
    mock_file.seek = AsyncMock()

    result = await get_file_details(mock_file)

    # Assertions
    assert isinstance(result, WidgetFileDetails)
    assert result.filename == "test.txt"
    assert result.widget_data == "Hello, World!"
    assert result.columns is None
    assert result.index is None


@pytest.mark.asyncio
async def test_get_file_details_markdown():
    """Test get_file_details with markdown file."""
    # Mock UploadFile
    mock_file = Mock(spec=UploadFile)
    mock_file.content_type = "text/markdown"
    mock_file.filename = "test.md"
    mock_file.read = AsyncMock(return_value=b"# Header\nContent")
    mock_file.seek = AsyncMock()

    result = await get_file_details(mock_file)

    # Assertions
    assert isinstance(result, WidgetFileDetails)
    assert result.filename == "test.md"
    assert result.widget_data == "# Header\nContent"
    assert result.columns is None
    assert result.index is None


@pytest.mark.asyncio
async def test_get_file_details_html_strips_markup():
    """Test get_file_details with HTML file."""
    mock_file = Mock(spec=UploadFile)
    mock_file.content_type = "text/html"
    mock_file.filename = "test.html"
    mock_file.read = AsyncMock(
        return_value=(
            b"<html><head><style>body{color:red;}</style></head><body><h1>Title</h1>"
            b"<p>Hello <strong>World</strong></p>"
            b"<script>alert('x')</script></body></html>"
        )
    )
    mock_file.seek = AsyncMock()

    result = await get_file_details(mock_file)

    assert isinstance(result, WidgetFileDetails)
    assert result.filename == "test.html"
    assert result.widget_data == "Title\n\nHello World"
    assert "<h1>" not in result.widget_data
    assert "alert('x')" not in result.widget_data
    assert "body{color:red;}" not in result.widget_data


def test_extract_text_from_html_strips_non_content_and_preserves_structure():
    html_content = """
    <!doctype html>
    <html>
      <head>
        <title>Tesla Overview</title>
        <style>body { color: red; }</style>
        <script>console.log("ignore me")</script>
      </head>
      <body>
        <h1>Tesla Overview</h1>
        <p>Tesla, Inc. was incorporated by <strong>Martin Eberhard</strong> and
        <em>Marc Tarpenning</em> in July 2003.</p>
        <ul>
          <li>Founded in 2003.</li>
          <li>Headquartered in Austin.</li>
        </ul>
        <table>
          <tr><th>City</th><th>State</th></tr>
          <tr><td>Austin</td><td>Texas</td></tr>
        </table>
      </body>
    </html>
    """

    result = extract_text_from_html(html_content)
    normalized_result = " ".join(result.split())
    expected_summary = (
        "Tesla, Inc. was incorporated by Martin Eberhard and "
        "Marc Tarpenning in July 2003."
    )

    assert "Tesla Overview" in result
    assert expected_summary in normalized_result
    assert "- Founded in 2003." in result
    assert "- Headquartered in Austin." in result
    assert "City" in result
    assert "State" in result
    assert "Austin" in result
    assert "Texas" in result
    assert "console.log" not in result
    assert "body { color: red; }" not in result
    assert "<strong>" not in result


@pytest.mark.asyncio
async def test_get_file_details_unsupported():
    """Test get_file_details with unsupported file type raises exception."""
    # Mock UploadFile
    mock_file = Mock(spec=UploadFile)
    mock_file.content_type = "application/unknown"
    mock_file.filename = "test.unknown"
    mock_file.seek = AsyncMock()

    with pytest.raises(HTTPException):  # Assert that it raises an HTTPException
        await get_file_details(mock_file)


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.stateful
@pytest.mark.integration
@pytest.mark.asyncio
async def test_rate_limit_below_limit(
    actual_valid_access_token, actual_set_copilot_call_count, mock_request
):
    mock_request.headers.get.return_value = actual_valid_access_token
    mock_request.json = AsyncMock(
        return_value={"messages": [{"role": "human", "content": "What is 1+1?"}]}
    )

    # Set the copilot call count to 0
    _ = await actual_set_copilot_call_count(actual_valid_access_token, count=0)

    @rate_limit()
    async def test_endpoint(request, *args, **kwargs):
        return "result"

    result = await test_endpoint(request=mock_request)
    assert result == "result"


@pytest.mark.skip(reason="Integration tests are skipped and will be rethought.")
@pytest.mark.stateful
@pytest.mark.integration
@pytest.mark.asyncio
async def test_rate_limit_above_limit(
    actual_valid_access_token, actual_set_copilot_call_count, mock_request
):
    mock_request.headers.get.return_value = actual_valid_access_token
    mock_request.json = AsyncMock(
        return_value={"messages": [{"role": "human", "content": "What is 1+1?"}]}
    )

    # Set the copilot call count to exceed the limit
    _ = await actual_set_copilot_call_count(actual_valid_access_token, count=100)

    @rate_limit()
    async def test_endpoint(request, *args, **kwargs):
        raise Exception("The rate limit should prevent this from being called.")

    result = await test_endpoint(request=mock_request)
    events: list[dict[str, Any]] = [y async for y in result.body_iterator]
    assert len(events) == 1
    assert events[0]["event"] == "copilotStatusUpdate"

    event_data_payload: dict = json.loads(events[0]["data"])
    assert event_data_payload["eventType"] == "ERROR"
    assert "You've reached your daily completion limit" in event_data_payload["message"]


@pytest.mark.asyncio
async def test_handle_metadata():
    metadata = {
        "key1": "value1",
        "key2": {"key3": "value3"},
    }

    expected_result = {
        "Key1": "value1",
        "Key3": "value3",
    }
    actual_result = flatten_and_format_dict(metadata)

    assert actual_result == expected_result


def test_find_longest_common_substring_indices():
    document_words = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    target_words = ["z", "b", "c", "d", "a", "b", "e", "x", "y", "z"]
    start_idx, end_idx = find_longest_common_substring_indices(
        document_words, target_words
    )
    assert start_idx == 1
    assert end_idx == 4  # Remember, the end index is exclusive


def test_get_current_datetime():
    actual_result = get_current_datetime(timezone="America/New_York")
    assert ("EST" in actual_result) or ("EDT" in actual_result)


def test_get_current_datetime_invalid_timezone():
    actual_result = get_current_datetime(timezone="This/Is/An/Invalid/Timezone")
    assert "UTC" in actual_result


@pytest.mark.parametrize(
    "max_retries, exceptions, expected_result",
    [
        (1, (httpx.RemoteProtocolError,), "Failure"),
        (2, (httpx.RemoteProtocolError,), "Failure"),
        (3, (httpx.RemoteProtocolError,), "Success"),
    ],
)
@pytest.mark.asyncio
async def test_retry_on_exception(max_retries, exceptions, expected_result):
    @retry_on_exception(max_retries=max_retries, exceptions=exceptions)
    async def unreliable_function():
        FORCE_FAIL_BEFORE_ATTEMPTS = 3
        unreliable_function.attempts += 1
        if unreliable_function.attempts < FORCE_FAIL_BEFORE_ATTEMPTS:
            raise exceptions[0]("Test exception message")
        return "Success"

    if expected_result == "Success":
        unreliable_function.attempts = 0
        result = await unreliable_function()
        assert result == "Success"
        assert unreliable_function.attempts == 3
    else:
        unreliable_function.attempts = 0
        with pytest.raises(
            expected_exception=RetryExceededError,
            match=f"Failed after {max_retries} attempts: Test exception message",
        ):
            await unreliable_function()


def test_handle_duplicate_columns_names():
    """handle_duplicate_columns_names has unique column names for JSON conversion."""
    df = pd.DataFrame(
        [[1, 4, 7, 10, 13], [2, 5, 8, 11, 14], [3, 6, 9, 12, 15]],
        columns=["col1", "col2", "col1", "col3", "col2"],
    )

    assert df.columns.duplicated().any(), "Test DataFrame should have duplicate columns"

    fixed_df = handle_duplicate_columns_names(df)

    assert not fixed_df.columns.duplicated().any(), (
        "Fixed DataFrame should have unique columns"
    )

    json_str = fixed_df.to_json(orient="records")
    assert json_str is not None, "Should be able to convert to JSON"

    expected_columns = ["col1", "col2", "col1_1", "col3", "col2_1"]
    assert list(fixed_df.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {list(fixed_df.columns)}"
    )


def test_handle_duplicate_columns_names_no_duplicates():
    """Test that handle_duplicate_columns_names doesn't modify DataFrames
    with unique columns."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})

    fixed_df = handle_duplicate_columns_names(df)

    assert list(fixed_df.columns) == list(df.columns), (
        "Unique columns should remain unchanged"
    )

    assert fixed_df is df, "Should return same DataFrame if no duplicates"


def test_handle_duplicate_columns_names_empty_dataframe():
    """Test that handle_duplicate_columns_names handles empty DataFrames."""
    df = pd.DataFrame()
    fixed_df = handle_duplicate_columns_names(df)
    assert fixed_df is df, "Empty DataFrame should be returned as-is"
