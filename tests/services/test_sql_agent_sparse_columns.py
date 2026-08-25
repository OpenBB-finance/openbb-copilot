"""Test SQL Agent handling of sparse columns."""

import json

import pandas as pd
import pytest
from openbb_ai.models import StatusUpdateSSE

from openbb_ada.services import SqlAgentService


@pytest.mark.asyncio
async def test_llm_peek_table_with_sparse_columns(
    test_sql_agent_service: SqlAgentService,
):
    """Test _llm_peek_table detects sparse columns that appear later in the data."""

    # Create a DataFrame with sparse columns similar to the user's example
    # First 49 rows have empty creator, traded, and date_added columns
    # Rows 50+ have populated data
    data = []
    for i in range(100):
        row = {
            "name": f"Protocol_{i}",
            "tvl": 1000000 * (i + 1),
            "change_1d": i % 10 - 5,
            "change_7d": i % 20 - 10,
        }

        # Add sparse columns - empty for first 49 rows, populated after
        if i >= 49:
            row["traded"] = "yes" if i % 2 == 0 else "no"
            row["creator"] = f"Creator_{i % 10}"
            row["date_added"] = f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
        else:
            row["traded"] = ""
            row["creator"] = ""
            row["date_added"] = ""

        data.append(row)

    df = pd.DataFrame(data)

    # Insert the table
    test_sql_agent_service.insert_table(df=df, table_name="sparse_table")

    # Call _llm_peek_table to peek at the table
    result_content = []
    artifacts = []

    async for event in test_sql_agent_service._llm_peek_table("sparse_table"):
        if isinstance(event, str):
            result_content.append(event)
        elif isinstance(event, StatusUpdateSSE):
            if event.data.artifacts:
                artifacts.extend(event.data.artifacts)

    # Check that artifacts were generated
    assert len(artifacts) > 0, "Should have generated table preview artifacts"

    # Check the table preview artifact
    table_artifact = None
    for artifact in artifacts:
        if artifact.name == "table_preview":
            table_artifact = artifact
            break

    assert table_artifact is not None, "Should have a table_preview artifact"
    assert table_artifact.type == "table"

    # Check that the sampled data includes rows with non-empty sparse columns
    content = table_artifact.content
    assert isinstance(content, list), "Content should be a list of rows"

    # Count rows with non-empty creator field
    rows_with_creator = [row for row in content if row.get("creator", "") != ""]

    # With our new sampling strategy (beginning, middle, end),
    # we should detect the sparse columns
    assert len(rows_with_creator) > 0, (
        "Should have detected rows with creator data using distributed sampling"
    )

    # Check that we sampled from different parts of the table
    # The middle and end samples should have the sparse data
    if len(content) >= 10:  # If we got enough samples
        # Last 5 rows should have creator data
        last_5_rows = content[-5:]
        creators_in_last_5 = [
            row for row in last_5_rows if row.get("creator", "") != ""
        ]
        assert len(creators_in_last_5) > 0, "End samples should have creator data"

        # Middle samples should also have creator data (since middle is around row 50)
        if len(content) >= 15:
            middle_rows = content[5:10]  # Middle 5 rows
            creators_in_middle = [
                row for row in middle_rows if row.get("creator", "") != ""
            ]
            assert len(creators_in_middle) > 0, (
                "Middle samples should have creator data"
            )


@pytest.mark.asyncio
async def test_llm_peek_table_small_table(
    test_sql_agent_service: SqlAgentService,
):
    """Test that _llm_peek_table returns all rows for small tables."""

    # Create a small DataFrame with only 20 rows
    df = pd.DataFrame(
        {
            "id": range(20),
            "name": [f"Item_{i}" for i in range(20)],
            "value": range(100, 120),
        }
    )

    # Insert the table
    test_sql_agent_service.insert_table(df=df, table_name="small_table")

    # Call _llm_peek_table
    artifacts = []

    async for event in test_sql_agent_service._llm_peek_table("small_table"):
        if isinstance(event, StatusUpdateSSE):
            if event.data.artifacts:
                artifacts.extend(event.data.artifacts)

    # Find the table preview artifact
    table_artifact = None
    for artifact in artifacts:
        if artifact.name == "table_preview":
            table_artifact = artifact
            break

    assert table_artifact is not None, "Should have a table_preview artifact"

    # For small tables (<=30 rows), we should get all rows
    content = table_artifact.content
    assert len(content) == 20, (
        f"Should have all 20 rows for small table, got {len(content)}"
    )


@pytest.mark.asyncio
async def test_llm_peek_table_sampling_distribution(
    test_sql_agent_service: SqlAgentService,
):
    """Test _llm_peek_table samples from beginning, middle, and end of large tables."""

    # Create a large DataFrame where we can verify the sampling
    df = pd.DataFrame(
        {
            "row_id": range(100),  # 0-99 to track which rows were sampled
            "section": ["beginning"] * 30 + ["middle"] * 40 + ["end"] * 30,
            "value": range(100, 200),
        }
    )

    # Insert the table
    test_sql_agent_service.insert_table(df=df, table_name="large_table")

    # Call _llm_peek_table
    artifacts = []

    async for event in test_sql_agent_service._llm_peek_table("large_table"):
        if isinstance(event, StatusUpdateSSE):
            if event.data.artifacts:
                artifacts.extend(event.data.artifacts)

    # Find the table preview artifact
    table_artifact = None
    for artifact in artifacts:
        if artifact.name == "table_preview":
            table_artifact = artifact
            break

    assert table_artifact is not None, "Should have a table_preview artifact"

    content = table_artifact.content
    assert len(content) == 15, (
        f"Should have 15 sampled rows for large table, got {len(content)}"
    )

    # Check that we have samples from each section
    sections = [row["section"] for row in content]
    assert "beginning" in sections, "Should have samples from beginning"
    assert "middle" in sections, "Should have samples from middle"
    assert "end" in sections, "Should have samples from end"

    # Verify distribution: first 5 from beginning, next 5 from middle, last 5 from end
    first_5_ids = [row["row_id"] for row in content[:5]]
    middle_5_ids = [row["row_id"] for row in content[5:10]]
    last_5_ids = [row["row_id"] for row in content[10:15]]

    # First 5 should be from beginning (0-4)
    assert max(first_5_ids) < 30, "First 5 samples should be from beginning"

    # Middle 5 should be from around the middle (around row 48-52)
    assert min(middle_5_ids) >= 30, "Middle samples should be from middle section"
    assert max(middle_5_ids) < 70, "Middle samples should be from middle section"

    # Last 5 should be from end (95-99)
    assert min(last_5_ids) >= 70, "Last 5 samples should be from end"


@pytest.mark.asyncio
async def test_preview_table_distributed_sampling(
    test_sql_agent_service: SqlAgentService,
):
    """Test preview_table uses distributed sampling for sparse column detection."""

    # Create DataFrame with sparse columns - same structure as the peek test
    data = []
    for i in range(100):
        row = {
            "name": f"Protocol_{i}",
            "tvl": 1000000 * (i + 1),
            "change_1d": i % 10 - 5,
            "change_7d": i % 20 - 10,
        }

        # Add sparse columns - empty for first 49 rows, populated after
        if i >= 49:
            row["traded"] = "yes" if i % 2 == 0 else "no"
            row["creator"] = f"Creator_{i % 10}"
            row["date_added"] = f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
        else:
            row["traded"] = ""
            row["creator"] = ""
            row["date_added"] = ""

        data.append(row)

    df = pd.DataFrame(data)

    # Insert the table
    test_sql_agent_service.insert_table(df=df, table_name="sparse_preview_table")

    # Call preview_table
    result = test_sql_agent_service.preview_table("sparse_preview_table")

    # The result should be JSON directly (without any prefix message)
    json_data = result.split("\n...[ANOTHER")[0]
    content = json.loads(json_data)

    # Should have 15 rows from distributed sampling
    assert len(content) == 15, f"Should have 15 sampled rows, got {len(content)}"

    # Count rows with non-empty creator field
    rows_with_creator = [row for row in content if row.get("creator", "") != ""]

    # With distributed sampling, we should detect the sparse columns
    assert len(rows_with_creator) > 0, (
        "preview_table should detect rows with creator data using distributed sampling"
    )

    # Verify we have samples from different parts of the table
    # Check for beginning samples (index < 10)
    beginning_samples = [row for row in content if row.get("index", 0) < 10]
    # Check for middle samples (index around 48-52)
    middle_samples = [row for row in content if 45 <= row.get("index", 0) <= 55]
    # Check for end samples (index >= 95)
    end_samples = [row for row in content if row.get("index", 0) >= 95]

    assert len(beginning_samples) > 0, "Should have samples from beginning"
    assert len(middle_samples) > 0, "Should have samples from middle"
    assert len(end_samples) > 0, "Should have samples from end"
