import pandas as pd

from openbb_ada.utils.child_table_schema_builder import ChildTableSchemaBuilder


def test_child_table_schema_builder_build():
    """Test basic schema building functionality."""
    # Create a simple DataFrame
    child_df = pd.DataFrame(
        {
            "name": ["Alice", "Bob"],
            "age": [25, 30],
            "group_path": ["1", "1"],
            "parent_path": [None, None],
        }
    )

    # Create builder
    builder = ChildTableSchemaBuilder(child_df, "parent_table", "nested_col")

    # Add columns
    builder.add_primary_key()
    builder.add_group_references()
    builder.add_data_columns({"data_type": "object"})

    # Build schema
    schema = builder.build()

    # Assert it's a CREATE TABLE statement
    assert schema.startswith("CREATE TABLE parent_table_nested_col (")
    assert "id INTEGER PRIMARY KEY" in schema
    assert "group_path TEXT NOT NULL" in schema
    assert "parent_path TEXT" in schema
    assert "name TEXT" in schema
    assert "age INTEGER" in schema


def test_generate_table_name():
    """Test table name generation."""
    child_df = pd.DataFrame({"col": [1, 2]})
    builder = ChildTableSchemaBuilder(child_df, "Test Table", "Test Column")
    assert builder.child_table_name == "test_table_test_column"
