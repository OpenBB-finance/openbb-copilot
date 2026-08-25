import pandas as pd


class ChildTableSchemaBuilder:
    """Builder pattern for creating child table schemas with proper data types."""

    def __init__(
        self, child_df: pd.DataFrame, parent_table_name: str, column_name: str
    ):
        self.child_df = child_df
        self.parent_table_name = parent_table_name
        self.column_name = column_name
        self.child_table_name = self._generate_table_name(
            parent_table_name, column_name
        )
        self.columns: list[str] = []
        self.has_group_path = "group_path" in child_df.columns

    @staticmethod
    def _generate_table_name(parent_table_name: str, column_name: str) -> str:
        """Generate a unique child table name for a nested column."""
        return f"{parent_table_name}_{column_name}".replace(" ", "_").lower()

    def add_primary_key(self) -> "ChildTableSchemaBuilder":
        """Add primary key column."""
        self.columns.append("id INTEGER PRIMARY KEY")
        return self

    def add_group_references(self) -> "ChildTableSchemaBuilder":
        """Add group reference columns based on normalization type."""
        if self.has_group_path:
            # Recursive normalization uses hierarchical paths
            self.columns.extend(
                [
                    "group_path TEXT NOT NULL",
                    "parent_path TEXT",
                ]
            )
        else:
            # Legacy normalization uses simple group_id
            self.columns.append("group_id INTEGER NOT NULL")
        return self

    def add_sequence_column(self, nested_info: dict) -> "ChildTableSchemaBuilder":
        """Add sequence column for array data types."""
        if nested_info.get("data_type") == "array":
            self.columns.append("sequence INTEGER NOT NULL")
        return self

    def add_data_columns(self, nested_info: dict) -> "ChildTableSchemaBuilder":
        """Add data columns based on DataFrame structure."""
        # Define reserved columns based on normalization type
        reserved_columns = ["id"]
        if self.has_group_path:
            reserved_columns.extend(["group_path", "parent_path"])
        else:
            reserved_columns.append("group_id")

        # Add sequence to reserved if it's an array type
        if nested_info.get("data_type") == "array":
            reserved_columns.append("sequence")

        for col in self.child_df.columns:
            if col in reserved_columns:
                continue

            # Infer column type from data
            column_def = self._infer_column_type(col)
            self.columns.append(column_def)
        return self

    def _infer_column_type(self, col: str) -> str:
        """Infer SQL column type from pandas DataFrame column."""
        dtype = str(self.child_df[col].dtype)

        # Handle object dtype specially (might be numeric stored as object)
        if dtype == "object":
            try:
                pd.to_numeric(self.child_df[col])
                return f"{col} REAL"
            except (ValueError, TypeError):
                return f"{col} TEXT"

        # Map pandas dtypes to SQLite types
        type_mapping = {
            "int64": "INTEGER",
            "int32": "INTEGER",
            "float64": "REAL",
            "float32": "REAL",
            "bool": "INTEGER",  # SQLite doesn't have native boolean
        }

        sql_type = type_mapping.get(dtype, "TEXT")
        return f"{col} {sql_type}"

    def build(self) -> str:
        """Build the complete CREATE TABLE statement."""
        columns_str = ",\n    ".join(self.columns)
        return f"CREATE TABLE {self.child_table_name} (\n    {columns_str}\n)"
