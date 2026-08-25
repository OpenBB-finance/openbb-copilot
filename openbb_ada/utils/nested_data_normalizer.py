"""
Nested data normalization module.

This module provides a unified approach to normalizing
nested data structures (arrays and objects) in DataFrames.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable

import pandas as pd

from openbb_ada.utils.utils import (
    MAX_NORMALIZATION_DEPTH,
    analyze_data_complexity,
    analyze_dictionary_schema,
    detect_nested_data_columns,
    generate_group_path,
)


def handle_normalization_errors(
    fallback_action: str = "serialize",
    log_errors: bool = True,
    return_empty_on_error: bool = False,
):
    """Decorator to consolidate error handling in normalization operations."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    # Use logging service if available in context
                    if (
                        hasattr(args[0], "_logging_service")
                        and args[0]._logging_service
                    ):
                        args[0]._logging_service.warning(
                            f"Error in {func.__name__}: {e}. "
                            f"Using fallback: {fallback_action}"
                        )

                # Handle different fallback strategies
                if fallback_action == "serialize":
                    # Return JSON serialized version for complex data
                    if len(args) > 1 and isinstance(args[1], (dict, list)):
                        return json.dumps(args[1])
                    return None
                elif fallback_action == "empty":
                    return {} if return_empty_on_error else None
                elif fallback_action == "skip":
                    return None
                else:
                    # Re-raise if no valid fallback
                    raise

        return wrapper

    return decorator


class NormalizationMode(Enum):
    """Normalization processing mode."""

    SIMPLE = "simple"
    HIERARCHICAL = "hierarchical"


@dataclass
class NormalizationResult:
    """Result container for normalization operations."""

    parent_df: pd.DataFrame
    child_dataframes: dict[str, pd.DataFrame]
    child_table_names: dict[str, str] | None = None

    def to_tuple(self) -> tuple:
        """Convert to tuple for backward compatibility."""
        if self.child_table_names is None:
            return (self.parent_df, self.child_dataframes)
        return (self.parent_df, self.child_dataframes, self.child_table_names)


@dataclass
class ProcessedRow:
    """Container for a processed row from normalization."""

    row_data: dict[str, Any]
    needs_recursion: bool = False
    original_value: Any = None


@dataclass
class NormalizationContext:
    """Context with consolidated mode logic."""

    mode: NormalizationMode
    current_depth: int
    parent_path: str | None
    group_id_counter: int

    @classmethod
    def create(
        cls,
        nested_columns: dict,
        current_depth: int = 0,
        parent_path: str | None = None,
        group_id_counter: int = 1,
    ) -> "NormalizationContext":
        """Factory method that consolidates mode determination."""
        # Single place for mode logic
        mode = (
            NormalizationMode.HIERARCHICAL
            if cls._should_use_hierarchical(nested_columns, current_depth, parent_path)
            else NormalizationMode.SIMPLE
        )

        return cls(mode, current_depth, parent_path, group_id_counter)

    @staticmethod
    def _should_use_hierarchical(
        nested_columns: dict, current_depth: int, parent_path: str | None
    ) -> bool:
        """Consolidated hierarchical mode decision logic."""
        return any(
            [
                # Has enhanced columns with complexity metrics
                any(
                    "should_normalize" in col_info
                    for col_info in nested_columns.values()
                ),
                # Is in a recursive call (depth > 0 or has parent path)
                current_depth > 0,
                parent_path is not None,
            ]
        )

    def next_group_id(self) -> int:
        """Get next group ID and increment counter."""
        current = self.group_id_counter
        self.group_id_counter += 1
        return current

    def create_group_reference(self) -> str:
        """Create group reference based on mode."""
        group_id = self.next_group_id()
        return (
            generate_group_path(self.parent_path, group_id, ".")
            if self.mode == NormalizationMode.HIERARCHICAL
            else str(group_id)
        )

    def should_recurse(self, value: Any) -> bool:
        """Determine if value needs recursive processing."""
        return (
            self.mode == NormalizationMode.HIERARCHICAL
            and self.current_depth < MAX_NORMALIZATION_DEPTH - 1
            and isinstance(value, (dict, list))
            and self._value_should_normalize(value)
        )

    @handle_normalization_errors(fallback_action="skip", return_empty_on_error=False)
    def _value_should_normalize(self, value: Any) -> bool:
        """Check if value complexity warrants normalization."""
        complexity = analyze_data_complexity(value)
        return complexity.should_normalize()


class ColumnProcessor(ABC):
    """Abstract base class for column processors."""

    def __init__(self, context: NormalizationContext):
        self.context = context

    @abstractmethod
    def process(
        self, column_name: str, column_info: dict, parent_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Process a column and return updated parent_df and optional child_df.
        """
        pass

    def parse_json_value(self, value: Any, expected_type: type) -> Any | None:
        """Parse JSON string if needed."""
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, expected_type):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return value if isinstance(value, expected_type) else None

    def is_null_value(self, value: Any) -> bool:
        """Check if value should be considered null."""
        if isinstance(value, (list, pd.Series)):
            return value is None or (hasattr(value, "__len__") and len(value) == 0)
        return pd.isna(value) or value is None


class UnifiedColumnProcessor(ColumnProcessor):
    """Unified processor for both array and object columns."""

    def process(
        self, column_name: str, column_info: dict, parent_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        parent_df = parent_df.copy()
        child_rows: list[dict[str, Any]] = []
        data_type = column_info["data_type"]

        for row_index, nested_value in enumerate(parent_df[column_name]):
            if self.is_null_value(nested_value):
                parent_df.at[row_index, column_name] = None
                continue

            # Parse JSON if needed - unified for both types
            parsed_value = self.parse_json_value(
                nested_value, list if data_type == "array" else dict
            )
            if parsed_value is None:
                parent_df.at[row_index, column_name] = None
                continue

            # Create group reference
            group_reference = self.context.create_group_reference()

            # Process elements based on type
            processed_rows = self._process_elements(
                parsed_value, group_reference, data_type
            )
            child_rows.extend(processed_rows)

            # Update parent with reference
            parent_df.at[row_index, column_name] = group_reference

        child_df = pd.DataFrame(child_rows) if child_rows else None
        return parent_df, child_df

    def _process_elements(
        self, value: list | dict, group_reference: str, data_type: str
    ) -> list[dict[str, Any]]:
        """Unified element processing for arrays and objects."""
        if data_type == "array":
            return self._process_array_elements(value, group_reference)  # type: ignore
        else:
            return self._process_object_elements(value, group_reference)  # type: ignore

    def _process_array_elements(
        self, array_value: list, group_reference: str
    ) -> list[dict[str, Any]]:
        """Process array elements with optional dictionary flattening."""
        rows = []
        schema_analysis = analyze_dictionary_schema(array_value)
        keys_to_flatten = schema_analysis.get("keys_to_flatten", set())
        should_flatten = schema_analysis.get("should_flatten", False)

        for sequence, value in enumerate(array_value):
            row = self._create_row(
                group_reference, sequence=sequence, data_type="array"
            )

            if isinstance(value, dict) and should_flatten and keys_to_flatten:
                # Flatten dictionary keys directly into row
                row.update({k: value.get(k) for k in keys_to_flatten})
            else:
                processed = self._process_single_value(value, sequence)
                row.update(processed.row_data)
                if processed.needs_recursion:
                    row.update(
                        {
                            "_needs_recursion": True,
                            "_original_value": processed.original_value,
                        }
                    )

            rows.append(row)
        return rows

    def _process_object_elements(
        self, dict_value: dict, group_reference: str
    ) -> list[dict[str, Any]]:
        """Process object key-value pairs."""
        rows = []
        for key, value in dict_value.items():
            row = self._create_row(group_reference, key=key, data_type="object")
            processed = self._process_single_value(value, key)
            row.update(processed.row_data)

            if processed.needs_recursion:
                row.update(
                    {
                        "_needs_recursion": True,
                        "_original_value": processed.original_value,
                    }
                )

            rows.append(row)
        return rows

    def _create_row(
        self,
        group_reference: str,
        sequence: int | None = None,
        key: str | None = None,
        data_type: str | None = None,
    ) -> dict:
        """Create base row structure based on context mode and data type."""
        # Base row structure depends on normalization mode
        row: dict[str, Any] = (
            {
                "group_path": group_reference,
                "parent_path": self.context.parent_path,
            }
            if self.context.mode == NormalizationMode.HIERARCHICAL
            else {"group_id": group_reference}
        )

        # Add type-specific fields
        if data_type == "array" and sequence is not None:
            row["sequence"] = sequence  # type: ignore[assignment]
        elif data_type == "object" and key is not None:
            row["key"] = key

        return row

    def _process_single_value(self, value: Any, identifier: str | int) -> ProcessedRow:
        """Process a single value, handling recursion and serialization."""
        if self.context.should_recurse(value):
            return ProcessedRow(
                {"value": f"__NESTED__{identifier}__"},
                needs_recursion=True,
                original_value=value,
            )

        # Serialize complex objects or return simple values
        processed_value = (
            json.dumps(value) if isinstance(value, (dict, list)) else value
        )
        return ProcessedRow({"value": processed_value})


class RecursiveProcessor:
    """Handles recursive normalization of nested values."""

    def __init__(self, context: NormalizationContext):
        self.context = context

    def process_child_dataframe(
        self, child_df: pd.DataFrame, _: str | None = None
    ) -> dict[str, pd.DataFrame]:
        """Process child dataframe for recursive normalization."""
        if "_needs_recursion" not in child_df.columns:
            return {}

        needs_recursion_mask = child_df["_needs_recursion"].fillna(False)
        if not needs_recursion_mask.any():
            return {}

        recursive_tables = {}
        rows_to_process = child_df[needs_recursion_mask]

        for _, row in rows_to_process.iterrows():
            original_value = row["_original_value"]
            group_path = row.get("group_path", row.get("group_id"))

            # Create temp dataframe for recursive processing
            temp_df = (
                pd.DataFrame([original_value])
                if isinstance(original_value, dict)
                else pd.DataFrame({"nested": [original_value]})
            )

            # Detect nested columns
            nested_columns = detect_nested_data_columns(
                temp_df,
                current_depth=self.context.current_depth + 1,
                parent_path=group_path,
            )

            if not nested_columns:
                continue

            # Create new context for recursion
            child_context = NormalizationContext.create(
                nested_columns,
                current_depth=self.context.current_depth + 1,
                parent_path=group_path,
                group_id_counter=self.context.group_id_counter,
            )

            # Process recursively
            normalizer = DataNormalizer(child_context)
            result = normalizer.normalize(temp_df, nested_columns)

            # Update counter and merge results
            self.context.group_id_counter = child_context.group_id_counter
            recursive_tables.update(result.child_dataframes)

        return recursive_tables


class DataNormalizer:
    """Main normalizer that orchestrates the normalization process."""

    def __init__(self, context: NormalizationContext):
        self.context = context
        self.processor = UnifiedColumnProcessor(context)
        self.recursive_processor = RecursiveProcessor(context)

    def normalize(
        self, df: pd.DataFrame, nested_columns: dict[str, dict]
    ) -> NormalizationResult:
        """Normalize nested data in dataframe."""
        parent_df = df.copy()
        child_dataframes = {}
        child_table_names: dict[str, str] | None = (
            {} if self.context.mode == NormalizationMode.HIERARCHICAL else None
        )

        for column_name, column_info in nested_columns.items():
            should_normalize = column_info.get("should_normalize", True)

            # Skip normalization if explicitly disabled for hierarchical mode
            if (
                self.context.mode == NormalizationMode.HIERARCHICAL
                and not should_normalize
            ):
                # Serialize nested data as JSON strings
                parent_df[column_name] = parent_df[column_name].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x
                )
                continue

            # Process column with unified processor
            parent_df, child_df = self.processor.process(
                column_name, column_info, parent_df
            )

            if child_df is not None:
                # Generate table name
                table_name = self._generate_table_name(column_name)

                # Handle recursive processing if needed
                if "_needs_recursion" in child_df.columns:
                    recursive_tables = self.recursive_processor.process_child_dataframe(
                        child_df
                    )
                    child_dataframes.update(recursive_tables)

                    # Clean up temporary columns efficiently
                    temp_columns = [
                        col for col in child_df.columns if col.startswith("_")
                    ]
                    if temp_columns:
                        child_df = child_df.drop(columns=temp_columns, errors="ignore")

                # Store processed child dataframe
                child_dataframes[table_name] = child_df

                # Track table names for hierarchical mode
                if child_table_names is not None:
                    child_table_names[table_name] = column_name

        return NormalizationResult(parent_df, child_dataframes, child_table_names)

    def _generate_table_name(self, column_name: str) -> str:
        """Generate table name based on mode and depth."""
        if self.context.mode == NormalizationMode.HIERARCHICAL:
            return f"{column_name}_level_{self.context.current_depth}"
        return column_name


def normalize_nested_data(
    df: pd.DataFrame,
    nested_columns: dict[str, dict],
    current_depth: int = 0,
    parent_path: str | None = None,
    group_id_counter: int = 1,
) -> tuple:
    """
    Main entry point for nested data normalization.

    Args:
        df: DataFrame containing nested data
        nested_columns: Dictionary mapping column names to their nested data info
        current_depth: Current nesting depth for recursion control
        parent_path: Path from root for hierarchical tracking
        group_id_counter: Counter for generating unique group IDs

    Returns:
        Tuple containing (parent_df, child_dataframes) or
        (parent_df, child_dataframes, child_table_names)
    """
    # Create context with automatic mode determination
    context = NormalizationContext.create(
        nested_columns, current_depth, parent_path, group_id_counter
    )

    # Create normalizer and process
    normalizer = DataNormalizer(context)
    result = normalizer.normalize(df, nested_columns)

    return result.to_tuple()
