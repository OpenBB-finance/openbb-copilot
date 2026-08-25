import json
import uuid
from typing import (
    Any,
    AsyncGenerator,
)

import pandas as pd
import tiktoken
from openbb_ai.models import (
    StatusUpdateSSE,
    StatusUpdateSSEData,
)

from ..errors import (
    StructuredContextError,
)
from ..models import (
    Citation,
    ContextStructuredQueryResult,
    ContextUnstructuredQueryResult,
    DataContent,
    ParsedContext,
    RawContext,
    RawObjectDataFormat,
    SourceInfo,
    SqlAgentQueryResult,
    StructuredContext,
    UnstructuredContext,
)
from ..utils.utils import (
    detect_nested_data_columns,
    flatten_and_format_dict,
    flatten_data,
)
from ._logging import LoggingService
from .sql_agent import SqlAgentService
from .template import TemplateService


class ContextService:
    """Handle additional Copilot context, including structured and unstructured data."""

    def __init__(
        self,
        template_service: TemplateService,
        sql_agent_service: SqlAgentService,
        logging_service: LoggingService,
    ):
        self._template_service = template_service
        self._sql_agent_service = sql_agent_service
        self._logging_service = logging_service

        self._unstructured_context: list[UnstructuredContext] = []
        self._structured_context: list[StructuredContext] = []

        # Dual-track approach: Store original column names for user display
        # Format: {table_name: {sql_column_name: original_display_name}}
        self._original_column_mappings: dict[str, dict[str, str]] = {}

    @property
    def unstructured_context(self) -> list[UnstructuredContext]:
        return self._unstructured_context

    @property
    def structured_context(self) -> list[StructuredContext]:
        return self._structured_context

    def _get_original_display_name_for_table(self, table_name: str) -> str:
        """
        Get the original display name for a table using metadata and stored mappings.

        This method first checks table metadata for relationship information,
        then falls back to stored mappings if needed.
        """
        # First, try to get the table info to check metadata
        try:
            table_info = self._sql_agent_service.get_sql_table_info(table_name)
            if table_info.metadata:
                # Use display_name from metadata if available
                if "display_name" in table_info.metadata:
                    return table_info.metadata["display_name"]

                # For child tables, use parent_column as display name
                if table_info.metadata.get("table_type") == "child":
                    parent_column = table_info.metadata.get("parent_column")
                    if parent_column:
                        return parent_column
        except ValueError:
            # Table not found in metadata, continue with mappings
            pass

        # Check if we have a direct mapping for this table
        if table_name in self._original_column_mappings:
            mappings = self._original_column_mappings[table_name]
            if mappings:
                # Return the first available original name
                return list(mappings.values())[0]

        # Fallback: try to find mapping in parent table based on metadata
        try:
            table_info = self._sql_agent_service.get_sql_table_info(table_name)
            if table_info.metadata and table_info.metadata.get("table_type") == "child":
                parent_table_name = table_info.metadata.get("parent_table")
                if (
                    parent_table_name
                    and parent_table_name in self._original_column_mappings
                ):
                    parent_mappings = self._original_column_mappings[parent_table_name]
                    parent_column = table_info.metadata.get("parent_column")

                    if parent_column:
                        # Try to find the original column name
                        parent_column_lower = parent_column.lower()
                        for sql_col, orig_col in parent_mappings.items():
                            if sql_col.lower() == parent_column_lower:
                                return orig_col

                        # If exact match not found, return the parent_column itself
                        return parent_column
        except ValueError:
            # Table not found, continue with final fallback
            pass

        # Final fallback: humanize the table name
        return table_name.replace("_", " ").title()

    def _extract_column_name_from_table(self, child_table_name: str) -> str:
        """Extract the column name from a child table using metadata."""

        try:
            # First, try to get from table metadata
            table_info = self._sql_agent_service.get_sql_table_info(child_table_name)
            if table_info.metadata:
                # For child tables, get the parent column directly
                if table_info.metadata.get("table_type") == "child":
                    parent_column = table_info.metadata.get("parent_column")
                    if parent_column:
                        return parent_column

                # Also check display_name
                display_name = table_info.metadata.get("display_name")
                if display_name:
                    return display_name
        except ValueError:
            # Table not found in metadata
            pass

        # Fallback: extract from table name structure
        # This is only used if metadata is not available
        if "_" in child_table_name:
            # Get the last meaningful part of the table name
            parts = child_table_name.split("_")
            # Skip common suffixes like "level", numbers
            meaningful_parts = []
            for part in reversed(parts):
                if part.lower() not in ["level", "0", "1", "2", "3", "4", "5"]:
                    meaningful_parts.append(part)
                    if len(meaningful_parts) >= 2:  # Usually column names are 1-2 parts
                        break

            if meaningful_parts:
                # Reverse to get original order and join
                return "_".join(reversed(meaningful_parts))

        return "nested_data"

    def preview_context(self, source_info_name: str) -> str:
        for sc in self._structured_context:
            if sc.source_info.name == source_info_name or (
                sc.source_info.metadata
                and sc.source_info.metadata.get("table_name") == source_info_name
            ):
                return self._sql_agent_service.preview_table(
                    sc.sql_table_info.table_name
                )
        for uc in self._unstructured_context:
            if uc.source_info.name == source_info_name:
                return uc.content[:256] + "..."
        return ""

    def get_full_table_data(self, source_info_name: str) -> list[dict] | None:
        """Resolve an artifact/table name to its full dataset.

        Returns the complete data as a list of dicts, or None if not found.
        """
        for sc in self._structured_context:
            if (
                sc.source_info.name == source_info_name
                or sc.sql_table_info.table_name == source_info_name
                or (
                    sc.source_info.metadata
                    and sc.source_info.metadata.get("table_name") == source_info_name
                )
            ):
                table_name = sc.sql_table_info.table_name
                df = self._sql_agent_service._execute_sql_query(
                    f"SELECT * FROM {table_name}"  # noqa: S608
                )
                return df.to_dict(orient="records")
        return None

    @staticmethod
    def _dump_roundtrip_context(parsed_context: ParsedContext) -> dict[str, Any]:
        return parsed_context.model_dump(mode="json")

    def dump_roundtrip_artifacts(self) -> dict[str, dict[str, Any]]:
        """Serialize artifacts that must survive a client request boundary."""
        artifacts: dict[str, dict[str, Any]] = {}

        for structured_context in self._structured_context:
            name = structured_context.source_info.name
            if not name or not name.startswith(("table_artifact_", "chart_artifact_")):
                continue

            data = self.get_full_table_data(name)
            if not data:
                continue

            artifacts[name] = self._dump_roundtrip_context(
                ParsedContext(
                    content=json.dumps(data),
                    source_info=structured_context.source_info,
                    data_format=structured_context.data_format,
                )
            )

        for unstructured_context in self._unstructured_context:
            name = unstructured_context.source_info.name
            metadata = unstructured_context.source_info.metadata or {}
            if not name or metadata.get("parse_as") != "snowflake_query":
                continue

            query_data_source = metadata.get("query_data_source")
            artifacts[name] = self._dump_roundtrip_context(
                ParsedContext(
                    content=unstructured_context.content,
                    source_info=unstructured_context.source_info,
                    data_format=RawObjectDataFormat(
                        parse_as="snowflake_query",
                        query_data_source=query_data_source
                        if isinstance(query_data_source, dict)
                        else None,
                    ),
                )
            )

        return artifacts

    def restore_roundtrip_artifacts(self, artifacts_data: dict[str, Any]) -> None:
        """Restore artifacts that were serialized before a client round-trip."""
        for artifact_name, artifact_payload in artifacts_data.items():
            if self.get_context_by_name(artifact_name) is not None:
                continue

            parsed_context = self._parse_roundtrip_artifact(
                artifact_name=artifact_name,
                artifact_payload=artifact_payload,
            )
            if parsed_context is None:
                continue

            self.load_context([parsed_context])

    def _parse_roundtrip_artifact(
        self,
        *,
        artifact_name: str,
        artifact_payload: Any,
    ) -> ParsedContext | None:
        if not isinstance(artifact_payload, dict):
            return None

        try:
            return ParsedContext.model_validate(artifact_payload)
        except ValueError:
            self._logging_service.warning(
                "Skipping invalid round-trip artifact payload for %s",
                artifact_name,
            )
            return None

    def load_explicit_context(self, context_elements: list[RawContext]) -> None:
        parsed_context_elements: list[ParsedContext] = []
        for raw_context in context_elements:
            if isinstance(raw_context.data, DataContent):
                for data_content_item in raw_context.data.items:
                    # Explicit context artifacts are user-provided and
                    # should always be citable (the user deliberately
                    # added them to the conversation). The non-citable
                    # logic for chart/table/html only applies to
                    # system-generated artifacts from tool calls.
                    citable = data_content_item.citable
                    data_format = (
                        data_content_item.data_format
                        if isinstance(
                            data_content_item.data_format, RawObjectDataFormat
                        )
                        else None
                    )
                    metadata = dict(raw_context.metadata or {})
                    if data_format is not None:
                        metadata["parse_as"] = data_format.parse_as
                        if data_format.query_data_source is not None:
                            metadata["query_data_source"] = (
                                data_format.query_data_source
                            )

                    parsed_context = ParsedContext(
                        content=data_content_item.content,
                        source_info=SourceInfo(
                            type="artifact",
                            uuid=raw_context.uuid,
                            name=raw_context.name,
                            description=raw_context.description,
                            metadata=metadata,
                            citable=citable,
                        ),
                        data_format=data_format,
                    )
                    parsed_context_elements.append(parsed_context)

        self.load_context(elements=parsed_context_elements)

    def get_structured_context_from_table_name(
        self, table_name: str
    ) -> StructuredContext:
        # First try to find exact match in structured context (for parent tables)
        for structured_context in self._structured_context:
            if structured_context.sql_table_info.table_name == table_name:
                return structured_context

        # If not found in structured context, try to get SqlTableInfo from self._tables
        # This handles child tables that have their own metadata with display_name
        try:
            child_table_info = self._sql_agent_service.get_sql_table_info(table_name)

            # Create a StructuredContext for the child table with its specific metadata
            child_context = StructuredContext(
                content="{}",  # Empty JSON content for child table
                sql_table_info=child_table_info,
                source_info=SourceInfo(
                    type="artifact",
                    uuid=uuid.uuid4(),  # Generate a new UUID for the child table
                    name=f"Child table: {table_name}",
                    description=f"Child table context for {table_name}",
                    metadata={},
                ),
                data_format=RawObjectDataFormat(parse_as="table"),
            )
            return child_context

        except ValueError:
            # Table not found in self._tables either, fall back to parent lookup
            pass

        # Fallback: If child table not found, try to find parent context
        # This is a last resort when metadata is not available

        # First, try to find any table with metadata that lists this as a child
        for sql_table in self._sql_agent_service.get_sql_tables_info():
            if sql_table.metadata and sql_table.metadata.get("table_type") == "parent":
                child_tables = sql_table.metadata.get("child_tables", [])
                if table_name in child_tables:
                    # Found the parent table
                    for structured_context in self._structured_context:
                        if (
                            structured_context.sql_table_info.table_name
                            == sql_table.table_name
                        ):
                            return structured_context

        # If still not found and table name suggests it might be a child
        # Try to find a parent by examining all tables
        if "__" in table_name or "_level_" in table_name:
            # Check all structured contexts to see if any could be the parent
            for structured_context in self._structured_context:
                parent_table_info = structured_context.sql_table_info
                if (
                    parent_table_info.metadata
                    and parent_table_info.metadata.get("table_type") == "parent"
                ):
                    # Check if the table name starts with the parent table name
                    if table_name.startswith(parent_table_info.table_name + "_"):
                        return structured_context

            # Last resort: progressive removal of segments (but this is fragile)
            # Only use this if no metadata is available
            potential_parents = set()
            parts = (
                table_name.split("__") if "__" in table_name else table_name.split("_")
            )
            for i in range(len(parts), 0, -1):
                potential_parents.add(
                    "__".join(parts[:i]) if "__" in table_name else "_".join(parts[:i])
                )

            # Try to find any of these potential parent tables
            for parent_candidate in potential_parents:
                for structured_context in self._structured_context:
                    if structured_context.sql_table_info.table_name == parent_candidate:
                        return structured_context

        raise ValueError(f"No structured context found for table name: {table_name}")

    @staticmethod
    def _process_df_list_columns(
        df: pd.DataFrame, max_length: int = 10
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Process DataFrame columns containing lists using new nested data normalization.

        Instead of dropping columns or converting to strings, this now:
        1. Detects nested data columns (arrays/objects)
        2. If nested data is found, returns the original DataFrame unchanged
        3. Lets the SqlAgentService.insert_table handle normalization properly
        4. Falls back to old behavior only for non-nested list columns

        This preserves nested data that should be normalized into child tables.
        """

        # First, detect if we have nested data that should be normalized
        nested_columns = detect_nested_data_columns(df)

        if nested_columns:
            # Return original DataFrame unchanged - let SqlAgentService handle it
            return df, []

        # If no nested data detected, proceed with original logic
        processed_df = df.copy()
        dropped_columns = []

        for column in processed_df.columns:
            if processed_df[column].apply(lambda x: isinstance(x, list)).any():
                if (
                    processed_df[column]
                    .apply(lambda x: len(x) if isinstance(x, list) else 0)
                    .max()
                    > max_length
                ):
                    processed_df.drop(columns=[column], inplace=True)
                    dropped_columns.append(column)
                else:
                    processed_df[column] = processed_df[column].apply(
                        lambda x: ", ".join(map(str, x)) if isinstance(x, list) else x
                    )
        return processed_df, dropped_columns

    @staticmethod
    def _is_vegalite_spec(data: Any) -> bool:
        """Check if the parsed JSON data is a Vega-Lite specification."""
        return (
            isinstance(data, dict)
            and isinstance(data.get("$schema"), str)
            and "vega-lite" in data["$schema"].lower()
        )

    @staticmethod
    def _extract_vegalite_values(spec: dict) -> list[dict]:
        """Extract the inline data values from a Vega-Lite spec.

        Handles top-level data.values and layered/concatenated specs.
        Returns an empty list if no inline values are found.
        """
        # Top-level data.values
        data_field = spec.get("data")
        if isinstance(data_field, dict):
            values = data_field.get("values")
            if isinstance(values, list) and len(values) > 0:
                return values

        # Layered specs: merge data from layers sharing the same schema
        for key in ("layer", "concat", "hconcat", "vconcat"):
            layers = spec.get(key)
            if isinstance(layers, list):
                merged: list[dict] = []
                for layer in layers:
                    if isinstance(layer, dict):
                        layer_data = layer.get("data")
                        if isinstance(layer_data, dict):
                            layer_values = layer_data.get("values")
                            if isinstance(layer_values, list):
                                merged.extend(layer_values)
                if merged:
                    return merged

        return []

    @staticmethod
    def _build_vegalite_metadata(spec: dict) -> str:
        """Build a short human-readable summary of a Vega-Lite chart spec."""
        title = spec.get("title", "Untitled")
        if isinstance(title, dict):
            title = title.get("text", "Untitled")
        mark = spec.get("mark", "unknown")
        if isinstance(mark, dict):
            mark = mark.get("type", "unknown")

        lines = [f'Vega-Lite chart: "{title}" (mark: {mark})']

        encoding = spec.get("encoding")
        if isinstance(encoding, dict):
            enc_parts = []
            for channel, enc_def in encoding.items():
                if isinstance(enc_def, dict):
                    field = enc_def.get("field", "?")
                    dtype = enc_def.get("type", "?")
                    enc_parts.append(f"{channel}={dtype}({field})")
            if enc_parts:
                lines.append(f"Encodings: {', '.join(enc_parts)}")

        return "\n".join(lines)

    def _parse_vegalite_spec(
        self, spec: dict, element: ParsedContext
    ) -> list[StructuredContext]:
        """Parse a Vega-Lite spec: extract data values as structured context
        and store chart metadata as unstructured context for LLM reasoning."""
        values = self._extract_vegalite_values(spec)
        if not values:
            raise StructuredContextError(
                "Vega-Lite spec has no inline data.values to extract."
            )

        mark = spec.get("mark", "unknown")
        if isinstance(mark, dict):
            mark = mark.get("type", "unknown")
        self._logging_service.info(
            "Detected Vega-Lite spec: title='%s', mark='%s', %d data values",
            spec.get("title", "Untitled"),
            mark,
            len(values),
        )

        # Store chart metadata as unstructured context
        metadata_text = self._build_vegalite_metadata(spec)
        self._logging_service.info("Vega-Lite chart metadata: %s", metadata_text)

        chart_metadata_source = element.source_info.model_copy(
            update={
                "uuid": uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{element.source_info.uuid}_vegalite_metadata",
                ),
                "name": f"{element.source_info.name} (chart metadata)",
            }
        )
        self._unstructured_context.append(
            UnstructuredContext(
                content=metadata_text,
                source_info=chart_metadata_source,
            )
        )

        # Re-route through the normal structured path with extracted values
        values_element = ParsedContext(
            content=json.dumps(values),
            source_info=element.source_info,
            data_format=element.data_format,
        )
        return self._try_parse_raw_element_to_table(values_element)

    def _try_parse_raw_element_to_table(
        self, element: ParsedContext
    ) -> list[StructuredContext]:
        try:
            data = json.loads(element.content)

            # Detect Vega-Lite spec and extract tabular data + chart metadata
            if self._is_vegalite_spec(data):
                return self._parse_vegalite_spec(data, element)

            # Extract original column names before processing
            original_column_names = {}
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                # Store original column names from raw data
                original_column_names = {col: col for col in data[0].keys()}

            try:
                df = pd.DataFrame(data)
                df, dropped_columns = self._process_df_list_columns(df)
            except ValueError:
                df = flatten_data(data)
                df, dropped_columns = self._process_df_list_columns(df)
        except (json.JSONDecodeError, ValueError, AttributeError) as err:
            raise StructuredContextError(err) from err

        if df.empty or not self._is_structured(df):
            raise StructuredContextError("Content better suited as unstructured.")

        try:
            table_name = element.source_info.name

            # Append input arguments to table name for uniqueness
            if (
                table_name is not None
                and element.source_info.metadata is not None
                and "input_args" in element.source_info.metadata
            ):
                input_args = element.source_info.metadata["input_args"]

                # Build suffix from all input arguments
                suffix_parts = []
                for _, value in input_args.items():
                    if value is not None and value != "":  # Skip None and empty values
                        # Truncate each value to max 20 chars and clean it
                        value_str = str(value)[:20].replace(" ", "-").replace("/", "-")
                        suffix_parts.append(value_str)

                # Join all parts with underscore and append to table name
                if suffix_parts:
                    table_name += "_" + "_".join(
                        suffix_parts[:5]
                    )  # Limit to 5 args max

            metadata = element.source_info.metadata or {}
            if dropped_columns:
                metadata["dropped_columns"] = dropped_columns
            # Add origin and widget_id for SQL query artifact generation
            if element.source_info.origin:
                metadata["origin"] = element.source_info.origin
            if element.source_info.widget_id:
                metadata["widget_id"] = element.source_info.widget_id

            sql_table_infos = self._sql_agent_service.insert_table_with_all_infos(
                df=df,
                table_name=table_name or "_table",
                description=element.source_info.description,
                metadata=metadata,
                source_uuid=str(element.source_info.uuid),
            )

            # Store original column mappings for created tables
            if original_column_names and sql_table_infos:
                for sql_table_info in sql_table_infos:
                    # Store original mappings for this table
                    table_mappings = {}

                    # For parent table, map direct columns
                    if sql_table_info == sql_table_infos[0]:  # Parent table
                        for original_col in original_column_names:
                            # Map original column name to itself (perfect 1:1 mapping)
                            table_mappings[original_col] = original_col

                    # For child tables, extract the original column name from table name
                    else:  # Child table
                        # Extract child column name from table structure
                        # e.g., "parent_table__childcol_level_0" -> "childcol" -> orig
                        if "__" in sql_table_info.table_name:
                            parts = sql_table_info.table_name.split("__")
                            if len(parts) >= 2:
                                child_part = parts[-1]
                                # Remove level suffix
                                if "_level_" in child_part:
                                    child_part = child_part.rsplit("_level_", 1)[0]

                                # Find original column name that matches this child part
                                for orig_col in original_column_names:
                                    # Case-insensitive match for tech vs display names
                                    if (
                                        child_part.lower() in orig_col.lower()
                                        or orig_col.lower() in child_part.lower()
                                    ):
                                        table_mappings[child_part] = orig_col
                                        break

                    if table_mappings:
                        self._original_column_mappings[sql_table_info.table_name] = (
                            table_mappings
                        )

        except Exception as err:
            self._logging_service.warning("Unable to create table: %s", err)
            raise StructuredContextError(err) from err
        # Handle multiple table infos (parent + children for nested data)
        structured_contexts = []

        # Create context only for the parent table to avoid duplicate citations
        # but store child table metadata for SQL queries to work properly
        if sql_table_infos:
            parent_table_info = sql_table_infos[0]
            source_info = element.source_info.model_copy()

            # Store child table names in metadata for reference but don't create
            if len(sql_table_infos) > 1:
                child_table_names = [table.table_name for table in sql_table_infos[1:]]
                metadata = source_info.metadata or {}
                metadata["child_tables"] = child_table_names
                source_info = source_info.model_copy(update={"metadata": metadata})

            # Store the table_name in metadata so artifact lookup can find it,
            # but preserve the original display name for citations
            metadata = source_info.metadata or {}
            metadata["table_name"] = parent_table_info.table_name
            source_info = source_info.model_copy(update={"metadata": metadata})

            structured_contexts.append(
                StructuredContext(
                    content=element.content,
                    sql_table_info=parent_table_info,
                    source_info=source_info,
                    data_format=RawObjectDataFormat(parse_as="table")
                    if not element.data_format
                    else element.data_format,
                )
            )

        # Always return list
        return structured_contexts

    def get_context_by_source_info_name(
        self, name: str
    ) -> StructuredContext | UnstructuredContext | None:
        for sc in self._structured_context:
            if sc.source_info.name == name:
                return sc
            # Also check table_name stored in metadata for artifact lookup
            if (
                sc.source_info.metadata
                and sc.source_info.metadata.get("table_name") == name
            ):
                return sc
        for uc in self._unstructured_context:
            if uc.source_info.name == name:
                return uc
        return None

    def get_context_by_id(
        self, uuid: str
    ) -> StructuredContext | UnstructuredContext | None:
        for unstructured_context in self._unstructured_context:
            if str(unstructured_context.source_info.uuid) == uuid:
                return unstructured_context
        for structured_context in self._structured_context:
            if str(structured_context.source_info.uuid) == uuid:
                return structured_context
        return None

    def get_context_by_name(
        self, name: str
    ) -> StructuredContext | UnstructuredContext | None:
        for unstructured_context in self._unstructured_context:
            if unstructured_context.source_info.name == name:
                return unstructured_context
        for structured_context in self._structured_context:
            if structured_context.source_info.name == name:
                return structured_context
            if (
                structured_context.source_info.metadata
                and structured_context.source_info.metadata.get("table_name") == name
            ):
                return structured_context
        return None

    def read_unstructured_context_by_ids(
        self, content_ids: list[str]
    ) -> list[ContextUnstructuredQueryResult]:
        context_lookup = {
            str(context.source_info.uuid): context
            for context in self._unstructured_context
        }

        context_unstructured_query_results: list[ContextUnstructuredQueryResult] = []

        for id_ in content_ids:
            if id_ not in context_lookup:
                raise ValueError(f"No unstructured context found with id: {id_}")

            context = context_lookup[id_]
            details = flatten_and_format_dict(
                {
                    "Data source": context.source_info.name,
                    "Origin": context.source_info.origin,
                    **(context.source_info.metadata or {}),
                }
            )
            context_unstructured_query_results.append(
                ContextUnstructuredQueryResult(
                    answer=context.content,
                    citations=[
                        Citation(
                            source_info=context.source_info,
                            details=[details],
                        )
                    ],
                )
            )

        return context_unstructured_query_results

    def _is_structured(
        self,
        df: pd.DataFrame,
        cell_length_threshold: int = 128,
        long_cells_ratio_threshold: float = 0.20,
        context_limit_threshold: int = 90_000,
    ) -> bool:
        try:
            df = df.copy()
            string_columns = df.select_dtypes(include=["object", "string"]).columns
            string_cell_count = df[string_columns].size

            # Check for nested data structures that should be processed as structured
            # even if they have long cells (normalized into child tables)
            nested_columns = detect_nested_data_columns(df)
            if nested_columns:
                self._logging_service.info(
                    "Detected nested data columns: %s. Processing as structured.",
                    list(nested_columns.keys()),
                )
                return True

            if string_cell_count == 0:
                self._logging_service.info(
                    "Checking context. Long cells count: 0, Total cells count: 0, Proportion: 0.0",  # noqa: E501
                )
                return True

            # Check if we exceed the token limit threshold.
            total_text = "\n".join(
                df[string_columns]
                .astype(str)
                .fillna("")
                .apply(lambda col: " ".join(col), axis=0)
            )
            encoding = tiktoken.get_encoding("o200k_base")
            token_count = len(encoding.encode(total_text))
            # This data is likely to exceed the token limit, so for safety we'll
            # treat is as structured. TODO: Feed unstructured context into a vectorDB
            # in future instead.
            if token_count > context_limit_threshold:
                self._logging_service.warning(
                    "Token count (%s) exceeds threshold for context, treating as structured.",  # noqa: E501
                    token_count,
                )
                return True

            # Count the number of string cells that exceed the threshold.
            long_cells_count = (
                df[string_columns]
                .apply(
                    lambda col: (
                        col.astype(str).str.len().gt(cell_length_threshold).sum()
                    ),
                    axis=0,
                )
                .sum()
            )
            self._logging_service.info(
                "Checking context. Long cells count: %s, Total cells count: %s, Proportion: %s",  # noqa: E501
                long_cells_count,
                string_cell_count,
                long_cells_count / string_cell_count,
            )
            if long_cells_count / string_cell_count > long_cells_ratio_threshold:
                self._logging_service.info(
                    "Long cells ratio exceeds threshold (%s). Returning False.",
                    long_cells_ratio_threshold,
                )
                return False
            return True

        except Exception as err:
            self._logging_service.warning("Error processing DataFrame: %s", err)
            return True

    def load_context(
        self, elements: list[ParsedContext]
    ) -> list[StructuredContext | UnstructuredContext]:
        """Load context from a list of elements and try to use them as structured."""  # noqa: E501
        loaded_context: list[StructuredContext | UnstructuredContext] = []
        for el in elements:
            try:
                structured_elements = self._try_parse_raw_element_to_table(el)
                # Handle structured contexts (always returns a list)
                for structured_element in structured_elements:
                    self._structured_context.append(structured_element)
                    loaded_context.append(structured_element)
                    self._logging_service.info(
                        "Created table: %s",
                        structured_element.sql_table_info.table_name,
                    )

            except StructuredContextError as err:
                unstructured_context = UnstructuredContext(
                    content=el.content,
                    source_info=el.source_info,
                )
                self._unstructured_context.append(unstructured_context)
                loaded_context.append(unstructured_context)
                self._logging_service.info(
                    "Unable to parse element as structured, using content as-is. Reason: %s",  # noqa: E501
                    err,
                    extra={"model_dump": f"{str(el.model_dump())[:100]}..."},
                )

        return loaded_context

    async def query_structured_context_with_natural_language(
        self, query: str
    ) -> AsyncGenerator[StatusUpdateSSE | ContextStructuredQueryResult, None]:
        self._logging_service.debug("=== STRUCTURED DATA QUERY START ===")
        self._logging_service.info("User query: %s", query)
        self._logging_service.debug(
            "This indicates global_search is ON and SQL Agent is being invoked"
        )

        # Log available tables from SQL agent
        sql_tables_info = self._sql_agent_service.get_sql_tables_info()
        self._logging_service.info(
            "SQL Agent has %d user tables available",
            len(sql_tables_info),
        )

        if not sql_tables_info:
            self._logging_service.warning(
                "No user tables available for SQL Agent. "
                "This might indicate widget data hasn't been loaded."
            )

        for table in sql_tables_info:
            self._logging_service.info(
                "  - Table: %s | Description: %s | Metadata: %s",
                table.table_name,
                table.description or "No description",
                table.metadata or {},
            )

        result: SqlAgentQueryResult | None = None
        async for event in self._sql_agent_service.query(query=query):
            if isinstance(event, StatusUpdateSSE):
                yield event
            elif isinstance(event, SqlAgentQueryResult):
                result = event

        self._logging_service.debug("=== SQL AGENT COMPLETED ===")
        if result:
            self._logging_service.info(
                "SQL agent completed and queried %d tables: %s",
                len(result.queried_tables),
                result.queried_tables,
            )

            # Log which tables are SQLite internals (which shouldn't happen)
            internal_tables = [
                t for t in result.queried_tables if t.startswith("sqlite_")
            ]
            if internal_tables:
                self._logging_service.error(
                    "SQL Agent queried SQLite internal tables (BUG): %s",
                    internal_tables,
                )
        else:
            self._logging_service.warning("SQL Agent returned no result")

        citations: list[Citation] = []

        # Group tables by parent to avoid duplicate citations
        parent_table_groups: dict[str, dict[str, Any]] = {}
        child_tables_used = set()

        for table_name in result.queried_tables if result else []:
            # Skip SQLite internal tables
            if table_name.startswith("sqlite_"):
                self._logging_service.warning(
                    "Skipping SQLite internal table in context: %s", table_name
                )
                continue

            # Get table info to check metadata
            try:
                table_info = self._sql_agent_service.get_sql_table_info(table_name)
                is_child = (
                    table_info.metadata is not None
                    and table_info.metadata.get("table_type") == "child"
                )

                if is_child:
                    # This is a child table - get its parent
                    parent_table_name = (
                        table_info.metadata.get("parent_table")
                        if table_info.metadata
                        else None
                    )
                    if parent_table_name:
                        # Get parent context
                        try:
                            parent_context = (
                                self.get_structured_context_from_table_name(
                                    parent_table_name
                                )
                            )
                        except ValueError as e:
                            self._logging_service.warning(
                                "Could not find parent context for child "
                                "table %s: %s. Skipping.",
                                table_name,
                                str(e),
                            )
                            continue

                        if parent_table_name not in parent_table_groups:
                            parent_table_groups[parent_table_name] = {
                                "context": parent_context,
                                "child_tables": [],
                            }

                        # Add this child table to the parent's list
                        parent_table_groups[parent_table_name]["child_tables"].append(
                            table_name
                        )
                        child_tables_used.add(table_name)
                else:
                    # This is a parent table (or regular table without nested data)
                    if table_name not in parent_table_groups:
                        try:
                            parent_context = (
                                self.get_structured_context_from_table_name(table_name)
                            )
                            parent_table_groups[table_name] = {
                                "context": parent_context,
                                "child_tables": [],
                            }
                        except ValueError as e:
                            self._logging_service.warning(
                                "Could not find context for table %s: %s. Skipping.",
                                table_name,
                                str(e),
                            )
                            continue
            except ValueError as e:
                # Table not found in metadata - skip it gracefully
                self._logging_service.warning(
                    "Table %s not found in SQL Agent tables: %s. Skipping citation.",
                    table_name,
                    str(e),
                )
                continue

        # Create citations for parent tables only
        for _parent_table_name, group_info in parent_table_groups.items():
            widget_context = group_info["context"]
            child_tables = group_info["child_tables"]

            # Prepare source info with nested table metadata if applicable
            source_info = widget_context.source_info.model_copy()

            if child_tables:
                # Generate user-friendly names for nested tables using original names
                nested_table_names = []
                for child_table in child_tables:
                    display_name = self._get_original_display_name_for_table(
                        child_table
                    )

                    nested_table_names.append(display_name)

                # Add nested tables info to metadata
                if source_info.metadata is None:
                    source_info.metadata = {}
                source_info.metadata["Nested tables"] = nested_table_names

            citations.append(
                Citation(
                    source_info=source_info,
                    details=[
                        flatten_and_format_dict(
                            {
                                "Source type": source_info.type,
                                "Origin": source_info.origin,
                                "Data source": source_info.name,
                                **{
                                    k: v
                                    for k, v in (source_info.metadata or {}).items()
                                    if k
                                    not in [
                                        "table_type",
                                        "parent_table",
                                        "parent_column",
                                        "child_tables",
                                        "normalization_approach",
                                        "widget_uuid",
                                        "table_name",
                                        "widget_id",
                                        "origin",
                                    ]
                                },
                            }
                        )
                    ],
                )
            )

        # Check if artifact is a single-row result (should not be included in response)
        # Single-row results have the value in the answer text already
        artifact_is_single_row = (
            result
            and result.artifact
            and isinstance(result.artifact.content, list)
            and len(result.artifact.content) <= 1
        )

        # Yield the result to Copilot's main execution loop
        # Don't include single-row artifacts - the value is in the answer
        yield ContextStructuredQueryResult(
            answer=result.answer if result else "",
            artifacts=[result.artifact]
            if result and result.artifact and not artifact_is_single_row
            else [],
            citations=citations,
        )

        # But also make sure we sent a status update to the client afterwards.
        # Yield "Artifact generated" status even for single-row (for UI feedback)
        if result and result.artifact:
            artifacts = [result.artifact]
            details = []
            for citation in citations:
                # Filter out internal table metadata fields that shouldn't be
                # shown to users
                filtered_metadata = {}
                if citation.source_info.metadata:
                    filtered_metadata = {
                        k: v
                        for k, v in citation.source_info.metadata.items()
                        if k
                        not in [
                            "table_type",
                            "parent_table",
                            "parent_column",
                            "child_tables",
                            "normalization_approach",
                            "hierarchical_table_name",
                            "original_column_name",
                            "reference_type",
                            "nesting_level",
                            "widget_uuid",
                            "table_name",
                            "widget_id",
                            "origin",
                        ]
                    }

                citation_details = {
                    "Source type": citation.source_info.type,
                    "Origin": citation.source_info.origin,
                    "Data source": citation.source_info.name,
                }

                # Handle nested tables specially
                if "Nested tables" in filtered_metadata:
                    nested_tables = filtered_metadata.pop("Nested tables")
                    if nested_tables:
                        citation_details["Nested tables"] = ", ".join(nested_tables)

                # Add remaining metadata
                citation_details.update(
                    {
                        k: v
                        for k, v in flatten_and_format_dict(filtered_metadata).items()
                    }
                )

                details.append(citation_details)
            # For single-row results, only show the status message without artifact data
            # (the value is already in the answer text)
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="Artifact generated",
                    details=[],
                    artifacts=[artifact.to_client_artifact() for artifact in artifacts]
                    if artifacts
                    else None,
                )
            )
