import json
import re
import uuid
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Callable, Dict, List, Tuple, cast

import httpx
import pandas as pd
import sqlparse
from fastapi.encoders import jsonable_encoder
from magentic import (
    AssistantMessage,
    FunctionCall,
    FunctionResultMessage,
    SystemMessage,
    UserMessage,
    chatprompt,
)
from magentic.chat_model.base import StringNotAllowedError
from openbb_ai.models import ChartParameters, StatusUpdateSSE, StatusUpdateSSEData
from pydantic import AliasChoices, Field
from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.schema import CreateTable

from ..constants import OPENBB_AGENT_MODEL_MAIN
from ..errors import SqlAgentError
from ..models import (
    ClientArtifact,
    CopilotArtifact,
    RawObjectDataFormat,
    SourceInfo,
    SqlAgentQueryResult,
    SqlTableInfo,
)
from ..utils.ai import get_llm
from ..utils.chart_generation import generate_chart_parameters
from ..utils.child_table_schema_builder import ChildTableSchemaBuilder
from ..utils.nested_data_normalizer import normalize_nested_data
from ..utils.utils import (
    detect_nested_data_columns,
    handle_datetime_columns,
    handle_df_orientation,
    handle_duplicate_columns_names,
    handle_nested_hashmaps,
    instrument_async_generator,
    retry_on_exception,
    sanitize_str,
)
from ._logging import LoggingService
from .template import TemplateService


class SqlAgentService:
    def __init__(
        self,
        template_service: TemplateService,
        logging_service: LoggingService,
        openai_api_key: str | None,
        engine: Engine | None = None,
    ):
        """Initialize SQL Agent Service.

        SQLAlchemy Engine Ownership Model:
        - If engine is None: Service creates and owns the engine
            (service responsible for cleanup)
        - If engine is provided: Service borrows the engine
            (caller responsible for cleanup)
        """
        self._template_service = template_service
        self._logging_service = logging_service
        self._openai_api_key = openai_api_key

        self._tables: list[SqlTableInfo] = []
        self._tables_metadata_path = None
        self._inserted_source_uuids: set[str] = set()  # Track inserted source UUIDs

        # Determine ownership: service owns engine only if it creates it
        if engine is None:
            self._engine = create_engine("sqlite:///:memory:")
            self._owns_engine = True
        else:
            self._engine = engine
            self._owns_engine = False

        self.final_query_result = None

    def _get_model(self, **kwargs):
        return get_llm(
            model=OPENBB_AGENT_MODEL_MAIN,
            temperature=0.1,
            api_key=self._openai_api_key,
            **kwargs,
        )

    def _load_table_metadata_if_exists(self, tables_metadata_path: Path):
        if tables_metadata_path.exists():
            with tables_metadata_path.open("r") as f:
                loaded_data = json.load(f)
                self._tables = [
                    SqlTableInfo(**table_info) for table_info in loaded_data
                ]

    def _write_table_metadata(self, tables_metadata_path: Path):
        self._logging_service.info(
            "Writing table metadata to disk: %s", tables_metadata_path
        )
        with open(tables_metadata_path, "w") as f:
            json.dump(jsonable_encoder(self._tables), f)

    def __del__(self):
        """Clean up owned resources.

        Only disposes the engine if this service created it (owns it).
        If the engine was injected via dependency injection, the caller
        is responsible for cleanup.

        Note:
          __del__ is called at unpredictable times and may not be called at all.
          Logging errors in the destructor may fail if logging service is unavailable
          (during interpreter shutdown), so we explicitly fall back to failing silently.
        """
        # Only dispose if we own the engine
        if (
            hasattr(self, "_owns_engine")
            and self._owns_engine
            and hasattr(self, "_engine")
            and self._engine is not None
        ):
            try:
                self._engine.dispose()
            except Exception as err:
                try:
                    if hasattr(self, "_logging_service") and self._logging_service:
                        self._logging_service.warning(
                            "Error during SQL engine cleanup (non-critical): %s", err
                        )
                except Exception:  # noqa: S110
                    pass  # Failing silently during interpreter shutdown

    @staticmethod
    def _is_sqlite_internal_table(table_name: str) -> bool:
        """Check if a table is a SQLite internal table that should be filtered."""
        return table_name.startswith("sqlite_")

    @staticmethod
    def _is_derived_artifact_table(table_name: str) -> bool:
        """Check if a table is a derived artifact from previous query results.

        Artifact tables are generated from SQL query results and should not be
        used as source data for subsequent queries. The LLM should always query
        the original source tables to get complete data.
        """
        artifact_prefixes = (
            "table_artifact_",
            "chart_artifact_",
        )
        return table_name.startswith(artifact_prefixes)

    def get_sql_tables_info(self) -> list[SqlTableInfo]:
        """Get all user tables, filtering out SQLite internal tables."""
        user_tables = [
            table
            for table in self._tables
            if not self._is_sqlite_internal_table(table.table_name)
        ]
        if len(self._tables) != len(user_tables):
            self._logging_service.info(
                "Filtered %d SQLite internal tables out of %d tables, "
                "returning %d user tables",
                len(self._tables) - len(user_tables),
                len(self._tables),
                len(user_tables),
            )
        return user_tables

    def get_sql_table_info(self, table_name: str) -> SqlTableInfo:
        """Get table info for a specific table.

        This includes internal SQLite tables if the callers explicitly ask for them.
        """
        for table in self._tables:
            if table.table_name == table_name:
                return table
        raise ValueError(f"The requested table does not exist: {table_name}")

    def get_table_schema(self, table_name: str) -> str:
        metadata = MetaData()
        metadata.reflect(bind=self._engine)

        table = metadata.tables.get(table_name)
        if table is not None:
            create_statement = str(
                CreateTable(table).compile(dialect=self._engine.dialect)
            )
            return create_statement
        else:
            raise ValueError(f"The requested table schema does not exist: {table_name}")

    def format_table_name(self, name: str) -> str:
        # Keep letters, numbers, and underscores; remove everything else
        # and convert to lowercase
        cleaned_name = name.replace(" ", "_")
        sql_name = re.sub(r"[^\w]", "", cleaned_name).lower()

        # Ensure the name is unique
        if sql_name in [table.table_name for table in self._tables]:
            return sql_name + "_" + str(uuid.uuid4())[-5:]
        return sql_name

    def insert_table_with_all_infos(
        self,
        df: pd.DataFrame,
        table_name: str,
        description: str | None = None,
        metadata: dict | None = None,
        source_uuid: str | None = None,
    ) -> list[SqlTableInfo]:
        """
        Insert a DataFrame into SQLite with automatic nested data normalization.
        Returns all table infos (parent + children).
        """
        # Check if we've already inserted data from this source UUID
        if source_uuid and source_uuid in self._inserted_source_uuids:
            self._logging_service.info(
                "Data from source UUID %s already inserted, skipping", source_uuid[:8]
            )
            # Return existing table info for this source
            for existing_table in self._tables:
                if (
                    existing_table.metadata
                    and existing_table.metadata.get("source_uuid") == source_uuid
                ):
                    return [existing_table]

        table_name = self.format_table_name(table_name)
        df = self._prepare_dataframe_for_sql(df)

        try:
            # Detect and normalize nested data
            has_nested_data, parent_df, child_dataframes, nested_columns = (
                self._detect_and_normalize_nested_data(df, metadata)
            )

            # If no nested data, use standard insertion
            if not has_nested_data:
                return [
                    self._insert_standard_table(df, table_name, description, metadata)
                ]

            # Insert all data transactionally
            table_infos = self._insert_normalized_data(
                parent_df,
                child_dataframes,
                table_name,
                nested_columns,
                self._get_literal_columns_and_values,
                self.get_table_schema,
            )

            # Add source UUID to metadata and track it
            if source_uuid:
                if not metadata:
                    metadata = {}
                metadata["source_uuid"] = source_uuid
                self._inserted_source_uuids.add(source_uuid)

                # Update the table infos with the source UUID
                for table_info in table_infos:
                    if table_info.metadata:
                        table_info.metadata["source_uuid"] = source_uuid
                    else:
                        table_info.metadata = {"source_uuid": source_uuid}

            # Add all tables to the service's tables list
            for sql_table_info in table_infos:
                self._tables.append(sql_table_info)

            # Only store to disk if we have an on-disk database
            if self._tables_metadata_path is not None:
                self._write_table_metadata(self._tables_metadata_path)

            self._logging_service.info(
                "Successfully inserted normalized data for %s with %d tables",
                table_name,
                len(table_infos),
            )

            # Return all table infos (parent + children for nested data)
            return table_infos

        except Exception as e:
            # Final fallback to standard table insertion
            self._logging_service.warning(
                "Nested data processing failed for table %s: %s. "
                "Using standard insertion.",
                table_name,
                str(e),
            )
            if source_uuid:
                if not metadata:
                    metadata = {}
                metadata["source_uuid"] = source_uuid
                self._inserted_source_uuids.add(source_uuid)

            return [self._insert_standard_table(df, table_name, description, metadata)]

    def _generate_enhanced_description(
        self,
        table_name: str,
        description: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Generate an enhanced description combining
        widget info and metadata context."""
        base_description = description or table_name

        if not metadata:
            return base_description

        enhanced_parts = [base_description]

        # Add filtering information
        if "filtered_by" in metadata and metadata["filtered_by"]:
            # Ensure it's a list of strings for safe joining
            filter_list = metadata["filtered_by"]
            if isinstance(filter_list, (list, tuple)):
                filters = ", ".join(str(f) for f in filter_list)
                enhanced_parts.append(f"filtered for {filters}")

        # Add grouping information
        if "grouped_by" in metadata and metadata["grouped_by"]:
            # Ensure it's a list of strings for safe joining
            group_list = metadata["grouped_by"]
            if isinstance(group_list, (list, tuple)):
                groups = ", ".join(str(g) for g in group_list)
                enhanced_parts.append(f"grouped by {groups}")

        # Add data scope information
        if "row_count_range" in metadata:
            enhanced_parts.append(f"({metadata['row_count_range']})")

        # Add display name if different from base
        if (
            "table_display_name" in metadata
            and metadata["table_display_name"] != table_name
        ):
            enhanced_parts.insert(1, f"from '{metadata['table_display_name']}'")

        return " - ".join(enhanced_parts)

    def insert_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        description: str | None = None,
        metadata: dict | None = None,
    ) -> SqlTableInfo:
        """
        Insert a DataFrame into SQLite with automatic nested data normalization.

        This method:
        1. Detects nested data columns (arrays/objects) via metadata hints or
           auto-inspection
        2. Normalizes nested data into separate child tables using
           group_id references
        3. Preserves original table structure with group_id references
           instead of nested data
        4. Inserts all data transactionally for consistency
        5. Falls back gracefully to standard processing if nested data
           handling fails

        Args:
            df: DataFrame to insert
            table_name: Name for the SQL table
            description: Optional description
            metadata: Optional metadata with column hints

        Returns:
            SqlTableInfo for the parent table
        """
        table_name = self.format_table_name(table_name)
        df = self._prepare_dataframe_for_sql(df)

        try:
            # Detect and normalize nested data
            has_nested_data, parent_df, child_dataframes, nested_columns = (
                self._detect_and_normalize_nested_data(df, metadata)
            )

            # If no nested data, use standard insertion
            if not has_nested_data:
                return self._insert_standard_table(
                    df, table_name, description, metadata
                )

            # Insert all data transactionally
            table_infos = self._insert_normalized_data(
                parent_df,
                child_dataframes,
                table_name,
                nested_columns,
                self._get_literal_columns_and_values,
                self.get_table_schema,
            )

            # Add all tables to the service's tables list
            for sql_table_info in table_infos:
                self._tables.append(sql_table_info)

            # Only store to disk if we have an on-disk database
            if self._tables_metadata_path is not None:
                self._write_table_metadata(self._tables_metadata_path)

            self._logging_service.info(
                "Successfully inserted normalized data for %s with %d tables",
                table_name,
                len(table_infos),
            )

            # Return main table (parent + children for nested data)
            return table_infos[0]

        except Exception as e:
            # Fallback to standard table insertion
            self._logging_service.warning(
                "Nested data processing failed for table %s: %s. "
                "Using standard insertion.",
                table_name,
                str(e),
            )
            return self._insert_standard_table(df, table_name, description, metadata)

    def _insert_standard_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        description: str | None = None,
        metadata: dict | None = None,
    ) -> SqlTableInfo:
        """
        Insert a DataFrame using standard processing (no nested data handling).

        This is the fallback method when nested data processing fails.
        """
        # Apply standard data processing
        self._logging_service.info("Inserting standard table %s", table_name)

        # Convert column names to lowercase
        df.columns = pd.Index([str(col).lower() for col in df.columns])

        # Apply standard transformations
        df = handle_nested_hashmaps(df=df)
        df = handle_df_orientation(df=df)
        df = handle_datetime_columns(df=df)
        unique_column_values = self._get_literal_columns_and_values(df=df)

        try:
            with self._engine.connect() as conn:
                df.to_sql(table_name, conn, index=True, if_exists="replace")
        except (ProgrammingError, OperationalError) as err:
            self._logging_service.warning("Unable to create table: %s", err)
            raise RuntimeError(f"Unable to create table: {err}") from err

        # Generate enhanced description combining widget info and metadata
        enhanced_description = self._generate_enhanced_description(
            table_name=table_name, description=description, metadata=metadata
        )

        self._logging_service.info(
            "Enhanced table description: original='%s' -> enhanced='%s'",
            description,
            enhanced_description,
        )

        sql_table_info = SqlTableInfo(
            table_name=table_name,
            sql_schema=self.get_table_schema(table_name),
            unique_column_values=unique_column_values,
            description=enhanced_description,
            metadata=metadata,
        )

        # Track source UUID if provided
        if metadata and "source_uuid" in metadata:
            self._inserted_source_uuids.add(metadata["source_uuid"])

        self._tables.append(sql_table_info)

        # Only store to disk if we have an on-disk database
        if self._tables_metadata_path is not None:
            self._write_table_metadata(self._tables_metadata_path)

        self._logging_service.info(
            "Inserted standard table %s with columns %s",
            sql_table_info.table_name,
            df.columns,
        )
        return sql_table_info

    def preview_table(self, table_name: str) -> str:
        for table in self._tables:
            if table.table_name == table_name:
                try:
                    num_rows = pd.read_sql_query(
                        f"select count(*) from {table_name}",  # noqa: S608
                        self._engine,
                    ).iloc[0, 0]

                    num_rows = cast(int, num_rows)

                    if num_rows <= 30:
                        df = pd.read_sql_query(
                            f"select * from {table_name}",  # noqa: S608
                            self._engine,
                        )
                    else:
                        # Sample from beginning, middle, and end of table
                        # Get first 5 rows
                        df_head = pd.read_sql_query(
                            f"select * from {table_name} limit 5",  # noqa: S608
                            self._engine,
                        )

                        # Get 5 rows from middle
                        middle_offset = num_rows // 2 - 2
                        df_middle = pd.read_sql_query(
                            (
                                f"select * from {table_name} "  # noqa: S608
                                f"limit 5 offset {middle_offset}"
                            ),
                            self._engine,
                        )

                        # Get last 5 rows
                        tail_offset = max(0, num_rows - 5)
                        df_tail = pd.read_sql_query(
                            f"select * from {table_name} limit 5 offset {tail_offset}",  # noqa: S608
                            self._engine,
                        )

                        # Combine samples
                        df = pd.concat([df_head, df_middle, df_tail], ignore_index=True)

                    df = handle_duplicate_columns_names(df)

                    json_data = df.to_json(orient="records", date_format="iso")
                    if num_rows > 30:
                        remaining = num_rows - 15
                        json_data += f"\n...[ANOTHER {remaining} ROWS]..."
                    return json_data
                except OperationalError as err:
                    raise SqlAgentError(f"Unable to peek table: {err}") from err

        return ""

    @instrument_async_generator("SqlAgentService.query")
    async def query(
        self,
        query: str,
    ) -> AsyncGenerator[SqlAgentQueryResult | StatusUpdateSSE, None]:
        self._logging_service.debug("=== SQL AGENT QUERY START ===")
        self._logging_service.info("Query: %s", query)
        self._logging_service.debug(
            "Available user tables: %s",
            [t.table_name for t in self.get_sql_tables_info()],
        )

        # # Enhanced logging for tables in context
        # # I'm leaving this because it is really nice to debug the SQL
        # self._logging_service.info("=== SQL AGENT TABLES IN CONTEXT ===")
        # if not self._tables:
        #     self._logging_service.info("No tables available in context")
        # else:
        #     self._logging_service.info(
        #         "Available tables: %s", [table.table_name for table in self._tables]
        #     )
        #     for table in self._tables:
        #         self._logging_service.info("Table: %s", table.table_name)
        #         if table.description:
        #             self._logging_service.info("  Description: %s", table.description)
        #         if table.metadata:
        #             self._logging_service.info("  Metadata: %s", table.metadata)
        #         self._logging_service.info("  Schema: %s", table.sql_schema)
        #         if table.unique_column_values:
        #             self._logging_service.info(
        #                 "  Unique values: %s", table.unique_column_values
        #             )
        # self._logging_service.info("=== END SQL AGENT TABLES ===")

        # # Print actual table data in memory before any queries
        # print("=== ACTUAL TABLE DATA IN MEMORY ===")
        # if not self._tables:
        #     print("No tables available in memory")
        # else:
        #     for table in self._tables:
        #         print(f"\n--- TABLE: {table.table_name} ---")
        #         try:
        #             # Get the actual data from the database
        #             df = pd.read_sql_query(
        #                 f"SELECT * FROM {table.table_name} LIMIT 20", self._engine
        #             )
        #             print(f"Shape: {df.shape}")
        #             print(f"Columns: {list(df.columns)}")
        #             print("First 20 rows:")
        #             print(df.to_string())
        #         except Exception as e:
        #             print(f"Error reading table {table.table_name}: {e}")
        # print("=== END ACTUAL TABLE DATA ===")

        self.main_copilot_query = query

        # Filter out derived artifact tables from previous query results.
        # The LLM should always query original source tables to get complete data,
        # not partial results stored in artifact tables.
        source_tables = [
            t for t in self._tables if not self._is_derived_artifact_table(t.table_name)
        ]

        messages: list[
            SystemMessage | UserMessage | AssistantMessage | FunctionResultMessage
        ] = [
            SystemMessage(
                sanitize_str(
                    self._template_service.render_copilot_sql_agent_system_prompt(
                        sql_tables_info=source_tables,
                    )
                )
            ),
            UserMessage("{query}"),
        ]
        MAX_CALLS = 20
        call_count = 0
        response = None
        early_exit = False
        while call_count < MAX_CALLS:
            call_count += 1

            @retry_on_exception(max_retries=3, exceptions=(httpx.RemoteProtocolError,))
            @chatprompt(
                *messages,
                functions=[
                    self._llm_peek_table,
                    self._llm_peek_column_unique_values,
                    self._llm_execute_sql,
                    self._llm_complete,
                ],
                model=self._get_model(),
            )
            async def _llm(query: str) -> FunctionCall: ...  # type: ignore[empty-body]

            self._logging_service.debug("Begin loop.")
            self._logging_service.debug("Executing LLM call...")
            try:
                response = await _llm(query=query)
            except (StringNotAllowedError, ValueError) as err:
                self._logging_service.warning(str(err))
                continue
            self._logging_service.debug("Received LLM response.")
            if isinstance(response, FunctionCall):
                self._logging_service.debug(
                    "LLM response is a function call: %s", response
                )
                try:
                    async for event in response():
                        if response._function.__name__ == self._llm_complete.__name__:
                            self._logging_service.debug(
                                "`complete` function called. Yielding...",
                            )
                            yield event
                            early_exit = True
                        elif isinstance(event, StatusUpdateSSE):
                            yield event
                        else:
                            messages.append(AssistantMessage(response))
                            messages.append(
                                FunctionResultMessage(
                                    content=sanitize_str(str(event)),
                                    function_call=response,
                                )
                            )

                    if early_exit:
                        break

                except SqlAgentError as err:
                    self._logging_service.warning("SQL query execution failed: %s", err)
                    messages.append(AssistantMessage(response))
                    messages.append(
                        FunctionResultMessage(
                            content=sanitize_str(str(err)),
                            function_call=response,
                        )
                    )
                    self._logging_service.debug("Retrying...")
                    continue
                self._logging_service.debug("End loop.")
        if not early_exit:
            self._logging_service.warning(
                "LLM failed to complete query after %s attempts...exiting.", MAX_CALLS
            )
            yield SqlAgentQueryResult(
                answer="I was unable to complete the query after reaching my iteration limit.",  # noqa: E501
                queried_tables=[],
            )

    def _raise_empty_query_result_error(self, sql_query: str) -> None:
        error_message = f"""
Your SQL query returned no data: {sql_query}

Here are some suggestions to help you fix this:
1. If you are filtering on a particular field, write a query that relaxes the constraints temporarily to see if you get any data.
2. Or, you can try getting the unique values for the field you are filtering on to see what values are available.
3. If you are filtering on a date range, consider using the most recent date range available in the data. In this case YOU MUST REPORT THAT YOU USED DIFFERENT DATE IN YOUR FINAL ANSWER BECAUSE DATA WASN'T AVAILABLE.
4. You can try peek the tables again to get an idea of the data available.
"""  # noqa: E501
        raise SqlAgentError(error_message)

    def _raise_empty_query_result_error_with_ssm(self, sql_query: str) -> None:
        error_message = f"""
Your query returned no data on the current page: {sql_query}

Here are some suggestions to help you fix this:
1. If you are filtering on a particular field, write a query that relaxes the constraints temporarily to see if you get any data.
2. Or, you can try getting the unique values for the field you are filtering on to see what values are available.
3. If you are filtering on a date range, consider using the most recent date range available in the data. In this case YOU MUST REPORT THAT YOU USED DIFFERENT DATE IN YOUR FINAL ANSWER BECAUSE DATA WASN'T AVAILABLE.
4. You can try peek the tables again to get an idea of the data available.
"""  # noqa: E501
        raise SqlAgentError(error_message)

    async def _llm_complete(
        self,
        # AliasChoices accommodates models that emit PascalCase or space-separated
        # parameter names instead of snake_case when invoking tool calls.
        final_answer: Annotated[
            str,
            Field(
                validation_alias=AliasChoices(
                    "final_answer", "Final Answer", "FinalAnswer"
                )
            ),
        ],
        final_sql_query: Annotated[
            str | None,
            Field(
                validation_alias=AliasChoices(
                    "final_sql_query",
                    "Final Sql Query",
                    "Final SQL Query",
                    "FinalSqlQuery",
                )
            ),
        ],
        create_artifact: Annotated[
            bool,
            Field(
                validation_alias=AliasChoices(
                    "create_artifact", "Create Artifact", "CreateArtifact"
                )
            ),
        ] = True,
        queried_tables: Annotated[
            list[str] | None,
            Field(
                validation_alias=AliasChoices(
                    "queried_tables", "Queried Tables", "QueriedTables"
                )
            ),
        ] = None,
        return_chart: Annotated[
            bool,
            Field(
                validation_alias=AliasChoices(
                    "return_chart", "Return Chart", "ReturnChart"
                )
            ),
        ] = False,
        summary: Annotated[
            str,
            Field(validation_alias=AliasChoices("summary", "Summary")),
        ] = "Providing final answer",
    ) -> AsyncGenerator[SqlAgentQueryResult | StatusUpdateSSE, None]:
        """Provide your final answer.

        Parameters
        ----------
        final_answer : str
            The final answer to the query.
        final_sql_query : str or None
            The final SQL query you executed.
            If None, no SQL query was executed.
        queried_tables : list[str] or None
            The list of tables you queried.
            If None, no tables were queried.
        return_chart : bool
            If True, the final answer will be returned as a chart artifact.
            If False, the final answer will be returned as a table artifact.
        create_artifact : bool
            If True, create an artifact from final_sql_query. Defaults to True
            for backward compatibility with existing tool calls.
            If False, do not create an artifact. Use False when no user-facing
            table or chart can be produced, or when final_sql_query was only
            used for diagnosis or schema inspection.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.

        Important
        ---------
        When calling this function, use the exact snake_case parameter names
        shown above (for example ``final_answer`` and ``final_sql_query``).
        """
        # Log and print which tables were actually used
        self._logging_service.debug("=== SQL AGENT TABLES ACTUALLY USED ===")
        if queried_tables:
            self._logging_service.debug("Tables used in query: %s", queried_tables)
            for table_name in queried_tables:
                try:
                    table_info = self.get_sql_table_info(table_name)
                    self._logging_service.debug(
                        "  - %s: %s",
                        table_name,
                        table_info.description or "No description",
                    )
                except ValueError:
                    self._logging_service.warning("  - %s: Table not found", table_name)
        else:
            self._logging_service.debug("No tables were used in the query")
        self._logging_service.debug("=== END TABLES USED ===")

        self._logging_service.info(
            "Completing query with final answer: %s", final_answer
        )
        if not create_artifact:
            self._logging_service.info(
                "Skipping final SQL artifact because create_artifact is false."
            )
            final_sql_query = None
        elif final_sql_query and self._sql_query_is_metadata_inspection(
            final_sql_query
        ):
            self._logging_service.info(
                "Skipping final SQL artifact because the query inspects metadata "
                "instead of returning user data."
            )
            final_sql_query = None
        elif final_sql_query and not self._sql_query_references_source_table(
            final_sql_query
        ):
            self._logging_service.info(
                "Skipping final SQL artifact because the query does not reference "
                "a loaded source table."
            )
            final_sql_query = None

        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                artifacts=[],
            )
        )
        artifact = None
        if final_sql_query:
            final_query_result = None
            self._logging_service.info("Creating result artifact...")
            try:
                final_query_result = self._execute_sql_query(final_sql_query)
                if len(final_query_result) == 0:
                    self._logging_service.warning(
                        "Query returned no data for artifact creation: %s. "
                        "Proceeding without artifact.",
                        final_sql_query,
                    )
                    final_query_result = None

                if final_query_result is not None:
                    final_query_result = handle_duplicate_columns_names(
                        final_query_result
                    )

            except (OperationalError, ProgrammingError) as err:
                raise SqlAgentError(f"Unable to execute SQL query: {err}") from err
            if final_query_result is not None and return_chart:
                chart_params: ChartParameters = await self._generate_chart_params(
                    final_query_result=final_query_result,
                    main_query=self.main_copilot_query,
                )

                artifact = CopilotArtifact(
                    data_format=RawObjectDataFormat(
                        parse_as="chart",
                        chart_params=chart_params,
                    ),
                    content=json.loads(
                        final_query_result.to_json(orient="records", date_format="iso")
                    ),
                    source_info=SourceInfo(
                        type="artifact",
                        uuid=uuid.uuid4(),
                        name=f"chart_artifact_{str(uuid.uuid4())[:5]}",
                        description=final_answer,
                        metadata={
                            "query": self._format_sql_query(final_sql_query),
                        },
                        citable=False,
                    ),
                )
            elif final_query_result is not None:
                artifact = CopilotArtifact(
                    data_format=RawObjectDataFormat(
                        parse_as="table",
                    ),
                    content=json.loads(
                        final_query_result.to_json(orient="records", date_format="iso")
                    ),
                    source_info=SourceInfo(
                        type="artifact",
                        uuid=uuid.uuid4(),
                        name=f"table_artifact_{str(uuid.uuid4())[:5]}",
                        description=final_answer,
                        metadata={
                            "query": self._format_sql_query(final_sql_query),
                            "parse_as": "table",
                        },
                        citable=False,
                    ),
                )
        else:
            self._logging_service.info("No SQL query received to create artifact.")
        yield SqlAgentQueryResult(
            answer=final_answer,
            queried_tables=queried_tables if queried_tables else [],
            artifact=artifact,
        )

    def _sql_query_references_source_table(self, sql_query: str) -> bool:
        source_table_names = {
            table.table_name.lower()
            for table in self.get_sql_tables_info()
            if not self._is_derived_artifact_table(table.table_name)
        }
        if not source_table_names:
            return False

        parsed_statements = sqlparse.parse(sql_query)
        query_tokens = {
            token.value.strip('"`[]').lower()
            for statement in parsed_statements
            for token in statement.flatten()
        }
        return bool(source_table_names.intersection(query_tokens))

    @staticmethod
    def _sql_query_is_metadata_inspection(sql_query: str) -> bool:
        parsed_statements = sqlparse.parse(sql_query)
        query_tokens = {
            token.value.strip('"`[]').lower()
            for statement in parsed_statements
            for token in statement.flatten()
        }
        metadata_tokens = {
            "pragma",
            "sqlite_master",
            "sqlite_schema",
            "information_schema",
            "typeof",
        }
        return bool(metadata_tokens.intersection(query_tokens))

    async def _generate_chart_params(
        self,
        final_query_result: pd.DataFrame,
        main_query: str,
    ) -> ChartParameters:
        """Convert a table artifact to a chart artifact."""
        final_query_result = handle_duplicate_columns_names(final_query_result)

        # Get chart parameters directly from LLM
        return await generate_chart_parameters(
            data=final_query_result,
            template_service=self._template_service,
            requested_chart_type=None,  # Let LLM decide based on context
            context=main_query,  # Pass the query as context for chart type hints
        )

    async def _llm_peek_table(
        self, table_name: str, summary: str = "Peeking table"
    ) -> AsyncGenerator[str | StatusUpdateSSE, None]:
        """Peek a sample of rows from a table to get an idea of the data.

        Samples rows from throughout the table to avoid missing sparse columns.

        Parameters
        ----------
        table_name : str
            The name of the table to peek.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """
        # Guard against peeking SQLite internal tables
        if self._is_sqlite_internal_table(table_name):
            error_msg = (
                f"Cannot peek SQLite internal table: {table_name}\n\n"
                "You should not attempt to peek or query SQLite system tables like:\n"
                "  - sqlite_master\n"
                "  - sqlite_sequence\n"
                "  - sqlite_stat1\n\n"
                "These are metadata tables managed by SQLite itself.\n"
                "Focus on the user tables provided in your context."
            )
            self._logging_service.warning(
                "SQL Agent attempted to peek SQLite internal table: %s", table_name
            )
            raise SqlAgentError(error_msg)

        self._logging_service.info("Peeking table: %s", table_name)
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                artifacts=[],
            )
        )
        try:
            # Get total row count
            num_rows = pd.read_sql_query(
                f"select count(*) from {table_name}",  # noqa: S608
                self._engine,
            ).iloc[0, 0]

            # Sample rows from throughout the table to catch sparse columns
            if num_rows <= 30:
                # If table is small, just get all rows
                df = pd.read_sql_query(f"select * from {table_name}", self._engine)  # noqa: S608
                result = f"Previewing all {num_rows} rows."
            else:
                # Sample from beginning, middle, and end of table
                # Get first 5 rows
                df_head = pd.read_sql_query(
                    f"select * from {table_name} limit 5",  # noqa: S608
                    self._engine,
                )

                # Get 5 rows from middle
                middle_offset = num_rows // 2 - 2
                df_middle = pd.read_sql_query(
                    f"select * from {table_name} limit 5 offset {middle_offset}",  # noqa: S608
                    self._engine,
                )

                # Get last 5 rows
                tail_offset = max(0, num_rows - 5)
                df_tail = pd.read_sql_query(
                    f"select * from {table_name} limit 5 offset {tail_offset}",  # noqa: S608
                    self._engine,
                )

                # Combine samples
                df = pd.concat([df_head, df_middle, df_tail], ignore_index=True)
                result = (
                    f"Previewing 15 sampled rows (from beginning, middle, and end) "
                    f"out of {num_rows} total rows."
                )

            df = handle_duplicate_columns_names(df)

            # Generate artifacts from the peeked table data
            content = json.loads(df.to_json(orient="records"))
            if content:
                yield StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="INFO",
                        message="Table preview artifact generated",
                        artifacts=[
                            ClientArtifact(
                                name="table_preview",
                                description=f"Preview of sampled rows from {table_name} (beginning, middle, and end)",  # noqa: E501
                                type="table",
                                content=content,
                            )
                        ],
                    )
                )
            else:
                yield StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="INFO",
                        message="Table preview artifact generated",
                        artifacts=[],
                    )
                )

            result = df.to_json(orient="records")
            num_rows = cast(int, num_rows)
            remaining = num_rows - 15
            result += f"\n\n...[ANOTHER {remaining} ROWS]...\n\n"
            self._logging_service.debug("Peeked table: %s ...", result[:1000])
            yield result
        except OperationalError as err:
            self._logging_service.error("Unable to peek table: %s", err)
            raise SqlAgentError(f"Unable to peek table: {err}") from err

    async def _llm_peek_column_unique_values(
        self, table_name: str, column_name: str, summary: str = "Getting unique values"
    ) -> AsyncGenerator[str | StatusUpdateSSE, None]:
        """Peek unique values in a column.

        Use this function when you need to filter on a categorical column (like
        'metric', 'category', 'type') and want to see all available values before
        writing a WHERE clause. This is essential because table peek only shows
        a sample of rows and may not include all unique values.

        Parameters
        ----------
        table_name : str
            The name of the table to query.
        column_name : str
            The name of the column to get unique values for.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """
        if self._is_sqlite_internal_table(table_name):
            error_msg = f"Cannot query SQLite internal table: {table_name}"
            self._logging_service.warning(error_msg)
            raise SqlAgentError(error_msg)

        self._logging_service.info(
            "Getting unique values for column '%s' in table '%s'",
            column_name,
            table_name,
        )
        yield StatusUpdateSSE(
            data=StatusUpdateSSEData(
                eventType="INFO",
                message=summary,
                artifacts=[],
            )
        )
        try:
            # Get total count first
            total_count = pd.read_sql_query(
                f"SELECT COUNT(DISTINCT {column_name}) as cnt FROM {table_name}",  # noqa: S608
                self._engine,
            ).iloc[0, 0]

            # Get limited unique values (max 50)
            df = pd.read_sql_query(
                f"SELECT DISTINCT {column_name} FROM {table_name} "  # noqa: S608
                f"ORDER BY {column_name} LIMIT 50",
                self._engine,
            )
            unique_values = df[column_name].tolist()

            if total_count > 50:
                result = (
                    f"Unique values in column '{column_name}' "
                    f"(showing 50 of {total_count} total): {unique_values}"
                )
            else:
                result = f"Unique values in column '{column_name}': {unique_values}"
            self._logging_service.debug("Unique values result: %s", result[:500])
            yield result
        except OperationalError as err:
            self._logging_service.error("Unable to get unique values: %s", err)
            raise SqlAgentError(f"Unable to get unique values: {err}") from err

    async def _llm_execute_sql(
        self,
        sql_query: str,
        queried_tables: list[str] | None = None,
        summary: str = "Executing SQL query",
    ) -> AsyncGenerator[StatusUpdateSSE | str, None]:
        """Execute a SQL query.

        Parameters
        ----------
        sql_query : str
            The SQL query to execute.
            The query should be a valid SQL query that can be executed on the database.
        queried_tables : list[str] or None
            The list of tables you queried.
            If None, no tables were queried.
        summary : str
            Very short summary of the action you are taking.
            The sentence should start with a gerund (3 to 5 words maximum).
            Do not include a period at the end of the sentence.
        """
        try:
            # Guard against Snowflake-style SQL that won't work in SQLite
            sql_upper = sql_query.strip().upper()
            if sql_upper.startswith("SHOW "):
                error_msg = (
                    f"Invalid SQL for SQLite: '{sql_query[:50]}...'\n\n"  # noqa: S608
                    "You are working with a SQLite database, not Snowflake.\n"
                    "Instead of 'SHOW DATABASES' or 'SHOW TABLES', use:\n"
                    "  - SELECT name FROM sqlite_master WHERE type='table'\n"
                    "  - PRAGMA database_list\n"
                    "  - PRAGMA table_info(table_name)"
                )
                self._logging_service.warning(
                    "SQL Agent attempted Snowflake-style query: %s", sql_query[:100]
                )
                raise SqlAgentError(error_msg)

            self._logging_service.info("Executing query: %s", sql_query)
            df = self._execute_sql_query(sql_query)
            if len(df) == 0:
                ssm_request_in_query = False
                if queried_tables:
                    for table_name in queried_tables:
                        try:
                            table_info = self.get_sql_table_info(table_name)
                            if (
                                table_info.metadata
                                and "ssmRequest" in table_info.metadata
                            ):
                                ssm_request_in_query = True
                                break
                        except ValueError:
                            continue
                if ssm_request_in_query:
                    self._raise_empty_query_result_error_with_ssm(sql_query)
                else:
                    self._raise_empty_query_result_error(sql_query)

            df = handle_duplicate_columns_names(df)

            if df.shape[0] > 100:
                head_str: str = df.head(50).to_json(orient="records", date_format="iso")
                tail_str: str = df.tail(50).to_json(orient="records", date_format="iso")
                # Result too large to display in full. Showing first and last 50 rows.
                result_str = (
                    "Result too large to display in full. "
                    "Showing first and last 50 rows."
                )
                result_str += (
                    f"{head_str}\n...[ANOTHER {df.shape[0] - 100} ROWS]...\n{tail_str}"
                )
            else:
                result_str = df.to_json(orient="records", date_format="iso")

            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="SQL query executed",
                    details=[self._format_sql_query(sql_query)],
                    artifacts=[],
                )
            )

            self._logging_service.debug("Query result: %s ...", result_str[:1000])
            yield result_str
        except (OperationalError, ProgrammingError) as err:
            self._logging_service.warning(
                "Unable to execute SQL query. Responding to LLM: %s", err
            )
            raise SqlAgentError(f"Unable to execute SQL query: {err}") from err

    def _execute_sql_query(self, sql_query: str) -> pd.DataFrame:
        df = pd.read_sql_query(sql_query, self._engine)
        return df

    @staticmethod
    def _prepare_dataframe_for_sql(df: pd.DataFrame) -> pd.DataFrame:
        """Sanitize column names and resolve index conflicts before SQL insertion."""
        sql_df = df.copy()

        cleaned_columns: list[str] = []
        duplicates: dict[str, int] = {}
        for column in sql_df.columns:
            cleaned = SqlAgentService._sanitize_column_name(column)
            if cleaned in duplicates:
                duplicates[cleaned] += 1
                cleaned = f"{cleaned}_{duplicates[cleaned]}"
            else:
                duplicates[cleaned] = 1
            cleaned_columns.append(cleaned)
        sql_df.columns = cleaned_columns

        index_candidates = [col for col in sql_df.columns if col.lower() == "index"]
        if index_candidates:
            index_col = index_candidates[0]
            series = sql_df[index_col]
            if series.is_unique and not series.isnull().all():
                sql_df = sql_df.set_index(index_col, drop=True)
                sql_df.index.name = "index"
            else:
                safe_name = "index_value"
                suffix = 1
                while safe_name in sql_df.columns:
                    suffix += 1
                    safe_name = f"index_value_{suffix}"
                sql_df = sql_df.rename(columns={index_col: safe_name})

        return sql_df

    @staticmethod
    def _sanitize_column_name(name: Any) -> str:
        """Sanitize column names for machine readability and SQL compatibility."""
        cleaned = str(name).strip()
        # Remove all types of quotes
        cleaned = cleaned.replace('"', "").replace("'", "")
        # Replace spaces and special characters with underscores
        cleaned = re.sub(r"[^\w]+", "_", cleaned)
        # Remove leading/trailing underscores and reduce multiple underscores
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        # Ensure it doesn't start with a number (SQL compatibility)
        if cleaned and cleaned[0].isdigit():
            cleaned = f"col_{cleaned}"
        return cleaned if cleaned else "column_0"

    @staticmethod
    def _get_literal_columns_and_values(df: pd.DataFrame) -> dict[str, list[str]]:
        unique_column_values = {}
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    unique_values = df[col].unique()
                    if len(unique_values) < 10:
                        unique_column_values[col] = unique_values.tolist()
                except TypeError:  # noqa: E501 Not all types are hashable (eg. lists), which is required for unique()
                    pass
        return unique_column_values

    @staticmethod
    def _format_sql_query(sql_query: str) -> str:
        """
        Format SQL query with proper indentation, spacing, and newlines
        to make it more readable in SQL editor style.

        Args:
            sql_query: Raw SQL query string

        Returns:
            Formatted SQL query string with proper indentation and newlines
        """
        try:
            # Use basic sqlparse formatting first
            formatted_query = sqlparse.format(
                sql_query,
                reindent=True,
                keyword_case="upper",
                identifier_case="lower",
                strip_comments=False,
                use_space_around_operators=True,
            )
            return f"```sql\n{formatted_query}\n```"

        except Exception:
            # If formatting fails, return the original query wrapped in code block
            return f"```sql\n{sql_query}\n```"

    def _detect_and_normalize_nested_data(
        self, df: pd.DataFrame, metadata: dict | None = None
    ) -> Tuple[bool, pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, dict]]:
        """
        Detect nested columns and normalize data if needed.

        Returns:
            Tuple of (has_nested_data, parent_df, child_dataframes, nested_columns)
        """
        nested_columns = detect_nested_data_columns(df, metadata)

        if not nested_columns:
            return False, df, {}, {}

        # Normalize nested data recursively into parent + child tables
        result = normalize_nested_data(df, nested_columns)
        if len(result) == 3:
            parent_df, child_dataframes, _ = result
        else:
            parent_df, child_dataframes = result

        return True, parent_df, child_dataframes, nested_columns

    def _create_child_table_schema(
        self,
        child_df: pd.DataFrame,
        parent_table_name: str,
        column_name: str,
        nested_info: dict,
    ) -> str:
        """Generate CREATE TABLE statement for a child table."""
        builder = ChildTableSchemaBuilder(child_df, parent_table_name, column_name)

        return (
            builder.add_primary_key()
            .add_group_references()
            .add_sequence_column(nested_info)
            .add_data_columns(nested_info)
            .build()
        )

    def _generate_child_table_name(
        self, parent_table_name: str, column_name: str
    ) -> str:
        """Generate a unique child table name for a nested column."""
        return ChildTableSchemaBuilder._generate_table_name(
            parent_table_name, column_name
        )

    def _insert_normalized_data(
        self,
        parent_df: pd.DataFrame,
        child_dataframes: Dict[str, pd.DataFrame],
        parent_table_name: str,
        nested_columns: Dict[str, dict],
        get_literal_columns_and_values_func: Callable,
        get_table_schema_func: Callable,
    ) -> List[SqlTableInfo]:
        """Insert parent and child data in a transaction."""

        # Store table names and schemas for later SqlTableInfo creation
        created_tables = []

        with self._engine.connect() as conn:
            # Start transaction
            trans = conn.begin()
            try:
                parent_df.to_sql(
                    parent_table_name, conn, index=True, if_exists="replace"
                )

                created_tables.append(
                    {
                        "name": parent_table_name,
                        "type": "parent",
                        "dataframe": parent_df,
                        "description": ("Main table with nested data normalized"),
                        "metadata": {
                            "table_type": "parent",
                            "child_tables": list(child_dataframes.keys()),
                            "normalization_approach": "group_id_references",
                        },
                    }
                )

                # Insert child tables with normalized data
                for col_name, child_df in child_dataframes.items():
                    child_table_name = self._generate_child_table_name(
                        parent_table_name, col_name
                    )

                    # Create copy to avoid modifying original DataFrame
                    child_df_copy = child_df.copy()

                    # Map hierarchical table names back to original columns for metadata
                    original_col_name = col_name
                    nested_column_info = nested_columns.get(col_name)

                    if not nested_column_info:
                        # Handle case-insensitive matching for hierarchical names
                        for base_col_name, col_info in nested_columns.items():
                            if col_name.lower().startswith(base_col_name.lower()):
                                original_col_name = base_col_name
                                nested_column_info = col_info
                                break

                    # If we still don't have metadata, create a default one
                    if nested_column_info is None:
                        nested_column_info = {
                            "data_type": "array",  # Default assumption
                            "detected_via": "recursive_normalization",
                        }

                    child_schema = self._create_child_table_schema(
                        child_df_copy,
                        parent_table_name,
                        col_name,
                        nested_column_info,
                    )
                    conn.execute(text(child_schema))

                    child_df_copy.to_sql(
                        child_table_name, conn, index=False, if_exists="append"
                    )

                    created_tables.append(
                        {
                            "name": child_table_name,
                            "type": "child",
                            "dataframe": child_df_copy,
                            "schema": child_schema,
                            "description": (
                                f"Child table for {original_col_name} nested data "
                                f"from {parent_table_name}"
                            ),
                            "metadata": {
                                "table_type": "child",
                                "parent_table": parent_table_name,
                                "parent_column": original_col_name,
                                "nested_data_type": nested_column_info.get(
                                    "data_type", "array"
                                ),
                                "reference_type": "group_id",
                                "hierarchical_table_name": col_name,  # Store full name
                                "original_column_name": original_col_name,  # Tech name
                                "display_name": original_col_name,  # Use raw name
                            },
                        }
                    )

                # Commit transaction
                trans.commit()

            except Exception as e:
                # Rollback on error
                trans.rollback()
                self._logging_service.error(
                    "Transaction failed during nested data insertion: %s", e
                )
                raise RuntimeError(f"Failed to insert normalized data: {e}") from e

        # Create SqlTableInfo objects with schema retrieval
        table_infos = []
        for table_info in created_tables:
            try:
                # Get schema dynamically for parent, use pre-generated for child
                if table_info["type"] == "parent":
                    sql_schema = get_table_schema_func(table_info["name"])
                else:
                    sql_schema = table_info["schema"]

                sql_table_info = SqlTableInfo(
                    table_name=table_info["name"],
                    sql_schema=sql_schema,
                    unique_column_values=get_literal_columns_and_values_func(
                        table_info["dataframe"]
                    ),
                    description=table_info["description"],
                    metadata=table_info["metadata"],
                )
                table_infos.append(sql_table_info)

            except Exception as e:
                self._logging_service.error(
                    "Failed to create SqlTableInfo for table %s: %s",
                    table_info["name"],
                    e,
                )
                # Continue with other tables even if one fails
                continue

        return table_infos

    def _get_table_relationships(self, tables: List[SqlTableInfo]) -> Dict[str, Dict]:
        """
        Get information about table relationships for normalized data.
        This helps the AI understand how to write proper JOIN queries.
        """
        relationships: Dict[str, Dict[str, Any]] = {}

        for table in tables:
            if table.metadata and table.metadata.get("table_type") == "parent":
                parent_name = table.table_name
                relationships[parent_name] = {"child_tables": []}

                # Find child table information
                for child_table in tables:
                    if (
                        child_table.metadata
                        and child_table.metadata.get("table_type") == "child"
                        and child_table.metadata.get("parent_table") == parent_name
                    ):
                        original_column = child_table.metadata.get("parent_column")
                        data_type = child_table.metadata.get("nested_data_type")

                        child_info = {
                            "table_name": child_table.table_name,
                            "original_column": original_column,
                            "join_condition": (
                                f"{child_table.table_name}.group_id = "
                                f"{parent_name}.{original_column}"
                            ),
                            "data_type": data_type,
                            "description": (
                                f"Child table containing {data_type} data "
                                f"from {original_column} column"
                            ),
                        }

                        relationships[parent_name]["child_tables"].append(child_info)

        return relationships

    def _generate_query_examples(self, relationships: Dict[str, Dict]) -> List[Dict]:
        """
        Generate example queries for normalized data to help the AI understand
        how to work with parent-child relationships.
        """
        examples: list[dict] = []

        # Generate simplified query examples (one per data type)
        for parent_table, info in relationships.items():
            for child_info in info["child_tables"]:
                child_table = child_info["table_name"]
                original_column = child_info["original_column"]
                data_type = child_info["data_type"]

                if data_type == "array" and not any(
                    ex.get("data_type") == "array" for ex in examples
                ):
                    examples.append(
                        {
                            "data_type": "array",
                            "description": "Join array data with parent table",
                            "query": (
                                f"SELECT parent.*, child.sequence, child.value\n"
                                f"FROM {parent_table} parent\n"
                                f"JOIN {child_table} child ON "
                                f"child.group_id = parent.{original_column}\n"
                                f"ORDER BY parent.id, child.sequence;"
                            ),
                        }
                    )
                elif data_type == "object" and not any(
                    ex.get("data_type") == "object" for ex in examples
                ):
                    examples.append(
                        {
                            "data_type": "object",
                            "description": "Join object data with parent table",
                            "query": (
                                f"SELECT parent.*, child.key, child.value\n"
                                f"FROM {parent_table} parent\n"
                                f"JOIN {child_table} child ON "
                                f"child.group_id = parent.{original_column};"
                            ),
                        }
                    )

        return examples

    def get_table_relationships(self) -> dict[str, dict]:
        """
        Get information about table relationships for normalized data.
        This helps the AI understand how to write proper JOIN queries.
        """
        return self._get_table_relationships(self._tables)

    def get_normalized_query_examples(self) -> list[dict]:
        """
        Generate example queries for normalized data to help the AI understand
        how to work with parent-child relationships.
        """
        relationships = self.get_table_relationships()
        return self._generate_query_examples(relationships)
