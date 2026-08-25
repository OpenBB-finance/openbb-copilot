"""MCP Data Service for processing Model Context Protocol responses."""

import json
import re
from collections import Counter
from typing import Any, AsyncGenerator, Literal
from uuid import UUID, uuid4

import openai
import pandas as pd
from fastapi import HTTPException
from magentic import SystemMessage, chatprompt
from openbb_ai.models import ClientArtifact, StatusUpdateSSE, StatusUpdateSSEData
from pydantic import ValidationError

from ..models import (
    RawObjectDataFormat,
    SourceInfo,
    StructuredContext,
    StructuredTableData,
)
from ..utils.ai import get_llm
from ..utils.utils import analyze_data_complexity, retry_on_exception
from ._logging import LoggingService
from .context import ContextService
from .template import TemplateService


def normalize_jsonish(value: Any) -> Any:
    """Recursively unwrap JSON-ish values.

    Handles single-element lists and double-encoded JSON strings that MCP
    tools sometimes return. For example:
      '{"success": true}'            -> {"success": True}
      '["{\\\"success\\\": true}"]'  -> {"success": True}
      '[{"success": true}]'          -> {"success": True}
      'plain text'                   -> "plain text"
    """
    while True:
        if isinstance(value, str):
            try:
                value = json.loads(value)
                continue
            except (json.JSONDecodeError, TypeError):
                return value

        if isinstance(value, list) and len(value) == 1:
            value = value[0]
            continue

        return value


class McpDataService:
    """Handle MCP data processing, artifact creation, and table conversion."""

    def __init__(
        self,
        context_service: ContextService,
        logging_service: LoggingService,
        template_service: TemplateService,
        openai_api_key: str | None,
    ):
        """Initialize MCP Data Service.

        Args:
            context_service: Context service for registering structured data
            logging_service: Logging service for debug output
            template_service: Template service for AI prompts
            openai_api_key: OpenAI API key for AI structuring
        """
        self._context_service = context_service
        self._logging_service = logging_service
        self._template_service = template_service
        self._openai_api_key = openai_api_key

    async def process_mcp_tool_response(
        self,
        tool_name: str,
        server_id: str,
        raw_content: Any,
        is_last_message: bool = True,
    ) -> AsyncGenerator[StatusUpdateSSE, None]:
        """Process MCP tool response and yield appropriate SSE events.

        Args:
            tool_name: Name of the MCP tool
            server_id: MCP server identifier
            raw_content: Raw response content from MCP tool
            is_last_message: Whether this is the last message in execution

        Yields:
            StatusUpdateSSE events with artifacts or errors
        """
        if not is_last_message or raw_content is None:
            return

        try:
            self._logging_service.debug(
                "mcp_result_processing_started",
                extra={
                    "event": "mcp_result_processing_started",
                    "server_id": server_id,
                    "tool_name": tool_name,
                    "is_last_message": is_last_message,
                    "raw_content_type": type(raw_content).__name__,
                    "raw_content_hash": self._logging_service.make_stable_hash(
                        raw_content
                    ),
                },
            )
            server_name = server_id
            actual_tool_name = tool_name
            if "_" in tool_name:
                parts = tool_name.split("_", 1)
                if len(parts) == 2:
                    server_name = parts[0]
                    actual_tool_name = parts[1]

            if self._is_mcp_error(raw_content):
                error_details = self._get_mcp_error_content(raw_content)
                self._logging_service.warning(
                    "mcp_error_payload_detected",
                    extra={
                        "event": "mcp_error_payload_detected",
                        "server_id": server_id,
                        "tool_name": tool_name,
                        "raw_content_type": type(raw_content).__name__,
                        "raw_content_hash": self._logging_service.make_stable_hash(
                            raw_content
                        ),
                        "error_details": error_details,
                    },
                )
                yield StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="ERROR",
                        message="Unexpected error",
                        details=[error_details] if error_details else [],
                        artifacts=[],
                    )
                )
                return

            original_artifact_type, original_artifact_content = (
                self._determine_mcp_artifact_type(raw_content)
            )
            original_artifact_uuid = uuid4()

            if (
                original_artifact_type == "table"
                and isinstance(original_artifact_content, list)
                and all(isinstance(row, dict) for row in original_artifact_content)
            ):
                self._register_mcp_table_as_structured_context(
                    table_data=original_artifact_content,
                    tool_name=actual_tool_name,
                    server_name=server_name,
                    artifact_uuid=original_artifact_uuid,
                )

            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="Artifact generated",
                    details=[],
                    artifacts=[
                        ClientArtifact(
                            type=original_artifact_type,
                            name=f"{server_name}_{actual_tool_name}_{str(original_artifact_uuid)[:8]}",
                            description=(
                                f"Data from MCP {server_name} {actual_tool_name}"
                            ),
                            uuid=original_artifact_uuid,
                            content=original_artifact_content,
                        )
                    ],
                )
            )

            (
                table_artifact_type,
                table_artifact_content,
            ) = await self._determine_mcp_artifact_type_with_ai(raw_content, tool_name)

            if (
                table_artifact_type == "table"
                and table_artifact_content != original_artifact_content
            ):
                yield StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="INFO",
                        message="Converting artifact to table",
                        details=[],
                        artifacts=[],
                    )
                )

                table_artifact_uuid = uuid4()

                if isinstance(table_artifact_content, list) and all(
                    isinstance(row, dict) for row in table_artifact_content
                ):
                    registered = self._register_mcp_table_as_structured_context(
                        table_data=table_artifact_content,
                        tool_name=actual_tool_name,
                        server_name=server_name,
                        artifact_uuid=table_artifact_uuid,
                    )

                    if registered:
                        self._logging_service.info(
                            "MCP table artifact registered as SQL-queryable context"
                        )

                yield StatusUpdateSSE(
                    data=StatusUpdateSSEData(
                        eventType="INFO",
                        message="Table artifact generated",
                        details=[],
                        artifacts=[
                            ClientArtifact(
                                type="table",
                                name=f"{server_name}_{actual_tool_name}_table_{str(table_artifact_uuid)[:8]}",
                                description=(
                                    f"Table view of data from MCP server "
                                    f"'{server_name}' with tool '{actual_tool_name}'"
                                ),
                                uuid=table_artifact_uuid,
                                content=table_artifact_content,
                            )
                        ],
                    )
                )

        except Exception as e:
            self._logging_service.error(
                "mcp_result_processing_failed",
                extra={
                    "event": "mcp_result_processing_failed",
                    "server_id": server_id,
                    "tool_name": tool_name,
                    "error": str(e),
                },
            )
            self._logging_service.error("Failed to process MCP response: %s", e)
            # Continue execution without artifact

    def _is_flat_array(self, arr: list) -> bool:
        """Check if array contains flat dicts (no nested objects/arrays)."""
        if not arr or not all(isinstance(item, dict) for item in arr):
            return False
        return analyze_data_complexity(arr, max_sample_size=5).max_nested_depth <= 2

    def _determine_mcp_artifact_type(
        self, raw_content
    ) -> tuple[Literal["text", "table", "chart"], Any]:
        """Determine the artifact type and properly formatted content for MCP responses.

        Returns:
            tuple: (artifact_type, artifact_content) where artifact_type is
                "text" or "table"
        """
        # Direct array of flat objects -> table
        if isinstance(raw_content, list) and self._is_flat_array(raw_content):
            return "table", raw_content

        elif isinstance(raw_content, dict):
            # Look for data/results wrapper with flat array
            for key in [
                "data",
                "results",
                "items",
                "rows",
                "records",
                "entries",
                "tables",
            ]:
                if key in raw_content and isinstance(raw_content[key], list):
                    if self._is_flat_array(raw_content[key]):
                        return "table", raw_content[key]

            # Flat dict -> single-row table
            if self._is_flat_dict(raw_content):
                return "table", [raw_content]

            # Complex nested structure -> formatted JSON
            return "text", self._format_as_json(raw_content)

        elif isinstance(raw_content, str):
            try:
                parsed_content = json.loads(raw_content)
                return self._determine_mcp_artifact_type(parsed_content)
            except (json.JSONDecodeError, TypeError):
                pass

        return "text", self._format_as_json(raw_content)

    def _format_as_json(self, obj: Any) -> str:
        """Format object as a JSON code block for clean display."""
        try:
            formatted = json.dumps(obj, indent=2, ensure_ascii=False, default=str)
            return f"```json\n{formatted}\n```"
        except (TypeError, ValueError):
            return f"```\n{str(obj)}\n```"

    async def _determine_mcp_artifact_type_with_ai(
        self, raw_content: Any, tool_name: str
    ) -> tuple[str, Any]:
        """Enhanced version that uses AI to structure complex data."""
        artifact_type, artifact_content = self._determine_mcp_artifact_type(raw_content)

        if artifact_type == "table":
            return artifact_type, artifact_content

        data_to_check = raw_content

        if isinstance(raw_content, str):
            try:
                data_to_check = json.loads(raw_content)
                self._logging_service.info(
                    f"Parsed JSON string, type: {type(data_to_check)}"
                )
            except json.JSONDecodeError:
                self._logging_service.info(
                    "Raw content is string but not valid JSON, keeping as text"
                )
                pass

        if isinstance(data_to_check, dict):
            for key in ["results", "data", "items", "rows", "records", "tables"]:
                if key in data_to_check and isinstance(data_to_check[key], list):
                    if data_to_check[key] and isinstance(data_to_check[key][0], dict):
                        self._logging_service.info(
                            f"Found table data in '{key}' field, extracting directly"
                        )
                        return "table", data_to_check[key]

        has_table_patterns = self._detect_tabular_patterns(raw_content)

        if not has_table_patterns:
            return artifact_type, artifact_content

        self._logging_service.debug(
            f"Attempting to structure MCP data from {tool_name}"
        )

        structured_data = await self._structure_mcp_data_with_ai(
            raw_data=raw_content, tool_name=tool_name
        )

        if structured_data and structured_data.rows:
            self._logging_service.debug(
                f"Successfully structured MCP data: {len(structured_data.rows)} rows"
            )
            return "table", structured_data.rows

        return artifact_type, artifact_content

    async def _structure_mcp_data_with_ai(
        self,
        raw_data: Any,
        tool_name: str,
    ) -> StructuredTableData | None:
        """Use AI to convert messy MCP data into structured table format."""
        self._logging_service.debug(
            f"Starting AI structuring for {tool_name}, raw_data type: {type(raw_data)}"
        )

        if not isinstance(raw_data, str):
            raw_str = (
                json.dumps(raw_data)
                if isinstance(raw_data, (dict, list))
                else str(raw_data)
            )
        else:
            raw_str = raw_data

        self._logging_service.debug(
            f"Raw data preview (first 500 chars): {raw_str[:500]}"
        )

        try:

            @retry_on_exception(
                max_retries=2,
                exceptions=(openai.APIError, ValidationError, HTTPException),
            )
            @chatprompt(
                SystemMessage(
                    self._template_service.render_mcp_to_table(
                        tool_name=tool_name, raw_output=raw_str
                    )
                ),
                model=get_llm(
                    model="gpt-4o-mini",
                    temperature=0.0,
                    api_key=self._openai_api_key,
                ),
            )
            async def _structure_data() -> StructuredTableData:  # type: ignore[empty-body]
                ...

            result = await _structure_data()

            self._logging_service.debug(
                f"Successfully structured {len(result.rows)} rows "
                f"with {len(result.column_order)} ordered columns"
            )
            return result

        except (openai.APIError, ValidationError) as e:
            self._logging_service.warning(
                f"Failed to structure MCP data with AI due to API/validation error: {e}"
            )
            return None
        except Exception as e:
            self._logging_service.error(f"Unexpected error structuring MCP data: {e}")
            return None

    def _detect_tabular_patterns(self, raw_content: Any) -> bool:
        """Detect if content likely contains tabular data using improved heuristics."""
        if isinstance(raw_content, str):
            json_patterns = any(
                pattern in raw_content
                for pattern in [
                    "[{",  # Array of objects
                    "[]",
                    '"results":',
                    '"data":',
                    '"items":',
                    '"rows":',
                    '"records":',
                ]
            )

            if json_patterns:
                return True

            return self._has_repeated_keywords(raw_content)

        elif isinstance(raw_content, list):
            if len(raw_content) < 2:
                return False

            sample_items = raw_content[: min(5, len(raw_content))]

            if not sample_items or not all(
                isinstance(item, dict) for item in sample_items
            ):
                return False

            first_keys = set(sample_items[0].keys())

            if not first_keys:
                return False

            consistent_keys = all(
                len(set(item.keys()).symmetric_difference(first_keys)) / len(first_keys)
                < 0.3
                for item in sample_items
            )

            return consistent_keys and 2 <= len(first_keys) <= 50

        elif isinstance(raw_content, dict):
            for value in raw_content.values():
                if isinstance(value, list) and len(value) > 1:
                    if self._detect_tabular_patterns(value):
                        return True
            return False

        return False

    def _has_repeated_keywords(self, text: str, min_repetitions: int = 3) -> bool:
        """Check if text contains repeated keywords that suggest tabular data."""
        field_pattern = r'"([a-zA-Z_][a-zA-Z0-9_]{1,20})"'
        matches = re.findall(field_pattern, text.lower())

        if not matches:
            return False

        field_counts = Counter(matches)

        repeated_fields = [
            field for field, count in field_counts.items() if count >= min_repetitions
        ]

        return len(repeated_fields) >= 2

    def _is_flat_dict(self, obj: dict) -> bool:
        """Check if a dictionary is flat (all values are simple types, not nested)."""
        return all(
            isinstance(v, (str, int, float, bool, type(None))) for v in obj.values()
        )

    def _is_mcp_error(self, raw_content) -> bool:
        """Check if raw content represents an error.

        Supports both:
        - MCP spec standard: {"isError": true, "content": [...]}
        - Legacy format: {"error_type": "...", "content": "..."}
        """
        if not isinstance(raw_content, dict):
            return False

        # MCP spec standard: isError field
        if raw_content.get("isError") is True:
            return True

        # Legacy format: error_type field
        if "error_type" in raw_content or "errortype" in raw_content:
            return True

        return False

    def _get_mcp_error_content(self, raw_content) -> str | None:
        """Get error content for display.

        Handles both MCP spec format and legacy format.
        """
        if not isinstance(raw_content, dict):
            return None

        # MCP spec format: content is an array of content items
        if "content" in raw_content:
            content = raw_content["content"]
            # MCP spec: content is array of {type, text} objects
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        texts.append(item["text"])
                    elif isinstance(item, str):
                        texts.append(item)
                return "\n".join(texts) if texts else None
            # Legacy format: content is a string
            return str(content)

        return None

    def _register_mcp_table_as_structured_context(
        self,
        table_data: list[dict],
        tool_name: str,
        server_name: str,
        artifact_uuid: UUID,
    ) -> bool:
        """Register MCP table artifact as structured context for SQL queries.

        Args:
            table_data: The table data as list of dictionaries
            tool_name: The MCP tool name
            server_name: The MCP server name
            artifact_uuid: The UUID of the artifact

        Returns:
            True if successfully registered, False otherwise
        """
        try:
            table_name = f"mcp_{server_name}_{tool_name}_{str(artifact_uuid)[:8]}"

            df = pd.DataFrame(table_data)

            sql_table_infos = (
                self._context_service._sql_agent_service.insert_table_with_all_infos(
                    df=df,
                    table_name=table_name,
                    description=(
                        f"Table data from MCP tool '{tool_name}' "
                        f"on server '{server_name}'"
                    ),
                    metadata={
                        "mcp_server": server_name,
                        "mcp_tool": tool_name,
                        "artifact_type": "table",
                        "source_uuid": str(artifact_uuid),
                    },
                    source_uuid=str(artifact_uuid),
                )
            )

            source_info = SourceInfo(
                uuid=artifact_uuid,
                type="artifact",
                origin=f"mcp_{server_name}",
                name=table_name,
                description=(
                    f"Table data from MCP tool '{tool_name}' on server '{server_name}'"
                ),
                citable=False,
            )

            for sql_table_info in sql_table_infos:
                structured_ctx = StructuredContext(
                    content=json.dumps(table_data),
                    source_info=source_info,
                    sql_table_info=sql_table_info,
                    data_format=RawObjectDataFormat(
                        data_type="object",
                        parse_as="table",
                    ),
                )

                # Directly append to structured context since SQL table already inserted
                self._context_service._structured_context.append(structured_ctx)

            self._logging_service.info(
                f"Registered MCP table artifact as structured context: {table_name} "
                f"(created {len(sql_table_infos)} table(s))"
            )

            return True

        except Exception as e:
            self._logging_service.error(
                f"Failed to register MCP table as structured context: {e}"
            )
            return False
