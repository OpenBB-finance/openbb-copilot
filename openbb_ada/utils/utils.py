import asyncio
import io
import json
import logging
import math
import os
import re
from contextlib import aclosing
from dataclasses import dataclass
from datetime import datetime
from functools import partial, update_wrapper, wraps
from inspect import signature
from json import JSONDecodeError
from typing import Any, Callable, Generator, Tuple
from uuid import UUID, uuid5

import httpx
import logfire
import openai
import pandas as pd
import pytz
from docx import Document as DocxDocument
from docx.document import Document as DocxDocumentObject
from docx.oxml.table import CT_Tbl as OxmlTable
from docx.oxml.text.paragraph import CT_P as OxmlParagraph
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from fastapi import HTTPException, Request, UploadFile, status
from openbb_ai.models import StatusUpdateSSE, StatusUpdateSSEData, UserAPIKeys
from sse_starlette import EventSourceResponse

from ..constants import (
    AUTH_ENABLED,
    OPENBB_PAYMENTS_API_SECRET_KEY,
    OPENBB_PAYMENTS_BASE_URL,
    RATE_LIMIT_ENABLED,
)
from ..errors import HTTPError, RetryExceededError
from ..models import WidgetFileDetails
from ..pdf import Pdf
from .html_text import extract_text_from_html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Nested data normalization constants
MAX_NORMALIZATION_DEPTH = 5
COMPLEXITY_THRESHOLD = 0.5

# Cache for analyze_data_complexity function
_complexity_cache: dict[int, Any] = {}


@dataclass
class ComplexityMetrics:
    """Simplified complexity metrics with factory methods."""

    unique_keys_count: int
    data_density: float
    structure_uniformity: float
    max_nested_depth: int
    avg_nested_elements: float

    @classmethod
    def from_dict(cls, data: dict) -> "ComplexityMetrics":
        """Create metrics from dictionary."""
        unique_keys = set(data.keys())
        max_depth = 2 if any(isinstance(v, (dict, list)) for v in data.values()) else 1

        return cls(
            unique_keys_count=len(unique_keys),
            data_density=1.0,
            structure_uniformity=1.0,
            max_nested_depth=max_depth,
            avg_nested_elements=float(len(data)),
        )

    @classmethod
    def from_list(cls, data: list, max_sample_size: int = 100) -> "ComplexityMetrics":
        """Create metrics from list."""
        if not data:
            return cls.empty()

        sample_data = data[:max_sample_size]
        unique_keys: set[str] = set()
        depths = []
        element_counts = []
        non_null_count = 0

        for item in sample_data:
            if item is not None:
                non_null_count += 1
                if isinstance(item, dict):
                    unique_keys.update(item.keys())
                    element_counts.append(len(item))
                    depths.append(_calculate_nested_depth(item))
                elif isinstance(item, list):
                    element_counts.append(len(item))
                    depths.append(_calculate_nested_depth(item))
                else:
                    element_counts.append(1)
                    depths.append(1)

        # Calculate uniformity efficiently
        structure_uniformity = cls._calculate_uniformity(sample_data)

        return cls(
            unique_keys_count=len(unique_keys),
            data_density=non_null_count / len(sample_data),
            structure_uniformity=structure_uniformity,
            max_nested_depth=max(depths) if depths else 1,
            avg_nested_elements=sum(element_counts) / len(element_counts)
            if element_counts
            else 0.0,
        )

    @classmethod
    def empty(cls) -> "ComplexityMetrics":
        """Create empty metrics."""
        return cls(0, 0.0, 0.0, 1, 1.0)

    @staticmethod
    def _calculate_uniformity(sample_data: list) -> float:
        """Calculate structure uniformity efficiently."""
        if not sample_data or not isinstance(sample_data[0], dict):
            # For non-dict arrays, check type consistency
            types = [type(item).__name__ for item in sample_data if item is not None]
            if not types:
                return 0.0
            most_common_type = max(set(types), key=types.count)
            return types.count(most_common_type) / len(types)

        # For dict arrays, check key consistency
        all_keys = [
            set(item.keys()) if isinstance(item, dict) else set()
            for item in sample_data
        ]
        if not all_keys:
            return 0.0

        common_keys = set.intersection(*all_keys)
        total_keys = set.union(*all_keys)
        return len(common_keys) / len(total_keys) if total_keys else 0.0

    def should_normalize(self) -> bool:
        """Simplified normalization decision logic."""
        return all(
            [
                self.structure_uniformity >= COMPLEXITY_THRESHOLD,
                self.data_density >= 0.1,
                self.unique_keys_count <= 100,
            ]
        )


def sanitize_str(message: str) -> str:
    """
    escapes odd sequences of '{' or '}' by appending one more brace.
    e.g. '{' -> '{{',  '}}}' -> '}}}}', etc.
    """

    def fix_sequence(seq: str) -> str:
        # if there's an odd number of braces, append one more of the same
        if len(seq) % 2 == 1:
            # append the same brace as the last character in seq
            return seq + seq[-1]
        return seq

    # replace any run of '{' or '}' via `fix_sequence`
    return re.sub(r"(\{+|\}+)", lambda m: fix_sequence(m.group(0)), message)


def get_current_datetime(timezone: str = "UTC") -> str:
    try:
        return datetime.now(pytz.timezone(timezone)).strftime("%Y-%m-%d %H:%M:%S %Z")
    except pytz.exceptions.UnknownTimeZoneError:
        # This way we can still return a timezone even if the timezone is invalid
        logger.warning("Invalid timezone received: %s", timezone)
        return datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S %Z")


async def validate_and_sync(access_token: str) -> tuple[bool, str]:
    """Validate an access token and retrieve information about the user.

    Returns a tuple of (is_valid, error_detail).
    """
    if OPENBB_PAYMENTS_BASE_URL is None:
        raise ValueError("OPENBB_PAYMENTS_BASE_URL is not set")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                OPENBB_PAYMENTS_BASE_URL + "/pro/validate-and-sync",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "X-OpenBB-Authorization": f"Bearer {OPENBB_PAYMENTS_API_SECRET_KEY}",  # noqa: E501
                },
                timeout=30,
            )
            if response.status_code == 200:
                return True, ""
        except (httpx.TimeoutException, httpx.ConnectError, JSONDecodeError) as err:
            logger.error(
                "While validating access token: [%s] %s", type(err).__name__, err
            )
            return (
                False,
                "Unable to reach validation service. Please try again shortly.",
            )
    return False, "Invalid access token."


async def get_copilot_call_count(access_token: str) -> dict:
    if OPENBB_PAYMENTS_BASE_URL is None:
        raise ValueError("OPENBB_PAYMENTS_BASE_URL is not set")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            OPENBB_PAYMENTS_BASE_URL + "/pro/usage",
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-OpenBB-Authorization": f"Bearer {OPENBB_PAYMENTS_API_SECRET_KEY}",
            },
            timeout=15,
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text,
            )
        return response.json()


async def is_rate_limited(access_token: str) -> bool:
    if os.environ.get("DISABLE_RATE_LIMIT", "false").lower() == "true":
        return False

    usage = await get_copilot_call_count(access_token)
    if usage["number_copilot_calls_day_count"] >= usage["copilot_calls_limit"]:
        return True
    return False


async def set_copilot_call_count(access_token: str, count: int):
    if OPENBB_PAYMENTS_BASE_URL is None:
        raise ValueError("OPENBB_PAYMENTS_BASE_URL is not set")
    async with httpx.AsyncClient() as client:
        response = await client.put(
            OPENBB_PAYMENTS_BASE_URL + "/pro/usage",
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-OpenBB-Authorization": f"Bearer {OPENBB_PAYMENTS_API_SECRET_KEY}",
            },
            json={"reset_copilot_count": True, "add_copilot_count": count},
        )
    return response


async def increment_copilot_call_count(access_token: str, increment_by: int = 1):
    if OPENBB_PAYMENTS_BASE_URL is None:
        raise ValueError("OPENBB_PAYMENTS_BASE_URL is not set")
    async with httpx.AsyncClient() as client:
        response = await client.put(
            OPENBB_PAYMENTS_BASE_URL + "/pro/usage",
            headers={
                "Authorization": f"Bearer {access_token}",
                "X-OpenBB-Authorization": f"Bearer {OPENBB_PAYMENTS_API_SECRET_KEY}",
            },
            json={"add_copilot_count": increment_by},
        )
    if response.status_code != 200:
        raise HTTPError(
            f"Received invalid response from Hub: {response.status_code}:{response.text}"  # noqa: E501
        )
    return response


def rate_limit():
    def decorator(func):
        @wraps(func)
        async def wrapper(
            request: Request,
            *args,
            **kwargs,
        ):
            if not AUTH_ENABLED or not RATE_LIMIT_ENABLED:
                return await func(request, *args, **kwargs)
            access_token = request.headers.get("Authorization", "").replace(
                "Bearer ", ""
            )
            if access_token:
                if os.environ.get("DISABLE_RATE_LIMIT", "false").lower() == "true":
                    return await func(request, *args, **kwargs)

                payload = await request.json()
                user_api_keys = UserAPIKeys(**payload.get("api_keys", {}))
                message_role = payload.get("messages", [{}])[-1].get("role", "human")

                # We rate limit on human messages, not on completions,
                # or user has an openai key
                if message_role != "human" or user_api_keys.openai_api_key:
                    return await func(request, *args, **kwargs)

                if await is_rate_limited(access_token):
                    status_update = StatusUpdateSSE(
                        event="copilotStatusUpdate",
                        data=StatusUpdateSSEData(
                            eventType="ERROR",
                            message="You've reached your daily completion limit of OpenBB Copilot. You can add your own OpenAI API key to continue, or wait for your completion limit to reset tomorrow.",  # noqa: E501
                        ),
                    )
                    return EventSourceResponse(
                        (
                            event.model_dump(exclude_none=True)
                            for event in [status_update]
                        ),
                        media_type="text/event-stream",
                    )
                result = await func(request, *args, **kwargs)
                try:
                    await increment_copilot_call_count(access_token, increment_by=1)
                except HTTPError as err:
                    logger.error(
                        "While incrementing copilot call count: %s",
                        err,
                        extra={"trace_id": request.headers.get("X-Trace-Id")},
                    )
                    # We still want to return the result, even if we couldn't
                    # increment the call count.
                    return result
                return result

        return wrapper

    return decorator


def split_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 100
) -> list[str]:
    output = []
    cursor = 0
    while cursor < len(text):
        output.append(text[cursor : cursor + chunk_size])
        cursor += chunk_size - chunk_overlap
    return output


def find_longest_common_substring_indices(
    document_words: list[str], target_words: list[str]
) -> tuple[int, int]:
    # TODO: In the case where we have multiple "best" matches,
    # we will only return the first one. We should instead
    # return all of them.
    max_len = 0
    best_start = 0
    for i in range(len(document_words)):
        for j in range(len(target_words)):
            k = 0
            while (
                i + k < len(document_words)
                and j + k < len(target_words)
                and document_words[i + k] == target_words[j + k]
            ):
                k += 1
            if k > max_len:
                max_len = k
                best_start = i
    return best_start, best_start + max_len


async def get_file_details(
    uploaded_file: UploadFile,
    n_peek_pages: int = 2,
    n_peek_sheet_rows: int = 10,
    n_peek_row_index: int = 5,
) -> WidgetFileDetails:
    content_type = uploaded_file.content_type

    await uploaded_file.seek(0)

    if content_type == "application/pdf":
        pdf_file = Pdf(await uploaded_file.read())
        file_text = pdf_file.get_text()
        return WidgetFileDetails(
            widget_data="\n".join(file_text[:n_peek_pages]),
            filename=uploaded_file.filename,
        )
    elif content_type in [
        "application/csv",
        "text/csv",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ]:
        if content_type in ["application/csv", "text/csv"]:
            df = pd.read_csv(
                io.BytesIO(await uploaded_file.read()),
                nrows=n_peek_sheet_rows,
                index_col=0,
            )
        elif content_type in [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]:
            df = pd.read_excel(
                io.BytesIO(await uploaded_file.read()),
                nrows=n_peek_sheet_rows,
                index_col=0,
            )

        filename = uploaded_file.filename
        columns = ", ".join(df.columns)
        index = (
            ", ".join(df.index[:n_peek_row_index].astype(str))
            + ", ... , "
            + ", ".join(df.index[-n_peek_row_index:].astype(str))
        )

        return WidgetFileDetails(
            widget_data=handle_duplicate_columns_names(df).to_json(orient="records"),
            filename=filename,
            columns=columns,
            index=index,
        )
    elif content_type in ["text/plain", "text/markdown", "text/html"]:
        file_data = await uploaded_file.read()
        widget_data = file_data.decode("utf-8")
        if content_type == "text/html":
            widget_data = extract_text_from_html(widget_data)
        return WidgetFileDetails(
            widget_data=widget_data, filename=uploaded_file.filename
        )
    elif content_type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        doc = DocxDocument(io.BytesIO(await uploaded_file.read()))
        text_chunks = extract_text_chunks_from_docx(doc)
        return WidgetFileDetails(
            widget_data="\n".join(text_chunks[:n_peek_pages]),
            filename=uploaded_file.filename,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unsupported file type",
        )


def get_human_readable_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"


def flatten_and_format_dict(input_dict: dict[str, Any] | None) -> dict[str, str]:
    if not input_dict:
        return {}
    dict_ = {}
    for k, v in input_dict.items():
        if isinstance(v, dict):
            nested_items = flatten_and_format_dict(v)
            dict_.update(nested_items)
        else:
            dict_[k.capitalize()] = str(v)
    return dict_


def flatten_data(data: dict[str, Any]) -> pd.DataFrame:
    result = []

    for key, value in data.items():
        if isinstance(value, list):
            try:
                df = pd.json_normalize(value)
            except Exception:
                df = pd.DataFrame(value, columns=[key])
        elif isinstance(value, dict):
            df = pd.json_normalize(value)
        else:  # Scalar value
            df = pd.DataFrame({key: [value]})

        df.columns = [f"{key}.{col}" for col in df.columns]  # type: ignore
        result.append(df)

    final_df = pd.concat(result, axis=1)
    return final_df


def handle_openai_error(error: openai.OpenAIError) -> StatusUpdateSSE:
    error_message = "AI service is temporarily unavailable"
    error_detail = str(error)

    if isinstance(error, openai.AuthenticationError):
        error_message = "Authentication error"
    elif isinstance(error, openai.APIConnectionError):
        error_message = "Connection error"
        # Log the underlying cause for debugging
        if hasattr(error, "__cause__") and error.__cause__:
            error_detail += f" (Cause: {error.__cause__})"
    elif isinstance(error, openai.RateLimitError):
        error_message = "Rate limit exceeded"

    return StatusUpdateSSE(
        data=StatusUpdateSSEData(
            eventType="ERROR",
            message=error_message,
        )
    )


def handle_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    for col in df_copy.columns:
        try:
            if (
                (
                    col.lower() == "date"
                    or col.lower() == "time"
                    or "_date" in col.lower()
                    or "_time" in col.lower()
                    or "datetime" in col.lower()
                    or "timestamp" in col.lower()
                )
                and df_copy[col].dtype == "object"
                and "timeline" not in col.lower()
            ):
                df_copy[col] = pd.to_datetime(df_copy[col])
        except ValueError:
            pass
    return df_copy


def handle_df_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """Transpose and reorient a dataframe if there is an index column.

    This occurs when a table is transposed, for example, "dates" are used as
    columns and each row contains a different "column" of data.
    """
    if df.columns[0].lower() == "index":
        if df[df.columns[0]].dtype == "object":
            try:
                df = df.set_index(df.columns[0])
                # Can we convert the columns to datetime?
                pd.to_datetime(df.columns)
                # If we can, then we can unpivot the table
                df = df.transpose()
                # And we can convert the index to a datetime
                df.index = pd.to_datetime(df.index)
            except ValueError:  # If not, nothing to do here.
                pass
    return df


def handle_nested_hashmaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Original implementation for backward compatibility.
    Converts nested dictionaries to JSON strings.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            df_copy[col] = df_copy[col].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else x
            )
    return df_copy


def detect_nested_data_columns(
    df: pd.DataFrame,
    metadata: dict | None = None,
    current_depth: int = 0,
    parent_path: str | None = None,
) -> dict[str, dict]:
    """Detect columns containing nested data (arrays/objects) that need normalization.

    Detection strategy:
    1. Check metadata for explicit hints (cellDataType: "array"/"object")
    2. Auto-detect by inspecting object-dtype columns for nested structures
    3. Add complexity analysis and normalization decisions when possible

    Args:
        df: DataFrame to analyze
        metadata: Optional metadata with column hints
        current_depth: Current nesting depth
        parent_path: Path from root for hierarchical tracking

    Returns:
        Dictionary mapping column names to their nested data info
    """
    # Stop recursion if we've reached max depth
    if current_depth >= MAX_NORMALIZATION_DEPTH:
        return {}

    nested_columns = {}

    # Method 1: Check metadata hints
    if metadata and "columns" in metadata:
        for col_name, col_info in metadata["columns"].items():
            if col_info.get("cellDataType") in ["array", "object"]:
                nested_columns[col_name] = {
                    "data_type": col_info["cellDataType"],
                    "sample_value": df[col_name].iloc[0]
                    if col_name in df.columns
                    else None,
                    "detected_via": "metadata",
                }

    # Method 2: Auto-detect with error handling
    for col_name in df.select_dtypes(include=["object"]).columns:
        if col_name in nested_columns:
            continue

        column_info = _analyze_column_with_fallback(df[col_name])
        if column_info:
            nested_columns[col_name] = column_info

    # Method 3: Enhanced analysis with error handling
    return _enhance_columns_with_complexity(
        nested_columns, df, current_depth, parent_path
    )


def generate_group_path(
    parent_path: str | None, group_id: int, separator: str = "."
) -> str:
    """Generate hierarchical group path."""
    if parent_path is None:
        return str(group_id)
    return f"{parent_path}{separator}{group_id}"


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
                    logger.warning(
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


def _looks_like_json(value: str) -> bool:
    """Quick heuristic for JSON-like strings."""
    stripped = value.strip()
    return len(stripped) > 2 and stripped[0] in "[{" and stripped[-1] in "]}"


@handle_normalization_errors(fallback_action="empty", return_empty_on_error=True)
def _analyze_column_with_fallback(column: pd.Series) -> dict | None:
    """Analyze column with automatic error handling."""
    sample_values = column.dropna().head(5).tolist()
    if not sample_values:
        return None

    nested_count = 0
    sample_data_type = None
    sample_value_for_column = None

    for sample_value in sample_values:
        if isinstance(sample_value, (list, dict)):
            nested_count += 1
            if sample_data_type is None:
                sample_data_type = (
                    "array" if isinstance(sample_value, list) else "object"
                )
                sample_value_for_column = sample_value
        elif isinstance(sample_value, str) and _looks_like_json(sample_value):
            parsed = json.loads(sample_value)  # This may raise, caught by decorator
            if isinstance(parsed, (list, dict)):
                nested_count += 1
                if sample_data_type is None:
                    sample_data_type = "array" if isinstance(parsed, list) else "object"
                    sample_value_for_column = parsed

    # Check if enough samples are nested (40% threshold)
    if nested_count > 0 and (nested_count / len(sample_values)) >= 0.4:
        return {
            "data_type": sample_data_type,
            "sample_value": sample_value_for_column,
            "detected_via": "auto_inspection",
        }
    return None


@handle_normalization_errors(fallback_action="serialize")
def _enhance_columns_with_complexity(
    nested_columns: dict, df: pd.DataFrame, current_depth: int, parent_path: str | None
) -> dict[str, dict]:
    """Add complexity analysis with error handling."""
    enhanced_columns = {}
    df_length = len(df)

    for col_name, col_info in nested_columns.items():
        enhanced_info = {
            **col_info,
            "current_depth": current_depth,
            "parent_path": parent_path,
        }

        # Complexity analysis with automatic fallback
        sample_data = col_info.get("sample_value")
        if sample_data is not None:
            complexity = analyze_data_complexity(sample_data)  # Protected by decorator
            should_normalize = complexity.should_normalize() and df_length >= 2

            enhanced_info.update(
                {
                    "complexity_metrics": complexity,
                    "should_normalize": should_normalize,
                    "normalization_strategy": "normalize"
                    if should_normalize
                    else "serialize",
                }
            )
        else:
            # Fallback values for missing sample data
            enhanced_info.update(
                {
                    "should_normalize": True,
                    "normalization_strategy": "normalize",
                }
            )

        enhanced_columns[col_name] = enhanced_info

    return enhanced_columns


def analyze_dictionary_schema(values: list, max_keys_to_flatten: int = 20) -> dict:
    """Simplified dictionary schema analysis using set operations."""
    # Early returns for invalid inputs
    dict_values = [v for v in values if isinstance(v, dict)]
    if not dict_values:
        return {
            "should_flatten": False,
            "keys_to_flatten": set(),
            "has_consistent_schema": False,
        }

    total_dicts = len(dict_values)
    min_frequency = max(1, int(total_dicts * 0.3))  # 30% threshold

    # Use Counter for efficient key frequency analysis
    from collections import Counter

    key_counter = Counter(key for d in dict_values for key in d.keys())

    # Get frequent keys efficiently
    frequent_keys = {
        key for key, count in key_counter.items() if count >= min_frequency
    }

    # Limit keys to prevent column explosion
    if len(frequent_keys) > max_keys_to_flatten:
        # Sort by frequency and take top keys
        frequent_keys = set(
            key
            for key, _ in key_counter.most_common(max_keys_to_flatten)
            if key_counter[key] >= min_frequency
        )

    # Check for simple values (non-nested) efficiently
    has_simple_values = not any(
        isinstance(d.get(key), (dict, list))
        for d in dict_values
        for key in frequent_keys
        if key in d
    )

    # Schema consistency check using set operations
    has_consistent_schema = (
        len(frequent_keys) > 0
        and sum(
            len(set(d.keys()) & frequent_keys) >= len(frequent_keys) * 0.5
            for d in dict_values
        )
        / total_dicts
        >= 0.8  # 80% of dicts have 50%+ of keys
    )

    should_flatten = (
        has_simple_values
        and len(frequent_keys) > 0
        and len(frequent_keys) <= max_keys_to_flatten
        and total_dicts >= 2
    )

    return {
        "should_flatten": should_flatten,
        "keys_to_flatten": frequent_keys,
        "has_consistent_schema": has_consistent_schema,
        "key_frequencies": dict(key_counter),
        "total_dictionaries": total_dicts,
    }


def analyze_data_complexity(
    data: list | dict, max_sample_size: int = 100
) -> ComplexityMetrics:
    """Simplified main function."""
    if isinstance(data, dict):
        return ComplexityMetrics.from_dict(data)
    elif isinstance(data, list):
        return ComplexityMetrics.from_list(data, max_sample_size)
    else:
        return ComplexityMetrics.empty()


def _calculate_nested_depth(obj: Any, current_depth: int = 1) -> int:
    """Calculate the maximum nesting depth of an object."""
    if not isinstance(obj, (dict, list)):
        return current_depth

    max_depth = current_depth

    if isinstance(obj, dict):
        for value in obj.values():
            depth = _calculate_nested_depth(value, current_depth + 1)
            max_depth = max(max_depth, depth)
    elif isinstance(obj, list):
        for item in obj:
            depth = _calculate_nested_depth(item, current_depth + 1)
            max_depth = max(max_depth, depth)

    return max_depth


def handle_duplicate_columns_names(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame columns are unique by appending suffixes to duplicates.

    This prevents ValueError when converting DataFrame to JSON with orient='records'
    which requires unique column names since JSON objects cannot have duplicate keys.

    Args:
        df: DataFrame that may have duplicate column names

    Returns:
        DataFrame with unique column names

    Example
    -------
    >>> df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col2": [7, 8, 9]})
    >>> handle_duplicate_columns_names(df)
    >>> pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col2_1": [7, 8, 9]})
    """
    if df.columns.duplicated().any():
        df_copy = df.copy()
        # Create unique column names by appending suffixes
        cols = df_copy.columns.tolist()
        seen: dict[str, int] = {}
        new_cols = []

        for col in cols:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)

        df_copy.columns = new_cols
        return df_copy
    return df


def wrapped_partial(func, *args, **kwargs):
    p = partial(func, *args, **kwargs)
    update_wrapper(p, func)
    return p


def masked_partial(func, *args, **kwargs) -> Callable:
    """Creates a partial that obfuscates the passed-in arguments from the
    function signature.

    This is useful when creating LLM functions where you want to pass in some
    extra data as input that you don't want the LLM to generate.

    Example
    -------
    def _tool_filter_values(query: str, options: list[str]) -> list[str]:
        ...

    # We want to pass in the `options` but not have the LLM generate it.
    masked_filter_values = masked_partial(_tool_filter_values, options=["a", "b", "c"])

    Now the `masked_filter_values` function will have the signature:
    def masked_filter_values(query: str) -> list[str]:
        ...
    But will have the `options` passed in when it is called.

    Now we can add this as a `magentic` tool and the `options` will be passed in
    but not be visible to the LLM.
    """
    p = partial(func, *args, **kwargs)
    wrapped_p = update_wrapper(p, func)
    original_signature = signature(func)
    parameters = [
        param
        for name, param in original_signature.parameters.items()
        if name not in kwargs
    ]
    new_signature = original_signature.replace(parameters=parameters)
    wrapped_p.__signature__ = new_signature  # type: ignore
    return wrapped_p


def instrument_async_generator(span_name: str):
    # Logfire requires generators to be treated as context managers to ensure
    # proper closure
    # and prevent logging issues when yielding SSEs to clients.
    # https://logfire.pydantic.dev/docs/reference/advanced/generators/#use-a-generator-as-a-context-manager
    def outer_wrapper(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async def generator(*args, **kwargs):
                with logfire.span(span_name):
                    async for event in func(*args, **kwargs):
                        yield event

            async with aclosing(generator(*args, **kwargs)) as gen:
                async for event in gen:
                    yield event

        return wrapper

    return outer_wrapper


def extract_text_chunks_from_docx(document: DocxDocumentObject) -> list[str]:
    def iterate_elements(
        parent: DocxDocumentObject,
    ) -> Generator[DocxParagraph | DocxTable, None, None]:
        for child in parent.element.body.iterchildren():
            if isinstance(child, OxmlParagraph):
                yield DocxParagraph(child, parent)
            elif isinstance(child, OxmlTable):
                yield DocxTable(child, parent)

    text_chunks = []
    # Iterating over the elements ensures they are in the correct order.
    for element in iterate_elements(document):
        if isinstance(element, DocxParagraph):
            text = element.text.strip().replace("\t", " ")
            if text:
                text_chunks.append(text)
        elif isinstance(element, DocxTable):
            rows = [
                " | ".join(cell.text.strip().replace("\t", " ") for cell in row.cells)
                for row in element.rows
            ]
            text = "\n".join(rows)
            text_chunks.append(text)
    return text_chunks


def retry_on_exception(max_retries: int, exceptions: Tuple[type[Exception], ...]):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    if attempt < max_retries - 1:
                        logger.warning(
                            "Error: %s. Retrying... (Attempt %d/%d)",
                            exc,
                            attempt + 1,
                            max_retries,
                        )
                        await asyncio.sleep(2**attempt)  # Exponential backoff
                    else:
                        logger.error(
                            "Failed after %d attempts: %s",
                            max_retries,
                            exc,
                        )
                        raise RetryExceededError(
                            f"Failed after {max_retries} attempts: {exc}"
                        ) from exc

        return wrapper

    return decorator


def sanitize_tool_name(
    tool_name: str,
    existing_names: set[str] | None = None,
    max_length: int = 64,
) -> str:
    """Create a valid Python identifier from an MCP tool name.

    Replaces spaces/dashes/dots with underscores, prefixes with ``mcp_``,
    and truncates to *max_length* characters.  When the result collides with
    an entry in *existing_names* a short numeric suffix is appended.

    Example
    -------
    >>> sanitize_tool_name("qtap duckdb mcp_list_columns")
    'mcp_qtap_duckdb_mcp_list_columns'
    """
    raw = f"{tool_name}"
    # Replace non-alphanumeric, non-underscore chars with '_'
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    sanitized = f"mcp_{sanitized}"

    # Truncate
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip("_")

    # Handle collisions
    if existing_names is not None:
        base = sanitized
        counter = 2
        while sanitized in existing_names:
            suffix = f"_{counter}"
            sanitized = base[: max_length - len(suffix)] + suffix
            counter += 1

    return sanitized


def build_context_uuid(
    widget_uuid: UUID,
    input_args: dict[str, Any] | None,
    item_index: int,
    extra_seed: Any | None = None,
) -> UUID:
    """Deterministically derive a context UUID for widget data snapshots."""

    payload: dict[str, Any] = {
        "input_args": input_args or {},
        "item_index": item_index,
    }
    if extra_seed is not None:
        payload["extra"] = extra_seed

    seed = json.dumps(payload, sort_keys=True, default=str)
    return uuid5(widget_uuid, seed)
