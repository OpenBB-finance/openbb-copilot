import json
import re
from typing import Any

import pytest


def parse_message_chunks(response_text: str) -> str:
    result = ""
    lines = response_text.split("\n")
    event = ""
    data = ""

    for line in lines:
        if line.startswith("event:"):
            event = line
        elif line.startswith("data:"):
            data = line[6:].strip()
            if "event: copilotMessageChunk" in event:
                try:
                    chunk_json = json.loads(data)
                    if "delta" in chunk_json:
                        result += chunk_json["delta"]
                except json.JSONDecodeError:
                    continue
            elif "event: copilotMessageArtifact" in event:
                artifact = json.loads(data)
                # Show a placeholder for the artifact
                result += f"<|TESTING_PLACEHOLDER_ARTIFACT:{artifact['uuid']}|>"
    return result


def parse_message_artifacts(response_text: str) -> list[dict]:
    result: list[dict] = []
    lines = response_text.split("\n")
    event = ""
    data = ""

    for line in lines:
        if line.startswith("event:"):
            event = line
        elif line.startswith("data:"):
            data = line[6:].strip()
            if "event: copilotMessageArtifact" in event:
                result.append(json.loads(data))
    return result


def parse_function_calls(response_text: str) -> list[dict]:
    result: list[dict] = []
    lines = response_text.split("\n")
    event = ""
    data = ""

    for line in lines:
        if line.startswith("event:"):
            event = line
        elif line.startswith("data:"):
            data = line[6:].strip()
            if "event: copilotFunctionCall" in event:
                result.append(json.loads(data))
    return result


def parse_status_updates(response_text: str) -> list[dict[str, Any]]:
    result: list[dict] = []
    lines = response_text.split("\n")
    event = ""
    data = ""

    for line in lines:
        if line.startswith("event:"):
            event = line
        elif line.startswith("data:"):
            data = line[6:].strip()
            if "event: copilotStatusUpdate" in event:
                payload = json.loads(data)
                if not payload["hidden"]:
                    result.append(payload)
    return result


def parse_citations(response_text: str) -> list[dict[str, Any]]:
    lines = response_text.split("\n")
    event = ""
    data = ""

    citations = []
    for line in lines:
        if line.startswith("event:"):
            event = line
        elif line.startswith("data:"):
            data = line[6:].strip()
            if "event: copilotCitationCollection" in event:
                citations.extend(json.loads(data)["citations"])
    return citations


def parse_prompt_suggestions(response_text: str) -> list[str]:
    event = ""
    suggestions: list[str] = []

    for line in response_text.split("\n"):
        if line.startswith("event:"):
            event = line
        elif line.startswith("data:"):
            data = line[6:].strip()
            if "event: copilotPromptSuggestions" in event:
                suggestions.extend(json.loads(data).get("suggestions", []))

    return suggestions


def assert_status_update_exists(
    status_updates: list[dict[str, Any]],
    message_pattern: str,
) -> dict[str, Any]:
    """Find a status update with a given message pattern (regex)."""
    for update in status_updates:
        if re.search(message_pattern, update.get("message", "")):
            return update
    pytest.fail(
        f"Status update with message '{message_pattern}' not found in status updates."
    )


def assert_status_update_exists_flexible(
    status_updates: list[dict[str, Any]], message: str
) -> dict[str, Any]:
    """Find a status update containing a given message."""
    for update in status_updates:
        if message in update.get("message", ""):
            return update
    pytest.fail(
        f"Status update containing message '{message}' not found in status updates."
    )


def assert_status_update_exists_optional(
    status_updates: list[dict[str, Any]], message: str
) -> dict[str, Any] | None:
    """Find a status update with a given message, but don't fail if not found."""
    for update in status_updates:
        if message in update.get("message", ""):
            return update
    return None


def assert_widget_operation_status_exists(
    status_updates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Find a status update indicating widget operations are happening."""
    operation_keywords = [
        "querying",
        "requesting",
        "retrieving",
        "asking",
        "processing",
        "calling",
        "generating",
        "searching",
        "preparing",
    ]
    for update in status_updates:
        message = update.get("message", "").lower()
        if any(keyword in message for keyword in operation_keywords):
            return update
    pytest.fail(
        f"No widget operation status update found. Available updates: "
        f"{[u.get('message', '') for u in status_updates]}"
    )


def assert_data_retrieval_status_exists(
    status_updates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Find a status update indicating data retrieval operations."""
    retrieval_keywords = [
        "retrieving",
        "fetching",
        "loading",
        "requesting",
        "querying",
        "getting",
        "obtaining",
        "collecting",
    ]
    for update in status_updates:
        message = update.get("message", "").lower()
        if any(keyword in message for keyword in retrieval_keywords):
            return update
    pytest.fail(
        f"No data retrieval status update found. Available updates: "
        f"{[u.get('message', '') for u in status_updates]}"
    )
