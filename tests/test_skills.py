"""
Tests for skill content handling in CopilotService._handle_function_call_result_message.

Covers the get_skill_content case (content extraction, error handling, and
status update generation) and the save_skill case (save confirmation, save
errors, and malformed payloads).
"""

from unittest.mock import MagicMock

import pytest
from openbb_ai.models import ClientCommandResult, StatusUpdateSSE

from openbb_ada.models import (
    LlmClientFunctionCallResultMessage,
    SkillCatalogEntry,
)


def _mock_function_call() -> MagicMock:
    fc = MagicMock()
    fc._unique_id = "test-function-call-id"
    return fc


def test_hydrate_loaded_skills_from_messages_restores_active_skill(
    test_copilot_service,
):
    test_copilot_service._skills_catalog = [
        SkillCatalogEntry(
            slug="snowflake-html-report",
            description="Generate HTML reports",
            updatedAt="2026-01-01T00:00:00Z",
        )
    ]
    message = LlmClientFunctionCallResultMessage(
        function="get_skill_content",
        input_arguments={"slug": "snowflake-html-report"},
        data=[
            ClientCommandResult(
                status="success",
                data={
                    "skill": {
                        "slug": "snowflake-html-report",
                        "description": "Generate HTML reports",
                        "contentMarkdown": "# Report Skill\n\nUse HTML.",
                    }
                },
            )
        ],
    )

    test_copilot_service._hydrate_loaded_skills_from_messages([message])

    assert [skill.slug for skill in test_copilot_service._selected_skills] == [
        "snowflake-html-report"
    ]
    assert test_copilot_service._selected_skills[0].source == "model_selected"
    assert test_copilot_service._skills_catalog == []


@pytest.mark.asyncio
async def test_handle_skill_content_result_populates_content(test_copilot_service):
    """Skill content from a successful response is injected into the LLM message."""
    message = LlmClientFunctionCallResultMessage(
        function="get_skill_content",
        input_arguments={"slug": "financial-analysis"},
        data=[
            ClientCommandResult(
                status="success",
                data={
                    "skill": {
                        "slug": "financial-analysis",
                        "description": "Analyze earnings and financial data",
                        "contentMarkdown": "# Financial Analysis\n\nFocus on EPS.",
                    }
                },
            )
        ],
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=_mock_function_call(),
        is_last_message=False,
    ):
        events.append(event)

    assert len(events) == 1
    result_messages = events[0]
    assert isinstance(result_messages, list)
    content = result_messages[0].content
    assert "## Skill Instructions: financial-analysis" in content
    assert "Analyze earnings and financial data" in content
    assert "# Financial Analysis" in content
    assert "Focus on EPS." in content


@pytest.mark.asyncio
async def test_handle_skill_content_result_yields_status_update_on_last_message(
    test_copilot_service,
):
    """A StatusUpdateSSE is emitted when is_last_message=True and content was found."""
    message = LlmClientFunctionCallResultMessage(
        function="get_skill_content",
        input_arguments={"slug": "my-skill"},
        data=[
            ClientCommandResult(
                status="success",
                data={
                    "skill": {
                        "slug": "my-skill",
                        "description": "A skill",
                        "contentMarkdown": "Some instructions",
                    }
                },
            )
        ],
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=_mock_function_call(),
        is_last_message=True,
    ):
        events.append(event)

    status_events = [e for e in events if isinstance(e, StatusUpdateSSE)]
    assert len(status_events) == 1
    assert "my-skill" in status_events[0].data.message
    assert status_events[0].data.eventType == "INFO"
    assert status_events[0].data.details is not None
    assert "Some instructions" in status_events[0].data.details[0]


@pytest.mark.asyncio
async def test_handle_skill_content_result_no_status_update_when_not_last_message(
    test_copilot_service,
):
    """No StatusUpdateSSE is emitted when is_last_message=False."""
    message = LlmClientFunctionCallResultMessage(
        function="get_skill_content",
        input_arguments={"slug": "my-skill"},
        data=[
            ClientCommandResult(
                status="success",
                data={
                    "skill": {
                        "slug": "my-skill",
                        "description": "A skill",
                        "contentMarkdown": "Instructions",
                    }
                },
            )
        ],
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=_mock_function_call(),
        is_last_message=False,
    ):
        events.append(event)

    status_events = [e for e in events if isinstance(e, StatusUpdateSSE)]
    assert len(status_events) == 0


@pytest.mark.asyncio
async def test_handle_skill_content_result_error_appended_to_content(
    test_copilot_service,
):
    """Error responses include the error message in the LLM content."""
    slug = "missing-skill"
    message = LlmClientFunctionCallResultMessage(
        function="get_skill_content",
        input_arguments={"slug": slug},
        data=[
            ClientCommandResult(
                status="error",
                message="Skill not found in library",
            )
        ],
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=_mock_function_call(),
        is_last_message=True,
    ):
        events.append(event)

    result_messages = [e for e in events if isinstance(e, list)]
    assert len(result_messages) == 1
    content = result_messages[0][0].content
    assert f"Skill '{slug}' not found" in content
    assert "Skill not found in library" in content

    # StatusUpdateSSE is still emitted on last message but without details
    status_events = [e for e in events if isinstance(e, StatusUpdateSSE)]
    assert len(status_events) == 1
    assert status_events[0].data.details is None


@pytest.mark.asyncio
async def test_handle_save_skill_result_success_yields_status_and_confirmation(
    test_copilot_service,
):
    """A successful save yields an INFO status update and a confirmation
    message for the LLM mentioning the slug."""
    message = LlmClientFunctionCallResultMessage(
        function="save_skill",
        input_arguments={"name": None, "instructions": None},
        data=[
            ClientCommandResult(
                status="success",
                data={
                    "skill": {
                        "slug": "earnings-analysis",
                        "description": "Analyze quarterly earnings",
                    }
                },
            )
        ],
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=_mock_function_call(),
        is_last_message=True,
    ):
        events.append(event)

    status_events = [e for e in events if isinstance(e, StatusUpdateSSE)]
    assert len(status_events) == 1
    assert status_events[0].data.eventType == "INFO"
    assert status_events[0].data.message == "Skill saved: /earnings-analysis"
    assert status_events[0].data.details is not None
    assert status_events[0].data.details[0]["skill_slug"] == "earnings-analysis"
    assert (
        status_events[0].data.details[0]["skill_description"]
        == "Analyze quarterly earnings"
    )

    result_messages = [e for e in events if isinstance(e, list)]
    assert len(result_messages) == 1
    content = result_messages[0][0].content
    assert "/earnings-analysis" in content
    assert "Analyze quarterly earnings" in content
    assert "confirm to the user" in content.lower()


@pytest.mark.asyncio
async def test_handle_save_skill_result_error_yields_warning(test_copilot_service):
    """An error result yields a WARNING status update and an error message
    for the LLM."""
    message = LlmClientFunctionCallResultMessage(
        function="save_skill",
        input_arguments={"name": None, "instructions": None},
        data=[
            ClientCommandResult(
                status="error",
                message="Storage quota exceeded",
            )
        ],
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=_mock_function_call(),
        is_last_message=True,
    ):
        events.append(event)

    status_events = [e for e in events if isinstance(e, StatusUpdateSSE)]
    assert len(status_events) == 1
    assert status_events[0].data.eventType == "WARNING"
    assert status_events[0].data.message == "Skill not saved"
    assert status_events[0].data.details is None

    result_messages = [e for e in events if isinstance(e, list)]
    assert len(result_messages) == 1
    content = result_messages[0][0].content
    assert "Failed to save the skill: Storage quota exceeded" in content


@pytest.mark.asyncio
async def test_handle_save_skill_result_with_non_dict_skill_does_not_crash(
    test_copilot_service,
):
    """A malformed success result where data.skill is not a dict is handled
    gracefully instead of crashing."""
    message = LlmClientFunctionCallResultMessage(
        function="save_skill",
        input_arguments={"name": None, "instructions": None},
        data=[
            ClientCommandResult(
                status="success",
                data={"skill": "not-a-dict"},
            )
        ],
    )

    events = []
    async for event in test_copilot_service._handle_function_call_result_message(
        message=message,
        function_call=_mock_function_call(),
        is_last_message=True,
    ):
        events.append(event)

    status_events = [e for e in events if isinstance(e, StatusUpdateSSE)]
    assert len(status_events) == 1
    assert status_events[0].data.eventType == "WARNING"
    assert status_events[0].data.message == "Skill not saved"

    result_messages = [e for e in events if isinstance(e, list)]
    assert len(result_messages) == 1
    assert "The skill was not saved." in result_messages[0][0].content
