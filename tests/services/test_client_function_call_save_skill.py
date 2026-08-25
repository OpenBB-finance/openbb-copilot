"""
Tests for save-skill functionality in ClientFunctionCallService.

Tests the save_skill function call handling including:
- Function call generation
- Client tool mapping
- Function call recognition and routing
"""

from unittest.mock import MagicMock, Mock

import pytest
from openbb_ai.models import (
    FunctionCallResponse,
    FunctionCallSSE,
    StatusUpdateSSE,
    StatusUpdateSSEData,
)

from openbb_ada.services import ClientFunctionCallService, LoggingService


@pytest.fixture
def client_function_call_service() -> ClientFunctionCallService:
    """Create a ClientFunctionCallService instance for testing."""
    return ClientFunctionCallService(
        copilot_data_service=Mock(),
        logging_service=LoggingService(),
    )


class TestLlmSaveSkill:
    """Tests for llm_save_skill function."""

    @pytest.mark.asyncio
    async def test_yields_status_update_then_function_call_response(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that llm_save_skill yields a StatusUpdateSSE then a
        FunctionCallResponse."""
        events = []
        async for event in client_function_call_service.llm_save_skill(
            name="earnings analysis",
            instructions="focus on the comparison table format",
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], StatusUpdateSSE)
        assert isinstance(events[1], FunctionCallResponse)

    @pytest.mark.asyncio
    async def test_status_update_uses_summary_message(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that the StatusUpdateSSE carries the summary as an INFO message."""
        events = []
        async for event in client_function_call_service.llm_save_skill():
            events.append(event)

        assert isinstance(events[0], StatusUpdateSSE)
        assert events[0].data.eventType == "INFO"
        assert events[0].data.message == "Saving skill"

    @pytest.mark.asyncio
    async def test_function_call_response_has_correct_function_name(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that the function name is 'save_skill'."""
        events = []
        async for event in client_function_call_service.llm_save_skill():
            events.append(event)

        assert isinstance(events[1], FunctionCallResponse)
        assert events[1].function == "save_skill"

    @pytest.mark.asyncio
    async def test_function_call_response_includes_name_and_instructions(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that the input_arguments include name and instructions."""
        events = []
        async for event in client_function_call_service.llm_save_skill(
            name="earnings analysis",
            instructions="focus on the comparison table format",
        ):
            events.append(event)

        assert isinstance(events[1], FunctionCallResponse)
        assert events[1].input_arguments is not None
        assert events[1].input_arguments["name"] == "earnings analysis"
        assert (
            events[1].input_arguments["instructions"]
            == "focus on the comparison table format"
        )

    @pytest.mark.asyncio
    async def test_input_arguments_default_to_none_when_not_provided(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that name and instructions are None when not provided."""
        events = []
        async for event in client_function_call_service.llm_save_skill():
            events.append(event)

        assert isinstance(events[1], FunctionCallResponse)
        assert events[1].input_arguments is not None
        assert events[1].input_arguments["name"] is None
        assert events[1].input_arguments["instructions"] is None

    @pytest.mark.asyncio
    async def test_extra_state_includes_copilot_function_call_arguments(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that extra_state carries summary, name and instructions."""
        events = []
        async for event in client_function_call_service.llm_save_skill(
            name="my skill",
            instructions="keep it short",
            summary="Saving my skill",
        ):
            events.append(event)

        assert isinstance(events[1], FunctionCallResponse)
        assert events[1].extra_state is not None
        assert events[1].extra_state["copilot_function_call_arguments"] == {
            "summary": "Saving my skill",
            "name": "my skill",
            "instructions": "keep it short",
        }


class TestIsClientFunctionCall:
    """Tests for is_client_function_call recognition of llm_save_skill."""

    def test_recognizes_llm_save_skill(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that llm_save_skill is recognized as a client function call."""
        mock_response = MagicMock()
        mock_response.function = MagicMock()
        mock_response.function.__name__ = "llm_save_skill"

        assert client_function_call_service.is_client_function_call(mock_response)

    def test_does_not_recognize_unknown_function(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that unknown function names are not client function calls."""
        mock_response = MagicMock()
        mock_response.function = MagicMock()
        mock_response.function.__name__ = "llm_unknown_function"

        assert not client_function_call_service.is_client_function_call(mock_response)


class TestGetClientTool:
    """Tests for get_client_tool function name mapping."""

    def test_maps_save_skill_to_llm_function(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that 'save_skill' maps to llm_save_skill."""
        tool = client_function_call_service.get_client_tool("save_skill")
        assert tool == client_function_call_service.llm_save_skill


class TestSaveSkillFunctionSignature:
    """Tests for llm_save_skill function signature and parameters."""

    def test_function_has_expected_parameters(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that llm_save_skill has the expected parameters."""
        import inspect

        sig = inspect.signature(client_function_call_service.llm_save_skill)
        param_names = list(sig.parameters.keys())

        assert "name" in param_names
        assert "instructions" in param_names
        assert "summary" in param_names
        assert "extra_state" in param_names

    def test_name_and_instructions_default_to_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that name and instructions parameters default to None."""
        import inspect

        sig = inspect.signature(client_function_call_service.llm_save_skill)
        assert sig.parameters["name"].default is None
        assert sig.parameters["instructions"].default is None


class TestHandleFunctionCallsSaveSkill:
    """Tests for handle_function_calls routing llm_save_skill."""

    @pytest.mark.asyncio
    async def test_handle_function_calls_yields_status_and_function_call_sse(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """handle_function_calls converts the llm_save_skill events to
        StatusUpdateSSE + FunctionCallSSE."""
        name = "earnings analysis"

        async def _fc_gen():
            yield StatusUpdateSSE(
                data=StatusUpdateSSEData(
                    eventType="INFO",
                    message="Saving skill",
                )
            )
            yield FunctionCallResponse(
                function="save_skill",
                input_arguments={"name": name, "instructions": None},
            )

        mock_response = MagicMock()
        mock_response.function = MagicMock()
        mock_response.function.__name__ = "llm_save_skill"
        mock_response.function.__qualname__ = "llm_save_skill"
        mock_response.arguments = {}
        mock_response.return_value = _fc_gen()

        events = []
        async for event in client_function_call_service.handle_function_calls(
            mock_response
        ):
            events.append(event)

        assert len(events) == 2
        assert isinstance(events[0], StatusUpdateSSE)
        assert events[0].data.message == "Saving skill"
        assert isinstance(events[1], FunctionCallSSE)
        assert events[1].data.function == "save_skill"
        assert events[1].data.input_arguments["name"] == name
