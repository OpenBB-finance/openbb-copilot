"""
Tests for skill-related functionality in ClientFunctionCallService.

Tests the get_skill_content function call handling including:
- Function call generation
- Client tool mapping
- Function call specification
"""

from unittest.mock import MagicMock, Mock

import pytest
from openbb_ai.models import FunctionCallResponse, FunctionCallSSE

from openbb_ada.services import ClientFunctionCallService, LoggingService


@pytest.fixture
def client_function_call_service() -> ClientFunctionCallService:
    """Create a ClientFunctionCallService instance for testing."""
    return ClientFunctionCallService(
        copilot_data_service=Mock(),
        logging_service=LoggingService(),
    )


class TestLlmGetSkillContent:
    """Tests for llm_get_skill_content function."""

    @pytest.mark.asyncio
    async def test_yields_function_call_response(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that llm_get_skill_content yields a FunctionCallResponse."""
        events = []
        async for event in client_function_call_service.llm_get_skill_content(
            slug="financial-analysis",
            reason="User needs financial analysis",
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], FunctionCallResponse)

    @pytest.mark.asyncio
    async def test_function_call_response_has_correct_function_name(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that the function name is 'get_skill_content'."""
        events = []
        async for event in client_function_call_service.llm_get_skill_content(
            slug="test-skill",
        ):
            events.append(event)

        assert isinstance(events[0], FunctionCallResponse)
        assert events[0].function == "get_skill_content"

    @pytest.mark.asyncio
    async def test_function_call_response_includes_slug(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that the input_arguments include the slug."""
        events = []
        async for event in client_function_call_service.llm_get_skill_content(
            slug="my-skill",
        ):
            events.append(event)

        assert isinstance(events[0], FunctionCallResponse)
        assert events[0].input_arguments is not None
        assert events[0].input_arguments["slug"] == "my-skill"

    @pytest.mark.asyncio
    async def test_function_call_response_includes_reason_when_provided(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that the input_arguments include reason when provided."""
        events = []
        async for event in client_function_call_service.llm_get_skill_content(
            slug="test-skill",
            reason="Testing the skill",
        ):
            events.append(event)

        assert isinstance(events[0], FunctionCallResponse)
        assert events[0].input_arguments is not None
        assert events[0].input_arguments["reason"] == "Testing the skill"

    @pytest.mark.asyncio
    async def test_function_call_response_omits_reason_when_not_provided(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that reason is None when not provided."""
        events = []
        async for event in client_function_call_service.llm_get_skill_content(
            slug="test-skill",
        ):
            events.append(event)

        assert isinstance(events[0], FunctionCallResponse)
        assert events[0].input_arguments is not None
        assert events[0].input_arguments.get("reason") is None


class TestGetClientTool:
    """Tests for get_client_tool function name mapping."""

    def test_maps_get_skill_content_to_llm_function(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that 'get_skill_content' maps to llm_get_skill_content."""
        tool = client_function_call_service.get_client_tool("get_skill_content")
        assert tool == client_function_call_service.llm_get_skill_content

    def test_raises_for_unknown_function(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that unknown function names raise HTTPException."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            client_function_call_service.get_client_tool("unknown_function")

        assert exc_info.value.status_code == 500


class TestGetSkillContentFunctionSignature:
    """Tests for llm_get_skill_content function signature and parameters."""

    def test_function_has_required_parameters(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that llm_get_skill_content has the expected parameters."""
        import inspect

        sig = inspect.signature(client_function_call_service.llm_get_skill_content)
        param_names = list(sig.parameters.keys())

        assert "slug" in param_names
        assert "reason" in param_names
        assert "summary" in param_names
        assert "extra_state" in param_names

    def test_reason_has_default_none(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """Test that reason parameter defaults to None."""
        import inspect

        sig = inspect.signature(client_function_call_service.llm_get_skill_content)
        reason_param = sig.parameters.get("reason")
        assert reason_param is not None
        assert reason_param.default is None


class TestHandleFunctionCallsSkillContent:
    """Tests for handle_function_calls routing llm_get_skill_content."""

    @pytest.mark.asyncio
    async def test_handle_function_calls_yields_function_call_sse(
        self, client_function_call_service: ClientFunctionCallService
    ):
        """handle_function_calls converts FunctionCallResponse to FunctionCallSSE."""
        slug = "financial-analysis"

        async def _fc_gen():
            yield FunctionCallResponse(
                function="get_skill_content",
                input_arguments={"slug": slug},
            )

        mock_response = MagicMock()
        mock_response.function = MagicMock()
        mock_response.function.__name__ = "llm_get_skill_content"
        mock_response.function.__qualname__ = "llm_get_skill_content"
        mock_response.arguments = {}
        mock_response.return_value = _fc_gen()

        events = []
        async for event in client_function_call_service.handle_function_calls(
            mock_response
        ):
            events.append(event)

        assert len(events) == 1
        assert isinstance(events[0], FunctionCallSSE)
        assert events[0].data.function == "get_skill_content"
        assert events[0].data.input_arguments["slug"] == slug
