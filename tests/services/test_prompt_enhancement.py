"""Tests for the prompt enhancement service availability throughout conversation."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from magentic import UserMessage

from openbb_ada.copilot import CopilotService
from openbb_ada.services import (
    ClientFunctionCallService,
    LoggingService,
    NativeFunctionCallService,
    PromptEnhancementService,
)


@pytest.fixture
def mock_prompt_enhancement_service():
    """Create a mock prompt enhancement service."""
    service = Mock(spec=PromptEnhancementService)
    service.enhance_prompt = AsyncMock(return_value="Enhanced query for better clarity")
    service.set_current_context = Mock()
    service.get_current_context = Mock(return_value=([], None, None, None))
    return service


@pytest.fixture
def copilot_with_enhancement(
    test_document_service,
    test_template_service,
    mock_prompt_enhancement_service,
):
    """Create a copilot service with prompt enhancement enabled."""
    # Create a proper mock context service
    mock_context_service = Mock()
    mock_context_service.unstructured_context = []
    mock_context_service.structured_context = []

    native_function_call_service = NativeFunctionCallService(
        document_service=test_document_service,
        logging_service=LoggingService(),
        web_search_llm_service=Mock(),
        context_service=mock_context_service,
        template_service=test_template_service,
        citation_service=Mock(),
        prompt_enhancement_service=mock_prompt_enhancement_service,
    )

    copilot = CopilotService(
        user_id="test-user-id",
        document_service=test_document_service,
        context_service=mock_context_service,
        template_service=test_template_service,
        url_retrieval_service=Mock(),
        copilot_data_service=Mock(),
        logging_service=LoggingService(),
        client_function_call_service=ClientFunctionCallService(
            copilot_data_service=Mock(),
            logging_service=LoggingService(),
        ),
        native_function_call_service=native_function_call_service,
        citation_service=Mock(),
        mcp_data_service=Mock(),
        openai_api_key="test-key",
        workspace_options={"workspace-web-search": True},
        prompt_enhancement_service=mock_prompt_enhancement_service,
    )
    return copilot


@pytest.mark.asyncio
async def test_prompt_enhancement_available_on_all_calls(copilot_with_enhancement):
    """Test that prompt enhancement is available on all calls, not just the first."""
    copilot = copilot_with_enhancement

    # Mock the copilot data service
    mock_widget_collection = Mock()
    mock_widget_collection.primary = None
    mock_widget_collection.secondary = None
    copilot._copilot_data_service.get_widget_collection.return_value = (
        mock_widget_collection
    )

    # Create chat messages for testing
    chat_messages = [UserMessage("analyze the data")]

    with patch.object(copilot, "_get_model") as mock_get_model:
        mock_model = Mock()
        mock_model.model = "gpt-4.1"
        mock_get_model.return_value = mock_model

        # Track function availability across calls
        functions_by_call = []

        # Patch chatprompt to capture functions
        with patch("openbb_ada.copilot.chatprompt") as mock_chatprompt:

            def capture_functions(*args, **kwargs):
                # Capture the functions argument
                if "functions" in kwargs and kwargs["functions"]:
                    has_enhancement = any(
                        func.__name__ == "llm_enhance_prompt"
                        for func in kwargs["functions"]
                        if hasattr(func, "__name__")
                    )
                    functions_by_call.append(has_enhancement)
                # Return a mock callable
                return Mock()

            mock_chatprompt.side_effect = capture_functions

            # First call - with original_messages passed
            await copilot._compose_chain(
                chat_messages=chat_messages,
                documents=None,
                web_pages=None,
                tools=None,
                original_messages=[],  # Messages are passed
                original_context=None,
            )

            # Second call - still with original_messages (per our fix)
            chat_messages.append(UserMessage("compare it with competitors"))
            await copilot._compose_chain(
                chat_messages=chat_messages,
                documents=None,
                web_pages=None,
                tools=None,
                original_messages=[],  # Messages still passed
                original_context=None,
            )

            # Third call - messages continue to be passed
            chat_messages.append(UserMessage("show me everything"))
            await copilot._compose_chain(
                chat_messages=chat_messages,
                documents=None,
                web_pages=None,
                tools=None,
                original_messages=[],  # Messages still passed
                original_context=None,
            )

            # Verify enhancement function was available in all calls
            assert len(functions_by_call) == 3, "Should have captured 3 calls"
            assert all(functions_by_call), (
                "Enhancement function should be available in all calls"
            )


@pytest.mark.asyncio
async def test_prompt_enhancement_service_setup():
    """Test that the prompt enhancement service is properly set up when provided."""
    # Create a mock enhancement service
    mock_enhancement_service = Mock(spec=PromptEnhancementService)
    mock_enhancement_service.enhance_prompt = AsyncMock(return_value="Enhanced query")
    mock_enhancement_service.get_current_context = Mock(
        return_value=(["msg"], ["ctx"], "widgets", ["tool"])
    )
    mock_enhancement_service.set_current_context = Mock()

    # Create a proper mock context service
    mock_context_service = Mock()
    mock_context_service.unstructured_context = []
    mock_context_service.structured_context = []

    # Create native function call service with enhancement
    native_service = NativeFunctionCallService(
        document_service=Mock(),
        logging_service=LoggingService(),
        web_search_llm_service=Mock(),
        context_service=mock_context_service,
        template_service=Mock(),
        citation_service=Mock(),
        prompt_enhancement_service=mock_enhancement_service,
    )

    # Verify the enhancement function exists
    assert hasattr(native_service, "llm_enhance_prompt"), (
        "Enhancement function should exist"
    )
    assert callable(native_service.llm_enhance_prompt), (
        "Enhancement function should be callable"
    )

    # Verify it can set context
    async for _ in native_service.llm_enhance_prompt(
        reasoning="Needs clarity",
    ):
        pass
    mock_enhancement_service.enhance_prompt.assert_awaited_once_with(
        messages=["msg"],
        context=["ctx"],
        widgets="widgets",
        tools=["tool"],
    )
