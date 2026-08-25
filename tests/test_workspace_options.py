from unittest.mock import patch

import pytest
from openbb_ai.models import WorkspaceState
from pydantic import ValidationError

from openbb_ada.models import AdaQueryRequest
from openbb_ada.services import TemplateService


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"generative-ui": True},
        {"workspace-web-search": True},
        {"widget-global-search": True},
        {"agent-orchestration": True},
        {"prompt-suggestions": True},
        {"generative-ui": True, "workspace-web-search": True},
        {
            "generative-ui": True,
            "workspace-web-search": True,
            "widget-global-search": True,
        },
    ],
    ids=[
        "no_options",
        "generative_ui",
        "web_search",
        "global_search",
        "agent_orchestration",
        "prompt_suggestions",
        "generative_ui_web_search",
        "generative_ui_web_search_global_search",
    ],
)
async def test_templates_render_without_errors(options):
    """Test that templates render successfully with different workspace options."""
    template_service = TemplateService(workspace_options=options)

    # Should render without errors
    context_result = template_service.render_copilot_context()
    system_result = template_service.render_copilot_system_prompt()

    # Basic structure checks
    assert isinstance(context_result, str)
    assert isinstance(system_result, str)
    assert len(context_result) > 10
    assert len(system_result) > 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workspace_options, should_have_generative_ui",
    [
        ({}, False),
        ({"generative-ui": True}, True),
    ],
    ids=["disabled", "enabled"],
)
async def test_generative_ui_dashboard_features(
    workspace_options, should_have_generative_ui
):
    """Test generative UI features on dashboard."""
    dashboard_state = WorkspaceState(
        current_page_context="dashboard",
        current_dashboard_uuid="550e8400-e29b-41d4-a716-446655440000",
        action_history=[],
        agents=[],
    )

    service = TemplateService(
        workspace_state=dashboard_state,
        workspace_options=workspace_options,
    )
    context = service.render_copilot_context()

    if should_have_generative_ui:
        assert "add new widgets to the dashboard" in context
        assert "llm_add_widget_to_dashboard" in context
        assert "Generative UI is currently disabled" not in context
    else:
        assert "Generative UI is currently disabled" in context


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workspace_options, should_contain",
    [
        ({}, False),
        ({"workspace-web-search": True}, True),
    ],
    ids=["disabled", "enabled"],
)
async def test_web_search_system_prompt(workspace_options, should_contain):
    """Test web search appears in system prompt when enabled."""
    with patch("openbb_ada.constants.LLM_WEB_SEARCH_ENABLED", True):
        service = TemplateService(workspace_options=workspace_options)
        system_result = service.render_copilot_system_prompt()

    if should_contain:
        assert "Web Search Strategy" in system_result
        assert "_llm_web_search" in system_result
    else:
        assert "Web Search Strategy" not in system_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workspace_options, should_contain",
    [
        ({}, False),
        ({"widget-global-search": True}, True),
    ],
    ids=["disabled", "enabled"],
)
async def test_global_data_system_prompt(workspace_options, should_contain):
    """Test global data appears in system prompt when enabled."""
    service = TemplateService(workspace_options=workspace_options)
    system_result = service.render_copilot_system_prompt()

    if should_contain:
        assert "No extra widgets found / added to the Copilot." in system_result
    else:
        assert "No extra widgets found / added to the Copilot." not in system_result
        assert "Global widget search is not enabled." in system_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workspace_options, should_contain",
    [
        ({}, False),
        ({"agent-orchestration": True}, True),
    ],
    ids=["disabled", "enabled"],
)
async def test_agent_orchestration_context(workspace_options, should_contain):
    """Test agent orchestration appears in context when enabled."""
    dashboard_state = WorkspaceState(
        current_page_context="dashboard",
        current_dashboard_uuid="550e8400-e29b-41d4-a716-446655440000",
        action_history=[],
        agents=[],
    )

    service = TemplateService(
        workspace_state=dashboard_state,
        workspace_options=workspace_options,
    )
    context = service.render_copilot_context()

    if should_contain:
        assert "Agent Orchestration Mode - ACTIVE" in context
    else:
        assert "Agent Orchestration Mode - ACTIVE" not in context


def test_prompt_suggestions_system_prompt_without_dashboard_context():
    service = TemplateService(
        workspace_state=WorkspaceState(current_page_context="settings"),
        workspace_options={"prompt-suggestions": True},
    )

    system_prompt = service.render_copilot_system_prompt()

    assert "Follow-up suggestions" in system_prompt
    assert "<suggestions>" in system_prompt
    assert "default to appending exactly one hidden `<suggestions>` block" in (
        system_prompt
    )
    assert "Actively look for useful, relevant follow-ups" in system_prompt
    assert "Only omit the entire `<suggestions>` block" in system_prompt
    assert 'Do not use the word "widget" in suggestions' in system_prompt
    assert "Since the user is not on a dashboard" in system_prompt
    assert "MCP tools, or selected skills" in system_prompt
    assert "OpenBB docs or marketplace apps" in system_prompt
    assert "Summarize the main takeaways from this dashboard" not in system_prompt


def test_prompt_suggestions_system_prompt_with_dashboard_context():
    service = TemplateService(
        workspace_state=WorkspaceState(
            current_page_context="dashboard",
            current_dashboard_uuid="550e8400-e29b-41d4-a716-446655440000",
            current_dashboard_info={
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "Markets",
                "current_tab_id": "overview",
                "tabs": [],
            },
            action_history=[],
            agents=[],
        ),
        workspace_options={"prompt-suggestions": True},
    )

    system_prompt = service.render_copilot_system_prompt()

    assert "Follow-up suggestions" in system_prompt
    assert "Since the user is not on a dashboard" not in system_prompt
    assert "Summarize the main takeaways from this dashboard" in system_prompt


@pytest.mark.asyncio
async def test_agent_orchestration_system_prompt_includes_available_agents():
    """Test live system prompt includes the concrete agent roster."""
    dashboard_state = WorkspaceState(
        current_page_context="dashboard",
        current_dashboard_uuid="550e8400-e29b-41d4-a716-446655440000",
        action_history=[],
        agents=[
            {
                "holder_url": "https://agents.example.com",
                "id": "research-agent",
                "name": "Research Agent",
                "description": "Researches topics and improves prompts.",
                "features": {"streaming": True, "agent-orchestration": False},
            }
        ],
    )

    service = TemplateService(
        workspace_state=dashboard_state,
        workspace_options={"agent-orchestration": True},
    )
    system_prompt = service.render_copilot_system_prompt()

    assert "### Available Agents" in system_prompt
    assert "<agents>" in system_prompt
    assert "holder_url: https://agents.example.com" in system_prompt
    assert "id: research-agent" in system_prompt
    assert "name: Research Agent" in system_prompt
    assert "description: Researches topics and improves prompts." in system_prompt
    assert (
        "features: {'streaming': True, 'agent-orchestration': False}" in system_prompt
    )


def test_request_strips_agent_orchestration_in_snowflake_native_app(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("openbb_ada.constants.SNOWFLAKE_NATIVE_APP", True)

    request = AdaQueryRequest(
        messages=[{"role": "human", "content": "hello"}],
        workspace_options={"agent-orchestration": True, "generative-ui": True},
    )

    assert request.workspace_options == {"generative-ui": True}


def test_request_keeps_agent_orchestration_outside_snowflake_native_app(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("openbb_ada.constants.SNOWFLAKE_NATIVE_APP", False)

    request = AdaQueryRequest(
        messages=[{"role": "human", "content": "hello"}],
        workspace_options={"agent-orchestration": True, "generative-ui": True},
    )

    assert request.workspace_options == {
        "agent-orchestration": True,
        "generative-ui": True,
    }


def test_request_rejects_legacy_workspace_options_list():
    with pytest.raises(ValidationError):
        AdaQueryRequest(
            messages=[{"role": "human", "content": "hello"}],
            workspace_options=["agent-orchestration", "generative-ui"],
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workspace_options, current_page_context",
    [
        ({"generative-ui": True}, "settings"),
        ({"generative-ui": True}, "dashboard"),
    ],
    ids=["settings_context", "dashboard_context"],
)
async def test_non_dashboard_context(workspace_options, current_page_context):
    """Test behavior when not on dashboard."""
    settings_state = WorkspaceState(
        current_page_context=current_page_context,
        current_dashboard_uuid=None
        if current_page_context != "dashboard"
        else "550e8400-e29b-41d4-a716-446655440000",
        action_history=[],
        agents=[],
    )

    service = TemplateService(
        workspace_state=settings_state,
        workspace_options=workspace_options,
    )
    context = service.render_copilot_context()

    if current_page_context != "dashboard":
        # Match intent-level constraints instead of exact sentence formatting.
        assert "current page" in context.lower()
        assert "not on a dashboard" in context.lower()
        assert "you cannot" in context.lower()
        assert "add" in context.lower() and "update" in context.lower()
        assert "dashboard" in context.lower()
    else:
        assert "dashboard state" in context.lower()
        assert "Current dashboard id:" in context
        assert (
            "You cannot add or update widgets since the user is not on a dashboard"
            not in context
        )
