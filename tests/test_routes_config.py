import pytest
from fastapi.testclient import TestClient


def test_agents_json_hides_agent_orchestration_in_snowflake_native_app(
    test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("openbb_ada.main.SNOWFLAKE_NATIVE_APP", True)

    response = test_client.get("/agents.json")

    assert response.status_code == 200
    features = response.json()["openbb_ada"]["features"]
    assert "agent-orchestration" not in features


def test_agents_json_includes_prompt_suggestions_feature(test_client: TestClient):
    response = test_client.get("/agents.json")

    assert response.status_code == 200
    feature = response.json()["openbb_ada"]["features"]["prompt-suggestions"]
    assert feature == {
        "label": "Follow-up Suggestions",
        "default": True,
        "description": "Show follow-up prompt suggestions after each response.",
    }


def test_query_disabled_web_search_returns_501(
    test_client: TestClient,
    no_rate_limit: None,
    mock_headers: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "openbb_ada.main.LLM_WEB_SEARCH_ENABLED",
        False,
    )

    payload = {
        "messages": [
            {
                "role": "human",
                "content": "What is the capital of France?",
            },
        ],
        "force_web_search": True,
    }
    response = test_client.post(
        "/v1/query",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 501


def test_query_disabled_url_retrieval_returns_501(
    test_client: TestClient,
    no_rate_limit: None,
    mock_headers: str,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "openbb_ada.main.URL_RETRIEVAL_ENABLED",
        False,
    )

    payload = {
        "messages": [
            {
                "role": "human",
                "content": "Tell me what's on this website?",
            },
        ],
        "urls": ["https://wikipedia.org"],
    }
    response = test_client.post(
        "/v1/query",
        headers=mock_headers,
        json=payload,
    )
    assert response.status_code == 501
