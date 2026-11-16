from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from task4_backend import (
    InMemoryStore,
    Settings,
    app,
    get_llm_client,
    get_settings,
    get_store,
)


API_KEY = "test-api-key"
AUTH = {"x-api-key": API_KEY}


@pytest.fixture
def setup_client():
    store = InMemoryStore()
    llm = SimpleNamespace(complete=AsyncMock(return_value="mock answer"))
    app.dependency_overrides[get_settings] = lambda: Settings(
        api_key=API_KEY,
        openai_api_key="unused-test-key",
        cors_origins=("http://localhost:3000", "http://localhost:5173"),
    )
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_llm_client] = lambda: llm
    with TestClient(app) as client:
        yield client, store, llm
    app.dependency_overrides.clear()


def create_session(client: TestClient) -> str:
    response = client.post("/sessions", headers=AUTH)
    assert response.status_code == 201
    return response.json()["session_id"]


def test_health_is_public(setup_client):
    client, _store, _llm = setup_client
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_missing_key_returns_401_with_standard_error_body(setup_client):
    client, _store, _llm = setup_client
    response = client.post("/sessions")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


def test_invalid_key_returns_403(setup_client):
    client, _store, _llm = setup_client
    response = client.post("/sessions", headers={"x-api-key": "wrong"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"


def test_unconfigured_auth_never_accepts_an_empty_header(setup_client):
    client, _store, _llm = setup_client
    app.dependency_overrides[get_settings] = lambda: Settings(api_key="", openai_api_key="")
    response = client.post("/sessions", headers={"x-api-key": ""})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "service_unavailable"


def test_create_session_returns_id(setup_client):
    client, store, _llm = setup_client
    response = client.post("/sessions", headers=AUTH)
    assert response.status_code == 201
    session_id = response.json()["session_id"]
    assert store.has_session(session_id)


def test_empty_message_returns_422(setup_client):
    client, _store, _llm = setup_client
    session_id = create_session(client)
    response = client.post(
        f"/sessions/{session_id}/chat", headers=AUTH, json={"message": ""}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


def test_message_over_2000_chars_returns_413(setup_client):
    client, _store, _llm = setup_client
    session_id = create_session(client)
    response = client.post(
        f"/sessions/{session_id}/chat",
        headers=AUTH,
        json={"message": "x" * 2001},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_chat_sends_history_to_mock_and_stores_both_messages(setup_client):
    client, store, llm = setup_client
    session_id = create_session(client)
    response = client.post(
        f"/sessions/{session_id}/chat", headers=AUTH, json={"message": "hello"}
    )
    assert response.status_code == 200
    assert [item["role"] for item in response.json()["messages"]] == ["user", "assistant"]
    llm.complete.assert_awaited_once_with([{"role": "user", "content": "hello"}])
    assert len(store.get_messages(session_id)) == 2


def test_get_session_messages_returns_history(setup_client):
    client, _store, _llm = setup_client
    session_id = create_session(client)
    client.post(f"/sessions/{session_id}/chat", headers=AUTH, json={"message": "hello"})
    response = client.get(f"/sessions/{session_id}/messages", headers=AUTH)
    assert response.status_code == 200
    assert response.json()["messages"][1]["content"] == "mock answer"


def test_unknown_session_returns_404(setup_client):
    client, _store, llm = setup_client
    response = client.get("/sessions/missing/messages", headers=AUTH)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    llm.complete.assert_not_awaited()


def test_llm_failure_returns_502_and_does_not_store_user_message(setup_client):
    client, store, llm = setup_client
    session_id = create_session(client)
    llm.complete.side_effect = RuntimeError("mock provider outage")
    response = client.post(
        f"/sessions/{session_id}/chat", headers=AUTH, json={"message": "hello"}
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "llm_error"
    assert store.get_messages(session_id) == []


def test_allowed_cors_origin_is_returned(setup_client):
    client, _store, _llm = setup_client
    response = client.options(
        "/sessions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key,content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_disallowed_cors_origin_is_not_reflected(setup_client):
    client, _store, _llm = setup_client
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
