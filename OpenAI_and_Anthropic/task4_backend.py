"""Task 4: dependency-injected, API-key-secured FastAPI chat backend."""

from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Protocol

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)


@dataclass(frozen=True)
class Settings:
    api_key: str
    openai_api_key: str
    model: str = "gpt-4o-mini"
    cors_origins: tuple[str, str] = (
        "http://localhost:3000",
        "http://localhost:5173",
    )


def get_settings() -> Settings:
    """Read the backend API key, provider key, model, and two allowed origins."""
    import os

    origins = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
        ).split(",")
        if origin.strip()
    )
    if len(origins) != 2:
        origins = ("http://localhost:3000", "http://localhost:5173")
    return Settings(
        api_key=os.getenv("APP_API_KEY", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        cors_origins=(origins[0], origins[1]),
    )


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2000)


class SessionCreated(BaseModel):
    session_id: str


class StoredMessage(BaseModel):
    role: str
    content: str
    created_at: datetime


class MessageList(BaseModel):
    session_id: str
    messages: list[StoredMessage]


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAIChatClient:
    """Small lazy client wrapper; tests replace it through dependency_overrides."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def complete(self, messages: list[dict[str, str]]) -> str:
        from openai import AsyncOpenAI

        async with AsyncOpenAI(
            api_key=self.settings.openai_api_key or None,
            max_retries=0,
        ) as client:
            response = await client.chat.completions.create(
                model=self.settings.model,
                messages=messages,
                max_tokens=1000,
            )
        return response.choices[0].message.content or ""


class InMemoryStore:
    """Minimal session store; replaceable with a database dependency later."""

    def __init__(self) -> None:
        self.sessions: dict[str, list[StoredMessage]] = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        return session_id

    def has_session(self, session_id: str) -> bool:
        return session_id in self.sessions

    def add_message(self, session_id: str, role: str, content: str) -> StoredMessage:
        message = StoredMessage(
            role=role,
            content=content,
            created_at=datetime.now(timezone.utc),
        )
        self.sessions[session_id].append(message)
        return message

    def get_messages(self, session_id: str) -> list[StoredMessage]:
        return list(self.sessions[session_id])


_DEFAULT_STORE = InMemoryStore()


def get_store() -> InMemoryStore:
    return _DEFAULT_STORE


def get_llm_client(settings: Annotated[Settings, Depends(get_settings)]) -> LLMClient:
    return OpenAIChatClient(settings)


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="x-api-key")] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    if not settings.api_key:
        raise HTTPException(status_code=503, detail="API key authentication is not configured")
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(status_code=403, detail="Invalid API key")


def create_app() -> FastAPI:
    application = FastAPI(title="Secured Chat Backend", version="1.0.0")
    settings = get_settings()
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "x-api-key"],
    )

    @application.exception_handler(HTTPException)
    async def http_error_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        messages = {
            401: ("unauthorized", "Authentication is required"),
            403: ("forbidden", "Authentication failed"),
            404: ("not_found", "The requested session was not found"),
            413: ("payload_too_large", "Message exceeds the 2000 character limit"),
            422: ("validation_error", "Request validation failed"),
            502: ("llm_error", "The language model request failed"),
            503: ("service_unavailable", "The service is not configured"),
        }
        code, default_message = messages.get(exc.status_code, ("http_error", "Request failed"))
        detail = exc.detail if isinstance(exc.detail, str) else default_message
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": code, "message": detail}},
            headers=exc.headers,
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        too_long = any(error.get("type") == "string_too_long" for error in errors)
        status = 413 if too_long else 422
        code = "payload_too_large" if too_long else "validation_error"
        message = (
            "Message exceeds the 2000 character limit"
            if too_long
            else "Request validation failed"
        )
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "details": [
                        {"location": error.get("loc", []), "type": error.get("type", "")}
                        for error in errors
                    ],
                }
            },
        )

    @application.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.post("/sessions", response_model=SessionCreated, status_code=201)
    async def create_session(
        _auth: Annotated[None, Depends(require_api_key)],
        store: Annotated[InMemoryStore, Depends(get_store)],
    ) -> SessionCreated:
        return SessionCreated(session_id=store.create_session())

    @application.post("/sessions/{session_id}/chat", response_model=MessageList)
    async def chat(
        session_id: str,
        body: ChatRequest,
        _auth: Annotated[None, Depends(require_api_key)],
        store: Annotated[InMemoryStore, Depends(get_store)],
        llm: Annotated[LLMClient, Depends(get_llm_client)],
    ) -> MessageList:
        if not store.has_session(session_id):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        history = [
            {"role": message.role, "content": message.content}
            for message in store.get_messages(session_id)
        ]
        history.append({"role": "user", "content": body.message})
        try:
            answer = await llm.complete(history)
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Language model request failed") from exc
        store.add_message(session_id, "user", body.message)
        store.add_message(session_id, "assistant", answer)
        return MessageList(session_id=session_id, messages=store.get_messages(session_id))

    @application.get("/sessions/{session_id}/messages", response_model=MessageList)
    async def get_session_messages(
        session_id: str,
        _auth: Annotated[None, Depends(require_api_key)],
        store: Annotated[InMemoryStore, Depends(get_store)],
    ) -> MessageList:
        if not store.has_session(session_id):
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return MessageList(session_id=session_id, messages=store.get_messages(session_id))

    return application


app = create_app()
