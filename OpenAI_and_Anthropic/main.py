"""OpenAI and Anthropic SDK examples plus a small FastAPI streaming API."""

import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, cast

import tiktoken
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
# This example is configured through its project .env file. Override stale
# shell-level placeholders so the key the user just saved here is actually used.
load_dotenv(BASE_DIR / ".env", override=True)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
SYSTEM_PROMPT = "You are a concise, helpful assistant."
TOKEN_LIMIT = int(os.getenv("TOKEN_LIMIT", "8000"))
USAGE_LOG = Path(os.getenv("USAGE_LOG", str(BASE_DIR / "usage.jsonl")))
OPENAI_INPUT_PER_MILLION = float(os.getenv("OPENAI_INPUT_PER_MILLION", "0.15"))
OPENAI_OUTPUT_PER_MILLION = float(os.getenv("OPENAI_OUTPUT_PER_MILLION", "0.60"))
ANTHROPIC_INPUT_PER_MILLION = float(os.getenv("ANTHROPIC_INPUT_PER_MILLION", "0.80"))
ANTHROPIC_OUTPUT_PER_MILLION = float(os.getenv("ANTHROPIC_OUTPUT_PER_MILLION", "4.00"))

app = FastAPI(title="SDK Calls, Streaming and Cost")


def _require_key(name: str) -> None:
    if not os.getenv(name):
        raise RuntimeError(f"Missing {name}; add it to {BASE_DIR / '.env'}.")


def _encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list[dict[str, str]], model: str = OPENAI_MODEL) -> int:
    """Estimate tokens in role/content messages before sending a request."""
    encoding = _encoding(model)
    # Include a small per-message allowance for role and message framing.
    return sum(4 + len(encoding.encode(str(m.get("role", "")))) +
               len(encoding.encode(str(m.get("content", "")))) for m in messages) + 2


def _cost(provider: str, input_tokens: int, output_tokens: int) -> float:
    if provider == "openai":
        input_rate, output_rate = OPENAI_INPUT_PER_MILLION, OPENAI_OUTPUT_PER_MILLION
    else:
        input_rate, output_rate = ANTHROPIC_INPUT_PER_MILLION, ANTHROPIC_OUTPUT_PER_MILLION
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _log_call(provider: str, model: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    cost = _cost(provider, input_tokens, output_tokens)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost,
    }
    USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with USAGE_LOG.open("a", encoding="utf-8") as log:
        log.write(json.dumps(record) + "\n")
    return record


def _check_limit(messages: list[dict[str, str]], model: str = OPENAI_MODEL) -> int:
    estimated = count_tokens(messages, model)
    if estimated > TOKEN_LIMIT:
        raise HTTPException(413, f"Request is about {estimated} tokens; limit is {TOKEN_LIMIT}.")
    return estimated


def ask_openai(prompt: str) -> tuple[str, str, dict[str, int]]:
    """Return answer, actual model, and usage from the OpenAI Chat Completions SDK."""
    _require_key("OPENAI_API_KEY")
    messages = [{"role": "user", "content": prompt}]
    _check_limit(messages)
    response = OpenAI().chat.completions.create(
        model=OPENAI_MODEL, messages=messages, temperature=0.7, max_tokens=300,
    )
    usage = response.usage
    tokens = {
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0,
    }
    _log_call("openai", response.model, tokens["input_tokens"], tokens["output_tokens"])
    return response.choices[0].message.content or "", response.model, tokens


def ask_claude(prompt: str, system: str) -> tuple[str, str, dict[str, int]]:
    """Return answer, actual model, and usage from the Anthropic Messages SDK."""
    _require_key("ANTHROPIC_API_KEY")
    messages = [{"role": "user", "content": prompt}]
    _check_limit([{"role": "system", "content": system}, *messages], ANTHROPIC_MODEL)
    response = Anthropic().messages.create(
        model=ANTHROPIC_MODEL, max_tokens=300, messages=messages, system=system,
    )
    answer = "\n".join(block.text for block in response.content if block.type == "text")
    usage = {"input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens,
             "total_tokens": response.usage.input_tokens + response.usage.output_tokens}
    _log_call("anthropic", response.model, usage["input_tokens"], usage["output_tokens"])
    return answer, response.model, usage


class ChatRequest(BaseModel):
    prompt: str
    system: str = SYSTEM_PROMPT


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Stream OpenAI text chunks as Server-Sent Events, followed by usage."""
    _require_key("OPENAI_API_KEY")
    messages = [{"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt}]
    _check_limit(messages)

    def events():
        model = OPENAI_MODEL
        input_tokens = output_tokens = 0
        try:
            stream = OpenAI().chat.completions.create(
                model=OPENAI_MODEL,
                messages=cast(list[ChatCompletionMessageParam], messages),
                max_tokens=300,
                stream=True,
                stream_options={"include_usage": True},
            )
            for chunk in stream:
                if chunk.model:
                    model = chunk.model
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or 0
                    output_tokens = chunk.usage.completion_tokens or 0
                if chunk.choices:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield f"data: {json.dumps({'text': content})}\n\n"
        except Exception:
            # Keep failures in the SSE protocol and avoid exposing provider details.
            yield f"event: error\ndata: {json.dumps({'error': 'Provider request failed. Check API key, model, and network.'})}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return
        usage = _log_call("openai", model, input_tokens, output_tokens)
        yield f"event: usage\ndata: {json.dumps(usage)}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/usage")
def get_usage():
    """Aggregate today's logged calls, using the machine's local calendar day."""
    today = date.today()
    total_tokens = 0
    total_cost = 0.0
    calls = 0
    if USAGE_LOG.exists():
        with USAGE_LOG.open(encoding="utf-8") as log:
            for line in log:
                try:
                    entry = json.loads(line)
                    timestamp = datetime.fromisoformat(entry["timestamp"])
                    if timestamp.astimezone().date() == today:
                        calls += 1
                        total_tokens += int(entry.get("total_tokens", 0))
                        total_cost += float(entry.get("cost_usd", 0))
                except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
    return {"date": today.isoformat(), "calls": calls, "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 8)}


QUESTIONS = [
    "Why is the sky blue?",
    "What is one practical benefit of learning Python?",
    "Explain photosynthesis in one sentence.",
]


def compare_streaming(prompt: str) -> None:
    """Print measured time to first content for regular and streaming calls."""
    _require_key("OPENAI_API_KEY")
    messages = [{"role": "user", "content": prompt}]
    _check_limit(messages)
    client = OpenAI()
    started = time.perf_counter()
    response = client.chat.completions.create(model=OPENAI_MODEL, messages=messages,
                                              max_tokens=100)
    nonstream_elapsed = time.perf_counter() - started
    nonstream_answer = response.choices[0].message.content or ""
    usage = response.usage
    _log_call("openai", response.model, usage.prompt_tokens if usage else 0,
              usage.completion_tokens if usage else 0)
    print(f"Non-streaming: first output after {nonstream_elapsed:.2f} s")
    print(nonstream_answer)

    started = time.perf_counter()
    stream = client.chat.completions.create(model=OPENAI_MODEL, messages=messages,
                                            max_tokens=100, stream=True,
                                            stream_options={"include_usage": True})
    first_token = None
    pieces = []
    input_tokens = output_tokens = 0
    for chunk in stream:
        if chunk.usage:
            input_tokens, output_tokens = chunk.usage.prompt_tokens or 0, chunk.usage.completion_tokens or 0
        if chunk.choices and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            if first_token is None:
                first_token = time.perf_counter() - started
            pieces.append(text)
    print(f"Streaming: first token after {(first_token or time.perf_counter() - started):.2f} s")
    print("".join(pieces))
    _log_call("openai", OPENAI_MODEL, input_tokens, output_tokens)


def main() -> None:
    for question in QUESTIONS:
        print(f"\nQUESTION: {question}")
        try:
            answer, model, usage = ask_openai(question)
        except RateLimitError as exc:
            details = str(exc).lower()
            if "no credits" in details or "insufficient_quota" in details:
                print(
                    "OpenAI API credits are exhausted. Add API billing credits or "
                    "enable billing for the API project, then run the program again."
                )
            else:
                print(f"OpenAI rate limit reached: {exc}")
            return
        print(f"\nOpenAI ({model})\n{answer}\nToken usage: {usage}")
        answer, model, usage = ask_claude(question, SYSTEM_PROMPT)
        print(f"\nAnthropic ({model})\n{answer}\nToken usage: {usage}")
    compare_streaming("In one sentence, explain why streaming can feel faster.")


if __name__ == "__main__":
    main()
