# SDK Calls, Streaming and Cost

This example uses the official OpenAI and Anthropic Python SDKs, adds OpenAI streaming over Server-Sent Events, estimates request tokens with `tiktoken`, and logs usage and estimated USD cost to `usage.jsonl`.

## Setup

1. Use Python 3.14 (the screenshot's Python 3.16 is a prerelease and the existing environment was created for 3.14). Create a fresh environment and install dependencies into it:

   ```sh
   python3.14 -m venv .venv314
   ./.venv314/bin/python -m pip install -r requirements.txt
   ```

   This installs `tiktoken` (and the SDK/FastAPI dependencies) into `.venv314`. Use this environment when running the app.
2. Copy `.env.example` to `.env`, then enter your API keys. Never commit `.env`.
3. Run both provider examples and compare time to first output/token: `./.venv314/bin/python main.py`.

The cost rates in `.env.example` are configurable estimates in USD per million tokens. Update them to match the model IDs and current provider pricing you use. The local usage log contains one JSON record per completed API call.

## FastAPI streaming endpoint

Start the server from this directory:

```sh
./.venv314/bin/python -m uvicorn main:app --reload
```

In another terminal, stream the SSE response:

```sh
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain photosynthesis in one sentence."}'
```

Text arrives in `data:` events, followed by a `usage` event and a `done` event. `curl -N` disables output buffering so chunks are visible as they arrive. `GET /usage` reports today's calls, tokens, and estimated cost.

## Functions and request formats

- `ask_openai(prompt)` uses `client.chat.completions.create(...)`.
- `ask_claude(prompt, system)` uses `client.messages.create(...)` with a separate `system` argument.
- Both return `(answer, model, usage)` and write a usage/cost record.
- `count_tokens(messages)` estimates the input size before sending; requests above `TOKEN_LIMIT` are refused.
- `compare_streaming(prompt)` measures time until a complete non-streaming response and time until the first streaming text chunk. The actual times depend on network, model, and load.

## Task 2: Reliability, retries, and async

Run the mock-only retry proof without contacting either provider or using API credits:

```sh
./.venv314/bin/python task2_reliability.py --mock-only
```

The mock responds with two 429 errors and then succeeds; the output records 1 s and 2 s exponential waits. A separate mocked 401 is called once, proving that authentication failures are not retried. The retry policy stops after five attempts and retries only 429 and HTTP 5xx errors. Every retry logs its attempt number, status, and planned wait.

Run the async timing comparison after the OpenAI API project has usable credits:

```sh
./.venv314/bin/python task2_reliability.py
```

This makes 20 live requests: ten sequential, then ten concurrent with a semaphore limit of three. It reports measured durations and successful result counts. Failed requests are reported individually; `asyncio.gather(return_exceptions=True)` preserves the other results. Live timings depend on API/network conditions.

Exponential backoff and jitter (4 lines):

1. Exponential backoff increases the delay after each retry, commonly 1, 2, 4, then 8 seconds.
2. It gives a temporarily overloaded service time to recover and reduces repeated pressure.
3. Jitter adds a small random offset to each delay so clients do not retry in lockstep.
4. This example logs deterministic exponential waits; jitter is a useful production addition.

## Task 3: Tools and conversation memory

Run the offline demonstration without API access or credits:

```sh
./.venv314/bin/python task3_tools_memory.py
```

It exercises all three tool schemas, executes a three-call round concurrently, checks four recoverable tool-error results (including their `tool_call_id`s), demonstrates that a stateless model forgets earlier input, and runs a 15-turn summarized conversation that retains an early name and order ID. The offline responder and summary are deterministic fixtures.

Run both the tool loop and 15-turn conversation through the OpenAI API:

```sh
./.venv314/bin/python task3_tools_memory.py --live
```

Or run only one live portion with `--live-tools` or `--live-conversation`. Live requests require a valid OpenAI API key and available API credits. The tool loop is capped at five rounds. Conversation requests use a 500-token system prompt budget, 3,000-token history budget, and 1,000-token output cap; older complete turns are summarized before sending when needed.

## Task 4: Secured FastAPI backend

Set `APP_API_KEY` in `.env` to a private random value. `OPENAI_API_KEY` is used separately by the live LLM dependency. Start the service with:

```sh
./.venv314/bin/python -m uvicorn task4_backend:app --reload
```

Authenticated routes require `x-api-key`; `/health` is public. Create a session with `POST /sessions`, send a `{"message":"hello"}` body to `POST /sessions/{id}/chat`, and fetch history with `GET /sessions/{id}/messages`. Empty messages return 422; messages over 2,000 characters return 413. CORS allows only `http://localhost:3000` and `http://localhost:5173` by default.

Run the mocked test suite with coverage:

```sh
./.venv314/bin/python -m pytest -q tests/test_task4_backend.py --cov=task4_backend --cov-report=term-missing --cov-fail-under=80
```
