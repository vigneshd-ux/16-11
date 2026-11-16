"""Task 2: bounded retries, async calls, and a mock-only retry proof."""

import argparse
import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from openai import AsyncOpenAI
from tenacity import (
    Retrying,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from unittest.mock import Mock

from main import OPENAI_MODEL, _log_call, _require_key

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("task2")


def _status_code(error: BaseException) -> int | None:
    """Read an HTTP status from an SDK error or a small test double."""
    return getattr(error, "status_code", None)


def _retryable(error: BaseException) -> bool:
    status = _status_code(error)
    return status == 429 or (status is not None and 500 <= status <= 599)


def _log_retry(retry_state: Any) -> None:
    error = retry_state.outcome.exception()
    status = _status_code(error) or "unknown"
    wait_seconds = retry_state.next_action.sleep
    logger.warning(
        "Attempt %s failed (%s) -> wait %g s",
        retry_state.attempt_number,
        status,
        wait_seconds,
    )


def _retrying(*, sleep: Callable[[float], None] = time.sleep) -> Retrying:
    """Build the shared retry policy; tests can replace sleep with a no-op."""
    return Retrying(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(5),
        before_sleep=_log_retry,
        sleep=sleep,
        reraise=True,
    )


def call_with_retry(operation: Callable[[], Any]) -> Any:
    """Call a synchronous API operation with 429/5xx-only retries."""
    return _retrying()(operation)


class MockHTTPError(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"mock HTTP {status_code}")


def prove_retry_behavior() -> None:
    """Prove two 429 retries succeed and a 401 is attempted only once."""
    success_after_two = Mock(
        side_effect=[MockHTTPError(429), MockHTTPError(429), "mock success"]
    )
    result = _retrying(sleep=lambda _seconds: None)(success_after_two)
    assert result == "mock success"
    assert success_after_two.call_count == 3
    logger.info("Attempt 3 succeeded: %s (mock called %s times)", result,
                success_after_two.call_count)

    unauthorized = Mock(side_effect=MockHTTPError(401))
    try:
        _retrying(sleep=lambda _seconds: None)(unauthorized)
    except MockHTTPError as error:
        assert error.status_code == 401
    else:
        raise AssertionError("Expected mock 401 to be raised")
    assert unauthorized.call_count == 1
    logger.info("401 was not retried (mock called %s time)", unauthorized.call_count)


QUESTIONS = [
    "Why is the sky blue?",
    "What is one practical benefit of learning Python?",
    "Explain photosynthesis in one sentence.",
    "What does a database index do?",
    "Name one way to make a password safer.",
    "What is the difference between a list and a tuple in Python?",
    "Why do plants need sunlight?",
    "What is an API in simple terms?",
    "Give one example of renewable energy.",
    "What does HTTP 404 mean?",
]


@retry(
    retry=retry_if_exception(_retryable),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
    reraise=True,
)
async def async_ask(client: AsyncOpenAI, prompt: str) -> tuple[str, dict[str, int]]:
    """Make one async request; tenacity retries only HTTP 429 and 5xx."""
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=160,
    )
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    _log_call("openai", response.model, input_tokens, output_tokens)
    return response.choices[0].message.content or "", {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


async def _run_sequential(client: AsyncOpenAI) -> tuple[float, list[str]]:
    started = time.perf_counter()
    answers = []
    for prompt in QUESTIONS:
        try:
            answer, _usage = await async_ask(client, prompt)
            answers.append(answer)
        except Exception as error:
            # Continue so a single exhausted retry sequence doesn't lose later results.
            logger.error("Sequential call failed for %r: %s", prompt, error)
            answers.append(f"ERROR: {error}")
    return time.perf_counter() - started, answers


async def _run_parallel(client: AsyncOpenAI) -> tuple[float, list[str]]:
    semaphore = asyncio.Semaphore(3)

    async def limited_call(prompt: str) -> str:
        async with semaphore:
            answer, _usage = await async_ask(client, prompt)
            return answer

    started = time.perf_counter()
    # return_exceptions retains successful answers if individual calls fail.
    results = await asyncio.gather(
        *(limited_call(prompt) for prompt in QUESTIONS), return_exceptions=True
    )
    answers = []
    for prompt, result in zip(QUESTIONS, results):
        if isinstance(result, BaseException):
            logger.error("Parallel call failed for %r: %s", prompt, result)
            answers.append(f"ERROR: {result}")
        else:
            answers.append(result)
    return time.perf_counter() - started, answers


async def compare_async() -> None:
    _require_key("OPENAI_API_KEY")
    # Disable the SDK's own retries so the tenacity policy owns retry counts/logs.
    async with AsyncOpenAI(max_retries=0) as client:
        sequential_seconds, sequential_answers = await _run_sequential(client)
        parallel_seconds, parallel_answers = await _run_parallel(client)

    print(f"Sequential ({len(sequential_answers)} questions): {sequential_seconds:.2f} s")
    print(f"Parallel (semaphore limit 3, {len(parallel_answers)} questions): {parallel_seconds:.2f} s")
    print("Sequential successes:", sum(not item.startswith("ERROR:") for item in sequential_answers))
    print("Parallel successes:", sum(not item.startswith("ERROR:") for item in parallel_answers))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mock-only", action="store_true",
        help="prove retry behavior without making API calls or using credits",
    )
    args = parser.parse_args()
    prove_retry_behavior()
    if not args.mock_only:
        await compare_async()


if __name__ == "__main__":
    asyncio.run(main())
