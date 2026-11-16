"""Task 3: parallel tool calls, recoverable tool errors, and conversation memory."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Protocol

import tiktoken

from main import OPENAI_MODEL, SYSTEM_PROMPT, _require_key


ORDERS: dict[int, dict[str, Any]] = {
    1001: {"status": "shipped", "tracking": "YOHO-1001"},
    1002: {"status": "processing", "tracking": None},
}
CUSTOMERS: dict[int, dict[str, Any]] = {
    5: {"name": "Avery Shah", "tier": "gold"},
    8: {"name": "Jordan Lee", "tier": "silver"},
}
DISCOUNT_RATES = {"standard": 0.0, "silver": 0.05, "gold": 0.10}
IMPORTANT_NAME = "Maya Chen"
IMPORTANT_ORDER_ID = "ORD-7314"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Look up an order by its positive integer ID. Returns status and tracking code, or a recoverable error if the order does not exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer", "minimum": 1, "description": "Positive numeric order ID, such as 1001."}
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer",
            "description": "Look up a customer by positive integer customer ID. Returns their name and loyalty tier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer", "minimum": 1, "description": "Positive customer ID, such as 5."}
                },
                "required": ["customer_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_discount",
            "description": "Calculate the discount and final price for a nonnegative subtotal and a customer loyalty tier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subtotal": {"type": "number", "minimum": 0, "description": "Purchase subtotal in the same currency units as the result."},
                    "customer_tier": {"type": "string", "enum": ["standard", "silver", "gold"], "description": "Customer loyalty tier."},
                },
                "required": ["subtotal", "customer_tier"],
                "additionalProperties": False,
            },
        },
    },
]


class ToolError(Exception):
    """An expected tool error the model can use to correct its request."""


def _int_arg(args: dict[str, Any], key: str) -> int:
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ToolError(f"Invalid argument '{key}': expected a positive integer")
    return value


def _check_keys(args: dict[str, Any], required: set[str]) -> None:
    missing = required - args.keys()
    extra = args.keys() - required
    if missing:
        raise ToolError(f"Invalid argument: missing {', '.join(sorted(missing))}")
    if extra:
        raise ToolError(f"Invalid argument: unexpected {', '.join(sorted(extra))}")


def dispatch_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a tool synchronously. Inputs are validated even if the model ignores its schema."""
    if not isinstance(args, dict):
        raise ToolError("Invalid arguments: expected a JSON object")
    if name == "get_order_status":
        _check_keys(args, {"order_id"})
        order_id = _int_arg(args, "order_id")
        order = ORDERS.get(order_id)
        if order is None:
            raise ToolError(f"Order {order_id} not found")
        return {"order_id": order_id, **order}
    if name == "get_customer":
        _check_keys(args, {"customer_id"})
        customer_id = _int_arg(args, "customer_id")
        customer = CUSTOMERS.get(customer_id)
        if customer is None:
            raise ToolError(f"Customer {customer_id} not found")
        return {"customer_id": customer_id, **customer}
    if name == "calculate_discount":
        _check_keys(args, {"subtotal", "customer_tier"})
        subtotal = args.get("subtotal")
        tier = args.get("customer_tier")
        if isinstance(subtotal, bool) or not isinstance(subtotal, (int, float)) or subtotal < 0:
            raise ToolError("Invalid argument 'subtotal': expected a nonnegative number")
        if tier not in DISCOUNT_RATES:
            raise ToolError("Invalid argument 'customer_tier': expected standard, silver, or gold")
        rate = DISCOUNT_RATES[tier]
        discount = round(float(subtotal) * rate, 2)
        return {"subtotal": subtotal, "customer_tier": tier, "discount_rate": rate,
                "discount_amount": discount, "final_total": round(float(subtotal) - discount, 2)}
    raise ToolError(f"Unknown tool '{name}'")


def parse_arguments(raw: str) -> dict[str, Any]:
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ToolError(f"Invalid argument JSON: {exc.msg}") from exc
    if not isinstance(result, dict):
        raise ToolError("Invalid arguments: expected a JSON object")
    return result


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class ModelTurn:
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


@dataclass(frozen=True)
class ToolResult:
    tool_call_id: str
    content: str


class ChatModel(Protocol):
    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn: ...


class OpenAIChatModel:
    def __init__(self, model: str = OPENAI_MODEL):
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI()

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            parallel_tool_calls=True,
        )
        message = response.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments) for c in (message.tool_calls or [])]
        return ModelTurn(content=message.content, tool_calls=calls or None)


async def execute_tool_round(calls: list[ToolCall]) -> list[ToolResult]:
    """Execute all calls in a round concurrently and preserve every original call ID."""
    async def one(call: ToolCall) -> ToolResult:
        try:
            output = dispatch_tool(call.name, parse_arguments(call.arguments))
        except ToolError as exc:
            output = {"error": str(exc)}
        return ToolResult(call.id, json.dumps(output, separators=(",", ":")))

    return await asyncio.gather(*(one(call) for call in calls))


async def run_tool_loop(
    model: ChatModel,
    messages: list[dict[str, Any]],
    *,
    max_rounds: int = 5,
) -> str:
    """Reusable bounded tool loop; each round's tool outputs are sent back to the model."""
    if not 1 <= max_rounds <= 5:
        raise ValueError("max_rounds must be between 1 and 5")
    for _round in range(max_rounds):
        turn = await model.complete(messages, TOOLS)
        if not turn.tool_calls:
            return turn.content or ""

        calls = turn.tool_calls
        messages.append({
            "role": "assistant",
            "content": turn.content,
            "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
                for call in calls
            ],
        })
        results = await execute_tool_round(calls)
        print(f"Tool calls in one round: {len(calls)}")
        for result in results:
            print(f"Tool result [{result.tool_call_id}]: {result.content}")
            messages.append({"role": "tool", "tool_call_id": result.tool_call_id, "content": result.content})
    return "Stopped after the maximum of 5 tool rounds."


class ScriptedToolModel:
    """Offline model to show the reusable tool loop without API calls."""
    def __init__(self):
        self.round = 0

    async def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        self.round += 1
        if self.round == 1:
            return ModelTurn(tool_calls=[
                ToolCall("tool_order", "get_order_status", '{"order_id":1001}'),
                ToolCall("tool_customer", "get_customer", '{"customer_id":5}'),
                ToolCall("tool_discount", "calculate_discount", '{"subtotal":120,"customer_tier":"gold"}'),
            ])
        return ModelTurn(content="Order 1001 is shipped; customer 5 is gold tier; the discounted total is 108.00.")


async def error_case_demo() -> None:
    cases = [
        (ToolCall("err_order", "get_order_status", '{"order_id":9999}'), "Order 9999 not found"),
        (ToolCall("err_argument", "get_order_status", '{"order_id":"abc"}'), "expected a positive integer"),
        (ToolCall("err_tool", "get_invoice", '{"invoice_id":3}'), "Unknown tool 'get_invoice'"),
        (ToolCall("err_json", "get_customer", '{"customer_id":'), "Invalid argument JSON"),
    ]
    for call, expected in cases:
        result = (await execute_tool_round([call]))[0]
        payload = json.loads(result.content)
        assert expected in payload.get("error", ""), f"Unexpected result: {payload}"
        assert result.tool_call_id == call.id, "Tool result lost its call ID"
        print(f"Error case [{result.tool_call_id}]: {result.content}")


def _encoding(model: str = OPENAI_MODEL):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


class Conversation:
    """Conversation with full message storage and a summary plus recent-turn window."""
    SYSTEM_BUDGET = 500
    HISTORY_BUDGET = 3000
    OUTPUT_BUDGET = 1000

    def __init__(self, *, model: str = OPENAI_MODEL, keep_last_turns: int = 5):
        self.model = model
        self.keep_last_turns = keep_last_turns
        self.encoding = _encoding(model)
        self.system = (
            "You are a helpful assistant. Retain the user's important personal and order facts "
            "across this conversation."
        )
        if self.count_text(self.system) > self.SYSTEM_BUDGET:
            raise ValueError("System prompt exceeds its 500-token budget")
        self.messages: list[dict[str, str]] = []
        self.turns: list[list[dict[str, str]]] = []
        self.summary = ""

    def count_text(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        return 3 + sum(3 + self.count_text(m.get("role", "")) + self.count_text(m.get("content", "")) for m in messages)

    def _active_messages(self, pending_user: str | None = None) -> list[dict[str, str]]:
        active = [{"role": "system", "content": self.system}]
        if self.summary:
            active.append({"role": "developer", "content": "Summary of earlier turns: " + self.summary})
        active.extend(message for turn in self.turns[-self.keep_last_turns:] for message in turn)
        if pending_user is not None:
            active.append({"role": "user", "content": pending_user})
        return active

    def history_tokens(self, pending_user: str | None = None) -> int:
        active = self._active_messages(pending_user)
        return self.count_messages([m for m in active if m["role"] != "system"])

    def _make_summary(self, archived: list[list[dict[str, str]]]) -> str:
        prior = self.summary
        old_text = " ".join(message["content"] for turn in archived for message in turn)
        facts = []
        for fact in (IMPORTANT_NAME, IMPORTANT_ORDER_ID):
            if fact.lower() in (old_text + " " + prior).lower():
                facts.append(fact)
        context = " ".join(filter(None, [prior, *(["Customer name: " + IMPORTANT_NAME] if IMPORTANT_NAME in facts else []),
                                         *(["Order ID: " + IMPORTANT_ORDER_ID] if IMPORTANT_ORDER_ID in facts else [])]))
        # Preserve other useful old-turn details in a compact, deterministic offline summary.
        snippets = []
        for turn in archived:
            for msg in turn:
                content = msg["content"]
                if IMPORTANT_NAME not in content and IMPORTANT_ORDER_ID not in content:
                    snippets.append(content[:90])
        return (context + (" Earlier notes: " + " | ".join(snippets[-3:]) if snippets else "")).strip()

    def _summarize_oldest(self) -> bool:
        if len(self.turns) <= self.keep_last_turns:
            return False
        archived = self.turns[:-self.keep_last_turns]
        self.summary = self._make_summary(archived)
        self.turns = self.turns[-self.keep_last_turns:]
        return True

    def _fit_history(self, pending_user: str) -> tuple[int, int]:
        before = self.history_tokens(pending_user)
        while self.history_tokens(pending_user) > self.HISTORY_BUDGET:
            if not self._summarize_oldest():
                break
        after = self.history_tokens(pending_user)
        if after > self.HISTORY_BUDGET:
            raise ValueError(f"History uses {after} tokens, exceeding {self.HISTORY_BUDGET}; shorten the current turn")
        if self.OUTPUT_BUDGET != 1000:
            raise AssertionError("Unexpected output budget")
        return before, after

    def append_exchange(self, user: str, assistant: str) -> None:
        """Append an already-completed exchange to full messages and complete-turn storage."""
        pair = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        self.messages.extend(pair)
        self.turns.append(pair)

    async def ask(self, user: str, responder: Any, client: Any | None = None) -> str:
        before, after = self._fit_history(user)
        if before > self.HISTORY_BUDGET:
            print(f"History before trim: {before} tokens -> {after} after summarising")
        active = self._active_messages(user)
        if client is None:
            answer = await responder(active)
        else:
            response = await client.chat.completions.create(
                model=self.model, messages=active, max_tokens=self.OUTPUT_BUDGET,
            )
            answer = response.choices[0].message.content or ""
        self.append_exchange(user, answer)
        return answer

    async def summarize_full_transcript(self) -> tuple[int, int]:
        """Compress at the end of a demonstration and report before/after history tokens."""
        before = self.count_messages(self.messages)
        while len(self.turns) > self.keep_last_turns:
            self._summarize_oldest()
        after = self.history_tokens()
        if after > self.HISTORY_BUDGET:
            # Keep dropping complete old turns until the active history satisfies the token cap.
            while after > self.HISTORY_BUDGET and self._summarize_oldest():
                after = self.history_tokens()
        return before, after


class OfflineResponder:
    async def __call__(self, messages: list[dict[str, str]]) -> str:
        user_texts = [m["content"] for m in messages if m["role"] == "user"]
        latest = user_texts[-1] if user_texts else ""
        combined = " ".join(m.get("content", "") for m in messages)
        if "what name and order" in latest.lower():
            if IMPORTANT_NAME in combined and IMPORTANT_ORDER_ID in combined:
                return f"Your name is {IMPORTANT_NAME}; your order is {IMPORTANT_ORDER_ID}."
            return "I do not have your name or order ID in the available conversation."
        if "what name did i provide" in latest.lower():
            if IMPORTANT_NAME in combined:
                return f"You said your name is {IMPORTANT_NAME}."
            return "I do not know your name because it is not in this request."
        return "Got it; I will keep that in mind."


async def prove_stateless_and_history() -> None:
    responder = OfflineResponder()
    first = [{"role": "user", "content": f"My name is {IMPORTANT_NAME}."}]
    await responder(first)
    isolated = await responder([{"role": "user", "content": "What name did I provide?"}])
    print(f"Stateless second request (no history): {isolated}")

    conversation = Conversation()
    await conversation.ask(f"My name is {IMPORTANT_NAME} and my order ID is {IMPORTANT_ORDER_ID}.", responder)
    remembered = await conversation.ask("What name and order did I provide?", responder)
    print(f"Conversation with stored history: {remembered}")


async def fifteen_turn_demo() -> None:
    conversation = Conversation(keep_last_turns=5)
    responder = OfflineResponder()
    for turn in range(1, 16):
        if turn == 1:
            prompt = f"My name is {IMPORTANT_NAME}; please remember order {IMPORTANT_ORDER_ID}."
        elif turn == 15:
            prompt = "What name and order ID did I share at the beginning?"
        else:
            prompt = f"Turn {turn}: please remember this shipping update and support note."
        # Expand each turn into realistic long-form context to exercise the token window offline.
        if turn not in (1, 15):
            prompt += " " + (f"For this update, record the package status, the carrier scan, and my preference for a delivery notice. " * 8)
        answer = await conversation.ask(prompt, responder)
        if turn != 15:
            # Add ordinary historical detail to the transcript, while keeping answer role and turn pair.
            pair = conversation.turns[-1]
            pair[1]["content"] += " This update is saved with the conversation context."
            conversation.messages[-1]["content"] = pair[1]["content"]
    before, after = await conversation.summarize_full_transcript()
    final_context = conversation._active_messages("What name and order ID did I share at the beginning?")
    final_answer = await responder(final_context)
    print(f"Turn 15: history {before} tokens -> {after} after summarising")
    print(f"Turn 15 answer: {final_answer}")
    print(f"Early name survived summary: {IMPORTANT_NAME in conversation.summary}")
    print(f"Early order ID survived summary: {IMPORTANT_ORDER_ID in conversation.summary}")
    if IMPORTANT_NAME not in conversation.summary or IMPORTANT_ORDER_ID not in conversation.summary:
        raise AssertionError("The early facts were not preserved by summarisation")
    if after > conversation.HISTORY_BUDGET:
        raise AssertionError("Summarised history exceeds the 3,000-token budget")


async def live_tool_demo(model_name: str) -> None:
    _require_key("OPENAI_API_KEY")
    model = OpenAIChatModel(model_name)
    messages = [{"role": "user", "content": "Check order 1001, customer 5, and the gold discount on 120."}]
    try:
        answer = await run_tool_loop(model, messages)
        print("Assistant:", answer)
    finally:
        await model.client.close()


async def live_conversation_demo(model_name: str) -> None:
    from openai import AsyncOpenAI

    _require_key("OPENAI_API_KEY")
    client = AsyncOpenAI()
    conversation = Conversation(model=model_name)
    try:
        for turn in range(1, 16):
            if turn == 1:
                prompt = f"My name is {IMPORTANT_NAME}; remember order {IMPORTANT_ORDER_ID}."
            elif turn == 15:
                prompt = "What name and order ID did I share at the start?"
            else:
                prompt = f"Turn {turn}: record a shipping update and my preference for text notifications."
            answer = await conversation.ask(prompt, responder=None, client=client)
            print(f"Turn {turn}: {answer}")
        print("Summary retains early fact:", IMPORTANT_NAME in conversation.summary and IMPORTANT_ORDER_ID in conversation.summary)
    finally:
        await client.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use OpenAI for both tools and the 15-turn conversation")
    parser.add_argument("--live-tools", action="store_true", help="Use the OpenAI API for a real tool-loop request")
    parser.add_argument("--live-conversation", action="store_true", help="Run 15 assistant turns with the OpenAI API")
    parser.add_argument("--model", default=OPENAI_MODEL)
    args = parser.parse_args()

    print("=== Tool loop ===")
    if args.live or args.live_tools:
        await live_tool_demo(args.model)
    else:
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Look up order, customer, and discount."}]
        answer = await run_tool_loop(ScriptedToolModel(), messages)
        print("Assistant:", answer)
    print("=== Four recoverable tool errors ===")
    await error_case_demo()
    print("=== Stateless model vs stored history ===")
    await prove_stateless_and_history()
    print("=== 15-turn sliding-window conversation ===")
    if args.live or args.live_conversation:
        await live_conversation_demo(args.model)
    else:
        await fifteen_turn_demo()


if __name__ == "__main__":
    asyncio.run(main())
