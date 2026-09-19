from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

# The schema every LLM call must follow: the model is forced to pick exactly
# one tool -- either this built-in "speak" tool (in-character dialogue), or
# one of the NPC's registered actions. Never both, never neither. Agent
# always prepends this to whatever action schemas it hands the LLM client.
SPEAK_TOOL_NAME = "speak"
SPEAK_TOOL_SCHEMA: dict[str, Any] = {
    "name": SPEAK_TOOL_NAME,
    "description": "Say something out loud, in character, for whoever you're talking to to hear.",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Exactly what your character says out loud."}},
        "required": ["text"],
    },
}

# Extra speak-tool field offered only on NPC-to-NPC conversation turns (see
# Agent.respond's `conversation`): the speaker marking a line as a goodbye.
ENDS_CONVERSATION_FIELD = "ends_conversation"


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResult:
    """Always exactly one tool call -- speak or an action, per SPEAK_TOOL_SCHEMA
    above. There's no separate free-text field anymore: a spoken line lives
    in a "speak" tool_call's `text` argument like any other tool's args.
    """

    tool_call: ToolCall


class LLMClient(Protocol):
    def complete(
        self, *, system: str, messages: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> LLMResult: ...


class MockLLMClient:
    """Deterministic, network-free stand-in so agent logic can be exercised
    without an API key. Always "speaks" -- echoes the stimulus back.
    """

    def complete(
        self, *, system: str, messages: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> LLMResult:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return LLMResult(tool_call=ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": f"(mock reply to: {last_user})"}))


class AnthropicLLMClient:
    """Real backend, via the Claude API's native tool-calling. tool_choice is
    forced to "any" so the model can't just answer in prose and skip the
    schema.
    """

    def __init__(self, *, model: str = "claude-sonnet-5", api_key: str | None = None) -> None:
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def complete(
        self, *, system: str, messages: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> LLMResult:
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
                **({"strict": True} if t.get("strict") else {}),
            }
            for t in tools
        ]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=messages,
            tools=anthropic_tools,
            tool_choice={"type": "any"},
        )
        for block in response.content:
            if block.type == "tool_use":
                return LLMResult(tool_call=ToolCall(name=block.name, arguments=block.input))
        raise RuntimeError("Anthropic response had no tool_use block despite tool_choice='any'")


class DeepSeekLLMClient:
    """Real backend for DeepSeek's OpenAI-compatible API. `deepseek-chat` and
    `deepseek-reasoner` are retired -- DeepSeek now ships a V4 model family
    selected directly by id. `deepseek-flash` (V4.1-Flash, their current
    default recommendation) is the default here; pass model="deepseek-v4-pro"
    for the other current tier if agentic/tool-use reliability matters more
    than latency for a given NPC. Check DeepSeek's docs if this drifts again.
    tool_choice is forced to "required" for the same reason as Anthropic's
    tool_choice="any" above.
    """

    def __init__(self, *, model: str = "deepseek-flash", api_key: str | None = None) -> None:
        import os

        import openai

        # The openai SDK defaults to reading OPENAI_API_KEY, which is the
        # wrong env var for a DeepSeek key -- read ours explicitly instead.
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com"
        )
        self._model = model

    def complete(
        self, *, system: str, messages: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> LLMResult:
        openai_tools = [
            {
                "type": "function",
                "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]},
            }
            for t in tools
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "system", "content": system}, *messages],
            tools=openai_tools,
            tool_choice="required",
        )
        message = response.choices[0].message
        if not message.tool_calls:
            raise RuntimeError("DeepSeek response had no tool call despite tool_choice='required'")
        import json

        call = message.tool_calls[0]
        return LLMResult(tool_call=ToolCall(name=call.function.name, arguments=json.loads(call.function.arguments)))
