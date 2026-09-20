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

# Extra speak-tool field offered only while talking with the player (see
# Agent.respond's `with_player`), and only until the NPC knows the name:
# the player's name, caught on the same line that heard it. The
# note_player_name tool does the same thing, but as an action it costs the
# whole turn -- so a model with something to say almost never spent its one
# call on it, and NPCs kept greeting someone they'd met repeatedly as a
# stranger.
PLAYER_NAME_FIELD = "player_name"


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


class _OpenAICompatibleLLMClient:
    """Shared Chat Completions logic for OpenAI and OpenAI-compatible APIs
    (DeepSeek). tool_choice is forced to "required" for the same reason as
    Anthropic's tool_choice="any" above. Subclasses set the key env var,
    base_url, and how the output-token cap is passed.
    """

    _KEY_ENV: str
    _BASE_URL: str | None = None
    _PROVIDER: str

    def __init__(self, *, model: str, api_key: str | None = None) -> None:
        import os

        import openai

        # Read our own env var explicitly -- the openai SDK would otherwise
        # fall back to OPENAI_API_KEY, which is wrong for DeepSeek.
        self._client = openai.OpenAI(api_key=api_key or os.environ.get(self._KEY_ENV), base_url=self._BASE_URL)
        self._model = model

    def _request_options(self) -> dict[str, Any]:
        return {"max_tokens": 512}

    def complete(
        self, *, system: str, messages: list[dict[str, str]], tools: list[dict[str, Any]]
    ) -> LLMResult:
        import json

        openai_tools = [
            {
                "type": "function",
                "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]},
            }
            for t in tools
        ]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, *messages],
            tools=openai_tools,
            tool_choice="required",
            **self._request_options(),
        )
        message = response.choices[0].message
        if not message.tool_calls:
            raise RuntimeError(f"{self._PROVIDER} response had no tool call despite tool_choice='required'")
        call = message.tool_calls[0]
        return LLMResult(tool_call=ToolCall(name=call.function.name, arguments=json.loads(call.function.arguments)))


class DeepSeekLLMClient(_OpenAICompatibleLLMClient):
    """DeepSeek's OpenAI-compatible API. `deepseek-chat` and
    `deepseek-reasoner` are retired -- DeepSeek now ships a V4 model family
    selected directly by id. `deepseek-flash` (V4.1-Flash, their current
    default recommendation) is the default here; pass model="deepseek-v4-pro"
    for the other current tier if agentic/tool-use reliability matters more
    than latency for a given NPC. Check DeepSeek's docs if this drifts again.
    """

    _KEY_ENV = "DEEPSEEK_API_KEY"
    _BASE_URL = "https://api.deepseek.com"
    _PROVIDER = "DeepSeek"

    def __init__(self, *, model: str = "deepseek-flash", api_key: str | None = None) -> None:
        super().__init__(model=model, api_key=api_key)


class OpenAILLMClient(_OpenAICompatibleLLMClient):
    """OpenAI's API. gpt-5.6-terra is the mid tier, closest to Sonnet 5 in
    price and positioning; gpt-5.6-luna is the ~10x cheaper budget tier.
    """

    _KEY_ENV = "OPENAI_API_KEY"
    _PROVIDER = "OpenAI"

    def __init__(self, *, model: str = "gpt-5.6-terra", api_key: str | None = None) -> None:
        super().__init__(model=model, api_key=api_key)

    def _request_options(self) -> dict[str, Any]:
        # Current OpenAI models reject max_tokens. Chat Completions also
        # refuses function tools alongside reasoning on gpt-5.6, so turn
        # reasoning off -- on par with the Anthropic backend, which doesn't
        # use extended thinking either.
        return {"max_completion_tokens": 512, "reasoning_effort": "none"}
