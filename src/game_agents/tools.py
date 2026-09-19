from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the tool's arguments
    handler: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        """Provider-neutral shape -- each LLMClient adapts this to whatever
        wire format its API actually wants (Anthropic's `input_schema` vs.
        OpenAI-style `function.parameters`, etc.).
        """
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


class ToolRegistry:
    """The actions one agent can take. Kept separate from Agent so different
    NPCs can be handed different action sets.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def all_tools(self) -> list[Tool]:
        """For copying a shared set of tools into a fresh, per-NPC registry
        -- see NPCRegistry, which needs each NPC to have its own registry
        instance (so it can additionally bind a self-aware initiate_conversation
        tool) while still offering whatever common actions were passed in.
        """
        return list(self._tools.values())

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"no tool registered as {name!r}")
        return tool.handler(**arguments)


def make_wait_tool() -> Tool:
    """A deliberate no-op. Without this, an autonomous tick (see
    mini_map.simulation.Simulation) forces every idle NPC to either move or
    speak to no one every time it's stimulated -- staying put is just as
    valid a choice as acting, so it needs to be an option on the same
    footing as any other tool, not something implied by ignoring a turn.
    """

    def handler() -> str:
        return "You stay where you are and keep doing what you were doing."

    return Tool(
        name="wait",
        description="Do nothing for the moment -- stay put and keep doing whatever you're already doing.",
        parameters={"type": "object", "properties": {}},
        handler=handler,
    )
