from __future__ import annotations

from dataclasses import dataclass, field

# Fallback used only if data/instructions.json is missing its
# "profile_template" key -- the real, editable copy of this lives there.
DEFAULT_PROFILE_TEMPLATE = (
    "You are {name}.\n"
    "Backstory: {backstory}\n"
    "Traits: {traits}\n"
    "Speech style: {speech_style}\n"
    "Current goals: {goals}\n"
    "Home: {home}\n"
    "Workplace: {workplace}\n"
    "Habits: {habits}"
)


@dataclass
class Identity:
    """An NPC's fixed personality: who they are, not what they're doing right now.
    `name` doubles as the identifier -- small demo, no need for a separate
    id as long as names stay unique.

    `home` and `workplace` are fixed map coordinates (see world.py for the
    map's bounds), not runtime state -- where an NPC currently stands is
    dynamic and lives on the Agent instead, starting at `home` and changing
    as the move tool is used.
    """

    name: str
    traits: list[str]
    backstory: str
    speech_style: str
    goals: list[str] = field(default_factory=list)
    home: tuple[int, int] = (0, 0)
    workplace: tuple[int, int] = (0, 0)
    habits: list[str] = field(default_factory=list)

    def prompt_block(self, template: str = DEFAULT_PROFILE_TEMPLATE) -> str:
        return template.format(
            name=self.name,
            backstory=self.backstory,
            traits=", ".join(self.traits),
            speech_style=self.speech_style,
            goals=", ".join(self.goals) if self.goals else "none",
            home=f"({self.home[0]}, {self.home[1]})",
            workplace=f"({self.workplace[0]}, {self.workplace[1]})",
            habits="; ".join(self.habits) if self.habits else "none",
        )
