from __future__ import annotations

from dataclasses import dataclass, field

# Fallback used only if data/instructions.json is missing its
# "profile_template" key -- the real, editable copy of this lives there.
DEFAULT_PROFILE_TEMPLATE = (
    "You are {name}.\n"
    "Backstory: {backstory}\n"
    "Traits: {traits}\n"
    "Speech style: {speech_style}\n"
    "Current goals: {goals}"
)


@dataclass
class Identity:
    """An NPC's fixed personality: who they are, not what they're doing right now.
    `name` doubles as the identifier -- small demo, no need for a separate
    id as long as names stay unique.
    """

    name: str
    traits: list[str]
    backstory: str
    speech_style: str
    goals: list[str] = field(default_factory=list)

    def prompt_block(self, template: str = DEFAULT_PROFILE_TEMPLATE) -> str:
        return template.format(
            name=self.name,
            backstory=self.backstory,
            traits=", ".join(self.traits),
            speech_style=self.speech_style,
            goals=", ".join(self.goals) if self.goals else "none",
        )
