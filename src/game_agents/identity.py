from __future__ import annotations

from dataclasses import dataclass, field

# Fallback used only if data/instructions.json is missing its
# "profile_template" key -- the real, editable copy of this lives there.
DEFAULT_PROFILE_TEMPLATE = (
    "You are {name}.\n"
    "Age: {age}\n"
    "Appearance: {appearance}\n"
    "Backstory: {backstory}\n"
    "Traits: {traits}\n"
    "Likes: {likes}\n"
    "Dislikes: {dislikes}\n"
    "Not your field (you know little about these): {unfamiliar_with}\n"
    "People you know: {relationships}\n"
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
    as the move tool is used. starting_money / starting_items follow the
    same split: they only seed the Agent's Inventory (see inventory.py),
    which is what actually changes as the NPC trades.
    """

    name: str
    traits: list[str]
    backstory: str
    speech_style: str
    goals: list[str] = field(default_factory=list)
    home: tuple[int, int] = (0, 0)
    workplace: tuple[int, int] = (0, 0)
    habits: list[str] = field(default_factory=list)
    starting_money: int = 0
    starting_items: dict[str, int] = field(default_factory=dict)
    # "male" or "female" -- picks the voice and face on the speech/animation
    # side (see conversation_export.py).
    gender: str = ""
    # Concrete personal facts, so the model has true things to say about
    # itself instead of inventing them. age 0 = unstated.
    age: int = 0
    appearance: str = ""
    likes: list[str] = field(default_factory=list)
    dislikes: list[str] = field(default_factory=list)
    # What they'd have no real knowledge of -- stops a blacksmith giving
    # confident herbal advice.
    unfamiliar_with: list[str] = field(default_factory=list)
    # Other NPC's name -> how this one knows and sees them. Anyone missing
    # is a stranger.
    relationships: dict[str, str] = field(default_factory=dict)

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
            age=str(self.age) if self.age else "unknown",
            appearance=self.appearance or "unremarkable",
            likes=", ".join(self.likes) if self.likes else "none in particular",
            dislikes=", ".join(self.dislikes) if self.dislikes else "none in particular",
            unfamiliar_with=", ".join(self.unfamiliar_with) if self.unfamiliar_with else "none",
            relationships=(
                "; ".join(f"{name} -- {how}" for name, how in self.relationships.items())
                if self.relationships
                else "no one here yet"
            ),
        )
