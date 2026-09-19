from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .agent import Agent, Scene


# Actions that don't end a conversation when taken mid-exchange: closing a
# deal just agreed on out loud, or checking your pack to answer a question.
# Anything else (walking off, waiting) still means the conversation's over.
IN_CONVERSATION_ACTIONS = frozenset({"buy_item", "sell_item", "check_inventory"})


@dataclass
class ConversationTurn:
    """One turn of a run_conversation() exchange -- everything needed to
    reconstruct what happened, in a directly JSON-serializable shape (see
    save_transcript()).
    """

    speaker: str
    listener: str
    stimulus: str
    utterance: str | None
    action: dict[str, Any] | None
    ends_conversation: bool = False


@dataclass
class ConversationHooks:
    """How NPCRegistry runs an NPC-to-NPC exchange. Every hook is optional;
    with none set, an exchange runs inline, unpaced, with no one watching --
    what tests and the CLI want. mini_map.Simulation fills them in so the
    world keeps moving during a conversation and the page can watch it.

    - run: schedules an exchange (a no-arg callable) somewhere else, e.g. a
      worker thread, instead of running it inside the initiator's tool call.
      Returns False if it couldn't be scheduled.
    - scene: the current time/weather, as each turn starts.
    - on_turn: called with each turn as soon as it's decided -- and may
      block, which is how the simulation paces lines out for reading.
    - on_end: called with the whole transcript, before both sides are
      released.
    """

    run: Callable[[Callable[[], None]], bool] | None = None
    scene: Callable[[], Scene] | None = None
    on_turn: Callable[[ConversationTurn], None] | None = None
    on_end: Callable[[list[ConversationTurn]], None] | None = None


def run_conversation(
    initiator: Agent,
    target: Agent,
    *,
    turns: int = 4,
    scene: Callable[[], Scene] | None = None,
    on_turn: Callable[[ConversationTurn], None] | None = None,
) -> list[ConversationTurn]:
    """Alternates turns between two agents, the initiator opening. Each
    reply becomes the other's next stimulus, each speaker sees the whole
    exchange so far in their scene, and each side's memory gets tagged with
    the other's name (so later retrieval can surface "what I remember
    talking to X" specifically).

    Ends when a side says goodbye (ends_conversation) -- the other side then
    gets one last line to answer it -- when a side acts instead of speaking
    (there's no coherent line left to hand the other side), or after
    `turns` turns, whichever comes first. The exception is an action that
    belongs in a conversation (IN_CONVERSATION_ACTIONS, e.g. closing a
    trade just agreed on): that's noted in the transcript and the same side
    gets to follow it up with a line.

    Caller is responsible for claiming/releasing busy state around this;
    this function only runs the turns. It never calls itself or lets either
    side re-initiate mid-exchange on its own -- see NPCRegistry's busy
    tracking, which is what actually makes that structurally impossible
    (a side already claimed busy can't successfully claim a new pair).
    """
    transcript: list[ConversationTurn] = []
    lines: list[str] = []
    stimulus = f"You've walked up to {target.identity.name}. Say your opening line."
    speaker, other = initiator, target
    closing = False
    for _ in range(turns):
        base = scene() if scene is not None else Scene()
        context = _conversation_context(speaker, other, lines, opening=not transcript, closing=closing)
        result = speaker.respond(
            stimulus,
            scene=Scene(time=base.time, location=base.location, context="\n".join(filter(None, (base.context, context)))),
            tags={other.identity.name},
            conversation=True,
        )
        turn = ConversationTurn(
            speaker=speaker.identity.name,
            listener=other.identity.name,
            stimulus=stimulus,
            utterance=result.utterance,
            action=result.action,
            ends_conversation=result.ends_conversation,
        )
        transcript.append(turn)
        if on_turn is not None:
            on_turn(turn)
        if result.utterance is None:
            acted_before = len(transcript) > 1 and transcript[-2].speaker == turn.speaker
            if closing or result.action is None or result.action["name"] not in IN_CONVERSATION_ACTIONS or acted_before:
                break
            # Part of the conversation, not a way out of it: note it in the
            # transcript both sides see, then the same speaker follows up
            # (hands over the goods, says what they found...). Twice in a
            # row still ends it -- no looping on actions.
            outcome = str(result.action["result"]).split(". ")[0].rstrip(".")
            lines.append(f"({speaker.identity.name}: {result.action['name']} -- {outcome}.)")
            stimulus = f"({result.action['result']}) Now say your next line to {other.identity.name}."
            continue
        if closing:
            break
        closing = result.ends_conversation
        lines.append(f"{speaker.identity.name}: {result.utterance}")
        stimulus = f'{speaker.identity.name} says: "{result.utterance}"'
        speaker, other = other, speaker
    return transcript


def _conversation_context(speaker: Agent, other: Agent, lines: list[str], *, opening: bool, closing: bool) -> str:
    other_name = other.identity.name
    if opening:
        parts = [f"You've chosen to talk to {other_name}, who is right in front of you. Open the conversation."]
    else:
        parts = [f"You're in a conversation with {other_name}."]
    if lines:
        parts.append("Conversation so far:\n" + "\n".join(lines))
    if closing:
        parts.append(f"{other_name} is wrapping up the conversation. Answer their goodbye in one short line.")
    elif not opening:
        parts.append(
            "Reply to what they just said. Once the conversation has run its course, say goodbye "
            "and mark that line as ending the conversation."
        )
    return "\n".join(parts)


def save_transcript(transcript: list[ConversationTurn], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(turn) for turn in transcript], indent=2), encoding="utf-8")
