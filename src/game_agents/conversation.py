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
    - on_start: called with (initiator, target) once the pair is claimed,
      just before the first turn.
    - scene: the current time/weather, as each turn starts.
    - on_turn: called with each turn as soon as it's decided -- and may
      block, which is how the simulation paces lines out for reading.
    - on_end: called with the whole transcript, before both sides are
      released.
    - refuse: asked before an exchange starts, with (initiator, target);
      a reason string turns the attempt down, handed back to the initiator
      as its tool result. None lets it go ahead.
    """

    run: Callable[[Callable[[], None]], bool] | None = None
    on_start: Callable[[str, str], None] | None = None
    scene: Callable[[], Scene] | None = None
    on_turn: Callable[[ConversationTurn], None] | None = None
    on_end: Callable[[list[ConversationTurn]], None] | None = None
    refuse: Callable[[str, str], str | None] | None = None


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
    trade just agreed on): the same side gets to follow it up with a line.
    Tool calls are private -- only the side that made one sees it (and its
    result) in the transcript; the other side only hears what's said.

    Caller is responsible for claiming/releasing busy state around this;
    this function only runs the turns. It never calls itself or lets either
    side re-initiate mid-exchange on its own -- see NPCRegistry's busy
    tracking, which is what actually makes that structurally impossible
    (a side already claimed busy can't successfully claim a new pair).
    """
    transcript: list[ConversationTurn] = []
    # (who can see it, line): None for what's said aloud, a name for that
    # side's own tool calls.
    lines: list[tuple[str | None, str]] = []
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
            # speaker's own view of the transcript, then the same speaker
            # follows up (hands over the goods, says what they found...).
            # Twice in a row still ends it -- no looping on actions.
            outcome = str(result.action["result"]).split(". ")[0].rstrip(".")
            lines.append((speaker.identity.name, f"(You: {result.action['name']} -- {outcome}.)"))
            stimulus = f"({result.action['result']}) Now say your next line to {other.identity.name}."
            continue
        if closing:
            break
        closing = result.ends_conversation
        lines.append((None, f"{speaker.identity.name}: {result.utterance}"))
        stimulus = f'{speaker.identity.name} says: "{result.utterance}"'
        speaker, other = other, speaker
    return transcript


def conversation_recap(transcript: list[ConversationTurn], name: str) -> str:
    """The whole exchange as `name` remembers it afterwards. Stored
    untagged (see NPCRegistry), so it surfaces whoever they talk to next --
    the per-turn memories are tagged with the partner and only come back
    around them, which left an NPC asking a second person what the first
    had just told it. Only `name`'s own tool calls are in it -- the other
    side's were never visible to them.
    """
    partner = next((t.listener if t.speaker == name else t.speaker for t in transcript), "someone")
    lines = []
    for turn in transcript:
        if turn.utterance is not None:
            lines.append(f"{turn.speaker}: {turn.utterance}")
        elif turn.action is not None and turn.speaker == name:
            lines.append(f"(You: {turn.action['name']} -- {turn.action['result']})")
    return f"Earlier you talked with {partner}:\n" + "\n".join(f"  {line}" for line in lines)


def _conversation_context(
    speaker: Agent, other: Agent, lines: list[tuple[str | None, str]], *, opening: bool, closing: bool
) -> str:
    other_name = other.identity.name
    if opening:
        parts = [f"You've chosen to talk to {other_name}, who is right in front of you. Open the conversation."]
    else:
        parts = [f"You're in a conversation with {other_name}."]
    visible = [line for owner, line in lines if owner is None or owner == speaker.identity.name]
    if visible:
        parts.append("Conversation so far:\n" + "\n".join(visible))
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
