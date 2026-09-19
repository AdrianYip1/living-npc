from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .agent import Agent


@dataclass
class ConversationTurn:
    """One turn of a run_conversation() exchange -- everything needed to
    reconstruct what happened, in a directly JSON-serializable shape (see
    save_transcript()).
    """

    speaker: str
    stimulus: str
    utterance: str | None
    action: dict[str, Any] | None


def run_conversation(initiator: Agent, target: Agent, *, turns: int = 4) -> list[ConversationTurn]:
    """Alternates turns between two agents: each one's reply becomes the
    other's next stimulus, and each side's memory gets tagged with the
    other's name (so later retrieval can surface "what I remember talking
    to X" specifically). Ends early if either side chooses to act instead
    of speak -- there's no coherent line left to hand the other side.

    Caller is responsible for claiming/releasing busy state around this;
    this function only runs the turns. It never calls itself or lets either
    side re-initiate mid-exchange on its own -- see NPCRegistry's busy
    tracking, which is what actually makes that structurally impossible
    (a side already claimed busy can't successfully claim a new pair).
    """
    transcript: list[ConversationTurn] = []
    stimulus = f"{initiator.identity.name} approaches you and starts a conversation."
    speaker, other = target, initiator
    for _ in range(turns):
        result = speaker.respond(stimulus, tags={other.identity.name})
        transcript.append(
            ConversationTurn(
                speaker=speaker.identity.name, stimulus=stimulus, utterance=result.utterance, action=result.action
            )
        )
        if result.utterance is None:
            break
        stimulus = result.utterance
        speaker, other = other, speaker
    return transcript


def save_transcript(transcript: list[ConversationTurn], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([asdict(turn) for turn in transcript], indent=2), encoding="utf-8")
