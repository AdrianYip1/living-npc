"""Local speech-to-text for the player's push-to-talk input.

The page records the mic while Tab is held and POSTs the clip (whatever
container MediaRecorder produced -- webm/opus in Chromium) to
/api/transcribe. faster-whisper decodes it through PyAV and transcribes it
on this machine, so nothing leaves it and no API key is needed.

The model is loaded on first use, not at startup: the first push-to-talk
pays the download/load cost (a few seconds, once), and sessions that never
speak never pay it at all. MINI_MAP_WHISPER_MODEL picks a different model
size (e.g. "tiny.en" for speed, "small.en" for accuracy).
"""

from __future__ import annotations

import io
import logging
import os
import threading

log = logging.getLogger(__name__)

DEFAULT_MODEL = "base.en"

_model = None
_model_lock = threading.Lock()
# One transcription at a time: whisper already uses every core, and the
# server is threaded, so overlapping requests would just thrash.
_transcribe_lock = threading.Lock()


class SpeechUnavailable(RuntimeError):
    pass


def _get_model():
    global _model
    with _model_lock:
        if _model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as err:
                raise SpeechUnavailable("faster-whisper is not installed") from err
            name = os.environ.get("MINI_MAP_WHISPER_MODEL", DEFAULT_MODEL)
            log.info("loading whisper model %s", name)
            _model = WhisperModel(name, device="cpu", compute_type="int8")
        return _model


def transcribe(audio: bytes) -> str:
    """Return the text spoken in `audio` (any container PyAV can decode)."""
    if not audio:
        return ""
    model = _get_model()
    with _transcribe_lock:
        segments, _info = model.transcribe(
            io.BytesIO(audio),
            beam_size=1,
            # Trims the silence around a quick press/release, and keeps
            # whisper from hallucinating words into an empty clip.
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
