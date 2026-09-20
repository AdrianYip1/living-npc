"""Entry point for the mini-map UI. Launching this builds the NPC registry
and the environment agent, starts the world tick loop on a background
thread, and serves the static page plus a small JSON API the page polls
for live state at http://127.0.0.1:8765.

Run with `python -m mini_map.game`, from src/ or (after a one-time `pip
install -e .` from the repo root -- see pyproject.toml) from anywhere. It
needs to be run as a package, not `python game.py` directly, since it
imports sibling packages (game_agents, environment_agent) that only
resolve once src/ is on sys.path, whether via cwd or the editable install.

Two tunables, independently overridable (see Simulation for what each one
actually paces):

    python -m mini_map.game --ticks-per-real-minute 30 --llm-calls-per-game-hour 6
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from environment_agent.agent import EnvironmentAgent

from game_agents.bootstrap import CONVERSATION_LOG_DIR, TRAVELER_NAMES_PATH, build_registry
from game_agents.conversation_export import ConversationExporter

from . import speech
from .simulation import Simulation

STATIC_DIR = Path(__file__).parent / "static"
HOST = "127.0.0.1"
PORT = 8765
# 60 ticks/real-min x 1 game-minute/tick (EnvironmentAgent's own default) =
# 1 game-minute per real second, i.e. a full in-game day in 24 real minutes
# -- matches the pace from before the clock became smooth (which was 15
# game-minutes every 15s: the same 1-minute-per-second rate, just chunkier).
TICKS_PER_REAL_MINUTE = 240.0
# 4 calls/game-hr x 1 game-hr/real-min (at the default tick rate above) = 4
# calls/real-min, i.e. one query round every 15s -- matches the old default.
LLM_CALLS_PER_GAME_HOUR = 3.0
# Inclusive (min, max) traveler arrivals per in-game day. Deliberately high
# for now so travelers are easy to spot: at the default clock speed (a day
# every 24 real minutes) 10-20 is one every ~96 real seconds on average.
# EnvironmentAgent's own default is a more realistic (2, 6).
TRAVELERS_PER_DAY = (10, 20)

# Set by run() before the server starts; the handler reads it per-request.
# A single-process script gets to have one simulation as shared state
# rather than threading it through the stdlib handler's constructor.
_simulation: Simulation | None = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if url.path == "/api/state":
            self._note_player_position(parse_qs(url.query))
            self._send_json(200, _simulation.state())
            return
        super().do_GET()

    def _note_player_position(self, query: dict[str, list[str]]) -> None:
        try:
            _simulation.set_player_position(float(query["px"][0]), float(query["py"][0]))
        except (KeyError, ValueError):
            pass

    def do_POST(self) -> None:
        # Raw audio, not JSON, so it skips the JSON routes below.
        if self.path == "/api/transcribe":
            self._handle_transcribe()
            return
        routes = {
            "/api/conversation/start": self._handle_conversation_start,
            "/api/conversation/end": self._handle_conversation_end,
            "/api/conversation/say": self._handle_conversation_say,
            "/api/pause": self._handle_pause,
            "/api/resume": self._handle_resume,
        }
        handler = routes.get(self.path)
        if handler is None:
            self.send_error(404)
            return
        handler(self._read_json_body())

    def _handle_conversation_start(self, body: dict) -> None:
        ok = _simulation.start_conversation(body.get("name", ""))
        self._send_json(200 if ok else 409, {"ok": ok})

    def _handle_conversation_end(self, body: dict) -> None:
        _simulation.end_conversation(body.get("name", ""))
        self._send_json(200, {"ok": True})

    def _handle_conversation_say(self, body: dict) -> None:
        result = _simulation.say(body.get("name", ""), body.get("text", ""))
        if result is None:
            self._send_json(404, {"error": "unknown npc"})
            return
        self._send_json(200, result)

    def _handle_pause(self, body: dict) -> None:
        _simulation.pause()
        self._send_json(200, {"paused": True})

    def _handle_resume(self, body: dict) -> None:
        _simulation.resume()
        self._send_json(200, {"paused": False})

    def _handle_transcribe(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        audio = self.rfile.read(length) if length else b""
        try:
            text = speech.transcribe(audio)
        except speech.SpeechUnavailable as err:
            self._send_json(503, {"error": str(err)})
            return
        except Exception:
            logging.exception("transcription failed")
            self._send_json(500, {"error": "transcription failed"})
            return
        self._send_json(200, {"text": text})

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        pass


def _is_port_still_occupied(host: str, port: int, *, timeout: float = 0.5) -> bool:
    """True if something accepts a connection at host:port. Only meaningful
    right after this process's own listening socket has been closed --
    Windows' SO_REUSEADDR (which ThreadingHTTPServer sets by default) lets
    a second process bind the same port without erroring, so a successful
    bind proves nothing; a successful *connect* does, since only an actual
    listener answers one.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def run(
    host: str = HOST,
    port: int = PORT,
    *,
    open_browser: bool = True,
    ticks_per_real_minute: float = TICKS_PER_REAL_MINUTE,
    llm_calls_per_game_hour: float = LLM_CALLS_PER_GAME_HOUR,
) -> None:
    global _simulation

    _print_actions()
    # Each run starts with an empty conversation_log/, so what's in there
    # is only ever this session's conversations.
    exporter = ConversationExporter(CONVERSATION_LOG_DIR)
    exporter.clear()
    registry, backend = build_registry()
    environment = EnvironmentAgent(travelers_per_day=TRAVELERS_PER_DAY)
    _simulation = Simulation(
        registry,
        environment,
        exporter=exporter,
        backend=backend,
        used_names_path=TRAVELER_NAMES_PATH,
        ticks_per_real_minute=ticks_per_real_minute,
        llm_calls_per_game_hour=llm_calls_per_game_hour,
    )
    # The UI's status panel starts on "Paused" -- match that on the server
    # so the world genuinely doesn't move until the player presses play,
    # rather than ticking silently behind an already-stale-looking label.
    _simulation.pause()

    sim_thread = threading.Thread(target=_simulation.run_forever, daemon=True)
    sim_thread.start()

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    game_minutes_per_real_minute = ticks_per_real_minute * environment.minutes_per_tick
    print(
        f"mini-map UI running at {url} ({backend} backend, {game_minutes_per_real_minute:g} game-min/real-min, "
        f"{llm_calls_per_game_hour:g} LLM calls/game-hr). Ctrl+C to stop"
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _simulation.stop()
        server.server_close()
        registry.save_all()
        # This process just released host:port -- if something still
        # answers there, it's a leftover instance from an earlier run
        # (e.g. one that outlived a closed terminal) rather than us.
        if _is_port_still_occupied(host, port):
            print(
                f"\nWarning: {host}:{port} is still answering after shutdown -- "
                "a previous ghost instance of this program is likely still running.\n"
                f"Find it with: netstat -ano | findstr {port}\n"
                "then stop it with: taskkill /F /PID <pid>"
            )


def _print_actions() -> None:
    """Every action an NPC takes (see game_agents.agent.action_log), one
    line each on the console. Only that logger -- the root logger stays
    quiet, or the HTTP client's per-request lines would bury these.
    """
    actions = logging.getLogger("game_agents.actions")
    if actions.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("  [action] %(message)s"))
    actions.addHandler(handler)
    actions.setLevel(logging.INFO)
    actions.propagate = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the living-npc mini-map.")
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"Port to serve the UI on. Default: {PORT}",
    )
    parser.add_argument(
        "--ticks-per-real-minute",
        type=float,
        default=TICKS_PER_REAL_MINUTE,
        metavar="COUNT",
        help=(
            "How many in-game ticks (day-phase/weather advances) happen per real minute -- "
            f"controls how fast the clock runs. Default: {TICKS_PER_REAL_MINUTE:g}"
        ),
    )
    parser.add_argument(
        "--llm-calls-per-game-hour",
        type=float,
        default=LLM_CALLS_PER_GAME_HOUR,
        metavar="COUNT",
        help=(
            "How many times each idle NPC gets queried per in-game hour, regardless of how "
            f"fast the clock itself is ticking. Default: {LLM_CALLS_PER_GAME_HOUR:g}"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(port=args.port, ticks_per_real_minute=args.ticks_per_real_minute, llm_calls_per_game_hour=args.llm_calls_per_game_hour)
