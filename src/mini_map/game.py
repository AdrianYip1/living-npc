"""Entry point for the mini-map UI. Launching this builds the NPC registry
and the environment agent, starts the world tick loop on a background
thread, and serves the static page plus a small JSON API the page polls
for live state at http://127.0.0.1:8765.

Run with `python -m mini_map.game`, from src/ or (after a one-time `pip
install -e .` from the repo root -- see pyproject.toml) from anywhere. It
needs to be run as a package, not `python game.py` directly, since it
imports sibling packages (game_agents, environment_agent) that only
resolve once src/ is on sys.path, whether via cwd or the editable install.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from environment_agent.agent import EnvironmentAgent

from game_agents.bootstrap import build_registry

from .simulation import Simulation

STATIC_DIR = Path(__file__).parent / "static"
HOST = "127.0.0.1"
PORT = 8765
TICK_INTERVAL_S = 15.0

# Set by run() before the server starts; the handler reads it per-request.
# A single-process script gets to have one simulation as shared state
# rather than threading it through the stdlib handler's constructor.
_simulation: Simulation | None = None


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/state":
            self._send_json(200, _simulation.state())
            return
        super().do_GET()

    def do_POST(self) -> None:
        routes = {
            "/api/conversation/start": self._handle_conversation_start,
            "/api/conversation/end": self._handle_conversation_end,
            "/api/conversation/say": self._handle_conversation_say,
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


def run(host: str = HOST, port: int = PORT, *, open_browser: bool = True) -> None:
    global _simulation

    registry, backend = build_registry()
    environment = EnvironmentAgent()
    _simulation = Simulation(registry, environment, tick_interval_s=TICK_INTERVAL_S)

    sim_thread = threading.Thread(target=_simulation.run_forever, daemon=True)
    sim_thread.start()

    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"mini-map UI running at {url} ({backend} backend, tick every {TICK_INTERVAL_S:.0f}s). Ctrl+C to stop")
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


if __name__ == "__main__":
    run()
