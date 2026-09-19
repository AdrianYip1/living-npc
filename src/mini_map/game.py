"""Entry point for the mini-map UI. Launching this serves a local page at
http://127.0.0.1:8765 -- a static placeholder for now, wired up to
game_agents once the 2D map and conversation panel exist to drive.

Run with `python game.py` from this folder, or `python -m mini_map.game`
from src/.
"""
from __future__ import annotations

import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"
HOST = "127.0.0.1"
PORT = 8765


class StaticHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:
        pass


def run(host: str = HOST, port: int = PORT, *, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer((host, port), StaticHandler)
    url = f"http://{host}:{port}"
    print(f"mini-map UI running at {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    run()
