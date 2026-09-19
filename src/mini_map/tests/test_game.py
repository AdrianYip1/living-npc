from __future__ import annotations

import socket
import unittest

from mini_map.game import _is_port_still_occupied


class IsPortStillOccupiedTests(unittest.TestCase):
    """The Ctrl+C shutdown check: after this program's own server closes,
    is host:port still answering (a leftover ghost instance)? Tested here
    against a real socket rather than the whole game server.
    """

    def test_true_while_something_is_listening(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        try:
            self.assertTrue(_is_port_still_occupied("127.0.0.1", port))
        finally:
            server.close()

    def test_false_once_nothing_is_listening(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        server.close()  # released before we ever check

        self.assertFalse(_is_port_still_occupied("127.0.0.1", port, timeout=0.2))


if __name__ == "__main__":
    unittest.main()
