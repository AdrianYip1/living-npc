from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from game_agents.bootstrap import start_fresh
from game_agents.storage import load_memory, load_player_names


class StartFreshTests(unittest.TestCase):
    """What --fresh does before build_registry() reads any of it back in."""

    def _data_dir(self, tmp: str) -> tuple[Path, tuple[Path, ...]]:
        data = Path(tmp)
        memory, inventory = data / "memory", data / "inventory"
        memory.mkdir()
        inventory.mkdir()
        (memory / "Mara.json").write_text(
            json.dumps([{"content": "the player is called Ray", "importance": 6, "tags": [], "timestamp": 1.0}]),
            encoding="utf-8",
        )
        (inventory / "Mara.json").write_text(json.dumps({"money": 3, "items": {}}), encoding="utf-8")
        names = data / "player_names.json"
        names.write_text(json.dumps({"Mara": "Ray"}), encoding="utf-8")
        return data, (memory, inventory, names)

    def test_the_town_no_longer_remembers_anyone(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, paths = self._data_dir(tmp)
            memory, _, names = paths

            start_fresh(paths, data_dir=data)

            # Every loader treats what's missing as "nothing yet", which is
            # the whole mechanism -- nothing else has to know about --fresh.
            self.assertEqual(load_memory("Mara", memory).all(), [])
            self.assertEqual(load_player_names(names), {})

    def test_nothing_is_deleted_only_moved(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, paths = self._data_dir(tmp)

            archived = start_fresh(paths, data_dir=data)

            self.assertTrue(archived.name.startswith("backup_"))
            self.assertEqual(json.loads((archived / "player_names.json").read_text(encoding="utf-8")), {"Mara": "Ray"})
            self.assertEqual(len(load_memory("Mara", archived / "memory").all()), 1)

    def test_a_second_fresh_start_keeps_the_first_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, paths = self._data_dir(tmp)

            first = start_fresh(paths, data_dir=data)
            self._data_dir(tmp)  # a second run's worth of state
            second = start_fresh(paths, data_dir=data)

            self.assertNotEqual(first, second)
            self.assertTrue(first.exists() and second.exists())

    def test_nothing_to_put_away_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)

            self.assertIsNone(start_fresh((data / "memory", data / "player_names.json"), data_dir=data))
            self.assertEqual(list(data.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
