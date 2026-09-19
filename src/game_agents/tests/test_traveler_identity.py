from __future__ import annotations

import unittest

from game_agents.identity import Identity
from game_agents.llm import SPEAK_TOOL_NAME, LLMResult, MockLLMClient, ToolCall
from game_agents.storage import identity_from_record, identity_to_record
from game_agents.traveler_identity import CREATE_IDENTITY_TOOL_NAME, generate_traveler_identity

_FALLBACK = Identity(name="Odell", traits=["weary"], backstory="fallback", speech_style="plain")

_GOOD_ARGUMENTS = {
    "name": "Selwyn",
    "traits": ["wry", "patient"],
    "backstory": "A cartographer mapping the coast road.",
    "speech_style": "precise, dry",
    "goals": ["finish the map"],
    "habits": ["Sketches the town square", "Asks everyone for directions"],
    "starting_money": 20,
    "starting_items": {"map": 1, "ink": 2},
}


class _FixedLLM:
    def __init__(self, tool_name, arguments=None, error=None):
        self._tool_name = tool_name
        self._arguments = arguments or {}
        self._error = error
        self.last_messages = None

    def complete(self, *, system, messages, tools):
        self.last_messages = messages
        if self._error is not None:
            raise self._error
        return LLMResult(tool_call=ToolCall(name=self._tool_name, arguments=self._arguments))


class GenerateTravelerIdentityTests(unittest.TestCase):
    def test_uses_the_llm_identity_when_valid(self):
        llm = _FixedLLM(CREATE_IDENTITY_TOOL_NAME, _GOOD_ARGUMENTS)
        identity = generate_traveler_identity(llm, "brief", fallback=_FALLBACK)

        self.assertEqual(identity.name, "Selwyn")
        self.assertEqual(identity.starting_items, {"map": 1, "ink": 2})
        # No home in town -- carried over from the fallback, not invented.
        self.assertEqual(identity.home, _FALLBACK.home)

    def test_result_round_trips_through_the_npcs_json_format(self):
        llm = _FixedLLM(CREATE_IDENTITY_TOOL_NAME, _GOOD_ARGUMENTS)
        identity = generate_traveler_identity(llm, "brief", fallback=_FALLBACK)
        self.assertEqual(identity_from_record(identity_to_record(identity)), identity)

    def test_mock_backend_falls_back(self):
        identity = generate_traveler_identity(MockLLMClient(), "brief", fallback=_FALLBACK)
        self.assertIs(identity, _FALLBACK)

    def test_wrong_tool_falls_back(self):
        llm = _FixedLLM(SPEAK_TOOL_NAME, {"text": "hi"})
        self.assertIs(generate_traveler_identity(llm, "brief", fallback=_FALLBACK), _FALLBACK)

    def test_llm_error_falls_back(self):
        llm = _FixedLLM(CREATE_IDENTITY_TOOL_NAME, error=RuntimeError("network down"))
        with self.assertLogs("game_agents.traveler_identity", level="ERROR"):
            self.assertIs(generate_traveler_identity(llm, "brief", fallback=_FALLBACK), _FALLBACK)

    def test_missing_field_falls_back(self):
        arguments = {k: v for k, v in _GOOD_ARGUMENTS.items() if k != "backstory"}
        llm = _FixedLLM(CREATE_IDENTITY_TOOL_NAME, arguments)
        self.assertIs(generate_traveler_identity(llm, "brief", fallback=_FALLBACK), _FALLBACK)

    def test_malformed_field_falls_back(self):
        for key, bad in [
            ("traits", 5),
            ("starting_money", -5),
            ("starting_items", {"map": "one"}),
            ("starting_items", [{"item": "map"}]),
            ("name", "  "),
        ]:
            with self.subTest(key=key, bad=bad):
                llm = _FixedLLM(CREATE_IDENTITY_TOOL_NAME, {**_GOOD_ARGUMENTS, key: bad})
                with self.assertLogs("game_agents.traveler_identity", level="WARNING"):
                    self.assertIs(generate_traveler_identity(llm, "brief", fallback=_FALLBACK), _FALLBACK)

    def test_items_as_list_of_pairs(self):
        items = [{"item": "map", "count": 1}, {"item": "ink", "count": 2}]
        llm = _FixedLLM(CREATE_IDENTITY_TOOL_NAME, {**_GOOD_ARGUMENTS, "starting_items": items})
        identity = generate_traveler_identity(llm, "brief", fallback=_FALLBACK)
        self.assertEqual(identity.starting_items, {"map": 1, "ink": 2})

    def test_comma_joined_list_fields_are_split(self):
        llm = _FixedLLM(CREATE_IDENTITY_TOOL_NAME, {**_GOOD_ARGUMENTS, "traits": "wry, patient; tired"})
        identity = generate_traveler_identity(llm, "brief", fallback=_FALLBACK)
        self.assertEqual(identity.traits, ["wry", "patient", "tired"])

    def test_extra_fields_are_ignored(self):
        llm = _FixedLLM(CREATE_IDENTITY_TOOL_NAME, {**_GOOD_ARGUMENTS, "home": [5, 5], "favorite_color": "red"})
        identity = generate_traveler_identity(llm, "brief", fallback=_FALLBACK)
        self.assertEqual(identity.name, "Selwyn")
        self.assertEqual(identity.home, _FALLBACK.home)

    def test_taken_name_falls_back_case_insensitively(self):
        llm = _FixedLLM(CREATE_IDENTITY_TOOL_NAME, _GOOD_ARGUMENTS)
        identity = generate_traveler_identity(llm, "brief", fallback=_FALLBACK, taken_names=["Mara", "selwyn"])
        self.assertIs(identity, _FALLBACK)

    def test_taken_names_are_listed_in_the_prompt(self):
        llm = _FixedLLM(CREATE_IDENTITY_TOOL_NAME, _GOOD_ARGUMENTS)
        generate_traveler_identity(llm, "brief", fallback=_FALLBACK, taken_names=["Mara", "Finn"])
        self.assertIn("Finn, Mara", llm.last_messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
