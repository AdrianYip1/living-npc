from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from game_agents.agent import Agent
from game_agents.chatty_mock import ChattyMockLLMClient
from game_agents.conversation import ConversationHooks, run_conversation
from game_agents.identity import Identity
from game_agents.llm import ENDS_CONVERSATION_FIELD, SPEAK_TOOL_NAME, LLMResult, MockLLMClient, ToolCall
from game_agents.registry import CONVERSATION_STARTED_PREFIX, NPCRegistry, conversation_happened
from game_agents.storage import save_identities
from game_agents.tools import Tool, ToolRegistry
from game_agents.world import (
    IN_REACH_LABEL,
    IN_VIEW_LABEL,
    INTERACTION_RANGE,
    PERSONAL_SPACE,
    distance,
    keep_personal_space,
    render_surroundings,
)


def _identity(name: str, **kwargs) -> Identity:
    return Identity(name=name, traits=[], backstory="", speech_style="", goals=[], **kwargs)


class _ScriptedLLM:
    """Speaks the given lines in order (a line ending in "!bye" is marked
    as a goodbye), and records every system prompt and tool list it saw.
    """

    def __init__(self, *lines: str):
        self._lines = list(lines)
        self.systems: list[str] = []
        self.tools: list[list[dict]] = []

    def complete(self, *, system, messages, tools):
        self.systems.append(system)
        self.tools.append(tools)
        text = self._lines.pop(0) if self._lines else "..."
        arguments = {"text": text.removesuffix("!bye")}
        if text.endswith("!bye"):
            arguments[ENDS_CONVERSATION_FIELD] = True
        return LLMResult(tool_call=ToolCall(name=SPEAK_TOOL_NAME, arguments=arguments))


class ConversationFlowTests(unittest.TestCase):
    def test_goodbye_gets_one_reply_then_ends(self):
        llm = _ScriptedLLM("Hi Bo.", "Hey Al.", "Must run!bye", "Bye then.", "never said")
        a, b = Agent(_identity("Al"), llm), Agent(_identity("Bo"), llm)

        transcript = run_conversation(a, b, turns=10)

        self.assertEqual([t.utterance for t in transcript], ["Hi Bo.", "Hey Al.", "Must run", "Bye then."])
        self.assertTrue(transcript[2].ends_conversation)
        self.assertIn("wrapping up the conversation", llm.systems[3])

    def test_each_speaker_sees_the_conversation_so_far(self):
        llm = _ScriptedLLM("Hi Bo.", "Hey Al.", "Nice day.")
        a, b = Agent(_identity("Al"), llm), Agent(_identity("Bo"), llm)

        run_conversation(a, b, turns=3)

        self.assertIn("Open the conversation", llm.systems[0])
        self.assertIn("Conversation so far:\nAl: Hi Bo.\nBo: Hey Al.", llm.systems[2])

    def test_goodbye_flag_is_only_offered_inside_a_conversation(self):
        llm = _ScriptedLLM()
        agent = Agent(_identity("Al"), llm)
        agent.respond("hello")
        run_conversation(agent, Agent(_identity("Bo"), MockLLMClient()), turns=1)

        self.assertNotIn(ENDS_CONVERSATION_FIELD, llm.tools[0][0]["parameters"]["properties"])
        self.assertIn(ENDS_CONVERSATION_FIELD, llm.tools[1][0]["parameters"]["properties"])
        self.assertFalse(agent.respond("hi").ends_conversation)  # ignored outside a conversation

    def test_memories_quote_who_said_what(self):
        llm = _ScriptedLLM("Hi Bo.", "Hey Al.")
        a, b = Agent(_identity("Al"), llm), Agent(_identity("Bo"), llm)

        run_conversation(a, b, turns=2)

        [memory] = b.memory.all()
        self.assertEqual(memory.content, 'Al says: "Hi Bo." -> Hey Al.')


class ConversationHooksTests(unittest.TestCase):
    def _registry(self, tmp, llm=None):
        npcs_path = Path(tmp) / "npcs.json"
        save_identities([_identity("Mara"), _identity("Finn")], npcs_path)
        return NPCRegistry(npcs_path, Path(tmp) / "memory", llm or MockLLMClient(), conversation_turns=2)

    def test_scheduled_exchange_keeps_both_busy_until_it_has_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp)
            scheduled = []
            ended = []
            registry.conversation_hooks = ConversationHooks(
                run=lambda exchange: scheduled.append(exchange) or True, on_end=ended.append
            )

            result = registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})

            self.assertTrue(result.startswith(CONVERSATION_STARTED_PREFIX))
            self.assertTrue(conversation_happened(result))
            self.assertEqual(registry.partner_of("Mara"), "Finn")
            self.assertEqual(registry.partner_of("Finn"), "Mara")
            self.assertEqual(registry.get("Finn").memory.all(), [])

            scheduled[0]()

            self.assertFalse(registry.is_busy("Mara"))
            self.assertIsNone(registry.partner_of("Finn"))
            self.assertEqual(len(ended), 1)
            self.assertEqual(len(registry.get("Finn").memory.all()), 1)

    def test_unschedulable_exchange_is_refused_and_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp)
            registry.conversation_hooks = ConversationHooks(run=lambda exchange: False)

            result = registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})

            self.assertFalse(conversation_happened(result))
            self.assertFalse(registry.is_busy("Mara"))
            self.assertFalse(registry.is_busy("Finn"))

    def test_on_turn_sees_every_turn_as_it_happens(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp)
            turns = []
            registry.conversation_hooks = ConversationHooks(on_turn=turns.append)

            registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})

            self.assertEqual([(t.speaker, t.listener) for t in turns], [("Mara", "Finn"), ("Finn", "Mara")])

    def test_player_conversation_has_no_partner(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self._registry(tmp)
            registry.try_occupy("Mara")
            self.assertIsNone(registry.partner_of("Mara"))


class SurroundingsTests(unittest.TestCase):
    def test_splits_people_by_reach_and_skips_the_far_away(self):
        text = render_surroundings(
            (0, 0), [("Far", (90, 0), False), ("View", (30, 0), False), ("Near", (5, 0), True)]
        )
        self.assertEqual(
            text, f"{IN_REACH_LABEL}: Near at (5, 0) (busy talking).\n{IN_VIEW_LABEL}: View at (30, 0)."
        )

    def test_empty_when_nobody_is_around(self):
        self.assertEqual(render_surroundings((0, 0), [("Far", (90, 90), False)]), "")


class ChattyMockTests(unittest.TestCase):
    def _tools(self, *names):
        return [{"name": name, "description": "", "parameters": {"type": "object", "properties": {}}} for name in names]

    def _speak(self, conversation=False):
        properties = {"text": {"type": "string"}}
        if conversation:
            properties[ENDS_CONVERSATION_FIELD] = {"type": "boolean"}
        return {"name": SPEAK_TOOL_NAME, "description": "", "parameters": {"type": "object", "properties": properties}}

    def _act(self, llm, system, stimulus="It is now 08:00 (morning)."):
        tools = [self._speak(), *self._tools("initiate_conversation", "move_to", "wait")]
        return llm.complete(system=system, messages=[{"role": "user", "content": stimulus}], tools=tools).tool_call

    def test_starts_a_conversation_with_someone_in_reach(self):
        system = "You are Mara.\nYou are currently at (0, 0).\n" + render_surroundings((0, 0), [("Finn", (4, 0), False)])
        calls = [self._act(ChattyMockLLMClient(seed=s), system) for s in range(10)]
        self.assertIn(ToolCall(name="initiate_conversation", arguments={"target_name": "Finn"}), calls)

    def test_never_approaches_someone_busy(self):
        system = "You are Mara.\nYou are currently at (0, 0).\n" + render_surroundings((0, 0), [("Finn", (4, 0), True)])
        for seed in range(10):
            self.assertNotEqual(self._act(ChattyMockLLMClient(seed=seed), system).name, "initiate_conversation")

    def test_resident_heads_to_work_in_the_morning(self):
        system = "You are Mara.\nWorkplace: (12, -30)\nYou are currently at (-40, 15).\nTime: 08:00 (morning)"
        self.assertEqual(
            self._act(ChattyMockLLMClient(seed=1), system), ToolCall(name="move_to", arguments={"x": 12, "y": -30})
        )

    _BUYER = (
        "You are Wren, a traveler.\nYour exit point: (100, 5), on the edge of the map.\n"
        "You've stopped in town to buy horseshoe from Mara, who works at The Forge at (12, -30).\n"
        "You are currently at (12, -22).\n"
    )

    def test_buyer_starts_talking_to_its_seller(self):
        system = self._BUYER + render_surroundings((12, -22), [("Mara", (12, -30), False), ("Finn", (14, -22), False)])
        call = self._act(ChattyMockLLMClient(seed=1), system, "You've reached (12, -22).")
        self.assertEqual(call, ToolCall(name="initiate_conversation", arguments={"target_name": "Mara"}))

    def test_buyer_heads_for_its_exit_once_it_has_bought(self):
        call = self._act(ChattyMockLLMClient(seed=1), self._BUYER, "You've bought the horseshoe you came for.")
        self.assertEqual(call, ToolCall(name="move_to", arguments={"x": 100, "y": 5}))

    def test_buyer_asks_the_price_then_buys_at_it(self):
        llm = ChattyMockLLMClient(seed=1)
        buyer = Agent(_identity("Wren"), llm, standing_context=self._BUYER)
        seller = Agent(_identity("Mara"), llm)
        tools = ToolRegistry()
        bought = []
        tools.register(
            Tool(
                name="buy_item",
                description="",
                parameters={"type": "object", "properties": {}},
                handler=lambda **kwargs: bought.append(kwargs) or "You bought it.",
            )
        )
        buyer.tools = tools

        transcript = run_conversation(buyer, seller, turns=6)

        self.assertEqual(transcript[0].utterance, "Hello, Mara. I'm after a horseshoe. What's your price?")
        self.assertRegex(transcript[1].utterance, r"^\d coins for the horseshoe")
        self.assertEqual(transcript[2].action["name"], "buy_item")
        self.assertEqual(bought[0]["item"], "horseshoe")
        self.assertEqual(bought[0]["seller_name"], "Mara")

    def test_holds_a_short_conversation_that_ends_on_a_goodbye(self):
        llm = ChattyMockLLMClient(seed=3)
        a, b = Agent(_identity("Al"), llm), Agent(_identity("Bo"), llm)

        transcript = run_conversation(a, b, turns=20)

        self.assertGreaterEqual(len(transcript), 3)
        self.assertLess(len(transcript), 20)
        self.assertTrue(any(t.ends_conversation for t in transcript))
        self.assertIn("Bo", transcript[0].utterance)

    def test_is_safe_to_share_across_threads(self):
        llm = ChattyMockLLMClient(seed=1)
        system = "You are Mara.\nYou are currently at (0, 0)."
        threads = [threading.Thread(target=self._act, args=(llm, system)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_identity_requests_fall_back_like_the_plain_mock(self):
        tools = self._tools("create_identity")
        call = ChattyMockLLMClient().complete(system="", messages=[], tools=tools).tool_call
        self.assertNotEqual(call.name, "create_identity")


class SoakRegressionTests(unittest.TestCase):
    """Issues found running the game against a real model."""

    def test_starting_a_conversation_is_not_offered_inside_one(self):
        # Claude answered "You walk up to Mara to start a conversation." by
        # calling initiate_conversation again -- ending the conversation
        # before a word was said, over and over.
        with tempfile.TemporaryDirectory() as tmp:
            npcs_path = Path(tmp) / "npcs.json"
            save_identities([_identity("Mara"), _identity("Finn")], npcs_path)
            llm = _ScriptedLLM("Hi Finn.")
            registry = NPCRegistry(npcs_path, Path(tmp) / "memory", llm, conversation_turns=1)

            registry.get("Mara").tools.execute("initiate_conversation", {"target_name": "Finn"})

            offered = [t["name"] for t in llm.tools[0]]
            self.assertNotIn("initiate_conversation", offered)
            self.assertIn("move_to", offered)
            self.assertIn("initiate_conversation", [t["name"] for t in registry.get("Mara").tools.schemas()])

    def _trader(self, name, script):
        """An agent that plays `script` in order: a string is a line, a
        dict is a tool call ({"name": ..., "arguments": ...})."""

        class _Script:
            def complete(self, *, system, messages, tools):
                step = script.pop(0) if script else "..."
                if isinstance(step, dict):
                    return LLMResult(tool_call=ToolCall(**step))
                return LLMResult(tool_call=ToolCall(name=SPEAK_TOOL_NAME, arguments={"text": step}))

        tools = ToolRegistry()
        for tool_name in ("sell_item", "move_to"):
            tools.register(
                Tool(name=tool_name, description="", parameters={"type": "object", "properties": {}},
                     handler=lambda **kwargs: "You sold 3 x fish to Yusef for 6 coins. You have 18 coins.")
            )
        return Agent(_identity(name), _Script(), tools)

    def test_a_trade_mid_conversation_is_followed_by_a_line_not_an_ending(self):
        # Finn sold his fish mid-conversation, which ended it on the spot;
        # Yusef's "Good chatting, Finn" then came out as a stray remark.
        sell = {"name": "sell_item", "arguments": {}}
        yusef = self._trader("Yusef", ["Three fish for six?", "Pleasure.", "Bye!"])
        finn = self._trader("Finn", ["Deal.", sell, "There you go, fresh as anything.", "Take care."])

        transcript = run_conversation(yusef, finn, turns=8)

        self.assertEqual(
            [(t.speaker, t.utterance or t.action["name"]) for t in transcript][:5],
            [
                ("Yusef", "Three fish for six?"),
                ("Finn", "Deal."),
                ("Yusef", "Pleasure."),
                ("Finn", "sell_item"),
                ("Finn", "There you go, fresh as anything."),
            ],
        )
        self.assertIn("You sold 3 x fish", transcript[4].stimulus)

    def test_two_actions_in_a_row_still_end_it(self):
        sell = {"name": "sell_item", "arguments": {}}
        yusef = self._trader("Yusef", ["Hi."])
        finn = self._trader("Finn", [dict(sell), dict(sell), "never said"])

        transcript = run_conversation(yusef, finn, turns=8)

        self.assertEqual([t.action["name"] for t in transcript[1:]], ["sell_item", "sell_item"])

    def test_walking_off_still_ends_it(self):
        yusef = self._trader("Yusef", ["Hi."])
        finn = self._trader("Finn", [{"name": "move_to", "arguments": {}}, "never said"])

        transcript = run_conversation(yusef, finn, turns=8)

        self.assertEqual(len(transcript), 2)

    def test_walking_to_someone_stops_in_front_of_them(self):
        self.assertEqual(keep_personal_space((12, -30), (40, -30), [(12, -30)]), (19, -30))

    def test_a_clear_destination_is_left_alone(self):
        self.assertEqual(keep_personal_space((12, -30), (40, -30), [(30, 30)]), (12, -30))

    def test_a_second_visitor_does_not_land_on_the_first(self):
        first = keep_personal_space((12, -30), (40, -30), [(12, -30)])
        second = keep_personal_space((12, -30), (40, -30), [(12, -30), first])
        for other in ((12, -30), first):
            self.assertGreaterEqual(distance(second, other), PERSONAL_SPACE - 0.5)
        self.assertLessEqual(distance(second, (12, -30)), INTERACTION_RANGE)

    def test_a_crowd_forms_a_ring_in_talking_range(self):
        # Several travelers all walking up to Mara: each gets its own spot,
        # none on top of another, all close enough to talk to her.
        mara = (12, -30)
        taken = [mara]
        for _ in range(5):
            spot = keep_personal_space(mara, (60, -30), taken)
            for other in taken:
                self.assertGreaterEqual(distance(spot, other), PERSONAL_SPACE - 0.5)
            self.assertLessEqual(distance(spot, mara), INTERACTION_RANGE)
            taken.append(spot)

    def test_icons_do_not_overlap_at_personal_space(self):
        # NPC_RADIUS 16px and WORLD_SCALE 6 in app.js: an icon is ~5.3 units wide.
        self.assertGreater(PERSONAL_SPACE, 2 * 16 / 6)
        self.assertLess(PERSONAL_SPACE, INTERACTION_RANGE)

    def test_move_tool_keeps_personal_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            npcs_path = Path(tmp) / "npcs.json"
            save_identities([_identity("Mara", home=(12, -30)), _identity("Wren", home=(40, -30))], npcs_path)
            registry = NPCRegistry(npcs_path, Path(tmp) / "memory", MockLLMClient())

            result = registry.get("Wren").tools.execute("move_to", {"x": 12, "y": -30})

            self.assertEqual(registry.get("Wren").destination, (19, -30))
            self.assertIn("(19, -30)", result)


if __name__ == "__main__":
    unittest.main()
