import asyncio
import json
import unittest
from types import SimpleNamespace

from subtask3.missing_tool_recovery import (
    build_recovery_prompt, recover_missing_tool_call, select_valid_call,
)
from subtask3.apply_missing_call_patch import patch_source


def parse_text(content):
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        return None
    if isinstance(data, dict) and isinstance(data.get("name"), str):
        return {"name": data["name"], "args": data.get("arguments", {})}
    return None


def ai(*, calls=None, content=""):
    return SimpleNamespace(tool_calls=calls or [], content=content)


def tool(name, description="Lookup data"):
    return SimpleNamespace(name=name, description=description)


class TestRecovery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = []
        self.kwargs = dict(
            allowed_tool_names=["get_employee"],
            offered_tools=[tool("get_employee", "Return employee by ID")],
            question="Find employee 123's manager",
            plan=["Look up employee", "Find manager"],
            current_step="Look up employee",
            step_index=0,
            history_text="(none yet)",
            parse_text_call=parse_text,
        )

    async def fake_retry(self, prompt):
        self.calls.append(prompt)
        return ai(calls=[{"name": "get_employee", "args": {"id": 123}, "id": "recovered-1"}])

    async def test_existing_call_no_retry(self):
        original = ai(calls=[{"name": "get_employee", "args": {"id": 123}, "id": "first"}])
        outcome = await recover_missing_tool_call(initial_response=original, invoke_retry=self.fake_retry, enabled=True, **self.kwargs)
        self.assertEqual(outcome.call["id"], "first")
        self.assertEqual(outcome.reason, "original_valid")
        self.assertFalse(outcome.attempted)
        self.assertEqual(self.calls, [])

    async def test_disabled_preserves_no_call(self):
        outcome = await recover_missing_tool_call(initial_response=ai(content="answer"), invoke_retry=self.fake_retry, enabled=False, **self.kwargs)
        self.assertIsNone(outcome.call)
        self.assertEqual(outcome.reason, "disabled")
        self.assertEqual(self.calls, [])

    async def test_enabled_one_retry_recovers(self):
        outcome = await recover_missing_tool_call(initial_response=ai(content="I cannot find it"), invoke_retry=self.fake_retry, enabled=True, **self.kwargs)
        self.assertTrue(outcome.attempted)
        self.assertTrue(outcome.recovered)
        self.assertEqual(outcome.call["args"], {"id": 123})
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Return employee by ID", self.calls[0])

    async def test_invalid_retry_does_not_fabricate_call(self):
        async def wrong(_):
            self.calls.append(1)
            return ai(calls=[{"name": "unoffered_tool", "args": {}}])
        outcome = await recover_missing_tool_call(initial_response=ai(), invoke_retry=wrong, enabled=True, **self.kwargs)
        self.assertFalse(outcome.recovered)
        self.assertIsNone(outcome.call)
        self.assertEqual(len(self.calls), 1)

    async def test_no_tools_no_retry(self):
        kwargs = {**self.kwargs, "allowed_tool_names": [], "offered_tools": []}
        outcome = await recover_missing_tool_call(initial_response=ai(), invoke_retry=self.fake_retry, enabled=True, **kwargs)
        self.assertEqual(outcome.reason, "no_offered_tools")
        self.assertEqual(self.calls, [])

    async def test_exceptions_propagate_for_budget_handling(self):
        class Budget(Exception):
            pass
        async def raise_budget(_):
            raise Budget()
        with self.assertRaises(Budget):
            await recover_missing_tool_call(initial_response=ai(), invoke_retry=raise_budget, enabled=True, **self.kwargs)

    async def test_text_json_call_no_retry(self):
        original = ai(content='{"name":"get_employee","arguments":{"id":123}}')
        outcome = await recover_missing_tool_call(initial_response=original, invoke_retry=self.fake_retry, enabled=True, **self.kwargs)
        self.assertEqual(outcome.call["args"], {"id": 123})
        self.assertFalse(outcome.attempted)

    async def test_first_allowed_tool(self):
        original = ai(calls=[
            {"name": "bad_tool", "args": {"id": 0}},
            {"name": "get_employee", "args": {"id": 123}},
        ])
        outcome = await recover_missing_tool_call(initial_response=original, invoke_retry=self.fake_retry, enabled=True, **self.kwargs)
        self.assertEqual(outcome.call["name"], "get_employee")
        self.assertFalse(outcome.attempted)

    async def test_missing_args_preserves_original_behavior(self):
        original = ai(calls=[{"name": "get_employee", "args": "oops"}])
        outcome = await recover_missing_tool_call(initial_response=original, invoke_retry=self.fake_retry, enabled=True, **self.kwargs)
        self.assertEqual(outcome.call["args"], {})
        self.assertFalse(outcome.attempted)


class TestHelpers(unittest.TestCase):
    def test_prompt_has_context(self):
        s = build_recovery_prompt(
            question="How many authors?", plan=["Find paper", "Count authors"],
            current_step="Count authors", step_index=1,
            history_text="Paper from step 1: FIBER", offered_tools=[tool("get_author_count", "Count authors")],
        )
        self.assertIn("Paper from step 1", s)
        self.assertIn("Current step 2", s)
        self.assertIn("get_author_count", s)

    def test_fallback_unoffered_rejected(self):
        result = select_valid_call(ai(content='{"name":"wrong","arguments":{}}'), ["get_employee"], 0, parse_text)
        self.assertIsNone(result)

    def test_patch_produces_parsable_agent(self):
        fake_source = '''\
import os
from agents.agent_interface import AgentInterface
async def foo():
    call = None
    for tc in getattr(ai, "tool_calls", None) or []:
        if tc.get("name") in names:
            a = tc.get("args")
            call = {"name": tc["name"], "args": a if isinstance(a, dict) else {}, "id": tc.get("id") or f"s{si}"}
            break
    if call is None:
        parsed = parse_text_tool_call(ai_text)
        if parsed and parsed["name"] in names:
            call = {**parsed, "id": f"s{si}"}
    # step header: carries the step text and the offered tools
    if call is None:
        continue
'''
        # The standalone fixture uses the same indentation anchors as the real source;
        # the trailing continue is syntactically invalid outside a loop, so change it.
        fake_source = fake_source.replace("        continue\n", "        pass\n")
        patched = patch_source(fake_source)
        self.assertIn("outcome = await recover_missing_tool_call", patched)
        self.assertIn("from subtask3.missing_tool_recovery import", patched)
        with self.assertRaisesRegex(ValueError, "already present"):
            patch_source(patched)


if __name__ == "__main__":
    unittest.main()
