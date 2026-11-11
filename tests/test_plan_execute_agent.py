import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.tools import StructuredTool
from subtask3.plan_execute_agent import PlanExecuteAgent, check_result


class FakeLLM:
    def __init__(self):
        self.turn = 0

    async def ainvoke(self, messages):
        outputs = [
            '{"steps":["Look up employee","Look up manager","Answer"]}',
            '{"action":"call","name":"employee","arguments":{"employee_id":123}}',
            '{"action":"call","name":"manager","arguments":{"manager_id":456}}',
            '{"action":"finish","answer":"The manager email is boss@example.org."}',
        ]
        output = outputs[self.turn]
        self.turn += 1
        return SimpleNamespace(content=output)


def employee(employee_id: int) -> str:
    return json.dumps({"manager_id": 456})


def manager(manager_id: int) -> str:
    return json.dumps({"email": "boss@example.org"})


class TestAgent(unittest.TestCase):
    def test_dependent_calls(self):
        tools = [
            StructuredTool.from_function(
                employee,
                description="Retrieve employee information using an employee ID."
            ),
            StructuredTool.from_function(
                manager,
                description="Retrieve a manager's email using their ID."
            ),
        ]
        agent = PlanExecuteAgent(FakeLLM(), tools=tools)
        result = asyncio.run(agent.run("Email of employee 123's manager?"))
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[1]["arguments"]["manager_id"], 456)
        self.assertIn("boss@example.org", result.content)
        self.assertEqual(len(result.metadata["step_checks"]), 2)

    def test_error_detection(self):
        self.assertFalse(check_result('{"error":"not found"}')[0])
        self.assertTrue(check_result('{"manager_id":456}')[0])


if __name__ == "__main__":
    unittest.main()
