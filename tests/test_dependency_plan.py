import json
import unittest

from subtask3.dependency_plan import (
    DEPENDENCY_PLAN_EX_A,
    dependency_hint,
    parse_dependency_plan,
)


class DependencyPlanTests(unittest.TestCase):
    def test_example(self):
        plan = parse_dependency_plan(DEPENDENCY_PLAN_EX_A)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[1]["inputs"][0]["source"], "step_1")
        self.assertIn("step 1", dependency_hint(plan[1]))

    def test_real_authors_example(self):
        raw = json.dumps([
            {"goal": "Find paper titles for the 2003 author affiliation", "inputs": [
                {"name": "affiliation", "source": "question"},
                {"name": "year", "source": "question"}]},
            {"goal": "Count authors at the requested laboratory", "inputs": [
                {"name": "title", "source": "step_1", "transform": "extract paper title"},
                {"name": "affiliation", "source": "question"}]},
        ])
        plan = parse_dependency_plan(raw)
        self.assertEqual(plan[1]["inputs"][0]["source"], "step_1")
        self.assertIn("extract paper title", dependency_hint(plan[1]))
        self.assertIn("use the question", dependency_hint(plan[1]))

    def test_date_format_transform_kept(self):
        plan = parse_dependency_plan('[{"goal":"Find records","inputs":[{"name":"date","source":"question","transform":"convert to YYYY-MM-DD"}]}]')
        self.assertIn("YYYY-MM-DD", dependency_hint(plan[0]))

    def test_forward_references_omitted(self):
        plan = parse_dependency_plan('[{"goal":"First","inputs":[{"name":"id","source":"step_2"}]},{"goal":"Second","inputs":[{"name":"id","source":"step_1"}]}]')
        self.assertEqual(plan[0]["inputs"], [])
        self.assertEqual(plan[1]["inputs"][0]["source"], "step_1")

    def test_unstructured_plan_falls_back(self):
        self.assertEqual(parse_dependency_plan('["Find papers", "Count authors"]'), [])

    def test_malformed_json_falls_back(self):
        self.assertEqual(parse_dependency_plan('not JSON'), [])

    def test_step_cap(self):
        data = json.dumps([{"goal": str(i), "inputs": []} for i in range(6)])
        self.assertEqual(len(parse_dependency_plan(data, max_steps=3)), 3)

    def test_no_unjustified_rejection(self):
        plan = parse_dependency_plan('[{"goal":"Find","inputs":[{"name":"city","source":"question","transform":"normalize spelling"}]}]')
        self.assertEqual(len(plan), 1)
        self.assertIn("normalize spelling", dependency_hint(plan[0]))


if __name__ == "__main__":
    unittest.main()
