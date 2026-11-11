import unittest
from subtask3.step_checker import check_step


class TestStepChecker(unittest.TestCase):
    def test_valid_json(self):
        self.assertTrue(check_step('{"manager_id": 456}').valid)

    def test_errors(self):
        for result in [None, "", [], {}, '{"error":"not found"}',
                       {"isError": True}, "Error: timeout",
                       [{"type":"text", "text": '{"error":"not found"}'}]]:
            with self.subTest(result=result):
                self.assertFalse(check_step(result).valid)

    def test_required_fields(self):
        result = '{"user":{"manager_id":456}}'
        self.assertTrue(check_step(result, required_fields=["user.manager_id"]).valid)
        check = check_step(result, required_fields=["user.email"])
        self.assertFalse(check.valid)
        self.assertIn("user.email", check.reason)

    def test_dependency_warning_not_failure(self):
        check = check_step(
            '{"email":"boss@example.org"}',
            next_arguments={"manager_id": 999},
            previous_results=['{"manager_id":456}'],
        )
        self.assertTrue(check.valid)
        self.assertEqual(len(check.warnings), 1)
        known = check_step(
            '{"email":"boss@example.org"}',
            next_arguments={"manager_id": 456},
            previous_results=['{"manager_id":456}'],
        )
        self.assertEqual(known.warnings, [])

    def test_zero_and_false_valid(self):
        self.assertTrue(check_step(0).valid)
        self.assertTrue(check_step(False).valid)


if __name__ == "__main__":
    unittest.main()
