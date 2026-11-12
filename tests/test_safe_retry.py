import unittest

from subtask3.safe_retry import should_retry, single_argument_variants


class SafeRetryTests(unittest.TestCase):
    def test_real_error_retried(self):
        self.assertTrue(should_retry("error"))

    def test_zero_can_be_correct(self):
        self.assertFalse(should_retry("zero"))

    def test_empty_can_be_correct(self):
        self.assertFalse(should_retry("empty"))

    def test_opt_in_zero_retry(self):
        self.assertTrue(should_retry("zero", retry_zero=True))

    def test_opt_in_empty_retry(self):
        self.assertTrue(should_retry("empty", retry_empty=True))

    def test_success_never_retried(self):
        self.assertFalse(should_retry(None, retry_zero=True, retry_empty=True))

    def test_only_one_field_changes(self):
        args = {"name": "ALICE", "affiliation": "YALE", "year": "2003"}
        variants = single_argument_variants(args, limit=5)
        self.assertGreater(len(variants), 0)
        for _, cand in variants:
            differences = [key for key in args if cand[key] != args[key]]
            self.assertEqual(len(differences), 1)
            self.assertEqual(cand["year"], "2003")

    def test_stringified_ids_unchanged(self):
        args = {"person_id": "ABC123", "year": "2003", "date": "2008-05-28"}
        self.assertEqual(single_argument_variants(args), [])

    def test_spaces_trimmed(self):
        args = {"name": "  Alice ", "other": 5}
        label, candidate = single_argument_variants(args, limit=1)[0]
        self.assertEqual(label, "strip:name")
        self.assertEqual(candidate, {"name": "Alice", "other": 5})

    def test_noop_skipped(self):
        self.assertEqual(single_argument_variants({"count": 1}), [])

    def test_limit(self):
        self.assertLessEqual(len(single_argument_variants({"name": "ABC", "city": "NYC"}, limit=2)), 2)


if __name__ == "__main__":
    unittest.main()
