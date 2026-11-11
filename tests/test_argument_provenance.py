import unittest
from subtask3.step_checker import check_step, trace_argument_provenance

QUESTION = (
    'Among the authors of the paper published in 2003 by an author with affiliation '
    'with Department of Network Science, Graduate School of Information Systems, '
    'The University of Electro-Communications, how many of them are affiliated '
    'with Cairo Microsoft Innovation Lab?'
)
TITLE = 'FIBER: A Generalized Framework for Auto-tuning Software'
PREVIOUS = [[[TITLE]]]


class TestArgumentProvenance(unittest.TestCase):
    def test_first_api_arguments_from_question(self):
        origins = trace_argument_provenance({
            'affiliation': 'Department of Network Science, Graduate School of Information Systems, The University of Electro-Communications',
            'year': '2003',
        }, question=QUESTION)
        self.assertEqual(origins['affiliation'], 'question')
        self.assertEqual(origins['year'], 'question')

    def test_second_api_dependencies(self):
        origins = trace_argument_provenance({
            'affiliation': 'Cairo Microsoft Innovation Lab',
            'title': TITLE,
        }, question=QUESTION, previous_results=PREVIOUS)
        self.assertEqual(origins['affiliation'], 'question')
        self.assertEqual(origins['title'], 'previous_response')

    def test_suspected_gold_inconsistency_warning(self):
        args = {'affiliation': 'University of Tokyo', 'title': TITLE}
        result = check_step([[1]], question=QUESTION,
                            previous_results=PREVIOUS, next_arguments=args)
        self.assertTrue(result.valid)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn('University of Tokyo', result.warnings[0])

    def test_nested_json_response_and_matching_id(self):
        origins = trace_argument_provenance({'manager_id': 456},
                                           question='Who manages employee 123?',
                                           previous_results=['{"user":{"manager_id":456}}'])
        self.assertEqual(origins['manager_id'], 'previous_response')

    def test_missing_source_warns_but_does_not_fail(self):
        result = check_step({'count': 1}, question='What is the count?',
                            next_arguments={'page': 20})
        self.assertTrue(result.valid)
        self.assertEqual(len(result.warnings), 1)

    def test_both_sources(self):
        origins = trace_argument_provenance({'year': '2003'},
                                           question='Find items from 2003',
                                           previous_results=[{'year': '2003'}])
        self.assertEqual(origins['year'], 'question_and_previous_response')


if __name__ == '__main__':
    unittest.main()
