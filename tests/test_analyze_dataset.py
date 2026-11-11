import unittest

from subtask3.analyze_dataset import analyze, response_contains


class AnalyzeDatasetTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [{
            'uuid': 'example1', 'domain': 'authors',
            'dialogue': {'turns': [{'turn_id': 0, 'query': 'How many from Cairo Microsoft Innovation Lab wrote the 2003 paper?'}]},
        }]
        self.gold = [{
            'uuid': 'example1', 'domain': 'authors',
            'output': [{
                'turn_id': 0,
                'answer': [[1]],
                'sequence': {
                    'tool_call': [
                        {'name': 'find_titles', 'arguments': {'year': '2003'}},
                        {'name': 'count_authors', 'arguments': {'affiliation': 'University of Tokyo', 'title': 'FIBER'}},
                    ],
                    'tool_response': [[['FIBER']], [[1]]],
                },
            }],
        }]

    def test_hops_and_domains(self):
        rows, flags, summary = analyze(self.inputs, self.gold)
        self.assertEqual(summary['gold_hop_distribution'], {'2': 1})
        self.assertEqual(summary['unique_domains'], 1)
        self.assertEqual(rows[0]['tool_names'], 'find_titles -> count_authors')

    def test_affiliation_review_flag(self):
        _, flags, _ = analyze(self.inputs, self.gold)
        self.assertEqual([f['type'] for f in flags], ['affiliation_needs_review'])

    def test_missing_gold(self):
        rows, flags, _ = analyze(self.inputs, [])
        self.assertFalse(rows[0]['has_gold'])
        self.assertEqual(flags[0]['type'], 'missing_gold')

    def test_response_provenance(self):
        self.assertTrue(response_contains('FIBER', [[['FIBER']]]))
        self.assertFalse(response_contains('Tokyo', [[['FIBER']]]))

    def test_no_flag_for_affiliation_from_question(self):
        self.gold[0]['output'][0]['sequence']['tool_call'][1]['arguments']['affiliation'] = 'Cairo Microsoft Innovation Lab'
        _, flags, _ = analyze(self.inputs, self.gold)
        self.assertEqual(flags, [])


if __name__ == '__main__':
    unittest.main()
