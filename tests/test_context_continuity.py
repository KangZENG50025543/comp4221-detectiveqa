"""Synthetic rule invariants, not claims about research observations."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from context_continuity import choose, ranked_ids


def chunks(native=False):
    return [{'id': i, 'spans': [{'pid': i if native else 0, 'start': 0, 'end': 1}]} for i in range(8)]


class Continuity(unittest.TestCase):
    def test_replaces_only_fifth_preserving_pool_and_capacity(self):
        scores = [(2, 9), (5, 8), (6, 7), (7, 6), (0, 5), (3, 1), (1, 0)]
        result = choose(chunks(), scores, 1)
        self.assertEqual(result['selected_ids'], [2, 3, 5, 6, 7])
        self.assertEqual((result['removed_id'], result['added_id']), (0, 3))
        self.assertEqual(choose(chunks(), scores, -1)['selected_ids'], [1, 2, 5, 6, 7])

    def test_never_reaches_outside_pool_or_replaces_existing_neighbor(self):
        scores = [(2, 9), (5, 8), (6, 7), (7, 6), (0, 5)]
        self.assertFalse(choose(chunks(), scores, 1)['changed'])
        scores.append((3, 10))
        self.assertFalse(choose(chunks(), scores, -1)['changed'])

    def test_native_document_boundary_and_long_native_fragment(self):
        scores = [(2, 9), (5, 8), (6, 7), (7, 6), (0, 5), (3, 1)]
        c = chunks(native=True)
        self.assertEqual(choose(c, scores, 1, True)['reason'], 'different_native_document')
        c[3]['spans'][0]['pid'] = 2
        self.assertTrue(choose(c, scores, 1, True)['changed'])

    def test_deterministic_ties_source_boundaries_and_short_pool(self):
        self.assertEqual(ranked_ids([(3, 1), (1, 1), (2, 2)]), [2, 1, 3])
        self.assertFalse(choose(chunks(), [(0, 1)], 1)['changed'])
        scores = [(0, 9), (4, 8), (5, 7), (6, 6), (7, 5)]
        self.assertEqual(choose(chunks(), scores, -1)['reason'], 'source_boundary')


if __name__ == '__main__':
    unittest.main()
