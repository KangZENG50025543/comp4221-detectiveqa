"""Synthetic invariants; no fixtures are claimed as research observations."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cross_dataset import ordinary_query, validate_input, coverage, retention, support_budget, chunks_for, require_frozen_hash


class TinyTokenizer:
    def encode(self, text, add_special_tokens=True):
        return list(text) + ([0, 0] if add_special_tokens else [])


def row():
    return {'qid': 'synthetic', 'dataset': 'musique', 'question': 'Which?', 'options': None,
            'group_id': 'synthetic', 'source_split': 'dev',
            'paragraphs': [{'pid': 0, 'text': 'first'}, {'pid': 1, 'text': 'second'}]}


class CrossDataset(unittest.TestCase):
    def test_frozen_hash_is_required_and_verified(self):
        digest = 'a' * 64
        require_frozen_hash({'hashes': {'inputs.jsonl': digest}}, 'inputs.jsonl', digest)
        for manifest in ({}, {'inputs_sha256': digest}, {'hashes': {'inputs.jsonl': 'b' * 64}}):
            with self.assertRaises(ValueError):
                require_frozen_hash(manifest, 'inputs.jsonl', digest)

    def test_native_documents_are_not_packed_together(self):
        chunks = chunks_for(row(), TinyTokenizer())
        self.assertEqual([c['id'] for c in chunks], [0, 1])
        self.assertEqual([len(c['spans']) for c in chunks], [1, 1])

    def test_labels_and_non_dev_inputs_rejected(self):
        q = row(); validate_input(q)
        q['answer'] = 'secret'
        self.assertNotIn('secret', ordinary_query(q))
        with self.assertRaises(ValueError): validate_input(q)
        del q['answer']; q['source_split'] = 'test'
        with self.assertRaises(ValueError): validate_input(q)

    def test_fragment_overlap_is_not_full_recall(self):
        paras = [{'pid': 1, 'text': '123456'}, {'pid': 2, 'text': 'abc'}]
        candidate = [{'id': 0, 'spans': [{'pid': 1, 'start': 0, 'end': 3}]},
                     {'id': 1, 'spans': [{'pid': 1, 'start': 3, 'end': 6}]},
                     {'id': 2, 'spans': [{'pid': 2, 'start': 0, 'end': 3}]}]
        cm = coverage(candidate, [1, 2], paras)
        fm = coverage(candidate[:1], [1, 2], paras)
        self.assertEqual(fm['full_paragraph_recall'], 0)
        self.assertEqual(fm['any_overlap_paragraph_recall'], .5)
        self.assertEqual(retention(fm, cm)['any_overlap_support_retention_given_candidates'], .5)
        self.assertFalse(support_budget(candidate, [1, 2], 2)['complete_support_paragraphs_fit_final_capacity'])


if __name__ == '__main__':
    unittest.main()
