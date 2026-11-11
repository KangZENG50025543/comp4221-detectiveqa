"""Synthetic cases only; no dataset answers loaded."""
import unittest
import sys
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from answer_diagnostic import validate_response, citation_checks, answer_scores
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import score_answer_diagnostic as pipeline


def response(**changes):
    r = dict(item_id='synthetic', answer='A', abstain=False,
             evidence_sufficient='unclear', citations=[], brief_reason='')
    r.update(changes)
    return r


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.item = dict(item_id='synthetic', options={'A': 'one'}, evidence=[
            dict(chunk_id=0, fragments=[dict(pid=1, start=0, end=3, text='abc'),
                                      dict(pid=1, start=8, end=11, text='xyz'),
                                      dict(pid=2, start=0, end=5, text='other')])])

    def test_schema(self):
        for bad in [None, 'not json', [], {}, response(extra=True), response(abstain=1),
                    response(answer='B'), response(citations=[{'pid': True, 'quote': 'abc'}]),
                    response(brief_reason=None), response(evidence_sufficient='maybe')]:
            self.assertTrue(validate_response(bad, self.item))
        self.assertFalse(validate_response(response(), self.item))

    def test_citations(self):
        r = response(citations=[{'pid': 99, 'quote': 'abc'}, {'pid': 1, 'quote': 'other'},
                                {'pid': 1, 'quote': 'abcxyz'}, {'pid': 1, 'quote': 'abc'}])
        c = citation_checks(r, self.item)
        self.assertEqual(c['citation_count'], 4)
        self.assertEqual(c['invalid_pid_count'], 1)
        self.assertEqual(c['invalid_quote_count'], 3)
        self.assertTrue(c['citations'][3]['quote_in_visible_fragment'])

    def test_empty_and_abstention(self):
        empty = dict(self.item, evidence=[])
        self.assertFalse(validate_response(response(), empty))
        self.assertTrue(validate_response(response(evidence_sufficient='yes'), empty))
        self.assertTrue(validate_response(response(abstain=True), empty))
        self.assertFalse(validate_response(response(abstain=True, answer=None), empty))
        self.assertEqual(answer_scores(response(abstain=True, answer=None), 'A')['exact_match'], 0)
        self.assertTrue(validate_response(response(answer=' '), empty))

    def test_scores(self):
        self.assertEqual(answer_scores(response(answer='The CAT!'), ['dog', 'cat'])['exact_match'], 1)
        self.assertAlmostEqual(answer_scores(response(answer='cat cat'), 'cat')['token_f1'], 2/3)
        self.assertEqual(answer_scores(response(), 'A', self.item['options'])['exact_match'], 1)
        self.assertEqual(answer_scores(response(answer='B'), 'B', self.item['options'])['exact_match'], 0)
        self.assertEqual(answer_scores(response(extra=1), 'A')['exact_match'], 0)
        self.assertEqual(answer_scores(response(answer=''), '')['exact_match'], 0)


class LockTests(unittest.TestCase):
    def make_run(self, base):
        run = Path(base)
        (run/'responses').mkdir()
        (run/'protocol.json').write_text('{}', encoding='utf8')
        (run/'condition_key.json').write_text(json.dumps([{'item_id':'one'}, {'item_id':'two'}]), encoding='utf8')
        (run/'responses/one.json').write_text('{}', encoding='utf8')
        registry = [dict(item_id='one', completed_utc='synthetic', output_exists=True),
                    dict(item_id='two', completed_utc='synthetic', output_exists=False)]
        (run/'agent_registry.json').write_text(json.dumps(registry), encoding='utf8')
        return run

    def test_lock_rejects_unfinished_or_duplicate_attempts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'checked_protocol', return_value={}):
            run = self.make_run(tmp)
            registry = json.loads((run/'agent_registry.json').read_text('utf8'))
            registry[1]['item_id'] = 'one'
            (run/'agent_registry.json').write_text(json.dumps(registry), encoding='utf8')
            with self.assertRaises(ValueError):
                pipeline.lock(run)
            self.assertFalse((run/'responses.lock.json').exists())

    def test_missing_answer_is_locked_not_silently_excluded(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'checked_protocol', return_value={}):
            run = self.make_run(tmp)
            pipeline.lock(run)
            lock = json.loads((run/'responses.lock.json').read_text('utf8'))
            self.assertEqual(lock['count'], 2)
            self.assertEqual(lock['present_count'], 1)
            self.assertEqual(lock['missing_outputs'], ['two'])
            self.assertIsNone(lock['outputs']['responses/two.json'])
            (run/'responses/two.json').write_text('{}', encoding='utf8')
            with patch.object(pipeline, 'read_jsonl', side_effect=AssertionError('gold must not be read')):
                with self.assertRaisesRegex(ValueError, 'filled after lock'):
                    pipeline.score(run)

    def test_changed_answer_rejected_before_gold_read(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'checked_protocol', return_value={}):
            run = self.make_run(tmp)
            pipeline.lock(run)
            (run/'responses/one.json').write_text('{"changed":true}', encoding='utf8')
            with patch.object(pipeline, 'read_jsonl', side_effect=AssertionError('gold must not be read')):
                with self.assertRaisesRegex(ValueError, 'Raw answer changed'):
                    pipeline.score(run)


if __name__ == '__main__':
    unittest.main()
