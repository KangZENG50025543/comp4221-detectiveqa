import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detectiveqa import ROOT,read_json,read_jsonl,prefix_for_question

class ActualData(unittest.TestCase):
    def test_book_split_and_gold_separation(self):
        split=read_json(ROOT/'configs/split.json')
        self.assertFalse(set(split['dev_books'])&set(split['test_books']))
        for group,path in [('dev','data/processed/dev_inputs.jsonl'),('test','data/restricted/test_inputs.jsonl')]:
            inputs=read_jsonl(ROOT/path);gold={r['qid']:r for r in read_jsonl(ROOT/f'data/restricted/{group}_gold.jsonl')}
            self.assertEqual({r['qid'] for r in inputs},set(split[group+'_ids']))
            for q in inputs:
                self.assertFalse({'answer','gold_pids','reference_reasoning','clue_positions'}&set(q))
                paras=read_json(ROOT/f'data/processed/books/{q["book_id"]}.json')
                prefix=prefix_for_question(paras,q['cutoff_exclusive'])
                self.assertTrue(all(p['pid']<q['cutoff_exclusive'] for p in prefix))
                self.assertTrue(set(gold[q['qid']]['gold_pids'])<={p['pid'] for p in prefix})

if __name__=='__main__':unittest.main()
