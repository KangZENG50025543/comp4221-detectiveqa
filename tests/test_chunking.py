import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detectiveqa import ROOT,read_json,read_jsonl,prefix_for_question,evidence_coverage
from retrieval import make_chunks

class Chunking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/embedding',local_files_only=True)
    def test_real_development_prefix_lossless(self):
        q=read_jsonl(ROOT/'data/processed/dev_inputs.jsonl')[0]
        prefix=prefix_for_question(read_json(ROOT/f'data/processed/books/{q["book_id"]}.json'),q['cutoff_exclusive'])
        chunks=make_chunks(prefix,self.tokenizer)
        self.assertTrue(all(len(self.tokenizer.encode(c['text']))<=512 for c in chunks))
        self.assertTrue(all(s['pid']<q['cutoff_exclusive'] for c in chunks for s in c['spans']))
        coverage=evidence_coverage(chunks,[p['pid'] for p in prefix],prefix)
        self.assertTrue(coverage['all_explicit_evidence_covered'])
    def test_long_paragraph_preserves_every_character(self):
        # A synthetic stress fixture only; never used as a research example or result.
        paras=[{'pid':1,'text':'A long paragraph with spaces. '*1000}]
        chunks=make_chunks(paras,self.tokenizer)
        self.assertGreater(len(chunks),1)
        self.assertEqual(''.join(paras[0]['text'][s['start']:s['end']] for c in chunks for s in c['spans']),paras[0]['text'])
        self.assertTrue(all(c['bge_tokens']<=512 for c in chunks))

if __name__=='__main__':unittest.main()
