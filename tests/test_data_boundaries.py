"""Synthetic fixtures test invariants; they are not research data or results."""
import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detectiveqa import parse_novel,prefix_for_question,query_text,evidence_coverage

class Boundaries(unittest.TestCase):
    def test_reveal_never_enters_prefix(self):
        novel=parse_novel('[1]first\n\n[2]clue\n[3]REVEAL\n[4]later')
        self.assertEqual([p['pid'] for p in prefix_for_question(novel,3)],[1,2])
    def test_duplicate_paragraph_ids_rejected(self):
        with self.assertRaises(ValueError):parse_novel('[1]a\n[1]b')
    def test_query_is_allowlisted(self):
        q={'question':'Who?','options':{k:'option'+k for k in 'ABCD'},'answer':'SECRET','reference_reasoning':'SECRET','gold_pids':['SECRET']}
        self.assertNotIn('SECRET',query_text(q))
    def test_partial_evidence_and_repeated_labels(self):
        p=[{'pid':1,'text':'0123456789'},{'pid':2,'text':'abcd'}]
        c=[{'spans':[{'pid':1,'start':0,'end':6}]}]
        m=evidence_coverage(c,[1,1,2],p)
        self.assertEqual(m['gold_paragraphs'],2)
        self.assertEqual(m['full_paragraph_recall'],0)
        self.assertAlmostEqual(m['mean_paragraph_character_coverage'],.3)
        c.append({'spans':[{'pid':1,'start':4,'end':10}]})
        self.assertEqual(evidence_coverage(c,[1],p)['full_paragraph_recall'],1)
    def test_missing_reveal_rejected(self):
        with self.assertRaises(ValueError):prefix_for_question([{'pid':1,'text':'a'}],2)

if __name__=='__main__':unittest.main()
