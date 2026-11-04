"""Data boundaries and evidence metrics. No answer-model API calls."""
from pathlib import Path
import json, re

ROOT=Path(__file__).resolve().parents[1]

def read_json(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line.strip()]
def write_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf8')

def parse_novel(text):
    """Use explicit [N] markers, never infer paragraph numbers from line counts."""
    matches=list(re.finditer(r'(?m)^[ \t]*\[(\d+)\]',text))
    if not matches:raise ValueError('No paragraph markers')
    ids=[int(m.group(1)) for m in matches]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate paragraph IDs')
    if ids!=list(range(1,len(ids)+1)):raise ValueError('Non-contiguous paragraph IDs')
    result=[]
    for i,m in enumerate(matches):
        body=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
        if not body:raise ValueError(f'Empty paragraph {ids[i]}')
        result.append({'pid':ids[i],'text':body})
    return result

def prefix_for_question(paragraphs,cutoff):
    if cutoff not in {p['pid'] for p in paragraphs}:raise ValueError('Missing reveal paragraph')
    prefix=[p for p in paragraphs if p['pid']<cutoff]
    if not prefix:raise ValueError('Empty safe prefix')
    return prefix

def query_text(question):
    # Deliberate allowlist: adding labels to a source record cannot leak them here.
    return question['question']+'\n'+'\n'.join(f'{k}. {question["options"][k]}' for k in 'ABCD')

def union_length(intervals):
    end=-1;total=0
    for start,stop in sorted(intervals):
        if start<0 or stop<start:raise ValueError('Invalid character interval')
        total+=max(0,stop-max(start,end));end=max(end,stop)
    return total

def evidence_coverage(chunks,gold_pids,paragraphs):
    gold=sorted(set(gold_pids));by_id={p['pid']:p['text'] for p in paragraphs}
    if not gold or any(pid not in by_id for pid in gold):raise ValueError('Missing explicit evidence')
    fractions={}
    for pid in gold:
        spans=[(s['start'],s['end']) for c in chunks for s in c['spans'] if s['pid']==pid]
        if any(b>len(by_id[pid]) for a,b in spans):raise ValueError('Span outside paragraph')
        covered=union_length(spans);fractions[str(pid)]=covered/len(by_id[pid])
    return {'gold_paragraphs':len(gold),'full_paragraph_recall':sum(f==1 for f in fractions.values())/len(gold),
            'mean_paragraph_character_coverage':sum(fractions.values())/len(gold),
            'all_explicit_evidence_covered':all(f==1 for f in fractions.values()),'per_paragraph':fractions}
