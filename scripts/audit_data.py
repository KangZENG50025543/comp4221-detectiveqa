"""Validate all metadata and paragraph boundaries, then split by book before inference."""
from pathlib import Path
import sys,hashlib,json,random,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detectiveqa import ROOT,read_json,write_json,write_jsonl,parse_novel,prefix_for_question

cfg=read_json(ROOT/'configs/protocol.json')
raw=ROOT/'data/raw'; excluded=[];file_checks=[];books={};valid=[];seen={};raw_questions=0
for file in sorted((raw/'anno_data_zh/human_anno').glob('*.json'),key=lambda p:int(p.stem)):
    arr=read_json(file)
    if not isinstance(arr,list) or len(arr)!=1:raise ValueError(f'Unexpected annotation schema: {file.name}')
    obj=arr[0];book=int(file.stem);raw_questions+=len(obj['questions'])
    fingerprint=hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if fingerprint in seen:
        excluded.extend({'qid':f'zh-human-{book}-{i}','reason':['duplicate_annotation_file'],'duplicate_of':seen[fingerprint]} for i in range(len(obj['questions'])))
        file_checks.append({'book':book,'status':'duplicate','duplicate_of':seen[fingerprint]});continue
    seen[fingerprint]=book
    if obj['novel_id']!=book:raise ValueError(f'Unexplained file/novel ID mismatch: {book}')
    novel_files=list((raw/'novel_data_zh').glob(f'{book}-*.txt'))
    if len(novel_files)!=1:raise ValueError(f'Expected exactly one source novel for {book}')
    text=novel_files[0].read_text(encoding='utf8')
    if '\ufffd' in text:raise ValueError(f'Corrupted UTF-8 content in book {book}')
    paras=parse_novel(text); actual=len(paras)
    books[book]=paras
    write_json(ROOT/f'data/processed/books/{book}.json',paras)
    file_checks.append({'book':book,'status':'parsed','actual_paragraphs':actual,'annotated_paragraphs':obj['num_paragraphs'],
                        'count_matches':actual==obj['num_paragraphs'],'source':str(novel_files[0].relative_to(ROOT))})
    for i,q in enumerate(obj['questions']):
        qid=f'zh-human-{book}-{i}';why=[];clues=q['clue_position'];cutoff=q['answer_position']
        if q['answer'] not in 'ABCD' or len(q['answer'])!=1:why.append('invalid_answer')
        if set(q['options'])!=set('ABCD') or not all(isinstance(v,str) and v.strip() for v in q['options'].values()):why.append('invalid_options')
        if not isinstance(q['question'],str) or not q['question'].strip():why.append('empty_question')
        if len(q['reasoning'])!=len(clues):why.append('reasoning_position_length_mismatch')
        if type(cutoff)!=int or not 1<cutoff<=actual:why.append('invalid_reveal_position')
        if any(type(x)!=int or (x!=-1 and not 1<=x<=actual) for x in clues):why.append('invalid_clue_position')
        if type(cutoff)==int and any(type(x)==int and x>=cutoff for x in clues):why.append('clue_at_or_after_reveal')
        gold=sorted(set(x for x in clues if type(x)==int and x>0))
        if not gold:why.append('no_explicit_clue')
        if why:excluded.append({'qid':qid,'reason':why});continue
        prefix=prefix_for_question(paras,cutoff)
        valid.append({'qid':qid,'book_id':book,'question':q['question'],'options':q['options'],
                      'cutoff_exclusive':cutoff,'prefix_paragraphs':len(prefix),'answer':q['answer'],
                      'gold_pids':gold,'reference_reasoning':q['reasoning'],'clue_positions':clues})

rng=random.Random(cfg['seed']);book_ids=sorted({q['book_id'] for q in valid});rng.shuffle(book_ids)
dev_books=book_ids[:cfg['development_books']];test_books=book_ids[cfg['development_books']:]
def sample(book_list,cap,seed):
    rng=random.Random(seed);queues={book:sorted([q for q in valid if q['book_id']==book],key=lambda q:q['qid']) for book in book_list}
    for queue in queues.values():rng.shuffle(queue)
    selected=[]
    while len(selected)<cap:
        added=False
        for book in book_list:
            if queues[book] and len(selected)<cap:selected.append(queues[book].pop());added=True
        if not added:break
    return selected
selected={'dev':sample(dev_books,cfg['development_question_target'],cfg['seed']+1),
          'test':sample(test_books,cfg['test_question_cap'],cfg['seed']+2)}
split={'seed':cfg['seed'],'algorithm':'numeric book sort, Python random shuffle, first five development books; round-robin questions',
       'dev_books':dev_books,'test_books':test_books,'dev_ids':[q['qid'] for q in selected['dev']],'test_ids':[q['qid'] for q in selected['test']],
       'all_valid_count':len(valid),'target_dev':cfg['development_question_target'],'target_test':cfg['test_question_cap']}
split['split_sha256']=hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()
dest=ROOT/'configs/split.json'
if dest.exists() and read_json(dest)!=split:raise RuntimeError('Existing split differs; review protocol before changing it')
write_json(dest,split)
review_queues={book:[q['qid'] for q in selected['test'] if q['book_id']==book] for book in test_books}
review_rng=random.Random(cfg['seed']+3)
for queue in review_queues.values():review_rng.shuffle(queue)
review_ids=[]
while len(review_ids)<min(cfg['human_review_questions'],len(selected['test'])):
    for book in test_books:
        if review_queues[book] and len(review_ids)<cfg['human_review_questions']:review_ids.append(review_queues[book].pop())
review={'seed':cfg['seed']+3,'question_ids':review_ids,'selected_before_test_inference':True,'ratings_completed':False}
review_path=ROOT/'configs/human_review_sample.json'
if review_path.exists() and read_json(review_path)!=review:raise RuntimeError('Existing human review sample differs')
write_json(review_path,review)
for name,rows in selected.items():
    public=[{k:q[k] for k in ['qid','book_id','question','options','cutoff_exclusive','prefix_paragraphs']} for q in rows]
    gold=[{k:q[k] for k in ['qid','answer','gold_pids','reference_reasoning','clue_positions']} for q in rows]
    write_jsonl(ROOT/('data/processed/dev_inputs.jsonl' if name=='dev' else 'data/restricted/test_inputs.jsonl'),public)
    write_jsonl(ROOT/f'data/restricted/{name}_gold.jsonl',gold)
write_jsonl(ROOT/'reports/exclusions.jsonl',excluded)
summary={'dataset_revision':cfg['dataset_revision'],'raw_annotation_files':len(file_checks),'raw_questions':raw_questions,
         'eligible_questions':len(valid),'eligible_books':len(book_ids),'exclusions':len(excluded),'file_checks':file_checks,
         'eligible_per_book':dict(collections.Counter(q['book_id'] for q in valid)),
         'dev_questions':len(selected['dev']),'test_questions':len(selected['test']),'dev_books':dev_books,'test_books':test_books,
         'source_text_checks':'UTF-8, contiguous unique paragraph IDs, existing reveal and explicit-clue paragraphs; no semantic validation claim',
         'human_semantic_audit_completed':False,'model_inference_on_test':False}
write_json(ROOT/'reports/data_audit.json',summary)
print(json.dumps({k:v for k,v in summary.items() if k not in ['file_checks','eligible_per_book']},ensure_ascii=True,indent=2))
