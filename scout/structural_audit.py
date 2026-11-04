import collections, hashlib, itertools, json, pathlib, statistics

BASE = pathlib.Path(__file__).resolve().parent
RAW = BASE / 'raw'
def digest(x):
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
def lengths(texts):
    out = {}
    for unit, vals in [('characters', [len(t) for t in texts]), ('whitespace_words', [len(t.split()) for t in texts])]:
        vals.sort()
        out[unit] = {'n':len(vals), 'min':min(vals) if vals else None, 'median':statistics.median(vals) if vals else None, 'mean':statistics.mean(vals) if vals else None, 'p95':vals[int((len(vals)-1)*.95)] if vals else None, 'max':max(vals) if vals else None}
    return out
def schema(rows):
    d = collections.defaultdict(collections.Counter)
    for r in rows:
        for k,v in r.items(): d[k][type(v).__name__] += 1
    return dict(d)
def read(p):
    rows, errors = [], []
    for lineno,line in enumerate(p.open(encoding='utf-8'),1):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            recovered=[]; pos=0
            try:
                while pos<len(line):
                    while pos<len(line) and line[pos].isspace(): pos+=1
                    if pos==len(line): break
                    obj,pos=json.JSONDecoder().raw_decode(line,pos); recovered.append(obj)
                rows.extend(recovered)
            except json.JSONDecodeError: recovered=[]
            errors.append({'physical_line':lineno,'error':e.msg,'column':e.colno,'recovered_objects':len(recovered)})
    return rows,errors
def common_audit(qs, texts, ids):
    ids = [digest(i) for i in ids if i is not None]
    qhash = [digest(q.get('question')) for q in qs]
    fullhash = [digest([q.get('question'),q.get('options')]) for q in qs]
    return {'qa_count':len(qs), 'unique_context_texts':len(set(map(digest,texts))), 'missing_ids':len(qs)-len(ids), 'duplicate_id_occurrences':len(ids)-len(set(ids)), 'duplicate_question_text_occurrences':len(qhash)-len(set(qhash)), 'duplicate_question_options_occurrences':len(fullhash)-len(set(fullhash)), 'empty_questions':sum(not isinstance(q.get('question'),str) or not q['question'].strip() for q in qs), 'empty_options_lists':sum(not isinstance(q.get('options'),list) or not q['options'] for q in qs), 'empty_option_entries':sum(not isinstance(o,str) or not o.strip() for q in qs for o in q.get('options',[]) if isinstance(q.get('options'),list)), 'option_count_histogram':collections.Counter(len(q.get('options',[])) for q in qs), 'empty_contexts':sum(not t.strip() for t in texts), 'context_lengths':lengths(texts)}, {'ids':set(ids),'questions':set(qhash),'question_options':set(fullhash),'contexts':set(map(digest,texts))}

report = {'definitions':{'qa_count':'DetectBench JSONL rows; QuALITY flattened questions, not article-set rows.', 'duplicates':'Exact strings; no lowercasing or whitespace normalization. Occurrences minus unique hashes.', 'independence':'Exact unique contexts are a structural proxy, not proof of independent stories. DetectBench test-hard/test-distract are variants and excluded from base split totals.', 'length':'Unicode Python string characters and whitespace-delimited words; neither is model tokens.', 'evidence':'Only aggregate schema/nonempty checks. Free-text evidence has no standardized offsets to range-check.'}, 'datasets':{}}

for name, folder in [('detectbench', RAW/'detectbench_official/DetectBench_eng_v1.3'), ('quality',RAW/'quality')]:
    result={'splits':{},'cross_split_exact_overlap':{}}
    sets={}
    for p in sorted(folder.iterdir()):
        if not p.is_file(): continue
        rows,errors=read(p)
        split=p.stem if name=='detectbench' else p.name.rsplit('.',1)[-1]
        if name=='detectbench':
            qs=rows; texts=[r.get('context','') for r in rows]; ids=[r.get('id') for r in qs]
        else:
            qs=[q for r in rows for q in r.get('questions',[])]; texts=[r.get('article','') for r in rows]; ids=[q.get('question_unique_id') for q in qs]
        audit,sets[split]=common_audit(qs,texts,ids)
        audit['unique_context_lengths']=lengths(list(set(texts)))
        if name=='detectbench':
            full=[digest([r.get('context'),r.get('question'),r.get('options')]) for r in rows]
        else:
            full=[digest([r.get('article'),q.get('question'),q.get('options')]) for r in rows for q in r.get('questions',[])]
        audit['duplicate_complete_input_occurrences']=len(full)-len(set(full))
        sets[split]['complete_inputs']=set(full)
        audit.update({'file':str(p.relative_to(BASE)), 'container_rows':len(rows), 'jsonl_nonstandard_lines':errors,'schema':schema(rows), 'question_schema':schema(qs)})
        if name=='detectbench':
            audit['label_check']={'present':sum('answer' in q for q in qs), 'integer_type':sum(type(q.get('answer')) is int for q in qs), 'out_of_zero_based_option_range':sum(type(q.get('answer')) is not int or not 0<=q['answer']<len(q.get('options',[])) for q in qs)}
            audit['option_quality']={'records_with_empty_option':sum(any(not isinstance(o,str) or not o.strip() for o in q.get('options',[])) for q in qs),'gold_points_to_empty_option':sum(type(q.get('answer')) is int and 0<=q['answer']<len(q.get('options',[])) and (not isinstance(q['options'][q['answer']],str) or not q['options'][q['answer']].strip()) for q in qs),'records_with_exact_duplicate_options':sum(len(q.get('options',[]))!=len(set(q.get('options',[]))) for q in qs),'records_with_trimmed_duplicate_options':sum(len(q.get('options',[]))!=len(set(o.strip() if isinstance(o,str) else str(o) for o in q.get('options',[]))) for q in qs)}
            ev=[r[k] for r in rows for k in ['clue_graph','reasoning_process'] if k in r]
            audit['evidence']={'present_rows':len(ev),'dict_rows':sum(isinstance(e,dict) for e in ev),'nested_schema':schema([e for e in ev if isinstance(e,dict)]),'empty_dict_rows':sum(isinstance(e,dict) and not e for e in ev), 'empty_string_fields':sum(isinstance(v,str) and not v.strip() for e in ev if isinstance(e,dict) for v in e.values()),'offset_range_check':'not applicable: observed free-text fields'}
        else:
            audit['unique_article_ids']=len(set(r.get('article_id') for r in rows))
            audit['duplicate_set_unique_id_occurrences']=len(rows)-len(set(r.get('set_unique_id') for r in rows))
            audit['label_check']={'gold_present':sum('gold_label' in q for q in qs), 'gold_out_of_one_based_option_range':sum(type(q['gold_label']) is not int or not 1<=q['gold_label']<=len(q.get('options',[])) for q in qs if 'gold_label' in q),'writer_present':sum('writer_label' in q for q in qs),'writer_out_of_one_based_option_range':sum(type(q['writer_label']) is not int or not 1<=q['writer_label']<=len(q.get('options',[])) for q in qs if 'writer_label' in q)}
            audit['evidence']={'supporting_chunk_annotations':'no dedicated supporting paragraph/sentence/offset field in observed schema; validation metadata is not gold chunk evidence'}
            sets[split]['article_ids']=set(digest(r.get('article_id')) for r in rows)
        result['splits'][split]=audit
    for a,b in itertools.combinations(sets,2): result['cross_split_exact_overlap'][a+' vs '+b]={k:len(sets[a][k]&sets[b][k]) for k in sets[a].keys()&sets[b].keys()}
    base=['train','dev','test']
    result['base_split_totals']={'qa_rows':sum(result['splits'][s]['qa_count'] for s in base), 'unique_question_texts':len(set.union(*(sets[s]['questions'] for s in base))), 'unique_exact_contexts':len(set.union(*(sets[s]['contexts'] for s in base)))}
    report['datasets'][name]=result

report['musique']={'status':'not audited unless official extracted JSONL exists; do not inspect partially downloaded archives', 'ready_files':[str(p.relative_to(BASE)) for p in RAW.rglob('musique*dev*.jsonl')]}
mp=RAW/'musique/musique_ans_v1.0_dev.jsonl'
if mp.exists():
    rows,errors=read(mp)
    paras=[p for r in rows for p in r.get('paragraphs',[])]
    decomp=[d for r in rows for d in r.get('question_decomposition',[])]
    contexts=[digest([(p.get('title'),p.get('paragraph_text')) for p in r.get('paragraphs',[])]) for r in rows]
    ids=[r.get('id') for r in rows]
    q=[r.get('question') for r in rows]
    def concat(r): return '\n'.join(p.get('paragraph_text','') for p in r.get('paragraphs',[]))
    report['musique']={'status':'audited community mirror dev only; not byte-verified against official ZIP','source_manifest':'musique_mirror_manifest.json','file':str(mp.relative_to(BASE)),'schema':schema(rows),'paragraph_schema':schema(paras),'decomposition_schema':schema(decomp),'qa_count':len(rows),'jsonl_nonstandard_lines':errors,'duplicate_id_occurrences':len(ids)-len(set(ids)),'duplicate_question_text_occurrences':len(q)-len(set(q)),'unique_context_sequences':len(set(contexts)),'unique_paragraph_texts':len(set(p.get('paragraph_text') for p in paras)),'paragraph_count_histogram':collections.Counter(len(r.get('paragraphs',[])) for r in rows),'support_count_histogram':collections.Counter(sum(p.get('is_supporting') is True for p in r.get('paragraphs',[])) for r in rows),'decomposition_count_histogram':collections.Counter(len(r.get('question_decomposition',[])) for r in rows),'empty_questions':sum(not isinstance(t,str) or not t.strip() for t in q),'empty_answers':sum(not isinstance(r.get('answer'),str) or not r['answer'].strip() for r in rows),'empty_paragraph_texts':sum(not isinstance(p.get('paragraph_text'),str) or not p['paragraph_text'].strip() for p in paras),'answerable_true_count':sum(r.get('answerable') is True for r in rows),'context_lengths':lengths([concat(r) for r in rows]),'paragraph_lengths':lengths([p.get('paragraph_text','') for p in paras]),'support_index_out_of_range':sum(type(d.get('paragraph_support_idx')) is not int or not 0<=d['paragraph_support_idx']<len(r.get('paragraphs',[])) for r in rows for d in r.get('question_decomposition',[])),'support_index_missing_from_paragraph_idx':sum(d.get('paragraph_support_idx') not in {p.get('idx') for p in r.get('paragraphs',[])} for r in rows for d in r.get('question_decomposition',[])),'duplicate_paragraph_idx_records':sum(len(r.get('paragraphs',[]))!=len({p.get('idx') for p in r.get('paragraphs',[])}) for r in rows),'paragraph_idx_not_sequence_records':sum([p.get('idx') for p in r.get('paragraphs',[])]!=list(range(len(r.get('paragraphs',[])))) for r in rows),'support_flag_nonboolean':sum(type(p.get('is_supporting')) is not bool for p in paras),'decomposition_ref_to_non_support_flag':sum(d.get('paragraph_support_idx') in {p.get('idx') for p in r.get('paragraphs',[])} and not any(p.get('idx')==d.get('paragraph_support_idx') and p.get('is_supporting') is True for p in r.get('paragraphs',[])) for r in rows for d in r.get('question_decomposition',[]))}
(BASE/'structural_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
