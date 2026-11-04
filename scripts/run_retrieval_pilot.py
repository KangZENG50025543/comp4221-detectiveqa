"""Development-only inference; no answer generation, no paid API, no test inference."""
from pathlib import Path
import argparse,sys,time,gc,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from detectiveqa import ROOT,read_json,read_jsonl,write_json,write_jsonl,prefix_for_question,query_text,evidence_coverage
from retrieval import make_chunks,embed_texts,rerank_scores,chunk_fingerprint

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=3);p.add_argument('--run-name',default='dev_retrieval_pilot');args=p.parse_args()
    cfg=read_json(ROOT/'configs/protocol.json');questions=read_jsonl(ROOT/'data/processed/dev_inputs.jsonl')[:args.limit]
    if not questions:raise ValueError('No development questions')
    split=read_json(ROOT/'configs/split.json')
    if any(q['qid'] not in split['dev_ids'] for q in questions):raise AssertionError('Non-development question')
    out=ROOT/'runs'/args.run_name
    if out.exists():raise FileExistsError('Choose a new run name; existing runs are immutable')
    out.mkdir(parents=True)
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);torch.set_num_threads(4)
    start=time.perf_counter();tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/embedding',local_files_only=True)
    model=AutoModel.from_pretrained(ROOT/'models/embedding',local_files_only=True,dtype=torch.float16).to('cuda').eval()
    torch.cuda.reset_peak_memory_stats();prepared=[];stages=[]
    for q in questions:
        t=time.perf_counter();paras=read_json(ROOT/f'data/processed/books/{q["book_id"]}.json')
        prefix=prefix_for_question(paras,q['cutoff_exclusive']);chunks=make_chunks(prefix,tokenizer,cfg['chunk_max_tokens'])
        assert all(s['pid']<q['cutoff_exclusive'] for c in chunks for s in c['spans'])
        cachekey=chunk_fingerprint(chunks,cfg['embedding']['revision']);cache=ROOT/f'cache/embeddings/{cachekey}.npy';cache.parent.mkdir(parents=True,exist_ok=True)
        reused=cache.exists()
        vectors=np.load(cache) if reused else embed_texts([c['text'] for c in chunks],model,tokenizer)
        if not reused:np.save(cache,vectors)
        query=query_text(q);qvec=embed_texts([query],model,tokenizer)[0];similarities=vectors@qvec
        ranked=sorted(range(len(chunks)),key=lambda i:(-float(similarities[i]),i))[:50]
        elapsed=time.perf_counter()-t
        write_json(out/f'{q["qid"]}_chunks.json',chunks)
        prepared.append({'q':q,'chunks':chunks,'query':query,'ranked':ranked,'dense_scores':{str(i):float(similarities[i]) for i in ranked},'prefix':prefix})
        stages.append({'qid':q['qid'],'prefix_paragraphs':len(prefix),'chunks':len(chunks),'dense_seconds':elapsed,'embedding_cache_hit':reused})
        print(f'DENSE {q["qid"]}: {len(chunks)} chunks in {elapsed:.1f}s',flush=True)
    dense_peak=torch.cuda.max_memory_allocated();del model;gc.collect();torch.cuda.empty_cache()
    rerank_tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/reranker',local_files_only=True)
    reranker=AutoModelForSequenceClassification.from_pretrained(ROOT/'models/reranker',local_files_only=True,dtype=torch.float16).to('cuda').eval()
    torch.cuda.reset_peak_memory_stats();retrieved=[];score_logs=[]
    for item,stage in zip(prepared,stages):
        chunks=item['chunks'];ranked=item['ranked'];t=time.perf_counter()
        # B and C run separately so their measured reranking time is not conflated.
        order10=ranked[:10];scores10=rerank_scores(item['query'],[chunks[i]['text'] for i in order10],reranker,rerank_tokenizer)
        b_seconds=time.perf_counter()-t;t=time.perf_counter()
        scores50=rerank_scores(item['query'],[chunks[i]['text'] for i in ranked],reranker,rerank_tokenizer)
        c_seconds=time.perf_counter()-t
        best10=sorted(zip(order10,scores10),key=lambda x:(-x[1],x[0]))[:5]
        best50=sorted(zip(ranked,scores50),key=lambda x:(-x[1],x[0]))[:5]
        conditions={'A':ranked[:5],'B':[i for i,_ in best10],'C':[i for i,_ in best50]}
        stage.update(rerank_B_seconds=b_seconds,rerank_C_seconds=c_seconds)
        for name,ids in conditions.items():
            chosen=[chunks[i] for i in sorted(ids)]
            record={'qid':item['q']['qid'],'book_id':item['q']['book_id'],'condition':name,
                    'question':item['q']['question'],'options':item['q']['options'],'selected_ids':sorted(ids),
                    'evidence':chosen,'total_bge_tokens':sum(c['bge_tokens'] for c in chosen)}
            retrieved.append(record)
        score_logs.append({'qid':item['q']['qid'],'dense_top50':ranked,'dense_scores':item['dense_scores'],
                           'rerank_top10':list(zip(order10,scores10)),'rerank_top50':list(zip(ranked,scores50))})
        print(f'RERANK {item["q"]["qid"]}: B {b_seconds:.1f}s, C {c_seconds:.1f}s',flush=True)
    rerank_peak=torch.cuda.max_memory_allocated();del reranker;gc.collect();torch.cuda.empty_cache()
    write_jsonl(out/'retrieval_records.jsonl',retrieved);write_json(out/'ranking_scores.json',score_logs)
    # Gold becomes visible only here, after ranking outputs have been committed.
    gold={r['qid']:r for r in read_jsonl(ROOT/'data/restricted/dev_gold.jsonl')};metrics=[]
    byqid={item['q']['qid']:item for item in prepared}
    for record in retrieved:
        item=byqid[record['qid']]
        m=evidence_coverage(record['evidence'],gold[record['qid']]['gold_pids'],item['prefix'])
        metrics.append({'qid':record['qid'],'book_id':record['book_id'],'condition':record['condition'],**m})
    candidate_metrics=[]
    for item in prepared:
        for k in [10,50]:
            m=evidence_coverage([item['chunks'][i] for i in item['ranked'][:k]],gold[item['q']['qid']]['gold_pids'],item['prefix'])
            candidate_metrics.append({'qid':item['q']['qid'],'candidate_k':k,**m})
    aggregates={}
    for name in 'ABC':
        rows=[r for r in metrics if r['condition']==name]
        aggregates[name]={metric:sum(r[metric] for r in rows)/len(rows) for metric in ['full_paragraph_recall','mean_paragraph_character_coverage','all_explicit_evidence_covered']}
    summary={'scope':'development pipeline pilot, not held-out research results','questions':len(questions),'books':len({q['book_id'] for q in questions}),
             'test_questions_run':0,'paid_api_requests':0,'answer_accuracy':None,'elapsed_seconds':time.perf_counter()-start,
             'torch_version':torch.__version__,'precision':'float16 weights, float32 normalized vectors and scores','cuda':torch.version.cuda,
             'embedding_peak_allocated_bytes':dense_peak,'reranker_peak_allocated_bytes':rerank_peak,
             'protocol_sha256':hashlib.sha256((ROOT/'configs/protocol.json').read_bytes()).hexdigest(),'split_sha256':split['split_sha256'],
             'aggregate_evidence_metrics':aggregates,'stages':stages}
    write_json(out/'summary.json',summary);write_jsonl(out/'evidence_metrics.jsonl',metrics);write_json(out/'candidate_coverage.json',candidate_metrics)
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':main()
