"""Run frozen-input development retrieval. Never generate answers or invoke APIs.

Raw contexts and rankings stay in the local run directory. MuSiQue supports are
read only after the retrieval outputs and their integrity lock are written.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from detectiveqa import ROOT, read_json, read_jsonl, write_json, write_jsonl
from retrieval import chunk_fingerprint, embed_texts, rerank_scores
from cross_dataset import ADAPTERS, validate_input, ordinary_query, chunks_for, coverage, retention, support_budget, require_frozen_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='cross_dataset_dev_20261111_01')
    parser.add_argument('--prepared', help='Relative directory containing frozen inputs.jsonl, gold.jsonl, manifest.json')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        raise ValueError('Run name must be a single simple directory name')
    run = ROOT / 'runs' / args.run_name
    prepared = (ROOT / args.prepared).resolve() if args.prepared else run / 'prepared'
    if not prepared.is_relative_to(run.resolve()):
        raise ValueError('Prepared data must be inside this run directory')
    input_path = prepared / 'inputs.jsonl'
    gold_path = prepared / 'gold.jsonl'
    manifest_path = prepared / 'manifest.json'
    manifest = read_json(manifest_path)
    manifest_sha = sha(manifest_path)
    if manifest.get('source_split', manifest.get('split')) != 'dev':
        raise ValueError('Frozen manifest must explicitly identify development data')
    require_frozen_hash(manifest, 'inputs.jsonl', sha(input_path))
    questions = read_jsonl(input_path)
    if not questions or len({q['qid'] for q in questions}) != len(questions):
        raise ValueError('Empty or duplicate-ID input')
    for q in questions:
        validate_input(q)
    if manifest.get('sample_ids') != [q['qid'] for q in questions]:
        raise ValueError('Inputs disagree with the frozen selection manifest')
    cfg_path = ROOT / 'configs/protocol.json'
    require_frozen_hash(manifest, 'protocol.json', sha(cfg_path))
    cfg = read_json(cfg_path)
    if cfg['chunk_max_tokens'] != 512 or cfg['final_chunks'] != 5:
        raise ValueError('This pilot was designed for a 512-token cap and 5 final chunks')
    out = run / 'retrieval'
    out.mkdir(exist_ok=False)
    write_json(out / 'run_start.json', {
        'created_utc': utc(), 'status': 'started',
        'scope': 'development retrieval pilot, not held-out evaluation or answer-model benchmark',
        'inputs_sha256': sha(input_path), 'manifest_sha256': sha(manifest_path),
        'protocol_sha256': sha(cfg_path), 'runner_sha256': sha(__file__),
        'adapter_code_sha256': sha(ROOT / 'src/cross_dataset.py'),
        'retrieval_code_sha256': sha(ROOT / 'src/retrieval.py'),
        'input_count': len(questions), 'source_split': 'dev', 'adapters': ADAPTERS,
        'model_specs': {k: cfg[k] for k in ('embedding', 'reranker')},
        'conditions': {'A': 'dense top min(5,N)', 'B': 'dense top min(10,N), rerank to min(5,N)',
                       'C': 'dense top min(50,N), rerank to min(5,N)'},
        'gold_policy': 'Only read after saved retrieval records, ranking scores and integrity lock',
    })
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.set_num_threads(4)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'models/embedding', local_files_only=True)
    model = AutoModel.from_pretrained(ROOT / 'models/embedding', local_files_only=True,
                                     dtype=torch.float16).to('cuda').eval()
    torch.cuda.reset_peak_memory_stats()
    items = []
    stages = []
    for n, q in enumerate(questions, 1):
        t = time.perf_counter()
        chunks = chunks_for(q, tokenizer, cfg['chunk_max_tokens'])
        if not chunks:
            raise ValueError('No chunks')
        cache_key = chunk_fingerprint(chunks, cfg['embedding']['revision'])
        cache = ROOT / 'cache/embeddings' / (cache_key + '.npy')
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache_hit = cache.exists()
        vectors = np.load(cache, allow_pickle=False) if cache_hit else embed_texts([c['text'] for c in chunks], model, tokenizer)
        if not cache_hit:
            np.save(cache, vectors)
        if len(vectors) != len(chunks) or not np.isfinite(vectors).all():
            raise ValueError('Invalid cached embedding vectors')
        query = ordinary_query(q)
        qvec = embed_texts([query], model, tokenizer)[0]
        dense = vectors @ qvec
        ranked = sorted(range(len(chunks)), key=lambda i: (-float(dense[i]), i))[:50]
        write_json(out / f'chunks_{n:03}.json', chunks)
        items.append({'q': q, 'chunks': chunks, 'query': query, 'ranked': ranked,
                      'dense_scores': {str(i): float(dense[i]) for i in ranked}})
        stages.append({'qid': q['qid'], 'dataset': q['dataset'], 'group_id': q['group_id'],
                       'source_paragraphs': len(q['paragraphs']), 'chunk_count': len(chunks),
                       'dense_seconds': time.perf_counter() - t, 'embedding_cache_hit': cache_hit})
        print(f'DENSE {n}/{len(questions)} {q["qid"]}: {len(chunks)} chunks', flush=True)
    dense_peak = torch.cuda.max_memory_allocated()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    rt = AutoTokenizer.from_pretrained(ROOT / 'models/reranker', local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(ROOT / 'models/reranker',
              local_files_only=True, dtype=torch.float16).to('cuda').eval()
    torch.cuda.reset_peak_memory_stats()
    records = []
    rankings = []
    for item, stage in zip(items, stages):
        q = item['q']; chunks = item['chunks']; ranked = item['ranked']
        candidate_ids = {'A': ranked[:5], 'B': ranked[:10], 'C': ranked}
        selections = {'A': ranked[:5]}
        rscores = {}
        for condition in ('B', 'C'):
            t = time.perf_counter()
            ids = candidate_ids[condition]
            scores = rerank_scores(item['query'], [chunks[i]['text'] for i in ids], model, rt)
            stage[f'rerank_{condition}_seconds'] = time.perf_counter() - t
            ordered = sorted(zip(ids, scores), key=lambda pair: (-pair[1], pair[0]))
            selections[condition] = [i for i, _ in ordered[:5]]
            rscores[condition] = list(zip(ids, scores))
        item['candidate_ids'] = candidate_ids
        item['selections'] = selections
        for condition in 'ABC':
            ids = sorted(selections[condition])
            selected = [chunks[i] for i in ids]
            records.append({'qid': q['qid'], 'dataset': q['dataset'], 'group_id': q['group_id'],
                'condition': condition, 'source_split': 'dev', 'question': q['question'], 'options': q['options'],
                'candidate_count': len(candidate_ids[condition]), 'selected_count': len(ids),
                'available_chunk_count': len(chunks), 'selected_ids': ids, 'evidence': selected,
                'total_bge_tokens': sum(c['bge_tokens'] for c in selected)})
        rankings.append({'qid': q['qid'], 'dataset': q['dataset'], 'candidate_ids': candidate_ids,
                         'dense_scores': item['dense_scores'], 'rerank_scores': rscores})
        print(f'RERANK {q["qid"]}: B pool {len(candidate_ids["B"])}, C pool {len(candidate_ids["C"])}', flush=True)
    rerank_peak = torch.cuda.max_memory_allocated()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    # Commit all model-selected evidence before reading labels.
    write_jsonl(out / 'retrieval_records.jsonl', records)
    write_json(out / 'ranking_scores.json', rankings)
    write_json(out / 'retrieval.lock.json', {'locked_utc': utc(),
        'retrieval_records_sha256': sha(out / 'retrieval_records.jsonl'),
        'ranking_scores_sha256': sha(out / 'ranking_scores.json'),
        'inputs_sha256': sha(input_path), 'question_count': len(questions)})
    require_frozen_hash(manifest, 'inputs.jsonl', sha(input_path))
    require_frozen_hash(manifest, 'protocol.json', sha(cfg_path))
    if sha(manifest_path) != manifest_sha:
        raise ValueError('Selection manifest changed during retrieval')
    gold_sha = sha(gold_path)
    require_frozen_hash(manifest, 'gold.jsonl', gold_sha)
    gold_rows = read_jsonl(gold_path)
    if len(gold_rows) != len(questions) or {r['qid'] for r in gold_rows} != {q['qid'] for q in questions}:
        raise ValueError('Gold and input question sets differ')
    gold = {r['qid']: r for r in gold_rows}
    metrics = []
    candidates = []
    for item in items:
        q = item['q']; chunks = item['chunks']; pids = gold[q['qid']]['gold_pids']
        if q['dataset'] == 'quality':
            if pids is not None:
                raise ValueError('QuALITY has no native paragraph-evidence gold')
            budget = None
        else:
            if not isinstance(pids, list) or not pids:
                raise ValueError('Missing MuSiQue supports')
            budget = support_budget(chunks, pids, 5)
        for condition in 'ABC':
            final_ids = item['selections'][condition]
            pool_ids = item['candidate_ids'][condition]
            common = {'qid': q['qid'], 'dataset': q['dataset'], 'group_id': q['group_id'], 'condition': condition,
                      'selected_count': len(final_ids), 'candidate_count': len(pool_ids), 'support_budget': budget}
            if pids is None:
                fm = cm = retained = None
            else:
                fm = coverage([chunks[i] for i in final_ids], pids, q['paragraphs'])
                cm = coverage([chunks[i] for i in pool_ids], pids, q['paragraphs'])
                retained = retention(fm, cm)
            metrics.append({**common, 'evidence_metrics': fm, 'candidate_evidence_metrics': cm, 'retention': retained})
            candidates.append({**common, 'evidence_metrics': cm})
    aggregates = {}
    for dataset in sorted({q['dataset'] for q in questions}):
        dataset_records = [r for r in records if r['dataset'] == dataset]
        by_condition = {}
        for condition in 'ABC':
            subset = [m for m in metrics if m['dataset'] == dataset and m['condition'] == condition]
            output_rows = [r for r in dataset_records if r['condition'] == condition]
            by_condition[condition] = {
                'questions': len(subset), 'evidence_annotated_questions': sum(m['evidence_metrics'] is not None for m in subset),
                'mean_selected_bge_tokens': mean([r['total_bge_tokens'] for r in output_rows]),
                'candidate_count_histogram': dict(Counter(str(m['candidate_count']) for m in subset)),
                'selected_count_histogram': dict(Counter(str(m['selected_count']) for m in subset)),
                'evidence_metrics': {key: mean([m['evidence_metrics'][key] for m in subset if m['evidence_metrics']])
                    for key in ('full_paragraph_recall', 'any_overlap_paragraph_recall',
                                'mean_paragraph_character_coverage', 'all_explicit_evidence_covered')},
                'candidate_evidence_metrics': {key: mean([m['candidate_evidence_metrics'][key] for m in subset if m['candidate_evidence_metrics']])
                    for key in ('full_paragraph_recall', 'any_overlap_paragraph_recall', 'all_explicit_evidence_covered')},
                'mean_retention_given_candidates': {key: mean([m['retention'][key] for m in subset if m['retention']])
                    for key in ('full_support_retention_given_candidates', 'any_overlap_support_retention_given_candidates')},
            }
        groups = {q['group_id'] for q in questions if q['dataset'] == dataset}
        aggregates[dataset] = {'questions': len(dataset_records) // 3, 'groups': len(groups), 'conditions': by_condition}
    write_jsonl(out / 'evidence_metrics.jsonl', metrics)
    write_jsonl(out / 'candidate_coverage.jsonl', candidates)
    summary = {'status': 'completed', 'completed_utc': utc(), 'scope': 'development-only retrieval diagnostic',
               'question_count': len(questions), 'retrieval_record_count': len(records), 'test_questions_run': 0,
               'paid_api_calls': 0, 'answer_accuracy': None, 'human_ratings': None,
               'elapsed_seconds': time.perf_counter() - start, 'torch_version': torch.__version__, 'cuda': torch.version.cuda,
               'precision': 'float16 model weights, float32 normalized embedding vectors and scoring',
               'embedding_peak_allocated_bytes': dense_peak, 'reranker_peak_allocated_bytes': rerank_peak,
               'inputs_sha256': sha(input_path), 'gold_sha256': gold_sha, 'manifest_sha256': sha(manifest_path),
               'protocol_sha256': sha(cfg_path), 'adapters': ADAPTERS, 'datasets': aggregates, 'stages': stages,
               'prespecified_strata': manifest.get('strata'),
               'limitations': ['Gold evidence quality is not certified by coverage metrics.',
                   'QuALITY evidence recall is null because no native source-span gold is provided.',
                   'Actual B/C pools may coincide when fewer than 11 chunks exist; no padding with artificial distractors.',
                   'Complete paragraph coverage is stricter than semantic evidence sufficiency.',
                   'No answer-generation or human evaluation has occurred in this run.']}
    write_json(out / 'summary.json', summary)
    print(json.dumps({'completed': str(out), 'questions': len(questions), 'datasets': aggregates}, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
