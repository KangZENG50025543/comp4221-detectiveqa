"""Frozen, posthoc development control: local adjacency without new model calls."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from detectiveqa import read_json, read_jsonl, write_json, write_jsonl, prefix_for_question
from cross_dataset import coverage
from context_continuity import choose

DIAGNOSTIC_SIX = ['zh-human-124-0', 'zh-human-209-1', 'zh-human-27-0',
                  'zh-human-209-3', 'zh-human-118-0', 'zh-human-118-2']
CONDITIONS = {'C': 0, 'E_next': 1, 'E_prev': -1}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='context_continuity_dev_20261111_01')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        raise ValueError('Run name must be one simple directory name')
    out = ROOT / 'runs' / args.run_name
    if out.exists():
        raise FileExistsError('Immutable run; choose a new run name')
    dq = ROOT / 'runs/dev_retrieval_20261111_01'
    cross = ROOT / 'runs/cross_dataset_dev_20261111_01'
    tracked = {}

    def tracked_json(p):
        tracked[str(p.relative_to(ROOT))] = sha(p)
        return read_json(p)

    def tracked_lines(p):
        tracked[str(p.relative_to(ROOT))] = sha(p)
        return read_jsonl(p)

    split = tracked_json(ROOT / 'configs/split.json')
    dq_inputs = {r['qid']: r for r in tracked_lines(ROOT / 'data/processed/dev_inputs.jsonl')}
    dq_ranks = {r['qid']: r for r in tracked_json(dq / 'ranking_scores.json')}
    source_records = tracked_lines(dq / 'retrieval_records.jsonl')
    manifest = tracked_json(cross / 'prepared/manifest.json')
    if manifest.get('source_split') != 'dev':
        raise ValueError('Supplemental inputs are not development data')
    cross_inputs = tracked_lines(cross / 'prepared/inputs.jsonl')
    if sha(cross / 'prepared/inputs.jsonl') != manifest['hashes']['inputs.jsonl']:
        raise ValueError('Cross inputs changed after preparation')
    if manifest['sample_ids'] != [q['qid'] for q in cross_inputs]:
        raise ValueError('Cross input IDs differ from frozen selection')
    cross_byid = {q['qid']: q for q in cross_inputs}
    cross_index = {q['qid']: n for n, q in enumerate(cross_inputs, 1)}
    cross_ranks = {r['qid']: r for r in tracked_json(cross / 'retrieval/ranking_scores.json')}
    source_records += tracked_lines(cross / 'retrieval/retrieval_records.jsonl')
    items = []
    for record in source_records:
        if record['condition'] != 'C':
            continue
        qid = record['qid']
        if qid in dq_inputs:
            if qid not in split['dev_ids']:
                raise ValueError('Non-development DetectiveQA question')
            q = dq_inputs[qid]
            chunks = tracked_json(dq / f'{qid}_chunks.json')
            paragraphs = prefix_for_question(tracked_json(ROOT / f'data/processed/books/{q["book_id"]}.json'), q['cutoff_exclusive'])
            if any(s['pid'] >= q['cutoff_exclusive'] for c in chunks for s in c['spans']):
                raise ValueError('Source chunks crossed existing cutoff')
            pairs = dq_ranks[qid]['rerank_top50']
            dataset = 'detectiveqa'
            group = q['book_id']
        else:
            q = cross_byid[qid]
            if q.get('source_split') != 'dev':
                raise ValueError('Non-development supplemental question')
            dataset, group = q['dataset'], q['group_id']
            chunks = tracked_json(cross / f'retrieval/chunks_{cross_index[qid]:03}.json')
            paragraphs = q['paragraphs']
            pairs = cross_ranks[qid]['rerank_scores']['C']
        if choose(chunks, pairs, 0)['selected_ids'] != record['selected_ids']:
            raise ValueError('Recomputed baseline C differs from saved source output')
        items.append({'qid': qid, 'dataset': dataset, 'group_id': group, 'question': q['question'],
                      'options': q['options'], 'paragraphs': paragraphs, 'chunks': chunks,
                      'pairs': pairs, 'original_record': record})
    if len(items) != 56 or len({i['qid'] for i in items}) != len(items):
        raise ValueError('Expected all 56 source development questions without duplicates')
    tracked[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    tracked['src/context_continuity.py'] = sha(ROOT / 'src/context_continuity.py')
    tracked['src/cross_dataset.py'] = sha(ROOT / 'src/cross_dataset.py')
    tracked['src/detectiveqa.py'] = sha(ROOT / 'src/detectiveqa.py')
    out.mkdir(parents=True)
    protocol = {
        'frozen_utc': utc(), 'scope': 'Posthoc fixed-rule development experiment motivated by six-case AI audit; not held-out validation',
        'question_count': 56, 'source_split': 'dev', 'sample_ids': [i['qid'] for i in items],
        'conditions': {'C': 'Reuse original reranker top5',
                       'E_next': 'Retain top4. Replace fifth only with top1 immediate successor in same source document and same original C candidate pool, if absent from original top5; otherwise C unchanged',
                       'E_prev': 'Same fixed rule using predecessor; direction control'},
        'native_document_rule': 'MuSiQue same original paragraph pid only; other two datasets one source document per question',
        'final_budget': 'Five original chunks, no added external contexts; actual token totals may differ',
        'selection_allowlist': ['chunk source order', 'chunk spans for document identity', 'original C candidate pool and rerank scores'],
        'selection_excludes': ['gold labels', 'reference reasoning', 'audit classifications', 'per-question tuned rules'],
        'diagnostic_six_for_metric_grouping_only': DIAGNOSTIC_SIX,
        'comparison_groups': 'All data; DetectiveQA diagnostic six; other fourteen development questions (not held out)',
        'source_sha256': tracked,
        'gold_policy': 'Read only after selection records and lock are saved. Preserve original annotations and scores.',
        'answer_generation': False, 'paid_api_calls': 0,
        'interpretation': 'Annotated paragraph coverage only; no answer accuracy or reasoning claims. MuSiQue may be a no-op because native paragraphs are separate documents.'
    }
    write_json(out / 'protocol.json', protocol)
    protocol_hash = sha(out / 'protocol.json')
    records = []
    for item in items:
        for name, direction in CONDITIONS.items():
            decision = choose(item['chunks'], item['pairs'], direction, item['dataset'] == 'musique')
            evidence = [item['chunks'][i] for i in decision['selected_ids']]
            records.append({'qid': item['qid'], 'dataset': item['dataset'], 'group_id': item['group_id'],
                            'condition': name, 'question': item['question'], 'options': item['options'],
                            'candidate_count': len(item['pairs']), 'selected_count': len(evidence),
                            'selected_ids': decision['selected_ids'], 'evidence': evidence,
                            'total_bge_tokens': sum(c['bge_tokens'] for c in evidence), 'decision': decision})
    write_jsonl(out / 'retrieval_records.jsonl', records)
    write_json(out / 'retrieval.lock.json', {'locked_utc': utc(), 'protocol_sha256': protocol_hash,
                'retrieval_records_sha256': sha(out / 'retrieval_records.jsonl'), 'record_count': len(records)})
    # No support labels were parsed before the line below.
    for name, digest in tracked.items():
        if sha(ROOT / name) != digest:
            raise ValueError('Source mutated during selection: ' + name)
    if sha(out / 'protocol.json') != protocol_hash:
        raise ValueError('Frozen protocol changed during selection')
    dq_gold_path = ROOT / 'data/restricted/dev_gold.jsonl'
    cross_gold_path = cross / 'prepared/gold.jsonl'
    if sha(cross_gold_path) != manifest['hashes']['gold.jsonl']:
        raise ValueError('Supplemental gold changed since source run preparation')
    gold = {r['qid']: r for r in read_jsonl(dq_gold_path) + read_jsonl(cross_gold_path)}
    byid = {i['qid']: i for i in items}
    metrics = []
    for record in records:
        item = byid[record['qid']]
        pids = gold[record['qid']]['gold_pids']
        if item['dataset'] == 'quality':
            if pids is not None:
                raise ValueError('QuALITY gold evidence must remain null')
            metric = None
        else:
            metric = coverage(record['evidence'], pids, item['paragraphs'])
        metrics.append({'qid': item['qid'], 'dataset': item['dataset'], 'group_id': item['group_id'],
                        'condition': record['condition'], 'evidence_metrics': metric,
                        'selected_ids': record['selected_ids'], 'total_bge_tokens': record['total_bge_tokens'],
                        'changed': record['decision']['changed'], 'decision_reason': record['decision']['reason']})
    write_jsonl(out / 'evidence_metrics.jsonl', metrics)
    baseline = {m['qid']: m for m in metrics if m['condition'] == 'C'}

    def aggregate(rows):
        result = {}
        for name in CONDITIONS:
            subset = [m for m in rows if m['condition'] == name]
            scored = [m for m in subset if m['evidence_metrics'] is not None]
            delta = [m['evidence_metrics']['full_paragraph_recall'] - baseline[m['qid']]['evidence_metrics']['full_paragraph_recall'] for m in scored]
            result[name] = {'questions': len(subset), 'changed_questions': sum(m['changed'] for m in subset),
                'decision_reasons': dict(Counter(m['decision_reason'] for m in subset)),
                'mean_selected_tokens': sum(m['total_bge_tokens'] for m in subset) / len(subset),
                'mean_full_paragraph_recall': sum(m['evidence_metrics']['full_paragraph_recall'] for m in scored) / len(scored) if scored else None,
                'all_supports_covered_count': sum(m['evidence_metrics']['all_explicit_evidence_covered'] for m in scored) if scored else None,
                'vs_C_full_recall': {'improved': sum(v > 1e-12 for v in delta), 'tied': sum(abs(v) <= 1e-12 for v in delta),
                                     'worsened': sum(v < -1e-12 for v in delta)} if scored else None}
        return result

    datasets = {dataset: aggregate([m for m in metrics if m['dataset'] == dataset]) for dataset in ('detectiveqa', 'musique', 'quality')}
    diagnosis = {'six_audited': aggregate([m for m in metrics if m['qid'] in DIAGNOSTIC_SIX]),
                 'other_fourteen_development': aggregate([m for m in metrics if m['dataset'] == 'detectiveqa' and m['qid'] not in DIAGNOSTIC_SIX])}
    summary = {'completed_utc': utc(), 'scope': protocol['scope'], 'protocol_sha256': protocol_hash,
                'question_count': len(items), 'record_count': len(records), 'gold_sha256': {'detectiveqa': sha(dq_gold_path), 'cross': sha(cross_gold_path)},
                'datasets': datasets, 'detectiveqa_diagnostic_groups': diagnosis, 'test_questions_run': 0,
                'answer_accuracy': None, 'human_ratings': None, 'paid_api_calls': 0,
                'limitations': ['Posthoc development rule, not independent confirmation; all 56 questions were already used in development.',
                    'No model inference was run: rankings and chunks reused exactly.',
                    'MuSiQue separate native document boundaries make adjacency a no-op in this selected batch.',
                    'The diagnostic six and other fourteen are both development groups; no held-out generalization claim.',
                    'Original evidence annotations are unchanged; recall does not establish semantic sufficiency or unique answer validity.',
                    'Equal number of chunks is not exactly equal token budget.']}
    write_json(out / 'summary.json', summary)
    print(json.dumps({'output': str(out), 'datasets': datasets, 'diagnostic_groups': diagnosis}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
