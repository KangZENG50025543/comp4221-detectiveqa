"""Freeze development-only inputs; keep reference answers/supports in a separate file."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 'COMP4221-cross-dev-20261111-4221'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', default='cross_dataset_dev_20261111_01')
    parser.add_argument('--scout-root', type=Path, default=ROOT / 'dataset_scout_20261013')
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in ('.', '..'):
        raise ValueError('run-name must be a single directory name')
    base = args.scout_root.resolve()
    out = ROOT / 'runs' / args.run_name / 'prepared'
    if out.exists():
        raise FileExistsError('Preparation is immutable; choose a new run name')
    exposure = read(base / 'exposure_registry.json')['candidate_samples']
    old_ids = {r['id'] for r in exposure}
    old_articles = {r['article_id'] for r in exposure if r.get('article_id')}
    old_components = {x for r in exposure for x in r.get('musique_component_ids', [])}
    rank = lambda ds, qid: hashlib.sha256(f'{SEED}:{ds}:{qid}'.encode()).hexdigest()
    inputs, gold, selection = [], [], []
    sources = [base / 'raw/musique/musique_ans_v1.0_dev.jsonl',
               base / 'raw/quality/QuALITY.v1.0.1.htmlstripped.dev']
    musique = [json.loads(s) for s in sources[0].read_text(encoding='utf8').splitlines() if s.strip()]
    eligible = [r for r in musique if 'musique:' + r['id'] not in old_ids
                and not old_components.intersection(step['id'] for step in r['question_decomposition'])]
    for hop in (2, 3, 4):
        chosen = sorted((r for r in eligible if r['id'].startswith(f'{hop}hop')), key=lambda r: rank('musique', r['id']))[:8]
        assert len(chosen) == 8
        for q in chosen:
            qid = 'musique:' + q['id']
            paras = [{'pid': p['idx'], 'text': p['title'] + '\n' + p['paragraph_text']} for p in q['paragraphs']]
            inputs.append({'qid': qid, 'dataset': 'musique', 'source_split': 'dev', 'question': q['question'],
                           'options': None, 'group_id': q['id'], 'paragraphs': paras})
            gold.append({'qid': qid, 'gold_pids': [p['idx'] for p in q['paragraphs'] if p['is_supporting']],
                         'answer': q['answer'], 'aliases': q['answer_aliases']})
            selection.append({'qid': qid, 'hops': hop, 'component_ids': [x['id'] for x in q['question_decomposition']]})
    rows = [json.loads(s) for s in sources[1].read_text(encoding='utf8').splitlines() if s.strip()]
    pairs = [(a, q) for a in rows if a['article_id'] not in old_articles for q in a['questions']
             if 'quality:' + q['question_unique_id'] not in old_ids]
    used = set()
    for difficult in (1, 0):
        added = 0
        for article, q in sorted((x for x in pairs if x[1]['difficult'] == difficult), key=lambda x: rank('quality', x[1]['question_unique_id'])):
            aid = article['article_id']
            if aid in used:
                continue
            used.add(aid)
            qid = 'quality:' + q['question_unique_id']
            paras = [{'pid': i + 1, 'text': text} for i, text in enumerate(s.strip() for s in article['article'].splitlines() if s.strip())]
            inputs.append({'qid': qid, 'dataset': 'quality', 'source_split': 'dev', 'question': q['question'],
                           'options': dict(zip('ABCD', q['options'])), 'group_id': aid, 'paragraphs': paras})
            gold.append({'qid': qid, 'gold_pids': None, 'answer': 'ABCD'[q['gold_label'] - 1]})
            selection.append({'qid': qid, 'article_id': aid, 'hard': bool(difficult)})
            added += 1
            if added == 6:
                break
        assert added == 6
    assert len(inputs) == 36 and len({r['qid'] for r in inputs}) == 36
    out.mkdir(parents=True)
    for name, content in [('inputs.jsonl', inputs), ('gold.jsonl', gold)]:
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in content), encoding='utf8')
    manifest = {
        'prepared_utc': datetime.now(timezone.utc).isoformat(), 'source_split': 'dev', 'seed': SEED,
        'selection_rule': 'SHA256(seed:dataset:source_id), take fixed 8 per 2/3/4-hop and 6 per hard/non-hard; one QuALITY question per article; no performance filtering',
        'counts': {'musique': 24, 'quality': 12}, 'sample_ids': [q['qid'] for q in inputs], 'strata': selection,
        'excluded_prior_question_ids': sorted(old_ids), 'excluded_quality_article_ids': sorted(old_articles),
        'excluded_musique_component_ids': sorted(old_components),
        'chunk_adapters': {'musique': 'title+body; preserve distinct document boundaries, split only over 512 tokens',
                           'quality': 'pack nonempty article lines in original order into <=512-token chunks; local pid, not gold evidence'},
        'conditions': {'A': 'dense top5', 'B': 'dense min(10,N), rerank top5', 'C': 'dense min(50,N), rerank top5'},
        'output_order': 'selected chunks in original source order', 'budget': 'five chunks; actual token totals recorded, not token-matched across datasets',
        'evidence_metrics': 'MuSiQue gold paragraph coverage only; QuALITY evidence metrics null pending human review',
        'answer_generation': False, 'paid_api_calls': 0, 'formal_protocol_changed': False,
        'provenance_caution': 'MuSiQue mirror identity must be checked against separate provenance report; missing confirmation limits interpretation',
        'hashes': {'inputs.jsonl': sha(out / 'inputs.jsonl'), 'gold.jsonl': sha(out / 'gold.jsonl'),
                   'protocol.json': sha(ROOT / 'configs/protocol.json')},
        'sources': [{'file': str(p.relative_to(base)), 'sha256': sha(p)} for p in sources],
        'usage': 'Development pilot only. Record exposure; do not claim fresh final validation or task-isolated causal effects.'
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'prepared': str(out), 'counts': manifest['counts'], 'inputs_sha256': manifest['hashes']['inputs.jsonl']}))


if __name__ == '__main__':
    main()
