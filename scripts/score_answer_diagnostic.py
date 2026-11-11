"""Lock raw isolated-agent answers, then score; never execute an answer API."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from detectiveqa import read_json, read_jsonl, write_json, write_jsonl
from prepare_answer_diagnostic import sha
from answer_diagnostic import validate_response, citation_checks, answer_scores


def utc():
    return datetime.now(timezone.utc).isoformat()


def checked_protocol(run):
    protocol = read_json(run / 'protocol.json')
    if sha(run / 'protocol.json') != read_json(run / 'input.lock.json')['protocol_sha256']:
        raise ValueError('Frozen protocol changed')
    for section, base in [('input_hashes', ROOT), ('gold_hashes', ROOT), ('prepared_hashes', run)]:
        for name, expected in protocol[section].items():
            if sha(base / name) != expected:
                raise ValueError('Frozen file changed: ' + name)
    return protocol


def lock(run):
    protocol = checked_protocol(run)
    dest = run / 'responses.lock.json'
    if dest.exists():
        raise FileExistsError('Responses already locked')
    expected = {r['item_id'] for r in read_json(run / 'condition_key.json')}
    actual = {p.stem for p in (run / 'responses').glob('*.json')}
    if actual - expected:
        raise ValueError(f'Unexpected outputs: {sorted(actual-expected)}')
    # The registry is provenance, not an inference that unknown runtime parameters were controlled.
    registry = run / 'agent_registry.json'
    if not registry.exists():
        raise ValueError('Execution registry required')
    attempts = read_json(registry)
    if (not isinstance(attempts, list) or len(attempts) != len(expected)
            or {a.get('item_id') for a in attempts} != expected
            or any(not a.get('completed_utc') for a in attempts)):
        raise ValueError('Each item needs exactly one completed attempt before locking')
    if any(a.get('output_exists') != (a['item_id'] in actual) for a in attempts):
        raise ValueError('Registry does not match response files')
    codepaths = [Path(__file__), ROOT/'src/answer_diagnostic.py']
    write_json(dest, {'locked_utc': utc(), 'protocol_sha256': sha(run/'protocol.json'),
        'outputs': {f'responses/{iid}.json': sha(run/'responses'/f'{iid}.json') if iid in actual else None for iid in sorted(expected)},
        'execution_registry_sha256': sha(registry), 'count': len(expected), 'present_count': len(actual),
        'scoring_code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in codepaths},
        'gold_parsed_by_this_script': False, 'missing_outputs': sorted(expected-actual),
        'note': 'Original responses are immutable. Scoring parses reference labels only in the separate score action.'})
    print(json.dumps({'locked_outputs': len(actual), 'missing_outputs': len(expected-actual), 'gold_parsed': False}))


def score(run):
    protocol = checked_protocol(run)
    frozen = read_json(run/'responses.lock.json')
    if (run/'summary.json').exists():
        raise FileExistsError('Scoring output already exists; do not overwrite a completed diagnostic')
    if sha(run/'protocol.json') != frozen['protocol_sha256'] or sha(run/'agent_registry.json') != frozen['execution_registry_sha256']:
        raise ValueError('Protocol or execution registry changed after response lock')
    for name, expected in frozen['outputs'].items():
        if expected is None:
            if (run/name).exists():
                raise ValueError('A missing response was filled after lock: ' + name)
        elif sha(run/name) != expected:
            raise ValueError('Raw answer changed: ' + name)
    for name, expected in frozen['scoring_code_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Scoring code changed after response lock')
    # First reference-label parse in this execution pipeline: strictly after verifying response lock.
    gold = {}
    for name in protocol['gold_hashes']:
        for row in read_jsonl(ROOT/name):
            if row['qid'] in gold:
                raise ValueError('Duplicate gold ID')
            gold[row['qid']] = row
    mappings = read_json(run/'condition_key.json')
    if len(frozen['outputs']) != len(mappings):
        raise ValueError('Incomplete output lock')
    metrics, response_view, evidence_view = [], {}, {}
    for meta in mappings:
        item = read_json(run/'items'/f'{meta["item_id"]}.json')
        try:
            response = read_json(run/'responses'/f'{meta["item_id"]}.json')
            errors = validate_response(response, item)
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
            response, errors = None, ['missing output or invalid JSON encoding/syntax']
        refs = gold[meta['qid']]
        aliases = [refs['answer']] + refs.get('aliases', [])
        values = answer_scores(response, aliases, item['options'])
        if errors:
            values['exact_match'] = values['token_f1'] = 0.0
        cited = citation_checks(response, item)
        answered = not errors and response['abstain'] is False
        response_view[meta['item_id']] = (response.get('answer'), response.get('abstain')) if isinstance(response,dict) else None
        evidence_view[meta['item_id']] = item['evidence']
        metrics.append({**meta, 'schema_valid': not errors, 'schema_errors': errors,
            'answered': answered, 'abstained': bool(not errors and response['abstain']),
            'exact_match': values['exact_match'],
            'token_f1': values['token_f1'] if meta['dataset'] == 'musique' else None,
            'self_reported_evidence_sufficient': response.get('evidence_sufficient') if isinstance(response, dict) else None,
            'citation_checks': cited, 'semantic_support': None})
    datasets = {}
    for dataset in ('detectiveqa', 'musique', 'quality'):
        conditions = {}
        for condition in 'ACN':
            rows = [r for r in metrics if r['dataset'] == dataset and r['condition'] == condition]
            n = len(rows)
            answered = [r for r in rows if r['answered']]
            count = sum(r['exact_match'] for r in rows)
            citation_count = sum(r['citation_checks']['citation_count'] for r in rows)
            conditions[condition] = {'n': n, 'correct': int(count), 'accuracy_or_em': count/n,
                'mean_token_f1': sum(r['token_f1'] for r in rows)/n if dataset == 'musique' else None,
                'answered': len(answered), 'answer_rate': len(answered)/n,
                'accuracy_or_em_among_answered': sum(r['exact_match'] for r in answered)/len(answered) if answered else None,
                'abstained': sum(r['abstained'] for r in rows), 'invalid_outputs': sum(not r['schema_valid'] for r in rows),
                'citation_count': citation_count,
                'invalid_pid_count': sum(r['citation_checks']['invalid_pid_count'] for r in rows),
                'invalid_quote_count': sum(r['citation_checks']['invalid_quote_count'] for r in rows),
                'answered_without_citations': sum(r['answered'] and r['citation_checks']['citation_count'] == 0 for r in rows),
                'self_report_sufficient_count': sum(r['self_reported_evidence_sufficient'] == 'yes' for r in rows)}
        by_qid = {}
        for row in metrics:
            if row['dataset'] == dataset:
                by_qid.setdefault(row['qid'], {})[row['condition']] = row
        delta = [r['C']['exact_match'] - r['A']['exact_match'] for r in by_qid.values()]
        same_inputs = [r for r in by_qid.values() if evidence_view[r['A']['item_id']] == evidence_view[r['C']['item_id']]]
        datasets[dataset] = {'conditions': conditions, 'paired_C_minus_A': {
            'improved': sum(d>0 for d in delta), 'tied': sum(d==0 for d in delta), 'worsened': sum(d<0 for d in delta),
            'mean_accuracy_or_em_difference': sum(delta)/len(delta),
            'identical_evidence_questions': len(same_inputs),
            'identical_evidence_different_predictions': sum(response_view[r['A']['item_id']] != response_view[r['C']['item_id']] for r in same_inputs)},
            'unique_source_groups': len({str(r['group_id']) for r in metrics if r['dataset'] == dataset})}
    write_jsonl(run/'answer_metrics.jsonl', metrics)
    summary = {'scored_utc': utc(), 'scope': 'Isolated-agent development diagnostic, not the proposed DeepSeek API benchmark or human review',
        'questions': 12, 'outputs': len(metrics), 'requested_model': protocol['requested_agent_model'],
        'backend_revision': None, 'decoding_parameters': None,
        'protocol_sha256': sha(run/'protocol.json'), 'response_lock_sha256': sha(run/'responses.lock.json'),
        'datasets': datasets, 'human_ratings': None, 'semantic_support_review': 'pending',
        'paid_api_calls': 0, 'additional_api_cost_cny': 0, 'agent_subscription_cost': None,
        'limitations': [protocol['sample_limits'], protocol['isolation_limit'],
            'One answer per item/condition, no decoding-variance estimate; exact backend revision, sampling parameters, token usage and subscription cost unavailable.',
            'No-evidence success is prior-answerability, not proof of training contamination or absence of reasoning.',
            'Self-reported sufficiency and matching citation text do not establish semantic support.',
            'Original reference answers preserved; post-answer quality findings do not alter these scores.',
            'Abstention is incorrect in the full-denominator metric; conditional accuracy is secondary.',
            'Not all instructions are enforced by the response schema; explanation length and semantic honesty need separate review.']}
    write_json(run/'summary.json', summary)
    print(json.dumps({'run': str(run), 'datasets': datasets}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['lock', 'score'])
    ap.add_argument('--run-name', default='answer_diagnostic_dev_20261111_01')
    args = ap.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {'.', '..'}:
        raise ValueError('Invalid run name')
    (lock if args.action == 'lock' else score)(ROOT/'runs'/args.run_name)
