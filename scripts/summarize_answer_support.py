"""Join separately recorded AI support judgments to locked answer metrics."""
import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from detectiveqa import read_json,read_jsonl,write_json
from prepare_answer_diagnostic import sha
from score_answer_diagnostic import checked_protocol
from answer_diagnostic import validate_response, answer_scores, citation_checks


def verify_metrics(run, metrics):
    """Recompute scores from locked raw replies, not a freely edited metric table."""
    protocol=checked_protocol(run)
    mappings={m['item_id']:m for m in read_json(run/'condition_key.json')}
    if len(metrics)!=len(mappings) or {m['item_id'] for m in metrics}!=set(mappings):
        raise ValueError('Metric IDs differ from the frozen question/condition mapping')
    gold={g['qid']:g for name in protocol['gold_hashes'] for g in read_jsonl(ROOT/name)}
    for metric in metrics:
        meta=mappings[metric['item_id']]
        if any(metric.get(k)!=v for k,v in meta.items()):
            raise ValueError('Metric identity or selected context changed')
        item=read_json(run/'items'/f'{meta["item_id"]}.json')
        response=read_json(run/'responses'/f'{meta["item_id"]}.json')
        errors=validate_response(response,item)
        g=gold[meta['qid']]
        scored=answer_scores(response,[g['answer']]+g.get('aliases',[]),item['options'])
        expected_em=0.0 if errors else scored['exact_match']
        expected_f1=(0.0 if errors else scored['token_f1']) if meta['dataset']=='musique' else None
        if (metric['schema_valid']!=(not errors) or metric['exact_match']!=expected_em
                or metric['token_f1']!=expected_f1 or metric['citation_checks']!=citation_checks(response,item)
                or metric['answered']!=(not errors and not response['abstain'])
                or metric['abstained']!=bool(not errors and response['abstain'])):
            raise ValueError('Metrics disagree with immutable raw replies')


def main(run):
    checked_protocol(run)
    lock=read_json(run/'responses.lock.json')
    for name,expected in lock['outputs'].items():
        if expected is not None and sha(run/name)!=expected:
            raise ValueError('Answer changed')
    destination=run/'support_summary.json'
    if destination.exists():raise FileExistsError('Support analysis already recorded')
    metrics=read_jsonl(run/'answer_metrics.jsonl')
    if len(metrics)!=36 or len({r['item_id'] for r in metrics})!=36:
        raise ValueError('Need every diagnostic item')
    verify_metrics(run,metrics)
    support_values={'full','partial','none','unclear','not_applicable'}
    citation_values={'full','partial','none','unclear','no_citations'}
    expected_keys={'item_id','answer_support','explanation_support','citation_entailment','possible_question_issue','note'}
    reviews={};hashes={}
    for dataset in ('detectiveqa','musique','quality'):
        path=run/'answer_review'/f'ai_support_{dataset}.json'
        value=read_json(path);hashes[str(path.relative_to(run))]=sha(path)
        if value['reviewer_type']!='AI' or value['reference_answers_seen'] is not False or value['condition_names_seen'] is not False:
            raise ValueError('Expected separate AI review without labels or method identities')
        ids={r['item_id'] for r in metrics if r['dataset']==dataset}
        if len(value['items'])!=len(ids) or {r['item_id'] for r in value['items']}!=ids:
            raise ValueError('Incomplete or duplicated AI review')
        for row in value['items']:
            if (set(row)!=expected_keys or row['answer_support'] not in support_values
                    or row['explanation_support'] not in support_values
                    or row['citation_entailment'] not in citation_values
                    or row['possible_question_issue'] not in {'yes','no','unclear'}
                    or not isinstance(row['note'],str) or not row['note'].strip()):
                raise ValueError('Invalid review schema')
            response=read_json(run/'responses'/f'{row["item_id"]}.json')
            if (row['answer_support']=='not_applicable')!=response['abstain']:
                raise ValueError('Answer-support applicability does not match abstention')
            if (row['citation_entailment']=='no_citations')!=(not response['citations']):
                raise ValueError('Citation applicability mismatch')
            reviews[row['item_id']]=row
    joined=[{**m,'ai_support_review':reviews[m['item_id']]} for m in metrics]
    datasets={}
    for dataset in ('detectiveqa','musique','quality'):
        datasets[dataset]={}
        for condition in 'ACN':
            rows=[r for r in joined if r['dataset']==dataset and r['condition']==condition]
            correct=[r for r in rows if r['exact_match']==1]
            datasets[dataset][condition]={'n':len(rows),'answered':sum(r['answered'] for r in rows),'original_label_correct':len(correct),
                'ai_answer_support':dict(Counter(r['ai_support_review']['answer_support'] for r in rows)),
                'ai_citation_entailment':dict(Counter(r['ai_support_review']['citation_entailment'] for r in rows)),
                'correct_and_ai_full_answer_support':sum(r['ai_support_review']['answer_support']=='full' for r in correct),
                'correct_but_ai_not_full_answer_support':sum(r['ai_support_review']['answer_support']!='full' for r in correct)}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Single-pass AI support triage, not human ground truth or a validated semantic metric',
        'review_process':'One new no-history agent per dataset; only masked A/C/N inputs, actual responses and rubric. Reference labels, correctness, method identities and answerer sufficiency self-ratings withheld. No-evidence arm remains visually apparent.',
        'response_lock_sha256':sha(run/'responses.lock.json'),'answer_metrics_sha256':sha(run/'answer_metrics.jsonl'),
        'review_hashes':hashes,'rubric_sha256':sha(run/'answer_review/rubric.json'),
        'datasets':datasets,'items':joined,'human_ratings':None,
        'limitations':['Same model family answered and reviewed, so correlated mistakes remain possible.',
            'One AI reviewer per dataset, no inter-rater reliability or human agreement measured.',
            'Strict support threshold can flag plausible inference or best-choice selection as partial; students must calibrate this judgment.',
            'Possible question issues are tentative observations on supplied evidence, not corrected labels or removal criteria.',
            'These post-answer findings do not change any original label, answer, score, source split or earlier protocol.']}
    write_json(destination,result)
    print(__import__('json').dumps({'datasets':datasets,'items':len(joined),'human_ratings':None},ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-name',default='answer_diagnostic_dev_20261111_01');args=ap.parse_args()
    if Path(args.run_name).name!=args.run_name or args.run_name in {'.','..'}:raise ValueError('Invalid run name')
    main(ROOT/'runs'/args.run_name)
