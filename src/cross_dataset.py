"""Adapters and diagnostics for the development-only cross-dataset pilot."""
from detectiveqa import evidence_coverage
from retrieval import make_chunks
import re

ADAPTERS = {
    'musique': 'Each native paragraph is a separate evidence document. Split a document at the token cap; never pack separate documents together. Input text includes the source title when supplied by preparation.',
    'quality': 'Pack the ordinary article paragraphs/lines in source order using the existing lossless chunker. Local paragraph IDs are not official evidence annotations.',
}


def require_frozen_hash(manifest, filename, actual):
    """A missing expected digest is an error, never a skipped freeze check."""
    hashes = manifest.get('hashes', {})
    expected = hashes.get(filename) if isinstance(hashes, dict) else None
    if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-fA-F]{64}', expected):
        raise ValueError(f'Missing/invalid frozen hash for {filename}')
    if expected.lower() != actual.lower():
        raise ValueError(f'Frozen hash mismatch for {filename}')


def validate_input(row):
    allowed = {'qid', 'dataset', 'question', 'options', 'group_id', 'paragraphs', 'source_split', 'source_id'}
    unexpected = set(row) - allowed
    if unexpected:
        raise ValueError(f'Non-allowlisted input fields: {sorted(unexpected)}')
    if row.get('source_split') != 'dev' or row.get('dataset') not in ADAPTERS:
        raise ValueError('Only MuSiQue/QuALITY development inputs are supported')
    if not isinstance(row.get('qid'), str) or not row['qid'] or not row.get('group_id'):
        raise ValueError('Missing stable question or group ID')
    if not isinstance(row.get('question'), str) or not row['question'].strip():
        raise ValueError('Missing question')
    options = row.get('options')
    if row['dataset'] == 'musique' and options is not None:
        raise ValueError('MuSiQue must retain its open-answer format')
    if row['dataset'] == 'quality' and (not isinstance(options, dict) or set(options) != set('ABCD')):
        raise ValueError('QuALITY must supply exactly four ordinary options')
    if options and any(not isinstance(t, str) or not t.strip() for t in options.values()):
        raise ValueError('Empty/invalid option')
    paras = row.get('paragraphs')
    if not isinstance(paras, list) or not paras:
        raise ValueError('Missing context')
    if any(set(p) != {'pid', 'text'} or not isinstance(p['pid'], int)
           or not isinstance(p['text'], str) or not p['text'] for p in paras):
        raise ValueError('Context must contain only integer pid and nonempty text')
    if len({p['pid'] for p in paras}) != len(paras):
        raise ValueError('Duplicate paragraph IDs')


def ordinary_query(row):
    """Only question and choices can enter embedding/reranking queries."""
    query = row['question']
    if row.get('options'):
        query += '\n' + '\n'.join(f'{k}. {row["options"][k]}' for k in 'ABCD')
    return query


def chunks_for(row, tokenizer, cap=512):
    if row['dataset'] == 'musique':
        chunks = []
        for para in row['paragraphs']:
            chunks.extend(make_chunks([para], tokenizer, cap))
        for i, chunk in enumerate(chunks):
            chunk['id'] = i
    else:
        chunks = make_chunks(row['paragraphs'], tokenizer, cap)
    return chunks


def coverage(chunks, pids, paragraphs):
    metric = evidence_coverage(chunks, pids, paragraphs)
    fractions = metric['per_paragraph']
    metric['any_overlap_paragraph_recall'] = sum(v > 0 for v in fractions.values()) / len(fractions)
    metric['full_support_count'] = sum(v == 1 for v in fractions.values())
    metric['any_overlap_support_count'] = sum(v > 0 for v in fractions.values())
    return metric


def support_budget(chunks, pids, final_capacity=5):
    """All chunks touching support are required for complete paragraph coverage.

    This relies on the existing lossless nonoverlapping chunker and measures
    complete annotated paragraphs, not semantic sufficiency of a smaller span.
    """
    gold = set(pids)
    required = [c['id'] for c in chunks if any(s['pid'] in gold for s in c['spans'])]
    return {'chunks_required_for_complete_support_paragraphs': len(required),
            'complete_support_paragraphs_fit_final_capacity': len(required) <= final_capacity}


def retention(final_metric, candidate_metric):
    result = {}
    for mode in ('full', 'any_overlap'):
        numerator = final_metric[f'{mode}_support_count']
        denominator = candidate_metric[f'{mode}_support_count']
        if numerator > denominator:
            raise AssertionError('Selected context contains evidence absent from candidates')
        result[f'{mode}_support_retention_given_candidates'] = numerator / denominator if denominator else None
    return result
