"""Pure diagnostic checks; no file access or semantic citation judging.

answer_scores(response, gold, options=None) returns exact_match/token_f1,
schema_valid and errors. gold is a string or sequence of answer aliases.
For MC, options is a dict and both scores use exact option-key equality.
Free-text scoring uses compatible MuSiQue-style normalization, NOT a verified
official scorer. citation_checks returns per-citation checks under `citations`
and aggregate counts. Pass the actual item to validate_response before scoring
to additionally check item identity and empty-evidence consistency.
"""
import re
import string
from collections import Counter

FIELDS = {'item_id', 'answer', 'abstain', 'evidence_sufficient', 'citations', 'brief_reason'}


def validate_response(response, item):
    """Return structural/consistency errors, not semantic support judgments."""
    if not isinstance(response, dict):
        return ['response must be an object']
    errors = []
    if set(response) != FIELDS:
        errors.append('response keys must exactly match schema')
    if not isinstance(response.get('item_id'), str) or response.get('item_id') != item.get('item_id'):
        errors.append('item_id must match item')
    ans = response.get('answer')
    if ans is not None and not isinstance(ans, str):
        errors.append('answer must be string or null')
    abstain = response.get('abstain')
    if type(abstain) is not bool:
        errors.append('abstain must be boolean')
    elif (abstain and ans is not None) or (not abstain and (not isinstance(ans, str) or not ans.strip())):
        errors.append('answer and abstain inconsistent')
    options = item.get('options')
    if options is not None and ans is not None and (not isinstance(ans, str) or ans not in options):
        errors.append('answer must be an option key')
    if response.get('evidence_sufficient') not in ('yes', 'no', 'unclear'):
        errors.append('invalid evidence_sufficient')
    if not item.get('evidence') and response.get('evidence_sufficient') == 'yes':
        errors.append('empty evidence cannot be sufficient')
    if not isinstance(response.get('brief_reason'), str):
        errors.append('brief_reason must be string')
    citations = response.get('citations')
    if not isinstance(citations, list):
        errors.append('citations must be list')
    else:
        for i, c in enumerate(citations):
            if (not isinstance(c, dict) or set(c) != {'pid', 'quote'}
                    or type(c.get('pid')) is not int or not isinstance(c.get('quote'), str)
                    or not c.get('quote', '').strip()):
                errors.append(f'invalid citation {i}')
    return errors


def citation_checks(response, item):
    """Exact visible-fragment checks only; never imply semantic support."""
    fragments = [f for c in item.get('evidence', []) for f in c.get('fragments', [])]
    citations = response.get('citations', []) if isinstance(response, dict) else []
    citations = citations if isinstance(citations, list) else []
    checks = []
    for c in citations:
        valid = isinstance(c, dict) and type(c.get('pid')) is int
        matches = [f for f in fragments if valid and f.get('pid') == c['pid']]
        quote = c.get('quote') if isinstance(c, dict) else None
        checks.append({'pid_present': bool(matches), 'quote_in_visible_fragment':
                       isinstance(quote, str) and bool(quote.strip()) and any(quote in f['text'] for f in matches)})
    return {'citations': checks, 'citation_count': len(checks),
            'invalid_pid_count': sum(not c['pid_present'] for c in checks),
            'invalid_quote_count': sum(not c['quote_in_visible_fragment'] for c in checks)}


def normalize_answer(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def answer_scores(response, gold, options=None):
    """Validate schema before awarding credit; callers validate actual item too."""
    # Evidence is unknown here: a placeholder prevents empty-evidence inference.
    item = {'item_id': response.get('item_id') if isinstance(response, dict) else None,
            'options': options, 'evidence': [{}]}
    errors = validate_response(response, item)
    result = {'exact_match': 0.0, 'token_f1': 0.0, 'schema_valid': not errors, 'errors': errors}
    if errors or response['abstain']:
        return result
    aliases = [gold] if isinstance(gold, str) else list(gold)
    if options is not None:
        score = float(response['answer'] in aliases)
        result.update(exact_match=score, token_f1=score)
        return result
    pred = normalize_answer(response['answer'])
    for alias in aliases:
        target = normalize_answer(alias)
        a, b = pred.split(), target.split()
        overlap = sum((Counter(a) & Counter(b)).values())
        f1 = (2 * overlap / (len(a) + len(b))) if a and b else float(a == b)
        result['exact_match'] = max(result['exact_match'], float(pred == target))
        result['token_f1'] = max(result['token_f1'], f1)
    return result
