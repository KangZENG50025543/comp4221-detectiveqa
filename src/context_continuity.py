"""Fixed adjacency selection controls; no question IDs or labels enter the rule."""
import math


def ranked_ids(score_pairs):
    if not score_pairs or any(not isinstance(i, int) or not math.isfinite(float(v)) for i, v in score_pairs):
        raise ValueError('Missing or invalid score pairs')
    if len({i for i, _ in score_pairs}) != len(score_pairs):
        raise ValueError('Duplicate candidate IDs')
    return [i for i, _ in sorted(score_pairs, key=lambda p: (-float(p[1]), p[0]))]


def choose(chunks, score_pairs, direction, native_documents=False):
    """Retain C top4; replace only its fifth slot with top1's next/previous chunk.

    The neighbor must be in the original C candidate pool and outside original
    C top5. MuSiQue neighbors must belong to the same native paragraph document.
    Other datasets have one source document per question. No gold-aware logic.
    """
    if direction not in (-1, 0, 1):
        raise ValueError('Direction must be -1, 0 or 1')
    if [c['id'] for c in chunks] != list(range(len(chunks))):
        raise ValueError('Expected consecutive source-order chunk IDs')
    order = ranked_ids(score_pairs)
    if any(i < 0 or i >= len(chunks) for i in order):
        raise ValueError('Candidate index outside source')
    original = order[:5]
    result = {'selected_ids': sorted(original), 'changed': False, 'anchor_id': order[0],
              'added_id': None, 'removed_id': None, 'reason': 'baseline'}
    if direction == 0:
        return result
    if len(original) != 5:
        result['reason'] = 'fewer_than_five_candidates'
        return result
    neighbor = order[0] + direction
    if neighbor < 0 or neighbor >= len(chunks):
        result['reason'] = 'source_boundary'
    elif neighbor not in order:
        result['reason'] = 'neighbor_not_in_original_candidate_pool'
    elif neighbor in original:
        result['reason'] = 'neighbor_already_in_original_top5'
    elif native_documents and (
        len({s['pid'] for s in chunks[order[0]]['spans']}) != 1 or
        {s['pid'] for s in chunks[neighbor]['spans']} != {s['pid'] for s in chunks[order[0]]['spans']}
    ):
        result['reason'] = 'different_native_document'
    else:
        result.update(selected_ids=sorted(original[:4] + [neighbor]), changed=True,
                      added_id=neighbor, removed_id=original[4], reason='fixed_adjacent_replacement')
    assert len(result['selected_ids']) == len(original)
    assert set(original[:4]).issubset(result['selected_ids'])
    assert set(result['selected_ids']).issubset(order)
    return result
