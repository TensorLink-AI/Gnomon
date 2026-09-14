"""Undeployed display experiment: compress only wholly unsupported pair cards.

Consumes an already computed 087/088 brief; does not query a ledger, choose a
forecast, change pagination, or change any score. Not imported by the live 093
runner. Full evidence remains authoritative at its original path and hash.
"""
from copy import deepcopy
import json

WINDOWS = ('last_4_origins', 'last_12_origins', 'lifetime')


def _support(windows, name, visited=()):
    if name not in WINDOWS or name in visited:
        raise ValueError('Unresolved or cyclic window reference')
    window = windows[name]
    if 'same_as' in window:
        if set(window) - {'same_as', 'evidence_pointer'}:
            raise ValueError('Window reference also supplies independent values')
        return _support(windows, window['same_as'], (*visited, name))
    count = window.get('matched_origins')
    if type(count) is not int or count < 0:
        raise ValueError('Explicit nonnegative matched-origin count required')
    if count == 0 and (window.get('scores') != {} or window.get('n') != 0
                       or window.get('lowest_error_config_ids') != []
                       or window.get('status') != 'insufficient_evidence'):
        raise ValueError('Zero-support card contains conflicting evidence')
    return count


def compact_unsupported_pairs(review):
    """Keep supported cards verbatim; summarize only pairs with zero in ALL windows.

    pagination.shown_pairs continues to count all represented source-page pairs,
    including unsupported_pairs. Source pagination and next-call arguments are
    preserved; omitted scores are never inferred to be zero or tied.
    """
    if review.get('schema_version') not in {'agent-review-087', 'agent-review-088'}:
        raise ValueError('Expected a frozen 087/088 review brief')
    if review.get('metric') != 'rmsle' or review.get('provider_calls') != 0:
        raise ValueError('Expected a read-only RMSLE review')
    if len(review['cards']) != review['pagination']['shown_pairs']:
        raise ValueError('Source page count does not match its cards')
    kept, unsupported = [], []
    for card in review['cards']:
        windows = card['windows']
        if set(windows) != set(WINDOWS):
            raise ValueError('All three evidence windows required')
        counts = [_support(windows, name) for name in WINDOWS]
        if any(counts):
            kept.append(deepcopy(card))
        else:
            item = deepcopy({k: v for k, v in card.items() if k != 'windows'})
            if not item.get('evidence_pointer'):
                raise ValueError('Full evidence pointer required')
            item['matched_origins_all_windows'] = 0
            unsupported.append(item)
    result = deepcopy(review)
    if not unsupported:
        return result
    result['source_schema_version'] = result['schema_version']
    result['schema_version'] = 'agent-review-sparse-094'
    result['cards'] = kept
    result['unsupported_pairs'] = unsupported
    result['display'] = {
        'supported_cards': len(kept), 'unsupported_pairs': len(unsupported),
        'source_page_pairs': len(review['cards']),
        'rule': 'Only zero support in all windows is compressed; no score-based filtering.',
        'pagination': 'shown_pairs counts cards plus unsupported_pairs; next_call is unchanged.',
        'recovery': 'Resolve each evidence_pointer against the full_evidence path and hash.',
        'unsupported_meaning': 'No matched numerical comparison, not zero error or a tie.',
    }
    # A small source response may cost less than the explanatory wrapper. Keep
    # it verbatim instead of expanding it. This depends only on serialized size,
    # not scores, winners, case outcomes, or provider identities.
    def size(value):
        return len(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode())
    if size(result) >= size(review):
        return deepcopy(review)
    return result
