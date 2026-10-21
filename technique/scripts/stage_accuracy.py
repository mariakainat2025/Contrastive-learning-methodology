

def _top_k_accuracy(rows, true_key, ks=(1, 3)):
    """k=1 uses the LOOSE rule: correct if the single top-ranked prediction
    (literally whichever label the ranking lists first -- ties are not
    re-examined, just whatever 'ranked[0]' already is) is ONE of the true
    labels. An exact subset match at k=1 is structurally impossible whenever
    a sample has more than 1 true label, since there's only one guess to
    work with, so top-1 needs its own looser rule to mean anything.

    k>1 uses the same precision/recall-hybrid rule as classify() in
    scoring_common.py: when a sample has no more true labels than k (room for
    full recall), correct = every true label was found among the top-k. When
    a sample has MORE true labels than k (full recall is structurally
    impossible no matter how good the model is), correct = every one of the
    k guesses was right -- i.e. no wrong/wasted slot -- even though some true
    labels inevitably got left out."""
    if not rows:
        return {k: 0.0 for k in ks}
    out = {}
    for k in ks:
        if k == 1:
            out[k] = sum(
                1 for r in rows
                if r['ranked'] and r['ranked'][0][0] in r[true_key]
            ) / len(rows)
        else:
            correct = 0
            for r in rows:
                true_set = r[true_key]
                top_set = {label for (label, _) in r['ranked'][:k]}
                if len(true_set) <= k:
                    if true_set <= top_set:
                        correct += 1
                elif top_set <= true_set:
                    correct += 1
            out[k] = correct / len(rows)
    return out


def is_correct(true_set, top_set, k):
    """Single-row version of the k>1 hybrid rule used by _top_k_accuracy:
    correct if every true label was found (when there's room, i.e.
    len(true_set) <= k), or if every one of the k guesses was right / no
    wrong guesses (when there isn't). Exposed so the live
    print_*_results_table() per-row WRONG flags can match the headline
    top-k accuracy, instead of the older any-overlap convention."""
    if len(true_set) <= k:
        return true_set <= top_set
    return top_set <= true_set


def classify(true_set, top_set, top_n):
    """correct / partial / wrong for one step at one level -- same logic as
    scoring_common.py's classify() on the CAM-LDS/PDF side."""
    total = len(true_set)
    if total == 0:
        return None
    found = len(true_set & top_set)
    if total <= top_n:
        if found == total:
            return 'correct'
        if found == 0:
            return 'wrong'
        return 'partial'
    else:
        extra_wrong = len(top_set - true_set)
        if found == 0:
            return 'wrong'
        if extra_wrong == 0:
            return 'correct'
        return 'partial'


def _accuracy_at_n_true(rows, true_key):
    """k adapts per sample instead of being fixed: a sample with n true
    labels only checks its own top-n predictions (one guess per true label
    it actually has), not a fixed top-1/top-3 for everyone. Strict: a sample
    only counts as correct if ALL n of its true labels are found among those
    n predictions -- partial overlap (some but not all) does not count."""
    if not rows:
        return 0.0
    n_correct = sum(
        1 for r in rows
        if r[true_key] <= {label for (label, _) in r['ranked'][:len(r[true_key])]}
    )
    return n_correct / len(rows)


def stage_top_k_accuracy(stage_rows, ks=(1, 3)):
    """Always computes accuracy for every k in ks (default: both top-1 and
    top-3) and returns {k: accuracy}, so callers never have to remember to
    call this twice."""
    return _top_k_accuracy(stage_rows, 'true_stages', ks)


def stage_accuracy_at_n_true(stage_rows):
    return _accuracy_at_n_true(stage_rows, 'true_stages')


def tactic_top_k_accuracy(tactic_rows, ks=(1, 3)):
    """Same as stage_top_k_accuracy, but at the tactic level."""
    return _top_k_accuracy(tactic_rows, 'true_tactics', ks)


def tactic_accuracy_at_n_true(tactic_rows):
    return _accuracy_at_n_true(tactic_rows, 'true_tactics')


def technique_top_k_accuracy(technique_rows, ks=(1, 3)):
    """Same as stage_top_k_accuracy, but at the technique level."""
    return _top_k_accuracy(technique_rows, 'true_techniques', ks)


def technique_accuracy_at_n_true(technique_rows):
    return _accuracy_at_n_true(technique_rows, 'true_techniques')
