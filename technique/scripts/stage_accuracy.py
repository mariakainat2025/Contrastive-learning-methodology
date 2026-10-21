

def _top_k_accuracy(rows, true_key, ks=(1, 3)):
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
    if len(true_set) <= k:
        return true_set <= top_set
    return top_set <= true_set


def classify(true_set, top_set, top_n):
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
    if not rows:
        return 0.0
    n_correct = sum(
        1 for r in rows
        if r[true_key] <= {label for (label, _) in r['ranked'][:len(r[true_key])]}
    )
    return n_correct / len(rows)


def stage_top_k_accuracy(stage_rows, ks=(1, 3)):
    return _top_k_accuracy(stage_rows, 'true_stages', ks)


def stage_accuracy_at_n_true(stage_rows):
    return _accuracy_at_n_true(stage_rows, 'true_stages')


def tactic_top_k_accuracy(tactic_rows, ks=(1, 3)):
    return _top_k_accuracy(tactic_rows, 'true_tactics', ks)


def tactic_accuracy_at_n_true(tactic_rows):
    return _accuracy_at_n_true(tactic_rows, 'true_tactics')


def technique_top_k_accuracy(technique_rows, ks=(1, 3)):
    return _top_k_accuracy(technique_rows, 'true_techniques', ks)


def technique_accuracy_at_n_true(technique_rows):
    return _accuracy_at_n_true(technique_rows, 'true_techniques')
