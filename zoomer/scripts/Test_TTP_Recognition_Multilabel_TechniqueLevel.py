"""
Same trained ZOOMER multilabel model as Test_TTP_Recognition_Multilabel.py, but scored
and ranked directly at TECHNIQUE level (LRAP/AUPR computed over the 16 technique classes
themselves) instead of collapsing predictions up to the 9 tactics. This is a separate
script, not a modification of the original, so the tactic-level evaluation stays intact
and comparable to what's already in the results/PDF.

Built to give a technique-vs-technique comparison against the CAM-LDS technique matcher
(CAM-LDS/technique/). One scope note: ZOOMER keeps all 16 technique classes distinct here
(T1059-000 and T1059-004 are NOT merged) -- ZOOMER's per-technique prototypes come from
the Deep+Wide graph model, not from MITRE template text, so it has no reason to merge
them. The CAM-LDS technique matcher does merge those two (see TECHNIQUE_MERGE in
train_camlds_matcher_technique.py) because it uses one shared MITRE template for both.
So a technique-level comparison table will have a 16-vs-15 class mismatch on that one
pair -- worth handling explicitly (e.g. merge ZOOMER's two down for the comparison, or
footnote it) when building the joint table, not something this script resolves itself.
"""
import sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score
from Deep_Wide_Model import TSGModel
from data_utils import GraphTensorCache, instance_from_filename
from create_data_split import get_zoomer_split_multilabel
from discretize_features import ALL_DIMS, fit_bins
from cross_product import generate_masks, DEFAULT_K as CROSS_PRODUCT_K
from Train_TTP_Recognition_Multilabel import IN_DIM, checkpoint_path


def embed_graph(model, cache, path, device):
    (h, adjacency, wide_x) = cache.get(path)
    with torch.no_grad():
        return model(h.to(device), adjacency.to(device), wide_x.to(device))


def build_final_prototypes(model, cache, split, classes, device):
    prototypes = []
    for technique in classes:
        paths = [path for (_, path) in split[technique]['train']]
        embeds = torch.stack([embed_graph(model, cache, p, device) for p in paths])
        prototypes.append(embeds.mean(dim=0))
    return torch.stack(prototypes)


def print_technique_results_table(rows, n_classes, top_n=3):
    col_file = 32
    col_true = 30
    col_score = 18
    print()
    print('  -- Test Results (scored against {} technique prototypes, technique-level LRAP/AUPR) --'.format(n_classes))
    header_fmt = '  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} ' + ' '.join(['{:<' + str(col_score) + '}'] * top_n) + ' {:<8}'
    print(header_fmt.format('#', 'File', 'True techniques', *['#{} (score)'.format(j + 1) for j in range(top_n)], 'Top3'))
    print('  ' + '-' * (4 + col_file + col_true + col_score * top_n + 8 + top_n + 3))
    n_wrong = 0
    for (i, r) in enumerate(rows, 1):
        ranked = r['ranked']
        top_techniques = {t for (t, _s) in ranked[:top_n]}
        cols = []
        for j in range(top_n):
            cols.append('{} {:.4f}'.format(*ranked[j]) if j < len(ranked) else '')
        fname = r['file'] if len(r['file']) <= col_file else r['file'][:col_file - 3] + '...'
        true_str = ','.join(sorted(r['true_techniques']))
        if len(true_str) > col_true:
            true_str = true_str[:col_true - 3] + '...'
        wrong = not r['true_techniques'] & top_techniques
        if wrong:
            n_wrong += 1
        print(header_fmt.format(i, fname, true_str, *cols, 'WRONG' if wrong else ''))
    print()
    print('  Wrong (true technique not in top-3) : {}/{} samples'.format(n_wrong, len(rows)))
    return n_wrong


def main(seed, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print('[multilabel/technique-level] Seed: {}  Device: {}'.format(seed, device))
    ckpt = torch.load(checkpoint_path(seed), map_location=device)
    classes = ckpt['classes']
    model = TSGModel(deep_in_dim=IN_DIM, wide_in_dim=ckpt['wide_in_dim']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    split = get_zoomer_split_multilabel(seed)
    train_paths = sorted({path for t in classes for (_, path) in split[t]['train']})
    bins = fit_bins(train_paths)
    h_cat_dim = sum(len(bins[dim]) for dim in ALL_DIMS)
    masks = generate_masks(h_cat_dim, seed=seed)
    cache = GraphTensorCache(bins, masks)

    prototypes = build_final_prototypes(model, cache, split, classes, device)
    unique_test = {}
    for technique in classes:
        for (filename, path) in split[technique]['test']:
            inst = instance_from_filename(filename)
            entry = unique_test.setdefault(inst, {'filename': filename, 'path': path, 'true_techniques': set()})
            entry['true_techniques'].add(technique)

    n_total = 0
    y_true = []
    y_score = []
    rows = []
    for inst in sorted(unique_test):
        filename = unique_test[inst]['filename']
        path = unique_test[inst]['path']
        true_techniques = unique_test[inst]['true_techniques']
        embed = embed_graph(model, cache, path, device)
        dists = torch.norm(prototypes - embed, dim=1)
        scores_by_class = (-dists).tolist()
        n_total += 1
        y_true.append([1 if c in true_techniques else 0 for c in classes])
        y_score.append(scores_by_class)
        ranked = sorted(zip(classes, scores_by_class), key=lambda x: x[1], reverse=True)
        rows.append({'file': inst, 'true_techniques': true_techniques, 'ranked': ranked})

    if verbose:
        n_wrong_top3 = print_technique_results_table(rows, len(classes))
    else:
        n_wrong_top3 = sum(
            1 for r in rows if not r['true_techniques'] & {t for (t, _s) in r['ranked'][:3]}
        )

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    lrap = label_ranking_average_precision_score(y_true, y_score)
    valid_cols = y_true.sum(axis=0) > 0
    aupr = average_precision_score(y_true[:, valid_cols], y_score[:, valid_cols], average='macro') if valid_cols.any() else 0.0

    technique_results = [{'file': r['file'], 'true_techniques': sorted(r['true_techniques']),
                           'ranked': [{'technique': t, 'score': s} for (t, s) in r['ranked']]}
                          for r in rows]

    metrics = {'seed': seed, 'n_total': n_total, 'n_classes': len(classes),
               'lrap': lrap, 'aupr': aupr, 'n_wrong_top3': n_wrong_top3,
               'technique_results': technique_results}

    if verbose:
        print()
        print('LRAP (Label Ranking Average Precision)          : {:.1f}%'.format(lrap * 100))
        print('AUPR (Area Under Precision-Recall Curve, macro) : {:.1f}%'.format(aupr * 100))
        print('Wrong (true technique not in top-3)             : {}/{} samples'.format(n_wrong_top3, n_total))
        print()
        print('Test samples      : {}'.format(n_total))
        print('Technique classes : {}'.format(len(classes)))

    return metrics


if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
