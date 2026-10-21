import json
import os
import random

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
MASKS_PATH = os.path.join(SCRIPTS_DIR, 'cross_product_masks.json')

DEFAULT_K = 64
DEFAULT_SEED = 0
DEFAULT_P = 0.01

def generate_masks(d, k=DEFAULT_K, p=DEFAULT_P, seed=DEFAULT_SEED):
    rng = random.Random(seed)
    return [[1 if rng.random() < p else 0 for _ in range(d)] for _ in range(k)]

def save_masks(masks):
    with open(MASKS_PATH, 'w') as f:
        json.dump(masks, f)

def load_masks():
    with open(MASKS_PATH) as f:
        return json.load(f)

def cross_product_features(h_cat, masks):
    features = []
    for mask in masks:
        on = True
        for (h_i, c_i) in zip(h_cat, mask):
            if c_i and not h_i:
                on = False
                break
        features.append(1 if on else 0)
    return features

def hwide_pre_projection(h_cat, masks):
    return list(h_cat) + cross_product_features(h_cat, masks)

if __name__ == '__main__':
    from discretize_features import ALL_DIMS, load_bins, discretize_vector
    from Wide_Model import h_cat_for_graph
    from Feature_Initialization import GRAPHS_DIR
    import json as _json

    with open(os.path.join(SCRIPTS_DIR, 'data_split_output.json')) as f:
        split_data = _json.load(f)
    graph_paths = sorted({row['path']
                           for pools in split_data['single_label_techniques'].values()
                           for row in pools['train'] + pools['test']})

    bins = load_bins()
    d = sum(len(bins[dim]) for dim in ALL_DIMS)

    for p in (0.1, 0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005):
        masks = generate_masks(d, k=DEFAULT_K, p=p)
        hcats = [h_cat_for_graph(fp, os.path.join(os.path.dirname(SCRIPTS_DIR), 'cache', 'wide_features'), GRAPHS_DIR)
                 for fp in graph_paths]

        all_phis = [cross_product_features(hc, masks) for hc in hcats]
        n_constant = 0
        for k_idx in range(DEFAULT_K):
            values = {phis[k_idx] for phis in all_phis}
            if len(values) == 1:
                n_constant += 1

        avg_selected = sum(sum(m) for m in masks) / len(masks)
        print('p={:<5} avg slots/mask={:6.1f}  useless features (constant across all 116 graphs, 0 or 1): {}/{}'.format(
            p, avg_selected, n_constant, DEFAULT_K))

    print()
    print('Committing final masks with p={} (see DEFAULT_P comment for why)...'.format(DEFAULT_P))
    final_masks = generate_masks(d, k=DEFAULT_K, p=DEFAULT_P)
    save_masks(final_masks)
    print('Saved {} masks (d={}) -> {}'.format(len(final_masks), d, MASKS_PATH))
