import os

import torch
import torch.nn as nn

from abstract_node import node_abstract_counts
from edge_feature import edge_type_counts
from ioc_feature import ioc_feature_counts, load_knowledge_base
from discretize_features import ALL_DIMS, discretize_vector, load_bins
from cross_product import load_masks, hwide_pre_projection

def cache_wide_features(graph_paths, graphs_dir, out_dir):
    import json
    from Feature_Initialization import load_graph
    knowledge_base = load_knowledge_base()
    n_ok = 0
    n_failed = 0
    n_skipped = 0
    for fp in sorted(graph_paths):
        rel = os.path.relpath(fp, graphs_dir)
        out_path = os.path.join(out_dir, os.path.dirname(rel), 'wide_' + os.path.basename(rel))
        if os.path.exists(out_path):
            n_skipped += 1
            continue
        try:
            with open(fp) as f:
                G = load_graph(json.load(f))
            node_counts = node_abstract_counts(G)
            edge_counts = edge_type_counts(G)
            ioc_counts = ioc_feature_counts(G, knowledge_base)
        except Exception as e:
            n_failed += 1
            print('  FAILED on {}: {}'.format(fp, e))
            continue
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump({'node_abstract_counts': node_counts, 'edge_type_counts': edge_counts,
                       'ioc_feature_counts': ioc_counts}, f, indent=2)
        n_ok += 1
    print('Wide features (node-abstract + edge-type + ioc) cached: {} ok, {} failed, {} skipped (already cached)'.format(n_ok, n_failed, n_skipped))
    print('Saved to                                              : {}/<tactic>/<technique>/wide_<name>.json'.format(out_dir))

def h_cat_for_graph(graph_path, wide_features_cache_dir, graphs_dir):
    import json
    rel = os.path.relpath(graph_path, graphs_dir)
    wide_path = os.path.join(wide_features_cache_dir, os.path.dirname(rel), 'wide_' + os.path.basename(rel))
    with open(wide_path) as f:
        cached = json.load(f)
    merged = {}
    merged.update(cached['node_abstract_counts'])
    merged.update(cached['edge_type_counts'])
    merged.update(cached['ioc_feature_counts'])
    raw_vector = [merged[dim] for dim in ALL_DIMS]
    bins = load_bins()
    return discretize_vector(raw_vector, bins)

def wide_input_vector(graph_path, wide_features_cache_dir, graphs_dir):
    h_cat = h_cat_for_graph(graph_path, wide_features_cache_dir, graphs_dir)
    masks = load_masks()
    return hwide_pre_projection(h_cat, masks)

class WideModel(nn.Module):

    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    import json
    from Feature_Initialization import GRAPHS_DIR

    wide_cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache', 'wide_features')

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_split_output.json')) as f:
        split_data = json.load(f)
    fp = list(split_data['single_label_techniques'].values())[0]['train'][0]['path']

    x = wide_input_vector(fp, wide_cache_dir, GRAPHS_DIR)
    print('Graph:', fp)
    print('Wide input vector length (226 h_cat + 64 cross-product):', len(x))

    model = WideModel(in_dim=len(x))
    h_wide = model(torch.tensor(x, dtype=torch.float32))
    print('h_wide shape:', tuple(h_wide.shape))
    print('h_wide:', h_wide)
