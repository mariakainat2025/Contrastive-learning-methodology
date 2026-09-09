import json
import os

from abstract_node import HNODE_DIMS, node_abstract_counts
from edge_feature import EDGE_TYPE_DIMS, edge_type_counts
from ioc_feature import IOC_DIMS, ioc_feature_counts, load_knowledge_base
from Feature_Initialization import load_graph

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_SPLIT_PATH = os.path.join(SCRIPTS_DIR, 'data_split_output.json')
BINS_PATH = os.path.join(SCRIPTS_DIR, 'discretization_bins.json')

ALL_DIMS = HNODE_DIMS + EDGE_TYPE_DIMS + IOC_DIMS
assert len(ALL_DIMS) == 83

DEFAULT_K = 4

_KNOWLEDGE_BASE = load_knowledge_base()

def raw_vector_for_graph(G):
    node_counts = node_abstract_counts(G)
    edge_counts = edge_type_counts(G)
    ioc_counts = ioc_feature_counts(G, _KNOWLEDGE_BASE)
    merged = {}
    merged.update(node_counts)
    merged.update(edge_counts)
    merged.update(ioc_counts)
    return [merged[dim] for dim in ALL_DIMS]

def load_raw_vector(graph_path):
    with open(graph_path) as f:
        G = load_graph(json.load(f))
    return raw_vector_for_graph(G)

def _train_graph_paths():
    with open(DATA_SPLIT_PATH) as f:
        split_data = json.load(f)
    paths = set()
    for pools in split_data['single_label_techniques'].values():
        for row in pools['train']:
            paths.add(row['path'])
    return sorted(paths)

def fit_bins(train_paths, k=DEFAULT_K):
    from sklearn.cluster import KMeans
    import numpy as np

    vectors = [load_raw_vector(p) for p in train_paths]

    bins = {}
    for (i, dim) in enumerate(ALL_DIMS):
        values = np.array([v[i] for v in vectors], dtype=float).reshape(-1, 1)
        n_distinct = len(set(values.flatten().tolist()))
        this_k = min(k, n_distinct) if n_distinct > 0 else 1
        km = KMeans(n_clusters=this_k, n_init=10, random_state=0).fit(values)
        centers = sorted(round(c[0], 6) for c in km.cluster_centers_)
        bins[dim] = centers
    return bins

def save_bins(bins):
    with open(BINS_PATH, 'w') as f:
        json.dump(bins, f, indent=2)

def load_bins():
    with open(BINS_PATH) as f:
        return json.load(f)

def _bucket_index(value, sorted_centers):
    distances = [abs(value - c) for c in sorted_centers]
    return distances.index(min(distances))

def discretize_vector(raw_vector, bins):
    h_cat = []
    for (i, dim) in enumerate(ALL_DIMS):
        centers = bins[dim]
        idx = _bucket_index(raw_vector[i], centers)
        h_cat.extend(1 if j == idx else 0 for j in range(len(centers)))
    return h_cat

if __name__ == '__main__':
    print('Fitting k-means bins (k={}) per dimension on training graphs only...'.format(DEFAULT_K))
    bins = fit_bins(_train_graph_paths())
    save_bins(bins)
    print('Saved ->', BINS_PATH)
    print()

    dim_sizes = {dim: len(centers) for (dim, centers) in bins.items()}
    total_hcat_dim = sum(dim_sizes.values())
    print('h_cat total dimensionality:', total_hcat_dim, '(from 83 raw dims)')
    print()
    print('Dimensions where k had to be reduced below {} (low variance in training data):'.format(DEFAULT_K))
    for (dim, k) in dim_sizes.items():
        if k < DEFAULT_K:
            print('  {} -> k={}, centers={}'.format(dim, k, bins[dim]))
