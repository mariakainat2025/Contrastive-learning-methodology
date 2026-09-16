"""
Bridges CAM-LDS matcher sequence instances to ZOOMER's existing Wide-feature pipeline
(node-abstract + edge-type + IoC counts, discretized, plus cross-product features).
Nothing here recomputes that logic -- it's a thin reuse layer over the already-working
code in CAM-LDS/zoomer/scripts/ (discretize_features.py, cross_product.py).

Every CAM-LDS matcher instance (identified by its 'step', e.g. "1_pwnkit_pam-41") is the
same underlying attack run as one of ZOOMER's graph files -- this module maps step name
-> graph file path, then produces the same wide feature vector ZOOMER's own Wide_Model.py
would compute for it. Bins/masks are refit fresh per seed from that seed's own training
set, matching how Train_TTP_Recognition_Multilabel.py does it (not a stale shared cache).
"""
import glob
import os
import sys

ZOOMER_SCRIPTS = '/csse/research/contructive-learning/CAM-LDS/zoomer/scripts'
if ZOOMER_SCRIPTS not in sys.path:
    sys.path.insert(0, ZOOMER_SCRIPTS)

from data_utils import instance_from_filename
from discretize_features import ALL_DIMS, fit_bins, load_raw_vector, discretize_vector, _bucket_index
from cross_product import generate_masks, hwide_pre_projection, DEFAULT_K as CROSS_PRODUCT_K
from abstract_node import HNODE_DIMS, node_abstract_counts
from edge_feature import EDGE_TYPE_DIMS, edge_type_counts
from Feature_Initialization import load_graph

GRAPHS_DIR = '/csse/research/contructive-learning/CAM-LDS/graphs'

NODE_EDGE_DIMS = HNODE_DIMS + EDGE_TYPE_DIMS


def raw_node_edge_vector(graph_path):
    import json
    with open(graph_path) as f:
        G = load_graph(json.load(f))
    node_counts = node_abstract_counts(G)
    edge_counts = edge_type_counts(G)
    merged = {}
    merged.update(node_counts)
    merged.update(edge_counts)
    return [merged[dim] for dim in NODE_EDGE_DIMS]


def fit_node_edge_bins(train_paths, k=4):
    """Same k-means-per-dimension recipe as discretize_features.fit_bins, but over
    NODE_EDGE_DIMS only -- IoC counts dropped. IoC is literally 'occurrence frequency
    of tactic-associated keywords', the most directly tactic-biased feature group, and
    a likely driver of the false-positive overfitting found in the full-wide run (two
    unrelated files getting inflated credential_access confidence on ~94 training
    examples). This isolates whether removing it fixes that."""
    from sklearn.cluster import KMeans
    import numpy as np

    vectors = [raw_node_edge_vector(p) for p in train_paths]
    bins = {}
    for (i, dim) in enumerate(NODE_EDGE_DIMS):
        values = np.array([v[i] for v in vectors], dtype=float).reshape(-1, 1)
        n_distinct = len(set(values.flatten().tolist()))
        this_k = min(k, n_distinct) if n_distinct > 0 else 1
        km = KMeans(n_clusters=this_k, n_init=10, random_state=0).fit(values)
        centers = sorted(round(c[0], 6) for c in km.cluster_centers_)
        bins[dim] = centers
    return bins


def discretize_node_edge_vector(raw_vector, bins):
    h_cat = []
    for (i, dim) in enumerate(NODE_EDGE_DIMS):
        centers = bins[dim]
        idx = _bucket_index(raw_vector[i], centers)
        h_cat.extend(1 if j == idx else 0 for j in range(len(centers)))
    return h_cat


def fit_node_edge_bins_and_masks(train_steps, step_to_path, seed):
    train_paths = sorted({step_to_path[s] for s in train_steps if s in step_to_path})
    bins = fit_node_edge_bins(train_paths)
    h_cat_dim = sum(len(bins[dim]) for dim in NODE_EDGE_DIMS)
    masks = generate_masks(h_cat_dim, seed=seed)
    wide_dim = h_cat_dim + CROSS_PRODUCT_K
    return bins, masks, wide_dim


def node_edge_vector_for_step(step, step_to_path, bins, masks):
    graph_path = step_to_path.get(step)
    if graph_path is None:
        raise KeyError('No graph file found for step {!r}'.format(step))
    raw = raw_node_edge_vector(graph_path)
    h_cat = discretize_node_edge_vector(raw, bins)
    return hwide_pre_projection(h_cat, masks)


NODE_ONLY_DIMS = HNODE_DIMS


def raw_node_only_vector(graph_path):
    import json
    with open(graph_path) as f:
        G = load_graph(json.load(f))
    node_counts = node_abstract_counts(G)
    return [node_counts[dim] for dim in NODE_ONLY_DIMS]


def fit_node_only_bins(train_paths, k=4):
    """Same recipe again, node-abstract counts only -- edge-type counts also dropped
    this time, to see whether they were adding signal or adding more of the same
    small-data overfitting risk that IoC did."""
    from sklearn.cluster import KMeans
    import numpy as np

    vectors = [raw_node_only_vector(p) for p in train_paths]
    bins = {}
    for (i, dim) in enumerate(NODE_ONLY_DIMS):
        values = np.array([v[i] for v in vectors], dtype=float).reshape(-1, 1)
        n_distinct = len(set(values.flatten().tolist()))
        this_k = min(k, n_distinct) if n_distinct > 0 else 1
        km = KMeans(n_clusters=this_k, n_init=10, random_state=0).fit(values)
        centers = sorted(round(c[0], 6) for c in km.cluster_centers_)
        bins[dim] = centers
    return bins


def discretize_node_only_vector(raw_vector, bins):
    h_cat = []
    for (i, dim) in enumerate(NODE_ONLY_DIMS):
        centers = bins[dim]
        idx = _bucket_index(raw_vector[i], centers)
        h_cat.extend(1 if j == idx else 0 for j in range(len(centers)))
    return h_cat


def fit_node_only_bins_and_masks(train_steps, step_to_path, seed):
    train_paths = sorted({step_to_path[s] for s in train_steps if s in step_to_path})
    bins = fit_node_only_bins(train_paths)
    h_cat_dim = sum(len(bins[dim]) for dim in NODE_ONLY_DIMS)
    masks = generate_masks(h_cat_dim, seed=seed)
    wide_dim = h_cat_dim + CROSS_PRODUCT_K
    return bins, masks, wide_dim


def node_only_vector_for_step(step, step_to_path, bins, masks):
    graph_path = step_to_path.get(step)
    if graph_path is None:
        raise KeyError('No graph file found for step {!r}'.format(step))
    raw = raw_node_only_vector(graph_path)
    h_cat = discretize_node_only_vector(raw, bins)
    return hwide_pre_projection(h_cat, masks)


def build_step_to_graph_path():
    step_to_path = {}
    for fp in sorted(glob.glob(os.path.join(GRAPHS_DIR, '*', '*', '*.json'))):
        inst = instance_from_filename(os.path.basename(fp))
        step_to_path.setdefault(inst, fp)
    return step_to_path


def fit_wide_bins_and_masks(train_steps, step_to_path, seed):
    """train_steps: the CAM-LDS matcher's own training instance names for this seed.
    Fits bins on exactly those instances' graphs -- same graphs, same seed's training
    split, kept in lockstep with what the text side is training on."""
    train_paths = sorted({step_to_path[s] for s in train_steps if s in step_to_path})
    bins = fit_bins(train_paths)
    h_cat_dim = sum(len(bins[dim]) for dim in ALL_DIMS)
    masks = generate_masks(h_cat_dim, seed=seed)
    wide_dim = h_cat_dim + CROSS_PRODUCT_K
    return bins, masks, wide_dim


def wide_vector_for_step(step, step_to_path, bins, masks):
    graph_path = step_to_path.get(step)
    if graph_path is None:
        raise KeyError('No graph file found for step {!r}'.format(step))
    raw = load_raw_vector(graph_path)
    h_cat = discretize_vector(raw, bins)
    return hwide_pre_projection(h_cat, masks)


if __name__ == '__main__':
    step_to_path = build_step_to_graph_path()
    print('mapped', len(step_to_path), 'instances to graph files')
    sample_steps = list(step_to_path.keys())[:20]
    bins, masks, wide_dim = fit_wide_bins_and_masks(sample_steps, step_to_path, seed=0)
    print('wide_dim =', wide_dim)
    v = wide_vector_for_step(sample_steps[0], step_to_path, bins, masks)
    print('example vector for', sample_steps[0], '-> length', len(v), 'sum', sum(v))
