import json
import torch
from Deep_Model import graph_to_tensors
from Feature_Initialization import load_graph
from discretize_features import (
    discretize_vector, raw_vector_for_graph,
    discretize_node_edge_vector, raw_node_edge_vector_for_graph,
)
from cross_product import hwide_pre_projection
import os
FOLDER_TACTIC_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'folder_tactic_map.json')
K_SHOT = 3

WIDE_FEATURE_FNS = {
    'all': (raw_vector_for_graph, discretize_vector),
    'node_edge': (raw_node_edge_vector_for_graph, discretize_node_edge_vector),
}

def wide_vector_for_graph(G, bins, masks, wide_features='all'):
    (raw_fn, discretize_fn) = WIDE_FEATURE_FNS[wide_features]
    h_cat = discretize_fn(raw_fn(G), bins)
    return hwide_pre_projection(h_cat, masks)
KNOWN_HOSTS = ('attacker', 'inetfw', 'videoserver', 'wazuh', 'reposerver', 'client', 'corpdns', 'linuxshare', 'docker-log', 'docker', 'adminpc')

def instance_from_filename(filename):
    stem = filename[len('graph_'):-len('.json')]
    for host in KNOWN_HOSTS:
        suffix = '_' + host
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem

def load_tactic_map():
    with open(FOLDER_TACTIC_MAP_PATH) as f:
        return json.load(f)

def base_technique(technique):
    return technique.split('-')[0]

def technique_tactics(tactic_map, technique):
    target = base_technique(technique)
    tactics = set()
    for (key, tacs) in tactic_map.items():
        key_technique = key.split('/', 1)[0]
        if base_technique(key_technique) == target:
            tactics.update(tacs)
    return tactics

def instance_tactics(tactic_map, technique, filename):
    instance = instance_from_filename(filename)
    key = '{}/{}'.format(technique, instance)
    if key in tactic_map:
        return set(tactic_map[key])
    return technique_tactics(tactic_map, technique)

class GraphTensorCache:

    def __init__(self, bins, masks, wide_features='all'):
        self.bins = bins
        self.masks = masks
        self.wide_features = wide_features
        self._cache = {}

    def get(self, path):
        if path not in self._cache:
            with open(path) as f:
                G = load_graph(json.load(f))
            (h, adjacency, _) = graph_to_tensors(G)
            wide_x = torch.tensor(
                wide_vector_for_graph(G, self.bins, self.masks, self.wide_features), dtype=torch.float32)
            self._cache[path] = (h, adjacency, wide_x)
        return self._cache[path]
