import os

import torch
import torch.nn as nn

from Deep_Model import DeepModel
from Wide_Model import WideModel, wide_input_vector

class TSGModel(nn.Module):

    def __init__(self, deep_in_dim, wide_in_dim, deep_hidden=32, deep_layers=2, deep_heads=4, wide_out_dim=64):
        super().__init__()
        self.deep_model = DeepModel(deep_in_dim, hidden_dim=deep_hidden, n_layers=deep_layers, n_heads=deep_heads)
        self.wide_model = WideModel(wide_in_dim, out_dim=wide_out_dim)

    def forward(self, h, adjacency, wide_x):
        h_deep = self.deep_model(h, adjacency)
        h_wide = self.wide_model(wide_x)
        return torch.cat([h_wide, h_deep], dim=-1)

def load_tsg_inputs(graph_path, graphs_dir, deep_tensors_cache_dir, wide_features_cache_dir):
    rel = os.path.relpath(graph_path, graphs_dir)
    tensors_path = os.path.join(deep_tensors_cache_dir, os.path.dirname(rel),
                                 'tensors_' + os.path.basename(rel)[:-len('.json')] + '.pt')
    cached = torch.load(tensors_path)
    wide_x = torch.tensor(wide_input_vector(graph_path, wide_features_cache_dir, graphs_dir), dtype=torch.float32)
    return (cached['h'], cached['adjacency'], wide_x)

if __name__ == '__main__':
    import json

    SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
    ZOOMER_DIR = os.path.dirname(SCRIPTS_DIR)
    DEEP_TENSORS_CACHE_DIR = os.path.join(ZOOMER_DIR, 'cache', 'deep_tensors')
    WIDE_FEATURES_CACHE_DIR = os.path.join(ZOOMER_DIR, 'cache', 'wide_features')

    from Feature_Initialization import GRAPHS_DIR

    with open(os.path.join(SCRIPTS_DIR, 'data_split_output.json')) as f:
        split_data = json.load(f)
    fp = list(split_data['single_label_techniques'].values())[0]['train'][0]['path']

    (h, adjacency, wide_x) = load_tsg_inputs(fp, GRAPHS_DIR, DEEP_TENSORS_CACHE_DIR, WIDE_FEATURES_CACHE_DIR)
    print('Graph:', fp)
    print('deep h:', tuple(h.shape), '| adjacency:', tuple(adjacency.shape), '| wide_x:', tuple(wide_x.shape))

    model = TSGModel(deep_in_dim=h.shape[1], wide_in_dim=wide_x.shape[0])
    h_tsg = model(h, adjacency, wide_x)
    print('h_TSG shape (h_wide 64 + h_deep {}):'.format(model.deep_model.layers[-1].head_fc[0].out_features * model.deep_model.layers[-1].n_heads),
          tuple(h_tsg.shape))
    print('h_TSG:', h_tsg)
