import torch
import torch.nn as nn
import torch.nn.functional as F
from Feature_Initialization import load_graph, node_features

def _scatter_softmax(scores, index, n):
    max_per_node = torch.full((n,), float('-inf'), device=scores.device, dtype=scores.dtype)
    max_per_node.scatter_reduce_(0, index, scores, reduce='amax', include_self=True)
    shifted = (scores - max_per_node[index]).exp()
    sum_per_node = torch.zeros(n, device=scores.device, dtype=scores.dtype).index_add_(0, index, shifted)
    return shifted / sum_per_node[index]

class GATLayer(nn.Module):

    def __init__(self, in_dim, out_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.attn_fc = nn.ModuleList((nn.Linear(2 * in_dim, 1) for _ in range(n_heads)))
        self.head_fc = nn.ModuleList((nn.Linear(in_dim, out_dim) for _ in range(n_heads)))

    def forward(self, h, adjacency):
        n = h.size(0)
        (dst, src) = adjacency.nonzero(as_tuple=True)
        h_i = h[dst]
        h_j = h[src]
        pair = torch.cat([h_i, h_j], dim=-1)
        heads = []
        for (attn_fc, head_fc) in zip(self.attn_fc, self.head_fc):
            e = F.relu(attn_fc(pair)).squeeze(-1)
            alpha = _scatter_softmax(e, dst, n)
            weighted = alpha.unsqueeze(-1) * h_j
            agg = torch.zeros(n, h_j.size(-1), device=h.device, dtype=h.dtype).index_add_(0, dst, weighted)
            heads.append(F.elu(head_fc(agg)))
        return torch.cat(heads, dim=-1)

class DeepModel(nn.Module):

    def __init__(self, in_dim, hidden_dim=32, n_layers=2, n_heads=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_dim, hidden_dim, n_heads))
        for _ in range(n_layers - 1):
            self.layers.append(GATLayer(hidden_dim * n_heads, hidden_dim, n_heads))

    def forward(self, h, adjacency):
        for layer in self.layers:
            h = layer(h, adjacency)
        return h.mean(dim=0)

def graph_to_tensors(G):
    feats = node_features(G)
    node_ids = list(feats)
    index = {n: i for (i, n) in enumerate(node_ids)}
    n = len(node_ids)
    h = torch.tensor([feats[n_id] for n_id in node_ids], dtype=torch.float32)
    adjacency = torch.eye(n, dtype=torch.bool)
    for (u, v) in G.edges():
        if u in index and v in index:
            (i, j) = (index[u], index[v])
            adjacency[i, j] = True
            adjacency[j, i] = True
    return (h, adjacency, node_ids)
if __name__ == '__main__':
    import json
    fp = '/csse/research/contructive-learning/CAM-LDS/graphs/persistence/T1078-002/graph_3_ssh_apt-1_reposerver.json'
    with open(fp) as f:
        G = load_graph(json.load(f))
    (h, adjacency, node_ids) = graph_to_tensors(G)
    print('nodes:', len(node_ids), '| h:', tuple(h.shape), '| adjacency:', tuple(adjacency.shape))
    model = DeepModel(in_dim=h.shape[1])
    h_deep = model(h, adjacency)
    print('h_deep:', tuple(h_deep.shape))
