import glob
import json
import os
import sys
from networkx.readwrite import json_graph
GRAPHS_DIR = '/csse/research/contructive-learning/CAM-LDS/graphs'
_CAM_LDS_SCRIPTS = '/csse/research/contructive-learning/CAM-LDS/scripts'
if _CAM_LDS_SCRIPTS not in sys.path:
    sys.path.insert(0, _CAM_LDS_SCRIPTS)
from parse_audit import RARE_EDGE_TYPES

def discover_edge_types(graphs_dir=GRAPHS_DIR):
    types = set()
    for fp in glob.glob(os.path.join(graphs_dir, '*', '*', '*.json')):
        with open(fp) as f:
            d = json.load(f)
        for e in d.get('links', []):
            et = e.get('edge_type')
            if et:
                types.add(et)
    return sorted(types)
ALL_EDGE_TYPES = discover_edge_types()
MASTER_EDGE_TYPES = [t for t in ALL_EDGE_TYPES if t not in RARE_EDGE_TYPES]
EVENT_PREFIX = 'EVENT_'

def _normalize(edge_type):
    if edge_type and edge_type.startswith(EVENT_PREFIX):
        return edge_type[len(EVENT_PREFIX):]
    return edge_type
EDGE_TYPE_INDEX = {_normalize(t): i for (i, t) in enumerate(MASTER_EDGE_TYPES)}
N_TYPES = len(MASTER_EDGE_TYPES)
STRUCTURAL_DIM = 2 * N_TYPES

def load_graph(graph_json):
    return json_graph.node_link_graph(graph_json)

def is_process(node_id, G):
    data = G.nodes[node_id]
    if data.get('type') != 'SUBJECT_PROCESS':
        return False
    return not data.get('name', '').startswith('syscall:')
FEATURIZABLE_TYPES = ('SUBJECT_PROCESS', 'FILE', 'NetFlowObject', 'PRINCIPAL_LOCAL')

def is_featurizable(node_id, G):
    data = G.nodes[node_id]
    t = data.get('type')
    if t == 'SUBJECT_PROCESS':
        return is_process(node_id, G)
    return t in ('FILE', 'NetFlowObject', 'PRINCIPAL_LOCAL')

def structural_features(G):
    features = {n: [0.0] * STRUCTURAL_DIM for n in G.nodes() if is_featurizable(n, G)}
    for (u, v, data) in G.edges(data=True):
        idx = EDGE_TYPE_INDEX.get(_normalize(data.get('edge_type')))
        if idx is None:
            continue
        if v in features:
            features[v][idx] += 1.0
        if u in features:
            features[u][N_TYPES + idx] += 1.0
    return features
SEMANTIC_DIM = 32

def hierarchical_substrings(s):
    if not s:
        return []
    pieces = []
    for (i, ch) in enumerate(s):
        if i > 0 and ch in ('/', ' '):
            piece = s[:i].rstrip(' /')
            if piece and piece not in pieces:
                pieces.append(piece)
    if s not in pieces:
        pieces.append(s)
    return pieces

def _delimited_hierarchy(text, sep):
    pieces = []
    for part in text.strip().split(sep):
        pieces.append(pieces[-1] + sep + part if pieces else part)
    return pieces

def _node_hierarchy(node_type, data):
    name = data.get('name') or ''
    if node_type == 'SUBJECT_PROCESS':
        text = data.get('cmd') or name
        return ('subject', hierarchical_substrings(text))
    if node_type == 'FILE':
        return ('file', _delimited_hierarchy(name, '/'))
    if node_type == 'NetFlowObject':
        ip = name.rsplit(':', 1)[0] if ':' in name else name
        hierarchy = _delimited_hierarchy(ip, '.')
        if ip != name:
            hierarchy.append(name)
        return ('netflow', hierarchy)
    if node_type == 'PRINCIPAL_LOCAL':
        return ('principal', _delimited_hierarchy(name, ':'))

def semantic_features(G):
    from sklearn.feature_extraction import FeatureHasher
    hasher = FeatureHasher(n_features=SEMANTIC_DIM, input_type='string')
    node_ids = [n for n in G.nodes() if is_featurizable(n, G)]
    char_lists = []
    for n in node_ids:
        data = G.nodes[n]
        (tag, hierarchy) = _node_hierarchy(data.get('type'), data)
        combined = tag + ''.join(hierarchy)
        char_lists.append(list(combined))
    matrix = hasher.transform(char_lists).toarray()
    return {n: matrix[i].tolist() for (i, n) in enumerate(node_ids)}

def l2_normalize(vec):
    length = sum((v * v for v in vec)) ** 0.5
    if length == 0:
        return list(vec)
    return [v / length for v in vec]

def node_features(G):
    t_feats = structural_features(G)
    s_feats = semantic_features(G)
    return {n: l2_normalize(t_feats[n]) + l2_normalize(s_feats[n]) for n in t_feats}
FEATURES_DIR = '/csse/research/contructive-learning/CAM-LDS/zoomer/features'

def _round_features(feats, ndigits=4):
    return {n: [round(v, ndigits) for v in vec] for (n, vec) in feats.items()}

def batch_main(graphs_dir=GRAPHS_DIR, out_dir=FEATURES_DIR):
    from tqdm import tqdm
    files = sorted(glob.glob(os.path.join(graphs_dir, '*', '*', '*.json')))
    print('Found {} graph files under {}'.format(len(files), graphs_dir))
    total_featurized_nodes = 0
    n_ok = 0
    n_failed = 0
    for fp in tqdm(files, desc='Building features'):
        try:
            with open(fp) as f:
                G = load_graph(json.load(f))
            feats = _round_features(node_features(G))
        except Exception as e:
            n_failed += 1
            print('  FAILED on {}: {}'.format(fp, e))
            continue
        rel = os.path.relpath(fp, graphs_dir)
        out_path = os.path.join(out_dir, os.path.dirname(rel), 'features_' + os.path.basename(rel))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(feats, f)
        total_featurized_nodes += len(feats)
        n_ok += 1
    print()
    print('Graphs processed  : {} ok, {} failed'.format(n_ok, n_failed))
    print('Featurized nodes  : {:,} total (process + file + socket + account)'.format(total_featurized_nodes))
    print('Feature dim      : {} (h_t={} + h_s={})'.format(STRUCTURAL_DIM + SEMANTIC_DIM, STRUCTURAL_DIM, SEMANTIC_DIM))
    print('Saved to         : {}/<tactic>/<technique>/features_<name>.json'.format(out_dir))
if __name__ == '__main__':
    assert len(ALL_EDGE_TYPES) == 59, len(ALL_EDGE_TYPES)
    assert len(RARE_EDGE_TYPES) == 12, len(RARE_EDGE_TYPES)
    assert N_TYPES == 47, N_TYPES
    assert RARE_EDGE_TYPES <= set(ALL_EDGE_TYPES)
    batch_main()
