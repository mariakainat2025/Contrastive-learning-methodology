import json
from Deep_Model import graph_to_tensors
from Feature_Initialization import load_graph
FOLDER_TACTIC_MAP_PATH = '/csse/research/contructive-learning/CAM-LDS/scripts/folder_tactic_map.json'
K_SHOT = 3
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

def technique_tactics(tactic_map, technique):
    tactics = set()
    prefix = technique + '/'
    for (key, tacs) in tactic_map.items():
        if key.startswith(prefix):
            tactics.update(tacs)
    return tactics

def instance_tactics(tactic_map, technique, filename):
    instance = instance_from_filename(filename)
    key = '{}/{}'.format(technique, instance)
    if key in tactic_map:
        return set(tactic_map[key])
    return technique_tactics(tactic_map, technique)
DATA_SPLIT_PATH = '/csse/research/contructive-learning/CAM-LDS/zoomer/scripts/data_split_output.json'

def load_split(path=DATA_SPLIT_PATH, single_label=True):
    with open(path) as f:
        data = json.load(f)
    key = 'single_label_techniques' if single_label else 'multi_label_techniques'
    return {technique: {'train': [(row['filename'], row['path']) for row in pools['train']], 'test': [(row['filename'], row['path']) for row in pools['test']]} for (technique, pools) in data[key].items()}

def load_technique_tactics(path=DATA_SPLIT_PATH):
    with open(path) as f:
        data = json.load(f)
    return {t: v['tactic'] for (t, v) in data['single_label_techniques'].items()}

class GraphTensorCache:

    def __init__(self):
        self._cache = {}

    def get(self, path):
        if path not in self._cache:
            with open(path) as f:
                G = load_graph(json.load(f))
            (h, adjacency, _) = graph_to_tensors(G)
            self._cache[path] = (h, adjacency)
        return self._cache[path]
