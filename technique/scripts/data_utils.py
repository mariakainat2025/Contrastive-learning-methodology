import json
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
