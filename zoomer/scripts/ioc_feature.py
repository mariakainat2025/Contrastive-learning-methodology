import json
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ATOMIC_IOCS_PATH = os.path.join(SCRIPTS_DIR, 'atomic_iocs.json')

OUR_TACTICS = ('collection', 'command_and_control', 'credential_access', 'discovery',
               'execution', 'initial_access', 'persistence', 'privilege_escalation', 'stealth')

IOC_DIMS = tuple(f'{tactic}_{kind}' for tactic in OUR_TACTICS for kind in ('file', 'ip'))
assert len(IOC_DIMS) == 18

def load_knowledge_base():
    with open(ATOMIC_IOCS_PATH) as f:
        data = json.load(f)
    return {tactic: {'file_paths': set(v['file_paths']), 'ips': set(v['ips'])}
            for (tactic, v) in data['by_tactic'].items()}

def _socket_ip(name):
    if '.' in name and ':' in name:
        return name.rsplit(':', 1)[0]
    return name

def ioc_feature_counts(G, knowledge_base):
    file_names = {data.get('name') for (_, data) in G.nodes(data=True) if data.get('type') == 'FILE'}
    ip_names = {_socket_ip(data.get('name', '')) for (_, data) in G.nodes(data=True)
                if data.get('type') == 'NetFlowObject'}

    counts = {}
    for tactic in OUR_TACTICS:
        kb = knowledge_base[tactic]
        counts[f'{tactic}_file'] = len(file_names & kb['file_paths'])
        counts[f'{tactic}_ip'] = len(ip_names & kb['ips'])
    return counts

def ioc_feature_vector(G, knowledge_base):
    counts = ioc_feature_counts(G, knowledge_base)
    return [counts[dim] for dim in IOC_DIMS]

if __name__ == '__main__':
    import glob
    from Feature_Initialization import load_graph

    kb = load_knowledge_base()
    files = glob.glob('/csse/research/contructive-learning/CAM-LDS/graphs/*/*/*.json')

    total_nonzero_graphs = 0
    dim_totals = {dim: 0 for dim in IOC_DIMS}

    for fp in files:
        with open(fp) as f:
            G = load_graph(json.load(f))
        counts = ioc_feature_counts(G, kb)
        if any(counts.values()):
            total_nonzero_graphs += 1
        for (dim, c) in counts.items():
            dim_totals[dim] += c

    print('Graphs scanned:', len(files))
    print('Graphs with at least one IoC hit:', total_nonzero_graphs)
    print()
    print('Total hits per dimension (across all graphs):')
    for dim in IOC_DIMS:
        if dim_totals[dim]:
            print(' ', dim, ':', dim_totals[dim])
