import ipaddress
import json
import os
import re

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ATOMIC_RED_TEAM_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), 'atomic-red-team')
ATOMICS_DIR = os.path.join(ATOMIC_RED_TEAM_DIR, 'atomics')
ENTERPRISE_ATTACK_PATH = os.path.join(ATOMIC_RED_TEAM_DIR, 'atomic_red_team', 'enterprise-attack.json')
OUT_PATH = os.path.join(SCRIPTS_DIR, 'atomic_iocs.json')

OUR_TACTICS = {'collection', 'command_and_control', 'credential_access', 'discovery',
               'execution', 'initial_access', 'persistence', 'privilege_escalation', 'stealth'}

TACTIC_RENAME = {'defense-evasion': 'stealth'}

IP_RE = re.compile(r'(?:\d{1,3}\.){3}\d{1,3}(?!\d)')
LOCAL_IPS = {'127.0.0.1', '0.0.0.0'}

def load_mitre_technique_tactics():
    with open(ENTERPRISE_ATTACK_PATH) as f:
        d = json.load(f)

    result = {}
    for o in d['objects']:
        if o.get('type') != 'attack-pattern':
            continue
        if o.get('revoked') or o.get('x_mitre_deprecated'):
            continue
        ext_ids = [r.get('external_id') for r in o.get('external_references', [])
                   if r.get('source_name') == 'mitre-attack']
        if not ext_ids:
            continue
        tactics = set()
        for phase in o.get('kill_chain_phases', []):
            name = phase['phase_name']
            name = TACTIC_RENAME.get(name, name).replace('-', '_')
            tactics.add(name)
        result[ext_ids[0]] = tactics
    return result

def find_yaml(atomic_id):
    base = atomic_id.split('.')[0]
    candidates = [
        os.path.join(ATOMICS_DIR, atomic_id, f'{atomic_id}.yaml'),
        os.path.join(ATOMICS_DIR, base, f'{base}.yaml'),
    ]
    return next((c for c in candidates if os.path.exists(c)), None)

def extract_file_paths(yaml_text):
    paths = []
    lines = yaml_text.splitlines()
    for (i, line) in enumerate(lines):
        if line.strip() == 'type: path':
            for j in range(i + 1, min(i + 4, len(lines))):
                m = re.match(r'\s*default:\s*(.+)', lines[j])
                if m:
                    paths.append(m.group(1).strip().strip('\'"'))
                    break
    return paths

SAFE_PRECEDING_CHARS = set(' \t\n\'"=:,(<[')

def extract_ips(yaml_text):
    ips = []
    for m in IP_RE.finditer(yaml_text):
        ip = m.group(0)
        if ip in LOCAL_IPS:
            continue
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            continue
        start = m.start()
        if start == 0:
            ips.append(ip)
            continue
        if yaml_text[start - 2:start] == '//' or yaml_text[start - 1] in SAFE_PRECEDING_CHARS:
            ips.append(ip)
    return ips

def main():
    mitre_tactics = load_mitre_technique_tactics()
    atomic_ids = sorted(d for d in os.listdir(ATOMICS_DIR)
                         if d.startswith('T') and os.path.isdir(os.path.join(ATOMICS_DIR, d)))

    by_technique = {}
    n_relevant = 0
    n_no_mitre_entry = 0

    for atomic_id in atomic_ids:
        tactics = mitre_tactics.get(atomic_id)
        if tactics is None:
            n_no_mitre_entry += 1
            continue
        relevant_tactics = tactics & OUR_TACTICS
        if not relevant_tactics:
            continue
        n_relevant += 1

        yaml_path = find_yaml(atomic_id)
        with open(yaml_path, errors='ignore') as f:
            text = f.read()

        file_paths = {}
        for p in extract_file_paths(text):
            file_paths[p] = file_paths.get(p, 0) + 1
        ips = {}
        for ip in extract_ips(text):
            ips[ip] = ips.get(ip, 0) + 1

        by_technique[atomic_id] = {'tactics': sorted(relevant_tactics), 'source_yaml': yaml_path,
                                    'file_paths': file_paths, 'ips': ips}

    by_tactic = {t: {'file_paths': {}, 'ips': {}} for t in OUR_TACTICS}
    for data in by_technique.values():
        for tactic in data['tactics']:
            for (p, c) in data['file_paths'].items():
                by_tactic[tactic]['file_paths'][p] = by_tactic[tactic]['file_paths'].get(p, 0) + c
            for (ip, c) in data['ips'].items():
                by_tactic[tactic]['ips'][ip] = by_tactic[tactic]['ips'].get(ip, 0) + c

    output = {'by_tactic': by_tactic, 'by_technique': by_technique}
    with open(OUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print('Total Atomic Red Team technique folders:', len(atomic_ids))
    print('  No MITRE entry found  :', n_no_mitre_entry)
    print('  Relevant to our 9 tactics:', n_relevant)
    print('Saved ->', OUT_PATH)
    print()
    print('Pooled per-tactic knowledge base (the actual hioc source):')
    for tactic in sorted(by_tactic):
        paths = by_tactic[tactic]['file_paths']
        ips = by_tactic[tactic]['ips']
        print('  {}: {} unique paths ({} occurrences), {} unique ips ({} occurrences)'.format(
            tactic, len(paths), sum(paths.values()), len(ips), sum(ips.values())))

if __name__ == '__main__':
    main()
