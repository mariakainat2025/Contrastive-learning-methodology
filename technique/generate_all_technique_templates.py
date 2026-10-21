import json
import os
import re

MITRE_PATH = '/csse/research/contructive-learning/CAM-LDS/enterprise-attack.json'
TACTIC_MAP_PATH = '/csse/research/contructive-learning/CAM-LDS/scripts/folder_tactic_map.json'
OUTPUT_DIR = '/csse/research/contructive-learning/CAM-LDS/technique/templates_all92'


def base_id(t):
    return t.split('-')[0]


def clean_description(text):
    text = re.sub(r'\(Citation:[^)]*\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'</?code>', '', text)
    text = text.replace('`', '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(TACTIC_MAP_PATH) as f:
        tactic_map = json.load(f)
    our_techniques = sorted({k.split('/')[0] for k in tactic_map})
    print('CAM-LDS techniques found: {}'.format(len(our_techniques)))

    with open(MITRE_PATH) as f:
        data = json.load(f)
    objects = data['objects']

    by_ext_id = {}
    for obj in objects:
        if obj.get('type') == 'attack-pattern':
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    by_ext_id[ref['external_id']] = obj

    base_ids = sorted({base_id(t) for t in our_techniques})
    print('Generating templates for {} base techniques...'.format(len(base_ids)))

    n_ok = 0
    missing = []
    for tid in base_ids:
        obj = by_ext_id.get(tid)
        if not obj:
            missing.append(tid)
            continue
        name = obj['name']
        description = clean_description(obj.get('description', ''))
        safe_name = name.replace(' ', '_').replace('/', '-')
        filename = '{}_{}.txt'.format(tid, safe_name)
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(description + '\n')
        print('  [{}] {} -> {}'.format(tid, name, filename))
        n_ok += 1

    print()
    print('Done. {}/{} templates written -> {}'.format(n_ok, len(base_ids), OUTPUT_DIR))
    if missing:
        print('WARNING: {} technique(s) not found in MITRE data: {}'.format(len(missing), missing))


if __name__ == '__main__':
    main()
