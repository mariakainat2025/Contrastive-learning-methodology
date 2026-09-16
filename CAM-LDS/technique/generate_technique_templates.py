import json
import os
import re

MITRE_PATH = '/csse/research/contructive-learning/CAM-LDS/enterprise-attack.json'
OUTPUT_DIR = '/csse/research/contructive-learning/CAM-LDS/technique/templates'

OUR_TECHNIQUES = ['T1003-008', 'T1018-000', 'T1033-000', 'T1036-005', 'T1053-003', 'T1056-001',
                   'T1059-000', 'T1059-004', 'T1070-004', 'T1078-003', 'T1083-000', 'T1105-000',
                   'T1219-000', 'T1543-002', 'T1546-000', 'T1548-003']


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
    with open(MITRE_PATH) as f:
        data = json.load(f)
    objects = data['objects']

    by_ext_id = {}
    for obj in objects:
        if obj.get('type') == 'attack-pattern':
            for ref in obj.get('external_references', []):
                if ref.get('source_name') == 'mitre-attack':
                    by_ext_id[ref['external_id']] = obj

    base_ids = sorted({base_id(t) for t in OUR_TECHNIQUES})
    print('Generating templates for {} base techniques...'.format(len(base_ids)))

    for tid in base_ids:
        obj = by_ext_id.get(tid)
        if not obj:
            print('  WARNING: {} not found in MITRE data, skipping.'.format(tid))
            continue
        name = obj['name']
        description = clean_description(obj.get('description', ''))
        safe_name = name.replace(' ', '_').replace('/', '-')
        filename = '{}_{}.txt'.format(tid, safe_name)
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(description + '\n')
        print('  [{}] {} -> {}'.format(tid, name, filename))

    print('Done. Templates -> {}'.format(OUTPUT_DIR))


if __name__ == '__main__':
    main()
