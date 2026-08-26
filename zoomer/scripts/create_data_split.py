import glob
import json
import os
import random
from collections import defaultdict
from data_utils import instance_from_filename, technique_tactics, load_tactic_map
ZOOMER_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
CAM_LDS_DIR = os.path.dirname(os.path.dirname(ZOOMER_SCRIPTS_DIR))
OUTPUT_PATH = os.path.join(ZOOMER_SCRIPTS_DIR, 'data_split_output.json')
GRAPHS_DIR = os.path.join(CAM_LDS_DIR, 'graphs')
SEED = 0
MIN_TRAIN_SAMPLES = 4
TEST_FRACTION = 0.2

def discover_technique_files(single_label=True):
    raw_by_technique = defaultdict(dict)
    for fp in glob.glob(os.path.join(GRAPHS_DIR, '*', '*', '*.json')):
        parts = fp.split(os.sep)
        (technique, filename) = (parts[-2], parts[-1])
        raw_by_technique[technique].setdefault(filename, fp)
    if not single_label:
        return raw_by_technique
    instance_to_techniques = defaultdict(set)
    instance_to_filename = {}
    for (technique, files) in raw_by_technique.items():
        for filename in files:
            inst = instance_from_filename(filename)
            instance_to_techniques[inst].add(technique)
            instance_to_filename[inst] = filename
    by_technique = defaultdict(dict)
    for (inst, techniques) in instance_to_techniques.items():
        winning_technique = sorted(techniques)[0]
        filename = instance_to_filename[inst]
        by_technique[winning_technique][filename] = raw_by_technique[winning_technique][filename]
    return by_technique

def technique_primary_tactic(tactic_map, technique):
    return sorted(technique_tactics(tactic_map, technique))[0]

def build_split(seed=SEED, single_label=True):
    rng = random.Random(seed)
    by_technique = discover_technique_files(single_label=single_label)
    qualifying = {t: files for (t, files) in by_technique.items() if len(files) >= MIN_TRAIN_SAMPLES}
    instance_assignment = {}
    split = {t: {'train': [], 'test': []} for t in qualifying}
    for technique in sorted(qualifying):
        items = list(qualifying[technique].items())
        rng.shuffle(items)
        n_test_target = max(round(len(items) * TEST_FRACTION), 1)
        undecided = []
        n_test_already = 0
        for (filename, path) in items:
            instance = instance_from_filename(filename)
            if instance in instance_assignment:
                label = instance_assignment[instance]
                split[technique][label].append((filename, path))
                if label == 'test':
                    n_test_already += 1
            else:
                undecided.append((filename, path, instance))
        n_test_needed = max(n_test_target - n_test_already, 0)
        for (i, (filename, path, instance)) in enumerate(undecided):
            label = 'test' if i < n_test_needed else 'train'
            instance_assignment[instance] = label
            split[technique][label].append((filename, path))
    return split

def _serialize(split):
    return {technique: {'train': [{'filename': fn, 'path': path} for (fn, path) in pools['train']], 'test': [{'filename': fn, 'path': path} for (fn, path) in pools['test']]} for (technique, pools) in split.items()}

def main():
    tactic_map = load_tactic_map()
    single_label_out = _serialize(build_split(seed=SEED, single_label=True))
    for technique in single_label_out:
        single_label_out[technique]['tactic'] = technique_primary_tactic(tactic_map, technique)
    multi_label_out = _serialize(build_split(seed=SEED, single_label=False))
    zoomer_tactics = sorted({v['tactic'] for v in single_label_out.values()})
    zoomer_test_instances = sorted({instance_from_filename(row['filename']) for v in single_label_out.values() for row in v['test']})
    output = {'seed': SEED, 'min_train_samples': MIN_TRAIN_SAMPLES, 'test_fraction': TEST_FRACTION, 'single_label_techniques': single_label_out, 'multi_label_techniques': multi_label_out, 'zoomer_tactics': zoomer_tactics, 'zoomer_test_instances': zoomer_test_instances}
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    n_zoomer_train = sum((len(v['train']) for v in single_label_out.values()))
    n_zoomer_test = sum((len(v['test']) for v in single_label_out.values()))
    n_multi_train = sum((len(v['train']) for v in multi_label_out.values()))
    n_multi_test = sum((len(v['test']) for v in multi_label_out.values()))
    print('ZOOMER split (single-label): {} techniques, {} tactics, {} train, {} test'.format(len(single_label_out), len(zoomer_tactics), n_zoomer_train, n_zoomer_test))
    print('Sequence split (own, multi-label): {} techniques, {} train rows, {} test rows'.format(len(multi_label_out), n_multi_train, n_multi_test))
    print('Saved -> {}'.format(OUTPUT_PATH))
if __name__ == '__main__':
    main()
