import glob
import json
import os
import random
from collections import defaultdict
from data_utils import instance_from_filename, technique_tactics, load_tactic_map
ZOOMER_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
CAM_LDS_DIR = os.path.dirname(os.path.dirname(ZOOMER_SCRIPTS_DIR))
OUTPUT_PATH = os.path.join(ZOOMER_SCRIPTS_DIR, 'data_split_output.json')
INSTANCE_TECHNIQUE_MAP_PATH = os.path.join(ZOOMER_SCRIPTS_DIR, 'instance_technique_map.json')
ZOOMER_INSTANCE_LABEL_PATH = os.path.join(ZOOMER_SCRIPTS_DIR, 'zoomer_instance_label.json')
GRAPHS_DIR = os.path.join(CAM_LDS_DIR, 'graphs')
SEED = 0
MIN_TRAIN_SAMPLES = 4
TEST_FRACTION = 0.2

def discover_technique_files(single_label=True):
    raw_by_technique = defaultdict(dict)
    for fp in sorted(glob.glob(os.path.join(GRAPHS_DIR, '*', '*', '*.json'))):
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

def get_zoomer_split(seed):
    tactic_map = load_tactic_map()
    split = build_split(seed=seed, single_label=True)
    return {technique: {'train': pools['train'], 'test': pools['test'], 'tactic': technique_primary_tactic(tactic_map, technique)}
            for (technique, pools) in split.items()}

def get_zoomer_split_multilabel(seed):
    scope_techniques = set(get_zoomer_split(seed).keys())
    tactic_map = load_tactic_map()

    raw_by_technique = discover_technique_files(single_label=False)
    qualifying = {t: files for (t, files) in raw_by_technique.items() if t in scope_techniques}

    rng = random.Random(seed)
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

    return {technique: {'train': pools['train'], 'test': pools['test'], 'tactic': technique_primary_tactic(tactic_map, technique)}
            for (technique, pools) in split.items()}

MIN_TRAIN_SAMPLES_SCENARIO = 2

def _scenario_assign(files_dict, scenario):
    scenario = str(scenario)
    prefix_us = scenario + '_'
    prefix_hy = scenario + '-'
    train, test = [], []
    for (filename, path) in files_dict.items():
        instance = instance_from_filename(filename)
        is_test = instance.startswith(prefix_us) or instance.startswith(prefix_hy)
        (test if is_test else train).append((filename, path))
    return train, test

def build_split_scenario(scenario, single_label=True):
    # keeps EVERY technique, even ones with 0 or 1 training samples -- callers decide
    # which techniques are trainable (classes_for_scenario_split below); test steps for
    # an excluded technique still show up here so no real step ever gets hidden
    by_technique = discover_technique_files(single_label=single_label)
    split = {}
    for technique in sorted(by_technique):
        train, test = _scenario_assign(by_technique[technique], scenario)
        split[technique] = {'train': train, 'test': test}
    return split

def get_zoomer_split_scenario(scenario, single_label=True):
    tactic_map = load_tactic_map()
    split = build_split_scenario(scenario, single_label=single_label)
    return {technique: {'train': pools['train'], 'test': pools['test'], 'tactic': technique_primary_tactic(tactic_map, technique)}
            for (technique, pools) in split.items()}

def get_zoomer_split_scenario_multilabel(scenario):
    tactic_map = load_tactic_map()
    raw_by_technique = discover_technique_files(single_label=False)
    split = {}
    for technique in sorted(raw_by_technique):
        train, test = _scenario_assign(raw_by_technique[technique], scenario)
        split[technique] = {'train': train, 'test': test}
    return {technique: {'train': pools['train'], 'test': pools['test'], 'tactic': technique_primary_tactic(tactic_map, technique)}
            for (technique, pools) in split.items()}

def get_zoomer_split_scenario_technique_singlelabel(scenario):
    # same single-label-train / multi-label-test idea as the tactic version: each
    # instance only trains toward its one "winning" technique (sorted first of its
    # true technique set), but still appears under EVERY technique it truly has on
    # the test side, so evaluation is scored against the full real label set.
    tactic_map = load_tactic_map()
    raw_by_technique = discover_technique_files(single_label=False)

    instance_techs_full = defaultdict(set)
    instance_file = {}
    for (technique, files) in raw_by_technique.items():
        for (filename, path) in files.items():
            instance = instance_from_filename(filename)
            instance_techs_full[instance].add(technique)
            instance_file[instance] = (filename, path)
    winning_technique_of = {inst: sorted(techs)[0] for (inst, techs) in instance_techs_full.items()}

    split = {}
    for technique in sorted(raw_by_technique):
        train_files = {fn: p for (inst, (fn, p)) in instance_file.items() if winning_technique_of[inst] == technique}
        (train, _) = _scenario_assign(train_files, scenario)
        (_, test) = _scenario_assign(raw_by_technique[technique], scenario)
        split[technique] = {'train': train, 'test': test}
    return {technique: {'train': pools['train'], 'test': pools['test'], 'tactic': technique_primary_tactic(tactic_map, technique)}
            for (technique, pools) in split.items()}

def _bucket_files_by_tactic(tactic_map, raw_by_technique):
    by_tactic = defaultdict(dict)
    for (technique, files) in raw_by_technique.items():
        for tactic in technique_tactics(tactic_map, technique):
            by_tactic[tactic].update(files)
    return by_tactic

def build_split_scenario_tactic(scenario):
    # tactic-level counterpart of build_split_scenario -- classes are tactics (14 of them)
    # instead of techniques (92), so each class gets far more training examples. Same
    # keep-every-class, filter-separately rule as the technique version: every tactic stays
    # in `split` even with 0 train samples, so no test step is ever hidden.
    tactic_map = load_tactic_map()
    raw_by_technique = discover_technique_files(single_label=False)
    by_tactic = _bucket_files_by_tactic(tactic_map, raw_by_technique)
    split = {}
    for tactic in sorted(by_tactic):
        train, test = _scenario_assign(by_tactic[tactic], scenario)
        split[tactic] = {'train': train, 'test': test}
    return split

def get_zoomer_split_scenario_tactic(scenario):
    return build_split_scenario_tactic(scenario)

def build_split_scenario_tactic_singlelabel(scenario):
    # TRAIN side: single-label -- each instance only trains toward its one "winning"
    # tactic (sorted first of its full tactic set), same rule as the technique-level
    # single-label split, so an episode never needs the sequential multi-label path.
    # TEST side: multi-label -- an instance still appears under EVERY tactic it truly
    # has, so evaluation is scored honestly against its full real label set, not just
    # whichever one tactic happened to win training.
    tactic_map = load_tactic_map()
    raw_by_technique = discover_technique_files(single_label=False)
    by_tactic_multi = _bucket_files_by_tactic(tactic_map, raw_by_technique)

    instance_tactics_full = defaultdict(set)
    instance_file = {}
    for (tactic, files) in by_tactic_multi.items():
        for (filename, path) in files.items():
            instance = instance_from_filename(filename)
            instance_tactics_full[instance].add(tactic)
            instance_file[instance] = (filename, path)
    winning_tactic_of = {inst: sorted(tacs)[0] for (inst, tacs) in instance_tactics_full.items()}

    split = {}
    for tactic in sorted(by_tactic_multi):
        train_files = {fn: p for (inst, (fn, p)) in instance_file.items() if winning_tactic_of[inst] == tactic}
        (train, _) = _scenario_assign(train_files, scenario)
        (_, test) = _scenario_assign(by_tactic_multi[tactic], scenario)
        split[tactic] = {'train': train, 'test': test}
    return split

def get_zoomer_split_scenario_tactic_singlelabel(scenario):
    return build_split_scenario_tactic_singlelabel(scenario)

def build_split_tactic(seed=SEED):
    tactic_map = load_tactic_map()
    raw_by_technique = discover_technique_files(single_label=False)
    by_tactic = _bucket_files_by_tactic(tactic_map, raw_by_technique)

    rng = random.Random(seed)
    instance_assignment = {}
    split = {t: {'train': [], 'test': []} for t in by_tactic}
    for tactic in sorted(by_tactic):
        items = list(by_tactic[tactic].items())
        rng.shuffle(items)
        n_test_target = max(round(len(items) * TEST_FRACTION), 1)
        undecided = []
        n_test_already = 0
        for (filename, path) in items:
            instance = instance_from_filename(filename)
            if instance in instance_assignment:
                label = instance_assignment[instance]
                split[tactic][label].append((filename, path))
                if label == 'test':
                    n_test_already += 1
            else:
                undecided.append((filename, path, instance))
        n_test_needed = max(n_test_target - n_test_already, 0)
        for (i, (filename, path, instance)) in enumerate(undecided):
            label = 'test' if i < n_test_needed else 'train'
            instance_assignment[instance] = label
            split[tactic][label].append((filename, path))
    return split

def get_zoomer_split_tactic(seed):
    return build_split_tactic(seed=seed)

def classes_for_scenario_split(split, min_train_samples=MIN_TRAIN_SAMPLES_SCENARIO):
    """Which techniques actually get a trained prototype -- >= min_train_samples
    training examples left after holding out the scenario. Techniques below that
    stay in `split` (their test steps still get scored, just against prototypes
    they were never trained on -- shown honestly as wrong, not hidden)."""
    classes = sorted(t for t in split if len(split[t]['train']) >= min_train_samples)
    dropped = sorted(t for t in split if len(split[t]['train']) < min_train_samples)
    if dropped:
        print('  Excluded from training (fewer than {} training examples): {}'.format(min_train_samples, dropped))
    return classes

def get_zoomer_split_singlelabel_matched(seed):
    tactic_map = load_tactic_map()
    multi_split = get_zoomer_split_multilabel(seed)
    test_instances = {instance_from_filename(fn) for pools in multi_split.values() for (fn, _) in pools['test']}

    by_technique = discover_technique_files(single_label=True)
    qualifying = {t: files for (t, files) in by_technique.items() if len(files) >= MIN_TRAIN_SAMPLES}
    split = {t: {'train': [], 'test': []} for t in qualifying}
    for (technique, files) in qualifying.items():
        for (filename, path) in files.items():
            instance = instance_from_filename(filename)
            label = 'test' if instance in test_instances else 'train'
            split[technique][label].append((filename, path))

    return {technique: {'train': pools['train'], 'test': pools['test'], 'tactic': technique_primary_tactic(tactic_map, technique)}
            for (technique, pools) in split.items()}

def build_instance_technique_map():
    raw_by_technique = discover_technique_files(single_label=False)
    instance_to_techniques = defaultdict(set)
    for (technique, files) in raw_by_technique.items():
        for filename in files:
            inst = instance_from_filename(filename)
            instance_to_techniques[inst].add(technique)
    return {inst: sorted(techs) for (inst, techs) in instance_to_techniques.items()}

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
    instance_technique_map = build_instance_technique_map()
    with open(INSTANCE_TECHNIQUE_MAP_PATH, 'w') as f:
        json.dump(instance_technique_map, f, indent=2)
    zoomer_instance_label = {}
    for (technique, pools) in single_label_out.items():
        for split_name in ('train', 'test'):
            for row in pools[split_name]:
                inst = instance_from_filename(row['filename'])
                zoomer_instance_label[inst] = {'technique': technique, 'tactic': pools['tactic'], 'split': split_name}
    with open(ZOOMER_INSTANCE_LABEL_PATH, 'w') as f:
        json.dump(zoomer_instance_label, f, indent=2)
    n_zoomer_train = sum((len(v['train']) for v in single_label_out.values()))
    n_zoomer_test = sum((len(v['test']) for v in single_label_out.values()))
    n_multi_train = sum((len(v['train']) for v in multi_label_out.values()))
    n_multi_test = sum((len(v['test']) for v in multi_label_out.values()))
    n_multi_technique_instances = sum((1 for techs in instance_technique_map.values() if len(techs) > 1))
    print('ZOOMER split (single-label): {} techniques, {} tactics, {} train, {} test'.format(len(single_label_out), len(zoomer_tactics), n_zoomer_train, n_zoomer_test))
    print('Sequence split (own, multi-label): {} techniques, {} train rows, {} test rows'.format(len(multi_label_out), n_multi_train, n_multi_test))
    print('Instance -> technique map: {} instances, {} with >1 true technique'.format(len(instance_technique_map), n_multi_technique_instances))
    print('ZOOMER instance label lookup: {} instances, exactly 1 technique + 1 tactic each'.format(len(zoomer_instance_label)))
    print('Saved -> {}'.format(OUTPUT_PATH))
    print('Saved -> {}'.format(INSTANCE_TECHNIQUE_MAP_PATH))
    print('Saved -> {}'.format(ZOOMER_INSTANCE_LABEL_PATH))
if __name__ == '__main__':
    main()
