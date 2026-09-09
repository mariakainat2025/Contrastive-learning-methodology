import json
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ZOOMER_DIR = os.path.dirname(SCRIPTS_DIR)

DATA_SPLIT_PATH = os.path.join(SCRIPTS_DIR, 'data_split_output.json')
FILE_PATHS_PATH = os.path.join(SCRIPTS_DIR, 'file_paths.json')

CACHE_DIR = os.path.join(ZOOMER_DIR, 'cache')
FEATURES_CACHE_DIR = os.path.join(CACHE_DIR, 'features')
DEEP_TENSORS_CACHE_DIR = os.path.join(CACHE_DIR, 'deep_tensors')
WIDE_FEATURES_CACHE_DIR = os.path.join(CACHE_DIR, 'wide_features')

SEEDS = (0, 1, 2)

def stage_data_split():
    print('== Stage 1: data split ==')
    if not os.path.exists(DATA_SPLIT_PATH):
        import create_data_split
        create_data_split.main()
    else:
        print('  cached ->', DATA_SPLIT_PATH)
    with open(DATA_SPLIT_PATH) as f:
        return json.load(f)

def scoped_graph_paths(split_data):
    paths = set()
    for pools in split_data['single_label_techniques'].values():
        for row in pools['train'] + pools['test']:
            paths.add(row['path'])
    return sorted(paths)

def stage_file_paths(graph_paths):
    print('== Stage 2: file paths ==')
    paths = set()
    for fp in graph_paths:
        with open(fp) as f:
            d = json.load(f)
        for n in d.get('nodes', []):
            if n.get('type') == 'FILE':
                name = n.get('name')
                if name:
                    paths.add(name)
    result = sorted(paths)
    with open(FILE_PATHS_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print('  {} unique FILE paths -> {}'.format(len(result), FILE_PATHS_PATH))
    return result

def stage_feature_initialization(graph_paths):
    print('== Stage 3: Feature_Initialization ==')
    from Feature_Initialization import GRAPHS_DIR, batch_main_scoped
    batch_main_scoped(graph_paths, graphs_dir=GRAPHS_DIR, out_dir=FEATURES_CACHE_DIR)

def stage_deep_model(graph_paths):
    print('== Stage 4: Deep Model (per-graph node/edge tensors) ==')
    from Feature_Initialization import GRAPHS_DIR
    from Deep_Model import cache_graph_tensors
    cache_graph_tensors(graph_paths, graphs_dir=GRAPHS_DIR, out_dir=DEEP_TENSORS_CACHE_DIR)

def stage_wide_model(graph_paths):
    print('== Stage 5: Wide Model (node-abstract + edge-type + ioc counts) ==')
    from Feature_Initialization import GRAPHS_DIR
    from Wide_Model import cache_wide_features
    cache_wide_features(graph_paths, graphs_dir=GRAPHS_DIR, out_dir=WIDE_FEATURES_CACHE_DIR)

def stage_train_test(seeds=SEEDS):
    print('== Stage 6: Combine + Train + Test ({} seeds) =='.format(len(seeds)))
    from run_multi_seed import main as run_multi_seed_main
    return run_multi_seed_main(seeds)

def main():
    split_data = stage_data_split()
    graph_paths = scoped_graph_paths(split_data)
    print("  {} graphs in ZOOMER's 9-tactic scope: {}".format(
        len(graph_paths), ', '.join(sorted(split_data['zoomer_tactics']))))
    print()

    stage_file_paths(graph_paths)
    stage_feature_initialization(graph_paths)
    stage_deep_model(graph_paths)
    stage_wide_model(graph_paths)

    print()
    stage_train_test()

    print()
    print('Pipeline complete.')

if __name__ == '__main__':
    main()
