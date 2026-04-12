import os
import sys
import pickle
import torch
os.environ.setdefault('PYTHONHASHSEED', '0')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import show

from scripts.parse_provenance          import run_parser
from scripts.extract_graphs            import read_single_graph, run_extract_windows
from scripts.reduce_graph              import reduce_graph
from scripts.extract_benign_subgraphs  import main as extract_benign
from scripts.label_subgraphs           import main as extract_attack
from scripts.filter_attack_subgraphs   import run_all as filter_attack
from scripts.subgraph_sequence_builder import run_sequences
from scripts.encode_sequences          import encode_sequences
from scripts.encode_cti                import encode_cti
from scripts.tokenize_sequences        import tokenize_all, tokenize_benign_testing
from scripts.train_detector            import run_contrastive_train
from scripts.evaluate                  import evaluate

CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')

def _save(name, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name + '.pkl')
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'  [cache] saved → {path}')

def _load(name):
    path = os.path.join(CACHE_DIR, name + '.pkl')
    if os.path.exists(path):
        print(f'  [cache] loading → {path}')
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def main(file=None):
    os.chdir(PROJECT_ROOT)
    tag = 'all'

    show('Stage 1 / 10 — parse_provenance')
    print()
    maps = _load('stage1_maps')
    if maps is None:
        maps = run_parser(file=file)
        _save('stage1_maps', maps)
    else:
        print('  [cache] skipping stage 1')
    print()

    edges_all = maps.get('edges_all', '')
    print('  Combined edges file : {}'.format(edges_all))
    print()

    show('Stage 2 / 10 — build graph  [{}]'.format(tag))
    print()
    cached = _load('stage2_graph_{}'.format(tag))
    if cached is None:
        node_map, g = read_single_graph('theia', edges_all, tag=tag)
        _save('stage2_graph_{}'.format(tag), (node_map, g))
    else:
        node_map, g = cached
        print('  [cache] skipping stage 2')
    print()

    show('Stage 3 / 10 — graph reduction  [{}]'.format(tag))
    print()
    cached = _load('stage3_graph_{}'.format(tag))
    if cached is None:
        g = reduce_graph(g)
        _save('stage3_graph_{}'.format(tag), g)
    else:
        g = cached
        print('  [cache] skipping stage 3')
    print()

    show('Stage 4 / 10 — subgraph partition  [{}]'.format(tag))
    print()
    cached = _load('stage4_subgraphs_{}'.format(tag))
    if cached is None:
        subgraphs = run_extract_windows(g, tag='subgraphs_{}'.format(tag))
        _save('stage4_subgraphs_{}'.format(tag), subgraphs)
    else:
        subgraphs = cached
        print('  [cache] skipping stage 4')
    print()

    show('Stage 5 / 10 — extract benign subgraphs')
    print()
    cached = _load('stage5_benign')
    if cached is None:
        extract_benign()
        _save('stage5_benign', True)
    else:
        print('  [cache] skipping stage 5')
    print()

    show('Stage 6 / 10 — attack subgraph extraction')
    print()
    cached = _load('stage6_attack')
    if cached is None:
        extract_attack()
        _save('stage6_attack', True)
    else:
        print('  [cache] skipping stage 6')
    print()

    show('Stage 6b / 10 — filter attack subgraphs')
    print()
    cached = _load('stage6b_filter_attack')
    if cached is None:
        filter_attack()
        _save('stage6b_filter_attack', True)
    else:
        print('  [cache] skipping stage 6b')
    print()

    show('Stage 7 / 10 — log sequence construction')
    print()
    cached = _load('stage7_sequences')
    if cached is None:
        run_sequences()
        _save('stage7_sequences', True)
    else:
        print('  [cache] skipping stage 7')
    print()

    show('Stage 8 / 10 — tokenize sequences')
    print()
    cached = _load('stage8_tokenize')
    if cached is None:
        tokenize_all()
        tokenize_benign_testing()
        _save('stage8_tokenize', True)
    else:
        print('  [cache] skipping stage 8')
    print()

    show('Stage 9 / 10 — contrastive training')
    print()
    cached = _load('stage9_training')
    if cached is None:
        run_contrastive_train()
        _save('stage9_training', True)
    else:
        print('  [cache] skipping stage 9')
    print()

    show('Stage 10 / 10 — evaluation')
    print()
    evaluate()
    print()

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', type=str, default=None,
                    help='Parse only this file. If omitted, parses all JSON files in input/theia/')
    args = ap.parse_args()
    main(file=args.file)
