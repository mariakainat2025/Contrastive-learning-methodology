import os
import sys
import pickle
import torch
os.environ.setdefault('PYTHONHASHSEED', '0')


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import show, OUTPUT_PARSED, OUTPUT_GRAPHS


from scripts.parse_provenance    import run_parser
from scripts.extract_graphs     import read_single_graph, run_extract_windows
from scripts.reduce_graph        import reduce_graph
from scripts.subgraph_sequence_builder import build_log_sequences
from scripts.encode_sequences          import encode_sequences

# ── Inactive stages — uncomment import when enabling the stage ────────────────
# from scripts.match_iocs          import run_ioc_matching
# from scripts.build_graphs        import run_graph_builder
# from scripts.embed_windows       import main as run_embed
# from scripts.encode_cti          import run_cti_encoding
# from scripts.train_detector      import run_contrastive_train
# from scripts.detect              import run_contrastive_test

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


def main():
    os.chdir(PROJECT_ROOT)

    show('Stage 1 / 8 — parse_provenance')
    print()
    maps = run_parser()
    _save('stage1_maps', maps)
    print()

    edges_all  = maps.get('edges_all', '')
    tag        = 'all'
    print('  Combined edges file : {}'.format(edges_all))
    print()

    show('Stage 3a / 8 — build graph  [{}]'.format(tag))
    print()
    node_map, g = read_single_graph('theia', edges_all, tag=tag)
    _save('stage3a_graph_{}'.format(tag), (node_map, g))
    print()

    show('Stage 3a.5 / 8 — graph reduction [{}]'.format(tag))
    print()
    report_path = os.path.join(OUTPUT_GRAPHS, 'reduction_report_{}.txt'.format(tag))
    g = reduce_graph(g, report_path=report_path)
    _save('stage3a5_graph_{}'.format(tag), g)
    print()

    show('Stage 3b / 8 — subgraph partition  [{}]'.format(tag))
    print()
    subgraphs = run_extract_windows(g, tag='subgraphs_{}'.format(tag))
    _save('stage3b_subgraphs_{}'.format(tag), subgraphs)
    print()

    show('Stage 4 / 8 — log sequence construction  [{}]'.format(tag))
    print()
    build_log_sequences(tag)
    print()

    # show('Stage 5 / 8 — encode sequences (RoBERTa)')
    # print()
    # encode_sequences(tag)
    # print()

    # show('Stage 4 / 8 — build_graphs')
    # print()
    # run_graph_builder(maps)
    # print()

    # show('Stage 5 / 8 — embed_windows ')
    # print()
    # run_embed()
    # print()

    # show('Stage 6 / 8 — encode_cti ')
    # print()
    # run_cti_encoding()
    # print()

    # show('Stage 7 / 8 — train_detector')
    # print()
    # run_contrastive_train()
    # print()


    # show('Stage 8 / 8 — detect  (window_3 and benign_4)')
    # print()
    # run_contrastive_test()
    # print()


if __name__ == '__main__':
    main()