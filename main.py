import os
import sys

# ── make sure 'scripts' package is importable ─────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.config import show

# ── pipeline stages ───────────────────────────────────────────────────────
from scripts.parse_json       import run_parser
from scripts.create_subgraph  import run_matching
from scripts.graph_builder    import build_graphs
from scripts.embed            import main as run_embed
from scripts.report_embedding import run_cti_encoding


def main():

    show('Stage 1 / 5 — parse_json')
    print()
    maps = run_parser()
    print()


    show('Stage 2 / 5 — create_subgraph')
    print()
    run_matching(maps)
    print()

  
    show('Stage 3 / 5 — graph_builder')
    print()
    build_graphs(maps)
    print()

  
    show('Stage 4 / 5 — embed  (SecureBERT + GAT)')
    print()
    run_embed()
    print()

   
    # show('Stage 5 / 5 — report_embedding  (CTI SecureBERT)')
    # print()
    # run_cti_encoding()
    # print()

    # print('=' * 65)
    # show('Pipeline complete')
    # print('=' * 65)


if __name__ == '__main__':
    main()
