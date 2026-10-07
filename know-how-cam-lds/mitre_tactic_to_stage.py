"""mitre_tactic_to_stage.py

Maps a MITRE technique -> MITRE tactic(s) -> CAM-LDS/ZOOMER stage, using the
SAME stage vocabulary and TACTIC_TO_STAGE table ZOOMER itself uses
(CAM-LDS/scripts/tactic_to_stage.py), instead of sean's separate
tech2tac.txt/tac2stage.txt text files -- so a technique found via CKD lands on
exactly the same stage name ZOOMER would report for it.

technique -> MITRE tactic still comes from tech2tac.txt (it's the general
549-technique MITRE universe the CKD gIoC database itself is built from;
CAM-LDS's own folder_tactic_map.json is ground-truth labels for its own
labeled instances only, not a general technique lookup).

CAM-LDS's TACTIC_TO_STAGE is keyed by its own 14 tactic folder names, which
match MITRE's 14 tactic names 1:1 after lowercasing + underscoring, with two
exceptions handled explicitly below:
  - "Defense Evasion" has no matching CAM-LDS key (CAM-LDS splits it into
    'stealth' and 'defense_impairment' based on its own ground-truth
    labeling) -- both of those map to the same stage, 'Maintain Persistence',
    so that's used directly, no disambiguation needed.
  - "Resource Development" has no matching CAM-LDS key either -- placed under
    'Establish Foothold', matching sean's own tac2stage.txt placement (the
    only existing reference for this tactic).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAM_LDS_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'scripts')
if CAM_LDS_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, CAM_LDS_SCRIPTS_DIR)

from tactic_to_stage import TACTIC_TO_STAGE, STAGE_ORDER  # noqa: E402

STAGE_OVERRIDES = {
    'Defense Evasion':      'Maintain Persistence',
    'Resource Development': 'Establish Foothold',
}


def mitre_tactic_to_stage(tactic):
    if tactic in STAGE_OVERRIDES:
        return STAGE_OVERRIDES[tactic]
    key = tactic.lower().replace(' ', '_')
    return TACTIC_TO_STAGE.get(key)


def load_tech2tac(path):
    tech2tac = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            tech, tac_str = line.split('\t')
            tech2tac[tech] = [t.strip() for t in tac_str.split(',')]
    return tech2tac


def technique_to_tactics_and_stages(technique, tech2tac):
    tactics = tech2tac.get(technique, [])
    stages = []
    for tac in tactics:
        stage = mitre_tactic_to_stage(tac)
        if stage and stage not in stages:
            stages.append(stage)
    stages.sort(key=STAGE_ORDER.index)
    return tactics, stages
