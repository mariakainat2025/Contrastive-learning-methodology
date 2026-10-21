
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAM_LDS_SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'CAM-LDS', 'scripts')
if CAM_LDS_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, CAM_LDS_SCRIPTS_DIR)

from tactic_to_stage import TACTIC_TO_STAGE, STAGE_ORDER

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
