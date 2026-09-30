TACTIC_TO_STAGE = {
    'reconnaissance':      'Initial Compromise',
    'initial_access':      'Initial Compromise',
    'execution':            'Establish Foothold',
    'command_and_control':  'Establish Foothold',
    'privilege_escalation': 'Escalate Privilege',
    'credential_access':    'Escalate Privilege',
    'discovery':            'Internal Reconnaissance',
    'collection':           'Internal Reconnaissance',
    'lateral_movement':     'Move Laterally',
    'persistence':          'Maintain Persistence',
    'stealth':              'Maintain Persistence',
    'defense_impairment':   'Maintain Persistence',
    'exfiltration':         'Complete Mission',
    'impact':               'Complete Mission',
}

STAGE_ORDER = [
    'Initial Compromise',
    'Establish Foothold',
    'Escalate Privilege',
    'Internal Reconnaissance',
    'Move Laterally',
    'Maintain Persistence',
    'Complete Mission',
]


def tactics_to_stages(tactics):
    return sorted({TACTIC_TO_STAGE[t] for t in tactics if t in TACTIC_TO_STAGE},
                  key=STAGE_ORDER.index)


def stage_scores_from_tactic_scores(tactic_scores):
    stage_scores = {}
    for tactic, score in tactic_scores.items():
        stage = TACTIC_TO_STAGE.get(tactic)
        if stage is None:
            continue
        stage_scores[stage] = stage_scores.get(stage, 0.0) + score
    return stage_scores
