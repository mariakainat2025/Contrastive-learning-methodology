import os
import sys
import json
import glob
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

TECHNIQUE_DIR = os.path.dirname(os.path.abspath(__file__))
if TECHNIQUE_DIR not in sys.path:
    sys.path.insert(0, TECHNIQUE_DIR)

from config import ROBERTA_MODEL
from encoder_utils import embed_text
from prototype_multilabel_loss import multilabel_prototype_loss, uniformity_loss, asymmetric_loss
from wide_feature_bridge import build_step_to_graph_path, fit_wide_bins_and_masks, wide_vector_for_step

CAM_LDS_DIR       = '/csse/research/contructive-learning/CAM-LDS'
SEQUENCES_DIR     = os.path.join(CAM_LDS_DIR, 'sequences')
TEMPLATE_DIR      = os.path.join(CAM_LDS_DIR, 'technique', 'templates_all92')
STEP_TACTICS_PATH = os.path.join(CAM_LDS_DIR, 'technique', 'step_tactics.json')
OUTPUT_TRAINING = os.path.join(CAM_LDS_DIR, 'technique', 'checkpoints')
RESULTS_DIR     = os.path.join(CAM_LDS_DIR, 'technique', 'results')
os.makedirs(OUTPUT_TRAINING, exist_ok=True)
os.makedirs(RESULTS_DIR,     exist_ok=True)

TACTIC_FOLDERS = ['privilege_escalation', 'command_and_control', 'credential_access', 'execution', 'initial_access', 'lateral_movement', 'persistence', 'stealth', 'defense_impairment', 'reconnaissance', 'impact', 'discovery', 'collection', 'exfiltration']

TECHNIQUE_MERGE = {}


def base_technique(t):
    return t.split('-')[0]


ZOOMER_TECHNIQUES = [
    'T1003', 'T1016', 'T1018', 'T1021', 'T1033', 'T1036',
    'T1041', 'T1046', 'T1053', 'T1056', 'T1057',
    'T1059', 'T1070', 'T1071', 'T1072', 'T1078',
    'T1082', 'T1083', 'T1098', 'T1105', 'T1110',
    'T1115', 'T1176', 'T1201', 'T1205', 'T1218', 'T1219',
    'T1222', 'T1543', 'T1546', 'T1547', 'T1548', 'T1556',
    'T1564', 'T1566', 'T1574',
    'T1039', 'T1040', 'T1120', 'T1133', 'T1136', 'T1190', 'T1213', 'T1486', 'T1497',
    'T1204', 'T1595',
    'T1590', 'T1591',
]

SEED       = 42
LR         = 1e-5
DROPOUT    = 0.5
N_EPOCHS   = 100
EMB_DIM    = 768
PROJ_DIM   = 128
WIDE_DIM_OUT = 64
COMBINED_DIM = PROJ_DIM + WIDE_DIM_OUT
PATIENCE   = 20
N_FREEZE   = 9
K_PER_TACTIC = 1
LAMBDA_UNIFORM = 0.1
MAX_CHUNKS = None

ZOOMER_SCOPE_MIN4 = [
    'T1003', 'T1018', 'T1021', 'T1033', 'T1036', 'T1053', 'T1056', 'T1057',
    'T1059', 'T1070', 'T1072', 'T1078', 'T1082', 'T1083', 'T1105', 'T1110',
    'T1115', 'T1176', 'T1204', 'T1219', 'T1222', 'T1543', 'T1546', 'T1548',
    'T1564', 'T1574',
]


random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False


class ProjectionNetwork(nn.Module):
    def __init__(self, in_dim=EMB_DIM, out_dim=PROJ_DIM, dropout=DROPOUT):
        super().__init__()
        self.fc1     = nn.Linear(in_dim, out_dim)
        self.fc2     = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout(h1)
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        return h1 + h2


class WideModel(nn.Module):
    def __init__(self, in_dim, out_dim=WIDE_DIM_OUT):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


def freeze_lower_layers(model, n_freeze=N_FREEZE):
    for param in model.embeddings.parameters():
        param.requires_grad = False
    for i in range(n_freeze):
        for param in model.encoder.layer[i].parameters():
            param.requires_grad = False


def load_step_techniques():
    with open(STEP_TACTICS_PATH) as f:
        report = json.load(f)
    step_techniques = {}
    step_techniques_all = {}
    for row in report:
        techniques = set()
        techniques_all = set()
        for occ in row['occurrences']:
            t = TECHNIQUE_MERGE.get(occ['technique'], base_technique(occ['technique']))
            techniques_all.add(t)
            if t in ZOOMER_TECHNIQUES:
                techniques.add(t)
        step_techniques[row['step']] = sorted(techniques)
        step_techniques_all[row['step']] = sorted(techniques_all)
    return step_techniques, step_techniques_all


def load_sequences(return_total=False, sequences_dir=None):
    sequences_dir = sequences_dir or SEQUENCES_DIR
    step_techniques, step_techniques_all = load_step_techniques()

    by_step = {}
    for folder in TACTIC_FOLDERS:
        for fpath in sorted(glob.glob(os.path.join(sequences_dir, folder, '*', 'sequence_*.json'))):
            with open(fpath) as f:
                seq = json.load(f)
            step = seq['step']
            if step in by_step:
                continue
            labels = step_techniques.get(step, [])
            labels_all = step_techniques_all.get(step, [])
            by_step[step] = {
                'sequence'      : ' '.join(seq['sequence']),
                'techniques'    : labels,
                'techniques_all': labels_all,
                'file'          : step,
                'n_events'      : len(seq['sequence']),
            }

    entries = list(by_step.values())
    n_total = len(entries)

    entries = [e for e in entries if e['techniques_all']]
    n_scoped = len([e for e in entries if e['techniques']])

    multi = [e for e in entries if len(e['techniques']) > 1]
    out_of_scope_only = [e for e in entries if not e['techniques']]
    print('  Loaded {} sequences with a real technique label ({} multi-technique in scope, {} entirely '
          'outside the 36-scope — kept for testing only, never trainable). {} total sequences, {} '
          'with no technique label at all.'.format(
              len(entries), len(multi), len(out_of_scope_only), n_total, n_total - len(entries)))
    print('  In-scope (trainable) sequences: {}'.format(n_scoped))

    if return_total:
        return entries, n_total
    return entries


def leave_out_split(entries, match_substrs, seed=SEED):
    if isinstance(match_substrs, str):
        match_substrs = [match_substrs]
    rng = random.Random(seed)
    train, test = [], []
    for e in entries:
        (test if any(m in e['file'] for m in match_substrs) else train).append(e)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def leave_out_scenario_split(entries, scenario, seed=SEED):
    scenario = str(scenario)
    prefix_us = scenario + '_'
    prefix_hy = scenario + '-'
    rng = random.Random(seed)
    train, test = [], []
    for e in entries:
        f = e['file']
        is_test = f.startswith(prefix_us) or f.startswith(prefix_hy)
        (test if is_test else train).append(e)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def random_split(entries, test_size=0.2, seed=SEED):
    rng = random.Random(seed)
    items = entries[:]
    rng.shuffle(items)
    n_test = max(1, int(len(items) * test_size))
    test  = items[:n_test]
    train = items[n_test:]
    return train, test


def stratified_split(entries, test_size=0.2, seed=SEED, min_test_per_technique=1):
    rng = random.Random(seed)
    items = entries[:]
    rng.shuffle(items)

    n_test_target = max(1, int(len(items) * test_size))
    all_techniques = sorted({t for e in items for t in e['techniques']})
    technique_total = {t: sum(1 for e in items if t in e['techniques']) for t in all_techniques}

    test_idx_set = set()
    technique_test_count = {t: 0 for t in all_techniques}

    for technique in all_techniques:
        target = min(min_test_per_technique, max(0, technique_total[technique] - 1))
        for i, e in enumerate(items):
            if technique_test_count[technique] >= target:
                break
            if i in test_idx_set or technique not in e['techniques']:
                continue
            test_idx_set.add(i)
            for t in e['techniques']:
                technique_test_count[t] += 1

    for i in range(len(items)):
        if len(test_idx_set) >= n_test_target:
            break
        if i not in test_idx_set:
            test_idx_set.add(i)

    test  = [items[i] for i in sorted(test_idx_set)]
    train = [items[i] for i in range(len(items)) if i not in test_idx_set]
    return train, test


GROUPED_DIR = os.path.join(CAM_LDS_DIR, 'grouped_by_tactic')
_CMD_RE = __import__('re').compile(r"Executing [\w-]+: '(.+?)'")


def _step_path_for_file(fname):
    matches = glob.glob(os.path.join(GROUPED_DIR, '*', '*', fname))
    return matches[0] if matches else None


def _attacker_command(step_path):
    log_path = os.path.join(step_path, 'attacker', 'logs', 'attackmate.log')
    if not os.path.isfile(log_path):
        return None
    with open(log_path, errors='replace') as f:
        for line in f:
            m = _CMD_RE.search(line)
            if m:
                return m.group(1).strip()
    return None


def _hosts_and_linecounts(step_path):
    result = {}
    if not os.path.isdir(step_path):
        return result
    for host in os.listdir(step_path):
        if host in ('attacker', 'wazuh'):
            continue
        audit_log = os.path.join(step_path, host, 'logs', 'log', 'audit', 'audit.log')
        if os.path.isfile(audit_log) and os.path.getsize(audit_log) > 0:
            with open(audit_log, errors='replace') as f:
                result[host] = sum(1 for _ in f)
    return result


def _looks_like_duplicate(ccmd, chosts, ecmd, ehosts):
    if not ccmd or ccmd != ecmd:
        return False
    common = set(chosts) & set(ehosts)
    if common:
        return any(ehosts[h] > 0 and abs(chosts[h] - ehosts[h]) / ehosts[h] < 0.05 for h in common)
    if chosts and ehosts:
        c_total, e_total = sum(chosts.values()), sum(ehosts.values())
        return e_total > 0 and abs(c_total - e_total) / e_total < 0.10
    return False


def build_duplicate_groups(entries):
    sigs = []
    for e in entries:
        path = _step_path_for_file(e['file'])
        cmd = _attacker_command(path) if path else None
        hosts = _hosts_and_linecounts(path) if path else {}
        sigs.append((e, cmd, hosts))

    groups = []
    for e, cmd, hosts in sigs:
        placed = False
        for group in groups:
            ge, gcmd, ghosts = group[0]
            if _looks_like_duplicate(cmd, hosts, gcmd, ghosts):
                group.append((e, cmd, hosts))
                placed = True
                break
        if not placed:
            groups.append([(e, cmd, hosts)])
    return [[e for e, _, _ in group] for group in groups]


def grouped_stratified_split(entries, test_size=0.2, seed=SEED, min_test_per_technique=1):
    rng = random.Random(seed)
    groups = build_duplicate_groups(entries)
    rng.shuffle(groups)

    n_test_target = max(1, int(len(entries) * test_size))
    all_techniques = sorted({t for e in entries for t in e['techniques']})
    technique_total = {t: sum(1 for e in entries if t in e['techniques']) for t in all_techniques}

    test_group_idx = set()
    technique_test_count = {t: 0 for t in all_techniques}

    def group_techs(gi):
        return {t for e in groups[gi] for t in e['techniques']}

    for technique in all_techniques:
        target = min(min_test_per_technique, max(0, technique_total[technique] - 1))
        for gi in range(len(groups)):
            if technique_test_count[technique] >= target:
                break
            if gi in test_group_idx or technique not in group_techs(gi):
                continue
            test_group_idx.add(gi)
            for t in group_techs(gi):
                technique_test_count[t] += len(groups[gi])

    n_test = sum(len(groups[gi]) for gi in test_group_idx)
    for gi in range(len(groups)):
        if n_test >= n_test_target:
            break
        if gi not in test_group_idx:
            test_group_idx.add(gi)
            n_test += len(groups[gi])

    test  = [e for gi in sorted(test_group_idx) for e in groups[gi]]
    train = [e for gi in range(len(groups)) if gi not in test_group_idx for e in groups[gi]]
    return train, test


def load_templates(template_dir=None, extra_techniques=None):
    template_dir = template_dir or TEMPLATE_DIR
    files_by_technique = {}
    for fname in sorted(os.listdir(template_dir)):
        if not fname.endswith('.txt'):
            continue
        technique_id = fname.split('_', 1)[0]
        files_by_technique[technique_id] = fname

    wanted = list(ZOOMER_TECHNIQUES) + list(extra_techniques or [])
    templates = {}
    for technique in wanted:
        fname = files_by_technique.get(technique)
        if fname is None:
            continue
        with open(os.path.join(template_dir, fname), encoding='utf-8') as f:
            templates[technique] = f.read().strip()
    return templates


def encode_one(tokenizer, encoder, text, device):
    enc  = tokenizer(text, padding=False, truncation=False, return_tensors='pt')
    rlen = int(enc['attention_mask'][0].sum())
    ids  = enc['input_ids'][0][:rlen].unsqueeze(0).to(device)
    mask = enc['attention_mask'][0][:rlen].unsqueeze(0).to(device)
    return embed_text(encoder, tokenizer, ids, mask, device, truncate=False, max_chunks=MAX_CHUNKS).squeeze(0)


def encode_sequence_combined(tokenizer, seq_encoder, log_proj, wide_model, entry, step_to_path, bins, masks, device):
    text_embed = log_proj(encode_one(tokenizer, seq_encoder, entry['sequence'], device).unsqueeze(0)).squeeze(0)
    if wide_model is None:
        return text_embed
    wide_vec = wide_vector_for_step(entry['file'], step_to_path, bins, masks)
    wide_t = torch.tensor(wide_vec, dtype=torch.float32, device=device)
    wide_embed = wide_model(wide_t)
    return torch.cat([text_embed, wide_embed], dim=-1)


def build_pos_mask(entries, all_techniques, technique_to_col):
    mask = torch.zeros(len(entries), len(all_techniques))
    for i, e in enumerate(entries):
        for t in e['techniques']:
            mask[i, technique_to_col[t]] = 1
    return mask


def build_technique_pools(entries, all_techniques):
    return {t: [i for i, e in enumerate(entries) if t in e['techniques']] for t in all_techniques}


def stratified_batch_indices(technique_pools, all_techniques, k_per_technique, rng):
    chosen = set()
    for t in all_techniques:
        pool = technique_pools[t]
        if not pool:
            continue
        if len(pool) >= k_per_technique:
            chosen.update(rng.sample(pool, k_per_technique))
        else:
            chosen.update(rng.choices(pool, k=k_per_technique))
    return sorted(chosen)


def run_contrastive_train(n_epochs=N_EPOCHS, test_file_match=None, test_scenario=None, patience=PATIENCE,
                           k_per_technique=K_PER_TACTIC,
                           test_size=0.2, split_seed=SEED, run_tag=None, template_dir=None,
                           stratified=True, lambda_uniform=LAMBDA_UNIFORM, class_reweight=False, reweight_cap=3.0,
                           sequences_dir=None, lr=LR, full_batch=False, n_freeze=N_FREEZE, dropout=DROPOUT,
                           zoomer_scope=False, use_wide=True,
                           loss_fn='contrastive', asl_gamma_pos=1.0, asl_gamma_neg=4.0, asl_clip=0.05,
                           grouped_split=False):
    device = torch.device('cuda')
    print('  Device   : {}'.format(device))
    print()

    entries, n_total_unfiltered = load_sequences(return_total=True, sequences_dir=sequences_dir)
    n_excluded = n_total_unfiltered - len(entries)

    all_techniques = sorted({t for e in entries for t in e['techniques']})
    print('  Techniques (from full true multi-label membership of our {} steps): {}'.format(
        len(entries), ', '.join(all_techniques)))
    if zoomer_scope:
        before = all_techniques
        all_techniques = [t for t in all_techniques if t in ZOOMER_SCOPE_MIN4]
        dropped = sorted(set(before) - set(all_techniques))
        print('  --zoomer-scope: training candidate set narrowed from {} to {} techniques '
              '(dropped: {}) -- same 26-technique scope ZOOMER min4 runs use.'.format(
                  len(before), len(all_techniques), ', '.join(dropped)))
    technique_to_col = {t: i for i, t in enumerate(all_techniques)}

    if run_tag is None:
        if test_scenario:
            run_tag = 'scenario{}'.format(test_scenario)
        elif test_file_match:
            run_tag = test_file_match
        else:
            run_tag = 'seed{}'.format(split_seed)

    if test_scenario:
        train_entries, test_entries = leave_out_scenario_split(entries, test_scenario, seed=split_seed)
        print('  Split mode: leave-one-scenario-out — test = ALL steps of scenario {}'.format(test_scenario))
    elif test_file_match:
        train_entries, test_entries = leave_out_split(entries, test_file_match)
        print('  Split mode: leave-out — test = every file matching "{}"'.format(test_file_match))
    elif stratified and grouped_split:
        train_entries, test_entries = grouped_stratified_split(entries, test_size=test_size, seed=split_seed)
        print('  Split mode: grouped stratified {:.0f}/{:.0f} (split_seed={}, near-duplicate siblings kept '
              'on the same side -- no leakage)'.format((1 - test_size) * 100, test_size * 100, split_seed))
    elif stratified:
        train_entries, test_entries = stratified_split(entries, test_size=test_size, seed=split_seed)
        print('  Split mode: stratified {:.0f}/{:.0f} (split_seed={}, every technique guaranteed >=1 test example)'.format(
            (1 - test_size) * 100, test_size * 100, split_seed))
    else:
        train_entries, test_entries = random_split(entries, test_size=test_size, seed=split_seed)
        print('  Split mode: random {:.0f}/{:.0f} (split_seed={})'.format(
            (1 - test_size) * 100, test_size * 100, split_seed))

    train_entries = [e for e in train_entries if e['techniques']]
    if zoomer_scope:
        scope_set = set(all_techniques)
        for e in train_entries:
            e['techniques'] = [t for t in e['techniques'] if t in scope_set]
        train_entries = [e for e in train_entries if e['techniques']]

    technique_counts = {t: 0 for t in all_techniques}
    for e in train_entries:
        for t in e['techniques']:
            technique_counts[t] += 1
    test_counts = {t: 0 for t in all_techniques}
    for e in test_entries:
        for t in e['techniques']:
            test_counts[t] += 1
    print('  Train sequences: {}  Test sequences: {}  Excluded (not in requested sequence pool): {}'.format(
        len(train_entries), len(test_entries), n_excluded))
    for t in all_techniques:
        print('    {} : {} train  {} test'.format(t, technique_counts[t], test_counts[t]))
    n_in_train = sum(1 for t in all_techniques if technique_counts[t] > 0)
    n_in_test = sum(1 for t in all_techniques if test_counts[t] > 0)
    print('  Techniques with >=1 training example : {} / {}'.format(n_in_train, len(all_techniques)))
    print('  Techniques with >=1 test example      : {} / {}'.format(n_in_test, len(all_techniques)))
    if not train_entries:
        print('  ERROR: no training sequences found.')
        return
    missing = [t for t in all_techniques if technique_counts[t] == 0]
    if missing:
        print('  WARNING: these techniques have zero training examples: {} '
              '(their prototype will only ever be learned as a negative).'.format(missing))

    class_weights = None
    if class_reweight:
        counts = torch.tensor([max(technique_counts[t], 1) for t in all_techniques], dtype=torch.float32, device=device)
        class_weights = (counts.mean() / counts).clamp(max=reweight_cap)
        print('  Class reweighting ON -- per-technique loss weights (mean count / count, capped at {:.1f}x, '
              'rare techniques weighted up):'.format(reweight_cap))
        for t, w in zip(all_techniques, class_weights.tolist()):
            print('    {} : weight={:.2f}'.format(t, w))

    templates = load_templates(template_dir=template_dir)
    print('  Templates: {} (dir={})'.format(', '.join(sorted(templates.keys())), template_dir or TEMPLATE_DIR))

    if use_wide:
        print('  Fitting wide-feature bins/masks on this seed\'s training instances (ZOOMER\'s own pipeline)...')
        step_to_path = build_step_to_graph_path()
        train_steps = [e['file'] for e in train_entries if e['techniques']]
        bins, masks, wide_dim = fit_wide_bins_and_masks(train_steps, step_to_path, seed=split_seed)
        print('  wide_dim={} (raw h_cat + cross-product, before the {}-dim wide projection)'.format(wide_dim, WIDE_DIM_OUT))
    else:
        step_to_path, bins, masks, wide_dim = None, None, None, 0
        print('  --no-wide: text-only embedding, skipping wide-feature bin fitting entirely.')
    print()

    print('  Loading RoBERTa encoders...')
    tokenizer    = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)
    seq_encoder  = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    freeze_lower_layers(seq_encoder, n_freeze=n_freeze)
    seq_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    seq_encoder.train()

    tmpl_encoder = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
    freeze_lower_layers(tmpl_encoder, n_freeze=n_freeze)
    tmpl_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    tmpl_encoder.train()
    combined_dim = COMBINED_DIM if use_wide else PROJ_DIM
    text_proj = ProjectionNetwork(EMB_DIM, combined_dim, dropout).to(device)

    n_layers = len(seq_encoder.encoder.layer)
    print('  Encoders loaded — layers 0-{} frozen, layers {}-{} trainable, gradient checkpointing ON.'.format(
        n_freeze - 1, n_freeze, n_layers - 1))
    print()

    log_proj  = ProjectionNetwork(EMB_DIM, PROJ_DIM, dropout).to(device)
    wide_model = WideModel(wide_dim, WIDE_DIM_OUT).to(device) if use_wide else None

    logit_scale = nn.Parameter(torch.ones([], device=device) * math.log(1 / 0.07))

    trainable_params = ([p for p in seq_encoder.parameters() if p.requires_grad] + list(log_proj.parameters()) +
                         (list(wide_model.parameters()) if use_wide else []) +
                         [p for p in tmpl_encoder.parameters() if p.requires_grad] + list(text_proj.parameters()) +
                         [logit_scale])

    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    scaler = torch.amp.GradScaler('cuda')

    best_loss  = float('inf')
    no_improve = 0
    best_state = None
    history    = []

    train_start = time.time()
    if full_batch:
        batch_size = len(train_entries)
        steps_per_epoch = 1
    else:
        batch_size = min(len(train_entries), k_per_technique * len(all_techniques))
        steps_per_epoch = max(1, -(-len(train_entries) // batch_size))
    print('  Training  epochs={} lr={} dropout={} patience={} batch_size={} full_batch={} n_freeze={}'.format(
        n_epochs, lr, dropout, patience, batch_size, full_batch, n_freeze))
    if loss_fn == 'asl':
        print('  Loss function: Asymmetric Loss (gamma_pos={} gamma_neg={} clip={})'.format(
            asl_gamma_pos, asl_gamma_neg, asl_clip))
    else:
        print('  Loss function: contrastive multilabel_prototype_loss (default)')
    print('  Start time: {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print()

    train_pos_mask = build_pos_mask(train_entries, all_techniques, technique_to_col).to(device)
    technique_pools = build_technique_pools(train_entries, all_techniques)
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        seq_encoder.train(); tmpl_encoder.train(); log_proj.train(); text_proj.train()
        if use_wide:
            wide_model.train()

        epoch_losses = []

        for _ in range(steps_per_epoch):
            if full_batch:
                batch_entries = train_entries
                pos_mask_batch = train_pos_mask
            else:
                batch_idxs = stratified_batch_indices(technique_pools, all_techniques, k_per_technique, random)
                batch_entries = [train_entries[i] for i in batch_idxs]
                pos_mask_batch = train_pos_mask[batch_idxs]

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                z = torch.stack([
                    encode_sequence_combined(tokenizer, seq_encoder, log_proj, wide_model, e,
                                              step_to_path, bins, masks, device)
                    for e in batch_entries
                ])
                prototypes = text_proj(torch.stack(
                    [encode_one(tokenizer, tmpl_encoder, templates[t], device) for t in all_techniques]))
                if loss_fn == 'asl':
                    main_loss = asymmetric_loss(z, prototypes, pos_mask_batch, logit_scale,
                                                 gamma_pos=asl_gamma_pos, gamma_neg=asl_gamma_neg, clip=asl_clip)
                else:
                    main_loss = multilabel_prototype_loss(z, prototypes, pos_mask_batch, logit_scale,
                                                           class_weights=class_weights)
                loss = main_loss + lambda_uniform * uniformity_loss(z)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                logit_scale.clamp_(0, math.log(100))

            epoch_losses.append(loss.item())

        epoch_loss = sum(epoch_losses) / len(epoch_losses)

        history.append({
            'epoch'      : epoch,
            'loss'       : round(epoch_loss, 6),
            'n_steps'    : len(epoch_losses),
            'logit_scale': round(logit_scale.exp().item(), 4),
        })

        epoch_time = (time.time() - epoch_start) / 60
        elapsed    = (time.time() - train_start) / 60
        if epoch % 5 == 0 or epoch == 1:
            print('  epoch {:>4d}  loss={:.6f}  steps={}  logit_scale={:.4f}  epoch_time={:.2f}m  total_time={:.2f}m'.format(
                      epoch, epoch_loss, len(epoch_losses), logit_scale.exp().item(), epoch_time, elapsed))

        if epoch_loss < best_loss - 1e-6:
            best_loss  = epoch_loss
            no_improve = 0
            best_state = {
                'seq_encoder' : {k: v.clone() for k, v in seq_encoder.state_dict().items()},
                'tmpl_encoder': {k: v.clone() for k, v in tmpl_encoder.state_dict().items()},
                'log_proj'    : {k: v.clone() for k, v in log_proj.state_dict().items()},
                'text_proj'   : {k: v.clone() for k, v in text_proj.state_dict().items()},
                'wide_model'  : ({k: v.clone() for k, v in wide_model.state_dict().items()} if use_wide else None),
                'logit_scale' : logit_scale.detach().clone(),
                'epoch'       : epoch,
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                print('  Early stopping at epoch {} (no improvement for {} epochs)'.format(epoch, patience))
                break

        if epoch % 10 == 0 and best_state:
            ckpt_name = 'camlds_technique_wide_{}_epoch{}.pt'.format(run_tag, epoch)
            ckpt_path = os.path.join(OUTPUT_TRAINING, ckpt_name)
            torch.save({
                'seq_encoder' : best_state['seq_encoder'],
                'tmpl_encoder': best_state['tmpl_encoder'],
                'log_proj'    : best_state['log_proj'],
                'text_proj'   : best_state['text_proj'],
                'wide_model'  : best_state['wide_model'],
                'logit_scale' : best_state['logit_scale'],
                'proj_dims'   : (EMB_DIM, PROJ_DIM, WIDE_DIM_OUT if use_wide else 0, wide_dim),
                'techniques'  : all_techniques,
                'best_loss'   : best_loss,
                'history'     : history[:epoch],
                'seed'        : SEED,
                'split_seed'  : split_seed,
                'test_size'   : test_size,
                'run_tag'     : run_tag,
                'test_file_match': test_file_match,
                'test_scenario': test_scenario,
                'template_dir': template_dir,
                'stratified'  : stratified,
                'grouped_split': grouped_split,
                'lambda_uniform': lambda_uniform,
                'class_reweight': class_reweight,
                'reweight_cap': reweight_cap,
                'sequences_dir': sequences_dir,
                'lr': lr,
                'full_batch': full_batch,
                'zoomer_scope': zoomer_scope,
                'use_wide': use_wide,
                'loss_fn': loss_fn,
            }, ckpt_path)
            print('  [checkpoint] {} saved (best_loss={:.6f})'.format(ckpt_name, best_loss))

    if best_state:
        seq_encoder.load_state_dict(best_state['seq_encoder'])
        tmpl_encoder.load_state_dict(best_state['tmpl_encoder'])
        log_proj.load_state_dict(best_state['log_proj'])
        text_proj.load_state_dict(best_state['text_proj'])
        if use_wide:
            wide_model.load_state_dict(best_state['wide_model'])
        with torch.no_grad():
            logit_scale.copy_(best_state['logit_scale'])

    total_time = (time.time() - train_start) / 60
    print()
    print('  Best train loss : {:.6f}'.format(best_loss))
    print('  logit_scale     : {:.4f}'.format(logit_scale.exp().item()))
    print('  End time        : {}'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    print('  Total time      : {:.2f}m'.format(total_time))

    model_path = os.path.join(OUTPUT_TRAINING, 'camlds_technique_wide_matcher_{}.pt'.format(run_tag))
    torch.save({
        'seq_encoder' : seq_encoder.state_dict(),
        'tmpl_encoder': tmpl_encoder.state_dict(),
        'log_proj'    : log_proj.state_dict(),
        'text_proj'   : text_proj.state_dict(),
        'wide_model'  : (wide_model.state_dict() if use_wide else None),
        'logit_scale' : logit_scale,
        'proj_dims'   : (EMB_DIM, PROJ_DIM, WIDE_DIM_OUT if use_wide else 0, wide_dim),
        'techniques'  : all_techniques,
        'best_loss'   : best_loss,
        'best_epoch'  : best_state['epoch'] if best_state else None,
        'history'     : history,
        'seed'        : SEED,
        'split_seed'  : split_seed,
        'test_size'   : test_size,
        'run_tag'     : run_tag,
        'test_file_match': test_file_match,
        'test_scenario': test_scenario,
        'template_dir': template_dir,
        'stratified'  : stratified,
        'grouped_split': grouped_split,
        'lambda_uniform': lambda_uniform,
        'class_reweight': class_reweight,
        'reweight_cap': reweight_cap,
        'sequences_dir': sequences_dir,
        'lr': lr,
        'full_batch': full_batch,
        'zoomer_scope': zoomer_scope,
        'use_wide': use_wide,
        'loss_fn': loss_fn,
    }, model_path)

    hist_path = os.path.join(RESULTS_DIR, 'train_history_camlds_technique_wide_{}.json'.format(run_tag))
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    print()
    print('  Checkpoint    → {}'.format(model_path))
    print('  Train history → {}'.format(hist_path))
    print()
    print('  Run test_camlds_matcher_technique_wide.py --run-tag {} for the detailed per-file test report.'.format(run_tag))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--k-per-technique', type=int, default=K_PER_TACTIC)
    ap.add_argument('--test-file', type=str, default=None)
    ap.add_argument('--test-scenario', type=str, default=None,
                     help='Hold out an ENTIRE scenario as test (e.g. "4"). Overrides --test-file when set.')
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--split-seed', type=int, default=SEED)
    ap.add_argument('--run-tag', type=str, default=None)
    ap.add_argument('--template-dir', type=str, default=None)
    ap.add_argument('--no-stratified', dest='stratified', action='store_false')
    ap.add_argument('--lambda-uniform', type=float, default=LAMBDA_UNIFORM)
    ap.add_argument('--class-reweight', action='store_true')
    ap.add_argument('--reweight-cap', type=float, default=3.0)
    ap.add_argument('--sequences-dir', type=str, default=None)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--full-batch', action='store_true',
                     help='Use ALL training examples every step instead of a small stratified '
                          'subsample -- much more gradient signal per step on small datasets.')
    ap.add_argument('--max-chunks', type=int, default=None,
                     help='Cap how many 512-token chunks any single long sequence contributes '
                          '(a few sequences run to 100k+ tokens and will OOM a full-batch step '
                          'if left uncapped). None = no cap (existing behavior).')
    ap.add_argument('--patience', type=int, default=PATIENCE,
                     help='Epochs to wait for improvement before early-stopping.')
    ap.add_argument('--n-freeze', type=int, default=N_FREEZE,
                     help='How many of RoBERTa\'s 12 layers to freeze (lower = more trainable capacity).')
    ap.add_argument('--dropout', type=float, default=DROPOUT)
    ap.add_argument('--n-epochs', type=int, default=N_EPOCHS)
    ap.add_argument('--zoomer-scope', action='store_true',
                     help="Train on ZOOMER's 26-technique min4 scope (>=4 dataset-wide samples) "
                          'instead of CAM-LDS\'s native 36-technique (>=3 samples) one -- for a '
                          '100%% identical training set between CAM-LDS and ZOOMER. Test time still '
                          'scores against all 36 templates by default (CAM-LDS can zero-shot-match '
                          'an unseen technique via its template text even without fine-tuning on it) '
                          'unless the test script\'s own --zoomer-scope also narrows it to 26.')
    ap.add_argument('--no-wide', dest='use_wide', action='store_false',
                     help='Text-only embedding, no ZOOMER wide (node/edge/IoC count) features '
                          'concatenated in at all -- an ablation to check whether the wide features '
                          'are helping or hurting, independent of the text side.')
    ap.add_argument('--loss', dest='loss_fn', choices=['contrastive', 'asl'], default='contrastive',
                     help="'contrastive' (default) = the existing prototype/logsumexp loss. "
                          "'asl' = Asymmetric Loss for Multi-Label Classification (Ben-Baruch et al. "
                          '2021) -- independent per-class sigmoid with asymmetric focal weighting, '
                          'built for severe per-sample positive/negative imbalance. Default stays '
                          "'contrastive' so omitting this flag reproduces the original behavior exactly.")
    ap.add_argument('--asl-gamma-pos', type=float, default=1.0, help='ASL positive focusing exponent.')
    ap.add_argument('--asl-gamma-neg', type=float, default=4.0, help='ASL negative focusing exponent.')
    ap.add_argument('--asl-clip', type=float, default=0.05, help='ASL negative probability-shift margin.')
    args = ap.parse_args()

    if args.max_chunks is not None:
        MAX_CHUNKS = args.max_chunks

    run_tag = args.run_tag
    if run_tag is None:
        if args.test_scenario:
            run_tag = 'scenario{}'.format(args.test_scenario)
        elif args.test_file:
            run_tag = args.test_file

    run_contrastive_train(test_file_match=args.test_file, test_scenario=args.test_scenario,
                           k_per_technique=args.k_per_technique,
                           test_size=args.test_size, split_seed=args.split_seed, run_tag=run_tag,
                           template_dir=args.template_dir,
                           stratified=args.stratified, lambda_uniform=args.lambda_uniform,
                           class_reweight=args.class_reweight, reweight_cap=args.reweight_cap,
                           lr=args.lr, full_batch=args.full_batch, patience=args.patience,
                           n_freeze=args.n_freeze, dropout=args.dropout, n_epochs=args.n_epochs,
                           sequences_dir=args.sequences_dir, zoomer_scope=args.zoomer_scope,
                           use_wide=args.use_wide, loss_fn=args.loss_fn,
                           asl_gamma_pos=args.asl_gamma_pos, asl_gamma_neg=args.asl_gamma_neg,
                           asl_clip=args.asl_clip)
