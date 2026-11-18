import argparse
import importlib.util
import os
import random
import re
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAM_LDS_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if CAM_LDS_DIR not in sys.path:
    sys.path.insert(0, CAM_LDS_DIR)

from train_camlds_matcher_technique_wide import ROBERTA_MODEL, encode_one, load_templates, TEMPLATE_DIR
from combine_test_pipeline import _load_knowhow_fasttext, make_knowhow_vectorizer


def load_technique_names(template_dir=None):
    template_dir = template_dir or TEMPLATE_DIR
    names = {}
    for fname in sorted(os.listdir(template_dir)):
        if not fname.endswith('.txt'):
            continue
        (technique_id, _, rest) = fname.partition('_')
        names[technique_id] = rest[:-len('.txt')].replace('_', ' ')
    return names


def _load_zoomer_data_utils():
    # technique/scripts/ and zoomer/scripts/ both have a data_utils.py; whichever gets
    # imported as plain "data_utils" first wins and gets cached under that name for the
    # whole process (combine_test_pipeline's own imports already claim it). Load zoomer's
    # copy by exact file path under a distinct module name instead, sidestepping the clash.
    # data_utils.py itself does bare "from discretize_features import ..." etc, so
    # zoomer/scripts needs to be on sys.path while it executes.
    zoomer_scripts_dir = os.path.join(CAM_LDS_DIR, 'zoomer', 'scripts')
    added = zoomer_scripts_dir not in sys.path
    if added:
        sys.path.append(zoomer_scripts_dir)
    try:
        path = os.path.join(zoomer_scripts_dir, 'data_utils.py')
        spec = importlib.util.spec_from_file_location('zoomer_data_utils_for_similarity_check', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if added:
            sys.path.remove(zoomer_scripts_dir)


_zoomer_data_utils = _load_zoomer_data_utils()
load_tactic_map = _zoomer_data_utils.load_tactic_map
technique_tactics = _zoomer_data_utils.technique_tactics


def pairwise_cosine_sims(vecs):
    keys = list(vecs.keys())
    sims = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (a, b) = (vecs[keys[i]], vecs[keys[j]])
            sims[(keys[i], keys[j])] = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return sims


def pick_same_diff_tactic_pairs(techs, n, seed=0):
    tactic_map = load_tactic_map()
    tactics_of = {t: technique_tactics(tactic_map, t) for t in techs}
    usable = [t for t in techs if tactics_of[t]]
    excluded = sorted(set(techs) - set(usable))
    if excluded:
        print('Excluded (no tactic mapping found): {}'.format(excluded))

    (same_pairs, diff_pairs) = ([], [])
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            (a, b) = (usable[i], usable[j])
            if tactics_of[a] & tactics_of[b]:
                same_pairs.append((a, b))
            else:
                diff_pairs.append((a, b))

    rng = random.Random(seed)
    rng.shuffle(same_pairs)
    rng.shuffle(diff_pairs)
    if len(same_pairs) < n:
        raise SystemExit('Only found {} same-tactic pairs among these techniques, need {}.'.format(len(same_pairs), n))
    if len(diff_pairs) < n:
        raise SystemExit('Only found {} different-tactic pairs among these techniques, need {}.'.format(len(diff_pairs), n))
    return (same_pairs[:n], diff_pairs[:n], tactics_of)


def main():
    ap = argparse.ArgumentParser(
        description='Compare how similar MITRE technique templates look to RoBERTa (pretrained, '
                    'untrained) vs KnowHow FastText, before any fine-tuning. Measures whether one '
                    'encoder starts with templates clustered close together (hard to tell apart) vs '
                    'spread apart (already distinguishable) -- the starting point an encoder has to '
                    'work from during contrastive training.')
    ap.add_argument('--techniques', type=str, default=None,
                     help='Comma-separated technique IDs to compare (e.g. "T1003,T1016,T1059"). '
                          'Default: all templates in technique/templates_all92.')
    ap.add_argument('--top', type=int, default=10,
                     help='How many most- and least-similar RoBERTa pairs to print (default 10). '
                          'Ignored when --same-diff-tactic is set.')
    ap.add_argument('--same-diff-tactic', type=int, default=None, metavar='N',
                     help='Instead of the full pairwise sweep, pick N technique pairs that share a '
                          'tactic and N pairs that share no tactic (tactics from folder_tactic_map.json, '
                          'same mapping ZOOMER\'s own split code uses), and compare only those 2*N pairs. '
                          'Tests whether the encoder\'s similarity score tracks real tactical relatedness '
                          'or is just uniformly high/low regardless of it.')
    ap.add_argument('--seed', type=int, default=0,
                     help='Random seed for picking which pairs to sample under --same-diff-tactic.')
    ap.add_argument('--checkpoint', type=str, default=None,
                     help='Path to a trained checkpoint_combinations/*.pt file for a template=mymodel '
                          '(RoBERTa) combo (e.g. camlds_technique_wide_matcher_mymodel_mymodel_mymodel_'
                          'k1_raw.pt). Loads its trained tmpl_encoder weights instead of the fresh '
                          'pretrained RoBERTa, so you can compare template similarity AFTER training vs '
                          'the untrained baseline.')
    ap.add_argument('--knowhow-checkpoint', type=str, default=None,
                     help='Path to a trained checkpoint_combinations/*.pt file for a template=knowhow '
                          'combo (e.g. knowhow_camlds_camlds_matcher_knowhow_mymodel_mymodel_k1_raw.pt). '
                          'Loads its trained template_adapter and applies it on top of the raw FastText '
                          'average, so FastText is compared on the same basis as --checkpoint does for '
                          'RoBERTa: the actual final embedding used downstream, not just the raw vector.')
    args = ap.parse_args()

    templates = load_templates()
    techs = args.techniques.split(',') if args.techniques else sorted(templates.keys())
    missing = [t for t in techs if t not in templates]
    if missing:
        raise SystemExit('Unknown technique id(s), not found in templates: {}'.format(missing))

    if args.same_diff_tactic:
        (same_pairs, diff_pairs, tactics_of) = pick_same_diff_tactic_pairs(
            techs, args.same_diff_tactic, seed=args.seed)
        needed = sorted({t for pair in (same_pairs + diff_pairs) for t in pair})
        print('Same-tactic pairs picked: {}'.format(same_pairs))
        print('Different-tactic pairs picked: {}'.format(diff_pairs))
        print()
    else:
        needed = techs
        print('Comparing {} templates.'.format(len(techs)))
        print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL)

    roberta_before = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device).eval()
    with torch.no_grad():
        rob_vecs_before = {t: encode_one(tokenizer, roberta_before, templates[t], device) for t in needed}

    rob_vecs_after = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'tmpl_encoder' not in ckpt:
            raise SystemExit('Checkpoint has no \'tmpl_encoder\' weights -- is this a template=mymodel '
                             '(RoBERTa) combo checkpoint? Keys found: {}'.format(sorted(ckpt.keys())))
        roberta_after = RobertaModel.from_pretrained(ROBERTA_MODEL).to(device)
        roberta_after.load_state_dict(ckpt['tmpl_encoder'])
        roberta_after.eval()
        with torch.no_grad():
            rob_vecs_after = {t: encode_one(tokenizer, roberta_after, templates[t], device) for t in needed}
        print('Loaded TRAINED template encoder from: {}  (run_tag={})'.format(
            args.checkpoint, ckpt.get('run_tag')))
        print()

    fasttext_model = _load_knowhow_fasttext()
    ft_vec_fn = make_knowhow_vectorizer(fasttext_model)
    ft_vecs_before = {t: ft_vec_fn(t, templates[t]) for t in needed}

    ft_vecs_after = None
    if args.knowhow_checkpoint:
        kh_ckpt = torch.load(args.knowhow_checkpoint, map_location=device, weights_only=False)
        if 'template_adapter' not in kh_ckpt:
            raise SystemExit('Checkpoint has no \'template_adapter\' weights -- is this a template=knowhow '
                             'combo checkpoint? Keys found: {}'.format(sorted(kh_ckpt.keys())))
        adapter_weight = kh_ckpt['template_adapter']['weight']
        (out_dim, in_dim) = adapter_weight.shape
        template_adapter = nn.Linear(in_dim, out_dim).to(device)
        template_adapter.load_state_dict(kh_ckpt['template_adapter'])
        template_adapter.eval()
        with torch.no_grad():
            ft_vecs_after = {t: template_adapter(v.to(device)) for (t, v) in ft_vecs_before.items()}
        print('Loaded TRAINED template_adapter from: {}  (run_tag={})'.format(
            args.knowhow_checkpoint, kh_ckpt.get('run_tag')))
        print()

    if args.same_diff_tactic:
        names = load_technique_names()

        def cos(vecs, a, b):
            return F.cosine_similarity(vecs[a].unsqueeze(0), vecs[b].unsqueeze(0)).item()

        def word_overlap(a, b):
            words_a = set(re.findall(r"[a-zA-Z']+", templates[a].lower()))
            words_b = set(re.findall(r"[a-zA-Z']+", templates[b].lower()))
            shared = words_a & words_b
            return (len(words_a), len(words_b), len(shared), sorted(shared))

        def report_group(label, pairs):
            print('{} pairs:'.format(label))
            for (a, b) in pairs:
                rv_before = cos(rob_vecs_before, a, b)
                fv_before = cos(ft_vecs_before, a, b)
                (n_a, n_b, n_shared, shared_words) = word_overlap(a, b)
                print('{}'.format('-'.join((a, b))))
                if rob_vecs_after is not None or ft_vecs_after is not None:
                    rv_after = cos(rob_vecs_after, a, b) if rob_vecs_after is not None else None
                    fv_after = cos(ft_vecs_after, a, b) if ft_vecs_after is not None else None
                    print('  RoBERTa:  before={:.4f}  after={}'.format(
                        rv_before, '{:.4f}'.format(rv_after) if rv_after is not None else 'n/a (no --checkpoint)'))
                    print('  FastText: before={:.4f}  after={}'.format(
                        fv_before, '{:.4f}'.format(fv_after) if fv_after is not None else 'n/a (no --knowhow-checkpoint)'))
                else:
                    print('  RoBERTa={:.4f}  FastText={:.4f}'.format(rv_before, fv_before))
                print('  {}: {}  [{}]'.format(a, names.get(a, '?'), ', '.join(sorted(tactics_of[a])) or 'none'))
                print('  {}: {}  [{}]'.format(b, names.get(b, '?'), ', '.join(sorted(tactics_of[b])) or 'none'))
                print('  Total words: {}={}  {}={}'.format(a, n_a, b, n_b))
                print('  Total shared words: {} ({})'.format(
                    n_shared, ', '.join(shared_words[:12]) + ('...' if len(shared_words) > 12 else '')))
                print()

        report_group('Same-tactic', same_pairs)
        report_group('Different-tactic', diff_pairs)
        print('If an encoder tracks real tactical relatedness, same-tactic similarity should be clearly '
              'higher than different-tactic. If both are close (or different-tactic is just as high), '
              'the encoder isn\'t distinguishing by meaning -- it\'s just generically collapsed.')
        return

    rob_sims = pairwise_cosine_sims(rob_vecs)
    ft_sims = pairwise_cosine_sims(ft_vecs)

    print('{} pairs compared.'.format(len(rob_sims)))
    print()
    print('{:12s} {:>10s} {:>10s} {:>10s}'.format('', 'mean', 'min', 'max'))
    print('{:12s} {:10.4f} {:10.4f} {:10.4f}'.format(
        'RoBERTa', statistics.mean(rob_sims.values()), min(rob_sims.values()), max(rob_sims.values())))
    print('{:12s} {:10.4f} {:10.4f} {:10.4f}'.format(
        'FastText', statistics.mean(ft_sims.values()), min(ft_sims.values()), max(ft_sims.values())))
    print()
    print('Higher = the encoder sees the two templates as more alike (before any training).')
    print()

    ranked = sorted(rob_sims.items(), key=lambda kv: kv[1], reverse=True)
    top_n = min(args.top, len(ranked))

    print('Most similar pairs under RoBERTa (top {}):'.format(top_n))
    print('{:24s} {:>10s} {:>10s}'.format('pair', 'RoBERTa', 'FastText'))
    for (pair, sim) in ranked[:top_n]:
        print('{:24s} {:10.4f} {:10.4f}'.format('-'.join(pair), sim, ft_sims[pair]))
    print()

    print('Least similar pairs under RoBERTa (bottom {}):'.format(top_n))
    print('{:24s} {:>10s} {:>10s}'.format('pair', 'RoBERTa', 'FastText'))
    for (pair, sim) in ranked[-top_n:]:
        print('{:24s} {:10.4f} {:10.4f}'.format('-'.join(pair), sim, ft_sims[pair]))


if __name__ == '__main__':
    main()
