import copy
import os
import random
import sys
import torch
import torch.nn.functional as F
from Deep_Wide_Model import TSGModel
from data_utils import GraphTensorCache, K_SHOT, instance_from_filename
from create_data_split import (
    get_zoomer_split_multilabel, get_zoomer_split_scenario_multilabel,
    classes_for_scenario_split, classes_for_fixed_scope,
    CAM_LDS_TECHNIQUE_SCOPE_MIN4,
)
from discretize_features import ALL_DIMS, fit_bins
from cross_product import generate_masks, DEFAULT_K as CROSS_PRODUCT_K
IN_DIM = 126
N_EPISODES = 2000
LR = 0.001
LOG_EVERY = 500
PATIENCE = 4
MIN_DELTA = 0.001
CHECKPOINT_DIR = '/csse/research/contructive-learning/CAM-LDS/zoomer/checkpoints'

def checkpoint_path(run_tag):
    return os.path.join(CHECKPOINT_DIR, 'ttp_recognition_multilabel_{}.pt'.format(run_tag))

def embed_graph(model, cache, path, device):
    (h, adjacency, wide_x) = cache.get(path)
    return model(h.to(device), adjacency.to(device), wide_x.to(device))

MAX_QUERY_PER_CLASS = 2

def support_and_query_size(train_count):
    """Support = K_SHOT (3), always. Query = whatever's left, capped at
    MAX_QUERY_PER_CLASS (2): 1 if only 1 remains, 2 if 2 or more remain --
    never uses ALL of a rich class's leftovers (our per-technique counts
    range from 1 to 16+ after a scenario holdout, very unlike the paper's
    fairly even ~5-6-per-class data, so an uncapped query would let a rich
    class dominate a given episode's compute even with per-class-mean loss)."""
    support_size = min(K_SHOT, max(1, train_count - 1))
    query_size = min(MAX_QUERY_PER_CLASS, train_count - support_size)
    return (support_size, query_size)

def print_sample_scarcity_summary(split, classes):
    groups = {}
    total_train_samples = 0
    total_test_samples = 0
    for t in classes:
        total = len(split[t]['train'])
        (support_size, query_size) = support_and_query_size(total)
        groups.setdefault(query_size, []).append((t, total, support_size, query_size))
        total_train_samples += total
        total_test_samples += len(split[t]['test'])
    print()
    print('Samples summary (K_SHOT={}, MAX_QUERY_PER_CLASS={}):'.format(K_SHOT, MAX_QUERY_PER_CLASS))
    for query_size in sorted(groups):
        entries = groups[query_size]
        label = '{} query sample{}'.format(query_size, '' if query_size == 1 else 's')
        print('  {} class(es) with {} every episode:'.format(len(entries), label))
        for (t, total, support, query) in entries:
            print('    {} (total={}, support={}, query={})'.format(t, total, support, query))
    unique_train = {instance_from_filename(fn) for t in classes for (fn, _) in split[t]['train']}
    unique_test = {instance_from_filename(fn) for t in classes for (fn, _) in split[t]['test']}
    unique_total = unique_train | unique_test
    print()
    print('  Total classes        : {}'.format(len(classes)))
    print('  Total train samples  : {} rows  ({} unique instances)'.format(total_train_samples, len(unique_train)))
    print('  Total test samples   : {} rows  ({} unique instances)'.format(total_test_samples, len(unique_test)))
    print('  Total samples overall: {} rows  ({} unique instances)'.format(total_train_samples + total_test_samples, len(unique_total)))
    print()

def print_orphaned_test_steps(split, classes):
    """Test steps where every true technique got excluded from training -- these will
    still be scored (nothing is hidden), but can only ever come out wrong since no
    prototype exists for any of their real labels."""
    classes_set = set(classes)
    step_techniques = {}
    for (technique, pools) in split.items():
        for (filename, _) in pools['test']:
            inst = instance_from_filename(filename)
            step_techniques.setdefault(inst, set()).add(technique)
    orphaned = sorted((inst, sorted(techs)) for (inst, techs) in step_techniques.items() if not (techs & classes_set))
    print()
    print('  Test steps with NO trained technique (will score as wrong, not hidden): {}/{}'.format(
        len(orphaned), len(step_techniques)))
    for (inst, techs) in orphaned:
        print('    {:20s} true technique(s): {}'.format(inst, techs))
    print()

def print_technique_training_summary(split, classes):
    classes_set = set(classes)
    all_techs = sorted(split.keys())
    skip_techs = sorted(t for t in all_techs if t not in classes_set)

    step_techniques = {}
    for (technique, pools) in split.items():
        for (filename, _) in pools['train']:
            inst = instance_from_filename(filename)
            step_techniques.setdefault(inst, set()).add(technique)
    include_steps = sorted(inst for (inst, techs) in step_techniques.items() if techs & classes_set)
    exclude_steps = sorted(inst for (inst, techs) in step_techniques.items() if not (techs & classes_set))

    print()
    print('-- Technique Training Summary --')
    print('Total techniques             : {}'.format(len(all_techs)))
    print('Include techniques (trained) : {}'.format(len(classes)))
    print('Skip techniques (not trained): {}'.format(len(skip_techs)))
    print()
    print('Total training steps : {}'.format(len(step_techniques)))
    print('Include steps ({}): {}'.format(len(include_steps), ', '.join(include_steps)))
    print('Exclude steps ({}): {}'.format(len(exclude_steps), ', '.join(exclude_steps)))
    print()

def run_episode(model, cache, split, classes, rng, device):
    prototypes = []
    query_items = []
    class_idx = 0
    for technique in classes:
        pool = list(split[technique]['train'])
        if not pool:
            continue
        rng.shuffle(pool)
        (support_size, query_size) = support_and_query_size(len(pool))
        support = pool[:support_size]
        remaining = pool[support_size:support_size + query_size]
        support_embeds = torch.stack([embed_graph(model, cache, path, device) for (_, path) in support])
        prototypes.append(support_embeds.mean(dim=0))
        for (_, path) in remaining:
            query_items.append((path, class_idx))
        class_idx += 1
    prototypes = torch.stack(prototypes)

    by_path = {}
    for (path, class_idx) in query_items:
        by_path.setdefault(path, []).append(class_idx)

    single_items = [(path, cidxs[0]) for (path, cidxs) in by_path.items() if len(cidxs) == 1]
    multi_items = [(path, cidxs) for (path, cidxs) in by_path.items() if len(cidxs) > 1]

    return (prototypes, single_items, multi_items)

def run_full_batch(model, cache, split, classes, device):
    """Full-batch alternative to run_episode(): every available training example
    for every class is embedded and used, instead of a random K_SHOT=3 support /
    2-query subsample reshuffled each episode. Each class's prototype is the mean
    of ALL its training examples; the loss for a given example uses a leave-one-
    out prototype for its own true class (mean of the OTHER examples, so the
    example never trivially matches itself) against the FULL prototypes of every
    other class as distractors. Multi-label graphs are still handled with the
    same per-label exemption as masked_single_label_loss (a graph's other true
    labels are masked out of the distractor set, not pushed away from).
    Returns a single scalar loss covering every training example exactly once."""
    class_paths = {t: [path for (_, path) in split[t]['train']] for t in classes if split[t]['train']}
    kept_classes = [t for t in classes if t in class_paths]
    all_paths = sorted({p for paths in class_paths.values() for p in paths})
    path_embed = {p: embed_graph(model, cache, p, device) for p in all_paths}

    class_embeds = [torch.stack([path_embed[p] for p in class_paths[t]]) for t in kept_classes]
    full_protos = torch.stack([e.mean(dim=0) for e in class_embeds])
    sums = [e.sum(dim=0) for e in class_embeds]
    counts = [e.shape[0] for e in class_embeds]
    n_classes = len(kept_classes)

    path_true_idxs = {}
    for ci, t in enumerate(kept_classes):
        for p in class_paths[t]:
            path_true_idxs.setdefault(p, set()).add(ci)

    losses = []
    for p, true_idxs in path_true_idxs.items():
        embed = path_embed[p]
        for true_idx in true_idxs:
            if counts[true_idx] <= 1:
                continue
            loo_proto = (sums[true_idx] - embed) / (counts[true_idx] - 1)
            protos = full_protos.clone()
            protos[true_idx] = loo_proto
            exempt = true_idxs - {true_idx}
            keep = [i for i in range(n_classes) if i == true_idx or i not in exempt]
            sub_protos = protos[torch.tensor(keep, device=device)]
            dists = torch.cdist(embed.unsqueeze(0), sub_protos) ** 2
            logits = -dists
            target = torch.tensor([keep.index(true_idx)], device=device)
            losses.append((true_idx, F.cross_entropy(logits, target)))

    by_class = {}
    for (true_idx, loss) in losses:
        by_class.setdefault(true_idx, []).append(loss)
    class_means = [torch.stack(class_losses).mean() for class_losses in by_class.values()]
    return torch.stack(class_means).mean()


def ttp_recognition_loss(prototypes, query_embeds, query_labels):
    """Per-class-mean cross-entropy: every class's query batch is averaged on
    its own first, then those per-class averages are averaged together -- so a
    class with many leftover query samples doesn't dominate the gradient over
    a class with only 1 or 2 (our per-technique sample counts vary a lot after
    a scenario holdout, unlike the paper's fairly even ~5-6-per-class data)."""
    dists = torch.cdist(query_embeds, prototypes) ** 2
    logits = -dists
    per_example_loss = F.cross_entropy(logits, query_labels, reduction='none')
    class_means = []
    for cidx in query_labels.unique():
        class_means.append(per_example_loss[query_labels == cidx].mean())
    return torch.stack(class_means).mean()

def masked_single_label_loss(embed, prototypes, true_idx, exempt_idxs):
    """Cross-entropy of one query embedding against a subset of prototypes: the true
    class plus every class NOT in exempt_idxs. exempt_idxs holds this same graph's
    OTHER true labels, which must never be pushed away from as if they were wrong."""
    n = prototypes.shape[0]
    keep = [i for i in range(n) if i == true_idx or i not in exempt_idxs]
    sub_protos = prototypes[torch.tensor(keep, device=embed.device)]
    dists = torch.cdist(embed.unsqueeze(0), sub_protos) ** 2
    logits = -dists
    target = torch.tensor([keep.index(true_idx)], device=embed.device)
    return F.cross_entropy(logits, target)

def main(seed, scenario=None, test_file=None, run_tag=None, min_train_samples=None, fixed_scope=False, fixed_scope_min4=False, full_batch=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    held_out = test_file or scenario
    if run_tag is None:
        run_tag = test_file if test_file else ('scenario{}'.format(scenario) if scenario else 'seed{}'.format(seed))
    if held_out:
        print('[multilabel] Held out: {}  Seed (bins/masks RNG): {}  Device: {}'.format(held_out, seed, device))
    else:
        print('[multilabel] Seed: {}  Device: {}'.format(seed, device))
    torch.manual_seed(seed)

    split = get_zoomer_split_scenario_multilabel(scenario, test_file=test_file) if held_out else get_zoomer_split_multilabel(seed)
    if held_out:
        if fixed_scope_min4:
            classes = classes_for_fixed_scope(split, scope=CAM_LDS_TECHNIQUE_SCOPE_MIN4)
        elif fixed_scope:
            classes = classes_for_fixed_scope(split)
        elif min_train_samples is not None:
            classes = classes_for_scenario_split(split, min_train_samples=min_train_samples)
        else:
            classes = classes_for_scenario_split(split)
    elif fixed_scope_min4:
        classes = classes_for_fixed_scope(split, scope=CAM_LDS_TECHNIQUE_SCOPE_MIN4)
    elif fixed_scope:
        classes = classes_for_fixed_scope(split)
    else:
        classes = sorted(split.keys())
    print('Training classes (techniques): {}'.format(len(classes)))
    for t in classes:
        print('  {:14s} train={} test={}'.format(t, len(split[t]['train']), len(split[t]['test'])))
    print_sample_scarcity_summary(split, classes)
    if held_out:
        print_technique_training_summary(split, classes)
        print_orphaned_test_steps(split, classes)

    train_paths = sorted({path for t in classes for (_, path) in split[t]['train']})
    print('Fitting k-means bins on {} training graphs (seed={})...'.format(len(train_paths), seed))
    bins = fit_bins(train_paths)
    h_cat_dim = sum(len(bins[dim]) for dim in ALL_DIMS)
    masks = generate_masks(h_cat_dim, seed=seed)
    wide_in_dim = h_cat_dim + CROSS_PRODUCT_K
    print('h_cat dim: {}  wide input dim (h_cat + cross-product): {}'.format(h_cat_dim, wide_in_dim))

    model = TSGModel(deep_in_dim=IN_DIM, wide_in_dim=wide_in_dim).to(device)
    deep_out_dim = model.deep_model.layers[-1].head_fc[0].out_features * model.deep_model.layers[-1].n_heads
    wide_out_dim = model.wide_model.linear.out_features
    print('h_deep dim: {}  h_wide dim: {}  h_TSG dim (h_wide + h_deep): {}'.format(
        deep_out_dim, wide_out_dim, wide_out_dim + deep_out_dim))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    cache = GraphTensorCache(bins, masks)
    rng = random.Random(seed)
    best_loss = float('inf')
    best_state = None
    stale_windows = 0
    recent_losses = []
    unit = 'epoch' if full_batch else 'episode'
    for episode in range(1, N_EPISODES + 1):
        if full_batch:
            loss = run_full_batch(model, cache, split, classes, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            episode_loss = loss.item()
        else:
            (prototypes, single_items, multi_items) = run_episode(model, cache, split, classes, rng, device)

            episode_loss = 0.0
            if single_items:
                query_embeds = torch.stack([embed_graph(model, cache, path, device) for (path, _) in single_items])
                query_labels = torch.tensor([cidx for (_, cidx) in single_items], device=device)
                loss = ttp_recognition_loss(prototypes, query_embeds, query_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                episode_loss += loss.item()

            protos_fixed = prototypes.detach()
            for (path, cidxs) in multi_items:
                for true_idx in cidxs:
                    exempt = set(cidxs) - {true_idx}
                    embed = embed_graph(model, cache, path, device)
                    step_loss = masked_single_label_loss(embed, protos_fixed, true_idx, exempt)
                    optimizer.zero_grad()
                    step_loss.backward()
                    optimizer.step()
                    episode_loss += step_loss.item()

        recent_losses.append(episode_loss)
        if episode % LOG_EVERY == 0 or episode == 1:
            avg_loss = sum(recent_losses) / len(recent_losses)
            recent_losses = []
            if avg_loss < best_loss - MIN_DELTA:
                best_loss = avg_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_windows = 0
            else:
                stale_windows += 1
            print('{} {:5d}  loss {:.4f}  best_loss {:.4f}'.format(unit, episode, episode_loss, best_loss))
            if stale_windows >= PATIENCE:
                print("loss hasn't improved for {} checks (best={:.4f}) -- stopping early at {} {}.".format(PATIENCE, best_loss, unit, episode))
                break
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    final_state = best_state if best_state is not None else model.state_dict()
    out_path = checkpoint_path(run_tag)
    torch.save({'model': final_state, 'classes': classes, 'seed': seed, 'scenario': scenario,
                'test_file': test_file, 'run_tag': run_tag, 'wide_in_dim': wide_in_dim}, out_path)
    print('Saved -> {}'.format(out_path))
    return out_path

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scenario', type=str, default=None,
                     help='Hold out this whole scenario (e.g. "4") instead of a random split.')
    ap.add_argument('--test-file', type=str, default=None,
                     help='Hold out only instances whose step id CONTAINS this substring '
                          '(e.g. "2_rootkit" or "6_macro_binary"), instead of the whole numbered '
                          'scenario -- mirrors CAM-LDS\'s --test-file exactly. Takes priority over '
                          '--scenario if both are passed.')
    ap.add_argument('--run-tag', type=str, default=None)
    ap.add_argument('--min-train-samples', type=int, default=None,
                     help='Override MIN_TRAIN_SAMPLES_SCENARIO (default 2) for scenario mode -- '
                          'how many training examples a technique needs left after the holdout to get trained.')
    ap.add_argument('--fixed-scope', action='store_true',
                     help="Use CAM-LDS's own fixed 36 base-technique scope directly (ignores "
                          "--min-train-samples) -- trains the exact same technique classes as "
                          "the CAM-LDS text+wide side, for a true side-by-side comparison.")
    ap.add_argument('--fixed-scope-min4', action='store_true',
                     help="Like --fixed-scope, but the 26-technique scope built from a >=4 "
                          "dataset-wide sample threshold instead of >=3 (drops 11 thin techniques, "
                          "picks up T1204). Takes priority over --fixed-scope if both are passed.")
    ap.add_argument('--full-batch', action='store_true',
                     help='Use every training example every epoch instead of a random K_SHOT=3 '
                          'support / 2-query subsample reshuffled each episode -- prototypes are '
                          'the mean of ALL training examples for a class, loss is leave-one-out. '
                          'No subsampling randomness; needs more GPU memory per step.')
    args = ap.parse_args()
    run_tag = args.run_tag
    held_out_label = args.test_file or args.scenario
    if run_tag is None and held_out_label:
        if args.fixed_scope_min4:
            run_tag = '{}_camldsscope_min4'.format(held_out_label)
        elif args.fixed_scope:
            run_tag = '{}_camldsscope'.format(held_out_label)
        elif args.min_train_samples is not None:
            run_tag = '{}_min{}'.format(held_out_label, args.min_train_samples)
        if args.full_batch:
            run_tag = (run_tag or held_out_label) + '_fullbatch'
    main(args.seed, scenario=args.scenario, test_file=args.test_file, run_tag=run_tag,
         min_train_samples=args.min_train_samples, fixed_scope=args.fixed_scope,
         fixed_scope_min4=args.fixed_scope_min4, full_batch=args.full_batch)
