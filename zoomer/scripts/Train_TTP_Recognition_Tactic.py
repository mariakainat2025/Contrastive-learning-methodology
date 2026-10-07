import copy
import os
import random
import sys
import torch
import torch.nn.functional as F
from Deep_Wide_Model import TSGModel
from data_utils import GraphTensorCache, K_SHOT, instance_from_filename
from create_data_split import get_zoomer_split_tactic, get_zoomer_split_scenario_tactic, classes_for_scenario_split
from discretize_features import ALL_DIMS, fit_bins
from cross_product import generate_masks, DEFAULT_K as CROSS_PRODUCT_K
IN_DIM = 126
N_EPISODES = 2000
LR = 0.001
MAX_QUERY_PER_CLASS = 2
LOG_EVERY = 500
PATIENCE = 4
MIN_DELTA = 0.001
CHECKPOINT_DIR = '/csse/research/contructive-learning/CAM-LDS/zoomer/checkpoints'

def checkpoint_path(run_tag):
    return os.path.join(CHECKPOINT_DIR, 'ttp_recognition_tactic_{}.pt'.format(run_tag))

def embed_graph(model, cache, path, device):
    (h, adjacency, wide_x) = cache.get(path)
    return model(h.to(device), adjacency.to(device), wide_x.to(device))

def support_and_query_size(train_count):
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
        if query_size == MAX_QUERY_PER_CLASS:
            label = 'full {} query samples'.format(MAX_QUERY_PER_CLASS)
        else:
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
    """Test steps where every true tactic got excluded from training -- these will
    still be scored (nothing is hidden), but can only ever come out wrong since no
    prototype exists for any of their real labels."""
    classes_set = set(classes)
    step_tactics = {}
    for (tactic, pools) in split.items():
        for (filename, _) in pools['test']:
            inst = instance_from_filename(filename)
            step_tactics.setdefault(inst, set()).add(tactic)
    orphaned = sorted((inst, sorted(tacs)) for (inst, tacs) in step_tactics.items() if not (tacs & classes_set))
    print()
    print('  Test steps with NO trained tactic (will score as wrong, not hidden): {}/{}'.format(
        len(orphaned), len(step_tactics)))
    for (inst, tacs) in orphaned:
        print('    {:20s} true tactic(s): {}'.format(inst, tacs))
    print()

def print_tactic_training_summary(split, classes):
    classes_set = set(classes)
    all_tacs = sorted(split.keys())
    skip_tacs = sorted(t for t in all_tacs if t not in classes_set)

    step_tactics = {}
    for (tactic, pools) in split.items():
        for (filename, _) in pools['train']:
            inst = instance_from_filename(filename)
            step_tactics.setdefault(inst, set()).add(tactic)
    include_steps = sorted(inst for (inst, tacs) in step_tactics.items() if tacs & classes_set)
    exclude_steps = sorted(inst for (inst, tacs) in step_tactics.items() if not (tacs & classes_set))

    print()
    print('-- Tactic Training Summary --')
    print('Total tactics             : {}'.format(len(all_tacs)))
    print('Include tactics (trained) : {}'.format(len(classes)))
    print('Skip tactics (not trained): {}'.format(len(skip_tacs)))
    print()
    print('Total training steps : {}'.format(len(step_tactics)))
    print('Include steps ({}): {}'.format(len(include_steps), ', '.join(include_steps)))
    print('Exclude steps ({}): {}'.format(len(exclude_steps), ', '.join(exclude_steps)))
    print()

def run_episode(model, cache, split, classes, rng, device):
    prototypes = []
    query_items = []
    for (class_idx, tactic) in enumerate(classes):
        pool = list(split[tactic]['train'])
        rng.shuffle(pool)
        (support_size, query_size) = support_and_query_size(len(pool))
        support = pool[:support_size]
        remaining = pool[support_size:support_size + query_size]
        support_embeds = torch.stack([embed_graph(model, cache, path, device) for (_, path) in support])
        prototypes.append(support_embeds.mean(dim=0))
        for (_, path) in remaining:
            query_items.append((path, class_idx))
    prototypes = torch.stack(prototypes)

    by_path = {}
    for (path, class_idx) in query_items:
        by_path.setdefault(path, []).append(class_idx)

    # a graph drawn as a query point under only one class this episode -- ordinary single-label case
    single_items = [(path, cidxs[0]) for (path, cidxs) in by_path.items() if len(cidxs) == 1]
    # a graph drawn as a query point under 2+ classes this episode (it genuinely has multiple true
    # labels) -- trained sequentially below, one label at a time, not batched together
    multi_items = [(path, cidxs) for (path, cidxs) in by_path.items() if len(cidxs) > 1]

    return (prototypes, single_items, multi_items)

def ttp_recognition_loss(prototypes, query_embeds, query_labels):
    dists = torch.cdist(query_embeds, prototypes) ** 2
    logits = -dists
    return F.cross_entropy(logits, query_labels)

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

def main(seed, scenario=None, run_tag=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if run_tag is None:
        run_tag = 'scenario{}'.format(scenario) if scenario else 'seed{}'.format(seed)
    if scenario:
        print('[tactic] Scenario held out: {}  Seed (bins/masks RNG): {}  Device: {}'.format(scenario, seed, device))
    else:
        print('[tactic] Seed: {}  Device: {}'.format(seed, device))
    torch.manual_seed(seed)

    split = get_zoomer_split_scenario_tactic(scenario) if scenario else get_zoomer_split_tactic(seed)
    classes = classes_for_scenario_split(split) if scenario else sorted(split.keys())
    print('Training classes (tactics): {}'.format(len(classes)))
    for t in classes:
        print('  {:24s} train={} test={}'.format(t, len(split[t]['train']), len(split[t]['test'])))
    print_sample_scarcity_summary(split, classes)
    if scenario:
        print_tactic_training_summary(split, classes)
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
    for episode in range(1, N_EPISODES + 1):
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

        # sequential, one label at a time, per multi-label query graph -- prototypes are
        # detached here since they've already been used in the backward pass above (and,
        # unlike the main step, these extra updates only refine the query path, not the
        # support/prototype path)
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
            print('episode {:5d}  loss {:.4f}  best_loss {:.4f}'.format(episode, episode_loss, best_loss))
            if stale_windows >= PATIENCE:
                print("loss hasn't improved for {} checks (best={:.4f}) -- stopping early at episode {}.".format(PATIENCE, best_loss, episode))
                break
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    final_state = best_state if best_state is not None else model.state_dict()
    out_path = checkpoint_path(run_tag)
    torch.save({'model': final_state, 'classes': classes, 'seed': seed, 'scenario': scenario,
                'run_tag': run_tag, 'wide_in_dim': wide_in_dim}, out_path)
    print('Saved -> {}'.format(out_path))
    return out_path

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--scenario', type=str, default=None,
                     help='Hold out this whole scenario (e.g. "4") instead of a random split.')
    ap.add_argument('--run-tag', type=str, default=None)
    args = ap.parse_args()
    main(args.seed, scenario=args.scenario, run_tag=args.run_tag)
