import copy
import random
import torch
import torch.nn.functional as F
from Deep_Model import DeepModel
from data_utils import GraphTensorCache, K_SHOT, load_split, instance_from_filename
IN_DIM = 126
N_EPISODES = 2000
LR = 0.001
MAX_QUERY_PER_CLASS = 2
LOG_EVERY = 500
PATIENCE = 4
MIN_DELTA = 0.001
CHECKPOINT_PATH = '/csse/research/contructive-learning/CAM-LDS/zoomer/checkpoints/ttp_recognition.pt'

def embed_graph(model, cache, path, device):
    (h, adjacency) = cache.get(path)
    return model(h.to(device), adjacency.to(device))

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

def run_episode(model, cache, split, classes, rng, device):
    prototypes = []
    query_embeds = []
    query_labels = []
    for (class_idx, technique) in enumerate(classes):
        pool = list(split[technique]['train'])
        rng.shuffle(pool)
        (support_size, query_size) = support_and_query_size(len(pool))
        support = pool[:support_size]
        remaining = pool[support_size:support_size + query_size]
        support_embeds = torch.stack([embed_graph(model, cache, path, device) for (_, path) in support])
        prototypes.append(support_embeds.mean(dim=0))
        for (_, path) in remaining:
            query_embeds.append(embed_graph(model, cache, path, device))
            query_labels.append(class_idx)
    prototypes = torch.stack(prototypes)
    query_embeds = torch.stack(query_embeds)
    query_labels = torch.tensor(query_labels, device=device)
    return (prototypes, query_embeds, query_labels)

def ttp_recognition_loss(prototypes, query_embeds, query_labels):
    dists = torch.cdist(query_embeds, prototypes) ** 2
    logits = -dists
    return F.cross_entropy(logits, query_labels)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))
    split = load_split()
    classes = sorted(split.keys())
    print('Training classes (techniques): {}'.format(len(classes)))
    for t in classes:
        print('  {:14s} train={} test={}'.format(t, len(split[t]['train']), len(split[t]['test'])))
    print_sample_scarcity_summary(split, classes)
    model = DeepModel(in_dim=IN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    cache = GraphTensorCache()
    rng = random.Random(1)
    best_loss = float('inf')
    best_state = None
    stale_windows = 0
    recent_losses = []
    for episode in range(1, N_EPISODES + 1):
        (prototypes, query_embeds, query_labels) = run_episode(model, cache, split, classes, rng, device)
        loss = ttp_recognition_loss(prototypes, query_embeds, query_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        recent_losses.append(loss.item())
        if episode % LOG_EVERY == 0 or episode == 1:
            avg_loss = sum(recent_losses) / len(recent_losses)
            recent_losses = []
            if avg_loss < best_loss - MIN_DELTA:
                best_loss = avg_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_windows = 0
            else:
                stale_windows += 1
            print('episode {:5d}  loss {:.4f}  best_loss {:.4f}'.format(episode, loss.item(), best_loss))
            if stale_windows >= PATIENCE:
                print("loss hasn't improved for {} checks (best={:.4f}) -- stopping early at episode {}.".format(PATIENCE, best_loss, episode))
                break
    import os
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    final_state = best_state if best_state is not None else model.state_dict()
    torch.save({'model': final_state, 'classes': classes}, CHECKPOINT_PATH)
    print('Saved -> {}'.format(CHECKPOINT_PATH))
if __name__ == '__main__':
    main()
