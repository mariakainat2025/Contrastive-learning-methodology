import sys
import numpy as np
import torch
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score
from Deep_Model import DeepModel
from data_utils import GraphTensorCache, load_split, instance_from_filename, load_tactic_map, instance_tactics, load_technique_tactics
_CAM_LDS_SCRIPTS = '/csse/research/contructive-learning/CAM-LDS/scripts'
if _CAM_LDS_SCRIPTS not in sys.path:
    sys.path.insert(0, _CAM_LDS_SCRIPTS)
from train_camlds_matcher import TACTIC_IDS
IN_DIM = 126
CHECKPOINT_PATH = '/csse/research/contructive-learning/CAM-LDS/zoomer/checkpoints/ttp_recognition.pt'

def embed_graph(model, cache, path, device):
    (h, adjacency) = cache.get(path)
    with torch.no_grad():
        return model(h.to(device), adjacency.to(device))

def build_final_prototypes(model, cache, split, classes, device):
    prototypes = []
    for technique in classes:
        paths = [path for (_, path) in split[technique]['train']]
        embeds = torch.stack([embed_graph(model, cache, p, device) for p in paths])
        prototypes.append(embeds.mean(dim=0))
    return torch.stack(prototypes)

def print_results_table(rows, n_classes, top_n=3):
    col_file = 32
    col_true = 24
    col_score = 18
    print()
    print('  -- Test Results (scored against {} technique prototypes) --'.format(n_classes))
    header_fmt = '  {:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} ' + ' '.join(['{:<' + str(col_score) + '}'] * top_n) + ' {:<8}'
    print(header_fmt.format('#', 'File', 'True', *['#{} (score)'.format(j + 1) for j in range(top_n)], 'Top1'))
    print('  ' + '-' * (4 + col_file + col_true + col_score * top_n + 8 + top_n + 3))
    n_wrong = 0
    for (i, r) in enumerate(rows, 1):
        ranked = r['ranked']
        cols = []
        for j in range(top_n):
            cols.append('{} {:.4f}'.format(*ranked[j]) if j < len(ranked) else '')
        fname = r['file'] if len(r['file']) <= col_file else r['file'][:col_file - 3] + '...'
        true_str = r['true_technique']
        if len(true_str) > col_true:
            true_str = true_str[:col_true - 3] + '...'
        top1_wrong = ranked[0][0] not in r['true_techniques']
        if top1_wrong:
            n_wrong += 1
        print(header_fmt.format(i, fname, true_str, *cols, 'WRONG' if top1_wrong else ''))
    print()
    print('  Totally wrong top-1 prediction : {}/{} samples'.format(n_wrong, len(rows)))

def tid(tactic):
    return TACTIC_IDS.get(tactic, tactic)

def print_tactic_results_table(tactic_rows, n_tactics, top_n=3):
    col_file = 26
    col_true = 24
    col_score = 16
    print()
    print('-- Test Results (scored against {} tactic prototypes) --'.format(n_tactics))
    header_fmt = '{:<4} {:<' + str(col_file) + '} {:<' + str(col_true) + '} ' + ' '.join(['{:<' + str(col_score) + '}'] * top_n) + ' {:<6}'
    print(header_fmt.format('#', 'File', 'True tactics', *['#{} (score)'.format(j + 1) for j in range(top_n)], 'Top{}'.format(top_n)))
    print('-' * (4 + col_file + col_true + col_score * top_n + 6 + top_n + 3))
    n_wrong = 0
    for (i, r) in enumerate(tactic_rows, 1):
        ranked = r['ranked']
        top_tactics = {t for (t, _) in ranked[:top_n]}
        cols = []
        for j in range(top_n):
            cols.append('{} {:.4f}'.format(tid(ranked[j][0]), ranked[j][1]) if j < len(ranked) else '')
        true_str = ','.join((tid(t) for t in sorted(r['true_tactics'])))
        if len(true_str) > col_true:
            true_str = true_str[:col_true - 3] + '...'
        fname = r['file'] if len(r['file']) <= col_file else r['file'][:col_file - 3] + '...'
        wrong = not r['true_tactics'] & top_tactics
        if wrong:
            n_wrong += 1
        print(header_fmt.format(i, fname, true_str, *cols, 'WRONG' if wrong else ''))
    return n_wrong

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: {}'.format(device))
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    classes = ckpt['classes']
    model = DeepModel(in_dim=IN_DIM).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    split = load_split()
    cache = GraphTensorCache()
    tactic_map = load_tactic_map()
    prototypes = build_final_prototypes(model, cache, split, classes, device)
    class_tactic = load_technique_tactics()
    all_tactics = sorted(set(class_tactic.values()))
    unique_test = {}
    for technique in classes:
        for (filename, path) in split[technique]['test']:
            inst = instance_from_filename(filename)
            entry = unique_test.setdefault(inst, {'filename': filename, 'path': path, 'true_techniques': set()})
            entry['true_techniques'].add(technique)
    n_total = 0
    n_correct_tech = 0
    n_correct_tac = 0
    y_true_tac = []
    y_score_tac = []
    rows = []
    tactic_rows = []
    for inst in sorted(unique_test):
        filename = unique_test[inst]['filename']
        path = unique_test[inst]['path']
        true_techniques = unique_test[inst]['true_techniques']
        embed = embed_graph(model, cache, path, device)
        dists = torch.norm(prototypes - embed, dim=1)
        pred_idx = int(torch.argmin(dists))
        pred_technique = classes[pred_idx]
        true_tactics = set()
        for technique in true_techniques:
            true_tactics |= instance_tactics(tactic_map, technique, filename)
        n_total += 1
        if pred_technique in true_techniques:
            n_correct_tech += 1
        if class_tactic[pred_technique] in true_tactics:
            n_correct_tac += 1
        scores_by_class = (-dists).tolist()
        tactic_scores = {t: -1000000000.0 for t in all_tactics}
        for (c, s) in zip(classes, scores_by_class):
            t = class_tactic[c]
            if s > tactic_scores[t]:
                tactic_scores[t] = s
        y_true_tac.append([1 if t in all_tactics and t in true_tactics else 0 for t in all_tactics])
        y_score_tac.append([tactic_scores[t] for t in all_tactics])
        ranked = sorted(zip(classes, scores_by_class), key=lambda x: x[1], reverse=True)
        rows.append({'file': filename, 'true_technique': '/'.join(sorted(true_techniques)), 'true_techniques': true_techniques, 'ranked': ranked})
        tactic_ranked = sorted(tactic_scores.items(), key=lambda x: x[1], reverse=True)
        tactic_rows.append({'file': inst, 'true_tactics': true_tactics, 'ranked': tactic_ranked})
    print_results_table(rows, len(classes))
    n_wrong_top3_tac = print_tactic_results_table(tactic_rows, len(all_tactics))
    tech_acc = n_correct_tech / n_total
    tac_acc = n_correct_tac / n_total
    y_true_tac = np.array(y_true_tac)
    y_score_tac = np.array(y_score_tac)
    lrap = label_ranking_average_precision_score(y_true_tac, y_score_tac)
    valid_cols = y_true_tac.sum(axis=0) > 0
    aupr = average_precision_score(y_true_tac[:, valid_cols], y_score_tac[:, valid_cols], average='macro') if valid_cols.any() else 0.0
    print()
    print('LRAP (Label Ranking Average Precision)          : {:.1f}%'.format(lrap * 100))
    print('AUPR (Area Under Precision-Recall Curve, macro) : {:.1f}%'.format(aupr * 100))
    print('Wrong (true label not in top-3)                 : {}/{} samples'.format(n_wrong_top3_tac, n_total))
    print()
    print('Test samples          : {}'.format(n_total))
    print('Technique classes     : {}'.format(len(classes)))
    print('Tactics represented   : {}'.format(len(all_tactics)))
    print()
    print('Technique Accuracy    : {:.1f}%'.format(tech_acc * 100))
    print('Tactic Accuracy       : {:.1f}%'.format(tac_acc * 100))
if __name__ == '__main__':
    main()
