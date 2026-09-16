import os
import sys
import json
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Levenshtein import ratio as lev_ratio

SIMILARITY_THRESHOLD = 0.7


def _to_sentences(text):
    if isinstance(text, list):
        text = '. '.join(str(s) for s in text)
    # return [s.strip() for s in str(text).split('.') if s.strip()]
    # correct one
    return [s.strip() for s in str(text).split('. ') if s.strip()]


def deduplicate_sequence(text, threshold=SIMILARITY_THRESHOLD):
    sentences = _to_sentences(text)
    kept = []
    for sent in sentences:
        if not any(lev_ratio(sent, k) >= threshold for k in kept):
            kept.append(sent)
    return '. '.join(kept)


def _deduplicate_with_report(text, threshold=SIMILARITY_THRESHOLD):
    sentences = _to_sentences(text)
    kept = []
    removed = []          # list of (removed_sent, matched_kept_sent, similarity)
    for sent in sentences:
        match = next((k for k in kept if lev_ratio(sent, k) >= threshold), None)
        if match is None:
            kept.append(sent)
        else:
            removed.append((sent, match, round(lev_ratio(sent, match), 3)))
    return '. '.join(kept), sentences, kept, removed


def deduplicate_sequences_file(in_path, out_path, threshold=SIMILARITY_THRESHOLD):
    with open(in_path) as f:
        data = json.load(f)

    records = data if isinstance(data, list) else data.get('sequences', data)

    report_entries = []

    for record in records:
        seq = record.get('sequence', '')
        deduped, original, kept, removed = _deduplicate_with_report(seq, threshold)
        record['sequence'] = deduped

        report_entries.append({
            'dep_id'           : record.get('dep_id'),
            'label'            : record.get('label', ''),
            'before_sentences' : len(original),
            'after_sentences'  : len(kept),
            'removed_sentences': len(removed),
            'reduction_pct'    : round(100 * len(removed) / len(original), 1) if original else 0,
            'kept'             : kept,
            'removed'          : removed,   # list of (removed, matched, sim)
        })

    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)

    return report_entries


def save_report(report_entries, report_path, file_name, threshold):
    lines = []
    lines.append('=' * 100)
    lines.append(f'  Deduplication Report — {file_name}')
    lines.append(f'  Threshold: {threshold}')
    lines.append('=' * 100)

    for entry in report_entries:
        lines.append('')
        label = entry.get('label') or f"dep={entry.get('dep_id')}"
        lines.append(f'  Sequence: {label}')
        lines.append(f'  Before : {entry["before_sentences"]} sentences')
        lines.append(f'  After  : {entry["after_sentences"]} sentences  ({entry["reduction_pct"]}% removed)')
        lines.append('-' * 100)

        lines.append('')
        lines.append('  KEPT SENTENCES:')
        for j, s in enumerate(entry.get('kept', []), 1):
            lines.append(f'    {j:4d}. {s}')

        lines.append('')
        lines.append('  REMOVED SENTENCES  (removed → matched kept  sim=similarity):')
        for j, item in enumerate(entry.get('removed', []), 1):
            if isinstance(item, (list, tuple)) and len(item) == 3:
                rem, matched, sim = item
            else:
                rem, matched, sim = item, '', ''
            lines.append(f'    {j:4d}.  REMOVED : {rem}')
            lines.append(f'           MATCHED : {matched}  (sim={sim})')

        lines.append('')
        lines.append('=' * 100)

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  Report saved: {report_path}')


def dedup_filename(name):
    return name.replace('sequences_', 'sequences_dedup_')


def _dedup_one_file(in_path, out_path, report_path, label, total_before, total_after):
    report_entries = deduplicate_sequences_file(in_path, out_path, SIMILARITY_THRESHOLD)
    save_report(report_entries, report_path, os.path.basename(in_path), SIMILARITY_THRESHOLD)
    for entry in report_entries:
        seq_label = entry.get('label') or f'dep={entry.get("dep_id")}'
        print(f'    {seq_label}: {entry["before_sentences"]} → {entry["after_sentences"]} '
              f'({entry["reduction_pct"]}% removed)')
        total_before += entry['before_sentences']
        total_after  += entry['after_sentences']
    return total_before, total_after


def run_deduplication():
    from scripts.config import OUTPUT_SEQUENCES, OUTPUT_TEST
    from scripts.filter_attack_subgraphs import ATTACKS
    import os
    os.makedirs(OUTPUT_TEST, exist_ok=True)

    total_before = 0
    total_after  = 0

    # Benign sequences
    benign_fname = 'sequences_benign.json'
    benign_in    = os.path.join(OUTPUT_SEQUENCES, benign_fname)
    benign_out   = os.path.join(OUTPUT_SEQUENCES, 'sequences_dedup_benign.json')
    benign_rep   = os.path.join(OUTPUT_TEST, 'dedup_benign_report.txt')

    if os.path.exists(benign_in):
        print(f'  [benign]')
        total_before, total_after = _dedup_one_file(
            benign_in, benign_out, benign_rep, 'benign', total_before, total_after)
    else:
        print(f'  skipping (not found): {benign_fname}')

    # Attack sequences
    for atk in ATTACKS:
        fname       = f'sequences_{atk["name"]}.json'
        out_fname   = dedup_filename(fname)
        in_path     = os.path.join(OUTPUT_SEQUENCES, fname)
        out_path    = os.path.join(OUTPUT_SEQUENCES, out_fname)
        report_path = os.path.join(OUTPUT_TEST, f'dedup_{atk["name"]}_report.txt')

        if not os.path.exists(in_path):
            print(f'  skipping (not found): {fname}')
            continue

        print(f'  [{atk["name"]}]')
        total_before, total_after = _dedup_one_file(
            in_path, out_path, report_path, atk['name'], total_before, total_after)

    print()
    if total_before > 0:
        pct = round(100 * (total_before - total_after) / total_before)
        print(f'  Total sentences: {total_before:,} → {total_after:,}  ({pct}% removed)')


if __name__ == '__main__':
    from scripts.config import INPUT_TEST, OUTPUT_TEST
    import os
    os.makedirs(OUTPUT_TEST, exist_ok=True)

    files = [
        ('abstracted_attack_sequences.json',  'dedup_attack_sequences.json',  'dedup_attack_report.txt'),
        ('abstracted_benign_sequences.json',   'dedup_benign_sequences.json',  'dedup_benign_report.txt'),
    ]

    print(f'  Threshold: {SIMILARITY_THRESHOLD}')
    print()

    for in_name, out_name, report_name in files:
        in_path     = os.path.join(INPUT_TEST, in_name)
        out_path    = os.path.join(INPUT_TEST, out_name)
        report_path = os.path.join(OUTPUT_TEST, report_name)

        if not os.path.exists(in_path):
            print(f'  skipping (not found): {in_path}')
            continue

        print(f'  Processing: {in_name}')
        report_entries = deduplicate_sequences_file(in_path, out_path, SIMILARITY_THRESHOLD)
        save_report(report_entries, report_path, in_name, SIMILARITY_THRESHOLD)

        for entry in report_entries:
            label = entry.get('label') or f"dep={entry.get('dep_id')}"
            print(f'    {label}: {entry["before_sentences"]} → {entry["after_sentences"]} sentences ({entry["reduction_pct"]}% removed)')
        print()
