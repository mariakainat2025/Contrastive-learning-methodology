import re

NODE_TYPE_RULES = [
    ('account',        re.compile(r'^acct:')),
    ('syscall',        re.compile(r'^syscall:')),
    ('not_found',      re.compile(r'\[not found\]')),
    ('credential_file', re.compile(r'\b(shadow|passwd|shells)\b')),
    ('network',        re.compile(r'network address|socket|sock file|/run/\w+/private|/run/user/')),
    ('config_file',    re.compile(r'\b(etc|profile\.d|crontabs)\b')),
    ('library_file',   re.compile(r'\blibrary\b')),
    ('user_file',      re.compile(r'(/home/|\.mozilla|\.local/share|user\s)')),
    ('system_binary',  re.compile(r'^/usr/(local/)?s?bin/|^/bin/|^/sbin/')),
]
NODE_TYPES = [name for (name, _pat) in NODE_TYPE_RULES] + ['other']


def classify_target(target):
    for (name, pattern) in NODE_TYPE_RULES:
        if pattern.search(target):
            return name
    return 'other'


def abstract_node_feature(sequence_lines):
    """sequence_lines: list of raw event strings like '/usr/sbin/cron openat shadow file.'
    Returns a dict {node_type: count} over NODE_TYPES, the same length for every sequence."""
    counts = {t: 0 for t in NODE_TYPES}
    for line in sequence_lines:
        parts = line.strip().rstrip('.').split(' ', 2)
        if len(parts) < 3:
            continue
        target = parts[2]
        counts[classify_target(target)] += 1
    return counts


if __name__ == '__main__':
    import json
    import glob
    from collections import Counter

    files = glob.glob('/csse/research/contructive-learning/CAM-LDS/sequences/*/*/sequence_*.json')
    other_examples = Counter()
    totals = {t: 0 for t in NODE_TYPES}

    for fp in files[:400]:
        with open(fp) as f:
            d = json.load(f)
        counts = abstract_node_feature(d['sequence'])
        for t, c in counts.items():
            totals[t] += c
        if counts['other']:
            for line in d['sequence']:
                parts = line.strip().rstrip('.').split(' ', 2)
                if len(parts) >= 3 and classify_target(parts[2]) == 'other':
                    other_examples[parts[2]] += 1

    print('NODE_TYPES ({}):'.format(len(NODE_TYPES)), NODE_TYPES)
    print()
    print('total events per type across 400 sequences:')
    for t, c in sorted(totals.items(), key=lambda x: -x[1]):
        print('  {:16s} {}'.format(t, c))
    print()
    print('most common UNMATCHED ("other") targets -- check these are genuinely miscellaneous:')
    for t, c in other_examples.most_common(20):
        print('  {:50s} {}'.format(t, c))
