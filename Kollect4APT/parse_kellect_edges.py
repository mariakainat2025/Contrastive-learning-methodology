import glob
import json
import os

from tqdm import tqdm

DATA_DIR = '/csse/research/contructive-learning/Kollect4APT/public_data'
OUTPUT_DIR = '/csse/research/contructive-learning/Kollect4APT/parsed_edges'

TREC_EVENT_MAP = {
    'ProcessStart': ('process', 'process', 'launch'),

    'FileIOCreate': ('process', 'file', 'create'),
    'FileIOFileCreate': ('process', 'file', 'create'),
    'FileIORead': ('process', 'file', 'read'),
    'FileIOWrite': ('process', 'file', 'write'),
    'FileIOClose': ('process', 'file', 'close'),
    'FileIOCleanup': ('process', 'file', 'close'),
    'FileIODelete': ('process', 'file', 'delete'),
    'FileIOFileDelete': ('process', 'file', 'delete'),
    'FileIODirEnum': ('process', 'file', 'enum'),

    'RegistryKCBCreate': ('process', 'registry', 'open'),
    'RegistryQuery': ('process', 'registry', 'query'),
    'RegistryQueryValue': ('process', 'registry', 'query'),
    'RegistryQueryMultipleValue': ('process', 'registry', 'query'),
    'RegistryEnumerateKey': ('process', 'registry', 'enumerate'),
    'RegistryEnumerateValueKey': ('process', 'registry', 'enumerate'),
    'RegistrySetValue': ('process', 'registry', 'modify'),
    'RegistrySetInformation': ('process', 'registry', 'modify'),
    'RegistryKCBDelete': ('process', 'registry', 'close'),
    'RegistryClose': ('process', 'registry', 'close'),
    'RegistryDeleteValue': ('process', 'registry', 'delete'),
    'RegistryDelete': ('process', 'registry', 'delete'),

    'TcpIpSendIPV4': ('process', 'socket', 'send'),
    'TcpIpSendIPV6': ('process', 'socket', 'send'),
    'TcpIpRecvIPV4': ('process', 'socket', 'receive'),
    'TcpIpRecvIPV6': ('process', 'socket', 'receive'),
    'TcpIpRetransmitIPV4': ('process', 'socket', 'retransmit'),
    'TcpIpConnectIPV4': ('process', 'socket', 'connect'),
    'TcpIpConnectIPV6': ('process', 'socket', 'connect'),
    'TcpIpDisconnectIPV4': ('process', 'socket', 'disconnect'),
    'TcpIpDisconnectIPV6': ('process', 'socket', 'disconnect'),
    'TcpIpAcceptIPV4': ('process', 'socket', 'accept'),
    'TcpIpAcceptIPV6': ('process', 'socket', 'accept'),
    'TcpIpReconnectIPV4': ('process', 'socket', 'reconnect'),
    'TcpIpReconnectIPV6': ('process', 'socket', 'reconnect'),
}

FILE_PATH_FIELD = {'FileIOCreate': 'OpenPath'}


def extract_process_node(pid, pname):
    return {'id': f'proc:{pid}', 'type': 'process', 'name': pname}


def extract_file_node(event, tag, counters):
    args = event.get('args', {})
    field = FILE_PATH_FIELD.get(event['Event'], 'FileName')
    path = args.get(field)
    if path:
        return {'id': f'file:{path}', 'type': 'file', 'name': path,
                'file_key': args.get('FileKey'), 'resolved': True}
    fkey = args.get('FileKey')
    fobj = args.get('FileObject')
    if fkey and fobj:
        uid = f'file:unresolved:{tag}:{fkey}:{fobj}'
    else:
        uid = f'file:unknown:{tag}'
    return {'id': uid, 'type': 'file', 'name': None,
            'file_key': fkey, 'file_object': fobj, 'resolved': False}


def extract_registry_node(event, tag, counters):
    args = event.get('args', {})
    key_name = args.get('KeyName')
    if key_name:
        key_lower = key_name.lower()
        return {'id': f'reg:{key_lower}', 'type': 'registry', 'name': key_lower,
                'key_handle': args.get('KeyHandle'), 'status': args.get('Status'), 'resolved': True}
    counters['registry'] += 1
    uid = f'reg:unresolved:{tag}:{counters["registry"]}'
    return {'id': uid, 'type': 'registry', 'name': None,
            'key_handle': args.get('KeyHandle'), 'status': args.get('Status'), 'resolved': False}


def extract_socket_node(event, tag, counters):
    args = event.get('args', {})
    saddr, sport = args.get('saddr'), args.get('sport')
    daddr, dport = args.get('daddr'), args.get('dport')
    identity = f'{saddr}:{sport}->{daddr}:{dport}'
    return {'id': f'sock:{identity}', 'type': 'socket', 'name': identity,
            'connid': args.get('connid')}


def parse_sample(json_path):
    tag = os.path.splitext(os.path.basename(json_path))[0]
    edges = []
    node_details = {}
    counters = {'file': 0, 'registry': 0, 'socket': 0}

    total_events = 0
    skipped_bad_json = 0
    skipped_unmapped_type = 0
    unresolved_target = 0

    with open(json_path, 'r', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_events += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                skipped_bad_json += 1
                continue

            evt_name = event.get('Event')
            mapping = TREC_EVENT_MAP.get(evt_name)
            if mapping is None:
                skipped_unmapped_type += 1
                continue
            src_cat, dst_cat, edge_type = mapping
            full_edge_type = f'{dst_cat}_{edge_type}'

            pid = event.get('PID')
            pname = event.get('PName')
            timestamp = event.get('TimeStamp')
            src_node = extract_process_node(pid, pname)

            if dst_cat == 'process':
                child_pid = event.get('args', {}).get('ProcessId', pid)
                dst_node = extract_process_node(child_pid, pname)
            elif dst_cat == 'file':
                dst_node = extract_file_node(event, tag, counters)
            elif dst_cat == 'registry':
                dst_node = extract_registry_node(event, tag, counters)
            elif dst_cat == 'socket':
                dst_node = extract_socket_node(event, tag, counters)
            else:
                continue

            if not dst_node.get('resolved', True):
                unresolved_target += 1

            node_details[src_node['id']] = src_node
            node_details[dst_node['id']] = dst_node

            edges.append((src_node['id'], src_node['type'],
                          dst_node['id'], dst_node['type'],
                          full_edge_type, timestamp))

    edges.sort(key=lambda e: e[5] if e[5] is not None else 0)
    skip_counts = {
        'bad_json': skipped_bad_json,
        'unmapped_type': skipped_unmapped_type,
        'unresolved_target': unresolved_target,
    }
    return edges, node_details, total_events, skip_counts


def main(data_dir=DATA_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(f'{data_dir}/*/*.json'))
    print(f'Found {len(files)} sample files to parse.')

    node_type_dict = {}
    edge_type_dict = {}
    all_node_details = {}
    edges_files = []

    grand_total_events = 0
    grand_skip_counts = {'bad_json': 0, 'unmapped_type': 0}
    grand_unresolved_target = 0

    for fpath in tqdm(files, desc='Parsing samples'):
        tag = os.path.splitext(os.path.basename(fpath))[0]
        edges, node_details, total_events, skip_counts = parse_sample(fpath)

        grand_total_events += total_events
        for k in grand_skip_counts:
            grand_skip_counts[k] += skip_counts[k]
        grand_unresolved_target += skip_counts['unresolved_target']

        for node_id, detail in node_details.items():
            ntype = detail['type']
            if ntype not in node_type_dict:
                node_type_dict[ntype] = len(node_type_dict)
        for e in edges:
            etype = e[4]
            if etype not in edge_type_dict:
                edge_type_dict[etype] = len(edge_type_dict)

        all_node_details.update(node_details)

        sample_dir = os.path.join(output_dir, tag)
        os.makedirs(sample_dir, exist_ok=True)

        edges_out = os.path.join(sample_dir, 'edges.txt')
        with open(edges_out, 'w', encoding='utf-8') as fw:
            for srcId, srcType, dstId, dstType, edgeType, ts in edges:
                fw.write(f'{srcId}\t{srcType}\t{dstId}\t{dstType}\t{edgeType}\t{ts}\n')
        edges_files.append((tag, edges_out, len(edges)))

    total_edges = sum(n for _, _, n in edges_files)
    total_skipped = sum(grand_skip_counts.values())
    print(f'\nParsed {len(files)} files.')
    print(f'  Total events scanned : {grand_total_events:,}')
    print(f'  Kept as edges        : {total_edges:,}  ({100 * total_edges / grand_total_events:.1f}%)')
    print(f'    - of which target was unresolved (placeholder node, event still kept) : '
          f'{grand_unresolved_target:,}  ({100 * grand_unresolved_target / total_edges:.1f}% of kept)')
    print(f'  Skipped total        : {total_skipped:,}  ({100 * total_skipped / grand_total_events:.1f}%)')
    print(f'    - unmapped event type (not in TREC_EVENT_MAP) : {grand_skip_counts["unmapped_type"]:,}')
    print(f'    - bad JSON line : {grand_skip_counts["bad_json"]:,}')

    with open(os.path.join(output_dir, 'node_type_map.json'), 'w', encoding='utf-8') as f:
        json.dump(node_type_dict, f, indent=2)
    print(f'node_type_map.json saved : {node_type_dict}')

    with open(os.path.join(output_dir, 'edge_type_map.json'), 'w', encoding='utf-8') as f:
        json.dump(edge_type_dict, f, indent=2)
    print(f'edge_type_map.json saved : {len(edge_type_dict)} types -> {sorted(edge_type_dict.keys())}')

    with open(os.path.join(output_dir, 'node_details.json'), 'w', encoding='utf-8') as f:
        json.dump(all_node_details, f, indent=2)
    print(f'node_details.json saved : {len(all_node_details):,} unique nodes across all samples')


if __name__ == '__main__':
    main()
