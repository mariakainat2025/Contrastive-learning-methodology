import os
import re
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import json


SCRIPTS_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR       = os.path.dirname(SCRIPTS_DIR)

DATA_PATH         = os.path.join(BASE_DIR, 'input', 'theia') + os.sep


OUTPUT_PATH       = os.path.join(BASE_DIR, 'output', 'theia') + os.sep
OUTPUT_PARSED     = os.path.join(BASE_DIR, 'output', 'theia', 'parsed')     + os.sep
OUTPUT_IOC_MATCH  = os.path.join(BASE_DIR, 'output', 'theia', 'ioc_match')  + os.sep
OUTPUT_WINDOWS    = os.path.join(BASE_DIR, 'output', 'theia', 'windows')    + os.sep
OUTPUT_GRAPHS     = os.path.join(BASE_DIR, 'output', 'theia', 'graphs')     + os.sep
OUTPUT_FEATURES   = os.path.join(BASE_DIR, 'output', 'theia', 'features')   + os.sep
OUTPUT_EMBEDDINGS = os.path.join(BASE_DIR, 'output', 'theia', 'embeddings') + os.sep
OUTPUT_CTI        = os.path.join(BASE_DIR, 'output', 'theia', 'cti')            + os.sep
OUTPUT_BENIGN     = os.path.join(BASE_DIR, 'output', 'theia', 'benign')         + os.sep
OUTPUT_TRAINING   = os.path.join(BASE_DIR, 'output', 'theia', 'trainingoutput') + os.sep

JSON_FILE      = 'file.json'
MALICIOUS_FILE = os.path.join(DATA_PATH, 'theia.txt')
EDGES_FILE     = OUTPUT_PARSED + JSON_FILE + '.txt'

#  Regex patterns (CDM18 schema)
pattern_uuid         = re.compile(r'uuid\":\"(.*?)\"')
pattern_type         = re.compile(r'type\":\"(.*?)\"')
pattern_time         = re.compile(r'timestampNanos\":(.*?),')
pattern_src          = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst1         = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_dst2         = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
pattern_file_path    = re.compile(r'\"path\":\"(.*?)\"')
pattern_file_name    = re.compile(r'\"filename\":\"(.*?)\"')
pattern_cmdline      = re.compile(r'cmdLine\":\{\"string\":\"(.*?)\"\}')
pattern_exe_path     = re.compile(r'\"path\":\"(/[^\"]+)\"')
pattern_netflow_name = re.compile(r'remoteAddress\":\"(.*?)\"')
pattern_pid          = re.compile(r'\"cid\":([\d]+)')
pattern_ppid         = re.compile(r'\"ppid\":\"([\d]+)\"')
pattern_start_ts     = re.compile(r'startTimestampNanos\":([\d]+)')
pattern_parent_uuid  = re.compile(
    r'parentSubject\":\{\"com\.bbn\.tc\.schema\.avro\.cdm18\.UUID\":\"(.*?)\"\}')
pattern_remote_port  = re.compile(r'remotePort\":([\d]+)')
pattern_local_addr   = re.compile(r'localAddress\":\"(.*?)\"')
pattern_local_port   = re.compile(r'localPort\":([\d]+)')
pattern_mem_addr     = re.compile(r'memoryAddress\":([\d]+)')


ATTACK_START_NS = 1523551440000000000   # 12:44 ET
ATTACK_END_NS   = 1523553960000000000   # 13:26 ET
WINDOW_SIZE_NS  = 15 * 60 * int(1e9)   # 15-minute tumbling windows

SCAN_TARGET_IPS = {
    '128.55.12.73',  '128.55.12.166', '128.55.12.67',
    '128.55.12.141', '128.55.12.110', '128.55.12.118',
    '128.55.12.10',  '128.55.12.1',   '128.55.12.55',
}
LOCAL_HOST_IP = '128.55.12.167'

IOC_IPS = {
    '5.214.163.155',
    '146.153.68.151',
    '104.228.117.212',
    '141.43.176.203',
    '149.52.198.23',
    '128.55.12.167',
    '128.55.12.73',  '128.55.12.166', '128.55.12.67',
    '128.55.12.141', '128.55.12.110', '128.55.12.118',
    '128.55.12.10',  '128.55.12.1',   '128.55.12.55',
}

IOC_FILES = {
    '/var/log/xdev',
    '/var/log/wdev',
    '/tmp/memtrace.so',
    '/var/log/mail',
    '/home/admin/profile',
    'libdrakon.linux.x64.so',
    'loaderDrakon.linux.x64',
    'microapt.linux.x64',
}

IOC_KEYWORDS = {
    'gatech',
    'sshd',
}

WINDOW_SEEDS = {
    1: [
        '/home/admin/profile',
        '128.55.12.110',
        '146.153.68.151',
        '141.43.176.203',
        '/var/log/wdev',
    ],
    2: [
        '/home/admin/profile',
        '128.55.12.110',
        '/tmp/memtrace.so',
        '/var/log/wdev',
        '146.153.68.151',
    ],
    3: [
        '/var/log/mail',
        '128.55.12.110',
        '/home/admin/profile',
    ],
}


CTI_REPORTS_DIR = os.path.join(BASE_DIR, 'input', 'cti_reports')
CTI_SCENARIO    = ''


def load_window_cti(scenario=None):
    if scenario is None:
        scenario = CTI_SCENARIO
    cti = {}
    for fname in sorted(os.listdir(CTI_REPORTS_DIR)):
        if fname.endswith('.txt') and not fname.startswith('.'):
            wkey  = fname.replace('.txt', '')
            fpath = os.path.join(CTI_REPORTS_DIR, fname)
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                cti[wkey] = f.read().strip()
    return cti



def show(msg):
    """Print message with current timestamp."""
    print(msg + '  [' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')


def update_ts(ts_map, uuid, timestamp):
    """Update first/last seen timestamps for a node."""
    if uuid not in ts_map:
        ts_map[uuid] = {'first': timestamp, 'last': timestamp}
    else:
        if timestamp < ts_map[uuid]['first']:
            ts_map[uuid]['first'] = timestamp
        if timestamp > ts_map[uuid]['last']:
            ts_map[uuid]['last'] = timestamp


def ns_to_et(ns):
    """Convert nanosecond timestamp → human-readable ET string (UTC-4, date-safe)."""
    if ns == '' or ns is None:
        return 'NO TIMESTAMP'
    utc = datetime.fromtimestamp(int(ns) / 1e9, tz=timezone.utc)
    et  = utc - timedelta(hours=4)
    return '{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d} ET'.format(
        et.year, et.month, et.day, et.hour, et.minute, et.second)


def matches_ioc(data, ioc_ips=None, ioc_files=None, ioc_keywords=None):
    if ioc_ips      is None: ioc_ips      = IOC_IPS
    if ioc_files    is None: ioc_files    = IOC_FILES
    if ioc_keywords is None: ioc_keywords = IOC_KEYWORDS

    for field in ('remote_ip', 'local_ip'):
        val = data.get(field, '')
        if val and val in ioc_ips:
            return True, val
    fp = data.get('file_path', '')
    if fp:
        fp_lower = fp.lower()
        for ioc in ioc_files:
            if ioc.lower() in fp_lower:
                return True, ioc
    for field in ('exe_path', 'cmdline'):
        val = data.get(field, '')
        if not val:
            continue
        v = val.lower()
        for ioc in ioc_files:
            if ioc.lower() in v:
                return True, ioc
    name = data.get('name', '')
    if name:
        n = name.lower()
        for ioc in ioc_files:
            if ioc.lower() in n:
                return True, ioc
        for kw in ioc_keywords:
            if kw in n:
                return True, kw
    return False, None


def load_adj():
    adj_path = OUTPUT_PARSE + 'adj.json'
    if not os.path.exists(adj_path):
        print('  adj.json not found: {}'.format(adj_path))
        return defaultdict(set)
    with open(adj_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    adj = defaultdict(set)
    for k, v in raw.items():
        adj[k] = set(v)
    print('  adj.json loaded : {} nodes'.format(len(adj)))
    return adj


def make_node_dict(uid, maps):
    id_nodetype_map   = maps['id_nodetype_map']
    id_nodename_map   = maps['id_nodename_map']
    id_exepath_map    = maps['id_exepath_map']
    id_cmdline_map    = maps['id_cmdline_map']
    id_pid_map        = maps['id_pid_map']
    id_ppid_map       = maps['id_ppid_map']
    id_parent_map     = maps['id_parent_map']
    id_remoteport_map = maps['id_remoteport_map']
    id_localaddr_map  = maps['id_localaddr_map']
    id_localport_map  = maps['id_localport_map']
    id_memaddr_map    = maps.get('id_memaddr_map', {})
    id_ts_map         = maps['id_ts_map']
    malicious_uuids   = maps['malicious_uuids']

    ntype   = id_nodetype_map.get(uid, 'UNKNOWN')
    ts_     = id_ts_map.get(uid, {})
    first_  = ts_.get('first', '')
    last_   = ts_.get('last',  '')
    has_ts_ = (first_ != '' and last_ != '')

    return {
        'node_type'      : ntype,
        'name'           : id_nodename_map.get(uid, ''),
        'exe_path'       : id_exepath_map.get(uid,    '') if ntype == 'SUBJECT_PROCESS' else '',
        'cmdline'        : id_cmdline_map.get(uid,    '') if ntype == 'SUBJECT_PROCESS' else '',
        'pid'            : id_pid_map.get(uid,        '') if ntype == 'SUBJECT_PROCESS' else '',
        'ppid'           : id_ppid_map.get(uid,       '') if ntype == 'SUBJECT_PROCESS' else '',
        'parent_uuid'    : id_parent_map.get(uid,     '') if ntype == 'SUBJECT_PROCESS' else '',
        'file_path'      : id_nodename_map.get(uid,   '') if 'FILE' in ntype            else '',
        'remote_ip'      : id_nodename_map.get(uid,   '') if ntype == 'NetFlowObject'   else '',
        'remote_port'    : id_remoteport_map.get(uid, '') if ntype == 'NetFlowObject'   else '',
        'local_ip'       : id_localaddr_map.get(uid,  '') if ntype == 'NetFlowObject'   else '',
        'local_port'     : id_localport_map.get(uid,  '') if ntype == 'NetFlowObject'   else '',
        'memory_address' : id_memaddr_map.get(uid,    '') if ntype == 'MemoryObject'    else '',
        'first_seen_ts'  : first_,
        'last_seen_ts'   : last_,
        'first_seen_et'  : ns_to_et(first_) if has_ts_ else 'NO TIMESTAMP',
        'last_seen_et'   : ns_to_et(last_)  if has_ts_ else 'NO TIMESTAMP',
        'in_theia_txt'   : 1 if uid in malicious_uuids else 0,
    }