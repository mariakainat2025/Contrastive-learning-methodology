import os
import re
import time
from datetime import datetime, timezone, timedelta


SCRIPTS_DIR    = os.path.dirname(os.path.abspath(__file__))
BASE_DIR       = os.path.dirname(SCRIPTS_DIR)


DATA_PATH = os.path.join(BASE_DIR, 'input', 'theia') + os.sep

OUTPUT_PATH       = os.path.join(BASE_DIR, 'output', 'theia') + os.sep
OUTPUT_PARSED     = os.path.join(BASE_DIR, 'output', 'theia', 'parsed')     + os.sep
OUTPUT_IOC_MATCH  = os.path.join(BASE_DIR, 'output', 'theia', 'ioc_match')  + os.sep
OUTPUT_WINDOWS    = os.path.join(BASE_DIR, 'output', 'theia', 'windows')    + os.sep
OUTPUT_GRAPHS     = os.path.join(BASE_DIR, 'output', 'theia', 'graphs')     + os.sep
OUTPUT_FEATURES   = os.path.join(BASE_DIR, 'output', 'theia', 'features')   + os.sep
OUTPUT_EMBEDDINGS = os.path.join(BASE_DIR, 'output', 'theia', 'embeddings') + os.sep
OUTPUT_CTI        = os.path.join(BASE_DIR, 'output', 'theia', 'cti')            + os.sep
OUTPUT_SEQUENCES  = os.path.join(BASE_DIR, 'output', 'theia', 'sequences')      + os.sep
OUTPUT_BENIGN     = os.path.join(BASE_DIR, 'output', 'theia', 'benign')         + os.sep
OUTPUT_TRAINING   = os.path.join(BASE_DIR, 'output', 'theia', 'trainingoutput') + os.sep

MALICIOUS_FILE = os.path.join(DATA_PATH, 'theia.txt')
EDGES_FILE     = OUTPUT_PARSED + 'edges.txt'

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


ROBERTA_MODEL = 'FacebookAI/roberta-base'
MAX_LEN       = 512
STRIDE        = 384  
EMB_DIM       = 768


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
    print(msg + '  [' + time.strftime('%Y-%m-%d %H:%M:%S') + ']')


def update_ts(ts_map, uuid, timestamp):
    if uuid not in ts_map:
        ts_map[uuid] = {'first': timestamp, 'last': timestamp}
    else:
        if timestamp < ts_map[uuid]['first']:
            ts_map[uuid]['first'] = timestamp
        if timestamp > ts_map[uuid]['last']:
            ts_map[uuid]['last'] = timestamp


def ns_to_et(ns):
    if ns == '' or ns is None:
        return 'NO TIMESTAMP'
    utc = datetime.fromtimestamp(int(ns) / 1e9, tz=timezone.utc)
    et  = utc - timedelta(hours=4)
    return '{}-{:02d}-{:02d} {:02d}:{:02d}:{:02d} ET'.format(
        et.year, et.month, et.day, et.hour, et.minute, et.second)




def make_node_dict(uid, maps):
    id_nodetype_map   = maps['id_nodetype_map']
    id_nodename_map   = maps['id_nodename_map']
    id_exepath_map    = maps['id_exepath_map']
    id_cmdline_map    = maps['id_cmdline_map']
    id_remoteport_map = maps.get('id_remoteport_map', {})
    id_localaddr_map  = maps.get('id_localaddr_map',  {})
    id_localport_map  = maps.get('id_localport_map',  {})
    id_memaddr_map    = maps.get('id_memaddr_map',    {})
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
        'file_path'      : id_nodename_map.get(uid,   '') if 'FILE' in ntype            else '',
        'remote_ip'      : (id_nodename_map.get(uid, '').split(',')[2]
                            if ntype == 'NetFlowObject' and
                            len(id_nodename_map.get(uid, '').split(',')) > 2 else ''),
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
