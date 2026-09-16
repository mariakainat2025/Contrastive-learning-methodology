import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_ATTACK_VERBS = {
    'execute', 'executed', 'run', 'ran', 'download', 'downloaded', 'upload', 'uploaded',
    'connect', 'connected', 'send', 'sent', 'receive', 'received', 'write', 'wrote',
    'read', 'inject', 'injected', 'exploit', 'exploited', 'install', 'installed',
    'drop', 'dropped', 'persist', 'escalate', 'steal', 'stole', 'exfiltrate',
    'bypass', 'evade', 'launch', 'launched', 'deploy', 'deployed', 'load', 'loaded',
    'open', 'opened', 'click', 'clicked', 'access', 'accessed', 'compromise',
    'compromised', 'infect', 'infected', 'modify', 'modified', 'delete', 'deleted',
    'create', 'created', 'spawn', 'spawned', 'browse', 'browsed', 'establish',
    'established', 'initiate', 'initiated', 'perform', 'performed', 'attempt',
    'attempted', 'trigger', 'triggered', 'gain', 'gained', 'maintain', 'maintained',
}

_ATTACK_NOUNS = {
    'malware', 'implant', 'backdoor', 'dropper', 'exploit', 'payload', 'shellcode',
    'credential', 'password', 'attacker', 'adversary', 'victim', 'target',
    'c2', 'command', 'control', 'attack', 'malicious', 'phishing', 'ransomware',
    'trojan', 'virus', 'worm', 'rootkit', 'keylogger', 'spyware', 'botnet',
    'privilege', 'escalation', 'persistence', 'lateral', 'movement', 'exfiltration',
}

_TECH_TERMS = {
    'browser', 'email', 'attachment', 'link', 'url', 'domain', 'ip', 'network',
    'file', 'registry', 'process', 'memory', 'shell', 'script', 'executable',
    'binary', 'disk', 'process', 'socket', 'port', 'server', 'client',
    'extension', 'plugin', 'macro', 'powershell', 'bash', 'cmd',
}

MIN_SCORE = 2


def score_sentence(sentence):
    words = re.findall(r'\b\w+\b', sentence.lower())
    score = 0
    for w in words:
        if w in _ATTACK_VERBS:
            score += 2
        elif w in _ATTACK_NOUNS:
            score += 2
        elif w in _TECH_TERMS:
            score += 1
    return score


def extract_attack_sentences(text, min_score=MIN_SCORE):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    kept = [s.strip() for s in sentences if score_sentence(s) >= min_score and s.strip()]
    return kept


def process_file(in_path, min_score=MIN_SCORE):
    with open(in_path, encoding='utf-8') as f:
        text = f.read()
    sentences = extract_attack_sentences(text, min_score)
    return sentences


if __name__ == '__main__':
    sample = open(os.path.join(PROJECT_ROOT, 'input', 'cti_reports',
                               'dep19710_browser_extension_drakon.txt')).read()
    print('=== Original ===')
    print(sample)
    print()
    print('=== Attack sentences only ===')
    for s in extract_attack_sentences(sample):
        print(f'  [{score_sentence(s)}]  {s}')
