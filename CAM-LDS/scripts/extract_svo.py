import os
import sys
import spacy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

nlp = spacy.load('en_core_web_sm')

# ── Abstract vocabulary mapping ───────────────────────────────────────────────

_PROCESS_MAP = {
    'browser': 'web_process', 'firefox': 'web_process', 'chrome': 'web_process',
    'extension': 'web_process',
    'email': 'email_process', 'thunderbird': 'email_process', 'outlook': 'email_process',
    'attacker': 'user_process', 'implant': 'user_process', 'malware': 'user_process',
    'dropper': 'user_process', 'apt': 'user_process',
    'shell': 'system_process', 'bash': 'system_process', 'cmd': 'system_process',
}

_OBJECT_MAP = {
    'executable': 'user_file', 'binary': 'user_file', 'disk': 'user_file',
    'file': 'user_file', 'drakon': 'user_file', 'implant': 'user_file',
    'memory': 'process_file', 'process': 'process_file',
    'registry': 'system_file', 'config': 'system_file',
    'library': 'library_file', 'dll': 'library_file',
    'website': 'public_netflow', 'server': 'public_netflow',
    'network': 'public_netflow', 'ip': 'public_netflow',
    'target': 'user_file', 'host': 'user_file',
}

_VERB_MAP = {
    'load': 'execute', 'inject': 'execute', 'launch': 'execute',
    'run': 'execute', 'start': 'execute', 'spawn': 'execute',
    'write': 'write', 'drop': 'write', 'place': 'write', 'install': 'write',
    'read': 'read', 'steal': 'read', 'collect': 'read', 'gather': 'read',
    'connect': 'connect', 'browse': 'connect', 'navigate': 'connect',
    'send': 'send', 'receive': 'receive', 'download': 'receive',
    'delete': 'delete', 'remove': 'delete', 'clone': 'clone',
    'exploit': 'execute', 'execute': 'execute',
}


def _abstract(word, mapping):
    w = word.lower().strip()
    for key, label in mapping.items():
        if key in w:
            return label
    return w


def _follow_chain(token):
    """Follow xcomp/pcomp chain to find the real action verb."""
    current = token
    for _ in range(3):
        children_verbs = [c for c in current.children
                          if c.dep_ in ('xcomp', 'pcomp') and c.pos_ == 'VERB']
        if children_verbs:
            current = children_verbs[0]
        else:
            break
    return current


def _get_objects(verb_token):
    """Get direct objects and prepositional objects of a verb."""
    objects = []
    for child in verb_token.children:
        if child.dep_ in ('dobj', 'attr'):
            objects.append(child)
        elif child.dep_ == 'prep':
            for grandchild in child.children:
                if grandchild.dep_ == 'pobj':
                    objects.append(grandchild)
    return objects


def extract_svos(text):
    doc = nlp(text)
    svos = []

    for sent in doc.sents:
        for token in sent:
            # find subject
            if token.dep_ in ('nsubj', 'nsubjpass') and token.head.pos_ == 'VERB':
                subj     = token
                root_verb = token.head

                # follow xcomp/pcomp chain to real action
                real_verb = _follow_chain(root_verb)
                action    = real_verb.lemma_.lower()

                if action not in _VERB_MAP and action == root_verb.lemma_.lower():
                    continue  # skip non-attack verbs

                objects = _get_objects(real_verb)
                if not objects and real_verb != root_verb:
                    objects = _get_objects(root_verb)

                for obj in objects:
                    svos.append((subj.text, action, obj.text))

        # also catch participial phrases without explicit subject (pcomp of "by")
        for token in sent:
            if token.dep_ == 'pcomp' and token.pos_ == 'VERB':
                action = token.lemma_.lower()
                if action in _VERB_MAP:
                    objects = _get_objects(token)
                    for obj in objects:
                        svos.append(('attacker', action, obj.text))

    return svos


def abstract_svos(svos):
    result = []
    for subj, verb, obj in svos:
        s = _abstract(subj, _PROCESS_MAP)
        v = _VERB_MAP.get(verb, verb)
        o = _abstract(obj,  _OBJECT_MAP)
        result.append((s, v, o))
    return result


def svos_to_sequence(abstract_svos):
    seen = set()
    sentences = []
    for s, v, o in abstract_svos:
        triple = f'{s} {v} {o}'
        if triple not in seen:
            seen.add(triple)
            sentences.append(triple)
    return '. '.join(sentences)


if __name__ == '__main__':
    text = """Continued attack against THEIA by exploiting the target via the malicious pass manager browser
extension in web browser. The attacker had previously tried to load drakon into the memory of the
browser extension on Windows, but this was unsuccessful. So, the attacker resorted to writing the
drakon implant executable to disk on the target upon exploiting the browser extension."""

    print('=== INPUT ===')
    print(text)
    print()

    raw_svos = extract_svos(text)
    print('=== RAW SVOs ===')
    for s, v, o in raw_svos:
        print(f'  ({s}, {v}, {o})')
    print()

    abs_svos = abstract_svos(raw_svos)
    print('=== ABSTRACTED SVOs ===')
    for s, v, o in abs_svos:
        print(f'  ({s}, {v}, {o})')
    print()

    seq = svos_to_sequence(abs_svos)
    print('=== LOG-FORMAT SEQUENCE ===')
    print(seq)
