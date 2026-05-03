import argparse
import json
import pickle
import random
import urllib.request
import urllib.error
from typing import List, Optional

from tqdm import tqdm

OLLAMA_URL    = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3"

DATASET = [
    {
        "idx": 2,
        "dep_id": 19799,
        "part_idx": 0,
        "total_parts": 1,
        "seed_uuid": "6818A92B-0000-0000-0000-000000000020",
        "seed_name": "/home/admin/clean",
        "n_triples": 42,
        "sequence": [
            "user_process clean execute user_file clean.",
            "user_process clean read library_file libc-2.",
            "user_process clean read library_file libpthread-2.",
            "user_process clean connect_recvfrom_sendto public_netflow.",
            "user_process clean read system_file urandom.",
            "user_process clean clone user_process clean.",
            "user_process clean recvfrom public_netflow.",
            "user_process clean read system_file online.",
            "user_process clean read system_file passwd.",
            "user_process clean sendto public_netflow.",
        ],
    },
]

SRC_TYPES = {
    "user_process",
    "util_process",
    "web_process",
    "email_process",
    "service_process",
    "system_process",
    "temp_process",
}

DST_TYPES = {
    "user_file",
    "library_file",
    "system_file",
    "process_file",
    "email_file",
    "temp_file",
    "public_netflow",
    "private_netflow",
    "local_connection",
    "network_failure",
    "kernel_socket",
    "user_process",
    "util_process",
    "web_process",
    "email_process",
    "service_process",
    "system_process",
    "temp_process",
}

OP_MAP = {
    "execute":                  "execute",
    "read":                     "read",
    "write":                    "write",
    "clone":                    "clone",
    "connect":                  "connect to",
    "recvfrom":                 "recvfrom",
    "sendto":                   "sendto",
    "connect_recvfrom_sendto":  "connect_recvfrom_sendto",
    "recvfrom_sendto":          "recvfrom_sendto",
    "connect_sendto":           "connect_sendto",
    "open":                     "open",
    "delete":                   "delete",
    "modify":                   "modify",
    "rename":                   "rename",
    "link":                     "link",
    "unlink":                   "unlink",
    "fork":                     "fork",
}

_PROC_SRCS = [
    "user_process", "util_process", "web_process",
    "email_process", "service_process", "system_process", "temp_process",
]
_FILE_DSTS = [
    "user_file", "library_file", "system_file",
    "process_file", "email_file", "temp_file",
]
_NET_DSTS = [
    "public_netflow", "private_netflow",
    "local_connection", "network_failure", "kernel_socket",
]
_PROC_DSTS = [
    "user_process", "util_process", "web_process",
    "email_process", "service_process", "system_process", "temp_process",
]

TEMPLATES: dict = {}

for _src in _PROC_SRCS:
    for _dst in _FILE_DSTS:
        TEMPLATES[f"{_src}_{_dst}"] = f"{_src} {{src}} {{op}} {_dst} {{dst}}"
    for _dst in _NET_DSTS:
        TEMPLATES[f"{_src}_{_dst}"] = f"{_src} {{src}} {{op}} {_dst}"
    for _dst in _PROC_DSTS:
        TEMPLATES[f"{_src}_{_dst}"] = f"{_src} {{src}} {{op}} {_dst} {{dst}}"

PROMPTS_PRE = [
    """
You are a professional cyber threat analyst. Your task is to convert structured behavioral interactions from audit logs into a clear and natural language narrative.

Please refer to the following examples for guidance:

Example 1:
Input:
1. user_process D execute user_file F.
2. user_process D execute user_file G.
3. user_process D read system_file A.
4. user_process D connect_recvfrom_sendto public_netflow.
5. user_process C modify system_file J and clone user_process K.
Output:
Here is the converted output: user_process D initiated by executing user_file F and G. It then read data from system_file A and exchanged data with public_netflow. Meanwhile, user_process C modified system_file J and cloned user_process K.

Example 2:
Input:
1. user_process A read system_file B.
2. user_process A write system_file C.
3. user_process A unlink system_file B.
4. user_process A unlink system_file C.
Output:
Here is the converted output: user_process A read system_file B and wrote to system_file C. It then unlinked system_file B and unlinked system_file C.

STRICT RULES — you must follow these exactly:
- Preserve every operation EXACTLY as given: read means read, write means write, unlink means unlink. Never confuse or swap them.
- When a process clones itself (src and dst are the same), say "cloned itself", not "was cloned".
- Always name every specific file for each operation. Never say "those files" or "the files".
- Exclude only repetitive, immediately consecutive identical actions and the initial self-execution of the process.
- Output the converted result directly, without any extra formatting or labels.

Input: {doc}
    """,

    """
You are a digital forensics expert. Parse structured audit trails into forensic narratives that document the chain of events for incident response.

Please refer to the following examples for guidance:

Example 1:
Input:
1. user_process A read system_file B.
2. user_process A execute user_file C.
3. user_process D write user_file C.
4. user_process D connect_recvfrom_sendto public_netflow.
5. user_process D recvfrom public_netflow.
6. user_process D write system_file F.
Output:
Here is the structured output: user_process A read system_file B and executed user_file C. user_process D wrote to user_file C, connected to public_netflow to receive data, then wrote the data to system_file F.

Example 2:
Input:
1. web_process firefox clone web_process firefox.
2. web_process firefox read system_file resolv.
3. web_process firefox connect_recvfrom_sendto public_netflow.
4. email_process thunderbird write email_file inbox.
5. email_process thunderbird sendto public_netflow.
Output:
Here is the structured output: web_process firefox cloned itself, read system_file resolv, and connected to public_netflow. email_process thunderbird wrote to email_file inbox and sent data to public_netflow.

STRICT RULES — you must follow these exactly:
- Preserve every operation EXACTLY as given: read means read, write means write, unlink means unlink. Never confuse or swap them.
- When a process clones itself (src and dst are the same), say "cloned itself", not "was cloned".
- Always name every specific file for each operation. Never say "those files" or "the files".
- Exclude repetitive actions and initial process instantiation.
- Output only the forensic narrative, without any extra formatting or labels.

Input: {doc}
    """,

    """
You are documenting system behavior for technical audiences. Transform structured interaction logs into detailed technical documentation that preserves important context.

Please refer to the following examples for guidance:

Example 1:
Input:
1. service_process sshd read system_file passwd.
2. service_process sshd read library_file libpam.
3. service_process sshd clone user_process bash.
4. user_process bash execute system_file ls.
5. user_process bash read process_file meminfo.
Output:
Here is the transformed output: service_process sshd read system_file passwd and library_file libpam, then cloned user_process bash. user_process bash executed system_file ls and read process_file meminfo.

Example 2:
Input:
1. system_process bash clone system_process sh.
2. system_process sh execute system_file uname.
3. system_process sh read library_file libc-2.
4. util_process gnome-terminal read system_file bash_completion.
Output:
Here is the transformed output: system_process bash cloned system_process sh. system_process sh executed system_file uname and read library_file libc-2. util_process gnome-terminal read system_file bash_completion.

STRICT RULES — you must follow these exactly:
- Preserve every operation EXACTLY as given: read means read, write means write, unlink means unlink. Never confuse or swap them.
- When a process clones itself (src and dst are the same), say "cloned itself", not "was cloned".
- Always name every specific file for each operation. Never say "those files" or "the files".
- Omit repetitive operations and process self-instantiation events.
- Provide documentation only, without any extra formatting or labels.

Input: {doc}
    """,

    """
You are a malware behavior analyst. Your objective is to synthesize structured system interactions into behavioral descriptions that highlight patterns and relationships.

Please refer to the following examples for guidance:

Example 1:
Input:
1. user_process clean execute user_file clean.
2. user_process clean read library_file libc-2.
3. user_process clean connect_recvfrom_sendto public_netflow.
4. user_process clean read system_file passwd.
5. user_process clean sendto public_netflow.
Output:
Here is the structured output: Behavioral analysis reveals user_process clean executing user_file clean and reading library_file libc-2. It then connected to public_netflow, read system_file passwd, and sent data to public_netflow.

Example 2:
Input:
1. temp_process tcexec execute temp_file jag_isht.
2. temp_process tcexec read library_file libqt5widgets.
3. temp_process tcexec read library_file libqt5core.
4. email_process thunderbird write email_file inbox.
5. email_process thunderbird connect_recvfrom_sendto public_netflow.
Output:
Here is the structured output: Behavioral analysis reveals temp_process tcexec executing temp_file jag_isht and reading library_file libqt5widgets and libqt5core. email_process thunderbird wrote to email_file inbox and connected to public_netflow.

STRICT RULES — you must follow these exactly:
- Preserve every operation EXACTLY as given: read means read, write means write, unlink means unlink. Never confuse or swap them.
- When a process clones itself (src and dst are the same), say "cloned itself", not "was cloned".
- Always name every specific file for each operation. Never say "those files" or "the files".
- Remove repetitive entries and self-referential process creation.
- Deliver only the behavioral description, without any extra formatting or labels.

Input: {doc}
    """,

    """
You are conducting a security investigation. Translate raw behavioral logs into investigative summaries that security teams can quickly understand.

Please refer to the following examples for guidance:

Example 1:
Input:
1. web_process firefox recvfrom public_netflow.
2. web_process firefox write user_file webappsstore.
3. user_process fluxbox execute email_process thunderbird.
4. email_process thunderbird connect_recvfrom_sendto public_netflow.
5. email_process thunderbird write email_file inbox.
Output:
Here is the translated output: Investigation identifies web_process firefox receiving data from public_netflow and writing to user_file webappsstore. user_process fluxbox executed email_process thunderbird. email_process thunderbird connected to public_netflow and wrote to email_file inbox.

Example 2:
Input:
1. system_process bash clone system_process bash.
2. system_process bash execute temp_file jag_isht.
3. system_process bash read library_file libqt5widgets.
4. system_process bash read library_file libqt5core.
5. system_process bash read library_file libc-2.
Output:
Here is the translated output: Investigation identifies system_process bash cloning itself and executing temp_file jag_isht. It then read library_file libqt5widgets, libqt5core, and libc-2.

STRICT RULES — you must follow these exactly:
- Preserve every operation EXACTLY as given: read means read, write means write, unlink means unlink. Never confuse or swap them.
- When a process clones itself (src and dst are the same), say "cloned itself", not "was cloned".
- Always name every specific file for each operation. Never say "those files" or "the files".
- Remove repetitive entries and self-referential process creation.
- Output only the investigative summary, without any extra formatting or labels.

Input: {doc}
    """,
]

OP_SYNONYMS = {
    "write":   ["wrote", "write", "written", "writes"],
    "read":    ["read", "reads", "reading"],
    "unlink":  ["unlink", "unlinked", "unlinks"],
    "sendto":  ["sent", "send", "sendto", "transmitted", "sends"],
    "clone":   ["cloned", "clone", "spawned", "clones"],
    "execute": ["executed", "execute", "executes", "executing"],
    "recvfrom":["received", "receive", "recvfrom", "receives"],
    "connect": ["connected", "connect", "connects"],
    "delete":  ["deleted", "delete", "deletes"],
    "modify":  ["modified", "modify", "modifies"],
    "open":    ["opened", "open", "opens"],
}


def parse_triplet(sentence: str) -> Optional[dict]:
    sentence = sentence.rstrip(". \n")
    parts    = sentence.split()

    if len(parts) < 4:
        return None

    src_type  = parts[0]
    src_name  = parts[1]
    operation = parts[2]
    dst_type  = parts[3]
    dst_name  = parts[4] if len(parts) > 4 else ""

    if src_type not in SRC_TYPES:
        return None

    return {
        "src_type":  src_type,
        "src_name":  src_name,
        "operation": operation,
        "dst_type":  dst_type,
        "dst_name":  dst_name,
    }


def generate_sentence(parsed: dict) -> Optional[str]:
    key      = f"{parsed['src_type']}_{parsed['dst_type']}"
    template = TEMPLATES.get(key)

    if template is None:
        return None

    op = OP_MAP.get(parsed["operation"], parsed["operation"])

    return template.format(
        src=parsed["src_name"],
        op=op,
        dst=parsed["dst_name"],
    )


def deduplicate_sentences(sentences: List[str]) -> List[str]:
    if not sentences:
        return []

    deduped = [sentences[0]]
    for s in sentences[1:]:
        if s != deduped[-1]:
            deduped.append(s)

    i = 0
    while i <= len(deduped) - 4:
        if deduped[i] == deduped[i + 2] and deduped[i + 1] == deduped[i + 3]:
            del deduped[i + 2: i + 4]
        else:
            i += 1

    return deduped


def sequence_to_formatted_doc(sequence: List[str]) -> str:
    sentences = []
    for raw in sequence:
        parsed = parse_triplet(raw)
        if parsed is None:
            continue
        sentence = generate_sentence(parsed)
        if sentence is None:
            continue
        sentences.append(sentence)

    deduped = deduplicate_sentences(sentences)
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(deduped))


def validate_narrative(narrative: str, sequence: List[str]) -> List[str]:
    issues = []
    narrative_lower = narrative.lower()

    for raw in sequence:
        parsed = parse_triplet(raw)
        if parsed is None:
            continue

        op  = parsed["operation"]
        dst = parsed["dst_name"]

        if dst and dst not in narrative_lower:
            issues.append(f"Missing target '{dst}' for operation '{op}'")

        if op in OP_SYNONYMS and dst:
            synonyms = OP_SYNONYMS[op]
            op_found = any(syn in narrative_lower for syn in synonyms)
            if not op_found:
                issues.append(f"Operation '{op}' not reflected in narrative")

    return issues


def clean_narrative(narrative: str) -> str:
    replacements = [
        ("was cloned",          "cloned itself"),
        ("were cloned",         "cloned themselves"),
        ("has been cloned",     "cloned itself"),
        ("got cloned",          "cloned itself"),
        ("those files",         "the specified files"),
        ("the above files",     "the specified files"),
        ("these files",         "the specified files"),
    ]
    for old, new in replacements:
        narrative = narrative.replace(old, new)
    return narrative


def check_ollama_running():
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=3)
    except Exception:
        raise RuntimeError(
            "\nOllama server is not running!\n"
            "Please open a NEW terminal and run:  ollama serve\n"
            "Then come back here and run the script again."
        )


def chat_with_ollama(prompt: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt},
        ],
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=None) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["message"]["content"].strip()
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def generate_narrative(sequence: List[str], model: str) -> str:
    fmt_doc = sequence_to_formatted_doc(sequence)
    if not fmt_doc.strip():
        return ""
    prompt = random.choice(PROMPTS_PRE).format(doc=fmt_doc)
    return chat_with_ollama(prompt, model)


def run_dry(dataset: list):
    for record in dataset:
        fmt_doc = sequence_to_formatted_doc(record["sequence"])
        n_raw   = record.get("n_triples", len(record["sequence"]))
        n_dedup = len(fmt_doc.splitlines())

        print("\n" + "=" * 60)
        print(f"idx        : {record.get('idx')}")
        print(f"seed       : {record.get('seed_name')}")
        print(f"raw triples: {n_raw}  →  after dedup: {n_dedup}")
        print("--- Formatted doc (would be sent to Llama 3) ---")
        print(fmt_doc)


def run_full(dataset: list, output_path: str, model: str):
    check_ollama_running()
    print(f"Using model : {model}")
    print(f"Ollama URL  : {OLLAMA_URL}\n")

    results = []
    for record in tqdm(dataset, desc="Generating narratives"):
        sequence  = record.get("sequence", [])
        fmt_doc   = sequence_to_formatted_doc(sequence)
        narrative = generate_narrative(sequence, model)
        narrative = clean_narrative(narrative)
        issues    = validate_narrative(narrative, sequence)

        print("\n" + "=" * 60)
        print(f"idx        : {record.get('idx')}")
        print(f"seed       : {record.get('seed_name')}")
        print("--- Formatted input ---")
        print(fmt_doc)
        print("--- Narrative ---")
        print(narrative)

        if issues:
            print("Validation issues detected:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("Narrative validated successfully.")

        results.append({
            "idx":       record.get("idx"),
            "dep_id":    record.get("dep_id"),
            "seed_name": record.get("seed_name"),
            "sequence":  sequence,
            "formatted": fmt_doc,
            "narrative": narrative,
            "validation": issues,
        })

    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved {len(results)} narratives → {output_path}")


def read_results(pickle_path: str):
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)

    for r in results:
        print("\n" + "=" * 60)
        print(f"idx        : {r['idx']}")
        print(f"seed       : {r['seed_name']}")
        print("--- Formatted doc ---")
        print(r["formatted"])
        print("--- Narrative ---")
        print(r["narrative"])
        issues = r.get("validation", [])
        if issues:
            print("Validation issues:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("Validated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert triplet sequences to natural language via Ollama."
    )
    parser.add_argument("--dry-run",  action="store_true", help="Show formatted docs only.")
    parser.add_argument("--model",    type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output",   type=str, default="narratives.pkl")
    parser.add_argument("--read",     type=str, default=None, metavar="PICKLE")
    args = parser.parse_args()

    if args.read:
        read_results(args.read)
    elif args.dry_run:
        run_dry(DATASET)
    else:
        run_full(DATASET, args.output, args.model)