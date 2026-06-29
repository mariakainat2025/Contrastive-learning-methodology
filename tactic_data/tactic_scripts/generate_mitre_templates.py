import json
import re
import os

PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MITRE_CTI_URL = os.path.join(PROJECT_ROOT, 'input', 'enterprise-attack.json')
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'templates')
OUTPUT_DIR_DC = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'templates_dc')

AUDITD_CHANNELS = {
    'auditd:SYSCALL',
}

_SYSCALL_PREFIX = re.compile(r'^(?:[\w/:]+(?:,\s*[\w/:]+)*)\s*:\s+', re.I)

_NOISE_DISCARD = re.compile(
    r'^(Correlates?\b|Defender\s+(?:correlates?\b|view\b)|Use\s+auditd\b|'
    r'Cross-correlate\b|Combine\b.*\blog\b|Detectable\s+via\b|'
    r'Monitors?\s+(?!for\b|audit\b))',
    re.I
)

_NOISE_STRIP = re.compile(
    r'^Monitors?\s+(?:audit\s+logs?\s+for|for)\s+',
    re.I
)

_CHAIN_PREFIX    = re.compile(r'^[^.!?]*?\bchain:\s*', re.I)
_STEP_CONNECTORS = ['', ' followed by ', ' then ', ' resulting in ', ' leading to ']


def _chain_to_sentence(text):
    text = _CHAIN_PREFIX.sub('', text).strip()
    parts = re.split(r'[;,]?\s*(?<!\w)\(?\d+\)\s*', text)
    parts = [p.strip().rstrip(';,.') for p in parts if p.strip()]
    if len(parts) <= 1:
        return text
    result = parts[0]
    for i, part in enumerate(parts[1:], 1):
        conn = _STEP_CONNECTORS[i] if i < len(_STEP_CONNECTORS) else ', '
        result += conn + part
    return result.strip().rstrip('.') + '.'


def _clean_an_description(desc):
    sentences = re.split(r'(?<=[.!?])\s+', desc.strip())
    kept = []
    for s in sentences:
        s = s.strip()
        if not s or _NOISE_DISCARD.match(s):
            continue
        s = _NOISE_STRIP.sub('', s).strip()
        if not s:
            continue
        s = s[0].upper() + s[1:]
        kept.append(_chain_to_sentence(s))
    return ' '.join(kept)


def _clean_channel(channel):
    text = channel.strip()
    text = re.sub(r'(?<![A-Za-z])odification\b', 'Modification', text)
    if text:
        text = text[0].upper() + text[1:]
    return text


_REGEX_PATTERN = re.compile(r'\.\*|(?<![/\w])\*(?!\w)|\[\^|\(\?|\\[dwsWDS]|\|(?=[a-z])')


def _is_descriptive(channel):
    if not channel or channel.lower() == 'none' or len(channel) <= 5:
        return False
    if re.match(r'^type\s*=\s*(EXECVE|SYSCALL)', channel, re.I):
        return False
    if re.search(r'\bIN\s*\(["\']', channel, re.I):
        return False
    tokens = re.split(r'[,/\s]+', channel.strip())
    if all(re.match(r'^[a-zA-Z_0-9\(\)]+$', t) for t in tokens if t) and len(tokens) <= 6:
        return False
    if _REGEX_PATTERN.search(channel):
        return False
    return True


def analytic_has_auditd(analytic):
    for lsr in analytic.get('x_mitre_log_source_references', []):
        if lsr.get('name', '') in AUDITD_CHANNELS:
            return True
    return False


def extract_auditd_text(analytic):
    lines = []
    bare  = False
    for lsr in analytic.get('x_mitre_log_source_references', []):
        if lsr.get('name', '') not in AUDITD_CHANNELS:
            continue
        cleaned = _clean_channel((lsr.get('channel') or '').strip())
        if _is_descriptive(cleaned):
            if cleaned not in lines:
                lines.append(cleaned)
        else:
            bare = True

    if lines:
        return lines

    desc = _clean_an_description((analytic.get('description') or '').strip())
    if desc:
        return [desc]

    return []


_DROP_SENTENCE = [
    r'^Correlate\b',
    r'^Focus\s+on\s+correlation',
    r'^Detection\s+involves',
]

_STRIP_PREFIX = [
    r'^Detect(?:s|(?:ion\s+of))?\s+',
    r'^Monitor(?:ing)?\s+(?:for\s+)?',
    r'^Behavioral\s+chain:\s+',
    r'^Chain:\s+',
    r'^Defender\s+sees:\s+',
    r'^Anomalies\s+include\s+',
    r'^Correlated\s+evidence\s+(?:of\s+)?',
    r'^Adversary\s+(?:may\s+)?(?:exploits?|uses?|leverages?)\s+',
]

_STRIP_INLINE = [
    r'\s*\(proxy/NGFW\s+logs[^)]*\)(?:\s*\+\s*)?',
    r'\s+by\s+monitoring\s+[\w_/]+\s+logs(?:,\s+[\w\s]+(?:events|logs|calls)[^,.]*)*',
    r',?\s*auditd\s+SYSCALL[^,;.]*',
    r',?\s+and\s+DHCP/Zeek',
    r'\(e\.g\.,\s*Thunderbird[^)]+\)',
]

_ATTACK_SIGNAL = re.compile(
    r'\b(spawn|execut|writ|connect|download|inject|escalat|steal|drop|'
    r'curl|wget|sshd?|sudo|python|/bin/sh|shell|credential|payload|'
    r'malware|webshell|outbound|/tmp|login|logon|auth|exploit|package|'
    r'phish|attachment|usb|hotplug|wireless|association|ssid|rogue|'
    r'process|file|network|command|script|binary|service|registry)\b', re.I)


def clean_linux_analytic(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    kept = []
    for s in sentences:
        if any(re.match(p, s, re.I) for p in _DROP_SENTENCE):
            continue
        for p in _STRIP_PREFIX:
            s = re.sub(p, '', s, flags=re.I)
        for p in _STRIP_INLINE:
            s = re.sub(p, ' ', s, flags=re.I)
        s = re.sub(r'\s+', ' ', s).strip().strip(',').strip()
        s = re.sub(r'\s+([.!?,])', r'\1', s)
        if len(s) > 15 and _ATTACK_SIGNAL.search(s):
            kept.append(s)
    return ' '.join(kept)


TACTIC_ORDER = [
    "TA0043", "TA0042", "TA0001", "TA0002", "TA0003",
    "TA0004", "TA0005", "TA0112", "TA0006", "TA0007",
    "TA0008", "TA0009", "TA0011", "TA0010", "TA0040"
]

TACTIC_NAMES = [
    "Reconnaissance", "Resource Development", "Initial Access", "Execution",
    "Persistence", "Privilege Escalation", "Defense Evasion", "Stealth",
    "Defense Impairment", "Credential Access", "Discovery", "Lateral Movement",
    "Collection", "Command and Control", "Exfiltration", "Impact"
]

OUTCOME_KEYWORDS      = ["goal", "gained through", "provide the opportunity"]
RELATIONSHIP_KEYWORDS = ["lifecycle", "paired with"]


def is_outcome(sentence, current_tactic_name=""):
    return any(kw in sentence.lower() for kw in OUTCOME_KEYWORDS)


def is_relationship(sentence, current_tactic_name=""):
    if any(kw in sentence.lower() for kw in RELATIONSHIP_KEYWORDS):
        return True
    for tname in TACTIC_NAMES:
        if current_tactic_name and tname.lower() == current_tactic_name.lower():
            continue
        if re.search(r'\b' + re.escape(tname.lower()) + r'\b', sentence.lower()):
            if re.search(r'\b' + re.escape(tname.lower()) + r'\s+(techniques|tactic|phase)', sentence.lower()):
                return True
    return False


def parse_tactic_description(description, tactic_name=""):
    raw = description.strip()
    raw = re.sub(r'(\*\s+[^\n*]+?)(\n)(?!\s*\*)', r'\1.\2', raw)
    raw = re.sub(r'\n\s*\*\s+', ', ', raw)
    raw = re.sub(r'\n', ' ', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', raw) if s.strip()]
    if not sentences:
        return "", [], [], []

    objective, technique_use, outcome, relationship = sentences[0], [], [], []
    for sentence in sentences[1:]:
        if is_outcome(sentence, current_tactic_name=tactic_name):
            outcome.append(sentence)
        elif is_relationship(sentence, current_tactic_name=tactic_name):
            relationship.append(sentence)
        else:
            technique_use.append(sentence)

    return objective, technique_use, outcome, relationship


def wrap(text, indent="  ", width=68):
    words = text.split()
    lines, cur = [], indent
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = indent + w
        else:
            cur += (" " if cur.strip() else "") + w
    if cur.strip():
        lines.append(cur)
    return lines


def build_tactic_output(tactic, sorted_techs, tech_analytics, dc_filter=False):
    tactic_id = ""
    for ref in tactic.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            tactic_id = ref["external_id"]

    tactic_name = tactic["name"]
    objective, technique_use, outcome, relationship = parse_tactic_description(
        tactic.get("description", ""), tactic_name=tactic_name
    )

    sep  = "=" * 70
    dash = "-" * 70
    lines = []
    lines.append(sep)
    lines.append(f"  MITRE ATT&CK — TACTIC: {tactic_id}")
    lines.append(sep)
    lines.append(f"  ID         : {tactic_id}")
    lines.append(f"  NAME       : {tactic_name}")
    lines.append("")
    lines.append(f"  OBJECTIVE  : {objective}")
    lines.append("")
    lines.append(dash)
    lines.append("  TECHNIQUES USE")
    lines.append(dash)
    lines.append("")
    if technique_use:
        combined = " ".join(technique_use)
        words = combined.split()
        current_line = "  "
        for word in words:
            if len(current_line) + len(word) + 1 > 68:
                lines.append(current_line)
                current_line = "  " + word
            else:
                current_line += (" " if current_line.strip() else "") + word
        if current_line.strip():
            lines.append(current_line)
    else:
        lines.append("  —")
    lines.append("")

    if outcome + relationship:
        lines.append(dash)
        lines.append("  OUTCOME / TACTIC LINKS")
        lines.append(dash)
        lines.append("")
        combined = " ".join(outcome + relationship)
        words = combined.split()
        current_line = "  "
        for word in words:
            if len(current_line) + len(word) + 1 > 68:
                lines.append(current_line)
                current_line = "  " + word
            else:
                current_line += (" " if current_line.strip() else "") + word
        if current_line.strip():
            lines.append(current_line)
        lines.append("")

    lines.append(dash)
    lines.append("  LINUX DETECTION STRATEGIES")
    lines.append(dash)
    lines.append("")

    if tech_analytics:
        for tid, tname, descriptions in tech_analytics:
            parts = []
            for desc in descriptions:
                if dc_filter:
                    s = desc.strip().rstrip('.')
                    if s:
                        parts.append(s)
                else:
                    cleaned = clean_linux_analytic(desc)
                    if cleaned:
                        parts.append(cleaned.rstrip('.'))
            if not parts:
                continue
            sentence = '. '.join(parts) + '.'
            for ln in wrap(sentence, indent="    "):
                lines.append(ln)
            lines.append("")
    else:
        lines.append("  No Linux detection strategies found for this tactic.")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines), tactic_id, tactic_name


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dc-filter', action='store_true',
                    help='Only include analytics with auditd/syscall channels')
    args = ap.parse_args()

    out_dir = OUTPUT_DIR_DC if args.dc_filter else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("Loading MITRE ATT&CK data...")
    with open(MITRE_CTI_URL, "r", encoding="utf-8") as f:
        data = json.load(f)
    objects = data["objects"]
    lookup  = {o["id"]: o for o in objects}

    tactic_map = {}
    for obj in objects:
        if obj.get("type") == "x-mitre-tactic":
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    tactic_map[ref["external_id"]] = obj

    mode = "auditd/syscall-only" if args.dc_filter else "full"
    print(f"Processing {len(TACTIC_ORDER)} tactics [{mode}]...\n")

    for tactic_id in TACTIC_ORDER:
        tactic = tactic_map.get(tactic_id)
        if not tactic:
            print(f"  WARNING: {tactic_id} not found, skipping.")
            continue

        tactic_shortname = tactic.get("x_mitre_shortname", "")
        tactic_name      = tactic["name"]

        tech_objects = {}
        for obj in objects:
            if (obj.get("type") == "attack-pattern"
                    and not obj.get("x_mitre_is_subtechnique", False)
                    and not obj.get("x_mitre_deprecated", False)
                    and not obj.get("revoked", False)):
                phases = [p["phase_name"] for p in obj.get("kill_chain_phases", [])]
                if tactic_shortname in phases:
                    tid = ""
                    for ref in obj.get("external_references", []):
                        if ref.get("source_name") == "mitre-attack":
                            tid = ref["external_id"]
                    tech_objects[obj["id"]] = {"tech_id": tid, "name": obj["name"]}

        sorted_techs   = sorted(tech_objects.items(), key=lambda x: x[1]["name"])
        tech_analytics = []

        for stix_id, tech in sorted_techs:
            descriptions = []
            for rel in objects:
                if (rel.get("type") == "relationship"
                        and rel.get("relationship_type") == "detects"
                        and rel.get("target_ref") == stix_id):
                    ds = lookup.get(rel["source_ref"], {})
                    for aref in ds.get("x_mitre_analytic_refs", []):
                        analytic = lookup.get(aref, {})
                        if "Linux" not in (analytic.get("x_mitre_platforms") or []):
                            continue
                        if args.dc_filter:
                            if not analytic_has_auditd(analytic):
                                continue
                            for cl in extract_auditd_text(analytic):
                                if cl not in descriptions:
                                    descriptions.append(cl)
                        else:
                            desc = analytic.get("description", "").strip()
                            if desc and desc not in descriptions:
                                descriptions.append(desc)

            if descriptions:
                tech_analytics.append((tech["tech_id"], tech["name"], descriptions))

        output, tid, tname = build_tactic_output(tactic, sorted_techs, tech_analytics, dc_filter=args.dc_filter)

        safe_name = tname.replace(" ", "_").replace("/", "-")
        filename  = f"{tid}_{safe_name}.txt"

        with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
            f.write(output)

        print(f"  [{tid}] {tname} — {len(sorted_techs)} techniques → {filename}")

    print(f"\nDone! Templates → {out_dir}")


if __name__ == "__main__":
    main()
