import json
import re
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MITRE_CTI_URL = os.path.join(PROJECT_ROOT, 'input', 'enterprise-attack.json')
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'output', 'theia', 'tactic_data', 'templates')

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

OUTCOME_KEYWORDS    = ["goal", "gained through", "provide the opportunity"]
RELATIONSHIP_KEYWORDS = ["lifecycle", "paired with"]


def is_outcome(sentence, current_tactic_name=""):
    s_lower = sentence.lower()
    return any(kw in s_lower for kw in OUTCOME_KEYWORDS)

def is_relationship(sentence, current_tactic_name=""):
    s_lower = sentence.lower()
    if any(kw in s_lower for kw in RELATIONSHIP_KEYWORDS):
        return True
    for tname in TACTIC_NAMES:
        if current_tactic_name and tname.lower() == current_tactic_name.lower():
            continue
        if re.search(r'\b' + re.escape(tname.lower()) + r'\b', s_lower):
            if re.search(r'\b' + re.escape(tname.lower()) + r'\s+(techniques|tactic|phase)', s_lower):
                return True
    return False

def parse_tactic_description(description, tactic_name=""):
    raw = description.strip()
    raw = re.sub(r'(\*\s+[^\n*]+?)(\n)(?!\s*\*)', r'\1.\2', raw)
    raw = re.sub(r'\n\s*\*\s+', ', ', raw)
    raw = re.sub(r'\n', ' ', raw)
    raw = re.sub(r'\s+', ' ', raw).strip()

    sentences = re.split(r'(?<=[.!?])\s+', raw)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return "", [], [], []

    objective      = sentences[0]
    technique_use  = []
    outcome        = []
    relationship   = []

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
    lines = []
    cur = indent
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = indent + w
        else:
            cur += (" " if cur.strip() else "") + w
    if cur.strip():
        lines.append(cur)
    return lines


def build_tactic_output(tactic, sorted_techs, tech_analytics):
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

    combined_or = outcome + relationship
    if combined_or:
        lines.append(dash)
        lines.append("  OUTCOME / TACTIC LINKS")
        lines.append(dash)
        lines.append("")
        combined = " ".join(combined_or)
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
            for desc in descriptions:
                for ln in wrap(desc, indent="  "):
                    lines.append(ln)
                lines.append("")
    else:
        lines.append("  No Linux detection strategies found for this tactic.")
        lines.append("")

    lines.append(sep)
    return "\n".join(lines), tactic_id, tactic_name


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    print(f"Processing {len(TACTIC_ORDER)} tactics...\n")

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
                        desc = analytic.get("description", "").strip()
                        if desc and desc not in descriptions:
                            descriptions.append(desc)
            if descriptions:
                tech_analytics.append((tech["tech_id"], tech["name"], descriptions))

        output, tid, tname = build_tactic_output(tactic, sorted_techs, tech_analytics)

        safe_name = tname.replace(" ", "_").replace("/", "-")
        filename  = f"{tid}_{safe_name}.txt"

        with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
            f.write(output)

        print(f"  [{tid}] {tname} — {len(sorted_techs)} techniques → {filename}")

    print(f"\nDone! Templates → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
