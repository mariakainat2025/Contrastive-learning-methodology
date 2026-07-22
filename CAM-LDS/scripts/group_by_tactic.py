import json
import shutil
from pathlib import Path

TECHNIQUES_DIR = Path("/csse/research/contructive-learning/aaaa/manifestations_filtered/manifestations_filtered/techniques")
OUTPUT_DIR     = Path("/csse/research/contructive-learning/CAM-LDS/grouped_by_tactic")
STIX_PATH      = Path(__file__).parent / "enterprise-attack.json"

with open(STIX_PATH) as f:
    stix = json.load(f)

matrix = {}
for obj in stix["objects"]:
    if obj["type"] != "attack-pattern":
        continue
    tid = None
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            tid = ref.get("external_id")
            break
    if not tid:
        continue
    tactics = [
        phase["phase_name"]
        for phase in obj.get("kill_chain_phases", [])
        if phase.get("kill_chain_name") == "mitre-attack"
    ]
    matrix[tid] = {"name": obj.get("name", ""), "tactics": tactics}


def folder_to_mitre_id(folder_name):
    parts = folder_name.split("-")
    base = parts[0]
    sub  = parts[1] if len(parts) > 1 else "000"
    return base if sub == "000" else f"{base}.{sub}"


def get_tactics(mitre_id):
    if mitre_id in matrix:
        return matrix[mitre_id]["tactics"]
    parent = mitre_id.split(".")[0]
    if parent in matrix:
        return matrix[parent]["tactics"]
    return None


summary = {}

for folder in sorted(TECHNIQUES_DIR.iterdir()):
    if not folder.is_dir():
        continue

    mitre_id = folder_to_mitre_id(folder.name)
    tactics  = get_tactics(mitre_id)

    if not tactics:
        continue

    for tactic in tactics:
        tactic_name = tactic.replace("-", "_")
        dest = OUTPUT_DIR / tactic_name / folder.name

        if not dest.exists():
            shutil.copytree(str(folder), str(dest))

        summary.setdefault(tactic_name, []).append(folder.name)
