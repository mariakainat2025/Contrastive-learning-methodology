
import os
import json
import glob

CAM_LDS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STIX_PATH   = os.path.join(CAM_LDS_DIR, "enterprise-attack.json")
GRAPHS_DIR  = os.path.join(CAM_LDS_DIR, "graphs")
OUT_DIR     = os.path.join(CAM_LDS_DIR, "templates_dc_b")

TACTICS = [
    "collection", "command_and_control", "credential_access", "defense_impairment",
    "discovery", "execution", "exfiltration", "impact", "initial_access",
    "lateral_movement", "persistence", "privilege_escalation", "reconnaissance", "stealth",
]


def folder_to_mitre_id(folder_name):
    base, sub = folder_name.split("-", 1)
    return base if sub == "000" else f"{base}.{sub}"


def dataset_techniques_for_tactic(tactic):
    folders = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(GRAPHS_DIR, tactic, "*")) if os.path.isdir(p)
    )
    return [folder_to_mitre_id(f) for f in folders]


def build_tech_lookup(objects):
    tech_by_id = {}
    for obj in objects:
        if obj.get("type") == "attack-pattern":
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    tech_by_id[ref["external_id"]] = obj
    return tech_by_id


def linux_analytics_for_technique(stix_id, objects, lookup):
    descriptions = []
    for rel in objects:
        if (rel.get("type") == "relationship" and rel.get("relationship_type") == "detects"
                and rel.get("target_ref") == stix_id):
            ds = lookup.get(rel["source_ref"], {})
            for aref in ds.get("x_mitre_analytic_refs", []):
                analytic = lookup.get(aref, {})
                if "Linux" not in (analytic.get("x_mitre_platforms") or []):
                    continue
                desc = analytic.get("description", "").strip()
                if desc:
                    descriptions.append(desc)
    return descriptions


def main():
    with open(STIX_PATH) as f:
        data = json.load(f)
    objects = data["objects"]
    lookup = {o["id"]: o for o in objects}
    tech_by_id = build_tech_lookup(objects)

    os.makedirs(OUT_DIR, exist_ok=True)

    for tactic in TACTICS:
        mitre_ids = dataset_techniques_for_tactic(tactic)
        sentences = []
        not_found = []
        no_linux = []

        for tid in mitre_ids:
            tech = tech_by_id.get(tid)
            if not tech:
                not_found.append(tid)
                continue
            descs = linux_analytics_for_technique(tech["id"], objects, lookup)
            if not descs:
                no_linux.append(tid)
            for d in descs:
                if d not in sentences:
                    sentences.append(d)

        text = " ".join(sentences)
        out_path = os.path.join(OUT_DIR, f"{tactic}_template_b.txt")
        with open(out_path, "w") as f:
            f.write(text + "\n")

        print(f"{tactic:22s} techniques={len(mitre_ids):2d}  "
              f"analytics={len(sentences):2d}  words={len(text.split()):4d}  "
              f"not_found={not_found}  no_linux_analytic={no_linux}")

    print(f"\nSaved -> {OUT_DIR}")


if __name__ == "__main__":
    main()

