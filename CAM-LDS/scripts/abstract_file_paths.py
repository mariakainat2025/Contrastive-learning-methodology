
import os
import json
from pathlib import PurePosixPath

CAM_LDS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH     = os.path.join(CAM_LDS_DIR, "parser", "file_paths.json")
OUT_PATH    = os.path.join(CAM_LDS_DIR, "parser", "file_paths_abstracted.json")

CONF_ROOTS = [PurePosixPath("/etc"), PurePosixPath("/var")]


def lift_etc_var(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None  
    path = PurePosixPath(raw_path)
    if len(path.suffix) <= 1:
        return None  

    for root in CONF_ROOTS:
        if root in path.parents:
            parents = list(path.parents)
            if len(parents) <= 2:
                return None  
            d = parents[-3].name  
            e = path.suffix[1:]
            return f"{root.stem} {d} {e} file"
    return None


def lift_etc_var_root(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None  
    path = PurePosixPath(raw_path)
    if len(path.suffix) > 1:
        return None  

    for root in CONF_ROOTS:
        if path.parent == root:
            return f"{path.name} file"
    return None


BIN_ROOTS = [
    PurePosixPath("/bin"), PurePosixPath("/sbin"),
    PurePosixPath("/usr/bin"), PurePosixPath("/usr/sbin"),
    PurePosixPath("/usr/local/bin"), PurePosixPath("/usr/local/sbin"),
]


def lift_bin(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None  
    path = PurePosixPath(raw_path)
    if len(path.suffix) <= 1:
        return None  

    for root in BIN_ROOTS:
        if root in path.parents:
            f = path.stem
            e = path.suffix[1:]
            return f"{f} {e} file"
    return None


HOME_ROOT = PurePosixPath("/home")


def lift_home(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None  
    path = PurePosixPath(raw_path)
    if len(path.suffix) <= 1:
        return None  
    if HOME_ROOT not in path.parents:
        return None

    parents = list(path.parents)
    if len(parents) <= 3:
        return None  
    d = parents[-4].name  
    f = path.stem
    e = path.suffix[1:]
    return f"user {d} {f} {e} file"


ROOT_HOME = PurePosixPath("/root")


def lift_root(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None  
    path = PurePosixPath(raw_path)
    if len(path.suffix) <= 1:
        return None  
    if ROOT_HOME not in path.parents:
        return None

    parents = list(path.parents)
    if len(parents) <= 2:
        return None  
    d = parents[-3].name  
    f = path.stem
    e = path.suffix[1:]
    return f"root user {d} {f} {e} file"


LIB_NAMES = {"lib", "lib32", "lib64"}


def lift_lib(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None  
    path = PurePosixPath(raw_path)
    if len(path.suffix) <= 1:
        return None  

    parents = list(path.parents)
    for i, p in enumerate(parents):
        if p.name in LIB_NAMES:
            if i == 0:
                return None  
            d = parents[i - 1].name
            return f"{d} library file"
    return None


def lift_other(raw_path):
    if not raw_path.startswith('/'):
        return None
    if raw_path.endswith('/'):
        return None  
    path = PurePosixPath(raw_path)
    if len(path.suffix) <= 1:
        return None  
    e = path.suffix[1:]
    return f"{e} file"


def main():
    with open(IN_PATH) as f:
        data = json.load(f)

    changed = {}
    for raw_path in data["paths"]:
        try:
            lifted = (lift_etc_var(raw_path) or lift_bin(raw_path) or lift_home(raw_path)
                      or lift_root(raw_path) or lift_lib(raw_path) or lift_other(raw_path))
        except Exception as exc:
            print(f"SKIPPED (error: {exc}) : {raw_path}")
            continue
        if lifted is not None and lifted != raw_path:
            changed[raw_path] = lifted
            print(f"CHANGED : {raw_path}  ->  {lifted}")

    print()
    print(f"Total paths checked : {len(data['paths'])}")
    print(f"Paths changed       : {len(changed)}")

    with open(OUT_PATH, "w") as f:
        json.dump(changed, f, indent=2)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()

