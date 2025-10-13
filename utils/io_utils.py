import os, orjson

DATA_DIR = "data"

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)

def write_jsonl(path: str, obj):
    with open(path, "ab") as f:
        f.write(orjson.dumps(obj) + b"\n")

def read_jsonl(path: str):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "rb") as f:
        for line in f:
            out.append(orjson.loads(line))
    return out
