"""Hash of the server code, so run.py can tell when a running server is outdated."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).parent
FILES = ["server.py", "predict.py", "bench.py", "version.py", "data/gpus.json", "data/models.json"]


def code_version() -> str:
    h = hashlib.sha256()
    for name in FILES:
        p = ROOT / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]
