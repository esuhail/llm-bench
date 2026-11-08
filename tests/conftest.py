import os
import sys
import tempfile
from pathlib import Path

# use a throwaway database, never the real bench.db
os.environ["LLM_BENCH_DB"] = str(Path(tempfile.mkdtemp()) / "test.db")
os.environ.pop("DATABASE_URL", None)
os.environ.pop("LLM_BENCH_LOCAL", None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
