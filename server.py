"""FastAPI server: stores benchmark results, serves predictions and the site.

    uvicorn server:app --reload
"""

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import bench
import predict
from version import code_version

DB_PATH = Path(os.environ.get("LLM_BENCH_DB") or Path(__file__).parent / "bench.db")
app = FastAPI(title="LLM Bench")
VERSION = code_version()


@app.get("/api/version")
def version():
    return {"app": "llm-bench", "version": VERSION}


# Limits reject impossible numbers (bugs or fake submissions).

class ModelResult(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    quantization: str | None = Field(default=None, max_length=50)
    parameter_size: str | None = Field(default=None, max_length=50)
    avg_tokens_per_sec: float = Field(gt=0, lt=2000)
    avg_ttft_ms: float = Field(ge=0, lt=600_000)
    vram_bytes: int | None = Field(default=None, ge=0)
    gpu_offload_pct: float | None = Field(default=None, ge=0, le=100)
    runs: int = Field(ge=1, le=100)


class Submission(BaseModel):
    timestamp: int
    os: str = Field(max_length=200)
    cpu: str = Field(max_length=200)
    gpu: str = Field(min_length=1, max_length=100)
    ollama_version: str | None = Field(default=None, max_length=50)
    results: list[ModelResult] = Field(min_length=1, max_length=50)


# One submission (a machine + run) has many model results.
# DATABASE_URL set (the hosted site) -> Postgres. Otherwise a local SQLite file.
DATABASE_URL = os.environ.get("DATABASE_URL")
PG = bool(DATABASE_URL)
ID = "SERIAL PRIMARY KEY" if PG else "INTEGER PRIMARY KEY AUTOINCREMENT"

SCHEMA = [
    f"""CREATE TABLE IF NOT EXISTS submissions (
        id             {ID},
        received_at    INTEGER NOT NULL,
        gpu            TEXT NOT NULL,
        cpu            TEXT,
        os             TEXT,
        ollama_version TEXT,
        shared_at      INTEGER
    )""",
    f"""CREATE TABLE IF NOT EXISTS results (
        id                 {ID},
        submission_id      INTEGER NOT NULL REFERENCES submissions(id),
        model              TEXT NOT NULL,
        quantization       TEXT,
        parameter_size     TEXT,
        avg_tokens_per_sec REAL NOT NULL,
        avg_ttft_ms        REAL NOT NULL,
        vram_bytes         BIGINT,
        gpu_offload_pct    REAL,
        runs               INTEGER NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_submissions_gpu ON submissions(gpu)",
]


class DB:
    """Small wrapper so the rest of the code writes one SQL dialect (? placeholders)."""

    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql: str, params=()):
        if PG:
            sql = sql.replace("?", "%s")
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return cur

    def all(self, sql: str, params=()) -> list[dict]:
        return [dict(r) for r in self.execute(sql, params).fetchall()]


@contextmanager
def db():
    if PG:
        import psycopg
        from psycopg.rows import dict_row
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
    try:
        yield DB(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


with db() as conn:
    for stmt in SCHEMA:
        conn.execute(stmt)
    # migration for databases created before this column existed
    if PG:
        conn.execute("ALTER TABLE results ADD COLUMN IF NOT EXISTS gpu_offload_pct REAL")
        conn.execute("ALTER TABLE submissions ADD COLUMN IF NOT EXISTS shared_at INTEGER")
    else:
        for table, col, kind in [("results", "gpu_offload_pct", "REAL"), ("submissions", "shared_at", "INTEGER")]:
            if col not in {r["name"] for r in conn.all(f"PRAGMA table_info({table})")}:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {kind}")


def save_submission(sub: Submission) -> int:
    with db() as conn:
        sid = conn.execute(
            "INSERT INTO submissions (received_at, gpu, cpu, os, ollama_version) "
            "VALUES (?, ?, ?, ?, ?) RETURNING id",
            (int(time.time()), sub.gpu.strip(), sub.cpu, sub.os, sub.ollama_version),
        ).fetchone()
        sid = sid["id"] if isinstance(sid, dict) else sid[0]
        for r in sub.results:
            conn.execute(
                "INSERT INTO results (submission_id, model, quantization, parameter_size, "
                "avg_tokens_per_sec, avg_ttft_ms, vram_bytes, gpu_offload_pct, runs) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (sid, r.model, r.quantization, r.parameter_size,
                 r.avg_tokens_per_sec, r.avg_ttft_ms, r.vram_bytes, r.gpu_offload_pct, r.runs),
            )
    return sid


# Uploads per IP per hour on the public site. In memory, which is fine for one instance.
UPLOAD_LIMIT = 20
_uploads: dict[str, list[float]] = {}


def client_ip(request: Request) -> str:
    # behind a host's proxy the real address is the first X-Forwarded-For entry
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def check_rate_limit(request: Request) -> None:
    now = time.time()
    ip = client_ip(request)
    recent = [t for t in _uploads.get(ip, []) if now - t < 3600]
    if len(recent) >= UPLOAD_LIMIT:
        raise HTTPException(429, "Too many uploads from this address. Try again in an hour.")
    recent.append(now)
    _uploads[ip] = recent


@app.post("/api/submissions", status_code=201)
def create_submission(sub: Submission, request: Request):
    check_rate_limit(request)
    return {"id": save_submission(sub)}


@app.get("/api/results")
def list_results(gpu: str | None = None):
    """One row per (GPU, model, quantization), averaged across submissions."""
    sql = """
        SELECT s.gpu, r.model, r.quantization,
               MAX(r.parameter_size)    AS parameter_size,
               AVG(r.avg_tokens_per_sec) AS tokens_per_sec,
               AVG(r.avg_ttft_ms)        AS ttft_ms,
               MAX(r.vram_bytes)         AS vram_bytes,
               MIN(r.gpu_offload_pct)    AS gpu_offload_pct,
               COUNT(*)                  AS submissions
        FROM results r JOIN submissions s ON s.id = r.submission_id
        {where}
        GROUP BY s.gpu, r.model, r.quantization
    """
    where, params = "", ()
    if gpu:
        where, params = "WHERE s.gpu = ?", (gpu,)
    with db() as conn:
        rows = conn.all(sql.format(where=where), params)
    for r in rows:  # rounding here because ROUND() differs between SQLite and Postgres
        r["tokens_per_sec"] = round(r["tokens_per_sec"], 1)
        r["ttft_ms"] = round(r["ttft_ms"])
    return sorted(rows, key=lambda r: (r["gpu"], -r["tokens_per_sec"]))


@app.get("/api/gpus")
def list_gpus():
    return predict.GPUS


@app.get("/api/categories")
def list_categories():
    return predict.TAGS


@app.get("/api/predictions")
def predictions(gpu: str):
    out = predict.predict_all(gpu, list_results())
    if out is None:
        raise HTTPException(404, f"Unknown GPU: {gpu}")
    return out


@app.get("/api/estimate")
def estimate(gpu: str, model: str, bits: float = predict.BITS_PER_WEIGHT, ctx: int = predict.DEFAULT_CTX):
    if not (1 <= bits <= 16 and 512 <= ctx <= 262_144):
        raise HTTPException(400, "bits must be 1-16 and ctx 512-262144")
    out = predict.estimate(gpu, model, bits, ctx, list_results())
    if out is None:
        raise HTTPException(404, "Unknown GPU or model")
    return out


@app.get("/api/accuracy")
def accuracy():
    return predict.accuracy(list_results())


# Local mode (set by run.py): lets the site download and benchmark models
# through the user's own Ollama. Never enabled on a public deployment.

LOCAL = os.environ.get("LLM_BENCH_LOCAL") == "1"
# public site that local installs can share results with
COMMUNITY_URL = os.environ.get("LLM_BENCH_COMMUNITY_URL", "").rstrip("/")
MODEL_NAME = re.compile(r"^[\w.\-/:]{1,200}$")
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def require_local(request: Request) -> None:
    if not LOCAL:
        raise HTTPException(403, "Only available when running LLM Bench on your own PC.")
    # block requests from other websites to 127.0.0.1
    origin = request.headers.get("origin")
    if origin and origin not in (f"http://{request.url.netloc}",):
        raise HTTPException(403, "Requests from other websites are not allowed.")


@lru_cache  # PowerShell detection is slow
def detected_gpu() -> str | None:
    return bench.detect_gpu()


def ollama_models() -> list[str] | None:
    try:
        return [m["name"] for m in bench.installed_models()]
    except requests.RequestException:
        return None


@app.get("/api/local")
def local_info():
    if not LOCAL:
        return {"local": False}
    gpu = detected_gpu()
    match = predict.find_gpu(gpu) if gpu else None
    installed = ollama_models()
    return {
        "local": True,
        "models_dir": os.environ.get("OLLAMA_MODELS") or str(Path.home() / ".ollama" / "models"),
        "gpu_detected": gpu,
        "gpu_catalog": match["name"] if match else None,
        "ollama_running": installed is not None,
        "installed": installed or [],
        "community_url": COMMUNITY_URL or None,
    }


class ModelRequest(BaseModel):
    model: str
    share: bool = False


def submission_payload(sid: int) -> dict:
    """Rebuild the upload format for a stored submission."""
    with db() as conn:
        s = conn.all("SELECT * FROM submissions WHERE id = ?", (sid,))[0]
        rows = conn.all("SELECT * FROM results WHERE submission_id = ?", (sid,))
    keep = ["model", "quantization", "parameter_size", "avg_tokens_per_sec",
            "avg_ttft_ms", "vram_bytes", "gpu_offload_pct", "runs"]
    return {"timestamp": s["received_at"], "os": s["os"] or "", "cpu": s["cpu"] or "",
            "gpu": s["gpu"], "ollama_version": s["ollama_version"],
            "results": [{k: r[k] for k in keep} for r in rows]}


def share(sid: int) -> None:
    """Send one local submission to the public site and remember that we did."""
    if not COMMUNITY_URL:
        raise RuntimeError("No public site is configured.")
    r = requests.post(f"{COMMUNITY_URL}/api/submissions", json=submission_payload(sid), timeout=90)
    if not r.ok:
        raise RuntimeError(f"Public site said {r.status_code}: {r.text[:200]}")
    with db() as conn:
        conn.execute("UPDATE submissions SET shared_at = ? WHERE id = ?", (int(time.time()), sid))


def start_job(kind: str, model: str, work) -> dict:
    """Run a download/benchmark in a background thread; the page polls its status.
    One at a time, since two jobs on the same GPU skew each other."""
    with JOBS_LOCK:
        if any(j["status"] == "running" for j in JOBS.values()):
            raise HTTPException(409, "Another download or benchmark is still running.")
        job = {"id": uuid.uuid4().hex[:8], "kind": kind, "model": model,
               "status": "running", "progress": 0, "message": "Starting..."}
        JOBS[job["id"]] = job

    def runner():
        try:
            work(job)
            job.update(status="done", progress=100)
        except Exception as e:
            job.update(status="error", message=str(e)[:300])

    threading.Thread(target=runner, daemon=True).start()
    return job


@app.post("/api/local/pull")
def pull_model(req: ModelRequest, request: Request):
    require_local(request)
    if not MODEL_NAME.match(req.model):
        raise HTTPException(400, "That doesn't look like a model name.")

    def work(job):
        with requests.post(f"{bench.OLLAMA}/api/pull", json={"model": req.model, "stream": True},
                           stream=True, timeout=(10, 600)) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                if msg.get("total"):
                    job["progress"] = round(99 * msg.get("completed", 0) / msg["total"])
                job["message"] = msg.get("status", "")
        if req.model not in (ollama_models() or []):
            raise RuntimeError("Download finished but Ollama doesn't list the model. Try `ollama pull` in a terminal.")
        job["message"] = "Saved to Ollama's model folder"

    return start_job("pull", req.model, work)


@app.post("/api/local/remove")
def remove_model(req: ModelRequest, request: Request):
    require_local(request)
    if any(j["status"] == "running" and j["model"] == req.model for j in JOBS.values()):
        raise HTTPException(409, "That model is busy right now.")
    r = requests.delete(f"{bench.OLLAMA}/api/delete", json={"model": req.model}, timeout=60)
    if not r.ok:
        raise HTTPException(r.status_code, r.text[:300])
    return {"removed": req.model}


@app.post("/api/local/benchmark")
def benchmark_model(req: ModelRequest, request: Request):
    require_local(request)
    installed = ollama_models()
    if installed is None:
        raise HTTPException(503, "Ollama isn't running. Open the Ollama app and try again.")
    if req.model not in installed:
        raise HTTPException(400, f"{req.model} isn't downloaded yet.")

    def work(job):
        def progress(done, total):
            job["progress"] = round(99 * done / total)
            job["message"] = "Loading model..." if done == 0 else f"Run {done} of {total}"
        result = bench.measure(req.model, progress)
        sub = bench.make_submission(detected_gpu() or "Unknown GPU", [result])
        sid = save_submission(Submission(**sub))
        job["result"] = result
        if req.share and COMMUNITY_URL:
            job["message"] = "Sharing with the public site..."
            try:
                share(sid)
                job["shared"] = True
            except Exception as e:  # sharing is optional; keep the local result either way
                job["shared"] = False
                job["share_error"] = str(e)[:200]

    return start_job("benchmark", req.model, work)


@app.post("/api/local/share-all")
def share_all(request: Request):
    """Upload every local result that hasn't been shared yet."""
    require_local(request)
    if not COMMUNITY_URL:
        raise HTTPException(400, "No public site is configured.")
    with db() as conn:
        ids = [r["id"] for r in conn.all("SELECT id FROM submissions WHERE shared_at IS NULL")]
    shared, errors = 0, []
    for sid in ids:
        try:
            share(sid)
            shared += 1
        except Exception as e:
            errors.append(str(e)[:200])
            break  # the site is probably asleep or down; don't hammer it
    return {"shared": shared, "remaining": len(ids) - shared, "error": errors[0] if errors else None}


@app.get("/api/local/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "No such job")
    return JOBS[job_id]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
