"""Speed and memory predictions.

Each token requires reading every active weight from memory once, so speed is
bound by memory bandwidth:

    time per token = weights (GB) / (efficiency * bandwidth) + fixed overhead

`efficiency` is learned from real benchmark results.
"""

import json
import re
from pathlib import Path
from statistics import median

DATA = Path(__file__).parent / "data"
GPUS = json.loads((DATA / "gpus.json").read_text())["gpus"]
_CATALOG = json.loads((DATA / "models.json").read_text())
MODELS = _CATALOG["models"]
TAGS = _CATALOG["tags"]

BITS_PER_WEIGHT = 4.85     # Q4_K_M average
RUNTIME_GB = 0.5
DEFAULT_CTX = 8192
DEFAULT_EFFICIENCY = 0.55  # until there's real data
RANGE = 0.20
SYSTEM_RAM_GBS = 40        # dual-channel DDR4, used when a model spills out of VRAM
# Fixed per-token cost. Matters for tiny models: without it a 0.6B model
# measured as half as efficient as a 9B one on the same GPU.
PER_TOKEN_OVERHEAD_S = 0.001


def weights_gb(params_b: float, bits: float = BITS_PER_WEIGHT) -> float:
    return params_b * bits / 8


def find_gpu(name: str) -> dict | None:
    """Longest match wins, so 'RTX 4060 Ti' doesn't match 'RTX 4060'."""
    norm = name.upper()
    hits = [g for g in GPUS if g["name"].upper() in norm]
    return max(hits, key=lambda g: len(g["name"])) if hits else None


def find_model(name: str) -> dict | None:
    return next((m for m in MODELS if m["name"] == name), None)


def params_from_text(size: str | None) -> float | None:
    """'9.0B' -> 9.0, '270M' -> 0.27"""
    m = re.match(r"([\d.]+)\s*([BM])", size or "", re.I)
    if not m:
        return None
    return float(m[1]) * (1 if m[2].upper() == "B" else 0.001)


def active_gb_for(result: dict) -> float | None:
    """Weights read per token. Uses Ollama's reported size, except for MoE models."""
    cat = find_model(result["model"])
    if cat and cat["active_params_b"] != cat["params_b"]:
        return weights_gb(cat["active_params_b"])
    p = params_from_text(result.get("parameter_size"))
    if p:
        return weights_gb(p)
    return weights_gb(cat["active_params_b"]) if cat else None


def speed(active_gb: float, eff: float, bandwidth_gbs: float) -> float:
    return 1 / (active_gb / (eff * bandwidth_gbs) + PER_TOKEN_OVERHEAD_S)


def efficiency_of(result: dict) -> float | None:
    """Efficiency implied by a measurement. Skips runs that spilled out of VRAM."""
    gpu = find_gpu(result["gpu"])
    gb = active_gb_for(result)
    if not gpu or not gb or (result.get("gpu_offload_pct") or 100) < 99:
        return None
    read_time = 1 / result["tokens_per_sec"] - PER_TOKEN_OVERHEAD_S
    if read_time <= 0:
        return None
    return gb / (gpu["bandwidth_gbs"] * read_time)


def learn_efficiency(measured: list[dict], gpu_name: str) -> tuple[float, str]:
    """Median efficiency for this GPU, falling back to all GPUs, then the default."""
    same = [e for r in measured if find_gpu(r["gpu"]) and find_gpu(r["gpu"])["name"] == gpu_name
            if (e := efficiency_of(r))]
    if same:
        return median(same), f"learned from {len(same)} result(s) on this GPU"
    every = [e for r in measured if (e := efficiency_of(r))]
    if every:
        return median(every), f"learned from {len(every)} result(s) on other GPUs"
    return DEFAULT_EFFICIENCY, "default estimate (no data yet)"


def kv_cache_gb(params_b: float, ctx: int) -> float:
    """Rough fp16 KV cache size. Fitted through ~32 KB/token at 1B,
    128 KB at 8B and 320 KB at 70B."""
    kb_per_token = 128 * (params_b / 8) ** 0.42
    return kb_per_token * 1024 * ctx / 1e9


def memory_breakdown(model: dict, bits: float = BITS_PER_WEIGHT, ctx: int = DEFAULT_CTX) -> dict:
    w = weights_gb(model["params_b"], bits)
    kv = kv_cache_gb(model["params_b"], ctx)
    return {"weights_gb": w, "kv_gb": kv, "runtime_gb": RUNTIME_GB, "total_gb": w + kv + RUNTIME_GB}


def predict(gpu: dict, model: dict, eff: float, bits: float = BITS_PER_WEIGHT, ctx: int = DEFAULT_CTX) -> dict:
    mem = memory_breakdown(model, bits, ctx)
    need_gb = mem["total_gb"]
    active_gb = weights_gb(model["active_params_b"], bits)
    fits = need_gb <= gpu["vram_gb"]

    if fits:
        tps = speed(active_gb, eff, gpu["bandwidth_gbs"])
        lo, hi, confidence = tps * (1 - RANGE), tps * (1 + RANGE), "good"
    else:
        # weights that don't fit are read from system RAM
        room_for_weights = max(0.0, gpu["vram_gb"] - mem["kv_gb"] - RUNTIME_GB)
        on_gpu = min(1.0, room_for_weights / mem["weights_gb"])
        t = (active_gb * on_gpu) / (eff * gpu["bandwidth_gbs"]) + \
            (active_gb * (1 - on_gpu)) / (eff * SYSTEM_RAM_GBS) + PER_TOKEN_OVERHEAD_S
        tps = 1 / t
        lo, hi, confidence = tps * 0.5, tps * 1.5, "rough"

    return {
        "model": model["name"],
        "tags": model.get("tags", []),
        "desc": model.get("desc", ""),
        "display": model.get("display", model["name"]),
        "maker": model.get("maker"),
        "params_b": model["params_b"],
        "moe": model["active_params_b"] != model["params_b"],
        "size_gb": round(mem["weights_gb"], 1),
        "needs_gb": round(need_gb, 1),
        "memory": {k: round(v, 2) for k, v in mem.items()},
        "fits": fits,
        "tokens_per_sec": round(tps, 1),
        "low": round(lo, 1),
        "high": round(hi, 1),
        "confidence": confidence,
    }


def estimate(gpu_name: str, model_name: str, bits: float, ctx: int, measured: list[dict]) -> dict | None:
    """Single model/GPU estimate at a given quantization and context length."""
    gpu = next((g for g in GPUS if g["name"] == gpu_name), None)
    model = find_model(model_name)
    if not gpu or not model:
        return None
    eff, _ = learn_efficiency(measured, gpu_name)
    return {"gpu": gpu, **predict(gpu, model, eff, bits, ctx)}


def predict_all(gpu_name: str, measured: list[dict]) -> dict | None:
    gpu = next((g for g in GPUS if g["name"] == gpu_name), None)
    if not gpu:
        return None
    eff, source = learn_efficiency(measured, gpu_name)
    on_this_gpu = [r for r in measured
                   if (g := find_gpu(r["gpu"])) and g["name"] == gpu_name]
    preds = [predict(gpu, m, eff) for m in MODELS]

    # benchmarked models that aren't in the catalog
    for name in {r["model"] for r in on_this_gpu} - {m["name"] for m in MODELS}:
        r = next(x for x in on_this_gpu if x["model"] == name)
        p = params_from_text(r.get("parameter_size"))
        size = weights_gb(p) if p else 0
        preds.append({
            "model": name, "tags": [], "desc": "Not in our catalog; measured on a real PC.",
            "display": name, "maker": None,
            "params_b": p, "moe": False, "size_gb": round(size, 1),
            "needs_gb": round(size + RUNTIME_GB + (kv_cache_gb(p, DEFAULT_CTX) if p else 0), 1),
            "fits": (r.get("gpu_offload_pct") or 100) >= 99,
            "tokens_per_sec": r["tokens_per_sec"], "low": None, "high": None,
            "confidence": "measured", "in_catalog": False,
        })

    for p in preds:
        real = [r for r in on_this_gpu if r["model"] == p["model"]]
        p["measured"] = round(median(r["tokens_per_sec"] for r in real), 1) if real else None
        p["measured_ttft_ms"] = round(median(r["ttft_ms"] for r in real)) if real else None
        p.setdefault("in_catalog", True)
    preds.sort(key=lambda p: (not p["fits"], -(p["measured"] or p["tokens_per_sec"])))
    return {"gpu": gpu, "efficiency": round(eff, 3), "efficiency_source": source, "models": preds}


def accuracy(measured: list[dict]) -> dict:
    """Leave-one-out: predict each result from the others, report mean % error."""
    usable = [r for r in measured if efficiency_of(r)]
    errors = []
    for i, r in enumerate(usable):
        others = [e for j, o in enumerate(usable) if j != i if (e := efficiency_of(o))]
        if not others:
            continue
        gpu = find_gpu(r["gpu"])
        predicted = speed(active_gb_for(r), median(others), gpu["bandwidth_gbs"])
        errors.append(abs(predicted - r["tokens_per_sec"]) / r["tokens_per_sec"])
    if not errors:
        return {"tested": 0, "mean_error_pct": None}
    return {"tested": len(errors), "mean_error_pct": round(100 * sum(errors) / len(errors), 1)}
