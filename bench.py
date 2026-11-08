"""Benchmark local models through Ollama.

    python bench.py                  pick models from a list
    python bench.py qwen3.5:9b ...   benchmark specific models
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import webbrowser

import requests

OLLAMA = "http://localhost:11434"
DEFAULT_SERVER = "http://127.0.0.1:8000"

# Same prompts everywhere so results are comparable. Short, medium, long output.
PROMPTS = [
    "Explain what a hash map is in two sentences.",
    "Write a Python function that checks if a string is a palindrome, with comments.",
    "Describe how a CPU executes an instruction, step by step, in about 300 words.",
]
RUNS_PER_PROMPT = 3

os.system("")  # enables ANSI colors in the Windows console
BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, RED, CYAN = "\033[92m", "\033[93m", "\033[91m", "\033[96m"


def verdict(tps: float) -> tuple[str, str, str]:
    """Label for a tokens/sec value. ~1 token = 0.75 words."""
    if tps >= 30:
        return "Fast", GREEN, "text appears much faster than you can read"
    if tps >= 15:
        return "Comfortable", GREEN, "faster than reading speed, feels smooth"
    if tps >= 7:
        return "Usable", YELLOW, "around reading speed, a bit of waiting"
    return "Slow", RED, "noticeably slower than reading, long waits"


def start_verdict(ms: float) -> str:
    if ms < 500:
        return "feels instant"
    if ms < 2000:
        return "short pause before it starts"
    return "long pause before any text appears"


def generate(model: str, prompt: str, nonce: str = "") -> dict:
    """One non-streaming generation. `nonce` goes at the start of the prompt
    so Ollama can't reuse the cached prompt from the previous run."""
    r = requests.post(
        f"{OLLAMA}/api/generate",
        json={
            "model": model,
            "prompt": f"[run {nonce}] {prompt}" if nonce else prompt,
            "stream": False,
            "think": False,
            # Override Modelfile defaults so every model runs with the same settings.
            "options": {
                "temperature": 0,
                "seed": 42,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "repeat_penalty": 1.0,
                "num_ctx": 8192,
                "num_predict": 1024,
            },
        },
        timeout=600,
    )
    if not r.ok:
        raise RuntimeError(f"Ollama error for {model}: {r.status_code} {r.text}")
    return r.json()


def installed_models() -> list[dict]:
    r = requests.get(f"{OLLAMA}/api/tags", timeout=10)
    r.raise_for_status()
    return r.json().get("models", [])


def model_info(model: str) -> dict:
    r = requests.post(f"{OLLAMA}/api/show", json={"model": model}, timeout=30)
    return r.json() if r.ok else {}


def can_generate(model: str) -> bool:
    caps = model_info(model).get("capabilities")
    return caps is None or "completion" in caps  # skips embedding models


def memory_split(model: str) -> tuple[int | None, float | None]:
    """(bytes in VRAM, % of the model in VRAM). Under 100% means it spilled into system RAM."""
    r = requests.get(f"{OLLAMA}/api/ps", timeout=10)
    for m in r.json().get("models", []):
        if m["name"] == model and m.get("size"):
            return m.get("size_vram"), round(100 * m["size_vram"] / m["size"], 1)
    return None, None


def ollama_version() -> str | None:
    r = requests.get(f"{OLLAMA}/api/version", timeout=10)
    return r.json().get("version") if r.ok else None


def detect_gpu() -> str | None:
    """GPU name from Windows, or None."""
    if platform.system() != "Windows":
        return None
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    names = [n.strip() for n in out.splitlines() if n.strip()]
    real = [n for n in names if not any(x in n for x in ("Basic", "Virtual", "Parsec", "Remote"))]
    for n in real:
        if any(x in n for x in ("RX", "RTX", "GTX", "Arc", "Radeon Pro")):
            return n
    return real[0] if real else None


def pick_models() -> list[str]:
    models = installed_models()
    usable = [m for m in models if can_generate(m["name"])]
    if not usable:
        sys.exit("No text models installed. Try:  ollama pull llama3.2:3b")

    print(f"\n{BOLD}Models on this computer:{RESET}")
    for i, m in enumerate(usable, 1):
        print(f"  {CYAN}{i}{RESET}. {m['name']:<42} {DIM}{m['size'] / 1e9:.1f} GB{RESET}")
    print(f"{DIM}  Tip: models bigger than your GPU's memory will run much slower.{RESET}")

    while True:
        raw = input(f"\nWhich to test? Numbers separated by commas, or {BOLD}a{RESET} for all: ").strip().lower()
        if raw == "a":
            return [m["name"] for m in usable]
        try:
            picks = [int(x) for x in raw.replace(" ", "").split(",") if x]
            if picks and all(1 <= p <= len(usable) for p in picks):
                return [usable[p - 1]["name"] for p in picks]
        except ValueError:
            pass
        print(f"{RED}Didn't catch that. Example: 1,3{RESET}")


def measure(model: str, progress=lambda done, total: None) -> dict:
    """Run the benchmark without printing. Used by the CLI and the web server."""
    generate(model, "Hi")  # warm-up so load time isn't measured

    tok_per_sec, ttft_ms = [], []
    n_runs = len(PROMPTS) * RUNS_PER_PROMPT
    for p_i, prompt in enumerate(PROMPTS):
        for i in range(RUNS_PER_PROMPT):
            progress(p_i * RUNS_PER_PROMPT + i, n_runs)
            s = generate(model, prompt, nonce=f"{time.time_ns()}-{i}")
            # durations are in nanoseconds
            tok_per_sec.append(s["eval_count"] / (s["eval_duration"] / 1e9))
            ttft_ms.append(s["prompt_eval_duration"] / 1e6)

    details = model_info(model).get("details", {})
    vram, gpu_pct = memory_split(model)
    result = {
        "model": model,
        "quantization": details.get("quantization_level"),
        "parameter_size": details.get("parameter_size"),
        "avg_tokens_per_sec": round(sum(tok_per_sec) / len(tok_per_sec), 2),
        "avg_ttft_ms": round(sum(ttft_ms) / len(ttft_ms), 1),
        "vram_bytes": vram,
        "gpu_offload_pct": gpu_pct,
        "runs": len(tok_per_sec),
    }
    progress(n_runs, n_runs)
    return result


def bench_model(model: str, index: int, total: int) -> dict:
    print(f"\n{BOLD}[{index}/{total}] {model}{RESET}")

    def show(done, n):
        msg = "Loading into memory..." if done == 0 else f"[{'#' * done}{'.' * (n - done)}] run {done}/{n}"
        print(f"  {msg:<40}", end="\r")

    result = measure(model, show)
    print(" " * 60, end="\r")
    explain(result)
    return result


def explain(r: dict) -> None:
    tps, ttft = r["avg_tokens_per_sec"], r["avg_ttft_ms"]
    label, color, meaning = verdict(tps)
    tags = " ".join(x for x in (r["parameter_size"], r["quantization"]) if x)
    print(f"  {DIM}{tags}{RESET}")
    print(f"  Overall:        {color}{BOLD}{label}{RESET}")
    print(f"  Writing speed:  {BOLD}{tps:.1f}{RESET} tokens/sec "
          f"{DIM}(~{tps * 0.75:.0f} words/sec, {meaning}){RESET}")
    print(f"  Response start: {BOLD}{ttft / 1000:.2f}s{RESET} {DIM}({start_verdict(ttft)}){RESET}")
    pct = r["gpu_offload_pct"]
    if pct is not None:
        gb = (r["vram_bytes"] or 0) / 1e9
        if pct >= 99:
            print(f"  GPU memory:     {BOLD}{gb:.1f} GB{RESET} {GREEN}fits entirely on your GPU{RESET}")
        else:
            print(f"  GPU memory:     {BOLD}{gb:.1f} GB{RESET} {YELLOW}only {pct:.0f}% fits on the GPU; "
                  f"the rest uses slower system RAM{RESET}")


def make_submission(gpu: str, model_results: list[dict]) -> dict:
    return {
        "timestamp": int(time.time()),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "gpu": gpu,
        "ollama_version": ollama_version(),
        "results": model_results,
    }


def summary(results: list[dict]) -> None:
    if len(results) < 2:
        return
    ranked = sorted(results, key=lambda r: r["avg_tokens_per_sec"], reverse=True)
    print(f"\n{BOLD}Comparison (fastest first){RESET}")
    top = ranked[0]["avg_tokens_per_sec"]
    for r in ranked:
        tps = r["avg_tokens_per_sec"]
        label, color, _ = verdict(tps)
        bar = "█" * max(1, round(25 * tps / top))
        print(f"  {r['model']:<38} {color}{bar}{RESET} {tps:.1f} tok/s  {DIM}{label}{RESET}")


def main():
    p = argparse.ArgumentParser(description="Benchmark local LLMs via Ollama.")
    p.add_argument("models", nargs="*", help="model names (leave empty to pick from a list)")
    p.add_argument("--submit", metavar="URL", help=f"results server (default tries {DEFAULT_SERVER})")
    p.add_argument("--no-submit", action="store_true", help="only save results.json")
    args = p.parse_args()
    interactive = not args.models

    print(f"\n{BOLD}{CYAN}LLM Bench{RESET} - how well does your PC run local AI models?")
    print(f"{DIM}Each model answers 3 test prompts, 3 times each, and we measure the speed.{RESET}")

    try:
        have = {m["name"] for m in installed_models()}
    except requests.ConnectionError:
        sys.exit(f"{RED}Can't reach Ollama.{RESET} Open the Ollama app (or run `ollama serve`) and try again.")

    if interactive:
        models = pick_models()
    else:
        models = [m if ":" in m else f"{m}:latest" for m in args.models]
        missing = [m for m in models if m not in have]
        if missing:
            sys.exit(f"Not installed: {', '.join(missing)}\nPull with: ollama pull {missing[0]}")

    gpu = detect_gpu()
    if gpu:
        typed = input(f"\nDetected GPU: {BOLD}{gpu}{RESET}. Press Enter if correct, or type the right name: ").strip()
        gpu = typed or gpu
    else:
        gpu = input("\nGPU name (e.g. RX 6750 XT): ").strip() or "Unknown GPU"

    results = make_submission(gpu, [bench_model(m, i, len(models)) for i, m in enumerate(models, 1)])
    summary(results["results"])

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{DIM}Saved raw numbers to results.json{RESET}")

    if not args.no_submit:
        server = (args.submit or DEFAULT_SERVER).rstrip("/")
        try:
            r = requests.post(f"{server}/api/submissions", json=results, timeout=30)
            if r.ok:
                print(f"{GREEN}Results uploaded.{RESET} Opening {server} ...")
                webbrowser.open(server)
            else:
                print(f"{YELLOW}Upload rejected:{RESET} {r.status_code} {r.text}")
        except requests.ConnectionError:
            print(f"{DIM}Results website isn't running, so results weren't uploaded.{RESET}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"\n{RED}Error:{RESET} {e}")
    finally:
        # keep the window open when launched by double-click
        if len(sys.argv) == 1:
            input("\nPress Enter to close...")
