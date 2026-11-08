"""Launcher used by LLM Bench.bat: checks setup, starts the server, opens the browser."""

import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser

HOST, PORT = "127.0.0.1", 8000
URL = f"http://{HOST}:{PORT}"


def port_in_use() -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((HOST, PORT)) == 0


def running_version() -> str | None:
    """Code version of whatever is on our port, "old" for a pre-versioning
    LLM Bench, or "other" for an unrelated program."""
    import requests
    try:
        r = requests.get(f"{URL}/api/version", timeout=3)
        if r.ok and r.json().get("app") == "llm-bench":
            return r.json()["version"]
        if requests.get(f"{URL}/api/gpus", timeout=3).ok:
            return "old"
    except (requests.RequestException, ValueError):
        pass
    return "other"


def stop_stale_server() -> bool:
    """Kill the outdated LLM Bench holding our port (Windows only)."""
    if os.name != "nt":
        return False
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    pids = {line.split()[-1] for line in out.splitlines()
            if f":{PORT} " in line and "LISTENING" in line}
    for pid in pids:
        subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
    for _ in range(20):
        if not port_in_use():
            return True
        time.sleep(0.25)
    return False


def ensure_ollama() -> None:
    """Ollama is optional: predictions work without it."""
    if shutil.which("ollama"):
        return
    print("Ollama isn't installed. You can still browse predictions,")
    print("but downloading and benchmarking models needs it.")
    if shutil.which("winget") and input("Install Ollama now with winget? [y/N] ").strip().lower() == "y":
        subprocess.run(["winget", "install", "-e", "--id", "Ollama.Ollama"])
        print("If Ollama doesn't show up yet, close this window and double-click LLM Bench again.")
    else:
        print("Get it later from https://ollama.com/download\n")


def open_when_ready() -> None:
    """Open the browser once the server is accepting connections."""
    for _ in range(60):
        if port_in_use():
            webbrowser.open(URL)
            print(f"\nLLM Bench is running at {URL}")
            print("Keep this window open while you use the site. Close it to stop.\n")
            return
        time.sleep(0.25)
    print("The website didn't start. Scroll up for the error.")


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if port_in_use():
        from version import code_version
        running = running_version()
        if running == code_version():
            print("LLM Bench is already running. Opening it...")
            webbrowser.open(URL)
            return
        if running == "other":
            print(f"Another program is using port {PORT}. Close it and try again.")
            input("Press Enter to close...")
            return
        # an older version is running; restart it so the page and API match
        print("An older LLM Bench is still running. Restarting it with the latest version...")
        if not stop_stale_server():
            print("Couldn't stop it. Close the old LLM Bench window (or end python.exe in Task Manager) and try again.")
            input("Press Enter to close...")
            return

    ensure_ollama()

    os.environ["LLM_BENCH_LOCAL"] = "1"

    threading.Thread(target=open_when_ready, daemon=True).start()
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nError: {e}")
        input("Press Enter to close...")
