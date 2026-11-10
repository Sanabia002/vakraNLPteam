"""Start (or check) everything a Subtask 4 run needs on the GPU machine, from the repository root:

    ollama        the agent's model, port 11434        (and the self-hosted judge, if used)
    proxy         thinking off, port 8080              (LITELLM_BASE_URL points here)
    rest          the domain APIs, port 8000           (environment/m3/rest)
    retriever     the document search, port 8001       (environment/retrievers)

    python -m subtask4.deploy.services start     start what is not running, then wait until all answer
    python -m subtask4.deploy.services check     report which answer

ENV is what the runner, evaluator and native MCP server expect; run and eval fill in any that are unset.
Logs go to output/subtask4/services/.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request

from ..data import REPO

ENV = {
    "PYTHONPATH": str(REPO),
    "CAPABILITY_ID": "4",
    "MCP_DB_ROOT": str(REPO / "data" / "databases"),
    "PRELOAD_COLLECTIONS": "false",
    "LITELLM_BASE_URL": "http://localhost:8080/v1",
    "LITELLM_API_KEY": "local",
}
SERVER_PYTHON = REPO / ".venv-env" / "bin" / "python"  # the servers' own environment (VAKRA's setup)
LOGS = REPO / "output" / "subtask4" / "services"
# The servers read their data from fixed folders next to their code; both are links into data/.
DATA_LINKS = {REPO / "environment" / "m3" / "rest" / "db": REPO / "data" / "databases",
              REPO / "environment" / "retrievers" / "chroma_data": REPO / "data" / "indexed_documents"}

# name: (health URL, command, working directory, extra environment)
SERVICES = {
    "ollama": ("http://localhost:11434/api/tags", ["ollama", "serve"], REPO,
               {"OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_CONTEXT_LENGTH": "32768", "OLLAMA_NUM_PARALLEL": "4",
                "OLLAMA_KEEP_ALIVE": "-1"}),
    "proxy": ("http://localhost:8080/api/tags", [sys.executable, "-m", "subtask4.deploy.nothink_proxy"], REPO, {}),
    "rest": ("http://localhost:8000/openapi.json",
             [str(SERVER_PYTHON), "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"],
             REPO / "environment" / "m3" / "rest", {}),
    "retriever": ("http://localhost:8001/health",
                  [str(SERVER_PYTHON), "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", "8001"],
                  REPO / "environment" / "retrievers", {"PRELOAD_COLLECTIONS": "false"}),
}


def apply_env() -> None:
    """Set any of ENV that is not already set (the runner, evaluator and MCP server read them)."""
    for key, value in ENV.items():
        os.environ.setdefault(key, value)


def answers(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3):
            return True
    except Exception:
        return False


def check() -> dict[str, bool]:
    return {name: answers(url) for name, (url, *_) in SERVICES.items()}


def start(timeout_s: int = 240) -> dict[str, bool]:
    LOGS.mkdir(parents=True, exist_ok=True)
    for link, target in DATA_LINKS.items():
        if not link.exists():
            link.symlink_to(target)
    for name, (url, command, cwd, extra) in SERVICES.items():
        if not answers(url):
            with open(LOGS / f"{name}.log", "ab") as log:
                subprocess.Popen(command, cwd=cwd, env={**os.environ, **ENV, **extra}, stdout=log,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    deadline, status = time.monotonic() + timeout_s, check()
    while not all(status.values()) and time.monotonic() < deadline:
        time.sleep(2)
        status = check()
    return status


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "check"
    status = start() if action == "start" else check()
    print("  ".join(f"{name}: {'up' if up else 'DOWN'}" for name, up in status.items()))
    sys.exit(0 if all(status.values()) else 1)
