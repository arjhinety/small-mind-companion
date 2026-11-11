"""Run a command with full logging: machine, environment, output, exit status, and an upload.

Every experiment, conversion, upload or CLI call worth keeping goes through this wrapper::

    uv run python scripts/run_logged.py --machine gpu-box --purpose "SFT v2 seed 1" -- \
        uv run python -m onebee.training.sft configs/training/sft_v2.yaml

What it records:

* **Committed** (small, reviewable, no secrets, no machine identifiers):
  ``runs/manifests/<run_id>.json`` -- command, purpose, machine, git commit and dirty state,
  start/end/duration, exit code, the log's SHA-256/size/line counts, and where the full log was
  uploaded (repo, path, revision); ``runs/environments/<env_id>.json`` -- Python, OS, GPU model,
  driver and CUDA versions, every installed package version, the uv.lock hash, allow-listed
  environment variables, and capability probes (sqlite-vec loads or not). Environments are
  content-addressed, so identical environments share one file.
* **Uploaded** to the private Hugging Face dataset repo (``RUN_LOG_REPO``) under
  ``small-mind-companion/runs/<run_id>/``: ``log.txt.gz`` (full output, both streams,
  timestamped, progress-bar redraws collapsed; the manifest's SHA-256 is of the uncompressed text)
  and ``environment_private.txt`` (full ``nvidia-smi -q``,
  hostname). Written locally first under ``runs/local/<run_id>/`` (gitignored), so a crash or a
  failed upload loses nothing; ``sync`` uploads anything still pending.

Secrets: the value of every environment variable whose name looks like a credential, and anything
shaped like an HF or OpenAI token, is replaced with ``[REDACTED:<NAME>]`` in the log, the command
and every file written, before anything leaves the machine.

The machine must be listed in ``registry/machines.yaml`` with ``runs_allowed: true`` and must match
its declared platform (and have ``nvidia-smi`` if it ``requires_gpu``); otherwise the wrapper
refuses before running anything.

Subcommands::

    run_logged.py --machine ID [--purpose TEXT] [--no-upload] -- COMMAND...
    run_logged.py sync      # upload every run whose manifest still says upload pending
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import IO, Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
MACHINES = ROOT / "registry" / "machines.yaml"
RUNS = ROOT / "runs"
MANIFESTS = RUNS / "manifests"
ENVIRONMENTS = RUNS / "environments"
LOCAL = RUNS / "local"
RUN_LOG_REPO = "arjhinety/run-logs"
PROJECT_PREFIX = "small-mind-companion"
SCHEMA_VERSION = 1

ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "HF_HUB_DISABLE_XET",
    "HF_HUB_ENABLE_HF_TRANSFER",
    "JUDGE_MODEL",
    "MODAL_CLOUD_PROVIDER",
    "MODAL_ENVIRONMENT",
    "MODAL_IMAGE_ID",
    "MODAL_IS_REMOTE",
    "MODAL_REGION",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TOKENIZERS_PARALLELISM",
    "TORCH_CUDNN_V8_API_DISABLED",
)
SECRET_NAME = re.compile(r"TOKEN|KEY|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH", re.IGNORECASE)
SECRET_SHAPES = (
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "[REDACTED:hf-token]"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "[REDACTED:openai-key]"),
)
WARNING_LINE = re.compile(r"\bwarn(ing)?\b", re.IGNORECASE)
ERROR_LINE = re.compile(r"\b(error|exception|traceback)\b", re.IGNORECASE)


# --------------------------------------------------------------------------------------------
# redaction
# --------------------------------------------------------------------------------------------
class Redactor:
    """Replaces credential values (from the environment) and token-shaped strings."""

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        env = dict(os.environ if environ is None else environ)
        pairs = [
            (value, name)
            for name, value in env.items()
            if SECRET_NAME.search(name) and value and len(value) >= 8
        ]
        # Longest first, so a value containing another is replaced whole.
        self._values = sorted(pairs, key=lambda p: len(p[0]), reverse=True)

    def __call__(self, text: str) -> str:
        for value, name in self._values:
            if value in text:
                text = text.replace(value, f"[REDACTED:{name}]")
        for pattern, replacement in SECRET_SHAPES:
            text = pattern.sub(replacement, text)
        return text


def collapse_carriage_returns(line: str) -> str:
    """Keep only what a terminal would show: the text after the last carriage return."""
    stripped = line.rstrip("\r")
    return stripped.rsplit("\r", 1)[-1] if "\r" in stripped else stripped


# --------------------------------------------------------------------------------------------
# machines
# --------------------------------------------------------------------------------------------
class RefusedError(RuntimeError):
    """The wrapper will not run on this machine."""


def load_machines(path: Path = MACHINES) -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {m["id"]: m for m in document["machines"]}


def check_machine(
    machine_id: str,
    machines: dict[str, dict[str, Any]],
    platform_name: str = sys.platform,
    has_nvidia_smi: bool | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = dict(os.environ if environ is None else environ)
    if machine_id not in machines:
        raise RefusedError(
            f"machine {machine_id!r} is not in registry/machines.yaml; add it there first"
        )
    machine = machines[machine_id]
    if not machine.get("runs_allowed", False):
        reason = machine.get("not_allowed_reason", "runs_allowed is false")
        raise RefusedError(f"runs are not allowed on {machine_id!r}: {reason}")
    if not platform_name.startswith(str(machine["platform"])):
        raise RefusedError(
            f"{machine_id!r} is declared as platform {machine['platform']!r}, "
            f"but this is {platform_name!r}: wrong --machine?"
        )
    if machine.get("requires_gpu"):
        present = (
            shutil.which("nvidia-smi") is not None if has_nvidia_smi is None else has_nvidia_smi
        )
        if not present:
            raise RefusedError(f"{machine_id!r} requires a GPU but nvidia-smi is not available")
    # A run must be logged where it executes: e.g. `--machine modal` only inside a Modal
    # container, never on the machine that launched it (that would record the launcher's
    # hardware under the Modal label).
    for name, value in (machine.get("requires_env") or {}).items():
        if env.get(name) != str(value):
            raise RefusedError(
                f"{machine_id!r} requires {name}={value} (this process is not running on that "
                "machine; start the wrapper inside it)"
            )
    for name in machine.get("forbids_env") or []:
        if name in env:
            raise RefusedError(
                f"{machine_id!r} must not see {name}: this process looks like it runs elsewhere"
            )
    return machine


# --------------------------------------------------------------------------------------------
# environment capture
# --------------------------------------------------------------------------------------------
def _run_quiet(argv: list[str], timeout: float = 60.0) -> str | None:
    if shutil.which(argv[0]) is None:
        return None
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout or "") + (out.stderr or "")


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(*args: str) -> str | None:
    out = _run_quiet(["git", "-C", str(ROOT), *args])
    return None if out is None else out.strip()


def git_state() -> dict[str, Any]:
    status = _git("status", "--porcelain") or ""
    diff = _git("diff", "HEAD") or ""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status.strip()),
        "changed_files": len([line for line in status.splitlines() if line.strip()]),
        "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None,
    }


def probe_sqlite_vec() -> dict[str, Any]:
    """Whether sqlite-vec loads, using the memory store's own loader (so the probe and the
    store cannot disagree about how to load it)."""
    try:
        import sqlite3

        from onebee.memory.store import load_sqlite_vec

        conn = sqlite3.connect(":memory:")
        try:
            error = load_sqlite_vec(conn)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - the probe reports, it does not fail the run
        error = f"{type(exc).__name__}: {exc}"
    return {"loads": error is None, "error": error}


def gpu_summary() -> list[dict[str, str]] | None:
    fields = "name,driver_version,memory.total,compute_cap"
    out = _run_quiet(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"])
    if out is None:
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            gpus.append(dict(zip(("name", "driver", "memory_total", "compute_cap"), parts)))
    return gpus


def cuda_runtime_version() -> str | None:
    out = _run_quiet(["nvidia-smi"])
    if out is None:
        return None
    match = re.search(r"CUDA Version:\s*([\d.]+)", out)
    return match.group(1) if match else None


def nvcc_version() -> str | None:
    out = _run_quiet(["nvcc", "--version"])
    if out is None:
        return None
    match = re.search(r"release ([\d.]+)", out)
    return match.group(1) if match else None


def installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name.lower()] = dist.version
    return dict(sorted(packages.items()))


def capture_environment(machine_id: str, redact: Redactor) -> tuple[dict[str, Any], str]:
    """(committed environment record, private detail text)."""
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "machine": machine_id,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "gpus": gpu_summary(),
        "cuda_driver_api": cuda_runtime_version(),
        "nvcc": nvcc_version(),
        "uv_lock_sha256": _sha256_file(ROOT / "uv.lock"),
        "env": {name: os.environ[name] for name in ENV_ALLOWLIST if name in os.environ},
        "capabilities": {"sqlite_vec": probe_sqlite_vec()},
        "packages": installed_packages(),
    }
    record = json.loads(redact(json.dumps(record)))
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    record["env_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    private = "\n".join(
        [
            f"hostname: {socket.gethostname()}",
            f"platform: {platform.platform()}",
            "",
            "== nvidia-smi -q ==",
            _run_quiet(["nvidia-smi", "-q"]) or "(nvidia-smi not available)",
            "",
            "== nvcc --version ==",
            _run_quiet(["nvcc", "--version"]) or "(nvcc not available)",
        ]
    )
    return record, redact(private)


# --------------------------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------------------------
def new_run_id(now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now(_dt.timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def _pump(
    stream: IO[bytes],
    tag: str,
    sink: IO[str],
    echo: IO[str],
    lock: threading.Lock,
    start: float,
    redact: Redactor,
    counts: dict[str, int],
) -> None:
    for raw in iter(stream.readline, b""):
        text = raw.decode("utf-8", errors="replace")
        echo.write(text)
        echo.flush()
        line = redact(collapse_carriage_returns(text.rstrip("\n")))
        with lock:
            sink.write(f"[{time.monotonic() - start:9.2f}s {tag}] {line}\n")
            counts["lines"] += 1
            if WARNING_LINE.search(line):
                counts["warning_lines"] += 1
            if ERROR_LINE.search(line):
                counts["error_lines"] += 1
    stream.close()


def run_command(
    command: list[str], log_path: Path, redact: Redactor
) -> tuple[int, dict[str, int], float]:
    """Run, tee both streams to the console and a timestamped log. (exit code, counts, seconds)"""
    counts = {"lines": 0, "warning_lines": 0, "error_lines": 0}
    lock = threading.Lock()
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8", newline="\n") as sink:
        sink.write(f"# command: {redact(' '.join(command))}\n")
        try:
            proc = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except OSError as exc:
            sink.write(f"# could not start: {redact(str(exc))}\n")
            return 127, counts, time.monotonic() - start
        assert proc.stdout is not None and proc.stderr is not None
        threads = [
            threading.Thread(
                target=_pump,
                args=(proc.stdout, "out", sink, sys.stdout, lock, start, redact, counts),
            ),
            threading.Thread(
                target=_pump,
                args=(proc.stderr, "err", sink, sys.stderr, lock, start, redact, counts),
            ),
        ]
        for thread in threads:
            thread.start()
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            code = proc.wait()
            sink.write("# interrupted by KeyboardInterrupt\n")
        for thread in threads:
            thread.join()
        sink.write(f"# exit code: {code}\n")
    return code, counts, time.monotonic() - start


# --------------------------------------------------------------------------------------------
# upload
# --------------------------------------------------------------------------------------------
def upload_run(run_id: str, files: dict[str, Path], repo_id: str = RUN_LOG_REPO) -> dict[str, Any]:
    """Upload a run's local files in one commit to the private dataset repo."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    prefix = f"{PROJECT_PREFIX}/runs/{run_id}"
    info = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=[
            CommitOperationAdd(path_in_repo=f"{prefix}/{name}", path_or_fileobj=str(path))
            for name, path in files.items()
        ],
        commit_message=f"{PROJECT_PREFIX} run {run_id}",
    )
    revision = getattr(info, "oid", None) or str(info).rstrip("/").rsplit("/", 1)[-1]
    return {"status": "uploaded", "repo": repo_id, "path": prefix, "revision": revision}


def _local_files(run_id: str) -> dict[str, Path]:
    folder = LOCAL / run_id
    return {
        name: folder / name
        for name in ("log.txt.gz", "environment_private.txt")
        if (folder / name).is_file()
    }


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("run_logged: no command given after --", file=sys.stderr)
        return 2
    try:
        check_machine(args.machine, load_machines())
    except RefusedError as exc:
        print(f"run_logged: refused: {exc}", file=sys.stderr)
        return 3
    redact = Redactor()
    run_id = new_run_id()
    folder = LOCAL / run_id
    folder.mkdir(parents=True, exist_ok=True)

    environment, private = capture_environment(args.machine, redact)
    env_path = ENVIRONMENTS / f"{environment['env_id']}.json"
    if not env_path.is_file():
        write_json(env_path, environment)
    (folder / "environment_private.txt").write_text(private, encoding="utf-8")

    started = _dt.datetime.now(_dt.timezone.utc)
    git_before = git_state()
    log_txt = folder / "log.txt"
    code, counts, seconds = run_command(command, log_txt, redact)
    ended = _dt.datetime.now(_dt.timezone.utc)

    log_bytes = log_txt.read_bytes()
    with gzip.open(folder / "log.txt.gz", "wb", compresslevel=9) as handle:
        handle.write(log_bytes)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "purpose": redact(args.purpose or ""),
        "machine": args.machine,
        "command": [redact(part) for part in command],
        "git": git_before,
        "environment": environment["env_id"],
        "started_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
        "duration_s": round(seconds, 2),
        "exit_code": code,
        "log": {
            "sha256": hashlib.sha256(log_bytes).hexdigest(),
            "bytes": len(log_bytes),
            **counts,
        },
        "upload": {"status": "pending"},
        "upstream_refs": [],
    }
    manifest_path = MANIFESTS / f"{run_id}.json"
    write_json(manifest_path, manifest)
    print(f"run_logged: {run_id} exit={code} manifest={manifest_path.relative_to(ROOT)}")
    if not args.no_upload:
        _upload_and_record(manifest_path)
    return code


def _upload_and_record(manifest_path: Path) -> bool:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = _local_files(manifest["run_id"])
    if "log.txt.gz" not in files:
        print(f"run_logged: {manifest['run_id']}: no local log to upload", file=sys.stderr)
        return False
    try:
        manifest["upload"] = upload_run(manifest["run_id"], files)
    except Exception as exc:  # noqa: BLE001 - a failed upload stays pending for `sync`
        print(f"run_logged: upload failed, left pending for `sync`: {exc}", file=sys.stderr)
        return False
    write_json(manifest_path, manifest)
    print(
        f"run_logged: uploaded to {manifest['upload']['repo']} @ {manifest['upload']['revision']}"
    )
    return True


def cmd_sync(_args: argparse.Namespace) -> int:
    pending = [
        path
        for path in sorted(MANIFESTS.glob("*.json"))
        if json.loads(path.read_text(encoding="utf-8"))["upload"].get("status") == "pending"
    ]
    failures = sum(0 if _upload_and_record(path) else 1 for path in pending)
    print(f"run_logged: {len(pending) - failures} of {len(pending)} pending runs uploaded")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["sync"]:
        return cmd_sync(argparse.Namespace())
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--machine", required=True, help="an id from registry/machines.yaml")
    parser.add_argument("--purpose", default="", help="why this run exists, in one line")
    parser.add_argument(
        "--no-upload", action="store_true", help="keep the log local; upload later with `sync`"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND...")
    return cmd_run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
