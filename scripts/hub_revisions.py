"""Snapshot the Hugging Face Hub facts behind the Study 002 revision pins (STUDIES item 13).

``configs/training/study002/*.yaml`` load every model from the Hub at a commit SHA. This script
records, for each pinned repository, what that commit is and why it was chosen:

* **Published checkpoints** (``arjhinety/onebee-gf-*``): the pinned commit, whether it is the one
  that added ``model.safetensors`` (its parent has none), the sha256 of that file at the pin and at
  the current head (so "unchanged since upload" is checkable), and the ``tokenizer.json`` blob id.
* **Gemma base models**: the pinned commit's date, the next commit on ``main`` (the date the pin
  stopped being ``main``), and the files that next commit changed.

The output is a dated snapshot of a remote service, so ``scripts/validate.py`` cannot recompute it
offline; it checks the configs against this file instead (claims ``study002-pin-*``). Rerun it to
refresh the snapshot. The sha256 values are also what to compare against
``sha256sum outputs/*/merged/model.safetensors`` on the GPU box, which is the only way to show a Hub
copy is byte-identical to the local checkpoint Study 001 trained from.

Standard library only.

Usage::

    uv run python scripts/hub_revisions.py          # (re)write the snapshot
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "audit" / "hub_revisions" / "hub_revisions.json"
API = "https://huggingface.co/api/models"

CHECKPOINTS = {
    "arjhinety/onebee-gf-sft-v0": "b7d767dcd8bc2af882a0542182f40926dcae3693",
    "arjhinety/onebee-gf-sft-v1": "8d0e632b02df9cdf8a06dc8408db57ee3e995357",
    "arjhinety/onebee-gf-dpo-v1-scale": "b984814810a1248f74349e2dcb549d6e9ddb329f",
}
GEMMA = {
    "google/gemma-4-E4B-it": "ee0ef6023621cff504d758262d4e04895a5af4a2",
    "google/gemma-4-E2B-it": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
}


def _get(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _commits(repo: str) -> list[dict[str, Any]]:
    """Every commit on main, newest first."""
    return _get(f"{API}/{repo}/commits/main")


def _tree(repo: str, revision: str) -> dict[str, dict[str, Any]]:
    entries = _get(f"{API}/{repo}/tree/{urllib.parse.quote(revision, safe='')}?recursive=1")
    return {e["path"]: e for e in entries if e.get("type") == "file"}


def _commit_row(commit: dict[str, Any]) -> dict[str, str]:
    return {"id": commit["id"], "date": commit["date"], "title": commit["title"]}


def checkpoint_record(repo: str, pin: str) -> dict[str, Any]:
    commits = _commits(repo)
    index = next(i for i, c in enumerate(commits) if c["id"] == pin)
    pinned = commits[index]
    at_pin, at_head = _tree(repo, pin), _tree(repo, commits[0]["id"])
    # The pin is the commit that added the weights: its parent (if any) has no model.safetensors.
    parent = commits[index + 1] if index + 1 < len(commits) else None
    weights_at_parent = (
        "model.safetensors" in _tree(repo, parent["id"]) if parent is not None else False
    )
    weights_pin = at_pin["model.safetensors"]["lfs"]["oid"]
    weights_head = at_head["model.safetensors"]["lfs"]["oid"]
    return {
        "pinned_revision": pin,
        "pinned_commit": _commit_row(pinned),
        "parent_commit": _commit_row(parent) if parent is not None else None,
        "pinned_commit_added_the_weights": not weights_at_parent,
        "head_commit": _commit_row(commits[0]),
        "model_safetensors_sha256_at_pin": weights_pin,
        "model_safetensors_sha256_at_head": weights_head,
        "weights_unchanged_since_pin": weights_pin == weights_head,
        "tokenizer_json_blob_at_pin": at_pin["tokenizer.json"]["oid"],
    }


def gemma_record(repo: str, pin: str) -> dict[str, Any]:
    commits = _commits(repo)
    index = next(i for i, c in enumerate(commits) if c["id"] == pin)
    record: dict[str, Any] = {"pinned_revision": pin, "pinned_commit": _commit_row(commits[index])}
    at_pin = _tree(repo, pin)
    record["tokenizer_json_blob_at_pin"] = at_pin["tokenizer.json"]["oid"]
    if index == 0:
        record["next_commit"] = None
        record["files_changed_by_next_commit"] = []
        return record
    successor = commits[index - 1]
    at_next = _tree(repo, successor["id"])
    record["next_commit"] = _commit_row(successor)
    record["files_changed_by_next_commit"] = sorted(
        path
        for path in set(at_pin) | set(at_next)
        if at_pin.get(path, {}).get("oid") != at_next.get(path, {}).get("oid")
    )
    return record


def main() -> int:
    snapshot = {
        "artifact_kind": "hub_revision_snapshot",
        "generated_on": dt.date.today().isoformat(),
        "note": (
            "Snapshot of Hugging Face Hub state for the Study 002 revision pins. Regenerate with "
            "scripts/hub_revisions.py (needs network)."
        ),
        "checkpoints": {repo: checkpoint_record(repo, pin) for repo, pin in CHECKPOINTS.items()},
        "gemma": {repo: gemma_record(repo, pin) for repo, pin in GEMMA.items()},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
