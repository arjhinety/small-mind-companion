"""Pair each PMB ``outdated_fact`` probe with the update turn it was generated from (ERRATA E43).

``scripts/build_pmb.py`` gives every update turn in a persona the same id (``pXXX_s5_t099``) and
picks the corrected fact with replacement, so a probe's gold turn ids cannot say which update it
means. The generator appends one update turn and one ``outdated_fact`` probe per loop iteration, so
a persona's k-th ``outdated_fact`` probe (in ``probes.jsonl`` order) belongs to its k-th turn with
that shared id (in session order). This script records that pairing and classifies each probe by
the sequence of corrections of its fact (the paired probes' gold answers, in order):

* ``single_correction``: the fact is corrected once.
* ``repeat_same_value``: corrected more than once, every time to the same value.
* ``latest_of_different_values``: corrected more than once to different values; this probe's is
  the last, so its gold answer is still true at the end.
* ``superseded``: a later correction gives a different value, so this probe's gold answer is no
  longer true at the end of the conversation.

``ambiguous_question`` is a separate flag: another ``outdated_fact`` probe of the persona asks the
same question with a different gold answer. ``noop_correction`` flags corrections that repeat the
fact's original value.

The paired turn's fact id must equal the stale gold turn's fact id, and the gold answer must occur
in the paired turn's text; the script fails otherwise. Reads only the frozen benchmark files.

Usage::

    uv run python scripts/pmb_update_pairing.py           # (re)write the output JSON
    uv run python scripts/pmb_update_pairing.py --check   # exit 1 if the output is stale
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "benchmarks" / "pmb_v0_full"
OUTPUT = ROOT / "results" / "audit" / "pmb_updates" / "pmb_update_pairing.json"
UPDATE_CATEGORY = "outdated_fact"


def _shared_update_id(persona_id: str, sessions: list[dict[str, Any]]) -> str:
    return f"{persona_id}_s{len(sessions) - 1}_t099"


def _training_personas() -> dict[str, Any]:
    """The same generator built the training personas: count the same two defects there."""
    files = sorted((ROOT / "data" / "benchmarks" / "sft_personas_v1" / "personas").glob("*.json"))
    shared = twice = 0
    doubled: set[int] = set()
    for f in files:
        persona = json.loads(f.read_text(encoding="utf-8"))
        attributes = Counter(
            re.sub(r"_\d+$", "", fact["predicate"]) for fact in persona["persona"]["fact_sheet"]
        )
        doubled.add(sum(1 for n in attributes.values() if n > 1))
        uid = _shared_update_id(persona["persona"]["persona_id"], persona["sessions"])
        updates = [t for s in persona["sessions"] for t in s["turns"] if t["turn_id"] == uid]
        shared += len(updates) > 1
        facts = Counter(t["revealed_fact_ids"][0] for t in updates if t["revealed_fact_ids"])
        twice += sum(1 for n in facts.values() if n > 1)
    return {
        "personas": len(files),
        "personas_sharing_one_update_id": shared,
        "facts_corrected_more_than_once": twice,
        "attributes_stated_twice_per_persona": sorted(doubled),
    }


def build() -> dict[str, Any]:
    probes = [
        json.loads(line)
        for line in (BENCH / "probes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows: list[dict[str, Any]] = []
    for persona_file in sorted((BENCH / "personas").glob("p*.json")):
        persona = json.loads(persona_file.read_text(encoding="utf-8"))
        pid = persona["persona"]["persona_id"]
        sessions = persona["sessions"]
        uid = _shared_update_id(pid, sessions)
        turns = [t for s in sessions for t in s["turns"]]
        updates = [t for t in turns if t["turn_id"] == uid]
        by_id = {t["turn_id"]: t for t in turns if t["turn_id"] != uid}
        upd_probes = [
            p for p in probes if p["persona_id"] == pid and p["category"] == UPDATE_CATEGORY
        ]
        if len(upd_probes) != len(updates):
            raise SystemExit(f"{pid}: {len(upd_probes)} probes but {len(updates)} update turns")
        sheet = {f["fact_id"]: f for f in persona["persona"]["fact_sheet"]}
        facts = [t["revealed_fact_ids"][0] for t in updates]
        golds = [p["gold_answer"].lower() for p in upd_probes]
        answers_by_question: dict[str, set[str]] = {}
        for p in upd_probes:
            answers_by_question.setdefault(p["question"].strip().lower(), set()).add(
                p["gold_answer"].lower()
            )
        for k, (probe, turn) in enumerate(zip(upd_probes, updates, strict=True)):
            stale_id = next(g for g in probe["gold_supporting_memory_ids"] if g != uid)
            fact = turn["revealed_fact_ids"][0]
            if fact not in by_id[stale_id]["revealed_fact_ids"]:
                raise SystemExit(
                    f"{probe['probe_id']}: paired turn reveals {fact}, not the stale one"
                )
            gold = probe["gold_answer"].lower()
            if gold not in turn["text"].lower():
                raise SystemExit(f"{probe['probe_id']}: gold answer not in the paired update turn")
            sequence = [golds[j] for j, f in enumerate(facts) if f == fact]
            if len(sequence) == 1:
                cls = "single_correction"
            elif gold != sequence[-1]:
                cls = "superseded"
            elif len(set(sequence)) == 1:
                cls = "repeat_same_value"
            else:
                cls = "latest_of_different_values"
            rows.append(
                {
                    "probe_id": probe["probe_id"],
                    "persona_id": pid,
                    "update_turn_id": uid,
                    "update_index": k,
                    "update_text": turn["text"],
                    "fact_id": fact,
                    "stale_turn_id": stale_id,
                    "class": cls,
                    "ambiguous_question": len(
                        answers_by_question[probe["question"].strip().lower()]
                    )
                    > 1,
                    "noop_correction": gold == str(sheet[fact]["object"]).lower(),
                }
            )
    counts = Counter(r["class"] for r in rows)
    return {
        "artifact_kind": "pmb-update-pairing",
        "generated_by": "scripts/pmb_update_pairing.py",
        "note": "Pairing of PMB outdated_fact probes to their update turns (ERRATA E43).",
        "update_probes": len(rows),
        "class_counts": {c: counts[c] for c in sorted(counts)},
        "ambiguous_question": sum(r["ambiguous_question"] for r in rows),
        "noop_correction": sum(r["noop_correction"] for r in rows),
        "personas_sharing_one_update_id": len({r["persona_id"] for r in rows}),
        "training_personas": _training_personas(),
        "pairs": rows,
    }


def serialise(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the output is stale")
    args = parser.parse_args(argv)
    text = serialise(build())
    rel = OUTPUT.relative_to(ROOT).as_posix()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{rel} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{rel} is current")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
