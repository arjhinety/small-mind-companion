"""Audit PMB's answer key against the end of each conversation (ERRATA E43). POST-HOC, read-only.

Every PMB probe is asked after the whole conversation, so its correct answer is an attribute's value
at the end. Two generator defects make some gold answers out of date. ``scripts/build_pmb.py``
corrects some facts in the last session (one correction per ``outdated_fact`` probe,
``scripts/pmb_update_pairing.py``) but gives the other probes about those facts the original value;
and it gives some personas the same attribute twice (``favorite color`` and ``favorite color_37``),
so a later turn can state a different value for it. This script ties every answerable probe to the
fact its gold answer names and compares the gold answer with the attribute's final value: the value
of the last turn, in conversation order, that states that fact or a twin of it (same predicate
without the ``_N`` suffix), where an update turn states its paired probe's gold answer.

Tie rule: an ``outdated_fact`` probe is tied through the pairing; any other probe to the fact
revealed by one of its gold turns whose fact-sheet value equals the gold answer (case-insensitive).
A probe tied to no fact or to more than one is ``untied``. Key status:

* ``current``: the gold answer is the attribute's final value;
* ``stale``: a later turn states a different value (``stale_route``: ``paired_correction`` when that
  turn corrects the probe's own fact, ``twin_restated_later`` when it states a twin);
* ``ambiguous_question`` (``outdated_fact`` only): another update probe of the persona asks the same
  question about a different fact;
* ``ambiguous_final``: the attribute's last statement is one turn giving two different values;
* ``untied``: the gold answer cannot be tied to exactly one fact.

``noop_correction`` flags ``outdated_fact`` probes whose correction repeats the original value. For
each full-harness run the script counts, per category and status, the probes the judge marked
correct and the responses that ``contains_answer`` finds the gold answer or the final value in.

Usage::

    uv run python scripts/pmb_answer_key_audit.py           # (re)write the output JSON
    uv run python scripts/pmb_answer_key_audit.py --check   # exit 1 if the output is stale
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "benchmarks" / "pmb_v0_full"
REGISTRY = ROOT / "results" / "registry.jsonl"
OUTPUT = ROOT / "results" / "audit" / "pmb_answer_key" / "pmb_answer_key_audit.json"


def _load(name: str, rel: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _base(predicate: str) -> str:
    """The predicate without the generator's ``_N`` suffix for a repeated predicate."""
    return re.sub(r"_\d+$", "", predicate)


def _q(probe: dict[str, Any]) -> str:
    return str(probe["question"]).strip().lower()


def build() -> dict[str, Any]:
    rule = _load("onebee_rule_by_path", "src/onebee/evaluation/graders/rule.py")
    pairing_module = _load("pmb_update_pairing_by_path", "scripts/pmb_update_pairing.py")
    pairing = pairing_module.build()
    pairs = {row["probe_id"]: row for row in pairing["pairs"]}
    every_probe = _rows(BENCH / "probes.jsonl")
    probes = [p for p in every_probe if p["answerable"]]

    status: dict[str, dict[str, Any]] = {}
    sheet_shape: set[tuple[int, int, int]] = set()  # (facts, attributes, attributes stated twice)
    for persona_file in sorted((BENCH / "personas").glob("p*.json")):
        persona = json.loads(persona_file.read_text(encoding="utf-8"))
        pid = persona["persona"]["persona_id"]
        sheet = {f["fact_id"]: f for f in persona["persona"]["fact_sheet"]}
        turns = {t["turn_id"]: t for s in persona["sessions"] for t in s["turns"]}
        mine = [p for p in probes if p["persona_id"] == pid]
        base_predicates = Counter(_base(f["predicate"]) for f in sheet.values())
        sheet_shape.add(
            (len(sheet), len(base_predicates), sum(1 for n in base_predicates.values() if n > 1))
        )
        answers_by_question: dict[str, set[str]] = defaultdict(set)
        for p in mine:
            answers_by_question[_q(p)].add(p["gold_answer"].lower())
        corrections: dict[str, list[tuple[int, str]]] = defaultdict(list)
        update_values: dict[int, str] = {}
        for p in mine:
            if p["probe_id"] in pairs:
                row = pairs[p["probe_id"]]
                corrections[row["fact_id"]].append((row["update_index"], p["gold_answer"]))
                update_values[row["update_index"]] = p["gold_answer"]
        for values in corrections.values():
            values.sort()
        # The last turn stating each attribute (base predicate), in conversation order, as the
        # (fact id, value) pairs it states. An update turn states its paired probe's gold answer.
        last: dict[str, list[tuple[str, str]]] = {}
        uid = pairing_module._shared_update_id(pid, persona["sessions"])
        k = 0
        for s in persona["sessions"]:
            for t in s["turns"]:
                is_update = t["turn_id"] == uid
                stated: dict[str, list[tuple[str, str]]] = defaultdict(list)
                for f in t["revealed_fact_ids"]:
                    value = update_values[k] if is_update else str(sheet[f]["object"])
                    stated[_base(str(sheet[f]["predicate"]))].append((f, value))
                last.update(stated)
                k += is_update
        for p in mine:
            gold = p["gold_answer"]
            row = pairs.get(p["probe_id"])
            if row is not None:
                facts = {row["fact_id"]}
            else:
                facts = {
                    f
                    for g in p["gold_supporting_memory_ids"]
                    if g in turns
                    for f in turns[g]["revealed_fact_ids"]
                    if str(sheet[f]["object"]).lower() == gold.lower()
                }
            shared = len(answers_by_question[_q(p)]) > 1
            if len(facts) != 1:
                status[p["probe_id"]] = {
                    "status": "untied",
                    "final": None,
                    "shared_question_different_gold": shared,
                    "sound": False,
                }
                continue
            fact = facts.pop()
            seq = corrections.get(fact, [])
            predicate = str(sheet[fact]["predicate"])
            final_statements = last[_base(predicate)]
            finals = {value.lower() for _, value in final_statements}
            final: str | None = final_statements[0][1] if len(finals) == 1 else None
            route = None
            if row is not None and row["ambiguous_question"]:
                key = "ambiguous_question"
            elif final is None:
                key = "ambiguous_final"
            elif final.lower() != gold.lower():
                key = "stale"
                own = any(f == fact for f, _ in final_statements)
                route = "paired_correction" if own else "twin_restated_later"
            else:
                key = "current"
            status[p["probe_id"]] = {
                "status": key,
                "final": final,
                "corrected": bool(seq),
                "stale_route": route,
                "predicate": predicate,
                "twin_predicate": base_predicates[_base(predicate)] > 1,
                "shared_question_different_gold": shared,
                "noop_correction": row is not None
                and gold.lower() == str(sheet[fact]["object"]).lower(),
                "suffixed_predicate": _base(predicate) != predicate,
                "sound": key == "current" and row is None and not shared,
            }

    by_cat: dict[str, Counter[str]] = defaultdict(Counter)
    for p in probes:
        by_cat[p["category"]][status[p["probe_id"]]["status"]] += 1
    runs: dict[str, Any] = {}
    for meta in (r for r in _rows(REGISTRY) if r.get("run_kind") == "pmb_full_harness"):
        rows = {r["probe"]["probe_id"]: r for r in _rows(ROOT / meta["run_path"] / "raw.jsonl")}
        table: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
        for p in probes:
            s = status[p["probe_id"]]
            r = rows[p["probe_id"]]
            table["_all_categories"][s["status"]]["probes"] += 1
            table["_all_categories"][s["status"]]["judge_correct"] += bool(r["lenient_correct"])
            cell = table[p["category"]][s["status"]]
            cell["probes"] += 1
            cell["judge_correct"] += bool(r["lenient_correct"])
            cell["response_contains_gold"] += rule.contains_answer(r["response"], p["gold_answer"])
            if s["status"] == "stale":
                has_final = rule.contains_answer(r["response"], s["final"])
                cell["response_contains_final"] += has_final
                cell["final_in_response_judged_correct"] += has_final and bool(r["lenient_correct"])
                every = table["_all_categories"]["stale"]
                every["response_contains_final"] += has_final
                every["final_in_response_judged_correct"] += has_final and bool(
                    r["lenient_correct"]
                )
            if s["sound"]:
                sound_cell = table["_sound_subset"]["all"]
                sound_cell["probes"] += 1
                sound_cell["judge_correct"] += bool(r["lenient_correct"])
        runs[meta["run_path"]] = {
            cat: {st: dict(sorted(c.items())) for st, c in sorted(cells.items())}
            for cat, cells in sorted(table.items())
        }
    totals = Counter(s["status"] for s in status.values())
    if len(sheet_shape) != 1:
        raise SystemExit(f"personas differ in fact-sheet shape: {sorted(sheet_shape)}")
    n_facts, n_attributes, n_twice = sheet_shape.pop()
    return {
        "artifact_kind": "pmb-answer-key-audit",
        "generated_by": "scripts/pmb_answer_key_audit.py",
        "status": "post-hoc: written after Study 001; describes PMB's answer key, changes no score",
        "answerable_probes": len(probes),
        "unanswerable_probes": len(every_probe) - len(probes),
        "fact_sheet_per_persona": {
            "facts": n_facts,
            "attributes": n_attributes,
            "attributes_stated_twice": n_twice,
        },
        "status_counts": dict(sorted(totals.items())),
        "noop_corrections": sum(1 for s in status.values() if s.get("noop_correction")),
        "noop_corrections_on_suffixed_predicates": sum(
            1 for s in status.values() if s.get("noop_correction") and s.get("suffixed_predicate")
        ),
        "twin_predicate_probes": sum(1 for s in status.values() if s.get("twin_predicate")),
        "shared_question_different_gold": sum(
            1 for s in status.values() if s["shared_question_different_gold"]
        ),
        "stale_by_route": dict(
            sorted(
                Counter(s["stale_route"] for s in status.values() if s.get("stale_route")).items()
            )
        ),
        "stale_by_predicate": dict(
            sorted(
                Counter(
                    _base(s["predicate"]) for s in status.values() if s["status"] == "stale"
                ).items()
            )
        ),
        "sound_subset": {
            "definition": "status current (the gold answer is the attribute's last-stated "
            "value), not an outdated_fact probe, question not shared with a probe of a different "
            "gold answer",
            "probes": sum(1 for s in status.values() if s["sound"]),
        },
        "status_by_category": {c: dict(sorted(v.items())) for c, v in sorted(by_cat.items())},
        "runs": runs,
        "probes": {pid: status[pid] for pid in sorted(status)},
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
