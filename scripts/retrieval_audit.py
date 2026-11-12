"""Audit of Study 001's memory retrieval: reproduce it, then show where the right memory is lost.

POST-HOC. Written after Study 001 froze, from its committed stores and responses. No model runs:
the probe questions were embedded once by ``scripts/embed_probe_questions.py``
(``results/audit/retrieval/question_embeddings.npy``) with the same model revision, library
versions, call pattern and (absent) prefix as the runs, but on CPU. Study 001 likely encoded on CUDA
(``run_system_d.py:34`` sets no device), so the vectors may differ from the runs' in the last bits.
The replay is therefore a close reproduction, not an exact one.

What it records, for answerable probes, with "gold" meaning a memory extracted from one of the
probe's gold supporting turns (``provenance.turn_ids``):

* **replay** -- the frozen ``HybridRetriever`` re-run on copies of ``data/stores/pmb_v0_full``,
  clock pinned to ``CLOCK``, compared list-for-list with each memory run's saved
  ``retrieved_memory_ids``.
* **ordering** -- ``MemoryStore.search`` selects the nearest vectors and the best FTS5 matches in a
  subquery but returns them through ``WHERE id IN (...)`` with no outer ``ORDER BY``, i.e. in table
  order. Gold hit@k is reported for each half as returned and in true rank order.
* **2x2** -- the full frozen pipeline with ordering as returned / in rank order, and the default
  tier quota (episodic 4, semantic 3, short_term 6) on / off; plus quota off truncated to each
  probe's quota-on list size, and a chance rate per arm from that arm's own list sizes.
* **k-sweep** -- whether the k=8 and k=16 runs retrieved different memories.

The stores are copied to a temporary directory first: opening them with ``MemoryStore`` (WAL mode)
rewrites header bytes of the committed files. Read-only for everything under ``results/`` except the
output below.

**Tie-breaking.** The frozen MMR step picks ``max`` over a Python ``set`` of candidates, so ties
are broken in set-iteration order, which depends on the process's string-hash seed. Study 001's
runs used Python's default random seed, so their retrieval order depended on tie-breaking. This
script runs at ``PYTHONHASHSEED=0`` (fixed before any seed's results were examined; it re-launches
itself if the variable is unset or different) and records, in ``hash_seed_spread``, the
min/median/max of every seed-dependent figure over seeds 0-20.

Usage::

    uv run python scripts/retrieval_audit.py                      # (re)write the output JSON
    uv run python scripts/retrieval_audit.py --check              # exit 1 if stale (21 seeds; slow)
    uv run python scripts/retrieval_audit.py --check --no-spread  # same, minus hash_seed_spread
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

from onebee.memory.store import MemoryStore, _sanitize_fts_query
from onebee.retrieval import fusion as fusion_mod
from onebee.retrieval import router as router_mod
from onebee.retrieval.fusion import FusionWeights, RetrievalCandidate
from onebee.retrieval.router import HybridRetriever

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "audit" / "retrieval"
OUTPUT = OUT_DIR / "retrieval_audit.json"
EMBEDDINGS = OUT_DIR / "question_embeddings.npy"
MANIFEST = OUT_DIR / "question_embeddings.json"
STORES = ROOT / "data" / "stores" / "pmb_v0_full"
PROBES = ROOT / "data" / "benchmarks" / "pmb_v0_full" / "probes.jsonl"
MEMORY_RUNS = (
    "results/v0.1/D_memory",
    "results/v0.1/E_sft_memory",
    "results/v1_scale/E_sft_memory",
    "results/v1_scale/E_distill",
)
KSWEEP = "results/v0.1/ksweep"
# The D_memory run's day; recency is the only clock-dependent term, and a clock within a few days
# of it gives the same retrieval.
CLOCK = "2026-08-13T12:00:00+00:00"
NO_QUOTA = {"episodic": 10**6, "semantic": 10**6, "short_term": 10**6}
K = 8
# Alternative clocks for the date-dependence check, compared with a replay at CLOCK, same seed.
ALT_CLOCKS = (
    "2026-08-12T12:00:00+00:00",
    "2026-08-13T00:00:00+00:00",
    "2026-08-15T12:00:00+00:00",
    "2026-09-13T12:00:00+00:00",
)
TERMS = ("similarity", "recency", "importance", "confidence", "entity")
HASH_SEED = "0"  # fixed before any seed's results were examined
SPREAD_SEEDS = range(21)


class _PinnedClock:
    def __init__(self, iso: str) -> None:
        self._t = datetime.fromisoformat(iso).astimezone(timezone.utc).timestamp()

    def time(self) -> float:
        return self._t


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _ranked_dense(store: MemoryStore, embedding: list[float], n: int) -> list[str]:
    rows = store._conn.execute(
        "SELECT m.id FROM memory_vec v JOIN memory m ON m.embedding_id = v.embedding_id "
        "WHERE v.embedding MATCH ? AND k = ? AND m.status = 'active' ORDER BY v.distance",
        [json.dumps(embedding), n],
    ).fetchall()
    return [r[0] for r in rows]


def _ranked_fts(store: MemoryStore, question: str, n: int) -> list[str]:
    query = _sanitize_fts_query(question)
    if not query:
        return []
    rows = store._conn.execute(
        "SELECT m.id FROM memory_fts f JOIN memory m ON m.rowid = f.rowid "
        "WHERE memory_fts MATCH ? AND m.status = 'active' ORDER BY f.rank LIMIT ?",
        [query, n],
    ).fetchall()
    return [r[0] for r in rows]


def _as_candidates(store: MemoryStore, ids: list[str], source: str) -> list[RetrievalCandidate]:
    out = []
    for i, memory_id in enumerate(ids):
        record = store.get_by_id(memory_id)
        assert record is not None
        out.append(
            RetrievalCandidate(
                record=record.model_dump(), memory_id=memory_id, source=source, rank=i + 1
            )
        )
    return out


class _RankOrdered:
    """Swap in rank-ordered dense and sparse strategies, keeping the store's candidate count."""

    def __init__(self, retriever: HybridRetriever) -> None:
        store = retriever.store
        retriever.dense.retrieve = lambda emb, tier=None, k=20: _as_candidates(  # type: ignore[method-assign]
            store, _ranked_dense(store, emb, k * 2), "dense"
        )
        retriever.sparse.retrieve = lambda q, tier=None, k=20: _as_candidates(  # type: ignore[method-assign]
            store, _ranked_fts(store, q, k * 2), "sparse"
        )


def _hit_at(ranks: list[int | None], k: int) -> float:
    return round(sum(1 for r in ranks if r is not None and r <= k) / len(ranks), 4)


def build() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if hashlib.sha256(EMBEDDINGS.read_bytes()).hexdigest() != manifest["sha256"]:
        raise SystemExit("question_embeddings.npy does not match its manifest")
    vectors = np.load(EMBEDDINGS)
    probes = _rows(PROBES)
    if [p["probe_id"] for p in probes] != manifest["probe_ids"]:
        raise SystemExit("probe order differs from the embeddings manifest")
    embedding = {p["probe_id"]: vectors[i].tolist() for i, p in enumerate(probes)}
    answerable = [p for p in probes if p["answerable"]]

    router_mod.time = _PinnedClock(CLOCK)  # type: ignore[assignment]
    with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stores: dict[str, MemoryStore] = {}
        turns: dict[str, set[str]] = {}
        persona_of: dict[str, str] = {}
        tiers: Counter[str] = Counter()
        sizes: dict[str, int] = {}
        # memories whose recency uses created_at (fusion.py:70 reads `event_time or created_at`,
        # so an event_time of 0 falls back to the wall-clock insert time)
        boosted: set[str] = set()
        for db in sorted(STORES.glob("p*.db")):
            copy = Path(tmp) / db.name
            shutil.copyfile(db, copy)
            stores[db.stem] = MemoryStore(str(copy))
            rows = stores[db.stem]._conn.execute(
                "SELECT id, provenance, tier, event_time FROM memory"
            )
            sizes[db.stem] = 0
            for memory_id, provenance, tier, event_time in rows:
                if not event_time:
                    boosted.add(memory_id)
                turns[memory_id] = set(json.loads(provenance or "{}").get("turn_ids") or [])
                persona_of[memory_id] = db.stem
                tiers[tier] += 1
                sizes[db.stem] += 1

        def is_gold(memory_id: str, probe: dict[str, Any]) -> bool:
            return bool(turns[memory_id] & set(probe["gold_supporting_memory_ids"]))

        def first_gold(ids: list[str], probe: dict[str, Any]) -> int | None:
            return next((i + 1 for i, m in enumerate(ids) if is_gold(m, probe)), None)

        def retrieve(
            probe: dict[str, Any],
            ordered: bool,
            quota: bool,
            weights: FusionWeights | None = None,
        ) -> list[str]:
            retriever = HybridRetriever(stores[probe["persona_id"]], weights=weights)
            if ordered:
                _RankOrdered(retriever)
            found = retriever.retrieve(
                probe["question"],
                query_embedding=embedding[probe["probe_id"]],
                k=K,
                tier_quotas=None if quota else NO_QUOTA,
            )
            return [c.memory_id for c in found]

        # 1. replay the frozen retriever against every memory run's saved lists
        replay = {}
        for run in MEMORY_RUNS:
            saved = _rows(ROOT / run / "raw.jsonl")
            same_list = same_set = 0
            mismatched = []
            for row in saved:
                got = retrieve(row["probe"], ordered=False, quota=True)
                same_list += got == row["retrieved_memory_ids"]
                same_set += set(got) == set(row["retrieved_memory_ids"])
                if got != row["retrieved_memory_ids"]:
                    mismatched.append(row["probe"]["probe_id"])
            replay[run] = {
                "probes": len(saved),
                "identical_lists": same_list,
                "identical_sets": same_set,
                "mismatched_probe_ids": mismatched,
                "saved_lists_with_boosted_memory": sum(
                    1 for row in saved if boosted & set(row["retrieved_memory_ids"])
                ),
            }

        # 1b. date dependence: replay D_memory at other clocks, same seed, vs the CLOCK replay
        d_rows = _rows(ROOT / MEMORY_RUNS[0] / "raw.jsonl")
        at_clock = [retrieve(row["probe"], ordered=False, quota=True) for row in d_rows]
        clock_shift = {}
        for alt in ALT_CLOCKS:
            router_mod.time = _PinnedClock(alt)  # type: ignore[assignment]
            moved = [retrieve(row["probe"], ordered=False, quota=True) for row in d_rows]
            clock_shift[alt] = sum(a != b for a, b in zip(at_clock, moved, strict=True))
        router_mod.time = _PinnedClock(CLOCK)  # type: ignore[assignment]

        # 1c. per-query spread of each weighted composite term over the scored candidate pool
        captured: list[dict[str, float]] = []
        original_score = router_mod.compute_composite_score

        def spy(*, record, rrf_score, query_entities, weights, now_ms, **kw):  # type: ignore[no-untyped-def]
            ts = record.get("event_time") or record.get("created_at", 0)
            entities = set(record.get("entities") or [])
            captured.append(
                {
                    "similarity": weights.w_sim * min(1.0, rrf_score),
                    "recency": weights.w_rec * math.exp(-(now_ms - ts) / (7 * 24 * 3600 * 1000)),
                    "importance": weights.w_imp * record.get("importance", 0.0),
                    "confidence": weights.w_conf * record.get("confidence", 0.0),
                    "entity": weights.w_ent * fusion_mod._jaccard(query_entities, entities),
                }
            )
            return original_score(
                record=record,
                rrf_score=rrf_score,
                query_entities=query_entities,
                weights=weights,
                now_ms=now_ms,
                **kw,
            )

        router_mod.compute_composite_score = spy
        term_spread = {}
        for ordered in (False, True):
            per_query: dict[str, list[float]] = {t: [] for t in TERMS}
            for p in answerable:
                captured.clear()
                retrieve(p, ordered=ordered, quota=True)
                for t in TERMS:
                    values = [c[t] for c in captured]
                    per_query[t].append(max(values) - min(values) if values else 0.0)
            sim = per_query["similarity"]
            term_spread["rank" if ordered else "as_returned"] = {
                t: {
                    "median_range": round(statistics.median(v), 5),
                    "max_range": round(max(v), 5),
                    "queries_range_exceeds_similarity": sum(
                        1 for a, b in zip(v, sim, strict=True) if a > b
                    ),
                }
                for t, v in per_query.items()
            }
        router_mod.compute_composite_score = original_score

        def chance_rate(lists: list[list[str]]) -> float:
            """Mean, over answerable probes, of P(a random draw of that probe's list size from
            its persona's store includes a gold memory)."""
            total = 0.0
            for p, ids in zip(answerable, lists, strict=True):
                n, k = sizes[p["persona_id"]], len(ids)
                g = sum(1 for m in turns if persona_of[m] == p["persona_id"] and is_gold(m, p))
                total += 0.0 if g == 0 else 1 - comb(n - g, k) / comb(n, k)
            return round(total / len(answerable), 4)

        # 2. chance level at the list sizes the D_memory run actually returned
        saved_d = {r["probe"]["probe_id"]: r for r in _rows(ROOT / MEMORY_RUNS[0] / "raw.jsonl")}
        d_lists = [saved_d[p["probe_id"]]["retrieved_memory_ids"] for p in answerable]
        observed = sum(
            any(is_gold(m, p) for m in ids) for p, ids in zip(answerable, d_lists, strict=True)
        )

        # 3. each half as returned vs in rank order
        halves: dict[str, list[int | None]] = {
            "dense_as_returned": [],
            "dense_rank_ordered": [],
            "keyword_as_returned": [],
            "keyword_rank_ordered": [],
        }
        for p in answerable:
            store = stores[p["persona_id"]]
            emb = embedding[p["probe_id"]]
            halves["dense_as_returned"].append(
                first_gold([r.id for r in store.search(query_embedding=emb, k=40)], p)
            )
            halves["dense_rank_ordered"].append(first_gold(_ranked_dense(store, emb, 80), p))
            halves["keyword_as_returned"].append(
                first_gold([r.id for r in store.search(query=p["question"], k=40)], p)
            )
            halves["keyword_rank_ordered"].append(
                first_gold(_ranked_fts(store, p["question"], 80), p)
            )

        # 4. full pipeline 2x2, plus quota-off truncated to each probe's quota-on list size, so
        # ordering and quota are also compared at equal list size; chance is per arm
        def arm(lists: list[list[str]]) -> dict[str, Any]:
            hits = sum(
                any(is_gold(m, p) for m in ids) for p, ids in zip(answerable, lists, strict=True)
            )
            return {
                "gold_retrieved": hits,
                "rate": round(hits / len(answerable), 4),
                "chance_rate": chance_rate(lists),
                "mean_list_size": round(sum(map(len, lists)) / len(lists), 3),
                "lists_with_boosted_memory": sum(1 for ids in lists if boosted & set(ids)),
            }

        grid = {}
        for ordered in (False, True):
            name = "rank" if ordered else "as_returned"
            on = [retrieve(p, ordered, True) for p in answerable]
            off = [retrieve(p, ordered, False) for p in answerable]
            grid[f"ordering={name},quota=on"] = arm(on)
            grid[f"ordering={name},quota=off"] = arm(off)
            grid[f"ordering={name},quota=off,truncated_to_quota_on_size"] = arm(
                [o[: len(q)] for o, q in zip(off, on, strict=True)]
            )
        # POST-HOC DIAGNOSTIC, not a 2x2 cell: rank order, quota off, recency weight 0
        grid["post-hoc diagnostic: ordering=rank,quota=off,w_rec=0"] = arm(
            [retrieve(p, True, False, FusionWeights(w_rec=0.0)) for p in answerable]
        )
        # POST-HOC DIAGNOSTIC: rank order, quota off, importance and confidence weights 0
        grid["post-hoc diagnostic: ordering=rank,quota=off,w_imp=0,w_conf=0"] = arm(
            [retrieve(p, True, False, FusionWeights(w_imp=0.0, w_conf=0.0)) for p in answerable]
        )
        for store in stores.values():
            store._conn.close()

    # 5. k-sweep: did k=16 retrieve anything k=8 did not?
    def ksweep_lists(k: str) -> dict[str, list[str]]:
        rows = _rows(ROOT / KSWEEP / k / "raw.jsonl")
        return {r["probe"]["probe_id"]: r["retrieved_memory_ids"] for r in rows}

    k8, k16 = ksweep_lists("k8"), ksweep_lists("k16")

    n_ans = len(answerable)
    return {
        "artifact_kind": "retrieval-audit-posthoc",
        "generated_by": "scripts/retrieval_audit.py",
        "status": "post-hoc audit of Study 001's frozen retriever; no model run",
        "clock": CLOCK,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "question_embeddings": {
            k: manifest[k]
            for k in (
                "model",
                "revision",
                "library_versions",
                "device",
                "device_differs_from_runs",
                "encoding",
                "sha256",
            )
        },
        "stores": {
            "memories": sum(sizes.values()),
            "per_persona": sizes,
            "tiers": dict(sorted(tiers.items())),
        },
        "answerable_probes": n_ans,
        "recency_boosted_memories": {
            "rule": "event_time is 0 or missing, so fusion.py:70 uses created_at (wall clock)",
            "count": len(boosted),
            "per_persona": dict(sorted(Counter(persona_of[m] for m in boosted).items())),
        },
        "replay": replay,
        "d_memory_clock_shift": {
            "compared_with": f"replay at {CLOCK}, same hash seed",
            "lists_changed_of_688": clock_shift,
        },
        "composite_term_range_per_query": {
            "note": "weighted term, max minus min over each answerable query's scored candidate"
            " pool (frozen pipeline, quota on); entity is always 0: no query entities are passed",
            **term_spread,
        },
        "d_memory_gold_retrieved": observed,
        "d_memory_chance_rate": chance_rate(d_lists),
        "halves_gold_hit_at": {
            name: {f"@{k}": _hit_at(ranks, k) for k in (1, 3, 8, 20)}
            for name, ranks in halves.items()
        },
        "pipeline_2x2": grid,
        "ksweep": {
            "probes": len(k8),
            "k8_k16_identical_lists": sum(k8[p] == k16[p] for p in k8),
            "k8_list_sizes": dict(sorted(Counter(len(v) for v in k8.values()).items())),
            "k16_list_sizes": dict(sorted(Counter(len(v) for v in k16.values()).items())),
        },
    }


def _differences(old: Any, new: Any, path: str = "") -> list[tuple[str, Any, Any]]:
    """Leaf-level differences between two JSON values, as (key path, old, new)."""
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[tuple[str, Any, Any]] = []
        for key in sorted(set(old) | set(new)):
            out += _differences(old.get(key), new.get(key), f"{path}/{key}")
        return out
    return [] if old == new else [(path or "/", old, new)]


def serialise(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def _worker_doc(seed: int) -> dict[str, Any]:
    env = {**os.environ, "PYTHONHASHSEED": str(seed)}
    out = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def _spread(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def hash_seed_spread() -> dict[str, Any]:
    """Re-run the seed-dependent figures under every seed in SPREAD_SEEDS, in subprocesses."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        docs = list(pool.map(_worker_doc, SPREAD_SEEDS))
    cells = docs[0]["pipeline_2x2"]
    return {
        "seeds": f"PYTHONHASHSEED {SPREAD_SEEDS.start}-{SPREAD_SEEDS.stop - 1}",
        "pipeline_2x2": {
            cell: {m: _spread([d["pipeline_2x2"][cell][m] for d in docs]) for m in cells[cell]}
            for cell in cells
        },
        "replay": {
            run: {
                **{
                    m: _spread([d["replay"][run][m] for d in docs])
                    for m in ("identical_lists", "identical_sets")
                },
                "probes_mismatched_under_some_seed": len(
                    set().union(*(d["replay"][run]["mismatched_probe_ids"] for d in docs))
                ),
                "probes_mismatched_under_every_seed": len(
                    set.intersection(*(set(d["replay"][run]["mismatched_probe_ids"]) for d in docs))
                ),
                "best_seed": min(
                    SPREAD_SEEDS,
                    key=lambda s: (len(docs[s]["replay"][run]["mismatched_probe_ids"]), s),
                ),
            }
            for run in docs[0]["replay"]
        },
        "d_memory_clock_shift": {
            alt: _spread([d["d_memory_clock_shift"]["lists_changed_of_688"][alt] for d in docs])
            for alt in ALT_CLOCKS
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the output is stale")
    parser.add_argument(
        "--no-spread", action="store_true", help="with --check: skip recomputing hash_seed_spread"
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    raw_args = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(raw_args)
    if args.worker:
        print(json.dumps(build()))
        return 0
    if os.environ.get("PYTHONHASHSEED") != HASH_SEED:
        env = {**os.environ, "PYTHONHASHSEED": HASH_SEED}
        script = str(Path(__file__).resolve())
        return subprocess.run([sys.executable, script, *raw_args], env=env, cwd=ROOT).returncode
    document = build()
    rel = OUTPUT.relative_to(ROOT).as_posix()
    if args.check and args.no_spread:
        committed = json.loads(OUTPUT.read_text(encoding="utf-8")) if OUTPUT.is_file() else {}
        committed.pop("hash_seed_spread", None)
        fresh = json.loads(serialise(document))
        if committed != fresh:
            print(
                f"{rel} is stale (excluding hash_seed_spread); differing fields:", file=sys.stderr
            )
            for path, old, new in _differences(committed, fresh):
                print(f"  {path}: committed {old!r} != recomputed {new!r}", file=sys.stderr)
            return 1
        print(f"{rel} is current (hash_seed_spread not rechecked)")
        return 0
    document["hash_seed_spread"] = hash_seed_spread()
    text = serialise(document)
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != text:
            print(f"{rel} is stale; re-run without --check", file=sys.stderr)
            return 1
        print(f"{rel} is current")
        return 0
    OUTPUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
