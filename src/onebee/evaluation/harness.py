from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from onebee.evaluation.graders.judge import Judge
from onebee.evaluation.graders.nli import NLIChecker
from onebee.evaluation.metrics.personalized import (
    Probe,
    ProbeResult,
    pra_lenient,
    pra_strict,
    score_probe,
    uar,
)
from onebee.evaluation.stats import bootstrap_ci


class SystemConfig(BaseModel):
    name: str
    description: str = ""


class HarnessResult(BaseModel):
    system: str
    n_probes: int
    metrics: dict[str, float]
    metrics_ci: dict[str, tuple[float, float, float]]
    per_category: dict[str, dict[str, float]]
    raw_results: list[dict[str, Any]]


def run_harness(
    probes: list[Probe],
    response_fn: Callable[[Probe], tuple[str, list[str]]],
    judge: Judge | None = None,
    nli: NLIChecker | None = None,
    system_name: str = "system",
    n_bootstrap: int = 10000,
    seed: int | None = None,
) -> HarnessResult:
    results: list[ProbeResult] = []
    for probe in probes:
        response_text, retrieved_ids = response_fn(probe)
        pr = score_probe(probe, response_text, retrieved_ids, judge=judge)
        results.append(pr)

    metrics: dict[str, float] = {}
    metrics_ci: dict[str, tuple[float, float, float]] = {}

    pra_s_val = pra_strict(results)
    metrics["pra_strict"] = pra_s_val

    if judge is not None:
        pra_l_val = pra_lenient(results)
        metrics["pra_lenient"] = pra_l_val

    uar_val = uar(results)
    metrics["uar"] = uar_val

    per_probe_strict = [1.0 if r.strict_correct else 0.0 for r in results if r.probe.answerable]
    if per_probe_strict:
        metrics_ci["pra_strict"] = bootstrap_ci(
            per_probe_strict, n_resamples=n_bootstrap, seed=seed
        )

    per_probe_uar = [1.0 if r.abstained else 0.0 for r in results if not r.probe.answerable]
    if per_probe_uar:
        metrics_ci["uar"] = bootstrap_ci(per_probe_uar, n_resamples=n_bootstrap, seed=seed)

    per_category: dict[str, dict[str, float]] = {}
    categories = {p.category for p in probes}
    for cat in categories:
        cat_results = [r for r in results if r.probe.category == cat]
        if not cat_results:
            continue
        cat_metrics: dict[str, float] = {}
        answerable_cat = [r for r in cat_results if r.probe.answerable]
        if answerable_cat:
            cat_metrics["pra_strict"] = pra_strict(answerable_cat)
            if judge is not None:
                cat_metrics["pra_lenient"] = pra_lenient(answerable_cat)
        unanswerable_cat = [r for r in cat_results if not r.probe.answerable]
        if unanswerable_cat:
            cat_metrics["uar"] = uar(unanswerable_cat)
        if cat_metrics:
            per_category[cat] = cat_metrics

    raw_results = [pr.model_dump() for pr in results]

    return HarnessResult(
        system=system_name,
        n_probes=len(probes),
        metrics=metrics,
        metrics_ci=metrics_ci,
        per_category=per_category,
        raw_results=raw_results,
    )


def save_harness_result(result: HarnessResult, path: str) -> None:
    os.makedirs(path, exist_ok=True)

    suffixless = {
        "system": result.system,
        "n_probes": result.n_probes,
        "metrics": result.metrics,
        "metrics_ci": {k: list(v) for k, v in result.metrics_ci.items()},
        "per_category": result.per_category,
    }

    metrics_path = os.path.join(path, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(suffixless, f, indent=2)

    raw_path = os.path.join(path, "raw.jsonl")
    with open(raw_path, "w") as f:
        for record in result.raw_results:
            f.write(json.dumps(record) + "\n")
