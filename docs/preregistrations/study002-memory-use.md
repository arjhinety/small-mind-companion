# Pre-registration: gold retrieval and accuracy-when-retrieved, replacing `mur` (Study 002)

**Registered:** on the date of the commit that adds this file. At that commit no Study 002
evaluation run is committed or indexed (`results/registry.jsonl` holds only Study 001 runs). Runs
happen on the GPU box, so this attests to the repository, not to the machine. Any later change to
this document is a dated amendment below, never an edit in place.

**Role: secondary and diagnostic.** `pra_lenient` stays the headline recall metric. These metrics
explain it: whether the needed memory was retrieved, whether it reached the model, and how often the
model answered correctly when it did. They do not test H5. H5 needs a generic-SFT arm that was
never trained (`reports/ERRATA.md` E36), and no metric supplies it.

## Why `mur` is retired

`mur` asked whether a gold memory was among the retrieved ones. It was 0.0 in every Study 001 run
because the probes cite benchmark turn ids (`p000_s003_t008`) while the store gives each memory a
random UUID, so the two never matched (E36). Even with the ids matched it measures only retrieval:
mapped correctly, the needed memory is retrieved for the same 156 of 608 answerable probes in all
four Study 001 memory runs (`results/audit/memory_use/memory_use.json`), so it cannot tell two
models that share a retriever apart. From this commit the harness no longer computes `mur`. Study
001's saved `metrics.json` files keep their recorded `mur: 0.0` unchanged; the name is not reused.

## Definitions

Each stored memory records the benchmark turns it was extracted from (`provenance.turn_ids`). For
one answerable probe with gold supporting turns `G`, two memory lists are recorded:

- `R_ret`: the ids, in order, that the retriever returned;
- `R_prompt`: the ids, in order, that reached the prompt (the context builder's `kept_memory_ids`,
  after any token-budget truncation).

**Needed turns.** For a PMB `outdated_fact` probe the gold turns are a fact and its later update,
and only the update answers the question, so the needed turns `N` are the latest gold turn alone
(by session, then turn). For every other probe `N = G`. Retrieving only the stale fact is not a hit;
it is counted separately as `stale_fact_only`.

```
gold_retrieved(R) = any(turn_map[m] ∩ N ≠ ∅ for m in R)
```

where `turn_map` is read from the store the run used. A retrieved id missing from that store is an
error, not a miss (it means the wrong store was given).

- **`gold_retrieval_rate`** = answerable probes with `gold_retrieved(R_ret)` / answerable probes.
  A **retriever metric**: neither the context builder nor the model's answer enters it.
- **`all_gold_retrieval_rate`** = answerable probes where every turn in `G` is covered by some memory
  in `R_ret` / answerable probes. Reported beside `gold_retrieval_rate`; it is not used for hit sets.
- **`gold_rank`** = the 1-based position in `R_ret` of the first memory whose turns meet `N`, so at
  most the list's length; `null` when there is none (a miss is never ranked 0 or k+1). Rank over the
  retriever's full candidate ranking belongs to the retrieval pre-registration, not here.
- **`stale_fact_only`** = `outdated_fact` probes where a stale gold turn is covered by `R_ret` and the
  update is not, as a count.
- **`accuracy_when_retrieved`** = among answerable probes with `gold_retrieved(R_prompt)` (the needed
  memory reached the model), the share marked correct. Reported twice: by the judge
  (`lenient_correct`) and by `contains_answer` (`docs/preregistrations/study002-answer-containment.md`).
- **`correct_without_retrieval`** = answerable probes marked correct where `gold_retrieved(R_prompt)`
  is false, as a count, by each of the two methods.

Unanswerable probes are excluded throughout. Systems with no memory have `gold_retrieval_rate` 0
and `accuracy_when_retrieved` undefined (`null`).

**Worked example (asserted literally in a unit test).** Here `R_ret = R_prompt`. Memories `u1`→
`p_s0_t1`, `u2`→`p_s0_t2`, `u3`→`p_s0_t9`, `u4`→`p_s1_t3` (a fact), `u5`→`p_s5_t7` (its update):

| retrieved | gold turns | category | judge | gold_retrieved | all gold | gold_rank | stale only |
|---|---|---|---|---|---|---|---|
| u3, u1 | p_s0_t1 | factual | correct | yes | yes | 2 | no |
| u2 | p_s0_t2, p_s0_t5 | factual | wrong | yes | no | 1 | no |
| u3 | p_s0_t3 | factual | correct | no | no | null | no |
| u4 | p_s1_t3, p_s5_t7 | outdated_fact | wrong | no | no | null | yes |
| u1, u5 | p_s1_t3, p_s5_t7 | outdated_fact | correct | yes | no | 2 | no |

`gold_retrieval_rate` = 3/5 = **0.6**; `all_gold_retrieval_rate` = 1/5 = **0.2**;
`accuracy_when_retrieved` = 2/3; `correct_without_retrieval` = **1**; `stale_fact_only` = **1**.

## Comparing systems

`accuracy_when_retrieved` is compared between two systems only on the probes where **both** runs had
the needed memory reach the prompt (the intersection of their hit sets on `R_prompt`), so the
comparison is paired. This fixes which probes are compared, not what reached the model: the two
prompts can still hold different other memories, in a different order. A difference is therefore
diagnostic; it does not show that a model used the memory, only how often it answered correctly when
the memory was present. When two runs share a retriever, stores and run settings, their hit sets are
identical (Study 001: the same 156 probes). `gold_retrieval_rate` is reported on all answerable
probes, so the intersection never hides a retrieval difference. Each comparison reports both runs'
`hit_set_sha256`, the size of the intersection (an empty one is reported as size 0), the exact
McNemar p-value on the discordant probes, and a paired bootstrap interval (`paired_bootstrap_diff`,
10 000 resamples, seed 1337). If Study 002 pre-registers a different interval method for
`pra_lenient`, these follow it by amendment.

`accuracy_when_retrieved` is **interpreted only when at least 30 probes** have the needed memory in
the prompt (in both runs, for a comparison). Below that it is reported and called uninterpretable.

## What every Study 002 report includes

For each system with memory: `gold_retrieval_rate`, `all_gold_retrieval_rate`, the distribution of
`gold_rank`, `stale_fact_only`, `accuracy_when_retrieved` by the judge and by `contains_answer`,
`correct_without_retrieval` by both, and `hit_set_sha256`. For each comparison: the paired figures
above.

**Run record.** Retrieval depends on more than the code: the process's hash seed (MMR
tie-breaking), the clock (recency) and the SQLite build (E42; across hash seeds 0–20 the audit's
replay matched D_memory's saved retrieved sets for 682 to 687 of 688 probes, `hash_seed_spread` in
`results/audit/retrieval/retrieval_audit.json`). Every Study 002 run therefore runs with
`PYTHONHASHSEED=0`, set before the interpreter starts (the run script re-executes itself, as
`scripts/retrieval_audit.py` does), and records the value in effect. Each memory run records, per
probe, `R_ret` and `R_prompt`, and per run: the sha256 of each store file, the embedding model and
revision, the retriever configuration, the retrieval clock, and the Python and SQLite versions.
Runs on the frozen Study 001 retriever also pin the clock at 2026-08-13T12:00:00+00:00.

**Store requirement.** The mapping depends on the store each run used. Its stores are committed (or
are the already committed `data/stores/` files with matching hashes). Stores built from data whose
license forbids redistribution (HaluMem, CC BY-NC-ND 4.0) are the exception: the run records the
pinned dataset revision, the committed build script and each store's sha256, and saved runs keep
memory ids and metrics, never the data's text. How such data's gold evidence maps to memories is
fixed in the retrieval pre-registration, since HaluMem's gold is memory text, not turn ids. A run
meeting neither condition cannot report these metrics.

## Known limitations

- **Gold turn lists may be incomplete.** A fact can be stated in more than one turn. Study 001 had
  11 to 35 judge-correct answers per memory run without the needed memory retrieved; some may be
  facts stated elsewhere, some guesses, some judge leniency. `correct_without_retrieval` is reported
  so this stays visible.
- **Extraction caps retrieval.** A memory was extracted from a needed turn for 561 of 608 answerable
  probes; the other 47 cannot be gold-retrieved by any retriever.
- **Probes are not independent** (the same persona fact recurs across categories), so probe-level
  intervals are likely too narrow.
- **One judge.** The judge-scored version inherits the judge's errors and possible preference for
  its own style (E38); the `contains_answer` version inherits that rule's failure modes.

**Designed after seeing Study 001.** These definitions and the 30-probe minimum were chosen with
Study 001's results and the retrieval audit (E42) in view. In Study 001 the context builder dropped
or reordered no memory in any run, so `R_prompt = R_ret` there. Under these definitions the needed
memory was retrieved for 156 of 608 answerable probes in every memory run (186 with any gold turn;
30 `outdated_fact` probes had only the stale fact); judge accuracy when retrieved was 51.92% (v0.1
D), 57.69% (v0.1 E), 41.03% (v1 E), 50.00% (v1 E_distill). Those figures are post-hoc description,
not evidence for these metrics. They are fixed here before Study 002.

## Implementation

- `src/onebee/evaluation/metrics/memory_use.py` (standard library only): `source_turn_map`,
  `needed_turns`, `gold_retrieved`, `gold_rank`, `all_gold_retrieved`, `stale_fact_only`,
  `gold_retrieval_rate`, `accuracy_when_retrieved`. Computed offline from each run's `raw.jsonl` and
  stores.
- Study 002 run scripts keep the context builder's trace and write `R_ret`, `R_prompt` and the run
  record above into `raw.jsonl`. Study 001's scripts are unchanged.
- `scripts/memory_use.py` produces the Study 001 post-hoc artifact with these same functions,
  including the context-builder replay.
- `tests/unit/test_memory_use.py` asserts the worked example. Changing the definitions fails it
  until an amendment below explains the change.

## Amendments

None.
