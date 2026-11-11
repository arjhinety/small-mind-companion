# Pre-registration: `pra_contain`, a judge-free cross-check for Study 002

**Registered:** 2026-11-11. At the commit that adds this file, no Study 002 evaluation run is
committed or indexed (`results/registry.jsonl` holds only Study 001 runs). Runs happen on the GPU
box, so this attests to the repository, not to the machine. Git history is the registration
record. Any later change to this document is a dated amendment below, never an edit in place.

**Role: secondary.** `pra_lenient` (the judge) stays the headline recall metric. `pra_contain` is
reported next to it and never overrides it. It does not replace a human evaluation (Study 002
item 14) or a second judge (item 15).

## Why it exists

Study 001's only judge-free recall metric, `pra_strict`, compares the *whole* response with the gold
answer, so it is ~0 on sentence-length answers (3 of 4816 responses; `reports/ERRATA.md` E41).
That leaves `pra_lenient` resting on a single LLM judge with no independent check. `pra_contain`
asks a simpler question that needs no model: does the gold answer appear in the response at all?

## Definition

For one answerable probe with gold answer `g` and model response `r`:

```
contains_answer(r, g) = normalize_text(g) != ""  and
                        re.search(r"(?<!\w)" + re.escape(normalize_text(g)) + r"(?!\w)",
                                  normalize_text(r)) is not None
```

`normalize_text` is the existing function in `src/onebee/evaluation/graders/rule.py` (lower-case,
delete ASCII punctuation, collapse whitespace). Deleting punctuation means `"9:00 AM"` becomes
`"900 am"` and `"hip-hop"` becomes `"hiphop"`; gold and response are normalised identically.

```
pra_contain = (number of answerable probes with contains_answer = true) / (number of answerable probes)
```

- **Denominator:** answerable probes (`probe.answerable` is true), the same denominator as
  `pra_lenient` and `pra_strict` (`src/onebee/evaluation/metrics/personalized.py:41-57`).
- **Unanswerable probes:** excluded from both numerator and denominator.
- **Empty gold:** counts as not contained. No answerable PMB probe has one today (0 of 608); the
  rule exists so the function is total.
- **Alternatives:** gold answer only. `acceptable_alternatives` is not consulted (it is empty in
  every PMB probe; E22, E41).
- **No answerable probes:** `pra_contain` is undefined (reported as `null`), not 0.

**Worked example (asserted literally in a unit test).** Six probes:

| gold | response | answerable | contains |
|---|---|---|---|
| Dublin | "You moved to Dublin." | yes | 1 |
| New York | "you live in new york." | yes | 1 |
| Harvard | "You graduated from either Berkeley or Harvard." | yes | 1 (known false positive) |
| April 30 | "Your birthday is April 30th." | yes | 0 (known false negative) |
| mystery | "You prefer mysteries." | yes | 0 (known false negative) |
| (any) | "I don't know." | **no** | excluded |

`pra_contain` = 3 / 5 = **0.6**.

## The rule is frozen as written, with known failure modes

The rule was written after reading Study 001's responses. Its agreement with the judge there
(Cohen's kappa 0.8887 to 0.9687 on the four memory runs;
`results/audit/pra_containment/pra_containment.json`) is motivation, not validation. Reading the 44
responses where it disagrees with the judge (33 where only the rule says correct, 11 where only the
judge does) showed these failure modes. They are **not** fixed,
because any fix would be tuned on those same rows:

- **Counts listed or hedged candidates as correct.** "either Berkeley or Harvard"; "Is it soccer,
  basketball, or something else?"; "Your hometown is Portland, though … Detroit". A model that lists
  more candidates scores higher without recalling more.
- **Counts negations and wrong attributions.** "I do not work for Apple"; "Diana considers Denver
  her hometown".
- **Misses inflections and suffixes.** "mysteries" for `mystery`; "April 30th" for `April 30`.
- **Misses partial answers the judge accepts.** "a cat" for `a tabby cat`.

The judge has its own errors in the same rows (it accepted "7:30 AM" for gold `8:00 AM`), so
neither method is ground truth.

## What every Study 002 report includes

For each evaluated system:

1. `pra_contain`, with a 95% bootstrap interval from `bootstrap_ci` in
   `src/onebee/evaluation/stats.py` over the per-probe 0/1 values (10 000 resamples, seed 1337).
   Differences between systems use `paired_bootstrap_diff` over the same probes. If Study 002
   later pre-registers a different interval method for `pra_lenient`, `pra_contain` follows it by
   amendment, so the two are always computed the same way.
2. The 2x2 table of `contains_answer` against the judge's `lenient_correct` on answerable probes.
3. Cohen's kappa for that table, **interpreted only when the judge marks at least 20 answerable
   probes correct and the rule, counted separately, also marks at least 20**. Below that, kappa is
   reported but called uninterpretable: on Study 001's three no-memory runs the judge marked 1
   probe correct in each and the rule 3, 3 and 4, giving kappas of 0.50, 0.50 and -0.003 that say
   nothing.
4. Every disagreement row (response, gold, both verdicts), committed with the run.

With several seeds (Study 002 item 2), each run is reported on its own. How runs are pooled is left
to that item's own pre-registration.

Probes are not independent: the same persona fact is asked in several categories, so a probe-level
bootstrap interval is likely narrower than the true uncertainty. Reports say so next to the interval.

## How a disagreement is read

- If kappa is interpretable and **below 0.6**, the system's `pra_lenient` is flagged in the report,
  and its disagreement rows are read and summarised before any claim rests on that number. The
  number itself is not changed.
- If the paired bootstrap of `contains_answer` minus the judge's `lenient_correct`, over the same
  answerable probes, gives a 95% interval that excludes 0, the report reads the disagreement rows on
  the larger side (candidate listing for contain-only, judge leniency for judge-only) before
  attributing the gap to either method.
- Neither condition turns `pra_contain` into the headline metric.

**The thresholds were chosen after seeing Study 001.** The 0.6 kappa line and the 20-positive
minimum were set with Study 001's kappas in view (0.89 to 0.97 with memory; uninterpretable
without). They are fixed here before Study 002, but they are not blind choices.

## Implementation

- `contains_answer(response, gold)` goes in `src/onebee/evaluation/graders/rule.py` next to
  `exact_match`. That file imports only `re` and `string`, so `scripts/validate.py` can keep loading
  it by path with no third-party package. `pra_contain(results)` goes in
  `src/onebee/evaluation/metrics/personalized.py` next to `pra_lenient`. Both are computed offline
  from each run's `raw.jsonl`; the harness, `metrics.json` and `pra_strict` are unchanged.
- `scripts/pra_containment.py` (exploratory, Study 001) switches to `contains_answer`, and its
  committed output must stay byte-identical (`--check` passes unchanged). That is the proof that
  the registered rule is the one analysed above.
- The rule is pinned by behaviour, not by file hash (`rule.py` also holds the abstention phrase
  list, which changes for unrelated reasons): unit tests assert the worked example's 0.6 and literal
  `normalize_text` outputs (`"9:00 AM"` -> `"900 am"`, `"Hip-Hop"` -> `"hiphop"`). Changing the rule
  fails those tests until an amendment below explains it.

## Amendments

None.
