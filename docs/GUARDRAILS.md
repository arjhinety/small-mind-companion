# Guardrails

Rules for future passes, each derived from a mistake this project actually made. The evidence is
in [`reports/audits/001-claim-audit/findings.json`](../reports/audits/001-claim-audit/findings.json);
the corrections are in [`reports/ERRATA.md`](../reports/ERRATA.md). Finding ids below are from that
ledger.

A rule is only useful if it is applied *before* the mistake. The two checklists at the end are part
of the process: one before results are written up, one before anything is published.

## Attribution

**G1. A result's label comes from the script that produced it, never from the write-up.** Findings
#1 and #2 — the highest-severity items in the audit — both came from trusting prose over code. The
70.0% UAR was credited to the DPO checkpoint for months because a table said "SFT+DPO+memory",
while `run_system_e_v1.py:35` loaded `outputs/sft/v1/merged` and named the system
`E_sft_memory_v1`. The one-line check that would have caught it is reading `HFEngine(...)` in the
generating script.
*Check:* for every number in a write-up, name the script and the checkpoint path that produced it.
If the artifact filename and the prose label disagree, the filename wins.

**G2. A two-stage comparison must isolate one stage.** Finding #2: distillation was credited with a
+3.3pp gain measured between `outputs/sft/v1/merged` and `outputs/distill/v1/merged` — but the
distilled model was trained *from the DPO output*, so the delta included DPO. The isolating
measurement was the pairwise run that held DPO constant on both sides.
*Check:* for any "X improved Y by Z" claim, list every stage that differs between the two systems.
If more than one differs, report it as a combined effect or run the isolating comparison.

**G3. Absence of a measurement is stated as absence.** There is no full-PMB `pra_lenient`/UAR
measurement for any DPO checkpoint. A blank cell in a results table reads as "not yet run", but a
borrowed number reads as a result.
*Check:* an em dash for a metric that was never computed, plus one sentence saying so, and never a
number carried over from a neighbouring row.

## Counts and state

**G4. Counts are generated, not typed.** Every stale count in this audit (#3, #14, #17, #21, #27,
#30) was a hand-written number that drifted: 40 personas vs 8, 451 tests vs 473, 21 bugs vs 27,
"15k lines" vs 66,383, two GGUF size sets in one file, and 2242 SFT examples vs the committed 2480.
*Check:* where a count can be derived from an artifact, generate it or say when it was last
verified. Datasheets are generated — fix the generator, not the generated file (finding #19 was
caused by a hardcoded title in `scripts/build_pmb.py`). **When a dataset is regenerated, grep for
its old size and row count** — the number changes in the data, not in the prose that quotes it.

**G5. "Done" and "not done" live in one place.** Finding #8: `docs/research_questions.md` still
described imatrix quantization as an undone next step while `docs/quantization_results.md`
documented it as completed. Finding #25: three files called `dpo-v1-scale` current-best while four
others named `distill-v1`.
*Check:* a project-state claim appears in exactly one document and is linked from anywhere else.
When state changes, the stale claims are found by grepping the old state's name.

**G6. Configuration claims are verified per file, not per pattern.** Finding #4: "base model
revision is pinned by commit SHA in each config" was true for 2 of 6 configs; the other four used
`"main"`.
*Check:* a claim about "all configs" is backed by a command that reads all of them.

## Evidence

**G7. A number with no committed artifact is labelled as reported, not measured.** Finding #6:
every quantization figure — throughput, perplexity, file sizes, parameter count — is restated from
run logs that are not in the repository, and `*.gguf` is gitignored so nothing can be recomputed.
*Check:* the docs carry an evidence note naming what is not reproducible from a fresh clone.

**G8. A referenced artifact must exist.** Findings #16 and #26: `make figures` was part of the
stated reproducibility chain and `scripts/make_figures.py` never existed; a results file was
referred to as "if it exists". A command in a reproducibility chain that cannot run invalidates the
claim.
*Check:* every path and command in a doc is executed or resolved during the same pass that writes
it.

**G9. A hash is recomputed from a clean checkout before it is trusted.** This rule was written
wrong the first time. It originally said "all six `hash.txt` files verify, once CRLF is normalised
to LF". Four of them do — the SFT/DPO datasets — and **four do not, under any algorithm**, which
was only established by testing them against a clean checkout of the commit that introduced them
(finding #29). The original claim came from a partial check that was generalised to all six.
*Check:* recompute every hash, from a clean checkout, before writing that they verify. Record the
algorithm and the line-ending assumption next to the hash; this repo had two undocumented and
incompatible algorithms in use. Where a hash cannot be reproduced, name the artifact that
supersedes it rather than leaving a broken pin in place, and remove the claim that it is a pin.

**G9b. A partial verification is not a verification.** The specific failure above: four hashes were
confirmed and the sentence was written about six. Nothing about the four that passed said anything
about the other two.
*Check:* a claim about "all N" is backed by a check that covered N, and the check's coverage is
stated.

**G9c. An unverified hash is not a pin, and it must not be cited as one.** Distinct from G9, which
is about how to *check* a hash; this is about what may be *claimed* from one. `README.md` described
the datasets as "hash-pinned" while four of the nine hashes matched their corpora under no tested
algorithm — a pin nobody had ever turned the key on. The remedy is two-part: verify it, or say
plainly that it does not verify and name the artifact that supersedes it (here,
`reports/data/study-001-freeze.json`).
*Check:* before the word "pinned" or "hash-verified" appears in a document, every hash it refers to
has been recomputed and the claim narrowed to the ones that passed. `scripts/recompute_hashes.py`
and `tests/unit/test_dataset_hashes.py` are this rule's enforcement.
*Numbering note:* G9b and G9c were both derived from the same finding (E29), so they are lettered
off G9 rather than appended at the end.

## Public surfaces

**G10. Published cards are corrected by re-push, and errata in the meantime.** Six model cards were
already on the Hub carrying the misattributed figure. The repository cannot fix a published card by
committing.
*Check:* after correcting a card source, re-push; if a number is already public and wrong, add it to
`reports/ERRATA.md` so the repository's own record supersedes it.

**G11. Contradictory status text is worse than no status text.** Finding #11: the SFT datasheets
said contamination was unchecked, the distill datasheet said it was checked clean, and it cited the
SFT datasheet as its source. The data was clean the whole time; only the documentation was wrong.
*Check:* a claim that cites another file must match that file. Where two files state the same fact,
generate one from the other or state it once.

**G12. A field documented as populated must be populated.** Finding #22: the README said probes
ship with "acceptable alternatives"; the field is empty in 688/688, which is precisely why
`pra_strict` is ~0 and `pra_lenient` is the reported metric.
*Check:* every documented schema field is either populated or documented as unpopulated with the
consequence spelled out.

## Before writing up results

1. For each number: which script produced it, and which checkpoint did that script load?
2. Does any comparison differ in more than one training stage?
3. Is every metric in the table actually measured, or borrowed from a neighbouring row?
4. Was every count regenerated from an artifact rather than typed?
5. Does any referenced file or command not exist?
6. Does any other document contradict this one about project state?

## Before publishing anything

1. Re-run `scripts/check_contamination.py` and record the output.
2. Recompute every `hash.txt`; confirm line-ending handling.
3. Re-resolve every link and path in the published document.
4. Confirm the licence statement matches the licence files.
5. If a number was previously public and is now wrong, add it to `reports/ERRATA.md`.
6. Confirm the document's headline table is the *current* one, not a superseded run.

## Known limitations

- The evaluation numbers themselves are sound: all seven runs recomputed from their own saved
  responses match `metrics.json` exactly. This audit found defects in the prose and metadata around
  them, not in the measurements.
- Every claim in this project rests on **a single seed and a single run**. No result here is
  replicated.
- No human evaluation exists. Every dataset is marked not human-reviewed, and no reviewer log or
  teacher transcript was retained — so the `gpt-5.6-luna` teacher identity is self-reported and
  unverifiable from the repository.
