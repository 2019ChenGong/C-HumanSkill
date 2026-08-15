# CMD — Cluster-Mixture De-identification of Distilled Expert "Skill Cards"

**The question.** A company distills an expert into a publishable **skill card** (a profile of how they
decide and work). Can we scrub **who it is** while keeping the card **useful**?

**The answer.** Don't rewrite one person's card to disguise them. Have **k people publish ONE identical
shared card**, then run a **per-line minimal-edit sanitize (V6)** that removes every ≥6-word verbatim run
from member text. The card can't be narrowed below the group of k, and it ships with a script-checkable
**0% verbatim certificate**.

To our knowledge this is the first de-identification of **LLM-distilled decision cards** (prior work
de-identifies raw text or demographics). Individual de-id fails on both axes; pooling + sanitize is
simultaneously anonymous and useful.

Full method and all numbers: **`docs/V6_METHOD_AND_DATA.md`**. Frozen designs: **`docs/ELEMK_DESIGN.md`**.

---

## How the experiment actually runs

Four stages. Each stage names the scripts it runs and the files it writes, so the code and the experiment
can be checked against each other. `results/` is regenerated locally and is **not** published.

### Stage 0 — build the datasets

Three datasets, each the **full qualifying set** (no sampling, no cap):

| Dataset | Domain | N | Bar |
|---|---|---|---|
| Enron emails | workplace email decisions | 116 | every sender with ≥20 distinct docs |
| 20-MAD SeaMonkey | bug-triage reasoning | 128 | every developer clearing the activity bar |
| CrossValidated | statistical-consulting Q&A | 77 | every expert with ≥15 gold answers |

`enron_collect_full.py` · `enron_nuwa100.py` · `util6_pool.py` · `mad_cmd_build.py` · `cv_build.py`

### Stage 1 — build the card

1. **Partition** authors randomly into clusters of size k (k=8 convention).
2. **Pool** — one shared card per cluster, only the group's common competence, nothing unique to any member.
   → `mad_synth_utility.py`
3. **Degeneracy fix** — re-distill any card that came out as a near-copy of one member.
   → `cmd_fix_degenerate.py`
4. **V6 sanitize** — per line: clean lines ship verbatim; lines carrying a ≥6-word run from member text get
   the smallest edit that breaks exactly those runs; lines where that is unsolvable are rewritten whole.
   **Five deterministic gates arbitrate every draft** (lexical re-check · numbers preserved · embedding
   fidelity · punt-status · length). The LLM only drafts — **every verdict is script-side.**
   → `v5_sanitize.py EDIT=min STAGE=build`
5. **Certificate** — recheck that 0% of card lines share any 6-word run with any member's text.
   → `elemk_v3_gates.py MODE=lex CARDS=v6min`

**Ships:** `data/*/cmd_shared_cards*__v6min{,_stats}.json` — committed, so the published artifact can be
inspected without rebuilding. **Never ships:** the per-line audit sidecar (`*_audit.json`), which maps card
lines back to member text and carries the same access level as raw member data.

Lines that can't pass the gates in 4 retries are dropped and counted. ≤10% auto-admits; 10–25% needs an
ablation-matched control and disclosure; >25% voids. This is an engineering admission line, **not** a
derived safety threshold (`docs/DROPGATE_DECISION.md`).

### Stage 2 — measure anonymity: the NAMING RULER (primary instrument)

A knockout re-identification tournament. The attacker sees one card plus a lettered list of writing samples
by different people, and must name which sample is by a contributor; survivors advance.

1. **Export** self-contained question packs — one positive-control arm (`indiv`, the raw individual card,
   which *must* leak or the wave is void) plus the arms under test. → `naming_export.py`
   (candidate-sample source is pinned by `naming_refsrc.py` / `cv_t2_verify.py`)
2. **Answer** — free Claude-Code sonnet subagents, one batch each, written to `ans_<i>.json`.
3. **Coverage gate** — `python scripts/naming_cov.py`, **no arguments**. Checks every answer file against
   its batch byte for byte. A subagent's self-reported count is never used; a bad batch is voided whole and
   re-dispatched from a frozen verbatim template, never edited into passing.
4. **Read out** — `naming_depth.py` (depth-excess, primary) · `naming_auc.py` (AUC) per directory, then
   `naming_auc_pooled.py` / `naming_rank_at_k.py` / `naming_pooled_depth.py` pooled across datasets.
5. **Verdict** — each experiment has a pre-registered verdict script that only re-scores existing answers:
   `naming_a12_verdict.py` · `naming_a3spot_verdict.py` · `naming_mseed_verdict.py`.

Everything is pre-registered before any run, including the conditions that would go against us. The earlier
pairwise **2AFC** instrument is **frozen** — kept as the historical record, not extended; where the two
disagree the naming ruler is the main criterion.

### Stage 3 — measure utility: FORCED CHOICE (FC)

One real work unit (a bug report / a stats question / an email task). The same downstream model drafts once
per arm; a blind judge must pick the better draft — **no abstention**. Win rate is the readout, null = .5.

1. **Export + draft** both arms per unit, plus a **placebo battery** that voids the whole wave if it fails
   (`pad` and `fmt` must show no preference · `cut` must show dose-response · `self` must land ≈.5).
   → `mad_fc_export.py` / `cv_fc_export.py` / `enron_fc_export.py`
2. **Judge** — blind subagents, both presentation orders, length-matched drafts.
3. **Coverage check** — `fc_status.py` (again: never the judge's own count).
4. **Verdict** — `cv_fc_score.py`, using a pre-registered δ=.10 dictionary: **SIG / certified-TIE /
   UNDERPOWERED+sMDE**. Cluster bootstrap, because a cluster shares one card and its members are not
   independent samples. Multi-wave pooling: `fc_multiseed_pool.py`. Cross-judge swap:
   `mad_fc_judge_qwen.py`.

"UNDERPOWERED" is a real verdict here — a failure to detect a difference is never reported as a tie.

---

## What came out

| | result |
|---|---|
| **Anonymity, ladder at equal n** | ship it raw → **10.0×** chance re-identification; wash it first with a SOTA per-person de-id method → **7.7–10.0×** (all paired diffs vs raw contain 0). CMD's sanitized card: **AUC .509, certified at δ=.10 and δ=.05** |
| **Anonymity, robustness** | certification holds on **3/3** independent card-build partitions (worst upper bound .5307); the residual is detected in **2/3** — it sits at the instrument's detection edge |
| **Utility** | sanitize costs nothing: ne−v6 **certified TIE on all 3 datasets**; pooling beats per-person de-id **SIG on all 3**; per-person de-id carries a **utility tax**; a card's value is domain-general — own card ≈ stranger's card everywhere |
| **Certificate** | 0.0% verbatim on all 3 datasets, vs 80–91% for the un-sanitized pooled card |
| **In progress** | k-gradient robustness sidebar (#152), k∈{4,6,8,10,12} — k12/k10/k6 closed, k4 at 242/455 batches. Headline claims do not depend on it. `results/K152_WAVE_LEDGER.md` |

Numbers, confidence intervals, per-dataset breakdowns and the full robustness matrix: `docs/V6_METHOD_AND_DATA.md`.

## Honest limits

- **≤1/k is a structural floor, not a privacy guarantee.** It holds by construction against a card-only
  adversary and says nothing about an adversary holding auxiliary data. Against someone who already has your
  raw text, the card adds no leak over raw — the residual is your own writing style.
- **"CMD is more anonymous than naive concatenation" is a 2AFC-only claim.** On the primary instrument the
  two are not separable and the point estimate runs the other way. What does hold on a main criterion is the
  verbatim channel and the certificate.
- **Cross-release linkage is unsolved by both operators** and does not vanish with larger k — a strong LLM
  links via decision architecture, not verbatim. This is the characterized open direction.
- **The "deepseek is the only bare-plug rewriter" claim is suspended** pending drop-rate-matched re-tests
  (`docs/DROPGATE_DECISION.md` §6).
- **Utility magnitudes are within-dataset only** — never averaged or ranked across datasets.
- **Threat model:** the deployment assumption is that the card-building corpus stays **private**, so the
  realistic adversary is corpus-blind. The corpus-holding adversary is a stress test (`results/THREAT_MODEL.md`).

## Repository layout

```
src/         llm.py (multi-provider client)  ·  attrib_metrics.py (cluster bootstrap, permutation null, Holm)

docs/        V6_METHOD_AND_DATA.md   # the method step by step + all data
             ELEMK_DESIGN.md         # frozen pre-registered designs
             DROPGATE_DECISION.md    # the drop-gate ruling + its replacement disciplines

data/        the committed V6 release cards + build-stat counters (the only tracked data)
             *_audit.json sidecars are hard-denied repo-wide and never ship

scripts/     build chain     mad_synth_utility · cmd_fix_degenerate · v5_sanitize · elemk_v3_gates
             naming ruler    naming_export/refsrc/cov/score/depth/auc/pooled_*/rank_at_k/pooled_gate
                             + naming_{a12,a3spot,mseed}_verdict · k152_{power,build_verify,gate_readout}
             forced choice   {mad,cv,enron}_fc_export · cv_fc_score · fc_status · fc_multiseed_pool
                             · mad_fc_judge_qwen
             verbatim/mech   vmia_verbatim · dropgate_{verdict,ablate} · cmd_dispersion · cmd_tcloseness
                             · cmd_xcard_* · petre_noop_census · cmd_gate · cmd_equiv_test
             2AFC (frozen)   neutral_2afc_export · cr_2afc_* · score_2afc_summary · r6_2afc_certify
             R-series        r9_rebuild_check · r13_* · r7_* · r6e_retry_probe
             builders/base   cv_build · cv_pilot · deid_enron · enron_* · mad_* · cmd_{build_shared,concat_build}
             superseded      cmd_utility · mad_utility · cv_util_judge_* · cmd_attack2afc* · cmd_openworld*
```

## Setup and reproduce

```bash
conda create -n anti-dis python=3.12 -y
conda run -n anti-dis python -m pip install -r requirements.txt
```

Copy `.env.example` and fill in keys: DeepSeek (card synthesis, sanitize rewriting, downstream drafting),
OpenAI (embeddings for the fidelity gate), OpenRouter (default router), Anthropic (token counting). A
missing key just disables that provider.

Run flags: `CONDA_NO_PLUGINS=true KMP_DUPLICATE_LIB_OK=TRUE PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8`

Then follow Stages 0→3 above in order. Every dump supports `PILOT_DRYRUN=1` / `COST=1`, which prints a
token and dollar estimate before spending anything. Raw datasets are git-ignored and rebuilt locally; the
**release cards are committed**, so the certificate can be re-verified against the shipped artifact.

## Artifact vs deployment

This repo is a **research reproduction artifact**, not a deployment — worth stating plainly, because two
things in it look like contradictions otherwise.

- **It ships member text; a deployment would not.** `gpt54_2afc_fixed_pkg/tasks/*/*/batch_*.json` (99 files,
  ~28.5 MB) contains the attack prompts verbatim, including the raw writing samples the cards were distilled
  from. That is deliberate: it makes the cited second-attacker certification independently re-runnable, and
  the underlying corpora are already public (Mozilla/Apache bug trackers, the Enron corpus, CrossValidated).
  A reader of this repo is by construction the corpus-holding adversary — the stress test, not the
  deployment scenario.
- **`results/` is not published.** It holds raw run outputs, per-run verdict documents and working notes;
  the scripts above regenerate it. Any `results/…` pointer in this README or in `docs/` refers to a locally
  regenerated file, not a shipped one.

## Note on module naming

This codebase uses the original research names (`deid_enron`, `enron_nuwa`, `src.attrib_metrics`). A
companion release uses refactored names (`deid.py`, `detective_*`, `src/stats.py`); the two are parallel and
`src/attrib_metrics.py` is byte-equivalent to that release's `src/stats.py`.
