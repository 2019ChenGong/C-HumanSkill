# gpt-5.4 second-attacker package (2AFC membership attack, FULL LADDER) — instructions for Codex

## What this is (read first)

A privacy paper claims a **pooled "skill card"** hides *who* contributed. Evidence so far rests on **one** attacker
(a sonnet-4.6 model doing a 2-alternative forced choice). A reviewer's #1 objection: *"the pooled card sitting at
chance (~0.5) might just mean that one attacker is weak, not that the card is anonymous."*

**Your job:** run **gpt-5.4 as a SECOND, stronger, different-lineage attacker** on the **exact same task and the exact
same pairs** sonnet answered. If gpt-5.4 **aces the positive control** (`indiv` cards, which SHOULD be re-identifiable),
**still leaks on per-person de-id cards** (`staab/petre_k4/presidio/tpar_t15`), **but cannot beat chance on the pooled
cards** (`shared`, `concat`), then "the attacker is just weak" is dead — a strong independent model reproduces the whole
ladder and agrees the pooled card is anonymous. That is the point.

**Orchestration model (important):** *you (Codex) are the orchestrator.* **Dispatch a gpt-5.4 sub-agent per batch** to
do the answering; you collect its answers and run the scorer. Use the strongest gpt-5.4 you have, and keep the model
**fixed to gpt-5.4** across every batch and dataset.

## Two hard rules — do not break either

1. **NEVER show `meta.json` to the gpt-5.4 worker.** `meta.json` is the ANSWER KEY (it contains `member_slot`). The
   worker sees ONLY `sys.txt` (its role) and one `batch_i.json` (the prompts). Leaking `meta.json` invalidates everything.
2. **A gpt-5.4 worker must do its whole batch ITSELF, inline. It must NOT spawn / dispatch / delegate to any further
   sub-agent, tool-agent, or helper.** One worker = one batch = one flat pass. (Nested sub-agents have silently
   mis-answered before; forbid them explicitly in the worker prompt.)

## Throughput rule (avoid rate-limits)

**Run at most 9 gpt-5.4 workers concurrently.** Process batches in waves of ≤9; start the next wave only after the
current wave's workers have written their `ans_i.json`. Total load: **92 batches** (MAD 36 + CV 20 + Enron 36),
~4,084 pairs → ~11 waves.

## Layout

```
gpt54_2afc_pkg/
  README_FOR_CODEX.md      <- this file
  sonnet_baseline.json     <- sonnet's per-channel numbers gpt-5.4 is compared against + the gate rule
  score_gpt54.py           <- scorer (numpy only); run after answers are in
  tasks/
    mad/    batch_0..35.json , sys.txt , meta.json(ANSWER KEY - never show worker)   [6 channels, 1536 pairs]
    cv/     batch_0..19.json , sys.txt , meta.json                                   [6 channels,  924 pairs]
    enron/  batch_0..35.json , sys.txt , meta.json                                   [7 channels, 1624 pairs]
```

Each `batch_i.json` is a list of `{"pid","prompt"}`. Each `prompt` is self-contained: a SKILL CARD (or an individual
card, or a de-id'd card) + two 250-char writing samples `[A]`/`[B]`; exactly one of A/B is a true contributor, the other
a same-topic stranger. `sys.txt` is the system role. The channel each pair belongs to lives only in `meta.json` (hidden
from the worker); the worker just answers what it sees.

Pairs are **byte-identical to what sonnet-4.6 answered** (k8; MAD/Enron seed 1, CV seed 0; same-topic negatives) — so
gpt-5.4's numbers are directly comparable to `sonnet_baseline.json`.

## Answer format the worker must produce

For every task in its batch, one record:

```json
{"pid": "P0007", "choice": "A", "conf": 72}
```

- `choice` = "A" or "B" — which writing sample contributed, judged ONLY by reasoning / decision style (ignore topic).
- `conf` = integer 50–100 (50 = pure guess, 100 = certain).

Write the batch's records as a JSON array to `tasks/<ds>/ans_<i>.json` (same folder, same index as the `batch_<i>.json`
it answered). e.g. answers to `tasks/mad/batch_3.json` → `tasks/mad/ans_3.json`.

## Worker prompt (hand this, plus `sys.txt`, to each gpt-5.4 sub-agent)

> You are the forensic authorship analyst described in the system role. You are given a JSON list of tasks; each task
> has a `pid` and a `prompt`. For EACH task, read its `prompt` — it shows a SKILL CARD (or a writing sample) and two
> writing samples `[A]` and `[B]` from two different people; exactly one of A/B contributed to / matches the card. Decide
> which — A or B — judged ONLY by reasoning / decision style and idiolect, IGNORING shared topic. Give a confidence
> 50–100. Answer EVERY task; do not skip any. Output ONLY a JSON array of `{"pid","choice","conf"}` (choice ∈ {A,B},
> conf ∈ 50..100), one object per task, and nothing else.
> Do the entire batch YOURSELF, inline. Do NOT spawn or call any sub-agent, worker, or tool-agent; do NOT split the
> batch across helpers. Just read and answer.

## Step-by-step

**Do MAD first** — it is the make-or-break gate (sonnet's positive control is strongest there, indiv 0.691).

1. In waves of ≤9: for each `tasks/mad/batch_i.json`, dispatch one gpt-5.4 worker (system = `tasks/mad/sys.txt`, user =
   the batch list + the worker prompt above). Save its JSON array to `tasks/mad/ans_i.json`.
2. When all 36 MAD batches are answered, run:
   ```
   python gpt54_2afc_pkg/score_gpt54.py
   ```
   It prints, per channel, gpt-5.4 vs sonnet, the **GATE**, the `shared` verdict, and the **paired indiv→shared drop**,
   and writes `results_gpt54_summary.json`.
3. **Read the GATE.** If it PASSES on MAD, do `cv/` (20 batches) then `enron/` (36 batches) the same way. (Enron is
   important: sonnet's indiv there is only 0.608 — the paper's softest spot — so gpt-5.4 acing indiv on Enron repairs it.)

## How to read the result (the gate is everything)

- **GATE PASS** = gpt-5.4 `indiv` acc ≥ sonnet's indiv on that dataset **and** its 95% CI excludes 0.5 ⇒ gpt-5.4 is a
  strong card-reading attacker; only then is `shared` interpretable.
  - **GATE PASS, and `shared`+`concat` ≈ chance (CI includes 0.5), and the de-id arms (staab/petre/presidio/tpar) still
    LEAK (CI>0.5)** → **the win**: a second, stronger, different-lineage attacker reproduces the entire ladder — it reads
    individual and per-person-de-id cards but cannot re-identify the pooled card. "sonnet is just weak" is dead.
  - **GATE PASS but `shared` LEAKS (CI>0.5)** → gpt-5.4 *does* crack the pooled card → a real anonymity problem; report
    it honestly, do not hide it.
- **GATE FAIL** = gpt-5.4 `indiv` weaker than sonnet's → this run doesn't strengthen the claim; report honestly, don't
  spin a FAIL as anonymity.

The headline number is the **paired indiv→shared drop** (same clusters): a large, significant drop = "the attacker reads
the individual card fine, then loses the person once the card is pooled."

## Notes

- No repo internals are needed to answer or score — task files are self-contained and `score_gpt54.py` uses only numpy.
- The scorer tolerates partial answers (it reports per-channel coverage), but trust a dataset's gate only once all its
  `indiv` and `shared` pairs are answered.
