# gpt-5.4 second-attacker package — FIXED pooled cards (2AFC membership attack) — instructions for Codex

## What this is (read first)

A privacy paper claims a **pooled "skill card"** (built by pooling k people into ONE shared card) hides *who*
contributed. An earlier gpt-5.4 run already showed a strong second attacker cannot re-identify the *base* pooled
cards. Since then the pooled cards were **rebuilt** (a degeneracy — a few pooled cards had copied one member
verbatim — was fixed; the honest fixed cards read slightly higher than the masked base ones). So the second-
attacker question must be **re-asked on the FIXED cards**: does gpt-5.4 *still* fail to re-identify?

**Your job:** run **gpt-5.4 as a second, strong, different-lineage attacker** on the **exact same pairs sonnet-4.6
answered** for the fixed pooled cards (channels `neutral` = the fixed CMD pooled card, and `concat` = a naive
concat-pool baseline), across **3 datasets × 3 random pooling seeds** (multiseed = the certification the paper
uses). If gpt-5.4 lands `neutral` at ~chance (certified anonymous) and reproduces the same `concat` pattern sonnet
found, then "the pooled card looks anonymous only because sonnet is a weak attacker" is dead.

**The positive-control GATE is already passed and is NOT in this package.** gpt-5.4's ability to *read* cards was
established in the base run: its `indiv` (individual-card) accuracy is **.711 / .721 / .655** (MAD / Enron / CV)
≥ sonnet and CI > 0.5. Those `indiv` cards are **per-person and were NOT changed** by the fix (and do not depend on
the pooling seed), so that gate transfers here unchanged. That is why a `neutral` ~chance result *here* means the
card is anonymous, not that the attacker is weak. (`sonnet_baseline.json` records this.)

**Orchestration model:** *you (Codex) are the orchestrator; you do NOT answer batches yourself.* **Dispatch a
gpt-5.4 SUB-AGENT per batch** (each sub-agent's model = gpt-5.4) to do the answering; you collect its answers and run
the scorer. Keep the model **fixed to gpt-5.4** for every sub-agent, across every batch, dataset, and seed.

## Two hard rules — do not break either

1. **NEVER show `meta.json` to the gpt-5.4 worker.** `meta.json` is the ANSWER KEY (it holds `member_slot`). The
   worker sees ONLY `sys.txt` (its role) and one `batch_i.json` (the prompts). Leaking `meta.json` invalidates the run.
2. **A worker does its whole batch ITSELF, inline. It must NOT spawn / delegate to any further sub-agent or helper.**
   One worker = one batch = one flat pass. (Nested sub-agents have silently mis-answered before; forbid them.)

## Throughput rule (Codex agent cap + rate-limits)

**Run at most 6 gpt-5.4 sub-agents concurrently** (Codex's max-agent limit). Process batches in waves of ≤6; start
the next wave only after the current wave's sub-agents have written their `ans_i.json`. Total load: **99 batches /
3,852 pairs** (MAD 39, CV 24, Enron 36) → ~17 waves (MAD ~7, CV ~4, Enron ~6).

## Layout

```
gpt54_2afc_fixed_pkg/
  README_FOR_CODEX.md        <- this file
  sonnet_baseline.json       <- sonnet's A1 multiseed numbers gpt-5.4 must reproduce + the gate + expected verdicts
  score_gpt54_fixed.py       <- scorer (numpy only); run after answers are in
  tasks/
    mad/    sys.txt   s0/{batch_0..12.json, meta.json}   s1/{...}   s2/{...}      [neutral+concat, 512 pairs/seed]
    cv/     sys.txt   s0/{batch_0..7.json,  meta.json}   s1/{...}   s2/{...}      [neutral+concat, 308 pairs/seed]
    enron/  sys.txt   s0/{batch_0..11.json, meta.json}   s1/{...}   s2/{...}      [neutral+concat, 464 pairs/seed]
```

Each `batch_i.json` is a list of `{"pid","prompt"}`. Each `prompt` is self-contained: a SKILL CARD (the pooled card)
+ two people's decision signatures `[A]` / `[B]`; exactly one of A/B is a true contributor to that card, the other a
same-topic non-member. `sys.txt` (one per dataset) is the system role. The channel and the answer key live only in
`meta.json` (hidden from the worker). Pairs are **byte-identical to what sonnet-4.6 answered** in A1, so the numbers
are directly comparable to `sonnet_baseline.json`.

## Answer format the worker must produce

For every task in its batch, one record:

```json
{"pid": "P0007", "choice": "A", "conf": 72}
```

- `choice` = "A" or "B" — which person contributed, judged ONLY by reasoning / decision style (ignore shared topic).
- `conf` = integer 50–100 (50 = pure guess, 100 = certain).

Write the batch's records as a JSON array to `ans_<i>.json` **in the same seed folder** as the `batch_<i>.json` it
answered. e.g. answers to `tasks/mad/s1/batch_3.json` → `tasks/mad/s1/ans_3.json`.
(The per-item prompt text ends with a legacy "answer one line `A 72`" instruction — ignore that wording; produce the
JSON array described here. The scorer reads only `choice`.)

## Worker prompt (hand this, plus the folder's `sys.txt`, to each gpt-5.4 sub-agent)

> You are the forensic authorship analyst described in the system role. You are given a JSON list of tasks; each task
> has a `pid` and a `prompt`. For EACH task, read its `prompt` — it shows a SKILL CARD and two people's decision
> signatures `[A]` and `[B]`; exactly one of A/B contributed to the card. Decide which — A or B — judged ONLY by
> reasoning / decision style and idiolect, IGNORING shared topic. Give a confidence 50–100. Answer EVERY task; do not
> skip any. Output ONLY a JSON array of `{"pid","choice","conf"}` (choice ∈ {A,B}, conf ∈ 50..100), one object per
> task, and nothing else. Do the entire batch YOURSELF, inline. Do NOT spawn or call any sub-agent, worker, or
> tool-agent; do NOT split the batch across helpers. Just read and answer.

## Step-by-step

**Do MAD first** — it is the make-or-break gate (sonnet's fixed `neutral` is lowest-variance there).

1. In waves of ≤6: for each `tasks/mad/s{0,1,2}/batch_i.json`, dispatch one gpt-5.4 sub-agent (system = the dataset's
   `tasks/mad/sys.txt`, user = the batch list + the worker prompt above). Save its JSON array to the same folder as
   `ans_i.json`. Do all three MAD seeds (s0, s1, s2 = 39 batches).
2. When all MAD batches are answered, run:  `python gpt54_2afc_fixed_pkg/score_gpt54_fixed.py`
   It prints, per dataset and channel, gpt-5.4's multiseed accuracy + certification beside sonnet's A1 number, and
   whether gpt-5.4 REPRODUCES the ladder. It writes `results_gpt54_fixed_summary.json`.
3. **Read the MAD result.** If gpt-5.4 `neutral` is CERTIFIED ANON (matches sonnet), do `cv/` (24 batches) then
   `enron/` (36 batches) the same way, and re-run the scorer.

## How to read the result

- The GATE is cited (gpt-5.4 reads individual cards fine: indiv .711/.721/.655 ≥ sonnet). So a `neutral` ~chance is
  meaningful.
- **The win (expected):** gpt-5.4 reproduces sonnet's A1 ladder on the FIXED cards —
  - `neutral` (fixed CMD): **CERTIFIED ANON on all 3** (both poolers up95 < 0.60, no leak) — sonnet was .479/.532/.535.
  - `concat` (naive pool): anon on **MAD & Enron**, and **LEAKS on CV** (same as sonnet .574) — CV concat is a real,
    honestly-reported weak spot of the *naive* pool that the CMD card does not share.
  A strong, independent-lineage attacker agreeing kills "sonnet is just weak."
- **If gpt-5.4 `neutral` LEAKS (CI-lo > 0.5 or up95 ≥ 0.60):** gpt-5.4 cracks the fixed pooled card → a real anonymity
  problem. Report it honestly; do not hide it.

## Notes

- No repo internals are needed to answer or score — task files are self-contained and `score_gpt54_fixed.py` uses only
  numpy + stdlib. Answers can arrive partially; the scorer reports per-seed coverage and only certifies a fully-answered
  channel.
- "Multiseed" = 3 random pooling partitions of the same people; the scorer pools them two ways (by (seed,cluster) and by
  person) and certifies a channel only if BOTH exclude a ≥0.60 leak — identical to the paper's `cmd_multiseed_pool.py`.
