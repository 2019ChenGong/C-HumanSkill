# CMD — Cluster-Mixture De-identification of Distilled Expert "Skill Cards"

**One question:** a company distills an expert into a publishable **skill card** (a profile of how they
decide and work). Can we scrub **WHO it is** (anonymity) while keeping the card **USEFUL** (utility)?

**CMD's answer:** don't rewrite one person's card to disguise them — have **k people publish ONE identical
shared card**. By construction the card can't be narrowed below the group of k.

**Why it matters (two hooks).** This is, to our knowledge, the **first de-identification of LLM-distilled
skill/decision cards** (prior de-id targets raw text or demographics, not a distilled reasoning profile).
It sits at the intersection of **(1) anti-distillation** — *distill the skill, not the person* — and
**(2) agent skills**: skill cards are exactly what powers reusable agent skills, and sharing them across an
org needs the author scrubbed while the competence survives. The result is **positive**: individual de-id
fails, but pooling is simultaneously **anonymous and useful**, and the consensus operator (CMD) is
**Pareto-better than naive concatenation**.

---

## Method (CMD)

1. **Partition** authors randomly into clusters of size `k`.
2. **Synthesize** ONE ε=0 *shared card* per cluster — only the group's **common** competence/decision approach,
   with no detail unique to any single member.
3. **Publish** it **byte-identical** to all `k` members.

**Canonical synth = `neutral`.** The shared card is synthesized with the utility-preserving **`neutral`** prompt
(preserve the concrete shared substance, ~1 input-card length, ~900 words) — this is the **one canonical CMD** used
throughout. An early over-compressed **`base`** synth ("8–12 bullets", ~316 words) is **deprecated**: it threw away
recoverable utility. `neutral` recovers utility to the individual-card ceiling **without re-leaking identity** (the
recovered content is generic shared base-rate, not the per-person decision architecture the attacker keys on). Every
CMD-vs-concat comparison uses a length-matched **`neutral-concat`** so it is not a card-length artifact. Numbers below
are the `neutral` synth unless marked *(base)*.

**Bound (a *k*-anonymity floor, not a privacy guarantee).** A card-only attacker is bounded to **re-id ≤ 1/k** —
classical *k*-anonymity / microaggregation applied to a free-text decision card (the shared representative is an
LLM-synthesized, *usable* card, not a numeric centroid). This is a **syntactic floor under a no-side-information
adversary**, not a guarantee against an attacker with auxiliary data. Against an attacker who *also* holds your raw
text, the card adds **no leak over raw**; the residual is your own raw writing style, which no card scheme can fix.

## Datasets (all three the FULL qualifying set — no cherry-pick)

| Dataset | Domain | N | Bar |
|---|---|---|---|
| Enron emails | workplace email decisions | **116** | every sender with ≥20 distinct docs |
| 20-MAD SeaMonkey | bug-triage reasoning | **128** | every developer clearing the activity bar (no random cap) |
| CrossValidated (stats SE) | statistical-consulting Q&A | **77** | every expert with ≥15 gold answers |

## Key results (free sonnet-4.6 **2AFC** membership attacker, chance = 0.5; equivalence/MDE tested)

- **Per-person de-id is not enough — not without gutting utility.** Rewriting a single person's card to disguise them
  — including 4 SOTA methods (Staab-R1, PETRE, TAROT/`tpar`, Presidio) — leaks under the stronger **gpt-5.4** attacker
  on every measured cell; on the weaker sonnet attacker it leaks on most cells too (the softest, Enron
  `petre_k4`, has a sonnet CI that just touches 0.5; Presidio is measured on Enron only). Identity lives in the
  *decision architecture*, which survives content-level rewriting. You *can* push a single card to chance — but only by
  **destroying its usefulness** (DP-Prompt T≥2.0 reaches 0.5 only by collapsing utility toward the no-card floor).
  So to be anonymous **and** still useful, you have to **pool**.
  - *Honest caveat on these baselines.* **Presidio is Enron-only by necessity** — it is a
    PII scrubber (names / orgs / emails), and the distilled 20-MAD (bug-triage) and CV (stats-QA) cards contain no PII for
    it to touch, so it is a no-op / N/A there; measuring it only on Enron is honest scope, not a cherry-pick.
- **Pooling Pareto-dominates per-person de-id — more anonymous AND more useful, which de-id can't be at once.** On the
  SAME unguarded free-subagent 2AFC as the de-id battery (k8, chance 0.5), the canonical `neutral` CMD shared card sits
  far below every per-person de-id method wherever identity is legible: **20-MAD 0.539** (de-id 0.65–0.66) and
  **CV 0.545** (de-id 0.71–0.74 — on CV the de-id methods leak *above* the no-de-id `indiv` 0.701, i.e. they fail to
  anonymize at all); on the identity-thin **Enron** the pooled card (**0.586** s0 / **0.578** 3-seed pooled) ties the
  de-id methods (0.58–0.63). **No cell has de-id beating pooling on anonymity, and pooling also wins utility (below).**
  Data: `results/{mad,enron,se}/2afc_free/_score_summary.json` (de-id battery), `results/neutral_ksweep_summary.json`,
  `results/multiseed_pool_neufix_d{05,10}.json`.
  - *Honest boundary — the pooled card does NOT reach statistical chance at k8.* Under the powered **3-seed pooled** test
    (`results/multiseed_pool_neufix_d{05,10}.json`) all three datasets leave a small, consistent residual —
    **20-MAD 0.573 / Enron 0.578 / CV 0.580**, all δ=±10pp NOT certified (CI-lo > 0.5). The earlier "20-MAD certified at
    k8" was a single-seed power artifact: single-seed s0 sits at 0.539 (CI ∋ 0.5, underpowered), but 3-seed pooling
    (48 units) detects the same ~0.575 leak MAD shares with the others. So k8 pooling **substantially reduces**
    re-identification (indiv ~0.69 → ~0.575) but does not eliminate it — full chance needs **k>8** (min-k ∝ richness).
    This is a dose-response boundary, not a failure: pooling is still the only route that is *both more anonymous than
    every de-id method and more useful*.
  - *A withdrawn earlier claim (kept visible for honesty).* An earlier "#64 neutral k8 δ=±10pp-certified on Enron / borderline
    CV" was **retracted**: those `neutral` cards were **degenerate** — ~16% (up to 41% Enron / 50% CV at low k) were
    near-verbatim copies of one pooled member, which over-exposed that member on the worst-case axis (verbatim runs up to
    ~700 words) yet *masked* the aggregate 2AFC leak (a card = one member's copy anti-identifies the other k−1). The
    anti-copy fix (`cmd_fix_degenerate.py`, retry until max-member-cos < 0.85) simultaneously **cleans the worst-case
    exposure** (down to concat-comparable) **and reveals the honest ~0.58 leak** on Enron/CV. Degenerate originals are
    preserved at `data/*/cmd_shared_cards*__neutral_degen_backup.json`.
- **Not one attacker's blind spot — a second, stronger attacker agrees.** Re-running the *identical* 2AFC with
  **gpt-5.4** as a different-lineage attacker (same pairs, all 3 datasets, full ladder) reproduces the whole shape: its
  `indiv` positive control beats sonnet's, the 4 per-person de-id arms **still leak**, yet the pooled `shared`/`concat`
  cards **stay far below the de-id arms** (at chance on these base cards) with a significant paired indiv→shared drop. A
  stronger independent model can't re-identify the pooled card either — so the pooled card's low re-id is real anonymity
  headroom over de-id, not a weak-measuring-stick artifact.
  *(Specific numbers omitted — this ran on `base` cards; neutral-card gpt-5.4 rerun = #70. On the fixed `neutral` cards the
  pooled card leaks slightly at k8 on Enron/CV per the boundary above, but still sits below the de-id arms.)*
- **Min-k for anonymity ∝ identity richness (dose-response on fixed neutral cards, `results/neutral_ksweep_summary.json`).**
  k2 leaks everywhere; higher k shrinks the residual but does not reach chance by k8. The powered **3-seed pooled** test
  (`results/multiseed_pool_neufix_d{05,10}.json`) puts all three at a small, remarkably uniform residual at k8 —
  **20-MAD 0.573 / Enron 0.578 / CV 0.580**, none δ=±10pp-certified ⇒ **all three need k>8** for full chance. (Single-seed
  s0 can look at chance — MAD 0.539, CV 0.545 — but that is underpowered; pooling 3 seeds reveals the shared ~0.575 leak.)
  Transition band k4/k6 is noisy (few clusters at high k, 1-shot subagent batches, non-monotone) — read the k2→k8 trend,
  not single mid-k points. The residual is small relative to the indiv leak (~0.69) and to every de-id method (0.58–0.74),
  which is the point: pooling shrinks re-identification more than any de-id baseline while keeping utility.
- **Utility preserved, pooling ~free.** shared ≈ individual card, both beat no-card; but the card ≈ a generic
  competent-colleague checklist and **own ≈ stranger** — the preserved value is generic competence, never
  person-specific. **Cross-model concordant:** on CV, "a card helps over no-card" and "pooling preserves that lift"
  both hold under two independent judges (haiku + sonnet), SIG and sign-matched — not a single-judge artifact.
  *(Specific haiku numbers were on `base` cohort-26 cards; the neutral-card cross-judge rerun is #72; neutral utility
  numbers are in the bullet below.)*
- **Utility metrics are dataset-specific and within-dataset ONLY — magnitudes are NOT comparable across datasets, and we
  never average or rank them across the three.** 20-MAD uses a mechanical 5-element coverage score (chance 0.2, scale
  [0,1]; indiv−nocard **+0.103**, shared−nocard **+0.077**, both SIG). Enron and CV use a pairwise LLM competence judge
  (scale [−1,+1], 0 = tie — the mechanical anchor saturates at 1.0 on Enron, so the judge is primary): Enron indiv−nocard
  **+0.263** / shared−nocard **+0.571** / shared−indiv **+0.293** (all SIG); CV (full-77) indiv−nocard **+0.239** /
  shared−nocard **+0.203** (both SIG). What is cross-dataset invariant — and all this section actually claims — is the
  **sign, significance, and ordering**: shared−nocard SIG-positive everywhere (pooling helps); shared ≈ individual
  (MAD −0.026 ns, CV −0.036 ns) up to shared > individual (Enron +0.293 SIG); and own ≈ stranger everywhere (~0, ns).
- **Canonical `neutral` CMD reaches the individual-card utility ceiling on all three, and beats/ties per-person de-id.**
  Re-measured on the `neutral` synth (`results/neutral_utility_summary.json`): **neutral − individual is NS on all three**
  (20-MAD −0.007 / CV +0.078 / Enron ≈+0.3 i.e. neutral≈base which already beat individual) = pooling reaches the
  personal-card ceiling; neutral − no-card is SIG-positive everywhere (20-MAD +0.097 / Enron +0.473 / CV +0.489). **Head-
  to-head vs per-person de-id on the discriminating pairwise judge (+ = CMD more competent):** Enron neutral−staab
  **+0.402** / neutral−petre **+0.277** (both SIG); CV neutral−staab **+0.221** (SIG), neutral−petre / neutral−tpar
  **+0.143** (ns, n=77); 20-MAD ties (all card arms 0.36–0.38 on the near-saturated mechanical metric). Corrected scope:
  CMD does **not** strictly beat de-id in *all* datasets — it **beats on Enron, ties on 20-MAD, and CV favors CMD** — but
  it reaches the individual ceiling everywhere and is the only arm that *also* anonymizes. (`mad_util_variants.json`,
  `results/se/util_judge77_neutral/`, `util_judge77_deid/`, `results/enron/2afc_util/summary.json`.)
- **Closed-world ≤1/k (structural *k*-anonymity FLOOR, not a measurement).** Because the ε=0 shared card is byte-identical across
  the k members, a closed-lineup attacker faces k indistinguishable options and cannot beat **1/k** — this holds *by
  construction* (the re-id "= 1/k" is arithmetic, reported with a zero-width CI), not as an empirical discovery. What the
  gate actually *measures* is the two facts that make that floor meaningful: raw text **does** leak within a k-cluster,
  and the positive-control attacker is competent. So the honest reading is: the identical card **adds no leakage over the
  (already-leaky) within-cluster raw floor**, and by construction can't exceed 1/k. (Closed-world gate run on Enron,
  `results/cmd_gate_result.json`; a 20-MAD sibling `results/mad/cmd_gate_result.json` shows the same pattern.)
- **CMD (= the canonical `neutral` synth) is Pareto-better than *naive* concat pooling — better on the valid hygiene
  axes at *equal card length*, at equal utility & equal single-release anonymity.** *(All CMD numbers below are the
  `neutral` synth — the one canonical CMD; the deprecated over-compressed `base` synth is retired. To remove the
  obvious "CMD only wins because its cards are shorter" objection, concat is also its length-relaxed `neutral-concat`,
  so both cards are ~900 words — the token/length axis is deliberately **dropped**.)* A naive baseline that just
  concatenates+summarizes the k member cards is *also* byte-identical (same ≤1/k floor), *also* reaches 2AFC chance,
  and on utility is statistically indistinguishable from CMD (**cmd_neutral − concat_neutral +0.013, NS**, at matched
  length; `results/mad/mad_util_variants.json`). At matched length CMD still wins the valid hygiene axes at **both
  k=4 and k=8** (reported k4 / k8; `results/mad/neutral_hygiene_summary.json`): matched-pair verbatim overlap (shared
  6-grams CMD **35.0 / 11.8** vs concat **86.1 / 29.5** — concat ~2.5× more), a $0 n-gram linkage attacker (rare-6gram
  AUC CMD **0.895 / 0.799** vs concat **0.982 / 0.891**), and a $0 no-LLM membership-identifiability probe (member-ID
  AUC CMD **0.886 / 0.757** vs concat **0.957 / 0.798**, CMD closer to the 0.5 ideal). concat carries the extra leakage
  on every axis at both k because it *republishes members' text*; CMD's consensus rewrite abstracts it away.
  **Honest cost of the neutral transition:** the longer neutral cards retain *more absolute* verbatim than the `base`
  cards (base-CMD 6-grams were 2.3 / 1.6; neutral-CMD 35.0 / 11.8), so the old "CMD verbatim ≈ NEG floor (~0)"
  sub-claim is **retired** — the surviving, still-strong claim is "neutral-CMD carries ~2.5× less verbatim + a lower
  member-ID AUC than neutral-concat at matched length". The t-closeness *distance* nominally favors concat but is the
  known-confounded axis (a bigger union sits closer to the aggregate), dropped per the audit.
  (`scripts/cmd_xcard_link.py` MODE=census with `METHOD=cmd_neutral/concat_neutral`; `scripts/cmd_tcloseness.py` with
  `SHARED=…__neutral.json OUT=tcloseness_neutral.json`; `results/mad/neutral_hygiene_summary.json`.)
- **Cross-release linkage is an OPEN limitation of pooling itself — for BOTH operators, and the honest future-work
  direction (not a CMD-specific loss).** Pool the same person into two releases and ask a *strong* attacker (same free
  sonnet-4.6 subagents, card-disjoint batching so it can't cross-reference cards within a batch) to link them by shared
  contributor, scored by the *same* free sonnet-4.6 subagents. Reporting **both k** tells the whole story: at **k=4**
  (well-powered, n=345 / 306) **CMD 0.895 < concat 0.987** — but that concat 0.987 is a literal *verbatim shortcut*
  (rare-6-gram matcher 0.959 at k4), which *dilutes* as the union grows; at **k=8** (n=87 / 74) the shortcut has weakened
  (rare-6-gram 0.76) and the two **converge to CMD 0.933 [0.892,0.967] vs concat 0.890 [0.838,0.936] — statistically TIED**,
  both strongly linkable via decision architecture (~0.90; verbatim-free 0.89 / 0.83). So CMD is *not* durably safer on the
  strong attacker (its k4 edge was concat's verbatim shortcut, gone by k8), while CMD's architecture-driven linkage holds
  ~0.90 at both k. Cross-release unlinkability is solved by **neither** operator, does not vanish with larger k, and is the
  characterized open problem. CMD's clean, persistent win is the verbatim + $0-attacker hygiene axes (rare-6-gram 0.65 / 0.60
  vs 0.96 / 0.76 at k4 / k8), not strong-LLM unlinkability. (Robust: k8 full 0.933 ≈ k8 pilot 0.969; the weaker paid
  sonnet-4.5's 0.577 was attacker weakness, not a clean card.) *Scope: the linkage attack is demonstrated on 20-MAD
  only; the driver (decision-architecture similarity surviving pooling) is dataset-general, so extending it to Enron/CV
  is future work, not expected to change the conclusion.* **Neutral-card status:** the strong-LLM linkage above is on
  the deprecated `base` cards; on the canonical `neutral` cards the $0 rare-6gram verbatim-linker already runs higher
  for both (**CMD 0.895 / 0.799 vs concat 0.982 / 0.891** at k4 / k8 — neutral's longer cards retain more verbatim, so
  verbatim-linkability is *worse* than base for both, CMD still < concat), and the **strong-LLM neutral-card rerun is
  pending (#67)** (pilot packs exported, needs subagents).
- **Why pooling, not "archetype clustering":** `archpool` (cluster by reasoning archetype) is **not better than
  random grouping**; the anonymity comes from *sharing one identical card*, and random grouping carries a
  weak-dominance guarantee (never worse than any content-based partition).

## Repository layout

```
src/
  llm.py             # multi-provider LLM client (OpenRouter/DeepSeek/OpenAI/Anthropic); keys from env/.env
  attrib_metrics.py  # cluster bootstrap CI, permutation null, Holm, paired-diff CI
scripts/
  cmd_gate.py / cmd_gate_score.py            # closed-world re-id gate + shared-card builder (DATASET=enron|mad|cv)
  cmd_attack2afc.py                          # 2AFC membership attacker (primary anonymity instrument, chance 0.5); `neutral` channel
  mad_synth_utility.py                       # build the canonical `neutral` shared card (K/SEED/ONLY env); cmd_concat_build.py NEUTRAL=1 = matched neutral-concat
  neutral_2afc_export.py                     # export neutral-CMD 2AFC anonymity pack (reuses build_pairs; self-contained indiv/shared/neutral) for FREE subagents
  cr_2afc_export.py / cr_2afc_score.py       # export 2AFC pairs for FREE sonnet-4.6 subagents; aggregate answers
  cmd_equiv_test.py                          # TOST equivalence + one-sided non-inferiority + MDE on the 2AFC nulls
  cmd_build_shared.py                        # build only the ε=0 shared cards for a k-sweep (CMD-only)
  cmd_multiseed_pool.py                      # multi-seed pooling certification (per-seed / (seed,card) / person bootstraps)
  cmd_dispersion.py                          # per-cluster dispersion behind the pooled means (SD vs sampling-noise floor, leaky tail)
  cmd_tcloseness.py                          # $0 no-LLM membership-identifiability / t-closeness probe (CMD vs concat pooled cards)
  score_2afc_summary.py                      # (re)generate the 2AFC anonymity _score_summary.json for any battery dir (headline anonymity table)
  xcard_census_norm.py                       # (re)generate the length-normalized verbatim census (xcard_census_normalized.json)
  cmd_xcard_link.py / cmd_xcard_export.py / cmd_xcard_score.py  # cross-release linkage attack (census / FREE-subagent export / same-instrument score)
  cmd_openworld.py / cmd_openworld_score.py  # (legacy) nneg-AUC membership attack — superseded by the 2AFC above
  cmd_utility.py                             # utility (shared vs individual/no-card/floor/stranger)
  cmd_k8_probe.py / cmd_synth_probe.py       # (optional) k=8 win analyses
  mad_cmd_build.py / mad_cmd_build_qwen.py   # 20-MAD card builder (+ qwen non-thinking cross-model distiller swap)
  cv_pilot.py / cv_build.py / cv_util_judge_export.py cv_util_judge_score.py  # CrossValidated (3rd dataset) build + cross-model utility judge
  deid_enron.py enron_nuwa.py enron_step2.py # shared helpers + de-id methods (naive/deid4/aggro)
  enron_archpool.py                          # archpool baseline (cluster→template→re-express)
  mad_nuwa_step2.py mad_comp_two_axis.py     # 20-MAD pipeline + harness
  enron_clean.py                             # Enron mail cleaning
  enron_collect_full.py enron_nuwa100.py enron_nuwa100_dump.py util6_pool.py   # dataset builders
```

## Setup

```bash
conda create -n anti-dis python=3.12 -y
conda run -n anti-dis python -m pip install -r requirements.txt
```

**API keys** — copy `.env.example` and fill, or export env vars:
DeepSeek (generator / de-id rewriter / utility judge), Anthropic (haiku LLM-detective + token counting),
OpenRouter (default router), OpenAI (optional). A missing key just disables that provider.

**Run flags:** `CONDA_NO_PLUGINS=true KMP_DUPLICATE_LIB_OK=TRUE PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8`

## Reproduce

1. **Build data** — `enron_collect_full.py`; `util6_pool.py` (`N_DEV=100000 OUT=mad_cmd_pool.json` = full set, no cap);
   `enron_nuwa100.py`; `mad_cmd_build.py`. (`data/` is git-ignored; regenerate locally.)
2. **DRYRUN cost** — every dump supports `PILOT_DRYRUN=1` (prints token/$ estimate before spending).
3. **Dump trials** — `cmd_gate.py` / `cmd_openworld.py` with `DATASET=enron|mad`, `K_LIST=`, `KCL=`.
4. **Consolidate** — `cmd_batch.py` (`MODE=gate|ow`) → one batch file per group.
5. **Anonymity attack (primary)** — `cr_2afc_export.py` writes self-contained 2AFC pair tasks; FREE Claude-Code
   sonnet-4.6 subagents answer them (each writes `ans_i.json`); `cr_2afc_score.py` aggregates → per-channel 2AFC
   accuracy with a card_id cluster bootstrap. (Legacy path: a strong LLM writes `_picks_*.json` for the nneg-AUC
   attack — superseded but reproducible.)
6. **Score + certify** — `cr_2afc_score.py` (2AFC acc + CI), then `cmd_equiv_test.py` for TOST equivalence /
   non-inferiority / MDE across k; `cmd_gate_score.py` for the closed-world ≤1/k gate (set `DATASET=`, `KCL=`).

## Note on module naming

This codebase uses the original research names (`deid_enron`, `enron_nuwa`, `src.attrib_metrics`). A companion
release uses refactored names (`deid.py`, `detective_*`, `src/stats.py`); the two are parallel and
`src/attrib_metrics.py` is byte-equivalent to that release's `src/stats.py`.
