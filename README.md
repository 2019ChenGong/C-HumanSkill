# CMD — Cluster-Mixture De-identification of Distilled Expert "Skill Cards"

**One question:** a company distills an expert into a publishable **skill card** (a profile of how they
decide and work). Can we scrub **WHO it is** (anonymity) while keeping the card **USEFUL** (utility)?

**CMD's answer:** don't rewrite one person's card to disguise them — have **k people publish ONE identical
shared card**, then pass it through a **per-line minimal-edit sanitize (V6)** that removes every ≥6-word
verbatim run from member text under five deterministic gates. By construction the card can't be narrowed
below the group of k, and it ships with a third-party-checkable **0% verbatim certificate**.

**Why it matters (two hooks).** This is, to our knowledge, the **first de-identification of LLM-distilled
skill/decision cards** (prior de-id targets raw text or demographics, not a distilled reasoning profile).
It sits at the intersection of **(1) anti-distillation** — *distill the skill, not the person* — and
**(2) agent skills**: skill cards are exactly what powers reusable agent skills, and sharing them across an
org needs the author scrubbed while the competence survives. The result is **positive**: individual de-id
fails, but pooling is simultaneously **anonymous and useful**, and the consensus operator (CMD) is
**Pareto-better than naive concatenation**.

---

## Method (CMD := pool + V6 sanitize; canonical as of 2026-07-18)

1. **Partition** authors randomly into clusters of size `k` (k=8 convention; k is a near-free knob, see below).
2. **Synthesize** ONE ε=0 *shared card* per cluster — only the group's **common** competence/decision approach,
   with no detail unique to any single member (utility-preserving `neutral` prompt + degeneracy fix = `neutral_fixed`).
3. **Sanitize (V6)**: per-line minimal edit of the pooled card — clean lines ship verbatim; lines containing a
   ≥6-consecutive-word run from any member's text get the smallest edit that breaks exactly those runs; lines
   where that is mathematically unsolvable (>90% coverage) are rewritten whole. Five deterministic gates arbitrate
   every draft (full-line lexical re-check · numeric preservation · embedding fidelity ≥.90/.75 · punt-status
   regex · length ±30%); a per-line provenance audit is kept **internal** (same access level as raw member data).
4. **Publish** the sanitized card **byte-identical** to all `k` members, with the **lexical certificate**:
   0% of lines share any 6-word run with any member's text — third-party verifiable by script, no LLM needed.

**Naming:** "CMD" in the paper = this full pipeline (pool + V6). The un-sanitized pooled card (`neutral_fixed`,
"ne") is the **baseline layer** — it matches CMD on both measured axes but is a collage (80–91% of its lines
share 6-word runs with member text) with no certificate. The early over-compressed **`base`** synth stays
deprecated. Every CMD-vs-concat comparison uses a length-matched **`neutral-concat`**.

**V6-era canonical verdicts** (all pre-registered; `results/V6_METHOD_AND_DATA.md` is the step-by-step +
data explainer, `results/ELEMK_DESIGN.md` the frozen designs, `results/V6R*_FINDINGS.md` the per-run verdicts):
- **Sanitize costs no utility**: forced-choice ne−v6 certified TIE on all 3 datasets (MAD .523 / CV .519 /
  Enron .468; δ=.10 equivalence with placebo battery). Direct FC vs the 3 SOTA per-person de-id arms
  (Staab / PETRE / TAROT; Presidio is an Enron-only PII scrubber kept as a footnote arm, not a comparison
  method): **8/9 cells not-worse, 5 SIG wins**; the utility constraint is certified against every arm that
  actually de-identifies (the single SIG-losing cell, CV·PETRE, is a quantified no-op arm — 49% of its CV
  cards are byte-identical to the source card, `results/petre_noop_census.json` — that leaks 2AFC .734).
- **Sanitize costs no anonymity**: v6 2AFC ∋.5 ×3 with paired ne−v6 ∋0 ×3; dual-attacker certified on MAD;
  point-certified at EVERY k∈{2,…,12} on MAD (sanitize kills the style channel, pooling kills the content
  channel — the low-k leak of raw pooled cards disappears). Honest residuals: Enron ~.545 (diffuse,
  input-side), CV ~.57 (two attackers agree; "no-worse" on CV is judge-dependent).
- **What the certificate does and does not buy**: exact + fuzzy near-verbatim channels dead (embedding-neighbor
  member−stranger gap collapses at τ≥.90 while τ=.70 semantic content is retained); but cross-release linkage
  is NOT reduced (paired Δ∋0 — a strong LLM links via decision architecture, not verbatim) ⇒
  **composition/multi-release stays the characterized OPEN direction for any pooled release form**.
- **Process robustness**: judge/attacker/distiller swaps reproduce; rewriter swaps (6 replacement models)
  show the gates are model-agnostic and fail loudly, while safe rebuild is a rewriter skill — deepseek is the
  only rewriter passing content/utility/style simultaneously **bare**, and the skill is **instruction-inducible**:
  one frozen "edit-economy" instruction lifts qwen3.7-max to a full four-axis pass (incl. pooled δ=.10
  anonymity certification) and glm-5.1 to 3/4 (the absolute-anonymity cell is judge-wave-dependent); the
  canonical pipeline does NOT use the instruction (it costs deepseek +1.5pp drop). Rebuild under fresh
  sampling is stable (certificate 0% reproduces; "where to edit" is a deterministic function of the input,
  "how to word it" varies within gates).

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

## Key results — pre-V6 measurement record (the pooling layer)

> ⚠ **This section is the `neutral`-era record of the POOLING layer** (why per-person de-id fails, why pooling
> wins, its honest k8 residual and boundaries). It remains valid for the un-sanitized baseline `ne`, but the
> canonical release form is now **pool + V6** (section above): several sub-claims are superseded there —
> notably, the k8 residual and the "min-k>8" reading apply to `ne` only (the sanitized v6 card is
> point-certified at every k∈{2..12} on MAD), and utility is now measured by the forced-choice instrument
> (`data_provenance.md` §C′), not the three legacy metrics quoted in some bullets below.

(free sonnet-4.6 **2AFC** membership attacker, chance = 0.5; equivalence/MDE tested)

- **Per-person de-id is not enough — not without gutting utility.** Rewriting a single person's card to disguise them
  — including 3 SOTA methods (Staab-R1, PETRE, TAROT/`tpar`) plus the Enron-only Presidio footnote arm — leaks under the stronger **gpt-5.4** attacker
  on every measured cell; on the weaker sonnet attacker it leaks on most cells too (the softest, Enron
  `petre_k4`, has a sonnet CI that just touches 0.5; Presidio is measured on Enron only). Identity lives in the
  *decision architecture*, which survives content-level rewriting. You *can* push a single card to chance — but only by
  **destroying its usefulness** (DP-Prompt T≥2.0 reaches 0.5 only by collapsing utility toward the no-card floor).
  So to be anonymous **and** still useful, you have to **pool**.
  - *Honest caveat on these baselines.* **Presidio is Enron-only by necessity** — it is a
    PII scrubber (names / orgs / emails), and the distilled 20-MAD (bug-triage) and CV (stats-QA) cards contain no PII for
    it to touch, so it is a no-op / N/A there; measuring it only on Enron is honest scope, not a cherry-pick.
    An edit-magnitude census (`scripts/petre_noop_census.py` → `results/petre_noop_census.json`) quantifies each
    baseline's touch: Staab word-similarity .54–.57 and TAROT .27–.36 are real rewrites (0% byte-identical),
    while **PETRE leaves 45–53% of cards byte-identical** to the source card (similarity .93–.95) — a frozen-config
    method property, uniform across datasets, not an implementation artifact.
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

data/  (committed RELEASE ARTIFACTS — the only tracked data)
  20mad/cmd_shared_cards_mad__v6min{,_stats}.json   # canonical V6 cards (k8_s0 + R13 k-grid) + build-stat counters
  se/cmd_shared_cards_cv__v6min{,_stats}.json       # CV V6 cards + stats
  enron/cmd_shared_cards__v6min{,_stats}.json       # Enron V6 cards + stats
  # per-line audit sidecars (*_audit.json) map card lines back to member text — NEVER tracked (hard-denied)

scripts/  — V6 CANONICAL PIPELINE (CMD := pool + V6 sanitize)
  # build chain: random grouping -> neutral pooling -> degeneracy fix -> V6 sanitize -> certificate
  mad_synth_utility.py                       # canonical `neutral` pooled-card synth (K/SEED/ONLY env)
  cmd_fix_degenerate.py                      # degeneracy fix (anti-copy re-distill; retry until max-member-cos < .85)
  cmd_consensus_pool.py                      # element decomposition + embed cache (v5_sanitize dependency; also the GTA consensus ablation)
  elemk_build.py                             # member-element extraction shared lib (v5_sanitize dependency)
  v5_sanitize.py                             # V6 minimal-edit sanitize (EDIT=min; five deterministic gates; SAN_GEN rewriter swap; SAN_GUIDE instruction arm; SAN_SAMPLE fresh-sampling rebuild)
  elemk_v3_gates.py                          # anatomy / lexical-certificate / census instruments (MODE=anatomy|lex|census; CARDS=<labels>)
  # anonymity instrument (free-subagent 2AFC, chance .5)
  neutral_2afc_export.py                     # export self-contained 2AFC packs (indiv gate + test channels; KCL/SEED/CONSPFC env)
  cr_2afc_export.py / cr_2afc_score.py       # generic 2AFC pair exporter / aggregator (de-id battery era, still the pair-builder base)
  score_2afc_summary.py                      # (re)generate _score_summary.json for any battery dir (headline anonymity table)
  r6_2afc_certify.py / r6f_pool_2afc.py      # δ=.10 dual-cluster certification (single wave / pre-committed multi-wave pooling)
  cmd_multiseed_pool.py                      # multi-seed pooling certification on the baseline layer (MSMODE=neufix|a1)
  cmd_equiv_test.py                          # TOST equivalence + non-inferiority + MDE across k
  # utility instrument (FC forced-choice, null .5, placebo battery + δ=.10 verdict dictionary)
  mad_fc_export.py / cv_fc_export.py / enron_fc_export.py   # per-dataset FC pack exporters (drafting + probes; CONTRASTS env)
  cv_fc_score.py                             # shared FC scorer (SIG / certified-TIE / UNDERPOWERED+sMDE)
  fc_status.py                               # programmatic judge-coverage check (never trust a judge's own count)
  fc_multiseed_pool.py                       # multi-wave FC pooling (R2/R11 cluster-wall breaker)
  mad_fc_judge_qwen.py                       # cross-JUDGE swap (qwen3.7-max re-judges the same packs)
  # mechanism / baseline tooling
  cmd_dispersion.py                          # per-cluster dispersion + leaky-tail behind pooled means
  cmd_tcloseness.py                          # $0 no-LLM membership-identifiability / t-closeness probe
  cmd_xcard_link.py / cmd_xcard_export.py / cmd_xcard_score.py  # cross-release linkage (census / FREE-subagent export / score)
  xcard_census_norm.py                       # length-normalized verbatim census
  petre_noop_census.py                       # $0 edit-magnitude census of the de-id baselines (PETRE no-op quantification)
  # R-series robustness verifiers
  r9_rebuild_check.py                        # rebuild-stability programmatic checks (R9)
  r13_neutral_build.py / r13_fc_curve.py     # k-gradient card builder + paired utility curve (R13)
  r7_fuzzy_2afc.py / r7_linkage_paired.py / r7_xcard_v6_census.py  # fuzzy-lexical attacker + v6 linkage (R7)
  r6e_retry_probe.py                         # retry-budget diagnostic on rewriter-swap drops (R6e; output INTERNAL)
  # cross-model swap packages
  build_gpt54_fixed_pkg.py                   # A2 second-attacker package builder (fixed cards); answers in gpt54_2afc_fixed_pkg/
  mad_cmd_build.py / mad_cmd_build_qwen.py / a3_qwen_neutral_build.py  # 20-MAD builder + qwen distiller-swap (A3)
  # dataset builders + shared libs + de-id baselines
  cmd_gate.py / cmd_gate_score.py            # closed-world ≤1/k re-id gate + shared-card builder (DATASET=enron|mad|cv)
  cmd_build_shared.py / cmd_concat_build.py  # k-sweep shared cards / matched neutral-concat baseline (NEUTRAL=1)
  cv_pilot.py / cv_build.py                  # CrossValidated (3rd dataset) build
  deid_enron.py enron_nuwa.py enron_step2.py # shared helpers + de-id baseline methods
  enron_archpool.py mad_nuwa_step2.py mad_comp_two_axis.py enron_clean.py
  enron_collect_full.py enron_nuwa100.py enron_nuwa100_dump.py util6_pool.py mad_step2_baselines.py
  # SUPERSEDED (kept for the retraction trail; see header comments in each file)
  cmd_utility.py mad_utility.py cv_util_judge_export.py cv_util_judge_score.py   # legacy utility instruments -> replaced by FC
  cmd_attack2afc_score.py cmd_attack2afc_grid.py cmd_attack_diag.py              # paid 2AFC runner -> replaced by free-subagent 2AFC
  cmd_attack2afc.py                          # paid runner retired, but build_pairs/SYS stay LIVE deps of neutral_2afc_export.py
  cmd_openworld.py / cmd_openworld_score.py  # legacy nneg-AUC attack -> replaced by 2AFC
  cmd_k8_probe.py / cmd_synth_probe.py cmd_tpr.py cmd_batch.py                   # early-era analyses
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

## Reproduce (V6 canonical chain)

1. **Build datasets** — `enron_collect_full.py`; `util6_pool.py` (`N_DEV=100000 OUT=mad_cmd_pool.json` = full set,
   no cap); `enron_nuwa100.py`; `mad_cmd_build.py`; `cv_build.py`. (raw `data/` is git-ignored; regenerate locally.
   The **V6 release cards + build stats ARE committed** — see `data/` in the layout above.)
2. **DRYRUN cost** — every dump supports `PILOT_DRYRUN=1` / `COST=1` (prints token/$ estimate before spending).
3. **Build the canonical card** — neutral pooling (`mad_synth_utility.py`, K/SEED env) → degeneracy fix
   (`cmd_fix_degenerate.py`) → **V6 sanitize** (`v5_sanitize.py EDIT=min STAGE=build`, five deterministic gates,
   drop >5% = build kill; writes `*__v6min{,_stats}.json` + an INTERNAL `*__v6min_audit.json` sidecar that is
   never shipped).
4. **Verify the lexical certificate (third-party check)** — `elemk_v3_gates.py MODE=lex CARDS=v6min`:
   0.0% of card lines share any 6-consecutive-word run with any member's text. Needs the member texts, i.e.
   step 1 rebuilt locally — the datasets are public; the committed cards let you diff/inspect without any build.
5. **Anonymity (2AFC)** — `neutral_2afc_export.py` writes self-contained packs (indiv positive gate + test
   channels); FREE Claude-Code sonnet subagents answer (`ans_i.json`); `score_2afc_summary.py` scores;
   `r6_2afc_certify.py` / `r6f_pool_2afc.py` run the δ=.10 dual-cluster certification.
6. **Utility (FC)** — `{mad,cv,enron}_fc_export.py` draft both arms per unit (+ pad/fmt/cut/self placebo probes);
   blind subagent judges force-choose; `fc_status.py` checks coverage programmatically; `cv_fc_score.py` emits
   SIG / certified-TIE / UNDERPOWERED verdicts (δ=.10 dictionary, cluster bootstrap).
7. **Structural gate** — `cmd_gate.py` + `cmd_gate_score.py` for the closed-world ≤1/k floor;
   `cmd_equiv_test.py` for TOST/MDE across k.

## Note on module naming

This codebase uses the original research names (`deid_enron`, `enron_nuwa`, `src.attrib_metrics`). A companion
release uses refactored names (`deid.py`, `detective_*`, `src/stats.py`); the two are parallel and
`src/attrib_metrics.py` is byte-equivalent to that release's `src/stats.py`.
