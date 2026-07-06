# CMD — Cluster-Mixture De-identification of Distilled Expert "Skill Cards"

**One question:** a company distills an expert into a publishable **skill card** (a profile of how they
decide and work). Can we scrub **WHO it is** (anonymity) while keeping the card **USEFUL** (utility)?

**CMD's answer:** don't rewrite one person's card to disguise them — have **k people publish ONE identical
shared card**. By construction the card can't be narrowed below the group of k.

---

## Method (CMD)

1. **Partition** authors randomly into clusters of size `k`.
2. **Synthesize** ONE ε=0 *shared card* per cluster — only the group's **common** competence/decision approach,
   with no detail unique to any single member.
3. **Publish** it **byte-identical** to all `k` members.

**Guarantee.** A card-only attacker is bounded to **re-id ≤ 1/k** — classical *k*-anonymity / microaggregation
applied to a free-text decision card (the shared representative is an LLM-synthesized, *usable* card, not a numeric
centroid). Against an attacker who *also* holds your raw text, the card adds **no leak over raw**; the residual is
your own raw writing style, which no card scheme can fix.

## Datasets (all three the FULL qualifying set — no cherry-pick)

| Dataset | Domain | N | Bar |
|---|---|---|---|
| Enron emails | workplace email decisions | **116** | every sender with ≥20 distinct docs |
| 20-MAD SeaMonkey | bug-triage reasoning | **128** | every developer clearing the activity bar (no random cap) |
| CrossValidated (stats SE) | statistical-consulting Q&A | **77** | every expert with ≥15 gold answers |

## Key results (free sonnet-4.6 **2AFC** membership attacker, chance = 0.5; equivalence/MDE tested)

- **Per-person de-id is not enough.** Rewriting a single person's card to disguise them — including 4 SOTA methods
  (Staab-R1, PETRE, TAROT/`tpar`, Presidio) — still leaks: the 2AFC stays above chance. Identity lives in the
  *decision architecture*, which survives content-level rewriting. So you have to **pool**.
- **Pooling reaches chance — how *firmly* depends on the domain (honest status, not a flat "always").** Shared-card
  2AFC ≈ 0.5 while the individual-card positive control leaks (**0.59–0.73**). Under equivalence (TOST) + MDE:
  **20-MAD is statistically *certified* anonymous** at the strict ±5pp margin (well-powered, leak ≥0.55 excluded);
  **Enron sits at chance (~0.50)** and **CV reaches chance on average at k≥8 but shows a real dose-response**
  (the shared card *leaks* at k≤6, CI excludes 0.5) **with high inter-cluster variance** (per-cluster acc spans .19–.75).
- **Multi-seed pooling certifies Enron + CV at the ±10pp practical margin.** Re-partitioning the same people under
  3 random seeds (k8) and pooling — with a person-clustered bootstrap that honestly absorbs the reused-writing-sample
  dependence — lifts both from "underpowered null" to **certified: a leak ≥0.60 is excluded** (Enron pooled 0.511,
  conservative up95 0.563; CV 0.545, up95 0.597; positive control still leaks 0.65 / 0.71). At the strict ±5pp margin
  they fall just short — Enron's estimate is **dead-on chance (0.511)** yet the most-conservative clustering misses by
  0.013, and **CV shows a small *stable* residual** (all 3 seeds land at 84/154 = 0.545; pooling makes it *precise*,
  not zero). That residual is itself the min-k law showing through: CV's richer identity needs k>8 to fully wash out.
- **Min-k for anonymity ∝ identity richness:** Enron (weak fingerprint) ≤k2 < 20-MAD ~k4 < CV (rich) ~k8 — a richer
  decision-fingerprint needs more people to wash out.
- **Model-agnostic (not a deepseek-consensus artifact).** Rebuilding the *entire* card stack with a different
  distiller (qwen3.7-max in non-thinking mode) reproduces it on 20-MAD: individual leaks (**0.68**), shared reaches
  chance (**0.49 ∋ 0.5**).
- **Utility preserved, pooling ~free.** shared ≈ individual card, both beat no-card; but the card ≈ a generic
  competent-colleague checklist and **own ≈ stranger** — the preserved value is generic competence, never
  person-specific. Concordant across judge models (deepseek / haiku / sonnet), same cohort (e.g. CV
  indiv−nocard SIG under both haiku +0.46 and sonnet +0.24; shared−nocard SIG +0.50 / +0.20).
- **Closed-world ≤1/k (structural).** In a closed lineup the shared-card re-id tracks **1/k** and Δ(card−raw) is
  significantly negative (safer than raw). The ε=0 byte-identical step is what makes this a *structural* ≤1/k bound
  rather than only an empirical near-chance.
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
  cmd_attack2afc.py                          # 2AFC membership attacker (primary anonymity instrument, chance 0.5)
  cr_2afc_export.py / cr_2afc_score.py       # export 2AFC pairs for FREE sonnet-4.6 subagents; aggregate answers
  cmd_equiv_test.py                          # TOST equivalence + one-sided non-inferiority + MDE on the 2AFC nulls
  cmd_build_shared.py                        # build only the ε=0 shared cards for a k-sweep (CMD-only)
  cmd_multiseed_pool.py                      # multi-seed pooling certification (per-seed / (seed,card) / person bootstraps)
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
