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

## Datasets (both the FULL qualifying set — no cherry-pick)

| Dataset | Domain | N | Bar |
|---|---|---|---|
| Enron emails | workplace email decisions | **116** | every sender with ≥20 distinct docs |
| 20-MAD SeaMonkey | bug-triage reasoning | **128** | every developer clearing the activity bar (no random cap) |

## Key results (strong-LLM attacker; both datasets)

- **Membership-inference (open-world, the headline):** shared-card ROC-AUC ≈ **0.5** at k∈{4,8,16} (undetectable);
  individual-card positive control AUC **0.67–0.73** (attacker works).
- **Closed-world re-id (the gate):** shared-card re-id tracks **1/k**; Δ(card−raw) **significantly negative**
  (the card is safer than raw); positive control valid at every k.
- **Utility (Enron):** shared ≈ individual (sharing is **free**), but ≈ a generic checklist and **own ≈ stranger**
  — the preserved value was generic competence, never person-specific.
- **Why pooling, not "archetype clustering":** `archpool` (cluster by reasoning archetype) is **not better than
  random grouping** on either axis; the anonymity comes from *sharing one identical card*, which only CMD's ε=0
  step makes a structural ≤1/k guarantee (vs an empirical near-chance).

## Repository layout

```
src/
  llm.py             # multi-provider LLM client (OpenRouter/DeepSeek/OpenAI/Anthropic); keys from env/.env
  attrib_metrics.py  # cluster bootstrap CI, permutation null, Holm, paired-diff CI
scripts/
  cmd_gate.py / cmd_gate_score.py            # closed-world re-id gate (DATASET=enron|mad)
  cmd_openworld.py / cmd_openworld_score.py  # membership-inference attack + ROC-AUC
  cmd_utility.py                             # utility (shared vs individual/no-card/floor/stranger)
  cmd_batch.py                               # consolidate per-trial dumps for one strong-attacker pass
  cmd_k8_probe.py / cmd_synth_probe.py       # (optional) k=8 win analyses
  mad_cmd_build.py                           # 20-MAD card builder
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
5. **Strong attacker** — a strong LLM (e.g. Opus) reads each batch and writes `_picks_*.json` (a single forced
   scoring pass, **not** an API call in these scripts; reproducible/auditable).
6. **Score** — `cmd_gate_score.py` / `cmd_openworld_score.py` (set `DATASET=`, `KCL=`).

## Note on module naming

This codebase uses the original research names (`deid_enron`, `enron_nuwa`, `src.attrib_metrics`). A companion
release uses refactored names (`deid.py`, `detective_*`, `src/stats.py`); the two are parallel and
`src/attrib_metrics.py` is byte-equivalent to that release's `src/stats.py`.
