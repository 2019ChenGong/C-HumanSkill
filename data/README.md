# data/ — raw datasets regenerated locally; V6 release cards COMMITTED

**What IS committed here (the release artifacts of CMD := pool + V6 sanitize):**

| file | content |
|---|---|
| `20mad/cmd_shared_cards_mad__v6min.json` | canonical V6-sanitized shared cards, 20-MAD (k8_s0 + the R13 k∈{2..12} grid), keyed `k{K}_s{seed}_G{cluster}` |
| `20mad/cmd_shared_cards_mad__v6min_stats.json` | per-card build-stat counters (`_config` thresholds + n_lines / n_dropped / retries / tiers / changed_frac) |
| `se/cmd_shared_cards_cv__v6min{,_stats}.json` | same for CrossValidated (k8_s0) |
| `enron/cmd_shared_cards__v6min{,_stats}.json` | same for Enron (k8_s1) |

Each card ships under the **lexical certificate**: 0.0% of lines share any 6-consecutive-word run with any
member's text (third-party verifiable: rebuild the datasets below, then `python scripts/elemk_v3_gates.py
MODE=lex CARDS=v6min`). The per-line **audit sidecars (`*_audit.json`) map card lines back to member text —
they carry the access level of raw member data, are hard-denied in `.gitignore`, and are NEVER shipped.**

**Everything else here is regenerated locally** (large / dataset-derived):

- **Enron** (`data/enron/`): place the public `enron_mail_20150507.tar.gz` here, then
  `python scripts/enron_collect_full.py` → `collected_ragfull_40.json` (116 authors, ≥20 distinct docs),
  then `python scripts/enron_nuwa100.py` (COLL=collected_ragfull_40.json) for the nuwa / archpool / random_pool cards.
- **20-MAD SeaMonkey** (`data/20mad/`): place the 20-MAD Bugzilla parquet dump here, then
  `N_DEV=100000 OUT=mad_cmd_pool.json python scripts/util6_pool.py` (full 128-dev set, no random cap),
  then `python scripts/mad_cmd_build.py` for the nuwa + aggro cards.
- **CrossValidated** (`data/se/`): `python scripts/cv_build.py` (public stats.stackexchange dump; 77 experts
  with ≥15 gold answers).

**Canonical card chain** (how the committed cards are produced): random k-grouping → neutral pooled synth
(`mad_synth_utility.py`) → degeneracy fix (`cmd_fix_degenerate.py`) → **V6 per-line minimal-edit sanitize**
(`v5_sanitize.py EDIT=min STAGE=build`, five deterministic gates, drop >5% = build kill). The un-sanitized
intermediates (`*__neutral{,_fixed}.json`) are internal baseline-layer files and stay untracked.

`results/` (scoring outputs, judge packs, attacker pick files) is likewise git-ignored.
