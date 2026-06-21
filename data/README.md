# data/ (git-ignored — regenerate locally)

All datasets and intermediate caches live here but are **not committed** (large / regenerable). Rebuild with the
builder scripts:

- **Enron** (`data/enron/`): place the public `enron_mail_20150507.tar.gz` here, then
  `python scripts/enron_collect_full.py` → `collected_ragfull_40.json` (116 authors, ≥20 distinct docs),
  then `python scripts/enron_nuwa100.py` (COLL=collected_ragfull_40.json) for the nuwa / archpool / random_pool cards.
- **20-MAD SeaMonkey** (`data/20mad/`): place the 20-MAD Bugzilla parquet dump here, then
  `N_DEV=100000 OUT=mad_cmd_pool.json python scripts/util6_pool.py` (full 128-dev set, no random cap),
  then `python scripts/mad_cmd_build.py` for the nuwa + aggro cards.

`results/` (scoring outputs, attacker pick files) is likewise git-ignored.
