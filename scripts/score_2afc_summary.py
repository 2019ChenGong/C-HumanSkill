"""Persist the 2AFC anonymity summary (`_score_summary.json`) for a battery dir — closes provenance gap G1/G3.

The headline per-channel 2AFC accuracy table (indiv / shared / concat / de-id arms, per dataset) previously had
NO committed writer: `cr_2afc_score.py` computed the identical numbers but only PRINTED them. This script uses the
byte-identical scoring (picked_member = choice==member_slot; acc per channel; cluster-bootstrap CI over card_id,
n=5000 seed=0 — verified to reproduce the existing MAD/Enron/CV summaries exactly) and WRITES the JSON.

Metadata (dataset/k/seed/nneg_match/instrument) is preserved from an existing `_score_summary.json` if present,
else taken from env (DATASET/K/SEED/NNEG/INSTRUMENT) — so re-running on the 3 canonical dirs regenerates them
losslessly, and pointing it at a new battery (e.g. the qwen swap) produces a fresh summary.

  BATCHDIR=results/mad/2afc_free python scripts/score_2afc_summary.py            # regenerate (preserves metadata)
  BATCHDIR=results/mad/qwen_2afc_k8 OUT=summary.json DATASET=mad-qwen K=8 SEED=0 \
      INSTRUMENT="free-subagent qwen-distilled-cards, sonnet-4.6 2AFC" python scripts/score_2afc_summary.py
"""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / os.environ["BATCHDIR"]
OUT = B / os.environ.get("OUT", "_score_summary.json")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

meta = json.loads((B / "meta.json").read_text(encoding="utf-8"))

ans = {}
for f in sorted(B.glob("ans_*.json")):
    if not re.fullmatch(r"ans_\d+", f.stem):          # stray-guard (ignore ans_backup etc.)
        continue
    for r in json.loads(f.read_text(encoding="utf-8")):
        if not isinstance(r, dict) or "pid" not in r:
            continue
        c = str(r.get("choice", "")).strip().upper()
        m = re.search(r"[AB]", c)
        if m:
            ans[r["pid"]] = (m.group(0), float(r.get("conf", 50) or 50))

rows = []
for pid, mt in meta.items():
    if pid not in ans:
        continue
    choice, _conf = ans[pid]
    rows.append({**mt, "picked_member": int(choice == mt["member_slot"])})

n_missing = len(meta) - len(rows)
print(f"scored {len(rows)}/{len(meta)} pairs ({n_missing} missing)")


def boot(sub, n=5000, seed=0):
    """acc + cluster-bootstrap 95% CI over card_id (n=5000, seed=0 — matches cr_2afc_score / the existing summaries)."""
    by = {}
    for r in sub:
        by.setdefault(r["card_id"], []).append(r["picked_member"])
    clus = list(by.values())
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(clus), len(clus))
        vals = [v for i in pick for v in clus[i]]
        means.append(np.mean(vals))
    acc = float(np.mean([r["picked_member"] for r in sub]))
    return acc, round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3), len(by)


# channel order: positive controls first, then hard-pools, then per-person de-id arms
present = list(dict.fromkeys(r["chan"] for r in rows))
PREF = ["indiv", "raw", "shared", "neutral", "concat"]
chans = [c for c in PREF if c in present] + sorted(c for c in present if c not in PREF)

channels, all_clusters = {}, set()
for chan in chans:
    sub = [r for r in rows if r["chan"] == chan]
    if not sub:
        continue
    acc, lo, hi, ncl = boot(sub)
    channels[chan] = {"acc": round(acc, 3), "ci": [lo, hi], "n": len(sub)}
    all_clusters.update(r["card_id"] for r in sub)

# metadata: preserve from an existing summary (lossless regen), else env
old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
def meta_field(key, env, default):
    if key in old:
        return old[key]
    v = os.environ.get(env)
    return type(default)(v) if v is not None else default

out = {
    "dataset": meta_field("dataset", "DATASET", "unknown"),
    "instrument": meta_field("instrument", "INSTRUMENT", "free-subagent sonnet-4.6 2AFC"),
    "k": meta_field("k", "K", 8),
    "seed": meta_field("seed", "SEED", 0),
    "nneg_match": meta_field("nneg_match", "NNEG", "member"),
    "n_clusters": len(all_clusters),
    "channels": channels,
}
OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"  ds={out['dataset']} k={out['k']} s={out['seed']} nclusters={out['n_clusters']}")
for c, v in channels.items():
    print(f"  {c:10s} acc={v['acc']:.3f} ci={v['ci']} n={v['n']}")
print(f"-> {OUT.relative_to(ROOT)}")
