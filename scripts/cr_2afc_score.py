"""Aggregate subagent 2AFC answers (ans_*.json) -> per-channel 2AFC accuracy + cluster-bootstrap CI.

picked_member = (choice == member_slot). acc per channel = mean over pairs. CI = bootstrap over CLUSTERS (card_id)
to respect author-cluster dependence (same instrument as cmd_attack2afc_score). Gate logic:
  indiv  = POSITIVE CONTROL: CI must EXCLUDE 0.5 above (attacker can read the card).
  shared = the test: CI ∋ 0.5 => pooling reaches chance-anonymity (the win); CI>0.5 => pooled card still leaks.

Run: python scripts/cr_2afc_score.py
"""
import sys
import json
import re
from pathlib import Path

import numpy as np

import os
ROOT = Path(__file__).resolve().parents[1]
B = ROOT / os.environ.get("BATCHDIR", "results/cr/2afc_batches")
meta = json.loads((B / "meta.json").read_text(encoding="utf-8"))

ans = {}
for f in sorted(B.glob("ans_*.json")):
    for r in json.loads(f.read_text(encoding="utf-8")):
        c = str(r.get("choice", "")).strip().upper()
        m = re.search(r"[AB]", c)
        if m:
            ans[r["pid"]] = (m.group(0), float(r.get("conf", 50) or 50))

rows = []
for pid, mt in meta.items():
    if pid not in ans:
        continue
    choice, conf = ans[pid]
    rows.append({**mt, "picked_member": int(choice == mt["member_slot"]), "conf": conf})

print(f"scored {len(rows)}/{len(meta)} pairs ({len(meta)-len(rows)} missing)\n")


def boot(sub, n=5000, seed=0):
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
    return float(np.mean([r["picked_member"] for r in sub])), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


# channel order: pos-controls, then hard-pools, then any per-person de-id arms (tpar/staab/petre/presidio/...)
present = list(dict.fromkeys(r["chan"] for r in rows))
PREF = ["indiv", "raw", "shared", "neutral", "concat"]
chans = [c for c in PREF if c in present] + sorted(c for c in present if c not in PREF)


def verdict(chan, lo, hi):
    if chan in ("indiv", "raw"):                                  # positive control: MUST exclude .5 above
        return "POS-CONTROL OK (reads card)" if lo > 0.5 else "WEAK — attacker barely reads card"
    if chan in ("shared", "concat", "neutral"):                  # hard-pool: ∋.5 = anonymized (the win)
        return "CHANCE = anonymized (the win)" if lo <= 0.5 <= hi else ("LEAKS (pooled card still re-IDs)" if lo > 0.5 else "below chance")
    # per-person de-id arm: expected to STILL leak (lo>.5); ∋.5 = it reached chance (would be a surprise)
    return "LEAKS (per-person de-id still re-IDs)" if lo > 0.5 else ("CHANCE (per-person reached anonymity?!)" if lo <= 0.5 <= hi else "below chance")


print(f"  {'channel':10s} {'2AFC acc':>9s} {'95% CI':>18s}  {'n':>4s}  verdict")
print("  " + "-" * 74)
for chan in chans:
    sub = [r for r in rows if r["chan"] == chan]
    if not sub:
        print(f"  {chan:10s}  (no answers)")
        continue
    acc, (lo, hi) = boot(sub)
    print(f"  {chan:10s} {acc:>9.3f}  [{lo:.3f},{hi:.3f}]  {len(sub):>4d}  {verdict(chan, lo, hi)}")
print("  " + "-" * 74)
print("  chance = 0.5. indiv/raw exclude .5 above (pos-control); shared/concat include .5 (pooling -> anonymity);")
print("  per-person de-id arms (tpar/staab/...) expected to STILL exclude .5 above (de-id leaks => need pooling).")
