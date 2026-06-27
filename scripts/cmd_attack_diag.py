"""$0 diagnostic: does the 'universal matcher' phenomenon exist in the EXISTING attacker picks?
If some candidate-authors get systematically HIGH scores even when they are NEGATIVES (not members),
then per-candidate calibration (subtract each author's baseline) has real confound to remove.
If every author's negative-baseline is ~equal, calibration is moot.

Also reports: tie-rate, score histogram, and a SIMULATED calibration on the existing raw scores
(subtract each author's mean-as-negative) to preview whether nneg-AUC / TPR@5% would move.
Reads results[/{ds}]/_picks_ow_k{K}_shared_s{S}.json + _ow_..._key.json. Set DATASET, KCL, SEED.
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DS = os.environ.get("DATASET", "mad")
KCL = os.environ.get("KCL", "8")
SEED = os.environ.get("SEED", "0")
RES = ROOT / "results" if DS == "enron" else ROOT / "results" / DS

key = json.loads((RES / f"_ow_k{KCL}_shared_s{SEED}_key.json").read_text(encoding="utf-8"))
picks = json.loads((RES / f"_picks_ow_k{KCL}_shared_s{SEED}.json").read_text(encoding="utf-8-sig"))

# collect (target_cluster, author, label, score) for every scored candidate
recs = []
for t, meta in key.items():
    sc = picks.get(t)
    if not sc:
        continue
    clus = meta["cluster"]
    for slot, lab in meta["labels"].items():
        if slot not in sc:
            continue
        b = meta["cands"][slot]
        try:
            recs.append((clus, b, lab, float(sc[slot])))
        except (TypeError, ValueError):
            pass

allscores = [r[3] for r in recs]
from collections import Counter
c = Counter(allscores)
tie = sum(v for v in c.values() if v > 1) / len(allscores)
print(f"=== DATASET={DS} k{KCL} s{SEED}: {len(recs)} scored candidates, {len(key)} trials ===")
print(f"score support: {sorted(c)[:12]}{' ...' if len(c)>12 else ''}  ({len(c)} distinct values)")
print(f"tie-rate = {tie:.3f}   mean={np.mean(allscores):.1f}  median={np.median(allscores):.0f}")

# universal-matcher test: per-author mean score WHEN A NEGATIVE (rneg/nneg only)
neg_by_author = defaultdict(list)
pos_by_author = defaultdict(list)
for clus, b, lab, s in recs:
    (pos_by_author if lab == "pos" else neg_by_author)[b].append(s)
neg_means = {b: np.mean(v) for b, v in neg_by_author.items() if len(v) >= 2}
vals = np.array(list(neg_means.values()))
print(f"\n--- universal-matcher test (per-author mean score AS A NEGATIVE, >=2 obs) ---")
print(f"n authors = {len(neg_means)}")
print(f"author negative-baseline: mean={vals.mean():.1f}  SD={vals.std():.1f}  range [{vals.min():.0f}, {vals.max():.0f}]")
print(f"  -> spread across authors (SD) is the confound calibration removes. SD>>0 => calibration has teeth.")
hi = sorted(neg_means.items(), key=lambda x: -x[1])[:5]
lo = sorted(neg_means.items(), key=lambda x: x[1])[:5]
print(f"  most 'matchy' negatives: " + ", ".join(f"{b}:{m:.0f}" for b, m in hi))
print(f"  least 'matchy' negatives: " + ", ".join(f"{b}:{m:.0f}" for b, m in lo))

# how much of the score variance is author (matchiness) vs label (real signal)?
# one-way: SS between authors (on negatives) vs total
neg_all = np.array([s for v in neg_by_author.values() for s in v])
grand = neg_all.mean()
ss_tot = ((neg_all - grand) ** 2).sum()
ss_auth = sum(len(v) * (np.mean(v) - grand) ** 2 for v in neg_by_author.values())
print(f"  fraction of NEGATIVE-score variance explained by author identity = {ss_auth/ss_tot:.3f}")
print(f"   (this is pure 'some authors look matchy' confound; calibration targets exactly this)")

# ---- simulated calibration on EXISTING raw scores ----
# calibrated_score(target C, cand b) = raw - mu_b, where mu_b = mean of b's scores when NEGATIVE in OTHER trials
# (leave-one-out so we don't subtract the trial itself)
def auc(pairs):
    from sklearn.metrics import roc_auc_score
    y = [p[0] for p in pairs]; s = [p[1] for p in pairs]
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s))

def tpr_at(pos, neg, target):
    pos = np.array(pos); neg = np.array(neg); best = 0.0
    for thr in sorted(set(neg) | set(pos)):
        fpr = float(np.mean(neg >= thr)); tpr = float(np.mean(pos >= thr))
        if fpr <= target and tpr > best:
            best = tpr
    return best

# build per-author negative-baseline from NEGATIVE observations (rneg+nneg), leave-one-out by trial
# (approximation: use global per-author neg mean; LOO matters little at these n)
mu = {b: np.mean(v) for b, v in neg_by_author.items()}
glob_mu = np.mean(neg_all)
def cal(b, s):
    return s - mu.get(b, glob_mu)

for neg_lab in ("rneg", "nneg"):
    raw_pos = [s for _c, _b, lab, s in recs if lab == "pos"]
    raw_neg = [s for _c, _b, lab, s in recs if lab == neg_lab]
    cal_pos = [cal(b, s) for _c, b, lab, s in recs if lab == "pos"]
    cal_neg = [cal(b, s) for _c, b, lab, s in recs if lab == neg_lab]
    a_raw = auc([(1, s) for s in raw_pos] + [(0, s) for s in raw_neg])
    a_cal = auc([(1, s) for s in cal_pos] + [(0, s) for s in cal_neg])
    print(f"\n[{neg_lab}] AUC raw={a_raw:.3f} -> calibrated={a_cal:.3f}   "
          f"TPR@5% raw={tpr_at(raw_pos,raw_neg,0.05):.2f} -> cal={tpr_at(cal_pos,cal_neg,0.05):.2f}  "
          f"TPR@10% raw={tpr_at(raw_pos,raw_neg,0.10):.2f} -> cal={tpr_at(cal_pos,cal_neg,0.10):.2f}")
print("\nNB: this is a CHEAP PREVIEW on existing raw subagent scores (global per-author mu, not LOO, not a fresh")
print("    calibrated query). It only tests whether the confound EXISTS and whether removing it MIGHT move the metric.")
