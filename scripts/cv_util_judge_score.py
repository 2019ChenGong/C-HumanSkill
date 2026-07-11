"""Aggregate free-sonnet-subagent CV utility judge answers (ans_*.json) -> indiv-nocard / indiv-stranger /
shared-nocard means + cluster-bootstrap CI over expert. Debiases each (unit,cmp) pair r1/r2 EXACTLY like
cv_pilot.judge():  r1 = _raw(X,Y) [+1 if 'A'(=X) better], r2 = _raw(Y,X) [+1 if 'A'(=Y) better];
Xwin = (r1>0)+(r2<0), Ywin = (r1<0)+(r2>0); verdict = +1 X better / -1 Y better / 0 tie.
in_no verdict = indiv-vs-nocard, sh_no = shared-vs-nocard (positive => card/pooling helps).

Run: BATCHDIR=results/se/util_judge python scripts/cv_util_judge_score.py   [compares vs haiku-26 if present]
"""
import os
import re
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / os.environ.get("BATCHDIR", "results/se/util_judge")
meta = json.loads((B / "meta.json").read_text(encoding="utf-8"))

ans = {}
for f in sorted(B.glob("ans_*.json")):
    for r in json.loads(f.read_text(encoding="utf-8")):
        c = str(r.get("choice", "")).strip().upper()
        if c.startswith("A"):
            ans[r["pid"]] = 1
        elif c.startswith("B"):
            ans[r["pid"]] = -1
        elif c.startswith("T"):
            ans[r["pid"]] = 0

# gather r1/r2 per (unit,cmp)
pairs = {}
for pid, mt in meta.items():
    if pid not in ans:
        continue
    key = (mt["unit"], mt["cmp"])
    pairs.setdefault(key, {"u": mt["u"]})[mt["role"]] = ans[pid]

CMPS_ALL = [("in_no", "indiv-nocard (card useful?)"), ("in_st", "indiv-stranger (person-specific?)"),
            ("sh_no", "shared-nocard (pooling preserves?)"),
            ("ne_no", "neutral-nocard (neutral pooling helps?)"), ("ne_in", "neutral-indiv (neutral vs ceiling)")]
_present = {mt["cmp"] for mt in meta.values()}
CMPS = [(c, nm) for c, nm in CMPS_ALL if c in _present]
# dynamic head-to-head arms: ne_{arm} = neutral vs a per-person de-id method (+mean => neutral MORE competent)
for c in sorted(_present):
    if c not in dict(CMPS_ALL):
        label = f"neutral-{c[3:]} (CMD vs de-id)" if c.startswith("ne_") else f"{c}"
        CMPS.append((c, label))
vals = {c: [] for c, _ in CMPS}
grps = {c: [] for c, _ in CMPS}
incomplete = 0
for (unit, cmp), d in pairs.items():
    if "r1" not in d or "r2" not in d:
        incomplete += 1
        continue
    r1, r2 = d["r1"], d["r2"]
    xw = (r1 > 0) + (r2 < 0)
    yw = (r1 < 0) + (r2 > 0)
    v = 1 if xw > yw else (-1 if yw > xw else 0)
    vals[cmp].append(v); grps[cmp].append(d["u"])


def boot(v, g, n=5000, seed=0):
    by = {}
    for x, u in zip(v, g):
        by.setdefault(u, []).append(x)
    clus = list(by.values())
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(clus), len(clus))
        xs = [x for i in pick for x in clus[i]]
        means.append(np.mean(xs))
    return float(np.mean(v)), (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


n_units = len({u for (u, _c) in pairs})
print(f"scored {len(ans)}/{len(meta)} directional tasks -> {n_units} (unit,cmp) pairs "
      f"({incomplete} pairs dropped: missing r1/r2)\n")
print(f"  {'comparison':40s} {'mean':>7s} {'95% CI':>18s}  {'n':>4s}")
print("  " + "-" * 74)
out = {"instrument": "free-subagent sonnet judge", "cohort": os.environ.get("NEXP", "26"), "cmp": {}}
for c, nm in CMPS:
    if not vals[c]:
        print(f"  {nm:40s}  (no answers)"); continue
    m, (lo, hi) = boot(vals[c], grps[c])
    sig = "SIG (CI excl 0)" if (lo > 0 or hi < 0) else "ns (CI incl 0)"
    out["cmp"][c] = {"mean": round(m, 3), "ci": [round(lo, 3), round(hi, 3)], "n": len(vals[c])}
    print(f"  {nm:40s} {m:+.3f}  [{lo:+.3f},{hi:+.3f}]  {len(vals[c]):>4d}  {sig}")
print("  " + "-" * 74)

# side-by-side vs the stored haiku baseline (same cohort) if available
hb = ROOT / "results" / "se" / ("cv_pilot_n26.json" if out["cohort"] == "26" else "cv_pilot.json")
if hb.exists():
    h = json.loads(hb.read_text(encoding="utf-8"))["all"]
    print(f"\n  vs haiku baseline ({hb.name}):")
    for c, nm in CMPS:
        hv = h.get(c, {}).get("mean")
        sv = out["cmp"].get(c, {}).get("mean")
        if hv is not None and sv is not None:
            print(f"    {c:8s}  sonnet {sv:+.3f}  |  haiku {hv:+.3f}  (sign {'MATCH' if (sv > 0) == (hv > 0) else 'FLIP!'})")

(B / "_util_judge_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nsaved -> {(B / '_util_judge_summary.json').relative_to(ROOT)}")
