"""R13 (#139) cross-k paired utility analysis — the review-M3 estimator.

Units (dev, bug) are IDENTICAL across the six fc_v6k{K} packs (NEXPERT=0, same first-8 bugs), so the
per-unit nec-vs-in win indicator pairs across packs. For each k != 8 this computes
Delta(k) = d_unit(k) - d_unit(8) with inference by a TWO-WAY cluster bootstrap (Cameron-Gelbach-Miller):
the k-partition clusters and the k8-partition clusters are resampled independently, a unit's replicate
weight = (draws of its k-cluster) x (draws of its k8-cluster). The two one-way clusterings are reported
as sensitivity. Delta CI containing 0 = "utility at k indistinguishable from the k8 anchor".

Run: python -P scripts/r13_fc_curve.py    [NBOOT=10000]
Out: results/mad/r13_fc_curve.json + a table.
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NBOOT = int(os.environ.get("NBOOT", "10000"))
KS = (2, 4, 6, 8, 10, 12)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load_pack(k):
    d = ROOT / f"results/mad/fc_v6k{k}"
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    ans = {}
    for f in sorted(d.glob("ans_*.json")):
        for r in json.loads(f.read_text(encoding="utf-8-sig")):
            ans[r["pid"]] = str(r.get("choice", "")).strip().upper()[:1]
    wins = defaultdict(list)                     # unit -> [win indicators over the 2 orders]
    for pid, mt in meta.items():
        if mt.get("kind") != "contrast" or mt.get("x") != "nec" or mt.get("y") != "in":
            continue
        c = ans.get(pid)
        assert c in ("A", "B"), f"pack k{k}: {pid} unanswered/malformed ({c!r})"
        nec_slot = "A" if mt["order"] == 0 else "B"
        wins[mt["unit"]].append(1.0 if c == nec_slot else 0.0)
    d_unit = {u: float(np.mean(v)) for u, v in wins.items()}
    assert all(len(v) == 2 for v in wins.values()), f"pack k{k}: units missing an order"
    return d_unit, cfg["unit_cluster"]


def two_way_boot(delta, ca, cb, nboot, seed=0):
    rng = np.random.default_rng(seed)
    ua, ub = sorted(set(ca)), sorted(set(cb))
    ia = {c: [i for i, x in enumerate(ca) if x == c] for c in ua}
    ib = {c: [i for i, x in enumerate(cb) if x == c] for c in ub}
    delta = np.asarray(delta)
    stats = []
    for _ in range(nboot):
        wa = np.zeros(len(delta)); wb = np.zeros(len(delta))
        for c in rng.choice(len(ua), len(ua), replace=True):
            wa[ia[ua[c]]] += 1.0
        for c in rng.choice(len(ub), len(ub), replace=True):
            wb[ib[ub[c]]] += 1.0
        w = wa * wb
        if w.sum() == 0:
            continue
        stats.append(float(np.average(delta, weights=w)))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def one_way_boot(delta, cl, nboot, seed=0):
    rng = np.random.default_rng(seed)
    uc = sorted(set(cl))
    idx = {c: [i for i, x in enumerate(cl) if x == c] for c in uc}
    delta = np.asarray(delta)
    stats = []
    for _ in range(nboot):
        pick = rng.choice(len(uc), len(uc), replace=True)
        sel = np.concatenate([idx[uc[c]] for c in pick])
        stats.append(float(delta[sel].mean()))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main():
    packs = {k: load_pack(k) for k in KS}
    d8, cl8 = packs[8]
    out = {"nboot": NBOOT, "delta_vs_k8": {}}
    print(f"R13 paired Delta(k) = d(k) - d(k8), identical (dev,bug) units, NBOOT={NBOOT}")
    print(f"{'k':>3} {'mean Delta':>11} {'two-way CI95':>20} {'one-way(k) CI':>20} {'one-way(k8) CI':>20}")
    for k in KS:
        if k == 8:
            continue
        dk, clk = packs[k]
        units = sorted(set(dk) & set(d8))
        assert len(units) == len(dk) == len(d8), f"unit mismatch k{k}: {len(units)}/{len(dk)}/{len(d8)}"
        delta = [dk[u] - d8[u] for u in units]
        ca = [clk[u] for u in units]
        cb = [cl8[u] for u in units]
        m = float(np.mean(delta))
        lo2, hi2 = two_way_boot(delta, ca, cb, NBOOT)
        lo_a, hi_a = one_way_boot(delta, ca, NBOOT)
        lo_b, hi_b = one_way_boot(delta, cb, NBOOT)
        out["delta_vs_k8"][f"k{k}"] = {
            "mean": round(m, 4), "ci95_twoway": [round(lo2, 4), round(hi2, 4)],
            "ci95_oneway_kpart": [round(lo_a, 4), round(hi_a, 4)],
            "ci95_oneway_k8part": [round(lo_b, 4), round(hi_b, 4)],
            "n_units": len(units), "contains_zero_twoway": bool(lo2 <= 0 <= hi2)}
        print(f"{k:>3} {m:>+11.4f} [{lo2:+.4f},{hi2:+.4f}]   [{lo_a:+.4f},{hi_a:+.4f}]   [{lo_b:+.4f},{hi_b:+.4f}]"
              f"   {'∋0' if lo2 <= 0 <= hi2 else 'EXCLUDES 0'}")
    of = ROOT / "results/mad/r13_fc_curve.json"
    of.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {of.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
