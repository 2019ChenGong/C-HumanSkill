"""R14 pooled re-analysis (pre-registered: results/NAMING_R14_PREREG.md).

Reads the six per-run sidecars written by scripts/naming_depth.py and produces:
  A  depth excess, pooled across the 3 datasets (units = "ds:cluster")           [pre-registered]
  B  final-hit, pooled across the 3 datasets vs the pooled chance line           [POST-HOC, see prereg B.3]
  C  variance-components power projection: smallest m (targets per cluster x m)
     at which the normal-approx CI-lo would clear the decision line             [PLANNING AID, not a result]

  python -P scripts/naming_pooled_depth.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = {"enron": "enron", "mad": "mad", "cv": "se"}
SUBDIRS = ("naming_v6", "naming_deid", "naming_indiv_full", "naming_v6_more", "naming_pool_ab")  # R17 increments
# A12 (#153/#154): pooled baselines ne/concat join the scan; POOLED mirrors naming_export.POOLED
# (kept local -- these pooled scripts sweep all 3 datasets and must not bind NE to one DATASET).
ARMS = ("indiv", "staab", "petre_k4", "tpar_t15", "v6", "ne", "concat")
POOLED = ("v6", "ne", "concat")
BOOT_N, BOOT_SEED = 5000, 0
MS = (1, 2, 4, 8)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def boot(ulist):
    allv = [x for u in ulist for x in u]
    rng = np.random.default_rng(BOOT_SEED)
    means = [np.mean([x for i in rng.integers(0, len(ulist), len(ulist)) for x in ulist[i]])
             for _ in range(BOOT_N)]
    return (float(np.mean(allv)), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), len(allv), len(ulist))


def project(ulist, line):
    """Variance-components projection of CI-lo when each cluster carries m x its current observations."""
    K = len(ulist)
    n = [len(u) for u in ulist]
    nbar = float(np.mean(n))
    mu = float(np.mean([x for u in ulist for x in u]))
    nobs = sum(n)
    sw2 = (sum(float(np.sum((np.array(u) - np.mean(u)) ** 2)) for u in ulist) / (nobs - K)) if nobs > K else 0.0
    msb = sum(len(u) * (np.mean(u) - mu) ** 2 for u in ulist) / (K - 1) if K > 1 else 0.0
    sb2 = max(0.0, (msb - sw2) / nbar)
    out = {}
    for m in MS:
        se = float(np.sqrt((sb2 + sw2 / (m * nbar)) / K))
        out[m] = {"ci_lo": round(mu - 1.96 * se, 4), "clears": bool(mu - 1.96 * se > line)}
    first = next((m for m in MS if out[m]["clears"]), None)
    return {"mu": round(mu, 4), "sigma2_between": round(sb2, 5), "sigma2_within": round(sw2, 5),
            "n_clusters": K, "obs_per_cluster": round(nbar, 2), "line": round(line, 5),
            "by_m": {str(m): v for m, v in out.items()}, "min_m_clearing": first}


def collect(arm, key):
    """-> (list of per-unit value lists, list of per-bracket chances) for one arm across the 3 datasets.

    R17: increment dirs carry MORE BRACKETS OF THE SAME CLUSTERS -> merge by "ds:unit", never append
    as separate units (that would resample one cluster twice and understate the clustered variance).
    """
    by_unit, chances = {}, []
    for ds, d in DIRS.items():
        for sub in SUBDIRS:
            p = ROOT / f"results/{d}/{sub}/_naming_depth.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text(encoding="utf-8"))
            sec = s.get("arms", {}).get(arm, {}).get(key)
            if not sec:
                continue
            for u, v in sec["units"].items():
                if v:
                    by_unit.setdefault(f"{ds}:{u}", []).extend(v)
            chances.extend(sec.get("chances", []))
    return list(by_unit.values()), chances


def main():
    out = {"prereg": "results/NAMING_R14_PREREG.md", "A_depth_pooled": {}, "B_final_pooled": {},
           "C_projection": {"note": "PLANNING AID ONLY — not a result; optimistic (treats point "
                                    "estimates as truth); m=4 == 2 targets/cluster -> 8 (full cohort)"}}

    print("=== R14-A  depth excess POOLED across 3 datasets (pre-registered; SIGNAL iff CI-lo > 0) ===")
    for arm in ARMS:
        u, _ = collect(arm, "depth")
        if not u:
            continue
        exc, lo, hi, n, k = boot(u)
        sig = lo > 0
        gate = ("   <-- POSITIVE-CONTROL GATE " + ("PASS" if sig else "**UND**")) if arm == "indiv" else ""
        out["A_depth_pooled"][arm] = {"excess": round(exc, 4), "ci": [round(lo, 4), round(hi, 4)],
                                      "n_members": n, "n_units": k, "signal": bool(sig)}
        print(f"  {arm:9s} excess {exc:+.3f} [{lo:+.3f},{hi:+.3f}]  (n={n} members, {k} clusters)  "
              f"{'SIGNAL' if sig else 'und':6s}{gate}")
        out["C_projection"][f"depth::{arm}"] = project(u, 0.0)

    print("\n=== R14-B  final-hit POOLED  [POST-HOC — chosen after seeing the per-dataset VOID] ===")
    for arm in ARMS:
        u, ch = collect(arm, "final")
        if not u or not ch:
            continue
        hit, lo, hi, n, k = boot(u)
        line = float(np.mean(ch))
        ok = lo > line
        gate = ("   <-- POSITIVE-CONTROL GATE " + ("PASS" if ok else "**VOID**")) if arm == "indiv" else ""
        out["B_final_pooled"][arm] = {"hit": round(hit, 4), "ci": [round(lo, 4), round(hi, 4)],
                                      "chance": round(line, 5), "n_brackets": n, "n_units": k,
                                      "clears": bool(ok)}
        print(f"  {arm:9s} hit {hit:.3f} [{lo:.3f},{hi:.3f}]  pooled chance {line:.4f}  "
              f"({n} brackets, {k} clusters)  {'PASS' if ok else 'VOID':4s}{gate}")
        out["C_projection"][f"final::{arm}"] = project(u, line)

    print("\n=== R14-C  power projection (PLANNING AID, not a result) — min m clearing the line ===")
    for key, pr in out["C_projection"].items():
        if not isinstance(pr, dict) or "by_m" not in pr:
            continue
        mm = pr["min_m_clearing"]
        cis = "  ".join(f"m{m}:{pr['by_m'][str(m)]['ci_lo']:+.3f}" for m in MS)
        print(f"  {key:20s} {cis}   -> min m = {mm if mm else '>8'}")

    p = ROOT / "results/naming_depth_pooled.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
