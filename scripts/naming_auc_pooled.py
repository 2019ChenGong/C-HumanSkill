"""R15 pooled — stratified AUC across the 3 datasets (pre-registered: results/NAMING_R15_AUC_PREREG.md).

Scores are NOT concatenated across datasets: an Enron member vs a CV non-member is not a meaningful
comparison. The pooled statistic is the STRATIFIED Mann-Whitney AUC = sum_ds (n_pos*n_neg)_ds * AUC_ds
/ sum_ds (n_pos*n_neg)_ds, i.e. the probability that a member outranks a non-member FROM THE SAME
DATASET. Bootstrap is stratified too: clusters are resampled within each dataset (14 / 16 / 9), so the
design is preserved. n=5000, seeds {0,1,2}.

Verdicts (pre-registered):
  GATE   indiv  CI-lo > .5
  LADDER each per-person arm CI-lo > .5
  CERT   v6 up95 < .5 + delta, delta = .10 (primary) and .05 (stricter)
  All three must hold under ALL SIX scoring variants; disagreement is printed, never filtered.

  python -P scripts/naming_auc_pooled.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from naming_auc import _auc, VARIANTS  # noqa: E402  (same rank/AUC implementation)

DIRS = {"enron": "enron", "mad": "mad", "cv": "se"}
# R17: the increment dirs hold MORE BRACKETS OF THE SAME CLUSTERS, so they must be merged BY CLUSTER
# (see load()); appending them as separate units would resample one cluster twice and understate
# the clustered variance.
# R18: naming_deid_full is the equal-n increment for the three per-person de-id arms (INDIV_SKIP=2),
# the counterpart of naming_indiv_full for indiv. Merged BY CLUSTER exactly like the others, so the
# three de-id arms now carry the same n (321) as indiv instead of the legacy 78.
SUBDIRS = ("naming_v6", "naming_deid", "naming_indiv_full", "naming_v6_more", "naming_deid_full", "naming_pool_ab")
# A12 (#153/#154): pooled baselines ne/concat join the scan; POOLED mirrors naming_export.POOLED
# (kept local -- these pooled scripts sweep all 3 datasets and must not bind NE to one DATASET).
ARMS = ("indiv", "staab", "petre_k4", "tpar_t15", "v6", "ne", "concat")
POOLED = ("v6", "ne", "concat")
BOOT_N, BOOT_SEEDS, DELTAS = 5000, (0, 1, 2), (0.10, 0.05)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load(arm, variant):
    """-> {ds: [ (scores, labels) per cluster ]}, merging increment dirs into the SAME cluster unit."""
    by_unit = {}
    for ds, d in DIRS.items():
        for sub in SUBDIRS:
            p = ROOT / f"results/{d}/{sub}/_naming_auc.json"
            if not p.exists():
                continue
            sec = json.loads(p.read_text(encoding="utf-8")).get("arms", {}).get(arm, {}).get(variant)
            if not sec:
                continue
            for u, vals in sec["units"].items():
                if vals:
                    by_unit.setdefault(ds, {}).setdefault(u, []).extend(vals)
    return {ds: [(np.array([a for a, _ in v], float), np.array([b for _, b in v], int))
                 for v in units.values()] for ds, units in by_unit.items()}


def strat_auc(per_ds):
    num = den = 0.0
    for units in per_ds.values():
        s = np.concatenate([u[0] for u in units]); y = np.concatenate([u[1] for u in units])
        a = _auc(s, y)
        if a is None:
            continue
        w = float(y.sum()) * float(len(y) - y.sum())
        num += w * a; den += w
    return num / den if den else None


def main():
    print("=== R15 pooled — STRATIFIED AUC over 3 datasets (cluster bootstrap within dataset) ===")
    out = {"prereg": "results/NAMING_R15_AUC_PREREG.md", "arms": {}}
    for arm in ARMS:
        out["arms"][arm] = {}
        for v in VARIANTS:
            per_ds = load(arm, v)
            if not per_ds:
                continue
            point = strat_auc(per_ds)
            cis = {}
            for bseed in BOOT_SEEDS:
                rng = np.random.default_rng(bseed)
                bs = []
                for _ in range(BOOT_N):
                    res = {ds: [units[i] for i in rng.integers(0, len(units), len(units))]
                           for ds, units in per_ds.items()}
                    a = strat_auc(res)
                    if a is not None:
                        bs.append(a)
                cis[bseed] = [round(float(np.percentile(bs, 2.5)), 4),
                              round(float(np.percentile(bs, 97.5)), 4)]
            lo, hi = cis[0]
            rec = {"auc": round(point, 4), "ci": cis[0], "ci_by_seed": cis,
                   "n_units": sum(len(u) for u in per_ds.values()),
                   "clears_chance": bool(lo > 0.5),
                   **{f"cert_d{round(d*100):02d}": bool(hi < 0.5 + d) for d in DELTAS}}
            out["arms"][arm][v] = rec
            tag = ""
            if arm == "indiv":
                tag = "  GATE " + ("PASS" if lo > 0.5 else "**FAIL**")
            elif arm in POOLED:
                tag = "  CERT d.10 " + ("OK" if hi < 0.60 else "NO") + " / d.05 " + \
                      ("OK" if hi < 0.55 else "NO")
            print(f"  {arm:9s} {v}  AUC {point:.3f} [{lo:.3f},{hi:.3f}]  "
                  f"({rec['n_units']} clusters){tag}")
        print()

    print("=== pre-registered verdicts, agreement across all six variants ===")
    for arm in ARMS:
        rows = out["arms"].get(arm, {})
        if not rows:
            continue
        if arm in POOLED:
            ok10 = [v for v in VARIANTS if rows.get(v, {}).get("cert_d10")]
            ok05 = [v for v in VARIANTS if rows.get(v, {}).get("cert_d05")]
            print(f"  {arm:9s} CERT delta=.10: {len(ok10)}/6 variants OK {ok10}")
            print(f"  {arm:9s} CERT delta=.05: {len(ok05)}/6 variants OK {ok05}")
        else:
            ok = [v for v in VARIANTS if rows.get(v, {}).get("clears_chance")]
            print(f"  {arm:9s} CI-lo>.5: {len(ok)}/6 variants {ok}")

    p = ROOT / "results/naming_auc_pooled.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
