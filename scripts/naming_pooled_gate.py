"""Pre-registered POOLED positive-control gate for the 3-arm naming run (see naming_export.py header).

GATE: petre_k4 (documented no-op arm) r1-conditional accuracy, pooled across the 3 datasets with
cluster bootstrap (units = ds-prefixed clusters, n=5000 seed=0), must clear the pooled chance line
(CI-lo > chance) = the wave's attacker has teeth. If VOID -> rerun the indiv arm before interpreting
any staab/tpar null. Secondary print: the no-op subset (cards byte-identical to the indiv card).

  python -P scripts/naming_pooled_gate.py            # reads results/{enron,mad,se}/naming_deid/_naming_summary.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = {"enron": "enron", "mad": "mad", "cv": "se"}
BOOT_N, BOOT_SEED = 5000, 0

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def pooled(section_key, label):
    units, chs = {}, []
    for ds, d in DIRS.items():
        p = ROOT / f"results/{d}/naming_deid/_naming_summary.json"
        if not p.exists():
            print(f"  [{label}] missing {p} — run naming_score.py for {ds} first"); return
        s = json.loads(p.read_text(encoding="utf-8"))
        sec = s.get(section_key) if section_key != "round1_conditional" \
            else s.get("round1_conditional", {}).get("petre_k4")
        if not sec or "units" not in sec:
            print(f"  [{label}] no units in {p}:{section_key}"); return
        for u, vals in sec["units"].items():
            units[f"{ds}:{u}"] = vals
        chs.extend([sec["chance"]] * sec["n_groups"])
    ulist = [v for v in units.values() if v]
    allv = [x for u in ulist for x in u]
    rng = np.random.default_rng(BOOT_SEED)
    means = []
    for _ in range(BOOT_N):
        pick = rng.integers(0, len(ulist), len(ulist))
        means.append(np.mean([x for i in pick for x in ulist[i]]))
    acc, lo, hi = float(np.mean(allv)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    chance = float(np.mean(chs))
    verdict = "GATE PASS" if lo > chance else "GATE **VOID** (rerun indiv arm before reading staab/tpar nulls)"
    print(f"  [{label}] acc {acc:.3f} [{lo:.3f},{hi:.3f}]  pooled chance {chance:.3f}  "
          f"(n={len(allv)} groups, {len(ulist)} clusters)  {verdict}")
    return {"acc": round(acc, 3), "ci": [round(lo, 3), round(hi, 3)], "chance": round(chance, 4),
            "n_groups": len(allv), "n_units": len(ulist), "gate_pass": lo > chance}


def main():
    print("=== pooled naming gate (pre-registered: petre_k4 r1-conditional across 3 datasets) ===")
    out = {"gate_petre_r1_pooled": pooled("round1_conditional", "petre_k4 r1-cond POOLED (binding gate)"),
           "secondary_noop_pooled": pooled("petre_noop_subset", "petre no-op subset POOLED (secondary)")}
    p = ROOT / "results/naming_deid_pooled_gate.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
