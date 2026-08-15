"""#152 extra-seed build acceptance — executes the frozen rules of `results/K152_KGRADIENT_PREREG.md` §7.

Adds no criterion of its own.  Three checks, in order:

  1. INTEGRITY   every pre-existing key in all three layers (base / neutral_fixed / v6min) is
                 byte-identical to its `*.pre_k152bak` backup, and nothing was deleted.
                 A single mutated key ABORTS -- published numbers must stay on the same cards.
  2. DROP GATE   per-cell drop rate vs the 10% auto-admit line (docs/DROPGATE_DECISION.md).
                 ★ Frozen failure branch: a cell over the line is NAMED and EXCLUDED from the
                 ladder.  No lowering the gate, no reseeding to find one that passes.
  3. POWER       recompute the trend-test sMDE with the cluster counts that ACTUALLY survived,
                 so the prereg's §2.3 number is replaced by the achieved one before any measurement.

  python -P scripts/k152_build_verify.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GATE = 0.10
LADDER = (4, 6, 8, 10, 12)
NEW = [(k, s) for k in (10, 12) for s in (1, 2)]
DS = [("mad", "data/20mad", "cmd_shared_cards_mad"),
      ("enron", "data/enron", "cmd_shared_cards"),
      ("cv", "data/se", "cmd_shared_cards_cv")]
# SD floor + noise coefficient fitted in scripts/k152_power.py (from the answered A12 packs)
SD_A, SD_B = 0.00706, 0.01831
BSEEDS = 8


def cell(key):
    m = re.match(r"k(\d+)_s(\d+)_", key)
    return (int(m.group(1)), int(m.group(2))) if m else None


def main():
    print("=== #152 build acceptance (prereg K152_KGRADIENT_PREREG.md §7) ===\n")

    # ---- 1. integrity -----------------------------------------------------------------
    print("--- 1. INTEGRITY: pre-existing keys unchanged across all three layers ---")
    bad = []
    for ds, d, base in DS:
        for suf in ("", "__neutral_fixed", "__v6min"):
            p = ROOT / d / f"{base}{suf}.json"
            b = Path(str(p) + ".pre_k152bak")
            if not b.exists():
                print(f"  {ds:6s} {suf or '(base)':16s} NO BACKUP -- cannot verify"); bad.append(str(p)); continue
            new, old = (json.loads(x.read_text(encoding="utf-8")) for x in (p, b))
            ch = [k for k in old if new.get(k) != old[k]]
            rm = [k for k in old if k not in new]
            add = sorted(set(new) - set(old))
            flag = "OK " if not ch and not rm else "**FAIL**"
            print(f"  {ds:6s} {suf or '(base)':16s} {flag} old {len(old):4d} changed {len(ch)} "
                  f"deleted {len(rm)} added {len(add)}")
            if ch or rm:
                bad.append(f"{p.name}: changed={ch[:3]} deleted={rm[:3]}")
    if bad:
        raise SystemExit(f"\n[ABORT] integrity failed -- published numbers would move: {bad}")
    print("  ⇒ pure-additive on all 9 files.\n")

    # ---- 2. drop gate -----------------------------------------------------------------
    print(f"--- 2. DROP GATE: new cells vs the {GATE:.0%} auto-admit line ---")
    print(f"  {'ds':6s} {'cell':9s} {'cards':>5s} {'dropped/content':>17s} {'rate':>7s}  verdict")
    surviving, voided = defaultdict(int), []
    for ds, d, base in DS:
        st = json.loads((ROOT / d / f"{base}__v6min_stats.json").read_text(encoding="utf-8"))
        agg = defaultdict(lambda: [0, 0, 0])
        for k, v in st.items():
            c = cell(k)
            if not c or not isinstance(v, dict):
                continue
            a = agg[c]; a[0] += v.get("n_dropped", 0); a[1] += v.get("n_content", 0); a[2] += 1
        for c in sorted(agg, key=lambda x: (x[0], x[1])):
            dr, nc, ncards = agg[c]
            rate = dr / max(1, nc)
            isnew = c in NEW
            if c[0] not in LADDER:
                continue
            ok = rate <= GATE
            if ok:
                surviving[c[0]] += ncards
            elif isnew:
                voided.append((ds, c, rate))
            tag = ("NEW " if isnew else "    ") + ("PASS" if ok else "**OVER -> EXCLUDED**")
            print(f"  {ds:6s} k{c[0]}_s{c[1]:<6d} {ncards:5d} {dr:8d}/{nc:<8d} {100*rate:6.2f}%  {tag}")
    if voided:
        print("\n  ★ EXCLUDED per the frozen failure branch (prereg §7): "
              + ", ".join(f"{d} k{c[0]}_s{c[1]} {100*r:.1f}%" for d, c, r in voided))
        print("    No gate lowering, no reseeding. These cells are reported by name as un-buildable.")
    else:
        print("\n  ★ all new cells passed the gate.")

    # ---- 3. achieved power ------------------------------------------------------------
    print(f"\n--- 3. ACHIEVED POWER (trend test, BSEEDS={BSEEDS}) ---")
    sd = float(np.sqrt(SD_A + SD_B / BSEEDS))
    kk = np.array(LADDER, float)
    nn = np.array([surviving[k] for k in LADDER], float)
    print(f"  surviving cards per k: " + "  ".join(f"k{k}={surviving[k]}" for k in LADDER)
          + f"   (total {int(nn.sum())})")
    kb = (kk * nn).sum() / nn.sum()
    sxx = (nn * (kk - kb) ** 2).sum()
    se = sd / np.sqrt(sxx) * (kk.max() - kk.min())
    print(f"  SD={sd:.4f}  Sxx={sxx:.1f}  ->  95% CI half-width = +-{1.96*se:.4f}   "
          f"sMDE(80%) = {2.8*se:.4f}")
    print(f"  prereg §2.3 planned: .039 (336 cards)  |  s0-only baseline: .053 (226 cards)")
    print(f"  v6 residual above chance (the whole available signal) = +.041 "
          f"[R17 +.003,+.083]  -> {'sMDE BELOW signal, max effect detectable' if 2.8*se < 0.041 else 'sMDE ABOVE signal, still a bounding experiment only'}")

    out = {"prereg": "results/K152_KGRADIENT_PREREG.md §7", "gate": GATE,
           "surviving_cards_by_k": {str(k): surviving[k] for k in LADDER},
           "voided_cells": [{"ds": d, "k": c[0], "seed": c[1], "drop_rate": r} for d, c, r in voided],
           "achieved": {"sd": sd, "bseeds": BSEEDS, "ci_halfwidth": 1.96 * se, "smde_80": 2.8 * se}}
    p = ROOT / "results/k152_build_verify.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
