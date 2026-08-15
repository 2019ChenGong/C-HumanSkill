"""#152 instrument-validity gate — executes `results/K152_KGRADIENT_PREREG.md` §4.1 and nothing else.

CRITERION (frozen before the wave): the same-wave `indiv` positive control must clear its own
chance line in EVERY dataset (3/3).  Cluster bootstrap over the owner's cluster, n=5000, seeds
{0,1,2}.  If it fails -> the wave has no teeth, the main k-gradient wave is NOT run (prereg §7
step 4 exists precisely so a toothless attacker is caught before the 5-figure question wave).

Reads the signal-bearing repack `results/{mad,enron,se}/naming_k152gate_sig/round1`.
`conf` is NOT used: the r1-conditional read never consumed it (see NAMING_R20_DISPATCH_TEMPLATE.md
revision v2 impact table), which is why the one disclosed out-of-band conf (Enron
`r1_indiv021_g12` = 40, `results/enron/naming_k152gate_sig/_STATUS.md`) cannot touch this verdict.

  python -P scripts/k152_gate_readout.py
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BOOT_N, SEEDS = 5000, (0, 1, 2)
CELLS = [("mad", 0, "results/mad"), ("enron", 1, "results/enron"), ("cv", 0, "results/se")]


def boot(by_unit, seed):
    units = [v for v in by_unit.values() if v]
    allv = [x for u in units for x in u]
    rng = np.random.default_rng(seed)
    means = [np.mean([x for i in rng.integers(0, len(units), len(units)) for x in units[i]])
             for _ in range(BOOT_N)]
    return (float(np.mean(allv)), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), len(allv), len(units))


def main():
    print("=== #152 instrument-validity gate (prereg K152_KGRADIENT_PREREG.md §4.1) ===")
    print("    criterion: same-wave `indiv` clears its own chance line in 3/3 datasets\n")
    print(f"  {'ds':6s} {'n':>4s} {'owners':>7s} {'acc':>7s} {'95% CI':>18s} {'chance':>8s}  verdict")
    out, allpass = {}, True
    for ds, seed, rd in CELLS:
        pack = ROOT / rd / "naming_k152gate_sig" / "round1"
        os.environ.update({"DATASET": ds, "SEED": str(seed), "KCL": "8", "GROUP": "random",
                           "ARMS": "indiv", "BSEEDS_V6": "1", "BSEEDS_ARM": "",
                           "INDIV_PER_CLUSTER": "2", "BATCHDIR": rd + "/naming_k152gate"})
        for m in [m for m in list(sys.modules) if m in ("naming_export", "cmd_gate")]:
            del sys.modules[m]
        import naming_export as NE

        brackets, _pool, _ref, full, _miss = NE.build_brackets()
        bidmap = {f"{NE.ARM_TAG.get(b['arm'], b['arm'])}{i:03d}": b for i, b in enumerate(brackets)}
        owner_cid = {m_: c for c, mem in full.items() for m_ in mem}
        meta, ans = NE.read_answers(pack)

        by_unit, chances = defaultdict(list), []
        for pid, mt in meta.items():
            ch = ans.get(pid)
            if ch is None or NE.LETTERS.index(ch) >= len(mt["cands"]):
                continue
            target = bidmap[mt["bid"]]["target"]           # indiv arm: the owner
            mem_in = [a for a in mt["cands"] if a == target]
            if not mem_in:
                continue                                   # not signal-bearing
            by_unit[owner_cid[target]].append(int(mt["cands"][NE.LETTERS.index(ch)] == target))
            chances.append(1 / len(mt["cands"]))

        acc, lo, hi, n, nu = boot(by_unit, 0)
        c = float(np.mean(chances))
        clears = lo > c
        stab = {str(s): [round(x, 4) for x in boot(by_unit, s)[:3]] for s in SEEDS}
        allpass &= clears
        out[ds] = {"acc": acc, "ci": [lo, hi], "chance": c, "n": n, "n_clusters": nu,
                   "clears": bool(clears), "stability_seeds": stab}
        print(f"  {ds:6s} {n:>4d} {nu:>7d} {acc:>7.3f} [{lo:>6.3f},{hi:>6.3f}] {c:>8.4f}  "
              f"{'PASS' if clears else '**FAIL**'}")

    print(f"\n  seed stability (acc, lo, hi) over bootstrap seeds {SEEDS}:")
    for ds in out:
        print(f"    {ds:6s} " + "  ".join(f"s{s}:{out[ds]['stability_seeds'][str(s)]}" for s in SEEDS))

    print(f"\n=== GATE: {'PASS 3/3 -> proceed to the main k-gradient wave (prereg §7 step 5)' if allpass else '**FAIL** -> the attacker has no teeth; DO NOT run the main wave, publish the failure as-is'} ===")
    p = ROOT / "results/k152_gate_readout.json"
    p.write_text(json.dumps({"prereg": "results/K152_KGRADIENT_PREREG.md §4.1",
                             "per_dataset": out, "gate_pass": bool(allpass)},
                            ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
