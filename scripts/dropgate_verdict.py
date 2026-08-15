"""Drop-gate experiment readout — executes the frozen rules of
`results/DROPGATE_THRESHOLD_PREREG.md`.  It adds no criterion of its own.

★ ARM RELABELLING (results/mad/dropgate_gate_sig/ARM_MAPPING.md)
The exporter only has three pooled slots (v6/ne/concat), so this wave borrows two of them:
    slot `v6` -> the ABLATED arm      (A20-t / A10-t / A10-r / ...)
    slot `ne` -> the UNABLATED BASE   (A0 = canonical v6, 3.3% pipeline drop)
Both slots share the pooled bracket salt, so for a given (cluster, bseed) they face byte-identical
brackets -> the difference is paired by construction, with no cross-wave licence needed.
This script prints the REAL names, never the slot names, so a reader cannot mistake the ablated
arm for the canonical card.

READS (prereg §4)
  gate-1  same-wave `indiv` positive control must clear its chance line (.1250)
  gate-2  ABL vs BASE must be DETECTED (paired CI excludes 0)
  -> either gate failing VOIDS the wave; the threshold stays 5% and "no power" is published as-is.

  ABL=A20-t BASE=A0 DATASET=mad SEED=0 KCL=8 INDIV_PER_CLUSTER=99 BSEEDS_V6=2 BSEED_SKIP=0 \
  ARMS=v6,ne,indiv V6C=cmd_shared_cards_mad__abl20t.json NEC=cmd_shared_cards_mad__v6min.json \
  BATCHDIR=results/mad/dropgate_gate PACK=results/mad/dropgate_gate_sig \
  python -P scripts/dropgate_verdict.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import naming_export as NE  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BOOT_N = 5000
SEEDS = (0, 1, 2)
PACK = ROOT / os.environ.get("PACK", "results/mad/dropgate_gate_sig")
LABEL = {"v6": os.environ.get("ABL", "ABL"), "ne": os.environ.get("BASE", "BASE"), "indiv": "indiv"}


def boot(by_unit, seed):
    units = [v for v in by_unit.values() if v]
    allv = [x for u in units for x in u]
    if not allv:
        return None
    rng = np.random.default_rng(seed)
    means = [np.mean([x for i in rng.integers(0, len(units), len(units)) for x in units[i]])
             for _ in range(BOOT_N)]
    return (float(np.mean(allv)), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), len(allv), len(units))


def main():
    brackets, pool, _ref, full, _miss = NE.build_brackets()
    bidmap = {f"{NE.ARM_TAG.get(b['arm'], b['arm'])}{i:03d}": b for i, b in enumerate(brackets)}
    owner_cid = {m: c for c, mem in full.items() for m in mem}
    members = {bid: (set(full[b["target"]]) if b["arm"] in NE.POOLED else {b["target"]})
               for bid, b in bidmap.items()}
    unit_of = {bid: (b["target"] if b["arm"] in NE.POOLED else owner_cid[b["target"]])
               for bid, b in bidmap.items()}

    meta, ans = NE.read_answers(PACK / "round1")
    n_ans = sum(1 for p in meta if ans.get(p) is not None)
    print(f"=== drop-gate readout  {PACK.relative_to(ROOT)}  {n_ans}/{len(meta)} answered ===")
    print(f"    slot v6 = {LABEL['v6']} (ablated)   slot ne = {LABEL['ne']} (unablated base)")
    if n_ans < len(meta):
        raise SystemExit(f"[HALT] {len(meta)-n_ans} unanswered — finish the pack before reading "
                         f"(a partial read is a second, undeclared look).")

    per_arm, per_card, chance = {}, {}, {}
    for pid, mt in sorted(meta.items()):
        ch = ans.get(pid)
        mem = members[mt["bid"]]
        mem_in = [a for a in mt["cands"] if a in mem]
        if not mem_in or ch is None or NE.LETTERS.index(ch) >= len(mt["cands"]):
            continue
        hit = int(mt["cands"][NE.LETTERS.index(ch)] in mem)
        per_arm.setdefault(mt["arm"], {}).setdefault(unit_of[mt["bid"]], []).append(hit)
        chance.setdefault(mt["arm"], []).append(len(mem_in) / len(mt["cands"]))
        if mt["arm"] in NE.POOLED:
            # Pair on (card, bracket-seed, group) — NOT on `bid`, which carries the arm tag
            # ("v6000" vs "ne032") and therefore never matches across slots. The pooled arms share
            # the bracket salt, so identical (target, bseed, gidx) == byte-identical question.
            per_card.setdefault(mt["arm"], {}).setdefault(
                (mt["target"], mt["bseed"], mt["gidx"]), []).append(hit)

    print("\n--- per-arm round-1-conditional accuracy ---")
    print(f"  {'arm':<8} {'acc':>6} {'95% CI':>18} {'chance':>8} {'n':>5} {'units':>6}")
    res = {}
    for slot in ("indiv", "ne", "v6"):
        if slot not in per_arm:
            continue
        acc, lo, hi, n, nu = boot(per_arm[slot], 0)
        c = float(np.mean(chance[slot]))
        res[LABEL[slot]] = {"acc": acc, "ci": [lo, hi], "chance": c, "n": n, "n_units": nu,
                            "above_chance": lo > c}
        print(f"  {LABEL[slot]:<8} {acc:>6.3f} [{lo:>6.3f},{hi:>6.3f}] {c:>8.4f} {n:>5} {nu:>6}")

    # ---- gate 1: positive control -------------------------------------------------------
    g1 = res.get("indiv", {}).get("above_chance", False)
    print(f"\n--- GATE 1  positive control `indiv` clears chance: "
          f"{'PASS' if g1 else '**FAIL**'} ---")

    # ---- gate 2: paired ABL vs BASE (same card, same bracket) ---------------------------
    print(f"--- GATE 2  {LABEL['v6']} vs {LABEL['ne']}, paired on (card, bracket) ---")
    g2 = None
    if "v6" in per_card and "ne" in per_card:
        base = {k: np.mean(v) for k, v in per_card["ne"].items()}
        by_unit = {}
        common = 0
        for k, v in per_card["v6"].items():
            if k in base:
                by_unit.setdefault(k[0], []).append(np.mean(v) - base[k]); common += 1
        b = boot(by_unit, 0)
        d, lo, hi, n, nu = b
        detected = lo > 0 or hi < 0
        g2 = {"diff": d, "ci": [lo, hi], "n_pairs": n, "n_units": nu, "detected": detected,
              "stability_seeds": {str(s): [round(x, 4) for x in boot(by_unit, s)[:3]] for s in SEEDS}}
        arrow = "more nameable" if d > 0 else "less nameable (MORE anonymous)"
        print(f"  diff = {d:+.4f} [{lo:+.4f},{hi:+.4f}]  ({n} paired brackets, {nu} cards)")
        print(f"  point estimate direction: ablated is {arrow}")
        print(f"  DETECTED: {'YES' if detected else 'NO'}")
        assert common == len(per_card["v6"]), "pairing incomplete — brackets differ between slots"

    ok = g1 and (g2 or {}).get("detected")
    print(f"\n=== VERDICT: {'GATES PASS -> proceed to the A10-t main read (prereg §4.2)' if ok else '**WAVE VOID** -> threshold stays 5%, publish "no power" as-is (prereg §4.1)'} ===")
    out = {"pack": str(PACK.relative_to(ROOT)).replace(os.sep, "/"),
           "arm_mapping": {"slot_v6": LABEL["v6"], "slot_ne": LABEL["ne"]},
           "prereg": "results/DROPGATE_THRESHOLD_PREREG.md §4.1",
           "per_arm": res, "gate1_positive_control": g1, "gate2_paired": g2, "gates_pass": bool(ok)}
    p = ROOT / f"results/dropgate_gate_{LABEL['v6']}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
