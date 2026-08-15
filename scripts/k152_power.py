"""#152 k-gradient POWER CALCULATION — run BEFORE freezing the design.

WHY THIS EXISTS: the drop-gate wave (results/DROPGATE_THRESHOLD_PREREG.md §4.1) came back VOID because
the design was frozen before anyone computed what it could resolve — paired-diff CI half-width +-.069
against a signal of .024.  That mistake is not to be repeated.  This script measures the naming ruler's
ACTUAL cluster-level dispersion from already-answered packs and projects it onto the k-gradient's
cluster counts.

METHOD (no new measurement, no cost):
  1. Re-derive per-CARD round-1-conditional excess (hit-rate minus that card's own chance line) from the
     A12 `naming_pool_ab` packs, which are 100% answered on 3 datasets x 3 pooled arms.
  2. Take the between-card SD of that excess.  That SD *is* the limiting quantity: the naming ruler
     cluster-bootstraps over cards, so precision is governed by CARD COUNT, not question count.
  3. Project the 95% CI half-width onto every (dataset, k) cell of the frozen ladder, whose card count
     is pool_n / k by construction.

  ARMS=v6,ne,concat python -P scripts/k152_power.py
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

BOOT_N = 5000
PACKS = [("mad", "results/mad/naming_pool_ab", 0, 128),
         ("enron", "results/enron/naming_pool_ab", 1, 116),
         ("cv", "results/se/naming_pool_ab", 0, 77)]
LADDER = (4, 6, 8, 10, 12)


def per_card(ds, packdir, seed, arms):
    """card_id -> {arm: (excess, n_questions)} for round-1-conditional groups."""
    # Reproduce the pack's export env EXACTLY from its own _config.json -- guessing it would silently
    # build a different bracket list and the bid lookup would KeyError (it did, on `cc112`).
    cfg = json.loads((ROOT / packdir / "round1" / "_config.json").read_text(encoding="utf-8"))
    os.environ.update({"DATASET": ds, "SEED": str(seed), "KCL": str(cfg["kcl"]), "GROUP": "random",
                       "ARMS": ",".join(cfg["arms"]), "BATCHDIR": packdir,
                       "BSEEDS_V6": str(cfg["bseeds_v6"]),
                       "BSEEDS_ARM": ",".join(f"{a}:{n}" for a, n in cfg.get("bseeds_arm", {}).items()),
                       "INDIV_PER_CLUSTER": str(cfg["indiv_per_cluster"])})
    for m in [m for m in list(sys.modules) if m in ("naming_export", "cmd_gate")]:
        del sys.modules[m]
    import naming_export as NE

    brackets, _pool, _ref, full, _miss = NE.build_brackets()
    bidmap = {f"{NE.ARM_TAG.get(b['arm'], b['arm'])}{i:03d}": b for i, b in enumerate(brackets)}
    members = {bid: (set(full[b["target"]]) if b["arm"] in NE.POOLED else {b["target"]})
               for bid, b in bidmap.items()}
    meta, ans = NE.read_answers(Path(ROOT / packdir) / "round1")

    acc = defaultdict(lambda: defaultdict(list))   # arm -> card -> [(hit, chance, bseed)]
    for pid, mt in meta.items():
        ch = ans.get(pid)
        if mt["arm"] not in NE.POOLED or ch is None:
            continue
        mem = members[mt["bid"]]
        mem_in = [a for a in mt["cands"] if a in mem]
        if not mem_in or NE.LETTERS.index(ch) >= len(mt["cands"]):
            continue                                    # not signal-bearing / unparsable
        acc[mt["arm"]][mt["target"]].append(
            (int(mt["cands"][NE.LETTERS.index(ch)] in mem), len(mem_in) / len(mt["cands"]),
             mt["bseed"]))
    return acc


def _cell_shape(ds, seed, k):
    """(#signal-bearing r1 groups, their chance lines, #cards, #r1 groups) for one (ds, k) cell.

    Deterministic -- built from the same bracket machinery the exporter uses, so the numbers are the
    wave that WOULD be exported.  Needs no answers and costs nothing.
    """
    os.environ.update({"DATASET": ds, "SEED": str(seed), "KCL": str(k), "GROUP": "random",
                       "ARMS": "v6", "BSEEDS_V6": "1", "BSEEDS_ARM": "", "INDIV_PER_CLUSTER": "1",
                       "BATCHDIR": f"results/_k152_plan_{ds}_k{k}"})
    for m in [m for m in list(sys.modules) if m in ("naming_export", "cmd_gate")]:
        del sys.modules[m]
    import naming_export as NE
    brackets, pool, _ref, full, _miss = NE.build_brackets()
    nsig, chances, ngrp = 0, [], 0
    for b in brackets:                                   # ARMS=v6, so every bracket is a pooled card
        mem = set(full[b["target"]])
        # exactly the exporter's round-1 candidate construction (_shuf + _partition into 8s)
        cands = NE._shuf(pool, NE._salt_arm(b["arm"]), b["target"], b["bseed"], "r1")
        for grp in NE._partition(cands, 8):
            ngrp += 1
            hits = [a for a in grp if a in mem]
            if hits:
                nsig += 1
                chances.append(len(hits) / len(grp))
    return nsig, chances, len({b["target"] for b in brackets}), ngrp


def excess(rows):
    return float(np.mean([h for h, _, _ in rows]) - np.mean([q for _, q, _ in rows]))


def sd_at_seeds(cards, nseeds):
    """between-card SD of excess when only the first `nseeds` bracket seeds are used."""
    vals = []
    for rows in cards.values():
        sub = [r for r in rows if r[2] < nseeds]
        if sub:
            vals.append(excess(sub))
    return (float(np.std(vals, ddof=1)), len(vals)) if len(vals) > 1 else (float("nan"), len(vals))


def main():
    arms = os.environ.get("ARMS", "v6,ne,concat").split(",")
    print("=== #152 POWER: cluster-level dispersion of the naming ruler (from answered A12 packs) ===\n")
    allpc = {ds: per_card(ds, pd_, sd_, arms) for ds, pd_, sd_, _ in PACKS}

    print(f"  {'ds':6s} {'arm':7s} {'cards':>5s} {'q/card':>7s} {'bseeds':>7s} {'mean excess':>12s} "
          f"{'SD(card)':>9s} {'->CI hw':>9s}")
    sd_by_arm = defaultdict(list)
    for ds, _p, _s, _n in PACKS:
        for arm in arms:
            cards = allpc[ds].get(arm)
            if not cards:
                continue
            vals = np.array([excess(v) for v in cards.values()])
            qs = np.mean([len(v) for v in cards.values()])
            nb = max(r[2] for v in cards.values() for r in v) + 1
            sd = float(np.std(vals, ddof=1))
            sd_by_arm[arm].append((ds, sd, len(vals)))
            print(f"  {ds:6s} {arm:7s} {len(vals):5d} {qs:7.1f} {nb:7d} {vals.mean():+12.4f} "
                  f"{sd:9.4f} {'+-'+format(1.96*sd/np.sqrt(len(vals)),'.4f'):>9s}")

    # ---- ★ variance decomposition: does adding bracket seeds buy precision, or is the SD real
    #         card-to-card heterogeneity (a floor no amount of free subagent labour can lower)?
    print("\n--- ★ VARIANCE DECOMPOSITION  Var(n_seeds) = A + B/n   (ne+concat, 4 seeds available) ---")
    print(f"  {'arm':7s} " + "".join(f"{'n=' + str(n):>10s}" for n in (1, 2, 3, 4))
          + f"{'A (floor)':>12s}{'SD floor':>10s}")
    floors, BFIT = {}, {}
    for arm in ("ne", "concat"):
        obs = []
        for n in (1, 2, 3, 4):
            v = [sd_at_seeds(allpc[ds][arm], n)[0] ** 2 for ds, _p, _s, _x in PACKS if arm in allpc[ds]]
            obs.append(float(np.mean(v)))
        # least squares on Var = A + B*(1/n)
        x = np.array([1 / n for n in (1, 2, 3, 4)]); y = np.array(obs)
        B, A = np.polyfit(x, y, 1)
        floors[arm] = max(A, 0.0); BFIT[arm] = max(B, 0.0)
        print(f"  {arm:7s} " + "".join(f"{o:10.5f}" for o in obs)
              + f"{A:12.5f}{np.sqrt(max(A,0)):10.4f}")

    SD_FLOOR = float(np.sqrt(np.mean([floors[a] for a in floors])))
    SD = pooled_v6 = float(np.sqrt(sum((n - 1) * s ** 2 for _, s, n in sd_by_arm["v6"])
                                   / sum(n - 1 for _, _, n in sd_by_arm["v6"])))
    print(f"\n  v6 observed SD @1 seed = {SD:.4f}   |   ne/concat extrapolated SD floor "
          f"(n_seeds -> inf) = {SD_FLOOR:.4f}")
    print("  ⇒ bracket seeds are FREE (subagent labour) and DO buy precision, but only down to the floor.")
    pooled = {a: float(np.sqrt(sum((n - 1) * s ** 2 for _, s, n in r) / sum(n - 1 for _, _, n in r)))
              for a, r in sd_by_arm.items()}
    SD = SD_FLOOR   # the projection uses the ACHIEVABLE floor, not the 1-seed number

    # ACTUAL card counts per cell, from the bracket machinery -- pool_n/k over-counts because the
    # remainder folds into a bottom group (R13 partition note), so never estimate what you can build.
    CARDS = {}
    for k in LADDER:
        for ds, _p, _cs, _pn in PACKS:
            try:
                CARDS[(k, ds)] = _cell_shape(ds, 0, k)[2]
            except Exception:
                CARDS[(k, ds)] = 0
    print(f"\n=== PROJECTION onto the frozen ladder k in {LADDER} (SD={SD:.4f}, v6 arm) ===")
    print("    cards per cell = pool_n / k BY CONSTRUCTION -- more questions cannot fix this.\n")
    print(f"  {'k':>3s} " + "".join(f"{n:>14s}" for n, _ in [('MAD(128)', 0), ('Enron(116)', 0), ('CV(77)', 0)])
          + f"{'3-家合并':>16s}")
    tot_by_k = {}
    for k in LADDER:
        cells, cols = [], ""
        for nm, ds in [("MAD", "mad"), ("Enron", "enron"), ("CV", "cv")]:
            nc = CARDS[(k, ds)]
            cells.append(nc)
            cols += f"{nc:4d}张 +-{1.96*SD/np.sqrt(max(nc,1)):.3f}"
        tot = sum(cells)
        tot_by_k[k] = tot
        print(f"  {k:>3d} {cols}   {tot:4d}张 +-{1.96*SD/np.sqrt(tot):.3f}")

    print("\n--- what a k4-vs-k12 endpoint contrast can resolve (3 datasets pooled, unpaired) ---")
    n4, n12 = tot_by_k[4], tot_by_k[12]
    se = SD * np.sqrt(1 / n4 + 1 / n12)
    print(f"  n(k4)={n4}  n(k12)={n12}   SE(diff) = {se:.4f}")
    print(f"  95% CI half-width          = +-{1.96*se:.4f}")
    print(f"  sMDE (80% power, two-sided)= {2.8 * se:.4f}   <- smallest k-effect this design can call")
    print(f"  sMDE (95% CI excludes 0 at truth=effect) ~ {1.96*se:.4f}")

    # ---- ★ linear-trend test over ALL 5 cells beats the endpoint contrast (uses every card) ----
    ks = np.array(LADDER, dtype=float)
    ncell = np.array([tot_by_k[int(k)] for k in ks], dtype=float)
    kbar = float((ks * ncell).sum() / ncell.sum())
    sxx = float((ncell * (ks - kbar) ** 2).sum())
    se_slope = SD / np.sqrt(sxx)
    span = ks.max() - ks.min()
    print(f"\n--- ★ linear trend over all {int(ncell.sum())} cards (primary, better than endpoints) ---")
    print(f"  Sxx = {sxx:.1f}   SE(slope) = {se_slope:.5f} per unit k")
    print(f"  over the k4->k12 span:  95% CI half-width = +-{1.96*se_slope*span:.4f}   "
          f"sMDE(80%) = {2.8*se_slope*span:.4f}")

    # ---- ★ per-k round-1 chance line + signal-bearing question counts (deterministic, no answers
    #         needed) -- the chance line RISES with k because a bigger card owns more of the 8 slots.
    # ---- ★ what the trend test can resolve AT A REALISTIC BRACKET-SEED COUNT, and what buying more
    #         BUILD seeds (fresh partitions => fresh CARDS, the binding quantity) would add.
    print("\n--- ★ sMDE vs the two levers (bracket seeds = free; build seeds = $) ---")
    Bfit = float(np.mean([b for b in (BFIT.values())]))
    print(f"  fitted Var(n_bracket_seeds) = {SD_FLOOR**2:.5f} + {Bfit:.5f}/n\n")
    print(f"  {'bseeds':>7s} {'SD':>7s} | {'s0 only (226 cards)':>26s} | "
          f"{'+ s1,s2 at k10,k12':>26s}")
    print(f"  {'':7s} {'':7s} | {'CI hw':>11s}{'sMDE80':>13s} | {'CI hw':>11s}{'sMDE80':>13s}")
    EXTRA = {10: 2, 12: 2}          # extra build seeds purchased at these k
    for nb in (2, 4, 8, 16):
        sd = float(np.sqrt(SD_FLOOR ** 2 + Bfit / nb))
        row = f"  {nb:7d} {sd:7.4f} |"
        for extra in ({}, EXTRA):
            nk = {k: sum(CARDS[(k, d)] for d in ("mad", "enron", "cv")) * (1 + extra.get(k, 0))
                  for k in LADDER}
            kk = np.array(LADDER, float); nn = np.array([nk[k] for k in LADDER], float)
            kb = (kk * nn).sum() / nn.sum()
            sxx_ = (nn * (kk - kb) ** 2).sum()
            se_ = sd / np.sqrt(sxx_) * (kk.max() - kk.min())
            row += f"{1.96*se_:11.4f}{2.8*se_:13.4f} |"
        print(row)
    ncards_extra = sum(CARDS[(k, d)] * n for k, n in EXTRA.items() for d in ("mad", "enron", "cv"))
    print(f"\n  '+ s1,s2 at k10,k12' = {ncards_extra} extra cards to BUILD "
          f"(base->neutral->degen-fix->V6). Price it before committing.")

    print("\n--- ★ per-k round-1 chance line + wave size (built from the actual bracket machinery) ---")
    print(f"  {'k':>3s} {'ds':6s} {'cards':>5s} {'r1 groups':>10s} {'signal-bearing':>15s} "
          f"{'mean chance':>12s} {'q @BSEEDS':>10s}")
    BS = int(os.environ.get("PLAN_BSEEDS", 4))
    # ★ THE WHOLE LADDER RUNS ON BUILD SEED 0. Enron's canonical k8 is s1, but s1 exists only at k=8;
    # mixing seeds across cells would confound k with the build partition. s0 exists at every cell of
    # the ladder on all three datasets, so s0 is the only choice that keeps k the sole varying factor.
    plan = {}
    for k in LADDER:
        for ds, _p, _canon_seed, pool_n in PACKS:
            seed = 0
            try:
                nsig, chances, ncards, ngrp = _cell_shape(ds, seed, k)
            except Exception as e:                       # cell not built / not in _KGRID
                print(f"  {k:>3d} {ds:6s}  -- unavailable: {type(e).__name__}: {e}")
                continue
            plan[f"k{k}_{ds}"] = {"cards": ncards, "r1_groups": ngrp, "signal": nsig,
                                  "chance": float(np.mean(chances)), "q_at_bseeds": nsig * BS}
            print(f"  {k:>3d} {ds:6s} {ncards:5d} {ngrp:10d} {nsig:15d} {np.mean(chances):12.4f} "
                  f"{nsig*BS:10d}")
    tot_q = sum(v["q_at_bseeds"] for v in plan.values())
    print(f"\n  TOTAL round-1 signal-bearing questions @ BSEEDS={BS}: {tot_q}"
          f"   (~{tot_q/15:.0f} batches @TPB=15)")

    print("\n--- reference points (what magnitudes exist on this ruler) ---")
    print("  indiv (per-person card) vs chance ....... ~+.30 to +.45   (positive control, huge)")
    print("  A12 ne-v6 paired ........................ +.0125 [-.0164,+.0440]  (n=16+14+9 cards)")
    print("  A12 concat-ne paired .................... -.0405 [-.0881,+.0047]")
    print("  R17 v6 residual above chance ............ +.041  [+.003,+.083]")
    out = {"pooled_sd_between_cards": pooled, "ladder": list(LADDER),
           "cards_per_cell": {str(k): {nm: pool // k + (1 if pool % k else 0)
                                       for nm, pool in [("mad", 128), ("enron", 116), ("cv", 77)]}
                              for k in LADDER},
           "pooled_cards_by_k": {str(k): v for k, v in tot_by_k.items()},
           "endpoint_contrast": {"n_k4": n4, "n_k12": n12, "se": se,
                                 "ci_halfwidth": 1.96 * se, "smde_80pct": 2.8 * se}}
    p = ROOT / "results/k152_power.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
