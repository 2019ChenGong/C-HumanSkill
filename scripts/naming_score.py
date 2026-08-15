"""Score the naming-attack tournament (all rounds present in BATCHDIR) -> _naming_summary.json + console table.

Pre-registered readings (see naming_export.py header):
  PRIMARY (illustration): final-pick hit rate per arm + cluster-bootstrap 95% CI vs the chance line.
    indiv arm = POSITIVE-CONTROL GATE: CI low must clear chance, else the whole wave is VOID.
    v6 arm: expected UNDERPOWERED — report the number, never upgrade a null into a claim.
  POWERED cross-checks: (a) round-1 conditional accuracy on groups containing >=1 true member
    (chance = n_members_in_group / group_size); (b) true-member survival depth across rounds.
Statistical unit = cluster (v6: the card's cid; indiv: the owner's cid). Bootstrap n=5000 seed=0.

  DATASET=enron SEED=1 python -P scripts/naming_score.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import naming_export as NE  # noqa: E402  (shares env config, build_brackets, read_answers)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BOOT_N, BOOT_SEED = 5000, 0


def boot(vals_by_unit):
    units = [v for v in vals_by_unit.values() if v]
    allv = [x for u in units for x in u]
    if not allv:
        return None
    rng = np.random.default_rng(BOOT_SEED)
    means = []
    for _ in range(BOOT_N):
        pick = rng.integers(0, len(units), len(units))
        vv = [x for i in pick for x in units[i]]
        means.append(np.mean(vv))
    return (round(float(np.mean(allv)), 3), round(float(np.percentile(means, 2.5)), 3),
            round(float(np.percentile(means, 97.5)), 3), len(allv), len(units))


def main():
    brackets, pool, _ref, full, _miss = NE.build_brackets()
    bidmap = {f"{NE.ARM_TAG.get(b['arm'], b['arm'])}{i:03d}": b for i, b in enumerate(brackets)}
    arms_present = [a for a in NE.ALL_ARMS
                    if any(b["arm"] == a for b in bidmap.values())]
    owner_cid = {m: cid for cid, mem in full.items() for m in mem}
    members = {bid: (set(full[b["target"]]) if b["arm"] in NE.POOLED else {b["target"]}) for bid, b in bidmap.items()}
    unit_of = {bid: (b["target"] if b["arm"] in NE.POOLED else owner_cid[b["target"]]) for bid, b in bidmap.items()}

    rounds = sorted(int(p.name[5:]) for p in NE.BASE.glob("round*") if (p / "meta.json").exists())
    assert rounds, f"no rounds under {NE.BASE}"
    picks = {}                                   # (bid, round) -> [picked author per group]
    r1meta = {}
    for r in rounds:
        meta, ans = NE.read_answers(NE.BASE / f"round{r}")
        if r == 1:
            r1meta = meta
        n_miss = 0
        for pid, mt in sorted(meta.items()):
            ch = ans.get(pid)
            if ch is None or NE.LETTERS.index(ch) >= len(mt["cands"]):
                n_miss += 1; continue
            picks.setdefault((mt["bid"], r), []).append(mt["cands"][NE.LETTERS.index(ch)])
        print(f"round{r}: {len(meta) - n_miss}/{len(meta)} answers usable")

    # final pick per bracket = the single survivor of the last round it appears in
    final = {}
    for bid in bidmap:
        last = None
        for r in rounds:
            w = picks.get((bid, r))
            if w:
                last = w
        if last and len(last) == 1:
            final[bid] = last[0]

    cfg = json.loads((NE.BASE / "round1" / "_config.json").read_text(encoding="utf-8"))
    assert cfg.get("arms", ["v6", "indiv"]) == NE.ARMS, \
        f"env ARMS {NE.ARMS} != exported arms {cfg.get('arms')} — rerun with the matching ARMS/BATCHDIR env"
    out = {"config": cfg, "arms": {}, "round1_conditional": {}, "survival": {}}
    print(f"\n=== NAMING attack  DS={NE.DS} k{NE.KCL} s{NE.SEED}  arms={arms_present}  pool={len(pool)}  "
          f"brackets resolved {len(final)}/{len(bidmap)} ===")
    if "indiv" not in arms_present:
        print("  NOTE: no in-run indiv arm — the pre-registered gate is the POOLED petre_k4 r1-conditional "
              "read across datasets (scripts/naming_pooled_gate.py); per-dataset reads below are transparency.")
    for arm in arms_present:
        by_unit, ch = {}, []
        for bid, p in final.items():
            if bidmap[bid]["arm"] != arm:
                continue
            by_unit.setdefault(unit_of[bid], []).append(int(p in members[bid]))
            ch.append(len(members[bid]) / len(pool))     # enron clusters are 8-9 members -> per-bracket chance
        chance = float(np.mean(ch)) if ch else 0.0
        b = boot(by_unit)
        if not b:
            continue
        acc, lo, hi, n, nu = b
        gate = ""
        if arm == "indiv":
            gate = "  GATE PASS" if lo > chance else "  GATE **VOID** (positive control does not clear chance)"
        out["arms"][arm] = {"hit": acc, "ci": [lo, hi], "chance": round(chance, 4), "n_brackets": n,
                            "n_units": nu, **({"gate_pass": lo > chance} if arm == "indiv" else {})}
        print(f"  {arm:5s}: hit {acc:.3f} [{lo:.3f},{hi:.3f}]  chance {chance:.4f}  "
              f"(n={n} brackets, {nu} clusters){gate}")
        if arm in NE.POOLED:                     # per-bseed transparency (pooled arms only)
            for bs in range(NE._bseeds(arm)):
                sub = [int(final[bid] in members[bid]) for bid in final
                       if bidmap[bid]["arm"] == arm and bidmap[bid]["bseed"] == bs]
                if sub:
                    print(f"         bseed{bs}: hit {np.mean(sub):.3f} (n={len(sub)})")

    # powered read (a): round-1 conditional accuracy on groups containing >=1 true member (per-pid, exact)
    meta1, ans1 = NE.read_answers(NE.BASE / "round1")

    def r1_cond(arm, owner_filter=None):
        acc_u, ch_list = {}, []
        for pid, mt in sorted(meta1.items()):
            if mt["arm"] != arm or (owner_filter is not None and mt["target"] not in owner_filter):
                continue
            mem_in = [a for a in mt["cands"] if a in members[mt["bid"]]]
            ch = ans1.get(pid)
            if not mem_in or ch is None or NE.LETTERS.index(ch) >= len(mt["cands"]):
                continue
            picked = mt["cands"][NE.LETTERS.index(ch)]
            acc_u.setdefault(unit_of[mt["bid"]], []).append(int(picked in members[mt["bid"]]))
            ch_list.append(len(mem_in) / len(mt["cands"]))
        return acc_u, ch_list

    for arm in arms_present:
        acc_u, ch_list = r1_cond(arm)
        b = boot(acc_u)
        if b:
            acc, lo, hi, n, nu = b
            out["round1_conditional"][arm] = {"acc": acc, "ci": [lo, hi],
                                              "chance": round(float(np.mean(ch_list)), 4), "n_groups": n,
                                              "units": {str(u): v for u, v in sorted(acc_u.items())}}
            print(f"  r1-conditional {arm:8s}: acc {acc:.3f} [{lo:.3f},{hi:.3f}]  "
                  f"chance {np.mean(ch_list):.3f}  (n={n} member-containing groups)")

    # petre_k4 no-op subset (owners whose petre card is byte-identical to their indiv card) — the in-run
    # teeth-check + the bridge to the legacy run's indiv arm (same owners, same groups up to slot order)
    if "petre_k4" in arms_present:
        deid = json.loads(NE.DEIDC.read_text(encoding="utf-8"))["petre_k4"]
        _d, _a, nuwa, _ag, _r2, _r3 = NE.CG.load()
        owners = {b["target"] for b in bidmap.values() if b["arm"] == "petre_k4"}
        noop = {m for m in owners if deid.get(m) == nuwa.get(m)}
        acc_u, ch_list = r1_cond("petre_k4", owner_filter=noop)
        b = boot(acc_u)
        if b:
            acc, lo, hi, n, nu = b
            out["petre_noop_subset"] = {"n_noop_owners": len(noop), "acc": acc, "ci": [lo, hi],
                                        "chance": round(float(np.mean(ch_list)), 4), "n_groups": n,
                                        "units": {str(u): v for u, v in sorted(acc_u.items())}}
            print(f"  petre NO-OP subset ({len(noop)}/{len(owners)} owners, card==indiv card): "
                  f"acc {acc:.3f} [{lo:.3f},{hi:.3f}]  chance {np.mean(ch_list):.3f}  (n={n})")
        oldb = Path(os.environ.get("OLDBASE", ROOT / f"results/{NE.DSDIR}/naming_v6"))
        if (oldb / "round1" / "_config.json").exists():
            ocfg = json.loads((oldb / "round1" / "_config.json").read_text(encoding="utf-8"))
            if (ocfg["ds"], ocfg["kcl"], ocfg["seed"]) != (NE.DS, NE.KCL, NE.SEED):
                print(f"  bridge SKIPPED: OLDBASE config {ocfg['ds']}/k{ocfg['kcl']}/s{ocfg['seed']} != current run")
                oldb = Path("__none__")
        if (oldb / "round1" / "meta.json").exists():
            ometa, oans = NE.read_answers(oldb / "round1")
            o_u, o_n = {}, 0
            for pid, mt in sorted(ometa.items()):
                if mt["arm"] != "indiv" or mt["target"] not in noop or mt["target"] not in mt["cands"]:
                    continue
                ch = oans.get(pid)
                if ch is None or NE.LETTERS.index(ch) >= len(mt["cands"]):
                    continue
                o_u.setdefault(owner_cid[mt["target"]], []).append(
                    int(mt["cands"][NE.LETTERS.index(ch)] == mt["target"])); o_n += 1
            ob = boot(o_u)
            if ob:
                out["bridge_legacy_indiv_noop"] = {"acc": ob[0], "ci": [ob[1], ob[2]], "n_groups": ob[3]}
                print(f"  bridge — legacy indiv on the SAME no-op owners: acc {ob[0]:.3f} "
                      f"[{ob[1]:.3f},{ob[2]:.3f}]  (n={ob[3]}; same groups up to slot order)")

    # powered read (b): true-member survival depth per arm
    meta_all = {r: NE.read_answers(NE.BASE / f"round{r}")[0] for r in rounds}
    for arm in arms_present:
        surv = {"eliminated_r1": 0, "reached_r2": 0, "reached_r3": 0, "named": 0}
        for bid, b in bidmap.items():
            if b["arm"] != arm or bid not in final:
                continue
            for m in members[bid]:
                depth = "eliminated_r1"
                for r in rounds[1:]:
                    if any(m in mt["cands"] for mt in meta_all[r].values() if mt["bid"] == bid):
                        depth = f"reached_r{r}"
                if final[bid] == m:
                    depth = "named"
                surv[depth] = surv.get(depth, 0) + 1
        out["survival"][f"{arm}_members"] = surv
        print(f"  {arm} member survival: {surv}")

    outp = NE.BASE / "_naming_summary.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {outp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
