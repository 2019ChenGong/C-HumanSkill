"""R14 re-analysis (pre-registered in results/NAMING_R14_PREREG.md, frozen 2026-07-28 BEFORE any run).

Ordinal survival-depth readout for the naming tournament + per-unit final-hit values for pooling.
NO new data, NO cost: pure re-scoring of the already-frozen answers.

Analysis A (depth):
  d_m in {0..R} = rounds the true member m survived in its bracket.
  EXACT structural null: with G_r groups in round r and pool size N, a random player survives
  through round r with prob prod_{j<=r} G_j/N_j = G_r/N (since N_1=N, N_r=G_{r-1}), so
      E[d | random] = (sum_r G_r) / N.
  Self-check: P[d=R] = G_R/N = 1/N = the final-hit chance line already in use.
  Primary statistic = excess depth e_m = d_m - E[d]. Unit = cluster. Cluster bootstrap n=5000 seed=0.
  SIGNAL iff CI-lo > 0. indiv = positive-control gate.

Analysis B feed (final):
  per-unit final-hit 0/1 values + per-bracket chance, so scripts/naming_pooled_depth.py can pool
  ds:cluster units across the 3 datasets. That pooled read is POST-HOC (see prereg B.3).

  DATASET=enron SEED=1 python -P scripts/naming_depth.py
  DATASET=mad SEED=0 ARMS=v6,indiv,staab,petre_k4,tpar_t15 BATCHDIR=results/mad/naming_deid \
      python -P scripts/naming_depth.py
  DRY=1 ... -> compute + print, write nothing.
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
DRY = os.environ.get("DRY", "") not in ("", "0")


def boot(vals_by_unit):
    """Cluster bootstrap over unit -> list of per-observation values. Returns (mean, lo, hi, n, n_units)."""
    units = [v for v in vals_by_unit.values() if v]
    allv = [x for u in units for x in u]
    if not allv:
        return None
    rng = np.random.default_rng(BOOT_SEED)
    means = []
    for _ in range(BOOT_N):
        pick = rng.integers(0, len(units), len(units))
        means.append(np.mean([x for i in pick for x in units[i]]))
    return (float(np.mean(allv)), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), len(allv), len(units))


def main():
    brackets, pool, _ref, full, _miss = NE.build_brackets()
    bidmap = {f"{NE.ARM_TAG.get(b['arm'], b['arm'])}{i:03d}": b for i, b in enumerate(brackets)}
    arms_present = [a for a in NE.ALL_ARMS
                    if any(b["arm"] == a for b in bidmap.values())]
    owner_cid = {m: cid for cid, mem in full.items() for m in mem}
    members = {bid: (set(full[b["target"]]) if b["arm"] in NE.POOLED else {b["target"]})
               for bid, b in bidmap.items()}
    unit_of = {bid: (b["target"] if b["arm"] in NE.POOLED else owner_cid[b["target"]])
               for bid, b in bidmap.items()}
    N = len(pool)

    rounds = sorted(int(p.name[5:]) for p in NE.BASE.glob("round*") if (p / "meta.json").exists())
    assert rounds, f"no rounds under {NE.BASE}"
    cfg = json.loads((NE.BASE / "round1" / "_config.json").read_text(encoding="utf-8"))
    assert cfg.get("arms", ["v6", "indiv"]) == NE.ARMS, \
        f"env ARMS {NE.ARMS} != exported arms {cfg.get('arms')} — rerun with the matching ARMS/BATCHDIR env"
    assert cfg.get("pool_n") == N, f"pool drift: config {cfg.get('pool_n')} != rebuilt {N}"

    cands_by, ngroups, picks, gsize = {}, {}, {}, {}
    for r in rounds:
        meta, ans = NE.read_answers(NE.BASE / f"round{r}")
        for pid, mt in sorted(meta.items()):
            cands_by.setdefault((mt["bid"], r), set()).update(mt["cands"])
            ngroups[(mt["bid"], r)] = ngroups.get((mt["bid"], r), 0) + 1
            for c in mt["cands"]:                      # exact per-member round-1 group size
                gsize[(mt["bid"], r, c)] = len(mt["cands"])
            ch = ans.get(pid)
            if ch is None or NE.LETTERS.index(ch) >= len(mt["cands"]):
                continue
            picks.setdefault((mt["bid"], r), []).append(mt["cands"][NE.LETTERS.index(ch)])

    final = {}
    for bid in bidmap:
        last = None
        for r in rounds:
            w = picks.get((bid, r))
            if w:
                last = w
        if last and len(last) == 1:
            final[bid] = last[0]

    out = {"config": cfg, "prereg": "results/NAMING_R14_PREREG.md", "N": N,
           "rounds": rounds, "arms": {}}
    print(f"\n=== R14 depth  DS={NE.DS} k{NE.KCL} s{NE.SEED}  arms={arms_present}  pool={N}  "
          f"brackets resolved {len(final)}/{len(bidmap)} ===")

    for arm in arms_present:
        dep_u, obs, nul = {}, [], []
        dep_id = {}          # A12: row LABELS parallel to dep_u — provenance only, no number depends on it
        fin_u, fin_id, chances = {}, {}, []
        n_short = 0
        for bid, b in bidmap.items():
            if b["arm"] != arm or bid not in final:
                continue
            G = [ngroups.get((bid, r), 0) for r in rounds]
            if 0 in G:
                n_short += 1; continue                 # bracket missing a round -> not comparable
            for m in sorted(members[bid]):
                # exact null: p1 from m's ACTUAL round-1 group size; p_j>=2 = G_j / G_{j-1}
                ps = [1.0 / gsize[(bid, rounds[0], m)]] + [G[j] / G[j - 1] for j in range(1, len(rounds))]
                enull, cum = 0.0, 1.0
                for pj in ps:
                    cum *= pj; enull += cum
                d = 0
                for i, r in enumerate(rounds):
                    if m not in cands_by.get((bid, r), set()):
                        break
                    nxt = rounds[i + 1] if i + 1 < len(rounds) else None
                    alive = (m in cands_by.get((bid, nxt), set())) if nxt else (final[bid] == m)
                    if not alive:
                        break
                    d += 1
                dep_u.setdefault(unit_of[bid], []).append(d - enull)
                dep_id.setdefault(unit_of[bid], []).append(f"{m}#{b['bseed']}")
                obs.append(d); nul.append(enull)
            fin_u.setdefault(unit_of[bid], []).append(int(final[bid] in members[bid]))
            fin_id.setdefault(unit_of[bid], []).append(f"{b['target']}#{b['bseed']}")
            chances.append(len(members[bid]) / N)

        bd, bf = boot(dep_u), boot(fin_u)
        if not bd or not bf:
            continue
        exc, elo, ehi, nm, nu = bd
        hit, hlo, hhi, nb, nu2 = bf
        chance = float(np.mean(chances))
        sig = elo > 0
        gate = ("  GATE PASS" if sig else "  GATE **UND**") if arm == "indiv" else ""
        out["arms"][arm] = {
            "depth": {"mean_obs": round(float(np.mean(obs)), 4), "mean_null": round(float(np.mean(nul)), 4),
                      "excess": round(exc, 4), "ci": [round(elo, 4), round(ehi, 4)],
                      "n_members": nm, "n_units": nu, "signal": bool(sig),
                      "units": {str(u): [round(x, 4) for x in v] for u, v in sorted(dep_u.items())},
                      "row_ids": {str(u): v for u, v in sorted(dep_id.items())}},
            "final": {"hit": round(hit, 4), "ci": [round(hlo, 4), round(hhi, 4)],
                      "chance": round(chance, 5), "n_brackets": nb, "n_units": nu2,
                      "units": {str(u): v for u, v in sorted(fin_u.items())},
                      "row_ids": {str(u): v for u, v in sorted(fin_id.items())}, "n_short_dropped": n_short,
                      "chances": [round(c, 5) for c in chances]},
        }
        print(f"  {arm:9s} depth  obs {np.mean(obs):.3f} vs null {np.mean(nul):.3f}  "
              f"excess {exc:+.3f} [{elo:+.3f},{ehi:+.3f}]  (n={nm} members, {nu} clusters){gate}")
        print(f"  {'':9s} final  hit {hit:.3f} [{hlo:.3f},{hhi:.3f}]  chance {chance:.4f}  "
              f"(n={nb} brackets" + (f", {n_short} dropped for missing rounds)" if n_short else ")"))

    if DRY:
        print("\nDRY=1 — nothing written")
        return
    outp = NE.BASE / "_naming_depth.json"
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {os.path.relpath(outp, ROOT)}")


if __name__ == "__main__":
    main()
