"""R15 — standard MIA readout for the naming tournament (pre-registered: results/NAMING_R15_AUC_PREREG.md).

Every candidate in the pool is scored for every bracket; label = is a true member of that card.
-> AUC (Mann-Whitney, ties 0.5) + TPR@1%/10% FPR + tie-mass diagnostics, cluster bootstrap.
SIX pre-registered scoring variants; the verdicts must agree across ALL of them (disagreement is
reported, never filtered). No new data: uses the `conf` field (50-100) present on 100% of answers.

  DATASET=enron SEED=1 python -P scripts/naming_auc.py
  ARMS=staab,petre_k4,tpar_t15 BATCHDIR=results/enron/naming_deid DATASET=enron SEED=1 \
      python -P scripts/naming_auc.py
"""
import os
import re
import sys
import json
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
BOOT_SEEDS = (0, 1, 2)
VARIANTS = ("S0", "S1", "S2", "S3", "S4", "S5")
W5 = (1.0, 2.0, 4.0)


def read_conf(rdir):
    meta = json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
    ans = {}
    for f in sorted(rdir.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", f.stem):
            continue
        for rec in json.loads(f.read_text(encoding="utf-8")):
            if isinstance(rec, dict) and "pid" in rec:
                m = re.search(r"[A-H]", str(rec.get("choice", "")).upper())
                if m:
                    c = rec.get("conf")
                    ans[rec["pid"]] = (m.group(0), float(c) if isinstance(c, (int, float)) else 50.0)
    return meta, ans


def _auc(s, y):
    """Mann-Whitney AUC with average ranks for ties (fully vectorised)."""
    npos = int(y.sum()); nneg = len(y) - npos
    if not npos or not nneg:
        return None
    order = np.argsort(s, kind="mergesort")
    _u, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    avg = starts[inv] + (counts[inv] + 1) / 2.0          # 1-based average rank
    ranks = np.empty(len(s), float)
    ranks[order] = avg
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def auc_tpr(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    a = _auc(s, y)
    if a is None:
        return None
    pos, neg = s[y == 1], s[y == 0]
    out = {"auc": a, "n_pos": int(len(pos)), "n_neg": int(len(neg))}
    for fpr in (0.01, 0.10):
        out[f"tpr@{int(fpr*100)}"] = float(np.mean(pos > np.quantile(neg, 1 - fpr)))
    return out


def main():
    brackets, pool, _ref, full, _miss = NE.build_brackets()
    bidmap = {f"{NE.ARM_TAG.get(b['arm'], b['arm'])}{i:03d}": b for i, b in enumerate(brackets)}
    arms = [a for a in NE.ALL_ARMS
            if any(b["arm"] == a for b in bidmap.values())]
    owner_cid = {m: cid for cid, mem in full.items() for m in mem}
    members = {bid: (set(full[b["target"]]) if b["arm"] in NE.POOLED else {b["target"]})
               for bid, b in bidmap.items()}
    unit_of = {bid: (b["target"] if b["arm"] in NE.POOLED else owner_cid[b["target"]])
               for bid, b in bidmap.items()}
    rounds = sorted(int(p.name[5:]) for p in NE.BASE.glob("round*") if (p / "meta.json").exists())
    cfg = json.loads((NE.BASE / "round1" / "_config.json").read_text(encoding="utf-8"))
    assert cfg.get("arms", ["v6", "indiv"]) == NE.ARMS, "ARMS drift vs exported config"

    # per (bid, round, candidate): did c win, the group's winner margin, the group size
    st = {}
    for r in rounds:
        meta, ans = read_conf(NE.BASE / f"round{r}")
        for pid, mt in sorted(meta.items()):
            got = ans.get(pid)
            if not got or NE.LETTERS.index(got[0]) >= len(mt["cands"]):
                continue
            w = mt["cands"][NE.LETTERS.index(got[0])]
            marg = (got[1] - 50.0) / 50.0
            for c in mt["cands"]:
                st[(mt["bid"], r, c)] = (c == w, marg, len(mt["cands"]))

    def scores_for(bid, c):
        depth, s0, s2, s3, s5 = 0, 0.0, 0.0, 0.0, 0.0
        lost_m, alive, won_r1 = 0.0, True, 0
        for i, r in enumerate(rounds):
            rec = st.get((bid, r, c))
            if not alive or rec is None:
                break
            wonr, marg, g = rec
            s3 += marg if wonr else (-marg / (g - 1) if g > 1 else 0.0)
            if wonr:
                depth += 1; s0 += marg
                s5 += W5[i] if i < len(W5) else W5[-1]
                if i == 0:
                    won_r1 = 1
            else:
                lost_m = marg; alive = False
        s2 = s0 - lost_m
        return {"S0": s0 + depth, "S1": float(depth), "S2": s2 + depth, "S3": s3,
                "S4": float(won_r1), "S5": s5}

    print(f"\n=== R15 AUC  DS={NE.DS} s{NE.SEED}  arms={arms}  pool={len(pool)} ===")
    out = {"config": cfg, "prereg": "results/NAMING_R15_AUC_PREREG.md", "arms": {}}
    for arm in arms:
        bids = [b for b in bidmap if bidmap[b]["arm"] == arm]
        per_unit = {v: {} for v in VARIANTS}
        for bid in bids:
            for c in pool:
                sc = scores_for(bid, c)
                lab = int(c in members[bid])
                for v in VARIANTS:
                    per_unit[v].setdefault(unit_of[bid], []).append((round(sc[v], 4), lab))
        out["arms"][arm] = {}
        for v in VARIANTS:
            units = list(per_unit[v].values())
            U = [(np.array([a for a, _ in u], float), np.array([b for _, b in u], int)) for u in units]
            sc_all = np.concatenate([u[0] for u in U]); lab_all = np.concatenate([u[1] for u in U])
            base = auc_tpr(sc_all, lab_all)
            cis = {}
            for bseed in BOOT_SEEDS:
                rng = np.random.default_rng(bseed)
                bs = []
                for _ in range(BOOT_N):
                    pick = rng.integers(0, len(U), len(U))
                    a = _auc(np.concatenate([U[i][0] for i in pick]),
                             np.concatenate([U[i][1] for i in pick]))
                    if a is not None:
                        bs.append(a)
                cis[bseed] = [round(float(np.percentile(bs, 2.5)), 4),
                              round(float(np.percentile(bs, 97.5)), 4)]
            floor = sc_all.min()
            tie = float(np.mean(sc_all == floor))
            keep = sc_all > floor
            above = auc_tpr(sc_all[keep], lab_all[keep]) if keep.sum() and lab_all[keep].sum() else None
            rec = {**{k: (round(x, 4) if isinstance(x, float) else x) for k, x in base.items()},
                   "ci": cis[0], "ci_by_seed": cis, "tie_mass_at_floor": round(tie, 4),
                   "auc_above_floor": round(above["auc"], 4) if above else None,
                   "n_units": len(units),
                   "units": {str(u): vals for u, vals in sorted(per_unit[v].items())}}
            out["arms"][arm][v] = rec
            print(f"  {arm:9s} {v}  AUC {base['auc']:.3f} {cis[0]}  "
                  f"TPR@1% {base['tpr@1']:.3f} @10% {base['tpr@10']:.3f}  "
                  f"tie@floor {tie:.2f}  AUC|above {rec['auc_above_floor']}")

    p = NE.BASE / "_naming_auc.json"
    p.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"saved -> {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
