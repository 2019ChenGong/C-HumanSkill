"""Probe the k=8 utility win: replicate shared@8 minus individual and test shared@8 vs shared@4 head-to-head."""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
import enron_nuwa as NW  # noqa: E402
import cmd_gate as CG  # noqa: E402
from src.attrib_metrics import cluster_mean_ci  # noqa: E402

SE = ROOT / "data" / "enron"
RES = ROOT / "results"
SHAREDC = SE / "cmd_shared_cards.json"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
MSEED = SEEDS[0]                                    # mechanism (sh8 vs sh4) on one seed
T = de.TASKS


def tl(c):
    return len(NW.ENC.encode(c or ""))


def main():
    docs, authors, nuwa, aggro, ref, raw_tgt = CG.load()
    cache = json.loads(SHAREDC.read_text(encoding="utf-8")) if SHAREDC.exists() else {}

    lay = {}                                        # (k,seed) -> (grp, byc); sh8 over SEEDS, sh4 over [MSEED]
    plan = []
    for (k, ss) in [(8, SEEDS), (4, [MSEED])]:
        for s in ss:
            grp, byc = CG.make_groups(aggro, authors, k, s)
            lay[(k, s)] = (grp, byc)
            for cid, mem in byc.items():
                ck = f"k{k}_s{s}_{cid}"
                if ck not in cache:
                    plan.append((ck, [aggro[a] for a in mem]))

    n_cl8 = sum(len(lay[(8, s)][1]) for s in SEEDS)
    if os.environ.get("PILOT_DRYRUN"):
        n_draft = 116 * len(T) + n_cl8 * len(T) + len(lay[(4, MSEED)][1]) * len(T)   # indiv + sh8(all seeds) + sh4(MSEED)
        n_judge = 116 * len(T) * len(SEEDS) + 116 * len(T)                            # sh8-indiv ×seeds + sh8-sh4 ×1
        cost = n_draft * 1700 / 1e6 * 0.6 + n_judge * 1400 / 1e6 * 1.0
        print(f"DRYRUN: synth {len(plan)} new cards; drafts≈{n_draft}; judge≈{n_judge} "
              f"(sh8-indiv ×{len(SEEDS)} + sh8-sh4 ×1, ×116×{len(T)}); est ~${cost:.1f}", flush=True)
        return

    if plan:
        print(f"synth {len(plan)} shared cards ...", flush=True)
        for (ck, _), card in zip(plan, de.pool(lambda pc: CG.synth_shared(pc[1]), plan)):
            cache[ck] = card
        SHAREDC.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    def shared(k, s, a):
        return cache[f"k{k}_s{s}_{lay[(k, s)][0][a]}"]

    def has_struct(c):
        return int(bool(re.search(r"(^|\n)#{1,3}\s|(^|\n)\d+\.\s|\*\*", c or "")))
    print("\n=== CARD structure (median tok / %structured headers·numbered·bold) — seed", MSEED, "===", flush=True)
    print(f"  indiv     tok={int(np.median([tl(nuwa[a]) for a in authors]))}  struct={np.mean([has_struct(nuwa[a]) for a in authors]):.2f}", flush=True)
    for k in (4, 8):
        cs = [shared(k, MSEED, a) for a in authors]
        print(f"  shared@{k} tok={int(np.median([tl(c) for c in cs]))}  struct={np.mean([has_struct(c) for c in cs]):.2f}", flush=True)

    # drafts: indiv (once), sh8 (per seed-cluster), sh4 (MSEED-cluster)
    djobs = [("indiv", None, a, t) for a in authors for t in range(len(T))]
    for (k, ss) in [(8, SEEDS), (4, [MSEED])]:
        for s in ss:
            seen = set()
            for a in authors:
                cid = lay[(k, s)][0][a]
                if cid not in seen:
                    seen.add(cid); djobs += [(f"sh{k}", s, a, t) for t in range(len(T))]
    print(f"\nbuilding {len(djobs)} drafts ...", flush=True)
    card_for = lambda arm, s, a: nuwa[a] if arm == "indiv" else shared(int(arm[2:]), s, a)
    D = {}
    for (arm, s, a, t), txt in zip(djobs, de.pool(lambda j: NW.draft(card_for(j[0], j[1], j[2]), T[j[3]]), djobs)):
        rep = a
        if arm != "indiv":
            k = int(arm[2:]); cid = lay[(k, s)][0][a]; rep = next(b for b in authors if lay[(k, s)][0][b] == cid)
        D[(arm, s, rep, t)] = txt

    def dr(arm, s, a, t):
        if arm == "indiv":
            return D[("indiv", None, a, t)]
        k = int(arm[2:]); cid = lay[(k, s)][0][a]; rep = next(b for b in authors if lay[(k, s)][0][b] == cid)
        return D[(arm, s, rep, t)]

    def judge(x, y, s, a, t):                                  # +1 = x better; NW.quality self-randomizes A/B
        return NW.quality(T[t], dr(x, s, a, t), dr(y, s, a, t), f"k8p-{x}{y}-{s}-{a}-{t}")

    units = [(a, t) for a in authors for t in range(len(T))]
    print("\n=== (A) REPLICATION shared@8 − indiv across seeds (CI resamples seed×cluster) ===", flush=True)
    allv, allg, percl = [], [], {}
    for s in SEEDS:
        v = [judge("sh8", "indiv", s, a, t) for (a, t) in units]
        g = [f"{s}_{lay[(8, s)][0][a]}" for (a, t) in units]
        ci = cluster_mean_ci(v, g, seed=0)
        print(f"  seed {s}: shared@8−indiv = {np.mean(v):+.3f} CI{ci}  (n_cl={len(set(g))})", flush=True)
        allv += v; allg += g
        for (a, t), vv in zip(units, v):
            percl.setdefault(f"{s}_{lay[(8,s)][0][a]}", []).append(vv)
    ci_all = cluster_mean_ci(allv, allg, seed=0)
    cl_means = sorted(float(np.mean(x)) for x in percl.values())
    print(f"  POOLED {SEEDS}: shared@8−indiv = {np.mean(allv):+.3f} CI{ci_all}  (n_cl={len(set(allg))})", flush=True)
    print(f"  per-cluster mean: min={cl_means[0]:+.2f} med={np.median(cl_means):+.2f} max={cl_means[-1]:+.2f} "
          f"| frac clusters >0 = {np.mean([m>0 for m in cl_means]):.2f}", flush=True)

    print(f"\n=== (B) MECHANISM shared@8 vs shared@4 head-to-head (seed {MSEED}) ===", flush=True)
    v = [judge("sh8", "sh4", MSEED, a, t) for (a, t) in units]
    g = [f"{MSEED}_{lay[(8, MSEED)][0][a]}" for (a, t) in units]
    ci_m = cluster_mean_ci(v, g, seed=0)
    fl = "  <-EXCL0" if (ci_m[0] > 0 or ci_m[1] < 0) else ""
    print(f"  shared@8 vs shared@4 = {np.mean(v):+.3f} CI{ci_m}  (n_cl={len(set(g))}){fl}", flush=True)

    out = {"seeds": SEEDS,
           "replication_sh8_indiv": {"diff": round(float(np.mean(allv)), 3), "ci": ci_all, "n_cl": len(set(allg)),
                                     "per_seed": "see log", "frac_clusters_pos": round(float(np.mean([m > 0 for m in cl_means])), 3)},
           "mechanism_sh8_vs_sh4": {"diff": round(float(np.mean(v)), 3), "ci": ci_m, "seed": MSEED},
           "note": "k=8 anomaly: shared@8 = richer structured card; indiv longer but loses (quality not length). "
                   "Replication + sh8-vs-sh4 test whether MORE pooling = better GENERIC playbook (anti-collapse) vs n_cl=14 noise."}
    (RES / "cmd_k8_probe.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> results/cmd_k8_probe.json", flush=True)


if __name__ == "__main__":
    main()
