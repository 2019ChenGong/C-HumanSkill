"""R15b — Rank@k / Precision@k on the naming tournament, in the metric convention of TSA
("De-Anonymization at Scale via Tournament-Style Attribution", arXiv 2601.12407).

TSA's definitions, verbatim from the paper:
  Rank@k      "the frequency that the top-k list contains at least one same-author document"
  Precision@k "the proportion of same-author documents among the top-k results"
Adapted: per released card (bracket) we rank the WHOLE pool by the candidate score and take top-k;
"same-author" becomes "is a true member of the card".

Primary score = S1 (rounds survived). Under top-1 advancement TSA's rule ("top-ranked documents
receive 2phi points") accumulates to 2*phi*depth, a strictly monotone transform of S1 -> identical
ranking, hence identical Rank@k / Precision@k / AUC. S0 is also printed for transparency.

TIES: S1 leaves ~87% of candidates tied at the floor, so "top-k" is ambiguous. We do NOT break ties
arbitrarily; we compute the EXACT EXPECTATION under uniformly random tie-breaking (closed form), which
is deterministic and cannot be gamed by ordering.

CHANCE lines (exact, per bracket, N = pool, M = #members):
  Precision@k = M/N        Rank@k = 1 - C(N-M, k)/C(N, k)

Reads the per-unit (score,label) lists already written by scripts/naming_auc.py -- no re-scoring.

  python -P scripts/naming_rank_at_k.py
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = {"enron": "enron", "mad": "mad", "cv": "se"}
SUBDIRS = ("naming_v6", "naming_deid", "naming_indiv_full", "naming_v6_more",
           "naming_deid_full", "naming_pool_ab")  # R17 increments + R18 equal-n increment for the 3 de-id arms
# (collect() is already keyed by unit id, so increment dirs merge into the same cluster correctly)
# A12 (#153/#154): pooled baselines ne/concat join the scan; POOLED mirrors naming_export.POOLED
# (kept local -- these pooled scripts sweep all 3 datasets and must not bind NE to one DATASET).
ARMS = ("indiv", "staab", "petre_k4", "tpar_t15", "v6", "ne", "concat")
POOLED = ("v6", "ne", "concat")
KS = (1, 5, 10, 20)
VARIANTS = ("S1", "S0")
BOOT_N, BOOT_SEED = 5000, 0

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def topk_expected(scores, labels, k):
    """Exact E[hit], E[precision] over uniformly random tie-breaking. Returns (rank_at_k, prec_at_k)."""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    uniq, counts = np.unique(-s, return_counts=True)          # descending groups
    filled, m_hi = 0, 0
    idx = 0
    for cnt in counts:
        grp_y = y[idx:idx + cnt]
        if filled + cnt <= k:                                  # whole group fits above the cut
            m_hi += int(grp_y.sum()); filled += cnt; idx += cnt
            if filled == k:
                return (1.0 if m_hi >= 1 else 0.0), m_hi / k
            continue
        r = k - filled                                          # partial draw from this tie group
        g, mg = int(cnt), int(grp_y.sum())
        exp_m = m_hi + r * mg / g
        if m_hi >= 1:
            hit = 1.0
        else:
            hit = 1.0 - (comb(g - mg, r) / comb(g, r) if g - mg >= r else 0.0)
        return hit, exp_m / k
    return (1.0 if m_hi >= 1 else 0.0), m_hi / max(filled, 1)   # k >= pool


def boot(vals_by_unit):
    units = [v for v in vals_by_unit.values() if v]
    allv = [x for u in units for x in u]
    rng = np.random.default_rng(BOOT_SEED)
    means = [np.mean(np.concatenate([units[i] for i in rng.integers(0, len(units), len(units))]))
             for _ in range(BOOT_N)]
    return (float(np.mean(allv)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)),
            len(allv), len(units))


def collect(arm, variant):
    """-> {ds: {unit: [(scores, labels) per bracket]}}, N per ds. Chunks the flat per-unit list by pool size."""
    per_ds, Ns = {}, {}
    for ds, d in DIRS.items():
        for sub in SUBDIRS:
            p = ROOT / f"results/{d}/{sub}/_naming_auc.json"
            if not p.exists():
                continue
            blob = json.loads(p.read_text(encoding="utf-8"))
            sec = blob.get("arms", {}).get(arm, {}).get(variant)
            if not sec:
                continue
            N = int(blob["config"]["pool_n"]); Ns[ds] = N
            for u, vals in sec["units"].items():
                assert len(vals) % N == 0, f"{ds}/{arm}/{variant} unit {u}: {len(vals)} not a multiple of {N}"
                brs = []
                for i in range(0, len(vals), N):                # each chunk of N = one bracket
                    chunk = vals[i:i + N]
                    brs.append(([c[0] for c in chunk], [c[1] for c in chunk]))
                per_ds.setdefault(ds, {}).setdefault(f"{ds}:{u}", []).extend(brs)
    return per_ds, Ns


def main():
    print("=== R15b  Rank@k / Precision@k  (TSA metric convention; exact expectation over tie-breaking) ===")
    out = {"convention": "TSA arXiv 2601.12407: Rank@k = top-k contains >=1 true member; "
                         "Precision@k = share of true members in top-k", "arms": {}}
    for variant in VARIANTS:
        print(f"\n--- score = {variant}" +
              ("  (PRIMARY: monotone-equivalent to TSA's 2*phi*depth under top-1 advancement)"
               if variant == "S1" else "  (transparency)"))
        for arm in ARMS:
            per_ds, Ns = collect(arm, variant)
            if not per_ds:
                continue
            row = {}
            for k in KS:
                rk, pk, ch_r, ch_p = {}, {}, [], []
                for ds, units in per_ds.items():
                    N = Ns[ds]
                    for u, brs in units.items():
                        for sc, lb in brs:
                            h, pr = topk_expected(sc, lb, k)
                            rk.setdefault(u, []).append(h); pk.setdefault(u, []).append(pr)
                            M = int(sum(lb))
                            ch_r.append(1.0 - (comb(N - M, k) / comb(N, k) if N - M >= k else 0.0))
                            ch_p.append(M / N)
                r, rlo, rhi, n, nu = boot(rk)
                p_, plo, phi_, _, _ = boot(pk)
                row[f"k{k}"] = {"rank_at_k": round(r, 4), "rank_ci": [round(rlo, 4), round(rhi, 4)],
                                "rank_chance": round(float(np.mean(ch_r)), 4),
                                "prec_at_k": round(p_, 4), "prec_ci": [round(plo, 4), round(phi_, 4)],
                                "prec_chance": round(float(np.mean(ch_p)), 4),
                                "n_brackets": n, "n_units": nu}
                lift = r / np.mean(ch_r) if np.mean(ch_r) else float("nan")
                flag = " *" if rlo > np.mean(ch_r) else "  "
                print(f"  {arm:9s} Rank@{k:<2d} {r:.3f} [{rlo:.3f},{rhi:.3f}] vs chance "
                      f"{np.mean(ch_r):.3f} ({lift:.1f}x){flag}  Prec@{k:<2d} {p_:.3f} "
                      f"vs {np.mean(ch_p):.3f}  (n={n})")
            out["arms"].setdefault(arm, {})[variant] = row
    p = ROOT / "results/naming_rank_at_k.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {p.relative_to(ROOT)}   (* = CI-lo clears the random-ranking chance line)")


if __name__ == "__main__":
    main()
