"""#2 Equivalence (TOST) + MDE re-analysis of the 2AFC anonymity nulls.

$0 — pure re-analysis of the EXISTING ans_*.json (no LLM calls). Turns every "CI ∋ 0.5" verdict into a
defensible statement instead of a bare failure-to-reject:

  (A) One-sided NON-INFERIORITY (the privacy-relevant test): is the upper 95% bound < 0.5+δ ?
      -> a meaningful leak (acc ≥ 0.5+δ) is EXCLUDED.  (below-chance acc is FINE for privacy, so the
         one-sided upper test is the right tool — not the symmetric TOST.)
  (B) Two-sided TOST equivalence to [0.5-δ, 0.5+δ] (δ=0.05 pre-registered): 90% CI ⊂ (L,U) ?  (stricter;
      reported because the checklist doc pre-registered [0.45,0.55]. Fails on the LOW side when the
      attacker is below chance — which is not a privacy problem, hence (A) is primary.)
  (C) MDE (α=.05, power=.80): smallest true acc above .5 this design can DETECT. If MDE_thresh ≥ 0.5+δ the
      study is UNDERPOWERED for a margin-sized leak => a non-significant null does NOT prove anonymity.

Cluster unit = card_id (group), identical to cr_2afc_score. SE + percentiles from the SAME cluster bootstrap
(respects author-group dependence; with few groups — CV k12→6, k16→~5 — the SE is honestly large and MDE blows up).

Run:  python scripts/cmd_equiv_test.py            [DELTA=0.05 NBOOT=20000]
Out:  results/equiv_mde_summary.json + a cross-dataset table.
"""
import os
import re
import sys
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DELTA = float(os.environ.get("DELTA", 0.05))   # equivalence margin around 0.5 -> [L, U]
NBOOT = int(os.environ.get("NBOOT", 20000))
L, U = 0.5 - DELTA, 0.5 + DELTA
Z_A = 1.6448536269514722   # z_{0.95}  (one-sided alpha=.05)
Z_B = 0.8416212335729143   # z_{0.80}  (power=.80)

POOL = {"shared", "concat", "neutral"}   # anonymity claim (want acc ≈ chance)
POS = {"indiv", "raw"}                    # positive control / leak-expected

# (label, batchdir) — the P0 CMD k-sweep (sonnet, s0) across 3 datasets + battery reference dirs.
# MAD k2/4/6/8 raw ans live in the old ksweep format (see results/mad/ksweep_summary.json), not a cr_2afc
# batchdir, so only MAD k10/k12 are re-analyzable here; their acc/CI at k2/4/6/8 come from ksweep_summary.
TARGETS = [
    # --- MAD k-sweep (s0), full raw ---
    ("MAD k2 s0", "results/mad/ksw_k2_s0"),
    ("MAD k4 s0", "results/mad/ksw_k4_s0"),
    ("MAD k6 s0", "results/mad/ksw_k6_s0"),
    ("MAD k8 s0", "results/mad/ksw_k8_s0"),
    ("MAD k10 s0", "results/mad/ksw_k10_s0"),
    ("MAD k12 s0", "results/mad/ksw_k12_s0"),
    # --- CV k-sweep (s0), full raw ---
    ("CV k2 s0", "results/se/ksw_k2_s0"),
    ("CV k4 s0", "results/se/ksw_k4_s0"),
    ("CV k6 s0", "results/se/ksw_k6_s0"),
    ("CV k8 s0", "results/se/2afc_battery"),
    ("CV k10 s0", "results/se/ksw_k10_s0"),
    ("CV k12 s0", "results/se/2afc_k12"),
    # --- Enron k-sweep (s0), full raw ---
    ("Enron k2 s0", "results/enron/ksw_k2_s0"),
    ("Enron k4 s0", "results/enron/ksw_k4_s0"),
    ("Enron k6 s0", "results/enron/ksw_k6_s0"),
    ("Enron k8 s0", "results/enron/ksw_k8_s0"),
    ("Enron k10 s0", "results/enron/ksw_k10_s0"),
    ("Enron k12 s0", "results/enron/ksw_k12_s0"),
    # --- battery reference (s1 pooling + per-person de-id arms) for the #2 table ---
    ("Enron k8 s1 (battery)", "results/enron/2afc_free"),
    ("MAD k8 s1 (battery)", "results/mad/2afc_free"),
]


def Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def load_rows(bdir):
    B = ROOT / bdir
    mp = B / "meta.json"
    if not mp.exists():
        return None
    meta = json.loads(mp.read_text(encoding="utf-8"))
    ans = {}
    for f in sorted(B.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", f.stem):     # stray-guard: skip ans_batchX / ans_3_part* leftovers
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            c = str(r.get("choice", "")).strip().upper()
            m = re.search(r"[AB]", c)
            if m:
                ans[r["pid"]] = m.group(0)
    rows = []
    for pid, mt in meta.items():
        if pid in ans:
            rows.append({**mt, "picked_member": int(ans[pid] == mt["member_slot"])})
    return rows


def analyze(sub, seed=0):
    by = {}
    for r in sub:
        by.setdefault(r["card_id"], []).append(r["picked_member"])
    clus = list(by.values())
    ncl = len(clus)
    acc = float(np.mean([v for c in clus for v in c]))
    rng = np.random.default_rng(seed)
    means = np.empty(NBOOT)
    for i in range(NBOOT):
        pick = rng.integers(0, ncl, ncl)
        means[i] = np.mean([v for j in pick for v in clus[j]])
    se = float(means.std(ddof=1))
    ci95 = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    ci90 = (float(np.percentile(means, 5)), float(np.percentile(means, 95)))
    up95 = float(np.percentile(means, 95))     # one-sided upper 95% bound
    # TOST normal-approx p = max of the two one-sided p's (supports the CI-based decision)
    p_below_U = Phi((acc - U) / se) if se > 0 else float(acc < U)   # reject H0: mu >= U
    p_above_L = 1.0 - Phi((acc - L) / se) if se > 0 else float(acc > L)  # reject H0: mu <= L
    tost_p = max(p_below_U, p_above_L)
    equiv = (ci90[0] > L) and (ci90[1] < U)    # two-sided TOST via 90% CI
    noninf = up95 < U                           # privacy: leak >= U excluded
    mde_delta = (Z_A + Z_B) * se
    mde_thresh = 0.5 + mde_delta
    underpowered = mde_thresh >= U
    return dict(acc=round(acc, 4), n=len(sub), ncl=ncl, se=round(se, 4),
                ci95=[round(ci95[0], 4), round(ci95[1], 4)], ci90=[round(ci90[0], 4), round(ci90[1], 4)],
                up95=round(up95, 4), tost_p=round(tost_p, 4), equiv=bool(equiv), noninf=bool(noninf),
                mde_delta=round(mde_delta, 4), mde_thresh=round(mde_thresh, 4), underpowered=bool(underpowered))


def anon_verdict(s):
    tost = "TOST=" + ("eq" if s["equiv"] else "no")
    if s["ci95"][0] > 0.5:                        # CI EXCLUDES .5 from below => a well-powered LEAK (check first,
        return f"LEAK detected (95%CI [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]>.5)  [{tost}]"   # else it was mislabeled 'underpowered')
    if s["noninf"]:
        return f"ANON ✓ leak≥{U:.2f} excluded  [{tost}]"
    if s["underpowered"]:
        return f"UNDERPOWERED can't exclude leak≤{s['mde_thresh']:.2f}  [{tost}]"
    return f"LEAK not excluded up95={s['up95']:.3f}≥{U:.2f}  [{tost}]"


def leak_verdict(s):
    if s["ci95"][0] > 0.5:
        return "LEAK detected (95%CI>.5) ✓ well-powered"
    if s["underpowered"]:
        return f"n.s. but UNDERPOWERED (MDE {s['mde_thresh']:.2f})"
    return "n.s. (no leak detected)"


PREF = ["indiv", "raw", "shared", "concat", "neutral"]
out = {"delta": DELTA, "margin": [L, U], "nboot": NBOOT, "targets": {}}
print(f"\n#2 TOST + MDE re-analysis | margin=[{L:.2f},{U:.2f}] (δ={DELTA})  NBOOT={NBOOT}")
print("  ANON verdict uses one-sided upper (privacy); TOST=two-sided equivalence; MDE≥"
      f"{U:.2f} => underpowered null.\n")
for label, bdir in TARGETS:
    rows = load_rows(bdir)
    if rows is None:
        print(f"== {label:22s} ({bdir})  -- meta.json MISSING, skip"); continue
    if not rows:
        print(f"== {label:22s} ({bdir})  -- 0 answered pairs, skip"); continue
    present = list(dict.fromkeys(r["chan"] for r in rows))
    chans = [c for c in PREF if c in present] + sorted(c for c in present if c not in PREF)
    print(f"== {label:22s} ({bdir})  n_pairs={len(rows)}")
    print(f"   {'chan':9s} {'acc':>6s} {'clus':>4s} {'se':>6s} {'95%CI':>16s} {'up95':>6s} {'MDE→':>6s}  verdict")
    tgt = {}
    for chan in chans:
        sub = [r for r in rows if r["chan"] == chan]
        s = analyze(sub)
        kind = "pool" if chan in POOL else ("pos" if chan in POS else "deid")
        s["kind"] = kind
        s["verdict"] = anon_verdict(s) if kind == "pool" else leak_verdict(s)
        tgt[chan] = s
        ci = f"[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]"
        print(f"   {chan:9s} {s['acc']:>6.3f} {s['ncl']:>4d} {s['se']:>6.3f} {ci:>16s} "
              f"{s['up95']:>6.3f} {s['mde_thresh']:>6.3f}  {s['verdict']}")
    out["targets"][label] = {"batchdir": bdir, "n_pairs": len(rows), "arms": tgt}
    print()

(ROOT / "results" / "equiv_mde_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("saved -> results/equiv_mde_summary.json")
print("\nNOTE: 'ANON ✓' = one-sided upper 95% bound < margin U => meaningful leak excluded (privacy-clean).")
print("      'UNDERPOWERED' = MDE threshold ≥ U => even a margin-sized real leak would be invisible here")
print("      (few groups: CV k12=6, k16=~5). A non-significant null there is NOT proof of anonymity.")
