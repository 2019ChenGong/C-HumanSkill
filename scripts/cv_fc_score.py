"""Score the CV forced-choice pack: win-rates vs the known null of 0.5, placebos, TWO MDEs, TOST.

Why forced choice (SKILL.md §3): an absolute score has no known null, so "arm A ties arm B" can only ever be
`p > 0.05` -- absence of evidence. A forced choice is null-0.5, so a tie becomes an equivalence statement.

THE TRAP THIS SCORER EXISTS TO AVOID. There are two different MDEs and only one of them is about the judge:
  * sampling MDE  = (Z_a+Z_b)*SE, from the bootstrap. It asks "is our n big enough?"
  * judge MDE     = how much content must differ before this judge notices. It asks "can the ruler see it?"
A judge that ignores content entirely and answers by position is LOW VARIANCE. Its sampling MDE is tiny, its
CI is narrow, and every contrast lands on 0.500. Gate on sampling MDE alone and you certify a blind judge's
0.500 as "TIE -- effects above delta excluded", which is the paper's headline manufactured out of nothing.
So the `cut@p` range probe is part of the BATTERY GATE, not a footnote: a contrast may only be called a tie
if the judge demonstrably resolves the smallest content difference we tested.

Reads every `ans*_<i>.json`, so judge replicates coexist:  ans_0.json (haiku) · ans_r2_0.json (sonnet).

Run:  BATCHDIR=results/se/cv_fc DELTA=0.10 python -P scripts/cv_fc_score.py
"""
import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.attrib_metrics import cluster_mean_ci   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

B = ROOT / os.environ.get("BATCHDIR", "results/se/cv_fc")
DELTA = float(os.environ.get("DELTA", "0.10"))     # SESOI on the win-rate scale, pre-registered
ONLY = os.environ.get("ONLY", "")                  # e.g. ONLY=r1 or ONLY=r2 -- score one judge's battery alone
Z_A, Z_B, Z_1 = 1.959964, 0.841621, 1.644854       # two-sided .05 / power .80 / one-sided .05
ANS = re.compile(r"ans(?:_(\w+?))?_(\d+)$")

meta = json.loads((B / "meta.json").read_text(encoding="utf-8"))
cfg = json.loads((B / "config.json").read_text(encoding="utf-8"))
POOLED = set(cfg.get("pooled_arms", ["ne", "cc"]))
UCLU = cfg.get("unit_cluster") or []

picks, reps = defaultdict(dict), set()
for f in sorted(B.glob("ans*.json")):
    m = ANS.fullmatch(f.stem)
    if not m:
        continue
    r = m.group(1) or "r1"
    if ONLY and r != ONLY:
        continue
    reps.add(r)
    for rec in json.loads(f.read_text(encoding="utf-8-sig")):   # subagents emit BOMs
        c = str(rec.get("choice", "")).strip().upper()[:1]
        if c in ("A", "B") and rec.get("pid") in meta:
            picks[rec["pid"]][r] = c
reps = sorted(reps)
missing = [p for p in meta if p not in picks]
print(f"judges: {reps}   items: {len(picks)}/{len(meta)}" + (f"   MISSING {len(missing)}" if missing else ""))
if missing:
    sys.exit(f"missing pids, e.g. {missing[:5]} -- re-dispatch those batches (run judge_qc.py first)")


def target_wins(pid, r):
    """The 'target' (arm X / the full draft / the padded copy) is A when order==0 and B when order==1."""
    return float((picks[pid][r] == "A") == (meta[pid].get("order", 0) == 0))


def agg(pids, group_by_cluster=False):
    """unit -> mean over orders x replicates. Returns (values, cluster labels)."""
    acc, gl = defaultdict(list), {}
    for pid in pids:
        u = meta[pid]["unit"]
        for r in picks[pid]:
            acc[u].append(target_wins(pid, r))
        gl[u] = UCLU[u] if (group_by_cluster and UCLU) else meta[pid]["u"]
    us = sorted(acc)
    return np.array([np.mean(acc[u]) for u in us]), [gl[u] for u in us]


kind = defaultdict(list)
for pid, m in meta.items():
    kind[m["kind"]].append(pid)

out = {"delta": DELTA, "judges": reps, "n_units": cfg.get("n_units"), "n_experts": cfg.get("n_experts"),
       "draft_len": json.loads((B / "draft_len.json").read_text(encoding="utf-8")) if (B / "draft_len.json").exists() else {},
       "placebo": {}, "contrast": {}}

# ---------------- placebos ----------------
print("\nPLACEBO BATTERY   self/pad/fmt must sit at 0.5 (bias probes); cut must rise above 0.5 (range probe)\n")
print(f"  {'probe':36s} {'rate':>6s} {'95% CI':>17s}  {'n':>4s}  verdict")
print("  " + "-" * 80)


def report(label, pids, expect, key, cluster=False):
    """expect: 'null' (two-sided, must contain .5) | 'not_above' (one-sided) | 'signal' (must exceed .5)"""
    if not pids:
        return None
    v, g = agg(pids, cluster)
    m, ci = float(np.mean(v)), cluster_mean_ci(v, g, seed=0)
    if expect == "null":
        ok, verdict = (ci[0] <= 0.5 <= ci[1]), None
        verdict = "PASS" if ok else "*** FAIL — biased"
    elif expect == "not_above":
        # The disease is preferring the LONGER text (the old CV judge: +0.893). A judge that PENALISES
        # content-free padding is not length-biased -- it is content-sensitive. Only an excess above .5 is
        # disqualifying. Note the asymmetry explicitly rather than pretending a two-sided null.
        ok = not (ci[0] > 0.5)
        verdict = "PASS (no length preference)" if ok else "*** FAIL — prefers the longer text"
        if ok and ci[1] < 0.5:
            verdict += "; penalises padding"
    else:
        ok = ci[0] > 0.5
        verdict = "detects" if ok else "BLIND at this level"
    out["placebo"][key] = {"rate": round(m, 3), "ci": [round(c, 3) for c in ci], "n": len(v),
                           "expect": expect, "ok": bool(ok)}
    print(f"  {label:36s} {m:6.3f}  [{ci[0]:+.3f},{ci[1]:+.3f}] {len(v):5d}  {verdict}")
    return ok


# `self`: A and B are the same text, so the rate IS P(choose A). Single order by construction.
sp = kind.get("self", [])
pos_bias = None
if sp:
    v = np.array([np.mean([1.0 if picks[p][r] == "A" else 0.0 for r in picks[p]]) for p in sp])
    g = [meta[p]["u"] for p in sp]
    m, ci = float(np.mean(v)), cluster_mean_ci(v, g, seed=0)
    pos_bias = m
    out["placebo"]["self"] = {"rate": round(m, 3), "ci": [round(c, 3) for c in ci], "n": len(v),
                              "expect": "diagnostic", "ok": None}
    print(f"  {'position: identical A and B':36s} {m:6.3f}  [{ci[0]:+.3f},{ci[1]:+.3f}] {len(v):5d}  "
          f"DIAGNOSTIC (not a gate — see below)")

pad_ok = report("length: +25% content-free filler", kind.get("pad", []), "not_above", "pad")
fmt_ok = report("format: markdown decoration stripped", kind.get("fmt", []), "null", "fmt")

cuts = defaultdict(list)
for pid in kind.get("cut", []):
    cuts[meta[pid]["p"]].append(pid)
detect = {p: report(f"range: full vs {int(100*p)}% of sentences cut", pids, "signal", f"cut@{p}")
          for p, pids in sorted(cuts.items())}

# The judge's own resolution floor: the smallest content deletion it reliably notices.
levels = sorted(cuts)
smallest = min([p for p, ok in detect.items() if ok], default=None)
finest = levels[0] if levels else None
resolves_finest = bool(smallest is not None and finest is not None and smallest <= finest + 1e-9)
out["placebo"]["smallest_detected_cut"] = smallest
out["placebo"]["finest_cut_tested"] = finest
out["placebo"]["resolves_finest"] = resolves_finest

# THE GATE. `self` is NOT in it. Under forced choice a judge shown two identical texts must pick one, and it
# picks A; P(A)=1.0 is the signature of a judge that reserves the default for genuine ties, not of a broken
# one. What actually distinguishes a good judge from a blind one is the pair (self, range):
#     self high + range high  -> decides on content when content differs, defaults to A when it doesn't. GOOD.
#     self high + range low   -> defaults to A even when content differs. BLIND, and its 0.500s are free.
# So the range probe carries the gate, and `self` is reported beside it as the thing that makes range matter.
bias_gates = [x for x in (pad_ok, fmt_ok) if x is not None]
battery = all(bias_gates) and resolves_finest
print("\n  " + ("BATTERY PASS" if battery else "*** BATTERY FAIL"))
if pos_bias is not None:
    print(f"  position: with identical texts the judge picks A {pos_bias:.0%} of the time. Forced choice makes")
    print("        that unavoidable, and the both-orders design cancels it exactly in every contrast. It is a")
    print("        problem ONLY if the judge is also content-blind, in which case every contrast lands on")
    print(f"        0.500 for free. The range probe is what rules that out — and it {'does' if resolves_finest else 'DOES NOT'} here.")
if not resolves_finest:
    print(f"  *** the judge does NOT resolve the finest content cut tested ({int(100*finest)}% of sentences"
          f" deleted{'' if smallest is None else f'; it needs {int(100*smallest)}%'}).")
    print("      Its resolution floor is coarser than the effects we are testing. NO contrast below may be")
    print("      reported as a tie — every null is a statement about the ruler, not about the arms.")

# ---------------- contrasts ----------------
cons = defaultdict(list)
for pid in kind.get("contrast", []):
    cons[(meta[pid]["x"], meta[pid]["y"])].append(pid)

print(f"\nCONTRASTS   win-rate of X over Y, null = 0.500, SESOI delta = {DELTA}\n")
print(f"  {'X':>9s} {'Y':>9s} {'win':>6s} {'95% CI':>17s} {'sMDE':>6s} {'agr':>5s}  verdict")
print("  " + "-" * 84)
CLAIM = {("ne", "nec"): "V4/elemk  black-box CMD vs decomposed-pipeline card",
         ("ne", "in"): "CLAIM 2  pooling vs individual card",
         ("ne", "cc"): "CLAIM 3  CMD vs concat",
         ("ne", "staab"): "CLAIM 4  CMD vs per-person de-id",
         ("ne", "tpar_t15"): "CLAIM 4  CMD vs per-person de-id",
         ("ne", "petre_k4"): "CLAIM 4  CMD vs per-person de-id",
         ("staab", "in"): "         does de-id cost utility?",
         ("in", "st"): "         own card vs a stranger's",
         ("in", "sham"): "         right domain vs a wrong-domain card",
         ("in", "no"): "[CONFOUNDED] a card vs no card"}
sds = []
for (x, y), pids in sorted(cons.items(), key=lambda kv: list(CLAIM).index(kv[0]) if kv[0] in CLAIM else 99):
    # A pooled card is shared by every member of its cluster, so its quality is a cluster-level random effect.
    # Cluster the bootstrap on the pooling cluster for those arms, or the CI is anti-conservative.
    pooled = bool({x, y} & POOLED)
    v, g = agg(pids, group_by_cluster=pooled)

    # cross-order agreement: does the same ARM win in both orders? 0 => the judge answered by position only.
    pair = defaultdict(dict)
    for pid in pids:
        for r in picks[pid]:
            pair[meta[pid]["unit"]][meta[pid]["order"]] = (x if picks[pid][r] == "A" else y) \
                if meta[pid]["order"] == 0 else (y if picks[pid][r] == "A" else x)
    both = [d for d in pair.values() if len(d) == 2]
    agree = float(np.mean([d[0] == d[1] for d in both])) if both else float("nan")

    m, ci = float(np.mean(v)), cluster_mean_ci(v, g, seed=0)
    se = (ci[1] - ci[0]) / (2 * Z_A)
    smde = (Z_A + Z_B) * se
    lo90, hi90 = m - Z_1 * se, m + Z_1 * se
    equiv = (lo90 > 0.5 - DELTA) and (hi90 < 0.5 + DELTA)
    diff = ci[0] > 0.5 or ci[1] < 0.5
    powered = smde < DELTA

    if diff:
        verdict = f"DIFFERENT ({'X' if m > .5 else 'Y'} better)"
    elif not battery:
        verdict = "NO VERDICT — battery failed"
    elif equiv and powered:
        verdict = f"TIE — |effect| > {DELTA} excluded"
    elif not powered:
        verdict = f"UNDERPOWERED — sampling MDE {smde:.3f} >= delta"
    else:
        verdict = "inconclusive"
    sds.append(float(np.std(v, ddof=1)))
    out["contrast"][f"{x}-{y}"] = {"win": round(m, 3), "ci": [round(c, 3) for c in ci], "se": round(se, 4),
                                   "sampling_mde": round(smde, 3), "cross_order_agreement": round(agree, 3),
                                   "n_units": len(v), "n_clusters": len(set(g)),
                                   "clustered_by": "pooling_cluster" if pooled else "expert",
                                   "equivalent": bool(equiv), "different": bool(diff), "powered": bool(powered),
                                   "battery_pass": bool(battery), "verdict": verdict,
                                   "claim": CLAIM.get((x, y), "")}
    print(f"  {x:>9s} {y:>9s} {m:6.3f}  [{ci[0]:.3f},{ci[1]:.3f}] {smde:6.3f} {agree:5.2f}  {verdict}")
    print(f"  {'':>19s} {CLAIM.get((x,y),'')}  [{len(set(g))} {'pooling clusters' if pooled else 'experts'}]")

print("\n  agr = cross-order agreement: how often the same ARM wins in both presentation orders.")
print("        ~0.50 = coin flips.  ~0.00 = the judge answered by POSITION and told us nothing about the arms.")

# ---------------- power planning ----------------
if sds:
    sd = float(np.mean(sds))
    n_now = out["contrast"][list(out["contrast"])[0]]["n_units"]
    print(f"\nPOWER PLANNING   per-unit SD of the win indicator = {sd:.3f}  "
          f"(n = {n_now} units, {len(reps)} judge(s) x 2 orders)")
    for d in (0.05, 0.10, 0.15):
        need = ((Z_A + Z_B) * sd / d) ** 2
        tag = "ALREADY THERE" if need <= n_now else f"need {need/n_now:.1f}x more units, or more replicates to shrink SD"
        print(f"  MDE < {d:.2f}: ~{need:.0f} units at this SD   ({tag})")
    out["power"] = {"per_unit_sd": round(sd, 3), "n_units": n_now, "n_replicates": len(reps),
                    "units_needed": {str(d): round(((Z_A + Z_B) * sd / d) ** 2) for d in (0.05, 0.10, 0.15)}}

out["battery_pass"] = bool(battery)
# ONLY-aware output name: a single-replicate re-score (e.g. ONLY=qwen) must NEVER overwrite the canonical
# all-replicate summary (B2 convention: _fc_summary_qwen.json).
_sumname = f"_fc_summary{('_' + ONLY) if ONLY else ''}.json"
(B / _sumname).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nsaved -> {(B / _sumname).relative_to(ROOT)}")
if not battery:
    sys.exit("BATTERY FAILED — the contrasts above carry no tie claim.")
