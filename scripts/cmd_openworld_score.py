"""Score the CMD membership-inference attack: pooled ROC-AUC, cluster-bootstrap CI, label-permutation null, tie-rate. Set KCL= and DATASET=."""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

try:                                                  # Windows consoles default to GBK -> non-ASCII prints (⚠/≈/→) crash
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("DATASET", "enron")
RES = ROOT / "results" if DATASET == "enron" else ROOT / "results" / DATASET
RES = (ROOT / os.environ["RESDIR"]) if os.environ.get("RESDIR") else RES   # cross-model run isolation
KCL = os.environ.get("KCL")        # REQUIRED: which k-namespace to score (files are _ow_k{KCL}_*); result -> _k{KCL}.json
if not KCL:
    sys.exit("set KCL env (e.g. KCL=8) — open-world files are namespaced by k to avoid cross-k overwrite")


def auc(pairs):
    """pairs = list of (y_true 0/1, score). Returns pooled ROC-AUC or None if a class is empty."""
    y = [p[0] for p in pairs]; s = [p[1] for p in pairs]
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def discover(chan):
    out = []
    for f in RES.glob(f"_ow_k{KCL}_{chan}_s*_key.json"):
        m = re.search(r"_s(\d+)_key$", f.stem)        # anchor to the seed token (avoid the '_s' inside '_shared')
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def load_records(chan):
    """Return list of (group, score, label) over all seeds; group = seed_cluster (shared) or author (raw/indiv)."""
    recs, miss = [], 0
    for s in discover(chan):
        key = json.loads((RES / f"_ow_k{KCL}_{chan}_s{s}_key.json").read_text())
        pf = RES / f"_picks_ow_k{KCL}_{chan}_s{s}.json"
        if not pf.exists():
            print(f"  ⚠ missing picks {pf.name}"); continue
        picks = json.loads(pf.read_text())
        for t, meta in key.items():
            sc = picks.get(t)
            if not sc:
                miss += 1; continue
            grp = f"{s}_{meta['cluster']}" if chan == "shared" else meta["author"]
            for slot, lab in meta["labels"].items():
                if slot in sc:
                    try:
                        recs.append((grp, float(sc[slot]), lab))
                    except (TypeError, ValueError):
                        pass
    if miss:
        print(f"  ⚠ {chan}: {miss} trials had no picks (excluded)")
    return recs


def pooled_auc(recs, neg):
    pairs = [(1, sc) for (_g, sc, lab) in recs if lab == "pos"] + [(0, sc) for (_g, sc, lab) in recs if lab == neg]
    return auc(pairs)


def boot_ci(recs, neg, nboot=2000, seed=0):
    by = {}
    for r in recs:
        by.setdefault(r[0], []).append(r)
    groups = list(by); rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nboot):
        samp = []
        for g in rng.choice(groups, len(groups), replace=True):
            samp += by[g]
        a = pooled_auc(samp, neg)
        if a is not None:
            vals.append(a)
    if not vals:
        return [None, None]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def perm_p(recs, neg, obs, nperm=2000, seed=0):
    """label-permutation null: shuffle pos/neg labels among the pos+neg candidates, recompute AUC."""
    sub = [(sc, lab) for (_g, sc, lab) in recs if lab in ("pos", neg)]
    y = np.array([1 if lab == "pos" else 0 for (_sc, lab) in sub]); s = np.array([sc for (sc, _lab) in sub])
    if len(set(y.tolist())) < 2:
        return None
    rng = np.random.default_rng(seed); null = []
    for _ in range(nperm):
        yp = rng.permutation(y)
        null.append(roc_auc_score(yp, s))
    null = np.array(null)
    return round(float((np.sum(np.abs(null - 0.5) >= abs(obs - 0.5)) + 1) / (nperm + 1)), 4)


def tie_rate(recs):
    s = [r[1] for r in recs]
    if not s:
        return 0.0
    from collections import Counter
    c = Counter(s)
    return round(sum(v for v in c.values() if v > 1) / len(s), 3)


def main():
    print("CMD OPEN-WORLD membership-verification AUC (pooled; CI=cluster bootstrap; null=label permutation)\n")
    summary = {}
    for chan in ("shared", "raw", "indiv", "staab", "staab_r1", "staab_g55_r1", "presidio", "tpar_t10", "tpar_t15", "petre_k4"):
        recs = load_records(chan)
        if not recs:
            print(f"{chan}: no records (run dump + Opus subagents first)\n"); continue
        ngrp = len(set(r[0] for r in recs)); tr = tie_rate(recs)
        npos = sum(r[2] == "pos" for r in recs)
        row = {"n_groups": ngrp, "n_pos": npos, "tie_rate": tr}
        print(f"=== {chan}  (n_groups={ngrp}, n_pos={npos}, tie_rate={tr}) ===")
        for neg in ("rneg", "nneg"):
            a = pooled_auc(recs, neg)
            if a is None:
                print(f"  vs {neg}: n/a"); continue
            ci = boot_ci(recs, neg); p = perm_p(recs, neg, a)
            sig = "SIG>0.5" if (ci[0] is not None and ci[0] > 0.5) else "≈0.5 (CI contains 0.5)"
            row[neg] = {"auc": round(a, 3), "ci": ci, "perm_p": p, "ci_width": round((ci[1] - ci[0]), 3) if ci[0] is not None else None}
            print(f"  AUC vs {neg:4s} = {a:.3f} CI{ci} (w={row[neg]['ci_width']}) perm_p={p}  {sig}")
        summary[chan] = row
        print()
    print("(note: perm_p ignores cluster blocks -> anticonservative; headline verdict uses the cluster-bootstrap CI.)")
    # verdict helpers (decision keyed on the block-aware bootstrap CI, NOT perm_p)
    sh = summary.get("shared", {})
    if sh.get("rneg") and sh.get("nneg"):
        ar, an = sh["rneg"]["auc"], sh["nneg"]["auc"]
        rlo, nlo = sh["rneg"]["ci"][0], sh["nneg"]["ci"][0]
        if rlo is not None and rlo <= 0.5:
            v = "shared membership UNDETECTABLE (rand-neg AUC CI contains 0.5) — open-world safe"
        elif an < ar - 0.05 and nlo is not None and nlo <= 0.5:
            v = "shared AUC>0.5 on random neg but collapses on content-near neg -> signal is TOPIC (G5 attribute), NOT identity"
        else:
            v = "shared AUC>0.5 on BOTH negs -> identity membership may leak open-world (check CI width / underpowered band 0.52-0.60)"
        print(f"VERDICT shared: {v}")
        summary["verdict_shared"] = v
    if summary.get("indiv", {}).get("rneg"):
        ic = summary["indiv"]["rneg"]["ci"]
        print(f"POS-CTRL indiv AUC(rand) CI{ic}: {'OK (>0.5)' if (ic[0] is not None and ic[0] > 0.5) else 'FAIL -> attacker too weak, result VOID'}")
    for arm in ("staab", "staab_r1", "staab_g55_r1", "presidio", "tpar_t10", "tpar_t15", "petre_k4"):   # per-person de-id vs indiv (pos-ctrl) vs shared (CMD)
        if summary.get(arm, {}).get("rneg"):
            st = summary[arm]["rneg"]
            iv = summary.get("indiv", {}).get("rneg", {}).get("auc"); sh = summary.get("shared", {}).get("rneg", {}).get("auc")
            leak = "still LEAKS (>0.5)" if (st["ci"][0] is not None and st["ci"][0] > 0.5) else "≈0.5"
            print(f"{arm.upper()} AUC(rand) = {st['auc']} CI{st['ci']} -> {leak}  (vs indiv {iv} / shared-CMD {sh})")
    (RES / f"cmd_openworld_result_k{KCL}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nsaved -> results/cmd_openworld_result_k{KCL}.json")


if __name__ == "__main__":
    main()
