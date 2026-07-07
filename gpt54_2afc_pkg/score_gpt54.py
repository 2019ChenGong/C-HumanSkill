"""Score the gpt-5.4 second-attacker 2AFC answers and compare head-to-head with the sonnet-4.6 baseline.

Self-contained: needs only numpy + the stdlib. Reads tasks/<ds>/{meta.json, ans_*.json} for ds in {mad,cv,enron}.
Each ans_i.json = [{"pid","choice","conf"}]; picked_member = (choice == meta[pid].member_slot). Chance = 0.5.
Cluster unit = card_id (author group), identical to the paper's cr_2afc_score / cmd_multiseed_pool bootstrap.

FULL LADDER (byte-identical to what sonnet answered): indiv (positive control, MUST leak) | shared, concat
(pooling -> want CHANCE) | staab, petre_k4, presidio, tpar_t15 (per-person de-id -> expected LEAK).

Per dataset it prints, for every channel, gpt-5.4 acc + 95% cluster-bootstrap CI beside sonnet's; then:
  - a POSITIVE-CONTROL GATE: gpt-5.4 `indiv` must clear CI-lo>0.5 AND acc >= sonnet's indiv (else `shared` is
    uninterpretable -- a weak attacker's ~0.5 proves nothing);
  - the `shared` verdict and the PAIRED indiv->shared drop (cluster-paired bootstrap) = the headline;
  - whether gpt-5.4 REPRODUCES the ladder shape (de-id arms leak, pooling doesn't) = instrument-validity.

Run (from repo root or inside gpt54_2afc_pkg/):  python score_gpt54.py
Out: results_gpt54_summary.json (next to this script).
"""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NBOOT = int(os.environ.get("NBOOT", 20000))
BASELINE = json.loads((HERE / "sonnet_baseline.json").read_text(encoding="utf-8"))
KIND = BASELINE["channel_kind"]      # chan -> "pos.."/"pool.."/"de-id.."
DATASETS = ["mad", "cv", "enron"]
ORDER = ["indiv", "shared", "concat", "staab", "petre_k4", "presidio", "tpar_t15"]


def load(ds):
    d = HERE / "tasks" / ds
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    ans = {}
    for f in sorted(d.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", f.stem):          # stray-guard: only ans_0.json, ans_1.json, ...
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            c = str(r.get("choice", "")).strip().upper()
            m = re.search(r"[AB]", c)
            if m:
                ans[r["pid"]] = m.group(0)
    rows = []
    for pid, mt in meta.items():
        if mt.get("neg") == "nneg" and pid in ans:
            rows.append({"chan": mt["chan"], "card_id": mt["card_id"],
                         "picked_member": int(ans[pid] == mt["member_slot"])})
    chans = [c for c in ORDER if any(v["chan"] == c for v in meta.values())]
    total = {c: sum(1 for v in meta.values() if v["chan"] == c and v.get("neg") == "nneg") for c in chans}
    return meta, rows, total, chans


def boot_acc(rows, seed=0):
    by = {}
    for r in rows:
        by.setdefault(r["card_id"], []).append(r["picked_member"])
    clus = list(by.values())
    if not clus:
        return None
    acc = float(np.mean([v for c in clus for v in c]))
    rng = np.random.default_rng(seed)
    means = np.empty(NBOOT)
    for i in range(NBOOT):
        pick = rng.integers(0, len(clus), len(clus))
        means[i] = np.mean([v for j in pick for v in clus[j]])
    return dict(acc=round(acc, 4), n=len(rows), ncl=len(clus),
                ci=[round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)])


def boot_paired_drop(rows, a="indiv", b="shared", seed=0):
    """cluster-paired bootstrap of acc(a) - acc(b) over card_id."""
    by = {}
    for r in rows:
        if r["chan"] in (a, b):
            by.setdefault(r["card_id"], {a: [], b: []})[r["chan"]].append(r["picked_member"])
    cids = [c for c in by if by[c][a] and by[c][b]]
    if not cids:
        return None

    def diff(sel):
        va = [v for c in sel for v in by[c][a]]
        vb = [v for c in sel for v in by[c][b]]
        return np.mean(va) - np.mean(vb)
    obs = float(diff(cids))
    rng = np.random.default_rng(seed)
    dd = np.empty(NBOOT)
    for i in range(NBOOT):
        sel = [cids[j] for j in rng.integers(0, len(cids), len(cids))]
        dd[i] = diff(sel)
    lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
    return dict(drop=round(obs, 4), ci=[round(lo, 4), round(hi, 4)], sig=bool(lo > 0))


def chan_verdict(chan, g):
    """g = boot_acc dict for gpt-5.4 on this channel."""
    lo, hi = g["ci"]
    k = KIND.get(chan, "")
    if k.startswith("pos"):
        return "LEAKS (reads indiv card)" if lo > 0.5 else "WEAK (barely reads indiv)"
    if k.startswith("pool"):
        return "CHANCE = anonymized" if lo <= 0.5 <= hi else ("LEAKS (pooled card still re-IDs)" if lo > 0.5 else "below chance")
    return "LEAKS (de-id still re-IDs)" if lo > 0.5 else ("CHANCE (de-id reached anonymity?!)" if lo <= 0.5 <= hi else "below chance")


def main():
    out = {"instrument": "gpt-5.4 2AFC (second attacker) vs sonnet-4.6 baseline", "nboot": NBOOT, "datasets": {}}
    print(f"\n=== gpt-5.4 SECOND-ATTACKER 2AFC (full ladder; chance=0.5; cluster-bootstrap over card_id, NBOOT={NBOOT}) ===")
    print("    GATE: gpt-5.4 `indiv` must beat sonnet's indiv AND exclude 0.5  ->  else `shared` is uninterpretable.\n")
    for ds in DATASETS:
        loaded = load(ds)
        if loaded is None:
            print(f"-- {ds.upper()}: tasks/{ds}/meta.json missing, skip"); continue
        meta, rows, total, chans = loaded
        base = BASELINE["datasets"][ds]; bch = base["channels"]
        answered = {c: sum(1 for r in rows if r["chan"] == c) for c in chans}
        cov = " ".join(f"{c}={answered[c]}/{total[c]}" for c in chans)
        print(f"-- {ds.upper()} (k8 s{base['seed']}, {base['n_clusters']} clusters)  coverage: {cov}")
        if sum(answered.values()) == 0:
            print("     (no answers yet)\n"); continue
        res = {"coverage": {c: [answered[c], total[c]] for c in chans}, "chan": {}}
        print(f"     {'channel':10s} {'kind':28s} {'gpt5.4':>21s}   {'sonnet-4.6':>21s}")
        for c in chans:
            sub = [r for r in rows if r["chan"] == c]
            g = boot_acc(sub) if sub else None
            b = bch.get(c)
            gs = f"{g['acc']:.3f} [{g['ci'][0]:.3f},{g['ci'][1]:.3f}]" if g else "(none)"
            bs = f"{b['acc']:.3f} [{b['ci'][0]:.3f},{b['ci'][1]:.3f}]" if b else "-"
            vv = f"  {chan_verdict(c, g)}" if g else ""
            print(f"     {c:10s} {KIND.get(c,''):28s} {gs:>21s}   {bs:>21s}{vv}")
            res["chan"][c] = {"gpt54": g, "sonnet": b}
        # GATE (indiv positive control)
        gi = res["chan"].get("indiv", {}).get("gpt54")
        gate = None
        if gi:
            strong = gi["ci"][0] > 0.5 and gi["acc"] >= bch["indiv"]["acc"]
            gate = {"pass": bool(strong), "gpt54_indiv": gi["acc"], "sonnet_indiv": bch["indiv"]["acc"]}
            print(f"     GATE: gpt5.4 indiv {gi['acc']:.3f} vs sonnet {bch['indiv']['acc']:.3f}, CI-lo {gi['ci'][0]:.3f}"
                  f"  ->  {'PASS (strong card attacker; shared interpretable)' if strong else 'FAIL (not strong enough on cards; do NOT read shared as anonymity)'}")
        # headline: paired indiv->shared drop
        drop = boot_paired_drop(rows, "indiv", "shared")
        if drop:
            print(f"     >>> PAIRED indiv->shared drop = {drop['drop']:+.3f} [{drop['ci'][0]:+.3f},{drop['ci'][1]:+.3f}]"
                  f"  {'SIG' if drop['sig'] else 'n.s.'}  (attacker reads indiv, loses shared)")
            res["paired_indiv_shared_drop"] = drop
        res["gate"] = gate
        out["datasets"][ds] = res
        print()
    (HERE / "results_gpt54_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved -> gpt54_2afc_pkg/results_gpt54_summary.json")
    print("\nWHAT TO CONCLUDE:")
    print("  * GATE PASS + `shared`/`concat` at CHANCE + de-id arms still LEAK  => gpt-5.4 reproduces the whole ladder:")
    print("    a stronger, different-lineage attacker reads individual & per-person-de-id cards but NOT the pooled card")
    print("    -> the 'sonnet is just weak' objection dies. This is the win.")
    print("  * GATE FAIL (gpt-5.4 indiv < sonnet) => report honestly; this run doesn't strengthen the claim.")
    print("  * `shared` LEAK (CI>0.5) => gpt-5.4 DOES re-identify the pooled card -> a real problem, report it.")


if __name__ == "__main__":
    main()
