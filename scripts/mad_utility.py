"""20-MAD OBJECTIVE utility: bug-resolution-prediction accuracy per arm — for CMD AND the comparison de-id methods.

Task (objective, NO LLM judge, so no judge-circularity): given a developer's skill card + a held-out solved bug
(report+stub), predict its RESOLUTION class (FIXED/WONTFIX/INVALID/DUPLICATE/WORKSFORME); score = accuracy vs the
real resolution. Answers "do these de-id arms (and CMD's shared card) keep the card's TASK utility on the 2nd dataset?"

Arms (all on the SAME 128 devs / same held-out bugs from data/20mad/mad_cmd_pool.json):
  nocard | indiv (nuwa) | cmd@k (ε=0 shared card, our method) | staab | staab_r1 | tpar_t10 | tpar_t15 | petre_k4
Reports accuracy per arm + paired diffs (arm−nocard = "helps?", arm−indiv = "vs personal card"), dev-cluster-bootstrap CI.

  MAXBUGS=8   held-out bugs per dev (deterministic first-N; caps cost)
  KCMD=4      which k for the CMD shared card
  PILOT_DRYRUN=1   cost/plan only

Run:  DATASET=mad MAXBUGS=8 python scripts/mad_utility.py   ->  results/mad/mad_utility.json
"""
import os
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
import cmd_gate as CG  # noqa: E402
from src.llm import chat  # noqa: E402
from src.attrib_metrics import cluster_mean_ci, cluster_paired_diff_ci  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

if os.environ.get("DATASET") != "mad":
    sys.exit("run with DATASET=mad (utility arms live in data/20mad/)")

GEN = os.environ.get("PREDICTOR", "deepseek-chat")    # cross-model predictor (e.g. openrouter/qwen/qwen3.7-max)
PRED_MAXTOK = int(os.environ.get("PRED_MAXTOK", 6))   # raise for THINKING predictors (gemini-flash needs ~500 to emit the class past its reasoning)
MAXBUGS = int(os.environ.get("MAXBUGS", 8))
KCMD = int(os.environ.get("KCMD", 4))
SEED = 0
CLASSES = ["FIXED", "WONTFIX", "INVALID", "DUPLICATE", "WORKSFORME"]
RES = (ROOT / os.environ["RESDIR"]) if os.environ.get("RESDIR") else ROOT / "results" / "mad"   # cross-model run isolation
RES.mkdir(parents=True, exist_ok=True)
# de-id comparison arms scored on the utility axis (must already be built in mad_cmd_step2.json)
DEID_ARMS = ["staab", "staab_r1", "tpar_t10", "tpar_t15", "petre_k4"]


def predict_res(card, report, stub):
    """Predict the most-likely resolution class (deepseek, temp 0 -> deterministic + cached). Same prompt as
    mad_comp_two_axis.predict_res so the metric matches the established 20-MAD utility protocol."""
    prof = f"Developer triage profile:\n{card}\n\n" if card else ""
    out = (chat([{"role": "system", "content": "You triage software bugs. Predict the most likely RESOLUTION."},
                 {"role": "user", "content": f"{prof}Bug:\n{stub}\n{report}\n\nWhat is the most likely resolution? "
                  f"Answer ONLY ONE of: {', '.join(CLASSES)}."}], model=GEN, temperature=0.0, max_tokens=PRED_MAXTOK) or "").upper()
    for c in CLASSES:
        if c in out:
            return c
    return None


def main():
    pool, authors, nuwa, aggro, _ref, _raw = CG.load()                       # mad: pool[d] has card_comments + solved_bugs
    step2 = json.loads(CG.STEP2C.read_text(encoding="utf-8"))                 # aggro/staab/.../petre_k4
    shared = json.loads(CG.SHAREDC.read_text(encoding="utf-8")) if CG.SHAREDC.exists() else {}
    cpath = CG.SE / ("cmd_concat_cards.json" if CG.DATASET == "enron" else "cmd_concat_cards_mad.json")
    concat = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else {}   # naive hard-pool baseline (group-level, same key scheme as shared)
    arms = ["nocard", "indiv", f"cmd@{KCMD}"] + ([f"concat@{KCMD}"] if concat else []) \
        + [a for a in DEID_ARMS if a in step2 and all(d in step2[a] for d in authors)]
    miss = [a for a in DEID_ARMS if a not in arms]
    if miss:
        print(f"  [!] de-id arms not built / incomplete -> EXCLUDED from utility: {miss}", flush=True)

    grp, byc = CG.make_groups(aggro, authors, KCMD, SEED)                     # CMD clusters (same builder as MIA/gate)
    def cmd_card(d):
        ck = f"k{KCMD}_s{SEED}_{grp[d]}"
        if ck not in shared:
            raise KeyError(f"missing CMD shared card {ck} (run cmd_gate/cmd_openworld to build it first)")
        return shared[ck]

    def concat_card(d):
        ck = f"k{KCMD}_s{SEED}_{grp[d]}"
        if ck not in concat:
            raise KeyError(f"missing concat card {ck} (run cmd_concat_build.py KCL={KCMD} SEED={SEED} first)")
        return concat[ck]

    import hashlib
    # stranger: a deterministic NON-cluster dev's nuwa card -> tests person-specificity (own vs stranger).
    # own>stranger => the +nocard gain is YOUR signal; own~stranger => it's GENERIC triage competence any card confers.
    stranger = {d: nuwa[sorted([b for b in authors if grp[b] != grp[d]],
                key=lambda b: hashlib.sha1(f"str-{d}-{b}".encode()).hexdigest())[0]] for d in authors}
    arms = arms + ["stranger"]

    def card_of(arm, d):
        if arm == "nocard":
            return None
        if arm == "indiv":
            return nuwa[d]
        if arm == "stranger":
            return stranger[d]
        if arm.startswith("cmd@"):
            return cmd_card(d)
        if arm.startswith("concat@"):
            return concat_card(d)
        return step2[arm][d]

    # held-out bugs: first MAXBUGS per dev (devs with >=1 solved bug)
    bugs = {d: pool[d].get("solved_bugs", [])[:MAXBUGS] for d in authors}
    devs = [d for d in authors if bugs[d]]
    units = [(d, bi) for d in devs for bi in range(len(bugs[d]))]            # held-out (dev,bug) — SAME across arms (paired)
    print(f"20-MAD utility: {len(devs)} devs x <= {MAXBUGS} bugs = {len(units)} held-out bugs | arms={arms} | "
          f"5-class chance=0.20 | predictor={GEN}", flush=True)

    if os.environ.get("PILOT_DRYRUN"):
        ncalls = len(arms) * len(units)
        print(f"DRYRUN: {ncalls} deepseek predictions (~${ncalls * 0.0002:.2f}; temp0 -> cached, reruns free).", flush=True)
        print(f"  -> results/mad/mad_utility.json (acc/arm + arm-nocard + arm-indiv, dev-cluster CI)", flush=True)
        return

    jobs = [(arm, d, bi) for arm in arms for (d, bi) in units]
    hits = {}
    for (arm, d, bi), hit in zip(jobs, de.pool(lambda j: predict_res(card_of(j[0], j[1]), bugs[j[1]][j[2]].get("report", ""),
                                                                      bugs[j[1]][j[2]].get("stub", "")) == bugs[j[1]][j[2]]["resolution"],
                                               jobs)):
        hits[(arm, d, bi)] = 1.0 if hit else 0.0

    g = [d for (d, bi) in units]
    acc = {arm: [hits[(arm, d, bi)] for (d, bi) in units] for arm in arms}
    out = {"N_devs": len(devs), "n_bugs": len(units), "maxbugs": MAXBUGS, "k_cmd": KCMD, "chance": 0.2, "per_arm": {}, "vs": {}}

    print(f"\n=== (acc; 5-class chance 0.20; CI resamples devs) ===", flush=True)
    for arm in arms:
        ci = cluster_mean_ci(acc[arm], g, seed=SEED)
        out["per_arm"][arm] = {"acc": round(float(np.mean(acc[arm])), 3), "ci": [round(c, 3) for c in ci]}
        print(f"  {arm:9s} acc={np.mean(acc[arm]):.3f} CI{[round(c,3) for c in ci]}", flush=True)

    print(f"\n=== paired diffs (CI resamples devs; EXCL0 = CI excludes 0 = significant) ===", flush=True)
    for x in arms:
        if x == "nocard":
            continue
        for y in ("nocard", "indiv"):
            if x == y:
                continue
            r = cluster_paired_diff_ci(acc[x], acc[y], g, seed=SEED)
            sig = "  <-EXCL0" if (r["ci"][0] > 0 or r["ci"][1] < 0) else ""
            out["vs"][f"{x}-{y}"] = {"diff": round(float(r["diff"]), 3), "ci": [round(c, 3) for c in r["ci"]], "sig": bool(sig)}
            print(f"  {x:9s} - {y:7s} = {r['diff']:+.3f} CI{[round(c,3) for c in r['ci']]}{sig}", flush=True)
    # person-specificity: own (indiv) vs a stranger's card. >0 SIG => YOUR card beats a stranger's = person-specific value;
    # ~0 => the +nocard gain is GENERIC triage competence any card confers (= the prior "20-MAD 无个人价值" claim).
    r = cluster_paired_diff_ci(acc["indiv"], acc["stranger"], g, seed=SEED)
    sig = "  <-EXCL0" if (r["ci"][0] > 0 or r["ci"][1] < 0) else ""
    out["vs"]["indiv-stranger"] = {"diff": round(float(r["diff"]), 3), "ci": [round(c, 3) for c in r["ci"]], "sig": bool(sig)}
    print(f"  {'OWN-vs-STRANGER (indiv-stranger)'} = {r['diff']:+.3f} CI{[round(c,3) for c in r['ci']]}{sig}", flush=True)

    out["note"] = ("20-MAD OBJECTIVE utility = bug-resolution-prediction accuracy (no judge). arm-nocard>0 SIG => card "
                   "helps the task; arm-indiv ~0 => keeps the personal card's utility. Compare to Enron pairwise-competence.")
    (RES / "mad_utility.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> results/mad/mad_utility.json", flush=True)


if __name__ == "__main__":
    main()
