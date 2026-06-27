"""Score the 2AFC paired-discrimination attacker. Metric = 2AFC accuracy (chance=0.5), cluster-bootstrap CI over
card_id. Gates: (1) indiv channel = POSITIVE CONTROL, must be >> 0.5 or the attacker is broken -> result VOID;
(2) shared/nneg = IDENTITY headline (topic controlled); (3) flip-null: re-label member<->stranger at random ->
must collapse to 0.5 (validates construction/parsing add no leak). Reads results[/{ds}]/_2afc_k{K}_s{S}_{model}.json.

Run: DATASET=mad KCL=8 SEED=0 ATTACK_MODEL=deepseek-chat python scripts/cmd_attack2afc_score.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DS = os.environ.get("DATASET", "mad")
RES = ROOT / "results" if DS == "enron" else ROOT / "results" / DS
RES = (ROOT / os.environ["RESDIR"]) if os.environ.get("RESDIR") else RES
KCL = os.environ.get("KCL", "8")
SEED = os.environ.get("SEED", "0")
MODEL = os.environ.get("ATTACK_MODEL", "deepseek-chat").replace("/", "_")
NNEG_MATCH = os.environ.get("NNEG_MATCH", "member")

recs = json.loads((RES / f"_2afc_k{KCL}_s{SEED}_{MODEL}_{NNEG_MATCH}.json").read_text(encoding="utf-8"))


def boot_acc(sub, nboot=3000, seed=0, flip=False):
    """cluster-bootstrap mean 2AFC accuracy over card_id. flip=True -> randomly relabel member/stranger (null)."""
    by = {}
    for r in sub:
        by.setdefault(r["card_id"], []).append(r)
    cards = list(by)
    rng = np.random.default_rng(seed)
    point = float(np.mean([_correct(r, rng, flip) for r in sub]))
    vals = []
    for _ in range(nboot):
        samp = []
        for c in rng.choice(cards, len(cards), replace=True):
            samp += by[c]
        vals.append(np.mean([_correct(r, rng, flip) for r in samp]))
    lo, hi = float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
    return point, lo, hi


def _correct(r, rng, flip):
    p = r["p_member"]
    if flip and rng.random() < 0.5:      # flip-null: half the time the "member" was actually the stranger
        p = 1.0 - p
    return 1.0 if p > 0.5 else (0.5 if p == 0.5 else 0.0)


def cell(chan, neg, flip=False):
    sub = [r for r in recs if r["chan"] == chan and r["neg"] == neg]
    if not sub:
        return None
    pt, lo, hi = boot_acc(sub, flip=flip)
    pm = float(np.mean([r["p_member"] for r in sub]))
    ncards = len(set(r["card_id"] for r in sub))
    sig = "SIG>0.5" if lo > 0.5 else ("SIG<0.5" if hi < 0.5 else "≈0.5 (CI∋0.5)")
    return {"acc": pt, "ci": [round(lo, 3), round(hi, 3)], "p_member": round(pm, 3),
            "n": len(sub), "ncards": ncards, "sig": sig}


def main():
    print(f"2AFC attacker scoring  DS={DS} k{KCL} s{SEED} model={MODEL} nneg~{NNEG_MATCH}  "
          f"(chance=0.5; CI=cluster-bootstrap over card)")
    print("ladder: raw (pure authorship) -> indiv (card abstraction) -> shared (pooling); "
          "nneg=topic-controlled IDENTITY, rneg=topic-gameable\n")
    summary = {}
    for chan in ("raw", "indiv", "shared"):
        for neg in ("nneg", "rneg"):
            c = cell(chan, neg)
            if c is None:
                continue
            summary[f"{chan}_{neg}"] = c
            print(f"  {chan:6s}/{neg}: acc={c['acc']:.3f} CI{c['ci']} P(member)={c['p_member']:.3f}  "
                  f"n={c['n']} cards={c['ncards']}  {c['sig']}")
    # flip-null sanity on the headline cell
    fn = cell("shared", "nneg", flip=True)
    if fn:
        print(f"\n  flip-null shared/nneg: acc={fn['acc']:.3f} CI{fn['ci']}  (must ∋0.5; else construction leaks)")

    print("\n--- verdict ---")
    pc = summary.get("raw_rneg") or summary.get("raw_nneg") or summary.get("indiv_rneg")
    if pc:
        ok = pc["ci"][0] > 0.5
        print(f"POS-CTRL raw: acc={pc['acc']:.3f} CI{pc['ci']} -> {'OK (attacker CAN do authorship at all)' if ok else 'FAIL -> attacker too weak / refs too short, VOID'}")
    # identity-destruction ladder on the clean topic-controlled metric (nneg)
    ladder = [(c, summary.get(f"{c}_nneg")) for c in ("raw", "indiv", "shared")]
    ladder = [(c, x["acc"]) for c, x in ladder if x]
    if len(ladder) >= 2:
        print("LADDER (nneg acc): " + " -> ".join(f"{c}:{a:.3f}" for c, a in ladder)
              + "   (drop = identity destroyed by that step)")
    sh = summary.get("shared_nneg")
    if sh:
        if sh["ci"][0] > 0.5:
            v = f"shared/nneg acc {sh['acc']:.3f} CI{sh['ci']} > 0.5 -> IDENTITY LEAKS under the card-conditioned attacker"
        elif sh["ci"][1] < 0.5:
            v = f"shared/nneg acc {sh['acc']:.3f} CI{sh['ci']} < 0.5 -> below chance (attacker anti-correlated; inspect)"
        else:
            v = f"shared/nneg acc {sh['acc']:.3f} CI{sh['ci']} ∋ 0.5 -> identity NOT recoverable (anonymity holds vs this attacker)"
        print(f"HEADLINE shared/nneg: {v}")
    summary["_meta"] = {"ds": DS, "k": KCL, "seed": SEED, "model": MODEL, "nneg_match": NNEG_MATCH}
    outp = RES / f"_2afc_score_k{KCL}_s{SEED}_{MODEL}_{NNEG_MATCH}.json"
    outp.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\nsaved -> {outp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
