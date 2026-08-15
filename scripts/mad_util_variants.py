"""Stage 1b: 20-MAD utility for the shared-card VARIANTS vs baseline vs comparison methods, all in ONE
paired bootstrap. predict_res is byte-identical to mad_utility.predict_res (same cached protocol).
New-variant cards -> cache miss -> real deepseek spend (~$0.2/arm x 3 = ~$0.6). Baseline/methods cached (free).

Reads the ISOLATED variant files built by mad_synth_utility.py; baseline cmd_shared_cards_mad.json untouched.

Run:  DATASET=mad python scripts/mad_util_variants.py   ->  results/mad/mad_util_variants.json
"""
import os
import sys
import json
import hashlib
import statistics
from pathlib import Path

import numpy as np

os.environ.setdefault("DATASET", "mad")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cmd_gate as CG  # noqa: E402
import deid_enron as de  # noqa: E402
from src.llm import chat  # noqa: E402
from src.attrib_metrics import cluster_mean_ci, cluster_paired_diff_ci  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GEN = "deepseek-chat"
PRED_MAXTOK = 6
MAXBUGS = 8
K, S = 8, 0
CLASSES = ["FIXED", "WONTFIX", "INVALID", "DUPLICATE", "WORKSFORME"]
METHODS = ["staab", "staab_r1", "tpar_t10", "tpar_t15", "petre_k4"]


def predict_res(card, report, stub):
    """EXACT replica of mad_utility.predict_res (deepseek temp0 -> cached)."""
    prof = f"Developer triage profile:\n{card}\n\n" if card else ""
    out = (chat([{"role": "system", "content": "You triage software bugs. Predict the most likely RESOLUTION."},
                 {"role": "user", "content": f"{prof}Bug:\n{stub}\n{report}\n\nWhat is the most likely resolution? "
                  f"Answer ONLY ONE of: {', '.join(CLASSES)}."}],
                model=GEN, temperature=0.0, max_tokens=PRED_MAXTOK) or "").upper()
    for c in CLASSES:
        if c in out:
            return c
    return None


def _load(name):
    p = CG.SE / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    pool, authors, nuwa, aggro, _r, _w = CG.load()
    step2 = json.loads(CG.STEP2C.read_text(encoding="utf-8"))
    grp, byc = CG.make_groups(aggro, authors, K, S)
    base = _load("cmd_shared_cards_mad.json")
    util = _load("cmd_shared_cards_mad__util.json")
    neut = _load(os.environ.get("NEUTRALC", "cmd_shared_cards_mad__neutral.json"))
    pad = _load("cmd_shared_cards_mad__pad.json")
    concat = _load("cmd_concat_cards_mad.json")
    concat_neutral = _load("cmd_concat_cards_mad__neutral.json")   # fair length-matched concat (same neutral relaxation)
    conspf = _load(os.environ.get("CONSPFC", "cmd_shared_cards_mad__conspf.json"))   # consensus_pf (full-replacement CMD)
    for nm, d in [("util", util), ("neutral", neut), ("pad", pad)]:
        miss = [c for c in byc if f"k{K}_s{S}_{c}" not in d]
        if miss:
            sys.exit(f"variant '{nm}' missing clusters {miss} — run mad_synth_utility.py first.")

    def shk(d):
        return f"k{K}_s{S}_{grp[d]}"

    import hashlib as _h
    stranger = {d: nuwa[sorted([b for b in authors if grp[b] != grp[d]],
                key=lambda b: _h.sha1(f"str-{d}-{b}".encode()).hexdigest())[0]] for d in authors}

    SRC = {
        "nocard": lambda d: None,
        "indiv": lambda d: nuwa[d],
        "cmd_base": lambda d: base[shk(d)],
        "cmd_neutral": lambda d: neut[shk(d)],
        "cmd_util": lambda d: util[shk(d)],
        "cmd_pad": lambda d: pad[shk(d)],
        "stranger": lambda d: stranger[d],
    }
    for m in METHODS:
        if m in step2 and all(x in step2[m] for x in authors):
            SRC[m] = (lambda mm: (lambda d: step2[mm][d]))(m)
    if concat and all(f"k{K}_s{S}_{c}" in concat for c in byc):
        SRC["concat"] = lambda d: concat[shk(d)]
    if concat_neutral and all(f"k{K}_s{S}_{c}" in concat_neutral for c in byc):
        SRC["concat_neutral"] = lambda d: concat_neutral[shk(d)]
    if conspf and all(f"k{K}_s{S}_{c}" in conspf for c in byc):
        SRC["cmd_conspf"] = lambda d: conspf[shk(d)]
    arms = list(SRC)

    bugs = {d: pool[d].get("solved_bugs", [])[:MAXBUGS] for d in authors}
    devs = [d for d in authors if bugs[d]]
    units = [(d, bi) for d in devs for bi in range(len(bugs[d]))]
    print(f"20-MAD util variants: {len(devs)} devs x <= {MAXBUGS} bugs = {len(units)} units | arms={arms}\n"
          f"  (new variant cards -> deepseek MISS = paid ~$0.6; baseline/methods cached)", flush=True)

    jobs = [(arm, d, bi) for arm in arms for (d, bi) in units]
    res = de.pool(lambda j: predict_res(SRC[j[0]](j[1]), bugs[j[1]][j[2]].get("report", ""),
                                        bugs[j[1]][j[2]].get("stub", "")), jobs)
    hits = {(arm, d, bi): (1.0 if r == bugs[d][bi]["resolution"] else 0.0)
            for (arm, d, bi), r in zip(jobs, res)}

    g = [d for (d, bi) in units]
    acc = {arm: [hits[(arm, d, bi)] for (d, bi) in units] for arm in arms}

    # median card lengths (None for nocard)
    def med_len(arm):
        vals = [len(SRC[arm](d)) for d in devs if SRC[arm](d)]
        return int(statistics.median(vals)) if vals else 0

    out = {"k": K, "seed": S, "n_units": len(units), "per_arm": {}, "vs": {}, "median_len": {}}
    print("\n=== per-arm acc (5-class chance 0.20) + median card len ===", flush=True)
    for arm in arms:
        ci = cluster_mean_ci(acc[arm], g, seed=S)
        ml = med_len(arm)
        out["per_arm"][arm] = {"acc": round(float(np.mean(acc[arm])), 3), "ci": [round(c, 3) for c in ci]}
        out["median_len"][arm] = ml
        print(f"  {arm:12s} acc={np.mean(acc[arm]):.3f} CI{[round(c,3) for c in ci]}  len={ml}", flush=True)

    def paired(x, y, label):
        r = cluster_paired_diff_ci(acc[x], acc[y], g, seed=S)
        excl0 = r["ci"][0] > 0 or r["ci"][1] < 0
        out["vs"][label] = {"diff": round(float(r["diff"]), 3), "ci": [round(c, 3) for c in r["ci"]], "sig": bool(excl0)}
        tag = "  EXCL0" if excl0 else "  ∋0"
        print(f"  {label:28s} = {r['diff']:+.3f} CI{[round(c,3) for c in r['ci']]}{tag}", flush=True)

    print("\n=== isolation (primary = cmd_neutral - cmd_base, pre-registered >=0) ===", flush=True)
    paired("cmd_neutral", "cmd_base", "cmd_neutral - cmd_base")      # total: length+substance, no leading words
    paired("cmd_neutral", "cmd_pad", "cmd_neutral - cmd_pad")        # PRIMARY: substance at MATCHED length (isolates from length)
    paired("cmd_pad", "cmd_base", "cmd_pad - cmd_base")              # pure-length effect (padded filler)
    paired("cmd_util", "cmd_neutral", "cmd_util - cmd_neutral")      # wording-inflation probe (should ~0)
    paired("cmd_util", "cmd_base", "cmd_util - cmd_base")            # total effect of util variant

    print("\n=== variants vs indiv ceiling ===", flush=True)
    for v in ("cmd_base", "cmd_neutral", "cmd_util", "cmd_pad"):
        paired(v, "indiv", f"{v} - indiv")

    print("\n=== best variant (neutral) vs each comparison method ===", flush=True)
    for m in METHODS + [x for x in ("concat", "concat_neutral") if x in arms]:
        if m in arms:
            paired("cmd_neutral", m, f"cmd_neutral - {m}")

    if "cmd_conspf" in arms:                                          # G3: consensus_pf full-replacement utility gate
        print("\n=== G3: consensus_pf (full-replacement CMD) ===", flush=True)
        paired("cmd_conspf", "cmd_neutral", "cmd_conspf - cmd_neutral")   # regression check (want ∋0)
        paired("cmd_conspf", "nocard", "cmd_conspf - nocard")            # utility preserved (want EXCL0 >0)
        paired("cmd_conspf", "indiv", "cmd_conspf - indiv")             # vs ceiling

    out["note"] = ("Stage-1 utility for synth variants. PRIMARY = cmd_neutral-cmd_base (>=0 pre-reg). "
                   "cmd_util-cmd_neutral ~0 => leading words don't inflate; cmd_pad-cmd_base ~0 => not just length. "
                   "Anonymity NOT tested here (Stage 2, sonnet 2AFC).")
    _outn = os.environ.get("OUT", "mad_util_variants.json")
    (ROOT / "results" / "mad" / _outn).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> results/mad/{_outn}", flush=True)


if __name__ == "__main__":
    main()
