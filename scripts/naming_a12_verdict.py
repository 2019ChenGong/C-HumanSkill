"""A12 (#153/#154) verdict computation — pre-registered in results/NAMING_A12_POOLBASE_PREREG.md.

Pure re-scoring of already-frozen answers: reads the `_naming_depth.json` / `_naming_auc.json`
sidecars and produces exactly the quantities the prereg names, in its order:

  §5.1 GATE    in-wave `indiv` pooled depth excess, CI-lo > 0 (else the WHOLE package is void)
  §5.2 LICENSE in-wave `indiv` vs legacy naming_v6 `indiv`, PER-OWNER paired depth diff,
               pooled CI must CONTAIN 0 -> unlocks pairing ne/concat against the legacy v6 answers
  §5.3 ANCHOR  in-wave `v6` (bseed 0 slice) vs legacy naming_v6 `v6` bseed-0 rows, per-cluster
               paired; CI containing 0 is corroboration only, CI excluding 0 downgrades C2/C3
  §6   C1 = concat - ne   (PRIMARY; both arms in-wave, does not depend on §5.2)
       C2 = ne     - v6   (secondary; needs §5.2)
       C3 = concat - v6   (descriptive)
  §6.4 the full 3 comparisons x 3 readouts x (pooled + 3 datasets) = 36-cell grid, listed in full.

Pairing is EXACT, never positional: depth/final rows carry `row_ids` ("{member}#{bseed}" and
"{target}#{bseed}") and two arms are compared only on the intersection of their row-id sets; the
symmetric difference is printed whenever it is non-empty.

Readouts (prereg §4, three in parallel, no cherry-picking):
  depth  paired per-member depth excess difference
  rank1  paired per-bracket final-hit difference (= TSA Rank@1 on this tournament)
  auc    per-cluster Mann-Whitney AUC (S1 score, ties 0.5) computed for each arm, then differenced.
         NOTE: this is the per-cluster-AUC contrast, NOT the stratified pooled AUC estimator that
         scripts/naming_auc_pooled.py reports; it is the pairable form of the same readout.

Statistic = cluster (ds:cluster, 39 pooled). Cluster bootstrap n=5000, seeds {0,1,2}; a cell counts
as SIGNAL only if all three seeds agree on the sign (prereg §4).

  python -P scripts/naming_a12_verdict.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = {"enron": "enron", "mad": "mad", "cv": "se"}
INWAVE = "naming_pool_ab"
LEGACY_V6 = ("naming_v6", "naming_v6_more")   # bseeds 0,1 + 2,3 -> the 4-bseed legacy v6
LEGACY_IND = "naming_v6"                      # byte-identical indiv brackets (verified pre-dispatch)
BOOT_N, SEEDS = 5000, (0, 1, 2)
DELTAS = (0.10, 0.05)
READOUTS = ("depth", "rank1", "auc")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ----------------------------------------------------------------- loaders
def load_rows(ds, sub, arm, sec_key):
    """-> {unit: {row_id: value}} from a `_naming_depth.json` section ('depth' or 'final')."""
    p = ROOT / f"results/{DIRS[ds]}/{sub}/_naming_depth.json"
    if not p.exists():
        return {}
    sec = json.loads(p.read_text(encoding="utf-8")).get("arms", {}).get(arm, {}).get(sec_key)
    if not sec:
        return {}
    assert "row_ids" in sec, f"{p} [{arm}/{sec_key}] predates row_ids — rerun scripts/naming_depth.py"
    out = {}
    for u, vals in sec["units"].items():
        ids = sec["row_ids"][u]
        assert len(ids) == len(vals), f"{p} {arm}/{sec_key} {u}: row_ids/units length mismatch"
        d = out.setdefault(u, {})
        for rid, v in zip(ids, vals):
            assert rid not in d, f"{p} {arm}/{sec_key} {u}: duplicate row id {rid}"
            d[rid] = v
    return out


def load_auc_pairs(ds, sub, arm):
    """-> {unit: [(score, label), ...]} from the S1 variant of `_naming_auc.json`."""
    p = ROOT / f"results/{DIRS[ds]}/{sub}/_naming_auc.json"
    if not p.exists():
        return {}
    sec = json.loads(p.read_text(encoding="utf-8")).get("arms", {}).get(arm, {}).get("S1")
    if not sec:
        return {}
    return {u: [(float(s), int(l)) for s, l in v] for u, v in sec["units"].items()}


def merge_rows(*maps):
    out = {}
    for m in maps:
        for u, rows in m.items():
            d = out.setdefault(u, {})
            for rid, v in rows.items():
                assert rid not in d, f"row id {rid} appears in two source dirs for unit {u}"
                d[rid] = v
    return out


def merge_pairs(*maps):
    out = {}
    for m in maps:
        for u, v in m.items():
            out.setdefault(u, []).extend(v)
    return out


def cluster_auc(pairs):
    """Mann-Whitney AUC with ties counted 0.5; None when a class is absent."""
    pos = np.array([s for s, l in pairs if l == 1], dtype=float)
    neg = np.array([s for s, l in pairs if l == 0], dtype=float)
    if not len(pos) or not len(neg):
        return None
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / (len(pos) * len(neg)))


# ----------------------------------------------------------------- statistics
def boot(ulist, seed):
    """Cluster bootstrap, same convention as naming_pooled_depth.boot."""
    allv = [x for u in ulist for x in u]
    rng = np.random.default_rng(seed)
    means = [np.mean([x for i in rng.integers(0, len(ulist), len(ulist)) for x in ulist[i]])
             for _ in range(BOOT_N)]
    return (float(np.mean(allv)), float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)), len(allv), len(ulist))


def multiseed(ulist, label):
    if not ulist:
        return {"label": label, "est": float("nan"), "ci": [float("nan")] * 2, "n_rows": 0,
                "n_units": 0, "seeds_agree": True, "verdict": "n/a"}
    runs = [boot(ulist, s) for s in SEEDS]
    lo, hi = min(r[1] for r in runs), max(r[2] for r in runs)      # conservative envelope
    signs = {(r[1] > 0) - (r[2] < 0) for r in runs}
    return {"label": label, "est": round(runs[0][0], 4), "ci": [round(lo, 4), round(hi, 4)],
            "n_rows": runs[0][3], "n_units": runs[0][4],
            "per_seed_ci": [[round(r[1], 4), round(r[2], 4)] for r in runs],
            "seeds_agree": len(signs) == 1,
            "verdict": "pos" if lo > 0 else "neg" if hi < 0 else "zero"}


def paired_units(a, b, restrict=None):
    """-> (per-cluster lists of exact paired row differences, list of unpaired-row reports)."""
    units, missing = [], []
    for u in sorted(set(a) & set(b)):
        ra, rb = a[u], b[u]
        if restrict is not None:
            ra = {k: v for k, v in ra.items() if restrict(k)}
            rb = {k: v for k, v in rb.items() if restrict(k)}
        if set(ra) != set(rb):
            missing.append((u, sorted(set(ra) ^ set(rb))[:4]))
        common = sorted(set(ra) & set(rb))
        if common:
            units.append([ra[k] - rb[k] for k in common])
    return units, missing


def paired_auc_units(a, b):
    """-> per-cluster lists holding ONE value each: AUC(a) - AUC(b) for that cluster."""
    units = []
    for u in sorted(set(a) & set(b)):
        x, y = cluster_auc(a[u]), cluster_auc(b[u])
        if x is not None and y is not None:
            units.append([x - y])
    return units, []


def show(r, rule=""):
    star = "" if r["seeds_agree"] else "  !SEED-DISAGREEMENT"
    print(f"  {r['label']:34s} {r['est']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  "
          f"(n={r['n_rows']}, {r['n_units']} clusters)  {r['verdict']:4s}{rule}{star}")
    if r.get("unpaired_rows"):
        print(f"      ! unpaired rows in {len(r['unpaired_rows'])} cluster(s): {r['unpaired_rows'][:2]}")


def main():
    inw, leg_v6, leg_ind = {}, {}, {}
    for key, sec in (("depth", "depth"), ("rank1", "final")):
        inw[key] = {a: {ds: load_rows(ds, INWAVE, a, sec) for ds in DIRS}
                    for a in ("indiv", "v6", "ne", "concat")}
        leg_v6[key] = {ds: merge_rows(*(load_rows(ds, s, "v6", sec) for s in LEGACY_V6)) for ds in DIRS}
        leg_ind[key] = {ds: load_rows(ds, LEGACY_IND, "indiv", sec) for ds in DIRS}
    inw["auc"] = {a: {ds: load_auc_pairs(ds, INWAVE, a) for ds in DIRS}
                  for a in ("indiv", "v6", "ne", "concat")}
    leg_v6["auc"] = {ds: merge_pairs(*(load_auc_pairs(ds, s, "v6") for s in LEGACY_V6)) for ds in DIRS}

    out = {"prereg": "results/NAMING_A12_POOLBASE_PREREG.md",
           "boot": {"n": BOOT_N, "seeds": list(SEEDS), "unit": "ds:cluster"}}

    # ---------- §5.1 positive-control gate (in-wave indiv only) ----------
    print("=== §5.1 GATE — IN-WAVE indiv pooled depth excess (CI-lo > 0 required) ===")
    pooled_ind = [list(rows.values()) for ds in DIRS for rows in inw["depth"]["indiv"][ds].values() if rows]
    g = multiseed(pooled_ind, "indiv (in-wave, pooled)")
    show(g, "  <-- GATE " + ("PASS" if g["verdict"] == "pos" and g["seeds_agree"] else "**VOID**"))
    out["gate_5_1"] = {"pooled": g, "per_dataset": {}}
    for ds in DIRS:
        r = multiseed([list(v.values()) for v in inw["depth"]["indiv"][ds].values() if v],
                      f"  per-dataset {ds}")
        show(r); out["gate_5_1"]["per_dataset"][ds] = r
    gate_ok = g["verdict"] == "pos" and g["seeds_agree"]
    indiv_effect = g["est"]

    # ---------- §5.2 cross-wave license (per-owner paired) ----------
    print("\n=== §5.2 LICENSE — in-wave indiv MINUS legacy naming_v6 indiv, per-owner paired "
          "(CI must CONTAIN 0) ===")
    out["license_5_2"] = {"per_dataset": {}}
    allu = []
    for ds in DIRS:
        u, miss = paired_units(inw["depth"]["indiv"][ds], leg_ind["depth"][ds])
        r = multiseed(u, f"  per-dataset {ds}"); r["unpaired_rows"] = miss
        show(r); out["license_5_2"]["per_dataset"][ds] = r; allu += u
    lic = multiseed(allu, "indiv in-wave - legacy (pooled)")
    show(lic, "  <-- LICENSE " + ("GRANTED" if lic["verdict"] == "zero" else "**REVOKED**"))
    out["license_5_2"]["pooled"] = lic
    license_ok = lic["verdict"] == "zero"

    # ---------- §5.3 in-wave v6 anchor (pooled channel) ----------
    print("\n=== §5.3 ANCHOR — in-wave v6 (bseed 0) MINUS legacy v6 (bseed 0), per-cluster paired ===")
    out["anchor_5_3"] = {"per_dataset": {}}
    b0 = lambda k: k.endswith("#0")                                       # noqa: E731
    allu = []
    for ds in DIRS:
        u, miss = paired_units(inw["depth"]["v6"][ds], leg_v6["depth"][ds], restrict=b0)
        r = multiseed(u, f"  per-dataset {ds}"); r["unpaired_rows"] = miss
        show(r); out["anchor_5_3"]["per_dataset"][ds] = r; allu += u
    anc = multiseed(allu, "v6 in-wave - legacy (pooled)")
    hw = (anc["ci"][1] - anc["ci"][0]) / 2
    show(anc, "  <-- ANCHOR " + ("consistent" if anc["verdict"] == "zero" else "**DRIFT**"))
    print(f"      power bound: CI half-width {hw:.4f} = {hw / indiv_effect:.2f} x the in-wave indiv "
          f"effect ({indiv_effect:+.4f}) — a drift smaller than this is NOT ruled out")
    anc["half_width"], anc["half_width_over_indiv_effect"] = round(hw, 4), round(hw / indiv_effect, 3)
    out["anchor_5_3"]["pooled"] = anc
    anchor_ok = anc["verdict"] == "zero"

    # ---------- §6 + §6.4: the full 3 x 3 x 4 grid ----------
    print("\n=== §6 / §6.4 — 3 comparisons x 3 readouts x (pooled + 3 datasets) = 36 cells, all listed ===")
    comps = [("C1", "concat", "ne", "concat", "ne", "PRIMARY (in-wave, no §5.2 dependency)"),
             ("C2", "ne", "v6(legacy)", "ne", None, "secondary — needs §5.2"),
             ("C3", "concat", "v6(legacy)", "concat", None, "descriptive")]
    out["comparisons"], ncell = {}, 0
    for tag, an, bn, a_arm, b_arm, role in comps:
        print(f"\n  ==== {tag}: {an} - {bn}   [{role}] ====")
        out["comparisons"][tag] = {"arms": [an, bn], "role": role, "readouts": {}}
        for ro in READOUTS:
            A = inw[ro][a_arm]
            B = inw[ro][b_arm] if b_arm else leg_v6[ro]
            pooled, cells = [], {}
            for ds in DIRS:
                u, miss = (paired_auc_units(A[ds], B[ds]) if ro == "auc"
                           else paired_units(A[ds], B[ds]))
                r = multiseed(u, f"  {ro:5s} {ds}"); r["unpaired_rows"] = miss
                cells[ds] = r; pooled += u; ncell += 1
            p = multiseed(pooled, f"  {ro:5s} POOLED")
            ncell += 1
            for ds in DIRS:
                show(cells[ds])
            main_tag = "  <== MAIN VERDICT" if (tag == "C1" and ro == "depth") else ""
            show(p, main_tag)
            if ro == "depth":
                hw = (p["ci"][1] - p["ci"][0]) / 2
                print(f"      power bound: CI half-width {hw:.4f} = {hw / indiv_effect:.2f} x the "
                      f"in-wave indiv effect")
                p["half_width_over_indiv_effect"] = round(hw / indiv_effect, 3)
            out["comparisons"][tag]["readouts"][ro] = {"pooled": p, "per_dataset": cells}
    print(f"\n  grid size actually printed: {ncell} cells (prereg §6.4 says 36)")
    out["grid_cells"] = ncell

    # ---------- §6.3 delta-certification ----------
    print("\n=== §6.3 delta-certification (AUC S1 per dataset; up95 over seeds {0,1,2} < .5 + delta) ===")
    out["certification"] = {}
    for ds in DIRS:
        p = ROOT / f"results/{DIRS[ds]}/{INWAVE}/_naming_auc.json"
        if not p.exists():
            print(f"  {ds}: no _naming_auc.json — run scripts/naming_auc.py first"); continue
        s = json.loads(p.read_text(encoding="utf-8"))
        for arm in ("v6", "ne", "concat"):
            rec = s.get("arms", {}).get(arm, {}).get("S1")
            if not rec:
                continue
            up = max([c[1] for c in rec.get("ci_by_seed", {}).values()] or [rec["ci"][1]])
            flags = "  ".join(f"d={d:.2f}:{'PASS' if up < .5 + d else 'FAIL'}" for d in DELTAS)
            print(f"  {ds:6s} {arm:7s} AUC {rec['auc']:.4f}  up95 {up:.4f}   {flags}")
            out["certification"].setdefault(ds, {})[arm] = {
                "auc": rec["auc"], "up95_max_over_seeds": up,
                "cert": {f"{d:.2f}": bool(up < .5 + d) for d in DELTAS}}

    # ---------- roll-up ----------
    print("\n=== ROLL-UP ===")
    print(f"  §5.1 gate      : {'PASS' if gate_ok else '**VOID — everything below is unusable**'}")
    print(f"  §5.2 license   : {'GRANTED (C2/C3 judgeable)' if license_ok else '**REVOKED**'}")
    print(f"  §5.3 anchor    : {'consistent' if anchor_ok else '**DRIFT — C2/C3 downgraded**'}")
    c1 = out["comparisons"]["C1"]["readouts"]["depth"]["pooled"]
    verdict = {"pos": "consensus synthesis significantly LOWERS re-identifiability vs naive union",
               "neg": "concat is MORE anonymous than CMD — core claim REVERSES on this ruler",
               "zero": "no detectable difference — the claim does NOT transfer to the naming ruler"}[c1["verdict"]]
    print(f"  §6.1 C1 (main) : {c1['est']:+.4f} [{c1['ci'][0]:+.4f},{c1['ci'][1]:+.4f}]  -> {verdict}")
    out["rollup"] = {"gate_pass": gate_ok, "license_granted": license_ok,
                     "anchor_consistent": anchor_ok, "c1_verdict": c1["verdict"],
                     "c1_point_sign": "negative (leans concat-more-anonymous)" if c1["est"] < 0
                                      else "positive (leans CMD-more-anonymous)"}

    p = ROOT / "results/_A12_poolbase_all.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
