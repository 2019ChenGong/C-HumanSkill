"""Pool a forced-choice contrast across multiple FC waves (seeds) and certify NONINFERIORITY at delta.

R11 (CV nec-tpar noninferiority sprint) and reusable for R2 (Enron ne-nec TIE sprint). Registered in
results/ELEMK_DESIGN.md, R11 appendix + review corrections. What this implements:

  * MAJOR-2: pid namespaces COLLIDE across packs (C{ci}... indexes each pack's own CONTRASTS list). So each
    pack is parsed in its own scope and merged only at the level of resolved per-(unit,wave) win values.
  * MAJOR-1: the per-person arm's drafts are byte-identical across waves (cached), so a unit's s0/s1 rows are
    correlated through the shared y-side text. Certification therefore requires BOTH poolers to clear:
      (B) cluster = (wave, pooling_cluster)   -- the seed-partition pooler (R1's B analog)
      (C) cluster = expert across waves       -- absorbs the reused-stimulus/person effect (R1's C analog)
    NONINFERIOR (X not worse than Y by >= delta) iff ci95-lo > 0.5 - delta on BOTH.
  * Battery gate: every pack's _fc_summary.json must exist with battery_pass true (score each wave first).
  * Per-wave estimates are reported alongside (selection transparency: the s0 wave was seen before this
    expansion was designed; the new wave's own row is the untouched-by-selection reading).

Unit value = mean over both orders x replicates (identical to cv_fc_score.agg). CI = cluster bootstrap
(cluster_mean_ci), se backed out of the CI width, sMDE = (Z_a+Z_b)*se, two-sided dictionary verdict beside
the noninferiority call.

Run:  PACKS="results/se/fc_v6r3=s0,results/se/fc_v6s1=s1" CONTRASTS="nec-tpar_t15,nec-staab" \
      [DELTA=0.10 NBOOT=20000 OUTFILE=results/cv_fc_multiseed_pool_d10.json] python -P scripts/fc_multiseed_pool.py
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

DELTA = float(os.environ.get("DELTA", "0.10"))
NBOOT = int(os.environ.get("NBOOT", "20000"))
ONLY = os.environ.get("ONLY", "")                 # e.g. ONLY=qwen -- pool a single judge replicate (R5)
PACKS = [p.split("=") for p in os.environ["PACKS"].split(",")]
CONTRASTS = [tuple(c.split("-")) for c in os.environ["CONTRASTS"].split(",")]
OUTFILE = ROOT / os.environ.get("OUTFILE", "results/cv_fc_multiseed_pool_d10.json")
# R12 (all three PACK-SCOPED, loud, and no-ops when unset):
#   DROP="r2:nec-tpar_t15"                  remove a contrast from ONE pack before aliasing (the R12 anchor
#                                           is analyzed separately and must never enter the pool)
#   ALIAS="r2:nec-tpar_t15_r2=nec-tpar_t15" rename a contrast key within ONE pack (MAJOR-2: a blanket rename
#                                           would silently blend the anchor and the rebuild rows)
#   TAGGROUP="s0=p0,r2=p0,s1=p1"            waves sharing a grouping PARTITION share x-side pooled cards (and
#                                           their cached drafts) -> add pooler (D) cluster=(partition,cluster)
#                                           (MAJOR-1); when set, certification needs (B)(C)(D) all clear.
DROP = defaultdict(list)
for _spec in filter(None, os.environ.get("DROP", "").split(",")):
    _tag, _key = _spec.split(":")
    DROP[_tag].append(tuple(_key.split("-")))
ALIAS = defaultdict(list)
for _spec in filter(None, os.environ.get("ALIAS", "").split(",")):
    _tag, _kv = _spec.split(":")
    _src, _dst = _kv.split("=")
    ALIAS[_tag].append((tuple(_src.split("-")), tuple(_dst.split("-"))))
TAGGROUP = dict(kv.split("=") for kv in filter(None, os.environ.get("TAGGROUP", "").split(",")))
Z_A, Z_B, Z_1 = 1.959964, 0.841621, 1.644854
ANS = re.compile(r"ans(?:_(\w+?))?_(\d+)$")
LO_BOUND = 0.5 - DELTA


def load_pack(path, tag):
    """One pack, own scope (pids never leave this function) -> {(x,y): [row]},
    row = {tag, unit, expert, cluster, value}."""
    B = ROOT / path
    meta = json.loads((B / "meta.json").read_text(encoding="utf-8"))
    cfg = json.loads((B / "config.json").read_text(encoding="utf-8"))
    summ = B / "_fc_summary.json"
    assert summ.exists(), f"{path}: score the wave first (cv_fc_score.py) -- battery gate needs _fc_summary.json"
    assert json.loads(summ.read_text(encoding="utf-8")).get("battery_pass") is True, \
        f"{path}: battery_pass is not true -- wave is void, re-judge before pooling"
    uclu, uexp = cfg["unit_cluster"], cfg["unit_expert"]
    picks = defaultdict(dict)
    for f in sorted(B.glob("ans*.json")):
        m = ANS.fullmatch(f.stem)
        if not m:
            continue
        rep = m.group(1) or "r1"
        if ONLY and rep != ONLY:
            continue
        for rec in json.loads(f.read_text(encoding="utf-8-sig")):
            c = str(rec.get("choice", "")).strip().upper()[:1]
            if c in ("A", "B") and rec.get("pid") in meta:
                picks[rec["pid"]][rep] = c
    out = defaultdict(lambda: defaultdict(list))     # (x,y) -> unit -> [win values]
    for pid, mt in meta.items():
        if mt["kind"] != "contrast" or pid not in picks:
            continue
        key = (mt["x"], mt["y"])
        for r, c in picks[pid].items():
            out[key][mt["unit"]].append(float((c == "A") == (mt.get("order", 0) == 0)))
    rows = {}
    for key, per_unit in out.items():
        rows[key] = [{"tag": tag, "unit": u, "expert": uexp[u], "cluster": uclu[u],
                      "value": float(np.mean(vs))} for u, vs in sorted(per_unit.items())]
    for key in DROP.get(tag, []):
        assert key in rows, f"{tag}: DROP {'-'.join(key)} not in pack (has {sorted('-'.join(k) for k in rows)})"
        print(f"  {tag}: DROPPED {'-'.join(key)} ({len(rows.pop(key))} units) -- excluded by registration")
    for src, dst in ALIAS.get(tag, []):
        assert src in rows, f"{tag}: ALIAS source {'-'.join(src)} not in pack"
        assert dst not in rows, (f"{tag}: ALIAS target {'-'.join(dst)} already present -- refusing to blend two "
                                 f"row-sets under one key (DROP the native rows first if that is registered)")
        rows[dst] = rows.pop(src)
        print(f"  {tag}: ALIASED {'-'.join(src)} -> {'-'.join(dst)} ({len(rows[dst])} units)")
    return rows


def estimate(rows, groups):
    v = np.array([r["value"] for r in rows])
    ci = cluster_mean_ci(v, groups, n_boot=NBOOT, seed=0)
    m = float(np.mean(v))
    se = (ci[1] - ci[0]) / (2 * Z_A)
    smde = (Z_A + Z_B) * se
    lo90, hi90 = m - Z_1 * se, m + Z_1 * se
    diff = ci[0] > 0.5 or ci[1] < 0.5
    equiv = (lo90 > LO_BOUND) and (hi90 < 0.5 + DELTA)
    noninf = ci[0] > LO_BOUND
    verdict = (f"DIFFERENT ({'X' if m > .5 else 'Y'} better)" if diff else
               f"TIE — |effect| > {DELTA} excluded" if equiv and smde < DELTA else
               f"UNDERPOWERED — sMDE {smde:.3f} >= delta" if smde >= DELTA else "inconclusive")
    return {"win": round(m, 3), "ci": [round(c, 3) for c in ci], "se": round(se, 4),
            "sampling_mde": round(smde, 3), "n_units": len(v), "n_clusters": len(set(groups)),
            "noninferior": bool(noninf), "two_sided_verdict": verdict}


def main():
    if TAGGROUP:
        unmapped = [t for _, t in PACKS if t not in TAGGROUP]
        assert not unmapped, f"TAGGROUP set but tags {unmapped} unmapped -- every pack needs a partition group"
    packs = {tag: load_pack(path, tag) for path, tag in PACKS}
    out = {"delta": DELTA, "lo_bound": LO_BOUND, "nboot": NBOOT,
           "packs": {t: p for p, t in PACKS}, "contrasts": {}}
    if DROP or ALIAS or TAGGROUP:      # provenance for the R12-style ops (absent when unset: gate-C neutral)
        out["ops"] = {"drop": {t: ["-".join(k) for k in v] for t, v in DROP.items()},
                      "alias": {t: [f"{'-'.join(s)}={'-'.join(d)}" for s, d in v] for t, v in ALIAS.items()},
                      "taggroup": TAGGROUP}
    npool = 3 if TAGGROUP else 2
    print(f"\nFC MULTISEED POOL  delta={DELTA}  noninferiority = ci95-lo > {LO_BOUND}  (all {npool} poolers must clear)")
    print("  (B) cluster=(wave,pooling_cluster)   (C) cluster=expert across waves"
          + ("   (D) cluster=(partition,pooling_cluster)" if TAGGROUP else "") + "\n")
    for key in CONTRASTS:
        name = "-".join(key)
        rows = [r for tag in packs for r in packs[tag].get(key, [])]
        if not rows:
            print(f"== {name}: no rows in any pack, skipped"); continue
        by_wave = {tag: [r for r in rows if r["tag"] == tag] for tag in packs}
        res = {"pooled_B": estimate(rows, [f"{r['tag']}|{r['cluster']}" for r in rows]),
               "pooled_C": estimate(rows, [r["expert"] for r in rows]),
               "per_wave": {tag: estimate(wr, [r["cluster"] for r in wr])
                            for tag, wr in by_wave.items() if wr}}
        if TAGGROUP:
            res["pooled_D"] = estimate(rows, [f"{TAGGROUP[r['tag']]}|{r['cluster']}" for r in rows])
        cert = all(res[p]["noninferior"] for p in ("pooled_B", "pooled_C") + (("pooled_D",) if TAGGROUP else ()))
        res["certified_noninferior"] = bool(cert)
        # R2 endpoint: a two-sided TIE claim reads THIS field, never the noninferiority boolean above
        # (m=.45 CI[.41,.49] is noninferior but NOT a tie -- it is DIFFERENT(Y better) and must be
        # reported by name).
        res["certified_tie_B"] = res["pooled_B"]["two_sided_verdict"].startswith("TIE")
        out["contrasts"][name] = res
        print(f"== {name}")
        for lab, e in (("(B) wave-cluster", res["pooled_B"]), ("(C) person     ", res["pooled_C"])) \
                + ((("(D) partition   ", res["pooled_D"]),) if TAGGROUP else ()):
            print(f"   {lab}  win={e['win']:.3f} CI[{e['ci'][0]:.3f},{e['ci'][1]:.3f}] "
                  f"sMDE={e['sampling_mde']:.3f} ncl={e['n_clusters']}  "
                  f"noninf={'YES' if e['noninferior'] else 'NO'}   [{e['two_sided_verdict']}]")
        for tag, e in res["per_wave"].items():
            print(f"   wave {tag:3s}         win={e['win']:.3f} CI[{e['ci'][0]:.3f},{e['ci'][1]:.3f}] "
                  f"ncl={e['n_clusters']}")
        print(f"   -> NONINFERIORITY {'CERTIFIED' if cert else 'NOT certified'} (needs all {npool} poolers)")
        print(f"   -> TWO-SIDED TIE (B) {'CERTIFIED' if res['certified_tie_B'] else 'NOT certified'} "
              f"(the R2-style endpoint; read this field for TIE claims)\n")
    out["caveats"] = [
        "Survivorship: this pooled estimate is conditioned on the seeds whose card sets passed the fid/lex "
        "build gate (CV: 2 of 3; s2 failed at 7.4% drop). The direction of that selection's effect on "
        "utility is not established but plausibly favors nec.",
        "Selection: the s0 wave was observed before this expansion was designed. nec-tpar_t15 was selected "
        "on power (point ~= null); nec-staab was selected on favorable direction and is a LABELED SENSITIVITY "
        "ANALYSIS ONLY -- it never upgrades a headline, whatever its verdict.",
    ]
    for c in out["caveats"]:
        print(f"  CAVEAT: {c}")
    OUTFILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved -> {OUTFILE.relative_to(ROOT) if OUTFILE.is_relative_to(ROOT) else OUTFILE}")


if __name__ == "__main__":
    main()
