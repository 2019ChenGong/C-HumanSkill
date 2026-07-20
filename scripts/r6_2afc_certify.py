"""R6 single-pack 2AFC certifier: dual-pooler channel accuracies + paired non-degradation diff.

The A2/R4 template as a reusable script, for packs exported by neutral_2afc_export.py (meta.json with
{chan, neg, card_id, member, stranger, member_slot} + ans_*.json with {pid, choice, conf}).

Per channel (nneg rows only): acc + two cluster bootstraps -- (B) cluster = card_id, (C) cluster =
member. Verdicts (delta = 0.10, R1/cmd_multiseed_pool semantics):
  TEST chan (conspf)  CERTIFIED_ANON iff BOTH poolers have up95 < .5+delta AND ci-lo <= .5
  REF  chan (neutral) REFERENCE-ONLY (never labeled certified here -- R4 MAJOR-2 discipline)
  GATE chan (indiv)   positive control: (B) ci-lo > .5 must hold or the wave is void
Paired diff REF-TEST over the shared (card_id, member, stranger) tuples (build_pairs constructs
identical tuples across channels), equal-weighted per-cluster mean + cluster bootstrap; CI containing
0 = "sanitized card no more identifying than the blackbox card" (P3).

Run:  BATCHDIR=results/mad/2afc_v6qwen [TEST=conspf REF=neutral GATECH=indiv DELTA=0.10
      NBOOT=20000 OUTFILE=results/mad/r6_2afc_certify.json] python -P scripts/r6_2afc_certify.py
"""
import os
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

B = ROOT / os.environ["BATCHDIR"]
TEST = os.environ.get("TEST", "conspf")
REF = os.environ.get("REF", "neutral")
GATECH = os.environ.get("GATECH", "indiv")
DELTA = float(os.environ.get("DELTA", "0.10"))
NBOOT = int(os.environ.get("NBOOT", "20000"))
OUTFILE = ROOT / os.environ.get("OUTFILE", "results/mad/r6_2afc_certify.json")

meta = json.loads((B / "meta.json").read_text(encoding="utf-8"))
ans = {}
for f in sorted(B.glob("ans_*.json")):
    for r in json.loads(f.read_text(encoding="utf-8-sig")):
        c = str(r.get("choice", "")).strip().upper()[:1]
        if c in ("A", "B"):
            ans[r["pid"]] = c
miss = [p for p in meta if p not in ans]
assert not miss, f"{len(miss)} pairs unanswered (e.g. {miss[:5]}) -- coverage first, then certify"

rows = defaultdict(list)             # chan -> [{card_id, member, stranger, ok}]
for pid, mt in meta.items():
    if mt.get("neg", "nneg") != "nneg":
        continue
    rows[mt["chan"]].append({"card_id": mt["card_id"], "member": mt["member"],
                             "stranger": mt["stranger"], "ok": float(ans[pid] == mt["member_slot"])})


def boot(rs, keyfn):
    v = np.array([r["ok"] for r in rs])
    g = [keyfn(r) for r in rs]
    lo, hi = cluster_mean_ci(v, g, n_boot=NBOOT, seed=0)
    return {"acc": round(float(v.mean()), 3), "ci": [round(lo, 3), round(hi, 3)],
            "n": len(v), "n_clusters": len(set(g))}


out = {"batchdir": str(B.relative_to(ROOT)), "delta": DELTA, "channels": {}}
print(f"R6 2AFC certify  {B.name}  delta={DELTA}  (B)=card  (C)=member  nneg only\n")
for chan in sorted(rows):
    bB = boot(rows[chan], lambda r: r["card_id"])
    bC = boot(rows[chan], lambda r: r["member"])
    res = {"B_card": bB, "C_member": bC}
    if chan == GATECH:
        res["gate_pass"] = bool(bB["ci"][0] > 0.5)
        lab = f"GATE {'PASS (attacker reads the card)' if res['gate_pass'] else 'FAIL -- wave void, re-judge'}"
    elif chan == TEST:
        res["certified_anon"] = bool(bB["ci"][1] < 0.5 + DELTA and bB["ci"][0] <= 0.5
                                     and bC["ci"][1] < 0.5 + DELTA and bC["ci"][0] <= 0.5)
        lab = "CERTIFIED_ANON" if res["certified_anon"] else "NOT certified"
    else:
        lab = "REFERENCE-ONLY"    # R4 MAJOR-2: the neutral channel is a same-wave reference, never certified here
    out["channels"][chan] = res
    print(f"  {chan:8s} (B) {bB['acc']:.3f} [{bB['ci'][0]:.3f},{bB['ci'][1]:.3f}] ncl={bB['n_clusters']}   "
          f"(C) {bC['acc']:.3f} [{bC['ci'][0]:.3f},{bC['ci'][1]:.3f}] ncl={bC['n_clusters']}   {lab}")

if REF in rows and TEST in rows:
    ref = {(r["card_id"], r["member"], r["stranger"]): r["ok"] for r in rows[REF]}
    tst = {(r["card_id"], r["member"], r["stranger"]): r["ok"] for r in rows[TEST]}
    common = sorted(set(ref) & set(tst))
    assert len(common) == len(ref) == len(tst), \
        f"pairing broken: ref {len(ref)} / test {len(tst)} / common {len(common)}"
    d = np.array([ref[k] - tst[k] for k in common])
    groups = [k[0] for k in common]
    by = defaultdict(list)
    for k in common:
        by[k[0]].append(ref[k] - tst[k])
    m_eq = float(np.mean([np.mean(v) for v in by.values()]))      # equal-weighted per-cluster (R4 formula)
    lo, hi = cluster_mean_ci(d, groups, n_boot=NBOOT, seed=0)
    out["paired_ref_minus_test"] = {"mean_eqweight": round(m_eq, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                    "n_pairs": len(common), "n_clusters": len(by),
                                    "contains_zero": bool(lo <= 0 <= hi)}
    print(f"\n  paired {REF}-{TEST}: {m_eq:+.4f} CI[{lo:+.4f},{hi:+.4f}] n={len(common)} "
          f"ncl={len(by)}  -> {'CI contains 0 (no degradation detected)' if lo <= 0 <= hi else 'CI EXCLUDES 0 -- report by name'}")

OUTFILE.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nsaved -> {OUTFILE.relative_to(ROOT) if OUTFILE.is_relative_to(ROOT) else OUTFILE}")
