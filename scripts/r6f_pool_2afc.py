"""R6f pooled two-wave 2AFC certification (ELEMK_DESIGN.md R6f wave-2 addendum).

Pools ALL judge waves run on the same exported pack (pre-committed: every wave counts, no wave
selection) and certifies the TEST channel with the same dual-cluster bootstrap as r6_2afc_certify:
  (B) clusters = card_id      (C) clusters = member
  cert = up95 < .5 + DELTA  AND  CI-lo <= .5      (delta=.10, R1 dictionary)
plus the pooled per-cluster equal-weight paired diff REF-TEST (R4 template).

Env: DIRS=comma list of pack dirs sharing one meta.json pid space (first dir's meta used, asserted
identical), TEST=conspf REF=neutral GATECH=indiv DELTA=0.10 NBOOT=20000
OUT=results/mad/r6f_pool_2afc.json

  DIRS=results/mad/2afc_v6qwenguided,results/mad/2afc_v6qwenguided_w2 python -P scripts/r6f_pool_2afc.py
"""
import os
import json
import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / d for d in os.environ["DIRS"].split(",")]
TEST = os.environ.get("TEST", "conspf")
REF = os.environ.get("REF", "neutral")
GATECH = os.environ.get("GATECH", "indiv")
DELTA = float(os.environ.get("DELTA", "0.10"))
NBOOT = int(os.environ.get("NBOOT", "20000"))
OUT = ROOT / os.environ.get("OUT", "results/mad/r6f_pool_2afc.json")
rng = np.random.default_rng(0)

meta = json.loads((DIRS[0] / "meta.json").read_text(encoding="utf-8"))
for d in DIRS[1:]:
    assert json.loads((d / "meta.json").read_text(encoding="utf-8")) == meta, f"meta mismatch: {d}"

# records: chan -> list of (card_id, member, correct01, wave_idx)
rec = {}
for wi, d in enumerate(DIRS):
    ans_files = sorted(glob.glob(str(d / "ans_*.json")))
    n = 0
    for ap in ans_files:
        for a in json.loads(Path(ap).read_text(encoding="utf-8")):
            m = meta[a["pid"]]
            rec.setdefault(m["chan"], []).append(
                (m["card_id"], m["member"], int(a["choice"] == m["member_slot"]), wi))
            n += 1
    print(f"[pool] wave{wi + 1} {d.name}: {n} answers")


def boot_acc(rows, key_idx):
    keys = sorted({r[key_idx] for r in rows})
    by = {k: [r[2] for r in rows if r[key_idx] == k] for k in keys}
    accs = []
    for _ in range(NBOOT):
        pick = rng.choice(len(keys), len(keys), replace=True)
        vals = [v for i in pick for v in by[keys[i]]]
        accs.append(np.mean(vals))
    return float(np.mean([r[2] for r in rows])), float(np.percentile(accs, 2.5)), \
        float(np.percentile(accs, 97.5)), len(keys)


out = {"dirs": [d.name for d in DIRS], "delta": DELTA, "nboot": NBOOT, "channels": {}}
for ch in (GATECH, REF, TEST):
    rows = rec[ch]
    accB, loB, hiB, nB = boot_acc(rows, 0)
    accC, loC, hiC, nC = boot_acc(rows, 1)
    out["channels"][ch] = {"n": len(rows),
                           "B": {"acc": round(accB, 4), "lo": round(loB, 4), "hi": round(hiB, 4), "ncl": nB},
                           "C": {"acc": round(accC, 4), "lo": round(loC, 4), "hi": round(hiC, 4), "ncl": nC}}
    tag = ""
    if ch == TEST:
        certB = hiB < 0.5 + DELTA and loB <= 0.5
        certC = hiC < 0.5 + DELTA and loC <= 0.5
        out["certified_B"], out["certified_C"] = certB, certC
        out["certified_dual"] = certB and certC
        tag = "  CERTIFIED (dual)" if (certB and certC) else "  NOT certified"
    if ch == GATECH:
        out["gate_pass"] = loB > 0.5
        tag = "  GATE " + ("PASS" if loB > 0.5 else "FAIL")
    print(f"  {ch:8s} (B) {accB:.3f} [{loB:.3f},{hiB:.3f}] ncl={nB}   "
          f"(C) {accC:.3f} [{loC:.3f},{hiC:.3f}] ncl={nC}{tag}")

# pooled per-card-cluster equal-weight paired diff REF-TEST (R4 template)
cards = sorted({r[0] for r in rec[TEST]})
dref = {c: [r[2] for r in rec[REF] if r[0] == c] for c in cards}
dtst = {c: [r[2] for r in rec[TEST] if r[0] == c] for c in cards}
diffs = np.array([np.mean(dref[c]) - np.mean(dtst[c]) for c in cards])
bs = [np.mean(diffs[rng.choice(len(cards), len(cards), replace=True)]) for _ in range(NBOOT)]
plo, phi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
out["paired_ref_minus_test"] = {"mean": round(float(diffs.mean()), 4),
                                "lo": round(plo, 4), "hi": round(phi, 4), "ncl": len(cards)}
print(f"  paired {REF}-{TEST}: {diffs.mean():+.4f} CI[{plo:+.4f},{phi:+.4f}] ncl={len(cards)}"
      f"  -> {'CI contains 0' if plo <= 0 <= phi else 'CI EXCLUDES 0'}")

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"saved -> {OUT.relative_to(ROOT)}")
