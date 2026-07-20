"""#5 Multi-seed pooling to CERTIFY pooled-card anonymity on Enron / CV (the underpowered-null datasets).

WHY: single-seed shared-card 2AFC nulls sit AT chance but are UNDERPOWERED (few clusters -> wide CI -> can't
exclude a <=0.55 leak; see cmd_equiv_test / checklist #2). Power scales with the number of (near-)independent
grouping units, and re-partitioning the SAME people under fresh random seeds adds units. This pools 3 seeds and
runs the privacy-relevant one-sided non-inferiority test (up95 < 0.5+delta) to turn "failure-to-reject" into a
defensible "meaningful leak excluded".

THE STAT CAVEAT (pre-registered in the checklist): the SAME person recurs across seeds, so seed batches are NOT
independent -> you CANNOT treat 3 x (N/k) clusters as 3x independent. Two valid poolers, reported together so the
certification does NOT hinge on the clustering choice (they bracket the true SE):

  (A) per-seed card-level   : each seed alone, bootstrap over card_id (few clusters, wide) -> CONSISTENCY check.
  (B) pooled (seed,card_id) : unit = (seed,card_id) cluster; 3x more units than one seed. Respects WITHIN-seed
                              co-member/shared-card correlation; cross-seed person recurrence only dilutes by 1/k
                              (each cluster shares <=1 person with a cluster in another seed) -> MORE conservative
                              (fewer units), the PRIMARY certification.
  (C) pooled person-level   : unit = member (person), carrying ALL their cross-seed obs. Absorbs the reused-ref
                              person effect (ref[m] is byte-identical across seeds); ignores within-seed co-member
                              correlation -> LESS conservative (more units), the SECONDARY/tighter estimate.

If BOTH (B) and (C) give up95 < U for `shared`, the pooled certification is robust to the clustering choice.

Reuses the SAME cluster-bootstrap machinery + non-inferiority/MDE definitions as cmd_equiv_test.py (delta=0.05).
$0 for pooling (pure re-analysis of ans_*.json); only cost was building the s1/s2 shared cards.

Run:  python scripts/cmd_multiseed_pool.py            [DELTA=0.05 NBOOT=20000]
Out:  results/multiseed_pool_summary.json + a per-dataset table.
"""
import os
import re
import sys
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DELTA = float(os.environ.get("DELTA", 0.05))
NBOOT = int(os.environ.get("NBOOT", 20000))
L, U = 0.5 - DELTA, 0.5 + DELTA
Z_A = 1.6448536269514722   # z_{0.95} one-sided
Z_B = 0.8416212335729143   # z_{0.80} power

# (dataset, [(seed_label, batchdir), ...]) — k8, GROUP=random, sonnet-4.6 free-subagent 2AFC.
# MSMODE=base -> original base-CMD batteries (shared channel certified).
# MSMODE=neutral -> neutral-CMD batteries built for #64 (neutral channel certified; base shared carried alongside).
MSMODE = os.environ.get("MSMODE", "base")
if MSMODE == "neufix":
    # #77: fixed (anti-copy, degeneracy-cleared) neutral CMD cards, neutral channel ONLY (indiv/shared reused from
    # the base/neutral batteries — unaffected by the degeneracy fix). Isolated neufix_* dirs; canonical dirs untouched.
    DATASETS = {
        "20-MAD k8": [("s0", "results/mad/neufix_k8_s0"),
                      ("s1", "results/mad/neufix_k8_s1"),
                      ("s2", "results/mad/neufix_k8_s2")],
        "Enron k8": [("s0", "results/enron/neufix_k8_s0"),
                     ("s1", "results/enron/neufix_k8_s1"),
                     ("s2", "results/enron/neufix_k8_s2")],
        "CV k8":    [("s0", "results/se/neufix_k8_s0"),
                     ("s1", "results/se/neufix_k8_s1"),
                     ("s2", "results/se/neufix_k8_s2")],
    }
elif MSMODE == "a1":
    # A1: concat_neutral vs neutral(CMD, fixed) on the SAME fixed cards, k8, 3 seeds, neutral+concat interleaved
    # in one pack (paired). Both channels are test channels (certified together); no indiv positive control in-pack
    # (validated separately via the base battery). Same free-subagent sonnet-4.6 2AFC instrument as neufix.
    DATASETS = {
        "20-MAD k8": [("s0", "results/mad/a1_ncc_k8_s0"), ("s1", "results/mad/a1_ncc_k8_s1"),
                      ("s2", "results/mad/a1_ncc_k8_s2")],
        "Enron k8": [("s0", "results/enron/a1_ncc_k8_s0"), ("s1", "results/enron/a1_ncc_k8_s1"),
                     ("s2", "results/enron/a1_ncc_k8_s2")],
        "CV k8":    [("s0", "results/se/a1_ncc_k8_s0"), ("s1", "results/se/a1_ncc_k8_s1"),
                     ("s2", "results/se/a1_ncc_k8_s2")],
    }
elif MSMODE == "v6":
    # R1 (#125): v6 (minimal-edit sanitize) three-channel packs, canonical wave + 2 new seeds.
    # Test chan = the CONSPFC slot, relabeled 'v6' (review MINOR-5); 'neutral' = same-wave reference
    # ONLY, excluded from certification (MINOR-6); 'indiv' = positive-control gate.
    DATASETS = {
        "20-MAD k8": [("s0", "results/mad/2afc_v6min"), ("s1", "results/mad/2afc_v6min_s1"),
                      ("s2", "results/mad/2afc_v6min_s2")],
        "Enron k8": [("s1", "results/enron/2afc_v6min"), ("s0", "results/enron/2afc_v6min_s0"),
                     ("s2", "results/enron/2afc_v6min_s2")],
        "CV k8":    [("s0", "results/se/2afc_v6min"), ("s1", "results/se/2afc_v6min_s1"),
                     ("s2", "results/se/2afc_v6min_s2")],
    }
elif MSMODE == "neutral":
    DATASETS = {
        "Enron k8": [("s0", "results/enron/2afc_util"),          # s0 (verified: grp matches seed 0, not s1)
                     ("s1", "results/enron/2afc_neutral_s1"),
                     ("s2", "results/enron/2afc_neutral_s2")],
        "CV k8":    [("s0", "results/se/2afc_neutral"),          # s0
                     ("s1", "results/se/2afc_neutral_s1"),
                     ("s2", "results/se/2afc_neutral_s2")],
    }
else:
    DATASETS = {
        "Enron k8": [("s0", "results/enron/ksw_k8_s0"),
                     ("s1", "results/enron/2afc_free"),      # battery (extra de-id arms ignored; we read shared+indiv)
                     ("s2", "results/enron/ms_k8_s2")],
        "CV k8":    [("s0", "results/se/2afc_battery"),
                     ("s1", "results/se/ms_k8_s1"),
                     ("s2", "results/se/ms_k8_s2")],
    }
# test channels (certified) + indiv positive control (must leak). neutral mode adds the neutral channel.
CHANS = os.environ.get("MSCHANS", "neutral,concat" if MSMODE == "a1"
                       else "neutral" if MSMODE == "neufix"
                       else "conspf,neutral,indiv" if MSMODE == "v6"
                       else "neutral,shared,indiv" if MSMODE == "neutral" else "shared,indiv").split(",")
# v6 mode: relabel the conspf slot to 'v6' everywhere (MINOR-5); neutral = reference-only (MINOR-6);
# canonical-wave seed per dataset for the new-seeds-only secondary estimate (MAJOR-1).
RELABEL = {"conspf": "v6"} if MSMODE == "v6" else {}
REF_CHANS = {"neutral"} if MSMODE == "v6" else set()
CANON_SEED = {"20-MAD k8": "s0", "Enron k8": "s1", "CV k8": "s0"} if MSMODE == "v6" else {}
CHANS = [RELABEL.get(c, c) for c in CHANS]


def load_seed(seed, bdir):
    """-> rows [{seed, chan, card_id, member, picked_member}] for shared+indiv nneg pairs answered in this dir."""
    B = ROOT / bdir
    meta = json.loads((B / "meta.json").read_text(encoding="utf-8"))
    ans = {}
    for f in sorted(B.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", f.stem):        # stray-guard
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            c = str(r.get("choice", "")).strip().upper()
            m = re.search(r"[AB]", c)
            if m:
                ans[r["pid"]] = m.group(0)
    rows = []
    for pid, mt in meta.items():
        ch = RELABEL.get(mt["chan"], mt["chan"])
        if ch not in CHANS or mt.get("neg") != "nneg" or pid not in ans:
            continue
        rows.append({"seed": seed, "chan": ch, "card_id": mt["card_id"],
                     "member": mt["member"], "picked_member": int(ans[pid] == mt["member_slot"])})
    return rows, len(meta)


def boot_by(rows, keyfn, seed=0):
    """Cluster bootstrap: group rows by keyfn(row), resample the GROUPS. -> acc, ci95, ci90(lo,hi), up95, se, n_units."""
    by = {}
    for r in rows:
        by.setdefault(keyfn(r), []).append(r["picked_member"])
    clus = list(by.values())
    ncl = len(clus)
    acc = float(np.mean([v for c in clus for v in c]))
    rng = np.random.default_rng(seed)
    means = np.empty(NBOOT)
    for i in range(NBOOT):
        pick = rng.integers(0, ncl, ncl)
        means[i] = np.mean([v for j in pick for v in clus[j]])
    se = float(means.std(ddof=1))
    return dict(acc=round(acc, 4), n=len(rows), n_units=ncl, se=round(se, 4),
                ci95=[round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)],
                up95=round(float(np.percentile(means, 95)), 4))


def certify(s):
    """Non-inferiority + MDE on a bootstrap result dict."""
    up95, se = s["up95"], s["se"]
    noninf = up95 < U
    mde_thresh = round(0.5 + (Z_A + Z_B) * se, 4)
    underpowered = mde_thresh >= U
    leak = s["ci95"][0] > 0.5
    if leak:
        verdict = f"LEAK (95%CI[{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]>.5)"
    elif noninf:
        verdict = f"ANON ✓ (leak≥{U:.2f} excluded, up95={up95:.3f})"
    elif underpowered:
        verdict = f"UNDERPOWERED (MDE≥{mde_thresh:.3f})"
    else:
        verdict = f"leak-not-excluded (up95={up95:.3f}≥{U:.2f})"
    return {**s, "noninf": bool(noninf), "mde_thresh": mde_thresh, "underpowered": bool(underpowered),
            "leak": bool(leak), "verdict": verdict}


out = {"delta": DELTA, "margin": [L, U], "nboot": NBOOT, "mode": MSMODE, "chans": CHANS, "datasets": {}}
print(f"\n#5 MULTI-SEED POOLING  margin=[{L:.2f},{U:.2f}] (δ={DELTA})  NBOOT={NBOOT}")
print("  cert = one-sided non-inferiority (up95<U => leak≥U excluded). (B)=(seed,card) primary; (C)=person tighter.\n")

for dslabel, seeds in DATASETS.items():
    allrows = []
    seed_present = []
    print(f"===== {dslabel} =====")
    for seed, bdir in seeds:
        if not (ROOT / bdir / "meta.json").exists():
            print(f"  [{seed}] {bdir}  -- MISSING, skip"); continue
        rows, nmeta = load_seed(seed, bdir)
        if not rows:
            print(f"  [{seed}] {bdir}  -- 0 answered shared/indiv nneg pairs, skip"); continue
        allrows += rows
        seed_present.append(seed)
    dsout = {"seeds": seed_present, "per_seed": {}, "pooled": {}}
    for chan in CHANS:
        crows = [r for r in allrows if r["chan"] == chan]
        if not crows:
            continue
        # (A) per-seed card-level
        print(f"  -- {chan} --")
        ps = {}
        for seed in seed_present:
            sr = [r for r in crows if r["seed"] == seed]
            a = certify(boot_by(sr, lambda r: r["card_id"], seed=int(re.sub(r"\D", "", seed) or 0)))
            ps[seed] = a
            print(f"    [{seed}] card-lvl acc={a['acc']:.3f} units={a['n_units']:>2d} "
                  f"95%CI[{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}] up95={a['up95']:.3f}  {a['verdict']}")
        # (B) pooled (seed,card_id)
        b = certify(boot_by(crows, lambda r: (r["seed"], r["card_id"])))
        # (C) pooled person
        c = certify(boot_by(crows, lambda r: r["member"]))
        print(f"    (B) POOL (seed,card) acc={b['acc']:.3f} units={b['n_units']:>2d} "
              f"95%CI[{b['ci95'][0]:.3f},{b['ci95'][1]:.3f}] up95={b['up95']:.3f} SE={b['se']:.3f}  {b['verdict']}")
        print(f"    (C) POOL person      acc={c['acc']:.3f} units={c['n_units']:>2d} "
              f"95%CI[{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}] up95={c['up95']:.3f} SE={c['se']:.3f}  {c['verdict']}")
        dsout["per_seed"][chan] = ps
        dsout["pooled"][chan] = {"seed_card": b, "person": c}
        # MAJOR-1 (R1 review): secondary estimate over the NEW seeds only (the canonical wave was
        # observed before R1 was registered -> show how much the headline leans on it).
        canon = CANON_SEED.get(dslabel)
        if canon is not None:
            nrows = [r for r in crows if r["seed"] != canon]
            if nrows and len({r["seed"] for r in nrows}) >= 1 and len(nrows) < len(crows):
                nb = certify(boot_by(nrows, lambda r: (r["seed"], r["card_id"])))
                nc = certify(boot_by(nrows, lambda r: r["member"]))
                print(f"    (B') NEW-SEEDS-ONLY  acc={nb['acc']:.3f} units={nb['n_units']:>2d} "
                      f"95%CI[{nb['ci95'][0]:.3f},{nb['ci95'][1]:.3f}] up95={nb['up95']:.3f}  {nb['verdict']}")
                print(f"    (C') NEW-SEEDS-ONLY  acc={nc['acc']:.3f} units={nc['n_units']:>2d} "
                      f"95%CI[{nc['ci95'][0]:.3f},{nc['ci95'][1]:.3f}] up95={nc['up95']:.3f}  {nc['verdict']}")
                dsout.setdefault("pooled_newonly", {})[chan] = {"seed_card": nb, "person": nc}
    out["datasets"][dslabel] = dsout
    # headline per test channel: certified robust iff BOTH pooled methods ANON (and not leaking);
    # v6 mode: 'neutral' is reference-only (review MINOR-6) -- never certified here.
    for rchan in [c for c in CHANS if c in REF_CHANS]:
        if rchan in dsout["pooled"]:
            print(f"  -- {rchan}: REFERENCE-ONLY (not a certification claim; A1 owns the ne cert)")
    for tchan in [c for c in CHANS if c != "indiv" and c not in REF_CHANS]:
        sh = dsout["pooled"].get(tchan, {})
        if not sh:
            continue
        both = sh["seed_card"]["noninf"] and sh["person"]["noninf"] and not sh["seed_card"]["leak"] and not sh["person"]["leak"]
        print(f"  => {tchan} CERTIFIED (robust): {both}  "
              f"[(B) {sh['seed_card']['verdict']} | (C) {sh['person']['verdict']}]")
    print()

_outname = os.environ.get("OUT", "multiseed_pool_summary.json")
(ROOT / "results" / _outname).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"saved -> results/{_outname}")
print("\nNOTE: (B) and (C) BRACKET the true SE under the crossed person×card structure. shared is certified only if")
print("      BOTH exclude a ≥U leak. indiv (pos-control) should show LEAK. Per-seed rows show cross-seed consistency.")
