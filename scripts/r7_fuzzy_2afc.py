"""R7 (#131) E1 — fuzzy-lexical attacker (mechanical, deterministic, no LLM judges).

E1a (R7MODE=e1a): threshold-sweep census on canonical k8 cards. Per card line, max-cos
(text-embedding-3-small, sqlite-cached) against (i) MEMBER pool = cluster members' elements ∪ aggro
lines vs (ii) STRANGER null pool = same construction from k seeded non-members; both pools seeded-
subsampled to equal LINE count (review M3). Curve = fraction of lines with max-cos >= tau,
tau in {.70...95}; headline = member−stranger gap, ne vs v6. Descriptive (no dictionary).

E1b rides the canonical G3-wave 2AFC pair sets (results/{mad,se,enron}/2afc_v6min), split into
three isolated passes (review M6):
  R7MODE=tier   — gate A (full PROMPT byte-rebuild vs pack batches, subsumes six-field + card-content
                  gates, review M1) then scores ONLY the indiv channel at 3 aggregation tiers
                  (top5mean/max/mean) -> results/r7_tier_stats_{DS}.json. No ne/v6 numbers.
  R7MODE=freeze — pools the 3 per-DS tier stats, applies the registered global rule (first tier in
                  top5mean->max->mean SIG>.5 on indiv in ALL datasets; else max-pass earliest tier,
                  failing datasets = instrument-limited) -> results/r7_tier_freeze.json.
  R7MODE=score  — requires the freeze; re-gates, scores neutral(ne)+conspf(v6) at the frozen tier
                  only + paired ne−v6 (cluster=card_id) -> results/r7_e1b_{DS}.json.
R7MODE=cost   — line inventory + embedding-token quote, no API.

Attacker statistic per candidate: m_i = max-cos of card line i to the candidate's ref lines
(the SAME _mref/_sref texts the LLM attacker saw); S = tier-aggregate of m; pick member iff
S_m > S_s (tie = .5). Accuracy CI = card_id-clustered bootstrap (5000, seed 0).

Recipes (one dataset per process; GROUP=random explicit per review MINOR-4; CV is DATASET=cv):
  DATASET=mad   KCL=8 SEED=0 CHANS=indiv,neutral,conspf GROUP=random M_NNEG=2 M_RNEG=0 \
    NEUTRALC=cmd_shared_cards_mad__neutral_fixed.json CONSPFC=cmd_shared_cards_mad__v6min.json \
    PACKDIR=results/mad/2afc_v6min R7MODE=tier python -P scripts/r7_fuzzy_2afc.py
  DATASET=cv    ... NEUTRALC=cmd_shared_cards_cv__neutral_fixed.json CONSPFC=cmd_shared_cards_cv__v6min.json \
    PACKDIR=results/se/2afc_v6min
  DATASET=enron KCL=8 SEED=1 ... NEUTRALC=cmd_shared_cards__neutral_fixed.json CONSPFC=cmd_shared_cards__v6min.json \
    PACKDIR=results/enron/2afc_v6min
  E1a: same env minus PACKDIR, plus K=8 SEED={0,0,1} (elemk_build binding) R7MODE=e1a
"""
import os
import re
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

R7MODE = os.environ.get("R7MODE", "cost")
TAUS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
TIERS = ("top5mean", "max", "mean")
NBOOT = 5000
RES = ROOT / "results"

_BULLET = re.compile(r"^\s*(?:[-*•>–—]+|\d+[.)]\s|#+\s|\*\*)\s*")
_WORD = re.compile(r"\w+")


def split_lines(text):
    """The single R7 splitter (cards, elements, aggro, refs all use it — review MINOR): newline
    split -> strip markdown bullet/number/emphasis prefixes -> keep lines with >=4 words."""
    out = []
    for ln in (text or "").splitlines():
        s = _BULLET.sub("", ln.strip()).strip("*_` ").strip()
        if len(_WORD.findall(s)) >= 4:
            out.append(s)
    return out


def embed_map(lines):
    import cmd_consensus_pool as CP
    uniq = sorted(set(lines))
    vecs = CP.embed(uniq)
    return dict(zip(uniq, vecs))


def mat(lines, V):
    return np.stack([V[x] for x in lines]) if lines else None


def maxcos(CM, RM):
    return (CM @ RM.T).max(axis=1)


def tier_agg(m):
    return {"top5mean": float(np.mean(np.sort(m)[-5:])), "max": float(np.max(m)), "mean": float(np.mean(m))}


def clus_boot(recs, nboot=NBOOT, seed=0):
    """recs = [(cluster_key, value)]; mean + clustered percentile CI (score_2afc_summary-style)."""
    by = defaultdict(list)
    for c, w in recs:
        by[c].append(w)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    point = float(np.mean([w for _, w in recs]))
    vals = []
    for _ in range(nboot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        vals.append(float(np.mean([w for k in pick for w in by[keys[k]]])))
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------- E1b: pair rebuild + full-prompt gate A ----------------

def rebuild_gated():
    """Rebuild the wave's pairs and byte-verify EVERY prompt against the exported pack (review M1:
    prompt equality subsumes six-field meta equality AND card/ref content equality — if any env
    (SEED, NEUTRALC, CONSPFC, M_NNEG...) differs from what the wave saw, some prompt differs)."""
    import neutral_2afc_export as NE
    A = NE.A
    print(f"[cfg] DATASET={A.DS} KCL={A.KCL} SEED={A.SEED} GROUP={os.environ.get('GROUP')} "
          f"CHANS={A.CHANS} NEUTRALC={os.environ.get('NEUTRALC')} CONSPFC={os.environ.get('CONSPFC')}")
    pack = ROOT / os.environ["PACKDIR"]
    meta = json.loads((pack / "meta.json").read_text(encoding="utf-8"))
    prompts = {}
    for f in sorted(pack.glob("batch_*.json")):
        for t in json.loads(f.read_text(encoding="utf-8")):
            prompts[t["pid"]] = t["prompt"]
    pairs, _n = A.build_pairs()
    assert len(pairs) == len(meta) == len(prompts), f"count mismatch {len(pairs)}/{len(meta)}/{len(prompts)}"
    bad_meta, bad_prompt = [], []
    for i, p in enumerate(pairs):
        pid = f"P{i:04d}"
        swap = NE._swap(p["chan"], p["card_id"], p["member"], p["stranger"])
        got = {"chan": p["chan"], "neg": p["neg"], "card_id": p["card_id"], "member": p["member"],
               "stranger": p["stranger"], "member_slot": "B" if swap else "A"}
        if any(meta[pid][k] != got[k] for k in got):
            bad_meta.append(pid)
        a, b = (p["_sref"], p["_mref"]) if swap else (p["_mref"], p["_sref"])
        if A.USR[p["_kind"]].format(card=p["_card"], a=a, b=b) != prompts[pid]:
            bad_prompt.append(pid)
    assert not bad_meta and not bad_prompt, \
        f"GATE A FAIL: meta mismatch {len(bad_meta)} {bad_meta[:5]} / prompt mismatch {len(bad_prompt)} {bad_prompt[:5]}"
    print(f"[gate A] {len(pairs)} pairs: meta fields 100% + PROMPT byte-identical 100% vs {pack.name}")
    return A, pairs


def pair_units(pairs, chans):
    """(chan, card_id, member, stranger) -> (card_lines, mref_lines, sref_lines); plus line pool."""
    units, all_lines, dropped = [], [], 0
    for p in pairs:
        if p["chan"] not in chans:
            continue
        cl, ml, sl = split_lines(p["_card"]), split_lines(p["_mref"]), split_lines(p["_sref"])
        if not cl or not ml or not sl:
            dropped += 1
            continue
        units.append((p, cl, ml, sl))
        all_lines += cl + ml + sl
    if dropped:
        print(f"  WARN: {dropped} pairs dropped (empty line set after split)")
    return units, all_lines


def score_chan(units, V, tiers):
    """-> {tier: {chan: [(card_id, member, stranger, win)]}}"""
    out = {t: defaultdict(list) for t in tiers}
    for p, cl, ml, sl in units:
        CM = mat(cl, V)
        sm, ss = tier_agg(maxcos(CM, mat(ml, V))), tier_agg(maxcos(CM, mat(sl, V)))
        for t in tiers:
            win = 1.0 if sm[t] > ss[t] else (0.5 if sm[t] == ss[t] else 0.0)
            out[t][p["chan"]].append((p["card_id"], p["member"], p["stranger"], win))
    return out


def mode_tier():
    A, pairs = rebuild_gated()
    units, lines = pair_units(pairs, {"indiv"})
    print(f"[tier] indiv pairs={len(units)}  unique lines={len(set(lines))}")
    V = embed_map(lines)
    scored = score_chan(units, V, TIERS)
    out = {"ds": A.DS, "n": len(units), "tiers": {}}
    for t in TIERS:
        recs = [(c, w) for c, _m, _s, w in scored[t]["indiv"]]
        acc, lo, hi = clus_boot(recs)
        out["tiers"][t] = {"acc": round(acc, 4), "ci": [round(lo, 4), round(hi, 4)], "sig": bool(lo > 0.5)}
        print(f"  indiv {t:9s} acc={acc:.4f} [{lo:.4f},{hi:.4f}]  {'SIG>.5' if lo > 0.5 else 'ns'}")
    (RES / f"r7_tier_stats_{A.DS}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved -> results/r7_tier_stats_{A.DS}.json   (no ne/v6 numbers in this pass)")


def mode_freeze():
    stats = {ds: json.loads((RES / f"r7_tier_stats_{ds}.json").read_text(encoding="utf-8"))
             for ds in ("mad", "cv", "enron")}
    chosen, trace = None, []
    for t in TIERS:
        ok = [ds for ds in stats if stats[ds]["tiers"][t]["sig"]]
        trace.append({"tier": t, "sig_datasets": ok})
        if len(ok) == 3 and chosen is None:
            chosen = {"tier": t, "instrument_limited": []}
    if chosen is None:
        best = max(trace, key=lambda r: len(r["sig_datasets"]))
        chosen = {"tier": best["tier"],
                  "instrument_limited": [ds for ds in stats if ds not in best["sig_datasets"]]}
    sha = {ds: hashlib.sha256((RES / f"r7_tier_stats_{ds}.json").read_bytes()).hexdigest()[:16]
           for ds in stats}
    out = {"rule": "first tier in top5mean->max->mean with indiv SIG>.5 (clustered CI) in ALL 3 datasets; "
                   "else earliest max-pass tier, failing datasets instrument-limited",
           **chosen, "trace": trace, "stats_sha": sha,
           "indiv": {ds: stats[ds]["tiers"] for ds in stats}}
    (RES / "r7_tier_freeze.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("saved -> results/r7_tier_freeze.json")


def mode_score():
    fz = json.loads((RES / "r7_tier_freeze.json").read_text(encoding="utf-8"))
    tier = fz["tier"]
    A, pairs = rebuild_gated()
    cur = hashlib.sha256((RES / f"r7_tier_stats_{A.DS}.json").read_bytes()).hexdigest()[:16]
    assert fz["stats_sha"][A.DS] == cur, \
        f"freeze is STALE: r7_tier_stats_{A.DS}.json changed after freeze (rerun R7MODE=freeze)"
    if A.DS in fz["instrument_limited"]:
        print(f"[score] {A.DS} is INSTRUMENT-LIMITED at frozen tier {tier} -> no ne/v6 claims; E1a only.")
        return
    units, lines = pair_units(pairs, {"neutral", "conspf"})
    print(f"[score] frozen tier={tier}  ne+v6 pairs={len(units)}  unique lines={len(set(lines))}")
    V = embed_map(lines)
    scored = score_chan(units, V, (tier,))[tier]
    out = {"ds": A.DS, "tier": tier, "chan": {}}
    for chan, lab in (("neutral", "ne"), ("conspf", "v6")):
        recs = [(c, w) for c, _m, _s, w in scored[chan]]
        acc, lo, hi = clus_boot(recs)
        out["chan"][lab] = {"acc": round(acc, 4), "ci": [round(lo, 4), round(hi, 4)], "n": len(recs),
                            "read": "SIG-leak" if lo > 0.5 else ("SIG-below" if hi < 0.5 else "ni .5")}
        print(f"  {lab:3s} acc={acc:.4f} [{lo:.4f},{hi:.4f}]  {out['chan'][lab]['read']}")
    ne = {(c, m, s): w for c, m, s, w in scored["neutral"]}
    v6 = {(c, m, s): w for c, m, s, w in scored["conspf"]}
    common = sorted(set(ne) & set(v6))
    assert len(common) == len(ne) == len(v6), f"unit mismatch {len(common)}/{len(ne)}/{len(v6)}"
    drecs = [(c, ne[(c, m, s)] - v6[(c, m, s)]) for c, m, s in common]
    d, dlo, dhi = clus_boot(drecs)
    out["paired_ne_minus_v6"] = {"mean": round(d, 4), "ci": [round(dlo, 4), round(dhi, 4)],
                                 "n": len(drecs), "contains_zero": bool(dlo <= 0 <= dhi)}
    print(f"  paired ne−v6 = {d:+.4f} [{dlo:+.4f},{dhi:+.4f}]  {'∋0' if dlo <= 0 <= dhi else 'EXCLUDES 0'}")
    (RES / f"r7_e1b_{A.DS}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved -> results/r7_e1b_{A.DS}.json")


# ---------------- E1a: threshold-sweep census ----------------

def _h(s):
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def mode_e1a():
    import elemk_build as EB
    import cmd_gate as CG
    DS = EB.DS
    print(f"[cfg] DATASET={DS} K={EB.K} SEED={EB.SEED} GROUP={os.environ.get('GROUP')} "
          f"NEUTRALC={os.environ.get('NEUTRALC')} CONSPFC={os.environ.get('CONSPFC')}"
          f"   (e1a binds K/SEED via elemk_build; the E1b KCL env is NOT read here)")
    aggro, byc = EB.load_clusters()
    cache = json.loads(EB.ELEMS_P.read_text(encoding="utf-8"))
    ne_f = json.loads((CG.SE / os.environ["NEUTRALC"]).read_text(encoding="utf-8"))
    v6_f = json.loads((CG.SE / os.environ["CONSPFC"]).read_text(encoding="utf-8"))
    authors = sorted(aggro)

    def pool_lines(people):
        out = []
        for m in people:
            for e in cache.get(m, []):
                out += split_lines(e)
            out += split_lines(aggro[m])
        return out

    per_arm = {"ne": [], "v6": []}          # (max-cos member, max-cos stranger) per card line
    all_lines = []
    plan = []
    for cid, mem in sorted(byc.items()):
        ck = f"k{EB.K}_s{EB.SEED}_{cid}"
        strangers = sorted((a for a in authors if a not in set(mem)),
                           key=lambda a: _h(f"r7null|{DS}|k{EB.K}|s{EB.SEED}|{cid}|{a}"))[:len(mem)]
        mp, sp = pool_lines(mem), pool_lines(strangers)
        n = min(len(mp), len(sp))
        rng = np.random.default_rng(_h(f"r7sub|{DS}|{ck}") % 2**32)
        mp = [mp[i] for i in rng.choice(len(mp), n, replace=False)]
        sp = [sp[i] for i in rng.choice(len(sp), n, replace=False)]
        cards = {"ne": ne_f[ck], "v6": v6_f[ck]}
        clines = {a: split_lines(cards[a]) for a in cards}
        for a, cl in clines.items():
            assert cl, f"{ck}/{a}: empty card line set"
        plan.append((ck, clines, mp, sp))
        assert plan[-1][2] and plan[-1][3], f"{ck}: empty member/stranger pool ({len(mp)}/{len(sp)} lines)"
        all_lines += mp + sp + [x for cl in clines.values() for x in cl]
    print(f"[e1a] clusters={len(plan)}  unique lines={len(set(all_lines))}  (pools line-matched per cluster)")
    V = embed_map(all_lines)
    for ck, cards, mp, sp in plan:
        MP, SP = mat(mp, V), mat(sp, V)
        for arm, cl in cards.items():
            CM = mat(cl, V)
            mA, mB = maxcos(CM, MP), maxcos(CM, SP)
            per_arm[arm] += list(zip(mA.tolist(), mB.tolist()))
    out = {"ds": DS, "k": EB.K, "seed": EB.SEED, "taus": list(TAUS), "arms": {}}
    print(f"\n[e1a] {DS}  member-vs-stranger fuzzy match curve (fraction of card lines, n_lines per arm)")
    print(f"{'arm':4s} {'n':>5s} " + " ".join(f"   tau={t:.2f} (mem/str/gap)" for t in TAUS))
    for arm, rows in per_arm.items():
        A_ = np.array([r[0] for r in rows]); B_ = np.array([r[1] for r in rows])
        curve = []
        for t in TAUS:
            fa, fb = float((A_ >= t).mean()), float((B_ >= t).mean())
            curve.append({"tau": t, "member": round(fa, 4), "stranger": round(fb, 4), "gap": round(fa - fb, 4)})
        out["arms"][arm] = {"n_lines": len(rows), "curve": curve,
                            "mean_maxcos_member": round(float(A_.mean()), 4),
                            "mean_maxcos_stranger": round(float(B_.mean()), 4)}
        print(f"{arm:4s} {len(rows):>5d} " + " ".join(f" {c['member']:.3f}/{c['stranger']:.3f}/{c['gap']:+.3f}" for c in curve))
    (RES / f"r7_e1a_{DS}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved -> results/r7_e1a_{DS}.json")


def mode_cost():
    n_lines, n_tok = 0, 0
    if os.environ.get("PACKDIR"):
        A, pairs = rebuild_gated()
        _u, lines = pair_units(pairs, set(A.CHANS))
        uq = set(lines)
        n_lines += len(uq); n_tok += sum(len(_WORD.findall(x)) for x in uq)
    print(f"COST: ~{n_lines} unique lines, ~{int(n_tok * 1.35)} tok -> text-embedding-3-small "
          f"~${n_tok * 1.35 / 1e6 * 0.02:.4f} (first run; reruns cached $0). Judges: none (mechanical).")


if __name__ == "__main__":
    {"cost": mode_cost, "e1a": mode_e1a, "tier": mode_tier, "freeze": mode_freeze, "score": mode_score}[R7MODE]()
