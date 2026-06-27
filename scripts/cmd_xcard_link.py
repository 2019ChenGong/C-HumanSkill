"""Experiment B — LLM cross-card LINKAGE attack (the differentiated attacker direction).

Question: can a strong LLM decide, from TWO pooled shared cards (different poolings), whether they SHARE a contributor,
judged by decision architecture? This is the linkage step cmd_composition.py's combinatorial intersection re-id ASSUMES
for free. Two-sided payoff: AUC>>0.5 = real multi-release attack; AUC≈0.5 = m-invariance margin > combinatorial worst case.

Cards come from cached random poolings across seeds s0/s1/s2 (a person lands in a different group per seed). A "card" =
(seed, cluster); its members = byc[cluster]; its text = cache[f"k{K}_s{seed}_{cid}"].
  POS  = two cross-seed cards sharing EXACTLY ONE member            (the attack)
  CTRL = two cross-seed cards sharing >=2 members                   (positive control: high overlap should be linkable)
  NEG  = two cross-seed cards sharing ZERO members, content-cosine-MATCHED to the POS pair (topic confound guard)

MODE=census ($0): pair census + overlap distribution + POS/NEG content-cosine matching + verbatim n-gram gate. NO spend.
MODE=run: score matched POS/CTRL/NEG with sonnet ("share a contributor, by decision architecture?"), AUC + bootstrap CI.
Review-mandated guards (pre-spend): run at k=4 (1/4 dilution, not k8's 1/8); NEG topic-matched to POS; n-gram gate;
positive control; pre-registered read. DATASET=mad K=4 SEEDS=0,1,2.
"""
import os
import re
import sys
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GROUP", "random")
import deid_enron as de  # noqa: E402
import cmd_gate as CG  # noqa: E402
from src.llm import chat  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DS = os.environ.get("DATASET", "mad")
RES = ROOT / "results" if DS == "enron" else ROOT / "results" / DS
RES.mkdir(parents=True, exist_ok=True)
K = int(os.environ.get("K", 4))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
MODE = os.environ.get("MODE", "census")
CALIPER = float(os.environ.get("CALIPER", 0.03))     # |content-cosine(NEG) - content-cosine(POS)| <= CALIPER
WORD = re.compile(r"\w+")


def words(t):
    return WORD.findall((t or "").lower())


def ngrams(t, n):
    w = words(t)
    return set(tuple(w[i:i + n]) for i in range(len(w) - n + 1)) if len(w) >= n else set()


def build_cards():
    """Return cards = list of dicts {id:(s,cid), seed, cid, members:set, text, cvec, g6, g8}."""
    _d, authors, _n, aggro, _r, _rt = CG.load()
    cache = json.loads(CG.SHAREDC.read_text(encoding="utf-8"))
    cards = []
    for s in SEEDS:
        grp, byc = CG.make_groups(aggro, authors, K, s)
        for cid, mem in byc.items():
            if len(mem) < K:
                continue
            ck = f"k{K}_s{s}_{cid}"
            txt = cache.get(ck)
            if not txt:
                continue
            cards.append({"id": f"s{s}:{cid}", "seed": s, "cid": cid, "members": set(mem), "text": txt,
                          "cvec": de._content_vec(txt), "g6": ngrams(txt, 6), "g8": ngrams(txt, 8)})
    return cards


def cross_pairs(cards):
    """All cross-seed card pairs with overlap count + content cosine + shared n-gram counts."""
    out = []
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            a, b = cards[i], cards[j]
            if a["seed"] == b["seed"]:
                continue                                   # same pooling can't share (disjoint partition)
            ov = len(a["members"] & b["members"])
            out.append({"i": i, "j": j, "overlap": ov,
                        "ccos": de._cosine(a["cvec"], b["cvec"]),
                        "n6": len(a["g6"] & b["g6"]), "n8": len(a["g8"] & b["g8"])})
    return out


def match_neg(pos, negpool, used):
    """For a POS pair, pick the nearest-content-cosine unused NEG (overlap 0) within CALIPER."""
    best, bd = None, 1e9
    for n in negpool:
        if n["_k"] in used:
            continue
        d = abs(n["ccos"] - pos["ccos"])
        if d < bd:
            bd, best = d, n
    if best is not None and bd <= CALIPER:
        used.add(best["_k"]); return best
    return None


SYS = ("You are a forensic analyst comparing the DECISION ARCHITECTURE of two skill cards — HOW each pooled group "
       "reasons, sequences steps, escalates, weighs constraints, prioritizes — NOT the topic/domain they work on. "
       "Shared topic is NOT evidence of a shared person.")
USR = ("Two skill cards, each distilled from a POOL of several different people:\n\nCARD 1:\n{a}\n\nCARD 2:\n{b}\n\n"
       "Judged ONLY by decision architecture (ignore shared topic/domain/tooling), do these two pools share AT LEAST "
       "ONE common contributor? Answer EXACTLY one line: YES or NO, a space, confidence 50-100 (50=pure guess). "
       "Example: `YES 65`. Output ONLY that line.")
NUM = re.compile(r"\d{2,3}")


def link_score(ta, tb, salt):
    """P(model says the two cards SHARE a contributor), A/B order randomized by salt."""
    if int(hashlib.sha1(("ord" + salt).encode()).hexdigest(), 16) % 2:
        ta, tb = tb, ta
    out = chat([{"role": "system", "content": SYS},
                {"role": "user", "content": USR.format(a=ta, b=tb)}],
               model=os.environ.get("ATTACK_MODEL", "anthropic/claude-sonnet-4.5"),
               temperature=0.0, max_tokens=12) or ""
    yes = bool(re.search(r"\bYES\b", out, re.I))
    nums = [float(x) for x in NUM.findall(out)]
    conf = next((n for n in nums if 50 <= n <= 100), 50.0)
    return (conf / 100.0 if yes else 1.0 - conf / 100.0), out


def _auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg)))


def _boot(pos, neg, nb=2000):
    rng = np.random.default_rng(0); v = []
    pos, neg = np.array(pos), np.array(neg)
    for _ in range(nb):
        pi = rng.integers(0, len(pos), len(pos)); ni = rng.integers(0, len(neg), len(neg))
        v.append(_auc(pos[pi], neg[ni]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def _boot_clustered(records, nb=3000):
    """cluster-bootstrap AUC over the POS shared-member (person). records = list of (cluster_key, y, score).
    Resample clusters with replacement; each POS row carries its matched NEG via the same cluster_key."""
    by = defaultdict(list)
    for ck, y, s in records:
        by[ck].append((y, s))
    keys = list(by); rng = np.random.default_rng(0); v = []
    point = _auc([s for _, y, s in records if y == 1], [s for _, y, s in records if y == 0])
    for _ in range(nb):
        ys, ss = [], []
        for k in rng.choice(keys, len(keys), replace=True):
            for y, s in by[k]:
                ys.append(y); ss.append(s)
        if len(set(ys)) == 2:
            from sklearn.metrics import roc_auc_score
            v.append(roc_auc_score(ys, ss))
    return point, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def run():
    """Score matched POS/NEG + graded positive controls (share-3 sibling, share-4 re-synth) with the sonnet linkage
    attacker. PERSISTS per-pair scores + the POS shared member so CIs can be person-clustered. Verbatim-free subset is
    the MATCHED rows where BOTH pos and neg share 0 6-grams (caliper pairing preserved)."""
    cen = json.loads((RES / f"_xcard_census_k{K}.json").read_text(encoding="utf-8"))
    cards = {c["id"]: c for c in build_cards()}
    _d, authors, _n, aggro, _r, _rt = CG.load()

    def shared_member(idA, idB):
        s = cards[idA]["members"] & cards[idB]["members"]
        return sorted(s)[0] if s else None

    # positive controls: share-3 (sibling, 1 member swapped) AND share-4 (SAME pool re-synthesized = sanity ceiling)
    N_CTRL = int(os.environ.get("N_CTRL", 15))
    base = [c for c in cards.values() if c["seed"] == SEEDS[0]][:N_CTRL]
    sib_jobs, re_jobs = [], []
    for c in base:
        mem = list(c["members"])
        pool_others = [a for a in authors if a not in c["members"]]
        new = sorted(pool_others, key=lambda a: hashlib.sha1(f"ctrl-{c['id']}-{a}".encode()).hexdigest())[0]
        sib_jobs.append((c["id"], [aggro[a] for a in mem[:-1] + [new]]))     # share 3 of 4
        re_jobs.append((c["id"], [aggro[a] for a in mem]))                    # share 4 of 4 (re-synth)
    print(f"synth {len(sib_jobs)} share-3 + {len(re_jobs)} share-4 positive-control cards (deepseek) ...", flush=True)
    sibs = de.pool(lambda j: CG.synth_shared(j[1]), sib_jobs)
    res4 = de.pool(lambda j: CG.synth_shared(j[1]), re_jobs)

    jobs = []   # (kind, salt, textA, textB, cluster_key, n6)
    for i, mp in enumerate(cen["matched"]):
        sm = shared_member(mp["pos"]["a"], mp["pos"]["b"])
        jobs.append(("pos", f"pos{i}", cards[mp["pos"]["a"]]["text"], cards[mp["pos"]["b"]]["text"], sm, mp["pos"]["n6"]))
        jobs.append(("neg", f"neg{i}", cards[mp["neg"]["a"]]["text"], cards[mp["neg"]["b"]]["text"], sm, mp["neg"]["n6"]))
    for (cid, _), sib in zip(sib_jobs, sibs):
        jobs.append(("ctrl3", f"c3-{cid}", cards[cid]["text"], sib, None, None))
    for (cid, _), r4 in zip(re_jobs, res4):
        jobs.append(("ctrl4", f"c4-{cid}", cards[cid]["text"], r4, None, None))

    print(f"scoring {len(jobs)} card pairs with {os.environ.get('ATTACK_MODEL','anthropic/claude-sonnet-4.5')} (cached) ...", flush=True)

    def do(job):
        kind, salt, ta, tb, ck, n6 = job
        p, raw = link_score(ta, tb, salt)
        return {"kind": kind, "score": p, "cluster": ck, "n6": n6, "raw": raw}

    res = de.pool(do, jobs)
    (RES / f"_xcard_scores_k{K}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    pos = [r["score"] for r in res if r["kind"] == "pos"]
    neg = [r["score"] for r in res if r["kind"] == "neg"]
    # person-clustered records: each matched POS/NEG shares the POS cluster key (shared member)
    clrec = [(r["cluster"], 1, r["score"]) for r in res if r["kind"] == "pos" and r["cluster"]] \
        + [(r["cluster"], 0, r["score"]) for r in res if r["kind"] == "neg" and r["cluster"]]
    # MATCHED verbatim-free: pair up pos[i]/neg[i] by order, keep rows where BOTH n6==0
    posr = [r for r in res if r["kind"] == "pos"]; negr = [r for r in res if r["kind"] == "neg"]
    vf = [(p["cluster"], p["score"], n["score"]) for p, n in zip(posr, negr) if (p["n6"] == 0 and n["n6"] == 0)]

    print(f"\n=== cross-card linkage RESULT  DS={DS} k{K} ===")
    pa, plo, phi = _boot_clustered(clrec)
    print(f"LLM linkage AUC (POS share-1 vs topic-matched NEG) = {pa:.3f}  person-clustered CI[{plo:.3f},{phi:.3f}]  (n={len(pos)} pairs)")
    print(f"  vs $0 rare-6gram trivial-matcher = 0.673 [.642,.701]  -> {'BEATS verbatim' if plo > 0.701 else 'WITHIN/BELOW verbatim baseline (no architecture signal beyond verbatim)' if phi < 0.701 else 'overlaps'}")
    if vf:
        vrec = [(c, 1, ps) for c, ps, ns in vf] + [(c, 0, ns) for c, ps, ns in vf]
        va, vlo, vhi = _boot_clustered(vrec)
        print(f"VERBATIM-FREE (MATCHED, both n6=0, n={len(vf)}): AUC = {va:.3f} clustered CI[{vlo:.3f},{vhi:.3f}]  "
              f"-> {'architecture links without verbatim' if vlo > 0.5 else 'NO architecture beyond verbatim (CI∋0.5)'}")
    for kind, lbl in (("ctrl3", "share-3"), ("ctrl4", "share-4 (re-synth same pool)")):
        cc = [r["score"] for r in res if r["kind"] == kind]
        if cc:
            print(f"POS-CONTROL {lbl}: {np.mean([c>0.5 for c in cc]):.2f} linked, mean P={np.mean(cc):.3f} (n={len(cc)})  "
                  f"-> {'OK' if np.mean(cc) > 0.65 else 'WEAK attacker'}")
    out = {"auc": pa, "ci_clustered": [plo, phi], "rare6_baseline": 0.673, "n_pos": len(pos),
           "auc_verbatim_free": (va if vf else None), "vf_ci": ([vlo, vhi] if vf else None), "n_vf": len(vf), "k": K}
    (RES / f"_xcard_result_k{K}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nsaved -> {(RES / f'_xcard_result_k{K}.json').relative_to(ROOT)} + per-pair scores")


def main():
    if MODE == "run":
        run(); return
    cards = build_cards()
    pairs = cross_pairs(cards)
    for p in pairs:
        p["_k"] = (p["i"], p["j"])
    dist = Counter(p["overlap"] for p in pairs)
    print(f"=== cross-card linkage census  DS={DS} k{K} seeds={SEEDS} ===")
    print(f"cards={len(cards)}  cross-seed pairs={len(pairs)}")
    print(f"overlap distribution: " + "  ".join(f"share-{o}:{dist[o]}" for o in sorted(dist)))

    POS = [p for p in pairs if p["overlap"] == 1]
    CTRL = [p for p in pairs if p["overlap"] >= 2]
    NEGpool = [p for p in pairs if p["overlap"] == 0]
    print(f"\nPOS (share=1): {len(POS)}   CTRL (share>=2): {len(CTRL)}   NEG pool (share=0): {len(NEGpool)}")

    # content-cosine matching: match each POS to a caliper-near NEG
    used = set()
    matched = []
    for p in sorted(POS, key=lambda x: x["ccos"]):
        n = match_neg(p, NEGpool, used)
        if n is not None:
            matched.append((p, n))
    pc = np.array([p["ccos"] for p, _ in matched]); nc = np.array([n["ccos"] for _, n in matched])
    print(f"\n--- GATE 1: topic-confound (POS vs content-matched NEG) ---")
    print(f"matched POS/NEG pairs: {len(matched)}  (caliper {CALIPER})")
    if len(matched):
        print(f"POS content-cosine: mean={pc.mean():.3f} [{np.percentile(pc,10):.3f},{np.percentile(pc,90):.3f}]")
        print(f"NEG content-cosine: mean={nc.mean():.3f} [{np.percentile(nc,10):.3f},{np.percentile(nc,90):.3f}]")
        print(f"  mean |Δcosine| = {np.abs(pc-nc).mean():.4f}  -> {'OK overlap (topic matched)' if abs(pc.mean()-nc.mean())<0.02 else 'WARN: POS/NEG cosine differ -> topic confound risk'}")

    # n-gram verbatim gate on the MATCHED sets
    print(f"\n--- GATE 2: verbatim n-gram leak (POS vs matched NEG) ---")
    if len(matched):
        p6 = np.mean([p["n6"] for p, _ in matched]); n6 = np.mean([n["n6"] for _, n in matched])
        p8 = np.mean([p["n8"] for p, _ in matched]); n8 = np.mean([n["n8"] for _, n in matched])
        print(f"shared word-6gram: POS {p6:.2f} vs NEG {n6:.2f} ;  word-8gram: POS {p8:.2f} vs NEG {n8:.2f}")
        print(f"  -> {'OK (POS not >> NEG, no verbatim shortcut)' if p6 <= n6 + 1.0 else 'WARN: POS shares more verbatim n-grams -> string-match shortcut, fix synth'}")

    # positive-control availability
    print(f"\n--- GATE 3: positive control (share>=2 pairs available?) ---")
    print(f"CTRL pairs (share>=2): {len(CTRL)}  -> {'OK, usable as positive control' if len(CTRL) >= 15 else 'TOO FEW; will need constructed high-overlap pairs'}")

    print(f"\n--- GATE 4: power (n) ---")
    print(f"usable matched POS/NEG = {len(matched)} pairs -> {'OK n for CI<0.5 test' if len(matched) >= 40 else 'LOW n'}")

    # persist the matched pairs + ctrl for the run step
    def serial(p):
        return {"a": cards[p["i"]]["id"], "b": cards[p["j"]]["id"], "overlap": p["overlap"],
                "ccos": round(p["ccos"], 4), "n6": p["n6"], "n8": p["n8"]}
    out = {"matched": [{"pos": serial(p), "neg": serial(n)} for p, n in matched],
           "ctrl": [serial(p) for p in CTRL], "k": K, "seeds": SEEDS}
    (RES / f"_xcard_census_k{K}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nsaved census -> {(RES / f'_xcard_census_k{K}.json').relative_to(ROOT)}")
    print("READ: all 4 gates must pass before MODE=run spends on sonnet.")


if __name__ == "__main__":
    main()
