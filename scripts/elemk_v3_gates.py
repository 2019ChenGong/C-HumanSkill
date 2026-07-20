# -*- coding: utf-8 -*-
"""Pinned measurement code for the ELEMK V3 fix-arm's pre-registered gates (ELEMK_DESIGN.md appendix).

Reviewer finding #1: the V2b attribution numbers (.542/.583 punt rates, near-verbatim census) lived in
session scratchpad scripts; re-deriving them post-hoc for V3 would allow silent judgment-call drift.
This file freezes both instruments and SELF-VALIDATES against the already-scored V2 q2 pack before any
V3 number is read:

  MODE=punt    BATCHDIR=results/mad/fc_elemk_q2   -> per-arm punt rate + unit-conditional table
  MODE=punt    BATCHDIR=results/mad/fc_elemk_q2v3 -> same, plus preamble-parrot check (reviewer #7)
  MODE=census                                     -> single-member near-verbatim %, q2 vs q2v3 SAME-RUN

Gate validity: MODE=punt on the V2 q2 pack must reproduce ne .542 / nec .583 (V2b baseline) or the
REQ regex is void. MODE=census computes both card sets in one invocation so the +3pp criterion compares
like with like. Order semantics (mad_fc_export.py:305): order 0 -> A=x(ne); order 1 -> A=y(nec).
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
MODE = os.environ.get("MODE", "punt")
# V4-X generalization: DATASET selects the prompt labels (each dataset's FC user prompt names the two
# candidates differently), the carrier file, and the census card files. MAD behavior is byte-identical.
DS = os.environ.get("DATASET", "mad")
_DATA = {"mad": os.path.join("data", "20mad"), "enron": os.path.join("data", "enron"),
         "cv": os.path.join("data", "se")}[DS]
_CARDBASE = {"mad": "cmd_shared_cards_mad", "enron": "cmd_shared_cards", "cv": "cmd_shared_cards_cv"}[DS]
_LAB = {"mad": "Assessment", "enron": "Reply", "cv": "Answer"}[DS]

# FROZEN (V2b, draft_engagement.py): produced ne .542 / nec .583 on fc_elemk_q2 and .542/.629 on q3.
REQ = re.compile(r"\b(request|ask|clarif|more information|more info|additional information"
                 r"|need.{0,20}(steps|details|info))\b", re.I)
# Reviewer #7: fragments of V3_PREAMBLE / rendering language that would be a diligence-tell if parroted.
PARROT = re.compile(r"background reference|engage the specific case|shared practice"
                    r"|apply a rule only|last[- ]resort conditional", re.I)


def carrier_shingles(n=6):
    """V4: 6-word shingles of the carrier text — a draft containing one is parroting the carrier verbatim."""
    p = os.path.join(_DATA, f"elemk_carrier_{DS}.txt")
    if not os.path.exists(p):
        return set()
    w = re.findall(r"[a-z']+", open(p, encoding="utf-8").read().lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def load_pack(d):
    meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
    ans = {}
    for f in glob.glob(os.path.join(d, "ans_*.json")):
        if not re.fullmatch(r"ans_\d+", os.path.splitext(os.path.basename(f))[0]):
            continue
        for r in json.load(open(f, encoding="utf-8-sig")):
            ans[r["pid"]] = r
    prompts = {}
    for f in glob.glob(os.path.join(d, "batch_*.json")):
        for it in json.load(open(f, encoding="utf-8")):
            prompts[it["pid"]] = it["prompt"]
    return meta, ans, prompts


def mode_punt():
    d = os.environ.get("BATCHDIR", "results/mad/fc_elemk_q2")
    meta, ans, prompts = load_pack(d)
    units, unit_win = {}, defaultdict(set)
    for pid, m in meta.items():
        if m.get("kind") != "contrast":
            continue
        if pid in ans:
            ch = ans[pid]["choice"].strip().upper()[:1]
            win = (m["x"] if ch == "A" else m["y"]) if m["order"] == 0 else (m["y"] if ch == "A" else m["x"])
            unit_win[m["unit"]].add(win)
        if m["order"] == 0 and pid in prompts:
            p = prompts[pid]
            a = p.split(f"{_LAB} A:\n", 1)[1].split(f"\n\n{_LAB} B:\n", 1)[0]
            b = p.split(f"\n\n{_LAB} B:\n", 1)[1].split(f"\n\nWhich {_LAB.lower()}", 1)[0]
            units[m["unit"]] = (a, b)      # order 0: A = x = ne, B = y = nec
    n = len(units)
    rq_ne = sum(bool(REQ.search(a)) for a, _ in units.values()) / n
    rq_nec = sum(bool(REQ.search(b)) for _, b in units.values()) / n
    print(f"[punt] {d}  units={n}")
    print(f"  punt rate: ne {rq_ne:.3f}   nec {rq_nec:.3f}")
    par = [u for u, (_, b) in units.items() if PARROT.search(b)]
    print(f"  preamble-parrot in nec drafts: {len(par)}/{n}" + (f"  e.g. units {par[:5]}" if par else ""))
    sh = carrier_shingles()
    if sh:
        def hits(t):
            w = re.findall(r"[a-z']+", t.lower())
            return any(" ".join(w[i:i + 6]) in sh for i in range(len(w) - 5))
        cp = [u for u, (_, b) in units.items() if hits(b)]
        print(f"  carrier 6-word-shingle parrot in nec drafts: {len(cp)}/{n}" + (f"  e.g. {cp[:5]}" if cp else ""))
    if unit_win:
        def share(sel):
            ss = [u for u in units if sel(units[u])]
            if not ss:
                return None, 0
            v = sum(1.0 if unit_win[u] == {"ne"} else 0.0 if unit_win[u] == {"nec"} else 0.5
                    for u in ss) / len(ss)
            return v, len(ss)
        for name, sel in [("nec punt only", lambda ab: REQ.search(ab[1]) and not REQ.search(ab[0])),
                          ("ne punt only", lambda ab: REQ.search(ab[0]) and not REQ.search(ab[1])),
                          ("both", lambda ab: REQ.search(ab[0]) and REQ.search(ab[1])),
                          ("neither", lambda ab: not REQ.search(ab[0]) and not REQ.search(ab[1]))]:
            v, k = share(sel)
            if v is not None:
                print(f"  {name:14s} n={k:3d}  ne-win-share={v:.3f}")


def _card_lines(card):
    out = []
    for ln in card.splitlines():
        s = re.sub(r"^\s*[-*•\d.)#]+\s*", "", ln).strip()
        if len(re.findall(r"\w+", s)) >= 5 and not s.isupper():
            out.append(s)
    return out


def _load_support(EB):
    """(ck, ei) -> support count, replaying stage_fuse's matched-set logic on the shipped adjudication."""
    recs = json.loads(EB.PAIRS_P.read_text(encoding="utf-8"))
    ans = {}
    for f in sorted(EB.ADJ.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", f.stem):
            continue
        for r in json.loads(f.read_text(encoding="utf-8-sig")):
            ans[r["pid"]] = bool(r.get("same"))
    matched = {}
    for r in recs:
        if r["cls"] == "yes" or (r["cls"] == "gray" and ans.get(r.get("pid", ""), False)):
            matched.setdefault((r["ck"], r["ei"]), set()).add(r["mo"])
    return matched


def _shingles(text, n=6):
    w = re.findall(r"[a-z']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def mode_anatomy():
    """V5 phase-0 ($0): where does each black-box ne card line come from?
    Per line: best-matching member element (cos), that element's support, #members at >=0.55,
    plus a LEXICAL 6-word-shingle check vs member elements AND member aggro cards.
    This is the black-box 'recipe' the V5 mimicry pipeline is parameterized from."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import numpy as np
    import elemk_build as EB
    _CANON = {"mad": (8, 0), "enron": (8, 1), "cv": (8, 0)}
    assert (EB.K, EB.SEED) == _CANON[DS], f"anatomy for {DS} needs K/SEED={_CANON[DS]}"
    aggro, byc = EB.load_clusters()
    cache = json.loads(EB.ELEMS_P.read_text(encoding="utf-8"))
    clus = EB.cluster_elements(byc, cache)
    matched = _load_support(EB)
    nef = os.path.join(_DATA, f"{_CARDBASE}__neutral_fixed.json")
    cards = json.loads(open(nef, encoding="utf-8").read())

    rows = []          # (top_cos, support_of_top, n_members_at_.55, lex_hit_elem, lex_hit_aggro)
    words, mem_cov = [], []
    for ck, (texts, owners) in clus.items():
        if not texts or ck not in cards:
            continue
        V = np.stack(EB.embed(texts))
        sup = [1 + len(matched.get((ck, i), ())) for i in range(len(texts))]
        mem = sorted(set(owners))
        oarr = np.array([mem.index(o) for o in owners])
        esh = [ _shingles(t) for t in texts ]
        ash = { m: _shingles(aggro[m]) for m in mem }
        L = _card_lines(cards[ck])
        words.append(len(re.findall(r"\w+", cards[ck])))
        S = np.stack(EB.embed(L)) @ V.T
        contrib = set()
        for i, s in enumerate(L):
            j = int(np.argmax(S[i]))
            per = [S[i, oarr == m].max() if (oarr == m).any() else 0.0 for m in range(len(mem))]
            n55 = sum(1 for v in per if v >= 0.55)
            lsh = _shingles(s)
            lex_e = any(lsh & es for es in esh)
            lex_a = any(lsh & ash[m] for m in mem)
            rows.append((float(S[i, j]), sup[j], n55, lex_e, lex_a))
            if S[i, j] >= 0.65:
                contrib.add(owners[j])
        mem_cov.append(len(contrib))

    n = len(rows)
    def pct(sel):
        return 100.0 * sum(1 for r in rows if sel(r)) / n
    print(f"[anatomy] {DS}  cards={len(words)}  lines={n}  words/card mean={np.mean(words):.0f} "
          f"(min {min(words)} max {max(words)})")
    print(f"  attribution by top element cos:  <.55 (unattributed/generic): {pct(lambda r: r[0] < .55):.1f}%   "
          f"[.55,.65): {pct(lambda r: .55 <= r[0] < .65):.1f}%   [.65,.8): {pct(lambda r: .65 <= r[0] < .8):.1f}%   "
          f">=.8: {pct(lambda r: r[0] >= .8):.1f}%")
    for lo, tag in [(0.65, ">=.65"), (0.8, ">=.8")]:
        sub = [r for r in rows if r[0] >= lo]
        if not sub:
            continue
        ns = len(sub)
        s1 = sum(1 for r in sub if r[1] == 1) / ns
        s2 = sum(1 for r in sub if r[1] == 2) / ns
        s3 = sum(1 for r in sub if r[1] >= 3) / ns
        m1 = sum(1 for r in sub if r[2] == 1) / ns
        print(f"  attributed {tag} ({ns} lines): support of best element  sup1 {s1:.1%} / sup2 {s2:.1%} / "
              f"sup>=3 {s3:.1%}   single-member@.55: {m1:.1%}")
    print(f"  LEXICAL 6-shingle overlap: vs member elements {pct(lambda r: r[3]):.1f}%   "
          f"vs member aggro cards {pct(lambda r: r[4]):.1f}%")
    print(f"  distinct members contributing lines@>=.65 per card: mean {np.mean(mem_cov):.1f} / k=8")


def mode_lex():
    """V5 lexical certificate, measured independently of the build (CARDS env = comma list of
    labels; default compares ne vs v5san). Per card file: % lines sharing a 6-word shingle with
    (a) cluster member elements, (b) cluster member aggro cards."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import elemk_build as EB
    _CANON = {"mad": (8, 0), "enron": (8, 1), "cv": (8, 0)}
    # R1 (#125): multi-seed certificates allowed -- K stays canonical, SEED may be 0/1/2 (the
    # shingle pool is rebuilt per cluster from THIS partition's members, so nothing is seed-stale).
    # R13 (#139): MAD k-gradient certificates allowed -- the same argument covers K (clusters,
    # elements, and shingle pools all derive from THIS (K,SEED) partition's files).
    _KGRID = (2, 4, 6, 8, 10, 12) if DS == "mad" else (_CANON[DS][0],)
    assert EB.K in _KGRID and EB.SEED in (0, 1, 2), \
        f"lex for {DS} needs K in {_KGRID} and SEED in 0/1/2"
    aggro, byc = EB.load_clusters()
    cache = json.loads(EB.ELEMS_P.read_text(encoding="utf-8"))
    clus = EB.cluster_elements(byc, cache)
    labels = os.environ.get("CARDS", "neutral_fixed,v5san").split(",")
    files = {lab: os.path.join(_DATA, f"{_CARDBASE}__{lab}.json") for lab in labels}
    cards = {lab: json.loads(open(p, encoding="utf-8").read()) for lab, p in files.items()}
    res = {lab: [0, 0, 0] for lab in files}          # lines, hit-elements, hit-aggro
    for ck, (texts, owners) in clus.items():
        mem = sorted(set(owners))
        esh = [_shingles(t) for t in texts]
        ash = {m: _shingles(aggro[m]) for m in mem}
        for lab in files:
            for s in _card_lines(cards[lab].get(ck, "")):
                lsh = _shingles(s)
                res[lab][0] += 1
                res[lab][1] += any(lsh & e for e in esh)
                res[lab][2] += any(lsh & ash[m] for m in mem)
    print(f"[lex] {DS}  6-word-shingle overlap of card lines with cluster member text:")
    for lab in files:
        n, he, ha = res[lab]
        print(f"  {lab:15s} lines={n:4d}   vs elements: {he/n:6.1%}   vs aggro cards: {ha/n:6.1%}")


def mode_census():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import numpy as np
    import elemk_build as EB
    # Reviewer MAJOR-2 (V4-X): a forgotten SEED env makes every card lookup miss (seed is baked into the key)
    # and the census would crash confusingly or, patched around, report a spurious 0%. Assert the canon.
    _CANON = {"mad": (8, 0), "enron": (8, 1), "cv": (8, 0)}
    assert (EB.K, EB.SEED) == _CANON[DS], (f"census for DATASET={DS} needs K/SEED={_CANON[DS]}, got "
                                           f"({EB.K},{EB.SEED}) -- set the SEED env (Enron canon = s1)")
    _aggro, byc = EB.load_clusters()
    cache = json.loads(EB.ELEMS_P.read_text(encoding="utf-8"))
    clus = EB.cluster_elements(byc, cache)
    if DS == "mad":
        files = {"q2": "data/20mad/cmd_shared_cards_mad__elemk_q2.json",
                 "q2v3": "data/20mad/cmd_shared_cards_mad__elemk_q2v3.json",
                 "q2v4": "data/20mad/cmd_shared_cards_mad__elemk_q2v4.json",
                 "ne": os.path.join(_DATA, f"{_CARDBASE}__neutral_fixed.json"),
                 "v5san": os.path.join(_DATA, f"{_CARDBASE}__v5san.json"),
                 "v6min": os.path.join(_DATA, f"{_CARDBASE}__v6min.json")}
    else:
        # V4-X G1' (ELEMK_DESIGN.md): the black-box ne card measured SAME-RUN is the census upper bound.
        files = {"ne": os.path.join(_DATA, f"{_CARDBASE}__neutral_fixed.json"),
                 "q2v4": os.path.join(_DATA, f"{_CARDBASE}__elemk_q2v4.json"),
                 "v5san": os.path.join(_DATA, f"{_CARDBASE}__v5san.json"),
                 "v6min": os.path.join(_DATA, f"{_CARDBASE}__v6min.json")}
    files = {k: p for k, p in files.items() if os.path.exists(p)}
    cards = {k: json.loads(open(p, encoding="utf-8").read()) for k, p in files.items()}

    def lines(card):
        out = []
        for ln in card.splitlines():
            s = re.sub(r"^\s*[-*•\d.)#]+\s*", "", ln).strip()
            if len(re.findall(r"\w+", s)) >= 5 and not s.isupper():
                out.append(s)
        return out

    res = {k: [0, 0, 0] for k in files}
    for ck, (texts, owners) in clus.items():
        V = np.stack(EB.embed(texts))
        mem = sorted(set(owners))
        oarr = np.array([mem.index(o) for o in owners])
        for k in files:
            L = lines(cards[k].get(ck, ""))
            if not L:
                continue
            S = np.stack(EB.embed(L)) @ V.T
            for i in range(len(L)):
                per = [S[i, oarr == m].max() if (oarr == m).any() else 0.0 for m in range(len(mem))]
                n55 = sum(1 for v in per if v >= 0.55)
                res[k][0] += 1
                res[k][1] += (n55 == 1 and max(per) >= 0.9)
                res[k][2] += (n55 == 1 and max(per) >= 0.8)
    print("[census] single-member near-verbatim line rate (same-run, q2 vs q2v3 comparable):")
    for k in files:
        n, v9, v8 = res[k]
        print(f"  {k:5s} lines={n:4d}  cos>=0.9: {v9/n:6.1%}   cos>=0.8: {v8/n:6.1%}")


if MODE == "punt":
    mode_punt()
elif MODE == "census":
    mode_census()
elif MODE == "anatomy":
    mode_anatomy()
elif MODE == "lex":
    mode_lex()
else:
    sys.exit(f"unknown MODE={MODE!r} (punt|census|anatomy|lex)")
