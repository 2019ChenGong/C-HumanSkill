"""CV (Cross Validated / statistics) drafting pilot — ALIGNED to the Enron protocol (no extra strictness).
Tests, on the SAME bar as Enron: (1) card useful? indiv-nocard; (2) person-specific? indiv-stranger;
(3) pooling preserves? shared_k-nocard; (4) anonymity? 2AFC re-id on the shared card. Reports the hard-question
(LLM-weak) split AND the unfiltered set as a bound. Judge = HAIKU (project judge, for cross-dataset comparability),
BIDIRECTIONAL-debiased (same haiku, both A/B orders averaged — an artifact fix, not a model swap).

Pipeline aligned: nuwa 2-call extract->assemble (deepseek t0.3, 600/1100) domain-tailored to statistical consulting;
pooling via cmd_gate.synth_shared; drafts deepseek; cluster CI over experts (people).

Run:  python scripts/cv_pilot.py <stats.7z>   [PILOT_DRYRUN=1 NEXP=26 K=6 NHELD=3]
Run from project root (clean cwd).
"""
import os
import re
import sys
import html
import json
from pathlib import Path
from collections import Counter, defaultdict
from xml.etree import ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from src.llm import chat  # noqa: E402
from src.attrib_metrics import cluster_mean_ci  # noqa: E402
import cmd_gate as CG  # noqa: E402  (synth_shared, group_random — canonical pooling)

try:                                    # Windows console/redirect defaults to GBK -> non-ASCII prints crash
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GEN = os.environ.get("GEN", "deepseek-chat")
JUDGE = os.environ.get("JUDGE", "openrouter/anthropic/claude-haiku-4.5")   # PROJECT judge (haiku); bidirectional below
NEXP = int(os.environ.get("NEXP", 26)); NHELD = int(os.environ.get("NHELD", 3)); NCARD = 12
K = int(os.environ.get("K", 6)); SEED = 0
QACC = int(os.environ.get("QACC", 8)); CHARS = int(os.environ.get("CHARS", 300)); MINEXP = 20
DRAFT_TOK = int(os.environ.get("DRAFT_TOK", 700))
REFCH = 600
TAG = re.compile(r"<[^>]+>")
CODE = re.compile(r"<code>|<pre>|CREATE\s|SELECT\s|WITH\s|INSERT\s|UPDATE\s|import\s|library\(|lm\(|glm\(", re.I)
TECH = re.compile(r"\b(how|why|which|when|calculat|test|model|regress|distribut|estimat|variance|interval|hypothesi|bayes|likelihood|sampl|correlat|significan|assumption|method|interpret|predict|classif|cluster|anova|mixed|effect)\b", re.I)
OPIN = re.compile(r"\b(should i|career|vs\.?|versus|better|worth|opinion|book|resource|software|which language|jobs?)\b", re.I)
WS = re.compile(r"\s+")


def plain(b):
    return WS.sub(" ", html.unescape(TAG.sub(" ", b or ""))).strip()


def cvec(t, n=4):
    t = WS.sub(" ", (t or "").lower())
    return Counter(t[i:i + n] for i in range(max(0, len(t) - n + 1)))


def cos(a, b):
    ks = set(a) & set(b)
    if not ks:
        return 0.0
    import math
    num = sum(a[w] * b[w] for w in ks)
    da = math.sqrt(sum(v * v for v in a.values())); db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


# ---- data ----
def _iterrows(posts):
    """Memory-safe iterparse: yield each <row>, then clear the root so processed elements don't
    accumulate. Plain `el.clear()` leaves the emptied element attached to root -> stdlib ET grows
    root's child list to millions of nodes -> GC thrash turns a ~30s parse into ~20min. Same rows,
    same output; just doesn't leak."""
    it = ET.iterparse(posts, events=("start", "end"))
    _, root = next(it)
    for ev, el in it:
        if ev == "end" and el.tag == "row":
            yield el
            root.clear()


def load(archive):
    posts = archive.parent / f"_{archive.stem}" / "Posts.xml"
    ans_by_user = Counter(); q_title = {}; q_body = {}; q_acc = {}; a_score = {}; a_body = {}
    for el in _iterrows(posts):
        pt = el.get("PostTypeId")
        if pt == "1":
            q_title[el.get("Id")] = el.get("Title", ""); q_body[el.get("Id")] = el.get("Body", ""); q_acc[el.get("Id")] = el.get("AcceptedAnswerId")
        elif pt == "2":
            a_score[el.get("Id")] = int(el.get("Score", 0)); a_body[el.get("Id")] = el.get("Body", "")
            u = el.get("OwnerUserId")
            if u:
                ans_by_user[u] += 1
    experts = {u for u, c in ans_by_user.items() if c >= MINEXP}
    gold_q = {qid for qid, acc in q_acc.items() if acc and a_score.get(acc, 0) >= QACC
              and TECH.search(q_title.get(qid, "")) and not OPIN.search(q_title.get(qid, ""))}
    by_user = defaultdict(list)
    for el in _iterrows(posts):
        if el.get("PostTypeId") == "2":
            u = el.get("OwnerUserId"); qid = el.get("ParentId"); b = el.get("Body", "")
            if u in experts and qid in gold_q and CODE.search(b) and len(plain(b)) >= CHARS:
                by_user[u].append((int(el.get("Id")), qid, int(el.get("Score", 0)), b))
    for u in by_user:
        by_user[u].sort()
    cohort = {u: v for u, v in by_user.items() if len(v) >= NCARD + NHELD}
    return q_title, q_body, q_acc, a_body, cohort


# ---- nuwa (stats-tailored, mirrors enron_nuwa 2-call) ----
def nuwa_extract(answers):
    body = "\n\n---\n\n".join(plain(a)[:1500] for a in answers)
    return chat([{"role": "system", "content": "You reverse-engineer a statistician / data analyst's COGNITIVE OPERATING SYSTEM for answering statistical questions, from their Q&A answers."},
                 {"role": "user", "content": f"Answers written by ONE person (identifiers masked):\n\n{body}\n\nIdentify the underlying DECISION FRAMEWORKS, mental models, statistical heuristics, characteristic moves, and failure modes these reveal -- when they choose a method, what assumptions/diagnostics they check, how they frame tradeoffs -- and tie each to the evidence. Output terse notes grouped: Frameworks / Heuristics / Characteristic moves / Failure modes. Derive only from the texts."}],
                model=GEN, temperature=0.3, max_tokens=600) or ""


def nuwa_assemble(notes):
    return chat([{"role": "system", "content": "You compile a REUSABLE statistical-consulting cognitive operating system a colleague could EXECUTE."},
                 {"role": "user", "content": f"Notes on how a statistician decides:\n\n{notes}\n\nCompile a COGNITIVE OPERATING SYSTEM card a colleague could EXECUTE -- second-person, situation-triggered procedures, NOT a biography. Sections: [Analysis Protocol] step-by-step approach to a new statistical question; [Decision Frameworks & Mental Models]; [Heuristics] each as 'When X -> do Y, watch for Z'; [Characteristic Moves]; [Failure Modes]. Write each as an EXECUTABLE instruction. Keep the statistical substance (methods, assumptions, diagnostics), but do NOT name any person, dataset, URL, or handle. ~800-1000 words."}],
                model=GEN, temperature=0.3, max_tokens=1100) or ""


# ---- draft + bidirectional haiku judge ----
def draft(card, q):
    prof = f"Statistical-consulting profile:\n{card}\n\n" if card else ""
    return chat([{"role": "system", "content": "You answer a statistics question competently and correctly."},
                 {"role": "user", "content": f"{prof}Question:\n{q}\n\nGive your answer: the recommended approach and the key reasoning. Be concise and correct."}],
                model=GEN, temperature=0.0, max_tokens=DRAFT_TOK) or ""


def _raw(q, first, second):
    out = (chat([{"role": "system", "content": "You compare two answers to the SAME statistics question and decide which is MORE competent and correct -- sounder method choice, correct assumptions, handles the subtlety. IGNORE writing style, length, tone."},
                 {"role": "user", "content": f"Question:\n{q}\n\nAnswer A:\n{first}\n\nAnswer B:\n{second}\n\nWhich is more competent? Answer ONLY 'A' or 'B' or 'TIE'."}],
                model=JUDGE, temperature=0.0, max_tokens=4) or "").strip().upper()
    return 1 if out.startswith("A") else (-1 if out.startswith("B") else 0)


def judge(q, A, B):
    """Bidirectional: +1 => A more competent (position-bias removed)."""
    r1 = _raw(q, A, B); r2 = _raw(q, B, A)
    aw = (r1 > 0) + (r2 < 0); bw = (r1 < 0) + (r2 > 0)
    return 1 if aw > bw else (-1 if bw > aw else 0)


def judge_correct(q, cand, ref):
    out = (chat([{"role": "system", "content": "You judge whether a candidate answer correctly and completely solves a statistics question. Use the REFERENCE expert answer to judge correctness. A candidate may use a DIFFERENT valid approach and still be fully correct."},
                 {"role": "user", "content": f"Question:\n{q}\n\nREFERENCE expert answer:\n{ref}\n\nCandidate:\n{cand}\n\nDoes the candidate fully and correctly solve it? Answer ONLY 'YES', 'PARTIAL', or 'NO'."}],
                model=JUDGE, temperature=0.0, max_tokens=4) or "").strip().upper()
    return 2 if out.startswith("YES") else (1 if out.startswith("PART") else 0)


def main():
    archive = Path(sys.argv[1])
    q_title, q_body, q_acc, a_body, cohort = load(archive)
    users = sorted(cohort, key=lambda u: -len(cohort[u]))[:NEXP]
    grp = CG.group_random(users, K, SEED)
    ngrp = len(set(grp.values()))
    print(f"=== CV pilot | cohort(>= {NCARD+NHELD} gold)={len(cohort)} using {len(users)} | K={K} groups={ngrp} judge={JUDGE} (bidir) ===", flush=True)
    n_units = sum(min(NHELD, len(cohort[u][NCARD:])) for u in users)
    if os.environ.get("PILOT_DRYRUN"):
        print(f"DRYRUN: nuwa {len(users)*2} + shared {ngrp} + drafts ~{n_units*4} deepseek; "
              f"judges ~{n_units*(3*2+1)} + 2AFC ~{len(users)*2} haiku", flush=True)
        return
    assert ngrp > 1, f"need >=2 groups for stranger/2AFC (got {ngrp}); lower K or raise NEXP"

    # cards
    indiv = {u: nuwa_assemble(nuwa_extract([b for (_, _, _, b) in cohort[u][:NCARD]])) for u in users}
    byc = defaultdict(list)
    for u in users:
        byc[grp[u]].append(u)
    shared = {g: CG.synth_shared([indiv[u] for u in mem]) for g, mem in byc.items()}
    vec = {u: cvec(indiv[u]) for u in users}
    stranger = {u: max((v for v in users if grp[v] != grp[u]), key=lambda v: cos(vec[u], vec[v])) for u in users}

    # held units + drafts + judges
    rows = []
    for u in users:
        for (aid, qid, sc, b) in cohort[u][NCARD:NCARD + NHELD]:
            acc = q_acc.get(qid)
            if not (acc and a_body.get(acc)):
                continue
            q = f"{q_title.get(qid,'')}\n{plain(q_body.get(qid,''))[:1200]}"
            astar = plain(a_body[acc])[:1400]
            d_no = plain(draft(None, q))[:1400]
            d_in = plain(draft(indiv[u], q))[:1400]
            d_st = plain(draft(indiv[stranger[u]], q))[:1400]
            d_sh = plain(draft(shared[grp[u]], q))[:1400]
            weak = judge_correct(q, d_no, astar) <= 1                # LLM-weak = nocard NO/PARTIAL vs A*
            rows.append({"u": u, "weak": weak,
                         "in_no": judge(q, d_in, d_no), "in_st": judge(q, d_in, d_st), "sh_no": judge(q, d_sh, d_no),
                         "len": {"no": len(d_no), "in": len(d_in), "st": len(d_st), "sh": len(d_sh)}})

    if not rows:
        print("no held units with accepted-answer bodies — nothing to report", flush=True); return

    def report(sub, lab):
        if not sub:
            print(f"  {lab}: (empty)", flush=True); return {}
        g = [r["u"] for r in sub]; out = {}
        for k, nm in [("in_no", "indiv-nocard (card useful?)"), ("in_st", "indiv-stranger (person-specific?)"), ("sh_no", "shared-nocard (pooling preserves?)")]:
            v = [r[k] for r in sub]; ci = cluster_mean_ci(v, g, seed=SEED)
            out[k] = {"mean": round(float(np.mean(v)), 3), "ci": [round(c, 3) for c in ci]}
            print(f"    {nm:38s} = {np.mean(v):+.3f} CI{[round(c,3) for c in ci]}", flush=True)
        return out
    print(f"\nunits={len(rows)}  (LLM-weak={sum(r['weak'] for r in rows)} / strong={sum(not r['weak'] for r in rows)})", flush=True)
    res = {}
    print("  --- ALL held (unfiltered bound) ---", flush=True); res["all"] = report(rows, "ALL")
    print("  --- LLM-WEAK (hard) subset ---", flush=True); res["weak"] = report([r for r in rows if r["weak"]], "WEAK")
    ml = {a: int(np.median([r["len"][a] for r in rows])) for a in ("no", "in", "st", "sh")}
    print(f"\n  draft len (chars) median: {ml}", flush=True)

    # ---- 2AFC anonymity on the shared card (member vs same-topic stranger, held writing sample) ----
    print("\n=== 2AFC re-id on shared card (chance 0.5) ===", flush=True)
    heldtxt = {u: plain(cohort[u][NCARD][3])[:REFCH] for u in users}       # a held answer as writing sample
    hv = {u: cvec(heldtxt[u]) for u in users}
    hits = 0; tot = 0
    for u in users:
        card = shared[grp[u]]
        strv = max((v for v in users if grp[v] != grp[u]), key=lambda v: cos(hv[u], hv[v]))   # same-topic non-member
        A, B = heldtxt[u], heldtxt[strv]
        def atk(first, second):
            out = (chat([{"role": "system", "content": "You are a forensic authorship analyst. You compare HOW people reason -- decision architecture, method choices, priorities -- NOT the topic."},
                         {"role": "user", "content": f"A SKILL CARD distilled from one or more people:\n{card}\n\nTwo writing samples from DIFFERENT people; EXACTLY ONE contributed to the card:\n\n[A] {first}\n\n[B] {second}\n\nWhich -- A or B -- contributed, judged ONLY by reasoning/decision style (ignore topic)? Answer ONLY 'A' or 'B'."}],
                        model=JUDGE, temperature=0.0, max_tokens=4) or "").strip().upper()
            return 1 if out.startswith("A") else (-1 if out.startswith("B") else 0)
        r1 = atk(A, B); r2 = atk(B, A)                                     # bidirectional; member is A in r1, B in r2
        mw = (r1 > 0) + (r2 < 0); sw = (r1 < 0) + (r2 > 0)
        hits += (mw > sw); tot += 1
    print(f"  shared-card 2AFC acc = {hits/tot:.3f}  (n={tot}; >0.5 = leaks, ~0.5 = anonymized)", flush=True)

    res["n"] = len(rows); res["twoafc_shared"] = hits / tot; res["draft_len"] = ml
    (ROOT / "results" / "se").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "se" / "cv_pilot.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> results/se/cv_pilot.json", flush=True)


if __name__ == "__main__":
    main()
