"""Export CV UTILITY pairwise-competence judge tasks for FREE sonnet subagents = cross-model check that the
haiku utility (indiv-nocard +0.564 @26 / +0.463 @77) is NOT a haiku-judge artifact (缺口2).

Reuses cv_pilot's load / nuwa cards / draft (drafts are CACHED deepseek t0 -> re-calling = cache hits, no spend).
Per held unit, three comparisons — in_no=(indiv,nocard), in_st=(indiv,stranger), sh_no=(shared,nocard) — each
judged BIDIRECTIONAL (role r1 = X-vs-Y, role r2 = Y-vs-X); the score script debiases exactly like cv_pilot.judge().
A subagent answers each directional task with 'A'/'B'/'TIE' (which answer is more competent). This mirrors
cv_pilot._raw byte-for-byte (same SYS + USR), only the JUDGE model changes deepseek/haiku -> free sonnet subagent.

Run:  NEXP=26 BATCHDIR=results/se/util_judge python scripts/cv_util_judge_export.py scratchpad/se/stats.7z
Then: N sonnet subagents each answer one batch -> cv_util_judge_score.py
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DATASET", "cv")
import cv_pilot as CVP  # noqa: E402
import cmd_gate as CG  # noqa: E402
import deid_enron as de  # noqa: E402  (de.pool = parallel draft pre-warm)
from src.llm import chat  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NBATCH = int(os.environ.get("NBATCH", 10))
OUT = ROOT / os.environ.get("BATCHDIR", "results/se/util_judge")
OUT.mkdir(parents=True, exist_ok=True)
NEUTRAL = os.environ.get("NEUTRAL", "") not in ("", "0")      # add the neutral-CMD utility arm
NEUTRAL_ONLY = os.environ.get("NEUTRAL_ONLY", "") not in ("", "0")   # emit ONLY ne_no/ne_in (in_no/sh_no already judged)
CONSENSUS = os.environ.get("CONSENSUS", "") not in ("", "0")  # add the consensus-aggregation pooling arm (co_no / co_ne)
CONSENSUS_ONLY = os.environ.get("CONSENSUS_ONLY", "") not in ("", "0")   # emit ONLY co_no/co_ne (fewest new drafts)
UNION = os.environ.get("UNION", "") not in ("", "0")         # add the ADDITIVE union-genericize arm (un_no / un_ne): keep full deduped union, genericize, NO deletion
UNION_ONLY = os.environ.get("UNION_ONLY", "") not in ("", "0")   # emit ONLY un_no/un_ne
CONS_TAU = float(os.environ.get("CONS_TAU", 0.55))
CONS_Q = int(os.environ.get("CONS_Q", 3))
CONS_PF = os.environ.get("CONS_PF", "1") not in ("", "0")   # post-filter assembled card (element-level <=1/k end-to-end)
CONS_HI = float(os.environ.get("CONS_HI", 0.75))
DEID = [a for a in os.environ.get("DEID", "").split(",") if a]  # e.g. staab,petre_k4,tpar_t15 — per-person de-id arms
DEID_ONLY = os.environ.get("DEID_ONLY", "") not in ("", "0")    # emit ONLY the ne_{arm} head-to-head comparisons
STEP2C = CG.SE / os.environ.get("STEP2C", "cv_cmd_step2.json")  # de-id cards (built by cv_step2_baselines.py)

# FROZEN neutral synth — byte-identical to mad_synth_utility.synth_neutral (the canonical utility-preserving CMD).
_SYS_KEEP = ("You distill ONE shared skill card capturing the full shared working knowledge and practices common to "
             "several colleagues, removing only what identifies any single one of them.")


def synth_neutral(member_cards):
    """Anti-copy neutral synth (byte-identical guard to cmd_fix_degenerate): a GENUINE synthesis that never copies a
    single member; retry with rising temperature until max member-cosine < 0.85 (else keep the least-copying draft)."""
    mvecs = [de._content_vec(m) for m in member_cards]
    best = None
    for t in range(4):
        body = "\n\n---\n\n".join(member_cards)
        msg = [{"role": "system", "content": _SYS_KEEP},
               {"role": "user", "content": f"Skill cards from several colleagues:\n\n{body}\n\nWrite ONE shared skill "
                "card that captures everything they have in COMMON — their shared knowledge, working approaches, and "
                "practices — while removing any phrasing or detail unique to any single person.\n\n"
                "CRITICAL — this must be a GENUINE SYNTHESIS, not a copy: do NOT reproduce, quote, or lightly paraphrase "
                "any single colleague's card. No sentence, list, or passage may be traceable to one person's card alone. "
                "Abstract and re-express the COMMON structure in neutral wording so that NO single contributor could be "
                "identified as its source. If one card is longer or more detailed than the others, do NOT let it "
                "dominate — include only what it SHARES with the rest.\n\n"
                "Preserve the concrete shared substance; do not compress into generic platitudes. Keep it comparable in "
                "length to ONE input card. It must read as if it could belong to any of them equally. Output ONLY the "
                "shared card."}]
        card = chat(msg, model="deepseek-chat", temperature=0.3 + 0.2 * t, max_tokens=1300) or ""
        mc = max((de._cosine(de._content_vec(card), mv) for mv in mvecs), default=0.0)
        if best is None or mc < best[1]:
            best = (card, mc)
        if mc < 0.85:
            break
    return best[0]


def synth_consensus(member_cards):
    """Consensus-Aggregation Pooling arm: decompose members into decision-elements, keep only elements shared by
    >= CONS_Q members (embedding cosine >= CONS_TAU), assemble a card from consensus-only elements. (see cmd_consensus_pool)"""
    from cmd_consensus_pool import elements as _elem, embed as _embed
    import numpy as np
    elems, owner = [], []
    for i, mc in enumerate(member_cards):
        for e in _elem(mc):
            elems.append(e); owner.append(i)
    if not elems:
        return ""
    V = _embed(elems); n = len(elems)

    def agree(i):
        return 1 + len({owner[j] for j in range(n) if owner[j] != owner[i] and float(V[i] @ V[j]) >= CONS_TAU})
    idxs = sorted([i for i in range(n) if agree(i) >= CONS_Q], key=lambda i: -agree(i))
    kept = []
    for i in idxs:
        if all(float(V[i] @ V[j]) < 0.80 for j in kept):
            kept.append(i)
    body = "\n".join(f"- {elems[i]}" for i in kept)
    card = chat([{"role": "system", "content": "You assemble a shared skill card for a team from a vetted list of "
                  "consensus points the members have in common."},
                 {"role": "user", "content": f"Consensus points shared by MULTIPLE members (member-specific detail "
                  f"already removed):\n\n{body}\n\nWrite ONE coherent shared skill card using ONLY these consensus "
                  "points; do not add new points or reintroduce member-specific detail. Output ONLY the card."}],
                model="deepseek-chat", temperature=0.3, max_tokens=1300) or ""
    if not CONS_PF:
        return card
    # POST-FILTER (deterministic, $0): drop assembled element-lines that are single-member-traceable
    # (support==1 at CONS_TAU AND top-member cos >= CONS_HI), so element-level <=1/k holds on the OUTPUT card.
    import re as _re
    lines = card.splitlines(); cand = []
    for li, ln in enumerate(lines):
        s = _re.sub(r"^\s*[-*•\d.)#]+\s*", "", ln).strip()
        if len(_re.findall(r"\w+", s)) >= 5 and not s.isupper():
            cand.append((li, s))
    if cand:
        EV = _embed([s for _, s in cand]); drop = set()
        for (li, _s), ev in zip(cand, EV):
            per = [max((float(ev @ V[j]) for j in range(n) if owner[j] == m), default=0.0) for m in set(owner)]
            if sum(1 for v in per if v >= CONS_TAU) == 1 and max(per) >= CONS_HI:
                drop.add(li)
        card = "\n".join(ln for li, ln in enumerate(lines) if li not in drop)
    return card


def synth_union(member_cards):
    """ADDITIVE de-identified aggregation (un_*): keep the FULL deduped union of the team's decision elements (breadth >
    any single card), then an LLM assembles a comprehensive card that GENERICIZES member-specific specifics. NO deletion,
    NO certificate -- validated at 2AFC chance on MAD despite retaining everything. Tests whether pooling >1 expert's
    combined knowledge beats the lossy one-shot neutral pool on a COMPETENCE metric (crowd-wisdom)."""
    from cmd_consensus_pool import elements as _elem, embed as _embed
    elems = []
    for mc in member_cards:
        elems += _elem(mc)
    if not elems:
        return ""
    V = _embed(elems); n = len(elems)
    kept = []
    for i in range(n):
        if all(float(V[i] @ V[j]) < 0.80 for j in kept):   # semantic dedupe (one representative per cluster)
            kept.append(i)
    body = "\n".join(f"- {elems[i]}" for i in kept)
    return chat([{"role": "system", "content": "You aggregate the COMPLETE decision expertise of a team into ONE shared "
                  "skill card, while de-identifying anything that points to a single member."},
                 {"role": "user", "content": "Below is the COMPLETE set of distinct decision/skill heuristics used across "
                  f"a team (deduplicated across all members):\n\n{body}\n\nWrite ONE comprehensive, well-organized shared "
                  "skill card that PRESERVES ALL of this decision knowledge -- do NOT drop useful heuristics and do NOT "
                  "collapse them into vague platitudes; keep every concrete, actionable rule. HOWEVER, GENERICIZE anything "
                  "that identifies a single member (personal/product/tool names, idiosyncratic phrasing, private one-off "
                  "specifics) so the card reads as the team's shared expertise and no single contributor is identifiable. "
                  "Organize by theme. Output ONLY the card."}],
                model="deepseek-chat", temperature=0.3, max_tokens=2400) or ""


SYS = ("You compare two answers to the SAME statistics question and decide which is MORE competent and correct "
       "-- sounder method choice, correct assumptions, handles the subtlety. IGNORE writing style, length, tone.")
USR = "Question:\n{q}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\nWhich is more competent? Answer ONLY 'A' or 'B' or 'TIE'."


def main():
    archive = Path(sys.argv[1])
    q_title, q_body, q_acc, a_body, cohort = CVP.load(archive)
    users = sorted(cohort, key=lambda u: -len(cohort[u]))[:CVP.NEXP]
    grp = CG.group_random(users, CVP.K, CVP.SEED)
    ngrp = len(set(grp.values()))
    print(f"CV util-judge export | using {len(users)} experts | K={CVP.K} groups={ngrp}", flush=True)

    indiv = {u: CVP.nuwa_assemble(CVP.nuwa_extract([b for (_, _, _, b) in cohort[u][:CVP.NCARD]])) for u in users}
    byc = defaultdict(list)
    for u in users:
        byc[grp[u]].append(u)
    shared = {g: CG.synth_shared([indiv[u] for u in mem]) for g, mem in byc.items()}
    neutral = {g: synth_neutral([indiv[u] for u in mem]) for g, mem in byc.items()} if NEUTRAL else {}
    if CONSENSUS:
        print("building consensus-aggregation cards ...", flush=True)
    consensus = {g: synth_consensus([indiv[u] for u in mem]) for g, mem in byc.items()} if CONSENSUS else {}
    if UNION:
        print("building union-genericize (additive, no-deletion) cards ...", flush=True)
    union = {g: synth_union([indiv[u] for u in mem]) for g, mem in byc.items()} if UNION else {}
    vec = {u: CVP.cvec(indiv[u]) for u in users}
    stranger = {u: max((v for v in users if grp[v] != grp[u]), key=lambda v: CVP.cos(vec[u], vec[v])) for u in users}
    deid = {}
    if DEID:
        step2 = json.loads(STEP2C.read_text(encoding="utf-8"))
        for arm in DEID:
            if arm not in step2:
                sys.exit(f"de-id arm '{arm}' not in {STEP2C.name} (have {list(step2)})")
            deid[arm] = {u: step2[arm].get(str(u)) or step2[arm].get(u) for u in users}

    # gather (unit, q); collect + PARALLEL pre-warm every draft (deepseek t0, sqlite-cached -> reruns free) so the
    # per-unit CVP.draft calls below all hit cache (serial drafting of the de-id arms is otherwise the bottleneck).
    units_q = []
    for u in users:
        for (aid, qid, sc, b) in cohort[u][CVP.NCARD:CVP.NCARD + CVP.NHELD]:
            acc = q_acc.get(qid)
            if acc and a_body.get(acc):
                units_q.append((u, qid, f"{q_title.get(qid, '')}\n{CVP.plain(q_body.get(qid, ''))[:1200]}"))
    jobs = []
    for (u, qid, q) in units_q:
        cards = [None, indiv[u], indiv[stranger[u]], shared[grp[u]]] + ([neutral[grp[u]]] if NEUTRAL else [])
        cards += [consensus[grp[u]]] if CONSENSUS else []
        cards += [union[grp[u]]] if UNION else []
        cards += [deid[arm][u] for arm in DEID]
        jobs += [(c, q) for c in cards]
    print(f"pre-warming {len(jobs)} drafts in parallel ...", flush=True)
    de.pool(lambda cq: CVP.draft(cq[0], cq[1]), jobs)

    tasks = []
    meta = {}
    i = 0
    for (u, qid, q) in units_q:
        d_no = CVP.plain(CVP.draft(None, q))[:1400]                       # cache hits (deepseek t0)
        d_in = CVP.plain(CVP.draft(indiv[u], q))[:1400]
        d_st = CVP.plain(CVP.draft(indiv[stranger[u]], q))[:1400]
        d_sh = CVP.plain(CVP.draft(shared[grp[u]], q))[:1400]
        d_ne = CVP.plain(CVP.draft(neutral[grp[u]], q))[:1400] if NEUTRAL else ""
        d_co = CVP.plain(CVP.draft(consensus[grp[u]], q))[:1400] if CONSENSUS else ""
        d_un = CVP.plain(CVP.draft(union[grp[u]], q))[:1400] if UNION else ""
        unit = f"{u}_{qid}"
        base_cmps = [("in_no", (d_in, d_no)), ("in_st", (d_in, d_st)), ("sh_no", (d_sh, d_no))]
        neutral_cmps = [("ne_no", (d_ne, d_no)), ("ne_in", (d_ne, d_in))]   # neutral helps? / neutral vs ceiling
        cons_cmps = [("co_no", (d_co, d_no)), ("co_ne", (d_co, d_ne))] if CONSENSUS else []   # consensus helps? / vs neutral
        union_cmps = ([("un_no", (d_un, d_no))] + ([("un_ne", (d_un, d_ne))] if NEUTRAL else [])
                      + ([("un_in", (d_un, d_in))])) if UNION else []   # union helps? / vs neutral / vs indiv ceiling
        deid_cmps = []
        for arm in DEID:
            d_ar = CVP.plain(CVP.draft(deid[arm][u], q))[:1400]
            deid_cmps.append((f"ne_{arm}", (d_ne, d_ar)))                  # neutral vs de-id (head-to-head, same-wave)
        if DEID_ONLY:
            cmps = deid_cmps
        elif UNION_ONLY:
            cmps = union_cmps
        elif CONSENSUS_ONLY:
            cmps = cons_cmps
        elif NEUTRAL_ONLY:
            cmps = neutral_cmps + deid_cmps
        else:
            cmps = base_cmps + (neutral_cmps if NEUTRAL else []) + cons_cmps + union_cmps + deid_cmps
        for cmp, (X, Y) in cmps:
            for role, (first, second) in [("r1", (X, Y)), ("r2", (Y, X))]:   # r1=X-vs-Y, r2=Y-vs-X (bidir)
                pid = f"T{i:04d}"; i += 1
                tasks.append({"pid": pid, "prompt": USR.format(q=q, a=first, b=second)})
                meta[pid] = {"u": str(u), "unit": unit, "cmp": cmp, "role": role}

    batches = [[] for _ in range(NBATCH)]
    for t in tasks:
        batches[int(t["pid"][1:]) % NBATCH].append(t)
    for j, b in enumerate(batches):
        (OUT / f"batch_{j}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "sys.txt").write_text(SYS, encoding="utf-8")
    n_units = len({m["unit"] for m in meta.values()})
    print(f"wrote {NBATCH} batches ({[len(b) for b in batches]}) = {len(tasks)} judge tasks over {n_units} units "
          f"(3 cmp x 2 orders) + meta + sys -> {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
