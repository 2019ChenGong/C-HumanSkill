"""ELEMK — decomposed pooling pipeline v2 (design frozen in results/ELEMK_DESIGN.md, #117).

Replaces the one-prompt black-box pool with an explicit, auditable pipeline:
  extract   deepseek decomposes each member's aggro card into SELF-CONTAINED atomic decision elements
            (no template scaffolding / headers — fixes the fake-support-from-skeleton bug of line-split conspf)
  pairs     per cluster: element x other-member best-match cosine (text-embedding-3-small);
            >=AUTO_YES auto-same, <AUTO_NO auto-diff, gray band -> batches for FREE sonnet-4.6 adjudication
            (conservative: over-merge inflates support = a PRIVACY bug)
  fuse      support(e) = 1 + #other-members matched; keep support>=Q (Q=2 and Q=3 arms); dedupe cos>=0.80
            keeping the highest-support representative; deepseek assembles (conspf ASSEMBLE prompt, verbatim);
            conspf anti-drift post-filter (assembled line support==1@TAU & top-cos>=HI -> drop)

Stages:  STAGE=cost | extract | pairs | fuse   (DATASET=mad K=8 SEED=0 fixed for the pilot; env-overridable)
Cards -> data/20mad/cmd_shared_cards_mad__elemk_q{2,3}.json (+ _stats sidecar with support histograms = audit artifact)
"""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GROUP", "random")
os.environ.setdefault("DATASET", "mad")
import cmd_gate as CG                            # noqa: E402
import deid_enron as de                          # noqa: E402  (de.pool = parallel LLM map)
from cmd_consensus_pool import embed             # noqa: E402  (text-embedding-3-small, L2-normalized)
from src.llm import chat                         # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

STAGE = os.environ.get("STAGE", "cost")
DS = os.environ.get("DATASET", "mad")
# V4-X generalization (ELEMK_DESIGN.md V4-X appendix): same pipeline on Enron (k8_s1, aggro source) and
# CV (k8_s0, aggro==nuwa -- registered difference). MAD paths are byte-identical to the pilot's.
_RES = {"mad": "mad", "enron": "enron", "cv": "se"}[DS]                      # results/<dir> convention
_CARDBASE = {"mad": "cmd_shared_cards_mad", "enron": "cmd_shared_cards", "cv": "cmd_shared_cards_cv"}[DS]
K = int(os.environ.get("K", 8))
SEED = int(os.environ.get("SEED", 0))
GEN = "deepseek-chat"
AUTO_YES = float(os.environ.get("AUTO_YES", 0.65))
AUTO_NO = float(os.environ.get("AUTO_NO", 0.45))
QS = [int(x) for x in os.environ.get("QS", "2,3").split(",")]
DEDUP = float(os.environ.get("DEDUP", 0.80))
TAU = float(os.environ.get("TAU", 0.55))     # post-filter only (instrument-consistent with conspf)
HI = float(os.environ.get("HI", 0.75))
BATCH = int(os.environ.get("BATCH", 50))

ELEMS_P = CG.SE / f"elemk_elements_k{K}_s{SEED}.json"
# Reviewer BLOCKER (V4-X): the adjudication dir must be namespaced by (K, SEED) -- a pairs.json built under a
# forgotten default SEED=0 would silently zero every support count in a SEED=1 fuse (cluster keys embed the
# seed, so no lookup ever hits). MAD keeps its legacy un-suffixed path (already adjudicated + shipped).
ADJ = ROOT / "results" / _RES / ("elemk_adj" if DS == "mad" else f"elemk_adj_k{K}_s{SEED}")
PAIRS_P = ADJ / "pairs.json"

# ---- FROZEN prompts (pre-registered in ELEMK_DESIGN.md; do not edit after seeing results) ----
EX_SYS = "You decompose a professional's skill/decision card into standalone atomic decision heuristics."
EX_USR = ("SKILL CARD:\n{card}\n\nList every distinct decision heuristic / working rule in this card as a JSON "
          "array of strings. Each string must be: (1) a single self-contained actionable rule understandable "
          "WITHOUT the card (no 'Step 3', no references to 'above' or 'this section'); (2) concrete — keep "
          "thresholds, conditions, and sequences; (3) 8-30 words. EXCLUDE: headers, section titles, generic "
          "scaffolding (e.g. 'follow a structured approach'), and anything that is not an actionable decision "
          "rule. Output ONLY the JSON array.")
ADJ_SYS = ("You judge whether two statements express the SAME actionable decision heuristic — the same underlying "
           "rule a professional would follow, such that having both on one skill card would be redundant. Be "
           "conservative: answer same=true ONLY if they describe the same rule/action; topical similarity, shared "
           "domain, or partial overlap is NOT enough. Judge each item in isolation. Output valid JSON only.")
ASSEMBLE_SYS = ("You assemble a shared skill card for a team from a vetted list of consensus points the team "
                "members have in common.")

# ---- V3 fix arm (V2b attribution, frozen in ELEMK_DESIGN.md appendix): same elements/filter, only the RENDERING
# changes — the card is framed as background reference and every process-gate is demoted to a last-resort
# conditional, because V2 cards rendered gates as unconditional policy and drafts obeyed them into misfires. ----
ASM = os.environ.get("ASM", "v2")
V3_PREAMBLE = ("Background reference distilled from the team's shared practice. Engage the specific case first: "
               "analyze what the report already provides before applying any rule below, and apply a rule only "
               "where its condition actually holds.")
ASSEMBLE_V3_SYS = ("You assemble a shared skill card for a team from a vetted list of consensus points the team "
                   "members have in common. The card will be handed to a colleague as BACKGROUND REFERENCE while "
                   "they work a specific case — it is not a procedure to execute.")

# ---- V4 carrier-content separation (frozen in ELEMK_DESIGN.md V4 appendix). The carrier is generated ONCE
# with NO member data in context (structurally zero-privacy), shared byte-identically by all clusters, and the
# card is composed DETERMINISTICALLY (no assembly LLM -> no drift channel; the certificate is exact by
# construction, so the conspf anti-drift post-filter does not apply to v4 -- G1' census is the guard). ----
# FROZEN per dataset (V4 appendix for mad; V4-X appendix for enron/cv -- each avoids its judge rubric's key
# phrases: Enron sound/risks/trade-offs/actionable, CV correct/caveats/conditions; residual overlap is a
# registered caveat, the ne-v4 and v4-carrier differences are immune to it).
CARRIER_PROMPTS = {"mad": (
    "Write a ~400-word general advisory guide for software bug triage, to serve as the surrounding context of a "
    "team skill card. Requirements: (1) encourage engaging with the specific bug report first — read what it "
    "already provides, form a hypothesis about the underlying mechanism, and reason about what will most likely "
    "happen to the bug and why; (2) present duplicates, regressions, environment-specific behavior, "
    "hardware/configuration factors, and severity/priority trade-offs as POSSIBILITIES worth weighing, never as "
    "procedures to enforce; (3) phrase everything as considerations ('it can help to…', 'worth weighing…'), "
    "never as commands; (4) do NOT include any instruction to request more information, ask for reproduction "
    "steps, defer or withhold action, or close reports — these process moves are covered elsewhere; (5) short "
    "plain-markdown sections with brief bullet groups; no numbered mandatory steps. Output ONLY the guide."),
    "enron": (
    "Write a ~400-word general advisory guide for handling everyday work-email situations at a large company "
    "(approvals, requests, scheduling, coordination, external partners), to serve as the surrounding context of "
    "a team skill card. Requirements: (1) encourage engaging with the specific message first — read what it "
    "already says, identify what the sender is actually asking for, and think through what each way of "
    "responding would mean for the people and the commitments involved; (2) present approving versus declining, "
    "committing now versus keeping options open, looping in colleagues, and the timing of a reply as "
    "POSSIBILITIES worth weighing, never as procedures to enforce; (3) phrase everything as considerations "
    "('it can help to…', 'worth weighing…'), never as commands; (4) do NOT include any instruction to request "
    "more information, defer or withhold a reply, or escalate by default — these process moves are covered "
    "elsewhere; (5) short plain-markdown sections with brief bullet groups; no numbered mandatory steps. "
    "Output ONLY the guide."),
    "cv": (
    "Write a ~400-word general advisory guide for answering applied statistics questions, to serve as the "
    "surrounding context of a team skill card. Requirements: (1) encourage engaging with the specific question "
    "first — read what it already provides about the data, the model, and the goal, and form a view of the "
    "practical problem behind the statistical one before reaching for any technique; (2) present model "
    "assumptions, sample size, measurement quality, multiple-testing exposure, and the choice between simpler "
    "and more elaborate methods as POSSIBILITIES worth weighing, never as procedures to enforce; (3) phrase "
    "everything as considerations ('it can help to…', 'worth weighing…'), never as commands; (4) do NOT include "
    "any instruction to request more information or clarification, defer answering, or decline to answer — "
    "these process moves are covered elsewhere; (5) short plain-markdown sections with brief bullet groups; "
    "no numbered mandatory steps. Output ONLY the guide.")}
CARRIER_PROMPT = CARRIER_PROMPTS[DS]


def carrier_text():
    p = CG.SE / f"elemk_carrier_{DS}.txt"
    if p.exists():
        return p.read_text(encoding="utf-8")
    msg = [{"role": "system", "content": "You write general professional craft guides."},
           {"role": "user", "content": CARRIER_PROMPT}]
    t = (chat(msg, model=GEN, temperature=0.3, max_tokens=900) or "").strip()
    assert t, "carrier generation returned empty"
    p.write_text(t, encoding="utf-8")
    return t


# Reviewer MAJOR-3 (V4-X): the prompt's word-avoidance is not reliably obeyed (the MAD carrier still echoed
# "duplicates"/"condition") -- scan the OUTPUT for the target judge-rubric's literal keywords and report.
RUBRIC_KW = {"mad": ("duplicate", "condition", "trade-off", "risk"),
             "enron": ("sound", "risk", "trade-off", "actionable"),
             "cv": ("correct", "caveat", "condition")}


def rubric_scan(t):
    low = t.lower()
    return [w for w in RUBRIC_KW[DS] if w in low]


def stage_carrieronly():
    """Write the carrier-only control card: every cluster key -> the SAME byte-identical carrier text."""
    _aggro, byc = load_clusters()
    t = carrier_text()
    hits = rubric_scan(t)
    if hits:
        # one frozen retry with an explicit literal word-ban appended; report both scans, keep the retry
        p = CG.SE / f"elemk_carrier_{DS}.txt"
        msg = [{"role": "system", "content": "You write general professional craft guides."},
               {"role": "user", "content": CARRIER_PROMPT + " Additional hard constraint: never use the words "
                + ", ".join(f"'{w}'" for w in RUBRIC_KW[DS]) + " or their derivatives."}]
        t2 = (chat(msg, model=GEN, temperature=0.3, max_tokens=900) or "").strip()
        h2 = [w for w in RUBRIC_KW[DS] if w in t2.lower()]
        print(f"[carrieronly] rubric-keyword scan: first draft hit {hits}; word-banned retry hit {h2 or 'none'}")
        if t2 and len(h2) < len(hits):
            t = t2
            p.write_text(t, encoding="utf-8")
    else:
        print("[carrieronly] rubric-keyword scan: clean")
    outp = CG.SE / f"{_CARDBASE}__carrier_only.json"
    cards = {f"k{K}_s{SEED}_{cid}": t for cid in sorted(byc)}
    outp.write_text(json.dumps(cards, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[carrieronly] {len(cards)} keys (byte-identical carrier, {len(t.split())} words) -> {outp.name}")


def compose_v4(kept_txt):
    paras = [b for b in carrier_text().split("\n\n") if b.strip()]
    cut = max(1, (len(paras) * 2) // 3)
    block = ("**Shared practices distilled from the team** (each point independently held by multiple members; "
             "member-specific detail already removed):\n" + "\n".join(f"- {t}" for t in kept_txt))
    return "\n\n".join(paras[:cut] + [block] + paras[cut:])


def assemble(kept_txt):
    body = "\n".join(f"- {t}" for t in kept_txt)
    if ASM == "v3":
        msg = [{"role": "system", "content": ASSEMBLE_V3_SYS},
               {"role": "user", "content": f"Consensus points shared by MULTIPLE members (member-specific detail "
                f"already removed):\n\n{body}\n\nWrite ONE coherent, well-organized shared skill card using ONLY "
                "these consensus points; do not add new points or reintroduce member-specific detail. Keep it "
                "concrete and usable. Two rendering requirements: (1) start the card with this exact preamble "
                f"line: \"{V3_PREAMBLE}\" (2) render every process-gating move (requesting more information or "
                "reproduction steps, withholding or deferring action, closing as incomplete or unreproducible) as "
                "an explicit LAST-RESORT conditional: state what to check in the report first, and apply the move "
                "only if that is truly absent from the report; never render a gate as an unconditional command or "
                "as the first step. Output ONLY the card."}]
    else:
        msg = [{"role": "system", "content": ASSEMBLE_SYS},
               {"role": "user", "content": f"Consensus points shared by MULTIPLE members (member-specific detail "
                f"already removed):\n\n{body}\n\nWrite ONE coherent, well-organized shared skill card using ONLY "
                "these consensus points; do not add new points or reintroduce member-specific detail. Keep it "
                "concrete and usable. Output ONLY the card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=1300) or ""


def parse_elems(out):
    """Robust JSON-array parse: strip fences, slice [ .. ], fallback to bullet lines."""
    if not out:
        return []
    t = re.sub(r"```(?:json)?", "", out).strip()
    i, j = t.find("["), t.rfind("]")
    if i >= 0 and j > i:
        try:
            arr = json.loads(t[i:j + 1])
            els = [str(x).strip() for x in arr if isinstance(x, str)]
            if els:
                return els
        except Exception:
            pass
    return [re.sub(r"^\s*[-*\d.)\"']+\s*", "", ln).strip().rstrip('",')
            for ln in t.splitlines() if len(re.findall(r"\w+", ln)) >= 5]


def clean(els):
    """Drop scaffolding: <5 content words, ALL-CAPS, or bare titles."""
    out = []
    for e in els:
        if len(re.findall(r"\w+", e)) < 5 or e.isupper():
            continue
        out.append(e)
    return out


def load_clusters():
    _d, authors, _n, aggro, _r, _rt = CG.load()
    grp, byc = CG.make_groups(aggro, authors, K, SEED)
    return aggro, {cid: mem for cid, mem in byc.items() if len(mem) >= K}


def stage_cost():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    aggro, byc = load_clusters()
    members = sorted({m for mem in byc.values() for m in mem})
    tin = sum(len(enc.encode(EX_SYS + EX_USR.format(card=aggro[m]))) for m in members)
    tout_est = sum(int(len(enc.encode(aggro[m])) * 0.6) for m in members)   # elements ~60% of card tokens
    ex_cost = tin / 1e6 * 0.28 + tout_est / 1e6 * 1.10
    # assemble: 16 clusters x len(QS) arms; input ~ kept elements (~40% of cluster elements), output <=1300
    n_cl = len(byc)
    asm_in = int(tout_est * 0.4) + n_cl * len(QS) * 200
    asm_out = n_cl * len(QS) * 1100
    asm_cost = asm_in / 1e6 * 0.28 + asm_out / 1e6 * 1.10
    print(f"[cost] extract: {len(members)} members, in={tin/1e3:.0f}k out~{tout_est/1e3:.0f}k tok -> ~${ex_cost:.2f}")
    print(f"[cost] assemble: {n_cl} clusters x {len(QS)} Q-arms -> ~${asm_cost:.2f}")
    print(f"[cost] embeddings (text-embedding-3-small): ~${(tout_est*1.2)/1e6*0.02:.3f}")
    print(f"[cost] TOTAL build ~= ${ex_cost + asm_cost + 0.03:.2f}   (adjudication/judges = free sonnet subagents)")


def stage_extract():
    aggro, byc = load_clusters()
    members = sorted({m for mem in byc.values() for m in mem})
    cache = json.loads(ELEMS_P.read_text(encoding="utf-8")) if ELEMS_P.exists() else {}
    todo = [m for m in members if m not in cache]
    print(f"[extract] {len(members)} members, cached={len(members)-len(todo)}, to-run={len(todo)}", flush=True)
    if todo:
        def one(m):
            out = chat([{"role": "system", "content": EX_SYS},
                        {"role": "user", "content": EX_USR.format(card=aggro[m])}],
                       model=GEN, temperature=0.2, max_tokens=2000)
            return clean(parse_elems(out))
        for m, els in zip(todo, de.pool(one, todo)):
            cache[m] = els
        ELEMS_P.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    ns = [len(cache[m]) for m in members]
    bad = [m for m in members if len(cache[m]) < 3]
    print(f"[extract] elements/member: mean={np.mean(ns):.1f} min={min(ns)} max={max(ns)}  -> {ELEMS_P.name}")
    if bad:
        print(f"  ⚠ members with <3 elements (check parses): {bad}")


def cluster_elements(byc, cache):
    """ck -> (texts, owners) with global order stable."""
    out = {}
    for cid, mem in sorted(byc.items()):
        texts, owners = [], []
        for m in mem:
            for e in cache.get(m, []):
                texts.append(e); owners.append(m)
        out[f"k{K}_s{SEED}_{cid}"] = (texts, owners)
    return out


def stage_pairs():
    _aggro, byc = load_clusters()
    cache = json.loads(ELEMS_P.read_text(encoding="utf-8"))
    ADJ.mkdir(parents=True, exist_ok=True)
    recs, gray_items = [], []
    for ck, (texts, owners) in cluster_elements(byc, cache).items():
        if not texts:
            continue
        V = np.stack(embed(texts))
        S = V @ V.T
        mem = sorted(set(owners))
        oarr = np.array([mem.index(o) for o in owners])
        for i in range(len(texts)):
            for mo_i, mo in enumerate(mem):
                if mo == owners[i]:
                    continue
                js = np.where(oarr == mo_i)[0]
                bj = js[int(np.argmax(S[i, js]))]
                c = float(S[i, bj])
                cls = "yes" if c >= AUTO_YES else ("no" if c < AUTO_NO else "gray")
                rec = {"ck": ck, "ei": i, "mo": mo, "cos": round(c, 4), "cls": cls}
                if cls == "gray":
                    pid = f"g{len(gray_items)}"
                    rec["pid"] = pid
                    gray_items.append({"pid": pid, "a": texts[i], "b": texts[int(bj)]})
                recs.append(rec)
    PAIRS_P.write_text(json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    nb = 0
    for nb, s in enumerate(range(0, len(gray_items), BATCH)):
        (ADJ / f"batch_{nb}.json").write_text(json.dumps(gray_items[s:s + BATCH], ensure_ascii=False, indent=1),
                                              encoding="utf-8")
        nb += 1
    (ADJ / "sys.txt").write_text(ADJ_SYS + "\n\nFor each item in the batch JSON, output one object "
                                 '{"pid": "...", "same": true|false}. Reply with ONLY a JSON array of these objects, '
                                 "one per item, covering every pid exactly once.", encoding="utf-8")
    from collections import Counter
    cc = Counter(r["cls"] for r in recs)
    print(f"[pairs] (element x other-member) records={len(recs)}  auto-yes={cc['yes']} auto-no={cc['no']} "
          f"gray={cc['gray']}  -> {max(nb,0)} batches of <= {BATCH} in {ADJ.relative_to(ROOT)}")


def stage_fuse():
    _aggro, byc = load_clusters()
    cache = json.loads(ELEMS_P.read_text(encoding="utf-8"))
    recs = json.loads(PAIRS_P.read_text(encoding="utf-8"))
    ans = {}
    for f in sorted(ADJ.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", f.stem):
            continue
        for r in json.loads(f.read_text(encoding="utf-8-sig")):
            ans[r["pid"]] = bool(r.get("same"))
    gray_pids = [r["pid"] for r in recs if r["cls"] == "gray"]
    missing = [p for p in gray_pids if p not in ans]
    if missing:
        sys.exit(f"[fuse] {len(missing)}/{len(gray_pids)} gray pairs unanswered (e.g. {missing[:5]}) — dispatch first.")
    # Reviewer BLOCKER (V4-X): a stale pairs.json from a different (K, SEED) has cluster keys that never match
    # this run's -- supports silently collapse to 1 and cards go near-empty. Fail loudly instead.
    _pair_cks = {r["ck"] for r in recs}
    _run_cks = set(cluster_elements(byc, cache))
    assert _pair_cks <= _run_cks, (f"pairs.json cluster keys do not match this run's partition "
                                   f"(pairs e.g. {sorted(_pair_cks)[:2]} vs run {sorted(_run_cks)[:2]}) -- "
                                   f"stale ADJ dir? re-run STAGE=pairs with the correct K/SEED")

    matched = {}   # (ck, ei) -> set(other members matched)
    for r in recs:
        ok = r["cls"] == "yes" or (r["cls"] == "gray" and ans[r["pid"]])
        if ok:
            matched.setdefault((r["ck"], r["ei"]), set()).add(r["mo"])

    clus = cluster_elements(byc, cache)
    for Q in QS:
        sfx = {"v3": "v3", "v4": "v4"}.get(ASM, "")
        outp = CG.SE / f"{_CARDBASE}__elemk_q{Q}{sfx}.json"
        statp = outp.with_name(outp.stem + "_stats.json")
        cards = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {}
        stats = {}
        plan = []
        for ck, (texts, owners) in clus.items():
            if not texts:
                continue
            V = np.stack(embed(texts))
            sup = [1 + len(matched.get((ck, i), ())) for i in range(len(texts))]
            keep = sorted([i for i in range(len(texts)) if sup[i] >= Q], key=lambda i: -sup[i])
            kept = []
            for i in keep:                                        # semantic dedupe, highest-support representative
                if all(float(V[i] @ V[j]) < DEDUP for j in kept):
                    kept.append(i)
            wdrop = 0
            if ASM == "v4":
                # step 5' wording-anonymity filter (V4 design amendment, ELEMK_DESIGN.md): v4 quotes elements
                # VERBATIM, so an element whose support came only from gray-band adjudication (content shared,
                # wording embedding-close to exactly ONE member at >=TAU with top>=HI) is a wording-level
                # exposure -- same TAU/HI as the conspf post-filter, applied pre-composition. Caught by the G1'
                # census gate at 5.2% (> q2+3pp) before any drafting spend.
                mem = sorted(set(owners))
                oarr = np.array([mem.index(o) for o in owners])
                safe = []
                for i in kept:
                    per = [float((V[i] @ V[oarr == m].T).max()) if (oarr == m).any() else 0.0
                           for m in range(len(mem))]
                    n55 = sum(1 for v in per if v >= TAU)
                    if n55 == 1 and max(per) >= HI:
                        wdrop += 1
                    else:
                        safe.append(i)
                kept = safe
            hist = {}
            for s in sup:
                hist[min(s, 8)] = hist.get(min(s, 8), 0) + 1
            stats[ck] = {"n_elements": len(texts), "support_hist": {str(k): hist[k] for k in sorted(hist)},
                         "n_geQ": len(keep), "n_kept": len(kept)}
            if ASM == "v4":
                stats[ck]["wording_dropped"] = wdrop
            plan.append((ck, [texts[i] for i in kept], V, owners, texts))
            print(f"  Q{Q} {ck}: {len(texts)} el, >=Q {len(keep)} ({100*len(keep)/len(texts):.0f}%), "
                  f"dedup-> {len(kept)}", flush=True)
        todo = [(ck, kt) for ck, kt, *_ in plan if ck not in cards]
        if todo:
            if ASM == "v4":
                print(f"  Q{Q}: composing {len(todo)} cards deterministically (carrier + certified elements) ...",
                      flush=True)
                for ck, kt in todo:
                    cards[ck] = compose_v4(kt)
            else:
                print(f"  Q{Q}: assembling {len(todo)} cards (deepseek) ...", flush=True)
                for (ck, kt), card in zip(todo, de.pool(lambda t: assemble(t[1]), todo)):
                    cards[ck] = card
        # anti-drift post-filter (conspf-identical: assembled line support==1@TAU & top-cos>=HI -> drop).
        # v4 skips it BY DESIGN: composition is deterministic (no LLM -> no drift channel); member-derived
        # lines are certified elements verbatim, guarded by the G1' census instead (ELEMK_DESIGN.md V4).
        for ck, kt, V, owners, texts in plan:
            if ASM == "v4":
                w = len(re.findall(r"\w+", cards[ck]))
                stats[ck].update({"dropped_lines": 0, "words": w})
                if stats[ck]["n_kept"] < 3 or w < 60:
                    stats[ck]["thin"] = True
                continue
            card = cards[ck]
            lines = card.splitlines()
            cand = [(li, re.sub(r"^\s*[-*•\d.)#]+\s*", "", ln).strip()) for li, ln in enumerate(lines)]
            cand = [(li, s) for li, s in cand if len(re.findall(r"\w+", s)) >= 5 and not s.isupper()]
            dropped = 0
            if cand:
                EV = embed([s for _, s in cand])
                mem = sorted(set(owners))
                drop = set()
                for (li, _s), ev in zip(cand, EV):
                    per = [max((float(ev @ V[j]) for j in range(len(texts)) if owners[j] == m), default=0.0)
                           for m in mem]
                    if sum(1 for v in per if v >= TAU) == 1 and max(per) >= HI:
                        drop.add(li)
                if drop:
                    cards[ck] = "\n".join(ln for li, ln in enumerate(lines) if li not in drop)
                    dropped = len(drop)
            w = len(re.findall(r"\w+", cards[ck]))
            stats[ck].update({"dropped_lines": dropped, "words": w})
            if stats[ck]["n_kept"] < 3 or w < 60:
                stats[ck]["thin"] = True
        outp.write_text(json.dumps(cards, ensure_ascii=False, indent=1), encoding="utf-8")
        statp.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
        ws = [stats[ck]["words"] for ck in stats]
        thin = [ck for ck in stats if stats[ck].get("thin")]
        print(f"[fuse] Q{Q}: {len(stats)} cards -> {outp.name}  words mean={np.mean(ws):.0f} "
              f"min={min(ws)}  thin={thin or 'none'}")


if __name__ == "__main__":
    {"cost": stage_cost, "extract": stage_extract, "pairs": stage_pairs, "fuse": stage_fuse,
     "carrieronly": stage_carrieronly}[STAGE]()
