"""V5 — sanitize, don't rebuild (design frozen in results/ELEMK_DESIGN.md V5 appendix, #123).

The black-box pooled card stays as selector/organizer (anatomy, elemk_v3_gates MODE=anatomy: 91-94%
of its lines trace to member elements, majority sup=1, 80-91% share a 6-word shingle with a member
aggro card); V5 rewrites its WORDING line by line under two deterministic gates and ships a per-line
provenance audit instead of rebuilding the card from consensus (the V4 route that loses on in>sham
datasets):

  lexical gate   the rewritten line shares NO 6-word shingle with {the original line, all cluster
                 member elements, all cluster member aggro cards}   (same _shingles(n=6) definition
                 as the carrier-leak instrument in elemk_v3_gates.py)
  fidelity gate  cos(original, rewrite) >= FID (0.75, text-embedding-3-small) -- content preserved

Gate violations retry (<=4, dry-run amendment) with explicit feedback naming the violation;
still-failing lines are DROPPED and counted (drop rate > 5% = G-lex FAIL-level anomaly per design).

STAGE=cost|build   DATASET=mad|cv|enron   K/SEED env (canon: mad 8,0 / cv 8,0 / enron 8,1)
Out: data/<ds>/<cardbase>__v5san.json + __v5san_audit.json + __v5san_stats.json
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
import elemk_build as EB                         # noqa: E402  (env-driven: DATASET/K/SEED shared)
import deid_enron as de                          # noqa: E402  (de.pool = parallel LLM map)
from cmd_consensus_pool import embed             # noqa: E402
from src.llm import chat, sample_one             # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

STAGE = os.environ.get("STAGE", "cost")
DS = EB.DS
K, SEED = EB.K, EB.SEED
_CANON = {"mad": (8, 0), "enron": (8, 1), "cv": (8, 0)}
# R1 (#125): multi-seed builds allowed -- K stays canonical, SEED may be 0/1/2 (cluster keys embed
# the seed so the shared OUT file cannot collide; the _config stamp enforces gate/prompt parity).
# R13 (#139): full k-gradient on MAD -- K may be any registered grid value (cluster keys embed k, so
# no collision; prompts/gates/thresholds identical). Non-MAD datasets stay canonical-K (R13 scope).
_KGRID = (2, 4, 6, 8, 10, 12) if DS == "mad" else (_CANON[DS][0],)
assert K in _KGRID and SEED in (0, 1, 2), \
    f"v5 for {DS} needs K in {_KGRID} and SEED in 0/1/2, got ({K},{SEED})"
if (K, SEED) != _CANON[DS]:
    print(f"[R1/R13] non-canonical (k{K}, s{SEED}) for {DS} (canonical k{_CANON[DS][0]}_s{_CANON[DS][1]})", flush=True)
# R6 (#130): the rewriter model is swappable (SAN_GEN); default = canonical deepseek, byte-neutral.
# qwen models run NON-thinking (same convention as mad_fc_judge_qwen / B2); extra is part of chat()'s
# cache key, and extra=None keeps every existing deepseek cache row valid.
GEN = os.environ.get("SAN_GEN", "deepseek-chat")
# R6c: every swapped rewriter runs NON-thinking so the swap holds the condition fixed (deepseek-chat
# is non-thinking); the family list covers models whose OpenRouter default may reason.
# R6d: "minimax" is deliberately NOT in the list — minimax-m2.7's endpoint rejects disabling reasoning
# (400 "Reasoning is mandatory"), so that arm runs with provider-default thinking; registered as a
# condition deviation in ELEMK_DESIGN.md R6d and reported flagged, never pooled with non-thinking arms.
SAN_EXTRA = ({"reasoning": {"enabled": False}}
             if any(f in GEN for f in ("qwen", "kimi", "grok", "glm")) else None)
FID = float(os.environ.get("FID", 0.75))
SHN = 6
MAXRETRY = 4
# V6 (ELEMK_DESIGN.md V6 appendix, #124): EDIT=min = minimal-edit sanitize. V5's whole-line
# fresh-wording rewrite was a NON-privacy constraint we added ourselves, and it is exactly where the
# MAD loss came from (wording noise + punt shift + drops). EDIT=min touches only the exact word runs
# that overlap member text; clean lines ship verbatim. EDIT=full keeps V5 byte-identical.
EDIT = os.environ.get("EDIT", "full")
assert EDIT in ("full", "min")
# R9 (#133): fresh-sampling rebuild. SAN_SAMPLE (int env) routes every EDIT=min rewriter call through
# sample_one — its cache key gains a _sample index plus a per-round salt (review M1: the empty->empty
# retry path repeats byte-identical messages, and a fixed key would replay one bad cached answer
# across rounds) — forcing fresh API draws instead of replaying the canonical chat() cache rows.
# Presence-tested (review MINOR-2): an explicit "0" counts as set. Unset = chat() path, byte-neutral.
_SS = os.environ.get("SAN_SAMPLE")
SAMPLE = int(_SS) if _SS is not None else None
if SAMPLE is not None:
    assert GEN == "deepseek-chat", "R9 fresh-sampling registered for the canonical rewriter only"
    assert SAN_EXTRA is None, "SAN_SAMPLE path uses sample_one, which does not forward `extra`"
    assert EDIT == "min", "SAN_SAMPLE is wired into the EDIT=min pipeline only (V5 full stays chat())"
MINFID = 0.90            # v6 fidelity gate (relaxed floor 0.80): edits are small, so demand closeness
MINFID_FLOOR = 0.80
# punt-status gate (v6): the rewrite must not change whether the line matches the punt REQ regex --
# frozen verbatim from elemk_v3_gates.py (V2b instrument); mechanically closes the V5-MAD punt channel.
REQ = re.compile(r"\b(request|ask|clarif|more information|more info|additional information"
                 r"|need.{0,20}(steps|details|info))\b", re.I)
# Reviewer MAJOR-4 (V6): the regex is hand-duplicated (elemk_v3_gates runs its MODE dispatch at import,
# so it cannot be imported from there). Pin the pattern so silent drift between the two copies fails loudly.
import hashlib as _hl                                    # noqa: E402
assert _hl.sha1(REQ.pattern.encode()).hexdigest()[:12] == "ae0b36dfa10e", \
    "REQ regex drifted from the elemk_v3_gates V2b frozen copy -- resync both and re-pin"

NE_P = EB.CG.SE / f"{EB._CARDBASE}__neutral_fixed.json"
_TAG = os.environ.get("TAG", "v5san" if EDIT == "full" else "v6min")
# R6 MAJOR-3: without this, a forgotten TAG under a swapped model resolves OUT_P to the CANONICAL
# file, whose existing cluster keys make the resume guard skip everything -- a $0 "successful" run
# that leaves every downstream axis silently testing deepseek cards.
assert GEN == "deepseek-chat" or _TAG not in ("v5san", "v6min"), \
    f"SAN_GEN={GEN!r} must write to its own TAG (e.g. TAG=v6min_qwen), not the canonical {_TAG!r} files"
# R9: same $0-silent-no-op family — under a canonical TAG the resume guard would skip every existing
# cluster and a "fresh-sampling rerun" would test nothing. Collisions with other historical TAGs
# (v6min_qwen etc.) are caught one layer down by the _config stamp (SAN_SAMPLE field mismatch).
assert SAMPLE is None or _TAG not in ("v5san", "v6min"), \
    f"SAN_SAMPLE={_SS} must write to its own TAG (e.g. TAG=v6min_rerun), not the canonical {_TAG!r} files"
OUT_P = EB.CG.SE / f"{EB._CARDBASE}__{_TAG}.json"
AUD_P = EB.CG.SE / f"{EB._CARDBASE}__{_TAG}_audit.json"
STAT_P = EB.CG.SE / f"{EB._CARDBASE}__{_TAG}_stats.json"

# ---- FROZEN prompt (pre-registered in ELEMK_DESIGN.md V5 appendix; do not edit after results) ----
RW_SYS = "You rewrite lines of a team skill card in fresh wording without changing their content."
RW_USR = ("Rewrite this line from a team skill card in completely fresh wording. Preserve the full "
          "actionable content — every condition, threshold, quantity, named method or tool, and "
          "sequence — but do NOT reuse any distinctive phrase of 4+ consecutive words from the "
          "original. If the line contains a person's name or an organization-internal project name, "
          "replace it with a generic role or descriptor. Keep roughly the same length and the same "
          "bullet/sentence form, and preserve any markdown emphasis markers (e.g. ** **) in place. "
          "Output ONLY the rewritten line.\n\nLINE:\n{line}")
FB_LEX = ("\n\nYour previous rewrite reused these exact phrases from the source material: {sh}. "
          "Produce a rewrite that shares NO run of 6 consecutive words with the source. Previous "
          "attempt:\n{prev}")
FB_FID = ("\n\nYour previous rewrite drifted too far from the original meaning — it must state the "
          "same rule with the same specifics, only in different words. Previous attempt:\n{prev}")
FB_NUM = ("\n\nYour previous rewrite dropped these exact values from the original: {toks}. They must "
          "appear unchanged in the rewrite. Previous attempt:\n{prev}")
FB_EMPTY = "\n\nYour previous response was empty. Return the rewritten line."

# ---- V6 FROZEN prompt + feedbacks (pre-registered in ELEMK_DESIGN.md V6 appendix) ----
MIN_SYS = "You make minimal edits to lines of a team skill card without changing their content."
MIN_USR = ("Minimally edit this line from a team skill card. The ONLY goal: it must no longer contain "
           "any of the following word sequences (rephrase just enough of each to break the sequence): "
           "{runs}. Keep every other word of the line VERBATIM — do not paraphrase anything you don't "
           "have to. Preserve all conditions, thresholds, quantities, named methods, markdown emphasis "
           "markers, and the line's bullet/sentence form and approximate length. Output ONLY the "
           "edited line.\n\nLINE:\n{line}")
FB_PUNT_ADD = ("\n\nYour previous edit ADDED request-for-information language that the original line "
               "does not have. Do not introduce words like 'request', 'ask', 'clarify', or 'more "
               "information' unless the original had them. Previous attempt:\n{prev}")
FB_PUNT_DROP = ("\n\nYour previous edit REMOVED the original line's request-for-information language. "
                "Keep that aspect: the edited line must still literally contain one of the words "
                "'request', 'ask', 'clarify', or the phrase 'more information'. Previous attempt:\n{prev}")
FB_LEN = "\n\nYour previous edit changed the line's length too much. Stay within the original length. Previous attempt:\n{prev}"

# R6f (#142): instruction-induced edit economy. ONE frozen GUIDE text (ELEMK_DESIGN.md R6f, byte-
# pinned there) appended to BOTH stage system prompts when SAN_GUIDE is set — the portability claim
# requires the identical text across models, so there is deliberately no per-model tailoring.
# Unset = byte-neutral (canonical builds untouched; prompt_sha1 stamps change with the patch, so a
# resume under the wrong prompt fails loudly via the existing _config assert).
GUIDE = (" Editing strategy, applied strictly: make the SMALLEST change that satisfies the request. "
         "Break a forbidden word sequence by changing roughly one word in every stretch of six — "
         "swap a single content word for an EQUALLY specific synonym, change word order, or shift a "
         "clause boundary — and keep the sentence skeleton, clause order, and every word you are "
         "not forced to change. Never replace a domain-specific term with a broader or vaguer one; "
         "never formalize or 'improve' the writing; never re-author the line from scratch. Preserve "
         "numbers, thresholds, conditions, roles, and the advice's direction exactly.")
_SG = os.environ.get("SAN_GUIDE")
if _SG is not None:
    assert _TAG not in ("v5san", "v6min"), \
        f"SAN_GUIDE must write to its own TAG (e.g. TAG=v6min_qwenguided), not the canonical {_TAG!r} files"
    RW_SYS += GUIDE
    MIN_SYS += GUIDE


def _shingles(text, n=SHN):
    """Identical to elemk_v3_gates._shingles (that file executes a MODE at import, so no import)."""
    w = re.findall(r"[a-z']+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _hit_runs(text, pool):
    """V6: the exact maximal word runs of `text` whose 6-shingles appear in the member pool —
    these are the ONLY spans the minimal edit is asked to break. Overlapping/adjacent shingle spans
    merge into ONE run (dry-run amendment 3: separate-but-overlapping runs double-counted coverage
    and mis-routed solvable lines to the full-rewrite path)."""
    w = re.findall(r"[a-z']+", text.lower())
    hits = [i for i in range(len(w) - SHN + 1) if " ".join(w[i:i + SHN]) in pool]
    runs, cur = [], None
    for i in hits:
        if cur is not None and i <= cur[1] + SHN:
            cur[1] = i
        else:
            cur = [i, i]
            runs.append(cur)
    return [" ".join(w[a:b + SHN]) for a, b in runs]


def _is_content(s):
    """Rewrite target = a line that can physically carry a 6-word shingle (>=6 [a-z'] tokens).
    Shorter lines (headers like '**Step 1: Reproduce Before Analyzing**') have ZERO 6-shingles, so the
    lexical certificate holds for them trivially — and paraphrasing 4-word titles just churns the
    fidelity gate (dry-run: half the drops were exactly these). Kept verbatim by design."""
    return len(re.findall(r"[a-z']+", s.lower())) >= SHN and not s.isupper()


# Reviewer BLOCKER-1: strip only true bullet/number/header markers. Bold markers (`**`) stay part of
# the content so `**Step 3: …**` reaches the rewriter INTACT (the original class ate the leading `**`
# and shipped a dangling trailing one -> ~50% of headers corrupted = a pure formatting confound).
# A `*` bullet requires trailing whitespace, which `**bold` never has.
_PREF = re.compile(r"^(\s*(?:[-•]\s*|\*\s+|\d+[.)]\s*|#+\s+)+)")


def _split_line(ln):
    m = _PREF.match(ln)
    pref = m.group(1) if m else ""
    return pref, ln[len(pref):].strip()


_NUM = re.compile(r"\d+(?:\.\d+)?%?")


def _nums(s):
    return set(_NUM.findall(s))


def _load_support():
    # R1 (#125) amendment ②: adj/pairs exist for canonical seeds only. They feed ONLY the audit
    # 'support' column (routing/gates never read them). Missing file -> None; a partition mismatch
    # is handled at the use site (support=null, never a fabricated 1).
    if not EB.PAIRS_P.exists():
        return None
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


def stage_cost():
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    cards = json.loads(NE_P.read_text(encoding="utf-8"))
    _aggro, byc = EB.load_clusters()
    cks = {f"k{K}_s{SEED}_{cid}" for cid in byc}
    lines = []
    for ck in sorted(cks & set(cards)):
        for ln in cards[ck].splitlines():
            _p, s = _split_line(ln)
            if _is_content(s):
                lines.append(s)
    # R6 MAJOR-5: price by the ACTUAL generator (the old constants were deepseek's), and for EDIT=min
    # size by the canonical v6min build's MEASURED call counts (clean lines never reach the LLM; the
    # per-content-line V5 sizing below over-counts lines and under-counts retries).
    PRICES = {"deepseek-chat": (0.28, 1.10), "openrouter/qwen/qwen3.7-max": (1.25, 5.00),
              # R6c: OpenRouter list prices fetched 2026-07-16; the 2x stop-gate covers drift
              "openrouter/moonshotai/kimi-k2.6": (0.60, 2.50),
              "openrouter/x-ai/grok-4.20": (1.25, 2.50),
              # R6d: list prices fetched 2026-07-18; minimax quote excludes its mandatory reasoning
              # tokens (billed as output) -- inflation unknown, the 2x stop-gate covers it.
              "openrouter/z-ai/glm-5.1": (0.97, 3.04),
              "openrouter/minimax/minimax-m2.7": (0.25, 1.00)}
    pin, pout = PRICES.get(GEN, (1.25, 5.00))
    mean_line = sum(len(enc.encode(s)) for s in lines) / max(len(lines), 1)
    ref = EB.CG.SE / f"{EB._CARDBASE}__v6min_stats.json"
    if EDIT == "min" and ref.exists():
        st = json.loads(ref.read_text(encoding="utf-8"))
        calls = sum((v["n_content"] - v["tiers"].get("verbatim", 0)) + v["retried"]
                    for k, v in st.items() if k.startswith(f"k{K}_s{SEED}_"))
        note = f"(EDIT=min, sized from canonical stats: {calls} calls x ~{mean_line:.0f}-tok lines)"
        if calls == 0:
            # R13 M4: no measured stats for this (K,SEED) -- the exact-prefix filter would silently
            # price $0 and void the quote-first gate. Extrapolate the calls-per-content-line rate
            # from every existing EDIT=min record instead, and say so loudly.
            rec = [v for k, v in st.items()
                   if k.startswith("k") and isinstance(v, dict) and "n_content" in v]
            tot = sum(v["n_content"] for v in rec)
            rate = (sum((v["n_content"] - v["tiers"].get("verbatim", 0)) + v["retried"] for v in rec)
                    / max(tot, 1))
            calls = int(round(rate * len(lines)))
            note = (f"(EDIT=min EXTRAPOLATED: no k{K}_s{SEED} stats; {rate:.2f} calls/line x "
                    f"{len(lines)} lines = {calls} calls)")
        tin = int(calls * (len(enc.encode(MIN_SYS + MIN_USR)) + mean_line * 1.5))
        tout = int(calls * mean_line * 1.2)
    else:
        tin = sum(len(enc.encode(RW_SYS + RW_USR.format(line=s))) for s in lines)
        tout = sum(len(enc.encode(s)) for s in lines)
        note = f"({len(lines)} content lines, +~30% retries not included)"
    c = tin / 1e6 * pin + tout / 1e6 * pout
    print(f"[cost] {DS} GEN={GEN}: in~{tin/1e3:.0f}k out~{tout/1e3:.0f}k tok -> ~${c:.2f} {note}, "
          f"embeddings ~${tout*2/1e6*0.02:.3f}")


def _rewrite_min(orig, OV, src_sh):
    """V6 minimal-edit pipeline (ELEMK_DESIGN.md V6 appendix). Clean lines ship VERBATIM; dirty lines
    get a minimal edit breaking exactly the member-overlapping runs, under five gates (lexical vs
    member pool ONLY, numeric, fidelity >= MINFID, punt-status preserved, length +-30%). Exhausted
    lines fall back to the V5 full-line rewrite (V5 gates, <=2 rounds) before dropping — a dropped
    line was V5's utility poison."""
    runs_of = {li: _hit_runs(orig[li], src_sh) for li in range(len(orig))}
    cur, tier, fail, best = {}, {}, {}, {}
    todo, route_rw = [], []
    for li in range(len(orig)):
        if not runs_of[li]:
            cur[li] = orig[li]
            tier[li] = "verbatim"
            continue
        # Dry-run amendments 2+4 (ELEMK_DESIGN.md): only lines whose runs cover ESSENTIALLY the whole
        # line (union cov > 0.90 -- every 6-word window must be broken, min-edit unsatisfiable) route
        # straight to the full-rewrite stage; other dirty lines attempt the min-edit first (dry run 1
        # showed ~half succeed there with changed_frac ~.2-.4, far below rewrite's ~.7).
        wl = len(re.findall(r"[a-z']+", orig[li].lower()))
        cov = sum(len(r.split()) for r in runs_of[li]) / max(wl, 1)
        if cov > 0.90:
            route_rw.append(li)
        else:
            todo.append(li)
            fail[li] = ("", None)
    retries = {li: 0 for li in range(len(orig))}
    # Reviewer MAJOR-3: mid-coverage lines (0.70 < cov <= 0.90) still need near-full rephrasing to
    # clear every overlapped window -- hold them to V5's fidelity thresholds, not MINFID's 0.90.
    fid_hi, fid_lo = {}, {}
    for li in todo:
        wl = len(re.findall(r"[a-z']+", orig[li].lower()))
        cov = sum(len(r.split()) for r in runs_of[li]) / max(wl, 1)
        fid_hi[li] = FID if cov > 0.70 else MINFID
        fid_lo[li] = 0.65 if cov > 0.70 else MINFID_FLOOR

    def _ask(sys_p, u, rnd=0):
        msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": u}]
        if SAMPLE is not None:      # R9 fresh-sampling path; salt=round per review M1
            return (sample_one(msgs, GEN, SAMPLE, temperature=0.3, max_tokens=400,
                               salt=f"r{rnd}") or "").strip()
        return (chat(msgs, model=GEN, temperature=0.3, max_tokens=400, extra=SAN_EXTRA) or "").strip()

    def _fb(li, u):
        mode, info = fail[li]
        if mode == "lex":
            shs, prev = info
            u += FB_LEX.format(sh=", ".join(f'"{s}"' for s in shs), prev=prev)
        elif mode == "num":
            toks, prev = info
            u += FB_NUM.format(toks=", ".join(sorted(toks)), prev=prev)
        elif mode == "fid":
            u += FB_FID.format(prev=info)
        elif mode == "punt_add":
            u += FB_PUNT_ADD.format(prev=info)
        elif mode == "punt_drop":
            u += FB_PUNT_DROP.format(prev=info)
        elif mode == "len":
            u += FB_LEN.format(prev=info)
        elif mode == "empty":
            u += FB_EMPTY
        return u

    # stage 1: minimal edits
    for rnd in range(1 + MAXRETRY):
        if not todo:
            break
        outs = de.pool(lambda li, _r=rnd: _ask(MIN_SYS, _fb(li, MIN_USR.format(
            runs="; ".join(f'"{r}"' for r in runs_of[li][:6]), line=orig[li])), _r), todo)
        news = []
        for o in outs:
            m = _PREF.match(o)
            news.append(o[len(m.group(1)):].strip() if m else o.strip())
        NV = np.stack(EB.embed([n or "x" for n in news]))
        nxt = []
        for li, new, nv in zip(todo, news, NV):
            hit = _shingles(new) & src_sh                 # v6: orig-line overlap is ALLOWED
            misn = _nums(orig[li]) - _nums(new)
            fid = float(nv @ OV[li])
            wo, wn = len(orig[li].split()), len(new.split())
            lenok = abs(wn - wo) <= max(3, int(0.30 * wo))
            po, pn = bool(REQ.search(orig[li])), bool(REQ.search(new))
            if new and not hit and not misn and po == pn and lenok and fid >= fid_hi[li]:
                cur[li] = new
                tier[li] = "strict"
                continue
            retries[li] += 1
            if not new:
                fail[li] = ("empty", None)
            elif hit:
                fail[li] = ("lex", (sorted(hit)[:3], new))
            elif misn:
                fail[li] = ("num", (misn, new))
            elif po != pn:
                fail[li] = ("punt_add" if pn else "punt_drop", new)
            elif not lenok:
                fail[li] = ("len", new)
            else:
                fail[li] = ("fid", new)
                if fid > best.get(li, (0.0, ""))[0]:
                    best[li] = (fid, new)
            nxt.append(li)
        todo = nxt
    still = []
    for li in todo:
        if li in best and best[li][0] >= fid_lo[li]:
            cur[li] = best[li][1]
            tier[li] = "relaxed"
        else:
            still.append(li)

    # stage 2: full-line rewrite (V5 gates: lexical vs orig+members, numeric, FID/FID_FLOOR)
    # + the punt-status gate (reviewer BLOCKER-1: the lines most likely to land here are exactly the
    # ones whose violating run IS the punt phrase -- without the gate the MAD punt channel reopens).
    # Serves BOTH the coverage-routed lines (tier "rewrite", 1+MAXRETRY rounds by design) and the
    # min-edit leftovers (tier "fallback").
    stage2 = route_rw + still
    fail2 = {li: ("", None) for li in stage2}
    best2 = {}
    todo = stage2
    for rnd in range(1 + MAXRETRY):
        if not todo:
            break
        def one2(li):
            u = RW_USR.format(line=orig[li])
            mode, info = fail2[li]
            if mode == "lex":
                shs, prev = info
                u += FB_LEX.format(sh=", ".join(f'"{s}"' for s in shs), prev=prev)
            elif mode == "num":
                toks, prev = info
                u += FB_NUM.format(toks=", ".join(sorted(toks)), prev=prev)
            elif mode == "fid":
                u += FB_FID.format(prev=info)
            elif mode == "punt_add":
                u += FB_PUNT_ADD.format(prev=info)
            elif mode == "punt_drop":
                u += FB_PUNT_DROP.format(prev=info)
            elif mode == "empty":
                u += FB_EMPTY
            # stage-2 rounds continue the salt sequence so a fallback line's round-r message can
            # never collide with its own stage-1 round-r message (different prompt template anyway;
            # offset keeps the invariant obvious)
            return _ask(RW_SYS, u, 1 + MAXRETRY + rnd)
        outs = de.pool(one2, todo)
        news = []
        for o in outs:
            m = _PREF.match(o)
            news.append(o[len(m.group(1)):].strip() if m else o.strip())
        NV = np.stack(EB.embed([n or "x" for n in news]))
        nxt = []
        for li, new, nv in zip(todo, news, NV):
            banned = _shingles(orig[li]) | src_sh
            hit = _shingles(new) & banned
            misn = _nums(orig[li]) - _nums(new)
            fid = float(nv @ OV[li])
            po, pn = bool(REQ.search(orig[li])), bool(REQ.search(new))
            if new and not hit and not misn and po == pn and fid >= FID:
                cur[li] = new
                tier[li] = "rewrite" if li in route_rw else "fallback"
                continue
            retries[li] += 1
            if not new:
                fail2[li] = ("empty", None)
            elif hit:
                fail2[li] = ("lex", (sorted(hit)[:3], new))
            elif misn:
                fail2[li] = ("num", (misn, new))
            elif po != pn:
                fail2[li] = ("punt_add" if pn else "punt_drop", new)
            else:
                fail2[li] = ("fid", new)
                if fid > best2.get(li, (0.0, ""))[0]:
                    best2[li] = (fid, new)
            nxt.append(li)
        todo = nxt
    dropped = set()
    route_set = set(route_rw)
    for li in todo:
        if li in best2 and best2[li][0] >= 0.65:
            cur[li] = best2[li][1]
            tier[li] = "rewrite_relaxed" if li in route_set else "fallback_relaxed"
        else:
            dropped.add(li)
            fail[li] = fail2[li]
    return cur, tier, retries, dropped, fail, runs_of


def stage_build():
    import hashlib
    aggro, byc = EB.load_clusters()
    cache = json.loads(EB.ELEMS_P.read_text(encoding="utf-8"))
    clus = EB.cluster_elements(byc, cache)
    matched = _load_support()
    # support is trustworthy only if the pairs file was adjudicated under THIS partition.
    _prefix = f"k{K}_s{SEED}_"
    pairs_ok = bool(matched) and all(k[0].startswith(_prefix) for k in matched)
    cards = json.loads(NE_P.read_text(encoding="utf-8"))
    out_cards = json.loads(OUT_P.read_text(encoding="utf-8")) if OUT_P.exists() else {}
    audit = json.loads(AUD_P.read_text(encoding="utf-8")) if AUD_P.exists() else {}
    stats = json.loads(STAT_P.read_text(encoding="utf-8")) if STAT_P.exists() else {}
    # Reviewer MINOR-14: config-stamp the artifacts; a resume under changed gates/prompt must not
    # silently mix old and new clusters.
    cfg = {"FID": FID, "FID_FLOOR": 0.65, "MAXRETRY": MAXRETRY, "SHN": SHN,
           "prompt_sha1": hashlib.sha1((RW_SYS + RW_USR).encode()).hexdigest()[:12]}
    if EDIT == "min":
        cfg.update({"EDIT": "min", "MINFID": MINFID, "MINFID_FLOOR": MINFID_FLOOR,
                    "min_prompt_sha1": hashlib.sha1((MIN_SYS + MIN_USR).encode()).hexdigest()[:12],
                    "req_sha1": hashlib.sha1(REQ.pattern.encode()).hexdigest()[:12]})
    if GEN != "deepseek-chat":      # R6 MAJOR-3: non-default generator self-documents in the sidecar;
        cfg["GEN"] = GEN            # conditional so canonical resumes keep their stamped config intact
    if SAMPLE is not None:          # R9: a resume under a different sample index must fail loudly
        cfg["SAN_SAMPLE"] = SAMPLE
    if _SG is not None:             # R6f: guided arms self-document (prompt_sha1 already differs)
        cfg["GUIDE"] = True
    if "_config" in stats:
        assert stats["_config"] == cfg, (f"resume under different config: {stats['_config']} vs {cfg} "
                                         f"-- delete {OUT_P.name}/{AUD_P.name}/{STAT_P.name} to rebuild")
    stats["_config"] = cfg
    if not pairs_ok:
        stats["_support_note"] = ("adj/pairs not adjudicated for this partition -> audit 'support' "
                                  "= null for these clusters (card routing unaffected; R1 amendment ②)")

    only = os.environ.get("ONLY", "")
    for ck in sorted(clus):
        texts, owners = clus[ck]
        if not texts or ck not in cards:
            continue
        if only and only not in ck:
            continue
        if ck in out_cards and ck in audit:
            continue
        V = np.stack(EB.embed(texts))
        sup = ([1 + len(matched.get((ck, i), ())) for i in range(len(texts))]
               if pairs_ok else [None] * len(texts))
        mem = sorted(set(owners))
        oarr = np.array([mem.index(o) for o in owners])
        # source shingle pool: cluster member elements + member aggro cards (privacy unit = cluster)
        src_sh = set()
        for t in texts:
            src_sh |= _shingles(t)
        for m in mem:
            src_sh |= _shingles(aggro[m])

        raw_lines = cards[ck].splitlines()
        parsed = [(_split_line(ln)) for ln in raw_lines]
        content_idx = [i for i, (_p, s) in enumerate(parsed) if _is_content(s)]
        orig = [parsed[i][1] for i in content_idx]
        OV = np.stack(EB.embed(orig)) if orig else np.zeros((0, 1))
        S = OV @ V.T if orig else None

        # provenance of the ORIGINAL lines (the audit sidecar)
        rows = []
        for li, i in enumerate(content_idx):
            j = int(np.argmax(S[li]))
            per = [S[li, oarr == m].max() if (oarr == m).any() else 0.0 for m in range(len(mem))]
            rows.append({"line": i, "top_ei": j, "top_owner": owners[j], "cos": round(float(S[li, j]), 4),
                         "support": sup[j], "n_members_55": int(sum(1 for v in per if v >= 0.55)),
                         "unattributed": bool(S[li, j] < 0.55)})

        runs_of = {}
        if EDIT == "min":
            # V6 (design appendix #124): minimal-edit pipeline; verbatim-clean lines untouched.
            cur, tier, retries, dropped, fail, runs_of = _rewrite_min(orig, OV, src_sh)
        else:
            # V5: rewrite rounds with deterministic gates: lexical (6-shingle vs orig line + member
            # text), numeric preservation (reviewer MAJOR-5), fidelity (cos >= FID). Empty gets its
            # own feedback (reviewer MINOR-13); lexical feedback names top 3 (reviewer MAJOR-8).
            cur = {li: None for li in range(len(orig))}          # li -> accepted rewrite
            fail = {li: ("", None) for li in range(len(orig))}   # li -> (mode, info)
            best = {}                                            # li -> (fid, new): lex+num-clean but < FID
            tier = {}                                            # li -> "strict" | "relaxed"
            todo = list(range(len(orig)))
            retries = {li: 0 for li in todo}
            for rnd in range(1 + MAXRETRY):
                if not todo:
                    break

                def one(li):
                    u = RW_USR.format(line=orig[li])
                    mode, info = fail[li]
                    if mode == "lex":
                        shs, prev = info
                        u += FB_LEX.format(sh=", ".join(f'"{s}"' for s in shs), prev=prev)
                    elif mode == "num":
                        toks, prev = info
                        u += FB_NUM.format(toks=", ".join(sorted(toks)), prev=prev)
                    elif mode == "fid":
                        u += FB_FID.format(prev=info)
                    elif mode == "empty":
                        u += FB_EMPTY
                    return (chat([{"role": "system", "content": RW_SYS}, {"role": "user", "content": u}],
                                 model=GEN, temperature=0.3, max_tokens=400, extra=SAN_EXTRA) or "").strip()

                outs = de.pool(one, todo)
                nxt = []
                news = []
                for o in outs:
                    m = _PREF.match(o)
                    news.append(o[len(m.group(1)):].strip() if m else o.strip())
                NV = np.stack(EB.embed([n or "x" for n in news]))
                for li, new, nv in zip(todo, news, NV):
                    banned = _shingles(orig[li]) | src_sh
                    hit = _shingles(new) & banned
                    misn = _nums(orig[li]) - _nums(new)
                    fid = float(nv @ OV[li])
                    if new and not hit and not misn and fid >= FID:
                        cur[li] = new
                        tier[li] = "strict"
                        continue
                    retries[li] += 1
                    if not new:
                        fail[li] = ("empty", None)
                    elif hit:
                        fail[li] = ("lex", (sorted(hit)[:3], new))
                    elif misn:
                        fail[li] = ("num", (misn, new))
                    else:
                        fail[li] = ("fid", new)
                        if fid > best.get(li, (0.0, ""))[0]:
                            best[li] = (fid, new)        # lex- and num-clean, only the cosine fell short
                    nxt.append(li)
                todo = nxt
            # graduated fallback (dry-run amendment, pre-G2): dropping a line REMOVES content, which
            # would confound P1 ("v5 loses because lines are missing, not because wording changed").
            # The lexical gate is the non-negotiable privacy claim; fidelity is our own quality guard —
            # accept the best lex+num-clean attempt at fid >= FID_FLOOR, tagged "relaxed"; else drop.
            FID_FLOOR = 0.65
            dropped = set()
            for li in todo:
                if li in best and best[li][0] >= FID_FLOOR:
                    cur[li] = best[li][1]
                    tier[li] = "relaxed"
                else:
                    dropped.add(li)

        # assemble: same order/structure; non-content lines verbatim; dropped lines omitted
        li_of = {i: li for li, i in enumerate(content_idx)}
        out_lines = []
        for i, ln in enumerate(raw_lines):
            if i not in li_of:
                out_lines.append(ln)
            elif li_of[i] in dropped:
                continue
            else:
                out_lines.append(parsed[i][0] + cur[li_of[i]])
        out_cards[ck] = "\n".join(out_lines)

        _cap = re.compile(r"\b[A-Z][A-Za-z0-9+.#-]{2,}\b")
        import difflib
        for li, r in enumerate(rows):
            r.update({"retries": retries[li], "dropped": li in dropped})
            if EDIT == "min":
                r["tier"] = tier.get(li, "dropped")
                r["n_runs"] = len(runs_of.get(li, []))
                if cur.get(li) is not None:
                    r["changed_frac"] = round(1 - difflib.SequenceMatcher(
                        None, orig[li].split(), cur[li].split()).ratio(), 3)
            if li in dropped:
                r["final_fail"] = fail[li][0]
            elif EDIT == "full" and tier.get(li) == "relaxed":
                r["fid_tier"] = "relaxed"
                r["fid"] = round(best[li][0], 4)
            if cur.get(li) is not None:                  # capitalized-name survival: audit only, no gate
                toks = orig[li].split()
                caps = set(_cap.findall(" ".join(toks[1:]))) if len(toks) > 1 else set()
                miss = sorted(t for t in caps if t not in cur[li])
                if miss:
                    r["caps_missing"] = miss
        audit[ck] = rows
        w_ne = len(re.findall(r"\w+", cards[ck]))
        w_v5 = len(re.findall(r"\w+", out_cards[ck]))
        stats[ck] = {"n_lines": len(raw_lines), "n_content": len(orig), "n_dropped": len(dropped),
                     "retried": sum(1 for v in retries.values() if v),
                     "caps_missing_lines": sum(1 for r in rows if r.get("caps_missing")),
                     "words_ne": w_ne, "words_v5": w_v5}
        if EDIT == "min":
            from collections import Counter
            stats[ck]["tiers"] = dict(Counter(tier.get(li, "dropped") for li in range(len(orig))))
            cf = [r["changed_frac"] for r in rows if "changed_frac" in r]
            stats[ck]["mean_changed_frac"] = round(float(np.mean(cf)), 3) if cf else 0.0
        print(f"  {ck}: {len(orig)} content lines, retried {stats[ck]['retried']}, "
              f"dropped {len(dropped)}, words {w_ne}->{w_v5}"
              + (f", tiers {stats[ck]['tiers']}, changed {stats[ck]['mean_changed_frac']:.0%}"
                 if EDIT == "min" else ""), flush=True)
        OUT_P.write_text(json.dumps(out_cards, ensure_ascii=False, indent=1), encoding="utf-8")
        AUD_P.write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")
        STAT_P.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    cs = {k: v for k, v in stats.items() if not k.startswith("_")}
    STAT_P.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    nc = sum(s["n_content"] for s in cs.values())
    nd = sum(s["n_dropped"] for s in cs.values())
    dw = [s["words_v5"] / s["words_ne"] for s in cs.values()]
    print(f"[build] {DS}: {len(cs)} cards -> {OUT_P.name}   content lines {nc}, dropped {nd} "
          f"({100*nd/max(nc,1):.1f}%)   words v5/ne mean {np.mean(dw):.2f}   "
          f"caps-missing lines {sum(s['caps_missing_lines'] for s in cs.values())}")
    if nd / max(nc, 1) > 0.05:
        print("  *** drop rate > 5% — G-lex FAIL-level anomaly per design, stop and inspect ***")


if __name__ == "__main__":
    {"cost": stage_cost, "build": stage_build}[STAGE]()
