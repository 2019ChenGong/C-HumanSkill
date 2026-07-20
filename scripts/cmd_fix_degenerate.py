"""Fix the neutral-CMD synthesis DEGENERATE tail (worst-case member exposure). ~6-12% of neutral CMD cards were found
to be near-verbatim COPIES of one dominant member's card (max-member-cos→1.0, longest verbatim run →86-100% of the
card) — the synth model took the lazy route and reproduced a single member instead of abstracting the common
structure. This EXPOSES that one member fully, even though the card is anonymous to the other k-1. concat never does
this (union-summary spreads copying across all members; worst single-member run ≤14% of card).

Fix = a synth prompt with an explicit ANTI-COPY GUARD ("must be a genuine synthesis; no single member reproducible"),
applied ONLY to cards that test degenerate. Non-degenerate cards are copied through unchanged. Retry with rising
temperature until the re-synth passes the guard (or MAX_TRIES). Writes an ISOLATED *_fixed.json — does NOT touch the
originals, so existing 2AFC/utility runs are unaffected until we validate the worst-case improvement.

DEGENERATE if max_m cosine(card,member) >= COS_TH  OR  longest verbatim run >= RUN_FRAC * card_len.

  DATASET=mad KS=4,8 SEEDS=0 python scripts/cmd_fix_degenerate.py
"""
import os
import re
import sys
import json
import difflib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GROUP", "random")
os.environ.setdefault("DATASET", "mad")
import deid_enron as de   # noqa: E402
import cmd_gate as CG     # noqa: E402
from src.llm import chat  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DS = os.environ.get("DATASET", "mad")
KS = [int(x) for x in os.environ.get("KS", "4,8").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
GEN = "deepseek-chat"
COS_TH = float(os.environ.get("COS_TH", 0.85))
RUN_FRAC = float(os.environ.get("RUN_FRAC", 0.40))
MAX_TRIES = int(os.environ.get("MAX_TRIES", 4))
DDIR = {"mad": "20mad", "enron": "enron", "cv": "se"}[DS]
# SRCNAME/DSTNAME env-overridable so an ISOLATED k-sweep file (e.g. k12) can be fixed without touching the canonical
# k8 neutral cards (the default). Default = canonical neutral -> *_fixed.json (unchanged behavior).
_SRCNAME = os.environ.get("SRCNAME") or {"mad": "cmd_shared_cards_mad__neutral.json",
                                         "enron": "cmd_shared_cards__neutral.json",
                                         "cv": "cmd_shared_cards_cv__neutral.json"}[DS]
SRC = ROOT / "data" / DDIR / _SRCNAME
DST = ROOT / "data" / DDIR / os.environ.get("DSTNAME", SRC.stem + "_fixed.json")

_SYS_KEEP = ("You distill ONE shared skill card capturing the full shared working knowledge and practices common to "
             "several colleagues, removing only what identifies any single one of them.")


def synth_anticopy(member_cards, temp):
    body = "\n\n---\n\n".join(member_cards)
    msg = [{"role": "system", "content": _SYS_KEEP},
           {"role": "user", "content": f"Skill cards from several colleagues:\n\n{body}\n\nWrite ONE shared skill card "
            "that captures everything they have in COMMON — their shared knowledge, working approaches, and practices "
            "— while removing any phrasing or detail unique to any single person.\n\n"
            "CRITICAL — this must be a GENUINE SYNTHESIS, not a copy: do NOT reproduce, quote, or lightly paraphrase "
            "any single colleague's card. No sentence, list, or passage may be traceable to one person's card alone. "
            "Abstract and re-express the COMMON structure in neutral wording so that NO single contributor could be "
            "identified as its source. If one card is longer or more detailed than the others, do NOT let it dominate "
            "— include only what it SHARES with the rest.\n\n"
            "Preserve the concrete shared substance; do not compress into generic platitudes. Keep it comparable in "
            "length to ONE input card. It must read as if it could belong to any of them equally. Output ONLY the "
            "shared card."}]
    return chat(msg, model=GEN, temperature=temp, max_tokens=1300) or ""


def toks(t):
    return re.findall(r"\w+", t.lower())


def degeneracy(card, member_texts):
    """-> (max_member_cos, longest_run_frac, worst_member_idx)."""
    cvec = de._content_vec(card); cw = toks(card); n = max(len(cw), 1)
    best_cos, best_run = 0.0, 0
    for mt in member_texts:
        cos = de._cosine(cvec, de._content_vec(mt))
        run = difflib.SequenceMatcher(None, cw, toks(mt), autojunk=False).find_longest_match(
            0, len(cw), 0, len(toks(mt))).size
        best_cos = max(best_cos, cos); best_run = max(best_run, run)
    return best_cos, best_run / n


def is_degenerate(cos, runfrac):
    return cos >= COS_TH or runfrac >= RUN_FRAC


def main():
    _d, authors, _n, aggro, _r, _rt = CG.load()
    src = json.loads(SRC.read_text(encoding="utf-8"))
    out = json.loads(DST.read_text(encoding="utf-8")) if DST.exists() else dict(src)  # start from a full copy
    fixed, checked, failed = [], 0, []
    for k in KS:
        for s in SEEDS:
            grp, byc = CG.make_groups(aggro, authors, k, s)
            for cid, mem in byc.items():
                if len(mem) < k:
                    continue
                ck = f"k{k}_s{s}_{cid}"
                if ck not in src:
                    continue
                checked += 1
                mtexts = [aggro[a] for a in mem]
                cos, rf = degeneracy(src[ck], mtexts)
                if not is_degenerate(cos, rf):
                    out[ck] = src[ck]        # keep original
                    continue
                # re-synth with the anti-copy guard, rising temperature until it passes
                print(f"  {ck}: DEGENERATE (cos={cos:.3f} run={rf:.0%}) -> re-synth", flush=True)
                best = None
                for t in range(MAX_TRIES):
                    card = synth_anticopy(mtexts, temp=0.3 + 0.2 * t)
                    c2, r2 = degeneracy(card, mtexts)
                    if best is None or (c2, r2) < (best[1], best[2]):
                        best = (card, c2, r2)
                    if not is_degenerate(c2, r2):
                        print(f"     try{t} temp={0.3+0.2*t:.1f}: OK cos={c2:.3f} run={r2:.0%}", flush=True)
                        break
                    print(f"     try{t} temp={0.3+0.2*t:.1f}: still cos={c2:.3f} run={r2:.0%}", flush=True)
                out[ck] = best[0]
                fixed.append((ck, round(cos, 3), round(rf, 2), round(best[1], 3), round(best[2], 2)))
                if is_degenerate(best[1], best[2]):
                    failed.append(ck)
    DST.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\nchecked {checked} cards | re-synthesized {len(fixed)} degenerate | {len(failed)} still degenerate after {MAX_TRIES} tries")
    for ck, c0, r0, c1, r1 in fixed:
        flag = "  <-- STILL DEGENERATE" if ck in failed else ""
        print(f"  {ck}: cos {c0}->{c1}  run {r0:.0%}->{r1:.0%}{flag}")
    print(f"\nsaved -> {DST.relative_to(ROOT)}  (originals untouched)")


if __name__ == "__main__":
    main()
