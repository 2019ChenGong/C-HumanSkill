"""Deterministic line-ablation arms for the drop-gate threshold experiment (prereg
`results/DROPGATE_THRESHOLD_PREREG.md`, 2026-08-05).

WHAT IT DOES — takes an ALREADY-BUILT v6 card set and deletes additional content lines to hit a
target total drop rate. It does NOT rebuild or re-sanitize anything: rebuilding would change both
"which lines are gone" AND "how the surviving lines were reworded", and the two factors could not be
separated. Ablation moves only the first.

  $0 — no LLM call, no embedding call. The per-line member-similarity used for TARGETED ablation is
  the `cos` field the sanitizer already wrote into the audit sidecar (max cosine of the ORIGINAL line
  against all member elements), so nothing is recomputed and nothing can drift.

TWO MODES (prereg §3.2)
  targeted  delete surviving lines in DESCENDING `cos` — mimics the pipeline, whose drops are exactly
            the lines too close to member text to reword. This is the realistic arm.
  random    delete an equal number at random (fixed seed) — the mechanism control. The targeted-minus-
            random gap IS the "drops are selective" effect, and is reportable on its own.

RECONSTRUCTION IS EXACT, NOT HEURISTIC: the shipped v6 card is
`[ne raw line]` for non-content lines and `[prefix + rewrite]` for surviving content lines, in raw
order, with dropped lines omitted (v5_sanitize.py assemble step). We walk the ne card and the v6 card
in lockstep to recover the content-line -> v6-line-position map, and ASSERT every non-content line is
byte-identical between the two. Any mismatch aborts — no silent re-parse.

  DATASET=mad K=8 SEED=0 RATE=0.10 MODE=targeted STAGE=cost python -P scripts/dropgate_ablate.py
  DATASET=mad K=8 SEED=0 RATE=0.10 MODE=targeted STAGE=build python -P scripts/dropgate_ablate.py
Out: data/<ds>/<cardbase>__abl{RATE}{t|r}.json  (+ _stats.json, same schema as the sanitizer's so
     the existing drop-rate readers work unchanged). Canonical files are never touched.
"""
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GROUP", "random")
os.environ.setdefault("DATASET", "mad")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DS = os.environ["DATASET"]
K = int(os.environ.get("K", 8))
SEED = int(os.environ.get("SEED", 0))
RATE = float(os.environ["RATE"])
MODE = os.environ.get("MODE", "targeted")
STAGE = os.environ.get("STAGE", "cost")
RSEED = int(os.environ.get("RSEED", 0))
assert MODE in ("targeted", "random"), MODE
assert 0 < RATE < 1, RATE

CARDBASE = {"mad": "cmd_shared_cards_mad", "enron": "cmd_shared_cards", "cv": "cmd_shared_cards_cv"}[DS]
SE = ROOT / {"mad": "data/20mad", "enron": "data/enron", "cv": "data/se"}[DS]
NE_P = SE / f"{CARDBASE}__neutral_fixed.json"
V6_P = SE / f"{CARDBASE}__v6min.json"
AUD_P = SE / f"{CARDBASE}__v6min_audit.json"
TAG = f"abl{int(RATE*100):02d}{'t' if MODE == 'targeted' else 'r'}"
OUT_P = SE / f"{CARDBASE}__{TAG}.json"
STAT_P = SE / f"{CARDBASE}__{TAG}_stats.json"

_PREF = re.compile(r"^(\s*(?:[-•]\s*|\*\s+|\d+[.)]\s*|#+\s+)+)")


def _split_line(ln):
    m = _PREF.match(ln)
    pref = m.group(1) if m else ""
    return pref, ln[len(pref):].strip()


def content_positions(ne_card, v6_card, rows):
    """content-line index li -> its position in v6_card.splitlines(); None if already dropped.

    Walks both cards in lockstep against the audit's `line` (= raw index in the ne card).
    Asserts byte-identity on every non-content line, so a drifted parser cannot pass silently.
    """
    raw = ne_card.splitlines()
    out = v6_card.splitlines()
    li_of_raw = {r["line"]: li for li, r in enumerate(rows)}
    dropped_li = {li for li, r in enumerate(rows) if r.get("dropped")}
    pos, j = {}, 0
    for i, ln in enumerate(raw):
        if i not in li_of_raw:                       # non-content: shipped verbatim
            assert j < len(out) and out[j] == ln, \
                f"non-content line {i} differs between ne and v6 -- structure drifted, ABORT"
            j += 1
            continue
        li = li_of_raw[i]
        if li in dropped_li:                         # already dropped by the sanitizer
            pos[li] = None
            continue
        assert j < len(out), f"ran out of v6 lines at raw {i} -- ABORT"
        pos[li] = j
        j += 1
    assert j == len(out), f"v6 has {len(out)-j} unconsumed trailing lines -- ABORT"
    return pos, dropped_li


def main():
    ne = json.loads(NE_P.read_text(encoding="utf-8"))
    v6 = json.loads(V6_P.read_text(encoding="utf-8"))
    aud = json.loads(AUD_P.read_text(encoding="utf-8"))
    keys = sorted(k for k in v6 if k.startswith(f"k{K}_s{SEED}_"))
    assert keys, f"no k{K}_s{SEED}_* cards in {V6_P.name}"

    rng = random.Random(RSEED)
    out_cards, stats = {}, {}
    tot_c = tot_d0 = tot_d1 = 0
    plan_preview = []
    for ck in keys:
        rows = aud[ck]
        pos, dropped_li = content_positions(ne[ck], v6[ck], rows)
        n_content = len(rows)
        surv = [li for li in range(n_content) if li not in dropped_li]
        want = int(round(RATE * n_content))                  # target TOTAL dropped for this card
        extra = max(0, want - len(dropped_li))
        extra = min(extra, len(surv) - 1)                    # never empty a card
        if MODE == "targeted":
            order = sorted(surv, key=lambda li: (-float(rows[li].get("cos", 0.0)), li))
        else:
            order = surv[:]; rng.shuffle(order)
        kill = set(order[:extra])

        lines = v6[ck].splitlines()
        keep = [ln for p, ln in enumerate(lines) if p not in {pos[li] for li in kill}]
        out_cards[ck] = "\n".join(keep)
        nd = len(dropped_li) + len(kill)
        stats[ck] = {"n_lines": len(ne[ck].splitlines()), "n_content": n_content, "n_dropped": nd,
                     "n_dropped_pipeline": len(dropped_li), "n_dropped_ablated": len(kill),
                     "abl_mode": MODE, "abl_rate_target": RATE,
                     "abl_cos_killed": sorted((round(float(rows[li].get("cos", 0.0)), 4) for li in kill),
                                              reverse=True)}
        tot_c += n_content; tot_d0 += len(dropped_li); tot_d1 += len(kill)
        if len(plan_preview) < 3:
            plan_preview.append((ck, n_content, len(dropped_li), len(kill),
                                 stats[ck]["abl_cos_killed"][:5]))

    rate0 = 100 * tot_d0 / max(1, tot_c)
    rate1 = 100 * (tot_d0 + tot_d1) / max(1, tot_c)
    print(f"=== ablate {DS} k{K}_s{SEED}  MODE={MODE}  RATE target {RATE:.0%}  -> {TAG} ===")
    print(f"  {len(keys)} cards, {tot_c} content lines")
    print(f"  pipeline drops {tot_d0} ({rate0:.1f}%)  + ablated {tot_d1}  = {tot_d0+tot_d1} ({rate1:.1f}%)")
    print(f"  LLM calls: 0   embedding calls: 0   COST: $0.00  (cos read from the existing audit)")
    print("  preview (card, n_content, pipeline_drop, ablated, top killed cos):")
    for p in plan_preview:
        print(f"    {p[0]}: {p[1]} lines, {p[2]} + {p[3]} killed, cos {p[4]}")
    if STAGE != "build":
        print("\nDRY RUN — nothing written. Re-run with STAGE=build.")
        return
    assert not OUT_P.exists(), f"{OUT_P.name} exists -- delete it to rebuild (no silent overwrite)"
    OUT_P.write_text(json.dumps(out_cards, ensure_ascii=False, indent=1), encoding="utf-8")
    STAT_P.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    back = json.loads(OUT_P.read_text(encoding="utf-8"))
    assert set(back) == set(keys) and all(back[k].strip() for k in keys), "readback failed"
    print(f"\nwrote {OUT_P.name} ({len(back)} cards) + {STAT_P.name}")


if __name__ == "__main__":
    main()
