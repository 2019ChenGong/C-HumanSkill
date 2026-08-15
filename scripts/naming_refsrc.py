"""Single source of truth for the naming ruler's CANDIDATE-SAMPLE source (R20 / #161).

The shipped naming wave hands the attacker a 250-char writing sample per candidate that is
LEAK-DISJOINT from the card-building corpus.  R20 adds conditions where the attacker holds
the card-building corpus instead -- the attacker our own scenario section describes.

Both `naming_refsrc_diag.py` (Stage 0, offline) and `naming_export.py` (the wave) import
THIS module, so the two can never drift.  Duplicating these constructions in two files is
exactly the desync hazard already fixed twice in this repo (cv_t2_verify RISK 5, and the
_shingles literal), so do not inline them anywhere.

CONDITIONS
----------
  held         CG.load()'s `ref`, returned UNTOUCHED         <- DEFAULT, byte-identical to legacy
  heldfull     same held-out text, untruncated               <- budget control
  cardmatched  card corpus, MATCHED unit count, untruncated  <- R20 MAIN condition
  cardfull     card corpus, all of it                        <- ceiling (diagnostic only)

`held` is an identity pass-through by construction, NOT a re-derivation: that is what makes
"REFSRC unset => byte-identical export" a guarantee rather than a test result.

UNIT-COUNT MATCHING (the point of `cardmatched`: differ from `held` only in WHICH text)
  Enron  held = docs[a][12]        (1 held-out doc)   | card = docs[a][0]        (1 card doc)
  MAD    held = comments[18:24]    (6 joined)         | card = comments[0:6]     (6 joined)
  CV     *** CANNOT be unit-matched *** -- CV's pool file stores only {ref, raw}, so the card
         corpus comes from data/se/cv_card_docs.json (the byte-verified 77/77 rebuild from the
         D1-VMIA CV backfill) and we CHAR-BUDGET match instead.  Disclosed in the prereg §7.1;
         CV's provenance delta is the least comparable of the three datasets.
"""
import json
from pathlib import Path

import cmd_gate as CG

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("held", "heldfull", "cardmatched", "cardfull")
CV_CARD_DOCS = ROOT / "data" / "se" / "cv_card_docs.json"


def _W(s):
    return CG.WS.sub(" ", s)


def _cv_card_docs(authors):
    if not CV_CARD_DOCS.exists():
        raise SystemExit(
            f"[FATAL] CV card corpus missing: {CV_CARD_DOCS}\n"
            "  Rebuild with scripts/cv_t2_verify.py (needs data/se_raw/_stats/Posts.xml).")
    cd = json.loads(CV_CARD_DOCS.read_text(encoding="utf-8"))
    miss = [a for a in authors if a not in cd]
    if miss:
        raise SystemExit(f"[FATAL] {len(miss)} CV authors absent from cv_card_docs.json: {miss[:5]}")
    return cd


def build(ds, docs, authors, ref_held, cond):
    """Return {author: candidate-sample text} for `cond`.

    `docs` is slot 0 of CG.load(): the doc dict (Enron) or the pool dict (MAD/CV).
    `ref_held` is slot 4 of CG.load(), returned as-is when cond == 'held'.
    """
    if cond not in CONDITIONS:
        raise SystemExit(f"[FATAL] REFSRC={cond!r} not in {CONDITIONS}")
    if cond == "held":
        return ref_held                      # identity: byte-identity is structural, not tested

    if ds == "enron":
        held_full = {a: _W(docs[a][CG.N_TRAIN]["text"]) for a in authors}
        card_m = {a: _W(docs[a][0]["text"]) for a in authors}
        card_full = {a: _W(" ".join(d["text"] for d in docs[a][:CG.N_TRAIN])) for a in authors}
    elif ds == "mad":
        t, r = CG.MAD_TRAIN, CG.N_REF
        held_full = {d: _W(" || ".join(docs[d]["card_comments"][t:t + r])) for d in authors}
        card_m = {d: _W(" || ".join(docs[d]["card_comments"][0:r])) for d in authors}
        card_full = {d: _W(" || ".join(docs[d]["card_comments"][:CG.N_TRAIN])) for d in authors}
    elif ds == "cv":
        cd = _cv_card_docs(authors)
        held_full = {a: _W(docs[a]["ref"]) for a in authors}
        card_m = {a: _W(cd[a])[:len(held_full[a])] for a in authors}   # char-budget match; see docstring
        card_full = {a: _W(cd[a]) for a in authors}
    else:
        raise SystemExit(f"[FATAL] unsupported dataset {ds!r} (R20 covers enron/mad/cv)")

    out = {"heldfull": held_full, "cardmatched": card_m, "cardfull": card_full}[cond]

    empty = [a for a in authors if not out.get(a, "").strip()]
    if empty:
        raise SystemExit(f"[FATAL] cond={cond}: {len(empty)} authors have empty text: {empty[:5]}")
    if cond.startswith("card"):
        same = [a for a in authors if out[a] == ref_held[a]]
        if len(same) > 0.05 * len(authors):
            raise SystemExit(f"[FATAL] cond={cond}: {len(same)}/{len(authors)} authors identical to the "
                             "held-out ref -- the conditions are not distinct.")
    return out


def provenance(ds, cond):
    """One-line human-readable description, written into _config.json and the Stage 0 JSON."""
    if cond == "held":
        return "held-out text [:250] (canonical, leak-disjoint from the card corpus)"
    unit = {"enron": "docs[a][12] (1 doc) vs docs[a][0] (1 doc)",
            "mad": f"comments[{CG.MAD_TRAIN}:{CG.MAD_TRAIN + CG.N_REF}] vs comments[0:{CG.N_REF}] "
                   f"({CG.N_REF} joined each)",
            "cv": "pool[a]['ref'] vs head of cv_card_docs[a] (CHAR-BUDGET matched only -- CV cannot "
                  "be unit-count matched; least comparable of the three)"}[ds]
    return f"{cond}: {unit}"
