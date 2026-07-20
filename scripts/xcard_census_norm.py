"""Regenerate the length-normalized verbatim census (`xcard_census_normalized.json`) — closes provenance gap G2.

The census that shows concat carries ~5-6x the CMD excess verbatim overlap (per 100 words) previously had NO
generating script — it was a hand aggregation of the raw per-method census files. This script rebuilds it
reproducibly from:
  - card caches data/20mad/{cmd_shared_cards_mad, cmd_concat_cards_mad}.json  -> mean card length (\\w+ tokens)
  - raw census  results/mad/_xcard_{method}_census_k{k}.json                  -> matched POS/NEG shared-6gram means
Word count = re.findall(r"\\w+", text); averaged over ALL k{k}_s* cards present in that method's cache
(cmd has seeds 0,1,2; concat's k8 cache only has seeds 0,1 — use whatever exists). Verified to reproduce the
canonical values exactly: mean_words cmd 284.0/316.3, concat 610.9/608.8; matched-POS 6grams cmd 2.29/1.55,
concat 25.84/7.66; per-100w excess cmd 0.69/0.20, concat 4.14/1.10.

  python scripts/xcard_census_norm.py
"""
import os
import re
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "mad"
CACHES = {"cmd": ROOT / "data/20mad/cmd_shared_cards_mad.json",
          "concat": ROOT / "data/20mad/cmd_concat_cards_mad.json"}
KS = [4, 8]
OUT = RES / "xcard_census_normalized.json"
WORD = re.compile(r"\w+")


def mean_words(cache, k):
    ws = [len(WORD.findall(v)) for kk, v in cache.items() if re.match(rf"k{k}_s\d+_", kk)]
    return round(float(np.mean(ws)), 1), len(ws)


def gate2_label(per100w_excess):
    # consistent rule: a verbatim-copy shortcut if the length-normalized POS excess is well above the NEG floor.
    # threshold 1.0 cleanly separates CMD (0.69/0.20 -> OK) from concat (4.14/1.10 -> WARN).
    return "WARN (verbatim shortcut)" if per100w_excess >= 1.0 else "OK (no verbatim shortcut; ~NEG baseline)"


def main():
    caches = {m: json.loads(p.read_text(encoding="utf-8")) for m, p in CACHES.items()}
    mean_card_words, gate2, seeds_used = {}, {}, {}
    for m, cache in caches.items():
        for k in KS:
            mw, n = mean_words(cache, k)
            mean_card_words[f"{m}_k{k}"] = mw
            seeds_used[f"{m}_k{k}"] = sorted({int(x) for kk in cache for x in re.findall(rf"^k{k}_s(\d+)_", kk)})
            cen = json.loads((RES / f"_xcard_{m}_census_k{k}.json").read_text(encoding="utf-8"))
            mp = cen["matched"]
            pos = round(float(np.mean([x["pos"]["n6"] for x in mp])), 2)
            neg = round(float(np.mean([x["neg"]["n6"] for x in mp])), 2)
            per100_pos = round(pos / (mw / 100.0), 3)   # from rounded pos/mw so it matches the canonical file
            per100_neg = round(neg / (mw / 100.0), 3)
            excess = round(per100_pos - per100_neg, 2)
            gate2[f"{m}_k{k}"] = {"pos": pos, "neg": neg,
                                  "per100w_pos": per100_pos, "per100w_neg": per100_neg,
                                  "per100w_excess": excess, "gate2": gate2_label(excess),
                                  "n_matched": len(mp)}

    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    out = {
        "experiment": old.get("experiment", "length-normalized verbatim cross-card census (CMD vs concat), 20-MAD"),
        "note": old.get("note", "n6 = matched-POS/NEG shared word-6gram counts from the raw census; per-100-word "
                                "normalization divides by mean card length (concat cards ~2x longer). Differentiation "
                                "survives normalization and holds at headline k8."),
        "generated_by": "scripts/xcard_census_norm.py",
        "seeds_used": seeds_used,
        "mean_card_words": mean_card_words,
        "gate2_matched_pos_neg_6gram": gate2,
        "findings": old.get("findings", [
            "Length-normalized (per 100 words) concat carries ~5-6x the CMD excess verbatim overlap at BOTH k4 and k8.",
            "At k8 CMD's absolute overlap ~= NEG baseline (gate2 OK) while concat still WARNs.",
        ]),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("mean_card_words:", mean_card_words)
    for a, v in gate2.items():
        print(f"  {a:10s} pos={v['pos']} neg={v['neg']} per100w_excess={v['per100w_excess']} {v['gate2']}")
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
