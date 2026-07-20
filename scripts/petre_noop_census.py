"""petre_noop_census.py -- $0 descriptive census: how much did each per-person de-id arm
actually CHANGE the individual card it was given?

Motivation (2026-07-19): the only SIG-losing utility cell in R3' is CV v6-petre (.410).
The registered attribution is "petre = no-op arm". This script turns that phrase into numbers:
per dataset x arm, the fraction of cards byte-identical to the indiv card, word-level
SequenceMatcher similarity, and retention of the original card's 6-word runs.

Pure string processing over existing card files -- no model calls, no spend.
Output: results/petre_noop_census.json
"""
import json, re, difflib, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def words(t):
    return re.findall(r"\w+", t.lower())

def sixg(ws):
    return set(tuple(ws[i:i + 6]) for i in range(len(ws) - 5))

def measure(nuwa, cards):
    ratios, keep6, lenr, ident, n = [], [], [], 0, 0
    for u, card in cards.items():
        if u not in nuwa or not isinstance(card, str):
            continue
        a, b = nuwa[u], card
        n += 1
        if a.strip() == b.strip():
            ident += 1
        wa, wb = words(a), words(b)
        ratios.append(difflib.SequenceMatcher(None, wa, wb).ratio())
        ga, gb = sixg(wa), sixg(wb)
        if ga:
            keep6.append(len(ga & gb) / len(ga))
        lenr.append(len(wb) / max(1, len(wa)))
    return {
        "n": n,
        "byte_identical": ident,
        "byte_identical_frac": round(ident / max(1, n), 4),
        "wordsim_mean": round(statistics.mean(ratios), 4),
        "wordsim_median": round(statistics.median(ratios), 4),
        "sixgram_keep_mean": round(statistics.mean(keep6), 4),
        "sixgram_keep_median": round(statistics.median(keep6), 4),
        "len_ratio_mean": round(statistics.mean(lenr), 4),
    }

def load(p, key=None):
    d = json.loads((ROOT / p).read_text(encoding="utf-8"))
    return d.get(key, d) if key else d

DATASETS = {
    "cv":    ("data/se/cv_cmd_nuwa.json",       "data/se/cv_cmd_step2.json"),
    "mad":   ("data/20mad/mad_cmd_nuwa.json",   "data/20mad/mad_cmd_step2.json"),
    "enron": ("data/enron/nuwa_cards_full.json", "data/enron/step2_cards_full.json"),
}
ARMS = ["petre_k4", "staab", "tpar_t15", "presidio"]

out = {}
for ds, (nuwa_p, step2_p) in DATASETS.items():
    nuwa = load(nuwa_p, "nuwa")
    step2 = load(step2_p)
    out[ds] = {}
    for arm in ARMS:
        if arm in step2:
            out[ds][arm] = measure(nuwa, step2[arm])
            r = out[ds][arm]
            print(f"{ds:5s} {arm:9s} n={r['n']:3d} identical={r['byte_identical']:3d} "
                  f"({100*r['byte_identical_frac']:.0f}%) wordsim={r['wordsim_mean']:.3f} "
                  f"6g-keep={r['sixgram_keep_mean']:.3f}")

dst = ROOT / "results/petre_noop_census.json"
dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"wrote {dst.relative_to(ROOT)}")
