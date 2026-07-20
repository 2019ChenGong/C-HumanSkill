"""R7 (#131) E2 — v6 linkage census + pack gates (MAD k8, seeds 0/1/2).

R7MODE=census (default): builds `_xcard_cmd_v6min_census_k8.json` by REUSING the canonical ne
census's id pairs verbatim (`_xcard_cmd_neutral_census_k8.json`; pool-identity level, card-text
independent) and recomputing ccos/n6/n8 on the v6 card texts. Review M2: this process binds ONE
METHOD (cmd_v6min) — the ne side is a plain json.load, never a second XL import. Gates (B):
id-pair set identity + overlap identity + ccos drift report + n6 before/after table.

R7MODE=packgate: independently rebuilds EVERY exported batch prompt from (census + intended card
file + swapped rule + XL.USR template) and byte-compares against the pack (review M1) — run once
per pack, each in its own process:
  DATASET=mad K=8 SEEDS=0,1,2 GROUP=random METHOD=cmd_neutral \
    CARDSRC=data/20mad/cmd_shared_cards_mad__neutral_fixed.json \
    PACK=results/mad/xcard_free_cmd_neutral_dj_k8_r7 R7MODE=packgate python -P scripts/r7_xcard_v6_census.py
  (and METHOD=cmd_v6min CARDSRC=data/20mad/cmd_shared_cards_mad__v6min.json PACK=..._cmd_v6min_dj_k8_r7)

census recipe:
  DATASET=mad K=8 SEEDS=0,1,2 GROUP=random METHOD=cmd_v6min \
    CARDSRC=data/20mad/cmd_shared_cards_mad__v6min.json R7MODE=census python -P scripts/r7_xcard_v6_census.py
"""
import os
import sys
import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GROUP", "random")
os.environ.setdefault("MODE", "export")          # keep cmd_xcard_link from scoring on import
import cmd_xcard_link as XL  # noqa: E402  (module-level env binding: ONE METHOD per process)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

R7MODE = os.environ.get("R7MODE", "census")
NE_CEN = XL.RES / "_xcard_cmd_neutral_census_k8.json"
print(f"[cfg] DATASET={XL.DS} K={XL.K} SEEDS={XL.SEEDS} GROUP={os.environ.get('GROUP')} "
      f"METHOD={XL.METHOD} CARDSRC={XL.CARDSRC.relative_to(ROOT)}")
assert XL.METHOD == os.environ.get("METHOD"), "tripwire: XL bound a different METHOD than env (stale import?)"


def serial(a_id, b_id, cards, overlap):
    a, b = cards[a_id], cards[b_id]
    return {"a": a_id, "b": b_id, "overlap": overlap,
            "ccos": round(XL.de._cosine(a["cvec"], b["cvec"]), 4),
            "n6": len(a["g6"] & b["g6"]), "n8": len(a["g8"] & b["g8"])}


def mode_census():
    assert XL.DS == "mad", f"R7 E2 is MAD-only by design (got DATASET={XL.DS})"
    assert XL.METHOD == "cmd_v6min" and str(XL.CARDSRC).endswith("__v6min.json"), \
        f"census mode must bind METHOD=cmd_v6min + v6min CARDSRC (got {XL.METHOD}, {XL.CARDSRC.name})"
    ne = json.loads(NE_CEN.read_text(encoding="utf-8"))
    cards = {c["id"]: c for c in XL.build_cards()}
    # sanity vs the sibling neutral_fixed file: same keys, different texts (M1-adjacent tripwire)
    nf = json.loads((ROOT / "data/20mad/cmd_shared_cards_mad__neutral_fixed.json").read_text(encoding="utf-8"))
    same = sum(1 for c in cards.values() if nf.get(f"k{XL.K}_s{c['seed']}_{c['cid']}") == c["text"])
    assert same == 0, f"{same} 'v6' cards are byte-identical to neutral_fixed -> wrong CARDSRC?"

    ids = {p[s]["a"] for mp in ne["matched"] for s in ("pos", "neg") for p in (mp,)} \
        | {p[s]["b"] for mp in ne["matched"] for s in ("pos", "neg") for p in (mp,)} \
        | {c["a"] for c in ne["ctrl"]} | {c["b"] for c in ne["ctrl"]}
    missing = sorted(i for i in ids if i not in cards)
    assert not missing, f"ne-census ids missing from v6 cards: {missing[:8]}"

    def overlap(a, b):
        return len(cards[a]["members"] & cards[b]["members"])

    matched, drift = [], []
    for mp in ne["matched"]:
        row = {}
        for s in ("pos", "neg"):
            ov = overlap(mp[s]["a"], mp[s]["b"])
            assert ov == mp[s]["overlap"], f"overlap drift {mp[s]} -> {ov} (grouping mismatch!)"
            row[s] = serial(mp[s]["a"], mp[s]["b"], cards, ov)
            drift.append(abs(row[s]["ccos"] - mp[s]["ccos"]))
        matched.append(row)
    ctrl = []
    for c in ne["ctrl"]:
        ov = overlap(c["a"], c["b"])
        assert ov == c["overlap"] and ov >= 2
        ctrl.append(serial(c["a"], c["b"], cards, ov))
    out = {"matched": matched, "ctrl": ctrl, "k": ne["k"], "seeds": ne["seeds"],
           "note": "id pairs REUSED verbatim from _xcard_cmd_neutral_census_k8.json (R7 E2 gate B); "
                   "ccos/n6/n8 recomputed on v6min card texts"}
    op = XL.RES / f"_xcard_{XL.METHOD}_census_k{XL.K}.json"
    op.write_text(json.dumps(out, indent=1), encoding="utf-8")

    d = np.array(drift)
    n6_ne = [mp[s]["n6"] for mp in ne["matched"] for s in ("pos", "neg")]
    n6_v6 = [mp[s]["n6"] for mp in matched for s in ("pos", "neg")]
    print(f"[gate B] id pairs: {len(matched)} matched + {len(ctrl)} ctrl — identical to ne census by construction, "
          f"overlaps verified against v6 groupings 100%")
    print(f"[gate B] ccos drift |v6-ne|: median={np.median(d):.4f} p90={np.percentile(d, 90):.4f} max={d.max():.4f}"
          f"  (topic-matching inherited from ne census; expect small)")
    print(f"[gate B] shared-6gram n6 mean: ne {np.mean(n6_ne):.2f} -> v6 {np.mean(n6_v6):.2f}   "
          f"(pairs with n6>0: ne {sum(1 for x in n6_ne if x)} -> v6 {sum(1 for x in n6_v6 if x)})")
    print(f"saved -> {op.relative_to(ROOT)}")


def swapped(ta, tb, salt):
    if int(hashlib.sha1(("ord" + salt).encode()).hexdigest(), 16) % 2:
        return tb, ta
    return ta, tb


def mode_packgate():
    pack = ROOT / os.environ["PACK"]
    cen = json.loads((XL.RES / f"_xcard_{XL.METHOD}_census_k{XL.K}.json").read_text(encoding="utf-8"))
    cache = json.loads(XL.CARDSRC.read_text(encoding="utf-8"))

    def text(card_id):                         # independent path: direct key map, NOT build_cards
        s, cid = card_id.split(":")
        return cache[f"k{XL.K}_{s}_{cid}"]

    cards = {c["id"]: c for c in XL.build_cards()}

    def shared_member(a, b):
        s = cards[a]["members"] & cards[b]["members"]
        return sorted(s)[0] if s else None

    want_meta, want_prompt = {}, {}
    for i, mp in enumerate(cen["matched"]):
        sm = shared_member(mp["pos"]["a"], mp["pos"]["b"])
        for kind, side in (("pos", mp["pos"]), ("neg", mp["neg"])):
            pid = f"{kind}{i}"
            ta, tb = swapped(text(side["a"]), text(side["b"]), pid)
            want_meta[pid] = {"kind": kind, "mi": i, "cluster": sm, "n6": side["n6"]}
            want_prompt[pid] = XL.USR.format(a=ta, b=tb)
    for i, c in enumerate(cen["ctrl"]):
        pid = f"ctrl{i}"
        ta, tb = swapped(text(c["a"]), text(c["b"]), pid)
        want_meta[pid] = {"kind": "ctrl", "mi": -1, "cluster": None, "n6": c.get("n6")}
        want_prompt[pid] = XL.USR.format(a=ta, b=tb)

    meta = json.loads((pack / "meta.json").read_text(encoding="utf-8"))
    got_prompt = {}
    for f in sorted(pack.glob("batch_*.json")):
        for t in json.loads(f.read_text(encoding="utf-8")):
            got_prompt[t["pid"]] = t["prompt"]
    assert (pack / "sys.txt").read_text(encoding="utf-8") == XL.SYS, "sys.txt != XL.SYS"
    assert set(meta) == set(want_meta) == set(got_prompt), \
        f"pid sets differ: meta {len(meta)} want {len(want_meta)} prompts {len(got_prompt)}"
    bm = [p for p in want_meta if meta[p] != want_meta[p]]
    bp = [p for p in want_prompt if got_prompt[p] != want_prompt[p]]
    assert not bm and not bp, f"PACKGATE FAIL: meta {len(bm)} {bm[:5]} / prompt {len(bp)} {bp[:5]}"
    print(f"[packgate] {pack.name}: {len(want_prompt)} prompts byte-identical, meta identical, sys identical "
          f"(rebuilt from census + {XL.CARDSRC.name} via independent key path)")


if __name__ == "__main__":
    {"census": mode_census, "packgate": mode_packgate}[R7MODE]()
