"""Export cross-card LINKAGE pairs as self-contained tasks for FREE sonnet-4.6 subagents (P8/H3).

Same job set cmd_xcard_link.py MODE=run would score with the PAID sonnet, but written out so the free
Claude-Code sonnet subagents can answer instead ($0). Reuses build_cards()/SYS/USR from cmd_xcard_link so
the pairs are byte-identical to the paid path. POS/NEG come from the census `matched`; positive controls =
the census `ctrl` real share>=2 pairs (no deepseek synth needed).

  DATASET=mad K=4 SEEDS=0,1,2 METHOD=concat CARDSRC=data/20mad/cmd_concat_cards_mad.json BATCH=26 \
    python scripts/cmd_xcard_export.py
Out: results/{ds}/xcard_free_{METHOD}_k{K}/  batch_i.json {pid,prompt} + sys.txt + meta.json(answer key)
"""
import os
import sys
import json
import random
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GROUP", "random")
os.environ.setdefault("DATASET", "mad")
os.environ.setdefault("METHOD", "concat")
os.environ.setdefault("MODE", "export")   # keep cmd_xcard_link from thinking it should score
import cmd_xcard_link as XL  # noqa: E402  (module-level globals read env on import)

BATCH = int(os.environ.get("BATCH", 26))
DISJOINT = bool(int(os.environ.get("DISJOINT", "0")))   # pack so no card_id repeats within a batch (kills within-batch cross-pair inference)
SAMPLE = int(os.environ.get("SAMPLE", "0"))             # >0 => random subsample this many matched pairs (seeded) for a small pilot
TAG = os.environ.get("TAG", "")                         # non-empty => isolate the pilot dir/result so a later FULL run never clobbers it
OUT = XL.RES / f"xcard_free_{XL.METHOD}{'_dj' if DISJOINT else ''}_k{XL.K}{('_' + TAG) if TAG else ''}"
OUT.mkdir(parents=True, exist_ok=True)


def swapped(ta, tb, salt):
    """Deterministic A/B order swap (same rule as cmd_xcard_link.link_score) so CARD 1 isn't always the same side."""
    if int(hashlib.sha1(("ord" + salt).encode()).hexdigest(), 16) % 2:
        return tb, ta
    return ta, tb


def main():
    cen = json.loads((XL.RES / f"_xcard_{XL.METHOD}_census_k{XL.K}.json").read_text(encoding="utf-8"))
    if SAMPLE > 0:                                       # small pilot: seeded random subsample of matched pairs (+few ctrl)
        rng = random.Random(0)
        m = cen["matched"]
        cen["matched"] = rng.sample(m, min(SAMPLE, len(m)))
        cen["ctrl"] = rng.sample(cen.get("ctrl", []), min(4, len(cen.get("ctrl", []))))
        print(f"PILOT subsample: {len(cen['matched'])} matched (+{len(cen['ctrl'])} ctrl) of {len(m)}")
    cards = {c["id"]: c for c in XL.build_cards()}

    def shared_member(idA, idB):
        s = cards[idA]["members"] & cards[idB]["members"]
        return sorted(s)[0] if s else None

    jobs = []   # (pid, kind, mi, cluster, n6, textA, textB, idA, idB)
    for i, mp in enumerate(cen["matched"]):
        sm = shared_member(mp["pos"]["a"], mp["pos"]["b"])
        for kind, side in (("pos", mp["pos"]), ("neg", mp["neg"])):
            pid = f"{kind}{i}"
            ta, tb = swapped(cards[side["a"]]["text"], cards[side["b"]]["text"], pid)
            jobs.append((pid, kind, i, sm, side["n6"], ta, tb, side["a"], side["b"]))
    for i, cp in enumerate(cen.get("ctrl", [])):
        pid = f"ctrl{i}"
        ta, tb = swapped(cards[cp["a"]]["text"], cards[cp["b"]]["text"], pid)
        jobs.append((pid, "ctrl", -1, None, cp.get("n6"), ta, tb, cp["a"], cp["b"]))

    meta = {}
    items = []   # (pid, prompt, idA, idB) in order
    for (pid, kind, mi, cluster, n6, ta, tb, idA, idB) in jobs:
        meta[pid] = {"kind": kind, "mi": mi, "cluster": cluster, "n6": n6}
        items.append((pid, XL.USR.format(a=ta, b=tb), idA, idB))

    if DISJOINT:
        # greedy first-fit so NO card_id repeats within a batch -> a worker sees each card at most once, so it
        # cannot cross-reference "which card recurs across pairs". AUC is rank-based, so any residual batch-level
        # calibration shift does not affect it -> this removes the within-batch inference confound.
        batches = []   # {"cards": set, "items": [task]}
        for (pid, prompt, idA, idB) in items:
            placed = False
            for bk in batches:
                if len(bk["items"]) < BATCH and idA not in bk["cards"] and idB not in bk["cards"]:
                    bk["items"].append({"pid": pid, "prompt": prompt}); bk["cards"].update((idA, idB)); placed = True; break
            if not placed:
                batches.append({"cards": {idA, idB}, "items": [{"pid": pid, "prompt": prompt}]})
        grouped = [bk["items"] for bk in batches]
        card_of = {pid: (idA, idB) for (pid, _p, idA, idB) in items}   # verify disjointness
        for g in grouped:
            seen = []
            for t in g:
                seen += list(card_of[t["pid"]])
            assert len(seen) == len(set(seen)), "DISJOINT violated: a card repeats within a batch"
    else:
        tasks = [{"pid": pid, "prompt": prompt} for (pid, prompt, _a, _b) in items]
        grouped = [tasks[b:b + BATCH] for b in range(0, len(tasks), BATCH)]

    (OUT / "sys.txt").write_text(XL.SYS, encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    for nb, g in enumerate(grouped):
        (OUT / f"batch_{nb}.json").write_text(json.dumps(g, ensure_ascii=False), encoding="utf-8")
    nb = len(grouped)
    npos = sum(1 for m in meta.values() if m["kind"] == "pos")
    nneg = sum(1 for m in meta.values() if m["kind"] == "neg")
    nctrl = sum(1 for m in meta.values() if m["kind"] == "ctrl")
    szs = sorted(set(len(g) for g in grouped))
    print(f"exported {len(items)} pairs (POS {npos} / NEG {nneg} / CTRL {nctrl}) -> {nb} batches"
          f"{' (CARD-DISJOINT, no card repeats in a batch; sizes '+str(szs)+')' if DISJOINT else ' of <= '+str(BATCH)}")
    print(f"  dir: {OUT.relative_to(ROOT)}   (sys.txt + meta.json[answer key, hide from workers] + batch_*.json)")


if __name__ == "__main__":
    main()
