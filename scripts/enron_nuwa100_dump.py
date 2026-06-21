"""Dump Enron re-identification lineups for strong-attacker scoring."""
import os
import re
import sys
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

SE = ROOT / "data" / "enron"
RES = ROOT / "results"
COLL = SE / os.environ.get("COLL", "collected_rag100_40.json")
NUWAC = SE / os.environ.get("NUWAC", "nuwa_cards_100.json")
STEP2C = SE / os.environ.get("STEP2C", "step2_cards_100.json")
N_TRAIN, N_REF, N_TGT = 12, 2, 2
REF_CHARS = int(os.environ.get("REF_CHARS", 400))
KLINE = int(os.environ.get("KLINE", 20))
WS = re.compile(r"\s+")


def main():
    COND = os.environ.get("COND", "archpool")
    docs = json.loads(COLL.read_text(encoding="utf-8"))
    authors = [a for a in docs if len(docs[a]) >= N_TRAIN + N_REF + N_TGT]
    ref = {a: WS.sub(" ", " || ".join(docs[a][N_TRAIN + r]["text"] for r in range(N_REF)))[:REF_CHARS] for a in authors}
    nuwa = json.loads(NUWAC.read_text(encoding="utf-8"))["nuwa"]
    step2 = json.loads(STEP2C.read_text(encoding="utf-8"))
    cards = {"nuwa": nuwa, **{k: step2[k] for k in ["aggro", "archpool", "random_pool", "adv_paraphrase"] if k in step2}}

    def target_of(a):
        if COND == "comment":
            return WS.sub(" ", docs[a][N_TRAIN + N_REF]["text"])[:900], "comment-100"
        if COND not in cards:
            raise SystemExit(f"COND={COND} not built (have {list(cards)})")
        return cards[COND][a], "card-100"

    def lineup(a, tag):
        # 19 deterministic distractors + true author, shuffled by a per-target hash (shared across card arms)
        others = sorted([b for b in authors if b != a],
                        key=lambda b: hashlib.sha1(f"distract-{a}-{b}".encode()).hexdigest())[:KLINE - 1]
        pool = others + [a]
        return sorted(pool, key=lambda b: hashlib.sha1(f"{tag}-{a}-{b}".encode()).hexdigest())

    kind = "a person's work EMAIL" if COND == "comment" else "a SKILL CARD (distilled working/decision heuristics)"
    key = {}
    for i, a in enumerate(authors, 1):
        tgt, tag = target_of(a)
        order = lineup(a, tag)
        key[f"T{i:03d}"] = {"cond": COND, "author": a, "true_candidate": order.index(a) + 1}
        lines = ["# Authorship re-identification -- single isolated trial",
                 f"The TARGET below is {kind} from ONE author. Identify which candidate is that SAME author,",
                 "by their reasoning, priorities, and decision style -- NOT by topic (topics overlap).\n",
                 "TARGET:", tgt, "", f"CANDIDATES (each a sample of their own writing; pick exactly ONE of 1..{len(order)}):"]
        for j, b in enumerate(order, 1):
            lines.append(f"[{j}] {ref[b]}")
        (RES / f"_e100k{KLINE}_single_{COND}_T{i:03d}.txt").write_text("\n".join(lines), encoding="utf-8")
    (RES / f"_e100k{KLINE}_single_{COND}_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"wrote {len(authors)} {COND} trials -> _e100k{KLINE}_single_{COND}_T001..  (K={KLINE}, chance={1/KLINE:.3f})", flush=True)


if __name__ == "__main__":
    main()
