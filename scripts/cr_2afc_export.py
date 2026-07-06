"""Export 2AFC pairs as self-contained text tasks for FREE Claude-Code subagents (no paid OpenRouter sonnet).

Reuses cmd_attack2afc.build_pairs() (pure construction, no API) to get the exact same nneg pairing/grouping as the
scripted attacker, then deterministically randomizes A/B slot per pair, formats the same SYS+USR_CARD prompt, and
splits into NBATCH batch files. A subagent reads one batch_i.json and writes ans_i.json = [{pid, choice, conf}].
Aggregate with cr_2afc_score.py.

Run: DATASET=cr KCL=8 SEED=0 GROUP=random CHANS=shared,indiv M_NNEG=2 M_RNEG=0 NBATCH=8 python scripts/cr_2afc_export.py
"""
import os
import sys
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("CHANS", "shared,indiv")
os.environ.setdefault("M_RNEG", "0")
import cmd_attack2afc as A2  # noqa: E402

NBATCH = int(os.environ.get("NBATCH", 8))
OUT = ROOT / os.environ.get("BATCHDIR", "results/cr/2afc_batches")
OUT.mkdir(parents=True, exist_ok=True)


def main():
    pairs, nclus = A2.build_pairs()
    print(f"built {len(pairs)} pairs over {nclus} clusters (chans={A2.CHANS})", flush=True)
    meta = {}
    tasks = []
    for i, p in enumerate(pairs):
        pid = f"P{i:04d}"
        swap = (int(hashlib.sha1(pid.encode()).hexdigest(), 16) % 2 == 1)
        a, b = (p["_sref"], p["_mref"]) if swap else (p["_mref"], p["_sref"])
        member_slot = "B" if swap else "A"
        prompt = A2.USR_CARD.format(card=p["_card"], a=a, b=b)
        tasks.append({"pid": pid, "prompt": prompt})
        meta[pid] = {"chan": p["chan"], "neg": p["neg"], "card_id": p["card_id"],
                     "member": p["member"], "stranger": p["stranger"], "member_slot": member_slot}
    # interleave so each batch has a mix of indiv/shared (balanced load + no per-batch chan bias)
    tasks.sort(key=lambda t: int(t["pid"][1:]) % NBATCH)
    batches = [[] for _ in range(NBATCH)]
    for t in tasks:
        batches[int(t["pid"][1:]) % NBATCH].append(t)
    for i, b in enumerate(batches):
        (OUT / f"batch_{i}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "sys.txt").write_text(A2.SYS, encoding="utf-8")
    print(f"wrote {NBATCH} batches ({[len(b) for b in batches]}) + meta.json + sys.txt -> {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
