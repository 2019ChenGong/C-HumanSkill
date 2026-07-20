"""Export a self-contained 2AFC anonymity pack (meta.json + batch_i.json + sys.txt) for FREE sonnet-4.6 subagents,
for the neutral-transition (#63). Reuses cmd_attack2afc.build_pairs() so the pairing logic (nneg same-topic
stranger, make_groups clusters) is byte-identical to the canonical battery; only bakes a fixed member_slot per pair
into meta.json (the subagents answer a static prompt) and writes prompts into batches.

Channels default to indiv,shared,neutral so each wave is INTERNALLY valid: indiv = positive control (must leak),
shared = base CMD, neutral = the utility-preserving CMD under test. nneg-only (M_RNEG=0) to match the canonical
packs (scorer pools all rows per channel). Scored afterward by score_2afc_summary.py.

Run:  DATASET=cv KCL=8 SEED=0 CHANS=indiv,shared,neutral NEUTRALC=cmd_shared_cards_cv__neutral.json \
      NBATCH=12 BATCHDIR=results/se/2afc_neutral python scripts/neutral_2afc_export.py
"""
import os
import sys
import json
import hashlib
from pathlib import Path

os.environ.setdefault("M_NNEG", "2")
os.environ.setdefault("M_RNEG", "0")            # nneg-only pack (matches canonical batteries; scorer pools per chan)
os.environ.setdefault("MODE", "dryrun")         # build_pairs never calls the API regardless

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cmd_attack2afc as A   # noqa: E402  (reuses build_pairs, SYS, USR_CARD/USR_WRITE, DATASET/KCL/SEED/CHANS)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NBATCH = int(os.environ.get("NBATCH", 12))
OUT = ROOT / os.environ.get("BATCHDIR", f"results/{A.DS}/2afc_neutral")
OUT.mkdir(parents=True, exist_ok=True)


def _swap(chan, cid, member, stranger):
    """Deterministic, balanced slot assignment (baked into meta so the subagent sees a static prompt)."""
    return int(hashlib.sha1(f"swap|{chan}|{cid}|{member}|{stranger}|{A.SEED}".encode()).hexdigest(), 16) % 2 == 1


def main():
    pairs, nclus = A.build_pairs()
    meta, tasks = {}, []
    for i, p in enumerate(pairs):
        pid = f"P{i:04d}"
        swap = _swap(p["chan"], p["card_id"], p["member"], p["stranger"])
        a, b = (p["_sref"], p["_mref"]) if swap else (p["_mref"], p["_sref"])
        member_slot = "B" if swap else "A"
        tmpl = A.USR[p["_kind"]]
        prompt = tmpl.format(card=p["_card"], a=a, b=b)
        meta[pid] = {"chan": p["chan"], "neg": p["neg"], "card_id": p["card_id"],
                     "member": p["member"], "stranger": p["stranger"], "member_slot": member_slot}
        tasks.append({"pid": pid, "prompt": prompt})

    # channel-interleave: build_pairs emits pairs grouped by (cluster,member,channel), so a raw pid%NBATCH can land
    # a whole batch in ONE channel (period-6 cycle resonates with NBATCH). Deterministically shuffle by sha1(pid)
    # then round-robin -> every batch gets a balanced channel mix (no subagent sees a homogeneous channel block).
    order = sorted(tasks, key=lambda t: hashlib.sha1(t["pid"].encode()).hexdigest())
    batches = [[] for _ in range(NBATCH)]
    for j, t in enumerate(order):
        batches[j % NBATCH].append(t)
    for i, b in enumerate(batches):
        (OUT / f"batch_{i}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "sys.txt").write_text(A.SYS, encoding="utf-8")
    (OUT / "samples_only.txt").write_text(
        "\n\n========\n\n".join(f"{t['pid']}\n{t['prompt']}" for t in tasks[:6]), encoding="utf-8")

    import collections
    bych = collections.Counter(m["chan"] for m in meta.values())
    print(f"DS={A.DS} k{A.KCL} s{A.SEED} clusters={nclus}  pairs={len(tasks)}  chans={dict(bych)}")
    print(f"  {NBATCH} batches ({[len(b) for b in batches]}) + meta + sys -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
