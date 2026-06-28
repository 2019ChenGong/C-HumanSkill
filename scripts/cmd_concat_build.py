"""Build naive concat-summarize-k pooled cards: a SECOND hard-pooling baseline beside CMD.

Unlike CMD `synth_shared` (common-ONLY consensus -> strips per-member detail), this NAIVELY MERGES all k member
cards into ONE combined card keeping union coverage. It is STILL byte-identical across the k members (=> same
≤1/k structural floor as CMD), but should leak MORE (per-member detail survives) + more verbatim -> demonstrates
the structural floor is a METHOD-CLASS property (not CMD-only), while consensus-synthesis quality matters ABOVE
the floor (CMD vs concat separation = the value of the consensus operator).

Run:  DATASET=mad KCL=8 SEED=1 MODE=run python scripts/cmd_concat_build.py
Out:  data/{ds}/cmd_concat_cards[_mad].json   keyed  k{K}_s{S}_{cid} -> card text  (same key scheme as SHAREDC).
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
import cmd_gate as CG  # noqa: E402
from src.llm import chat  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DS = CG.DATASET
KCL = int(os.environ.get("KCL", 8))
SEED = int(os.environ.get("SEED", 1))
MODE = os.environ.get("MODE", "dryrun")
GEN = "deepseek-chat"
OUT = CG.SE / os.environ.get("CONCATC", "cmd_concat_cards.json" if DS == "enron" else "cmd_concat_cards_mad.json")


def synth_concat(member_cards):
    """Naive union merge (NOT consensus) -> one byte-identical card that keeps each member's distinct moves."""
    body = "\n\n---\n\n".join(member_cards)
    msg = [{"role": "system", "content": "You merge several colleagues' skill cards into ONE combined skill card."},
           {"role": "user", "content": f"Skill cards from several colleagues:\n\n{body}\n\nWrite ONE combined skill "
            "card (working/decision heuristics) that MERGES and COVERS all of their approaches. Summarize for brevity "
            "but KEEP the union of their distinct heuristics, sequencing, and decision moves. Output ONLY the card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=900) or ""


def main():
    _d, authors, _n, aggro, _r, _t = CG.load()
    grp, byc = CG.make_groups(aggro, authors, KCL, SEED)
    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    plan = []
    for cid, mem in byc.items():
        if len(mem) < KCL:
            continue
        ck = f"k{KCL}_s{SEED}_{cid}"
        if ck not in cache:
            plan.append((ck, [aggro[a] for a in mem]))
    print(f"concat-build DS={DS} k{KCL} s{SEED}: groups={len(byc)} to-synth={len(plan)} cached={len(cache)} "
          f"(deepseek ~${len(plan) * 0.001:.3f})")
    if MODE != "run":
        print("dryrun: set MODE=run to synthesize.")
        return
    if plan:
        for (ck, _), card in zip(plan, de.pool(lambda pc: synth_concat(pc[1]), plan)):
            cache[ck] = card
        OUT.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"saved {len(cache)} concat cards -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
