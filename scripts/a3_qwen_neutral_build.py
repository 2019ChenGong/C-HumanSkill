"""A3 — build the qwen3.7-max-DISTILLED fixed-neutral MAD pooled cards (cross-DISTILLER robustness).

The pooled-anonymity result is measured on cards distilled by deepseek-chat. A2 swapped the ATTACKER (gpt-5.4), B2
the JUDGE (qwen); A3 swaps the DISTILLER: rebuild the MAD pooled cards with qwen3.7-max (non-thinking) and re-run
the same 2AFC. If qwen-distilled pooled cards are ALSO anonymous, the anonymity is a property of POOLING, not of
deepseek's card-writing style. A base-card qwen result exists (#4: pooled .488 ~chance); this is the honest,
degeneracy-fixed, neutral-synth version aligned to the current canonical.

Reuses the CANONICAL prompts verbatim (no p-hacking): `mad_synth_utility.synth_neutral` (the utility-preserving
pool) + `cmd_fix_degenerate.{degeneracy,is_degenerate,synth_anticopy}` (the anti-copy fix). Only the model is
swapped to qwen (non-thinking, aligned to how the base qwen build ran). The per-person qwen nuwa/aggro cards are
CACHED from the base build (`mad_cmd_{nuwa,step2}__qwen.json`) → only the ~16 cluster neutral-synths + a few fixes
are new spend.

Run:  COST=1 python -P scripts/a3_qwen_neutral_build.py      # price the qwen synth calls first, NO spend
      python -P scripts/a3_qwen_neutral_build.py             # build -> data/20mad/cmd_shared_cards_mad__qwen__neutral_fixed.json
Then 2AFC (free sonnet), pointing the harness at the qwen cards via env:
  DATASET=mad KCL=8 SEED=0 CHANS=indiv,neutral NEUTRALC=cmd_shared_cards_mad__qwen__neutral_fixed.json \
    NUWAC=mad_cmd_nuwa__qwen.json STEP2C=mad_cmd_step2__qwen.json \
    NBATCH=12 BATCHDIR=results/mad/a3_qwen_neufix python scripts/neutral_2afc_export.py
"""
import os
import sys
import json
from pathlib import Path

import tiktoken

os.environ["DATASET"] = "mad"
os.environ.setdefault("GROUP", "random")
# point CG.load at the qwen-distilled per-person cards (cached from the base qwen build)
os.environ.setdefault("NUWAC", "mad_cmd_nuwa__qwen.json")
os.environ.setdefault("STEP2C", "mad_cmd_step2__qwen.json")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import src.llm as L                # noqa: E402
import deid_enron as de            # noqa: E402
import cmd_gate as CG              # noqa: E402
import mad_synth_utility as MSU    # noqa: E402  (synth_neutral — canonical pooling prompt)
import cmd_fix_degenerate as FIX   # noqa: E402  (degeneracy + synth_anticopy — canonical fix)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENC = tiktoken.get_encoding("cl100k_base")
QWEN = os.environ.get("QWEN_MODEL", "openrouter/qwen/qwen3.7-max")
K = int(os.environ.get("K", 8))
S = int(os.environ.get("SEED", 0))
COST = os.environ.get("COST", "") not in ("", "0")
PRICE_IN, PRICE_OUT = 1.25, 3.75
OUT = CG.SE / os.environ.get("OUT", "cmd_shared_cards_mad__qwen__neutral_fixed.json")

_orig_chat = L.chat


def qchat(messages, model=None, temperature=0.0, max_tokens=1024, use_cache=True, retries=5, extra=None):
    """Force qwen NON-thinking (reasoning.enabled=False), aligned to the base qwen distiller build."""
    if extra is None and "qwen" in QWEN.lower():
        extra = {"reasoning": {"enabled": False}}
    return _orig_chat(messages, model=QWEN, temperature=temperature, max_tokens=max_tokens,
                      use_cache=use_cache, retries=retries, extra=extra)


# swap the model in every module that synthesizes a pooled card
MSU.chat = qchat; MSU.GEN = QWEN
FIX.chat = qchat; FIX.GEN = QWEN


def main():
    _pool, authors, nuwa, aggro, _r, _w = CG.load()   # nuwa/aggro = qwen (env-pointed)
    grp, byc = CG.make_groups(aggro, authors, K, S)
    clusters = [c for c in sorted(byc) if len(byc[c]) >= K]
    print(f"qwen distiller = {QWEN} (non-thinking) | {len(clusters)} clusters k{K}_s{S} | aggro from {os.environ['STEP2C']}", flush=True)

    if COST:
        tin = tout = 0
        for c in clusters:
            body = "\n\n---\n\n".join(aggro[a] for a in byc[c])
            # synth_neutral prompt = _SYS_KEEP + body + fixed instruction (~120 tok); output cap 1300
            tin += len(ENC.encode(MSU._SYS_KEEP)) + len(ENC.encode(body)) + 120
            tout += 1300
        usd = tin / 1e6 * PRICE_IN + tout / 1e6 * PRICE_OUT
        print(f"=== COST (qwen neutral synth, {len(clusters)} clusters; degeneracy fixes extra ~2-4 calls) ===")
        print(f"  input ~{tin:,} tok @ ${PRICE_IN}/M = ${tin/1e6*PRICE_IN:.3f}")
        print(f"  output <= {tout:,} tok @ ${PRICE_OUT}/M = ${tout/1e6*PRICE_OUT:.3f}")
        print(f"  TOTAL <= ${usd:.2f}  (+ ~$0.02 per degenerate re-synth; nuwa/aggro already cached = $0)")
        return

    # ---- 1. neutral synth (qwen), idempotent ----
    cache = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    plan = [c for c in clusters if f"k{K}_s{S}_{c}" not in cache]
    if plan:
        print(f"building {len(plan)} qwen neutral cards ...", flush=True)
        cards = de.pool(lambda c: MSU.synth_neutral([aggro[a] for a in byc[c]]), plan)
        for c, card in zip(plan, cards):
            cache[f"k{K}_s{S}_{c}"] = card
        OUT.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    # ---- 2. degeneracy fix (qwen anti-copy), in place ----
    nfix = 0
    for c in clusters:
        ck = f"k{K}_s{S}_{c}"
        mtexts = [aggro[a] for a in byc[c]]
        cos, rf = FIX.degeneracy(cache[ck], mtexts)
        if not FIX.is_degenerate(cos, rf):
            continue
        print(f"  {ck}: DEGENERATE (cos={cos:.3f} run={rf:.0%}) -> qwen anti-copy re-synth", flush=True)
        best = None
        for t in range(FIX.MAX_TRIES):
            card = FIX.synth_anticopy(mtexts, temp=0.3 + 0.2 * t)
            c2, r2 = FIX.degeneracy(card, mtexts)
            if best is None or (c2, r2) < (best[1], best[2]):
                best = (card, c2, r2)
            if not FIX.is_degenerate(c2, r2):
                break
        cache[ck] = best[0]; nfix += 1
        OUT.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    import numpy as np
    lens = [len(cache[f"k{K}_s{S}_{c}"] or "") for c in clusters]
    print(f"\nDONE: {len(clusters)} qwen neutral_fixed cards ({nfix} anti-copy fixed) -> {OUT.relative_to(ROOT)}")
    print(f"  char median {int(np.median(lens))}  min {min(lens)}  max {max(lens)}")
    empties = [c for c in clusters if not cache.get(f'k{K}_s{S}_{c}')]
    print(f"  empties: {empties or 'none'}")


if __name__ == "__main__":
    main()
