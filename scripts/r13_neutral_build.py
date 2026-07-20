"""R13 (#139) — build the MISSING MAD neutral pooled cards at k10/k12 (s0), canonical deepseek.

Same recipe as the shipped neutral cards, reused VERBATIM (no p-hacking): `mad_synth_utility.synth_neutral`
(the utility-preserving pool synth) + `cmd_fix_degenerate.{degeneracy,is_degenerate,synth_anticopy}` (the
anti-copy degeneracy fix), functions imported directly (NOT the cmd_fix_degenerate CLI, whose main() skips
keys absent from its src file — R13 review MINOR-2). Model stays deepseek-chat (the canonical distiller).

Appends k{K}_s0 keys into BOTH canonical multi-k files:
  data/20mad/cmd_shared_cards_mad__neutral.json        (raw synth)
  data/20mad/cmd_shared_cards_mad__neutral_fixed.json  (post degeneracy-fix)
Never touches existing keys: one-time .r13bak backups + byte-identical assert over the pre-existing key set.
Partition sanity: every target cluster must already have a base shared card k{K}_s0_* (built by the old
k-sweep via the same GROUP=random make_groups), else the partition drifted and we stop before spending.

Run:  COST=1 K=10 python -P scripts/r13_neutral_build.py     # price first, no spend
      K=10 python -P scripts/r13_neutral_build.py
      (then K=12; idempotent — existing k{K}_s0 keys are skipped)
"""
import os
import sys
import json
import shutil
from pathlib import Path

import tiktoken

os.environ["DATASET"] = "mad"
os.environ.setdefault("GROUP", "random")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de            # noqa: E402  (de.pool parallel map)
import cmd_gate as CG              # noqa: E402  (load, make_groups)
import mad_synth_utility as MSU    # noqa: E402  (synth_neutral — canonical pooling prompt)
import cmd_fix_degenerate as FIX   # noqa: E402  (degeneracy + synth_anticopy — canonical fix)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ENC = tiktoken.get_encoding("cl100k_base")
K = int(os.environ["K"])
S = int(os.environ.get("SEED", 0))
assert K in (10, 12) and S == 0, f"R13 scope: MAD k10/k12 s0 only, got (k{K}, s{S})"
COST = os.environ.get("COST", "") not in ("", "0")
PIN, POUT = 0.28, 1.10             # deepseek-chat list prices (same constants as the canonical builders)
RAW_P = CG.SE / "cmd_shared_cards_mad__neutral.json"
FIX_P = CG.SE / "cmd_shared_cards_mad__neutral_fixed.json"


def main():
    _pool, authors, nuwa, aggro, _r, _w = CG.load()
    grp, byc = CG.make_groups(aggro, authors, K, S)
    clusters = [c for c in sorted(byc) if len(byc[c]) >= K]
    covered = sum(len(byc[c]) for c in clusters)
    sizes = sorted(len(byc[c]) for c in clusters)
    print(f"MAD k{K}_s{S}: {len(clusters)} clusters (sizes {sizes[0]}-{sizes[-1]}), "
          f"{covered}/{len(authors)} devs covered (make_groups spreads the remainder across groups)", flush=True)

    base = json.loads((CG.SE / "cmd_shared_cards_mad.json").read_text(encoding="utf-8"))
    miss = [c for c in clusters if f"k{K}_s{S}_{c}" not in base]
    assert not miss, (f"base shared cards missing for {miss} -- make_groups no longer reproduces the "
                      f"k-sweep partition; STOP (cards would attach to the wrong members)")

    if COST:
        tin = tout = 0
        for c in clusters:
            body = "\n\n---\n\n".join(aggro[a] for a in byc[c])
            tin += len(ENC.encode(MSU._SYS_KEEP)) + len(ENC.encode(body)) + 120
            tout += 1300
        usd = tin / 1e6 * PIN + tout / 1e6 * POUT
        print(f"=== COST (deepseek neutral synth, {len(clusters)} clusters k{K}_s{S}) ===")
        print(f"  input ~{tin:,} tok @ ${PIN}/M + output <= {tout:,} tok @ ${POUT}/M -> TOTAL <= ${usd:.2f}"
              f"  (+ ~$0.01 per degenerate re-synth)")
        return

    for path in (RAW_P, FIX_P):
        bak = path.with_name(path.name + ".r13bak")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"  backup -> {bak.name}", flush=True)
    raw = json.loads(RAW_P.read_text(encoding="utf-8"))
    fixed = json.loads(FIX_P.read_text(encoding="utf-8"))
    before_raw = dict(raw)
    before_fixed = dict(fixed)

    # ---- 1. neutral synth (canonical prompt/model), idempotent on existing keys ----
    plan = [c for c in clusters if f"k{K}_s{S}_{c}" not in raw]
    if plan:
        print(f"synthesizing {len(plan)} neutral cards (deepseek, canonical prompt) ...", flush=True)
        cards = de.pool(lambda c: MSU.synth_neutral([aggro[a] for a in byc[c]]), plan)
        for c, card in zip(plan, cards):
            assert card and card.strip(), f"EMPTY synth for k{K}_s{S}_{c} -- stop, do not ship"
            raw[f"k{K}_s{S}_{c}"] = card
        assert all(raw[k] == before_raw[k] for k in before_raw), "pre-existing raw keys mutated -- ABORT"
        RAW_P.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    # ---- 2. degeneracy fix -> neutral_fixed (canonical detector + anti-copy re-synth) ----
    nfix = 0
    for c in clusters:
        ckey = f"k{K}_s{S}_{c}"
        if ckey in fixed:
            continue                               # resume: already fixed in an earlier run
        mtexts = [aggro[a] for a in byc[c]]
        card = raw[ckey]
        cos, rf = FIX.degeneracy(card, mtexts)
        if FIX.is_degenerate(cos, rf):
            print(f"  {ckey}: DEGENERATE (cos={cos:.3f} run={rf:.0%}) -> anti-copy re-synth", flush=True)
            best = None
            for t in range(FIX.MAX_TRIES):
                cand = FIX.synth_anticopy(mtexts, temp=0.3 + 0.2 * t)
                c2, r2 = FIX.degeneracy(cand, mtexts)
                if best is None or (c2, r2) < (best[1], best[2]):
                    best = (cand, c2, r2)
                if not FIX.is_degenerate(c2, r2):
                    break
            card = best[0]
            nfix += 1
        fixed[ckey] = card
        assert all(fixed[k] == before_fixed[k] for k in before_fixed), "pre-existing fixed keys mutated -- ABORT"
        FIX_P.write_text(json.dumps(fixed, ensure_ascii=False), encoding="utf-8")

    import numpy as np
    lens = [len(fixed[f"k{K}_s{S}_{c}"] or "") for c in clusters]
    empt = [c for c in clusters if not fixed.get(f"k{K}_s{S}_{c}")]
    print(f"\nDONE k{K}_s{S}: {len(clusters)} neutral+fixed cards ({nfix} anti-copy fixed) -> "
          f"{RAW_P.name} + {FIX_P.name}")
    print(f"  char median {int(np.median(lens))}  min {min(lens)}  max {max(lens)}  empties: {empt or 'none'}")


if __name__ == "__main__":
    main()
