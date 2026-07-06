"""Build ONLY the CMD ε=0 shared cards (synth_shared) for a k-sweep into SHAREDC — no _cmdgate lineup dump.
Idempotent: reuses cmd_gate.build_shared (skips already-cached k{k}_s{s}_{cid} keys). CMD-only (no concat / de-id).

Run: DATASET=mad GROUP=random K_LIST=10,12 SEEDS=0 SHAREDC=cmd_shared_cards_mad.json python scripts/cmd_build_shared.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GROUP", "random")
import cmd_gate as CG  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

docs, authors, nuwa, aggro, ref, raw_tgt = CG.load()
print(f"DATASET={os.environ.get('DATASET','mad')} N={len(authors)} K_LIST={CG.K_LIST} SEEDS={CG.SEEDS} "
      f"GROUP={CG.GROUP} SHAREDC={CG.SHAREDC.name}", flush=True)
cache, layout = CG.build_shared(authors, aggro)
for (k, s), (grp, byc) in sorted(layout.items()):
    miss = [f"k{k}_s{s}_{cid}" for cid in byc if f"k{k}_s{s}_{cid}" not in cache]
    assert not miss, f"missing shared cards: {miss}"
    sizes = sorted(len(v) for v in byc.values())
    print(f"  k={k} s={s}: {len(byc)} groups OK  (sizes min/max={sizes[0]}/{sizes[-1]})", flush=True)
print(f"total keys in {CG.SHAREDC.name}: {len(cache)}", flush=True)
