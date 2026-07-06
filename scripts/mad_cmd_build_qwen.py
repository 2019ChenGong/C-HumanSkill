"""#4 cross-model distiller: rebuild the FULL 20-MAD card stack (nuwa -> aggro -> shared) with
qwen3.7-max in NON-thinking mode (reasoning.enabled=False -> aligned to deepseek-chat), writing to
__qwen files so the deepseek originals are NEVER touched. Then the 2AFC anonymity harness is re-run
with NUWAC/STEP2C/SHAREDC pointed at these files to test whether "pooling -> anonymity" is a
deepseek-consensus artifact or model-agnostic.

Isolation: monkeypatch each reused module's `chat` (to inject reasoning-off for qwen) + its `GEN`
constant -> qwen. No permanent edits to mad_nuwa_step2 / enron_step2 / cmd_gate. Same build LOGIC as
the audited mad_cmd_build.py (same dev filter, same tr[:18], same 2-call nuwa + 1-call aggro).

Run:  NDEV=3 python scripts/mad_cmd_build_qwen.py      # smoke: 3 devs, eyeball cards
      python scripts/mad_cmd_build_qwen.py             # full 128 devs (~$1.6 qwen non-thinking)
Out:  data/20mad/{mad_cmd_nuwa__qwen,mad_cmd_step2__qwen}.json + cmd_shared_cards_mad__qwen.json
"""
import os
import sys
import json
from pathlib import Path

os.environ["DATASET"] = "mad"
os.environ.setdefault("GROUP", "random")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

QWEN = os.environ.get("QWEN_MODEL", "openrouter/qwen/qwen3.7-max")
NDEV = int(os.environ.get("NDEV", "0"))   # 0 = all devs; >0 = smoke subset
KCL = int(os.environ.get("KCL", "8"))
SEED = int(os.environ.get("SEED", "0"))

import src.llm as L  # noqa: E402
_orig_chat = L.chat


def qchat(messages, model=L.DEFAULT_MODEL, temperature=0.0, max_tokens=1024, use_cache=True, retries=5, extra=None):
    """Force qwen to NON-thinking (reasoning.enabled=False) so it matches deepseek-chat's mode."""
    if extra is None and "qwen" in str(model).lower():
        extra = {"reasoning": {"enabled": False}}
    return _orig_chat(messages, model=model, temperature=temperature, max_tokens=max_tokens,
                      use_cache=use_cache, retries=retries, extra=extra)


L.chat = qchat
import deid_enron as de            # noqa: E402  (pool, 12 workers)
import mad_nuwa_step2 as MN        # noqa: E402  (nuwa_extract, nuwa_assemble)
import enron_step2 as ES          # noqa: E402  (aggro_card)
import cmd_gate as CG             # noqa: E402  (synth_shared, build_shared, make_groups)
import mad_cmd_build as MB        # noqa: E402  (POOL, BAR, N_TRAIN — reuse the audited constants)

# --- swap model + chat in every module that distills a card ---
for m in (MN, ES, CG):
    m.chat = qchat
    m.GEN = QWEN

MAD = ROOT / "data" / "20mad"
NUWAC_Q = MAD / "mad_cmd_nuwa__qwen.json"
STEP2C_Q = MAD / "mad_cmd_step2__qwen.json"
SHAREDC_Q = MAD / "cmd_shared_cards_mad__qwen.json"

pool = json.loads(MB.POOL.read_text(encoding="utf-8"))["pool"]
devs = [d for d in pool if len(pool[d]["card_comments"]) >= MB.BAR]
if NDEV:
    devs = devs[:NDEV]
tr = {d: pool[d]["card_comments"][:MB.N_TRAIN] for d in devs}
print(f"qwen distiller = {QWEN} (non-thinking) | devs={len(devs)}{' (SMOKE)' if NDEV else ''} | k={KCL} s={SEED}", flush=True)

# ---- nuwa (2-call), idempotent ----
nuwa = json.loads(NUWAC_Q.read_text(encoding="utf-8"))["nuwa"] if NUWAC_Q.exists() else {}
miss = [d for d in devs if d not in nuwa]
if miss:
    print(f"building {len(miss)} nuwa cards (2-call qwen) ...", flush=True)
    for d, card in zip(miss, de.pool(lambda d: MN.nuwa_assemble(MN.nuwa_extract(tr[d])), miss)):
        nuwa[d] = card
    NUWAC_Q.write_text(json.dumps({"nuwa": nuwa}, ensure_ascii=False), encoding="utf-8")

# ---- aggro (1-call on nuwa), idempotent ----
step2 = json.loads(STEP2C_Q.read_text(encoding="utf-8")) if STEP2C_Q.exists() else {}
aggro = step2.get("aggro", {})
miss = [d for d in devs if d not in aggro]
if miss:
    print(f"building {len(miss)} aggro cards (1-call qwen) ...", flush=True)
    for d, card in zip(miss, de.pool(lambda d: ES.aggro_card(nuwa[d]), miss)):
        aggro[d] = card
    step2["aggro"] = aggro
    STEP2C_Q.write_text(json.dumps(step2, ensure_ascii=False), encoding="utf-8")

# ---- shared (synth_shared over qwen aggro), idempotent ----
CG.SHAREDC = SHAREDC_Q
CG.K_LIST = [KCL]
CG.SEEDS = [SEED]
cache, layout = CG.build_shared(devs, aggro)
for (k, s), (grp, byc) in sorted(layout.items()):
    sizes = sorted(len(v) for v in byc.values())
    print(f"  shared k={k} s={s}: {len(byc)} clusters (sizes {sizes[0]}/{sizes[-1]})", flush=True)

# ---- sanity: card lengths + a sample head (catch empties/truncation/thinking-leak) ----
import numpy as np  # noqa: E402
def _ml(cards, keys):
    return int(np.median([len(cards[k] or "") for k in keys]))
print(f"\nDONE qwen: nuwa={len(nuwa)} aggro={len(aggro)} shared={len(cache)}", flush=True)
print(f"  [char median] nuwa={_ml(nuwa, devs)} aggro={_ml(aggro, devs)} "
      f"shared={int(np.median([len(v or '') for v in cache.values()]))}", flush=True)
empties = [d for d in devs if not (nuwa.get(d) and aggro.get(d))]
print(f"  empties: {empties or 'none'}", flush=True)
sd = devs[0]
print(f"\n---- sample nuwa[{sd}] head ----\n{(nuwa[sd] or '')[:400]}", flush=True)
print(f"\n---- sample shared head ----\n{(list(cache.values())[0] or '')[:400]}", flush=True)
print(f"\nsaved -> {NUWAC_Q.name} + {STEP2C_Q.name} + {SHAREDC_Q.name}", flush=True)
