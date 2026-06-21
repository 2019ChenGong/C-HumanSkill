"""Build nuwa (individual) + aggro (de-identified) cards for the full 128-developer 20-MAD SeaMonkey set."""
import os
import re
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de            # noqa: E402  (pool)
import enron_step2 as ES          # noqa: E402  (aggro_card)
from mad_nuwa_step2 import nuwa_extract, nuwa_assemble  # noqa: E402  (SAME nuwa construction as the audited run)

MAD = ROOT / "data" / "20mad"
POOL = MAD / "mad_cmd_pool.json"
NUWAC = MAD / "mad_cmd_nuwa.json"
STEP2C = MAD / "mad_cmd_step2.json"
N_TRAIN, N_REF, N_TGT = 18, 6, 2
BAR = N_TRAIN + N_REF + N_TGT       # 26
WS = re.compile(r"\s+")


def main():
    pool = json.loads(POOL.read_text(encoding="utf-8"))["pool"]
    devs = [d for d in pool if len(pool[d]["card_comments"]) >= BAR]
    tr = {d: pool[d]["card_comments"][:N_TRAIN] for d in devs}
    print(f"FULL SeaMonkey set: N={len(devs)} devs (>= {BAR} card_comments)", flush=True)

    if os.environ.get("PILOT_DRYRUN"):
        import tiktoken
        ENC = tiktoken.get_encoding("cl100k_base")
        tl = lambda c: len(ENC.encode(c or ""))
        ev = sum(tl("\n\n---\n\n".join(tr[d][:12])) for d in devs)                    # nuwa_extract input
        n = len(devs)
        ds_in = ev + n * 600 + n * 850          # extract_in + assemble_in(~notes 600) + aggro_in(~nuwa 850)
        ds_out = n * 600 + n * 900 + n * 700    # extract_out + assemble_out + aggro_out
        cost = ds_in / 1e6 * 0.28 + ds_out / 1e6 * 1.10
        print(f"DRYRUN: {n} devs × (nuwa_extract + nuwa_assemble + aggro_card) = {3*n} deepseek calls; "
              f"est deepseek in~{ds_in/1e6:.2f}M out~{ds_out/1e6:.2f}M -> ~${cost:.2f}. Opus attack FREE later.", flush=True)
        print(f"  -> {NUWAC.name} (nuwa) + {STEP2C.name} (aggro)", flush=True)
        return

    # ---- nuwa (2-call), cache-aware ----
    nuwa = json.loads(NUWAC.read_text(encoding="utf-8"))["nuwa"] if NUWAC.exists() else {}
    miss = [d for d in devs if d not in nuwa]
    if miss:
        print(f"building {len(miss)} nuwa cards (2-call deepseek) ...", flush=True)
        for d, card in zip(miss, de.pool(lambda d: nuwa_assemble(nuwa_extract(tr[d])), miss)):
            nuwa[d] = card
        NUWAC.write_text(json.dumps({"nuwa": nuwa}, ensure_ascii=False), encoding="utf-8")

    # ---- aggro (1-call) on nuwa, cache-aware ----
    step2 = json.loads(STEP2C.read_text(encoding="utf-8")) if STEP2C.exists() else {}
    aggro = step2.get("aggro", {})
    miss = [d for d in devs if d not in aggro]
    if miss:
        print(f"building {len(miss)} aggro cards (1-call deepseek) ...", flush=True)
        for d, card in zip(miss, de.pool(lambda d: ES.aggro_card(nuwa[d]), miss)):
            aggro[d] = card
        step2["aggro"] = aggro
        STEP2C.write_text(json.dumps(step2, ensure_ascii=False), encoding="utf-8")

    import numpy as np
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    tl = lambda c: len(ENC.encode(c or ""))
    print(f"\nDONE: nuwa={len(nuwa)} aggro={len(aggro)} (target {len(devs)})", flush=True)
    print(f"  [tok median] nuwa={int(np.median([tl(nuwa[d]) for d in devs]))} "
          f"aggro={int(np.median([tl(aggro[d]) for d in devs]))}", flush=True)
    print(f"saved -> {NUWAC.name} + {STEP2C.name}", flush=True)


if __name__ == "__main__":
    main()
