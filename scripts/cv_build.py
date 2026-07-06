"""Build the CV (Cross Validated / statistics) card files for the CANONICAL cmd_gate/cmd_attack2afc harness.

Reuses cv_pilot's Posts.xml parse + nuwa distillation (IDENTICAL prompts -> llm_cache hits, ~$0) so the cards match
the frozen pilot exactly. Writes to data/se/:
  cv_cmd_pool.json          {"pool": {author: {ref, raw}}}   ref/raw = held-answer writing samples (leak-disjoint from card)
  cv_cmd_nuwa.json          {"nuwa": {author: card}, "aggro": {author: card}}   aggro=nuwa (synth_shared genericizes)
  cmd_shared_cards_cv.json  {f"k{K}_s{SEED}_{cid}": shared_card}   built via CG.make_groups + synth_shared (GROUP=random)

Grouping: make_groups(nuwa, users, K, SEED) with GROUP=random == cv_pilot's group_random(users,K,0) -> SAME layout ->
synth_shared inputs byte-identical -> cache hits -> shared cards match the pilot exactly.

Run:  DATASET=cv python scripts/cv_build.py <stats.7z>   [NEXP=26 KS=6,8 SEED=0 REF_FRAG=78]
Run from project root (clean cwd, NOT scratchpad).
"""
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("DATASET", "cv")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cv_pilot as CVP      # noqa: E402  (load, nuwa_extract, nuwa_assemble, plain, NEXP/NCARD/NHELD)
import cmd_gate as CG       # noqa: E402  (SE dir, make_groups, synth_shared)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

KS = [int(x) for x in os.environ.get("KS", "6,8").split(",")]
SEED = int(os.environ.get("SEED", 0))
FRAG = int(os.environ.get("REF_FRAG", 78))     # per-answer fragment so a 3-frag ref survives the 250 cap (wpse-style)


def main():
    archive = Path(sys.argv[1])
    q_title, q_body, q_acc, a_body, cohort = CVP.load(archive)
    users = sorted(cohort, key=lambda u: -len(cohort[u]))[:CVP.NEXP]
    print(f"cv_build: cohort(>= {CVP.NCARD+CVP.NHELD} gold)={len(cohort)} using {len(users)} experts | KS={KS} seed={SEED}", flush=True)

    # indiv nuwa cards (IDENTICAL to cv_pilot -> llm_cache hits)
    nuwa = {u: CVP.nuwa_assemble(CVP.nuwa_extract([b for (_, _, _, b) in cohort[u][:CVP.NCARD]])) for u in users}

    # pool: ref (multi-fragment held-answer writing sample) + raw (a later held answer), both leak-disjoint from card[:12]
    pool = {}
    for u in users:
        held = [CVP.plain(b) for (_, _, _, b) in cohort[u][CVP.NCARD:]]     # all held answers (index >= 12)
        ref = " || ".join(h[:FRAG] for h in held[:3])
        raw = held[3] if len(held) >= 4 else held[-1]
        pool[u] = {"ref": ref, "raw": raw}

    se = CG.SE
    se.mkdir(parents=True, exist_ok=True)
    (se / "cv_cmd_pool.json").write_text(json.dumps({"pool": pool}, ensure_ascii=False, indent=1), encoding="utf-8")
    (se / "cv_cmd_nuwa.json").write_text(json.dumps({"nuwa": nuwa, "aggro": nuwa}, ensure_ascii=False, indent=1), encoding="utf-8")

    # shared (pooled) cards for each K via the canonical grouping (GROUP=random -> same layout as the pilot)
    shared = {}
    for K in KS:
        grp, byc = CG.make_groups(nuwa, users, K, SEED)
        for cid, mem in byc.items():
            shared[f"k{K}_s{SEED}_{cid}"] = CG.synth_shared([nuwa[u] for u in mem])
        print(f"  k{K}: {len(byc)} clusters (sizes {sorted(len(m) for m in byc.values())})", flush=True)
    (se / "cmd_shared_cards_cv.json").write_text(json.dumps(shared, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved -> data/se/{{cv_cmd_pool,cv_cmd_nuwa,cmd_shared_cards_cv}}.json  ({len(shared)} shared cards)", flush=True)


if __name__ == "__main__":
    main()
