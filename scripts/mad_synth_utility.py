"""Stage 1a: build utility-preserving shared-card VARIANTS for 20-MAD k8/s0, into ISOLATED cache
files (NEVER touches baseline cmd_shared_cards_mad.json). Prompts are FROZEN here (pre-registered;
do not edit after seeing results — that would be p-hacking). cmd_gate.py is left UNMODIFIED.

Three variants isolate "substance vs length vs wording":
  neutral : relax length + keep concrete shared substance, NO result-relevant leading words   (PRIMARY candidate)
  util    : neutral + explicit "heuristics/patterns/criteria/domain knowledge" words           (wording-inflation probe)
  pad     : baseline card elaborated to ~2x length with generic filler, NO new specifics        (length control / blank-card)

Run:  DATASET=mad python scripts/mad_synth_utility.py
Then eval:  DATASET=mad python scripts/mad_util_variants.py
"""
import os
import sys
import json
import statistics
from pathlib import Path

os.environ.setdefault("DATASET", "mad")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cmd_gate as CG          # noqa: E402
import deid_enron as de        # noqa: E402  (de.pool parallel map)
from src.llm import chat       # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GEN = "deepseek-chat"
K = int(os.environ.get("K", 8))              # M2 headline k=8; env-overridable for the neutral-transition backfill (k4 etc.)
S = int(os.environ.get("SEED", 0))           # seed 0 = Stage-1 headline; seed 1 = tighten-CI backfill
ONLY = os.environ.get("ONLY", "")            # e.g. ONLY=neutral -> skip util/pad (neutral-transition build; saves synth $)

# ---- FROZEN prompts (M4: exact strings pre-registered) ----
_SYS_KEEP = ("You distill ONE shared skill card capturing the full shared working knowledge and practices common to "
             "several colleagues, removing only what identifies any single one of them.")


def synth_neutral(member_cards):
    body = "\n\n---\n\n".join(member_cards)
    msg = [{"role": "system", "content": _SYS_KEEP},
           {"role": "user", "content": f"Skill cards from several colleagues:\n\n{body}\n\nWrite ONE shared skill card "
            "that captures everything they have in COMMON — their shared knowledge, working approaches, and practices "
            "— while removing any phrasing or detail unique to any single person. Preserve the concrete shared "
            "substance; do not compress or over-abstract it into generic platitudes. Keep it comparable in length to "
            "ONE of the input cards — do not bloat or pad it. It must read as if it could "
            "belong to any of them equally. Output ONLY the shared card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=1300) or ""


def synth_util(member_cards):
    body = "\n\n---\n\n".join(member_cards)
    msg = [{"role": "system", "content": _SYS_KEEP},
           {"role": "user", "content": f"Skill cards from several colleagues:\n\n{body}\n\nWrite ONE shared skill card "
            "that captures everything they have in COMMON — their shared knowledge, working approaches, practices, "
            "specific heuristics, recurring patterns, decision criteria, and domain knowledge — while removing any "
            "phrasing or detail unique to any single person. Preserve the concrete shared substance; do not compress "
            "or over-abstract it into generic platitudes. Keep it comparable in length to ONE of the input cards — do "
            "not bloat or pad it. It must read as if it could belong to any of them equally. "
            "Output ONLY the shared card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=1300) or ""


def synth_pad(baseline_card):
    msg = [{"role": "system", "content": "You lightly expand a skill card to be more verbose WITHOUT adding any new specific content."},
           {"role": "user", "content": f"Here is a shared skill card:\n\n{baseline_card}\n\nRewrite it to be roughly "
            "twice as long by elaborating each point with generic, standard professional phrasing. Do NOT add any new "
            "specific facts, heuristics, criteria, names, numbers, or details — only pad the existing content with "
            "generic elaboration. Output ONLY the expanded card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=1300) or ""


VARIANTS = [("neutral", synth_neutral, "members"),
            ("util", synth_util, "members"),
            ("pad", synth_pad, "base")]


def main():
    pool, authors, nuwa, aggro, _r, _w = CG.load()
    grp, byc = CG.make_groups(aggro, authors, K, S)
    clusters = sorted(byc)
    base_path = CG.SHAREDC                     # dataset-aware: mad=cmd_shared_cards_mad.json, enron=cmd_shared_cards.json
    if not base_path.exists():
        sys.exit(f"baseline {base_path} missing — build it first (cmd_gate/cmd_openworld/cmd_utility).")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    missb = [c for c in clusters if f"k{K}_s{S}_{c}" not in base]
    if missb:
        sys.exit(f"baseline missing clusters {missb}")
    print(f"{len(clusters)} clusters k{K}_s{S} | building util/neutral/pad into ISOLATED files | baseline untouched", flush=True)

    variants = [v for v in VARIANTS if not ONLY or v[0] in ONLY.split(",")]
    rows = {}
    for variant, fn, src in variants:
        outpath = CG.SE / f"{base_path.stem}__{variant}.json"    # mad -> cmd_shared_cards_mad__neutral.json; enron -> cmd_shared_cards__neutral.json
        cache = json.loads(outpath.read_text(encoding="utf-8")) if outpath.exists() else {}
        plan = []
        for c in clusters:
            ck = f"k{K}_s{S}_{c}"
            if ck in cache:
                continue
            plan.append((ck, [aggro[a] for a in byc[c]] if src == "members" else base[ck]))
        if plan:
            print(f"  synth {variant}: {len(plan)} cards (deepseek) ...", flush=True)
            for (ck, _), card in zip(plan, de.pool(lambda p, fn=fn: fn(p[1]), plan)):
                cache[ck] = card
            outpath.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        lens = sorted(len(cache[f"k{K}_s{S}_{c}"]) for c in clusters)
        rows[variant] = int(statistics.median(lens))
        print(f"  {variant:8s} -> {outpath.name}  median_len={rows[variant]}  range[{lens[0]},{lens[-1]}]", flush=True)

    blens = sorted(len(base[f"k{K}_s{S}_{c}"]) for c in clusters)
    print(f"\n  baseline median_len={int(statistics.median(blens))}  (indiv nuwa ~4967 for reference)", flush=True)
    print(f"  M5 length gate: median vs baseline {int(statistics.median(blens))} -> "
          f"neutral={rows.get('neutral','-')} util={rows.get('util','-')} pad={rows.get('pad','-')}", flush=True)


if __name__ == "__main__":
    main()
