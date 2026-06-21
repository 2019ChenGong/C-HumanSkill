"""Probe whether the k=8 utility win is robust to the shared-card synthesis prompt (terse vs rich)."""
import os
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
import enron_nuwa as NW  # noqa: E402
import cmd_gate as CG  # noqa: E402
from src.llm import chat  # noqa: E402
from src.attrib_metrics import cluster_mean_ci  # noqa: E402

SE = ROOT / "data" / "enron"
RES = ROOT / "results"
SHAREDC = SE / "cmd_shared_cards.json"
ALTC = SE / "cmd_shared_cards_alt.json"
GEN = "deepseek-chat"
SEED = 0
K = 8
T = de.TASKS


def alt_synth(member_cards):
    """Deliberately TERSE/minimal synthesis — the opposite style of the original 'cognitive OS' prompt."""
    body = "\n\n---\n\n".join(member_cards)
    msg = [{"role": "system", "content": "You extract the few shared essentials common to several colleagues."},
           {"role": "user", "content": f"Cards from several colleagues:\n\n{body}\n\nList the MAX 5 most important "
            "working/decision points they have in COMMON, as a short flat bullet list, total <=120 words. No headers, "
            "no sub-structure, no detail unique to any one person. Output ONLY the short list."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=300) or ""


def tl(c):
    return len(NW.ENC.encode(c or ""))


def main():
    docs, authors, nuwa, aggro, ref, raw_tgt = CG.load()
    cache = json.loads(SHAREDC.read_text(encoding="utf-8"))
    alt = json.loads(ALTC.read_text(encoding="utf-8")) if ALTC.exists() else {}
    grp, byc = CG.make_groups(aggro, authors, K, SEED)         # SAME clusters as the original shared@8
    plan = [(cid, [aggro[a] for a in mem]) for cid, mem in byc.items() if f"k{K}_s{SEED}_{cid}" not in alt]

    if os.environ.get("PILOT_DRYRUN"):
        n_draft = 116 * len(T) + 2 * len(byc) * len(T)         # indiv + (orig + alt) sh8 per cluster-task
        n_judge = 2 * 116 * len(T)                             # alt-indiv + alt-orig
        cost = n_draft * 1700 / 1e6 * 0.6 + n_judge * 1400 / 1e6 * 1.0
        print(f"DRYRUN: synth {len(plan)} alt cards; drafts≈{n_draft}; judge≈{n_judge} (alt-indiv + alt-orig); est ~${cost:.1f}", flush=True)
        return

    if plan:
        print(f"synth {len(plan)} TERSE alt shared@8 cards ...", flush=True)
        for (cid, _), card in zip(plan, de.pool(lambda pc: alt_synth(pc[1]), plan)):
            alt[f"k{K}_s{SEED}_{cid}"] = card
        ALTC.write_text(json.dumps(alt, ensure_ascii=False), encoding="utf-8")

    def orig(a):
        return cache[f"k{K}_s{SEED}_{grp[a]}"]

    def altc(a):
        return alt[f"k{K}_s{SEED}_{grp[a]}"]

    print(f"\n[tok] indiv={int(np.median([tl(nuwa[a]) for a in authors]))} "
          f"orig_sh8={int(np.median([tl(orig(a)) for a in authors]))} "
          f"alt_sh8={int(np.median([tl(altc(a)) for a in authors]))}", flush=True)

    # drafts: indiv (per author), orig/alt sh8 (per cluster)
    djobs = [("indiv", a, t) for a in authors for t in range(len(T))]
    for arm in ("orig", "alt"):
        seen = set()
        for a in authors:
            if grp[a] not in seen:
                seen.add(grp[a]); djobs += [(arm, a, t) for t in range(len(T))]
    print(f"building {len(djobs)} drafts ...", flush=True)
    cf = {"indiv": nuwa, "orig": None, "alt": None}
    card_for = lambda arm, a: nuwa[a] if arm == "indiv" else (orig(a) if arm == "orig" else altc(a))
    D = {}
    for (arm, a, t), txt in zip(djobs, de.pool(lambda j: NW.draft(card_for(j[0], j[1]), T[j[2]]), djobs)):
        rep = a if arm == "indiv" else next(b for b in authors if grp[b] == grp[a])
        D[(arm, rep, t)] = txt

    def dr(arm, a, t):
        rep = a if arm == "indiv" else next(b for b in authors if grp[b] == grp[a])
        return D[(arm, rep, t)]

    def judge(x, y, a, t):
        return NW.quality(T[t], dr(x, a, t), dr(y, a, t), f"synthp-{x}{y}-{a}-{t}")

    units = [(a, t) for a in authors for t in range(len(T))]
    g = [f"{grp[a]}" for (a, t) in units]
    print("\n=== alt(TERSE) shared@8 utility (CI resamples clusters, seed 0) ===", flush=True)
    out = {}
    for x, y in [("alt", "indiv"), ("alt", "orig")]:
        v = [judge(x, y, a, t) for (a, t) in units]
        ci = cluster_mean_ci(v, g, seed=0); fl = "  <-EXCL0" if (ci[0] > 0 or ci[1] < 0) else ""
        out[f"{x}-{y}"] = {"diff": round(float(np.mean(v)), 3), "ci": ci}
        print(f"  {x} vs {y:6s} = {np.mean(v):+.3f} CI{ci}  (n_cl={len(set(g))}){fl}", flush=True)
    out["note"] = ("k=8 win prompt-dependence: TERSE synth (max5 bullets, <=120w) vs original 'cognitive OS'. "
                   "alt-indiv survives -> robust (pooled content helps); vanishes -> orig win was rich-structuring artifact. "
                   "Reference: orig sh8-indiv pooled +0.193 (cmd_k8_probe).")
    (RES / "cmd_synth_probe.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> results/cmd_synth_probe.json", flush=True)


if __name__ == "__main__":
    main()
