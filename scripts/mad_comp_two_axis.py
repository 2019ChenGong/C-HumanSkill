"""20-MAD harness: lineup / detective / resolution-prediction and the shared 20-MAD constants."""
import os
import re
import sys
import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402  (pool, _content_vec, _cosine)
from src.llm import chat, sample_one  # noqa: E402
from src.attrib_metrics import cluster_mean_ci, cluster_paired_diff_ci  # noqa: E402

GEN = "deepseek-chat"
HEADLINE = "claude-haiku-4-5"
SECOND = "deepseek-chat"
MAD = ROOT / "data" / "20mad"
POOL = MAD / "solve_pool.json"
COMPC = MAD / "comp_cards.json"
DEIDC = MAD / "comp_deid_cards.json"
N_TRAIN = 18          # comments[:18] distill the card
N_REF = 6             # comments[18:24] = candidate reference (attacker corpus)
N_TGT = 2             # comments[24:26] = positive-control comment targets
B_CARDS = int(os.environ.get("B_CARDS", 2))
SUBSET = 12
REF_CHARS = 900
RESAMPLE = int(os.environ.get("RESAMPLE", 2))
SEED = 0
CLASSES = ["FIXED", "WONTFIX", "INVALID", "DUPLICATE", "WORKSFORME"]
WS = re.compile(r"\s+")


def comp_card(comments):
    body = "\n- ".join(comments[:SUBSET])
    msg = [{"role": "system", "content": "You distill a reusable triage/review PLAYBOOK from a developer's past "
            "bug-tracker comments -- the transferable way THIS developer handles bugs."},
           {"role": "user", "content":
            f"Below are bug-tracker comments by the SAME developer:\n\n- {body}\n\nDistill a concise skill card "
            "(5-8 bullets) of the TRANSFERABLE judgment a colleague could APPLY to handle NEW bugs the way this "
            "developer does -- how they judge severity/priority, when they resolve FIXED vs WONTFIX vs INVALID vs "
            "DUPLICATE vs WORKSFORME, what they check before deciding, how they push back or ask for info. Ground "
            "each bullet in the CHARACTERISTIC way THIS developer decides (e.g. 'when X, they tend to ...'), NOT "
            "generic best-practice: if two developers would write the same bullet it is too generic. Do NOT include "
            "names or writing-style description -- capture decision substance. Output ONLY the card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=420) or ""


def deid_comp_card(card):
    msg = [{"role": "system", "content": "You generalize a skill card so it no longer reveals the developer's "
            "specific software module, component, or technical domain."},
           {"role": "user", "content":
            f"Skill card describing how a developer triages bugs:\n\n{card}\n\nRewrite it to REMOVE everything that "
            "reveals this developer's specific MODULE, COMPONENT, technical DOMAIN, or the particular kinds of bugs/"
            "subsystems they work on. Keep ONLY generic, transferable bug-triage and decision heuristics that could "
            "apply to a developer on ANY component. Strip component names, technology nouns, and domain markers; "
            "generalize each bullet to its domain-neutral form while keeping it a concrete, usable heuristic (do NOT "
            "collapse to empty platitudes). Keep the same bullet structure. Output ONLY the rewritten card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=420) or ""


def _subset(items, seed):
    if len(items) <= SUBSET:
        return items
    idx = np.random.default_rng(seed).permutation(len(items))[:SUBSET]
    return [items[i] for i in sorted(idx)]


def build_cards(pool, devs):
    if COMPC.exists() and DEIDC.exists() and not os.environ.get("REBUILD"):
        return json.loads(COMPC.read_text(encoding="utf-8")), json.loads(DEIDC.read_text(encoding="utf-8"))
    tr = {d: pool[d]["card_comments"][:N_TRAIN] for d in devs}
    print(f"distilling {B_CARDS} comp cards/dev (deepseek) ...", flush=True)
    comp = {d: de.pool(lambda s: comp_card(_subset(tr[d], s)), list(range(B_CARDS))) for d in devs}
    COMPC.write_text(json.dumps(comp, ensure_ascii=False), encoding="utf-8")
    print("building deidcomp (module-stripped) cards (from card 0) ...", flush=True)
    deid = dict(zip(devs, de.pool(lambda d: deid_comp_card(comp[d][0]), devs)))
    DEIDC.write_text(json.dumps(deid, ensure_ascii=False), encoding="utf-8")
    return comp, deid


def detective(target, refs, model, salt):
    block = "\n".join(f"[{slot}] {rt}" for slot, _a, rt in refs)
    msg = [{"role": "system", "content": "You identify which candidate developer authored a target, from how the "
            "developer thinks, triages, and decides -- NOT from the topic/component (components overlap)."},
           {"role": "user", "content":
            f"TARGET (written/derived from ONE developer):\n{target}\n\nCANDIDATES (each shown by a sample of their "
            f"own bug comments):\n{block}\n\nWhich candidate [number] is the SAME developer as the TARGET? Match by "
            "triage reasoning, priorities, and decision style, NOT topic. Answer ONLY the number."}]
    out = sample_one(msg, model, s=0, temperature=0.7, max_tokens=6, salt=salt) or ""
    nums = [int(x) for x in re.findall(r"\d+", out) if 1 <= int(x) <= len(refs)]
    return refs[nums[-1] - 1][1] if nums else None


def topic_pick(target, refs):
    qt = de._content_vec(target)
    best, bs = None, 0.0
    for _slot, a, rt in refs:
        s = de._cosine(qt, de._content_vec(rt))
        if s > bs:
            bs, best = s, a
    return best


def lineup(devs, ref, target_dev, tag):
    order = sorted(devs, key=lambda b: hashlib.sha1(f"{tag}-{target_dev}-{b}".encode()).hexdigest())
    return [(i + 1, b, ref[b]) for i, b in enumerate(order)]


def predict_res(card, report, stub):
    sys_m = "You triage software bugs. Predict the most likely RESOLUTION."
    prof = f"Developer triage profile:\n{card}\n\n" if card else ""
    out = (chat([{"role": "system", "content": sys_m},
                 {"role": "user", "content": f"{prof}Bug:\n{stub}\n{report}\n\nWhat is the most likely resolution? "
                  f"Answer ONLY ONE of: {', '.join(CLASSES)}."}], model=GEN, temperature=0.0, max_tokens=6) or "").upper()
    for c in CLASSES:
        if c in out:
            return c
    return None


def main():
    pool = json.loads(POOL.read_text(encoding="utf-8"))["pool"]
    devs = [d for d in pool if len(pool[d]["card_comments"]) >= N_TRAIN + N_REF + N_TGT]
    N = len(devs)
    chance = 1.0 / N
    print(f"devs={N} chance={chance:.3f} train={N_TRAIN} ref={N_REF} K(full)={N} B={B_CARDS} resample={RESAMPLE}", flush=True)
    ref = {d: WS.sub(" ", " || ".join(pool[d]["card_comments"][N_TRAIN:N_TRAIN + N_REF]))[:REF_CHARS] for d in devs}

    comp, deid = build_cards(pool, devs)

    def mpc(cards):
        vs = [de._content_vec(c) for c in cards]
        s = [de._cosine(vs[i], vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
        return round(float(np.mean(s)), 3)
    vac_c, vac_d = mpc([comp[d][0] for d in devs]), mpc([deid[d] for d in devs])

    def module_legible(card, salt):
        out = (chat([{"role": "user", "content": f"Skill card:\n{card}\n\nFrom this card alone, can you tell what "
                      "specific software module, component, or technical area this developer works on? Answer ONLY 'YES' or 'NO'."}],
                    model=HEADLINE, temperature=0.0, max_tokens=4) or "").strip().lower()
        return 1.0 if out.startswith("y") else 0.0
    ml_c = float(np.mean(de.pool(lambda d: module_legible(comp[d][0], f"mc-{d}"), devs)))
    ml_d = float(np.mean(de.pool(lambda d: module_legible(deid[d], f"md-{d}"), devs)))
    print(f"\n[VACUITY] pairwise cosine comp={vac_c} deidcomp={vac_d}", flush=True)
    print(f"[MODULE-LEGIBLE] 'can you tell the component?' comp={ml_c:.2f} deidcomp={ml_d:.2f} (deid<comp => module stripped)", flush=True)

    # anonymity units
    units = []   # (arm, dev, k, target)
    for d in devs:
        for j in range(N_TGT):
            units.append(("comment", d, j, WS.sub(" ", pool[d]["card_comments"][N_TRAIN + N_REF + j])[:500]))
        for b in range(B_CARDS):
            units.append(("comp", d, b, comp[d][b]))
        units.append(("deidcomp", d, 0, deid[d]))
    ARMS = ["comment", "comp", "deidcomp"]

    if os.environ.get("PILOT_DRYRUN"):
        per = {arm: sum(1 for u in units if u[0] == arm) for arm in ARMS}
        nb = sum(len(pool[d]["solved_bugs"]) for d in devs)
        print(f"\nDRYRUN anon units/arm={per} total={len(units)}; detective calls={len(units)}*{RESAMPLE}*2={len(units)*RESAMPLE*2}; "
              f"util drafts(deepseek)= {nb}*3 = {nb*3}.", flush=True)
        return

    def run_det(i):
        arm, d, k, tgt = units[i]
        lu = lineup(devs, ref, d, f"{arm}-{k}")
        ph = [detective(tgt, lu, HEADLINE, f"{arm}-{d}-{k}-h{r}") for r in range(RESAMPLE)]
        pd = [detective(tgt, lu, SECOND, f"{arm}-{d}-{k}-d{r}") for r in range(RESAMPLE)]
        return i, float(np.mean([p == d for p in ph])), float(np.mean([p == d for p in pd])), float(topic_pick(tgt, lu) == d)
    res = {}
    for i, sh, sd, tp in de.pool(run_det, list(range(len(units)))):
        res[i] = (sh, sd, tp)
    print("anonymity detectives done", flush=True)

    def arm_idx(arm):
        return [i for i in range(len(units)) if units[i][0] == arm]
    def grp(idxs):
        return [units[i][1] for i in idxs]

    summary = {}
    print(f"\n=== ANONYMITY single-shot (full K={N}, chance={chance:.3f}, soft over {RESAMPLE}) ===", flush=True)
    print(f"{'arm':9s} {'haiku':>20s} {'deepseek*':>11s} {'topic':>8s}", flush=True)
    for arm in ARMS:
        idxs = arm_idx(arm); g = grp(idxs)
        h = [res[i][0] for i in idxs]; d_ = [res[i][1] for i in idxs]; t = [res[i][2] for i in idxs]
        ci = cluster_mean_ci(h, g, seed=SEED)
        summary[arm] = {"n": len(idxs), "haiku": round(float(np.mean(h)), 3), "haiku_ci": ci,
                        "deepseek": round(float(np.mean(d_)), 3), "topic": round(float(np.mean(t)), 3)}
        print(f"{arm:9s} {np.mean(h):.3f} CI{ci!s:>13s} {np.mean(d_):>11.3f} {np.mean(t):>8.3f}  (n={len(idxs)})", flush=True)
    # comp vs deidcomp paired (same devs, card 0 only for comp to match deidcomp 1/dev)
    ci_comp0 = [i for i in arm_idx("comp") if units[i][2] == 0]
    di = arm_idx("deidcomp")
    if grp(ci_comp0) == grp(di):
        gd = cluster_paired_diff_ci([res[i][0] for i in ci_comp0], [res[i][0] for i in di], grp(di), seed=SEED)
        print(f"  comp(card0) - deidcomp (haiku) = {gd['diff']:+.3f} CI{gd['ci']} "
              f"{'(de-id REDUCES re-id: handle=separable module)' if gd['ci'][0] > 0 else '(de-id does NOT reduce: deep behavioral fingerprint)'}", flush=True)
    print(f"  TOPIC-DECOMP comp: haiku={summary['comp']['haiku']:.3f} topic={summary['comp']['topic']:.3f}", flush=True)

    out = {"N": N, "chance": round(chance, 4), "summary": summary,
           "vacuity": {"comp": vac_c, "deidcomp": vac_d},
           "module_legible": {"comp": round(ml_c, 3), "deidcomp": round(ml_d, 3)}}

    # === UTILITY: objective resolution prediction ===
    if not os.environ.get("SKIP_UTIL"):
        ubugs = [(d, b) for d in devs for b in pool[d]["solved_bugs"]]
        def do_u(job):
            (d, b), arm = job
            card = {"nocard": None, "comp": comp[d][0], "deidcomp": deid[d]}[arm]
            p = predict_res(card, b.get("report", ""), b.get("stub", ""))
            return job, 1.0 if p == b["resolution"] else 0.0
        UARMS = ["nocard", "comp", "deidcomp"]
        acc = {arm: [] for arm in UARMS}; ug = {arm: [] for arm in UARMS}
        for ((d, b), arm), hit in de.pool(do_u, [((d, b), arm) for (d, b) in ubugs for arm in UARMS]):
            acc[arm].append(hit); ug[arm].append(d)
        print(f"\n=== UTILITY resolution-prediction accuracy (objective; n={len(ubugs)} bugs, {N} devs) ===", flush=True)
        ures = {}
        for arm in UARMS:
            ci = cluster_mean_ci(acc[arm], ug[arm], seed=SEED)
            ures[arm] = {"acc": round(float(np.mean(acc[arm])), 3), "ci": ci}
            print(f"  {arm:9s} acc={np.mean(acc[arm]):.3f} CI{ci}", flush=True)
        for x, y in [("comp", "nocard"), ("deidcomp", "nocard"), ("deidcomp", "comp")]:
            r = cluster_paired_diff_ci(acc[x], acc[y], ug[x], seed=SEED)
            fl = "  <-EXCL0" if (r["ci"][0] > 0 or r["ci"][1] < 0) else ""
            print(f"    {x}-{y} = {r['diff']:+.3f} CI{r['ci']}{fl}", flush=True)
        out["utility"] = {"n_bugs": len(ubugs), "per_arm": ures}

    out["note"] = ("20-MAD third-domain competence-card two-axis. comp vs deidcomp(module-stripped) anonymity + "
                   "objective resolution-prediction utility. Compare to enron_comp_*/wpse_competence.")
    (ROOT / "results" / "mad_comp_two_axis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> results/mad_comp_two_axis.json", flush=True)


if __name__ == "__main__":
    main()
