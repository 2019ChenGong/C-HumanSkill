"""20-MAD pipeline: distill a developer's bug-triage cognitive-OS card (its nuwa_extract/nuwa_assemble are reused by mad_cmd_build)."""
import os
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("K", "5")                       # archpool cluster count (read by enron_archpool at call time)
import deid_enron as de                               # noqa: E402  (pool, _content_vec, _cosine)
import mad_comp_two_axis as MC                         # noqa: E402  (lineup/detective/topic_pick/predict_res/harness)
import enron_step2 as ES                               # noqa: E402  (naive_card/deid4_card/aggro_card -- domain-agnostic)
import enron_archpool as AP                            # noqa: E402  (cluster_authors/cluster_template/pool_card)
from src.llm import chat                               # noqa: E402
from src.attrib_metrics import cluster_mean_ci, cluster_paired_diff_ci  # noqa: E402

AP.K = int(os.environ.get("K", 5))                    # rebind archpool cluster count
GEN = MC.GEN                                           # deepseek-chat
HEADLINE = MC.HEADLINE                                 # claude-haiku-4-5
SECOND = MC.SECOND                                     # deepseek-chat
MAD = ROOT / "data" / "20mad"
POOL = MAD / "solve_pool.json"
NUWAC = MAD / "mad_nuwa_cards.json"
STEP2C = MAD / "mad_step2_cards.json"
N_TRAIN, N_REF, N_TGT, REF_CHARS = MC.N_TRAIN, MC.N_REF, MC.N_TGT, MC.REF_CHARS   # 18 / 6 / 2 / 900
NUWA_EVID = 12                                          # comments[:12] feed nuwa_extract (match Enron evidence budget)
RESAMPLE = int(os.environ.get("RESAMPLE", 2))
SEED = 0
WS = MC.WS
CLASSES = MC.CLASSES


# ---------------- STEP 1: nuwa cognitive-OS for bug triage (2-call, FAITHFUL) ----------------
def nuwa_extract(comments):
    body = "\n\n---\n\n".join(comments[:NUWA_EVID])
    msg = [{"role": "system", "content": "You reverse-engineer a software developer's COGNITIVE OPERATING SYSTEM for "
            "triaging and resolving bugs, from their bug-tracker comments."},
           {"role": "user", "content": f"Bug-tracker comments by ONE developer:\n\n{body}\n\nIdentify the underlying "
            "DECISION FRAMEWORKS, mental models, triage heuristics, characteristic moves, and failure modes these "
            "reveal -- when they resolve FIXED vs WONTFIX vs INVALID vs DUPLICATE vs WORKSFORME, what they check before "
            "deciding, how they push back or ask for info -- and tie each to the evidence. Output terse notes grouped: "
            "Frameworks / Heuristics / Characteristic moves / Failure modes. Derive only from the texts."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=600) or ""


def nuwa_assemble(notes):
    msg = [{"role": "system", "content": "You compile a REUSABLE bug-triage cognitive operating system a colleague "
            "could EXECUTE."},
           {"role": "user", "content": f"Notes on how a developer triages and decides on bugs:\n\n{notes}\n\nCompile a "
            "COGNITIVE OPERATING SYSTEM card a colleague could EXECUTE -- second-person, situation-triggered procedures, "
            "NOT a biography. Sections: [Triage Protocol] step-by-step approach to a new bug; [Decision Frameworks & "
            "Mental Models]; [Heuristics] each as 'When X -> do Y, watch for Z'; [Characteristic Moves]; [Failure "
            "Modes]. Write each as an EXECUTABLE instruction. Keep the technical decision substance (you MAY reference "
            "the kinds of components/bugs this developer handles -- that is their competence), but do NOT name the "
            "developer. ~600-900 words of card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=1100) or ""


def main():
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    def tlen(c):
        return len(ENC.encode(c or ""))

    pool = json.loads(POOL.read_text(encoding="utf-8"))["pool"]
    devs = [d for d in pool if len(pool[d]["card_comments"]) >= N_TRAIN + N_REF + N_TGT]
    N = len(devs); chance = 1.0 / N
    print(f"devs={N} chance={chance:.3f} train={N_TRAIN} ref={N_REF} tgt={N_TGT} K(cluster)={AP.K} resample={RESAMPLE}", flush=True)
    ref = {d: WS.sub(" ", " || ".join(pool[d]["card_comments"][N_TRAIN:N_TRAIN + N_REF]))[:REF_CHARS] for d in devs}
    tr = {d: pool[d]["card_comments"][:N_TRAIN] for d in devs}

    # ---- DRYRUN: estimate BEFORE building anything (cost discipline) ----
    if os.environ.get("PILOT_DRYRUN"):
        tl = lambda c: len(ENC.encode(c or ""))
        s = devs[0]
        lu = MC.lineup(devs, ref, s, "dry-0")
        det_in = tl("\n".join(f"[{slot}] {rt}" for slot, _a, rt in lu)) + 400   # +target +prompt (real 24-ref block)
        nbugs = sum(len(pool[d]["solved_bugs"]) for d in devs)
        card_sz, util_in = 850, 1150                          # assembled-card / (card+report+stub) estimate, pre-build
        n_units = N * (N_TGT + 5)
        haiku_calls = ds_det = n_units * RESAMPLE
        build_calls = 2 * N + N + 3 * N + N + (1 + AP.K + N)  # nuwa(2)+naive(1)+deid4(3)+aggro(1)+archpool(1+K+N)
        util_calls = nbugs * 6
        haiku_in, haiku_out = haiku_calls * det_in, haiku_calls * 6
        ds_in = ds_det * det_in + build_calls * card_sz + util_calls * util_in
        ds_out = ds_det * 6 + build_calls * 900 + util_calls * 6
        usd = lambda ti, to, ri, ro: ti / 1e6 * ri + to / 1e6 * ro
        cost = usd(haiku_in, haiku_out, 1.0, 5.0) + usd(ds_in, ds_out, 0.28, 1.10)
        print(f"DRYRUN N={N}: build={build_calls} deepseek | anon units={n_units} -> haiku={haiku_calls}+deepseek={ds_det} "
              f"(det_in~{det_in}tok) | util={util_calls} deepseek (in~{util_in}tok)", flush=True)
        print(f"  est tokens: haiku in~{haiku_in/1e6:.2f}M out~{haiku_out/1e3:.0f}k | deepseek in~{ds_in/1e6:.2f}M out~{ds_out/1e6:.2f}M", flush=True)
        print(f"  est cost ~ ${cost:.2f} (haiku $1/$5, deepseek $0.28/$1.10 per M; Opus strong-attack FREE via subagents)", flush=True)
        return

    # ---- Step-1 nuwa ----
    if NUWAC.exists() and not os.environ.get("REBUILD"):
        nuwa = {d: json.loads(NUWAC.read_text(encoding="utf-8"))["nuwa"][d] for d in devs}
    else:
        print("STEP1: building nuwa cognitive-OS cards (2-call deepseek) ...", flush=True)
        nuwa = dict(zip(devs, de.pool(lambda d: nuwa_assemble(nuwa_extract(tr[d])), devs)))
        NUWAC.write_text(json.dumps({"nuwa": nuwa}, ensure_ascii=False), encoding="utf-8")

    # ---- Step-2: naive / deid4 / aggro / archpool (all on nuwa) ----
    step2 = json.loads(STEP2C.read_text(encoding="utf-8")) if STEP2C.exists() else {}

    def build(key, fn):
        if step2.get(key) and all(d in step2[key] for d in devs) and not os.environ.get("REBUILD"):
            return {d: step2[key][d] for d in devs}
        print(f"building {key} ...", flush=True)
        out = dict(zip(devs, de.pool(fn, devs)))
        step2[key] = out
        STEP2C.write_text(json.dumps(step2, ensure_ascii=False), encoding="utf-8")
        return out

    others = {d: [nuwa[b] for b in devs if b != d] for d in devs}   # contrast set for deid4 markers
    naive = build("naive", lambda d: ES.naive_card(nuwa[d]))
    deid4 = build("deid4", lambda d: ES.deid4_card(nuwa[d], others[d]))
    aggro = build("aggro", lambda d: ES.aggro_card(nuwa[d]))

    # archpool: cluster aggro cards by reasoning archetype -> per-cluster template -> re-express
    if step2.get("archpool") and all(d in step2["archpool"] for d in devs) and not os.environ.get("REBUILD"):
        archpool = {d: step2["archpool"][d] for d in devs}
        clusters = step2.get("archpool_clusters", {d: "?" for d in devs})
    else:
        print(f"clustering {N} aggro cards into {AP.K} reasoning archetypes ...", flush=True)
        clusters = AP.cluster_authors(aggro, devs)
        byc = {}
        for d in devs:
            byc.setdefault(clusters[d], []).append(d)
        for c, mem in sorted(byc.items(), key=lambda x: -len(x[1])):
            print(f"    [{len(mem)}] {c}: {', '.join(mem)}", flush=True)
        print("building cluster templates + re-expressing ...", flush=True)
        templates = {c: AP.cluster_template([aggro[d] for d in mem]) for c, mem in byc.items()}
        archpool = dict(zip(devs, de.pool(lambda d: AP.pool_card(aggro[d], templates[clusters[d]]), devs)))
        step2["archpool"] = archpool
        step2["archpool_clusters"] = clusters
        STEP2C.write_text(json.dumps(step2, ensure_ascii=False), encoding="utf-8")

    byc = {}
    for d in devs:
        byc.setdefault(clusters.get(d, "?"), []).append(d)
    cards = {"nuwa": nuwa, "naive": naive, "deid4": deid4, "aggro": aggro, "archpool": archpool}
    print("\n[TOK median] " + "  ".join(f"{k}={int(np.median([tlen(cards[k][d]) for d in devs]))}" for k in cards), flush=True)
    print(f"[clusters] { {c: len(m) for c, m in byc.items()} }", flush=True)

    out = {"dataset": "20mad", "N": N, "chance": round(chance, 4), "K_cluster": AP.K,
           "tok": {k: int(np.median([tlen(cards[k][d]) for d in devs])) for k in cards},
           "clusters": {c: len(m) for c, m in byc.items()}}

    # ===== ANONYMITY scripted (Opus single-shot via subagents, separate) =====
    if not os.environ.get("SKIP_ANON"):
        CARD_ARMS = ["nuwa", "naive", "deid4", "aggro", "archpool"]
        units = []
        for d in devs:
            for j in range(N_TGT):
                units.append(("comment", d, j, WS.sub(" ", pool[d]["card_comments"][N_TRAIN + N_REF + j])[:500]))
            for arm in CARD_ARMS:
                units.append((arm, d, 0, cards[arm][d]))
        AARMS = ["comment"] + CARD_ARMS

        def rd(i):
            arm, d, kk, tgt = units[i]
            lu = MC.lineup(devs, ref, d, f"{arm}-{kk}")
            ph = [MC.detective(tgt, lu, HEADLINE, f"mn-{arm}-{d}-{kk}-h{r}") for r in range(RESAMPLE)]
            pdd = [MC.detective(tgt, lu, SECOND, f"mn-{arm}-{d}-{kk}-d{r}") for r in range(RESAMPLE)]
            return i, float(np.mean([p == d for p in ph])), float(np.mean([p == d for p in pdd])), float(MC.topic_pick(tgt, lu) == d)
        R = {}
        for i, sh, sd, tp in de.pool(rd, list(range(len(units)))):
            R[i] = (sh, sd, tp)
        print(f"\n=== ANONYMITY scripted (K={N}, chance={chance:.3f}, soft over {RESAMPLE}) ===", flush=True)
        print(f"{'arm':9s} {'haiku':>22s} {'deepseek':>10s} {'topic':>8s}", flush=True)
        asum = {}
        for arm in AARMS:
            idxs = [i for i in range(len(units)) if units[i][0] == arm]; gg = [units[i][1] for i in idxs]
            h = [R[i][0] for i in idxs]; dd = [R[i][1] for i in idxs]; t = [R[i][2] for i in idxs]
            ci = cluster_mean_ci(h, gg, seed=SEED)
            asum[arm] = {"haiku": round(float(np.mean(h)), 3), "haiku_ci": ci,
                         "deepseek": round(float(np.mean(dd)), 3), "topic": round(float(np.mean(t)), 3), "n": len(idxs)}
            print(f"{arm:9s} {np.mean(h):.3f} CI{ci!s:>15s} {np.mean(dd):>10.3f} {np.mean(t):>8.3f}  (n={len(idxs)})", flush=True)
        print("  STRONG attacker (Opus single-shot K=24) via mad_nuwa_dump.py + subagents -- NOT scripted here.", flush=True)
        out["anonymity_scripted"] = asum

    # ===== UTILITY: objective resolution prediction =====
    if not os.environ.get("SKIP_UTIL"):
        ubugs = [(d, b) for d in devs for b in pool[d]["solved_bugs"]]
        UARMS = ["nocard", "nuwa", "naive", "deid4", "aggro", "archpool"]
        cardof = {"nocard": None, **cards}
        jobs = [((d, b), arm) for (d, b) in ubugs for arm in UARMS]

        def do_u(job):
            (d, b), arm = job
            card = None if arm == "nocard" else cardof[arm][d]
            p = MC.predict_res(card, b.get("report", ""), b.get("stub", ""))
            return job, 1.0 if p == b["resolution"] else 0.0
        acc = {arm: [] for arm in UARMS}; ug = {arm: [] for arm in UARMS}
        for ((d, b), arm), hit in de.pool(do_u, jobs):
            acc[arm].append(hit); ug[arm].append(d)
        print(f"\n=== UTILITY resolution-prediction (objective; n={len(ubugs)} bugs, {N} devs) ===", flush=True)
        ures = {}
        for arm in UARMS:
            ci = cluster_mean_ci(acc[arm], ug[arm], seed=SEED)
            ures[arm] = {"acc": round(float(np.mean(acc[arm])), 3), "ci": ci}
            print(f"  {arm:9s} acc={np.mean(acc[arm]):.3f} CI{ci}", flush=True)
        PAIRS = [("nuwa", "nocard"), ("naive", "nuwa"), ("deid4", "nuwa"), ("aggro", "nuwa"),
                 ("archpool", "nuwa"), ("archpool", "deid4")]
        upair = {}
        for x, y in PAIRS:
            r = cluster_paired_diff_ci(acc[x], acc[y], ug[x], seed=SEED)
            fl = "  <-EXCL0" if (r["ci"][0] > 0 or r["ci"][1] < 0) else ""
            upair[f"{x}-{y}"] = {"diff": round(r["diff"], 3), "ci": r["ci"]}
            print(f"    {x:9s} - {y:9s} = {r['diff']:+.3f} CI{r['ci']}{fl}", flush=True)
        print("  nuwa-nocard = does the rich card help at all (tests the old null)? others = does anonymization preserve it?", flush=True)
        out["utility"] = {"n_bugs": len(ubugs), "per_arm": ures, "pairs": upair}

    out["note"] = ("20-MAD migration of Enron Step1(nuwa)->Step2(naive/deid4/aggro/archpool). Strong attacker = Opus "
                   "single-shot K=24 via mad_nuwa_dump.py + subagents. UTILITY = objective resolution prediction.")
    (ROOT / "results" / "mad_nuwa_step2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> results/mad_nuwa_step2.json", flush=True)


if __name__ == "__main__":
    main()
