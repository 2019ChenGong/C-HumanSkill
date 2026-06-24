"""CMD Step-0 gate: build size-k clusters, synthesize one e=0 shared card per cluster, dump intra-cluster re-identification lineups (card/raw/indiv). Set DATASET=enron|mad."""
import os
import re
import sys
import json
import math
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
import enron_nuwa as NW  # noqa: E402
from src.llm import chat  # noqa: E402

GEN = "deepseek-chat"
DATASET = os.environ.get("DATASET", "enron")           # enron | mad (20-MAD SeaMonkey cross-dataset replication)
RES = ROOT / "results" if DATASET == "enron" else ROOT / "results" / DATASET
RES.mkdir(parents=True, exist_ok=True)
if DATASET == "enron":
    SE = ROOT / "data" / "enron"
    COLL = SE / "collected_ragfull_40.json"
    NUWAC = SE / "nuwa_cards_full.json"
    STEP2C = SE / "step2_cards_full.json"
    SHAREDC = SE / "cmd_shared_cards.json"
    MAD_TRAIN = None
else:                                                  # 20-MAD: full 128-dev SeaMonkey set, fresh CMD cards
    SE = ROOT / "data" / "20mad"
    COLL = SE / "mad_cmd_pool.json"
    # card files env-overridable for cross-model BUILDER checks (e.g. NUWAC=mad_cmd_nuwa__sonnet.json); defaults unchanged
    NUWAC = SE / os.environ.get("NUWAC", "mad_cmd_nuwa.json")
    STEP2C = SE / os.environ.get("STEP2C", "mad_cmd_step2.json")
    SHAREDC = SE / os.environ.get("SHAREDC", "cmd_shared_cards_mad.json")
    MAD_TRAIN = 18                                      # MAD bar 18/6/2; ref=comments[18:24], raw target=comments[24]
N_TRAIN, N_REF, N_TGT = 12, 6, 2                        # Enron doc indexing (MAD overrides via MAD_TRAIN in load_mad)
REF_CHARS = int(os.environ.get("REF_CHARS", 250))      # candidate-display budget — HELD IDENTICAL across datasets
RAW_CHARS = int(os.environ.get("RAW_CHARS", 900))
K_LIST = [int(x) for x in os.environ.get("K_LIST", "4,8").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
GROUP = os.environ.get("GROUP", "random")
CONDS = ["card", "raw", "indiv"]
WS = re.compile(r"\s+")


# ---------- grouping (size-k clusters; random default, knn optional) ----------
def group_random(authors, k, seed):
    rng = np.random.default_rng(seed)
    order = list(authors); rng.shuffle(order)
    groups = {}
    g = 0
    i = 0
    while i < len(order):
        chunk = order[i:i + k]
        if len(chunk) < k and groups:                 # tail < k -> merge into previous group (keep all >= k)
            for a in chunk:
                groups[a] = f"G{g-1}"
        else:
            for a in chunk:
                groups[a] = f"G{g}"
            g += 1
        i += k
    return groups


def group_knn(cards, authors, k, seed):
    vecs = {a: de._content_vec(cards[a]) for a in authors}
    rng = np.random.default_rng(seed)
    pool = list(authors); rng.shuffle(pool)
    pool = set(pool)
    groups, g = {}, 0
    order = list(authors); rng.shuffle(order)
    for a in order:
        if a not in pool:
            continue
        pool.discard(a)
        nbrs = sorted(pool, key=lambda b: -de._cosine(vecs[a], vecs[b]))[:k - 1]
        members = [a] + nbrs
        for b in members:
            groups[b] = f"G{g}"; pool.discard(b)
        g += 1
    # ensure every group >= k: merge any short final group into the most-similar full group
    byc = {}
    for a, gid in groups.items():
        byc.setdefault(gid, []).append(a)
    full = [gid for gid, m in byc.items() if len(m) >= k]
    for gid, m in list(byc.items()):
        if len(m) < k and full:
            rep = m[0]
            tgt = max([g for g in full if g != gid], key=lambda fg: de._cosine(vecs[rep], vecs[byc[fg][0]]), default=full[0])
            for a in m:
                groups[a] = tgt
    return groups


def make_groups(cards, authors, k, seed):
    grp = group_knn(cards, authors, k, seed) if GROUP == "knn" else group_random(authors, k, seed)
    byc = {}
    for a in authors:
        byc.setdefault(grp[a], []).append(a)
    return grp, byc


# ---------- ε=0 shared card (identical for all members of a cluster) ----------
def synth_shared(member_cards):
    """ONE shared card capturing the cluster's common working/decision approach, with NO detail unique to any
    single member -> published byte-identical to every member (the ε=0 mechanism)."""
    body = "\n\n---\n\n".join(member_cards)
    msg = [{"role": "system", "content": "You distill ONE shared skill card common to several colleagues, removing "
            "anything that identifies any single one of them."},
           {"role": "user", "content": f"Skill cards from several colleagues:\n\n{body}\n\nWrite ONE shared skill card "
            "(working/decision heuristics, 8-12 bullets) that captures ONLY what is COMMON across them — the shared "
            "competence and decision approach — with NO phrasing, move, priority, or detail unique to any single "
            "person. It must read as if it could belong to any of them equally. Output ONLY the shared card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=900) or ""


# ---------- char-4gram stylometric floor (cheap, no API) ----------
def char_vec(t, n=4):
    t = WS.sub(" ", (t or "").lower())
    return Counter(t[i:i + n] for i in range(max(0, len(t) - n + 1)))


def cos_counter(a, b):
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    num = sum(a[w] * b[w] for w in keys)
    da = math.sqrt(sum(v * v for v in a.values())); db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def load_enron():
    docs = json.loads(COLL.read_text(encoding="utf-8"))
    authors = [a for a in docs if len(docs[a]) >= N_TRAIN + N_REF + N_TGT]
    nuwa = json.loads(NUWAC.read_text(encoding="utf-8"))["nuwa"]
    aggro = json.loads(STEP2C.read_text(encoding="utf-8"))["aggro"]
    ref = {a: WS.sub(" ", docs[a][N_TRAIN]["text"])[:REF_CHARS] for a in authors}            # candidate ref (idx 12)
    raw_tgt = {a: WS.sub(" ", docs[a][N_TRAIN + 1]["text"])[:RAW_CHARS] for a in authors}     # held-out raw target (idx 13)
    return docs, authors, nuwa, aggro, ref, raw_tgt


def load_mad():
    """20-MAD SeaMonkey: docs = card_comments lists. ref = join(comments[18:24])[:250] (matches existing MAD harness),
    raw target = comments[24][:900]. Leak-disjoint: nuwa evidence ⊆ comments[:12] ⊂ train[:18]; ref=[18:24]; tgt=[24].
    Same return shape as load_enron (docs is a placeholder here — unused downstream in dump/run_floor)."""
    pool = json.loads(COLL.read_text(encoding="utf-8"))["pool"]
    t = MAD_TRAIN                                                                 # 18
    authors = [d for d in pool if len(pool[d]["card_comments"]) >= t + N_REF + N_TGT]
    nuwa = json.loads(NUWAC.read_text(encoding="utf-8"))["nuwa"]
    aggro = json.loads(STEP2C.read_text(encoding="utf-8"))["aggro"]
    for d in authors:
        assert len(pool[d]["card_comments"]) > t + N_REF, f"{d} too few comments for raw target idx {t+N_REF}"
    ref = {d: WS.sub(" ", " || ".join(pool[d]["card_comments"][t:t + N_REF]))[:REF_CHARS] for d in authors}
    raw_tgt = {d: WS.sub(" ", pool[d]["card_comments"][t + N_REF])[:RAW_CHARS] for d in authors}
    return pool, authors, nuwa, aggro, ref, raw_tgt


def load():
    return load_mad() if DATASET == "mad" else load_enron()


# ---------- scripted raw-only floor sweep (MODE=floor) ----------
def run_floor():
    _docs, authors, _nuwa, aggro, ref, raw_tgt = load()
    n_seeds = int(os.environ.get("SEEDS_FLOOR", 12))
    rvec = {a: char_vec(ref[a]) for a in authors}
    qvec = {a: char_vec(raw_tgt[a]) for a in authors}
    print(f"[FLOOR] char-4gram raw-only intra-cluster re-id, GROUP={GROUP}, {n_seeds} seeds", flush=True)
    for k in K_LIST:
        rates = []
        for s in range(n_seeds):
            grp, byc = make_groups(aggro, authors, k, s)
            hit = 0
            for a in authors:
                mem = byc[grp[a]]
                pick = max(mem, key=lambda b: cos_counter(qvec[a], rvec[b]))
                hit += (pick == a)
            rates.append(hit / len(authors))
        rates = np.array(rates)
        chance = 1.0 / k  # nominal; tail-merged groups are >=k so true chance <= 1/k -> 1/k is the CONSERVATIVE bar
        n_above = int((rates > chance).sum())
        verdict = ("raw-floor > 1/k on ALL seeds (raw already leaks)" if (rates > chance).all()
                   else "raw-floor <= 1/k on ALL seeds (style not separating within cluster)" if (rates <= chance).all()
                   else f"UNSTABLE across seeds ({n_above}/{len(rates)} > 1/k) — decision direction not seed-stable")
        print(f"  k={k} chance={chance:.3f}: re-id seed-mean={rates.mean():.3f} "
              f"[min/med/max={rates.min():.3f}/{np.median(rates):.3f}/{rates.max():.3f}] "
              f"seeds>1/k={n_above}/{len(rates)} -> {verdict}", flush=True)
    print("  NOTE: scripted char-4gram is a WEAK stylometer; the Opus dump is the strong-attacker floor. "
          "Use this only for SEED-STABILITY of the decision direction.", flush=True)


# ---------- build shared cards + dump Opus trials ----------
def build_shared(authors, aggro):
    cache = json.loads(SHAREDC.read_text(encoding="utf-8")) if SHAREDC.exists() else {}
    plan = []  # (key, member_cards)
    layout = {}  # (k,seed) -> (grp, byc)
    for k in K_LIST:
        for s in SEEDS:
            grp, byc = make_groups(aggro, authors, k, s)
            layout[(k, s)] = (grp, byc)
            for cid, mem in byc.items():
                ck = f"k{k}_s{s}_{cid}"
                if ck not in cache:
                    plan.append((ck, [aggro[a] for a in mem]))
    if plan:
        print(f"synthesizing {len(plan)} ε=0 shared cards (deepseek) ...", flush=True)
        for (ck, _), card in zip(plan, de.pool(lambda pc: synth_shared(pc[1]), plan)):
            cache[ck] = card
        SHAREDC.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache, layout


def dump():
    docs, authors, nuwa, aggro, ref, raw_tgt = load()
    N = len(authors)
    if os.environ.get("PILOT_DRYRUN"):
        n_clusters = sum(len(make_groups(aggro, authors, k, s)[1]) for k in K_LIST for s in SEEDS)
        n_trials = N * len(CONDS) * len(K_LIST) * len(SEEDS)
        ds_in = sum(len(make_groups(aggro, authors, k, s)[1]) * (k * 600 + 200) for k in K_LIST for s in SEEDS)
        ds_out = n_clusters * 700
        cost = ds_in / 1e6 * 0.28 + ds_out / 1e6 * 1.10
        print(f"DRYRUN N={N}: ε=0 shared cards to synth={n_clusters} (deepseek ~${cost:.2f}); "
              f"Opus trials={n_trials} ({len(CONDS)} conds × {len(K_LIST)} k × {len(SEEDS)} seeds × {N}) — Opus FREE via subagents.",
              flush=True)
        print(f"  trial files -> results/_cmdgate_k<k>_s<seed>_<cond>_T###.txt ; keys _..._key.json", flush=True)
        return
    shared, layout = build_shared(authors, aggro)
    # ε=0 guard: every cluster must have exactly ONE built shared card (reused byte-identically for all members)
    for (k, s), (grp, byc) in layout.items():
        for cid in byc:
            assert f"k{k}_s{s}_{cid}" in shared, f"missing ε=0 shared card k{k}_s{s}_{cid}"
    manifest = {}
    for k in K_LIST:
        for s in SEEDS:
            grp, byc = layout[(k, s)]
            for cond in CONDS:
                key = {}
                for idx, a in enumerate(authors, 1):
                    mem = byc[grp[a]]
                    lu = NW.lineup(mem, ref, a, f"gate-{k}-{s}-{a}")           # intra-cluster, SAME order across conds
                    true_slot = next(slot for slot, b, _ in lu if b == a)
                    if cond == "card":
                        tgt, kind = shared[f"k{k}_s{s}_{grp[a]}"], "a SHARED SKILL CARD (one card published to a whole group)"
                    elif cond == "indiv":
                        tgt, kind = nuwa[a], "an INDIVIDUAL SKILL CARD"
                    else:
                        tgt, kind = raw_tgt[a], "a RAW WORK EMAIL"
                    lines = ["# Authorship re-identification — single isolated trial",
                             f"The TARGET below is {kind} from ONE author. Identify which candidate is that SAME author,",
                             "by their reasoning, priorities, and decision style — NOT by topic (topics overlap).\n",
                             "TARGET:", tgt, "",
                             f"CANDIDATES (each a sample of their own writing; pick exactly ONE of 1..{len(lu)}):"]
                    for slot, _b, rt in lu:
                        lines.append(f"[{slot}] {rt}")
                    (RES / f"_cmdgate_k{k}_s{s}_{cond}_T{idx:03d}.txt").write_text("\n".join(lines), encoding="utf-8")
                    key[f"T{idx:03d}"] = {"author": a, "true_candidate": true_slot, "k": k, "seed": s,
                                          "cond": cond, "cluster": grp[a], "cluster_size": len(mem)}
                (RES / f"_cmdgate_k{k}_s{s}_{cond}_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
                manifest[f"k{k}_s{s}_{cond}"] = len(authors)
    (RES / "_cmdgate_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"dumped {len(manifest)} conditions × {N} trials -> results/_cmdgate_*  (GROUP={GROUP}, K={K_LIST}, seeds={SEEDS})", flush=True)
    print("  next: dispatch Opus subagents over _cmdgate_k<k>_s<seed>_<cond>_T*.txt (≤3 concurrent), then cmd_gate_score.py", flush=True)


if __name__ == "__main__":
    if os.environ.get("MODE") == "floor":
        run_floor()
    else:
        dump()
