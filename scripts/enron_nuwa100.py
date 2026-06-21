"""Build Enron nuwa cards plus the archpool / random_pool Step-2 anonymization frontier."""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
import enron_nuwa as NW  # noqa: E402  (nuwa_extract, nuwa_assemble, draft, quality, ENC, HEADLINE)
import enron_step2 as ES  # noqa: E402  (aggro_card)
import enron_archpool as AP  # noqa: E402  (cluster_template, pool_card)
from src.llm import chat  # noqa: E402
from src.attrib_metrics import cluster_mean_ci, cluster_paired_diff_ci  # noqa: E402

GEN = "deepseek-chat"
SE = ROOT / "data" / "enron"
COLL = SE / os.environ.get("COLL", "collected_rag100_40.json")
NUWAC = SE / os.environ.get("NUWAC", "nuwa_cards_100.json")
STEP2C = SE / os.environ.get("STEP2C", "step2_cards_100.json")
N_TRAIN, N_REF, N_TGT, REF_CHARS = 12, 6, 2, 500
KM = int(os.environ.get("KM", 16))
MIN = int(os.environ.get("MIN", 3))
SEED = 0
WS = re.compile(r"\s+")


def balanced_kmeans(texts, k, min_size):
    X = TfidfVectorizer(stop_words="english", max_features=600, ngram_range=(1, 2)).fit_transform(texts)
    lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
    Xd = X.toarray()
    cents = {c: Xd[lab == c].mean(0) for c in set(lab)}
    def cos(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        return float(u @ v / (nu * nv)) if nu and nv else 0.0
    while True:
        sizes = {c: int((lab == c).sum()) for c in set(lab)}
        small = [c for c, s in sizes.items() if s < min_size]
        if not small or len(set(lab)) <= 1:
            break
        c = small[0]; others = [o for o in set(lab) if o != c]
        tgt = max(others, key=lambda o: cos(cents[c], cents[o]))
        lab[lab == c] = tgt
        cents = {cc: Xd[lab == cc].mean(0) for cc in set(lab)}
    return lab


def adv_paraphrase_card(card):
    msg = [{"role": "system", "content": "You are an expert at anonymizing text to DEFEAT authorship attribution while preserving its usefulness."},
           {"role": "user", "content": f"Below is a skill card distilled from ONE person's work. Rewrite it so an "
            f"authorship-attribution system (matching writing AND reasoning style) CANNOT tell whose it is. Remove or "
            f"alter every individual signature — phrasing, tone, characteristic move-SEQUENCES, idiosyncratic priorities "
            f"and decision patterns — while PRESERVING all substantive decision procedures, frameworks, and heuristics "
            f"so it stays fully usable. Output ONLY the rewritten card.\n\nCard:\n{card}"}]
    return chat(msg, model=GEN, temperature=0.4, max_tokens=1100) or ""


def tlen(c):
    return len(NW.ENC.encode(c or ""))


def main():
    docs = json.loads(COLL.read_text(encoding="utf-8"))
    authors = [a for a in docs if len(docs[a]) >= N_TRAIN + N_REF + N_TGT]
    N = len(authors); chance = 1.0 / N
    tr = {a: [docs[a][j]["text"] for j in range(N_TRAIN)] for a in authors}
    print(f"N={N} chance={chance:.3f} KM={KM} MIN={MIN}", flush=True)

    # ---- DRYRUN cost estimate BEFORE building (cost discipline) ----
    if os.environ.get("PILOT_DRYRUN"):
        T = len(de.TASKS)
        ext_in = tlen("\n\n---\n\n".join(tr[authors[0]][:N_TRAIN])) + 150
        card_sz = 1100
        build_calls = 2 * N + N + (KM + N) + (KM + N) + N        # nuwa+aggro+archpool+random_pool+adv
        UARMS_CARD = ["nuwa", "archpool", "random_pool", "adv_paraphrase"]
        n_draft = len(UARMS_CARD) * N * T + T                    # +nocard
        PAIRS = 6
        n_judge = PAIRS * N * T
        ds_in = build_calls * 1500 + n_draft * (card_sz + 200)   # build + drafts (deepseek)
        ds_out = build_calls * 900 + n_draft * 400
        hk_in = n_judge * (card_sz + 200); hk_out = n_judge * 4   # judge (haiku)
        usd = lambda ti, to, ri, ro: ti / 1e6 * ri + to / 1e6 * ro
        cost = usd(ds_in, ds_out, 0.28, 1.10) + usd(hk_in, hk_out, 1.0, 5.0)
        print(f"DRYRUN N={N}: build={build_calls} deepseek (ext_in~{ext_in}tok) | util drafts={n_draft} deepseek + "
              f"judges={n_judge} haiku ({PAIRS} pairs × {N} × {T})", flush=True)
        print(f"  est: deepseek in~{ds_in/1e6:.1f}M out~{ds_out/1e6:.1f}M | haiku in~{hk_in/1e6:.1f}M", flush=True)
        print(f"  est cost ~ ${cost:.1f} (deepseek $0.28/$1.10, haiku $1/$5 per M; Opus K=100 attack FREE via subagents, separate)", flush=True)
        return

    step2 = json.loads(STEP2C.read_text(encoding="utf-8")) if STEP2C.exists() else {}

    # ---- Step-1 nuwa (INCREMENTAL: reuse cached authors, build only missing) ----
    cache = json.loads(NUWAC.read_text(encoding="utf-8"))["nuwa"] if NUWAC.exists() else {}
    miss = authors if os.environ.get("REBUILD") else [a for a in authors if a not in cache]
    if miss:
        print(f"STEP1: building {len(miss)} nuwa cards (2-call; reusing {len(authors)-len(miss)} cached) ...", flush=True)
        built = dict(zip(miss, de.pool(lambda a: NW.nuwa_assemble(NW.nuwa_extract(tr[a])), miss)))
        cache.update(built)
        NUWAC.write_text(json.dumps({"nuwa": cache}, ensure_ascii=False), encoding="utf-8")
    nuwa = {a: cache[a] for a in authors}

    def build(key, fn):
        if step2.get(key) and all(a in step2[key] for a in authors) and not os.environ.get("REBUILD"):
            return {a: step2[key][a] for a in authors}
        print(f"building {key} ...", flush=True)
        out = fn(); step2[key] = out
        STEP2C.write_text(json.dumps(step2, ensure_ascii=False), encoding="utf-8")
        return out

    aggro = build("aggro", lambda: dict(zip(authors, de.pool(lambda a: ES.aggro_card(nuwa[a]), authors))))

    # archpool: balanced KMeans clusters -> per-cluster template -> pool
    if step2.get("archpool") and all(a in step2["archpool"] for a in authors) and not os.environ.get("REBUILD"):
        archpool = {a: step2["archpool"][a] for a in authors}; clusters = step2["archpool_clusters"]
    else:
        lab = balanced_kmeans([aggro[a] for a in authors], KM, MIN)
        clusters = {authors[i]: f"C{int(lab[i])}" for i in range(N)}
        byc = {}
        for a in authors:
            byc.setdefault(clusters[a], []).append(a)
        print(f"archpool clusters (KM={KM}->{len(byc)}): { sorted((len(m) for m in byc.values()), reverse=True) }", flush=True)
        templ = {c: AP.cluster_template([aggro[a] for a in mem]) for c, mem in byc.items()}
        archpool = dict(zip(authors, de.pool(lambda a: AP.pool_card(aggro[a], templ[clusters[a]]), authors)))
        step2["archpool"] = archpool; step2["archpool_clusters"] = clusters
        STEP2C.write_text(json.dumps(step2, ensure_ascii=False), encoding="utf-8")

    sizes = sorted([sum(1 for a in authors if clusters[a] == c) for c in set(clusters[a] for a in authors)], reverse=True)

    def build_randompool():
        rng = np.random.default_rng(SEED); order = list(authors); rng.shuffle(order)
        groups, i = {}, 0
        for gi, s in enumerate(sizes):
            for a in order[i:i + s]:
                groups[a] = f"R{gi}"
            i += s
        byg = {}
        for a in authors:
            byg.setdefault(groups[a], []).append(a)
        templ = {g: AP.cluster_template([aggro[a] for a in mem]) for g, mem in byg.items()}
        return dict(zip(authors, de.pool(lambda a: AP.pool_card(aggro[a], templ[groups[a]]), authors)))
    random_pool = build("random_pool", build_randompool)
    adv = build("adv_paraphrase", lambda: dict(zip(authors, de.pool(lambda a: adv_paraphrase_card(nuwa[a]), authors))))

    cards = {"nuwa": nuwa, "aggro": aggro, "archpool": archpool, "random_pool": random_pool, "adv_paraphrase": adv}
    print("[TOK median] " + "  ".join(f"{k}={int(np.median([tlen(cards[k][a]) for a in authors]))}" for k in cards), flush=True)
    print(f"[archpool cluster sizes] {sizes}", flush=True)

    out = {"N": N, "chance": round(chance, 4), "cluster_sizes": sizes,
           "tok": {k: int(np.median([tlen(cards[k][a]) for a in authors])) for k in cards}}

    # ---- UTILITY: Enron 8-task pairwise competence (KEY: archpool vs random_pool at high power) ----
    if not os.environ.get("SKIP_UTIL"):
        T = de.TASKS
        units = [(a, t) for a in authors for t in range(len(T))]
        nocard_d = dict(zip(range(len(T)), de.pool(lambda t: NW.draft(None, T[t]), list(range(len(T))))))
        UARMS_CARD = ["nuwa", "archpool", "random_pool", "adv_paraphrase"]
        dj = [(arm, a, t) for arm in UARMS_CARD for (a, t) in units]
        D = {}
        for (arm, a, t), txt in zip(dj, de.pool(lambda j: NW.draft(cards[j[0]][j[1]], T[j[2]]), dj)):
            D[(arm, a, t)] = txt
        dof = lambda arm, a, t: nocard_d[t] if arm == "nocard" else D[(arm, a, t)]
        PAIRS = [("archpool", "random_pool"), ("archpool", "nocard"), ("random_pool", "nocard"),
                 ("archpool", "nuwa"), ("adv_paraphrase", "nocard"), ("archpool", "adv_paraphrase")]
        jj = [(a, t, x, y) for (a, t) in units for (x, y) in PAIRS]
        J = {}
        for (a, t, x, y), v in zip(jj, de.pool(lambda j: NW.quality(T[j[1]], dof(j[2], j[0], j[1]), dof(j[3], j[0], j[1]),
                                                                     f"e100-{j[0]}-{j[1]}-{j[2]}-{j[3]}"), jj)):
            J[(a, t, x, y)] = v
        g = [a for (a, t) in units]
        print(f"\n=== UTILITY (Enron pairwise competence; +1 first better; n={len(units)}) ===", flush=True)
        ures = {}
        for x, y in PAIRS:
            v = [J[(a, t, x, y)] for (a, t) in units]
            ci = cluster_mean_ci(v, g, seed=SEED)
            fl = "  <-EXCL0" if (ci[0] > 0 or ci[1] < 0) else ""
            ures[f"{x}-{y}"] = {"diff": round(float(np.mean(v)), 3), "ci": ci}
            print(f"  {x:14s} vs {y:14s} = {np.mean(v):+.3f} CI{ci}{fl}", flush=True)
        print("  KEY: archpool-random_pool — at N=100 does the archetype edge clear significance, or confirm ≈random (powered null)?", flush=True)
        out["utility"] = ures

    out["note"] = "Enron N=100 full-scale nuwa->pooling. Hardens the pooling headline + resolves archetype-vs-random. Opus K=100 anonymity via enron_nuwa100_dump.py + subagents."
    (ROOT / "results" / "enron_nuwa100.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> results/enron_nuwa100.json", flush=True)


if __name__ == "__main__":
    main()
