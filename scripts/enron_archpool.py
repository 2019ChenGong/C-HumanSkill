"""archpool baseline: cluster cards by reasoning archetype, build one per-cluster template, re-express each card into it."""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
import enron_nuwa as NW  # noqa: E402
from src.llm import chat  # noqa: E402
from src.attrib_metrics import cluster_mean_ci  # noqa: E402

GEN = "deepseek-chat"
SE = ROOT / "data" / "enron"
NUWAC = SE / "nuwa_cards.json"
STEP2C = SE / "step2_cards.json"
N_AUTH, N_TRAIN, N_TGT, REF_CHARS = 18, 12, 2, 500
RESAMPLE = int(os.environ.get("RESAMPLE", 2))
K = int(os.environ.get("K", 4))
SEED = 0
WS = re.compile(r"\s+")
LEAKERS = ["haedicke-m", "love-p", "bass-e", "kaminski-v", "jones-t", "germany-c"]


def cluster_authors(aggro, authors):
    body = "\n\n".join(f"=== CARD {i + 1} ===\n{aggro[a][:700]}" for i, a in enumerate(authors))
    msg = [{"role": "system", "content": "You group skill cards by their underlying REASONING ARCHETYPE (the shape of "
            "how they decide), NOT by topic."},
           {"role": "user", "content": f"{body}\n\nGroup these {len(authors)} cards into exactly {K} archetype clusters "
            "by similarity of REASONING STYLE / decision architecture (e.g. risk-recommender, internal-control/process, "
            "network-router/connector, coordinator/administrator). Every card goes in EXACTLY one cluster; each cluster "
            f"MUST have at least 3 cards. Output ONLY lines of the form:\nClusterName: 1,4,7,12\n(one line per cluster, "
            f"using card numbers 1-{len(authors)})."}]
    out = chat(msg, model=GEN, temperature=0.2, max_tokens=400) or ""
    clusters, seen = {}, set()
    for line in out.splitlines():
        m = re.match(r"\s*([^:]+):\s*(\d[\d,\s]*)", line)      # tolerate trailing prose after the number list
        if not m:
            continue
        name = m.group(1).strip()
        for i in re.findall(r"\d+", m.group(2)):
            i = int(i)
            if 1 <= i <= len(authors) and i not in seen:
                clusters[authors[i - 1]] = name; seen.add(i)
    for a in authors:                                          # any unassigned -> own catch-all
        clusters.setdefault(a, "unassigned")
    return clusters


def cluster_template(member_cards):
    body = "\n\n---\n\n".join(member_cards)
    msg = [{"role": "system", "content": "You distill a SHARED decision template common to several people of the same "
            "professional archetype."},
           {"role": "user", "content": f"Cards from several people of the SAME reasoning archetype:\n\n{body}\n\nWrite "
            "ONE shared, ordered decision template (a fixed sequence of 6-9 named steps) that captures the COMMON "
            "decision architecture of this archetype -- the move-sequence they share -- with NO detail unique to any "
            "single one of them. Output ONLY the numbered step template."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=500) or ""


def pool_card(card, template):
    msg = [{"role": "system", "content": "You re-express a skill card into a shared archetype template so that members "
            "of the same archetype look structurally identical."},
           {"role": "user", "content": f"Shared archetype template:\n{template}\n\nSkill card to re-express:\n{card}\n\n"
            "Re-express ALL the card's substance STRICTLY within the shared template above, in its exact step order, one "
            "short paragraph per step. Keep the archetype's shared methods, but REMOVE anything that would distinguish "
            "THIS person from others of the same archetype (no individual-specific move, phrasing, or distinctive "
            "sequence). Output ONLY the templated card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=900) or ""


def toklen(c):
    return len(NW.ENC.encode(c or ""))


def vacuity(cards):
    vs = [de._content_vec(c) for c in cards]
    s = [de._cosine(vs[i], vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
    return round(float(np.mean(s)), 3) if s else 0.0


def main():
    docs = de.get_docs()
    need = N_TRAIN + 1 + N_TGT
    authors = [a for a in sorted(docs, key=lambda a: -len(docs[a])) if len(docs[a]) >= need][:N_AUTH]
    N = len(authors); chance = 1.0 / N
    ref = {a: WS.sub(" ", docs[a][N_TRAIN]["text"])[:REF_CHARS] for a in authors}
    nuwa = json.loads(NUWAC.read_text(encoding="utf-8"))["nuwa"]
    step2 = json.loads(STEP2C.read_text(encoding="utf-8"))
    aggro = {a: step2["aggro"][a] for a in authors}
    archnorm = {a: step2["archnorm"][a] for a in authors}

    if step2.get("archpool") and not os.environ.get("REBUILD"):
        archpool = {a: step2["archpool"][a] for a in authors}
        clusters = step2.get("archpool_clusters", {a: "?" for a in authors})
    else:
        print("clustering 18 aggro cards by reasoning archetype ...", flush=True)
        clusters = cluster_authors(aggro, authors)
        byc = {}
        for a in authors:
            byc.setdefault(clusters[a], []).append(a)
        print("  clusters:", flush=True)
        for c, mem in sorted(byc.items(), key=lambda x: -len(x[1])):
            print(f"    [{len(mem)}] {c}: {', '.join(mem)}", flush=True)
        print("building cluster templates ...", flush=True)
        templates = {c: cluster_template([aggro[a] for a in mem]) for c, mem in byc.items()}
        print("re-expressing each card into its cluster template ...", flush=True)
        archpool = dict(zip(authors, de.pool(lambda a: pool_card(aggro[a], templates[clusters[a]]), authors)))
        step2["archpool"] = archpool
        step2["archpool_clusters"] = clusters
        STEP2C.write_text(json.dumps(step2, ensure_ascii=False), encoding="utf-8")

    byc = {}
    for a in authors:
        byc.setdefault(clusters.get(a, "?"), []).append(a)
    print(f"\n[STATS] tok aggro={int(np.median([toklen(aggro[a]) for a in authors]))} "
          f"archnorm={int(np.median([toklen(archnorm[a]) for a in authors]))} "
          f"archpool={int(np.median([toklen(archpool[a]) for a in authors]))}; "
          f"clusters={ {c: len(m) for c, m in byc.items()} }", flush=True)
    print(f"  leaker clusters: " + ", ".join(f"{a}->{clusters.get(a,'?')}(size {len(byc.get(clusters.get(a,'?'),[]))})" for a in LEAKERS), flush=True)

    if os.environ.get("PILOT_DRYRUN"):
        nb = N * len(de.TASKS)
        print(f"DRYRUN N={N}; UTILITY drafts~{nb*2} judges~{nb*3}; ANON units~{N*N_TGT + N} x{RESAMPLE}x2.", flush=True)
        return

    out = {"N": N, "chance": round(chance, 4), "clusters": {c: m for c, m in byc.items()},
           "tok": {"aggro": int(np.median([toklen(aggro[a]) for a in authors])),
                   "archpool": int(np.median([toklen(archpool[a]) for a in authors]))}}

    # ===== ANONYMITY scripted (Opus separate) =====
    if not os.environ.get("SKIP_ANON"):
        units = []
        for a in authors:
            for j in range(N_TGT):
                units.append(("comment", a, j, WS.sub(" ", docs[a][N_TRAIN + 1 + j]["text"])[:1200]))
            units.append(("archpool", a, 0, archpool[a]))
        AARMS = ["comment", "archpool"]

        def rd(i):
            arm, a, kk, tgt = units[i]
            lu = NW.lineup(authors, ref, a, f"{arm}-{kk}")
            ph = [NW.detective(tgt, lu, NW.HEADLINE, f"ap-{arm}-{a}-{kk}-h{r}") for r in range(RESAMPLE)]
            return i, float(np.mean([p == a for p in ph])), float(NW.topic_pick(tgt, lu) == a)
        R = {}
        for i, sh, tp in de.pool(rd, list(range(len(units)))):
            R[i] = (sh, tp)
        print(f"\n=== ANONYMITY scripted (K={N}, chance={chance:.3f}) ===", flush=True)
        asum = {}
        for arm in AARMS:
            idxs = [i for i in range(len(units)) if units[i][0] == arm]; gg = [units[i][1] for i in idxs]
            h = [R[i][0] for i in idxs]; t = [R[i][1] for i in idxs]
            ci = cluster_mean_ci(h, gg, seed=SEED)
            asum[arm] = {"haiku": round(float(np.mean(h)), 3), "haiku_ci": ci, "topic": round(float(np.mean(t)), 3), "n": len(idxs)}
            print(f"  {arm:9s} haiku={np.mean(h):.3f} CI{ci} topic={np.mean(t):.3f} (n={len(idxs)})", flush=True)
        print("  ref Opus(reused): aggro 0.222 / aggro_short 0.167 / archnorm 0.111. archpool Opus via enron_step2_dump.py COND=archpool.", flush=True)
        out["anonymity"] = asum

    # ===== UTILITY (vs nocard / aggro / archnorm) =====
    if not os.environ.get("SKIP_UTIL"):
        T = de.TASKS
        units = [(a, t) for a in authors for t in range(len(T))]
        nocard_d = dict(zip(range(len(T)), de.pool(lambda t: NW.draft(None, T[t]), list(range(len(T))))))
        cardof = {"aggro": aggro, "archnorm": archnorm, "archpool": archpool}
        dj = [(arm, a, t) for arm in cardof for (a, t) in units]
        D = {}
        for (arm, a, t), txt in zip(dj, de.pool(lambda j: NW.draft(cardof[j[0]][j[1]], T[j[2]]), dj)):
            D[(arm, a, t)] = txt

        def dof(arm, a, t):
            return nocard_d[t] if arm == "nocard" else D[(arm, a, t)]
        PAIRS = [("archpool", "nocard"), ("archpool", "aggro"), ("archpool", "archnorm")]
        jj = [(a, t, x, y) for (a, t) in units for (x, y) in PAIRS]
        J = {}
        for (a, t, x, y), v in zip(jj, de.pool(lambda j: NW.quality(T[j[1]], dof(j[2], j[0], j[1]),
                                                                     dof(j[3], j[0], j[1]),
                                                                     f"ap-{j[0]}-{j[1]}-{j[2]}-{j[3]}"), jj)):
            J[(a, t, x, y)] = v
        g = [a for (a, t) in units]
        print(f"\n=== UTILITY (pairwise; +1 first better; n={len(units)}) ===", flush=True)
        ures = {}
        for x, y in PAIRS:
            v = [J[(a, t, x, y)] for (a, t) in units]
            ci = cluster_mean_ci(v, g, seed=SEED)
            fl = "  <-EXCL0" if (ci[0] > 0 or ci[1] < 0) else ""
            ures[f"{x}-{y}"] = {"diff": round(float(np.mean(v)), 3), "ci": ci}
            print(f"  {x:9s} vs {y:9s} = {np.mean(v):+.3f} CI{ci}{fl}", flush=True)
        print("  archpool-aggro (cost vs best-util) ; archpool-archnorm >0 (recovered utility vs the extreme).", flush=True)
        out["utility"] = ures

    out["note"] = "archetype-pooling: cluster by reasoning archetype, per-cluster shared template. Opus via enron_step2_dump.py COND=archpool; leakers=" + ",".join(LEAKERS)
    (ROOT / "results" / "enron_archpool.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> results/enron_archpool.json", flush=True)


if __name__ == "__main__":
    main()
