"""De-identification card methods: naive paraphrase, deid4 marker-aware content bottleneck, and aggro genericization."""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
import enron_nuwa as NW  # noqa: E402  (draft, quality, detective, topic_pick, lineup, HEADLINE, SECOND, ENC)
from src.llm import chat  # noqa: E402
from src.attrib_metrics import cluster_mean_ci  # noqa: E402

GEN = "deepseek-chat"
SE = ROOT / "data" / "enron"
NUWAC = SE / "nuwa_cards.json"
AGGROC = SE / "nuwa_aggro_cards.json"
CARDS = SE / "step2_cards.json"
N_AUTH, N_TRAIN, N_TGT, REF_CHARS = 18, 12, 2, 500
RESAMPLE = int(os.environ.get("RESAMPLE", 2))
SEED = 0
WS = re.compile(r"\s+")
SECTIONS = "[Analysis Protocol]; [Decision Frameworks & Mental Models]; [Heuristics] each as 'When X -> do Y, " \
           "watch for Z'; [Characteristic Moves]; [Failure Modes]"


# ---------------- STEP 2a: deid4 (marker-aware content bottleneck) adapted to a CARD ----------------
def card_markers(card, others):
    contrast = "\n\n".join(f"(another person's card)\n{c[:600]}" for c in others)
    msg = [{"role": "system", "content": "You spot what makes a skill card identifiable as a specific individual."},
           {"role": "user", "content": f"TARGET card:\n{card}\n\nOTHER people's cards (for contrast):\n{contrast}\n\n"
            "List the 3-6 distinctive habits / characteristic moves / phrasings / recurring decision-patterns that "
            "single out the TARGET's author from the others -- what would let someone recognize this author. Output "
            "a short bullet list, no preamble."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=300) or ""


def card_abstract_marker_aware(card, markers):
    """deid4 step1 on a card: strip personal PHRASING of the markers, KEEP every procedure/decision."""
    msg = [{"role": "system", "content": "You extract the essential, transferable procedures of a skill card."},
           {"role": "user", "content": f"Skill card:\n{card}\n\nThis author's IDENTIFYING habits/phrasings to "
            f"neutralize:\n{markers}\n\nList the substantive, EXECUTABLE decision procedures this card conveys -- the "
            "analysis steps, frameworks, heuristics, characteristic moves, and failure modes -- as terse NEUTRAL "
            "bullets grouped by section. Strip the author's personal PHRASING, tone, and stylistic tics (including the "
            "habits above), BUT preserve every procedure, framework, heuristic, and decision -- remove only the "
            "personal wording, never the substance. Output only the bullets."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=700) or ""


def card_rewrite_from_points(points):
    """deid4 step2: regenerate a FRESH neutral card from points only (no access to original -> style can't carry)."""
    msg = [{"role": "system", "content": "You write a plain, neutral skill card from a list of procedures."},
           {"role": "user", "content": f"Procedures to convey:\n{points}\n\nWrite a single skill card with sections "
            f"{SECTIONS} that conveys EXACTLY these procedures, in a neutral generic professional style with NO "
            "distinctive habits, phrasings, or personal tics. Keep each procedure concrete and executable. Add no "
            "content beyond the points. Output ONLY the card."}]
    return chat(msg, model=GEN, temperature=0.5, max_tokens=1100) or ""


def deid4_card(card, others):
    return card_rewrite_from_points(card_abstract_marker_aware(card, card_markers(card, others)))


# ---------------- STEP 2b: aggro (genericize the DECISION procedures themselves) -- reused from nuwa_aggro ----------
def aggro_card(card):
    msg = [{"role": "system", "content": "You strip a skill card to ONLY fully generic, person- and "
            "domain-independent decision procedures."},
           {"role": "user", "content": f"Skill card:\n\n{card}\n\nRewrite so EVERY procedure is abstracted to its "
            "most GENERIC, universally-applicable form -- a decision procedure ANY competent professional in ANY "
            "field could follow, with ZERO trace of the specific situations, characteristic moves, phrasings, or "
            "idiosyncrasies of the person it came from. Replace each specific move with the general principle behind "
            "it. The result must look ESSENTIALLY THE SAME no matter whose work it was distilled from -- no individual "
            "signature. Keep it concrete and executable (real procedures, not vague platitudes), but fully "
            "de-individualized. Same section structure. Output ONLY the card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=1100) or ""


# ---------------- STEP 2c: naive floor ----------------
def naive_card(card):
    msg = [{"role": "system", "content": "You anonymize a skill card."},
           {"role": "user", "content": f"Skill card:\n\n{card}\n\nRewrite this card to remove anything that could "
            "identify who it was distilled from, while keeping it useful. Output ONLY the card."}]
    return chat(msg, model=GEN, temperature=0.5, max_tokens=1100) or ""


# ---------------- card stats (anti-gutting) ----------------
_BULLET = re.compile(r"^\s*([-*•–]|\d+[.)]|when\b)", re.I)


def propcount(card):
    return len([ln for ln in (card or "").splitlines() if _BULLET.match(ln)])


def toklen(card):
    return len(NW.ENC.encode(card or ""))


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
    V = {a: de._content_vec(ref[a]) for a in authors}
    strg = {a: max((b for b in authors if b != a), key=lambda b: de._cosine(V[a], V[b])) for a in authors}

    nuwa = json.loads(NUWAC.read_text(encoding="utf-8"))["nuwa"]   # Step-1 anchor (the real distillation product)

    # ---- build / load Step-2 cards (all on the nuwa card) ----
    need_keys = ("nuwa", "deid4", "aggro", "naive")
    if CARDS.exists() and not os.environ.get("REBUILD") and all(json.loads(CARDS.read_text(encoding="utf-8")).get(k) for k in need_keys):
        C = json.loads(CARDS.read_text(encoding="utf-8"))
    else:
        print("STEP2a: building deid4(nuwa) (marker-aware bottleneck) ...", flush=True)
        othr = {a: [nuwa[b] for b in authors if b != a][:4] for a in authors}
        deid4 = dict(zip(authors, de.pool(lambda a: deid4_card(nuwa[a], othr[a]), authors)))
        if AGGROC.exists():
            print("STEP2b: reusing aggro(nuwa) from nuwa_aggro_cards.json ...", flush=True)
            aggro = json.loads(AGGROC.read_text(encoding="utf-8"))
            aggro = {a: aggro[a] for a in authors}
        else:
            print("STEP2b: building aggro(nuwa) ...", flush=True)
            aggro = dict(zip(authors, de.pool(lambda a: aggro_card(nuwa[a]), authors)))
        print("STEP2c: building naive(nuwa) floor ...", flush=True)
        naive = dict(zip(authors, de.pool(lambda a: naive_card(nuwa[a]), authors)))
        C = {"nuwa": nuwa, "deid4": deid4, "aggro": aggro, "naive": naive}
        CARDS.write_text(json.dumps(C, ensure_ascii=False), encoding="utf-8")

    ARMS = list(need_keys)
    stats = {}
    for k in ARMS:
        d = C[k]
        stats[k] = {"tok": int(np.median([toklen(d[a]) for a in authors])),
                    "props": int(np.median([propcount(d[a]) for a in authors])),
                    "vacuity": vacuity([d[a] for a in authors])}
    print(f"\n[CARD STATS median] (anti-gutting: tok/props ~level; vacuity->high = collapsed to sameness; "
          f"the {{arm}}-nuwa utility pair is the real gutting guard)", flush=True)
    for k in ARMS:
        print(f"  {k:7s} tok={stats[k]['tok']:5d} props={stats[k]['props']:3d} vacuity={stats[k]['vacuity']}", flush=True)

    if os.environ.get("PILOT_DRYRUN"):
        nb = N * len(de.TASKS)
        print(f"DRYRUN N={N}; UTILITY drafts~{nb*4+len(de.TASKS)} judges~{nb*8}; ANON units~{N*N_TGT + N*4} x{RESAMPLE}x2.", flush=True)
        return

    out = {"N": N, "chance": round(chance, 4), "card_stats": stats}

    # ===== ANONYMITY (scripted, full K=18) =====
    if not os.environ.get("SKIP_ANON"):
        units = []
        for a in authors:
            for j in range(N_TGT):
                units.append(("comment", a, j, WS.sub(" ", docs[a][N_TRAIN + 1 + j]["text"])[:1200]))
            for k in ARMS:
                units.append((k, a, 0, C[k][a]))
        AARMS = ["comment"] + ARMS

        def rd(i):
            arm, a, kk, tgt = units[i]
            lu = NW.lineup(authors, ref, a, f"{arm}-{kk}")
            ph = [NW.detective(tgt, lu, NW.HEADLINE, f"s2-{arm}-{a}-{kk}-h{r}") for r in range(RESAMPLE)]
            pdz = [NW.detective(tgt, lu, NW.SECOND, f"s2-{arm}-{a}-{kk}-d{r}") for r in range(RESAMPLE)]
            return i, float(np.mean([p == a for p in ph])), float(np.mean([p == a for p in pdz])), float(NW.topic_pick(tgt, lu) == a)
        R = {}
        for i, sh, sd, tp in de.pool(rd, list(range(len(units)))):
            R[i] = (sh, sd, tp)
        print(f"\n=== ANONYMITY scripted (full K={N}, chance={chance:.3f}, soft over {RESAMPLE}) ===", flush=True)
        print(f"{'arm':9s} {'haiku':>20s} {'deepseek*':>11s} {'topic':>8s}", flush=True)
        asum = {}
        for arm in AARMS:
            idxs = [i for i in range(len(units)) if units[i][0] == arm]; gg = [units[i][1] for i in idxs]
            h = [R[i][0] for i in idxs]; d_ = [R[i][1] for i in idxs]; t = [R[i][2] for i in idxs]
            ci = cluster_mean_ci(h, gg, seed=SEED)
            asum[arm] = {"haiku": round(float(np.mean(h)), 3), "haiku_ci": ci,
                         "deepseek": round(float(np.mean(d_)), 3), "topic": round(float(np.mean(t)), 3), "n": len(idxs)}
            print(f"{arm:9s} {np.mean(h):.3f} CI{ci!s:>13s} {np.mean(d_):>11.3f} {np.mean(t):>8.3f}  (n={len(idxs)})", flush=True)
        print("  Opus single-shot (the headline) via enron_step2_dump.py COND=deid4/naive; nuwa 0.389 / aggro 0.222 reused.", flush=True)
        out["anonymity"] = asum

    # ===== UTILITY (same run; G2 = nuwa-stranger) =====
    if not os.environ.get("SKIP_UTIL"):
        T = de.TASKS
        nocard_d = dict(zip(range(len(T)), de.pool(lambda t: NW.draft(None, T[t]), list(range(len(T))))))
        units = [(a, t) for a in authors for t in range(len(T))]
        CARM = ["nuwa", "deid4", "aggro", "naive"]
        draft_jobs = []
        for i, (a, t) in enumerate(units):
            for arm in CARM:
                draft_jobs.append((i, arm, C[arm][a]))
            draft_jobs.append((i, "strg_nuwa", nuwa[strg[a]]))
        D = {}
        for jb, txt in zip(draft_jobs, de.pool(lambda jb: NW.draft(jb[2], T[units[jb[0]][1]]), draft_jobs)):
            D[(jb[0], jb[1])] = txt

        def dof(arm, i):
            return nocard_d[units[i][1]] if arm == "nocard" else D[(i, arm)]
        PAIRS = [("nuwa", "nocard"), ("nuwa", "strg_nuwa"),
                 ("deid4", "nocard"), ("deid4", "nuwa"),
                 ("aggro", "nocard"), ("aggro", "nuwa"),
                 ("naive", "nocard"), ("naive", "nuwa")]
        jobs = [(i, p) for i in range(len(units)) for p in PAIRS]
        J = {}
        for (i, p), v in zip(jobs, de.pool(lambda jb: NW.quality(T[units[jb[0]][1]], dof(jb[1][0], jb[0]),
                                                                  dof(jb[1][1], jb[0]),
                                                                  f"s2-{jb[0]}-{jb[1][0]}-{jb[1][1]}"), jobs)):
            J[(i, p)] = v
        g = [u[0] for u in units]
        print(f"\n=== UTILITY (pairwise competence, +1 first better; n={len(units)}, {N} authors) ===", flush=True)
        ures = {}
        for x, y in PAIRS:
            v = [J[(i, (x, y))] for i in range(len(units))]
            ci = cluster_mean_ci(v, g, seed=SEED)
            fl = "  <-EXCL0" if (ci[0] > 0 or ci[1] < 0) else ""
            tag = "  <<G2" if (x, y) == ("nuwa", "strg_nuwa") else ("  <<anchor" if (x, y) == ("nuwa", "nocard") else "")
            ures[f"{x}-{y}"] = {"diff": round(float(np.mean(v)), 3), "ci": ci}
            print(f"  {x:6s} vs {y:11s} = {np.mean(v):+.3f} CI{ci}{fl}{tag}", flush=True)
        print("  anchor nuwa-nocard>0 => useful Step-1 to preserve. G2 nuwa-stranger>0 => person-specific. "
              "{arm}-nuwa ~0 => Step-2 preserved utility; <0 => it gutted content.", flush=True)
        out["utility"] = ures

    out["note"] = "Step-1=nuwa anchor; Step-2 frontier {deid4 / aggro / naive} on the nuwa card. Opus via enron_step2_dump.py."
    (ROOT / "results" / "enron_step2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> results/enron_step2.json", flush=True)


if __name__ == "__main__":
    main()
