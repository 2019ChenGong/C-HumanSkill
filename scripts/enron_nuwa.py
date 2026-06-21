"""Shared helpers: skill-card drafting, pairwise quality judge, re-identification lineup and the LLM detective."""
import os
import re
import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
from src.llm import chat, sample_one  # noqa: E402
from src.attrib_metrics import cluster_mean_ci, cluster_paired_diff_ci  # noqa: E402

GEN = "deepseek-chat"
HEADLINE = "claude-haiku-4-5"
SECOND = "deepseek-chat"
SE = ROOT / "data" / "enron"
COMPC = SE / "comp_anon_cards.json"
NUWAC = SE / "nuwa_cards.json"
N_AUTH, N_TRAIN, N_TGT, REF_CHARS = 18, 12, 2, 500
RESAMPLE = int(os.environ.get("RESAMPLE", 2))
SEED = 0
ENC = tiktoken.get_encoding("cl100k_base")
WS = re.compile(r"\s+")


# ---------------- STEP 1: faithful nuwa cognitive-OS (2-call) ----------------
def nuwa_extract(emails):
    body = "\n\n---\n\n".join(emails[:12])
    msg = [{"role": "system", "content": "You reverse-engineer an expert's COGNITIVE OPERATING SYSTEM from their work."},
           {"role": "user", "content": f"Work emails by ONE person (identifiers masked):\n\n{body}\n\nIdentify the "
            "underlying DECISION FRAMEWORKS, mental models, heuristics, characteristic moves, and failure modes these "
            "reveal -- and tie each to the evidence. Output terse notes grouped: Frameworks / Heuristics / Characteristic "
            "moves / Failure modes. Derive only from the texts."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=600) or ""


def nuwa_assemble(notes):
    msg = [{"role": "system", "content": "You compile a REUSABLE cognitive operating system a stranger could EXECUTE."},
           {"role": "user", "content": f"Notes on how a person decides:\n\n{notes}\n\nCompile a COGNITIVE OPERATING "
            "SYSTEM card a colleague could EXECUTE -- second-person, situation-triggered procedures, NOT a biography. "
            "Sections: [Analysis Protocol] step-by-step approach to a work decision; [Decision Frameworks & Mental "
            "Models]; [Heuristics] each as 'When X -> do Y, watch for Z'; [Characteristic Moves]; [Failure Modes]. "
            "Write each as an EXECUTABLE instruction generic enough that a stranger could run it. Do NOT name any "
            "person, company, or specific deal. ~800-1000 words of card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=1100) or ""


def nuwa_deid_card(card):
    msg = [{"role": "system", "content": "You generalize a cognitive-OS card so it reveals no specific person/desk/domain."},
           {"role": "user", "content": f"Cognitive-OS card:\n\n{card}\n\nRewrite to REMOVE anything revealing the "
            "person's specific job, department, domain, or the kinds of situations/deals/systems they handle. Keep ONLY "
            "generic, transferable, EXECUTABLE decision procedures applicable in ANY role. Strip domain nouns and role "
            "markers; keep concrete usable heuristics (no empty platitudes). Keep the section structure and roughly the "
            "same length. Output ONLY the rewritten card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=1100) or ""


def padded_comp_card(card, target_tok):
    msg = [{"role": "system", "content": "You expand a skill card to a target length WITHOUT adding new content."},
           {"role": "user", "content": f"Skill card:\n\n{card}\n\nExpand it to about {target_tok} tokens: elaborate "
            "each existing bullet into a fuller paragraph with generic illustrative detail, but add NO new heuristics, "
            "NO new decisions, and NO identity/names. Same substance, just longer prose. Output ONLY the expanded card."}]
    return chat(msg, model=GEN, temperature=0.3, max_tokens=1100) or ""


# ---------------- utility (pairwise competence judge) ----------------
def draft(card, task):
    sysm = ("You are an Enron employee handling work email. Respond to the situation the way a competent employee "
            "would. Output only the email reply.")
    if card:
        sysm = ("You are an Enron employee. Use the working/decision profile below to respond. Output only the email reply.")
        user = f"Your profile:\n{card}\n\nSituation:\n{task}\n\nWrite ONLY your email reply, following any format the situation specifies."
    else:
        user = f"Situation:\n{task}\n\nWrite ONLY your email reply, following any format the situation specifies."
    return chat([{"role": "system", "content": sysm}, {"role": "user", "content": user}], model=GEN, temperature=0.5, max_tokens=400) or ""


def quality(task, a, b, salt):
    swap = int(hashlib.sha1(salt.encode()).hexdigest(), 16) % 2 == 0
    A, B = (b, a) if swap else (a, b)
    out = (chat([{"role": "system", "content": "You compare two email replies to the SAME work situation and decide "
                  "which handles it MORE competently and usefully -- better judgment, more complete, more actionable. "
                  "IGNORE writing style, length, tone, identity."},
                 {"role": "user", "content": f"Situation:\n{task}\n\nReply A:\n{A}\n\nReply B:\n{B}\n\nWhich is more "
                  "competent? Answer ONLY 'A' or 'B' (or 'TIE')."}], model=HEADLINE, temperature=0.0, max_tokens=4) or "").strip().lower()
    v = 1 if out.startswith("a") else (-1 if out.startswith("b") else 0)
    return -v if swap else v


# ---------------- anonymity detective ----------------
def detective(target, refs, model, salt):
    block = "\n".join(f"[{slot}] {rt}" for slot, _a, rt in refs)
    msg = [{"role": "system", "content": "You identify which candidate authored a target, from how the author thinks, "
            "decides, and writes -- NOT from the topic (topics overlap)."},
           {"role": "user", "content": f"TARGET (from ONE author):\n{target}\n\nCANDIDATES (each a sample of their own "
            f"writing):\n{block}\n\nWhich candidate [number] is the SAME author? Match by reasoning/decision style, NOT "
            "topic. Answer ONLY the number."}]
    out = sample_one(msg, model, s=0, temperature=0.7, max_tokens=6, salt=salt) or ""
    nums = [int(x) for x in re.findall(r"\d+", out) if 1 <= int(x) <= len(refs)]
    return refs[nums[-1] - 1][1] if nums else None


def topic_pick(target, refs):
    qt = de._content_vec(target); best, bs = None, 0.0
    for _s, a, rt in refs:
        s = de._cosine(qt, de._content_vec(rt))
        if s > bs:
            bs, best = s, a
    return best


def lineup(authors, ref, ta, tag):
    order = sorted(authors, key=lambda b: hashlib.sha1(f"{tag}-{ta}-{b}".encode()).hexdigest())
    return [(i + 1, b, ref[b]) for i, b in enumerate(order)]


def main():
    docs = de.get_docs()
    need = N_TRAIN + 1 + N_TGT
    authors = [a for a in sorted(docs, key=lambda a: -len(docs[a])) if len(docs[a]) >= need][:N_AUTH]
    N = len(authors); chance = 1.0 / N
    ref = {a: WS.sub(" ", docs[a][N_TRAIN]["text"])[:REF_CHARS] for a in authors}
    tr = {a: [docs[a][j]["text"] for j in range(N_TRAIN)] for a in authors}
    comp = {a: json.loads(COMPC.read_text(encoding="utf-8"))["comp"][a][0] for a in authors}
    # topic-matched stranger (max content-word cosine over ref)
    V = {a: de._content_vec(ref[a]) for a in authors}
    strg = {a: max((b for b in authors if b != a), key=lambda b: de._cosine(V[a], V[b])) for a in authors}

    if NUWAC.exists() and not os.environ.get("REBUILD"):
        C = json.loads(NUWAC.read_text(encoding="utf-8"))
        nuwa, nuwa_deid, padded = C["nuwa"], C["nuwa_deid"], C["padded"]
    else:
        print("STEP1: building nuwa cognitive-OS cards (2-call) ...", flush=True)
        nuwa = dict(zip(authors, de.pool(lambda a: nuwa_assemble(nuwa_extract(tr[a])), authors)))
        print("STEP2: building nuwa_deid ...", flush=True)
        nuwa_deid = dict(zip(authors, de.pool(lambda a: nuwa_deid_card(nuwa[a]), authors)))
        print("LENGTH-CONTROL: building padded_comp ...", flush=True)
        tgt_tok = int(np.median([len(ENC.encode(nuwa[a])) for a in authors]))
        padded = dict(zip(authors, de.pool(lambda a: padded_comp_card(comp[a], tgt_tok), authors)))
        NUWAC.write_text(json.dumps({"nuwa": nuwa, "nuwa_deid": nuwa_deid, "padded": padded}, ensure_ascii=False), encoding="utf-8")

    def toklen(d):
        return int(np.median([len(ENC.encode(d[a])) for a in authors]))
    lens = {"comp": toklen(comp), "nuwa": toklen(nuwa), "nuwa_deid": toklen(nuwa_deid), "padded_comp": toklen(padded)}
    print(f"\n[CARD TOKEN LENGTHS median] {lens}  (P3: padded_comp ~ nuwa so re-id diff != length)", flush=True)

    if os.environ.get("PILOT_DRYRUN"):
        nb = N * len(de.TASKS)
        print(f"DRYRUN authors={N}; UTILITY drafts={nb*4+len(de.TASKS)} judges={nb*4}; "
              f"ANON units=({N*N_TGT}+{N*3}) resample {RESAMPLE} x2 models.", flush=True)
        return

    out = {"N": N, "chance": round(chance, 4), "card_tok_len": lens}

    # ===== UTILITY (same-run; GATE = nuwa-nocard) =====
    if not os.environ.get("SKIP_UTIL"):
        T = de.TASKS
        UARMS = ["nocard", "comp", "nuwa", "strg_nuwa", "nuwa_deid"]
        nocard_d = {t: draft(None, T[t]) for t in range(len(T))}
        def card_u(arm, a):
            return {"comp": comp[a], "nuwa": nuwa[a], "strg_nuwa": nuwa[strg[a]], "nuwa_deid": nuwa_deid[a]}[arm]
        units = [(a, t) for a in authors for t in range(len(T))]
        D = {}
        for (i, arm), txt in zip([(i, arm) for i in range(len(units)) for arm in UARMS[1:]],
                                 de.pool(lambda j: draft(card_u(j[1], units[j[0]][0]), T[units[j[0]][1]]),
                                         [(i, arm) for i in range(len(units)) for arm in UARMS[1:]])):
            D[(i, arm)] = txt
        def dof(arm, i):
            return nocard_d[units[i][1]] if arm == "nocard" else D[(i, arm)]
        PAIRS = [("comp", "nocard"), ("nuwa", "nocard"), ("nuwa", "strg_nuwa"), ("nuwa_deid", "nuwa")]
        J = {}
        for (i, p), v in zip([(i, p) for i in range(len(units)) for p in PAIRS],
                             de.pool(lambda j: quality(T[units[j[0]][1]], dof(j[1][0], j[0]), dof(j[1][1], j[0]),
                                                       f"u-{j[0]}-{j[1][0]}-{j[1][1]}"),
                                     [(i, p) for i in range(len(units)) for p in PAIRS])):
            J[(i, p)] = v
        g = [u[0] for u in units]
        print(f"\n=== UTILITY (pairwise competence, +1 first better; n={len(units)}, {N} authors) ===", flush=True)
        ures = {}
        for x, y in PAIRS:
            v = [J[(i, (x, y))] for i in range(len(units))]
            ci = cluster_mean_ci(v, g, seed=SEED)
            fl = "  <-EXCL0" if (ci[0] > 0 or ci[1] < 0) else ""
            ures[f"{x}-{y}"] = {"diff": round(float(np.mean(v)), 3), "ci": ci}
            tag = "  <<GATE" if (x, y) == ("nuwa", "nocard") else ""
            print(f"  {x:10s} vs {y:10s} = {np.mean(v):+.3f} CI{ci}{fl}{tag}", flush=True)
        print("  GATE nuwa-nocard>0 => utility EXISTS to preserve; else nuwa_deid-nuwa~0 is vacuous (P1).", flush=True)
        out["utility"] = ures

    # ===== ANONYMITY (same-run scripted, length-controlled) =====
    if not os.environ.get("SKIP_ANON"):
        units = []
        for a in authors:
            for j in range(N_TGT):
                units.append(("comment", a, j, WS.sub(" ", docs[a][N_TRAIN + 1 + j]["text"])[:1200]))
            units.append(("comp", a, 0, comp[a]))
            units.append(("nuwa", a, 0, nuwa[a]))
            units.append(("nuwa_deid", a, 0, nuwa_deid[a]))
            units.append(("padded_comp", a, 0, padded[a]))
        AARMS = ["comment", "comp", "nuwa", "nuwa_deid", "padded_comp"]
        def rd(i):
            arm, a, k, tgt = units[i]
            lu = lineup(authors, ref, a, f"{arm}-{k}")
            ph = [detective(tgt, lu, HEADLINE, f"{arm}-{a}-{k}-h{r}") for r in range(RESAMPLE)]
            pd = [detective(tgt, lu, SECOND, f"{arm}-{a}-{k}-d{r}") for r in range(RESAMPLE)]
            return i, float(np.mean([p == a for p in ph])), float(np.mean([p == a for p in pd])), float(topic_pick(tgt, lu) == a)
        R = {}
        for i, sh, sd, tp in de.pool(rd, list(range(len(units)))):
            R[i] = (sh, sd, tp)
        print(f"\n=== ANONYMITY single-shot (full K={N}, chance={chance:.3f}, soft over {RESAMPLE}) ===", flush=True)
        print(f"{'arm':12s} {'haiku':>20s} {'deepseek*':>11s} {'topic':>8s}", flush=True)
        asum = {}
        for arm in AARMS:
            idxs = [i for i in range(len(units)) if units[i][0] == arm]; gg = [units[i][1] for i in idxs]
            h = [R[i][0] for i in idxs]; d_ = [R[i][1] for i in idxs]; t = [R[i][2] for i in idxs]
            ci = cluster_mean_ci(h, gg, seed=SEED)
            asum[arm] = {"haiku": round(float(np.mean(h)), 3), "haiku_ci": ci, "deepseek": round(float(np.mean(d_)), 3), "topic": round(float(np.mean(t)), 3), "n": len(idxs)}
            print(f"{arm:12s} {np.mean(h):.3f} CI{ci!s:>13s} {np.mean(d_):>11.3f} {np.mean(t):>8.3f}  (n={len(idxs)})", flush=True)
        print("  read: nuwa vs comp = does richer/abstracted card change re-id? padded_comp = length control "
              "(if padded~nuwa, identity gain is LENGTH not depth). nuwa_deid = step-2 anonymization.", flush=True)
        out["anonymity"] = asum

    out["note"] = ("nuwa(faithful cognitive-OS) step1 + our step2 anonymization on Enron, same-run head-to-head vs comp "
                   "+ padded_comp length control. Opus single-shot via enron_nuwa_dump.py.")
    (ROOT / "results" / "enron_nuwa.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> results/enron_nuwa.json", flush=True)


if __name__ == "__main__":
    main()
