# SUPERSEDED (2026-07): legacy utility instrument, kept for the retraction trail. Canonical utility = FC forced-choice ({mad,cv,enron}_fc_export.py + cv_fc_score.py; placebo battery + delta=.10 verdict dictionary).
"""CMD utility: pairwise-judge a shared card against individual / no-card / floor / stranger arms across tasks."""
import os
import re
import sys
import json
import hashlib
from pathlib import Path
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
import enron_nuwa as NW  # noqa: E402
import cmd_gate as CG  # noqa: E402  (make_groups, synth_shared, load)
from src.attrib_metrics import cluster_mean_ci, cluster_paired_diff_ci  # noqa: E402

SE = ROOT / "data" / "enron"
RES = ROOT / "results"
SHAREDC = SE / "cmd_shared_cards.json"
K_LIST = [int(x) for x in os.environ.get("K_LIST", "2,4,8").split(",")]        # judged k
KCONC = [int(x) for x in os.environ.get("KCONC", "2,4,8,16").split(",")]        # concreteness-only k (free)
GROUP = os.environ.get("GROUP", "random")
SEED = int(os.environ.get("SEED", 0))
WS = re.compile(r"\s+")
NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b")
NUM_RE = re.compile(r"\b\d[\d,.]*\b|\$\d|%")
COND_RE = re.compile(r"\b(if|unless|when|whenever|provided|otherwise|else|in case|should)\b", re.I)
# one MECHANICAL task: a competent reply MUST cover these 5 concrete elements (keyword-scored, no judge)
MECH_TASK = ("A counterparty asks to sign a draft agreement by Friday. Reply covering: the FILING DEADLINE, the "
             "DOLLAR threshold needing approval, WHO must sign off, the FALLBACK if they refuse, and the "
             "CONFIDENTIALITY constraint.")
MECH_ELEMS = {"deadline": r"deadline|friday|by \w+day|due",
              "dollar": r"\$|dollar|threshold|amount|limit",
              "approver": r"sign[- ]?off|approv|authoriz|escalat",
              "fallback": r"fallback|if .*refus|alternativ|otherwise|backup|contingen",
              "confidential": r"confidential|nda|non-?disclos|privile"}


def tlen(c):
    return len(NW.ENC.encode(c or ""))


def concreteness(card):
    n = max(1, tlen(card))
    return {"named": len(NAME_RE.findall(card or "")) / n * 100,
            "num": len(NUM_RE.findall(card or "")) / n * 100,
            "cond": len(COND_RE.findall(card or "")) / n * 100}


def build_floor_inf(shared_cards):
    """Mechanical k→∞ floor: pool ALL shared cards, strip named entities / numbers / conditionals -> generic platitudes.
    Deterministic, reproducible; the literal limit shared@k converges toward."""
    text = "\n".join(shared_cards)
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if len(s) < 15:
            continue
        s = NAME_RE.sub("the relevant party", s)
        s = NUM_RE.sub("the agreed amount", s)
        s = COND_RE.sub("as appropriate", s)
        lines.append(s)
    seen, dedup = set(), []
    for s in lines:
        key = re.sub(r"\W+", " ", s.lower())[:60]
        if key not in seen:
            seen.add(key); dedup.append(s)
    return "\n".join(dedup[:14]) or "Act professionally and document the rationale for each decision as appropriate."


def mech_score(draft):
    d = (draft or "").lower()
    return sum(bool(re.search(pat, d)) for pat in MECH_ELEMS.values()) / len(MECH_ELEMS)


def main():
    docs, authors, nuwa, aggro, ref, raw_tgt = CG.load()
    N = len(authors)
    cache = json.loads(SHAREDC.read_text(encoding="utf-8")) if SHAREDC.exists() else {}
    SP = SE / "step2_cards_full.json"
    _S2 = json.loads(SP.read_text(encoding="utf-8")) if SP.exists() else {}
    STAAB = _S2.get("staab", {})
    HAS_STAAB = bool(STAAB) and all(a in STAAB for a in authors)            # Staab per-person arm (haiku judge ⊥ gpt-4o adversary)
    PETRE = _S2.get("petre_k4", {})
    HAS_PETRE = bool(PETRE) and all(a in PETRE for a in authors)            # PETRE suppression arm (utility pair = the frontier point)
    # ---- FILL_METHODS=1: fill the remaining comparison arms, SAME per-author UPAIRS rule as staab/petre ----
    FILLM = bool(os.environ.get("FILL_METHODS"))
    PRESIDIO = _S2.get("presidio", {}); HAS_PRESIDIO = FILLM and all(a in PRESIDIO for a in authors)
    TPAR15 = _S2.get("tpar_t15", {}); HAS_TPAR15 = FILLM and all(a in TPAR15 for a in authors)
    ARCHPOOL = _S2.get("archpool", {}); HAS_ARCH = FILLM and all(a in ARCHPOOL for a in authors)   # pooled card, published per-author (byte-identical within its archetype cluster)
    _CC = json.loads((SE / "cmd_concat_cards.json").read_text(encoding="utf-8")) if FILLM and (SE / "cmd_concat_cards.json").exists() else {}

    # ---- groups + shared cards per k (reuse gate builder; build missing) ----
    layouts = {}
    plan = []
    for k in sorted(set(K_LIST) | set(KCONC)):
        grp, byc = CG.make_groups(aggro, authors, k, SEED)
        layouts[k] = (grp, byc)
        for cid, mem in byc.items():
            ck = f"k{k}_s{SEED}_{cid}" if GROUP == "random" else f"{GROUP}_k{k}_s{SEED}_{cid}"
            if ck not in cache:
                plan.append((ck, [aggro[a] for a in mem]))
    T = de.TASKS
    # concat pooled card, looked up per-author via the standard k=8 s{SEED} clustering (same clustering as shared@8)
    concat_of = {a: _CC.get(f"k8_s{SEED}_{layouts[8][0][a]}") for a in authors} if (FILLM and _CC and 8 in layouts) else {}
    HAS_CONCAT = FILLM and bool(concat_of) and all(concat_of.get(a) for a in authors)
    UPAIRS = [("indiv", "nocard"), ("indiv", "floor"), ("own", "stranger")]            # once-pairs (author-level)
    if HAS_STAAB:
        UPAIRS += [("staab", "nocard"), ("staab", "indiv")]                            # still-useful? + didn't-gut-decisions?
    if HAS_PETRE:
        UPAIRS += [("petre_k4", "nocard"), ("petre_k4", "indiv")]                      # suppression cost: how much utility left vs gutted
    for _arm, _has in [("presidio", HAS_PRESIDIO), ("tpar_t15", HAS_TPAR15), ("archpool", HAS_ARCH), ("concat", HAS_CONCAT)]:
        if _has:
            UPAIRS += [(_arm, "nocard")]                                                # same X-vs-nocard rule as staab/petre

    if os.environ.get("PILOT_DRYRUN"):
        n_synth = len(plan)
        n_cl = {k: len(layouts[k][1]) for k in K_LIST}
        n_draft = (len(T) * 2                                   # nocard, floor per task
                   + N * len(T)                                 # indiv per author-task (stranger reuses indiv)
                   + sum(n_cl[k] for k in K_LIST) * len(T)      # shared per cluster-task (judged k only)
                   + len(MECH_ELEMS) * 0 + (N + 2 + sum(n_cl[k] for k in K_LIST)))  # mech-task drafts per arm
        n_judge = (len(UPAIRS) * N * len(T)                     # once-pairs per author-task
                   + sum((2 * n_cl[k] + N) * len(T) for k in K_LIST))  # shared vs nocard/floor per CLUSTER + vs indiv per author
        ds = n_draft * 1300; ds_out = n_draft * 400; hk = n_judge * 1400
        cost = ds / 1e6 * 0.28 + ds_out / 1e6 * 1.10 + hk / 1e6 * 1.0
        print(f"DRYRUN N={N}: shared cards to synth={n_synth}; drafts≈{n_draft} (deepseek); judge≈{n_judge} (haiku) "
              f"[{len(UPAIRS)} once-pairs×{N} + per-k(2 cluster-pairs + 1 author-pair), ×{len(T)} tasks]", flush=True)
        print(f"  concreteness on k={KCONC} (FREE); judge on k={K_LIST}; est ~${cost:.1f} "
              f"(deepseek .28/1.10, haiku 1/M; trim K_LIST or pairs if high)", flush=True)
        return

    if plan:
        print(f"synthesizing {len(plan)} ε=0 shared cards ...", flush=True)
        for (ck, _), card in zip(plan, de.pool(lambda pc: CG.synth_shared(pc[1]), plan)):
            cache[ck] = card
        SHAREDC.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    def shared_of(k, a):
        grp = layouts[k][0]
        ck = f"k{k}_s{SEED}_{grp[a]}" if GROUP == "random" else f"{GROUP}_k{k}_s{SEED}_{grp[a]}"
        return cache[ck]

    floor = build_floor_inf([cache[f"k4_s{SEED}_{c}"] for c in layouts[4][1]] if 4 in layouts
                            else list(cache.values())[:30])
    # stranger: deterministic non-cluster indiv (use k=4 clustering to define "different cluster")
    g4 = layouts[4][0] if 4 in layouts else CG.make_groups(aggro, authors, 4, SEED)[0]
    stranger = {a: nuwa[sorted([b for b in authors if g4[b] != g4[a]],
                key=lambda b: hashlib.sha1(f"str-{a}-{b}".encode()).hexdigest())[0]] for a in authors}

    # ---- concreteness vs k (FREE) ----
    print("\n=== (B) CONCRETENESS per 100 tok (named / num / cond) vs k ===", flush=True)
    conc = {"indiv": np.mean([[*concreteness(nuwa[a]).values()] for a in authors], 0).round(3).tolist()}
    print(f"  indiv          named/num/cond = {conc['indiv']}  tok={int(np.median([tlen(nuwa[a]) for a in authors]))}", flush=True)
    for k in KCONC:
        cs = [concreteness(shared_of(k, a)) for a in authors]
        m = [round(float(np.mean([c[x] for c in cs])), 3) for x in ("named", "num", "cond")]
        conc[f"shared@{k}"] = m
        print(f"  shared@{k:<3}      named/num/cond = {m}  tok={int(np.median([tlen(shared_of(k,a)) for a in authors]))}", flush=True)
    fc = concreteness(floor); print(f"  floor∞         named/num/cond = {[round(fc[x],3) for x in ('named','num','cond')]}  tok={tlen(floor)}", flush=True)

    # ---- drafts ----
    def card_of(arm, a, k=None):
        if arm == "nocard":
            return None
        if arm == "floor":
            return floor
        if arm == "shared":
            return shared_of(k, a)
        if arm == "stranger":
            return stranger[a]
        if arm == "staab":
            return STAAB[a]
        if arm == "petre_k4":
            return PETRE[a]
        if arm == "presidio":
            return PRESIDIO[a]
        if arm == "tpar_t15":
            return TPAR15[a]
        if arm == "archpool":
            return ARCHPOOL[a]
        if arm == "concat":
            return concat_of[a]
        return nuwa[a]                      # indiv / own
    draft_arms = [("nocard", None), ("indiv", None), ("floor", None), ("stranger", None)] + [("shared", k) for k in K_LIST]
    if HAS_STAAB:
        draft_arms += [("staab", None)]
    if HAS_PETRE:
        draft_arms += [("petre_k4", None)]
    for _arm, _has in [("presidio", HAS_PRESIDIO), ("tpar_t15", HAS_TPAR15), ("archpool", HAS_ARCH), ("concat", HAS_CONCAT)]:
        if _has:
            draft_arms += [(_arm, None)]
    djobs = []
    for arm, k in draft_arms:
        if arm in ("nocard", "floor"):
            djobs += [(arm, k, None, t) for t in range(len(T))]                      # author-independent
        elif arm == "shared":
            seen = set()
            for a in authors:                                                         # one per cluster
                cid = layouts[k][0][a]
                if (k, cid) not in seen:
                    seen.add((k, cid)); djobs += [(arm, k, a, t) for t in range(len(T))]
        else:
            djobs += [(arm, k, a, t) for a in authors for t in range(len(T))]
    print(f"\nbuilding {len(djobs)} drafts ...", flush=True)
    D = {}
    for (arm, k, a, t), txt in zip(djobs, de.pool(lambda j: NW.draft(card_of(j[0], j[2], j[1]), T[j[3]]), djobs)):
        D[(arm, k, a if arm not in ("nocard", "floor") else None, t)] = txt

    def dof(arm, a, t, k=None):
        if arm == "own":
            arm = "indiv"
        if arm == "shared":
            cid = layouts[k][0][a]; rep = next(b for b in authors if layouts[k][0][b] == cid)
            return D[("shared", k, rep, t)]
        a_slot = None if arm in ("nocard", "floor") else a    # all non-shared arms stored with k=None
        return D[(arm, None, a_slot, t)]

    print(f"[draft tok] " + " ".join(f"{arm}{'@'+str(k) if k else ''}="
          f"{int(np.median([tlen(v) for (ar,kk,_,_),v in D.items() if ar==arm and kk==k] or [0]))}"
          for arm, k in draft_arms), flush=True)

    # ---- (C) MECHANICAL coverage (no judge); author-independent arms drafted once, shared per-cluster ----
    print("\n=== (C) MECHANICAL required-element coverage (0..1, no judge) ===", flush=True)
    mech = {}
    for arm, k in [("nocard", None), ("indiv", None), ("floor", None)] + [("shared", k) for k in K_LIST]:
        if arm in ("nocard", "floor"):
            reps = authors[:1]                                  # author-independent card -> one draft
        elif arm == "shared":
            seen = set(); reps = [a for a in authors if (layouts[k][0][a] not in seen and not seen.add(layouts[k][0][a]))]
        else:
            reps = authors                                      # indiv per author
        sc = [mech_score(d) for d in de.pool(lambda a: NW.draft(card_of(arm, a, k), MECH_TASK), reps)]
        mech[f"{arm}{'@'+str(k) if k else ''}"] = round(float(np.mean(sc)), 3)
        print(f"  {arm}{('@'+str(k)) if k else '':<4} coverage={np.mean(sc):.3f}  (n={len(reps)})", flush=True)

    # ---- (A) pairwise judge, A/B order randomized, CLUSTER CI ----
    def judge(x, y, a, t, k=None):
        dx, dy = dof(x, a, t, k), dof(y, a, t, k)
        flip = int(hashlib.sha1(f"{x}-{y}-{a}-{t}-{k}".encode()).hexdigest(), 16) & 1
        salt = f"cmdu-{x}{k}-{y}-{a}-{t}"
        v = NW.quality(T[t], dy, dx, salt) if flip else NW.quality(T[t], dx, dy, salt)   # returns {-1,0,+1}, +1=first better
        return (-v) if flip else v                                                       # normalize to +1 = x better

    units = [(a, t) for a in authors for t in range(len(T))]
    g = [a for (a, t) in units]
    excl = lambda ci: "  <-EXCL0" if (ci[0] > 0 or ci[1] < 0) else ""
    print("\n=== (A) UTILITY pairwise (+1 = first better; CI resamples CLUSTERS) ===", flush=True)
    out = {"N": N, "group": GROUP, "concreteness": conc, "mech": mech}
    ures = {}
    ures_raw = {}                                                   # ADDITIVE: per-(author,task) raw vectors for variance decomposition
    for x, y in UPAIRS:                                              # once-pairs (author-level)
        v = [judge(x, y, a, t) for (a, t) in units]
        ci = cluster_mean_ci(v, g, seed=SEED)
        ures[f"{x}-{y}"] = {"diff": round(float(np.mean(v)), 3), "ci": ci}
        ures_raw[f"{x}-{y}"] = {"a": [a for (a, t) in units], "t": [int(t) for (a, t) in units],
                                "g4": [str(g4[a]) for (a, t) in units], "v": [int(vv) for vv in v]}
        print(f"  {x:8s} vs {y:9s} = {np.mean(v):+.3f} CI{ci}{excl(ci)}", flush=True)
    (RES / "cmd_utility_raw.json").write_text(json.dumps(ures_raw, ensure_ascii=False), encoding="utf-8")
    for k in K_LIST:
        grp = layouts[k][0]
        seen = set(); reps = [a for a in authors if (grp[a] not in seen and not seen.add(grp[a]))]
        for y in ("nocard", "floor"):                               # shared vs author-indep: judge per (cluster, task)
            ru = [(rep, t) for rep in reps for t in range(len(T))]
            v = [judge("shared", y, rep, t, k) for (rep, t) in ru]
            gg = [grp[rep] for (rep, t) in ru]
            ci = cluster_mean_ci(v, gg, seed=SEED)
            ures[f"shared@{k}-{y}"] = {"diff": round(float(np.mean(v)), 3), "ci": ci, "n_clusters": len(reps)}
            print(f"  shared@{k:<2} vs {y:9s} = {np.mean(v):+.3f} CI{ci}  (n_cl={len(reps)}){excl(ci)}", flush=True)
        v = [judge("shared", "indiv", a, t, k) for (a, t) in units]  # shared vs indiv: per author, cluster CI
        gg = [grp[a] for (a, t) in units]
        ci = cluster_mean_ci(v, gg, seed=SEED)
        ures[f"shared@{k}-indiv"] = {"diff": round(float(np.mean(v)), 3), "ci": ci, "n_clusters": len(reps)}
        print(f"  shared@{k:<2} vs {'indiv':9s} = {np.mean(v):+.3f} CI{ci}  (n_cl={len(reps)}){excl(ci)}", flush=True)
    out["utility"] = ures
    out["note"] = ("CMD utility: shared@k preserves GENERIC competence? own-vs-stranger re-confirms own≈stranger in-experiment; "
                   "concreteness vs k = mechanism; floor∞ = k→inf limit; mech = non-judge anchor. CI resamples clusters (n_eff).")
    (RES / "cmd_utility.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nsaved -> results/cmd_utility.json", flush=True)


if __name__ == "__main__":
    main()
