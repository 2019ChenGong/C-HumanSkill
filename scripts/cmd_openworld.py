"""CMD membership-inference attack (open-world): dump per-candidate 0-100 membership-scoring trials with random + topic-near negatives, namespaced by k."""
import os
import re
import sys
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
import cmd_gate as CG  # noqa: E402

DATASET = os.environ.get("DATASET", "enron")
RES = ROOT / "results" if DATASET == "enron" else ROOT / "results" / DATASET
RES = (ROOT / os.environ["RESDIR"]) if os.environ.get("RESDIR") else RES   # cross-model run isolation (e.g. results/mad/sonnet)
RES.mkdir(parents=True, exist_ok=True)
SHAREDC = CG.SHAREDC                                  # follow cmd_gate's dataset-specific shared-card cache
KCL = int(os.environ.get("KCL", 4))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
N_RNEG = int(os.environ.get("N_RNEG", 4))
N_NNEG = int(os.environ.get("N_NNEG", 4))
REF_CHARS = int(os.environ.get("REF_CHARS", 250))
WS = re.compile(r"\s+")


def main():
    docs, authors, nuwa, aggro, ref_raw, raw_tgt = CG.load()
    cache = json.loads(SHAREDC.read_text(encoding="utf-8")) if SHAREDC.exists() else {}
    # SINGLE dataset-agnostic code path: use cmd_gate.load()'s ref (candidate, REF_CHARS=250) + raw_tgt (held raw, 900).
    # Enron regression tripwire: these MUST byte-match the old docs-recompute, else the Enron number silently shifts.
    if DATASET == "enron":
        assert all(ref_raw[a] == WS.sub(" ", docs[a][CG.N_TRAIN]["text"])[:REF_CHARS] for a in authors), "ref drift"
        assert all(raw_tgt[a] == WS.sub(" ", docs[a][CG.N_TRAIN + 1]["text"])[:CG.RAW_CHARS] for a in authors), "raw drift"
    ref = ref_raw                                                                    # candidate display (250 chars)
    held = raw_tgt                                                                   # raw target (900 chars)
    refvec = {a: de._content_vec(ref[a]) for a in authors}

    def shared_card(k, s, a, grp):
        return cache[f"k{k}_s{s}_{grp[a]}"]

    # Per-person de-id baseline arms (Staab/PETRE/Presidio/DP-Prompt): same estimand as `indiv`, only the card differs.
    # Dataset-agnostic: read from cmd_gate's STEP2C (enron -> data/enron/step2_cards_full.json byte-identical to the old
    # hardcoded path; mad -> data/20mad/mad_cmd_step2.json built by mad_step2_baselines.py).
    PP = {}
    sj = json.loads(CG.STEP2C.read_text(encoding="utf-8")) if CG.STEP2C.exists() else {}
    for arm in ("staab", "staab_r1", "presidio", "tpar_t10", "tpar_t15", "petre_k4"):
        d = {a: c for a, c in sj.get(arm, {}).items() if isinstance(c, str) and c.strip()}
        if not d:
            continue
        if all(a in d for a in authors):
            PP[arm] = d
        else:                                                        # M2: present-but-incomplete -> loud, don't silently drop
            print(f"  [!] arm '{arm}' has {len(d)}/{len(authors)} cards -> EXCLUDED from MIA (rebuild it for the full set)", flush=True)

    # channel -> list of (seed,) to run
    CHAN = [("shared", SEEDS), ("raw", [SEEDS[0]]), ("indiv", [SEEDS[0]])]
    CHAN += [(arm, [SEEDS[0]]) for arm in PP]

    if os.environ.get("PILOT_DRYRUN"):
        n_trials = sum(len(authors) * len(ss) for _, ss in CHAN)
        ncand = KCL + N_RNEG + N_NNEG
        need = sum(1 for s in SEEDS for cid in CG.make_groups(aggro, authors, KCL, s)[1]
                   if f"k{KCL}_s{s}_{cid}" not in cache)
        print(f"DRYRUN: {n_trials} verification trials (shared×{len(SEEDS)} + raw×1 + indiv×1, ×{len(authors)}); "
              f"~{ncand} candidates/trial (={KCL}pos+{N_RNEG}rneg+{N_NNEG}nneg for shared); "
              f"{need} shared cards to synth (~${need*0.002:.2f}); Opus attack FREE via subagents.", flush=True)
        print(f"  trial files -> results/_ow_k{KCL}_{{chan}}_s{{seed}}_T###.txt ; key _ow_k{KCL}_{{chan}}_s{{seed}}_key.json", flush=True)
        return

    # ensure shared cards exist for all needed (k,seed)
    layouts = {}
    plan = []
    for s in SEEDS:
        grp, byc = CG.make_groups(aggro, authors, KCL, s)
        layouts[s] = (grp, byc)
        for cid, mem in byc.items():
            if f"k{KCL}_s{s}_{cid}" not in cache:
                plan.append((f"k{KCL}_s{s}_{cid}", [aggro[a] for a in mem]))
    if plan:
        print(f"synth {len(plan)} missing shared cards ...", flush=True)
        for (ck, _), card in zip(plan, de.pool(lambda pc: CG.synth_shared(pc[1]), plan)):
            cache[ck] = card
        SHAREDC.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    def pool_for(a, s, grp):
        """one seeded nested pool of other-cluster authors for target a (reused across channels)."""
        return [b for b in authors if grp[b] != grp[a]]

    def negatives(a, s, grp, target_text):
        elig = pool_for(a, s, grp)
        rneg = sorted(elig, key=lambda b: hashlib.sha1(f"ow-{s}-{a}-{b}".encode()).hexdigest())[:N_RNEG]
        tv = de._content_vec(target_text)
        npool = [b for b in elig if b not in rneg]                                        # avoid overlap with rneg
        nneg = sorted(npool, key=lambda b: -de._cosine(tv, refvec[b]))[:N_NNEG]
        ncos = [de._cosine(tv, refvec[b]) for b in nneg]                                  # diagnostic: spot cross-author near-dups
        return rneg, nneg, ncos

    nmade = 0
    nneg_cos_all = []
    for chan, seeds in CHAN:
        for s in seeds:
            grp, byc = layouts[s]
            key = {}
            for i, a in enumerate(authors, 1):
                if chan == "shared":
                    target = shared_card(KCL, s, a, grp); pos = byc[grp[a]]              # k contributors
                    tgt_kind = "a SHARED SKILL CARD distilled from a GROUP of people"
                    ttext = target
                elif chan == "indiv":
                    target = nuwa[a]; pos = [a]
                    tgt_kind = "an INDIVIDUAL SKILL CARD distilled from ONE person"
                    ttext = target
                elif chan in PP:
                    target = PP[chan][a]; pos = [a]
                    tgt_kind = "an INDIVIDUAL (de-identified) SKILL CARD distilled from ONE person"
                    ttext = target
                else:  # raw
                    target = held[a]; pos = [a]
                    tgt_kind = "a RAW WORK EMAIL by ONE person"
                    ttext = held[a]
                rneg, nneg, ncos = negatives(a, s, grp, ttext)
                nneg_cos_all += ncos
                cands = [(b, "pos") for b in pos] + [(b, "rneg") for b in rneg] + [(b, "nneg") for b in nneg]
                order = sorted(cands, key=lambda x: hashlib.sha1(f"owc-{chan}-{s}-{a}-{x[0]}".encode()).hexdigest())
                lines = ["# Authorship MEMBERSHIP verification — score EVERY candidate",
                         f"The TARGET below is {tgt_kind}. For EACH candidate, score 0-100 = how likely that candidate is",
                         "one of the people the TARGET was distilled from, judged by REASONING / DECISION STYLE — NOT by",
                         "topic (topics overlap, ignore them). Many or even ALL candidates may be unrelated; score them low.\n",
                         "TARGET:", target, "",
                         f"CANDIDATES (each a sample of their own writing); output a 0-100 score for EACH of 1..{len(order)}:"]
                for j, (b, _lab) in enumerate(order, 1):
                    lines.append(f"[{j}] {ref[b]}")
                (RES / f"_ow_k{KCL}_{chan}_s{s}_T{i:03d}.txt").write_text("\n".join(lines), encoding="utf-8")
                key[f"T{i:03d}"] = {"author": a, "cluster": grp[a], "chan": chan, "seed": s,
                                    "labels": {str(j): lab for j, (_b, lab) in enumerate(order, 1)},
                                    "cands": {str(j): b for j, (b, _lab) in enumerate(order, 1)}}
                nmade += 1
            (RES / f"_ow_k{KCL}_{chan}_s{s}_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
            print(f"  dumped {chan} s{s}: {len(authors)} trials", flush=True)
    if nneg_cos_all:
        import numpy as _np
        cc = _np.array(nneg_cos_all)
        print(f"\n[nneg content-cosine] median={_np.median(cc):.3f} p95={_np.percentile(cc,95):.3f} max={cc.max():.3f} "
              f"(>0.7 => spot-check cross-author near-dup threads; high = topic-control is strong, biases toward 'safe')", flush=True)
    print(f"dumped {nmade} verification trials -> results/_ow_k{KCL}_*  (KCL={KCL}, seeds shared={SEEDS}, neg={N_RNEG}rand+{N_NNEG}near)", flush=True)
    print(f"  next: Opus subagents score each candidate 0-100 -> _picks_ow_k{KCL}_{{chan}}_s{{seed}}.json ; then KCL={KCL} cmd_openworld_score.py", flush=True)


if __name__ == "__main__":
    main()
