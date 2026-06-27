"""2AFC paired-discrimination membership attacker (reproducible, scripted, cached) for CMD pooled cards.

WHY: the legacy open-world attacker (cmd_openworld.py -> Opus subagent 0-100 picks) DEGENERATES on de-identified
cards: it stops conditioning on the (generic, near-identical) target card and scores each candidate's 250-char ref
in ISOLATION. Evidence (MAD k8 s0): one candidate gets the SAME score across 14 distinct target clusters; tie-rate
1.000; candidate identity explains 100% of negative-score variance; nneg-AUC 0.262 (below chance) = artifact.

FIX (resolution, not calibration): force the attacker to USE the card via a 2-alternative forced choice. Show the
TARGET card + two writing samples A,B (exactly one is a true contributor, the other a stranger). Ask which one
contributed, judged by REASONING/DECISION STYLE (topic held constant for the nneg pairing). This yields a
continuous, tie-free statistic with chance = 0.5, no per-candidate calibration, and (by construction) the model
cannot answer without reading the card. Ensemble T temperature-samples for a continuous confidence + variance cut.

NEG TYPES per (card C, member m):
  nneg = same-TOPIC stranger (topically near C, not a co-member)  -> IDENTITY headline (topic controlled)
  rneg = random stranger (not a co-member)                        -> auxiliary (topic-gameable)
CHANNELS:
  shared = CMD pooled card (the thing under test)
  indiv  = the member's OWN individual card  -> POSITIVE CONTROL (a working attacker MUST win this; chance=0.5)

Run:  DATASET=mad KCL=8 SEED=0 ATTACK_MODEL=deepseek-chat T_ENS=3 M_NNEG=2 M_RNEG=2 MODE=run  python scripts/cmd_attack2afc.py
Modes: dryrun (count calls + cost, no spend) | sample (print a few real prompts+answers) | run (full, cached).
Output: results[/{ds}]/_2afc_k{K}_s{S}_{model}.json  = list of pair records {chan,neg,card_id,member,stranger,p_member,acc,n}.
"""
import os
import re
import sys
import json
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import deid_enron as de  # noqa: E402
import cmd_gate as CG  # noqa: E402
from src.llm import sample_one  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DS = os.environ.get("DATASET", "mad")
RES = ROOT / "results" if DS == "enron" else ROOT / "results" / DS
RES = (ROOT / os.environ["RESDIR"]) if os.environ.get("RESDIR") else RES
RES.mkdir(parents=True, exist_ok=True)
KCL = int(os.environ.get("KCL", 8))
SEED = int(os.environ.get("SEED", 0))
MODEL = os.environ.get("ATTACK_MODEL", "deepseek-chat")
T_ENS = int(os.environ.get("T_ENS", 3))
M_NNEG = int(os.environ.get("M_NNEG", 2))
M_RNEG = int(os.environ.get("M_RNEG", 2))
NNEG_MATCH = os.environ.get("NNEG_MATCH", "member")   # member = stranger ~ MEMBER's topic (clean 2AFC control);
#                                                       card = stranger ~ CARD's topic (legacy; confounded for pooled)
TEMP = float(os.environ.get("ATTACK_TEMP", 0.7))
MODE = os.environ.get("MODE", "dryrun")
CHANS = os.environ.get("CHANS", "shared,indiv").split(",")

SYS = ("You are a forensic authorship analyst. You compare HOW people reason and make decisions — their decision "
       "architecture, sequencing, escalation, risk posture, what they prioritize — NOT what topic they write about. "
       "Topics overlap and are NOT evidence of shared authorship.")
USR_CARD = ("A SKILL CARD below was distilled from the working/decision style of one or more people.\n\n"
            "SKILL CARD:\n{card}\n\n"
            "Two writing samples follow, from two DIFFERENT people. EXACTLY ONE of them is a person whose work was "
            "used to build the card; the OTHER writes about similar topics but did NOT contribute.\n\n"
            "[A] {a}\n\n[B] {b}\n\n"
            "Which person — A or B — contributed to the card, judged ONLY by reasoning / decision style (ignore "
            "topic)? Answer with EXACTLY one line: the letter (A or B), a space, then a confidence 50-100 (50=pure "
            "guess, 100=certain). Example: `A 72`. Output ONLY that line.")
USR_WRITE = ("A WRITING SAMPLE below is by ONE person.\n\n"
             "WRITING SAMPLE:\n{card}\n\n"
             "Two more writing samples follow, from two DIFFERENT people. EXACTLY ONE of them is by the SAME person "
             "as the sample above; the OTHER is a different person who writes about similar topics.\n\n"
             "[A] {a}\n\n[B] {b}\n\n"
             "Which — A or B — is by the SAME author as the sample above, judged by reasoning / decision style and "
             "writing idiolect (ignore shared topic)? Answer with EXACTLY one line: the letter (A or B), a space, "
             "then a confidence 50-100 (50=pure guess, 100=certain). Example: `A 72`. Output ONLY that line.")
USR = {"card": USR_CARD, "writing": USR_WRITE}


def _h(*xs):
    return hashlib.sha1("|".join(map(str, xs)).encode()).hexdigest()


def parse(out):
    """-> (choice in {'A','B',None}, confidence 50-100). Robust to chatter."""
    if not out:
        return None, 50.0
    m = re.search(r"\b([AB])\b", out)
    choice = m.group(1) if m else None
    nums = [float(x) for x in re.findall(r"\d{2,3}", out)]
    conf = next((n for n in nums if 50 <= n <= 100), 50.0)
    return choice, conf


def p_member_one(card, member_ref, stranger_ref, salt, s, kind="card"):
    """One sampled 2AFC -> P(model says the MEMBER contributed) in [0,1], order randomized by salt."""
    swap = (int(_h("swap", salt), 16) % 2 == 1)        # randomize which slot is the member
    a, b = (stranger_ref, member_ref) if swap else (member_ref, stranger_ref)
    member_slot = "B" if swap else "A"
    msg = [{"role": "system", "content": SYS},
           {"role": "user", "content": USR[kind].format(card=card, a=a, b=b)}]
    out = sample_one(msg, MODEL, s=s, temperature=TEMP, max_tokens=12, salt=salt)
    choice, conf = parse(out)
    if choice is None:
        return 0.5, out
    picked_member = (choice == member_slot)
    # confidence-weighted prob the MEMBER is the contributor
    return (conf / 100.0 if picked_member else 1.0 - conf / 100.0), out


def build_pairs():
    """Construct all 2AFC pairs (no API). Returns list of dicts + the card/ref text lookups."""
    _docs, authors, nuwa, aggro, ref, raw_tgt = CG.load()
    cache = json.loads(CG.SHAREDC.read_text(encoding="utf-8")) if CG.SHAREDC.exists() else {}
    grp, byc = CG.make_groups(aggro, authors, KCL, SEED)
    refvec = {a: de._content_vec(ref[a]) for a in authors}
    aset = set(authors)

    pairs = []
    for cid, mem in byc.items():
        if len(mem) < KCL:
            continue
        ck = f"k{KCL}_s{SEED}_{cid}"
        shared = cache.get(ck)
        non = [b for b in authors if grp[b] != cid]                 # non-members (other clusters) = stranger pool
        for m in mem:
            # same-topic strangers. NNEG_MATCH=member (clean): nearest the MEMBER's ref -> member & stranger share
            # topic, card fixed, only cue is identity. =card (legacy/confounded): nearest the generic pooled CARD,
            # which for a pooled card selects a stranger that out-matches the member by content (drives acc<0.5).
            matchvec = refvec[m] if NNEG_MATCH == "member" else (de._content_vec(shared) if shared else refvec[m])
            nn = sorted(non, key=lambda b: -de._cosine(matchvec, refvec[b]))[:M_NNEG]
            # random strangers: hash-ordered non-members disjoint from nn
            rr = [b for b in sorted(non, key=lambda b: _h("r", SEED, cid, m, b)) if b not in nn][:M_RNEG]
            for chan in CHANS:
                if chan == "shared":
                    card, kind = shared, "card"
                elif chan == "indiv":
                    card, kind = nuwa.get(m), "card"
                else:  # raw = pure-authorship 2AFC (target = member's held-out raw writing) — easiest positive control
                    card, kind = raw_tgt.get(m), "writing"
                if not card:
                    continue
                for nlab, strangers in (("nneg", nn), ("rneg", rr)):
                    for st in strangers:
                        pairs.append({"chan": chan, "neg": nlab, "card_id": cid, "member": m, "stranger": st,
                                      "_card": card, "_mref": ref[m], "_sref": ref[st], "_kind": kind})
    return pairs, len(byc)


def main():
    pairs, nclus = build_pairs()
    ncalls = len(pairs) * T_ENS
    by = {}
    for p in pairs:
        by[(p["chan"], p["neg"])] = by.get((p["chan"], p["neg"]), 0) + 1
    print(f"=== 2AFC attacker  DS={DS} k{KCL} s{SEED} model={MODEL} T={T_ENS} ===")
    print(f"clusters={nclus}  pairs={len(pairs)}  ({', '.join(f'{k[0]}/{k[1]}:{v}' for k,v in sorted(by.items()))})")
    print(f"LLM calls = pairs×T = {ncalls}  (~{ncalls*0.7/1000:.1f}k in-tok @ ~700/call; deepseek ≈ ${ncalls*700/1e6*0.3:.2f}, cached reruns free)")
    if MODE == "dryrun":
        print("DRYRUN: no spend. Set MODE=sample to see real Q&A, MODE=run for the full cached scoring.")
        return

    todo = pairs if MODE == "run" else pairs[:4]
    # flatten (pair, sample) into one job list -> parallel via de.pool (ThreadPoolExecutor)
    jobs = [(i, s) for i in range(len(todo)) for s in range(T_ENS)]

    def do(job):
        i, s = job
        p = todo[i]
        salt_base = _h(p["chan"], p["neg"], p["card_id"], p["member"], p["stranger"], SEED)
        pm, raw = p_member_one(p["_card"], p["_mref"], p["_sref"], salt_base + f"-{s}", s, kind=p["_kind"])
        return i, pm, raw

    results = de.pool(do, jobs) if MODE == "run" else [do(j) for j in jobs]
    per = {}
    for i, pm, raw in results:
        per.setdefault(i, {"ps": [], "raws": []})
        per[i]["ps"].append(pm); per[i]["raws"].append(raw)

    out_recs = []
    for i, p in enumerate(todo):
        ps, raws = per[i]["ps"], per[i]["raws"]
        rec = {k: v for k, v in p.items() if not k.startswith("_")}
        rec["p_member"] = float(np.mean(ps)); rec["acc"] = float(np.mean([x > 0.5 for x in ps])); rec["n"] = len(ps)
        out_recs.append(rec)
        if MODE == "sample":
            print(f"\n--- pair {i}: {p['chan']}/{p['neg']} card={p['card_id']} member={p['member']} vs stranger={p['stranger']}")
            print(f"    raws={raws}  -> p_member(mean)={rec['p_member']:.2f} acc={rec['acc']:.2f}")
    if MODE == "sample":
        print("\n(sample mode: scored 4 pairs only — eyeball the format/answers before MODE=run)")
        return

    safe_model = MODEL.replace("/", "_")
    outp = RES / f"_2afc_k{KCL}_s{SEED}_{safe_model}_{NNEG_MATCH}.json"
    outp.write_text(json.dumps(out_recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nsaved {len(out_recs)} pair records -> {outp.relative_to(ROOT)}")
    # quick on-the-spot summary (full scoring/CI/gates in cmd_attack2afc_score.py)
    for chan in CHANS:
        for nlab in ("nneg", "rneg"):
            sub = [r for r in out_recs if r["chan"] == chan and r["neg"] == nlab]
            if sub:
                acc = np.mean([r["acc"] for r in sub]); pm = np.mean([r["p_member"] for r in sub])
                print(f"  {chan:6s}/{nlab}: 2AFC acc={acc:.3f}  mean P(member)={pm:.3f}  (n={len(sub)}, chance=0.5)")


if __name__ == "__main__":
    main()
