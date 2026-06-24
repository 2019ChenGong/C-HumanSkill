"""Build the 4 external per-person de-id baselines (Staab / PETRE / Presidio / DP-Prompt) on the 20-MAD
nuwa cards, mirroring the Enron build so the headline MIA verdict — "every per-person de-id leaks (AUC
0.62-0.72), only CMD k-anon ≈0.5" — replicates on the SECOND dataset, not just Enron.

It REUSES the exact Enron builder FUNCTIONS (imported from enron_staab / enron_petre / enron_presidio /
enron_tpar), only swapping in 20-MAD data so the mechanism is byte-for-byte the same per-person de-id:
  - nuwa cards   : data/20mad/mad_cmd_nuwa.json   (128 devs, the indiv positive-control cards)
  - in-loop ref  : card_comments[18:24][:250]      (leak-disjoint from nuwa evidence comments[:12];
                                                     IDENTICAL construction to cmd_gate.load_mad's ref)
  - othr contrast: 4 other devs' nuwa cards         (same as the Enron builders)
Merge-appends arms staab / staab_r1 / presidio / tpar_t10 / tpar_t15 / petre_k4 into
data/20mad/mad_cmd_step2.json (NEVER replaces the existing 'aggro' key).

  ARMS=staab,petre,presidio,tpar   (default all; comma list to subset)
  PILOT_DRYRUN=1                    (cost/plan only, no API)

next: cmd_openworld.py (DATASET=mad) now loads these arms via CG.STEP2C -> dump -> Opus subagents ->
      cmd_openworld_score.py (DATASET=mad KCL=4).
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

try:                                                  # Windows consoles default to GBK -> non-ASCII prints crash
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MAD = ROOT / "data" / "20mad"
POOL = MAD / "mad_cmd_pool.json"
NUWAC = MAD / os.environ.get("NUWAC", "mad_cmd_nuwa.json")     # env-overridable for cross-model base cards (e.g. __sonnet)
STEP2C = MAD / os.environ.get("STEP2C", "mad_cmd_step2.json")  # de-id arms merge here (point at the matching base's step2)
LOCK = MAD / f".step2_build_{STEP2C.stem}.lock"                # per-output lock; merge() is read-modify-write of STEP2C
                                                      # (two builders interleaving read->write would silently drop an arm or 'aggro')
MAD_TRAIN, N_REF, REF_CHARS, KLINE = 18, 6, 250, 20   # mirror cmd_gate.load_mad ref = comments[18:24][:250]; in-loop lineup K=20
ARMS = set(x.strip() for x in os.environ.get("ARMS", "staab,petre,presidio,tpar").split(",") if x.strip())
WS = re.compile(r"\s+")


def _h(s):
    return hashlib.sha1(s.encode()).hexdigest()


def lineup(a, authors, tag):
    """19 deterministic distractors + true author, hash-shuffled — SAME construction as enron_staab/enron_petre."""
    others = sorted([b for b in authors if b != a], key=lambda b: _h(f"distract-{a}-{b}"))[:KLINE - 1]
    return sorted(others + [a], key=lambda b: _h(f"{tag}-{a}-{b}"))


def load():
    pool = json.loads(POOL.read_text(encoding="utf-8"))["pool"]
    nuwa = json.loads(NUWAC.read_text(encoding="utf-8"))["nuwa"]
    authors = [d for d in sorted(nuwa) if d in pool and len(pool[d]["card_comments"]) >= MAD_TRAIN + N_REF]
    ref = {d: WS.sub(" ", " || ".join(pool[d]["card_comments"][MAD_TRAIN:MAD_TRAIN + N_REF]))[:REF_CHARS] for d in authors}
    return nuwa, authors, ref


def merge(key, cards):
    """Merge-append ONE arm into mad_cmd_step2.json without touching 'aggro' or other arms."""
    S = json.loads(STEP2C.read_text(encoding="utf-8")) if STEP2C.exists() else {}
    S[key] = cards
    STEP2C.write_text(json.dumps(S, ensure_ascii=False), encoding="utf-8")
    print(f"  merged '{key}' (N={len(cards)}) -> {STEP2C.name}", flush=True)


# ---------------- per-arm builders (each reuses the Enron function on 20-MAD data) ----------------
def build_presidio(nuwa, authors):
    from enron_presidio import presidio_card                       # lazy: only inits spaCy if presidio is requested
    pairs = de.pool(lambda a: presidio_card(nuwa[a]), authors)
    cards = dict(zip(authors, [p[0] for p in pairs]))
    nred = [p[1] for p in pairs]
    chg = float(np.median([1.0 - len(cards[a]) / max(1, len(nuwa[a])) for a in authors]))
    print(f"  [presidio] entities/card median={int(np.median(nred))} | zero-entity cards={sum(1 for x in nred if x == 0)}/{len(authors)} "
          f"| char-change median={chg:.4f} (~0 => near no-op => MIA must ~ indiv)", flush=True)
    merge("presidio", cards)


def build_tpar(nuwa, authors):
    from enron_tpar import tpar_card
    for T, arm in [(1.0, "tpar_t10"), (1.5, "tpar_t15")]:
        cards = dict(zip(authors, de.pool(lambda a: tpar_card(nuwa[a], T), authors)))
        bad = [a for a in authors if not cards[a].strip()]
        if bad:
            print(f"  [!] {arm}: {len(bad)} EMPTY paraphrases (refusal at T={T}?) -> arm incomplete: {bad[:5]}", flush=True)
        merge(arm, cards)


def build_staab(nuwa, authors, ref):
    from enron_staab import staab_card                              # gpt-4o in-loop adversary + deepseek rewrite, R rounds
    order = {a: lineup(a, authors, "staab-inloop") for a in authors}
    othr = {a: [nuwa[b] for b in authors if b != a][:4] for a in authors}
    res = de.pool(lambda a: (a, *staab_card(a, nuwa[a], othr[a], ref, order[a])), authors)
    merge("staab", {a: fin for (a, fin, _r1, _tr) in res})
    merge("staab_r1", {a: r1 for (a, _fin, r1, _tr) in res})


def build_petre(nuwa, authors, ref):
    import enron_petre as EP                                        # qwen3.7-max in-loop attacker-guided suppression, COST_CAP
    order = {a: lineup(a, authors, "petre-inloop") for a in authors}
    othr = {a: [nuwa[b] for b in authors if b != a][:4] for a in authors}
    res = de.pool(lambda a: (a, *EP.petre_card(a, nuwa[a], othr[a], ref, order[a])), authors)
    cards = {a: c for (a, c, _i) in res}
    info = {a: i for (a, _c, i) in res}
    arm = f"petre_k{EP.K}"
    skipped = [a for a in authors if info[a].get("skipped_cap")]
    conv = sum(1 for a in authors if info[a].get("converged"))
    # cap-skipped cards are the UNMODIFIED nuwa card (== indiv) -> they would leak MAXIMALLY and FALSELY inflate the
    # "petre leaks" verdict, while still passing cmd_openworld's all-authors completeness gate. Drop them so the arm is
    # left INCOMPLETE and the MIA EXCLUDES petre_k4 entirely (honest) until COST_CAP is raised and a (cached, cheap) rerun finishes.
    for a in skipped:
        cards.pop(a, None)
    print(f"  [petre] spend ~${EP._spent['usd']:.2f}/{EP.COST_CAP} over {EP._spent['calls']} calls | "
          f"converged {conv}/{len(authors) - len(skipped)} | skipped@cap {len(skipped)}", flush=True)
    if skipped:
        print(f"  [!] {len(skipped)} devs SKIPPED at cost cap -> DROPPED from arm (now incomplete -> MIA will EXCLUDE petre_k4, "
              f"NOT mix in no-op leaky cards). Raise COST_CAP and rerun `ARMS=petre` (cached -> cheap) to complete.", flush=True)
    merge(arm, cards)


def main():
    nuwa, authors, ref = load()
    print(f"20-MAD: N={len(authors)} devs | arms={sorted(ARMS)} | ref=comments[18:24][:250] | in-loop KLINE={KLINE}", flush=True)

    if os.environ.get("PILOT_DRYRUN"):
        n = len(authors)
        est = []
        if "tpar" in ARMS:
            est.append(("tpar_t10+t15", f"{n * 2} deepseek paraphrase ~${n * 2 * 0.0008:.2f}"))
        if "staab" in ARMS:
            est.append(("staab(R=3)", f"{n * (3 * 2 + 1)} gpt-4o + {n * 3 * 2} deepseek ~$4-7"))
        if "petre" in ARMS:
            est.append(("petre_k4", f"~{n * (8 * 2 + 1)} qwen3.7-max worst / ~3 rounds typical ~$3 (cap $9)"))
        if "presidio" in ARMS:
            est.append(("presidio", f"{n} local spaCy passes FREE (needs presidio_analyzer + en_core_web_sm)"))
        print("DRYRUN cost plan:", flush=True)
        for k, v in est:
            print(f"  {k:14s} {v}", flush=True)
        print("  Opus MIA scoring FREE later via subagents.", flush=True)
        return

    try:                                                  # exclusive lock: refuse to run if another build is mid-write
        os.close(os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        sys.exit(f"another mad_step2_baselines.py is writing {STEP2C.name} (lock {LOCK.name}); run arms SERIALLY, never "
                 f"concurrently. if it crashed, delete {LOCK} and retry.")
    try:
        if "presidio" in ARMS:
            try:
                build_presidio(nuwa, authors)
            except Exception as e:  # noqa: BLE001
                print(f"  [!] presidio SKIPPED ({type(e).__name__}: {e}). pip install presidio_analyzer presidio_anonymizer && "
                      f"python -m spacy download en_core_web_sm  -> then ARMS=presidio rerun.", flush=True)
        if "tpar" in ARMS:
            build_tpar(nuwa, authors)
        if "staab" in ARMS:
            build_staab(nuwa, authors, ref)
        if "petre" in ARMS:
            build_petre(nuwa, authors, ref)
    finally:
        LOCK.unlink(missing_ok=True)

    S = json.loads(STEP2C.read_text(encoding="utf-8")) if STEP2C.exists() else {}
    have = [k for k in ("aggro", "staab", "staab_r1", "presidio", "tpar_t10", "tpar_t15", "petre_k4") if k in S]
    print(f"\n{STEP2C.name} now has arms: {have}", flush=True)
    print("  next: DATASET=mad KCL=4 python scripts/cmd_openworld.py  (dump) -> cmd_batch.py -> Opus subagents -> cmd_openworld_score.py", flush=True)


if __name__ == "__main__":
    main()
