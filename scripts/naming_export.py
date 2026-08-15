"""Full-cohort NAMING attack (knockout-tournament re-identification) — the 2AFC's intuitive companion read.

PRE-REGISTERED DESIGN (2026-07-21, chat-approved; ILLUSTRATION, not a new headline instrument):
- Question: given a released card, can an attacker NAME its contributors out of the ENTIRE clustered cohort?
- Form: knockout tournament of (up-to-)8-way forced choices. Reuses the proven read-the-card FC primitive; the
  retired 0-100 independent scorer is NOT resurrected (it degenerates on de-identified cards: ignores the card,
  tie-rate 1.0, the legacy .262 artifact). One long 100+-candidate prompt is also banned (attention dilution =
  weak attacker = fake null).
- Arms: indiv = a member's OWN individual card, INDIV_PER_CLUSTER hash-picked members/cluster, 1 bracket seed
          -> POSITIVE CONTROL (scorer gate: hit-rate CI must clear chance, else the whole wave is VOID);
        v6    = canonical sanitized pooled card (cmd_shared_cards*__v6min.json), every full cluster,
          BSEEDS_V6 bracket seeds -> the illustration under test.
- Hit: v6 = final pick is ANY of the card's k members (chance = k/pool); indiv = final pick IS the card's owner
  (chance = 1/pool). Topic is deliberately NOT controlled — attacker-friendly, conservative for the defense claim.
- Stats plan (naming_score.py): cluster bootstrap over card_id. The v6 null is expected UNDERPOWERED and stays an
  illustration; the POWERED cross-checks are (a) round-1 conditional 8-way accuracy on groups containing >=1 true
  member and (b) member survival depth. Do not upgrade a null here into a claim.
- Canonical seeds/pools (k8): enron SEED=1 pool 116 (chance .0714/.0086); mad SEED=0 pool 128 (.0625/.0078);
  cv SEED=0 pool 77 (.1111/.0130).

EXTENSION (2026-07-22, chat-approved; per-person de-id arms — separate BATCHDIR, legacy run untouched):
- ARMS=staab,petre_k4,tpar_t15 BATCHDIR=results/{ds}/naming_deid: owner-type arms. Same hash-picked
  INDIV_PER_CLUSTER owners/cluster as the legacy indiv arm AND the same partition salt ("indiv") -> each owner
  faces the identical r1 candidate groups across arms/runs; only the card text differs (slot order is
  re-shuffled per (arm, group) so paired groups don't share the letter layout). Cards come straight from the
  per-dataset STEP2C de-id files, canonical 2AFC-ladder channels staab / petre_k4 / tpar_t15 (full-pool
  coverage verified 2026-07-22). No indiv or v6 rerun.
- GATE (pre-registered): petre_k4 is a documented no-op arm (45-53% cards byte-identical to the owner's indiv
  card, results/petre_noop_census.json) -> its r1-conditional read, POOLED across the 3 datasets (cluster
  bootstrap, scripts/naming_pooled_gate.py), must clear chance (CI-lo > pooled chance) = attacker-has-teeth.
  Per-dataset and no-op-subset reads are printed as transparency and as the bridge to the legacy indiv arm
  (same owners, same groups up to slot order). If the pooled gate fails -> rerun the indiv arm before
  interpreting any staab/tpar null.
- H2 (prediction from the 2AFC ladder .58-.74): staab/tpar r1-conditional falls between the legacy indiv read
  and chance; petre ~= indiv. Final-hit reads expected und (cluster-count destiny, 3/3 precedent) — narrative
  only. Comparisons via per-arm lift vs own chance line; never upgrade a null.

Rounds are SEQUENTIAL free-sonnet-subagent waves over self-contained {pid,prompt} batches (ans_*.json protocol,
same as the 2AFC packs; subagents read ONLY sys.txt + their batch file):
  DATASET=enron SEED=1 ROUND=1 python -P scripts/naming_export.py     # export round-1 pack -> run wave
  DATASET=enron SEED=1 ROUND=2 python -P scripts/naming_export.py     # consumes round1 ans_*.json
  DATASET=enron SEED=1 ROUND=3 python -P scripts/naming_export.py     # final 2-way pick
  DATASET=enron SEED=1 python -P scripts/naming_score.py
"""
import os
import re
import sys
import json
import math
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cmd_gate as CG  # noqa: E402  (DATASET env must be set before this import; it is, via the run command)
import naming_refsrc as RS  # noqa: E402  R20 candidate-sample source (shared with naming_refsrc_diag.py)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DS = os.environ.get("DATASET", "enron")
KCL = int(os.environ.get("KCL", 8))
SEED = int(os.environ.get("SEED", 1 if DS == "enron" else 0))   # canonical v6 seeds: enron s1, mad/cv s0
ROUNDN = int(os.environ.get("ROUND", 1))
BSEEDS_V6 = int(os.environ.get("BSEEDS_V6", 2))
INDIV_PC = int(os.environ.get("INDIV_PER_CLUSTER", 2))
# R17 scale-up (prereg results/NAMING_R17_SCALEUP_PREREG.md): owners/seeds are taken as a RANK SLICE,
# so an already-measured slice can be skipped and only the increment exported. A given owner's r1
# partition is salted by (arm-salt, target, bseed, round) ONLY -> independent of how many owners run,
# hence the legacy slice stays bit-identical and both directories merge at analysis time.
INDIV_SKIP = int(os.environ.get("INDIV_SKIP", 0))
BSEED_SKIP = int(os.environ.get("BSEED_SKIP", 0))
TPB = int(os.environ.get("TASKS_PER_BATCH", 40))
V6C_DEFAULT = {"enron": "cmd_shared_cards__v6min.json", "mad": "cmd_shared_cards_mad__v6min.json",
               "cv": "cmd_shared_cards_cv__v6min.json"}
V6C = CG.SE / os.environ.get("V6C", V6C_DEFAULT[DS])
ARMS = [a.strip() for a in os.environ.get("ARMS", "v6,indiv").split(",") if a.strip()]
DEIDC_DEFAULT = {"enron": "step2_cards_full.json", "mad": "mad_cmd_step2.json", "cv": "cv_cmd_step2.json"}
DEIDC = CG.SE / os.environ.get("DEIDC", DEIDC_DEFAULT[DS])  # de-id cards loaded directly; CG.STEP2C env is NOT
                                                            # overridden, so CG.load()/make_groups stay canonical
# A1/A2 (#153/#154, prereg results/NAMING_A12_POOLBASE_PREREG.md): `ne` (un-sanitized CMD) and `concat`
# (naive union-merge) are POOLED-card arms exactly like v6 -- one card per CLUSTER, hit = ANY of the k
# members, chance = k/pool, statistical unit = the cluster. POOLED is exported so every scorer decides
# "pooled vs owner-type" from ONE definition instead of re-testing `arm == "v6"` (which would silently
# mis-score a new pooled arm as owner-type: wrong hit set AND wrong bootstrap unit).
POOLED = ("v6", "ne", "concat")
NEC_DEFAULT = {"enron": "cmd_shared_cards__neutral_fixed.json",
               "mad": "cmd_shared_cards_mad__neutral_fixed.json",
               "cv": "cmd_shared_cards_cv__neutral_fixed.json"}
CONCATC_DEFAULT = {"enron": "cmd_concat_cards__neutral.json", "mad": "cmd_concat_cards_mad__neutral.json",
                   "cv": "cmd_concat_cards_cv__neutral.json"}
POOLC = {"v6": V6C,
         "ne": CG.SE / os.environ.get("NEC", NEC_DEFAULT[DS]),
         "concat": CG.SE / os.environ.get("CONCATC", CONCATC_DEFAULT[DS])}
ARM_TAG = {"v6": "v6", "indiv": "indiv", "staab": "stb", "petre_k4": "pet", "tpar_t15": "tpr",
           "ne": "ne", "concat": "cc"}
# Canonical scan/display order for every scorer -- one list so adding an arm never needs a sweep of
# hand-copied tuples (the A12 correction: five scorers each carried their own copy).
ALL_ARMS = ("indiv", "staab", "petre_k4", "tpar_t15", "v6", "ne", "concat")
# Per-arm bracket-seed override, e.g. BSEEDS_ARM="v6:1,ne:4,concat:4" -> the A12 package runs ne/concat at
# the full 4 seeds (equal-n with the shipped v6 measurement) while v6 contributes only the bseed-0 in-wave
# ANCHOR (prereg §5.3). Unset => every pooled arm uses BSEEDS_V6, i.e. legacy behaviour bit-for-bit.
BSEEDS_ARM = {p.split(":")[0].strip(): int(p.split(":")[1])
              for p in os.environ.get("BSEEDS_ARM", "").split(",") if p.strip()}
# R20 (#161, prereg results/NAMING_R20_FULLACCESS_PREREG.md): which text the attacker holds per
# candidate.  `held` (default) = the shipped leak-disjoint 250-char held-out sample.  The R20
# conditions hand it the CARD-BUILDING corpus instead -- the attacker our scenario section describes.
# ref text is consumed at EXACTLY ONE site (the rendered candidate list in main()); it never enters
# the salt, the bracket list, the partition, the slot shuffle, or the batch ordering.  Therefore the
# new conditions face BIT-IDENTICAL brackets => the old/new comparison is exactly paired per owner.
# Unset => RS.build() returns CG.load()'s `ref` object untouched => byte-identical legacy export.
REFSRC = os.environ.get("REFSRC", "held")


def _bseeds(arm):
    return BSEEDS_ARM.get(arm, BSEEDS_V6)
DSDIR = {"enron": "enron", "mad": "mad", "cv": "se"}[DS]
BASE = ROOT / os.environ.get("BATCHDIR", f"results/{DSDIR}/naming_v6")
if set(ARMS) != {"v6", "indiv"} and "BATCHDIR" not in os.environ:
    sys.exit("non-legacy ARMS requires an explicit BATCHDIR (refusing to touch the legacy naming_v6 dir)")
if (INDIV_SKIP or BSEED_SKIP or INDIV_PC != 2 or BSEEDS_V6 != 2) and "BATCHDIR" not in os.environ:
    sys.exit("a non-default owner/seed slice requires an explicit BATCHDIR "
             "(refusing to overwrite the legacy naming_v6 dir with a different bracket set)")
LETTERS = "ABCDEFGH"

SYS = ("You are a forensic analyst re-identifying who contributed to a distilled skill card. You may use ANY "
       "cue: reasoning and decision style, sequencing, escalation habits, risk posture, priorities, phrasing "
       "habits and idiolect, and topical fingerprints.")
USR_V6 = ("A SKILL CARD below was distilled from the combined working/decision style of a small group of "
          "people.\n\nSKILL CARD:\n{card}\n\n"
          "{n} writing samples follow, each by a DIFFERENT person. Some, one, or none of them may be among the "
          "people whose work was used to build the card.\n\n{cands}\n\n"
          "Which person is MOST LIKELY to be one of the card's contributors? You MUST pick exactly one. Answer "
          "with EXACTLY one line: the letter, a space, then a confidence 50-100 (50=pure guess, 100=certain). "
          "Example: `C 62`. Output ONLY that line.")
USR_INDIV = ("A SKILL CARD below was distilled from the working/decision style of ONE person.\n\n"
             "SKILL CARD:\n{card}\n\n"
             "{n} writing samples follow, each by a DIFFERENT person. One of them may be the person the card "
             "was distilled from — or none of them is.\n\n{cands}\n\n"
             "Which person is MOST LIKELY the one the card was distilled from? You MUST pick exactly one. Answer "
             "with EXACTLY one line: the letter, a space, then a confidence 50-100 (50=pure guess, 100=certain). "
             "Example: `C 62`. Output ONLY that line.")


def _h(*xs):
    return hashlib.sha1("|".join(map(str, xs)).encode()).hexdigest()


def _shuf(items, *salt):
    return sorted(items, key=lambda x: _h("shuf", *salt, x))


def _partition(items, gsize=8):
    """Contiguous slices of the (already hash-shuffled) list, sizes as even as possible, each <= gsize."""
    n = math.ceil(len(items) / gsize)
    base, extra = divmod(len(items), n)
    out, i = [], 0
    for g in range(n):
        sz = base + (1 if g < extra else 0)
        out.append(items[i:i + sz]); i += sz
    return out


def _salt_arm(arm):
    """Owner-type arms (indiv + per-person de-id) share the 'indiv' salt -> a given owner faces the identical
    candidate partition in every owner arm and across runs (per-owner pairing + the no-op bridge).
    POOLED arms (v6/ne/concat) likewise share the 'v6' salt -> for a given (cluster, bseed) all three face
    the SAME r1 partition, which is what makes the A12 per-cluster pairing exact (slots still re-shuffled
    per arm below, so paired groups never share a letter layout)."""
    return "v6" if arm in POOLED else "indiv"


def build_brackets():
    """Deterministic bracket list (index-stable across rounds) + candidate pool + ref texts."""
    _docs, authors, nuwa, aggro, ref, _raw = CG.load()
    _grp, byc = CG.make_groups(aggro, authors, KCL, SEED)
    full = {cid: mem for cid, mem in byc.items() if len(mem) >= KCL}
    pool = sorted(a for mem in full.values() for a in mem)
    pooled_cards = {a: json.loads(POOLC[a].read_text(encoding="utf-8")) for a in ARMS if a in POOLED}
    deid = json.loads(DEIDC.read_text(encoding="utf-8")) \
        if any(a not in POOLED and a != "indiv" for a in ARMS) else {}
    brackets, miss = [], {}
    for arm in ARMS:
        if arm in POOLED:
            mk = "card" if arm == "v6" else arm   # legacy _config.json miss key preserved for v6
            for cid in sorted(full):
                card = pooled_cards[arm].get(f"k{KCL}_s{SEED}_{cid}")
                if not card:
                    miss[mk] = miss.get(mk, 0) + 1; continue
                for b in range(BSEED_SKIP, _bseeds(arm)):
                    brackets.append({"arm": arm, "target": cid, "bseed": b, "card": card})
        else:
            src = nuwa if arm == "indiv" else deid.get(arm, {})
            for cid in sorted(full):
                for m in sorted(full[cid], key=lambda x: _h("ipick", SEED, cid, x))[INDIV_SKIP:INDIV_PC]:
                    card = src.get(m)
                    if not card:
                        miss[arm] = miss.get(arm, 0) + 1; continue
                    brackets.append({"arm": arm, "target": m, "bseed": 0, "card": card})
    if any(miss.get(a) for a in ARMS if a not in ("v6", "indiv")):
        # covers the de-id owner arms (owner sets would diverge from the indiv pairing) AND the A12 pooled
        # arms ne/concat (a missing cluster card = fewer clusters than v6 = unequal n = the R18 artefact).
        # v6 keeps its legacy non-raising behaviour (its miss key is "card").
        raise SystemExit(f"arm(s) missing cards {miss} — arm coverage would diverge; "
                         f"fix DEIDC/NEC/CONCATC/ARMS before exporting or scoring")
    return brackets, pool, ref, full, miss


def read_answers(rdir):
    meta = json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
    ans = {}
    for f in sorted(rdir.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", f.stem):        # stray-guard (ignore ans_backup etc.)
            continue
        for rec in json.loads(f.read_text(encoding="utf-8")):
            if isinstance(rec, dict) and "pid" in rec:
                m = re.search(r"[A-H]", str(rec.get("choice", "")).upper())
                if m:
                    ans[rec["pid"]] = m.group(0)
    return meta, ans


def winners_of(rdir):
    """bid -> [winning author per group]; skips missing/unparseable/out-of-range answers (reported)."""
    meta, ans = read_answers(rdir)
    win, missing = {}, 0
    for pid, mt in sorted(meta.items()):
        ch = ans.get(pid)
        if ch is None or LETTERS.index(ch) >= len(mt["cands"]):
            missing += 1; continue
        win.setdefault(mt["bid"], []).append(mt["cands"][LETTERS.index(ch)])
    return win, missing, len(meta)


def main():
    brackets, pool, ref, full, miss = build_brackets()
    if REFSRC != "held":
        _docs, _authors = CG.load()[0], sorted(pool)
        ref = RS.build(DS, _docs, _authors, ref, REFSRC)
        print(f"[R20] REFSRC={REFSRC} -- {RS.provenance(DS, REFSRC)}", flush=True)
    bidmap = {f"{ARM_TAG.get(b['arm'], b['arm'])}{i:03d}": b for i, b in enumerate(brackets)}
    rdir = BASE / f"round{ROUNDN}"

    if ROUNDN == 1:
        cand_lists = {bid: _shuf(pool, _salt_arm(b["arm"]), b["target"], b["bseed"], "r1")
                      for bid, b in bidmap.items()}
    else:
        prev = BASE / f"round{ROUNDN - 1}"
        pcfg = json.loads((prev / "_config.json").read_text(encoding="utf-8"))
        assert (pcfg["ds"], pcfg["kcl"], pcfg["seed"]) == (DS, KCL, SEED), f"config drift vs {prev}: {pcfg}"
        assert pcfg.get("arms", ["v6", "indiv"]) == ARMS, f"ARMS drift vs {prev}: {pcfg.get('arms')} != {ARMS}"
        assert pcfg.get("bseeds_arm", {}) == BSEEDS_ARM, \
            f"per-arm bseed drift vs {prev}: exported {pcfg.get('bseeds_arm', {})} != env {BSEEDS_ARM}"
        assert (pcfg.get("indiv_skip", 0), pcfg.get("bseed_skip", 0),
                pcfg.get("indiv_per_cluster"), pcfg.get("bseeds_v6")) == \
               (INDIV_SKIP, BSEED_SKIP, INDIV_PC, BSEEDS_V6), \
            (f"owner/seed slice drift vs {prev}: exported "
             f"skip={pcfg.get('indiv_skip', 0)}/{pcfg.get('bseed_skip', 0)} "
             f"pc={pcfg.get('indiv_per_cluster')} bs={pcfg.get('bseeds_v6')} != env "
             f"{INDIV_SKIP}/{BSEED_SKIP} pc={INDIV_PC} bs={BSEEDS_V6}")
        win, missing, ntot = winners_of(prev)
        print(f"round{ROUNDN - 1}: winners in {len(win)} brackets; {missing}/{ntot} answers missing/unusable")
        cand_lists = {bid: _shuf(w, _salt_arm(bidmap[bid]["arm"]), bidmap[bid]["target"],
                                 bidmap[bid]["bseed"], f"r{ROUNDN}")
                      for bid, w in win.items() if len(w) >= 2}

    meta, tasks = {}, []
    for bid in sorted(cand_lists):
        b = bidmap[bid]
        for g, grp_c in enumerate(_partition(cand_lists[bid], 8)):
            if b["arm"] not in ("v6", "indiv"):   # every arm that SHARES another arm's partition (de-id arms
                # share indiv's; ne/concat share v6's) re-shuffles its slots, so paired groups never present
                # the same letter layout and the attacker cannot recognise a bracket it already answered.
                grp_c = _shuf(grp_c, "slots", b["arm"], b["target"], ROUNDN, g)
            pid = f"r{ROUNDN}_{bid}_g{g:02d}"
            body = "\n\n".join(f"[{LETTERS[j]}] {ref[a]}" for j, a in enumerate(grp_c))
            tmpl = USR_V6 if b["arm"] in POOLED else USR_INDIV
            meta[pid] = {"bid": bid, "arm": b["arm"], "target": b["target"], "bseed": b["bseed"],
                         "round": ROUNDN, "gidx": g, "cands": grp_c}
            tasks.append({"pid": pid, "prompt": tmpl.format(card=b["card"], n=len(grp_c), cands=body)})

    if not tasks:
        print("nothing to export (all brackets already resolved to a single winner?)"); return
    rdir.mkdir(parents=True, exist_ok=True)
    order = sorted(tasks, key=lambda t: _h(t["pid"]))     # interleave brackets/arms across batches
    nbatch = math.ceil(len(order) / TPB)
    batches = [[] for _ in range(nbatch)]
    for j, t in enumerate(order):
        batches[j % nbatch].append(t)
    for i, bb in enumerate(batches):
        (rdir / f"batch_{i}.json").write_text(json.dumps(bb, ensure_ascii=False, indent=1), encoding="utf-8")
    (rdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (rdir / "sys.txt").write_text(SYS, encoding="utf-8")
    (rdir / "samples_only.txt").write_text(
        "\n\n========\n\n".join(f"{t['pid']}\n{t['prompt']}" for t in tasks[:3]), encoding="utf-8")
    cfg = {"ds": DS, "kcl": KCL, "seed": SEED, "round": ROUNDN, "arms": ARMS, "v6c": V6C.name,
           "deidc": DEIDC.name, "bseeds_v6": BSEEDS_V6,
           "indiv_per_cluster": INDIV_PC, "indiv_skip": INDIV_SKIP, "bseed_skip": BSEED_SKIP,
           "pool_n": len(pool), "clusters": len(full),
           "chance_v6": round(KCL / len(pool), 4), "chance_indiv": round(1 / len(pool), 4),
           "brackets": len(bidmap), "tasks": len(tasks), "miss": miss}
    # New keys are added ONLY when non-default, so a legacy export still writes a byte-identical _config.json.
    if BSEEDS_ARM:
        cfg["bseeds_arm"] = BSEEDS_ARM
    if REFSRC != "held":
        cfg["refsrc"] = REFSRC
        cfg["refsrc_provenance"] = RS.provenance(DS, REFSRC)
    if any(a in POOLED and a != "v6" for a in ARMS):
        cfg["poolc"] = {a: POOLC[a].name for a in ARMS if a in POOLED}
    (rdir / "_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    toks = sum(len(t["prompt"]) for t in tasks) // 4
    by = {}
    for m in meta.values():
        by[m["arm"]] = by.get(m["arm"], 0) + 1
    print(f"DS={DS} k{KCL} s{SEED} ROUND={ROUNDN}: pool={len(pool)} clusters={len(full)} "
          f"brackets={len(bidmap)} (miss={miss})")
    print(f"  tasks={len(tasks)} ({by})  ~{toks/1000:.0f}k prompt-tokens  chance v6={cfg['chance_v6']} "
          f"indiv={cfg['chance_indiv']}")
    print(f"  {nbatch} batches ({[len(x) for x in batches]}) -> {os.path.relpath(rdir, ROOT)}")


if __name__ == "__main__":
    main()
