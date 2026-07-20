"""20-MAD utility, on the SAME rebuilt instrument as CV/Enron: reference-free FORCED CHOICE + placebo battery.

Third-dataset replication of THE LAW -- "a distilled card carries DOMAIN-GENERAL judgement, not PERSON-SPECIFIC
judgement" -- to retire MAD's DEAD objective instrument (5-class bug-resolution prediction: majority baseline
0.461 beats every arm; card dynamic range indiv-stranger +0.017 < noise 0.026, so it cannot arbitrate any
operator). Forced choice has no class prior to collapse onto. See .claude/skills/judge-card-utility/SKILL.md;
this is a straight port of scripts/enron_fc_export.py (which ported cv_fc_export.py). Reviewed by an Opus
adversary before spending (scratchpad/mad_fc_design.md): leakage cleared (card source is issue-id-DISJOINT from
the test bugs by construction, util6_pool.py:96; and FC is reference-free so there is no answer to leak).

STRUCTURAL DIFFERENCE from Enron (Plan A): the task is NOT a shared generic prompt. Unit = (dev e, bug i), where
the task text is dev e's OWN held-out solved bug (stub+report). So `in` triages its own project's bug (a mild
home-field edge, CV-like) and `in ~= st` is a slightly weaker tie test than Enron's -- report the direction
honestly; a small `in > st` here is genuine mild person/component-specificity, NOT leakage.

Because tasks are per-dev, NO two units share (cluster-card, bug): the Enron pooled-arm DEDUP yields zero
savings and would COLLIDE if kept (two devs in a cluster both keying bug#0 as (ne, cluster, 0)). So every arm
drafts once per (arm, e, i); `dkey` is gone. `POOLED_ARMS` survives for its SECOND, unrelated role only: the
scorer reads config["pooled_arms"] to bootstrap ne/cc by the 16 pooling CLUSTERS (a cluster-shared card is not
16 independent draws), while in/st/staab/sham bootstrap by the 128 DEVS.

ARMS: in / st / sham (a wrong-domain CV statistics card) / staab / ne (pooled CMD, neutral_fixed) / cc (concat).
CONTRASTS: ne-staab (CMD beats de-id, SIG>.5) ; ne-cc (CMD vs concat; 16 clusters = best power of the 3
datasets) ; staab-in (de-id cost, per-dataset) ; in-st (own vs stranger, TIE) ; in-sham (domain skill, SIG>.5).

Run:  DATASET=mad python -P scripts/mad_fc_export.py            [COST=1 to price the deepseek drafting first]
      DATASET=mad NEXPERT=0 python -P scripts/mad_fc_export.py  [full 128-dev cohort]
      -> results/mad/fc/{meta.json, config.json, sys.txt, draft_len.json, batch_i.json}
      then free `sonnet` subagents write ans_i.json (SKILL.md 7), then: BATCHDIR=results/mad/fc python -P scripts/cv_fc_score.py
"""
import os
import re
import sys
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ["DATASET"] = "mad"                 # utility arms live in data/20mad/
os.environ.setdefault("GROUP", "random")      # data-independent partition; MUST match the shipped pooled cards
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import deid_enron as de          # noqa: E402  (de.pool parallel map)
import cmd_gate as CG            # noqa: E402  (load, make_groups)
from src.llm import chat         # noqa: E402
from skill_prompts import load_prompts, fill    # noqa: E402

K = int(os.environ.get("K", "8"))                    # R13 (#139): k-gradient packs override; default = canonical 8.
SEED = int(os.environ.get("SEED", "1"))              # MAD canonical partition = k8_s1 (matches the ANONYMITY
#            battery AND the only k8 seed where BOTH neutral_fixed and concat cards exist -> both axes, one seed.
GEN = os.environ.get("PREDICTOR", "deepseek-chat")
MAXBUGS = int(os.environ.get("MAXBUGS", "8"))        # held-out bugs per dev (first-N; the card source is disjoint)
NEXPERT = int(os.environ.get("NEXPERT", "30"))       # 0 = full 128-dev cohort
DRAFT_TOK = int(os.environ.get("DRAFT_TOK", "700"))  # triage assessments are short; headroom + untruncation guard
TEMP = float(os.environ.get("DRAFT_TEMP", "0.0"))    # fixed => cached => reproducible re-exports  (NOT "TEMP":
#                     that is a standard Windows env var, the temp directory, and would collide.)
NBATCH = int(os.environ.get("NBATCH", "0"))          # 0 = auto (<= MAXITEM items/batch)
MAXITEM = int(os.environ.get("MAXITEM", "30"))
NPLA = int(os.environ.get("NPLA", "32"))             # units carrying self/pad/fmt
NCUT = int(os.environ.get("NCUT", "16"))             # units carrying the cut@p range probe
COST = os.environ.get("COST", "") not in ("", "0")
OUT = ROOT / os.environ.get("BATCHDIR", "results/mad/fc")

_DEF_CONTRASTS = [("ne", "staab"), ("ne", "cc"), ("staab", "in"), ("in", "st"), ("in", "sham")]
# CONTRASTS env-overridable: "ne-nec,nec-staab" -> the clean-rebuild de-risk arms. Default = canonical 5.
CONTRASTS = ([tuple(c.split("-")) for c in os.environ["CONTRASTS"].split(",")]
             if os.environ.get("CONTRASTS") else _DEF_CONTRASTS)
# nec = clean-rebuilt CMD (un-truncated); a POOLED card like ne/cc -> cluster-bootstrap.
POOLED_ARMS = tuple(a for a in ("ne", "cc", "nec") if any(a in c for c in CONTRASTS))  # bootstrap-clustering role
#              ONLY (config["pooled_arms"]); NOT a draft-dedup key -- on MAD every draft is per-(arm, dev, bug).

CUTS = (0.10, 0.25, 0.50)
# Content-free padding: vacuous truisms that carry NO triage-actionable value (the pad probe measures a length
# bias, so the filler must add nothing a judge could legitimately credit). An earlier version mentioned
# reproduction steps / duplicate issues / severity -- those ARE triage actions the rubric rewards, so the judge
# rightly preferred the padded copy and the length gate failed on the FILLER, not on a real bias. These are
# deliberately generic and non-actionable.
FILLER = ["There is, of course, more that could be said about each of these points.",
          "In the end, software tends to behave the way it was written to behave.",
          "Every project carries a long history that shapes how things end up.",
          "As with most things in engineering, one view rarely captures everything.",
          "This could be looked at from several different angles, as such things often can.",
          "Reasonable people will, as always, weigh the various considerations differently.",
          "It is worth remembering that context is a large part of any picture.",
          "Ultimately the situation is what it is, and matters will unfold from there."]
MD = re.compile(r"(\*\*|`|^#{1,6}\s*|^\s*[-+]\s+(?=\S))", re.M)
ENDS = re.compile(r"[.!?][\s*_`\"'’”)\]]*$")
NUM = re.compile(r"(?:(?<=\s)|^)\d+\.(?=\s)")


def strip_md(t):
    return re.sub(r"[ \t]{2,}", " ", MD.sub("", t)).strip()


def denum(t):
    return re.sub(r"\s{2,}", " ", NUM.sub("", t)).strip()


def sentences(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 3]


def cut(t, p, seed):
    ss = sentences(t)
    n = len(ss)
    k = max(1, int(round(p * n)))
    if n - k < 2:
        return None
    rng = random.Random(seed)
    drop = set(rng.sample(range(n), k))
    return " ".join(s for i, s in enumerate(ss) if i not in drop)


def pad(t, target_chars):
    out, i = t, 0
    while len(out) < target_chars and i < len(FILLER):
        out += " " + FILLER[i]
        i += 1
    return out.strip()


def draft_msgs(card, task):
    """The MAD triage drafter. The card is a distilled triage/decision profile; the output is a short competence
    assessment of the bug, NOT a bare resolution class (a one-word FIXED/WONTFIX would let the judge pattern-match
    a label rather than judge competence). Reference-free: the real `resolution` is never shown."""
    if card:
        return [{"role": "system", "content": "You are a software developer triaging a bug report. Use the "
                 "triage/decision profile below to decide how to handle it. Output only your triage assessment."},
                {"role": "user", "content": f"Your profile:\n{card}\n\nBug report:\n{task}\n\nWrite ONLY your "
                 "triage assessment (2-4 sentences): what you judge will most likely happen to this bug and why, "
                 "and the single recommended next step. Do not output a bare status word."}]
    return [{"role": "system", "content": "You are a competent software developer triaging a bug report. Output "
             "only your triage assessment."},
            {"role": "user", "content": f"Bug report:\n{task}\n\nWrite ONLY your triage assessment (2-4 "
             "sentences): what you judge will most likely happen to this bug and why, and the single recommended "
             "next step. Do not output a bare status word."}]


def truncated(text, cap):
    """Long, and not ending in a finished sentence -- a runaway. A short assessment that finishes cleanly passes."""
    return len(ENC.encode(text)) >= 0.80 * cap and not ENDS.search(text)


def draft_untruncated(card, task):
    """draft; resample at a higher cap if the model runs off the end. max_tokens is part of the llm cache key, so
    each retry is a genuine new sample AND stays cached (idempotent re-exports). Returns (draft, n_resamples) or
    (None, 3)."""
    msgs = draft_msgs(card, task)
    d = (chat(msgs, model=GEN, temperature=TEMP, max_tokens=DRAFT_TOK) or "").strip()
    if d and not truncated(d, DRAFT_TOK):
        return d, 0
    for i, cap in enumerate((DRAFT_TOK + 500, DRAFT_TOK + 1000), start=1):
        d = (chat(msgs, model=GEN, temperature=TEMP, max_tokens=cap) or "").strip()
        if d and not truncated(d, cap):
            return d, i
    return None, 3


def main():
    P = load_prompts()
    SYS, USR = P["forced_choice_triage.system"], P["forced_choice_triage.user"]

    # ---- cohort + grouping (must match the shipped pooled cards) ----
    assert CG.GROUP == "random", (f"GROUP={CG.GROUP!r} -- the shipped neutral_fixed/concat cards were built on "
                                  f"the random (data-independent) partition; any other grouping attaches the "
                                  f"wrong cluster's card to every dev")
    pool, authors, nuwa, aggro, _ref, _raw = CG.load()
    grp, byc = CG.make_groups(aggro, authors, K, SEED)
    D20 = ROOT / "data" / "20mad"
    neutral = json.loads((D20 / "cmd_shared_cards_mad__neutral_fixed.json").read_text(encoding="utf-8"))
    concat = json.loads((D20 / "cmd_concat_cards_mad.json").read_text(encoding="utf-8"))
    step2 = json.loads((D20 / "mad_cmd_step2.json").read_text(encoding="utf-8"))
    # R13 M2: existence asserts are conditional on the arms the pack ACTUALLY uses (the concat file has no
    # k2/k6/k10/k12 keys and neutral_fixed gains k10/k12 only after the R13 build -- an unconditional assert
    # would kill a nec-in gradient pack that never touches ne/cc/staab).
    _use = {a for c in CONTRASTS for a in c}
    _use_nec = "nec" in _use
    neclean = json.loads((D20 / os.environ.get("NEUTRALCLEAN", "cmd_shared_cards_mad__neutral_cleanpilot.json")
                          ).read_text(encoding="utf-8")) if _use_nec else {}
    ck = {a: f"k{K}_s{SEED}_{grp[a]}" for a in authors}
    for a in authors:
        assert "ne" not in _use or ck[a] in neutral, \
            f"pooled card {ck[a]} missing from neutral_fixed -- grouping != shipped cards"
        assert "cc" not in _use or ck[a] in concat, f"concat card {ck[a]} missing for {a}"
        assert not _use_nec or ck[a] in neclean, f"clean card {ck[a]} missing from NEUTRALCLEAN -- build it first"
    miss = [a for a in authors if a not in step2["staab"]]
    assert "staab" not in _use or not miss, f"staab de-id card missing for {len(miss)} devs, e.g. {miss[:3]}"

    # Membership FREEZE (ported from Enron: key existence is not enough -- all 16 k8_s1 keys exist for ANY
    # 128-dev partition into 16 labelled groups, so a re-ordered `authors` would silently attach the wrong
    # cluster's card to every member while the existence asserts stay green). group_random is data-independent
    # and reproducible from the (unchanged) pool, so drift risk is low; freeze on first run and assert after.
    manifest = D20 / f"mad_groups_k{K}_s{SEED}.json"
    members = {cid: sorted(ms) for cid, ms in byc.items()}
    if manifest.exists():
        rec = json.loads(manifest.read_text(encoding="utf-8"))
        assert rec == members, (f"grouping drifted from {manifest.name} -- the shipped neutral_fixed/concat cards "
                                f"were built for a DIFFERENT partition; every ne/cc/staab draft would use the "
                                f"wrong cluster's card")
    else:
        manifest.write_text(json.dumps(members, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  froze cluster membership -> {manifest.relative_to(ROOT)}")
    print(f"grouping: {len(byc)} clusters of k={K}; used-arm cards present for all {len(authors)} devs "
          f"(arms checked: {sorted(_use)})")

    # ---- held-out test bugs: first MAXBUGS solved bugs per dev (disjoint from the card source) ----
    bugs = {a: pool[a].get("solved_bugs", [])[:MAXBUGS] for a in authors}
    short = [a for a in authors if len(bugs[a]) < MAXBUGS]
    assert not short, f"{len(short)} devs have < {MAXBUGS} solved bugs, e.g. {short[:3]} -- uneven unit blocks"

    def task_text(e, i):
        b = bugs[e][i]
        return f"{str(b.get('stub', '')).strip()}\n\n{str(b.get('report', '')).strip()}".strip()

    # ---- expert subsample: round-robin across clusters so every cluster is represented for the ne/cc bootstrap
    by_cluster = defaultdict(list)
    for a in authors:
        by_cluster[grp[a]].append(a)
    for c in by_cluster:
        by_cluster[c].sort(key=lambda a: hashlib.sha1(a.encode()).hexdigest())
    if NEXPERT and NEXPERT < len(authors):
        experts, idx = [], 0
        while len(experts) < NEXPERT:
            progressed = False
            for c in sorted(by_cluster):
                if idx < len(by_cluster[c]):
                    experts.append(by_cluster[c][idx]); progressed = True
                    if len(experts) >= NEXPERT:
                        break
            idx += 1
            if not progressed:
                break
        experts = sorted(experts)
    else:
        experts = sorted(authors)
    ncl = len(set(grp[e] for e in experts))
    assert not NEXPERT or NEXPERT >= len(byc), (f"NEXPERT={NEXPERT} < {len(byc)} clusters -- some pooling "
                                                f"clusters would be absent from the ne/cc bootstrap")
    print(f"experts: {len(experts)}/{len(authors)} over {ncl} clusters  (NEXPERT={NEXPERT or 'all'})")

    # ---- stranger (random outsider, full dev pool) + sham (a CV statistics card in the triage slot) ----
    rng = np.random.default_rng(SEED)
    stranger = {}
    for a in authors:
        popn = [b for b in authors if grp[b] != grp[a]]
        stranger[a] = popn[int(rng.integers(len(popn)))]
    cv = json.loads((ROOT / "data/se/cv_cmd_nuwa.json").read_text(encoding="utf-8"))["nuwa"]  # wrong-domain source
    cv_keys = sorted(cv)
    sham = {a: cv[cv_keys[int(hashlib.sha1(f"sham-{a}".encode()).hexdigest(), 16) % len(cv_keys)]]
            for a in authors}

    CARD = {"in": lambda e: nuwa[e], "st": lambda e: nuwa[stranger[e]], "sham": lambda e: sham[e],
            "staab": lambda e: step2["staab"][e], "ne": lambda e: neutral[ck[e]], "cc": lambda e: concat[ck[e]],
            # R3' (#127): full per-person de-id battery vs v6 -- same author-indexed pattern as staab.
            "petre_k4": lambda e: step2["petre_k4"][e], "tpar_t15": lambda e: step2["tpar_t15"][e],
            "nec": lambda e: neclean[ck[e]]}
    arms = sorted({a for c in CONTRASTS for a in c}, key=list(CARD).index)

    if COST:
        # one draft per DISTINCT (arm, dev, bug) -- no dedup on MAD (per-dev tasks)
        keys = [(arm, e, i) for arm in arms for e in experts for i in range(len(bugs[e]))]
        tin = sum(len(ENC.encode((CARD[arm](e) or "") + task_text(e, i))) for (arm, e, i) in keys)
        print(f"\nCOST: {len(keys)} drafts | input ~{tin:,} tok | output <= {len(keys)*DRAFT_TOK:,} tok "
              f"({GEN}); judges are free subagents")
        return

    # ---- drafts: one per DISTINCT (arm, dev, bug); the pooled card is SELECTED by arm but the draft key is
    # always per-(arm, e, i) -- two devs in a cluster never collapse (their bugs differ). ----
    jobs = [(arm, e, i, CARD[arm](e)) for arm in arms for e in experts for i in range(len(bugs[e]))]
    print(f"\ndrafting {len(jobs)} distinct (arm,dev,bug) drafts over {len(experts)} devs x {MAXBUGS} bugs "
          f"x {len(arms)} arms ({arms}) ...", flush=True)
    Dd, resampled = {}, defaultdict(int)
    for (arm, e, i, _card), (txt, nr) in zip(jobs, de.pool(lambda j: draft_untruncated(j[3], task_text(j[1], j[2])),
                                                           jobs)):
        assert txt, (f"arm {arm} dev {e} bug {i}: draft still truncated after 3 resamples -- this bug makes the "
                     f"model run away; raise DRAFT_TOK or drop the bug")
        Dd[(arm, e, i)] = txt
        if nr:
            resampled[arm] += 1
    for arm in arms:
        L = np.array([len(Dd[(arm, e, i)]) for e in experts for i in range(len(bugs[e]))])
        print(f"  {arm:6s} mean {L.mean():6.0f} chars  median {np.median(L):6.0f}  max {L.max():5d}  "
              f"resampled {resampled[arm]}", flush=True)

    units = [(e, i) for e in experts for i in range(len(bugs[e]))]

    def draft_of(arm, u):
        e, i = units[u]
        return Dd[(arm, e, i)]

    meta, items = {}, []

    def add(pid, kind, u, A, B, **kw):
        e, i = units[u]
        meta[pid] = {"kind": kind, "unit": u, "u": e, **kw}       # "u"=dev -> per-person bootstrap key in scorer
        items.append({"pid": pid, "prompt": fill(USR, q=task_text(e, i), a=A, b=B, pid=pid)})

    # real contrasts, both orders. No dedup on MAD: every unit is a distinct dev-bug, so every ne-cc pair is
    # unique (unlike Enron's shared tasks). Both orders of a pair go to different batches (below).
    for ci, (x, y) in enumerate(CONTRASTS):
        for u in range(len(units)):
            for o in (0, 1):
                A, B = (draft_of(x, u), draft_of(y, u)) if o == 0 else (draft_of(y, u), draft_of(x, u))
                add(f"C{ci}{u:04d}{o}", "contrast", u, A, B, x=x, y=y, order=o)

    # Placebo/cut carriers: ONE unit per expert, spread (NOT a contiguous prefix -- the unit list is dev-major,
    # MAXBUGS units per dev, so range(NPLA) would land the battery on the first 2-4 devs, and the scorer clusters
    # the battery by dev, so its resolves_finest gate could FAIL over 2 clusters and void every contrast AFTER
    # the paid drafting). carriers[j] = unit index of (experts[j], bug j % MAXBUGS): rotate the bug so the
    # battery is not single-bug. Every dev has exactly MAXBUGS units, so the blocks are uniform.
    carriers = [j * MAXBUGS + (j % MAXBUGS) for j in range(len(experts))]
    # PROBE_ARM: which arm's drafts carry the self/pad/fmt/cut probes. Default "in" (canonical B1). A pack whose
    # CONTRASTS excludes `in` (e.g. the elemk ne-nec gate) sets PROBE_ARM to a drafted arm -- the battery
    # validates the JUDGE on draft-vs-modified-self, so the carrier arm is immaterial to what it measures.
    PROBE_ARM = os.environ.get("PROBE_ARM", "in")
    assert PROBE_ARM in arms, f"PROBE_ARM={PROBE_ARM!r} not among drafted arms {arms}"
    for u in carriers[:NPLA]:
        d = draft_of(PROBE_ARM, u)
        add(f"S{u:04d}0", "self", u, d, d)                                        # position
        tgt = int(len(d) * 1.25)
        for o in (0, 1):
            A, B = (pad(d, tgt), d) if o == 0 else (d, pad(d, tgt))               # length: A longer iff o==0
            add(f"P{u:04d}{o}", "pad", u, A, B, order=o)
            A, B = (d, strip_md(d)) if o == 0 else (strip_md(d), d)               # format: A rich iff o==0
            add(f"F{u:04d}{o}", "fmt", u, A, B, order=o)
    for u in carriers[:NCUT]:                                                      # range probe
        d = denum(draft_of(PROBE_ARM, u))
        for j, p in enumerate(CUTS):
            c = cut(d, p, seed=u * 100 + j)
            if c is None:
                continue
            for o in (0, 1):
                A, B = (d, c) if o == 0 else (c, d)                               # A is the full draft iff o==0
                add(f"X{j}{u:04d}{o}", "cut", u, A, B, p=p, order=o)

    # BATCHING (same two rules as cv/enron): scatter by pair so a judge never sees one unit's items massed (it
    # would spot the near-identical placebos), and route the two ORDERS of a pair to different batches so no
    # single judge can flip a pair mechanically. Both orders are still measured, so position bias cancels.
    pairs = defaultdict(list)
    for t in items:
        m = meta[t["pid"]]
        pairs[(m["unit"], m["kind"], m.get("x"), m.get("y"), m.get("p"))].append(t)
    nb = NBATCH or max(2, -(-len(items) // MAXITEM))
    assert nb >= 2, "need >= 2 batches to separate the two orders of a pair"
    assert (nb // 2) % nb != 0, "offset collapses; the two orders would land in the same batch"
    batches = [[] for _ in range(nb)]
    for key, ts in pairs.items():
        base = int(hashlib.sha1("|".join(map(str, key)).encode()).hexdigest()[:8], 16) % nb
        for t in ts:
            o = meta[t["pid"]].get("order", 0)
            batches[(base + (nb // 2 if o == 1 else 0)) % nb].append(t)
    for bi, b in enumerate(batches):
        random.Random(SEED * 1000 + bi).shuffle(b)

    OUT.mkdir(parents=True, exist_ok=True)
    # A pid is (contrast, unit, order) and does NOT depend on the draft text, so a re-export that changes a
    # prompt (a resampled draft) would leave any `ans` already judged against the OLD prompt looking done to
    # fc_status (pid-coverage only). Detect changed batches and stale their answers out.
    changed = []
    ANSF = re.compile(r"ans(?:_\w+?)?_(\d+)$")
    for i, b in enumerate(batches):
        txt = json.dumps(b, ensure_ascii=False, indent=1)
        p = OUT / f"batch_{i}.json"
        if p.exists() and p.read_text(encoding="utf-8") != txt:
            changed.append(i)
        p.write_text(txt, encoding="utf-8")
    if changed:
        n = 0
        for f in OUT.glob("ans*.json"):
            m = ANSF.fullmatch(f.stem)
            if m and int(m.group(1)) in changed:
                f.rename(f.with_suffix(".json.stale")); n += 1
        print(f"\n  ⚠ {len(changed)} batches changed on re-export -> staled {n} answer file(s).", flush=True)

    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "sys.txt").write_text(SYS, encoding="utf-8")
    (OUT / "draft_len.json").write_text(json.dumps(
        {a: int(np.mean([len(draft_of(a, u)) for u in range(len(units))])) for a in arms}, indent=1),
        encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps(
        {"dataset": "mad", "nexpert": NEXPERT, "subsampled": len(experts) < len(authors),
         "n_units": len(units), "n_experts": len(experts), "maxbugs": MAXBUGS, "contrasts": CONTRASTS,
         "arms": arms, "cuts": list(CUTS), "k": K, "seed": SEED, "pooled_arms": list(POOLED_ARMS),
         "draft_tok": DRAFT_TOK, "temp": TEMP, "resampled_runaways": dict(resampled),
         "unit_expert": [e for e, i in units], "unit_cluster": [grp[e] for e, i in units],
         "units": [f"{e}|bug{i}" for e, i in units]}, indent=1), encoding="utf-8")

    kinds = defaultdict(int)
    for m in meta.values():
        kinds[m["kind"]] += 1
    print(f"\n{len(items)} items {dict(kinds)} -> {nb} batches ({[len(b) for b in batches]}) "
          f"-> {OUT.relative_to(ROOT)}")
    print(f"  {len(units)} units / {len(experts)} devs / {ncl} clusters")


if __name__ == "__main__":
    main()
