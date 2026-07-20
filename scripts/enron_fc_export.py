"""Enron utility, on the SAME rebuilt instrument as CV: reference-free FORCED CHOICE + the placebo battery.

The point is to replicate THE LAW -- "a distilled card carries DOMAIN-GENERAL judgement, not PERSON-SPECIFIC
judgement" -- on a SECOND dataset. If Enron reproduces the four decisive CV contrasts, the law is
cross-dataset, not a CV artifact. See .claude/skills/judge-card-utility/SKILL.md and scripts/cv_fc_export.py
(this is a straight port of that file).

Two things are DIFFERENT from CV, and both simplify:
  1. REFERENCE-FREE. The 8 Enron tasks are generic work-email decisions (GRANT/DENY-style), identical for every
     author, and there is no expert "reference answer". So CV's gate zero (the reference must be written by an
     outsider) simply does not apply -- there is nothing for a cohort member to have authored. The judge picks
     the more competent reply directly. Prompt = `forced_choice_noref.*` (no {ref} field).
  2. Generic tasks => `in` (the author's own card) has NO home-field edge on the task, because the task is not
     the author's own question. That makes `in ~= st` a PURER tie test and `in > sham` a purer domain-skill
     test than on CV.

STREAMLINED to the four decisive contrasts + the CMD-vs-concat parity, dropping the underpowered `ne-in` /
`ne-{tpar,petre}` (CV showed those sit on a between-cluster variance floor that more judges cannot move) and
the confounded `in-no`:
    ne - staab   CLAIM 4  pooled CMD beats per-person de-id            SIG > .5 expected
    ne - cc      CLAIM 3  CMD vs the naive concat baseline (parity)    TIE expected
    staab - in            does per-person de-id cost utility?          SIG < .5 expected
    in - st               own card vs a stranger's                     TIE expected  (the only tie claim)
    in - sham             right domain vs a wrong-domain (20-MAD) card SIG > .5 expected

ARMS: in / st / sham / staab / ne (pooled CMD, k8_s1, neutral_fixed) / cc (concat, k8_s0). Every per-person arm
is indexed by the author `e`. The pooled arms ne/cc are SHARED by every member of a cluster, so their draft is
keyed by (cluster, task) -- drafted ONCE and reused byte-identical, since deepseek at t=0 is not deterministic
and re-drafting per member would fabricate phantom-distinct `ne` texts and inflate its variance.

Unit = (author, task). NEXPERT authors are sampled ROUND-ROBIN across pooling clusters (even cluster coverage
for the ne/cc bootstrap); NEXPERT=0 uses the full cohort. The ne-cc contrast (both arms per-cluster) is
deduped to one item per (cluster, task) -- otherwise the same A/B pair is judged once per member for nothing.

Run:  DATASET=enron python -P scripts/enron_fc_export.py            [COST=1 to price the deepseek drafting first]
      NEXPERT=0 python -P scripts/enron_fc_export.py                [full 116-author cohort]
      -> results/enron/fc/{meta.json, config.json, sys.txt, draft_len.json, batch_i.json}
      then free `sonnet` subagents write ans_i.json (SKILL.md §7), then: BATCHDIR=results/enron/fc python -P scripts/cv_fc_score.py
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
os.environ.setdefault("DATASET", "enron")
os.environ.setdefault("GROUP", "random")     # data-independent partition; MUST match the shipped pooled cards
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import deid_enron as de          # noqa: E402  (TASKS, pool)
import enron_nuwa as NW          # noqa: E402  (GEN, the Enron draft prompt)
import cmd_gate as CG            # noqa: E402  (load, make_groups)
from src.llm import chat         # noqa: E402
from skill_prompts import load_prompts, fill    # noqa: E402

K = 8
SEED = int(os.environ.get("SEED", "1"))              # Enron canonical partition = k8_s1 (matches the anonymity
#            battery AND the only seed where BOTH neutral_fixed and concat cards exist at k8). CV used s0.
GEN = NW.GEN                                          # deepseek-chat
NEXPERT = int(os.environ.get("NEXPERT", "30"))       # 0 = full cohort
DRAFT_TOK = int(os.environ.get("DRAFT_TOK", "700"))  # email replies are short; headroom + untruncation guard
TEMP = float(os.environ.get("DRAFT_TEMP", "0.0"))    # fixed => cached => reproducible re-exports
#                     ^ NOT "TEMP": that is a standard Windows env var (the temp directory) and collides.
NBATCH = int(os.environ.get("NBATCH", "0"))          # 0 = auto (<= MAXITEM items/batch)
MAXITEM = int(os.environ.get("MAXITEM", "30"))
NPLA = int(os.environ.get("NPLA", "32"))             # units carrying self/pad/fmt
NCUT = int(os.environ.get("NCUT", "16"))             # units carrying the cut@p range probe
COST = os.environ.get("COST", "") not in ("", "0")
OUT = ROOT / os.environ.get("BATCHDIR", "results/enron/fc")

CONTRASTS = [("ne", "staab"), ("ne", "cc"), ("staab", "in"), ("in", "st"), ("in", "sham")]
# CONTRASTS env-overridable ("ne-nec" -> the V4/elemk vs-black-box packs; ported from mad_fc_export).
# nec = an injected pooled-card arm loaded from NEUTRALCLEAN (same key scheme as ne).
if os.environ.get("CONTRASTS"):
    CONTRASTS = [tuple(c.split("-")) for c in os.environ["CONTRASTS"].split(",")]
_USE_NEC = any("nec" in c for c in CONTRASTS)
PERCLUSTER = {"ne", "cc"} | ({"nec"} if _USE_NEC else set())   # cards shared by every member of a cluster
#   nec in PERCLUSTER => drafted once per (cluster, task) AND the ne-nec contrast dedupes to one item per
#   (cluster, task), exactly like ne-cc.

CUTS = (0.10, 0.25, 0.50)
FILLER = ["Note that the right call depends on the specifics of the situation.",
          "It is worth weighing the context in which this request arose.",
          "As always, the details of the deal and the relationship matter here.",
          "Different desks may reasonably emphasise different aspects of this.",
          "The best decision will depend on what outcome you are protecting.",
          "None of this removes the need to use your own judgement on the case.",
          "There is, of course, more that could be said about each of these points.",
          "In practice one balances these considerations against the goal at hand."]
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
    """The Enron drafter's prompt, verbatim from enron_nuwa.draft -- so these drafts are the project's Enron
    drafts, not a new dialect. Only temperature and max_tokens are lifted out (for reproducible caching and the
    untruncation guard). Every arm here is card-bearing, so the no-card branch is only a safety net."""
    if card:
        return [{"role": "system", "content": "You are an Enron employee. Use the working/decision profile below "
                 "to respond. Output only the email reply."},
                {"role": "user", "content": f"Your profile:\n{card}\n\nSituation:\n{task}\n\nWrite ONLY your email "
                 "reply, following any format the situation specifies."}]
    return [{"role": "system", "content": "You are an Enron employee handling work email. Respond to the situation "
             "the way a competent employee would. Output only the email reply."},
            {"role": "user", "content": f"Situation:\n{task}\n\nWrite ONLY your email reply, following any format "
             "the situation specifies."}]


def truncated(text, cap):
    """Long, and not ending in a finished sentence. A short reply that signs off with 'Best,' is under the
    length bar and passes -- only a long AND unfinished draft is a runaway."""
    return len(ENC.encode(text)) >= 0.80 * cap and not ENDS.search(text)


def draft_untruncated(card, task):
    """draft the reply; resample at a higher cap if the model runs off the end instead of finishing.

    max_tokens is part of the llm cache key, so each retry is a genuine new sample AND stays cached -- the run
    is idempotent and re-exports do not invalidate drafts already on disk. Returns (draft, n_resamples), or
    (None, 3) if it never finished a sentence."""
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
    SYS, USR = P["forced_choice_noref.system"], P["forced_choice_noref.user"]
    T = de.TASKS

    # ---- cohort + grouping (must match the shipped pooled cards) ----
    assert CG.GROUP == "random", (f"GROUP={CG.GROUP!r} -- the shipped neutral_fixed/concat cards were built on "
                                  f"the random (data-independent) partition; any other grouping attaches the "
                                  f"wrong cluster's card to every author")
    docs, authors, nuwa, aggro, ref, raw = CG.load()
    grp, byc = CG.make_groups(aggro, authors, K, SEED)
    neutral = json.loads((ROOT / "data/enron/cmd_shared_cards__neutral_fixed.json").read_text(encoding="utf-8"))
    concat = json.loads((ROOT / "data/enron/cmd_concat_cards.json").read_text(encoding="utf-8"))
    step2 = json.loads((ROOT / "data/enron/step2_cards_full.json").read_text(encoding="utf-8"))
    neclean = (json.loads((ROOT / "data/enron" / os.environ["NEUTRALCLEAN"]).read_text(encoding="utf-8"))
               if _USE_NEC else {})
    ck = {a: f"k{K}_s{SEED}_{grp[a]}" for a in authors}
    # R2: card-existence asserts scoped to the arms this pack actually drafts (concat's k8 cards exist only
    # for s1; a ne-nec pack at SEED=0/2 must not trip over them).
    _used = {x for c in CONTRASTS for x in c}
    for a in authors:
        assert "ne" not in _used or ck[a] in neutral, \
            f"pooled card {ck[a]} missing from neutral_fixed -- grouping != shipped cards"
        assert "cc" not in _used or ck[a] in concat, f"concat card {ck[a]} missing for {a}"
        assert not _USE_NEC or ck[a] in neclean, f"nec card {ck[a]} missing from NEUTRALCLEAN -- build it first"
    if "staab" in _used:
        miss = [a for a in authors if a not in step2["staab"]]
        assert not miss, f"staab de-id card missing for {len(miss)} authors, e.g. {miss[:3]}"
    # Key existence is NOT enough (ported from cv_fc_export): all 14 k8_s1 keys exist for ANY 116-author
    # partition into 14 labelled groups, so a re-ordered `authors` (a regenerated collection, a stray
    # GROUP env, a library bump) would silently attach some other cluster's pooled/concat/staab card to every
    # member while the existence asserts stay green. Freeze the exact MEMBERSHIP and assert against it. The
    # happy path was verified to match the card-build call chain (CG.load + make_groups, deterministic) at
    # freeze time; drift after that trips this assert instead of shipping a confidently-wrong headline.
    # R2 (mirrors R11 MAJOR-3): for a non-canonical seed the freeze below is a no-op on first run, so the
    # grouping must FIRST be cross-checked against the 2AFC audit trail of that seed's shipped cards.
    # Hard-fail on any diff: a drifted partition would attach the wrong cluster's ne/nec card everywhere.
    if SEED != 1:
        xc = ROOT / "results" / "enron" / f"2afc_v6min_s{SEED}" / "meta.json"
        assert xc.exists(), f"no 2AFC audit trail {xc} to cross-check the s{SEED} grouping against"
        seen = defaultdict(set)
        for mt in json.loads(xc.read_text(encoding="utf-8")).values():
            seen[mt["card_id"]].add(mt["member"])
        for cid, ms in sorted(seen.items()):
            extra = ms - set(byc.get(cid, []))
            assert not extra, (f"s{SEED} grouping mismatch on {cid}: 2AFC members {sorted(extra)} not in "
                               f"this export's cluster -- ne/nec drafts would use the wrong card")
        print(f"  grouping cross-checked against {xc.relative_to(ROOT)}: "
              f"{sum(len(v) for v in seen.values())} member slots consistent")
    manifest = ROOT / "data" / "enron" / f"enron_groups_k{K}_s{SEED}.json"
    members = {cid: sorted(ms) for cid, ms in byc.items()}
    if manifest.exists():
        rec = json.loads(manifest.read_text(encoding="utf-8"))
        assert rec == members, (f"grouping drifted from {manifest.name} -- the shipped neutral_fixed/concat "
                                f"cards were built for a DIFFERENT partition; every ne/cc/staab draft would "
                                f"use the wrong cluster's card")
    else:
        manifest.write_text(json.dumps(members, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  froze cluster membership -> {manifest.relative_to(ROOT)}")
    print(f"grouping: {len(byc)} clusters of k={K}; neutral_fixed + concat present for all {len(authors)} authors")

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

    # ---- stranger (random outsider, over the full author pool) + sham (a 20-MAD card in the email slot) ----
    rng = np.random.default_rng(SEED)
    stranger = {}
    for a in authors:
        pop = [b for b in authors if grp[b] != grp[a]]
        stranger[a] = pop[int(rng.integers(len(pop)))]
    mad = json.loads((ROOT / "data/20mad/mad_cmd_nuwa.json").read_text(encoding="utf-8"))["nuwa"]
    mad_keys = sorted(mad)
    sham = {a: mad[mad_keys[int(hashlib.sha1(f"sham-{a}".encode()).hexdigest(), 16) % len(mad_keys)]]
            for a in authors}

    CARD = {"in": lambda e: nuwa[e], "st": lambda e: nuwa[stranger[e]], "sham": lambda e: sham[e],
            "staab": lambda e: step2["staab"][e], "ne": lambda e: neutral[ck[e]], "cc": lambda e: concat[ck[e]],
            # R3' (#127): full per-person de-id battery vs v6 -- same author-indexed pattern as staab.
            "petre_k4": lambda e: step2["petre_k4"][e], "tpar_t15": lambda e: step2["tpar_t15"][e],
            "presidio": lambda e: step2["presidio"][e],
            "nec": lambda e: neclean[ck[e]]}
    arms = sorted({a for c in CONTRASTS for a in c}, key=list(CARD).index)
    # PROBE_ARM: which arm's drafts carry the self/pad/fmt/cut probes (ported from mad_fc_export). Default
    # "in"; a pack whose CONTRASTS exclude `in` (e.g. ne-nec) sets it to a drafted arm -- the battery
    # validates the JUDGE on draft-vs-modified-self, so the carrier arm is immaterial.
    PROBE_ARM = os.environ.get("PROBE_ARM", "in")
    assert PROBE_ARM in arms, f"PROBE_ARM={PROBE_ARM!r} not among drafted arms {arms}"

    def dkey(arm, e):
        """what an arm's draft actually depends on: the cluster for pooled arms, the author otherwise."""
        return grp[e] if arm in PERCLUSTER else e

    if COST:
        # one draft per DISTINCT (arm, dkey, task)
        keys = {(arm, dkey(arm, e), t) for arm in arms for e in experts for t in range(len(T))}
        tin = sum(len(ENC.encode((CARD[arm](next(e for e in experts if dkey(arm, e) == dk)) or "") + T[t]))
                  for (arm, dk, t) in keys)
        print(f"\nCOST: {len(keys)} drafts | input ~{tin:,} tok | output <= {len(keys)*DRAFT_TOK:,} tok "
              f"({GEN}); judges are free subagents")
        return

    # ---- drafts: one per DISTINCT (arm, dkey, task); pooled arms drafted once per cluster and reused ----
    jobs, seen = [], set()
    for arm in arms:
        for e in experts:
            dk = dkey(arm, e)
            for t in range(len(T)):
                if (arm, dk, t) in seen:
                    continue
                seen.add((arm, dk, t))
                jobs.append((arm, dk, t, CARD[arm](e)))
    print(f"\ndrafting {len(jobs)} distinct (arm,key,task) drafts over {len(experts)} experts x {len(T)} tasks "
          f"x {len(arms)} arms ({arms}) ...", flush=True)
    D, resampled = {}, defaultdict(int)
    for (arm, dk, t, _card), (txt, nr) in zip(jobs, de.pool(lambda j: draft_untruncated(j[3], T[j[2]]), jobs)):
        assert txt, (f"arm {arm} key {dk} task {t}: draft still truncated after 3 resamples -- this task makes "
                     f"the model run away; raise DRAFT_TOK or drop the task")
        D[(arm, dk, t)] = txt
        if nr:
            resampled[arm] += 1
    for arm in arms:
        L = np.array([len(D[(arm, dkey(arm, e), t)]) for e in experts for t in range(len(T))])
        print(f"  {arm:6s} mean {L.mean():6.0f} chars  median {np.median(L):6.0f}  max {L.max():5d}  "
              f"resampled {resampled[arm]}", flush=True)

    units = [(e, t) for e in experts for t in range(len(T))]

    def draft_of(arm, i):
        e, t = units[i]
        return D[(arm, dkey(arm, e), t)]

    meta, items = {}, []

    def add(pid, kind, i, A, B, **kw):
        e, t = units[i]
        meta[pid] = {"kind": kind, "unit": i, "u": e, **kw}
        items.append({"pid": pid, "prompt": fill(USR, q=T[t], a=A, b=B, pid=pid)})

    # real contrasts, both orders. A per-cluster x per-cluster contrast (ne-cc) is deduped to one item per
    # (cluster, task): its A/B pair is identical for every member of a cluster, so judging it per member is
    # wasted. Per-person contrasts differ for every expert and are never skipped.
    seen_pair = set()
    for ci, (x, y) in enumerate(CONTRASTS):
        for i in range(len(units)):
            e, t = units[i]
            sig = (x, y, dkey(x, e), dkey(y, e), t)
            if sig in seen_pair:
                continue
            seen_pair.add(sig)
            for o in (0, 1):
                A, B = (draft_of(x, i), draft_of(y, i)) if o == 0 else (draft_of(y, i), draft_of(x, i))
                add(f"C{ci}{i:04d}{o}", "contrast", i, A, B, x=x, y=y, order=o)

    # Placebo/cut carriers: ONE unit per expert, NOT a contiguous prefix. The unit list is expert-major
    # (8 tasks per expert in a block), so `range(NPLA)` would land the whole battery on the first 2-4
    # experts -- and the scorer clusters the battery by expert, so its `resolves_finest` gate would bootstrap
    # over 2 clusters and a single atypical expert could FAIL the battery and void every contrast AFTER the
    # paid drafting. Spread across experts (rotating the task so the battery is not single-question) so the
    # gate has ~NPLA / ~NCUT clusters. carriers[j] = unit index of (experts[j], task j % len(T)).
    carriers = [j * len(T) + (j % len(T)) for j in range(len(experts))]
    for i in carriers[:NPLA]:
        d = draft_of(PROBE_ARM, i)
        add(f"S{i:04d}0", "self", i, d, d)                                        # position
        tgt = int(len(d) * 1.25)
        for o in (0, 1):
            A, B = (pad(d, tgt), d) if o == 0 else (d, pad(d, tgt))               # length: A longer iff o==0
            add(f"P{i:04d}{o}", "pad", i, A, B, order=o)
            A, B = (d, strip_md(d)) if o == 0 else (strip_md(d), d)               # format: A rich iff o==0
            add(f"F{i:04d}{o}", "fmt", i, A, B, order=o)
    for i in carriers[:NCUT]:                                                     # range probe
        d = denum(draft_of(PROBE_ARM, i))
        for j, p in enumerate(CUTS):
            c = cut(d, p, seed=i * 100 + j)
            if c is None:
                continue
            for o in (0, 1):
                A, B = (d, c) if o == 0 else (c, d)                               # A is the full draft iff o==0
                add(f"X{j}{i:04d}{o}", "cut", i, A, B, p=p, order=o)

    # BATCHING (same two rules as cv_fc_export): scatter by pair so a judge never sees one unit's items massed
    # (it would spot the near-identical placebos), and route the two ORDERS of a pair to different batches so
    # no single judge can flip a pair mechanically. Both orders are still measured for every pair, so position
    # bias cancels in the contrast.
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
    # A pid is (contrast, unit, order); it does NOT depend on the draft text. So if a re-export changes a
    # prompt (a draft resampled), any `ans` already written for that batch was judged against the OLD prompt,
    # and fc_status only checks pid coverage -- it would call it done. Detect the change and stale those out.
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
        {a: int(np.mean([len(draft_of(a, i)) for i in range(len(units))])) for a in arms}, indent=1),
        encoding="utf-8")
    (OUT / "config.json").write_text(json.dumps(
        {"dataset": "enron", "nexpert": NEXPERT, "subsampled": len(experts) < len(authors),
         "n_units": len(units), "n_experts": len(experts), "contrasts": CONTRASTS, "arms": arms,
         "cuts": list(CUTS), "k": K, "seed": SEED, "pooled_arms": sorted(PERCLUSTER),
         "draft_tok": DRAFT_TOK, "temp": TEMP, "resampled_runaways": dict(resampled),
         "unit_expert": [e for e, t in units], "unit_cluster": [grp[e] for e, t in units],
         "units": [f"{e}|task{t}" for e, t in units]}, indent=1), encoding="utf-8")

    kinds = defaultdict(int)
    for m in meta.values():
        kinds[m["kind"]] += 1
    outname = OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT
    print(f"\n{len(items)} items {dict(kinds)} -> {nb} batches ({[len(b) for b in batches]}) "
          f"-> {outname}")
    print(f"  {len(units)} units / {len(experts)} experts / {ncl} clusters")


if __name__ == "__main__":
    main()
