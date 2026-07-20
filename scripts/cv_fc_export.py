"""CV utility, rebuilt: reference-anchored FORCED CHOICE + the placebo battery.

See .claude/skills/judge-card-utility/SKILL.md. Two properties motivate the rewrite:

  1. An absolute score has NO KNOWN NULL, so a tie between arms can never be tested for equivalence.
     A forced choice has a null of exactly 0.5 -- and `scripts/cmd_equiv_test.py` already implements
     TOST / non-inferiority / MDE against that null for the anonymity axis. Same machinery, same claim shape.
  2. The reference must be written by an OUTSIDER (gate zero, cv_refs_build.py). Otherwise the `in` arm is
     graded against its own author's idiom -- 58.4% of CV units, pre-repair.

Each (unit, contrast) is judged in BOTH orders, so position bias cancels within the unit rather than being
hoped away. Arms are never labelled; the judge sees only "Answer A" / "Answer B".

PLACEBOS (mixed into the same batches, indistinguishable from real items). They are orthogonal by design:
  self   A and B are the same draft                     -> P(choose A) must be 0.5      [position]
  pad    B is A plus content-free filler                -> win-rate must be 0.5         [length]
  fmt    B is A with markdown/LaTeX markers stripped    -> win-rate must be 0.5         [format]
  cut@p  B is A with a fraction p of sentences deleted  -> win-rate must RISE with p    [range / MDE]

`pad` adds length without information; `cut` removes information; the old CV pairwise judge conflated the two
(it scored +0.893 preferring "the longer copy", but that copy was clipped, so it had lost content too).

ARMS: no / in / st / ne (pooled CMD, k8_s0) / cc (concat, k8_s0) / staab / tpar_t15 / petre_k4.
Every per-person arm indexes `u` -- NOT the cluster representative (that bug made `staab - in` a comparison
against a stranger's card for 22 of 26 experts).

Run:  python -P scripts/cv_fc_export.py            [PILOT=20 JUDGES=2 COST=1 to price it first]
      -> results/se/cv_fc/{meta.json, sys.txt, batch_i.json}
      then free subagents write ans_i.json, then: python -P scripts/cv_fc_score.py
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

ENC = tiktoken.get_encoding("cl100k_base")   # not deepseek's tokenizer, but a tight enough upper bound
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DATASET", "cv"); os.environ.setdefault("GROUP", "random")
# BEFORE importing cv_pilot, which reads DRAFT_TOK at import time.
#
# 700 cut off 3/106 `st` drafts. Raising it to 1000 cut off a `no` draft instead -- at 967 tokens, ending
# `w = S_W^(-1) * (0.0833`, mid-formula -- because a higher cap lets the model write longer, so "close to the
# cap" is a target you never catch. Truncation whose RATE depends on the arm is a fabricated effect (cf. the
# `[:1400]` clip: 8/20 card drafts vs 1/20 nocard). So: give the cap real headroom, and detect truncation by
# whether the draft ENDS IN A COMPLETE SENTENCE, not by how close it got to the limit.
#
# A high cap is nearly free: you pay for tokens emitted, not for the ceiling. It only costs a re-draft, since
# max_tokens is part of the llm cache key.
os.environ.setdefault("DRAFT_TOK", "2000")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import cv_pilot as CVP          # noqa: E402
import cmd_gate as CG           # noqa: E402
import deid_enron as de         # noqa: E402
from src.llm import chat        # noqa: E402
from skill_prompts import load_prompts, fill    # noqa: E402

K = 8
SEED = int(os.environ.get("SEED", "0"))   # R11: multi-seed FC waves; canonical (unset) behavior unchanged
PILOT = int(os.environ.get("PILOT", "0"))            # 0 = full cohort
NBATCH = int(os.environ.get("NBATCH", "0"))          # 0 = auto (<=MAXITEM items/batch)
MAXITEM = int(os.environ.get("MAXITEM", "30"))       # 60 near-identical items made graders default to one label
COST = os.environ.get("COST", "") not in ("", "0")
NCUT = int(os.environ.get("NCUT", "16"))             # units carrying the cut@p range probe
NPLA = int(os.environ.get("NPLA", "32"))             # units carrying self/pad/fmt (the battery needs power,
OUT = ROOT / os.environ.get("BATCHDIR", "results/se/cv_fc")   # not every unit; contrasts need every unit)

# The contrasts the paper actually claims. `in`-vs-`no` is reported but confounded (a wrong-domain card
# helps as much as the right one), so it never carries a claim on its own.
CONTRASTS = [("ne", "in"),        # CLAIM 2: pooling costs nothing vs the individual card
             ("ne", "cc"),        # CLAIM 3: CMD vs the naive concat baseline
             ("ne", "staab"),     # CLAIM 4: CMD vs per-person de-id
             ("ne", "tpar_t15"),
             ("ne", "petre_k4"),
             ("staab", "in"),     # does per-person de-id cost utility?
             ("in", "st"),        # control: is it THIS person's card?
             ("in", "sham"),      # control: does the RIGHT domain beat a bug-triage card? if not, `in - no`
             ("in", "no")]        #          measures "a persona block is present", not domain skill
PILOT_CONTRASTS = [("ne", "in"), ("in", "no")]
# CONTRASTS env-overridable ("ne-nec" -> the V4/elemk vs-black-box packs; ported from mad_fc_export).
# nec = an injected pooled-card arm loaded from NEUTRALCLEAN (same key scheme as ne); cluster-bootstrapped.
if os.environ.get("CONTRASTS"):
    CONTRASTS = [tuple(c.split("-")) for c in os.environ["CONTRASTS"].split(",")]

CUTS = (0.10, 0.25, 0.50)
# Distinct sentences, not one sentence repeated: a smoke-test judge noticed the repetition and called the
# padded answer out for saying the same thing twice. Each of these is content-free w.r.t. any statistics
# question -- they add length and no claim.
FILLER = ["Note that the appropriate choice depends on the specifics of your setting.",
          "It is worth considering the context in which the data were collected.",
          "As always, the details of the application matter for how you proceed.",
          "Different practitioners may reasonably emphasise different aspects here.",
          "The right call will depend on what you are ultimately trying to learn.",
          "None of this removes the need to think carefully about your own case.",
          "There is, of course, more that could be said about each of these points.",
          "In practice one weighs these considerations against the goal at hand."]
# Only markdown DECORATION. A bare `*` is multiplication (`2 * 3`) and `$` delimits LaTeX; stripping either
# damages content, which would turn this null probe into a content probe that passes by cancellation
# (format-preference netting against damage-aversion).
MD = re.compile(r"(\*\*|`|^#{1,6}\s*|^\s*[-+]\s+(?=\S))", re.M)
# "ends like a finished sentence": terminator, allowing trailing markdown/quote/bracket clutter.
ENDS = re.compile(r"[.!?][\s*_`\"'’”)\]]*$")
NUM = re.compile(r"(?:(?<=\s)|^)\d+\.(?=\s)")     # inline list numbering: "... convenience. 1. Differentiability"


def strip_md(t):
    return re.sub(r"[ \t]{2,}", " ", MD.sub("", t)).strip()


def denum(t):
    """Drop inline list numbering. cv_pilot.plain() collapses newlines, so `1.` `2.` sit mid-paragraph and
    the sentence splitter treats them as their own sentences. Deleting one then leaves `2. 3. 4.` dangling --
    and a judge spots the broken numbering, so `cut` would measure 'is the text damaged?' instead of
    'is content missing?'. Strip the numbers from BOTH sides of the cut probe."""
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


def truncated(text, cap):
    """Long, and not ending in a finished sentence. A SHORT draft ending in a formula is fine."""
    return len(ENC.encode(text)) >= 0.80 * cap and not ENDS.search(text)


def draft_untruncated(card, q):
    """cv_pilot.draft, but resample if the model runs away instead of finishing.

    deepseek at temperature=0 is NOT deterministic, and `max_tokens` changes the CONTENT, not just where it
    stops. The same question (a binary-similarity-coefficient question) produced 1965 tokens of runaway matrix
    algebra at cap=2000 and a clean 182-token answer at cap=3000. Raising the ceiling therefore never
    converges -- some other unit runs away at the new ceiling.

    So: detect the runaway and resample. Each retry uses a different `max_tokens`, which is part of the llm
    cache key, so it is a genuine new sample AND still cached -- the run stays idempotent and the drafts
    already on disk are not invalidated.

    Returns (draft, n_resamples) -- or (None, 3) if it never finished a sentence.
    """
    # First attempt goes through cv_pilot.draft verbatim, so it hits the cache entries already on disk.
    d = CVP.plain(CVP.draft(card, q) or "")
    if d.strip() and not truncated(d, CVP.DRAFT_TOK):
        return d, 0
    # Retries: identical messages, different max_tokens => different cache key => a real resample, still cached.
    prof = f"Statistical-consulting profile:\n{card}\n\n" if card else ""
    msgs = [{"role": "system", "content": "You answer a statistics question competently and correctly."},
            {"role": "user", "content": f"{prof}Question:\n{q}\n\nGive your answer: the recommended approach "
                                        f"and the key reasoning. Be concise and correct."}]
    for i, cap in enumerate((CVP.DRAFT_TOK + 1000, CVP.DRAFT_TOK + 1500), start=1):
        d = CVP.plain(chat(msgs, model=CVP.GEN, temperature=0.0, max_tokens=cap) or "")
        if d.strip() and not truncated(d, cap):
            return d, i
    return None, 3


def main():
    P = load_prompts()
    SYS, USR = P["forced_choice.system"], P["forced_choice.user"]

    U = json.loads((ROOT / "data/se/cv_fc_units.json").read_text(encoding="utf-8"))
    units = U["units"]
    print(f"units file: {U['meta']}")
    if PILOT:
        # deterministic spread over experts, not the first N (which would be one expert's three questions)
        units = sorted(units, key=lambda v: hashlib.sha1(f"{v['u']}|{v['qid']}".encode()).hexdigest())[:PILOT]
    contrasts = PILOT_CONTRASTS if PILOT else CONTRASTS

    # ---- cards. Every per-person arm is indexed by u. ----
    nuwa = json.loads((ROOT / "data/se/cv_cmd_nuwa.json").read_text(encoding="utf-8"))["nuwa"]
    pool = json.loads((ROOT / "data/se/cv_cmd_pool.json").read_text(encoding="utf-8"))["pool"]
    authors = [a for a in pool if a in nuwa]                 # exactly cmd_gate.load_cv's order
    grp, byc = CG.make_groups({a: "" for a in authors}, authors, K, SEED)
    neutral = json.loads((ROOT / "data/se/cmd_shared_cards_cv__neutral_fixed.json").read_text(encoding="utf-8"))
    concat = json.loads((ROOT / "data/se/cmd_concat_cards_cv.json").read_text(encoding="utf-8"))
    step2 = json.loads((ROOT / "data/se/cv_cmd_step2.json").read_text(encoding="utf-8"))
    _use_nec = any("nec" in c for c in contrasts)
    neclean = (json.loads((ROOT / "data/se" / os.environ["NEUTRALCLEAN"]).read_text(encoding="utf-8"))
               if _use_nec else {})
    ck = {a: f"k{K}_s{SEED}_{grp[a]}" for a in authors}
    # R11: assert card existence only for arms this pack actually drafts (`arms` is derived later, so derive
    # here from contrasts). The CV concat file only has k8_s0 keys; a pack that never drafts `cc` must not
    # trip over that.
    _used = {x for c in contrasts for x in c}
    for a in authors:
        assert "ne" not in _used or ck[a] in neutral, \
            f"pooled card {ck[a]} missing -- grouping does not match the shipped cards"
        assert "cc" not in _used or ck[a] in concat, f"concat card {ck[a]} missing"
        assert not _use_nec or ck[a] in neclean, f"nec card {ck[a]} missing from NEUTRALCLEAN -- build it first"
    # Key existence is NOT enough: all 9 keys exist for ANY 77-author partition, so a re-ordered `pool` would
    # silently attach some other cluster's card to every member. Freeze the membership and assert it.
    # R11 MAJOR-3: for a non-canonical seed the manifest freeze below is a no-op on first run (nothing to
    # compare against), so the grouping MUST first be cross-checked against the 2AFC audit trail of the same
    # seed's shipped cards (card_id -> member sets). Hard-fail on any diff: a drifted partition would attach
    # the WRONG cluster's card to every nec draft.
    if SEED != 0:
        xc = ROOT / "results" / "se" / f"2afc_v6min_s{SEED}" / "meta.json"
        assert xc.exists(), f"no 2AFC audit trail {xc} to cross-check the s{SEED} grouping against"
        seen = defaultdict(set)
        for mt in json.loads(xc.read_text(encoding="utf-8")).values():
            seen[mt["card_id"]].add(mt["member"])
        for cid, ms in sorted(seen.items()):
            extra = ms - set(byc.get(cid, []))
            assert not extra, (f"s{SEED} grouping mismatch on {cid}: 2AFC members {sorted(extra)} not in "
                               f"this export's cluster -- nec drafts would use the wrong card")
        print(f"  grouping cross-checked against {xc.relative_to(ROOT)}: "
              f"{sum(len(v) for v in seen.values())} member slots consistent")
    manifest = ROOT / "data" / "se" / f"cv_groups_k{K}_s{SEED}.json"
    members = {cid: sorted(ms) for cid, ms in byc.items()}
    if manifest.exists():
        rec = json.loads(manifest.read_text(encoding="utf-8"))
        assert rec == members, (f"grouping drifted from {manifest.name} -- the shipped pooled/concat cards "
                                f"were built for a DIFFERENT partition; every `ne`/`cc` draft would use the "
                                f"wrong cluster's card")
    else:
        manifest.write_text(json.dumps(members, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  froze cluster membership -> {manifest.relative_to(ROOT)}")
    # R12 MINOR-1: also cover any drafted arm that lives in step2 (e.g. tpar_t15_r2), with a friendly
    # message instead of a raw KeyError when the arm was never built.
    for m in sorted({"staab", "tpar_t15", "petre_k4"} | (_used & set(step2))):
        miss = [v["u"] for v in units if v["u"] not in step2.get(m, {})]
        assert not miss, f"de-id arm {m} missing for {len(miss)} experts, e.g. {miss[:3]} -- build it first"
    print(f"grouping: {len(byc)} clusters of k={K}; pooled+concat cards present for all {len(authors)} authors")

    # stranger = a RANDOM member of another cluster. (Picking the most cosine-similar outsider, as the old
    # code did, biases `in - st` toward zero by construction -- it hand-picks the hardest stranger.)
    rng = np.random.default_rng(SEED)
    stranger = {}
    for a in authors:
        pop = [b for b in authors if grp[b] != grp[a]]
        stranger[a] = pop[int(rng.integers(len(pop)))]

    # sham = a 20-MAD BUG-TRIAGE card injected where the statistical-consulting profile goes. If `in ~= sham`,
    # the card-vs-nocard channel is measuring "a structured expert-persona block is in the prompt", not skill.
    mad = json.loads((ROOT / "data/20mad/mad_cmd_nuwa.json").read_text(encoding="utf-8"))["nuwa"]
    mad_keys = sorted(mad)
    sham = {u: mad[mad_keys[int(hashlib.sha1(f"sham-{u}".encode()).hexdigest(), 16) % len(mad_keys)]]
            for u in authors}

    CARD = {"no": lambda u: None, "in": lambda u: nuwa[u], "st": lambda u: nuwa[stranger[u]],
            "ne": lambda u: neutral[ck[u]], "cc": lambda u: concat[ck[u]], "nec": lambda u: neclean[ck[u]],
            "staab": lambda u: step2["staab"][u], "tpar_t15": lambda u: step2["tpar_t15"][u],
            "petre_k4": lambda u: step2["petre_k4"][u], "sham": lambda u: sham[u],
            "tpar_t15_r2": lambda u: step2["tpar_t15_r2"][u]}   # R12: fresh-sample tpar rebuild
    arms = sorted({a for c in contrasts for a in c}, key=list(CARD).index)
    # PROBE_ARM: which arm's drafts carry the self/pad/fmt/cut probes (ported from mad_fc_export). Default
    # "in" (canonical). A pack whose contrasts exclude `in` (e.g. ne-nec) sets it to a drafted arm -- the
    # battery validates the JUDGE on draft-vs-modified-self, so the carrier arm is immaterial.
    PROBE_ARM = os.environ.get("PROBE_ARM", "in")
    assert PROBE_ARM in arms, f"PROBE_ARM={PROBE_ARM!r} not among drafted arms {arms}"

    if COST:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tin = sum(len(enc.encode((CARD[a](v["u"]) or "") + v["q"])) for a in arms for v in units)
        n = len(arms) * len(units)
        print(f"\nCOST: {n} drafts | input ~{tin:,} tok | output <= {n*CVP.DRAFT_TOK:,} tok "
              f"({CVP.GEN}); judges are free subagents")
        return

    # ---- drafts (deepseek t=0, cached) ----
    # NO char clip. A `[:1400]` slice used to sit here, and it truncated 8/20 `in` and 8/20 `ne` drafts
    # mid-word against 1/20 `no` -- card arms are ~30% longer, so the clip fired on them ~8x more often, and
    # a judge that sees a mangled tail marks that arm down. An arm-correlated truncation is a fabricated
    # effect pointing straight at the arms whose ties are the headline. `max_tokens=DRAFT_TOK` already bounds
    # generation, and no draft reaches it (verified: 0/60 at >=690 tok).
    print(f"\ndrafting {len(units)} units x {len(arms)} arms ({arms}) ...", flush=True)
    drafts, resampled = {}, {}
    for a in arms:
        pairs = list(de.pool((lambda aa: (lambda v: draft_untruncated(CARD[aa](v["u"]), v["q"])))(a), units))
        bad = [i for i, (x, _) in enumerate(pairs) if not x]
        assert not bad, (f"arm {a}: {len(bad)} drafts still truncated after 3 resamples, e.g. unit "
                         f"{units[bad[0]]['u']} -- this question makes the model run away; drop the unit "
                         f"explicitly rather than scoring a mangled draft")
        outs = [x for x, _ in pairs]
        rs = sum(1 for _, n in pairs if n > 0)
        resampled[a] = rs
        drafts[a] = outs
        L = np.array([len(x) for x in outs])
        tok = np.array([len(ENC.encode(x)) for x in outs])
        print(f"  {a:9s} mean {L.mean():6.0f} chars  median {np.median(L):6.0f}  max {L.max():5d}"
              f"  max_tok {tok.max():4d}  resampled {rs}/{len(outs)}", flush=True)

    meta, items = {}, []

    def add(pid, kind, i, A, B, **kw):
        meta[pid] = {"kind": kind, "unit": i, "u": units[i]["u"], **kw}
        items.append({"pid": pid, "prompt": fill(USR, q=units[i]["q"], ref=units[i]["ref"], a=A, b=B, pid=pid)})

    # real contrasts, both orders
    for ci, (x, y) in enumerate(contrasts):
        for i in range(len(units)):
            for o in (0, 1):
                A, B = (drafts[x][i], drafts[y][i]) if o == 0 else (drafts[y][i], drafts[x][i])
                add(f"C{ci}{i:03d}{o}", "contrast", i, A, B, x=x, y=y, order=o)

    # Placebos ride on a fixed PREFIX of the unit list, not on every unit: the battery needs enough n to put a
    # CI around 0.5, the contrasts need every unit. The prefix is deterministic, so a resumed run rebuilds the
    # same items.
    for i in range(min(NPLA, len(units))):
        d = drafts[PROBE_ARM][i]
        add(f"S{i:03d}0", "self", i, d, d)                                        # position
        tgt = int(len(d) * 1.25)
        for o in (0, 1):
            A, B = (pad(d, tgt), d) if o == 0 else (d, pad(d, tgt))               # length: A longer iff o==0
            add(f"P{i:03d}{o}", "pad", i, A, B, order=o)
            A, B = (d, strip_md(d)) if o == 0 else (strip_md(d), d)               # format: A rich iff o==0
            add(f"F{i:03d}{o}", "fmt", i, A, B, order=o)
    for i in range(min(NCUT, len(units))):                                        # range probe
        d = denum(drafts[PROBE_ARM][i])     # BOTH sides de-numbered: see denum()
        for j, p in enumerate(CUTS):
            c = cut(d, p, seed=i * 100 + j)
            if c is None:
                continue
            for o in (0, 1):
                A, B = (d, c) if o == 0 else (c, d)                               # A is the full draft iff o==0
                add(f"X{j}{i:03d}{o}", "cut", i, A, B, p=p, order=o)

    # BATCHING. Two rules, each bought with a bug.
    #
    # (1) DO NOT batch by unit. The absolute-score instrument had to, because an absolute grader has a
    #     leniency parameter that lands inside every paired difference. A forced choice has no leniency --
    #     the judge must pick one of two -- so within-judge pairing buys nothing here, and it costs: a smoke
    #     test judge given one unit's ~13 items saw the same question six times, noticed that most A/B pairs
    #     were near-identical, and reported "this appears to be an intentional QA test". A judge that has
    #     recognised the placebos is no longer measuring anything. Scatter by pair instead.
    #
    # (2) The two orders of a pair must reach DIFFERENT judges. One judge seeing (X,Y) then (Y,X) can flip
    #     mechanically. Split across judges, the two judgements are independent; position bias still cancels
    #     because both orders are measured for every pair.
    pairs = defaultdict(list)
    for t in items:
        m = meta[t["pid"]]
        pairs[(m["unit"], m["kind"], m.get("x"), m.get("y"), m.get("p"))].append(t)
    nb = NBATCH or max(2, -(-len(items) // MAXITEM))
    assert nb >= 2, "need >= 2 batches to separate the two orders of a pair"
    batches = [[] for _ in range(nb)]
    for key, ts in pairs.items():
        # R12 MINOR-4: sibling contrasts sharing the x-side draft (nec-tpar_t15 vs nec-tpar_t15_r2) must
        # not co-batch -- one judge seeing the same nec draft against two paraphrases of one card is a
        # blinding risk. Hash on the rebuild-stripped key and shift the rebuild sibling one slot. Packs
        # without _rN arms hash exactly as before (gate B byte-repro verifies).
        ck = tuple(re.sub(r"_r\d+$", "", v) if isinstance(v, str) else v for v in key)
        base = (int(hashlib.sha1("|".join(map(str, ck)).encode()).hexdigest()[:8], 16) + (ck != key)) % nb
        for t in ts:
            o = meta[t["pid"]].get("order", 0)
            batches[(base + (nb // 2 if o == 1 else 0)) % nb].append(t)
    assert (nb // 2) % nb != 0, "offset collapses; need nb >= 2"
    for bi, b in enumerate(batches):        # order within a batch must carry no information
        random.Random(SEED * 1000 + bi).shuffle(b)

    OUT.mkdir(parents=True, exist_ok=True)
    # A pid is (contrast, unit, order) -- it does NOT depend on the draft text. So if a re-export changes a
    # prompt (e.g. a draft that wasn't cached the first time got resampled), any `ans` already written for
    # that batch was judged against the OLD prompt, and fc_status.py -- which only checks pid coverage --
    # would happily call it done. Detect the change and stale out those answers.
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
        print(f"\n  ⚠ {len(changed)} batches changed content on re-export -> staled {n} answer file(s). "
              f"They were judged against different prompts.", flush=True)
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "sys.txt").write_text(SYS, encoding="utf-8")
    (OUT / "draft_len.json").write_text(json.dumps(
        {a: int(np.mean([len(x) for x in drafts[a]])) for a in arms}, indent=1), encoding="utf-8")
    # `unit_cluster` matters for the scorer: the `ne`/`cc` arms inject ONE card shared by every member of a
    # pooling cluster, so a good/bad pooled card moves all of that cluster's units together. Bootstrapping by
    # drafter alone would not absorb that card-level random effect and would give the HEADLINE contrasts CIs
    # that are too narrow -- i.e. it would over-declare "TIE".
    (OUT / "config.json").write_text(json.dumps(
        {"pilot": PILOT, "n_units": len(units), "n_experts": len(set(v["u"] for v in units)),
         "contrasts": contrasts, "arms": arms, "cuts": list(CUTS), "k": K, "seed": SEED,
         "pooled_arms": ["ne", "cc"] + (["nec"] if _use_nec else []),
         "draft_tok": CVP.DRAFT_TOK, "resampled_runaways": resampled,
         "unit_expert": [v["u"] for v in units],
         "unit_cluster": [grp[v["u"]] for v in units],
         "units": [f"{v['u']}|{v['qid']}" for v in units]}, indent=1), encoding="utf-8")

    kinds = defaultdict(int)
    for m in meta.values():
        kinds[m["kind"]] += 1
    outname = OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT
    print(f"\n{len(items)} items {dict(kinds)} -> {nb} batches ({[len(b) for b in batches]}) "
          f"-> {outname}")
    print(f"  {len(units)} units / {len(set(v['u'] for v in units))} experts")


if __name__ == "__main__":
    main()
