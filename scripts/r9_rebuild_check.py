"""R9 (#133) — rebuild-stability programmatic checks (P2-P5; P1 = elemk_v3_gates MODE=lex, run
separately with CARDS=v6min_rerun).

Compares the fresh-sampling rerun build (TAG=v6min_rerun) against the canonical v6min build for ONE
dataset per process, verifying FROM FILES (never trusting the build's own printouts):

  P2  drop gate <= 5% of content lines
  P3  per-line gate re-checks on every non-dropped rerun line: numbers preserved, punt status
      preserved, fidelity >= .65 global floor (embeddings recomputed), length +-30%+3w for the
      stage-1 tiers (strict/relaxed) only — stage-2 has no length gate by construction; plus
      deterministic-routing asserts: verbatim line set identical to canonical, and the
      {rewrite, rewrite_relaxed} UNION identical (review M2: per-tier comparison would flag
      expected sampling jitter)
  P5  anti-cache tripwire: every rerun card byte-differs from its canonical sibling
  P4  stability descriptives + the review-M3 VETO thresholds (|mean_changed_frac drift| > .10 or
      fallback-family tier share drift > 15pp voids the no-utility-retest argument for that dataset)

  DATASET=mad K=8 SEED=0 GROUP=random python -P scripts/r9_rebuild_check.py
  (cv: K=8 SEED=0 / enron: K=8 SEED=1)
"""
import os
import sys
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GROUP", "random")
os.environ.setdefault("EDIT", "min")
import v5_sanitize as VS                         # noqa: E402  (frozen REQ/_shingles/_split_line/... )
import elemk_build as EB                         # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DS, K, SEED = EB.DS, EB.K, EB.SEED
PREFIX = f"k{K}_s{SEED}_"
CANON = {"mad": (8, 0), "cv": (8, 0), "enron": (8, 1)}
assert (K, SEED) == CANON[DS], f"R9 runs on the canonical partition only, got ({K},{SEED}) for {DS}"

ne_cards = json.loads(VS.NE_P.read_text(encoding="utf-8"))
# member shingle pools (mirror stage_build) — dry-run amendment: routing is verified against a
# fresh INPUT-side recomputation, not against the canonical audit's tier labels. A dropped line
# keeps tier "dropped" and loses its route label, so comparing tier-label sets across builds
# falsely fails whenever the two builds drop different lines (seen on CV G2: canon 2 drops, rerun 0).
import re as _re                                 # noqa: E402
aggro, byc = EB.load_clusters()
clus = EB.cluster_elements(byc, json.loads(EB.ELEMS_P.read_text(encoding="utf-8")))
C = {n: json.loads((EB.CG.SE / f"{EB._CARDBASE}__v6min{n and '_' or ''}{n}.json").read_text(encoding="utf-8"))
     for n in ("", "audit", "stats")}
R = {n: json.loads((EB.CG.SE / f"{EB._CARDBASE}__v6min_rerun{n and '_' or ''}{n}.json").read_text(encoding="utf-8"))
     for n in ("", "audit", "stats")}

cks = sorted(k for k in R[""] if k.startswith(PREFIX))
canon_cks = sorted(k for k in C[""] if k.startswith(PREFIX))
assert cks == canon_cks and cks, f"cluster key sets differ: rerun {len(cks)} vs canonical {len(canon_cks)}"

cfg_r = R["stats"]["_config"]
assert cfg_r.get("SAN_SAMPLE") is not None, "rerun stats carry no SAN_SAMPLE stamp — was this a fresh-sampling build?"
for f in ("prompt_sha1", "min_prompt_sha1", "req_sha1", "FID", "MINFID", "MINFID_FLOOR", "MAXRETRY", "SHN"):
    assert C["stats"]["_config"][f] == cfg_r[f], f"config drift on {f}: {C['stats']['_config'][f]} vs {cfg_r[f]}"


def extract(card_text, aud, raw, parsed, content_set):
    """Invert stage_build's assembly: recover {line_idx: new content text} from the card file.
    Round-trip quirk (dry-run amendment 2): '\n'.join(out_lines).splitlines() eats exactly ONE
    trailing empty element, so when the card's last surviving assembled line is an empty
    non-content line (e.g. the final content line after it was dropped — CV G4), the file is one
    line shorter than the surviving-line count. That single pattern is legal; anything else fails."""
    lines = card_text.splitlines()
    drop = {r["line"] for r in aud if r["dropped"]}
    out, j = {}, 0
    for i, ln in enumerate(raw):
        pref, _s = parsed[i]
        if i in content_set and i in drop:
            continue
        if j == len(lines):
            surviving_rest = [x for x in range(i, len(raw)) if not (x in content_set and x in drop)]
            assert surviving_rest == [i] and i not in content_set and ln == "", \
                f"card exhausted at line {i} ({ln[:40]!r}) — not the trailing-empty round-trip pattern"
            return out
        if i in content_set:
            got = lines[j]
            assert got.startswith(pref), f"prefix mismatch at line {i}: {got[:40]!r}"
            out[i] = got[len(pref):]
            j += 1
        else:
            assert lines[j] == ln, f"non-content line {i} not verbatim: {lines[j][:40]!r}"
            j += 1
    assert j == len(lines), f"card has {len(lines)} lines, consumed {j}"
    return out


STAGE1 = ("strict", "relaxed")
FALLBACK_FAMILY = ("relaxed", "fallback", "fallback_relaxed", "rewrite_relaxed")
fails, pairs = [], []          # pairs: (ck, i, orig, new) for changed rerun lines -> fidelity batch
per, tier_c_all, tier_r_all = {}, Counter(), Counter()
n_content = n_drop_r = n_ident = n_common_changed = 0
cf_c, cf_r = [], []

for ck in cks:
    aud_c, aud_r = C["audit"][ck], R["audit"][ck]
    assert [r["line"] for r in aud_c] == [r["line"] for r in aud_r], f"{ck}: audit line indices differ"
    raw = ne_cards[ck].splitlines()
    parsed = [VS._split_line(ln) for ln in raw]
    content_idx = [i for i, (_p, s) in enumerate(parsed) if VS._is_content(s)]
    assert content_idx == [r["line"] for r in aud_r], f"{ck}: content-line set drifted from audit"
    cset = set(content_idx)

    # P5 anti-cache tripwire
    if R[""][ck] == C[""][ck]:
        fails.append(f"P5 {ck}: rerun card byte-identical to canonical — sampling did not dodge the cache")

    ex_c = extract(C[""][ck], aud_c, raw, parsed, cset)
    ex_r = extract(R[""][ck], aud_r, raw, parsed, cset)
    tc = {r["line"]: r["tier"] for r in aud_c}
    tr = {r["line"]: r["tier"] for r in aud_r}
    tier_c_all.update(tc.values()); tier_r_all.update(tr.values())
    cf_c += [r["changed_frac"] for r in aud_c if "changed_frac" in r]
    cf_r += [r["changed_frac"] for r in aud_r if "changed_frac" in r]

    # P3-5 deterministic routing — recompute the route from inputs (_hit_runs + cov>0.90, exactly
    # as _rewrite_min does) and hold BOTH builds' audits to it; dropped lines are exempt from the
    # route-label check (their label is "dropped") but must come from a non-verbatim route.
    texts, owners = clus[ck]
    src_sh = set()
    for t in texts:
        src_sh |= VS._shingles(t)
    for m in sorted(set(owners)):
        src_sh |= VS._shingles(aggro[m])
    RW_T, MIN_T = ("rewrite", "rewrite_relaxed"), ("strict", "relaxed", "fallback", "fallback_relaxed")
    for i in content_idx:
        orig = parsed[i][1]
        runs = VS._hit_runs(orig, src_sh)
        if not runs:
            route = "verbatim"
        else:
            wl = len(_re.findall(r"[a-z']+", orig.lower()))
            route = "rw" if sum(len(r.split()) for r in runs) / max(wl, 1) > 0.90 else "min"
        for side, t in (("canon", tc[i]), ("rerun", tr[i])):
            ok = ((route == "verbatim" and t == "verbatim")
                  or (route == "rw" and (t in RW_T or t == "dropped"))
                  or (route == "min" and (t in MIN_T or t == "dropped")))
            if not ok:
                fails.append(f"P3-route {ck} L{i} ({side}): input route {route} but tier {t}")

    for i in content_idx:
        n_content += 1
        if tr[i] == "dropped":
            n_drop_r += 1
            continue
        orig, new = parsed[i][1], ex_r[i]
        if tr[i] == "verbatim":
            if new != orig:
                fails.append(f"P3 {ck} L{i}: verbatim line not byte-identical")
            continue
        if VS._nums(orig) - VS._nums(new):
            fails.append(f"P3-num {ck} L{i}: lost {VS._nums(orig) - VS._nums(new)}")
        if bool(VS.REQ.search(orig)) != bool(VS.REQ.search(new)):
            fails.append(f"P3-punt {ck} L{i}: punt status flipped")
        if tr[i] in STAGE1:
            wo, wn = len(orig.split()), len(new.split())
            if abs(wn - wo) > max(3, int(0.30 * wo)):
                fails.append(f"P3-len {ck} L{i}: {wo}->{wn} words (tier {tr[i]})")
        pairs.append((ck, i, orig, new))
        if i in ex_c and tc[i] != "verbatim":
            n_common_changed += 1
            n_ident += ex_r[i] == ex_c[i]

# P3 fidelity floor — recompute embeddings (sqlite-cached; identical vectors to the build's own)
if pairs:
    OV = np.stack(EB.embed([p[2] for p in pairs]))
    NV = np.stack(EB.embed([p[3] for p in pairs]))
    fid = (OV * NV).sum(axis=1)
    for (ck, i, _o, _n), f in zip(pairs, fid):
        if f < 0.65 - 1e-6:
            fails.append(f"P3-fid {ck} L{i}: cos {f:.3f} < .65 floor")

# P2 drop gate
drop_rate = n_drop_r / max(n_content, 1)
if drop_rate > 0.05:
    fails.append(f"P2: rerun drop rate {drop_rate:.1%} > 5% gate")

# P4 descriptives + review-M3 veto
assert cf_c and cf_r, "no changed_frac rows — NaN would silently disarm the M3 veto"
mcf_c, mcf_r = float(np.mean(cf_c)), float(np.mean(cf_r))
fam_c = sum(tier_c_all[t] for t in FALLBACK_FAMILY) / max(sum(tier_c_all.values()), 1)
fam_r = sum(tier_r_all[t] for t in FALLBACK_FAMILY) / max(sum(tier_r_all.values()), 1)
veto = abs(mcf_r - mcf_c) > 0.10 or abs(fam_r - fam_c) > 0.15
wr_c = float(np.mean([C["stats"][ck]["words_v5"] / C["stats"][ck]["words_ne"] for ck in cks]))
wr_r = float(np.mean([R["stats"][ck]["words_v5"] / R["stats"][ck]["words_ne"] for ck in cks]))
ident_frac = n_ident / max(n_common_changed, 1)

out = {"dataset": DS, "k": K, "seed": SEED, "clusters": len(cks), "n_content": n_content,
       "drop_rate_rerun": round(drop_rate, 4),
       "drop_rate_canon": round(sum(1 for v in tier_c_all.elements() if v == "dropped")
                                / max(sum(tier_c_all.values()), 1), 4),
       "tiers_canon": dict(tier_c_all), "tiers_rerun": dict(tier_r_all),
       "mean_changed_frac": {"canon": round(mcf_c, 4), "rerun": round(mcf_r, 4)},
       "fallback_family_share": {"canon": round(fam_c, 4), "rerun": round(fam_r, 4)},
       "word_ratio_v6_over_ne": {"canon": round(wr_c, 4), "rerun": round(wr_r, 4)},
       "identical_changed_line_frac": round(ident_frac, 4),
       "n_changed_lines_common": n_common_changed,
       "m3_veto_triggered": bool(veto), "hard_gate_failures": fails}
op = ROOT / "results" / f"r9_rebuild_check_{DS}.json"
op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"[r9] {DS} k{K}_s{SEED}: {len(cks)} clusters, {n_content} content lines")
print(f"  P2 drop rate: rerun {drop_rate:.1%} (canon {out['drop_rate_canon']:.1%})  gate<=5%: "
      f"{'PASS' if drop_rate <= 0.05 else 'FAIL'}")
print(f"  P3 gates re-checked on {len(pairs)} changed lines + verbatim byte-checks: "
      f"{'ALL GREEN' if not any(f.startswith('P3') for f in fails) else 'FAILURES (see below)'}")
print(f"  P5 anti-cache: {'PASS (all cards differ)' if not any(f.startswith('P5') for f in fails) else 'FAIL'}")
print(f"  P4 tiers canon {dict(tier_c_all)}")
print(f"     tiers rerun {dict(tier_r_all)}")
print(f"     changed_frac {mcf_c:.3f}->{mcf_r:.3f}  fallback-family {fam_c:.1%}->{fam_r:.1%}  "
      f"words/ne {wr_c:.2f}->{wr_r:.2f}  identical-changed-line {ident_frac:.1%}")
print(f"  M3 veto: {'TRIGGERED — no-utility-retest argument VOID for this dataset' if veto else 'not triggered'}")
for f in fails:
    print(f"  !! {f}")
print(f"saved -> {op.relative_to(ROOT)}")
sys.exit(1 if fails else 0)
