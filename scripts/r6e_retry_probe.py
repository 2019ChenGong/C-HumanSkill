"""R6e (ELEMK_DESIGN.md R6e) — retry-budget probe for the fid-death rewriter family.

Would MORE retry rounds rescue the lines a swapped rewriter dropped, or is the 5-round ladder not
the bottleneck? Reruns v5_sanitize._rewrite_min VERBATIM on the previously-dropped lines only, with
MAXRETRY monkeypatched 4 -> 11 (12 rounds/stage; per the user's rule, beyond 12 = model problem,
untested). Rounds 1-5 replay byte-identically from the chat() cache (each line's feedback chain
depends only on its own cached attempts), so round 6 continues the build's exact ladder state and
only rounds 6+ spend money. Frozen prompts/gates/thresholds/temperature/SAN_EXTRA untouched.

Sub-list legality: _rewrite_min treats lines independently (per-line messages depend only on
(orig line, its runs, its own fail chain)); batching does not couple lines.

Sentinel: an outright pass derived at round <= 5 would mean the cache-replay assumption broke
(the build already proved those rounds fail) -> hard failure, do not trust the run.

Env: PROBE_GEN (full model id, e.g. openrouter/z-ai/glm-5.1), PROBE_TAG (source arm tag, e.g.
v6min_glm51), CLUSTERS (optional comma list of cluster suffixes, e.g. G8,G13,G11), MAXR (default 11).
Output: results/r6e_retry_probe_<slug>.json — INTERNAL (contains member-overlapping original lines),
never released with cards. No data/ writes.

  PROBE_GEN=openrouter/z-ai/glm-5.1 PROBE_TAG=v6min_glm51 python -P scripts/r6e_retry_probe.py
"""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

PROBE_GEN = os.environ["PROBE_GEN"]
PROBE_TAG = os.environ["PROBE_TAG"]
CLUS_F = set(filter(None, os.environ.get("CLUSTERS", "").split(",")))
MAXR = int(os.environ.get("MAXR", "11"))
assert PROBE_GEN != "deepseek-chat", "probe targets the swapped-rewriter arms"

# v5_sanitize reads its config from env at import time; TAG=r6e_probe only satisfies the
# non-canonical-TAG assert — the probe never calls stage_build, so nothing is written under it.
os.environ.update({"SAN_GEN": PROBE_GEN, "TAG": "r6e_probe", "EDIT": "min",
                   "DATASET": "mad", "K": "8", "SEED": "0", "GROUP": "random"})
import v5_sanitize as VS                         # noqa: E402
EB = VS.EB

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

assert VS.GEN == PROBE_GEN and VS.EDIT == "min" and VS.SAMPLE is None
VS.MAXRETRY = MAXR                               # the ONLY registered deviation from the build
R = 1 + MAXR                                     # rounds per stage (12 at the default)

# ---- source arm audit: which lines died, and where -------------------------------------------
# Review MAJOR-1: the arm's stats _config must name PROBE_GEN — a mismatched tag means NONE of the
# original cache rows can hit (model is in the cache key): all 12 rounds become fresh paid calls and
# the sentinel cannot catch it (a different model can fail its first 5 fresh rounds too).
_st = json.loads((EB.CG.SE / f"{EB._CARDBASE}__{PROBE_TAG}_stats.json").read_text(encoding="utf-8"))
_arm_gen = _st.get("_config", {}).get("GEN", "deepseek-chat")
assert _arm_gen == PROBE_GEN, f"PROBE_TAG={PROBE_TAG} was built with {_arm_gen!r}, not {PROBE_GEN!r}"
aud_p = EB.CG.SE / f"{EB._CARDBASE}__{PROBE_TAG}_audit.json"
aud = json.loads(aud_p.read_text(encoding="utf-8"))
targets = {}                                     # ck -> [(li, final_fail)]
for ck, rows in aud.items():
    if ck.startswith("_") or not ck.startswith("k8_s0_"):
        continue
    if CLUS_F and ck.split("_")[-1] not in CLUS_F:
        continue
    dead = [(li, r.get("final_fail", "?")) for li, r in enumerate(rows) if r.get("tier") == "dropped"]
    if dead:
        targets[ck] = dead
n_lines = sum(len(v) for v in targets.values())
print(f"[r6e] {PROBE_GEN} on {PROBE_TAG}: {n_lines} dropped lines in {len(targets)} clusters; "
      f"{R} rounds/stage; fresh-call bound ~{n_lines * 2 * (R - 5)} (rounds 6+ only)")

# ---- rebuild the exact _rewrite_min inputs (mirrors stage_build) -----------------------------
aggro, byc = EB.load_clusters()
cache = json.loads(EB.ELEMS_P.read_text(encoding="utf-8"))
clus = EB.cluster_elements(byc, cache)
cards = json.loads(VS.NE_P.read_text(encoding="utf-8"))

records, sentinel_fail = [], []
for ck in sorted(targets):
    texts, owners = clus[ck]
    src_sh = set()
    for t in texts:
        src_sh |= VS._shingles(t)
    for m in sorted(set(owners)):
        src_sh |= VS._shingles(aggro[m])
    parsed = [VS._split_line(ln) for ln in cards[ck].splitlines()]
    content = [s for _p, s in parsed if VS._is_content(s)]
    orig = [content[li] for li, _f in targets[ck]]
    OV = np.stack(EB.embed(orig))

    cur, tier, retries, dropped, fail, runs_of = VS._rewrite_min(orig, OV, src_sh)

    for j, (li, build_fail) in enumerate(targets[ck]):
        wl = len(re.findall(r"[a-z']+", orig[j].lower()))
        cov = sum(len(r.split()) for r in runs_of[j]) / max(wl, 1)
        route = cov > 0.90
        t = tier.get(j, "dropped")
        ret = retries[j]
        if t == "strict":
            rnd, label = ret + 1, f"s1@r{ret + 1}"
        elif t == "relaxed":
            rnd, label = R, f"s1-salvage@{R}"
        elif t == "rewrite":                      # route lines never enter stage 1
            rnd, label = ret + 1, f"s2@r{ret + 1}"
        elif t == "fallback":
            rnd, label = ret - R + 1, f"s2@r{ret - R + 1}"
        elif t in ("rewrite_relaxed", "fallback_relaxed"):
            rnd, label = R, f"s2-salvage@{R}"
        else:
            # review MINOR-4: any tier other than a terminal drop here means the input
            # reconstruction diverged — exactly what this probe must not paper over
            assert t == "dropped", f"unexpected tier {t!r} for probed line {ck}:{li}"
            rnd, label = None, "still_dropped"
        # cache-replay sentinel: outright passes can only happen on fresh rounds (>= 6)
        if t in ("strict", "rewrite", "fallback") and rnd is not None and rnd <= 5:
            sentinel_fail.append((ck, li, t, rnd))
        # review MINOR-3: fail[] is only meaningful for terminal drops (rescued lines carry a
        # stale stage-1 entry or none at all) — blank the diagnostic fields for rescued lines
        mode, info = fail.get(j, ("", None)) if j in dropped else ("", None)
        last = (info[1] if mode in ("lex", "num") and info else
                info if isinstance(info, str) else None)
        last_fid = None
        if j in dropped and isinstance(last, str) and last:
            lv = np.stack(EB.embed([last]))[0]
            last_fid = round(float(lv @ OV[j]), 4)
        records.append({
            "cluster": ck, "li": li, "cov": round(cov, 3), "route": route,
            "build_final_fail": build_fail, "probe_tier": t, "probe_round": label,
            "retries": ret, "rescued": j not in dropped,
            "orig": orig[j], "rescued_text": cur.get(j),
            "last_fail_mode": mode or None, "last_attempt": last, "last_attempt_fid": last_fid,
        })
    print(f"  {ck}: {len(targets[ck])} probed, rescued "
          f"{len(targets[ck]) - len(dropped)}/{len(targets[ck])}", flush=True)

if sentinel_fail:
    print(f"[r6e] SENTINEL FAIL — outright pass at replayed round <= 5, cache-replay assumption "
          f"broken: {sentinel_fail}")
    sys.exit(1)

# ---- summary ---------------------------------------------------------------------------------
resc = [r for r in records if r["rescued"]]
dead = [r for r in records if not r["rescued"]]
from collections import Counter                  # noqa: E402
print(f"\n[r6e] {PROBE_GEN}: rescued {len(resc)}/{len(records)} "
      f"({100 * len(resc) / max(len(records), 1):.0f}%) at {R} rounds/stage")
print("  rescue rounds:", dict(Counter(r["probe_round"] for r in resc).most_common()))
print("  rescue tiers:", dict(Counter(r["probe_tier"] for r in resc).most_common()))
if dead:
    fids = sorted((r["last_attempt_fid"] for r in dead if r["last_attempt_fid"] is not None),
                  reverse=True)
    print(f"  still dead {len(dead)}: last-attempt fid top5 {fids[:5]} "
          f"(salvage floors .65/.80) | modes {dict(Counter(r['last_fail_mode'] for r in dead))}")

slug = re.sub(r"[^a-z0-9]+", "_", PROBE_GEN.split("/")[-1].lower()).strip("_")
out_p = ROOT / "results" / f"r6e_retry_probe_{slug}.json"
out_p.write_text(json.dumps({
    "_config": {"PROBE_GEN": PROBE_GEN, "PROBE_TAG": PROBE_TAG, "MAXR": MAXR,
                "rounds_per_stage": R, "clusters_filter": sorted(CLUS_F) or None,
                "note": "INTERNAL file (member-overlapping original lines); never released"},
    "records": records,
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[r6e] wrote {out_p}")
