"""Programmatic pid-coverage / format check for a naming round dir (never trust a subagent's own count).

Checks, per round dir:
  - stray answer files (ans_*part*, ans_chunk*, ...) that the scorer would silently ignore
  - every batch has an ans_<N>.json
  - answered pids == batch pids exactly (missing / extra / duplicate)
  - choice letter is within THAT item's candidate count (a letter past the group size = unusable)
  - conf present and an integer in [45,100]  (prompt asks 50-100; 45 seen historically)

  python -P scripts/naming_cov.py results/enron/naming_indiv_full/round1 [more dirs...]
Exit code 1 if anything is wrong, so it can gate the next round.
"""
import json
import re
import sys
from pathlib import Path

LETTERS = "ABCDEFGH"
ROOT = Path(__file__).resolve().parents[1]


def check(rdir):
    rdir = Path(rdir)
    meta = json.loads((rdir / "meta.json").read_text(encoding="utf-8"))
    bad = []
    stray = [p.name for p in rdir.glob("ans_*") if not re.fullmatch(r"ans_\d+\.json", p.name)]
    if stray:
        bad.append(f"STRAY answer files (scorer ignores these): {stray}")

    batches = sorted(rdir.glob("batch_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    for bp in batches:                                   # every batch needs an answer file
        n = int(bp.stem.split("_")[1])
        if not (rdir / f"ans_{n}.json").exists():
            items = json.loads(bp.read_text(encoding="utf-8"))
            bad.append(f"batch_{n}: NO ans_{n}.json ({len(items)} items unanswered)")

    # EFFECTIVE answer per pid = last record wins, in the SAME file order the scorer uses
    # (sorted(glob) -> ans_10 < ans_2 < ans_33 < ans_4 ...). A patch file written with a higher
    # index therefore supersedes an earlier bad record; only the effective answer is validated,
    # superseded ones are reported as info, not as errors.
    seen, superseded = {}, 0
    for ap in sorted(rdir.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", ap.stem):
            continue
        try:
            recs = json.loads(ap.read_text(encoding="utf-8"))
        except Exception as e:
            bad.append(f"{ap.name}: unparseable ({e})"); continue
        for r in recs:
            if not isinstance(r, dict) or "pid" not in r:
                bad.append(f"{ap.name}: malformed record {str(r)[:80]}"); continue
            if r["pid"] in seen:
                superseded += 1
            seen[r["pid"]] = (ap.name, r)
    extra = set(seen) - set(meta)
    if extra:
        bad.append(f"{len(extra)} answered pids not in meta.json e.g. {sorted(extra)[:3]}")
    for pid, (fn, r) in seen.items():
        if pid not in meta:
            continue
        ch = str(r.get("choice", "")).strip().upper()
        if len(ch) != 1 or ch not in LETTERS:
            bad.append(f"{pid} [{fn}]: bad choice {r.get('choice')!r}"); continue
        if LETTERS.index(ch) >= len(meta[pid]["cands"]):
            bad.append(f"{pid} [{fn}]: choice {ch} past group size {len(meta[pid]['cands'])}")
        c = r.get("conf")
        if not isinstance(c, (int, float)) or not (45 <= c <= 100):
            bad.append(f"{pid} [{fn}]: bad conf {c!r}")
    if superseded:
        print(f"      i {superseded} superseded record(s) overridden by a later ans_* file (patches)")

    cov = len(set(seen) & set(meta)) / len(meta) if meta else 0
    if cov < 1.0:                                    # partial coverage is a failure, not a warning
        miss = sorted(set(meta) - set(seen))
        bad.append(f"{len(miss)} pid(s) never answered e.g. {miss[:3]}")
    tag = "OK " if not bad else "BAD"
    print(f"[{tag}] {rdir.as_posix().split('results/')[-1]:44s} "
          f"{len(batches):3d} batches  coverage {len(set(seen) & set(meta)):5d}/{len(meta):5d} = {cov:6.1%}")
    for b in bad[:12]:
        print(f"      - {b}")
    if len(bad) > 12:
        print(f"      - ... and {len(bad) - 12} more")
    return not bad and cov == 1.0


if __name__ == "__main__":
    dirs = sys.argv[1:] or [str(p) for p in sorted(ROOT.glob("results/*/naming_*/round*")) if (p / "meta.json").exists()]
    ok = all([check(d) for d in dirs])
    sys.exit(0 if ok else 1)
