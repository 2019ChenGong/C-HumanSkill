"""Resumable-run bookkeeping for a forced-choice judge pack. Answers: what is still missing, and what is rotten.

A full CV run is ~140 free-subagent judge calls. It WILL be interrupted -- by a killed session, a subagent
that dies mid-write, a rate limit. The only thing that makes that survivable is being able to ask, at any
moment, exactly which (replicate, batch) pairs still need a judge, and to distrust the ones that came back
malformed rather than silently scoring them.

A batch is DONE only if its `ans` file parses, covers every pid in the batch exactly once, and every verdict
is a clean "A" or "B". Anything else is quarantined to `<name>.bad` so the next dispatch re-runs it. (A
half-written file that merely *parses* is the dangerous case: `cv_fc_score.py` would exit on the missing pids,
but only after you had already believed a `judge_qc` PASS on the batches that did land.)

Run:  BATCHDIR=results/se/cv_fc_full REPS=r1,r2 python -P scripts/fc_status.py
      BATCHDIR=... REPS=r1,r2 QUARANTINE=1 python -P scripts/fc_status.py     # move bad files aside
Exit code 0 when nothing is left to do, 1 otherwise -- so a driver loop can poll it.
"""
import os
import re
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / os.environ["BATCHDIR"]
REPS = [r.strip() for r in os.environ.get("REPS", "r1").split(",") if r.strip()]
QUAR = os.environ.get("QUARANTINE", "") not in ("", "0")
VERBOSE = os.environ.get("VERBOSE", "") not in ("", "0")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def ans_path(rep, i):
    return B / (f"ans_{i}.json" if rep == "r1" else f"ans_{rep}_{i}.json")


def check(batch_pids, p):
    """-> (ok, note). Never trust a file that exists; trust one that covers its batch."""
    if not p.exists():
        return False, "missing"
    if p.stat().st_size == 0:
        return False, "empty file"
    try:
        recs = json.loads(p.read_text(encoding="utf-8-sig"))     # subagents emit BOMs
    except Exception as e:
        return False, f"unparseable ({type(e).__name__}) -- truncated mid-write?"
    if not isinstance(recs, list):
        return False, "not a JSON array"
    got, bad = [], 0
    for r in recs:
        if not isinstance(r, dict):
            bad += 1
            continue
        c = str(r.get("choice", "")).strip().upper()
        if c not in ("A", "B"):
            bad += 1
            continue
        got.append(r.get("pid"))
    want = set(batch_pids)
    miss = want - set(got)
    extra = set(got) - want
    dup = len(got) - len(set(got))
    if bad:
        return False, f"{bad} verdicts not A/B"
    if miss:
        return False, f"missing {len(miss)}/{len(want)} pids (e.g. {sorted(miss)[:3]})"
    if extra:
        return False, f"{len(extra)} pids not in this batch"
    if dup:
        return False, f"{dup} duplicate pids"
    return True, f"{len(got)} items"


batches = sorted((f for f in B.glob("batch_*.json")), key=lambda f: int(f.stem.split("_")[1]))
if not batches:
    sys.exit(f"no batch_*.json in {B}")
pids = {int(f.stem.split("_")[1]): [t["pid"] for t in json.loads(f.read_text(encoding="utf-8"))]
        for f in batches}
total_items = sum(len(v) for v in pids.values())

todo, rotten, done = [], [], 0
for rep in REPS:
    for i in sorted(pids):
        p = ans_path(rep, i)
        ok, note = check(pids[i], p)
        if ok:
            done += 1
            if VERBOSE:
                print(f"  ok    {rep} batch_{i:<3d} {note}")
        else:
            todo.append((rep, i))
            if note != "missing":
                rotten.append((rep, i, p, note))

print(f"{B.name}: {len(batches)} batches x {len(REPS)} replicate(s) = {len(batches)*len(REPS)} judge calls, "
      f"{total_items} items each replicate")
print(f"  done {done}   todo {len(todo)}   of which rotten {len(rotten)}")

for rep, i, p, note in rotten:
    print(f"  ROTTEN  {rep} batch_{i}: {note}")
    if QUAR:
        dst = p.with_suffix(".json.bad")
        n = 1
        while dst.exists():
            dst = p.with_suffix(f".json.bad{n}"); n += 1
        p.rename(dst)
        print(f"          -> quarantined to {dst.name}")

if todo:
    print("\nTODO (replicate, batch):")
    for rep in REPS:
        xs = [i for r, i in todo if r == rep]
        if xs:
            print(f"  {rep}: {' '.join(map(str, xs))}   ({len(xs)} calls)")
    (B / "_todo.json").write_text(json.dumps([{"rep": r, "batch": i, "out": ans_path(r, i).name}
                                              for r, i in todo], indent=1), encoding="utf-8")
    print(f"\nwrote {(B/'_todo.json').relative_to(ROOT)}")
    sys.exit(1)

print("\nALL DONE — every batch of every replicate is complete and well-formed.")
