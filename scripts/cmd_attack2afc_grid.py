"""Aggregate all 2AFC score files (results[/{ds}]/_2afc_score_*.json) into one master table:
the identity-destruction ladder raw -> indiv -> shared, per (dataset, k, seed, model, nneg-match), for nneg & rneg.
Pure $0 reader. Run: python scripts/cmd_attack2afc_grid.py
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def fmt(cell):
    if not cell:
        return "   --       "
    a, ci = cell["acc"], cell["ci"]
    star = "*" if (ci[0] > 0.5 or ci[1] < 0.5) else " "
    return f"{a:.3f}{star}[{ci[0]:.2f},{ci[1]:.2f}]"


def main():
    files = list((ROOT / "results").glob("_2afc_score_*.json"))
    files += list((ROOT / "results").glob("*/_2afc_score_*.json"))
    rows = []
    for f in sorted(files):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = s.get("_meta", {})
        rows.append((m.get("ds", "?"), m.get("k", "?"), m.get("seed", "?"),
                     m.get("model", "?"), m.get("nneg_match", "?"), s))
    if not rows:
        print("no _2afc_score_*.json found — run cmd_attack2afc_score.py first.")
        return

    print("2AFC identity-destruction ladder  (acc; * = CI excludes 0.5; chance=0.5)")
    print("nneg = topic-controlled IDENTITY (headline) | rneg = topic-gameable (aux)\n")
    hdr = f"{'dataset':7s} {'k':>2s} {'s':>1s} {'model':20s} {'nneg~':6s} | {'neg':4s} | {'raw':14s} {'indiv':14s} {'shared':14s}"
    print(hdr); print("-" * len(hdr))
    for ds, k, seed, model, nm, s in sorted(rows, key=lambda r: (r[0], str(r[1]), str(r[2]), r[3], r[4])):
        for neg in ("nneg", "rneg"):
            cells = [s.get(f"{c}_{neg}") for c in ("raw", "indiv", "shared")]
            line = (f"{ds:7s} {str(k):>2s} {str(seed):>1s} {model:20s} {nm:6s} | {neg:4s} | "
                    + " ".join(fmt(c) for c in cells))
            # pooling effect on the topic-controlled identity metric: shared - indiv (negative = pooling de-ids)
            if neg == "nneg" and cells[1] and cells[2]:
                delta = cells[2]["acc"] - cells[1]["acc"]
                line += f"  pooling Δ(shared-indiv)={delta:+.3f}" + ("  (pooling de-ids)" if delta < 0 else "")
            print(line)
        print()


if __name__ == "__main__":
    main()
