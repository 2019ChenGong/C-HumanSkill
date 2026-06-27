"""Consolidate per-trial dump files into one batch per group so a single strong-attacker pass can score the whole group."""
import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict

DATASET = os.environ.get("DATASET", "enron")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
if DATASET != "enron":
    RES = RES / DATASET
if os.environ.get("RESDIR"):
    RES = ROOT / os.environ["RESDIR"]              # cross-model run isolation (match cmd_openworld)
MODE = os.environ.get("MODE", "gate")
MARK = "\n\n@@@@@ {tag} @@@@@\n"


def main():
    if MODE == "gate":
        pat = re.compile(r"_cmdgate_(k\d+)_(s\d+)_(card|raw|indiv)_T(\d+)$")
        groups = defaultdict(list)
        for f in RES.glob("_cmdgate_k*_s*_*_T*.txt"):
            m = pat.match(f.stem)
            if m:
                groups[(m.group(1), m.group(2), m.group(3))].append((int(m.group(4)), f))
        manifest = []
        for (k, s, cond), files in sorted(groups.items()):
            files.sort()
            parts = [f"# BATCH {k} {s} {cond}: {len(files)} INDEPENDENT re-identification trials. "
                     "For EACH trial pick exactly ONE candidate number.\n"]
            for idx, f in files:
                parts.append(MARK.format(tag=f"T{idx:03d}") + f.read_text(encoding="utf-8"))
            bf = RES / f"_batch_cmdgate_{k}_{s}_{cond}.txt"
            bf.write_text("".join(parts), encoding="utf-8")
            manifest.append({"batch": bf.name, "picks": f"_picks_cmdgate_{k}_{s}_{cond}.json",
                             "n": len(files), "kind": "gate-pick"})
        (RES / "_batch_manifest_gate.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"gate: {len(manifest)} batches -> results/_batch_cmdgate_*.txt  (manifest _batch_manifest_gate.json)")
        for m in manifest:
            print(f"  {m['batch']}  ({m['n']} trials) -> {m['picks']}")
    else:
        # ORDERING INVARIANT: longest/most-specific channel token first so no token is a prefix of another
        # (staab_r1 before staab; tpar_t\d+ matches any temperature incl _t10 without _t1 eating it). Else files drop silently.
        pat = re.compile(r"_ow_(k\d+)_(staab_g55_r1|tpar_t\d+|petre_k\d+|staab_r1|staab|presidio|shared|raw|indiv)_(s\d+)_T(\d+)$")
        groups = defaultdict(list)
        for f in RES.glob("_ow_k*_*_s*_T*.txt"):
            m = pat.match(f.stem)
            if m:
                groups[(m.group(1), m.group(2), m.group(3))].append((int(m.group(4)), f))
        manifest = []
        for (k, chan, s), files in sorted(groups.items()):
            files.sort()
            parts = [f"# BATCH {k} {chan} {s}: {len(files)} INDEPENDENT membership-scoring trials. "
                     "For EACH trial output a 0-100 score for EVERY candidate.\n"]
            for idx, f in files:
                parts.append(MARK.format(tag=f"T{idx:03d}") + f.read_text(encoding="utf-8"))
            bf = RES / f"_batch_ow_{k}_{chan}_{s}.txt"
            bf.write_text("".join(parts), encoding="utf-8")
            manifest.append({"batch": bf.name, "picks": f"_picks_ow_{k}_{chan}_{s}.json",
                             "n": len(files), "kind": "ow-score"})
        (RES / "_batch_manifest_ow.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print(f"ow: {len(manifest)} batches -> results/_batch_ow_*.txt  (manifest _batch_manifest_ow.json)")
        for m in manifest:
            print(f"  {m['batch']}  ({m['n']} trials) -> {m['picks']}")


if __name__ == "__main__":
    main()
