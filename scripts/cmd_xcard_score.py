"""Score the FREE-subagent cross-card linkage answers (P8/H3) -> AUC / clustered CI / verbatim-free / ctrl.

Reads results/{ds}/xcard_free_{METHOD}_k{K}/{meta.json, ans_*.json} and writes _xcard_{METHOD}_result_k{K}.json
with the SAME fields as the paid cmd_xcard_link.py run(), so concat's linkage AUC is directly comparable to CMD's.

  DATASET=mad K=4 METHOD=concat python scripts/cmd_xcard_score.py
"""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DATASET", "mad")
os.environ.setdefault("METHOD", "concat")
os.environ.setdefault("MODE", "score")
import cmd_xcard_link as XL  # noqa: E402  (reuse _auc / _boot_clustered / _boot)

DISJOINT = bool(int(os.environ.get("DISJOINT", "0")))
TAG = os.environ.get("TAG", "")
PKG = XL.RES / f"xcard_free_{XL.METHOD}{'_dj' if DISJOINT else ''}_k{XL.K}{('_' + TAG) if TAG else ''}"


def _p(ans):
    """P(share a contributor) from a worker answer dict {verdict, conf} or a raw 'YES 65' line."""
    v = ans.get("verdict") if isinstance(ans, dict) else None
    conf = ans.get("conf") if isinstance(ans, dict) else None
    if v is None:                                   # tolerate a raw string answer
        s = ans if isinstance(ans, str) else json.dumps(ans)
        v = "YES" if re.search(r"\bYES\b", s, re.I) else "NO"
        m = re.findall(r"\d{2,3}", s); conf = float(m[0]) if m else 50.0
    yes = bool(re.search(r"\bYES\b", str(v), re.I))
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = 50.0
    conf = min(100.0, max(50.0, conf))
    return conf / 100.0 if yes else 1.0 - conf / 100.0


def main():
    meta = json.loads((PKG / "meta.json").read_text(encoding="utf-8"))
    P = {}
    for f in sorted(PKG.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", f.stem):     # stray-guard
            continue
        for rec in json.loads(f.read_text(encoding="utf-8")):
            if isinstance(rec, dict) and "pid" in rec:
                P[rec["pid"]] = _p(rec)
    have = [pid for pid in meta if pid in P]
    miss = [pid for pid in meta if pid not in P]
    print(f"answered {len(have)}/{len(meta)} pairs" + (f"  MISSING {len(miss)}: {miss[:12]}..." if miss else ""))

    pos = {pid: P[pid] for pid in have if meta[pid]["kind"] == "pos"}
    neg = {pid: P[pid] for pid in have if meta[pid]["kind"] == "neg"}
    ctrl = [P[pid] for pid in have if meta[pid]["kind"] == "ctrl"]

    pv = list(pos.values()); nv = list(neg.values())
    auc = XL._auc(pv, nv)
    # clustered CI over the POS shared-member (each matched pair's pos+neg carry the same cluster)
    records = [(meta[pid]["cluster"], 1, P[pid]) for pid in pos] + [(meta[pid]["cluster"], 0, P[pid]) for pid in neg]
    point, clo, chi = XL._boot_clustered(records)

    # verbatim-free subset: matched indices where BOTH pos and neg share 0 six-grams
    mis = sorted(set(meta[pid]["mi"] for pid in have if meta[pid]["mi"] >= 0))
    vf_pos, vf_neg = [], []
    for mi in mis:
        pk, nk = f"pos{mi}", f"neg{mi}"
        if pk in P and nk in P and meta[pk]["n6"] == 0 and meta[nk]["n6"] == 0:
            vf_pos.append(P[pk]); vf_neg.append(P[nk])
    vf = None
    if vf_pos and vf_neg and len(set([1] * len(vf_pos) + [0] * len(vf_neg))) == 2:
        va = XL._auc(vf_pos, vf_neg); vlo, vhi = XL._boot(vf_pos, vf_neg)
        vf = {"auc": round(va, 4), "ci": [round(vlo, 4), round(vhi, 4)], "n": len(vf_pos)}

    # $0 rare-verbatim baseline: n6 count as the score (pos n6 vs neg n6)
    r6p = [meta[pid]["n6"] for pid in pos]; r6n = [meta[pid]["n6"] for pid in neg]
    rare6 = round(XL._auc(r6p, r6n), 4) if (r6p and r6n) else None

    out = {"method": XL.METHOD, "instrument": "FREE claude-code sonnet-4.6 subagents (2AFC linkage, YES/NO+conf)",
           "auc": round(auc, 4), "ci_clustered": [round(clo, 4), round(chi, 4)],
           "rare6_baseline": rare6, "n_pos": len(pv), "n_neg": len(nv),
           "auc_verbatim_free": (vf["auc"] if vf else None), "vf_ci": (vf["ci"] if vf else None),
           "n_vf": (vf["n"] if vf else 0),
           "ctrl_mean": (round(float(np.mean(ctrl)), 4) if ctrl else None), "n_ctrl": len(ctrl),
           "k": XL.K, "coverage": [len(have), len(meta)]}
    outname = f"_xcard_{XL.METHOD}_result_free{'_dj' if DISJOINT else ''}_k{XL.K}{('_' + TAG) if TAG else ''}.json"   # "_free"/"_dj"/TAG never clobbers the PAID run()
    (XL.RES / outname).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\n-> results/{'mad/' if XL.DS!='enron' else ''}{outname}")


if __name__ == "__main__":
    main()
