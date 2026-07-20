"""R7 (#131) E2 — paired Δ(ne−v6) cross-release linkage analysis (MAD k8).

Loads the two same-wave packs (identical id pairs, identical DISJOINT batch shapes, batch_i pairs
dispatched in the same wave) and computes:
  - per-channel absolute AUC + shared-member-clustered CI (same stats as cmd_xcard_score)
  - PRIMARY: paired Δ(ne−v6) AUC — one bootstrap over shared-member clusters; each draw evaluates
    BOTH channels on the identical resampled row multiset, Δ per draw, percentile CI.
    Guardrail (review MINOR-3): Δ always runs on the FULL matched set; verbatim-free never filters it.
  - ctrl positive controls per channel: mean + UNclustered bootstrap CI (review MINOR-2); ctrl is
    excluded from Δ by design.
  - rare6 baseline per channel from each pack's own census n6.

Run: python -P scripts/r7_linkage_paired.py   [NBOOT=5000]
     (imports cmd_xcard_link only for pure stat helpers; no METHOD-dependent state is used)
Out: results/mad/r7_linkage_paired.json
"""
import os
import re
import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DATASET", "mad"); os.environ.setdefault("METHOD", "cmd_neutral")
os.environ.setdefault("MODE", "export"); os.environ.setdefault("GROUP", "random")
import cmd_xcard_link as XL  # noqa: E402  (pure helpers: _auc/_boot_clustered)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NBOOT = int(os.environ.get("NBOOT", "5000"))
PACKS = {"ne": ROOT / "results/mad/xcard_free_cmd_neutral_dj_k8_r7",
         "v6": ROOT / "results/mad/xcard_free_cmd_v6min_dj_k8_r7"}


def _p(ans):
    v = ans.get("verdict") if isinstance(ans, dict) else None
    conf = ans.get("conf") if isinstance(ans, dict) else None
    if v is None:
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


def load_pack(d):
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    P = {}
    for f in sorted(d.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", f.stem):
            continue
        for rec in json.loads(f.read_text(encoding="utf-8-sig")):
            if isinstance(rec, dict) and "pid" in rec:
                P[rec["pid"]] = _p(rec)
    miss = [pid for pid in meta if pid not in P]
    assert not miss, f"{d.name}: MISSING answers {len(miss)}: {miss[:10]}"
    return meta, P


def main():
    packs = {ch: load_pack(d) for ch, d in PACKS.items()}
    m_ne, m_v6 = packs["ne"][0], packs["v6"][0]
    assert set(m_ne) == set(m_v6), "pid sets differ between packs"
    assert all(m_ne[p]["kind"] == m_v6[p]["kind"] and m_ne[p]["cluster"] == m_v6[p]["cluster"] for p in m_ne), \
        "kind/cluster mismatch between packs"
    out = {"nboot": NBOOT, "chan": {}}
    print(f"R7 E2 paired linkage  ({len(m_ne)} pids/channel, NBOOT={NBOOT})")
    for ch, (meta, P) in packs.items():
        recs = [(meta[p]["cluster"], 1, P[p]) for p in meta if meta[p]["kind"] == "pos"] \
             + [(meta[p]["cluster"], 0, P[p]) for p in meta if meta[p]["kind"] == "neg"]
        point, lo, hi = XL._boot_clustered(recs, nb=NBOOT)
        r6p = [meta[p]["n6"] for p in meta if meta[p]["kind"] == "pos"]
        r6n = [meta[p]["n6"] for p in meta if meta[p]["kind"] == "neg"]
        rare6 = XL._auc(r6p, r6n) if (any(r6p) or any(r6n)) else 0.5
        # verbatim-free (per-channel descriptive ONLY; never used for the paired Δ)
        mis = sorted(set(meta[p]["mi"] for p in meta if meta[p]["mi"] >= 0))
        vfp = [P[f"pos{i}"] for i in mis if meta[f"pos{i}"]["n6"] == 0 and meta[f"neg{i}"]["n6"] == 0]
        vfn = [P[f"neg{i}"] for i in mis if meta[f"pos{i}"]["n6"] == 0 and meta[f"neg{i}"]["n6"] == 0]
        vf = round(XL._auc(vfp, vfn), 4) if vfp else None
        ctrl = [P[p] for p in meta if meta[p]["kind"] == "ctrl"]
        rng = np.random.default_rng(0)
        cb = [float(np.mean(rng.choice(ctrl, len(ctrl), replace=True))) for _ in range(NBOOT)]
        out["chan"][ch] = {"auc": round(point, 4), "ci_clustered": [round(lo, 4), round(hi, 4)],
                           "rare6_baseline": round(rare6, 4), "vf_auc": vf, "n_vf": len(vfp),
                           "ctrl_mean": round(float(np.mean(ctrl)), 4),
                           "ctrl_ci_unclustered": [round(float(np.percentile(cb, 2.5)), 4),
                                                   round(float(np.percentile(cb, 97.5)), 4)],
                           "n_pos": len(r6p), "n_ctrl": len(ctrl)}
        c = out["chan"][ch]
        print(f"  {ch}: AUC={c['auc']} {c['ci_clustered']}  rare6={c['rare6_baseline']}  "
              f"vf={c['vf_auc']} (n={c['n_vf']})  ctrl={c['ctrl_mean']} {c['ctrl_ci_unclustered']}")

    # PRIMARY paired Δ: one cluster bootstrap, both channels evaluated on the same row multiset
    meta = m_ne
    by = defaultdict(list)                    # cluster -> [(y, pid)]
    for p in meta:
        if meta[p]["kind"] in ("pos", "neg"):
            by[meta[p]["cluster"]].append((1 if meta[p]["kind"] == "pos" else 0, p))
    keys = sorted(by)
    Pne, Pv6 = packs["ne"][1], packs["v6"][1]

    def auc_rows(rows, P):
        return XL._auc([P[p] for y, p in rows if y == 1], [P[p] for y, p in rows if y == 0])

    all_rows = [r for k in keys for r in by[k]]
    d_point = auc_rows(all_rows, Pne) - auc_rows(all_rows, Pv6)
    rng = np.random.default_rng(0)
    ds = []
    for _ in range(NBOOT):
        rows = [r for k in rng.choice(keys, len(keys), replace=True) for r in by[k]]
        if len({y for y, _ in rows}) < 2:
            continue
        ds.append(auc_rows(rows, Pne) - auc_rows(rows, Pv6))
    dlo, dhi = float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
    out["paired_delta_ne_minus_v6"] = {"point": round(d_point, 4), "ci": [round(dlo, 4), round(dhi, 4)],
                                       "n_clusters": len(keys),
                                       "read": "v6 LESS linkable (SIG)" if dlo > 0 else
                                               ("v6 MORE linkable (SIG)" if dhi < 0 else "no paired difference")}
    print(f"  PRIMARY paired Δ(ne−v6) AUC = {d_point:+.4f} [{dlo:+.4f},{dhi:+.4f}]  "
          f"({out['paired_delta_ne_minus_v6']['read']}, {len(keys)} shared-member clusters)")
    op = ROOT / "results/mad/r7_linkage_paired.json"
    op.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"saved -> {op.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
