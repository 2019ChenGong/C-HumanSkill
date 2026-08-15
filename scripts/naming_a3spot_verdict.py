"""A3-spot verdict (#155): does the second card-build partition reproduce the canonical one?

Pre-registration: `results/NAMING_A3SPOT_SEED2_PREREG.md` (frozen before export).
Run with no arguments:  python -P scripts/naming_a3spot_verdict.py

The two partitions have DISJOINT cluster sets, so there is no pairing structure — per prereg §5 we
compare two INDEPENDENT estimates and are forbidden from manufacturing a paired difference.

Estimators are imported from `scripts/naming_a12_verdict.py` verbatim (boot / multiseed /
cluster_auc / load_rows / load_auc_pairs) so the second partition is read on exactly the same ruler
as the canonical one — no re-implementation.

Frozen criteria (prereg §5):
  (1) PRIMARY   second-partition pooled AUC(S1) up95 < .5 + delta, delta = .10  -> certification holds
  (2) SECONDARY second-partition pooled depth-excess POINT estimate is positive (same sign as canon)
  (3) COMPAT    the two partitions' depth-excess CIs overlap
Verdict:
  (1) ok and (3) ok            -> "reproduces on the second partition"; do NOT escalate
  (1) fails                    -> escalate to the full multi-seed run; until then every certification
                                  claim must carry a "single partition" qualifier
  (1) ok but (3) fails         -> register as evidence of partition variance; escalate
"""
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import naming_a12_verdict as A12          # noqa: E402  (estimators reused verbatim)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DIRS = A12.DIRS                            # {"mad": "mad", "enron": "enron", "cv": "se"}
SPOT = "naming_seed2"
LEGACY_V6 = A12.LEGACY_V6                  # ("naming_v6", "naming_v6_more") + pool_ab increment
POOL_AB = "naming_pool_ab"
DELTA = 0.10
SEED2 = {"mad": 1, "enron": 0, "cv": 1}    # frozen in prereg §3
CANON = {"mad": 0, "enron": 1, "cv": 0}


def pooled_units(per_ds_maps):
    """[{unit: {row_id: val}}] per dataset -> list of per-(ds,cluster) value lists."""
    out = []
    for ds, m in per_ds_maps.items():
        for u, rows in m.items():
            if rows:
                out.append(list(rows.values()))
    return out


# ★ The canonical side must be pooled with `naming_pooled_depth.collect`'s rule — "same cluster,
#   more brackets", i.e. EXTEND the per-unit list across sub-dirs with NO row-id uniqueness check.
#   (A12's `merge_rows` asserts uniqueness, which is right for naming_v6+naming_v6_more — disjoint
#   bseeds — but wrong here: `naming_pool_ab` re-measures v6 at bseed 0 in-wave, so the row ids
#   legitimately collide. That in-wave repeat is exactly what turned the published pooled v6 from
#   +0.0412 (n=1284) into +0.0442 (n=1605).)
#
# ★★ And the second partition must NEVER be folded into those SUBDIRS: the unit key is `ds:G0`,
#   but cluster "G0" at seed 1 holds DIFFERENT people than "G0" at seed 0 — merging would silently
#   fuse two partitions into one "cluster". The two sides are pooled separately, by construction.
CANON_SUBDIRS = ("naming_v6", "naming_v6_more", "naming_pool_ab")


def _per_ds_canon_depth(ds):
    """-> {unit: [values]} for ONE dataset, canonical partition."""
    by_unit = {}
    for sub in CANON_SUBDIRS:
        p = ROOT / f"results/{DIRS[ds]}/{sub}/_naming_depth.json"
        if not p.exists():
            continue
        sec = json.loads(p.read_text(encoding="utf-8")).get("arms", {}).get("v6", {}).get("depth")
        if not sec:
            continue
        for u, v in sec["units"].items():
            if v:
                by_unit.setdefault(u, []).extend(v)
    return by_unit


def canon_depth_units():
    return [v for ds in DIRS for v in _per_ds_canon_depth(ds).values()]


def canon_auc_units():
    out = []
    for ds, d in DIRS.items():
        by_unit = {}
        for sub in CANON_SUBDIRS:
            p = ROOT / f"results/{d}/{sub}/_naming_auc.json"
            if not p.exists():
                continue
            sec = json.loads(p.read_text(encoding="utf-8")).get("arms", {}).get("v6", {}).get("S1")
            if not sec:
                continue
            for u, v in sec["units"].items():
                by_unit.setdefault(u, []).extend((float(s), int(l)) for s, l in v)
        for u, pairs in by_unit.items():
            a = A12.cluster_auc(pairs)
            if a is not None:
                out.append([a])
    return out


def strat_auc(per_ds_pairs):
    """Stratified pooled AUC: cluster-level AUCs averaged, bootstrapped over (ds,cluster)."""
    units = []
    for ds, m in per_ds_pairs.items():
        for u, pairs in m.items():
            a = A12.cluster_auc(pairs)
            if a is not None:
                units.append([a])
    return units


def main():
    # ---- second partition (this experiment) -------------------------------------------------
    spot_depth = {ds: A12.load_rows(ds, SPOT, "v6", "depth") for ds in DIRS}
    spot_auc = {ds: A12.load_auc_pairs(ds, SPOT, "v6") for ds in DIRS}
    # ---- canonical partition (already published) -- pooled with the canonical "extend" rule ----
    canon_depth = {ds: {u: dict(enumerate(v)) for u, v in
                        _per_ds_canon_depth(ds).items()} for ds in DIRS}

    res = {"experiment": "A3-spot (#155) second build partition vs canonical",
           "prereg": "results/NAMING_A3SPOT_SEED2_PREREG.md",
           "partitions": {"canonical": CANON, "second": SEED2},
           "delta": DELTA, "boot": {"n": A12.BOOT_N, "seeds": list(A12.SEEDS), "unit": "ds:cluster"}}

    print("=" * 78)
    print("A3-spot 判决:第二建卡分区 vs 正典分区(判据见预注册 §5,跑前冻结)")
    print("=" * 78)

    # ---------- depth excess, per dataset then pooled ----------
    print("\n--- 深度超额(v6),逐家 ---")
    print(f"{'数据集':8s} {'分区':>10s} {'est':>9s} {'CI':>22s} {'簇':>4s} {'行':>6s}")
    per_ds = {}
    for ds in DIRS:
        for tag, src, seedmap in (("canonical", canon_depth, CANON), ("second", spot_depth, SEED2)):
            u = [list(v.values()) for v in src[ds].values() if v]
            r = A12.multiseed(u, f"{ds}/{tag}")
            per_ds[f"{ds}/{tag}"] = r
            print(f"{ds:8s} {tag+' s'+str(seedmap[ds]):>10s} {r['est']:+9.4f} "
                  f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] {r['n_units']:4d} {r['n_rows']:6d}")
    res["depth_per_dataset"] = per_ds

    d_canon = A12.multiseed(canon_depth_units(), "pooled/canonical")
    d_spot = A12.multiseed(pooled_units(spot_depth), "pooled/second")
    res["depth_pooled"] = {"canonical": d_canon, "second": d_spot}
    print("\n--- 深度超额(v6),合并 ---")
    for r in (d_canon, d_spot):
        print(f"  {r['label']:20s} {r['est']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  "
              f"簇={r['n_units']} 行={r['n_rows']}  跨种子同号={r['seeds_agree']}  判定={r['verdict']}")

    # ---------- stratified AUC ----------
    print("\n--- 分层合并 AUC(S1)---")
    a_canon = A12.multiseed(canon_auc_units(), "pooled/canonical")
    a_spot = A12.multiseed(strat_auc(spot_auc), "pooled/second")
    res["auc_pooled"] = {"canonical": a_canon, "second": a_spot}
    for r in (a_canon, a_spot):
        print(f"  {r['label']:20s} AUC {r['est']:.4f} [{r['ci'][0]:.4f},{r['ci'][1]:.4f}]  簇={r['n_units']}")

    # ---------- frozen criteria ----------
    up95 = a_spot["ci"][1]
    c1 = up95 < 0.5 + DELTA
    c2 = d_spot["est"] > 0
    lo_a, hi_a = d_canon["ci"]
    lo_b, hi_b = d_spot["ci"]
    c3 = not (hi_b < lo_a or hi_a < lo_b)

    if not c1:
        verdict = "ESCALATE — 认证失守 ⇒ 升级完整多种子;补完前所有认证主张须加「单分区」限定"
    elif not c3:
        verdict = "ESCALATE — 认证守住但两分区 CI 不重叠 ⇒ 登记为分区方差实证,升级"
    else:
        verdict = "REPRODUCES — 写「第二分区上复现」,收进 limitations,不升级"

    res["criteria"] = {"C1_cert_holds": bool(c1), "C1_up95": round(up95, 4),
                       "C1_bound": 0.5 + DELTA,
                       "C2_sign_positive": bool(c2), "C2_est": d_spot["est"],
                       "C3_ci_overlap": bool(c3),
                       "canon_ci": d_canon["ci"], "second_ci": d_spot["ci"]}
    res["verdict"] = verdict

    print("\n" + "=" * 78)
    print("冻结判据(预注册 §5)")
    print(f"  ① 主判据  第二分区 AUC up95 = {up95:.4f}  <  {0.5+DELTA:.2f} ?   -> {'✔ 认证守住' if c1 else '✘ 失守'}")
    print(f"  ② 次判据  第二分区深度超额点估计 = {d_spot['est']:+.4f}  为正 ?   -> {'✔' if c2 else '✘(仅方向性,不单独当结论)'}")
    print(f"  ③ 相容    正典 [{lo_a:+.4f},{hi_a:+.4f}] vs 第二 [{lo_b:+.4f},{hi_b:+.4f}] 重叠 ? -> {'✔' if c3 else '✘'}")
    print(f"\n判决: {verdict}")
    print("=" * 78)

    out = ROOT / "results/_A3SPOT_seed2_all.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
