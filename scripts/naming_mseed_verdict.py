"""MSEED verdict (#158): the complete three-card-build-partition read of the naming ruler.

Pre-registration: `results/NAMING_MSEED_S2_PREREG.md` (frozen before the s2 export).
Run with no arguments:  python -P scripts/naming_mseed_verdict.py

Estimators (`boot` / `multiseed` / `cluster_auc` / `load_rows` / `load_auc_pairs`) are imported from
`scripts/naming_a12_verdict.py` verbatim, so all three partitions are read on exactly the same ruler
as the published canonical numbers — nothing re-implemented.

★ The three partitions are RE-PARTITIONS OF THE SAME 321 PEOPLE. Two consequences, both frozen in
  prereg §3 and enforced here:
  1. They must never be pooled as 3x39 independent clusters (each person would be counted three
     times and the CI would be under-estimated). The MAIN report is three separate estimates plus a
     descriptive cross-partition envelope.
  2. The unit key is `ds:G0`, and cluster "G0" holds DIFFERENT PEOPLE in each partition — so the
     per-cluster maps are never merged across partitions either.

Frozen criteria (prereg §4):
  (1) CERT      per-partition pooled AUC(S1) up95 < .5 + delta, delta = .10 (report .05 too).
                3/3 -> "certification is independent of the build partition"; any miss -> report by
                name "N of 3 failed" and re-qualify the certification claim.
  (2) DETECT    per-partition pooled depth-excess CI excludes 0. Report how many of 3, honestly.
                3/3 -> may write "robust detection"; 1/3 or 2/3 -> "partition-dependent".
                Reporting only the favourable partition is forbidden.
  (3) MARGINAL  per-person mean across the three partitions, then bootstrap over the 321 people.
                Descriptive only — no pass/fail — and must appear together with the main report.
                Carries the approximation flag: "inter-person correlation after cross-partition
                averaging is assumed negligible".
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import naming_a12_verdict as A12          # noqa: E402  (estimators reused verbatim)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DIRS = A12.DIRS                            # {"enron": "enron", "mad": "mad", "cv": "se"}
DELTAS = (0.10, 0.05)
ARM = "v6"

# ★ The canonical side is pooled with `naming_pooled_depth.collect`'s rule — "same cluster, more
#   brackets", i.e. EXTEND across sub-dirs with NO row-id uniqueness check. `naming_pool_ab`
#   legitimately re-measures v6 at bseed 0 in-wave, so row ids collide by construction; that in-wave
#   repeat is what turned the published pooled v6 from +0.0412 (n=1284) into +0.0442 (n=1605).
PARTITIONS = [
    ("canonical", {"mad": 0, "enron": 1, "cv": 0},
     ("naming_v6", "naming_v6_more", "naming_pool_ab")),
    ("second",    {"mad": 1, "enron": 0, "cv": 1}, ("naming_seed2",)),
    ("third",     {"mad": 2, "enron": 2, "cv": 2}, ("naming_seed3",)),
]


def depth_by_unit(ds, subs):
    """-> {unit: [(row_id, value), ...]} for ONE dataset in ONE partition (extend across sub-dirs)."""
    by_unit = {}
    for sub in subs:
        p = ROOT / f"results/{DIRS[ds]}/{sub}/_naming_depth.json"
        if not p.exists():
            continue
        sec = json.loads(p.read_text(encoding="utf-8")).get("arms", {}).get(ARM, {}).get("depth")
        if not sec:
            continue
        for u, vals in sec["units"].items():
            ids = sec["row_ids"][u]
            assert len(ids) == len(vals), f"{p} {u}: row_ids/units length mismatch"
            by_unit.setdefault(u, []).extend(zip(ids, vals))
    return by_unit


def auc_by_unit(ds, subs):
    """-> {unit: [(score, label), ...]} for ONE dataset in ONE partition."""
    by_unit = {}
    for sub in subs:
        p = ROOT / f"results/{DIRS[ds]}/{sub}/_naming_auc.json"
        if not p.exists():
            continue
        sec = json.loads(p.read_text(encoding="utf-8")).get("arms", {}).get(ARM, {}).get("S1")
        if not sec:
            continue
        for u, v in sec["units"].items():
            by_unit.setdefault(u, []).extend((float(s), int(l)) for s, l in v)
    return by_unit


def cluster_units(by_unit_per_ds):
    """{ds: {unit: [(rid, val)]}} -> [[val, ...], ...] one list per (ds, cluster)."""
    return [[v for _, v in rows] for m in by_unit_per_ds.values() for rows in m.values() if rows]


def auc_units(by_unit_per_ds):
    """{ds: {unit: [(score,label)]}} -> [[cluster_auc], ...] one singleton list per (ds, cluster)."""
    out = []
    for m in by_unit_per_ds.values():
        for pairs in m.values():
            a = A12.cluster_auc(pairs)
            if a is not None:
                out.append([a])
    return out


def person_means(by_unit_per_ds):
    """{ds: {unit: [(rid, val)]}} -> {(ds, person): mean of that person's rows in this partition}.

    row id is "{member}#{bseed}"; the person is the part before the LAST '#'. Keyed with the dataset
    so member ids can never collide across the three corpora.
    """
    acc = {}
    for ds, m in by_unit_per_ds.items():
        for rows in m.values():
            for rid, v in rows:
                acc.setdefault((ds, rid.rsplit("#", 1)[0]), []).append(v)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main():
    res = {"experiment": "MSEED (#158) — three card-build partitions, naming ruler, arm v6",
           "prereg": "results/NAMING_MSEED_S2_PREREG.md",
           "partitions": {name: seeds for name, seeds, _ in PARTITIONS},
           "deltas": list(DELTAS),
           "boot": {"n": A12.BOOT_N, "seeds": list(A12.SEEDS), "unit": "ds:cluster"}}

    print("=" * 84)
    print("MSEED 判决:点名尺 v6 三个建卡分区(判据见 NAMING_MSEED_S2_PREREG.md §4,跑前冻结)")
    print("=" * 84)

    depth = {name: {ds: depth_by_unit(ds, subs) for ds in DIRS} for name, _, subs in PARTITIONS}
    aucs = {name: {ds: auc_by_unit(ds, subs) for ds in DIRS} for name, _, subs in PARTITIONS}

    # ★ HARD GATE — a MISSING sidecar must never be rendered as a FAILED criterion. Without this,
    #   an un-run partition silently prints "certification lost" / "detection absent", which is the
    #   most dangerous possible failure mode for this script.
    holes = [f"{name}/{ds}" for name, _, _ in PARTITIONS for ds in DIRS
             if not depth[name][ds] or not aucs[name][ds]]
    if holes:
        print("\n★ ABORT — 以下 (分区,数据集) 缺 `_naming_depth.json` / `_naming_auc.json`:")
        for h in holes:
            print(f"    {h}")
        print("  判据未计算(缺数 ≠ 失守)。先跑 naming_depth.py / naming_auc.py 补齐再来。")
        sys.exit(2)

    # ---------------- 主报 A: depth excess, per dataset ----------------
    print("\n--- 深度超额(v6),逐家 × 逐分区 ---")
    print(f"{'数据集':6s} {'分区':>12s} {'est':>9s} {'CI':>21s} {'簇':>4s} {'行':>6s}")
    res["depth_per_dataset"] = {}
    for ds in DIRS:
        for name, seeds, _ in PARTITIONS:
            u = [[v for _, v in rows] for rows in depth[name][ds].values() if rows]
            r = A12.multiseed(u, f"{ds}/{name}")
            res["depth_per_dataset"][f"{ds}/{name}"] = r
            print(f"{ds:6s} {name + ' s' + str(seeds[ds]):>12s} {r['est']:+9.4f} "
                  f"[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] {r['n_units']:4d} {r['n_rows']:6d}")
        print()

    # ---------------- 主报 B: depth excess pooled, per partition ----------------
    print("--- 深度超额(v6),每分区合并 ★ 判据 ② ---")
    res["depth_pooled"], det_pass = {}, []
    for name, _, _ in PARTITIONS:
        r = A12.multiseed(cluster_units(depth[name]), name)
        res["depth_pooled"][name] = r
        excl0 = r["ci"][0] > 0 or r["ci"][1] < 0
        det_pass.append(excl0)
        print(f"  {name:10s} {r['est']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  "
              f"簇={r['n_units']} 行={r['n_rows']}  跨种子同号={r['seeds_agree']}  "
              f"排除0={'✔' if excl0 else '✘'}")
    d_lo = min(res["depth_pooled"][n]["ci"][0] for n, _, _ in PARTITIONS)
    d_hi = max(res["depth_pooled"][n]["ci"][1] for n, _, _ in PARTITIONS)
    res["depth_envelope"] = [round(d_lo, 4), round(d_hi, 4)]
    print(f"  {'包络':10s} [{d_lo:+.4f},{d_hi:+.4f}]   (描述性:三分区不独立,不是严格覆盖的 CI)")

    # ---------------- 主报 C: stratified pooled AUC, per partition ----------------
    print("\n--- 分层合并 AUC(S1),每分区 ★ 判据 ① ---")
    res["auc_pooled"], cert = {}, {f"{d:.2f}": [] for d in DELTAS}
    for name, _, _ in PARTITIONS:
        r = A12.multiseed(auc_units(aucs[name]), name)
        res["auc_pooled"][name] = r
        up95 = r["ci"][1]
        flags = []
        for d in DELTAS:
            ok = up95 < 0.5 + d
            cert[f"{d:.2f}"].append(ok)
            flags.append(f"δ={d:.2f}:{'✔' if ok else '✘失守'}")
        r["cert"] = {f"{d:.2f}": bool(up95 < 0.5 + d) for d in DELTAS}
        print(f"  {name:10s} AUC {r['est']:.4f} [{r['ci'][0]:.4f},{r['ci'][1]:.4f}]  "
              f"簇={r['n_units']}  up95={up95:.4f}   " + "  ".join(flags))
    a_hi = max(res["auc_pooled"][n]["ci"][1] for n, _, _ in PARTITIONS)
    res["auc_up95_worst"] = round(a_hi, 4)
    print(f"  {'最差上界':8s} up95 = {a_hi:.4f}")

    # ---------------- 次报: per-person marginalization ----------------
    print("\n--- ③ 边际化(次报,描述性):逐人跨三分区平均 → 按人自助 ---")
    pm = {name: person_means(depth[name]) for name, _, _ in PARTITIONS}
    common = set.intersection(*(set(m) for m in pm.values()))
    allp = set.union(*(set(m) for m in pm.values()))
    per_person = {p: sum(pm[n][p] for n, _, _ in PARTITIONS) / len(PARTITIONS) for p in common}
    r = A12.multiseed([[v] for v in per_person.values()], "per-person (3-partition mean)")
    res["marginal_per_person"] = r
    res["marginal_n_people"] = len(per_person)
    res["marginal_people_dropped"] = sorted(f"{d}:{p}" for d, p in (allp - common))
    print(f"  n={len(per_person)} 人(三分区都出现的交集;并集 {len(allp)})   "
          f"{r['est']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]  排除0="
          f"{'✔' if (r['ci'][0] > 0 or r['ci'][1] < 0) else '✘'}")
    if allp - common:
        print(f"  ! {len(allp - common)} 人未在全部三分区出现,已剔除:"
              f"{res['marginal_people_dropped'][:6]}")
    print("  ★ 假设标注:逐人跨分区平均后「人际相关可忽略」——这是近似,n=3 个分区仍估不出分区方差。")

    # ---------------- frozen verdict ----------------
    n_cert10, n_cert05 = sum(cert["0.10"]), sum(cert["0.05"])
    n_det = sum(det_pass)
    K = len(PARTITIONS)
    if n_cert10 == K:
        cert_verdict = "认证与建卡分区无关 —— 去掉一切「单分区」限定"
    else:
        names = [n for (n, _, _), ok in zip(PARTITIONS, cert["0.10"]) if not ok]
        cert_verdict = f"★ 认证在 {K - n_cert10}/{K} 个分区失守({', '.join(names)})—— 认证主张退回加限定"
    if n_det == K:
        det_verdict = "稳健检出 —— 三分区全部排除 0"
    elif n_det == 0:
        det_verdict = "三分区全部含 0 —— 不得声称检出"
    else:
        names = [n for (n, _, _), ok in zip(PARTITIONS, det_pass) if ok]
        det_verdict = (f"★ 分区依赖 —— 只有 {n_det}/{K} 个分区排除 0({', '.join(names)});"
                       f"禁止只报有利的那个")

    res["criteria"] = {
        "C1_cert_pass_delta10": f"{n_cert10}/{K}", "C1_cert_pass_delta05": f"{n_cert05}/{K}",
        "C1_per_partition": {n: res["auc_pooled"][n]["cert"] for n, _, _ in PARTITIONS},
        "C2_detect_excludes_zero": f"{n_det}/{K}",
        "C2_per_partition": {n: bool(ok) for (n, _, _), ok in zip(PARTITIONS, det_pass)},
        "C3_marginal_descriptive_only": True}
    res["verdict"] = {"certification": cert_verdict, "detection": det_verdict}

    print("\n" + "=" * 84)
    print("冻结判据(预注册 §4)")
    print(f"  ① 认证  δ=.10 通过 {n_cert10}/{K} 个分区(δ=.05 通过 {n_cert05}/{K})")
    print(f"          -> {cert_verdict}")
    print(f"  ② 检出  深度超额 CI 排除 0 的分区数 = {n_det}/{K}")
    print(f"          -> {det_verdict}")
    print(f"  ③ 边际化 {r['est']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}](n={len(per_person)} 人)"
          f" —— 描述性,不设通过/失败")
    print("=" * 84)

    out = ROOT / "results/_MSEED_3partition_all.json"
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    try:
        print(f"\n-> {out.relative_to(ROOT)}")
    except ValueError:
        print(f"\n-> {out}")


if __name__ == "__main__":
    main()
