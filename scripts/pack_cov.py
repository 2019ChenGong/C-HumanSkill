"""Coverage gate for the 2AFC / forced-choice / linkage pack families.

`scripts/naming_cov.py` gates the naming tournament. Nothing gated the other families, which is
where most published numbers live — this closes that gap. Audit 2026-08-04 found the historical
data clean, so this script exists to keep it that way, not to fix anything.

  python -P scripts/pack_cov.py            # gate every pack directory under results/
  python -P scripts/pack_cov.py results/mad/fc     # gate one directory

## Why coverage < 100% is only a WARN here

The naming waves were run to completion by design, so `naming_cov.py` rightly calls anything below
1.0 BAD. The 2AFC/FC packs have a handful of historical single-digit dropouts (9 directories at
95.4–99.9%). A gate that painted those permanently red would train everyone to ignore it. What
actually biases a comparison is not the dropout rate but its CONCENTRATION: losing 5% of every arm
costs sample size, losing 5% of one arm moves the estimate. So dropout is a WARN and arm-skewed
dropout is a BAD.

## BAD (something that can change a conclusion)

  ORPHAN     an answered pid is absent from meta.json  -> meta was overwritten by a later export,
             so the answers on disk no longer describe the questions on disk
  CONFLICT   two answer generations answer the SAME pid DIFFERENTLY -> whichever scorer merges them
             gets a number that depends on filename sort order. This is the decidable form of the
             "mixed generations" hazard: two generations that merely coexist without disagreeing
             cannot change any estimate, and which scorer reads a directory is not statically
             knowable anyway (most take BATCHDIR from the environment), so coexistence alone is a
             WARN and only actual disagreement is a BAD.
  UNREADABLE an ans_*.json that is empty or not valid JSON
  SKEWED     one arm's dropout rate exceeds SKEW_MULT x the directory's overall rate (and that arm
             lost at least MIN_SKEW_N items) -> the surviving comparison is biased, not just smaller

## WARN (costs sample size, cannot flip a sign on its own)

  DROPOUT    coverage < 100% but spread evenly across arms
  UNRUN      coverage < UNRUN_FRAC — an exported pack that was never dispatched is not a partial
             measurement, it is a non-measurement; reporting it as a "83% dropout" would be wrong
  EXTRA-GEN  a non-standard answer generation (ans_qwen_pilot, ans_r0, ...) sits alongside the
             numbered one. Coverage is NOT computed for these: a pilot slice covers a slice by
             design, so scoring it against the full meta would manufacture a fake dropout rate.
  NO-META    answers present with no meta.json to check them against

Read-only: this script never writes to results/.
"""
import os
import re
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SKEW_MULT = 3.0        # an arm is "skewed" at >3x the directory-wide dropout rate ...
MIN_SKEW_N = 3         # ... but only once it has actually lost >=3 items (guards tiny arms)
UNRUN_FRAC = 0.5       # below this the pack was never really dispatched -> UNRUN, not DROPOUT
ARM_KEYS = ("chan", "arm", "kind", "cond", "channel", "method", "card_kind")

# ans_3 -> generation "r1" ; ans_qwen_3 -> "qwen" ; ans_r2_3 -> "r2" ; anything else -> None
GEN = re.compile(r"^ans(?:_([A-Za-z][\w]*?))?_(\d+)$")
STRAY_GUARD = re.compile(r'fullmatch\(\s*r?[\'"]ans_\\d\+')
ANS_GLOB = re.compile(r'glob\(\s*f?[\'"]ans_?\*\.json[\'"]\s*\)')
# a scorer that captures the generation tag in its own regex separates them correctly
GEN_AWARE = re.compile(r'ans\(\?:_\(')

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def unguarded_scorers():
    """Scripts that glob ans files without either the stray-guard or a generation-aware regex."""
    out = []
    for p in sorted((ROOT / "scripts").glob("*.py")):
        try:
            t = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if ANS_GLOB.search(t) and not (STRAY_GUARD.search(t) or GEN_AWARE.search(t)):
            out.append(p.name)
    return out


def audit_dir(d):
    """-> (bad, warn, info) lists of strings for one pack directory."""
    bad, warn, info = [], [], []
    ansf = sorted(d.glob("ans*.json"))
    if not ansf:
        return None

    gens, unreadable = defaultdict(dict), []       # gen -> {pid: choice}
    for f in ansf:
        try:
            raw = f.read_text(encoding="utf-8-sig")
            recs = json.loads(raw) if raw.strip() else None
        except Exception as e:
            unreadable.append(f"{f.name} ({type(e).__name__})"); continue
        if recs is None:
            unreadable.append(f"{f.name} (empty)"); continue
        if not isinstance(recs, list):
            unreadable.append(f"{f.name} (not a list)"); continue
        m = GEN.fullmatch(f.stem)
        g = (m.group(1) or "r1") if m else f"?{f.stem}"
        for rec in recs:
            if isinstance(rec, dict) and "pid" in rec:
                gens[g][rec["pid"]] = str(rec.get("choice", "")).strip().upper()[:1]
    if unreadable:
        bad.append(f"UNREADABLE {unreadable[:4]}")

    named = sorted(g for g in gens if not g.startswith("?"))
    odd = sorted(g for g in gens if g.startswith("?"))
    if odd:
        warn.append(f"EXTRA-GEN 非常规命名的答案代 {odd[:3]}"
                    + (f"(与 {named} 同处一个目录)" if named else "")
                    + " —— 不为其计算覆盖率")
    # CONFLICT — only counts when at least one side is a ?-generation.
    #
    # Disagreement between PROPERLY NAMED generations is the instrument, not a defect: the FC packs
    # carry repeated replicates (r1/r2/r3) and a cross-judge pass (qwen) of the SAME items, and the
    # FC scorers key them apart (picks[pid][rep]) precisely so the disagreement can be measured.
    # Flagging those would condemn the design. A ?-generation is different: its name parses into no
    # rep, so every scorer either skips it (best case) or swallows it into whichever rep it happens
    # to sort next to (worst case) — that is the case where a number depends on filename order.
    allg = sorted(gens)
    for i in range(len(allg)):
        for j in range(i + 1, len(allg)):
            gi, gj = allg[i], allg[j]
            if not (gi.startswith("?") or gj.startswith("?")):
                continue
            a, b = gens[gi], gens[gj]
            both = set(a) & set(b)
            dis = [p for p in both if a[p] != b[p]]
            if dis:
                bad.append(f"CONFLICT [{gi}] vs [{gj}]: {len(dis)}/{len(both)} 个共同 pid 答案不同,"
                           f"例 {sorted(dis)[:3]} —— 其中一代名字解析不出 rep,合并结果取决于文件名排序")

    mp = d / "meta.json"
    if not mp.exists():
        warn.append("NO-META 有答案但没有 meta.json,无法校验")
        info.append(f"答案代: " + ", ".join(f"{g}={len(v)}" for g, v in sorted(gens.items())))
        return bad, warn, info
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except Exception as e:
        bad.append(f"UNREADABLE meta.json ({type(e).__name__})")
        return bad, warn, info
    if not isinstance(meta, dict) or not meta:
        return bad, warn, info

    mset = set(meta)
    probe = meta[next(iter(meta))]
    akey = next((k for k in ARM_KEYS if isinstance(probe, dict) and k in probe), None)

    for g, gmap in sorted(gens.items()):
        answered = set(gmap)
        hit, orph = answered & mset, answered - mset
        cov = len(hit) / len(mset)
        info.append(f"[{g}] meta={len(mset)} 已答={len(hit)} ({cov:.1%})")
        if orph:
            bad.append(f"ORPHAN [{g}] {len(orph)} 个已答 pid 不在 meta 里,例 {sorted(orph)[:3]}")
        if g.startswith("?"):
            continue                       # non-standard generation: a slice by design, see EXTRA-GEN
        if cov >= 1.0:
            continue
        if cov < UNRUN_FRAC:
            warn.append(f"UNRUN [{g}] 只答了 {cov:.1%} —— 这不是「部分测量」而是「没派单」,"
                        f"不得当作结果引用")
            continue
        miss = [p for p in mset if p not in answered]
        rate = len(miss) / len(mset)
        if not akey:
            warn.append(f"DROPOUT [{g}] 缺 {len(miss)} ({rate:.2%});meta 无臂字段,无法查是否集中")
            continue
        tot, mis = Counter(meta[p][akey] for p in mset), Counter(meta[p][akey] for p in miss)
        skew = [(a, mis[a], tot[a]) for a in tot
                if mis.get(a, 0) >= MIN_SKEW_N and mis[a] / tot[a] > SKEW_MULT * rate]
        if skew:
            bad.append("SKEWED [{}] 掉题集中在 {}(总体 {:.2%})".format(
                g, ", ".join(f"{a}:{m}/{n}={m/n:.2%}" for a, m, n in skew), rate))
        else:
            worst = max(((a, mis.get(a, 0) / tot[a]) for a in tot), key=lambda x: x[1])
            warn.append(f"DROPOUT [{g}] 缺 {len(miss)} ({rate:.2%}),各臂均衡"
                        f"(最偏 {worst[0]}={worst[1]:.2%})")
    return bad, warn, info


def main():
    roots = [Path(a) for a in sys.argv[1:]] or [RESULTS]
    dirs = set()
    for r in roots:
        r = r if r.is_absolute() else ROOT / r
        if list(r.glob("ans*.json")):
            dirs.add(r)
        for p in r.rglob("ans*.json"):
            if not re.fullmatch(r"round\d+", p.parent.name):   # naming rounds -> naming_cov.py
                dirs.add(p.parent)

    ug = unguarded_scorers()
    n_bad = n_warn = n_ok = 0
    for d in sorted(dirs):
        res = audit_dir(d)
        if res is None:
            continue
        bad, warn, info = res
        rel = str(d.relative_to(ROOT)).replace("\\", "/")
        tag = "BAD " if bad else ("WARN" if warn else "OK  ")
        n_bad += bool(bad); n_warn += bool(warn) and not bad; n_ok += not bad and not warn
        if bad or warn:
            print(f"[{tag}] {rel}")
            for m in bad:
                print(f"        ✗ {m}")
            for m in warn:
                print(f"        ! {m}")
            for m in info:
                print(f"          {m}")

    print(f"\n=== {len(dirs)} 个打包目录: {n_ok} OK / {n_warn} WARN / {n_bad} BAD ===")
    print(f"(点名淘汰赛的 round* 目录不在此列,由 scripts/naming_cov.py 无参跑负责)")
    if ug:
        print(f"\n提示(静态线索,不是判决): {len(ug)} 个脚本 glob ans 文件时既无残片守卫也无分代正则。")
        print(f"      多数是导出器(只用来看哪些批已答完),且它们的输入目录由 BATCHDIR 决定,")
        print(f"      静态判不出实际读了谁 —— 只有当被指向上面报 EXTRA-GEN/CONFLICT 的目录时才要紧。")
        print(f"      {', '.join(ug)}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
