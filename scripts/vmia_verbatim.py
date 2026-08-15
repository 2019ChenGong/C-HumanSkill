"""D1-VMIA (#159) — verbatim-channel membership inference against pooled skill cards.

Pre-registration: `results/VMIA_D1_PREREG.md`, INCLUDING its "修订 A" amendment, which was appended
BEFORE any number was produced and which supersedes the original primary criterion.
Run with no arguments:  python -P scripts/vmia_verbatim.py

WHY THIS INSTRUMENT EXISTS
--------------------------
The naming ruler (primary, attribution axis) hands the attacker a writing sample DELIBERATELY
disjoint from the card-building corpus.  The lexical certificate measures overlap on the BUILD-INPUT
side.  So `concat`'s 76-88% verbatim overlap lives in text the naming attacker never sees — that
ruler is structurally incapable of rewarding sanitization.  This script supplies the missing ruler:
an attacker who HOLDS the candidate's text and searches for verbatim reuse.

  question:  "is person a one of the k contributors behind card C?"
  score:     deterministic n-gram overlap between a's text and C   (no model, no judge, no waves)
  readout:   per-cluster AUC -> pooled, cluster bootstrap — the SAME estimator as the naming ruler.

★★ THE TAUTOLOGY, AND WHY THE PRIMARY TARGET IS `docs` AND NOT `card` ★★
`v5_sanitize.py` builds the V6 arm under a lexical gate with SHN = 6, where `src_sh` is the union of
every cluster member's elements AND every cluster member's aggro card; stage-1 minimal edit and
stage-2 fallback rewrite BOTH admit a line only when `_shingles(new) & src_sh` is empty (the relaxed
branches lower only the fidelity threshold, never the lexical gate), and a line that cannot pass is
DROPPED.  Therefore, for a true positive a, `cand_sh[a] is a subset of src_sh`, so
   S1(v6 card, a) == 0 on target `card` at n >= 6  IS AN ANALYTIC IDENTITY, NOT A MEASUREMENT.
n >= 7 adds nothing: any shared 7-gram contains a shared 6-gram.  Target `card` at n >= 6 is
therefore reported as the GATE'S POSTCONDITION and is barred from carrying the claim.
Target `docs` (the candidate's raw source documents) is NEVER part of `src_sh` — it is the only
target the sanitizer was not optimised against, so it is the primary one.  n in {3,4,5} on `card` is
likewise ungated (5-clean does not imply 6-clean) and is the honest robustness band there.

ARMS:  `v6` = ours, `concat` = naive-pooling baseline, `ne` = ATTRIBUTION ABLATION ONLY.
`ne` is never our method (user ruling 2026-08-04).  Its only job is fixed in the prereg amendment:
ne and concat are both UNSANITIZED but synthesised under opposite instructions, so if ne is as loud
as concat the advantage is attributable to line-level sanitization, and if ne is already as quiet as
v6 the writeup must say the advantage comes from the consensus synthesis choice instead.
"""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import naming_a12_verdict as A12          # noqa: E402  estimators reused verbatim
import elemk_v3_gates as EG               # noqa: E402  _shingles / _card_lines reused verbatim
                                          # (note: importing EG chdir's to the repo root; harmless
                                          #  here because every path constant is absolute)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------- frozen config (prereg + 修订 A)
DIRS = A12.DIRS                                   # {"enron": "enron", "mad": "mad", "cv": "se"}
ARMS = ("v6", "concat", "ne")
CLAIM_ARMS = ("v6", "concat")
KCL = 8
BUILD_SEEDS = (0, 1, 2)
NGRAMS = (3, 4, 5, 6, 8, 10)
NPRIMARY = 6
UNGATED_N = (3, 4, 5)                             # on target `card`, the band the sanitizer never saw
SCORES = ("S1", "S2", "S3", "S4")
SPRIMARY = "S1"
TPRIMARY = "docs"                                 # 修订 A: raw source documents carry the claim
DF_MAX = 2
MAXSHINGLE = 20
TPR_FPRS = (0.01, 0.10)
# env vars whose stale values would make `naming_export` sys.exit() at import (WARN-5)
_HOSTILE_ENV = ("ARMS", "BATCHDIR", "INDIV_SKIP", "BSEED_SKIP", "INDIV_PER_CLUSTER",
                "BSEEDS_V6", "ROUND", "SEED", "KCL")


# ---------------------------------------------------------------- corpora
def _load_ds(ds):
    """-> (aggro, byc, pool, pooled_cards, raw) for ONE dataset at the frozen KCL."""
    for v in _HOSTILE_ENV:
        os.environ.pop(v, None)
    os.environ["DATASET"] = ds
    for m in ("cmd_gate", "naming_export"):
        sys.modules.pop(m, None)
    import cmd_gate as CG                          # noqa: E402  re-imported per dataset
    import naming_export as NE                     # noqa: E402  for POOLC — same files as A12

    docs, authors, _nuwa, aggro, _ref, _raw = CG.load()
    pooled_cards = {a: json.loads(NE.POOLC[a].read_text(encoding="utf-8")) for a in ARMS}
    byc, pool = {}, {}
    for seed in BUILD_SEEDS:
        _grp, b = CG.make_groups(aggro, authors, KCL, seed)
        byc[seed] = {cid: mem for cid, mem in b.items() if len(mem) >= KCL}
        # ★ candidate pool == the NAMING RULER's pool (naming_export.build_brackets flattens `full`
        #   AFTER dropping short clusters). Scoring extra authors would change the negative set and
        #   make the two axes' AUCs silently incomparable.
        pool[seed] = sorted(a for mem in byc[seed].values() for a in mem)
    return aggro, byc, pool, pooled_cards, _raw_source_docs(ds, CG, docs, authors)


def _raw_source_docs(ds, CG, docs, authors):
    """T2: the documents that ACTUALLY reached the card-building LLM call, or None.

    ★ These bounds are the *nuwa evidence* budget, NOT cmd_gate's ref/raw split bar.  For MAD the two
    differ and an earlier version of this file got it wrong: `mad_cmd_build` passes
    card_comments[:18], but `mad_nuwa_step2.nuwa_extract` re-slices to [:NUWA_EVID] = [:12], so
    comments[12:18] never entered any card-building call.  Read the constant from the builder rather
    than restating it, so a change there cannot silently desynchronise this scorer.
    CV (修订 B): its source answers are not in the repo — they live in an 819 MB Posts.xml from the
    stats.stackexchange dump.  Rather than re-derive them here (an unverified rebuild would silently
    score a *different* corpus), we read the materialised file produced by `scripts/cv_t2_verify.py`,
    which only writes it after PROVING byte-identity to the original card-building inputs via an
    llm_cache key recomputation.  A missing file therefore still means uncovered, exactly as before.
    ★ Do not point this at `data/stackexchange/Posts.xml` — that is WPSE, NOT CV (registered gotcha).
    """
    if ds == "enron":
        assert CG.N_TRAIN == 12, f"enron nuwa evidence budget moved to {CG.N_TRAIN} — re-verify"
        return {a: " ".join(d.get("text", "") for d in docs[a][:CG.N_TRAIN]) for a in authors}
    if ds == "mad":
        from mad_nuwa_step2 import NUWA_EVID       # noqa: E402  the real card-building boundary
        return {a: " ".join(docs[a]["card_comments"][:NUWA_EVID]) for a in authors}
    if ds == "cv":
        p = ROOT / "data" / "se" / "cv_card_docs.json"
        if not p.exists():
            return None
        verified = json.loads(p.read_text(encoding="utf-8"))
        # Filter to `authors` and let G5 count/assert on anyone missing — 修订 B.3's SUBSET branch
        # expects a shortfall to surface as a printed drop count, never as a silent smaller denominator.
        return {a: verified[a] for a in authors if a in verified}
    return None


# ---------------------------------------------------------------- scoring (deterministic)
def _card_units(card, n):
    """-> (per-line shingle sets, per-line word lists, whole-card shingle set) for lines that can
    physically carry an n-gram.  ★ Lines with < n words can never be hit yet would inflate the S1/S2
    denominator; since line-length distribution differs BY ARM, keeping them would systematically
    depress the score of the shorter-lined arm (i.e. bias toward v6).  Prereg 修订 A requires the
    denominator to count only >= n-word lines."""
    lines, words = [], []
    for ln in EG._card_lines(card):
        w = re.findall(r"[a-z']+", ln.lower())        # same tokenisation as EG._shingles
        if len(w) >= n:
            lines.append(EG._shingles(ln, n)); words.append(w)
    allsh = set().union(*lines) if lines else set()
    return lines, words, allsh


class LazyShingles:
    """Candidate shingle sets, materialised per (author, m) ON DEMAND.

    ★ Eagerly precomputing m = 3..20 for every candidate would hold ~18 shingle sets over each
    candidate's full source corpus (12 Enron emails / 12 MAD comments) x ~120 candidates — several GB.
    S4 only ever ascends above n for pairs that ALREADY share an n-gram, which is rare (and, for the
    v6 arm, expected to be empty), so on-demand filling keeps the working set tiny.
    """

    def __init__(self, texts):
        self._t, self._c = texts, {}

    def get(self, a, m):
        k = (a, m)
        if k not in self._c:
            self._c[k] = EG._shingles(self._t[a], m)
        return self._c[k]

    def at(self, m):
        return {a: self.get(a, m) for a in self._t}


def _longest_shared_in_lines(words, cand, a, n):
    """S4: longest m >= n such that SOME line shares an m-gram with candidate `a`, else 0.

    Scoped to the same filtered lines as S1/S2 (no cross-line shingles, no header lines) so all four
    scores describe one text, and ascending from n so a pair with no n-gram overlap costs nothing.
    """
    best = 0
    for w in words:
        m = n
        while m < MAXSHINGLE and len(w) > m:
            nxt = {" ".join(w[i:i + m + 1]) for i in range(len(w) - m)}
            if not (nxt & cand.get(a, m + 1)):
                break
            m += 1
        best = max(best, m)
    return best


def score_card(card, cand_sh, cand, n, rare):
    """-> {score_name: {author: value}} for ONE card against every candidate, at one n."""
    lines, words, allsh = _card_units(card, n)
    nline = len(lines)
    out = {s: {} for s in SCORES}
    for a, sh in cand_sh.items():
        hit_idx = [i for i, ln in enumerate(lines) if ln & sh]
        out["S1"][a] = len(hit_idx) / nline if nline else 0.0
        out["S2"][a] = (sum(1 for i in hit_idx if lines[i] & sh & rare) / nline) if nline else 0.0
        out["S3"][a] = len(allsh & sh) / len(allsh) if allsh else 0.0
        # no n-gram shared => no (n+1)-gram shared => S4 is 0 without touching the text again
        out["S4"][a] = float(_longest_shared_in_lines([words[i] for i in hit_idx], cand, a, n)
                             if hit_idx else 0.0)
    return out


def _df_counts(cand_sh):
    df = {}
    for sh in cand_sh.values():
        for g in sh:
            df[g] = df.get(g, 0) + 1
    return df


# ---------------------------------------------------------------- readouts
def tpr_at_fpr(scores, labels, fpr):
    """DESCRIPTIVE ONLY (prereg 修订 A limitation 10): pools (card, candidate) pairs across clusters,
    so its unit is not the cluster and it carries no CI. Never a paper claim."""
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    neg, pos = s[y == 0], s[y == 1]
    if not len(neg) or not len(pos):
        return float("nan")
    t = np.quantile(neg, 1.0 - fpr, method="higher")
    return float((pos > t).mean())


def pooled_auc(per_cluster_pairs):
    units = []
    for cid in sorted(per_cluster_pairs):
        a = A12.cluster_auc(per_cluster_pairs[cid])
        if a is not None:
            units.append([a])
    return A12.multiseed(units, "pooled")


def paired_diff(pa, pb):
    """EXACT per-cluster paired difference AUC(a) - AUC(b): same cluster, same candidate set, only
    the card text differs."""
    assert set(pa) == set(pb), f"G4: arm cluster sets differ ({sorted(set(pa) ^ set(pb))[:5]})"
    units = []
    for cid in sorted(pa):
        x, y = A12.cluster_auc(pa[cid]), A12.cluster_auc(pb[cid])
        if x is not None and y is not None:
            units.append([x - y])
    return A12.multiseed(units, "paired")


def verdict_word(diff, ref_effect):
    """Prereg 修订 A judgement dictionary, ported from the FC utility instrument."""
    lo, hi = diff["ci"]
    hw = (hi - lo) / 2
    if lo > 0 or hi < 0:
        return "SIG", hw
    if ref_effect and hw < abs(ref_effect) / 2:
        return "TIE", hw
    return "UNDERPOWERED", hw


# ---------------------------------------------------------------- driver
def run_target(ds_data, target):
    res, diag, uncovered, dropped = {}, {}, set(), {}
    for ds, (aggro, byc, pool, pooled_cards, raw) in ds_data.items():
        src = aggro if target == "card" else raw
        if src is None:
            uncovered.add(ds)
            continue
        for seed in BUILD_SEEDS:
            cand_text = {a: src[a] for a in pool[seed] if src.get(a)}
            miss = [a for a in pool[seed] if not src.get(a)]
            if miss:                                                        # G5
                dropped[f"{ds}/s{seed}"] = len(miss)
                for cid, mem in byc[seed].items():
                    kept = sum(1 for m in mem if m in cand_text)
                    assert kept >= KCL, (f"G5: {ds}/s{seed}/{cid} lost positives to missing text "
                                         f"({kept} < {KCL})")
            cand = LazyShingles(cand_text)
            for n in NGRAMS:
                cand_sh = cand.at(n)
                df = _df_counts(cand_sh)
                rare = {g for g, c in df.items() if c <= DF_MAX}
                cid_sets = {}
                for arm in ARMS:
                    cards = pooled_cards[arm]
                    for cid, mem in sorted(byc[seed].items()):
                        card = cards.get(f"k{KCL}_s{seed}_{cid}")
                        if not card:
                            continue
                        cid_sets.setdefault(arm, set()).add(cid)
                        sc = score_card(card, cand_sh, cand, n, rare)
                        memset = set(mem)
                        for s in SCORES:
                            pairs = [(sc[s][a], int(a in memset)) for a in sorted(cand_sh)]
                            res.setdefault((ds, n, seed, arm, s), {})[cid] = pairs
                        if n == NPRIMARY:
                            diag.setdefault((ds, seed, arm), []).append(
                                (len(_card_units(card, n)[0]), len(card.split()),
                                 sum(1 for a in cand_sh if sc[SPRIMARY][a] == 0.0), len(cand_sh)))
                sets = list(cid_sets.values())                              # G4
                assert all(s == sets[0] for s in sets), \
                    f"G4: {ds}/s{seed}/n{n} arms cover different clusters: " \
                    f"{ {a: len(v) for a, v in cid_sets.items()} }"
    return res, diag, uncovered, dropped


def independence_diag(ds_data, n=NPRIMARY):
    """修订 A limitation 8: how much of a candidate's own aggro card is verbatim from their raw docs.
    High values mean T2's signal partly flows through text the sanitizer's gate DID cover."""
    out = {}
    for ds, (aggro, byc, pool, _cards, raw) in ds_data.items():
        if raw is None:
            out[ds] = None
            continue
        vals = []
        for a in pool[BUILD_SEEDS[0]]:
            if not (aggro.get(a) and raw.get(a)):
                continue
            lines, _w, _all = _card_units(aggro[a], n)
            rsh = EG._shingles(raw[a], n)
            if lines:
                vals.append(sum(1 for ln in lines if ln & rsh) / len(lines))
        out[ds] = round(float(np.mean(vals)), 4) if vals else None
    return out


def main():
    print("=" * 96)
    print("D1-VMIA 逐字通道成员推断(判据 = results/VMIA_D1_PREREG.md + 修订 A,均在跑数前冻结)")
    print(f"主 target = {TPRIMARY}(原始源文档,消毒器闸门未覆盖) | 主 n = {NPRIMARY} | 主分数 = {SPRIMARY}")
    print("=" * 96)

    ds_data = {ds: _load_ds(ds) for ds in DIRS}
    out = {"experiment": "D1-VMIA (#159) verbatim-channel membership inference",
           "prereg": "results/VMIA_D1_PREREG.md (incl. 修订 A)",
           "primary": {"target": TPRIMARY, "n": NPRIMARY, "score": SPRIMARY},
           "arms": list(ARMS), "claim_arms": list(CLAIM_ARMS), "ablation_only": ["ne"],
           "kcl": KCL, "build_seeds": list(BUILD_SEEDS), "ngrams": list(NGRAMS),
           "scores": list(SCORES), "df_max": DF_MAX,
           "boot": {"n": A12.BOOT_N, "seeds": list(A12.SEEDS), "unit": "ds:seed:cluster"},
           "targets": {}}

    ind = independence_diag(ds_data)
    out["independence_diag_aggro_vs_raw"] = ind
    print("\n--- 修订A-限制8 诊断:候选人 aggro 卡中逐字来自其原始文档的行占比(n=6)---")
    for ds, v in ind.items():
        print(f"  {ds:6s} {'无 T2 覆盖' if v is None else f'{v:.1%}'}"
              + ("   <- 越高说明 T2 的信号越多流经闸门覆盖过的文本" if v else ""))

    for target in ("docs", "card"):
        res, diag, uncovered, dropped = run_target(ds_data, target)
        tsec = {"per_cell": {}, "gates": {}, "criteria": {}, "dropped_candidates": dropped}
        head = ("候选人原始源文档 —— ★主 target,消毒器闸门未覆盖" if target == "docs"
                else "候选人逐人卡 —— ⚠ n≥6 由消毒器闸门构造保证为 0,非测量结果")
        print(f"\n{'#' * 96}\n### TARGET = {target}   ({head})\n{'#' * 96}")
        if dropped:
            print(f"  G5 因文本缺失被剔除的候选人: {dropped}")

        print("\n--- G2/G3 披露:卡长度 与 分数恰为 0 的候选人比例(n=6, S1)---")
        print(f"{'ds':6s} {'seed':>4s} {'arm':>7s} {'簇':>3s} {'≥6词行/卡':>10s} {'词/卡':>7s} {'零分占比':>9s} {'档':>6s}")
        for (ds, seed, arm), v in sorted(diag.items()):
            nl = np.mean([x[0] for x in v]); nw = np.mean([x[1] for x in v])
            z = sum(x[2] for x in v) / max(1, sum(x[3] for x in v))
            band = "解析零" if z == 1.0 else ("近乎零" if z >= 0.90 else "常规")
            print(f"{ds:6s} {seed:4d} {arm:>7s} {len(v):3d} {nl:10.1f} {nw:7.0f} {z:8.1%} {band:>6s}")
            tsec["gates"].setdefault("length_and_zeros", {})[f"{ds}/s{seed}/{arm}"] = {
                "clusters": len(v), "mean_scorable_lines": round(float(nl), 1),
                "mean_words": round(float(nw), 0), "zero_score_frac": round(float(z), 4),
                "band": band}

        for ds in DIRS:
            if ds in uncovered:
                print(f"\n  ★ {ds}: TARGET={target} 无覆盖(源语料本地不可得,预注册 §7-2)——本节不出数")
                tsec["per_cell"][ds] = {"uncovered": True}
                continue
            for n in NGRAMS:
                for s in SCORES:
                    flag = ""
                    if n == NPRIMARY and s == SPRIMARY:
                        flag = "  ★主格" if target == TPRIMARY else "  ⚠闸门后置条件(非测量)"
                    elif target == "card" and n >= 6:
                        flag = "  ⚠闸门覆盖"
                    elif target == "card" and n in UNGATED_N:
                        flag = "  (闸门未覆盖)"
                    print(f"\n--- {ds} | n={n} | {s}{flag} ---")
                    for seed in BUILD_SEEDS:
                        row = {}
                        for arm in ARMS:
                            key = (ds, n, seed, arm, s)
                            if key in res:
                                row[arm] = pooled_auc(res[key])
                                tsec["per_cell"][f"{ds}/n{n}/{s}/s{seed}/{arm}"] = row[arm]
                        if not row:
                            continue
                        print(f"  s{seed} 簇={row[ARMS[0]]['n_units']}  AUC  " + "  ".join(
                            f"{a}={row[a]['est']:.4f}[{row[a]['ci'][0]:.4f},{row[a]['ci'][1]:.4f}]"
                            for a in ARMS if a in row))
                        if "concat" in row and "v6" in row:
                            d = paired_diff(res[(ds, n, seed, "concat", s)],
                                            res[(ds, n, seed, "v6", s)])
                            w, hw = verdict_word(d, row["concat"]["est"] - 0.5)
                            print(f"       配对差 concat−v6 = {d['est']:+.4f} "
                                  f"[{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]  {w}  (半宽/sMDE={hw:.4f})")
                            d["verdict_word"] = w; d["half_width"] = round(hw, 4)
                            tsec["per_cell"][f"{ds}/n{n}/{s}/s{seed}/paired_concat_minus_v6"] = d

        print(f"\n--- TPR@FPR(n={NPRIMARY}, {SPRIMARY})—— ★描述性,不进任何论文主张(修订A-限制10)---")
        for ds in DIRS:
            if ds in uncovered:
                continue
            for seed in BUILD_SEEDS:
                cells = []
                for arm in ARMS:
                    key = (ds, NPRIMARY, seed, arm, SPRIMARY)
                    if key not in res:
                        continue
                    allp = [p for cid in sorted(res[key]) for p in res[key][cid]]
                    t = {f"tpr@{f:.0%}": round(tpr_at_fpr([x for x, _ in allp],
                                                          [y for _, y in allp], f), 4)
                         for f in TPR_FPRS}
                    cells.append(f"{arm}: " + " ".join(f"{k}={v}" for k, v in t.items()))
                    tsec.setdefault("tpr_descriptive", {})[f"{ds}/s{seed}/{arm}"] = t
                if cells:
                    print(f"  {ds:6s} s{seed}   " + "   |   ".join(cells))
        out["targets"][target] = tsec

    # ---------------- gates + frozen criteria ----------------
    print("\n" + "=" * 96)
    print("跑前硬闸 与 冻结判据(修订 A)")
    print("=" * 96)

    prim, sec = out["targets"][TPRIMARY], out["targets"]["card"]
    cov = [ds for ds in DIRS if not sec["per_cell"].get(ds, {}).get("uncovered")
           and not prim["per_cell"].get(ds, {}).get("uncovered")]

    g1 = {f"{ds}/s{seed}": bool(r["ci"][0] > 0.5)
          for ds in DIRS for seed in BUILD_SEEDS
          if (r := sec["per_cell"].get(f"{ds}/n{NPRIMARY}/{SPRIMARY}/s{seed}/concat"))}
    assert len(g1) == len(DIRS) * len(BUILD_SEEDS), \
        f"G6: G1 只取到 {len(g1)} 格,期望 {len(DIRS) * len(BUILD_SEEDS)} 格 —— 有格缺失,不得按更小分母判 PASS"
    g1_pass = all(g1.values())
    print(f"  G1 仪器闸(在 target=card 上,concat 合并 AUC 的 CI-lo > 0.5): "
          f"{sum(g1.values())}/{len(g1)} 格 -> {'✔ 打分管线有效' if g1_pass else '✘ **打分器坏了,下面一律不可读**'}")
    if not g1_pass:
        print("     " + "  ".join(f"{k}:{'✔' if v else '✘'}" for k, v in sorted(g1.items())))
    out["gates"] = {"G1_instrument": {"per_cell": g1, "pass": g1_pass}}

    print(f"\n  ① 主判据  target={TPRIMARY} / n={NPRIMARY} / {SPRIMARY} 逐簇配对差 concat − v6:")
    c1 = {}
    for ds in cov:
        for seed in BUILD_SEEDS:
            d = prim["per_cell"].get(f"{ds}/n{NPRIMARY}/{SPRIMARY}/s{seed}/paired_concat_minus_v6")
            if not d:
                continue
            c1[f"{ds}/s{seed}"] = d["verdict_word"]
            print(f"     {ds:6s} s{seed}  {d['est']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]  "
                  f"簇={d['n_units']}  -> {d['verdict_word']}")
    nsig = sum(1 for v in c1.values() if v == "SIG")
    verdict = (f"逐字通道上 v6 严格优于朴素池化({nsig}/{len(c1)} 格 SIG)" if c1 and nsig == len(c1)
               else f"★ {nsig}/{len(c1)} 格 SIG —— 按预注册如实报,禁止只报有利格;"
                    f"UNDERPOWERED 格不得写成「无差别」")
    print(f"     -> {verdict}")
    prim["criteria"]["C1_primary"] = c1

    print("\n  ② `ne` 归因(修订 A 锁死的解释规则):")
    for ds in cov:
        for seed in BUILD_SEEDS:
            k = f"{ds}/n{NPRIMARY}/{SPRIMARY}/s{seed}"
            r = {a: prim["per_cell"].get(f"{k}/{a}") for a in ARMS}
            if not all(r.values()):
                continue
            near = abs(r["ne"]["est"] - r["v6"]["est"]) < abs(r["concat"]["est"] - r["v6"]["est"]) / 2
            print(f"     {ds:6s} s{seed}  v6={r['v6']['est']:.4f}  ne={r['ne']['est']:.4f}  "
                  f"concat={r['concat']['est']:.4f}  -> "
                  f"{'共识合成已消掉大部分 ⇒ 归因于合成选择' if near else '归因于逐行消毒'}")
            prim["criteria"].setdefault("C2_ne_attribution", {})[f"{ds}/s{seed}"] = \
                "synthesis_choice" if near else "line_sanitization"

    out["verdict"] = {
        "G1_instrument_pass": g1_pass, "C1": verdict,
        "tautology_note": ("target=card 且 n>=6 的 v6 零分是 v5_sanitize 词面闸门的后置条件,"
                           "是解析恒等式不是测量结果;该格不得承载任何主张(预注册 修订 A)。"),
        "scope_note": ("只测逐字通道。R7 已测出消毒后语义通道仍保留 τ=.70 差 +.58–.69 ⇒ "
                       "零信号不等于「没有 MIA 能打 v6」;更不得写成「v6 匿名」。")}

    p = ROOT / "results/_VMIA_D1_all.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        print(f"\nsaved -> {p.relative_to(ROOT)}")
    except ValueError:
        print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
