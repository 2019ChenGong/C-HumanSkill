"""Reusable, dataset-agnostic attribution metrics for the de-id experiments.

This is the SHARED CORE both testbeds import (synthetic + Enron now, DICES later). It never touches
emails / labels / cards -- it operates only on, per UNIT, the detective's predicted author ids vs the
ground-truth author. So porting to DICES later reuses this file unchanged.

A "unit" = one (author, held-item) cell. For each unit the detective produced K (re-shuffled,
temperature-sampled) guesses; each guess is the predicted author id, or None if unparsed.

  O1  soft accuracy  : per-unit pick-fraction in [0,1] (continuity for free from the K resamples);
                       aggregate = mean over units (== overall accuracy, but unit-resolved).
  O7  empirical null : permute the AUTHOR labels (random relabeling sigma: author->author) many times,
                       recompute the metric -> the EMPIRICAL chance distribution. Replaces the asserted
                       1/N: it captures the detective's guess-bias and the finite-sample variance, and
                       works for any metric. Author-level (not unit-level) permutation respects that
                       identity lives at the author block and handles within-author correlation.
  boot bootstrap CI  : paired bootstrap over units (RNG built ONCE).
  O8  multiple-comp  : Holm-Bonferroni / Benjamini-Hochberg, ready for the per-author/per-metric family.

All randomness uses an explicit seed and an RNG built once, so results are reproducible.
"""
import numpy as np


# ---------- O1: soft (continuous) accuracy ----------
def soft_acc_units(preds, truth):
    """preds: list over units of [K predicted author ids] (None entries allowed, count as wrong).
    truth: list over units of the true author id. Returns per-unit soft accuracy (np.ndarray)."""
    out = np.empty(len(preds))
    for i, (ps, t) in enumerate(zip(preds, truth)):
        out[i] = np.mean([1.0 if p == t else 0.0 for p in ps]) if ps else 0.0
    return out


def mean_soft_acc(preds, truth):
    u = soft_acc_units(preds, truth)
    return float(u.mean()) if len(u) else float("nan")


# ---------- O7: empirical permutation null ----------
def empirical_null(preds, truth, authors=None, n_perm=5000, seed=0, block=True):
    """Empirical permutation null for mean soft accuracy -> the chance line is MEASURED, not asserted 1/N.
    It conditions on the detective's observed guesses, so it captures the detective's marginal guess-bias
    (verbosity / position priors) and the finite-sample variance.

    block=True  (DEFAULT, primary): AUTHOR-BLOCK permutation. Draw a random bijection sigma over the
        distinct authors and carry the new label to ALL of that author's units. This respects that units
        of one author share the same lineup/cards (their guesses are correlated), so it is the
        statistically correct, CONSERVATIVE null.
    block=False (sensitivity only): unit-level shuffle of the truth labels across units. Treats units as
        i.i.d.-exchangeable, which underestimates the variance when within-author guesses are correlated
        -> anti-conservative. Keep only as a sensitivity check.

    Decisions use CI-CONTAINMENT (matches the design's "obs inside the empirical null CI"), which a
    BELOW-chance arm correctly fails (a one-sided upper test would wrongly pass it):
      above_chance   = obs ABOVE the null 97.5th pct (significantly identifiable)
      reached_chance = obs INSIDE the 95% null CI [2.5, 97.5] (indistinguishable from chance)
    Also returns the one-sided permutation p (P[null >= obs], +1 smoothed). `authors` overrides the
    author set (else inferred from truth)."""
    truth = list(truth)
    if authors is None:
        authors = sorted(set(truth))
    A = list(authors)
    obs = mean_soft_acc(preds, truth)
    g = np.random.default_rng(seed)                    # built ONCE
    null = np.empty(n_perm)
    if block:
        for b in range(n_perm):
            perm = g.permutation(len(A))
            sigma = {A[i]: A[perm[i]] for i in range(len(A))}
            null[b] = mean_soft_acc(preds, [sigma[t] for t in truth])
    else:
        t_arr = np.array(truth, dtype=object)
        for b in range(n_perm):
            null[b] = mean_soft_acc(preds, list(g.permutation(t_arr)))
    lo, hi = np.percentile(null, [2.5, 97.5])
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"observed": round(obs, 4),
            "null_mean": round(float(null.mean()), 4),
            "null_ci": [round(float(lo), 4), round(float(hi), 4)],
            "null_hi95": round(float(np.percentile(null, 95)), 4),
            "p_value": round(p, 4),
            "block": block,
            "above_chance": bool(obs > hi),              # above the null 97.5th pct
            "reached_chance": bool(lo <= obs <= hi)}     # inside the 95% null CI (containment)


# ---------- bootstrap CIs (paired over units) ----------
def paired_diff_ci(a_units, b_units, n_boot=2000, seed=0, alpha=0.05):
    """95% CI for mean(a - b) over PAIRED units (same unit index in both). RNG built once.
    gap excludes 0 => the two arms differ on the same units."""
    d = np.asarray(a_units, float) - np.asarray(b_units, float)
    n = len(d)
    g = np.random.default_rng(seed)
    bs = np.array([d[g.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


def mean_ci(values, n_boot=2000, seed=0, alpha=0.05):
    """95% bootstrap CI for a single arm's mean over units."""
    v = np.asarray(values, float)
    n = len(v)
    g = np.random.default_rng(seed)
    bs = np.array([v[g.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


# ---------- CLUSTER (author-level) bootstrap ----------
# When several units share one author's card their outcomes are correlated; resampling UNITS i.i.d. treats
# them as independent -> understates variance -> false significance. These resample whole AUTHOR BLOCKS (each
# drawn author contributes ALL its units), so the effective sample size is #authors, not #units. RNG built once.
def _clusters(groups):
    """groups: per-unit author id (same order as the value arrays). -> list of np index-arrays, one per author."""
    idx = {}
    for i, a in enumerate(groups):
        idx.setdefault(a, []).append(i)
    return [np.asarray(v) for v in idx.values()]


def cluster_mean_ci(values, groups, n_boot=2000, seed=0, alpha=0.05):
    """95% CI for an arm's mean, resampling AUTHORS (with all their units) -> honest under within-author corr."""
    v = np.asarray(values, float)
    cl = _clusters(groups)
    nA = len(cl)
    g = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([cl[p] for p in g.integers(0, nA, nA)])
        bs[b] = v[idx].mean()
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


def cluster_paired_diff_ci(a_units, b_units, groups, n_boot=2000, seed=0, alpha=0.05):
    """Author-cluster paired bootstrap of mean(a - b) over units. Resamples AUTHOR blocks (carrying all their
    paired per-unit diffs). Returns {diff, ci=[lo,hi], p} where p = 2-sided bootstrap p (proportion of resample
    means on the far side of 0, doubled, capped at 1) for the Holm family. CI excludes 0 <=> arms differ."""
    d = np.asarray(a_units, float) - np.asarray(b_units, float)
    cl = _clusters(groups)
    nA = len(cl)
    g = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([cl[p] for p in g.integers(0, nA, nA)])
        bs[b] = d[idx].mean()
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    ge, le = int((bs >= 0).sum()), int((bs <= 0).sum())     # +1 smoothing -> no indefensible p=0.0
    p = min(1.0, 2.0 * (min(ge, le) + 1) / (n_boot + 1))
    return {"diff": round(float(d.mean()), 4), "ci": [round(float(lo), 4), round(float(hi), 4)],
            "p": round(p, 4)}


# ---------- TWO-WAY (rater x item) crossed bootstrap ----------
# For a CROSSED design (every rater judges every item, e.g. DICES-350), BOTH rater variance and item variance are
# real and per-item outcomes are correlated across raters. Resampling raters ALONE (cluster_* above) treats items
# as fixed/exhausted -> anticonservative for item-level effects. These resample rater indices AND item indices
# independently (multiplier bootstrap): a replicate weights cell (r,i) by rmult[r]*imult[i]. RNG built once.
def _twoway_boot(d, rater_of, item_of, n_boot, seed):
    raters = sorted(set(rater_of)); items = sorted(set(item_of))
    rid = {r: i for i, r in enumerate(raters)}; iid = {t: i for i, t in enumerate(items)}
    ru = np.array([rid[r] for r in rater_of]); iu = np.array([iid[t] for t in item_of])
    R, I = len(raters), len(items)
    g = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        rm = np.bincount(g.integers(0, R, R), minlength=R).astype(float)
        im = np.bincount(g.integers(0, I, I), minlength=I).astype(float)
        w = rm[ru] * im[iu]
        sw = w.sum()
        bs[b] = float((d * w).sum() / sw) if sw > 0 else np.nan
    return bs


def two_way_mean_ci(values, rater_of, item_of, n_boot=2000, seed=0, alpha=0.05):
    """95% CI for an arm's mean over a crossed rater x item design, resampling BOTH axes -> honest under item- AND
    rater-level correlation. rater_of/item_of are per-unit id lists (same order as values)."""
    bs = _twoway_boot(np.asarray(values, float), rater_of, item_of, n_boot, seed)
    lo, hi = np.nanpercentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


def two_way_paired_diff_ci(a_units, b_units, rater_of, item_of, n_boot=2000, seed=0, alpha=0.05):
    """Crossed (rater x item) paired bootstrap of mean(a - b). Returns {diff, ci=[lo,hi], p} (2-sided, +1 smoothed).
    CI excludes 0 <=> arms differ, honest to BOTH rater and item variance."""
    d = np.asarray(a_units, float) - np.asarray(b_units, float)
    bs = _twoway_boot(d, rater_of, item_of, n_boot, seed)
    lo, hi = np.nanpercentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    ge, le = int((bs >= 0).sum()), int((bs <= 0).sum())
    p = min(1.0, 2.0 * (min(ge, le) + 1) / (n_boot + 1))
    return {"diff": round(float(d.mean()), 4), "ci": [round(float(lo), 4), round(float(hi), 4)], "p": round(p, 4)}


# ---------- O8: multiple-comparison correction ----------
def holm(pvals):
    """Holm-Bonferroni adjusted p-values (FWER), returned in the INPUT order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return [round(float(x), 4) for x in adj]


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values (FDR), returned in the INPUT order."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * m / (rank + 1))
        adj[i] = min(prev, 1.0)
    return [round(float(x), 4) for x in adj]


# ---------- self-test ----------
def _selftest():
    rng = np.random.default_rng(42)
    N, n_held = 6, 6
    authors = list(range(N))
    truth = [a for a in authors for _ in range(n_held)]   # balanced: each author n_held units
    K = 3

    # perfect detective -> obs=1, far above null
    perfect = [[t] * K for t in truth]
    r = empirical_null(perfect, truth, authors, n_perm=3000, seed=0)
    assert r["observed"] == 1.0 and r["above_chance"] and not r["reached_chance"], r
    assert abs(r["null_mean"] - 1.0 / N) < 0.03, r

    # random detective -> obs ~ 1/N, NOT above null, reached chance
    rand = [[int(rng.integers(0, N)) for _ in range(K)] for _ in truth]
    r2 = empirical_null(rand, truth, authors, n_perm=3000, seed=0)
    assert abs(r2["observed"] - 1.0 / N) < 0.08, r2
    assert r2["reached_chance"] and not r2["above_chance"], r2

    # None guesses count as wrong; null still ~ valid_frac / N
    halfnone = [[t, None, None] for t in truth]            # 1/3 valid, always correct when valid
    r3 = empirical_null(halfnone, truth, authors, n_perm=2000, seed=0)
    assert abs(r3["observed"] - 1.0 / 3) < 1e-3, r3        # 4-dp rounded
    assert r3["above_chance"], r3
    assert abs(r3["null_mean"] - (1.0 / 3) / N) < 0.02, r3  # null ~ valid_frac / N

    # paired diff CI: a strictly above b -> CI excludes 0 and is positive
    a = [1.0] * 30; b = [0.0] * 30
    ci = paired_diff_ci(a, b, seed=0)
    assert ci[0] > 0.5, ci

    # Holm/BH exact values (locks correctness, not just bounds)
    hp = holm([0.01, 0.04, 0.03, 0.5]); bp = bh([0.01, 0.04, 0.03, 0.5])
    assert hp == [0.04, 0.09, 0.09, 0.5], hp          # Holm step-down, input order preserved
    assert bp == [0.04, 0.0533, 0.0533, 0.5], bp      # BH step-up, suffix-min from largest rank
    assert all(0 <= x <= 1 for x in hp + bp), (hp, bp)

    # cluster bootstrap: with WITHIN-author correlation, cluster CI must be WIDER than naive unit CI (honest)
    g2 = np.random.default_rng(1)
    grp = [a for a in range(20) for _ in range(6)]               # 20 authors x 6 units
    base = {a: float(g2.normal(0, 1)) for a in range(20)}        # author-level effect (units share it)
    vals = [base[a] + float(g2.normal(0, 0.05)) for a in range(20) for _ in range(6)]
    naive = mean_ci(vals, seed=0); clust = cluster_mean_ci(vals, grp, seed=0)
    assert (clust[1] - clust[0]) > (naive[1] - naive[0]), (naive, clust)   # cluster CI WIDER
    pr = cluster_paired_diff_ci([1.0] * 120, [0.0] * 120, grp, seed=0)
    assert pr["ci"][0] > 0.5 and pr["p"] < 0.05, pr             # all-1 vs all-0 -> clearly significant

    print("attrib_metrics self-test PASSED")
    print("  cluster CI wider than naive:", clust, ">", naive)
    print("  perfect:", r)
    print("  random :", r2)
    print("  Holm   :", hp, " BH:", bp)


if __name__ == "__main__":
    _selftest()
