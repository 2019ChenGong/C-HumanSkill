"""Per-person / per-cluster dispersion behind the pooled-anonymity MEANS (P1/#48, $0, no spend).

The headline "pooling reaches chance" numbers are MEANS. A mean at chance can still hide an identity-rich
MINORITY that leaks single-release. This script decomposes each mean into its per-cluster / per-person
distribution, tests whether the spread exceeds pure sampling noise, and checks whether the leaky tail is a
STABLE identity-driven set (indiv-vs-pooled correlation) rather than noise.

Two modes (env MODE):

  MODE=fixed  (DEFAULT, task A4 / #108) -- the CANONICAL CMD result on FIXED cards.
    Reads the `neutral` channel from the neufix packs, AGGREGATED across 3 seeds:
      MAD  results/mad/neufix_k8_s{0,1,2}   Enron results/enron/neufix_k8_s{0,1,2}   CV results/se/neufix_k8_s{0,1,2}
    Each pooled card = one (seed, card_id) cluster (random grouping differs per seed -> distinct units).
    The `indiv` leaky reference is read PER-PERSON from the base battery (indiv acc is grouping-independent):
      MAD results/mad/2afc_free   Enron results/enron/2afc_free   CV results/se/2afc_battery
    A person appears once per seed (3 pooled cards, 2 trials each = ~6 neutral trials) -> per-person leak is
    robust to WHO they were pooled with. Cross-channel corr uses base indiv acc vs neutral acc.
    -> results/dispersion_2afc_fixed.json

  MODE=base   -- legacy: the base-battery `shared` (pre-degeneracy-fix) pooled card, indiv from same battery.
    -> results/dispersion_2afc.json

  MODE=fixed python scripts/cmd_dispersion.py
"""
import os
import re
import sys
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODE = os.environ.get("MODE", "fixed")

# legacy base-battery pooled card (pre-degeneracy-fix `shared` channel)
DIRS = [("enron", ROOT / "results/enron/2afc_free"),
        ("mad", ROOT / "results/mad/2afc_free"),
        ("cv", ROOT / "results/se/2afc_battery")]

# fixed cards: neutral channel from neufix packs (3 seeds) + indiv reference from base battery
FIXED = [
    ("enron", [ROOT / f"results/enron/neufix_k8_s{s}" for s in (0, 1, 2)], ROOT / "results/enron/2afc_free"),
    ("mad",   [ROOT / f"results/mad/neufix_k8_s{s}"   for s in (0, 1, 2)], ROOT / "results/mad/2afc_free"),
    ("cv",    [ROOT / f"results/se/neufix_k8_s{s}"    for s in (0, 1, 2)], ROOT / "results/se/2afc_battery"),
]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(d):
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    ans = {}
    for f in sorted(d.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", f.stem):
            continue
        for rec in json.loads(f.read_text(encoding="utf-8")):
            if isinstance(rec, dict) and "pid" in rec:
                ans[rec["pid"]] = rec
    rows = []   # (chan, member, card, correct)
    for pid, m in meta.items():
        a = ans.get(pid)
        if not a or "choice" not in a:
            continue
        correct = int(str(a["choice"]).strip().upper()[:1] == str(m["member_slot"]).strip().upper()[:1])
        rows.append((m["chan"], m["member"], m["card_id"], correct))
    return rows


def chan_stats(rows, chan):
    r = [x for x in rows if x[0] == chan]
    n = len(r)
    acc = np.mean([c for *_x, c in r]) if n else float("nan")
    # per-cluster
    byc = defaultdict(list)
    for _ch, _m, card, c in r:
        byc[card].append(c)
    caccs = {card: np.mean(v) for card, v in byc.items()}
    csizes = {card: len(v) for card, v in byc.items()}
    cvals = np.array(list(caccs.values()))
    # per-person
    bym = defaultdict(list)
    for _ch, m, _card, c in r:
        bym[m].append(c)
    paccs = {m: np.mean(v) for m, v in bym.items()}
    pvals = np.array(list(paccs.values()))
    # expected cross-cluster SD if every trial were iid Bernoulli(acc) (pure sampling noise)
    nbar = np.mean(list(csizes.values()))
    sd_binom = math.sqrt(acc * (1 - acc) / nbar) if 0 < acc < 1 else 0.0
    # leaky tail: clusters whose Wilson-lower > 0.5
    leaky = []
    for card, v in byc.items():
        lo, hi = wilson(int(sum(v)), len(v))
        if lo > 0.5:
            leaky.append((card, float(np.mean(v)), len(v), round(lo, 3)))
    leaky.sort(key=lambda x: -x[1])
    return {"n": n, "acc": float(acc), "caccs": caccs, "csizes": csizes,
            "cluster_sd": float(cvals.std()) if len(cvals) else float("nan"),
            "cluster_min": float(cvals.min()) if len(cvals) else float("nan"),
            "cluster_max": float(cvals.max()) if len(cvals) else float("nan"),
            "sd_binom_expected": sd_binom, "n_clusters": len(byc),
            "paccs": paccs,
            "p_full": float(np.mean(pvals == 1.0)) if len(pvals) else float("nan"),
            "p_zero": float(np.mean(pvals == 0.0)) if len(pvals) else float("nan"),
            "n_persons": len(bym), "leaky_clusters": leaky}


def load_rows_robust(d):
    """(chan, member, card, correct) for one pack dir; utf-8-sig tolerant, skips non-AB choices."""
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    ans = {}
    for f in sorted(d.glob("ans_*.json")):
        if not re.fullmatch(r"ans_\d+", f.stem):
            continue
        for rec in json.loads(f.read_text(encoding="utf-8-sig")):
            if isinstance(rec, dict) and "pid" in rec:
                ans[rec["pid"]] = rec
    rows = []
    for pid, m in meta.items():
        a = ans.get(pid)
        if not a or "choice" not in a:
            continue
        ch = str(a["choice"]).strip().upper()
        if not ch or ch[0] not in "AB":
            continue
        correct = int(ch[0] == str(m["member_slot"]).strip().upper()[:1])
        rows.append((m["chan"], m["member"], m["card_id"], correct))
    return rows


def indiv_by_person(base_dir):
    """per-person indiv acc {member: acc} from a base battery (indiv is grouping-independent)."""
    pm = defaultdict(lambda: [0, 0])
    for chan, member, _card, correct in load_rows_robust(base_dir):
        if chan != "indiv":
            continue
        pm[member][0] += correct
        pm[member][1] += 1
    return {m: k / n for m, (k, n) in pm.items() if n}


def fixed_stats(neu_dirs):
    """decompose the neutral pooled mean; cluster = (seed_idx, card_id). returns per-cluster & per-person tables."""
    rows = []                        # (cluster_key, member, correct)
    for si, d in enumerate(neu_dirs):
        for chan, member, card, correct in load_rows_robust(d):
            if chan != "neutral":
                continue
            rows.append(((si, card), member, correct))
    n = len(rows)
    acc = np.mean([c for *_x, c in rows]) if n else float("nan")
    byc = defaultdict(list)
    bym = defaultdict(list)
    for ckey, member, c in rows:
        byc[ckey].append(c)
        bym[member].append(c)
    caccs = {k: float(np.mean(v)) for k, v in byc.items()}
    csizes = {k: len(v) for k, v in byc.items()}
    cvals = np.array(list(caccs.values()))
    nbar = float(np.mean(list(csizes.values()))) if csizes else float("nan")
    sd_binom = math.sqrt(acc * (1 - acc) / nbar) if 0 < acc < 1 and nbar else 0.0
    # leaky clusters (Wilson-lower>0.5)
    leaky_c = []
    for k, v in byc.items():
        lo, hi = wilson(int(sum(v)), len(v))
        if lo > 0.5:
            leaky_c.append((f"s{k[0]}:{k[1]}", float(np.mean(v)), len(v), round(lo, 3)))
    leaky_c.sort(key=lambda x: -x[1])
    # leaky persons (Wilson-lower>0.5, ~6 trials each across seeds -> robust to co-members)
    paccs = {m: float(np.mean(v)) for m, v in bym.items()}
    leaky_p = []
    for m, v in bym.items():
        lo, hi = wilson(int(sum(v)), len(v))
        if lo > 0.5:
            leaky_p.append((m, float(np.mean(v)), len(v), round(lo, 3)))
    leaky_p.sort(key=lambda x: -x[1])
    return {"n": n, "acc": float(acc), "nbar": nbar,
            "caccs": caccs, "csizes": csizes, "n_clusters": len(byc),
            "cluster_sd": float(cvals.std()) if len(cvals) else float("nan"),
            "cluster_min": float(cvals.min()) if len(cvals) else float("nan"),
            "cluster_max": float(cvals.max()) if len(cvals) else float("nan"),
            "sd_binom_expected": sd_binom,
            "paccs": paccs, "n_persons": len(bym),
            "leaky_clusters": leaky_c, "leaky_persons": leaky_p,
            "cluster_members": {k: sorted({m for kk, m, _c in rows if kk == k}) for k in byc}}


def main_fixed():
    out = {}
    for name, neu_dirs, base_dir in FIXED:
        if not all((d / "meta.json").exists() for d in neu_dirs):
            print(f"[skip] {name}: missing a neufix seed under {[str(d) for d in neu_dirs]}")
            continue
        st = fixed_stats(neu_dirs)
        indiv = indiv_by_person(base_dir) if (base_dir / "meta.json").exists() else {}
        indiv_mean = float(np.mean(list(indiv.values()))) if indiv else float("nan")
        lo, hi = wilson(int(round(st["acc"] * st["n"])), st["n"])
        sd_ratio = st["cluster_sd"] / st["sd_binom_expected"] if st["sd_binom_expected"] else float("nan")
        # cross-channel: per-PERSON indiv acc vs neutral acc (sharpest identity-stability test)
        common_p = [m for m in st["paccs"] if m in indiv]
        if len(common_p) >= 3:
            xs = np.array([indiv[m] for m in common_p]); ys = np.array([st["paccs"][m] for m in common_p])
            r_person = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() and ys.std() else float("nan")
        else:
            r_person = float("nan")
        # cross-channel: per-CLUSTER mean-indiv-of-members vs neutral acc
        cl_x, cl_y = [], []
        for ckey, members in st["cluster_members"].items():
            mi = [indiv[m] for m in members if m in indiv]
            if mi:
                cl_x.append(float(np.mean(mi))); cl_y.append(st["caccs"][ckey])
        if len(cl_x) >= 3 and np.std(cl_x) and np.std(cl_y):
            r_cluster = float(np.corrcoef(cl_x, cl_y)[0, 1])
        else:
            r_cluster = float("nan")

        print(f"\n===== {name.upper()}  (neutral FIXED CMD card, k8, seeds s0+s1+s2 pooled) =====")
        print(f"  MEAN acc = {st['acc']:.3f}  Wilson95 CI[{lo:.3f},{hi:.3f}]  "
              f"(n={st['n']} trials, {st['n_persons']} persons, {st['n_clusters']} pooled cards over 3 seeds)")
        print(f"  per-CLUSTER acc: SD {st['cluster_sd']:.3f}  range [{st['cluster_min']:.3f},{st['cluster_max']:.3f}]  "
              f"(binom-noise SD {st['sd_binom_expected']:.3f} at nbar={st['nbar']:.0f}; observed/expected = {sd_ratio:.2f}x)")
        if st["leaky_clusters"]:
            tops = "; ".join(f"{c} acc={a:.2f}(n={n},lo={l})" for c, a, n, l in st["leaky_clusters"][:4])
            print(f"  LEAKY CLUSTERS: {len(st['leaky_clusters'])}/{st['n_clusters']} Wilson-lo>0.5 -> {tops}")
        else:
            print(f"  LEAKY CLUSTERS: 0/{st['n_clusters']} individually CI-above-0.5 (tail = spread, not per-cluster sig at this n)")
        if st["leaky_persons"]:
            tops = "; ".join(f"{m} acc={a:.2f}(n={n},lo={l})" for m, a, n, l in st["leaky_persons"][:5])
            print(f"  LEAKY PERSONS: {len(st['leaky_persons'])}/{st['n_persons']} Wilson-lo>0.5 across 3 pooling contexts -> {tops}")
        else:
            print(f"  LEAKY PERSONS: 0/{st['n_persons']} individually CI-above-0.5 (no person robustly re-id'd across co-member sets)")
        print(f"  tail STABLE? corr(indiv acc, neutral acc): per-PERSON r={r_person:+.2f}  per-CLUSTER r={r_cluster:+.2f}  "
              f"(indiv mean {indiv_mean:.3f}) -> {'identity-rich persons stay the leaky ones' if (r_person==r_person and r_person>0.3) else 'weak/no cross-channel structure'}")
        out[name] = {
            "neutral_mean": round(st["acc"], 4), "neutral_ci": [round(lo, 4), round(hi, 4)],
            "n_trials": st["n"], "n_persons": st["n_persons"], "n_clusters": st["n_clusters"], "nbar": round(st["nbar"], 2),
            "cluster_sd": round(st["cluster_sd"], 4), "cluster_range": [round(st["cluster_min"], 4), round(st["cluster_max"], 4)],
            "sd_binom_expected": round(st["sd_binom_expected"], 4), "sd_ratio_obs_over_binom": round(sd_ratio, 3),
            "n_leaky_clusters": len(st["leaky_clusters"]), "leaky_clusters": st["leaky_clusters"],
            "n_leaky_persons": len(st["leaky_persons"]), "leaky_persons": st["leaky_persons"][:12],
            "indiv_mean": round(indiv_mean, 4),
            "corr_indiv_neutral_person": round(r_person, 3), "corr_indiv_neutral_cluster": round(r_cluster, 3),
        }
    (ROOT / "results" / "dispersion_2afc_fixed.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> results/dispersion_2afc_fixed.json")


def main():
    out = {}
    for name, d in DIRS:
        if not (d / "meta.json").exists():
            print(f"[skip] {name}: no meta at {d}")
            continue
        rows = load(d)
        sh = chan_stats(rows, "shared")
        iv = chan_stats(rows, "indiv")
        # is the leaky tail STABLE / identity-driven? per-cluster indiv-acc vs shared-acc correlation
        common = [c for c in sh["caccs"] if c in iv["caccs"]]
        if len(common) >= 3:
            xs = np.array([iv["caccs"][c] for c in common]); ys = np.array([sh["caccs"][c] for c in common])
            r_cluster = float(np.corrcoef(xs, ys)[0, 1]) if xs.std() and ys.std() else float("nan")
        else:
            r_cluster = float("nan")
        lo, hi = wilson(int(round(sh["acc"] * sh["n"])), sh["n"])
        sd_ratio = sh["cluster_sd"] / sh["sd_binom_expected"] if sh["sd_binom_expected"] else float("nan")

        print(f"\n===== {name.upper()}  (shared pooled card, k8) =====")
        print(f"  MEAN acc = {sh['acc']:.3f}  Wilson95 CI[{lo:.3f},{hi:.3f}]  (n={sh['n']} trials, {sh['n_persons']} persons, {sh['n_clusters']} clusters)")
        print(f"  per-CLUSTER acc: mean {sh['acc']:.3f}  SD {sh['cluster_sd']:.3f}  range [{sh['cluster_min']:.3f},{sh['cluster_max']:.3f}]  "
              f"(binom-noise SD would be {sh['sd_binom_expected']:.3f}; observed/expected = {sd_ratio:.2f}x)")
        print(f"  per-PERSON (2 trials each): fully-ID'd both trials = {sh['p_full']*100:.0f}%  (chance 25%),  0/2 = {sh['p_zero']*100:.0f}% (chance 25%)")
        if sh["leaky_clusters"]:
            tops = "; ".join(f"{c} acc={a:.2f} (n={n}, lo={lo3})" for c, a, n, lo3 in sh["leaky_clusters"][:4])
            print(f"  LEAKY TAIL: {len(sh['leaky_clusters'])}/{sh['n_clusters']} clusters with Wilson-lower>0.5 -> {tops}")
        else:
            print(f"  LEAKY TAIL: 0/{sh['n_clusters']} clusters individually CI-above-0.5 (tail visible as spread, not per-cluster significance at this n)")
        print(f"  tail STABLE? per-cluster corr(indiv acc, shared acc) r = {r_cluster:+.2f}  "
              f"(indiv mean {iv['acc']:.3f}) -> {'identity-rich clusters stay the leaky ones' if r_cluster>0.3 else 'weak/no cross-channel structure'}")
        out[name] = {"shared_mean": round(sh["acc"], 4), "shared_ci": [round(lo, 4), round(hi, 4)],
                     "cluster_sd": round(sh["cluster_sd"], 4), "cluster_range": [round(sh["cluster_min"], 4), round(sh["cluster_max"], 4)],
                     "sd_binom_expected": round(sh["sd_binom_expected"], 4), "sd_ratio_obs_over_binom": round(sd_ratio, 3),
                     "person_full_id_frac": round(sh["p_full"], 4), "person_full_id_chance": 0.25,
                     "n_leaky_clusters": len(sh["leaky_clusters"]), "n_clusters": sh["n_clusters"],
                     "leaky_clusters": sh["leaky_clusters"],
                     "indiv_mean": round(iv["acc"], 4), "cluster_corr_indiv_shared": round(r_cluster, 3),
                     "n_trials": sh["n"], "n_persons": sh["n_persons"]}
    (ROOT / "results" / "dispersion_2afc.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> results/dispersion_2afc.json")


if __name__ == "__main__":
    if MODE == "fixed":
        main_fixed()
    else:
        main()
