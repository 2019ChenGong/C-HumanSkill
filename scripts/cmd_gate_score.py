"""Score the CMD gate: per-cluster chance, paired Delta(card-raw) cluster-bootstrap CI, absolute positive-control lift floor, three-outcome decision."""
import os
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.attrib_metrics import cluster_paired_diff_ci, cluster_mean_ci  # noqa: E402

DATASET = os.environ.get("DATASET", "enron")
RES = ROOT / "results" if DATASET == "enron" else ROOT / "results" / DATASET
# absolute positive-control floor (k-scale-invariant): the indiv card must recover a non-trivial fraction of the
# identifiability HEADROOM above chance — normalized lift = (re-id − chance)/(1 − chance) >= MIN_LIFT. A multiplicative
# "x×chance" floor is unusable across k (at k=4, 3×chance=0.75≈perfect; at k=16, 3×chance=0.19). If indiv can't clear
# MIN_LIFT the k-way lineup is too hard to tell a SAFE card from a STARVED attacker -> gate INCONCLUSIVE at that k.
MIN_LIFT = float(os.environ.get("POSCTRL_MIN_LIFT", 0.10))


def load_cond(k, s, cond):
    """Return (rows, n_key, n_pick): rows = {author: {'hit':0/1, 'cluster':id}} for trials present in BOTH key & picks."""
    kp = RES / f"_cmdgate_k{k}_s{s}_{cond}_key.json"
    pp = RES / f"_picks_cmdgate_k{k}_s{s}_{cond}.json"
    if not kp.exists():
        return None, 0, 0
    key = json.loads(kp.read_text())
    picks = json.loads(pp.read_text()) if pp.exists() else {}
    rows = {}
    for t, meta in key.items():
        if t in picks:
            rows[meta["author"]] = {"hit": int(picks[t] == meta["true_candidate"]), "cluster": meta["cluster"],
                                    "size": meta.get("cluster_size", None)}
    return rows, len(key), len(picks)


def eff_chance(rows, authors, k):
    """Author-weighted random-guess accuracy = mean(1/lineup_size). Tail-merge makes one cluster ~2k, so its members
    truly face 1/(2k), not 1/k; using the global 1/k over-credits them. Falls back to 1/k if sizes are absent."""
    sizes = [rows[a].get("size") for a in authors]
    if any(s is None for s in sizes):
        return 1.0 / k
    return float(np.mean([1.0 / s for s in sizes]))


def discover():
    ks, seeds = set(), set()
    for f in RES.glob("_cmdgate_k*_s*_card_key.json"):
        parts = f.stem.split("_")               # ['', 'cmdgate', 'k4', 's0', 'card', 'key']
        ks.add(int(parts[2][1:])); seeds.add(int(parts[3][1:]))
    return sorted(ks), sorted(seeds)


def reid(rows, authors):
    hits = [rows[a]["hit"] for a in authors]
    cl = [rows[a]["cluster"] for a in authors]
    ci = cluster_mean_ci(hits, cl)              # resample CLUSTERS, not authors
    return float(np.mean(hits)), ci, len(set(cl))


def main():
    ks, seeds = discover()
    if not ks:
        print("no _cmdgate_*_card_key.json found — run cmd_gate.py dump + Opus subagents first."); return
    print(f"CMD Step-0 GATE scoring — k∈{ks}, seeds∈{seeds}  (CIs resample CLUSTERS = n_eff)\n")
    summary = {}
    for k in ks:
        chance = 1.0 / k
        per_seed = []
        for s in seeds:
            card, nk_c, np_c = load_cond(k, s, "card")
            raw, nk_r, np_r = load_cond(k, s, "raw")
            indiv, nk_i, np_i = load_cond(k, s, "indiv")
            if not card or not raw:
                continue
            # COMPLETENESS: warn loudly if any condition's picks are incomplete (missing picks bias re-id upward)
            incomplete = [(c, nk, npk) for c, nk, npk in [("card", nk_c, np_c), ("raw", nk_r, np_r),
                          ("indiv", nk_i, np_i)] if npk < nk]
            if incomplete:
                print(f"  ⚠ k={k} s={s} INCOMPLETE picks: " +
                      ", ".join(f"{c} {npk}/{nk}" for c, nk, npk in incomplete) +
                      "  -> re-id may be biased; treat row as PROVISIONAL", flush=True)
            authors = [a for a in card if a in raw]          # paired set (card ∩ raw)
            ch = eff_chance(card, authors, k)                 # author-weighted 1/lineup_size (tail-merge aware)
            pc, cic, ncl = reid(card, authors); pr, cir, _ = reid(raw, authors)
            hc = [card[a]["hit"] for a in authors]; hr = [raw[a]["hit"] for a in authors]
            cls = [card[a]["cluster"] for a in authors]
            dlt = cluster_paired_diff_ci(hc, hr, cls)        # card − raw, paired, resampling clusters
            row = {"k": k, "seed": s, "n": len(authors), "n_clusters": ncl, "incomplete": bool(incomplete),
                   "chance": round(ch, 4),
                   "card": round(pc, 3), "card_ci": cic, "raw": round(pr, 3), "raw_ci": cir,
                   "delta": dlt["diff"], "delta_ci": dlt["ci"], "delta_p": dlt["p"]}
            if indiv:
                ai = [a for a in indiv if a in card]
                pi, cii, _ = reid(indiv, ai)
                row["indiv"] = round(pi, 3); row["indiv_lo"] = cii[0]
                lift = (pi - ch) / (1 - ch) if ch < 1 else 0.0     # normalized lift above chance (k-scale-invariant)
                row["indiv_lift"] = round(lift, 3)
                row["pos_abs_ok"] = bool(lift >= MIN_LIFT)         # absolute floor: attacker COMPETENT, not merely SIG
            per_seed.append(row)
            ind = (f" indiv={row.get('indiv','—')}(lo {row.get('indiv_lo','—')}, "
                   f"lift={row.get('indiv_lift','—')}{'OK' if row.get('pos_abs_ok') else ' STARVED'})") if "indiv" in row else ""
            print(f"  k={k} s={s} n={len(authors)} n_clusters={ncl} eff_chance={ch:.3f}: "
                  f"card={pc:.3f}{cic}  raw={pr:.3f}{cir}  Δ(card−raw)={dlt['diff']:+.3f} CI{dlt['ci']} p={dlt['p']}{ind}")
        if not per_seed:
            continue
        raw_leaks = [r["raw_ci"][0] > r["chance"] for r in per_seed]               # per-seed eff_chance, not global 1/k
        delta_pos = [r["delta_ci"][0] > 0 for r in per_seed]                       # Δ CI excludes 0, positive
        delta_zero = [r["delta_ci"][0] <= 0 <= r["delta_ci"][1] for r in per_seed]  # Δ CI ∋ 0
        delta_neg = [r["delta_ci"][1] < 0 for r in per_seed]                       # Δ CI excludes 0, negative (card SAFER)
        pos = [r.get("indiv_lo", 0) > r["chance"] for r in per_seed if "indiv" in r]            # relative: SIG > chance
        pos_abs = [r.get("pos_abs_ok", False) for r in per_seed if "indiv" in r]                # absolute: >= FLOOR_MULT×chance
        stable = lambda xs: all(xs) or not any(xs)
        print(f"  --- k={k} decision across {len(per_seed)} seeds (n_clusters≈{per_seed[0]['n_clusters']}) ---")
        print(f"      raw_lo>chance: {sum(raw_leaks)}/{len(raw_leaks)} | Δ>0(stop): {sum(delta_pos)}/{len(delta_pos)} | "
              f"Δ∋0: {sum(delta_zero)}/{len(delta_zero)} | Δ<0(safer): {sum(delta_neg)}/{len(delta_neg)} | "
              f"pos-ctrl SIG>ch: {sum(pos)}/{len(pos) if pos else 0} | pos-ctrl lift>={MIN_LIFT:g}: {sum(pos_abs)}/{len(pos_abs) if pos_abs else 0}")
        if any(r["incomplete"] for r in per_seed):
            print("      (some seeds have INCOMPLETE picks — verdict PROVISIONAL until pick files complete)")
        if pos_abs and not all(pos_abs):
            verdict = (f"POS-CTRL STARVED — indiv normalized lift < {MIN_LIFT:g} on some seed: the k-way lineup is too hard "
                       "to tell a SAFE card from a STARVED attacker. Gate INCONCLUSIVE at this k -> lean on lineup-free open-world AUC")
        elif any(delta_pos):
            verdict = "(iii) STOP-AND-FIX — identical card still leaks (Δ CI excludes 0 positive) on some seed: bug or ε≠0"
        elif pos and not all(pos):
            verdict = "POS-CTRL FAIL on some seed — attacker too weak there; result void until fixed"
        elif not all(delta_zero) and not any(delta_neg):
            verdict = "Δ not cleanly ∋0 nor <0 — inspect per-seed Δ before deciding"
        elif not stable(raw_leaks):
            verdict = "UNSTABLE — raw-floor decision flips across seeds; NOT a decision (add seeds / inspect clustering)"
        elif all(raw_leaks):
            verdict = "(ii) raw already leaks within cluster -> headline weakens to 'card adds no leak' (G3/T1). PROCEED-REFRAMED"
        else:
            note = " [Δ<0: card is SAFER than raw floor]" if all(delta_neg) else ""
            verdict = f"(i) card was binding (raw <= 1/k) and Δ not-positive -> CMD strong in T2. PROCEED to main experiment{note}"
        print(f"      VERDICT k={k}: {verdict}\n")
        summary[f"k{k}"] = {"per_seed": per_seed, "verdict": verdict}
    (RES / "cmd_gate_result.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("saved -> results/cmd_gate_result.json")


if __name__ == "__main__":
    main()
