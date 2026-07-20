"""t-closeness distance of pooled cards (P1/#47, $0, no spend) — earn-or-drop the "CMD = t-closeness refinement" analogy.

Classic t-closeness (Li et al. 2007): an equivalence class satisfies t-closeness if the distance between the
distribution of the sensitive attribute IN THE CLASS and the distribution in the WHOLE TABLE (population base-rate)
is <= t. The release satisfies t-closeness at t = the MAX over classes.

Mapping to our setting:
  equivalence class            = a pool of k people
  released class representation = the pooled card (concat = union summary / CMD = consensus)
  class attribute distribution  = the pooled card's content-word distribution
  population / common reference = the base-rate content distribution over ALL individual member cards (aggro)
  t                             = distance(pooled-card dist, population dist)

Two complementary $0 measurements over the SAME content representation the paper already uses (de._content_vec:
lowercase content words, PII placeholders + stopwords dropped):

  (A) t-closeness proper  t = D(pooled card, population base-rate).  LOWER = more t-close = reveals less about
      which specific members are inside.  Report mean AND max (the t parameter = worst class).
  (B) member-ID AUC       for a pooled card, rank ALL authors by content similarity to the card; AUC of
      "true member vs non-member".  1.0 = the card's distribution tracks its own members (leaks membership,
      = naive k-anonymity that republishes members); 0.5 = card equally close to members and strangers
      (t-close / anonymous).  This is the DISTRIBUTIONAL analog of our LLM linkage / 2AFC own~stranger result.

Prediction if the analogy is real: CMD has SMALLER t and member-ID AUC CLOSER to 0.5 than concat.
If not, we DROP the analogy and say so. Honesty > narrative.

  DATASET=mad python scripts/cmd_tcloseness.py
"""
import os
import sys
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GROUP", "random")
os.environ.setdefault("DATASET", "mad")
import deid_enron as de  # noqa: E402  (reuse _content_vec / _cosine — same content metric as the rest of the paper)
import cmd_gate as CG    # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DS = os.environ.get("DATASET", "mad")
RES = ROOT / "results" if DS == "enron" else ROOT / "results" / DS
KS = [int(x) for x in os.environ.get("KS", "4,8").split(",")]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
# data dir follows the dataset (MAD hardcoded historically); CV=data/se, Enron=data/enron
DATADIR = ROOT / "data" / {"mad": "20mad", "enron": "enron", "se": "se", "cv": "se"}.get(DS, DS)
SHARED = DATADIR / os.environ.get("SHARED", "cmd_shared_cards_mad.json")
CONCAT = DATADIR / os.environ.get("CONCAT", "cmd_concat_cards_mad.json")


def _norm(counter):
    tot = sum(counter.values())
    return {w: c / tot for w, c in counter.items()} if tot else {}


def _jsd(p, q):
    """Jensen-Shannon divergence (log base 2, range [0,1]); the equal-ground-distance special case of EMD."""
    vocab = set(p) | set(q)
    m = {w: 0.5 * (p.get(w, 0.0) + q.get(w, 0.0)) for w in vocab}

    def _kl(a):
        s = 0.0
        for w in vocab:
            pa = a.get(w, 0.0)
            if pa > 0:
                s += pa * math.log2(pa / m[w])
        return s
    return 0.5 * _kl(p) + 0.5 * _kl(q)


def _auc(pos, neg):
    """P(a random pos scores above a random neg); ties count 0.5. pos/neg are similarity scores."""
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for a in pos:
        for b in neg:
            if a > b:
                wins += 1
            elif a == b:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    _d, authors, _n, aggro, _r, _rt = CG.load()
    shared = json.loads(SHARED.read_text(encoding="utf-8"))
    concat = json.loads(CONCAT.read_text(encoding="utf-8"))

    # per-author individual-card content vectors + normalized distributions (the population data)
    ivec = {a: de._content_vec(aggro[a]) for a in authors}
    idist = {a: _norm(ivec[a]) for a in authors}

    # population base-rate = aggregate content distribution over ALL individual cards (the "whole table")
    Qc = Counter()
    for a in authors:
        Qc.update(ivec[a])
    Qvec = dict(Qc)
    Qdist = _norm(Qc)

    caches = {"concat": concat, "cmd": shared}
    rows = []
    per_card = defaultdict(list)   # (method,k) -> list of {t_jsd, t_cos, midauc}

    for method, cache in caches.items():
        for k in KS:
            for s in SEEDS:
                grp, byc = CG.make_groups(aggro, authors, k, s)
                for cid, mem in byc.items():
                    if len(mem) < k:
                        continue
                    ck = f"k{k}_s{s}_{cid}"
                    txt = cache.get(ck)
                    if not txt:
                        continue
                    pv = de._content_vec(txt)
                    pd = _norm(pv)
                    members = set(mem)
                    # (A) t-closeness: distance of the pooled card to the population base-rate
                    t_jsd = _jsd(pd, Qdist)
                    t_cos = 1.0 - de._cosine(pv, Qvec)
                    # (B) member-ID AUC: does card-to-author similarity rank true members above strangers?
                    sim = {a: de._cosine(pv, ivec[a]) for a in authors}
                    pos = [sim[a] for a in authors if a in members]
                    neg = [sim[a] for a in authors if a not in members]
                    midauc = _auc(pos, neg)
                    # own vs stranger mean cosine distance (interpretable companion to the AUC)
                    own = float(np.mean([1 - sim[a] for a in authors if a in members]))
                    oth = float(np.mean([1 - sim[a] for a in authors if a not in members]))
                    per_card[(method, k)].append(
                        {"t_jsd": t_jsd, "t_cos": t_cos, "midauc": midauc, "own": own, "oth": oth})

    def agg(lst, key):
        v = np.array([x[key] for x in lst if not math.isnan(x[key])])
        return v

    print(f"=== t-closeness of pooled cards  DS={DS}  seeds={SEEDS} ===")
    print(f"population base-rate = content dist over {len(authors)} individual cards; "
          f"metric: JSD (faithful t-closeness) + cosine-dist (paper content metric)\n")
    hdr = f"{'method':6} k  n   | t_JSD mean/max      t_cos mean/max     | member-ID AUC mean [dispersion]   own/stranger cos-dist"
    print(hdr); print("-" * len(hdr))
    summary = {}
    for method in ("concat", "cmd"):
        for k in KS:
            lst = per_card[(method, k)]
            if not lst:
                continue
            tj = agg(lst, "t_jsd"); tc = agg(lst, "t_cos"); mid = agg(lst, "midauc")
            own = agg(lst, "own"); oth = agg(lst, "oth")
            summary[f"{method}_k{k}"] = {
                "n_cards": len(lst),
                "t_jsd_mean": round(float(tj.mean()), 4), "t_jsd_max": round(float(tj.max()), 4),
                "t_cos_mean": round(float(tc.mean()), 4), "t_cos_max": round(float(tc.max()), 4),
                "member_id_auc_mean": round(float(mid.mean()), 4),
                "member_id_auc_p10": round(float(np.percentile(mid, 10)), 4),
                "member_id_auc_p90": round(float(np.percentile(mid, 90)), 4),
                "own_cosdist_mean": round(float(own.mean()), 4),
                "stranger_cosdist_mean": round(float(oth.mean()), 4),
                "own_minus_stranger": round(float(own.mean() - oth.mean()), 4)}
            print(f"{method:6} {k:<2} {len(lst):<3} | "
                  f"{tj.mean():.3f}/{tj.max():.3f}        {tc.mean():.3f}/{tc.max():.3f}       | "
                  f"{mid.mean():.3f} [{np.percentile(mid,10):.3f},{np.percentile(mid,90):.3f}]"
                  f"          {own.mean():.3f}/{oth.mean():.3f}")

    # verdict per k: is CMD more t-close (lower t) AND less member-identifying (AUC closer .5) than concat?
    print("\n--- verdict (CMD vs concat) ---")
    verdicts = {}
    for k in KS:
        c = summary.get(f"concat_k{k}"); m = summary.get(f"cmd_k{k}")
        if not (c and m):
            continue
        d_t = m["t_cos_mean"] - c["t_cos_mean"]
        d_mid = abs(m["member_id_auc_mean"] - 0.5) - abs(c["member_id_auc_mean"] - 0.5)
        earn = (m["t_cos_mean"] <= c["t_cos_mean"] + 1e-9) and (abs(m["member_id_auc_mean"] - 0.5) <= abs(c["member_id_auc_mean"] - 0.5) + 1e-9)
        verdicts[f"k{k}"] = {"delta_t_cos_cmd_minus_concat": round(d_t, 4),
                             "delta_member_auc_dev_from_half": round(d_mid, 4),
                             "cmd_more_t_close": earn}
        print(f"  k={k}: Δt_cos(CMD−concat)={d_t:+.4f}  (CMD {'closer to base-rate' if d_t<0 else 'FARTHER'});  "
              f"member-ID AUC concat {c['member_id_auc_mean']:.3f} vs CMD {m['member_id_auc_mean']:.3f}  "
              f"-> {'EARNED (CMD more t-close)' if earn else 'NOT earned'}")

    out = {"dataset": DS, "seeds": SEEDS, "population_n_authors": len(authors),
           "metric_note": "t = distance(pooled card, base-rate over all individual cards); JSD + cosine-dist. "
                          "member_id_auc = AUC ranking true members above strangers by card-to-author cosine.",
           "by_arm": summary, "verdict": verdicts}
    outp = RES / os.environ.get("OUT", "tcloseness.json")
    outp.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {outp.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
