"""Pooled TPR@low-FPR for the CMD membership-inference attack — the realistic, topic-controlled identity metric
that complements ROC-AUC (Carlini et al. 2022, "MIA From First Principles": report TPR@low-FPR, not just AUC).

For each (picks, key) pair we pool every candidate across all trials, split by true label (pos / rneg / nneg),
then for a chosen negative type compute TPR at FPR<=target:
  FPR(t) = frac(neg >= t)   TPR(t) = frac(pos >= t)   TPR@X% = max TPR over thresholds with FPR<=X%
Chance => TPR ~= FPR (so TPR@5% ~= 5%). rneg = random strangers (topic gameable); nneg = same-topic strangers
(topic held constant => isolates IDENTITY; the headline). Also prints pooled ROC-AUC for reference.

Reproduces the §5.5 ④ 8-cell table. Run: python scripts/cmd_tpr.py
"""
import os
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FPRS = (0.01, 0.05, 0.10)

# (label, picks, key) — the 8 CMD-shared cells + per-person leak references. picks/key relative to repo root.
CELLS = [
    ("Enron  CMD   Opus  k4", "results/_picks_ow_k4_shared_s0.json", "results/_ow_k4_shared_s0_key.json"),
    ("Enron  CMD   Opus  k8", "results/_picks_ow_k8_shared_s0.json", "results/_ow_k8_shared_s0_key.json"),
    ("Enron  CMD   gpt55 k4", "results/enron_gpt55/_picks_ow_k4_shared_s0.json", "results/enron_gpt55/_ow_k4_shared_s0_key.json"),
    ("Enron  CMD   gpt55 k8", "results/enron_gpt55_k8/_picks_ow_k8_shared_s0.json", "results/enron_gpt55_k8/_ow_k8_shared_s0_key.json"),
    ("Enron  indiv Opus  k4", "results/_picks_ow_k4_indiv_s0.json", "results/_ow_k4_indiv_s0_key.json"),
    ("20MAD  CMD   Opus  k4", "results/mad/_picks_ow_k4_shared_s0.json", "results/mad/_ow_k4_shared_s0_key.json"),
    ("20MAD  CMD   Opus  k8", "results/mad/_picks_ow_k8_shared_s0.json", "results/mad/_ow_k8_shared_s0_key.json"),
    ("20MAD  CMD   gpt55 k4", "results/mad/gpt55/_picks_ow_k4_shared_s0.json", "results/mad/gpt55/_ow_k4_shared_s0_key.json"),
    ("20MAD  CMD   gpt55 k8", "results/mad/gpt55_k8/_picks_ow_k8_shared_s0.json", "results/mad/gpt55_k8/_ow_k8_shared_s0_key.json"),
    ("20MAD  indiv Opus  k4", "results/mad/_picks_ow_k4_indiv_s0.json", "results/mad/_ow_k4_indiv_s0_key.json"),
    # ---- method comparison @ headline Opus k4 (per-person de-id baselines; k-independent) ----
    ("Enron  staab    Opus k4", "results/_picks_ow_k4_staab_s0.json", "results/_ow_k4_staab_s0_key.json"),
    ("Enron  staab_r1 Opus k4", "results/_picks_ow_k4_staab_r1_s0.json", "results/_ow_k4_staab_r1_s0_key.json"),
    ("Enron  presidio Opus k4", "results/_picks_ow_k4_presidio_s0.json", "results/_ow_k4_presidio_s0_key.json"),
    ("Enron  tpar_t10 Opus k4", "results/_picks_ow_k4_tpar_t10_s0.json", "results/_ow_k4_tpar_t10_s0_key.json"),
    ("Enron  tpar_t15 Opus k4", "results/_picks_ow_k4_tpar_t15_s0.json", "results/_ow_k4_tpar_t15_s0_key.json"),
    ("Enron  petre    Opus k4", "results/_picks_ow_k4_petre_k4_s0.json", "results/_ow_k4_petre_k4_s0_key.json"),
    ("20MAD  staab    Opus k4", "results/mad/_picks_ow_k4_staab_s0.json", "results/mad/_ow_k4_staab_s0_key.json"),
    ("20MAD  staab_r1 Opus k4", "results/mad/_picks_ow_k4_staab_r1_s0.json", "results/mad/_ow_k4_staab_r1_s0_key.json"),
    ("20MAD  tpar_t10 Opus k4", "results/mad/_picks_ow_k4_tpar_t10_s0.json", "results/mad/_ow_k4_tpar_t10_s0_key.json"),
    ("20MAD  tpar_t15 Opus k4", "results/mad/_picks_ow_k4_tpar_t15_s0.json", "results/mad/_ow_k4_tpar_t15_s0_key.json"),
    ("20MAD  petre    Opus k4", "results/mad/_picks_ow_k4_petre_k4_s0.json", "results/mad/_ow_k4_petre_k4_s0_key.json"),
    # ---- per-person arms scored at Opus k8 (k-independent; k4/k8 diff = scoring noise) ----
    ("Enron  indiv Opus  k8", "results/_picks_ow_k8_indiv_s0.json", "results/_ow_k8_indiv_s0_key.json"),
    ("20MAD  indiv Opus  k8", "results/mad/_picks_ow_k8_indiv_s0.json", "results/mad/_ow_k8_indiv_s0_key.json"),
    # ---- gpt-5.5 k4 secondary attacker (subset of arms; noisy/unreliable, direction-check only) ----
    ("Enron  indiv    gpt55 k4", "results/enron_gpt55/_picks_ow_k4_indiv_s0.json", "results/enron_gpt55/_ow_k4_indiv_s0_key.json"),
    ("Enron  staab    gpt55 k4", "results/enron_gpt55/_picks_ow_k4_staab_s0.json", "results/enron_gpt55/_ow_k4_staab_s0_key.json"),
    ("Enron  tpar_t10 gpt55 k4", "results/enron_gpt55/_picks_ow_k4_tpar_t10_s0.json", "results/enron_gpt55/_ow_k4_tpar_t10_s0_key.json"),
    ("Enron  petre    gpt55 k4", "results/enron_gpt55/_picks_ow_k4_petre_k4_s0.json", "results/enron_gpt55/_ow_k4_petre_k4_s0_key.json"),
    ("20MAD  indiv    gpt55 k4", "results/mad/gpt55/_picks_ow_k4_indiv_s0.json", "results/mad/gpt55/_ow_k4_indiv_s0_key.json"),
    ("20MAD  staab    gpt55 k4", "results/mad/gpt55/_picks_ow_k4_staab_s0.json", "results/mad/gpt55/_ow_k4_staab_s0_key.json"),
    ("20MAD  tpar_t10 gpt55 k4", "results/mad/gpt55/_picks_ow_k4_tpar_t10_s0.json", "results/mad/gpt55/_ow_k4_tpar_t10_s0_key.json"),
    ("20MAD  petre    gpt55 k4", "results/mad/gpt55/_picks_ow_k4_petre_k4_s0.json", "results/mad/gpt55/_ow_k4_petre_k4_s0_key.json"),
    # ---- gap-fill: gpt-5.5 k4 on the remaining de-id arms (staab_r1/tpar_t15/presidio) ----
    ("Enron  staab_r1 gpt55 k4", "results/enron_gpt55/_picks_ow_k4_staab_r1_s0.json", "results/enron_gpt55/_ow_k4_staab_r1_s0_key.json"),
    ("Enron  tpar_t15 gpt55 k4", "results/enron_gpt55/_picks_ow_k4_tpar_t15_s0.json", "results/enron_gpt55/_ow_k4_tpar_t15_s0_key.json"),
    ("Enron  presidio gpt55 k4", "results/enron_gpt55/_picks_ow_k4_presidio_s0.json", "results/enron_gpt55/_ow_k4_presidio_s0_key.json"),
    ("20MAD  staab_r1 gpt55 k4", "results/mad/gpt55/_picks_ow_k4_staab_r1_s0.json", "results/mad/gpt55/_ow_k4_staab_r1_s0_key.json"),
    ("20MAD  tpar_t15 gpt55 k4", "results/mad/gpt55/_picks_ow_k4_tpar_t15_s0.json", "results/mad/gpt55/_ow_k4_tpar_t15_s0_key.json"),
]


def load(picks, key):
    P = json.loads((ROOT / picks).read_text(encoding="utf-8-sig"))  # tolerate BOM (gpt-5.5 sub-agent output)
    K = json.loads((ROOT / key).read_text(encoding="utf-8"))
    pos, rneg, nneg = [], [], []
    for t, kk in K.items():
        if t not in P:
            continue
        for s, l in kk["labels"].items():
            if s not in P[t]:
                continue
            (pos if l == "pos" else rneg if l == "rneg" else nneg).append(float(P[t][s]))
    return np.array(pos), np.array(rneg), np.array(nneg)


def tpr_at(pos, neg, target):
    best = (0.0, 1.0)
    for s in sorted(set(neg) | set(pos)):
        fpr = float(np.mean(neg >= s)); tpr = float(np.mean(pos >= s))
        if fpr <= target and tpr > best[0]:
            best = (tpr, fpr)
    return best[0]


def main():
    print("Pooled TPR@{1,5,10}% FPR (chance=1/5/10). nneg=same-topic(IDENTITY headline), rneg=random(aux).")
    print("AUC column lives in the official scorer (cluster-bootstrap): DATASET=.. RESDIR=.. KCL=.. cmd_openworld_score.py")
    print("NB: only 4 neg/trial -> @1% is coarse (~5 samples define the 1% line); read @5%/@10%.\n")
    hdr = f"{'cell':22s} | {'nneg TPR@1/5/10':>16s} | {'rneg TPR@1/5/10':>16s}"
    print(hdr); print("-" * len(hdr))
    for label, picks, key in CELLS:
        try:
            pos, rneg, nneg = load(picks, key)
        except FileNotFoundError:
            print(f"{label:22s} | (missing file)"); continue
        nt = "/".join(f"{tpr_at(pos, nneg, f)*100:.0f}" for f in FPRS)
        rt = "/".join(f"{tpr_at(pos, rneg, f)*100:.0f}" for f in FPRS)
        print(f"{label:22s} | {nt:>16s} | {rt:>16s}")


if __name__ == "__main__":
    main()
