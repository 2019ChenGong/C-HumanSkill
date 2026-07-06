"""Export CV UTILITY pairwise-competence judge tasks for FREE sonnet subagents = cross-model check that the
haiku utility (indiv-nocard +0.564 @26 / +0.463 @77) is NOT a haiku-judge artifact (缺口2).

Reuses cv_pilot's load / nuwa cards / draft (drafts are CACHED deepseek t0 -> re-calling = cache hits, no spend).
Per held unit, three comparisons — in_no=(indiv,nocard), in_st=(indiv,stranger), sh_no=(shared,nocard) — each
judged BIDIRECTIONAL (role r1 = X-vs-Y, role r2 = Y-vs-X); the score script debiases exactly like cv_pilot.judge().
A subagent answers each directional task with 'A'/'B'/'TIE' (which answer is more competent). This mirrors
cv_pilot._raw byte-for-byte (same SYS + USR), only the JUDGE model changes deepseek/haiku -> free sonnet subagent.

Run:  NEXP=26 BATCHDIR=results/se/util_judge python scripts/cv_util_judge_export.py scratchpad/se/stats.7z
Then: N sonnet subagents each answer one batch -> cv_util_judge_score.py
"""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("DATASET", "cv")
import cv_pilot as CVP  # noqa: E402
import cmd_gate as CG  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

NBATCH = int(os.environ.get("NBATCH", 10))
OUT = ROOT / os.environ.get("BATCHDIR", "results/se/util_judge")
OUT.mkdir(parents=True, exist_ok=True)

SYS = ("You compare two answers to the SAME statistics question and decide which is MORE competent and correct "
       "-- sounder method choice, correct assumptions, handles the subtlety. IGNORE writing style, length, tone.")
USR = "Question:\n{q}\n\nAnswer A:\n{a}\n\nAnswer B:\n{b}\n\nWhich is more competent? Answer ONLY 'A' or 'B' or 'TIE'."


def main():
    archive = Path(sys.argv[1])
    q_title, q_body, q_acc, a_body, cohort = CVP.load(archive)
    users = sorted(cohort, key=lambda u: -len(cohort[u]))[:CVP.NEXP]
    grp = CG.group_random(users, CVP.K, CVP.SEED)
    ngrp = len(set(grp.values()))
    print(f"CV util-judge export | using {len(users)} experts | K={CVP.K} groups={ngrp}", flush=True)

    indiv = {u: CVP.nuwa_assemble(CVP.nuwa_extract([b for (_, _, _, b) in cohort[u][:CVP.NCARD]])) for u in users}
    byc = defaultdict(list)
    for u in users:
        byc[grp[u]].append(u)
    shared = {g: CG.synth_shared([indiv[u] for u in mem]) for g, mem in byc.items()}
    vec = {u: CVP.cvec(indiv[u]) for u in users}
    stranger = {u: max((v for v in users if grp[v] != grp[u]), key=lambda v: CVP.cos(vec[u], vec[v])) for u in users}

    tasks = []
    meta = {}
    i = 0
    for u in users:
        for (aid, qid, sc, b) in cohort[u][CVP.NCARD:CVP.NCARD + CVP.NHELD]:
            acc = q_acc.get(qid)
            if not (acc and a_body.get(acc)):
                continue
            q = f"{q_title.get(qid, '')}\n{CVP.plain(q_body.get(qid, ''))[:1200]}"
            d_no = CVP.plain(CVP.draft(None, q))[:1400]                       # cache hits (deepseek t0)
            d_in = CVP.plain(CVP.draft(indiv[u], q))[:1400]
            d_st = CVP.plain(CVP.draft(indiv[stranger[u]], q))[:1400]
            d_sh = CVP.plain(CVP.draft(shared[grp[u]], q))[:1400]
            unit = f"{u}_{qid}"
            for cmp, (X, Y) in [("in_no", (d_in, d_no)), ("in_st", (d_in, d_st)), ("sh_no", (d_sh, d_no))]:
                for role, (first, second) in [("r1", (X, Y)), ("r2", (Y, X))]:   # r1=X-vs-Y, r2=Y-vs-X (bidir)
                    pid = f"T{i:04d}"; i += 1
                    tasks.append({"pid": pid, "prompt": USR.format(q=q, a=first, b=second)})
                    meta[pid] = {"u": str(u), "unit": unit, "cmp": cmp, "role": role}

    batches = [[] for _ in range(NBATCH)]
    for t in tasks:
        batches[int(t["pid"][1:]) % NBATCH].append(t)
    for j, b in enumerate(batches):
        (OUT / f"batch_{j}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "sys.txt").write_text(SYS, encoding="utf-8")
    n_units = len({m["unit"] for m in meta.values()})
    print(f"wrote {NBATCH} batches ({[len(b) for b in batches]}) = {len(tasks)} judge tasks over {n_units} units "
          f"(3 cmp x 2 orders) + meta + sys -> {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
