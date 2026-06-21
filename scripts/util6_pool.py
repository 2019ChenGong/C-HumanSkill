"""Build the 20-MAD SeaMonkey developer pool (N_DEV / OUT configurable; full set = no random cap)."""
import os
import re
import sys
import json
import glob
import random
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "20mad"
PRODUCT = "SeaMonkey"
SEED = 0
N_DEV, C_MIN, T_MIN, CAP = int(os.environ.get("N_DEV", 110)), 30, 14, 22  # N_DEV=big -> NO random cap (full qualifier set)
C_TAKE = 40                                    # how many card comments to keep per dev
MIN_JUDG = 5
MIN_CHARS, MAX_CHARS = 60, 1500
CLASSES = ["FIXED", "WONTFIX", "INVALID", "DUPLICATE", "WORKSFORME"]
JUDGMENT = ["WONTFIX", "INVALID", "DUPLICATE", "WORKSFORME"]
AUTO = re.compile(r"^\s*\*\*\*|has been marked as a duplicate|Created attachment|\(In reply to", re.I)
LEAK = re.compile(r"dup(?:licate)?\b|wontfix|won'?t fix|invalid|works ?for ?me|worksforme|not a bug", re.I)


def clean(t):
    lines = [ln for ln in str(t).splitlines() if not ln.lstrip().startswith(">")]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()[:MAX_CHARS]


def main():
    rng = random.Random(SEED)
    idm = pd.read_parquet(D / "idmerging.parquet", columns=["key_hash", "merged_id"])
    g = idm.groupby("key_hash")["merged_id"].nunique()
    k2m = idm[idm.key_hash.isin(g[g == 1].index)].drop_duplicates("key_hash").set_index("key_hash")["merged_id"]

    prod_dir = D / "bugzilla" / "mozilla" / PRODUCT
    if not list(prod_dir.glob("*_nlcomments.parquet")):
        subprocess.run(["tar", "-xf", str(D / "bugzilla.tar"), f"bugzilla/mozilla/{PRODUCT}/"], cwd=str(D), check=True)
    nl = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(prod_dir / "*_nlcomments.parquet")))],
                   ignore_index=True).sort_values(["issue_id", "comment_id", "paragraph_id"])
    com = nl.groupby(["issue_id", "comment_id"])["text"].apply(lambda s: "\n".join(map(str, s))).reset_index()
    meta = pd.read_parquet(D / "comments.parquet",
                           columns=["source", "product", "issue_id", "comment_id", "author_key", "created"])
    meta = meta[(meta.source == "mozilla") & (meta["product"] == PRODUCT)]
    df = com.merge(meta, on=["issue_id", "comment_id"], how="inner")
    df["mid"] = df.author_key.map(k2m)
    df["auto"] = df["text"].str.contains(AUTO, na=False)
    df["clean"] = df["text"].map(clean)

    first = df.sort_values("created").groupby("issue_id").first().reset_index()
    report = first[["issue_id", "clean", "mid", "auto"]].rename(
        columns={"clean": "report", "mid": "rep_mid", "auto": "rep_auto"})

    cc = df[df.mid.notna() & ~df.auto & (df["clean"].str.len() >= MIN_CHARS)]      # usable dev comments

    iss = pd.read_parquet(D / "issues.parquet",
                          columns=["source", "product", "issue_id", "resolution", "component", "severity",
                                   "priority", "summary", "assignee_key"])
    iss = iss[(iss.source == "mozilla") & (iss["product"] == PRODUCT)
              & iss["resolution"].notna() & iss["summary"].notna()].copy()
    for c in ["resolution", "component", "severity", "priority", "summary"]:
        iss[c] = iss[c].map(str)
    iss["r"] = iss["resolution"].str.upper().str.strip()
    iss["mid"] = iss["assignee_key"].map(k2m)
    iss = iss[iss["r"].isin(CLASSES) & iss["mid"].notna()].merge(report, on="issue_id", how="inner")
    iss = iss[(iss["rep_mid"] != iss["mid"]) & (~iss["rep_auto"]) & (iss["report"].str.len() >= MIN_CHARS)
              & (~iss["summary"].str.contains(LEAK, na=False))]    # leak-safe

    cand = []
    for m, gb in iss.groupby("mid"):
        njudg = gb["r"].isin(JUDGMENT).sum()
        if len(gb) >= T_MIN and njudg >= MIN_JUDG:
            myc = cc[(cc.mid == m) & (~cc.issue_id.isin(set(gb.issue_id)))]
            if myc["issue_id"].nunique() >= C_MIN:
                cand.append(m)
    print(f"{PRODUCT}: {len(cand)} candidate devs (>= {T_MIN} solved bugs w/ >= {MIN_JUDG} judgment & "
          f">= {C_MIN} off-issue comments)", flush=True)
    if os.environ.get("PILOT_DRYRUN"):
        print("DRYRUN -> stop."); return
    chosen = rng.sample(sorted(cand), min(N_DEV, len(cand)))

    pool = {}
    for m in chosen:
        gb = iss[iss.mid == m].drop_duplicates("issue_id")
        judg = gb[gb["r"].isin(JUDGMENT)]
        nonj = gb[~gb["r"].isin(JUDGMENT)]
        take = pd.concat([judg.sample(min(len(judg), CAP), random_state=SEED),
                          nonj.sample(min(len(nonj), max(0, CAP - min(len(judg), CAP))), random_state=SEED)])
        solved = [{"issue_id": int(r.issue_id), "resolution": r.r, "report": r.report,
                   "stub": f"component: {r.component} | severity: {r.severity} | priority: {r.priority}\n"
                           f"summary: {r.summary[:200]}"}
                  for r in take.itertuples()]
        myc = cc[(cc.mid == m) & (~cc.issue_id.isin(set(gb.issue_id)))].drop_duplicates("issue_id").sort_values("created")
        pool[str(int(m))] = {"card_comments": myc["clean"].tolist()[:C_TAKE], "solved_bugs": solved}

    out = {"product": PRODUCT, "n_devs": len(pool), "classes": CLASSES, "judgment": JUDGMENT, "pool": pool}
    OUT = D / os.environ.get("OUT", "util6_pool.json")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    allr = Counter(b["resolution"] for v in pool.values() for b in v["solved_bugs"])
    print(f"\nSELECTED {len(pool)} devs | solved-bug resolution mix: {dict(allr)}")
    print(f"avg solved bugs/dev = {sum(len(v['solved_bugs']) for v in pool.values())/len(pool):.1f}")
    print("saved -> data/20mad/util6_pool.json")


if __name__ == "__main__":
    main()
