"""Consensus-Aggregation Pooling (principled pooling method, prototype) — replaces the crude one-prompt pool.

METHODOLOGY (the "scientific" pooling, vs the naive 'ask an LLM to summarize k cards'):
  1. DECOMPOSE each of the k member cards into atomic decision-elements (bullet/heuristic lines).
  2. AGREEMENT: embed every element; for each element, count how many DISTINCT members have a semantically-matching
     element (cosine >= TAU). agreement(e) = # supporting members. This is an explicit cross-member consensus measure.
  3. AGGREGATE: keep only elements with agreement >= Q (shared by >= Q members); DROP single-/few-member idiosyncrasies
     (exactly the material the one-prompt pool wrongly imports — e.g. one member's MICE/Amelia, offset, rare-events).
     Dedupe near-duplicate consensus elements (keep one representative). Then an LLM writes a coherent card from ONLY
     the consensus elements. Q is a knob tracing an anonymity-utility curve.

Rationale: members are k noisy samples of a shared group decision-policy; the consensus set estimates that policy while
suppressing member-specific detail -> more principled than one prompt + a small anonymity gain (drops the single-member
leak component) + better worst-case hygiene (no single member's card can dominate). Does NOT claim a privacy floor.

Run:  DATASET=cv K=8 SEED=0 CLUS=3 TAU=0.55 Q=3 python scripts/cmd_consensus_pool.py
Out:  results/{se}/consensus_pool_cv_k8_s0_G{CLUS}.json  {neutral_card, consensus_card, elements, stats}
"""
import os
import re
import sys
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("GROUP", "random")
import cmd_gate as CG            # noqa: E402
from src.llm import chat, _openai_client   # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DS = os.environ.get("DATASET", "cv")
K = int(os.environ.get("K", 8)); SEED = int(os.environ.get("SEED", 0)); CLUS = int(os.environ.get("CLUS", 3))
TAU = float(os.environ.get("TAU", 0.55))       # cosine >= TAU => "same decision element" (diff moves ~0.27-0.29; same ~0.6+)
Q = int(os.environ.get("Q", 3))                # keep elements shared by >= Q of the k members
DEDUP = float(os.environ.get("DEDUP", 0.80))   # drop a consensus element if ~duplicate (cos>=DEDUP) of a kept one
WRITER = os.environ.get("WRITER_MODEL", "deepseek-chat")
FIXED = {"cv": "cmd_shared_cards_cv__neutral_fixed.json", "mad": "cmd_shared_cards_mad__neutral_fixed.json",
         "enron": "cmd_shared_cards__neutral_fixed.json"}[DS]
FLAGS = ["mice", "amelia", "orthogonal polynomial", "offset", "rare-events", "rare events", "king", "10 events",
         "events per predictor", "cross-validation as the out"]   # Opus-flagged single-member imports to watch


def elements(card):
    """Atomic decision-elements = substantive lines (bullets/heuristics), headers & fragments dropped."""
    out = []
    for ln in (card or "").splitlines():
        s = re.sub(r"^\s*[-*•\d.)#]+\s*", "", ln).strip()
        if len(re.findall(r"\w+", s)) >= 5 and not s.isupper():
            out.append(s)
    return out


def embed(texts):
    """L2-normalized text-embedding-3-small vectors, sqlite-cached per text (R6 MAJOR-6: uncached
    embeddings made every sanitize re-run pay real dollars AND left fidelity-gate cosines exposed to
    API float noise across runs; the cache freezes each text's vector after first sight)."""
    import hashlib
    from src.llm import _conn, _lock
    keys = {t: hashlib.sha256(f"embed:text-embedding-3-small:{t}".encode("utf-8")).hexdigest()
            for t in set(texts)}
    out = {}
    with _lock:
        c = _conn()
        for t, k in keys.items():
            row = c.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
            if row is not None:
                out[t] = np.array(json.loads(row[0]))
        c.close()
    todo = [t for t in keys if t not in out]
    for i in range(0, len(todo), 256):
        chunk = todo[i:i + 256]
        r = _openai_client.embeddings.create(model="text-embedding-3-small", input=chunk)
        with _lock:
            c = _conn()
            for t, d in zip(chunk, r.data):
                out[t] = np.array(d.embedding)
                c.execute("INSERT OR REPLACE INTO cache (k, v) VALUES (?, ?)",
                          (keys[t], json.dumps(d.embedding)))
            c.commit()
            c.close()
    return [out[t] / (np.linalg.norm(out[t]) + 1e-12) for t in texts]


def main():
    _p, authors, nuwa, aggro, _r, _w = CG.load()
    grp, byc = CG.make_groups(aggro, authors, K, SEED)
    cid = sorted(byc)[CLUS]; members = byc[cid]
    fixed = json.loads((CG.SE / FIXED).read_text(encoding="utf-8"))
    neutral_card = fixed[f"k{K}_s{SEED}_{cid}"]

    # 1. decompose (track which member each element came from)
    elems, owner = [], []
    for m in members:
        for e in elements(aggro[m]):
            elems.append(e); owner.append(m)
    V = embed(elems)
    n = len(elems)
    print(f"=== consensus_pool  {DS} k{K} s{SEED} {cid}  members={len(members)}  elements={n}  TAU={TAU} Q={Q} ===")

    # 2. agreement: for each element, # DISTINCT members with a matching element (cos>=TAU)
    agree = []
    for i in range(n):
        sup = set()
        for j in range(n):
            if owner[j] != owner[i] and float(V[i] @ V[j]) >= TAU:
                sup.add(owner[j])
        agree.append(1 + len(sup))            # +1 = its own member
    agree = np.array(agree)
    dist = {q: int((agree == q).sum()) for q in range(1, len(members) + 1)}
    print(f"  agreement histogram (elements shared by exactly q members): {dist}")

    # 3a. consensus = agreement>=Q ; dedupe near-duplicates (keep the highest-agreement representative)
    idx = [i for i in range(n) if agree[i] >= Q]
    idx.sort(key=lambda i: -agree[i])
    kept, kept_txt = [], []
    for i in idx:
        if all(float(V[i] @ V[j]) < DEDUP for j in kept):
            kept.append(i); kept_txt.append(elems[i])
    dropped_single = [elems[i] for i in range(n) if agree[i] < Q]
    print(f"  kept consensus elements (>= {Q} members, deduped): {len(kept)}   | dropped (< {Q}): {len(dropped_single)}")

    def flags_in(lst):
        return sorted({f for f in FLAGS if any(f in e.lower() for e in lst)})
    print(f"  Opus-flagged single-member imports — in DROPPED: {flags_in(dropped_single)}")
    print(f"                                       in KEPT:    {flags_in(kept_txt)}")

    # 3b. LLM writes a coherent shared card from ONLY the consensus elements
    body = "\n".join(f"- {t}" for t in kept_txt)
    msg = [{"role": "system", "content": "You assemble a shared skill card for a team from a vetted list of consensus "
            "points that the team members have in common."},
           {"role": "user", "content": f"These are the decision/skill points shared by MULTIPLE members of a team "
            f"(consensus points only — anything specific to a single member has already been removed):\n\n{body}\n\n"
            "Write ONE coherent, well-organized shared skill card using ONLY these consensus points (do not add new "
            "points, do not reintroduce member-specific detail). Keep it concrete and usable. Output ONLY the card."}]
    consensus_card = chat(msg, model=WRITER, temperature=0.3, max_tokens=1300) or ""
    print(f"  consensus card: {len(re.findall(r'\\w+', consensus_card))} words  (neutral: {len(re.findall(r'\\w+', neutral_card))})")

    outdir = ROOT / "results" / ("se" if DS == "cv" else DS)
    (outdir / f"consensus_pool_{DS}_k{K}_s{SEED}_G{CLUS}.json").write_text(json.dumps({
        "cluster": cid, "members": members, "tau": TAU, "q": Q,
        "n_elements": n, "agreement_hist": dist, "n_kept": len(kept), "n_dropped": len(dropped_single),
        "neutral_card": neutral_card, "consensus_card": consensus_card,
        "kept_elements": kept_txt, "dropped_single_elements": dropped_single}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"  saved -> results/{'se' if DS=='cv' else DS}/consensus_pool_{DS}_k{K}_s{SEED}_G{CLUS}.json")


if __name__ == "__main__":
    main()
