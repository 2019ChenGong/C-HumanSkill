"""CV T2 backfill — rebuild the card-building corpus from Posts.xml and PROVE it is byte-identical
to what actually reached the card-building LLM calls (预注册 results/VMIA_D1_PREREG.md 修订 B).

WHY THIS EXISTS
  D1-VMIA's primary criterion needs `target=docs` = the raw documents that entered the card-building
  call.  Enron/MAD keep their source text in the repo; CV never did — `cv_build.py` persisted only the
  held-out `ref`/`raw` samples, so the 12 answers per author were reconstructible only from the
  original stats.stackexchange dump.  That dump is now back on disk (md5-verified, see 修订 B.1).

WHY MONKEY-PATCHING, NOT RE-TYPING THE PROMPT
  The proof is a cache-key recomputation, and the key covers the *exact* messages list.  Re-typing
  `nuwa_extract`'s prompt here would make a transcription slip indistinguishable from a data mismatch.
  Instead we replace `cv_pilot.chat` with a capture stub and call the REAL `cv_pilot.nuwa_extract` /
  `nuwa_assemble`.  Two consequences, both load-bearing:
    * zero transcription risk — the messages are whatever the builder actually constructs;
    * no LLM call is reachable, so the run is structurally $0.
  ★ Scope of that second point, stated precisely (a review flagged the looser wording): the patch
  rebinds `cv_pilot`'s OWN `chat` name.  `cmd_gate`, `deid_enron`, `enron_nuwa` and `mad_nuwa_step2`
  each hold their own `from src.llm import chat` binding, and those are NOT patched.  The guarantee
  therefore holds because this script's call graph is exactly {load, nuwa_extract, nuwa_assemble,
  plain} and none of them route through those modules — NOT because `chat` is globally unavailable.
  Anything added here that calls into those modules would make paid calls again; patch them too.
  The cache is opened `mode=ro` so a bug here cannot write a row and manufacture its own "hit".

VERIFICATION IS TWO-STEP (修订 B.2)
  step 1  rebuilt 12 answers -> nuwa_extract key -> cache hit    (proves the INPUTS match)
  step 2  that cached output -> nuwa_assemble key -> cache hit, and the retrieved card must equal
          cv_cmd_nuwa.json["nuwa"][author] byte-for-byte                (end-to-end closure)
  Step 1 alone suffices for the claim; step 2 catches "key computed correctly but slice off-by-one".

Run:  python -P scripts/cv_t2_verify.py            [DRYRUN=1 -> first 5 authors, writes nothing]
Run from project root.
"""
import os
import re
import sys
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

# ★ cv_pilot.load()'s cohort and nuwa_extract's model depend on these; a stale value would silently
#   rebuild a DIFFERENT corpus (or key on a different model) and surface only as a mass cache miss.
#   Clear them so the documented defaults apply.
for _v in ("NEXP", "NHELD", "QACC", "CHARS", "K", "SEED", "DRAFT_TOK", "GEN", "JUDGE"):
    os.environ.pop(_v, None)
# DATASET is NOT popped: cv_pilot imports cmd_gate, which defaults to enron paths when unset.
# cv_build.py sets it to "cv" before importing cv_pilot — mirror that exactly.
os.environ["DATASET"] = "cv"

import src.llm as LLM                       # noqa: E402  for _key + CACHE_DB (NOT for calling)
import cv_pilot as CVP                      # noqa: E402  load / nuwa_extract / nuwa_assemble / plain

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ARCHIVE = Path(os.environ.get("CV_ARCHIVE", ROOT / "data" / "se_raw" / "stats.7z"))
NUWA_JSON = ROOT / "data" / "se" / "cv_cmd_nuwa.json"
OUT_DOCS = ROOT / "data" / "se" / "cv_card_docs.json"
OUT_AUDIT = ROOT / "results" / "se" / "cv_t2_verify.json"
DRYRUN = os.environ.get("DRYRUN", "").strip().lower() not in ("", "0", "false", "no")
DRY_N = 5

TRUNC = 1500          # cv_pilot.nuwa_extract: plain(a)[:1500] (修订 B.4). EXPECTED value only —
                      # the bound actually used is measured by _probe_trunc() and must match this.
NCARD = CVP.NCARD     # 12
MINWORDS = 50         # G8


# ---------------------------------------------------------------- capture stub (no network, ever)
class _Captured(str):
    """Sentinel returned by the stub. Subclasses str so any incidental string use downstream is safe."""


_CAP = {}


def _capture_chat(messages, model=LLM.DEFAULT_MODEL, temperature=0.0, max_tokens=1024,
                  use_cache=True, retries=5, extra=None):
    """Mirrors src.llm.chat's signature and its cache-key construction — and returns without calling
    anything.  params must be built EXACTLY as chat() builds it or the key will not match."""
    params = {"temperature": temperature, "max_tokens": max_tokens}
    if extra:
        params["_extra"] = extra
    _CAP["model"], _CAP["messages"], _CAP["params"] = model, messages, params
    return _Captured("<captured>")


def _key_for(fn, *args):
    _CAP.clear()
    fn(*args)
    if not _CAP:
        raise RuntimeError(f"{fn.__name__} did not call chat() — capture failed, cannot verify")
    return LLM._key(_CAP["model"], _CAP["messages"], _CAP["params"]), _CAP["model"], _CAP["params"]


def _probe_trunc():
    """Measure cv_pilot's per-answer truncation instead of trusting a copied literal.

    The T2 corpus is built here, separately from `nuwa_extract`, so the cache-key gate does NOT
    protect it: if `nuwa_extract`'s `[:1500]` ever changed, the key check would still validate the
    (new) real call while this file kept slicing at the old bound, silently breaking the "exactly the
    bytes that entered the call" claim.  So derive the bound from the real function: feed one
    synthetic answer far longer than any plausible limit and read the boundary off the captured
    message.  `plain()` is the identity on a run of 'w' (no tags, no whitespace to collapse), and the
    prompt template itself has no 'w' run longer than 1, so the longest run IS the truncation point.
    """
    _key_for(CVP.nuwa_extract, ["w" * 40000])
    body = " ".join(m["content"] for m in _CAP["messages"])
    return max((len(m) for m in re.findall(r"w+", body)), default=0)


# ---------------------------------------------------------------- read-only cache
def _ro_cache():
    if not LLM.CACHE_DB.exists():
        sys.exit(f"ABORT: cache not found: {LLM.CACHE_DB}")
    uri = "file:" + LLM.CACHE_DB.as_posix().replace("?", "%3f").replace("#", "%23") + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30)


def main():
    if not ARCHIVE.exists():
        sys.exit(f"ABORT: archive not found: {ARCHIVE}")
    posts = ARCHIVE.parent / f"_{ARCHIVE.stem}" / "Posts.xml"
    if not posts.exists():
        sys.exit(f"ABORT: Posts.xml not found where cv_pilot.load expects it: {posts}")

    if not NUWA_JSON.exists():
        sys.exit(f"ABORT: shipped cards not found: {NUWA_JSON}")
    shipped = json.loads(NUWA_JSON.read_text(encoding="utf-8"))["nuwa"]
    authors = list(shipped)
    print("=" * 96)
    print("CV T2 补齐 — 字节级验证(预注册 VMIA_D1_PREREG.md 修订 B)")
    print(f"  archive : {ARCHIVE}")
    print(f"  Posts   : {posts}  ({posts.stat().st_size:,} bytes)")
    print(f"  shipped : {NUWA_JSON.name}  n={len(authors)} authors")
    print(f"  mode    : {'DRYRUN (前 %d 位, 不写盘)' % DRY_N if DRYRUN else 'FULL'}")
    print("=" * 96)

    print("\n[1/3] 按 cv_pilot.load() 原样重建 cohort(两趟 iterparse, 约 1-2 分钟)...", flush=True)
    _qt, _qb, _qa, _ab, cohort = CVP.load(ARCHIVE)
    print(f"      cohort(>= {NCARD + CVP.NHELD} gold answers) = {len(cohort)} authors", flush=True)

    missing = [a for a in authors if a not in cohort]
    short = [a for a in authors if a in cohort and len(cohort[a]) < NCARD]
    print(f"      shipped 作者在 cohort 中: {len(authors) - len(missing)}/{len(authors)}"
          f"   缺席={len(missing)}  答案不足12={len(short)}")
    if missing[:5]:
        print(f"      缺席样例: {missing[:5]}")

    # -------------------------------------------------------------- verify
    todo = authors[:DRY_N] if DRYRUN else authors
    print(f"\n[2/3] 逐作者双步验证({len(todo)} 位)...", flush=True)
    CVP.chat = _capture_chat            # ★ cv_pilot's chat is now a capture stub (scope: see header)
    trunc = _probe_trunc()
    print(f"      截断边界(从 cv_pilot.nuwa_extract 实测,非照抄常量)= {trunc}  期望={TRUNC}", flush=True)
    if trunc != TRUNC:
        sys.exit(f"ABORT: cv_pilot.nuwa_extract now truncates at {trunc}, not {TRUNC}. The T2 corpus "
                 f"is built here separately, so the cache-key gate would NOT catch this desync — "
                 f"update 修订 B.4 and TRUNC together, deliberately.")
    con = _ro_cache()
    rows, docs = {}, {}
    for i, a in enumerate(todo, 1):
        rec = {"in_cohort": a in cohort, "n_answers": len(cohort.get(a, []))}
        if not rec["in_cohort"] or rec["n_answers"] < NCARD:
            rec["status"] = "NO_COHORT"
            rows[a] = rec
            continue
        bodies = [b for (_, _, _, b) in cohort[a][:NCARD]]
        rec["answer_ids"] = [int(t[0]) for t in cohort[a][:NCARD]]

        k1, model, params = _key_for(CVP.nuwa_extract, bodies)
        rec["extract_key"] = k1
        r1 = con.execute("SELECT v FROM cache WHERE k=?", (k1,)).fetchone()
        if r1 is None:
            rec["status"] = "MISS_EXTRACT"
            rows[a] = rec
            continue
        notes = r1[0]

        k2, _m2, _p2 = _key_for(CVP.nuwa_assemble, notes)
        rec["assemble_key"] = k2
        r2 = con.execute("SELECT v FROM cache WHERE k=?", (k2,)).fetchone()
        if r2 is None:
            rec["status"] = "HIT_EXTRACT_MISS_ASSEMBLE"
            rows[a] = rec
            continue
        rec["card_identical"] = (r2[0] == shipped[a])
        rec["status"] = "VERIFIED" if rec["card_identical"] else "CARD_MISMATCH"

        # T2 text = exactly the bytes that entered the extract call (修订 B.4: plain() + [:1500])
        t2 = " ".join(CVP.plain(b)[:trunc] for b in bodies)
        rec["t2_words"] = len(t2.split())
        rec["t2_chars"] = len(t2)
        if rec["status"] == "VERIFIED":
            docs[a] = t2
        rows[a] = rec
        if i % 20 == 0 or i == len(todo):
            print(f"      {i}/{len(todo)} ...", flush=True)
    con.close()

    # -------------------------------------------------------------- report + gate
    from collections import Counter
    tally = Counter(r["status"] for r in rows.values())
    nver = tally.get("VERIFIED", 0)
    rate = nver / len(todo) if todo else 0.0
    print(f"\n[3/3] 结果  (n={len(todo)})")
    for st, c in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"      {st:26s} {c:4d}")
    print(f"\n      ★ 验证通过率 = {nver}/{len(todo)} = {rate:.1%}")

    thin = sorted(a for a in docs if len(docs[a].split()) < MINWORDS)          # G8
    print(f"      G8 T2 非空闸: <{MINWORDS} 词的作者 = {len(thin)}"
          f"{' -> ' + str(thin[:5]) if thin else '  ✔'}")
    if docs:
        ws = sorted(len(v.split()) for v in docs.values())
        print(f"      T2 词数: min={ws[0]} p50={ws[len(ws)//2]} max={ws[-1]}")

    # 修订 B.3 — frozen BEFORE the rate was known
    if rate >= 1.0:
        gate = "FULL"; msg = "= 100% ⇒ CV 全量进主判据"
    elif rate >= 0.80:
        gate = "SUBSET"; msg = f"≥80% 且 <100% ⇒ 只用通过的 {nver} 位,FINDINGS 必须披露通过率与缩小的候选池"
    else:
        gate = "UNCOVERED"; msg = "<80% ⇒ CV 仍记 uncovered,不做半吊子测量"
    print(f"\n      纳入规则(修订 B.3,跑前冻结): {gate} — {msg}")

    audit = {"prereg": "results/VMIA_D1_PREREG.md 修订 B",
             "archive": str(ARCHIVE), "posts_bytes": posts.stat().st_size,
             "cohort_size": len(cohort), "n_shipped": len(authors), "n_checked": len(todo),
             "trunc": trunc, "trunc_expected": TRUNC, "ncard": NCARD, "tally": dict(tally),
             "verified": nver, "rate": rate, "gate": gate, "g8_thin": thin,
             "dryrun": DRYRUN, "per_author": rows}

    if DRYRUN:
        print(f"\nDRYRUN — 不写盘。全量跑将写 {OUT_DOCS.relative_to(ROOT)}"
              f"({len(docs)} 位作者的 T2)与 {OUT_AUDIT.relative_to(ROOT)}")
        return
    if gate == "UNCOVERED":
        OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
        OUT_AUDIT.write_text(json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")
        sys.exit(f"\nGATE=UNCOVERED — 按修订 B.3 不产出 T2 语料。审计已写 {OUT_AUDIT.relative_to(ROOT)}")

    OUT_DOCS.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOCS.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.write_text(json.dumps(audit, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {OUT_DOCS.relative_to(ROOT)}  ({len(docs)} authors)")
    print(f"saved -> {OUT_AUDIT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
