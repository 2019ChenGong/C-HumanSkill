"""B2 cross-JUDGE robustness: re-judge B1's MAD forced-choice battery with qwen3.7-max instead of sonnet.

The utility headline (ne > staab: pooled CMD card beats per-person de-id) was measured with sonnet-4.6 as the
free-subagent judge (B1, results/mad/fc, sonnet = replicate "r1"). B2 asks the reviewer's question: is that an
artifact of the sonnet judge, or does a DIFFERENT model reach the same verdict? We reuse the EXACT same pack --
same cached deepseek drafts, same sys.txt framing, same per-item prompts sonnet saw -- and only swap the judge.

qwen3.7-max runs in NON-thinking mode (reasoning.enabled=False, aligned to how the drafter/attacker use it), via
OpenRouter (openrouter/qwen/qwen3.7-max). Each forced-choice item is one independent call (the batch grouping in
the pack was only for subagent context hygiene; a programmatic judge has no such need), but answers are written
per batch as ans_qwen_<i>.json so fc_status.py (REPS=qwen) and cv_fc_score.py (ONLY=qwen) work unchanged.

Run:  BATCHDIR=results/mad/fc COST=1 python -P scripts/mad_fc_judge_qwen.py   # price it first, NO calls
      BATCHDIR=results/mad/fc         python -P scripts/mad_fc_judge_qwen.py   # judge (resumable; cached)
  then: ONLY=qwen BATCHDIR=results/mad/fc python -P scripts/cv_fc_score.py
        REPS=qwen BATCHDIR=results/mad/fc python -P scripts/fc_status.py       # coverage / quarantine
"""
import os
import re
import sys
import json
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import deid_enron as de           # noqa: E402  (de.pool -> 12-worker thread pool, order-preserving)
from src.llm import chat          # noqa: E402

ENC = tiktoken.get_encoding("cl100k_base")     # qwen tokenizer differs slightly; fine for a cost ESTIMATE
B = ROOT / os.environ.get("BATCHDIR", "results/mad/fc")
QWEN = os.environ.get("QWEN_MODEL", "openrouter/qwen/qwen3.7-max")
REASONING_OFF = {"reasoning": {"enabled": False}}
MAXOUT = int(os.environ.get("MAXOUT", "256"))          # non-thinking -> the short JSON verdict; headroom only
TEMP = float(os.environ.get("JUDGE_TEMP", "0.0"))
PRICE_IN = float(os.environ.get("PRICE_IN", "1.25"))   # qwen3.7-max OpenRouter list price, $/M tokens
PRICE_OUT = float(os.environ.get("PRICE_OUT", "3.75"))
EST_OUT = int(os.environ.get("EST_OUT", "48"))         # per-item output estimate ({"pid","choice","why<=15w"}）
COST = os.environ.get("COST", "") not in ("", "0")
ONLYB = os.environ.get("BATCHES", "")                  # optional "0,1,2" subset of batch indices (cheaper run)

PILOT = os.environ.get("PILOT", "") not in ("", "0")    # judge ONLY the battery probes -> competence gate, ~$0.15
SYS = (B / "sys.txt").read_text(encoding="utf-8")
batch_files = sorted(B.glob("batch_*.json"), key=lambda f: int(f.stem.split("_")[1]))
if ONLYB:
    keep = {int(x) for x in ONLYB.split(",")}
    batch_files = [f for f in batch_files if int(f.stem.split("_")[1]) in keep]
if not batch_files:
    sys.exit(f"no batch_*.json in {B}")

_CHOICE = re.compile(r'"choice"\s*:\s*"?\s*([ABab])')
_BARE = re.compile(r'\b([AB])\b')


def parse_choice(out):
    """Extract A/B from the judge output. Prefer the JSON 'choice' field; fall back to a bare A/B; else None."""
    if not out:
        return None, ""
    s = out.strip().strip("`")
    # try strict/loose JSON first (captures 'why' too)
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        try:
            o = json.loads(m.group(0))
            c = str(o.get("choice", "")).strip().upper()[:1]
            if c in ("A", "B"):
                return c, str(o.get("why", ""))[:120]
        except Exception:
            pass
    m = _CHOICE.search(s)
    if m:
        return m.group(1).upper(), ""
    m = _BARE.search(s)
    if m:
        return m.group(1).upper(), ""
    return None, ""


def judge_item(item):
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": item["prompt"]}]
    out = chat(msgs, model=QWEN, temperature=TEMP, max_tokens=MAXOUT, extra=REASONING_OFF) or ""
    c, why = parse_choice(out)
    return {"pid": item["pid"], "choice": c, "why": why}


def batch_ok(bi, want_pids):
    p = B / f"ans_qwen_{bi}.json"
    if not p.exists():
        return False
    try:
        recs = json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    got = [r.get("pid") for r in recs if isinstance(r, dict)
           and str(r.get("choice", "")).strip().upper()[:1] in ("A", "B")]
    return set(got) == set(want_pids) and len(got) == len(want_pids)


def pilot():
    """Competence gate: judge ONLY the battery probes (self/pad/fmt/cut) and check qwen is content-sensitive.
    order o is the LAST char of the pid; for a cut pid X{j}{u}{o}, j (=pid[1]) picks the level .10/.25/.50.
    'target' (full draft / padded copy / rich format) is A iff o==0 (mirrors the export construction)."""
    items = []
    for f in batch_files:
        for it in json.loads(f.read_text(encoding="utf-8")):
            if it["pid"][0] in ("S", "P", "F", "X"):
                items.append(it)
    print(f"PILOT: judging {len(items)} battery probes with {QWEN} (non-thinking) ...", flush=True)
    recs = de.pool(judge_item, items)
    by = {r["pid"]: r for r in recs}
    unparsed = [r["pid"] for r in recs if r["choice"] not in ("A", "B")]
    (B / "ans_qwen_pilot.json").write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    from collections import defaultdict
    CUTLVL = {"0": ".10", "1": ".25", "2": ".50"}
    self_A, pad_win, fmt_richwin, cut_fullwin = [], [], [], defaultdict(list)
    for it in items:
        pid = it["pid"]; r = by[pid]; c = r["choice"]
        if c not in ("A", "B"):
            continue
        o = int(pid[-1]) if pid[0] != "S" else 0
        target_A = (o == 0)                        # target = full/padded/rich is in slot A when o==0
        target_won = float((c == "A") == target_A)
        if pid[0] == "S":
            self_A.append(float(c == "A"))
        elif pid[0] == "P":
            pad_win.append(target_won)              # target here = the PADDED copy
        elif pid[0] == "F":
            fmt_richwin.append(target_won)          # target = the markdown-rich copy
        elif pid[0] == "X":
            cut_fullwin[CUTLVL.get(pid[1], "?")].append(target_won)   # target = the FULL draft

    def m(v):
        return (sum(v) / len(v), len(v)) if v else (float("nan"), 0)
    print(f"\n=== qwen competence gate (parsed {len(items)-len(unparsed)}/{len(items)}, "
          f"unparsed {len(unparsed)}) ===")
    r, n = m(self_A);        print(f"  self  P(pick A), identical texts   {r:.3f}  n={n}   (diagnostic; ~1.0 = reserves default)")
    r, n = m(pad_win);       print(f"  pad   P(prefer +25% filler)        {r:.3f}  n={n}   (want <= 0.5; >0.5 = length-biased)")
    r, n = m(fmt_richwin);   print(f"  fmt   P(prefer markdown-rich)      {r:.3f}  n={n}   (want ~0.5)")
    for lvl in (".10", ".25", ".50"):
        r, n = m(cut_fullwin.get(lvl, []))
        if n:
            print(f"  cut@{lvl} P(prefer FULL vs cut)       {r:.3f}  n={n}   (want > 0.5; @.10 = finest resolution test)")
    c10, _ = m(cut_fullwin.get(".10", []))
    padr, _ = m(pad_win)
    verdict = (c10 > 0.5 and not (padr > 0.6))
    print(f"\n  VERDICT: qwen {'looks COMPETENT (resolves finest cut, not length-biased) -> proceed' if verdict else 'FAILS the gate (blind at .10 cut or length-biased) -> do NOT scale up'}")
    print(f"  saved raw picks -> {(B/'ans_qwen_pilot.json').relative_to(ROOT)}")


def main():
    if PILOT and not COST:
        pilot(); return
    sys_tok = len(ENC.encode(SYS))
    total_items = in_tok = 0
    pending = []   # (bi, items)
    for f in batch_files:
        bi = int(f.stem.split("_")[1])
        items = json.loads(f.read_text(encoding="utf-8"))
        total_items += len(items)
        in_tok += sum(sys_tok + len(ENC.encode(it["prompt"])) for it in items)
        if not batch_ok(bi, [it["pid"] for it in items]):
            pending.append((bi, items))

    if COST:
        out_tok = total_items * EST_OUT
        usd = in_tok / 1e6 * PRICE_IN + out_tok / 1e6 * PRICE_OUT
        print(f"=== COST estimate: qwen judge over {len(batch_files)} batches / {total_items} items ===")
        print(f"  model   {QWEN}  (non-thinking)")
        print(f"  input   ~{in_tok:,} tok  (sys {sys_tok} tok x {total_items} items + prompts)  @ ${PRICE_IN}/M = ${in_tok/1e6*PRICE_IN:.2f}")
        print(f"  output  ~{out_tok:,} tok  ({EST_OUT}/item est)                          @ ${PRICE_OUT}/M = ${out_tok/1e6*PRICE_OUT:.2f}")
        print(f"  TOTAL   ~${usd:.2f}   (cached calls are free on re-run; {len(pending)}/{len(batch_files)} batches still pending)")
        return

    print(f"qwen judge: {len(pending)}/{len(batch_files)} batches pending ({total_items} items total)", flush=True)
    done = 0
    for bi, items in pending:
        recs = de.pool(judge_item, items)
        miss = [r["pid"] for r in recs if r["choice"] not in ("A", "B")]
        if miss:
            print(f"  batch_{bi}: {len(miss)} unparsed (e.g. {miss[:3]}) -- left for a retry pass", flush=True)
        clean = [{"pid": r["pid"], "choice": r["choice"], "why": r["why"]} for r in recs if r["choice"] in ("A", "B")]
        (B / f"ans_qwen_{bi}.json").write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
        done += 1
        if done % 20 == 0:
            print(f"  ...{done}/{len(pending)} batches written", flush=True)
    print(f"DONE: wrote {done} batches -> {B.relative_to(ROOT)}/ans_qwen_*.json", flush=True)


if __name__ == "__main__":
    main()
