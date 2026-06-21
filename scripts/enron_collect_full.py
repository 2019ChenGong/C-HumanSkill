"""Collect the full qualifying Enron author set (every sender with >=20 distinct docs; no 100-author cap)."""
import sys
import json
import tarfile
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from enron_clean import load_enron_cleaned, SENT  # noqa: E402
from deid_enron import _dedup  # noqa: E402

SE = ROOT / "data" / "enron"
TAR = SE / "enron_mail_20150507.tar.gz"
EXIST = SE / "collected_rag100_40.json"
OUT = SE / "collected_ragfull_40.json"
N_PER = 40
BAR = 20  # N_TRAIN(12)+N_REF(6)+N_TGT(2) = the enron_nuwa100 main-script author bar


def all_users():
    cnt = Counter()
    with tarfile.open(TAR, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            p = m.name.split("/")
            if len(p) >= 4 and p[0] == "maildir" and p[2].lower() in SENT:
                cnt[p[1]] += 1
    return [u for u, _ in cnt.most_common()]  # all, most-prolific first


def main():
    existing = json.loads(EXIST.read_text(encoding="utf-8"))
    keep_existing = {a: v for a, v in existing.items() if len(v) >= BAR}
    print(f"existing kept (>= {BAR}): {len(keep_existing)}/{len(existing)}", flush=True)

    users = all_users()
    new_cands = [u for u in users if u not in existing]
    print(f"maildir senders total={len(users)}; new candidates to collect={len(new_cands)}", flush=True)

    got = load_enron_cleaned(new_cands, n_per=N_PER, do_scrub=True)
    new_qual = {}
    for a in new_cands:
        d = _dedup(got.get(a, []))
        if len(d) >= BAR:
            new_qual[a] = d
    print(f"new qualifiers (>= {BAR} distinct): {len(new_qual)}", flush=True)
    print(f"  -> {sorted(new_qual)}", flush=True)

    merged = {**keep_existing, **new_qual}
    OUT.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    sizes = sorted((len(v) for v in merged.values()))
    print(f"\nFULL set N={len(merged)} (was {len(keep_existing)}); doc-count[min..max]={sizes[0]}..{sizes[-1]}", flush=True)
    print(f"saved -> {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
