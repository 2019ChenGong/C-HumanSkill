"""Assemble the A2 package: gpt-5.4 SECOND ATTACKER on the FIXED (degeneracy-cleared) pooled cards.

The base gpt-5.4 pkg (gpt54_2afc_pkg) tested the BASE `shared`/`concat` cards. The degeneracy fix changed the
pooled cards (base CMD was masked-degenerate, sonnet read it .469; the FIXED neutral CMD reads .479-.535). So the
2nd-attacker question must be re-asked on the FIXED cards: does gpt-5.4 ALSO fail to re-identify on them?

We reuse the EXACT pairs sonnet answered in A1 (results/{mad,se,enron}/a1_ncc_k8_s{0,1,2}: neutral_fixed CMD +
concat_neutral, k8, 3 seeds, byte-identical prompts). gpt-5.4 answers the SAME batches; multiseed pooling
certifies (δ=.10) exactly as A1 did for sonnet. The indiv positive-control GATE is CITED from the base pkg
(gpt-5.4 indiv .711/.721/.655 >= sonnet, CI>0.5) — the per-person cards are unchanged by the degeneracy fix and
pooling-seed-independent, so that gate transfers; no need to re-run indiv.

This builds gpt54_2afc_fixed_pkg/ (self-contained for Codex): tasks/{ds}/s{seed}/{batch_*.json,meta.json},
tasks/{ds}/sys.txt, sonnet_baseline.json. (README + scorer are written separately.)

Run:  python -P scripts/build_gpt54_fixed_pkg.py
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "gpt54_2afc_fixed_pkg"
SEEDS = ["s0", "s1", "s2"]
BATCH_TARGET = 40                       # rebatch to ~40 items/batch -> fewer files for Codex waves
SRC = {"mad": "results/mad/a1_ncc_k8_{s}", "cv": "results/se/a1_ncc_k8_{s}",
       "enron": "results/enron/a1_ncc_k8_{s}"}

# A1 sonnet-4.6 multiseed baseline (delta=0.10), from results/a1_multiseed_pool_d10.json — what gpt-5.4 must reproduce.
SONNET = {
    "mad":   {"neutral": {"acc": 0.479, "up95_B": 0.513, "up95_C": 0.522, "verdict": "ANON"},
              "concat":  {"acc": 0.508, "up95_B": 0.543, "up95_C": 0.552, "verdict": "ANON"}},
    "enron": {"neutral": {"acc": 0.532, "up95_B": 0.564, "up95_C": 0.583, "verdict": "ANON"},
              "concat":  {"acc": 0.536, "up95_B": 0.568, "up95_C": 0.586, "verdict": "ANON"}},
    "cv":    {"neutral": {"acc": 0.535, "up95_B": 0.570, "up95_C": 0.587, "verdict": "ANON"},
              "concat":  {"acc": 0.574, "up95_B": 0.619, "up95_C": 0.626, "verdict": "LEAK (CV concat)"}},
}
# base pkg gpt-5.4 indiv positive control (per-person cards, unchanged by the fix) — the GATE, cited not re-run.
GATE = {"mad": {"gpt54_indiv": 0.711, "sonnet_indiv": 0.691},
        "cv": {"gpt54_indiv": 0.655, "sonnet_indiv": 0.701},
        "enron": {"gpt54_indiv": 0.721, "sonnet_indiv": 0.608}}


def rebatch(items, target):
    n = max(1, round(len(items) / target))
    out = [[] for _ in range(n)]
    for i, it in enumerate(items):
        out[i % n].append(it)
    return out


def main():
    if PKG.exists():
        shutil.rmtree(PKG)
    total_pairs = total_batches = 0
    manifest = {}
    for ds, tmpl in SRC.items():
        dsdir = PKG / "tasks" / ds
        dsdir.mkdir(parents=True, exist_ok=True)
        sys_written = False
        manifest[ds] = {}
        for s in SEEDS:
            src = ROOT / tmpl.format(s=s)
            meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
            items = []
            for bf in sorted(src.glob("batch_*.json"), key=lambda f: int(f.stem.split("_")[1])):
                items.extend(json.loads(bf.read_text(encoding="utf-8")))
            assert len(items) == len(meta), f"{ds}/{s}: {len(items)} items vs {len(meta)} meta"
            sd = dsdir / s
            sd.mkdir(parents=True, exist_ok=True)
            batches = rebatch(items, BATCH_TARGET)
            for i, b in enumerate(batches):
                (sd / f"batch_{i}.json").write_text(json.dumps(b, ensure_ascii=False, indent=1), encoding="utf-8")
            (sd / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
            if not sys_written:
                shutil.copy(src / "sys.txt", dsdir / "sys.txt")
                sys_written = True
            manifest[ds][s] = {"pairs": len(items), "batches": len(batches)}
            total_pairs += len(items); total_batches += len(batches)
    # baseline file
    baseline = {
        "instrument_sonnet": "free-subagent sonnet-4.6 2AFC (the single attacker); A1 multiseed pool, delta=0.10",
        "chance": 0.5, "delta": 0.10, "margin_U": 0.60,
        "gate_rule": ("gpt-5.4 is already validated as a STRONG card-reader by the base 2nd-attacker run: its "
                      "indiv positive control (per-person cards, UNCHANGED by the degeneracy fix, pooling-seed "
                      "independent) is .711/.721/.655 (MAD/Enron/CV) >= sonnet and CI>0.5. That gate TRANSFERS to "
                      "the fixed pooled cards here; so gpt-5.4 reading neutral/concat ~chance means the card is "
                      "anonymous, not that the attacker is weak."),
        "channel_kind": {"neutral": "pooled CMD, fixed (want <=de-id, ideally chance)",
                         "concat": "pooled naive concat, fixed (want <=de-id)"},
        "gate_indiv_from_base_pkg": GATE,
        "sonnet_baseline_multiseed_d10": SONNET,
        "note": ("gpt-5.4 answers the SAME a1_ncc pairs sonnet did (byte-identical prompts). The win: gpt-5.4 "
                 "reproduces A1 -> neutral CERTIFIED anon on all 3 (up95<0.60 for both poolers B & C); concat anon "
                 "on MAD/Enron and (like sonnet) LEAKS on CV. That kills 'the pooled card is anonymous only "
                 "because sonnet is a weak attacker'."),
    }
    (PKG / "sonnet_baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=1), encoding="utf-8")
    (PKG / "_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"built {PKG.relative_to(ROOT)}  | {total_pairs} pairs, {total_batches} batches over 3 ds x 3 seeds")
    for ds in SRC:
        print(f"  {ds}: " + "  ".join(f"{s}={manifest[ds][s]['pairs']}p/{manifest[ds][s]['batches']}b" for s in SEEDS))


if __name__ == "__main__":
    main()
