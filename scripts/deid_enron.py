"""Shared helpers: parallel LLM pool, content vectors and cosine, the task list, and document dedup."""
import os
import re
import sys
import json
import math
import random
import hashlib
import logging
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

os.environ["LITELLM_LOG"] = "ERROR"
logging.getLogger("httpx").setLevel(logging.ERROR)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.llm import chat, sample_one  # noqa: E402
from src.attrib_metrics import empirical_null, paired_diff_ci, holm  # noqa: E402
from enron_clean import load_enron_cleaned  # noqa: E402

MODEL_GEN = os.environ.get("GEN_MODEL", "deepseek-chat")  # executor override (weak-model probe): GEN_MODEL=qwen/...
MODEL_JUDGE = "deepseek-chat"                             # NOTE: == MODEL_GEN -> self-consistency bias (likely
#                                                          overstates utility). Utility is judge-sensitive; read
#                                                          across judges (gpt 0.95 / haiku 0.51 / deepseek ?), not absolute.
ATTRIBUTORS = ["deepseek-chat", "claude-haiku-4-5"]      # independent attributor switched GPT -> Haiku
CANDIDATES = ["dasovich-j", "germany-c", "kaminski-v", "mann-k", "shackleton-s", "beck-s", "jones-t",
              "taylor-m", "kean-s", "symes-k", "sanders-r", "nemec-g", "haedicke-m", "scott-s",
              "dorland-c", "lokay-m", "bass-e", "love-p"]
N_AUTHORS = 10
N_PER = 24            # collect extra so dedup still leaves >= N_TRAIN+N_HELD distinct docs
N_TRAIN = 8
N_HELD = 6
K = 3
WORKERS = 12
ARMS = ["original", "pii", "deid", "ctrl", "deid2", "deid3", "deid4"]   # appended -> earlier arms keep cache seeds
DEID_ARMS = ["pii", "deid", "ctrl", "deid2", "deid3", "deid4"]
# deid  = marker-based surgical removal (preserves content -> diagnosed to PROTECT the content fingerprint)
# deid2 = abstract->rewrite: distill the email to neutral points, then regenerate fresh (no original phrasing)
# deid3 = marker removal THEN a full paraphrase pass (paraphrase-dominant, like ctrl + extra removal).
#         INTENDED to dominate ctrl heuristically; NOT a rigorous superset (the two paraphrases see
#         different inputs), so deid3 CAN still land above ctrl -- if it does, that's a real result.
# deid4 = MARKER-AWARE abstract -> rewrite: deid2's bottleneck BUT the abstraction step is told the author's
#         markers and strips them. SHARES deid2's step 2 (rewrite_from_points) verbatim, so deid4-vs-deid2
#         isolates whether knowing the markers adds removal ON TOP of the blind content bottleneck.

# O4 utility = does the DE-ID skill still DO THE WORK? Run original-skill vs deid-skill on these NEW shared
# work situations (domain-generic, hand-written -> reproducible, no generation loop, disjoint from emails);
# a judge decides whether the two replies have the SAME PRACTICAL EFFECT, ignoring identity/style.
# DISCRETE-VERDICT tasks (from the task-design sweep: open tasks let one card draft two different decisions
# -> noise floor ~0.2 -> unreadable. Each task pins the situation and forces ONE of 3 options on the first
# line, so the same card converges across redrafts (higher floor) while different people still pick different
# verdicts (cross stays low). Each keeps a genuine BORDERLINE tension so judgment styles diverge.) Paired with
# the decision-focused effect_same judge.
def _verdict(ctx, opts):
    return (f"{ctx} Decide: {opts}. Put your decision on the FIRST line (exactly one of the options), then "
            "2-3 sentences to the sender.")


TASKS = [
    _verdict("A counterparty requests a 3-day extension to deliver trade confirmations. Your standard policy is "
             "firm deadlines, but this counterparty has been reliable and the slip is small.",
             "GRANT / GRANT WITH CONDITIONS / DENY the extension"),
    _verdict("A counterparty disputes the gas volume on a recent deal. Your records show your number is correct, "
             "but their figure is plausible and they want a corrected statement by end of week.",
             "HOLD FIRM ON OUR NUMBER / OFFER A JOINT RECONCILIATION / CONCEDE AND REISSUE"),
    _verdict("A junior analyst wants to book a trade that is within position limits but relies on an aggressive "
             "price assumption you find borderline.",
             "APPROVE / APPROVE WITH A CAVEAT / SEND BACK FOR REVISION"),
    _verdict("A colleague forwards a draft agreement and asks you to confirm by Friday whether you can sign "
             "as-is. You have a concern about one liability clause in Section 12.",
             "SIGN AS-IS / SIGN BUT REQUEST A CARVE-OUT / DO NOT SIGN YET"),
    _verdict("Two internal systems report different volumes for the same trade. An analyst asks how to reconcile "
             "them before settlement.",
             "RECONCILE IT YOURSELF / ASK THE ANALYST TO FIX IT / ESCALATE TO THE DESK"),
    _verdict("A risk model is about to go up the chain. One key assumption looks optimistic to you, though it is "
             "defensible.",
             "APPROVE TO GO UP / GO UP WITH A FLAGGED CAVEAT / HOLD FOR REVISION"),
    _verdict("A vendor invoice arrives 15% above the quoted price, citing scope changes you were never told "
             "about. They want payment this week.",
             "PAY IN FULL / PAY THE QUOTED AMOUNT AND DISPUTE THE REST / HOLD PAYMENT PENDING REVIEW"),
    _verdict("A trader asks you to release a position limit by 10% for one day to close a profitable deal; it is "
             "within risk appetite but breaches the standing cap.",
             "APPROVE THE ONE-DAY EXCEPTION / APPROVE WITH A HEDGE REQUIREMENT / DENY"),
]
JUDGES = ["claude-haiku-4-5", "deepseek-chat"]   # haiku = independent (headline); deepseek = generator (self-consistent, secondary)
JUDGE0 = next((j for j in JUDGES if not j.startswith("deepseek")), JUDGES[0])


def pool(fn, items):
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(fn, items))


def _dedup(recs):
    """Drop near-duplicate / same-thread emails (re-sends, quoted self-replies, blasts) so train and
    held cannot leak near-identical text. Keep a doc only if it's < 0.5 token-Jaccard to every kept doc
    (and not a same-subject >0.3 overlap). Greedy in collection order."""
    kept, kept_toks, kept_subj = [], [], []
    for r in recs:
        toks = set(re.findall(r"\w+", (r.get("raw_text") or r["text"]).lower()))
        if not toks:
            continue
        subj = re.sub(r"^\s*(re|fw|fwd)\s*:\s*", "", (r.get("subject") or "").lower()).strip()
        dup = False
        for k, s in zip(kept_toks, kept_subj):
            j = len(toks & k) / len(toks | k)
            if j > 0.5 or (subj and subj == s and j > 0.3):
                dup = True
                break
        if not dup:
            kept.append(r)
            kept_toks.append(toks)
            kept_subj.append(subj)
    return kept


def get_docs():
    key = hashlib.sha1((",".join(sorted(CANDIDATES)) + f"-{N_PER}-dedup").encode()).hexdigest()[:10]
    cache = ROOT / "data" / "enron" / f"collected_{key}.json"
    if cache.exists():
        print(f"using cached collection {cache.name}", flush=True)
        return json.loads(cache.read_text(encoding="utf-8"))
    print("streaming tarball to collect emails (slow, one-time)...", flush=True)
    got = load_enron_cleaned(CANDIDATES, n_per=N_PER, do_scrub=True)
    deduped = {a: _dedup(v) for a, v in got.items()}
    for a in got:
        print(f"  {a}: {len(got[a])} -> {len(deduped[a])} after dedup", flush=True)
    out = {a: v for a, v in deduped.items() if len(v) >= N_TRAIN + N_HELD}
    print(f"kept {len(out)} authors with >= {N_TRAIN + N_HELD} distinct docs", flush=True)
    cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


# ---- pipeline (email-adapted copies of the deid_attribution functions) ----
def induce_skill(docs):
    body = "\n\n---\n\n".join(docs)
    msg = [{"role": "system", "content": "You profile an employee from samples of their work emails."},
           {"role": "user", "content":
            f"Below are several emails, ALL written by the SAME employee.\n\n{body}\n\nWrite a concise "
            "PROFILE (5-8 bullets) of this employee's RECOGNIZABLE working habits -- how they decide / "
            "handle requests / what they consistently do, plus tone/format -- so a reader could recognize "
            "this employee's FUTURE emails. Base it only on the texts."}]
    return chat(msg, model=MODEL_GEN, temperature=0.3, max_tokens=400) or ""


def identify_markers(target_docs, contrast_docs):
    target = "\n\n---\n\n".join(target_docs)
    contrast = "\n\n".join(f"(another employee) {d}" for d in contrast_docs)
    user = (f"TARGET employee's emails:\n{target}\n\nOTHER employees' emails (for contrast):\n{contrast}\n\n"
            "List the 3-6 distinctive habits / phrases / recurring moves that single out the TARGET employee "
            "from the others -- the things that would let someone recognize the target's FUTURE emails. "
            "Output a short bullet list, no preamble.")
    return chat([{"role": "system", "content": "You spot what makes an employee's emails identifiable."},
                 {"role": "user", "content": user}], model=MODEL_GEN, temperature=0.3, max_tokens=300) or ""


def rewrite_email(subject, doc, markers):
    msg = [{"role": "system", "content":
            "You rewrite a work email to remove the employee's identifying habits while preserving its "
            "substantive content and intent."},
           {"role": "user", "content":
            f"Email subject: {subject}\n\nEmail:\n{doc}\n\nThe employee's identifying habits to REMOVE:\n"
            f"{markers}\n\nRewrite the email so it NO LONGER reveals these habits or who wrote it, BUT it "
            "MUST preserve the substantive content & intent (the information conveyed, requests made, "
            "decisions stated). Do NOT add any new distinctive habit. Output ONLY the rewritten email."}]
    return chat(msg, model=MODEL_GEN, temperature=0.5, max_tokens=500) or ""


def reword_email(subject, doc):
    msg = [{"role": "system", "content": "You paraphrase a work email without changing its meaning."},
           {"role": "user", "content":
            f"Email subject: {subject}\n\nEmail:\n{doc}\n\nParaphrase this email: reword it while keeping its "
            "meaning, tone, decisions, and ALL substantive content the same. Output ONLY the paraphrased email."}]
    return chat(msg, model=MODEL_GEN, temperature=0.5, max_tokens=500) or ""


def abstract_email(subject, doc):
    """deid2 step 1: distill the email to neutral, transferable POINTS (info / requests / decisions),
    stripped of the writer's phrasing and stylistic tics -- the content bottleneck that breaks carry-through."""
    msg = [{"role": "system", "content": "You extract the essential, transferable content of a work email."},
           {"role": "user", "content":
            f"Email subject: {subject}\n\nEmail:\n{doc}\n\nList ONLY the substantive points this email must "
            "convey -- the facts/information, the requests/asks, and any decisions -- as 3-6 terse, NEUTRAL "
            "bullets. Strip the writer's personal phrasing, tone, and stylistic tics; keep what the recipient "
            "needs in order to act. Output only the bullets."}]
    return chat(msg, model=MODEL_GEN, temperature=0.3, max_tokens=400) or ""


def rewrite_from_points(subject, points):
    """deid2 step 2: write a FRESH plain email from the points only (NO access to the original text),
    so the author's vocabulary/structure cannot carry through."""
    msg = [{"role": "system", "content": "You write a plain, neutral work email from a list of points."},
           {"role": "user", "content":
            f"Email subject: {subject}\n\nPoints to convey:\n{points}\n\nWrite a single professional email that "
            "conveys EXACTLY these points. Use a neutral, generic business style with NO distinctive habits, "
            "phrasings, or personal tics. Do not add information beyond the points. Output ONLY the email."}]
    return chat(msg, model=MODEL_GEN, temperature=0.5, max_tokens=500) or ""


def abstract_email_marker_aware(subject, doc, markers):
    """deid4 step 1: SAME as abstract_email, but the abstractor is TOLD the author's identifying markers and
    instructed to strip their STYLE while KEEPING any substance they reference. deid4's step 2 reuses
    rewrite_from_points VERBATIM, so deid4 differs from deid2 ONLY in whether the abstraction knows the markers.
    NOTE (review fix): markers may name content-bearing habits; instructing 'strip completely' would drop
    content (failure mode B). So we strip the PERSONAL PHRASING of each marker but PRESERVE every fact/request/
    number/deadline/decision -- isolating 'know which style to remove' from 'told to delete content'."""
    msg = [{"role": "system", "content": "You extract the essential, transferable content of a work email."},
           {"role": "user", "content":
            f"Email subject: {subject}\n\nEmail:\n{doc}\n\nThis writer's IDENTIFYING habits/phrasings to "
            f"neutralize:\n{markers}\n\nList ONLY the substantive points this email must convey -- the "
            "facts/information, the requests/asks, and any decisions -- as 3-6 terse, NEUTRAL bullets. Strip "
            "the writer's personal PHRASING, tone, and stylistic tics (including the habits listed above), BUT "
            "you MUST still preserve every fact, request, number, deadline, name, and decision -- even if a "
            "listed habit refers to one; remove only the personal wording, never the substance. Output only "
            "the bullets."}]
    return chat(msg, model=MODEL_GEN, temperature=0.3, max_tokens=400) or ""


def attribute(cards, query, seed, tag, attr_model):
    ids = sorted(cards)
    perm = ids[:]
    random.Random(seed).shuffle(perm)
    block = "\n\n".join(f"Employee {i+1}:\n{cards[perm[i]]}" for i in range(len(perm)))
    msg = [{"role": "system", "content": "You identify which employee wrote an email, from employee profiles."},
           {"role": "user", "content":
            f"{block}\n\n=== EMAIL TO IDENTIFY ===\n{query}\n\nBased on working/writing habits, which employee "
            f"most likely wrote this email? Answer with ONLY the number (1-{len(perm)})."}]
    out = sample_one(msg, attr_model, s=0, temperature=0.7, max_tokens=6, salt=f"{tag}-{seed}")
    cand = [int(x) for x in re.findall(r"\d+", out or "") if 1 <= int(x) <= len(perm)]
    return perm[cand[-1] - 1] if cand else None


def utility_yes(subject, orig, new):  # DEAD in this script (superseded by effect_same/content_preserved); kept for reference
    msg = [{"role": "system", "content":
            "You judge whether a rewritten work email still conveys the original's substantive content and "
            "intent. Ignore tone, wording, and identity; judge ONLY whether the information / requests / "
            "decisions are preserved."},
           {"role": "user", "content":
            f"Subject: {subject}\n\nORIGINAL email:\n{orig}\n\nREWRITTEN email:\n{new}\n\nDoes the REWRITTEN "
            "email preserve the original's substantive content & intent? Answer ONLY YES or NO."}]
    return "yes" in (chat(msg, model=MODEL_JUDGE, temperature=0.0, max_tokens=4) or "").lower()


def content_preserved(subject, orig, new, model):
    """Per-email content-preservation judge for the B-guard (deid4-vs-deid2 review fix). Read on the ACTUAL
    rewritten TRAINING emails (where content-dropping happens), NOT on downstream skill drafts. Ignore style/
    identity; YES only if all facts/requests/numbers/deadlines/decisions survive. Comparing deid4 to deid2 on
    the SAME 60 emails controls for the bottleneck's own baseline loss -> a deid4 anonymity gain is 'real'
    (mode A) only if deid4 preserves content about as well as deid2 does here."""
    msg = [{"role": "system", "content":
            "You judge whether a rewritten work email keeps ALL of the original's substantive content. Ignore "
            "style, tone, wording, and identity (masked names/placeholders are fine). Answer NO if ANY fact, "
            "request, number, deadline, name, or decision from the original is missing or changed."},
           {"role": "user", "content":
            f"Subject: {subject}\n\nORIGINAL email:\n{orig}\n\nREWRITTEN email:\n{new}\n\nDoes the REWRITTEN "
            "email preserve EVERY substantive fact, request, number, deadline, and decision of the original "
            "(ignoring style)? Answer ONLY YES or NO."}]
    return "yes" in (chat(msg, model=model, temperature=0.0, max_tokens=4) or "").lower()


def draft_reply(card, task):
    """Execute a skill: the induced working-style profile drafts a reply to a NEW work situation."""
    msg = [{"role": "system", "content":
            "You are an Enron employee handling work email. Use the working-style profile below to respond to "
            "the situation the way THIS employee would. Output only the email reply, nothing else."},
           {"role": "user", "content":
            f"Your working-style profile:\n{card}\n\nSituation:\n{task}\n\n"
            "Write ONLY your email reply, following any length or format the situation specifies."}]
    return chat(msg, model=MODEL_GEN, temperature=0.5, max_tokens=400) or ""


def draft_reply_alt(card, task):
    """A SECOND, independent draft of the SAME skill (sampling-varied) -> noise-floor baseline: two open-ended
    drafts of one skill already differ in specifics, so this is the ceiling SAME-rate can reach."""
    msg = [{"role": "system", "content":
            "You are an Enron employee handling work email. Use the working-style profile below to respond to "
            "the situation the way THIS employee would. Output only the email reply, nothing else."},
           {"role": "user", "content":
            f"Your working-style profile:\n{card}\n\nSituation:\n{task}\n\n"
            "Write ONLY your email reply, following any length or format the situation specifies."}]
    return sample_one(msg, MODEL_GEN, s=1, temperature=0.5, max_tokens=400, salt="draft-alt") or ""


def effect_same(task, a, b, model):
    """Judge: do two replies to the SAME situation have the SAME PRACTICAL EFFECT, ignoring identity/style?
    (info conveyed / requests / decisions / actionable outcome). Masked names/placeholders != lost info.
    NOTE: paired with the DISCRETE-VERDICT tasks (which lift the noise floor 0.04->0.46). A 'focus only on the
    first-line decision' variant was tested in the task-design sweep and REJECTED: with just 3 options people's
    decisions often collide, so ignoring the rationale let DIFFERENT cards read as SAME -> cross 0.23->0.40,
    margin 0.22->0.06 (leakage). This original 'practical effect' wording -- which separates colliding
    decisions by their material rationale differences -- gave the best signal (margin 0.22), so keep it."""
    msg = [{"role": "system", "content":
            "You compare two email replies written for the SAME work situation. Decide ONLY whether they would "
            "have the SAME PRACTICAL EFFECT: the same information conveyed, the same requests/decisions, the same "
            "actionable outcome. IGNORE writing style, tone, identity, and names; masked names or placeholders do "
            "NOT count as lost information. Different wording with the same substance = SAME; but a different "
            "decision, a missing or added request, or a changed deadline/number/commitment = DIFFERENT."},
           {"role": "user", "content":
            f"Situation:\n{task}\n\nReply A:\n{a}\n\nReply B:\n{b}\n\nDo A and B have essentially the SAME "
            "practical effect (same substance/decisions/info), ignoring style and identity? "
            "Answer ONLY 'SAME' or 'DIFFERENT'."}]
    out = (chat(msg, model=model, temperature=0.0, max_tokens=4) or "").strip().lower()
    return "same" in out


# ---- O3: topic-only channel -- deterministic content-word TF-IDF attributor (NO style) ----
# Measures how identifiable an author is by TOPIC/content alone. If an arm's LLM (style) attribution
# tracks its topic-only accuracy, the residual identity is TOPIC, not removable style (the dataset confound).
_STOP = set((
    "the a an and or but if then else of to in on for with at by from as is are was were be been being this "
    "that these those it its they them their he she his her you your we our us i me my mine will would can could "
    "should shall may might must do does did done have has had having not no nor so than too very just about "
    "into over under out up down off above below again further once here there when where why how all any both "
    "each few more most other some such only own same now also per re fw fwd dear thanks thank regards please "
    "let know need get got like want make made see send sent email message").split())


def _content_vec(text):
    t = re.sub(r"\[[a-z]+\]", " ", (text or "").lower())   # drop PII placeholders ([person]/[org]/...) FIRST
    return Counter(w for w in re.findall(r"[a-z]{3,}", t) if w not in _STOP)


def _cosine(u, v):
    dot = sum(c * v.get(w, 0.0) for w, c in u.items())
    nu = math.sqrt(sum(c * c for c in u.values()))
    nv = math.sqrt(sum(c * c for c in v.values()))
    return dot / (nu * nv) if nu and nv else 0.0


def build_topic_profiles(train_by_author):
    """Per-author content-word TF-IDF profile (rare topic words dominate -- the diagnosed fingerprint)."""
    docs = {a: [_content_vec(t) for t in texts] for a, texts in train_by_author.items()}
    ndoc = sum(len(v) for v in docs.values()) or 1
    df = Counter()
    for cs in docs.values():
        for c in cs:
            df.update(c.keys())
    idf = {w: math.log((ndoc + 1) / (df[w] + 1)) + 1.0 for w in df}
    profiles = {}
    for a, cs in docs.items():
        prof = Counter()
        for c in cs:
            for w, n in c.items():
                prof[w] += n * idf[w]
        profiles[a] = prof
    return profiles, idf


def topic_predict(profiles, idf, query_text):
    qc = _content_vec(query_text)
    qv = {w: n * idf.get(w, 0.0) for w, n in qc.items()}
    best, bs = None, 0.0          # require POSITIVE overlap; empty/tie -> None (counts wrong, no authors[0] bias)
    for a, pv in profiles.items():
        s = _cosine(qv, pv)
        if s > bs:
            bs, best = s, a
    return best


def main():
    docs = get_docs()
    authors = sorted(docs, key=lambda a: -len(docs[a]))[:N_AUTHORS]
    if len(authors) < 4:
        print(f"only {len(authors)} usable authors -> abort", flush=True)
        return
    N = len(authors)
    chance = 1.0 / N
    if N < N_AUTHORS:
        print(f"WARNING: only {N} authors cleared the doc bar (wanted {N_AUTHORS}); chance now 1/{N}", flush=True)
    print(f"authors={authors} N={N} chance={chance:.3f} train={N_TRAIN} held={N_HELD} K={K}", flush=True)

    # per author: raw/scrubbed train + raw held + subjects
    raw_tr = {a: [docs[a][j]["raw_text"] for j in range(N_TRAIN)] for a in authors}
    scr_tr = {a: [docs[a][j]["text"] for j in range(N_TRAIN)] for a in authors}
    subj_tr = {a: [docs[a][j]["subject"] for j in range(N_TRAIN)] for a in authors}
    raw_held = {a: [docs[a][N_TRAIN + j]["raw_text"] for j in range(N_HELD)] for a in authors}
    scr_held = {a: [docs[a][N_TRAIN + j]["text"] for j in range(N_HELD)] for a in authors}   # for O3 topic channel

    # de-identify (deid) and reword (ctrl) the SCRUBBED train docs
    markers = {a: identify_markers(scr_tr[a], [d for o in authors if o != a for d in scr_tr[o][:2]])
               for a in authors}
    print("identified markers", flush=True)

    def filt(job):
        arm, a, j = job
        if arm == "deid":
            return job, rewrite_email(subj_tr[a][j], scr_tr[a][j], markers[a])
        if arm == "deid2":
            return job, rewrite_from_points(subj_tr[a][j], abstract_email(subj_tr[a][j], scr_tr[a][j]))
        if arm == "deid3":   # SUPERSET of ctrl: targeted marker removal THEN a full paraphrase pass
            return job, reword_email(subj_tr[a][j], rewrite_email(subj_tr[a][j], scr_tr[a][j], markers[a]))
        if arm == "deid4":   # deid2's bottleneck but MARKER-AWARE abstraction; step 2 == deid2's verbatim
            return job, rewrite_from_points(subj_tr[a][j],
                                            abstract_email_marker_aware(subj_tr[a][j], scr_tr[a][j], markers[a]))
        return job, reword_email(subj_tr[a][j], scr_tr[a][j])

    deid_tr = {a: [None] * N_TRAIN for a in authors}
    deid2_tr = {a: [None] * N_TRAIN for a in authors}
    deid3_tr = {a: [None] * N_TRAIN for a in authors}
    deid4_tr = {a: [None] * N_TRAIN for a in authors}
    ctrl_tr = {a: [None] * N_TRAIN for a in authors}
    by = {"deid": deid_tr, "deid2": deid2_tr, "deid3": deid3_tr, "deid4": deid4_tr, "ctrl": ctrl_tr}
    for (arm, a, j), txt in pool(filt, [(arm, a, j) for arm in ("deid", "deid2", "deid3", "deid4", "ctrl")
                                        for a in authors for j in range(N_TRAIN)]):
        by[arm][a][j] = txt
    print("rewrote deid/deid2/deid3/deid4/ctrl", flush=True)

    train_by_arm = {"original": raw_tr, "pii": scr_tr, "deid": deid_tr, "ctrl": ctrl_tr,
                    "deid2": deid2_tr, "deid3": deid3_tr, "deid4": deid4_tr}
    skills = {arm: {a: induce_skill(train_by_arm[arm][a]) for a in authors} for arm in ARMS}
    print("induced all skills", flush=True)

    # attribution: raw held query vs each arm's cards, per attributor
    arm_i = {a: i for i, a in enumerate(ARMS)}
    ajobs = [(am, arm, a, j, k) for am in ATTRIBUTORS for arm in ARMS for a in authors
             for j in range(N_HELD) for k in range(K)]

    def do_attr(job):
        am, arm, a, j, k = job
        seed = arm_i[arm] * 1000000 + authors.index(a) * 10000 + j * 100 + k
        pred = attribute(skills[arm], raw_held[a][j], seed, f"{am}-{arm}", am)
        return am, arm, a, j, pred                         # keep WHICH author (needed for empirical null)

    preds = {am: {arm: {} for arm in ARMS} for am in ATTRIBUTORS}  # -> list of K predicted author NAMES (None ok)
    for am, arm, a, j, pred in pool(do_attr, ajobs):
        preds[am][arm].setdefault((a, j), []).append(pred)
    # O1 soft accuracy: per-unit pick-fraction over the K resamples
    acc_unit = {am: {arm: {u: float(np.mean([1.0 if p == u[0] else 0.0 for p in ps]))
                           for u, ps in preds[am][arm].items()} for arm in ARMS} for am in ATTRIBUTORS}
    acc = {am: {arm: float(np.mean(list(acc_unit[am][arm].values()))) for arm in ARMS} for am in ATTRIBUTORS}

    # O3 TOPIC-ONLY channel: deterministic content-word TF-IDF attributor (no style). Profiles built from
    # the arm's PUBLISHED train text; query = SCRUBBED held (PII removed -> isolates TOPIC, not names).
    topic = {}
    for arm in ARMS:
        profiles, idf = build_topic_profiles(train_by_arm[arm])
        tpreds = {(a, j): [topic_predict(profiles, idf, scr_held[a][j])]
                  for a in authors for j in range(N_HELD)}
        tunits = sorted(tpreds)
        ttruth = [u[0] for u in tunits]
        tacc = float(np.mean([1.0 if tpreds[u][0] == u[0] else 0.0 for u in tunits]))
        tn = empirical_null([tpreds[u] for u in tunits], ttruth, authors=authors, n_perm=5000, seed=0, block=True)
        topic[arm] = {"acc": round(tacc, 3), "null_ci": tn["null_ci"], "p_value": tn["p_value"],
                      "above_chance": tn["above_chance"], "reached_chance": tn["reached_chance"]}

    # === B-GUARD (deid4 review fix): per-email content preservation on the ACTUAL rewrites ===
    # The skill-draft utility below is measured on NEW tasks via the induced 5-8 bullet profile, which is
    # robust to a few gutted source emails -> it can MISS content that deid4 dropped from the rewrites
    # (failure mode B). So judge content preservation directly on the rewrites (N_TRAIN per author) vs their
    # PII-scrubbed originals. Compare deid4 to deid2 on the SAME emails: a deid4 anonymity gain is 'real' (A)
    # only if deid4 keeps content about as well as deid2 here. Both judges (haiku headline + deepseek).
    CG_ARMS = ["deid2", "deid4"]
    cg_jobs = [(arm, a, j, jm) for arm in CG_ARMS for a in authors for j in range(N_TRAIN) for jm in JUDGES]
    cg_raw = {arm: {jm: [] for jm in JUDGES} for arm in CG_ARMS}
    for arm, jm, keep in pool(lambda jb: (jb[0], jb[3],
                                          content_preserved(subj_tr[jb[1]][jb[2]], scr_tr[jb[1]][jb[2]],
                                                            by[jb[0]][jb[1]][jb[2]], jb[3])), cg_jobs):
        cg_raw[arm][jm].append(1 if keep else 0)
    cguard = {arm: {jm: round(float(np.mean(cg_raw[arm][jm])), 3) for jm in JUDGES} for arm in CG_ARMS}
    print(f"content-preservation B-guard (per-email, on the rewrites, N_TRAIN={N_TRAIN}/author): "
          f"{ {arm: cguard[arm] for arm in CG_ARMS} }", flush=True)

    # === O4 UTILITY: does the DE-ID skill still DO THE WORK? ===
    # original-skill vs each de-id-skill draft a reply to the SAME new tasks; judges decide whether the two
    # replies have the SAME PRACTICAL EFFECT (info/decisions/outcome), IGNORING identity/style. SAME = utility
    # kept. Multi-judge -> band(min..max) + agreement; disagreement cases dumped for "why is utility unstable".
    draft = {arm: {} for arm in ARMS}
    for (arm, a, t), txt in pool(lambda jb: (jb, draft_reply(skills[jb[0]][jb[1]], TASKS[jb[2]])),
                                 [(arm, a, t) for arm in ARMS for a in authors for t in range(len(TASKS))]):
        draft[arm][(a, t)] = txt
    # NOISE FLOOR: a SECOND independent draft of the ORIGINAL skill (resampled). Two open-ended drafts of the
    # SAME skill already differ in specifics -> this is the ceiling SAME-rate can reach; read de-id RELATIVE to it.
    draft_alt = {}
    for (a, t), txt in pool(lambda jb: (jb, draft_reply_alt(skills["original"][jb[0]], TASKS[jb[1]])),
                            [(a, t) for a in authors for t in range(len(TASKS))]):
        draft_alt[(a, t)] = txt
    print("drafted skill replies on new tasks (+ original self-baseline)", flush=True)

    COMPARE = DEID_ARMS + ["__self__"]                 # __self__ = original vs its own 2nd draft = noise floor

    def do_util(job):
        arm, a, t, jm = job
        o = draft["original"][(a, t)]
        d = draft_alt[(a, t)] if arm == "__self__" else draft[arm][(a, t)]
        swap = int(hashlib.sha1(f"ord-{arm}-{a}-{t}".encode()).hexdigest(), 16) % 2 == 0
        A, B = (d, o) if swap else (o, d)            # randomize order; effect_same is symmetric
        return arm, jm, (a, t), effect_same(TASKS[t], A, B, jm)

    same_by = {arm: {jm: [] for jm in JUDGES} for arm in COMPARE}
    cells = {}                                        # (arm,a,t) -> {jm: bool}
    for arm, jm, at, same in pool(do_util, [(arm, a, t, jm) for arm in COMPARE for a in authors
                                            for t in range(len(TASKS)) for jm in JUDGES]):
        same_by[arm][jm].append(1 if same else 0)
        cells.setdefault((arm,) + at, {})[jm] = same
    util_judge = {arm: {jm: round(float(np.mean(same_by[arm][jm])), 3) for jm in JUDGES} for arm in COMPARE}
    util_band = {arm: [min(util_judge[arm].values()), max(util_judge[arm].values())] for arm in COMPARE}
    util_rate = {arm: util_judge[arm][JUDGE0] for arm in COMPARE}       # headline = independent judge (haiku)
    floor = util_rate["__self__"]                                       # noise floor (same skill, two drafts)
    agree_n = sum(1 for v in cells.values() if len(set(v.values())) == 1)
    agreement = round(agree_n / max(1, len(cells)), 3)
    disagreements = [{"arm": k[0], "author": k[1], "task": TASKS[k[2]],
                      "original_reply": draft["original"][(k[1], k[2])],
                      "deid_reply": (draft_alt[(k[1], k[2])] if k[0] == "__self__" else draft[k[0]][(k[1], k[2])]),
                      "verdicts": {jm: ("SAME" if vv else "DIFFERENT") for jm, vv in v.items()}}
                     for k, v in cells.items() if len(set(v.values())) > 1][:15]

    print(f"\n=== ATTRIBUTION (chance={chance:.3f}) ===", flush=True)
    print(f"{'attributor':22s} " + " ".join(f"{arm:>9s}" for arm in ARMS), flush=True)
    for am in ATTRIBUTORS:
        print(f"{am:22s} " + " ".join(f"{acc[am][arm]:9.3f}" for arm in ARMS), flush=True)

    print("\n=== LADDER & DE-ID (per attributor; empirical author-block null replaces 1/N) ===", flush=True)
    out_res = {}
    for am in ATTRIBUTORS:
        units = sorted(acc_unit[am]["original"].keys())
        truth_list = [u[0] for u in units]
        nullam = {arm: empirical_null([preds[am][arm][u] for u in units], truth_list, authors=authors,
                                      n_perm=5000, seed=0, block=True) for arm in ARMS}
        null_unit = {arm: empirical_null([preds[am][arm][u] for u in units], truth_list, authors=authors,
                     n_perm=5000, seed=0, block=False) for arm in ("original", "deid", "deid2", "deid3", "deid4")}
        holm_arm = dict(zip(ARMS, holm([nullam[arm]["p_value"] for arm in ARMS])))
        gap_ctrl = {arm: paired_diff_ci([acc_unit[am]["ctrl"][u] for u in units],
                                        [acc_unit[am][arm][u] for u in units])
                    for arm in ("deid", "deid2", "deid3", "deid4")}
        # headline of THIS experiment: deid4 (marker-aware) vs deid2 (blind) bottleneck. >0 => deid4 more
        # anonymous => marker knowledge adds removal ON TOP of the bottleneck. Contains 0 => bottleneck does it all.
        gap_deid2 = {arm: paired_diff_ci([acc_unit[am]["deid2"][u] for u in units],
                                         [acc_unit[am][arm][u] for u in units]) for arm in ("deid", "deid3", "deid4")}
        no = nullam["original"]
        out_res[am] = {"acc": {arm: round(acc[am][arm], 3) for arm in ARMS},
                       "empirical_null": nullam, "empirical_null_unit": null_unit,
                       "holm_p": {arm: round(holm_arm[arm], 4) for arm in ARMS}, "gap_vs_ctrl": gap_ctrl,
                       "gap_vs_deid2": gap_deid2,
                       "original_above_chance": no["above_chance"],
                       "reached_chance": {arm: nullam[arm]["reached_chance"] for arm in DEID_ARMS}}
        print(f"  [{am}] original={acc[am]['original']:.2f}(p={no['p_value']:.3f},above={no['above_chance']}) "
              f"-> pii={acc[am]['pii']:.2f} -> deid={acc[am]['deid']:.2f} -> deid2={acc[am]['deid2']:.2f} "
              f"-> deid3={acc[am]['deid3']:.2f} -> deid4={acc[am]['deid4']:.2f} -> ctrl={acc[am]['ctrl']:.2f}",
              flush=True)
        for arm in ("deid", "deid2", "deid3", "deid4"):
            na = nullam[arm]
            extra = f" gap_vs_deid2={gap_deid2[arm]}" if arm in gap_deid2 else ""
            print(f"        {arm:5s}: null95CI={na['null_ci']} p={na['p_value']:.3f} "
                  f"reached_chance={na['reached_chance']} gap_vs_ctrl={gap_ctrl[arm]}{extra} "
                  f"(unit-null reached={null_unit[arm]['reached_chance']})", flush=True)

    print("\n=== O3 TOPIC-ONLY CHANNEL (deterministic content-word TF-IDF; PII placeholders stripped) ===",
          flush=True)
    for arm in ARMS:
        t = topic[arm]
        note = "  (raw feature space -- NOT comparable to scrubbed arms)" if arm == "original" else ""
        print(f"  {arm:8s} topic_acc={t['acc']:.3f} null95CI={t['null_ci']} p={t['p_value']:.3f}{note}", flush=True)
    print("  (read: compare ONLY the post-PII arms (pii/deid/ctrl/deid2/deid3 = same scrubbed space). pii = "
          "topic identity AFTER PII removal; if pii topic_acc >> its own null (p small), identity is largely "
          "TOPIC. An arm lowers the topic channel only if its topic_acc falls vs pii. K=1 null is weak -> trust "
          "p/acc, not above_chance. SEPARATE attributor from the LLM -> read each channel vs its OWN null; do "
          "NOT equate topic_acc and style_acc levels.)", flush=True)

    print(f"\n=== O4 UTILITY (de-id skill vs original skill: SAME practical effect on new tasks? judges={JUDGES}) ===",
          flush=True)
    print(f"  judge agreement = {agreement} over {len(cells)} (arm,author,task) cells; {len(disagreements)} "
          "disagreements dumped", flush=True)
    fj = " ".join(f"{jm.split('/')[-1]}={util_judge['__self__'][jm]:.2f}" for jm in JUDGES)
    print(f"  NOISE FLOOR __self__ (original skill vs its own 2nd draft): {fj}  band={util_band['__self__']} "
          f"-> read de-id arms RELATIVE to this (D~0 = no extra degradation beyond redraft noise)", flush=True)
    for arm in DEID_ARMS:
        jd = " ".join(f"{jm.split('/')[-1]}={util_judge[arm][jm]:.2f}" for jm in JUDGES)
        print(f"  {arm:5s} effect-kept: {jd}  band={util_band[arm]}  headline={util_rate[arm]:.2f} "
              f"(floor {floor:.2f}, D{util_rate[arm]-floor:+.2f})", flush=True)
    print("  (read: SAME = de-id did NOT change the skill's practical effect. ABSOLUTE rate is low because two "
          "open-ended drafts rarely match exactly -> compare each arm to the __self__ FLOOR, not to 1.0. "
          "D~0 = de-id no worse than redrafting the same skill = effect preserved.)", flush=True)

    # headline read = the INDEPENDENT attributor (deepseek is the generator/de-identifier -> overstates the
    # de-id drop; gpt-4o-mini never generated/scrubbed the text).
    am0 = next((m for m in ATTRIBUTORS if not m.startswith("deepseek")), ATTRIBUTORS[0])  # independent attributor
    R = out_res[am0]
    print(f"\n=== READ (primary attributor = independent {am0}) ===", flush=True)
    print(f"style ladder: original {R['acc']['original']} -> pii {R['acc']['pii']} -> deid {R['acc']['deid']} "
          f"-> deid2 {R['acc']['deid2']} -> deid3 {R['acc']['deid3']} -> deid4 {R['acc']['deid4']} "
          f"-> ctrl {R['acc']['ctrl']} (empirical-chance upper ~ {R['empirical_null']['ctrl']['null_ci'][1]})",
          flush=True)
    for arm in ("deid", "deid2", "deid3", "deid4", "ctrl"):
        g = R["gap_vs_ctrl"].get(arm)
        rel = "" if arm == "ctrl" else (" BEATS ctrl" if g[0] > 0 else " WORSE than ctrl" if g[1] < 0 else " ~ctrl")
        print(f"  {arm:6s}: style_acc={R['acc'][arm]} reached={R['empirical_null'][arm]['reached_chance']} "
              f"(p={R['empirical_null'][arm]['p_value']}) | topic_acc={topic[arm]['acc']} "
              f"topic_reached={topic[arm]['reached_chance']} | effect-util={util_rate[arm]:.2f}"
              f"(floorD{util_rate[arm]-floor:+.2f}){rel}", flush=True)
    print(f"  [effect-utility floor (original skill vs its own 2nd draft) = {floor:.2f}; D~0 = de-id preserved "
          "the skill's practical effect as well as merely redrafting it does]", flush=True)
    # deid3 vs ctrl on identifiability + effect-utility vs the noise floor
    g3 = R["gap_vs_ctrl"]["deid3"]
    rel3 = "BEATS ctrl" if g3[0] > 0 else "WORSE than ctrl" if g3[1] < 0 else "matches ctrl"
    d3_ok = util_rate["deid3"] >= floor - 0.05
    print(f"-> deid3 identifiability: {rel3} ctrl (gap={g3}, reached_chance="
          f"{R['empirical_null']['deid3']['reached_chance']}); effect-utility {util_rate['deid3']:.2f} vs floor "
          f"{floor:.2f} -> {'~floor: effect preserved' if d3_ok else 'below floor: effect degraded'}.", flush=True)
    # === HEADLINE OF THIS EXPERIMENT: does MARKER-AWARE abstraction (deid4) beat the BLIND bottleneck (deid2)? ===
    # This is the SINGLE PRE-REGISTERED CONFIRMATORY comparison (review fix #1): all other gap_vs_ctrl /
    # gap_vs_deid2 CIs are EXPLORATORY/descriptive. So the one 95% CI below is not an uncorrected cherry-pick.
    gd = R["gap_vs_deid2"]["deid4"]                       # CI of mean(deid2_acc - deid4_acc); >0 => deid4 more anon
    # B-guard (review fix #3): arbitrate A-vs-B on the per-email content preservation of the ACTUAL rewrites,
    # NOT the downstream skill-draft utility. deid4 'really' more anonymous only if it keeps content about as
    # well as deid2 on the same 60 emails. Require it under BOTH judges (review fix: not single-judge fragile).
    cg2, cg4 = cguard["deid2"], cguard["deid4"]
    content_ok = all(cg4[jm] >= cg2[jm] - 0.10 for jm in JUDGES)      # deid4 not meaningfully worse than deid2 (both judges)
    util_ok = all(util_judge["deid4"][jm] >= util_judge["__self__"][jm] - 0.05 for jm in JUDGES)
    if gd[0] > 0:
        verdict = ("A: marker-aware ADDS removal on top of the bottleneck (deid4 more anonymous than deid2, "
                   "and content preserved as well as deid2)" if content_ok else
                   "B: deid4 more anonymous but its per-email content preservation fell below deid2's -> it "
                   "won by DROPPING CONTENT, not a clean de-id gain")
    elif gd[1] < 0:
        verdict = "deid4 WORSE than deid2 (marker-aware abstraction hurt -- unexpected)"
    else:
        verdict = ("C: deid4 ~ deid2 (CI contains 0) -> once you regenerate from points, knowing the markers "
                   "adds nothing; the BOTTLENECK does all the work")
    print(f"-> [CONFIRMATORY] deid4 vs deid2 (marker-aware vs blind bottleneck): style {R['acc']['deid4']} vs "
          f"{R['acc']['deid2']}, gap_vs_deid2={gd}", flush=True)
    print(f"   B-guard per-email content-preserved deid2={cg2} deid4={cg4} (content_ok={content_ok}); "
          f"skill-effect util deid4={util_judge['deid4']} (util_ok={util_ok})", flush=True)
    print(f"   VERDICT {verdict}", flush=True)
    # dataset diagnosis (O3): is the identity mostly TOPIC?  (compare only post-PII = same scrubbed space)
    print(f"[O3 dataset] topic channel (scrubbed space): pii={topic['pii']['acc']}(p={topic['pii']['p_value']}) "
          f"-> deid={topic['deid']['acc']} deid2={topic['deid2']['acc']} deid3={topic['deid3']['acc']} "
          f"deid4={topic['deid4']['acc']} ctrl={topic['ctrl']['acc']}. If pii topic_acc >> its null, identity is "
          "largely TOPIC; if ctrl lowers "
          "topic_acc below the content-preserving arms, ctrl wins by DRIFTING TOPIC -- the dataset confound that "
          "caps how much targeted style-removal can beat paraphrase here.", flush=True)

    out = {"authors": authors, "N": N, "chance": round(chance, 3), "K": K,
           "attributors": {am: out_res[am] for am in ATTRIBUTORS},
           "topic_only": topic,
           "utility": {"definition": "de-id skill vs original skill draft replies to shared new tasks; judge "
                       "decides SAME practical effect (ignoring identity/style). SAME = utility kept.",
                       "judges": JUDGES, "headline_judge": JUDGE0, "per_judge": util_judge, "band": util_band,
                       "noise_floor_self": floor, "agreement": agreement, "n_tasks": len(TASKS)},
           "content_guard": {"definition": "per-email content preservation on the rewrites (N_TRAIN/author) vs "
                             "PII-scrubbed originals (B-guard for deid4-vs-deid2): YES = all facts/requests/numbers/deadlines/"
                             "decisions kept, ignoring style. deid4 'real' anonymity gain only if it preserves "
                             "content about as well as deid2.", "judges": JUDGES, "per_arm": cguard,
                             "confirmatory_comparison": "deid4 vs deid2 (gap_vs_deid2); all other gaps exploratory"},
           "per_author": {am: {arm: {a: round(float(np.mean([acc_unit[am][arm][(a, j)]
                          for j in range(N_HELD)])), 3) for a in authors}
                          for arm in ("deid", "deid2", "deid3", "deid4")} for am in ATTRIBUTORS}}
    fn = ROOT / "results" / "deid_enron.json"
    fn.parent.mkdir(parents=True, exist_ok=True)
    fn.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results" / "deid_enron_utility_disagree.json").write_text(
        json.dumps(disagreements, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {fn}  (+ utility disagreements dump)", flush=True)


if __name__ == "__main__":
    main()
