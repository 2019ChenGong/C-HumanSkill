"""Load and clean Enron sent-mail (PII scrub, quoted-line strip, dedup)."""
import re
import email
import tarfile
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
TARBALL = ROOT / "data" / "enron" / "enron_mail_20150507.tar.gz"

TARGET = ["dasovich-j", "germany-c", "kaminski-v", "mann-k", "shackleton-s"]
SENT = {"_sent_mail", "sent", "sent_items"}
N_PER = 22
N_TRAIN = 10
MIN_WORDS, MAX_WORDS, MIN_CHARS = 40, 1500, 200

# ---------------------------------------------------------------- regex bank
EMAIL_RE = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
# enron internal routing handles like  Chris Germany@ECT  /  Vince J Kaminski/HOU/ECT@ECT
ENRON_HANDLE_RE = re.compile(r"\b[\w. ]+?/[A-Z]{2,}(?:/[A-Z]{2,})*@[A-Z]{2,}\b")
ENRON_AT_RE = re.compile(r"\b[A-Z][A-Za-z]+(?: [A-Z]\.?)? [A-Z][A-Za-z]+@(?:ECT|EES|ENRON|Enron|Corp|NA)\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
# phone / fax: 713-853-1234, (713) 853 1234, 713.853.1234, x-1234 ext
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[ .\-]?)?(?:\(?\d{3}\)?[ .\-]?)?\d{3}[ .\-]\d{4}(?!\d)")
EXT_RE = re.compile(r"\b(?:ext|x|extension)\.?\s*[:#]?\s*\d{3,5}\b", re.I)
# trailing parenthetical / bracket placeholder collapsing
PLACEHOLDER_TOKENS = ["EMAIL", "PHONE", "URL", "PERSON", "ORG", "LOC", "ENTITY"]

# Enron-specific codenames / systems / SPEs / products -> [ENTITY]
ENRON_ENTITIES = [
    "Raptor", "Raptors", "Chewco", "JEDI", "JEDI II", "Azurix", "Sitara", "GMS", "Enpower",
    "EnronOnline", "Enron Online", "EOL", "LJM", "LJM1", "LJM2", "Whitewing", "Osprey",
    "Mariner", "Condor", "Talon", "Timberwolf", "Porcupine", "Braveheart", "Rawhide",
    "Yosemite", "Marlin", "Firefly", "EES", "ENA", "Enron Net Works", "ENW", "EBS",
    "Enron Broadband", "TWA", "Dabhol", "DPC", "Transwestern", "Northern Border",
    "RAC", "Cinergy", "CERA", "FERC", "CPUC", "ISO", "PX", "Sundevil", "Backbone",
    "RADR", "Cuiaba", "Nahanni", "Nile", "Hawaii", "Bob West", "Margaux", "Catalytica",
    "Risk Management Protocol", "DASH", "DPR", "tagg", "TAGG",
]
# build a case-insensitive word-boundary alternation, longest first
_ent_sorted = sorted(set(ENRON_ENTITIES), key=len, reverse=True)
ENTITY_RE = re.compile(r"\b(" + "|".join(re.escape(e) for e in _ent_sorted) + r")\b")

# sign-off cue words that precede a name line we should drop the *name* of
SIGNOFF_CUES = re.compile(
    r"^(best( regards)?|regards|thanks( a lot| again)?|thank you|cheers|sincerely|"
    r"warm(est)? regards|talk soon|take care|love|yours( truly)?|respectfully|"
    r"best wishes|ciao|later|cordially|kind regards)[\s,.!]*$", re.I)

# legal / disclaimer / marketing footer triggers -> cut from here down
FOOTER_TRIGGERS = re.compile(
    r"(?i)(this e-?mail.*(confiden|intend|privile|proprietary)|"
    r"the information (contained|transmitted) in this|"
    r"if you (are not|have received) this (message|e-?mail|communication) in error|"
    r"named addressee|legally exempt from disclosure|"
    r"if you wish not to receive|donotemail|to unsubscribe|"
    r"future e-?mail notifications|\*{6,}|={6,}|_{10,}|-{20,})")


def author_name_from_addr(addr):
    """jeff.dasovich@enron.com -> {'jeff','dasovich','jeff dasovich'} tokens to mask.
    Used uniformly for every author; no special-casing."""
    if not addr:
        return []
    m = EMAIL_RE.search(addr)
    local = (m.group(0).split("@")[0] if m else addr).lower()
    local = re.sub(r"[^a-z.]+", ".", local)
    parts = [p for p in local.split(".") if len(p) >= 2]
    names = set(parts)
    if len(parts) >= 2:
        names.add(" ".join(parts))
        names.add(parts[0] + " " + parts[-1])
    return names


def _strip_quotes_and_headers(payload):
    """Drop quoted text, forwarded/original-message blocks, embedded reply headers,
    and routing lines -- BEFORE masking. Keeps only the author's own prose."""
    out = []
    for ln in payload.splitlines():
        s = ln.strip()
        if re.match(r"^-+\s*(Original Message|Forwarded by|----)", s, re.I):
            break
        if re.match(r"^On .+wrote:$", s):
            break
        if s.startswith(">"):
            continue
        if re.match(r"^(From|To|Sent|Cc|Bcc|Subject|Date)\s*:", s, re.I):
            continue
        # embedded Notes-style reply header line: a SHORT line that STARTS with a routing handle
        #   "Chris Germany@ECT"  /  "Dave Scott@EES  12/19/2000 10:09 AM"  /  "Vince J Kaminski/HOU/ECT@ECT"
        # (only drop header-LIKE lines; a body sentence that merely mentions a handle is masked later)
        if len(s.split()) <= 8 and (ENRON_AT_RE.match(s) or ENRON_HANDLE_RE.match(s)):
            continue
        # bare date/time stamp lines that head a quoted block
        if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*(AM|PM)?\s*$", s, re.I):
            continue
        out.append(ln)
    return "\n".join(out)


def _cut_footer(text):
    """Remove signature blocks / legal disclaimers / marketing footers from first trigger down."""
    m = FOOTER_TRIGGERS.search(text)
    if m:
        text = text[:m.start()]
    return text


def _mask_signoff_names(text, author_names):
    """Handle 'Best,\\nJeff' / 'Thanks, Margaret' / 'Love, Mother': the name(s) following a
    sign-off cue (or a trailing short name line) -> [PERSON]. Done for ALL authors uniformly."""
    lines = text.splitlines()
    n = len(lines)
    for i, ln in enumerate(lines):
        s = ln.strip()
        # inline 'Best, Jeff' / 'Thanks a lot ...Margaret'
        m = re.match(r"^(best( regards)?|regards|thanks(?:[ ,.!a-z]*)?|thank you|cheers|sincerely|"
                     r"love|cordially|warm(?:est)? regards|kind regards|yours(?: truly)?)"
                     r"[\s,.!]*[-.]*\s*([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)\s*$", s, re.I)
        if m:
            lines[i] = re.sub(re.escape(m.group(3)) + r"\s*$", "[PERSON]", ln)
            continue
        # cue word on its own line -> next non-empty short line is the name
        if SIGNOFF_CUES.match(s):
            for j in range(i + 1, min(i + 3, n)):
                t = lines[j].strip()
                if not t:
                    continue
                if len(t.split()) <= 3 and re.match(r"^[A-Z][a-zA-Z.'\- ]+$", t):
                    lines[j] = re.sub(r"\S.*", "[PERSON]", lines[j])
                break
    text = "\n".join(lines)
    # also: a final line that's just a short capitalized name (no cue) -> sign-off
    tl = text.rstrip().splitlines()
    if tl:
        last = tl[-1].strip()
        if 0 < len(last.split()) <= 3 and re.match(r"^[A-Z][a-zA-Z.'\-]+(?: [A-Z][a-zA-Z.'\-]+)?$", last) \
                and last.lower() not in {"thanks", "ok", "thank you"}:
            tl[-1] = "[PERSON]"
            text = "\n".join(tl)
    return text


# spaCy NER is loaded lazily; if unavailable we fall back to a regex name masker.
@lru_cache(maxsize=1)
def _nlp():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["lemmatizer", "tagger", "parser", "attribute_ruler"])
    except Exception:  # noqa: BLE001
        return None


# fallback / residual-name 2nd pass: TitleCase token PAIRS that look like person names.
NAME_PAIR_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+\b")
# words that, if EITHER token of a TitleCase pair is one of them, mark the pair NON-name
# (so "Payment Date", "Managing Director", "Service List", "Master Swap" are spared).
NONNAME_WORDS = set(w.lower() for w in """
Monday Tuesday Wednesday Thursday Friday Saturday Sunday Jan Feb Mar Apr May Jun Jul Aug Sep Sept Oct Nov Dec
January February March April May June July August September October November December
The This That These Those We You They He She It If And But Or So For Please Thanks Thank Best Regards Dear Hi Hello
Date Time Number No Order Payment Invoice Account Service Master Swap Agreement Agreements Rate Dept Department
Director Manager Managing President Vice Chief Officer Counsel Group Team Desk Zone Comm Committee List Path
Inquiries Inquiry Report Protocol Notice Subject Update Status Summary Draft Final Version Section Schedule Exhibit
North South East West Left Right Next Last First Second Third Other Same New Old Mr Mrs Ms Dr Re Fw Fwd
""".split())


def _looks_like_name_pair(m):
    toks = m.group(0).split()
    words = [t.rstrip(".").lower() for t in toks]
    return not any(w in NONNAME_WORDS for w in words)


def _mask_residual_name_pairs(text):
    """2nd pass over what NER missed: TitleCase pairs unlikely to be business phrases -> [PERSON].
    Conservative: spares calendar/business/common bigrams (NONNAME_WORDS)."""
    return NAME_PAIR_RE.sub(lambda m: "[PERSON]" if _looks_like_name_pair(m) else m.group(0), text)


def _mask_persons_regex(text):
    return _mask_residual_name_pairs(text)


def _mask_with_ner(text, extra_names):
    nlp = _nlp()
    if nlp is None:
        out = _mask_persons_regex(text)
    else:
        doc = nlp(text)
        spans = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                spans.append((ent.start_char, ent.end_char, "[PERSON]"))
            elif ent.label_ in ("ORG", "NORP", "FAC"):
                spans.append((ent.start_char, ent.end_char, "[ORG]"))
            elif ent.label_ in ("GPE", "LOC"):
                spans.append((ent.start_char, ent.end_char, "[LOC]"))
        spans.sort(reverse=True)
        for s, e, tag in spans:
            text = text[:s] + tag + text[e:]
        # 2nd pass: TitleCase name-pairs NER missed (small model misses uncommon/non-Western names)
        text = _mask_residual_name_pairs(text)
    # extra explicit names (author own name + first names) -> [PERSON]
    for nm in sorted(extra_names, key=len, reverse=True):
        if not nm:
            continue
        text = re.sub(r"\b" + re.escape(nm) + r"\b", "[PERSON]", text, flags=re.I)
    return text


def _collapse_placeholders(text):
    """Stop placeholder COUNTS from encoding identity: collapse adjacent duplicates and never
    number them. '[PERSON] [PERSON]' / '[PERSON], [PERSON]' / '[PERSON][PERSON]' -> '[PERSON]'."""
    tok = "|".join(PLACEHOLDER_TOKENS)
    # remove any stray [PERSON]'s -> [PERSON]
    text = re.sub(r"\[(" + tok + r")\]('s|s)?", r"[\1]", text)
    # collapse runs separated by spaces/punctuation
    text = re.sub(r"(\[(?:" + tok + r")\])(\s*[,/&]?\s*\1)+", r"\1", text)
    # collapse mixed PERSON/ORG/LOC runs to a single token of the first kind (avoid count leak)
    text = re.sub(r"(\[(?:" + tok + r")\])(\s*[,/&]?\s*\[(?:" + tok + r")\])+",
                  lambda m: m.group(1), text)
    return text


def scrub(text, author_name=None):
    """Mask every identifier in `text` with a TYPED placeholder, uniformly.
    author_name: optional set/list/str of name tokens (the author's own + first-name sign-offs)
                 to force-mask in addition to NER. Pass the SAME derivation for every author."""
    if author_name is None:
        extra = set()
    elif isinstance(author_name, str):
        extra = {author_name}
    else:
        extra = set(author_name)

    t = text
    # 1. structural: drop quoted/forwarded/header/routing lines, then cut footers/disclaimers
    t = _strip_quotes_and_headers(t)
    t = _cut_footer(t)
    # 2. contact identifiers
    t = EMAIL_RE.sub("[EMAIL]", t)
    t = ENRON_HANDLE_RE.sub("[PERSON]", t)
    t = ENRON_AT_RE.sub("[PERSON]", t)
    t = URL_RE.sub("[URL]", t)
    t = EXT_RE.sub("[PHONE]", t)
    t = PHONE_RE.sub("[PHONE]", t)
    # 3. Enron codenames/systems/SPEs -> [ENTITY] (before NER so NER doesn't mislabel them)
    t = ENTITY_RE.sub("[ENTITY]", t)
    # 4. sign-off names (Best,\nJeff / Love, Mother) -> [PERSON]
    t = _mask_signoff_names(t, extra)
    # 5. NER persons/orgs/locations + explicit author names -> typed placeholders
    t = _mask_with_ner(t, extra)
    # 6. collapse placeholder runs so counts can't encode identity
    t = _collapse_placeholders(t)
    # 7. tidy whitespace
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def parse_email(raw):
    """Return (from_addr, subject, body) from an RFC822 string, or None."""
    try:
        msg = email.message_from_string(raw)
    except Exception:  # noqa: BLE001
        return None
    subject = (msg.get("Subject") or "").strip()
    frm = msg.get("From") or ""
    payload = msg.get_payload()
    if not isinstance(payload, str):
        return None
    return frm, subject, payload


def clean_email_raw(raw):
    """RAW-condition cleaner = probe_enron's clean_email (strip quotes/headers/sig, length filter).
    NO PII masking. Returns {text, subject, frm} or None."""
    parsed = parse_email(raw)
    if not parsed:
        return None
    frm, subject, payload = parsed
    body = _strip_quotes_and_headers(payload)
    body = re.split(r"(?i)this e-?mail.*?(confidential|intended)", body)[0]
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    w = len(body.split())
    if w < MIN_WORDS or w > MAX_WORDS or len(body) < MIN_CHARS:
        return None
    return {"text": body, "subject": subject, "frm": frm}


def load_enron_cleaned(target=None, n_per=N_PER, do_scrub=True, tarball=TARBALL):
    """Collect sent emails for each author in `target` and return per-author docs:
        { author: [ {text, subject, frm, raw_text}, ... ] }
    Length-filter is applied on the RAW body (same as probe_enron) so RAW and CLEANED
    use the SAME doc set. If do_scrub: text = scrub(raw_body, author_names); else text = raw_body.
    The author_name set is derived UNIFORMLY from the From: address of each author."""
    target = target or TARGET
    got = {a: [] for a in target}
    with tarfile.open(tarball, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")
            if len(parts) < 4 or parts[0] != "maildir":
                continue
            a, folder = parts[1], parts[2].lower()
            if a not in target or folder not in SENT or len(got[a]) >= n_per:
                continue
            try:
                raw = tf.extractfile(m).read().decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                continue
            rec = clean_email_raw(raw)
            if not rec:
                continue
            names = author_name_from_addr(rec["frm"])
            rec["raw_text"] = rec["text"]
            if do_scrub:
                rec["text"] = scrub(rec["raw_text"], author_name=names)
                # after scrub, re-check it still has enough words (don't keep gutted docs)
                if len(rec["text"].split()) < 20:
                    continue
            got[a].append(rec)
            if all(len(got[x]) >= n_per for x in target):
                break
    return got


if __name__ == "__main__":
    # quick smoke
    sample = ("Best,\nJeff\n\nPlease call Keith if he has not contacted you. Vince Kaminski "
              "at Enron in Houston re Raptor and Chewco. Reach me at 713-853-1234 or "
              "jeff.dasovich@enron.com. Chris Germany@ECT wrote earlier.")
    print("RAW:\n", sample)
    print("\nSCRUBBED:\n", scrub(sample, author_name=author_name_from_addr("jeff.dasovich@enron.com")))
