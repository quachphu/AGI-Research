"""
Build the CSULB training lanes, held-out set and test questions.

    .venv/bin/python tools/csulb/build_corpus.py      # after extract.py

From school_policies_data/{documents,sections}.jsonl it writes:

    train/csulb_policies/   the cleaned documents: what the policies say
    train/csulb_qa/         questions and answers in the chat format the
                            server uses: how to answer from that knowledge
    train/csulb_qa_open/    the same, with the matching policy passage in the
                            question: how to answer from a passage put in front
    val/csulb_qa*/          one wording of every fact, never trained on
    heldout_mix/            data/val (hard links) + val/: what train.py scores
    eval/questions.jsonl    a sample of the held-out wordings for eval.py

Every question and answer comes from rules over the policy text - headings,
definitions, policy numbers, dates - and every answer is the policy's own
words. No other model is used. Each fact is asked several ways, as in
corpora/self_knowledge.yaml: one wording repeated is what a model recites,
several wordings of one fact is what it learns.
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qa_format import U0, U1, B0, B1, open_book_prompt  # noqa: E402

ANSWER_MAX = 520
GENERIC_HEADINGS = {"preamble", "introduction", "purpose", "general", "background",
                    "scope", "policy", "overview", "statement", "policy statement",
                    "summary", "authority", "applicability", "rationale"}
SKIP_HEADINGS = re.compile(r"^(table of contents|contents|appendix|attachment|"
                           r"exhibit|signature|approved|history|revision|page \d)",
                           re.I)


# ---------------------------------------------------------------- names
def clean_title(t):
    t = " ".join(t.split())
    t = re.sub(r"^Policy Statement\s*[-:]?\s*\d{2}-\d{2}\s*", "", t, flags=re.I)
    t = re.sub(r"\s*\((Supersedes|See|Replaces|Formerly)[^)]*\)", "", t, flags=re.I)
    t = re.sub(r"\s*[(\[](PDF|DOCX?|RTF)[)\]]", "", t, flags=re.I)
    # "Double Majors, Policy on" -> "Policy on Double Majors"
    head, sep, tail = t.rpartition(", ")
    if sep and head and tail.split() and len(tail.split()) <= 7 and \
            tail.split()[-1].lower() in ("on", "in", "of", "for", "to", "the"):
        tail = " ".join(w.lower() if i and w.lower() in ("on", "in", "of", "for", "to", "the", "and")
                        else w for i, w in enumerate(tail.split()))
        t = f"{tail[:1].upper() + tail[1:]} {head}"
    t = t.strip(" -:,")
    return t[:1].upper() + t[1:] if t else t


def clean_heading(h):
    if not h:
        return None
    h = re.sub(r"^(\d+(\.\d+)*\.?|[IVX]+\.|[A-Z]\.|Article [IVX\d]+\.?|"
               r"Section \d+(\.\d+)*\.?|Regulation [IVX]+\.?)\s*[-:]?\s*", "", h.strip(),
               flags=re.I)
    h = h.strip(" .:-")
    if len(h) < 5 or len(h) > 90 or "|" in h or SKIP_HEADINGS.match(h):
        return None
    letters = [c for c in h if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8:
        small = {"and", "or", "of", "the", "for", "to", "in", "on", "a", "an", "by", "with"}
        words = h.lower().split()
        h = " ".join(w if (i and w in small) else w[:1].upper() + w[1:]
                     for i, w in enumerate(words))
        h = re.sub(r"\b(Csulb|Csu|Asi|Ferpa|Rsca|Ps)\b", lambda m: m.group(1).upper(), h)
    return h


OFFICES = [  # (words in the index link or address, who publishes it)
    (("academic senate", "/academic-senate"), "the CSULB Academic Senate"),
    (("associated students", "asicsulb.org"), "Associated Students, Inc. (ASI)"),
    (("research foundation",), "the CSULB Research Foundation"),
    (("foundation",), "the Long Beach State Foundation"),
    (("police", "clery", "safety plan", "security report"), "the CSULB University Police"),
    (("research and sponsored", "office-of-research"), "the CSULB Office of Research and Economic Development"),
    (("government relations",), "CSULB Government Relations"),
    (("campus rules", "student-affairs", "free speech"), "CSULB Student Affairs"),
    (("social media",), "CSULB Strategic Communications"),
    (("nondiscrimination", "non-discrimination", "title ix", "equity-compliance"),
     "the CSULB Office of Equity and Compliance"),
    (("calstate.edu", "systemwide", "faqs"), "the California State University (CSU) system"),
    (("accessibility",), "CSULB Information Technology (accessibility)"),
]


def office_of(d):
    hay = f"{d.get('section') or ''} {d.get('url') or ''} {d.get('src') or ''}".lower()
    for keys, name in OFFICES:
        if any(k in hay for k in keys):
            return name
    return "CSULB"


def ref_of(d):
    """How an answer names its source."""
    num = d.get("policy_number")
    t = d["ctitle"]
    if num:
        return f"CSULB Policy Statement {num[3:]} ({t})"
    return f"the {t}" if not t.lower().startswith(("the ", "csu ", "csulb ")) else t


def in_sentence(h):
    """A heading as it reads inside a question: lower case, acronyms kept."""
    return " ".join(w if (len(w) > 1 and w.isupper()) or re.match(r"^[A-Z]{2}", w) else w.lower()
                    for w in h.split())


STOP = {"the", "and", "for", "with", "from", "that", "this", "policy", "policies",
        "csulb", "university", "section", "procedures", "general"}


def heading_fits(h, body):
    """A heading some of whose words the text under it actually uses."""
    words = [w for w in re.findall(r"[a-z]{4,}", h.lower()) if w not in STOP]
    if not words:
        return True
    low = body.lower()
    return any(w[:6] in low for w in words)


def junk(text):
    """A table of contents or a form, not policy text."""
    return len(re.findall(r"\.{5,}|_{5,}", text)) >= 3 or \
        sum(c.isalpha() for c in text) < 0.55 * len(text)


# ---------------------------------------------------------------- answers
def sentences(text):
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = re.sub(r"\n- ", "; ", text)
    text = re.sub(r"\s*\n\s*", " ", text).strip()
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])", text) if s.strip()]


def excerpt(text, limit=ANSWER_MAX):
    out = ""
    for s in sentences(text):
        if out and len(out) + len(s) + 1 > limit:
            break
        out = (out + " " + s).strip()
        if len(out) >= limit * 0.55:
            break
    if len(out) > limit:
        cut = out.rfind(" ", 0, limit)
        out = out[:cut if cut > 0 else limit].rstrip(",;:") + " ..."
    return out


def answer(d, label, body):
    lead = f"According to {ref_of(d)}"
    if label:
        lead += f", {label}"
    return f"{lead}: {body}\nSource: {d['url']}"


# ---------------------------------------------------------------- facts
DEF_RES = [
    re.compile(r"(?:^|\. )\"?([A-Z][A-Za-z/ -]{2,50}?)\"? (?:is|are|shall be) defined as ([^.]{15,400}\.)"),
    re.compile(r"(?:^|\. )\"([A-Za-z][A-Za-z/ -]{2,50}?)\" (?:means|refers to|is) ([^.]{15,400}\.)"),
    re.compile(r"(?:^|\n)(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][A-Za-z/ -]{2,40})\.\s+((?:An?|The) [^.]{15,400}\.)"),
    re.compile(r"(?:^|\n)(?:\d+(?:\.\d+)*\.?\s+)?([A-Z][A-Za-z/ -]{2,40}):\s+((?:An?|The) [^.]{15,400}\.)"),
]
NOT_TERMS = {"the", "this", "it", "there", "that", "these", "policy", "note", "example",
             "please note", "purpose", "background", "scope", "summary", "result",
             "exception", "exceptions", "rationale", "response", "authority"}


def definitions(text):
    seen = set()
    for i, rx in enumerate(DEF_RES):
        for m in rx.finditer(text):
            term, body = m.group(1).strip(" -"), m.group(2).strip()
            key = term.lower()
            if key in seen or key in NOT_TERMS or len(term.split()) > 6 or key.startswith(("the ", "this ")):
                continue
            # "Center. An academic center is ..." - a defined term is a Title Case name
            if i >= 2 and not all(w[:1].isupper() for w in term.split() if len(w) > 3):
                continue
            seen.add(key)
            yield term, body


def section_facts(d, s):
    """Questions about one section of a policy."""
    t = d["ctitle"]
    if junk(s["text"]):
        return []
    h = clean_heading(s.get("heading"))
    if h and (h.lower() == t.lower() or h.lower() in t.lower() or not heading_fits(h, s["text"])):
        h = None
    label = None
    if s.get("number") and h:
        label = f"section {s['number']} ({h})"
    elif h:
        label = f"section \"{h}\""
    body = excerpt(s["text"])
    if len(body) < 60:
        return []
    facts = []
    if h and h.lower() not in GENERIC_HEADINGS:
        hl = in_sentence(h)
        qs = [f"What does the CSULB {t} say about {hl}?",
              f"What is the CSULB policy on {hl}?",
              f"Tell me about {hl} in the {t}.",
              f"Can you explain the \"{h}\" part of the {t}?",
              f"I have a question about {hl} at CSULB. What does the policy say?",
              f"{h} - what are the rules at CSULB?"]
        facts.append(dict(kind="topic", questions=qs, answer=answer(d, label, body)))
    elif h:
        qs = [f"What is the {h.lower()} of the CSULB {t}?",
              f"What does the {h.lower()} section of the {t} say?",
              f"Explain the {h.lower()} of the {t}."]
        facts.append(dict(kind="topic", questions=qs, answer=answer(d, label, body)))
    for term, dbody in definitions(s["text"]):
        tl = in_sentence(term)
        qs = [f"What is {tl} at CSULB?",
              f"How does CSULB define {tl}?",
              f"What does \"{term}\" mean in the {t}?",
              f"Define {tl} according to CSULB policy.",
              f"What counts as {tl} under the {t}?"]
        facts.append(dict(kind="definition", questions=qs,
                          answer=answer(d, label, f"{term}: {excerpt(dbody, 420)}")))
    for f in facts:
        f["section"] = s["id"]
        f["passage"] = s["text"]
        f["heading"] = h
    return facts


def doc_facts(d, first):
    """Questions about a document as a whole: what it is, its number, date, source."""
    t, num, url = d["ctitle"], d.get("policy_number"), d["url"]
    facts = []
    summary = excerpt(first["text"]) if first else ""
    head = []
    if d.get("supersedes"):
        head.append(f"Supersedes: {d['supersedes']}")
    if d.get("approved_by"):
        head.append(f"Approved by: the {d['approved_by']}")
    head.append(f"Published by: {office_of(d)}")
    passage = "\n".join(head) + ("\n" + first["text"] if first else "")
    sid = first["id"] if first else None
    if summary:
        qs = [f"What is the CSULB {t} about?",
              f"Summarize the {t}.",
              f"Give me an overview of the CSULB {t}.",
              f"What is the purpose of the {t}?",
              f"Explain the CSULB {t} in short."]
        if num:
            qs += [f"What is CSULB policy {num}?", f"What is Policy Statement {num[3:]} about?"]
        facts.append(dict(kind="overview", questions=qs, answer=answer(d, None, summary)))
    head = f"{t}"
    if num:
        bits = [f"The {head} is CSULB Policy Statement {num[3:]}"]
        if d.get("date"):
            bits.append(f"dated {d['date']}")
        a = ", ".join(bits) + f".\nSource: {url}"
        facts.append(dict(kind="number", answer=a,
                          questions=[f"What is the policy number of the CSULB {t}?",
                                     f"Which policy statement number is the {t}?",
                                     f"What PS number covers the {t}?",
                                     f"Which CSULB policy statement is about {t.lower() if not t[:2].isupper() else t}?"]))
    if d.get("date"):
        who = f"Policy Statement {num[3:]}" if num else f"the {t}"
        facts.append(dict(kind="date", answer=f"{who[:1].upper() + who[1:]} ({t}) is dated {d['date']}.\nSource: {url}",
                          questions=[f"When was the CSULB {t} issued?",
                                     f"What is the date of the {t}?",
                                     f"How recent is the CSULB {t}?"]
                          + ([f"When was {num} approved?"] if num else [])))
    if d.get("supersedes"):
        facts.append(dict(kind="supersedes",
                          answer=f"The {t} supersedes {d['supersedes'].rstrip('.')}.\nSource: {url}",
                          questions=[f"What does the CSULB {t} supersede?",
                                     f"Which older policies does the {t} replace?",
                                     f"Is the {t} replacing an earlier policy?"]))
    if d.get("approved_by"):
        rec = f", recommended by the {d['recommended_by']}" if d.get("recommended_by") else ""
        facts.append(dict(kind="approval",
                          answer=f"The {t} was approved by the {d['approved_by']}{rec}.\nSource: {url}",
                          questions=[f"Who approved the CSULB {t}?",
                                     f"Who signed off on the {t}?",
                                     f"Which body approved the {t}?"]))
    office = office_of(d)
    facts.append(dict(kind="where",
                      answer=f"The official text of the {t} is published by {office} at {url}",
                      questions=[f"Where can I find the CSULB {t}?",
                                 f"Where is the official {t} posted?",
                                 f"Give me the link to the CSULB {t}.",
                                 f"Which office publishes the {t}?"]
                      + ([f"Where can I read {num}?"] if num else [])))
    for f in facts:
        f["section"] = sid
        f["passage"] = passage
    return facts


# ---------------------------------------------------------------- writing
def chat(q, a):
    return f"{U0}\n{q}\n{U1}\n{B0}\n{a}\n{B1}\n"


def write_shards(folder, items, shard_chars=400_000):
    os.makedirs(folder, exist_ok=True)
    for old in glob.glob(os.path.join(folder, "*.txt")):
        os.remove(old)
    buf, n, total = [], 0, 0
    for it in items:
        buf.append(it)
        total += len(it)
        if sum(len(x) for x in buf) >= shard_chars:
            with open(os.path.join(folder, f"part-{n:04d}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(buf))
            buf, n = [], n + 1
    if buf:
        with open(os.path.join(folder, f"part-{n:04d}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(buf))
        n += 1
    return n, total


def hardlink_tree(src, dst):
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        os.makedirs(os.path.join(dst, rel), exist_ok=True)
        for fn in files:
            a, b = os.path.join(root, fn), os.path.join(dst, rel, fn)
            if not os.path.exists(b):
                try:
                    os.link(a, b)
                except OSError:
                    shutil.copy2(a, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--data", default="school_policies_data")
    ap.add_argument("--general-val", default="data/val")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wordings", type=int, default=3,
                    help="training wordings kept per fact (one more is held out)")
    ap.add_argument("--eval-n", type=int, default=60)
    a = ap.parse_args()
    D, rng = a.data, random.Random(a.seed)

    docs = {}
    with open(os.path.join(D, "documents.jsonl"), encoding="utf-8") as f:
        for l in f:
            d = json.loads(l)
            if d.get("kept"):
                d["ctitle"] = clean_title(d["title"])
                docs[d["id"]] = d
    secs = {}
    with open(os.path.join(D, "sections.jsonl"), encoding="utf-8") as f:
        for l in f:
            s = json.loads(l)
            secs.setdefault(s["doc"], []).append(s)

    # ---- lane 1: the documents themselves
    pol = os.path.join(D, "train", "csulb_policies")
    os.makedirs(pol, exist_ok=True)
    for old in glob.glob(os.path.join(pol, "*.txt")):
        os.remove(old)
    raw_chars = 0
    for d in docs.values():
        shutil.copyfile(os.path.join(D, d["file"]), os.path.join(pol, os.path.basename(d["file"])))
        raw_chars += d["chars"]

    # ---- facts
    facts = []
    for doc_id, d in docs.items():
        ss = secs.get(doc_id, [])
        if not ss:
            continue
        first = next((s for s in ss if len(s["text"]) > 200 and not junk(s["text"])), None)
        for f in doc_facts(d, first):
            f["doc"] = doc_id
            facts.append(f)
        for s in ss:
            for f in section_facts(d, s):
                f["doc"] = doc_id
                facts.append(f)
    for i, f in enumerate(facts):
        f["id"] = i
        f["questions"] = list(dict.fromkeys(f["questions"]))

    train_closed, train_open, val_closed, val_open, held = [], [], [], [], []
    for f in facts:
        qs = f["questions"][:]
        rng.shuffle(qs)
        hold = qs.pop() if len(qs) >= 3 else None
        d = docs[f["doc"]]
        hd = f.get("heading")
        for q in qs[:a.wordings]:
            train_closed.append(chat(q, f["answer"]))
            train_open.append(chat(open_book_prompt(q, d, f["passage"], hd), f["answer"]))
        if hold:
            val_closed.append(chat(hold, f["answer"]))
            val_open.append(chat(open_book_prompt(hold, d, f["passage"], hd), f["answer"]))
            held.append(dict(q=hold, kind=f["kind"], fact=f["id"], section=f["section"],
                             doc=f["doc"], title=d["ctitle"], url=d["url"],
                             policy_number=d.get("policy_number"), answer=f["answer"],
                             heading=hd, passage=f["passage"]))
    rng.shuffle(train_closed)
    rng.shuffle(train_open)

    out = {}
    out["train/csulb_qa"] = write_shards(os.path.join(D, "train", "csulb_qa"), train_closed)
    out["train/csulb_qa_open"] = write_shards(os.path.join(D, "train", "csulb_qa_open"), train_open)
    out["val/csulb_qa"] = write_shards(os.path.join(D, "val", "csulb_qa"), val_closed)
    out["val/csulb_qa_open"] = write_shards(os.path.join(D, "val", "csulb_qa_open"), val_open)

    # ---- what train.py scores: the general held-out plus ours
    mix = os.path.join(D, "heldout_mix")
    shutil.rmtree(mix, ignore_errors=True)
    os.makedirs(mix)
    for sub in sorted(os.listdir(a.general_val)):
        if os.path.isdir(os.path.join(a.general_val, sub)):
            hardlink_tree(os.path.join(a.general_val, sub), os.path.join(mix, sub))
    for sub in ("csulb_qa", "csulb_qa_open"):
        hardlink_tree(os.path.join(D, "val", sub), os.path.join(mix, sub))

    # ---- the test questions: a spread over kinds and documents
    os.makedirs(os.path.join(D, "eval"), exist_ok=True)
    with open(os.path.join(D, "eval", "questions_all.jsonl"), "w", encoding="utf-8") as f:
        for h in held:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    by_kind = {}
    for h in held:
        by_kind.setdefault(h["kind"], []).append(h)
    for v in by_kind.values():
        rng.shuffle(v)
    share = {"topic": 0.35, "definition": 0.2, "overview": 0.2, "number": 0.08,
             "date": 0.05, "where": 0.07, "supersedes": 0.03, "approval": 0.02}
    pick, used_docs = [], {}
    for k, frac in share.items():
        want = max(1, round(a.eval_n * frac))
        for h in by_kind.get(k, []):
            if want == 0:
                break
            if used_docs.get(h["doc"], 0) >= 2:
                continue
            used_docs[h["doc"]] = used_docs.get(h["doc"], 0) + 1
            pick.append(h)
            want -= 1
    pick = pick[:a.eval_n]
    with open(os.path.join(D, "eval", "questions.jsonl"), "w", encoding="utf-8") as f:
        for h in pick:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    mine = os.path.join(D, "eval", "my_questions.jsonl")
    if not os.path.exists(mine):
        with open(mine, "w", encoding="utf-8") as f:
            f.write(json.dumps(dict(q="What happens if I am caught cheating on an exam?",
                                    note="your own questions: one JSON per line, "
                                         "'q' is required, 'answer' and 'url' are optional"))
                    + "\n")

    kinds = {}
    for f in facts:
        kinds[f["kind"]] = kinds.get(f["kind"], 0) + 1
    print(f"{len(docs)} documents, {raw_chars / 1e6:.2f}M chars -> train/csulb_policies")
    print(f"{len(facts)} facts: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    for k, (n, c) in out.items():
        print(f"  {k:22s} {n:3d} files  {c / 1e6:6.2f}M chars")
    print(f"held-out wordings {len(held)}, eval sample {len(pick)} -> eval/questions.jsonl")
    print(f"heldout_mix: {', '.join(sorted(os.listdir(mix)))}")


if __name__ == "__main__":
    main()
