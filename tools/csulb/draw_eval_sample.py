"""
Draw the frozen evaluation sample.

    .venv/bin/python tools/csulb/draw_eval_sample.py

Clean questions are held-out wordings whose exact question string does not
appear in train/csulb_qa. The draw is 50 of each kind, at most two from one
document, seed 20261029. Twenty more clean questions, outside that 400, are
the prompt-development set. The rendered gold-passage prompt is stored, so a
later edit to qa_format.py does not change this evaluation.
"""
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from build_corpus import clean_title  # noqa: E402
from qa_format import open_book_prompt  # noqa: E402
from watch_8h import questions_in_train  # noqa: E402

SEED = 20261029
PER_KIND = 50
MAX_PER_DOC = 2
DEV_N = 20
KINDS = ["topic", "overview", "where", "date", "number",
         "definition", "approval", "supersedes"]
DATA = os.path.join(ROOT, "school_policies_data")
OUT_DIR = os.path.join(ROOT, "research", "eval")


def load_docs():
    docs = {}
    with open(os.path.join(DATA, "documents.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("kept"):
                d["ctitle"] = clean_title(d.get("title") or "")
                docs[d["id"]] = d
    return docs


def clean_rows():
    train_q = questions_in_train()
    rows = []
    with open(os.path.join(DATA, "eval", "questions_all.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    facts = [r["fact"] for r in rows]
    if len(facts) != len(set(facts)):
        raise SystemExit("fact ids are not unique")
    clean = [r for r in rows if r["q"].strip() not in train_q]
    return rows, clean


def row_out(r, docs):
    d = docs[r["doc"]]
    gold = open_book_prompt(r["q"], d, r.get("passage") or "", r.get("heading"))
    return {
        "id": f"f{int(r['fact']):05d}",
        "fact": r["fact"],
        "kind": r["kind"],
        "doc": r["doc"],
        "q": r["q"],
        "answer": r["answer"],
        "url": r.get("url") or d.get("url"),
        "policy_number": r.get("policy_number") or d.get("policy_number"),
        "date": d.get("date"),
        "title": d.get("ctitle") or r.get("title"),
        "heading": r.get("heading"),
        "passage": r.get("passage") or "",
        "gold_user": gold,
    }


def draw(clean, docs):
    rng = random.Random(SEED)
    picked = []
    used = set()
    for kind in KINDS:
        pool = sorted((r for r in clean if r["kind"] == kind),
                      key=lambda r: (r["fact"], r["q"]))
        rng.shuffle(pool)
        n_doc = {}
        got = []
        for r in pool:
            if len(got) >= PER_KIND:
                break
            if n_doc.get(r["doc"], 0) >= MAX_PER_DOC:
                continue
            got.append(r)
            n_doc[r["doc"]] = n_doc.get(r["doc"], 0) + 1
            used.add(r["fact"])
        if len(got) != PER_KIND:
            raise SystemExit(f"{kind}: drew {len(got)}, need {PER_KIND}")
        picked.extend(got)
    rest = sorted((r for r in clean if r["fact"] not in used),
                  key=lambda r: (r["fact"], r["q"]))
    rng.shuffle(rest)
    dev = rest[:DEV_N]
    if len(dev) != DEV_N:
        raise SystemExit(f"prompt-dev set has {len(dev)}")
    return [row_out(r, docs) for r in picked], [row_out(r, docs) for r in dev]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    docs = load_docs()
    all_rows, clean = clean_rows()
    sample, dev = draw(clean, docs)
    write_jsonl(os.path.join(OUT_DIR, "questions_400.jsonl"), sample)
    write_jsonl(os.path.join(OUT_DIR, "questions_prompt_dev.jsonl"), dev)
    print(f"all {len(all_rows)} clean {len(clean)} sample {len(sample)} dev {len(dev)}")
    from collections import Counter
    print("sample kinds", dict(Counter(r["kind"] for r in sample)))


if __name__ == "__main__":
    main()
