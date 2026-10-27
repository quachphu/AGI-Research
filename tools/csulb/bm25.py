"""
Word-match search over the policy sections: Okapi BM25, pure Python.

    .venv/bin/python tools/csulb/bm25.py "late add deadline"        # top 3
    .venv/bin/python tools/csulb/bm25.py --check                     # hit rate

This is what finds the official passage shown beside every answer. It is not
a model and learns nothing; it counts shared words, weighted by how rare each
word is across all sections. The section's title and heading are counted as
part of its text, so a question naming a policy finds that policy.
"""
import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

STOP = set("""a an and are as at be but by can could do does for from had has have how
i if in into is it its me my of on or our shall should so that the their them then
there these they this to under up us was we were what when where which who whom why
will with would you your about any all also may must not no csulb csu university
long beach california state policy policies tell explain please say says""".split())


def words(text):
    out = []
    for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()):
        if w in STOP or len(w) < 2:
            continue
        # a crude stem: enough that "withdrawals" finds "withdrawal"
        for suf in ("ings", "ing", "ies", "es", "s", "ed"):
            if len(w) > len(suf) + 3 and w.endswith(suf):
                w = w[: -len(suf)] + ("y" if suf == "ies" else "")
                break
        out.append(w)
    return out


class BM25:
    def __init__(self, sections, k1=1.2, b=0.6, title_boost=2, heading_boost=4):
        self.sections = sections
        self.k1, self.b = k1, b
        self.tf, self.len = [], []
        df = Counter()
        for s in sections:
            title = f"{s.get('title') or ''} {s.get('policy_number') or ''}"
            toks = (words(s["text"]) + words(title) * title_boost
                    + words(s.get("heading") or "") * heading_boost)
            c = Counter(toks)
            self.tf.append(c)
            self.len.append(len(toks))
            df.update(c.keys())
        n = len(sections)
        self.avg = sum(self.len) / max(1, n)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for w in c:
                self.post[w].append(i)

    @classmethod
    def load(cls, path="school_policies_data/sections.jsonl"):
        with open(path, encoding="utf-8") as f:
            return cls([json.loads(l) for l in f if l.strip()])

    def search(self, query, k=3):
        q = words(query)
        # a policy number in the question ("PS 21-01", "21-01") is a near-certain match
        nums = set(re.findall(r"\b(\d{2}-\d{2})\b", query))
        score = defaultdict(float)
        for w in set(q):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i in self.post[w]:
                f = self.tf[i][w]
                score[i] += idf * f * (self.k1 + 1) / (
                    f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        if nums:
            for i, s in enumerate(self.sections):
                if (s.get("policy_number") or "")[3:] in nums:
                    score[i] += 10.0
        best = sorted(score.items(), key=lambda kv: -kv[1])[:k]
        return [dict(self.sections[i], score=round(sc, 3)) for i, sc in best]


def check(index, questions):
    """How often the section a held-out question was built from comes back."""
    n = top1 = top3 = doc1 = 0
    for h in questions:
        if not h.get("section"):
            continue
        hits = index.search(h["q"], k=3)
        n += 1
        ids = [x["id"] for x in hits]
        top1 += bool(ids) and ids[0] == h["section"]
        top3 += h["section"] in ids
        doc1 += bool(hits) and hits[0]["doc"] == h["doc"]
    return dict(n=n, top1=top1 / max(1, n), top3=top3 / max(1, n), doc_top1=doc1 / max(1, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("query", nargs="?")
    ap.add_argument("--data", default="school_policies_data")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--check", action="store_true",
                    help="score retrieval on eval/questions_all.jsonl")
    a = ap.parse_args()
    idx = BM25.load(os.path.join(a.data, "sections.jsonl"))
    if a.check:
        with open(os.path.join(a.data, "eval", "questions_all.jsonl"), encoding="utf-8") as f:
            qs = [json.loads(l) for l in f]
        r = check(idx, qs)
        print(f"{r['n']} held-out questions: right section first {r['top1']:.1%}, "
              f"in top 3 {r['top3']:.1%}, right document first {r['doc_top1']:.1%}")
        by = defaultdict(list)
        for h in qs:
            by[h["kind"]].append(h)
        for kind, hs in sorted(by.items()):
            r = check(idx, hs)
            print(f"  {kind:11s} {r['n']:5d}  section top1 {r['top1']:.1%}  top3 {r['top3']:.1%}"
                  f"  document top1 {r['doc_top1']:.1%}")
        return
    for h in idx.search(a.query or "", k=a.k):
        print(f"[{h['score']}] {h['title']} {h.get('policy_number') or ''} - {h.get('heading') or ''}")
        print(f"   {h['url']}")
        print(f"   {' '.join(h['text'].split())[:300]}\n")


if __name__ == "__main__":
    main()
