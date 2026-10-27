"""
Ask a weights folder the held-out CSULB questions and score the answers.

    .venv/bin/python tools/csulb/eval.py --weights weights --tag before
    .venv/bin/python tools/csulb/eval.py --weights weights_csulb --tag after
    .venv/bin/python tools/csulb/eval.py --report        # runs/csulb_eval.md

Each question is asked twice:
    closed-book   the question alone - what the model remembers
    open-book     the best BM25 passage put in front of the question, in the
                  format train/csulb_qa_open taught - what it can read

and scored against the answer it was built from:
    word F1       shared content words, answer vs reference (0..1)
    cites number  the policy statement number appears (when it has one)
    cites source  the policy's address appears
    answer form   the reply opens "According to ..." or ends with its source

Generation is greedy, so the same weights give the same answers every run.
Learning is off: asking never changes the weights.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from bm25 import BM25, words  # noqa: E402
from build_corpus import clean_heading, clean_title  # noqa: E402
from qa_format import open_book_prompt  # noqa: E402


def f1(pred, ref):
    p, r = words(pred), words(ref)
    if not p or not r:
        return 0.0
    from collections import Counter
    common = sum((Counter(p) & Counter(r)).values())
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def load_docs(data):
    docs = {}
    with open(os.path.join(data, "documents.jsonl"), encoding="utf-8") as f:
        for l in f:
            d = json.loads(l)
            if d.get("kept"):
                d["ctitle"] = clean_title(d["title"])
                docs[d["id"]] = d
    return docs


def open_prompt(question, index, docs):
    hit = index.search(question, k=1)
    if not hit:
        return question, None
    s = hit[0]
    return open_book_prompt(question, docs[s["doc"]], s["text"], clean_heading(s.get("heading"))), s


def ask(serve, text, max_new):
    model = serve.STATE["model"]
    prompt = serve.build_prompt([{"role": "user", "content": text}], int(model.cfg.block * 0.9))
    out = []
    for ev in serve.stream(prompt, max_new):
        if "t" in ev:
            out.append(ev["t"])
    return "".join(out).strip()


def run(a):
    import serve
    serve.load(a.weights, device=a.device, learn=False)
    docs = load_docs(a.data)
    index = BM25.load(os.path.join(a.data, "sections.jsonl"))
    qs = []
    with open(os.path.join(a.data, "eval", "questions.jsonl"), encoding="utf-8") as f:
        qs += [json.loads(l) for l in f if l.strip()]
    if a.limit:
        qs = qs[:a.limit]
    mine = os.path.join(a.data, "eval", "my_questions.jsonl")
    extra = []
    if os.path.exists(mine):
        with open(mine, encoding="utf-8") as f:
            extra = [json.loads(l) for l in f if l.strip()]

    rows, t0 = [], time.time()
    for i, h in enumerate(qs + extra):
        row = dict(q=h["q"], kind=h.get("kind", "yours"), ref=h.get("answer"),
                   url=h.get("url"), number=h.get("policy_number"), doc=h.get("doc"))
        row["closed"] = ask(serve, h["q"], a.max_new)
        prompt, hit = open_prompt(h["q"], index, docs)
        row["open"] = ask(serve, prompt, a.max_new)
        row["retrieved"] = dict(doc=hit["doc"], title=hit["title"], heading=hit.get("heading"),
                                url=hit["url"]) if hit else None
        if row["ref"]:
            for mode in ("closed", "open"):
                ans = row[mode]
                row[f"{mode}_f1"] = round(f1(ans, row["ref"]), 4)
                row[f"{mode}_url"] = bool(row["url"]) and row["url"] in ans
                row[f"{mode}_num"] = (row["number"][3:] in ans) if row["number"] else None
                row[f"{mode}_form"] = ans.startswith("According to") or "Source: http" in ans
            row["retrieved_doc_ok"] = bool(hit) and hit["doc"] == row["doc"]
        rows.append(row)
        el = time.time() - t0
        print(f"[{i + 1}/{len(qs) + len(extra)}] {el / 60:.1f} min  "
              f"closed F1 {row.get('closed_f1', '-')}  open F1 {row.get('open_f1', '-')}  "
              f"{h['q'][:70]}", flush=True)

    scored = [r for r in rows if r["ref"]]
    summ = dict(weights=a.weights, tag=a.tag, n=len(scored), minutes=round((time.time() - t0) / 60, 1),
                retrieval_doc_top1=mean(r["retrieved_doc_ok"] for r in scored))
    for mode in ("closed", "open"):
        summ[mode] = dict(
            f1=mean(r[f"{mode}_f1"] for r in scored),
            cites_source=mean(r[f"{mode}_url"] for r in scored),
            cites_number=mean(r[f"{mode}_num"] for r in scored if r[f"{mode}_num"] is not None),
            answer_form=mean(r[f"{mode}_form"] for r in scored))
    out = os.path.join(ROOT, "runs", f"csulb_eval_{a.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(summary=summ, rows=rows), f, ensure_ascii=False, indent=1)
    print(json.dumps(summ, indent=1))
    print(f"-> {out}")


def mean(xs):
    xs = [float(x) for x in xs]
    return round(sum(xs) / len(xs), 4) if xs else None


# ---------------------------------------------------------------- report
def report(a):
    res = {}
    for tag in ("before", "after"):
        p = os.path.join(ROOT, "runs", f"csulb_eval_{tag}.json")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                res[tag] = json.load(f)
    if not res:
        sys.exit("no runs/csulb_eval_*.json yet")
    tags = list(res)
    L = ["# CSULB policy chatbot: evaluation", ""]
    for t in tags:
        s = res[t]["summary"]
        L.append(f"- **{t}**: `{s['weights']}`, {s['n']} held-out questions, {s['minutes']} min")
    L += ["", "Held-out questions are wordings never trained on. Scores are 0-100%. "
          "*Word F1* is how much of the reference answer's wording the reply uses.", ""]
    L.append("| measure | " + " | ".join(tags) + " |")
    L.append("|---|" + "---|" * len(tags))
    for mode, label in (("closed", "closed-book (memory)"), ("open", "open-book (passage given)")):
        for k, name in (("f1", "word F1"), ("cites_number", "cites the policy number"),
                        ("cites_source", "cites the source URL"),
                        ("answer_form", "answers in the trained form")):
            vals = [res[t]["summary"][mode][k] for t in tags]
            L.append(f"| {label}: {name} | " + " | ".join(
                "-" if v is None else f"{v * 100:.1f}%" for v in vals) + " |")
    L += [f"| retrieval: right policy found first (same for both) | " +
          " | ".join(f"{res[t]['summary']['retrieval_doc_top1'] * 100:.1f}%" for t in tags) + " |", ""]

    rows = {t: res[t]["rows"] for t in tags}
    L += ["## Examples", ""]
    n = min(len(r) for r in rows.values())
    pick = list(range(0, n, max(1, n // 15)))[:15]
    for i in pick:
        r0 = rows[tags[0]][i]
        L += [f"### {i + 1}. {r0['q']}", ""]
        if r0.get("ref"):
            L += [f"*Reference:* {short(r0['ref'], 400)}", ""]
        if r0.get("retrieved"):
            rt = r0["retrieved"]
            L += [f"*Passage found:* {rt['title']} - {rt.get('heading') or ''} ({rt['url']})", ""]
        for t in tags:
            r = rows[t][i]
            L += [f"- **{t}, closed-book**: {short(r['closed'], 400)}",
                  f"- **{t}, open-book**: {short(r['open'], 400)}"]
        L.append("")
    yours = [i for i, r in enumerate(rows[tags[0]]) if r["kind"] == "yours"]
    if yours:
        L += ["## Your questions (eval/my_questions.jsonl)", ""]
        for i in yours:
            L += [f"### {rows[tags[0]][i]['q']}", ""]
            for t in tags:
                r = rows[t][i]
                L += [f"- **{t}, closed-book**: {short(r['closed'], 400)}",
                      f"- **{t}, open-book**: {short(r['open'], 400)}"]
            L.append("")
    out = os.path.join(ROOT, "runs", "csulb_eval.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L[:30]))
    print(f"-> {out}")


def short(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + " ..."


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--weights", default="weights_csulb")
    ap.add_argument("--tag", default="after")
    ap.add_argument("--data", default=os.path.join(ROOT, "school_policies_data"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-new", type=int, default=360)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report(a)
    os.chdir(ROOT)
    run(a)


if __name__ == "__main__":
    main()
