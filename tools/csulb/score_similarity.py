"""
Score the frozen 400 questions.

Cosine to the official reference is the headline. Exact match and word F1 use
the raw answer, after the end-of-reply cut and before stripping. Embeddings
see the stripped text, cut to that question's cleaned reference length.

    .venv/bin/python tools/csulb/score_similarity.py
    .venv/bin/python tools/csulb/score_similarity.py --self-test
"""
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from eval import f1  # noqa: E402

QUESTIONS = os.path.join(ROOT, "research", "eval", "questions_400.jsonl")
LOCAL = os.path.join(ROOT, "research", "eval", "local", "answers.jsonl")
CLAUDE = os.path.join(ROOT, "research", "eval", "claude", "answers.jsonl")
OUT = os.path.join(ROOT, "research", "eval", "scores.json")
QWEN = "Qwen/Qwen3-Embedding-0.6B"
MPNET = "sentence-transformers/all-mpnet-base-v2"
SEED = 20261029
N_SHUFFLE = 10
N_BOOT = 10000

TAGS = re.compile(r"</?user>|</?bot>")
SOURCE_LINE = re.compile(r"^Source:.*$", re.M)


def strip_for_embed(text):
    """Drop the trained lead-in, source lines, and chat tags."""
    text = text or ""
    text = TAGS.sub(" ", text)
    text = SOURCE_LINE.sub(" ", text)
    text = re.sub(r"Source:\s*\S+", " ", text)
    stripped = text.lstrip()
    if stripped.startswith("According to "):
        colon = stripped.find(": ")
        text = stripped[colon + 2:] if colon >= 0 else stripped
    return " ".join(text.split())


def number_needle(policy_number):
    if not policy_number:
        return None
    if policy_number.startswith("PS "):
        return policy_number[3:]
    return policy_number


def contains(answer, needle):
    if not needle:
        return None
    return needle in (answer or "")


def exact_fields(row, answer):
    num = contains(answer, number_needle(row.get("policy_number")))
    date = contains(answer, row.get("date")) if row.get("kind") == "date" else None
    url = contains(answer, row.get("url"))
    return {"number": num, "date": date, "url": url}


def prepare(row, answer):
    """Raw metrics, then the embedding string cut to the cleaned reference."""
    raw = answer or ""
    ref_clean = strip_for_embed(row["answer"])
    n = len(ref_clean)
    return {
        "raw": raw,
        "f1": f1(raw, row["answer"]),
        "exact": exact_fields(row, raw),
        "embed": strip_for_embed(raw)[:n],
        "ref_embed": ref_clean,
        "embed_chars": n,
    }


def top_quarter(ids, scores):
    """The highest ceil(n/4). A tie at that boundary stays in the panel."""
    ranked = sorted(ids, key=lambda i: (-scores[i], i))
    k = max(1, (len(ranked) + 3) // 4)
    cut = scores[ranked[k - 1]]
    return [i for i in ranked if scores[i] >= cut - 1e-12]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def bootstrap_diff(a, b, seed=SEED):
    """95% interval for the mean of a minus the mean of b, paired."""
    rng = random.Random(seed)
    n = len(a)
    if n == 0:
        return None
    diffs = []
    idx = list(range(n))
    for _ in range(N_BOOT):
        draw = [idx[rng.randrange(n)] for _ in range(n)]
        diffs.append(mean(a[i] - b[i] for i in draw))
    diffs.sort()
    lo = diffs[int(0.025 * (N_BOOT - 1))]
    hi = diffs[int(0.975 * (N_BOOT - 1))]
    return {"mean": mean(x - y for x, y in zip(a, b)), "lo": lo, "hi": hi, "n": n}


def shuffles(ids):
    rng = random.Random(SEED)
    out = []
    for _ in range(N_SHUFFLE):
        perm = ids[:]
        for _try in range(20):
            rng.shuffle(perm)
            if all(perm[i] != ids[i] for i in range(len(ids))):
                break
            for i, q in enumerate(ids):
                if perm[i] == q:
                    j = (i + 1) % len(ids)
                    perm[i], perm[j] = perm[j], perm[i]
        out.append(perm)
    return out


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def valid_claude(rec):
    if rec.get("valid") is False:
        return False
    reason = (rec.get("stop_reason") or "").lower()
    if reason in ("max_tokens", "refusal", "model_refusal"):
        return False
    if not (rec.get("answer") or "").strip():
        return False
    return True


def encode_all(text_of):
    """L2-normalized vectors, no instruction on either embedding."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    keys = list(text_of)
    texts = [text_of[k] for k in keys]
    device = "cuda" if _cuda() else "cpu"
    out = {}
    for name in (QWEN, MPNET):
        model = SentenceTransformer(name, device=device)
        kwargs = dict(batch_size=8, normalize_embeddings=True,
                      show_progress_bar=True, convert_to_numpy=True)
        # Qwen3's default query prompt would make the two sides asymmetric.
        # Both texts are answers, so neither gets an instruction.
        try:
            vecs = model.encode(texts, prompt=None, **kwargs)
        except TypeError:
            vecs = model.encode(texts, **kwargs)
        vecs = np.asarray(vecs, dtype="float64")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.clip(norms, 1e-12, None)
        out[name] = {k: vecs[i] for i, k in enumerate(keys)}
        del model
    return out


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def dot(a, b):
    return float((a * b).sum())


def rate(flags):
    used = [f for f in flags if f is not None]
    if not used:
        return None
    return mean(1.0 if f else 0.0 for f in used)


def self_test():
    row = {
        "answer": "According to Policy Statement 01-01 (Example), the rule: "
                  "Students may enroll.\nSource: https://example.edu/a",
        "policy_number": "PS 01-01",
        "date": "March 4, 1974",
        "kind": "date",
        "url": "https://example.edu/a",
    }
    raw = "According to Policy Statement 01-01, the rule: Students may enroll."
    got = prepare(row, raw)
    assert got["embed"] == "Students may enroll."
    assert got["ref_embed"].startswith("Students may enroll.")
    assert got["exact"]["number"] is True
    assert got["exact"]["url"] is False
    assert got["exact"]["date"] is False
    missing = exact_fields({"policy_number": None, "kind": "topic", "url": None}, "x")
    assert missing == {"number": None, "date": None, "url": None}
    assert top_quarter(["b", "a", "c", "d"], {"a": 0.1, "b": 0.9, "c": 0.9, "d": 0.2}) == ["b", "c"]
    print("self-test ok")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return
    questions = load_jsonl(QUESTIONS)
    by_id = {r["id"]: r for r in questions}
    order = [r["id"] for r in questions]
    local = defaultdict(dict)
    for r in load_jsonl(LOCAL):
        local[(r["checkpoint"], r["mode"])][r["id"]] = r
    claude = defaultdict(dict)
    if os.path.exists(CLAUDE):
        for r in load_jsonl(CLAUDE):
            claude[(r.get("run") or "main", r["mode"])][r["id"]] = r

    systems = {
        "author_closed": ("weights", "closed", "main"),
        "hour8_closed": ("weights_csulb", "closed", "main"),
        "claude_closed": (None, "closed", "main"),
        "author_gold": ("weights", "gold", "main"),
        "hour8_gold": ("weights_csulb", "gold", "main"),
        "claude_gold": (None, "gold", "main"),
        "claude_closed_ceiling": (None, "closed", "ceiling"),
        "claude_gold_ceiling": (None, "gold", "ceiling"),
    }

    prepared = {}
    invalid = Counter()
    for name, (ckpt, mode, run) in systems.items():
        prepared[name] = {}
        for qid, row in by_id.items():
            if ckpt:
                rec = local[(ckpt, mode)].get(qid)
                ok = bool(rec and (rec.get("answer") or "").strip())
                answer = rec.get("answer") if rec else ""
            else:
                rec = claude[(run, mode)].get(qid)
                ok = bool(rec and valid_claude(rec))
                answer = rec.get("answer") if rec else ""
                if rec and not ok:
                    invalid[rec.get("stop_reason") or "empty"] += 1
            if ok:
                prepared[name][qid] = prepare(row, answer)

    texts = {}
    for name, pack in prepared.items():
        for qid, p in pack.items():
            texts[(name, qid)] = p["embed"]
            texts[("ref", qid)] = p["ref_embed"]
    vectors = encode_all(texts)

    per = []
    summary = {"embeddings": {"primary": QWEN, "secondary": MPNET},
               "excluded_claude": dict(invalid),
               "modes": {}}
    for name in systems:
        pack = prepared[name]
        summary["modes"][name] = {
            "n_valid": len(pack),
            "f1": mean(p["f1"] for p in pack.values()),
            "number": rate(p["exact"]["number"] for p in pack.values()),
            "date": rate(p["exact"]["date"] for p in pack.values()),
            "url": rate(p["exact"]["url"] for p in pack.values()),
        }

    def cos(model, a_name, a_id, b_name, b_id):
        return dot(vectors[model][(a_name, a_id)], vectors[model][(b_name, b_id)])

    for model in (QWEN, MPNET):
        for name, pack in prepared.items():
            if not pack:
                continue
            vals = {qid: cos(model, name, qid, "ref", qid) for qid in pack}
            summary["modes"][name].setdefault("cosine", {})[model] = mean(vals.values())
            high = top_quarter(list(vals), vals)
            panel = []
            for qid in high:
                ex = pack[qid]["exact"]
                if ex["number"] is False or ex["date"] is False:
                    panel.append({"id": qid, "cosine": vals[qid],
                                  "number": ex["number"], "date": ex["date"]})
            summary["modes"][name].setdefault("error_panel", {})[model] = {
                "high_n": len(high), "wrong_n": len(panel), "rows": panel}
            ids = list(pack)
            chance = []
            for perm in shuffles(ids):
                chance.append(mean(cos(model, name, q, "ref", p)
                                   for q, p in zip(ids, perm)))
            summary["modes"][name].setdefault("chance", {})[model] = mean(chance)

    # Ceiling and agreement use questions where both answers are valid.
    pairs = {
        "ceiling_closed": ("claude_closed", "claude_closed_ceiling"),
        "ceiling_gold": ("claude_gold", "claude_gold_ceiling"),
        "agree_author_closed": ("author_closed", "claude_closed"),
        "agree_hour8_closed": ("hour8_closed", "claude_closed"),
        "agree_author_gold": ("author_gold", "claude_gold"),
        "agree_hour8_gold": ("hour8_gold", "claude_gold"),
    }
    summary["paired"] = {}
    for label, (left, right) in pairs.items():
        both = [q for q in order if q in prepared[left] and q in prepared[right]]
        summary["paired"][label] = {"n": len(both)}
        for model in (QWEN, MPNET):
            if not both:
                continue
            summary["paired"][label][model] = mean(
                cos(model, left, q, right, q) for q in both)

    def paired_cos(model, left, right):
        both = [q for q in order if q in prepared[left] and q in prepared[right]]
        a = [cos(model, left, q, "ref", q) for q in both]
        b = [cos(model, right, q, "ref", q) for q in both]
        return both, a, b

    summary["bootstrap"] = {}
    contrasts = [
        ("hour8_minus_author_closed", "hour8_closed", "author_closed"),
        ("hour8_minus_claude_closed", "hour8_closed", "claude_closed"),
        ("author_minus_claude_closed", "author_closed", "claude_closed"),
        ("hour8_minus_author_gold", "hour8_gold", "author_gold"),
        ("hour8_minus_claude_gold", "hour8_gold", "claude_gold"),
        ("author_minus_claude_gold", "author_gold", "claude_gold"),
    ]
    for label, left, right in contrasts:
        summary["bootstrap"][label] = {}
        for model in (QWEN, MPNET):
            both, a, b = paired_cos(model, left, right)
            summary["bootstrap"][label][model] = bootstrap_diff(a, b)

    def order_of(mode_names, model):
        scored = [(summary["modes"][n]["cosine"][model], n)
                  for n in mode_names if "cosine" in summary["modes"].get(n, {})]
        return [n for _, n in sorted(scored, reverse=True)]

    closed_names = ["author_closed", "hour8_closed", "claude_closed"]
    gold_names = ["author_gold", "hour8_gold", "claude_gold"]
    summary["ranking"] = {
        "closed": {"qwen": order_of(closed_names, QWEN),
                   "mpnet": order_of(closed_names, MPNET)},
        "gold": {"qwen": order_of(gold_names, QWEN),
                 "mpnet": order_of(gold_names, MPNET)},
    }
    summary["ranking"]["closed"]["agree"] = (
        summary["ranking"]["closed"]["qwen"] == summary["ranking"]["closed"]["mpnet"])
    summary["ranking"]["gold"]["agree"] = (
        summary["ranking"]["gold"]["qwen"] == summary["ranking"]["gold"]["mpnet"])

    by_kind = defaultdict(lambda: defaultdict(list))
    cos_kind = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name in ("author_closed", "hour8_closed", "claude_closed",
                 "author_gold", "hour8_gold", "claude_gold"):
        for qid, p in prepared[name].items():
            kind = by_id[qid]["kind"]
            by_kind[name][kind].append(p["f1"])
            for model in (QWEN, MPNET):
                cos_kind[name][model][kind].append(cos(model, name, qid, "ref", qid))
    summary["f1_by_kind"] = {
        name: {k: mean(v) for k, v in kinds.items()} for name, kinds in by_kind.items()}
    summary["cosine_by_kind"] = {
        name: {model: {k: mean(v) for k, v in kinds.items()}
               for model, kinds in models.items()}
        for name, models in cos_kind.items()}

    # Per question, for the demo and the error panel's context.
    rows_out = []
    for qid in order:
        rec = {"id": qid, "kind": by_id[qid]["kind"]}
        for model in (QWEN, MPNET):
            short = "qwen" if model == QWEN else "mpnet"
            for name in ("hour8_closed", "author_closed", "claude_closed",
                         "hour8_gold", "author_gold", "claude_gold"):
                if qid in prepared[name]:
                    rec[f"{name}__{short}"] = cos(model, name, qid, "ref", qid)
        if qid in prepared["hour8_closed"]:
            rec["hour8_closed_f1"] = prepared["hour8_closed"][qid]["f1"]
            rec["hour8_closed_exact"] = prepared["hour8_closed"][qid]["exact"]
        rows_out.append(rec)
    summary["per_question"] = rows_out

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(summary, f)
    print(f"wrote {OUT}")
    for name in closed_names + gold_names:
        m = summary["modes"][name]
        cos_q = m.get("cosine", {}).get(QWEN)
        f1s = m["f1"]
        f1_txt = "-" if f1s is None else f"{f1s:.3f}"
        print(f"  {name}: n={m['n_valid']} f1={f1_txt} "
              f"qwen={None if cos_q is None else round(cos_q, 3)}")


if __name__ == "__main__":
    main()
