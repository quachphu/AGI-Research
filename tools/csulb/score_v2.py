"""
Rescore with closed-book Claude v2.

Gold-passage Claude answers stay the version-1 records. Closed-book answers
come from answers_v2.jsonl. The output is scores_v2.json. scores.json and
answers.jsonl are not written.

    .venv/bin/python tools/csulb/score_v2.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import score_similarity as s  # noqa: E402

V1 = os.path.join(ROOT, "research", "eval", "claude", "answers.jsonl")
V2 = os.path.join(ROOT, "research", "eval", "claude", "answers_v2.jsonl")
MERGED = os.path.join(ROOT, "research", "eval", "claude", "answers_v2_scored.jsonl")
OUT = os.path.join(ROOT, "research", "eval", "scores_v2.json")


def main():
    v2 = s.load_jsonl(V2)
    closed = [r for r in v2 if r.get("mode") == "closed" and r.get("prompt_version") == "v2"]
    if not closed:
        raise SystemExit(f"no v2 closed answers in {V2}")
    gold = [r for r in s.load_jsonl(V1)
            if r.get("mode") == "gold" and r.get("run") in ("main", "ceiling")]
    with open(MERGED, "w", encoding="utf-8") as f:
        for r in gold + closed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s.CLAUDE = MERGED
    s.OUT = OUT
    s.main()


if __name__ == "__main__":
    main()
