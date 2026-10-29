"""
Greedy answers for the frozen 400 questions.

Author's weights and hour 8 only. Closed-book is the question. Gold passage
is the stored gold_user string. The limit is 800 characters. Learning is off.

    .venv/bin/python tools/csulb/generate_answers.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

QUESTIONS = os.path.join(ROOT, "research", "eval", "questions_400.jsonl")
OUT = os.path.join(ROOT, "research", "eval", "local", "answers.jsonl")
CHECKPOINTS = ["weights", "weights_csulb"]
MAX_NEW = 800
MARKERS = ("</bot>", "</bo", "<user>", "<bot>")


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_rows():
    with open(QUESTIONS, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_keys():
    found = set()
    if not os.path.exists(OUT):
        return found
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            found.add((r["checkpoint"], r["mode"], r["id"]))
    return found


def cut_reply(text, n_yielded):
    at = None
    for marker in MARKERS:
        i = text.find(marker)
        if i >= 0 and (at is None or i < at):
            at = i
    answer = text if at is None else text[:at]
    stop = "end_marker" if n_yielded < MAX_NEW else "limit"
    return answer.strip(), stop


def generate(serve, user_text):
    model = serve.STATE["model"]
    prompt = serve.build_prompt(
        [{"role": "user", "content": user_text}],
        int(model.cfg.block * 0.9))
    parts = []
    for ev in serve.stream(prompt, MAX_NEW):
        if "t" in ev:
            parts.append(ev["t"])
    yielded = "".join(parts)
    answer, stop = cut_reply(yielded, len(parts))
    return {
        "answer": answer,
        "stop": stop,
        "yielded_chars": len(yielded),
        "new_chars": len(parts),
    }


def main():
    import serve
    rows = load_rows()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    have = done_keys()
    head = git_head()
    for weights in CHECKPOINTS:
        pending = []
        for row in rows:
            for mode in ("closed", "gold"):
                if (weights, mode, row["id"]) not in have:
                    pending.append((row, mode))
        if not pending:
            print(f"{weights}: already complete", flush=True)
            continue
        print(f"{weights}: {len(pending)} answers", flush=True)
        serve.load(os.path.join(ROOT, weights), device="cuda", learn=False)
        with open(OUT, "a", encoding="utf-8") as f:
            for n, (row, mode) in enumerate(pending, 1):
                user = row["q"] if mode == "closed" else row["gold_user"]
                gen = generate(serve, user)
                rec = {
                    "id": row["id"],
                    "checkpoint": weights,
                    "mode": mode,
                    "git": head,
                    "max_new": MAX_NEW,
                    "learn": False,
                    **gen,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if n % 10 == 0 or n == len(pending):
                    print(f"  {weights} {n}/{len(pending)} {mode} {row['id']} "
                          f"{gen['stop']} {gen['new_chars']} chars", flush=True)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
