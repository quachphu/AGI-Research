"""
Claude Opus 5.5 answers for the frozen questions.

No temperature is sent. Effort is low. One question per request. The 20
prompt-development questions run first, one at a time. The 400 are batched.
A second batch repeats the first 50 for the run-to-run ceiling.

The key is read from the environment or from an untracked .env file.
It is never written into a result file.

    .venv/bin/python tools/csulb/claude_answers.py dev
    .venv/bin/python tools/csulb/claude_answers.py batch
    .venv/bin/python tools/csulb/claude_answers.py ceiling
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL = os.path.join(ROOT, "research", "eval")
CLAUDE = os.path.join(EVAL, "claude")
MODEL = "claude-opus-5-5"
EFFORT = "low"
MAX_TOKENS = 16000
URL = "https://api.anthropic.com/v1/messages"
BATCH_URL = "https://api.anthropic.com/v1/messages/batches"

CLOSED = (
    "Answer this question from your own knowledge in 2 to 5 sentences, then stop. "
    "Do not use a heading or a bullet list. "
    "Mention a policy number or a source only if you know it.\n\n"
    "Question: {question}"
)
GOLD = (
    "Answer this question only from the passage below, in 2 to 5 sentences, then stop. "
    "Do not use a heading or a bullet list. "
    "Mention a policy number or a source only if the passage states it.\n\n"
    "{gold_user}"
)


def load_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    env = os.path.join(ROOT, ".env")
    if not key and os.path.exists(env):
        for line in open(env, encoding="utf-8"):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "ANTHROPIC_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set and .env does not contain it")
    return key


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_for(row, mode):
    if mode == "closed":
        return CLOSED.format(question=row["q"])
    if mode == "gold":
        return GOLD.format(gold_user=row["gold_user"])
    raise SystemExit(mode)


def params(prompt):
    # No temperature, top_p, or top_k. Opus 5.5 rejects them.
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "output_config": {"effort": EFFORT},
        "messages": [{"role": "user", "content": prompt}],
    }


def post(url, key, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("content-type", "application/json")
    req.add_header("x-api-key", key)
    req.add_header("anthropic-version", "2023-06-01")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r), r.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {e.code} from {url}: {body[:2000]}") from e


def get(url, key):
    req = urllib.request.Request(url, method="GET")
    req.add_header("x-api-key", key)
    req.add_header("anthropic-version", "2023-06-01")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read()
            return raw, r.status, r.headers.get_content_type()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise SystemExit(f"HTTP {e.code} from {url}: {body[:2000]}") from e


def text_of(message):
    parts = []
    for block in message.get("content") or []:
        if block.get("type") == "text":
            parts.append(block.get("text") or "")
    return "\n".join(parts).strip()


def record_from_message(custom_id, message, mode, run, qid):
    stop = message.get("stop_reason") or ""
    answer = text_of(message)
    valid = bool(answer) and stop not in ("max_tokens", "refusal")
    return {
        "id": qid,
        "custom_id": custom_id,
        "mode": mode,
        "run": run,
        "model": message.get("model") or MODEL,
        "date": time.strftime("%Y-%m-%d"),
        "request_id": message.get("id"),
        "stop_reason": stop,
        "usage": message.get("usage"),
        "answer": answer,
        "valid": valid,
    }


def custom_id(qid, mode, run):
    return f"{qid}__{mode}__{run}"


def parse_custom(cid):
    qid, mode, run = cid.split("__")
    return qid, mode, run


def run_dev(key):
    os.makedirs(CLAUDE, exist_ok=True)
    out = os.path.join(CLAUDE, "dev.jsonl")
    dev = rows(os.path.join(EVAL, "questions_prompt_dev.jsonl"))
    done = set()
    if os.path.exists(out):
        for r in rows(out):
            done.add((r["id"], r["mode"]))
    with open(out, "a", encoding="utf-8") as f:
        for row in dev:
            for mode in ("closed", "gold"):
                if (row["id"], mode) in done:
                    continue
                prompt = prompt_for(row, mode)
                message, _ = post(URL, key, params(prompt))
                if message.get("model") and not str(message["model"]).startswith(MODEL):
                    raise SystemExit(
                        f"model id came back as {message.get('model')}, not {MODEL}. "
                        "Stop and amend the protocol. Do not substitute.")
                rec = record_from_message(
                    custom_id(row["id"], mode, "dev"), message, mode, "dev", row["id"])
                rec["prompt"] = prompt
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{row['id']} {mode} stop={rec['stop_reason']} "
                      f"chars={len(rec['answer'])}", flush=True)
                print(rec["answer"][:400].replace("\n", " "), flush=True)
                print("---", flush=True)
    freeze_prompt()


def freeze_prompt():
    path = os.path.join(CLAUDE, "prompt.json")
    doc = {
        "model": MODEL,
        "effort": EFFORT,
        "max_tokens": MAX_TOKENS,
        "temperature": None,
        "closed": CLOSED,
        "gold": GOLD,
        "note": "Frozen after the 20-question development pass. "
                "Those 20 are not scored.",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {path}", flush=True)


def submit(key, requests, tag):
    os.makedirs(CLAUDE, exist_ok=True)
    raw_dir = os.path.join(CLAUDE, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    created, _ = post(BATCH_URL, key, {"requests": requests})
    bid = created["id"]
    with open(os.path.join(raw_dir, f"{tag}_batch.json"), "w") as f:
        json.dump(created, f, indent=1)
    print(f"batch {bid} status {created.get('processing_status')}", flush=True)
    while True:
        body, _, _ = get(f"{BATCH_URL}/{bid}", key)
        info = json.loads(body)
        status = info.get("processing_status")
        counts = info.get("request_counts")
        print(f"  {status} {counts}", flush=True)
        with open(os.path.join(raw_dir, f"{tag}_batch.json"), "w") as f:
            json.dump(info, f, indent=1)
        if status == "ended":
            return info
        time.sleep(30)


def download_results(key, info, tag):
    url = info["results_url"]
    req = urllib.request.Request(url, method="GET")
    # A signed results URL does not need the API key. Sending it is harmless
    # only when the host is Anthropic's. Try the signed URL as it stands.
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read()
    except urllib.error.HTTPError:
        raw, _, _ = get(url, key)
    path = os.path.join(CLAUDE, "raw", f"{tag}_results.jsonl")
    with open(path, "wb") as f:
        f.write(raw)
    return path


def append_results(path, run):
    dest = os.path.join(CLAUDE, "answers.jsonl")
    have = set()
    if os.path.exists(dest):
        for rec in rows(dest):
            have.add(rec["custom_id"])
    n = 0
    with open(dest, "a", encoding="utf-8") as f:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            cid = item["custom_id"]
            if cid in have:
                continue
            qid, mode, run_id = parse_custom(cid)
            result = item.get("result") or {}
            if result.get("type") != "succeeded":
                rec = {
                    "id": qid, "custom_id": cid, "mode": mode, "run": run_id,
                    "model": MODEL, "date": time.strftime("%Y-%m-%d"),
                    "request_id": None, "stop_reason": result.get("type") or "error",
                    "usage": None, "answer": "", "valid": False,
                    "error": result.get("error"),
                }
            else:
                message = result["message"]
                got = message.get("model") or ""
                if got and not str(got).startswith(MODEL):
                    raise SystemExit(
                        f"model id came back as {got}, not {MODEL}. "
                        "Stop and amend the protocol. Do not substitute.")
                rec = record_from_message(cid, message, mode, run_id, qid)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"appended {n} -> {dest}", flush=True)


def requests_for(questions, modes, run):
    out = []
    for row in questions:
        for mode in modes:
            cid = custom_id(row["id"], mode, run)
            out.append({"custom_id": cid, "params": params(prompt_for(row, mode))})
    return out


def run_batch(key, ceiling=False):
    questions = rows(os.path.join(EVAL, "questions_400.jsonl"))
    if ceiling:
        questions = questions[:50]
        tag, run = "ceiling", "ceiling"
    else:
        tag, run = "main", "main"
    reqs = requests_for(questions, ("closed", "gold"), run)
    info = submit(key, reqs, tag)
    path = download_results(key, info, tag)
    append_results(path, run)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("dev", "batch", "ceiling", "freeze"):
        raise SystemExit("usage: claude_answers.py dev|batch|ceiling|freeze")
    if sys.argv[1] == "freeze":
        freeze_prompt()
        return
    key = load_key()
    if sys.argv[1] == "dev":
        run_dev(key)
    elif sys.argv[1] == "batch":
        run_batch(key, ceiling=False)
    else:
        run_batch(key, ceiling=True)


if __name__ == "__main__":
    main()
