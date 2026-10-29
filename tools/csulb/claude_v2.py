"""
Closed-book Claude answers with the campus sentence added.

Does not write research/eval/claude/prompt.json or answers.jsonl.
Version 1 stays where it is. This writes prompt_v2.json, answers_v2.jsonl,
and raw/main_v2_* plus raw/ceiling_v2_*.

    .venv/bin/python tools/csulb/claude_v2.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import claude_answers as c  # noqa: E402

SENTENCE = (
    "These questions are about policies of California State University, "
    "Long Beach (CSULB). "
)
CLOSED_V2 = SENTENCE + c.CLOSED
DEST = os.path.join(c.CLAUDE, "answers_v2.jsonl")
PROMPT_PATH = os.path.join(c.CLAUDE, "prompt_v2.json")


def write_prompt():
    doc = {
        "model": c.MODEL,
        "effort": c.EFFORT,
        "max_tokens": c.MAX_TOKENS,
        "temperature": None,
        "closed": CLOSED_V2,
        "gold": None,
        "note": "Version 2, 29 October 2026. The frozen closed-book prompt "
                "plus one sentence. Gold is unchanged and was not re-run. "
                "Version 1 remains prompt.json and answers.jsonl.",
    }
    with open(PROMPT_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")


def have_ids():
    if not os.path.exists(DEST):
        return set()
    return {r["custom_id"] for r in c.rows(DEST)}


def append_v2(path):
    got = have_ids()
    n = 0
    invalid = 0
    with open(DEST, "a", encoding="utf-8") as f:
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            item = json.loads(line)
            cid = item["custom_id"]
            if cid in got:
                continue
            qid, mode, run_id = c.parse_custom(cid)
            result = item.get("result") or {}
            if result.get("type") != "succeeded":
                rec = {
                    "id": qid, "custom_id": cid, "mode": mode, "run": run_id,
                    "prompt_version": "v2", "model": c.MODEL,
                    "date": time.strftime("%Y-%m-%d"), "request_id": None,
                    "stop_reason": result.get("type") or "error",
                    "usage": None, "answer": "", "valid": False,
                    "error": result.get("error"),
                }
                invalid += 1
            else:
                message = result["message"]
                got_model = message.get("model") or ""
                if got_model and not str(got_model).startswith(c.MODEL):
                    raise SystemExit(
                        f"model id came back as {got_model}, not {c.MODEL}. "
                        "Stop and amend the protocol. Do not substitute.")
                rec = c.record_from_message(cid, message, mode, run_id, qid)
                reason = (rec.get("stop_reason") or "").lower()
                if reason in ("max_tokens", "refusal", "model_refusal") or not rec["answer"]:
                    rec["valid"] = False
                    invalid += 1
                rec["prompt_version"] = "v2"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            got.add(cid)
            n += 1
    print(f"appended {n} invalid {invalid} -> {DEST}", flush=True)


def finish_existing(key, tag):
    """Poll a batch that was already submitted. Do not open a second one."""
    path = os.path.join(c.CLAUDE, "raw", f"{tag}_batch.json")
    if not os.path.exists(path):
        return None
    info = json.load(open(path, encoding="utf-8"))
    bid = info.get("id")
    if not bid:
        return None
    while info.get("processing_status") != "ended":
        body, _, _ = c.get(f"{c.BATCH_URL}/{bid}", key)
        info = json.loads(body)
        print(f"  {info.get('processing_status')} {info.get('request_counts')}",
              flush=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=1)
        if info.get("processing_status") != "ended":
            time.sleep(30)
    results = os.path.join(c.CLAUDE, "raw", f"{tag}_results.jsonl")
    if not os.path.exists(results):
        results = c.download_results(key, info, tag)
    append_v2(results)
    return info


def run_one(key, ceiling):
    questions = c.rows(os.path.join(c.EVAL, "questions_400.jsonl"))
    if ceiling:
        questions = questions[:50]
        tag, run = "ceiling_v2", "ceiling"
    else:
        tag, run = "main_v2", "main"
    need = {c.custom_id(row["id"], "closed", run) for row in questions}
    if need <= have_ids():
        print(f"{tag} already in {DEST}", flush=True)
        return
    # The imported prompt builder reads claude_answers.CLOSED.
    c.CLOSED = CLOSED_V2
    if os.path.exists(os.path.join(c.CLAUDE, "raw", f"{tag}_batch.json")):
        print(f"resuming {tag}", flush=True)
        finish_existing(key, tag)
        return
    reqs = c.requests_for(questions, ("closed",), run)
    info = c.submit(key, reqs, tag)
    path = c.download_results(key, info, tag)
    append_v2(path)


def main():
    os.makedirs(c.CLAUDE, exist_ok=True)
    write_prompt()
    key = c.load_key()
    run_one(key, ceiling=False)
    run_one(key, ceiling=True)


if __name__ == "__main__":
    main()
