"""
Per-piece enwik9 loss for the author's weights and hour 8.

Uses the same 2,000 spread pieces as tools/measure_enwik9.py. Writes only
under research/eval/diagnostics/. Does not call the benchmark recorder, so
runs/enwik9_measured.json and the history file are left alone.

    .venv/bin/python tools/csulb/enwik9_pieces.py
"""
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import measure_heldout as MH  # noqa: E402

SLICE = os.path.join(ROOT, "data", "enwik9")
PIECE = 2048
N = 2000
OUT_DIR = os.path.join(ROOT, "research", "eval", "diagnostics")
LN2 = math.log(2)


def kind_of(text):
    """One label. A table wins, then a template, then any wiki markup."""
    table = ("{|" in text) or ("|}" in text) or (
        text.count("\n|") + text.count("\n!")) >= 4
    if table:
        return "table"
    if text.count("{{") >= 2:
        return "template"
    if (text.count("[[") >= 1 or text.count("==") >= 2
            or "<page" in text or "<text" in text or "<revision" in text):
        return "markup"
    return "prose"


def decode(arr):
    raw = bytes(int(x) for x in arr[:PIECE] if int(x) < 256)
    return raw.decode("utf-8", "replace")


def losses_for(weights):
    import torch
    import train as T
    from minagi.config import load, get
    from minagi.precision import set_compute_dtype
    from minagi.stream import FolderEvaluator

    if not torch.cuda.is_available():
        raise SystemExit("no GPU")
    pid = MH.training_run(weights)
    if pid:
        raise SystemExit(f"training is running (pid {pid})")
    set_compute_dtype(str(get(load(), "training.precision", "bf16")))
    device = torch.device("cuda")
    model, cfg, pool, man = T.build_paged(
        weights, device,
        ram_capacity=int(get(load(), "pool.ram_cache", 96)),
        read_only=True)
    model.eval()
    ev = FolderEvaluator(model, SLICE, PIECE, PIECE, device, whole=False)
    name = next(iter(ev.groups))
    texts = ev._texts(name, N)
    if len(texts) != N:
        raise SystemExit(f"expected {N} pieces, got {len(texts)}")
    out = []
    for i, data in enumerate(texts):
        with torch.no_grad():
            got = ev.score_block(data, name)
        if len(got) != 1:
            raise SystemExit(f"piece {i} returned {len(got)} chunks")
        text = decode(data)
        out.append({"i": i, "kind": kind_of(text), "nats": got[0]})
        if (i + 1) % 200 == 0:
            print(f"  {os.path.basename(weights)} {i + 1}/{N}", flush=True)
    del model, pool
    torch.cuda.empty_cache()
    return man.get("step"), out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    author_step, author = losses_for(os.path.join(ROOT, "weights"))
    hour_step, hour = losses_for(os.path.join(ROOT, "weights_csulb"))
    if [r["kind"] for r in author] != [r["kind"] for r in hour]:
        raise SystemExit("the two runs did not label the same pieces")
    rows = []
    for a, h in zip(author, hour):
        rows.append({
            "i": a["i"], "kind": a["kind"],
            "author_nats": a["nats"], "hour8_nats": h["nats"],
        })
    path = os.path.join(OUT_DIR, "enwik9_pieces.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r)
    summary = {
        "pieces": N, "piece_bytes": PIECE,
        "author_step": author_step, "hour8_step": hour_step,
        "author_bpb": sum(r["author_nats"] for r in rows) / N / LN2,
        "hour8_bpb": sum(r["hour8_nats"] for r in rows) / N / LN2,
        "kinds": {},
    }
    for kind, group in sorted(by.items()):
        n = len(group)
        a = sum(r["author_nats"] for r in group) / n / LN2
        h = sum(r["hour8_nats"] for r in group) / n / LN2
        summary["kinds"][kind] = {
            "n": n, "author_bpb": a, "hour8_bpb": h, "delta_bpb": h - a,
        }
    sp = os.path.join(OUT_DIR, "enwik9_pieces_summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
        f.write("\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
