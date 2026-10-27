"""
Record the 8-hour CSULB run without ever signaling the trainer.

During the run it waits until training time crosses 2, 4, and 6 hours, then
until the next "weights/ checkpointed" line, hardlinks that checkpoint, and
appends the losses to runs/csulb_8h_review.md. At the final save it appends
hour 8. After the four evaluations, `--scores` adds the question scores.

    .venv/bin/python tools/csulb/watch_8h.py --pid PID --sample-from BYTES
    .venv/bin/python tools/csulb/watch_8h.py --scores
"""
import argparse
import json
import os
import re
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG = os.path.join(ROOT, "runs", "csulb_8h.log")
SAMPLES = os.path.join(ROOT, "runs", "csulb_samples.txt")
REVIEW = os.path.join(ROOT, "runs", "csulb_8h_review.md")
WEIGHTS = os.path.join(ROOT, "weights_csulb")
EXPECTED = ["chat", "chess", "code", "csulb_policies", "csulb_qa",
            "csulb_qa_open", "reasoning", "wikipedia"]
# training minutes at which a hardlink snapshot is taken
MARKS = [(120, "h2"), (240, "h4"), (360, "h6")]
BAR = "=" * 78


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def append(text):
    with open(REVIEW, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def log_text():
    if not os.path.exists(LOG):
        return ""
    with open(LOG, encoding="utf-8", errors="replace") as f:
        return f.read()


def new_samples(offset):
    with open(SAMPLES, encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        return f.read()


def sample_blocks(text):
    """Header of each sample entry written after `offset`, in order."""
    out = []
    parts = text.split(BAR)
    # entries are BAR, header, BAR, generations, BAR, header, ...
    for i in range(1, len(parts), 2):
        header = parts[i]
        if "step " in header and " min" in header:
            out.append(header.strip("\n"))
    return out


def parse_block(header):
    m = re.search(r"step ([\d,]+)\s+([\d.]+)M\b.*?([\d.]+) min", header)
    step = chars = minutes = None
    if m:
        step = int(m.group(1).replace(",", ""))
        chars = float(m.group(2))
        minutes = float(m.group(3))
    train = re.search(r"train loss ([\d.]+)", header)
    held = re.search(r"held-out loss ([\d.]+)(?:\s+\+/-([\d.]+))?", header)
    rate = re.search(r"rate x([\d.]+)", header)
    lr = re.search(r"\blr ([\d.eE+-]+)", header)
    dom = ""
    for line in header.splitlines():
        if line.startswith("  ") and "csulb_qa" in line:
            dom = line.strip()
    return dict(step=step, chars_m=chars, minutes=minutes,
                train=float(train.group(1)) if train else None,
                held=float(held.group(1)) if held else None,
                held_se=float(held.group(2)) if held and held.group(2) else None,
                rate=float(rate.group(1)) if rate else None,
                lr=lr.group(1) if lr else None,
                domains=dom, raw=header)


def manifest_scale():
    try:
        with open(os.path.join(WEIGHTS, "manifest.json"), encoding="utf-8") as f:
            p = json.load(f).get("plasticity") or {}
        return p.get("scale")
    except OSError:
        return None


def session_base_m(text):
    m = re.search(r"# session started .*? ([\d.]+)M of", text)
    return float(m.group(1)) if m else None


def section(title, info, note, base_m=None):
    se = f" +/-{info['held_se']:.4f}" if info.get("held_se") is not None else ""
    session = None
    if base_m is not None and info.get("chars_m") is not None:
        session = info["chars_m"] - base_m
    lines = [
        f"## {title}",
        "",
        f"- Recorded: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Trainer still running: {note}",
        f"- Step: {info['step']}",
        f"- Characters read this session: {None if session is None else round(session, 2)}M",
        f"- Characters read over the model's life: {info['chars_m']}M",
        f"- Minutes into this session: {info['minutes']}",
        f"- Minutes of the 8-hour plan: {info.get('plan_minutes', info['minutes'])}",
        f"- Train loss: {info['train']}",
        f"- Held-out loss: {info['held']}{se}",
        f"- Learning-rate scale: {info['rate'] if info['rate'] is not None else manifest_scale()}",
        f"- Base learning rate: {info['lr']}",
        f"- Per subject: {info['domains'] or '(not in this sample)'}",
        "",
        "```",
        info["raw"],
        "```",
        "",
    ]
    return "\n".join(lines)


def hardlink(tag):
    dest = os.path.join(ROOT, f"weights_csulb_{tag}")
    if os.path.exists(dest):
        return f"refused: {dest} already exists"
    subprocess.check_call(["cp", "-al", WEIGHTS, dest], cwd=ROOT)
    n = sum(len(files) for _, _, files in os.walk(dest))
    return f"hardlinked {n} files to {os.path.basename(dest)}"


def untouch_note(stamps):
    changed = []
    for path, old in stamps.items():
        try:
            now = os.stat(path).st_mtime_ns
        except OSError:
            changed.append(path + " missing")
            continue
        if now != old:
            changed.append(path)
    if changed:
        return "CHANGED (this run must not write these): " + ", ".join(changed)
    return "unchanged: weights/manifest.json and weights_csulb_65min/manifest.json"


def watch(pid, offset, stamps, already=0.0):
    done = {}
    pending = None          # (tag, log length when the sample was first seen)
    hour8 = False
    while not hour8:
        text = log_text()
        fresh = new_samples(offset)
        base_m = session_base_m(fresh)
        blocks = [parse_block(b) for b in sample_blocks(fresh)]
        blocks = [b for b in blocks if b["minutes"] is not None]
        if pending is None:
            for minutes, tag in MARKS:
                if tag in done:
                    continue
                if any(b["minutes"] + already >= minutes for b in blocks):
                    pending = (tag, len(text), minutes)
                    break
        if pending is not None:
            tag, at, minutes = pending
            if "checkpointed" in text[at:]:
                info = dict([b for b in blocks if b["minutes"] + already >= minutes][-1])
                info["plan_minutes"] = round(info["minutes"] + already, 1)
                try:
                    snap = hardlink(tag)
                except Exception as e:
                    snap = f"SNAPSHOT FAILED: {e}"
                note = f"yes (pid {pid})" if alive(pid) else f"NO (pid {pid} not running)"
                append(section(f"Hour {minutes / 60:.0f} ({tag})", info,
                               note, base_m) + f"- Snapshot: {snap}\n- Author weights: {untouch_note(stamps)}\n\n")
                done[tag] = snap
                pending = None
        if "weights/ updated" in text or (not alive(pid) and "reached " in text):
            info = blocks[-1] if blocks else dict(
                step=None, chars_m=None, minutes=None, train=None, held=None,
                held_se=None, rate=None, lr=None, domains="", raw="(no sample)")
            if blocks:
                info = dict(info)
                info["plan_minutes"] = round((info["minutes"] or 0) + already, 1)
            tail = "\n".join(text.splitlines()[-12:])
            body = section("Hour 8 (final save)", info, "no, the trainer has exited", base_m)
            body += f"- Author weights: {untouch_note(stamps)}\n\nLast log lines:\n\n```\n{tail}\n```\n\n"
            if "weights/ updated" not in text:
                body += "The trainer exited without `weights/ updated`. The hour-8 weights were not confirmed saved.\n\n"
            append(body)
            hour8 = True
            break
        if not alive(pid) and "weights/ updated" not in text and "reached " not in text:
            tail = "\n".join(text.splitlines()[-20:])
            waiting = f"\n\nWas waiting to snapshot {pending[0]}.\n" if pending else "\n"
            append(f"## Trainer exited early{waiting}\n```\n{tail}\n```\n\n")
            break
        time.sleep(20)


def questions_in_train():
    found = set()
    base = os.path.join(ROOT, "school_policies_data", "train", "csulb_qa")
    for root, _, files in os.walk(base):
        for fn in files:
            txt = open(os.path.join(root, fn), encoding="utf-8", errors="replace").read()
            i = 0
            while True:
                a = txt.find("<user>\n", i)
                if a < 0:
                    break
                b = txt.find("\n</user>", a)
                if b < 0:
                    break
                q = txt[a + 7:b].strip()
                if "[Question]\n" in q:
                    q = q.split("[Question]\n", 1)[1].strip()
                found.add(q)
                i = b + 7
    return found


def mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def score_rows(rows, train_q):
    scored = [r for r in rows if "open_f1" in r]
    clean = [r for r in scored if r["q"].strip() not in train_q]
    leaked = [r for r in scored if r["q"].strip() in train_q]

    def pack(group):
        if not group:
            return None
        out = {}
        for mode in ("closed", "open"):
            out[mode] = dict(
                f1=mean(r[f"{mode}_f1"] for r in group),
                num=mean(r[f"{mode}_num"] for r in group if r[f"{mode}_num"] is not None),
                url=mean(bool(r[f"{mode}_url"]) for r in group),
                form=mean(bool(r[f"{mode}_form"]) for r in group),
            )
        return out

    return pack(scored), pack(clean), len(scored), len(clean), len(leaked)


def write_scores():
    train_q = questions_in_train()
    lines = ["## Question scores after the run", "",
             "Scored after training finished. Open-book shows the model the top search hit. "
             "The clean set is the test questions whose wording does not appear in "
             "`train/csulb_qa`. The other test questions use a wording that training pairs "
             "with a different document.", "",
             "| checkpoint | set | n | closed F1 | closed number | closed URL | closed form | open F1 | open number | open URL | open form |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in ("h2", "h4", "h6", "h8"):
        path = os.path.join(ROOT, "runs", f"csulb_eval_{tag}.json")
        if not os.path.exists(path):
            lines.append(f"| {tag} | missing | | | | | | | | | |")
            continue
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)["rows"]
        all_, clean, n, nc, nl = score_rows(rows, train_q)
        for label, pack, count in ((f"all (leaked {nl})", all_, n), ("clean wording", clean, nc)):
            if pack is None:
                continue
            c, o = pack["closed"], pack["open"]
            lines.append(
                f"| {tag} | {label} | {count} | {pct(c['f1'])} | {pct(c['num'])} | {pct(c['url'])} | "
                f"{pct(c['form'])} | {pct(o['f1'])} | {pct(o['num'])} | {pct(o['url'])} | {pct(o['form'])} |")
    lines.append("")
    append("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pid", type=int)
    ap.add_argument("--sample-from", type=int, default=0)
    ap.add_argument("--offset-minutes", type=float, default=0.0,
                    help="minutes already trained in this 8-hour plan before this process")
    ap.add_argument("--scores", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.scores:
        write_scores()
        return
    if not a.pid:
        ap.error("--pid is required")
    stamps = {}
    for rel in ("weights/manifest.json", "weights_csulb_65min/manifest.json"):
        stamps[os.path.join(ROOT, rel)] = os.stat(os.path.join(ROOT, rel)).st_mtime_ns
    watch(a.pid, a.sample_from, stamps, a.offset_minutes)


if __name__ == "__main__":
    main()
