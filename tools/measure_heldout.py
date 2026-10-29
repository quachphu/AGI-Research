#!/usr/bin/env python3
"""
Held-out for the README, measured on the model as it is now.

A run's checks read a few blocks of each subject and are reported at the
level its session started with (minagi/heldout.py). What the README says
about where the model is should be a measurement, not a check carried
forward: this reads the blocks the run's own start and end evaluations read
- `data.eval_chars` of each subject, as whole blocks spread over all of it -
on weights/ as it is now, and writes runs/heldout_measured.json, which
tools/refresh_readme.py builds the README's table from. --all reads every
block of every subject instead: the whole held-out set, no sample at all,
for about twenty-five times as long.

NOTHING IS MEASURED TWICE. If these weights were measured already - by this
tool, or by the run itself, which ends a cleanly stopped session with the
same evaluation on the weights it leaves behind - that measurement is used
and the card is not touched.

It needs the card and the weights to itself, so it will not run beside a
training run - and publish.sh will not publish without it: it refuses to
start while a run is training (--check) and stops if this fails.

    python3 tools/measure_heldout.py
    python3 tools/measure_heldout.py --all
    python3 tools/measure_heldout.py --check     # is a run training?

--held-out names a folder other than data.held_out. --out names the file.
--force measures again even when a measurement of these weights exists.
With none of those three, this tool does what it did before they existed.
A held-out folder other than data.held_out is not the run's own evaluation,
so the sample log is not reused for it.
"""
import argparse
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from minagi import heldout                                  # noqa: E402

OUT = os.path.join(ROOT, "runs", "heldout_measured.json")
LOG = os.path.join(ROOT, "runs", "samples.txt")
HEAD = re.compile(r"^step ([\d,]+)\s+([\d.]+)M (?:of [\d,]+M )?characters"
                  r"[^\n]*?(\d+) experts", re.M)


def training_run(wdir):
    """The pid of another python process running train.py from this
    repository, or naming `wdir` on its command line; None if there is none."""
    me, here = os.getpid(), os.path.realpath(ROOT)
    w = os.path.realpath(wdir).encode()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().split(b"\0")
            cwd = os.path.realpath(f"/proc/{pid}/cwd")
        except OSError:
            continue
        if (int(pid) == me or not os.path.basename(cmd[0]).startswith(b"python")
                or not any(x.endswith(b"train.py") for x in cmd[1:])):
            continue
        if cwd == here or any(w in x for x in cmd):
            return int(pid)
    return None


def sample(cfg_block):
    """How the run's evaluation reads each subject, from config.yaml: (blocks
    a subject, characters a block)."""
    from minagi.config import load, get
    c = load()
    chunk = int(get(c, "training.chunk", 2048))
    per = max(1, int(cfg_block) // chunk)
    steps = max(1, int(get(c, "data.eval_chars", 240 * 512)) // chunk)
    return -(-steps // per), chunk * per


def same_weights(d, man):
    """Whether figures `d` describe the weights `man` describes: the same
    step, pool and characters read."""
    return (d.get("step") == man.get("step")
            and d.get("experts") == int(man.get("n_experts") or 0)
            and abs((d.get("chars") or 0) - (man.get("read_chars") or 0)) < 1e5)


def already(man, want_all=False, out=OUT, log=LOG, log_reuse=True):
    """A measurement of the weights `man` describes that exists already, or
    None: this tool's own, or the evaluation the run ended on - the last
    entry of its sample log, taken after checks were spread over the whole
    set, without the evidence line a check carries, on these same weights.
    Only this tool's own reading of every block will do for --all, and none
    taken before held-out counted whole blocks only (`whole` unset).
    `log_reuse` false skips the sample log, which describes data.held_out
    and not some other folder."""
    try:
        with open(out) as f:
            d = json.load(f)
        if (same_weights(d, man) and d.get("whole")
                and (d.get("all") or not want_all)):
            return d
    except (OSError, ValueError):
        pass
    if want_all or not log_reuse:
        return None
    try:
        with open(log, errors="ignore") as f:
            text = f.read()
    except OSError:
        return None
    heads = list(HEAD.finditer(text))
    first = heldout.first_shift(text)
    whole = heldout.part_shift(text)
    # an evaluation from before held-out counted whole blocks only is a
    # different measurement from the one this tool makes
    if (not heads or first is None or whole is None
            or heads[-1].start() < max(first[0], whole[0])):
        return None
    m = heads[-1]
    head = text[m.start():text.find("\n" + "=" * 78, m.start())]
    v = re.search(r"^held-out loss ([\d.]+) \+/-([\d.]+)", head, re.M)
    dom = re.search(r"^  ([a-z_]+ [\d.]+(?:   [a-z_]+ [\d.]+)*)\s*$", head, re.M)
    d = {"step": int(m.group(1).replace(",", "")),
         "chars": float(m.group(2)) * 1e6, "experts": int(m.group(3))}
    if heldout.is_check(head) or not v or not dom or not same_weights(d, man):
        return None
    parts = dom.group(1).split()
    per = {parts[i]: float(parts[i + 1]) for i in range(0, len(parts) - 1, 2)}
    n, size = sample(man["cfg"]["block"])
    # the log rounds characters read to a tenth of a million; the manifest
    # has them exactly
    d.update(chars=man.get("read_chars"), val=float(v.group(1)),
             se=float(v.group(2)), per_subject=per,
             blocks={k: n for k in per}, block_chars=size, all=False,
             whole=True, date=time.strftime("%Y-%m-%d %H:%M"),
             source="the run's own evaluation at the end of its session")
    return d


def write(d, path=None):
    path = path or OUT
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(d, f, indent=1)
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights"))
    ap.add_argument("--all", action="store_true",
                    help="every block of every subject, not the run's sample")
    ap.add_argument("--device", default=None,
                    help="cuda unless said otherwise; the CPU is for tests")
    ap.add_argument("--check", action="store_true",
                    help="only say whether a training run is going: exit 3 "
                         "if one is, 0 if not")
    ap.add_argument("--held-out", default=None,
                    help="held-out folder; default is data.held_out in "
                         "config.yaml")
    ap.add_argument("--out", default=None,
                    help="where to write the measurement; default "
                         "runs/heldout_measured.json")
    ap.add_argument("--force", action="store_true",
                    help="measure even if these weights were measured "
                         "already")
    a = ap.parse_args()

    pid = training_run(a.weights)
    if a.check:
        if pid:
            print(f"a training run is going (pid {pid})")
            sys.exit(3)
        return
    if pid:
        sys.exit(f"  a training run is going (pid {pid}) - stop it to measure "
                 f"held-out")
    if any(n.endswith(".tmp") or ".tmp." in n for n in os.listdir(a.weights)):
        sys.exit(f"  {a.weights} has a save in progress - not measuring")

    with open(os.path.join(a.weights, "manifest.json")) as f:
        man = json.load(f)
    out_path = (a.out if not a.out or os.path.isabs(a.out)
                else os.path.join(ROOT, a.out)) or OUT
    log_reuse = True
    held = None
    if a.held_out:
        from minagi.config import get as _get, load as _load
        default_held = os.path.join(
            ROOT, str(_get(_load(), "data.held_out", "data/val")))
        held = (a.held_out if os.path.isabs(a.held_out)
                else os.path.join(ROOT, a.held_out))
        log_reuse = os.path.realpath(held) == os.path.realpath(default_held)
    if not a.force:
        have = already(man, a.all, out=out_path, log_reuse=log_reuse)
        if have is not None:
            if have.get("source"):
                write(have, out_path)
            print(f"  held-out {have['val']:.4f} +/-{have['se']:.4f} at step "
                  f"{have['step']:,} is already measured on these weights"
                  + (f" - {have['source']}" if have.get("source") else "")
                  + "; not measuring again")
            return

    import numpy as np
    import torch
    import train as T
    from minagi.config import load, get
    from minagi.precision import set_compute_dtype
    from minagi.stream import FolderEvaluator

    if a.device is None and not torch.cuda.is_available():
        sys.exit("  no GPU - held-out on the CPU would take hours")
    c = load()
    if held is None:
        held = os.path.join(ROOT, str(get(c, "data.held_out", "data/val")))
    chunk = int(get(c, "training.chunk", 2048))
    chars = int(get(c, "data.eval_chars", 240 * 512))
    set_compute_dtype(str(get(c, "training.precision", "bf16")))
    device = torch.device(a.device or "cuda")
    t0 = time.time()
    model, cfg, pool, man = T.build_paged(
        a.weights, device, ram_capacity=int(get(c, "pool.ram_cache", 96)),
        read_only=True)
    ev = FolderEvaluator(model, held, chunk, cfg.block, device)
    steps = 10 ** 9 if a.all else max(1, chars // max(1, chunk))
    v = ev.run(steps, spread=True)
    se = v.pop("stderr", 0.0)
    n = {k: len(ev._texts(k, max(1, -(-steps // ev.per)))) for k in v}
    val = float(np.mean(list(v.values())))
    out = {"step": man.get("step"), "chars": man.get("read_chars"),
           "experts": int(man.get("n_experts") or 0), "val": val, "se": se,
           "per_subject": v, "blocks": n, "block_chars": ev.block,
           "all": bool(a.all), "whole": True,
           "date": time.strftime("%Y-%m-%d %H:%M"),
           "seconds": round(time.time() - t0)}
    write(out, out_path)
    print(f"  held-out {val:.4f} +/-{se:.4f} at step {out['step']:,}, "
          f"{'every block' if a.all else f'{min(n.values())}-{max(n.values())} blocks'} "
          f"of each subject, in {out['seconds']} s -> {os.path.relpath(out_path, ROOT)}")


if __name__ == "__main__":
    main()
