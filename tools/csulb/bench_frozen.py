"""
Run one frozen benchmark and keep the record.

    .venv/bin/python tools/csulb/bench_frozen.py \
        --tool heldout --weights weights --held-out data/val --repeat 1

Always passes --force. A tool that reports an existing measurement, instead
of computing one, fails this protocol. The copy under research/eval/benchmarks
is the record that is reported. The shared files under runs/ are not.
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEST = os.path.join(ROOT, "research", "eval", "benchmarks")
PY = os.path.join(ROOT, ".venv", "bin", "python")


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def stamp(path):
    st = os.stat(path)
    return {"mtime_ns": st.st_mtime_ns, "size": st.st_size}


def snapshot(weights):
    files = [
        os.path.join(ROOT, "weights", "manifest.json"),
        os.path.join(ROOT, weights, "manifest.json"),
        os.path.join(ROOT, weights, "core.npz"),
    ]
    return {f: stamp(f) for f in files}


def run_one(tool, weights, repeat, held_out=None):
    os.makedirs(DEST, exist_ok=True)
    tag = {"heldout": "heldout", "enwik9": "enwik9", "pg19": "pg19"}[tool]
    mix = ""
    if tool == "heldout":
        mix = "__val" if held_out.rstrip("/").endswith("data/val") else "__mix"
    name = f"{tag}{mix}__{weights}__r{repeat}.json"
    dest = os.path.join(DEST, name)
    if os.path.exists(dest):
        old = json.load(open(dest))
        if old.get("fresh") is True:
            print(f"  already fresh: {name}", flush=True)
            return dest
    tmp = os.path.join(DEST, f".tmp_{tag}_{weights}_r{repeat}.json")
    if tool == "heldout":
        cmd = [PY, "-u", os.path.join(ROOT, "tools", "measure_heldout.py"),
               "--weights", os.path.join(ROOT, weights),
               "--held-out", held_out, "--out", tmp, "--force"]
    elif tool == "enwik9":
        cmd = [PY, "-u", os.path.join(ROOT, "tools", "measure_enwik9.py"),
               "--weights", os.path.join(ROOT, weights),
               "--pieces", "2000", "--out", tmp, "--force"]
    else:
        cmd = [PY, "-u", os.path.join(ROOT, "tools", "eval_pg19.py"),
               "--weights", os.path.join(ROOT, weights),
               "--books", os.path.join(ROOT, "data", "eval", "pg19"),
               "--fraction", "1", "--out", tmp, "--force"]
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import measure_heldout as MH
    pid = MH.training_run(os.path.join(ROOT, weights))
    if pid:
        raise SystemExit(f"a training run is going (pid {pid})")
    before = snapshot(weights)
    t0 = time.time()
    print(f"\n== {name} ==", flush=True)
    print(" ", " ".join(cmd), flush=True)
    log_path = os.path.join(DEST, "logs", name + ".log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = []
    with open(log_path, "w") as logf:
        for line in proc.stdout:
            lines.append(line)
            logf.write(line)
            logf.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
    code = proc.wait()
    if code != 0:
        raise SystemExit(f"{name} exited {code}")
    blob = "".join(lines)
    if "not measuring again" in blob:
        raise SystemExit(f"{name} reused a measurement; the protocol forbids that")
    if not os.path.exists(tmp):
        raise SystemExit(f"{name} produced no record")
    after = snapshot(weights)
    if before != after:
        raise SystemExit(f"{name} changed a checkpoint file")
    rec = json.load(open(tmp))
    if tool == "pg19" and rec.get("window") not in (0, 4096):
        raise SystemExit(f"{name} window is {rec.get('window')}, not 4096")
    wrapped = {
        "fresh": True,
        "git": git_head(),
        "tool": tool,
        "weights": weights,
        "repeat": repeat,
        "held_out": held_out,
        "seconds_wall": round(time.time() - t0, 1),
        "weights_manifest_mtime_ns": before[os.path.join(ROOT, "weights", "manifest.json")]["mtime_ns"],
        "measurement": rec,
    }
    with open(dest + ".tmp", "w") as f:
        json.dump(wrapped, f, indent=1)
    os.replace(dest + ".tmp", dest)
    os.remove(tmp)
    print(f"  fresh record -> {os.path.relpath(dest, ROOT)}", flush=True)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True, choices=["heldout", "enwik9", "pg19"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--held-out", default=None)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    if a.tool == "heldout" and not a.held_out:
        raise SystemExit("heldout needs --held-out")
    run_one(a.tool, a.weights, a.repeat, a.held_out)


if __name__ == "__main__":
    main()
