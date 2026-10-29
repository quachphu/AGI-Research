"""
The frozen-benchmark sequence in research/eval_protocol.md.

Held-out on data/val and on the CSULB mix, then enwik9, for all six
checkpoints. PG19 for the author's weights, the 65-minute checkpoint, and
hour 8. The author's weights and hour 8 are measured twice. A finished
fresh record is left in place, so a stopped suite can be started again.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_frozen as B  # noqa: E402

SIX = ["weights", "weights_csulb_65min", "weights_csulb_h2",
       "weights_csulb_h4", "weights_csulb_h6", "weights_csulb"]
PG = ["weights", "weights_csulb_65min", "weights_csulb"]


def jobs():
    out = []
    for held in ("data/val", "school_policies_data/heldout_mix"):
        for w in SIX:
            reps = 2 if w in ("weights", "weights_csulb") else 1
            for r in range(1, reps + 1):
                out.append(("heldout", w, r, held))
    for w in SIX:
        reps = 2 if w in ("weights", "weights_csulb") else 1
        for r in range(1, reps + 1):
            out.append(("enwik9", w, r, None))
    for w in PG:
        reps = 2 if w in ("weights", "weights_csulb") else 1
        for r in range(1, reps + 1):
            out.append(("pg19", w, r, None))
    return out


def main():
    todo = jobs()
    print(f"{len(todo)} measurements, code {B.git_head()}", flush=True)
    for i, (tool, weights, repeat, held) in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}]", flush=True)
        B.run_one(tool, weights, repeat, held)
    print("\nsuite finished", flush=True)


if __name__ == "__main__":
    main()
