    # Caller information (for building call graph edges)
    caller_func_name: str | None = None
    caller_filename: str | None = None
    caller_firstlineno: int | None = None
    # Whether this is a primitive (non-recursive) call
    # A call is primitive if the function doesn't appear anywhere in the call stack
    is_primitive_call: bool = True
    # Full call stack at the time of this call (for proper snakeviz drill-down)
    # Each entry is (func_name, filename, firstlineno)
    call_stack: tuple[tuple[str, str, int], ...] = ()

    # Backwards compatibility alias
    @property
    def trace_time_ns(self) -> int:
        return self.cumtime_ns

    @property
    def trace_time_ms(self) -> float:
        return self.cumtime_ns / 1e6

    @property
    def cumtime_ms(self) -> float:
        return self.cumtime_ns / 1e6

    @property
    def tottime_ms(self) -> float:
        return self.tottime_ns / 1e6

    @property
    def caller_key(self) -> tuple[str, int, str] | None:
        """Return caller as a pstats-compatible key tuple."""
        if self.caller_func_name is not None:
            return (
                self.caller_filename or "",
                self.caller_firstlineno or 0,
                self.caller_func_name,
            )
        return None

    @property
    def func_key(self) -> tuple[str, int, str]:
        """Return this function as a pstats-compatible key tuple."""
        return (self.filename, self.firstlineno, self.func_name)

    def __repr__(self) -> str:
        return (
            f"FunctionTraceTiming({self.func_name} at {self.filename}:{self.firstlineno}, "
            f"cumtime={self.cumtime_ms:.2f}ms, tottime={self.tottime_ms:.2f}ms, "
            f"bytecode={self.bytecode_count}, depth={self.inline_depth})"
        )


@dataclass
class ProfilerStackEntry:
    """Stack entry for tracking function timing in the Dynamo profiler."""

    func_name: str
    filename: str
    firstlineno: int
    start_time_ns: int
    child_time_ns: int  # Accumulated time spent in traced children
    is_primitive_call: bool = True  # Whether this is a non-recursive call


class DynamoProfilerState:
    """State for Dynamo profiler tracking function trace timings."""

    def __init__(self) -> None:
        self.timings: list[FunctionTraceTiming] = []
        self.stack: list[ProfilerStackEntry] = []

    def record_timing(self, timing: FunctionTraceTiming) -> None:
        """Record timing data for a traced function."""
        self.timings.append(timing)

    def get_timings(self) -> list[FunctionTraceTiming]:
        """Get all recorded timings."""
        return self.timings

    def push(
        self, func_name: str, filename: str, firstlineno: int, start_time_ns: int
    ) -> None:
        """Push a new entry onto the timing stack."""
        # Check if this function already exists in the stack (indirect recursion)
        is_primitive = not any(
            entry.func_name == func_name
            and entry.filename == filename
            and entry.firstlineno == firstlineno
            for entry in self.stack
        )
        self.stack.append(
            ProfilerStackEntry(
                func_name=func_name,
                filename=filename,
                firstlineno=firstlineno,
                start_time_ns=start_time_ns,
                child_time_ns=0,
                is_primitive_call=is_primitive,
            )
        )

    def pop(self) -> ProfilerStackEntry | None:
        """Pop the top entry from the timing stack."""
        if self.stack:
            return self.stack.pop()
        return None

    def add_child_time(self, child_cumtime_ns: int) -> None:
        """Add the child's cumulative time to the parent's child_time accumulator."""
        if self.stack:
            self.stack[-1].child_time_ns += child_cumtime_ns

    def get_current_caller(self) -> tuple[str, str, int] | None:
        """Get the current caller (top of stack) as (func_name, filename, firstlineno)."""
        if self.stack:
            entry = self.stack[-1]
            return (entry.func_name, entry.filename, entry.firstlineno)
        return None

    def get_call_stack(self) -> tuple[tuple[str, str, int], ...]:
        """Get the full current call stack as tuple of (func_name, filename, firstlineno)."""
        return tuple(
            (entry.func_name, entry.filename, entry.firstlineno) for entry in self.stack
        )

    def generate_pstats(
        self, output_file: str | None = None, print_raw: bool = False
    ) -> pstats.Stats:
        """Generate pstats.Stats object from recorded timings.

        Args:
            output_file: Optional file path to save the stats.
            print_raw: If True, print raw aggregate timings before returning.
        """
        import cProfile
        import io
        import logging
        import pstats

        log = logging.getLogger(__name__)

        # Aggregate by (filename, lineno, func_name)
        aggregated: dict[tuple[str, int, str], dict[str, Any]] = {}
        # caller_edges[callee_key][caller_key] -> edge stats
        caller_edges: dict[
            tuple[str, int, str], dict[tuple[str, int, str], dict[str, Any]]
        ] = {}

        for t in self.timings:
            key = (t.filename, t.firstlineno, t.func_name)

            if key not in aggregated:
                aggregated[key] = {
                    "ncalls": 0,
                    "pcalls": 0,
                    "tottime": 0.0,
                    "cumtime": 0.0,
                }
                caller_edges[key] = {}

            agg = aggregated[key]
            agg["ncalls"] += 1
            agg["tottime"] += t.tottime_ns / 1e9

            if t.is_primitive_call:
                agg["pcalls"] += 1
                agg["cumtime"] += t.cumtime_ns / 1e9

            # Build caller edge
            if t.caller_filename is not None:
                caller_key = (
                    t.caller_filename,
                    t.caller_firstlineno or 0,
                    t.caller_func_name or "",
                )
                if caller_key not in caller_edges[key]:
                    caller_edges[key][caller_key] = {
                        "ncalls": 0,
                        "pcalls": 0,
                        "tottime": 0.0,
                        "cumtime": 0.0,
                    }
                edge = caller_edges[key][caller_key]
                edge["ncalls"] += 1
                edge["tottime"] += t.tottime_ns / 1e9
                # Always add cumtime to edges for visualization (gprof2dot)
                # Function-level cumtime is already correct (only primitive calls)
                edge["cumtime"] += t.cumtime_ns / 1e9
                if t.is_primitive_call:
                    edge["pcalls"] += 1

        if print_raw:
            sorted_items = sorted(
                aggregated.items(), key=lambda x: x[1]["cumtime"], reverse=True
            )
            print("\n=== Aggregate Timings (raw) ===")
            print(
                f"{'ncalls':>8} {'pcalls':>8} {'tottime':>12} {'cumtime':>12}  function"
            )
            print("-" * 80)
            total_cumtime = 0.0
            total_tottime = 0.0
            for (filename, lineno, func_name), agg in sorted_items:
                ncalls = agg["ncalls"]
                pcalls = agg["pcalls"]
                tottime = agg["tottime"] * 1000  # convert to ms
                cumtime = agg["cumtime"] * 1000
                total_cumtime += cumtime
                total_tottime += tottime
                short_file = filename.split("/")[-1] if "/" in filename else filename
                print(
                    f"{ncalls:>8} {pcalls:>8} {tottime:>10.2f}ms {cumtime:>10.2f}ms  "
                    f"{func_name} ({short_file}:{lineno})"
                )
            print("-" * 80)
            print(
                f"Total timings: {len(self.timings)}, unique functions: {len(aggregated)}"
            )
            print(
                f"Sum tottime: {total_tottime:.2f}ms, Sum cumtime: {total_cumtime:.2f}ms"
            )

        # Ensure caller-only functions have a top-level entry.
        # gprof2dot expects every function referenced as a caller to also
        # exist as a top-level entry in the stats dict with timing data.
        for key in list(caller_edges.keys()):
            for caller_key in caller_edges[key]:
                if caller_key not in aggregated:
                    aggregated[caller_key] = {
                        "ncalls": 0,
                        "pcalls": 0,
                        "tottime": 0.0,
                        "cumtime": 0.0,
                    }
                    caller_edges[caller_key] = {}

        # Build the stats dict in pstats format
        stats_dict: dict[
            tuple[str, int, str], tuple[int, int, float, float, dict[Any, Any]]
        ] = {}

        for key, agg in aggregated.items():
            callers: dict[tuple[str, int, str], tuple[int, int, float, float]] = {}
            for caller_key, edge in caller_edges[key].items():
                callers[caller_key] = (
                    edge["ncalls"],
                    edge["pcalls"],
                    edge["tottime"],
                    edge["cumtime"],
                )

            stats_dict[key] = (
                agg["pcalls"],
                agg["ncalls"],
                agg["tottime"],
                agg["cumtime"],
                callers,
            )

        # Create a pstats.Stats object
        dummy_profile = cProfile.Profile()
        dummy_profile.enable()
        dummy_profile.disable()
        stats = pstats.Stats(dummy_profile, stream=io.StringIO())

        stats.stats = stats_dict  # type: ignore[attr-defined]
        stats.total_calls = sum(s[1] for s in stats_dict.values())  # type: ignore[attr-defined]
        stats.prim_calls = sum(s[0] for s in stats_dict.values())  # type: ignore[attr-defined]
        stats.total_tt = sum(s[2] for s in stats_dict.values())  # type: ignore[attr-defined]

        if output_file:
            stats.dump_stats(output_file)
            log.info(
                "Saved pstats to %s. Visualize with: snakeviz %s",
                output_file,
                output_file,
            )

        return stats

    def generate_svg(
        self, profile_file: str, svg_file: str | None = None
    ) -> str | None:
        """Generate an SVG call graph from a profile file using gprof2dot and graphviz.

        Args:
            profile_file: Path to the pstats profile file.
            svg_file: Optional path for the output SVG. If not provided, uses
                profile_file with .svg extension.

        Returns:
            Path to the generated SVG file, or None if generation failed.
        """
        import os
        import shutil
        import subprocess

        if not shutil.which("gprof2dot"):
            print("gprof2dot not found. Install with: pip install gprof2dot")
            return None

        if not shutil.which("dot"):
            print("graphviz 'dot' not found. Install graphviz package.")
            return None

        if svg_file is None:
            svg_file = profile_file.rsplit(".", 1)[0] + ".svg"

        try:
            # gprof2dot -f pstats profile.prof | dot -Tsvg -o profile.svg
            gprof2dot = subprocess.Popen(
                [
                    "gprof2dot",
                    "-f",
                    "pstats",
                    "--node-label=total-time-percentage",
                    "--node-label=self-time-percentage",
                    "--node-label=total-time",
                    profile_file,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            dot = subprocess.Popen(
                ["dot", "-Tsvg", "-o", svg_file],
                stdin=gprof2dot.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            gprof2dot.stdout.close()  # type: ignore[union-attr]
            _, dot_err = dot.communicate()
            _, gprof2dot_err = gprof2dot.communicate()

            if gprof2dot.returncode != 0:
                print(f"gprof2dot failed: {gprof2dot_err.decode()}")
                return None

            if dot.returncode != 0:
                print(f"graphviz dot failed: {dot_err.decode()}")
                return None

            if not os.path.isfile(svg_file):
                print(f"SVG file was not created: {svg_file}")
                return None

            print(f"SVG call graph saved to: {svg_file}")
            return svg_file

        except Exception as e:
            print(f"Failed to generate SVG: {e}")
            return None

    def dump_stats(
        self, output_file: str | None = None, generate_svg: bool = True
    ) -> None:
        """Print profiler stats to stdout and optionally save to file.

        Args:
            output_file: Optional path to save the pstats profile.
            generate_svg: If True and output_file is provided, also generate an SVG
                call graph using gprof2dot and graphviz.
        """
        import sys

        if not self.timings:
            return

        stats = self.generate_pstats(output_file, print_raw=True)
        print("\n=== Dynamo Profiler (pstats) ===")
        stats.stream = sys.stdout  # type: ignore[attr-defined]
        stats.sort_stats("cumulative").print_stats()

        if output_file:
            print(f"\nProfile saved to: {output_file}")
            print(f"Visualize with: snakeviz {output_file}")

            if generate_svg:
                self.generate_svg(output_file)
<|endoftext|>"""
This module provides utilities for analyzing and optimizing Python bytecode.
Key functionality includes:
- Dead code elimination
- Jump instruction optimization
- Stack size analysis and verification
- Live variable analysis
- Line number propagation and cleanup
- Exception table handling for Python 3.11+

The utilities in this module are used to analyze and transform bytecode
for better performance while maintaining correct semantics.
"""

import bisect
import dataclasses
import dis
import itertools
import sys
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    # TODO(lucaskabela): consider moving Instruction into this file
    # and refactoring in callsite; that way we don't have to guard this import
    from .bytecode_transformation import Instruction

TERMINAL_OPCODES = {
    dis.opmap["RETURN_VALUE"],
    dis.opmap["JUMP_FORWARD"],
    dis.opmap["RAISE_VARARGS"],
    # TODO(jansel): double check exception handling
}
TERMINAL_OPCODES.add(dis.opmap["RERAISE"])
if sys.version_info >= (3, 11):
    TERMINAL_OPCODES.add(dis.opmap["JUMP_BACKWARD"])
    TERMINAL_OPCODES.add(dis.opmap["JUMP_FORWARD"])
else:
    TERMINAL_OPCODES.add(dis.opmap["JUMP_ABSOLUTE"])

if (3, 12) <= sys.version_info < (3, 14):
    TERMINAL_OPCODES.add(dis.opmap["RETURN_CONST"])
if sys.version_info >= (3, 13):
    TERMINAL_OPCODES.add(dis.opmap["JUMP_BACKWARD_NO_INTERRUPT"])
JUMP_OPCODES = set(dis.hasjrel + dis.hasjabs)
JUMP_OPNAMES = {dis.opname[opcode] for opcode in JUMP_OPCODES}
HASLOCAL = set(dis.haslocal)
HASFREE = set(dis.hasfree)

stack_effect = dis.stack_effect


def get_indexof(insts: list["Instruction"]) -> dict["Instruction", int]:
    """
    Get a mapping from instruction memory address to index in instruction list.
    Additionally checks that each instruction only appears once in the list.
    """
    # pyrefly: ignore [implicit-any]
    indexof = {}
    for i, inst in enumerate(insts):
        if inst in indexof:
            raise AssertionError(
                f"duplicate instruction at index {i} in instruction list"
            )
        indexof[inst] = i
    return indexof


def remove_dead_code(instructions: list["Instruction"]) -> list["Instruction"]:
    """Dead code elimination"""
    indexof = get_indexof(instructions)
    live_code = set()

    def find_live_code(start: int) -> None:
        for i in range(start, len(instructions)):
            if i in live_code:
                return
            live_code.add(i)
            inst = instructions[i]
            if inst.exn_tab_entry:
                find_live_code(indexof[inst.exn_tab_entry.target])
            if inst.opcode in JUMP_OPCODES:
                if inst.target is None:
                    raise AssertionError(
                        f"jump instruction {inst.opname} has no target"
                    )
                find_live_code(indexof[inst.target])
            if inst.opcode in TERMINAL_OPCODES:
                return

    find_live_code(0)

    # change exception table entries if start/end instructions are dead
    # assumes that exception table entries have been propagated,
    # e.g. with bytecode_transformation.propagate_inst_exn_table_entries,
    # and that instructions with an exn_tab_entry lies within its start/end.
    if sys.version_info >= (3, 11):
        live_idx = sorted(live_code)
        for i, inst in enumerate(instructions):
            if i in live_code and inst.exn_tab_entry:
                # find leftmost live instruction >= start
                start_idx = bisect.bisect_left(
                    live_idx, indexof[inst.exn_tab_entry.start]
                )
                if start_idx >= len(live_idx):
                    raise AssertionError(
                        "no live instruction found at or after exn_tab_entry start"
                    )
                # find rightmost live instruction <= end
                end_idx = (
                    bisect.bisect_right(live_idx, indexof[inst.exn_tab_entry.end]) - 1
                )
                if end_idx < 0:
                    raise AssertionError(
                        "no live instruction found at or before exn_tab_entry end"
                    )
                if not (live_idx[start_idx] <= i <= live_idx[end_idx]):
                    raise AssertionError(
                        f"instruction {i} not within live range "
                        f"[{live_idx[start_idx]}, {live_idx[end_idx]}]"
                    )
                inst.exn_tab_entry.start = instructions[live_idx[start_idx]]
                inst.exn_tab_entry.end = instructions[live_idx[end_idx]]

    return [inst for i, inst in enumerate(instructions) if i in live_code]


def remove_pointless_jumps(instructions: list["Instruction"]) -> list["Instruction"]:
    """Eliminate jumps to the next instruction"""
    pointless_jumps = {
        id(a)
        for a, b in itertools.pairwise(instructions)
        if a.opname == "JUMP_ABSOLUTE" and a.target is b
    }
    return [inst for inst in instructions if id(inst) not in pointless_jumps]


def propagate_line_nums(instructions: list["Instruction"]) -> None:
    """Ensure every instruction has line number set in case some are removed"""
    cur_line_no = None

    def populate_line_num(inst: "Instruction") -> None:
        nonlocal cur_line_no
        if inst.starts_line:
            cur_line_no = inst.starts_line

        inst.starts_line = cur_line_no

    for inst in instructions:
        populate_line_num(inst)


def remove_extra_line_nums(instructions: list["Instruction"]) -> None:
    """Remove extra starts line properties before packing bytecode"""

    cur_line_no = None

    def remove_line_num(inst: "Instruction") -> None:
        nonlocal cur_line_no
        if inst.starts_line is None:
            return
        elif inst.starts_line == cur_line_no:
            inst.starts_line = None
        else:
            cur_line_no = inst.starts_line

    for inst in instructions:
        remove_line_num(inst)


@dataclasses.dataclass
class ReadsWrites:
    reads: set[Any]
    writes: set[Any]
    visited: set[Any]


def livevars_analysis(
    instructions: list["Instruction"], instruction: "Instruction"
) -> set[Any]:
    indexof = get_indexof(instructions)
    must = ReadsWrites(set(), set(), set())
    may = ReadsWrites(set(), set(), set())

    def walk(state: ReadsWrites, start: int) -> None:
        if start in state.visited:
            return
        state.visited.add(start)

        for i in range(start, len(instructions)):
            inst = instructions[i]
            if inst.opcode in HASLOCAL or inst.opcode in HASFREE:
                if "LOAD" in inst.opname or "DELETE" in inst.opname:
                    if inst.argval not in must.writes:
                        state.reads.add(inst.argval)
                elif "STORE" in inst.opname:
                    state.writes.add(inst.argval)
                elif inst.opname == "MAKE_CELL":
                    pass
                else:
                    raise NotImplementedError(f"unhandled {inst.opname}")
            if inst.exn_tab_entry:
                walk(may, indexof[inst.exn_tab_entry.target])
            if inst.opcode in JUMP_OPCODES:
                if inst.target is None:
                    raise AssertionError(
                        f"jump instruction {inst.opname} has no target"
                    )
                walk(may, indexof[inst.target])
                state = may
            if inst.opcode in TERMINAL_OPCODES:
                return

    walk(must, indexof[instruction])
    return must.reads | may.reads


@dataclasses.dataclass
class FixedPointBox:
    value: bool = True


@dataclasses.dataclass
class StackSize:
    low: int | float
    high: int | float
    fixed_point: FixedPointBox

    def zero(self) -> None:
        self.low = 0
        self.high = 0
        self.fixed_point.value = False

    def offset_of(self, other: "StackSize", n: int) -> None:
        prior = (self.low, self.high)
        self.low = min(self.low, other.low + n)
        self.high = max(self.high, other.high + n)
        if (self.low, self.high) != prior:
            self.fixed_point.value = False

    def exn_tab_jump(self, depth: int) -> None:
        prior = (self.low, self.high)
        self.low = min(self.low, depth)
        self.high = max(self.high, depth)
        if (self.low, self.high) != prior:
            self.fixed_point.value = False


def stacksize_analysis(instructions: list["Instruction"]) -> int | float:
    if not instructions:
        raise AssertionError("instructions list must not be empty")
    fixed_point = FixedPointBox()
    stack_sizes = {
        inst: StackSize(float("inf"), float("-inf"), fixed_point)
        for inst in instructions
    }
    stack_sizes[instructions[0]].zero()

    for _ in range(100):
        if fixed_point.value:
            break
        fixed_point.value = True

        for inst, next_inst in zip(instructions, instructions[1:] + [None]):
            stack_size = stack_sizes[inst]
            if inst.opcode not in TERMINAL_OPCODES:
                if next_inst is None:
                    raise AssertionError(f"missing next inst: {inst}")
                eff = stack_effect(inst.opcode, inst.arg, jump=False)
                stack_sizes[next_inst].offset_of(stack_size, eff)
            if inst.opcode in JUMP_OPCODES:
                if inst.target is None:
                    raise AssertionError(f"missing target: {inst}")
                stack_sizes[inst.target].offset_of(
                    stack_size, stack_effect(inst.opcode, inst.arg, jump=True)
                )
            if inst.exn_tab_entry:
                # see https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt
                # on why depth is computed this way.
                depth = inst.exn_tab_entry.depth + int(inst.exn_tab_entry.lasti) + 1
                stack_sizes[inst.exn_tab_entry.target].exn_tab_jump(depth)

    low = min(x.low for x in stack_sizes.values())
    high = max(x.high for x in stack_sizes.values())

    if not fixed_point.value:
        raise AssertionError("failed to reach fixed point")
    if low < 0:
        raise AssertionError(f"stack size analysis produced negative low value: {low}")
    return high
<|endoftext|>"""
Debug utilities for TorchDynamo compilation and execution.

This module provides various debugging tools and utilities for TorchDynamo, including:

- Minification support for reducing test cases while preserving bugs
- Input/output handling via InputReader and InputWriter for reproducible testing
- Accuracy checking between original and compiled models
- Neural network module string conversion via NNModuleToString
- Profiling tools and system information collection
- Buck build system integration for Meta-internal testing

Key classes:
- InputReader/InputWriter: Handle serialization of model inputs/outputs
- NNModuleToString: Converts nn.Modules to string representations
- BuckTargetWriter: Manages Buck build system integration
"""

from __future__ import annotations

import atexit
import copy
import cProfile
import functools
import getpass
import inspect
import itertools
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from importlib import import_module
from typing import Any, cast, TYPE_CHECKING, TypeVar

import torch
import torch._prims_common as utils
import torch._subclasses.meta_utils
from torch import Tensor
from torch._dynamo.testing import rand_strided
from torch._inductor.cpp_builder import normalize_path_separator
from torch._prims_common import is_float_dtype
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._config_module import ConfigModule
from torch.utils._content_store import ContentStoreReader, ContentStoreWriter

from . import config
from .utils import clone_inputs, get_debug_dir, warn_once


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.hub import tqdm
    from torch.storage import UntypedStorage


log = logging.getLogger(__name__)

T = TypeVar("T")


inductor_config = import_module("torch._inductor.config")
use_buck = inductor_config.is_fbcode()

if use_buck:
    import libfb.py.build_info


# pyrefly: ignore [implicit-any]
extra_deps = []
extra_imports = ""
cur_target = ""
if use_buck:
    extra_deps = [
        "//caffe2/torch/fb/sparsenn:sparsenn_operators_gpu",
        "//caffe2/torch/fb/sparsenn:sparsenn_operators",
        "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu",
        "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops",
    ]
    cur_target = libfb.py.build_info.BuildInfo.get_build_rule().replace("fbcode:", "//")  # type: ignore[possibly-undefined]
    # Preload common fbcode custom-op libraries so repros that use those ops
    # work out of the box. Best-effort: a repro whose graph doesn't use these
    # ops (or that is run outside a buck target linking them) must not fail
    # just because the library isn't present.
    _extra_deps_list = "\n".join(f'    "{x}",' for x in extra_deps)
    extra_imports = (
        f"for _extra_dep in [\n{_extra_deps_list}\n]:\n"
        "    try:\n"
        "        torch.ops.load_library(_extra_dep)\n"
        "    except OSError:\n"
        "        pass\n"
    )


BUCK_CMD_PREFIX = ["buck2", "run", "@mode/dev-nosan"]


class BuckTargetWriter:
    def __init__(self, filename: str) -> None:
        self.subdir, self.py_file = os.path.split(os.path.abspath(filename))
        self.target = self.py_file.replace(".py", "")

        # Get main_module path from fbcode
        self.path = f"{self.subdir.replace('/', '.')}.{self.target}"
        self.path = self.path[self.path.find("fbcode.") :]
        self.path = self.path[7:]

        # Get cmd line path
        tmp = self.subdir
        tmp = tmp[tmp.find("fbcode/") :][7:]
        self.cmd_line_path = f"//{tmp}:{self.target}"

    def build(self) -> str:
        extra_cpp_deps = "\n".join([f'        "{x}",' for x in extra_deps])
        return textwrap.dedent(
            f"""
load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")

python_binary(
    name="{self.target}",
    srcs = ["{self.py_file}"],
    compile = False,
    deps = [
        "//caffe2:torch",
        "//caffe2:libtorch",
        "//caffe2/functorch:functorch",
        "//triton:triton",
        "{cur_target}",
    ],
    cpp_deps = [
{extra_cpp_deps}
    ],
    main_module = "{self.path}",
    par_style = "xar",
)
"""
        )

    def write(self, print_msg: bool = True) -> list[str]:
        target_file = os.path.join(self.subdir, "TARGETS")
        with open(target_file, "w") as fd:
            fd.write(self.build())
        # log.warning("Wrote isolation TARGETS file at %s", target_file)
        cmd_split = BUCK_CMD_PREFIX + [self.cmd_line_path]
        if print_msg:
            log.warning(
                "Found an example that reproduces the error. Run this cmd to repro - %s",
                " ".join(cmd_split),
            )
        return cmd_split


def minifier_dir() -> str:
    path = os.path.join(get_debug_dir(), "minifier")
    if path is None:
        path = f"{tempfile.gettempdir()}/minifier_{getpass.getuser()}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


MAX_CONSTANT_NUMEL_INLINE = 4


class UnsupportedNNModuleError(AssertionError):
    pass


class NNModuleToString:
    fake_quant_modules = {
        "torch.ao.quantization.fake_quantize.FakeQuantize",
        "torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize",
    }

    qat_conv_modules = {
        "torch.ao.nn.qat.modules.conv.Conv1d",
        "torch.ao.nn.qat.modules.conv.Conv2d",
        "torch.ao.nn.qat.modules.conv.Conv3d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn1d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn2d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn3d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU1d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU3d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU1d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU3d",
    }

    qat_linear_modules = {
        "torch.ao.nn.qat.modules.linear.Linear",
        "torch.ao.nn.intrinsic.qat.modules.linear_relu.LinearReLU",
    }

    # x86 currently aliases fbgemm in repr, so emit fbgemm as the canonical
    # constructor for both until those default qconfigs diverge.
    default_qat_qconfig_backends = ("fbgemm", "x86", "qnnpack", "onednn")

    safe_reprs = [
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.Softmax,
        torch.nn.ReLU,
        torch.nn.GELU,
        torch.nn.Identity,
        torch.nn.MaxPool2d,
        torch.nn.Embedding,
        torch.nn.Tanh,
        torch.nn.ConvTranspose1d,
        torch.nn.GLU,
        torch.nn.LSTM,
        torch.nn.Flatten,
        torch.nn.AdaptiveAvgPool2d,
    ]

    @staticmethod
    def can_convert_to_string(gm: torch.fx.GraphModule) -> bool:
        cant_convert = set()
        for _, module in gm.named_children():
            # Nested GraphModules (e.g. torch.cond / torch.while_loop subgraphs)
            # are serialized recursively by convert().
            if isinstance(module, torch.fx.GraphModule):
                continue
            if NNModuleToString._module_constructor_string(module) is None:
                cant_convert.add(module)

        if len(cant_convert) > 0:
            log.warning(
                "Cannot safely convert some modules to strings - %s", cant_convert
            )
            return False
        return True

    @staticmethod
    def _module_type_name(module: torch.nn.Module) -> str:
        return f"{type(module).__module__}.{type(module).__name__}"

    @staticmethod
    def _type_name(type_: type[object]) -> str:
        return f"{type_.__module__}.{type_.__name__}"

    @staticmethod
    def _fake_quant_constructor(module: torch.nn.Module) -> str | None:
        module_type = NNModuleToString._module_type_name(module)
        if module_type not in NNModuleToString.fake_quant_modules:
            return None

        observer = module.activation_post_process  # type: ignore[attr-defined]
        quant_min = cast(int, module.quant_min)  # type: ignore[attr-defined]
        quant_max = cast(int, module.quant_max)  # type: ignore[attr-defined]
        if getattr(observer, "reduce_range", False):
            quant_min *= 2
            quant_max = quant_max * 2 + 1
        args = [
            f"observer={NNModuleToString._type_name(type(observer))}",
            f"quant_min={quant_min!r}",
            f"quant_max={quant_max!r}",
            f"dtype={module.dtype!r}",  # type: ignore[attr-defined]
            f"qscheme={module.qscheme!r}",  # type: ignore[attr-defined]
            f"is_dynamic={module.is_dynamic!r}",  # type: ignore[attr-defined]
        ]
        if hasattr(observer, "reduce_range"):
            args.append(f"reduce_range={observer.reduce_range!r}")
        if hasattr(observer, "ch_axis"):
            args.append(f"ch_axis={observer.ch_axis!r}")
        return f"{module_type}({', '.join(args)})"

    @staticmethod
    def _qat_qconfig_constructor(module: torch.nn.Module) -> str | None:
        qconfig = getattr(module, "qconfig", None)
        for backend in NNModuleToString.default_qat_qconfig_backends:
            try:
                default_qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
            except AssertionError:
                continue
            if repr(qconfig) == repr(default_qconfig):
                return (
                    "qconfig=torch.ao.quantization.get_default_qat_qconfig"
                    f"({backend!r})"
                )
        return None

    @staticmethod
    def _qat_conv_constructor(module: torch.nn.Module) -> str | None:
        module_type = NNModuleToString._module_type_name(module)
        if module_type not in NNModuleToString.qat_conv_modules:
            return None
        qconfig = NNModuleToString._qat_qconfig_constructor(module)
        if qconfig is None:
            return None

        args = [
            repr(module.in_channels),  # type: ignore[attr-defined]
            repr(module.out_channels),  # type: ignore[attr-defined]
            f"kernel_size={module.kernel_size!r}",  # type: ignore[attr-defined]
            f"stride={module.stride!r}",  # type: ignore[attr-defined]
            f"padding={module.padding!r}",  # type: ignore[attr-defined]
            f"dilation={module.dilation!r}",  # type: ignore[attr-defined]
            f"groups={module.groups!r}",  # type: ignore[attr-defined]
            f"bias={module.bias is not None}",  # type: ignore[attr-defined]
            f"padding_mode={module.padding_mode!r}",  # type: ignore[attr-defined]
        ]
        bn = getattr(module, "bn", None)
        if bn is not None:
            args.extend(
                [
                    f"eps={bn.eps!r}",
                    f"momentum={bn.momentum!r}",
                    f"freeze_bn={module.freeze_bn!r}",  # type: ignore[attr-defined]
                ]
            )
        args.append(qconfig)
        return f"{module_type}({', '.join(args)})"

    @staticmethod
    def _qat_linear_constructor(module: torch.nn.Module) -> str | None:
        module_type = NNModuleToString._module_type_name(module)
        if module_type not in NNModuleToString.qat_linear_modules:
            return None
        qconfig = NNModuleToString._qat_qconfig_constructor(module)
        if qconfig is None:
            return None

        args = [
            repr(module.in_features),  # type: ignore[attr-defined]
            repr(module.out_features),  # type: ignore[attr-defined]
            f"bias={module.bias is not None}",  # type: ignore[attr-defined]
            qconfig,
        ]
        return f"{module_type}({', '.join(args)})"

    @staticmethod
    def _can_emit_module_constructor(module: torch.nn.Module, module_str: str) -> bool:
        if type(module) not in NNModuleToString.safe_reprs:
            return False

        try:
            compile(f"mod = {module_str}", "<module repr>", "exec")
        except SyntaxError:
            return False
        return True

    @staticmethod
    def _module_constructor_string(
        module: torch.nn.Module, *, allow_unsafe_repr: bool = False
    ) -> str | None:
        module_str = repr(module)
        if NNModuleToString._can_emit_module_constructor(module, module_str):
            return module_str
        constructor_str = (
            NNModuleToString._fake_quant_constructor(module)
            or NNModuleToString._qat_conv_constructor(module)
            or NNModuleToString._qat_linear_constructor(module)
        )
        if constructor_str is not None:
            return constructor_str
        if allow_unsafe_repr:
            return module_str
        return None

    @staticmethod
    def convert(gm: torch.fx.GraphModule, *, allow_unsafe_repr: bool = False) -> str:
        from torch.nn.modules.module import _addindent

        tab = " " * 4

        # Higher-order ops (torch.cond, torch.while_loop, ...) attach their
        # subgraphs as nested GraphModule children. These have no
        # constructor-style repr, so serialize each nested GraphModule as its
        # own top-level class (defined before Repro so it is in scope) and
        # instantiate it in __init__. Leaf nn.Modules use the constructor-string
        # path below.
        subgraph_classes: dict[int, str] = {}
        subgraph_defs: list[str] = []

        def register_subgraph(name_hint: str, sub: torch.fx.GraphModule) -> str:
            # Reserve the name before recursing so nested/shared subgraphs are
            # deduplicated by identity and cannot recurse infinitely.
            if id(sub) in subgraph_classes:
                return subgraph_classes[id(sub)]
            class_name = f"{name_hint}_{len(subgraph_classes)}"
            subgraph_classes[id(sub)] = class_name
            body = emit_module_body(sub)
            subgraph_defs.append(f"class {class_name}(torch.nn.Module):\n{body}")
            return class_name

        def emit_module_body(module: torch.nn.Module) -> str:
            model_str = (
                f"{tab}def __init__(self) -> None:\n{tab * 2}super().__init__()\n"
            )

            for module_name, child in module.named_children():
                if isinstance(child, torch.fx.GraphModule):
                    class_name = register_subgraph(module_name, child)
                    model_str += f"{tab * 2}self.{module_name} = {class_name}()\n"
                    continue
                module_str = NNModuleToString._module_constructor_string(
                    child, allow_unsafe_repr=allow_unsafe_repr
                )
                if module_str is None:
                    raise UnsupportedNNModuleError(
                        f"Cannot convert module to string: {child!r}"
                    )
                # module should be a core torch.nn.Module, so all parameters
                # and buffers should be on the same device.
                example_param = next(child.parameters(), None)
                example_tensor = (
                    example_param
                    if example_param is not None
                    else next(child.buffers(), None)
                )
                if example_tensor is not None and example_tensor.device.type != "cpu":
                    module_str = f'{module_str}.to("{example_tensor.device}")'
                model_str += f"{tab * 2}self.{module_name} = {module_str}\n"

            for buffer_name, buffer in module._buffers.items():
                if buffer is None:
                    continue
                # Serialize full data for small buffers
                if buffer.numel() <= MAX_CONSTANT_NUMEL_INLINE:
                    from torch._tensor_str import PRINT_OPTS

                    if PRINT_OPTS.threshold < MAX_CONSTANT_NUMEL_INLINE:
                        raise AssertionError(
                            f"PRINT_OPTS.threshold ({PRINT_OPTS.threshold}) must be >= "
                            f"MAX_CONSTANT_NUMEL_INLINE ({MAX_CONSTANT_NUMEL_INLINE})"
                        )
                    tensor_str = repr(buffer)
                elif torch.is_floating_point(buffer):
                    tensor_str = (
                        f"torch.randn({list(buffer.shape)}, dtype={buffer.dtype})"
                    )
                else:
                    tensor_str = f"torch.randint(1, size={list(buffer.shape)}, dtype={buffer.dtype})"
                if buffer.device.type != "cpu":
                    tensor_str = f'{tensor_str}.to("{buffer.device}")'
                model_str += (
                    f"{tab * 2}self.register_buffer('{buffer_name}', {tensor_str})\n"
                )

            for param_name, param in module._parameters.items():
                if param is None:
                    continue
                maybe_device = ""
                if param.device.type != "cpu":
                    maybe_device = f', device="{param.device}"'
                tensor_str = f"torch.nn.Parameter(torch.randn({list(param.shape)}, dtype={param.dtype}{maybe_device}))"
                model_str += f"{tab * 2}self.{param_name} = {tensor_str}\n"

            model_str += f"{_addindent(module.code, len(tab))}\n"
            return model_str

        repro_class = f"class Repro(torch.nn.Module):\n{emit_module_body(gm)}"
        # Subgraph classes must be defined before Repro instantiates them.
        return "\nfrom torch.nn import *\n" + "\n".join(subgraph_defs + [repro_class])


@functools.cache  # subprocess is expensive
def _cuda_system_info_comment() -> str:
    if not torch.cuda.is_available():
        return "# torch.cuda.is_available()==False, no GPU info collected\n"

    model_str = "# CUDA Info: \n"
    try:
        if torch.version.hip is None:
            cuda_version_out = subprocess.check_output(["nvcc", "--version"])
            cuda_version_lines = cuda_version_out.decode().split("\n")
            comment = "".join([f"# {s} \n" for s in cuda_version_lines if s != ""])
            model_str += f"{comment}\n"
        else:
            model_str += "# Not searching for nvcc on ROCM setup\n"
    except (OSError, subprocess.CalledProcessError):
        model_str += "# nvcc not found\n"

    gpu_names = Counter(
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    )

    model_str += "# GPU Hardware Info: \n"
    for name, count in gpu_names.items():
        model_str += f"# {name} : {count} \n"
    model_str += "\n"
    return model_str


def generate_env_vars_string(*, stable_output: bool = False) -> str:
    """
    Generate a string configuration for environment variables related to Dynamo, Inductor, and Triton.
    """
    if stable_output:
        return "# env var omitted due to stable_output=True"

    allow_list = ["TORCH", "DYNAMO", "INDUCTOR", "TRITON"]
    skip_list = ["TRITON_LIBDEVICE_PATH", "TRITON_PTXAS_PATH", "TRITON_LIBCUDA_PATH"]
    repro_env_vars = ["TORCHDYNAMO_REPRO_AFTER", "TORCHDYNAMO_REPRO_LEVEL"]

    def filter(key: str) -> bool:
        return (
            any(string in key for string in allow_list)
            and key not in skip_list
            and key not in repro_env_vars
        )

    config_lines = [
        f"""os.environ['{key}'] = '{value.replace("'", '"')}'"""
        for key, value in os.environ.items()
        if filter(key)
    ]
    # Repro scripts should not recursively generate more repro scripts when run.
    config_lines.extend(f"os.environ.pop('{key}', None)" for key in repro_env_vars)
    config_string = "\n".join(config_lines)
    return normalize_path_separator(f"""\
import os
{config_string}
    """)


def generate_config_string(*, stable_output: bool = False) -> str:
    import torch._functorch.config
    import torch._inductor.config
    from torch._inductor.codegen import common

    if stable_output:
        return "# config omitted due to stable_output=True"

    # Third-party Inductor backends can register their own ConfigModule.
    # Repros need to replay non-default values from those modules, and
    # ConfigModule.codegen_config() emits assignments but not the module import.
    extra_codegen_configs = []
    for c in common.custom_backend_codegen_configs.values():
        if isinstance(c, ConfigModule):
            codegen_config = c.codegen_config()
            if codegen_config:
                extra_codegen_configs.append(f"import {c.__name__}\n{codegen_config}")
    extra_codegen_configs_str = "\n".join(extra_codegen_configs)

    experimental_config = torch.fx.experimental._config.codegen_config()  # type: ignore[attr-defined]

    return f"""\
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
{torch._dynamo.config.codegen_config()}
{torch._inductor.config.codegen_config()}
{torch._functorch.config.codegen_config()}
{experimental_config}
{extra_codegen_configs_str}
"""


def get_minifier_repro_path() -> str:
    return os.path.join(minifier_dir(), "minifier_launcher.py")


def helper_for_dump_minify(contents: str) -> None:
    minified_repro_path = get_minifier_repro_path()
    log.warning("Writing minified repro to:\n%s", minified_repro_path)

    if use_buck:
        BuckTargetWriter(minified_repro_path).write()
    try:
        with open(minified_repro_path, "w") as fd:
            fd.write(contents)

    except OSError as e:
        log.exception("")
        raise NotImplementedError(f"Could not write to {minified_repro_path}") from e


class AccuracyError(Exception):
    pass


def clone_inputs_retaining_gradness(example_inputs: Sequence[Any]) -> list[Any]:
    """
    This clone inputs is different from utils clone_input. In case of minifier,
    all the tensors are leaf tensors while creating a new graph. So, we set the
    requires_grad field w/o checking the leafness of the tensor.
    """
    cloned_inputs = clone_inputs(example_inputs)
    for idx in range(len(example_inputs)):
        if isinstance(cloned_inputs[idx], torch.Tensor):
            cloned_inputs[idx].requires_grad_(example_inputs[idx].requires_grad)
    return cloned_inputs  # type: ignore[return-value]


def run_fwd_maybe_bwd(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    only_fwd: bool = False,
    disable_clone: bool = False,
) -> Any:
    """
    Runs a forward and possibly backward iteration for a given mod and args.

    When disable_clone is True, we will use args as-is without cloning.
    This is higher fidelity but we may destroy the args in the process.
    """
    from .testing import collect_results, reduce_to_scalar_loss, requires_bwd_pass

    gm = copy.deepcopy(gm)
    if not disable_clone:
        args = clone_inputs_retaining_gradness(args)

    if hasattr(gm, "zero_grad"):
        gm.zero_grad(True)

    # TorchInductor returned callable expects lists. So, may need a boxed calling convention.
    out = gm(args) if getattr(gm, "_boxed_call", False) else gm(*args)

    if only_fwd:
        return out
    if requires_bwd_pass(out):
        loss = reduce_to_scalar_loss(out)
        loss.backward()
    return collect_results(gm, out, None, args)


def same_two_models(
    gm: torch.fx.GraphModule,
    opt_gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    only_fwd: bool = False,
    *,
    require_fp64: bool = False,
    ignore_non_fp: bool = False,
) -> bool:
    """
    Check two models have same accuracy.

    require_fp64: if True, raise an error if we unable to calculate the fp64 reference
    ignore_non_fp: if True, do not compare outputs which are not floating point.  This
        is mostly useful for the minifier (which wants to avoid quantizing floating point
        error into integer/boolean error)
    """
    from .utils import same

    ref = run_fwd_maybe_bwd(gm, example_inputs, only_fwd)

    fp64_ref = None
    if config.same_two_models_use_fp64:
        try:
            fp64_model, fp64_examples = cast_to_fp64(
                copy.deepcopy(gm), clone_inputs_retaining_gradness(example_inputs)
            )
            fp64_ref = run_fwd_maybe_bwd(fp64_model, fp64_examples, only_fwd)
        except Exception:
            if require_fp64:
                raise RuntimeError(  # noqa: B904
                    "Could not generate fp64 outputs, workaround with torch._dynamo.config.same_two_models_use_fp64 = False"
                )
            log.warning("Could not generate fp64 outputs")

    try:
        res = run_fwd_maybe_bwd(opt_gm, example_inputs, only_fwd)
    except Exception:
        # This means that the minified graph is bad/exposes a different problem.
        # As we are checking accuracy here, lets log the exception and return True.
        log.exception(
            "While minifying the program in accuracy minification mode, "
            "ran into a runtime exception which is likely an unrelated issue."
            " Skipping this graph."
        )
        return True

    passing = same(
        ref,
        res,
        fp64_ref,
        tol=config.repro_tolerance,
        equal_nan=True,
        ignore_non_fp=ignore_non_fp,
    )
    return passing


def cast_dtype_args_to_fp64(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for node in model.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.prims.convert_element_type.default
        ):
            if len(node.args) != 2:
                raise AssertionError(
                    f"Expected node to have 2 args, got {len(node.args)}"
                )
            if is_float_dtype(node.args[1]) and node.args[1] != torch.float64:
                node.args = (node.args[0], torch.float64)
        if node.op == "call_function":
            dtype = node.kwargs.get("dtype")
            if dtype is not None and is_float_dtype(dtype):
                new_kwargs = dict(node.kwargs)
                new_kwargs["dtype"] = torch.float64
                node.kwargs = new_kwargs

    model.graph.lint()
    model.recompile()
    return model


def cast_to(
    dtype: torch.dtype, model: torch.fx.GraphModule, inputs: list[Any]
) -> tuple[torch.fx.GraphModule, list[Any]]:
    from torch.utils._pytree import tree_map

    model = model.to(dtype)
    if dtype == torch.float64:
        # If casting to fp64 for accuracy comparison, we need to
        # replace dtype arguments embedded in the graph with fp64
        model = cast_dtype_args_to_fp64(model)

    inputs = tree_map(
        lambda x: x.to(dtype)
        if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x,
        inputs,
    )
    return model, inputs


def cast_to_fp64(
    model: torch.fx.GraphModule, inputs: list[Any]
) -> tuple[torch.fx.GraphModule, list[Any]]:
    return cast_to(torch.float64, model, inputs)


def backend_accuracy_fails(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    compiler_fn: Callable[[torch.fx.GraphModule, list[Any]], torch.fx.GraphModule],
    only_fwd: bool = False,
    *,
    require_fp64: bool = False,
    ignore_non_fp: bool = False,
) -> bool:
    try:
        compiled_gm = compiler_fn(
            copy.deepcopy(gm), clone_inputs_retaining_gradness(example_inputs)
        )
        return not same_two_models(
            gm,
            compiled_gm,
            example_inputs,
            only_fwd,
            require_fp64=require_fp64,
            ignore_non_fp=ignore_non_fp,
        )
    except Exception:
        # This means that the minified graph is bad/exposes a different problem.
        # As we are checking accuracy here, lets log the exception and return False.
        log.exception(
            "While minifying the program in accuracy minification mode, "
            "ran into a runtime exception which is likely an unrelated issue."
            " Skipping this graph"
        )
        return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       REPRO SUPPORT CODE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# Helper functions for computing what the default values of tensor
# values should be.  These all coincide with factory functions, e.g., torch.empty


def _stride_or_default(
    stride: torch._prims_common.StrideType | None,
    *,
    shape: torch._prims_common.ShapeType,
) -> torch._prims_common.StrideType:
    return stride if stride is not None else utils.make_contiguous_strides_for(shape)


def _mk_defaulter(d: T) -> Callable[[T | None], T]:
    return lambda x: x if x is not None else d


_dtype_or_default = _mk_defaulter(torch.float32)
_device_or_default = _mk_defaulter(torch.device("cpu"))
_storage_offset_or_default = _mk_defaulter(0)
_requires_grad_or_default = _mk_defaulter(False)
_is_leaf_or_default = _mk_defaulter(False)


class NopInputReader:
    def __init__(self) -> None:
        self.total = 0

    def storage(
        self,
        storage_hash: str | None,
        nbytes: int,
        *,
        device: torch._prims_common.DeviceLikeType | None = None,
        dtype_hint: torch.dtype | None = None,
    ) -> None:
        self.total += 1

    def tensor(self, *args: Any, **kwargs: Any) -> torch.Tensor | None:
        pass

    def symint(self, *args: Any, **kwargs: Any) -> int | None:
        pass

    def const(self, name: str) -> None:
        pass

    def unsupported(self, name: str) -> None:
        pass

    def generator(self, device_type: str, device_index: int) -> None:
        pass

    def opaque(self, script_class_name: str) -> None:
        self.total += 1


# TODO: Support bundling the entire repro into a zip file for ease of
# transferring around
class InputReader:
    def __init__(
        self, save_dir: str | None = None, *, pbar: tqdm | None = None
    ) -> None:
        # If None, we will generate random data instead.  It's important
        # to natively support this use case as it will allow people to
        # share repros without including the real data, if the problem
        # reproduces even on random data.
        if save_dir is None:
            log.warning("no save_dir specified, will generate random data")
        self.store = ContentStoreReader(save_dir) if save_dir is not None else None
        self.args: list[Any] = []
        self.pbar = pbar

    def storage(
        self,
        storage_hash: str | None,
        nbytes: int,
        *,
        device: torch._prims_common.DeviceLikeType | None = None,
        dtype_hint: torch.dtype | None = None,
    ) -> UntypedStorage:
        if self.pbar is not None:
            self.pbar.update(1)
        device = _device_or_default(device)  # type: ignore[arg-type]
        dtype_hint = _dtype_or_default(dtype_hint)
        if self.store is not None and storage_hash is not None:
            try:
                storage = self.store.read_storage(storage_hash)
            except FileNotFoundError:
                pass
            else:
                if device != storage.device:
                    log.warning("device mismatch: %s != %s", device, storage.device)
                    # TODO: transfer it to the right device?  But failing this
                    # way would be very mysterious!  Would have been better
                    # not to store device in the serialized format...
                return storage
        warn_once(f"could not load {storage_hash}, generating random data instead")
        shape = (nbytes // dtype_hint.itemsize,)
        stride = _stride_or_default(None, shape=shape)
        return rand_strided(shape, stride, dtype_hint, device).untyped_storage()

    def tensor(
        self,
        storage: UntypedStorage,
        shape: torch._prims_common.ShapeType,
        stride: torch._prims_common.StrideType | None = None,
        *,
        storage_offset: int | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool | None = None,
        is_leaf: bool | None = None,
        **metadata: Any,
    ) -> torch.Tensor:
        stride = _stride_or_default(stride, shape=shape)
        storage_offset = _storage_offset_or_default(storage_offset)
        dtype = _dtype_or_default(dtype)
        is_leaf = _is_leaf_or_default(is_leaf)
        requires_grad = _requires_grad_or_default(requires_grad)
        t = torch.tensor(
            [], dtype=dtype, device=storage.device, requires_grad=requires_grad
        )
        with torch.no_grad():
            t.set_(storage, storage_offset, shape, stride)
        if not is_leaf:
            # Fake up some autograd history in a very naughty way
            with torch.enable_grad():
                t = t.clone(memory_format=torch.preserve_format)
            with torch.no_grad():
                t.set_(storage, storage_offset, shape, stride)
        if torch._subclasses.meta_utils.safe_is_leaf(t) != is_leaf:
            raise AssertionError(
                f"Tensor leaf status mismatch: safe_is_leaf(t) = "
                f"{torch._subclasses.meta_utils.safe_is_leaf(t)}, expected {is_leaf}"
            )
        torch._utils.set_tensor_metadata(t, metadata)
        self.args.append(t)
        return t  # for BC

    def symint(self, val: Any, *, expr: str | None = None) -> Any:
        self.args.append(val)
        if expr is not None:
            if not hasattr(self, "symint_exprs"):
                self.symint_exprs: dict[int, str] = {}
            self.symint_exprs[len(self.args) - 1] = expr
        return val  # for BC

    def const(self, name: str) -> None:
        self.args.append(None)

    def unsupported(self, name: str) -> None:
        self.args.append(None)

    def generator(self, device_type: str, device_index: int) -> Any:
        device = torch.device(device_type, device_index)
        gen = torch.Generator(device=device)
        self.args.append(gen)
        return gen

    def opaque(self, script_class_name: str) -> None:
        self.args.append(None)


# Here is our writer strategy:
#  1. We will stream all of the inputs to disk
#  2. You can now deterministically randomize the inputs, or reload
#     the inputs from disk
#  3. You can YOLO run the script without the inputs, in which case
#     we'll fill the inputs with random data and pray.  This is the
#     legacy behavior, but it's also useful if you want to find out
#     if we're so broken even random inputs trigger it
#  4. We could offer an in process "check if the randomized thing
#     works too" but this is delicate so we don't do it


def _serialize_storage_nbytes(nbytes: int | torch.SymInt) -> str:
    if isinstance(nbytes, torch.SymInt):
        from torch.fx.experimental.symbolic_shapes import SymExprPrinter

        return SymExprPrinter().doprint(nbytes.node.expr)
    return repr(nbytes)


class InputWriter:
    def __init__(self, save_dir: str | None, *, stable_hash: bool = False) -> None:
        self._lines: list[str] = []
        # TODO: consider ensuring tensor and storage counters line up?
        self.storage_counter = itertools.count()
        self.save_dir = save_dir
        self.store = (
            ContentStoreWriter(save_dir, stable_hash=stable_hash)
            if save_dir is not None
            else None
        )
        self.seen_storages: dict[StorageWeakRef, str] = {}

    def lines(self) -> list[str]:
        r = [
            "def load_args(reader):",
        ]
        r.extend(f"    {l}" for l in self._lines)
        # In case we need to change the internal format of load_args
        # in an FC-breaking way
        r.append("load_args._version = 0")
        return r

    # Storages are untyped, but we need to initialize them with data if
    # we don't have the real data, so we give a hint saying what kind
    # of initialization may be appropriate
    #
    # If we had a FakeTensor, device_hint tells us what device should be
    def storage(
        self,
        untyped_storage: UntypedStorage,
        *,
        device_hint: torch._prims_common.DeviceLikeType | None = None,
        dtype_hint: torch.dtype | None = None,
    ) -> str:
        ws = StorageWeakRef(untyped_storage)
        v = self.seen_storages.get(ws)
        if v is not None:
            return v
        v = f"buf{next(self.storage_counter)}"
        maybe_dtype_hint = ""
        if _dtype_or_default(None) != _dtype_or_default(dtype_hint):
            maybe_dtype_hint = f", dtype_hint={dtype_hint!r}"
        # TODO: being optional on device is kind of pointless as the default
        # is CPU but most repros we care about are CUDA
        maybe_device = ""
        device = untyped_storage.device
        if device.type == "meta":
            if device_hint is None:
                raise AssertionError(
                    "device_hint must be provided when storage device is 'meta'"
                )
            device = device_hint  # type: ignore[assignment]
        if _device_or_default(None) != device:
            maybe_device = f", device={device!r}"
        nbytes = untyped_storage.nbytes()
        nbytes_source = _serialize_storage_nbytes(nbytes)
        storage_hash = None
        if self.store is not None and untyped_storage.device.type != "meta":
            storage_hash = self.store.write_storage(untyped_storage)
        self._lines.append(
            f"{v} = reader.storage({storage_hash!r}, {nbytes_source}{maybe_device}{maybe_dtype_hint})"
        )
        self.seen_storages[ws] = v
        return v

    def tensor(self, name: str, t: torch.Tensor) -> None:
        from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

        storage = self.storage(
            t.untyped_storage(), dtype_hint=t.dtype, device_hint=t.device
        )
        args = []
        # NB: this is positional, must come first
        if not statically_known_true(
            sym_eq(_stride_or_default(None, shape=t.shape), t.stride())
        ):
            args.append(str(tuple(t.stride())))
        if _dtype_or_default(None) != t.dtype:
            args.append(f"dtype={t.dtype!r}")
        if not statically_known_true(
            _storage_offset_or_default(None) == t.storage_offset()
        ):
            args.append(f"storage_offset={t.storage_offset()!r}")
        tensor_metadata = torch._utils.get_tensor_metadata(t)
        if tensor_metadata:
            args.extend(f"{k}={v!r}" for k, v in tensor_metadata.items())
        if _requires_grad_or_default(None) != t.requires_grad:
            args.append(f"requires_grad={t.requires_grad!r}")
        is_leaf = torch._subclasses.meta_utils.safe_is_leaf(t)
        if _is_leaf_or_default(None) != is_leaf:
            args.append(f"is_leaf={is_leaf!r}")
        self._lines.append(
            "reader.tensor("
            + ", ".join([storage, str(tuple(t.shape)), *args])
            + f")  # {name}"
        )

    def unsupported(self, name: str, arg: Any) -> None:
        # NB: Try hard not to /print/ a tensor, that will be very slow
        self._lines.append(
            f"reader.unsupported({name!r})  # unsupported type for dumping: {type(arg)}"
        )
        # Best effort dump as much useful stuff we can lol, in case you want
        # to repair the repro
        if isinstance(arg, (list, tuple)):
            self._lines.append('"""')
            for i, a in enumerate(arg):
                name_i = f"{name}[{i}]"
                if isinstance(a, torch.Tensor):
                    self.tensor(name_i, a)
                elif isinstance(a, (int, torch.SymInt)):
                    self.symint(name_i, a)
                else:
                    self.unsupported(name_i, a)
            self._lines.append('"""')

    # write out that the arg was filtered out as it is constant
    def const(self, name: str) -> None:
        self._lines.append(
            f"reader.const({name!r})  # {name}, filtered out during compilation"
        )

    # TODO: this doesn't actually symint atm
    def symint(self, name: str, val: Any) -> None:
        if isinstance(val, torch.SymInt):
            expr_str = str(val.node.expr)
            hint = val.node.hint
            self._lines.append(f"reader.symint({hint!r}, expr={expr_str!r})  # {name}")
        else:
            self._lines.append(f"reader.symint({val!r})  # {name}")

    def generator(self, name: str, arg: torch._C.Generator) -> None:
        device = arg.device
        self._lines.append(
            f"reader.generator({device.type!r}, {device.index!r})  # {name}"
        )

    def opaque(self, name: str, script_class_name: str) -> None:
        self._lines.append(f"reader.opaque({script_class_name!r})  # {name}")


def aot_graph_input_parser(
    func: Callable[[list[Tensor]], list[Tensor]],
    device: str = "cuda",
    sym_shapes: dict[str, int] | None = None,
    default_sym_shape: int | None = None,
) -> dict[str, Any]:
    """
    Takes in a function which has been printed with print_readable() and constructs kwargs to run it.

    Handles Tensor inputs, Symints, and a graph module which might have tensor constants.

    Consider a function `forward` defined as follows:

    def forward(self, primals_1: "f32[1001, 6]", primals_2: "f32[s0]", primals_3: "Sym(s0)",):
        _tensor_constant0: "i64[4190]" = self._tensor_constant0
        # Further implementation

    kwargs = aot_graph_input_parser(forward)
    forward(**kwargs)
    """

    from torch.utils._dtype_abbrs import dtype_abbrs

    dtype_map: dict[str, torch.dtype] = {
        value: key for key, value in dtype_abbrs.items()
    }
    dtype_pattern: str = "|".join(dtype_abbrs.values())

    # Extracting the source code from the function
    source = inspect.getsource(func)

    # Regular expressions
    tensor_assignment_regex = rf"(_tensor_constant\d+): \"({dtype_pattern})\[\s*(.*?)\s*\]\" = self\.(_tensor_constant\d+)"
    tensor_regex = rf"({dtype_pattern})\[\s*(.*?)\s*\]"
    sym_shape_regex = r"Sym\((s\d+)\)"

    class TensorContainer:
        "Container for tensors as attributes"

    # Dictionary for tensors from annotations
    kwargs: dict[str, Any] = {}

    sym_shapes_dict: dict[str, int] = sym_shapes or {}

    def get_sym_int(symint: str) -> int:
        torch._check(
            symint in sym_shapes_dict or default_sym_shape is not None,
            lambda: f"{symint} not in symbolic_shapes and default sym shape not passed in",
        )
        return sym_shapes_dict.get(symint, default_sym_shape)  # type: ignore[return-value]

    def gen_tensor(shape: torch._prims_common.ShapeType, dtype: torch.dtype) -> Tensor:
        # Resolve symbolic shapes to concrete values
        resolved_shape = []
        dynamic_dims = []
        for i, dim in enumerate(shape):
            dim = dim.strip()  # type: ignore[attr-defined]
            if "s" in dim:
                s = get_sym_int(dim)
                resolved_shape.append(s)
                dynamic_dims.append(i)
            else:
                if dim:
                    resolved_shape.append(int(dim))

        constructor = torch.randn if dtype.is_floating_point else torch.zeros
        out = constructor(resolved_shape, dtype=dtype, device=device)  # type: ignore[call-arg]
        for d in dynamic_dims:
            torch._dynamo.mark_dynamic(out, d)
        return out

    # Parse function annotations for tensor generation
    annotations = func.__annotations__
    for param, annotation in annotations.items():
        # Skip 'return' annotation
        if param == "return":
            continue

        match = re.search(tensor_regex, annotation)
        if match:
            data_type, shape_str = match.groups()
            shape = tuple(shape_str.split(","))
            dtype = dtype_map[data_type]
            # pyrefly: ignore [bad-argument-type]
            kwargs[param] = gen_tensor(shape, dtype)

        match = re.search(sym_shape_regex, annotation)
        if match:
            kwargs[param] = get_sym_int(match.group(1))

    if "self" in inspect.signature(func).parameters:
        container = TensorContainer()
        kwargs["self"] = container
        for match in re.finditer(tensor_assignment_regex, source):
            attr_name, data_type, shape_str, _ = match.groups()
            shape = tuple(shape_str.split(","))
            dtype = dtype_map[data_type]
            # pyrefly: ignore [bad-argument-type]
            setattr(container, attr_name, gen_tensor(shape, dtype))

    return kwargs


def profile_to_file(filename: str) -> Callable[[T], T]:
    """
    Decorator to cProfile a given function and save the result to disk on process exit.

    Args:
        filename: filename to save profile to
    """
    prof = cProfile.Profile()
    filename = os.path.abspath(os.path.expanduser(filename))

    def decorator(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            prof.enable()
            try:
                return fn(*args, **kwargs)
            finally:
                prof.disable()

        return wrapper

    def save_it() -> None:
        prof.dump_stats(filename)
        sys.stderr.write(
            textwrap.dedent(
                f"""\
                Wrote profile to {filename}, view with:

                    snakeviz {filename}

                """
            )
        )

    atexit.register(save_it)
    return decorator
<|endoftext|>"""
Provides thread-local scope identification for SubgraphTracer instances.

This module implements a thread-safe mechanism for tracking nested tracing contexts,
which is essential when multiple SubgraphTracer instances are active. The scope ID
helps identify which tracer context is currently active when direct access to the
InstructionTranslator is difficult.

Key components:
- Thread-local scope ID storage (_current_scope_id)
- Getter function (current_scope_id) to safely access the current scope
- Context manager (enter_new_scope) for managing nested scope transitions

The scope ID increments when entering a new context and decrements when exiting,
allowing proper tracking of nested tracing operations across different threads.
"""

import contextlib
import threading
from collections.abc import Generator


# Global variable to identify which SubgraphTracer we are in.
# It is sometimes difficult to find an InstructionTranslator to use.
_current_scope_id = threading.local()


def current_scope_id() -> int:
    global _current_scope_id
    if not hasattr(_current_scope_id, "value"):
        _current_scope_id.value = 1
    return _current_scope_id.value


@contextlib.contextmanager
def enter_new_scope() -> Generator[None, None, None]:
    global _current_scope_id
    try:
        _current_scope_id.value = current_scope_id() + 1
        yield
    finally:
        _current_scope_id.value = current_scope_id() - 1
<|endoftext|>"""
Functions used to test torch._dynamo.dont_skip_tracing.
This file is located in torch/_dynamo so that it is skipped by trace rules.
There is a special rule in trace_rules that doesn't skip this file when
dont_skip_tracing is active.
"""

import torch


def f1(x: torch.Tensor) -> torch.Tensor:
    return x + 1


def f2(x: torch.Tensor) -> torch.Tensor:
    return x + 1


def f3(x: torch.Tensor) -> torch.Tensor:
    return f2(x)


def f4(x: torch.Tensor) -> torch.Tensor:
    x = f5(x, 1)
    x = torch._dynamo.dont_skip_tracing(f6)(x)
    x = f5(x, 8)
    return x


def f5(x: torch.Tensor, n: int) -> torch.Tensor:
    if torch.compiler.is_compiling():
        return x + n
    return x


def f6(x: torch.Tensor) -> torch.Tensor:
    x = f5(x, 2)
    torch._dynamo.graph_break()
    x = f5(x, 4)
    return x
<|endoftext|>"""
This module implements graph deduplication functionality for TorchDynamo's optimization pipeline.
Graph deduplication identifies identical subgraphs in the computational graph and merges them
to reduce redundancy and improve performance. The process involves analyzing regions of the graph,
identifying structurally equivalent regions, and replacing them with a single shared implementation.
This optimization is particularly effective for models with repeated patterns or similar computational
structures across different parts of the network.
"""

import logging
import operator
from collections import defaultdict, deque
from collections.abc import Generator, Iterable

import torch
import torch.fx
from torch._dynamo import config
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._ordered_set import OrderedSet

from .graph_region_tracker import Node, Region
from .graph_utils import _detect_cycles, _get_flat_args, _get_flat_args_unique


# Represents an index into the region
# to select a node and then
# an index into that node's
# flattened arguments
UsageIndex = tuple[int, int]

log = logging.getLogger(__name__)

last_node_to_additional_deps: dict[Node, OrderedSet[Node]] | None = None


def apply_graph_deduplication(output_graph) -> dict[str, torch.fx.GraphModule]:  # type: ignore[no-untyped-def]
    """
    This is the main entry point for applying the graph deduplication pass. \
Deduplication occurs in two phases:
    1. Subgraph creation:
        Subgraph creation works by taking one representative region from each region \
group and creating a subgraph from it, which will then be used to replace all regions \
in the group. This is implemented by first copying all nodes of the region to the new \
subgraph and then finding all inputs which are not within the region and creating placeholders \
for them. For the outputs, all regions in a region group need to be scanned to ensure the \
largest set of outputs is found, and then an output node is created which returns \
a tuple of all outputs.

    2. Graph replacement:
        To replace each region with the extracted subgraph, the node index in the region \
and argument index within the node's flattened args and kwargs are recorded once during \
subgraph creation. This allows us to determine which (external to the region) nodes and \
in which order these nodes are passed as inputs. For the outputs, getitem nodes are created \
for each output, and all nodes in the region with external outputs are replaced by the proper \
getitem node. Finally, all original nodes are erased (there should be no uses of these \
left in the graph).

The deduplication mutates the output_graph argument in place.

Returns a mapping of nodes to their subgraph output replacement node to remap outputs
when they are created in output_graph.
    """

    duplicated_region_groups = output_graph.region_tracker.get_identical_regions(
        output_graph.graph
    )
    node_to_mutated_arg_positions = (
        output_graph.region_tracker.node_to_mutated_arg_positions
    )
    node_to_additional_deps = _populate_additional_deps(
        output_graph.graph, output_graph.region_tracker.node_to_mutated_arg_positions
    )

    sub_gms: dict[str, torch.fx.GraphModule] = {}

    for region_group in duplicated_region_groups:
        inds_with_external_users = _get_all_output_indices(region_group)
        region = region_group[0]
        (
            subgraph,
            external_node_usages,
            node_usage_to_tuple_elems,
            ind_to_tuple_spec,
        ) = _create_subgraph(region, inds_with_external_users)

        # Ignore regions with no args for now, could they possibly be evaluated at compile time?
        if not list(external_node_usages):
            continue

        sub_gm = torch.fx.GraphModule(output_graph.nn_modules, subgraph)
        subgraph_name = output_graph.install_subgraph("subgraph", sub_gm)
        sub_gms[subgraph_name] = sub_gm
        with output_graph.graph.inserting_before():
            get_subgraph_node = output_graph.graph.create_node(
                "get_attr", subgraph_name, (), {}
            )

        for region in region_group:
            _replace_region_with_subgraph(
                output_graph.graph,
                region,
                get_subgraph_node,
                external_node_usages,
                node_usage_to_tuple_elems,
                ind_to_tuple_spec,
                inds_with_external_users,
                subgraph_name,
                node_to_additional_deps,
                node_to_mutated_arg_positions,
            )

    # This is to expose the updated node_to_additional_deps to tests
    global last_node_to_additional_deps
    last_node_to_additional_deps = node_to_additional_deps

    _stable_topological_sort(
        output_graph.graph,
        node_to_additional_deps,
    )
    return sub_gms


def _replace_region_with_subgraph(
    graph: torch.fx.Graph,
    region: Region,
    get_subgraph_node: Node,
    external_node_usages: Iterable[OrderedSet[UsageIndex]],
    node_usage_to_tuple_elems: dict[UsageIndex, OrderedSet[int]],
    ind_to_tuple_spec: dict[int, dict[tuple[int, ...], int]],
    inds_with_external_users: list[int],
    subgraph_name: str,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
    node_to_mutated_arg_positions: dict[Node, OrderedSet[int]],
) -> None:
    sub_args = []
    flattened_getitem_nodes: OrderedSet[Node] = OrderedSet()
    for usages in external_node_usages:
        usage = next(iter(usages))
        node_ind, usage_ind = usage
        node = region[node_ind]
        flattened_args_kwargs = _get_flat_args(node, {})
        for user_ind, node_usage_ind in usages:
            user = region[user_ind]
            if user in node_to_mutated_arg_positions:
                if node_usage_ind in node_to_mutated_arg_positions[user]:
                    log.debug(
                        "NYI: Failed to substitute region %s due to mutation", region
                    )
                    return
        if usage in node_usage_to_tuple_elems:
            tuple_elems = [region[i] for i in node_usage_to_tuple_elems[usage]]
            flattened_getitem_nodes.update(tuple_elems)
            sub_args.extend(tuple_elems)
        else:
            sub_args.append(flattened_args_kwargs[usage_ind])

    # Input/Output aliasing not supported in HOPs today
    # Note: we should use the nodes in the original graph (the region here)
    # because we use the original traced example values for this check
    if _has_aliasing(
        region, sub_args, inds_with_external_users, flattened_getitem_nodes
    ):
        return

    invoke_args = (get_subgraph_node, subgraph_name, *sub_args)

    invoke_subgraph_node = graph.create_node(
        "call_function",
        torch.ops.higher_order.invoke_subgraph,
        invoke_args,  # type: ignore[arg-type]
        {},
    )

    ind = 0
    flattened_output_nodes: OrderedSet[Node] = OrderedSet()
    for external_user_ind in inds_with_external_users:
        node = region[external_user_ind]
        if _is_tuple_node(node):
            tuple_spec = ind_to_tuple_spec[external_user_ind]
            flattened_output_nodes.update(
                _replace_tuple_outputs(
                    node, ind, tuple_spec, invoke_subgraph_node, graph
                )
            )
            ind += len(tuple_spec)
        else:
            subgraph_output = graph.create_node(
                "call_function", operator.getitem, (invoke_subgraph_node, ind), {}
            )
            node.replace_all_uses_with(subgraph_output, propagate_meta=True)
            ind += 1

    # Erase in reverse topological order
    for node in reversed(region):
        if node in flattened_getitem_nodes:
            # Don't erase these, since they will still be used
            continue

        if node not in flattened_output_nodes:
            graph.erase_node(node)

        # Remove any nodes with additional deps
        # This is safe; we've guaranteed that there is
        # no input mutation, so all additional deps
        # will be internal to the subgraph
        node_to_additional_deps.pop(node, None)
        for deps in node_to_additional_deps.values():
            try:
                deps.remove(node)
                deps.add(invoke_subgraph_node)
            except KeyError:
                pass

    if config.graph_deduplication_lint:
        print(_detect_cycles(graph, node_to_additional_deps))
        _stable_topological_sort(graph, node_to_additional_deps)
        graph.lint()


def _get_external_inputs(
    region: Region,
) -> dict[Node, OrderedSet[UsageIndex]]:
    external_node_to_usages = defaultdict[Node, OrderedSet[UsageIndex]](OrderedSet)
    region_unique = set(region)
    for node_ind, node in enumerate(region):
        flattened_args_kwargs = _get_flat_args(node, {})
        for arg_ind, in_node in enumerate(flattened_args_kwargs):
            if isinstance(in_node, Node) and in_node not in region_unique:
                # in_node may occur in multiple nodes' flat_args
                # track this so we can check if the arg is mutated
                # Previously, we only needed to track one occurrence
                # to be able to map that node to a placeholder
                external_node_to_usages[in_node].add((node_ind, arg_ind))

    return external_node_to_usages


def _get_all_output_indices(regions: list[Region]) -> list[int]:
    # Scan all regions to get the set of all possible output nodes indices in the region
    # perhaps we can record this information during region creation for more efficiency?
    inds_with_external_users: set[int] = set()
    for region in regions:
        _get_inds_with_external_users(region, inds_with_external_users)

    return sorted(inds_with_external_users)


def _get_inds_with_external_users(region: Region, inds_unique: set[int]) -> None:
    for ind, node in enumerate(region):
        for user in node.users:
            if user not in region:
                if ind not in inds_unique:
                    inds_unique.add(ind)


def _create_subgraph(
    region: Region,
    inds_with_external_users: list[int],
) -> tuple[
    torch.fx.Graph,
    list[OrderedSet[UsageIndex]],
    dict[UsageIndex, OrderedSet[int]],
    dict[int, dict[tuple[int, ...], int]],
]:
    subgraph: torch.fx.Graph = torch.fx.Graph()
    external_input_to_usages = _get_external_inputs(region)
    external_node_usages = list[OrderedSet[UsageIndex]]()
    region_to_subgraph_node = {}
    flattened_getitem_nodes: OrderedSet[Node] = OrderedSet()
    node_usage_to_tuple_elems: dict[UsageIndex, OrderedSet[int]] = {}

    for node, usage_indices in external_input_to_usages.items():
        # We don't handle tuples as inputs today
        if _is_tuple_node(node):
            # If a node is a tuple we will possibly create multiple placeholders for them
            # and track which nodes we won't copy into the subgraph because they are flattened away
            # Later, when replacing each region with this subgraph, we will create a getitem node
            # externally which will perform the flattening on the outer nodes.
            flattened_node_indices = _get_flattened_node_indices(node, region)
            for ind in flattened_node_indices:
                placeholder = subgraph.placeholder(
                    f"supgraph_input_{node.name}_flattened_{ind}"
                )
                region_to_subgraph_node[region[ind]] = placeholder
                flattened_getitem_nodes.add(region[ind])
            node_usage_to_tuple_elems[next(iter(usage_indices))] = (
                flattened_node_indices
            )
        else:
            placeholder = subgraph.placeholder(f"subgraph_input_{node.name}")
            region_to_subgraph_node[node] = placeholder

        external_node_usages.append(usage_indices)

    def map_arg(node: Node) -> Node:
        if node in region_to_subgraph_node:
            return region_to_subgraph_node[node]
        else:
            return node

    def copy_to_subgraph(node: Node) -> Node:
        subgraph_node = subgraph.node_copy(node, lambda old: map_arg(old))
        region_to_subgraph_node[node] = subgraph_node
        return subgraph_node

    output_list = []
    ind_to_tuple_spec = {}
    for ind, node in enumerate(region):
        if node not in flattened_getitem_nodes:
            subgraph_node = copy_to_subgraph(node)
            if ind in inds_with_external_users:
                # flatten tuple outputs by generating a getitem node tree
                if _is_tuple_node(node):
                    getitem_nodes, ind_to_tuple_spec[ind] = _create_getitem_nodes(
                        node, subgraph_node, subgraph
                    )
                    output_list.extend(getitem_nodes)
                else:
                    output_list.append(subgraph_node)

    subgraph.output(tuple(output_list))

    return subgraph, external_node_usages, node_usage_to_tuple_elems, ind_to_tuple_spec


def _stable_topological_sort_impl(
    graph: torch.fx.Graph,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
    do_sort: bool = True,
    region: OrderedSet[Node] | None = None,
) -> bool:
    # Nodes are in exactly one of these four collections:

    # - Nodes in `pending` are waiting to be processed (in reverse order):
    pending = list(reversed(region or graph.nodes))

    # - Nodes in `ready` have been processed and are already in the correct
    #   order.  When sorting a region, nodes outside the region are
    #   implicitly ready (filtered out in the waiting_for check below).
    ready: set[Node] = set()

    # - `waiting` is a mapping from a dependency to nodes which depend on that
    #   dependency.
    waiting = defaultdict(list)

    # - `outputs` are always at the end of the graph
    outputs = OrderedSet[Node]()

    has_additional_deps = bool(node_to_additional_deps)

    # The cursor indicates the last processed node so we can add new nodes
    # after it.
    cursor = None
    while pending:
        node = pending.pop()

        if node.op == "output":
            outputs.add(node)
            if node.users:
                raise AssertionError("output nodes should have no users")
            continue

        # node._input_nodes is maintained by FX and already contains the
        # unique set of input nodes — avoid rebuilding it via map_arg.
        if has_additional_deps:
            deps = _get_flat_args_unique(node, node_to_additional_deps)
        else:
            deps = node._input_nodes

        last_unready = None
        for x in deps:
            if x not in ready and (region is None or x in region):
                last_unready = x
        if last_unready is not None:
            # We have unprocessed input nodes. Wait for the last unready
            # arg so an already sorted list will only recheck this node once.
            waiting[last_unready].append(node)
        else:
            ready.add(node)
            if cursor and cursor.next is not node and do_sort:
                cursor.append(node)
            cursor = node
            # Mark the nodes that have been waiting for this node to finish as
            # ready to check again.
            pending.extend(reversed(waiting.pop(node, ())))

    ready.update(outputs)
    expected_len = len(region) if region is not None else len(graph.nodes)
    return not waiting and len(ready) == expected_len


def _stable_topological_sort(
    graph: torch.fx.Graph,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
) -> None:
    if not _stable_topological_sort_impl(graph, node_to_additional_deps):
        raise AssertionError("stable topological sort failed")


def _stable_topological_sort_region(
    graph: torch.fx.Graph,
    region: OrderedSet[Node],
) -> None:
    if not _stable_topological_sort_impl(graph, {}, region=region):
        raise AssertionError("stable topological sort of region failed")


def _has_cycle(
    graph: torch.fx.Graph,
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
) -> bool:
    return not _stable_topological_sort_impl(
        graph, node_to_additional_deps, do_sort=False
    )


def _populate_additional_deps(
    graph: torch.fx.Graph, node_to_mutated_arg_positions: dict[Node, OrderedSet[int]]
) -> dict[Node, OrderedSet[Node]]:
    node_to_additional_deps: dict[Node, OrderedSet[Node]] = defaultdict(OrderedSet)
    _add_mutation_dependencies(node_to_mutated_arg_positions, node_to_additional_deps)
    _add_global_state_dependencies(graph, node_to_additional_deps)
    return node_to_additional_deps


def _add_global_state_dependencies(
    graph: torch.fx.Graph, node_to_additional_deps: dict[Node, OrderedSet[Node]]
) -> None:
    import torch.amp

    all_nodes = list(graph.nodes)

    # These are targets of the nodes which need to stay in the same relative place in the graph
    global_state_targets = {torch.amp._enter_autocast, torch.amp._exit_autocast}
    all_nodes_dep_on: list[Node] = []

    def prev_cur_nodes(
        all_nodes: list[Node],
    ) -> Generator[tuple[list[Node], Node], None, None]:
        prev_nodes: list[Node] = []
        next_nodes = list(reversed(all_nodes))

        while next_nodes:
            cur_node = next_nodes.pop()
            yield prev_nodes, cur_node
            prev_nodes.append(cur_node)

    for prev_nodes, cur_node in prev_cur_nodes(all_nodes):
        args_unique = _get_flat_args_unique(cur_node, {})
        new_deps = [n for n in all_nodes_dep_on if n not in args_unique]

        if new_deps:
            additional_deps = node_to_additional_deps[cur_node]
            additional_deps.update(new_deps)

        if cur_node.target in global_state_targets:
            additional_deps = node_to_additional_deps[cur_node]
            additional_deps.update(n for n in prev_nodes if n not in args_unique)
            all_nodes_dep_on.append(cur_node)


def _add_mutation_dependencies(
    node_to_mutated_arg_positions: dict[Node, OrderedSet[int]],
    node_to_additional_deps: dict[Node, OrderedSet[Node]],
) -> None:
    for node, indices in node_to_mutated_arg_positions.items():
        flat_args_kwargs = _get_flat_args(node, {})

        # for all mutated args,
        # add dependency on usages which occur after node to ensure
        # node will always be ordered before them
        # also add node as a dependency on usages which
        # occur before node to ensure node is ordered after them
        for index in indices:
            mutated_arg = flat_args_kwargs[index]
            for user in mutated_arg.users:
                if user is node:
                    continue

                elif user < node:
                    node_to_additional_deps[node].add(user)

                elif user > node:
                    node_to_additional_deps[user].add(node)


def _has_aliasing(
    region: Region,
    inputs: list[Node],
    inds_with_external_users: list[int],
    flattened_getitem_nodes: OrderedSet[Node],
) -> bool:
    input_storages: dict[StorageWeakRef, Node] = dict()
    for node in inputs:
        if node in flattened_getitem_nodes:
            continue
        example_value = node.meta["example_value"]
        if isinstance(example_value, torch.Tensor):
            storage = StorageWeakRef(example_value._typed_storage())
            if storage in input_storages:
                # input-input aliasing
                log.debug(
                    "NYI: Failed to substitute region %s due to input-output aliasing detected at nodes %s, %s",
                    region,
                    input_storages[storage],
                    node,
                )
                return True
            input_storages[storage] = node
    output_storages: dict[StorageWeakRef, Node] = dict()
    for i in inds_with_external_users:
        out_node = region[i]
        if out_node in flattened_getitem_nodes:
            continue
        if out_node:
            example_value = out_node.meta["example_value"]
            if isinstance(example_value, list):
                raise AssertionError("expected example_value to not be a list")
            if isinstance(example_value, torch.Tensor):
                storage = StorageWeakRef(example_value._typed_storage())
                if storage in output_storages:
                    # output-output aliasing
                    log.debug(
                        "NYI: Failed to substitute region %s due to output-output aliasing detected at nodes %s, %s",
                        region,
                        output_storages[storage],
                        out_node,
                    )
                    return True
                output_storages[storage] = out_node
    intersected_storages = input_storages.keys() & output_storages.keys()
    if len(intersected_storages) > 0:
        # input-output aliasing
        aliased = [
            (input_storages[s], output_storages[s]) for s in intersected_storages
        ]
        aliased = ", ".join([f"{i} and {o}" for i, o in aliased])
        log.debug(
            "NYI: Failed to substitute region %s due to input-output aliasing detected at nodes %s",
            region,
            aliased,
        )
        return True
    return False


def _is_tuple_node(node: Node) -> bool:
    return isinstance(node.meta["example_value"], tuple)


def _get_children_getitems(node: Node) -> Generator[Node, None, None]:
    for user in node.users:
        if user.target is operator.getitem and isinstance(user.args[1], int):
            yield user


def _get_flattened_node_indices(node: Node, region: Region) -> OrderedSet[int]:
    """Returns an ordered set of indices, each representing a node in the region which will be flattened"""
    flattened_node_to_ind = {n: i for i, n in enumerate(region)}
    node_indices: OrderedSet[int] = OrderedSet()
    queue = deque(_get_children_getitems(node))
    while queue:
        cur_node = queue.popleft()
        if any(user in region for user in cur_node.users):
            node_indices.add(flattened_node_to_ind[cur_node])
        for child in _get_children_getitems(cur_node):
            queue.append(child)
    return node_indices


def _create_getitem_nodes(
    node: Node, subgraph_tuple_node: Node, subgraph: torch.fx.Graph
) -> tuple[list[Node], dict[tuple[int, ...], int]]:
    tup = node.meta["example_value"]
    if not isinstance(tup, tuple):
        raise AssertionError("_get_getitem_children expects tuple")

    getitem_nodes: list[Node] = []
    queue = deque([(e, (i,), subgraph_tuple_node) for i, e in enumerate(tup)])
    path_to_output_index = {}

    while queue:
        cur_elem, path, parent = queue.popleft()

        with subgraph.inserting_after(parent):
            new_getitem_node = subgraph.create_node(
                "call_function", operator.getitem, (parent, path[-1]), {}
            )
        new_getitem_node.meta["example_value"] = cur_elem

        path_to_output_index[path] = len(getitem_nodes)
        getitem_nodes.append(new_getitem_node)

        if isinstance(cur_elem, tuple):
            queue.extend(
                [(e, path + (i,), new_getitem_node) for i, e in enumerate(cur_elem)]  # type: ignore[arg-type,misc]
            )

    return getitem_nodes, path_to_output_index  # type: ignore[return-value]


def _replace_tuple_outputs(
    node: Node,
    output_index: int,
    tuple_spec: dict[tuple[int, ...], int],
    invoke_subgraph_node: Node,
    graph: torch.fx.Graph,
) -> OrderedSet[Node]:
    if not _is_tuple_node(node):
        raise AssertionError("_replace_tuple_outputs expects a tuple node")

    queue = deque((c, (c.args[1],)) for c in _get_children_getitems(node))
    erased_nodes: OrderedSet[Node] = OrderedSet()
    while queue:
        cur_node, path = queue.pop()

        for c in _get_children_getitems(cur_node):
            queue.append((c, path + (c.args[1],)))  # type: ignore[return-value, arg-type]

        with graph.inserting_after(invoke_subgraph_node):
            subgraph_output = graph.create_node(
                "call_function",
                operator.getitem,
                (invoke_subgraph_node, output_index + tuple_spec[path]),  # type: ignore[index]
                {},
            )
        cur_node.replace_all_uses_with(subgraph_output, propagate_meta=True)
        graph.erase_node(cur_node)
        erased_nodes.add(cur_node)

    graph.erase_node(node)
    erased_nodes.add(node)
    return erased_nodes
<|endoftext|>from __future__ import annotations

import copy
import dataclasses
import dis
import functools
import logging
import sys
from collections import Counter
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import types

    from .symbolic_convert import InstructionTranslatorBase

from .bytecode_transformation import (
    create_copy,
    create_dup_top,
    create_instruction,
    create_swap,
    Instruction,
    unique_id,
)
from .codegen import PyCodegen
from .exc import unimplemented
from .graph_break_hints import DYNAMO_BUG
from .output_graph import GraphCompileReason, StackLocalsMetadata
from .variables.misc import NullVariable, UnknownVariable


log = logging.getLogger(__name__)


@functools.cache
def _get_comprehension_bytecode_prefix() -> list[str]:
    """Get the bytecode instructions that precede BUILD_LIST in a list comprehension."""

    if sys.version_info < (3, 12):
        raise AssertionError(
            f"comprehension bytecode prefix requires Python 3.12+, got {sys.version_info}"
        )

    def fn() -> list[int]:
        return [i for i in range(1)]  # noqa: C416

    insts = [inst.opname for inst in dis.get_instructions(fn)]

    start_idx = len(insts) - 1 - insts[::-1].index("LOAD_FAST_AND_CLEAR")
    end_idx = insts.index("BUILD_LIST")

    return insts[start_idx:end_idx]


@dataclasses.dataclass
class ComprehensionAnalysis:
    """Metadata about a comprehension's bytecode structure.

    Attributes:
        end_ip: Instruction pointer after all comprehension bytecode
        result_var: Name of result variable, or None if result stays on stack
        result_on_stack: True if result stays on stack (discarded, returned, or in expression)
        iterator_vars: Variables from LOAD_FAST_AND_CLEAR (need restoration)
        walrus_vars: Variables assigned via walrus operator (:=) inside comprehension
        captured_vars: Variables read from outer scope via LOAD_FAST inside comprehension
    """

    end_ip: int
    result_var: str | None
    result_on_stack: bool
    iterator_vars: list[str]
    walrus_vars: list[str]
    captured_vars: list[str]


def _is_comprehension_start(tx: InstructionTranslatorBase) -> bool:
    """Detect if we're at the start of a list/dict comprehension in 3.12+.

    In Python 3.12+, comprehensions are inlined with a bytecode pattern that
    precedes BUILD_LIST/BUILD_MAP.
    """
    if sys.version_info < (3, 12):
        raise AssertionError(
            f"comprehension start detection requires Python 3.12+, got {sys.version_info}"
        )

    if tx.instruction_pointer is None:
        raise AssertionError("instruction_pointer must not be None")
    ip = tx.instruction_pointer - 1

    pattern = _get_comprehension_bytecode_prefix()
    prefix = [inst.opname for inst in tx.instructions[ip - len(pattern) : ip]]

    return prefix == pattern


def _find_comprehension_end_for_ip(tx: InstructionTranslatorBase) -> int:
    """Find the instruction pointer of the outermost END_FOR for current comprehension."""
    if sys.version_info < (3, 12):
        raise AssertionError(
            f"comprehension END_FOR search requires Python 3.12+, got {sys.version_info}"
        )
    if tx.instruction_pointer is None:
        raise AssertionError("instruction_pointer must not be None")

    nesting_depth = 0
    for search_ip in range(tx.instruction_pointer, len(tx.instructions)):
        inst = tx.instructions[search_ip]
        if inst.opname == "FOR_ITER":
            nesting_depth += 1
        elif inst.opname == "END_FOR":
            nesting_depth -= 1
            if nesting_depth == 0:
                return search_ip
    return -1


def _analyze_comprehension(tx: InstructionTranslatorBase) -> ComprehensionAnalysis:
    """Analyze comprehension bytecode to determine result handling pattern."""
    if sys.version_info < (3, 12):
        raise AssertionError(
            f"comprehension analysis requires Python 3.12+, got {sys.version_info}"
        )
    if tx.instruction_pointer is None:
        raise AssertionError("instruction_pointer must not be None")

    start_ip = tx.instruction_pointer - 1  # BUILD_LIST/BUILD_MAP

    iterator_vars: list[str] = []
    walrus_vars: list[str] = []
    captured_vars: list[str] = []
    defined_inside: set[str] = set()

    # Collect iterator variables from LOAD_FAST_AND_CLEAR before BUILD_LIST/BUILD_MAP
    iter_scan_ip = start_ip - 1
    while iter_scan_ip >= 0:
        inst = tx.instructions[iter_scan_ip]
        if inst.opname == "LOAD_FAST_AND_CLEAR":
            iterator_vars.insert(0, inst.argval)
            iter_scan_ip -= 1
        elif inst.opname in ("SWAP", "GET_ITER"):
            iter_scan_ip -= 1
        else:
            break
    defined_inside.update(iterator_vars)

    end_for_ip = _find_comprehension_end_for_ip(tx)
    if end_for_ip == -1:
        unimplemented(
            gb_type="Comprehension analysis failed: No END_FOR",
            context="",
            explanation="Could not find END_FOR instruction in comprehension bytecode.",
            hints=[],
        )

    # Find first FOR_ITER to know where loop body starts
    for_iter_ip = next(
        i
        for i in range(start_ip, end_for_ip)
        if tx.instructions[i].opname == "FOR_ITER"
    )

    # Single pass through loop body to detect walrus vars and captured vars
    for body_ip in range(for_iter_ip + 1, end_for_ip):
        inst = tx.instructions[body_ip]

        # Detect walrus pattern: COPY 1 followed by STORE_FAST
        if inst.opname == "COPY" and inst.arg == 1 and body_ip + 1 < end_for_ip:
            next_inst = tx.instructions[body_ip + 1]
            if next_inst.opname == "STORE_FAST":
                var_name = next_inst.argval
                if var_name not in iterator_vars and var_name not in walrus_vars:
                    walrus_vars.append(var_name)
                    defined_inside.add(var_name)

        # Track variables defined inside the loop
        if inst.opname == "STORE_FAST":
            defined_inside.add(inst.argval)

        # Detect LOAD_FAST referencing outer variables
        elif inst.opname.startswith("LOAD_FAST"):
            var_names = (
                inst.argval if isinstance(inst.argval, tuple) else (inst.argval,)
            )
            for var_name in var_names:
                if var_name not in defined_inside and var_name not in captured_vars:
                    captured_vars.append(var_name)

    # Extract pre_store_ops: all opcodes from END_FOR+1 until first STORE_FAST
    pre_store_ops: list[str] = []
    scan_ip = end_for_ip + 1
    # In 3.13 END_FOR is followed by POP_TOP to pop the iter.  In 3.12, END_FOR is responsible for doing the pop, and in
    # 3.14+ POP_ITER is used instead.  We skip over the mandatory POP_TOP in 3.13 to avoid confusing the analysis below
    # (which looks for an additional POP_TOP if the result is discarded).
    if sys.version_info[:2] == (3, 13):
        scan_ip += 1
    while (
        scan_ip < len(tx.instructions)
        and tx.instructions[scan_ip].opname != "STORE_FAST"
    ):
        pre_store_ops.append(tx.instructions[scan_ip].opname)
        scan_ip += 1

    store_fast_ip = scan_ip

    # Skip all STORE_FASTs (iteration-variable restores plus an optional result
    # assignment) to find the end of the comprehension epilogue.
    while (
        scan_ip < len(tx.instructions)
        and tx.instructions[scan_ip].opname == "STORE_FAST"
    ):
        scan_ip += 1

    # The comprehension epilogue always restores the iteration variables that
    # were saved by LOAD_FAST_AND_CLEAR (one STORE_FAST each). The disposition
    # of the result is determined relative to those restores:
    #   - a POP_TOP right after the loop teardown discards the result
    #     (bare statement)
    #   - one extra STORE_FAST beyond the iteration vars is a result
    #     assignment (the extra store may reuse an iteration var's name)
    #   - no extra store leaves the result on the stack (returned/consumed)
    # We use this to classify the handling of the result as discarded, stored,
    # or returned/consumed
    n_iter = len(iterator_vars)
    store_targets = Counter(
        tx.instructions[i].argval for i in range(store_fast_ip, scan_ip)
    )

    result_var: str | None = None
    if pre_store_ops[-1:] == ["POP_TOP"]:  # discarded
        result_var = None
        result_on_stack = False
    elif store_targets.total() > n_iter:  # stored
        store_targets.subtract(iterator_vars)
        if store_targets.total() != 1:
            unimplemented(
                gb_type="Comprehension analysis failed: ambiguous result store",
                context=f"store_targets={store_targets}, iterator_vars={iterator_vars}",
                explanation="Expected exactly one comprehension result assignment.",
                hints=DYNAMO_BUG,
            )
        result_var = next(iter(store_targets))
        result_on_stack = False
    elif store_targets.total() == n_iter:  # returned/consumed
        result_var = None
        result_on_stack = True
    else:
        unimplemented(
            gb_type="Comprehension analysis failed: No matches",
            context=f"pre_store_ops={pre_store_ops}, store_targets={store_targets}, "
            f"iterator_vars={iterator_vars}",
            explanation="Comprehension does not match any known bytecode pattern.",
            hints=DYNAMO_BUG,
        )

    return ComprehensionAnalysis(
        end_ip=scan_ip,
        result_var=result_var,
        # pyrefly: ignore [unbound-name]
        result_on_stack=result_on_stack,
        iterator_vars=iterator_vars,
        walrus_vars=walrus_vars,
        captured_vars=captured_vars,
    )


def _handle_comprehension_graph_break(
    tx: InstructionTranslatorBase, inst: Instruction
) -> None:
    """Handle list/dict comprehension graph break.

    Builds a synthetic function wrapping the comprehension bytecode,
    calls it via codegen_call_resume, then chains into the resume
    function for the post-comprehension code.
    """
    if sys.version_info < (3, 12):
        raise AssertionError(
            f"comprehension graph break requires Python 3.12+, got {sys.version_info}"
        )
    if tx.instruction_pointer is None:
        raise AssertionError("instruction_pointer must not be None")

    start_ip = tx.instruction_pointer - 1  # BUILD_LIST/BUILD_MAP
    analysis = _analyze_comprehension(tx)
    stack_pops = 1 + len(analysis.iterator_vars)
    reason = GraphCompileReason("comprehension_graph_break", [tx.frame_summary()])
    log.debug("comprehension triggered compile")

    # --- Step 1: Compile the graph up to the comprehension ---

    all_stack_locals_metadata = tx.output.compile_subgraph(
        tx,
        reason=reason,
        stack_pops=stack_pops,
    )
    # Record which stack_pops items are NULL before popn loses the info.
    # NULLs on the CPython stack can't be passed as function arguments.
    stack_pops_null_mask = [
        isinstance(tx.stack[len(tx.stack) - stack_pops + i], NullVariable)
        for i in range(stack_pops)
    ]

    tx.popn(stack_pops)
    meta = all_stack_locals_metadata[0]
    cg = PyCodegen(tx.output.root_tx)

    # Runtime stack after compile_subgraph:
    #   cells, [frame_values], *(non-popped items), *(stack_pops items w/ NULLs)
    # frame_values[0] = [frame N locals] (no stack items yet)

    nonnull_count = sum(1 for m in stack_pops_null_mask if not m)

    # live_stack_depth: stack items above cells/frame_values excluding NULLs
    # that compile_subgraph didn't codegen (tracked in stack_null_idxes).
    live_stack_depth = len(tx.stack) - len(meta.stack_null_idxes)

    # --- Step 2: Pop stack_pops items and append non-nulls to frame_values[0] ---
    # SWAP each item to TOS then LIST_APPEND or pop_null; fv_list stays at
    # TOS throughout. Items append in TOS-first (reversed) order;
    # _build_comprehension_fn compensates by loading in reverse.
    cg.extend_output(
        [
            # frame_values[0] to TOS
            *create_copy(live_stack_depth + stack_pops + 1),
            cg.create_load_const(0),
            cg.create_binary_subscr(),
        ]
    )
    for i in reversed(range(stack_pops)):
        cg.extend_output(create_swap(2))
        if stack_pops_null_mask[i]:
            cg.extend_output(cg.pop_null())
        else:
            cg.extend_output(cg.create_list_append())
    cg.extend_output([create_instruction("POP_TOP")])

    # Stack: cells, [frame_values], *(non-popped items)

    # --- Step 3: Build comprehension function ---
    new_code, fn_name = _build_comprehension_fn(
        tx,
        analysis,
        start_ip,
        stack_pops,
        stack_pops_null_mask,
        nonnull_count,
        meta,
    )

    # --- Step 4: Extract [cells[0]] and [frame_values[0]] for codegen_call_resume ---
    cg.extend_output(
        [
            *create_copy(live_stack_depth + 2),
            cg.create_load_const(0),
            cg.create_binary_subscr(),
            create_instruction("BUILD_LIST", arg=1),
            *create_copy(live_stack_depth + 2),
            cg.create_load_const(0),
            cg.create_binary_subscr(),
            create_instruction("BUILD_LIST", arg=1),
        ]
    )

    # Stack: ..., *(non-popped), [cells[0]], [frame_values[0]]

    # --- Step 5: Call comprehension function via codegen_call_resume ---
    tx.codegen_call_resume([new_code], [fn_name], cg)

    # Stack: ..., *(non-popped), comp_result

    # --- Step 6: Remove appended stack_pops items from frame_values[0] ---
    if nonnull_count > 0:
        frame_values_pos = live_stack_depth + 1 + 1  # +1 result, +1 frame_values
        cg.extend_output(
            [
                *create_copy(frame_values_pos),
                cg.create_load_const(0),
                cg.create_binary_subscr(),
                # frame_values[0] on TOS
                create_dup_top(),
                # frame_values[0], frame_values[0]
                cg.create_load_const(-nonnull_count),
                cg.create_load_const(None),
                create_instruction("BUILD_SLICE", arg=2),
                create_instruction("DELETE_SUBSCR"),
                # del frame_values[0][-nonnull_count:]
                create_instruction("POP_TOP"),
            ]
        )

    # --- Step 7: Pass comprehension outputs to frame_values[0] ---
    # Walrus vars first, then result_var.
    vars_to_pass = analysis.walrus_vars + (
        [analysis.result_var] if analysis.result_var else []
    )

    existing_vars: dict[str, int] = {}
    for var_name in vars_to_pass:
        tx.symbolic_locals[var_name] = UnknownVariable()
        if var_name in meta.locals_names:
            existing_vars[var_name] = meta.locals_names[var_name]
        else:
            # The var was cleared to NULL at the break (e.g. a comprehension
            # iteration variable reused as the result target). We are binding
            # it to a real value, so drop the NULL restoration; otherwise the
            # resume function would reset it to NULL and clobber the result.
            if var_name in meta.locals_null_keys:
                meta.locals_null_keys.remove(var_name)
            meta.locals_names[var_name] = len(meta.locals_names)

    fv_depth = live_stack_depth + 2  # comp_result + frame_values

    # --- Walrus vars: extract from comp_result tuple ---
    if analysis.walrus_vars:
        # comp_result is (result, *walrus_vars).
        cg.extend_output(
            [
                *create_copy(fv_depth),
                cg.create_load_const(0),
                cg.create_binary_subscr(),
            ]
        )
        # Stack: ..., comp_tuple, fv0
        for j, walrus_var in enumerate(analysis.walrus_vars):
            cg.extend_output(
                [
                    *create_copy(2),
                    cg.create_load_const(j + 1),
                    cg.create_binary_subscr(),
                ]
            )
            # Stack: ..., comp_tuple, fv0, walrus_value
            if walrus_var in existing_vars:
                # fv0[idx] = walrus_value
                cg.extend_output(
                    [
                        *create_copy(2),  # copy fv0
                        cg.create_load_const(existing_vars[walrus_var]),
                        create_instruction("STORE_SUBSCR"),
                    ]
                )
            else:
                cg.extend_output(cg.create_list_append())
            # Stack: ..., comp_tuple, fv0
        cg.extend_output(
            [
                create_instruction("POP_TOP"),  # pop fv0
                # Extract the result from the tuple.
                cg.create_load_const(0),
                cg.create_binary_subscr(),
            ]
        )
        # Stack: ..., result

    # --- Result: keep on stack, overwrite/append to fv[0], or discard ---
    if analysis.result_on_stack:
        tx.push(UnknownVariable())
    elif analysis.result_var:
        cg.extend_output(
            [
                *create_copy(fv_depth),
                cg.create_load_const(0),
                cg.create_binary_subscr(),
                # Stack: ..., result, fv0
            ]
        )
        if analysis.result_var in existing_vars:
            cg.extend_output(
                [
                    cg.create_load_const(existing_vars[analysis.result_var]),
                    create_instruction("STORE_SUBSCR"),
                    # fv0[idx] = result
                ]
            )
        else:
            cg.extend_output(
                [
                    *create_swap(2),
                    *cg.create_list_append(),
                    create_instruction("POP_TOP"),
                ]
            )
    else:
        cg.extend_output([create_instruction("POP_TOP")])

    # Stack: cells, [frame_values], *(non-popped stack)
    tx.output.add_output_instructions(cg.get_instructions())

    # --- Step 8: Create resume function chain ---
    resume_inst = tx.instructions[analysis.end_ip]
    tx.output.add_output_instructions(
        tx.create_call_resume_at(resume_inst, all_stack_locals_metadata)
    )

    tx.instruction_pointer = None


def _build_comprehension_fn(
    tx: InstructionTranslatorBase,
    analysis: ComprehensionAnalysis,
    start_ip: int,
    stack_pops: int,
    stack_pops_null_mask: list[bool],
    nonnull_count: int,
    meta: StackLocalsMetadata,
) -> tuple[types.CodeType, str]:
    """Build a synthetic function wrapping comprehension bytecode.

    Uses the same calling convention as resume functions created by
    create_resume / ContinueExecutionCache.generate: the first two args
    are __nested_resume_fns and __nested_frame_values (ignored here),
    followed by stack items and live locals.

    Returns (code, name) where name is the global name for the function.
    """
    from .bytecode_transformation import transform_code_object
    from .eval_frame import skip_code
    from .resume_execution import CO_VARARGS, CO_VARKEYWORDS

    # Args follow frame_values layout: locals first, then stack_pops items
    # (appended to end of frame_values[0] by the caller).
    # codegen_call_resume unpacks frame_values[0] as positional args.
    argnames = tuple(k for k in meta.locals_names if k not in tx.cell_and_freevars())
    args = (
        ["__nested_resume_fns", "__nested_frame_values"]
        + list(argnames)
        + [f"___stack{i}" for i in range(nonnull_count)]
    )

    freevars = tuple(
        sorted(list(tx.f_code.co_cellvars or []) + list(tx.f_code.co_freevars or []))
    )

    lineno = tx.lineno if tx.lineno is not None else tx.f_code.co_firstlineno
    fn_name = unique_id(f"__comprehension_{tx.f_code.co_name}_at_{lineno}")

    comprehension_body_vars = (
        analysis.iterator_vars
        + analysis.walrus_vars
        + ([analysis.result_var] if analysis.result_var else [])
        + analysis.captured_vars
    )

    def update(instructions: list[Instruction], code_options: dict[str, Any]) -> None:
        code_options["co_name"] = fn_name
        if sys.version_info >= (3, 11):
            code_options["co_qualname"] = fn_name
        code_options["co_firstlineno"] = lineno
        code_options["co_cellvars"] = ()
        code_options["co_freevars"] = freevars
        code_options["co_argcount"] = len(args)
        code_options["co_posonlyargcount"] = 0
        code_options["co_kwonlyargcount"] = 0
        code_options["co_varnames"] = tuple(
            args + [v for v in comprehension_body_vars if v not in args]
        )
        code_options["co_flags"] = code_options["co_flags"] & ~(
            CO_VARARGS | CO_VARKEYWORDS
        )

        prefix: list[Instruction] = []
        if freevars:
            prefix.append(create_instruction("COPY_FREE_VARS", arg=len(freevars)))
        prefix.append(create_instruction("RESUME", arg=0))

        # Push stack_pops items onto operand stack so the comprehension
        # bytecode finds them where it expects (iterator + saved vars).
        # NULL positions get PUSH_NULL, non-null get LOAD_FAST.
        # Items were appended to frame_values[0] in TOS-first order,
        # so load in reverse to reconstruct the original stack layout.
        nonnull_i = nonnull_count - 1
        for i in range(stack_pops):
            if stack_pops_null_mask[i]:
                prefix.append(create_instruction("PUSH_NULL"))
            else:
                prefix.append(
                    create_instruction("LOAD_FAST", argval=f"___stack{nonnull_i}")
                )
                nonnull_i -= 1

        comp_insts = _copy_comprehension_bytecode(tx, start_ip, analysis.end_ip)

        # Epilogue: ensure result is on stack, pack walrus vars, return.
        epilogue: list[Instruction] = []
        if not analysis.result_on_stack:
            if analysis.result_var:
                epilogue.append(
                    create_instruction("LOAD_FAST", argval=analysis.result_var)
                )
            else:
                epilogue.append(create_instruction("LOAD_CONST", argval=None))
        if analysis.walrus_vars:
            for var_name in analysis.walrus_vars:
                epilogue.append(create_instruction("LOAD_FAST", argval=var_name))
            epilogue.append(
                create_instruction(
                    "BUILD_TUPLE",
                    arg=1 + len(analysis.walrus_vars),
                )
            )
        epilogue.append(create_instruction("RETURN_VALUE"))

        instructions[:] = prefix + comp_insts + epilogue

    new_code, _ = transform_code_object(tx.f_code, update)
    skip_code(new_code)

    # Install as global
    tx.output.install_resume_function_global(fn_name, new_code, tx.f_globals)

    return new_code, fn_name


def _copy_comprehension_bytecode(
    tx: InstructionTranslatorBase, start_ip: int, end_ip: int
) -> list[Instruction]:
    """Copy comprehension bytecode instructions, updating jump targets."""
    inst_map: dict[Instruction, Instruction] = {}
    copied_insts: list[Instruction] = []

    for ip in range(start_ip, end_ip):
        original_inst = tx.instructions[ip]
        copied_inst = copy.copy(original_inst)
        copied_inst.exn_tab_entry = None
        inst_map[original_inst] = copied_inst
        copied_insts.append(copied_inst)

    for copied_inst in copied_insts:
        if copied_inst.target is not None and copied_inst.target in inst_map:
            copied_inst.target = inst_map[copied_inst.target]

    return copied_insts


def maybe_setup_comprehension_speculation(
    tx: InstructionTranslatorBase, inst: Instruction
) -> bool:
    """
    Handle comprehension start for Python 3.12+ BUILD_LIST/BUILD_MAP with argval 0.
    Returns True if a graph break was triggered and the caller should return early.
    """
    if not (sys.version_info >= (3, 12) and inst.argval == 0):
        return False

    if not _is_comprehension_start(tx):
        return False

    can_speculate = (
        all(b.can_restore() for b in tx.block_stack)
        and not tx.one_graph
        and not tx.error_on_graph_break
        and not tx.is_tracing_resume_prologue
        and not tx.active_generic_context_managers
        and tx.output.current_tracer.parent is None
    )

    if can_speculate and tx.parent is not None:
        can_speculate = tx._can_speculate_comprehension_nested()
    # Only set up speculation at depth 0 (outermost comprehension)
    if can_speculate and tx._comprehension_depth == 0:
        speculation = tx.speculate()
        if speculation.failed(tx):
            _handle_comprehension_graph_break(tx, inst)
            return True
        tx.current_speculation = speculation
    end_for_ip = _find_comprehension_end_for_ip(tx)
    if end_for_ip < 0:
        raise AssertionError("failed to find END_FOR for comprehension")
    tx._comprehension_end_for_ips.add(end_for_ip)
    tx._comprehension_depth += 1
    return False
<|endoftext|>"""
Profile Guided Optimization (PGO) implementation for Dynamo.

This module provides functionality for caching and managing code state profiles
that guide optimization decisions in Dynamo. It implements both local and remote
caching mechanisms for storing profile information across runs, handles profile
merging across distributed ranks, and manages the lifecycle of profile data
during compilation. The profiles track dynamic vs static properties of tensors
and help Dynamo make better specialization decisions.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import enum
import functools
import logging
import os
import pickle
import re
import zlib
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar
from typing_extensions import override, Self

import torch._dynamo.config
import torch._utils_internal
import torch.compiler.config
import torch.distributed as dist
from torch._dynamo.utils import (
    CompileEventLogger,
    dynamo_timed,
    set_feature_use,
    warn_once,
)
from torch._environment import is_fbcode
from torch._logging._internal import trace_structured_artifact
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactRecorder,
)
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    import types

    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._inductor.remote_cache import JsonDataTy, RemoteCache


class ReservedWorkflowIdUserError(ValueError):
    pass


log = logging.getLogger(__name__)
symbolic_shape_log = logging.getLogger("torch.fx.experimental.symbolic_shapes")

LOCK_TIMEOUT = 10

# How does in memory representation work?  Concretely, this module is
# responsible for holding GLOBAL state representing the state it holds, no
# other copies permitted.  So we retire frame_state entirely and store it
# here.  This should be reset when Dynamo is reset.  We never GC information
# (similar to how the filesystem doesn't get cleaned up except by tmp
# cleaner), so the expectation is the information is relatively cheap and we
# don't mind leaking it.


# How exactly did we design the cache key?  Here are some of the questions:
#
# - JOB_ID: Do we have a unique identifier for the "training run"  (such that
#   it stays the same if we're running the same code, and changes if we're
#   running something different).
#
# - RANK: Are we sharing the cache across ranks, or does each rank get
#   an individual cache?
#
# We choose to require job_id for PGO cache.  This is to prevent
# situations where unrelated invocations of PyTorch unpredictably cause
# changes to each other's behavior.  With a job_id, at least you know there
# is some "state" associated with it.  (State dict might be another way to
# tell if a run is related or not.)  You can opt-in to YOLO everything
# aliases everything by passing a shared job_id for all your invocations.
#
# We choose to NOT share PGO cache across ranks.  With no RANK_SHARING, there
# is never contention between runs, so we can leisurely update a bundle with
# information we need.  Because we are grouped by job_id, we can have a single
# consolidated bundle for everything (or not; maybe worry about O(n^2) IO if
# we updated every compile--let's just instrument this.)  Can even take a
# filelock for extra safety (expect no contention); expect 50ns overhead from
# uncontended filelock.
#
# If we did share ranks, everyone is storming to modify the same cache files.
# We can do this by having folks atomic write to a CAS-store and then having
# readers do on-the-fly merging (this can be implemented in remote using
# prefix iteration).  As an optional optimization, one rank can be elected to
# handling bundling post facto (ideally, this is done async, after quiescence,
# without compiler collective need to wait for everyone to finish writing
# their bits.) Not sure how you can avoid a listdir because if some rank shows
# up with some new entries we need to pull them in ASAP (unless you want to
# delay bundling).
#
# But compiler collectives fill a similar niche:  compilers chat with each
# other so rank 0 has collected everything.  So elect rank 0 only to write the
# bundle.  Don't even need CAS-store atomic write; just one rank writing an
# updating bundles.  The point is that use compiler collectives to share
# profiles across ranks, but use the PGO cache to persist profiles per rank
# across attempts.  No need to have one mechanism to do everything.


@functools.cache
def _hash_containing_file(filepath: str) -> str:
    # if the file does not exist we consider filepath to be the hash.
    if not os.path.exists(filepath):
        return filepath

    with open(filepath, "rb") as file:
        content = file.read()
        crc32_value = zlib.crc32(content)
        hash = format(crc32_value & 0xFFFFFFFF, "08x")
        return hash


def _get_closure_content(content: types.CellType) -> object:
    if callable(content) and hasattr(content, "__code__"):
        return content.__code__
    return None


def _get_cell_hash_content(content: types.CellType) -> object:
    # Safely extract cell contents without blowing up the entire tuple hash
    try:
        return _get_closure_content(content.cell_contents)
    except ValueError:
        return None


@dataclasses.dataclass(frozen=True)
class CodeId:
    filename: str
    firstlineno: int
    name: str
    # When a job restart, the code can be copied to a different path than the previous attempt. In that case
    # self.filename will have a different value,  we do not want to consider those differences. Instead we
    # hash the content of the file and use it as an identifier of the file.
    #
    # self.filename is kept in the object to give readable information/pointer to the actual file, in a local
    # code state it will refer to the first seen file path.
    file_hash: str
    closure_hash: int | None = None

    # Exclude file name.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CodeId):
            return False
        return (
            self.file_hash == other.file_hash
            and self.firstlineno == other.firstlineno
            and self.name == other.name
            and self.closure_hash == other.closure_hash
        )

    # Ensure if two CodeIds are the same, then they have the same hash by excluding filename.
    def __hash__(self) -> int:
        return hash((self.file_hash, self.name, self.firstlineno, self.closure_hash))

    def __str__(self) -> str:
        return f"hash({self.file_hash}){self.filename}:{self.firstlineno}:{self.name}"

    @staticmethod
    def make(
        code: types.CodeType, closure: tuple[types.CellType, ...] | None = None
    ) -> CodeId:
        closure_hash = None
        if closure:
            closure_hash = hash(tuple(_get_cell_hash_content(c) for c in closure))

        return CodeId(
            code.co_filename,
            code.co_firstlineno,
            code.co_name,
            _hash_containing_file(code.co_filename),
            closure_hash,
        )


@dataclasses.dataclass
class CodeState:
    automatic_dynamic: defaultdict[str, FrameStateSizeEntry] = dataclasses.field(
        default_factory=lambda: defaultdict(FrameStateSizeEntry)
    )


_INIT_CODE_STATE: defaultdict[CodeId, CodeState] | None = None
_CODE_STATE: defaultdict[CodeId, CodeState] | None = None
_LOGGED_DYNAMIC_ALLOWLIST: bool = False
_KNOWN_DYNAMIC_SOURCES: set[str] = set()


@dataclasses.dataclass(frozen=True)
class InferStride:
    """
    Denotes the quantity stride[dim] * size[dim], which is what the stride would
    be for the next physical dimension that results in a contiguous layout.

    For example, given size = [2, 3], stride = [3, 1], we can replace this with
    stride = [InferStride(1), 1], because InferStride(1) = stride[1] * size[1] = 1 * 3 = 3

    Indirecting the representation in this way is important for the join operation
    on strides as if we join [2, 3][3, 1] and [2, 4][4, 1],
    we don't want [2, None][None, 1] which would get eventually symbolized into
    [2, s0][s1, 1] (notice that the relationship between s0 and s1 is broken).
    If we instead rewrite the expressions as InferStride so we have [2, 3][InferStride(1), 1]
    and [2, 4][InferStride(1), 1] we now join to [2, None][InferStride(1), 1] will
    result in [2, s0][s0, 1], as desired.
    """

    dim: int


_T = TypeVar("_T")


class AutoUnset(enum.Enum):
    """
    The identity element of our semilattice, a generic "don't know" element that
    is always subsumed when we get more information.
    """

    token = 0


auto_unset = AutoUnset.token


class AutoDynamic(enum.Enum):
    """
    The top element of our (bounded) semilattice, whenever you merge this with
    any other element you always get it again
    """

    token = 0


auto_dynamic = AutoDynamic.token


@dataclasses.dataclass
class FrameStateSizeEntry:
    scalar: int | AutoDynamic | AutoUnset = dataclasses.field(default=auto_unset)
    # NB: We don't have cases where we have a known dimensionality but
    # we know NOTHING about the individual sizes
    size: AutoDynamic | AutoUnset | tuple[int | AutoDynamic, ...] = dataclasses.field(
        default=auto_unset
    )
    stride: AutoDynamic | AutoUnset | tuple[int | AutoDynamic | InferStride, ...] = (
        dataclasses.field(default=auto_unset)
    )
    excluded_sizes: tuple[int | None, ...] | None = dataclasses.field(
        default=None, compare=False
    )
    excluded_scalar: int | None = dataclasses.field(default=None, compare=False)

    def render(self) -> str:
        # Special cases
        def render_single(s: int | AutoDynamic | AutoUnset | InferStride) -> str:
            if s is auto_dynamic:
                return "?"
            elif s is auto_unset:
                # This basically shouldn't happen, this is for debugging
                return "auto unset"
            elif isinstance(s, InferStride):
                return f"S({s.dim})"
            else:
                return str(s)

        def render_tuple(ss: tuple[int | AutoDynamic | InferStride, ...]) -> str:
            return "[" + ", ".join(render_single(s) for s in ss) + "]"

        # Common cases
        if self.size is auto_dynamic and self.stride is auto_dynamic:
            if self.scalar is auto_dynamic:
                return "fully dynamic scalar or tensor"
            else:
                return f"scalar {self.scalar}"
        elif self.scalar is auto_dynamic:
            if isinstance(self.size, tuple) and isinstance(self.stride, tuple):
                return f"tensor size={render_tuple(self.size)} stride={render_tuple(self.stride)}"

        # Fallback
        return f"unusual {repr(self)}"

    def __post_init__(self) -> None:
        if isinstance(self.scalar, torch.SymInt):
            raise AssertionError(self.scalar)
        if isinstance(self.size, tuple):
            for s in self.size:
                if isinstance(s, torch.SymInt):
                    raise AssertionError(s)
        if isinstance(self.stride, tuple):
            for s1 in self.stride:
                if isinstance(s1, torch.SymInt):
                    raise AssertionError(s1)

    def is_size_dynamic(self, dim: int) -> bool:
        if self.size is auto_dynamic:
            return True
        if self.size is auto_unset:
            return False
        return self.size[dim] is auto_dynamic

    def is_stride_dynamic(self, dim: int) -> bool:
        # At the moment, dynamic strides is a bit buggy.  Good test case
        # here is `PYTORCH_TEST_WITH_DYNAMO=1 python test/test_autograd.py
        # TestAutograd.test_gradcheck_jacobian_mismatch`
        #
        # This if statement preserves historical behavior, which is that we
        # ONLY make strides dynamic if the size is exactly static everywhere.
        # We could potentially relax this but in general we should be very
        # careful about when to infer dynamic strides.
        #
        # Actually, the existing algorithm is already somewhat problematic.
        # Suppose a tensor that is sometimes:
        # f32[2, 3, 5][15, 5, 1] and other times
        # f32[2, 3, 5][5, 10, 1] (specifically, dim 0 and 1 are physically transposed).
        # If we infer strides should be (DYNAMIC, DYNAMIC, 1).  But this is
        # silly: we really should have just guarded on dim order.
        if not (
            isinstance(self.size, tuple) and all(type(s) is int for s in self.size)
        ):
            return False
        if self.stride is auto_dynamic:
            return True
        if self.stride is auto_unset:
            return False
        return self.stride[dim] is auto_dynamic

    @staticmethod
    def _munge_symint(xs: tuple[int, ...]) -> tuple[AutoDynamic | int, ...]:
        return tuple(auto_dynamic if isinstance(x, torch.SymInt) else x for x in xs)

    @classmethod
    def make_scalar(cls, x: int) -> FrameStateSizeEntry:
        return FrameStateSizeEntry(scalar=x, size=auto_dynamic, stride=auto_dynamic)

    @classmethod
    def make_tensor(
        cls, size: tuple[int, ...], stride: tuple[int, ...]
    ) -> FrameStateSizeEntry:
        return FrameStateSizeEntry(
            scalar=auto_dynamic,
            size=cls._munge_symint(size),
            stride=cls._munge_symint(stride),
        )

    @classmethod
    def make_size(cls, size: tuple[int, ...]) -> FrameStateSizeEntry:
        return FrameStateSizeEntry(
            scalar=auto_unset,
            size=cls._munge_symint(size),
            stride=auto_unset,
        )

    @staticmethod
    def _merge_atom(x: _T, y: _T) -> AutoDynamic | _T:
        if x is auto_unset:
            return y
        if y is auto_unset:
            return x
        if x is auto_dynamic or y is auto_dynamic or x != y:
            return auto_dynamic
        return x

    @classmethod
    def _merge_atom_tup(
        cls,
        xs: AutoDynamic | AutoUnset | tuple[_T, ...],
        ys: AutoDynamic | AutoUnset | tuple[_T, ...],
    ) -> AutoDynamic | AutoUnset | tuple[AutoDynamic | _T, ...]:
        if xs is auto_unset:
            return ys
        if ys is auto_unset:
            return xs
        if xs is auto_dynamic or ys is auto_dynamic:
            return auto_dynamic
        if len(xs) != len(ys):
            return auto_dynamic
        return tuple(cls._merge_atom(x, y) for x, y in zip(xs, ys))

    def __ior__(self, other: Self) -> Self:
        # Record current static sizes before merge. For dims that become
        # dynamic, the exclusion guard will reject these values so inputs
        # fall through to the earlier, more specialized cache entry.
        # Already-dynamic dims become None and are ignored by the guard.
        # When no dim transitions, clear stale excluded_sizes so later
        # compilations don't inherit exclusions from earlier transitions.
        new_size = self._merge_atom_tup(self.size, other.size)
        if isinstance(self.size, tuple):
            if new_size != self.size:
                self.excluded_sizes = tuple(
                    s if type(s) is int else None for s in self.size
                )
            elif self.excluded_sizes is not None:
                self.excluded_sizes = None
        self.size = new_size
        # Same idea for scalars: record the static value about to become dynamic.
        # Re-derive like excluded_sizes: only set when transitioning from a
        # concrete int, clear when already dynamic.
        if (
            type(self.scalar) is int
            and type(other.scalar) is int
            and self.scalar != other.scalar
        ):
            self.excluded_scalar = self.scalar
        elif self.scalar is auto_dynamic and self.excluded_scalar is not None:
            self.excluded_scalar = None
        self.scalar = self._merge_atom(self.scalar, other.scalar)
        self.stride = self._merge_atom_tup(self.stride, other.stride)
        return self


def update_automatic_dynamic(
    tx: InstructionTranslatorBase,
    name: str,
    entry: FrameStateSizeEntry,
    *,
    is_unspecialized_nn_module: bool = False,
) -> FrameStateSizeEntry:
    code_id = CodeId.make(tx.f_code, tx.closure)
    frame_state = get_code_state()[code_id]
    if torch._dynamo.config.automatic_dynamic_shapes:
        is_update = name in frame_state.automatic_dynamic
        mut_entry = frame_state.automatic_dynamic[name]
        old_entry = copy.copy(mut_entry)
        mut_entry |= entry

        # Do some logs (damn, I spend more code logging than I do actually doing
        # the updates lol)
        if is_update and old_entry.scalar != mut_entry.scalar:
            symbolic_shape_log.info(
                "marking %s as dynamic (from automatic dynamic/PGO)",
                name,
            )
            log.debug(
                "automatic dynamic int %s val %s != %s",
                name,
                entry.scalar,
                old_entry.scalar,
            )
            CompileEventLogger.instant(
                "automatic_dynamic",
                {
                    "name": name,
                    "dim_changed": "scalar",
                    "reason": "scalar change",
                    "cached": str(old_entry.scalar),
                    "new": str(entry.scalar),
                },
            )
            if is_unspecialized_nn_module:
                log.info(
                    "%s is converted to a symbolic integer. It is an attribute of a "
                    "user defined nn module class. If you wish to keep it static, you can "
                    "mark the nn module class as `torch._dynamo.mark_static`.",
                    name,
                )

        def log_tup(
            tup_name: str, short_reason: str, long_reason: str, i: int | None = None
        ) -> None:
            entry_tup = (
                getattr(entry, tup_name) if i is None else getattr(entry, tup_name)[i]
            )
            old_entry_tup = (
                getattr(old_entry, tup_name)
                if i is None
                else getattr(old_entry, tup_name)[i]
            )
            symbolic_shape_log.info(
                "marking %s as dynamic (from automatic dynamic/PGO)",
                name,
            )
            log.debug(
                "automatic dynamic %s %s %s %s != %s",
                tup_name,
                name,
                short_reason,
                # NB: We used to only report len(...) here for dim mismatch
                entry_tup,
                old_entry_tup,
            )
            CompileEventLogger.instant(
                "automatic_dynamic",
                {
                    "name": name,
                    "dim_changed": "all" if i is None else i,
                    "reason": long_reason,
                    "cached": str(old_entry_tup),
                    "new": str(entry_tup),
                },
            )

        if is_update and old_entry.size != mut_entry.size:
            if isinstance(old_entry.size, tuple) and isinstance(entry.size, tuple):
                if len(old_entry.size) != len(entry.size):
                    log_tup("size", "dim", "dimensionality change")
                else:
                    for i in range(len(entry.size)):
                        if old_entry.size[i] != entry.size[i]:
                            log_tup("size", f"size({i})", "size change", i)
            else:
                log_tup("size", "other", "other")

        if is_update and old_entry.stride != mut_entry.stride:
            if isinstance(old_entry.stride, tuple) and isinstance(entry.stride, tuple):
                if len(old_entry.stride) != len(entry.stride):
                    log_tup("stride", "dim", "dimensionality change")
                else:
                    for i in range(len(entry.stride)):
                        if old_entry.stride[i] != entry.stride[i]:
                            log_tup("stride", f"stride({i})", "stride change", i)
            else:
                log_tup("stride", "other", "other")
    else:
        old_entry = frame_state.automatic_dynamic[name]
        log.debug(
            "automatic dynamic is off, overwriting int %s val %s -> %s",
            name,
            old_entry.scalar,
            entry.scalar,
        )
        frame_state.automatic_dynamic[name] = entry
        mut_entry = entry

    return mut_entry


def process_automatic_dynamic(
    tx: InstructionTranslatorBase,
    name: str,
    entry: FrameStateSizeEntry,
    *,
    is_unspecialized_nn_module: bool = False,
) -> FrameStateSizeEntry:
    if (st := tx.distributed_state) is None:
        return update_automatic_dynamic(
            tx,
            name,
            entry,
            is_unspecialized_nn_module=is_unspecialized_nn_module,
        )
    elif st.all_states is None:
        # Preflight, always pretend as if it's static.  The point here
        # is we want to get through the preflight quickly, and static
        # will run faster.  The preexisting frame state will get
        # applied anyway after we do compiler collectives.
        # TODO: I'm not sure if we should just bong the entire pgo
        # state here, it kind of depends if we're going to have other
        # things that talk in compiler collective.  Also, the PGO
        # state, if we've already inferred something is automatic
        # dynamic, will have lost the actual input sizes, which might
        # be useful for debugging purposes (e.g., observing 0/1
        # specialization).  Bonging the entire PGO state here would
        # let us delete this logic here; the compiler collective
        # would just directly update_automatic_dynamic
        st.local_state.automatic_dynamic[name] = entry
        return entry
    else:
        # Apply the updates.  NB: all_states includes the local state
        # too.
        res = None
        for sub_state in st.all_states:
            if name in sub_state.automatic_dynamic:
                res = update_automatic_dynamic(
                    tx,
                    name,
                    sub_state.automatic_dynamic[name],
                    is_unspecialized_nn_module=is_unspecialized_nn_module,
                )
        if res is None:
            raise AssertionError(
                "expected at least one matching automatic_dynamic entry across ranks"
            )
        return res


def format_cache_key(key: str) -> str:
    # NB: We always use global rank for keys, even though they are overkill
    # for local only cache
    rank = None
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()

    tag = torch.compiler.config.cache_key_tag
    return f"{key}:{rank}:{tag}"


def get_cache_key() -> str | None:
    # TODO: info versions of these logs that log only once
    if torch.compiler.config.force_disable_caches:
        warn_once(
            "dynamo_pgo force disabled by torch.compiler.config.force_disable_caches"
        )
        return None

    # NB: We namespace the cache keys so that only user-specified job id
    # can alias with each other.
    if (r := torch.compiler.config.job_id) is not None:
        if r.startswith("mast:"):
            raise ReservedWorkflowIdUserError(
                "torch.compiler.config.job_id with prefix 'mast:' is reserved for "
                "automatically generated job id associated with a specific MAST job "
                "name and version."
            )
        return format_cache_key(r)

    if (name_version := torch._utils_internal.get_mast_job_name_version()) is not None:
        mast_job_name, mast_job_version = name_version
        return format_cache_key(f"mast:{mast_job_name}:{mast_job_version}")

    return None


def get_extra_cache_key(sticky_key: str) -> str | None:
    if torch.compiler.config.force_disable_caches:
        warn_once(
            "dynamo_pgo force disabled by torch.compiler.config.force_disable_caches"
        )
        return None

    return format_cache_key(sticky_key)


# This solely controls local PGO
def code_state_path(cache_key: str) -> str | None:
    if not torch._dynamo.config.automatic_dynamic_local_pgo:
        log.debug("automatic_dynamic_local_pgo not enabled")
        return None

    from torch._inductor.runtime.runtime_utils import cache_dir

    code_state_key = re.sub(r'[<>:"/\\|?*]', "_", f"code_state_{cache_key}.pkl")
    return os.path.join(cache_dir(), "dynamo", code_state_key)


def should_use_remote_dynamo_pgo_cache() -> bool:
    if torch.compiler.config.force_disable_caches:
        return False

    if (r := torch._dynamo.config.automatic_dynamic_remote_pgo) is not None:
        return r

    if not is_fbcode():
        return False

    if torch._utils_internal.is_fb_unit_test():
        return False

    try:
        from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
    except ModuleNotFoundError:
        return False

    return REMOTE_CACHE_VERSION >= torch._utils_internal.justknobs_getval_int(
        "pytorch/remote_cache:dynamo_pgo_version"
    )


def get_remote_cache() -> RemoteCache[JsonDataTy] | None:
    from torch._inductor.remote_cache import create_cache

    if not should_use_remote_dynamo_pgo_cache():
        return None

    return create_cache(
        "dynamo-pgo",
        is_fbcode(),
        "FbRemoteDynamoPGOCache",
        "RemoteDynamoPGOCache",
    )


def _collect_dynamic_sources(code_state: CodeState) -> OrderedSet[str]:
    dynamic_sources: OrderedSet[str] = OrderedSet()
    for src, fs in code_state.automatic_dynamic.items():
        dynamic = False
        if isinstance(fs.size, tuple):
            dynamic = auto_dynamic in fs.size  # type: ignore[operator]
        elif fs.scalar == auto_dynamic:
            dynamic = True
        if dynamic:
            dynamic_sources.add(src)
    return dynamic_sources


def _collect_missing_sources(all_sources: OrderedSet[str]) -> OrderedSet[str]:
    from torch._dynamo.variables.builder import is_dynamic_source

    global _KNOWN_DYNAMIC_SOURCES
    missing_sources: OrderedSet[str] = OrderedSet()
    for src in all_sources:
        if src in _KNOWN_DYNAMIC_SOURCES:
            continue
        elif is_dynamic_source(src):
            _KNOWN_DYNAMIC_SOURCES.add(src)
            continue
        missing_sources.add(src)
    return missing_sources


def log_frame_dynamic_whitelist(f_code: types.CodeType) -> None:
    global _KNOWN_DYNAMIC_SOURCES
    code_id = CodeId.make(f_code, None)
    frame_state = get_code_state()[code_id]
    all_dynamic_sources = _collect_dynamic_sources(frame_state)
    frame_whitelist = ",".join(all_dynamic_sources)
    missing_whitelist = ",".join(_collect_missing_sources(all_dynamic_sources))
    if frame_whitelist:
        with dynamo_timed(name := "pgo.dynamic_whitelist", log_pt2_compile_event=True):
            CompileEventLogger.pt2_compile(
                name,
                recompile_dynamic_whitelist=frame_whitelist,
                missing_dynamic_whitelist=missing_whitelist,
            )


def _log_size_mismatch_recompile() -> None:
    global _LOGGED_DYNAMIC_ALLOWLIST
    if not _LOGGED_DYNAMIC_ALLOWLIST:
        torch._utils_internal.add_mlhub_insight(
            category="dynamic_shapes_analysis",
            insight="Dynamic shape recompilation detected",
            insight_description="PGO detected a recompilation due to dynamic shapes. \
            Please follow the instruction from the action link to reduce \
            recompilation overhead.",
        )
        # add mlhub insight only once per rank
        _LOGGED_DYNAMIC_ALLOWLIST = True


def render_code_state(cs: defaultdict[CodeId, CodeState]) -> str:
    code_state_str = "\n".join(
        f"{k}:\n"
        + "\n".join(
            f"  {src}: {fs.render()}" for src, fs in v.automatic_dynamic.items()
        )
        for k, v in cs.items()
    )
    dynamic_sources: OrderedSet[str] = OrderedSet()
    for state in cs.values():
        dynamic_sources.update(_collect_dynamic_sources(state))
    if dynamic_sources:
        code_state_str += (
            "\n\nPGO detected a recompilation due to dynamic shapes. "
            "To reduce shape recompilations by compiling dynamically to start, "
            f'set environment variable TORCH_COMPILE_DYNAMIC_SOURCES="{",".join(dynamic_sources)}"'
        )
    return code_state_str


@CacheArtifactFactory.register
class PGOCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None:
        meta = write_local_impl(
            self._rewrite_cache_key_for_mega_cache(self.key), self.content
        )
        if meta is None:
            raise AssertionError(
                "write_local_impl returned None during PGO cache population"
            )

    @override
    @staticmethod
    def type() -> str:
        return "pgo"

    @staticmethod
    def _rewrite_cache_key_for_mega_cache(original_key: str) -> str:
        """
        The PGO cache artifact key for a MAST job contains the job name and the version.
        When we want to use the cache artifact on a different MAST job, we need to
        update the key to use the new MAST job's name and version.
        """
        if not original_key.startswith("mast:"):
            # if original_key is overridden, then don't change it
            return original_key
        if (new_key := get_cache_key()) is not None:
            return new_key
        return original_key


def hit(key: str, ty: str) -> defaultdict[CodeId, CodeState]:
    global _INIT_CODE_STATE
    if not isinstance(_CODE_STATE, defaultdict):
        raise AssertionError(
            f"expected _CODE_STATE to be a defaultdict, got {type(_CODE_STATE)}"
        )
    log.info("get_code_state %s hit %s, %d entries", key, ty, len(_CODE_STATE))
    trace_structured_artifact(
        f"get_{ty}_code_state",
        "string",
        lambda: render_code_state(_CODE_STATE),  # type: ignore[arg-type]
    )
    set_feature_use("pgo", True)
    _INIT_CODE_STATE = copy.deepcopy(_CODE_STATE)
    return _CODE_STATE


def get_local_code_state(cache_key: str) -> defaultdict[CodeId, CodeState] | None:
    global _CODE_STATE
    path = code_state_path(cache_key)
    if path is not None and os.path.exists(path):
        with dynamo_timed(
            name := "pgo.get_local_code_state", log_pt2_compile_event=True
        ):
            CompileEventLogger.pt2_compile(name, cache_key=cache_key)
            # Read lock not necessary as we always write atomically write to
            # the actual location
            with open(path, "rb") as f:
                try:
                    content = f.read()
                    _CODE_STATE = pickle.loads(content)
                    CompileEventLogger.pt2_compile(name, cache_size_bytes=f.tell())
                except Exception:
                    log.warning(
                        "get_code_state failed while reading %s", path, exc_info=True
                    )
                else:
                    CacheArtifactRecorder(PGOCacheArtifact.type(), cache_key).record(
                        content
                    )
                    return hit(path, "local")
    return None


def lookup_remote_cache_entry(
    remote_cache: RemoteCache[JsonDataTy],
    cache_key: str,
    event_name: str | None = None,
) -> defaultdict[CodeId, CodeState] | None:
    code_state = None
    try:
        cache_data = remote_cache.get(cache_key)
    except Exception:
        log.warning("get_code_state failed remote read on %s", cache_key, exc_info=True)
    else:
        if cache_data is not None:
            try:
                if not isinstance(cache_data, dict):
                    raise AssertionError(
                        f"expected cache_data to be a dict, got {type(cache_data)}"
                    )
                data = cache_data["data"]
                if not isinstance(data, str):
                    raise AssertionError(
                        f"expected cache_data['data'] to be a str, got {type(data)}"
                    )
                payload = base64.b64decode(data)
                if event_name is not None:
                    CompileEventLogger.pt2_compile(
                        event_name, cache_size_bytes=len(payload)
                    )
                code_state = pickle.loads(payload)
            except Exception:
                log.warning(
                    "get_code_state failed parsing remote result on %s",
                    cache_key,
                    exc_info=True,
                )
            else:
                CacheArtifactRecorder(PGOCacheArtifact.type(), cache_key).record(
                    payload
                )
        else:
            log.info("get_code_state remote miss on %s", cache_key)
    return code_state


def get_remote_code_state(cache_key: str) -> defaultdict[CodeId, CodeState] | None:
    global _CODE_STATE
    remote_cache = get_remote_cache()
    if remote_cache is not None:
        with dynamo_timed(
            name := "pgo.get_remote_code_state",
            log_pt2_compile_event=True,
            dynamo_compile_column_us="pgo_get_remote_code_state_time_us",
        ):
            CompileEventLogger.pt2_compile(name, cache_key=cache_key)
            code_state = lookup_remote_cache_entry(remote_cache, cache_key, name)
            if code_state is not None:
                _CODE_STATE = code_state
                return hit(cache_key, "remote")
    return None


def get_extra_remote_code_state(cache_key: str) -> None:
    """
    Reads an additional PGO profile from the given cache key, and merges it with the default PGO profile.
    """
    global _CODE_STATE
    if _CODE_STATE is None:
        raise AssertionError(
            "_CODE_STATE must be initialized before reading extra remote code state"
        )

    remote_cache = get_remote_cache()
    if remote_cache is not None:
        with dynamo_timed(
            name := "pgo.get_extra_remote_code_state",
            log_pt2_compile_event=True,
            dynamo_compile_column_us="pgo_get_remote_code_state_time_us",
        ):
            CompileEventLogger.pt2_compile(name, cache_key=cache_key)
            code_state = lookup_remote_cache_entry(remote_cache, cache_key)
            log.info(
                "get_extra_code_state %s hit, %d entries",
                cache_key,
                len(code_state) if code_state is not None else 0,
            )
            if code_state is not None:
                if _CODE_STATE:
                    raise AssertionError(
                        "expected _CODE_STATE to be empty before merging extra remote state"
                    )
                _CODE_STATE = code_state
                # log to tlparse
                trace_structured_artifact(
                    "get_extra_remote_code_state",
                    "string",
                    lambda: render_code_state(code_state),
                )


def get_code_state() -> defaultdict[CodeId, CodeState]:
    global _CODE_STATE, _INIT_CODE_STATE
    if _CODE_STATE is not None:
        return _CODE_STATE

    # Initialize it (even if we don't look up profile)
    _CODE_STATE = defaultdict(CodeState)

    cache_key = get_cache_key()
    if cache_key is None:
        return _CODE_STATE

    # Attempt local
    local_code_state = get_local_code_state(cache_key)

    # Attempt remote
    if local_code_state is None:
        get_remote_code_state(cache_key)

    # Attempt additional remote if neither local/default remote succeeded
    if (
        not _CODE_STATE
        and (sticky_read := torch.compiler.config.pgo_extra_read_key) is not None
    ):
        extra_read_key = get_extra_cache_key(sticky_read)
        if extra_read_key is not None:
            get_extra_remote_code_state(extra_read_key)

    log.info("get_code_state using default")

    if _CODE_STATE is None:
        raise AssertionError("_CODE_STATE should have been initialized above")
    return _CODE_STATE


def put_code_state() -> None:
    if _CODE_STATE is None:
        log.info("put_code_state: never initialized, will not write")
        return

    if _CODE_STATE == _INIT_CODE_STATE:
        log.info("put_code_state: no change, skipping")
        return

    cache_key = get_cache_key()
    if cache_key is None:
        log.info("put_code_state: no cache key, skipping")
        return

    put_local_code_state(cache_key)
    put_remote_code_state(cache_key)
    if (sticky_write := torch.compiler.config.pgo_extra_write_key) is not None:
        extra_write_key = get_extra_cache_key(sticky_write)
        if extra_write_key is not None:
            put_remote_code_state(extra_write_key)


def write_local_impl(cache_key: str, pickled_code: bytes) -> tuple[str, int] | None:
    path = code_state_path(cache_key)

    if path is None:
        return None

    # If the user isn't misusing our API, we should have exclusive access to
    # this directory.  But it's not too hard

    tmp_path = path + ".tmp"
    lock_path = path + ".lock"
    # We /mostly/ don't need the lock but the tmp file could be clobbered
    # TODO: use a safe tempfile create to eliminate lock
    from torch.utils._filelock import FileLock

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with FileLock(lock_path, timeout=LOCK_TIMEOUT):
        with open(tmp_path, "wb") as f:
            f.write(pickled_code)
            size = f.tell()
        os.replace(tmp_path, path)
    return path, size


def put_local_code_state(cache_key: str) -> None:
    with dynamo_timed(name := "pgo.put_local_code_state", log_pt2_compile_event=True):
        CompileEventLogger.pt2_compile(name, cache_key=cache_key)
        if _CODE_STATE is None:
            raise AssertionError(
                "_CODE_STATE must be initialized before writing local code state"
            )

        pickled_code = pickle.dumps(_CODE_STATE)

        CacheArtifactRecorder(PGOCacheArtifact.type(), cache_key).record(pickled_code)

        meta = write_local_impl(cache_key, pickled_code)
        if meta is None:
            log.info("put_code_state: local cache disabled")
            return
        path, size = meta

        CompileEventLogger.pt2_compile(name, cache_size_bytes=size)
        log.info("put_code_state: wrote local %s, %d entries", path, len(_CODE_STATE))
        trace_structured_artifact(
            "put_local_code_state",
            "string",
            lambda: render_code_state(_CODE_STATE),
        )


def put_remote_code_state(cache_key: str, extra_code_state: bool = False) -> None:
    event_name = (
        "put_remote_code_state"
        if not extra_code_state
        else "put_extra_remote_code_state"
    )
    with dynamo_timed(
        name := f"pgo.{event_name}",
        log_pt2_compile_event=True,
        dynamo_compile_column_us="pgo_put_remote_code_state_time_us",
    ):
        CompileEventLogger.pt2_compile(name, cache_key=cache_key)
        if _CODE_STATE is None:
            raise AssertionError(
                "_CODE_STATE must be initialized before writing remote code state"
            )

        remote_cache = get_remote_cache()

        if remote_cache is None:
            log.info("%s: remote cache disabled", event_name)
            return

        content = pickle.dumps(_CODE_STATE)
        CompileEventLogger.pt2_compile(name, cache_size_bytes=len(content))
        cache_data: JsonDataTy = {
            "data": base64.b64encode(content).decode("ascii"),
        }
        remote_cache.put(cache_key, cache_data)
        log.info(
            "%s: wrote remote %s, %d entries", event_name, cache_key, len(_CODE_STATE)
        )
        # TODO: don't log this multiple times
        trace_structured_artifact(
            event_name,
            "string",
            lambda: render_code_state(_CODE_STATE),
        )


# NB: this does NOT reset the cached code state on disk
def reset_code_state() -> None:
    global _CODE_STATE, _INIT_CODE_STATE, _LOGGED_DYNAMIC_ALLOWLIST
    _CODE_STATE = None
    _INIT_CODE_STATE = None
    _LOGGED_DYNAMIC_ALLOWLIST = False
<|endoftext|>"""
This module provides utilities for analyzing, transforming and manipulating Python bytecode.
It includes functionality for:
- Converting between different bytecode formats and versions
- Virtualizing jumps and managing jump targets
- Handling exception tables and their entries
- Managing instruction offsets and extended arguments
- Providing a clean API for bytecode modification and transformation
- Supporting Python version-specific bytecode features
- Generating bytecode from template functions

The module is designed to work across different Python versions (3.7+) and handles
version-specific bytecode differences transparently.
"""

import copy
import dataclasses
import dis
import functools
import itertools
import re
import sys
import types
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, cast, Optional, TYPE_CHECKING

from . import config
from .bytecode_analysis import (
    get_indexof,
    propagate_line_nums,
    remove_extra_line_nums,
    stacksize_analysis,
)
from .utils import is_safe_constant


if TYPE_CHECKING:
    from .output_graph import DynamoTracerOutput


@dataclasses.dataclass(slots=True)
class InstructionExnTabEntry:
    start: "Instruction"
    end: "Instruction"
    target: "Instruction"
    depth: int
    lasti: bool

    def __repr__(self) -> str:
        return (
            f"InstructionExnTabEntry(start={self.start.short_inst_repr()}, "
            f"end={self.end.short_inst_repr()}, "
            f"target={self.target.short_inst_repr()}, "
            f"depth={self.depth}, lasti={self.lasti})"
        )

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, InstructionExnTabEntry):
            return False
        return (
            self.start is o.start
            and self.end is o.end
            and self.target is o.target
            and self.depth == o.depth
            and self.lasti == o.lasti
        )


@dataclasses.dataclass(slots=True)
class Instruction:
    """A mutable version of dis.Instruction"""

    opcode: int
    opname: str
    arg: int | None
    argval: Any
    offset: int | None = None
    starts_line: int | None = None
    is_jump_target: bool = False

    positions: Optional["dis.Positions"] = None
    # extra fields to make modification easier:
    target: Optional["Instruction"] = None
    exn_tab_entry: InstructionExnTabEntry | None = None
    argrepr: str | None = None

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)

    def short_inst_repr(self) -> str:
        return f"Instruction(opname={self.opname}, offset={self.offset})"

    def copy_positions(self, other: "Instruction") -> None:
        self.starts_line = other.starts_line
        self.positions = other.positions


if sys.version_info >= (3, 13):

    def convert_instruction(i: dis.Instruction) -> Instruction:
        return Instruction(
            i.opcode,
            i.opname,
            i.arg,
            i.argval,
            i.offset,
            i.line_number,
            i.is_jump_target,
            i.positions,
        )

elif sys.version_info >= (3, 11):

    def convert_instruction(i: dis.Instruction) -> Instruction:
        return Instruction(
            i.opcode,
            i.opname,
            i.arg,
            i.argval,
            i.offset,
            i.starts_line,
            i.is_jump_target,
            i.positions,
        )

else:

    def convert_instruction(i: dis.Instruction) -> Instruction:
        return Instruction(
            i.opcode,
            i.opname,
            i.arg,
            i.argval,
            i.offset,
            i.starts_line,
            i.is_jump_target,
            None,
        )


class _NotProvided:
    def __repr__(self) -> str:
        return "_NotProvided"


if sys.version_info >= (3, 12):

    def inst_has_op_bits(name: str) -> bool:
        return name in ("LOAD_ATTR", "LOAD_GLOBAL", "LOAD_SUPER_ATTR")

elif sys.version_info >= (3, 11):

    def inst_has_op_bits(name: str) -> bool:
        return name == "LOAD_GLOBAL"

else:

    def inst_has_op_bits(name: str) -> bool:
        return False


def create_instruction(
    name: str,
    *,
    arg: int | None = None,
    argval: Any | None = _NotProvided,
    target: Instruction | None = None,
) -> Instruction:
    """
    At most one of `arg`, `argval`, and `target` can be not None/_NotProvided.
    This is to prevent ambiguity, e.g. does
        create_instruction("LOAD_CONST", 5)
    mean load the constant at co_consts[5], or load the constant 5?

    If `arg` is not provided, it will be computed during assembly from
    `argval` or `target`.

    Bits in the args of instructions LOAD_GLOBAL, LOAD_ATTR (3.12+), and LOAD_SUPER_ATTR
    modify the behavior of the instruction. In this case, we allow both `arg`
    and `argval` to be set. The value of `arg` here is expected to be the value of
    the op bits and the true value of `arg` will be computed during assembly.
    If `arg` is not set, the bits are assumed to be 0.
    """

    # allow for instructions with op bits to have both arg and argval specified
    if inst_has_op_bits(name):
        if target is not None:
            raise RuntimeError("target cannot be specified for instruction")
        if arg is None:
            arg = 0
    else:
        cnt = (arg is not None) + (argval is not _NotProvided) + (target is not None)
        if cnt > 1:
            raise RuntimeError(
                "only one of arg, argval, and target can be not None/_NotProvided"
            )
    if arg is not None and not isinstance(arg, int):
        raise RuntimeError("instruction arg must be int or None")
    return Instruction(
        opcode=dis.opmap[name], opname=name, arg=arg, argval=argval, target=target
    )


# Python 3.11 remaps
def create_jump_absolute(target: Instruction) -> Instruction:
    inst = "JUMP_FORWARD" if sys.version_info >= (3, 11) else "JUMP_ABSOLUTE"
    return create_instruction(inst, target=target)


def is_jump_absolute(target: Instruction) -> bool:
    return target.opname in ("JUMP_FORWARD", "JUMP_ABSOLUTE")


def create_load_const(val: Any, checked: bool = True) -> Instruction:
    """
    In general we should only create `LOAD_CONST` for immutable objects, but
    sometimes it's convenient _and safe_ for Dynamo create `LOAD_CONST` for
    mutable objects. In such cases, use `checked=False`.
    """
    if checked:
        if not is_safe_constant(val):
            raise AssertionError(f"unsafe constant {val}")
    return create_instruction("LOAD_CONST", argval=val)


def create_breakpoint() -> list[Instruction]:
    """
    Create instructions that trigger the bytecode debugger to stop.

    Usage:
        codegen.extend_output(create_breakpoint())

    When the bytecode debugger is active, execution will pause at this point.
    At runtime, these instructions have no effect on program state.
    """
    from .bytecode_debugger import BREAKPOINT_MARKER

    return [
        create_instruction("LOAD_CONST", argval=BREAKPOINT_MARKER),
        create_instruction("POP_TOP"),
    ]


def create_dup_top() -> Instruction:
    if sys.version_info >= (3, 11):
        return create_instruction("COPY", arg=1)
    return create_instruction("DUP_TOP")


def create_rot_n(n: int) -> list[Instruction]:
    """
    Returns a "simple" sequence of instructions that rotates TOS to the n-th
    position in the stack. For Python < 3.11, returns a single ROT_*
    instruction. If no such instruction exists, an error is raised and the
    caller is expected to generate an equivalent sequence of instructions.
    For Python >= 3.11, any rotation can be expressed as a simple sequence of
    swaps.
    """
    if n <= 1:
        # don't rotate
        return []

    if sys.version_info >= (3, 11):
        # rotate can be expressed as a sequence of swap operations
        # e.g. rotate 3 is equivalent to swap 3, swap 2
        return [create_instruction("SWAP", arg=i) for i in range(n, 1, -1)]

    if n <= 4:
        return [create_instruction("ROT_" + ["TWO", "THREE", "FOUR"][n - 2])]
    return [create_instruction("ROT_N", arg=n)]


def add_push_null(
    inst_or_insts: Instruction | list[Instruction],
) -> list[Instruction]:
    """
    Appends or prepends a PUSH_NULL instruction to `inst_or_insts`,
    depending on Python version. Used when you know that
    `inst_or_insts` generates a callable that will be called.

    NOTE: Assumes `inst_or_insts` is a single instruction or sequence of
    instructions that pushes exactly 1 object to the stack that is to
    be called. It is important that you include ALL instructions that
    construct the callable - not just the first instruction/a prefix.

    Will attempt to use the NULL push bit for instructions
    with such bits (LOAD_GLOBAL 3.11+, LOAD_ATTR 3.12+, LOAD_SUPER_ATTR).
    In this case, instructions WILL be modified.
    """
    if isinstance(inst_or_insts, Instruction):
        insts: list[Instruction] = [inst_or_insts]
    else:
        if not isinstance(inst_or_insts, list):
            raise AssertionError(
                f"expected inst_or_insts to be a list, got {type(inst_or_insts)}"
            )
        insts = inst_or_insts

    def inst_has_bit_set(idx: int) -> bool:
        if insts[idx].arg is None:
            raise AssertionError(f"expected insts[{idx}].arg to be not None")
        return insts[idx].arg & 1 == 1  # type: ignore[operator]

    def set_inst_bit(idx: int) -> None:
        if insts[idx].arg is None:
            raise AssertionError(f"expected insts[{idx}].arg to be not None")
        insts[idx].arg |= 1  # type: ignore[operator]

    if sys.version_info >= (3, 13):
        # In 3.13, NULL follows the callable
        if inst_has_op_bits(insts[-1].opname) and not inst_has_bit_set(-1):
            # All insts with op bits have the push_null bit as the last one.
            # Only set the bit if it hasn't been set - otherwise, we need
            # to add another PUSH_NULL.
            set_inst_bit(-1)
        else:
            insts = insts + [create_instruction("PUSH_NULL")]
    elif sys.version_info >= (3, 12):
        # LOAD_ATTR/LOAD_SUPER_ATTR at the end
        # We assume that `insts` will only load 1 object, so
        # LOAD_GLOBAL at the end doesn't need to be checked
        if inst_has_op_bits(insts[-1].opname) and not inst_has_bit_set(-1):
            set_inst_bit(-1)
        elif insts[0].opname == "LOAD_GLOBAL" and not inst_has_bit_set(0):
            set_inst_bit(0)
        else:
            insts = [create_instruction("PUSH_NULL")] + insts
    elif sys.version_info >= (3, 11):
        # 3.11 introduced NULL preceding callable
        if inst_has_op_bits(insts[0].opname) and not inst_has_bit_set(0):
            set_inst_bit(0)
        else:
            insts = [create_instruction("PUSH_NULL")] + insts
    return insts


def add_push_null_call_function_ex(
    inst_or_insts: Instruction | list[Instruction],
) -> list[Instruction]:
    """Like add_push_null, but the low bit of LOAD_ATTR/LOAD_SUPER_ATTR
    is not set, due to an expected CALL_FUNCTION_EX instruction.
    """
    if isinstance(inst_or_insts, Instruction):
        insts: list[Instruction] = [inst_or_insts]
    else:
        if not isinstance(inst_or_insts, list):
            raise AssertionError(
                f"expected inst_or_insts to be a list, got {type(inst_or_insts)}"
            )
        insts = inst_or_insts

    if sys.version_info < (3, 11):
        return insts

    idx = -1 if sys.version_info >= (3, 13) else 0
    if insts[idx].opname == "LOAD_GLOBAL":
        if insts[idx].arg is None:
            raise AssertionError(
                f"expected insts[{idx}].arg to be not None for LOAD_GLOBAL"
            )
        if insts[idx].arg & 1 == 0:  # type: ignore[operator]
            insts[idx].arg |= 1  # type: ignore[operator]
            return insts

    if sys.version_info >= (3, 13):
        insts = insts + [create_instruction("PUSH_NULL")]
    else:
        insts = [create_instruction("PUSH_NULL")] + insts

    return insts


def create_call_function(nargs: int, push_null: bool) -> list[Instruction]:
    """
    Creates a sequence of instructions that makes a function call.

    `push_null` is used in Python 3.11+ only. It is used in codegen when
    a function call is intended to be made with the NULL + fn convention,
    and we know that the NULL has not been pushed yet. We will push a
    NULL and rotate it to the correct position immediately before making
    the function call.

    `push_null` should be True if no NULL is pushed for the callable.
    Conversely, `push_null` should be False if a NULL was pushed for the callable.
    Prefer using `push_null=False` when possible since we will not need to rotate
    NULL to the right place, which is less efficient.

    Generally, you should codegen a function by using `add_push_null` then
    `create_call_function` with `push_null=False`.

    Example of when to set push_null False:

    insts = [
        create_instruction("LOAD_GLOBAL", argval="torch"),
        create_instruction("LOAD_ATTR", argval="nn"),
        create_instruction("LOAD_ATTR", argval="functional"),
        create_instruction("LOAD_ATTR", argval="relu"),
    ]
    insts = add_push_null(insts)
    insts.append(create_instruction("LOAD_FAST", argval="x"))
    insts.extend(create_call_function(1, False))

    Example of when to set push_null True:

    insts = [create_instruction("LOAD_FAST", x)]
    for should_wrap, wrapper_name in wrappers:
        if should_wrap:
            insts.extend([
                create_instruction("LOAD_GLOBAL", argval="wrapper1"),
                create_instruction("SWAP", arg=2),
                *create_call_function(1, True),
            )
    """
    if sys.version_info >= (3, 11):
        output = []
        if push_null:
            output.append(create_instruction("PUSH_NULL"))
            # 3.13 swapped NULL and callable
            rots = nargs + 1 if sys.version_info >= (3, 13) else nargs + 2
            output.extend(create_rot_n(rots))
        if sys.version_info < (3, 12):
            output.append(create_instruction("PRECALL", arg=nargs))
        output.append(create_instruction("CALL", arg=nargs))
        return output
    return [create_instruction("CALL_FUNCTION", arg=nargs)]


def create_call_function_ex(
    has_kwargs: bool, push_null: bool, ignore_314_kwargs_push: bool = False
) -> list[Instruction]:
    """
    Assumes that in 3.14+, if has_kwargs=False, there is NOT a NULL
    on the TOS for the kwargs. This utility function will add a PUSH_NULL.

    If the caller has already pushed a NULL for the kwargs, then set ignore_314_kwargs_push=True
    so we don't push another NULL for the kwargs.
    """
    if sys.version_info >= (3, 11):
        output = []
        if (
            sys.version_info >= (3, 14)
            and not has_kwargs
            and not ignore_314_kwargs_push
        ):
            output.append(create_instruction("PUSH_NULL"))
            has_kwargs = True
        if push_null:
            output.append(create_instruction("PUSH_NULL"))
            # 3.13 swapped NULL and callable
            # if flags == 1, 2 values popped - otherwise if flags == 0, 1 value
            rots = (
                int(has_kwargs) + 2
                if sys.version_info >= (3, 13)
                else int(has_kwargs) + 3
            )
            output.extend(create_rot_n(rots))
        output.append(create_instruction("CALL_FUNCTION_EX", arg=int(has_kwargs)))
        return output
    return [create_instruction("CALL_FUNCTION_EX", arg=int(has_kwargs))]


def create_call_method(nargs: int) -> list[Instruction]:
    if sys.version_info >= (3, 12):
        return [create_instruction("CALL", arg=nargs)]
    if sys.version_info >= (3, 11):
        return [
            create_instruction("PRECALL", arg=nargs),
            create_instruction("CALL", arg=nargs),
        ]
    return [create_instruction("CALL_METHOD", arg=nargs)]


def create_load_method(name: str) -> Instruction:
    if sys.version_info >= (3, 12):
        # in 3.12, create a LOAD_ATTR instruction with the low bit set
        return create_instruction("LOAD_ATTR", arg=1, argval=name)
    return create_instruction("LOAD_METHOD", argval=name)


def create_setup_with(target: Instruction) -> Instruction:
    opname = "BEFORE_WITH" if sys.version_info >= (3, 11) else "SETUP_WITH"
    return create_instruction(opname, target=target)


def create_swap(n: int) -> list[Instruction]:
    if sys.version_info >= (3, 11):
        return [create_instruction("SWAP", arg=n)]
    # in Python < 3.11, SWAP is a macro that expands to multiple instructions
    if n == 1:
        return []
    elif n == 2:
        return [create_instruction("ROT_TWO")]
    elif n == 3:
        return [create_instruction("ROT_THREE"), create_instruction("ROT_TWO")]
    """
    e.g. swap "a" and "b" in this stack:
    0 a 1 2 3 b
    0 a [1 2 3 b]
    0 a [1 2 3 b] [1 2 3 b]
    0 a [1 2 3 b] [1 2 3 b] -1
    0 a [1 2 3 b] b
    0 b a [1 2 3 b]
    0 b a [1 2 3 b] [1 2 3 b]
    0 b [1 2 3 b] a [1 2 3 b]
    0 b [1 2 3 b] a [1 2 3 b] -1
    0 b [1 2 3 a]
    0 b [1 2 3 a] [1 2 3 a]
    0 b [1 2 3 a] [1 2 3 a] reverse
    0 b [a 3 2 1] None
    0 b [a 3 2 1]
    0 b 1 2 3 a
    """
    return [
        create_instruction("BUILD_LIST", arg=n - 1),
        create_instruction("DUP_TOP"),
        create_instruction("LOAD_CONST", argval=-1),
        create_binary_subscr(),
        create_instruction("ROT_THREE"),
        create_instruction("DUP_TOP"),
        create_instruction("ROT_THREE"),
        create_instruction("LOAD_CONST", argval=-1),
        create_instruction("STORE_SUBSCR"),
        create_instruction("DUP_TOP"),
        create_load_method("reverse"),
        *create_call_method(0),
        create_instruction("POP_TOP"),
        create_instruction("UNPACK_SEQUENCE", arg=n - 1),
    ]


def get_call_callable_depth(opname: str, arg: int) -> int:
    """Return the depth of the callable from TOS for a call instruction.

    Depth is 1-indexed: TOS = 1, below-TOS = 2, etc.  This encodes the
    per-version stack layouts for every call instruction variant.

    Note: on 3.11-3.12, CPython's CALL can have either NULL or self at the
    bottom depending on function vs method call.  Dynamo's LOAD_METHOD
    normalizes method calls to NULL + bound_method, so the bottom is always
    NULL and the callable is always at depth arg + 1.

    Stack layouts (bottom-to-top within the call's frame):
      CALL (3.13+):            callable  NULL  arg0 ... argN-1
      CALL (3.11-3.12):        NULL  callable  arg0 ... argN-1
      CALL_KW (3.13+):         callable  NULL  arg0 ... argN-1  kw_names
      CALL_FUNCTION (3.10):    callable  arg0 ... argN-1
      CALL_FUNCTION_KW (3.10): callable  arg0 ... argN-1  kw_names
      CALL_FUNCTION_EX:
        3.14+:                 callable  NULL  args_tuple  kwargs_or_null
        3.13:                  callable  NULL  args_tuple [kwargs_dict]
        3.11-3.12:             NULL  callable  args_tuple [kwargs_dict]
        3.10:                  callable  args_tuple [kwargs_dict]
    """
    if opname in ("CALL", "CALL_KW"):
        kw_extra = 1 if opname == "CALL_KW" else 0
        if sys.version_info >= (3, 13):
            return arg + 2 + kw_extra
        else:
            # 3.11-3.12: NULL is always at bottom (Dynamo normalizes
            # LOAD_METHOD to NULL + bound_method)
            return arg + 1 + kw_extra
    elif opname in ("CALL_FUNCTION", "CALL_FUNCTION_KW"):
        kw_extra = 1 if opname == "CALL_FUNCTION_KW" else 0
        return arg + 1 + kw_extra
    elif opname == "CALL_FUNCTION_EX":
        if sys.version_info >= (3, 14):
            return 4
        elif sys.version_info >= (3, 13):
            return 3 + (arg & 1)
        elif sys.version_info >= (3, 11):
            return 2 + (arg & 1)
        else:
            return 2 + (arg & 1)
    else:
        raise ValueError(f"not a call instruction: {opname}")


def create_binary_slice(
    start: int | None, end: int | None, store: bool = False
) -> list[Instruction]:
    """
    BINARY_SLICE and STORE_SLICE (if `set` is True) for all Python versions
    """
    if sys.version_info >= (3, 14):
        subscr_inst = (
            create_instruction("STORE_SUBSCR") if store else create_binary_subscr()
        )
        return [
            create_load_const(slice(start, end)),
            subscr_inst,
        ]
    elif sys.version_info >= (3, 12):
        inst_name = "STORE_SLICE" if store else "BINARY_SLICE"
        return [
            create_load_const(start),
            create_load_const(end),
            create_instruction(inst_name),
        ]
    else:
        inst_name = "STORE_SUBSCR" if store else "BINARY_SUBSCR"
        return [
            create_load_const(start),
            create_load_const(end),
            create_instruction("BUILD_SLICE", arg=2),
            create_instruction(inst_name),
        ]


def create_copy(i: int) -> list[Instruction]:
    if sys.version_info >= (3, 11):
        return [create_instruction("COPY", arg=i)]
    if i == 1:
        return [create_instruction("DUP_TOP")]
    # COPY 4
    # 0 1 2 3
    # 3 1 2 0
    # 3 1 2 0 0
    # 0 1 2 0 3
    # 0 1 2 3 0
    return [
        *create_swap(i),
        create_dup_top(),
        *create_swap(i + 1),
        *create_swap(2),
    ]


# mainly for debugging generated bytecode
def create_print_on_stack(depth: int) -> list[Instruction]:
    return [
        *add_push_null(create_instruction("LOAD_CONST", argval=print)),
        *create_copy(depth + (2 if sys.version_info >= (3, 11) else 1)),
        *create_call_function(1, False),
        create_instruction("POP_TOP"),
    ]


# mainly for debugging generated bytecode
def create_print_value(value: Any) -> list[Instruction]:
    return [
        *add_push_null(create_instruction("LOAD_CONST", argval=print)),
        create_instruction("LOAD_CONST", argval=value),
        *create_call_function(1, False),
        create_instruction("POP_TOP"),
    ]


def create_binary_subscr() -> Instruction:
    if sys.version_info < (3, 14):
        return create_instruction("BINARY_SUBSCR")
    # https://github.com/python/cpython/blob/0e46c0499413bc5f9f8336fe76e2e67cf93f64d8/Include/opcode.h#L36
    return create_instruction("BINARY_OP", arg=26)


def create_build_tuple(n: int) -> Instruction:
    if sys.version_info >= (3, 14) and n == 0:
        return create_load_const(())
    return create_instruction("BUILD_TUPLE", arg=n)


def linetable_writer(
    first_lineno: int,
) -> tuple[list[int], Callable[[int, int], None], Callable[[int], None]]:
    """
    Used to create typing.CodeType.co_linetable
    See https://github.com/python/cpython/blob/3.10/Objects/lnotab_notes.txt
    This is the internal format of the line number table for Python 3.10
    """
    if sys.version_info[:2] != (3, 10):
        raise AssertionError(
            f"linetable_writer requires Python 3.10, got {sys.version_info[:2]}"
        )
    linetable: list[int] = []
    lineno = first_lineno
    lineno_delta = 0
    byteno = 0

    def _update(byteno_delta: int, lineno_delta: int) -> None:
        while byteno_delta != 0 or lineno_delta != 0:
            byte_offset = max(0, min(byteno_delta, 254))
            line_offset = max(-127, min(lineno_delta, 127))
            if byte_offset == 0 and line_offset == 0:
                raise AssertionError("byte_offset and line_offset cannot both be 0")
            byteno_delta -= byte_offset
            lineno_delta -= line_offset
            linetable.extend((byte_offset, line_offset & 0xFF))

    def update(lineno_new: int, byteno_new: int) -> None:
        nonlocal lineno, lineno_delta, byteno
        byteno_delta = byteno_new - byteno
        byteno = byteno_new
        _update(byteno_delta, lineno_delta)
        lineno_delta = lineno_new - lineno
        lineno = lineno_new

    def end(total_bytes: int) -> None:
        _update(total_bytes - byteno, lineno_delta)

    return linetable, update, end


def encode_varint(n: int) -> list[int]:
    """
    6-bit chunk encoding of an unsigned integer
    See https://github.com/python/cpython/blob/3.11/Objects/locations.md
    """
    if n < 0:
        raise AssertionError(f"encode_varint requires non-negative n, got {n}")
    b = [n & 63]
    n >>= 6
    while n > 0:
        b[-1] |= 64
        b.append(n & 63)
        n >>= 6
    return b


if sys.version_info >= (3, 11):

    def linetable_311_writer(
        first_lineno: int,
    ) -> tuple[list[int], Callable[[dis.Positions | None, int], None]]:
        """
        Used to create typing.CodeType.co_linetable
        See https://github.com/python/cpython/blob/3.11/Objects/locations.md
        This is the internal format of the line number table for Python 3.11
        """
        if sys.version_info < (3, 11):
            raise AssertionError(
                f"linetable_311_writer requires Python 3.11+, got {sys.version_info}"
            )
        linetable = []
        lineno = first_lineno

        def update(positions: dis.Positions | None, inst_size: int) -> None:
            nonlocal lineno
            lineno_new = positions.lineno if positions else None

            def _update(delta: int, size: int) -> None:
                if not (0 < size <= 8):
                    raise AssertionError(f"size must be in range (0, 8], got {size}")
                # first byte - use 13 (no column info) is positions is
                # malformed, otherwise use 14 (long form)
                other_varints: tuple[int, ...] = ()
                if (
                    positions
                    and positions.lineno is not None
                    and positions.end_lineno is not None
                    and positions.col_offset is not None
                    and positions.end_col_offset is not None
                ):
                    linetable.append(0b1_1110_000 + size - 1)
                    # for whatever reason, column offset needs `+ 1`
                    # https://github.com/python/cpython/blob/1931c2a438c50e6250725c84dff94fc760b9b951/Python/compile.c#L7603
                    other_varints = (
                        positions.end_lineno - positions.lineno,
                        positions.col_offset + 1,
                        positions.end_col_offset + 1,
                    )
                else:
                    linetable.append(0b1_1101_000 + size - 1)
                # encode signed int
                if delta < 0:
                    delta = ((-delta) << 1) | 1
                else:
                    delta <<= 1
                # encode unsigned int
                linetable.extend(encode_varint(delta))
                for n in other_varints:
                    linetable.extend(encode_varint(n))

            if lineno_new is None:
                lineno_delta = 0
            else:
                lineno_delta = lineno_new - lineno
                lineno = lineno_new
            while inst_size > 8:
                _update(lineno_delta, 8)
                inst_size -= 8
            _update(lineno_delta, inst_size)

        return linetable, update


@dataclasses.dataclass(slots=True)
class ExceptionTableEntry:
    start: int
    end: int
    target: int
    depth: int
    lasti: bool


def encode_exception_table_varint(n: int) -> list[int]:
    """
    Similar to `encode_varint`, but the 6-bit chunks are ordered in reverse.
    """
    if n < 0:
        raise AssertionError(
            f"encode_exception_table_varint requires non-negative n, got {n}"
        )
    b = [n & 63]
    n >>= 6
    while n > 0:
        b.append(n & 63)
        n >>= 6
    b.reverse()
    for i in range(len(b) - 1):
        b[i] |= 64
    return b


def decode_exception_table_varint(bytes_iter: Iterator[int]) -> int:
    """
    Inverse of `encode_exception_table_varint`.
    """
    b = next(bytes_iter)
    val = b & 63
    while b & 64:
        val <<= 6
        b = next(bytes_iter)
        val |= b & 63
    return val


def check_exception_table(tab: list[ExceptionTableEntry]) -> None:
    """
    Verifies that a list of ExceptionTableEntries will make a well-formed
    jump table: entries are non-empty, sorted, and do not overlap.
    """
    for i in range(len(tab) - 1):
        if not (
            tab[i].start <= tab[i].end
            and tab[i].end < tab[i + 1].start
            and tab[i + 1].start <= tab[i + 1].end
        ):
            raise AssertionError(
                f"exception table entries are not well-formed at index {i}: "
                f"entry[{i}]=(start={tab[i].start}, end={tab[i].end}), "
                f"entry[{i + 1}]=(start={tab[i + 1].start}, end={tab[i + 1].end})"
            )


def parse_exception_table(exntab: bytes) -> list[ExceptionTableEntry]:
    """
    Parse the exception table according to
    https://github.com/python/cpython/blob/3.11/Objects/exception_handling_notes.txt
    """
    exntab_iter = iter(exntab)
    tab = []
    try:
        while True:
            start = decode_exception_table_varint(exntab_iter) * 2
            length = decode_exception_table_varint(exntab_iter) * 2
            end = start + length - 2
            target = decode_exception_table_varint(exntab_iter) * 2
            dl = decode_exception_table_varint(exntab_iter)
            depth = dl >> 1
            lasti = bool(dl & 1)
            tab.append(ExceptionTableEntry(start, end, target, depth, lasti))
    except StopIteration:
        check_exception_table(tab)
        return tab


def assemble_exception_table(tab: list[ExceptionTableEntry]) -> bytes:
    """
    Inverse of parse_exception_table - encodes list of exception
    table entries into bytes.
    """
    b = []
    for entry in tab:
        first_entry = encode_exception_table_varint(entry.start // 2)
        first_entry[0] |= 1 << 7
        b.extend(first_entry)
        length = entry.end - entry.start + 2
        b.extend(encode_exception_table_varint(length // 2))
        b.extend(encode_exception_table_varint(entry.target // 2))
        dl = (entry.depth << 1) + entry.lasti
        b.extend(encode_exception_table_varint(dl))
    return bytes(b)


def assemble(instructions: list[Instruction], firstlineno: int) -> tuple[bytes, bytes]:
    """Do the opposite of dis.get_instructions()"""
    code: list[int] = []
    if sys.version_info >= (3, 11):
        lnotab, update_lineno = linetable_311_writer(firstlineno)
        num_ext = 0
        for i, inst in enumerate(instructions):
            if inst.opname == "EXTENDED_ARG":
                inst_size = 1
                num_ext += 1
                # copy positions from the actual instruction
                for j in (1, 2, 3):
                    if instructions[i + j].opname != "EXTENDED_ARG":
                        inst.positions = instructions[i + j].positions
                        break
            else:
                inst_size = instruction_size(inst) // 2 + num_ext
                num_ext = 0
            update_lineno(inst.positions, inst_size)
            num_ext = 0
            arg = inst.arg or 0
            code.extend((inst.opcode, arg & 0xFF))
            for _ in range(instruction_size(inst) // 2 - 1):
                code.extend((0, 0))
    else:
        lnotab, update_lineno, end = linetable_writer(firstlineno)

        for inst in instructions:
            if inst.starts_line is not None:
                update_lineno(inst.starts_line, len(code))
            arg = inst.arg or 0
            code.extend((inst.opcode, arg & 0xFF))

        end(len(code))

    return bytes(code), bytes(lnotab)


def _get_instruction_by_offset(
    offset_to_inst: dict[int, Instruction], offset: int
) -> Instruction | None:
    """
    Get the instruction located at a given offset, accounting for EXTENDED_ARGs
    """
    for n in (0, 2, 4, 6):
        if offset_to_inst[offset + n].opcode != dis.EXTENDED_ARG:
            return offset_to_inst[offset + n]
    return None


def virtualize_jumps(instructions: Iterable[Instruction]) -> None:
    """Replace jump targets with pointers to make editing easier"""
    jump_targets = {
        inst.offset: inst for inst in instructions if inst.offset is not None
    }

    for inst in instructions:
        if inst.opcode in dis.hasjabs or inst.opcode in dis.hasjrel:
            inst.target = _get_instruction_by_offset(jump_targets, inst.argval)


_REL_JUMPS = set(dis.hasjrel)


def flip_jump_direction(instruction: Instruction) -> None:
    if sys.version_info < (3, 11):
        raise RuntimeError("Cannot flip jump direction in Python < 3.11")
    if "FORWARD" in instruction.opname:
        instruction.opname = instruction.opname.replace("FORWARD", "BACKWARD")
    elif "BACKWARD" in instruction.opname:
        instruction.opname = instruction.opname.replace("BACKWARD", "FORWARD")
    else:
        raise AttributeError("Instruction is not a forward or backward jump")
    instruction.opcode = dis.opmap[instruction.opname]
    if instruction.opcode not in _REL_JUMPS:
        raise AssertionError(
            f"expected {instruction.opname} to be a relative jump instruction"
        )


def _get_instruction_front(instructions: list[Instruction], idx: int) -> Instruction:
    """
    i.e. get the first EXTENDED_ARG instruction (if any) when targeting
    instructions[idx] with a jump.
    """
    target = instructions[idx]
    for offset in (1, 2, 3):
        if idx >= offset and instructions[idx - offset].opcode == dis.EXTENDED_ARG:
            target = instructions[idx - offset]
        else:
            break
    return target


def devirtualize_jumps(instructions: list[Instruction]) -> None:
    """Fill in args for virtualized jump target after instructions may have moved"""
    jumps = set(dis.hasjabs).union(set(dis.hasjrel))

    # check for negative jump args and fix them
    for inst in instructions:
        if inst.opcode in jumps:
            if inst.opcode not in dis.hasjabs:
                if inst.target is None:
                    raise AssertionError(
                        f"jump instruction {inst.opname} has no target"
                    )
                if inst.target.offset is None:
                    raise AssertionError(f"jump target for {inst.opname} has no offset")
                if inst.offset is None:
                    raise AssertionError(
                        f"jump instruction {inst.opname} has no offset"
                    )
                if inst.target.offset < inst.offset:
                    if sys.version_info < (3, 11):
                        raise RuntimeError("Got negative jump offset for Python < 3.11")
                    # forward jumps become backward
                    if "FORWARD" in inst.opname:
                        flip_jump_direction(inst)
                else:
                    # backward jumps become forward
                    if sys.version_info >= (3, 11) and "BACKWARD" in inst.opname:
                        flip_jump_direction(inst)

    # jump instruction size may have changed due to flips
    update_offsets(instructions)
    indexof = get_indexof(instructions)

    # compute jump instruction arg
    for inst in instructions:
        if inst.opcode in jumps:
            if inst.target is None:
                raise AssertionError(f"jump instruction {inst.opname} has no target")
            target = _get_instruction_front(instructions, indexof[inst.target])
            if inst.opcode in dis.hasjabs:
                if sys.version_info < (3, 11):
                    # `arg` is expected to be bytecode offset, whereas `offset` is byte offset.
                    # Divide since bytecode is 2 bytes large.
                    if target.offset is None:
                        raise AssertionError("absolute jump target has no offset")
                    inst.arg = int(target.offset / 2)
                else:
                    raise RuntimeError("Python 3.11+ should not have absolute jumps")
            else:  # relative jump
                # byte offset between target and next instruction
                if target.offset is None:
                    raise AssertionError("relative jump target has no offset")
                if inst.offset is None:
                    raise AssertionError(
                        f"relative jump instruction {inst.opname} has no offset"
                    )
                inst.arg = abs(
                    int(target.offset - inst.offset - instruction_size(inst))
                )
                # pyrefly: ignore [unsupported-operation]
                inst.arg //= 2
            inst.argval = target.offset
            inst.argrepr = f"to {target.offset}"


def virtualize_exception_table(
    exn_tab_bytes: bytes, instructions: list[Instruction]
) -> None:
    """Replace exception table entries with pointers to make editing easier"""
    exn_tab = parse_exception_table(exn_tab_bytes)
    offset_to_inst = {cast(int, inst.offset): inst for inst in instructions}
    offsets = sorted(offset_to_inst.keys())
    end_offset_idx = 0
    exn_tab_iter = iter(exn_tab)
    try:

        def step() -> tuple[ExceptionTableEntry, InstructionExnTabEntry]:
            nonlocal end_offset_idx
            entry = next(exn_tab_iter)
            # find rightmost offset <= entry.end, since entry.end may not be
            # an actual instruction, e.g. if the end instruction is LOAD_GLOBAL,
            # which takes more than 2 bytes, then entry.end points to the end
            # of the LOAD_GLOBAL instruction, not the beginning.
            while (
                end_offset_idx < len(offsets) and offsets[end_offset_idx] <= entry.end
            ):
                end_offset_idx += 1
            if end_offset_idx <= 0:
                raise AssertionError(
                    "end_offset_idx must be positive in exception table virtualization"
                )
            end_offset = offsets[end_offset_idx - 1]
            inst_entry = InstructionExnTabEntry(
