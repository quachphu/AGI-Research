from torch.utils.benchmark.op_fuzzers.spectral import SpectralOpFuzzer


def _dim_options(ndim):
    if ndim == 1:
        return [None]
    elif ndim == 2:
        return [0, 1, None]
    elif ndim == 3:
        return [0, 1, 2, (0, 1), (0, 2), None]
    raise ValueError(f"Expected ndim in range 1-3, got {ndim}")


def run_benchmark(name: str, function: object, dtype: torch.dtype, seed: int, device: str, samples: int,
                  probability_regular: float):
    cuda = device == 'cuda'
    spectral_fuzzer = SpectralOpFuzzer(seed=seed, dtype=dtype, cuda=cuda,
                                       probability_regular=probability_regular)
    results = []
    for tensors, tensor_params, params in spectral_fuzzer.take(samples):
        shape = [params['k0'], params['k1'], params['k2']][:params['ndim']]
        str_shape = ' x '.join([f"{s:<4}" for s in shape])
        sub_label = f"{str_shape} {'' if tensor_params['x']['is_contiguous'] else '(discontiguous)'}"
        for dim in _dim_options(params['ndim']):
            for nthreads in (1, 4, 16) if not cuda else (1,):
                measurement = benchmark.Timer(
                    stmt='func(x, dim=dim)',
                    globals={'func': function, 'x': tensors['x'], 'dim': dim},
                    label=f"{name}_{device}",
                    sub_label=sub_label,
                    description=f"dim={dim}",
                    num_threads=nthreads,
                ).blocked_autorange(min_run_time=1)
                measurement.metadata = {
                    'name': name,
                    'device': device,
                    'dim': dim,
                    'shape': shape,
                }
                measurement.metadata.update(tensor_params['x'])
                results.append(measurement)
    return results


Benchmark = namedtuple('Benchmark', ['name', 'function', 'dtype'])
BENCHMARKS = [
    Benchmark('fft_real', torch.fft.fftn, torch.float32),
    Benchmark('fft_complex', torch.fft.fftn, torch.complex64),
    Benchmark('ifft', torch.fft.ifftn, torch.complex64),
    Benchmark('rfft', torch.fft.rfftn, torch.float32),
    Benchmark('irfft', torch.fft.irfftn, torch.complex64),
]
BENCHMARK_MAP = {b.name: b for b in BENCHMARKS}
BENCHMARK_NAMES = [b.name for b in BENCHMARKS]
DEVICE_NAMES = ['cpu', 'cuda']

def _output_csv(file, results) -> None:
    file.write('benchmark,device,num_threads,numel,shape,contiguous,dim,mean (us),median (us),iqr (us)\n')
    for measurement in results:
        metadata = measurement.metadata
        device, dim, shape, name, numel, contiguous = (
            metadata['device'], metadata['dim'], metadata['shape'],
            metadata['name'], metadata['numel'], metadata['is_contiguous'])

        if isinstance(dim, Iterable):
            dim_str = '-'.join(str(d) for d in dim)
        else:
            dim_str = str(dim)
            shape_str = 'x'.join(str(s) for s in shape)

        print(name, device, measurement.task_spec.num_threads, numel, shape_str, contiguous, dim_str,  # type: ignore[possibly-undefined]
              measurement.mean * 1e6, measurement.median * 1e6, measurement.iqr * 1e6,
              sep=',', file=file)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=str, choices=DEVICE_NAMES, nargs='+', default=DEVICE_NAMES)
    parser.add_argument('--bench', type=str, choices=BENCHMARK_NAMES, nargs='+', default=BENCHMARK_NAMES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--probability-regular', '--probability_regular', type=float, default=1.0)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    num_benchmarks = len(args.device) * len(args.bench)
    i = 0
    results = []
    for device in args.device:
        for bench in (BENCHMARK_MAP[b] for b in args.bench):
            results += run_benchmark(
                name=bench.name, function=bench.function, dtype=bench.dtype,
                seed=args.seed, device=device, samples=args.samples,
                probability_regular=args.probability_regular)
            i += 1
            print(f'Completed {bench.name} benchmark on {device} ({i} of {num_benchmarks})')

    if args.output is not None:
        with open(args.output, 'w') as f:
            _output_csv(f, results)

    compare = benchmark.Compare(results)
    compare.trim_significant_figures()
    compare.colorize()
    compare.print()
<|endoftext|>"""Trivial use of Timer API:

$ python -m examples.simple_timeit
"""

import torch

import torch.utils.benchmark as benchmark_utils


def main() -> None:
    timer = benchmark_utils.Timer(
        stmt="x + y",
        globals={"x": torch.ones((4, 8)), "y": torch.ones((1, 8))},
        label="Broadcasting add (4x8)",
    )

    for i in range(3):
        print(f"Run: {i}\n{'-' * 40}")
        print(f"timeit:\n{timer.timeit(10000)}\n")
        print(f"autorange:\n{timer.blocked_autorange()}\n\n")


if __name__ == "__main__":
    main()
<|endoftext|># mypy: allow-untyped-defs
"""Example of Timer and Compare APIs:

$ python -m examples.compare
"""

import pickle
import sys
import time

import torch

import torch.utils.benchmark as benchmark_utils


class FauxTorch:
    """Emulate different versions of pytorch.

    In normal circumstances this would be done with multiple processes
    writing serialized measurements, but this simplifies that model to
    make the example clearer.
    """
    def __init__(self, real_torch, extra_ns_per_element) -> None:
        self._real_torch = real_torch
        self._extra_ns_per_element = extra_ns_per_element

    def extra_overhead(self, result):
        # time.sleep has a ~65 us overhead, so only fake a
        # per-element overhead if numel is large enough.
        numel = int(result.numel())
        if numel > 5000:
            time.sleep(numel * self._extra_ns_per_element * 1e-9)
        return result

    def add(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.add(*args, **kwargs))

    def mul(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.mul(*args, **kwargs))

    def cat(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.cat(*args, **kwargs))

    def matmul(self, *args, **kwargs):
        return self.extra_overhead(self._real_torch.matmul(*args, **kwargs))


def main() -> None:
    tasks = [
        ("add", "add", "torch.add(x, y)"),
        ("add", "add (extra +0)", "torch.add(x, y + zero)"),
    ]

    serialized_results = []
    repeats = 2
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,
            globals={
                "torch": torch if branch == "master" else FauxTorch(torch, overhead_ns),
                "x": torch.ones((size, 4)),
                "y": torch.ones((1, 4)),
                "zero": torch.zeros(()),
            },
            label=label,
            sub_label=sub_label,
            description=f"size: {size}",
            env=branch,
            num_threads=num_threads,
        )
        for branch, overhead_ns in [("master", None), ("my_branch", 1), ("severe_regression", 5)]
        for label, sub_label, stmt in tasks
        for size in [1, 10, 100, 1000, 10000, 50000]
        for num_threads in [1, 4]
    ]

    for i, timer in enumerate(timers * repeats):
        serialized_results.append(pickle.dumps(
            timer.blocked_autorange(min_run_time=0.05)
        ))
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
    print()

    comparison = benchmark_utils.Compare([
        pickle.loads(i) for i in serialized_results
    ])

    print("== Unformatted " + "=" * 80 + "\n" + "/" * 95 + "\n")
    comparison.print()

    print("== Formatted " + "=" * 80 + "\n" + "/" * 93 + "\n")
    comparison.trim_significant_figures()
    comparison.colorize()
    comparison.print()


if __name__ == "__main__":
    main()
<|endoftext|># mypy: allow-untyped-defs
"""Example use of Timer and op fuzzers to measure kernel performance.

$ python -m examples.op_benchmark
"""

import numpy as np
import torch

from torch.utils.benchmark import Timer
from torch.utils.benchmark.op_fuzzers.binary import BinaryOpFuzzer
from torch.utils.benchmark.op_fuzzers.unary import UnaryOpFuzzer
import operator


_MEASURE_TIME = 1.0


def assert_dicts_equal(dict_0, dict_1) -> None:
    """Builtin dict comparison will not compare numpy arrays.

    e.g.::

        x = {"a": np.ones((2, 1))}
        x == x  # Raises ValueError
    """
    if set(dict_0.keys()) != set(dict_0.keys()):
        raise AssertionError("dicts must have the same keys")
    if all(np.all(v != dict_1[k]) for k, v in dict_0.items() if k != "dtype"):
        raise AssertionError("dict values differ for keys other than 'dtype'")


def run(n, stmt, fuzzer_cls) -> None:
    float_iter = fuzzer_cls(seed=0, dtype=torch.float32).take(n)
    int_iter = fuzzer_cls(seed=0, dtype=torch.int32).take(n)
    raw_results = []
    for i, (float_values, int_values) in enumerate(zip(float_iter, int_iter, strict=True)):
        float_tensors, float_tensor_params, float_params = float_values
        int_tensors, int_tensor_params, int_params = int_values

        # This benchmark assumes that the two fuzzers generate identically
        # sized and strided Tensors, since the same seed is used.
        assert_dicts_equal(float_params, int_params)
        assert_dicts_equal(float_tensor_params["x"], int_tensor_params["x"])

        float_measurement, int_measurement = (
            Timer(
                stmt,
                globals=tensors,
            ).blocked_autorange(min_run_time=_MEASURE_TIME)
            for tensors in (float_tensors, int_tensors)
        )

        descriptions = []
        for name in float_tensors:
            shape_str = "(" + ", ".join([
                f"2 ** {int(np.log2(i))}"
                if 2 ** int(np.log2(i)) == i and i > 1
                else str(i)
                for i in float_tensors[name].shape
            ]) + ")"
            order = float_tensor_params[name]["order"]
            order_str = ("" if all(order == np.arange(len(order))) else str(tuple(order)))
            steps = float_tensor_params[name]["steps"]
            steps_str = str(steps) if sum(steps) > len(steps) else ""
            descriptions.append((name, shape_str, order_str, steps_str))
        raw_results.append((float_measurement, int_measurement, descriptions))

        print(f"\r{i + 1} / {n}", end="")
    print()

    parsed_results, name_len, shape_len, order_len, steps_len = [], 0, 0, 0, 0
    for float_measurement, int_measurement, descriptions in raw_results:
        t_float = float_measurement.median * 1e6
        t_int = int_measurement.median * 1e6
        rel_diff = abs(t_float - t_int) / (t_float + t_int) * 2
        parsed_results.append((t_float, t_int, rel_diff, descriptions))
        for name, shape, order, steps in descriptions:
            name_len = max(name_len, len(name))
            shape_len = max(shape_len, len(shape))
            order_len = max(order_len, len(order))
            steps_len = max(steps_len, len(steps))

    parsed_results.sort(key=operator.itemgetter(2))

    print(f"stmt: {stmt}")
    print(f" diff    faster{'':>17}{' ' * name_len} ", end="")
    print(f"{'shape'.ljust(shape_len)}{'':>16}{'order'.ljust(order_len)}", end="")
    print(f"          steps\n{'-' * 100}")
    for results, spacer in [(parsed_results[:10], "..."), (parsed_results[-10:], "")]:
        for t_float, t_int, rel_diff, descriptions in results:
            time_str = [f"{rel_diff * 100:>4.1f}%    {'int' if t_int < t_float else 'float':<20}"]
            time_str.extend(["".ljust(len(time_str[0])) for _ in descriptions[:-1]])
            for t_str, (name, shape, order, steps) in zip(time_str, descriptions, strict=True):
                name = f"{name}:".ljust(name_len + 1)
                shape = shape.ljust(shape_len + 10)
                order = order.ljust(order_len)
                print(f"{t_str} {name}  {shape}|     {order}      |   {steps}")
        print(spacer)


def main() -> None:
    run(n=100, stmt="torch.median(x, dim=0)", fuzzer_cls=UnaryOpFuzzer)
    run(n=100, stmt="torch.square(x)", fuzzer_cls=UnaryOpFuzzer)
    run(n=100, stmt="x + y", fuzzer_cls=BinaryOpFuzzer)


if __name__ == "__main__":
    main()
<|endoftext|># mypy: allow-untyped-defs
"""Example of the Timer and Fuzzer APIs:

$ python -m examples.fuzzer
"""

import sys

import torch.utils.benchmark as benchmark_utils


def main() -> None:
    add_fuzzer = benchmark_utils.Fuzzer(
        parameters=[
            [
                benchmark_utils.FuzzedParameter(
                    name=f"k{i}",
                    minval=16,
                    maxval=16 * 1024,
                    distribution="loguniform",
                ) for i in range(3)
            ],
            benchmark_utils.FuzzedParameter(
                name="d",
                distribution={2: 0.6, 3: 0.4},
            ),
        ],
        tensors=[
            [
                benchmark_utils.FuzzedTensor(
                    name=name,
                    size=("k0", "k1", "k2"),
                    dim_parameter="d",
                    probability_contiguous=0.75,
                    min_elements=64 * 1024,
                    max_elements=128 * 1024,
                ) for name in ("x", "y")
            ],
        ],
        seed=0,
    )

    n = 250
    measurements = []
    for i, (tensors, tensor_properties, _) in enumerate(add_fuzzer.take(n=n)):
        x, x_order = tensors["x"], str(tensor_properties["x"]["order"])
        y, y_order = tensors["y"], str(tensor_properties["y"]["order"])
        shape = ", ".join(tuple(f'{i:>4}' for i in x.shape))

        description = "".join([
            f"{x.numel():>7} | {shape:<16} | ",
            f"{'contiguous' if x.is_contiguous() else x_order:<12} | ",
            f"{'contiguous' if y.is_contiguous() else y_order:<12} | ",
        ])

        timer = benchmark_utils.Timer(
            stmt="x + y",
            globals=tensors,
            description=description,
        )

        measurements.append(timer.blocked_autorange(min_run_time=0.1))
        measurements[-1].metadata = {"numel": x.numel()}
        print(f"\r{i + 1} / {n}", end="")
        sys.stdout.flush()
    print()

    # More string munging to make pretty output.
    print(f"Average attempts per valid config: {1. / (1. - add_fuzzer.rejection_rate):.1f}")

    def time_fn(m):
        return m.median / m.metadata["numel"]
    measurements.sort(key=time_fn)

    template = f"{{:>6}}{' ' * 19}Size    Shape{' ' * 13}X order        Y order\n{'-' * 80}"
    print(template.format("Best:"))
    for m in measurements[:15]:
        print(f"{time_fn(m) * 1e9:>4.1f} ns / element     {m.description}")

    print("\n" + template.format("Worst:"))
    for m in measurements[-15:]:
        print(f"{time_fn(m) * 1e9:>4.1f} ns / element     {m.description}")


if __name__ == "__main__":
    main()
<|endoftext|>#!/usr/bin/env python3
# mypy: allow-untyped-defs
""" The Python Hipify script.
##
# Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
#               2017-2018 Advanced Micro Devices, Inc. and
#                         Facebook Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
import argparse
import fnmatch
import re
import shutil
import sys
import os
import warnings

from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from .cuda_to_hip_mappings import CAFFE2_PATH_MAPPINGS

from collections.abc import Iterator
from collections.abc import Mapping, Iterable
from enum import Enum
import functools
import hashlib

def _deprecated(name):
    warnings.warn(f"hipify version 2.0.0 no longer uses function {name}", FutureWarning, stacklevel=2)

class CurrentState(Enum):
    INITIALIZED = 1
    DONE = 2

class HipifyResult:
    def __init__(self, current_state, hipified_path) -> None:
        self.current_state = current_state
        self.hipified_path = hipified_path
        self.status = ""

    def __str__(self) -> str:
        return (f"HipifyResult:: current_state: {self.current_state}, hipified_path : {self.hipified_path}, status: {self.status}")

HipifyFinalResult = dict[str, HipifyResult]
HIPIFY_C_BREADCRUMB = "// !!! This is a file automatically generated by hipify!!!\n"
HIPIFY_FINAL_RESULT: HipifyFinalResult = {}

# Hardcode the PyTorch template map
"""This dictionary provides the mapping from PyTorch kernel template types
to their actual types."""
PYTORCH_TEMPLATE_MAP = {"Dtype": "scalar_t", "T": "scalar_t"}

__all__ = ['InputError', 'openf', 'bcolors', 'GeneratedFileCleaner', 'match_extensions', 'matched_files_iter',
           'preprocess_file_and_save_result', 'compute_stats', 'add_dim3', 'processKernelLaunches', 'find_closure_group',
           'find_bracket_group', 'find_parentheses_group', 'replace_math_functions', 'hip_header_magic', 'replace_extern_shared',
           'get_hip_file_path', 'is_out_of_place', 'is_pytorch_file', 'is_cusparse_file', 'is_special_file', 'is_caffe2_gpu_file',
           'Trie', 'preprocessor', 'file_specific_replacement', 'file_add_header',
           'fix_static_global_kernels', 'extract_arguments', 'str2bool', 'CurrentState', 'HipifyResult', 'hipify']


class InputError(Exception):
    # Exception raised for errors in the input.

    def __init__(self, message) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"Input error: {self.message}"


def openf(filename, mode):
    return open(filename, mode, errors='ignore')


# Color coding for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# To the programmer, the output of hipify most likely are intermediates.
# This class allows users of hipify to ask for a cleanup by running the
# hipify and compilation in a with instantiating this context manager class
# with keep_intermediates=False.
# The main usecase is the cpp_extensions, specifically the load method.
# It is a good idea to keep intermediates (in case of errors or to
# not recompile unchanged files), but in cases where you don't want to
# keep them (e.g. in the CI), this can be used to remove files.
class GeneratedFileCleaner:
    """Context Manager to clean up generated files"""
    def __init__(self, keep_intermediates=False) -> None:
        self.keep_intermediates = keep_intermediates
        self.files_to_clean = set()
        self.dirs_to_clean = []

    def __enter__(self):
        return self

    def open(self, fn, *args, **kwargs):
        if not os.path.exists(fn):
            self.files_to_clean.add(os.path.abspath(fn))

        return open(fn, *args, **kwargs)

    def makedirs(self, dn, exist_ok=False) -> None:
        parent, n = os.path.split(dn)
        if not n:
            parent, n = os.path.split(parent)
        if parent and n and not os.path.exists(parent):
            self.makedirs(parent, exist_ok=True)
        if not os.path.isdir(dn) or not exist_ok:
            os.mkdir(dn)
            self.dirs_to_clean.append(os.path.abspath(dn))

    def __exit__(self, type, value, traceback):
        if not self.keep_intermediates:
            for f in self.files_to_clean:
                os.unlink(f)
            for d in self.dirs_to_clean[::-1]:
                os.rmdir(d)

# Follow UNIX convention for paths to use '/' instead of '\\' on Windows
def _to_unix_path(path: str) -> str:
    return path.replace(os.sep, '/')

def match_extensions(filename: str, extensions: Iterable) -> bool:
    """Helper method to see if filename ends with certain extension"""
    return any(filename.endswith(e) for e in extensions)


def _fnmatch(filepath, patterns):
    return any(fnmatch.fnmatch(filepath, pattern) for pattern in patterns)


def matched_files_iter(
        root_path: str,
        includes: Iterable = (),
        ignores: Iterable = (),
        extensions: Iterable = (),
        out_of_place_only: bool = False,
        is_pytorch_extension: bool = False) -> Iterator[str]:

    exact_matches = set(includes)

    # This is a very rough heuristic; really, we want to avoid scanning
    # any file which is not checked into source control, but this script
    # needs to work even if you're in a Git or Hg checkout, so easier to
    # just block the biggest time sinks that won't matter in the
    # end.
    for (abs_dirpath, dirs, filenames) in os.walk(root_path, topdown=True):
        rel_dirpath = os.path.relpath(abs_dirpath, root_path)
        if rel_dirpath == '.':
            # Blah blah blah O(n) blah blah
            if ".git" in dirs:
                dirs.remove(".git")
            if "build" in dirs:
                dirs.remove("build")
            if "third_party" in dirs:
                dirs.remove("third_party")
                dirs.append("third_party/nvfuser")
        for filename in filenames:
            filepath = _to_unix_path(os.path.join(abs_dirpath, filename))
            # We respect extensions, UNLESS you wrote the entire
            # filename verbatim, in which case we always accept it
            if (
                _fnmatch(filepath, includes)
                and (not _fnmatch(filepath, ignores))
                and (match_extensions(filepath, extensions) or filepath in exact_matches)
            ):
                yield filepath


def preprocess_file_and_save_result(
        output_directory: str,
        filepath: str,
        all_files: Iterable,
        header_include_dirs: Iterable,
        stats: dict[str, list],
        hip_clang_launch: bool,
        is_pytorch_extension: bool,
        clean_ctx: GeneratedFileCleaner,
        show_progress: bool) -> None:
    fin_path = os.path.abspath(os.path.join(output_directory, filepath))
    hipify_result = HipifyResult(current_state=CurrentState.INITIALIZED, hipified_path=fin_path)
    HIPIFY_FINAL_RESULT[fin_path] = hipify_result
    result = preprocessor(output_directory, filepath, all_files, header_include_dirs, stats,
                          hip_clang_launch, is_pytorch_extension, clean_ctx, show_progress)

    # Show what happened
    if show_progress and "ignored" not in result.status:
        print(
            fin_path, "->",
            result.hipified_path, result.status, flush=True)

    HIPIFY_FINAL_RESULT[fin_path] = result


def compute_stats(stats) -> None:
    unsupported_calls = {cuda_call for (cuda_call, _filepath) in stats["unsupported_calls"]}

    # Print the number of unsupported calls
    print(f"Total number of unsupported CUDA function calls: {len(unsupported_calls):d}")

    # Print the list of unsupported calls
    print(", ".join(unsupported_calls))

    # Print the number of kernel launches
    print(f"\nTotal number of replaced kernel launches: {len(stats['kernel_launches']):d}")


def add_dim3(kernel_string, cuda_kernel):
    '''adds dim3() to the second and third arguments in the kernel launch'''
    count = 0
    closure = 0
    kernel_string = kernel_string.replace("<<<", "").replace(">>>", "")
    arg_locs: list[dict[str, int]] = [{} for _ in range(2)]
    arg_locs[count]['start'] = 0
    for ind, c in enumerate(kernel_string):
        if count > 1:
            break
        if c == "(":
            closure += 1
        elif c == ")":
            closure -= 1
        if (c == "," or ind == len(kernel_string) - 1) and closure == 0:
            arg_locs[count]['end'] = ind + (c != ",")
            count += 1
            if count < 2:
                arg_locs[count]['start'] = ind + 1

    first_arg_raw = kernel_string[arg_locs[0]['start']:arg_locs[0]['end'] + 1]
    second_arg_raw = kernel_string[arg_locs[1]['start']:arg_locs[1]['end']]

    first_arg_clean = kernel_string[arg_locs[0]['start']:arg_locs[0]['end']].replace("\n", "").strip(" ")
    second_arg_clean = kernel_string[arg_locs[1]['start']:arg_locs[1]['end']].replace("\n", "").strip(" ")

    first_arg_dim3 = f"dim3({first_arg_clean})"
    second_arg_dim3 = f"dim3({second_arg_clean})"

    first_arg_raw_dim3 = first_arg_raw.replace(first_arg_clean, first_arg_dim3)
    second_arg_raw_dim3 = second_arg_raw.replace(second_arg_clean, second_arg_dim3)
    cuda_kernel = cuda_kernel.replace(first_arg_raw + second_arg_raw, first_arg_raw_dim3 + second_arg_raw_dim3)
    return cuda_kernel


RE_KERNEL_LAUNCH = re.compile(r'([ ]+)(detail?)::[ ]+\\\n[ ]+')


def processKernelLaunches(string, stats):
    """ Replace the CUDA style Kernel launches with the HIP style kernel launches."""
    # Concat the namespace with the kernel names. (Find cleaner way of doing this later).
    string = RE_KERNEL_LAUNCH.sub(lambda inp: f"{inp.group(1)}{inp.group(2)}::", string)

    def grab_method_and_template(in_kernel):
        # The positions for relevant kernel components.
        pos = {
            "kernel_launch": {"start": in_kernel["start"], "end": in_kernel["end"]},
            "kernel_name": {"start": -1, "end": -1},
            "template": {"start": -1, "end": -1}
        }

        # Count for balancing template
        count = {"<>": 0}

        # Status for whether we are parsing a certain item.
        START = 0
        AT_TEMPLATE = 1
        AFTER_TEMPLATE = 2
        AT_KERNEL_NAME = 3

        status = START

        # Parse the string character by character
        for i in range(pos["kernel_launch"]["start"] - 1, -1, -1):
            char = string[i]

            # Handle Templating Arguments
            if status in (START, AT_TEMPLATE):
                if char == ">":
                    if status == START:
                        status = AT_TEMPLATE
                        pos["template"]["end"] = i
                    count["<>"] += 1

                if char == "<":
                    count["<>"] -= 1
                    if count["<>"] == 0 and (status == AT_TEMPLATE):
                        pos["template"]["start"] = i
                        status = AFTER_TEMPLATE

            # Handle Kernel Name
            if status != AT_TEMPLATE:
                if string[i].isalnum() or string[i] in {'(', ')', '_', ':', '#'}:
                    if status != AT_KERNEL_NAME:
                        status = AT_KERNEL_NAME
                        pos["kernel_name"]["end"] = i

                    # Case: Kernel name starts the string.
                    if i == 0:
                        pos["kernel_name"]["start"] = 0

                        # Finished
                        return [(pos["kernel_name"]), (pos["template"]), (pos["kernel_launch"])]

                else:
                    # Potential ending point if we're already traversing a kernel's name.
                    if status == AT_KERNEL_NAME:
                        pos["kernel_name"]["start"] = i

                        # Finished
                        return [(pos["kernel_name"]), (pos["template"]), (pos["kernel_launch"])]

    def find_kernel_bounds(string):
        """Finds the starting and ending points for all kernel launches in the string."""
        kernel_end = 0
        kernel_positions = []

        # Continue until we cannot find any more kernels anymore.
        while string.find("<<<", kernel_end) != -1:
            # Get kernel starting position (starting from the previous ending point)
            kernel_start = string.find("<<<", kernel_end)

            # Get kernel ending position (adjust end point past the >>>)
            kernel_end = string.find(">>>", kernel_start) + 3
            if kernel_end <= 0:
                raise InputError("no kernel end found")

            # Add to list of traversed kernels
            kernel_positions.append({"start": kernel_start, "end": kernel_end,
                                     "group": string[kernel_start: kernel_end]})

        return kernel_positions

    # Replace comments and string literals from the code so that find_kernel_bounds does not
    # wrongly capture kernels in comments and string literals.
    # This function replaces them with "x" to keep positions.
    def mask_comments(string):
        in_comment = ''
        prev_c = ''
        new_string = ''
        for c in string:
            if in_comment == '':
                # Outside comments
                if c == '/' and prev_c == '/':
                    in_comment = '//'
                elif c == '*' and prev_c == '/':
                    in_comment = '/*'
                elif c == '"' and prev_c != '\\' and prev_c != "'":
                    in_comment = '"'
            elif in_comment == '//':
                # In // xxx
                if c == '\r' or c == '\n':
                    in_comment = ''
            elif in_comment == '/*':
                # In /* xxx */
                if c == '/' and prev_c == '*':
                    in_comment = ''
            elif in_comment == '"':
                # In ""
                if c == '"' and prev_c != '\\':
                    in_comment = ''
            prev_c = c
            if in_comment == '':
                new_string += c
            else:
                new_string += 'x'
        return new_string

    # Grab positional ranges of all kernel launches
    get_kernel_positions = list(find_kernel_bounds(mask_comments(string)))
    output_string = string

    # Replace each CUDA kernel with a HIP kernel.
    for kernel in get_kernel_positions:
        # Get kernel components
        params = grab_method_and_template(kernel)

        # Find parenthesis after kernel launch
        parenthesis = string.find("(", kernel["end"])

        # Extract cuda kernel
        cuda_kernel = string[params[0]["start"]:parenthesis + 1]
        kernel_string = string[kernel['start']:kernel['end']]
        end_param_index = 0 if params[1]['end'] == -1 else 1
        kernel_name_with_template = string[params[0]['start']:params[end_param_index]['end'] + 1]
        cuda_kernel_dim3 = add_dim3(kernel_string, cuda_kernel)
        # Keep number of kernel launch params consistent (grid dims, group dims, stream, dynamic shared size)
        num_klp = len(extract_arguments(0, kernel["group"].replace("<<<", "(").replace(">>>", ")")))

        hip_kernel = "hipLaunchKernelGGL(" + cuda_kernel_dim3[0:-1].replace(
            ">>>", ", 0" * (4 - num_klp) + ">>>").replace("<<<", ", ").replace(
            ">>>", ", ").replace(kernel_name_with_template, "(" + kernel_name_with_template + ")")

        # Replace cuda kernel with hip kernel
        output_string = output_string.replace(cuda_kernel, hip_kernel)

        # Update the statistics
        stats["kernel_launches"].append(hip_kernel)

    return output_string


def find_closure_group(input_string, start, group):
    """Generalization for finding a balancing closure group

         if group = ["(", ")"], then finds the first balanced parentheses.
         if group = ["{", "}"], then finds the first balanced bracket.

    Given an input string, a starting position in the input string, and the group type,
    find_closure_group returns the positions of group[0] and group[1] as a tuple.

    Example:
        >>> find_closure_group("(hi)", 0, ["(", ")"])
        (0, 3)
    """

    inside_parenthesis = False
    parens = 0
    pos = start
    p_start, p_end = -1, -1

    while pos < len(input_string):
        if input_string[pos] == group[0]:
            if inside_parenthesis is False:
                inside_parenthesis = True
                parens = 1
                p_start = pos
            else:
                parens += 1
        elif input_string[pos] == group[1] and inside_parenthesis:
            parens -= 1

            if parens == 0:
                p_end = pos
                return p_start, p_end

        pos += 1
    return None, None


def find_bracket_group(input_string, start):
    """Finds the first balanced parentheses."""
    return find_closure_group(input_string, start, group=["{", "}"])


def find_parentheses_group(input_string, start):
    """Finds the first balanced bracket."""
    return find_closure_group(input_string, start, group=["(", ")"])


RE_ASSERT = re.compile(r"\bassert[ ]*\(")


def replace_math_functions(input_string):
    """FIXME: Temporarily replace std:: invocations of math functions
        with non-std:: versions to prevent linker errors NOTE: This
        can lead to correctness issues when running tests, since the
        correct version of the math function (exp/expf) might not get
        called.  Plan is to remove this function once HIP supports
        std:: math function calls inside device code

    """
    output_string = input_string
    for func in MATH_TRANSPILATIONS:
        output_string = output_string.replace(fr'{func}(', f'{MATH_TRANSPILATIONS[func]}(')

    return output_string


RE_SYNCTHREADS = re.compile(r":?:?\b(__syncthreads)\b(\w*\()")


def hip_header_magic(input_string):
    """If the file makes kernel builtin calls and does not include the cuda_runtime.h header,
    then automatically add an #include to match the "magic" includes provided by NVCC.
    TODO:
        Update logic to ignore cases where the cuda_runtime.h is included by another file.
    """

    # Copy the input.
    output_string = input_string

    # Check if one of the following headers is already included.
    headers = ["hip/hip_runtime.h", "hip/hip_runtime_api.h"]
    if any(re.search(fr'#include ("{ext}"|<{ext}>)', output_string) for ext in headers):
        return output_string

    # Rough logic to detect if we're inside device code
    hasDeviceLogic: int
    hasDeviceLogic = "hipLaunchKernelGGL" in output_string
    hasDeviceLogic += "__global__" in output_string
    hasDeviceLogic += "__shared__" in output_string
    hasDeviceLogic += RE_SYNCTHREADS.search(output_string) is not None

    # If device logic found, provide the necessary header.
    if hasDeviceLogic:
        output_string = '#include "hip/hip_runtime.h"\n' + input_string

    return output_string


RE_EXTERN_SHARED = re.compile(r"extern\s+([\w\(\)]+)?\s*__shared__\s+([\w:<>\s]+)\s+(\w+)\s*\[\s*\]\s*;")


def replace_extern_shared(input_string):
    """
    Match 'extern __shared__ type foo[];' syntax and use HIP_DYNAMIC_SHARED() MACRO instead.
    See: https://github.com/ROCm/hip/blob/master/docs/markdown/hip_kernel_language.md#__shared__
    Examples:
        "extern __shared__ char smemChar[];"
            => "HIP_DYNAMIC_SHARED( char, smemChar)"
        "extern __shared__ unsigned char smem[];"
            => "HIP_DYNAMIC_SHARED( unsigned char, my_smem)"
    """
    output_string = input_string
    output_string = RE_EXTERN_SHARED.sub(
        lambda inp: f"HIP_DYNAMIC_SHARED({inp.group(1) or ''} {inp.group(2)}, {inp.group(3)})", output_string)

    return output_string


def get_hip_file_path(rel_filepath, is_pytorch_extension=False):
    """
    Returns the new name of the hipified file
    """
    # At the moment, some PyTorch source files are HIPified in place.  The predicate
    # is_out_of_place tells us if this is the case or not.
    if os.path.isabs(rel_filepath):
        raise AssertionError("rel_filepath must be a relative path")
    if not is_pytorch_extension and not is_out_of_place(rel_filepath):
        return rel_filepath

    dirpath, filename = os.path.split(rel_filepath)
    root, ext = os.path.splitext(filename)

    # Here's the plan:
    #
    # In general, we need to disambiguate the HIPified filename so that
    # it gets a different name from the original filename, so
    # that we don't overwrite the original file
    #
    # There's a lot of different naming conventions across PyTorch,
    # but the general recipe is to convert occurrences
    # of cuda/gpu to hip, and add hip if there are no occurrences
    # of cuda/gpu anywhere.
    #
    # Concretely, we do the following:
    #
    #   - If there is a directory component named "cuda", replace
    #     it with "hip", AND
    #
    #   - If the file name contains "CUDA", replace it with "HIP", AND
    #
    #   - ALWAYS replace '.cu' with '.hip', because those files
    #     contain CUDA kernels that needs to be hipified and processed with
    #     hip compiler
    #
    #   - If we are not hipifying a PyTorch extension, and the parent
    #     directory name did not change as a result of the above
    #     transformations, insert "hip" in the file path
    #     as the direct parent folder of the file
    #
    #   - If we are hipifying a PyTorch extension, and the parent directory
    #     name as well as the filename (incl. extension) did not change as
    #     a result of the above transformations, insert "_hip" in the filename
    #
    # This isn't set in stone; we might adjust this to support other
    # naming conventions.

    if ext == '.cu':
        ext = '.hip'

    orig_filename = filename
    orig_dirpath = dirpath

    dirpath = dirpath.replace('cuda', 'hip')
    dirpath = dirpath.replace('CUDA', 'HIP')
    dirpath = dirpath.replace('THC', 'THH')

    root = root.replace('cuda', 'hip')
    root = root.replace('CUDA', 'HIP')
    # Special case to handle caffe2/core/THCCachingAllocator
    if dirpath != "caffe2/core":
        root = root.replace('THC', 'THH')

    if not is_pytorch_extension and dirpath == orig_dirpath:
        dirpath = os.path.join(dirpath, 'hip')

    if is_pytorch_extension and dirpath == orig_dirpath and (root + ext) == orig_filename:
        root = root + "_hip"

    return os.path.join(dirpath, root + ext)


def is_out_of_place(rel_filepath) -> bool:
    if os.path.isabs(rel_filepath):
        raise AssertionError("rel_filepath must be a relative path")
    if rel_filepath.startswith("torch/"):
        return False
    if rel_filepath.startswith("third_party/nvfuser/"):
        return False
    if rel_filepath.startswith("tools/autograd/templates/"):
        return False
    return True


# Keep this synchronized with includes/ignores in build_amd.py
def is_pytorch_file(rel_filepath) -> bool:
    _deprecated("is_pytorch_file")
    if os.path.isabs(rel_filepath):
        raise AssertionError("rel_filepath must be a relative path")
    if rel_filepath.startswith("aten/"):
        if rel_filepath.startswith("aten/src/ATen/core/"):
            return False
        return True
    if rel_filepath.startswith("torch/"):
        return True
    if rel_filepath.startswith("third_party/nvfuser/"):
        return True
    if rel_filepath.startswith("third_party/fbgemm/"):
        return True
    if rel_filepath.startswith("third_party/mslk/"):
        return True
    if rel_filepath.startswith("tools/autograd/templates/"):
        return True
    if rel_filepath.startswith("test/cpp/c10d/"):
        return True
    return False


def is_cusparse_file(rel_filepath):
    _deprecated("is_cusparse_file")
    if is_pytorch_file(rel_filepath):
        return "sparse" in rel_filepath.lower()
    return False


def is_special_file(rel_filepath) -> bool:
    _deprecated("is_special_file")
    if is_pytorch_file(rel_filepath):
        if "sparse" in rel_filepath.lower():
            return True
        elif "linalg" in rel_filepath.lower():
            if "batchlinearalgebralibblas" in rel_filepath.lower():
                return False  # don't use "special" mappings for this specific linalg cublas file
            return True
    return False


def is_caffe2_gpu_file(rel_filepath):
    _deprecated("is_caffe2_gpu_file")
    if os.path.isabs(rel_filepath):
        raise AssertionError("rel_filepath must be a relative path")
    if rel_filepath.startswith("c10/cuda"):
        return True
    filename = os.path.basename(rel_filepath)
    _, ext = os.path.splitext(filename)

    return ('gpu' in filename or ext in ['.cu', '.cuh']) and ('cudnn' not in filename)


class TrieNode:
    """A Trie node whose children are represented as a dictionary of char: TrieNode.
       A special char '' represents end of word
    """

    def __init__(self) -> None:
        self.children = {}


class Trie:
    """Creates a Trie out of a list of words. The trie can be exported to a Regex pattern.
    The corresponding Regex should match much faster than a simple Regex union."""

    def __init__(self) -> None:
        """Initialize the trie with an empty root node."""
        self.root = TrieNode()
        self._hash = hashlib.md5(usedforsecurity=False)
        self._digest = self._hash.digest()

    def add(self, word) -> None:
        """Add a word to the Trie. """
        self._hash.update(word.encode())
        self._digest = self._hash.digest()
        node = self.root

        for char in word:
            node.children.setdefault(char, TrieNode())
            node = node.children[char]
        node.children[''] = True    # Mark the end of the word

    def dump(self):
        """Return the root node of Trie. """
        return self.root

    def quote(self, char):
        """ Escape a char for regex. """
        return re.escape(char)

    def search(self, word):
        """Search whether word is present in the Trie.
        Returns True if yes, else return False"""
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                return False

        # make sure to check the end-of-word marker present
        return '' in node.children

    @functools.lru_cache  # noqa: B019
    def _pattern(self, root, digest):
        """Convert a Trie into a regular expression pattern

        Memoized on the hash digest of the trie, which is built incrementally
        during add().
        """
        node = root

        if "" in node.children and len(node.children.keys()) == 1:
            return None

        alt = []    # store alternative patterns
        cc = []     # store char to char classes
        q = 0       # for node representing the end of word
        for char in sorted(node.children.keys()):
            if isinstance(node.children[char], TrieNode):
                try:
                    recurse = self._pattern(node.children[char], self._digest)
                    alt.append(self.quote(char) + recurse)
                except Exception:
                    cc.append(self.quote(char))
            else:
                q = 1
        cconly = not len(alt) > 0

        if len(cc) > 0:
            if len(cc) == 1:
                alt.append(cc[0])
            else:
                alt.append('[' + ''.join(cc) + ']')

        if len(alt) == 1:
            result = alt[0]
        else:
            result = "(?:" + "|".join(alt) + ")"

        if q:
            if cconly:
                result += "?"
            else:
                result = f"(?:{result})?"
        return result

    def pattern(self):
        """Export the Trie to a regex pattern."""
        return self._pattern(self.root, self._digest)

    def export_to_regex(self):
        """Export the Trie to a regex pattern."""
        return self._pattern(self.root, self._digest)

PYTORCH_TRIE = Trie()
PYTORCH_MAP: dict[str, object] = {}

for mapping in CUDA_TO_HIP_MAPPINGS:
    if not isinstance(mapping, Mapping):
        raise TypeError("Expected each mapping in CUDA_TO_HIP_MAPPINGS to be a Mapping")
    for src, dst in mapping.items():
        PYTORCH_TRIE.add(src)
        PYTORCH_MAP[src] = dst

RE_PYTORCH_PREPROCESSOR = re.compile(fr'(?<=\W)({PYTORCH_TRIE.export_to_regex()})(?=\W)')

RE_QUOTE_HEADER = re.compile(r'#include "([^"]+)"')
RE_ANGLE_HEADER = re.compile(r'#include <([^>]+)>')
RE_THC_GENERIC_FILE = re.compile(r'#define THC_GENERIC_FILE "([^"]+)"')
RE_CU_SUFFIX = re.compile(r'\.cu\b')  # be careful not to pick up .cuh

"""
Returns a HipifyResult object with the following details:
    "hipified_path" : absolute path of hipified source file
    "status"        : "ok"      if hipified file was written out
                      "skipped" if an identical hipified file already existed or hipified file couldn't be written out
                      "ignored" if the source file was a hipified file itself or not meant to be hipified
    "current_state" : CurrentState.INITIALIZED if source file is first ready to be hipified
                      CurrentState.DONE if source file is done with hipification process
"""


def preprocessor(
        output_directory: str,
        filepath: str,
        all_files: Iterable,
        header_include_dirs: Iterable,
        stats: dict[str, list],
        hip_clang_launch: bool,
        is_pytorch_extension: bool,
        clean_ctx: GeneratedFileCleaner,
        show_progress: bool) -> HipifyResult:
    """ Executes the CUDA -> HIP conversion on the specified file. """
    fin_path = os.path.abspath(os.path.join(output_directory, filepath))
    filepath = _to_unix_path(filepath)
    hipify_result = HIPIFY_FINAL_RESULT[fin_path]
    if filepath not in all_files:
        hipify_result.hipified_path = None
        hipify_result.status = "[ignored, not to be hipified]"
        hipify_result.current_state = CurrentState.DONE
        return hipify_result

    rel_filepath = _to_unix_path(os.path.relpath(filepath, output_directory))

    with open(fin_path, encoding='utf-8') as fin:
        if fin.readline() == HIPIFY_C_BREADCRUMB:
            hipify_result.hipified_path = None
            hipify_result.status = "[ignored, input is hipified output]"
            hipify_result.current_state = CurrentState.DONE
            return hipify_result
        fin.seek(0)
        output_source = fin.read()

    orig_output_source = output_source

    # get_hip_file_path needs a relative path to work correctly
    fout_path = os.path.abspath(os.path.join(output_directory, get_hip_file_path(rel_filepath, is_pytorch_extension)))
    if not os.path.exists(os.path.dirname(fout_path)):
        clean_ctx.makedirs(os.path.dirname(fout_path))

    # unsupported_calls statistics reporting is broken atm
    def pt_repl(m):
        return PYTORCH_MAP[m.group(0)]

    output_source = RE_PYTORCH_PREPROCESSOR.sub(pt_repl, output_source)

    # TODO: Remove CAFFE2_PATH_MAPPINGS. They were necessary for Meta-internal builds.
    # Apply CAFFE2 path mappings (simple string replacement for paths containing slashes)
    # Need to be careful to avoid double-transformations when source file has #ifdef blocks
    # with HIP-specific paths already in them (e.g., caffe2/core/hip/context_gpu.h)
    for cuda_path, hip_path in CAFFE2_PATH_MAPPINGS.items():
        # Use regex to ensure we don't match paths that already have been hipified
        # We need to avoid transforming "caffe2/core/hip/context_gpu.h" when looking for "caffe2/core/context_gpu.h"
        # The key insight: if hip_path contains /hip/ and cuda_path doesn't, we need to be careful
        if "/hip/" in hip_path and "/hip/" not in cuda_path:
            # Only replace cuda_path if it's not preceded by "/hip/"
            # Use negative lookbehind to prevent matching already-hipified paths
            # The pattern checks that the cuda_path is not immediately preceded by "/hip/"
            pattern = r'(?<!/hip/)' + re.escape(cuda_path)
            output_source = re.sub(pattern, hip_path, output_source)
        else:
            # Simple replacement when no /hip/ involved or both have it
            output_source = output_source.replace(cuda_path, hip_path)

    # Header rewrites
    def mk_repl(templ, include_current_dir=True):
        def repl(m):
            f = m.group(1)
            filename = os.path.basename(f)
            if (
                f.startswith(("ATen/cuda",
                              "ATen/native/cuda",
                              "ATen/native/nested/cuda",
                              "ATen/native/quantized/cuda",
                              "ATen/native/sparse/cuda",
                              "ATen/native/transformers/cuda",
                              "THC/")) or
                (f.startswith("THC") and not f.startswith("THCP"))
            ):
                return templ.format(get_hip_file_path(m.group(1), is_pytorch_extension))
            # if filename is one of the files being hipified for this extension
            if (is_pytorch_extension and any(s.endswith(filename) for s in all_files)):
                header_dir = None
                header_filepath = None
                # If include_current_dir True, look first in same dir as the including source file
                if include_current_dir:
                    header_dir_to_check = os.path.dirname(fin_path)
                    header_path_to_check = os.path.abspath(os.path.join(header_dir_to_check, f))
                    if os.path.exists(header_path_to_check):
                        header_dir = header_dir_to_check
                        header_filepath = header_path_to_check
                # If not found, look in include dirs one by one and first match wins
                if header_filepath is None:
                    for header_include_dir in header_include_dirs:
                        header_dir_to_check = os.path.join(output_directory, header_include_dir)
                        header_path_to_check = os.path.abspath(os.path.join(header_dir_to_check, f))
                        if os.path.exists(header_path_to_check):
                            header_dir = header_dir_to_check
                            header_filepath = header_path_to_check
                # If header file not found, keep as is
                if header_filepath is None:
                    return m.group(0)
                # Hipify header file first if needed
                if header_filepath not in HIPIFY_FINAL_RESULT:
                    preprocess_file_and_save_result(output_directory,
                                                    header_filepath,
                                                    all_files, header_include_dirs, stats, hip_clang_launch,
                                                    is_pytorch_extension, clean_ctx, show_progress)
                elif header_filepath in HIPIFY_FINAL_RESULT:
                    header_result = HIPIFY_FINAL_RESULT[header_filepath]
                    if header_result.current_state == CurrentState.INITIALIZED:
                        # get_hip_file_path needs a relative path to work correctly
                        header_rel_path = os.path.relpath(header_filepath, output_directory)
                        header_fout_path = os.path.abspath(os.path.join(output_directory,
                                                                        get_hip_file_path(header_rel_path, is_pytorch_extension)))
                        header_result.hipified_path = header_fout_path
                        HIPIFY_FINAL_RESULT[header_filepath] = header_result
                        return templ.format(os.path.relpath(header_fout_path if header_fout_path is not None
                                                            else header_filepath, header_dir))
                hipified_header_filepath = HIPIFY_FINAL_RESULT[header_filepath].hipified_path
                return templ.format(_to_unix_path(os.path.relpath(hipified_header_filepath if hipified_header_filepath is not None
                                                                  else header_filepath, header_dir)))

            return m.group(0)
        return repl
    output_source = RE_QUOTE_HEADER.sub(mk_repl('#include "{0}"', True), output_source)
    output_source = RE_ANGLE_HEADER.sub(mk_repl('#include <{0}>', False), output_source)
    output_source = RE_THC_GENERIC_FILE.sub(mk_repl('#define THC_GENERIC_FILE "{0}"'), output_source)

    # CMakeLists.txt rewrites
    if filepath.endswith('CMakeLists.txt'):
        output_source = output_source.replace('CUDA', 'HIP')
        output_source = output_source.replace('THC', 'THH')
        output_source = RE_CU_SUFFIX.sub('.hip', output_source)

    # Perform Kernel Launch Replacements
    if not hip_clang_launch:
        output_source = processKernelLaunches(output_source, stats)

    # Replace std:: with non-std:: versions
    if (filepath.endswith((".cu", ".cuh"))) and "PowKernel" not in filepath:
        output_source = replace_math_functions(output_source)

    # Include header if device code is contained.
    output_source = hip_header_magic(output_source)

    # Replace the extern __shared__
    # NOTE: No longer needed after transition from hcc to hipclang.
    # output_source = replace_extern_shared(output_source)

    # Don't write out identical hipified files for extensions if dirpath has not changed
    if (
        is_pytorch_extension
        and orig_output_source == output_source
        and os.path.dirname(fin_path) == os.path.dirname(fout_path)
    ):
        hipify_result.hipified_path = fin_path
        hipify_result.status = "[skipped, no changes]"
        hipify_result.current_state = CurrentState.DONE
        return hipify_result

    # Add hipify breadcrumb for C-style files to avoid re-hipification
    if fin_path != fout_path and match_extensions(fin_path, (".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".hpp")):
        output_source = HIPIFY_C_BREADCRUMB + output_source

    do_write = True
    if os.path.exists(fout_path):
        with open(fout_path, encoding='utf-8') as fout_old:
            do_write = fout_old.read() != output_source
    if do_write:
        try:
            with clean_ctx.open(fout_path, 'w', encoding='utf-8') as fout:
                fout.write(output_source)
            hipify_result.hipified_path = fout_path
            hipify_result.status = "[ok]"
            hipify_result.current_state = CurrentState.DONE
            return hipify_result
        except OSError as e:
            print(f'{bcolors.WARNING}Failed to save {fout_path} with "{e.strerror}", leaving {fin_path} unchanged.{bcolors.ENDC}',
                  file=sys.stderr)
            hipify_result.hipified_path = fin_path
            hipify_result.status = "[skipped, no permissions]"
            hipify_result.current_state = CurrentState.DONE
            return hipify_result
    else:
        hipify_result.hipified_path = fout_path
        hipify_result.status = "[skipped, already hipified]"
        hipify_result.current_state = CurrentState.DONE
        return hipify_result

def file_specific_replacement(filepath, search_string, replace_string, strict=False) -> None:
    with openf(filepath, "r+") as f:
        contents = f.read()
        if strict:
            contents = re.sub(fr'\b({re.escape(search_string)})\b', lambda x: replace_string, contents)
        else:
            contents = contents.replace(search_string, replace_string)
        f.seek(0)
        f.write(contents)
        f.truncate()


def file_add_header(filepath, header) -> None:
    with openf(filepath, "r+") as f:
        contents = f.read()
        if header[0] != "<" and header[-1] != ">":
            header = f'"{header}"'
        contents = (f'#include {header} \n') + contents
        f.seek(0)
        f.write(contents)
        f.truncate()


def fix_static_global_kernels(in_txt):
    """Static global kernels in HIP results in a compilation error."""
    in_txt = in_txt.replace(" __global__ static", "__global__")
    return in_txt


RE_INCLUDE = re.compile(r"#include .*\n")


def extract_arguments(start, string):
    """
    Return the list of arguments in the upcoming function parameter closure.
    Example:
    string (input): '(blocks, threads, 0, THCState_getCurrentStream(state))'
    arguments (output): [{'start': 1, 'end': 7}, {'start': 8, 'end': 16}, \
        {'start': 17, 'end': 19}, {'start': 20, 'end': 53}]
    """

    arguments = []
    closures = {
        "<": 0,
        "(": 0
    }
    current_position = start
    argument_start_pos = current_position + 1

    # Search for final parenthesis
    while current_position < len(string):
        if string[current_position] == "(":
            closures["("] += 1
        elif string[current_position] == ")":
            closures["("] -= 1
        elif string[current_position] == "<":
            closures["<"] += 1
        elif string[current_position] == ">" and string[current_position - 1] != "-" and closures["<"] > 0:
            closures["<"] -= 1

        # Finished all arguments
        if closures["("] == 0 and closures["<"] == 0:
            # Add final argument
            arguments.append({"start": argument_start_pos, "end": current_position})
            break

        # Finished current argument
        if closures["("] == 1 and closures["<"] == 0 and string[current_position] == ",":
            arguments.append({"start": argument_start_pos, "end": current_position})
            argument_start_pos = current_position + 1

        current_position += 1

    return arguments


def str2bool(v : str) -> bool:
    """ArgumentParser doesn't support type=bool. Thus, this helper method will convert
    from possible string types to True / False."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def hipify(
    project_directory: str,
    show_detailed: bool = False,
    extensions: Iterable = (".cu", ".cuh", ".c", ".cc", ".cpp", ".h", ".in", ".hpp"),
    header_extensions: Iterable = (".cuh", ".h", ".hpp"),
    output_directory: str = "",
    header_include_dirs: Iterable = (),
    includes: Iterable = ('*',),
    extra_files: Iterable = (),
    out_of_place_only: bool = False,
    ignores: Iterable = (),
    show_progress: bool = True,
    hip_clang_launch: bool = False,
    is_pytorch_extension: bool = False,
    hipify_extra_files_only: bool = False,
    clean_ctx: GeneratedFileCleaner | None = None
) -> HipifyFinalResult:
    if project_directory == "":
        project_directory = os.getcwd()

    # Verify the project directory exists.
    if not os.path.exists(project_directory):
        print("The project folder specified does not exist.")
        sys.exit(1)

    # If no output directory, provide a default one.
    if not output_directory:
        project_directory.rstrip("/")
        output_directory = project_directory + "_amd"

    if project_directory != output_directory:
        includes = [include.replace(project_directory, output_directory) for include in includes]
        ignores = [ignore.replace(project_directory, output_directory) for ignore in ignores]

    # Copy from project directory to output directory if not done already.
    if not os.path.exists(output_directory):
        shutil.copytree(project_directory, output_directory)

    includes = list(map(_to_unix_path, includes))
    ignores = list(map(_to_unix_path, ignores))

    all_files = list(matched_files_iter(output_directory, includes=includes,
                                        ignores=ignores, extensions=extensions,
                                        out_of_place_only=out_of_place_only,
                                        is_pytorch_extension=is_pytorch_extension))
    all_files_set = set(all_files)

    for f in extra_files:
        if not os.path.isabs(f):
            f = os.path.join(output_directory, f)
        if f not in all_files_set:
            all_files.append(f)

    # List all files in header_include_paths to ensure they are hipified
    from pathlib import Path
    for header_include_dir in header_include_dirs:
        if os.path.isabs(header_include_dir):
            header_include_dir_path = Path(header_include_dir)
        else:
            header_include_dir_path = Path(os.path.join(output_directory, header_include_dir))
        all_files.extend(
            str(path) for path in header_include_dir_path.rglob('*') if path.is_file()
            and _fnmatch(str(path), includes)
            and (not _fnmatch(str(path), ignores))
            and match_extensions(path.name, header_extensions)
        )

    if clean_ctx is None:
        clean_ctx = GeneratedFileCleaner(keep_intermediates=True)

    # Preprocessing statistics.
    stats: dict[str, list] = {"unsupported_calls": [], "kernel_launches": []}

    for filepath in (all_files if not hipify_extra_files_only else extra_files):
        preprocess_file_and_save_result(output_directory, filepath, all_files, header_include_dirs,
                                        stats, hip_clang_launch, is_pytorch_extension, clean_ctx, show_progress)

    print(bcolors.OKGREEN + "Successfully preprocessed all matching files." + bcolors.ENDC, file=sys.stderr)

    # Show detailed summary
    if show_detailed:
        compute_stats(stats)

    return HIPIFY_FINAL_RESULT
<|endoftext|># mypy: allow-untyped-defs
import gc
import sys
from typing import Any, NamedTuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
logger = logging.getLogger(__name__)

def observe_garbage(observer):
    enabled = True

    def disable() -> None:
        # when GC runs during exit, things like `sys` will already be unloaded
        # so we have to disable the callback to avoid hitting errors.
        nonlocal enabled
        enabled = False
    atexit.register(disable)

    def gc_callback(phase, info) -> None:
        nonlocal enabled
        if not enabled:
            return
        if phase == "start":
            gc.set_debug(gc.DEBUG_SAVEALL)
        elif phase == "stop":
            orig_trace = sys.getprofile()
            self_return = [False]

            def do_collect(*args, **kwargs):
                nonlocal enabled
                if not self_return[0]:
                    self_return[0] = True
                else:
                    sys.setprofile(orig_trace)
                    enabled = False
                    try:
                        # things in gc.garbage have survived a collection
                        # so to free them we have to collect a generation greater than them
                        # but that might _also_ free other stuff and we don't want to miss
                        # that stuff. So we have to now force gc at the highest level here,
                        # report all of what we found, _then_ we can free it up.
                        if info['generation'] != 2:
                            gc.collect()
                        observer(gc.garbage)
                        gc.garbage.clear()
                        # we have to re-run GC to clean up the cycles
                        # we saved from before.
                        gc.set_debug(0)
                        before = torch.cuda.memory_allocated()
                        gc.collect()
                        after = torch.cuda.memory_allocated()
                        if before != after:
                            logger.warning("CUDA Memory changed during GC, %d bytes freed.", before - after)
                    finally:
                        enabled = True
                if orig_trace is not None:
                    return orig_trace(*args, **kwargs)
            sys.setprofile(do_collect)

    gc.callbacks.append(gc_callback)

    # provide a way to disarm the callback
    def remove() -> None:
        gc.callbacks.remove(gc_callback)
    return remove

# Function to visualize cycles adapted from refcycle:
# Copyright 2013 Mark Dickinson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def _get_cell_type():
    def f(x=None):
        return lambda: x
    return type(f().__closure__[0])

CellType = _get_cell_type()

def annotated_references(obj):
    """
    Return known information about references held by the given object.

    Returns a mapping from referents to lists of descriptions.  Note that there
    may be more than one edge leading to any particular referent; hence the
    need for a list.  Descriptions are currently strings.

    """
    references: dict[int, list[str]] = {}

    def add_reference(name, obj) -> None:
        references.setdefault(id(obj), []).append(name)

    def add_attrs(*attrs) -> None:
        for attr in attrs:
            if hasattr(obj, attr):
                add_reference(attr, getattr(obj, attr))

    def add_cell_references() -> None:
        try:
            add_attrs("cell_contents")
        except ValueError:
            # if cell_contents is empty,
            # accessing it raises ValueError
            # in this case there is no object to
            # annotate
            pass

    def add_function_references() -> None:
        add_attrs("__defaults__",
                  "__closure__",
                  "__globals__",
                  "__code__",
                  "__name__",
                  "__module__",
                  "__doc__"
                  "__qualname__",
                  "__annotations__",
                  "__kwdefaults__")


    def add_sequence_references() -> None:
        for position, item in enumerate(obj):
            add_reference(f"[{position}]", item)

    def add_dict_references() -> None:
        for key, value in obj.items():
            add_reference("key", key)
            add_reference(f"[{repr(key)}]", value)

    def add_set_references() -> None:
        for elt in obj:
            add_reference("element", elt)

    def add_bound_method_references() -> None:
        add_attrs("__self__", "__func__", "im_class")

    def add_weakref_references() -> None:
        # For subclasses of weakref, we can't reliably distinguish the
        # callback (if any) from other attributes.
        if type(obj) is weakref.ref:
            referents = gc.get_referents(obj)
            if len(referents) == 1:
                target = referents[0]
                add_reference("__callback__", target)


    def add_frame_references() -> None:
        f_locals = obj.f_locals
        add_attrs("f_back", "f_code", "f_builtins", "f_globals", "f_trace", "f_locals")
        # Some badly-behaved code replaces the f_locals dict with
        # something that doesn't support the full dict interface.  So we
        # only continue with the annotation if f_locals is a Python dict.
        if type(f_locals) is dict:
            for name, local in obj.f_locals.items():
                add_reference(f"local {name}", local)

    def add_getset_descriptor_references() -> None:
        add_attrs("__objclass__", "__name__", "__doc__")

    type_based_references = {
        tuple: add_sequence_references,
        list: add_sequence_references,
        dict: add_dict_references,
        set: add_set_references,
        frozenset: add_set_references,
        types.FunctionType: add_function_references,
        types.FrameType: add_frame_references,
        CellType: add_cell_references,
        types.MethodType: add_bound_method_references,
        weakref.ref: add_weakref_references,
        types.GetSetDescriptorType: add_getset_descriptor_references,
    }

    for type_ in type(obj).__mro__:
        if type_ in type_based_references:
            type_based_references[type_]()

    add_attrs("__dict__", "__class__")
    if isinstance(obj, type):
        add_attrs("__mro__")

    return references

###############################################################################
# Object annotations.


BASE_TYPES = (int, float, complex, type(None), str, bytes)
FRAME_FILENAME_LIMIT = 32

def object_annotation(obj):
    """
    Return a string to be used for Graphviz nodes.

    The string should be short but as informative as possible.
    """

    def format_sequence(obj):
        body = ','.join(repr(x) if isinstance(x, BASE_TYPES) else type(x).__name__ for x in obj[:8])
        if len(obj) > 8:
            body = f'{body}, ...{len(obj) - 8}'
        return body

    # For basic types, use the repr.
    if isinstance(obj, BASE_TYPES):
        return repr(obj)
    if type(obj).__name__ == 'function':
        return f"function\n{obj.__name__}"
    elif isinstance(obj, types.MethodType):
        try:
            func_name = obj.__func__.__qualname__
        except AttributeError:
            func_name = "<anonymous>"
        return f"instancemethod\n{func_name}"
    elif isinstance(obj, list):
        return f"[{format_sequence(obj)}]"
    elif isinstance(obj, tuple):
        return f"({format_sequence(obj)})"
    elif isinstance(obj, dict):
        return f"dict[{len(obj)}]"
    elif isinstance(obj, types.ModuleType):
        return f"module\n{obj.__name__}"
    elif isinstance(obj, type):
        return f"type\n{obj.__name__}"
    elif isinstance(obj, weakref.ref):
        referent = obj()
        if referent is None:
            return "weakref (dead referent)"
        else:
            return f"weakref to id 0x{id(referent):x}"
    elif isinstance(obj, types.FrameType):
        filename = obj.f_code.co_filename
        if len(filename) > FRAME_FILENAME_LIMIT:
            filename = "..." + filename[-(FRAME_FILENAME_LIMIT - 3):]
        return f"frame\n{filename}:{obj.f_lineno}"
    elif is_cuda_tensor(obj):
        return f"object\n{type(obj).__module__}.{type(obj).__name__} ({obj.shape})"
    else:
        return f"object\n{type(obj).__module__}.{type(obj).__name__}"



class Node(NamedTuple):
    label: str
    context: str | None
    root: bool
    referrents: list[tuple[str, int]]

def create_graph(objects, *, context=None, filter=None):
    if context is None:
        context = cuda_allocation_context()
    if filter is None:
        filter = is_cuda_tensor

    objects = [obj for obj in objects if not isinstance(obj, weakref.ProxyTypes)]
    nodes = [Node(object_annotation(obj), context(obj), filter(obj), []) for obj in objects]
    node_referrers: list[list[int]] = [[] for obj in objects]

    id_to_node = {id(obj): i for i, obj in enumerate(objects)}
    for obj in objects:
        fidx = id_to_node[id(obj)]
        f = nodes[fidx]
        references = annotated_references(obj)
        for referrent in gc.get_referents(obj):
            rid = id(referrent)
            tidx = id_to_node.get(rid)
            if tidx is None:
                continue
            labels = references.get(rid, ["?"])
            node_referrers[tidx].append(fidx)
            for label in labels:
                f.referrents.append((label, tidx))

    to_search = [i for i, n in enumerate(nodes) if n.root]
    to_keep = set()
    while to_search:
        idx = to_search.pop()
        if idx in to_keep:
            continue
        to_keep.add(idx)
        referrers = node_referrers[idx]
        to_search.extend(referrers)
    id_to_filtered_id: dict[int, int] = {}
    filtered: list[Any] = []
    for i, n in enumerate(nodes):
        if i in to_keep:
            id_to_filtered_id[i] = len(id_to_filtered_id)
            filtered.append(n)
    for n in filtered:
        n.referrents[:] = [(label, id_to_filtered_id[idx])
                           for (label, idx) in n.referrents
                           if idx in id_to_filtered_id]
    return filtered

def escape(n):
    return json.dumps(n)


def is_cuda_tensor(obj):
    return (
        isinstance(obj, torch.Tensor) and
        obj.device.type == "cuda" and
        not torch._subclasses.fake_tensor.is_fake_tensor(obj)
    )

def cuda_allocation_context():
    snapshot = torch.cuda.memory._snapshot()
    addr_to_frame = {}
    for seg in snapshot['segments']:
        addr = seg['address']
        for blk in seg['blocks']:
            if blk['state'] == 'active_allocated':
                frames, _real_size = _block_extra(blk)
                addr_to_frame[addr] = frames
            addr += blk['size']

    def object_context(obj):
        if is_cuda_tensor(obj):
            addr = obj.untyped_storage().data_ptr()
            frames = addr_to_frame.get(addr)
            if frames is not None:
                return '\n'.join(_frames_fmt(frames, full_filename=True))
        return None
    return object_context

def to_dot(nodes):
    lines = ["digraph GraphName {", "node [shape=rect];", 'rankdir=LR;']
    for i, n in enumerate(nodes):
        lines.append(f'{i} [label={escape(n.label)}, color={"red" if n.root else "black"}];')

    for i, f in enumerate(nodes):
        for label, j in f.referrents:
            lines.append(f'{i} -> {j} [label = {escape(label)}]')
    lines.append("}\n")
    return '\n'.join(lines)

_template = """
<!DOCTYPE html>
<html>
<head>
  <style>
    body {
      margin: 0;
      padding: 0;
      overflow: hidden;
    }

    #container {
      display: flex;
      flex-direction: column;
      height: 100vh;
    }

    #main {
      flex: 2;
      height: 60vh;
      overflow: clip;
    }

    #preContainer {
      flex: 1;
      height: 40vh;
      overflow: auto;
    }

    pre {
      margin: 0;
      padding: 10px;
    }
  </style>
</head>
<body>
  <div id="container">
    <div id="main">
    </div>
    <div id="preContainer">
      <pre id="stacktrace">Mouse over tensor objects to see where they were allocated.</pre>
    </div>
  </div>
<script src='https://cdnjs.cloudflare.com/ajax/libs/viz.js/1.8.0/viz-lite.js'></script>
<script>
let dot = $DOT
let image = Viz(dot, {format: 'svg', 'totalMemory': 1024*1024*1024});
let main = document.getElementById('main')
main.innerHTML = image
let svg = main.firstElementChild
// Panning and zooming logic
let isPanning = false;
let startX, startY;
let viewBox = { x: 0, y: 0, width: parseFloat(svg.getAttribute('width')), height: parseFloat(svg.getAttribute('height')) };
svg.removeAttribute('width');
svg.removeAttribute('height');
function updateViewBox() {
    svg.setAttribute('viewBox', `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`);
}
updateViewBox()
svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
svg.addEventListener('mousedown', function(e) {
    isPanning = true;
    startX = e.clientX;
    startY = e.clientY;
});
svg.addEventListener('mousemove', function(e) {
    if (!isPanning) return;
    const dx = (e.clientX - startX) * (viewBox.width / svg.clientWidth);
    const dy = (e.clientY - startY) * (viewBox.height / svg.clientHeight);
    viewBox.x -= dx;
    viewBox.y -= dy;
    startX = e.clientX;
    startY = e.clientY;
    updateViewBox();
});
svg.addEventListener('mouseup', function() {
    isPanning = false;
});
svg.addEventListener('mouseleave', function() {
    isPanning = false;
});
svg.addEventListener('wheel', function(e) {
    e.preventDefault();
    const zoomFactor = 0.1;
    const zoomAmount = e.deltaY > 0 ? 1 + zoomFactor : 1 - zoomFactor;
    // Calculate mouse position relative to the SVG
    const rect = svg.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const mouseXRel = mouseX / svg.clientWidth;
    const mouseYRel = mouseY / svg.clientHeight;
    // Adjust viewBox to zoom around the mouse position
    const newWidth = viewBox.width * zoomAmount;
    const newHeight = viewBox.height * zoomAmount;
    viewBox.x += (viewBox.width - newWidth) * mouseXRel;
    viewBox.y += (viewBox.height - newHeight) * mouseYRel;
    viewBox.width = newWidth;
    viewBox.height = newHeight;
    updateViewBox();
});
$LISTENERS
</script>
</body>
</html>
"""
_listener_template = """
document.getElementById('node{id}').addEventListener('mouseover', function(event) {{
  document.getElementById("stacktrace").textContent = {stack}
}})
"""
def to_html(nodes):
    listeners = []
    for i, n in enumerate(nodes):
        if n.context is None:
            continue
        s = _listener_template.format(id=str(i + 1), stack=escape(f'{n.label}:\n{n.context}'))
        listeners.append(s)
    dot = to_dot(nodes)
    return _template.replace('$DOT', repr(dot)).replace('$LISTENERS', '\n'.join(listeners))

def observe_tensor_cycles(callback):
    torch.cuda.memory._record_memory_history(max_entries=100000)

    def observer(garbage) -> None:
        if garbage:
            if not any(is_cuda_tensor(obj) for obj in garbage):
                logger.info("No CUDA Tensors found in garbage")
                return
            callback(to_html(create_graph(garbage)))
    return observe_garbage(observer)


def warn_tensor_cycles():
    """
    Install a warning that reports whenever a cycle that is holding CUDA memory is observed.

    The warning produces an .html file that visualizes the cycle,
    and links it to the stack frame that allocated the CUDA tensor.

    Reference cycles are freed by the cycle collector rather than being cleaned up
    when the objects in the cycle first become unreachable. If a cycle points to a tensor,
    the CUDA memory for that tensor will not be freed until garbage collection runs.
    Accumulation of CUDA allocations can lead to out of memory errors (OOMs), as well as
    non-deterministic allocation behavior which is harder to debug.
    """
    logger.info("Watching Python reference cycles for CUDA Tensors.")

    def write_and_log(html) -> None:
        with NamedTemporaryFile('w', suffix='.html') as f:
            f.write(html)
            logger.warning('Reference cycle includes a CUDA Tensor see visualization of cycle %s', f.name)
    return observe_tensor_cycles(write_and_log)
<|endoftext|># mypy: allow-untyped-defs
"""
Call record classes for DebugMode: _DebugCall and its subclasses.
"""

import math
from typing import Any, TYPE_CHECKING

import torch

# Import _utils module for mutable globals
from torch.utils._debug_mode import _utils
from torch.utils._debug_mode._utils import (
    _arg_to_str,
    _get_op_name,
    _get_stack_trace,
    _maybe_get_autograd_trace,
    REDISTRIBUTE_FUNC,
    TensorIdTracker,
)
from torch.utils._pytree import tree_all, tree_map


if TYPE_CHECKING:
    from torch._dynamo.device_interface import DeviceInterface


class _DebugCall:
    """Base class for tracking operator calls in DebugMode"""

    def __init__(
        self,
        call_depth: int,
        record: dict[str, Any] | None = None,
        log: dict[str, Any] | None = None,
        stack: bool = False,
    ) -> None:
        self.call_depth = call_depth
        if stack:
            self.stack_trace = _get_stack_trace()
            self.fwd_stack_trace = _maybe_get_autograd_trace()

        # results from dispatch hooks
        self.record = record
        self.log = log
        self.output_str: str | None = None

    def stringify_args(
        self, attributes: list[str], tensor_memo: TensorIdTracker | None = None
    ) -> None:
        """
        To reduce memory consumption, this method stringifies args/kwargs, stores the result, and deletes original args/kwargs.
        """
        raise NotImplementedError(
            "Subclasses must implement stringify_args(), even if no-op"
        )

    def stringify_output(
        self,
        output: Any,
        attributes: list[str],
        tensor_memo: TensorIdTracker | None = None,
    ) -> None:
        """Store stringified version of call output in self.output_str"""
        if tree_all(lambda x: x is None, output):
            return
        output_str = tree_map(lambda x: _arg_to_str(x, attributes, tensor_memo), output)
        self.output_str = f"  ->  {str(output_str)}"

    def render(self, attributes: list[str]) -> str:
        raise NotImplementedError("Subclasses must implement string render()")

    def __repr__(self) -> str:
        return self.render([])


class _OpCall(_DebugCall):
    """Normal operator call"""

    def __init__(
        self,
        op,
        args: tuple,
        kwargs: dict,
        call_depth: int,
        stack: bool = False,
    ) -> None:
        super().__init__(call_depth, stack=stack)
        self.op = op
        self.args = args
        self.kwargs = kwargs

        self.args_str: str | None = None
        self.kwargs_str: str | None = None

    def stringify_args(
        self, attributes: list[str], tensor_memo: TensorIdTracker | None = None
    ) -> None:
        self.args_str = ", ".join(
            _arg_to_str(arg, attributes, tensor_memo) for arg in self.args
        )
        if self.kwargs:
            self.kwargs_str = ", " + ", ".join(
                f"{k}={_arg_to_str(v, attributes, tensor_memo)}"
                for k, v in self.kwargs.items()
            )
        else:
            self.kwargs_str = ""
        del self.args
        del self.kwargs

    def render(self, attributes: list[str]) -> str:
        if self.args_str is not None:
            args_str = self.args_str
        else:
            args_str = ", ".join(_arg_to_str(arg, attributes) for arg in self.args)

        if self.kwargs_str is not None:
            kwargs_str = self.kwargs_str
        else:
            if self.kwargs:
                kwargs_str = ", " + ", ".join(
                    f"{k}={_arg_to_str(v, attributes)}" for k, v in self.kwargs.items()
                )
            else:
                kwargs_str = ""

        if isinstance(self.op, torch._ops.OpOverload):
            op_name = self.op.__qualname__
        elif hasattr(self.op, "__module__") and hasattr(self.op, "__name__"):
            op_name = f"{self.op.__module__}.{self.op.__name__}"
        else:
            op_name = str(self.op)

        base_str = f"{op_name}({args_str}{kwargs_str})"

        if self.output_str:
            base_str += self.output_str
        if self.log:
            base_str += f"  # {self.log}"
        return base_str

    def __iter__(self):
        # for BC; tuple(self) returns (op, args, kwargs, call_depth)
        if self.args_str is not None:
            yield from [self.op, self.args_str, self.kwargs_str, self.call_depth]
        else:
            yield from [self.op, self.args, self.kwargs, self.call_depth]


class _RedistributeCall(_DebugCall):
    def __init__(
        self,
        arg,
        src_placement,
        dst_placement,
        transform_info_str,
        call_depth,
        stack=False,
        is_explicit=False,
    ) -> None:
        super().__init__(call_depth, stack=stack)
        self.arg = arg
        self.src_placement = src_placement
        self.dst_placement = dst_placement
        self.transform_info_str = transform_info_str
        self.is_explicit = is_explicit
        self.is_outer_call = isinstance(arg, int)

        self.arg_str: str | None = None

    def stringify_args(
        self, attributes: list[str], tensor_memo: TensorIdTracker | None = None
    ) -> None:
        self.arg_str = f"{_arg_to_str(self.arg, attributes, tensor_memo)}"
        del self.arg

    def render(self, attributes: list[str]) -> str:
        if self.arg_str is not None:
            arg_str = self.arg_str
        else:
            arg_str = f"{_arg_to_str(self.arg, attributes)}"

        if self.transform_info_str is not None:  # prioritize over src/dst placements
            placement_str = f"trace: {self.transform_info_str}"
        else:
            src_placement_str = _arg_to_str(self.src_placement, attributes)
            dst_placement_str = _arg_to_str(self.dst_placement, attributes)
            placement_str = f"{src_placement_str} -> {dst_placement_str}"

        # DebugMode will add redistribute_input logs at 2 levels,
        # once per redistribute decision, and once per redistributed input.
        # We only annotate [implicit/explicit] logs on the former (outer-level call).
        if self.is_outer_call:
            annotation = " [implicit] "
        elif self.is_explicit:
            annotation = " [explicit] "
        else:
            annotation = ""

        base_str = f"{REDISTRIBUTE_FUNC}{annotation}({arg_str}, {placement_str})"
        if self.output_str:
            base_str += self.output_str
        return base_str

    def __iter__(self):
        # for BC; tuple(self) returns (op, placement info, kwargs, call_depth)
        if self.arg_str is not None:
            arg = self.arg_str
        else:
            arg = self.arg

        yield REDISTRIBUTE_FUNC
        if self.transform_info_str:
            yield [arg, self.transform_info_str]
        else:
            yield [arg, self.src_placement, self.dst_placement]
        yield {}
        yield self.call_depth


class _OutputPlacementCall(_DebugCall):
    """Records output placement for a DTensor op."""

    def __init__(self, placements_str: str, call_depth: int) -> None:
        super().__init__(call_depth)
        self.placements_str = placements_str

    def stringify_args(
        self, attributes: list[str], tensor_memo: TensorIdTracker | None = None
    ) -> None:
        pass  # Already stringified

    def render(self, attributes: list[str]) -> str:
        return f"-> output: {self.placements_str}"


class _TritonKernelCall(_DebugCall):
    """Triton kernel call from Inductor"""

    def __init__(
        self,
        kernel_name: str,
        kwargs: dict[str, Any],
        call_depth: int,
    ):
        super().__init__(call_depth)
        self.kernel_name = kernel_name
        self.kwargs = kwargs
        self.kwargs_str: str | None = None

        self.pre_hashes: dict[str, Any] | None = None
        self.post_hashes: dict[str, Any] | None = None

    def stringify_args(
        self, attributes: list[str], tensor_memo: TensorIdTracker | None = None
    ) -> None:
        # Optionally hash kernel inputs before launch
        if hash_fn := _utils._TRITON_INPUT_HASH_FN:
            self.pre_hashes = {
                k: hash_fn(v)
                for k, v in self.kwargs.items()
                if isinstance(v, torch.Tensor)
            }

        if self.kwargs:
            self.kwargs_str = ", ".join(
                f"{k}={_arg_to_str(v, attributes, tensor_memo)}"
                for k, v in self.kwargs.items()
            )
        else:
            self.kwargs_str = ""

    def render(self, attributes: list[str]) -> str:
        base_str = f"[triton] {self.kernel_name}({self.kwargs_str})"
        if self.pre_hashes:
            pre_hashes_str = ", ".join(f"{k}: {v}" for k, v in self.pre_hashes.items())
            pre_hashes_str = (
                "\n  "
                + "  " * self.call_depth
                + f"# pre-kernel hashes: {{{pre_hashes_str}}}"
            )
        else:
            pre_hashes_str = ""
        if self.post_hashes:
            post_hashes_str = ", ".join(
                f"{k}: {v}" for k, v in self.post_hashes.items()
            )
            post_hashes_str = (
                "\n  "
                + "  " * self.call_depth
                + f"# post-kernel hashes: {{{post_hashes_str}}}"
            )
        else:
            post_hashes_str = ""
        return f"{base_str}{pre_hashes_str}{post_hashes_str}\n"

    def finalize(self, device_interface: "DeviceInterface"):
        # synchronize -> hash/store kernel results
        device_interface.synchronize(device_interface.current_device())
        if _utils._RECORD_TRITON_OUTPUTS:
            self.record = {
                "output": {
                    k: v.clone() if isinstance(v, torch.Tensor) else v
                    for k, v in self.kwargs.items()
                }
            }
        if hash_fn := _utils._TRITON_OUTPUT_HASH_FN:
            self.post_hashes = {
                k: hash_fn(v)
                for k, v in self.kwargs.items()
                if isinstance(v, torch.Tensor)
            }

        # don't store tensors
        del self.kwargs

    def __iter__(self):
        yield from [self.kernel_name, (), self.kwargs_str, self.call_depth]


class _AnnotateCall(_DebugCall):
    """Custom annotation call"""

    def __init__(
        self, tag: Any, header: str, call_depth: int, stack: bool = False
    ) -> None:
        super().__init__(call_depth, stack=stack)
        self.tag = tag
        self.header = header

    def render(self, attributes: list[str]) -> str:
        return f"[{self.header}] {self.tag}"

    def __iter__(self):
        yield from [
            f"[{self.header}] {self.tag}",
            (),
            {},
            self.call_depth,
        ]


def _serialize_debug_value(value: Any) -> dict[str, Any]:
    if value is None or isinstance(value, (bool, int, str)):
        return {"type": "scalar", "value": value}
    if isinstance(value, float):
        if math.isnan(value):
            return {"type": "nonfinite_float", "value": "nan"}
        if math.isinf(value):
            return {
                "type": "nonfinite_float",
                "value": "inf" if value > 0 else "-inf",
            }
        return {"type": "scalar", "value": value}
    if isinstance(value, tuple):
        return {
            "type": "tuple",
            "items": [_serialize_debug_value(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "items": [_serialize_debug_value(item) for item in value],
        }
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "DebugMode log serialization only supports dicts with string keys, "
                    f"but found key {key!r} of type {type(key).__name__}"
                )
            items.append([key, _serialize_debug_value(item)])
        return {"type": "dict", "items": items}
    raise TypeError(
        "DebugMode log serialization only supports JSON-compatible scalar, "
        f"tuple, list, and dict values, but found {type(value).__name__}"
    )


def _deserialize_debug_value(value: Any) -> Any:
    if not isinstance(value, dict):
        raise ValueError(
            f"Serialized DebugMode value must be a dict, but found {type(value).__name__}"
        )
    value_type = value.get("type")
    if value_type == "scalar":
        return value.get("value")
    if value_type == "nonfinite_float":
        nonfinite_value = value.get("value")
        if nonfinite_value == "nan":
            return float("nan")
        if nonfinite_value == "inf":
            return float("inf")
        if nonfinite_value == "-inf":
            return float("-inf")
        raise ValueError(
            f"Unknown serialized DebugMode non-finite float: {nonfinite_value!r}"
        )
    if value_type == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("Serialized DebugMode tuple must contain an items list")
        return tuple(_deserialize_debug_value(item) for item in items)
    if value_type == "list":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("Serialized DebugMode list must contain an items list")
        return [_deserialize_debug_value(item) for item in items]
    if value_type == "dict":
        items = value.get("items")
        if not isinstance(items, list):
            raise ValueError("Serialized DebugMode dict must contain an items list")
        result = {}
        for item in items:
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(
                    "Serialized DebugMode dict items must be [key, value] pairs"
                )
            key, item_value = item
            if not isinstance(key, str):
                raise ValueError(
                    "Serialized DebugMode dict keys must be strings, "
                    f"but found {type(key).__name__}"
                )
            result[key] = _deserialize_debug_value(item_value)
        return result
    raise ValueError(f"Unknown serialized DebugMode value type: {value_type!r}")


def _serialize_optional_debug_value(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return _serialize_debug_value(value)


def _deserialize_optional_debug_value(value: Any | None) -> Any | None:
    if value is None:
        return None
    return _deserialize_debug_value(value)


def _serialize_common_call_fields(call: _DebugCall) -> dict[str, Any]:
    return {
        "call_depth": call.call_depth,
        "log": _serialize_optional_debug_value(call.log),
        "output_str": call.output_str,
        "stack_trace": getattr(call, "stack_trace", None),
        "fwd_stack_trace": getattr(call, "fwd_stack_trace", None),
    }


def _restore_common_call_fields(call: _DebugCall, data: dict[str, Any]) -> None:
    call.log = _deserialize_optional_debug_value(data.get("log"))
    call.output_str = data.get("output_str")
    stack_trace = data.get("stack_trace")
    if stack_trace is not None:
        call.stack_trace = stack_trace
    fwd_stack_trace = data.get("fwd_stack_trace")
    if fwd_stack_trace is not None:
        call.fwd_stack_trace = fwd_stack_trace


def _op_call_args_str(call: _OpCall) -> str:
    if call.args_str is not None:
        return call.args_str
    return ", ".join(_arg_to_str(arg, []) for arg in call.args)


def _op_call_kwargs_str(call: _OpCall) -> str:
    if call.kwargs_str is not None:
        return call.kwargs_str
    if call.kwargs:
        return ", " + ", ".join(
            f"{key}={_arg_to_str(value, [])}" for key, value in call.kwargs.items()
        )
    return ""


def _redistribute_arg_str(call: _RedistributeCall) -> str:
    if call.arg_str is not None:
        return call.arg_str
    return _arg_to_str(call.arg, [])


def _triton_kwargs_str(call: _TritonKernelCall) -> str:
    if call.kwargs_str is not None:
        return call.kwargs_str
    if call.kwargs:
        return ", ".join(
            f"{key}={_arg_to_str(value, [])}" for key, value in call.kwargs.items()
        )
    return ""


def _serialized_str(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str):
        raise ValueError(
            f"Serialized DebugMode call field {field!r} must be a string, "
            f"but found {type(value).__name__}"
        )
    return value


def _serialized_optional_str(data: dict[str, Any], field: str) -> str | None:
    value = data.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(
            f"Serialized DebugMode call field {field!r} must be a string or None, "
            f"but found {type(value).__name__}"
        )
    return value


def _serialize_debug_call(call: _DebugCall) -> dict[str, Any]:
    data = _serialize_common_call_fields(call)
    if isinstance(call, _OpCall):
        data.update(
            {
                "type": "op",
                "op_name": _get_call_name(call),
                "args_str": _op_call_args_str(call),
                "kwargs_str": _op_call_kwargs_str(call),
            }
        )
    elif isinstance(call, _RedistributeCall):
        data.update(
            {
                "type": "redistribute",
                "arg_str": _redistribute_arg_str(call),
                "src_placement_str": _arg_to_str(call.src_placement, []),
                "dst_placement_str": _arg_to_str(call.dst_placement, []),
                "transform_info_str": call.transform_info_str,
                "is_explicit": call.is_explicit,
                "is_outer_call": call.is_outer_call,
            }
        )
    elif isinstance(call, _OutputPlacementCall):
        data.update(
            {
                "type": "output_placement",
                "placements_str": call.placements_str,
            }
        )
    elif isinstance(call, _TritonKernelCall):
        data.update(
            {
                "type": "triton_kernel",
                "kernel_name": call.kernel_name,
                "kwargs_str": _triton_kwargs_str(call),
                "pre_hashes": _serialize_optional_debug_value(call.pre_hashes),
                "post_hashes": _serialize_optional_debug_value(call.post_hashes),
            }
        )
    elif isinstance(call, _AnnotateCall):
        data.update(
            {
                "type": "annotate",
                "tag": str(call.tag),
                "header": call.header,
            }
        )
    else:
        raise TypeError(f"Unsupported DebugMode call type: {type(call).__name__}")
    return data


def _deserialize_debug_call(data: Any) -> _DebugCall:
    if not isinstance(data, dict):
        raise ValueError(
            f"Serialized DebugMode call must be a dict, but found {type(data).__name__}"
        )
    call_type = data.get("type")
    call_depth = data.get("call_depth")
    if not isinstance(call_depth, int):
        raise ValueError("Serialized DebugMode call must contain an integer call_depth")

    if call_type == "op":
        call = _OpCall(_serialized_str(data, "op_name"), (), {}, call_depth)
        call.args_str = _serialized_str(data, "args_str")
        call.kwargs_str = _serialized_str(data, "kwargs_str")
    elif call_type == "redistribute":
        arg_str = _serialized_str(data, "arg_str")
        call = _RedistributeCall(
            arg_str,
            _serialized_str(data, "src_placement_str"),
            _serialized_str(data, "dst_placement_str"),
            _serialized_optional_str(data, "transform_info_str"),
            call_depth,
            is_explicit=bool(data.get("is_explicit")),
        )
        call.arg_str = arg_str
        # Serialized redistribute calls only keep rendered args, so restore the
        # outer-call classification instead of relying on __init__'s arg type.
        call.is_outer_call = bool(data.get("is_outer_call"))
    elif call_type == "output_placement":
        call = _OutputPlacementCall(_serialized_str(data, "placements_str"), call_depth)
    elif call_type == "triton_kernel":
        call = _TritonKernelCall(_serialized_str(data, "kernel_name"), {}, call_depth)
        call.kwargs_str = _serialized_str(data, "kwargs_str")
        call.pre_hashes = _deserialize_optional_debug_value(data.get("pre_hashes"))
        call.post_hashes = _deserialize_optional_debug_value(data.get("post_hashes"))
    elif call_type == "annotate":
        call = _AnnotateCall(
            _serialized_str(data, "tag"), _serialized_str(data, "header"), call_depth
        )
    else:
        raise ValueError(f"Unknown serialized DebugMode call type: {call_type!r}")

    _restore_common_call_fields(call, data)
    return call


def _get_call_name(call: _DebugCall) -> str:
    """String identifying _DebugCall (e.g. func, kernel, module name)"""
    if isinstance(call, _OpCall):
        return _get_op_name(call.op)
    elif isinstance(call, _TritonKernelCall):
        return call.kernel_name
    elif isinstance(call, _AnnotateCall):
        return f"[{call.header}] {call.tag}"
    elif isinstance(call, _RedistributeCall):
        return REDISTRIBUTE_FUNC
    else:
        return str(call)
<|endoftext|># mypy: allow-untyped-defs
"""
DebugMode: a debugging TorchDispatchMode that intercepts and logs runtime calls.

See torch.utils._debug_mode._mode for the full implementation and docstring.
"""

from torch.utils._debug_mode._calls import (
    _AnnotateCall,
    _DebugCall,
    _get_call_name,
    _OpCall,
    _OutputPlacementCall,
    _RedistributeCall,
    _TritonKernelCall,
)
from torch.utils._debug_mode._mode import (
    DebugInterpreter,
    DebugMode,
    get_active_debug_mode,
)
from torch.utils._debug_mode._utils import (
    _stringify_shape,
    hash_tensor_fn,
    norm_hash_fn,
    TensorIdTracker,
)


__all__ = ["DebugMode", "get_active_debug_mode"]
<|endoftext|># mypy: allow-untyped-defs
"""
Utility functions for DebugMode: tensor formatting, hashing, stack traces, and hook runners.
"""

import inspect
import os
import traceback
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.fx.graph import _parse_stack_trace
from torch.utils._dtype_abbrs import dtype_abbrs
from torch.utils._pytree import tree_map
from torch.utils._traceback import CapturedTraceback
from torch.utils.weak import WeakIdRef


if TYPE_CHECKING:
    from torch.utils._debug_mode._calls import _DebugCall


REDISTRIBUTE_FUNC = "redistribute_input"

# Tracks if we're in inductor benchmarking, and temporarily disables logging
# (for ignoring autotuning kernel launches which don't affect the user-facing result)
_IN_INDUCTOR_BENCHMARK: bool = False
# For record_outputs, log_tensor_hashes hooks for triton kernels.
# Stores kernel outputs in call.record["output"]
_RECORD_TRITON_OUTPUTS: bool = False
# Annotates kernel output hashes, and stores them in call.post_hashes
_TRITON_OUTPUT_HASH_FN: Callable | None = None
# Annotates kernel input hashes, and stores them in call.pre_hashes
_TRITON_INPUT_HASH_FN: Callable | None = None

# registered dispatch call hooks
_DISPATCH_RECORD_HOOKS: list[Callable] = []
_DISPATCH_LOG_HOOKS: list[Callable] = []
_DISPATCH_PRE_LOG_HOOKS: list[Callable] = []


def _stringify_shape(shape) -> str:
    return f"[{', '.join([str(x) for x in shape])}]"


def _stringify_device_mesh(mesh) -> str:
    return f"DM({', '.join([str(s) for s in mesh.shape])})"


def _stringify_placement(placement) -> str:
    return f"[{', '.join([str(p) for p in placement])}]"


def _stringify_attributes(tensor, attributes) -> str:
    pairs = {}
    for attr in attributes:
        if hasattr(tensor, attr):
            pairs[attr] = getattr(tensor, attr)
    if len(pairs) == 0:
        return ""
    return f"{{{', '.join([f'{k}={v}' for k, v in pairs.items()])}}}"


def _stringify_dtensor_spec(spec) -> str:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    return DTensorSpec.format_shard_order_str(spec.placements, spec.shard_order)


class TensorIdTracker:
    def __init__(self) -> None:
        self.tensor_memo: dict[WeakIdRef, int] = {}
        self.next_tensor_id = 0

    def _id(self, tensor) -> int:
        with torch._C._DisablePythonDispatcher():
            o = WeakIdRef(tensor)

            def del_memo() -> None:
                self.tensor_memo.pop(o, None)

            weakref.finalize(tensor, del_memo)
            if o not in self.tensor_memo:
                self.tensor_memo[o] = self.next_tensor_id
                self.next_tensor_id += 1
            return self.tensor_memo[o]


def _tensor_debug_string(tensor, attributes, tensor_memo=None) -> str:
    """Convert tensor to debug string representation."""

    if isinstance(tensor, torch.Tensor):
        tensor_debug_str = f"{dtype_abbrs[tensor.dtype]}{_stringify_shape(tensor.shape)}{_stringify_attributes(tensor, attributes)}"
        id_str = f"${tensor_memo._id(tensor)}" if tensor_memo is not None else ""
        if isinstance(tensor, torch.distributed.tensor.DTensor):
            # omitted device mesh
            return f"dt{id_str}: {tensor_debug_str}| {_stringify_dtensor_spec(tensor._spec)}"
        elif is_fake_tensor(tensor):
            return f"ft{id_str}: {tensor_debug_str}"
        else:
            return f"t{id_str}: {tensor_debug_str}"
    else:
        raise RuntimeError(f"Unsupported tensor type: {type(tensor)}")


def _arg_to_str(arg, attributes, tensor_memo=None) -> str:
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    def to_str(x):
        if isinstance(x, torch.Tensor):
            return _tensor_debug_string(x, attributes, tensor_memo)
        elif isinstance(x, DTensorSpec):
            return _stringify_dtensor_spec(x)
        return x

    arg = tree_map(to_str, arg)
    return str(arg)


def norm_hash_fn(t: torch.Tensor, use_scalar: bool = False) -> torch.Tensor | float:
    """
    from Observer. Computes a hash for a tensor by converting it to float (if needed), making it contiguous,
    replacing NaN/inf values with fixed numbers, and then computing the L1 norm in float64 or complex128.
    This is used to generate a deterministic summary value for tensor comparison.
    """
    with torch._C._DisablePythonDispatcher():
        if not (t.is_floating_point() or t.is_complex()):
            t = t.float()
        t = t.contiguous()

        if t.is_complex():
            t_float = t.to(dtype=torch.complex128)
        else:
            t_float = t.to(dtype=torch.float64)

        out = t_float.norm(p=1)
        if use_scalar:
            return out.item()
        return out


def _compute_rel_diff(hash1, hash2):
    # Relative difference: |hash1 - hash2| / max(|hash1|, |hash2|, eps)
    numerator = abs(hash1 - hash2)
    denominator = max(abs(hash1), abs(hash2), 1e-10)
    return numerator / denominator


def hash_tensor_fn(t: torch.Tensor, use_scalar: bool = False) -> torch.Tensor | int:
    """
    wrapper over torch.hash_tensor
    """
    if isinstance(t, torch.distributed.tensor.DTensor):
        t = t.to_local()

    if t.is_floating_point():
        t_clean = t.to(dtype=torch.float64)
    elif t.is_complex():
        t_clean = t.to(dtype=torch.complex128).view(torch.float64)
    else:
        t_clean = t.to(dtype=torch.int64)

    if t.numel() > 0:
        out = torch.hash_tensor(t_clean)
    else:
        out = torch.zeros((), device=t_clean.device, dtype=torch.uint64)

    if use_scalar:
        return out.item()  # type: ignore[attribute]
    return out


def _get_stack_trace() -> str:
    from torch.fx.experimental.symbolic_shapes import uninteresting_files

    summary = CapturedTraceback.extract().summary()
    summary = summary[:-4]  # filter out DebugMode frames
    summary = [
        frame for frame in summary if frame.filename not in uninteresting_files()
    ]
    summary = traceback.StackSummary.from_list(summary)
    return "".join(summary.format())


def _get_user_stack_trace(stack_trace_str: str) -> str | None:
    # Extract user code stack trace, filtering out torch internals.
    torch_dir = os.path.dirname(inspect.getfile(torch))
    filter_fn = lambda file, name, code: not file.startswith(torch_dir + os.path.sep)  # noqa: E731
    trace = _parse_stack_trace(stack_trace_str, filter_fn=filter_fn)
    if trace:
        return f"File: {trace.file}:{trace.lineno} in {trace.name}, code: {trace.code}"
    return None


def _maybe_get_autograd_trace() -> str | None:
    if torch._C._current_autograd_node() is not None:
        tb = torch._C._current_autograd_node().metadata.get("traceback_")  # type: ignore[attr-defined]
        if tb:
            return "".join(tb)
    return None


def _get_op_name(op) -> str:
    if isinstance(op, torch._ops.OpOverload):
        op_name = op.__qualname__
    elif hasattr(op, "__module__") and hasattr(op, "__name__"):
        op_name = f"{op.__module__}.{op.__name__}"
    else:
        op_name = str(op)
    return op_name


def _run_hook(hook, *args):
    out = hook(*args)
    if out is not None and not isinstance(out, dict):
        raise AssertionError(f"hook must return None or dict, got {type(out).__name__}")
    return out


def _run_dispatch_pre_log_hooks(call: "_DebugCall", func, types, args, kwargs) -> None:
    if _DISPATCH_PRE_LOG_HOOKS:
        for hook in _DISPATCH_PRE_LOG_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, call)
            if hook_out is not None:
                # Store pre-hook results in call.log
                if call.log is None:
                    call.log = {}
                call.log.update(hook_out)


def _run_dispatch_hooks(call: "_DebugCall", func, types, args, kwargs, result) -> None:
    if _DISPATCH_RECORD_HOOKS:
        record = {}
        for hook in _DISPATCH_RECORD_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, result)
            if hook_out is not None:
                record.update(hook_out)
        if record:
            call.record = record

    if _DISPATCH_LOG_HOOKS:
        # Preserve existing log from pre-hooks (e.g., input_hash)
        if call.log is None:
            call.log = {}
        for hook in _DISPATCH_LOG_HOOKS:
            hook_out = _run_hook(hook, func, types, args, kwargs, result)
            if hook_out is not None:
                call.log.update(hook_out)
<|endoftext|># mypy: allow-untyped-defs
"""
DebugMode is a debugging TorchDispatchMode that intercepts and logs runtime calls
to a hierarchical string dump. It logs real tensor, DTensor, and optionally FakeTensor
operations, with some additional handling for DTensor internals.

An example dump from an eager mode DTensor matmul:

    torch.mm(dt$0: f32[8, 8]| S(0), dt$1: f32[8, 32]| S(0))  ->  dt$6: f32[8, 32]| S(0)
      aten::mm(dt$0: f32[8, 8]| S(0), dt$1: f32[8, 32]| S(0))
        redistribute_input(1, S(0) -> R)
          redistribute_input(t$2: f32[1, 32], trace: S(0)->R)
            _c10d_functional::all_gather_into_tensor(t$2: f32[1, 32], 8, 0)  ->  t$3: f32[8, 32]
            _c10d_functional::wait_tensor(t$3: f32[8, 32])  ->  t$3: f32[8, 32]
        aten::mm(t$4: f32[1, 8], t$3: f32[8, 32])  ->  t$5: f32[1, 32]

This mode runs "under" compile, which means it hides itself during compilation, and is re-enabled
at runtime, and DebugMode-related operations won't show up in the compiled region.
DebugMode also provides some visibility into non-torch-dispatch calls (e.g. DTensor redistribute calls,
inductor-generated triton kernels), but requires special handling for these, since dispatch modes
can't intercept them by default.

The mode also provides some extensions for custom debugging (e.g. adding custom dispatch call hooks
via dispatch_hooks), or numerics debugging (e.g. tensor hashing for bitwise equivalence/closeness,
via log_tensor_hashes). These decorators allow annotating string dumps with additional per-call information,
for any region of runtime code.

Usage::

    with DebugMode() as debug_mode:
        result = some_pytorch_operation(tensor_input)
    print(debug_mode.debug_string())
"""

import contextlib
import functools
import json
import logging
import os
from collections.abc import Callable
from typing import Any, IO, TYPE_CHECKING

import torch
from torch._logging import warning_once
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

# Import _utils module for mutable globals
from torch.utils._debug_mode import _utils
from torch.utils._debug_mode._calls import (
    _AnnotateCall,
    _deserialize_debug_call,
    _get_call_name,
    _OpCall,
    _OutputPlacementCall,
    _RedistributeCall,
    _serialize_debug_call,
    _TritonKernelCall,
)
from torch.utils._debug_mode._utils import (
    _compute_rel_diff,
    _get_user_stack_trace,
    _run_dispatch_hooks,
    _run_dispatch_pre_log_hooks,
    _stringify_dtensor_spec,
    hash_tensor_fn,
    norm_hash_fn,
    TensorIdTracker,
)
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _get_current_dispatch_mode_stack,
    TorchDispatchMode,
)
from torch.utils._pytree import (
    keystr,
    tree_all,
    tree_map,
    tree_map_only,
    tree_map_with_path,
)


if TYPE_CHECKING:
    from torch.distributed._tools.mod_tracker import ModTracker


log = logging.getLogger(__name__)

_SERIALIZED_LOG_FORMAT = "torch.utils._debug_mode.DebugMode.logs"
_SERIALIZED_LOG_VERSION = 1


# Counter for active DebugMode instances (fast path for get_active_debug_mode)
_ACTIVE_DEBUG_MODE_COUNT = 0

_annotate_decorated = False


def _ensure_annotate_decorated():
    """
    Lazily apply dont_skip_tracing decorator to DebugMode._annotate, to avoid circular import/initialization issues.
    """
    global _annotate_decorated
    if not _annotate_decorated:
        DebugMode._annotate = torch._dynamo.dont_skip_tracing(DebugMode._annotate)  # type: ignore[has-type]

        # Mark annotate as side-effectful so aot_eager doesn't DCE it.
        from torch.fx.node import _side_effectful_functions

        _side_effectful_functions.add(torch.ops.debug_mode_ops.annotate.default)

        # Register no-op lowering for inductor backend
        from torch._inductor.lowering import register_lowering

        @register_lowering(torch.ops.debug_mode_ops.annotate)
        def _annotate_lowering(tag: str) -> None:
            warning_once(log, 'DebugMode._annotate() is a no-op for backend="inductor"')
            return None

        _annotate_decorated = True


@torch.library.custom_op("debug_mode_ops::annotate", mutates_args=())
def _annotate(tag: str) -> None:
    # This is special-cased in DebugMode.__torch_dispatch__
    return None


@_annotate.register_fake
def _annotate_fake(tag: str) -> None:
    return None


class DebugInterpreter(torch.fx.Interpreter):
    """
    Interpreter class for running aot_eager compiled regions when DebugMode is active,
    instead of using the compiled code. This gives us access to fx.Node metadata to decorate
    and contextualize DebugMode logs (e.g. nn_module_stack, stack_trace, compiled region boundaries).

    Note: this is currently only enabled with DebugMode(run_compile_with_interpreter=True).
    """

    def __init__(self, module, backend):
        super().__init__(module)
        self.mode = get_active_debug_mode()
        if self.mode is None:
            raise RuntimeError("No DebugMode is currently active")

        # for tracking initial nn_module_stack
        self.base_nn_module_stack = list(self.mode.current_nn_module_stack)

        # annotate start of region
        self.backend = backend
        self.mode.operators.append(
            _AnnotateCall(
                "enter", f"{self.backend} region (compile)", self.mode.call_depth
            )
        )

    def run_node(self, n: torch.fx.Node) -> Any:
        if self.mode is None:
            raise RuntimeError("No DebugMode is currently active")

        # handling of nn.Module stack
        if self.mode.record_nn_module and n.op not in ["placeholder", "output"]:
            self.mode._handle_fx_nn_module_stack(
                self.base_nn_module_stack,
                n.meta.get("nn_module_stack", {}),
                n.meta.get("fwd_nn_module_stack", {}),
            )

        # override stack trace with n.meta
        if (
            self.mode.record_stack_trace
            and n.op not in ["placeholder", "output"]
            and (stack_trace := n.meta.get("stack_trace", None)) is not None
        ):
            with self.mode.set_fx_stack_trace(stack_trace):
                return super().run_node(n)
        else:
            return super().run_node(n)

    def run(self, *args, **kwargs):
        if self.mode is None:
            raise RuntimeError("No DebugMode is currently active")
        result = super().run(*args)

        # reset nn.Module stack to pre-compiled region value
        if len(self.mode.current_nn_module_stack) < len(self.base_nn_module_stack):
            warning_once(
                log, "unexpected handling of nn_module_stack in DebugInterpreter"
            )
        while len(self.mode.current_nn_module_stack) > len(self.base_nn_module_stack):
            self.mode._exit_nn_module_call()

        # annotate end of region
        self.mode.operators.append(
            _AnnotateCall(
                "exit", f"{self.backend} region (compile)", self.mode.call_depth
            )
        )

        return result


class DebugMode(TorchDispatchMode):
    def __init__(
        self,
        *,
        record_torchfunction=False,
        record_faketensor=False,
        record_realtensor=True,
        record_tensor_attributes=None,
        record_nn_module=False,
        store_original_args=False,
        record_stack_trace=False,
        record_output=True,
        record_ids=False,
        record_profiler_context=True,
        record_localtensor=True,
        run_compile_with_interpreter=False,
    ) -> None:
        super().__init__()
        import torch.distributed.tensor  # noqa: F401

        _ensure_annotate_decorated()
        self.supports_higher_order_operators = True

        # Pushes DebugMode onto the torchfunction stack, and records __torch_function__ calls as well.
        # WARNING: currently incompatible with torch.compile due to dynamo guard failures.
        self.record_torchfunction = record_torchfunction

        # Records __torch_dispatch__ calls on FakeTensors.
        self.record_faketensor = record_faketensor

        # Records __torch_dispatch__ calls on real tensors.
        self.record_realtensor = record_realtensor

        # Records __torch_dispatch__ calls on LocalTensor.
        self.record_localtensor = record_localtensor

        # Optional list[str] of tensor attributes, to be annotated in the string dump.
        self.record_tensor_attributes = record_tensor_attributes or []

        # Uses ModTracker to record nn.Module entrances.
        # This flag currently has no effect on torch.compiled-regions.
        self.record_nn_module = record_nn_module

        self.module_tracker: ModTracker | None = None
        if self.record_nn_module:
            self.module_tracker_setup()

        # If True, stores call args/kwargs in logs, without immediately stringifying.
        # Defaults to False for memory concerns.
        self.store_original_args = store_original_args

        # For stack trace recording, stores log call stack traces in .stack_trace.
        # For backward graph nodes, will also store the corresponding forward stack traces in .fwd_stack_trace.
        # NOTE: this is only available if autograd tracebacks are being set during the forward pass,
        # e.g. via DebugMode(record_stack_trace=True), or torch.autograd.set_detect_anomaly().
        self.record_stack_trace = record_stack_trace

        # Records call outputs in logs (e.g. for __torch_dispatch__, __torch_function__, redistribute_input)
        self.record_output: bool = record_output

        # Annotates string dumps with graph-style tensor ids, e.g. op($1, $2) -> $3.
        self.record_ids: bool = record_ids

        # Annotates string dumps with profiler.record_function contexts from runtime code.
        # Currently does not preserve contexts inside torch.compile-d regions.
        self.record_profiler_context: bool = record_profiler_context

        # For aot_eager compiled regions, wraps the compiled fx.GraphModule with a DebugInterpreter,
        # and uses it at runtime for node metadata visibility.
        self.run_compile_with_interpreter: bool = run_compile_with_interpreter

        self.reset()

    def reset(self) -> None:
        self.operators = []
        self.call_depth = 0
        self._tensor_memo = TensorIdTracker()
        self._output_info: dict[int, object] = {}
        self.ignored_record_functions = 0
        self.current_nn_module_stack = []
        self.fx_stack_trace = None

    def _track_op_output(self, op_index, result) -> None:
        """Assign IDs to output tensors and store in output_info"""
        self._output_info[op_index] = result

    # Without this override, running torch.compile under DebugMode
    # will force torch.compile to always use the "eager" backend
    # With this, DebugMode will not take effect on torch.compile
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def _record_call(self, call) -> None:
        if _utils._IN_INDUCTOR_BENCHMARK:
            return

        if str(call).startswith("profiler::_record_function"):
            return

        if not self.store_original_args:
            call.stringify_args(
                self.record_tensor_attributes,
                self._tensor_memo if self.record_ids else None,
            )
        if self.fx_stack_trace:
            call.stack_trace = call.fwd_stack_trace = self.fx_stack_trace
        self.operators.append(call)

    def _record_call_output(self, call, output) -> None:
        if not self.record_output:
            return
        call.stringify_output(
            output,
            self.record_tensor_attributes,
            self._tensor_memo if self.record_ids else None,
        )

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        call = _OpCall(
            func, args, kwargs, self.call_depth, stack=self.record_stack_trace
        )
        self._record_call(call)

        try:
            self.call_depth += 1
            result = func(*args, **kwargs)
            self._record_call_output(call, result)
            return result
        finally:
            self.call_depth -= 1

    def _maybe_record_function(self, tag):
        # filter out tags that appear noisy, or aren't runtime-related
        if any(
            tag.startswith(prefix)
            for prefix in [
                # assuming these are from benchmarking, not the actual runtime call
                "CachingAutotuner.",
                "InductorBenchmarker.",
                # inductor compilation
                "compile_fx.<locals>.",
            ]
        ):
            self.ignored_record_functions += 1
            return

        call = _AnnotateCall(
            tag, "record function", self.call_depth, stack=self.record_stack_trace
        )
        self.operators.append(call)
        self.call_depth += 1

    def _maybe_exit_record_function(self):
        if self.ignored_record_functions < 0:
            raise AssertionError(
                f"ignored_record_functions is negative: {self.ignored_record_functions}"
            )
        if self.ignored_record_functions > 0:
            self.ignored_record_functions -= 1
        else:
            self.call_depth -= 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Handle record_function entries
        if self.record_profiler_context:
            if func == torch.ops.profiler._record_function_enter_new.default:
                if len(args) != 1:
                    raise AssertionError(f"expected 1 arg, got {len(args)}")
                self._maybe_record_function(args[0])
            elif func == torch.ops.profiler._record_function_exit._RecordFunction:
                self._maybe_exit_record_function()

        # Handle DebugMode._annotate()
        if func is torch.ops.debug_mode_ops.annotate.default:
            if len(args) != 1:
                raise AssertionError(f"expected 1 arg, got {len(args)}")
            self._handle_annotate(args[0])
            return

        from torch.distributed._functional_collectives import AsyncCollectiveTensor
        from torch.distributed._local_tensor import LocalTensor

        # Record the operation with its call depth
        call = None
        if torch.distributed.tensor.DTensor in types:
            call = _OpCall(
                func, args, kwargs, self.call_depth, stack=self.record_stack_trace
            )
            self._record_call(call)
            return NotImplemented
        elif FakeTensor in types or isinstance(
            _get_current_dispatch_mode(), FakeTensorMode
        ):
            if self.record_faketensor:
                if func != torch.ops.prim.device.default:
                    call = _OpCall(
                        func,
                        args,
                        kwargs,
                        self.call_depth + 1,
                        stack=self.record_stack_trace,
                    )
                    self._record_call(call)
        # TODO: check the context manager
        elif LocalTensor in types:
            if self.record_localtensor:
                call = _OpCall(
                    func,
                    args,
                    kwargs,
                    self.call_depth + 1,
                    stack=self.record_stack_trace,
                )
                self._record_call(call)
        elif AsyncCollectiveTensor in types:
            # Record AsyncCollectiveTensor operations so debugging/tracing tools can see them
            if self.record_realtensor:
                call = _OpCall(
                    func,
                    args,
                    kwargs,
                    self.call_depth + 1,
                    stack=self.record_stack_trace,
                )
                self._record_call(call)
        elif len(types) == 0:
            if self.record_realtensor:
                call = _OpCall(
                    func,
                    args,
                    kwargs,
                    self.call_depth + 1,
                    stack=self.record_stack_trace,
                )
                self._record_call(call)

        # Run pre-hooks before executing the operation to hash inputs
        # We have to run before the func() call in case there's any
        # in-place mutation
        if call:
            _run_dispatch_pre_log_hooks(call, func, types, args, kwargs)

        result = func(*args, **kwargs)
        if call:
            self._record_call_output(call, result)
            _run_dispatch_hooks(call, func, types, args, kwargs, result)

        return result

    def __enter__(self):
        global _ACTIVE_DEBUG_MODE_COUNT
        _ACTIVE_DEBUG_MODE_COUNT += 1
        if self.record_torchfunction:
            torch._C._push_on_torch_function_stack(self)

        super().__enter__()
        if self.record_nn_module:
            self.module_tracker.__enter__()  # type: ignore[attribute, union-attr]

        if self.record_stack_trace:
            self.anomaly_for_traces = torch.autograd.set_detect_anomaly(
                True, check_nan=False
            )
            self.anomaly_for_traces.__enter__()
        return self

    # pyrefly: ignore [bad-override]
    def __exit__(self, *args):
        global _ACTIVE_DEBUG_MODE_COUNT
        _ACTIVE_DEBUG_MODE_COUNT -= 1
        super().__exit__(*args)
        if self.record_nn_module:
            self.module_tracker.__exit__()  # type: ignore[attribute, union-attr]
        if self.record_torchfunction:
            torch._C._pop_torch_function_stack()
        if self.record_stack_trace:
            self.anomaly_for_traces.__exit__(*args)

    @contextlib.contextmanager
    def set_fx_stack_trace(self, stack_trace):
        self.fx_stack_trace = stack_trace
        try:
            yield
        finally:
            self.fx_stack_trace = None

    def _enter_nn_module_call(self, fqn, header):
        call = _AnnotateCall(
            fqn, header, self.call_depth + 1, stack=self.record_stack_trace
        )
        self.operators.append(call)
        self.current_nn_module_stack.append(fqn)
        self.call_depth += 1

    def _exit_nn_module_call(self):
        self.call_depth -= 1
        self.current_nn_module_stack.pop()

    def module_tracker_setup(self) -> None:
        from torch.distributed._tools.mod_tracker import ModTracker

        self.module_tracker = ModTracker()

        # module pre-fw hook: record module call
        def pre_fw_hook(module, input) -> None:
            fqn = self.module_tracker._get_mod_name(module)  # type: ignore[attribute, union-attr]
            self._enter_nn_module_call(fqn, "nn.Mod")

        # module post-fw hook: decrement call depth
        def post_fw_hook(module, input, output) -> None:
            self._exit_nn_module_call()

        self.module_tracker.register_user_hooks(pre_fw_hook, post_fw_hook)

    def _handle_fx_nn_module_stack(
        self,
        base_stack: list[str],
        nn_module_stack: dict[str, tuple[str, Any]] | None,
        fwd_nn_module_stack: dict[str, tuple[str, Any]] | None,
    ) -> None:
        """
        Called when DebugInterpreter observes nn_module_stack or fwd_nn_module_stack metadata
        from executing the compiled GraphModule.

        If the current module stack is mismatched with what's currently tracked in DebugMode
        (current_nn_module_stack), we adjust call depth and add new [nn.Module] log entries accordingly.
        """

        nn_module_stack = nn_module_stack or {}
        fwd_nn_module_stack = fwd_nn_module_stack or {}
        if nn_module_stack and fwd_nn_module_stack:
            raise AssertionError(
                "Expecting at most one of nn_module_stack and fwd_nn_module_stack."
            )

        is_fwd = nn_module_stack
        stack = nn_module_stack if is_fwd else fwd_nn_module_stack

        # forward stack
        current_stack = self.current_nn_module_stack
        new_stack = base_stack + [v[0] for v in stack.values()]

        entered = set(new_stack) - set(current_stack)
        exited = set(current_stack) - set(new_stack)

        # Decrement depth for exited modules
        for _ in exited:
            self._exit_nn_module_call()
        if self.call_depth < 0:
            raise AssertionError("Unexpectedly, DebugMode call_depth is negative")

        # Add [nn.Module] entries for newly entered modules
        for fqn in sorted(entered):
            self._enter_nn_module_call(
                fqn, "nn.Mod (compile)" if is_fwd else "nn.Mod (compile bwd)"
            )

        self.current_nn_module_stack = new_stack

    @contextlib.contextmanager
    def record_redistribute_calls(
        self,
        arg,
        src_placement,
        dst_placement,
        transform_info_str: str | None = None,
        is_explicit: bool = False,
    ):
        try:
            self._record_call(
                _RedistributeCall(
                    arg,
                    src_placement=src_placement,
                    dst_placement=dst_placement,
                    transform_info_str=transform_info_str,
                    call_depth=self.call_depth + 1,
                    stack=self.record_stack_trace,
                    is_explicit=is_explicit,
                )
            )
            self.call_depth += 1
            yield
        finally:
            self.call_depth -= 1

    def record_output_placements(self, output_spec) -> None:
        """Record output placements for a DTensor op as a separate line."""
        if not self.record_output:
            return
        from torch.distributed.tensor._dtensor_spec import DTensorSpec

        placements_str = str(
            tree_map_only(DTensorSpec, _stringify_dtensor_spec, output_spec)
        )
        call = _OutputPlacementCall(placements_str, self.call_depth + 1)
        self._record_call(call)

    def record_triton_kernel(
        self, kernel_name: str, kwargs: dict[str, Any]
    ) -> _TritonKernelCall:
        call = _TritonKernelCall(kernel_name, kwargs, self.call_depth + 1)
        call.stringify_args(self.record_tensor_attributes)
        self.operators.append(call)
        return call

    def debug_string(self, show_stack_trace: bool | None = None) -> str:
        """
        show_stack_trace: option to display one-line stack trace summaries above groups
                        of operations (similar to gm.print_readable() style).
                        Requires record_stack_trace=True.
                        if None, uses self.record_stack_trace, otherwise overrides it.
        """
        show_stack_trace = (
            self.record_stack_trace if show_stack_trace is None else show_stack_trace
        )

        with torch._C.DisableTorchFunction():
            if not show_stack_trace:
                result = "\n".join(
                    "  "
                    + "  " * op.call_depth
                    + op.render(self.record_tensor_attributes)
                    for op in self.operators
                )
                return result

            # Group operations by stack trace
            lines = []
            prev_stack_summary = None

            for op in self.operators:
                # Get the stack trace: prefer fwd_stack_trace, fallback to stack_trace
                stack_trace = None
                if hasattr(op, "fwd_stack_trace") and op.fwd_stack_trace:
                    stack_trace = op.fwd_stack_trace
                elif hasattr(op, "stack_trace") and op.stack_trace:
                    stack_trace = op.stack_trace

                stack_summary = None
                if stack_trace:
                    stack_summary = _get_user_stack_trace(stack_trace)

                if stack_summary and stack_summary != prev_stack_summary:
                    # add blank line before stack trace comment for readability
                    if lines:  # don't add blank line at the very start
                        lines.append("")
                    indent = "  " * (op.call_depth + 1)
                    lines.append(indent + "# " + stack_summary)
                    prev_stack_summary = stack_summary

                # Add the operation line
                line = (
                    "  "
                    + "  " * op.call_depth
                    + op.render(self.record_tensor_attributes)
                )
                lines.append(line)

            return "\n".join(lines)

    @staticmethod
    @contextlib.contextmanager
    def dispatch_hooks(
        record_hook: Callable | None = None,
        log_hook: Callable | None = None,
        pre_log_hook: Callable | None = None,
    ):
        """
        Allows installing post-hooks on arguments to intercepted __torch_dispatch__ calls;
        hook signatures are expected as (func, types, args, kwargs, result),
        i.e. __torch_dispatch__ args + return value.

        Logging hook outputs are stored in call.log and annotate calls in debug_string(),
        while recording hook outputs are just stored in call.record.
        For now hooks are expected to return dictionaries.

        pre_log_hook signature is (func, types, args, kwargs, call) and is executed before
        the operation. It allows capturing state before in-place mutations.
        """
        if record_hook:
            _utils._DISPATCH_RECORD_HOOKS.append(record_hook)
        if log_hook:
            _utils._DISPATCH_LOG_HOOKS.append(log_hook)
        if pre_log_hook:
            _utils._DISPATCH_PRE_LOG_HOOKS.append(pre_log_hook)
        try:
            yield
        finally:
            if record_hook:
                _utils._DISPATCH_RECORD_HOOKS.pop()
            if log_hook:
                _utils._DISPATCH_LOG_HOOKS.pop()
            if pre_log_hook:
                _utils._DISPATCH_PRE_LOG_HOOKS.pop()

    @staticmethod
    @contextlib.contextmanager
    def record_outputs():
        """
        Hook for storing cloned output tensors in .record["output"].
        """

        def dispatch_hook(func, types, args, kwargs, result):
            out = tree_map(
                lambda x: x.clone() if isinstance(x, torch.Tensor) else x, result
            )
            return {"output": out}

        try:
            _old_record_triton = _utils._RECORD_TRITON_OUTPUTS
            _utils._RECORD_TRITON_OUTPUTS = True
            with DebugMode.dispatch_hooks(record_hook=dispatch_hook):
                yield
        finally:
            _utils._RECORD_TRITON_OUTPUTS = _old_record_triton

    @staticmethod
    @contextlib.contextmanager
    def log_tensor_hashes(
        hash_fn: Callable | str | list[str] = "norm", hash_inputs: bool = False
    ):
        """
        Installs hook for tensor hash logging.

        hash_fn: One of:
            - Custom-defined hash function
            - String: one of ("norm", "hash_tensor")
                - "norm": uses norm_hash_fn; basically tensor's L1 norm
                - "hash_tensor": uses torch.hash_tensor (XOR sum reduction)
            - List of strings: returns tuple of hashes from above options
        hash_inputs: if True, also hashes tensors in (args, kwargs), storing them in "input_hash".
        Input hashes are captured before the operation executes, so they reflect the state before
        any in-place mutations.
        """

        def hash_fn_option(hash_type):
            if not isinstance(hash_type, str) or hash_type not in [
                "norm",
                "hash_tensor",
            ]:
                raise AssertionError(
                    f"hash_type must be 'norm' or 'hash_tensor', got {hash_type!r}"
                )
            return functools.partial(
                norm_hash_fn if hash_type == "norm" else hash_tensor_fn, use_scalar=True
            )

        if callable(hash_fn):
            fn = hash_fn
        elif isinstance(hash_fn, str):
            fn = hash_fn_option(hash_fn)
        elif isinstance(hash_fn, list):
            fns = [hash_fn_option(fn) for fn in hash_fn]
            fn = lambda x: tuple(fn(x) for fn in fns)  # noqa: E731
        else:
            raise NotImplementedError(
                f"log_tensor_hashes() expected hash_fn to be callable, str, or list[str], but found {type(hash_fn)}"
            )

        def _tree_hash(obj):
            return tree_map(
                lambda x: fn(x) if isinstance(x, torch.Tensor) else None, obj
            )

        def _dispatch_pre_log_hook(func, types, args, kwargs, call):
            """Pre-hook to capture input hashes before operation executes"""
            if "empty" in str(func) or "profiler" in str(func):
                return None

            if hash_inputs:
                # Capture input hashes before the operation
                input_hash = _tree_hash((args, kwargs))
                if not tree_all(lambda x: x is None, input_hash):
                    return {"input_hash": input_hash}
            return None

        def _dispatch_post_hook(func, types, args, kwargs, result):
            """Post-hook to capture output hashes after operation executes"""
            if "empty" in str(func) or "profiler" in str(func):
                return None

            out = {}
            out["hash"] = _tree_hash(result)

            if tree_all(lambda x: x is None, out.values()):
                return None
            return out

        try:
            if hash_inputs:
                _old_input_hfn = _utils._TRITON_INPUT_HASH_FN
                _utils._TRITON_INPUT_HASH_FN = fn
            _old_output_hfn = _utils._TRITON_OUTPUT_HASH_FN
            _utils._TRITON_OUTPUT_HASH_FN = fn
            with DebugMode.dispatch_hooks(
                log_hook=_dispatch_post_hook,
                pre_log_hook=_dispatch_pre_log_hook if hash_inputs else None,
            ):
                yield
        finally:
            if hash_inputs:
                _utils._TRITON_INPUT_HASH_FN = _old_input_hfn  # type: ignore[assignment]
            _utils._TRITON_OUTPUT_HASH_FN = _old_output_hfn

    @staticmethod
    @contextlib.contextmanager
    def _benchmarking_inductor():
        """
        Context manager for disabling logging during inductor benchmarking,
        so logs don't contain all kernels launched from autotuning.
        """
        try:
            _utils._IN_INDUCTOR_BENCHMARK = True
            yield
        finally:
            _utils._IN_INDUCTOR_BENCHMARK = False

    @property
    def logs(self):
        return list(self.operators)

    @staticmethod
    def save_logs(logs: list, destination: str | os.PathLike | IO[str]) -> None:
        """
        Save DebugMode logs to a JSON file so they can be loaded and compared in
        another Python process.

        Args:
            logs: logs from a DebugMode run (from debug_mode.logs)
            destination: path or text file object to write
        """
        data = {
            "format": _SERIALIZED_LOG_FORMAT,
            "version": _SERIALIZED_LOG_VERSION,
            "logs": [_serialize_debug_call(call) for call in logs],
        }
        if isinstance(destination, (str, os.PathLike)):
            with open(destination, "w", encoding="utf-8") as f:
                json.dump(data, f, allow_nan=False)
                f.write("\n")
        else:
            json.dump(data, destination, allow_nan=False)
            destination.write("\n")

    @staticmethod
    def load_logs(source: str | os.PathLike | IO[str]) -> list:
        """
        Load logs saved by DebugMode.save_logs().

        Args:
            source: path or text file object to read

        Returns:
            A list of DebugMode log records suitable for check_hash_mismatches().
        """
        if isinstance(source, (str, os.PathLike)):
            with open(source, encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = json.load(source)

        if not isinstance(data, dict):
            raise ValueError(
                f"Serialized DebugMode logs must be a dict, but found {type(data).__name__}"
            )
        if data.get("format") != _SERIALIZED_LOG_FORMAT:
            raise ValueError(
                "Serialized DebugMode logs have an unrecognized format: "
                f"{data.get('format')!r}"
            )
        if data.get("version") != _SERIALIZED_LOG_VERSION:
            raise ValueError(
                "Serialized DebugMode logs have an unsupported version: "
                f"{data.get('version')!r}"
            )
        logs = data.get("logs")
        if not isinstance(logs, list):
            raise ValueError("Serialized DebugMode logs must contain a logs list")
        return [_deserialize_debug_call(call) for call in logs]

    def _handle_annotate(self, tag):
        """Handles DebugMode._annotate()"""
        call = _AnnotateCall(tag, "annotate", self.call_depth, self.record_stack_trace)
        self.operators.append(call)

    @staticmethod
    def _annotate(tag: Any) -> None:
        """
        If an active DebugMode exists, adds an "[annotate] <tag>" entry to the logs. Useful for contextualizing logs.
        Implemented with a custom op.
        """
        torch.ops.debug_mode_ops.annotate(tag)

    @staticmethod
    def check_hash_mismatches(
        logs1: list, logs2: list, compare_inputs: bool = False
    ) -> list[dict]:
        """
        Compares tensor hashes between two DebugMode runs, for checking run-to-run numerical divergence.

        This first validates the two log sequences have identical structure (same operations, input shapes/dtypes, etc.),
        then compares tensor hash values, and returns a list of call outputs where mismatches were found.
        Expects input logs to have been run with log_tensor_hashes, and looks for hashes in .log["hash"] & .log["input_hash"]
        (or .post_hashes & .pre_hashes for triton kernels).

        note: skips checking log pairs where hashes aren't present, but will raise if present in one & not the other.

        Args:
            logs1: logs from the first DebugMode run (from debug_mode.logs)
            logs2: logs from the second DebugMode run
            compare_inputs: If True, also compare input tensor hashes (default: only output checking)

        Returns:
            List of dictionaries describing hash mismatches. Each dict contains:
                - call_type: "torch op" or "triton kernel"
                - call: Operator/kernel name
                - arg_name: For triton kernels, the argument name; None for torch ops
                - pytree_path: For torch ops, the pytree path to the differing tensor; None for kernels
                - hash1: Hash value from the first run
                - hash2: Hash value from the second run
                - rel_diff: Relative difference between hash values
                - is_input_hash: True if this is an input hash, False for output hash

        Raises:
            ValueError: If logs have different lengths, call types, operator names, or call depths

        Usage::

            # Process 1: run the first model config and save its logs
            with DebugMode() as debug_mode, DebugMode.log_tensor_hashes():
                model_config_1(x)
            DebugMode.save_logs(debug_mode.logs, "run1.json")

            # Process 2: run the second model config and save its logs
            with DebugMode() as debug_mode, DebugMode.log_tensor_hashes():
                model_config_2(x)
            DebugMode.save_logs(debug_mode.logs, "run2.json")

            logs1 = DebugMode.load_logs("run1.json")
            logs2 = DebugMode.load_logs("run2.json")
            mismatches = DebugMode.check_hash_mismatches(logs1, logs2)
            for m in mismatches:
                print(f"{m['call']}: hash diff {m['rel_diff']:.2e}")
        """
        if len(logs1) != len(logs2):
            raise ValueError(f"Log lengths don't match: {len(logs1)} vs {len(logs2)}")

        difference_info = []
        for i, (log1, log2) in enumerate(zip(logs1, logs2)):
            # check call type
            call1_type = type(log1).__name__
            call2_type = type(log2).__name__
            if call1_type != call2_type:
                raise ValueError(
                    f"Call types don't match at index {i}: {call1_type} vs {call2_type}"
                )
            call_type = call1_type

            # check call name
            op1_name, op2_name = _get_call_name(log1), _get_call_name(log2)
            if op1_name != op2_name:
                raise ValueError(
                    f"Operators don't match at index {i}: {call_type}[{op1_name}] vs {call_type}[{op2_name}]"
                )
            op_name = op1_name

            # check call depth
            if log1.call_depth != log2.call_depth:
                raise ValueError(
                    f"Call depths for {call_type}[{op_name}] don't match at index {i}: {log1.call_depth} vs {log2.call_depth}"
                )

            # Redistribute: call args should be the same
            if isinstance(log1, _RedistributeCall):
                if log1.render([]) != log2.render([]):
                    raise ValueError(
                        f"Redistribute calls don't match at index {i}: {log1} vs {log2}"
                    )

            # Triton kernel: same arg names, arg types
            elif isinstance(log1, _TritonKernelCall):
                if log1.kwargs_str != log2.kwargs_str:
                    raise ValueError(
                        f"Triton kernel call args don't match for {log1.kernel_name} at index {i}:"
                        f"\n\nlog1: {log1.kwargs_str}\n\nlog2: {log2.kwargs_str}"
                    )

                def compare_triton_hashes(hashes1, hashes2, is_input):
                    if set(hashes1.keys()) != set(hashes2.keys()):  # type: ignore[union-attr]
                        raise AssertionError(
                            f"hash key mismatch: {set(hashes1.keys())} vs {set(hashes2.keys())}"
                        )
                    for key in hashes1:
                        if hashes1[key] != hashes2[key]:
                            difference_info.append(
                                {
                                    "call_type": "triton kernel",
                                    "call": op_name,
                                    "arg_name": key,
                                    "pytree_path": None,
                                    "hash1": hashes1[key],
                                    "hash2": hashes2[key],
                                    "rel_diff": _compute_rel_diff(
                                        hashes1[key], hashes2[key]
                                    ),
                                    "is_input_hash": is_input,
                                }
                            )

                # check output hashes
                has_post_1, has_post_2 = (
                    log1.post_hashes is not None,
                    log2.post_hashes is not None,
                )
                if has_post_1 != has_post_2:
                    raise ValueError(
                        f"Triton kernel post-hash presence inconsistent for {log1.kernel_name} "
                        f"at index {i}: log1 has post_hashes={has_post_1}, log2 has post_hashes={has_post_2}"
                    )

                if has_post_1:
                    compare_triton_hashes(
                        log1.post_hashes, log2.post_hashes, is_input=False
                    )

                # maybe check input hashes
                if compare_inputs:
                    has_pre_1, has_pre_2 = (
                        log1.pre_hashes is not None,
                        log2.pre_hashes is not None,
                    )
                    if has_pre_1 != has_pre_2:
                        raise ValueError(
                            f"Triton kernel pre-hash presence inconsistent for {log1.kernel_name} "
                            f"at index {i}: log1 has pre_hashes={has_pre_1}, log2 has pre_hashes={has_pre_2}"
                        )

                    if has_pre_1:
                        compare_triton_hashes(
                            log1.pre_hashes, log2.pre_hashes, is_input=True
                        )

            # regular log calls
            elif isinstance(log1, _OpCall):

                def compare_op_hashes(hashes1, hashes2, is_input):
                    def _helper(keypath, hash1, hash2):
                        if hash1 != hash2:
                            difference_info.append(
                                {
                                    "call_type": "torch op",
                                    "call": op_name,
                                    "arg_name": None,
                                    "pytree_path": keystr(keypath),
                                    "hash1": hash1,
                                    "hash2": hash2,
                                    "rel_diff": _compute_rel_diff(hash1, hash2),
                                    "is_input_hash": is_input,
                                }
                            )

                    tree_map_with_path(_helper, hashes1, hashes2)

                # check output hashes
                has_hash1 = log1.log is not None and "hash" in log1.log
                has_hash2 = log2.log is not None and "hash" in log2.log
                if has_hash1 != has_hash2:
                    raise ValueError(
                        f"Output hash presence inconsistent for triton kernel {call_type}[{op_name}] "
                        f"at index {i}: log1 has hash={has_hash1}, log2 has hash={has_hash2}"
                    )

                if has_hash1:
                    compare_op_hashes(
                        log1.log["hash"],  # type: ignore[union-attr]
                        log2.log["hash"],
                        is_input=False,
                    )

                # maybe check input hashes
                if compare_inputs:
                    has_hash1 = log1.log is not None and "input_hash" in log1.log
                    has_hash2 = log2.log is not None and "input_hash" in log2.log
                    if has_hash1 != has_hash2:
                        raise ValueError(
                            f"Input hash presence inconsistent for triton kernel {call_type}[{op_name}] "
                            f"at index {i}: log1 has input_hash={has_hash1}, log2 has input_hash={has_hash2}"
                        )

                    if has_hash1:
                        compare_op_hashes(
                            log1.log["input_hash"],  # type: ignore[union-attr]
                            log2.log["input_hash"],
                            is_input=True,
                        )

        return difference_info


def get_active_debug_mode() -> DebugMode | None:
    # Fast path: if no DebugMode is active, skip the stack walk
    if _ACTIVE_DEBUG_MODE_COUNT == 0:
        return None
    debug_mode = None
    for mode in _get_current_dispatch_mode_stack():
        if isinstance(mode, DebugMode):
            debug_mode = mode
            break
    return debug_mode
<|endoftext|># mypy: allow-untyped-defs
from contextlib import contextmanager
from typing import Any, cast
import random
import torch
import time
from torch.utils.benchmark import Timer

def extract_ir(filename: str) -> list[str]:
    BEGIN = "<GRAPH_EXPORT>"
    END = "</GRAPH_EXPORT>"
    pfx = None
    graphs = []
    with open(filename) as f:
        split_strs = f.read().split(BEGIN)
        for i, split_str in enumerate(split_strs):
            if i == 0:
                continue
            end_loc = split_str.find(END)
            if end_loc == -1:
                continue
            s = split_str[:end_loc]
            pfx = split_strs[i - 1].splitlines()[-1]
            lines = [x[len(pfx):] for x in s.splitlines(keepends=True)]
            graphs.append(''.join(lines))

    return graphs


def make_tensor_from_type(inp_type: torch._C.TensorType):
    size = inp_type.sizes()
    stride = inp_type.strides()
    device = inp_type.device()
    dtype = inp_type.dtype()
    if size is None:
        raise AssertionError("make_tensor_from_type: 'size' is None (inp_type.sizes() returned None)")
    if stride is None:
        raise AssertionError("make_tensor_from_type: 'stride' is None (inp_type.strides() returned None)")
    if device is None:
        raise AssertionError("make_tensor_from_type: 'device' is None (inp_type.device() returned None)")
    if dtype is None:
        raise AssertionError("make_tensor_from_type: 'dtype' is None (inp_type.dtype() returned None)")
    return torch.empty_strided(size=size, stride=stride, device=device, dtype=dtype)

def load_graph_and_inputs(ir: str) -> tuple[Any, list[Any]]:
    graph = torch._C.parse_ir(ir, parse_tensor_constants=True)
    graph.makeMultiOutputIntoTuple()
    inputs = []
    for inp in graph.inputs():
        if isinstance(inp.type(), torch._C.FloatType):
            inputs.append(random.uniform(.1, 100))
        elif isinstance(inp.type(), torch._C.IntType):
            inputs.append(random.randint(1, 100))
        elif isinstance(inp.type(), torch._C.TensorType):
            tensorType = cast(torch._C.TensorType, inp.type())
            inputs.append(make_tensor_from_type(tensorType))
        elif isinstance(inp.type(), torch._C.BoolType):
            inputs.append(random.randint(0, 1) == 1)
        else:
            raise NotImplementedError(f"A default value is not implemented for type {inp.type()}")

    func = torch._C._create_function_from_graph("forward", graph)
    torch._C._jit_pass_erase_shape_information(func.graph)
    return (func, inputs)

def time_cuda(fn, inputs, test_runs):
    t = Timer(stmt="fn(*inputs)", globals={"fn": fn, "inputs" : inputs})
    times = t.blocked_autorange()
    return times.median * 1000  # time in ms

def time_cpu(fn, inputs, test_runs):
    s = time.perf_counter()
    for _ in range(test_runs):
        fn(*inputs)
    e = time.perf_counter()
    return (e - s) / test_runs * 1000  # time in ms

def run_test(ir, inputs, *, warmup_runs=10, test_runs=20) -> float:
    graph, _ = load_graph_and_inputs(ir)
    for _ in range(warmup_runs):
        graph(*inputs)

    is_cpu = None
    for input in inputs:
        if isinstance(input, torch.Tensor):
            is_cpu = input.device.type == "cpu"
            break
    if is_cpu is None:
        raise AssertionError("No tensor found in inputs")

    out = time_cpu(graph, inputs, test_runs) if is_cpu else time_cuda(graph, inputs, test_runs)
    return out

@contextmanager
def no_fuser(*args, **kwargs):
    old_optimize = torch._C._get_graph_executor_optimize(False)
    try:
        yield
    finally:
        torch._C._get_graph_executor_optimize(old_optimize)

def run_baseline_no_fusion(ir, inputs) -> float:
    with no_fuser():
        return run_test(ir, inputs)


def run_nnc(ir, inputs, dynamic) -> float:
    try:
        strat = [("DYNAMIC", 10)] if dynamic else [("STATIC", 10)]
        old_strat = torch.jit.set_fusion_strategy(strat)
        with torch.jit.fuser("fuser1"):
            return run_test(ir, inputs)
    finally:
        torch.jit.set_fusion_strategy(old_strat)

def run_nvfuser(ir, inputs) -> float:
    with torch.jit.fuser("fuser2"):
        return run_test(ir, inputs)
<|endoftext|># mypy: disallow-untyped-defs

import functools
import logging
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from threading import Lock
from typing import Any, TypeVar
from typing_extensions import ParamSpec


logger = logging.getLogger("strobelight_function_profiler")

console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(name)s, line %(lineno)d, %(asctime)s, %(levelname)s: %(message)s"
)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

_P = ParamSpec("_P")
_R = TypeVar("_R")


class StrobelightCLIProfilerError(Exception):
    """
    Raised when an error happens during strobelight profiling
    """


def _pid_namespace_link(pid: int | None = None) -> str:
    """Returns the link to the process's namespace, example: pid:[4026531836]"""
    PID_NAMESPACE_PATH = "/proc/{}/ns/pid"
    pid = pid or os.getpid()
    return os.readlink(PID_NAMESPACE_PATH.format(pid))


def _pid_namespace(pid: int | None = None) -> int:
    """Returns the process's namespace id"""
    pid = pid or os.getpid()
    link = _pid_namespace_link(pid)
    return int(link[link.find("[") + 1 : -1])


def _command_to_string(command: Sequence[str]) -> str:
    return " ".join(command)


class StrobelightCLIFunctionProfiler:
    """
    Note: this is a meta only tool.

    StrobelightCLIFunctionProfiler can be used to profile a python function and
    generate a strobelight link with the results. It works on meta servers but
    does not requires an fbcode target.
    When stop_at_error is false(default), error during profiling does not prevent
    the work function from running.

    Check function_profiler_example.py for an example.
    """

    # This lock is used to make sure only one thread is running the profiler at any point.
    _lock = Lock()

    def __init__(
        self,
        *,
        stop_at_error: bool = False,
        max_profile_duration_sec: int = 60 * 10,
        sample_each: float = 1e7,  # sample each sample_each cycles.
        run_user_name: str = "pytorch-strobelight-ondemand",
        timeout_wait_for_running_sec: int = 60,
        timeout_wait_for_finished_sec: int = 60,
        recorded_env_variables: list[str] | None = None,
        sample_tags: list[str] | None = None,
        stack_max_len: int = 127,
        async_stack_max_len: int = 127,
    ) -> None:
        self.stop_at_error = stop_at_error
        self.max_profile_duration_sec = max_profile_duration_sec
        self.sample_each = sample_each
        self.run_user_name = run_user_name
        self.timeout_wait_for_running_sec = timeout_wait_for_running_sec
        self.timeout_wait_for_finished_sec = timeout_wait_for_finished_sec
        # Results of the most recent run.
        # Tracks the strobelight run id of the most recent run
        self.current_run_id: int | None = None
        self.sample_tags = sample_tags

    def _run_async(self) -> None:
        processId = os.getpid()
        namespace = _pid_namespace(processId)
        command = [
            "strobeclient",
            "run",
            "--profiler",
            "pyperf",
            "--event",
            "cycles",
            "--async",
            "--sample-interval",
            f"{int(self.sample_each)}",
            "--duration-ms",
            f"{int(self.max_profile_duration_sec * 1000)}",
            "--pid",
            f"{namespace}:{processId}",
        ]

        if self.sample_tags:
            command.append("--sample-tags")
            command.append(",".join(self.sample_tags))

        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to start strobelight profiling, error in run_async:{output}"
            )

        if match := re.search(r"INFO Run Id: (-?\d+)", output):
            self.current_run_id = int(match.group(1))
            return

        raise StrobelightCLIProfilerError(
            f"failed to start strobelight profiling, unexpected result {output}"
        )

    def _wait_for_running(self, counter: int = 0) -> None:
        if counter > 20:
            raise StrobelightCLIProfilerError(
                "wait_for_running called more than 20 times"
            )

        command = ["strobeclient", "getRunStatus", "--run-id", f"{self.current_run_id}"]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to start strobelight profiling, error in wait_for_running:{output}"
            )

        if match := re.search("Profile run status: (.*)", output):
            current_status = match.group(1)
            if current_status == "RUNNING":
                return
            elif current_status == "PREPARING":
                time.sleep(10)
                self._wait_for_running(counter + 1)
                return
            else:
                raise StrobelightCLIProfilerError(f"unexpected {current_status} phase")

        raise StrobelightCLIProfilerError(f"unexpected output\n: {output} ")

    def _stop_run(self) -> None:
        command = ["strobeclient", "stopRun", "--run-id", str(self.current_run_id)]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to stop strobelight profiling, return code is not 0 :{output}"
            )

        if match := re.search("INFO ::1:(.*)", output):
            current_status = match.group(1)
            if current_status.__contains__("Success!"):
                return
            else:
                raise StrobelightCLIProfilerError(
                    f"failed to stop strobelight profiling, got {current_status} result"
                )

        raise StrobelightCLIProfilerError(f"unexpected output\n: {output} ")

    def _get_results(self) -> None:
        command = ["strobeclient", "getRunStatus", "--run-id", str(self.current_run_id)]
        logger.debug("running command: %s", _command_to_string(command))
        result = subprocess.run(command, capture_output=True)
        output = result.stderr.decode("utf-8")
        logger.debug("output:\n{%s}", output)

        if result.returncode != 0:
            raise StrobelightCLIProfilerError(
                f"failed to extract profiling results, return code is not 0 : {output}"
            )

        if match := re.search("INFO ::1:(.*)", output):
            current_status = match.group(1)
            if current_status.__contains__("Profile run status: PROCESSING"):
                time.sleep(10)
                self._get_results()
                return
            elif not current_status.__contains__("Profile run finished with SUCCESS"):
                raise StrobelightCLIProfilerError(
                    f"failed to extract profiling results, unexpected response {output}"
                )

        for item in re.findall(
            r"(Total samples(.*)|GraphProfiler(.*)|Icicle view \(python stack\)(.*))",
            output,
        ):
            logger.info(item[0])

    def _stop_strobelight_no_throw(
        self,
        collect_results: bool,
    ) -> None:
        try:
            # call stop run
            self._stop_run()
            logger.info("strobelight profiling stopped")

            logger.debug("collection stopped")

            if not collect_results:
                return

            self._get_results()
        except Exception:
            logger.warning("error during stop_strobelight", exc_info=True)

    # Return true if strobelight started and is running. Never throw.
    def _start_strobelight(self) -> bool:
        strobelight_started = False
        try:
            self._run_async()
            strobelight_started = True
            logger.info("strobelight run id is: %s", self.current_run_id)
            self._wait_for_running()
            logger.info("strobelight profiling running")
            return True

        except Exception:
            logger.warning("error during start_strobelight:", exc_info=True)
            if strobelight_started:
                self._stop_strobelight_no_throw(collect_results=False)
            return False

    def profile(
        self, work_function: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
    ) -> _R | None:
        self.current_run_id = None

        if locked := StrobelightCLIFunctionProfiler._lock.acquire(False):
            if not locked:
                if self.stop_at_error:
                    raise StrobelightCLIProfilerError("concurrent runs not supported")

                logger.warning("concurrent runs not supported")
                return work_function(*args, **kwargs)

            started = self._start_strobelight()
            if not started:
                if self.stop_at_error:
                    StrobelightCLIFunctionProfiler._lock.release()
                    raise StrobelightCLIProfilerError(
                        "failed to start strobelight profiling"
                    )
                result = work_function(*args, **kwargs)
                StrobelightCLIFunctionProfiler._lock.release()
                return result

            try:
                logger.debug("collection started")
                result = work_function(*args, **kwargs)
                self._stop_strobelight_no_throw(collect_results=True)
                StrobelightCLIFunctionProfiler._lock.release()
                return result
            except Exception as error:
                logger.warning("work function throw exception", exc_info=True)
                self._stop_strobelight_no_throw(collect_results=False)
                StrobelightCLIFunctionProfiler._lock.release()
                raise error
        return None


# A function decorator that wraps profile, if no profiler is provided one with
# default args is created. A function can be annotated as:
# @strobelight()
# @strobelight(profiler = StrobelightFunctionProfiler(stop_at_error=True,..))
# @strobelight(stop_at_error=True,...)
def strobelight(
    profiler: StrobelightCLIFunctionProfiler | None = None, **kwargs: Any
) -> Callable[[Callable[_P, _R]], Callable[_P, _R | None]]:
    if not profiler:
        profiler = StrobelightCLIFunctionProfiler(**kwargs)

    def strobelight_inner(
        work_function: Callable[_P, _R],
    ) -> Callable[_P, _R | None]:
        @functools.wraps(work_function)
        def wrapper_function(*args: _P.args, **kwargs: _P.kwargs) -> _R | None:
            # pyrefly: ignore [bad-argument-type]
            return profiler.profile(work_function, *args, **kwargs)

        return wrapper_function

    return strobelight_inner
<|endoftext|>"""This module converts objects into numpy array."""

import numpy as np

import torch


def make_np(x: torch.Tensor) -> np.ndarray:
    """
    Convert an object into numpy array.

    Args:
      x: An instance of torch tensor

    Returns:
        numpy.array: Numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    if np.isscalar(x):
        return np.array([x])
    if isinstance(x, torch.Tensor):
        if x.device.type == "meta":
            return np.random.randn(1)
        return _prepare_pytorch(x)
    raise NotImplementedError(
        f"Got {type(x)}, but numpy array or torch tensor are expected."
    )


def _prepare_pytorch(x: torch.Tensor) -> np.ndarray:
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float16)
    # pyrefly: ignore [bad-assignment]
    x = x.detach().cpu().numpy()
    # pyrefly: ignore [bad-return]
    return x
<|endoftext|># mypy: allow-untyped-defs
from collections import OrderedDict
import contextlib
from typing import Any

from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.versions_pb2 import VersionDef

import torch
from ._proto_graph import node_proto

methods_OP = [
    "attributeNames",
    "hasMultipleOutputs",
    "hasUses",
    "inputs",
    "kind",
    "outputs",
    "outputsSize",
    "scopeName",
]
# Some additional methods to explore for methods_IO are
#
#   'unique' (type int)
#   'type' (type <Tensor<class 'torch._C.Type'>>)
#
# But the below are sufficient for now.
methods_IO = ["node", "offset", "debugName"]

GETATTR_KIND = "prim::GetAttr"
CLASSTYPE_KIND = "ClassType"


class NodeBase:
    def __init__(
        self,
        debugName=None,
        inputs=None,
        scope=None,
        tensor_size=None,
        op_type="UnSpecified",
        attributes="",
    ) -> None:
        # TODO; Specify a __slots__ for this class or potentially
        # used namedtuple instead
        self.debugName = debugName
        self.inputs = inputs
        self.tensor_size = tensor_size
        self.kind = op_type
        self.attributes = attributes
        self.scope = scope

    def __repr__(self) -> str:
        repr = []
        repr.append(str(type(self)))
        repr.extend(
            m + ": " + str(getattr(self, m)) + str(type(getattr(self, m)))
            for m in dir(self)
            if "__" not in m
        )
        return "\n".join(repr) + "\n\n"


class NodePy(NodeBase):
    def __init__(self, node_cpp, valid_methods) -> None:
        super().__init__(node_cpp)
        valid_methods = valid_methods[:]
        self.inputs = []

        for m in valid_methods:
            if m == "inputs" or m == "outputs":
                list_of_node = list(getattr(node_cpp, m)())
                io_unique_names = []
                io_tensor_sizes = []
                for n in list_of_node:
                    io_unique_names.append(n.debugName())
                    if n.isCompleteTensor():
                        io_tensor_sizes.append(n.type().sizes())
                    else:
                        io_tensor_sizes.append(None)

                setattr(self, m, io_unique_names)
                setattr(self, m + "tensor_size", io_tensor_sizes)

            else:
                setattr(self, m, getattr(node_cpp, m)())


class NodePyIO(NodePy):
    def __init__(self, node_cpp, input_or_output=None) -> None:
        super().__init__(node_cpp, methods_IO)
        try:
            tensor_size = node_cpp.type().sizes()
        except RuntimeError:
            tensor_size = [
                1,
            ]  # fail when constant model is used.
        self.tensor_size = tensor_size
        # Kind attribute string is purely descriptive and will be shown
        # in detailed information for the node in TensorBoard's graph plugin.
        #
        # NodePyOP nodes get this from their kind() method.
        self.kind = "Parameter"
        if input_or_output:
            self.input_or_output = input_or_output
            self.kind = "IO Node"


class NodePyOP(NodePy):
    def __init__(self, node_cpp) -> None:
        super().__init__(node_cpp, methods_OP)
        # Replace single quote which causes strange behavior in TensorBoard
        # TODO: See if we can remove this in the future
        self.attributes = str(
            {k: _node_get(node_cpp, k) for k in node_cpp.attributeNames()}
        ).replace("'", " ")
        self.kind = node_cpp.kind()


class GraphPy:
    """Helper class to convert torch.nn.Module to GraphDef proto and visualization with TensorBoard.

    GraphDef generation operates in two passes:

    In the first pass, all nodes are read and saved to two lists.
    One list is for input/output nodes (nodes_io), which only have inbound
    or outbound connections, but not both. Another list is for internal
    operator nodes (nodes_op). The first pass also saves all scope name
    appeared in the nodes in scope_name_appeared list for later processing.

    In the second pass, scope names are fully applied to all nodes.
    debugNameToScopedName is a mapping from a node's ID to its fully qualified
    scope name. e.g. Net1/Linear[0]/1. Unfortunately torch.jit doesn't have
    totally correct scope output, so this is nontrivial. The function
    populate_namespace_from_OP_to_IO and find_common_root are used to
    assign scope name to a node based on the connection between nodes
    in a heuristic kind of way. Bookkeeping is done with shallowest_scope_name
    and scope_name_appeared.
    """

    def __init__(self) -> None:
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = "default"
        self.scope_name_appeared = []

    def append(self, x) -> None:
        if isinstance(x, NodePyIO):
            self.nodes_io[x.debugName] = x
        if isinstance(x, NodePyOP):
            self.nodes_op.append(x)

    def printall(self) -> None:
        print("all nodes")
        for node in self.nodes_op:
            print(node)
        for key in self.nodes_io:
            print(self.nodes_io[key])

    def find_common_root(self) -> None:
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split("/")[0]

    def populate_namespace_from_OP_to_IO(self) -> None:
        for node in self.nodes_op:
            for node_output, outputSize in zip(node.outputs, node.outputstensor_size, strict=True):
                self.scope_name_appeared.append(node.scopeName)
                self.nodes_io[node_output] = NodeBase(
                    node_output,
                    node.inputs,
                    node.scopeName,
                    outputSize,
                    op_type=node.kind,
                    attributes=node.attributes,
                )

        self.find_common_root()

        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.unique_name_to_scoped_name[input_node_id] = (
                    node.scopeName + "/" + input_node_id
                )

        for key, node in self.nodes_io.items():
            if type(node) is NodeBase:
                # pyrefly: ignore [unsupported-operation]
                self.unique_name_to_scoped_name[key] = node.scope + "/" + node.debugName
            if hasattr(node, "input_or_output"):
                self.unique_name_to_scoped_name[key] = (
                    node.input_or_output + "/" + node.debugName
                )

            if hasattr(node, "scope") and node.scope is not None:
                self.unique_name_to_scoped_name[key] = node.scope + "/" + node.debugName
                if node.scope == "" and self.shallowest_scope_name:
                    self.unique_name_to_scoped_name[node.debugName] = (

                        self.shallowest_scope_name + "/" + node.debugName
                    )

        # replace name
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = [
                self.unique_name_to_scoped_name[node_input_id]
                for node_input_id in node.inputs
            ]
            if node.debugName in self.unique_name_to_scoped_name:
                self.nodes_io[key].debugName = self.unique_name_to_scoped_name[
                    node.debugName
                ]

    def to_proto(self):
        """Convert graph representation of GraphPy object to TensorBoard required format."""
        # TODO: compute correct memory usage and CPU time once
        # PyTorch supports it
        nodes = [
            node_proto(
                v.debugName,
                input=v.inputs,
                outputsize=v.tensor_size,
                op=v.kind,
                attributes=v.attributes,
            )
            for v in self.nodes_io.values()
        ]
        return nodes


def parse(graph, trace, args=None, omit_useless_nodes=True):
    """Parse an optimized PyTorch model graph and produces a list of nodes and node stats.

    Useful for eventual conversion to TensorBoard protobuf format.

    Args:
      graph (PyTorch module): The model graph to be parsed.
      trace (PyTorch JIT TracedModule): The model trace to be parsed.
      args (tuple): input tensor[s] for the model.
      omit_useless_nodes (boolean): Whether to remove nodes from the graph.
    """
    nodes_py = GraphPy()
    for node in graph.inputs():
        if omit_useless_nodes:
            if (
                len(node.uses()) == 0
            ):  # number of user of the node (= number of outputs/ fanout)
                continue

        if node.type().kind() != CLASSTYPE_KIND:
            nodes_py.append(NodePyIO(node, "input"))

    attr_to_scope: dict[Any, str] = {}
    for node in graph.nodes():
        if node.kind() == GETATTR_KIND:
            attr_name = node.s("name")
            attr_key = node.output().debugName()
            parent = node.input().node()
            if (
                parent.kind() == GETATTR_KIND
            ):  # If the parent node is not the top-level "self" node
                parent_attr_key = parent.output().debugName()
                parent_scope = attr_to_scope[parent_attr_key]
                attr_scope = parent_scope.split("/")[-1]
                attr_to_scope[attr_key] = f"{parent_scope}/{attr_scope}.{attr_name}"
            else:
                attr_to_scope[attr_key] = f"__module.{attr_name}"
            # We don't need classtype nodes; scope will provide this information
            if node.output().type().kind() != CLASSTYPE_KIND:
                node_py = NodePyOP(node)
                node_py.scopeName = attr_to_scope[attr_key]  # type: ignore[attr-defined]
                nodes_py.append(node_py)
        else:
            nodes_py.append(NodePyOP(node))

    for i, node in enumerate(graph.outputs()):  # Create sink nodes for output ops
        node_pyio = NodePyIO(node, "output")
        node_pyio.debugName = f"output.{i + 1}"
        node_pyio.inputs = [node.debugName()]
        nodes_py.append(node_pyio)

    def parse_traced_name(module):
        if isinstance(module, torch.jit.TracedModule):
            module_name = module._name
        else:
            module_name = getattr(module, "original_name", "Module")
        return module_name

    alias_to_name = {}
    base_name = parse_traced_name(trace)
    for name, module in trace.named_modules(prefix="__module"):
        mod_name = parse_traced_name(module)
        attr_name = name.split(".")[-1]
        alias_to_name[name] = f"{mod_name}[{attr_name}]"

    for node in nodes_py.nodes_op:
        module_aliases = node.scopeName.split("/")
        replacements = [
            alias_to_name[alias] if alias in alias_to_name else alias.split(".")[-1]
            for alias in module_aliases
        ]
        node.scopeName = base_name
        if any(replacements):
            node.scopeName += "/" + "/".join(replacements)

    nodes_py.populate_namespace_from_OP_to_IO()
    return nodes_py.to_proto()


def graph(model, args, verbose=False, use_strict_trace=True):
    """
    Process a PyTorch model and produces a `GraphDef` proto that can be logged to TensorBoard.

    Args:
      model (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      verbose (bool): Whether to print out verbose information while
        processing.
      use_strict_trace (bool): Whether to pass keyword argument `strict` to
        `torch.jit.trace`. Pass False when you want the tracer to
        record your mutable container types (list, dict)
    """
    with _set_model_to_eval(model):
        try:
            trace = torch.jit.trace(model, args, strict=use_strict_trace)
            graph = trace.graph
            torch._C._jit_pass_inline(graph)
        except RuntimeError as e:
            print(e)
            print("Error occurs, No graph saved")
            raise e

    if verbose:
        print(graph)
    list_of_nodes = parse(graph, trace, args)
    # We are hardcoding that this was run on CPU even though it might have actually
    # run on GPU. Note this is what is shown in TensorBoard and has no bearing
    # on actual execution.
    # TODO: See if we can extract GPU vs CPU information from the PyTorch model
    # and pass it correctly to TensorBoard.
    #
    # Definition of StepStats and DeviceStepStats can be found at
    # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/plugins/graph/tf_graph_common/proto.ts
    # and
    # https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/step_stats.proto
    stepstats = RunMetadata(
        step_stats=StepStats(dev_stats=[DeviceStepStats(device="/device:CPU:0")])
    )
    return GraphDef(node=list_of_nodes, versions=VersionDef(producer=22)), stepstats
    # The producer version has been reverse engineered from standard
    # TensorBoard logged data.


@contextlib.contextmanager
def _set_model_to_eval(model):
    """Context manager to temporarily set the training mode of ``model`` to eval."""
    if not isinstance(model, torch.jit.ScriptFunction):
        originally_training = model.training
        model.train(False)
        try:
            yield
        finally:
            model.train(originally_training)
    else:
        # Do nothing for ScriptFunction
        try:
            yield
        finally:
            pass


def _node_get(node: torch._C.Node, key: str):
    """Get attributes of a node which is polymorphic over return type."""
    sel = node.kindOf(key)
    return getattr(node, sel)(key)
<|endoftext|># mypy: allow-untyped-defs
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.versions_pb2 import VersionDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto


def load_onnx_graph(fname):
    import onnx
    m = onnx.load(fname)  # type: ignore[attr-defined]
    g = m.graph
    return parse(g)


def parse(graph):
    nodes = []
    import itertools

    nodes_proto = list(itertools.chain(graph.input, graph.output))

    for node in nodes_proto:
        print(node.name)
        shapeproto = TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=d.dim_value)
                for d in node.type.tensor_type.shape.dim
            ]
        )
        nodes.append(
            NodeDef(
                name=node.name.encode(encoding="utf_8"),
                op="Variable",
                input=[],
                attr={
                    "dtype": AttrValue(type=node.type.tensor_type.elem_type),
                    "shape": AttrValue(shape=shapeproto),
                },
            )
        )

    for node in graph.node:
        _attr = [" = ".join([str(f[1]) for f in s.ListFields()]) for s in node.attribute]
        attr = ", ".join(_attr).encode(encoding="utf_8")
        print(node.output[0])
        nodes.append(
            NodeDef(
                name=node.output[0].encode(encoding="utf_8"),
                op=node.op_type,
                input=node.input,
                attr={"parameters": AttrValue(s=attr)},
            )
        )

    # two pass token replacement, appends opname to object id
    mapping = {}
    for node in nodes:
        mapping[node.name] = node.op + "_" + node.name

    return GraphDef(node=nodes, versions=VersionDef(producer=22))
<|endoftext|># mypy: allow-untyped-defs
import math
import numpy as np
from ._convert_np import make_np
from ._utils import make_grid
from tensorboard.compat import tf
from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo


_HAS_GFILE_JOIN = hasattr(tf.io.gfile, "join")


def _gfile_join(a, b):
    # The join API is different between tensorboard's TF stub and TF:
    # https://github.com/tensorflow/tensorboard/issues/6080
    # We need to try both because `tf` may point to either the stub or the real TF.
    if _HAS_GFILE_JOIN:
        return tf.io.gfile.join(a, b)
    else:
        fs = tf.io.gfile.get_filesystem(a)
        return fs.join(a, b)


def make_tsv(metadata, save_path, metadata_header=None) -> None:
    if not metadata_header:
        metadata = [str(x) for x in metadata]
    else:
        if len(metadata_header) != len(
            metadata[0]
        ):
            raise AssertionError("len of header must be equal to the number of columns in metadata")
        metadata = ["\t".join(str(e) for e in l) for l in [metadata_header] + metadata]

    metadata_bytes = tf.compat.as_bytes("\n".join(metadata) + "\n")
    with tf.io.gfile.GFile(_gfile_join(save_path, "metadata.tsv"), "wb") as f:
        f.write(metadata_bytes)


# https://github.com/tensorflow/tensorboard/issues/44 image label will be squared
def make_sprite(label_img, save_path) -> None:
    from PIL import Image
    from io import BytesIO

    # this ensures the sprite image has correct dimension as described in
    # https://www.tensorflow.org/get_started/embedding_viz
    nrow = math.ceil((label_img.size(0)) ** 0.5)
    arranged_img_CHW = make_grid(make_np(label_img), ncols=nrow)

    # augment images so that #images equals nrow*nrow
    arranged_augment_square_HWC = np.zeros(
        (arranged_img_CHW.shape[2], arranged_img_CHW.shape[2], 3)
    )
    arranged_img_HWC = arranged_img_CHW.transpose(1, 2, 0)  # chw -> hwc
    arranged_augment_square_HWC[: arranged_img_HWC.shape[0], :, :] = arranged_img_HWC
    im = Image.fromarray(np.uint8((arranged_augment_square_HWC * 255).clip(0, 255)))

    with BytesIO() as buf:
        im.save(buf, format="PNG")
        im_bytes = buf.getvalue()

    with tf.io.gfile.GFile(_gfile_join(save_path, "sprite.png"), "wb") as f:
        f.write(im_bytes)


def get_embedding_info(metadata, label_img, subdir, global_step, tag):
    info = EmbeddingInfo()
    info.tensor_name = f"{tag}:{str(global_step).zfill(5)}"
    info.tensor_path = _gfile_join(subdir, "tensors.tsv")
    if metadata is not None:
        info.metadata_path = _gfile_join(subdir, "metadata.tsv")
    if label_img is not None:
        info.sprite.image_path = _gfile_join(subdir, "sprite.png")
        info.sprite.single_image_dim.extend([label_img.size(3), label_img.size(2)])
    return info


def write_pbtxt(save_path, contents) -> None:
    config_path = _gfile_join(save_path, "projector_config.pbtxt")
    with tf.io.gfile.GFile(config_path, "wb") as f:
        f.write(tf.compat.as_bytes(contents))


def make_mat(matlist, save_path) -> None:
    with tf.io.gfile.GFile(_gfile_join(save_path, "tensors.tsv"), "wb") as f:
        for x in matlist:
            x = [str(i.item()) for i in x]
            f.write(tf.compat.as_bytes("\t".join(x) + "\n"))
<|endoftext|>import tensorboard
from torch._vendor.packaging.version import Version

if not hasattr(tensorboard, "__version__") or Version(
    tensorboard.__version__
) < Version("1.15"):
    raise ImportError("TensorBoard logging requires TensorBoard version 1.15 or above")

del Version
del tensorboard

from .writer import FileWriter, SummaryWriter
from tensorboard.summary.writer.record_writer import RecordWriter

__all__ = [
    "FileWriter",
    "RecordWriter",
    "SummaryWriter",
]
<|endoftext|># mypy: allow-untyped-defs
import json
import logging
import struct

from typing import Any

import torch
import numpy as np


from google.protobuf import struct_pb2

from tensorboard.compat.proto.summary_pb2 import (
    HistogramProto,
    Summary,
    SummaryMetadata,
)
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.plugins.custom_scalar import layout_pb2
from tensorboard.plugins.pr_curve.plugin_data_pb2 import PrCurvePluginData
from tensorboard.plugins.text.plugin_data_pb2 import TextPluginData

from ._convert_np import make_np
from ._utils import _prepare_video, convert_to_HWC

__all__ = [
    "half_to_int",
    "int_to_half",
    "hparams",
    "scalar",
    "histogram_raw",
    "histogram",
    "make_histogram",
    "image",
    "image_boxes",
    "draw_boxes",
    "make_image",
    "video",
    "make_video",
    "audio",
    "custom_scalars",
    "text",
    "tensor_proto",
    "pr_curve_raw",
    "pr_curve",
    "compute_curve",
    "mesh",
]

logger = logging.getLogger(__name__)

def half_to_int(f: float) -> int:
    """Casts a half-precision float value into an integer.

    Converts a half precision floating point value, such as `torch.half` or
    `torch.bfloat16`, into an integer value which can be written into the
    half_val field of a TensorProto for storage.

    To undo the effects of this conversion, use int_to_half().

    """
    buf = struct.pack("f", f)
    return struct.unpack("i", buf)[0]

def int_to_half(i: int) -> float:
    """Casts an integer value to a half-precision float.

    Converts an integer value obtained from half_to_int back into a floating
    point value.

    """
    buf = struct.pack("i", i)
    return struct.unpack("f", buf)[0]

def _tensor_to_half_val(t: torch.Tensor) -> list[int]:
    return [half_to_int(x) for x in t.flatten().tolist()]

def _tensor_to_complex_val(t: torch.Tensor) -> list[float]:
    return torch.view_as_real(t).flatten().tolist()

def _tensor_to_list(t: torch.Tensor) -> list[Any]:
    return t.flatten().tolist()

# type maps: torch.Tensor type -> (protobuf type, protobuf val field)
_TENSOR_TYPE_MAP = {
    torch.half: ("DT_HALF", "half_val", _tensor_to_half_val),
    torch.float16: ("DT_HALF", "half_val", _tensor_to_half_val),
    torch.bfloat16: ("DT_BFLOAT16", "half_val", _tensor_to_half_val),
    torch.float32: ("DT_FLOAT", "float_val", _tensor_to_list),
    torch.float: ("DT_FLOAT", "float_val", _tensor_to_list),
    torch.float64: ("DT_DOUBLE", "double_val", _tensor_to_list),
    torch.double: ("DT_DOUBLE", "double_val", _tensor_to_list),
    torch.int8: ("DT_INT8", "int_val", _tensor_to_list),
    torch.uint8: ("DT_UINT8", "int_val", _tensor_to_list),
    torch.qint8: ("DT_UINT8", "int_val", _tensor_to_list),
    torch.int16: ("DT_INT16", "int_val", _tensor_to_list),
    torch.short: ("DT_INT16", "int_val", _tensor_to_list),
    torch.int: ("DT_INT32", "int_val", _tensor_to_list),
    torch.int32: ("DT_INT32", "int_val", _tensor_to_list),
    torch.qint32: ("DT_INT32", "int_val", _tensor_to_list),
    torch.int64: ("DT_INT64", "int64_val", _tensor_to_list),
    torch.complex32: ("DT_COMPLEX32", "scomplex_val", _tensor_to_complex_val),
    torch.chalf: ("DT_COMPLEX32", "scomplex_val", _tensor_to_complex_val),
    torch.complex64: ("DT_COMPLEX64", "scomplex_val", _tensor_to_complex_val),
    torch.cfloat: ("DT_COMPLEX64", "scomplex_val", _tensor_to_complex_val),
    torch.bool: ("DT_BOOL", "bool_val", _tensor_to_list),
    torch.complex128: ("DT_COMPLEX128", "dcomplex_val", _tensor_to_complex_val),
    torch.cdouble: ("DT_COMPLEX128", "dcomplex_val", _tensor_to_complex_val),
    torch.uint8: ("DT_UINT8", "uint32_val", _tensor_to_list),
    torch.quint8: ("DT_UINT8", "uint32_val", _tensor_to_list),
    torch.quint4x2: ("DT_UINT8", "uint32_val", _tensor_to_list),
}


def _calc_scale_factor(tensor) -> int:
    converted = tensor.numpy() if not isinstance(tensor, np.ndarray) else tensor
    return 1 if converted.dtype == np.uint8 else 255


def _draw_single_box(
    image,
    xmin,
    ymin,
    xmax,
    ymax,
    display_str,
    color="black",
    color_text="black",
    thickness=2,
):
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color,
    )
    if display_str:
        text_bottom = bottom
        # Reverse list and print from bottom to top.
        _left, _top, _right, _bottom = font.getbbox(display_str)
        text_width, text_height = _right - _left, _bottom - _top
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [
                (left, text_bottom - text_height - 2 * margin),
                (left + text_width, text_bottom),
            ],
            fill=color,
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill=color_text,
            font=font,
        )
    return image


def hparams(hparam_dict=None, metric_dict=None, hparam_domain_discrete=None):
    """Output three `Summary` protocol buffers needed by hparams plugin.

    `Experiment` keeps the metadata of an experiment, such as the name of the
      hyperparameters and the name of the metrics.
    `SessionStartInfo` keeps key-value pairs of the hyperparameters
    `SessionEndInfo` describes status of the experiment e.g. STATUS_SUCCESS

    Args:
      hparam_dict: A dictionary that contains names of the hyperparameters
        and their values.
      metric_dict: A dictionary that contains names of the metrics
        and their values.
      hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
        contains names of the hyperparameters and all discrete values they can hold

    Returns:
      The `Summary` protobufs for Experiment, SessionStartInfo and
        SessionEndInfo
    """
    import torch
    from tensorboard.plugins.hparams.api_pb2 import (
        DataType,
        Experiment,
        HParamInfo,
        MetricInfo,
        MetricName,
        Status,
    )
    from tensorboard.plugins.hparams.metadata import (
        EXPERIMENT_TAG,
        PLUGIN_DATA_VERSION,
        PLUGIN_NAME,
        SESSION_END_INFO_TAG,
        SESSION_START_INFO_TAG,
    )
    from tensorboard.plugins.hparams.plugin_data_pb2 import (
        HParamsPluginData,
        SessionEndInfo,
        SessionStartInfo,
    )

    # TODO: expose other parameters in the future.
    # hp = HParamInfo(name='lr',display_name='learning rate',
    # type=DataType.DATA_TYPE_FLOAT64, domain_interval=Interval(min_value=10,
    # max_value=100))
    # mt = MetricInfo(name=MetricName(tag='accuracy'), display_name='accuracy',
    # description='', dataset_type=DatasetType.DATASET_VALIDATION)
    # exp = Experiment(name='123', description='456', time_created_secs=100.0,
    # hparam_infos=[hp], metric_infos=[mt], user='tw')

    if not isinstance(hparam_dict, dict):
        logger.warning("parameter: hparam_dict should be a dictionary, nothing logged.")
        raise TypeError(
            "parameter: hparam_dict should be a dictionary, nothing logged."
        )
    if not isinstance(metric_dict, dict):
        logger.warning("parameter: metric_dict should be a dictionary, nothing logged.")
        raise TypeError(
            "parameter: metric_dict should be a dictionary, nothing logged."
        )

    hparam_domain_discrete = hparam_domain_discrete or {}
    if not isinstance(hparam_domain_discrete, dict):
        raise TypeError(
            "parameter: hparam_domain_discrete should be a dictionary, nothing logged."
        )
    for k, v in hparam_domain_discrete.items():
        if (
            k not in hparam_dict
            or not isinstance(v, list)
            or not all(isinstance(d, type(hparam_dict[k])) for d in v)
        ):
            raise TypeError(
                f"parameter: hparam_domain_discrete[{k}] should be a list of same type as hparam_dict[{k}]."
            )
    hps = []

    ssi = SessionStartInfo()
    for k, v in hparam_dict.items():
        if v is None:
            continue
        if isinstance(v, (int, float)):
            ssi.hparams[k].number_value = v

            if k in hparam_domain_discrete:
                domain_discrete: struct_pb2.ListValue | None = struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(number_value=d)
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            hps.append(
                HParamInfo(
                    name=k,
                    type=DataType.Value("DATA_TYPE_FLOAT64"),
                    domain_discrete=domain_discrete,
                )
            )
            continue

        if isinstance(v, str):
            ssi.hparams[k].string_value = v

            if k in hparam_domain_discrete:
                domain_discrete = struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(string_value=d)
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            hps.append(
                HParamInfo(
                    name=k,
                    type=DataType.Value("DATA_TYPE_STRING"),
                    domain_discrete=domain_discrete,
                )
            )
            continue

        if isinstance(v, bool):
            ssi.hparams[k].bool_value = v

            if k in hparam_domain_discrete:
                domain_discrete = struct_pb2.ListValue(
                    values=[
                        struct_pb2.Value(bool_value=d)
                        for d in hparam_domain_discrete[k]
                    ]
                )
            else:
                domain_discrete = None

            hps.append(
                HParamInfo(
                    name=k,
                    type=DataType.Value("DATA_TYPE_BOOL"),
                    domain_discrete=domain_discrete,
                )
            )
            continue

        if isinstance(v, torch.Tensor):
            v = make_np(v)[0]
            ssi.hparams[k].number_value = v
            hps.append(HParamInfo(name=k, type=DataType.Value("DATA_TYPE_FLOAT64")))
            continue
        raise ValueError(
            "value should be one of int, float, str, bool, or torch.Tensor"
        )

    content = HParamsPluginData(session_start_info=ssi, version=PLUGIN_DATA_VERSION)
    smd = SummaryMetadata(
        plugin_data=SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=content.SerializeToString()
        )
    )
