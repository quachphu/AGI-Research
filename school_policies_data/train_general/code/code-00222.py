        """
        Formats a single torch._C._profiler.CapturedTraceback into a list of
        strings equivalent to the output of traceback.format_list.  Note that if
        pass it CapturedTraceback with C++ traces,  it is better not to use this
        function and use the batch formatting API format_captured_tbs to amortize
        the cost of symbolization
        """
        return traceback.format_list(self.summary())

    @staticmethod
    def format_all(tbs):
        """
        Bulk version of CapturedTraceback.format.  Returns a list of list of strings.
        """
        import torch._C._profiler

        # Directly populate tracebacks that already have cached summaries
        rs: list[list[str] | None] = []
        delayed_idxs = []
        for i, tb in enumerate(tbs):
            if tb.tb is None:
                rs.append([])
            else:
                rs.append(None)
                delayed_idxs.append(i)

        torch._C._profiler.symbolize_tracebacks([tbs[i].tb for i in delayed_idxs])
        for i in delayed_idxs:
            rs[i] = traceback.format_list(tbs[i].summary())

        return rs


def _extract_symbolized_tb(tb, skip):
    """
    Given a symbolized traceback from symbolize_tracebacks, return a StackSummary object of
    pre-processed stack trace entries.
    """
    stack = traceback.StackSummary()
    for f in reversed(tb[skip:]):
        stack.append(traceback.FrameSummary(f["filename"], f["line"], f["name"]))
    return stack
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import collections.abc as _collections_abc
import weakref
from collections.abc import Mapping, MutableMapping
from weakref import ref

from torch import Tensor


WeakRef = ref


__all__ = [
    "TensorWeakRef",
    "WeakIdRef",
    "WeakIdKeyDictionary",
    "WeakTensorKeyDictionary",
]


# TODO: make weakref properly thread safe following
# https://github.com/python/cpython/pull/125325
class _IterationGuard:
    # This context manager registers itself in the current iterators of the
    # weak container, such as to delay all removals until the context manager
    # exits.
    # This technique should be relatively thread-safe (since sets are).

    def __init__(self, weakcontainer) -> None:
        # Don't create cycles
        self.weakcontainer = ref(weakcontainer)

    def __enter__(self):
        w = self.weakcontainer()
        if w is not None:
            w._iterating.add(self)
        return self

    def __exit__(self, e, t, b):
        w = self.weakcontainer()
        if w is not None:
            s = w._iterating
            s.remove(self)
            if not s:
                w._commit_removals()


# This file defines a variant of WeakKeyDictionary that overrides the hashing
# behavior of the key to use object identity, rather than the builtin
# __eq__/__hash__ functions.  This is useful for Tensor weak keys, as their
# __eq__ implementation return a Tensor (elementwise equality), which means
# you can't use them directly with the WeakKeyDictionary in standard library.
#
# Our implementation strategy is to create a wrapper weak key object, which we
# use as a key in a stock Python dictionary.  This is similar to how weakref
# implements WeakKeyDictionary, but instead of using weakref.ref as the
# wrapper, we use a custom wrapper that has different __eq__ and __hash__
# behavior.  Note that we subsequently store this weak key directly in an
# ORDINARY dictionary, since the newly constructed WeakIdKey's only use would
# be a dictionary so it would have no strong references.  Ensuring that
# only live WeakIdKeys are in the map is handled by putting finalizers on the
# original key object.


# It is simpler to implement this with composition, but if we want to
# directly reuse the callback mechanism on weakref, we need the weakref
# and the key to be exactly the same object.  Reusing the callback mechanism
# minimizes the divergence between our implementation and Lib/weakref.py
#
# NB: Prefer using this when working with weakrefs of Tensors; e.g., do
# WeakIdRef(tensor) rather than weakref.ref(tensor); it handles a number of
# easy to get wrong cases transparently for you.
class WeakIdRef(weakref.ref):
    __slots__ = ["_id"]

    def __init__(self, key, callback=None) -> None:
        # Unlike stock weakref, which preserves hash semantics of the
        # original object but lazily defers hash calls until the first
        # time the user attempts to hash the weakref, we can eagerly
        # cache the id of the key as we know this is definitely the hash
        # method
        self._id = id(key)
        super().__init__(key, callback)  # type: ignore[call-arg]

    def __call__(self):
        r = super().__call__()
        # Special logic for Tensor PyObject resurrection
        if hasattr(r, "_fix_weakref"):
            r._fix_weakref()  # type: ignore[union-attr]
        return r

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        # An attractive but wrong alternate implementation is to only test if
        # the stored _ids match.  This can lead to an ABA problem if you have:
        #
        #   a1 = A()
        #   w1 = WeakIdRef(a1)
        #   del a1
        #   a2 = A()  # suppose it gets the same ID as a1
        #   w2 = WeakIdRef(a2)
        #   print(w1 == w2)
        #
        # This should be False, as a1 and a2 are unrelated (and a1 is
        # dead anyway)
        a = self()
        b = other()
        if a is not None and b is not None:
            return a is b
        return self is other


# This is the same as WeakIdRef but equality is checked using hash() rather than id.
# This will be equivalent to the one above except for classes where hash is not their id.
class _WeakHashRef(weakref.ref):
    __slots__ = ["_id"]

    def __init__(self, key, callback=None) -> None:
        # Unlike stock weakref, which preserves hash semantics of the
        # original object but lazily defers hash calls until the first
        # time the user attempts to hash the weakref, we can eagerly
        # cache the id of the key as we know this is definitely the hash
        # method
        self._id = hash(key)
        super().__init__(key, callback)  # type: ignore[call-arg]

    def __call__(self):
        r = super().__call__()
        # Special logic for Tensor PyObject resurrection
        if hasattr(r, "_fix_weakref"):
            r._fix_weakref()  # type: ignore[union-attr]
        return r

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        # Use hash equality to determine ref equality.
        # ScriptObject implements __hash__ to return the wrapped IValue's id, so
        # this is equivalent to doing an identity comparison.
        a = self()
        b = other()
        if a is not None and b is not None:
            return hash(a) == hash(b)
        return self is other


# This is directly adapted from cpython/Lib/weakref.py
class WeakIdKeyDictionary(MutableMapping):
    def __init__(self, dict=None, ref_type=WeakIdRef) -> None:  # CHANGED
        self.data = {}

        self.ref_type = ref_type  # CHANGED

        def remove(k, selfref=ref(self)) -> None:
            self = selfref()
            if self is not None:
                if self._iterating:
                    self._pending_removals.append(k)
                else:
                    try:
                        del self.data[k]
                    except KeyError:
                        pass

        self._remove = remove
        # A list of dead weakrefs (keys to be removed)
        self._pending_removals = []
        self._iterating = set()
        self._dirty_len = False
        if dict is not None:
            self.update(dict)

    def _commit_removals(self) -> None:
        # NOTE: We don't need to call this method before mutating the dict,
        # because a dead weakref never compares equal to a live weakref,
        # even if they happened to refer to equal objects.
        # However, it means keys may already have been removed.
        pop = self._pending_removals.pop
        d = self.data
        while True:
            try:
                key = pop()
            except IndexError:
                return

            try:
                del d[key]
            except KeyError:
                pass

    def _scrub_removals(self) -> None:
        d = self.data
        self._pending_removals = [k for k in self._pending_removals if k in d]
        self._dirty_len = False

    def __delitem__(self, key) -> None:
        self._dirty_len = True
        del self.data[self.ref_type(key)]  # CHANGED

    def __getitem__(self, key):
        return self.data[self.ref_type(key)]  # CHANGED

    def __len__(self) -> int:
        if self._dirty_len and self._pending_removals:
            # self._pending_removals may still contain keys which were
            # explicitly removed, we have to scrub them (see issue #21173).
            self._scrub_removals()
        return len(self.data) - len(self._pending_removals)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} at {id(self):#x}>"

    def __setitem__(self, key, value) -> None:
        self.data[self.ref_type(key, self._remove)] = value  # CHANGED

    def copy(self):
        new = WeakIdKeyDictionary()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = value
        return new

    __copy__ = copy

    def __deepcopy__(self, memo):
        from copy import deepcopy

        new = self.__class__()
        with _IterationGuard(self):
            for key, value in self.data.items():
                o = key()
                if o is not None:
                    new[o] = deepcopy(value, memo)
        return new

    def get(self, key, default=None):
        return self.data.get(self.ref_type(key), default)  # CHANGED

    def __contains__(self, key) -> bool:
        try:
            wr = self.ref_type(key)  # CHANGED
        except TypeError:
            return False
        return wr in self.data

    def items(self):
        with _IterationGuard(self):
            for wr, value in self.data.items():
                key = wr()
                if key is not None:
                    yield key, value

    def keys(self):
        with _IterationGuard(self):
            for wr in self.data:
                obj = wr()
                if obj is not None:
                    yield obj

    __iter__ = keys

    def values(self):
        with _IterationGuard(self):
            for wr, value in self.data.items():
                if wr() is not None:
                    yield value

    def keyrefs(self):
        """Return a list of weak references to the keys.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the keys around longer than needed.

        """
        return list(self.data)

    def popitem(self):
        self._dirty_len = True
        while True:
            key, value = self.data.popitem()
            o = key()
            if o is not None:
                return o, value

    # pyrefly: ignore [bad-override]
    def pop(self, key, *args):
        self._dirty_len = True

        return self.data.pop(self.ref_type(key), *args)  # CHANGED

    def setdefault(self, key, default=None):
        return self.data.setdefault(
            self.ref_type(key, self._remove), default
        )  # CHANGED

    def update(self, dict=None, **kwargs) -> None:  # type: ignore[override]
        d = self.data
        if dict is not None:
            if not hasattr(dict, "items"):
                dict = type({})(dict)
            for key, value in dict.items():
                d[self.ref_type(key, self._remove)] = value  # CHANGED
        if kwargs:
            self.update(kwargs)

    def __ior__(self, other):
        self.update(other)
        return self

    def __or__(self, other):
        if isinstance(other, _collections_abc.Mapping):
            c = self.copy()
            c.update(other)
            return c
        return NotImplemented

    def __ror__(self, other):
        if isinstance(other, _collections_abc.Mapping):
            c = self.__class__()
            c.update(other)
            c.update(self)
            return c
        return NotImplemented

    # Default Mapping equality will tests keys for equality, but
    # we want to test ids for equality
    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return {id(k): v for k, v in self.items()} == {
            id(k): v for k, v in other.items()
        }


# Convenience alias
WeakTensorKeyDictionary = WeakIdKeyDictionary


class TensorWeakRef:
    """Wrapper around a weak ref of a Tensor that handles the _fix_weakref() call required when unwrapping a Tensor weakref."""

    ref: WeakRef[Tensor]

    def __init__(self, tensor: Tensor) -> None:
        if not isinstance(tensor, Tensor):
            raise AssertionError(f"expected torch.Tensor, got {type(tensor)}.")
        self.ref = weakref.ref(tensor)

    def __call__(self):
        out = self.ref()
        if out is None:
            return out
        if not isinstance(out, Tensor):
            raise AssertionError(f"expected torch.Tensor, got {type(out)}.")
        # TODO, add _fix_weakref type binding
        out._fix_weakref()  # type: ignore[attr-defined]
        return out
<|endoftext|># mypy: allow-untyped-defs
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def remove_triton_function_declaration(source_code: str) -> str:
    remove_head = re.sub(r"(\n.+\s\'\'\'\n)", "\n", source_code)
    remove_tail = re.sub(r"(\'\'\'\,.+)", "\n", remove_head)
    return remove_tail


def remove_async_compile(source_code: str) -> str:
    remove_top_level = str.replace(source_code, "async_compile = AsyncCompile()", "")
    remove_compile = str.replace(remove_top_level, "async_compile.wait(globals())", "")
    remove_del = str.replace(remove_compile, "del async_compile", "")
    return remove_del


def rename_kernels(source_code: str) -> str:
    pattern = r"(\w+)\s*=\s*async_compile\.triton\('triton_',\s"
    triton_kernel_decl = "def triton_"
    matches = [
        (match.end(), match.group(1))
        for match in re.finditer(pattern, source_code, re.DOTALL)
    ]

    # Starting from the last match to avoid issues with shifting indices after replacements
    for end_index, captured_string in reversed(matches):
        # Find the index of the next "B" after the current match
        index_of_B = source_code.find(triton_kernel_decl, end_index)
        if index_of_B != -1:
            # Replace the triton_kernel_decl with the captured string
            source_code = (
                source_code[:index_of_B]
                + f"def {captured_string}"
                + source_code[index_of_B + len(triton_kernel_decl) :]
            )
        else:
            # If triton_kernel_decl is not found after the current match, continue to the next
            continue

    return source_code


def merge_params(original_params: list[str], new_params: list[str]) -> list[str]:
    for idx in range(len(new_params)):
        if new_params[idx] == "T":
            new_params[idx] = original_params[idx]
    return new_params


def add_launch_params(
    original: str, kernel_to_params: dict[str, tuple[str, str]]
) -> str:
    # Regex to match the function call in the original string
    pattern = r"(\w+)\.run\((.*)\)"

    def replace(match) -> str:
        # Extract parts from the regex match
        func_name = match.group(1)
        params = match.group(2)
        new_params, grid = kernel_to_params[func_name]
        new_params = merge_params(params.split(", "), new_params.split(", "))

        # Format the new function call
        new_string = f"{func_name}[{grid}]({', '.join(new_params)})"
        return new_string

    transformed = re.sub(pattern, replace, original)

    remove_inductor_wrappers = re.sub(
        r"@triton_heuristics[^@]*@triton.jit",
        r"@triton.jit",
        transformed,
        flags=re.DOTALL,
    )

    return remove_inductor_wrappers


def process_file(
    input_filename: str, output_filename: str, auto_generate_params: bool = True
) -> str:
    with open(input_filename) as file:
        source_code = file.read()

    transformed_code = source_code
    if "def triton_(" in source_code:
        raise RuntimeError(
            "Need to run original Pytorch code generating kernels with TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1"
        )
    # transformed_code = rename_kernels(transformed_code)
    transformed_code = remove_triton_function_declaration(transformed_code)
    transformed_code = remove_async_compile(transformed_code)

    launch_params_filename = f"{input_filename}.launch_params"

    # Auto-generate launch_params if they don't exist and auto_generate_params is True
    if not os.path.exists(launch_params_filename) and auto_generate_params:
        print(f"Launch params file {launch_params_filename} not found. Generating...")
        try:
            # Set environment variable and run the input file
            env = os.environ.copy()
            env["TORCHINDUCTOR_DUMP_LAUNCH_PARAMS"] = "1"

            result = subprocess.run(
                [sys.executable, input_filename],
                env=env,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(input_filename) or ".",
            )

            if result.returncode != 0:
                print(f"Error running {input_filename}:")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                raise RuntimeError(
                    f"Failed to generate launch params. Command failed with return code {result.returncode}"
                )

            print(f"Successfully generated {launch_params_filename}")

        except Exception as e:
            raise RuntimeError(
                f"Failed to generate launch params by running {input_filename}: {str(e)}"
            ) from e

    if not os.path.exists(launch_params_filename):
        raise RuntimeError(
            f"Missing {launch_params_filename}. Run `TORCHINDUCTOR_DUMP_LAUNCH_PARAMS=1 python {input_filename}` first."
        )

    with open(launch_params_filename) as f:
        launch_params_meta = f.readlines()

    split_params = [i.split("|") for i in launch_params_meta]
    kernel_args_grid = {a.strip(): (b.strip(), c.strip()) for a, b, c in split_params}
    transformed_code = add_launch_params(transformed_code, kernel_args_grid)

    with open(output_filename, "w") as file:
        file.write(transformed_code)
    print(f"Successfully generated {output_filename}")
    return transformed_code


def get_clean_triton(
    input_path: Path,
    output_path: Path = Path("triton_only_repro.py"),
    auto_generate_params: bool = True,
):
    """Run experiments and output results to file

    Args:
        input_path (Path): Path to inductor generated output code
        output_path (Path): Path to write out the new python file
        auto_generate_params (bool): Whether to automatically generate launch_params if missing
    """
    return process_file(str(input_path), str(output_path), auto_generate_params)


if __name__ == "__main__":
    """Sample usage:
    # Running sweep
    python _get_clean_triton.py output_code.py

    # To disable auto-generation of launch params:
    python _get_clean_triton.py output_code.py --no-auto-generate
    """
    parser = argparse.ArgumentParser(
        description="Clean Inductor generated code to remove Inductor dependencies"
    )

    # Add the arguments
    parser.add_argument(
        "input_path", type=Path, help="Path to inductor generated output code"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("triton_only_repro.py"),
        help="Path to write out the clean triton output",
    )
    parser.add_argument(
        "--no-auto-generate",
        action="store_true",
        help="Disable automatic generation of launch_params file",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    result = get_clean_triton(
        args.input_path, args.output_path, not args.no_auto_generate
    )
<|endoftext|># mypy: allow-untyped-defs
import functools

import torch


# The Mkldnn* classes subclass torch.jit.ScriptModule, whose metaclass compiles
# @torch.jit.script_method members at class-definition time. Building them lazily
# keeps `import torch.utils.mkldnn` (and its transitive importers) free of any
# TorchScript compilation until MKLDNN conversion is actually used.
@functools.lru_cache(None)
def _get_mkldnn_classes():
    class MkldnnLinear(torch.jit.ScriptModule):
        def __init__(self, dense_module, dtype) -> None:
            super().__init__()
            self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))
            if dense_module.bias is not None:
                # Bias can be fp32 or bf16 for OneDNN bf16 path, but for good accuracy,
                # we use fp32 dtype.
                self.register_buffer('bias', dense_module.bias.to_mkldnn())
            else:
                # TODO: Remove this once ScriptModule supports registering None buffer
                self.register_buffer(
                    'bias',
                    torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

        @torch.jit.script_method
        def __getstate__(self):
            return (self.weight.to_dense(), self.bias.to_dense(), self.training)

        @torch.jit.script_method
        def __setstate__(self, state):
            self.weight = state[0].to_mkldnn()
            self.bias = state[1].to_mkldnn()
            self.training = state[2]

        @torch.jit.script_method
        def forward(self, x):
            x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
            y_mkldnn = torch._C._nn.mkldnn_linear(x_mkldnn, self.weight, self.bias)
            y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
            return y

    class _MkldnnConvNd(torch.jit.ScriptModule):
        """Common base of MkldnnConv1d and MkldnnConv2d."""

        __constants__ = ['stride', 'padding', 'dilation', 'groups']

        def __init__(self, dense_module) -> None:
            super().__init__()

            self.stride = dense_module.stride
            self.padding = dense_module.padding
            self.dilation = dense_module.dilation
            self.groups = dense_module.groups

            if dense_module.bias is not None:
                self.register_buffer('bias', dense_module.bias.to_mkldnn())
            else:
                # Bias can be fp32 or bf16 for OneDNN bf16 path, but for good accuracy,
                # we use fp32 dtype.
                # TODO: Remove this once ScriptModule supports registering None buffer
                self.register_buffer(
                    'bias',
                    torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

        @torch.jit.script_method
        def __getstate__(self):
            return (self.weight.to_dense(), self.bias.to_dense(), self.training)

        @torch.jit.script_method
        def forward(self, x):
            return torch.mkldnn_convolution(
                x,
                self.weight,
                self.bias,
                self.padding,
                self.stride,
                self.dilation,
                self.groups)

    class MkldnnConv1d(_MkldnnConvNd):
        def __init__(self, dense_module, dtype) -> None:
            super().__init__(dense_module)

            self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

        @torch.jit.script_method
        def __setstate__(self, state):
            self.weight = state[0].to_mkldnn()
            self.bias = state[1].to_mkldnn()
            self.training = state[2]

    class MkldnnConv2d(_MkldnnConvNd):
        def __init__(self, dense_module, dtype) -> None:
            super().__init__(dense_module)

            self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv2d_weight(
                dense_module.weight.to_mkldnn(dtype),
                self.padding,
                self.stride,
                self.dilation,
                self.groups))

        @torch.jit.script_method
        def __setstate__(self, state):
            self.weight = torch._C._nn.mkldnn_reorder_conv2d_weight(
                state[0].to_mkldnn(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups)
            self.bias = state[1].to_mkldnn()
            self.training = state[2]

    class MkldnnConv3d(_MkldnnConvNd):
        def __init__(self, dense_module, dtype) -> None:
            super().__init__(dense_module)

            self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv3d_weight(
                dense_module.weight.to_mkldnn(dtype),
                self.padding,
                self.stride,
                self.dilation,
                self.groups))

        @torch.jit.script_method
        def __setstate__(self, state):
            self.weight = torch._C._nn.mkldnn_reorder_conv3d_weight(
                state[0].to_mkldnn(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups)
            self.bias = state[1].to_mkldnn()
            self.training = state[2]

    class MkldnnBatchNorm(torch.jit.ScriptModule):
        __constants__ = ['exponential_average_factor', 'eps']

        def __init__(self, dense_module) -> None:
            super().__init__()

            if dense_module.training:
                raise AssertionError("Only support eval mode batchnorm for mkldnn path now")
            if not dense_module.track_running_stats:
                raise AssertionError("Only support track_running_stats=True for mkldnn path now")
            if not dense_module.affine:
                raise AssertionError("Only support affine=True for mkldnn path now")

            if dense_module.momentum is None:
                self.exponential_average_factor = 0.0
            else:
                self.exponential_average_factor = dense_module.momentum
            self.eps = dense_module.eps

            self.register_buffer('weight', dense_module.weight.to_mkldnn())
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
            self.register_buffer('running_mean', dense_module.running_mean.to_mkldnn())
            self.register_buffer('running_var', dense_module.running_var.to_mkldnn())

        @torch.jit.script_method
        def __getstate__(self):
            weight = self.weight.to_dense()
            bias = self.bias.to_dense()
            running_mean = self.running_mean.to_dense()
            running_var = self.running_var.to_dense()
            return (weight, bias, running_mean, running_var, self.training)

        @torch.jit.script_method
        def __setstate__(self, state):
            self.weight = state[0].to_mkldnn()
            self.bias = state[1].to_mkldnn()
            self.running_mean = state[2].to_mkldnn()
            self.running_var = state[3].to_mkldnn()
            self.training = state[4]

        @torch.jit.script_method
        def forward(self, x):
            return torch.batch_norm(
                x,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_var,
                False,  # training
                self.exponential_average_factor,
                self.eps,
                False,  # cuda_enabled
            )

    class MkldnnPrelu(torch.jit.ScriptModule):
        def __init__(self, dense_module, dtype) -> None:
            super().__init__()
            self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

        @torch.jit.script_method
        def __getstate__(self):
            return (self.weight.to_dense(), self.training)

        @torch.jit.script_method
        def __setstate__(self, state):
            self.weight = state[0].to_mkldnn()
            self.training = state[1]

        @torch.jit.script_method
        def forward(self, x):
            x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
            y_mkldnn = torch.prelu(x_mkldnn, self.weight)
            y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
            return y

    return {
        'MkldnnLinear': MkldnnLinear,
        '_MkldnnConvNd': _MkldnnConvNd,
        'MkldnnConv1d': MkldnnConv1d,
        'MkldnnConv2d': MkldnnConv2d,
        'MkldnnConv3d': MkldnnConv3d,
        'MkldnnBatchNorm': MkldnnBatchNorm,
        'MkldnnPrelu': MkldnnPrelu,
    }


def __getattr__(name):
    classes = _get_mkldnn_classes()
    if name in classes:
        return classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def to_mkldnn(module, dtype=torch.float):
    if dtype not in (torch.float, torch.bfloat16, torch.half):
        raise AssertionError("MKLDNN only support float, bfloat16, and half path now")

    classes = _get_mkldnn_classes()
    MkldnnLinear = classes['MkldnnLinear']
    MkldnnConv1d = classes['MkldnnConv1d']
    MkldnnConv2d = classes['MkldnnConv2d']
    MkldnnConv3d = classes['MkldnnConv3d']
    MkldnnBatchNorm = classes['MkldnnBatchNorm']
    MkldnnPrelu = classes['MkldnnPrelu']

    def m_fn(m, d):
        if isinstance(m, torch.nn.Linear):
            return MkldnnLinear(m, d)
        elif isinstance(m, torch.nn.Conv1d):
            return MkldnnConv1d(m, d)
        elif isinstance(m, torch.nn.Conv2d):
            return MkldnnConv2d(m, d)
        elif isinstance(m, torch.nn.Conv3d):
            return MkldnnConv3d(m, d)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            # For batchnorm bf16 path, OneDNN requires weight and bias need fp32 dtype.
            # so it doesn't need dtype argument.
            return MkldnnBatchNorm(m)
        elif isinstance(m, torch.nn.PReLU):
            return MkldnnPrelu(m, d)
        else:
            return m

    def m_fn_rec(m, d):
        new_m = m_fn(m, d)
        for name, sub_m in m.named_children():
            setattr(new_m, name, m_fn_rec(sub_m, d))
        return new_m

    return m_fn_rec(module, dtype)
<|endoftext|># mypy: allow-untyped-defs
import logging
import weakref
from typing import TYPE_CHECKING

import torch
from torch.autograd.graph import register_multi_grad_hook
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.utils._pytree import tree_flatten


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


logger = logging.getLogger(__name__)


__all__ = ["ModuleTracker"]


class ModuleTracker:
    """
    ``ModuleTracker`` is a context manager that tracks the nn.Module hierarchy during execution
    so that other system can query which Module is currently being executed (or its backward is being
    executed).

    You can access the ``parents`` attribute on this context manager to get the set of all the
    Modules currently being executed via their fqn (fully qualified name, also used as the key within
    the state_dict).
    You can access the ``is_bw`` attribute to know if you are currently running in backward or not.

    Note that ``parents`` is never empty and always contains the "Global" key. The ``is_bw`` flag
    will remain ``True`` after the forward until another Module is executed. If you need it to be
    more accurate, please submit an issue requesting this. Adding a map from fqn to the module instance
    is possible but not done yet, please submit an issue requesting this if you need it.

    Example usage

    .. code-block:: python

        mod = torch.nn.Linear(2, 2)

        with ModuleTracker() as tracker:
            # Access anything during the forward pass
            def my_linear(m1, m2, bias):
                print(f"Current modules: {tracker.parents}")
                return torch.mm(m1, m2.t()) + bias

            torch.nn.functional.linear = my_linear

            mod(torch.rand(2, 2))

    """

    parents: set[str]
    """
    A Set containing the fqn for each module currently running their forward
    """

    def __init__(self) -> None:
        self.parents = {"Global"}
        self._known_modules: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._seen_modules: weakref.WeakSet = weakref.WeakSet()
        self._has_callback = False
        self._hooks: list[RemovableHandle] = []

    def _maybe_set_engine_callback(self) -> None:
        # This assumes no concurrent calls to backward
        if self._has_callback:
            return

        def callback() -> None:
            self.parents = {"Global"}
            self._has_callback = False

        torch.autograd.Variable._execution_engine.queue_callback(callback)
        self._has_callback = True

    @property
    def is_bw(self):
        """
        A boolean marking if this is currently running during the backward pass or not
        """
        return torch._C._current_graph_task_id() != -1

    def _get_mod_name(self, mod):
        if mod not in self._known_modules:
            self._known_modules[mod] = type(mod).__name__
        mod_name = self._known_modules[mod]
        if mod not in self._seen_modules:
            for name, submod in mod.named_children():
                self._known_modules[submod] = f"{mod_name}.{name}"
                self._get_mod_name(submod)
            self._seen_modules.add(mod)
        return mod_name

    def _get_append_fn(self, name, is_bw):
        def fn(*args) -> None:
            if is_bw:
                self._maybe_set_engine_callback()
            if name in self.parents:
                logger.info(
                    "The module hierarchy tracking seems to be broken as this Module was already entered. %s during %s",
                    name,
                    "backward" if is_bw else "forward",
                )
            self.parents.add(name)

        return fn

    def _get_pop_fn(self, name, is_bw):
        def fn(*args) -> None:
            if name in self.parents:
                self.parents.remove(name)
            else:
                logger.info(
                    "The Module hierarchy tracking is confused as we're exiting a Module that was never entered. %s during %s",
                    name,
                    "backward" if is_bw else "forward",
                )

        return fn

    def _fw_pre_hook(self, mod, input) -> None:
        name = self._get_mod_name(mod)
        self._get_append_fn(name, False)()

        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            self._hooks.append(
                register_multi_grad_hook(tensors, self._get_pop_fn(name, True))
            )

    def _fw_post_hook(self, mod, input, output) -> None:
        name = self._get_mod_name(mod)
        self._get_pop_fn(name, False)()

        args, _ = tree_flatten(output)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            self._hooks.append(
                register_multi_grad_hook(tensors, self._get_append_fn(name, True))
            )

    def __enter__(self):
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(self._fw_post_hook)
        return self

    def __exit__(self, *args):
        self._fw_pre_handle.remove()
        self._fw_post_handle.remove()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
<|endoftext|>from collections.abc import Sequence
from pathlib import Path
from re import match as _match


def read_file(fname: Path | str) -> list[str]:
    with open(fname, encoding="utf-8") as f:
        return f.readlines()


def _embed_headers(
    content: list[str], include_dirs: list[Path], processed_files: set[str]
) -> str:
    for line_idx, cur_line in enumerate(content):
        # Eliminate warning: `#pragma once in main file`
        if cur_line.startswith("#pragma once"):
            content[line_idx] = ""
            continue
        m = _match('^\\s*#include\\s*[<"]([^>"]+)[>"]', cur_line)
        if m is None:
            continue
        for include_dir in include_dirs:
            path = include_dir / m[1]
            if not path.exists():
                continue
            if str(path) in processed_files:
                content[line_idx] = ""
                continue
            processed_files.add(str(path))
            content[line_idx] = _embed_headers(
                read_file(path), include_dirs, processed_files
            )
            break
    return "".join(content)


def embed_headers(
    fname: str, include_dirs: Sequence[str] | Sequence[Path] | str | None = None
) -> str:
    if include_dirs is None:
        base_dir = Path(__file__).parent.parent.parent
        include_dirs = [base_dir, base_dir / "aten" / "src"]
    elif isinstance(include_dirs, str):
        include_dirs = [Path(include_dirs)]
    else:
        include_dirs = [Path(x) for x in include_dirs]

    return _embed_headers(read_file(fname), include_dirs, {fname})


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(f"Usage:\n {sys.argv[0]} filename")
        sys.exit(1)
    print(embed_headers(sys.argv[1]))
<|endoftext|># mypy: allow-untyped-defs
from typing import TypeVar

import torch


T = TypeVar("T")


# returns if all are the same mode
def all_same_mode(modes):
    return all(tuple(mode == modes[0] for mode in modes))


no_dispatch = torch._C._DisableTorchDispatch
<|endoftext|># mypy: allow-untyped-defs
from torch._C import _get_cpp_backtrace

def get_cpp_backtrace(frames_to_skip=0, maximum_number_of_frames=64) -> str:
    r"""
    Return a string containing the C++ stack trace of the current thread.

    Args:
        frames_to_skip (int): the number of frames to skip from the top of the stack
        maximum_number_of_frames (int): the maximum number of frames to return
    """
    return _get_cpp_backtrace(frames_to_skip, maximum_number_of_frames)
<|endoftext|># mypy: allow-untyped-defs

import torch._C


def format_time(time_us=None, time_ms=None, time_s=None) -> str:
    """Define time formatting."""
    if sum([time_us is not None, time_ms is not None, time_s is not None]) != 1:
        raise AssertionError("Expected only one of time_us, time_ms, time_s is given.")

    US_IN_SECOND = 1e6
    US_IN_MS = 1e3

    if time_us is None:
        if time_ms is not None:
            time_us = time_ms * US_IN_MS
        elif time_s is not None:
            time_us = time_s * US_IN_SECOND
        else:
            raise AssertionError("Shouldn't reach here :)")

    if time_us >= US_IN_SECOND:
        return f'{time_us / US_IN_SECOND:.3f}s'
    if time_us >= US_IN_MS:
        return f'{time_us / US_IN_MS:.3f}ms'
    return f'{time_us:.3f}us'


class ExecutionStats:
    def __init__(self, c_stats, benchmark_config) -> None:
        self._c_stats = c_stats
        self.benchmark_config = benchmark_config

    @property
    def latency_avg_ms(self):
        return self._c_stats.latency_avg_ms

    @property
    def num_iters(self):
        return self._c_stats.num_iters

    @property
    def iters_per_second(self):
        """Return total number of iterations per second across all calling threads."""
        return self.num_iters / self.total_time_seconds

    @property
    def total_time_seconds(self):
        return self.num_iters * (
            self.latency_avg_ms / 1000.0) / self.benchmark_config.num_calling_threads

    def __str__(self) -> str:
        return '\n'.join([
            "Average latency per example: " + format_time(time_ms=self.latency_avg_ms),
            f"Total number of iterations: {self.num_iters}",
            f"Total number of iterations per second (across all threads): {self.iters_per_second:.2f}",
            "Total time: " + format_time(time_s=self.total_time_seconds)
        ])


class ThroughputBenchmark:
    """
    This class is a wrapper around a c++ component throughput_benchmark::ThroughputBenchmark.

    This wrapper on the throughput_benchmark::ThroughputBenchmark component is responsible
    for executing a PyTorch module (nn.Module or ScriptModule) under an inference
    server like load. It can emulate multiple calling threads to a single module
    provided. In the future we plan to enhance this component to support inter and
    intra-op parallelism as well as multiple models running in a single process.

    Please note that even though nn.Module is supported, it might incur an overhead
    from the need to hold GIL every time we execute Python code or pass around
    inputs as Python objects. As soon as you have a ScriptModule version of your
    model for inference deployment it is better to switch to using it in this
    benchmark.

    Example::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> from torch.utils import ThroughputBenchmark
        >>> bench = ThroughputBenchmark(my_module)
        >>> # Pre-populate benchmark's data set with the inputs
        >>> for input in inputs:
        ...     # Both args and kwargs work, same as any PyTorch Module / ScriptModule
        ...     bench.add_input(input[0], x2=input[1])
        >>> # Inputs supplied above are randomly used during the execution
        >>> stats = bench.benchmark(
        ...     num_calling_threads=4,
        ...     num_warmup_iters = 100,
        ...     num_iters = 1000,
        ... )
        >>> print("Avg latency (ms): {}".format(stats.latency_avg_ms))
        >>> print("Number of iterations: {}".format(stats.num_iters))
    """

    def __init__(self, module) -> None:
        if isinstance(module, torch.jit.ScriptModule):
            self._benchmark = torch._C.ThroughputBenchmark(module._c)
        else:
            self._benchmark = torch._C.ThroughputBenchmark(module)

    def run_once(self, *args, **kwargs):
        """
        Given input id (input_idx) run benchmark once and return prediction.

        This is useful for testing that benchmark actually runs the module you
        want it to run. input_idx here is an index into inputs array populated
        by calling add_input() method.
        """
        return self._benchmark.run_once(*args, **kwargs)

    def add_input(self, *args, **kwargs) -> None:
        """
        Store a single input to a module into the benchmark memory and keep it there.

        During the benchmark execution every thread is going to pick up a
        random input from all the inputs ever supplied to the benchmark via
        this function.
        """
        self._benchmark.add_input(*args, **kwargs)

    def benchmark(
            self,
            num_calling_threads=1,
            num_warmup_iters=10,
            num_iters=100,
            profiler_output_path=""):
        """
        Run a benchmark on the module.

        Args:
            num_warmup_iters (int): Warmup iters are used to make sure we run a module
                a few times before actually measuring things. This way we avoid cold
                caches and any other similar problems. This is the number of warmup
                iterations for each of the threads separately

            num_iters (int): Number of iterations the benchmark should run with.
                This number is separate from the warmup iterations. Also the number is
                shared across all the threads. Once the num_iters iterations across all
                the threads is reached, we will stop execution. Though total number of
                iterations might be slightly larger. Which is reported as
                stats.num_iters where stats is the result of this function

            profiler_output_path (str): Location to save Autograd Profiler trace.
                If not empty, Autograd Profiler will be enabled for the main benchmark
                execution (but not the warmup phase). The full trace will be saved
                into the file path provided by this argument


        This function returns BenchmarkExecutionStats object which is defined via pybind11.
        It currently has two fields:
            - num_iters - number of actual iterations the benchmark have made
            - avg_latency_ms - average time it took to infer on one input example in milliseconds
        """
        config = torch._C.BenchmarkConfig()
        config.num_calling_threads = num_calling_threads
        config.num_warmup_iters = num_warmup_iters
        config.num_iters = num_iters
        config.profiler_output_path = profiler_output_path
        c_stats = self._benchmark.benchmark(config)
        return ExecutionStats(c_stats, config)
<|endoftext|>from __future__ import annotations

import functools
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Concatenate, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec


if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future


_P = ParamSpec("_P")
_T = TypeVar("_T")
_C = TypeVar("_C")

# Sentinel used to indicate that cache lookup failed.
_cache_sentinel = object()

_prefetch_executor = ThreadPoolExecutor(max_workers=2)


def prefetchable_cache(func: Callable[[], _T]) -> Callable[[], _T]:
    """
    Like functools.cache but with a prefetch() method that starts computing
    the value in a background thread, and a set() method for prepopulating.
    """
    _cache: _T | object = _cache_sentinel
    _lock = threading.Lock()
    _future: Future[_T] | None = None

    def wrapper() -> _T:
        nonlocal _cache, _future
        with _lock:
            if _cache is not _cache_sentinel:
                return _cache  # type: ignore[return-value]
            if _future is not None:
                _cache = _future.result()
                _future = None
                return _cache  # type: ignore[return-value]
            _cache = func()
            return _cache  # type: ignore[return-value]

    def set_val(val: _T) -> None:
        nonlocal _cache
        with _lock:
            if _cache is not _cache_sentinel:
                raise RuntimeError("prefetchable_cache value already set")
            _cache = val

    def clear() -> None:
        nonlocal _cache, _future
        with _lock:
            _cache = _cache_sentinel
            _future = None

    def prefetch() -> None:
        nonlocal _future
        with _lock:
            if _cache is not _cache_sentinel or _future is not None:
                return
            _future = _prefetch_executor.submit(func)

    wrapper.set = set_val  # type: ignore[attr-defined]
    wrapper.clear = clear  # type: ignore[attr-defined]
    wrapper.prefetch = prefetch  # type: ignore[attr-defined]
    return wrapper


def cache_method(
    f: Callable[Concatenate[_C, _P], _T],
) -> Callable[Concatenate[_C, _P], _T]:
    """
    Like `@functools.cache` but for methods.

    `@functools.cache` (and similarly `@functools.lru_cache`) shouldn't be used
    on methods because it caches `self`, keeping it alive
    forever. `@cache_method` ignores `self` so won't keep `self` alive (assuming
    no cycles with `self` in the parameters).

    Footgun warning: This decorator completely ignores self's properties so only
    use it when you know that self is frozen or won't change in a meaningful
    way (such as the wrapped function being pure).
    """
    cache_name = "_cache_method_" + f.__name__

    @functools.wraps(f)
    def wrap(self: _C, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        if kwargs:
            raise AssertionError("cache_method does not accept keyword arguments")
        if not (cache := getattr(self, cache_name, None)):
            cache = {}
            setattr(self, cache_name, cache)
        cached_value = cache.get(args, _cache_sentinel)
        if cached_value is not _cache_sentinel:
            return cached_value
        value = f(self, *args, **kwargs)
        cache[args] = value
        return value

    return wrap
<|endoftext|># mypy: allow-untyped-defs
# Extra utilities for working with context managers that should have been
# in the standard library but are not

import functools
import inspect
import sys
import warnings
from collections.abc import Callable
from typing import Any, cast, overload, TypeVar
from typing_extensions import Self


# Used for annotating the decorator usage of _DecoratorContextManager (e.g.,
# 'no_grad' and 'enable_grad').
# See https://mypy.readthedocs.io/en/latest/generics.html#declaring-decorators
FuncType = Callable[..., Any]
F = TypeVar("F", bound=FuncType)


def _wrap_generator(ctx_factory, func):
    """
    Wrap each generator invocation with the context manager factory.

    The input should be a function that returns a context manager,
    not a context manager itself, to handle one-shot context managers.
    """

    @functools.wraps(func)
    def generator_context(*args, **kwargs):
        gen = func(*args, **kwargs)

        # Generators are suspended and unsuspended at `yield`, hence we
        # make sure the grad mode is properly set every time the execution
        # flow returns into the wrapped generator and restored when it
        # returns through our `yield` to our caller (see PR #49017).
        try:
            # Issuing `None` to a generator fires it up
            with ctx_factory():
                response = gen.send(None)

            while True:
                try:
                    # Forward the response to our caller and get its next request
                    request = yield response

                except GeneratorExit:
                    # Inform the still active generator about its imminent closure
                    with ctx_factory():
                        gen.close()
                    raise

                except BaseException:
                    # Propagate the exception thrown at us by the caller
                    with ctx_factory():
                        response = gen.throw(*sys.exc_info())

                else:
                    # Pass the last request to the generator and get its response
                    with ctx_factory():
                        response = gen.send(request)

        # We let the exceptions raised above by the generator's `.throw` or
        # `.send` methods bubble up to our caller, except for StopIteration
        except StopIteration as e:
            # The generator informed us that it is done: take whatever its
            # returned value (if any) was and indicate that we're done too
            # by returning it (see docs for python's return-statement).
            return e.value

    return generator_context


def context_decorator(ctx, func):
    """
    Like contextlib.ContextDecorator.

    But with the following differences:
    1. Is done by wrapping, rather than inheritance, so it works with context
       managers that are implemented from C and thus cannot easily inherit from
       Python classes
    2. Wraps generators in the intuitive way (c.f. https://bugs.python.org/issue37743)
    3. Errors out if you try to wrap a class, because it is ambiguous whether
       or not you intended to wrap only the constructor

    The input argument can either be a context manager (in which case it must
    be a multi-shot context manager that can be directly invoked multiple times)
    or a callable that produces a context manager.
    """
    if callable(ctx) and hasattr(ctx, "__enter__"):
        raise AssertionError(
            f"Passed in {ctx} is both callable and also a valid context manager "
            "(has __enter__), making it ambiguous which interface to use.  If you "
            "intended to pass a context manager factory, rewrite your call as "
            "context_decorator(lambda: ctx()); if you intended to pass a context "
            "manager directly, rewrite your call as context_decorator(lambda: ctx)"
        )

    if not callable(ctx):

        def ctx_factory():
            return ctx

    else:
        ctx_factory = ctx

    if inspect.isclass(func):
        raise RuntimeError(
            "Cannot decorate classes; it is ambiguous whether or not only the "
            "constructor or all methods should have the context manager applied; "
            "additionally, decorating a class at definition-site will prevent "
            "use of the identifier as a conventional type.  "
            "To specify which methods to decorate, decorate each of them "
            "individually."
        )

    if inspect.isgeneratorfunction(func):
        return _wrap_generator(ctx_factory, func)

    @functools.wraps(func)
    def decorate_context(*args, **kwargs):
        # pyrefly: ignore [bad-context-manager]
        with ctx_factory():
            return func(*args, **kwargs)

    return decorate_context


class _DecoratorContextManager:
    """Allow a context manager to be used as a decorator."""

    def __call__(self, orig_func: F) -> F:
        if inspect.isclass(orig_func):
            warnings.warn(
                "Decorating classes is deprecated and will be disabled in "
                "future versions. You should only decorate functions or methods. "
                "To preserve the current behavior of class decoration, you can "
                "directly decorate the `__init__` method and nothing else.",
                FutureWarning,
                stacklevel=2,
            )
            func = cast(F, lambda *args, **kwargs: orig_func(*args, **kwargs))
        else:
            func = orig_func

        return cast(F, context_decorator(self.clone, func))

    def __enter__(self) -> None:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        raise NotImplementedError

    def clone(self):
        # override this method if your children class takes __init__ parameters
        return self.__class__()


class _NoParamDecoratorContextManager(_DecoratorContextManager):
    """Allow a context manager to be used as a decorator without parentheses."""

    @overload
    def __new__(cls, orig_func: F) -> F: ...  # type: ignore[misc]
    @overload
    def __new__(cls, orig_func: None = None) -> Self: ...

    def __new__(cls, orig_func: F | None = None) -> Self | F:  # type: ignore[misc]
        if orig_func is None:
            return super().__new__(cls)
        return cls()(orig_func)
<|endoftext|>from torch.utils.benchmark.utils.common import *  # noqa: F403
from torch.utils.benchmark.utils.timer import *  # noqa: F403
from torch.utils.benchmark.utils.compare import *  # noqa: F403
from torch.utils.benchmark.utils.fuzzer import *  # noqa: F403
from torch.utils.benchmark.utils.valgrind_wrapper.timer_interface import *  # noqa: F403
from torch.utils.benchmark.utils.sparse_fuzzer import *  # noqa: F403
<|endoftext|># mypy: allow-untyped-defs
from typing import Any, cast
from collections.abc import Callable

import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounterWithBackend
from torch.utils.benchmark import Timer


__all__ = ["bench_all", "benchmark_compile"]


_warned_tensor_cores = False
_default_float_32_precision = torch.get_float32_matmul_precision()

try:

    from tabulate import tabulate

    HAS_TABULATE = True
except ModuleNotFoundError:
    HAS_TABULATE = False
    tabulate = None  # type: ignore[assignment]
    print("tabulate is not installed, please pip install tabulate to use this utility")

if HAS_TABULATE:
    def _enable_tensor_cores() -> None:
        global _warned_tensor_cores

        if torch.cuda.is_available():
            if torch.backends.cuda.matmul.allow_tf32 is False and torch.cuda.get_device_capability() >= (8, 0):
                torch.set_float32_matmul_precision("high")
                if not _warned_tensor_cores:
                    print("Your GPU supports tensor cores")
                    print("we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`")
                    _warned_tensor_cores = True

    def _disable_tensor_cores() -> None:
        torch.set_float32_matmul_precision(_default_float_32_precision)

    def bench_loop(
        model: torch.nn.Module | Callable,
        sample_input: torch.Tensor | Any,
        num_iters: int = 5,
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn: Callable | None = None,
    ):
        # Define the statement and setup for the benchmark
        if optimizer and loss_fn:
            # Training mode
            stmt = """
    output = model(sample_input)
    loss = loss_fn(output) if loss_fn else output.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
            """
        else:
            # Inference mode
            stmt = "model(sample_input)"

        # Create the Timer object
        timer = Timer(
            stmt=stmt,
            globals={"model": model, "sample_input": sample_input, "optimizer": optimizer, "loss_fn": loss_fn},
        )


        result = timer.timeit(number=num_iters)

        # Get the average time per iteration in milliseconds
        avg_time = result.mean * 1000
        return round(avg_time, 2)

    def benchmark_compile(
        model: torch.nn.Module | Callable,
        sample_input: torch.Tensor | Any,
        num_iters: int = 5,
        backend: str | None = None,
        mode: str | None = "default",
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn : torch.nn.Module | Callable | None = None,
    ):
        """
        Use this utility to benchmark torch.compile
        """
        if backend:
            try:
                torch._dynamo.reset()
                compile_counter_with_backend = CompileCounterWithBackend(backend)
                opt_model = torch.compile(model, backend=compile_counter_with_backend, mode=mode)

                # Compilation only happens after the first inference
                compilation_time = bench_loop(opt_model, sample_input, 1, optimizer, loss_fn)

                running_time = bench_loop(opt_model, sample_input, num_iters, optimizer, loss_fn)

                if compile_counter_with_backend.frame_count == 0:
                    raise RuntimeError("No compilation occurred during benchmarking.")

                if compile_counter_with_backend.frame_count > 1:
                    raise RuntimeError("Recompilation occurred during benchmarking.")

            except Exception as e:
                print(e)
                print(f"Failed to compile {backend} with mode {mode}")
                return None, None
        else:
            opt_model = model
            compilation_time = None
            running_time = bench_loop(opt_model, sample_input, num_iters, optimizer, loss_fn)

        compilation_time = round(compilation_time, 2) if compilation_time else None
        running_time = round(running_time, 2) if running_time else None


        return compilation_time, running_time


    def bench_all(
        model : torch.nn.Module | Callable,
        sample_input: torch.Tensor | Any,
        num_iters : int = 5,
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn : torch.nn.Module | Callable | None = None,
    ):
        """
        This is a simple utility that can be used to benchmark torch.compile
        In particular it ensures that your GPU is setup to use tensor cores if it supports its
        It also tries out all the main backends and prints a table of results so you can easily compare them all
        Many of the backends have their own optional dependencies so please pip install them separately

        You will get one table for inference and another for training
        If you'd like to leverage this utility for training make sure to pass in a torch.optim.Optimizer

        The important warnings are
        Your GPU supports tensor cores
        we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`

        If a compilation fails for any reason including the dependency not being included
        then we will print Failed to compile {backend} with mode {mode}
        """
        field_names = ["Train/Inference", "Backend", "Mode", "Compilation Time", "Average Running Time"]
        table = []


        eager_time = None
        torch._dynamo.reset()
        _, eager_time = benchmark_compile(model, sample_input, num_iters, None, None, optimizer)
        table.append(
            [("Training" if optimizer else "Inference"), "Eager", "-", "-", f"{eager_time} ms"]
        )

        for backend in torch._dynamo.list_backends():

            if backend == "inductor":
                mode_options = cast(list[str | None], list(torch._inductor.list_mode_options().keys())) + [None]
                for mode in mode_options:
                    if mode == "default":
                        continue
                    torch._dynamo.reset()
                    try:
                        if torch.cuda.is_available():
                            _enable_tensor_cores()
                        compilation_time, running_time = benchmark_compile(
                            model, sample_input, num_iters, backend, mode, optimizer, loss_fn)
                    finally:
                        if torch.cuda.is_available():
                            _disable_tensor_cores()
                    table.append([
                        ("Training" if optimizer else "Inference"),
                        backend if backend else "-",
                        mode if mode is not None else "-",
                        f"{compilation_time} ms " if compilation_time else "-",
                        f"{running_time} ms " if running_time else "-",
                    ])

            else:
                torch._dynamo.reset()
                compilation_time, running_time = benchmark_compile(
                    model, sample_input, num_iters, backend, None, optimizer, loss_fn)

                if running_time is not None:
                    table.append([
                        ("Training" if optimizer else "Inference"),
                        backend, "-",
                        f"{compilation_time} ms " or "-",
                        f"{running_time} ms ",
                    ])


        # pyrefly: ignore [not-callable]
        return tabulate(table, headers=field_names, tablefmt="github")
<|endoftext|>"""JIT C++ strings into executables."""
import atexit
import os
import re
import shutil
import textwrap
import threading
from typing import Any

import torch
from torch.utils.benchmark.utils._stubs import CallgrindModuleType, TimeitModuleType
from torch.utils.benchmark.utils.common import _make_temp_dir
from torch.utils import cpp_extension


LOCK = threading.Lock()
SOURCE_ROOT = os.path.split(os.path.abspath(__file__))[0]

# We calculate uuid once at import time so that separate processes will have
# separate build roots, but threads will share the same build root.
# `cpp_extension` uses build root as part of the cache key, so per-invocation
# uuid's (e.g. different build root per _compile_template call) would lead to
# a 0% cache hit rate and spurious recompilation. Consider the following:
#   ```
#   setup = "auto x = torch::ones({1024, 1024});"
#   stmt = "torch::mm(x, x);"
#   for num_threads in [1, 2, 4, 8]:
#     print(Timer(stmt, setup, num_threads=num_threads, language="c++").blocked_autorange())
#   ````
# `setup` and `stmt` do not change, so we can reuse the executable from the
# first pass through the loop.
_BUILD_ROOT: str | None = None

def _get_build_root() -> str:
    global _BUILD_ROOT
    if _BUILD_ROOT is None:
        _BUILD_ROOT = _make_temp_dir(prefix="benchmark_utils_jit_build")
        # pyrefly: ignore [missing-argument]
        atexit.register(shutil.rmtree, _BUILD_ROOT)
    return _BUILD_ROOT


# BACK_TESTING_NOTE:
#   There are two workflows where this code could be used. One is the obvious
#   case where someone simply builds or installs PyTorch and uses Timer.
#   The other is that the entire `torch/utils/benchmark` folder from a CURRENT
#   PyTorch checkout is copy-pasted into a much OLDER version of the PyTorch
#   source code. This is what we refer to here as "back testing". The rationale
#   is that we might want to use current tooling to study some aspect of an
#   earlier version of PyTorch. (e.g. a regression.)
#
#   The problem is that Timer relies on several aspects of core PyTorch, namely
#   some binding functions for Valgrind symbols in `torch._C` and the
#   `torch.__config__._cxx_flags()` method. If we were to naively copy code
#   around this wouldn't work as the symbols of interest aren't present in
#   earlier versions of PyTorch. In order to work around this, we must add back
#   testing shims. These shims will never activate during normal use, but will
#   allow Timer to function outside of the "correct" version of PyTorch by
#   emulating functionality that was added later.
#
#   These shims are temporary, and as Timer becomes more integrated with
#   PyTorch the cost and complexity of such shims will increase. Once back
#   testing is no longer required (which is to say we have done enough historic
#   analysis and the shims no longer justify their maintenance and code
#   complexity costs) back testing paths will be removed.

CXX_FLAGS: list[str] | None
if hasattr(torch.__config__, "_cxx_flags"):
    try:
        CXX_FLAGS = torch.__config__._cxx_flags().strip().split()
        if CXX_FLAGS is not None and "-g" not in CXX_FLAGS:
            CXX_FLAGS.append("-g")
        # remove "-W" flags to allow build benchmarks
        # with a relaxed constraint of compiler versions
        if CXX_FLAGS is not None:
            CXX_FLAGS = list(filter(lambda x: not x.startswith("-W"), CXX_FLAGS))

    except RuntimeError:
        # We are in FBCode.
        CXX_FLAGS = None
else:
    # FIXME: Remove when back testing is no longer required.
    CXX_FLAGS = ["-O2", "-fPIC", "-g"]

EXTRA_INCLUDE_PATHS: list[str] = [os.path.join(SOURCE_ROOT, "valgrind_wrapper")]
CONDA_PREFIX = os.getenv("CONDA_PREFIX")
if CONDA_PREFIX is not None:
    # Load will automatically search /usr/include, but not conda include.
    EXTRA_INCLUDE_PATHS.append(os.path.join(CONDA_PREFIX, "include"))


COMPAT_CALLGRIND_BINDINGS: CallgrindModuleType | None = None
def get_compat_bindings() -> CallgrindModuleType:
    with LOCK:
        global COMPAT_CALLGRIND_BINDINGS
        if COMPAT_CALLGRIND_BINDINGS is None:
            COMPAT_CALLGRIND_BINDINGS = cpp_extension.load(
                name="callgrind_bindings",
                sources=[os.path.join(
                    SOURCE_ROOT,
                    "valgrind_wrapper",
                    "compat_bindings.cpp"
                )],
                extra_cflags=CXX_FLAGS,
                extra_include_paths=EXTRA_INCLUDE_PATHS,
            )
    return COMPAT_CALLGRIND_BINDINGS


def _compile_template(
    *,
    stmt: str,
    setup: str,
    global_setup: str,
    src: str,
    is_standalone: bool
) -> Any:
    for before, after, indentation in (
        ("// GLOBAL_SETUP_TEMPLATE_LOCATION", global_setup, 0),
        ("// SETUP_TEMPLATE_LOCATION", setup, 4),
        ("// STMT_TEMPLATE_LOCATION", stmt, 8)
    ):
        # C++ doesn't care about indentation so this code isn't load
        # bearing the way it is with Python, but this makes the source
        # look nicer if a human has to look at it.
        src = re.sub(
            before,
            textwrap.indent(after, " " * indentation)[indentation:],
            src
        )

    # We want to isolate different Timers. However `cpp_extension` will
    # cache builds which will significantly reduce the cost of repeated
    # invocations.
    with LOCK:
        name = f"timer_cpp_{abs(hash(src))}"
        build_dir = os.path.join(_get_build_root(), name)
        os.makedirs(build_dir, exist_ok=True)

        src_path = os.path.join(build_dir, "timer_src.cpp")
        with open(src_path, "w") as f:
            f.write(src)

    # `cpp_extension` has its own locking scheme, so we don't need our lock.
    return cpp_extension.load(
        name=name,
        sources=[src_path],
        build_directory=build_dir,
        extra_cflags=CXX_FLAGS,
        extra_include_paths=EXTRA_INCLUDE_PATHS,
        is_python_module=not is_standalone,
        is_standalone=is_standalone,
    )


def compile_timeit_template(*, stmt: str, setup: str, global_setup: str) -> TimeitModuleType:
    template_path: str = os.path.join(SOURCE_ROOT, "timeit_template.cpp")
    with open(template_path) as f:
        src: str = f.read()

    module = _compile_template(stmt=stmt, setup=setup, global_setup=global_setup, src=src, is_standalone=False)
    if not isinstance(module, TimeitModuleType):
        raise AssertionError("compiled module is not a TimeitModuleType")
    return module


def compile_callgrind_template(*, stmt: str, setup: str, global_setup: str) -> str:
    template_path: str = os.path.join(SOURCE_ROOT, "valgrind_wrapper", "timer_callgrind_template.cpp")
    with open(template_path) as f:
        src: str = f.read()

    target = _compile_template(stmt=stmt, setup=setup, global_setup=global_setup, src=src, is_standalone=True)
    if not isinstance(target, str):
        raise AssertionError("compiled target path is not a string")
    return target
<|endoftext|>"""Base shared classes and utilities."""

import collections
import contextlib
import dataclasses
import os
import shutil
import tempfile
import textwrap
import time
from typing import cast, Any
from collections.abc import Iterable, Iterator
import uuid

import torch


__all__ = ["TaskSpec", "Measurement", "select_unit", "unit_to_english", "trim_sigfig", "ordered_unique", "set_torch_threads"]


_MAX_SIGNIFICANT_FIGURES = 4
_MIN_CONFIDENCE_INTERVAL = 25e-9  # 25 ns

# Measurement will include a warning if the distribution is suspect. All
# runs are expected to have some variation; these parameters set the
# thresholds.
_IQR_WARN_THRESHOLD = 0.1
_IQR_GROSS_WARN_THRESHOLD = 0.25


@dataclasses.dataclass(init=True, repr=False, eq=True, frozen=True)
class TaskSpec:
    """Container for information used to define a Timer. (except globals)"""
    stmt: str
    setup: str
    global_setup: str = ""
    label: str | None = None
    sub_label: str | None = None
    description: str | None = None
    env: str | None = None
    num_threads: int = 1

    @property
    def title(self) -> str:
        """Best effort attempt at a string label for the measurement."""
        if self.label is not None:
            return self.label + (f": {self.sub_label}" if self.sub_label else "")
        elif "\n" not in self.stmt:
            return self.stmt + (f": {self.sub_label}" if self.sub_label else "")
        return (
            f"stmt:{f' ({self.sub_label})' if self.sub_label else ''}\n"
            f"{textwrap.indent(self.stmt, '  ')}"
        )

    def setup_str(self) -> str:
        return (
            "" if (self.setup == "pass" or not self.setup)
            else f"setup:\n{textwrap.indent(self.setup, '  ')}" if "\n" in self.setup
            else f"setup: {self.setup}"
        )

    def summarize(self) -> str:
        """Build TaskSpec portion of repr string for other containers."""
        sections = [
            self.title,
            self.description or "",
            self.setup_str(),
        ]
        return "\n".join([f"{i}\n" if "\n" in i else i for i in sections if i])

_TASKSPEC_FIELDS = tuple(i.name for i in dataclasses.fields(TaskSpec))


@dataclasses.dataclass(init=True, repr=False)
class Measurement:
    """The result of a Timer measurement.

    This class stores one or more measurements of a given statement. It is
    serializable and provides several convenience methods
    (including a detailed __repr__) for downstream consumers.
    """
    number_per_run: int
    raw_times: list[float]
    task_spec: TaskSpec
    metadata: dict[Any, Any] | None = None  # Reserved for user payloads.

    def __post_init__(self) -> None:
        self._sorted_times: tuple[float, ...] = ()
        self._warnings: tuple[str, ...] = ()
        self._median: float = -1.0
        self._mean: float = -1.0
        self._p25: float = -1.0
        self._p75: float = -1.0

    def __getattr__(self, name: str) -> Any:
        # Forward TaskSpec fields for convenience.
        if name in _TASKSPEC_FIELDS:
            return getattr(self.task_spec, name)
        return super().__getattribute__(name)

    # =========================================================================
    # == Convenience methods for statistics ===================================
    # =========================================================================
    #
    # These methods use raw time divided by number_per_run; this is an
    # extrapolation and hides the fact that different number_per_run will
    # result in different amortization of overheads, however if Timer has
    # selected an appropriate number_per_run then this is a non-issue, and
    # forcing users to handle that division would result in a poor experience.
    @property
    def times(self) -> list[float]:
        return [t / self.number_per_run for t in self.raw_times]

    @property
    def median(self) -> float:
        self._lazy_init()
        return self._median

    @property
    def mean(self) -> float:
        self._lazy_init()
        return self._mean

    @property
    def iqr(self) -> float:
        self._lazy_init()
        return self._p75 - self._p25

    @property
    def significant_figures(self) -> int:
        """Approximate significant figure estimate.

        This property is intended to give a convenient way to estimate the
        precision of a measurement. It only uses the interquartile region to
        estimate statistics to try to mitigate skew from the tails, and
        uses a static z value of 1.645 since it is not expected to be used
        for small values of `n`, so z can approximate `t`.

        The significant figure estimation used in conjunction with the
        `trim_sigfig` method to provide a more human interpretable data
        summary. __repr__ does not use this method; it simply displays raw
        values. Significant figure estimation is intended for `Compare`.
        """
        self._lazy_init()
        n_total = len(self._sorted_times)
        lower_bound = int(n_total // 4)
        upper_bound = int(torch.tensor(3 * n_total / 4).ceil())
        interquartile_points: tuple[float, ...] = self._sorted_times[lower_bound:upper_bound]
        std = torch.tensor(interquartile_points).std(unbiased=False).item()
        sqrt_n = torch.tensor(len(interquartile_points)).sqrt().item()

        # Rough estimates. These are by no means statistically rigorous.
        confidence_interval = max(1.645 * std / sqrt_n, _MIN_CONFIDENCE_INTERVAL)
        relative_ci = torch.tensor(self._median / confidence_interval).log10().item()
        num_significant_figures = int(torch.tensor(relative_ci).floor())
        return min(max(num_significant_figures, 1), _MAX_SIGNIFICANT_FIGURES)

    @property
    def has_warnings(self) -> bool:
        self._lazy_init()
        return bool(self._warnings)

    def _lazy_init(self) -> None:
        if self.raw_times and not self._sorted_times:
            self._sorted_times = tuple(sorted(self.times))
            _sorted_times = torch.tensor(self._sorted_times, dtype=torch.float64)
            self._median = _sorted_times.quantile(.5).item()
            self._mean = _sorted_times.mean().item()
            self._p25 = _sorted_times.quantile(.25).item()
            self._p75 = _sorted_times.quantile(.75).item()

            def add_warning(msg: str) -> None:
                rel_iqr = self.iqr / self.median * 100
                self._warnings += (
                    f"  WARNING: Interquartile range is {rel_iqr:.1f}% "
                    f"of the median measurement.\n           {msg}",
                )

            if not self.meets_confidence(_IQR_GROSS_WARN_THRESHOLD):
                add_warning("This suggests significant environmental influence.")
            elif not self.meets_confidence(_IQR_WARN_THRESHOLD):
                add_warning("This could indicate system fluctuation.")


    def meets_confidence(self, threshold: float = _IQR_WARN_THRESHOLD) -> bool:
        return self.iqr / self.median < threshold

    @property
    def title(self) -> str:
        return self.task_spec.title

    @property
    def env(self) -> str:
        return (
            "Unspecified env" if self.taskspec.env is None
            else cast(str, self.taskspec.env)
        )

    @property
    def as_row_name(self) -> str:
        return self.sub_label or self.stmt or "[Unknown]"

    def __repr__(self) -> str:
        """
        Example repr:
            <utils.common.Measurement object at 0x7f395b6ac110>
              Broadcasting add (4x8)
              Median: 5.73 us
              IQR:    2.25 us (4.01 to 6.26)
              372 measurements, 100 runs per measurement, 1 thread
              WARNING: Interquartile range is 39.4% of the median measurement.
                       This suggests significant environmental influence.
        """
        self._lazy_init()
        skip_line, newline = "MEASUREMENT_REPR_SKIP_LINE", "\n"
        n = len(self._sorted_times)
        time_unit, time_scale = select_unit(self._median)
        iqr_filter = '' if n >= 4 else skip_line

        repr_str = f"""
{super().__repr__()}
{self.task_spec.summarize()}
  {'Median: ' if n > 1 else ''}{self._median / time_scale:.2f} {time_unit}
  {iqr_filter}IQR:    {self.iqr / time_scale:.2f} {time_unit} ({self._p25 / time_scale:.2f} to {self._p75 / time_scale:.2f})
  {n} measurement{'s' if n > 1 else ''}, {self.number_per_run} runs {'per measurement,' if n > 1 else ','} {self.num_threads} thread{'s' if self.num_threads > 1 else ''}
{newline.join(self._warnings)}""".strip()

        return "\n".join(l for l in repr_str.splitlines(keepends=False) if skip_line not in l)

    @staticmethod
    def merge(measurements: Iterable["Measurement"]) -> list["Measurement"]:
        """Convenience method for merging replicates.

        Merge will extrapolate times to `number_per_run=1` and will not
        transfer any metadata. (Since it might differ between replicates)
        """
        grouped_measurements: collections.defaultdict[TaskSpec, list[Measurement]] = collections.defaultdict(list)
        for m in measurements:
            grouped_measurements[m.task_spec].append(m)

        def merge_group(task_spec: TaskSpec, group: list["Measurement"]) -> "Measurement":
            times: list[float] = []
            for m in group:
                # Different measurements could have different `number_per_run`,
                # so we call `.times` which normalizes the results.
                times.extend(m.times)

            return Measurement(
                number_per_run=1,
                raw_times=times,
                task_spec=task_spec,
                metadata=None,
            )

        return [merge_group(t, g) for t, g in grouped_measurements.items()]


def select_unit(t: float) -> tuple[str, float]:
    """Determine how to scale times for O(1) magnitude.

    This utility is used to format numbers for human consumption.
    """
    time_unit = {-3: "ns", -2: "us", -1: "ms"}.get(int(torch.tensor(t).log10().item() // 3), "s")
    time_scale = {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1}[time_unit]
    return time_unit, time_scale


def unit_to_english(u: str) -> str:
    return {
        "ns": "nanosecond",
        "us": "microsecond",
        "ms": "millisecond",
        "s": "second",
    }[u]


def trim_sigfig(x: float, n: int) -> float:
    """Trim `x` to `n` significant figures. (e.g. 3.14159, 2 -> 3.10000)"""
    if n != int(n):
        raise AssertionError("Number of significant figures must be an integer")
    magnitude = int(torch.tensor(x).abs().log10().ceil().item())
    scale = 10 ** (magnitude - n)
    return float(torch.tensor(x / scale).round() * scale)


def ordered_unique(elements: Iterable[Any]) -> list[Any]:
    return list(collections.OrderedDict(dict.fromkeys(elements)).keys())


@contextlib.contextmanager
def set_torch_threads(n: int) -> Iterator[None]:
    prior_num_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(n)
        yield
    finally:
        torch.set_num_threads(prior_num_threads)


def _make_temp_dir(prefix: str | None = None, gc_dev_shm: bool = False) -> str:
    """Create a temporary directory. The caller is responsible for cleanup.

    This function is conceptually similar to `tempfile.mkdtemp`, but with
    the key additional feature that it will use shared memory if the
    `BENCHMARK_USE_DEV_SHM` environment variable is set. This is an
    implementation detail, but an important one for cases where many Callgrind
    measurements are collected at once. (Such as when collecting
    microbenchmarks.)

    This is an internal utility, and is exported solely so that microbenchmarks
    can reuse the util.
    """
    use_dev_shm: bool = (os.getenv("BENCHMARK_USE_DEV_SHM") or "").lower() in ("1", "true")
    if use_dev_shm:
        root = "/dev/shm/pytorch_benchmark_utils"
        if os.name != "posix":
            raise AssertionError(f"tmpfs (/dev/shm) is POSIX only, current platform is {os.name}")
        if not os.path.exists("/dev/shm"):
            raise AssertionError("This system does not appear to support tmpfs (/dev/shm).")
        os.makedirs(root, exist_ok=True)

        # Because we're working in shared memory, it is more important than
        # usual to clean up ALL intermediate files. However we don't want every
        # worker to walk over all outstanding directories, so instead we only
        # check when we are sure that it won't lead to contention.
        if gc_dev_shm:
            for i in os.listdir(root):
                owner_file = os.path.join(root, i, "owner.pid")
                if not os.path.exists(owner_file):
                    continue

                with open(owner_file) as f:
                    owner_pid = int(f.read())

                if owner_pid == os.getpid():
                    continue

                try:
                    # https://stackoverflow.com/questions/568271/how-to-check-if-there-exists-a-process-with-a-given-pid-in-python
                    os.kill(owner_pid, 0)

                except OSError:
                    print(f"Detected that {os.path.join(root, i)} was orphaned in shared memory. Cleaning up.")
                    shutil.rmtree(os.path.join(root, i))

    else:
        root = tempfile.gettempdir()

    # We include the time so names sort by creation time, and add a UUID
    # to ensure we don't collide.
    name = f"{prefix or tempfile.gettempprefix()}__{int(time.time())}__{uuid.uuid4()}"
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=False)

    if use_dev_shm:
        with open(os.path.join(path, "owner.pid"), "w") as f:
            f.write(str(os.getpid()))

    return path
<|endoftext|>"""Timer class based on the timeit.Timer class, but torch aware."""
import enum
import timeit
import textwrap
from typing import overload, Any, NoReturn
from collections.abc import Callable

import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import TimerClass, TimeitModuleType
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface


__all__ = ["Timer", "timer", "Language"]


def timer() -> float:
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()
    return timeit.default_timer()


class Language(enum.Enum):
    PYTHON = 0
    CPP = 1


class CPPTimer:
    def __init__(
        self,
        stmt: str,
        setup: str,
        global_setup: str,
        timer: Callable[[], float],
        globals: dict[str, Any],
    ) -> None:
        if timer is not timeit.default_timer and torch.accelerator.is_available():
            raise NotImplementedError(
                "PyTorch was built with accelerators and an accelerator is present; however "
                "Timer does not yet support accelerator measurements. If your "
                "code is CPU only, pass `timer=timeit.default_timer` to the "
                "Timer's constructor to indicate this. (Note that this will "
                "produce incorrect results if an accelerator is in fact used, as "
                "Timer will not synchronize the accelerator.)"
            )

        if globals:
            raise ValueError("C++ timing does not support globals.")

        self._stmt: str = textwrap.dedent(stmt)
        self._setup: str = textwrap.dedent(setup)
        self._global_setup: str = textwrap.dedent(global_setup)
        self._timeit_module: TimeitModuleType | None = None

    def timeit(self, number: int) -> float:
        if self._timeit_module is None:
            self._timeit_module = cpp_jit.compile_timeit_template(
                stmt=self._stmt,
                setup=self._setup,
                global_setup=self._global_setup,
            )

        return self._timeit_module.timeit(number)


class Timer:
    """Helper class for measuring execution time of PyTorch statements.

    For a full tutorial on how to use this class, see:
    https://pytorch.org/tutorials/recipes/recipes/benchmark.html

    The PyTorch Timer is based on `timeit.Timer` (and in fact uses
    `timeit.Timer` internally), but with several key differences:

    1) Runtime aware:
        Timer will perform warmups (important as some elements of PyTorch are
        lazily initialized), set threadpool size so that comparisons are
        apples-to-apples, and synchronize asynchronous accelerator functions when
        necessary.

    2) Focus on replicates:
        When measuring code, and particularly complex kernels / models,
        run-to-run variation is a significant confounding factor. It is
        expected that all measurements should include replicates to quantify
        noise and allow median computation, which is more robust than mean.
        To that effect, this class deviates from the `timeit` API by
        conceptually merging `timeit.Timer.repeat` and `timeit.Timer.autorange`.
        (Exact algorithms are discussed in method docstrings.) The `timeit`
        method is replicated for cases where an adaptive strategy is not
        desired.

    3) Optional metadata:
        When defining a Timer, one can optionally specify `label`, `sub_label`,
        `description`, and `env`. (Defined later) These fields are included in
        the representation of result object and by the `Compare` class to group
        and display results for comparison.

    4) Instruction counts
        In addition to wall times, Timer can run a statement under Callgrind
        and report instructions executed.

    Directly analogous to `timeit.Timer` constructor arguments:

        `stmt`, `setup`, `timer`, `globals`

    PyTorch Timer specific constructor arguments:

        `label`, `sub_label`, `description`, `env`, `num_threads`

    Args:
        stmt: Code snippet to be run in a loop and timed.

        setup: Optional setup code. Used to define variables used in `stmt`

        global_setup: (C++ only)
            Code which is placed at the top level of the file for things like
            `#include` statements.

        timer:
            Callable which returns the current time. If PyTorch was built
            without accelerators or there is no accelerator present, this defaults to
            `timeit.default_timer`; otherwise it will synchronize accelerators before
            measuring the time.

        globals:
            A dict which defines the global variables when `stmt` is being
            executed. This is the other method for providing variables which
            `stmt` needs.

        label:
            String which summarizes `stmt`. For instance, if `stmt` is
            "torch.nn.functional.relu(torch.add(x, 1, out=out))"
            one might set label to "ReLU(x + 1)" to improve readability.

        sub_label:
            Provide supplemental information to disambiguate measurements
            with identical stmt or label. For instance, in our example
            above sub_label might be "float" or "int", so that it is easy
            to differentiate:
            "ReLU(x + 1): (float)"

            "ReLU(x + 1): (int)"
            when printing Measurements or summarizing using `Compare`.

        description:
            String to distinguish measurements with identical label and
            sub_label. The principal use of `description` is to signal to
            `Compare` the columns of data. For instance one might set it
            based on the input size  to create a table of the form: ::

                                        | n=1 | n=4 | ...
                                        ------------- ...
                ReLU(x + 1): (float)    | ... | ... | ...
                ReLU(x + 1): (int)      | ... | ... | ...


            using `Compare`. It is also included when printing a Measurement.

        env:
            This tag indicates that otherwise identical tasks were run in
            different environments, and are therefore not equivalent, for
            instance when A/B testing a change to a kernel. `Compare` will
            treat Measurements with different `env` specification as distinct
            when merging replicate runs.

        num_threads:
            The size of the PyTorch threadpool when executing `stmt`. Single
            threaded performance is important as both a key inference workload
            and a good indicator of intrinsic algorithmic efficiency, so the
            default is set to one. This is in contrast to the default PyTorch
            threadpool size which tries to utilize all cores.
    """

    _timer_cls: type[TimerClass] = timeit.Timer

    def __init__(
        self,
        stmt: str = "pass",
        setup: str = "pass",
        global_setup: str = "",
        timer: Callable[[], float] = timer,
        globals: dict[str, Any] | None = None,
        label: str | None = None,
        sub_label: str | None = None,
        description: str | None = None,
        env: str | None = None,
        num_threads: int = 1,
        language: Language | str = Language.PYTHON,
    ) -> None:
        if not isinstance(stmt, str):
            raise ValueError("Currently only a `str` stmt is supported.")

        # We copy `globals` to prevent mutations from leaking.
        # (For instance, `eval` adds the `__builtins__` key)
        self._globals = dict(globals or {})

        timer_kwargs = {}
        if language in (Language.PYTHON, "py", "python"):
            # Include `torch` if not specified as a convenience feature.
            self._globals.setdefault("torch", torch)
            self._language: Language = Language.PYTHON
            if global_setup:
                raise ValueError(
                    f"global_setup is C++ only, got `{global_setup}`. Most "
                    "likely this code can simply be moved to `setup`."
                )

        elif language in (Language.CPP, "cpp", "c++"):
            if self._timer_cls is not timeit.Timer:
                raise AssertionError("_timer_cls has already been swapped.")
            self._timer_cls = CPPTimer
            setup = ("" if setup == "pass" else setup)
            self._language = Language.CPP
            timer_kwargs["global_setup"] = global_setup

        else:
            raise ValueError(f"Invalid language `{language}`.")

        # Convenience adjustment so that multi-line code snippets defined in
        # functions do not IndentationError (Python) or look odd (C++). The
        # leading newline removal is for the initial newline that appears when
        # defining block strings. For instance:
        #   textwrap.dedent("""
        #     print("This is a stmt")
        #   """)
        # produces '\nprint("This is a stmt")\n'.
        #
        # Stripping this down to 'print("This is a stmt")' doesn't change
        # what gets executed, but it makes __repr__'s nicer.
        stmt = textwrap.dedent(stmt)
        stmt = (stmt[1:] if stmt and stmt[0] == "\n" else stmt).rstrip()
        setup = textwrap.dedent(setup)
        setup = (setup[1:] if setup and setup[0] == "\n" else setup).rstrip()


        self._timer = self._timer_cls(
            stmt=stmt,
            setup=setup,
            timer=timer,
            globals=valgrind_timer_interface.CopyIfCallgrind.unwrap_all(self._globals),
            **timer_kwargs,
        )
        self._task_spec = common.TaskSpec(
            stmt=stmt,
            setup=setup,
            global_setup=global_setup,
            label=label,
            sub_label=sub_label,
            description=description,
            env=env,
            num_threads=num_threads,
        )

    def _timeit(self, number: int) -> float:
        # Even calling a timer in C++ takes ~50 ns, so no real operation should
        # take less than 1 ns. (And this prevents divide by zero errors.)
        return max(self._timer.timeit(number), 1e-9)

    def timeit(self, number: int = 1000000) -> common.Measurement:
        """Mirrors the semantics of timeit.Timer.timeit().

        Execute the main statement (`stmt`) `number` times.
        https://docs.python.org/3/library/timeit.html#timeit.Timer.timeit
        """
        with common.set_torch_threads(self._task_spec.num_threads):
            # Warmup
            self._timeit(number=max(int(number // 100), 2))

            return common.Measurement(
                number_per_run=number,
                raw_times=[self._timeit(number=number)],
                task_spec=self._task_spec
            )

    def repeat(self, repeat: int = -1, number: int = -1) -> None:
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def autorange(self, callback: Callable[[int, float], NoReturn] | None = None) -> None:
        raise NotImplementedError("See `Timer.blocked_autorange.`")

    def _threaded_measurement_loop(
        self,
        number: int,
        time_hook: Callable[[], float],
        stop_hook: Callable[[list[float]], bool],
        min_run_time: float,
        max_run_time: float | None = None,
        callback: Callable[[int, float], NoReturn] | None = None
    ) -> list[float]:
        total_time = 0.0
        can_stop = False
        times: list[float] = []
        with common.set_torch_threads(self._task_spec.num_threads):
            while (total_time < min_run_time) or (not can_stop):
                time_spent = time_hook()
                times.append(time_spent)
                total_time += time_spent
                if callback:
                    callback(number, time_spent)
                can_stop = stop_hook(times)
                if max_run_time and total_time > max_run_time:
                    break
        return times

    def _estimate_block_size(self, min_run_time: float) -> int:
        with common.set_torch_threads(self._task_spec.num_threads):
            # Estimate the block size needed for measurement to be negligible
            # compared to the inner loop. This also serves as a warmup.
            overhead = torch.tensor([self._timeit(0) for _ in range(5)]).median().item()
            number = 1
            while True:
                time_taken = self._timeit(number)
                relative_overhead = overhead / time_taken
                if relative_overhead <= 1e-4 and time_taken >= min_run_time / 1000:
                    break
                if time_taken > min_run_time:
                    break
                # Avoid overflow in C++ pybind11 interface
                if number * 10 > 2147483647:
                    break
                number *= 10
        return number

    def blocked_autorange(
        self,
        callback: Callable[[int, float], NoReturn] | None = None,
        min_run_time: float = 0.2,
    ) -> common.Measurement:
        """Measure many replicates while keeping timer overhead to a minimum.

        At a high level, blocked_autorange executes the following pseudo-code::

            `setup`

            total_time = 0
            while total_time < min_run_time
                start = timer()
                for _ in range(block_size):
                    `stmt`
                total_time += (timer() - start)

        Note the variable `block_size` in the inner loop. The choice of block
        size is important to measurement quality, and must balance two
        competing objectives:

            1) A small block size results in more replicates and generally
               better statistics.

            2) A large block size better amortizes the cost of `timer`
               invocation, and results in a less biased measurement. This is
               important because accelerator synchronization time is non-trivial
               (order single to low double digit microseconds) and would
               otherwise bias the measurement.

        blocked_autorange sets block_size by running a warmup period,
        increasing block size until timer overhead is less than 0.1% of
        the overall computation. This value is then used for the main
        measurement loop.

        Returns:
            A `Measurement` object that contains measured runtimes and
            repetition counts, and can be used to compute statistics.
            (mean, median, etc.)
        """
        number = self._estimate_block_size(min_run_time)

        def time_hook() -> float:
            return self._timeit(number)

        def stop_hook(times: list[float]) -> bool:
            return True

        times = self._threaded_measurement_loop(
            number, time_hook, stop_hook,
            min_run_time=min_run_time,
            callback=callback)

        return common.Measurement(
            number_per_run=number,
            raw_times=times,
            task_spec=self._task_spec
        )

    def adaptive_autorange(
            self,
            threshold: float = 0.1,
            *,
            min_run_time: float = 0.01,
            max_run_time: float = 10.0,
            callback: Callable[[int, float], NoReturn] | None = None,
    ) -> common.Measurement:
        """Similar to `blocked_autorange` but also checks for variability in measurements
        and repeats until iqr/median is smaller than `threshold` or `max_run_time` is reached.


        At a high level, adaptive_autorange executes the following pseudo-code::

            `setup`

            times = []
            while times.sum < max_run_time
                start = timer()
                for _ in range(block_size):
                    `stmt`
                times.append(timer() - start)

                enough_data = len(times)>3 and times.sum > min_run_time
                small_iqr=times.iqr/times.mean<threshold

                if enough_data and small_iqr:
                    break

        Args:
            threshold: value of iqr/median threshold for stopping

            min_run_time: total runtime needed before checking `threshold`

            max_run_time: total runtime  for all measurements regardless of `threshold`

        Returns:
            A `Measurement` object that contains measured runtimes and
            repetition counts, and can be used to compute statistics.
            (mean, median, etc.)
        """
        number = self._estimate_block_size(min_run_time=0.05)

        def time_hook() -> float:
            return self._timeit(number)

        def stop_hook(times: list[float]) -> bool:
            if len(times) > 3:
                return common.Measurement(
                    number_per_run=number,
                    raw_times=times,
                    task_spec=self._task_spec
                ).meets_confidence(threshold=threshold)
            return False
        times = self._threaded_measurement_loop(
            number, time_hook, stop_hook, min_run_time, max_run_time, callback=callback)

        return common.Measurement(
            number_per_run=number,
            raw_times=times,
            task_spec=self._task_spec
        )

    @overload
    def collect_callgrind(
        self,
        number: int,
        *,
        repeats: None,
        collect_baseline: bool,
        retain_out_file: bool,
    ) -> valgrind_timer_interface.CallgrindStats:
        ...

    @overload
    def collect_callgrind(
        self,
        number: int,
        *,
        repeats: int,
        collect_baseline: bool,
        retain_out_file: bool,
    ) -> tuple[valgrind_timer_interface.CallgrindStats, ...]:
        ...

    def collect_callgrind(
        self,
        number: int = 100,
        *,
        repeats: int | None = None,
        collect_baseline: bool = True,
        retain_out_file: bool = False,
    ) -> Any:
        """Collect instruction counts using Callgrind.

        Unlike wall times, instruction counts are deterministic
        (modulo non-determinism in the program itself and small amounts of
        jitter from the Python interpreter.) This makes them ideal for detailed
        performance analysis. This method runs `stmt` in a separate process
        so that Valgrind can instrument the program. Performance is severely
        degraded due to the instrumentation, however this is ameliorated by
        the fact that a small number of iterations is generally sufficient to
        obtain good measurements.

        In order to use this method `valgrind`, `callgrind_control`, and
        `callgrind_annotate` must be installed.

        Because there is a process boundary between the caller (this process)
        and the `stmt` execution, `globals` cannot contain arbitrary in-memory
        data structures. (Unlike timing methods) Instead, globals are
        restricted to builtins, `nn.Modules`'s, and TorchScripted functions/modules
        to reduce the surprise factor from serialization and subsequent
        deserialization. The `GlobalsBridge` class provides more detail on this
        subject. Take particular care with nn.Modules: they rely on pickle and
        you may need to add an import to `setup` for them to transfer properly.

        By default, a profile for an empty statement will be collected and
        cached to indicate how many instructions are from the Python loop which
        drives `stmt`.

        Returns:
            A `CallgrindStats` object which provides instruction counts and
            some basic facilities for analyzing and manipulating results.
        """
        if not isinstance(self._task_spec.stmt, str):
            raise ValueError("`collect_callgrind` currently only supports string `stmt`")

        if repeats is not None and repeats < 1:
            raise ValueError("If specified, `repeats` must be >= 1")

        # Check that the statement is valid. It doesn't guarantee success, but it's much
        # simpler and quicker to raise an exception for a faulty `stmt` or `setup` in
        # the parent process rather than the valgrind subprocess.
        self._timeit(1)
        is_python = (self._language == Language.PYTHON)
        if not is_python and self._globals:
            raise AssertionError("_timer globals are only supported for Python timers")
        result = valgrind_timer_interface.wrapper_singleton().collect_callgrind(
            task_spec=self._task_spec,
            globals=self._globals,
            number=number,
            repeats=repeats or 1,
            collect_baseline=collect_baseline and is_python,
            is_python=is_python,
            retain_out_file=retain_out_file,
        )

        return (result[0] if repeats is None else result)
<|endoftext|>from typing import Any
from collections.abc import Callable
from typing_extensions import Protocol, runtime_checkable


class TimerClass(Protocol):
    """This is the portion of the `timeit.Timer` API used by benchmark utils."""
    def __init__(
        self,
        stmt: str,
        setup: str,
        timer: Callable[[], float],
        globals: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        ...

    def timeit(self, number: int) -> float:
        ...


@runtime_checkable
class TimeitModuleType(Protocol):
    """Modules generated from `timeit_template.cpp`."""
    def timeit(self, number: int) -> float:
        ...


class CallgrindModuleType(Protocol):
    """Replicates the valgrind endpoints in `torch._C`.

    These bindings are used to collect Callgrind profiles on earlier versions
    of PyTorch and will eventually be removed.
    """
    __file__: str
    __name__: str

    def _valgrind_supported_platform(self) -> bool:
        ...

    def _valgrind_toggle(self) -> None:
        ...
<|endoftext|># mypy: allow-untyped-defs
from numbers import Number
import torch
from torch.utils.benchmark import FuzzedTensor
import math

class FuzzedSparseTensor(FuzzedTensor):
    def __init__(
        self,
        name: str,
        size: tuple[str | int, ...],
        min_elements: int | None = None,
        max_elements: int | None = None,
        dim_parameter: str | None = None,
        sparse_dim: str | None = None,
        nnz: str | None = None,
        density: str | None = None,
        coalesced: str | None = None,
        dtype=torch.float32,
        cuda=False
    ) -> None:
        """
        Args:
            name:
                A string identifier for the generated Tensor.
            size:
                A tuple of integers or strings specifying the size of the generated
                Tensor. String values will replaced with a concrete int during the
                generation process, while ints are simply passed as literals.
            min_elements:
                The minimum number of parameters that this Tensor must have for a
                set of parameters to be valid. (Otherwise they are resampled.)
            max_elements:
                Like `min_elements`, but setting an upper bound.
            dim_parameter:
                The length of `size` will be truncated to this value.
                This allows Tensors of varying dimensions to be generated by the
                Fuzzer.
            sparse_dim:
                The number of sparse dimensions in a sparse tensor.
            density:
                This value allows tensors of varying sparsities to be generated by the Fuzzer.
            coalesced:
                The sparse tensor format permits uncoalesced sparse tensors,
                where there may be duplicate coordinates in the indices.
            dtype:
                The PyTorch dtype of the generated Tensor.
            cuda:
                Whether to place the Tensor on a GPU.
        """
        super().__init__(name=name, size=size, min_elements=min_elements,
                         max_elements=max_elements, dim_parameter=dim_parameter, dtype=dtype, cuda=cuda)
        self._density = density
        self._coalesced = coalesced
        self._sparse_dim = sparse_dim

    @staticmethod
    def sparse_tensor_constructor(size, dtype, sparse_dim, nnz, is_coalesced):
        """sparse_tensor_constructor creates a sparse tensor with coo format.

        Note that when `is_coalesced` is False, the number of elements is doubled but the number of indices
        represents the same amount of number of non zeros `nnz`, i.e, this is virtually the same tensor
        with the same sparsity pattern. Moreover, most of the sparse operation will use coalesce() method
        and what we want here is to get a sparse tensor with the same `nnz` even if this is coalesced or not.

        In the other hand when `is_coalesced` is True the number of elements is reduced in the coalescing process
        by an unclear amount however the probability to generate duplicates indices are low for most of the cases.
        This decision was taken on purpose to maintain the construction cost as low as possible.
        """
        if isinstance(size, Number):
            size = [size] * sparse_dim
        if all(size[d] <= 0 for d in range(sparse_dim)) and nnz != 0:
            raise AssertionError('invalid arguments')
        v_size = [nnz] + list(size[sparse_dim:])
        if dtype.is_floating_point:
            v = torch.rand(size=v_size, dtype=dtype, device="cpu")
        else:
            v = torch.randint(1, 127, size=v_size, dtype=dtype, device="cpu")

        i = torch.rand(sparse_dim, nnz, device="cpu")
        i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
        i = i.to(torch.long)

        if not is_coalesced:
            v = torch.cat([v, torch.randn_like(v)], 0)
            i = torch.cat([i, i], 1)

        x = torch.sparse_coo_tensor(i, v, torch.Size(size))
        if is_coalesced:
            x = x.coalesce()
        return x

    def _make_tensor(self, params, state):
        size, _, _ = self._get_size_and_steps(params)
        density = params['density']
        nnz = math.ceil(sum(size) * density)
        if nnz > sum(size):
            raise AssertionError('nnz cannot exceed total number of elements')

        is_coalesced = params['coalesced']
        sparse_dim = params['sparse_dim'] if self._sparse_dim else len(size)
        sparse_dim = min(sparse_dim, len(size))
        tensor = self.sparse_tensor_constructor(size, self._dtype, sparse_dim, nnz, is_coalesced)

        if self._cuda:
            tensor = tensor.cuda()
        sparse_dim = tensor.sparse_dim()
        dense_dim = tensor.dense_dim()
        is_hybrid = len(size[sparse_dim:]) > 0

        properties = {
            "numel": int(tensor.numel()),
            "shape": tensor.size(),
            "is_coalesced": tensor.is_coalesced(),
            "density": density,
            "sparsity": 1.0 - density,
            "sparse_dim": sparse_dim,
            "dense_dim": dense_dim,
            "is_hybrid": is_hybrid,
            "dtype": str(self._dtype),
        }
        return tensor, properties
<|endoftext|># mypy: allow-untyped-defs
"""Display class to aggregate and print the results of many measurements."""
import collections
import enum
import itertools as it

from torch.utils.benchmark.utils import common
from torch import tensor as _tensor
import operator

__all__ = ["Colorize", "Compare"]

BEST = "\033[92m"
GOOD = "\033[34m"
BAD = "\033[2m\033[91m"
VERY_BAD = "\033[31m"
BOLD = "\033[1m"
TERMINATE = "\033[0m"


class Colorize(enum.Enum):
    NONE = "none"
    COLUMNWISE = "columnwise"
    ROWWISE = "rowwise"


# Classes to separate internal bookkeeping from what is rendered.
class _Column:
    def __init__(
        self,
        grouped_results: list[tuple[common.Measurement | None, ...]],
        time_scale: float,
        time_unit: str,
        trim_significant_figures: bool,
        highlight_warnings: bool,
    ) -> None:
        self._grouped_results = grouped_results
        self._flat_results = [*it.chain.from_iterable(grouped_results)]
        self._time_scale = time_scale
        self._time_unit = time_unit
        self._trim_significant_figures = trim_significant_figures
        self._highlight_warnings = (
            highlight_warnings
            and any(r.has_warnings for r in self._flat_results if r)
        )
        leading_digits = [
            int(_tensor(r.median / self._time_scale).log10().ceil()) if r else None
            for r in self._flat_results
        ]
        unit_digits = max(d for d in leading_digits if d is not None)
        decimal_digits = min(
            max(m.significant_figures - digits, 0)
            for digits, m in zip(leading_digits, self._flat_results, strict=True)
            if (m is not None) and (digits is not None)
        ) if self._trim_significant_figures else 1
        length = unit_digits + decimal_digits + (1 if decimal_digits else 0)
        self._template = f"{{:>{length}.{decimal_digits}f}}{{:>{7 if self._highlight_warnings else 0}}}"

    def get_results_for(self, group):
        return self._grouped_results[group]

    def num_to_str(self, value: float | None, estimated_sigfigs: int, spread: float | None):
        if value is None:
            return " " * len(self.num_to_str(1, estimated_sigfigs, None))

        if self._trim_significant_figures:
            value = common.trim_sigfig(value, estimated_sigfigs)

        return self._template.format(
            value,
            f" (! {spread * 100:.0f}%)" if self._highlight_warnings and spread is not None else "")


def optional_min(seq):
    l = list(seq)
    return None if len(l) == 0 else min(l)


class _Row:
    def __init__(self, results, row_group, render_env, env_str_len,
                 row_name_str_len, time_scale, colorize, num_threads=None) -> None:
        super().__init__()
        self._results = results
        self._row_group = row_group
        self._render_env = render_env
        self._env_str_len = env_str_len
        self._row_name_str_len = row_name_str_len
        self._time_scale = time_scale
        self._colorize = colorize
        self._columns: tuple[_Column, ...] = ()
        self._num_threads = num_threads

    def register_columns(self, columns: tuple[_Column, ...]) -> None:
        self._columns = columns

    def as_column_strings(self):
        concrete_results = [r for r in self._results if r is not None]
        env = f"({concrete_results[0].env})" if self._render_env else ""
        env = env.ljust(self._env_str_len + 4)
        output = ["  " + env + concrete_results[0].as_row_name]
        for m, col in zip(self._results, self._columns or (), strict=False):
            if m is None:
                output.append(col.num_to_str(None, 1, None))
            else:
                output.append(col.num_to_str(
                    m.median / self._time_scale,
                    m.significant_figures,
                    m.iqr / m.median if m.has_warnings else None
                ))
        return output

    @staticmethod
    def color_segment(segment, value, best_value):
        if value <= best_value * 1.01 or value <= best_value + 100e-9:
            return BEST + BOLD + segment + TERMINATE * 2
        if value <= best_value * 1.1:
            return GOOD + BOLD + segment + TERMINATE * 2
        if value >= best_value * 5:
            return VERY_BAD + BOLD + segment + TERMINATE * 2
        if value >= best_value * 2:
            return BAD + segment + TERMINATE * 2

        return segment

    def row_separator(self, overall_width):
        return (
            [f"{self._num_threads} threads: ".ljust(overall_width, "-")]
            if self._num_threads is not None else []
        )

    def finalize_column_strings(self, column_strings, col_widths):
        best_values = [-1 for _ in column_strings]
        if self._colorize == Colorize.ROWWISE:
            row_min = min(r.median for r in self._results if r is not None)
            best_values = [row_min for _ in column_strings]
        elif self._colorize == Colorize.COLUMNWISE:
            best_values = [
                optional_min(r.median for r in column.get_results_for(self._row_group) if r is not None)
                for column in (self._columns or ())
            ]

        row_contents = [column_strings[0].ljust(col_widths[0])]
        for col_str, width, result, best_value in zip(column_strings[1:], col_widths[1:], self._results, best_values, strict=False):
            col_str = col_str.center(width)
            if self._colorize != Colorize.NONE and result is not None and best_value is not None:
                col_str = self.color_segment(col_str, result.median, best_value)
            row_contents.append(col_str)
        return row_contents


class Table:
    def __init__(
            self,
            results: list[common.Measurement],
            colorize: Colorize,
            trim_significant_figures: bool,
            highlight_warnings: bool
    ) -> None:
        if len({r.label for r in results}) != 1:
            raise AssertionError("All results must share the same label")

        self.results = results
        self._colorize = colorize
        self._trim_significant_figures = trim_significant_figures
        self._highlight_warnings = highlight_warnings
        self.label = results[0].label
        self.time_unit, self.time_scale = common.select_unit(
            min(r.median for r in results)
        )

        self.row_keys = common.ordered_unique([self.row_fn(i) for i in results])
        self.row_keys.sort(key=operator.itemgetter(slice(2)))  # preserve stmt order
        self.column_keys = common.ordered_unique([self.col_fn(i) for i in results])
        self.rows, self.columns = self.populate_rows_and_columns()

    @staticmethod
    def row_fn(m: common.Measurement) -> tuple[int, str | None, str]:
        return m.num_threads, m.env, m.as_row_name

    @staticmethod
    def col_fn(m: common.Measurement) -> str | None:
        return m.description

    def populate_rows_and_columns(self) -> tuple[tuple[_Row, ...], tuple[_Column, ...]]:
        rows: list[_Row] = []
        columns: list[_Column] = []
        ordered_results: list[list[common.Measurement | None]] = [
            [None for _ in self.column_keys]
            for _ in self.row_keys
        ]
        row_position = {key: i for i, key in enumerate(self.row_keys)}
        col_position = {key: i for i, key in enumerate(self.column_keys)}
        for r in self.results:
            i = row_position[self.row_fn(r)]
            j = col_position[self.col_fn(r)]
            ordered_results[i][j] = r

        unique_envs = {r.env for r in self.results}
        render_env = len(unique_envs) > 1
        env_str_len = max(len(i) for i in unique_envs) if render_env else 0

        row_name_str_len = max(len(r.as_row_name) for r in self.results)

        prior_num_threads = -1
        prior_env = ""
        row_group = -1
        rows_by_group: list[list[list[common.Measurement | None]]] = []
        for (num_threads, env, _), row in zip(self.row_keys, ordered_results, strict=True):
            thread_transition = (num_threads != prior_num_threads)
            if thread_transition:
                prior_num_threads = num_threads
                prior_env = ""
                row_group += 1
                rows_by_group.append([])
            rows.append(
                _Row(
                    results=row,
                    row_group=row_group,
                    render_env=(render_env and env != prior_env),
                    env_str_len=env_str_len,
                    row_name_str_len=row_name_str_len,
                    time_scale=self.time_scale,
                    colorize=self._colorize,
                    num_threads=num_threads if thread_transition else None,
                )
            )
            rows_by_group[-1].append(row)
            prior_env = env

        for i in range(len(self.column_keys)):
            grouped_results = [tuple(row[i] for row in g) for g in rows_by_group]
            column = _Column(
                grouped_results=grouped_results,
                time_scale=self.time_scale,
                time_unit=self.time_unit,
                trim_significant_figures=self._trim_significant_figures,
                highlight_warnings=self._highlight_warnings,)
            columns.append(column)

        rows_tuple, columns_tuple = tuple(rows), tuple(columns)
        for ri in rows_tuple:
            ri.register_columns(columns_tuple)
        return rows_tuple, columns_tuple

    def render(self) -> str:
        string_rows = [[""] + self.column_keys]
        string_rows.extend(r.as_column_strings() for r in self.rows)
        num_cols = max(len(i) for i in string_rows)
        for sr in string_rows:
            sr.extend(["" for _ in range(num_cols - len(sr))])

        col_widths = [max(len(j) for j in i) for i in zip(*string_rows, strict=True)]
        finalized_columns = ["  |  ".join(i.center(w) for i, w in zip(string_rows[0], col_widths, strict=True))]
        overall_width = len(finalized_columns[0])
        for string_row, row in zip(string_rows[1:], self.rows, strict=True):
            finalized_columns.extend(row.row_separator(overall_width))
            finalized_columns.append("  |  ".join(row.finalize_column_strings(string_row, col_widths)))

        newline = "\n"
        has_warnings = self._highlight_warnings and any(ri.has_warnings for ri in self.results)
        return f"""
[{(' ' + (self.label or '') + ' ').center(overall_width - 2, '-')}]
{newline.join(finalized_columns)}

Times are in {common.unit_to_english(self.time_unit)}s ({self.time_unit}).
{'(! XX%) Measurement has high variance, where XX is the IQR / median * 100.' + newline if has_warnings else ""}"""[1:]


class Compare:
    """Helper class for displaying the results of many measurements in a
    formatted table.

    The table format is based on the information fields provided in
    :class:`torch.utils.benchmark.Timer` (`description`, `label`, `sub_label`,
    `num_threads`, etc).

    The table can be directly printed using :meth:`print` or casted as a `str`.

    For a full tutorial on how to use this class, see:
    https://pytorch.org/tutorials/recipes/recipes/benchmark.html

    Args:
        results: List of Measurement to display.
    """
    def __init__(self, results: list[common.Measurement]) -> None:
        self._results: list[common.Measurement] = []
        self.extend_results(results)
        self._trim_significant_figures = False
        self._colorize = Colorize.NONE
        self._highlight_warnings = False

    def __str__(self) -> str:
        return "\n".join(self._render())

    def extend_results(self, results) -> None:
        """Append results to already stored ones.

        All added results must be instances of ``Measurement``.
        """
        for r in results:
            if not isinstance(r, common.Measurement):
                raise ValueError(
                    "Expected an instance of `Measurement`, " f"got {type(r)} instead."
                )
        self._results.extend(results)

    def trim_significant_figures(self) -> None:
        """Enables trimming of significant figures when building the formatted table."""
        self._trim_significant_figures = True

    def colorize(self, rowwise=False) -> None:
        """Colorize formatted table.

        Colorize columnwise by default.
        """
        self._colorize = Colorize.ROWWISE if rowwise else Colorize.COLUMNWISE

    def highlight_warnings(self) -> None:
        """Enables warning highlighting when building formatted table."""
        self._highlight_warnings = True

    def print(self) -> None:
        """Print formatted table"""
        print(str(self))

    def _render(self):
        results = common.Measurement.merge(self._results)
        grouped_results = self._group_by_label(results)
        output = [self._layout(group) for group in grouped_results.values()]
        return output

    def _group_by_label(self, results: list[common.Measurement]):
        grouped_results: collections.defaultdict[str, list[common.Measurement]] = collections.defaultdict(list)
        for r in results:
            grouped_results[r.label].append(r)
        return grouped_results

    def _layout(self, results: list[common.Measurement]):
        table = Table(
            results,
            self._colorize,
            self._trim_significant_figures,
            self._highlight_warnings
        )
        return table.render()
<|endoftext|># mypy: allow-untyped-defs
import functools
import itertools as it
from typing import Any
from collections.abc import Callable

import torch


__all__ = [
    "Fuzzer",
    "FuzzedParameter", "ParameterAlias",
    "FuzzedTensor",
]


_DISTRIBUTIONS = (
    "loguniform",
    "uniform",
)


class FuzzedParameter:
    """Specification for a parameter to be generated during fuzzing."""
    def __init__(
        self,
        name: str,
        minval: int | float | None = None,
        maxval: int | float | None = None,
        distribution: str | dict[Any, float] | None = None,
        strict: bool = False,
    ) -> None:
        """
        Args:
            name:
                A string name with which to identify the parameter.
                FuzzedTensors can reference this string in their
                specifications.
            minval:
                The lower bound for the generated value. See the description
                of `distribution` for type behavior.
            maxval:
                The upper bound for the generated value. Type behavior is
                identical to `minval`.
            distribution:
                Specifies the distribution from which this parameter should
                be drawn. There are three possibilities:
                    - "loguniform"
                        Samples between `minval` and `maxval` (inclusive) such
                        that the probabilities are uniform in log space. As a
                        concrete example, if minval=1 and maxval=100, a sample
                        is as likely to fall in [1, 10) as it is [10, 100].
                    - "uniform"
                        Samples are chosen with uniform probability between
                        `minval` and `maxval` (inclusive). If either `minval`
                        or `maxval` is a float then the distribution is the
                        continuous uniform distribution; otherwise samples
                        are constrained to the integers.
                    - dict:
                        If a dict is passed, the keys are taken to be choices
                        for the variables and the values are interpreted as
                        probabilities. (And must sum to one.)
                If a dict is passed, `minval` and `maxval` must not be set.
                Otherwise, they must be set.
            strict:
                If a parameter is strict, it will not be included in the
                iterative resampling process which Fuzzer uses to find a
                valid parameter configuration. This allows an author to
                prevent skew from resampling for a given parameter (for
                instance, a low size limit could inadvertently bias towards
                Tensors with fewer dimensions) at the cost of more iterations
                when generating parameters.
        """
        self._name = name
        self._minval = minval
        self._maxval = maxval
        self._distribution = self._check_distribution(distribution)
        self.strict = strict

    @property
    def name(self):
        return self._name

    def sample(self, state):
        if self._distribution == "loguniform":
            return self._loguniform(state)

        if self._distribution == "uniform":
            return self._uniform(state)

        if isinstance(self._distribution, dict):
            return self._custom_distribution(state)

    def _check_distribution(self, distribution):
        if not isinstance(distribution, dict):
            if distribution not in _DISTRIBUTIONS:
                raise AssertionError(f"Unknown distribution: {distribution}")
        else:
            if any(i < 0 for i in distribution.values()):
                raise AssertionError("Probabilities cannot be negative")
            if not abs(sum(distribution.values()) - 1) > 1e-5:
                raise AssertionError("Distribution is not normalized")
            if self._minval is not None:
                raise AssertionError("When passing a custom distribution, 'minval' must be None")
            if self._maxval is not None:
                raise AssertionError("When passing a custom distribution, 'maxval' must be None")

        return distribution

    def _loguniform(self, state):
        import numpy as np
        output = int(2 ** state.uniform(
            low=np.log2(self._minval) if self._minval is not None else None,
            high=np.log2(self._maxval) if self._maxval is not None else None,
        ))
        if self._minval is not None and output < self._minval:
            return self._minval
        if self._maxval is not None and output > self._maxval:
            return self._maxval
        return output

    def _uniform(self, state):
        if isinstance(self._minval, int) and isinstance(self._maxval, int):
            return int(state.randint(low=self._minval, high=self._maxval + 1))
        return state.uniform(low=self._minval, high=self._maxval)

    def _custom_distribution(self, state):
        import numpy as np
        # If we directly pass the keys to `choice`, numpy will convert
        # them to numpy dtypes.
        index = state.choice(
            np.arange(len(self._distribution)),
            p=tuple(self._distribution.values()))
        return list(self._distribution.keys())[index]


class ParameterAlias:
    """Indicates that a parameter should alias the value of another parameter.

    When used in conjunction with a custom distribution, this allows fuzzed
    tensors to represent a broader range of behaviors. For example, the
    following sometimes produces Tensors which broadcast:

    Fuzzer(
        parameters=[
            FuzzedParameter("x_len", 4, 1024, distribution="uniform"),

            # `y` will either be size one, or match the size of `x`.
            FuzzedParameter("y_len", distribution={
                0.5: 1,
                0.5: ParameterAlias("x_len")
            }),
        ],
        tensors=[
            FuzzedTensor("x", size=("x_len",)),
            FuzzedTensor("y", size=("y_len",)),
        ],
    )

    Chains of alias' are allowed, but may not contain cycles.
    """
    def __init__(self, alias_to) -> None:
        self.alias_to = alias_to

    def __repr__(self) -> str:
        return f"ParameterAlias[alias_to: {self.alias_to}]"


def dtype_size(dtype):
    if dtype == torch.bool:
        return 1
    if dtype.is_floating_point or dtype.is_complex:
        return int(torch.finfo(dtype).bits / 8)
    return int(torch.iinfo(dtype).bits / 8)


def prod(values, base=1):
    """np.prod can overflow, so for sizes the product should be done in Python.

    Even though np.prod type promotes to int64, it can still overflow in which
    case the negative value will pass the size check and OOM when attempting to
    actually allocate the Tensor.
    """
    return functools.reduce(lambda x, y: int(x) * int(y), values, base)


class FuzzedTensor:
    def __init__(
        self,
        name: str,
        size: tuple[str | int, ...],
        steps: tuple[str | int, ...] | None = None,
        probability_contiguous: float = 0.5,
        min_elements: int | None = None,
        max_elements: int | None = None,
        max_allocation_bytes: int | None = None,
        dim_parameter: str | None = None,
        roll_parameter: str | None = None,
        dtype=torch.float32,
        cuda=False,
        tensor_constructor: Callable | None = None
    ) -> None:
        """
        Args:
            name:
                A string identifier for the generated Tensor.
            size:
                A tuple of integers or strings specifying the size of the generated
                Tensor. String values will replaced with a concrete int during the
                generation process, while ints are simply passed as literals.
            steps:
                An optional tuple with the same length as `size`. This indicates
                that a larger Tensor should be allocated, and then sliced to
                produce the generated Tensor. For instance, if size is (4, 8)
                and steps is (1, 4), then a tensor `t` of size (4, 32) will be
                created and then `t[:, ::4]` will be used. (Allowing one to test
                Tensors with strided memory.)
            probability_contiguous:
                A number between zero and one representing the chance that the
                generated Tensor has a contiguous memory layout. This is achieved by
                randomly permuting the shape of a Tensor, calling `.contiguous()`,
                and then permuting back. This is applied before `steps`, which can
                also cause a Tensor to be non-contiguous.
            min_elements:
                The minimum number of parameters that this Tensor must have for a
                set of parameters to be valid. (Otherwise they are resampled.)
            max_elements:
                Like `min_elements`, but setting an upper bound.
            max_allocation_bytes:
                Like `max_elements`, but for the size of Tensor that must be
                allocated prior to slicing for `steps` (if applicable). For
                example, a FloatTensor with size (1024, 1024) and steps (4, 4)
                would have 1M elements, but would require a 64 MB allocation.
            dim_parameter:
                The length of `size` and `steps` will be truncated to this value.
                This allows Tensors of varying dimensions to be generated by the
                Fuzzer.
            dtype:
                The PyTorch dtype of the generated Tensor.
            cuda:
                Whether to place the Tensor on a GPU.
            tensor_constructor:
                Callable which will be used instead of the default Tensor
                construction method. This allows the author to enforce properties
                of the Tensor (e.g. it can only have certain values). The dtype and
                concrete shape of the Tensor to be created will be passed, and
                concrete values of all parameters will be passed as kwargs. Note
                that transformations to the result (permuting, slicing) will be
                performed by the Fuzzer; the tensor_constructor is only responsible
                for creating an appropriately sized Tensor.
        """
        self._name = name
        self._size = size
        self._steps = steps
        self._probability_contiguous = probability_contiguous
        self._min_elements = min_elements
        self._max_elements = max_elements
        self._max_allocation_bytes = max_allocation_bytes
        self._dim_parameter = dim_parameter
        self._dtype = dtype
        self._cuda = cuda
        self._tensor_constructor = tensor_constructor

    @property
    def name(self):
        return self._name

    @staticmethod
    def default_tensor_constructor(size, dtype, **kwargs):
        if dtype.is_floating_point or dtype.is_complex:
            return torch.rand(size=size, dtype=dtype, device="cpu")
        else:
            return torch.randint(1, 127, size=size, dtype=dtype, device="cpu")

    def _make_tensor(self, params, state):
        import numpy as np
        size, steps, allocation_size = self._get_size_and_steps(params)
        constructor = (
            self._tensor_constructor or
            self.default_tensor_constructor
        )

        raw_tensor = constructor(size=allocation_size, dtype=self._dtype, **params)
        if self._cuda:
            raw_tensor = raw_tensor.cuda()

        # Randomly permute the Tensor and call `.contiguous()` to force re-ordering
        # of the memory, and then permute it back to the original shape.
        dim = len(size)
        order = np.arange(dim)
        if state.rand() > self._probability_contiguous:
            while dim > 1 and np.all(order == np.arange(dim)):
                order = state.permutation(raw_tensor.dim())

            raw_tensor = raw_tensor.permute(tuple(order)).contiguous()
            raw_tensor = raw_tensor.permute(tuple(np.argsort(order)))

        slices = [slice(0, size * step, step) for size, step in zip(size, steps, strict=True)]
        tensor = raw_tensor[tuple(slices)]

        properties = {
            "numel": int(tensor.numel()),
            "order": order,
            "steps": steps,
            "is_contiguous": tensor.is_contiguous(),
            "dtype": str(self._dtype),
        }

        return tensor, properties

    def _get_size_and_steps(self, params):
        dim = (
            params[self._dim_parameter]
            if self._dim_parameter is not None
            else len(self._size)
        )

        def resolve(values, dim):
            """Resolve values into concrete integers."""
            values = tuple(params.get(i, i) for i in values)
            if len(values) > dim:
                values = values[:dim]
            if len(values) < dim:
                values = values + tuple(1 for _ in range(dim - len(values)))
            return values

        size = resolve(self._size, dim)
        steps = resolve(self._steps or (), dim)
        allocation_size = tuple(size_i * step_i for size_i, step_i in zip(size, steps, strict=True))
        return size, steps, allocation_size

    def satisfies_constraints(self, params) -> bool:
        size, _, allocation_size = self._get_size_and_steps(params)
        # Product is computed in Python to avoid integer overflow.
        num_elements = prod(size)
        if num_elements < 0:
            raise AssertionError("Computed number of elements is negative")

        allocation_bytes = prod(allocation_size, base=dtype_size(self._dtype))

        def nullable_greater(left, right):
            if left is None or right is None:
                return False
            return left > right

        return not any((
            nullable_greater(num_elements, self._max_elements),
            nullable_greater(self._min_elements, num_elements),
            nullable_greater(allocation_bytes, self._max_allocation_bytes),
        ))


class Fuzzer:
    def __init__(
        self,
        parameters: list[FuzzedParameter | list[FuzzedParameter]],
        tensors: list[FuzzedTensor | list[FuzzedTensor]],
        constraints: list[Callable] | None = None,
        seed: int | None = None
    ) -> None:
        """
        Args:
            parameters:
                List of FuzzedParameters which provide specifications
                for generated parameters. Iterable elements will be
                unpacked, though arbitrary nested structures will not.
            tensors:
                List of FuzzedTensors which define the Tensors which
                will be created each step based on the parameters for
                that step. Iterable elements will be unpacked, though
                arbitrary nested structures will not.
            constraints:
                List of callables. They will be called with params
                as kwargs, and if any of them return False the current
                set of parameters will be rejected.
            seed:
                Seed for the RandomState used by the Fuzzer. This will
                also be used to set the PyTorch random seed so that random
                ops will create reproducible Tensors.
        """
        import numpy as np
        if seed is None:
            seed = int(np.random.RandomState().randint(0, 2 ** 32 - 1, dtype=np.int64))
        self._seed = seed
        self._parameters = Fuzzer._unpack(parameters, FuzzedParameter)
        self._tensors = Fuzzer._unpack(tensors, FuzzedTensor)
        self._constraints = constraints or ()

        p_names = {p.name for p in self._parameters}
        t_names = {t.name for t in self._tensors}
        name_overlap = p_names.intersection(t_names)
        if name_overlap:
            raise ValueError(f"Duplicate names in parameters and tensors: {name_overlap}")

        self._rejections = 0
        self._total_generated = 0

    @staticmethod
    def _unpack(values, cls):
        return tuple(it.chain.from_iterable(
            [[i] if isinstance(i, cls) else i for i in values]
        ))

    def take(self, n):
        import numpy as np
        state = np.random.RandomState(self._seed)
        torch.manual_seed(state.randint(low=0, high=2 ** 63, dtype=np.int64))
        for _ in range(n):
            params = self._generate(state)
            tensors = {}
            tensor_properties = {}
            for t in self._tensors:
                tensor, properties = t._make_tensor(params, state)
                tensors[t.name] = tensor
                tensor_properties[t.name] = properties
            yield tensors, tensor_properties, params

    @property
    def rejection_rate(self):
        if not self._total_generated:
            return 0.
        return self._rejections / self._total_generated

    def _generate(self, state):
        strict_params: dict[str, float | int | ParameterAlias] = {}
        for _ in range(1000):
            candidate_params: dict[str, float | int | ParameterAlias] = {}
            for p in self._parameters:
                if p.strict:
                    if p.name in strict_params:
                        candidate_params[p.name] = strict_params[p.name]
                    else:
                        candidate_params[p.name] = p.sample(state)
                        strict_params[p.name] = candidate_params[p.name]
                else:
                    candidate_params[p.name] = p.sample(state)

            candidate_params = self._resolve_aliases(candidate_params)

            self._total_generated += 1
            if not all(f(candidate_params) for f in self._constraints):
                self._rejections += 1
                continue

            if not all(t.satisfies_constraints(candidate_params) for t in self._tensors):
                self._rejections += 1
                continue

            return candidate_params
        raise ValueError("Failed to generate a set of valid parameters.")

    @staticmethod
    def _resolve_aliases(params):
        params = dict(params)
        alias_count = sum(isinstance(v, ParameterAlias) for v in params.values())

        keys = list(params.keys())
        while alias_count:
            for k in keys:
                v = params[k]
                if isinstance(v, ParameterAlias):
                    params[k] = params[v.alias_to]
            alias_count_new = sum(isinstance(v, ParameterAlias) for v in params.values())
            if alias_count == alias_count_new:
                raise ValueError(f"ParameterAlias cycle detected\n{params}")

            alias_count = alias_count_new

        return params
<|endoftext|>"""Intermediate layer between `Timer` and `valgrind`."""
from __future__ import annotations

import collections
import enum
import dataclasses
import itertools as it
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
from typing import (
    cast, Any, NamedTuple,
    TYPE_CHECKING)

import torch
from torch.utils.benchmark.utils import common, cpp_jit
import operator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from torch.utils.benchmark.utils._stubs import CallgrindModuleType


__all__ = ["FunctionCount", "FunctionCounts", "CallgrindStats", "CopyIfCallgrind"]


if TYPE_CHECKING:
    CompletedProcessType = subprocess.CompletedProcess[str]
else:
    CompletedProcessType = subprocess.CompletedProcess


class FunctionCount(NamedTuple):
    # TODO(#105471): Rename the count field
    count: int  # type: ignore[assignment]
    function: str


@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class FunctionCounts:
    """Container for manipulating Callgrind results.

    It supports:
        1) Addition and subtraction to combine or diff results.
        2) Tuple-like indexing.
        3) A `denoise` function which strips CPython calls which are known to
           be non-deterministic and quite noisy.
        4) Two higher order methods (`filter` and `transform`) for custom
           manipulation.
    """
    _data: tuple[FunctionCount, ...]
    inclusive: bool
    truncate_rows: bool = True

    # For normal use, torch._tensor_str.PRINT_OPTS.linewidth determines
    # the print settings. This is simply to allow hermetic unit tests.
    _linewidth: int | None = None

    def __iter__(self) -> Iterator[FunctionCount]:
        yield from self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: Any) -> FunctionCount | FunctionCounts:
        data: FunctionCount | tuple[FunctionCount, ...] = self._data[item]
        return (
            FunctionCounts(cast(tuple[FunctionCount, ...], data), self.inclusive, truncate_rows=False)
            if isinstance(data, tuple) else data
        )

    def __repr__(self) -> str:
        count_len = 0
        for c, _ in self:
            # Account for sign in string length.
            count_len = max(count_len, len(str(c)) + int(c < 0))

        lines = []
        linewidth = self._linewidth or torch._tensor_str.PRINT_OPTS.linewidth
        fn_str_len = max(linewidth - count_len - 4, 40)
        for c, fn in self:
            if len(fn) > fn_str_len:
                left_len = int((fn_str_len - 5) // 2)
                fn = fn[:left_len] + " ... " + fn[-(fn_str_len - left_len - 5):]
            lines.append(f"  {c:>{count_len}}  {fn}")

        if self.truncate_rows and len(lines) > 18:
            lines = lines[:9] + ["...".rjust(count_len + 2)] + lines[-9:]

        if not self.inclusive:
            lines.extend(["", f"Total: {self.sum()}"])

        return "\n".join([super().__repr__()] + lines)

    def __add__(
        self,
        other: FunctionCounts,
    ) -> FunctionCounts:
        return self._merge(other, lambda c: c)

    def __sub__(
        self,
        other: FunctionCounts,
    ) -> FunctionCounts:
        return self._merge(other, operator.neg)

    def __mul__(self, other: int | float) -> FunctionCounts:
        return self._from_dict({
            fn: int(c * other) for c, fn in self._data
        }, self.inclusive)

    def transform(self, map_fn: Callable[[str], str]) -> FunctionCounts:
        """Apply `map_fn` to all of the function names.

        This can be used to regularize function names (e.g. stripping irrelevant
        parts of the file path), coalesce entries by mapping multiple functions
        to the same name (in which case the counts are added together), etc.
        """
        counts: collections.defaultdict[str, int] = collections.defaultdict(int)
        for c, fn in self._data:
            counts[map_fn(fn)] += c

        return self._from_dict(counts, self.inclusive)

    def filter(self, filter_fn: Callable[[str], bool]) -> FunctionCounts:
        """Keep only the elements where `filter_fn` applied to function name returns True."""
        return FunctionCounts(tuple(i for i in self if filter_fn(i.function)), self.inclusive)

    def sum(self) -> int:
        return sum(c for c, _ in self)

    def denoise(self) -> FunctionCounts:
        """Remove known noisy instructions.

        Several instructions in the CPython interpreter are rather noisy. These
        instructions involve unicode to dictionary lookups which Python uses to
        map variable names. FunctionCounts is generally a content agnostic
        container, however this is sufficiently important for obtaining
        reliable results to warrant an exception."""
        return self.filter(lambda fn: "dictobject.c:lookdict_unicode" not in fn)

    def _merge(
        self,
        second: FunctionCounts,
        merge_fn: Callable[[int], int]
    ) -> FunctionCounts:
        if self.inclusive != second.inclusive:
            raise AssertionError("Cannot merge inclusive and exclusive counts.")
        counts: collections.defaultdict[str, int] = collections.defaultdict(int)
        for c, fn in self:
            counts[fn] += c

        for c, fn in second:
            counts[fn] += merge_fn(c)

        return self._from_dict(counts, self.inclusive)

    @staticmethod
    def _from_dict(counts: dict[str, int], inclusive: bool) -> FunctionCounts:
        flat_counts = (FunctionCount(c, fn) for fn, c in counts.items() if c)
        return FunctionCounts(tuple(sorted(flat_counts, reverse=True)), inclusive)


@dataclasses.dataclass(repr=False, eq=False, frozen=True)
class CallgrindStats:
    """Top level container for Callgrind results collected by Timer.

    Manipulation is generally done using the FunctionCounts class, which is
    obtained by calling `CallgrindStats.stats(...)`. Several convenience
    methods are provided as well; the most significant is
    `CallgrindStats.as_standardized()`.
    """
    task_spec: common.TaskSpec
    number_per_run: int
    built_with_debug_symbols: bool
    baseline_inclusive_stats: FunctionCounts
    baseline_exclusive_stats: FunctionCounts
    stmt_inclusive_stats: FunctionCounts
    stmt_exclusive_stats: FunctionCounts
    stmt_callgrind_out: str | None

    def __repr__(self) -> str:
        base_stats = self.baseline_exclusive_stats
        output = f"""
{super().__repr__()}
{self.task_spec.summarize()}
  {'':>25}All{'':>10}Noisy symbols removed
    Instructions: {self.counts(denoise=False):>12}{'':>15}{self.counts(denoise=True):>12}
    Baseline:     {base_stats.sum():>12}{'':>15}{base_stats.denoise().sum():>12}
{self.number_per_run} runs per measurement, {self.task_spec.num_threads} thread{'s' if self.task_spec.num_threads > 1 else ''}
""".strip()
        if not self.built_with_debug_symbols:
            output += textwrap.dedent("""
            Warning: PyTorch was not built with debug symbols.
                     Source information may be limited. Rebuild with
                     REL_WITH_DEB_INFO=1 for more detailed results.""")
        return output

    def stats(self, inclusive: bool = False) -> FunctionCounts:
        """Returns detailed function counts.

        Conceptually, the FunctionCounts returned can be thought of as a tuple
        of (count, path_and_function_name) tuples.

        `inclusive` matches the semantics of callgrind. If True, the counts
        include instructions executed by children. `inclusive=True` is useful
        for identifying hot spots in code; `inclusive=False` is useful for
        reducing noise when diffing counts from two different runs. (See
        CallgrindStats.delta(...) for more details)
        """
        return self.stmt_inclusive_stats if inclusive else self.stmt_exclusive_stats

    def counts(self, *, denoise: bool = False) -> int:
        """Returns the total number of instructions executed.

        See `FunctionCounts.denoise()` for an explanation of the `denoise` arg.
        """
        stats = self.stmt_exclusive_stats
        return (stats.denoise() if denoise else stats).sum()

    # FIXME: Once 3.7 is the minimum version, type annotate `other` per PEP 563
    def delta(
        self,
        other: CallgrindStats,
        inclusive: bool = False,
    ) -> FunctionCounts:
        """Diff two sets of counts.

        One common reason to collect instruction counts is to determine the
        effect that a particular change will have on the number of instructions
        needed to perform some unit of work. If a change increases that number, the
        next logical question is "why". This generally involves looking at what part
        if the code increased in instruction count. This function automates that
        process so that one can easily diff counts on both an inclusive and
        exclusive basis.
        """
        return self.stats(inclusive=inclusive) - other.stats(inclusive=inclusive)

    def as_standardized(self) -> CallgrindStats:
        """Strip library names and some prefixes from function strings.

        When comparing two different sets of instruction counts, on stumbling
        block can be path prefixes. Callgrind includes the full filepath
        when reporting a function (as it should). However, this can cause
        issues when diffing profiles. If a key component such as Python
        or PyTorch was built in separate locations in the two profiles, which
        can result in something resembling::

            23234231 /tmp/first_build_dir/thing.c:foo(...)
             9823794 /tmp/first_build_dir/thing.c:bar(...)
              ...
               53453 .../aten/src/Aten/...:function_that_actually_changed(...)
              ...
             -9823794 /tmp/second_build_dir/thing.c:bar(...)
            -23234231 /tmp/second_build_dir/thing.c:foo(...)

        Stripping prefixes can ameliorate this issue by regularizing the
        strings and causing better cancellation of equivalent call sites
        when diffing.
        """
        def strip(stats: FunctionCounts) -> FunctionCounts:
            transforms = (
                # PyTorch may have been built in different locations.
                (r"^.+build/\.\./", "build/../"),
                (r"^.+/" + re.escape("build/aten/"), "build/aten/"),

                # "Python" and "Objects" come from CPython.
                (r"^.+/" + re.escape("Python/"), "Python/"),
                (r"^.+/" + re.escape("Objects/"), "Objects/"),

                # Strip library name. e.g. `libtorch.so`
                (r"\s\[.+\]$", ""),
            )

            for before, after in transforms:
                stats = stats.transform(lambda fn: re.sub(before, after, fn))

            return stats

        return CallgrindStats(
            task_spec=self.task_spec,
            number_per_run=self.number_per_run,
            built_with_debug_symbols=self.built_with_debug_symbols,
            baseline_inclusive_stats=strip(self.baseline_inclusive_stats),
            baseline_exclusive_stats=strip(self.baseline_exclusive_stats),
            stmt_inclusive_stats=strip(self.stmt_inclusive_stats),
            stmt_exclusive_stats=strip(self.stmt_exclusive_stats),

            # `as_standardized` will change symbol names, so the contents will
            # no longer map directly to `callgrind.out`
            stmt_callgrind_out=None,
        )


class Serialization(enum.Enum):
    PICKLE = 0
    TORCH = 1
    TORCH_JIT = 2


_GLOBALS_ALLOWED_TYPES: dict[Serialization, tuple[Any, ...]] = {
    Serialization.PICKLE: (str, bytes, bool, int, float, complex),
    Serialization.TORCH_JIT: (torch.jit.ScriptFunction, torch.jit.ScriptModule),
    Serialization.TORCH: (torch.nn.Module,),
}


class CopyIfCallgrind:
    """Signal that a global may be replaced with a deserialized copy.

    See `GlobalsBridge` for why this matters.
    """
    def __init__(self, value: Any, *, setup: str | None = None) -> None:
        for method, supported_types in _GLOBALS_ALLOWED_TYPES.items():
            if any(isinstance(value, t) for t in supported_types):
                self._value: Any = value
                self._setup: str | None = setup
                self._serialization: Serialization = method
                break
        else:
            supported_str = "\n".join([
                getattr(t, "__name__", repr(t))
                for t in it.chain(_GLOBALS_ALLOWED_TYPES.values())])

            raise ValueError(
                f"Unsupported type: {type(value)}\n"
                f"`collect_callgrind` restricts globals to the following types:\n"
                f"{textwrap.indent(supported_str, '  ')}"
            )

    @property
    def value(self) -> Any:
        return self._value

    @property
    def setup(self) -> str | None:
        return self._setup

    @property
    def serialization(self) -> Serialization:
        return self._serialization

    @staticmethod
    def unwrap_all(globals: dict[str, Any]) -> dict[str, Any]:
        return {
            k: (v.value if isinstance(v, CopyIfCallgrind) else v)
            for k, v in globals.items()
        }


class GlobalsBridge:
    """Handle the transfer of (certain) globals when collecting Callgrind statistics.

    Key takeaway: Any globals passed must be wrapped in `CopyIfCallgrind` to
                  work with `Timer.collect_callgrind`.

    Consider the following code snippet:
    ```
        import pickle
        import timeit

        class Counter:
            value = 0

            def __call__(self):
                self.value += 1

        counter = Counter()
        timeit.Timer("counter()", globals={"counter": counter}).timeit(10)
        print(counter.value)  # 10

        timeit.Timer(
            "counter()",
            globals={"counter": pickle.loads(pickle.dumps(counter))}
        ).timeit(20)
        print(counter.value)  # Still 10
    ```

    In the first case, `stmt` is executed using the objects in `globals`;
    however, the addition of serialization and deserialization changes the
    semantics and may meaningfully change behavior.

    This is a practical consideration when collecting Callgrind statistics.
    Unlike `exec` based execution (which `timeit` uses under the hood) which
    can share in-memory data structures with the caller, Callgrind collection
    requires an entirely new process in order to run under Valgrind. This means
    that any data structures used for statement execution will have to be
    serialized and deserialized in the subprocess.

    In order to avoid surprising semantics from (user invisible) process
    boundaries, what can be passed through `globals` is severely restricted
    for `Timer.collect_callgrind`. It is expected that most setup should be
    achievable (albeit perhaps less ergonomically) by passing a `setup`
    string.

    There are, however, exceptions. One such class are TorchScripted functions.
    Because they require a concrete file with source code it is not possible
    to define them using a `setup` string. Another group are torch.nn.Modules,
    whose construction can be complex and prohibitively cumbersome to coerce
    into a `setup` string. Finally, most builtin types are sufficiently well
    behaved and sufficiently common to warrant allowing as well. (e.g.
    `globals={"n": 1}` is very convenient.)

    Fortunately, all have well defined serialization semantics. This class
    is responsible for enabling the Valgrind subprocess to use elements in
    `globals` so long as they are an allowed type.

    Caveats:
        The user is required to acknowledge this serialization by wrapping
        elements in `globals` with `CopyIfCallgrind`.

        While ScriptFunction and ScriptModule are expected to save and load
        quite robustly, it is up to the user to ensure that an nn.Module can
        un-pickle successfully.

        `torch.Tensor` and `np.ndarray` are deliberately excluded. The
        serialization/deserialization process perturbs the representation of a
        tensor in ways that could result in incorrect measurements. For example,
        if a tensor lives in pinned CPU memory, this fact would not be preserved
        by a dump, and that will in turn change the performance of certain CUDA
        operations.
    """

    def __init__(self, globals: dict[str, Any], data_dir: str) -> None:
        self._globals: dict[str, CopyIfCallgrind] = {}
        self._data_dir = data_dir
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        if globals.get("torch", torch) is not torch:
            raise ValueError("`collect_callgrind` does not support mocking out `torch`.")

        for name, value in globals.items():
            if name in ("torch", "__builtins__"):
                # Torch will be imported by the collection script, and
                # __builtins__ is added by Timer.
                continue

            if not isinstance(value, CopyIfCallgrind):
                raise ValueError(
                    "`collect_callgrind` requires that globals be wrapped in "
                    "`CopyIfCallgrind` so that serialization is explicit."
                )

            self._globals[name] = value

    def construct(self) -> str:
        load_lines = []
        for name, wrapped_value in self._globals.items():
            if wrapped_value.setup is not None:
                load_lines.append(textwrap.dedent(wrapped_value.setup))

            if wrapped_value.serialization == Serialization.PICKLE:
                path = os.path.join(self._data_dir, f"{name}.pkl")
                load_lines.append(
                    f"with open({repr(path)}, 'rb') as f:\n    {name} = pickle.load(f)")
                with open(path, "wb") as f:
                    pickle.dump(wrapped_value.value, f)

            elif wrapped_value.serialization == Serialization.TORCH:
                path = os.path.join(self._data_dir, f"{name}.pt")
                # TODO: Figure out if we can use torch.serialization.add_safe_globals here
                # Using weights_only=False after the change in
                # https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573
                load_lines.append(f"{name} = torch.load({repr(path)}, weights_only=False)")
                torch.save(wrapped_value.value, path)

            elif wrapped_value.serialization == Serialization.TORCH_JIT:
                path = os.path.join(self._data_dir, f"{name}.pt")
                load_lines.append(f"{name} = torch.jit.load({repr(path)})")
                with open(path, "wb") as f:
                    torch.jit.save(wrapped_value.value, f)  # type: ignore[no-untyped-call]

            else:
                raise NotImplementedError(
                    f"Unknown serialization method: {wrapped_value.serialization}")

        return "\n".join(load_lines)


class _ValgrindWrapper:
    def __init__(self) -> None:
        self._bindings_module: CallgrindModuleType | None = None
        valgrind_symbols = (
            "_valgrind_supported_platform",
            "_valgrind_toggle",
            "_valgrind_toggle_and_dump_stats",
        )
        if all(hasattr(torch._C, symbol) for symbol in valgrind_symbols):
            self._supported_platform: bool = torch._C._valgrind_supported_platform()

        else:
            print("Callgrind bindings are not present in `torch._C`. JIT-ing bindings.")
            self._bindings_module = cpp_jit.get_compat_bindings()
            if not all(hasattr(self._bindings_module, symbol) for symbol in valgrind_symbols):
                raise AssertionError("JIT-compiled callgrind bindings are missing required symbols")
            self._supported_platform = self._bindings_module._valgrind_supported_platform()

        self._commands_available: dict[str, bool] = {}
        if self._supported_platform:
            # Only bother checking on supported platforms.
            for cmd in ("valgrind", "callgrind_control", "callgrind_annotate"):
                self._commands_available[cmd] = not subprocess.run(
                    ["which", cmd],
                    capture_output=True,
                    check=False,
                ).returncode

        self._build_type: str | None = None
        build_search = re.search("BUILD_TYPE=(.+),", torch.__config__.show())  # type: ignore[no-untyped-call]
        if build_search is not None:
            self._build_type = build_search.groups()[0].split(",")[0]

    def _validate(self) -> None:
        if not self._supported_platform:
            raise OSError("Valgrind is not supported on this platform.")

        missing_cmds = [cmd for cmd, available in self._commands_available.items() if not available]
        if missing_cmds:
            raise OSError("Missing: " + ", ".join(missing_cmds))

    def collect_callgrind(
        self,
        task_spec: common.TaskSpec,
        globals: dict[str, Any],
        *,
        number: int,
        repeats: int,
        collect_baseline: bool,
        is_python: bool,
        retain_out_file: bool,
    ) -> tuple[CallgrindStats, ...]:
        """Collect stats, and attach a reference run which can be used to filter interpreter overhead."""
        self._validate()
        if not is_python and collect_baseline:
            raise AssertionError("collect_baseline is only supported for Python timers")

        *task_stats, baseline_stats = self._invoke(
            task_spec=task_spec,
            globals=globals,
            number=number,
            repeats=repeats,
            collect_baseline=collect_baseline,
            is_python=is_python,
            retain_out_file=retain_out_file,
        )
        if len(task_stats) != repeats:
            raise AssertionError("Unexpected number of task stats returned from _invoke")

        return tuple(
            CallgrindStats(
                task_spec=task_spec,
                number_per_run=number,
                built_with_debug_symbols=self._build_type == "RelWithDebInfo",
                baseline_inclusive_stats=baseline_stats[0],
                baseline_exclusive_stats=baseline_stats[1],
                stmt_inclusive_stats=stmt_inclusive_stats,
                stmt_exclusive_stats=stmt_exclusive_stats,
                stmt_callgrind_out=out_contents,
            )
            for stmt_inclusive_stats, stmt_exclusive_stats, out_contents in task_stats
        )

    def _invoke(
        self,
        *,
        task_spec: common.TaskSpec,
        globals: dict[str, Any],
        number: int,
        repeats: int,
        collect_baseline: bool,
        is_python: bool,
        retain_out_file: bool,
    ) -> tuple[tuple[FunctionCounts, FunctionCounts, str | None], ...]:
        """Core invocation method for Callgrind collection.

        Valgrind operates by effectively replacing the CPU with an emulated
        version which allows it to instrument any code at the cost of severe
        performance degradation. This has the practical effect that in order
        to collect Callgrind statistics, a new process has to be created
        running under `valgrind`. The steps for this process are:

        1) Create a scratch directory.
        2) Codegen a run script. (_ValgrindWrapper._construct_script)
            Inside the run script:
                * Validate that Python and torch match the parent process
                * Validate that it is indeed running under valgrind
                * Execute `setup` and warm up `stmt`
                * Begin collecting stats
                * Run the `stmt` loop
                * Stop collecting stats
        3) Parse the run results.
        4) Cleanup the scratch directory.
        """
        working_dir = common._make_temp_dir(prefix="callgrind")
        data_dir = os.path.join(working_dir, "data")
        script_file = os.path.join(working_dir, "timer_callgrind.py")
        callgrind_out = os.path.join(working_dir, "callgrind.out")
        error_log = os.path.join(working_dir, "error.txt")
        stat_log = os.path.join(working_dir, "callgrind_stat.txt")
        stdout_stderr_log = os.path.join(working_dir, "stdout_stderr.log")

        def run(args: list[str], **kwargs: Any) -> tuple[CompletedProcessType, str]:
            # https://thraxil.org/users/anders/posts/2008/03/13/Subprocess-Hanging-PIPE-is-your-enemy/
            with open(stdout_stderr_log, "wb") as f_stdout_stderr:
                invocation = subprocess.run(
                    args,
                    stdout=f_stdout_stderr,
                    stderr=subprocess.STDOUT,
                    **kwargs,
                )
                with open(stdout_stderr_log) as f:
                    return invocation, f.read()

        try:
            if is_python:
                if self._bindings_module is not None:
                    shutil.copy(
                        self._bindings_module.__file__,
                        os.path.join(working_dir, os.path.split(self._bindings_module.__file__)[1])
                    )

                script_file = os.path.join(working_dir, "timer_callgrind.py")
                with open(script_file, "w") as f:
                    f.write(self._construct_script(
                        task_spec,
                        globals=GlobalsBridge(globals, data_dir),
                        number=number,
                        repeats=repeats,
                        collect_baseline=collect_baseline,
                        error_log=error_log,
                        stat_log=stat_log,
                        bindings=self._bindings_module))

                run_loop_cmd = ["python", script_file]
            else:
                if collect_baseline:
                    raise AssertionError("collect_baseline must be False for non-Python timers")
                run_loop_exec = cpp_jit.compile_callgrind_template(
                    stmt=task_spec.stmt,
                    setup=task_spec.setup,
                    global_setup=task_spec.global_setup,
                )
                run_loop_cmd = [
                    run_loop_exec,
                    "--number", str(number),
                    "--number-warmup", str(min(number, 10)),
                    "--repeats", str(repeats),
                    "--number-threads", str(task_spec.num_threads),
                ]

            valgrind_invocation, valgrind_invocation_output = run([
                "valgrind",
                "--tool=callgrind",
                f"--callgrind-out-file={callgrind_out}",
                "--dump-line=yes",
                "--dump-instr=yes",
                "--instr-atstart=yes",
                "--collect-atstart=no",
            ] + run_loop_cmd)

            if valgrind_invocation.returncode:
                error_report = ""
                if os.path.exists(error_log):
                    with open(error_log) as f:
                        error_report = f.read()
                if not error_report:
                    error_report = "Unknown error.\n" + valgrind_invocation_output

                raise OSError(f"Failed to collect callgrind profile:\n{error_report}")

            def parse_output(fpath: str, inclusive: bool) -> FunctionCounts:
                _annotate_invocation, annotate_invocation_output = run([
                    "callgrind_annotate",
                    f"--inclusive={'yes' if inclusive else 'no'}",
                    "--threshold=100",
                    "--show-percs=no",
                    fpath
                ], check=True)

                total_pattern = re.compile(r"^([0-9,]+)\s+PROGRAM TOTALS")
                begin_pattern = re.compile(r"Ir\s+file:function")
                function_pattern = re.compile(r"^\s*([0-9,]+)\s+(.+:.+)$")

                class ScanState(enum.Enum):
                    SCANNING_FOR_TOTAL = 0
                    SCANNING_FOR_START = 1
                    PARSING = 2

                scan_state = ScanState.SCANNING_FOR_TOTAL
                fn_counts = []
                for l in annotate_invocation_output.splitlines(keepends=False):
                    if scan_state == ScanState.SCANNING_FOR_TOTAL:
                        total_match = total_pattern.match(l)
                        if total_match:
                            program_totals = int(total_match.groups()[0].replace(",", ""))
                            scan_state = ScanState.SCANNING_FOR_START

                    elif scan_state == ScanState.SCANNING_FOR_START:
                        if begin_pattern.match(l):
                            scan_state = ScanState.PARSING

                    else:
                        if scan_state != ScanState.PARSING:
                            raise AssertionError("Failed to enter PARSING state while parsing callgrind_annotate output")
                        fn_match = function_pattern.match(l)
                        if fn_match:
                            ir_str, file_function = fn_match.groups()
                            ir = int(ir_str.replace(",", ""))
                            if ir == program_totals:  # type: ignore[possibly-undefined]
                                # Callgrind includes some top level red herring symbols when
                                # a program dumps multiple profiles.
                                continue
                            fn_counts.append(FunctionCount(ir, file_function))

                        elif re.match(r"-+", l):
                            # Ignore heading separator lines.
                            continue

                        else:
                            break

                if scan_state != ScanState.PARSING:
                    raise AssertionError(f"Failed to parse {fpath}")
                return FunctionCounts(tuple(sorted(fn_counts, reverse=True)), inclusive=inclusive)

            def read_results(i: int) -> tuple[FunctionCounts, FunctionCounts, str | None]:
                if i == repeats and not collect_baseline:
                    # Null baseline.
                    return (
                        FunctionCounts((), inclusive=True),
                        FunctionCounts((), inclusive=False),
                        None,
                    )

                fpath = f"{callgrind_out}.{i + 1}"  # Callgrind one-indexes files.
                callgrind_out_contents: str | None = None
                if retain_out_file:
                    with open(fpath) as f:
                        callgrind_out_contents = f.read()

                return (
                    parse_output(fpath, inclusive=True),
                    parse_output(fpath, inclusive=False),
                    callgrind_out_contents
                )

            return tuple(read_results(i) for i in range(repeats + 1))
        finally:
            shutil.rmtree(working_dir)

    @staticmethod
    def _construct_script(
        task_spec: common.TaskSpec,
        globals: GlobalsBridge,
        *,
        number: int,
        repeats: int,
        collect_baseline: bool,
        error_log: str,
        stat_log: str,
        bindings: CallgrindModuleType | None,
    ) -> str:
        def block_stmt(stmt: str, indent: int = 0) -> str:
            """Partially unroll benchmark loop.

            The naive template looks something like:
                "for _ in range({number}): {stmt}"

            However a loop in Python is surprisingly expensive, and significantly
            increases the number of background Python instructions. So instead we
            partially unroll the loops, with a block size of 100 chosen to keep
            the instruction overhead from `range` low while also not ballooning
            the size of the generated file.
            """
            block_size = 100
            loop_count = number // block_size
            if loop_count == 1:
                # There is no point in having `for _ in range(1): ...` rather
                # than just `...`, and this lets us save shave a few background
                # instructions.
                loop_count = 0
            remainder = number - block_size * loop_count
            blocked_stmt = ""

            if loop_count:
                unrolled_stmts = textwrap.indent("\n".join([stmt] * block_size), " " * 4)
                blocked_stmt += f"for _ in range({loop_count}):\n{unrolled_stmts}\n"

            if remainder:
                blocked_stmt += "\n".join([stmt] * remainder)

            return textwrap.indent(blocked_stmt, " " * indent)

        pass_baseline = (
            "callgrind_bindings._valgrind_toggle()\n"
            f"{block_stmt('pass')}\n"
            "callgrind_bindings._valgrind_toggle_and_dump_stats()"
        )

        return textwrap.dedent(r"""
            import gc
            import os
            import pickle
            import subprocess
            import sys
            import time

            # Mitigate https://github.com/pytorch/pytorch/issues/37377
            # which can sometimes cause the subprocess call to fail.
            import numpy as np

            import torch
            torch.set_num_threads({num_threads})

            {bindings_import}

            PID = os.getpid()

            def log_failure(msg):
                with open({error_log_repr}, "wt") as f:
                    f.write(msg)
                sys.exit(1)

            def check_result(completed_process):
                if completed_process.returncode:
                    log_failure(f"Command failed: {{' '.join(completed_process.args)}}")
                return completed_process

            # =============================================================================
            # == Check that subprocess matches parent =====================================
            # =============================================================================
            if os.path.realpath(sys.executable) != "{parent_interpreter}":
                log_failure(
                    "Interpreter mismatch:\n"
                    f"  {{os.path.realpath(sys.executable)}}\n    vs.\n  {parent_interpreter}"
                )

            if torch.__file__ != "{torch_file}":
                log_failure(
                    "PyTorch does not match expected file:\n"
                    f"  {{torch.__file__}}\n    vs.\n  {torch_file}"
                )

            # =============================================================================
            # == User specified setup =====================================================
            # =============================================================================
            # Load serialized globals
            {load_globals}

            # User setup str
            {setup}

            for _ in range({warmup_number}):
            {indented_stmt}

            # =============================================================================
            # == Callgrind management =====================================================
            # =============================================================================
            with open("{stat_log}", "wb") as stat_file:
                # If many instances of callgrind are running at once, the output of
                # `callgrind_control` may exceed 16kb which would cause `subprocess.PIPE`
                # to deadlock. So instead we use a file.
                callgrind_stat = check_result(subprocess.run(
                    ["callgrind_control", "--stat"],
                    stdout=stat_file,
                    stderr=subprocess.STDOUT,
                ))

            with open("{stat_log}", "rt") as stat_file:
                stat_lines = stat_file.read().splitlines()

            if f"PID {{PID}}: python {{__file__}}" not in stat_lines:
                log_failure("Process does not appear to be running callgrind.")

            gc.collect()
            time.sleep(0.01)

            # =============================================================================
            # == User code block ==========================================================
            # =============================================================================
            for _ in range({repeats}):
                callgrind_bindings._valgrind_toggle()
            {blocked_stmt}
                callgrind_bindings._valgrind_toggle_and_dump_stats()
                gc.collect()

            {baseline}
        """).strip().format(
            indented_stmt=textwrap.indent(task_spec.stmt, " " * 4),
            blocked_stmt=block_stmt(task_spec.stmt, indent=4),
            baseline=(pass_baseline if collect_baseline else ""),
            number=number,
            repeats=repeats,
            load_globals=globals.construct(),
            setup=task_spec.setup,
            warmup_number=min(number, 10),
            num_threads=task_spec.num_threads,
            error_log_repr=repr(error_log),
            stat_log=stat_log,
            parent_interpreter=os.path.realpath(sys.executable),
            torch_file=torch.__file__,
            bindings_import=(
                "import torch._C as callgrind_bindings" if bindings is None
                else f"import {bindings.__name__} as callgrind_bindings"),
        )


CALLGRIND_SINGLETON: _ValgrindWrapper | None = None
def wrapper_singleton() -> _ValgrindWrapper:
    global CALLGRIND_SINGLETON
    if CALLGRIND_SINGLETON is None:
        CALLGRIND_SINGLETON = _ValgrindWrapper()
    return CALLGRIND_SINGLETON
<|endoftext|># mypy: allow-untyped-defs
import numpy as np
import torch

from torch.utils.benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedTensor


_MIN_DIM_SIZE = 16
_MAX_DIM_SIZE = 16 * 1024 ** 2
_POW_TWO_SIZES = tuple(2 ** i for i in range(
    int(np.log2(_MIN_DIM_SIZE)),
    int(np.log2(_MAX_DIM_SIZE)) + 1,
))


class BinaryOpFuzzer(Fuzzer):
    def __init__(self, seed, dtype=torch.float32, cuda=False) -> None:
        super().__init__(
            parameters=[
                # Dimensionality of x and y. (e.g. 1D, 2D, or 3D.)
                FuzzedParameter("dim", distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True),

                # Shapes for `x` and `y`.
                #       It is important to test all shapes, however
                #   powers of two are especially important and therefore
                #   warrant special attention. This is done by generating
                #   both a value drawn from all integers between the min and
                #   max allowed values, and another from only the powers of two
                #   (both distributions are loguniform) and then randomly
                #   selecting between the two.
                #       Moreover, `y` will occasionally have singleton
                #   dimensions in order to test broadcasting.
                [
                    FuzzedParameter(
                        name=f"k_any_{i}",
                        minval=_MIN_DIM_SIZE,
                        maxval=_MAX_DIM_SIZE,
                        distribution="loguniform",
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k_pow2_{i}",
                        distribution={size: 1. / len(_POW_TWO_SIZES) for size in _POW_TWO_SIZES}
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={
                            ParameterAlias(f"k_any_{i}"): 0.8,
                            ParameterAlias(f"k_pow2_{i}"): 0.2,
                        },
                        strict=True,
                    ) for i in range(3)
                ],

                [
                    FuzzedParameter(
                        name=f"y_k{i}",
                        distribution={
                            ParameterAlias(f"k{i}"): 0.8,
                            1: 0.2,
                        },
                        strict=True,
                    ) for i in range(3)
                ],

                # Steps for `x` and `y`. (Benchmarks strided memory access.)
                [
                    FuzzedParameter(
                        name=f"{name}_step_{i}",
                        distribution={1: 0.8, 2: 0.06, 4: 0.06, 8: 0.04, 16: 0.04},
                    )
                    for i in range(3)
                    for name in ("x", "y")
                ],

                # Repeatable entropy for downstream applications.
                FuzzedParameter(name="random_value", minval=0, maxval=2 ** 32 - 1, distribution="uniform"),
            ],
            tensors=[
                FuzzedTensor(
                    name="x",
                    size=("k0", "k1", "k2"),
                    steps=("x_step_0", "x_step_1", "x_step_2"),
                    probability_contiguous=0.75,
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    max_allocation_bytes=2 * 1024**3,  # 2 GB
                    dim_parameter="dim",
                    dtype=dtype,
                    cuda=cuda,
                ),
                FuzzedTensor(
                    name="y",
                    size=("y_k0", "y_k1", "y_k2"),
                    steps=("x_step_0", "x_step_1", "x_step_2"),
                    probability_contiguous=0.75,
                    max_allocation_bytes=2 * 1024**3,  # 2 GB
                    dim_parameter="dim",
                    dtype=dtype,
                    cuda=cuda,
                ),
            ],
            seed=seed,
        )
<|endoftext|># mypy: allow-untyped-defs
import math

import torch
from torch.utils import benchmark
from torch.utils.benchmark import FuzzedParameter, FuzzedTensor, ParameterAlias


__all__ = ['SpectralOpFuzzer']

MIN_DIM_SIZE = 16
MAX_DIM_SIZE = 16 * 1024

def power_range(upper_bound, base):
    return (base ** i for i in range(int(math.log(upper_bound, base)) + 1))

# List of regular numbers from MIN_DIM_SIZE to MAX_DIM_SIZE
# These numbers factorize into multiples of prime factors 2, 3, and 5 only
# and are usually the fastest in FFT implementations.
REGULAR_SIZES = []
for i in power_range(MAX_DIM_SIZE, 2):
    for j in power_range(MAX_DIM_SIZE // i, 3):
        ij = i * j
        for k in power_range(MAX_DIM_SIZE // ij, 5):
            ijk = ij * k
            if ijk > MIN_DIM_SIZE:
                REGULAR_SIZES.append(ijk)
REGULAR_SIZES.sort()

class SpectralOpFuzzer(benchmark.Fuzzer):
    def __init__(self, *, seed: int, dtype=torch.float64,
                 cuda: bool = False, probability_regular: float = 1.0) -> None:
        super().__init__(
            parameters=[
                # Dimensionality of x. (e.g. 1D, 2D, or 3D.)
                FuzzedParameter("ndim", distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True),

                # Shapes for `x`.
                #   It is important to test all shapes, however
                #   regular sizes are especially important to the FFT and therefore
                #   warrant special attention. This is done by generating
                #   both a value drawn from all integers between the min and
                #   max allowed values, and another from only the regular numbers
                #   (both distributions are loguniform) and then randomly
                #   selecting between the two.
                [
                    FuzzedParameter(
                        name=f"k_any_{i}",
                        minval=MIN_DIM_SIZE,
                        maxval=MAX_DIM_SIZE,
                        distribution="loguniform",
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k_regular_{i}",
                        distribution={size: 1. / len(REGULAR_SIZES) for size in REGULAR_SIZES}
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={
                            ParameterAlias(f"k_regular_{i}"): probability_regular,
                            ParameterAlias(f"k_any_{i}"): 1 - probability_regular,
                        },
                        strict=True,
                    ) for i in range(3)
                ],

                # Steps for `x`. (Benchmarks strided memory access.)
                [
                    FuzzedParameter(
                        name=f"step_{i}",
                        distribution={1: 0.8, 2: 0.06, 4: 0.06, 8: 0.04, 16: 0.04},
                    ) for i in range(3)
                ],
            ],
            tensors=[
                FuzzedTensor(
                    name="x",
                    size=("k0", "k1", "k2"),
                    steps=("step_0", "step_1", "step_2"),
                    probability_contiguous=0.75,
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    max_allocation_bytes=2 * 1024**3,  # 2 GB
                    dim_parameter="ndim",
                    dtype=dtype,
                    cuda=cuda,
                ),
            ],
            seed=seed,
        )
<|endoftext|># mypy: allow-untyped-defs
import numpy as np
import torch

from torch.utils.benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedTensor


_MIN_DIM_SIZE = 16
_MAX_DIM_SIZE = 16 * 1024 ** 2
_POW_TWO_SIZES = tuple(2 ** i for i in range(
    int(np.log2(_MIN_DIM_SIZE)),
    int(np.log2(_MAX_DIM_SIZE)) + 1,
))


class UnaryOpFuzzer(Fuzzer):
    def __init__(self, seed, dtype=torch.float32, cuda=False) -> None:
        super().__init__(
            parameters=[
                # Dimensionality of x. (e.g. 1D, 2D, or 3D.)
                FuzzedParameter("dim", distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True),

                # Shapes for `x`.
                #   It is important to test all shapes, however
                #   powers of two are especially important and therefore
                #   warrant special attention. This is done by generating
                #   both a value drawn from all integers between the min and
                #   max allowed values, and another from only the powers of two
                #   (both distributions are loguniform) and then randomly
                #   selecting between the two.
                [
                    FuzzedParameter(
                        name=f"k_any_{i}",
                        minval=_MIN_DIM_SIZE,
                        maxval=_MAX_DIM_SIZE,
                        distribution="loguniform",
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k_pow2_{i}",
                        distribution={size: 1. / len(_POW_TWO_SIZES) for size in _POW_TWO_SIZES}
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={
                            ParameterAlias(f"k_any_{i}"): 0.8,
                            ParameterAlias(f"k_pow2_{i}"): 0.2,
                        },
                        strict=True,
                    ) for i in range(3)
                ],

                # Steps for `x`. (Benchmarks strided memory access.)
                [
                    FuzzedParameter(
                        name=f"x_step_{i}",
                        distribution={1: 0.8, 2: 0.06, 4: 0.06, 8: 0.04, 16: 0.04},
                    ) for i in range(3)
                ],

                # Repeatable entropy for downstream applications.
                FuzzedParameter(name="random_value", minval=0, maxval=2 ** 32 - 1, distribution="uniform"),
            ],
            tensors=[
                FuzzedTensor(
                    name="x",
                    size=("k0", "k1", "k2"),
                    steps=("x_step_0", "x_step_1", "x_step_2"),
                    probability_contiguous=0.75,
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    max_allocation_bytes=2 * 1024**3,  # 2 GB
                    dim_parameter="dim",
                    dtype=dtype,
                    cuda=cuda,
                ),
            ],
            seed=seed,
        )
<|endoftext|># mypy: allow-untyped-defs
import numpy as np
import torch

from torch.utils.benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedSparseTensor


_MIN_DIM_SIZE = 16
_MAX_DIM_SIZE = 16 * 1024 ** 2
_POW_TWO_SIZES = tuple(2 ** i for i in range(
    int(np.log2(_MIN_DIM_SIZE)),
    int(np.log2(_MAX_DIM_SIZE)) + 1,
))


class BinaryOpSparseFuzzer(Fuzzer):
    def __init__(self, seed, dtype=torch.float32, cuda=False) -> None:
        super().__init__(
            parameters=[
                # Dimensionality of x and y. (e.g. 1D, 2D, or 3D.)
                FuzzedParameter("dim_parameter", distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True),
                FuzzedParameter(
                    name="sparse_dim",
                    distribution={1: 0.4, 2: 0.4, 3: 0.2},
                    strict=True
                ),
                # Shapes for `x` and `y`.
                #       It is important to test all shapes, however
                #   powers of two are especially important and therefore
                #   warrant special attention. This is done by generating
                #   both a value drawn from all integers between the min and
                #   max allowed values, and another from only the powers of two
                #   (both distributions are loguniform) and then randomly
                #   selecting between the two.
                #       Moreover, `y` will occasionally have singleton
                #   dimensions in order to test broadcasting.
                [
                    FuzzedParameter(
                        name=f"k_any_{i}",
                        minval=_MIN_DIM_SIZE,
                        maxval=_MAX_DIM_SIZE,
                        distribution="loguniform",
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k_pow2_{i}",
                        distribution={size: 1. / len(_POW_TWO_SIZES) for size in _POW_TWO_SIZES}
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={
                            ParameterAlias(f"k_any_{i}"): 0.8,
                            ParameterAlias(f"k_pow2_{i}"): 0.2,
                        },
                        strict=True,
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"y_k{i}",
                        distribution={
                            ParameterAlias(f"k{i}"): 1.0},
                        strict=True,
                    ) for i in range(3)
                ],
                FuzzedParameter(
                    name="density",
                    distribution={0.1: 0.4, 0.05: 0.3, 0.01: 0.3},
                ),
                FuzzedParameter(
                    name="coalesced",
                    distribution={True: 0.5, False: 0.5},
                ),
                # Repeatable entropy for downstream applications.
                FuzzedParameter(name="random_value", minval=0, maxval=2 ** 32 - 1, distribution="uniform"),
            ],
            tensors=[
                FuzzedSparseTensor(
                    name="x",
                    size=("k0", "k1", "k2"),
                    dim_parameter="dim_parameter",
                    sparse_dim="sparse_dim",
                    density="density",
                    coalesced="coalesced",
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    dtype=dtype,
                    cuda=cuda,
                ),
                FuzzedSparseTensor(
                    name="y",
                    size=("y_k0", "y_k1", "y_k2"),
                    dim_parameter="dim_parameter",
                    sparse_dim="sparse_dim",
                    density="density",
                    coalesced="coalesced",
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    dtype=dtype,
                    cuda=cuda,
                ),
            ],
            seed=seed,
        )
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.types import _dtype

from torch.utils.benchmark import Fuzzer, FuzzedParameter, ParameterAlias, FuzzedSparseTensor

__all__ = ["UnaryOpSparseFuzzer"]

_MIN_DIM_SIZE = 16
_MAX_DIM_SIZE = 16 * 1024 ** 2
_POW_TWO_SIZES = tuple(2 ** i for i in range(
    int(np.log2(_MIN_DIM_SIZE)),
    int(np.log2(_MAX_DIM_SIZE)) + 1,
))

class UnaryOpSparseFuzzer(Fuzzer):
    def __init__(self, seed: int | None, dtype: _dtype | None = None, cuda: bool = False) -> None:
        if dtype is None:
            dtype = getattr(torch, 'float32', None)
        super().__init__(
            parameters=[
                # Sparse dim parameter of x. (e.g. 1D, 2D, or 3D.)
                FuzzedParameter("dim_parameter", distribution={1: 0.3, 2: 0.4, 3: 0.3}, strict=True),
                FuzzedParameter(
                    name="sparse_dim",
                    distribution={1: 0.4, 2: 0.4, 3: 0.2},
                    strict=True
                ),
                # Shapes for `x`.
                #   It is important to test all shapes, however
                #   powers of two are especially important and therefore
                #   warrant special attention. This is done by generating
                #   both a value drawn from all integers between the min and
                #   max allowed values, and another from only the powers of two
                #   (both distributions are loguniform) and then randomly
                #   selecting between the two.
                [
                    FuzzedParameter(
                        name=f"k_any_{i}",
                        minval=_MIN_DIM_SIZE,
                        maxval=_MAX_DIM_SIZE,
                        distribution="loguniform",
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k_pow2_{i}",
                        distribution={size: 1. / len(_POW_TWO_SIZES) for size in _POW_TWO_SIZES}
                    ) for i in range(3)
                ],
                [
                    FuzzedParameter(
                        name=f"k{i}",
                        distribution={
                            ParameterAlias(f"k_any_{i}"): 0.8,
                            ParameterAlias(f"k_pow2_{i}"): 0.2,
                        },
                        strict=True,
                    ) for i in range(3)
                ],
                FuzzedParameter(
                    name="density",
                    distribution={0.1: 0.4, 0.05: 0.3, 0.01: 0.3},
                ),
                FuzzedParameter(
                    name="coalesced",
                    distribution={True: 0.5, False: 0.5},
                ),
                FuzzedParameter(name="random_value", minval=0, maxval=2 ** 32 - 1, distribution="uniform"),
            ],
            tensors=[
                FuzzedSparseTensor(
                    name="x",
                    size=("k0", "k1", "k2"),
                    dim_parameter="dim_parameter",
                    sparse_dim="sparse_dim",
                    min_elements=4 * 1024,
                    max_elements=32 * 1024 ** 2,
                    density="density",
                    coalesced="coalesced",
                    dtype=dtype,
                    cuda=cuda,
                ),
            ],
            seed=seed,
        )
<|endoftext|># mypy: allow-untyped-defs
"""Microbenchmarks for the torch.fft module"""
from argparse import ArgumentParser
from collections import namedtuple
from collections.abc import Iterable

import torch
import torch.fft
from torch.utils import benchmark
