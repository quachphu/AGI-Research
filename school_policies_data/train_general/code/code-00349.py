        # for point-wise ops the two estimates will be the same, they matter for reductions only
        self.outside_loop = MemoryEstimate()
        self.loops = [MemoryEstimate()]
        self.persistent = MemoryEstimate()
        self.simulate_codegen()
        self.remove_kernel_local()

    def simulate_codegen(self) -> None:
        from .simd import SIMDKernel

        kernel_size_outside_loop = (*self.groups[:-1], sympy.S.One)
        kernel_size_inside_loop = tuple(self.groups)
        self.kernel_sizes = kernel_size_inside_loop

        for node in self.features.node_schedule:
            if node is DisableReduction:
                self.inside_reduction = False
                self.kernel_sizes = kernel_size_outside_loop
                continue
            elif node is EnableReduction:
                self.inside_reduction = True
                self.kernel_sizes = kernel_size_inside_loop
                self.loops.append(MemoryEstimate())
                continue
            if not isinstance(node, SchedulerNode):
                raise AssertionError(f"expected SchedulerNode, got {type(node)}")
            rw = extract_loop_body_with_args(
                node._body,
                SIMDKernel.map_kernel_groups_to_node_sizes(
                    self.kernel_sizes, node.get_ranges(), self.set_ranges
                ),
                dict(zip(self.symbols, self.kernel_sizes)),
            )

            for dep in rw._reads:
                if not isinstance(dep, MemoryDep):
                    continue
                dep = dep.simplify_with_ranges()
                if not self.persistent.writes.get(dep.name):  # cache miss?
                    self.persistent.reads[dep.name].add(dep)
                # the cache behavior of looped kernels is more complex than the persistent case above
                # some operations are lifted outside the loop (if they don't use the reduction dimension)
                # other operations are inside the loop, and can only be reused within the same loop
                if not (
                    self.outside_loop.writes.get(dep.name)
                    or self.loops[-1].writes.get(dep.name)
                ):
                    self.scope(dep).reads[dep.name].add(dep)
                    if dep.name in self.store_buffer_names and self.loops[-1].reads.get(
                        dep.name
                    ):
                        self.must_keep_buffers.add(dep.name)

            for dep in rw._writes:
                if not isinstance(dep, MemoryDep):
                    continue
                dep = dep.simplify_with_ranges()
                self.store_buffer_names.add(dep.name)
                self.persistent.writes[dep.name].add(dep)
                self.scope(dep).writes[dep.name].add(dep)

    def remove_kernel_local(self) -> None:
        # Remove any kernel-local buffers
        fused_node_names = OrderedSet(
            [n.get_name() for n in self.features.scheduler_nodes()]
        )
        for name in self.store_buffer_names:
            if not self.persistent.reads.get(
                name
            ) and V.graph.scheduler.can_buffer_be_removed_through_fusion(
                name, fused_node_names
            ):
                self.persistent.remove(name)
                if name not in self.must_keep_buffers:
                    # we can also remove this from the looped kernel
                    self.outside_loop.remove(name)
                    for loop in self.loops:
                        loop.remove(name)

        if not self.loops[-1]:
            self.loops.pop()  # for pointwise ops

    def scope(self, dep: MemoryDep) -> MemoryEstimate:
        """Determine how a read/write should be categorized"""
        if self.inside_reduction and (
            self.has_reduction_var(dep.index) or dep.is_indirect()
        ):
            return self.loops[-1]
        return self.outside_loop

    def has_reduction_var(self, index: sympy.Expr) -> bool:
        for sym in self.symbols[-self.num_reductions_dims :]:
            if isinstance(sym, sympy.Symbol) and sym in index.free_symbols:
                return True
        return False

    def set_ranges(self, *lengths: list[list[sympy.Expr]]) -> list[list[sympy.Expr]]:
        if len(self.kernel_sizes) != len(lengths):
            raise AssertionError(
                f"expected len(kernel_sizes) == len(lengths), got "
                f"{len(self.kernel_sizes)} != {len(lengths)}"
            )
        return [
            self.make_flat_range(sym, numel, length)
            for sym, numel, length in zip(self.symbols, self.kernel_sizes, lengths)
        ]

    @staticmethod
    def make_flat_range(
        sym: sympy.Symbol, numel: sympy.Expr, lengths: list[sympy.Expr]
    ) -> list[sympy.Expr]:
        if len(lengths) == 1 and numel == lengths[0]:
            return [sym]
        divisor = sympy.S.One
        itervars = []
        for length in reversed(lengths):
            if V.graph.sizevars.statically_known_equals(divisor * length, numel):
                expr = FloorDiv(sym, divisor)
            else:
                expr = ModularIndexing(sym, divisor, length)
            itervars.append(expr)
            divisor = divisor * length
        return [*reversed(itervars)]


@dataclasses.dataclass
class MemoryEstimate:
    """Tracks the memory usage of a single loop in the generated kernel"""

    reads: dict[str, OrderedSet[MemoryDep]] = dataclasses.field(
        default_factory=functools.partial(collections.defaultdict, OrderedSet)
    )
    writes: dict[str, OrderedSet[MemoryDep]] = dataclasses.field(
        default_factory=functools.partial(collections.defaultdict, OrderedSet)
    )

    def remove(self, name: str) -> None:
        self.reads.pop(name, None)
        self.writes.pop(name, None)

    def __bool__(self) -> bool:
        return bool(self.reads or self.writes)

    def __repr__(self) -> str:
        return f"""MemoryEstimate(
            reads={[*itertools.chain.from_iterable(self.reads.values())]!r},
            writes={[*itertools.chain.from_iterable(self.writes.values())]!r}
        )"""


@dataclasses.dataclass
class StatsForDim:
    """Memory usage stats for a block dimension in the generated kernel (different from user dimensions)"""

    # the number of load/store ops
    count_per_thread_contiguous: int = 0
    count_per_thread_broadcast: int = 0
    count_per_thread_non_contiguous: int = 0  # excludes broadcast

    # total bytes in each load/store op for a single element
    bytes_per_thread_contiguous: int = 0
    bytes_per_thread_broadcast: int = 0
    bytes_per_thread_non_contiguous: int = 0  # excludes broadcast

    # total bytes read by entire kernel
    bytes_contiguous_or_broadcast: sympy.Expr = sympy.S.Zero
    bytes_non_contiguous: sympy.Expr = sympy.S.Zero

    def __add__(self, other: typing.Self) -> StatsForDim:
        return StatsForDim(
            count_per_thread_contiguous=self.count_per_thread_contiguous
            + other.count_per_thread_contiguous,
            count_per_thread_broadcast=self.count_per_thread_broadcast
            + other.count_per_thread_broadcast,
            count_per_thread_non_contiguous=self.count_per_thread_non_contiguous
            + other.count_per_thread_non_contiguous,
            bytes_per_thread_contiguous=self.bytes_per_thread_contiguous
            + other.bytes_per_thread_contiguous,
            bytes_per_thread_broadcast=self.bytes_per_thread_broadcast
            + other.bytes_per_thread_broadcast,
            bytes_per_thread_non_contiguous=self.bytes_per_thread_non_contiguous
            + other.bytes_per_thread_non_contiguous,
            bytes_contiguous_or_broadcast=self.bytes_contiguous_or_broadcast
            + other.bytes_contiguous_or_broadcast,
            bytes_non_contiguous=self.bytes_non_contiguous + other.bytes_non_contiguous,
        )

    @property
    def count_per_thread(self) -> int:
        return (
            self.count_per_thread_contiguous
            + self.count_per_thread_broadcast
            + self.count_per_thread_non_contiguous
        )

    @property
    def bytes_per_thread(self) -> int:
        return (
            self.bytes_per_thread_contiguous
            + self.bytes_per_thread_broadcast
            + self.bytes_per_thread_non_contiguous
        )

    @property
    def bytes(self) -> sympy.Expr:
        return self.bytes_contiguous_or_broadcast + self.bytes_non_contiguous

    @property
    def contiguous_score(self) -> float:
        return 1.0 - self.count_per_thread_non_contiguous / max(
            self.count_per_thread, 1
        )


@dataclasses.dataclass
class StatsForLoop:
    """Memory usage stats for single loop in the generated kernel"""

    # load/store ops
    count_per_thread: int = 0
    bytes_per_thread: int = 0

    def __add__(self, other: typing.Self) -> StatsForLoop:
        return StatsForLoop(
            count_per_thread=self.count_per_thread + other.count_per_thread,
            bytes_per_thread=self.bytes_per_thread + other.bytes_per_thread,
        )


@dataclasses.dataclass
class StatsForReadsOrWrites:
    """Memory usage stats that are collected for reads/writes/both"""

    dim: list[StatsForDim]
    loop: list[StatsForLoop]
    # total bytes contiguous in any dimension
    bytes_contiguous_or_broadcast: sympy.Expr = sympy.S.Zero
    bytes_non_contiguous: sympy.Expr = sympy.S.Zero

    def __add__(self, other: typing.Self) -> StatsForReadsOrWrites:
        if len(self.dim) != len(other.dim):
            raise AssertionError(
                f"expected len(self.dim) == len(other.dim), got "
                f"{len(self.dim)} != {len(other.dim)}"
            )
        if len(self.loop) != len(other.loop):
            raise AssertionError(
                f"expected len(self.loop) == len(other.loop), got "
                f"{len(self.loop)} != {len(other.loop)}"
            )
        return StatsForReadsOrWrites(
            dim=[a + b for a, b in zip(self.dim, other.dim)],
            loop=[a + b for a, b in zip(self.loop, other.loop)],
            bytes_contiguous_or_broadcast=self.bytes_contiguous_or_broadcast
            + self.bytes_contiguous_or_broadcast,
            bytes_non_contiguous=self.bytes_non_contiguous + other.bytes_non_contiguous,
        )

    @property
    def count_per_thread(self) -> int:
        return self.dim[0].count_per_thread

    @property
    def bytes_per_thread(self) -> int:
        return self.dim[0].bytes_per_thread

    @property
    def bytes(self) -> sympy.Expr:
        return self.bytes_contiguous_or_broadcast + self.bytes_non_contiguous

    @classmethod
    def compute(
        cls,
        loop_deps: list[dict[str, OrderedSet[MemoryDep]]],
        index_symbols: list[sympy.Symbol],
    ) -> typing.Self:
        ndim = len(index_symbols)
        result = cls(dim := [StatsForDim() for _ in range(ndim)], [])
        for dep_group in loop_deps:
            result.loop.append(loop_stats := StatsForLoop())
            for name, deps in dep_group.items():
                if not deps:
                    raise AssertionError(f"expected non-empty deps for {name}")
                contiguous_or_broadcast = [True] * ndim
                numel = sympy.S.Zero
                itemsize = V.graph.get_dtype(name).itemsize
                loop_stats.count_per_thread += len(deps)
                loop_stats.bytes_per_thread += itemsize * len(deps)
                for dep in deps:
                    strides: list[sympy.Expr] = V.graph.sizevars.stride_vars(
                        dep.index, index_symbols
                    )
                    for i in range(ndim):
                        if V.graph.sizevars.statically_known_equals(strides[i], 1):
                            dim[i].count_per_thread_contiguous += 1
                            dim[i].bytes_per_thread_contiguous += itemsize
                        elif (
                            V.graph.sizevars.statically_known_equals(strides[i], 0)
                            and not dep.is_indirect()
                        ):
                            dim[i].count_per_thread_broadcast += 1
                            dim[i].bytes_per_thread_broadcast += itemsize
                        else:
                            dim[i].count_per_thread_non_contiguous += 1
                            dim[i].bytes_per_thread_non_contiguous += itemsize
                            contiguous_or_broadcast[i] = False
                    numel += dep.get_numel()
                if len(deps) > 1:
                    # can't read more elements than exist in the buffer
                    numel = Min(numel, V.graph.get_numel(name))
                nbytes = numel * itemsize
                for i in range(ndim):
                    if contiguous_or_broadcast[i]:
                        dim[i].bytes_contiguous_or_broadcast += nbytes
                    else:
                        dim[i].bytes_non_contiguous += nbytes
                if any(contiguous_or_broadcast):
                    result.bytes_contiguous_or_broadcast += nbytes
                else:
                    result.bytes_non_contiguous += nbytes
        if len(result.loop) > 1:
            # the first loop represent the "outside of the loop" compute which could be long lived
            result.loop = [result.loop[0] + x for x in result.loop[1:]]
        return result


@dataclasses.dataclass
class StatsForKernelType:
    """Memory usage stats that are collected for both persistent and looped kernels"""

    reads: StatsForReadsOrWrites
    writes: StatsForReadsOrWrites
    memory: StatsForReadsOrWrites

    @classmethod
    def compute(
        cls, loops: list[MemoryEstimate], estimator: MemoryEstimator
    ) -> typing.Self:
        reads = StatsForReadsOrWrites.compute(
            [loop.reads for loop in loops], estimator.symbols
        )
        writes = StatsForReadsOrWrites.compute(
            [loop.writes for loop in loops], estimator.symbols
        )
        return cls(
            reads=reads,
            writes=writes,
            memory=reads + writes,
        )


@dataclasses.dataclass
class MemoryStats:
    """Memory usage stats collected for each generated kernel"""

    persistent: StatsForKernelType
    looped: StatsForKernelType

    def get(self, persistent: bool) -> StatsForKernelType:
        return self.persistent if persistent else self.looped

    @classmethod
    def compute(cls, estimator: MemoryEstimator) -> typing.Self:
        persistent = StatsForKernelType.compute([estimator.persistent], estimator)
        if len(estimator.loops) == 1 and not (
            estimator.outside_loop and estimator.loops[0]
        ):
            looped = persistent  # loops/persistent is the same in this common case
        else:
            looped = StatsForKernelType.compute(
                [estimator.outside_loop, *estimator.loops], estimator
            )
        return cls(
            persistent=persistent,
            looped=looped,
        )
<|endoftext|>"""
Custom extern kernel codegen registry.

This module provides a registry that maps operators to their custom codegen
implementations. This allows us to keep all custom implementations in one place
and easily extend support for both Python and C++ wrappers.

To add a new custom implementation:
1. Create a codegen function with signature:
   def my_codegen(node: ir.FallbackKernel, writeline: Callable[[str], None]) -> None:
2. Register it in CUSTOM_EXTERN_KERNEL_CODEGEN with the operator name as key

Example:
    CUSTOM_EXTERN_KERNEL_CODEGEN = {
        "torch.ops.higher_order.print": CustomCodegen(
            python=generate_print_python,
            cpp=generate_print_cpp,  # Optional
        ),
    }

Usage:
    codegen = CUSTOM_EXTERN_KERNEL_CODEGEN[op]
    codegen.python(node, writeline)  # Direct attribute access
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from .. import ir

    # Type alias for codegen function signature (only used for type checking)
    CodegenFunc = Callable[[ir.FallbackKernel, Callable[[str], None]], None]


@dataclass
class CustomCodegen:
    """
    Container for custom codegen implementations.

    Attributes:
        python: Codegen function for Python wrapper (optional)
        cpp: Codegen function for C++ wrapper (optional)
    """

    python: Any = None
    cpp: Any = None


def generate_print_python(
    node: ir.FallbackKernel,
    writeline: Callable[[str], None],
) -> None:
    """
    Generate a builtin print call for the print HOP fallback (Python wrapper).

    This function generates Python code that calls the builtin print function
    with format string interpolation.

    Args:
        node: The FallbackKernel IR node representing the print HOP call.
        writeline: A function that writes a line of code to the output buffer.

    Example generated code:
        print('x = {}, y = {}'.format(buf0, buf1))
        print('x = {x}, y = {y}'.format(x=buf0, y=buf1))
        print('x = {}, y = {y}'.format(buf0, y=buf1))
    """
    codegen_args: list[str] = node.codegen_args()
    codegen_kwargs: list[str] = node.codegen_kwargs()

    # First arg is the format string
    if not codegen_args:
        raise ValueError(
            "generate_print_python requires a format string as the first positional argument"
        )
    format_str: str = codegen_args[0]

    # Remaining args are positional arguments for .format()
    positional_args = codegen_args[1:]

    args_str = ", ".join(positional_args + codegen_kwargs)
    writeline(
        f"print({format_str}.format({args_str}))"
        if args_str
        else f"print({format_str})"
    )


# Registry mapping operator names to their custom codegen implementations
# Usage: CUSTOM_EXTERN_KERNEL_CODEGEN[op_name].python(node, writeline)
CUSTOM_EXTERN_KERNEL_CODEGEN: dict[str, CustomCodegen] = {
    "torch.ops.higher_order.print": CustomCodegen(
        python=generate_print_python,
    ),
}
<|endoftext|>from collections.abc import Callable
from typing import Generic, TypeVar


T = TypeVar("T")


def _value_or(opt: T | None, default: T) -> T:
    return opt if opt is not None else default


class SegmentedTree(Generic[T]):
    def __init__(
        self,
        values: list[T],
        update_op: Callable[[T, T], T],
        summary_op: Callable[[T, T], T],
        identity_element: T,
    ):
        """
        Initialize a segment tree with the given values and operations.

        Args:
            values: list of initial values
            update_op: Function to apply when updating a value (e.g., addition)
            summary_op: Function to summarize two values (e.g., min, max, sum)
            identity_element: Identity element for the summary_op (e.g., 0 for sum, float('inf') for min)

        Raises:
            ValueError: If the input values list is empty
        """
        if not values:
            raise ValueError("Cannot create a segment tree with empty values list")

        self.n = len(values)
        self.update_op = update_op
        self.summary_op = summary_op
        self.identity = identity_element

        # Size of segment tree array (next power of 2 * 2)
        # The tree follows a standard heap layout where
        # node `n`'s children are at `2*n` and `2*n+1`.
        # Index 0 is unused.
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.size *= 2

        # Initialize tree and lazy arrays
        self.tree = [identity_element] * self.size
        # The lazy array contains updates to the given node
        # Upon update, we only push updates to the top-most
        # nodes that fully receive the update. We then
        # propagate the update down as required (i.e., when
        # we receive an interval query that neither fully
        # contains the node nor fully doesn't contain the
        # node
        self.lazy: list[T | None] = [None] * self.size

        # Build the tree
        self._build(values, 1, 0, self.n - 1)

    def _build(self, values: list[T], node: int, start: int, end: int) -> None:
        """
        Build the segment tree recursively.

        Args:
            values: Original array of values
            node: Current node index in the segment tree
            start: Start index of the segment
            end: End index of the segment
        """
        if start == end:
            # Leaf node
            if start < len(values):
                self.tree[node] = values[start]
            return

        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        # Recursively build left and right subtrees
        self._build(values, left_child, start, mid)
        self._build(values, right_child, mid + 1, end)

        # Update current node with summary of children
        self.tree[node] = self.summary_op(self.tree[left_child], self.tree[right_child])

    def _children(self, node: int) -> list[int]:
        return [2 * node, 2 * node + 1]

    def _push_lazy(self, node: int, start: int, end: int) -> None:
        """
        Push lazy updates down to children.

        Args:
            node: Current node index
            start: Start index of the segment
            end: End index of the segment
        """
        lazy_node = self.lazy[node]
        if lazy_node is None:
            return

        # Apply lazy update to current node
        self.tree[node] = self.update_op(self.tree[node], lazy_node)

        if start != end:  # Not a leaf node
            # Propagate to children
            for child in self._children(node):
                self.lazy[child] = self.update_op(
                    _value_or(self.lazy[child], self.identity), lazy_node
                )

        # Clear the lazy value
        self.lazy[node] = None

    def _update_range_helper(
        self, node: int, start: int, end: int, left: int, right: int, value: T
    ) -> None:
        """
        Helper method to update a range of values in the segment tree.

        Args:
            node: Current node index
            start: Start index of the current segment
            end: End index of the current segment
            left: Start index of the range to update
            right: End index of the range to update
            value: Value to apply to the range
        """
        # Push lazy updates before processing this node
        self._push_lazy(node, start, end)

        # No overlap
        if start > right or end < left:
            return

        # Complete overlap
        if start >= left and end <= right:
            # Apply update to current node
            self.lazy[node] = value
            self._push_lazy(node, start, end)
            return

        # Partial overlap, recurse to children
        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        self._update_range_helper(left_child, start, mid, left, right, value)
        self._update_range_helper(right_child, mid + 1, end, left, right, value)

        # Update current node based on children
        self.tree[node] = self.summary_op(self.tree[left_child], self.tree[right_child])

    def _query_range_helper(
        self, node: int, start: int, end: int, left: int, right: int
    ) -> T:
        """
        Helper method to query a range of values in the segment tree.

        Args:
            node: Current node index
            start: Start index of the current segment
            end: End index of the current segment
            left: Start index of the range to query
            right: End index of the range to query

        Returns:
            Summary value for the range
        """
        # No overlap
        if start > right or end < left:
            return self.identity

        # Push lazy updates before processing this node
        self._push_lazy(node, start, end)

        # Complete overlap
        if start >= left and end <= right:
            return self.tree[node]

        # Partial overlap, recurse to children
        mid = (start + end) // 2
        left_child = 2 * node
        right_child = 2 * node + 1

        left_result = self._query_range_helper(left_child, start, mid, left, right)
        right_result = self._query_range_helper(right_child, mid + 1, end, left, right)

        # Combine results from children
        return self.summary_op(left_result, right_result)

    def update_range(self, start: int, end: int, value: T) -> None:
        """
        Update a range of values in the segment tree.

        Args:
            start: Start index of the range to update (inclusive)
            end: End index of the range to update (inclusive)
            value: Value to apply to the range

        Raises:
            ValueError: If start > end or indices are out of bounds
        """
        if start > end:
            raise ValueError("Start index must be less than or equal to end index")

        if start < 0 or start >= self.n:
            raise ValueError(f"Start index {start} out of bounds [0, {self.n - 1}]")

        if end < 0 or end >= self.n:
            raise ValueError(f"End index {end} out of bounds [0, {self.n - 1}]")

        self._update_range_helper(1, 0, self.n - 1, start, end, value)

    def summarize_range(self, start: int, end: int) -> T:
        """
        Query a range of values in the segment tree.

        Args:
            start: Start index of the range to query (inclusive)
            end: End index of the range to query (inclusive)

        Returns:
            Summary value for the range according to the summary operation

        Raises:
            ValueError: If start > end or indices are out of bounds
        """
        if start > end:
            raise ValueError("Start index must be less than or equal to end index")

        if start < 0 or start >= self.n:
            raise ValueError(f"Start index {start} out of bounds [0, {self.n - 1}]")

        if end < 0 or end >= self.n:
            raise ValueError(f"End index {end} out of bounds [0, {self.n - 1}]")

        return self._query_range_helper(1, 0, self.n - 1, start, end)
<|endoftext|>from __future__ import annotations

from textwrap import dedent

from .common import DeviceIdx, DeviceOpOverrides, register_device_op_overrides


class CpuDeviceOpOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self, name: str) -> str:
        return dedent(
            """
            def get_raw_stream(_):
                return 0
            """
        )

    def cpp_kernel_type(self) -> str:
        return "void*"

    def set_device(self, device_idx: DeviceIdx) -> str:
        return "pass"

    def synchronize(self) -> str:
        return "pass"

    def device_guard(self, device_idx: DeviceIdx) -> str:
        return "torch._ops.contextlib.nullcontext()"


register_device_op_overrides("cpu", CpuDeviceOpOverrides())
<|endoftext|>import re

import torch


# It is not a good idea to directly apply hipify_torch to codegen, which will be vulnerable to cases like:
#   "...
#    from ..codecache import CudaKernelParamCache
#   ..."
# In such cases, we do not need to hipify_torch the original class/file name in codegen/codecache


def maybe_hipify_code_wrapper(source_codes: str, force_hipify: bool = False) -> str:
    if torch.version.hip is None and not force_hipify:
        return source_codes

    try:
        from torch.utils.hipify.hipify_python import PYTORCH_MAP, PYTORCH_TRIE
    except ImportError:
        # hipify not available for non-AMD builds
        return source_codes

    def c2_repl(m: re.Match[str]) -> object:
        return PYTORCH_MAP[m.group(0)]

    # We need to redefine RE_PYTORCH_PREPROCESSOR here since in hipify_torch,
    # it will apply positive lookbehind (?<=\W) to the pattern to avoid matching
    # keyword at the beginning of code line. However, this can happen in codegen,
    # which will cause the pattern to not match.

    # Note that lookahead (?=\W|$) is still needed to keep hipification idempotent, for example
    # we need to skip replacing "getStreamFromExternal" in "getStreamFromExternalMasqueradingAsCUDA".
    # End-of-string must also count as a boundary so bare-token inputs like "CUfunction"
    # (returned by device_codegen.cpp_kernel_type()) get hipified too.
    RE_PYTORCH_PREPROCESSOR = re.compile(rf"({PYTORCH_TRIE.export_to_regex()})(?=\W|$)")

    source_codes = RE_PYTORCH_PREPROCESSOR.sub(c2_repl, source_codes)  # type: ignore[arg-type]
    return source_codes
<|endoftext|># This is not a feature-complete compiler backend
# Just an early prototype that shows that one can compile elementwise ops into a Metal shader
from __future__ import annotations

import functools
import itertools
import logging
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

import sympy
from sympy.printing.precedence import PRECEDENCE

import torch
from torch.utils._cpp_embed_headers import _embed_headers
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Min
from torch.utils._sympy.printers import CppPrinter, ExprPrinter as ExprPrinter_
from torch.utils._sympy.value_ranges import ValueRanges

from ..utils import ceildiv, get_bounds_index_expr, get_kernel_metadata
from ..virtualized import ops, OpsWrapper, V
from .common import (
    CSEVariable,
    DeferredLine,
    DTYPE_TO_COMPUTATION_DTYPE,
    IndentedBuffer,
    OpOverrides,
    PythonPrinter,
)
from .simd import IterationRangesEntry, SIMDKernel, SIMDScheduling


if TYPE_CHECKING:
    from ..ops_handler import ReductionType, StoreMode
    from ..scheduler import Scheduler, SchedulerNode
    from .common import OpVarT

log = logging.getLogger(__name__)

DTYPE_TO_METAL = {
    torch.bool: "bool",
    torch.int8: "char",
    torch.int16: "short",
    torch.int32: "int",
    torch.int64: "long",
    torch.uint8: "uchar",
    torch.uint16: "ushort",
    torch.uint32: "uint",
    torch.uint64: "ulong",
    torch.float: "float",
    torch.half: "half",
    torch.bfloat16: "bfloat",
}


def value_to_metal(val: float | int | bool | str | CSEVariable) -> str:
    if isinstance(val, float):
        if val == torch.inf:
            return "HUGE_VALF"
        elif val == -torch.inf:
            return "-HUGE_VALF"
        elif val != val:  # Only float that not equal to self is nan
            return "NAN"
        return str(val)
    elif isinstance(val, bool):
        return "true" if val else "false"
    return str(val)


class MetalExprPrinter(ExprPrinter_):
    """Converts sympy expression to Metal code snippet"""

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        x = self.doprint(x)
        div = self.doprint(div)
        if expr.is_integer:
            return f"c10::metal::floor_divide({x}, {div})"
        return f"metal::floor({x}) / ({div})"

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = expr.args
        # Workaround for Metal compiler bug with fused (x / A) % B, see PR 175481
        use_safe_mod = div == 65536 and (mod & (mod - 1)) != 0

        x = self.doprint(x)
        if div != 1:
            div = self.doprint(div)
            if expr.is_integer:
                x = f"({x}) / ({div})"
            else:
                x = f"metal::floor({x}) / ({div})"
        mod = self.doprint(mod)
        if use_safe_mod:
            return f"c10::metal::safe_mod({x}, {mod})"
        return f"({x}) % ({mod})"

    def _print_min_max(self, expr: sympy.Expr, fn: str) -> str:
        # pyrefly: ignore [missing-attribute]
        args = list(map(self._print, expr.args))
        result = args[0]
        for arg in args[1:]:
            result = self._print_binary_min_max(result, arg, fn)
        return result

    @staticmethod
    def _print_binary_min_max(a: str, b: str, fn: str) -> str:
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"metal::{fn}({typecast_a}, {typecast_b})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        return self._print_min_max(expr, "min")

    def _print_Max(self, expr: sympy.Expr) -> str:
        return self._print_min_max(expr, "max")

    def _print_Abs(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"metal::abs({self._print(expr.args[0])})"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"static_cast<long>(metal::rint({self._print(expr.args[0])}))"

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 2:
            raise AssertionError(f"expected 2 args, got {len(expr.args)}")
        number, ndigits = expr.args
        if number.is_integer:
            # ndigits < 0 should have been filtered by the sympy function
            if ndigits >= 0:
                raise AssertionError(f"expected ndigits < 0, got {ndigits}")
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )
        number_str = self.parenthesize(number, PRECEDENCE["Mul"])
        return f"static_cast<float>(metal::rint(1e{ndigits} * {number_str}) * 1e{-ndigits})"

    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        lhs, rhs = expr.args
        # TODO: This is only accurate up to 2**23
        # pyrefly: ignore [missing-attribute]
        return f"static_cast<float>({self._print(lhs)}) / static_cast<float>({self._print(rhs)})"

    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 2:
            raise AssertionError(f"expected 2 args, got {len(expr.args)}")
        x, y = map(self.doprint, expr.args)
        return f"metal::precise::pow(static_cast<float>({x}), static_cast<float>({y}))"

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        x = self.doprint(expr.args[0])
        return f"static_cast<float>({x})"

    def _print_Float(self, expr: sympy.Expr) -> str:
        if expr.is_integer:
            # sympy considers 0.0 to be integer, but Metal doesn't.
            # this workaround prints the float as an integer
            # xref: https://github.com/sympy/sympy/issues/26620
            return str(int(expr))
        else:
            return str(expr)

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        x = self.doprint(expr.args[0])
        return f"static_cast<int>(metal::floor(static_cast<float>({x})))"

    _print_floor = _print_FloorToInt

    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        x = self.doprint(expr.args[0])
        return f"static_cast<int>(metal::trunc({x}))"

    def _print_OpaqueUnaryFn_log2(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        x = self.doprint(expr.args[0])
        return f"metal::precise::log2({x})"

    def _print_Where(self, expr: sympy.Expr) -> str:
        c, p, q = (
            self.parenthesize(arg, PRECEDENCE["Atom"] - 0.5) for arg in expr.args
        )
        return f"{c} ? {p} : {q}"


class MetalOverrides(OpOverrides):
    """Implements Metal-specific overrides for ops. Base class emits Python-friendly overrides."""

    @staticmethod
    def to_dtype(
        x: CSEVariable,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> str:
        if dtype == torch.double:
            log.warning(
                "float64 cast requested, probably from tensorify_python_scalars"
            )
            return f"static_cast<float>({x})"
        return f"static_cast<{DTYPE_TO_METAL[dtype]}>({x})"

    @staticmethod
    def to_dtype_bitcast(
        x: CSEVariable, dtype: torch.dtype, src_dtype: torch.dtype
    ) -> str:
        return f"as_type<{DTYPE_TO_METAL[dtype]}>(static_cast<{DTYPE_TO_METAL[src_dtype]}>({x}))"

    @staticmethod
    def constant(val: bool | float | int, dtype: torch.dtype) -> str:
        return value_to_metal(val)

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        idx_str = V.kernel.index_to_str(V.kernel.prepare_indexing(expr))
        var = V.kernel.cse.generate(
            V.kernel.compute, idx_str, bounds=get_bounds_index_expr(expr)
        )
        return ops.to_dtype(var, dtype)

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> str:
        # Metal index_expr already emits the requested dtype, so value_expr has
        # the same lowering here.
        return MetalOverrides.index_expr(expr, dtype)

    @staticmethod
    def masked(mask: CSEVariable, body: sympy.Expr, other: CSEVariable) -> str:
        # TODO: Type annotation for other is wrong, it's often float or int
        # TODO: Should it be converted to lambda on MacOS-15+?

        other_str = value_to_metal(other)
        scoped_body = IndentedBuffer()
        with V.kernel.swap_buffers(scoped_body), scoped_body.indent():
            # Reset the scoped variable counter so that each invocation of the same body
            # generates identical variable names. Without this reset, repeated calls to
            # body() would keep incrementing the counter, resulting in different cache key.
            V.kernel.cse.iter_buffer_ids = itertools.count()
            # Append "_scoped" to the current prefix so each nesting level gets unique vars
            V.kernel.cse.name_prefix += "_scoped"
            rc = body()

        # Compute cache key manually as variable name is needed to actually generate the code
        cache_key = f"{mask}:{scoped_body.getvalue()}:{other_str}"
        var = V.kernel.cse.try_get(cache_key)
        if not var:
            var = V.kernel.cse.newvar(dtype=rc.dtype)
            V.kernel.cse.put(cache_key, var)
            V.kernel.compute.writelines(
                [f"{DTYPE_TO_METAL[rc.dtype]} {var};", f"if ({mask}) {{"]
            )
            with V.kernel.compute.indent():
                V.kernel.compute.splice(scoped_body)
                V.kernel.compute.writeline(
                    f"{var} = static_cast<decltype({var})>({rc});"
                )
            V.kernel.compute.writeline(
                f"}} else {var} = static_cast<decltype({var})>({other_str});"
            )
        return var

    @staticmethod
    # pyrefly: ignore [bad-override]
    def where(a: OpVarT, b: OpVarT, c: OpVarT) -> str:
        return f"{a} ? {b} : static_cast<decltype({b})>({value_to_metal(c)})"

    @staticmethod
    def remainder(a: OpVarT, b: OpVarT) -> str:
        return f"c10::metal::remainder({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def maximum(a: CSEVariable, b: CSEVariable) -> str:
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"c10::metal::max({typecast_a}, {typecast_b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def minimum(a: CSEVariable, b: CSEVariable) -> str:
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"c10::metal::min({typecast_a}, {typecast_b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_or(a: CSEVariable, b: CSEVariable) -> str:
        return f"{a} || {b}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_and(a: CSEVariable, b: CSEVariable) -> str:
        return f"{a} && {b}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def isnan(x: CSEVariable) -> str:
        return f"metal::isnan({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def isinf(x: CSEVariable) -> str:
        return f"metal::isinf({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log(x: CSEVariable) -> str:
        return f"metal::precise::log({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp(x: CSEVariable) -> str:
        return f"metal::precise::exp({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def abs(x: CSEVariable) -> str:
        return f"metal::abs({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def signbit(x: CSEVariable) -> str:
        return f"metal::signbit({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sin(x: CSEVariable) -> str:
        return f"metal::precise::sin({x})"

    @staticmethod
    def sinc(x: CSEVariable) -> str:
        return f"c10::metal::sinc({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def cos(x: CSEVariable) -> str:
        return f"metal::precise::cos({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tan(x: CSEVariable) -> str:
        return f"metal::precise::tan({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def asin(x: CSEVariable) -> str:
        return f"metal::precise::asin({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def acos(x: CSEVariable) -> str:
        return f"metal::precise::acos({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan(x: CSEVariable) -> str:
        return f"metal::precise::atan({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan2(x: CSEVariable, y: CSEVariable) -> str:
        return f"::metal::precise::atan2({x}, {y})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sqrt(x: CSEVariable) -> str:
        return f"metal::precise::sqrt({x})"

    @staticmethod
    def neg(x: CSEVariable) -> str:
        # TODO: Does it rely on undefined behavior?
        # If so, add special logic for unsigned types
        return f"static_cast<decltype({x})>(-{x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def rsqrt(x: CSEVariable) -> str:
        return f"metal::precise::rsqrt({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tanh(x: CSEVariable) -> str:
        return f"metal::precise::tanh({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atanh(x: CSEVariable) -> str:
        return f"metal::precise::atanh({x})"

    @staticmethod
    def floordiv(a: CSEVariable, b: CSEVariable) -> str:
        # a and b must be of integer type
        return f"c10::metal::floor_divide({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def floor(x: CSEVariable) -> str:
        return f"metal::floor({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sign(x: CSEVariable) -> str:
        return f"metal::sign({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def fmod(a: CSEVariable, b: CSEVariable) -> str:
        typecast_a = f"static_cast<decltype({a}+{b})>({a})"
        typecast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"metal::fmod({typecast_a}, {typecast_b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def trunc(x: CSEVariable) -> str:
        return f"metal::trunc({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def truncdiv(a: CSEVariable, b: CSEVariable) -> str:
        quot = f"{a} / {b}"
        if (a.dtype is not None and a.dtype.is_floating_point) or (
            b.dtype is not None and b.dtype.is_floating_point
        ):
            return f"metal::trunc({quot})"
        return quot

    @staticmethod
    # pyrefly: ignore [bad-override]
    def ceil(x: CSEVariable) -> str:
        return f"metal::ceil({x})"

    @staticmethod
    def rand(seed: CSEVariable, offset: CSEVariable) -> str:
        V.kernel.headers.add("random")
        return f"c10::metal::rand({seed}, {offset})"

    @staticmethod
    def randn(seed: CSEVariable, offset: CSEVariable) -> str:
        V.kernel.headers.add("random")
        return f"c10::metal::randn({seed}, {offset})"

    @staticmethod
    def randint64(
        seed: CSEVariable, offset: CSEVariable, low: CSEVariable, high: CSEVariable
    ) -> str:
        V.kernel.headers.add("random")
        return f"c10::metal::randint64({seed}, {offset}, {low}, {high})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def round(x: CSEVariable) -> str:
        return f"metal::rint({x})"

    @staticmethod
    def pow(a: CSEVariable, b: CSEVariable) -> str:
        cast_a = f"static_cast<decltype({a}+{b})>({a})"
        cast_b = f"static_cast<decltype({a}+{b})>({b})"
        return f"metal::pow({cast_a}, {cast_b})"

    def _special_unary(self, a: CSEVariable, name: str) -> str:
        V.kernel.headers.add("special_math")
        return f"c10::metal::{name}({a})"

    def _special_binary(self, a: CSEVariable, b: CSEVariable, name: str) -> str:
        V.kernel.headers.add("special_math")
        return f"c10::metal::{name}({a}, {b})"

    @classmethod
    def _initialize_special_ops(cls) -> None:
        # Unary special ops
        for name in [
            "erf",
            "erfc",
            "erfinv",
            "i0",
            "i0e",
            "i1",
            "i1e",
            "digamma",
            "spherical_bessel_j0",
        ]:
            setattr(cls, name, functools.partialmethod(cls._special_unary, name=name))

        cls.lgamma = functools.partialmethod(cls._special_unary, name="log_gamma")  # type: ignore[assignment]

        # Unary special ops with forward in method name
        for name in [
            "bessel_j0",
            "bessel_j1",
            "bessel_y0",
            "bessel_y1",
            "modified_bessel_i0",
            "modified_bessel_i1",
            "modified_bessel_k0",
            "modified_bessel_k1",
            "scaled_modified_bessel_k0",
            "scaled_modified_bessel_k1",
        ]:
            setattr(
                cls,
                name,
                functools.partialmethod(cls._special_unary, name=name + "_forward"),
            )

        # Binary special ops
        for name in [
            "polygamma",
            "igamma",
            "igammac",
            "zeta",
        ]:
            setattr(cls, name, functools.partialmethod(cls._special_binary, name=name))

        # Binary special ops with forward in method name
        for name in [
            "chebyshev_polynomial_t",
            "chebyshev_polynomial_u",
            "chebyshev_polynomial_v",
            "chebyshev_polynomial_w",
            "hermite_polynomial_h",
            "hermite_polynomial_he",
            "shifted_chebyshev_polynomial_t",
            "shifted_chebyshev_polynomial_u",
            "shifted_chebyshev_polynomial_v",
            "shifted_chebyshev_polynomial_w",
        ]:
            setattr(
                cls,
                name,
                functools.partialmethod(cls._special_binary, name=name + "_forward"),
            )


MetalOverrides._initialize_pointwise_overrides("mps")
MetalOverrides._initialize_special_ops()


class MetalKernel(SIMDKernel):
    """Implement Metal codegen based on the SIMDKernel abstraction"""

    overrides = MetalOverrides  # type: ignore[assignment]
    suffix = ";"
    newvar_prefix = "auto "
    max_threadgroup_size = 1024
    simd_group_size = 32
    pexpr = PythonPrinter().doprint
    cexpr = CppPrinter().doprint
    sexpr = MetalExprPrinter().doprint
    kexpr = sexpr
    headers: OrderedSet[str] = OrderedSet(["utils"])
    multistage_reduction_entry: list[IterationRangesEntry] = []

    def __init__(
        self,
        tiling: dict[str, sympy.Expr],
        **kwargs: Any,
    ) -> None:
        super().__init__(tiling, **kwargs)
        self.acc_var_ids = itertools.count()

    def dtype_to_str(self, dtype: torch.dtype) -> str:
        return DTYPE_TO_METAL[dtype]

    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        """Codegen a load from an InputBuffer"""
        var = self.args.input(name)
        index = self.prepare_indexing(index)
        dtype = V.graph.get_dtype(name)
        line = f"{var}[{self.index_to_str(index)}]"
        if dtype in [torch.float16, torch.bfloat16]:
            # TODO(NS): Figure out the right balance between optype casts
            # op_math_t for half-precision floats should be float32
            # Otherwise it can lead to a correctness issues with eager
            line = f"static_cast<float>({line})"
            dtype = torch.float32
        return self.cse.generate(self.loads, line, dtype=dtype)

    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> None:
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        dtype_str = self.dtype_to_str(V.graph.get_dtype(name))
        cast_val = f"static_cast<{dtype_str}>({value})"
        if mode is None:
            line = f"{var}[{self.index_to_str(index)}] = {cast_val};"
        elif mode == "atomic_add":
            self.headers.add("atomic")
            atomic_type = f"c10::metal::AtomicType<{dtype_str}>"
            cast_var = f"reinterpret_cast<device {atomic_type}::type *>({var})"
            line = f"{atomic_type}::atomic_add({cast_var}, {self.index_to_str(index)}, {cast_val});"
        else:
            raise RuntimeError(f"Unimplemented store mode {mode}")
        if self.inside_reduction:
            self.compute.writeline(DeferredLine(name, line))
        else:
            self.stores.writeline(DeferredLine(name, line))

    def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
        var = self.args.output(name)
        index = self.prepare_indexing(index)
        dtype_str = self.dtype_to_str(V.graph.get_dtype(name))

        reduction_dim = next(t for t in self.range_trees if t.is_reduction)
        # Only one thread in the reduction group needs to store the results
        line = f"{var}[{self.index_to_str(index)}] = static_cast<{dtype_str}>({value});"
        line = f"if ({reduction_dim.name} == 0) {line}"
        self.stores.writeline(DeferredLine(name, line))

    def _new_idxvar(
        self,
        dtype: str | torch.dtype,
        elem_count: int | None = None,
        default_value: Any | None = None,
        is_threadgroup: bool = True,
        bounds: ValueRanges[Any] = ValueRanges.unknown(),
    ) -> CSEVariable:
        if isinstance(dtype, torch.dtype):
            dtype = self.dtype_to_str(dtype)
        var_name = f"tmp_acc_{next(self.acc_var_ids)}"
        var = V.kernel.create_cse_var(var_name, bounds, dtype)
        var_def = "threadgroup " if is_threadgroup else ""
        var_def += f"{dtype} {var_name}"
        if elem_count:
            var_def += f"[{self.sexpr(elem_count)}]"
        if default_value is not None:
            if is_threadgroup:
                raise AssertionError("Thread group var can not have default value")
            var_def += f" = {default_value}"
        self.indexing_code.writeline(var_def + self.suffix)
        return var

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        "Caching wrapper around _reduction_nocache"
        cache_key = (src_dtype, reduction_type, value)
        # Return cached reduction
        if cache_key in self.cse.reduction_cache:
            return self.cse.reduction_cache[cache_key]
        result = self._reduction_nocache(dtype, src_dtype, reduction_type, value)
        self.cse.reduction_cache[cache_key] = result  # type: ignore[assignment]
        return result

    def _reduction_nocache(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: CSEVariable | tuple[CSEVariable, ...],
    ) -> CSEVariable | tuple[CSEVariable, ...]:
        """Codegen a reduction operation.
        Only sum and prod operations are somewhat reasonable optimized"""
        if not self.inside_reduction:
            raise AssertionError("expected to be inside reduction")
        if self._load_mask:
            raise AssertionError("expected no load mask during reduction")

        def _unwrap_helper(res3: CSEVariable) -> tuple[CSEVariable, ...]:
            # Unwraps vec3 dtype into individual components
            return OpsWrapper._unwrap(
                [CSEVariable(f"{res3}.{t}", res3.bounds, res3.dtype) for t in "xyz"]
            )

        # Establish reduction buffer size and index expression
        reduction_idx = ""
        acc_buf_size = 1
        for rd in self.range_trees:
            if not rd.is_reduction:
                continue
            if reduction_idx:
                reduction_idx += " + "
            reduction_idx += f"{rd.name} * {acc_buf_size}"

            if isinstance(rd.numel, sympy.Integer):
                acc_buf_size *= rd.numel
            else:
                acc_buf_size *= sympy.Symbol(
                    f"{rd.prefix}numel", integer=True, positive=True
                )

        acc_buf_size = Min(acc_buf_size, self.max_threadgroup_size)
        acc_buf_size_str = self.sexpr(acc_buf_size)
        # metal threadgroup arrays need a compile time constant size, so
        # fall back to the static upper bound when acc buf size is symbolic
        # happens when dynamic=True
        acc_buf_alloc_size = (
            acc_buf_size
            if isinstance(acc_buf_size, sympy.Integer)
            else self.max_threadgroup_size
        )
        shmem_buf_size = (
            ceildiv(acc_buf_size, self.simd_group_size)
            if isinstance(acc_buf_size, sympy.Integer)
            else self.simd_group_size
        )

        if reduction_type == "any":
            acc = self._new_idxvar(dtype)
            self.indexing_code.writeline(f"{acc} = false;")
            self.indexing_code.writeline(
                "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
            )
            self.compute.splice(
                f"""
                if ({value}) {{
                    {acc} = true;
                }}
            """
            )
            self.stores.writeline(
                "threadgroup_barrier(metal::mem_flags::mem_threadgroup);"
            )
            return acc

        self.headers.add("reduction_utils")

        if reduction_type in ["prod", "sum"]:
            acc_dtype = DTYPE_TO_COMPUTATION_DTYPE[src_dtype]
            acc_buf = self._new_idxvar(acc_dtype, shmem_buf_size)
            if not self.multistage_reduction_entry:
                val = value
            else:
                default_val, reduction_op = (
                    (0, "+") if reduction_type == "sum" else (1, "*")
                )
                val = self._new_idxvar(
                    acc_dtype, default_value=default_val, is_threadgroup=False
                )
                self.compute.splice(f"{val} {reduction_op}= {value};")

            return self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_{reduction_type}({acc_buf}, {val}, {reduction_idx}, {acc_buf_size_str})",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[dtype],
            )
        if reduction_type in ["max", "min"]:
            acc_buf = self._new_idxvar(src_dtype, shmem_buf_size)
            src_metal_type = DTYPE_TO_METAL[src_dtype]
            cast_value = f"static_cast<{src_metal_type}>({value})"
            if not self.multistage_reduction_entry:
                val = cast_value  # type: ignore[assignment]
            else:
                op_struct = "MaxOp" if reduction_type == "max" else "MinOp"
                limit_val = f"::c10::metal::{op_struct}<{src_metal_type}>::identity()"
                val = self._new_idxvar(
                    src_dtype, default_value=limit_val, is_threadgroup=False
                )
                self.compute.splice(
                    f"{val} = ::c10::metal::{reduction_type}({val}, {cast_value});"
                )
            return self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_{reduction_type}({acc_buf}, {val}, {reduction_idx}, {acc_buf_size_str})",
                dtype=DTYPE_TO_COMPUTATION_DTYPE[dtype],
            )
        if reduction_type in ["argmin", "argmax"]:
            data_acc_buf = self._new_idxvar(src_dtype, shmem_buf_size)
            idx_acc_buf = self._new_idxvar(dtype, shmem_buf_size)
            src_metal_type = DTYPE_TO_METAL[src_dtype]
            cast_value = f"static_cast<{src_metal_type}>({value})"
            if not self.multistage_reduction_entry:
                val = cast_value  # type: ignore[assignment]
                idx_val = f"static_cast<{DTYPE_TO_METAL[dtype]}>({reduction_idx})"
            else:
                op_struct = "MaxOp" if reduction_type == "argmax" else "MinOp"
                limit_val = f"::c10::metal::{op_struct}<{src_metal_type}>::identity()"
                val = self._new_idxvar(
                    src_dtype, default_value=limit_val, is_threadgroup=False
                )
                idx_val = self._new_idxvar(dtype, default_value=0, is_threadgroup=False)  # type: ignore[assignment]
                idx_var = next(
                    t for t in self.range_tree_nodes.values() if t.is_reduction
                )
                self.compute.splice(f"""
                if (::c10::metal::{op_struct}<{src_metal_type}>::replace({cast_value}, {val})) {{
                    {val} = {cast_value};
                    {idx_val} = {idx_var.name};
                }}
                """)
            return self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_{reduction_type}({data_acc_buf}, {idx_acc_buf}, "
                f"{val}, {idx_val}, {reduction_idx}, {acc_buf_size_str})",
                dtype=dtype,
            )
        if reduction_type == "welford_reduce":
            if not self.multistage_reduction_entry:
                acc_buf = self._new_idxvar("float3", acc_buf_alloc_size)
                self.compute.splice(
                    f"{acc_buf}[{reduction_idx}] = float3({value}, 0.0, 1.0);"
                )
                wf_res = self.cse.generate(
                    self.compute,
                    f"c10::metal::threadgroup_welford_combine({acc_buf}, {reduction_idx}, {acc_buf_size_str})",
                    dtype=torch.float32,
                )
                return _unwrap_helper(wf_res)
            acc_buf = self._new_idxvar("float3", acc_buf_alloc_size)
            acc_thread_var = f"{acc_buf}[{reduction_idx}]"
            self.indexing_code.splice(f"{acc_thread_var} = 0.0;")
            self.compute.writeline(
                f"{acc_thread_var} = ::c10::metal::welford_combine({acc_thread_var}, float3({value}, 0.0, 1.0));"
            )
            wf_res = self.cse.generate(
                self.stores,
                f"c10::metal::threadgroup_welford_combine({acc_buf}, {reduction_idx}, {acc_buf_size_str})",
                dtype=torch.float32,
            )
            return _unwrap_helper(wf_res)
        if reduction_type == "welford_combine":
            if not isinstance(value, tuple):
                raise AssertionError("Input to welford combine must be tuple")
            acc_buf = self._new_idxvar("float3", acc_buf_alloc_size)
            acc_thread_var = f"{acc_buf}[{reduction_idx}]"
            inp_value = f"float3({value[0]}, {value[1]}, {value[2]})"
            self.indexing_code.splice(f"{acc_thread_var} = 0.0;")
            if self.multistage_reduction_entry:
                self.indexing_code.splice(f"{acc_thread_var} = 0.0;")
                self.compute.writeline(
                    f"{acc_thread_var} = ::c10::metal::welford_combine({acc_thread_var}, {inp_value});"
                )
            else:
                self.compute.writeline(f"{acc_thread_var} = {inp_value};")
            wf_res = self.cse.generate(
                self.stores if self.multistage_reduction_entry else self.compute,
                f"c10::metal::threadgroup_welford_combine({acc_buf}, {reduction_idx}, {acc_buf_size_str})",
                dtype=torch.float32,
            )
            return _unwrap_helper(wf_res)
        raise NotImplementedError(reduction_type)

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry) -> None:
        index_expr = self.rename_indexing(entry.expr)
        index_str = self.sexpr(index_expr)  # type: ignore[misc]

        if not entry.is_reduction or (
            isinstance(entry.root.numel, sympy.Integer)
            and entry.root.numel <= self.max_threadgroup_size
        ):
            self.indexing_code.writeline(
                f"{self.index_dtype} {entry.name} = {index_str};"
            )
            return

        acc_size = (
            entry.root.numel
            if isinstance(entry.root.numel, sympy.Integer)
            else sympy.Symbol(f"{entry.root.prefix}numel", integer=True, positive=True)
        )

        # Check if we've already generated a loop for this reduction root
        root_already_processed = any(
            e.root is entry.root for e in self.multistage_reduction_entry
        )

        linear_idx_name = f"{entry.root.prefix}_linear_idx"

        if not root_already_processed:
            self.multistage_reduction_entry.append(entry)
            # When reducing the tensor whose size exceeds max threadgroup size
            # loop over extra indices per reduction thread and perform part of the operation
            # using values in the shared memory

            # Use floats so that it doesn't do integer division
            loop_size = (acc_size + float(self.max_threadgroup_size - 1)) // float(
                self.max_threadgroup_size
            )
            loop_size_str = self.sexpr(loop_size)

            root_name = entry.root.name

            self.body.writeline(
                f"for(auto {entry.root.prefix}_cnt = 0; {entry.root.prefix}_cnt < {loop_size_str}; ++{entry.root.prefix}_cnt) {{"
            )
            with self.body.indent():
                if isinstance(acc_size, sympy.Symbol):
                    self.body.writeline(
                        f"{self.index_dtype} {linear_idx_name} = "
                        f"{self.max_threadgroup_size} * {entry.root.prefix}_cnt + {root_name};"
                    )
                else:
                    self.body.writeline(
                        f"{self.index_dtype} {linear_idx_name} = {loop_size_str} * {root_name} + {entry.root.prefix}_cnt;"
                    )

                # Check that reduction is performed only within tensor boundary
                if (
                    isinstance(acc_size, sympy.Symbol)
                    or loop_size * self.max_threadgroup_size != acc_size
                ):
                    self.body.writeline(f"if ({linear_idx_name} >= {acc_size}) break;")

                # Compute entry value from linear index by substituting root name
                sub_index_str = index_str.replace(entry.root.name, linear_idx_name)
                self.body.writeline(
                    f"{self.index_dtype} {entry.name} = {sub_index_str};"
                )
        else:
            # root is already processed so just need to compute this entry's value inside the existing loop
            with self.body.indent():
                sub_index_str = index_str.replace(entry.root.name, linear_idx_name)
                self.body.writeline(
                    f"{self.index_dtype} {entry.name} = {sub_index_str};"
                )

    def codegen_body(self) -> None:
        """
        Concat output code from index_code, loads, compute, stores,
        suffix into self.body.

        For pointwise kernels, this is called just once at the end.

        For reduction kernels, this generates a loop over the reduction
        axis.
        """
        if self.multistage_reduction_entry:
            with self.body.indent():
                self.body.splice(self.loads)
                self.body.splice(self.compute)
            self.body.writeline("}" * len(self.multistage_reduction_entry))
            # Invalidate variables instantiated inside loop
            # But results of reduction alive. Reduction cache values can be
            # either CSEVariable or tuple of CSEVariables, in which case all
            # variables in the tuple must be preserved
            self.cse.invalidate(
                OrderedSet(
                    v
                    for item in self.cse.reduction_cache.values()
                    for v in (item if isinstance(item, tuple) else (item,))
                )
            )
            # And loop codegen
            roots = [e.root for e in self.multistage_reduction_entry]
            self.multistage_reduction_entry.clear()
            for node in self.range_tree_nodes.values():
                if any(node.root is r for r in roots):
                    node.cache_clear()
        else:
            self.body.splice(self.loads)
            self.body.splice(self.compute)
        self.body.splice(self.stores)
        self.loads.clear()
        self.compute.clear()
        self.stores.clear()

    def codegen_kernel(self, name: str = "generated_kernel") -> str:
        """Called at the end to generate a final kernel string"""
        self.codegen_body()
        code = IndentedBuffer()
        fn_name = name

        if V.graph.cpp_wrapper:
            code.writeline('(R"MTL(')
        else:
            code.writeline("compile_mps_shader('''")

        idx_vars = self.active_range_trees()
        with code.indent():
            if not V.graph.cpp_wrapper:
                for header in self.headers:
                    code.writeline(f"#include <c10/metal/{header}.h>")
            else:
                headers = [
                    f"#include <c10/metal/{header}.h>" for header in self.headers
                ]
                header_contents = _embed_headers(
                    headers,
                    [Path(__file__).parent.parent.parent / "include"],
                    OrderedSet(),  # type: ignore[arg-type]
                )
                code.writeline(header_contents)

            if self.inside_reduction:
                total_reduction_size = math.prod(
                    t.numel for t in self.range_trees if t.is_reduction
                )
                # If using dynamic shapes, set the threadgroup size to be the
                # max possible size
                threadgroup_size = (
                    min(total_reduction_size, self.max_threadgroup_size)
                    if isinstance(total_reduction_size, sympy.Integer)
                    else self.max_threadgroup_size
                )
                code.writeline(
                    f"[[max_total_threads_per_threadgroup({threadgroup_size})]]"
                )
            code.writeline(f"kernel void {fn_name}(")
            with code.indent():
                for outer, inner in self.args.output_buffers.items():
                    if outer in self.removed_buffers:
                        continue
                    dtype_str = self.dtype_to_str(V.graph.get_dtype(outer))
                    code.writeline(f"device {dtype_str}* {inner},")
                for outer, inner in self.args.input_buffers.items():
                    dtype = V.graph.get_dtype(outer)
                    # MPS does not support float64, but scalar inputs are fine
                    if dtype == torch.float64:
                        outer_buf = V.graph.try_get_buffer(outer)
                        if outer_buf is None or outer_buf.get_size() != []:
                            raise RuntimeError("float64 is not supported by MPS")
                        dtype_str = "float"
                    else:
                        dtype_str = self.dtype_to_str(dtype)
                    code.writeline(f"constant {dtype_str}* {inner},")
                for inner in self.args.sizevars.values():
                    code.writeline(f"constant long& {inner},")

                # Write dynamic values as inputs
                for idx_var in idx_vars:
                    if isinstance(idx_var.numel, sympy.Integer):
                        pass
                    else:
                        code.writeline(f"constant long& {idx_var.prefix}numel,")

                # Add error buffer parameter if error header is used
                if "error" in self.headers:
                    code.writeline("device c10::metal::ErrorMessages* error_buf,")

                if len(idx_vars) >= 4:
                    raise AssertionError("Up to 3 index variables are supported")
                thread_pos_dtype = (
                    f"uint{len(idx_vars)}" if len(idx_vars) > 1 else "uint"
                )
                thread_pos_var_name = (
                    idx_vars[0].name if len(idx_vars) == 1 else "thread_pos"
                )
                thread_pos_suffix = "," if self.inside_reduction else ""
                code.writeline(
                    f"{thread_pos_dtype} {thread_pos_var_name} [[thread_position_in_grid]]{thread_pos_suffix}"
                )
                if self.inside_reduction:
                    code.writeline(
                        f"{thread_pos_dtype} group_pos [[thread_position_in_threadgroup]]"
                    )
            code.writeline(") {")
            with code.indent():
                if len(idx_vars) > 1:
                    for idx, var in enumerate(idx_vars):
                        code.writeline(
                            f"auto {var.name} = thread_pos.{chr(120 + idx)};"
                        )
                code.splice(self.indexing_code)
                code.splice(self.body)
            code.writeline("}")

        if V.graph.cpp_wrapper:
            code.writeline(')MTL");')
        else:
            code.writeline("''')")

        return code.getvalue()

    def call_kernel(
        self, name: str, node: Any = None, deallocate_ws: bool = True
    ) -> None:
        """
        Codegens a call to this kernel
        """
        wrapper = V.graph.wrapper_code
        # Make sure sizevars has been computed
        for v in self.args.sizevars:
            wrapper.ensure_size_computed(v)

        _, call_args, _, arg_types = self.args.python_argdefs()
        arg_name_to_type = {
            str(call_arg): arg_type for call_arg, arg_type in zip(call_args, arg_types)
        }

        args = [*self.args.output_buffers.keys(), *self.args.input_buffers.keys()]
        args = [arg for arg in args if arg not in self.removed_buffers]
        args += [str(v) for v in self.args.sizevars]
        arg_types = [arg_name_to_type[arg] for arg in args]

        # Add any dynamic ints as inputs
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, int)):
                # Don't need to pass in integers as inputs
                continue
            elif isinstance(tree.numel, sympy.Symbol):
                expr = tree.numel
            else:
                expr = V.graph.wrapper_code.generate_numel_expr(name, tree).inner

            if not tree.is_reduction or self.inside_reduction:
                args.append(str(expr))
                arg_types.append(int)

        expr_printer = self.cexpr if V.graph.cpp_wrapper else self.pexpr

        def format_threads(threads: list[str], kwarg: str) -> str:
            if V.graph.cpp_wrapper:
                threads = [f"static_cast<uint64_t>({t})" for t in threads]
                return f"{{{', '.join(threads)}}}"
            else:
                return f"{kwarg}=[{', '.join(threads)}]"

        # For reduction kernels, limit the maximum size over reduction dimensions to
        # a maximum threadgroup size
        if len(self.active_range_trees()) > 0:
            threads = [
                expr_printer(
                    Min(v.numel, self.max_threadgroup_size)  # type: ignore[misc]
                    if v.is_reduction
                    else v.numel
                )
                for v in self.active_range_trees()
            ]

            args.append(format_threads(threads, "threads"))
            arg_types.append(list)
        else:
            if V.graph.cpp_wrapper:
                raise RuntimeError("We should always have threads?")

        if self.inside_reduction:
            threads = [
                expr_printer(Min(v.numel, self.max_threadgroup_size))  # type: ignore[misc]
                if v.is_reduction
                else "1"
                for v in self.active_range_trees()
            ]
            args.append(format_threads(threads, "group_size"))
            arg_types.append(list)
        else:
            if V.graph.cpp_wrapper:
                # Add a None so that we always have a group_size in the
                # arguments. We won't use it if the value is None.
                args += [None]  # type: ignore[list-item]
                arg_types.append(None)

        # Add error buffer index if error reporting is used
        # TODO(malfet) Figure out how to do it for aoti
        if "error" in self.headers and not V.graph.cpp_wrapper:
            args.append(
                f"error_buf_idx={len([arg for arg in args if arg is not None and '=' not in arg])}"
            )

        wrapper.generate_kernel_call(
            name,
            args,
            device=torch.device("mps"),
            triton=False,
            arg_types=arg_types,
        )

    def device_assert_async(self, cond: CSEVariable, msg: str) -> None:
        if V.graph.cpp_wrapper:
            self.cse.generate(self.compute, f"if (!{cond}) return", assignment=False)
        else:
            self.headers.add("error")
            self.compute.writelines(
                [
                    f"if (!{cond}) {{",
                    f"    TORCH_REPORT_ERROR(error_buf, {repr(msg)});",
                    "    return;",
                    "}",
                ]
            )

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        if not (lower or upper):
            return

        expr_str = self.index_to_str(expr)
        size_str = self.index_to_str(size)

        # Generate bounds checking with error reporting
        # TODO(malfet): Is upper bound inclusive or exclusive?
        if lower and upper:
            # Check both lower and upper bounds
            condition = f"({expr_str} < 0 || {expr_str} >= {size_str})"
        elif lower:
            condition = f"{expr_str} < 0"
        else:
            condition = f"{expr_str} >= {size_str}"

        # Generate error reporting code
        if V.graph.cpp_wrapper:
            self.cse.generate(
                self.compute, f"if ({condition}) return", assignment=False
            )
        else:
            # Add error header for error reporting
            self.headers.add("error")
            self.compute.writelines(
                [
                    f"if ({condition}) {{",
                    f'    TORCH_REPORT_ERROR(error_buf, "Index ", {expr_str}, " out of range [0, ", {size_str}, ")");',
                    "    return;",
                    "}",
                ]
            )


class MetalScheduling(SIMDScheduling):
    kernel_type = MetalKernel  # type: ignore[assignment]

    def __init__(self, scheduler: Scheduler | None) -> None:
        super().__init__(scheduler)

    def define_kernel(
        self, src_code: str, node_schedule: list[SchedulerNode], kernel: MetalKernel
    ) -> str:
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        if V.graph.cpp_wrapper:
            # C++ path: one library per kernel (each has a single "generated_kernel" function)
            mps_lib_name = f"mps_lib_{wrapper.next_kernel_suffix()}"
            kernel_name = mps_lib_name
            wrapper.src_to_kernel[src_code] = kernel_name
            src_code = f"const char* {mps_lib_name}_source = " + src_code
            origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
            wrapper.define_kernel(
                mps_lib_name, src_code, f"{origins}\n{detailed_origins}", gpu=False
            )
            return kernel_name

        # Python path: register kernel with async_compile; wait() will compile all
        # accumulated Metal kernels into a single library and replace each placeholder.
        fn_name = f"generated_kernel_{wrapper.next_kernel_suffix()}"
        wrapper.src_to_kernel[src_code] = fn_name

        # Extract Metal source from compile_mps_shader('''...''') call
        metal_src_start = "compile_mps_shader('''"
        start = src_code.index(metal_src_start) + len(metal_src_start)
        end = src_code.rindex("''')")
        metal_src = src_code[start:end]

        # Strip #include lines and rename the kernel function
        body_lines = []
        for line in metal_src.split("\n"):
            if line.strip().startswith("#include"):
                continue
            body_lines.append(
                line.replace("kernel void generated_kernel(", f"kernel void {fn_name}(")
            )

        headers_repr = repr(sorted(kernel.headers))
        wrapper.header.writeline(f"{fn_name} = async_compile.metal({fn_name!r}, '''")
        for line in body_lines:
            wrapper.header.writeline(line)
        wrapper.header.writeline(f"''', {headers_repr})")

        return fn_name
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import collections
import dataclasses
import itertools
import pprint
from typing import Any, Protocol, TYPE_CHECKING

import torch
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import Max

from .. import config
from ..utils import _align, align, cache_on_self, CachedMethod, IndentedBuffer
from ..virtualized import V
from .wrapper import (
    AllocateLine,
    BufferLike,
    FreeIfNotReusedLine,
    MemoryPlanningLine,
    NullLine,
    ReuseLine,
)


if TYPE_CHECKING:
    from collections.abc import Iterable

    import sympy


@dataclasses.dataclass
class LiveRange:
    """
    A range where a given tensor is live.  Begin and end are both counters
    representing points in the program of grouped memory operations.
    Begin is inclusive, end is exclusive.

    Invariant: begin <= end
    """

    begin: float  # int | +/-inf
    end: float  # int | +/-inf

    def contains(self, other: LiveRange):
        """Is other entirely within self"""
        return self.begin <= other.begin and other.end <= self.end

    def join(self, other: LiveRange):
        """Combine two ranges using a union operation"""
        return LiveRange(min(self.begin, other.begin), max(self.end, other.end))

    def __len__(self):
        return self.end - self.begin


class LiveRanges:
    """
    A collection of LiveRange regions, allowing for non-contiguous
    live regions.

    Invariant: LiveRanges.ranges is in sorted order and non-overlapping
    """

    def __init__(self, ranges: Iterable[LiveRange]):
        ranges = [*sorted(ranges, key=lambda x: x.begin)]
        self.ranges = ranges[:1]
        for r in ranges[1:]:
            if self.ranges[-1].begin > r.begin:
                raise AssertionError("ranges must be sorted by begin")
            if self.ranges[-1].end >= r.begin:
                self.ranges[-1] = LiveRange.join(self.ranges[-1], r)
            else:
                self.ranges.append(r)

    def overlaps(self, other: LiveRanges):
        """Check if any pair of ranges in self and other overlap"""
        left = collections.deque(self.ranges)
        right = collections.deque(other.ranges)
        while left and right:
            if left[0].begin > right[0].begin:
                left, right = right, left
            if left[0].begin > right[0].begin:
                raise AssertionError("left should begin no later than right")
            if left[0].end > right[0].begin:
                return True
            left.popleft()
        return False

    @property
    def begin(self):
        return self.ranges[0].begin

    @property
    def end(self):
        return self.ranges[-1].end

    def __repr__(self):
        return f"{self.__class__.__name__}([{', '.join(map(repr, self.ranges))}])"


class AllocationTreeNode:
    """
    Abstract base class for nodes in allocation pool.
    """

    def allocate(self, block: Allocation, is_last: bool) -> bool:
        """
        Try to assign block to a memory location in this pool.  Return True if
        an assignment was made.
        """
        return False

    def get_live_ranges(self) -> LiveRanges:
        """Aggregate LiveRanges for all objects below this in tree"""
        raise NotImplementedError

    def get_size_hint(self) -> int:
        """Number of bytes used for example inputs"""
        raise NotImplementedError

    def get_symbolic_size(self) -> sympy.Expr:
        """Number of bytes needed at runtime"""
        raise NotImplementedError

    def finalize(self, pool, offset) -> AllocationTreeNode:
        """Called after all allocations have been made"""
        return self

    def is_empty(self):
        return False


@dataclasses.dataclass
class Allocation(AllocationTreeNode):
    """
    Represents memory allocated to a given node in the allocation pool.
    """

    node: BufferLike
    live_range: LiveRange
    size_hint: int
    symbolic_size: sympy.Expr
    allocated: bool = False
    pool: AllocationPool | None = None
    offset: sympy.Expr | None = None
    earliest_available: float | None = None

    def __post_init__(self) -> None:
        has_unbacked_sym = False
        for s in self.node.get_layout().size:
            if free_unbacked_symbols(s):
                has_unbacked_sym = True
                break

        if has_unbacked_sym:
            self.earliest_available = self.get_live_ranges().begin

    @property
    def device(self):
        return self.node.get_device()

    def get_live_ranges(self):
        return LiveRanges([self.live_range])

    def get_size_hint(self):
        return self.size_hint

    def get_symbolic_size(self):
        return self.symbolic_size

    def mark_allocated(self):
        if self.allocated:
            raise AssertionError("block already allocated")
        self.allocated = True

    def finalize(self, pool, offset):
        if not (self.pool is None and self.offset is None):
            raise AssertionError("block already finalized")
        self.pool = pool
        self.offset = offset
        return self

    def codegen_alloc_from_pool(self, wrapper):
        if not self.pool:
            raise AssertionError("block has no pool assigned")
        node = self.node
        shape = tuple(node.get_size())
        stride = tuple(node.get_stride())
        return wrapper.codegen_alloc_from_pool(
            self.pool.name, self.offset, node.get_dtype(), shape, stride
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"node={self.node.get_name()}, "
            f"live_range={self.live_range}, "
            f"size_hint={self.size_hint}, "
            f"symbolic_size={self.symbolic_size}, "
            f"pool={self.pool.name if self.pool else None}, "
            f"offset={self.offset})"
        )

    def get_earliest_available(self):
        return self.earliest_available


@dataclasses.dataclass
class Empty(AllocationTreeNode):
    """
    Placeholder to represent empty space in the allocation pool.
    Only exists to get the size_hint correct in parent nodes.
    """

    size_hint: int

    def get_live_ranges(self):
        return LiveRanges([])

    def get_size_hint(self):
        return self.size_hint

    def get_symbolic_size(self):
        return 0

    def is_empty(self):
        return True


class MemorySplitProtocol(Protocol):
    get_live_ranges: CachedMethod[[], LiveRanges]
    get_size_hint: CachedMethod[[], int]
    get_symbolic_size: CachedMethod[[], sympy.Expr]

    def _allocate(self, block: Allocation, is_last: bool) -> bool: ...


class ClearCacheOnAllocateMixin(MemorySplitProtocol):
    """
    Helper to assist in caching get_live_ranges, get_size_hint, and
    get_symbolic_size.
    """

    def allocate(self, block: Allocation, is_last: bool):
        is_allocated = self._allocate(block, is_last)
        if is_allocated:
            self.clear_cache()
        return is_allocated

    def clear_cache(self):
        self.get_live_ranges.clear_cache(self)
        self.get_size_hint.clear_cache(self)
        self.get_symbolic_size.clear_cache(self)


@dataclasses.dataclass
class TemporalSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    """
    Contains a list of allocations not overlapping in LiveRanges.

    Invariant: no pair (a,b) in self.allocations will have:
         a.get_live_ranges().overlaps(b.get_live_ranges())
    """

    allocations: list[AllocationTreeNode]

    def _allocate(self, block: Allocation, is_last: bool):
        slot_size = self.get_size_hint()
        block_size = block.get_size_hint()
        if not is_last and block_size > slot_size:
            return False  # doesn't fit

        block_live = block.get_live_ranges()
        overlapping = [
            s for s in self.allocations if s.get_live_ranges().overlaps(block_live)
        ]
        if len(overlapping) > 1:
            # TODO(jansel): we could try harder here by merging overlapping in space
            return False
        elif len(overlapping) == 1:
            return overlapping[0].allocate(block, is_last)
        else:
            block.mark_allocated()

            if len(self.allocations) == 1 and isinstance(self.allocations[-1], Empty):
                self.allocations.pop()

            if slot_size == block_size:
                # perfect fit
                self.allocations.append(block)
            elif slot_size > block_size:
                self.allocations.append(
                    SpatialSplit.create(block, slot_size - block_size)
                )
            else:  # grow this allocation
                if not is_last:
                    raise AssertionError("can only grow allocation when is_last")
                self.allocations = [
                    *(
                        SpatialSplit.create(a, block_size - slot_size)
                        for a in self.allocations
                    ),
                    block,
                ]
            return True

    @cache_on_self
    def get_live_ranges(self) -> LiveRanges:
        return LiveRanges(
            itertools.chain.from_iterable(
                x.get_live_ranges().ranges for x in self.allocations
            )
        )

    @cache_on_self
    def get_size_hint(self) -> int:
        if not self.allocations:
            return 0
        return max(x.get_size_hint() for x in self.allocations)

    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr:
        if not self.allocations:
            return 0  # type: ignore[return-value]
        return Max(*[x.get_symbolic_size() for x in self.allocations])

    def is_empty(self):
        return len(self.allocations) == 1 and self.allocations[0].is_empty()

    def finalize(self, pool, offset):
        self.allocations = [block.finalize(pool, offset) for block in self.allocations]
        self.clear_cache()
        if len(self.allocations) == 1:
            return self.allocations[0]
        return self


@dataclasses.dataclass
class SpatialSplit(ClearCacheOnAllocateMixin, AllocationTreeNode):
    """
    Contains two allocations, left and right, that do not overlap in space.
    Right will be allocated immediately after left in memory.
    """

    left: TemporalSplit
    right: TemporalSplit

    @staticmethod
    def create(left, extra_space):
        if not isinstance(left, AllocationTreeNode):
            raise AssertionError(f"expected AllocationTreeNode, got {type(left)}")
        if not (isinstance(extra_space, int) and extra_space >= 1):
            raise AssertionError(
                f"expected positive int extra_space, got {extra_space}"
            )
        return SpatialSplit(TemporalSplit([left]), TemporalSplit([Empty(extra_space)]))

    def _allocate(self, block: Allocation, is_last: bool):
        return self.left.allocate(block, False) or self.right.allocate(block, is_last)

    @cache_on_self
    def get_live_ranges(self):
        return LiveRanges(
            itertools.chain(
                self.left.get_live_ranges().ranges, self.right.get_live_ranges().ranges
            )
        )

    @cache_on_self
    def get_size_hint(self) -> int:
        return _align(self.left.get_size_hint()) + self.right.get_size_hint()

    @cache_on_self
    def get_symbolic_size(self) -> sympy.Expr:
        return align(self.left.get_symbolic_size()) + self.right.get_symbolic_size()

    def finalize(self, pool, offset):
        self.left = self.left.finalize(pool, offset)
        self.right = self.right.finalize(
            pool, offset + align(self.left.get_symbolic_size())
        )
        self.clear_cache()
        if self.right.is_empty():
            return self.left
        return self


@dataclasses.dataclass
class AllocationPool:
    """
    Represents a pool of allocations that will be generated by a single
    call to torch.empty.
    """

    device: torch.device
    root: TemporalSplit
    can_expand: bool = True
    restrict_live_range: LiveRange | None = None
    name: str | None = None
    names_to_del: list[str] = dataclasses.field(default_factory=list)
    creation_cache: dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        for block in self.root.allocations:
            if isinstance(block, Allocation):
                self.update_restrict_live_range(block)

    def allocate(self, block: Allocation, is_last: bool):
        if (
            self.restrict_live_range is not None
            and not self.restrict_live_range.contains(block.live_range)
        ):
            return False

        block_earliest_available = block.get_earliest_available()
        pool_begin = self.root.get_live_ranges().begin
        if block_earliest_available and block_earliest_available > pool_begin:
            return False

        is_last = self.can_expand and is_last
        if self.root.allocate(block, is_last):
            self.update_restrict_live_range(block)
            return True

        if is_last:
            return self.allocate_at_end(block)

        return False

    def update_restrict_live_range(self, block: Allocation):
        if block_earliest_available := block.get_earliest_available():
            if self.restrict_live_range is None:
                self.restrict_live_range = LiveRange(
                    block_earliest_available, float("inf")
                )
            else:
                self.restrict_live_range = LiveRange(
                    min(self.restrict_live_range.begin, block_earliest_available),
                    self.restrict_live_range.end,
                )

    def allocate_at_end(self, block):
        block.mark_allocated()
        self.root = TemporalSplit([SpatialSplit(self.root, TemporalSplit([block]))])
        self.update_restrict_live_range(block)
        return True

    def finalize(self, name):
        if self.name:
            raise AssertionError("pool already finalized")
        self.name = name
        self.names_to_del.append(name)
        self.root.finalize(self, 0)

    def codegen_create(self, wrapper, code: IndentedBuffer):
        if not self.name:
            raise AssertionError("pool must be finalized before codegen_create")
        nbytes = self.root.get_symbolic_size()
        for block in self.root.allocations:
            if isinstance(block, Allocation) and nbytes == block.get_symbolic_size():
                node = block.node
                code.writeline(
                    wrapper.make_allocation(
                        self.name,
                        device=self.device,
                        dtype=node.get_dtype(),
                        shape=tuple(node.get_size()),
                        stride=tuple(node.get_stride()),
                    )
                )
                return
        else:
            code.writeline(
                wrapper.make_allocation(
                    self.name,
                    device=self.device,
                    dtype=torch.uint8,
                    shape=(nbytes,),
                    stride=(1,),
                )
            )

    def codegen_destroy(self, wrapper, code: IndentedBuffer):
        code.writeline(wrapper.make_free_by_names(self.names_to_del))

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


@dataclasses.dataclass
class AllocationPools:
    """
    Collection of many AllocationPool objects grouped by device.
    """

    device_to_pools: dict[torch.device, list[AllocationPool]] = dataclasses.field(
        default_factory=dict
    )

    def get_pools(self, block):
        if block.device not in self.device_to_pools:
            self.device_to_pools[block.device] = []
        return self.device_to_pools[block.device]

    def allocate(self, block: Allocation):
        pools = self.get_pools(block)

        for pool in pools:
            if pool.allocate(block, is_last=pool is pools[-1]):
                return

        # everything is full, make a new pool
        pools.append(
            AllocationPool(
                block.device,
                TemporalSplit([block]),
                can_expand=config.memory_pool != "none",
            )
        )
        block.mark_allocated()

    def allocate_output(self, block: Allocation):
        """Outputs get different pools so memory gets freed properly"""
        pools = self.get_pools(block)
        if pools and config.memory_pool in ("outputs", "combined"):
            pools[-1].allocate_at_end(block)
        else:
            # create a new pool
            block.mark_allocated()
            pools.append(
                AllocationPool(
                    block.device,
                    TemporalSplit([block]),
                    can_expand=config.memory_pool == "combined",
                )
            )

    def finalize(self):
        """Called at the end of allocation process"""
        for i, pool in enumerate(
            itertools.chain.from_iterable(self.device_to_pools.values())
        ):
            pool.finalize(f"pool{i}")

    def pprint(self):
        for pool in itertools.chain.from_iterable(self.device_to_pools.values()):
            print()
            print(pool.name)
            print(pool.root.get_live_ranges())
            pprint.pprint(pool.root)


class BufferGroup:
    """
    Due to inplace reuse an allocated buffer can have many names.
    This tracks these collections of buffers sharing underlying memory.
    """

    def __init__(self, node: BufferLike):
        self.node = node
        self.names = [node.get_name()]
        self.is_output = False
        self.allocation: Allocation | None = None
        self.live_range = LiveRange(float("inf"), -float("inf"))

    def update_usage(self, timestep: int):
        """Expand self.live_range to include timestep"""
        self.live_range = LiveRange(
            min(timestep, self.live_range.begin),
            max(timestep, self.live_range.end),
        )

    def sym_nbytes(self):
        return self.node.get_layout().storage_size() * self.node.get_dtype().itemsize

    def make_allocation(self):
        if self.allocation:
            raise AssertionError("multiple allocations")
        if not isinstance(self.live_range.begin, int):
            raise AssertionError("live ranges not computed")
        nbytes = self.sym_nbytes()
        # For now, fallback value will be used if we encounter an unbacked SymInt. The longer-term plan is to have
        # size_hint() use better heuristics for unbackeds, at which point the fallback value will be ignored.
        size_hint = V.graph.sizevars.optimization_hint(nbytes, fallback=64)
        self.allocation = Allocation(
            self.node,
            self.live_range,
            size_hint=size_hint,
            symbolic_size=nbytes,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.names!r}, is_output={self.is_output}, "
            f"live_range={self.live_range}"
        )


@dataclasses.dataclass
class PoolMemoryPlanningLine(MemoryPlanningLine):
    """Abstract base class for {Alloc,Dealloc}FromPoolLine"""

    group: BufferGroup
    timestep: int | None = None

    @property
    def node(self):
        return self.group.node


@dataclasses.dataclass
class AllocFromPoolLine(PoolMemoryPlanningLine):
    """Similar to AllocationLine, but takes memory from a pool"""

    is_first_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        allocation = self.group.allocation
        if not (allocation and allocation.pool):
            raise AssertionError("group must have an allocation with a pool")
        pool = allocation.pool
        name = self.node.get_name()

        if self.is_first_pool_usage:
            pool.codegen_create(self.wrapper, code)

        pool.names_to_del.extend(self.group.names)
        alloc_from_pool, allocation_lines_to_write = allocation.codegen_alloc_from_pool(
            self.wrapper
        )
        code.writelines(allocation_lines_to_write)
        if alloc_from_pool in pool.creation_cache:
            code.writeline(
                self.wrapper.make_tensor_alias(
                    name, pool.creation_cache[alloc_from_pool], "alloc"
                )
            )
        else:
            pool.creation_cache[alloc_from_pool] = name
            code.writeline(
                f"{self.wrapper.declare}{name} = {alloc_from_pool}{self.wrapper.ending}"
            )


@dataclasses.dataclass
class DeallocFromPoolLine(PoolMemoryPlanningLine):
    """Similar to FreeIfNotReusedLine, but takes memory from a pool"""

    is_last_pool_usage: bool = False

    def codegen(self, code: IndentedBuffer):
        if self.is_last_pool_usage:
            if not (self.group.allocation and self.group.allocation.pool):
                raise AssertionError("group must have an allocation with a pool")
            self.group.allocation.pool.codegen_destroy(self.wrapper, code)


@dataclasses.dataclass
class MemoryPlanner:
    """
    Coordination object to run memory planning passes during wrapper
    codegen.
    """

    wrapper: Any
    pools: AllocationPools = dataclasses.field(default_factory=AllocationPools)
    buffer_groups: list[BufferGroup] | None = None

    def plan(self, lines: list[Any]) -> list[Any]:
        """Call all the memory planning passes in sequence"""
        lines = [*lines]
        self.drop_removed_buffers(lines)
        self.convert_to_pool_lines(lines)
        self.compute_live_ranges(lines)
        self.allocate_groups()
        self.mark_first_last_usage(lines)
        return lines

    def drop_removed_buffers(self, lines):
        """
        Replace any memory planning lines in V.graph.removed_buffers with NullLine
        """
        # drop any removed buffers
        for i, line in enumerate(lines):
            if isinstance(line, (AllocateLine, FreeIfNotReusedLine, ReuseLine)):
                if line.node.get_name() in V.graph.removed_buffers:
                    lines[i] = NullLine(self.wrapper)

    def compute_buffer_groups(self, lines):
        """
        Populates self.buffer_groups with BufferGroup objects that join
        allocations with common storage (due to inplace reuse) into a
        single object.
        """
        name_to_group = {}
        for line in lines:
            if isinstance(line, AllocateLine):
                name = line.node.get_name()
                if name in name_to_group:
                    raise AssertionError(f"duplicate allocation for {name}")
                name_to_group[name] = BufferGroup(line.node)
            elif isinstance(line, ReuseLine):
                old_name = line.node.get_name()
                new_name = line.reused_as.get_name()
                if new_name in name_to_group:
                    raise AssertionError(f"duplicate group for reused {new_name}")
                # TODO(jansel): we should support reusing buffers created via ExternKernelAlloc
                if old_name in name_to_group:
                    name_to_group[old_name].names.append(new_name)
                    name_to_group[new_name] = name_to_group[old_name]

        outputs = OrderedSet(V.graph.get_output_names())
        unique_groups = [*{id(g): g for g in name_to_group.values()}.values()]
        for group in unique_groups:
            group.is_output = any(x in outputs for x in group.names) or any(
                any(user.node.get_name() == "OUTPUT" for user in buf.users)
                for name in group.names
                if (buf := V.graph.scheduler.name_to_buf.get(name)) is not None
            )

        if self.buffer_groups is not None:
            raise AssertionError("buffer_groups already computed")
        self.buffer_groups = unique_groups
        return name_to_group

    def convert_to_pool_lines(self, lines):
        """
        Convert AllocateLine/FreeIfNotReusedLine/ReuseLine into their
        pool-based counterparts.
        """
        name_to_group = self.compute_buffer_groups(lines)
        for i, line in enumerate(lines):
            if isinstance(line, AllocateLine):
                if line.node.get_name() in name_to_group:
                    lines[i] = AllocFromPoolLine(
                        self.wrapper, name_to_group[line.node.get_name()]
                    )
            elif isinstance(line, FreeIfNotReusedLine):
                if line.is_reused:
                    raise AssertionError("expected line not to be reused")
                if line.node.get_name() in name_to_group:
                    lines[i] = DeallocFromPoolLine(
                        self.wrapper, name_to_group[line.node.get_name()]
                    )
            elif isinstance(line, ReuseLine):
                if line.node.get_name() in name_to_group:
                    line.delete_old = False

    def compute_live_ranges(self, lines):
        """Populate every BufferGroup.live_ranges field based on first/last usage"""
        timestep = 0
        worklist = collections.deque(lines)
        while worklist:
            if isinstance(worklist[0], MemoryPlanningLine):
                timestep += 1
                while worklist and isinstance(worklist[0], MemoryPlanningLine):
                    line = worklist.popleft()
                    if isinstance(line, PoolMemoryPlanningLine):
                        line.group.update_usage(timestep)
                        line.timestep = timestep
            else:
                worklist.popleft()

        timestep += 1
        if self.buffer_groups is None:
            raise AssertionError("buffer_groups not computed")
        for group in self.buffer_groups:
            if group.is_output:
                group.update_usage(timestep)

    def allocate_groups(self):
        """
        Assign every allocation to a specific location in a specific AllocationPool.
        """
        if config.memory_pool not in ("none", "intermediates", "outputs", "combined"):
            raise AssertionError(f"invalid memory_pool {config.memory_pool}")
        if self.buffer_groups is None:
            raise AssertionError("buffer_groups not computed")

        for group in self.buffer_groups:
            group.make_allocation()

        outputs: list[Allocation] = []
        intermediates: list[Allocation] = []
        for group in self.buffer_groups:
            if not group.allocation:
                raise AssertionError("group has no allocation")
            if group.is_output and config.memory_pool != "combined":
                outputs.append(group.allocation)
            else:
                intermediates.append(group.allocation)

        for block in sorted(
            outputs,
            key=lambda x: (
                x.size_hint,
                -len(x.live_range),
            ),
        ):
            self.pools.allocate_output(block)

        for block in sorted(
            intermediates,
            key=lambda x: (
                -x.size_hint,
                -len(x.live_range),
            ),
        ):
            self.pools.allocate(block)

        self.pools.finalize()

    def mark_first_last_usage(self, lines):
        """
        Populate the AllocFromPoolLine.is_first_pool_usage and
        DeallocFromPoolLine.is_last_pool_usage fields so that pools
        are created/destroyed.
        """
        seen = OrderedSet[AllocationPool]()
        for line in lines:
            if isinstance(line, AllocFromPoolLine):
                if not line.group.allocation:
                    raise AssertionError("group has no allocation")
                pool = line.group.allocation.pool
                if pool is None:
                    raise AssertionError("allocation has no pool")
                if pool not in seen:
                    line.is_first_pool_usage = True
                    seen.add(pool)

        seen = OrderedSet[AllocationPool]()
        for line in reversed(lines):
            if isinstance(line, DeallocFromPoolLine):
                if not line.group.allocation:
                    raise AssertionError("group has no allocation")
                pool = line.group.allocation.pool
                if pool is None:
                    raise AssertionError("allocation has no pool")
                if pool not in seen:
                    line.is_last_pool_usage = (
                        pool.root.get_live_ranges().end <= line.timestep
                    )
                    seen.add(pool)
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import contextlib
import ctypes
import dataclasses
import functools
import math
import os
import sys
import textwrap
from itertools import chain, count
from typing import Any, Literal, TYPE_CHECKING

import sympy

import torch
import torch._higher_order_ops.torchbind
import torch._inductor.async_compile
import torch._ops
from torch import dtype as torch_dtype
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch.fx.experimental.symbolic_shapes import ConvertIntKey, DivideByKey, SymTypes
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CleanDiv, FloorDiv, Mod, ModularIndexing
from torch.utils._sympy.symbol import symbol_is_type, SymT

from .. import config, ir
from ..ir import ExternKernel
from ..utils import (
    _align,
    AotOnlyBuffer,
    DualIndentedBuffer,
    make_codegen_buffer,
    normalize_name,
)
from ..virtualized import V
from .aoti_hipify_utils import maybe_hipify_code_wrapper
from .common import get_device_op_overrides, IndentedBuffer, Kernel
from .cpp_utils import (
    cexpr,
    DEVICE_TO_ATEN,
    DEVICE_TO_INT,
    DTYPE_TO_ATEN,
    DTYPE_TO_CPP,
    LAYOUT_TO_ATEN,
)
from .wrapper import (
    _rewrite_symbol_solution_for_int_codegen,
    codegen_reinterpret_view_helper,
    EnterSubgraphLine,
    ExitSubgraphLine,
    HasWriteLine,
    PythonWrapperCodegen,
    SymbolicCallArg,
    WrapperLine,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ..graph import GraphLowering
    from ..runtime.hints import TritonMeta

    # At most, the list nesting can go one layer deep.
    _OUTPUT_ARGS_TYPE = list[str | None | list[str | None]]

    from ..scheduler import BaseSchedulerNode


@dataclasses.dataclass
class DualWrapperCodeLine(WrapperLine):
    jit_code: IndentedBuffer
    aot_code: AotOnlyBuffer

    def codegen(self, code: DualIndentedBuffer) -> None:
        # In dual-wrapper replay, these route each captured snippet to the
        # matching side. If replayed on a plain IndentedBuffer, splice_jit writes
        # normally and splice_aot is a no-op, preserving the same selection.
        code.splice_jit(self.jit_code)
        code.splice_aot(self.aot_code)


@dataclasses.dataclass
class DeferredCpuTritonCallWrapper:
    """CPU counterpart of DeferredTritonCallWrapper in cpp_wrapper_gpu.py.

    Deferred so wrapper emission can read `CpuTritonKernelCache` entries
    after the autotune block has populated them.

    NOTE: CPU doesn't actively use autotuning, but we mirror
    the GPU flow for symmetry.
    """

    wrapper_name: str
    kernel_name: str
    arg_types: list[Any]
    triton_meta: TritonMeta | None = None
    inductor_meta: dict[str, Any] | None = None

    def _get_cpp_param_type(self, name: str, arg_type: Any) -> str:
        if isinstance(arg_type, torch_dtype):
            return f"const {name}_type_& {name}"
        elif issubclass(arg_type, (SymbolicCallArg, sympy.Expr, int)):
            return f"int64_t {name}"
        elif arg_type is float:
            return f"float {name}"
        elif arg_type is bool:
            return f"bool {name}"
        else:
            raise ValueError(
                f"Unexpected arg type {arg_type} for CPU triton wrapper arg '{name}'"
            )

    def _resolve_arg_names(
        self,
    ) -> tuple[list[str], list[str], list[str]]:
        """Return (wrapper_arg_names, kernel_arg_names, grid_arg_names).

        Wrapper args exclude *declared* constexpr (the call site doesn't pass
        them); kernel args additionally exclude *value-specialized* constexpr
        (baked into the binary). Grid args are appended for `run_from_nativert`.
        """
        from torch._inductor.codecache import CpuTritonKernelCache

        info = CpuTritonKernelCache.get(self.kernel_name)
        if info is None:
            raise AssertionError(
                f"CpuTritonKernelCache not populated for {self.kernel_name}"
            )
        signature = info["signature"]
        declared_constexpr_names = OrderedSet(
            (self.inductor_meta or {}).get("declared_constexpr_names", [])
        )
        wrapper_arg_names = [
            name for name in signature if name not in declared_constexpr_names
        ]
        kernel_arg_names = [name for name, ty in signature.items() if ty != "constexpr"]
        grid_arg_names = list((self.inductor_meta or {}).get("extra_launcher_args", []))
        return wrapper_arg_names + grid_arg_names, kernel_arg_names, grid_arg_names

    def _write_wrapper_signature(
        self,
        prefix: IndentedBuffer,
        wrapper_arg_names: list[str],
    ) -> None:
        # `cubin_dir_` name reused from GPU so the same AOTI runtime member feeds both.
        if len(wrapper_arg_names) != len(self.arg_types):
            raise AssertionError(
                (
                    wrapper_arg_names,
                    self.arg_types,
                )
            )
        template_types = [
            f"typename {name}_type_"
            for name, arg_type in zip(wrapper_arg_names, self.arg_types)
            if isinstance(arg_type, torch_dtype)
        ]
        if V.graph.aot_mode:
            template_types.append("typename kernels_type_")
        if template_types:
            prefix.writeline(f"template <{', '.join(template_types)}>")

        param_lines = [
            self._get_cpp_param_type(name, arg_type)
            for name, arg_type in zip(wrapper_arg_names, self.arg_types)
        ]
        if V.graph.aot_mode:
            param_lines.append("kernels_type_& kernels_")
        param_lines.append(
            "const std::optional<std::string>& cubin_dir_ = std::nullopt"
        )

        prefix.writeline(f"static __attribute__((noinline)) void {self.wrapper_name}(")
        with prefix.indent():
            for i, param in enumerate(param_lines):
                comma = "," if i < len(param_lines) - 1 else ""
                prefix.writeline(f"{param}{comma}")
        prefix.writeline("){")

    def _generate_load_kernel(
        self,
        prefix: IndentedBuffer,
        kernel_so: str,
        launcher_so: str,
        kernel_symbol: str,
    ) -> None:
        # `std::call_once` keeps init thread-safe under concurrent first calls.
        kernel_member = f"kernels_.{self.kernel_name}_kernel_fn"
        launcher_member = f"kernels_.{self.kernel_name}_launcher_fn"
        prefix.splice(f"""
            using namespace torch::aot_inductor;
            static std::once_flag {self.kernel_name}_init_flag;
            std::call_once({self.kernel_name}_init_flag, [&]() {{
                {kernel_member} = loadCpuTritonKernel(
                    "{kernel_so}", "{kernel_symbol}", cubin_dir_);
                {launcher_member} = loadCpuTritonLauncher(
                    "{launcher_so}", cubin_dir_);
            }});
            """)

    def _generate_launch(
        self,
        prefix: IndentedBuffer,
        kernel_arg_names: list[str],
        grid_arg_names: list[str],
        signature: dict[str, Any],
    ) -> None:
        # The launcher uses literal `args[i]` indices over the *original*
        # signature, so constexpr slots must be padded (nullptr) to keep
        # non-constexpr args at the expected positions.
        kernel_member = f"kernels_.{self.kernel_name}_kernel_fn"
        launcher_member = f"kernels_.{self.kernel_name}_launcher_fn"

        args_array_entries: list[str] = []
        for name, ty in signature.items():
            if ty == "constexpr":
                args_array_entries.append("nullptr")
            elif isinstance(ty, str) and ty.startswith("*"):
                args_array_entries.append(f"{name}.data_ptr()")
            else:
                args_array_entries.append(f"&{name}")
        args_arr_str = ", ".join(args_array_entries)

        # Grid args are int64_t in the wrapper but the launcher ABI is uint32_t.
        grid_x = grid_arg_names[0] if len(grid_arg_names) > 0 else "1"
        grid_y = grid_arg_names[1] if len(grid_arg_names) > 1 else "1"
        grid_z = grid_arg_names[2] if len(grid_arg_names) > 2 else "1"

        # TODO: thread `num_cpu_threads` from inductor_meta (autotune support).
        prefix.splice(f"""
            void* args_arr[] = {{ {args_arr_str} }};
            {launcher_member}(
                static_cast<uint32_t>({grid_x}),
                static_cast<uint32_t>({grid_y}),
                static_cast<uint32_t>({grid_z}),
                /*num_cpu_threads=*/1,
                args_arr,
                {kernel_member});
            """)

    def generate(self, wrapper: CppWrapperCpu) -> None:
        from torch._inductor.codecache import CpuTritonKernelCache

        info = CpuTritonKernelCache.get(self.kernel_name)
        if info is None:
            raise RuntimeError(
                f"CpuTritonKernelCache not populated for '{self.kernel_name}': "
                "the autotune block did not save this kernel. Check that "
                "config.triton.autotune_at_compile_time is True and the Triton "
                "CPU backend emits a launcher .so exporting `run_from_nativert`."
            )

        signature = info["signature"]
        wrapper_arg_names, kernel_arg_names, grid_arg_names = self._resolve_arg_names()

        prefix = wrapper.prefix
        self._write_wrapper_signature(prefix, wrapper_arg_names)
        with prefix.indent():
            self._generate_load_kernel(
                prefix,
                info["kernel_so_path"],
                info["launcher_so_path"],
                info["kernel_symbol"],
            )
            self._generate_launch(prefix, kernel_arg_names, grid_arg_names, signature)
        prefix.writeline("}")

        # Register .so artifacts so the AOTI packager picks them up.
        V.graph.wrapper_code.additional_files.append(info["kernel_so_path"])
        V.graph.wrapper_code.additional_files.append(info["launcher_so_path"])


class CppWrapperCpu(PythonWrapperCodegen):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels
    """

    def __init__(self):
        if not hasattr(self, "device"):
            self.device = "cpu"
        # must be initialized prior to calling super().__init__()
        self.included_devices: OrderedSet[str] = OrderedSet()
        self.model_class_name_suffix = (
            ""
            if config.aot_inductor.dynamic_linkage
            else config.aot_inductor.model_name_for_generated_files
        )
        self.aoti_model_class_name = f"AOTInductorModel{self.model_class_name_suffix}"

        super().__init__()

        self.declare = "auto "
        self.declare_maybe_reference = "decltype(auto) "
        self.ending = ";"
        self.comment = "//"
        self.none_str = "nullptr"
        self.supports_intermediate_hooks = False
        self.kernel_callsite_id = count()
        self.int_array_id = count()  # for int array local variable declarations
        self.declared_int_array_vars: OrderedSet[str] = OrderedSet()
        self.tmp_tensor_id = count()  # for tmp tensor local variable declarations
        self.arg_var_id = count()
        self._aot_output: str | None = None
        # Defined kernels drive member declarations; called kernels drive
        # `call_<k>` emission. A kernel may be defined without being called.
        self._cpu_triton_kernel_names: OrderedSet[str] = OrderedSet()
        self._cpu_triton_call_wrappers: dict[str, DeferredCpuTritonCallWrapper] = {}
        self.used_cached_devices: OrderedSet[str] = OrderedSet()
        self.used_cached_dtypes: OrderedSet[str] = OrderedSet()
        self.used_cached_layouts: OrderedSet[str] = OrderedSet()
        self.used_cached_memory_formats: OrderedSet[str] = OrderedSet()
        self.used_switch_selector: OrderedSet[str] = OrderedSet()
        self.cached_output_id = count()
        self.scalar_to_tensor_id = count()
        self.custom_op_wrapper_loaded = False
        # For GEMM kernels that must be initialized and are resolved at linking.
        self.initialized_kernels: dict[str, Kernel] = {}
        self.device_codegen = get_device_op_overrides(self.device)
        self._included_extra_headers: OrderedSet[str] = OrderedSet()
        self.codegen_int_array_var_cache = {}
        self.needs_vec_isa = self.device == "cpu"

    @contextlib.contextmanager
    def _preserve_device_guard_state(self):
        last_seen_device_guard_index = self.last_seen_device_guard_index
        try:
            yield
        finally:
            self.last_seen_device_guard_index = last_seen_device_guard_index

    @contextlib.contextmanager
    def _preserve_codegen_state(self, graphs):
        wrapper_state = (
            OrderedSet(self.allocated),
            OrderedSet(self.freed),
            dict(self.reuses),
            dict(self.allocated_workspaces),
            OrderedSet(self.declared_int_array_vars),
            dict(self.codegen_int_array_var_cache),
            OrderedSet(self.kernel_numel_expr),
            OrderedSet(self.used_switch_selector),
        )
        subgraph_state = [
            (
                graph,
                OrderedSet(graph.removed_buffers),
                OrderedSet(graph.inplaced_to_remove),
            )
            for graph in graphs
        ]
        try:
            yield
        finally:
            (
                self.allocated,
                self.freed,
                self.reuses,
                self.allocated_workspaces,
                self.declared_int_array_vars,
                self.codegen_int_array_var_cache,
                self.kernel_numel_expr,
                self.used_switch_selector,
            ) = wrapper_state
            for graph, removed_buffers, inplaced_to_remove in subgraph_state:
                graph.removed_buffers = removed_buffers
                graph.inplaced_to_remove = inplaced_to_remove

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: str | None,
        parent_wrapper: PythonWrapperCodegen | None,
        partition_signatures: ir.GraphPartitionSignature | None = None,
    ):
        # TODO - support subgraph codegen by lifting functions. Check the
        # comment at CppWrapperCpu `codegen_subgraph` function.
        return CppWrapperCpu()

    @contextlib.contextmanager
    def _target_buf(self, attr: str, buf: IndentedBuffer):
        """Temporarily redirect self.<attr> to a specific buffer."""
        saved = getattr(self, attr)
        setattr(self, attr, buf)
        try:
            yield
        finally:
            setattr(self, attr, saved)

    @staticmethod
    def _generate_temporary_array_pointer(
        c_type: str, elements: Sequence[str], *, force_mutable: bool = False
    ) -> str:
        """Get a pointer to an array that only exists for the duration of the C++
        statement it's used in."""
        # If the c_type is already a pointer, return a mutable pointer to the array.
        # Otherwise, return a const pointer.  In the C-shim API, pointer types are only
        # const-qualified with respect to the underlying value, not any nested pointers.
        # e.g. const double** is possible, but not const double* const*.  This means
        # that an array containing pointers must _already_ be properly const-qualified
        # by the c_type, and not add additional const-ness.
        array_expr = f"std::array<{c_type}, {len(elements)}>{{{', '.join(elements)}}}"
        if force_mutable or c_type.endswith("*"):
            return f"{array_expr}.data()"

        return f"static_cast<const {c_type}*>({array_expr}.data())"

    def _generate_kernel_call_helper(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton=True,
        arg_types=None,
        raw_keys=None,
        raw_args=None,
        triton_meta: TritonMeta | None = None,
        inductor_meta=None,
        graph_name="",
        original_fxnode_name=None,
        current_stream_idx=None,
    ):
        """
        Generates kernel call code.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the CUDA language for codegen.
                Only valid when cuda == True.
        """
        if not (arg_types is not None and len(call_args) == len(arg_types)):
            raise AssertionError(
                "Mismatch call_args and arg_types in generate_kernel_call:\n"
                f"call_args: {call_args}\n"
                f"arg_types: {arg_types}"
            )

        # CPU Triton kernel: run the autotune block (populates
        # `CpuTritonKernelCache`) and emit `call_<k>(...)`; the body is
        # defined later by `DeferredCpuTritonCallWrapper` in `finalize_prefix`.
        if triton and kernel_name in self._cpu_triton_kernel_names:
            if (
                config.triton.autotune_at_compile_time
                and kernel_name not in self.kernel_autotune_names
            ):
                PythonWrapperCodegen._generate_kernel_call_helper(
                    self,
                    kernel_name,
                    call_args,
                    device=device,
                    triton=triton,
                    arg_types=arg_types,
                    raw_keys=raw_keys,
                    raw_args=raw_args,
                    triton_meta=triton_meta,
                    inductor_meta=inductor_meta,
                    original_fxnode_name=original_fxnode_name,
                )

            wrapper_name = f"call_{kernel_name}"
            if wrapper_name not in self._cpu_triton_call_wrappers:
                self._cpu_triton_call_wrappers[wrapper_name] = (
                    DeferredCpuTritonCallWrapper(
                        wrapper_name=wrapper_name,
                        kernel_name=kernel_name,
                        arg_types=list(arg_types),
                        triton_meta=triton_meta,
                        inductor_meta=inductor_meta,
                    )
                )

            stringified_args = [
                self._stringify_cpu_triton_call_arg(a) for a in call_args
            ]
            if V.graph.aot_mode:
                stringified_args.append("kernels")
                stringified_args.append("this->cubin_dir_")
            self.writeline(self.wrap_kernel_call(wrapper_name, stringified_args))
            return

        new_args = []
        for idx, arg in enumerate(call_args):
            if isinstance(arg_types[idx], str) and "*" in arg_types[idx]:
                new_args.append(f"({arg_types[idx]})({arg}.data_ptr())")
            else:
                # arg is a scalar - ensure it's a string for C++ codegen
                # With Triton support, arg might be a SymPy expression or other type
                new_args.append(str(arg) if not isinstance(arg, str) else arg)
        # debug printer related logic for cpp kernel type.
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            call_args,
            kernel_name,
            None,
            None,
            "cpp",
        )
        with debug_printer_manager:
            self.writeline(self.wrap_kernel_call(kernel_name, new_args))

    def write_constant(self, name, hashed):
        # include a hash so our code cache gives different constants different files
        self.header.writeline(f"// {name} {hashed}")

    @staticmethod
    def get_device_include_path_jit(device: str) -> str:
        return f"#include <torch/csrc/inductor/cpp_wrapper/{device}.h>"

    @staticmethod
    def get_device_include_path_aot(device: str) -> str:
        return f"#include <torch/csrc/inductor/aoti_include/{device}.h>"

    def add_device_include(self, device: str) -> None:
        if device in self.included_devices:
            return

        self.included_devices.add(device)

        # Add the default header for this device, plus any C-shim extensions that are
        # present.
        self.header.splice_jit(self.get_device_include_path_jit(device))
        self.header.splice_aot(self.get_device_include_path_aot(device))
        extend_aoti_c_shim_include = (
            f"torch/csrc/inductor/aoti_torch/generated/extend/c_shim_{self.device}.h"
        )
        extend_aoti_c_shim_path = os.path.join(
            os.path.dirname(torch.__file__),
            "include",
            extend_aoti_c_shim_include,
        )
        if os.path.exists(extend_aoti_c_shim_path):
            self.header.splice(f"#include <{extend_aoti_c_shim_include}>")

    def _write_aoti_interface_header(self):
        """Emit the AOTI runtime interface include or inline."""
        if config.aot_inductor.dynamic_linkage:
            with open(
                os.path.join(os.path.dirname(__file__), "aoti_runtime", "interface.cpp")
            ) as f:
                self.header.splice_aot(f.read())
        else:
            # we produce a separate model header for each model in static linkage
            self.header.splice_aot(f"""#include \"{self.model_class_name_suffix}.h\"""")
        self.header.splice_aot("\n")

    def write_header(self):
        if V.graph.is_const_graph:
            # We do not write header for constant graph, it will be written by main module.
            return

        if not V.graph.aot_mode:
            self.header.splice('''
                import torch
                from torch._inductor.codecache import CppWrapperCodeCache

                cpp_wrapper_src = (
                r"""
                ''')

        for device in V.graph.device_types:
            if device != "meta":
                self.add_device_include(device)
        if not self.included_devices:
            # When only meta device is present, add_device_include is never
            # called so we need to pull in the common header explicitly.
            if V.graph.aot_mode:
                self.header.writeline(
                    "#include <torch/csrc/inductor/aoti_include/common.h>"
                )
            else:
                self.header.writeline(
                    "#include <torch/csrc/inductor/cpp_wrapper/common.h>"
                )

        if V.graph.aot_mode:
            self._write_aoti_interface_header()

        if config.cpp.enable_kernel_profile and config.cpp.enable_kernel_context_guard:
            self.header.splice(
                "#include <torch/csrc/inductor/aoti_runtime/kernel_context_tls.h>"
            )
            self.header.splice("""
                namespace torch::aot_inductor {
                thread_local KernelContext* tls_kernel_context = nullptr;
                }
                """)

    def include_extra_header(self, header: str):
        # This is needed for cpp to python dtype conversion
        if header in self._included_extra_headers:
            return
        self._included_extra_headers.add(header)
        self.header.splice(f"#include <{header}>")

    def mark_output_type(self):
        # mark output type to unwrap tensor back to python scalar
        from ..ir import ShapeAsConstantBuffer

        output_is_tensor = {}
        for idx, x in enumerate(V.graph.graph_outputs):
            if isinstance(x, ShapeAsConstantBuffer):
                output_is_tensor[idx] = False
            else:
                output_is_tensor[idx] = True

        self.output_is_tensor = output_is_tensor

    def write_prefix(self):
        if V.graph.is_const_graph:
            # We do not write prefix for constant graph, it will be written by main module.
            return
        if config.aot_inductor.custom_ops_to_c_shims:
            # custom_ops_to_c_shims contains declaration of custom ops with C shim.
            # TODO: this could be auto-generated from a passed-in custom op schema
            custom_c_shims = list(
                chain(*config.aot_inductor.custom_ops_to_c_shims.values())
            )
            declarations = "\n".join(
                [f"extern {textwrap.dedent(shim)};" for shim in custom_c_shims]
            )
            self.prefix.splice(f"""
                extern "C" {{
                    {declarations}
                }}
                """)
        self.prefix.writeline_aot("namespace torch::aot_inductor {")
        if not V.graph.is_const_graph:
            self.codegen_input_size_and_nan_asserts()

    def write_input_output_info(
        self,
        info_kind: str,
        idx: int,
        name: str,
    ):
        self.prefix.writeline(f"""{info_kind}[{idx}].name = "{name}";""")

    def codegen_input_symbol_assignment(  # type: ignore[override]
        self,
        name: str,
        value: ir.TensorBox,
        bound_vars: OrderedSet[sympy.Symbol],
        deferred_symbol_assignments=None,
    ):
        """Assign symbolic shapes to C++ locals, with replacement aliases."""
        code = self.prefix

        @functools.cache
        def sizeof(name):
            self.codegen_input_size_var_decl(code, name)
            return f"{name}_size"

        @functools.cache
        def strideof(name):
            self.codegen_input_stride_var_decl(code, name)
            return f"{name}_stride"

        def codegen_symbol(
            sym_or_exp: sympy.Symbol | sympy.Expr,
            base_name: str,
            name_fn: Callable[[str], str],
            dim: int,
            deferred_symbol_assignments=None,
        ) -> bool:
            if isinstance(sym_or_exp, sympy.Symbol):
                if sym_or_exp in bound_vars:
                    return False
                code.writeline(f"int64_t {sym_or_exp} = {name_fn(base_name)}[{dim}];")
                bound_vars.add(sym_or_exp)
                if symbol_is_type(sym_or_exp, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)):
                    self.unbacked_symbol_decls.add(str(sym_or_exp))
                self.maybe_emit_replacement_aliases(sym_or_exp, bound_vars)
                return True
            elif isinstance(sym_or_exp, sympy.Expr):
                undefined_symbols = [
                    sym for sym in sym_or_exp.free_symbols if sym not in bound_vars
                ]
                if len(undefined_symbols) != 1:
                    # Skip if expression contains no symbols or if multiple
                    # symbols exists since we assume each base symbol is defined
                    # by other codegen_symbol calls.
                    if (
                        len(undefined_symbols) > 1
                        and deferred_symbol_assignments is not None
                    ):

                        def retry(deferred_symbol_assignments):
                            return codegen_symbol(
                                sym_or_exp,
                                base_name,
                                name_fn,
                                dim,
                                deferred_symbol_assignments,
                            )

                        deferred_symbol_assignments.append(retry)
                    return False

                from torch.utils._sympy.solve import try_solve

                free_symbol = undefined_symbols.pop()
                base_name = name_fn(base_name)
                # Use a size symbol to solve the free symbol
                size_symbol = sympy.Symbol(f"{base_name}_{dim}", integer=True)
                solution = try_solve(sympy.Eq(sym_or_exp, size_symbol), free_symbol)
                if solution is not None:
                    code.writeline(f"int64_t {size_symbol} = {base_name}[{dim}];")
                    expr = _rewrite_symbol_solution_for_int_codegen(solution[1])
                    code.writeline(f"int64_t {free_symbol} = {cexpr(expr)};")
                    bound_vars.add(free_symbol)
                    if symbol_is_type(
                        free_symbol, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)
                    ):
                        self.unbacked_symbol_decls.add(str(free_symbol))
                    self.maybe_emit_replacement_aliases(free_symbol, bound_vars)
                    return True
                return False
            return False

        if isinstance(value, sympy.Expr):
            if not isinstance(value, sympy.Symbol) or value in bound_vars:
                return
            if symbol_is_type(value, (SymT.FLOAT, SymT.UNBACKED_FLOAT)):
                decl = "double"
            else:
                decl = "int64_t"
            code.writeline(f"{decl} {value} = {name};")
            bound_vars.add(value)
            if symbol_is_type(value, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)):
                self.unbacked_symbol_decls.add(str(value))
            self.maybe_emit_replacement_aliases(value, bound_vars)
        elif isinstance(value, ir.TensorBox):
            for dim, size in enumerate(value.get_size()):
                codegen_symbol(size, name, sizeof, dim, deferred_symbol_assignments)
            for dim, stride in enumerate(value.get_stride()):
                codegen_symbol(stride, name, strideof, dim, deferred_symbol_assignments)
        elif isinstance(
            value, (ir.TorchBindObject, ir.GeneratorState, ir.OpaqueObjectState)
        ):
            # Python-only inputs do not define C++ shape symbols.
            pass
        else:
            raise AssertionError(f"Unknown value type: {type(value)}")

    def maybe_emit_replacement_aliases(  # type: ignore[override]
        self,
        sym: sympy.Symbol,
        bound_vars: OrderedSet[sympy.Symbol],
    ) -> None:
        sizevars = getattr(V.graph, "sizevars", None)
        shape_env = getattr(sizevars, "shape_env", None)
        if shape_env is None:
            return

        def is_backed_symbol(s: sympy.Symbol) -> bool:
            return not symbol_is_type(s, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT))

        def decl_type(s: sympy.Symbol) -> str:
            if symbol_is_type(s, (SymT.FLOAT, SymT.UNBACKED_FLOAT)):
                return "double"
            return "int64_t"

        for src, tgt in shape_env.replacements.items():
            if (
                tgt == sym
                and isinstance(src, sympy.Symbol)
                and src not in bound_vars
                and is_backed_symbol(src)
                and is_backed_symbol(sym)
            ):
                self.prefix.writeline(f"{decl_type(src)} {src} = {sym};")
                bound_vars.add(src)
            elif (
                src == sym
                and isinstance(tgt, sympy.Symbol)
                and tgt not in bound_vars
                and is_backed_symbol(sym)
                and is_backed_symbol(tgt)
            ):
                self.prefix.writeline(f"{decl_type(tgt)} {tgt} = {sym};")
                bound_vars.add(tgt)

    def bind_input_symbol(
        self,
        sym: sympy.Symbol,
        input_name: str,
        kind: Literal["size", "stride"],
        dim: int,
        bound_vars: OrderedSet[sympy.Symbol],
    ) -> None:
        if sym in bound_vars:
            return
        accessor = "sizes" if kind == "size" else "strides"
        self.prefix.writeline(f"int64_t {sym} = {input_name}.{accessor}()[{dim}];")
        bound_vars.add(sym)
        self.maybe_emit_replacement_aliases(sym, bound_vars)

    def generate_input_output_runtime_checks(self):
        """
        In debug_compile mode, we generate checks to ensure the dtype/shape/stride/device of each
        real input/output tensor match ones provided at compile time via sample
        input/output.
        """

        def gen_check(handle_kind, idx, name, tensor):
            # Wrap AtenTensorHandle with ConstantHandle for cleaner utility function access
            self.prefix.writeline_aot(
                f"ConstantHandle {name} = ConstantHandle({handle_kind}[{idx}]);"
            )
            self.prefix.writeline_aot(f"int32_t {name}_dtype;")
            self.prefix.writeline_aot(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype({name}, &{name}_dtype));"
            )
            expected_dtype_name = DTYPE_TO_ATEN[tensor.dtype]
            dtype_str = str(tensor.dtype).split(".")[-1]
            self.prefix.splice_aot(f"""
                    int32_t {name}_expected_dtype = aoti_torch_dtype_{dtype_str}();
                    if ({name}_expected_dtype != {name}_dtype) {{
                        std::stringstream ss;
                        ss << "{handle_kind}[{idx}]: unmatched dtype, "
                           << "expected: " << {name}_expected_dtype << "({expected_dtype_name}), "
                           << "but got: " << {name}_dtype << "\\n";
                        throw std::runtime_error(std::move(ss).str());
                    }}
                """)
            self.prefix.writeline_aot(f"auto {name}_size = {name}.sizes();")
            for dim_idx, d in enumerate(tensor.get_size()):
                if isinstance(d, (int, sympy.Integer)):
                    self.prefix.splice_aot(f"""
                            if ({d} != {name}_size[{dim_idx}]) {{
                                std::stringstream ss;
                                ss << "{handle_kind}[{idx}]: unmatched dim value at {dim_idx}, "
                                   << "expected: {d}, " << "but got: " << {name}_size[{dim_idx}]
                                   << "\\n";
                                throw std::runtime_error(std::move(ss).str());
                            }}
                        """)
                else:
                    from torch.utils._sympy.value_ranges import bound_sympy

                    sym_range = bound_sympy(d, V.graph.sizevars.shape_env.var_to_range)
                    if config.aot_inductor.check_lowerbound and not math.isinf(
                        sym_range.lower
                    ):
                        self.prefix.splice_aot(f"""
                                if ({name}_size[{dim_idx}] < {sym_range.lower}) {{
                                    std::stringstream ss;
                                    ss << "{handle_kind}[{idx}]: dim value is too small at {dim_idx}, "
                                       << "expected it to be >= {sym_range.lower}, " << "but got: "
                                       << {name}_size[{dim_idx}] << "\\n";
                                    throw std::runtime_error(std::move(ss).str());
                                }}
                            """)
                    if not math.isinf(sym_range.upper):
                        # Limit upper bound to max C long long value (2^63 - 1)
                        max_long_long = ctypes.c_longlong(2**63 - 1).value
                        upper_bound = min(sym_range.upper, max_long_long)
                        self.prefix.splice_aot(f"""
                                if ({name}_size[{dim_idx}] > {upper_bound}) {{
                                    std::stringstream ss;
                                    ss << "{handle_kind}[{idx}]: dim value is too large at {dim_idx}, "
                                       << "expected to be <= {upper_bound}, " << "but got: "
                                       << {name}_size[{dim_idx}] << "\\n";
                                    throw std::runtime_error(std::move(ss).str());
                                }}
                            """)

            self.prefix.writeline_aot(f"auto {name}_stride = {name}.strides();")
            for stride_idx, s in enumerate(tensor.get_stride()):
                if not isinstance(s, (int, sympy.Integer)):
                    continue
                self.prefix.splice_aot(f"""
                        if ({s} != {name}_stride[{stride_idx}]) {{
                            std::stringstream ss;
                            ss << "{handle_kind}[{idx}]: unmatched stride value at {stride_idx}, "
                               << "expected: {s}, " << "but got: " << {name}_stride[{stride_idx}]
                               << "\\n";
                            throw std::runtime_error(std::move(ss).str());
                        }}
                    """)

            # check input device type
            if isinstance(tensor, ir.TensorBox):
                tensor_device = tensor.get_device()
                if tensor_device is not None:
                    expected_device_type = DEVICE_TO_INT.get(tensor_device.type)
                    if expected_device_type is not None:
                        self.prefix.writeline_aot(f"int32_t {name}_device_type;")
                        self.prefix.writeline_aot(
                            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type({name}, &{name}_device_type));"
                        )
                        device_type_str = tensor_device.type
                        self.prefix.splice_aot(f"""
                                int32_t {name}_expected_device_type = {expected_device_type};
                                if ({name}_expected_device_type != {name}_device_type) {{
                                    std::stringstream ss;
                                    ss << "{handle_kind}[{idx}]: unmatched device type, "
                                    << "expected: " << {name}_expected_device_type << "{expected_device_type}({device_type_str}), "
                                    << "but got: " << {name}_device_type << "\\n";
                                    throw std::runtime_error(std::move(ss).str());
                                }}
                            """)

        # Create a separate function for each input check to avoid "too big to optimize" error
        for idx, (name, tensor) in enumerate(V.graph.graph_inputs.items()):
            self.prefix.splice_aot(f"""
                AOTI_NOINLINE static void check_input_{idx}(
                    AtenTensorHandle* input_handles
                ) {{
                """)
            with self.prefix.indent():
                gen_check("input_handles", idx, name, tensor)
            self.prefix.writeline_aot("}")

        # force noinline to avoid any potential compilation slowdown due to aggressive
        # inline done by the host compiler
        self.prefix.splice_aot("""
            static bool _check_aoti_runtime_check_inputs_env() {
                const static char* env_var_value = getenv("AOTI_RUNTIME_CHECK_INPUTS");
                const static bool result = env_var_value != nullptr && env_var_value[0] != '0';
                return result;
            }

            AOTI_NOINLINE static void __check_inputs_outputs(
                AtenTensorHandle* input_handles,
                AtenTensorHandle* output_handles) {
                if (!_check_aoti_runtime_check_inputs_env()){
                    return;
                }
            """)
        with self.prefix.indent():
            for idx in range(len(V.graph.graph_inputs)):
                self.prefix.writeline_aot(f"check_input_{idx}(input_handles);")
        self.prefix.writeline_aot("}")

    def _write_aoti_entry_point_signature(self):
        """Emit the AOTI entry point: run_impl method and _const_run_impl."""
        self.codegen_additional_funcs()

        if V.graph.const_module:
            # In dual-wrapper-mode, the const graph's JIT output is emitted separately,
            # while its AOTI body is spliced below into the main AOTI source.
            # Keep the const graph header out of the main output to avoid
            # duplicate standalone header content.
            if not V.graph.is_dual_wrapper_mode:
                self.header.splice(V.graph.const_module.wrapper_code.header)
            if V.graph.const_wrapper_code is None:
                raise AssertionError("expected const_wrapper_code to be set")
            self.prefix.splice_aot(V.graph.const_wrapper_code)
            if V.graph.const_kernel_code is None:
                raise AssertionError("expected const_kernel_code to be set")
            self.kernel_declarations.splice(V.graph.const_kernel_code)

        if V.graph.is_const_graph:
            self.prefix.splice_aot(f"""
                void {self.aoti_model_class_name}::_const_run_impl(
                    std::vector<AtenTensorHandle>& output_handles,
                    DeviceStreamType stream,
                    AOTIProxyExecutorHandle proxy_executor
                ) {{
                """)
        else:
            if not config.aot_inductor.use_runtime_constant_folding:
                # If we do not split the constant graph, we'll just create
                # an empty implementation when wrapping the main module.
                self.prefix.splice_aot(f"""
                    void {self.aoti_model_class_name}::_const_run_impl(
                        std::vector<AtenTensorHandle>& output_handles,
                        DeviceStreamType stream,
                        AOTIProxyExecutorHandle proxy_executor
                    ) {{}}

                    """)

            run_impl_proto = f"""
                void {self.aoti_model_class_name}::run_impl(
                    AtenTensorHandle*
                        input_handles, // array of input AtenTensorHandle; handles
                                        // are stolen; the array itself is borrowed
                    AtenTensorHandle*
                        output_handles, // array for writing output AtenTensorHandle; handles
                                        // will be stolen by the caller; the array itself is
                                        // borrowed
                    DeviceStreamType stream,
                    AOTIProxyExecutorHandle proxy_executor
                ) {{
                    __check_inputs_outputs(input_handles, output_handles);
                """

            self.generate_input_output_runtime_checks()
            self.prefix.splice_aot(run_impl_proto)

    def _write_jit_entry_point_signature(self):
        """Emit the JIT entry point: inductor_entry_impl free function."""
        self.prefix.splice_jit("""
            void inductor_entry_impl(
                AtenTensorHandle*
                    input_handles, // array of input AtenTensorHandle; handles
                                    // are stolen; the array itself is borrowed
                AtenTensorHandle*
                    output_handles  // array for writing output AtenTensorHandle; handles
                                    // will be stolen by the caller; the array itself is
                                    // borrowed)
            ) {
            """)

    def _write_entry_point_signature(self):
        """Emit the entry point function/method signature.

        AOTI: AOTInductorModel::run_impl with stream and proxy_executor params.
        JIT: inductor_entry_impl free function.
        Dual-wrapper-mode: both signatures, to their respective buffers.
        """
        if V.graph.aot_mode:
            self._write_aoti_entry_point_signature()
        self._write_jit_entry_point_signature()

    def _write_input_preamble(self):
        """Emit input stealing and (in JIT mode) GIL release."""
        # Weights are promoted in the JIT mode
        jit_num_args = len(V.graph.graph_inputs) + len(V.graph.constants)
        # AOTI const graphs enter through _const_run_impl and have no
        # input_handles. Their JIT entry point still receives constants as
        # appended input handles.
        if not V.graph.is_const_graph:
            aoti_num_args = len(V.graph.graph_inputs)
            self.prefix.splice_aot(
                f"auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, {aoti_num_args});"
            )
        # release GIL to support multiple instances inference (in different threads of the same process)
        self.prefix.splice_jit("py::gil_scoped_release_simple release;")
        self.prefix.splice_jit(
            f"auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, {jit_num_args});"
        )

    def _write_input_unpacking(self):
        """Emit code to unpack graph inputs from the handles array."""
        inputs_len = len(V.graph.graph_inputs.keys())
        if inputs_len == 0:
            return
        for idx, input_key in enumerate(V.graph.graph_inputs.keys()):
            # unwrap input tensor back to scalar
            if isinstance(V.graph.graph_inputs[input_key], sympy.Expr):
                from ..graph import may_get_constant_buffer_dtype

                dtype = may_get_constant_buffer_dtype(
                    V.graph.graph_inputs[input_key]  # type: ignore[arg-type]
                )
                if dtype is None:
                    raise AssertionError("Fails to get the dtype of the sympy.Expr")
                self.codegen_tensor_item(
                    dtype, f"inputs[{idx}]", input_key, self.prefix
                )
            else:
                self.prefix.writeline(f"auto {input_key} = std::move(inputs[{idx}]);")
        # debug printing for all input args to AOTI model
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.codegen_model_inputs_value_print(
            input_args_to_print=[
                input_key
                for input_key in V.graph.graph_inputs
                if input_key.startswith("arg")
            ]
        )

    def _write_constants_unpacking(self):
        """Emit code to unpack constants — from constants_ (AOTI) or inputs (JIT)."""
        inputs_len = len(V.graph.graph_inputs.keys())
        if not all(
            isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
        ):
            raise AssertionError("Expect all constants to be Tensor")
        for idx, constants_key in enumerate(V.graph.constants.keys()):
            # AOTI: weights live in constants_ owned by ConstantHandle; never std::move.
            # JIT: weights are appended as inputs to the graph.
            constants_idx = inputs_len + idx
            self.prefix.writeline_aot(
                f"[[maybe_unused]] auto& {constants_key} = constants_->at({idx});"
            )
            self.prefix.writeline_jit(
                f"[[maybe_unused]] auto {constants_key} = std::move(inputs[{constants_idx}]);"
            )

    def _write_entry_point_postamble(self):
        """Emit AOTI-specific post-input setup (inputs.clear, kernels ref)."""
        if not V.graph.is_const_graph:
            self.prefix.writeline_aot("inputs.clear();")
        self.prefix.writeline_aot(
            "[[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());"
        )

    def write_wrapper_decl(self):
        self._write_entry_point_signature()
        with self.prefix.indent():
            self._codegen_entry_impl_prologue()
            self._write_input_preamble()
            self._write_input_unpacking()
            self._write_constants_unpacking()
            self.codegen_inputs()
            self._write_entry_point_postamble()

    def codegen_tensor_dtype_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"int32_t {name}_dtype;")
        code.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype({name}, &{name}_dtype));"
        )

    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"auto {name}_size = {name}.sizes();")

    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"auto {name}_stride = {name}.strides();")

    def codegen_input_device_type_var_decl(self, code: IndentedBuffer, name):
        code.writeline(f"int32_t {name}_device_type;")
        code.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type({name}, &{name}_device_type));"
        )

    def codegen_additional_funcs(self):
        pass

    def codegen_initialized_kernel_decls(self):
        # Tell compiler we need to link with the non-mangled symbols
        for kernel in self.initialized_kernels.values():
            if not hasattr(kernel, "get_signature"):
                raise AssertionError(f"{kernel} must have get_signature implemented")
            signature = kernel.get_signature()
            self.prefix.writeline(f'extern "C" {signature};')

    def codegen_model_kernels(self):
        self.prefix.writeline("namespace {")
        self.codegen_initialized_kernel_decls()
        self.prefix.writeline(
            "class AOTInductorModelKernels : public AOTInductorModelKernelsBase {"
        )
        self.prefix.writeline("  public:")
        declare_kernel = OrderedSet(self.src_to_kernel.values()) - OrderedSet(
            self.initialized_kernels.keys()
        )
        declare_kernel.update(
            entry[0] for entry in self.user_defined_kernel_cache.values()
        )
        if V.graph.const_module:
            declare_kernel.update(
                V.graph.const_module.wrapper_code.src_to_kernel.values()
            )
        # CPU Triton kernels need a (kernel_fn, launcher_fn) pair, declared
        # separately below; skip them in the default single-`void*` loop.
        cpu_triton_names = self._cpu_triton_kernel_names
        for kernel in sorted(declare_kernel):
            if kernel in cpu_triton_names:
                continue
            self.prefix.writeline(
                maybe_hipify_code_wrapper(
                    f"    {self.device_codegen.cpp_kernel_type()} {kernel}{{nullptr}};"
                )
            )
        for kernel in sorted(cpu_triton_names):
            self.prefix.writeline(f"    void* {kernel}_kernel_fn{{nullptr}};")
            self.prefix.writeline(
                f"    torch::aot_inductor::CpuTritonLauncherFn "
                f"{kernel}_launcher_fn{{nullptr}};"
            )
        for name, kernel in self.initialized_kernels.items():
            if not hasattr(kernel, "get_signature"):
                raise AssertionError(f"{kernel} must have get_signature implemented")
            kernel_ptr = f"(*{name})"
            signature = kernel.get_signature().replace(name, kernel_ptr)
            self.prefix.writeline(f"    {signature} = torch::aot_inductor::{name};")
        self.prefix.writeline("};")
        self.prefix.writeline("}  // namespace\n\n")

        if config.aot_inductor.embed_kernel_binary:
            self.prefix.writeline('extern "C" {')
            for name in sorted(declare_kernel):
                self.prefix.writeline(
                    f"    extern const unsigned char __{name}_start[];"
                )
                if torch.xpu.is_available():
                    self.prefix.writeline(
                        f"    extern const unsigned char __{name}_end[];"
                    )
            self.prefix.writeline("}")

    # MSVC string was longer than the limit of 16380 single-byte characters.
    # https://learn.microsoft.com/en-us/cpp/error-messages/compiler-errors-1/compiler-error-c2026
    MSVC_C2026_MAX_STRING_LENGTH = 16000

    def codegen_write_arg_with_large_length_string(
        self,
        arg_name: str,
        arg_str_val: str,
        max_truncate_length: int = MSVC_C2026_MAX_STRING_LENGTH,
    ):
        def truncate_string(s: str, length: int) -> list[str]:
            return [s[i : i + length] for i in range(0, len(s), length)]

        if len(arg_str_val) > max_truncate_length:
            truncated_strs = truncate_string(arg_str_val, max_truncate_length)
            self.prefix.writeline(f"{arg_name} =")
            for truncate_str in truncated_strs:
                self.prefix.writeline(f'R"({truncate_str})"')
            self.prefix.writeline(";")
        else:
            self.prefix.writeline(f'{arg_name} = R"({arg_str_val})";')

    def codegen_model_constructor(self):
        """
        // Generated code example
        AOTInductorModel::AOTInductorModel()
            : AOTInductorModelBase(4, 1) {
        inputs_info_[0].name = "input0";
        inputs_info_[0].dtype = "torch.float16";
        ...
        constants_info_[0].name = "L__self___weight";
        constants_info_[0].dtype = at::kFloat;
        constants_info_[0].offset = 0;
        constants_info_[0].data_size = 8192;
        constants_info_[0].shape = {64, 32};
        constants_info_[0].stride = {32, 1};
        ...
        outputs_info_[0].name = "output0";
        outputs_info_[0].dtype = "torch.float16";
        }
        """

        num_inputs = len(V.graph.graph_inputs)
        num_outputs = len(V.graph.graph_outputs)
        num_constants = len(V.graph.constants)
        include_weights = (
            "true"
            if config.aot_inductor.package_constants_in_so
            and config.aot_inductor.package_constants_on_disk_format != "binary_blob"
            else "false"
        )
        self.prefix.splice(f"""
            {self.aoti_model_class_name}::{self.aoti_model_class_name}(std::shared_ptr<ConstantMap> constants_map,
                                               std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                               const std::string& device_str,
                                               std::optional<std::string> cubin_dir)
                : AOTInductorModelBase({num_inputs},
                                       {num_outputs},
                                       {num_constants},
                                       device_str,
                                       std::move(cubin_dir),
                                       {include_weights}) {{
            """)

        with self.prefix.indent():
            for idx, (name, inp) in enumerate(V.graph.graph_inputs.items()):
                if isinstance(inp, sympy.Expr):
                    raise AssertionError(f"input {name=} cannot be symbolic")
                self.write_input_output_info("inputs_info_", idx, name)

            all_cuda = all(
                V.graph.get_original_value_of_constant(name).is_cuda
                for name in V.graph.constants
                if name not in V.graph.folded_constants
            )
            for idx, name in enumerate(V.graph.constants.keys()):
                tensor = V.graph.get_original_value_of_constant(name)
                if not isinstance(tensor, torch.Tensor):
                    raise AssertionError(
                        f"expected constant {name} to be a Tensor, got {type(tensor)}"
                    )
                self.prefix.writeline(f"""constants_info_[{idx}].name = "{name}";""")
                self.prefix.writeline(
                    f"constants_info_[{idx}].dtype = {self.codegen_dtype(tensor.dtype)};"
                )
                # Mixed-device constants are only supported when the secondary device is CPU
                if tensor.device.type != self.device and tensor.device.type != "cpu":
                    raise AssertionError(
                        f"Mixed-device constants are only supported when the secondary "
                        f"device is CPU. Model device is '{self.device}', but constant "
                        f"'{name}' is on device '{tensor.device}'."
                    )
                # device_index is not needed because it can be set at runtime
                device_type, _ = self.codegen_device(tensor.device).split(", ")
                self.prefix.writeline(
                    f"constants_info_[{idx}].device_type = {device_type};"
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].offset = {tensor.storage_offset()};"
                )

                # If constants to serialize contain cpu tensors, we always align data_size it to 64.
                # When loading the constants, the valid data will depends on the size
                # not the data_size so there won't be correctness issue.
                data_size = (
                    torch.ops.mkldnn._nbytes(tensor)
                    if tensor.is_mkldnn
                    else tensor.untyped_storage().nbytes()
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].data_size = {data_size if all_cuda else _align(data_size)};"
                )

                from_folded = "true" if name in V.graph.folded_constants else "false"
                self.prefix.writeline(
                    f"constants_info_[{idx}].from_folded = {from_folded};"
                )

                if name in V.graph.folded_constants:
                    constant_type_str = "FoldedConstant"
                elif name.startswith("_tensor_constant"):
                    constant_type_str = "TensorConstant"
                elif any(
                    name == normalize_name(parameter_name)
                    for parameter_name in V.graph.named_parameters
                ):
                    constant_type_str = "Parameter"
                elif any(
                    name == normalize_name(buffer_name)
                    for buffer_name in V.graph.named_buffers
                ):
                    constant_type_str = "Buffer"
                else:
                    constant_type_str = "Unknown"
                self.prefix.writeline(
                    f"constants_info_[{idx}].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::{constant_type_str});"
                )

                size_str = ", ".join([str(s) for s in tensor.size()])
                self.prefix.writeline(f"constants_info_[{idx}].shape = {{{size_str}}};")

                stride_str = ", ".join([str(s) for s in tensor.stride()])
                self.prefix.writeline(
                    f"constants_info_[{idx}].stride = {{{stride_str}}};"
                )
                self.prefix.writeline(
                    f"constants_info_[{idx}].layout = static_cast<int32_t>({self.codegen_layout(tensor.layout)});"
                )

                if tensor.is_mkldnn:
                    opaque_metadata_tensor = torch.ops.mkldnn._get_mkldnn_serialized_md(
                        tensor
                    )
                    if opaque_metadata_tensor.dim() != 1:
                        raise AssertionError("Expect opaque_metadata_tensor to be 1-D")

                    opaque_metadata_list = opaque_metadata_tensor.tolist()
                    opaque_metadata_str = self.codegen_shape_tuple(opaque_metadata_list)
                    self.prefix.writeline(
                        f"constants_info_[{idx}].opaque_metadata = {opaque_metadata_str};"
                    )
                if name in V.graph.dynamo_flat_name_to_original_fqn:
                    original_fqn = V.graph.dynamo_flat_name_to_original_fqn.get(
                        name, name
                    )
                elif name in V.graph.allocated_constant_name:
                    original_fqn = V.graph.allocated_constant_name[name]
                else:
                    raise AssertionError("original_fqn must be set for constant")
                self.prefix.writeline(
                    f"""constants_info_[{idx}].original_fqn = "{original_fqn}";"""
                )
            self.prefix.writeline("update_constants_map(std::move(constants_map));")
            self.prefix.writeline("update_constants_array(std::move(constants_array));")

            def escape_string(x):
                return (
                    x.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\t", "\\t")
                )

            # Origin code: self.prefix.writeline(f'in_spec_ = R"({config.aot_inductor.serialized_in_spec})";')
            # Fix msvc C2026 error via codegen_write_arg_with_large_length_string
            self.codegen_write_arg_with_large_length_string(
                arg_name="in_spec_", arg_str_val=config.aot_inductor.serialized_in_spec
            )
            # Origin code: self.prefix.writeline(f'out_spec_ = R"({config.aot_inductor.serialized_out_spec})";')
            # Fix msvc C2026 error via codegen_write_arg_with_large_length_string
            self.codegen_write_arg_with_large_length_string(
                arg_name="out_spec_",
                arg_str_val=config.aot_inductor.serialized_out_spec,
            )

            for idx, output in enumerate(V.graph.graph_outputs):
                if isinstance(output, sympy.Expr):
                    raise AssertionError(f"output {name=} cannot be symbolic")
                name = f"output{idx}"
                self.write_input_output_info("outputs_info_", idx, name)

            self.prefix.writeline(
                "this->kernels_ = std::make_unique<AOTInductorModelKernels>();"
            )

        self.prefix.writeline("}")

    def codegen_const_run_driver(self):
        """
        // Generated code example
        std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
            DeviceStreamType stream,
            AOTIProxyExecutorHandle proxy_executor,
            bool initialization
        ) {
            std::unordered_map<std::string, AtenTensorHandle> folded_constants_map;
            std::vector<AtenTensorHandle> output_handles;
            // build up output_handles over here.
            _const_run_impl(output_handles, stream, proxy_executor);
            // build up folded_constants_map
            return folded_constants_map;
        }
        """

        self.prefix.splice(f"""
            std::unordered_map<std::string, AtenTensorHandle> {self.aoti_model_class_name}::const_run_impl(
                DeviceStreamType stream,
                AOTIProxyExecutorHandle proxy_executor,
                bool initialization
            ) {{
            """)
        if not config.aot_inductor.use_runtime_constant_folding:
            self.prefix.splice("""
                    if (!initialization) {
                        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                                  << "aot_inductor.use_runtime_constant_folding=False\\n";
                    }
                    return {};
                }
                """)
            return

        with self.prefix.indent():
            # This is a mapping to the index of constant folding graph's output
            const_index_mapping: list[tuple[int, str] | None] = [None] * len(
                V.graph.const_output_index
            )
            for idx, (name, _) in enumerate(V.graph.constants.items()):
                if name in V.graph.const_output_index:
                    const_index_mapping[V.graph.const_output_index[name]] = (idx, name)  # type: ignore[call-overload]
            if None in const_index_mapping:
                raise AssertionError(
                    "Not all constant gets mapped for constant folding graph."
                )

            self.prefix.writeline(f"""
                std::unordered_map<std::string, AtenTensorHandle> folded_constants_map;
                folded_constants_map.reserve({len(const_index_mapping)});
                std::vector<AtenTensorHandle> output_handles({len(const_index_mapping)});
                """)

            self.prefix.splice("""
                // The below assignment of output_handles to constants is not used directly.
                // It's only used to memo the correspondence of handle and constants.
                """)

            for output_idx, (const_idx, _) in enumerate(const_index_mapping):  # type: ignore[misc]
                self.prefix.writeline(
                    f"output_handles[{output_idx}] = constants_->at({const_idx});"
                )

            self.prefix.writeline(
                "_const_run_impl(output_handles, stream, proxy_executor);"
            )

            for output_idx, (_, const_name) in enumerate(const_index_mapping):  # type: ignore[misc]
                self.prefix.writeline(
                    f'folded_constants_map["{const_name}"] = output_handles[{output_idx}];'
                )
            self.prefix.writeline("return folded_constants_map;")

        self.prefix.writeline("}")

    def generate(self, is_inference):
        with dynamo_timed("CppWrapperCpu.generate", log_pt2_compile_event=True):
            self.write_wrapper_decl()
            return super().generate(is_inference)

    def _codegen_aoti_model_class(self):
        """Emit AOTI model class, constructor, and const_run driver into self.prefix."""
        self.prefix.writeline("namespace torch::aot_inductor {")
        self.codegen_model_kernels()
        self.codegen_model_constructor()
        self.codegen_const_run_driver()
        self.prefix.writeline("} // namespace torch::aot_inductor")
        self.prefix.writeline("using namespace torch::aot_inductor;")

    def finalize_prefix(self):
        prior = self.prefix

        # Build AOTI model class into a temporary buffer (AOTI-only content).
        # _codegen_aoti_model_class calls methods that write to self.prefix,
        # so redirect self.prefix to the temporary buffer during the call.
        aot_mode_decls = IndentedBuffer()
        if V.graph.aot_mode and not V.graph.is_const_graph:
            with self._target_buf("prefix", aot_mode_decls):
                self._codegen_aoti_model_class()
        elif not V.graph.is_const_graph and self.initialized_kernels:
            # JIT cpp_wrapper: emit extern "C" decls for kernels (CUTLASS,
            # ROCm CK) whose precompiled .so files are linked into the
            # wrapper via extra_flags.
            self.codegen_initialized_kernel_decls()

        # Reset self.prefix and write shared cache decls + CPU triton content.
        self.prefix = make_codegen_buffer()
        for dtype in self.used_cached_dtypes:
            self.prefix.writeline(f"CACHE_TORCH_DTYPE({dtype});")
        for device in self.used_cached_devices:
            self.prefix.writeline(f"CACHE_TORCH_DEVICE({device});")
        for layout in self.used_cached_layouts:
            self.prefix.writeline(f"CACHE_TORCH_LAYOUT({layout});")
        for memory_format in self.used_cached_memory_formats:
            self.prefix.writeline(f"CACHE_TORCH_MEMORY_FORMAT({memory_format});")

        # Includes are needed whenever a kernel was *defined* (the model
        # kernels class references `CpuTritonLauncherFn`); `call_<k>` bodies
        # only for kernels that were actually *called*.
        if self._cpu_triton_kernel_names:
            self._write_cpu_triton_runtime_includes(self.prefix)
        if self._cpu_triton_call_wrappers:
            for wrapper in self._cpu_triton_call_wrappers.values():
                self.prefix.writeline("\n")
                wrapper.generate(self)

        # aot_mode_decls is AOTI-only; splice_aot routes correctly across
        # all three modes (no-op on plain IndentedBuffer in pure-JIT).
        self.prefix.splice_aot(aot_mode_decls)
        self.prefix.splice(prior)

    @staticmethod
    def _write_cpu_triton_runtime_includes(prefix: IndentedBuffer) -> None:
        """One-time includes/guards needed by emitted CPU Triton wrappers."""
        prefix.writeline("// CPU AOTI Triton kernel wrappers")
        prefix.writeline("#ifdef _WIN32")
        prefix.writeline(
            '#error "CPU AOTI Triton kernels are not supported on Windows"'
        )
        prefix.writeline("#endif")
        prefix.writeline("#include <mutex>")
        prefix.writeline(
            "#include <torch/csrc/inductor/aoti_runtime/cpu_triton_runtime_wrappers.h>"
        )

    @staticmethod
    def _stringify_cpu_triton_call_arg(arg: Any) -> str:
        """Render a Triton kernel call argument as a C++ expression."""
        if isinstance(arg, str):
            return arg
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float, SymbolicCallArg)):
            return str(arg)
        return cexpr(V.graph.sizevars.simplify(arg))

    def _define_kernel_helper(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: str | None = None,
        gpu: bool = False,
        cpp_definition: str | None = None,
    ):
        # Misnomer: `gpu` actually means "is this a Triton kernel?"
        if gpu:
            # Register as a CPU Triton kernel and let the autotune block
            # compile it; the C++ wrapper is emitted later in finalize_prefix.
            self._cpu_triton_kernel_names.add(kernel_name)
            if config.triton.autotune_at_compile_time:
                PythonWrapperCodegen._define_kernel_helper(
                    self, kernel_name, kernel_body, metadata, gpu, cpp_definition
                )
            return
        if cpp_definition is not None:
            self.needs_vec_isa = True
            self.header.splice(cpp_definition)
            self.kernel_declarations.splice(f"\n{kernel_body}\n")
        else:
            self.header.splice(f"\n{kernel_body}\n")

    def codegen_scalar_to_tensor(self, output: str):
        name = f"scalar_to_tensor_{next(self.scalar_to_tensor_id)}"
        self.wrapper_call.writeline(
            f"RAIIAtenTensorHandle {name} = scalar_to_tensor_handle({output});"
        )
        return name

    def codegen_tensor_item(
        self, dtype: torch.dtype, tensor: str, scalar: str, indented_buffer=None
    ):
        dtype_str = str(dtype).split(".")[-1]
        writer = indented_buffer or self

        if dtype == torch.float16 or dtype == torch.bfloat16:
            scalar_tmp = f"{scalar}_tmp"
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar_tmp};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar_tmp}));"
            )
            writer.writeline(f"float {scalar} = float({scalar_tmp});")
        else:
            writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar};")
            writer.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar}));"
            )

    def generate_return(self, output_refs: list[str]):
        cst_names = V.graph.constants.keys()
        output2idx: dict[str, int] = {}

        # If any output ref represents an rvalue tensor, materialize it to an lvalue
        # RAIIAtenTensorHandle first.  This prevents situations where the code for the
        # rvalue tensor references tensor handles whose contents are modified below.
        output_refs = [
            self.create_tmp_raii_handle_var_if_needed(o, self.wrapper_call)
            for o in output_refs
        ]

        for idx, output in enumerate(output_refs):
            if output == "nullptr":
                continue

            is_constant_buffer = output in cst_names
            output_buffer = V.graph.graph_outputs[idx]
            if isinstance(output_buffer, ir.BaseView):
                output_storage = output_buffer.unwrap_view()
                if not isinstance(output_storage, (ir.BaseView, ir.MutableBox)):
                    raise AssertionError(
                        f"expected output_storage to be BaseView or MutableBox, "
                        f"got {type(output_storage)}"
                    )
                if isinstance(output_storage.data, ir.ConstantBuffer):
                    is_constant_buffer = True

            if isinstance(output_buffer, ir.ShapeAsConstantBuffer):
                # Need to wrap scalar into tensor as the main function returns a vector of tensors
                output_tensor = self.codegen_scalar_to_tensor(output)
                self.wrapper_call.writeline(
                    f"output_handles[{idx}] = {output_tensor}.release();"
                )
                continue

            if is_constant_buffer:
                # See NOTE(return_constant) above.
                self.wrapper_call.writeline(
                    "AOTI_TORCH_ERROR_CODE_CHECK("
                    f"aoti_torch_clone({output}, &output_handles[{idx}]));"
                )
            else:
                if output in output2idx:
                    src_idx = output2idx[output]
                    self.wrapper_call.writeline(
                        f"output_handles[{idx}] = output_handles[{src_idx}];"
                    )
                else:
                    self.wrapper_call.writeline(
                        f"output_handles[{idx}] = {output}.release();"
                    )

            if output not in output2idx:
                output2idx[output] = idx

    def _codegen_entry_impl_prologue(self):
        """Hook for subclasses to emit code at the top of inductor_entry_impl,
        before the GIL is released. Default no-op."""
        if config.triton.debug_sync_graph:
            self.generate_debug_sync(self.prefix)

    def generate_before_suffix(self, result):
        # Close the entry function. In dual-wrapper-mode `result` is the JIT side;
        # the AOTI side close is emitted by _generate_end_aoti.
        if V.graph.aot_mode and not V.graph.is_dual_wrapper_mode:
            entry = "_const_run_impl" if V.graph.is_const_graph else "run_impl"
            result.writeline(f"}} // {self.aoti_model_class_name}::{entry}")
        else:
            result.writeline("} // inductor_entry_impl")

    def _generate_end_aoti(self, result):
        """Close AOTI namespace. In dual-wrapper-mode, also assemble the
        AOTI source string from `.aot` buffers into self._aot_output; the
        input `result` is the JIT side and is left untouched in that case.
        """
        if V.graph.is_dual_wrapper_mode:
            if not isinstance(self.header, DualIndentedBuffer):
                raise AssertionError(
                    f"expected DualIndentedBuffer, got {type(self.header)}"
                )
            if not isinstance(self.prefix, DualIndentedBuffer):
                raise AssertionError(
                    f"expected DualIndentedBuffer, got {type(self.prefix)}"
                )
            if not isinstance(self.wrapper_call, DualIndentedBuffer):
                raise AssertionError(
                    f"expected DualIndentedBuffer, got {type(self.wrapper_call)}"
                )
            aot_result = IndentedBuffer()
            aot_result.splice(self.header.aot)
            aot_result.splice(self.prefix.aot)
            with aot_result.indent():
                aot_result.splice(self.wrapper_call.aot)
            entry = "_const_run_impl" if V.graph.is_const_graph else "run_impl"
            aot_result.writeline(f"}} // {self.aoti_model_class_name}::{entry}")
            aot_result.splice(self.suffix)
            if not V.graph.is_const_graph:
                aot_result.writeline("} // namespace torch::aot_inductor\n\n\n")
            self._aot_output = aot_result.getvalue()
        elif not V.graph.is_const_graph:
            result.writeline("} // namespace torch::aot_inductor\n\n\n")

    def _generate_end_jit(self, result):
        """Close JIT C++ block and emit Python wrapper for loading/calling."""
        if config.cpp_wrapper_build_separate:
            # Close the wrapper code block, then write any kernel definitions.
            result.splice('"""\n)')
            if self.kernel_declarations:
                result.splice('\nkernel_src = (\nr"""')
                result.splice(self.kernel_declarations.getvalue())
                result.splice('"""\n)')
            else:
                result.splice("""
                    kernel_src = ''
                    """)
        else:
            # Merge main code and kernel code
            result.splice(self.kernel_declarations.getvalue())
            self.kernel_declarations.clear()
            # Close the wrapper code block
            result.splice('"""\n)')

        if config.cpp_wrapper_build_separate:
            kernel_code = "kernel_src"
            needs_vec_isa = "False"
            kernel_needs_vec_isa = str(self.needs_vec_isa)
        else:
            kernel_code = "None"
            needs_vec_isa = str(self.needs_vec_isa)
            kernel_needs_vec_isa = "None"
        # Link any precompiled kernel .so files (CUTLASS, ROCm CK) into the
        # wrapper. The .so files are built without -Wl,-soname, so passing
        # their absolute paths positionally to the linker encodes them as
        # absolute DT_NEEDED entries that resolve at module load (same
        # mechanism HalideCodeCache uses for its standalone runtime).
        extra_flags_repr = repr(tuple(self.external_kernel_libs))
        # Cpp entry function for JIT with cpp wrapper
        result.splice(f"""
            inductor_entry = CppWrapperCodeCache.load_pybinding(
                argtypes=["std::vector<AtenTensorHandle>"],
                main_code=cpp_wrapper_src,
                device_type="{self.device}",
                needs_vec_isa={needs_vec_isa},
                kernel_needs_vec_isa={kernel_needs_vec_isa},
                num_outputs={len(V.graph.graph_outputs)},
                kernel_code={kernel_code},
                extra_flags={extra_flags_repr},
            )
            """)

        input_python_indices = [
            idx
            for idx, value in enumerate(V.graph.graph_inputs.values())
            if isinstance(
                value,
                (ir.TorchBindObject, ir.GeneratorState, ir.OpaqueObjectState),
            )
        ]
        python_only_input_indices = (
            "set()"
            if not input_python_indices
            else "{" + ", ".join(map(str, input_python_indices)) + "}"
        )
        wrapper_body = f"""
                    python_only_input_indices = {python_only_input_indices}
                    input_tensors = []
                    input_handle_positions = []
                    # Preserve graph input slots: Python-only inputs stay None
                    # and are unpacked as nullptrs by the JIT cpp wrapper.
                    input_handles = [None] * len(args)
                    for idx, arg in enumerate(args):
                        if idx in python_only_input_indices:
                            continue
                        input_handle_positions.append(idx)
                        input_tensors.append(
                            arg if isinstance(arg, torch.Tensor) else torch.tensor(arg, device='cpu')
                        )
        """
        if V.graph.constants:
            # Append constants to the input args for cpp wrapper.
            # Python wrapper directly gets the value inside the wrapper call
            # as a global variable passed when calling exec(code, mod.__dict__, mod.__dict__).
            # For cpp wrapper, we need to pass this python value to the inductor_entry_impl function explicitly.
            if not all(
                isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
            ):
                raise AssertionError("Expect all constants to be Tensor")
            constants_str = f"[{', '.join(V.graph.constants.keys())}]"
            wrapper_body += f"""
                    constants_tensor = {constants_str}
                    constants_start_idx = len(input_handles)
                    input_handles.extend([None] * len(constants_tensor))
                    input_handle_positions.extend(
                        range(constants_start_idx, constants_start_idx + len(constants_tensor))
                    )
                    input_tensors.extend(constants_tensor)
            """
        # Convert vector of at::Tensor to vector of AtenTensorHandle.
        # If we pass at::Tensor, the compilation will be too slow.
        wrapper_body += """
                    packed_input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(input_tensors)
                    for idx, handle in zip(input_handle_positions, packed_input_handles):
                        input_handles[idx] = handle
        """
        # Release the inputs for memory reuse.
        wrapper_body += """
                    args.clear()
                    del input_handle_positions
                    del packed_input_handles
                    del input_tensors
        """

        # unwrap output tensor back to python scalar
        if all(x for x in self.output_is_tensor.values()):
            # If no ShapeAsConstantBuffer in the output, directly return the output as tensors
            outputs_str = "output_tensors"
        else:
            outputs = [
                (
                    f"output_tensors[{i}]"
                    if self.output_is_tensor[i]
                    else f"output_tensors[{i}].item()"
                )
                for i in range(len(V.graph.graph_outputs))
            ]
            outputs_str = f"[{', '.join(outputs)}]"
        wrapper_body += f"""
                    output_handles = f(input_handles)
                    output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(output_handles)
                    return {outputs_str}
        """

        # Wrap the func to support setting result._boxed_call = True
        result.splice(f"""
            def _wrap_func(f):
                def g(args):
                    {wrapper_body}
                return g

            call = _wrap_func(inductor_entry)
            """)

    def generate_end(self, result):
        """Generates the end of the code block, and any code needed to call it."""
        if V.graph.aot_mode:
            self._generate_end_aoti(result)
        else:
            self._generate_end_jit(result)

    @staticmethod
    def get_c_shim_func_name(kernel: str, device: str) -> str:
        if kernel.startswith("aoti_torch_"):
            return kernel

        if "::" not in kernel:
            raise AssertionError(
                "Cpp kernel name: " + kernel + " does not contain '::'"
            )
        kernel_tokens = kernel.split("::")
        kernel_suffix = kernel_tokens[-1]
        if kernel_suffix == "call":
            kernel_suffix = kernel_tokens[-2]

        shim_fn = f"aoti_torch_{device}_{kernel_suffix}"
        return shim_fn

    def generate_c_shim_extern_kernel_call(
        self,
        kernel: str,
        args: list[str],
        device: str,
        *,
        debug_args: list[str] | None = None,
        stack_traces: OrderedSet[str] | None = None,
    ) -> None:
        """debug_args kwarg allows CppWrapperCpuArrayRef to pass in wrapped arguments in
        place of args while preserving debug printer output."""
        # We can do this unconditionally, since we cache this call.
        self.add_device_include(device)

        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(
            debug_args if debug_args is not None else args, kernel, None, None, "extern"
        )
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        enable_kernel_context_guard = (
            enable_kernel_profile and config.cpp.enable_kernel_context_guard
        )
        with debug_printer_manager:
            shim_fn = self.get_c_shim_func_name(kernel, device)
            shim_fn_codes = [
                f"AOTI_TORCH_ERROR_CODE_CHECK({shim_fn}({', '.join(args)}));"
            ]
            if enable_kernel_profile:
                call_code = shim_fn_codes[0]
                shim_fn_codes = ["{"]
                if enable_kernel_context_guard:
                    stack_trace_str = 'R"('
                    if stack_traces:
                        for stack_trace in stack_traces:
                            for line in stack_trace.split("\n"):
                                stack_trace_str += f"\n{line}"
                            stack_trace_str += "\n"
                    stack_trace_str += ')"'
                    shim_fn_codes.append(
                        f"""KernelContextGuard _ctx("{shim_fn}", {stack_trace_str});"""
                    )
                shim_fn_codes.extend(
                    [
                        f"""RAIIAtenRecordFunctionHandle record_{shim_fn}_("{shim_fn}", nullptr);""",
                        call_code,
                        "}",
                    ]
                )
            self.writelines(shim_fn_codes)

    def generate_c_shim_extern_kernel_alloc(
        self, extern_kernel: ir.ExternKernelAlloc, args: list[str]
    ) -> None:
        # registered output buffer name
        name = extern_kernel.name
        output_handle_name = f"{name}_handle"
        is_inplace = (
            isinstance(extern_kernel.op_overload, torch._ops.OpOverload)
            and torch.Tag.inplace_view in extern_kernel.op_overload.tags
        )

        if not is_inplace:
            self.writeline(f"AtenTensorHandle {output_handle_name};")
            args = [*args, f"&{output_handle_name}"]

        device = d.type if (d := extern_kernel.get_device()) else self.device

        self.generate_c_shim_extern_kernel_call(
            extern_kernel.get_kernel_name(), args, device
        )

        if extern_kernel.python_kernel_name in (
            "torch.ops._c10d_functional.all_reduce_.default",
            "torch.ops._c10d_functional.wait_tensor.default",
        ):
            # all_reduce_ is an inplace op and its returned tensor is not used anywhere.
            # wait_tensor returns its input without any modification and the returned tensor is not used anywhere.
            # In both cases, we can immediately delete the returned AtenTensorHandle to reduce its lifetime.
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_delete_tensor_object({output_handle_name}));"
            )
        elif not is_inplace:
            self.writeline(f"RAIIAtenTensorHandle {name}({output_handle_name});")

    def _generate_extern_kernel_alloc_helper(self, extern_kernel, args):
        if getattr(extern_kernel, "outputs", None):
            # ir.ExternKernelAlloc may have outputs if it returns a tuple
            self.generate_c_shim_fallback_kernel(extern_kernel, args)
        else:
            self.generate_c_shim_extern_kernel_alloc(extern_kernel, args)

    def generate_c_shim_fallback_kernel(
        self, fallback_kernel: ir.FallbackKernel, args: list[str]
    ) -> None:
        output_args = []
        output_raii_handles = []
        output_name_base = fallback_kernel.get_name()
        for idx, output in enumerate(fallback_kernel.outputs):
            if isinstance(output, ir.MultiOutput):
                # TODO: handle integer output (e.g., as in attention)
                name = f"{output.get_name()}"
                output_handle_name = f"{name}_handle"
                if output.indices:
                    if output.indices[0][1] != idx:
                        raise AssertionError(
                            f"expected {output.indices[0][1]=} == {idx=} for {output_name_base=}"
                        )
                self.writeline(f"AtenTensorHandle {output_handle_name};")
                output_args.append(f"&{output_handle_name}")
                output_raii_handles.append(
                    f"RAIIAtenTensorHandle {name}({output_handle_name});"
                )
            elif isinstance(output, int):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"int64_t {output_name} = {output};")
                output_args.append(f"&{output_name}")
            elif isinstance(output, sympy.Expr):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"auto {output_name} = {cexpr(output)};")
                output_args.append(f"&{output_name}")
            elif output is None:
                output_args.append("nullptr")
            else:
                raise NotImplementedError(f"unsupported type of {output=}")
        args = args + output_args
        device = d.type if (d := fallback_kernel.get_device()) else self.device

        self.generate_c_shim_extern_kernel_call(
            fallback_kernel.cpp_kernel_name,  # type: ignore[arg-type]
            args,
            device,
        )
        for raii_handle in output_raii_handles:
            self.writeline(raii_handle)

    def _generate_extern_kernel_out_helper(
        self,
        kernel: str,
        out: str,
        out_view: str | None,
        args: list[str],
        device: str,
        stack_traces: OrderedSet[str] | None = None,
    ) -> None:
        if out_view:
            out_name = f"{out}_as_strided"
            self.writeline(f"auto {out_name} = {out_view};")
            args.insert(0, out_name)
        else:
            args.insert(0, out)

        self.generate_c_shim_extern_kernel_call(
            kernel, args, device, stack_traces=stack_traces
        )

    def _get_scatter_reduce_enum(self, reduce):
        # Follow aten/src/ATen/native/ReductionType.h:get_operator_enum
        get_operator_enum = {"add": "sum", "multiply": "prod"}
        if reduce in get_operator_enum:
            reduce = get_operator_enum[reduce]

        return reduce

    def _generate_scatter_fallback_args(self, inputs: Sequence[Any]) -> list[str]:
        return [str(x) for x in inputs]

    def _generate_scatter_fallback(
        self,
        output,
        inputs,
        cpp_kernel_name,
        python_kernel_name,
        src_is_tensor,
        reduce,
        kwargs,
        device,
    ):
        reduce = self._get_scatter_reduce_enum(reduce)

        # call the ABI shim function instead of the ATen one
        self.add_device_include(device)
        cpp_kernel_name = self.get_c_shim_func_name(cpp_kernel_name, device)
        # TODO: consider remove "_out" and add missing inplace variants to fallback_ops.py
        cpp_kernel_name = cpp_kernel_name.replace("__", "_") + "_out"
        # str(output) ensures that CppWrapperCpuArrayRef borrows the output tensor
        args_wrapped = self._generate_scatter_fallback_args((str(output), *inputs))
        # Wrap in AOTI_TORCH_ERROR_CODE_CHECK so a shim failure
        # surfaces instead of being silently swallowed.
        line = f"{cpp_kernel_name}({','.join(args_wrapped)}"

        if python_kernel_name.startswith("aten.scatter_reduce"):
            line += f", {','.join(kwargs)}"
        elif src_is_tensor:
            if reduce:
                line += f", {V.graph.wrapper_code.val_to_arg_str(reduce)}"
        elif reduce is not None:
            raise AssertionError(
                "Expect reduce to be None for aten.scatter_ with scalar src"
            )
        line += ")"
        self.writeline(f"AOTI_TORCH_ERROR_CODE_CHECK({line});")

    def _generate_index_put_fallback(self, kernel, x, indices, values, accumulate):
        # TODO: update aoti_torch_index_put_out in ir.py to use autogen out version
        # See the comment in codegen_reinterpret_view about why having something like
        # RAIIAtenTensorHandle(tmp_tensor_handle_2) in a tmp array can cause the corresponding
        # tensor prematurely deallocated, thus the temporary array trick here.
        indices_str = self._generate_temporary_array_pointer(
            "AtenTensorHandle", indices
        )
        args = [
            x,
            indices_str,
            str(len(indices)),
            values,
            accumulate,
        ]
        args.insert(0, x)  # set x as the output tensor, this fallback mutates x.
        # Wrap in AOTI_TORCH_ERROR_CODE_CHECK so a shim failure surfaces instead
        # of being silently swallowed.
        self.writeline(f"AOTI_TORCH_ERROR_CODE_CHECK({kernel}({', '.join(args)}));")

    def add_benchmark_harness(self, output):
        if V.graph.aot_mode:
            return
        super().add_benchmark_harness(output)

    def _extract_divisors_from_expr(
        self, expr: sympy.Expr
    ) -> list[tuple[sympy.Expr, str]]:
        """
        Walk the sympy expression and extract all divisors/modulos from
        FloorDiv, CleanDiv, Mod, and ModularIndexing nodes.

        Returns a list of (divisor_expr, operation_name) tuples for divisors
        that are not constant positive integers (which cannot be zero).
        """
        divisors: list[tuple[sympy.Expr, str]] = []
        seen: OrderedSet[str] = OrderedSet()

        def is_constant_nonzero(e: sympy.Expr) -> bool:
            """Check if expression is a constant that cannot be zero."""
            if isinstance(e, sympy.Integer):
                return int(e) != 0
            if isinstance(e, sympy.Number):
                return float(e) != 0
            return False

        def maybe_add_divisor(divisor: sympy.Expr, op_name: str) -> None:
            """Add divisor to list if it could potentially be zero."""
            if not is_constant_nonzero(divisor):
                key = str(divisor)
                if key not in seen:
                    seen.add(key)
                    divisors.append((divisor, op_name))

        def walk(e: sympy.Expr) -> None:
            if isinstance(e, (FloorDiv, CleanDiv)):
                _, div = e.args
                maybe_add_divisor(div, "floor division")
            elif isinstance(e, Mod):
