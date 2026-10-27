    output = semantic.dot_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, fast_math=False, lhs_k_pack=True,
                                 rhs_k_pack=True, out_dtype=ttgl.float32)
    return ttgl.tensor(output.handle, acc.type)


def _scaled_upcast(src, scale, elem_type, axis, semantic):
    _check(isinstance(src.type, ttgl.distributed_type),
           lambda: f"Expected src to have a distributed_type but got {src.type}")
    _check(isinstance(scale.type, ttgl.distributed_type),
           lambda: f"Expected scale to have a distributed_type but got {scale.type}")
    _check(elem_type in {ttgl.float16, ttgl.bfloat16},
           lambda: f"Expected elem_type to be fp16 or bf16 but got {elem_type}")

    if src.dtype in {ttgl.float8e4nv, ttgl.float8e5}:
        _check(axis is None, lambda: "axis must be None for fp8 scaled_upcast")
        _check(scale.type.shape == src.type.shape,
               lambda: f"Expected scale shape for fp8 scaled_upcast to be {src.type.shape} but got {scale.type.shape}")
        _check(
            scale.type.layout == src.type.layout,
            lambda: f"Expected scale layout for fp8 scaled_upcast to be {src.type.layout} but got {scale.type.layout}")
        # Note: bf16 is allowed due to CDNA3/CDNA4 conversion before passing to scaled_upcast
        _check(scale.dtype in {ttgl.int8, ttgl.uint8, ttgl.bfloat16},
               lambda: f"Unsupported scale dtype for fp8 scaled_upcast: {scale.dtype}")
        ret_ty = scale.type.with_element_ty(elem_type)
        handle = semantic.builder.create_scaled_upcast_fp8(ret_ty.to_ir(semantic.builder), src.handle, scale.handle)
        return _wrap_scaled_upcast_result(handle, elem_type, semantic)

    _check(src.dtype in {ttgl.int8, ttgl.uint8},
           lambda: f"Expected packed fp4 input in int8/uint8 or fp8 input, but got {src.dtype}")
    _check(axis is not None, lambda: "axis is required for packed fp4 scaled_upcast")

    rank = len(src.type.shape)
    _check(-rank <= axis < rank, lambda: f"axis {axis} out of range for rank {rank}")
    if axis < 0:
        axis += rank

    expected_shape = list(src.type.shape)
    expected_shape[axis] *= 2
    _check(scale.type.shape == expected_shape,
           lambda: f"Expected scale shape for fp4 scaled_upcast to be {expected_shape} but got {scale.type.shape}")
    _check(scale.dtype in {ttgl.int8, ttgl.uint8, ttgl.bfloat16},
           lambda: f"Unsupported scale dtype for fp4 scaled_upcast: {scale.dtype}")

    handle = semantic.builder.create_scaled_upcast_fp4(src.handle, scale.handle, elem_type.to_ir(semantic.builder),
                                                       axis)
    return _wrap_scaled_upcast_result(handle, elem_type, semantic)
<|endoftext|>from ..._core import builtin
from .._ops import _wmma

__all__ = ["wmma"]


@builtin
def wmma(a, b, acc, _semantic=None):
    """
    Computes matrix-multiplication of a * b + acc using AMD WMMA instruction.

    Args:
        a (tensor): The operand a to be multiplied.
        b (tensor): The operand b to be multiplied.
        acc (tensor): The accumulator tensor.
    """
    return _wmma(2, a, b, acc, _semantic)
<|endoftext|>from triton.experimental.gluon.language._core import builtin

__all__ = ["arrive", "wait"]


@builtin
def arrive(_semantic=None):
    """
    Signals that the cluster has arrived at a cluster barrier, used to synchronize execution of CTAs within the same cluster.
    """
    _semantic.builder.create_amd_cluster_arrive()


@builtin
def wait(_semantic=None):
    """
    Wait on a cluster barrier to be arrived by all CTAs within the same cluster.
    Arrive and wait operations must come in pairs. Waiting before arriving or arriving more than once
    without a corresponding wait will result in undefined behavior.
    """
    _semantic.builder.create_amd_cluster_wait()
<|endoftext|>from ..._core import ir, builtin, _unwrap_if_constexpr
from ..._semantic import _check
from triton.experimental.gluon.language._layouts import DistributedLayout
from ..cdna4.async_copy import commit_group, wait_group

__all__ = ["global_to_shared", "shared_to_global", "commit_group", "wait_group", "mbarrier_arrive"]


@builtin
def global_to_shared(smem, pointer, mask=None, other=None, cache_modifier="", _semantic=None):
    """
    Asynchronously copy elements from global memory to shared memory. Requires manual syncronization via `wait_group` before accessing the loaded data.

    Args:
        smem (shared_memory_descriptor): Destination shared memory descriptor.
        pointer (tensor): Source pointer tensor.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor or scalar, optional): Tensor or scalar providing default values for masked elements. Defaults to None(0).
        cache_modifier (str): Cache modifier specifier. Defaults to "".
        eviction_policy (str): Eviction policy specifier. Defaults to "".
    """
    _check(pointer.type.is_block(), lambda: "expected ptr to be a tensor")
    _check(isinstance(pointer.type.layout, DistributedLayout),
           lambda: "expected ptr type layout to be a DistributedLayout")
    _check(
        smem.shape == pointer.shape, lambda:
        f"expected smem shape to match pointer shape but got smem.shape = {smem.shape}, pointer.shape = {pointer.shape}"
    )
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
    other = _unwrap_if_constexpr(other)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, pointer.dtype.element_ty)
        pointer, other = _semantic.broadcast_impl_value(pointer, other)
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    mask_handle = mask.handle if mask is not None else ir.value()
    other_handle = other.handle if other is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(smem.handle, pointer.handle, mask_handle, other_handle,
                                                        cache_modifier, ir.EVICTION_POLICY.NORMAL, False)


@builtin
def shared_to_global(pointer, smem, mask=None, cache_modifier="", _semantic=None):
    """
    Asynchronously copy elements from shared memory to global memory. Requires manual syncronization via `wait_group` before accessing the stored data.

    Args:
        pointer (tensor): Destination pointer tensor.
        smem (shared_memory_descriptor): Source shared memory descriptor.
        mask (tensor, optional): Mask tensor for predicated stores. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    _check(pointer.type.is_block(), lambda: "expected ptr to be a tensor")
    _check(isinstance(pointer.type.layout, DistributedLayout),
           lambda: "expected ptr type layout to be a DistributedLayout")
    _check(
        smem.shape == pointer.shape, lambda:
        f"expected smem shape to match pointer shape but got smem.shape = {smem.shape}, pointer.shape = {pointer.shape}"
    )
    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        pointer, mask = _semantic.broadcast_impl_value(pointer, mask)
    cache_modifier = _semantic._str_to_store_cache_modifier(cache_modifier)
    mask_handle = mask.handle if mask is not None else ir.value()
    _semantic.builder.create_async_copy_local_to_global(smem.handle, pointer.handle, mask_handle, cache_modifier,
                                                        ir.EVICTION_POLICY.NORMAL)


@builtin
def mbarrier_arrive(mbarrier, _semantic=None):
    """
    Arrive on the mbarrier once all outstanding async copies are complete.
    Args:
        mbarrier (shared_memory_descriptor): Barrier object to arrive on.
    """
    _semantic.builder.create_async_copy_lds_barrier_arrive(mbarrier.handle)
<|endoftext|>from __future__ import annotations

from dataclasses import dataclass
from triton.language.core import _unwrap_if_constexpr
from triton.runtime.jit import constexpr_function

from triton.experimental.gluon.language._layouts import PaddedSharedLayout, SharedLayout

__all__ = [
    "PartitionedSharedLayout",
    "make_partitioned_dot_layouts",
]


@dataclass(frozen=True)
class PartitionedSharedLayout(SharedLayout):
    """
    Represents a partitioned shared memory layout that splits a tensor
    across multiple physical shared memory partitions.

    This reduces shared memory partition conflicts by placing different
    pieces of a tensor in separate physical memory slots.

    Args:
        num_partitions (int): Number of physical memory partitions.
        num_groups (int): Number of groups (each group has num_partitions pieces).
        partition_dim (int): Dimension along which to partition.
        partition_layout (SharedLayout): Inner layout for each piece
            (e.g., SwizzledSharedLayout or PaddedSharedLayout).
    """
    num_partitions: int
    num_groups: int
    partition_dim: int
    partition_layout: SharedLayout

    def __post_init__(self):
        super().__setattr__("num_partitions", _unwrap_if_constexpr(self.num_partitions))
        super().__setattr__("num_groups", _unwrap_if_constexpr(self.num_groups))
        super().__setattr__("partition_dim", _unwrap_if_constexpr(self.partition_dim))
        super().__setattr__("partition_layout", _unwrap_if_constexpr(self.partition_layout))

        is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
        assert is_power_of_2(self.num_partitions), \
            f"PartitionedSharedLayout num_partitions must be a power of two, got {self.num_partitions}"
        assert is_power_of_2(self.num_groups), \
            f"PartitionedSharedLayout num_groups must be a power of two, got {self.num_groups}"
        assert self.partition_dim >= 0, \
            f"PartitionedSharedLayout partition_dim must be non-negative, got {self.partition_dim}"
        assert isinstance(self.partition_layout, SharedLayout), \
            f"PartitionedSharedLayout partition_layout must be a SharedLayout, got {type(self.partition_layout)}"

    def _to_ir(self, builder):
        partition_layout_ir = self.partition_layout._to_ir(builder)
        return builder.get_partitioned_shared_layout(
            self.num_partitions,
            self.num_groups,
            self.partition_dim,
            partition_layout_ir,
        )

    def mangle(self) -> str:
        partition_mangle = self.partition_layout.mangle()
        return f"Partitioned_{self.num_partitions}_{self.num_groups}_{self.partition_dim}_{partition_mangle}_Partitioned"

    def __hash__(self):
        return hash((
            self.num_partitions,
            self.num_groups,
            self.partition_dim,
            self.partition_layout,
        ))


@constexpr_function
def _make_partitioned_dot_operand_layout(sublayout, partition_dim, num_partitions, block_mn_size, warp_coverage, order):
    """Build a ``PartitionedSharedLayout`` for one GEMM operand.

    The operand tile (``sublayout``) is split along ``partition_dim`` into
    ``num_partitions * num_groups`` logical pieces, where
    ``num_groups = block_mn_size // warp_coverage``.  Each piece carries an
    inner ``PaddedSharedLayout`` rebuilt with an identity mapping using the
    original sublayout's ``interval_padding_pairs`` and ``cga_layout``.
    """
    is_power_of_2 = lambda n: n > 0 and n & (n - 1) == 0
    num_groups = block_mn_size // warp_coverage
    assert num_groups >= 1, f"block dim ({block_mn_size}) must be >= {warp_coverage}"
    assert is_power_of_2(num_groups), \
        f"block_dim / warp_coverage = {num_groups} must be a power of 2"

    num_logical_pieces = num_partitions * num_groups
    inner_shape = list(sublayout.shape)
    assert inner_shape[partition_dim] % num_logical_pieces == 0
    inner_shape[partition_dim] //= num_logical_pieces

    # TODO: when the partitioned dimension is the contiguous (fastest-varying)
    # dimension, the interval_padding_pairs may need adjustment to preserve
    # bank-conflict avoidance properties for the smaller piece shape.
    inner = PaddedSharedLayout.with_identity_for(sublayout.interval_padding_pairs, inner_shape, order,
                                                 sublayout.cga_layout)
    return PartitionedSharedLayout(num_partitions, num_groups, partition_dim, inner)


@constexpr_function
def make_partitioned_dot_layouts(block_m, block_n, original_layout_a, original_layout_b, num_warps, instr_shape,
                                 a_transposed=False, b_transposed=False):
    """Create partitioned shared memory layouts and WMMA layout for a GFX1250 GEMM
       in order to avoid LDS partition conflicts.

    Args:
        block_m: M dimension tile size.  Must be at least
            ``WARP_TILES_M * instr_shape[0]`` (the per-CTA M extent covered by
            one partition group), and a power-of-2 multiple thereof.
        block_n: N dimension tile size.  Must be at least
            ``WARP_TILES_N * instr_shape[1]``, and a power-of-2 multiple
            thereof.
        original_layout_a: ``PaddedSharedLayout`` for operand A.  Shape is
            ``[block_m, block_k]`` when not transposed (K contiguous) and
            ``[block_k, block_m]`` when transposed (M contiguous).
        original_layout_b: ``PaddedSharedLayout`` for operand B.  Shape is
            ``[block_k, block_n]`` when not transposed (N contiguous) and
            ``[block_n, block_k]`` when transposed (K contiguous).
        num_warps: Number of warps per CTA.  Currently must be 4 or 8.
        instr_shape: WMMA instruction shape as ``[M, N, K]``.
        a_transposed: Whether A is transposed in shared memory, i.e. M is
            the contiguous axis instead of K.
        b_transposed: Whether B is transposed in shared memory, i.e. K is
            the contiguous axis instead of N.

    Returns:
        A tuple ``(shared_layout_a, shared_layout_b, wmma_layout)``.
    """
    from triton.experimental.gluon.language.amd._layouts import AMDWMMALayout

    INSTR_SHAPE = list(instr_shape)

    assert num_warps in (4, 8), f"Only 4 or 8 warps are currently supported, got {num_warps}"

    NUM_PARTITIONS = 2

    # The caller passes each sublayout with its contiguous axis at dim 1 of the
    # tile (i.e. the tile's shape is already expressed in memory order):
    #   A non-transposed: [block_m, block_k] — K contiguous at dim 1
    #   A transposed:     [block_k, block_m] — M contiguous at dim 1
    #   B non-transposed: [block_k, block_n] — N contiguous at dim 1
    #   B transposed:     [block_n, block_k] — K contiguous at dim 1
    # So the linear component is always built with order [1, 0] regardless of
    # transposition. TDM additionally requires order [rank-1, ..., 0].
    order = [1, 0]

    # WMMA CTA layout: Below, M runs vertically (rows) and N
    # horizontally (cols); each cell is one INSTR_SHAPE-sized instruction tile,
    # labelled with the warp that computes it.  ``warp_bases`` map warp-id bits
    # to (M, N) tile offsets, ``reg_bases`` map register-id
    # (more specifically, instruction repetition registers) bits the same way.
    #
    # 4-warp case (warp_bases = [[2, 1], [1, 0]], reg_bases = [[2, 0]]):
    #
    #   M=0:  w0 w1   <- second tile computed by w1 (reg=1)
    #   M=1:  w2 w3
    #   M=2:  w0 w1   <- first tile computed by w1 (reg=0)
    #   M=3:  w2 w3
    #
    # Such layout allows w0 and w1 as well as w2 and w3 to read different A/B
    # operand data blocks in a single instruction, which is necessary precondition
    # for avoiding LDS partition conflicts.
    if num_warps == 4:
        warp_bases = [[2, 1], [1, 0]]
        reg_bases = [[2, 0]]
    else:  # num_warps == 8
        # Same idea as the 4-warp case, but the third warp bit replaces the
        # register bit. This means there's no need to define repetition registers,
        # single instruction CTA layout is enough to describe the layout we need.
        # Each warp now owns a single instruction tile and the extra warp dimension
        # is folded into the M-axis of the warp grid.
        warp_bases = [[2, 1], [1, 0], [2, 0]]
        reg_bases = []

    # The tile extent along each dimension is 2^m, where m is the largest
    # basis component in that dimension across both warp and register bases.
    def _tile_extent(dim):
        m = max((b[dim] for b in warp_bases + reg_bases), default=0)
        return 1 << m

    WARP_TILES_M = _tile_extent(0)
    WARP_TILES_N = _tile_extent(1)

    wmma_layout = AMDWMMALayout(3, True, warp_bases, reg_bases, INSTR_SHAPE)

    # Per-CTA extent that one warp+register cycle covers in each dimension.
    warp_coverage_m = WARP_TILES_M * INSTR_SHAPE[0]
    warp_coverage_n = WARP_TILES_N * INSTR_SHAPE[1]

    # Partition A along its M axis and B along its N axis.  These dims live
    # at different positions in the tile depending on transposition (the
    # contiguous axis is always at dim 1, so M / N moves to dim 0 or dim 1).
    a_partition_dim = 1 if a_transposed else 0
    b_partition_dim = 0 if b_transposed else 1

    shared_layout_a = _make_partitioned_dot_operand_layout(original_layout_a, partition_dim=a_partition_dim,
                                                           num_partitions=NUM_PARTITIONS, block_mn_size=block_m,
                                                           warp_coverage=warp_coverage_m, order=order)
    shared_layout_b = _make_partitioned_dot_operand_layout(original_layout_b, partition_dim=b_partition_dim,
                                                           num_partitions=NUM_PARTITIONS, block_mn_size=block_n,
                                                           warp_coverage=warp_coverage_n, order=order)

    return shared_layout_a, shared_layout_b, wmma_layout
<|endoftext|>from triton.runtime.jit import constexpr_function
from triton._C.libtriton.gluon_ir import get_amd_wmma_scale_layout as _get_wmma_scale_layout

from ..._core import builtin, int8, uint8, int32, float8e4nv, tensor, _unwrap_if_constexpr
from .._ops import _wmma, _verify_wmma, _mma_scaled, _scaled_upcast
from .._layouts import AMDWMMALayout
from ..cdna3 import buffer_load, buffer_store
from ._layouts import PartitionedSharedLayout, make_partitioned_dot_layouts
from . import tdm
from . import async_copy
from . import mbarrier
from . import cluster

__all__ = [
    "async_copy", "tdm", "mbarrier", "cluster", "wmma", "wmma_scaled", "scaled_upcast", "buffer_load", "buffer_store",
    "get_wmma_scale_layout", "PartitionedSharedLayout", "make_partitioned_dot_layouts"
]


@builtin
def wmma(a, b, acc, _semantic=None):
    """
    Computes matrix-multiplication of a * b + acc using AMD WMMA instruction.

    Args:
        a (tensor): The operand a to be multiplied.
        b (tensor): The operand b to be multiplied.
        acc (tensor): The accumulator tensor.
    """
    return _wmma(3, a, b, acc, _semantic)


# List all valid combinations directly for readability
_valid_dtype_combinations = set([(dtype_a, dtype_b, "e8m0", "e8m0")
                                 for dtype_a in ("e4m3", "e5m2", "e2m1")
                                 for dtype_b in ("e4m3", "e5m2", "e2m1")] + [(dtype_a, "e2m1", "e8m0", dtype_b_scale)
                                                                             for dtype_a in ("e4m3", "e5m2")
                                                                             for dtype_b_scale in ("e4m3", )] +
                                [("e2m1", dtype_b, dtype_a_scale, "e8m0")
                                 for dtype_b in ("e4m3", "e5m2")
                                 for dtype_a_scale in ("e4m3", )] + [("e2m1", "e2m1", dtype_scale, dtype_scale)
                                                                     for dtype_scale in ("e4m3", )])


@builtin
def wmma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _semantic=None):
    """
    AMD Scaled WMMA operation.

    ```
    c = a * a_scale @ b * b_scale + acc
    ```

    `a` and `b` use microscaling formats described in
    "OCP Microscaling Formats (MX) Specification":
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.

    Args:
        a (tensor): The operand A to be multiplied.
        a_scale (Optional[tensor]): Scale factor for operand A.
        a_format (str): Format of the operand A. Available formats: `e2m1`, `e4m3`, `e5m2`.
        b (tensor): The operand B to be multiplied.
        b_scale (Optional[tensor]): Scale factor for operand B.
        b_format (str): Format of the operand B. Available formats: `e2m1`, `e4m3`, `e5m2`.
        acc (tensor): Accumulator tensor.
    """
    _verify_wmma(3, a, b, acc)
    a_format = _unwrap_if_constexpr(a_format)
    b_format = _unwrap_if_constexpr(b_format)
    a_scale = _unwrap_if_constexpr(a_scale)
    b_scale = _unwrap_if_constexpr(b_scale)

    if a_format == "e2m1":
        wmma_layout = a.type.layout.parent
        assert isinstance(wmma_layout, AMDWMMALayout) and wmma_layout.instr_shape in [[16, 16, 64], [32, 16, 64]], \
            "e2m1 format expects instr_shape to be [16, 16, 64] or [32, 16, 64]"
    if b_format == "e2m1":
        wmma_layout = b.type.layout.parent
        assert isinstance(wmma_layout, AMDWMMALayout) and wmma_layout.instr_shape in [[16, 16, 64], [32, 16, 64]], \
            "e2m1 format expects instr_shape to be [16, 16, 64] or [32, 16, 64]"
    acc_shapes = [[16, 16, 128]]
    if a_format == "e2m1" and b_format == "e2m1":
        acc_shapes.append([32, 16, 128])

    acc_layout = acc.type.layout
    assert isinstance(acc_layout, AMDWMMALayout) and acc_layout.instr_shape in acc_shapes, \
        f"accumulator tensor's layout must be one of {acc_shapes}"

    assert a_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported lhs_format: {a_format}"
    assert b_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported rhs_format: {b_format}"

    scale_dtype_to_format = {float8e4nv: "e4m3"}

    # E8M0 scale has various representation in frontend.
    scale_dtype_to_format.update({x: "e8m0" for x in (int8, uint8)})

    if isinstance(a_scale, tensor) and isinstance(b_scale, tensor):
        assert a_scale.dtype in scale_dtype_to_format, f"Unsupported a_scale dtype: {a_scale.dtype}"
        assert b_scale.dtype in scale_dtype_to_format, f"Unsupported b_scale dtype: {b_scale.dtype}"

        a_scale_format = scale_dtype_to_format[a_scale.dtype]
        b_scale_format = scale_dtype_to_format[b_scale.dtype]

        assert (a_format, b_format, a_scale_format, b_scale_format) in _valid_dtype_combinations, \
            f"Unsupported dtype combination: {a_format}, {b_format}, {a_scale_format}, {b_scale_format}."

    return _mma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, get_wmma_scale_layout, _semantic)


@builtin
def scaled_upcast(src, scale, elem_type, axis=None, _semantic=None):
    """
    Upcast an fp4 or fp8 tensor and fold raw E8M0 scale payload into the
    GFX1250 scaled-upcast op.

    The scale tensor must use raw E8M0 payload in `int8` or `uint8`, and must
    already have the expanded output shape and scaled-upcast result layout.
    For fp4 inputs, that is the canonical unpacked layout implied by `src`
    and `axis`. `elem_type` must be `fp16` or `bf16`. GFX1250 keeps those
    bytes in the native `cvt.scale.pk8` payload form.
    """
    axis = _unwrap_if_constexpr(axis)
    elem_type = _unwrap_if_constexpr(elem_type)
    assert scale.dtype in (int8, uint8), \
        f"Expected scale to use raw E8M0 payload in int8/uint8 but got {scale.dtype}"
    return _scaled_upcast(src, scale, elem_type, axis, _semantic)


def _get_wmma_scale_layout_impl(*args, **kwargs):
    return _get_wmma_scale_layout(*args, **kwargs)


_get_wmma_scale_layout_impl.__triton_builtin__ = True


@constexpr_function
def get_wmma_scale_layout(dot_operand_layout, shape, scale_factor=32):
    """ Get the scale layout for WMMA scaled operands.

    Args:
        dot_operand_layout (DotOperandLayout): The dot operand layout.
        shape (List[int]): The shape of the scale tensor.
        scale_factor (int): The scale factor, i.e. the number of elements of operand sharing a single scale.

    Return:
        layout (DistributedLinearLayout): The scale layout.
    """
    assert scale_factor in (16, 32), "Only support 16 or 32 scale factor"
    op_idx = dot_operand_layout.operand_index
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDWMMALayout), "Expected parent to be an instance of AMDWMMALayout"
    mdim = parent.instr_shape[0]
    ndim = parent.instr_shape[1]
    transposed = parent.transposed
    reg_bases = parent.reg_bases
    warp_bases = parent.warp_bases
    cga_bases = parent.cga_layout
    return _get_wmma_scale_layout_impl(op_idx, shape, mdim, ndim, transposed, scale_factor, reg_bases, warp_bases,
                                       cga_bases)
<|endoftext|>from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._layouts import PaddedSharedLayout, SwizzledSharedLayout
from triton.experimental.gluon.language.amd.gfx1250 import PartitionedSharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

if TYPE_CHECKING:
    from triton._C import ir
    from triton.experimental.gluon.language._core import shared_memory_descriptor

__all__ = [
    "update_tensor_descriptor", "async_load", "async_wait", "make_tensor_descriptor", "tensor_descriptor",
    "tensor_descriptor_type", "prefetch", "async_scatter"
]


@dataclass(eq=True)
class tensor_descriptor_type(ttgl.base_type):
    """The type for a tensor descriptor."""

    block_type: ttgl.block_type
    shape_type: ttgl.tuple_type
    strides_type: ttgl.tuple_type
    layout: PaddedSharedLayout | SwizzledSharedLayout | PartitionedSharedLayout

    def __str__(self) -> str:
        return f"tensor_descriptor<{self.block_type}, {self.layout}>"

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[tensor_descriptor, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        value = tensor_descriptor(handle, shape, strides, self)
        return value, cursor

    def _to_ir(self, builder: ir.builder) -> ir.type:
        is_signed = self.block_type.element_ty.is_int_signed()
        return builder.get_tensor_descriptor_layout_type(
            self.block_type.to_ir(builder),
            is_signed,
            self.layout._to_ir(builder),
        )

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self._to_ir(builder))
        self.shape_type._flatten_ir_types(builder, out)
        self.strides_type._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        return f"TD{self.block_type.mangle()}_{self.shape_type.mangle()}_{self.strides_type.mangle()}_{self.layout.mangle()}TD"


@dataclass
class tensor_descriptor(ttgl.base_value):
    """A descriptor representing a tensor in global memory."""

    handle: ir.value
    shape: ttgl.tuple
    strides: ttgl.tuple
    type: tensor_descriptor_type

    def _set_name(self, builder: ir.builder, name: str) -> None:
        self.handle.set_loc(builder.create_name_loc(name, self.handle.get_loc()))
        self.shape._set_name(builder, name + ".shape")
        self.strides._set_name(builder, name + ".stride")

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)
        self.shape._flatten_ir(handles)
        self.strides._flatten_ir(handles)

    @property
    def block_type(self):
        return self.type.block_type

    @property
    def block_shape(self):
        return self.type.block_type.shape

    @property
    def dtype(self):
        return self.type.block_type.element_ty

    @property
    def layout(self):
        return self.type.layout


@builtin
def make_tensor_descriptor(base: ttgl.tensor, shape: List[ttgl.constexpr | ttgl.tensor],
                           strides: List[ttgl.constexpr | ttgl.tensor], block_shape: List[ttgl.constexpr],
                           layout: PaddedSharedLayout | SwizzledSharedLayout | PartitionedSharedLayout,
                           _semantic=None) -> tensor_descriptor:
    """Make a tensor descriptor object.

    Args:
        base (tensor): base pointer of the tensor in global memory.
        shape (List[int]): shape of the tensor.
        strides (List[int]): strides of the tensor.
        block_shape (List[int]): block shape of the tensor.
        layout (PaddedSharedLayout | SwizzledSharedLayout | PartitionedSharedLayout): the layout of the tensor in shared memory.

    Returns:
        tensor_descriptor: the created tensor descriptor object
    """
    ndim = len(shape)
    assert 1 <= ndim <= 5, f"Expected 1 <= ndim <= 5 but got {ndim} dimensions"
    assert len(strides) == ndim, f"Expected {ndim} strides but got {len(strides)}"
    block_shape = _unwrap_if_constexpr(block_shape)
    assert len(block_shape) == ndim, f"Expected block_shape to have {ndim} dimensions but got {len(block_shape)}"
    assert isinstance(base.dtype, ttgl.pointer_type), "Expected base to be a pointer"

    layout = _unwrap_if_constexpr(layout)
    assert isinstance(layout, (PaddedSharedLayout, SwizzledSharedLayout, PartitionedSharedLayout)), \
        "Expected layout to be a PaddedSharedLayout, SwizzledSharedLayout, or PartitionedSharedLayout"
    if isinstance(layout, SwizzledSharedLayout):
        assert layout.max_phase == 1, "Expected max_phase to be 1 for SwizzledSharedLayout"
    elif isinstance(layout, PartitionedSharedLayout):
        assert isinstance(layout.partition_layout, (PaddedSharedLayout, SwizzledSharedLayout)), \
            "PartitionedSharedLayout partition_layout must be PaddedSharedLayout or SwizzledSharedLayout"
        if isinstance(layout.partition_layout, SwizzledSharedLayout):
            assert layout.partition_layout.max_phase == 1, \
                "Expected max_phase to be 1 for SwizzledSharedLayout in PartitionedSharedLayout"

    base_handle = base.handle
    shape_handles = _semantic._convert_to_ir_values(shape, require_i64=False)  # i32 shape
    stride_handles = _semantic._convert_to_ir_values(strides, require_i64=True)  # i64 stride

    shape = ttgl.tuple(shape)
    strides = ttgl.tuple(strides)
    block_type = ttgl.block_type(base.type.element_ty, block_shape)
    type = tensor_descriptor_type(block_type, shape.type, strides.type, layout)

    padding = _semantic._str_to_padding_option("zero")
    handle = _semantic.builder.create_make_tensor_descriptor(type._to_ir(_semantic.builder), base_handle, shape_handles,
                                                             stride_handles, padding)

    return tensor_descriptor(handle, shape, strides, type)


def _handle_i32_pred(pred, _semantic):
    pred = _unwrap_if_constexpr(pred)
    if isinstance(pred, bool):
        pred = int(pred)
    pred = _semantic.to_tensor(pred)
    if pred.type.is_int1():
        pred = _semantic.cast(pred, ttgl.int32)
    assert pred.type.is_int32(), f"Expected pred to be an int32 or int1 value, but got {pred.type}"
    return pred


@builtin
def update_tensor_descriptor(desc: tensor_descriptor, add_offsets: List[ttgl.constexpr | ttgl.tensor] = None,
                             set_bounds: List[ttgl.constexpr | ttgl.tensor] = None, pred=None,
                             clamp_bounds: bool = False, _semantic=None) -> tensor_descriptor:
    """Update selected fields of a TDM descriptor; return a new descriptor SSA value.

    Each parameter is independently optional; only the fields the caller
    names are written.  Everything else is inherited from the input
    descriptor.

    NOTE: Unlike the standard tensor-descriptor mental model, `add_offsets`
    here moves the tile position only.  It does NOT update the descriptor's
    bounds.  If the loop crosses an OOB boundary, you must also pass
    `set_bounds` explicitly, or pass `clamp_bounds=True` to derive the OOB
    extent from `add_offsets`.

    Args:
        desc (tensor_descriptor): the input descriptor.
        add_offsets (List[int], optional): per-dim deltas in element units
            that move the tile position.  Does not touch the bounds.
        set_bounds (List[int], optional): per-dim absolute rewrite of the
            descriptor's bounds.  Use to install OOB extent at a peel
            epilogue.
        pred (int, optional): set the descriptor's predicate.
        clamp_bounds (bool, optional): if True, also shrink the descriptor's
            bounds by `add_offsets` (tensor_dim[i] = max(0, tensor_dim[i] -
            add_offsets[i])) to derive the advanced tile's OOB extent.
            Requires `add_offsets`; mutually exclusive with `set_bounds`.

    Returns:
        tensor_descriptor: a new descriptor SSA value with the requested
        fields rewritten.

    Raises:
        ValueError: if no parameter is provided (no-op updates are forbidden).

    Example:
        # K-loop interior: bump tile position only
        d = tdm.update_tensor_descriptor(d, add_offsets=[0, BLOCK_K])

        # Prologue: position at first tile + set the predicate
        d = tdm.update_tensor_descriptor(d, add_offsets=[pid_m * BLOCK_M, 0],
                                         pred=do_load)

        # Peel epilogue: install real OOB extent for the partial last tile
        d = tdm.update_tensor_descriptor(d, set_bounds=[M - pid_m * BLOCK_M, K - k_main])
    """
    if add_offsets is None and set_bounds is None and pred is None:
        raise ValueError("tdm.update_tensor_descriptor requires at least one of add_offsets, "
                         "set_bounds, pred")

    clamp_bounds = _unwrap_if_constexpr(clamp_bounds)
    if clamp_bounds:
        if add_offsets is None:
            raise ValueError("tdm.update_tensor_descriptor: clamp_bounds requires add_offsets")
        if set_bounds is not None:
            raise ValueError("tdm.update_tensor_descriptor: clamp_bounds and set_bounds are mutually exclusive")

    rank = len(desc.block_shape)

    add_offset_handles = []
    if add_offsets is not None:
        if len(add_offsets) != rank:
            raise ValueError(f"add_offsets must have length {rank} (descriptor rank), got {len(add_offsets)}")
        add_offset_handles = _semantic._convert_to_ir_values(add_offsets, require_i64=False)

    set_bounds_handles = []
    if set_bounds is not None:
        if len(set_bounds) != rank:
            raise ValueError(f"set_bounds must have length {rank} (descriptor rank), got {len(set_bounds)}")
        set_bounds_handles = _semantic._convert_to_ir_values(set_bounds, require_i64=False)

    pred_handle = ttgl.ir.value()
    if pred is not None:
        pred_handle = _handle_i32_pred(pred, _semantic).handle

    new_handle = _semantic.builder.create_update_tensor_descriptor(desc.handle, add_offset_handles, set_bounds_handles,
                                                                   pred_handle, bool(clamp_bounds))
    # Rebuild shape/strides tuples so the returned descriptor's tuple objects
    # don't alias the original's (the frontend's SSA tracking assumes distinct
    # tuples per descriptor value).
    new_shape = ttgl.tuple(list(desc.shape))
    new_strides = ttgl.tuple(list(desc.strides))
    return tensor_descriptor(new_handle, new_shape, new_strides, desc.type)


@builtin
def async_load(src: tensor_descriptor, offsets: List[ttgl.constexpr | ttgl.tensor] = None,
               dest: shared_memory_descriptor = None, pred=None, mbarrier: shared_memory_descriptor = None,
               warp_used_hint=None, cache_modifier="", _semantic=None) -> None:
    """Load a block of tensor specified in tensor descriptor from global memory to shared memory asynchronously.

    This op expects a prior ``update_tensor_descriptor`` to position the
    descriptor for the load offsets and bounds.  Doing this explicitly lets the
    developer control when and where the descriptor update happens, which has
    performance implications for downstream code generation.  For convenience,
    ``offsets`` (and optionally ``pred``) may be passed here, in which case an
    ``update_tensor_descriptor`` is emitted right before the load.

    Args:
        src (tensor_descriptor): the source tensor descriptor.
        offsets (List[int], optional): if given, the offsets from the base
            pointer used to position the descriptor before the load.
        dest (shared_memory_descriptor): the shared memory destination to store the loaded data.
        pred (bool, optional): if given, predicate to enable or disable the load.
        mbarrier (shared_memory_descriptor, optional): The barrier object to signal "arrive" on.
        warp_used_hint (int, optional): Bitmask selecting which warps issue
            the TDM copy (bit ``n`` => warp ``n``); cleared warps become HW
            no-ops.  Doesn't affect the data in ``dest``, only the work split.
            The number of active warps must be a power of two, and the active
            warps must follow a regular bit pattern for efficient lowering.
            Examples: ``0b00001111`` (warps 0..3), ``0b11110000`` (warps
            4..7), ``0b01010101`` (warps 0,2,4,6).  Omit / ``None`` = all
            warps participate; explicit ``0`` and other invalid hints are
            rejected by the verifier.
        cache_modifier (str, optional): Cache behavior.
    """
    assert dest is not None, "async_load requires a dest shared_memory_descriptor"
    if offsets:
        eff_pred = True if pred is None else pred
        src = update_tensor_descriptor(src, add_offsets=offsets, pred=eff_pred, clamp_bounds=True, _semantic=_semantic)
    elif pred is not None:
        src = update_tensor_descriptor(src, pred=pred, _semantic=_semantic)

    mbarrier = _unwrap_if_constexpr(mbarrier)
    mbarrier_handle = mbarrier.handle if mbarrier is not None else ttgl.ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)

    warp_used_hint = _unwrap_if_constexpr(warp_used_hint)
    if warp_used_hint is not None:
        warp_used_hint = int(warp_used_hint)

    _semantic.builder.create_async_tdm_copy_global_to_local(src.handle, dest.handle, mbarrier_handle, cache_modifier,
                                                            warp_used_hint)


@builtin
def async_store(dest: tensor_descriptor, offsets: List[ttgl.constexpr | ttgl.tensor] = None,
                src: shared_memory_descriptor = None, mbarrier: shared_memory_descriptor = None, cache_modifier="",
                _semantic=None) -> None:
    """Store a block of tensor specified in tensor descriptor from shared memory to global memory asynchronously.

    See :func:`async_load` for how the descriptor is positioned (a prior
    ``update_tensor_descriptor`` or the ``offsets`` convenience).  Stores take no
    ``pred``.

    Args:
        dest (tensor_descriptor): the destination tensor descriptor.
        offsets (List[int], optional): if given, the offsets from the base
            pointer used to position the descriptor before the store.
        src (shared_memory_descriptor): the shared memory source to load the data.
        mbarrier (shared_memory_descriptor, optional): The barrier object to signal "arrive" on.
    """
    assert src is not None, "async_store requires a src shared_memory_descriptor"
    if offsets:
        dest = update_tensor_descriptor(dest, add_offsets=offsets, clamp_bounds=True, _semantic=_semantic)

    mbarrier = _unwrap_if_constexpr(mbarrier)
    mbarrier_handle = mbarrier.handle if mbarrier is not None else ttgl.ir.value()
    cache_modifier = _semantic._str_to_store_cache_modifier(cache_modifier)
    _semantic.builder.create_async_tdm_copy_local_to_global(dest.handle, src.handle, mbarrier_handle, cache_modifier)


@builtin
def async_wait(num_outstanding=0, _semantic=None) -> None:
    """Wait for the outstanding asynchronous tensor operations to complete.

    Args:
        num_outstanding (int): number of outstanding async tensor operations to wait for.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_tdm_wait(num_outstanding)


@builtin
def async_scatter(desc: tensor_descriptor, dst_row_indices: ttgl.tensor, src: shared_memory_descriptor,
                  mbarrier: shared_memory_descriptor = None, _semantic=None) -> None:
    """Scatter data from shared memory to non-contiguous rows in global memory asynchronously.

    This operation uses TDM scatter mode to write data to non-contiguous rows in global memory.
    Unlike async_store which writes to contiguous rows, scatter allows writing to arbitrary
    rows specified by the dst_row_indices tensor.

    The dtype of dst_row_indices determines the index size:
    - int16: up to 16 rows can be scattered per TDM instruction
    - int32: up to 8 rows can be scattered per TDM instruction
    If more rows are needed, multiple TDM instructions will be automatically issued.

    The column offset is carried by the descriptor: position it beforehand with
    ``update_tensor_descriptor(desc, add_offsets=[0, col])``.

    Args:
        desc (tensor_descriptor): the destination tensor descriptor. Must be 2D and positioned
                                  (column via update_tensor_descriptor).
        dst_row_indices (tensor): 1D tensor of row indices (int16 or int32) in the destination tensor.
        src (shared_memory_descriptor): the shared memory source containing data to scatter. Must be 2D.
        mbarrier (shared_memory_descriptor, optional): The barrier object to signal "arrive" on.
    """
    ndim = len(desc.block_shape)
    assert ndim == 2, f"TDM scatter only supports 2D tensors, got {ndim}D"

    src_ndim = len(src.shape)
    assert src_ndim == 2, f"TDM scatter src must be 2D, got {src_ndim}D"

    mbarrier = _unwrap_if_constexpr(mbarrier)
    mbarrier_handle = mbarrier.handle if mbarrier is not None else ttgl.ir.value()

    _semantic.builder.create_async_tdm_scatter(desc.handle, dst_row_indices.handle, src.handle, mbarrier_handle)


@builtin
def async_gather(desc: tensor_descriptor, src_row_indices: ttgl.tensor, dst: shared_memory_descriptor,
                 mbarrier: shared_memory_descriptor = None, _semantic=None) -> None:
    """Gather data from non-contiguous rows in global memory to shared memory asynchronously.

    This operation uses TDM gather mode to read data from non-contiguous rows in global memory.
    Unlike async_load which reads from contiguous rows, gather allows reading from arbitrary
    rows specified by the src_row_indices tensor.

    The dtype of src_row_indices determines the index size:
    - int16: up to 16 rows can be gathered per TDM instruction
    - int32: up to 8 rows can be gathered per TDM instruction
    If more rows are needed, multiple TDM instructions will be automatically issued.

    The column offset and predicate are carried by the descriptor: position it
    beforehand with ``update_tensor_descriptor(desc, add_offsets=[0, col], pred=p)``.

    Args:
        desc (tensor_descriptor): the source tensor descriptor. Must be 2D and positioned
                                  (column / pred via update_tensor_descriptor).
        src_row_indices (tensor): 1D tensor of row indices (int16 or int32) in the source tensor.
        dst (shared_memory_descriptor): the shared memory destination to store gathered data. Must be 2D.
        mbarrier (shared_memory_descriptor, optional): The barrier object to signal "arrive" on.
    """
    ndim = len(desc.block_shape)
    assert ndim == 2, f"TDM gather only supports 2D tensors, got {ndim}D"

    dst_ndim = len(dst.shape)
    assert dst_ndim == 2, f"TDM gather dst must be 2D, got {dst_ndim}D"

    mbarrier = _unwrap_if_constexpr(mbarrier)
    mbarrier_handle = mbarrier.handle if mbarrier is not None else ttgl.ir.value()

    _semantic.builder.create_async_tdm_gather(desc.handle, src_row_indices.handle, dst.handle, mbarrier_handle)


@builtin
def prefetch(src: tensor_descriptor, offsets: List[ttgl.constexpr | ttgl.tensor], pred: bool = True,
             speculative: bool = False, _semantic=None) -> None:
    """Prefetches a block of tensor specified in tensor descriptor from global memory into L2. Speculative prefetches can generate more
    efficient assembly because they do not require out of bounds checks. However, they are dropped by the hardware if their virtual address translation is not cached.
    So speculative should only be set if previous iterations have accessed the same virtual page (e.g. column major)
    Args:
        src (tensor_descriptor): the source tensor descriptor.
        offsets (List[int]): the offsets from the base pointer in the tensor descriptor.
        pred (bool, optional): Predicate to enable or disable the prefetch. Defaults to True.
        speculative (bool, optional): Whether the prefetch is speculative. Defaults to False.
    """
    offset_handles = _semantic._convert_to_ir_values(offsets, require_i64=False)
    pred = _semantic.to_tensor(pred)
    pred_handle = pred.handle
    speculative = _unwrap_if_constexpr(speculative)
    _semantic.builder.create_tdm_prefetch(src.handle, offset_handles, pred_handle, speculative, False)


@builtin
def _test_prefetch_with_offsets(src: tensor_descriptor, offsets: List[ttgl.constexpr | ttgl.tensor], pred: bool = True,
                                speculative: bool = False, _semantic=None) -> ttgl.tensor:
    """Test-only prefetch variant that returns offsets for validation."""
    offset_handles = _semantic._convert_to_ir_values(offsets, require_i64=False)
    pred = _semantic.to_tensor(pred)
    pred_handle = pred.handle
    speculative = _unwrap_if_constexpr(speculative)
    handle = _semantic.builder.create_tdm_prefetch(src.handle, offset_handles, pred_handle, speculative, True)
    shape = _semantic.builder.get_shape_from_tensor(handle)
    layout = _semantic.builder.get_gluon_layout_from_tensor(handle)
    ret_ty = ttgl.distributed_type(ttgl.int64, shape, layout)
    tensor = ttgl.tensor(handle, ret_ty)
    return tensor
<|endoftext|>import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._layouts import SwizzledSharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

__all__ = ["MBarrierLayout", "init", "wait", "arrive"]


class MBarrierLayout(SwizzledSharedLayout):
    """
    Layout for mbarrier synchronization.

    Args:
        cga_layout (List[List[int]]): CGA layout bases. Defaults to [].
    """

    def __init__(self, cga_layout=None):
        super().__init__(vec=1, per_phase=1, max_phase=1, order=[0], cga_layout=cga_layout or [])


@builtin
def init(mbarrier, count, _semantic=None):
    """
    Initialize an mbarrier with a specified count. An mbarrier consists of an init count, a pending count and a phase.
    At initialization, the init count and pending count are initialized with the given 'count' and the phase is initialized to 0.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to initialize.
        count (int): The initial count for the barrier. Must be a positive integer.
    """
    count = _unwrap_if_constexpr(count)
    _semantic.builder.create_lds_barrier_init(mbarrier.handle, count)


@builtin
def wait(mbarrier, phase, _semantic=None):
    """
    Wait until the mbarrier's phase differs from the provided phase value.
    This means that the given 'phase' has completed.

    Args:
        mbarrier (shared_memory_descriptor): The barrier object to wait on.
        phase (int): The phase value to compare against. The wait completes when
        the barrier's phase becomes different from this value.
    """
    phase = _semantic.to_tensor(phase)

    _semantic.builder.create_lds_barrier_wait(mbarrier.handle, phase.handle)


@builtin
def arrive(mbarrier, *, count=1, _semantic=None):
    """
    Arrive at an mbarrier with a specified count. The operation requires a `count` attribute
    of at least 1, and decreases the pending arrival count of the mbarrier by the specific count.
    If the pending count reaches zero, the phase changes (is decremented in a wraparound manner) and the
    pending count is reloaded with the init count value. Returns the mbarrier's phase parity (0 for even, 1 for odd) prior to the "arrive" operation.

    Args:
        mbarrier (shared_memory_descriptor): Barrier to be signalled.
        count (int): Count to arrive with. Defaults to 1.

    Returns:
        prior phase (int): phase of mbarrier, prior to "arrive" operation.
    """
    count = _unwrap_if_constexpr(count)
    handle = _semantic.builder.create_lds_barrier_arrive(mbarrier.handle, count)
    return ttgl.tensor(handle, ttgl.int32)
<|endoftext|>from __future__ import annotations
from typing import TYPE_CHECKING

from triton import knobs
from triton.experimental.gluon.language import _core as ttgl
from triton._C.libtriton import ir
from ..._core import builtin, int8, uint8, uint16, bfloat16, _unwrap_if_constexpr
from .._ops import _scaled_upcast

if TYPE_CHECKING:
    from ..._semantic import GluonSemantic

__all__ = [
    "buffer_atomic_add", "buffer_atomic_and", "buffer_atomic_min", "buffer_atomic_max", "buffer_atomic_or",
    "buffer_atomic_xor", "buffer_atomic_xor", "buffer_load", "buffer_store", "mfma", "scaled_upcast"
]

_atomic_op_str_to_op = {
    "smax": ir.ATOMIC_OP.MAX, "smin": ir.ATOMIC_OP.MIN, "umax": ir.ATOMIC_OP.UMAX, "umin": ir.ATOMIC_OP.UMIN, "fadd":
    ir.ATOMIC_OP.FADD, "iadd": ir.ATOMIC_OP.ADD, "and": ir.ATOMIC_OP.AND, "or": ir.ATOMIC_OP.OR, "xor":
    ir.ATOMIC_OP.XOR, "xchg": ir.ATOMIC_OP.XCHG
}


def _verify_buffer_ops(ptr, offsets, mask=None, other=None):
    assert ptr.type.is_ptr(), "ptr must be a scalar pointer type"

    assert isinstance(offsets.type, ttgl.distributed_type), "expected offsets type to be a distributed_type"
    assert offsets.dtype.is_int32() or offsets.dtype.is_uint32(), "offsets element type must be int32 or uint32"

    mask = _unwrap_if_constexpr(mask)
    other = _unwrap_if_constexpr(other)
    if other is not None:
        assert mask is not None, "when other is not None, mask should not be None"


def _verify_element_type_and_dispatch_op(op, elem_type, arch):
    supported_types = [
        ttgl.float16, ttgl.float32, ttgl.bfloat16, ttgl.float64, ttgl.int32, ttgl.int64, ttgl.uint32, ttgl.uint64
    ]
    assert elem_type in supported_types, f"{elem_type} is not supported in buffer atomic on {arch}."

    if op in ['and', 'or', 'xor', 'xchg']:
        assert elem_type in [ttgl.int32, ttgl.int64], f"{op} with {elem_type} is not supported on CDNA3 or CDNA4"
        return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]

    if op in ['max', 'min']:
        if elem_type in [ttgl.int32, ttgl.int64, ttgl.float64]:
            op = 's' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type in [ttgl.uint32, ttgl.uint64]:
            op = 'u' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        else:
            raise ValueError(f"{op} with {elem_type} is not supported on CDNA3 and CDNA4")

    if op == 'add':
        if elem_type in [ttgl.uint32, ttgl.uint64]:
            op = 'i' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type in [ttgl.float16, ttgl.float32, ttgl.float64]:
            op = 'f' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        elif elem_type is ttgl.bfloat16:
            assert arch == "cdna4", "Buffer atomic fadd with bf16 is only supported on CDNA4 for now."
            op = 'f' + op
            return _atomic_op_str_to_op[_unwrap_if_constexpr(op)]
        else:
            raise ValueError(f"{op} with {elem_type} is not supported on CDNA3 and CDNA4")

    raise ValueError(f"Unknown {op} on CDNA3 or CDNA4")


def _buffer_atomic_rmw_impl(op, ptr, offsets, value, arch, mask, sem, scope, _semantic):
    _verify_buffer_ops(ptr, offsets, mask)

    op = _verify_element_type_and_dispatch_op(op, ptr.type.scalar.element_ty, arch)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
        mask = _semantic.cast(mask, ttgl.int1)
        _, mask = _semantic.broadcast_impl_value(offsets, mask)
    mask = mask.handle if mask is not None else ir.value()

    value = _unwrap_if_constexpr(value)
    value = _semantic.to_tensor(value)
    _, value = _semantic.broadcast_impl_value(offsets, value)

    sem = _semantic._str_to_sem(sem)
    scope = _semantic._str_to_scope(scope)
    return _semantic.tensor(
        _semantic.builder.create_buffer_atomic_rmw(op, ptr.handle, offsets.handle, value.handle, sem, scope, mask),
        value.type)


@builtin
def buffer_load(ptr, offsets, mask=None, other=None, cache=None, _semantic=None):
    """
    AMD buffer load from global memory via a scalar base pointer and a tensor of
    offsets instead of a tensor of pointers. This operation will load data
    directly into registers.

    Args:
        ptr (pointer to scalar): Global memory scalar base pointer to load from.
        offsets (tensor): Offsets tensor for the load operation.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor or scalar, optional): Tensor or scalar providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    _verify_buffer_ops(ptr, offsets, mask, other)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)

    other = _unwrap_if_constexpr(other)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, ptr.dtype.element_ty)
        offsets, other = _semantic.broadcast_impl_value(offsets, other)

    other = other.handle if other is not None else ir.value()
    mask = mask.handle if mask is not None else ir.value()
    cache = _unwrap_if_constexpr(cache)
    cache_modifier = _semantic._str_to_load_cache_modifier(cache) if cache is not None else ir.CACHE_MODIFIER.NONE

    ret_ty = offsets.type.with_element_ty(ptr.type.scalar.element_ty)
    builder = _semantic.builder
    handle = builder.create_buffer_load(ret_ty.to_ir(builder), ptr.handle, offsets.handle, mask, other, cache_modifier)
    return ttgl.tensor(handle, ret_ty)


@builtin
def buffer_store(stored_value, ptr, offsets, mask=None, cache=None, _semantic: GluonSemantic = None):
    """
    AMD buffer store a tensor directly to global memory via a scalar base pointer and a tensor of
    offsets instead of a tensor of pointers.

    Args:
        stored_value (tensor to be stored): The tensor to be stored to global memory.
        ptr (pointer to scalar): Global memory scalar base pointer to store to.
        offsets (tensor): Offsets tensor for the store operation.
        mask (tensor, optional): Mask tensor for predicated store. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    _verify_buffer_ops(ptr, offsets, mask)

    offsets_shape = offsets.shape
    mask = _unwrap_if_constexpr(mask)
    if mask is None:
        offsets, stored_value = _semantic.broadcast_tensors(offsets, stored_value)
    else:
        offsets, stored_value, mask = _semantic.broadcast_tensors(offsets, stored_value, mask)
    if offsets_shape != offsets.shape:
        raise ValueError(f"Expected `offsets` argument to have shape {offsets.shape} but got {offsets_shape}")

    mask = mask.handle if mask is not None else ir.value()
    cache = _unwrap_if_constexpr(cache)
    cache_modifier = _semantic._str_to_store_cache_modifier(cache) if cache is not None else ir.CACHE_MODIFIER.NONE

    _semantic.builder.create_buffer_store(stored_value.handle, ptr.handle, offsets.handle, mask, cache_modifier)


@builtin
def mfma(a, b, acc, _semantic: GluonSemantic = None):
    """
    Computes matrix-multiplication of a * b + acc using AMD native matrix core units.
    Args:
        a (tensor): The first operand of mfma.
        b (tensor): The second operand of mfma.
        acc (tensor): The accumulator tensor.
    """
    assert acc is not None, "acc is required"
    ret_type = acc.type
    acc = ttgl._unwrap_if_constexpr(acc)

    handle = _semantic.dot(a, b, acc, input_precision=knobs.language.fp32_default, max_num_imprecise_acc=None,
                           out_dtype=acc.dtype).handle
    return ttgl.tensor(handle, ret_type)


def _convert_e8m0_scale_to_bf16(scale, _semantic=None):
    # Mirror scaleTo16() for BF16 compute: reinterpret raw E8M0 bytes as the
    # shifted BF16 payload expected by the non-gfx1250 scaled-upcast path.
    if scale.dtype == int8:
        scale = _semantic.bitcast(scale, uint8)
    scale = _semantic.cast(scale, uint16)
    shift = _semantic.cast(_semantic.to_tensor(7), uint16)
    scale = _semantic.shl(scale, shift)
    return _semantic.bitcast(scale, bfloat16)


@builtin
def scaled_upcast(src, scale, elem_type, axis=None, _semantic=None):
    """
    Upcast an fp4 or fp8 tensor and fold raw E8M0 scale payload into the
    CDNA3 scaled-upcast op.

    CDNA3 lowers this through the software-emulated scaled-upcast path; it
    does not use native hardware scaled-upcast instructions.

    The scale tensor must use raw E8M0 payload in `int8` or `uint8`, and must
    already have the expanded output shape and scaled-upcast result layout.
    For fp4 inputs, that is the canonical unpacked layout implied by `src`
    and `axis`. `elem_type` must be `fp16` or `bf16`. CDNA3 converts those
    bytes to the internal `bf16` scale form expected by the AMD op.
    """
    axis = _unwrap_if_constexpr(axis)
    elem_type = _unwrap_if_constexpr(elem_type)
    assert scale.dtype in (int8, uint8), \
        f"Expected scale to use raw E8M0 payload in int8/uint8 but got {scale.dtype}"
    scale = _convert_e8m0_scale_to_bf16(scale, _semantic)
    return _scaled_upcast(src, scale, elem_type, axis, _semantic)


"""
AMD Buffer Atomic RMW operations.
The supported operatios are max, min, add, and, or, xor, xchg.
Similar to normal atomic ops: it loads data at ptr plus offsets, do `op` with `value`, and store result to `ptr` plus `offsets` with
the specified memory semantics and scope.

Buffer atomics access global memory via a scalar base pointer and a tensor of offsets instead of a tensor of pointers.
Similar to other buffer ops, the `mask` is a boolean vector that determines if a given element should be processed with
the atomic RMW op. Elements with `mask[i] == 0` are dropped (i.e., the atomic is not executed).

Buffer Atomic RMW ops return the pre-op value in the global memory.

Args:
    ptr (pointer to scalar): Global memory scalar base pointer to load from.
    offsets (tensor): Offsets tensor for the load operation.
    value (tensor): Another operand of `op`.
    mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
    sem (str, optional): Memory Semantic Descriptor. Default is None which means acq_rel memory semantic.
    scope (str, optional): Memory Sync Scope for atomic accesses. Default is None and it will be mapped to `gpu`, which is called `agent` for AMDGPU. Please ref https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942 for details.
"""


@builtin
def buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):
    return _buffer_atomic_rmw_impl('max', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('min', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('add', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('and', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('or', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xor', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xchg(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xchg', ptr, offsets, value, "cdna3", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)
<|endoftext|>from ..._core import builtin
from .._ops import _wmma

__all__ = ["wmma"]


@builtin
def wmma(a, b, acc, _semantic=None):
    """
    Computes matrix-multiplication of a * b + acc using AMD WMMA instruction.

    Args:
        a (tensor): The operand a to be multiplied.
        b (tensor): The operand b to be multiplied.
        acc (tensor): The accumulator tensor.
    """
    return _wmma(1, a, b, acc, _semantic)
<|endoftext|>from ..._core import ir, builtin, _unwrap_if_constexpr
from ..._semantic import _check
from ..._layouts import DistributedLayout
from ..cdna3 import _verify_buffer_ops

__all__ = [
    "global_load_to_shared",
    "buffer_load_to_shared",
    "commit_group",
    "wait_group",
    "load_shared_relaxed",
]


@builtin
def global_load_to_shared(dest, ptr, mask=None, other=None, cache_modifier="", _semantic=None):
    """
    AMD global load to shared operation. This operation loads data directly
    from global memory to shared memory without going through registers. It
    happens asynchronously and requires a subsequent `async_wait` to ensure the
    data is available in shared memory. Note that this operation does still
    complete in order with ttgl.loads/stores or buffer_loads/stores on CDNA4,
    so interleaving with them will hurt performance.

    Compared to `buffer_load_to_shared`, it requires a tensor pointer which
    supports 64-bit indexing range for each thread in a block, which gives more
    flexibility, but at the cost of higher register pressure and no hardware
    out-of-bound masking support. Prefer to use `buffer_load_to_shared` when
    possible for better performance.

    The underlying hardware instruction uses separate registers for global
    memory address for each thread but the same register for local memory
    address for the whole warp. Therefore, while using this operation
    the following conditions must be met or lowering to LLVM will fail:

    - For the `ptr` layout, size per thread * bits per element must be 128 or 32.
      To get ideal performance, it is recommended to use 128 bits per element.
    - Writes to `dest` must be coalesced.
    - If `dest` is swizzled, it only can be swizzled within warp boundary.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        ptr (pointer tensor): Tensor of pointers to global memory to load from.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor or scalar, optional): Tensor or scalar providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    _check(ptr.type.is_block(), lambda: "expected ptr to be a tensor")
    _check(isinstance(ptr.type.layout, DistributedLayout), lambda: "expected ptr type layout to be a DistributedLayout")
    _check(
        dest.shape == ptr.shape, lambda:
        f"expected dest shape to match pointer shape but got dest.shape = {dest.shape}, pointer.shape = {ptr.shape}")

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        ptr, mask = _semantic.broadcast_impl_value(ptr, mask)
    other = _unwrap_if_constexpr(other)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, ptr.dtype.element_ty)
        ptr, other = _semantic.broadcast_impl_value(ptr, other)

    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)
    mask_handle = mask.handle if mask is not None else ir.value()
    other_handle = other.handle if other is not None else ir.value()
    _semantic.builder.create_async_copy_global_to_local(dest.handle, ptr.handle, mask_handle, other_handle,
                                                        cache_modifier, ir.EVICTION_POLICY.NORMAL, False)


@builtin
def buffer_load_to_shared(dest, ptr, offsets, mask=None, other=None, cache_modifier="", _semantic=None):
    """
    AMD buffer load to shared operation. Buffer load is similar to global load
    but it accesses global memory via a scalar base pointer and a tensor of
    32-bit offsets instead of a tensor of pointers. This operation loads data
    directly from global memory to shared memory without going through
    registers. It happens asynchronously and requires a subsequent `async_wait`
    to ensure thedata is available in shared memory. Note that this operation
    does still complete in order with ttgl.loads/stores or buffer_loads/stores
    on CDNA4, so interleaving with them will hurt performance.

    Compared to `global_load_to_shared`, it has better performance and also
    supports hardware out-of-bound masking. But it strictly requires a
    32-bit offset instead of a 64-bit tensor pointer.

    The underlying hardware instruction uses separate registers for global
    memory address for each thread but the same register for local memory
    address for the whole warp. Therefore, while using this operation
    the following conditions must be met or lowering to LLVM will fail:

    - For the `offsets` layout, size per thread * bits per element must be 128 or 32.
      To get ideal performance, it is recommended to use 128 bits per element.
    - Writes to `dest` must be coalesced.
    - If `dest` is swizzled, it only can be swizzled within warp boundary.

    Args:
        dest (shared_memory_descriptor): Destination shared memory descriptor.
        ptr (pointer to scalar): Global memory scalar base pointer to load from.
        offsets (tensor): Offsets tensor for the load operation.
        mask (tensor, optional): Mask tensor for predicated loads. Defaults to None.
        other (tensor or scalar, optional): Tensor or scalar providing default values for masked elements. Defaults to None.
        cache_modifier (str): Cache modifier specifier. Defaults to "".
    """
    _check(isinstance(offsets.type.layout, DistributedLayout),
           lambda: "expected offsets type layout to be a DistributedLayout")
    _verify_buffer_ops(ptr, offsets, mask, other)

    mask = _unwrap_if_constexpr(mask)
    if mask is not None:
        offsets, mask = _semantic.broadcast_impl_value(offsets, mask)
    other = _unwrap_if_constexpr(other)
    if other is not None:
        other = _semantic.to_tensor(other)
        other = _semantic.cast(other, ptr.type.scalar.element_ty)
        offsets, other = _semantic.broadcast_impl_value(offsets, other)

    mask = mask.handle if mask is not None else ir.value()
    other = other.handle if other is not None else ir.value()
    stride = ir.value()
    cache_modifier = _semantic._str_to_load_cache_modifier(cache_modifier)

    _semantic.builder.create_buffer_load_to_local(dest.handle, ptr.handle, offsets.handle, mask, other, stride,
                                                  cache_modifier)


@builtin
def commit_group(_semantic=None):
    """
    Commit oustanding async operations.

    This finalizes a set of async copy operations which can be waited upon via `wait_group`.
    """
    _semantic.builder.create_async_commit_group()


@builtin
def wait_group(num_outstanding=0, _semantic=None):
    """
    Wait for outstanding commit groups. It will block until the number of
    outstanding commit groups is less than or equal to `num_outstanding`. Note that uncommited
    async operations will be waited upon even if `num_outstanding` is 0.

    Args:
        num_outstanding (int): The number of outstanding commit groups to wait for. Defaults to 0.
    """
    num_outstanding = _unwrap_if_constexpr(num_outstanding)
    _semantic.builder.create_async_wait_group(num_outstanding)


@builtin
def load_shared_relaxed(smem, layout, _semantic=None):
    """
    Load a tensor from shared memory with extra hints for the underlying
    compiler to avoid emitting unnecessary waits before loading from the target
    shared memory.

    Args:
        smem (shared_memory_descriptor): Shared memory descriptor to load from.
        layout (DistributedLayout): The destination layout of the tensor.

    Returns:
        tensor: A Gluon tensor containing the loaded data.
    """
    SYNCED_VIA_WAIT_ATTR_NAME = "ttg.amdg.syncedViaAsyncWait"

    layout = _unwrap_if_constexpr(layout)
    ret = _semantic.shared_load(smem, layout)
    ret.handle.set_attr(SYNCED_VIA_WAIT_ATTR_NAME, _semantic.builder.get_bool_attr(True))
    return ret
<|endoftext|>from triton.runtime.jit import constexpr_function
from triton._C.libtriton.gluon_ir import (
    get_amd_mfma_scale_layout as _get_mfma_scale_layout,
    compute_amd_efficient_padded_shared_layout as _compute_efficient_padded_shared_layout,
)

from ..._core import builtin, int8, uint8, _unwrap_if_constexpr
from ..._layouts import DotOperandLayout
from .._layouts import AMDMFMALayout
from .._ops import _mma_scaled, _scaled_upcast
from ..cdna3 import _buffer_atomic_rmw_impl, _convert_e8m0_scale_to_bf16
from ..cdna3 import *  # NOQA: F403
from ..cdna3 import __all__ as __cdna3_all
from . import async_copy

__all__ = [
    *__cdna3_all,
    "async_copy",
    "mfma_scaled",
    "scaled_upcast",
    "get_mfma_scale_layout",
    "compute_efficient_padded_shared_layout",
]


@builtin
def mfma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, _semantic=None):
    """
    AMD Scaled MFMA operation.

    ```
    c = a * a_scale @ b * b_scale + acc
    ```

    `a` and `b` use microscaling formats described in
    "OCP Microscaling Formats (MX) Specification":
    https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf.
    Currently supported only on CDNA4 hardware.

    Args:
        a (tensor): The operand A to be multiplied.
        a_scale (Optional[tensor]): Scale factor for operand A.
        a_format (str): Format of the operand A. Available formats: `e2m1`, `e4m3`, `e5m2`.
        b (tensor): The operand B to be multiplied.
        b_scale (Optional[tensor]): Scale factor for operand B.
        b_format (str): Format of the operand B. Available formats: `e2m1`, `e4m3`, `e5m2`.
        acc (tensor): Accumulator tensor.
    """
    layout = acc.type.layout
    assert isinstance(layout, AMDMFMALayout), "Expected layout to be an instance of AMDMFMALayout"
    assert (isinstance(a.type.layout, DotOperandLayout) and a.type.layout.parent== layout), \
            "Expected lhs layout to be a DotOperandLayout with parent matching MFMA layout"
    assert (isinstance(b.type.layout, DotOperandLayout) and b.type.layout.parent == layout), \
            "Expected rhs layout to be a DotOperandLayout with parent matching MFMA layout"

    a_format = _unwrap_if_constexpr(a_format)
    b_format = _unwrap_if_constexpr(b_format)
    a_scale = _unwrap_if_constexpr(a_scale)
    b_scale = _unwrap_if_constexpr(b_scale)
    assert a_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported lhs_format: {a_format}"
    assert b_format in {"e2m1", "e4m3", "e5m2"}, f"Unsupported rhs_format: {b_format}"

    return _mma_scaled(a, a_scale, a_format, b, b_scale, b_format, acc, get_mfma_scale_layout, _semantic)


@builtin
def scaled_upcast(src, scale, elem_type, axis=None, _semantic=None):
    """
    Upcast an fp4 or fp8 tensor and fold raw E8M0 scale payload into the
    CDNA4 scaled-upcast op.

    The scale tensor must use raw E8M0 payload in `int8` or `uint8`, and must
    already have the expanded output shape and scaled-upcast result layout.
    For fp4 inputs, that is the canonical unpacked layout implied by `src`
    and `axis`. `elem_type` must be `fp16` or `bf16`. CDNA4 converts those
    bytes to the internal `bf16` scale form expected by the AMD op.
    """
    axis = _unwrap_if_constexpr(axis)
    elem_type = _unwrap_if_constexpr(elem_type)
    assert scale.dtype in (int8, uint8), \
        f"Expected scale to use raw E8M0 payload in int8/uint8 but got {scale.dtype}"
    scale = _convert_e8m0_scale_to_bf16(scale, _semantic=_semantic)
    return _scaled_upcast(src, scale, elem_type, axis, _semantic)


def _get_mfma_scale_layout_impl(*args, **kwargs):
    return _get_mfma_scale_layout(*args, **kwargs)


_get_mfma_scale_layout_impl.__triton_builtin__ = True


@constexpr_function
def get_mfma_scale_layout(dot_operand_layout, shape, scale_factor=32):
    """ Get the scale layout for MFMA scaled operands.

    Args:
        dot_operand_layout (DotOperandLayout): The dot operand layout.
        shape (List[int]): The shape of the scale tensor.
        scale_factor (int): The scale factor.
    Return:
        layout (DistributedLinearLayout): The scale layout.
    """
    assert scale_factor == 32, "Only scale factor 32 is supported for CDNA4 Scaled MFMA"
    op_idx = dot_operand_layout.operand_index
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDMFMALayout), "Expected parent to be an instance of AMDMFMALayout"
    mdim = parent.instr_shape[0]
    tiles_per_warp = parent.tiles_per_warp
    warps_per_cta = parent.warps_per_cta
    return _get_mfma_scale_layout_impl(op_idx, shape, mdim, tiles_per_warp, warps_per_cta)


def _compute_efficient_padded_shared_layout_impl(*args, **kwargs):
    return _compute_efficient_padded_shared_layout(*args, **kwargs)


_compute_efficient_padded_shared_layout_impl.__triton_builtin__ = True


@constexpr_function
def compute_efficient_padded_shared_layout(dot_operand_layout, shape, dtype, is_k_contig=True):
    """Compute an efficient padded shared layout for the given parameters
    that avoids bank conflicts as much as possible.

    Args:
        dot_operand_layout (DotOperandLayout): The layout for the dot operand
            that will be copied to shared memory with padding. Must have an
            AMDMFMALayout v4 (CDNA4) parent.
        shape (List[int]): Shared memory tile shape for the dot operand —
            ``[BM, BK]`` for operand A or ``[BK, BN]`` for operand B.
        dtype (dtype): Element type of the tensor that will live in this
            shared memory allocation (e.g. ``ttgl.float16``, ``ttgl.float8e4nv``).
            Only types with bitwidth in {4, 8, 16} are supported. For packed
            fp4 (two values per byte), pass ``ttgl.uint8`` — at the LDS level
            4-bit shares the 8-bit padding pattern.
        is_k_contig (bool): K is the contiguous dim in shared memory.
    Return:
        layout (PaddedSharedLayout): or None if the input falls outside the
            supported set. Common reasons for None: ``k_width`` not in
            {4, 8, 16}; element bitwidth not in {4, 8, 16}; MFMA instruction
            shape or kWidth combination not handled by the underlying
            algorithm.
    """
    parent = dot_operand_layout.parent
    assert isinstance(parent, AMDMFMALayout), \
        "Expected dot operand's parent to be an AMDMFMALayout"
    assert parent.version == 4, \
        "compute_efficient_padded_shared_layout only supports MFMA v4 (CDNA4)"
    return _compute_efficient_padded_shared_layout_impl(
        dot_operand_layout.operand_index,
        dot_operand_layout.k_width,
        parent.version,
        list(parent.warps_per_cta),
        list(parent.instr_shape),
        parent.transposed,
        list(parent.tiles_per_warp),
        parent.element_bitwidth,
        list(parent.cga_layout),
        list(shape),
        dtype.primitive_bitwidth,
        is_k_contig,
    )


"""
buffer_atomic_rmw of cnda4 shares the same signature and functionalities as cdna3.buffer_atomic_rmw.
The cdna4 version additionally supports `fadd` with `bf16`.
"""


@builtin
def buffer_atomic_max(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):
    return _buffer_atomic_rmw_impl('max', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_min(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('min', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_add(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('add', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_and(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('and', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_or(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('or', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xor(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xor', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)


@builtin
def buffer_atomic_xchg(ptr, offsets, value, mask=None, sem=None, scope=None, _semantic=None):

    return _buffer_atomic_rmw_impl('xchg', ptr, offsets, value, "cdna4", mask=mask, sem=sem, scope=scope,
                                   _semantic=_semantic)
<|endoftext|>from dataclasses import dataclass
from typing import List, Any, Optional
from triton._utils import validate_block_shape, canonicalize_dtype, get_primitive_bitwidth
from triton.experimental.gluon.language._layouts import NVMMASharedLayout
import triton.language as tl

__all__ = ["TensorDescriptor", "TensorDescriptorIm2Col"]


def _validate_common_descriptor(tensor, shape, strides, layout, padding, round_f32_to_tf32, block_shape):
    rank = len(shape)
    assert len(strides) == rank, "strides rank mismatch"
    assert 0 < rank <= 5, "rank must be 1-5"
    assert tensor.data_ptr() % 16 == 0, "base must be 16-byte aligned"

    dtype_str = canonicalize_dtype(tensor.dtype)
    elem_bytes = get_primitive_bitwidth(dtype_str) // 8
    for stride in strides[:-1]:
        assert (stride * elem_bytes) % 16 == 0, "strides must be 16-byte aligned"
    for shape_dim in shape:
        assert shape_dim > 0, "shape must be positive"
    assert strides[-1] == 1, "Last dimension must be contiguous"
    assert isinstance(layout, NVMMASharedLayout), "Layout must be NVMMASharedLayout"
    assert padding == "zero" or padding == "nan", "Illegal value for padding"
    if padding == "nan":
        assert tensor.dtype.is_floating_point, "Padding option `nan` is only supported for floating point tensors"
    if round_f32_to_tf32:
        assert dtype_str == "fp32", "round_f32_to_tf32 is only supported for float32 tensors"
    assert elem_bytes * 8 == layout.element_bitwidth
    padding_factor = 2 if layout.fp4_padded else 1
    min_block = layout.swizzle_byte_width // (elem_bytes * padding_factor)
    assert block_shape[-1] >= min_block, \
        f"Expected block_shape[-1] to be at least {min_block} but got {block_shape[-1]}"
    if layout.fp4_padded:
        assert tensor.data_ptr() % 32 == 0, "For fp4_padded, base must 32-byte aligned"
        for stride in strides[:-1]:
            assert (stride * elem_bytes) % 32 == 0, "For fp4_padded, tensor strides must be 32-byte aligned"
        assert tl.target_info.cuda_capability_geq(10, 0), "fp4_padded requires blackwell or newer"
    assert not layout.fp4_padded or layout.swizzle_byte_width == 128, (
        f"FP4 padded operands must be swizzled with 128-byte width, but got {layout.swizzle_byte_width}")
    assert layout.element_bitwidth in [
        8, 16, 32
    ], (f"tensor descriptor dtype must be 8, 16, or 32 bits, but got {layout.element_bitwidth}")
    return rank


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    layout: NVMMASharedLayout
    padding: str = "zero"
    round_f32_to_tf32: bool = False

    def __post_init__(self):
        rank = len(self.shape)
        assert len(self.block_shape) == rank, f"tiled: block_shape must match rank {rank}"
        rank = _validate_common_descriptor(
            self.base,
            self.shape,
            self.strides,
            self.layout,
            self.padding,
            self.round_f32_to_tf32,
            self.block_shape,
        )
        validate_block_shape(self.block_shape)

    @property
    def mode(self) -> str:
        return "tiled"

    def __mangle__(self):
        """Generate a type string matching MLIR types (!ttng.tensordesc or !ttng.tensordesc_im2col)."""
        dtype_str = canonicalize_dtype(self.base.dtype)
        block_shape_str = ','.join(map(str, self.block_shape))
        return f"tensordesc<{dtype_str}[{block_shape_str}],{repr(self.layout)}>"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], layout: NVMMASharedLayout, padding="zero",
                    round_f32_to_tf32=False):
        """
        Create a TensorDescriptor from a tensor.

        Args:
            tensor: Input tensor
            block_shape: Block dimensions for TMA copy.
                Tiled mode: must match tensor rank.
            layout: NVMMASharedLayout for shared memory
            padding: "zero" (default) or "nan" for out-of-bounds padding
            round_f32_to_tf32: Round float32 to TF32 precision (default False)
        """
        return TensorDescriptor(
            tensor,
            tensor.shape,
            tensor.stride(),
            block_shape,
            layout,
            padding,
            round_f32_to_tf32,
        )


@dataclass
class TensorDescriptorIm2Col:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    layout: NVMMASharedLayout
    padding: str = "zero"
    round_f32_to_tf32: bool = False
    element_strides: Optional[List[int]] = None  # Element strides per dimension (optional)
    pixel_box_lower_corner: Optional[List[int]] = None  # Im2col: box start offsets (DHW)
    pixel_box_upper_corner: Optional[List[int]] = None  # Im2col: box end offsets (DHW)

    def __post_init__(self):
        assert len(self.block_shape) == 2, "im2col: block_shape must be 2D"
        rank = _validate_common_descriptor(
            self.base,
            self.shape,
            self.strides,
            self.layout,
            self.padding,
            self.round_f32_to_tf32,
            self.block_shape,
        )
        # Validate element_strides if provided
        if self.element_strides is not None:
            assert len(self.element_strides
                       ) == rank, f"element_strides length mismatch: expected {rank}, got {len(self.element_strides)}"
            for i, s in enumerate(self.element_strides):
                assert 0 < s <= 8, f"element_strides[{i}] must be in (0, 8], got {s}"
            assert self.element_strides[-1] == 1, f"element_strides[-1] must be 1, got {self.element_strides[-1]}"

        assert rank in [3, 4, 5], f"im2col mode requires rank 3, 4, or 5, got {rank}"
        spatial_rank = rank - 2

        assert self.pixel_box_lower_corner is not None, "pixel_box_lower_corner required for im2col"
        assert self.pixel_box_upper_corner is not None, "pixel_box_upper_corner required for im2col"
        assert len(self.pixel_box_lower_corner) == spatial_rank, "pixel_box_lower_corner length mismatch"
        assert len(self.pixel_box_upper_corner) == spatial_rank, "pixel_box_upper_corner length mismatch"

        # Validate box corner ranges based on rank
        offset_ranges = {3: (-32768, 32767), 4: (-128, 127), 5: (-16, 15)}
        lo, hi = offset_ranges[rank]
        for corner, name in [(self.pixel_box_lower_corner, "Lower"), (self.pixel_box_upper_corner, "Upper")]:
            for i, val in enumerate(corner):
                assert lo <= val <= hi, f"pixel_box_{name.lower()}_corner[{i}] must be in [{lo}, {hi}], got {val}"

        # block_shape is [pixelsPerColumn, channelsPerPixel], both must be powers of 2
        def is_power_of_2(n):
            return n > 0 and (n & (n - 1)) == 0

        assert is_power_of_2(self.block_shape[0]), f"block_shape[0] must be power of 2, got {self.block_shape[0]}"
        assert is_power_of_2(self.block_shape[1]), f"block_shape[1] must be power of 2, got {self.block_shape[1]}"

    @property
    def mode(self) -> str:
        return "im2col"

    def __mangle__(self):
        """Generate a type string matching MLIR types (!ttng.tensordesc or !ttng.tensordesc_im2col)."""
        dtype_str = canonicalize_dtype(self.base.dtype)
        block_shape_str = ','.join(map(str, self.block_shape))
        return f"tensordesc_im2col<{dtype_str}[{block_shape_str}],{repr(self.layout)}>"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int], layout: NVMMASharedLayout, padding="zero",
                    round_f32_to_tf32=False, element_strides=None, pixel_box_lower_corner=None,
                    pixel_box_upper_corner=None):
        """
        Create a TensorDescriptorIm2Col from a tensor.

        Args:
            tensor: Input tensor
            block_shape: Block dimensions for TMA copy (2D [pixelsPerColumn, channelsPerPixel])
            layout: NVMMASharedLayout for shared memory
            padding: "zero" (default) or "nan" for out-of-bounds padding
            round_f32_to_tf32: Round float32 to TF32 precision (default False)
            element_strides: Element strides per dimension (optional, each in range (0, 8])
            pixel_box_lower_corner: Im2col mode - box start offsets (DHW dimensions)
            pixel_box_upper_corner: Im2col mode - box end offsets (DHW dimensions)
        """
        return TensorDescriptorIm2Col(
            tensor,
            tensor.shape,
            tensor.stride(),
            block_shape,
            layout,
            padding,
            round_f32_to_tf32,
            element_strides,
            pixel_box_lower_corner,
            pixel_box_upper_corner,
        )
<|endoftext|>from dataclasses import dataclass
from typing import List, Any
from triton._utils import validate_block_shape
from triton.experimental.gluon.language._layouts import PaddedSharedLayout, SwizzledSharedLayout
from triton.experimental.gluon.language.amd.gfx1250 import PartitionedSharedLayout

__all__ = ["TensorDescriptor"]


@dataclass
class TensorDescriptor:
    base: Any
    shape: List[int]
    strides: List[int]
    block_shape: List[int]
    layout: PaddedSharedLayout | SwizzledSharedLayout | PartitionedSharedLayout
    padding: str = "zero"

    def __post_init__(self):
        ndim = len(self.shape)
        assert 1 <= ndim <= 5, f"Expected 1-5 dimensions but got {ndim} dimensions"
        assert len(self.strides) == ndim, f"Expected {ndim} strides but got {len(self.strides)}"
        assert len(self.block_shape) == ndim, \
            f"Expected block_shape to have {ndim} dimensions but got {len(self.strides)}"
        validate_block_shape(self.block_shape)
        assert self.strides[-1] == 1, "Last dimension must be contiguous"
        assert isinstance(self.layout, (PaddedSharedLayout, SwizzledSharedLayout, PartitionedSharedLayout)), \
            "Expected layout to be a PaddedSharedLayout, SwizzledSharedLayout, or PartitionedSharedLayout"
        if isinstance(self.layout, SwizzledSharedLayout):
            assert self.layout.max_phase == 1, "Expected max_phase to be 1 for SwizzledSharedLayout"
        elif isinstance(self.layout, PartitionedSharedLayout):
            assert isinstance(self.layout.partition_layout, (PaddedSharedLayout, SwizzledSharedLayout)), \
                "PartitionedSharedLayout partition_layout must be PaddedSharedLayout or SwizzledSharedLayout"
            if isinstance(self.layout.partition_layout, SwizzledSharedLayout):
                assert self.layout.partition_layout.max_phase == 1, \
                    "Expected max_phase to be 1 for SwizzledSharedLayout in PartitionedSharedLayout"
        assert self.padding == "zero", "Only 'zero' padding is supported"

    @staticmethod
    def from_tensor(tensor: Any, block_shape: List[int],
                    layout: PaddedSharedLayout | SwizzledSharedLayout | PartitionedSharedLayout):
        """ Create a TensorDescriptor object from a tensor.

        Args:
            tensor (torch.Tensor): The input tensor.
            block_shape (List[int]): The block shape of the tensor.
            layout (PaddedSharedLayout | SwizzledSharedLayout | PartitionedSharedLayout): The layout of the tensor in shared memory.

        Returns:
            tensor_descriptor: the created TensorDescriptor object

        """
        return TensorDescriptor(tensor, tensor.shape, tensor.stride(), block_shape, layout)
<|endoftext|>import binascii
import hashlib
import importlib.util
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import List

import triton
import triton.backends


@dataclass
class CompileArgs:
    '''
    A class to contain arguments from command-line parser.
    '''
    path: str = ''
    kernel_name: str = ''
    signature: str = ''
    grid: str = ''
    target: str | None = None
    num_warps: int = 1
    num_stages: int = 3
    out_name: str | None = None
    out_path: Path | None = None


desc = """
Triton ahead-of-time compiler:

This program compiles the kernel with name `kernel-name` in the file at the
provided `path` into self-contained C source-code that embeds the `cubin`
data along with utilities to load, unload and launch the kernel.

signature is provided as a list of (optionally divisibility-hinted) types
or constexpr values, e.g.

`compile.py --kernel-name kernel --signature "*fp32:16, i32:16, 1024, i32" --out-name kernel /path/to/kernel.py`

will compile triton.JITFunction of name `kernel` inside the file `/path/to/kernel.py`.
Said kernel will be specialized such that argument 0, 1 are assumed to be multiple of 16,
and argument 2 is assumed to be a compile-time constant of value 1024, i.e. it won't be part of the generated prototype.

The resulting entry point will have signature

CUresult kernel_{specialization_suffix}(CUstream stream, unsigned gX, unsigned gY, unsigned gZ, float* arg0, int32_t arg1, int32_t arg2)

Different such specialized entry points can be combined using the `linker.py` script.

NOTE: when resolving the scope of /path/to/kernel.py, the file will be executed from within its parent directory with the python interpreter
used to run this `compile.py` script
"""


def main():
    # command-line arguments
    parser = ArgumentParser(description=desc)
    parser.add_argument("path",
                        help="Path to Python source containing desired kernel in its scope. File will be executed.")
    parser.add_argument("--kernel-name", "-n", type=str, default="", help="Name of the kernel to compile",
                        required=True)
    parser.add_argument(
        "--target", "-t", type=str, default=None,
        help="The target to compile towards, in format of '<backend>:<arch>:<warp-size>'; "
        "e.g., 'cuda:80:32', 'hip:gfx942:64'. Default to None, which means using current machine's GPU target")
    parser.add_argument("--num-warps", "-w", type=int, default=1, help="Number of warps to launch the kernel")
    parser.add_argument("--num-stages", "-ns", type=int, default=3,
                        help="Number of stages (meta-parameter of the kernel)")
    parser.add_argument("--out-name", "-on", type=str, default=None, help="Out name for the compiled kernel")
    parser.add_argument("--out-path", "-o", type=Path, default=None, help="Out filename")
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    parser.add_argument("--grid", "-g", type=str, help="Launch grid of the kernel", required=True)
    cli_args = parser.parse_args()
    args = CompileArgs(**vars(cli_args))  # A sanity check to ensure class CompileArgs is updated as well.
    compile_kernel(args)


def compile_kernel(args: CompileArgs):
    out_name = args.out_name if args.out_name else args.kernel_name
    out_path = args.out_path if args.out_path else Path(out_name)

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path)
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, args.kernel_name)
    grid = args.grid.split(",")
    assert len(grid) == 3

    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def hash_signature(signature: List[str]):
        m = hashlib.sha256()
        m.update(" ".join(signature).encode())
        return m.hexdigest()[:8]

    meta_sig = f"warps{args.num_warps}xstages{args.num_stages}"
    sig_hash = hash_signature(signature + [meta_sig])

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        return None

    hints = {(i, ): constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    for key, value in hints.items():
        if value == 1:
            constants[kernel.arg_names[key[0]]] = value
    signature = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature)}
    for key in constants:
        signature[key] = 'constexpr'
    const_sig = 'x'.join([str(v) for v in constants.values()])
    doc_string = [f"{k}={v}" for k, v in constants.items()]
    doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]
    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    attrs = {k: [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}
    kernel.create_binder()
    src = kernel.ASTSource(fn=kernel, constexprs=constants, signature=signature, attrs=attrs)
    target = triton.backends.compiler.GPUTarget(*args.target.split(":")) \
        if args.target else triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.make_backend(target)
    kwargs = {"num_warps": args.num_warps, "num_stages": args.num_stages}
    options = backend.parse_options(kwargs)
    ccinfo = triton.compile(src, target=target, options=options.__dict__)

    if getattr(ccinfo.metadata, "global_scratch_size", 0) > 0:
        raise RuntimeError("AOT compiling kernels with global scratch requirements is not yet implemented")
    if ccinfo.metadata.profile_scratch_size > 0:
        raise RuntimeError("AOT compiling kernels with profile scratch requirements is not yet implemented")

    arg_names = []
    arg_types = []
    arg_names_not_1 = []
    arg_types_not_1 = []
    for i, arg_name in enumerate(kernel.arg_names):
        if arg_name not in constants:
            arg_names.append(arg_name)
            arg_types.append(signature[arg_name])
            arg_names_not_1.append(arg_name)
            arg_types_not_1.append(signature[arg_name])
        elif hints.get((i, ), None) == 1:
            arg_names.append(arg_name)
            arg_types.append("i32")

    # dump C stub code
    suffix = ''
    for i, ty in enumerate(signature.values()):
        if hints.get((i, ), None) == 1:
            suffix += f'{i}c'
        if hints.get((i, ), None) == 16:
            suffix += f'{i}d'
    func_name = '_'.join([out_name, sig_hash, suffix])
    asm = ccinfo.asm[backend.binary_ext]  # store binary data once

    hex_ = str(binascii.hexlify(asm))[2:-1]

    ty_to_cpp = triton.runtime.driver.active.map_python_to_cpp_type
    backend_name = target.backend

    params = {
        "kernel_name": func_name,
        "triton_kernel_name": args.kernel_name,
        "bin_size": len(asm),
        "bin_data": ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]),
        "signature": ", ".join([f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names_not_1, arg_types_not_1)]),
        "full_signature": ", ".join([f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, arg_types)]),
        "arg_pointers": ", ".join([f"&{arg}" for arg in arg_names_not_1] + ["&global_scratch"] + ["&profile_scratch"]),
        "num_args": len(arg_names_not_1) + 2,  # +2 for global and profile scratch
        "kernel_docstring": doc_string,
        "shared": ccinfo.metadata.shared,
        "num_warps": args.num_warps,
        "algo_info": "_".join([const_sig, meta_sig]),
        "gridX": grid[0],
        "gridY": grid[1],
        "gridZ": grid[2],
        "_placeholder": "",
        "warp_size": target.warp_size,
        "backend_name": backend_name,
    }
    output_files = []
    template_dir = Path(__file__).parent / "extra" / backend_name
    for template_path in template_dir.glob('compile.*'):
        ext = template_path.suffix
        output_file = out_path.with_suffix(f".{sig_hash}_{suffix}{ext}")
        with output_file.open("w") as fp:
            fp.write(template_path.read_text().format(**params))
        output_files.append(output_file)

    return func_name, output_files


if __name__ == "__main__":
    main()
<|endoftext|>"""
Helper classes for working with low precision floating point types that
align with the opencompute (OCP) microscaling (MX) specification.
  * MXFP4Tensor: 4-bit E2M1 floating point data
  * MXScaleTensor: 8-bit E8M0 floating point data
Reference: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
"""

import torch


class MXFP4Tensor:

    def __init__(self, data=None, size=None, device=None):
        """
        Tensor class for working with four bit E2M1 floating point data as defined by the
        opencompute microscaling specification.


        Parameters:
        - data: A torch tensor of float32 numbers to convert to fp4e2m1 microscaling format.
        - size: The size of the tensor to create.
        - device: The device on which to create the tensor.
        """
        self.device = device
        if data is not None:
            assert isinstance(data, torch.Tensor), "Parameter data must be a torch tensor"
            self.device = data.device
            self.data = self._from_float(data)
        elif size is not None:
            self.size = size if isinstance(size, tuple) else (size, )
        else:
            raise ValueError("Either parameter data or size must be provided")

    def random(self):
        S = torch.randint(0, 2, size=self.size, dtype=torch.uint8, device=self.device)
        E = torch.randint(0, 4, size=self.size, dtype=torch.uint8, device=self.device)
        M = torch.randint(0, 2, size=self.size, dtype=torch.uint8, device=self.device)

        self.data = ((S << 3) | (E << 1) | M).type(torch.uint8)
        return self

    def to(self, dtype):
        """
        Convert fp4e2m1 data to float32.

        Returns:
        - A torch tensor of type dtype representing the fp4e2m1 data.
        """
        assert dtype == torch.float32, "Currently only float32 is supported for fp4e2m1 to float conversion"

        data = self.data
        S = ((data >> 3) & 0x1).type(dtype)
        E = ((data >> 1) & 0x3).type(dtype)
        M = (data & 0x1).type(dtype)

        # The MXF4 E2M1 spec defines 0bS000 as zero
        value = torch.zeros_like(S)
        is_zero = (E == 0) & (M == 0)
        non_zero_mask = ~is_zero
        if non_zero_mask.any():
            S_nz = S[non_zero_mask]
            E_nz = E[non_zero_mask]
            M_nz = M[non_zero_mask]

            sign = torch.pow(-1, S_nz)
            # Normal and subnormal handling for the exponent and mantissa
            exponent = torch.where(E_nz == 0, E_nz, E_nz - 1)
            mantissa = torch.where(E_nz == 0, M_nz * 0.5, 1.0 + M_nz * 0.5)
            value_nz = sign * torch.pow(2, exponent) * mantissa

            value[non_zero_mask] = value_nz

        # For zeros, the values must remain zero with the correct sign
        value[is_zero & (S == 1)] *= -1
        return value.type(torch.float32)

    def _from_float(self, values):
        """
        Convert float32 numbers to mxf4 e2m1 format.
        * No encodings are reserved for Inf or NaN in mxf4.
        * Conversion from float supports roundTiesToEven rounding mode.
        * If a value exceeds the mxf4 representable range after rounding,
          clamps to the maximum mxf4 magnitude, preserving the sign.
        * If a value has magnitude less than the minimum subnormal magnitude
          in mxf4 after rounding, converts to zero.

        Parameters:
        - values: A torch tensor of float32 numbers to convert to fp4 format.
        """
        S = torch.signbit(values).type(torch.uint8)
        abs_values = torch.abs(values)

        is_zero = (abs_values == 0)
        is_invalid = torch.isnan(values) | torch.isinf(values)

        # Enumerate all possible E2M1 exponent and mantissa values. We will
        # use these to compare the distance between float32 and all possible
        # E2M1 floats to find the nearest E2M1 representable value
        E_bits = torch.tensor([0, 1, 2, 3], dtype=torch.uint8, device=self.device)
        M_bits = torch.tensor([0, 1], dtype=torch.uint8, device=self.device)

        candidate_values = []
        candidate_E = []
        candidate_M = []

        for E in E_bits:
            if E == 0:
                # Subnormals
                exponent = 0
                for M in M_bits:
                    significand = M * 0.5
                    value = significand * (2**exponent)
                    candidate_values.append(value)
                    candidate_E.append(E)
                    candidate_M.append(M)
            else:
                # Normals
                exponent = E.item() - 1
                for M in M_bits:
                    significand = 1.0 + M * 0.5
                    value = significand * (2**exponent)
                    candidate_values.append(value)
                    candidate_E.append(E)
                    candidate_M.append(M)

        candidates = torch.tensor(candidate_values, dtype=torch.float32, device=self.device)
        candidate_E = torch.tensor(candidate_E, dtype=torch.uint8, device=self.device)
        candidate_M = torch.tensor(candidate_M, dtype=torch.uint8, device=self.device)

        abs_values_flat = abs_values.view(-1)
        N = abs_values_flat.shape[0]
        abs_values_expanded = abs_values_flat.unsqueeze(1)

        # Clamp invalid values to the max e2m1 representable value
        max_candidate_value = candidates.max().item()
        abs_values_flat[is_invalid.view(-1)] = max_candidate_value

        # Compute distance between all abs_values and candidate e2m1 values
        errors = torch.abs(abs_values_expanded - candidates.unsqueeze(0))

        # To implement roundTiesToEven, we need to break ties by preferring
        # even mantissas (M == 0). We do so by adding an epsilon bias to shift
        # the closest candidate with an even mantissa closer to the float value
        min_errors, _ = torch.min(errors, dim=1, keepdim=True)
        is_tie = (errors == min_errors)
        # More than one candidate has the min error for some float value
        if is_tie.sum() > 1:
            M_bits_expanded = candidate_M.unsqueeze(0).expand(N, -1)
            tie_breaker = (M_bits_expanded == 0).type(torch.int32)

            errors = errors - (tie_breaker * 1e-6)

        best_indices = torch.argmin(errors, dim=1)

        E_selected = candidate_E[best_indices]
        M_selected = candidate_M[best_indices]
        E = E_selected.view(abs_values.shape)
        M = M_selected.view(abs_values.shape)

        E[is_zero] = 0
        M[is_zero] = 0

        return ((S << 3) | (E << 1) | M).type(torch.uint8)

    def to_packed_tensor(self, dim):
        """
        Packs two e2m1 elements into a single uint8 along the specified dimension.

        Parameters:
        - dim: The dimension along which to pack the elements.

        Returns:
        - A torch tensor of dtype uint8 with two e2m1 elements packed into one uint8.
        """
        data = self.data
        assert 0 <= dim < data.ndim, \
            "The dimension to pack along is not within the range of tensor dimensions"

        size_along_dim = data.size(dim)
        new_size_along_dim = (size_along_dim + 1) // 2

        # If the size is odd, we pad the data along dim with zeros at the end
        if size_along_dim % 2 != 0:
            pad_sizes = [0] * (2 * data.ndim)
            pad_index = (data.ndim - dim - 1) * 2 + 1
            pad_sizes[pad_index] = 1
            data = torch.nn.functional.pad(data, pad_sizes, mode='constant', value=0)

        new_shape = list(data.shape)
        new_shape[dim] = new_size_along_dim
        new_shape.insert(dim + 1, 2)  # packed dimension of length 2
        data = data.reshape(*new_shape)

        low = data.select(dim + 1, 0)
        high = data.select(dim + 1, 1)
        packed = (high << 4) | low

        return packed

    def unpack_packed_tensor(self, packed_tensor, dim, original_shape):
        """
        Unpacks a tensor where two fp4 elements are packed into a single uint8.

        Parameters:
        - packed_tensor: The packed tensor
        - dim: The dimension along which the tensor was packed.
        - original_shape: The shape of the original tensor before packing.

        Returns:
        - A tensor with the original data unpacked into uint8 elements containing one
          fp4e2m1 element in the least significant bits.
        """
        high = (packed_tensor >> 4) & 0xF
        low = packed_tensor & 0xF

        stacked = torch.stack((low, high), dim=dim + 1)

        # Flatten along dim and dim+1 and then merge
        shape = list(stacked.shape)
        new_shape = shape[:dim] + [shape[dim] * 2] + shape[dim + 2:]
        data = stacked.reshape(*new_shape)

        # Remove any padding
        if original_shape[dim] % 2 != 0:
            indices = [slice(None)] * data.ndim
            indices[dim] = slice(0, original_shape[dim])
            data = data[tuple(indices)]

        return data.type(torch.uint8)


class MXScaleTensor:

    def __init__(self, data=None, size=None, device=None):
        """
        Tensor class for working with microscaling E8M0 block scale factors.

        Parameters:
        - data: A torch tensor of float32 numbers to convert to fp8e8m0 microscaling format.
        - size: The size of the tensor to create.
        - device: The device on which to create the tensor.
        """
        self.device = device
        if data is not None:
            assert isinstance(data, torch.Tensor), "Parameter data must be a torch tensor"
            self.device = data.device
            self.data = self._from_float(data)
        elif size is not None:
            self.size = size if isinstance(size, tuple) else (size, )
        else:
            raise ValueError("Either parameter data or size must be provided")

    def random(self, low=None, high=None):
        """
        Generate random E8M0 data within a specified range.
        * Excludes the NaN encoding (255).
        """
        bias = 127

        min_exponent = 0 if low is None else max(0, int(torch.log2(torch.tensor(low))) + bias)
        max_exponent = 254 if high is None else min(254, max(0, int(torch.log2(torch.tensor(high))) + bias))
        assert min_exponent <= max_exponent, "Low must be less than or equal to high"

        E = torch.randint(min_exponent, max_exponent + 1, size=self.size, dtype=torch.uint8, device=self.device)
        self.data = E
        return self

    def to(self, dtype):
        assert dtype == torch.float32, "Currently only float32 is supported for f8e8m0 to float conversion"
        data = self.data.type(dtype)
        is_nan = (data == 255)
        e_biased = data.clone()
        e_biased[is_nan] = 0
        e = e_biased - 127
        value = torch.pow(2.0, e)
        value[is_nan] = torch.nan
        return value.type(dtype)

    def _from_float(self, values):
        """
        Convert float32 numbers to E8M0 format.
        * Values <= 0, NaNs, and Infs are converted to the NaN encoding (255).
        * Positive values are converted by computing the floor of log2(value) to get the exponent.

        Parameters:
        - values: A torch tensor of float32 numbers to convert to E8M0 format.
        """
        result = torch.empty_like(values, dtype=torch.uint8, device=self.device)

        is_invalid = torch.isnan(values) | torch.isinf(values) | (values <= 0)
        result[is_invalid] = 255

        valid_values = values[~is_invalid]
        e = torch.floor(torch.log2(valid_values))
        e_biased = e + 127
        e_biased_int = e_biased.type(torch.int32)
        e_biased_clamped = torch.clamp(e_biased_int, 0, 254)
        result[~is_invalid] = e_biased_clamped.type(torch.uint8)

        return result
<|endoftext|>import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

# fmt: off


def create_ragged_descriptor(T, block_shape, ragged_dim=0):
    """
    Given a 2- or 3-dimensional tensor T, this creates a 'ragged descriptor'
    which behaves like a concatenation (along the first axis) of subarrays
    of potentially unequal size.

    The load_ragged and store_ragged device functions can be used to read
    and write from subarrays T[slice_off : slice_off + slice_size]
    with hardware bounds-checking preventing any sort of leakage outside
    the subarray.
    """

    block_shape = list(block_shape)
    tensor_shape = list(T.shape)
    rank = len(tensor_shape)

    if ragged_dim < 0:
        ragged_dim += rank

    assert 0 <= ragged_dim < rank - 1, "last dimension cannot be ragged"
    assert rank <= 3, "read-write ragged descriptors must have at most 3 dimensions"

    assert len(block_shape) == rank, "block shape must have same length as tensor shape"

    max_int = 0x7fff0000
    billion = 0x40000000  # == 2**30

    assert tensor_shape[ragged_dim] <= billion, "number of rows may not exceed 2**30"
    tensor_shape[ragged_dim] = billion
    ragged_stride = T.stride(ragged_dim)

    # we prepend an extra two dimensions and rely on the fact that pointers
    # have 64-bit wraparound semantics:
    tma_stride = [2**34 - ragged_stride, ragged_stride] + [T.stride(i) for i in range(rank)]
    tma_shape  = [max_int, max_int] + tensor_shape
    box_shape  = [1, 1] + block_shape

    return TensorDescriptor(T, tma_shape, tma_stride, box_shape)


@triton.jit
def to_ragged_indices(slice_off, slice_size, row):
    """
    Helper function for load_ragged and store_ragged.
    """

    billion = 0x40000000  # == 2**30
    x = billion - slice_size + row
    y = slice_off + slice_size

    return billion, y, x


@triton.jit
def load_ragged(TMA, slice_off, slice_size, coords, ragged_dim: tl.constexpr = 0):
    """
    Read from a subarray T[slice_off : slice_off + slice_size] with
    hardware bounds-checking, where reading outside the subarray gives zeros.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.load().
    """

    tl.static_assert(len(TMA.shape) == len(coords) + 2, "TMA must be a read-write ragged descriptor")

    c0, c1, c2 = to_ragged_indices(slice_off, slice_size, coords[ragged_dim])
    data = TMA.load([c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:])
    data = tl.reshape(data, data.shape[2:])
    return data


@triton.jit
def store_ragged(TMA, slice_off, slice_size, coords, data, ragged_dim: tl.constexpr = 0):
    """
    Write to a subarray T[slice_off : slice_off + slice_size] with
    hardware bounds-checking, where writes outside the subarray are masked
    correctly.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.store().
    """

    c0, c1, c2 = to_ragged_indices(slice_off, slice_size, coords[ragged_dim])
    data = tl.reshape(data, [1, 1] + data.shape)
    TMA.store([c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:], data)


@triton.jit
def atomic_add_ragged(TMA, slice_off, slice_size, coords, data, ragged_dim: tl.constexpr = 0):
    """
    Atomic add into a subarray T[slice_off : slice_off + slice_size] with
    hardware bounds-checking, where adds outside the subarray are masked
    correctly.

    Coords should be an appropriately-sized list of integers, just like in
    TMA.atomic_add().
    """

    c0, c1, c2 = to_ragged_indices(slice_off, slice_size, coords[ragged_dim])
    data = tl.reshape(data, [1, 1] + data.shape)
    TMA.atomic_add([c0, c1] + coords[:ragged_dim] + [c2] + coords[ragged_dim + 1:], data)
<|endoftext|>"""Run a Python script under Triton Global Memory Sanitizer.

This module provides a small command-line wrapper runs the wrapped script with GSan enabled, e.g.
    python -m triton.tools.gsan my_script.py
"""

from __future__ import annotations

import argparse
import contextlib
import runpy
import sys
from pathlib import Path
from typing import Sequence

import triton
import torch
from triton.experimental.gsan._allocator import create_mem_pool


def _parse_args(argv: Sequence[str] | None = None) -> tuple[Path, list[str], str]:
    parser = argparse.ArgumentParser(description="Run a Python script with Triton Global Memory Sanitizer.")
    parser.add_argument("script", help="Python script to execute")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments forwarded to the target script")
    args = parser.parse_args(argv)

    script_args = list(args.script_args)
    if script_args[:1] == ["--"]:
        script_args = script_args[1:]

    return Path(args.script), script_args


@contextlib.contextmanager
def _script_context(script_path: Path, script_args: Sequence[str]):
    original_argv = sys.argv[:]
    original_path = sys.path[:]

    sys.argv = [str(script_path), *script_args]
    sys.path.insert(0, str(script_path.parent))
    try:
        yield script_path
    finally:
        sys.argv = original_argv
        sys.path[:] = original_path


def main(argv: Sequence[str] | None = None) -> int:
    script_path, script_args = _parse_args(argv)

    script_path = script_path.resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"Script not found: {script_path}")

    triton.knobs.compilation.instrumentation_mode = "gsan"

    with torch.cuda.use_mem_pool(create_mem_pool()), \
        _script_context(script_path, script_args):
        runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
<|endoftext|>from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union

from dataclasses import dataclass


def _exists(x):
    return x is not None


class LinkerError(Exception):
    pass


@dataclass
class KernelLinkerMeta:
    orig_kernel_name: str
    arg_names: Sequence[str]
    arg_ctypes: Sequence[str]
    sizes: Sequence[Union[int, None]]
    sig_hash: str
    triton_suffix: str
    suffix: str
    num_specs: int
    """ number of specialized arguments """


class HeaderParser:

    def __init__(self) -> None:
        import re

        # [kernel_name, c signature]
        self.linker_directives = re.compile("//[\\s]*tt-linker:[\\s]*([\\w]+):(.+):(.+)")
        # [name, hash, suffix]
        self.kernel_name = re.compile("^([\\w]+)_([\\w]+)_([\\w]*)$")
        # [(type, name)]
        self.c_sig = re.compile("[\\s]*(\\w+)\\s(\\w+)[,]?")
        # [d|c]
        self.arg_suffix = re.compile("[c,d]")
        # [backend_name]
        self.backend_name_re = re.compile("//[\\s]*tt-linker-backend:[\\s]*([\\w]+)")

        self.kernels = defaultdict(list)
        self.backend_name = None

    def extract_linker_meta(self, header: str):
        for ln in header.splitlines():
            if ln.startswith("//"):
                m = self.linker_directives.match(ln)
                if _exists(m):
                    ker_name, c_sig, algo_info = m.group(1), m.group(2), m.group(3)
                    name, sig_hash, suffix = self._match_name(ker_name)
                    c_types, arg_names = self._match_c_sig(c_sig)
                    num_specs, sizes = self._match_suffix(suffix, c_sig)
                    self._add_kernel(
                        "_".join([name, algo_info]),
                        KernelLinkerMeta(
                            orig_kernel_name=name,
                            arg_names=arg_names,
                            arg_ctypes=c_types,
                            sizes=sizes,
                            sig_hash=sig_hash,
                            triton_suffix=suffix,
                            suffix=suffix,
                            num_specs=num_specs,
                        ),
                    )
                else:
                    m = self.backend_name_re.match(ln)
                    if _exists(m):
                        backend_name = m.group(1)
                        if self.backend_name is None:
                            self.backend_name = backend_name
                        elif self.backend_name != backend_name:
                            raise RuntimeError(f"differing backend {self.backend_name} vs. {backend_name}")

    def _match_name(self, ker_name: str):
        m = self.kernel_name.match(ker_name)
        if _exists(m):
            name, sig_hash, suffix = m.group(1), m.group(2), m.group(3)
            return name, sig_hash, suffix
        raise LinkerError(f"{ker_name} is not a valid kernel name")

    def _match_c_sig(self, c_sig: str):
        m = self.c_sig.findall(c_sig)
        if len(m):
            tys, args = [], []
            for ty, arg_name in m:
                tys.append(ty)
                args.append(arg_name)
            return tys, args

        raise LinkerError(f"{c_sig} is not a valid argument signature")

    def _match_suffix(self, suffix: str, c_sig: str):
        args = c_sig.split(",")
        s2i = {"c": 1, "d": 16}
        num_specs = 0
        sizes = []
        # scan through suffix, suffix only includes indexes followed by d or c.
        for i in range(len(args)):
            pos = 0
            idx_matched = suffix.startswith(str(i))
            if not idx_matched:
                continue
            pos += len(str(i))
            if self.arg_suffix.match(suffix, pos):
                num_specs += 1
                sizes.extend([None] * (i - len(sizes)))
                sizes.append(s2i[suffix[pos]])
                pos += 1
            suffix = suffix[pos:]

        if len(suffix) > 0:
            raise Exception(f"Has invalid extra suffix: {suffix}")
        sizes.extend([None] * (len(args) - len(sizes)))

        return num_specs, sizes

    def _add_kernel(self, name: str, ker: KernelLinkerMeta):
        if name in self.kernels:
            last: KernelLinkerMeta = self.kernels[name][-1]

            for cur, new_ in zip(last.arg_ctypes, ker.arg_ctypes):
                if cur != new_:
                    raise LinkerError(
                        f"Mismatched signature for kernel {name}: \n\texisting sig is: {','.join(last.arg_ctypes)}\n\tcurrent is: {','.join(ker.arg_ctypes)}"
                    )

        self.kernels[name].append(ker)


def gen_signature_with_full_args(m):
    return ", ".join([f"{ty} {arg}" for ty, arg in zip(m.arg_ctypes, m.arg_names)])


def gen_signature(m):
    arg_types = [ty for ty, hint in zip(m.arg_ctypes, m.sizes) if hint != 1]
    arg_names = [arg for arg, hint in zip(m.arg_names, m.sizes) if hint != 1]
    sig = ", ".join([f"{ty} {arg}" for ty, arg in zip(arg_types, arg_names)])
    return sig


# generate declarations of kernels with meta-parameter and constant values
def make_algo_decls(name: str, metas: Sequence[KernelLinkerMeta]) -> str:
    return f"""
TT_ResultTy {name}(TT_StreamTy stream, {gen_signature_with_full_args(metas[-1])});
void load_{name}();
void unload_{name}();
    """


# generate declarations of kernels with meta-parameter and constant values
def make_global_decl(meta: KernelLinkerMeta) -> str:
    return f"""
TT_ResultTy {meta.orig_kernel_name}_default(TT_StreamTy stream, {gen_signature_with_full_args(meta)});
TT_ResultTy {meta.orig_kernel_name}(TT_StreamTy stream, {gen_signature_with_full_args(meta)}, int algo_id);
void load_{meta.orig_kernel_name}();
void unload_{meta.orig_kernel_name}();
    """


# generate dispatcher function for kernels with different meta-parameter and constant values
def make_default_algo_kernel(meta: KernelLinkerMeta) -> str:
    src = f"TT_ResultTy {meta.orig_kernel_name}_default(TT_StreamTy stream, {gen_signature_with_full_args(meta)}){{\n"
    src += (f"  return {meta.orig_kernel_name}(stream, {', '.join(meta.arg_names)}, 0);\n")
    src += "}\n"
    return src


# generate dispatcher function for kernels with different integer value hints
def make_kernel_hints_dispatcher(name: str, metas: Sequence[KernelLinkerMeta]) -> str:
    src = f"// launcher for: {name}\n"
    for meta in sorted(metas, key=lambda m: -m.num_specs):
        src += f"TT_ResultTy {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(TT_StreamTy stream, {gen_signature(meta)});\n"
    src += "\n"

    src += (f"TT_ResultTy {name}(TT_StreamTy stream, {gen_signature_with_full_args(metas[-1])}){{")
    src += "\n"
    for meta in sorted(metas, key=lambda m: -m.num_specs):
        cond_fn = (  #
            lambda val, hint: f"((uintptr_t){val} % {hint} == 0)"  #
            if hint == 16  #
            else f"({val} == {hint})"  #
            if hint == 1  #
            else None)
        conds = " && ".join([  #
            cond_fn(val, hint)  #
            for val, hint in zip(meta.arg_names, meta.sizes)  #
            if hint is not None
        ])
        src += (f"  if ({conds})\n" if any(meta.sizes) else "if (1)\n"
                )  # Edge case where no specializations hence no dispatching required
        arg_names = [arg for arg, hint in zip(meta.arg_names, meta.sizes) if hint != 1]
        src += f"    return {meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}(stream, {', '.join(arg_names)});\n"
    src += "\n"
    src += "  return TT_ERROR_INVALID_VALUE;\n"
    src += "}\n"

    for mode in ["load", "unload"]:
        src += f"\n// {mode} for: {name}\n"
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            src += f"void {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n"
        src += f"void {mode}_{name}() {{"
        src += "\n"
        for meta in sorted(metas, key=lambda m: -m.num_specs):
            src += (f"  {mode}_{meta.orig_kernel_name}_{meta.sig_hash}_{meta.suffix}();\n")
        src += "}\n"
    return src


# generate dispatcher function for kernels with different meta-parameter and constant values
def make_kernel_meta_const_dispatcher(meta: KernelLinkerMeta) -> str:
    src = f"TT_ResultTy {meta.orig_kernel_name}(TT_StreamTy stream, {gen_signature_with_full_args(meta)}, int algo_id){{\n"
    src += f"  assert (algo_id < (int)sizeof({meta.orig_kernel_name}_kernels));\n"
    src += f"  return {meta.orig_kernel_name}_kernels[algo_id](stream, {', '.join(meta.arg_names)});\n"
    src += "}\n"
    return src


# generate definition of function pointers of kernel dispatchers based on meta-parameter and constant values
def make_func_pointers(names: str, meta: KernelLinkerMeta) -> str:
    # the table of hint dispatchers
    src = f"typedef TT_ResultTy (*kernel_func_t)(TT_StreamTy stream, {gen_signature_with_full_args(meta)});\n"
    src += f"kernel_func_t {meta.orig_kernel_name}_kernels[] = {{\n"
    for name in names:
        src += f"  {name},\n"
    src += "};\n"
    return src


# generate definition for load/unload functions for kernels with different meta-parameter and constant values
def make_kernel_load_def(names: str, meta: KernelLinkerMeta) -> str:
    src = ""
    for mode in ["load", "unload"]:
        src += f"void {mode}_{meta.orig_kernel_name}(void){{\n"
        for name in names:
            src += f"  {mode}_{name}();\n"
        src += "}\n\n"
    return src


def make_get_num_algos_decl(meta: KernelLinkerMeta) -> str:
    src = f"int {meta.orig_kernel_name}_get_num_algos(void);"
    return src


def make_get_num_algos_def(meta: KernelLinkerMeta) -> str:
    src = f"int {meta.orig_kernel_name}_get_num_algos(void){{\n"
    src += f"  return (int)(sizeof({meta.orig_kernel_name}_kernels) / sizeof({meta.orig_kernel_name}_kernels[0]));\n"
    src += "}\n"
    return src


desc = """
Triton ahead-of-time linker:

This program takes in header files generated by compile.py, and generates a
single entry-point responsible for dispatching the user's input to the right
kernel given the specializations that were compiled.

Example usage:
python link.py /path/to/headers/*.h -o kernel_name
"""

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description=desc)
    parser.add_argument(
        "headers",
        nargs="+",
        help="Paths to header files to link. Must include linker directive annotations (autogenerated by ttc)",
    )
    parser.add_argument("--out", "-o", type=Path, help="Out filename")
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="String to prefix kernel dispatcher names",
    )
    args = parser.parse_args()

    # metadata
    parser = HeaderParser()
    includes = []
    for header in args.headers:
        h_path = Path(header)
        h_str = h_path.read_text()
        includes.append(h_path.name)
        parser.extract_linker_meta(h_str)

    # generate headers
    algo_decls = [make_algo_decls(name, meta) for name, meta in parser.kernels.items()]
    meta_lists = [meta for name, meta in parser.kernels.items()]
    meta = meta_lists[0][0]
    get_num_algos_decl = make_get_num_algos_decl(meta)
    global_decl = make_global_decl(meta)
    backend_prelude = (Path(__file__).parent / "extra" / parser.backend_name / "link.h").read_text()
    with args.out.with_suffix(".h").open("w") as fp:
        out = backend_prelude
        out += "\n".join(algo_decls)
        out += "\n"
        out += get_num_algos_decl
        out += "\n"
        out += global_decl
        fp.write(out)

    # generate source
    defs = [make_kernel_hints_dispatcher(name, meta) for name, meta in parser.kernels.items()]
    names = [name for name in parser.kernels.keys()]
    func_pointers_def = make_func_pointers(names, meta)
    meta_const_def = make_kernel_meta_const_dispatcher(meta)
    load_unload_def = make_kernel_load_def(names, meta)
    get_num_algos_def = make_get_num_algos_def(meta)
    default_algo_kernel = make_default_algo_kernel(meta)
    with args.out.with_suffix(".c").open("w") as fp:
        out = backend_prelude
        out += "#include <stdint.h>\n"
        out += "#include <assert.h>\n"
        out += "\n"
        out += "\n".join(defs)
        out += "\n"
        out += func_pointers_def
        out += "\n"
        out += get_num_algos_def
        out += "\n"
        out += meta_const_def
        out += "\n"
        out += load_unload_def
        out += "\n"
        out += default_algo_kernel
        fp.write(out)
<|endoftext|>import argparse
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class Symbol:
    _name: str
    _op_name: str
    _ret_type: str
    _arg_names: List[str]
    _arg_types: List[str]

    def __init__(
        self,
        name: str,
        op_name: str,
        ret_type: str,
        arg_names: List[str],
        arg_types: List[str],
    ) -> None:
        '''
        A symbol is a function declaration.
        :param name: name of the symbol
        :param op_name: name of the operation
        :param ret_type: return type of the operation
        :param arg_names: names of the arguments
        :param arg_types: types of the arguments
        '''
        self._name = name
        self._op_name = op_name
        self._ret_type = ret_type
        self._arg_names = list(arg_names)
        self._arg_types = list(arg_types)

    @property
    def name(self) -> str:
        return self._name

    @property
    def op_name(self) -> str:
        return self._op_name

    @property
    def ret_type(self) -> str:
        return self._ret_type

    @property
    def arg_names(self) -> List[str]:
        return self._arg_names

    @property
    def arg_types(self) -> List[str]:
        return self._arg_types


def convert_type(type_str) -> Optional[str]:
    if type_str == "i32":
        return "int32"
    elif type_str == "u32":
        return "uint32"
    elif type_str == "i64":
        return "int64"
    elif type_str == "u64":
        return "uint64"
    elif type_str == "float":
        return "fp32"
    elif type_str == "double":
        return "fp64"
    else:
        # ignore other types, such as pointer types
        return None


def to_unsigned(type_str) -> str:
    if type_str == "int32":
        return "uint32"
    elif type_str == "int64":
        return "uint64"
    else:
        return type_str


class ExternLibrary(ABC):
    _name: str
    _path: str
    _symbols: Dict[str, Symbol]
    _format: bool
    _grouping: bool

    def __init__(
        self,
        name: str,
        path: str,
        format: bool = True,
        grouping: bool = True,
    ) -> None:
        '''
        Abstract class for extern library.
        :param name: name of the library
        :param path: path of the library
        :param format: whether to format the generated stub file
        '''
        self._name = name
        self._path = path
        self._symbols = {}
        self._format = format
        self._grouping = grouping

    @property
    def name(self) -> str:
        return self._name

    @property
    def path(self) -> str:
        return self._path

    @property
    def symbols(self) -> Dict[str, Symbol]:
        return self._symbols

    @property
    def grouping(self) -> bool:
        return self._grouping

    @abstractmethod
    def parse_symbols(self, input_file) -> None:
        pass

    @abstractmethod
    def _output_stubs(self) -> str:
        pass

    def generate_stub_file(self, output_dir) -> None:
        file_str = self._output_stubs()
        if file_str is None or len(file_str) == 0:
            raise Exception("file_str is empty")

        output_file = f"{output_dir}/{self._name}.py"
        with open(output_file, "w") as f:
            f.write(file_str)
            f.close()
            if self._format:
                subprocess.Popen(["autopep8", "-a", "-r", "-i", output_file], stdout=subprocess.PIPE).communicate()
                subprocess.Popen(["isort", output_file], stdout=subprocess.PIPE).communicate()


class Libdevice(ExternLibrary):
    _symbol_groups: Dict[str, List[Symbol]]

    def __init__(self, path) -> None:
        '''
        Constructor for Libdevice.
        :param path: path of the libdevice library
        '''
        super().__init__("libdevice", path)
        self._symbol_groups = {}
        self.is_pure = True

    @staticmethod
    def _extract_symbol(line) -> Optional[Symbol]:
        # Extract symbols from line in the following format:
        # "define [internal] <ret_type> @<name>(<arg_types>,)"
        entries = line.split("@")
        ret_str = entries[0]
        func_str = entries[1]
        # Get ret_type, skip internal symbols
        ret_strs = ret_str.split()
        if ret_strs[1] == "internal":
            return None
        ret_type = convert_type(ret_strs[1])
        if ret_type is None:
            return None
        # Get function name
        func_strs = func_str.split("(")
        func_name = func_strs[0].replace("@", "")
        op_name = func_name.replace("__nv_", "")
        if 'ieee' in op_name:
            return None
        # Get arg_types
        arg_strs = func_strs[1].split(",")
        arg_types = []
        arg_names = []
        for i, arg_str in enumerate(arg_strs):
            arg_type = convert_type(arg_str.split()[0])
            if arg_type is None:
                return None
            arg_name = 'arg' + str(i)
            arg_types.append(arg_type)
            arg_names.append(arg_name)
        if op_name == "sad":
            # Special case for sad, where the last argument is an unsigned int
            arg_types[-1] = to_unsigned(arg_types[-1])
        elif op_name.startswith("u"):
            # LLVM does not differentiate between signed and unsigned integer type.
            # We have to convert the types to unsigned
            ret_type = to_unsigned(ret_type)
            for i, arg_type in enumerate(arg_types):
                arg_types[i] = to_unsigned(arg_type)
        return Symbol(func_name, op_name, ret_type, arg_names, arg_types)

    def _group_symbols(self) -> None:
        symbol_set = {}
        for symbol in self._symbols.values():
            op_name = symbol.op_name
            symbol_set[op_name] = symbol

        # Group functions together by renaming.
        renaming = {
            'llabs': 'abs', 'acosf': 'acos', 'acoshf': 'acosh', 'dadd_rd': 'add_rd', 'fadd_rd': 'add_rd', 'dadd_rn':
            'add_rn', 'fadd_rn': 'add_rn', 'dadd_ru': 'add_ru', 'fadd_ru': 'add_ru', 'dadd_rz': 'add_rz', 'fadd_rz':
            'add_rz', 'asinf': 'asin', 'asinhf': 'asinh', 'atanf': 'atan', 'atan2f': 'atan2', 'atanhf': 'atanh',
            'brevll': 'brev', 'cbrtf': 'cbrt', 'ceilf': 'ceil', 'clzll': 'clz', 'copysignf': 'copysign', 'cosf': 'cos',
            'coshf': 'cosh', 'cospif': 'cospi', 'cyl_bessel_i0f': 'cyl_bessel_i0', 'cyl_bessel_i1f': 'cyl_bessel_i1',
            'fdiv_rd': 'div_rd', 'ddiv_rd': 'div_rd', 'fdiv_rn': 'div_rn', 'ddiv_rn': 'div_rn', 'fdiv_ru': 'div_ru',
            'ddiv_ru': 'div_ru', 'fdiv_rz': 'div_rz', 'ddiv_rz': 'div_rz', 'erff': 'erf', 'erfcf': 'erfc', 'erfcinvf':
            'erfcinv', 'erfcxf': 'erfcx', 'erfinvf': 'erfinv', 'expf': 'exp', 'exp10f': 'exp10', 'exp2f': 'exp2',
            'expm1f': 'expm1', 'fabsf': 'abs', 'fabs': 'abs', 'fast_fdividef': 'fast_dividef', 'fdimf': 'fdim', 'ffsll':
            'ffs', 'floorf': 'floor', 'fmaf': 'fma', 'fmaf_rd': 'fma_rd', 'fmaf_rn': 'fma_rn', 'fmaf_ru': 'fma_ru',
            'fmaf_rz': 'fma_rz', 'fmodf': 'fmod', 'uhadd': 'hadd', 'hypotf': 'hypot', 'ilogbf': 'ilogb', 'isinff':
            'isinf', 'isinfd': 'isinf', 'isnanf': 'isnan', 'isnand': 'isnan', 'j0f': 'j0', 'j1f': 'j1', 'jnf': 'jn',
            'ldexpf': 'ldexp', 'lgammaf': 'lgamma', 'llrintf': 'llrint', 'llroundf': 'llround', 'logf': 'log', 'log10f':
            'log10', 'log1pf': 'log1p', 'log2f': 'log2', 'logbf': 'logb', 'umax': 'max', 'llmax': 'max', 'ullmax':
            'max', 'fmaxf': 'max', 'fmax': 'max', 'umin': 'min', 'llmin': 'min', 'ullmin': 'min', 'fminf': 'min',
            'fmin': 'min', 'dmul_rd': 'mul_rd', 'fmul_rd': 'mul_rd', 'dmul_rn': 'mul_rn', 'fmul_rn': 'mul_rn',
            'dmul_ru': 'mul_ru', 'fmul_ru': 'mul_ru', 'dmul_rz': 'mul_rz', 'fmul_rz': 'mul_rz', 'umul24': 'mul24',
            'umulhi': 'mulhi', 'mul64hi': 'mulhi', 'umul64hi': 'mulhi', 'nearbyintf': 'nearbyint', 'nextafterf':
            'nextafter', 'norm3df': 'norm3d', 'norm4df': 'norm4d', 'normcdff': 'normcdf', 'normcdfinvf': 'normcdfinv',
            'popcll': 'popc', 'powif': 'pow', 'powi': 'pow', 'powf': 'pow', 'rcbrtf': 'rcbrt', 'frcp_rd': 'rcp_rd',
            'drcp_rd': 'rcp_rd', 'frcp_rn': 'rcp_rn', 'drcp_rn': 'rcp_rn', 'frcp_ru': 'rcp_ru', 'drcp_ru': 'rcp_ru',
            'frcp_rz': 'rcp_rz', 'drcp_rz': 'rcp_rz', 'remainderf': 'remainder', 'urhadd': 'rhadd', 'rhypotf': 'rhypot',
            'rintf': 'rint', 'rnorm3df': 'rnorm3d', 'rnorm4df': 'rnorm4d', 'roundf': 'round', 'rsqrtf': 'rsqrt',
            'frsqrt_rn': 'rsqrt_rn', 'usad': 'sad', 'scalbnf': 'scalbn', 'signbitf': 'signbit', 'signbitd': 'signbit',
            'sinf': 'sin', 'sinhf': 'sinh', 'sinpif': 'sinpi', 'sqrtf': 'sqrt', 'fsqrt_rd': 'sqrt_rd', 'dsqrt_rd':
            'sqrt_rd', 'fsqrt_rn': 'sqrt_rn', 'dsqrt_rn': 'sqrt_rn', 'fsqrt_ru': 'sqrt_ru', 'dsqrt_ru': 'sqrt_ru',
            'fsqrt_rz': 'sqrt_rz', 'dsqrt_rz': 'sqrt_rz', 'fsub_rd': 'sub_rd', 'dsub_rd': 'sub_rd', 'fsub_rn': 'sub_rn',
            'dsub_rn': 'sub_rn', 'fsub_ru': 'sub_ru', 'dsub_ru': 'sub_ru', 'fsub_rz': 'sub_rz', 'dsub_rz': 'sub_rz',
            'tanf': 'tan', 'tanhf': 'tanh', 'tgammaf': 'tgamma', 'truncf': 'trunc', 'y0f': 'y0', 'y1f': 'y1', 'ynf':
            'yn'
        }

        for symbol in self._symbols.values():
            op_name = symbol.op_name
            if op_name in renaming:
                op_name = renaming[op_name]
                symbol._op_name = op_name
            if op_name in self._symbol_groups:
                self._symbol_groups[op_name].append(symbol)
            else:
                self._symbol_groups[op_name] = [symbol]

    def parse_symbols(self, input_file) -> None:
        if len(self.symbols) > 0:
            return
        output = subprocess.check_output(["grep", "define", input_file]).decode().splitlines()
        for line in output:
            symbol = self._extract_symbol(line)
            if symbol is None:
                continue
            self._symbols[symbol.name] = symbol

        self._group_symbols()

    def _output_stubs(self) -> str:
        # Generate python functions in the following format:
        # @extern.extern
        # def <op_name>(<args>, _builder=None):
        #   arg_type_symbol_dict = {[arg_type]: {(symbol, ret_type)}}
        #   return core.extern_elementwise("libdevice", <path>, <args>, <arg_type_symbol_dict>, _builder)
        import_str = "from . import core\n"

        header_str = ""
        func_str = ""
        for symbols in self._symbol_groups.values():
            func_str += "@core.extern\n"
            func_name_str = f"def {symbols[0].op_name}("
            for arg_name in symbols[0].arg_names:
                func_name_str += f"{arg_name}, "
            func_name_str += "_builder=None):\n"

            return_str = f"\treturn core.extern_elementwise(\"{self._name}\", libdevice_path(), ["
            for arg_name in symbols[0].arg_names:
                return_str += f"{arg_name}, "
            return_str += "], \n"

            arg_type_symbol_dict_str = "{"
            for symbol in symbols:
                arg_type_symbol_dict_str += "("
                for arg_type in symbol.arg_types:
                    arg_type_symbol_dict_str += f'core.dtype("{arg_type}"),'
                ret_type = f'core.dtype("{symbol.ret_type}")'
                arg_type_symbol_dict_str += "): (\"" + symbol.name + "\", " + ret_type + "),\n"
            arg_type_symbol_dict_str += "}"

            return_str += arg_type_symbol_dict_str
            return_str += f", is_pure={self.is_pure}"
            return_str += ", _builder=_builder)\n"

            func_str += func_name_str + return_str + "\n"
        file_str = import_str + header_str + func_str

        return file_str


class LLVMDisassembler:
    _path: str
    _ll_file: str

    def __init__(self, path) -> None:
        '''
        Invoke llvm-dis to disassemble the given file.
        :param path: path to llvm-dis
        '''
        self._path = path
        self._ll_file = "/tmp/extern_lib.ll"

    def disasm(self, lib_path: str) -> None:
        subprocess.Popen([self._path, lib_path, "-o", self.ll_file], stdout=subprocess.PIPE).communicate()

    @property
    def ll_file(self) -> str:
        return self._ll_file

    @property
    def path(self) -> str:
        return self._path


extern_libs = ["libdevice"]


def build(
    llvm_dis_path: str,
    lib_path: str,
    lib_name: str,
    output_dir: str,
) -> None:
    '''
      Interface function to build the library file.
      :param llvm_dis_path: path to the llvm-dis binary
      :param lib_path: path to the external library file
      :param lib_name: name of the library
      :param output_dir: path to the output directory
    '''
    if lib_name == "libdevice":
        extern_lib = Libdevice(lib_path)
    else:
        raise Exception(f"Unknown extern library: {lib_name}")

    llvm_disassembler = LLVMDisassembler(llvm_dis_path)
    llvm_disassembler.disasm(lib_path)

    extern_lib.parse_symbols(llvm_disassembler.ll_file)
    extern_lib.generate_stub_file(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llvm-dis", dest="llvm_dis_path", help="Path to llvm-dis", default="llvm-dis")
    parser.add_argument("--lib-path", dest="lib_path", help="Path to the extern library")
    parser.add_argument("--lib-name", dest="lib_name", help="Name of the extern library")
    parser.add_argument("--output", dest="output_dir", help="Output file path", default="/tmp/")
    args = parser.parse_args()

    build(args.llvm_dis_path, args.lib_path, args.lib_name, args.output_dir)
<|endoftext|># MIT License

# Copyright (c) 2020 Da Yan @ HKUST

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functools
import os
import re
import subprocess
import tempfile

FLINE_RE = re.compile(r'\s*/\*\w{4,}\*/\s*([^;]*;)\s*/\* 0x(\w{16}) \*/\s*')
SLINE_RE = re.compile(r'\s*/\* 0x(\w{16}) \*/\s*')
FNAME_RE = re.compile(r'\s*Function : (\w+)\s*')
BRA_RE = re.compile(r'(.*BRA(?:\.U)? )(0x\w+);')


def parseCtrl(sline):
    enc = int(SLINE_RE.match(sline).group(1), 16)
    stall = (enc >> 41) & 0xf
    yld = (enc >> 45) & 0x1
    wrtdb = (enc >> 46) & 0x7
    readb = (enc >> 49) & 0x7
    watdb = (enc >> 52) & 0x3f

    yld_str = 'Y' if yld == 0 else '-'
    wrtdb_str = '-' if wrtdb == 7 else str(wrtdb)
    readb_str = '-' if readb == 7 else str(readb)
    watdb_str = '--' if watdb == 0 else f'{watdb:02d}'
    return f'{watdb_str}:{readb_str}:{wrtdb_str}:{yld_str}:{stall:x}'


def processSassLines(fline, sline, labels):
    asm = FLINE_RE.match(fline).group(1)
    # Remove tailing space
    if asm.endswith(" ;"):
        asm = asm[:-2] + ";"
    ctrl = parseCtrl(sline)
    # BRA target address
    if BRA_RE.match(asm) is not None:
        target = int(BRA_RE.match(asm).group(2), 16)
        if target in labels:
            pass
        else:
            labels[target] = len(labels)
    return (f'{ctrl}', f'{asm}')


@functools.lru_cache()
def get_sass(cubin_asm, fun=None):
    fd, path = tempfile.mkstemp()
    try:
        with open(fd, 'wb') as cubin:
            cubin.write(cubin_asm)
        sass = extract(path, fun)
    finally:
        os.remove(path)
    return sass


def path_to_cuobjdump():
    from triton import knobs
    return knobs.nvidia.cuobjdump.path


def extract(file_path, fun):
    cuobjdump = path_to_cuobjdump()
    if fun is None:
        sass_str = subprocess.check_output([cuobjdump, "-sass", file_path])
    else:
        sass_str = subprocess.check_output([cuobjdump, "-fun", fun, "-sass", file_path])
    sass_lines = sass_str.splitlines()
    line_idx = 0
    while line_idx < len(sass_lines):
        line = sass_lines[line_idx].decode()
        # format:
        # function : <function_name>
        # .headerflags: ...
        # /*0000*/ asmstr /*0x...*/
        #                 /*0x...*/

        # Looking for new function header (function: <name>)
        while FNAME_RE.match(line) is None:
            line_idx += 1
            if line_idx < len(sass_lines):
                line = sass_lines[line_idx].decode()
            else:
                return

        fname = FNAME_RE.match(line).group(1)
        ret = ''
        ret += f'Function:{fname}\n'
        line_idx += 2  # bypass .headerflags
        line = sass_lines[line_idx].decode()
        # Remapping address to label
        labels = {}  # address -> label_idx
        # store sass asm in buffer and them print them (for labels)
        # (ctrl, asm)
        asm_buffer = []
        while FLINE_RE.match(line) is not None:
            # First line (Offset ASM Encoding)
            fline = sass_lines[line_idx].decode()
            line_idx += 1
            # Second line (Encoding)
            sline = sass_lines[line_idx].decode()
            line_idx += 1
            asm_buffer.append(processSassLines(fline, sline, labels))
            # peek the next line
            if line_idx >= len(sass_lines):
                break
            line = sass_lines[line_idx].decode()
        # Print sass
        # label naming convention: LBB#i
        for idx, (ctrl, asm) in enumerate(asm_buffer):
            # Print label if this is BRA target
            offset = idx * 16
            if offset in labels:
                label_name = f'LBB{labels[offset]}'
                ret += f'{label_name}:\n'
            ret += ctrl + '\t'
            # if this is BRA, remap offset to label
            if BRA_RE.match(asm):
                target = int(BRA_RE.match(asm).group(2), 16)
                target_name = f'LBB{labels[target]}'
                asm = BRA_RE.sub(rf'\1{target_name};', asm)
            ret += asm + '\n'
        ret += '\n'
        return ret
<|endoftext|>from __future__ import annotations

from collections import deque
from collections.abc import Mapping

from triton.tools.triton_to_gluon_translator.ordered_set import ordered_set


def stable_toposort(edges: Mapping[int, ordered_set[int]]) -> list[int]:  # noqa: C901
    nodes: list[int] = list(edges.keys())
    order = {node: idx for idx, node in enumerate(nodes)}
    index: dict[int, int] = {}
    low: dict[int, int] = {}
    stack: list[int] = []
    on_stack: ordered_set[int] = ordered_set()
    comp_of: dict[int, int] = {}
    comps: list[tuple[int, ...]] = []

    def dfs(node: int) -> None:
        idx = len(index)
        index[node] = low[node] = idx
        stack.append(node)
        on_stack.add(node)
        for nxt in edges[node]:
            if nxt not in index:
                dfs(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], index[nxt])
        if low[node] == index[node]:
            comp: list[int] = []
            while True:
                cur = stack.pop()
                on_stack.remove(cur)
                comp.append(cur)
                comp_of[cur] = len(comps)
                if cur == node:
                    break
            comps.append(tuple(sorted(comp, key=order.__getitem__)))

    for node in nodes:
        if node not in index:
            dfs(node)

    priority = [min(order[n] for n in comp) for comp in comps]
    indegree = [0] * len(comps)
    outgoing: list[ordered_set[int]] = [ordered_set() for _ in comps]
    for node in nodes:
        src = comp_of[node]
        for nbr in edges[node]:
            dst = comp_of[nbr]
            if src != dst and dst not in outgoing[src]:
                outgoing[src].add(dst)
                indegree[dst] += 1

    queue = deque(sorted((cid for cid, deg in enumerate(indegree) if deg == 0), key=priority.__getitem__))
    result: list[int] = []
    while queue:
        cid = queue.popleft()
        result.extend(comps[cid])
        for nxt in sorted(outgoing[cid], key=priority.__getitem__):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    assert len(result) == len(nodes), "component graph contains a cycle"
    return result
<|endoftext|>"""
Copied from https://github.com/bustawin/ordered-set-37
"""

# Unlicense
# This is free and unencumbered software released into the public domain.

# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.

# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

# For more information, please refer to <http://unlicense.org/>

import itertools
from collections import OrderedDict
from typing import TYPE_CHECKING, MutableSet

if TYPE_CHECKING:
    # trying to avoid polluting the global namespace with typing names.
    from typing import Iterable, Iterator, TypeVar

    T = TypeVar("T")


class ordered_set(MutableSet["T"]):
    """
    A set that preserves insertion order by internally using a dict.
    """

    __slots__ = ("values", )

    def __init__(self, elements: "Iterable[T] | None" = None) -> None:
        self.values = OrderedDict.fromkeys(elements or [])

    def add(self, value: "T") -> None:
        self.values[value] = None

    def update(self, values: "Iterable[T]") -> None:
        self.values.update((k, None) for k in values)

    def clear(self) -> None:
        self.values.clear()

    def discard(self, value: "T") -> None:
        self.values.pop(value, None)

    def __getitem__(self, index: int) -> "T":
        try:
            return next(itertools.islice(self.values, index, index + 1))
        except StopIteration as exc:
            raise IndexError(f"index {index} out of range") from exc

    def __contains__(self, x: object) -> bool:
        return self.values.__contains__(x)

    def __add__(self, other: "ordered_set[T]") -> "ordered_set[T]":
        return ordered_set(itertools.chain(self, other))

    def __len__(self) -> int:
        return self.values.__len__()

    def __iter__(self) -> "Iterator[T]":
        return self.values.__iter__()

    def __str__(self) -> str:
        return f"{{{', '.join(str(i) for i in self)}}}"

    def __repr__(self) -> str:
        return f"<ordered_set {self}>"
<|endoftext|># type: ignore

import builtins

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.hopper import (
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
    warpgroup_mma_init,
)

from triton.tools.triton_to_gluon_translator.common_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.common_helpers import (
    tl_dot_decomposed_block_scales_impl,
    default_blocked_layout,
)
from triton.tools.triton_to_gluon_translator.nvidia_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.nvidia_helpers import (
    tl_dot_mma_sync,
    get_shared_memory_mma_operand,
)

# ---- NVIDIA Hopper dot ----


@gluon.constexpr_function
def _get_default_max_num_imprecise_acc(
    a_ty: ttgl.block_type,
    b_ty: ttgl.block_type,
    max_num_imprecise_acc: int | None,
) -> int:
    # FIXME: Get this from builder options. Default is 2**30 for Hopper.
    max_num_imprecise_acc_default = 2**30
    if max_num_imprecise_acc is not None:
        return max_num_imprecise_acc
    if a_ty.element_ty.is_fp8() and b_ty.element_ty.is_fp8():
        return max_num_imprecise_acc_default
    return 0


@gluon.constexpr_function
def _get_default_input_precision(
    allow_tf32: bool | None,
    input_precision: str | None,
) -> str:
    assert input_precision is None or allow_tf32 is None, "Only one of input_precision and allow_tf32 can be specified"
    if input_precision is not None:
        return input_precision

    allow_tf32 = allow_tf32 or True
    return "tf32" if allow_tf32 else "ieee"


@gluon.constexpr_function
def _operand_supports_mmav3(dtype: ttgl.dtype) -> bool:
    return (dtype.is_fp8() or dtype.is_fp16() or dtype.is_bf16() or dtype.is_fp32() or dtype.is_fp64()
            or dtype.is_int8() or dtype.is_uint8())


@gluon.constexpr_function
def tl_dot_mmav3_supported(
    a_ty: ttgl.block_type,
    b_ty: ttgl.block_type,
    num_warps: int,
    input_precision: str,
    max_num_imprecise_acc: int,
    out_dtype: ttgl.dtype,
) -> bool:

    M = a_ty.shape[0]
    N = b_ty.shape[1]
    K = a_ty.shape[1]

    a_dtype = a_ty.element_ty
    b_dtype = b_ty.element_ty

    # Minimum MMA instruction K shape.
    mma_min_K = 256 // a_dtype.primitive_bitwidth
    if K < mma_min_K:
        return False

    # Only rank 2 supported.
    if len(a_ty.shape) != 2 or len(b_ty.shape) != 2:
        return False

    # Minimum 4 warps.
    if num_warps % 4 != 0:
        return False

    # Minimum MMA instruction shape along M and N.
    if M % 64 != 0 or N % 8 != 0:
        return False

    # Accepted LHS dtypes: float8e5m2, float8e4m3fn, u/int8, float16, bfloat16, float32.
    if a_dtype not in [
            ttgl.float8e5,
            ttgl.float8e4nv,
            ttgl.int8,
            ttgl.uint8,
            ttgl.float16,
            ttgl.bfloat16,
            ttgl.float32,
    ]:
        return False

    # Check float8 -> float32 accumulation with imprecise acc.
    if max_num_imprecise_acc < 32 and a_dtype in [ttgl.float8e5, ttgl.float8e4nv] and out_dtype.is_fp32():
        return False

    # fp32 operands require tf32.
    if a_dtype.is_fp32() and b_dtype.is_fp32():
        return input_precision == "tf32"

    # Check supported operand dtypes.
    return _operand_supports_mmav3(a_dtype) and _operand_supports_mmav3(b_dtype)


@gluon.constexpr_function
def _mmav3_acc_layout(
    num_warps: int,
    c_shape: list[int],
    a_dtype: ttgl.dtype,
    b_dtype: ttgl.dtype,
    out_dtype: ttgl.dtype,
):
    k = 256 // a_dtype.primitive_bitwidth
    assert c_shape[0] % 64 == 0 and c_shape[1] % 8 == 0, "c_shape must be divisible [64, 8]"

    # fmt: off
    if a_dtype.is_floating():
        valid_n = [256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176,
                   168, 160, 152, 144, 136, 128, 120, 112, 104, 96,  88,  # noqa: E241
                    80,  72,  64,  56,  48,  40,  32,  24,  16,  8]       # noqa: E241, E127
    else:
        assert a_dtype.is_int()
        valid_n = [224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 24, 16, 8]
    # fmt: on

    m = 16
    m_warps = max(c_shape[0] // m, 1)
    n_warps = max(num_warps // m_warps, 1)
    max_n = max(c_shape[1] // n_warps, 8)
    instr_shape = None
    for n in valid_n:
        if c_shape[1] % n == 0 and n <= max_n:
            instr_shape = [m, n, k]
            break

    assert instr_shape is not None, "Failed to find valid instruction shape"

    warps_per_tile = [4, 1]
    shape_per_warp = [16, instr_shape[1]]
    while True:
        if warps_per_tile[0] * warps_per_tile[1] >= num_warps:
            break
        if c_shape[0] > shape_per_warp[0] * warps_per_tile[0]:
            warps_per_tile[0] *= 2
        else:
            warps_per_tile[1] *= 2

    return ttgl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_tile,
        instr_shape=instr_shape,
    )


@gluon.jit
def tl_dot_mmav3(
    a: ttgl.tensor,
    b: ttgl.tensor,
    acc: ttgl.tensor | None,
    input_precision: builtins.str,
    max_num_imprecise_acc: int,
    out_dtype: ttgl.dtype,
):
    el_ty: ttgl.constexpr = a.type.element_ty
    allow_transpose: ttgl.constexpr = el_ty.is_fp16() or el_ty.is_bf16()
    a_smem = get_shared_memory_mma_operand(a, 0, allow_transpose, is_fp4_padded=False, force_transpose=False)
    b_smem = get_shared_memory_mma_operand(b, 1, allow_transpose, is_fp4_padded=False, force_transpose=False)

    c_shape: ttgl.constexpr = a.shape[:-1] + [b.shape[-1]]
    if acc is not None:
        ttgl.static_assert(acc.shape == c_shape, "accumulator shape is incompatible")
        ttgl.static_assert(acc.type.element_ty == out_dtype, "accumulator dtype is incompatible")

    mma_layout: ttgl.constexpr = _mmav3_acc_layout(
        ttgl.num_warps(),
        c_shape,
        a.type.element_ty,
        b.type.element_ty,
        out_dtype,
    )

    if acc is None:
        ret_layout: ttgl.constexpr = default_blocked_layout(c_shape, ttgl.num_warps())
        acc = ttgl.zeros(c_shape, out_dtype, layout=mma_layout)
    else:
        ret_layout: ttgl.constexpr = acc.type.layout
        acc = ttgl.convert_layout(acc, mma_layout)

    fence_async_shared()
    wgmma_acc = warpgroup_mma_init(acc)
    wgmma_acc = warpgroup_mma(a_smem, b_smem, wgmma_acc, is_async=True)
    acc = warpgroup_mma_wait(num_outstanding=0, deps=(wgmma_acc, ))

    return ttgl.convert_layout(acc, ret_layout)


@gluon.jit
def tl_dot(
    a: ttgl.tensor,
    b: ttgl.tensor,
    acc: ttgl.tensor | None = None,
    input_precision: builtins.str | None = None,
    allow_tf32: builtins.bool | None = None,
    max_num_imprecise_acc: int | None = None,
    out_dtype: ttgl.dtype = ttgl.float32,
):
    input_prec: ttgl.constexpr = _get_default_input_precision(
        allow_tf32,
        input_precision,
    )
    max_num_imprecise: ttgl.constexpr = _get_default_max_num_imprecise_acc(
        a.type,
        b.type,
        max_num_imprecise_acc,
    )
    num_warps: ttgl.constexpr = ttgl.num_warps()

    if tl_dot_mmav3_supported(a.type, b.type, num_warps, input_prec, max_num_imprecise, out_dtype):
        return tl_dot_mmav3(a, b, acc, input_prec, max_num_imprecise, out_dtype)
    else:
        return tl_dot_mma_sync(a, b, acc, input_precision, out_dtype)


# ---- NVIDIA Hopper dot-scaled ----


@gluon.jit
def tl_dot_scaled(
    lhs,
    lhs_scale,
    lhs_format,
    rhs,
    rhs_scale,
    rhs_format,
    acc=None,
    fast_math=False,
    lhs_k_pack=True,
    rhs_k_pack=True,
    out_dtype=ttgl.float32,
):
    return tl_dot_decomposed_block_scales_impl(
        tl_dot_scaled,
        tl_dot,
        lhs,
        lhs_scale,
        lhs_format,
        rhs,
        rhs_scale,
        rhs_format,
        acc=acc,
        fast_math=fast_math,
        lhs_k_pack=lhs_k_pack,
        rhs_k_pack=rhs_k_pack,
        out_dtype=out_dtype,
    )


# ---- NVIDIA obj dispatch ----


@gluon.jit
def tl_obj_gather(obj, x_offsets, y_offset):
    ttgl.static_assert(not isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor),
                       "descriptor gather is not supported on Hopper")
    return obj.gather(x_offsets, y_offset)


@gluon.jit
def tl_obj_scatter(obj, value, x_offsets, y_offset):
    ttgl.static_assert(not isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor),
                       "descriptor scatter is not supported on Hopper")
    return obj.scatter(value, x_offsets, y_offset)
<|endoftext|># type: ignore

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

# hack to workaround limited dependencies tracking.
# TODO: fix this by pulling imports into the generated file.
from triton.language.target_info import current_target  # noqa: F401

# ---- layout utilities ----


@gluon.constexpr_function
def get_num_threads_per_warp(target=None) -> ttgl.constexpr:
    if target is None:
        target = current_target()
    if target is not None and target.backend == "hip":
        gfx_major = int(target.arch[3:-2])
        return ttgl.constexpr(32 if gfx_major >= 10 else 64)
    return ttgl.constexpr(32)


@gluon.jit
def get_num_threads_per_program():
    return ttgl.num_warps() * get_num_threads_per_warp(current_target())


@gluon.constexpr_function
def default_blocked_layout(shape: ttgl.constexpr, num_warps: ttgl.constexpr, target=None) -> ttgl.constexpr:
    rank = len(shape)
    size_per_thread = [1] * rank
    threads_per_warp = [1] * rank
    # TODO: pick a better layout based on shape. Using this allows to not have to convert layout when broadcasting but may blow up register pressure.
    threads_per_warp[rank - 1] = get_num_threads_per_warp(target)
    warps_per_cta = [1] * rank
    warps_per_cta[0] = num_warps
    order = list(range(rank - 1, -1, -1))
    return ttgl.BlockedLayout(
        size_per_thread=size_per_thread,
        threads_per_warp=threads_per_warp,
        warps_per_cta=warps_per_cta,
        order=order,
    )


@gluon.constexpr_function
def get_swizzle_byte_width(bitwidth):
    swizzle = min(bitwidth, 128)
    swizzle = 0 if swizzle < 32 else swizzle
    return swizzle


@gluon.constexpr_function
def get_int_type(bitwidth):
    if bitwidth == 64:
        return ttgl.int64
    elif bitwidth == 32:
        return ttgl.int32
    elif bitwidth == 16:
        return ttgl.int16
    elif bitwidth == 8:
        return ttgl.int8
    else:
        assert False, f"Unsupported bitwidth: {bitwidth}"


# ---- portable ops ----


@gluon.jit
def tl_arange(start: ttgl.constexpr, stop: ttgl.constexpr = None):
    layout: ttgl.constexpr = default_blocked_layout([stop - start], ttgl.num_warps())
    return ttgl.arange(start, stop, layout=layout)


@gluon.jit
def tl_full(shape, value, dtype=None):
    layout: ttgl.constexpr = default_blocked_layout(shape, ttgl.num_warps())
    return ttgl.full(shape, value, dtype, layout=layout)


@gluon.jit
def tl_trans(value, *dims):
    return value.trans(*dims)


@gluon.constexpr_function
def cat_permute_order(rank, dim):
    order = list(range(rank))
    order.insert(dim, rank)
    return order


@gluon.constexpr_function
def cat_result_shape(input_shape, dim):
    result_shape = list(input_shape)
    result_shape[dim] *= 2
    return result_shape


@gluon.jit
def tl_cat(input, other, can_reorder=False, dim=0):
    c = ttgl.join(input, other)
    order: ttgl.constexpr = cat_permute_order(len(input.shape), dim)
    c = ttgl.permute(c, order)
    shape: ttgl.constexpr = cat_result_shape(input.shape, dim)
    c = ttgl.reshape(c, shape)
    return reset_to_default_layout(c)


@gluon.jit
def reset_to_default_layout(value):
    ty: ttgl.constexpr = value.type
    if isinstance(ty, ttgl.tuple_type):
        out = ()
        for i in ttgl.static_range(len(value)):
            r = ttgl.convert_layout(value[i], layout=default_blocked_layout(value[i].type.shape, ttgl.num_warps()))
            out = out + (r, )
        return out
    elif isinstance(value, ttgl.tensor) and isinstance(value.type, ttgl.distributed_type):
        layout: ttgl.constexpr = default_blocked_layout(ty.shape, ttgl.num_warps())
        return ttgl.convert_layout(value, layout=layout)
    else:
        return value


@gluon.constexpr_function
def get_split_src_layout(shape: ttgl.constexpr, num_warps: ttgl.constexpr, target=None) -> ttgl.constexpr:
    rank = len(shape)
    size_per_thread = [1 if i != rank - 1 else 2 for i in range(rank)]
    threads_per_warp = [1 for _ in range(rank)]
    remaining_threads = get_num_threads_per_warp(target)
    for dim in range(rank - 2, -1, -1):
        threads_per_warp[dim] = min(shape[dim], remaining_threads)
        remaining_threads = remaining_threads // threads_per_warp[dim]
    warps_per_cta = [1 for _ in range(rank)]
    warps_per_cta[0] = num_warps
    order = list(range(rank - 1, -1, -1))
    return ttgl.BlockedLayout(
        size_per_thread=size_per_thread,
        threads_per_warp=threads_per_warp,
        warps_per_cta=warps_per_cta,
        order=order,
    )


@gluon.jit
def set_split_src_layout(value):
    layout: ttgl.constexpr = get_split_src_layout(value.type.shape, ttgl.num_warps())
    return ttgl.convert_layout(value, layout=layout)


@gluon.constexpr_function
def build_expand_dims_layout(shape, expand_dims, num_warps):
    if isinstance(shape, ttgl.tuple):
        shape = shape.values
    assert isinstance(shape, list), (f"expected shape to be a list, got {shape} which is {type(shape)}")
    parent_shape = list(shape)
    for dim in expand_dims:
        parent_shape.insert(dim, 1)
    layout = default_blocked_layout(parent_shape, num_warps)
    for dim in reversed(expand_dims):
        layout = ttgl.SliceLayout(dim=dim, parent=layout)
    return layout


@gluon.jit
def convert_to_expand_dims_layout(value, expand_dims: list[int]):
    layout: ttgl.constexpr = build_expand_dims_layout(value.shape, expand_dims, ttgl.num_warps())
    return ttgl.convert_layout(value, layout)


# ---- dot-scaled sub-helpers (vendor-neutral) ----


@gluon.jit
def tl_dot_decomposed_scale_to_16(scale, compute_type: ttgl.constexpr):
    large_fp_type: ttgl.constexpr = ttgl.float32 if compute_type == ttgl.float16 else compute_type
    int_width: ttgl.constexpr = large_fp_type.primitive_bitwidth
    int_type: ttgl.constexpr = get_int_type(int_width)

    zexted = ttgl.cast(scale, int_type)
    shift_value: ttgl.constexpr = large_fp_type.fp_mantissa_width
    shl_res = zexted << shift_value
    scale_fp = ttgl.cast(shl_res, large_fp_type, bitcast=True)
    if large_fp_type != compute_type:
        scale_fp = ttgl.cast(scale_fp, compute_type)
    return scale_fp


@gluon.constexpr_function
def tl_dot_get_expand_dims_layout(scale_ty, num_warps, rank):
    shape = scale_ty.shape.values + [1]
    blocked = default_blocked_layout(shape, num_warps)
    slice = ttgl.SliceLayout(rank, blocked)
    return slice


@gluon.constexpr_function
def tl_dot_get_permute_order(rank, dim):
    order = list(range(rank))
    order.insert(dim + 1, rank)
    return order


@gluon.constexpr_function
def tl_dot_get_reshape_shape(scale_ty, dim, scale_factor):
    shape = list(scale_ty.shape.values)
    shape.pop()
    shape[dim] *= scale_factor
    return shape


@gluon.jit
def tl_dot_decomposed_broadcast_scale(scale, dim, scale_factor: ttgl.constexpr):
    scale_ty: ttgl.constexpr = scale.type
    rank: ttgl.constexpr = len(scale_ty.shape)

    num_warps: ttgl.constexpr = ttgl.num_warps()
    slice_enc: ttgl.constexpr = tl_dot_get_expand_dims_layout(scale_ty, num_warps, rank)
    scale = ttgl.convert_layout(scale, slice_enc)
    expand_scale = scale.expand_dims(rank)
    broadcast_scale = expand_scale.broadcast_to(scale.type.shape + (scale_factor, ))
    permute_order: ttgl.constexpr = tl_dot_get_permute_order(rank, dim)
    transposed_scale = broadcast_scale.permute(permute_order)
    reshape_shape: ttgl.constexpr = tl_dot_get_reshape_shape(broadcast_scale.type, dim, scale_factor)
    return transposed_scale.reshape(reshape_shape)


@gluon.constexpr_function
def tl_dot_decomposed_get_transposed_order(rank):
    assert rank >= 2
    order = list(range(rank - 2))
    order += [rank - 1, rank - 2]
    return order


@gluon.jit
def tl_dot_decomposed_extend_and_broadcast_scale(v, scale, compute_type: ttgl.constexpr, operand_index: ttgl.constexpr,
                                                 scale_factor: ttgl.constexpr):
    rank: ttgl.constexpr = len(v.type.shape)
    k_dim: ttgl.constexpr = rank - 1 if operand_index == 0 else rank - 2

    if operand_index == 1:
        order: ttgl.constexpr = tl_dot_decomposed_get_transposed_order(rank)
        scale = ttgl.permute(scale, order)

    scale16 = tl_dot_decomposed_scale_to_16(scale, compute_type)
    reshape_scale = tl_dot_decomposed_broadcast_scale(scale16, k_dim, scale_factor)
    return ttgl.convert_layout(reshape_scale, v.type.layout), scale


@gluon.jit
def tl_dot_decomposed_mask_nan(mxfp, scale, fast_math):
    ttgl.static_assert(fast_math, "TODO: support non-fast-math")
    return mxfp


@gluon.jit
def tl_dot_decomposed_scale_arg(v, scale, arg_format: ttgl.constexpr, operand_index: ttgl.constexpr,
                                compute_type: ttgl.constexpr, fast_math: ttgl.constexpr, scale_factor: ttgl.constexpr):
    is_fp4: ttgl.constexpr = arg_format == "e2m1"
    rank: ttgl.constexpr = len(v.type.shape)
    k_dim: ttgl.constexpr = rank - 1 if operand_index == 0 else rank - 2

    if is_fp4:
        v = ttgl.fp4_to_fp(v, compute_type, k_dim)
    else:
        v = ttgl.cast(v, compute_type)
    if scale is None:
        return v
    else:
        reshape_scale, scale = tl_dot_decomposed_extend_and_broadcast_scale(v, scale, compute_type, operand_index,
                                                                            scale_factor)
        mxfp = ttgl.mul(v, reshape_scale)
        return tl_dot_decomposed_mask_nan(mxfp, scale, fast_math)


@gluon.constexpr_function
def tl_dot_decomposed_deduce_scale_factor(v, scale, arg_format, operand_index, k_pack):
    if scale is None:
        return 0
    if scale.numel == 1:
        return 0

    k_dim = len(v.shape) - 1 if operand_index == 0 else len(v.shape) - 2
    unpack_factor = 2 if arg_format == "e2m1" and k_pack else 1
    k_size = v.shape[k_dim] * unpack_factor
    scale_factor = k_size // scale.shape[-1]
    assert scale_factor in (16, 32), f"scale factor must be 16 or 32. Got {scale_factor}"
    return scale_factor


@gluon.jit
def tl_dot_decomposed_block_scales_impl(
    tl_dot_scaled_fn: ttgl.constexpr,
    tl_dot_fn: ttgl.constexpr,
    lhs,
    lhs_scale,
    lhs_format,
    rhs,
    rhs_scale,
    rhs_format,
    acc=None,
    fast_math=False,
    lhs_k_pack=True,
    rhs_k_pack=True,
    out_dtype=ttgl.float32,
):
    if lhs_scale is None and rhs_scale is not None:
        lhs_trans = tl_trans(lhs)
        rhs_trans = tl_trans(rhs)
        if acc is not None:
            orig_layout: ttgl.constexpr = acc.type.layout
            acc = tl_trans(acc)
        result = tl_dot_scaled_fn(
            rhs_trans,
            rhs_scale,
            rhs_format,
            lhs_trans,
            lhs_scale,
            lhs_format,
            acc,
            fast_math,
            lhs_k_pack,
            rhs_k_pack,
            out_dtype,
        )
        result = tl_trans(result)
        if acc is not None:
            result = ttgl.convert_layout(result, orig_layout)
        return result
    else:
        ttgl.static_assert(not (not lhs_k_pack or not rhs_k_pack), "TODO: support m/n packed formats")
        compute_type: ttgl.constexpr = (ttgl.float16 if
                                        (lhs_format == "fp16" or rhs_format == "fp16") else ttgl.bfloat16)
        lhs_scale_factor: ttgl.constexpr = tl_dot_decomposed_deduce_scale_factor(lhs, lhs_scale, lhs_format, 0,
                                                                                 lhs_k_pack)
        rhs_scale_factor: ttgl.constexpr = tl_dot_decomposed_deduce_scale_factor(rhs, rhs_scale, rhs_format, 1,
                                                                                 rhs_k_pack)
        scale_factor: ttgl.constexpr = lhs_scale_factor or rhs_scale_factor or 32
        ttgl.static_assert(
            lhs_scale_factor == 0 or rhs_scale_factor == 0 or lhs_scale_factor == rhs_scale_factor,
            "Operands must have the same scale factor",
        )

        scale_a = tl_dot_decomposed_scale_arg(lhs, lhs_scale, lhs_format, 0, compute_type, fast_math, scale_factor)
        scale_b = tl_dot_decomposed_scale_arg(rhs, rhs_scale, rhs_format, 1, compute_type, fast_math, scale_factor)

        return tl_dot_fn(scale_a, scale_b, acc, out_dtype=out_dtype)
<|endoftext|># type: ignore

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    tcgen05_commit,
    tcgen05_mma,
    tcgen05_mma_scaled,
)
from triton.experimental.gluon.language.nvidia.blackwell import tma as tma_blackwell

from triton.tools.triton_to_gluon_translator.common_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.common_helpers import (
    default_blocked_layout,
    get_num_threads_per_warp,
    tl_dot_decomposed_block_scales_impl,
)
from triton.tools.triton_to_gluon_translator.nvidia_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.nvidia_helpers import (
    tl_dot_mma_sync,
    get_shared_memory_mma_operand,
)

# ---- NVIDIA Blackwell dot ----


@gluon.constexpr_function
def tl_dot_mmav5_supported(a_ty, b_ty, num_warps, input_precision, allow_tf32, max_num_imprecise_acc):
    assert max_num_imprecise_acc in [0, None], ("max_num_imprecise_acc only applies to Hopper warp_group_dot")
    assert input_precision is None or allow_tf32 is None, (
        "Only one of input_precision and allow_tf32 can be specified")
    if input_precision is None and (allow_tf32 or allow_tf32 is None):
        input_precision = "tf32"

    M = a_ty.shape[0]
    N = b_ty.shape[1]
    K = a_ty.shape[1]
    min_K = 256 // a_ty.element_ty.primitive_bitwidth
    if a_ty.element_ty.is_int() or b_ty.element_ty.is_int():
        return False
    if (min(a_ty.element_ty.primitive_bitwidth, b_ty.element_ty.primitive_bitwidth) >= 32
            and input_precision != "tf32"):
        return False
    return (num_warps in [4, 8] and len(a_ty.shape) == 2 and len(b_ty.shape) == 2 and K >= min_K and M >= 64
            and N >= 16)


@gluon.jit
def tl_dot_blackwell(
    a,
    b,
    acc=None,
    input_precision=None,
    allow_tf32=None,
    max_num_imprecise_acc=None,
    out_dtype=ttgl.float32,
):
    M: ttgl.constexpr = a.type.shape[0]
    N: ttgl.constexpr = b.type.shape[1]

    allow_transpose = not a.type.element_ty.is_fp32()
    a_smem = get_shared_memory_mma_operand(a, 0, allow_transpose)
    b_smem = get_shared_memory_mma_operand(b, 1, allow_transpose)

    m: ttgl.constexpr = 128 if M >= 128 else 64
    n: ttgl.constexpr = 256 if N >= 256 else N

    acc_dtype: ttgl.constexpr = acc.dtype if acc is not None else out_dtype
    col_stride: ttgl.constexpr = 32 // acc_dtype.primitive_bitwidth
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)
    acc_tmem = allocate_tensor_memory(acc_dtype, [M, N], acc_tmem_layout)
    tmem_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    if acc is not None:
        acc_temp = ttgl.convert_layout(acc, tmem_reg_layout)
    else:
        acc_temp = ttgl.zeros([M, N], out_dtype, layout=tmem_reg_layout)
    acc_tmem.store(acc_temp)
    fence_async_shared()
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    tcgen05_mma(a_smem, b_smem, acc_tmem, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    out = acc_tmem.load()
    ret_layout: ttgl.constexpr = default_blocked_layout([M, N], ttgl.num_warps())
    out = ttgl.convert_layout(out, ret_layout)
    return out


# ---- NVIDIA dot dispatch ----


@gluon.jit
def tl_dot(
    a,
    b,
    acc=None,
    input_precision=None,
    allow_tf32=None,
    max_num_imprecise_acc=None,
    out_dtype=ttgl.float32,
):
    num_warps: ttgl.constexpr = ttgl.num_warps()
    if tl_dot_mmav5_supported(a.type, b.type, num_warps, input_precision, allow_tf32, max_num_imprecise_acc):
        return tl_dot_blackwell(a, b, acc, input_precision, allow_tf32, max_num_imprecise_acc, out_dtype)
    else:
        return tl_dot_mma_sync(a, b, acc, input_precision, out_dtype)


# ---- NVIDIA dot-scaled ----


@gluon.constexpr_function
def tl_dot_scaled_mmav5_supported(a_ty, b_ty, num_warps):
    M = a_ty.shape[0]
    N = b_ty.shape[1]
    K = a_ty.shape[1]
    min_K = 256 // a_ty.element_ty.primitive_bitwidth
    return (num_warps in [4, 8] and len(a_ty.shape) == 2 and len(b_ty.shape) == 2 and K >= min_K and M >= 128
            and N >= 16)


@gluon.jit
def tl_dot_scaled_blackwell(
    lhs,
    lhs_scale,
    lhs_format,
    rhs,
    rhs_scale,
    rhs_format,
    acc=None,
    fast_math=False,
    lhs_k_pack=True,
    rhs_k_pack=True,
    out_dtype=ttgl.float32,
):
    is_a_fp4: ttgl.constexpr = lhs_format == "e2m1"
    is_b_fp4: ttgl.constexpr = rhs_format == "e2m1"

    mixed_prec: ttgl.constexpr = lhs_format != rhs_format
    is_a_mixed_prec_fp4: ttgl.constexpr = mixed_prec and is_a_fp4
    is_b_mixed_prec_fp4: ttgl.constexpr = mixed_prec and not is_a_fp4 and is_b_fp4

    is_mmav5_fp4_padded_a: ttgl.constexpr = is_a_mixed_prec_fp4 or not lhs_k_pack
    is_mmav5_fp4_padded_b: ttgl.constexpr = is_b_mixed_prec_fp4 or not rhs_k_pack

    a_smem = get_shared_memory_mma_operand(
        lhs,
        0,
        allow_transpose=not is_a_fp4,
        is_fp4_padded=is_mmav5_fp4_padded_a,
        force_transpose=not lhs_k_pack,
    )
    b_smem = get_shared_memory_mma_operand(
        rhs,
        1,
        allow_transpose=not is_b_fp4,
        is_fp4_padded=is_mmav5_fp4_padded_b,
        force_transpose=not rhs_k_pack,
    )

    M: ttgl.constexpr = lhs.type.shape[0]
    N: ttgl.constexpr = rhs.type.shape[1]

    m: ttgl.constexpr = 128
    n: ttgl.constexpr = 256 if N >= 256 else N

    acc_dtype: ttgl.constexpr = acc.dtype if acc is not None else out_dtype
    col_stride: ttgl.constexpr = 32 // acc_dtype.primitive_bitwidth
    acc_tmem_layout: ttgl.constexpr = TensorMemoryLayout([m, n], col_stride=col_stride)
    acc_tmem = allocate_tensor_memory(acc_dtype, [M, N], acc_tmem_layout)
    tmem_reg_layout: ttgl.constexpr = acc_tmem.get_reg_layout()
    if acc is not None:
        acc_temp = ttgl.convert_layout(acc, tmem_reg_layout)
    else:
        acc_temp = ttgl.zeros([M, N], out_dtype, layout=tmem_reg_layout)
    acc_tmem.store(acc_temp)
    fence_async_shared()

    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    scale_layout: ttgl.constexpr = TensorMemoryScalesLayout()
    a_scale_tmem = allocate_tensor_memory(lhs_scale.dtype, lhs_scale.shape, scale_layout)
    b_scale_tmem = allocate_tensor_memory(rhs_scale.dtype, rhs_scale.shape, scale_layout)
    scale_layout_reg_lhs: ttgl.constexpr = a_scale_tmem.get_reg_layout()
    scale_layout_reg_rhs: ttgl.constexpr = b_scale_tmem.get_reg_layout()
    lhs_scale = ttgl.convert_layout(lhs_scale, scale_layout_reg_lhs)
    rhs_scale = ttgl.convert_layout(rhs_scale, scale_layout_reg_rhs)
    a_scale_tmem.store(lhs_scale)
    b_scale_tmem.store(rhs_scale)

    tcgen05_mma_scaled(a_smem, b_smem, acc_tmem, a_scale_tmem, b_scale_tmem, lhs_format, rhs_format, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    out = acc_tmem.load()
    ret_layout: ttgl.constexpr = default_blocked_layout([M, N], ttgl.num_warps())
    out = ttgl.convert_layout(out, ret_layout)
    return out


@gluon.jit
def tl_dot_decomposed_block_scales(
    lhs,
    lhs_scale,
    lhs_format,
    rhs,
    rhs_scale,
    rhs_format,
    acc=None,
    fast_math=False,
    lhs_k_pack=True,
    rhs_k_pack=True,
    out_dtype=ttgl.float32,
):
    return tl_dot_decomposed_block_scales_impl(
        tl_dot_scaled,
        tl_dot,
        lhs,
        lhs_scale,
        lhs_format,
        rhs,
        rhs_scale,
        rhs_format,
        acc=acc,
        fast_math=fast_math,
        lhs_k_pack=lhs_k_pack,
        rhs_k_pack=rhs_k_pack,
        out_dtype=out_dtype,
    )


@gluon.jit
def tl_dot_scaled(
    lhs,
    lhs_scale,
    lhs_format,
    rhs,
    rhs_scale,
    rhs_format,
    acc=None,
    fast_math=False,
    lhs_k_pack=True,
    rhs_k_pack=True,
    out_dtype=ttgl.float32,
):
    if (tl_dot_scaled_mmav5_supported(lhs.type, rhs.type, ttgl.num_warps()) and lhs_scale is not None
            and rhs_scale is not None):
        return tl_dot_scaled_blackwell(
            lhs,
            lhs_scale,
            lhs_format,
            rhs,
            rhs_scale,
            rhs_format,
            acc,
            fast_math,
            lhs_k_pack,
            rhs_k_pack,
            out_dtype,
        )
    else:
        return tl_dot_decomposed_block_scales(
            lhs,
            lhs_scale,
            lhs_format,
            rhs,
            rhs_scale,
            rhs_format,
            acc,
            fast_math,
            lhs_k_pack,
            rhs_k_pack,
            out_dtype,
        )


# ---- NVIDIA TMA tensor descriptors ----


@gluon.jit
def tl_gather_tensor_descriptor(desc, x_offsets, y_offset):
    desc_shape: ttgl.constexpr = [x_offsets.shape[0], desc.block_shape[1]]
    alloc = ttgl.allocate_shared_memory(desc.dtype, desc_shape, desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    x_offsets_layout: ttgl.constexpr = ttgl.SliceLayout(
        0,
        ttgl.BlockedLayout([1, 4], [get_num_threads_per_warp(), 1], [1, ttgl.num_warps()], [1, 0]),
    )
    x_offsets = ttgl.convert_layout(x_offsets, x_offsets_layout)
    mbarrier.expect(bar, x_offsets.shape[0] * desc.block_type.nbytes)
    tma_blackwell.async_gather(desc, x_offsets, y_offset, bar, alloc)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    ret_layout: ttgl.constexpr = default_blocked_layout(desc.block_shape, ttgl.num_warps())
    out = alloc.load(ret_layout)
    return out


@gluon.jit
def tl_scatter_tensor_descriptor(desc, value, x_offsets, y_offset):
    desc_shape: ttgl.constexpr = [x_offsets.shape[0], desc.block_shape[1]]
    alloc = ttgl.allocate_shared_memory(desc.dtype, desc_shape, desc.layout, value)
    fence_async_shared()
    x_offsets_layout: ttgl.constexpr = ttgl.SliceLayout(
        0,
        ttgl.BlockedLayout([1, 4], [get_num_threads_per_warp(), 1], [1, ttgl.num_warps()], [1, 0]),
    )
    x_offsets = ttgl.convert_layout(x_offsets, x_offsets_layout)
    tma_blackwell.async_scatter(desc, x_offsets, y_offset, alloc)
    tma.store_wait(0, read_only=False)


# ---- NVIDIA obj dispatch ----


@gluon.jit
def tl_obj_gather(obj, x_offsets, y_offset):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        return tl_gather_tensor_descriptor(obj, x_offsets, y_offset)
    else:
        return obj.gather(x_offsets, y_offset)


@gluon.jit
def tl_obj_scatter(obj, value, x_offsets, y_offset):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        return tl_scatter_tensor_descriptor(obj, value, x_offsets, y_offset)
    else:
        return obj.scatter(value, x_offsets, y_offset)
<|endoftext|># type: ignore

from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma

from triton.tools.triton_to_gluon_translator.common_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.common_helpers import (
    default_blocked_layout, )

# ---- NVIDIA MMA sync (Ampere) ----


@gluon.constexpr_function
def tl_dot_mma_sync_layout(shape, num_warps):
    rank = len(shape)
    assert rank in [
        2,
        3,
    ], "MMA sync only supports 2D shapes or 3D shapes with a batch outer dimension"
    if rank == 2:
        return ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[num_warps, 1], instr_shape=[16, 8])
    return ttgl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[num_warps, 1, 1], instr_shape=[1, 16, 8])


@gluon.constexpr_function
def tl_dot_mma_sync_k_width(a_ty, b_ty):
    a_bitwidth = a_ty.element_ty.primitive_bitwidth
    b_bitwidth = b_ty.element_ty.primitive_bitwidth
    min_bitwidth = min(a_bitwidth, b_bitwidth)
    return max(32 // min_bitwidth, 1)


@gluon.jit
def tl_dot_mma_sync(a, b, acc_init=None, input_precision=None, out_dtype=ttgl.float32):
    mma_layout: ttgl.constexpr = tl_dot_mma_sync_layout(a.type.shape, ttgl.num_warps())
    k_width: ttgl.constexpr = tl_dot_mma_sync_k_width(a.type, b.type)
    a_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=mma_layout, operand_index=0, k_width=k_width)
    b_layout: ttgl.constexpr = ttgl.DotOperandLayout(parent=mma_layout, operand_index=1, k_width=k_width)
    a = ttgl.convert_layout(a, a_layout)
    b = ttgl.convert_layout(b, b_layout)
    if acc_init is not None:
        acc = ttgl.convert_layout(acc_init, mma_layout)
    else:
        acc = ttgl.full([a.shape[0], a.shape[1], b.shape[2]], 0.0, out_dtype, layout=mma_layout)
    result = mma_v2(a, b, acc, input_precision)
    if acc_init is not None:
        layout: ttgl.constexpr = acc_init.type.layout
    else:
        layout: ttgl.constexpr = default_blocked_layout(result.type.shape, ttgl.num_warps())
    result = ttgl.convert_layout(result, layout)
    return result


@gluon.constexpr_function
def get_shared_memory_mma_layout(type, operand_index, allow_transpose, is_fp4_padded=False, force_transpose=False):
    if not allow_transpose:
        if operand_index == 1:
            transposed = True
        else:
            transposed = False
        if force_transpose:
            transposed = not transposed
    else:
        transposed = operand_index == 1

    shape = type.shape
    swizzle_byte_width = 0
    ele_bit_width = type.element_ty.primitive_bitwidth
    packing_factor = 2 if is_fp4_padded else 1

    contig_dim_size_in_byte = ((shape[0] if transposed else shape[1]) * packing_factor * ele_bit_width // 8)
    if contig_dim_size_in_byte >= 128 and contig_dim_size_in_byte % 128 == 0:
        swizzle_byte_width = 128
    elif contig_dim_size_in_byte >= 64 and contig_dim_size_in_byte % 64 == 0:
        swizzle_byte_width = 64
    elif contig_dim_size_in_byte >= 32 and contig_dim_size_in_byte % 32 == 0:
        swizzle_byte_width = 32
    else:
        swizzle_byte_width = 0

    flatten_outer_dim = 1
    for dim in shape:
        flatten_outer_dim *= dim
    if len(shape) < 2 or flatten_outer_dim < 8:
        swizzle_byte_width = 0
    return ttgl.NVMMASharedLayout(
        swizzle_byte_width=swizzle_byte_width,
        transposed=transposed,
        element_bitwidth=ele_bit_width,
        rank=len(shape),
        fp4_padded=is_fp4_padded,
    )


@gluon.jit
def get_shared_memory_mma_operand(value, operand_index, allow_transpose, is_fp4_padded=False, force_transpose=False):
    layout: ttgl.constexpr = get_shared_memory_mma_layout(value.type, operand_index, allow_transpose, is_fp4_padded,
                                                          force_transpose)
    return ttgl.allocate_shared_memory(value.dtype, value.shape, layout, value)


# ---- NVIDIA TMA tensor descriptors ----


@gluon.jit
def tl_make_tensor_descriptor(base, shape, strides, block_shape, padding_option: ttgl.constexpr = "zero"):
    layout: ttgl.constexpr = ttgl.NVMMASharedLayout.get_default_for(block_shape, base.dtype.element_ty)
    return tma.make_tensor_descriptor(base, shape, strides, block_shape, layout, padding_option)


@gluon.jit
def tl_store_tensor_descriptor(desc, offsets, value):
    alloc = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout, value)
    fence_async_shared()
    tma.async_copy_shared_to_global(desc, offsets, alloc)
    tma.store_wait(0, read_only=False)
    alloc._keep_alive()


@gluon.jit
def tl_load_tensor_descriptor(desc, offsets):
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    bar = ttgl.allocate_shared_memory(ttgl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, offsets, bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    ret_layout: ttgl.constexpr = default_blocked_layout(desc.block_shape, ttgl.num_warps())
    out = smem.load(ret_layout)
    return out


# ---- NVIDIA obj dispatch ----


@gluon.jit
def tl_obj_store(obj, offsets, value):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        return tl_store_tensor_descriptor(obj, offsets, value)
    else:
        return obj.store(offsets, value)


@gluon.jit
def tl_obj_load(obj, offsets):
    if isinstance(obj, ttgl.nvidia.hopper.tma.tensor_descriptor):
        return tl_load_tensor_descriptor(obj, offsets)
    else:
        return obj.load(offsets)


# ---- NVIDIA host-side descriptor ----


def convert_host_descriptor(desc):

    def torch_dtype_to_triton(dtype):
        import torch

        if dtype == torch.float8_e5m2:
            return ttgl.float8e5
        if dtype == torch.float8_e4m3fn:
            return ttgl.float8e4nv
        return getattr(ttgl, str(dtype).split(".")[1])

    from triton.tools.tensor_descriptor import TensorDescriptor

    assert isinstance(desc, TensorDescriptor)
    block_shape = desc.block_shape
    dtype = desc.base.dtype
    tensor = desc.base

    layout = ttgl.NVMMASharedLayout.get_default_for(block_shape, torch_dtype_to_triton(dtype))
    return gluon.nvidia.hopper.TensorDescriptor(tensor, desc.shape, desc.strides, block_shape, layout)
<|endoftext|># type: ignore

import math

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.amd.gfx1250 import wmma as amd_wmma
from triton.experimental.gluon.language.amd.gfx1250 import tdm as amd_tdm
from triton.experimental.gluon.language.amd.cdna3 import mfma as amd_mfma
from triton.language.target_info import current_target

from triton.tools.triton_to_gluon_translator.common_helpers import *  # noqa: F401,F403
from triton.tools.triton_to_gluon_translator.common_helpers import (
    default_blocked_layout,
    get_num_threads_per_warp,
    tl_dot_decomposed_block_scales_impl,
)

# ---- architecture detection ----


@gluon.constexpr_function
def _is_gfx1250(target=None):
    return target is not None and target.arch == "gfx1250"


@gluon.constexpr_function
def _is_cdna(target=None):
    return target is not None and target.arch in ("gfx942", "gfx950")


@gluon.constexpr_function
def _cdna_version(target=None):
    """Returns 3 for gfx942, 4 for gfx950."""
    return 4 if target is not None and target.arch == "gfx950" else 3


# ---- AMD WMMA layout helpers (gfx1250) ----


@gluon.constexpr_function
def compute_warp_bases(num_warps):
    """Distribute warps across M/N: first bit to N, rest to M."""
    n_bits = int(math.log2(num_warps))
    if n_bits == 0:
        return []
    warp_bases = [[0, 1]]
    for i in range(n_bits - 1):
        warp_bases.append([1 << i, 0])
    return warp_bases


@gluon.constexpr_function
def get_wmma_layout(shape, num_warps):
    warp_bases = compute_warp_bases(num_warps)
    return ttgl.amd.AMDWMMALayout(3, True, warp_bases, [], [16, 16, 32])


@gluon.constexpr_function
def get_wmma_k_width(a_ty, b_ty):
    min_bitwidth = min(a_ty.element_ty.primitive_bitwidth, b_ty.element_ty.primitive_bitwidth)
    return max(128 // min_bitwidth, 1)


# ---- AMD MFMA layout helpers (cdna3/cdna4) ----


@gluon.constexpr_function
def get_mfma_instr_k(element_bitwidth, target=None):
    """K dimension of the MFMA instruction for [32, 32, K]."""
    k_bits = 128 if _cdna_version(target) == 3 else 256
    return k_bits // element_bitwidth


@gluon.constexpr_function
def get_mfma_layout(num_warps, element_bitwidth, target=None):
    instr_k = get_mfma_instr_k(element_bitwidth, target)
    return ttgl.amd.AMDMFMALayout(
        version=_cdna_version(target),
        instr_shape=[32, 32, instr_k],
        transposed=True,
        warps_per_cta=[num_warps, 1],
    )


@gluon.constexpr_function
def get_mfma_k_width(a_ty, b_ty, target=None):
    min_bitwidth = min(a_ty.element_ty.primitive_bitwidth, b_ty.element_ty.primitive_bitwidth)
    instr_k = get_mfma_instr_k(min_bitwidth, target)
    return instr_k // 2


# ---- AMD dot paths ----


@gluon.jit
def tl_dot_wmma(a, b, acc, out_dtype):
    """gfx1250 WMMA path."""
    M: ttgl.constexpr = a.type.shape[0]
    N: ttgl.constexpr = b.type.shape[1]
    num_warps: ttgl.constexpr = ttgl.num_warps()

    wmma_layout: ttgl.constexpr = get_wmma_layout([M, N], num_warps)
    k_width: ttgl.constexpr = get_wmma_k_width(a.type, b.type)
    a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=k_width)
    b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=k_width)

    a = ttgl.convert_layout(a, a_layout)
    b = ttgl.convert_layout(b, b_layout)

    if acc is not None:
        accumulator = ttgl.convert_layout(acc, wmma_layout)
    else:
        accumulator = ttgl.zeros([M, N], out_dtype, layout=wmma_layout)

    result = amd_wmma(a, b, accumulator)

    if acc is not None:
        ret_layout: ttgl.constexpr = acc.type.layout
    else:
        ret_layout: ttgl.constexpr = default_blocked_layout(result.type.shape, num_warps)
    return ttgl.convert_layout(result, ret_layout)


@gluon.jit
def tl_dot_mfma(a, b, acc, out_dtype):
    """CDNA3/CDNA4 MFMA path."""
    M: ttgl.constexpr = a.type.shape[0]
    N: ttgl.constexpr = b.type.shape[1]
    num_warps: ttgl.constexpr = ttgl.num_warps()
    min_bitwidth: ttgl.constexpr = min(a.type.element_ty.primitive_bitwidth, b.type.element_ty.primitive_bitwidth)
    target: ttgl.constexpr = current_target()

    mfma_layout: ttgl.constexpr = get_mfma_layout(num_warps, min_bitwidth, target)
    k_width: ttgl.constexpr = get_mfma_k_width(a.type, b.type, target)
    a_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=k_width)
    b_layout: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=k_width)

    a = ttgl.convert_layout(a, a_layout)
    b = ttgl.convert_layout(b, b_layout)

    if acc is not None:
        accumulator = ttgl.convert_layout(acc, mfma_layout)
    else:
        accumulator = ttgl.zeros([M, N], out_dtype, layout=mfma_layout)

    result = amd_mfma(a, b, accumulator)

    if acc is not None:
        ret_layout: ttgl.constexpr = acc.type.layout
    else:
        ret_layout: ttgl.constexpr = default_blocked_layout(result.type.shape, num_warps)
    return ttgl.convert_layout(result, ret_layout)


# ---- AMD dot dispatch ----


@gluon.jit
def tl_dot(
    a,
    b,
    acc=None,
    input_precision=None,
    allow_tf32=None,
    max_num_imprecise_acc=None,
    out_dtype=ttgl.float32,
):
    target: ttgl.constexpr = current_target()
    if _is_gfx1250(target):
        return tl_dot_wmma(a, b, acc, out_dtype)
    elif _is_cdna(target):
        return tl_dot_mfma(a, b, acc, out_dtype)


@gluon.jit
def tl_dot_scaled(
    lhs,
    lhs_scale,
    lhs_format,
    rhs,
    rhs_scale,
    rhs_format,
    acc=None,
    fast_math=False,
    lhs_k_pack=True,
    rhs_k_pack=True,
    out_dtype=ttgl.float32,
):
    return tl_dot_decomposed_block_scales_impl(
        tl_dot_scaled,
        tl_dot,
        lhs,
        lhs_scale,
        lhs_format,
        rhs,
        rhs_scale,
        rhs_format,
        acc,
        fast_math,
        lhs_k_pack,
        rhs_k_pack,
        out_dtype,
    )


# ---- AMD TDM tensor descriptors (gfx1250 only) ----


@gluon.constexpr_function
def get_default_tdm_layout(*block_shape):
    block_shape = list(block_shape)
    return ttgl.PaddedSharedLayout.with_identity_for(
        [[block_shape[-1], 4]],
        block_shape,
        list(range(len(block_shape) - 1, -1, -1)),
    )


@tl.core._aggregate
class AMDTensorDescriptorArgs:
    """Wraps a real TDM descriptor alongside the original base pointer.

    The base_ptr is needed by gather/scatter to recreate the descriptor with a different
    block_shape -- Triton uses block_shape=[1, N] but TDM hardware requires [num_indices, N].
    Shape, strides, and block_shape are read from desc (type metadata gives plain Python ints
    for block_shape, tuples for shape/strides)."""
    desc: amd_tdm.tensor_descriptor
    base_ptr: tl.core.tensor


@gluon.jit
def tl_make_tensor_descriptor(base, shape, strides, block_shape, padding_option: ttgl.constexpr = "zero"):
    ttgl.static_assert(_is_gfx1250(current_target()), "tl_make_tensor_descriptor requires gfx1250 target")
    layout: ttgl.constexpr = get_default_tdm_layout(*block_shape)
    desc = amd_tdm.make_tensor_descriptor(base, shape, strides, block_shape, layout)
    return AMDTensorDescriptorArgs(desc, base)


# ---- AMD obj dispatch ----


@gluon.jit
def tl_obj_load_amd(desc, offsets):
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    amd_tdm.async_load(desc, offsets, smem)
    amd_tdm.async_wait(0)
    ret_layout: ttgl.constexpr = default_blocked_layout(desc.block_shape, ttgl.num_warps())
    return smem.load(ret_layout)


@gluon.jit
def tl_obj_store_amd(desc, offsets, value):
    smem = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout, value)
    amd_tdm.async_store(desc, offsets, smem)
    amd_tdm.async_wait(0)


@gluon.jit
def tl_obj_store(obj, offsets, value):
    if isinstance(obj, AMDTensorDescriptorArgs):
        tl_obj_store_amd(obj.desc, offsets, value)
    elif isinstance(obj, amd_tdm.tensor_descriptor):
        tl_obj_store_amd(obj, offsets, value)
    else:
        return obj.store(offsets, value)


@gluon.jit
def tl_obj_load(obj, offsets):
    if isinstance(obj, AMDTensorDescriptorArgs):
