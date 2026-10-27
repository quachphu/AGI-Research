in flight alongside an in-progress bulk-reduce. An epilogue drains
the final outstanding load.

Chunks along D at a compile-time ``chunk_elems``: small rows travel as
a single chunk, large rows chunk at 512 B. The TMA descriptor OOB-clamp
handles rows whose length isn't a multiple of ``chunk_elems``.

Grid layout is 2D to maintain SM utilization for shapes with small N
and large D: ``(grid_x, grid_y)``, where ``grid_y`` partitions the
chunk-index axis. When N is large the host sets ``grid_y = 1`` so the
inner chunk loop recovers the classic schedule. When N is tiny (say
100 rows, D=640K), ``grid_y`` grows so the grid still fills the
machine.

Restrictions (enforced by the host cond in ``cutedsl_impl.py``):
  - sm_90+ (cp.reduce.async.bulk availability)
  - dim == 0, rank >= 2, self/src inner-contiguous
  - index is the expanded-1D pattern (same shape as src, stride 0 on every
    axis except 0)
  - dtype in {fp32, fp16, bf16}
  - ``row_bytes % 16 == 0`` (for both the TMA load operand and the
    cp.reduce.async.bulk gmem operand)
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.cute.testing as cute_testing
import cutlass.pipeline as pipeline
from cutlass import BFloat16, Float16, Float32, Int32, Int64

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache

from ._ptx import cvta_smem, make_bulk_reduce_add


_MAX_CHUNK_BYTES = 512
_THREADS_PER_CTA = 32  # one warp per CTA
# cp.async.bulk requires 128B-aligned smem destinations.
_SMEM_ALIGN_BYTES = 128


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


_TORCH_TO_CUTE = {
    torch.float32: Float32,
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
}


_bulk_reduce_add_f32 = make_bulk_reduce_add("f32")
_bulk_reduce_add_f16 = make_bulk_reduce_add("noftz.f16")
_bulk_reduce_add_bf16 = make_bulk_reduce_add("noftz.bf16")


def _reduce_op_for(dtype):
    if dtype is Float32:
        return _bulk_reduce_add_f32
    if dtype is Float16:
        return _bulk_reduce_add_f16
    if dtype is BFloat16:
        return _bulk_reduce_add_bf16
    raise ValueError(f"unsupported dtype: {dtype}")


def _make_kernel(dtype, elem_bytes: int, chunk_elems: int, reduce_op):
    """Build a dtype-specialized kernel closure. dtype/elem_bytes
    /chunk_elems are Python-time constants that the preprocessor folds
    at cute.compile time; ``N`` and ``num_chunks`` are runtime args so
    one compile serves every row length.

    ``chunk_elems`` is baked in at compile time (it is the static TMA
    box dim). The TMA descriptor built in ``_launch`` (via
    ``make_tiled_tma_atom`` on the source tensor) enables
    OOB-clamp-to-zero for reads past column ``N``; the reduce side then
    writes only the actual valid byte count, so partial final chunks --
    and rows shorter than a full box -- are handled natively.

    One CTA = one warp. The single driver thread (tidx == 0) serves as
    both producer (issues the TMA load) and consumer (issues the
    bulk-reduce). Both CooperativeGroups are ``Agent.Thread, size=1``
    so the mbarrier arrive counts match the single-threaded flow.

    The reduce side always uses the runtime ``out_row_stride`` arg so
    outer-strided outputs (e.g. slices) work; the contiguous case just
    passes ``out.stride(0) == N`` in as that value.
    """

    chunk_bytes = chunk_elems * elem_bytes

    # 2-stage pipeline buffer is laid out column-major; stage i starts
    # at offset i * stage_stride_elems * elem_bytes, so the stride must
    # round chunk_bytes up to a multiple of _SMEM_ALIGN_BYTES. Otherwise
    # stage 1 is misaligned and the kernel faults the first time a CTA
    # writes it. Bites both chunk_bytes < 128 (small D) and chunk_bytes
    # not a multiple of 128 (e.g. fp32 N=36 -> 144 B).
    stage_stride_elems = _round_up(chunk_elems, _SMEM_ALIGN_BYTES // elem_bytes)

    @cute.kernel
    def _kernel(
        tma_atom: cute.CopyAtom,
        tma_tensor_src: cute.Tensor,  # TMA-view of mSrc
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        N: Int32,
        num_chunks: Int32,
        chunks_per_cta: Int32,
        out_row_stride: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdim_x, _, _ = cute.arch.grid_dim()

        # num_entries (M_src) comes from mIndex's shape; mIndex is 1D
        # of length M_src after host-side flattening.
        num_entries = mIndex.shape[0]

        # Chunk-index range assigned to this CTA. bidy partitions the
        # chunk axis into disjoint slices of chunks_per_cta; when
        # chunks_per_cta == num_chunks, grid_y == 1 and every CTA sees
        # the whole D.
        chunk_start = bidy * chunks_per_cta
        chunk_end = chunk_start + chunks_per_cta
        if chunk_end > num_chunks:
            chunk_end = num_chunks

        out_rs = out_row_stride

        smem = cutlass.utils.SmemAllocator()
        sBuf = smem.allocate_tensor(
            dtype,
            cute.make_layout((chunk_elems, 2), stride=(1, stage_stride_elems)),
            _SMEM_ALIGN_BYTES,
        )
        mbar_storage = smem.allocate_array(cutlass.Uint64, num_elems=2 * 2)

        pipe = pipeline.PipelineTmaAsync.create(
            num_stages=2,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
            tx_count=chunk_bytes,
            barrier_storage=mbar_storage,
            tidx=tidx,
        )

        # Tile the source (M, N) by (1, chunk_elems) to get
        # (1, chunk_elems, M, num_chunks); tma_partition collapses and
        # returns tma_gmem[None, row_idx, chunk_idx] and tma_smem[None,
        # stage].
        tiled_gmem = cute.local_tile(tma_tensor_src, (1, chunk_elems), (None, None))
        tma_smem, tma_gmem = cpasync.tma_partition(
            tma_atom,
            Int32(0),
            cute.make_layout(1),
            cute.group_modes(sBuf, 0, 1),
            cute.group_modes(tiled_gmem, 0, 2),
        )

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, 2
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 2
        )

        # Software-pipelined schedule: at each iteration, issue the TMA
        # load for the current (entry, chunk) pair, then consume the
        # previous pair (bulk-reduce). With num_stages=2 this keeps one
        # TMA load in flight while a bulk-reduce is running. Final
        # iteration's load is drained in an epilogue after the main loop.
        # pair_count is a runtime Int32 so the ``pair_count > 0`` branch
        # stays in the compiled IR (a Python bool would be baked in at
        # trace time).
        pair_count = Int32(0)
        prev_chunk_idx = Int32(0)
        prev_r = Int64(0)

        base = bidx
        while base < num_entries:
            entry_id = base

            chunk_idx = chunk_start
            while chunk_idx < chunk_end:
                if tidx == Int32(0):
                    r = Int64(mIndex[entry_id])
                    # Bounds check: index values must be valid output
                    # rows. Compiling with ``--enable-assertions`` turns
                    # this into a device-side trap; otherwise it folds
                    # away. Driver thread only -- no need to replicate
                    # the check across all 32 lanes.
                    cute_testing.assert_(r >= Int64(0))
                    cute_testing.assert_(r < Int64(mOut.shape[0]))

                    pipe.producer_acquire(producer_state)
                    cute.copy(
                        tma_atom,
                        tma_gmem[None, entry_id, chunk_idx],
                        tma_smem[None, producer_state.index],
                        tma_bar_ptr=pipe.producer_get_barrier(producer_state),
                    )
                    pipe.producer_commit(producer_state)
                    producer_state.advance()

                    if pair_count > Int32(0):
                        pipe.consumer_wait(consumer_state)
                        cbuf_ptr = sBuf[None, consumer_state.index].iterator

                        # Partial-chunk handling: actual valid element
                        # count is min(chunk_elems, N - off). TMA
                        # OOB-clamped the tail to 0 in smem, we reduce
                        # only the valid bytes.
                        off = prev_chunk_idx * Int32(chunk_elems)
                        cur_elems = N - off
                        if cur_elems > Int32(chunk_elems):
                            cur_elems = Int32(chunk_elems)
                        cur_bytes = cur_elems * Int32(elem_bytes)

                        dst_off = prev_r * out_rs + Int64(off)
                        gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                        reduce_op(gmem_dst_u64, cvta_smem(cbuf_ptr), cur_bytes)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                        pipe.consumer_release(consumer_state)
                        consumer_state.advance()

                    prev_chunk_idx = chunk_idx
                    prev_r = r
                    pair_count = pair_count + Int32(1)

                chunk_idx = chunk_idx + Int32(1)

            base = base + gdim_x

        # Epilogue: drain the last outstanding TMA load.
        if tidx == Int32(0):
            if pair_count > Int32(0):
                pipe.consumer_wait(consumer_state)
                cbuf_ptr = sBuf[None, consumer_state.index].iterator

                off = prev_chunk_idx * Int32(chunk_elems)
                cur_elems = N - off
                if cur_elems > Int32(chunk_elems):
                    cur_elems = Int32(chunk_elems)
                cur_bytes = cur_elems * Int32(elem_bytes)

                dst_off = prev_r * out_rs + Int64(off)
                gmem_dst_u64 = Int64((mOut.iterator + dst_off).toint())
                reduce_op(gmem_dst_u64, cvta_smem(cbuf_ptr), cur_bytes)
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=False)
                pipe.consumer_release(consumer_state)

    @cute.jit
    def _launch(
        mSrc: cute.Tensor,
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        N: Int32,
        num_chunks: Int32,
        chunks_per_cta: Int32,
        grid_x: Int32,
        grid_y: Int32,
        out_row_stride: Int64,
    ):
        # Build the tile-mode TMA descriptor over the (M_src, N) source.
        # Both global dims are dynamic; the ``(1, chunk_elems)`` box is
        # static. TMA clamps OOB column reads to 0, so rows shorter than
        # a full box and rows with ``N % chunk_elems != 0`` are handled
        # natively on the load.
        tma_atom, tma_tensor_src = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mSrc,
            cute.make_layout((1, chunk_elems)),
            (1, chunk_elems),
        )
        _kernel(
            tma_atom,
            tma_tensor_src,
            mIndex,
            mOut,
            N,
            num_chunks,
            chunks_per_cta,
            out_row_stride,
        ).launch(
            grid=[grid_x, grid_y, 1],
            block=[_THREADS_PER_CTA, 1, 1],
            stream=stream,
        )

    return _launch


def _chunk_elems_for(torch_dtype: torch.dtype) -> int:
    """Compile-time ``chunk_elems`` (the static TMA box dim): always
    ``_MAX_CHUNK_BYTES // elem_bytes``. Since ``N`` is now a runtime arg
    the box can't shrink to fit small rows -- instead a short row loads
    one box and the TMA descriptor OOB-clamps the tail to 0, and the
    reduce writes only ``min(chunk_elems, N)`` valid elements. 512 B is
    a multiple of 16 for every supported dtype, so the box always meets
    the reduce's 16-byte alignment."""
    return _MAX_CHUNK_BYTES // torch_dtype.itemsize


@instrumented_cutedsl_cache(
    "aten::scatter_add",
    key_fn=lambda torch_dtype: f"tma {torch_dtype}",
)
def _compile_tma_scatter(torch_dtype: torch.dtype):
    # N and num_chunks are runtime args, so one compile per dtype serves
    # every row length. chunk_elems (the static TMA box) depends only on
    # the dtype's element size.
    dtype = _TORCH_TO_CUTE[torch_dtype]
    elem_bytes = dtype.width // 8
    chunk_elems = _chunk_elems_for(torch_dtype)
    reduce_op = _reduce_op_for(dtype)
    launcher = _make_kernel(dtype, elem_bytes, chunk_elems, reduce_op)

    mSrc_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    # Index is guaranteed contiguous by _flatten_for_expanded_1d; fix
    # stride=1 so `mIndex[i]` doesn't emit a runtime stride multiply.
    mIndex_fake = cute.runtime.make_fake_tensor(Int64, (cute.sym_int(),), stride=(1,))
    mOut_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    return cute.compile(
        launcher,
        mSrc_fake,
        mIndex_fake,
        mOut_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),  # N
        Int32(0),  # num_chunks
        Int32(0),  # chunks_per_cta
        Int32(0),  # grid_x
        Int32(0),  # grid_y
        Int64(0),  # out_row_stride
        # ``--enable-assertions`` keeps the ``cute_testing.assert_``
        # bounds checks on ``r`` live in production. Cost is roughly
        # +1-10% (geomean +7.7%) on most shapes; the safety net is
        # worth it because an OOB ``r`` would otherwise silently
        # corrupt unrelated gmem via ``cp.reduce.async.bulk``.
        options="--enable-tvm-ffi --enable-assertions",
    )


def min_d_divisor_for(dtype: torch.dtype) -> int:
    """Smallest value D must be divisible by so ``cp.reduce.async.bulk``
    gmem operands stay 16-byte aligned: ``D * sizeof(dtype) % 16 == 0``,
    i.e. ``D % (16 / sizeof(dtype)) == 0``. fp32: %4, bf16/fp16: %8.
    Supported dtypes' itemsize divides 16, so plain // suffices.
    """
    return 16 // dtype.itemsize


def row_shape_supported(dtype: torch.dtype, N: int) -> bool:
    """Host-side check: can the TMA kernel handle an N-element row?

    Only requirement is that ``row_bytes = N * elem_size`` is a multiple
    of 16, which is the PTX operand alignment for both the tile-mode
    TMA load (per-row stride must be 16-aligned) and
    ``cp.reduce.async.bulk`` (gmem address + byte count must be
    16-aligned). Rows that aren't multiples of ``chunk_elems`` are
    handled via the TMA descriptor's OOB-clamp-to-zero behavior.
    """
    return (N * dtype.itemsize) % 16 == 0


def _plan_grid(M: int, D: int, chunk_elems: int, sm: int) -> tuple[int, int, int]:
    """Pick ``(grid_x, grid_y, chunks_per_cta)``.

    Strategy: keep the classic 1D schedule (grid_y=1, whole chunk range
    per CTA with internal double-buffering) whenever the row-axis alone
    saturates the GPU. When M is too small for that, split the
    chunk-axis across grid_y so every SM gets work.
    """
    n_chunks = (D + chunk_elems - 1) // chunk_elems
    # 1 warp per CTA: need many more CTAs than an 8-warp layout to keep
    # occupancy up. sm*32 target with a sm*64 clamp works well across
    # uniform / high_cont / few_idx on B200.
    row_ctas = M
    target_ctas = sm * 32
    if row_ctas >= target_ctas:
        grid_x = min(row_ctas, sm * 64)
        return grid_x, 1, n_chunks
    # Split the chunk axis until we hit the target.
    want_y = max(1, target_ctas // max(row_ctas, 1))
    grid_y = min(n_chunks, want_y)
    chunks_per_cta = (n_chunks + grid_y - 1) // grid_y
    # Recompute grid_y now that each y-slot holds chunks_per_cta chunks.
    grid_y = (n_chunks + chunks_per_cta - 1) // chunks_per_cta
    grid_x = row_ctas
    return grid_x, grid_y, chunks_per_cta


def tma_scatter_add_into(
    out: torch.Tensor,
    index_1d: torch.Tensor,
    src: torch.Tensor,
) -> None:
    """In-place: ``out[index_1d[i], :] += src[i, :]`` for every i.

    ``out`` / ``src`` are 2D with inner-dim stride 1 (outer row stride
    can differ from N, e.g. a slice of a wider buffer). ``index_1d`` is
    1D int64 of length M_src. ``row_bytes = N * elem_size`` must be a
    multiple of 16; the host cond enforces this.
    """
    M, N = src.shape
    chunk_elems = _chunk_elems_for(src.dtype)
    compiled = _compile_tma_scatter(src.dtype)
    sm = torch.cuda.get_device_properties(out.device).multi_processor_count

    grid_x, grid_y, chunks_per_cta = _plan_grid(M, N, chunk_elems, sm)
    num_chunks = (N + chunk_elems - 1) // chunk_elems
    compiled(
        src,
        index_1d,
        out,
        N,
        num_chunks,
        chunks_per_cta,
        grid_x,
        grid_y,
        out.stride(0),
    )
<|endoftext|>"""Inline-PTX helpers shared across the TMA and vec-scatter kernels.

Each helper wraps ``llvm.inline_asm`` with the options we always use
(AT&T dialect, side-effecting) and normalizes argument types (cute's
``Numeric`` wrappers get converted to ``ir.Value`` automatically). This
keeps the kernel files focused on algorithm instead of MLIR plumbing.
"""

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op, T


def as_ir(v, *, loc=None, ip=None):
    """Normalize ``v`` to ``ir.Value``: cute Numeric wrappers (Int32,
    Float16, etc.) get unwrapped via ``.ir_value()``; other values pass
    through unchanged (they're already ir.Values)."""
    if hasattr(v, "ir_value"):
        return v.ir_value(loc=loc, ip=ip)
    return v


def _inline_asm(result_type, operands, asm, constraint, *, loc=None, ip=None):
    """``llvm.inline_asm`` with the flags we always use."""
    return llvm.inline_asm(
        result_type,
        operands,
        asm,
        constraint,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cvta_smem(ptr, *, loc=None, ip=None):
    """Shared-memory ptr -> its ``.shared::cta`` state-space u64 address."""
    raw = ptr.toint(loc=loc, ip=ip).ir_value()
    # cvta is pure, not side-effecting; use inline_asm directly.
    return llvm.inline_asm(
        T.i64(),
        [raw],
        "cvta.to.shared::cta.u64 $0, $1;",
        "=l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def make_bulk_reduce_add(ptx_suffix: str):
    """Emit ``cp.reduce.async.bulk.global.shared::cta.bulk_group.add.<suffix>``.
    Suffix selects the dtype: ``f32`` / ``noftz.f16`` / ``noftz.bf16``."""

    @dsl_user_op
    def _op(gmem_addr_u64, smem_u64, size_bytes, *, loc=None, ip=None):
        _inline_asm(
            None,
            [
                as_ir(gmem_addr_u64, loc=loc, ip=ip),
                as_ir(smem_u64, loc=loc, ip=ip),
                as_ir(size_bytes, loc=loc, ip=ip),
            ],
            f"cp.reduce.async.bulk.global.shared::cta.bulk_group.add.{ptx_suffix}"
            " [$0], [$1], $2;",
            "l,l,r",
            loc=loc,
            ip=ip,
        )

    return _op


def make_packed_half_atomic_add(ptx_suffix: str):
    """Emit ``red.global.add.noftz.<suffix>`` with a single ``i32`` packed
    register holding two halves. Used by the vec-scatter kernel for
    ``f16x2`` / ``bf16x2`` paired atomics."""

    @dsl_user_op
    def _op(gmem_ptr_i64, packed_i32, *, loc=None, ip=None):
        _inline_asm(
            None,
            [as_ir(gmem_ptr_i64, loc=loc, ip=ip), as_ir(packed_i32, loc=loc, ip=ip)],
            f"red.global.add.noftz.{ptx_suffix} [$0], $1;",
            "l,r",
            loc=loc,
            ip=ip,
        )

    return _op
<|endoftext|>"""Vectorized scatter_add with per-element atomics (pre-sm_90 fallback).

Same input restriction as the TMA path (expanded-1D index, dim=0, 2D
contiguous tensors), but works everywhere scalar atomicAdd does. Matches
the fallback path in https://github.com/pytorch/pytorch/pull/182675.

Algorithm: warp-per-source-row. Each warp reads ``idx[entry]`` once, then
the 32 lanes chunk through N with vectorized gather (4 fp32 or 8 halves
per lane per step) and issue atomics into ``out[ind, :]``:

- fp32: per-element ``atomicAdd``.
- fp16/bf16: paired x2 atomics
  (``red.global.add.noftz.{f16x2,bf16x2}``), one instruction per pair.
  Matches the instruction count aten gets from ``atomicAdd(__half2*, ...)``.
"""

import cuda.bindings.driver as cuda  # pyrefly: ignore[missing-import]
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as cute_testing
from cutlass import BFloat16, const_expr, Float16, Float32, Int32, Int64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector as mlir_vector
from cutlass.cutlass_dsl import dsl_user_op, T

import torch
from torch._native.instrumentation import instrumented_cutedsl_cache

from ._ptx import make_packed_half_atomic_add


_WARPS_PER_BLOCK = 8
_THREADS_PER_BLOCK = 32 * _WARPS_PER_BLOCK
_VEC_BYTES = 16  # 16-byte vector gather = LDG.128

_TORCH_TO_CUTE = {
    torch.float32: Float32,
    torch.float16: Float16,
    torch.bfloat16: BFloat16,
}


def _pack_half2_as_i32(val_a, val_b, elem_mlir_type, *, loc=None, ip=None):
    """Build a <2 x elem> vector from two scalar halves and bitcast to i32.
    Matches the layout PTX red.global.add.noftz.{f16x2,bf16x2} expects."""
    vec_type = ir.VectorType.get([2], elem_mlir_type, loc=loc)
    vec = mlir_vector.from_elements(
        vec_type,
        [val_a.ir_value(loc=loc, ip=ip), val_b.ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )
    return llvm.bitcast(T.i32(), vec, loc=loc, ip=ip)


_packed_atomic_add_f16x2 = make_packed_half_atomic_add("f16x2")
_packed_atomic_add_bf16x2 = make_packed_half_atomic_add("bf16x2")


@dsl_user_op
def _atomic_add_f16x2(ptr, val_a: Float16, val_b: Float16, *, loc=None, ip=None):
    """Packed x2 f16 atomic add. Equivalent to atomicAdd(__half2*, ...)."""
    gmem_ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    packed = _pack_half2_as_i32(val_a, val_b, Float16.mlir_type, loc=loc, ip=ip)
    _packed_atomic_add_f16x2(gmem_ptr_i64, packed, loc=loc, ip=ip)


@dsl_user_op
def _atomic_add_bf16x2(ptr, val_a: BFloat16, val_b: BFloat16, *, loc=None, ip=None):
    """Packed x2 bf16 atomic add (sm_90+)."""
    gmem_ptr_i64 = ptr.toint(loc=loc, ip=ip).ir_value()
    packed = _pack_half2_as_i32(val_a, val_b, BFloat16.mlir_type, loc=loc, ip=ip)
    _packed_atomic_add_bf16x2(gmem_ptr_i64, packed, loc=loc, ip=ip)


def _make_kernel(dtype, elem_bytes: int, vec_elems: int, contig: bool):
    """Build a dtype-specialized vectorized scatter-add kernel.

    Each warp handles one source row. 32 lanes chunk through N, each lane
    issuing a single vectorized LDG.128 (via ``cute.autovec_copy``) for
    ``vec_elems`` consecutive elements and atomic-adding them into
    ``out[ind, :]``.

    fp16/bf16 use paired x2 atomics (``red.global.add.noftz.{f16x2,bf16x2}``)
    to match the instruction count aten gets with ``atomicAdd(__half2*, ...)``.
    fp32 uses scalar ``atomicAdd``.
    """
    is_half = dtype is Float16 or dtype is BFloat16

    @cute.kernel
    def _kernel(
        mSrc: cute.Tensor,
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        num_entries: Int32,
        D: Int32,
        src_row_stride: Int64,
        out_row_stride: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()

        warp_in_block = tidx // Int32(32)
        lane = tidx % Int32(32)

        # Contiguous fast path: row stride is D at compile time.
        if const_expr(contig):
            src_row_stride = Int64(D)
            out_row_stride = Int64(D)

        entries_per_block = Int32(_WARPS_PER_BLOCK)
        base = bidx * entries_per_block
        while base < num_entries:
            entry_id = base + warp_in_block
            if entry_id < num_entries:
                r = Int64(mIndex[entry_id])
                # Bounds check: out-of-range ``r`` would make the
                # atomicAdd below corrupt unrelated memory. Compiling
                # with ``--enable-assertions`` turns this into a
                # device-side trap; otherwise it folds away.
                cute_testing.assert_(r >= Int64(0))
                cute_testing.assert_(r < Int64(mOut.shape[0]))

                lane_offset = lane * Int32(vec_elems)
                stride_elems = Int32(32 * vec_elems)

                off = lane_offset
                while off < D:
                    src_off = Int64(entry_id) * src_row_stride + Int64(off)
                    dst_off = r * out_row_stride + Int64(off)

                    # Vectorized gmem -> register load. ``autovec_copy``
                    # picks the widest safe instruction given the static
                    # shape (vec_elems x elem_bytes == 16 bytes, so this
                    # becomes a single LDG.128).
                    gsrc = cute.make_tensor(
                        mSrc.iterator + src_off,
                        cute.make_layout(vec_elems),
                    )
                    rvals = cute.make_rmem_tensor(vec_elems, dtype)
                    cute.autovec_copy(gsrc, rvals)

                    if const_expr(is_half):
                        # Paired atomics: one x2 atomic per 2 elements.
                        # vec_elems is even for halves (8 at 16 bytes),
                        # so vec_elems // 2 is exact.
                        for p in cutlass.range_constexpr(vec_elems // 2):
                            v0 = 2 * p
                            dst_ptr = mOut.iterator + (dst_off + Int64(v0))
                            if const_expr(dtype is Float16):
                                _atomic_add_f16x2(dst_ptr, rvals[v0], rvals[v0 + 1])
                            else:
                                _atomic_add_bf16x2(dst_ptr, rvals[v0], rvals[v0 + 1])
                    else:
                        for v in cutlass.range_constexpr(vec_elems):
                            dst_ptr_v = mOut.iterator + (dst_off + Int64(v))
                            cute.arch.atomic_add(dst_ptr_v, rvals[v])

                    off = off + stride_elems

            base = base + gdim * entries_per_block

    @cute.jit
    def _launch(
        mSrc: cute.Tensor,
        mIndex: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
        num_entries: Int32,
        D: Int32,
        grid_x: Int32,
        src_row_stride: Int64,
        out_row_stride: Int64,
    ):
        _kernel(
            mSrc,
            mIndex,
            mOut,
            num_entries,
            D,
            src_row_stride,
            out_row_stride,
        ).launch(
            grid=[grid_x, 1, 1],
            block=[_THREADS_PER_BLOCK, 1, 1],
            stream=stream,
        )

    return _launch


@instrumented_cutedsl_cache(
    "aten::scatter_add",
    key_fn=lambda torch_dtype, contig: f"vec {torch_dtype} contig={contig}",
)
def _compile_vec_scatter(torch_dtype: torch.dtype, contig: bool):
    # N is a runtime arg (``D``), never a compile-time constant here: the
    # kernel loops ``while off < D`` and loads fixed ``vec_elems`` chunks off
    # a raw iterator, so the inner dim can stay symbolic. One compile per
    # (dtype, contig) serves every N.
    dtype = _TORCH_TO_CUTE[torch_dtype]
    elem_bytes = dtype.width // 8
    vec_elems = _VEC_BYTES // elem_bytes
    launcher = _make_kernel(dtype, elem_bytes, vec_elems, contig)

    mSrc_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    # Index is contiguous (see _flatten_for_expanded_1d); fix stride=1
    # so `mIndex[i]` doesn't emit a runtime stride multiply.
    mIndex_fake = cute.runtime.make_fake_tensor(Int64, (cute.sym_int(),), stride=(1,))
    mOut_fake = cute.runtime.make_fake_tensor(
        dtype, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    return cute.compile(
        launcher,
        mSrc_fake,
        mIndex_fake,
        mOut_fake,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        Int32(0),
        Int32(0),
        Int32(0),
        Int64(0),
        Int64(0),
        # ``--enable-assertions`` keeps the ``cute_testing.assert_``
        # bounds checks on ``r`` live in production. Cost is roughly
        # +1-10% (geomean +7.7%) on most shapes; the safety net is
        # worth it because an OOB ``r`` would otherwise silently
        # corrupt unrelated gmem via the per-element ``atomicAdd``.
        options="--enable-tvm-ffi --enable-assertions",
    )


def vec_elems_for(dtype: torch.dtype) -> int:
    """Elements per lane per step (16-byte vector gather)."""
    return _VEC_BYTES // dtype.itemsize


def vec_scatter_add_into(
    out: torch.Tensor,
    index_1d: torch.Tensor,
    src: torch.Tensor,
) -> None:
    """In-place: ``out[index_1d[i], :] += src[i, :]`` for every i.

    ``out`` / ``src`` are 2D with inner-dim stride 1; outer row stride
    can differ from N. Requires N divisible by ``vec_elems_for(src.dtype)``
    (the cond enforces this). When N < 32 * vec_elems the loop has fewer
    active lanes per warp; lanes whose starting offset is >= N skip.
    """
    M, N = src.shape
    contig = src.stride(0) == N and out.stride(0) == N
    compiled = _compile_vec_scatter(src.dtype, contig)
    sm = torch.cuda.get_device_properties(out.device).multi_processor_count
    grid = min((M + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK, sm * 8)
    compiled(src, index_1d, out, M, N, grid, src.stride(0), out.stride(0))
<|endoftext|>from __future__ import annotations

import functools
import math
from typing import ParamSpec, TYPE_CHECKING, TypeVar

import torch

from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, KeepDims, normalizer, OutArray


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_P = ParamSpec("_P")
_R = TypeVar("_R")


class LinAlgError(Exception):
    pass


def _atleast_float_1(a: torch.Tensor) -> torch.Tensor:
    if not (a.dtype.is_floating_point or a.dtype.is_complex):
        a = a.to(_dtypes_impl.default_dtypes().float_dtype)
    return a


def _atleast_float_2(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dtyp = _dtypes_impl.result_type_impl(a, b)
    if not (dtyp.is_floating_point or dtyp.is_complex):
        dtyp = _dtypes_impl.default_dtypes().float_dtype

    a = _util.cast_if_needed(a, dtyp)
    b = _util.cast_if_needed(b, dtyp)
    return a, b


def linalg_errors(func: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(func)
    def wrapped(*args: _P.args, **kwds: _P.kwargs) -> _R:
        try:
            return func(*args, **kwds)
        except torch._C._LinAlgError as e:  # pyrefly: ignore[missing-attribute]  # TODO
            raise LinAlgError(*e.args)  # noqa: B904

    return wrapped


# ### Matrix and vector products ###


@normalizer
@linalg_errors
def matrix_power(a: ArrayLike, n: int) -> torch.Tensor:
    a = _atleast_float_1(a)
    return torch.linalg.matrix_power(a, n)


@normalizer
@linalg_errors
def multi_dot(
    inputs: Sequence[ArrayLike], *, out: OutArray | None = None
) -> torch.Tensor:
    return torch.linalg.multi_dot(inputs)


# ### Solving equations and inverting matrices ###


@normalizer
@linalg_errors
def solve(a: ArrayLike, b: ArrayLike) -> torch.Tensor:
    a, b = _atleast_float_2(a, b)
    return torch.linalg.solve(a, b)


@normalizer
@linalg_errors
def lstsq(
    a: ArrayLike, b: ArrayLike, rcond: float | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = _atleast_float_2(a, b)
    # NumPy is using gelsd: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/umath_linalg.cpp#L3991
    # on CUDA, only `gels` is available though, so use it instead
    driver = "gels" if a.is_cuda or b.is_cuda else "gelsd"
    return torch.linalg.lstsq(a, b, rcond=rcond, driver=driver)


@normalizer
@linalg_errors
def inv(a: ArrayLike) -> torch.Tensor:
    a = _atleast_float_1(a)
    result = torch.linalg.inv(a)
    return result


@normalizer
@linalg_errors
def pinv(a: ArrayLike, rcond: float = 1e-15, hermitian: bool = False) -> torch.Tensor:
    a = _atleast_float_1(a)
    return torch.linalg.pinv(a, rtol=rcond, hermitian=hermitian)


@normalizer
@linalg_errors
def tensorsolve(
    a: ArrayLike, b: ArrayLike, axes: Sequence[int] | None = None
) -> torch.Tensor:
    a, b = _atleast_float_2(a, b)
    return torch.linalg.tensorsolve(a, b, dims=axes)


@normalizer
@linalg_errors
def tensorinv(a: ArrayLike, ind: int = 2) -> torch.Tensor:
    a = _atleast_float_1(a)
    return torch.linalg.tensorinv(a, ind=ind)


# ### Norms and other numbers ###


@normalizer
@linalg_errors
def det(a: ArrayLike) -> torch.Tensor:
    a = _atleast_float_1(a)
    return torch.linalg.det(a)


@normalizer
@linalg_errors
def slogdet(a: ArrayLike) -> tuple[torch.Tensor, torch.Tensor]:
    a = _atleast_float_1(a)
    return torch.linalg.slogdet(a)


@normalizer
@linalg_errors
def cond(x: ArrayLike, p: int | str | None = None) -> torch.Tensor:
    x = _atleast_float_1(x)

    # check if empty
    # cf: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1744
    if x.numel() == 0 and math.prod(x.shape[-2:]) == 0:
        raise LinAlgError("cond is not defined on empty arrays")

    result = torch.linalg.cond(x, p=p)

    # Convert nans to infs (numpy does it in a data-dependent way, depending on
    # whether the input array has nans or not)
    # XXX: NumPy does this: https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1744
    return torch.where(torch.isnan(result), float("inf"), result)


@normalizer
@linalg_errors
def matrix_rank(
    a: ArrayLike, tol: float | None = None, hermitian: bool = False
) -> torch.Tensor | int:
    a = _atleast_float_1(a)

    if a.ndim < 2:
        return int((a != 0).any())

    if tol is None:
        # follow https://github.com/numpy/numpy/blob/v1.24.0/numpy/linalg/linalg.py#L1885
        atol = 0
        rtol = max(a.shape[-2:]) * torch.finfo(a.dtype).eps
    else:
        atol, rtol = tol, 0
    return torch.linalg.matrix_rank(a, atol=atol, rtol=rtol, hermitian=hermitian)


@normalizer
@linalg_errors
def norm(
    x: ArrayLike,
    ord: int | float | str | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: KeepDims = False,
) -> torch.Tensor:
    x = _atleast_float_1(x)
    return torch.linalg.norm(x, ord=ord, dim=axis)


# ### Decompositions ###


@normalizer
@linalg_errors
def cholesky(a: ArrayLike) -> torch.Tensor:
    a = _atleast_float_1(a)
    return torch.linalg.cholesky(a)


@normalizer
@linalg_errors
def qr(
    a: ArrayLike, mode: str = "reduced"
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    a = _atleast_float_1(a)
    result = torch.linalg.qr(a, mode=mode)
    if mode == "r":
        # match NumPy
        return result.R
    return result


@normalizer
@linalg_errors
def svd(
    a: ArrayLike,
    full_matrices: bool = True,
    compute_uv: bool = True,
    hermitian: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
    a = _atleast_float_1(a)
    if not compute_uv:
        return torch.linalg.svdvals(a)

    # NB: ignore the hermitian= argument (no pytorch equivalent)
    result = torch.linalg.svd(a, full_matrices=full_matrices)
    return result


# ### Eigenvalues and eigenvectors ###


@normalizer
@linalg_errors
def eig(a: ArrayLike) -> tuple[torch.Tensor, torch.Tensor]:
    a = _atleast_float_1(a)
    w, vt = torch.linalg.eig(a)

    if not a.is_complex() and w.is_complex() and (w.imag == 0).all():
        w = w.real
        vt = vt.real
    return w, vt


@normalizer
@linalg_errors
def eigh(a: ArrayLike, UPLO: str = "L") -> tuple[torch.Tensor, torch.Tensor]:
    a = _atleast_float_1(a)
    return torch.linalg.eigh(a, UPLO=UPLO)


@normalizer
@linalg_errors
def eigvals(a: ArrayLike) -> torch.Tensor:
    a = _atleast_float_1(a)
    result = torch.linalg.eigvals(a)
    if not a.is_complex() and result.is_complex() and (result.imag == 0).all():
        result = result.real
    return result


@normalizer
@linalg_errors
def eigvalsh(a: ArrayLike, UPLO: str = "L") -> torch.Tensor:
    a = _atleast_float_1(a)
    return torch.linalg.eigvalsh(a, UPLO=UPLO)
<|endoftext|># mypy: ignore-errors

"""A thin pytorch / numpy compat layer.

Things imported from here have numpy-compatible signatures but operate on
pytorch tensors.
"""

# Contents of this module ends up in the main namespace via _funcs.py
# where type annotations are used in conjunction with the @normalizer decorator.
from __future__ import annotations

import builtins
import itertools
import operator
from typing import TYPE_CHECKING

import torch

from . import _dtypes_impl, _util


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._normalizations import (
        ArrayLike,
        ArrayLikeOrScalar,
        CastingModes,
        DTypeLike,
        NDArray,
        NotImplementedType,
        OutArray,
    )


def copy(
    a: ArrayLike, order: NotImplementedType = "K", subok: NotImplementedType = False
):
    return a.clone()


def copyto(
    dst: NDArray,
    src: ArrayLike,
    casting: CastingModes | None = "same_kind",
    where: NotImplementedType = None,
):
    (src,) = _util.typecast_tensors((src,), dst.dtype, casting=casting)
    dst.copy_(src)


def atleast_1d(*arys: ArrayLike):
    res = torch.atleast_1d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def atleast_2d(*arys: ArrayLike):
    res = torch.atleast_2d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def atleast_3d(*arys: ArrayLike):
    res = torch.atleast_3d(*arys)
    if isinstance(res, tuple):
        return list(res)
    else:
        return res


def _concat_check(tup, dtype, out):
    if tup == ():
        raise ValueError("need at least one array to concatenate")

    """Check inputs in concatenate et al."""
    if out is not None and dtype is not None:
        # mimic numpy
        raise TypeError(
            "concatenate() only takes `out` or `dtype` as an "
            "argument, but both were provided."
        )


def _concat_cast_helper(tensors, out=None, dtype=None, casting="same_kind"):
    """Figure out dtypes, cast if necessary."""

    if out is not None or dtype is not None:
        # figure out the type of the inputs and outputs
        out_dtype = out.dtype.torch_dtype if dtype is None else dtype
    else:
        out_dtype = _dtypes_impl.result_type_impl(*tensors)

    # cast input arrays if necessary; do not broadcast them against `out`
    tensors = _util.typecast_tensors(tensors, out_dtype, casting)

    return tensors


def _concatenate(
    tensors, axis=0, out=None, dtype=None, casting: CastingModes | None = "same_kind"
):
    # pure torch implementation, used below and in cov/corrcoef below
    tensors, axis = _util.axis_none_flatten(*tensors, axis=axis)
    tensors = _concat_cast_helper(tensors, out, dtype, casting)
    return torch.cat(tensors, axis)


def concatenate(
    ar_tuple: Sequence[ArrayLike],
    axis=0,
    out: OutArray | None = None,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    _concat_check(ar_tuple, dtype, out=out)
    result = _concatenate(ar_tuple, axis=axis, out=out, dtype=dtype, casting=casting)
    return result


def vstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.vstack(tensors)


row_stack = vstack


def hstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.hstack(tensors)


def dstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    # XXX: in numpy 1.24 dstack does not have dtype and casting keywords
    # but {h,v}stack do.  Hence add them here for consistency.
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.dstack(tensors)


def column_stack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    # XXX: in numpy 1.24 column_stack does not have dtype and casting keywords
    # but row_stack does. (because row_stack is an alias for vstack, really).
    # Hence add these keywords here for consistency.
    _concat_check(tup, dtype, out=None)
    tensors = _concat_cast_helper(tup, dtype=dtype, casting=casting)
    return torch.column_stack(tensors)


def stack(
    arrays: Sequence[ArrayLike],
    axis=0,
    out: OutArray | None = None,
    *,
    dtype: DTypeLike | None = None,
    casting: CastingModes | None = "same_kind",
):
    _concat_check(arrays, dtype, out=out)

    tensors = _concat_cast_helper(arrays, dtype=dtype, casting=casting)
    result_ndim = tensors[0].ndim + 1
    axis = _util.normalize_axis_index(axis, result_ndim)
    return torch.stack(tensors, axis=axis)


def append(arr: ArrayLike, values: ArrayLike, axis=None):
    if axis is None:
        if arr.ndim != 1:
            arr = arr.flatten()
        values = values.flatten()
        axis = arr.ndim - 1
    return _concatenate((arr, values), axis=axis)


# ### split ###


def _split_helper(tensor, indices_or_sections, axis, strict=False):
    if isinstance(indices_or_sections, int):
        return _split_helper_int(tensor, indices_or_sections, axis, strict)
    elif isinstance(indices_or_sections, (list, tuple)):
        # NB: drop split=..., it only applies to split_helper_int
        return _split_helper_list(tensor, list(indices_or_sections), axis)
    else:
        raise TypeError(f"split_helper: {type(indices_or_sections)}")


def _split_helper_int(tensor, indices_or_sections, axis, strict=False):
    if not isinstance(indices_or_sections, int):
        raise NotImplementedError("split: indices_or_sections")

    axis = _util.normalize_axis_index(axis, tensor.ndim)

    # numpy: l%n chunks of size (l//n + 1), the rest are sized l//n
    l, n = tensor.shape[axis], indices_or_sections

    if n <= 0:
        raise ValueError

    if l % n == 0:
        num, sz = n, l // n
        lst = [sz] * num
    else:
        if strict:
            raise ValueError("array split does not result in an equal division")

        num, sz = l % n, l // n + 1
        lst = [sz] * num

    lst += [sz - 1] * (n - num)

    return torch.split(tensor, lst, axis)


def _split_helper_list(tensor, indices_or_sections, axis):
    if not isinstance(indices_or_sections, list):
        raise NotImplementedError("split: indices_or_sections: list")
    # numpy expects indices, while torch expects lengths of sections
    # also, numpy appends zero-size arrays for indices above the shape[axis]
    lst = [x for x in indices_or_sections if x <= tensor.shape[axis]]
    num_extra = len(indices_or_sections) - len(lst)

    lst.append(tensor.shape[axis])
    lst = [
        lst[0],
    ] + [a - b for a, b in zip(lst[1:], lst[:-1])]
    lst += [0] * num_extra

    return torch.split(tensor, lst, axis)


def array_split(ary: ArrayLike, indices_or_sections, axis=0):
    return _split_helper(ary, indices_or_sections, axis)


def split(ary: ArrayLike, indices_or_sections, axis=0):
    return _split_helper(ary, indices_or_sections, axis, strict=True)


def hsplit(ary: ArrayLike, indices_or_sections):
    if ary.ndim == 0:
        raise ValueError("hsplit only works on arrays of 1 or more dimensions")
    axis = 1 if ary.ndim > 1 else 0
    return _split_helper(ary, indices_or_sections, axis, strict=True)


def vsplit(ary: ArrayLike, indices_or_sections):
    if ary.ndim < 2:
        raise ValueError("vsplit only works on arrays of 2 or more dimensions")
    return _split_helper(ary, indices_or_sections, 0, strict=True)


def dsplit(ary: ArrayLike, indices_or_sections):
    if ary.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return _split_helper(ary, indices_or_sections, 2, strict=True)


def kron(a: ArrayLike, b: ArrayLike):
    return torch.kron(a, b)


def vander(x: ArrayLike, N=None, increasing=False):
    return torch.vander(x, N, increasing)


# ### linspace, geomspace, logspace and arange ###


def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num=50,
    endpoint=True,
    retstep=False,
    dtype: DTypeLike | None = None,
    axis=0,
):
    if axis != 0 or retstep or not endpoint:
        raise NotImplementedError
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    # XXX: raises TypeError if start or stop are not scalars
    return torch.linspace(start, stop, num, dtype=dtype)


def geomspace(
    start: ArrayLike,
    stop: ArrayLike,
    num=50,
    endpoint=True,
    dtype: DTypeLike | None = None,
    axis=0,
):
    if axis != 0 or not endpoint:
        raise NotImplementedError
    base = torch.pow(stop / start, 1.0 / (num - 1))
    logbase = torch.log(base)
    return torch.logspace(
        torch.log(start) / logbase,
        torch.log(stop) / logbase,
        num,
        base=base,
    )


def logspace(
    start,
    stop,
    num=50,
    endpoint=True,
    base=10.0,
    dtype: DTypeLike | None = None,
    axis=0,
):
    if axis != 0 or not endpoint:
        raise NotImplementedError
    return torch.logspace(start, stop, num, base=base, dtype=dtype)


def arange(
    start: ArrayLikeOrScalar | None = None,
    stop: ArrayLikeOrScalar | None = None,
    step: ArrayLikeOrScalar | None = 1,
    dtype: DTypeLike | None = None,
    *,
    like: NotImplementedType = None,
):
    if step == 0:
        raise ZeroDivisionError
    if stop is None and start is None:
        raise TypeError
    if stop is None:
        # XXX: this breaks if start is passed as a kwarg:
        # arange(start=4) should raise (no stop) but doesn't
        start, stop = 0, start
    if start is None:
        start = 0

    # the dtype of the result
    if dtype is None:
        dtype = (
            _dtypes_impl.default_dtypes().float_dtype
            if any(_dtypes_impl.is_float_or_fp_tensor(x) for x in (start, stop, step))
            else _dtypes_impl.default_dtypes().int_dtype
        )
    work_dtype = torch.float64 if dtype.is_complex else dtype

    # RuntimeError: "lt_cpu" not implemented for 'ComplexFloat'. Fall back to eager.
    if any(_dtypes_impl.is_complex_or_complex_tensor(x) for x in (start, stop, step)):
        raise NotImplementedError

    if (step > 0 and start > stop) or (step < 0 and start < stop):
        # empty range
        return torch.empty(0, dtype=dtype)

    result = torch.arange(start, stop, step, dtype=work_dtype)
    result = _util.cast_if_needed(result, dtype)
    return result


# ### zeros/ones/empty/full ###


def empty(
    shape,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.empty(shape, dtype=dtype)


# NB: *_like functions deliberately deviate from numpy: it has subok=True
# as the default; we set subok=False and raise on anything else.


def empty_like(
    prototype: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape=None,
):
    result = torch.empty_like(prototype, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def full(
    shape,
    fill_value: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    if isinstance(shape, int):
        shape = (shape,)
    if dtype is None:
        dtype = fill_value.dtype
    if not isinstance(shape, (tuple, list)):
        shape = (shape,)
    return torch.full(shape, fill_value, dtype=dtype)


def full_like(
    a: ArrayLike,
    fill_value,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape=None,
):
    # XXX: fill_value broadcasts
    result = torch.full_like(a, fill_value, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def ones(
    shape,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.ones(shape, dtype=dtype)


def ones_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape=None,
):
    result = torch.ones_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


def zeros(
    shape,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.zeros(shape, dtype=dtype)


def zeros_like(
    a: ArrayLike,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "K",
    subok: NotImplementedType = False,
    shape=None,
):
    result = torch.zeros_like(a, dtype=dtype)
    if shape is not None:
        result = result.reshape(shape)
    return result


# ### cov & corrcoef ###


def _xy_helper_corrcoef(x_tensor, y_tensor=None, rowvar=True):
    """Prepare inputs for cov and corrcoef."""

    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/function_base.py#L2636
    if y_tensor is not None:
        # make sure x and y are at least 2D
        ndim_extra = 2 - x_tensor.ndim
        if ndim_extra > 0:
            x_tensor = x_tensor.view((1,) * ndim_extra + x_tensor.shape)
        if not rowvar and x_tensor.shape[0] != 1:
            x_tensor = x_tensor.mT
        x_tensor = x_tensor.clone()

        ndim_extra = 2 - y_tensor.ndim
        if ndim_extra > 0:
            y_tensor = y_tensor.view((1,) * ndim_extra + y_tensor.shape)
        if not rowvar and y_tensor.shape[0] != 1:
            y_tensor = y_tensor.mT
        y_tensor = y_tensor.clone()

        x_tensor = _concatenate((x_tensor, y_tensor), axis=0)

    return x_tensor


def corrcoef(
    x: ArrayLike,
    y: ArrayLike | None = None,
    rowvar=True,
    bias=None,
    ddof=None,
    *,
    dtype: DTypeLike | None = None,
):
    if bias is not None or ddof is not None:
        # deprecated in NumPy
        raise NotImplementedError
    xy_tensor = _xy_helper_corrcoef(x, y, rowvar)

    is_half = (xy_tensor.dtype == torch.float16) and xy_tensor.is_cpu
    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32

    xy_tensor = _util.cast_if_needed(xy_tensor, dtype)
    result = torch.corrcoef(xy_tensor)

    if is_half:
        result = result.to(torch.float16)

    return result


def cov(
    m: ArrayLike,
    y: ArrayLike | None = None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights: ArrayLike | None = None,
    aweights: ArrayLike | None = None,
    *,
    dtype: DTypeLike | None = None,
):
    m = _xy_helper_corrcoef(m, y, rowvar)

    if ddof is None:
        ddof = 1 if bias == 0 else 0

    is_half = (m.dtype == torch.float16) and m.is_cpu
    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32

    m = _util.cast_if_needed(m, dtype)
    result = torch.cov(m, correction=ddof, aweights=aweights, fweights=fweights)

    if is_half:
        result = result.to(torch.float16)

    return result


def _conv_corr_impl(a, v, mode):
    dt = _dtypes_impl.result_type_impl(a, v)
    a = _util.cast_if_needed(a, dt)
    v = _util.cast_if_needed(v, dt)

    padding = v.shape[0] - 1 if mode == "full" else mode

    if padding == "same" and v.shape[0] % 2 == 0:
        # UserWarning: Using padding='same' with even kernel lengths and odd
        # dilation may require a zero-padded copy of the input be created
        # (Triggered internally at pytorch/aten/src/ATen/native/Convolution.cpp:1010.)
        raise NotImplementedError("mode='same' and even-length weights")

    # NumPy only accepts 1D arrays; PyTorch requires 2D inputs and 3D weights
    aa = a[None, :]
    vv = v[None, None, :]

    result = torch.nn.functional.conv1d(aa, vv, padding=padding)

    # torch returns a 2D result, numpy returns a 1D array
    return result[0, :]


def convolve(a: ArrayLike, v: ArrayLike, mode="full"):
    # NumPy: if v is longer than a, the arrays are swapped before computation
    if a.shape[0] < v.shape[0]:
        a, v = v, a

    # flip the weights since numpy does and torch does not
    v = torch.flip(v, (0,))

    return _conv_corr_impl(a, v, mode)


def correlate(a: ArrayLike, v: ArrayLike, mode="valid"):
    v = torch.conj_physical(v)
    return _conv_corr_impl(a, v, mode)


# ### logic & element selection ###


def bincount(x: ArrayLike, /, weights: ArrayLike | None = None, minlength=0):
    if x.numel() == 0:
        # edge case allowed by numpy
        x = x.new_empty(0, dtype=int)

    int_dtype = _dtypes_impl.default_dtypes().int_dtype
    (x,) = _util.typecast_tensors((x,), int_dtype, casting="safe")

    return torch.bincount(x, weights, minlength)


def where(
    condition: ArrayLike,
    x: ArrayLikeOrScalar | None = None,
    y: ArrayLikeOrScalar | None = None,
    /,
):
    if (x is None) != (y is None):
        raise ValueError("either both or neither of x and y should be given")

    if condition.dtype != torch.bool:
        condition = condition.to(torch.bool)

    if x is None and y is None:
        result = torch.where(condition)
    else:
        result = torch.where(condition, x, y)
    return result


# ###### module-level queries of object properties


def ndim(a: ArrayLike):
    return a.ndim


def shape(a: ArrayLike):
    return tuple(a.shape)


def size(a: ArrayLike, axis=None):
    if axis is None:
        return a.numel()
    else:
        return a.shape[axis]


# ###### shape manipulations and indexing


def expand_dims(a: ArrayLike, axis):
    shape = _util.expand_shape(a.shape, axis)
    return a.view(shape)  # never copies


def flip(m: ArrayLike, axis=None):
    # XXX: semantic difference: np.flip returns a view, torch.flip copies
    if axis is None:
        axis = tuple(range(m.ndim))
    else:
        axis = _util.normalize_axis_tuple(axis, m.ndim)
    return torch.flip(m, axis)


def flipud(m: ArrayLike):
    return torch.flipud(m)


def fliplr(m: ArrayLike):
    return torch.fliplr(m)


def rot90(m: ArrayLike, k=1, axes=(0, 1)):
    axes = _util.normalize_axis_tuple(axes, m.ndim)
    return torch.rot90(m, k, axes)


# ### broadcasting and indices ###


def broadcast_to(array: ArrayLike, shape, subok: NotImplementedType = False):
    return torch.broadcast_to(array, size=shape)


# This is a function from tuples to tuples, so we just reuse it.  However,
# dynamo expects its __module__ to be torch._numpy
def broadcast_shapes(*args):
    return torch.broadcast_shapes(*args)


def broadcast_arrays(*args: ArrayLike, subok: NotImplementedType = False):
    return torch.broadcast_tensors(*args)


def meshgrid(*xi: ArrayLike, copy=True, sparse=False, indexing="xy"):
    ndim = len(xi)

    if indexing not in ["xy", "ij"]:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1 :]) for i, x in enumerate(xi)]

    if indexing == "xy" and ndim > 1:
        # switch first and second axis
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])

    if not sparse:
        # Return the full N-D matrix (not only the 1-D vector)
        output = torch.broadcast_tensors(*output)

    if copy:
        output = [x.clone() for x in output]

    return list(output)  # match numpy, return a list


def indices(dimensions, dtype: DTypeLike | None = int, sparse=False):
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1691-L1791
    dimensions = tuple(dimensions)
    N = len(dimensions)
    shape = (1,) * N
    if sparse:
        res = ()
    else:
        res = torch.empty((N,) + dimensions, dtype=dtype)
    for i, dim in enumerate(dimensions):
        idx = torch.arange(dim, dtype=dtype).reshape(
            shape[:i] + (dim,) + shape[i + 1 :]
        )
        if sparse:
            res = res + (idx,)
        else:
            res[i] = idx
    return res


# ### tri*-something ###


def tril(m: ArrayLike, k=0):
    return torch.tril(m, k)


def triu(m: ArrayLike, k=0):
    return torch.triu(m, k)


def tril_indices(n, k=0, m=None):
    if m is None:
        m = n
    return torch.tril_indices(n, m, offset=k)


def triu_indices(n, k=0, m=None):
    if m is None:
        m = n
    return torch.triu_indices(n, m, offset=k)


def tril_indices_from(arr: ArrayLike, k=0):
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    # Return a tensor rather than a tuple to avoid a graphbreak
    return torch.tril_indices(arr.shape[0], arr.shape[1], offset=k)


def triu_indices_from(arr: ArrayLike, k=0):
    if arr.ndim != 2:
        raise ValueError("input array must be 2-d")
    # Return a tensor rather than a tuple to avoid a graphbreak
    return torch.triu_indices(arr.shape[0], arr.shape[1], offset=k)


def tri(
    N,
    M=None,
    k=0,
    dtype: DTypeLike | None = None,
    *,
    like: NotImplementedType = None,
):
    if M is None:
        M = N
    tensor = torch.ones((N, M), dtype=dtype)
    return torch.tril(tensor, diagonal=k)


# ### equality, equivalence, allclose ###


def isclose(a: ArrayLike, b: ArrayLike, rtol=1.0e-5, atol=1.0e-8, equal_nan=False):
    dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    return torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def allclose(a: ArrayLike, b: ArrayLike, rtol=1e-05, atol=1e-08, equal_nan=False):
    dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def _tensor_equal(a1, a2, equal_nan=False):
    # Implementation of array_equal/array_equiv.
    if a1.shape != a2.shape:
        return False
    cond = a1 == a2
    if equal_nan:
        cond = cond | (torch.isnan(a1) & torch.isnan(a2))
    return cond.all().item()


def array_equal(a1: ArrayLike, a2: ArrayLike, equal_nan=False):
    return _tensor_equal(a1, a2, equal_nan=equal_nan)


def array_equiv(a1: ArrayLike, a2: ArrayLike):
    # *almost* the same as array_equal: _equiv tries to broadcast, _equal does not
    try:
        a1_t, a2_t = torch.broadcast_tensors(a1, a2)
    except RuntimeError:
        # failed to broadcast => not equivalent
        return False
    return _tensor_equal(a1_t, a2_t)


def nan_to_num(
    x: ArrayLike, copy: NotImplementedType = True, nan=0.0, posinf=None, neginf=None
):
    # work around RuntimeError: "nan_to_num" not implemented for 'ComplexDouble'
    if x.is_complex():
        re = torch.nan_to_num(x.real, nan=nan, posinf=posinf, neginf=neginf)
        im = torch.nan_to_num(x.imag, nan=nan, posinf=posinf, neginf=neginf)
        return re + 1j * im
    else:
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


# ### put/take_along_axis ###


def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis=None,
    out: OutArray | None = None,
    mode: NotImplementedType = "raise",
):
    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)
    idx = (slice(None),) * axis + (indices, ...)
    result = a[idx]
    return result


def take_along_axis(arr: ArrayLike, indices: ArrayLike, axis):
    (arr,), axis = _util.axis_none_flatten(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)
    return torch.take_along_dim(arr, indices, axis)


def put(
    a: NDArray,
    indices: ArrayLike,
    values: ArrayLike,
    mode: NotImplementedType = "raise",
):
    v = values.type(a.dtype)
    # If indices is larger than v, expand v to at least the size of indices. Any
    # unnecessary trailing elements are then trimmed.
    if indices.numel() > v.numel():
        ratio = (indices.numel() + v.numel() - 1) // v.numel()
        v = v.unsqueeze(0).expand((ratio,) + v.shape)
    # Trim unnecessary elements, regardless if v was expanded or not. Note
    # np.put() trims v to match indices by default too.
    if indices.numel() < v.numel():
        v = v.flatten()
        v = v[: indices.numel()]
    a.put_(indices, v)
    return None


def put_along_axis(arr: ArrayLike, indices: ArrayLike, values: ArrayLike, axis):
    (arr,), axis = _util.axis_none_flatten(arr, axis=axis)
    axis = _util.normalize_axis_index(axis, arr.ndim)

    indices, values = torch.broadcast_tensors(indices, values)
    values = _util.cast_if_needed(values, arr.dtype)
    result = torch.scatter(arr, axis, indices, values)
    arr.copy_(result.reshape(arr.shape))
    return None


def choose(
    a: ArrayLike,
    choices: Sequence[ArrayLike],
    out: OutArray | None = None,
    mode: NotImplementedType = "raise",
):
    # First, broadcast elements of `choices`
    choices = torch.stack(torch.broadcast_tensors(*choices))

    # Use an analog of `gather(choices, 0, a)` which broadcasts `choices` vs `a`:
    # (taken from https://github.com/pytorch/pytorch/issues/9407#issuecomment-1427907939)
    idx_list = [
        torch.arange(dim).view((1,) * i + (dim,) + (1,) * (choices.ndim - i - 1))
        for i, dim in enumerate(choices.shape)
    ]

    idx_list[0] = a
    return choices[tuple(idx_list)].squeeze(0)


# ### unique et al. ###


def unique(
    ar: ArrayLike,
    return_index: NotImplementedType = False,
    return_inverse=False,
    return_counts=False,
    axis=None,
    *,
    equal_nan: NotImplementedType = True,
):
    (ar,), axis = _util.axis_none_flatten(ar, axis=axis)
    axis = _util.normalize_axis_index(axis, ar.ndim)

    result = torch.unique(
        ar, return_inverse=return_inverse, return_counts=return_counts, dim=axis
    )

    return result


def nonzero(a: ArrayLike):
    return torch.nonzero(a, as_tuple=True)


def argwhere(a: ArrayLike):
    return torch.argwhere(a)


def flatnonzero(a: ArrayLike):
    return torch.flatten(a).nonzero(as_tuple=True)[0]


def clip(
    a: ArrayLike,
    min: ArrayLike | None = None,
    max: ArrayLike | None = None,
    out: OutArray | None = None,
):
    return torch.clamp(a, min, max)


def repeat(a: ArrayLike, repeats: ArrayLikeOrScalar, axis=None):
    return torch.repeat_interleave(a, repeats, axis)


def tile(A: ArrayLike, reps):
    if isinstance(reps, int):
        reps = (reps,)
    return torch.tile(A, reps)


def resize(a: ArrayLike, new_shape=None):
    # implementation vendored from
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/fromnumeric.py#L1420-L1497
    if new_shape is None:
        return a

    if isinstance(new_shape, int):
        new_shape = (new_shape,)

    a = a.flatten()

    new_size = 1
    for dim_length in new_shape:
        new_size *= dim_length
        if dim_length < 0:
            raise ValueError("all elements of `new_shape` must be non-negative")

    if a.numel() == 0 or new_size == 0:
        # First case must zero fill. The second would have repeats == 0.
        return torch.zeros(new_shape, dtype=a.dtype)

    repeats = -(-new_size // a.numel())  # ceil division
    a = concatenate((a,) * repeats)[:new_size]

    return reshape(a, new_shape)


# ### diag et al. ###


def diagonal(a: ArrayLike, offset=0, axis1=0, axis2=1):
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    return torch.diagonal(a, offset, axis1, axis2)


def trace(
    a: ArrayLike,
    offset=0,
    axis1=0,
    axis2=1,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
):
    result = torch.diagonal(a, offset, dim1=axis1, dim2=axis2).sum(-1, dtype=dtype)
    return result


def eye(
    N,
    M=None,
    k=0,
    dtype: DTypeLike | None = None,
    order: NotImplementedType = "C",
    *,
    like: NotImplementedType = None,
):
    if dtype is None:
        dtype = _dtypes_impl.default_dtypes().float_dtype
    if M is None:
        M = N
    z = torch.zeros(N, M, dtype=dtype)
    z.diagonal(k).fill_(1)
    return z


def identity(n, dtype: DTypeLike | None = None, *, like: NotImplementedType = None):
    return torch.eye(n, dtype=dtype)


def diag(v: ArrayLike, k=0):
    return torch.diag(v, k)


def diagflat(v: ArrayLike, k=0):
    return torch.diagflat(v, k)


def diag_indices(n, ndim=2):
    idx = torch.arange(n)
    return (idx,) * ndim


def diag_indices_from(arr: ArrayLike):
    if not arr.ndim >= 2:
        raise ValueError("input array must be at least 2-d")
    # For more than d=2, the strided formula is only valid for arrays with
    # all dimensions equal, so we check first.
    s = arr.shape
    if s[1:] != s[:-1]:
        raise ValueError("All dimensions of input must be of equal length")
    return diag_indices(s[0], arr.ndim)


def fill_diagonal(a: ArrayLike, val: ArrayLike, wrap=False):
    if a.ndim < 2:
        raise ValueError("array must be at least 2-d")
    if val.numel() == 0 and not wrap:
        a.fill_diagonal_(val)
        return a

    if val.ndim == 0:
        val = val.unsqueeze(0)

    # torch.Tensor.fill_diagonal_ only accepts scalars
    # If the size of val is too large, then val is trimmed
    if a.ndim == 2:
        tall = a.shape[0] > a.shape[1]
        # wrap does nothing for wide matrices...
        if not wrap or not tall:
            # Never wraps
            diag = a.diagonal()
            diag.copy_(val[: diag.numel()])
        else:
            # wraps and tall... leaving one empty line between diagonals?!
            max_, min_ = a.shape
            idx = torch.arange(max_ - max_ // (min_ + 1))
            mod = idx % min_
            div = idx // min_
            a[(div * (min_ + 1) + mod, mod)] = val[: idx.numel()]
    else:
        idx = diag_indices_from(a)
        # a.shape = (n, n, ..., n)
        a[idx] = val[: a.shape[0]]

    return a


def vdot(a: ArrayLike, b: ArrayLike, /):
    # 1. torch only accepts 1D arrays, numpy flattens
    # 2. torch requires matching dtype, while numpy casts (?)
    t_a, t_b = torch.atleast_1d(a, b)
    if t_a.ndim > 1:
        t_a = t_a.flatten()
    if t_b.ndim > 1:
        t_b = t_b.flatten()

    dtype = _dtypes_impl.result_type_impl(t_a, t_b)
    is_half = dtype == torch.float16 and (t_a.is_cpu or t_b.is_cpu)
    is_bool = dtype == torch.bool

    # work around torch's "dot" not implemented for 'Half', 'Bool'
    if is_half:
        dtype = torch.float32
    elif is_bool:
        dtype = torch.uint8

    t_a = _util.cast_if_needed(t_a, dtype)
    t_b = _util.cast_if_needed(t_b, dtype)

    result = torch.vdot(t_a, t_b)

    if is_half:
        result = result.to(torch.float16)
    elif is_bool:
        result = result.to(torch.bool)

    return result


def tensordot(a: ArrayLike, b: ArrayLike, axes=2):
    if isinstance(axes, (list, tuple)):
        axes = [[ax] if isinstance(ax, int) else ax for ax in axes]

    target_dtype = _dtypes_impl.result_type_impl(a, b)
    a = _util.cast_if_needed(a, target_dtype)
    b = _util.cast_if_needed(b, target_dtype)

    return torch.tensordot(a, b, dims=axes)


def dot(a: ArrayLike, b: ArrayLike, out: OutArray | None = None):
    dtype = _dtypes_impl.result_type_impl(a, b)
    is_bool = dtype == torch.bool
    if is_bool:
        dtype = torch.uint8

    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    if a.ndim == 0 or b.ndim == 0:
        result = a * b
    else:
        result = torch.matmul(a, b)

    if is_bool:
        result = result.to(torch.bool)

    return result


def inner(a: ArrayLike, b: ArrayLike, /):
    dtype = _dtypes_impl.result_type_impl(a, b)
    is_half = dtype == torch.float16 and (a.is_cpu or b.is_cpu)
    is_bool = dtype == torch.bool

    if is_half:
        # work around torch's "addmm_impl_cpu_" not implemented for 'Half'"
        dtype = torch.float32
    elif is_bool:
        dtype = torch.uint8

    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    result = torch.inner(a, b)

    if is_half:
        result = result.to(torch.float16)
    elif is_bool:
        result = result.to(torch.bool)
    return result


def outer(a: ArrayLike, b: ArrayLike, out: OutArray | None = None):
    return torch.outer(a, b)


def cross(a: ArrayLike, b: ArrayLike, axisa=-1, axisb=-1, axisc=-1, axis=None):
    # implementation vendored from
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1486-L1685
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3

    # Check axisa and axisb are within bounds
    axisa = _util.normalize_axis_index(axisa, a.ndim)
    axisb = _util.normalize_axis_index(axisb, b.ndim)

    # Move working axis to the end of the shape
    a = torch.moveaxis(a, axisa, -1)
    b = torch.moveaxis(b, axisb, -1)
    msg = "incompatible dimensions for cross product\n(dimension must be 2 or 3)"
    if a.shape[-1] not in (2, 3) or b.shape[-1] not in (2, 3):
        raise ValueError(msg)

    # Create the output array
    shape = broadcast_shapes(a[..., 0].shape, b[..., 0].shape)
    if a.shape[-1] == 3 or b.shape[-1] == 3:
        shape += (3,)
        # Check axisc is within bounds
        axisc = _util.normalize_axis_index(axisc, len(shape))
    dtype = _dtypes_impl.result_type_impl(a, b)
    cp = torch.empty(shape, dtype=dtype)

    # recast arrays as dtype
    a = _util.cast_if_needed(a, dtype)
    b = _util.cast_if_needed(b, dtype)

    # create local aliases for readability
    a0 = a[..., 0]
    a1 = a[..., 1]
    if a.shape[-1] == 3:
        a2 = a[..., 2]
    b0 = b[..., 0]
    b1 = b[..., 1]
    if b.shape[-1] == 3:
        b2 = b[..., 2]
    if cp.ndim != 0 and cp.shape[-1] == 3:
        cp0 = cp[..., 0]
        cp1 = cp[..., 1]
        cp2 = cp[..., 2]

    if a.shape[-1] == 2:
        if b.shape[-1] == 2:
            # a0 * b1 - a1 * b0
            cp[...] = a0 * b1 - a1 * b0
            return cp
        else:
            if b.shape[-1] != 3:
                raise AssertionError(f"b.shape[-1] must be 3, got {b.shape[-1]}")
            # cp0 = a1 * b2 - 0  (a2 = 0)
            # cp1 = 0 - a0 * b2  (a2 = 0)
            # cp2 = a0 * b1 - a1 * b0
            cp0[...] = a1 * b2
            cp1[...] = -a0 * b2
            cp2[...] = a0 * b1 - a1 * b0
    else:
        if a.shape[-1] != 3:
            raise AssertionError(f"a.shape[-1] must be 3, got {a.shape[-1]}")
        if b.shape[-1] == 3:
            cp0[...] = a1 * b2 - a2 * b1
            cp1[...] = a2 * b0 - a0 * b2
            cp2[...] = a0 * b1 - a1 * b0
        else:
            if b.shape[-1] != 2:
                raise AssertionError(f"b.shape[-1] must be 2, got {b.shape[-1]}")
            cp0[...] = -a2 * b1
            cp1[...] = a2 * b0
            cp2[...] = a0 * b1 - a1 * b0

    return torch.moveaxis(cp, -1, axisc)


def einsum(*operands, out=None, dtype=None, order="K", casting="safe", optimize=False):
    # Have to manually normalize *operands and **kwargs, following the NumPy signature
    # We have a local import to avoid polluting the global space, as it will be then
    # exported in funcs.py
    from ._ndarray import ndarray
    from ._normalizations import (
        maybe_copy_to,
        normalize_array_like,
        normalize_casting,
        normalize_dtype,
        wrap_tensors,
    )

    dtype = normalize_dtype(dtype)
    casting = normalize_casting(casting)
    if out is not None and not isinstance(out, ndarray):
        raise TypeError("'out' must be an array")
    if order != "K":
        raise NotImplementedError("'order' parameter is not supported.")

    # parse arrays and normalize them
    sublist_format = not isinstance(operands[0], str)
    if sublist_format:
        # op, str, op, str ... [sublistout] format: normalize every other argument

        # - if sublistout is not given, the length of operands is even, and we pick
        #   odd-numbered elements, which are arrays.
        # - if sublistout is given, the length of operands is odd, we peel off
        #   the last one, and pick odd-numbered elements, which are arrays.
        #   Without [:-1], we would have picked sublistout, too.
        array_operands = operands[:-1][::2]
    else:
        # ("ij->", arrays) format
        subscripts, array_operands = operands[0], operands[1:]

    tensors = [normalize_array_like(op) for op in array_operands]
    target_dtype = _dtypes_impl.result_type_impl(*tensors) if dtype is None else dtype

    # work around 'bmm' not implemented for 'Half' etc
    is_half = target_dtype == torch.float16 and all(t.is_cpu for t in tensors)
    if is_half:
        target_dtype = torch.float32

    is_short_int = target_dtype in [torch.uint8, torch.int8, torch.int16, torch.int32]
    if is_short_int:
        target_dtype = torch.int64

    tensors = _util.typecast_tensors(tensors, target_dtype, casting)

    from torch.backends import opt_einsum

    try:
        # set the global state to handle the optimize=... argument, restore on exit
        if opt_einsum.is_available():
            old_strategy = torch.backends.opt_einsum.strategy
            old_enabled = torch.backends.opt_einsum.enabled

            # torch.einsum calls opt_einsum.contract_path, which runs into
            # https://github.com/dgasmith/opt_einsum/issues/219
            # for strategy={True, False}
            if optimize is True:
                optimize = "auto"
            elif optimize is False:
                torch.backends.opt_einsum.enabled = False

            torch.backends.opt_einsum.strategy = optimize

        if sublist_format:
            # recombine operands
            sublists = operands[1::2]
            has_sublistout = len(operands) % 2 == 1
            if has_sublistout:
                sublistout = operands[-1]
            operands = list(itertools.chain.from_iterable(zip(tensors, sublists)))
            if has_sublistout:
                operands.append(sublistout)

            result = torch.einsum(*operands)
        else:
            result = torch.einsum(subscripts, *tensors)

    finally:
        if opt_einsum.is_available():
            torch.backends.opt_einsum.strategy = old_strategy
            torch.backends.opt_einsum.enabled = old_enabled

    result = maybe_copy_to(out, result)
    return wrap_tensors(result)


# ### sort and partition ###


def _sort_helper(tensor, axis, kind, order):
    if tensor.dtype.is_complex:
        raise NotImplementedError(f"sorting {tensor.dtype} is not supported")
    (tensor,), axis = _util.axis_none_flatten(tensor, axis=axis)
    axis = _util.normalize_axis_index(axis, tensor.ndim)

    stable = kind == "stable"

    return tensor, axis, stable


def sort(a: ArrayLike, axis=-1, kind=None, order: NotImplementedType = None):
    # `order` keyword arg is only relevant for structured dtypes; so not supported here.
    a, axis, stable = _sort_helper(a, axis, kind, order)
    result = torch.sort(a, dim=axis, stable=stable)
    return result.values


def argsort(a: ArrayLike, axis=-1, kind=None, order: NotImplementedType = None):
    a, axis, stable = _sort_helper(a, axis, kind, order)
    return torch.argsort(a, dim=axis, stable=stable)


def searchsorted(
    a: ArrayLike, v: ArrayLike, side="left", sorter: ArrayLike | None = None
):
    if a.dtype.is_complex:
        raise NotImplementedError(f"searchsorted with dtype={a.dtype}")

    return torch.searchsorted(a, v, side=side, sorter=sorter)


# ### swap/move/roll axis ###


def moveaxis(a: ArrayLike, source, destination):
    source = _util.normalize_axis_tuple(source, a.ndim, "source")
    destination = _util.normalize_axis_tuple(destination, a.ndim, "destination")
    return torch.moveaxis(a, source, destination)


def swapaxes(a: ArrayLike, axis1, axis2):
    axis1 = _util.normalize_axis_index(axis1, a.ndim)
    axis2 = _util.normalize_axis_index(axis2, a.ndim)
    return torch.swapaxes(a, axis1, axis2)


def rollaxis(a: ArrayLike, axis, start=0):
    # Straight vendor from:
    # https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numeric.py#L1259
    #
    # Also note this function in NumPy is mostly retained for backwards compat
    # (https://stackoverflow.com/questions/29891583/reason-why-numpy-rollaxis-is-so-confusing)
    # so let's not touch it unless hard pressed.
    n = a.ndim
    axis = _util.normalize_axis_index(axis, n)
    if start < 0:
        start += n
    msg = "'%s' arg requires %d <= %s < %d, but %d was passed in"
    if not (0 <= start < n + 1):
        raise _util.AxisError(msg % ("start", -n, "start", n + 1, start))
    if axis < start:
        # it's been removed
        start -= 1
    if axis == start:
        # numpy returns a view, here we try returning the tensor itself
        # return tensor[...]
        return a
    axes = list(range(n))
    axes.remove(axis)
    axes.insert(start, axis)
    return a.view(axes)


def roll(a: ArrayLike, shift, axis=None):
    if axis is not None:
        axis = _util.normalize_axis_tuple(axis, a.ndim, allow_duplicate=True)
        if not isinstance(shift, tuple):
            shift = (shift,) * len(axis)
    return torch.roll(a, shift, axis)


# ### shape manipulations ###


def squeeze(a: ArrayLike, axis=None):
    if axis == ():
        result = a
    elif axis is None:
        result = a.squeeze()
    else:
        if isinstance(axis, tuple):
            result = a
            for ax in axis:
                result = a.squeeze(ax)
        else:
            result = a.squeeze(axis)
    return result


def reshape(a: ArrayLike, newshape, order: NotImplementedType = "C"):
    # if sh = (1, 2, 3), numpy allows both .reshape(sh) and .reshape(*sh)
    newshape = newshape[0] if len(newshape) == 1 else newshape
    return a.reshape(newshape)


# NB: cannot use torch.reshape(a, newshape) above, because of
# (Pdb) torch.reshape(torch.as_tensor([1]), 1)
# *** TypeError: reshape(): argument 'shape' (position 2) must be tuple of SymInts, not int


def transpose(a: ArrayLike, axes=None):
    # numpy allows both .transpose(sh) and .transpose(*sh)
    # also older code uses axes being a list
    if axes in [(), None, (None,)]:
        axes = tuple(reversed(range(a.ndim)))
    elif len(axes) == 1:
        axes = axes[0]
    return a.permute(axes)


def ravel(a: ArrayLike, order: NotImplementedType = "C"):
    return torch.flatten(a)


def diff(
    a: ArrayLike,
    n=1,
    axis=-1,
    prepend: ArrayLike | None = None,
    append: ArrayLike | None = None,
):
    axis = _util.normalize_axis_index(axis, a.ndim)

    if n < 0:
        raise ValueError(f"order must be non-negative but got {n}")

    if n == 0:
        # match numpy and return the input immediately
        return a

    if prepend is not None:
        shape = list(a.shape)
        shape[axis] = prepend.shape[axis] if prepend.ndim > 0 else 1
        prepend = torch.broadcast_to(prepend, shape)

    if append is not None:
        shape = list(a.shape)
        shape[axis] = append.shape[axis] if append.ndim > 0 else 1
        append = torch.broadcast_to(append, shape)

    return torch.diff(a, n, axis=axis, prepend=prepend, append=append)


# ### math functions ###


def angle(z: ArrayLike, deg=False):
    result = torch.angle(z)
    if deg:
        result = result * (180 / torch.pi)
    return result


def sinc(x: ArrayLike):
    return torch.sinc(x)


# NB: have to normalize *varargs manually
def gradient(f: ArrayLike, *varargs, axis=None, edge_order=1):
    N = f.ndim  # number of dimensions

    varargs = _util.ndarrays_to_tensors(varargs)

    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _util.normalize_axis_tuple(axis, N)

    len_axes = len(axes)
    n = len(varargs)
    if n == 0:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 1 and (_dtypes_impl.is_scalar(varargs[0]) or varargs[0].ndim == 0):
        # single scalar or 0D tensor for all axes (np.ndim(varargs[0]) == 0)
        dx = varargs * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(varargs)
        for i, distances in enumerate(dx):
            distances = torch.as_tensor(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != f.shape[axes[i]]:
                raise ValueError(
                    "when 1d, distances must match "
                    "the length of the corresponding dimension"
                )
            if not (distances.dtype.is_floating_point or distances.dtype.is_complex):
                distances = distances.double()

            diffx = torch.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            if (diffx == diffx[0]).all():
                diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    otype = f.dtype
    if _dtypes_impl.python_type_for_torch(otype) in (int, bool):
        # Convert to floating point.
        # First check if f is a numpy integer type; if so, convert f to float64
        # to avoid modular arithmetic when computing the changes in f.
        f = f.double()
        otype = torch.float64

    for axis, ax_dx in zip(axes, dx):
        if f.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required."
            )
        # result allocation
        out = torch.empty_like(f, dtype=otype)

        # spacing for the current axis (NB: np.ndim(ax_dx) == 0)
        uniform_spacing = _dtypes_impl.is_scalar(ax_dx) or ax_dx.ndim == 0

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)

        if uniform_spacing:
            out[tuple(slice1)] = (f[tuple(slice4)] - f[tuple(slice2)]) / (2.0 * ax_dx)
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -(dx2) / (dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))
            # fix the shape for broadcasting
            shape = [1] * N
            shape[axis] = -1
            a = a.reshape(shape)
            b = b.reshape(shape)
            c = c.reshape(shape)
            # 1D equivalent -- out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
            out[tuple(slice1)] = (
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )

        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out[tuple(slice1)] = (f[tuple(slice2)] - f[tuple(slice3)]) / dx_n

        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2.0 / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = -dx1 / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
            out[tuple(slice1)] = (
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )

            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2.0 / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = (dx2) / (dx1 * (dx1 + dx2))
                b = -(dx2 + dx1) / (dx1 * dx2)
                c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            out[tuple(slice1)] = (
                a * f[tuple(slice2)] + b * f[tuple(slice3)] + c * f[tuple(slice4)]
            )

        outvals.append(out)

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    else:
        return outvals


# ### Type/shape etc queries ###


def round(a: ArrayLike, decimals=0, out: OutArray | None = None):
    if a.is_floating_point():
        result = torch.round(a, decimals=decimals)
    elif a.is_complex():
        # RuntimeError: "round_cpu" not implemented for 'ComplexFloat'
        result = torch.complex(
            torch.round(a.real, decimals=decimals),
            torch.round(a.imag, decimals=decimals),
        )
    else:
        # RuntimeError: "round_cpu" not implemented for 'int'
        result = a
    return result


around = round
round_ = round


def real_if_close(a: ArrayLike, tol=100):
    if not torch.is_complex(a):
        return a
    if tol > 1:
        # Undocumented in numpy: if tol < 1, it's an absolute tolerance!
        # Otherwise, tol > 1 is relative tolerance, in units of the dtype epsilon
        # https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/type_check.py#L577
        tol = tol * torch.finfo(a.dtype).eps

    mask = torch.abs(a.imag) < tol
    return a.real if mask.all() else a


def real(a: ArrayLike):
    return torch.real(a)


def imag(a: ArrayLike):
    if a.is_complex():
        return a.imag
    return torch.zeros_like(a)


def iscomplex(x: ArrayLike):
    if torch.is_complex(x):
        return x.imag != 0
    return torch.zeros_like(x, dtype=torch.bool)


def isreal(x: ArrayLike):
    if torch.is_complex(x):
        return x.imag == 0
    return torch.ones_like(x, dtype=torch.bool)


def iscomplexobj(x: ArrayLike):
    return torch.is_complex(x)


def isrealobj(x: ArrayLike):
    return not torch.is_complex(x)


def isneginf(x: ArrayLike, out: OutArray | None = None):
    return torch.isneginf(x)


def isposinf(x: ArrayLike, out: OutArray | None = None):
    return torch.isposinf(x)


def i0(x: ArrayLike):
    return torch.special.i0(x)


def isscalar(a):
    # We need to use normalize_array_like, but we don't want to export it in funcs.py
    from ._normalizations import normalize_array_like

    try:
        t = normalize_array_like(a)
        return t.numel() == 1
    except Exception:
        return False


# ### Filter windows ###


def hamming(M):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.hamming_window(M, periodic=False, dtype=dtype)


def hanning(M):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.hann_window(M, periodic=False, dtype=dtype)


def kaiser(M, beta):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.kaiser_window(M, beta=beta, periodic=False, dtype=dtype)


def blackman(M):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.blackman_window(M, periodic=False, dtype=dtype)


def bartlett(M):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    return torch.bartlett_window(M, periodic=False, dtype=dtype)


# ### Dtype routines ###

# vendored from https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/type_check.py#L666


array_type = [
    [torch.float16, torch.float32, torch.float64],
    [None, torch.complex64, torch.complex128],
]
array_precision = {
    torch.float16: 0,
    torch.float32: 1,
    torch.float64: 2,
    torch.complex64: 1,
    torch.complex128: 2,
}


def common_type(*tensors: ArrayLike):
    is_complex = False
    precision = 0
    for a in tensors:
        t = a.dtype
        if iscomplexobj(a):
            is_complex = True
        if not (t.is_floating_point or t.is_complex):
            p = 2  # array_precision[_nx.double]
        else:
            p = array_precision.get(t)
            if p is None:
                raise TypeError("can't get common type for non-numeric array")
        precision = builtins.max(precision, p)
    if is_complex:
        return array_type[1][precision]
    else:
        return array_type[0][precision]


# ### histograms ###


def histogram(
    a: ArrayLike,
    bins: ArrayLike = 10,
    range=None,
    normed=None,
    weights: ArrayLike | None = None,
    density=None,
):
    if normed is not None:
        raise ValueError("normed argument is deprecated, use density= instead")

    if weights is not None and weights.dtype.is_complex:
        raise NotImplementedError("complex weights histogram.")

    is_a_int = not (a.dtype.is_floating_point or a.dtype.is_complex)
    is_w_int = weights is None or not weights.dtype.is_floating_point
    if is_a_int:
        a = a.double()

    if weights is not None:
        weights = _util.cast_if_needed(weights, a.dtype)

    if isinstance(bins, torch.Tensor):
        if bins.ndim == 0:
            # bins was a single int
            bins = operator.index(bins)
        else:
            bins = _util.cast_if_needed(bins, a.dtype)

    if range is None:
        h, b = torch.histogram(a, bins, weight=weights, density=bool(density))
    else:
        h, b = torch.histogram(
            a, bins, range=range, weight=weights, density=bool(density)
        )

    if not density and is_w_int:
        h = h.long()
    if is_a_int:
        b = b.long()

    return h, b


def histogram2d(
    x,
    y,
    bins=10,
    range: ArrayLike | None = None,
    normed=None,
    weights: ArrayLike | None = None,
    density=None,
):
    # vendored from https://github.com/numpy/numpy/blob/v1.24.0/numpy/lib/twodim_base.py#L655-L821
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    try:
        N = len(bins)
    except TypeError:
        N = 1

    if N != 1 and N != 2:
        bins = [bins, bins]

    h, e = histogramdd((x, y), bins, range, normed, weights, density)

    return h, e[0], e[1]


def histogramdd(
    sample,
    bins=10,
    range: ArrayLike | None = None,
    normed=None,
    weights: ArrayLike | None = None,
    density=None,
):
    # have to normalize manually because `sample` interpretation differs
    # for a list of lists and a 2D array
    if normed is not None:
        raise ValueError("normed argument is deprecated, use density= instead")

    from ._normalizations import normalize_array_like, normalize_seq_array_like

    if isinstance(sample, (list, tuple)):
        sample = normalize_array_like(sample).T
    else:
        sample = normalize_array_like(sample)

    sample = torch.atleast_2d(sample)

    if not (sample.dtype.is_floating_point or sample.dtype.is_complex):
        sample = sample.double()

    # bins is either an int, or a sequence of ints or a sequence of arrays
    bins_is_array = not (
        isinstance(bins, int) or builtins.all(isinstance(b, int) for b in bins)
    )
    if bins_is_array:
        bins = normalize_seq_array_like(bins)
        bins_dtypes = [b.dtype for b in bins]
        bins = [_util.cast_if_needed(b, sample.dtype) for b in bins]

    if range is not None:
        range = range.flatten().tolist()

    if weights is not None:
        # range=... is required : interleave min and max values per dimension
        mm = sample.aminmax(dim=0)
        range = torch.cat(mm).reshape(2, -1).T.flatten()
        range = tuple(range.tolist())
        weights = _util.cast_if_needed(weights, sample.dtype)
        w_kwd = {"weight": weights}
    else:
        w_kwd = {}

    h, b = torch.histogramdd(sample, bins, range, density=bool(density), **w_kwd)

    if bins_is_array:
        b = [_util.cast_if_needed(bb, dtyp) for bb, dtyp in zip(b, bins_dtypes)]

    return h, b


# ### odds and ends


def min_scalar_type(a: ArrayLike, /):
    # https://github.com/numpy/numpy/blob/maintenance/1.24.x/numpy/core/src/multiarray/convert_datatype.c#L1288

    from ._dtypes import DType

    if a.numel() > 1:
        # numpy docs: "For non-scalar array a, returns the vector's dtype unmodified."
        return DType(a.dtype)

    if a.dtype == torch.bool:
        dtype = torch.bool

    elif a.dtype.is_complex:
        fi = torch.finfo(torch.float32)
        fits_in_single = a.dtype == torch.complex64 or (
            fi.min <= a.real <= fi.max and fi.min <= a.imag <= fi.max
        )
        dtype = torch.complex64 if fits_in_single else torch.complex128

    elif a.dtype.is_floating_point:
        for dt in [torch.float16, torch.float32, torch.float64]:
            fi = torch.finfo(dt)
            if fi.min <= a <= fi.max:
                dtype = dt
                break
    else:
        # must be integer
        for dt in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]:
            # Prefer unsigned int where possible, as numpy does.
            ii = torch.iinfo(dt)
            if ii.min <= a <= ii.max:
                dtype = dt
                break

    return DType(dtype)


def pad(array: ArrayLike, pad_width: ArrayLike, mode="constant", **kwargs):
    if mode != "constant":
        raise NotImplementedError
    value = kwargs.get("constant_values", 0)
    # `value` must be a python scalar for torch.nn.functional.pad
    typ = _dtypes_impl.python_type_for_torch(array.dtype)
    value = typ(value)

    pad_width = torch.broadcast_to(pad_width, (array.ndim, 2))
    pad_width = torch.flip(pad_width, (0,)).flatten()

    return torch.nn.functional.pad(array, tuple(pad_width), value=value)
<|endoftext|># mypy: ignore-errors

"""Wrapper to mimic (parts of) np.random API surface.

NumPy has strict guarantees on reproducibility etc; here we don't give any.

Q: default dtype is float64 in numpy

"""

from __future__ import annotations

import functools
from math import sqrt

import torch

from . import _dtypes_impl, _util
from ._normalizations import array_or_scalar, ArrayLike, normalizer


__all__ = [
    "seed",
    "random_sample",
    "sample",
    "random",
    "rand",
    "randn",
    "normal",
    "choice",
    "randint",
    "shuffle",
    "uniform",
]


def use_numpy_random():
    # local import to avoid ref cycles
    import torch._dynamo.config as config

    return config.use_numpy_random_stream


def deco_stream(func):
    @functools.wraps(func)
    def inner(*args, **kwds):
        if not use_numpy_random():
            return func(*args, **kwds)
        else:
            import numpy

            from ._ndarray import ndarray

            f = getattr(numpy.random, func.__name__)

            # numpy funcs accept numpy ndarrays, unwrap
            args = tuple(
                arg.tensor.numpy() if isinstance(arg, ndarray) else arg for arg in args
            )
            kwds = {
                key: val.tensor.numpy() if isinstance(val, ndarray) else val
                for key, val in kwds.items()
            }

            value = f(*args, **kwds)

            # `value` can be either numpy.ndarray or python scalar (or None)
            if isinstance(value, numpy.ndarray):
                value = ndarray(torch.as_tensor(value))

            return value

    return inner


@deco_stream
def seed(seed=None):
    if seed is not None:
        torch.random.manual_seed(seed)


@deco_stream
def random_sample(size=None):
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).uniform_()
    return array_or_scalar(values, return_scalar=size == ())


def rand(*size):
    if size == ():
        size = None
    return random_sample(size)


sample = random_sample
random = random_sample


@deco_stream
def uniform(low=0.0, high=1.0, size=None):
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).uniform_(low, high)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def randn(*size):
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.randn(size, dtype=dtype)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def normal(loc=0.0, scale=1.0, size=None):
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).normal_(loc, scale)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def shuffle(x):
    # no @normalizer because we do not cast e.g. lists to tensors
    from ._ndarray import ndarray

    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, ndarray):
        tensor = x.tensor
    else:
        raise NotImplementedError("We do not random.shuffle lists in-place")

    perm = torch.randperm(tensor.shape[0])
    xp = tensor[perm]
    tensor.copy_(xp)


@deco_stream
def randint(low, high=None, size=None):
    if size is None:
        size = ()
    if not isinstance(size, (tuple, list)):
        size = (size,)
    if high is None:
        low, high = 0, low
    values = torch.randint(low, high, size=size)
    return array_or_scalar(values, int, return_scalar=size == ())


@deco_stream
@normalizer
def choice(a: ArrayLike, size=None, replace=True, p: ArrayLike | None = None):
    # https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
    if a.numel() == 1:
        a = torch.arange(a)

    # TODO: check a.dtype is integer -- cf np.random.choice(3.4) which raises

    # number of draws
    if size is None:
        num_el = 1
    elif _util.is_sequence(size):
        num_el = 1
        for el in size:
            num_el *= el
    else:
        num_el = size

    # prepare the probabilities
    if p is None:
        p = torch.ones_like(a) / a.shape[0]

    # cf https://github.com/numpy/numpy/blob/main/numpy/random/mtrand.pyx#L973
    atol = sqrt(torch.finfo(p.dtype).eps)
    if abs(p.sum() - 1.0) > atol:
        raise ValueError("probabilities do not sum to 1.")

    # actually sample
    indices = torch.multinomial(p, num_el, replacement=replace)

    if _util.is_sequence(size):
        indices = indices.reshape(size)

    samples = a[indices]

    return samples
<|endoftext|># mypy: ignore-errors

from __future__ import annotations

import torch

from . import _binary_ufuncs_impl, _dtypes_impl, _unary_ufuncs_impl, _util
from ._normalizations import (
    ArrayLike,
    ArrayLikeOrScalar,
    CastingModes,
    DTypeLike,
    normalizer,
    NotImplementedType,
    OutArray,
)


def _ufunc_postprocess(result, out, casting):
    if out is not None:
        result = _util.typecast_tensor(result, out.dtype.torch_dtype, casting)
        result = torch.broadcast_to(result, out.shape)
    return result


# ############# Binary ufuncs ######################

_binary = [
    name
    for name in dir(_binary_ufuncs_impl)
    if not name.startswith("_") and name not in ["torch", "matmul", "divmod", "ldexp"]
]


NEP50_FUNCS = (
    "add",
    "subtract",
    "multiply",
    "floor_divide",
    "true_divide",
    "divide",
    "remainder",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "hypot",
    "arctan2",
    "logaddexp",
    "logaddexp2",
    "heaviside",
    "copysign",
    "fmax",
    "minimum",
    "fmin",
    "maximum",
    "fmod",
    "gcd",
    "lcm",
    "pow",
)


def deco_binary_ufunc(torch_func):
    """Common infra for binary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """

    @normalizer
    def wrapped(
        x1: ArrayLikeOrScalar,
        x2: ArrayLikeOrScalar,
        /,
        out: OutArray | None = None,
        *,
        where: NotImplementedType = True,
        casting: CastingModes | None = "same_kind",
        order: NotImplementedType = "K",
        dtype: DTypeLike | None = None,
        subok: NotImplementedType = False,
        signature: NotImplementedType = None,
        extobj: NotImplementedType = None,
    ):
        if dtype is not None:

            def cast(x, dtype):
                if isinstance(x, torch.Tensor):
                    return _util.typecast_tensor(x, dtype, casting)
                else:
                    return torch.as_tensor(x, dtype=dtype)

            x1 = cast(x1, dtype)
            x2 = cast(x2, dtype)
        elif isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            dtype = _dtypes_impl.result_type_impl(x1, x2)
            x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)
        else:
            x1, x2 = _dtypes_impl.nep50_to_tensors(
                x1, x2, torch_func.__name__ in NEP50_FUNCS, torch_func.__name__
            )

        result = torch_func(x1, x2)

        return _ufunc_postprocess(result, out, casting)

    wrapped.__qualname__ = torch_func.__name__
    wrapped.__name__ = torch_func.__name__

    return wrapped


# matmul's signature is _slightly_ different from other ufuncs:
# - no where=...
# - additional axis=..., axes=...
# - no NEP50 scalars in or out
@normalizer
def matmul(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    out: OutArray | None = None,
    *,
    casting: CastingModes | None = "same_kind",
    order: NotImplementedType = "K",
    dtype: DTypeLike | None = None,
    subok: NotImplementedType = False,
    signature: NotImplementedType = None,
    extobj: NotImplementedType = None,
    axes: NotImplementedType = None,
    axis: NotImplementedType = None,
):
    if dtype is None:
        dtype = _dtypes_impl.result_type_impl(x1, x2)
    x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)

    result = _binary_ufuncs_impl.matmul(x1, x2)

    result = _ufunc_postprocess(result, out, casting)
    return result


# ldexp casting is special : the dtype of the result == dtype of the 1st arg
@normalizer
def ldexp(
    x1: ArrayLikeOrScalar,
    x2: ArrayLikeOrScalar,
    /,
    out: OutArray | None = None,
    *,
    where: NotImplementedType = True,
    casting: CastingModes | None = "same_kind",
    order: NotImplementedType = "K",
    dtype: DTypeLike | None = None,
    subok: NotImplementedType = False,
    signature: NotImplementedType = None,
    extobj: NotImplementedType = None,
):
    if dtype is not None:
        if isinstance(x1, torch.Tensor):
            x1 = _util.typecast_tensor(x1, dtype, casting)
        else:
            x1 = torch.as_tensor(x1, dtype=dtype)
    else:
        if not isinstance(x1, torch.Tensor):
            x1 = torch.as_tensor(x1)
            x1 = _util.cast_int_to_float(x1)

    x2 = torch.as_tensor(x2)
    # the second arg must be integer
    if _dtypes_impl._category(x2.dtype) != 1:
        raise ValueError("ldexp 2nd arg must be integer")

    result = _binary_ufuncs_impl.ldexp(x1, x2)

    if x1.dtype == torch.float16:
        # torch.ldexp(f16, int) -> f32, undo it
        result = result.to(torch.float16)

    return _ufunc_postprocess(result, out, casting)


# nin=2, nout=2
@normalizer
def divmod(
    x1: ArrayLike,
    x2: ArrayLike,
    out1: OutArray | None = None,
    out2: OutArray | None = None,
    /,
    out: tuple[OutArray | None, OutArray | None] = (None, None),
    *,
    where: NotImplementedType = True,
    casting: CastingModes | None = "same_kind",
    order: NotImplementedType = "K",
    dtype: DTypeLike | None = None,
    subok: NotImplementedType = False,
    signature: NotImplementedType = None,
    extobj: NotImplementedType = None,
):
    # make sure we either have no out arrays at all, or there is either
    # out1, out2, or out=tuple, but not both
    num_outs = sum(x is not None for x in [out1, out2])
    if num_outs == 1:
        raise ValueError("both out1 and out2 need to be provided")
    elif num_outs == 2:
        o1, o2 = out
        if o1 is not None or o2 is not None:
            raise TypeError(
                "cannot specify 'out' as both a positional and keyword argument"
            )
    else:
        out1, out2 = out

    if dtype is None:
        dtype = _dtypes_impl.result_type_impl(x1, x2)
    x1, x2 = _util.typecast_tensors((x1, x2), dtype, casting)

    quot, rem = _binary_ufuncs_impl.divmod(x1, x2)

    quot = _ufunc_postprocess(quot, out1, casting)
    rem = _ufunc_postprocess(rem, out2, casting)
    return quot, rem


#
# Attach ufuncs to this module, for a further export to the public namespace in __init__.py
#
for name in _binary:
    ufunc = getattr(_binary_ufuncs_impl, name)
    vars()[name] = deco_binary_ufunc(ufunc)


def modf(x, /, *args, **kwds):
    quot, rem = divmod(x, 1, *args, **kwds)
    return rem, quot


_binary = _binary + ["divmod", "modf", "matmul", "ldexp"]


# ############# Unary ufuncs ######################


_unary = [
    name
    for name in dir(_unary_ufuncs_impl)
    if not name.startswith("_") and name != "torch"
]


# these are ufunc(int) -> float
_fp_unary = [
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "cbrt",
    "cos",
    "cosh",
    "deg2rad",
    "degrees",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log10",
    "log1p",
    "log2",
    "rad2deg",
    "radians",
    "reciprocal",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
]


def deco_unary_ufunc(torch_func):
    """Common infra for unary ufuncs.

    Normalize arguments, sort out type casting, broadcasting and delegate to
    the pytorch functions for the actual work.
    """

    @normalizer
    def wrapped(
        x: ArrayLike,
        /,
        out: OutArray | None = None,
        *,
        where=True,
        casting: CastingModes | None = "same_kind",
        order="K",
        dtype: DTypeLike | None = None,
        subok: NotImplementedType = False,
        signature=None,
        extobj=None,
    ):
        if dtype is not None:
            x = _util.typecast_tensor(x, dtype, casting)

        if torch_func.__name__ in _fp_unary:
            x = _util.cast_int_to_float(x)

        result = torch_func(x)
        result = _ufunc_postprocess(result, out, casting)
        return result

    wrapped.__qualname__ = torch_func.__name__
    wrapped.__name__ = torch_func.__name__

    return wrapped


#
# Attach ufuncs to this module, for a further export to the public namespace in __init__.py
#
for name in _unary:
    ufunc = getattr(_unary_ufuncs_impl, name)
    vars()[name] = deco_unary_ufunc(ufunc)


__all__ = _binary + _unary  # noqa: PLE0605
<|endoftext|>"""Define analogs of numpy dtypes supported by pytorch.
Define the scalar types and supported dtypes and numpy <--> torch dtype mappings.
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, TypeGuard, TypeVar

import torch

from . import _dtypes_impl


if TYPE_CHECKING:
    from collections.abc import Callable

    from ._ndarray import ndarray


_T = TypeVar("_T")


# ### Scalar types ###


class generic:
    name = "generic"
    # Concrete scalar types below set these; declared here so the dtype machinery
    # can read them off any scalar type.
    typecode: str
    torch_dtype: torch.dtype

    def __new__(cls, value: object) -> ndarray:
        # NumPy scalars are modelled as 0-D arrays
        # so a call to np.float32(4) produces a 0-D array.

        from ._ndarray import asarray, ndarray

        if isinstance(value, str) and value in ["inf", "nan"]:
            value = {"inf": torch.inf, "nan": torch.nan}[value]

        if isinstance(value, ndarray):
            return value.astype(cls)
        else:
            return asarray(value, dtype=cls)


##################
# abstract types #
##################


class number(generic):
    name = "number"


class integer(number):
    name = "integer"


class inexact(number):
    name = "inexact"


class signedinteger(integer):
    name = "signedinteger"


class unsignedinteger(integer):
    name = "unsignedinteger"


class floating(inexact):
    name = "floating"


class complexfloating(inexact):
    name = "complexfloating"


_abstract_dtypes = [
    "generic",
    "number",
    "integer",
    "signedinteger",
    "unsignedinteger",
    "inexact",
    "floating",
    "complexfloating",
]

# ##### concrete types

# signed integers


class int8(signedinteger):
    name = "int8"
    typecode = "b"
    torch_dtype = torch.int8


class int16(signedinteger):
    name = "int16"
    typecode = "h"
    torch_dtype = torch.int16


class int32(signedinteger):
    name = "int32"
    typecode = "i"
    torch_dtype = torch.int32


class int64(signedinteger):
    name = "int64"
    typecode = "l"
    torch_dtype = torch.int64


# unsigned integers


class uint8(unsignedinteger):
    name = "uint8"
    typecode = "B"
    torch_dtype = torch.uint8


class uint16(unsignedinteger):
    name = "uint16"
    typecode = "H"
    torch_dtype = torch.uint16


class uint32(signedinteger):
    name = "uint32"
    typecode = "I"
    torch_dtype = torch.uint32


class uint64(signedinteger):
    name = "uint64"
    typecode = "L"
    torch_dtype = torch.uint64


# floating point


class float16(floating):
    name = "float16"
    typecode = "e"
    torch_dtype = torch.float16


class float32(floating):
    name = "float32"
    typecode = "f"
    torch_dtype = torch.float32


class float64(floating):
    name = "float64"
    typecode = "d"
    torch_dtype = torch.float64


class complex64(complexfloating):
    name = "complex64"
    typecode = "F"
    torch_dtype = torch.complex64


class complex128(complexfloating):
    name = "complex128"
    typecode = "D"
    torch_dtype = torch.complex128


class bool_(generic):
    name = "bool_"
    typecode = "?"
    torch_dtype = torch.bool


# name aliases
_name_aliases = {
    "intp": int64,
    "int_": int64,
    "intc": int32,
    "byte": int8,
    "short": int16,
    "longlong": int64,  # XXX: is this correct?
    "ulonglong": uint64,
    "ubyte": uint8,
    "half": float16,
    "single": float32,
    "double": float64,
    "float_": float64,
    "csingle": complex64,
    "singlecomplex": complex64,
    "cdouble": complex128,
    "cfloat": complex128,
    "complex_": complex128,
}
# We register float_ = float32 and so on
for name, obj in _name_aliases.items():
    vars()[name] = obj


# Replicate this NumPy-defined way of grouping scalar types,
# cf tests/core/test_scalar_methods.py
sctypes = {
    "int": [int8, int16, int32, int64],
    "uint": [uint8, uint16, uint32, uint64],
    "float": [float16, float32, float64],
    "complex": [complex64, complex128],
    "others": [bool_],
}


# Support mappings/functions

_names = {st.name: st for cat in sctypes for st in sctypes[cat]}
_typecodes = {st.typecode: st for cat in sctypes for st in sctypes[cat]}
_torch_dtypes = {st.torch_dtype: st for cat in sctypes for st in sctypes[cat]}


_aliases = {
    "u1": uint8,
    "i1": int8,
    "i2": int16,
    "i4": int32,
    "i8": int64,
    "b": int8,  # XXX: srsly?
    "f2": float16,
    "f4": float32,
    "f8": float64,
    "c8": complex64,
    "c16": complex128,
    # numpy-specific trailing underscore
    "bool_": bool_,
}


_python_types = {
    int: int64,
    float: float64,
    complex: complex128,
    builtins.bool: bool_,
    # also allow stringified names of python types
    int.__name__: int64,
    float.__name__: float64,
    complex.__name__: complex128,
    builtins.bool.__name__: bool_,
}


def sctype_from_string(s: object) -> type[generic]:
    """Normalize a string value: a type 'name' or a typecode or a width alias."""
    if s in _names:
        return _names[s]
    if s in _name_aliases:
        return _name_aliases[s]
    if s in _typecodes:
        return _typecodes[s]
    if s in _aliases:
        return _aliases[s]
    if s in _python_types:
        return _python_types[s]
    raise TypeError(f"data type {s!r} not understood")


def sctype_from_torch_dtype(torch_dtype: torch.dtype) -> type[generic]:
    return _torch_dtypes[torch_dtype]


# ### DTypes. ###


def dtype(arg: object) -> DType:
    if arg is None:
        arg = _dtypes_impl.default_dtypes().float_dtype
    return DType(arg)


class DType:
    _scalar_type: type[generic]

    def __init__(self, arg: object) -> None:
        # a pytorch object?
        if isinstance(arg, torch.dtype):
            sctype = _torch_dtypes[arg]
        elif isinstance(arg, torch.Tensor):
            sctype = _torch_dtypes[arg.dtype]
        # a scalar type?
        elif issubclass_(arg, generic):
            sctype = arg
        # a dtype already?
        elif isinstance(arg, DType):
            sctype = arg._scalar_type
        # a has a right attribute?
        elif hasattr(arg, "dtype"):
            sctype = arg.dtype._scalar_type
        else:
            sctype = sctype_from_string(arg)
        self._scalar_type = sctype

    @property
    def name(self) -> str:
        return self._scalar_type.name

    @property
    def type(self) -> type[generic]:
        return self._scalar_type

    @property
    def kind(self) -> str:
        # https://numpy.org/doc/stable/reference/generated/numpy.dtype.kind.html
        return _torch_dtypes[self.torch_dtype].name[0]

    @property
    def typecode(self) -> str:
        return self._scalar_type.typecode

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DType):
            return self._scalar_type == other._scalar_type
        try:
            other_instance = DType(other)
        except TypeError:
            return False
        return self._scalar_type == other_instance._scalar_type

    @property
    def torch_dtype(self) -> torch.dtype:
        return self._scalar_type.torch_dtype

    def __hash__(self) -> int:
        return hash(self._scalar_type.name)

    def __repr__(self) -> str:
        return f'dtype("{self.name}")'

    __str__ = __repr__

    @property
    def itemsize(self) -> int:
        elem = self.type(1)
        return elem.tensor.element_size()

    def __getstate__(self) -> builtins.type[generic]:
        return self._scalar_type

    def __setstate__(self, value: builtins.type[generic]) -> None:
        self._scalar_type = value


typecodes = {
    "All": "efdFDBbhil?",
    "AllFloat": "efdFD",
    "AllInteger": "Bbhil",
    "Integer": "bhil",
    "UnsignedInteger": "B",
    "Float": "efd",
    "Complex": "FD",
}


# ### Defaults and dtype discovery


def set_default_dtype(
    fp_dtype: object = "numpy", int_dtype: object = "numpy"
) -> Callable[[], _dtypes_impl.DefaultDTypes]:
    """Set the (global) defaults for fp, complex, and int dtypes.

    The complex dtype is inferred from the float (fp) dtype. It has
    a width at least twice the width of the float dtype,
    i.e., it's complex128 for float64 and complex64 for float32.

    Parameters
    ----------
    fp_dtype
        Allowed values are "numpy", "pytorch" or dtype_like things which
        can be converted into a DType instance.
        Default is "numpy" (i.e. float64).
    int_dtype
        Allowed values are "numpy", "pytorch" or dtype_like things which
        can be converted into a DType instance.
        Default is "numpy" (i.e. int64).

    Returns
    -------
    The old default dtype state: a namedtuple with attributes ``float_dtype``,
    ``complex_dtypes`` and ``int_dtype``. These attributes store *pytorch*
    dtypes.

    Notes
    ------------
    This function has a side effect: it sets the global state with the provided dtypes.

    The complex dtype has bit width of at least twice the width of the float
    dtype, i.e. it's complex128 for float64 and complex64 for float32.

    """
    if fp_dtype not in ["numpy", "pytorch"]:
        fp_dtype = dtype(fp_dtype).torch_dtype
    if int_dtype not in ["numpy", "pytorch"]:
        int_dtype = dtype(int_dtype).torch_dtype

    if fp_dtype == "numpy":
        float_dtype = torch.float64
    elif fp_dtype == "pytorch":
        float_dtype = torch.float32
    else:
        float_dtype = fp_dtype

    complex_dtype = {
        torch.float64: torch.complex128,
        torch.float32: torch.complex64,
        torch.float16: torch.complex64,
    }[float_dtype]

    if int_dtype in ["numpy", "pytorch"]:
        int_dtype = torch.int64

    new_defaults = _dtypes_impl.DefaultDTypes(
        float_dtype=float_dtype, complex_dtype=complex_dtype, int_dtype=int_dtype
    )

    # set the new global state and return the old state
    old_defaults = _dtypes_impl.default_dtypes
    # pyrefly: ignore[bad-assignment]  # TODO
    _dtypes_impl._default_dtypes = new_defaults
    return old_defaults


def issubclass_(arg: object, klass: type[_T]) -> TypeGuard[type[_T]]:
    # issubclass() raises TypeError if arg is not a class; guard explicitly so a
    # non-class arg returns False (and narrows arg to type[_T] on success).
    return isinstance(arg, type) and issubclass(arg, klass)


def issubdtype(arg1: object, arg2: object) -> bool:
    # cf https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numerictypes.py#L356-L420

    # We also accept strings even if NumPy doesn't as dtypes are serialized as their
    # string representation in dynamo's graph
    def str_to_abstract(t: object) -> object:
        if isinstance(t, str) and t in _abstract_dtypes:
            return globals()[t]
        return t

    arg1 = str_to_abstract(arg1)
    arg2 = str_to_abstract(arg2)

    if not issubclass_(arg1, generic):
        arg1 = dtype(arg1).type
    if not issubclass_(arg2, generic):
        arg2 = dtype(arg2).type
    return issubclass(arg1, arg2)


__all__ = ["dtype", "DType", "typecodes", "issubdtype", "set_default_dtype", "sctypes"]
__all__ += list(_names.keys())
__all__ += list(_name_aliases.keys())
__all__ += _abstract_dtypes  # noqa: PLE0605
<|endoftext|># mypy: ignore-errors

from . import fft, linalg, random
from ._dtypes import *  # noqa: F403
from ._funcs import *  # noqa: F403
from ._getlimits import finfo, iinfo
from ._ndarray import (
    array,
    asarray,
    ascontiguousarray,
    can_cast,
    from_dlpack,
    ndarray,
    newaxis,
    result_type,
)
from ._ufuncs import *  # noqa: F403
from ._util import AxisError, UFuncTypeError


from math import pi, e  # usort: skip


all = all  # noqa: PLW0127
alltrue = all

any = any  # noqa: PLW0127
sometrue = any

inf = float("inf")
nan = float("nan")

False_ = False
True_ = True
<|endoftext|>from __future__ import annotations

import functools
from typing import Concatenate, ParamSpec, TYPE_CHECKING, TypeVar

import torch

from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, normalizer


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


_P = ParamSpec("_P")
_R = TypeVar("_R")


def upcast(
    func: Callable[Concatenate[torch.Tensor, _P], _R],
) -> Callable[Concatenate[torch.Tensor, _P], _R]:
    """NumPy fft casts inputs to 64 bit and *returns 64-bit results*."""

    @functools.wraps(func)
    def wrapped(tensor: torch.Tensor, *args: _P.args, **kwds: _P.kwargs) -> _R:
        target_dtype = (
            _dtypes_impl.default_dtypes().complex_dtype
            if tensor.is_complex()
            else _dtypes_impl.default_dtypes().float_dtype
        )
        tensor = _util.cast_if_needed(tensor, target_dtype)
        return func(tensor, *args, **kwds)

    return wrapped


@normalizer
@upcast
def fft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    return torch.fft.fft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ifft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    return torch.fft.ifft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def rfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    return torch.fft.rfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def irfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    return torch.fft.irfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def fftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    return torch.fft.fftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    return torch.fft.ifftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    return torch.fft.rfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfftn(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: str | None = None,
) -> torch.Tensor:
    return torch.fft.irfftn(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def fft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    return torch.fft.fft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def ifft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    return torch.fft.ifft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def rfft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    return torch.fft.rfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def irfft2(
    a: ArrayLike,
    s: Sequence[int] | None = None,
    axes: Sequence[int] = (-2, -1),
    norm: str | None = None,
) -> torch.Tensor:
    return torch.fft.irfft2(a, s, dim=axes, norm=norm)


@normalizer
@upcast
def hfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    return torch.fft.hfft(a, n, dim=axis, norm=norm)


@normalizer
@upcast
def ihfft(
    a: ArrayLike, n: int | None = None, axis: int = -1, norm: str | None = None
) -> torch.Tensor:
    return torch.fft.ihfft(a, n, dim=axis, norm=norm)


@normalizer
def fftfreq(n: int, d: float = 1.0) -> torch.Tensor:
    return torch.fft.fftfreq(n, d)


@normalizer
def rfftfreq(n: int, d: float = 1.0) -> torch.Tensor:
    return torch.fft.rfftfreq(n, d)


@normalizer
def fftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> torch.Tensor:
    return torch.fft.fftshift(x, axes)


@normalizer
def ifftshift(x: ArrayLike, axes: int | Sequence[int] | None = None) -> torch.Tensor:
    return torch.fft.ifftshift(x, axes)
<|endoftext|># mypy: ignore-errors

"""Dtypes/scalar type implementations with torch dtypes.

Here `dtype` is always a torch.dtype, this module knows nothing about
scalar types, wrapper dtypes or anything like that. PyTorch only.
"""

from collections import namedtuple

import torch


# defaults : mimic NumPy, allow user control
DefaultDTypes = namedtuple(
    "DefaultDTypes", ["float_dtype", "complex_dtype", "int_dtype"]
)

# a global state
# We set it the first time we call default_dtypes() to avoid importing
# torch._dynamo.config and create a circular reference
_default_dtypes = None


def default_dtypes():
    global _default_dtypes
    if _default_dtypes is None:
        import torch._dynamo.config as config

        _default_dtypes = DefaultDTypes(
            float_dtype=getattr(torch, config.numpy_default_float),
            complex_dtype=getattr(torch, config.numpy_default_complex),
            int_dtype=getattr(torch, config.numpy_default_int),
        )
        if not isinstance(_default_dtypes.float_dtype, torch.dtype):
            raise AssertionError(
                f"float_dtype must be a torch.dtype, got {type(_default_dtypes.float_dtype)}"
            )
        if not isinstance(_default_dtypes.complex_dtype, torch.dtype):
            raise AssertionError(
                f"complex_dtype must be a torch.dtype, got {type(_default_dtypes.complex_dtype)}"
            )
        if not isinstance(_default_dtypes.int_dtype, torch.dtype):
            raise AssertionError(
                f"int_dtype must be a torch.dtype, got {type(_default_dtypes.int_dtype)}"
            )
    return _default_dtypes


def get_default_dtype_for(dtype):
    """Default scalar type given sctype category."""
    if dtype == torch.bool:
        return dtype
    if dtype.is_complex:
        return default_dtypes().complex_dtype
    if dtype.is_floating_point:
        return default_dtypes().float_dtype
    # else, it must be (some) integer
    return default_dtypes().int_dtype


from . import _casting_dicts as _cd


def can_cast_impl(from_torch_dtype, to_torch_dtype, casting):
    return _cd._can_cast_dict[casting][from_torch_dtype][to_torch_dtype]


def result_type_impl(*tensors):
    # NB: torch dtypes here
    dtyp = tensors[0].dtype
    if len(tensors) == 1:
        return dtyp

    for curr in tensors[1:]:
        dtyp = _cd._result_type_dict[dtyp][curr.dtype]

    return dtyp


def python_type_for_torch(dtyp):
    """Get a python scalar type a torch dtype"""
    if dtyp.is_floating_point:
        typ = float
    elif dtyp.is_complex:
        typ = complex
    elif dtyp == torch.bool:
        typ = bool
    else:
        typ = int
    return typ


# ### NEP 50 helpers ###

_SCALAR_TYPES = (int, bool, float, complex)

_SCALAR_AND_SYMBOLIC_TYPES = (
    *_SCALAR_TYPES,
    torch.SymInt,
    torch.SymFloat,
    torch.SymBool,
)

_NEP50_FUNCS_TENSOR_ONLY = (
    "minimum",
    "maximum",
    "logaddexp",
    "logaddexp2",
    "lcm",
    "gcd",
    "hypot",
    "heaviside",
    "fmod",
    "fmin",
    "fmax",
    "copysign",
    "arctan2",
)


def is_scalar(x):
    return isinstance(x, _SCALAR_TYPES)


def is_scalar_or_symbolic(x):
    return isinstance(x, _SCALAR_AND_SYMBOLIC_TYPES)


def _dtype_for_scalar(py_type):
    return {
        bool: torch.bool,
        torch.SymBool: torch.bool,
        int: torch.int64,
        torch.SymInt: torch.int64,
        float: torch.float64,
        torch.SymFloat: torch.float64,
        complex: torch.complex128,
    }[py_type]


def _dtype_for_scalar_or_tensor(x):
    return x.dtype if isinstance(x, torch.Tensor) else _dtype_for_scalar(type(x))


def is_float_or_fp_tensor(x):
    return _dtype_for_scalar_or_tensor(x).is_floating_point


def is_complex_or_complex_tensor(x):
    return _dtype_for_scalar_or_tensor(x).is_complex


def _category(dtype):
    return {
        torch.bool: 0,
        torch.SymBool: 0,
        # int
        torch.uint8: 1,
        torch.int8: 1,
        torch.int16: 1,
        torch.int32: 1,
        torch.int64: 1,
        torch.SymInt: 1,
        # float
        torch.float16: 2,
        torch.float32: 2,
        torch.float64: 2,
        torch.SymFloat: 2,
        # complex
        torch.complex64: 3,
        torch.complex128: 3,
    }[dtype]


def nep50_to_tensors(x1, x2, handle_weaks, function_name):
    """If either of inputs is a python scalar, type-promote with NEP 50."""

    def to_tensor(scalar, dtype=None):
        if dtype is None:
            dtype = _dtype_for_scalar(type(scalar))
            dtype = get_default_dtype_for(dtype)
        return torch.as_tensor(scalar, dtype=dtype)

    x1_is_weak = not isinstance(x1, torch.Tensor)
    x2_is_weak = not isinstance(x2, torch.Tensor)
    if not handle_weaks or (x1_is_weak and x2_is_weak):
        x1 = to_tensor(x1) if x1_is_weak else x1
        x2 = to_tensor(x2) if x2_is_weak else x2
        return x1, x2

    # scalar <op> tensor: NEP 50
    if x1_is_weak == x2_is_weak:
        raise AssertionError(
            f"Expected exactly one weak type, got x1_is_weak={x1_is_weak}, x2_is_weak={x2_is_weak}"
        )

    weak, not_weak = (x1, x2) if x1_is_weak else (x2, x1)

    # find the dtype for the weak's type
    weak_dtype = _dtype_for_scalar(type(weak))

    cat_weak = _category(weak_dtype)
    cat_not_weak = _category(not_weak.dtype)

    dt = not_weak.dtype if cat_weak <= cat_not_weak else None

    # special-case complex + float32
    if weak_dtype.is_complex and not_weak.dtype == torch.float32:
        dt = torch.complex64

    # detect overflows: in PyTorch, uint8(-1) wraps around to 255,
    # while NEP50 mandates an exception.
    #
    # Note that we only check if each element of the binop overflows,
    # not the result. Consider, e.g. `uint8(100) + 200`. Operands are OK
    # in uint8, but the result overflows and wrap around 255.
    # Numpy emits a RuntimeWarning, PyTorch does not, and we do not either.
    if cat_weak == 1 and cat_not_weak == 1:
        # integers
        iinfo = torch.iinfo(not_weak.dtype)
        if not (iinfo.min <= weak <= iinfo.max):
            raise OverflowError(
                f"Python integer {weak} out of bounds for {not_weak.dtype}"
            )
    if weak_dtype != dt or function_name in _NEP50_FUNCS_TENSOR_ONLY:
        # finally, can make `weak` into a 0D tensor, if both parameters are required to be tensor.
        weak = to_tensor(weak, dt)

    return (weak, not_weak) if x1_is_weak else (not_weak, weak)
<|endoftext|># mypy: ignore-errors

import inspect
import itertools

from . import _funcs_impl, _reductions_impl
from ._normalizations import normalizer


# _funcs_impl.py contains functions which mimic NumPy's eponymous equivalents,
# and consume/return PyTorch tensors/dtypes.
# They are also type annotated.
# Pull these functions from _funcs_impl and decorate them with @normalizer, which
# - Converts any input `np.ndarray`, `torch._numpy.ndarray`, list of lists, Python scalars, etc into a `torch.Tensor`.
# - Maps NumPy dtypes to PyTorch dtypes
# - If the input to the `axis` kwarg is an ndarray, it maps it into a tuple
# - Implements the semantics for the `out=` arg
# - Wraps back the outputs into `torch._numpy.ndarrays`


def _public_functions(mod):
    def is_public_function(f):
        return inspect.isfunction(f) and not f.__name__.startswith("_")

    return inspect.getmembers(mod, is_public_function)


# We fill in __all__ in the loop below
__all__ = []

# decorate implementer functions with argument normalizers and export to the top namespace
for name, func in itertools.chain(
    _public_functions(_funcs_impl), _public_functions(_reductions_impl)
):
    if name in ["percentile", "quantile", "median"]:
        decorated = normalizer(func, promote_scalar_result=True)
    elif name == "einsum":
        # normalized manually
        decorated = func
    else:
        decorated = normalizer(func)

    decorated.__qualname__ = name
    decorated.__name__ = name
    vars()[name] = decorated
    __all__.append(name)


"""
Vendored objects from numpy.lib.index_tricks
"""


class IndexExpression:
    """
    Written by Konrad Hinsen <hinsen@cnrs-orleans.fr>
    last revision: 1999-7-23

    Cosmetic changes by T. Oliphant 2001
    """

    def __init__(self, maketuple):
        self.maketuple = maketuple

    def __getitem__(self, item):
        if self.maketuple and not isinstance(item, tuple):
            return (item,)
        else:
            return item


index_exp = IndexExpression(maketuple=True)
s_ = IndexExpression(maketuple=False)


__all__ += ["index_exp", "s_"]
<|endoftext|>""" "Normalize" arguments: convert array_likes to tensors, dtypes to torch dtypes and so on."""

from __future__ import annotations

import functools
import inspect
import operator
import typing
from typing import TYPE_CHECKING

import torch

from . import _dtypes, _dtypes_impl, _util


if TYPE_CHECKING:
    from collections.abc import Callable

    from ._ndarray import ndarray


# These names are used both as static type annotations and as runtime markers:
# the `normalizer` decorator dispatches on the *string* form of each parameter's
# annotation (e.g. "ArrayLike", "OutArray | None"). Because `from __future__
# import annotations` is active everywhere these are used, the annotations are
# never evaluated, so the static type below is what each parameter resolves to
# *after* normalization runs (e.g. an ArrayLike argument is a torch.Tensor by the
# time the implementer body sees it).
ArrayLike: typing.TypeAlias = torch.Tensor
Scalar = int | float | complex | bool
ArrayLikeOrScalar = ArrayLike | Scalar

# DTypeLike normalizes to a torch.dtype (or None when no dtype is given).
DTypeLike: typing.TypeAlias = "torch.dtype | None"
# AxisLike is an axis specification: a single axis, a tuple of axes, or None.
AxisLike: typing.TypeAlias = "int | tuple[int, ...] | None"
# NDArray normalizes to the wrapped tensor.
NDArray: typing.TypeAlias = torch.Tensor
# CastingModes is one of the numpy casting strings.
CastingModes: typing.TypeAlias = str
# KeepDims is the keepdims flag; handled entirely by the normalizer.
KeepDims: typing.TypeAlias = bool

# OutArray is to annotate the out= array argument.
#
# This one is special is several respects:
# First, It needs to be an NDArray, and we need to preserve the `result is out`
# semantics. Therefore, we cannot just extract the Tensor from the out array.
# So we never pass the out array to implementer functions and handle it in the
# `normalizer` below.
# Second, the out= argument can be either keyword or positional argument, and
# as a positional arg, it can be anywhere in the signature.
# To handle all this, we define a special `OutArray` annotation and dispatch on it.
#
OutArray: typing.TypeAlias = "ndarray"

# NotImplementedType marks unsupported parameters; the normalizer raises unless
# the argument equals the (always None) default, so the value is never used.
NotImplementedType: typing.TypeAlias = None


def normalize_array_like(
    x: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.Tensor:
    from ._ndarray import asarray

    return asarray(x).tensor


def normalize_array_like_or_scalar(
    x: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.Tensor | Scalar:
    if _dtypes_impl.is_scalar_or_symbolic(x):
        return x  # pyrefly: ignore[bad-return]
    return normalize_array_like(x, parm)  # codespell:ignore


def normalize_optional_array_like_or_scalar(
    x: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.Tensor | Scalar | None:
    if x is None:
        return None
    return normalize_array_like_or_scalar(x, parm)  # codespell:ignore


def normalize_optional_array_like(
    x: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.Tensor | None:
    # This explicit normalizer is needed because otherwise normalize_array_like
    # does not run for a parameter annotated as Optional[ArrayLike]
    return None if x is None else normalize_array_like(x, parm)  # codespell:ignore


def normalize_seq_array_like(
    x: typing.Iterable[object],
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> tuple[torch.Tensor, ...]:
    return tuple(normalize_array_like(value) for value in x)


def normalize_dtype(
    dtype: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.dtype | None:
    # cf _decorators.dtype_to_torch
    torch_dtype = None
    if dtype is not None:
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
    return torch_dtype


def normalize_not_implemented(
    arg: object,
    parm: inspect.Parameter,  # codespell:ignore
) -> None:
    if arg != parm.default:  # codespell:ignore
        raise NotImplementedError(
            f"'{parm.name}' parameter is not supported."  # codespell:ignore
        )


def normalize_axis_like(
    arg: AxisLike | ndarray,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> AxisLike:
    from ._ndarray import ndarray

    if isinstance(arg, ndarray):
        return operator.index(arg)
    return arg


def normalize_ndarray(
    arg: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> torch.Tensor | None:
    # check the arg is an ndarray, extract its tensor attribute
    if arg is None:
        return arg

    from ._ndarray import ndarray

    if not isinstance(arg, ndarray):
        name = parm.name if parm is not None else "argument"  # codespell:ignore
        raise TypeError(f"'{name}' must be an array")
    return arg.tensor


def normalize_outarray(
    arg: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> ndarray | None:
    # almost normalize_ndarray, only return the array, not its tensor
    if arg is None:
        return arg
    from ._ndarray import ndarray

    # Dynamo can pass torch tensors as out arguments,
    # wrap it in an ndarray before processing
    if isinstance(arg, torch.Tensor):
        arg = ndarray(arg)

    if not isinstance(arg, ndarray):
        name = parm.name if parm is not None else "argument"  # codespell:ignore
        raise TypeError(f"'{name}' must be an array")
    return arg


def normalize_casting(
    arg: object,
    parm: inspect.Parameter | None = None,  # codespell:ignore
) -> CastingModes:
    valid = ("no", "equiv", "safe", "same_kind", "unsafe")
    if not isinstance(arg, str) or arg not in valid:
        raise ValueError(
            f"casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe' (got '{arg}')"
        )
    return arg


# The registered normalizers have intentionally heterogeneous signatures
# (e.g. normalize_seq_array_like takes an iterable), so the common callable
# type erases the parameter list.
normalizers: dict[str, Callable[..., object]] = {
    "ArrayLike": normalize_array_like,
    "ArrayLikeOrScalar": normalize_array_like_or_scalar,
    "Optional[ArrayLike]": normalize_optional_array_like,
    "ArrayLike | None": normalize_optional_array_like,
    "Sequence[ArrayLike]": normalize_seq_array_like,
    "Optional[ArrayLikeOrScalar]": normalize_optional_array_like_or_scalar,
    "ArrayLikeOrScalar | None": normalize_optional_array_like_or_scalar,
    "Optional[NDArray]": normalize_ndarray,
    "NDArray | None": normalize_ndarray,
    "Optional[OutArray]": normalize_outarray,
    "OutArray | None": normalize_outarray,
    "NDArray": normalize_ndarray,
    "Optional[DTypeLike]": normalize_dtype,
    "DTypeLike | None": normalize_dtype,
    "AxisLike": normalize_axis_like,
    "NotImplementedType": normalize_not_implemented,
    "Optional[CastingModes]": normalize_casting,
    "CastingModes | None": normalize_casting,
}


def maybe_normalize(arg: object, parm: inspect.Parameter) -> object:  # codespell:ignore
    """Normalize arg if a normalizer is registered."""
    normalizer = normalizers.get(parm.annotation)  # codespell:ignore
    return normalizer(arg, parm) if normalizer else arg  # codespell:ignore


# ### Return value helpers ###


def maybe_copy_to(
    out: ndarray | None, result: object, promote_scalar_result: bool = False
) -> object:
    # NB: here out is either an ndarray or None
    if out is None:
        return result
    elif isinstance(result, torch.Tensor):
        if result.shape != out.shape:
            can_fit = result.numel() == 1 and out.ndim == 0
            if promote_scalar_result and can_fit:
                result = result.squeeze()
            else:
                raise ValueError(
                    f"Bad size of the out array: out.shape = {out.shape}"
                    f" while result.shape = {result.shape}."
                )
        out.tensor.copy_(result)
        return out
    elif isinstance(result, (tuple, list)):
        return type(result)(
            maybe_copy_to(o, r, promote_scalar_result) for o, r in zip(out, result)
        )
    else:
        raise AssertionError  # We should never hit this path


def wrap_tensors(result: object) -> object:
    from ._ndarray import ndarray

    if isinstance(result, torch.Tensor):
        return ndarray(result)
    elif isinstance(result, (tuple, list)):
        result = type(result)(wrap_tensors(x) for x in result)
    return result


def array_or_scalar(
    values: torch.Tensor,
    py_type: type[int | float | complex | bool] = float,
    return_scalar: bool = False,
) -> int | float | complex | bool | ndarray:
    if return_scalar:
        return py_type(values.item())
    else:
        from ._ndarray import ndarray

        return ndarray(values)


# ### The main decorator to normalize arguments / postprocess the output ###

_P = typing.ParamSpec("_P")
_R = typing.TypeVar("_R")


@typing.overload
def normalizer(_func: Callable[_P, _R]) -> Callable[_P, _R]: ...


@typing.overload
def normalizer(
    _func: None = None, *, promote_scalar_result: bool = False
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...


def normalizer(
    _func: Callable[_P, _R] | None = None, *, promote_scalar_result: bool = False
) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    def normalizer_inner(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def wrapped(*args: _P.args, **kwds: _P.kwargs) -> _R:
            sig = inspect.signature(func)
            params = sig.parameters
            first_param = next(iter(params.values()))

            # NumPy's API does not have positional args before variadic positional args
            norm_args: tuple[object, ...]
            if first_param.kind == inspect.Parameter.VAR_POSITIONAL:
                norm_args = tuple(maybe_normalize(arg, first_param) for arg in args)
            else:
                # NB: extra unknown arguments: pass through, will raise in func(*args) below
                norm_args = (
                    tuple(
                        maybe_normalize(arg, parm)  # codespell:ignore
                        for arg, parm in zip(args, params.values())  # codespell:ignore
                    )
                    + args[len(params.values()) :]
                )

            norm_kwds = {
                name: maybe_normalize(arg, params[name]) if name in params else arg
                for name, arg in kwds.items()
            }

            # The normalizer rebuilds args/kwds, so they are no longer the
            # ParamSpec-tracked *args/**kwargs; the call is correct at runtime.
            # pyrefly: ignore[invalid-param-spec]
            result: object = func(*norm_args, **norm_kwds)

            # keepdims
            bound_args = None
            if "keepdims" in params and params["keepdims"].annotation == "KeepDims":
                # keepdims can be in any position so we need sig.bind
                bound_args = sig.bind(*norm_args, **norm_kwds).arguments
                if bound_args.get("keepdims", False):
                    # In this case the first arg is the initial tensor and
                    # the second arg is (optionally) the axis
                    tensor = norm_args[0]
                    axis = bound_args.get("axis")
                    if not isinstance(tensor, torch.Tensor):
                        raise AssertionError("keepdims requires a tensor first arg")
                    # pyrefly: ignore[bad-argument-type]
                    result = _util.apply_keepdims(result, axis, tensor.ndim)

            # out
            if "out" in params:
                # out can be in any position so we need sig.bind
                if bound_args is None:
                    bound_args = sig.bind(*norm_args, **norm_kwds).arguments
                out = bound_args.get("out")
                result = maybe_copy_to(out, result, promote_scalar_result)
            result = wrap_tensors(result)

            return result  # pyrefly: ignore[bad-return]

        return wrapped

    if _func is None:
        return normalizer_inner
    else:
        return normalizer_inner(_func)
<|endoftext|># mypy: ignore-errors

from __future__ import annotations

import builtins
import math
import operator
from collections.abc import Sequence

import torch

from . import _dtypes, _dtypes_impl, _funcs, _ufuncs, _util
from ._normalizations import (
    ArrayLike,
    normalize_array_like,
    normalizer,
    NotImplementedType,
)


newaxis = None

FLAGS = [
    "C_CONTIGUOUS",
    "F_CONTIGUOUS",
    "OWNDATA",
    "WRITEABLE",
    "ALIGNED",
    "WRITEBACKIFCOPY",
    "FNC",
    "FORC",
    "BEHAVED",
    "CARRAY",
    "FARRAY",
]

SHORTHAND_TO_FLAGS = {
    "C": "C_CONTIGUOUS",
    "F": "F_CONTIGUOUS",
    "O": "OWNDATA",
    "W": "WRITEABLE",
    "A": "ALIGNED",
    "X": "WRITEBACKIFCOPY",
    "B": "BEHAVED",
    "CA": "CARRAY",
    "FA": "FARRAY",
}


class Flags:
    def __init__(self, flag_to_value: dict):
        invalid_keys = [k for k in flag_to_value if k not in FLAGS]
        if invalid_keys:
            raise AssertionError(f"Invalid flag keys: {invalid_keys}")
        self._flag_to_value = flag_to_value

    def __getattr__(self, attr: str):
        if attr.islower() and attr.upper() in FLAGS:
            return self[attr.upper()]
        else:
            raise AttributeError(f"No flag attribute '{attr}'")

    def __getitem__(self, key):
        if key in SHORTHAND_TO_FLAGS:
            key = SHORTHAND_TO_FLAGS[key]
        if key in FLAGS:
            try:
                return self._flag_to_value[key]
            except KeyError as e:
                raise NotImplementedError(f"{key=}") from e
        else:
            raise KeyError(f"No flag key '{key}'")

    def __setattr__(self, attr, value):
        if attr.islower() and attr.upper() in FLAGS:
            self[attr.upper()] = value
        else:
            super().__setattr__(attr, value)

    def __setitem__(self, key, value):
        if key in FLAGS or key in SHORTHAND_TO_FLAGS:
            raise NotImplementedError("Modifying flags is not implemented")
        else:
            raise KeyError(f"No flag key '{key}'")


def create_method(fn, name=None):
    name = name or fn.__name__

    def f(*args, **kwargs):
        return fn(*args, **kwargs)

    f.__name__ = name
    f.__qualname__ = f"ndarray.{name}"
    return f


# Map ndarray.name_method -> np.name_func
# If name_func == None, it means that name_method == name_func
methods = {
    "clip": None,
    "nonzero": None,
    "repeat": None,
    "round": None,
    "squeeze": None,
    "swapaxes": None,
    "ravel": None,
    # linalg
    "diagonal": None,
    "dot": None,
    "trace": None,
    # sorting
    "argsort": None,
    "searchsorted": None,
    # reductions
    "argmax": None,
    "argmin": None,
    "any": None,
    "all": None,
    "max": None,
    "min": None,
    "ptp": None,
    "sum": None,
    "prod": None,
    "mean": None,
    "var": None,
    "std": None,
    # scans
    "cumsum": None,
    "cumprod": None,
    # advanced indexing
    "take": None,
    "choose": None,
}

dunder = {
    "abs": "absolute",
    "invert": None,
    "pos": "positive",
    "neg": "negative",
    "gt": "greater",
    "lt": "less",
    "ge": "greater_equal",
    "le": "less_equal",
}

# dunder methods with right-looking and in-place variants
ri_dunder = {
    "add": None,
    "sub": "subtract",
    "mul": "multiply",
    "truediv": "divide",
    "floordiv": "floor_divide",
    "pow": "power",
    "mod": "remainder",
    "and": "bitwise_and",
    "or": "bitwise_or",
    "xor": "bitwise_xor",
    "lshift": "left_shift",
    "rshift": "right_shift",
    "matmul": None,
}


def _upcast_int_indices(index):
    if isinstance(index, torch.Tensor):
        if index.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            return index.to(torch.int64)
    elif isinstance(index, tuple):
        return tuple(_upcast_int_indices(i) for i in index)
    return index


def _has_advanced_indexing(index):
    """Check if there's any advanced indexing"""
    return any(
        isinstance(idx, (Sequence, bool))
        or (isinstance(idx, torch.Tensor) and (idx.dtype == torch.bool or idx.ndim > 0))
        for idx in index
    )


def _numpy_compatible_indexing(index):
    """Convert scalar indices to lists when advanced indexing is present for NumPy compatibility."""
    if not isinstance(index, tuple):
        index = (index,)

    # Check if there's any advanced indexing (sequences, booleans, or tensors)
    has_advanced = _has_advanced_indexing(index)

    if not has_advanced:
        return index

    # Convert integer scalar indices to single-element lists when advanced indexing is present
    # Note: Do NOT convert boolean scalars (True/False) as they have special meaning in NumPy
    converted = []
    for idx in index:
        if isinstance(idx, int) and not isinstance(idx, bool):
            # Integer scalars should be converted to lists
            converted.append([idx])
        elif (
            isinstance(idx, torch.Tensor)
            and idx.ndim == 0
            and not torch.is_floating_point(idx)
            and idx.dtype != torch.bool
        ):
            # Zero-dimensional tensors holding integers should be treated the same as integer scalars
            converted.append([idx])
        else:
            # Everything else (booleans, lists, slices, etc.) stays as is
            converted.append(idx)

    return tuple(converted)


def _get_bool_depth(s):
    """Returns the depth of a boolean sequence/tensor"""
    if isinstance(s, bool):
        return True, 0
    if isinstance(s, torch.Tensor) and s.dtype == torch.bool:
        return True, s.ndim
    if not (isinstance(s, Sequence) and s and s[0] != s):
        return False, 0
    is_bool, depth = _get_bool_depth(s[0])
    return is_bool, depth + 1


def _numpy_empty_ellipsis_patch(index, tensor_ndim):
    """
    Patch for NumPy-compatible ellipsis behavior when ellipsis doesn't match any dimensions.

    In NumPy, when an ellipsis (...) doesn't actually match any dimensions of the input array,
    it still acts as a separator between advanced indices. PyTorch doesn't have this behavior.

    This function detects when we have:
    1. Advanced indexing on both sides of an ellipsis
    2. The ellipsis doesn't actually match any dimensions
    """
    if not isinstance(index, tuple):
        index = (index,)

    # Find ellipsis position
    ellipsis_pos = None
    for i, idx in enumerate(index):
        if idx is Ellipsis:
            ellipsis_pos = i
            break

    # If no ellipsis, no patch needed
    if ellipsis_pos is None:
        return index, lambda x: x, lambda x: x

    # Count non-ellipsis dimensions consumed by the index
    consumed_dims = 0
    for idx in index:
        is_bool, depth = _get_bool_depth(idx)
        if is_bool:
            consumed_dims += depth
        elif idx is Ellipsis or idx is None:
            continue
        else:
            consumed_dims += 1

    # Calculate how many dimensions the ellipsis should match
    ellipsis_dims = tensor_ndim - consumed_dims

    # Check if ellipsis doesn't match any dimensions
    if ellipsis_dims == 0:
        # Check if we have advanced indexing on both sides of ellipsis
        left_advanced = _has_advanced_indexing(index[:ellipsis_pos])
        right_advanced = _has_advanced_indexing(index[ellipsis_pos + 1 :])

        if left_advanced and right_advanced:
            # This is the case where NumPy and PyTorch differ
            # We need to ensure the advanced indices are treated as separated
            new_index = index[:ellipsis_pos] + (None,) + index[ellipsis_pos + 1 :]
            end_ndims = 1 + sum(
                1 for idx in index[ellipsis_pos + 1 :] if isinstance(idx, slice)
            )

            def squeeze_fn(x):
                return x.squeeze(-end_ndims)

            def unsqueeze_fn(x):
                if isinstance(x, torch.Tensor) and x.ndim >= end_ndims:
                    return x.unsqueeze(-end_ndims)
                return x

            return new_index, squeeze_fn, unsqueeze_fn

    return index, lambda x: x, lambda x: x


# Used to indicate that a parameter is unspecified (as opposed to explicitly
# `None`)
class _Unspecified:
    pass


_Unspecified.unspecified = _Unspecified()

###############################################################
#                      ndarray class                          #
###############################################################


class ndarray:
    def __init__(self, t=None):
        if t is None:
            self.tensor = torch.Tensor()
        elif isinstance(t, torch.Tensor):
            self.tensor = t
        else:
            raise ValueError(
                "ndarray constructor is not recommended; prefer"
                "either array(...) or zeros/empty(...)"
            )

    # Register NumPy functions as methods
    for method, name in methods.items():
        fn = getattr(_funcs, name or method)
        vars()[method] = create_method(fn, method)

    # Regular methods but coming from ufuncs
    conj = create_method(_ufuncs.conjugate, "conj")
    conjugate = create_method(_ufuncs.conjugate)

    for method, name in dunder.items():
        fn = getattr(_ufuncs, name or method)
        method = f"__{method}__"
        vars()[method] = create_method(fn, method)

    for method, name in ri_dunder.items():
        fn = getattr(_ufuncs, name or method)
        plain = f"__{method}__"
        vars()[plain] = create_method(fn, plain)
        rvar = f"__r{method}__"
        vars()[rvar] = create_method(lambda self, other, fn=fn: fn(other, self), rvar)
        ivar = f"__i{method}__"
        vars()[ivar] = create_method(
            lambda self, other, fn=fn: fn(self, other, out=self), ivar
        )

    # There's no __idivmod__
    __divmod__ = create_method(_ufuncs.divmod, "__divmod__")
    __rdivmod__ = create_method(
        lambda self, other: _ufuncs.divmod(other, self), "__rdivmod__"
    )

    # prevent loop variables leaking into the ndarray class namespace
    del ivar, rvar, name, plain, fn, method

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def size(self):
        return self.tensor.numel()

    @property
    def ndim(self):
        return self.tensor.ndim

    @property
    def dtype(self):
        return _dtypes.dtype(self.tensor.dtype)

    @property
    def strides(self):
        elsize = self.tensor.element_size()
        return tuple(stride * elsize for stride in self.tensor.stride())

    @property
    def itemsize(self):
        return self.tensor.element_size()

    @property
    def flags(self):
        # Note contiguous in torch is assumed C-style
        return Flags(
            {
                "C_CONTIGUOUS": self.tensor.is_contiguous(),
                "F_CONTIGUOUS": self.T.tensor.is_contiguous(),
                "OWNDATA": self.tensor._base is None,
                "WRITEABLE": True,  # pytorch does not have readonly tensors
            }
        )

    @property
    def data(self):
        return self.tensor.data_ptr()

    @property
    def nbytes(self):
        return self.tensor.storage().nbytes()

    @property
    def T(self):
        return self.transpose()

    @property
    def real(self):
        return _funcs.real(self)

    @real.setter
    def real(self, value):
        self.tensor.real = asarray(value).tensor

    @property
    def imag(self):
        return _funcs.imag(self)

    @imag.setter
    def imag(self, value):
        self.tensor.imag = asarray(value).tensor

    @property
    def flat(self):
        return self.ravel()

    # ctors
    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        if order != "K":
            raise NotImplementedError(f"astype(..., order={order} is not implemented.")
        if casting != "unsafe":
            raise NotImplementedError(
                f"astype(..., casting={casting} is not implemented."
            )
        if not subok:
            raise NotImplementedError(f"astype(..., subok={subok} is not implemented.")
        if not copy:
            raise NotImplementedError(f"astype(..., copy={copy} is not implemented.")
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        t = self.tensor.to(torch_dtype)
        return ndarray(t)

    @normalizer
    def copy(self: ArrayLike, order: NotImplementedType = "C"):
        return self.clone()

    @normalizer
    def flatten(self: ArrayLike, order: NotImplementedType = "C"):
        return torch.flatten(self)

    def resize(self, *new_shape, refcheck=False):
        # NB: differs from np.resize: fills with zeros instead of making repeated copies of input.
        if refcheck:
            raise NotImplementedError(
                f"resize(..., refcheck={refcheck} is not implemented."
            )
        if new_shape in [(), (None,)]:
            return

        # support both x.resize((2, 2)) and x.resize(2, 2)
        if len(new_shape) == 1:
            new_shape = new_shape[0]
        if isinstance(new_shape, int):
            new_shape = (new_shape,)

        if builtins.any(x < 0 for x in new_shape):
            raise ValueError("all elements of `new_shape` must be non-negative")

        new_numel, old_numel = math.prod(new_shape), self.tensor.numel()

        self.tensor.resize_(new_shape)

        if new_numel >= old_numel:
            # zero-fill new elements
            if not self.tensor.is_contiguous():
                raise AssertionError("tensor must be contiguous for resize with growth")
            b = self.tensor.flatten()  # does not copy
            b[old_numel:].zero_()

    def view(self, dtype=_Unspecified.unspecified, type=_Unspecified.unspecified):
        if dtype is _Unspecified.unspecified:
            dtype = self.dtype
        if type is not _Unspecified.unspecified:
            raise NotImplementedError(f"view(..., type={type} is not implemented.")
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        tview = self.tensor.view(torch_dtype)
        return ndarray(tview)

    @normalizer
    def fill(self, value: ArrayLike):
        # Both Pytorch and NumPy accept 0D arrays/tensors and scalars, and
        # error out on D > 0 arrays
        self.tensor.fill_(value)

    def tolist(self):
        return self.tensor.tolist()

    def __iter__(self):
        return (ndarray(x) for x in self.tensor.__iter__())

    def __str__(self):
        return (
            str(self.tensor)
            .replace("tensor", "torch.ndarray")
            .replace("dtype=torch.", "dtype=")
        )

    __repr__ = create_method(__str__)

    def __eq__(self, other):
        try:
            return _ufuncs.equal(self, other)
        except (RuntimeError, TypeError):
            # Failed to convert other to array: definitely not equal.
            falsy = torch.full(self.shape, fill_value=False, dtype=bool)
            return asarray(falsy)

    def __ne__(self, other):
        return ~(self == other)

    def __index__(self):
        try:
            return operator.index(self.tensor.item())
        except Exception as exc:
            raise TypeError(
                "only integer scalar arrays can be converted to a scalar index"
            ) from exc

    def __bool__(self):
        return bool(self.tensor)

    def __int__(self):
        return int(self.tensor)

    def __float__(self):
        return float(self.tensor)

    def __complex__(self):
        return complex(self.tensor)

    def is_integer(self):
        try:
            v = self.tensor.item()
            result = int(v) == v
        except Exception:
            result = False
        return result

    def __len__(self):
        return self.tensor.shape[0]

    def __contains__(self, x):
        return self.tensor.__contains__(x)

    def transpose(self, *axes):
        # np.transpose(arr, axis=None) but arr.transpose(*axes)
        return _funcs.transpose(self, axes)

    def reshape(self, *shape, order="C"):
        # arr.reshape(shape) and arr.reshape(*shape)
        return _funcs.reshape(self, shape, order=order)

    def sort(self, axis=-1, kind=None, order=None):
        # ndarray.sort works in-place
        _funcs.copyto(self, _funcs.sort(self, axis, kind, order))

    def item(self, *args):
        # Mimic NumPy's implementation with three special cases (no arguments,
        # a flat index and a multi-index):
        # https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/methods.c#L702
        if args == ():
            return self.tensor.item()
        elif len(args) == 1:
            # int argument
            return self.ravel()[args[0]]
        else:
            return self.__getitem__(args)

    def __getitem__(self, index):
        tensor = self.tensor

        def neg_step(i, s):
            if not (isinstance(s, slice) and s.step is not None and s.step < 0):
                return s

            nonlocal tensor
            tensor = torch.flip(tensor, (i,))

            # Account for the fact that a slice includes the start but not the end
            if not (isinstance(s.start, int) or s.start is None):
                raise AssertionError(
                    f"slice start must be int or None, got {type(s.start).__name__}"
                )
            if not (isinstance(s.stop, int) or s.stop is None):
                raise AssertionError(
                    f"slice stop must be int or None, got {type(s.stop).__name__}"
                )
            start = s.stop + 1 if s.stop else None
            stop = s.start + 1 if s.start else None

            return slice(start, stop, -s.step)

        if isinstance(index, Sequence):
            index = type(index)(neg_step(i, s) for i, s in enumerate(index))
        else:
            index = neg_step(0, index)
        index = _util.ndarrays_to_tensors(index)
        index = _upcast_int_indices(index)
        # Apply NumPy-compatible indexing conversion
        index = _numpy_compatible_indexing(index)
        # Apply NumPy-compatible empty ellipsis behavior
        index, maybe_squeeze, _ = _numpy_empty_ellipsis_patch(index, tensor.ndim)
        return maybe_squeeze(ndarray(tensor.__getitem__(index)))

    def __setitem__(self, index, value):
        index = _util.ndarrays_to_tensors(index)
        index = _upcast_int_indices(index)
        # Apply NumPy-compatible indexing conversion
        index = _numpy_compatible_indexing(index)
        # Apply NumPy-compatible empty ellipsis behavior
        index, _, maybe_unsqueeze = _numpy_empty_ellipsis_patch(index, self.tensor.ndim)

        if not _dtypes_impl.is_scalar(value):
            value = normalize_array_like(value)
            value = _util.cast_if_needed(value, self.tensor.dtype)

        return self.tensor.__setitem__(index, maybe_unsqueeze(value))

    take = _funcs.take
    put = _funcs.put

    def __dlpack__(self, *, stream=None):
        return self.tensor.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def _tolist(obj):
    """Recursively convert tensors into lists."""
    a1 = []
    for elem in obj:
        if isinstance(elem, (list, tuple)):
            elem = _tolist(elem)
        if isinstance(elem, ndarray):
            a1.append(elem.tensor.tolist())
        else:
            a1.append(elem)
    return a1


# This is the ideally the only place which talks to ndarray directly.
# The rest goes through asarray (preferred) or array.


def array(obj, dtype=None, *, copy=True, order="K", subok=False, ndmin=0, like=None):
    if subok is not False:
        raise NotImplementedError("'subok' parameter is not supported.")
    if like is not None:
        raise NotImplementedError("'like' parameter is not supported.")
    if order != "K":
        raise NotImplementedError

    # a happy path
    if (
        isinstance(obj, ndarray)
        and copy is False
        and dtype is None
        and ndmin <= obj.ndim
    ):
        return obj

    if isinstance(obj, (list, tuple)):
        # FIXME and they have the same dtype, device, etc
        if obj and all(isinstance(x, torch.Tensor) for x in obj):
            # list of arrays: *under torch.Dynamo* these are FakeTensors
            obj = torch.stack(obj)
        else:
            # XXX: remove tolist
            # lists of ndarrays: [1, [2, 3], ndarray(4)] convert to lists of lists
            obj = _tolist(obj)

    # is obj an ndarray already?
    if isinstance(obj, ndarray):
        obj = obj.tensor

    # is a specific dtype requested?
    torch_dtype = None
    if dtype is not None:
        torch_dtype = _dtypes.dtype(dtype).torch_dtype

    tensor = _util._coerce_to_tensor(obj, torch_dtype, copy, ndmin)
    return ndarray(tensor)


def asarray(a, dtype=None, order="K", *, like=None):
    return array(a, dtype=dtype, order=order, like=like, copy=False, ndmin=0)


def ascontiguousarray(a, dtype=None, *, like=None):
    arr = asarray(a, dtype=dtype, like=like)
    if not arr.tensor.is_contiguous():
        arr.tensor = arr.tensor.contiguous()
    return arr


def from_dlpack(x, /):
    t = torch.from_dlpack(x)
    return ndarray(t)


def _extract_dtype(entry):
    try:
        dty = _dtypes.dtype(entry)
    except Exception:
        dty = asarray(entry).dtype
    return dty


def can_cast(from_, to, casting="safe"):
    from_ = _extract_dtype(from_)
    to_ = _extract_dtype(to)

    return _dtypes_impl.can_cast_impl(from_.torch_dtype, to_.torch_dtype, casting)


def result_type(*arrays_and_dtypes):
    tensors = []
    for entry in arrays_and_dtypes:
        try:
            t = asarray(entry).tensor
        except (RuntimeError, ValueError, TypeError):
            dty = _dtypes.dtype(entry)
            t = torch.empty(1, dtype=dty.torch_dtype)
        tensors.append(t)

    torch_dtype = _dtypes_impl.result_type_impl(*tensors)
    return _dtypes.dtype(torch_dtype)
<|endoftext|># mypy: ignore-errors

import torch

from . import _dtypes


def finfo(dtyp):
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    return torch.finfo(torch_dtype)


def iinfo(dtyp):
    torch_dtype = _dtypes.dtype(dtyp).torch_dtype
    return torch.iinfo(torch_dtype)
<|endoftext|>"""Assorted utilities, which do not need anything other than torch and stdlib."""

import operator
from collections.abc import Iterable, Sequence, Sized
from typing import TypeGuard

import torch

from . import _dtypes_impl


# https://github.com/numpy/numpy/blob/v1.23.0/numpy/distutils/misc_util.py#L497-L504
def is_sequence(seq: object) -> TypeGuard[Sized]:
    if isinstance(seq, str):
        return False
    if not isinstance(seq, Sized):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True


class AxisError(ValueError, IndexError):
    pass


class UFuncTypeError(TypeError, RuntimeError):
    pass


def cast_if_needed(tensor: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    # NB: no casting if dtype=None
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor


def cast_int_to_float(x: torch.Tensor) -> torch.Tensor:
    # cast integers and bools to the default float dtype
    if _dtypes_impl._category(x.dtype) < 2:
        x = x.to(_dtypes_impl.default_dtypes().float_dtype)
    return x


# a replica of the version in ./numpy/numpy/core/src/multiarray/common.h
def normalize_axis_index(ax: int, ndim: int, argname: str | None = None) -> int:
    if not (-ndim <= ax < ndim):
        raise AxisError(f"axis {ax} is out of bounds for array of dimension {ndim}")
    if ax < 0:
        ax += ndim
    return ax


# from https://github.com/numpy/numpy/blob/main/numpy/core/numeric.py#L1378
def normalize_axis_tuple(
    axis: int | Iterable[int],
    ndim: int,
    argname: str | None = None,
    allow_duplicate: bool = False,
) -> tuple[int, ...]:
    """
    Normalizes an axis argument into a tuple of non-negative integer axes.

    This handles shorthands such as ``1`` and converts them to ``(1,)``,
    as well as performing the handling of negative indices covered by
    `normalize_axis_index`.

    By default, this forbids axes from being specified multiple times.
    Used internally by multi-axis-checking logic.

    Parameters
    ----------
    axis : int, iterable of int
        The un-normalized index or indices of the axis.
    ndim : int
        The number of dimensions of the array that `axis` should be normalized
        against.
    argname : str, optional
        A prefix to put before the error message, typically the name of the
        argument.
    allow_duplicate : bool, optional
        If False, the default, disallow an axis from being specified twice.

    Returns
    -------
    normalized_axes : tuple of int
        The normalized axis index, such that `0 <= normalized_axis < ndim`
    """
    # Optimization to speed-up the most common cases.
    axes: Iterable[int]
    if isinstance(axis, (tuple, list)):
        axes = axis
    else:
        # operator.index handles the scalar shorthand (int and __index__-like
        # objects); a TypeError means axis is already an iterable of ints.
        try:
            axes = [operator.index(axis)]  # pyrefly: ignore[bad-argument-type]
        except TypeError:
            axes = axis  # pyrefly: ignore[bad-assignment]
    # Going via an iterator directly is slower than via list comprehension.
    result = tuple(normalize_axis_index(ax, ndim, argname) for ax in axes)
    if not allow_duplicate and len(set(map(int, result))) != len(result):
        if argname:
            raise ValueError(f"repeated axis in `{argname}` argument")
        else:
            raise ValueError("repeated axis")
    return result


def allow_only_single_axis(axis: int | tuple[int, ...] | None) -> int | None:
    if axis is None:
        return axis
    if isinstance(axis, int):
        return axis
    if len(axis) != 1:
        raise NotImplementedError("does not handle tuple axis")
    return axis[0]


def expand_shape(
    arr_shape: Sequence[int], axis: int | tuple[int, ...] | list[int]
) -> list[int]:
    # taken from numpy 1.23.x, expand_dims function
    axis_seq = axis if isinstance(axis, (list, tuple)) else (axis,)
    out_ndim = len(axis_seq) + len(arr_shape)
    normalized = normalize_axis_tuple(axis_seq, out_ndim)
    shape_it = iter(arr_shape)
    shape = [1 if ax in normalized else next(shape_it) for ax in range(out_ndim)]
    return shape


def apply_keepdims(
    tensor: torch.Tensor, axis: int | tuple[int, ...] | None, ndim: int
) -> torch.Tensor:
    if axis is None:
        # tensor was a scalar
        shape = (1,) * ndim
        tensor = tensor.expand(shape).contiguous()
    else:
        shape = expand_shape(tensor.shape, axis)
        tensor = tensor.reshape(shape)
    return tensor


def axis_none_flatten(
    *tensors: torch.Tensor, axis: int | None = None
) -> tuple[tuple[torch.Tensor, ...], int]:
    """Flatten the arrays if axis is None."""
    if axis is None:
        tensors = tuple(ar.flatten() for ar in tensors)
        return tensors, 0
    else:
        return tensors, axis


def typecast_tensor(
    t: torch.Tensor, target_dtype: torch.dtype, casting: str
) -> torch.Tensor:
    """Dtype-cast tensor to target_dtype.

    Parameters
    ----------
    t : torch.Tensor
        The tensor to cast
    target_dtype : torch dtype object
        The array dtype to cast all tensors to
    casting : str
        The casting mode, see `np.can_cast`

     Returns
     -------
    `torch.Tensor` of the `target_dtype` dtype

     Raises
     ------
     ValueError
        if the argument cannot be cast according to the `casting` rule

    """
    can_cast = _dtypes_impl.can_cast_impl

    if not can_cast(t.dtype, target_dtype, casting=casting):
        raise TypeError(
            f"Cannot cast array data from {t.dtype} to"
            f" {target_dtype} according to the rule '{casting}'"
        )
    return cast_if_needed(t, target_dtype)


def typecast_tensors(
    tensors: Sequence[torch.Tensor], target_dtype: torch.dtype, casting: str
) -> tuple[torch.Tensor, ...]:
    return tuple(typecast_tensor(t, target_dtype, casting) for t in tensors)


def _try_convert_to_tensor(obj: object) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(obj)
    except Exception as e:
        mesg = f"failed to convert {obj} to ndarray. \nInternal error is: {str(e)}."
        raise NotImplementedError(mesg)  # noqa: B904
    return tensor


def _coerce_to_tensor(
    obj: object,
    dtype: torch.dtype | None = None,
    copy: bool = False,
    ndmin: int = 0,
) -> torch.Tensor:
    """The core logic of the array(...) function.

    Parameters
    ----------
    obj : tensor_like
        The thing to coerce
    dtype : torch.dtype object or None
        Coerce to this torch dtype
    copy : bool
        Copy or not
    ndmin : int
        The results as least this many dimensions
    is_weak : bool
        Whether obj is a weakly typed python scalar.

    Returns
    -------
    tensor : torch.Tensor
        a tensor object with requested dtype, ndim and copy semantics.

    Notes
    -----
    This is almost a "tensor_like" coercive function. Does not handle wrapper
    ndarrays (those should be handled in the ndarray-aware layer prior to
    invoking this function).
    """
    if isinstance(obj, torch.Tensor):
        tensor = obj
    else:
        # tensor.dtype is the pytorch default, typically float32. If obj's elements
        # are not exactly representable in float32, we've lost precision:
        # >>> torch.as_tensor(1e12).item() - 1e12
        # -4096.0
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(_dtypes_impl.get_default_dtype_for(torch.float32))
        try:
            tensor = _try_convert_to_tensor(obj)
        finally:
            torch.set_default_dtype(default_dtype)

    # type cast if requested
    tensor = cast_if_needed(tensor, dtype)

    # adjust ndim if needed
    ndim_extra = ndmin - tensor.ndim
    if ndim_extra > 0:
        tensor = tensor.view((1,) * ndim_extra + tensor.shape)

    # special handling for np._CopyMode
    try:
        copy = bool(copy)
    except ValueError:
        # TODO handle _CopyMode.IF_NEEDED correctly
        copy = False
    # copy if requested
    if copy:
        tensor = tensor.clone()

    return tensor


def ndarrays_to_tensors(*inputs: object) -> object:
    """Convert all ndarrays from `inputs` to tensors. (other things are intact)"""
    from ._ndarray import ndarray

    if len(inputs) == 0:
        return ValueError()
    elif len(inputs) == 1:
        input_ = inputs[0]
        if isinstance(input_, ndarray):
            return input_.tensor
        elif isinstance(input_, tuple):
            result = []
            for sub_input in input_:
                sub_result = ndarrays_to_tensors(sub_input)
                result.append(sub_result)
            return tuple(result)
        else:
            return input_
    else:
        if not isinstance(inputs, tuple):
            raise AssertionError(
                f"Expected inputs to be a tuple, got {type(inputs).__name__}"
            )
        return ndarrays_to_tensors(inputs)
<|endoftext|># mypy: ignore-errors

"""Export torch work functions for binary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `torch._numpy/_ufuncs.py` module.
"""

import torch
from torch import (  # noqa: F401
    add,
    arctan2,
    bitwise_and,
    bitwise_left_shift as left_shift,
    bitwise_or,
    bitwise_right_shift as right_shift,
    bitwise_xor,
    copysign,
    divide,
    eq as equal,
    float_power,
    floor_divide,
    fmax,
    fmin,
    fmod,
    gcd,
    greater,
    greater_equal,
    heaviside,
    hypot,
    lcm,
    ldexp,
    less,
    less_equal,
    logaddexp,
    logaddexp2,
    logical_and,
    logical_or,
    logical_xor,
    maximum,
    minimum,
    multiply,
    nextafter,
    not_equal,
    pow as power,
    remainder,
    remainder as mod,
    subtract,
    true_divide,
)

from . import _dtypes_impl, _util


# work around torch limitations w.r.t. numpy
def matmul(x, y):
    # work around:
    #  - RuntimeError: expected scalar type Int but found Double
    #  - RuntimeError: "addmm_impl_cpu_" not implemented for 'Bool'
    #  - RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    dtype = _dtypes_impl.result_type_impl(x, y)
    is_bool = dtype == torch.bool
    is_half = (x.dtype == torch.float16 or y.dtype == torch.float16) and (
        x.is_cpu or y.is_cpu
    )

    work_dtype = dtype
    if is_bool:
        work_dtype = torch.uint8
    if is_half:
        work_dtype = torch.float32

    x = _util.cast_if_needed(x, work_dtype)
    y = _util.cast_if_needed(y, work_dtype)

    result = torch.matmul(x, y)

    if work_dtype != dtype:
        result = result.to(dtype)

    return result


# a stub implementation of divmod, should be improved after
# https://github.com/pytorch/pytorch/issues/90820 is fixed in pytorch
def divmod(x, y):
    return x // y, x % y
<|endoftext|># mypy: ignore-errors

"""Export torch work functions for unary ufuncs, rename/tweak to match numpy.
This listing is further exported to public symbols in the `_numpy/_ufuncs.py` module.
"""

import torch
from torch import (  # noqa: F401
    absolute as fabs,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    bitwise_not,
    bitwise_not as invert,
    ceil,
    conj_physical as conjugate,
    cos,
    cosh,
    deg2rad,
    deg2rad as radians,
    exp,
    exp2,
    expm1,
    floor,
    isfinite,
    isinf,
    isnan,
    log,
    log10,
    log1p,
    log2,
    logical_not,
    negative,
    rad2deg,
    rad2deg as degrees,
    reciprocal,
    round as fix,
    round as rint,
    sign,
    signbit,
    sin,
    sinh,
    sqrt,
    square,
    tan,
    tanh,
    trunc,
)


# special cases: torch does not export these names
def cbrt(x):
    return torch.pow(x, 1 / 3)


def positive(x):
    return +x


def absolute(x):
    # work around torch.absolute not impl for bools
    if x.dtype == torch.bool:
        return x
    return torch.absolute(x)


# TODO set __name__ and __qualname__
abs = absolute
conj = conjugate
<|endoftext|>"""Implementation of reduction operations, to be wrapped into arrays, dtypes etc
in the 'public' layer.

Anything here only deals with torch objects, e.g. "dtype" is a torch.dtype instance etc
"""

from __future__ import annotations

import functools
from typing import Concatenate, ParamSpec, TYPE_CHECKING, TypeVar

import torch

from . import _dtypes_impl, _util


if TYPE_CHECKING:
    from collections.abc import Callable

    from ._normalizations import (
        ArrayLike,
        AxisLike,
        DTypeLike,
        KeepDims,
        NotImplementedType,
        OutArray,
    )


_P = ParamSpec("_P")
_R = TypeVar("_R")


def _deco_axis_expand(
    func: Callable[Concatenate[torch.Tensor, AxisLike, _P], _R],
) -> Callable[Concatenate[torch.Tensor, AxisLike, _P], _R]:
    """
    Generically handle axis arguments in reductions.
    axis is *always* the 2nd arg in the function so no need to have a look at its signature
    """

    @functools.wraps(func)
    def wrapped(
        a: torch.Tensor, axis: AxisLike = None, *args: _P.args, **kwds: _P.kwargs
    ) -> _R:
        if axis is not None:
            axis = _util.normalize_axis_tuple(axis, a.ndim)

        if axis == ():
            # So we insert a length-one axis and run the reduction along it.
            # We cannot return a.clone() as this would sidestep the checks inside the function
            newshape = _util.expand_shape(a.shape, axis=0)
            a = a.reshape(newshape)
            axis = (0,)

        return func(a, axis, *args, **kwds)

    return wrapped


def _atleast_float(dtype: torch.dtype | None, other_dtype: torch.dtype) -> torch.dtype:
    """Return a dtype that is real or complex floating-point.

    For inputs that are boolean or integer dtypes, this returns the default
    float dtype; inputs that are complex get converted to the default complex
    dtype; real floating-point dtypes (`float*`) get passed through unchanged
    """
    if dtype is None:
        dtype = other_dtype
    if not (dtype.is_floating_point or dtype.is_complex):
        return _dtypes_impl.default_dtypes().float_dtype
    return dtype


@_deco_axis_expand
def count_nonzero(
    a: ArrayLike, axis: AxisLike = None, *, keepdims: KeepDims = False
) -> torch.Tensor:
    return a.count_nonzero(axis)


@_deco_axis_expand
def argmax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    *,
    keepdims: KeepDims = False,
) -> torch.Tensor:
    if a.is_complex():
        raise NotImplementedError(f"argmax with dtype={a.dtype}.")

    axis = _util.allow_only_single_axis(axis)

    if a.dtype == torch.bool:
        # RuntimeError: "argmax_cpu" not implemented for 'Bool'
        a = a.to(torch.uint8)

    return torch.argmax(a, axis)


@_deco_axis_expand
def argmin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    *,
    keepdims: KeepDims = False,
) -> torch.Tensor:
    if a.is_complex():
        raise NotImplementedError(f"argmin with dtype={a.dtype}.")

    axis = _util.allow_only_single_axis(axis)

    if a.dtype == torch.bool:
        # RuntimeError: "argmin_cpu" not implemented for 'Bool'
        a = a.to(torch.uint8)

    return torch.argmin(a, axis)


@_deco_axis_expand
def any(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
) -> torch.Tensor:
    axis = _util.allow_only_single_axis(axis)
    return torch.any(a) if axis is None else torch.any(a, dim=axis)


@_deco_axis_expand
def all(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
) -> torch.Tensor:
    axis = _util.allow_only_single_axis(axis)
    return torch.all(a) if axis is None else torch.all(a, dim=axis)


@_deco_axis_expand
def amax(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
) -> torch.Tensor:
    if a.is_complex():
        raise NotImplementedError(f"amax with dtype={a.dtype}")

    return a.amax() if axis is None else a.amax(axis)


max = amax


@_deco_axis_expand
def amin(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
) -> torch.Tensor:
    if a.is_complex():
        raise NotImplementedError(f"amin with dtype={a.dtype}")

    return a.amin() if axis is None else a.amin(axis)


min = amin


@_deco_axis_expand
def ptp(
    a: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
) -> torch.Tensor:
    amax = a.amax() if axis is None else a.amax(axis)
    amin = a.amin() if axis is None else a.amin(axis)
    return amax - amin


@_deco_axis_expand
def sum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
) -> torch.Tensor:
    if dtype is not None and not isinstance(dtype, torch.dtype):
        raise AssertionError(
            f"dtype must be None or a torch.dtype, got {type(dtype).__name__}"
        )

    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype

    if axis is None:
        return a.sum(dtype=dtype)
    return a.sum(dim=axis, dtype=dtype)


@_deco_axis_expand
def prod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    initial: NotImplementedType = None,
    where: NotImplementedType = None,
) -> torch.Tensor:
    axis = _util.allow_only_single_axis(axis)

    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype

    if axis is None:
        return a.prod(dtype=dtype)
    return a.prod(dim=axis, dtype=dtype)


product = prod


@_deco_axis_expand
def mean(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
) -> torch.Tensor:
    dtype = _atleast_float(dtype, a.dtype)

    if axis is None:
        return a.mean(dtype=dtype)
    return a.mean(dim=axis, dtype=dtype)


@_deco_axis_expand
def std(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
    ddof: float = 0,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
) -> torch.Tensor:
    in_dtype = dtype
    dtype = _atleast_float(dtype, a.dtype)
    tensor = _util.cast_if_needed(a, dtype)
    result = tensor.std(dim=axis, correction=ddof)
    return _util.cast_if_needed(result, in_dtype)


@_deco_axis_expand
def var(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
    ddof: float = 0,
    keepdims: KeepDims = False,
    *,
    where: NotImplementedType = None,
) -> torch.Tensor:
    in_dtype = dtype
    dtype = _atleast_float(dtype, a.dtype)
    tensor = _util.cast_if_needed(a, dtype)
    result = tensor.var(dim=axis, correction=ddof)
    return _util.cast_if_needed(result, in_dtype)


# cumsum / cumprod are almost reductions:
#   1. no keepdims
#   2. axis=None flattens


def cumsum(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
) -> torch.Tensor:
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype
    if dtype is None:
        dtype = a.dtype

    # pyrefly: ignore[bad-argument-type]
    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)

    # torch accepts the numpy-compat `axis=` alias, but its stub only lists `dim`.
    # pyrefly: ignore[missing-argument, unexpected-keyword]
    return a.cumsum(axis=axis, dtype=dtype)


def cumprod(
    a: ArrayLike,
    axis: AxisLike = None,
    dtype: DTypeLike | None = None,
    out: OutArray | None = None,
) -> torch.Tensor:
    if dtype == torch.bool:
        dtype = _dtypes_impl.default_dtypes().int_dtype
    if dtype is None:
        dtype = a.dtype

    # pyrefly: ignore[bad-argument-type]
    (a,), axis = _util.axis_none_flatten(a, axis=axis)
    axis = _util.normalize_axis_index(axis, a.ndim)

    # torch accepts the numpy-compat `axis=` alias, but its stub only lists `dim`.
    # pyrefly: ignore[missing-argument, unexpected-keyword]
    return a.cumprod(axis=axis, dtype=dtype)


cumproduct = cumprod


def average(
    a: ArrayLike,
    axis: int | None = None,
    weights: ArrayLike | None = None,
    returned: bool = False,
    *,
    keepdims: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if weights is None:
        result = mean(a, axis)
        wsum = torch.as_tensor(a.numel() / result.numel(), dtype=result.dtype)
    else:
        if not a.dtype.is_floating_point:
            a = a.double()

        # axis & weights
        if a.shape != weights.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights differ."
                )
            if weights.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ."
                )
            if weights.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis."
                )

            # setup weight to broadcast along axis
            weights = torch.broadcast_to(weights, (a.ndim - 1) * (1,) + weights.shape)
            weights = weights.swapaxes(-1, axis)

        # do the work
        result_dtype = _dtypes_impl.result_type_impl(a, weights)
        numerator = sum(a * weights, axis, dtype=result_dtype)
        wsum = sum(weights, axis, dtype=result_dtype)
        result = numerator / wsum

    # We process keepdims manually because the decorator does not deal with variadic returns
    if keepdims:
        result = _util.apply_keepdims(result, axis, a.ndim)

    if returned:
        if wsum.shape != result.shape:
            wsum = torch.broadcast_to(wsum, result.shape).clone()
        return result, wsum
    else:
        return result


# Not using deco_axis_expand as it assumes that axis is the second arg
def quantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    interpolation: NotImplementedType = None,
) -> torch.Tensor:
    if overwrite_input:
        # raise NotImplementedError("overwrite_input in quantile not implemented.")
        # NumPy documents that `overwrite_input` MAY modify inputs:
        # https://numpy.org/doc/stable/reference/generated/numpy.percentile.html#numpy-percentile
        # Here we choose to work out-of-place because why not.
        pass

    if not a.dtype.is_floating_point:
        dtype = _dtypes_impl.default_dtypes().float_dtype
        a = a.to(dtype)

    # edge case: torch.quantile only supports float32 and float64
    if a.dtype == torch.float16:
        a = a.to(torch.float32)

    if axis is None:
        a = a.flatten()
        q = q.flatten()
        axis = (0,)
    else:
        axis = _util.normalize_axis_tuple(axis, a.ndim)

    # FIXME(Mario) Doesn't np.quantile accept a tuple?
    # torch.quantile does accept a number. If we don't want to implement the tuple behaviour
    # (it's deffo low prio) change `normalize_axis_tuple` into a normalize_axis index above.
    axis = _util.allow_only_single_axis(axis)

    q = _util.cast_if_needed(q, a.dtype)

    return torch.quantile(a, q, axis, interpolation=method)


def percentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: AxisLike = None,
    out: OutArray | None = None,
    overwrite_input: bool = False,
    method: str = "linear",
    keepdims: KeepDims = False,
    *,
    interpolation: NotImplementedType = None,
) -> torch.Tensor:
    # np.percentile(float_tensor, 30) : q.dtype is int64 => q / 100.0 is float32
    if _dtypes_impl.python_type_for_torch(q.dtype) is int:
        q = q.to(_dtypes_impl.default_dtypes().float_dtype)
    qq = q / 100.0

    return quantile(
        a,
        qq,
        axis=axis,
        overwrite_input=overwrite_input,
        method=method,
        keepdims=keepdims,
        interpolation=interpolation,
    )


def median(
    a: ArrayLike,
    axis: int | None = None,
    out: OutArray | None = None,
    overwrite_input: bool = False,
    keepdims: KeepDims = False,
) -> torch.Tensor:
    return quantile(
        a,
        torch.as_tensor(0.5),
        axis=axis,
        overwrite_input=overwrite_input,
        out=out,
        keepdims=keepdims,
    )
<|endoftext|># mypy: allow-untyped-defs
# pylint: disable=useless-parent-delegation
from __future__ import annotations

from typing import cast, Generic, TYPE_CHECKING, TypeVar

import torch


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["Future", "collect_all", "wait_all"]


T = TypeVar("T")
S = TypeVar("S")


class Future(torch._C.Future, Generic[T]):
    r"""
    Wrapper around a ``torch._C.Future`` which encapsulates an asynchronous
    execution of a callable, e.g. :meth:`~torch.distributed.rpc.rpc_async`. It
    also exposes a set of APIs to add callback functions and set results.

    .. warning:: GPU support is a beta feature, subject to changes.
    """

    def __init__(self, *, devices: list[int | str | torch.device] | None = None):
        r"""
        Create an empty unset ``Future``. If the future is intended to hold
        values containing CUDA tensors, (a superset of) their CUDA devices must
        be specified at construction. (This is only supported if
        ``torch.cuda.is_available()`` returns ``True``). This is needed to
        ensure proper CUDA stream synchronization. The child futures, returned
        by the ``then`` method, will inherit these devices.

        Args:
            devices(``List[Union[int, str, torch.device]]``, optional): the set
                of devices on which tensors contained in this future's value are
                allowed to reside and on which callbacks are allowed to operate.
        """
        if devices is None:
            devices = []
        super().__init__([torch.device(d) for d in devices])

    def done(self) -> bool:
        r"""
        Return ``True`` if this ``Future`` is done. A ``Future`` is done if it
        has a result or an exception.

        If the value contains tensors that reside on GPUs, ``Future.done()``
        will return ``True`` even if the asynchronous kernels that are
        populating those tensors haven't yet completed running on the device,
        because at such stage the result is already usable, provided one
        performs the appropriate synchronizations (see :meth:`wait`).
        """
        return super().done()

    def wait(self) -> T:
        r"""
        Block until the value of this ``Future`` is ready.

        If the value contains tensors that reside on GPUs, then an additional
        synchronization is performed with the kernels (executing on the device)
        which may be asynchronously populating those tensors. Such sync is
        non-blocking, which means that ``wait()`` will insert the necessary
        instructions in the current streams to ensure that further operations
        enqueued on those streams will be properly scheduled after the async
        kernels but, once that is done, ``wait()`` will return, even if those
        kernels are still running. No further synchronization is required when
        accessing and using the values, as long as one doesn't change streams.

        Returns:
            The value held by this ``Future``. If the function (callback or RPC)
            creating the value has thrown an error, this ``wait`` method will
            also throw an error.
        """
        return super().wait()

    def value(self) -> T:
        r"""
        Obtain the value of an already-completed future.

        This method should only be called after a call to :meth:`wait` has
        completed, or inside a callback function passed to :meth:`then`. In
        other cases this ``Future`` may not yet hold a value and calling
        ``value()`` could fail.

        If the value contains tensors that reside on GPUs, then this method will
        *not* perform any additional synchronization. This should be done
        beforehand, separately, through a call to :meth:`wait` (except within
        callbacks, for which it's already being taken care of by :meth:`then`).

        Returns:
            The value held by this ``Future``. If the function (callback or RPC)
            creating the value has thrown an error, this ``value()`` method will
            also throw an error.
        """
        return super().value()

    def then(self, callback: Callable[[Future[T]], S]) -> Future[S]:
        r"""
        Append the given callback function to this ``Future``, which will be run
        when the ``Future`` is completed.  Multiple callbacks can be added to
        the same ``Future``, but the order in which they will be executed cannot
        be guaranteed (to enforce a certain order consider chaining:
        ``fut.then(cb1).then(cb2)``). The callback must take one argument, which
        is the reference to this ``Future``. The callback function can use the
        :meth:`value` method to get the value. Note that if this ``Future`` is
        already completed, the given callback will be run immediately inline.

        If the ``Future``'s value contains tensors that reside on GPUs, the
        callback might be invoked while the async kernels that are populating
        those tensors haven't yet finished executing on the device. However, the
        callback will be invoked with some dedicated streams set as current
        (fetched from a global pool) which will be synchronized with those
        kernels. Hence any operation performed by the callback on these tensors
        will be scheduled on the device after the kernels complete. In other
        words, as long as the callback doesn't switch streams, it can safely
        manipulate the result without any additional synchronization. This is
        similar to the non-blocking behavior of :meth:`wait`.

        Similarly, if the callback returns a value that contains tensors that
        reside on a GPU, it can do so even if the kernels that are producing
        these tensors are still running on the device, as long as the callback
        didn't change streams during its execution. If one wants to change
        streams, one must be careful to re-synchronize them with the original
        streams, that is, those that were current when the callback was invoked.

        Args:
            callback(``Callable``): a ``Callable`` that takes this ``Future`` as
