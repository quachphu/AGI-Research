            nelems: Number of elements to broadcast.
            pe_root: PE number of the root PE that provides the source data.

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - Automatically calculates byte size from tensor type and element count.
            - This is a collective operation - all PEs in the team must participate.
            - Must be called from kernels launched with cooperative launch.

        Example:
            ```
            # Broadcast 100 elements from PE 0 to all PEs
            nvshmem.broadcast(0, dest_tensor, src_tensor, 100, 0)
            ```
        """
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return broadcastmem_block_extern_wrapper(
            team, dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe_root
        )

    @core.extern  # type: ignore[misc]
    def broadcastmem_block_extern_wrapper(
        team: Any,
        dest: Any,
        source: Any,
        size_bytes: Any,
        pe_root: Any,
        _semantic: Any = None,
    ) -> None:
        """Low-level extern wrapper for NVSHMEM broadcast"""
        return core.extern_elementwise(
            "",
            "",
            [team, dest, source, size_bytes, pe_root],
            {
                (
                    core.dtype("int32"),  # team handle
                    core.dtype("int64"),  # dest ptr
                    core.dtype("int64"),  # source ptr
                    core.dtype("int64"),  # size in bytes
                    core.dtype("int32"),  # pe_root
                ): ("nvshmemx_broadcastmem_block", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    # Reduction Operation
    @triton.jit  # type: ignore[misc]
    def reduce(team, dest, source, nreduce, operation: tl.constexpr):  # type: ignore[no-untyped-def]
        """
        Performs a collective reduction on tensors across a team of PEs.

        This high-level function provides a tensor-aware interface for NVSHMEM
        reduction operations. It automatically infers the data type from the
        input tensors and calls the appropriate underlying NVSHMEM function.

        Args:
            team: The team handle for the collective (0 for NVSHMEM_TEAM_WORLD).
            dest: Destination tensor for the reduction results.
            source: Source tensor containing data to be reduced. Must be the same type as dest.
            nreduce: The number of elements in the source tensor to reduce.
            operation: The reduction operation to perform ("sum", "max", "min", "prod").

        Notes:
            - Performs compile-time type checking between dest and source tensors.
            - This is a collective operation that must be called by all PEs in the team.
            - Requires a cooperative grid launch.

        Example:
            ```
            # Perform a sum reduction on two tensors
            nvshmem.reduce(0, dest_tensor, src_tensor, 100, "sum")
            ```
        """
        tl.static_assert(dest.type == source.type)
        dtype = dest.type.element_ty
        return reduce_extern_wrapper(
            team,
            dest.to(tl.int64),
            source.to(tl.int64),
            nreduce.to(tl.int64),
            operation,
            dtype,
        )

    @core.extern  # type: ignore[misc]
    def reduce_extern_wrapper(
        team: Any,
        dest: Any,
        source: Any,
        nreduce: Any,
        operation: str,
        dtype: Any,
        _semantic: Any = None,
    ) -> None:
        """
        Low-level extern wrapper for NVSHMEM reduction operations.

        This function provides a generic interface to NVSHMEM reduction operations,
        automatically selecting the appropriate NVSHMEM function based on the data type
        and operation specified.
        Args:
            team (int64): The team handle (0 for NVSHMEM_TEAM_WORLD).
            dest (pointer): Destination pointer where reduction results are stored.
            source (pointer): Source pointer containing data to be reduced.
            nreduce (int64): Number of elements to reduce.
            operation (str): Reduction operation ("sum", "max", "min", "prod").
            dtype: Data type specification - accepts torch.dtype, tl.dtype, str, or constexpr.
            _semantic: Optional semantic information for Triton compilation.

        Raises:
            ValueError: If the operation is not supported.
            TypeError: If the data type is not supported.

        Example:
            nvshmem.reduce(0, dest_ptr, src_ptr, 100, "sum", torch.float32)
        """
        # Mapping from Triton dtype names to NVSHMEM typenames
        DTYPE_TO_NVSHMEM_MAP = {
            "int8": "int8",
            "int16": "int16",
            "int32": "int32",
            "int64": "int64",
            "uint8": "uint8",
            "uint16": "uint16",
            "uint32": "uint32",
            "uint64": "uint64",
            "fp16": "half",
            "bf16": "bfloat16",
            "fp32": "float",
            "fp64": "double",
        }

        # Triton dtype names are standardized as fp16, bf16, fp32, etc.
        dtype_name = str(dtype).replace("tl.", "")

        if dtype_name not in DTYPE_TO_NVSHMEM_MAP:
            raise TypeError(
                f"Unsupported reduction dtype: {dtype_name}. Supported dtypes: {list(DTYPE_TO_NVSHMEM_MAP.keys())}"
            )

        # Extract operation name from constexpr if needed
        op_name = operation.value if hasattr(operation, "value") else operation

        # Validate operation is supported
        supported_ops = {"sum", "max", "min", "prod"}
        if op_name not in supported_ops:
            raise ValueError(
                f"Unsupported reduction operation: '{op_name}'. Supported ops are {supported_ops}"
            )

        # Map to NVSHMEM typename and validate dtype is supported
        nvshmem_typename = DTYPE_TO_NVSHMEM_MAP.get(dtype_name)
        if nvshmem_typename is None:
            raise TypeError(
                f"Unsupported reduction dtype: {dtype_name}. Supported dtypes are {list(DTYPE_TO_NVSHMEM_MAP.keys())}"
            )

        # Generate NVSHMEM function name
        nvshmem_func = f"nvshmem_{nvshmem_typename}_{op_name}_reduce"

        # Define function signature - all parameters are int64 in Triton (they are just ptrs)
        signature = (
            core.dtype("int32"),  # team handle
            core.dtype("int64"),  # destination pointer
            core.dtype("int64"),  # source pointer
            core.dtype("int64"),  # number of elements
        )

        return core.extern_elementwise(
            "",
            "",
            [team, dest, source, nreduce],
            {signature: (nvshmem_func, core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    # Utility for inspecting Triton kernels

    triton_kernels: dict = {}

    def _log_triton_kernel(kernel) -> None:  # type: ignore[no-untyped-def]
        import atexit
        import tempfile

        if dist.is_initialized() and dist.get_rank() != 0:
            return

        def on_exit() -> None:
            logger.info("PTX files:")
            for kernel in triton_kernels:
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    f.write(kernel.asm["ptx"].encode("utf-8"))
                    logger.info(f"+- {kernel.name}: {f.name}")  # noqa: G004

        if len(triton_kernels) == 0:
            atexit.register(on_exit)

        if kernel not in triton_kernels:
            triton_kernels[kernel] = None
<|endoftext|>from __future__ import annotations

import math
import os
import socket
import uuid
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import timedelta
from enum import Enum
from typing import Any, Literal
from typing_extensions import deprecated

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory, Work as _Work
from torch._prims_common import make_contiguous_strides_for
from torch.utils._triton import has_triton


_group_name_to_store: dict[str, c10d.Store] = {}


@deprecated(
    "`enable_symm_mem_for_group` is deprecated. There is no need to call this function anymore.",
    category=FutureWarning,
)
def enable_symm_mem_for_group(group_name: c10d.GroupName) -> None:
    """
    Enables symmetric memory for a process group.

    Args:
        group_name (str): the name of the process group.
    """
    if group_name in _group_name_to_store:
        return

    group = c10d._resolve_process_group(group_name)
    global_ranks = sorted(c10d._world.pg_group_ranks[group].keys())
    # Different subgroups with the same name should use different stores
    global_ranks_str = "_".join(map(str, global_ranks))
    store = c10d.PrefixStore(
        f"symmetric_memory-{global_ranks_str}",
        c10d._get_process_group_store(group),
    )
    _group_name_to_store[group_name] = store
    _SymmetricMemory.set_group_info(
        group_name,
        group.rank(),
        group.size(),
        store,
    )


_is_test_mode: bool = False
_mocked_group_names: set[str] | None = None


@contextmanager
def _test_mode(group_names: set[str] | None = None) -> Generator[None, None, None]:
    """
    Forces ``is_symm_mem_enabled_for_group()`` to return ``True`` and the ops
    defined in the ``symm_mem`` namespace to use fallback implementations.

    The context manager is not thread safe.
    """
    global _is_test_mode
    global _mocked_group_names
    prev = _is_test_mode
    prev_group_names = _mocked_group_names
    try:
        _is_test_mode = True
        _mocked_group_names = group_names
        yield
    finally:
        _is_test_mode = prev
        _mocked_group_names = prev_group_names


def is_symm_mem_enabled_for_group(group_name: c10d.GroupName) -> bool:
    """
    Check if symmetric memory is enabled for a process group.

    Args:
        group_name (str): the name of the process group.
    """
    if _is_test_mode:
        return _mocked_group_names is None or group_name in _mocked_group_names
    return group_name in _group_name_to_store


_group_name_to_workspace_tensor: dict[str, torch.Tensor | None] = {}


def get_symm_mem_workspace(
    group_name: c10d.GroupName, min_size: int
) -> _SymmetricMemory:
    """
    Get the symmetric memory workspace associated with the process group. If
    ``min_size`` is greater than the workspace associated with ``group_name``,
    the workspace will be re-allocated and re-rendezvous'd.

    Args:
        group_name (str): the name of the process group.
        min_size (int): the size requirement for the workspace in bytes.

    Returns:
        _SymmetricMemory: the symmetric memory workspace associated with the
        group.
    """
    tensor = _group_name_to_workspace_tensor.get(group_name)
    size = tensor.numel() * tensor.element_size() if tensor is not None else 0
    if tensor is None or size < min_size:
        if (
            torch.accelerator.is_available()
            and torch.accelerator.current_stream().is_capturing()
        ):
            curr_size = 0 if tensor is None else tensor.numel() * tensor.element_size()
            raise RuntimeError(
                f"get_symm_mem_workspace(): the requested size ({min_size} bytes) "
                "is greater than the size of the currently allocated workspace "
                f"({curr_size} bytes). It's currently not possible to expand the "
                "workspace size during graph capture. Please invoke "
                f'`get_symm_mem_workspace(group_name="{group_name}", '
                f'min_size="{min_size}")` before initiating the graph capture '
                "and try again."
            )
        tensor = _SymmetricMemory.empty_strided_p2p(
            (max(size, min_size),),
            [1],
            torch.uint8,
            torch.device(torch.accelerator.current_device_index()),
            group_name,
        )
        _group_name_to_workspace_tensor[group_name] = tensor
    return _SymmetricMemory.rendezvous(tensor)


_backend_streams: dict[int, torch.Stream] = {}


def _get_backend_stream(priority: int = 0) -> torch.Stream:
    if priority not in _backend_streams:
        _backend_streams[priority] = torch.Stream(priority=priority)
    return _backend_streams[priority]


def _pipelined_multi_all_gather_and_consume(
    shard: list[torch.Tensor],
    shard_consumer: Callable[[list[torch.Tensor], int], None],
    ag_out: list[torch.Tensor],
    group_name: c10d.GroupName,
    ag_out_needed: bool = True,
) -> None:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        gathered = [
            all_gather_single(x, gather_dim=0, group=group)
            for x in shard
        ]

        shards = [[] for _ in range(group_size)]
        for x in ag_out:
            for i, y in enumerate(x.chunk(group_size)):
                shards[i].append(y)

        for src_rank, shard in enumerate(shards):
            shard_consumer(shard, src_rank)
    """
    p2p_workspace_size_req = 0
    for x in shard:
        p2p_workspace_size_req += x.numel() * x.element_size()
    symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    symm_mem.barrier(channel=0)
    backend_stream = _get_backend_stream()
    backend_stream.wait_stream(torch.accelerator.current_stream())

    for x, y in zip(shard, ag_out):
        if not x.is_contiguous():
            raise AssertionError(
                "_pipelined_all_gather_and_consume: all tensors "
                "in `shard` must be contiguous"
            )
        if not y.is_contiguous():
            raise AssertionError(
                "_pipelined_all_gather_and_consume: all tensors "
                "in `ag_out` must be contiguous"
            )
        if x.shape[0] * group_size != y.shape[0]:
            raise AssertionError
        if x.shape[1:] != y.shape[1:]:
            raise AssertionError

    def copy_shard(dst: list[torch.Tensor], src: list[torch.Tensor]) -> None:
        for d, s in zip(dst, src):
            d.copy_(s)

    def get_p2p_bufs(remote_rank: int) -> list[torch.Tensor]:
        offset_bytes = 0
        bufs = []
        for x in shard:
            buf = symm_mem.get_buffer(
                remote_rank,
                x.shape,
                x.dtype,
                storage_offset=offset_bytes // x.element_size(),
            )
            bufs.append(buf)
            offset_bytes += buf.numel() * buf.element_size()
        return bufs

    local_p2p_bufs = get_p2p_bufs(rank)

    # shards[i] => shard from rank i
    shards: list[list[torch.Tensor]] = [[] for _ in range(group_size)]
    for x in ag_out:
        for i, y in enumerate(x.chunk(group_size)):
            shards[i].append(y)

    # Parallelization strategy: after each rank copies its shard into its local
    # p2p buffer, every rank issues independent p2p copy -> shard_consumer
    # sequences to two streams. In addition to computation/communication
    # overlapping, the strategy allows for computation/computation overlapping,
    # greatly reducing quantization inefficiency.
    #
    # Notation:
    # - "mv" for the copy to local buffer
    # - "cp" for p2p copies
    # - "b" for barriers
    #
    # Constraints:
    # - The GPU scheduler may or may not overlap "mv" with the first shard_consumer.
    # - "cp" from different streams cannot overlap.
    #
    # Ideal scenario 0 - "mv" overlaps with the first shard_consumer:
    #
    # stream 0: [ shard_consumer ][ cp ][ shard_consumer ]
    # stream 1: [ mv ][b][ cp ][ shard_consumer ]
    #
    # Ideal scenario 1 - "mv" is scheduled before the first shard_consumer:
    #
    # stream 0:       [ shard_consumer ][ cp ][ shard_consumer ]
    # stream 1: [ mv ][b][ cp ][ shard_consumer ]
    #
    # Suboptimal scenario 0 - "mv" is scheduled after the first shard_consumer:
    #
    # stream 0: [ shard_consumer ]               [ cp ][ shard_consumer ]
    # stream 1:                   [ mv ][b][ cp ][ shard_consumer ]
    #
    # Suboptimal scenario 0 - "b" is scheduled after the first shard_consumer:
    #
    # stream 0:       [ shard_consumer ]         [ cp ][ shard_consumer ]
    # stream 1: [ mv ]                  [b][ cp ][ shard_consumer ]
    #
    # We haven't yet figured out a way to ensure "mv" and "b" are either
    # overlapped with or scheduled before the first shard_consumer. Thus, to
    # prevent suboptimal scenarios, we are giving up the chance to overlap "mv"
    # and "b" with the first shard_consumer for now.
    copy_shard(dst=local_p2p_bufs, src=shard)
    symm_mem.barrier(channel=1)
    backend_stream.wait_stream(torch.accelerator.current_stream())

    # At this point, all ranks have copied their local shard to
    # their local p2p buffer. Each rank can now copy and consume
    # remote shards.
    shard_consumer(shard, rank)

    for step in range(1, group_size):
        if step % 2 == 0:
            stream = torch.accelerator.current_stream()
        else:
            stream = backend_stream
        remote_rank = (step + rank) % group_size
        remote_p2p_bufs = get_p2p_bufs(remote_rank)
        with stream:
            copy_shard(dst=shards[remote_rank], src=remote_p2p_bufs)
            shard_consumer(shards[remote_rank], remote_rank)

    if ag_out_needed:
        # Copy from input to the all-gather output. Opportunistically overlap
        # it with the last shard_consumer.
        if group_size % 2 == 0:
            stream = torch.accelerator.current_stream()
        else:
            stream = backend_stream
        with stream:
            copy_shard(dst=shards[rank], src=shard)

    torch.accelerator.current_stream().wait_stream(backend_stream)
    symm_mem.barrier(channel=0)


def _pipelined_all_gather_and_consume(
    shard: torch.Tensor,
    shard_consumer: Callable[[torch.Tensor, int], None],
    ag_out: torch.Tensor,
    group_name: c10d.GroupName,
    ag_out_needed: bool = True,
) -> None:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        ag_out = all_gather_single(shard, gather_dim=0, group=group)
        shards = ag_out.chunk(group.size())
        for src_rank, shard in enumerate(shards):
            shard_consumer(shard, src_rank)
    """

    def adapter(shard: list[torch.Tensor], rank: int) -> None:
        shard_consumer(shard[0], rank)

    _pipelined_multi_all_gather_and_consume(
        [shard],
        adapter,
        [ag_out],
        group_name,
        ag_out_needed,
    )


def _pipelined_produce_and_all2all(
    chunk_producer: Callable[[int, torch.Tensor], None],
    output: torch.Tensor,
    group_name: c10d.GroupName,
    out_chunk_dim: int = 0,
) -> None:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        chunks = [
            chunk_producer(dst_rank, chunks[dst_rank])
            for dst_rank in range(group_size):
        ]
        dist.all_to_all_single(output=output, input=torch.cat(chunks))
    """
    out_chunks = output.chunk(
        c10d._get_group_size_by_name(group_name), dim=out_chunk_dim
    )
    p2p_workspace_size_req = out_chunks[0].numel() * out_chunks[0].element_size() * 2
    symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    symm_mem.barrier(channel=0)
    backend_stream = _get_backend_stream()
    backend_stream.wait_stream(torch.accelerator.current_stream())

    def get_p2p_buf(rank: int, idx: int) -> torch.Tensor:
        if idx not in (0, 1):
            raise AssertionError
        offset = 0 if idx == 0 else out_chunks[0].numel()
        return symm_mem.get_buffer(
            rank, out_chunks[0].shape, out_chunks[0].dtype, offset
        )

    # Prepare two local p2p buffers, so that a remote rank can pull the result
    # of step [i] in one p2p buffer while the local rank can compute the
    # result of step [i+1] and write it directly the other p2p buffer.
    local_p2p_buf_0 = get_p2p_buf(rank, 0)
    local_p2p_buf_1 = get_p2p_buf(rank, 1)

    for step in range(1, group_size):
        remote_rank = (rank - step) % group_size
        if step % 2 == 0:
            stream = torch.accelerator.current_stream()
            p2p_buf = local_p2p_buf_1
            remote_p2p_buf = get_p2p_buf(remote_rank, 1)
        else:
            stream = backend_stream
            p2p_buf = local_p2p_buf_0
            remote_p2p_buf = get_p2p_buf(remote_rank, 0)
        with stream:
            # Parallelization strategy: every rank issues independent compute
            # -> barrier -> p2p copy sequences on two streams. In addition to
            # computation/communication overlapping, the strategy allows for
            # computation/computation overlapping, greatly reducing
            # quantization inefficiency.
            #
            # Ideally, stream activities would look like this ("b" for
            # barriers, "cp" for p2p copies):
            #
            # [rank 0]
            # stream 0:         [  chunk_producer  ][b][ cp ][  chunk_producer ][b][ cp ]
            # stream 1: [  chunk_producer  ][b][ cp ][  chunk_producer  ][b][ cp ]
            #
            # [rank 1]
            # stream 0:         [  chunk_producer  ][b][ cp ][  chunk_producer ][b][ cp ]
            # stream 1: [  chunk_producer  ][b][ cp ][  chunk_producer  ][b][ cp ]
            #
            # Note that the barriers synchronize streams with the same ID
            # across ranks. They don't synchronize streams on the same rank.
            #
            # Since the work on both streams is independent, there's no
            # guarantee that the chunk_producer from stream 0 or stream 1 will
            # be scheduled first. If there is a scheduling mismatch across
            # ranks, the barrier forces all ranks to wait for the slowest.
            #
            # When scheduling mismatches occur among ranks, the stream
            # activities might look like this (note that p2p copies from
            # different streams cannot overlap with each other):
            #
            # [rank 0]
            # stream 0: [  chunk_producer  ][b        ][ cp ][  chunk_producer ][b       ][ cp ]
            # stream 1:         [  chunk_producer  ][b]      [ cp ][  chunk_producer  ][b]      [ cp ]
            #
            # [rank 1]
            # stream 0:         [  chunk_producer  ][b]      [ cp ][  chunk_producer  ][b]      [ cp ]
            # stream 1: [  chunk_producer  ][b        ][ cp ][  chunk_producer  ][b      ][ cp ]
            #
            # To prevent this, we need to ensure that the chunk_producer on
            # stream 1 gets scheduled first on every rank. Without access to
            # the underlying kernels, CUDA offers no API to control the
            # scheduling order of two independent, overlapping kernels. Our
            # solution is to issue a small sleep kernel in stream 0. The sleep
            # duration is insignificant, but having an extra task in stream 0
            # will almost guarantee that the chunk_producer on stream 1 gets
            # scheduled first. Once the first chunk_producer is scheduled in
            # the correct order, there's very little room for the scheduling
            # order of subsequent kernels to be inconsistent across ranks.
            if step == 2 and torch.cuda.is_available():
                torch.cuda._sleep(100)
            chunk_producer((rank + step) % group_size, p2p_buf)
            symm_mem.barrier(channel=step % 2)
            out_chunks[remote_rank].copy_(remote_p2p_buf)
            # The local P2P buffer can only be overwritten by the next
            # chunk_producer after all peers have finished reading from it.
            symm_mem.barrier(channel=step % 2)

    # If the sleep wasn't issued in the above loop, do it now.
    if group_size == 2 and torch.cuda.is_available():
        torch.cuda._sleep(100)

    chunk_producer(rank, out_chunks[rank])
    torch.accelerator.current_stream().wait_stream(backend_stream)
    symm_mem.barrier(channel=0)


def reduce_partials(
    partials: torch.Tensor,
    *,
    dim: int,
    reduce_op: str,
    output_dtype: torch.dtype,
    group_size: int,
) -> torch.Tensor:
    if output_dtype in (torch.float16, torch.bfloat16):
        normalized_dim = dim if dim >= 0 else dim + partials.dim()
        # The common RS layout is [rank, local_M, N]. Fuse the final rank
        # reduction into one kernel so low-precision partials accumulate in fp32
        # and round only once when written to the output dtype.
        if normalized_dim == 0:
            reduced = triton_reduce_partials_first_dim(
                partials,
                reduce_op=reduce_op,
                output_dtype=output_dtype,
                group_size=group_size,
            )
            if reduced is not None:
                return reduced
        if reduce_op == "sum":
            return torch.sum(partials, dim=dim)
        if reduce_op == "avg":
            return torch.mean(partials, dim=dim)
    if reduce_op == "sum":
        return torch.sum(partials, dim=dim)
    if reduce_op == "avg":
        return torch.mean(partials, dim=dim)
    raise ValueError("reduce_op must be sum or avg")


def triton_reduce_partials_first_dim(
    partials: torch.Tensor,
    *,
    reduce_op: str,
    output_dtype: torch.dtype,
    group_size: int,
) -> torch.Tensor | None:
    if (
        partials.device.type != "cuda"
        or not has_triton()
        or not partials.is_contiguous()
        or partials.dim() < 2
        or partials.shape[0] != group_size
        or output_dtype not in (torch.float16, torch.bfloat16)
        or partials.dtype not in (torch.float16, torch.bfloat16)
    ):
        return None

    out = partials.new_empty(partials.shape[1:], dtype=output_dtype)
    shard_elems = out.numel()
    block = 1024
    import triton

    from torch.distributed._symmetric_memory._triton_kernels import (
        reduce_partials_first_dim_kernel,
    )

    grid = (triton.cdiv(shard_elems, block),)
    reduce_partials_first_dim_kernel[grid](
        partials,
        out,
        shard_elems,
        group_size,
        reduce_op == "avg",
        BLOCK=block,
    )
    return out


lib = torch.library.Library("symm_mem", "DEF")
lib.define(
    "fused_all_gather_matmul("
    "Tensor A, Tensor[] Bs, int gather_dim, str group_name, *, bool return_A = True) -> (Tensor?, Tensor[])",
    tags=[torch._C.Tag.needs_fixed_stride_order],
)
lib.define(
    "fused_all_gather_scaled_matmul("
    "Tensor A, Tensor[] Bs, Tensor A_scale, Tensor[] B_scales, "
    "int gather_dim, str group_name, "
    "Tensor?[] biases, "
    "Tensor?[] result_scales, "
    "ScalarType?[] out_dtypes, "
    "bool[] use_fast_accum) -> (Tensor, Tensor[])",
    tags=[torch._C.Tag.needs_fixed_stride_order],
)
lib.define(
    "fused_matmul_reduce_scatter(Tensor A, Tensor B, str reduce_op, int scatter_dim, str group_name) -> Tensor",
    tags=[torch._C.Tag.needs_fixed_stride_order],
)
lib.define(
    "fused_scaled_matmul_reduce_scatter("
    "Tensor A, Tensor B, Tensor A_scale, Tensor B_scale, "
    "str reduce_op, int orig_scatter_dim, int scatter_dim_after_maybe_reshape, str group_name, SymInt[]? output_shape, "
    "Tensor? bias = None, "
    "Tensor? result_scale = None, "
    "ScalarType? out_dtype = None, "
    "bool use_fast_accum = False) -> Tensor",
    tags=[torch._C.Tag.needs_fixed_stride_order],
)
lib.define("_low_contention_all_gather(Tensor tensor, str group_name) -> Tensor")
lib.define(
    "_low_contention_reduce_scatter(Tensor tensor, str reduce_op, str group_name) -> Tensor"
)
# Copy-engine multicast low-contention all-gather that allocates symm_mem output.
lib.define(
    "_low_contention_all_gather_ce_multicast(Tensor tensor, str group_name) -> Tensor"
)
# Out variant for preallocated symm_mem output, including CUDA graph capture.
lib.define(
    "_low_contention_all_gather_ce_multicast_out("
    "Tensor tensor, str group_name, Tensor(a!) out) -> Tensor(a!)"
)

lib.define("get_remote_tensors(Tensor x, str group_name) -> Tensor[]")
"""
Given a local tensor and a group name, return a tuple of tensors that are
symmetric on other devices. The returned tensors are ordered by rank IDs. The
length of the tuple equals to the size of the group.

Note: this API works only when `world_within_direct_access()` returns True, i.e.
only when the group is within NVLink domain or similar. It does not work across
network interfaces.
"""


@torch.library.impl(lib, "get_remote_tensors", "CUDA")
@torch.library.impl(lib, "get_remote_tensors", "XPU")
def _get_remote_tensors_default(
    local: torch.Tensor, group_name: c10d.GroupName
) -> tuple[torch.Tensor, ...]:
    hdl = rendezvous(local, group_name)
    if hdl is None:
        raise ValueError("Tensor is not allocated from Symmetric Memory")

    return tuple(
        hdl.get_remote_tensor(peer, local.size(), local.dtype)
        for peer in range(hdl.world_size)
    )


@torch.library.impl(lib, "get_remote_tensors", "Meta")
def _get_remote_tensors_meta(
    local: torch.Tensor, group_name: c10d.GroupName
) -> tuple[torch.Tensor, ...]:
    group = c10d._resolve_process_group(group_name)
    return tuple(torch.empty_like(local) for _ in range(group.size()))


class _ScaleMode(Enum):
    UNSCALED = "unscaled"
    TENSOR_WISE = "tensor-wise"
    ROW_WISE_SHARDED = "row-wise-sharded"
    ROW_WISE_REPLICATED = "row-wise-replicated"


def _check_and_verify_fp8_all_gather_scale_mode(
    shard: torch.Tensor, scale: torch.Tensor | None, gather_dim: int, group_size: int
) -> _ScaleMode:
    full_shape = list(shard.shape)
    full_shape[gather_dim] *= group_size

    if scale is None:
        return _ScaleMode.UNSCALED
    elif scale.shape[:-1] == shard.shape[:-1] and scale.shape[-1] == 1:
        # Row-wise scaling
        #
        # NOTE: when the last dim of both A_shard and A_scale is one, we can't
        # tell if A_scale is replicated tensor-wise scale or sharded row-wise
        # scale. Treating it as row-wise scaling for safety.
        return _ScaleMode.ROW_WISE_SHARDED
    elif scale.numel() == 1:
        return _ScaleMode.TENSOR_WISE
    elif list(scale.shape[:-1]) == full_shape[:-1]:
        return _ScaleMode.ROW_WISE_REPLICATED
    else:
        raise ValueError(
            "Invalid scale shape for fp8 all-gather "
            f"(shard shape: {shard.shape}, scale shape: {scale.shape})"
        )


def _fused_all_gather_matmul_impl(
    mm_out_op: torch._ops.OpOverload,
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    A_scale: torch.Tensor | None,
    kwargs_list: list[dict[str, Any]],
    out_dtypes: list[torch.dtype | None],
    gather_dim: int,
    group_name: c10d.GroupName,
    return_A: bool,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    if A_shard.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    for B in Bs:
        if B.dim() != 2:
            raise ValueError("B must be a matrix")
    if len(out_dtypes) != len(Bs):
        raise ValueError("len(out_types) must be the same as len(Bs)")
    if len(kwargs_list) != len(Bs):
        raise ValueError("len(kwargs_list) must be the same as len(Bs)")
    if gather_dim < 0 or gather_dim >= A_shard.dim():
        raise ValueError("Invalid gather_dim")

    group = c10d._resolve_process_group(group_name)

    if gather_dim == A_shard.ndim - 1 or gather_dim == -1:
        return _fused_all_gather_matmul_last_gather_dim_impl(
            mm_out_op,
            A_shard,
            Bs,
            A_scale,
            kwargs_list,
            out_dtypes,
            gather_dim,
            group_name,
            return_A,
        )

    # Move the gather_dim to the front and flatten the tensor into a 2D matrix.
    # The flattened tensor doesn't need to be contiguous (for computation
    # efficiency), as _pipelined_all_gather_and_consume guarantees that shards
    # passed to shard_consumer are contiguous.
    A_shard_flat = A_shard.movedim(gather_dim, 0)
    leading_dims = [group.size()] + list(A_shard_flat.shape[:-1])
    A_shard_flat = A_shard_flat.flatten(0, -2)

    # Helper function for reverting the above transformation
    def unflatten(t: torch.Tensor) -> torch.Tensor:
        return t.view(*leading_dims, -1).flatten(0, 1).movedim(0, gather_dim)

    A_flat = A_shard_flat.new_empty(
        A_shard_flat.shape[0] * group.size(),
        A_shard_flat.shape[1],
    )

    outputs = [
        A_flat.new_empty(A_flat.shape[0], B.shape[1], dtype=out_dtype or B.dtype)
        for B, out_dtype in zip(Bs, out_dtypes)
    ]
    output_shards = [output.chunk(group.size()) for output in outputs]

    scale_mode = _check_and_verify_fp8_all_gather_scale_mode(
        shard=A_shard, scale=A_scale, gather_dim=gather_dim, group_size=group.size()
    )

    # Computing block-wise matmul along the first dim of A
    if scale_mode == _ScaleMode.ROW_WISE_SHARDED:
        if A_scale is None:
            raise AssertionError
        A_scale_shard = A_scale.movedim(gather_dim, 0).flatten(0, -2)
        A_scale_flat = A_scale_shard.new_empty(
            A_scale_shard.shape[0] * group.size(),
            A_scale_shard.shape[1],
        )

        def row_wise_sharded_consumer(shard: list[torch.Tensor], rank: int) -> None:
            for idx, (B, kwargs) in enumerate(zip(Bs, kwargs_list)):
                mm_out_op(
                    shard[0],
                    B,
                    scale_a=shard[1],
                    **kwargs,
                    out=output_shards[idx][rank],
                )

        _pipelined_multi_all_gather_and_consume(
            [A_shard_flat, A_scale_shard],
            row_wise_sharded_consumer,
            [A_flat, A_scale_flat],
            group_name,
            return_A,
        )
    elif scale_mode == _ScaleMode.ROW_WISE_REPLICATED:
        if A_scale is None:
            raise AssertionError
        A_scale_shards = (
            A_scale.movedim(gather_dim, 0).flatten(0, -2).chunk(group.size())
        )

        def row_wise_replicated_consumer(shard: torch.Tensor, rank: int) -> None:
            for idx, (B, kwargs) in enumerate(zip(Bs, kwargs_list)):
                mm_out_op(
                    shard,
                    B,
                    scale_a=A_scale_shards[rank],
                    **kwargs,
                    out=output_shards[idx][rank],
                )

        _pipelined_all_gather_and_consume(
            A_shard_flat,
            row_wise_replicated_consumer,
            A_flat,
            group_name,
            return_A,
        )
    else:
        if scale_mode == _ScaleMode.TENSOR_WISE:
            if A_scale is None:
                raise AssertionError
            for kwargs in kwargs_list:
                kwargs["scale_a"] = A_scale
        else:
            if scale_mode != _ScaleMode.UNSCALED:
                raise AssertionError

        def default_consumer(shard: torch.Tensor, rank: int) -> None:
            for idx, (B, kwargs) in enumerate(zip(Bs, kwargs_list)):
                mm_out_op(shard, B, **kwargs, out=output_shards[idx][rank])

        _pipelined_all_gather_and_consume(
            A_shard_flat,
            default_consumer,
            A_flat,
            group_name,
            return_A,
        )

    A = unflatten(A_flat) if return_A else None
    return A, [unflatten(output) for output in outputs]


def _pipelined_all_gather_and_consume_last_dim(
    shard: torch.Tensor,
    shard_consumer: Callable[[torch.Tensor, int], None],
    ag_out: torch.Tensor,
    group_name: c10d.GroupName,
    ag_out_needed: bool = True,
) -> None:
    p2p_workspace_size_req = 0
    p2p_workspace_size_req = shard.numel() * shard.element_size()
    symm_mem = get_symm_mem_workspace(group_name, min_size=p2p_workspace_size_req)
    group_size = symm_mem.world_size
    rank = symm_mem.rank

    symm_mem.barrier(channel=0)
    backend_stream = _get_backend_stream()
    backend_stream.wait_stream(torch.accelerator.current_stream())

    def copy_shard(dst: torch.Tensor, src: torch.Tensor) -> None:
        dst.copy_(src)

    def get_p2p_buf(remote_rank: int) -> torch.Tensor:
        buf = symm_mem.get_buffer(
            remote_rank,
            shard.shape,
            shard.dtype,
        )
        return buf

    local_p2p_buf = get_p2p_buf(rank)

    shards = ag_out.chunk(group_size)

    copy_shard(dst=local_p2p_buf, src=shard)
    symm_mem.barrier(channel=1)
    backend_stream.wait_stream(torch.accelerator.current_stream())

    # At this point, all ranks have copied their local shard to
    # their local p2p buffer. Each rank can now copy and consume
    # remote shards.
    shard_consumer(shard, rank)

    for step in range(1, group_size):
        if step % 2 == 0:
            stream = torch.accelerator.current_stream()
        else:
            stream = backend_stream
        remote_rank = (step + rank) % group_size
        remote_p2p_buf = get_p2p_buf(remote_rank)
        with stream:
            copy_shard(dst=shards[remote_rank], src=remote_p2p_buf)
            shard_consumer(shards[remote_rank], remote_rank)

    if ag_out_needed:
        # Copy from input to the all-gather output. Opportunistically overlap
        # it with the last shard_consumer.
        if group_size % 2 == 0:
            stream = torch.accelerator.current_stream()
        else:
            stream = backend_stream
        with stream:
            copy_shard(dst=shards[rank], src=shard)

    torch.accelerator.current_stream().wait_stream(backend_stream)
    symm_mem.barrier(channel=0)


def _fused_all_gather_matmul_last_gather_dim_impl(
    mm_out_op: torch._ops.OpOverload,
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    A_scale: torch.Tensor | None,
    kwargs_list: list[dict[str, Any]],
    out_dtypes: list[torch.dtype | None],
    gather_dim: int,
    group_name: c10d.GroupName,
    return_A: bool,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    group = c10d._resolve_process_group(group_name)
    group_size = group.size()

    B_shards = [B.chunk(group.size()) for B in Bs]

    leading_dims = list(A_shard.shape[:-1])
    A_shard_flat = A_shard.flatten(0, -2)

    def unflatten(t: torch.Tensor) -> torch.Tensor:
        return t.view(*leading_dims, -1)

    A_flat_out = A_shard_flat.new_empty(
        A_shard_flat.shape[0] * group.size(),
        A_shard_flat.shape[1],
    )

    outputs = [
        torch.empty(
            (A_shard_flat.shape[0], B.shape[1]),
            dtype=out_dtype or B.dtype,
            device=A_shard.device,
        )
        for B, out_dtype in zip(Bs, out_dtypes)
    ]

    first = True
    events = [torch.Event() for _ in outputs]

    def default_consumer(shard: torch.Tensor, rank: int) -> None:
        nonlocal first
        for out, event, B_shard, kwargs in zip(outputs, events, B_shards, kwargs_list):
            event.wait()
            if first:
                torch.ops.aten.mm.out(shard, B_shard[rank], **kwargs, out=out)
            else:
                out.addmm_(shard, B_shard[rank])
            event.record()

        first = False

    _pipelined_all_gather_and_consume_last_dim(
        A_shard_flat,
        default_consumer,
        A_flat_out,
        group_name,
        return_A,
    )
    ret_A = None
    if return_A:
        # This path is inefficient and will be filtered out at passes stage
        # Added only for completeness.
        A_split_cat_out_flat = torch.cat(A_flat_out.chunk(group_size), dim=-1)
        ret_A = unflatten(A_split_cat_out_flat)

    return ret_A, [unflatten(output) for output in outputs]


@torch.library.impl(lib, "fused_all_gather_matmul", "Meta")
def _fused_all_gather_matmul_fallback(
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    gather_dim: int,
    group_name: c10d.GroupName,
    *,
    return_A: bool = True,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    group_size = c10d._get_group_size_by_name(group_name)
    A = torch.ops._c10d_functional.all_gather_into_tensor(
        A_shard.contiguous(), group_size, group_name
    )
    A = torch.ops._c10d_functional.wait_tensor(A)
    if gather_dim == A.ndim - 1 or gather_dim == -1:
        A_splits = A.chunk(group_size)
        A_mm = torch.cat(A_splits, dim=-1)
        res = [torch.matmul(A_mm, B) for B in Bs]
        if return_A:
            return A_mm, res
        else:
            return None, res

    A = A.view(group_size, *A_shard.shape).movedim(gather_dim + 1, 1).flatten(0, 1)
    res = [torch.matmul(A, B).movedim(0, gather_dim) for B in Bs]
    if return_A:
        return A.movedim(0, gather_dim), res
    else:
        return None, res


@torch.library.impl(lib, "fused_all_gather_matmul", "CUDA")
@torch.library.impl(lib, "fused_all_gather_matmul", "XPU")
def _fused_all_gather_matmul(
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    gather_dim: int,
    group_name: c10d.GroupName,
    *,
    return_A: bool = True,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        all_gather_single(A_shard, gather_dim, group_name) @ B

    Optimal stride order for A_shard - if A_shard.movedim(gather_dim, 0) is
    contiguous, no extra copy is required for input layout transformation.
    Otherwise A_shard needs to be copied once.
    """
    if _is_test_mode:
        return _fused_all_gather_matmul_fallback(
            A_shard, Bs, gather_dim, group_name, return_A=return_A
        )

    if _should_use_fused_all_gather_matmul_native(A_shard, Bs, gather_dim, group_name):
        group = c10d._resolve_process_group(group_name)
        leading_dims = list(A_shard.shape[:-1])
        leading_dims[0] *= group.size()
        A, out = _fused_all_gather_matmul_native(
            A_shard.flatten(0, -2), Bs[0], group_name
        )
        return A.view(*leading_dims, -1), [out.view(*leading_dims, -1)]

    if _should_use_multimem_all_gather_matmul(
        A_shard, gather_dim, group_name, return_A
    ):
        return None, _multimem_all_gather_matmul(A_shard, Bs, group_name)

    with torch.profiler.record_function("fused_all_gather_matmul"):
        return _fused_all_gather_matmul_impl(
            torch.ops.aten.mm.out,
            A_shard,
            Bs,
            None,
            [{} for B in Bs],
            [B.dtype for B in Bs],
            gather_dim,
            group_name,
            return_A,
        )


def _should_use_fused_all_gather_matmul_native(
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    gather_dim: int,
    group_name: c10d.GroupName,
) -> bool:
    group = c10d._resolve_process_group(group_name)
    local_M = math.prod(A_shard.shape[:-1])

    return (
        "TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP" in os.environ
        and torch.cuda.is_available()
        and A_shard.device.type == "cuda"
        and A_shard.is_contiguous()
        and gather_dim == 0
        # _async_input_mm requires local_M to be divisible by world_size.
        and local_M % group.size() == 0
        # _async_input_mm outperforms the decomposition-based approach when the
        # global M is small.
        and 2048 < local_M * group.size() <= 4096
        # _async_input_mm only supports a single B.
        and len(Bs) == 1
    )


def _fused_all_gather_matmul_native(
    A_shard: torch.Tensor,
    B: torch.Tensor,
    group_name: c10d.GroupName,
) -> tuple[torch.Tensor, torch.Tensor]:
    symm_mem = rendezvous(A_shard, group_name)
    if symm_mem is None:
        symm_mem = get_symm_mem_workspace(
            group_name, A_shard.numel() * A_shard.element_size()
        )
        symm_mem.barrier()
        buf = symm_mem.get_buffer(symm_mem.rank, A_shard.shape, A_shard.dtype)
        buf.copy_(A_shard)
        A_shard = buf

    rank = symm_mem.rank
    world_size = symm_mem.world_size

    current_stream = torch.cuda.current_stream()
    backend_stream = _get_backend_stream(priority=-1)

    symm_mem.barrier()
    backend_stream.wait_stream(current_stream)
    current_stream.wait_stream(backend_stream)

    A = A_shard.new_empty(A_shard.shape[0] * world_size, A_shard.shape[1])
    A_signals = torch.zeros(world_size, dtype=torch.uint32, device=A_shard.device)
    A_shards = A.chunk(world_size)

    A_shards[rank].copy_(A_shard)
    if not torch.cuda.is_current_stream_capturing():
        _SymmetricMemory.stream_write_value32(A_signals, rank, 1)
    else:
        _SymmetricMemory.memset32(A_signals, offset=rank, val=1, count=1)

    out = torch.ops.symm_mem._async_input_mm(A, B, A_signals, rank)
    for step in range(1, world_size):
        src_rank = (rank + step) % world_size
        src_buf = symm_mem.get_buffer(src_rank, A_shard.shape, A_shard.dtype)
        with backend_stream:
            A_shards[src_rank].copy_(src_buf)
            if not torch.cuda.is_current_stream_capturing():
                # cuStreamWriteValue32 issues a system level fence before the write
                _SymmetricMemory.stream_write_value32(A_signals, src_rank, 1)
            else:
                _SymmetricMemory.memset32(A_signals, offset=src_rank, val=1, count=1)

    current_stream.wait_stream(backend_stream)
    backend_stream.wait_stream(current_stream)

    symm_mem.barrier()
    return A, out


def _should_use_multimem_all_gather_matmul(
    A_shard: torch.Tensor,
    gather_dim: int,
    group_name: c10d.GroupName,
    return_A: bool,
) -> bool:
    group = c10d._resolve_process_group(group_name)
    local_M = math.prod(A_shard.shape[:-1])
    has_multicast_support = (
        A_shard.device.type == "cuda"
        and _SymmetricMemory.has_multicast_support(
            DeviceType.CUDA, A_shard.device.index
        )
    )

    return (
        has_multicast_support
        and not return_A
        and A_shard.is_contiguous()
        and gather_dim == 0
        # The heuristic is empirical. We could refine it with a more
        # sophisticated perf model.
        and local_M * group.size() <= 2048
    )


def _multimem_all_gather_matmul(
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    group_name: c10d.GroupName,
) -> list[torch.Tensor]:
    group = c10d._resolve_process_group(group_name)
    A_shape = torch.Size((A_shard.shape[0] * group.size(), *A_shard.shape[1:]))
    symm_mem = get_symm_mem_workspace(
        group_name, A_shape.numel() * A_shard.element_size()
    )
    A = symm_mem.get_buffer(symm_mem.rank, A_shape, A_shard.dtype)
    torch.ops.symm_mem.multimem_all_gather_out(A_shard, group_name, A)
    return [torch.matmul(A, B) for B in Bs]


@torch.library.impl(lib, "fused_all_gather_scaled_matmul", "Meta")
def _fused_all_gather_scaled_matmul_fallback(
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    A_scale: torch.Tensor,
    B_scales: list[torch.Tensor],
    gather_dim: int,
    group_name: c10d.GroupName,
    biases: list[torch.Tensor | None],
    result_scales: list[torch.Tensor | None],
    out_dtypes: list[torch.dtype | None],
    use_fast_accum: list[bool],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    out_dtypes = _maybe_convert_scalar_types_to_dtypes(out_dtypes)

    group_size = c10d._get_group_size_by_name(group_name)
    A = torch.ops._c10d_functional.all_gather_into_tensor(
        A_shard.contiguous(), group_size, group_name
    )
    A = torch.ops._c10d_functional.wait_tensor(A)
    A = A.view(group_size, *A_shard.shape).movedim(gather_dim + 1, 1).flatten(0, 1)

    scale_mode = _check_and_verify_fp8_all_gather_scale_mode(
        shard=A_shard, scale=A_scale, gather_dim=gather_dim, group_size=group_size
    )
    if scale_mode == _ScaleMode.ROW_WISE_SHARDED:
        A_scale_shard = A_scale
        A_scale = torch.ops._c10d_functional.all_gather_into_tensor(
            A_scale.contiguous(), group_size, group_name
        )
        A_scale = torch.ops._c10d_functional.wait_tensor(A_scale)
        A_scale = (
            A_scale.view(group_size, *A_scale_shard.shape)
            .movedim(gather_dim + 1, 1)
            .flatten(0, -2)
        )
    elif scale_mode == _ScaleMode.ROW_WISE_REPLICATED:
        A_scale = A_scale.movedim(gather_dim, 0).flatten(0, -2)
    else:
        if scale_mode != _ScaleMode.TENSOR_WISE:
            raise AssertionError

    def scaled_matmul(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        bias: torch.Tensor | None,
        result_scale: torch.Tensor | None,
        out_dtype: torch.dtype | None,
        use_fast_accum: bool,
    ) -> torch.Tensor:
        leading_dims = A.shape[:-1]
        res = torch.ops.aten._scaled_mm(
            A.flatten(0, -2),
            B,
            A_scale,
            B_scale,
            bias,
            result_scale,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
        return res.unflatten(0, leading_dims)

    return A.movedim(0, gather_dim), [
        scaled_matmul(
            A, B, A_scale, B_scale, bias, result_scale, out_dtype, fast_accum
        ).movedim(0, gather_dim)
        for B, B_scale, bias, result_scale, out_dtype, fast_accum in zip(
            Bs, B_scales, biases, result_scales, out_dtypes, use_fast_accum
        )
    ]


@torch.library.impl(lib, "fused_all_gather_scaled_matmul", "CUDA")
@torch.library.impl(lib, "fused_all_gather_scaled_matmul", "XPU")
def _fused_all_gather_scaled_matmul(
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    A_scale: torch.Tensor,
    B_scales: list[torch.Tensor],
    gather_dim: int,
    group_name: c10d.GroupName,
    biases: list[torch.Tensor | None],
    result_scales: list[torch.Tensor | None],
    out_dtypes: list[torch.dtype | None],
    use_fast_accum: list[bool],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        A = all_gather_single(A_shard, gather_dim, group_name)
        leading_dims = A.shape[:-1]
        res = torch.ops.aten._scaled_mm(A.flatten(0, -2), B, A_scale, B_scale)
        res = res.unflatten(0, leading_dims)

    The input `A_scale` can be tensor-wise, row-wise-sharded or
    row-wise-replicated.

    Optimal stride order for `A_shard` - if `A_shard.movedim(gather_dim, 0)` is
    contiguous, no extra copy is required for input layout transformation.
    Otherwise A_shard needs to be copied once.
    """
    out_dtypes = _maybe_convert_scalar_types_to_dtypes(out_dtypes)

    if len(biases) != len(Bs):
        raise ValueError("len(biases) must be the same as len(Bs)")
    if len(result_scales) != len(Bs):
        raise ValueError("len(result_scales) must be the same as len(Bs)")
    if len(out_dtypes) != len(Bs):
        raise ValueError("len(out_dtypes) must be the same as len(Bs)")
    if len(use_fast_accum) != len(Bs):
        raise ValueError("len(use_gast_accum_list) must be the same as len(Bs)")

    if _is_test_mode:
        return _fused_all_gather_scaled_matmul_fallback(
            A_shard,
            Bs,
            A_scale,
            B_scales,
            gather_dim,
            group_name,
            biases,
            result_scales,
            out_dtypes,
            use_fast_accum,
        )

    with torch.profiler.record_function("fused_all_gather_scaled_matmul"):
        A, res = _fused_all_gather_matmul_impl(
            torch.ops.aten._scaled_mm.out,
            A_shard,
            Bs,
            A_scale,
            [
                {
                    "scale_b": B_scale,
                    "bias": bias,
                    "scale_result": result_scale,
                    "out_dtype": out_dtype,
                    "use_fast_accum": fast_accum,
                }
                for B_scale, bias, result_scale, out_dtype, fast_accum in zip(
                    B_scales, biases, result_scales, out_dtypes, use_fast_accum
                )
            ],
            out_dtypes,
            gather_dim,
            group_name,
            True,
        )
        if A is None:
            raise AssertionError
        return A, res


def make_contiguous_for_perm(
    t: torch.Tensor,
    perm: list[int],
) -> torch.Tensor:
    """
    Restride `t` such that `t.permute(perm)` is contiguous.
    """
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    return t.permute(perm).contiguous().permute(inv_perm)


def restride_A_shard_for_fused_all_gather_matmul(
    t: torch.Tensor,
    gather_dim: int,
) -> torch.Tensor:
    """
    Restride the `A_shard` arg of `fused_all_gather_matmul` for optimal perf.
    See the doc for `fused_all_gather_matmul` for detail.
    """
    perm = list(range(len(t.shape)))
    perm.insert(0, perm.pop(gather_dim))
    return make_contiguous_for_perm(t, perm)


@torch.library.impl(lib, "fused_matmul_reduce_scatter", "CUDA")
@torch.library.impl(lib, "fused_matmul_reduce_scatter", "XPU")
def _fused_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: c10d.GroupName,
) -> torch.Tensor:
    """
    Perform the following logic with micro-pipelined computation and
    communication:

        reduce_scatter_single(A @ B, reduce_op, scatter_dim, group_name)

    Optimal stride order for A - if A.movedim(scatter_dim, 0) is contiguous, no
    extra copy is required for input layout transformation. Otherwise A needs
    to be copied once.
    """
    if _is_test_mode:
        return _fused_matmul_reduce_scatter_fallback(
            A, B, reduce_op, scatter_dim, group_name
        )

    with torch.profiler.record_function("fused_matmul_reduce_scatter"):
        return _fused_matmul_reduce_scatter_impl(
            mm_out_op=torch.ops.aten.mm.out,
            A=A,
            B=B,
            kwargs={},
            out_dtype=A.dtype,
            reduce_op=reduce_op,
            scatter_dim=scatter_dim,
            group_name=group_name,
        )


@torch.library.impl(lib, "fused_matmul_reduce_scatter", "Meta")
def _fused_matmul_reduce_scatter_fallback(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: c10d.GroupName,
) -> torch.Tensor:
    res = funcol.reduce_scatter_single(A @ B, reduce_op, scatter_dim, group_name)
    res = funcol.wait_tensor(res)
    return res


def _fused_matmul_reduce_scatter_impl(
    mm_out_op: torch._ops.OpOverload,
    A: torch.Tensor,
    B: torch.Tensor,
    kwargs: dict[str, Any],
    out_dtype: torch.dtype | None,
    reduce_op: str,
    scatter_dim: int,
    group_name: c10d.GroupName,
) -> torch.Tensor:
    if A.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    if scatter_dim < 0 or scatter_dim >= A.dim():
        raise ValueError("Invalid gather_dim")
    if B.dim() != 2:
        raise ValueError("B must be a matrix")
    if reduce_op not in ("sum", "avg"):
        raise ValueError("reduce_op must be sum or avg")
    group = c10d._resolve_process_group(group_name)
    out_shape = [*A.shape[:-1], B.shape[1]]
    out_shape[scatter_dim] //= group.size()
    output_dtype = out_dtype or A.dtype

    if scatter_dim == A.ndim - 1:
        B_shards = B.chunk(group.size(), dim=B.ndim - 1)
        A_flat = A.flatten(0, -2)

        def _chunk_producer(rank: int, out: torch.Tensor) -> None:
            mm_out_op(A_flat, B_shards[rank], **kwargs, out=out)

        leading_dims = list(A.shape[:-1])

        stacked_partials = torch.empty(
            (A_flat.shape[0], B.shape[1]),
            dtype=out_dtype or A.dtype,
            device=A.device,
        )

        _pipelined_produce_and_all2all(
            _chunk_producer,
            stacked_partials,
            group_name,
            out_chunk_dim=1,
        )

        stacked_partials_view = stacked_partials.reshape(
            *leading_dims, group.size(), -1
        )
        return reduce_partials(
            stacked_partials_view,
            dim=-2,
            reduce_op=reduce_op,
            output_dtype=output_dtype,
            group_size=group.size(),
        )

    # Move the scatter_dim to the front and flatten the tensor into a 2D matrix
    x = A.movedim(scatter_dim, 0)
    leading_dims = [group.size()] + list(x.shape[:-1])
    leading_dims[1] //= group.size()
    x = x.flatten(0, -2)
    A_shards = x.chunk(group.size())

    # Computing block-wise matmul along the first dim of A
    def chunk_producer(rank: int, out: torch.Tensor) -> None:
        mm_out_op(A_shards[rank], B, **kwargs, out=out)

    stacked_partials = x.new_empty(x.shape[0], B.shape[1], dtype=output_dtype)

    _pipelined_produce_and_all2all(
        chunk_producer,
        stacked_partials,
        group_name,
    )

    # Ensures that the transpose and reduction produce contiguous result
    # in a single reduction kernel.
    return reduce_partials(
        stacked_partials.view(*leading_dims, -1)
        .movedim(1, scatter_dim + 1)
        .movedim(0, scatter_dim),
        dim=scatter_dim,
        reduce_op=reduce_op,
        output_dtype=output_dtype,
        group_size=group.size(),
    )


@torch.library.impl(lib, "fused_scaled_matmul_reduce_scatter", "CUDA")
@torch.library.impl(lib, "fused_scaled_matmul_reduce_scatter", "XPU")
def _fused_scaled_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: c10d.GroupName,
    output_shape: list[int],
    bias: torch.Tensor | None = None,
    result_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    if _is_test_mode:
        return _fused_scaled_matmul_reduce_scatter_fallback(
            A,
            B,
            A_scale,
            B_scale,
            reduce_op,
            orig_scatter_dim,
            scatter_dim_after_maybe_reshape,
            group_name,
            output_shape,
            bias,
            result_scale,
            out_dtype,
            use_fast_accum,
        )
    with torch.profiler.record_function("fused_scaled_matmul_reduce_scatter"):
        return _fused_scaled_matmul_reduce_scatter_impl(
            mm_out_op=torch.ops.aten._scaled_mm.out,
            A=A,
            B=B,
            A_scale=A_scale,
            kwargs={
                "scale_b": B_scale,
                "bias": bias,
                "scale_result": result_scale,
                "out_dtype": out_dtype,
                "use_fast_accum": use_fast_accum,
            },
            out_dtype=out_dtype,
            reduce_op=reduce_op,
            orig_scatter_dim=orig_scatter_dim,
            scatter_dim_after_maybe_reshape=scatter_dim_after_maybe_reshape,
            group_name=group_name,
            output_shape=output_shape,
        )


@torch.library.impl(lib, "fused_scaled_matmul_reduce_scatter", "Meta")
def _fused_scaled_matmul_reduce_scatter_fallback(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: c10d.GroupName,
    output_shape: list[int],
    bias: torch.Tensor | None = None,
    result_scale: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    if A_scale.numel() > 1:
        if A_scale.shape[:-1] != A.shape[:-1]:
            raise ValueError(
                "For row-wise scaling, the leading dims of A_scale "
                "must match the leading dims of A "
                f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
            )
        A_scale = A_scale.flatten(0, -2).contiguous()
    elif A_scale.numel() != 1:
        raise ValueError(
            "Invalid A_scale shape "
            f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
        )

    C = torch._scaled_mm(
        A.flatten(0, -2).contiguous(),
        B,
        A_scale,
        B_scale,
        bias,
        result_scale,
        out_dtype,
        use_fast_accum,
    )
    C = C.view(*output_shape[:-1], B.shape[1])
    res = funcol.reduce_scatter_single(
        C,
        reduce_op,
        orig_scatter_dim,  # need original scatter dim for 3D+ output tensor here
        group_name,
    )
    res = funcol.wait_tensor(res)
    return res


def _fused_scaled_matmul_reduce_scatter_impl(
    mm_out_op: torch._ops.OpOverload,
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    kwargs: dict[str, Any],
    out_dtype: torch.dtype | None,
    reduce_op: str,
    orig_scatter_dim: int,
    scatter_dim_after_maybe_reshape: int,
    group_name: c10d.GroupName,
    output_shape: list[int],
) -> torch.Tensor:
    if A.dim() < 2:
        raise ValueError("A_shard must be a matrix")
    if (
        scatter_dim_after_maybe_reshape < 0
        or scatter_dim_after_maybe_reshape >= A.dim()
    ):
        raise ValueError("Invalid scatter dim for 2D tensor input to scaled_mm")
    if orig_scatter_dim < 0 or orig_scatter_dim >= len(output_shape):
        raise ValueError("Invalid scatter dim for 3D+ output tensor")
    if B.dim() != 2:
        raise ValueError("B must be a matrix")
    if reduce_op not in ("sum", "avg"):
        raise ValueError("reduce_op must be sum or avg")
    output_dtype = out_dtype or A.dtype

    group = c10d._resolve_process_group(group_name)

    # Move scatter to first dim, then shard the tensor along the first dim, so the chunk producer
    # can perform matmuls along the first dim.
    A_with_scatter_dim_0 = A.movedim(scatter_dim_after_maybe_reshape, 0)

    # To handle case where A is 3D+, reshape to 2D to prepare for mm which requires 2D inputs.
    A_2D_with_scatter_dim_0 = A_with_scatter_dim_0.flatten(0, -2)

    # Partition A along the first dim to prepare for sharding across TP process group.
    A_shards = A_2D_with_scatter_dim_0.chunk(group.size())

    # Now that 'A' is sharded along the first dim, we need to update its scale(s) accordingly.
    # How we do this depends on if we are using tensorwise scaling, rowwise scaling, or no scaling.
    tensorwise_scaling = A_scale is not None and A_scale.numel() == 1
    rowwise_scaling = A_scale is not None and A_scale.numel() > 1

    # For tensorwise scaling, the scale should be replicated so each shard has a copy.
    if tensorwise_scaling:
        A_scale_shards = [A_scale] * group.size()

    # For rowwise scaling, we need to move the scatter dim to the first dim to match the
    # dim swap of the 'A' tensor. Then we can shard the scales along the first dim, just like
    # the 'A' tensor.
    elif rowwise_scaling:
        if A_scale.shape[:-1] != A.shape[:-1]:
            raise ValueError(
                "For row-wise scaling, the leading dims of A_scale "
                "must match the leading dims of A "
                f"(A shape: {A.shape}, A_scale shape: {A_scale.shape})"
            )
        A_scale = (
            A_scale.movedim(scatter_dim_after_maybe_reshape, 0)
            .contiguous()
            .flatten(0, -2)
        )
        A_scale_shards = list(A_scale.chunk(group.size()))
        # cuBLAS's row-wise kernel requires scales to be aligned to 16 bytes.
        # When we slice them we might break this and need to reallocate them.
        A_scale_shards = [
            t if t.data_ptr() % 16 == 0 else t.clone() for t in A_scale_shards
        ]
    else:
        raise ValueError("A_scale cannot be none for scaled_mm")

    # Computing block-wise matmul along the first dim of A
    def chunk_producer(rank: int, out: torch.Tensor) -> None:
        mm_out_op(A_shards[rank], B, scale_a=A_scale_shards[rank], **kwargs, out=out)

    # Stacked partials will be the 2D outputs of the pipelined scaled mm, and will
    # have the shape (A_with_scatter_dim_0_tensor.shape[0], B.shape[1]) to align with the formula:
    # (a*b,c) @ (c,d) = (a*b,d)
    stacked_partials = A_with_scatter_dim_0.new_empty(
        A_2D_with_scatter_dim_0.shape[0], B.shape[1], dtype=output_dtype
    )

    # Execute the pipelined mm/scaled_mm.
    _pipelined_produce_and_all2all(
        chunk_producer,
        stacked_partials,
        group_name,
    )

    # We now need to transform the *unreduced* stacked 2D partial mm outputs to an *unreduced* 3D+ output,
    # then reduce-scatter. To do this, we first need to determine the shape of the unreduced 3D+ output,
    # to reshape our stacked partials so we can apply the reduce-scatter.
    #
    # The *unreduced* 3D+ tensor will have dim 0 = `group_size`, as we have `group_size` instances of
    # stacked partial outputs. The next dims will be A's leading dims (sharded along the original scatter dim),
    # as it was the left operand of the mm op. We can use -1 as the final dim of the view to populate the rest.
    stacked_partials_3D_leading_dims = [group.size()] + list(
        # We use A from after the dim swap 0<=>scatter_dim, but before the flatten,
        # to get the leading dims of the 3D+ view of stacked partials.
        A_with_scatter_dim_0.shape[:-1]
    )

    # The `group_size` leading dim has been prepended to `stacked_partials_3D_leading_dims`,
    # to capture the partial output from each rank. We need to divide the sharding/scatter dim
    # by the group size. If the original scatter dim was 0, then it is now dim 1 in this
    # tensor, since this new `group_size` dim was prepended.
    stacked_partial_scatter_dim = orig_scatter_dim if orig_scatter_dim > 0 else 1
    stacked_partials_3D_leading_dims[stacked_partial_scatter_dim] //= group.size()

    # Ensures that the transpose and reduction produce contiguous result
    # in a single reduction kernel.
    reduced_out = reduce_partials(
        # View 2D stacked partials as 3D+ tensor of shape (`group_size`, ...)
        stacked_partials.view(*stacked_partials_3D_leading_dims, -1)
        # We originally swapped 0<=>scatter_dim_after_maybe_reshape. Now after
        # prepending the `group_size` dim, to undo this original swap, we
        # must swap 1<=>scatter_dim_after_maybe_reshape+1.
        .movedim(1, scatter_dim_after_maybe_reshape + 1),
        # Reduce along the `group_size` dim (0).
        dim=0,
        reduce_op=reduce_op,
        output_dtype=output_dtype,
        group_size=group.size(),
    )

    # Output shape must be scattered along original scatter dim as well.
    output_shape[orig_scatter_dim] //= group.size()
    out = reduced_out.view(*output_shape)
    return out


def restride_A_for_fused_matmul_reduce_scatter(
    t: torch.Tensor,
    scatter_dim: int,
) -> torch.Tensor:
    """
    Restride the `A_shard` arg of `fused_matmul_reduce_scatter` for optimal
    perf. See the doc for `fused_matmul_reduce_scatter` for detail.
    """
    perm = list(range(len(t.shape)))
    perm.insert(0, perm.pop(scatter_dim))
    return make_contiguous_for_perm(t, perm)


def _maybe_convert_scalar_types_to_dtypes(
    scalar_types: list[Any],
) -> list[torch.dtype | None]:
    """
    When a list of `torch.dtype`s is passed through the dispatcher as
    `ScalarType[]`, it is converted to a list of scalar type enum values. This
    function converts it back to a list of `torch.dtype`s.
    """
    # Order defined in https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
    _SCALAR_TYPE_TO_DTYPE = {
        0: torch.uint8,
        1: torch.int8,
        2: torch.short,
        3: torch.int,
        4: torch.int64,
        5: torch.half,
        6: torch.float,
        7: torch.double,
        8: torch.complex32,
        9: torch.complex64,
        10: torch.complex128,
        11: torch.bool,
        12: torch.qint8,
        13: torch.quint8,
        14: torch.qint32,
        15: torch.bfloat16,
        16: torch.float8_e5m2,
        17: torch.float8_e4m3fn,
        18: torch.float8_e5m2fnuz,
        19: torch.float8_e4m3fnuz,
    }
    if any(not isinstance(x, (type(None), int)) for x in scalar_types):
        return scalar_types

    dtypes: list[torch.dtype | None] = []
    for scalar_type in scalar_types:
        if scalar_type is None:
            dtypes.append(scalar_type)
        elif scalar_type not in _SCALAR_TYPE_TO_DTYPE:
            raise ValueError(f"Unrecognized scalar type {scalar_type}")
        else:
            dtypes.append(_SCALAR_TYPE_TO_DTYPE[scalar_type])
    return dtypes


class Work(_Work):
    def __init__(self) -> None:
        super().__init__()
        self.event = torch.Event()
        self.event.record()

    def wait(self, timeout: timedelta = timedelta(seconds=0)) -> bool:
        self.event.wait()
        return True


"""
NOTE [low-contention collectives]
When a collective is overlapped with abundant compute, it makes sense to
prioritize reducing the contention between the collective and the overlapped
compute, even at the cost of a slightly slower collective.

Common collective implementations (e.g., NCCL without user buffer
registration) optimize for throughput with no ambient compute. However, such
implementations may not be optimal when they are overlapped with compute:
- These implementations typically fuse the entire collective into a single
kernel and reserve SM resources based on the most demanding portion of the
collective, even when a large portion of the collective does not require this
much resource.
- These implementations often use SM-based P2P copy as opposed to copy
engine-based P2P copy. Copy engine-based P2P copy may not have a significant
advantage when there's no ambient compute. However, it may significantly
improve overall resource utilization in the presence of ambient compute.

When overlapped with intensive compute (e.g., persistent matmul kernels), the
SM-usage of a collective can lead to inefficient overlapping.

Low-contention collectives achieve their goals with the following strategies:
- Use copy engine-based copy whenever possible.
- Break down portions of a collective with different resource requirements
into multiple kernels. This improves the overlapping efficiency at the cost
of additional launching overhead.
"""


@torch.library.impl(lib, "_low_contention_all_gather", "Meta")
def _low_contention_all_gather_meta(
    tensor: torch.Tensor,
    group_name: c10d.GroupName,
) -> torch.Tensor:
    group_size = c10d._get_group_size_by_name(group_name)
    return tensor.new_empty(tensor.shape[0] * group_size, *tensor.shape[1:])


@torch.library.impl(lib, "_low_contention_all_gather", "CUDA")
def _low_contention_all_gather(
    tensor: torch.Tensor,
    group_name: c10d.GroupName,
) -> torch.Tensor:
    """
    Performs all-gather with symmetric memory in a low-contention fashion.

    When `tensor` is already in symmetric memory:
        - The collective is carried out without using SMs.
        - No symmetric memory workspace is required.

    When `tensor` is not in symmetric memory:
        - An extra SM-based copy is performed to copy the input data into the
          symmetric memory workspace.
        - Symmetric memory workspace size requirement: the size of `tensor`.
    """
    symm_mem = rendezvous(tensor, group_name)
    if symm_mem is not None:
        input_is_symm_mem = True
    else:
        symm_mem = get_symm_mem_workspace(
            group_name, tensor.numel() * tensor.element_size()
        )
        input_is_symm_mem = False

    rank = symm_mem.rank
    world_size = symm_mem.world_size

    output = tensor.new_empty(tensor.shape[0] * world_size, *tensor.shape[1:])
    chunks = output.chunk(world_size)

    _get_backend_stream().wait_stream(torch.accelerator.current_stream())
    with _get_backend_stream():
        if not input_is_symm_mem:
            local_buf = symm_mem.get_buffer(rank, tensor.shape, tensor.dtype)
            local_buf.copy_(tensor)
        # pull
        symm_mem.barrier()
        for step in range(world_size):
            remote_rank = (rank - step) % world_size
            src_buf = symm_mem.get_buffer(remote_rank, tensor.shape, tensor.dtype)
            chunks[remote_rank].copy_(src_buf)
        symm_mem.barrier()
        torch._C._distributed_c10d._register_work(output, Work())
        return output


def _require_multicast(device_index: int, op_name: str) -> None:
    if not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device_index):
        raise RuntimeError(f"{op_name} requires multicast support (NVSwitch / NVLS).")


# Use one dedicated signal-pad channel for CE multicast, separate from existing
# low-contention collectives. Its two barriers are ordered on the same stream;
# if a peer has not consumed the first signal yet, the second barrier's CAS put
# spins until that peer slot is cleared.
_CE_MULTICAST_BARRIER_CHANNEL = 3


def _lc_ag_out_shape(tensor: torch.Tensor, world_size: int) -> tuple[int, ...]:
    if tensor.dim() == 0:
        return (world_size,)
    return (tensor.shape[0] * world_size, *tensor.shape[1:])


def _check_lc_ag_out(
    tensor: torch.Tensor,
    output: torch.Tensor,
    world_size: int,
    op_name: str,
) -> None:
    expected_shape = _lc_ag_out_shape(tensor, world_size)
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"{op_name}: expected out shape {expected_shape}, "
            f"but got {tuple(output.shape)}."
        )
    if output.dtype != tensor.dtype:
        raise RuntimeError(
            f"{op_name}: expected out dtype {tensor.dtype}, but got {output.dtype}."
        )
    if output.device != tensor.device:
        raise RuntimeError(
            f"{op_name}: expected out device {tensor.device}, but got {output.device}."
        )
    if not output.is_contiguous():
        raise RuntimeError(f"{op_name}: out must be contiguous.")


def _check_lc_signal_pad_capacity(symm_mem: _SymmetricMemory) -> None:
    required_bytes = (_CE_MULTICAST_BARRIER_CHANNEL + 1) * symm_mem.world_size * 4
    actual_bytes = _SymmetricMemory.signal_pad_size
    if actual_bytes < required_bytes:
        raise RuntimeError(
            f"low_contention all-gather requires signal_pad_size >= "
            f"{required_bytes} bytes for world_size={symm_mem.world_size}, but "
            f"the current size is {actual_bytes} bytes."
        )


@torch.library.impl(lib, "_low_contention_all_gather_ce_multicast", "Meta")
def _low_contention_all_gather_ce_multicast_meta(
    tensor: torch.Tensor,
    group_name: c10d.GroupName,
) -> torch.Tensor:
    world_size = c10d._get_group_size_by_name(group_name)
    return tensor.new_empty(_lc_ag_out_shape(tensor, world_size))


@torch.library.impl(lib, "_low_contention_all_gather_ce_multicast", "CUDA")
def _low_contention_all_gather_ce_multicast(
    tensor: torch.Tensor,
    group_name: c10d.GroupName,
) -> torch.Tensor:
    """Direct-input copy-engine multicast all-gather into a fresh output."""
    device_index = (
        tensor.device.index
        if tensor.device.index is not None
        else torch.cuda.current_device()
    )
    _require_multicast(device_index, "_low_contention_all_gather_ce_multicast")
    world_size = c10d._get_group_size_by_name(group_name)
    out_shape = _lc_ag_out_shape(tensor, world_size)
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "_low_contention_all_gather_ce_multicast cannot allocate its output "
            "during CUDA graph capture. Use "
            "_low_contention_all_gather_ce_multicast_out with a preallocated "
            "symmetric-memory output."
        )
    device = torch.device("cuda", device_index)
    with torch.cuda.use_mem_pool(get_mem_pool(device)):
        output = torch.empty_strided(
            out_shape,
            make_contiguous_strides_for(out_shape),
            dtype=tensor.dtype,
            device=device,
        )
    return _low_contention_all_gather_ce_multicast_impl(tensor, group_name, output)


@torch.library.impl(lib, "_low_contention_all_gather_ce_multicast_out", "Meta")
def _low_contention_all_gather_ce_multicast_out_meta(
    tensor: torch.Tensor,
    group_name: c10d.GroupName,
    out: torch.Tensor,
) -> torch.Tensor:
    world_size = c10d._get_group_size_by_name(group_name)
    _check_lc_ag_out(tensor, out, world_size, "_low_contention_all_gather_ce_multicast")
    return out


@torch.library.impl(lib, "_low_contention_all_gather_ce_multicast_out", "CUDA")
def _low_contention_all_gather_ce_multicast_out(
    tensor: torch.Tensor,
    group_name: c10d.GroupName,
    out: torch.Tensor,
) -> torch.Tensor:
    """Direct-input copy-engine multicast all-gather into caller-provided output."""
    world_size = c10d._get_group_size_by_name(group_name)
    _check_lc_ag_out(tensor, out, world_size, "_low_contention_all_gather_ce_multicast")
    return _low_contention_all_gather_ce_multicast_impl(tensor, group_name, out)


def _low_contention_all_gather_ce_multicast_impl(
    tensor: torch.Tensor,
    group_name: c10d.GroupName,
    output: torch.Tensor,
) -> torch.Tensor:
    device_index = (
        output.device.index
        if output.device.index is not None
        else torch.cuda.current_device()
    )
    _require_multicast(device_index, "_low_contention_all_gather_ce_multicast")
    symm_mem = rendezvous(output, group_name)
    if symm_mem is None:
        raise RuntimeError(
            "ce_multicast output must be allocated from symmetric memory"
        )
    _check_lc_signal_pad_capacity(symm_mem)

    rank = symm_mem.rank
    shard_bytes = tensor.numel() * tensor.element_size()

    ag_input = tensor if tensor.is_contiguous() else tensor.contiguous()
    symm_mem.barrier(channel=_CE_MULTICAST_BARRIER_CHANNEL)
    torch.ops.symm_mem.memcpy_to_multicast_(
        output, ag_input, rank * shard_bytes, group_name
    )
    symm_mem.barrier(channel=_CE_MULTICAST_BARRIER_CHANNEL)
    return output


@torch.library.impl(lib, "_low_contention_reduce_scatter", "Meta")
def _low_contention_reduce_scatter_meta(
    tensor: torch.Tensor,
    reduce_op: str,
    group_name: c10d.GroupName,
) -> torch.Tensor:
    group_size = c10d._get_group_size_by_name(group_name)
    return tensor.unflatten(0, (group_size, -1)).mean(dim=0)


def _low_contention_reduce_scatter_with_symm_mem_input(
    tensor: torch.Tensor,
    reduce_op: str,
    symm_mem: _SymmetricMemory,
) -> torch.Tensor:
    rank = symm_mem.rank
    world_size = symm_mem.world_size

    if tensor.shape[0] % world_size != 0:
        raise AssertionError
    a2a_res = torch.empty_like(tensor)
    chunks = a2a_res.chunk(world_size)

    _get_backend_stream().wait_stream(torch.accelerator.current_stream())
    with _get_backend_stream():
        # pull + offline reduction
        symm_mem.barrier()
        for step in range(world_size):
            remote_rank = (rank - step) % world_size
            src_buf = symm_mem.get_buffer(
                remote_rank,
                chunks[0].shape,
                chunks[0].dtype,
                chunks[0].numel() * rank,
            )
            chunks[remote_rank].copy_(src_buf)
        symm_mem.barrier()

        ret = a2a_res.unflatten(0, (world_size, -1))
        if reduce_op == "sum":
            ret = ret.sum(dim=0)
        elif reduce_op == "avg":
            ret = ret.mean(dim=0)
        else:
            raise ValueError(f"reduce_op ({reduce_op}) is not supported")
        torch._C._distributed_c10d._register_work(ret, Work())
        return ret


def _low_contention_reduce_scatter_with_workspace(
    tensor: torch.Tensor,
    reduce_op: str,
    workspace: _SymmetricMemory,
) -> torch.Tensor:
    rank = workspace.rank
    world_size = workspace.world_size

    if tensor.shape[0] % world_size != 0:
        raise AssertionError
    chunks = tensor.chunk(world_size)

    _get_backend_stream().wait_stream(torch.accelerator.current_stream())
    with _get_backend_stream():
        # push + offline reduction
        workspace.barrier()
        for step in range(world_size):
            remote_rank = (rank - step) % world_size
            dst_buf = workspace.get_buffer(
                remote_rank, chunks[0].shape, chunks[0].dtype, chunks[0].numel() * rank
            )
            dst_buf.copy_(chunks[remote_rank])
        workspace.barrier()

        buf = workspace.get_buffer(rank, tensor.shape, tensor.dtype)
        ret = buf.unflatten(0, (world_size, -1))
        if reduce_op == "sum":
            ret = ret.sum(dim=0)
        elif reduce_op == "avg":
            ret = ret.mean(dim=0)
        else:
            raise ValueError(f"reduce_op ({reduce_op}) is not supported")
        torch._C._distributed_c10d._register_work(ret, Work())
        return ret


@torch.library.impl(lib, "_low_contention_reduce_scatter", "CUDA")
def _low_contention_reduce_scatter(
    tensor: torch.Tensor,
    reduce_op: str,
    group_name: c10d.GroupName,
) -> torch.Tensor:
    """
    Performs reduce-scatter with symmetric memory in a low-contention fashion.

    This implementation performs a P2P-based all-to-all followed by an offline
    reduction.

    When `tensor` is already in symmetric memory:
        - Pull-based all-to-all is used.
        - No symmetric memory workspace is required.

    When `tensor` is not in symmetric memory:
        - Push-based all-to-all is used.
        - Symmetric memory workspace size requirement: the size of `tensor`.

    SM-usage:
        - SM-based copy of the rank's own chunk for the all-to-all.
        - Reduction on the all-to-all result.

    TODO(yifu): the SM-based copy can be avoided with a list-based reduction
    kernel.
    """
    symm_mem = rendezvous(tensor, group_name)
    if symm_mem is not None:
        return _low_contention_reduce_scatter_with_symm_mem_input(
            tensor, reduce_op, symm_mem
        )
    else:
        workspace = get_symm_mem_workspace(
            group_name, tensor.numel() * tensor.element_size()
        )
        return _low_contention_reduce_scatter_with_workspace(
            tensor, reduce_op, workspace
        )


@torch.library.impl(lib, "all_to_all_vdev_2d", "Meta")
def _all_to_all_vdev_2d_meta(
    input: torch.Tensor,
    out: torch.Tensor,
    in_splits: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group_name: c10d.GroupName,
    major_align: int | None = None,
) -> None:
    return None


@torch.library.impl(lib, "all_to_all_vdev_2d_offset", "Meta")
def _all_to_all_vdev_2d_offset_meta(
    input: torch.Tensor,
    out: torch.Tensor,
    in_splits_offsets: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    group_name: c10d.GroupName,
) -> None:
    return None


# =============================================================================
# User-facing APIs
# =============================================================================


from collections.abc import Sequence
from typing import overload, TYPE_CHECKING, Union


if TYPE_CHECKING:
    from torch._C._distributed_c10d import ProcessGroup
    from torch.types import _device, _dtype, _int


_use_implicit_mempool: bool | None = None  # type: ignore[assignment]


def _should_use_implicit_mempool() -> bool:
    r"""
    Check if the implicit memory pool should be used for symmetric memory allocations.

    Returns:
        bool: True if the implicit memory pool should be used, False otherwise.

    By default, use implicit memory pool for `symm_mem.empty`.  Users can
    disable this by setting the environment variable `TORCH_SYMMMEM_IMPLICIT_POOL` to `0`.
    """
    global _use_implicit_mempool
    if _use_implicit_mempool is None:
        _use_implicit_mempool = os.getenv("TORCH_SYMMMEM_IMPLICIT_POOL", "1") == "1"

    return _use_implicit_mempool


@overload
def empty(
    *size: _int, dtype: _dtype | None = None, device: _device | None = None
) -> torch.Tensor: ...


@overload
# pyrefly: ignore [inconsistent-overload]
def empty(
    size: Sequence[_int],
    *,
    dtype: _dtype | None = None,
    device: _device | None = None,
) -> torch.Tensor: ...


def empty(  # type: ignore[misc]
    *size: Any,
    dtype: _dtype | None = None,
    device: _device | None = None,
) -> torch.Tensor:
    r"""
    Similar to :func:`torch.empty()`. The returned tensor can be used by
    :func:`torch._distributed._symmetric_memory.rendezvous()` to establish a
    symmetric memory tensor among participating processes.

    .. note::
        This is a host-synchronous allocation. Together with
        :func:`rendezvous`, it is intended as an initialization-time
        operation: allocate a symmetric memory tensor once and reuse it,
        rather than allocating in hot code paths.

    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
            for CPU tensor types, the current CUDA device for CUDA tensor types,
            and the current XPU device for XPU tensor types.
    """
    if len(size) == 1 and isinstance(size[0], Sequence):
        size = tuple(size[0])
    else:
        size = tuple(size)

    if dtype is None:
        dtype = torch.get_default_dtype()

    if device is None:
        device = torch.get_default_device()
    else:
        device = torch.device(device)

    stride = torch._prims_common.make_contiguous_strides_for(size)

    if _should_use_implicit_mempool() and device.type == "cuda":
        # Allocate tensor from an implicit memory pool
        mempool = get_mem_pool(device)
        # TODO: this path can be made device-agnostic if `use_mem_pool` is
        # elevated from torch.cuda to torch accelerator.
        with torch.cuda.use_mem_pool(mempool):
            return _SymmetricMemory.empty_strided_p2p(size, stride, dtype, device)
    else:
        return _SymmetricMemory.empty_strided_p2p(size, stride, dtype, device)


def _resolve_group_name(group: c10d.GroupName | ProcessGroup) -> c10d.GroupName:
    from torch._C._distributed_c10d import ProcessGroup

    if isinstance(group, str):
        return c10d.GroupName(group)
    if isinstance(group, ProcessGroup):
        return group.group_name
    raise TypeError(f"unsupported group type: {type(group)}")


def rendezvous(
    tensor: torch.Tensor, group: c10d.GroupName | ProcessGroup
) -> _SymmetricMemory:
    r"""
    rendezvous(tensor, group) -> _SymmetricMemory

    Establish a symmetric memory tensor among participating processes. This is
    a collective operation.

    .. note::
        This is a host-blocking initialization operation: the first rendezvous
        of a tensor performs handle exchange and mapping across processes, and
        synchronizes the host with the device. It cannot be ordered onto a
        CUDA stream or captured in a CUDA graph. Rendezvous a buffer once and
        reuse the returned handle rather than calling this in hot code paths;
        subsequent calls on the same tensor return the cached handle.

    Args:
        tensor (:class:`torch.Tensor`): the local tensor used to establish the symmetric memory tensor.
            It must be allocated via :func:`torch._distributed._symmetric_memory.empty()`. The shape,
            dtype, and device type must be identical across all participating processes.
        group (Union[str, :class:`torch.distributed.ProcessGroup`]): The group identifying the
            participating processes. This can be either a group name or a process group object.
    """
    group_name = _resolve_group_name(group)
    return _SymmetricMemory.rendezvous(tensor, group_name)


def is_nvshmem_available() -> bool:
    r"""
    is_nvshmem_available() -> bool

    Check if NVSHMEM (CUDA) or rocSHMEM (ROCm) is available in the current
    build and usable at runtime. On ROCm, rocSHMEM ``VERSION`` must be at
    least 3.3.0 (see ``rocshmem/rocshmem.hpp``).
    """
    try:
        from torch._C._distributed_c10d import _is_nvshmem_available
    except ImportError:
        # Not all builds have NVSHMEM support.
        return False

    # Check if NVSHMEM is available on current system.
    return _is_nvshmem_available()


def set_backend(name: Literal["NVSHMEM", "CUDA", "NCCL"]) -> None:
    r"""
    Set the backend for symmetric memory allocation. This is a global setting
    and affects all subsequent calls to
    :func:`torch._distributed._symmetric_memory.empty()`.  Note that the backend
    cannot be changed once a symmetric memory tensor has been allocated.

    Args:
        backend (str): the backend for symmetric memory allocation. Currently,
            only `"NVSHMEM"`, `"CUDA"`, `"NCCL"` are supported.
    """
    _SymmetricMemory.set_backend(name)


def get_backend(device: _device) -> str | None:
    r"""
    Get the backend for symmetric memory allocation for a given device. If not
    found, return None.

    Args:
        device (`torch.device` or str): the device for which to get the backend.
    """
    return _SymmetricMemory.get_backend(torch.device(device))


def get_mempool_allocator(device: _device):  # type: ignore[no-untyped-def]
    r"""
    Get the MemPool allocator for symmetric memory for a given device.

    Args:
        device (`torch.device` or str): the device for which to get the MemPool
            allocator.
    """
    return _SymmetricMemory.get_mempool_allocator(torch.device(device))


def set_signal_pad_size(size: int) -> None:
    r"""
    Set the signal pad size for future symmetric memory allocations.

    Signal pads are P2P-accessible memory regions used for synchronization in
    symmetric memory. This function allows users to configure
    the signal pad size to be proportional to their workload requirements.

    .. warning::
        This must be called before any symmetric memory allocations are made.
        The size cannot be changed after allocations have been performed.

    Args:
        size (int): the signal pad size in bytes. The size should be
            proportional to the number of blocks launched and the world size.

    Example::

        >>> # doctest: +SKIP
        >>> # Set a larger signal pad size before any allocations
        >>> torch.distributed._symmetric_memory.set_signal_pad_size(1024 * 1024)  # 1MB
    """
    _SymmetricMemory.signal_pad_size = size


def get_signal_pad_size() -> int:
    r"""
    Get the current signal pad size for symmetric memory allocations.

    Returns the user-configured size if set via :func:`set_signal_pad_size`,
    otherwise returns the default size.

    Returns:
        int: the signal pad size in bytes.

    Example::

        >>> # doctest: +SKIP
        >>> size = torch.distributed._symmetric_memory.get_signal_pad_size()
        >>> print(f"Signal pad size: {size} bytes")
    """
    return _SymmetricMemory.signal_pad_size


# An internal map from device to the symmetric memory pool for that device.
_symm_mem_pools: dict[_device, torch.cuda.MemPool] = {}


def get_mem_pool(device: _device) -> torch.cuda.MemPool:
    """
    Get the symmetric memory pool for a given device. If not found, create a new
    pool.

    The tensor allocations with this pool must be symmetric across ranks.  The
    allocated tensors can be used with symmetric operations, for example,
    operations defined under `torch.ops.symm_mem`.

    Args:
        device (`torch.device` or str): the device for which to get the symmetric memory pool.

    Returns:
        `torch.cuda.MemPool`: the symmetric memory pool for the given device.

    Example::

        >>> # doctest: +SKIP
        >>> pool = torch.distributed._symmetric_memory.get_mem_pool("cuda:0")
        >>> with torch.cuda.use_mem_pool(pool):
        >>>     tensor = torch.randn(1000, device="cuda:0")
        >>> tensor = torch.ops.symm_mem.one_shot_all_reduce(tensor, "sum", group_name)

    """
    # This function is a wrapper around the `torch.cuda.MemPool` constructor.
    # Due to special requirements of SymmetricMemory, we preset certain options for the pool.
    # - use_on_oom=False: we don't want to lend the space of the pool for
    # non-symmetric allocations because this could desync the allocation state
    # across ranks.
    # - no_split=True: we don't want to split segments, because today a segment
    # is associated with a signal pad, if two allocated tensors share a segment
    # and their kernels concurrently use (the same) signal pad, this could cause
    # undefined behaviors. We could consider relaxing this in the future if we
    # establish stream tracking and implicit synchronization around an
    # allocation.
    if device not in _symm_mem_pools:
        allocator = get_mempool_allocator(device)
        # Create a new pool with the given allocator and the preset options.
        _symm_mem_pools[device] = torch.cuda.MemPool(
            allocator,
            use_on_oom=False,
            no_split=True,
        )

    return _symm_mem_pools[device]


def _cuda_get_out(
    dst: torch.Tensor, hdl: _SymmetricMemory, offset: int, size: int, peer: int
) -> None:
    storage_offset = hdl.offset // dst.element_size() + offset
    remote_src = hdl.get_buffer(peer, (size,), dst.dtype, storage_offset)
    dst.view(-1)[:size].copy_(remote_src)


# One-sided communication APIs.
def get(
    dst: torch.Tensor,
    hdl: _SymmetricMemory,
    peer: int,
    offset: int = 0,
) -> None:
    r"""
    get(dst, hdl, peer, offset=0) -> ()

    Copy ``dst.numel()`` elements starting at ``offset`` from ``peer``'s
    symmetric allocation into local ``dst`` using one-sided symmetric memory
    access.

    ``hdl`` is the symmetric memory handle returned by
    :func:`torch.distributed._symmetric_memory.rendezvous`; the remote source
    is ``peer``'s allocation backing that handle. The number of elements copied
    is inferred from ``dst``; pass a view (e.g. ``dst[:n]``) to fill only part
    of a tensor. ``offset`` is expressed in elements of ``dst``'s dtype.
    ``dst`` can be a regular CUDA tensor or a symmetric-memory tensor; it must
    be on the same device as ``hdl`` and backed by contiguous memory. The copy
    is issued on the current CUDA stream.

    Args:
        dst (Tensor): local destination tensor.
        hdl (SymmetricMemory): handle whose peer allocation is the remote
            source.
        peer (int): rank to copy from.
        offset (int, optional): element offset into the peer allocation to
            start reading from. Defaults to ``0``.
    """
    if dst.device != hdl.device:
        raise ValueError("get: dst must be on the same device as hdl")
    if not dst.is_contiguous():
        raise ValueError("get: dst must be backed by contiguous memory")
    if offset < 0:
        raise ValueError("get: offset must be non-negative")
    if peer < 0 or peer >= hdl.world_size:
        raise ValueError("get: invalid peer")
    size = dst.numel()
    element_size = dst.element_size()
    if hdl.offset % element_size != 0:
        raise RuntimeError("get: handle offset is not element-aligned")
    start = hdl.offset + offset * element_size
    end = start + size * element_size
    if start > hdl.buffer_size or end > hdl.buffer_size:
        raise ValueError("get: requested range exceeds symmetric allocation")

    backend = get_backend(dst.device)
    if backend == "CUDA":
        _cuda_get_out(dst, hdl, offset, size, peer)
        return

    # `hdl` is a pybind `_SymmetricMemory` object. Dispatcher expects the
    # TorchBind custom class type `__torch__.torch.classes.c10d.SymmetricMemory`.
    # Convert via `.boxed()`.
    hdl_boxed = hdl.boxed() if hasattr(hdl, "boxed") else hdl
    if backend == "NVSHMEM":
        torch.ops.symm_mem.nvshmem_get_out(dst, hdl_boxed, offset, size, peer)
    elif backend == "NCCL":
        torch.ops.symm_mem.nccl_get_out(dst, hdl_boxed, offset, size, peer)
    else:
        raise ValueError(f"get: unsupported backend: {backend}")


def put_signal(src: torch.Tensor, hdl: _SymmetricMemory, peer: int) -> None:
    r"""
    put_signal(src, hdl, peer) -> None

    Put data to a peer's symmetric memory and signal the peer.

    Args:
        src (torch.Tensor): the source tensor to read data from.
        hdl (SymmetricMemory): the symmetric memory to put data to.
        peer (int): the peer to put data to.
    """
    backend = get_backend(src.device)
    # `hdl` is a pybind `_SymmetricMemory` object. Dispatcher expects the
    # TorchBind custom class type `__torch__.torch.classes.c10d.SymmetricMemory`.
    # Convert via `.boxed()`.
    hdl_boxed = hdl.boxed() if hasattr(hdl, "boxed") else hdl
    if backend == "NCCL":
        torch.ops.symm_mem.nccl_put_signal(src, hdl_boxed, peer)
    # TODO: other backends' dispatch goes here
    else:
        raise ValueError(f"put_signal: unsupported backend: {backend}")


def wait_signal(hdl: _SymmetricMemory, peer: int) -> None:
    r"""
    wait_signal(hdl, peer) -> None

    Wait for a signal from a peer.

    Args:
        hdl (SymmetricMemory): the symmetric memory handle on which to wait for a signal.
        peer (int): the peer to wait for a signal from.
    """
    backend = get_backend(hdl.device)
    # See note in `put_signal` about `_SymmetricMemory` vs TorchBind type.
    hdl_boxed = hdl.boxed() if hasattr(hdl, "boxed") else hdl
    if backend == "NCCL":
        torch.ops.symm_mem.nccl_wait_signal(hdl_boxed, peer)
    # TODO: other backends' dispatch goes here
    else:
        raise ValueError(f"wait_signal: unsupported backend: {backend}")


def reduce_scatter_offset(
    input: torch.Tensor,
    out: list[torch.Tensor],
    group: str,
    *,
    dim: int,
    offsets: list[int] | None = None,
    dst_ranks: list[int] | None = None,
    red_op: str = "sum",
) -> None:
    r"""
    reduce_scatter_offset(input, out, group, *, dim, offsets, dst_ranks, red_op='sum') -> None

    Simultaneously reduce N blocks of a 2-D ``input`` tensor from a symmetric
    memory buffer, routing each block to a specific destination rank.  Only
    ``dst_ranks[i]`` writes the reduced result for block ``i``; the result is
    written to a contiguous output tensor, with the same shape as block ``i``.

    The ``dim`` argument controls which dimension is sharded:

    - ``dim=0`` (row sharding): block ``i`` spans
      ``input[offsets[i-1] : offsets[i], :]``.  Each ``out[j]`` has shape
      ``(size_j, input.size(1))``.
    - ``dim=1`` (column sharding): block ``i`` spans
      ``input[:, offsets[i-1] : offsets[i]]``.  Each ``out[j]`` has shape
      ``(input.size(0), size_j)``.

    Blocks are described by ``offsets``, an inclusive prefix-sum of block sizes
    along ``dim`` (first block starts at index 0 by convention).  Block offsets
    can be even or uneven; when uneven, the following condition must be met: for
    each ``j``, the ``j``-th owned block must have the same size across all
    ranks (so that ``out[j]`` has a uniform shape); different ``j``'s may
    differ.

    Args:
        input (Tensor): 2-D tensor allocated via symmetric memory (innermost
            dimension must be contiguous).
        out (list[Tensor]): Output tensors for this rank's owned blocks.  Must
            have length equal to the number of blocks owned by this rank (i.e.
            the count of ``i`` where ``dst_ranks[i] == my_rank``).  Each
            ``out[j]`` must be contiguous with the same dtype as ``input``.
        group (str): The name of the ``ProcessGroup`` to perform the operation on.
        dim (int): Dimension along which blocks are defined (0 or 1).
        offsets (list[int] | None): Inclusive prefix-sum of block sizes along
            ``dim``, length N.  If not provided, ``input.size(dim)`` is divided
            into equal-size blocks based on the size of the ``group``.
        dst_ranks (list[int] | None): Destination rank for each block.  If not
            provided, blocks are distributed round-robin across ranks.
        red_op (str): Reduction operation; currently only ``'sum'`` is supported.

    Example::

        >>> # doctest: +SKIP
        >>> # Each rank holds a Grouped GEMM gradient buffer in symmetric memory.
        >>> # The buffer has W experts laid out as equal column blocks; each expert
        >>> # is reduced to a specific rank (dst_ranks[i] == i % world_size).
        >>> buf = symm_mem.empty(H, W * C, dtype=torch.bfloat16, device="cuda")
        >>> symm_mem.rendezvous(buf, group=group_name)
        >>> offsets = [i * C for i in range(1, W + 1)]  # inclusive prefix-sum
        >>> dst_ranks = [i % world_size for i in range(W)]
        >>> n_owned = sum(r == rank for r in dst_ranks)
        >>> out = [torch.empty(H, C, dtype=torch.bfloat16, device="cuda") for _ in range(n_owned)]
        >>> symm_mem.reduce_scatter_offset(buf, out, group_name, dim=1, offsets=offsets, dst_ranks=dst_ranks)
    """
    backend = get_backend(input.device)
    if backend == "NCCL":
        torch.ops.symm_mem.nccl_reduce_scatter_offset(
            input, out, group, dim, offsets, dst_ranks, red_op
        )
    else:
        raise NotImplementedError(
            f"reduce_scatter_offset: unsupported backend: {backend}"
        )


def is_symm_mem_tensor(tensor: torch.Tensor) -> bool:
    r"""
    is_symm_mem_tensor(tensor) -> bool

    Returns ``True`` if ``tensor`` was allocated via symmetric memory
    (i.e. via :func:`torch.distributed._symmetric_memory.empty` or
    :meth:`_SymmetricMemory.empty_strided_p2p`).

    This is a non-collective, O(1) check.

    Args:
        tensor (:class:`torch.Tensor`): the tensor to check.
    """
    return _SymmetricMemory.is_symm_mem_tensor(tensor)


def all_to_all_nd(
    input: torch.Tensor,
    out: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    *,
    group: str,
) -> None:
    r"""
    all_to_all_nd(input, out, scatter_dim, gather_dim, *, group) -> None

    Permute-free all-to-all: shards along ``scatter_dim`` of ``input`` are exchanged
    so each rank receives one shard per peer; results are laid out along ``gather_dim``
    of ``out``. Supported pairs are ``(scatter_dim=1, gather_dim=0)`` and
    ``(scatter_dim=0, gather_dim=1)``.

    For ``scatter_dim=1``, ``gather_dim=0`` (let ``G`` be the group size): ``input`` is
    ``[rows, G * local_cols]`` or equivalently ``[rows, G, local_cols]``;
    each rank ``r`` reads column block ``r`` from every peer and writes peer ``j`` into
    ``out[j]``. ``out`` must be contiguous with shape ``[G, rows, local_cols]`` or the
    flattened equivalent ``[G * rows, local_cols]``.

    For ``scatter_dim=0``, ``gather_dim=1`` (``G`` is the group size): ``input`` is
    ``[G * local_rows, cols]`` or equivalently ``[G, local_rows, cols]``;
    each rank ``r`` reads row block ``r`` from every peer; peer ``j`` is stored in
    ``out[:, j, :]``. ``out`` must be contiguous with shape ``[local_rows, G, cols]`` or
    ``[local_rows, G * cols]`` (flattened gather and inner dims).

    Args:
        input (Tensor): For ``(scatter_dim=1, gather_dim=0)``, 2-D ``[rows, G*local_cols]`` or
            3-D ``[rows, G, local_cols]`` in symmetric memory (innermost dimension contiguous).
            For ``(scatter_dim=0, gather_dim=1)``, 2-D ``[G*local_rows, cols]`` or 3-D
            ``[G, local_rows, cols]`` (same layout).
        out (Tensor): Contiguous buffer (see shapes above; 2-D allowed where noted).
        scatter_dim (int): ``0`` or ``1`` — dimension along which the input is partitioned.
        gather_dim (int): ``0`` or ``1`` — dimension along which peer chunks are
            concatenated in ``out``.
        group (str): The name of the ``ProcessGroup`` to perform the operation on.
    """
    backend = get_backend(input.device)
    if backend == "NCCL":
        torch.ops.symm_mem.nccl_all_to_all_nd(
            input, out, scatter_dim, gather_dim, group
        )
    else:
        raise NotImplementedError(f"all_to_all_nd: unsupported backend: {backend}")


__all__ = [
    "empty",
    "is_symm_mem_tensor",
    "rendezvous",
    "is_nvshmem_available",
    "set_backend",
    "get_backend",
    "get",
    "set_signal_pad_size",
    "get_signal_pad_size",
    "get_mem_pool",
    "reduce_scatter_offset",
    "all_to_all_nd",
]
<|endoftext|>from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import Self

import torch


if TYPE_CHECKING:
    from torch.types import _device


__all__ = ["NcclCommRegistration", "register_external_nccl_comm"]


class NcclCommRegistration:
    r"""
    Handle returned by :func:`register_external_nccl_comm`.

    Keeps an externally-owned NCCL communicator published in PyTorch's
    symmetric memory registry for as long as this object is alive. Dropping it
    (via ``del``, exiting its ``with`` block, or calling :meth:`unregister`)
    removes the entry so a successor producer can register cleanly under the
    same ``group_name``.

    The comm's lifetime is *not* owned by this object; the registration only
    publishes a borrowed pointer. Optionally a strong reference to the source
    comm object is held so a stray ``del comm`` cannot dangle the registered
    pointer for the lifetime of the registration.
    """

    def __init__(
        self,
        group_name: str,
        device: torch.device,
        comm: object | None = None,
    ) -> None:
        self._group_name = group_name
        self._device = device
        self._comm = comm
        self._active = True

    def unregister(self) -> None:
        """Remove the registration. Idempotent."""
        if not self._active:
            return
        # Imported lazily: only present in NCCL builds with symmetric-memory
        # device support.
        from torch._C._distributed_c10d import _unregister_external_nccl_comm

        _unregister_external_nccl_comm(self._group_name, self._device)
        self._active = False
        self._comm = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.unregister()

    def __del__(self) -> None:
        try:
            self.unregister()
        except Exception:
            # Best-effort cleanup; never raise from __del__ (e.g. during
            # interpreter teardown the C extension may already be gone).
            pass


def register_external_nccl_comm(
    group_name: str,
    comm_ptr: int,
    device: _device,
    comm: object | None = None,
) -> NcclCommRegistration:
    r"""
    Publish an externally-owned ``ncclComm_t`` into PyTorch's symmetric memory
    registry under ``group_name``.

    This lets producers that are not a ``ProcessGroupNCCL`` (e.g. torchcomms
    backends) supply the host NCCL communicator that
    :func:`rendezvous` needs, without symmetric memory having to
    ``dynamic_cast`` to a specific process-group implementation. After this
    call, ``rendezvous(tensor, group=group_name)`` on a tensor in a process
    group registered under ``group_name`` will find this comm.

    Args:
        group_name (str): the process-group name to register the comm under.
        comm_ptr (int): the host ``ncclComm_t`` as an opaque integer pointer
            (e.g. from ``torchcomms`` ``get_nccl_comm_ptr()``).
        device (`torch.device` or str): the CUDA device the comm belongs to.
        comm (object, optional): the source comm object. When provided, a
            strong reference is held by the returned registration so the comm
            cannot be garbage-collected while still registered.

    Returns:
        NcclCommRegistration: keep this alive for as long as symmetric memory
        should use this comm; drop it (or call ``unregister()``) to remove the
        registration.

    .. note::
        This is only available in NCCL builds with symmetric-memory device
        support; otherwise it raises :class:`ImportError`.
    """
    from torch._C._distributed_c10d import (
        _register_external_nccl_comm as _register_external_nccl_comm_impl,
    )

    device = torch.device(device)
    _register_external_nccl_comm_impl(group_name, comm_ptr, device)
    return NcclCommRegistration(group_name, device, comm)
<|endoftext|>from torch.utils._triton import has_triton


if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_partials_first_dim_kernel(
        partials,
        out,
        shard_elems: tl.constexpr,
        group_size: tl.constexpr,
        do_avg: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < shard_elems
        acc = tl.zeros((BLOCK,), tl.float32)
        for rank in tl.static_range(0, group_size):
            vals = tl.load(
                partials + rank * shard_elems + offsets,
                mask=mask,
                other=0.0,
            )
            acc += vals.to(tl.float32)
        if do_avg:
            acc /= group_size
        # The accumulation stays in fp32 and is rounded exactly once, here.
        tl.store(out + offsets, acc.to(out.dtype.element_ty), mask=mask)
<|endoftext|>"""
Shared utilities for SHMEM Triton backends (_nvshmem_triton, _rocshmem_triton).

This is a leaf module — it does NOT import any backend or dispatch module,
so backend modules can safely import from here without circular dependencies.
"""

from collections.abc import Callable
from typing import Any

from torch.utils._triton import has_triton


class ShmemKernelRegistry:
    """Base registry tracking Triton kernels that need SHMEM module init."""

    _to_init: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str) -> None:
        cls._to_init.setdefault(name)

    @classmethod
    def deregister(cls, name: str) -> None:
        cls._to_init.pop(name, None)

    @classmethod
    def has(cls, name: str) -> bool:
        return name in cls._to_init


def run_shmem_init_hook(
    *,
    kwargs: dict[str, Any],
    registry: type[ShmemKernelRegistry],
    module_init: Callable[[Any], None],
    logger: Any,
) -> None:
    """Common post-compile hook logic shared by NVSHMEM and rocSHMEM backends."""
    jit_function = kwargs["fn"].jit_function
    fn_name = jit_function.fn.__name__

    if not registry.has(fn_name):
        return

    key = kwargs["key"]
    device = kwargs["compile"]["device"]
    kernel_cache = jit_function.device_caches[device][0]
    kernel = kernel_cache.get(key, None)
    if kernel is not None:
        _ = kernel.run
        module_init(kernel.module)
    else:
        logger.warning(
            "It seems Triton hasn't created a kernel for function %s. "
            "Please report this issue to Triton.",
            fn_name,
        )


if has_triton():
    from triton.runtime.jit import JITFunction, KernelInterface

    class GridCallableWithExtern(KernelInterface):
        """
        KernelInterface invokes self.run in __getitem__, i.e. [].
        We direct the call to JITFunction.run with extern_libs injected,
        so that users don't have to pass it.
        """

        def __init__(self, jit_func: JITFunction, extern_libs: dict[str, str]) -> None:
            self.jit_func = jit_func
            self.extern_libs = extern_libs

        def run(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return self.jit_func.run(*args, **kwargs, extern_libs=self.extern_libs)

    def build_requires_shmem_decorator(  # type: ignore[no-untyped-def]
        *,
        jit_func,
        find_device_library: Callable[[], str],
        extern_libs_key: str,
        registry: type[ShmemKernelRegistry],
        init_hook: Callable[..., None],
        error_prefix: str,
    ):
        """Build the @requires_<backend> decorator for a Triton JIT function."""
        import triton
        from triton.runtime.jit import JITFunction

        if not isinstance(jit_func, JITFunction):
            raise TypeError(
                f"{error_prefix} must be applied to a @triton.jit function, "
                f"got {type(jit_func)}"
            )

        lib_path = find_device_library()
        extern_libs = {extern_libs_key: lib_path}
        registry.register(jit_func.fn.__name__)
        triton.knobs.runtime.jit_post_compile_hook = init_hook
        return GridCallableWithExtern(jit_func, extern_libs)
<|endoftext|>"""
SHMEM Triton backend dispatch.

Thin abstraction layer that routes to the correct platform-specific module:
  - ROCm  - _rocshmem_triton
  - CUDA  - _nvshmem_triton


End users import from here to get backend-agnostic functionality:
  - get_shmem_backend_module(): returns the active backend module
  - requires_shmem(): backend-agnostic @requires_shmem decorator
"""

import torch


def get_shmem_backend_module():
    if torch.version.hip is not None:
        from torch.distributed._symmetric_memory import _rocshmem_triton

        return _rocshmem_triton
    from torch.distributed._symmetric_memory import _nvshmem_triton

    return _nvshmem_triton


def requires_shmem(jit_func):  # type: ignore[no-untyped-def]
    """
    Backend-agnostic Triton decorator for SHMEM kernels.

    Delegates to ``@requires_rocshmem`` on ROCm and ``@requires_nvshmem``
    on CUDA.
    """
    backend = get_shmem_backend_module()
    if torch.version.hip is not None:
        return backend.requires_rocshmem(jit_func)
    return backend.requires_nvshmem(jit_func)
<|endoftext|>"""
rocSHMEM Triton integration for AMD GPU (ROCm) builds.

This module provides the rocSHMEM-specific counterpart to the NVSHMEM Triton
integration in _nvshmem_triton.py.  It is only imported on ROCm builds
(i.e. when ``torch.version.hip is not None``).
"""

import logging
import os
import sysconfig
from typing import Any

import torch
from torch.distributed._symmetric_memory._shmem_triton_utils import (
    run_shmem_init_hook,
    ShmemKernelRegistry,
)
from torch.utils._triton import has_triton


logger = logging.getLogger(__name__)


class RocshmemLibFinder:
    """
    Find the architecture-specific rocSHMEM device bitcode library.

    Environment variable:
        ``ROCSHMEM_LIB_DIR`` (Optional[str]): directory containing
        ``librocshmem_device_{arch}.bc``.  When not set, the standard
        ROCm installation at ``/opt/rocm/lib`` is searched.

    Example::
        export ROCSHMEM_LIB_DIR=/opt/rocm/lib
    """

    found_device_lib_path: str | None = None

    @classmethod
    def find_device_library(cls) -> str:
        if cls.found_device_lib_path is not None:
            return cls.found_device_lib_path

        if not torch.cuda.is_available():
            raise RuntimeError(
                "ROCm/CUDA not available — cannot detect GPU architecture"
            )

        props = torch.cuda.get_device_properties(0)
        # gcnArchName returns e.g. "gfx942:sramecc+:xnack-"
        arch = props.gcnArchName.split(":")[0]
        logger.info("Detected GPU architecture: %s", arch)

        lib_name = f"librocshmem_device_{arch}.bc"

        search_paths = [
            os.path.join(sysconfig.get_path("purelib"), "amd", "rocshmem", "lib"),
            os.path.join(os.environ.get("ROCM_PATH") or "/opt/rocm", "lib"),
            "/usr/local/lib",
            "/usr/lib",
        ]

        user_lib_dir = os.environ.get("ROCSHMEM_LIB_DIR")
        if user_lib_dir is not None:
            lib_path = os.path.join(user_lib_dir, lib_name)
            if not os.path.exists(lib_path):
                raise RuntimeError(
                    f"rocSHMEM device library not found at ROCSHMEM_LIB_DIR: {lib_path}"
                )
            logger.info("Found rocSHMEM device library: %s", lib_path)
            cls.found_device_lib_path = lib_path
            return lib_path

        lib_path = None
        for path in search_paths:
            candidate = os.path.join(path, lib_name)
            if os.path.exists(candidate):
                lib_path = candidate
                break

        if lib_path is None:
            raise RuntimeError(
                f"rocSHMEM device library '{lib_name}' not found.\n"
                f"Searched: {search_paths}\n"
                "Set ROCSHMEM_LIB_DIR to the directory containing it."
            )
        logger.info("Found rocSHMEM device library: %s", lib_path)
        cls.found_device_lib_path = lib_path
        return lib_path


class RocshmemKernelRegistry(ShmemKernelRegistry):
    """Track Triton kernels that need rocSHMEM HIP-module initialization."""

    _to_init: dict[str, Any] = {}


def _rocshmem_init_hook(*args, **kwargs) -> None:  # type: ignore[no-untyped-def]
    """
    Post-compile hook that initializes rocSHMEM device context in the
    compiled HIP module using the existing c10d NVSHMEM entrypoint.
    """
    from torch._C._distributed_c10d import _nvshmemx_cumodule_init

    run_shmem_init_hook(
        kwargs=kwargs,
        registry=RocshmemKernelRegistry,
        module_init=_nvshmemx_cumodule_init,
        logger=logger,
    )


def requires_rocshmem(  # type: ignore[no-untyped-def]
    jit_func,
):
    """
    Decorator to mark a Triton kernel as requiring rocSHMEM device APIs.

    Finds the architecture-specific rocSHMEM bitcode library, registers
    the kernel for post-compile HIP-module initialization, and wraps the
    function so that ``extern_libs`` is injected automatically.

    Example::

        @requires_rocshmem
        @triton.jit
        def my_kernel(...):
            pe = rocshmem_my_pe()
            rocshmem_putmem_wg(dest, src, nbytes, target_pe)

    Set ``ROCSHMEM_LIB_DIR`` to override the default library search path.
    """
    from torch.distributed._symmetric_memory._shmem_triton_utils import (
        build_requires_shmem_decorator,
    )

    return build_requires_shmem_decorator(
        jit_func=jit_func,
        find_device_library=RocshmemLibFinder.find_device_library,
        extern_libs_key="rocshmem",
        registry=RocshmemKernelRegistry,
        init_hook=_rocshmem_init_hook,
        error_prefix="@requires_rocshmem",
    )


if has_triton():
    import triton
    import triton.language as tl
    from triton.language import core

    # -----------------------------------------------------------------------
    # rocSHMEM device API — Triton-callable device functions.
    #
    # All symbol names are the direct rocSHMEM device-bitcode names; no
    # NVSHMEM → rocSHMEM translation layer is needed here.
    #
    # RMA (block/wg-scoped):
    #   nvshmemx_putmem_block  → rocshmem_putmem_wg
    #   nvshmemx_getmem_block  → rocshmem_getmem_wg
    #   nvshmemx_getmem_nbi_block → rocshmem_getmem_nbi_wg
    #   nvshmemx_putmem_signal_block → rocshmem_putmem_signal_wg
    # Wait / signal:
    #   nvshmem_int_wait_until    → rocshmem_int_wait_until
    #   nvshmem_signal_wait_until → rocshmem_uint64_wait_until
    # Memory ordering, PE info, barriers: names are identical.
    # -----------------------------------------------------------------------

    @triton.jit
    def put(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        """Put *nelems* elements from local *source* to *dest* on remote *pe*."""
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return _putmem_wg(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def _putmem_wg(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int32"),
                ): ("rocshmem_putmem_wg", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def get(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        """Get *nelems* elements from *source* on remote *pe* into local *dest* (blocking)."""
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return _getmem_wg(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def _getmem_wg(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int32"),
                ): ("rocshmem_getmem_wg", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def get_nbi(dest, source, nelems, pe):  # type: ignore[no-untyped-def]
        """Non-blocking get; call quiet() for completion."""
        tl.static_assert(dest.type == source.type)
        nbytes = nelems * dest.type.element_ty.itemsize
        return _getmem_nbi_wg(
            dest.to(tl.int64), source.to(tl.int64), nbytes.to(tl.int64), pe
        )

    @core.extern
    def _getmem_nbi_wg(dest, source, size_bytes, pe, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [dest, source, size_bytes, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int32"),
                ): ("rocshmem_getmem_nbi_wg", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def putmem_signal_block(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
    ):
        """Put data to remote PE and atomically update a signal variable."""
        sig_val = 0 << 32 | sig_val
        return _putmem_signal_wg(
            dst.to(tl.int64),
            src.to(tl.int64),
            size_bytes.to(tl.int64),
            signal.to(tl.int64),
            sig_val.to(tl.uint64),
            sig_op,
            pe,
        )

    @core.extern
    def _putmem_signal_wg(  # type: ignore[no-untyped-def]
        dst,
        src,
        size_bytes,
        signal,
        sig_val,
        sig_op,
        pe,
        _semantic=None,
    ):
        return core.extern_elementwise(
            "",
            "",
            [dst, src, size_bytes, signal, sig_val, sig_op, pe],
            {
                (
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("int64"),
                    core.dtype("uint64"),
                    core.dtype("int32"),
                    core.dtype("int32"),
                ): ("rocshmem_putmem_signal_wg", core.dtype("int32"))
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def wait_until(ivar, cmp_op, cmp_val):  # type: ignore[no-untyped-def]
        """Block until *ivar* satisfies the comparison condition."""
        tl.static_assert(
            ivar.type.element_ty.itemsize == 4,
            "wait_until expects a 32-bit type for the synchronization variable",
        )
        return _int_wait_until(ivar.to(tl.int64), cmp_op, cmp_val)

    @core.extern
    def _int_wait_until(ivar, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [ivar, cmp, cmp_val],
            {
                (core.dtype("int64"), core.dtype("int32"), core.dtype("int32")): (
                    "rocshmem_int_wait_until",
                    core.dtype("int32"),
                )
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def signal_wait_until(signal, cmp, cmp_val):  # type: ignore[no-untyped-def]
        """Block until a uint64 signal variable satisfies the comparison condition."""
        cmp_val = 0 << 32 | cmp_val
        return _uint64_wait_until(signal.to(tl.int64), cmp, cmp_val.to(tl.uint64))

    @core.extern
    def _uint64_wait_until(signal, cmp, cmp_val, _semantic=None):  # type: ignore[no-untyped-def]
        return core.extern_elementwise(
            "",
            "",
            [signal, cmp, cmp_val],
            {
                (core.dtype("int64"), core.dtype("int32"), core.dtype("uint64")): (
                    "rocshmem_uint64_wait_until",
                    core.dtype("int32"),
                )
            },
            is_pure=False,
            _semantic=_semantic,
        )

    @triton.jit
    def signal_op(sig_addr, signal, sig_op, pe):  # type: ignore[no-untyped-def]
        """Not available in current rocSHMEM device bitcode."""
        tl.static_assert(
            False,
            "rocshmem has no device-bitcode equivalent for signal_op. "
            "Use rocshmem_uint64_atomic_set or rocshmem_uint64_atomic_add instead.",
        )

    @core.extern
    def fence(_semantic=None):  # type: ignore[no-untyped-def]
        """Ensure ordering of issued remote-memory operations to each target PE."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_fence", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def quiet(_semantic=None):  # type: ignore[no-untyped-def]
        """Wait for completion of all outstanding remote-memory operations."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_quiet", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def my_pe(_semantic=None):  # type: ignore[no-untyped-def]
        """Return the PE number of the calling PE."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_my_pe", core.dtype("int32"))},
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def n_pes(_semantic=None):  # type: ignore[no-untyped-def]
        """Return the total number of PEs."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_n_pes", core.dtype("int32"))},
            is_pure=True,
            _semantic=_semantic,
        )

    @core.extern
    def barrier_all(_semantic=None):  # type: ignore[no-untyped-def]
        """Barrier across all PEs with completion guarantee (workgroup-scoped)."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_barrier_all_wg", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    @core.extern
    def sync_all(_semantic=None):  # type: ignore[no-untyped-def]
        """Lightweight synchronization barrier across all PEs (workgroup-scoped)."""
        return core.extern_elementwise(
            "",
            "",
            [],
            {(): ("rocshmem_sync_all_wg", core.dtype("int32"))},
            is_pure=False,
            _semantic=_semantic,
        )

    # Collective stubs: rocSHMEM *_wg collectives are not exposed in current
    # device bitcode, so Triton kernels cannot call them yet.

    @triton.jit
    def alltoall(team, dest, source, nelems_per_pe):  # type: ignore[no-untyped-def]
        """Not available: rocshmem_alltoallmem_wg is not in current device bitcode."""
        tl.static_assert(
            False,
            "rocshmem_alltoallmem_wg is not available in current device bitcode. "
            "Use host-side rocshmem_alltoallmem_on_stream instead.",
        )

    @triton.jit
    def broadcast(team, dest, source, nelems, pe_root):  # type: ignore[no-untyped-def]
        """Not available: rocshmem_broadcastmem_wg is not in current device bitcode."""
        tl.static_assert(
            False,
            "rocshmem_broadcastmem_wg is not available in current device bitcode. "
            "Use host-side rocshmem_broadcastmem_on_stream instead.",
        )

    @triton.jit
    def reduce(team, dest, source, nreduce, operation: tl.constexpr):  # type: ignore[no-untyped-def]
        """Not available: rocshmem team reduce wg ops are not in current device bitcode."""
        tl.static_assert(
            False,
            "rocshmem team reduce is not available in current device bitcode. "
            "Use host-side rocshmem reduce API instead.",
        )
<|endoftext|># mypy: allow-untyped-defs
import warnings

import torch
import torch.distributed as dist
from torch.autograd import Function

# The two imports below are not always available depending on the
# USE_DISTRIBUTED compile flag. Make sure they raise import error
# if we're trying to use them.
from torch.distributed import group, ReduceOp


def _not_supported_under_compile(name, *, suggestion=None):
    msg = (
        f"torch.distributed.nn.functional.{name} is not supported under torch.compile."
    )
    if suggestion:
        msg += f" Use {suggestion} instead."
    raise RuntimeError(msg)


def _deprecated(name, suggestion):
    warnings.warn(
        f"torch.distributed.nn.functional.{name} is deprecated, "
        f"use {suggestion} instead.",
        category=FutureWarning,
        stacklevel=3,
    )


def broadcast(tensor, src, group=group.WORLD):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Received tensor from the broadcast op.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "broadcast",
            suggestion="torch.distributed._functional_collectives.broadcast",
        )
    _deprecated("broadcast", "torch.distributed._functional_collectives.broadcast")
    return _Broadcast.apply(src, group, tensor)


def gather(tensor, dst=0, group=group.WORLD):
    """
    Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        dst (int, optional): Destination rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple[Tensor]: List of appropriately-sized tensors with the gathered data.
    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile("gather")
    return _Gather.apply(dst, group, tensor)


def scatter(tensors, src=0, group=group.WORLD):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensors (list[Tensor]): List of tensors to scatter on the source rank.
            Receivers must pass ``None`.
        src (int, optional): Source rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output tensor from the scatter operation.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile("scatter")
    return _Scatter.apply(src, group, *tensors)


def reduce(tensor, dst, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input of the collective.
        dst (int): Destination rank.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile("reduce")
    return _Reduce.apply(dst, op, group, tensor)


def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Arguments:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "reduce_scatter",
            suggestion="torch.distributed._functional_collectives.reduce_scatter_single",
        )
    _deprecated(
        "reduce_scatter",
        "torch.distributed._functional_collectives.reduce_scatter_single",
    )
    return _Reduce_Scatter.apply(op, group, output, *input_list)


def all_gather(tensor, group=group.WORLD):
    """
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "all_gather",
            suggestion="torch.distributed._functional_collectives.all_gather_single",
        )
    _deprecated(
        "all_gather", "torch.distributed._functional_collectives.all_gather_single"
    )
    return _AllGather.apply(group, tensor)


def _all_gather_base(output_tensor, input_tensor, group=group.WORLD):
    """
    Single tensor all gather. Gathers a single tensor from all ranks, and puts them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Examples:
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> # xdoctest: +SKIP("incorrect want text")
        >>> output_tensor = torch.zeros(2, dtype=torch.int64)
        >>> output_tensor
        [tensor([0, 0])] # Rank 0 and 1
        >>> tensor = torch.arange(1, dtype=torch.int64) + 1 + rank
        >>> tensor
        tensor([1]) # Rank 0
        tensor([2]) # Rank 1
        >>> dist.all_gather_base(output_tensor, tensor)
        >>> output_tensor
        tensor([1,2]) # Rank 0
        tensor([1,2]) # Rank 1

    .. warning::
        `_all_gather_base` is experimental and subject to change.
        It is the caller's responsibility to ensure the output_tensor
        is correctly sized.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile("_all_gather_base")
    return _AllGatherBase.apply(output_tensor, input_tensor, group)


def all_to_all(output_tensor_list, input_tensor_list, group=group.WORLD):
    """
    Each process scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

    Arguments:
        output_tensor_list (list[Tensor]): list of tensors to gather one per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile("all_to_all")
    return _AlltoAll.apply(group, output_tensor_list, *input_tensor_list)


def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=group.WORLD,
):
    """
    Each process splits input tensor and then scatters the split list to all processes in a group.

    Then concatenate the received tensors from all the processes in the group and return single output tensor.

    Arguments:
        output (Tensor): Gathered concatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.

    Returns:
        Tensor: Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "all_to_all_single",
            suggestion="torch.distributed._functional_collectives.all_to_all_single",
        )
    _deprecated(
        "all_to_all_single",
        "torch.distributed._functional_collectives.all_to_all_single",
    )
    return _AlltoAllSingle.apply(
        group, output, output_split_sizes, input_split_sizes, input
    )


def all_reduce(tensor, op=ReduceOp.SUM, group=group.WORLD):
    """
    Reduces the tensor data across all machines in such a way that all get the final result.

    After the call the returned tensor is going to be bitwise
    identical in all processes.

    Arguments:
        tensor (Tensor): Input of the collective.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "all_reduce",
            suggestion="torch.distributed._functional_collectives.all_reduce",
        )
    _deprecated("all_reduce", "torch.distributed._functional_collectives.all_reduce")
    return _AllReduce.apply(op, group, tensor)


class _Broadcast(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, src, group, tensor):
        ctx.src = src
        ctx.group = group
        ctx.global_rank = dist.get_rank()
        # torch.distributed makes all the calls in place
        # we allocate new tensors to avoid this
        tensor = tensor.clone()
        dist.broadcast(tensor, src, group=group)
        return tensor

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        gx = _Reduce.apply(ctx.src, ReduceOp.SUM, ctx.group, grad_output)
        if ctx.src != ctx.global_rank:
            gx.zero_()
        return (None, None, gx)


class _Gather(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, dst, group, tensor):
        ctx.dst = dst
        ctx.group = group
        # Need to create a list of tensors here to do the
        # aggregation, get it from the group size
        # tensor should be correctly sized for the method
        # gathering
        tensor_list = [
            torch.zeros_like(tensor) for i in range(dist.get_world_size(group=group))
        ]

        tensor = tensor.contiguous()
        if dist.get_rank(group=group) == dst:
            dist.gather(tensor, tensor_list, dst, group=group)
        else:
            dist.gather(tensor, None, dst, group=group)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None) + (_Scatter.apply(ctx.dst, ctx.group, *grad_outputs),)


class _Scatter(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, src, group, *tensors):
        ctx.src = src
        ctx.group = group
        if not all(t.size() == tensors[0].size() for t in tensors):
            raise AssertionError
        output = torch.zeros_like(tensors[0])
        if dist.get_rank(group=group) == src:
            dist.scatter(output, list(tensors), src, group=group)
        else:
            dist.scatter(output, None, src, group=group)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        return (None, None) + _Gather.apply(ctx.src, ctx.group, grad_output)


class _Reduce(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, src, op, group, tensor):
        ctx.src = src
        ctx.group = group
        tensor = tensor.clone()
        dist.reduce(tensor, src, op=op, group=group)
        return tensor

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        return (None, None, None) + (_Broadcast.apply(ctx.src, ctx.group, grad_output),)


class _Reduce_Scatter(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, op, group, tensor, *input_tensor_list):
        ctx.group = group
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()
        input_tensor_list = tuple(t.contiguous() for t in input_tensor_list)
        dist.reduce_scatter(tensor, list(input_tensor_list), op=op, group=group)
        return tensor

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        return (None, None, None) + _AllGather.apply(ctx.group, grad_output)


class _AllGather(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, group, tensor):
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()

        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]

        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        if dist.get_backend(group=ctx.group) in (dist.Backend.NCCL, dist.Backend.XCCL):
            rank = dist.get_rank(group=ctx.group)
            gx = torch.empty_like(grad_outputs[rank])
            gx = _Reduce_Scatter.apply(ReduceOp.SUM, ctx.group, gx, *grad_outputs)
        else:
            # As many backends doesn't support ReduceScatter, we use AlltoAll with .sum()
            # to emulate the ReduceScatter behavior
            tensor_list = [torch.empty_like(tensor) for tensor in grad_outputs]
            gxs = _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)
            gx = torch.sum(torch.stack(gxs), dim=0)
        return (None, gx)


class _AllGatherBase(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, output_tensor, input_tensor, group):
        ctx.group = group
        dist._all_gather_base(output_tensor, input_tensor.contiguous(), group=group)
        return output_tensor

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        if dist.get_backend(group=ctx.group) in (dist.Backend.NCCL, dist.Backend.XCCL):
            world_size = dist.get_world_size(group=ctx.group)
            out_size = list(grad_output.size())
            if out_size[0] % world_size != 0:
                raise RuntimeError(
                    f"Tensor with dimensions: {out_size} does "
                    f"not have first dimension divisible by world_size: {world_size}"
                )
            out_size[0] = out_size[0] // dist.get_world_size(group=ctx.group)
            gx = torch.empty(
                out_size, device=grad_output.device, dtype=grad_output.dtype
            )
            dist._reduce_scatter_base(gx, grad_output, ReduceOp.SUM, ctx.group)
        else:
            raise RuntimeError("Backend not supported!")
        return (None, gx, None)


class _AlltoAll(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, group, out_tensor_list, *tensors):
        ctx.group = group
        ctx.input_tensor_size_list = [
            tensors[i].size() for i in range(dist.get_world_size(group=group))
        ]
        my_rank = dist.get_rank(group=group)
        tensors = tuple(t.contiguous() for t in tensors)
        # Implement it on means of scatter/gather, send/recv async operations have issues
        if dist.get_backend(group=group) is dist.Backend.GLOO:
            for i in range(dist.get_world_size(group=group)):
                to_send = None
                if i == my_rank:
                    to_send = list(tensors)
                dist.scatter(out_tensor_list[i], to_send, i, group=group)
        else:
            dist.all_to_all(
                out_tensor_list,
                list(tensors),
                group=group,
            )
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        tensor_list = [
            torch.empty(
                size, device=grad_outputs[0].device, dtype=grad_outputs[0].dtype
            )
            for size in ctx.input_tensor_size_list
        ]
        return (None, None) + _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)


class _AlltoAllSingle(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, group, output, output_split_sizes, input_split_sizes, input):
        ctx.group = group
        ctx.input_size = input.size()
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        tensor = torch.empty(
            ctx.input_size, device=grad_output.device, dtype=grad_output.dtype
        )
        return (None, None, None, None) + (
            _AlltoAllSingle.apply(
                ctx.group,
                tensor,
                ctx.output_split_sizes,
                ctx.input_split_sizes,
                grad_output.contiguous(),
            ),
        )


class _AllReduce(Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, op, group, tensor):
        ctx.group = group
        ctx.op = op
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)
<|endoftext|>#!/usr/bin/python3
# mypy: allow-untyped-defs
import importlib.abc
import importlib.util
import sys

import torch
from torch.distributed.nn.jit.templates.remote_module_template import (
    get_remote_module_template,
)


_FILE_PREFIX = "_remote_module_"


def get_arg_return_types_from_interface(module_interface):
    if not getattr(module_interface, "__torch_script_interface__", False):
        raise AssertionError(
            "Expect a TorchScript class interface decorated by @torch.jit.interface."
        )
    qualified_name = torch._jit_internal._qualified_name(module_interface)
    cu = torch.jit._state._python_cu
    module_interface_c = cu.get_interface(qualified_name)
    if "forward" not in module_interface_c.getMethodNames():
        raise AssertionError(
            f"Expect forward in interface methods, while it has {module_interface_c.getMethodNames()}"
        )
    method_schema = module_interface_c.getMethod("forward")

    arg_str_list = []
    arg_type_str_list = []
    if method_schema is None:
        raise AssertionError
    for argument in method_schema.arguments:
        arg_str_list.append(argument.name)

        if argument.has_default_value():
            default_value_str = f" = {argument.default_value}"
        else:
            default_value_str = ""
        arg_type_str = f"{argument.name}: {argument.type}{default_value_str}"
        arg_type_str_list.append(arg_type_str)

    arg_str_list = arg_str_list[1:]  # Remove "self".
    args_str = ", ".join(arg_str_list)

    arg_type_str_list = arg_type_str_list[1:]  # Remove "self".
    arg_types_str = ", ".join(arg_type_str_list)

    if len(method_schema.returns) != 1:
        raise AssertionError
    argument = method_schema.returns[0]
    return_type_str = str(argument.type)

    return args_str, arg_types_str, return_type_str


class _StringLoader(importlib.abc.SourceLoader):
    """
    A custom loader for dynamically generated Python source code.

    Inherits from SourceLoader for API compatibility but overrides exec_module()
    to avoid bytecode caching issues. The default SourceLoader.exec_module() calls
    cache_from_source() which fails with IndexError when the filename doesn't
    correspond to a real filesystem path with a .py extension.
    """

    def __init__(self, data: str) -> None:
        self.data = data

    def get_source(self, fullname: str) -> str:
        return self.data

    def get_data(self, path: str) -> bytes:
        return self.data.encode("utf-8")

    def get_filename(self, fullname: str) -> str:
        return f"<{fullname}>.py"

    def path_stats(self, path: str) -> dict:
        # Raise OSError since source is dynamically generated (no filesystem stats)
        raise OSError("dynamically generated module has no filesystem stats")

    def exec_module(self, module) -> None:
        """
        Execute the module by compiling and running the source directly.

        This overrides SourceLoader.exec_module() to bypass the problematic
        get_code() -> cache_from_source() code path that fails on dynamic modules.
        """
        source = self.get_source(module.__name__)
        filename = self.get_filename(module.__name__)
        code = compile(source, filename, "exec", dont_inherit=True)
        exec(code, module.__dict__)


def _do_instantiate_remote_module_template(
    generated_module_name, str_dict, enable_moving_cpu_tensors_to_cuda
):
    if generated_module_name in sys.modules:
        return sys.modules[generated_module_name]

    loader = _StringLoader(
        get_remote_module_template(enable_moving_cpu_tensors_to_cuda).format(**str_dict)
    )
    spec = importlib.util.spec_from_loader(
        generated_module_name, loader, origin="torch-git"
    )
    if spec is None:
        raise AssertionError
    module = importlib.util.module_from_spec(spec)
    sys.modules[generated_module_name] = module
    loader.exec_module(module)
    return module


def instantiate_scriptable_remote_module_template(
    module_interface_cls, enable_moving_cpu_tensors_to_cuda=True
):
    if not getattr(module_interface_cls, "__torch_script_interface__", False):
        raise ValueError(
            f"module_interface_cls {module_interface_cls} must be a type object decorated by "
            "@torch.jit.interface"
        )

    # Generate the template instance name.
    module_interface_cls_name = torch._jit_internal._qualified_name(
        module_interface_cls
    ).replace(".", "_")
    generated_module_name = f"{_FILE_PREFIX}{module_interface_cls_name}"

    # Generate type annotation strs.
    assign_module_interface_cls_str = (
        f"from {module_interface_cls.__module__} import "
        f"{module_interface_cls.__name__} as module_interface_cls"
    )
    args_str, arg_types_str, return_type_str = get_arg_return_types_from_interface(
        module_interface_cls
    )
    kwargs_str = ""
    arrow_and_return_type_str = f" -> {return_type_str}"
    arrow_and_future_return_type_str = f" -> Future[{return_type_str}]"

    str_dict = dict(
        assign_module_interface_cls=assign_module_interface_cls_str,
        arg_types=arg_types_str,
        arrow_and_return_type=arrow_and_return_type_str,
        arrow_and_future_return_type=arrow_and_future_return_type_str,
        args=args_str,
        kwargs=kwargs_str,
        jit_script_decorator="@torch.jit.script",
    )
    return _do_instantiate_remote_module_template(
        generated_module_name, str_dict, enable_moving_cpu_tensors_to_cuda
    )


def instantiate_non_scriptable_remote_module_template():
    generated_module_name = f"{_FILE_PREFIX}non_scriptable"
    str_dict = dict(
        assign_module_interface_cls="module_interface_cls = None",
        args="*args",
        kwargs="**kwargs",
        arg_types="*args, **kwargs",
        arrow_and_return_type="",
        arrow_and_future_return_type="",
        jit_script_decorator="",
    )
    # For a non-scriptable template, always enable moving CPU tensors to a cuda device,
    # because there is no syntax limitation on the extra handling caused by the script.
    return _do_instantiate_remote_module_template(generated_module_name, str_dict, True)
<|endoftext|>#!/usr/bin/python3
# mypy: allow-untyped-defs


def get_remote_module_template(enable_moving_cpu_tensors_to_cuda: bool):
    return _TEMPLATE_PREFIX + (
        _REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA
        if enable_moving_cpu_tensors_to_cuda
        else _REMOTE_FORWARD_TEMPLATE
    )


_TEMPLATE_PREFIX = """from typing import *

import torch
import torch.distributed.rpc as rpc
from torch import Tensor
from torch._jit_internal import Future
from torch.distributed.rpc import RRef
from typing import Tuple  # pyre-ignore: unused import


{assign_module_interface_cls}


def forward_async(self, {arg_types}){arrow_and_future_return_type}:
    args = (self.module_rref, self.device, self.is_device_map_set, {args})
    kwargs = {{{kwargs}}}
    return rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args,
        kwargs,
    )


def forward(self, {arg_types}){arrow_and_return_type}:
    args = (self.module_rref, self.device, self.is_device_map_set, {args})
    kwargs = {{{kwargs}}}
    ret_fut = rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args,
        kwargs,
    )
    return ret_fut.wait()


_generated_methods = [
    forward_async,
    forward,
]


{jit_script_decorator}
"""

# This template may cause typing error (the mismatch between ``Tuple[()]`` and ``Tuple[Any]``)
# even if the code is only used for instantiation but not execution.
# Therefore, only include handling moving CPU tensors to a cuda device if necessary.
# TODO: Merge these two templates together in the future once TorchScript syntax is improved.
_REMOTE_FORWARD_TEMPLATE_ENABLE_MOVING_CPU_TENSORS_TO_CUDA = """
def _remote_forward(
    module_rref: RRef[module_interface_cls], device: str, is_device_map_set: bool, {arg_types}){arrow_and_return_type}:
    module = module_rref.local_value()
    device = torch.device(device)

    if device.type != "cuda":
        return module.forward({args}, {kwargs})

    # If the module is on a cuda device,
    # move any CPU tensor in args or kwargs to the same cuda device.
    # Since torch script does not support generator expression,
    # have to use concatenation instead of
    # ``tuple(i.to(device) if isinstance(i, Tensor) else i for i in *args)``.
    args = ({args},)
    out_args: Tuple[()] = ()
    for arg in args:
        arg = (arg.to(device),) if isinstance(arg, Tensor) else (arg,)
        out_args = out_args + arg

    kwargs = {{{kwargs}}}
    for k, v in kwargs.items():
        if isinstance(v, Tensor):
            kwargs[k] = kwargs[k].to(device)

    if is_device_map_set:
        return module.forward(*out_args, {kwargs})

    # If the device map is empty, then only CPU tensors are allowed to send over wire,
    # so have to move any GPU tensor to CPU in the output.
    # Since torch script does not support generator expression,
    # have to use concatenation instead of
    # ``tuple(i.cpu() if isinstance(i, Tensor) else i for i in module.forward(*out_args, {kwargs}))``.
    ret: Tuple[()] = ()
    for i in module.forward(*out_args, {kwargs}):
        i = (i.cpu(),) if isinstance(i, Tensor) else (i,)
        ret = ret + i
    return ret
"""

_REMOTE_FORWARD_TEMPLATE = """
def _remote_forward(
    module_rref: RRef[module_interface_cls], device: str, is_device_map_set: bool, {arg_types}){arrow_and_return_type}:
    module = module_rref.local_value()

    return module.forward({args}, {kwargs})
"""
<|endoftext|>#!/usr/bin/python3
# mypy: allow-untyped-defs
import collections
import io
import sys
import types
from collections.abc import Callable, Iterator, Mapping
from typing import Any, TypeVar
from typing_extensions import Self

import torch
import torch.distributed.rpc as rpc
from torch import device, dtype, nn, Tensor
from torch.distributed import _remote_device
from torch.distributed.nn.jit import instantiator
from torch.distributed.rpc.internal import _internal_rpc_pickler
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle


__all__ = ["RemoteModule"]

_grad_t = tuple[Tensor, ...] | Tensor
# See https://mypy.readthedocs.io/en/latest/generics.html#generic-methods-and-generic-self for the use
# of `T` to annotate `self`. Many methods of `Module` return `self` and we want those return values to be
# the type of the subclass, not the looser type of `Module`.
T = TypeVar("T", bound="Module")

_NON_SCRIPTABLE_REMOTE_MODULE_MODULE = (
    instantiator.instantiate_non_scriptable_remote_module_template()
)

_REMOTE_MODULE_PICKLED_ATTRIBUTES = (
    "on",
    "device",
    "is_device_map_set",
    "is_scriptable",
    "generated_methods",
    "module_rref",
)

_SerializedRemoteModule = collections.namedtuple(  # type: ignore[misc]
    "_SerializedRemoteModule",
    _REMOTE_MODULE_PICKLED_ATTRIBUTES,
)

# These attributes are mostly from RemoteModule's parent class and are intentionally not pickled.
# A new attribute of RemoteModule should be either in _REMOTE_MODULE_PICKLED_ATTRIBUTES
# or _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING.
# Otherwise, it will not be pickled.
_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING = (
    "training",
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_is_full_backward_hook",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_state_dict_pre_hooks",
    "_modules",
    # The two attributes below are generated methods, not available at pickling time.
    "forward_async",
    "forward",
)


# RPC handler.
def _instantiate_template(module_interface_cls, enable_moving_cpu_tensors_to_cuda):
    instantiator.instantiate_scriptable_remote_module_template(
        module_interface_cls, enable_moving_cpu_tensors_to_cuda
    )


def _create_module(module_cls, args, kwargs, device):
    module = module_cls(*args, **kwargs)
    if not isinstance(module, nn.Module):
        raise ValueError(
            "Expect `module_cls(*args, **kwargs)` returns an instance of <class nn.Module>, "
            f"but it returns an instance of {type(module)}."
        )
    module.to(device)
    return module


def _create_module_with_interface(
    module_cls, args, kwargs, device, module_interface_cls
):
    module = _create_module(module_cls, args, kwargs, device)
    if module_interface_cls is not None:
        module = torch.jit.script(module)
    return rpc.RRef(module, module_interface_cls)


def _param_rrefs(module_rref, recurse) -> list[rpc.RRef[Parameter]]:
    ret: list[rpc.RRef[Parameter]] = [
        rpc.RRef(param) for param in module_rref.local_value().parameters(recurse)
    ]
    return ret


def _raise_not_supported(name: str) -> None:
    raise ValueError(f"Method ``{name}`` not supported for RemoteModule")


class _RemoteModule(nn.Module):
    def __new__(cls, *args, **kwargs):
        # Use __new__ for logging purposes.
        torch._C._log_api_usage_once("torch.distributed.nn.api.remote_module")
        return super().__new__(cls)

    def __init__(
        self,
        remote_device: str,
        module_cls: type[nn.Module],
        args: tuple | None = None,
        kwargs: dict[str, Any] | None = None,
        _module_interface_cls: Any = None,
    ):
        """
        RemoteModule instance can only be created after RPC initialization.

        It creates a user-specified module on a specified remote node.
        It behaves like a regular ``nn.Module`` except that the ``forward`` method is
        executed on the remote node.
        It takes care of autograd recording to ensure the backward pass propagates
        gradients back to the corresponding remote module.
        It can be shared across processors using `RPC framework <https://pytorch.org/docs/stable/rpc.html>`__,
        without incurring any overheads of copying the actual module,
        which is equivalent to an :class:`~torch.distributed.rpc.RRef`
        pointing to the remote module.

        The arguments of ``forward_async`` and ``forward`` are the same as
        the ``forward`` method of the module returned by the ``module_cls``.

        Apart from ``forward_async`` and ``forward``, no other methods are supported from nn.Module for now.

        Particularly, to create a hybrid model, typically the local modules should be
        created outside of remote modules, rather than as submodules of any remote module (by calling ``add_module``).
        Hybrid Example:
                >>> class HybridModel(nn.Module):
                >>>     def __init__(self) -> None:
                >>>         nn.Module.__init__(self)
                >>>         self.remote_embedding = RemoteModule(...)
                >>>         self.local_linear = nn.Linear(...)

        For example, if ``module_cls`` returns an instance of ``nn.Linear``,
        that has ``forward`` method signature, ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods in signature of
        ``def forward(input: Tensor) -> Tensor:`` and
        ``def forward_async(input: Tensor) -> Future[Tensor]:``.

        .. note::
            If the remote module is placed on a cuda device,
            any input CPU tensors will be automatically moved to the same cuda device,
            and GPU tensors are returned over the wire according to the device map of the remote worker on TensorPipe RPC backend.

        Args:
            remote_device (str): Device on the destination worker where we'd like to place this module.
                The device can be a local device or a remote device specified by one of the following remote
                formats:

                    1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
                    2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").

                In addition, the device field can be optional and the default value is "cpu".
            module_cls (nn.Module): For example,
                >>> class MyModule(nn.Module):
                >>>     def forward(input):
                >>>         return input + 1
                >>>
                >>> module_cls = MyModule
            args (Sequence, optional): args to be passed to ``module_cls``.
            kwargs (Dict, optional): kwargs to be passed to ``module_cls``.
            _module_interface_cls (type, optional): The TorchScript interface type for the module
                to be created. The type object should be decorated by @torch.jit.interface.
                If not provided, the generated RemoteModule is not torchscript-able.
                Warning, this is an experimental API and susceptible to frequent changes.

        Returns:
            A remote module instance which wraps the :class:`~nn.Module` created by the
            user-provided ``module_cls``, it has a blocking ``forward`` method and an
            asynchronous ``forward_async`` method that returns a future of the ``forward`` call
            on the user-provided module on the remote side.

        Example::
            Run the following code in two different processes:

            >>> # xdoctest: +SKIP("distributed")
            >>> # On worker 0:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>> from torch import nn, Tensor
            >>> from torch.distributed.nn.api.remote_module import RemoteModule
            >>>
            >>> rpc.init_rpc("worker0", rank=0, world_size=2)
            >>> remote_linear_module = RemoteModule(
            >>>     "worker1/cpu", nn.Linear, args=(20, 30),
            >>> )
            >>> input = torch.randn(128, 20)
            >>> ret_fut = remote_linear_module.forward_async(input)
            >>> ret = ret_fut.wait()
            >>> rpc.shutdown()

            >>> # On worker 1:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>>
            >>> rpc.init_rpc("worker1", rank=1, world_size=2)
            >>> rpc.shutdown()
        """
        super().__init__()

        enable_moving_cpu_tensors_to_cuda = self._prepare_init(remote_device)

        # Default arguments preparation.
        args = args if args is not None else ()
        kwargs = kwargs if kwargs is not None else {}

        if _module_interface_cls is not None:
            # Users reply on this field to know if this generated RemoteModule is TorchScript-able.
            self.is_scriptable = True

            # Instantiate template on remote side.
            fut = rpc.rpc_async(
                self.on,
                _instantiate_template,
                (_module_interface_cls, enable_moving_cpu_tensors_to_cuda),
            )

            self._init_template(
                _module_interface_cls, enable_moving_cpu_tensors_to_cuda
            )

            # Instantiate template on remote side.
            fut = rpc.rpc_async(
                self.on,
                _instantiate_template,
                (_module_interface_cls, enable_moving_cpu_tensors_to_cuda),
            )

            # Create the module on the remote side.
            fut.wait()  # Ensure remote_module_cls is available on remote side.

            # TODO: We need to change this to rpc.remote, and make it async (see the else branch below).
            # For that we need to be able to apply _module_interface_cls to the RRef returned by rpc.remote
            # See https://github.com/pytorch/pytorch/issues/58098 for more context.
            self.module_rref = rpc.rpc_sync(
                self.on,
                _create_module_with_interface,
                (module_cls, args, kwargs, self.device, _module_interface_cls),
            )
        else:
            self.is_scriptable = False
            self.generated_methods = (
                _NON_SCRIPTABLE_REMOTE_MODULE_MODULE._generated_methods
            )
            # Create the module on the remote side.
            self.module_rref = rpc.remote(
                self.on,
                _create_module,
                (module_cls, args, kwargs, self.device),
            )

        self._install_generated_methods()
        self._check_attribute_picklability()

    def remote_parameters(self, recurse: bool = True) -> list[rpc.RRef[Parameter]]:
        """
        Return a list of :class:`~torch.distributed.rpc.RRef` pointing to the remote module's parameters.

        This can typically be used in conjunction
        with :class:`~torch.distributed.optim.DistributedOptimizer`.

        Args:
            recurse (bool): if True, then returns parameters of the remote
                module and all submodules of the remote module. Otherwise,
                returns only parameters that are direct members of the
                remote module.

        Returns:
            A list of :class:`~torch.distributed.rpc.RRef` (``List[RRef[nn.Parameter]]``)
            to remote module's parameters.
        """
        return rpc.rpc_sync(self.on, _param_rrefs, args=(self.module_rref, recurse))

    def get_module_rref(self) -> rpc.RRef[nn.Module]:
        """Return an :class:`~torch.distributed.rpc.RRef` (``RRef[nn.Module]``) pointing to the remote module."""
        return self.module_rref

    @torch.jit.export
    def __getstate__(self):
        raise RuntimeError(
            "Cannot pickle RemoteModule in python pickler. RemoteModule can only be pickled when using RPC"
        )

    @torch.jit.export
    def __setstate__(self, state):
        raise RuntimeError(
            "Cannot unpickle RemoteModule in python pickler. RemoteModule can only be unpickled when using RPC"
        )

    def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None:
        _raise_not_supported(self.register_buffer.__name__)

    def register_parameter(self, name: str, param: Parameter | None) -> None:
        _raise_not_supported(self.register_parameter.__name__)

    def add_module(self, name: str, module: Module | None) -> None:
        _raise_not_supported(self.add_module.__name__)

    def apply(self, fn: Callable[[Module], None]) -> Self:  # type: ignore[return]
        _raise_not_supported(self.apply.__name__)

    def cuda(self, device: int | device | None = None) -> Self:  # type: ignore[return]
        _raise_not_supported(self.cuda.__name__)

    def ipu(self, device: int | device | None = None) -> Self:  # type: ignore[return]
        _raise_not_supported(self.ipu.__name__)

    def xpu(self, device: int | device | None = None) -> Self:  # type: ignore[return]
        _raise_not_supported(self.xpu.__name__)

    def cpu(self) -> Self:  # type: ignore[return]
        _raise_not_supported(self.cpu.__name__)

    def type(self, dst_type: dtype | str) -> Self:  # type: ignore[return]
        _raise_not_supported(self.type.__name__)

    def float(self) -> Self:  # type: ignore[return]
        _raise_not_supported(self.float.__name__)

    def double(self) -> Self:  # type: ignore[return]
        _raise_not_supported(self.double.__name__)

    def half(self) -> Self:  # type: ignore[return]
        _raise_not_supported(self.half.__name__)

    def bfloat16(self) -> Self:  # type: ignore[return]
        _raise_not_supported(self.bfloat16.__name__)

    def to(self, *args, **kwargs) -> T:  # type: ignore[misc, return, type-var]
        _raise_not_supported(self.to.__name__)

    def register_backward_hook(  # type: ignore[return]
        self,
        hook: Callable[[Module, _grad_t, _grad_t], _grad_t | None],
        # pyrefly: ignore [bad-return]
    ) -> RemovableHandle:
        _raise_not_supported(self.register_backward_hook.__name__)

    def register_forward_pre_hook(  # type: ignore[return]
        self,
        hook: Callable[[T, tuple[Any, ...]], Any | None]
        | Callable[
            [T, tuple[Any, ...], dict[str, Any]], tuple[Any, dict[str, Any]] | None
        ],
        prepend: bool = False,
        with_kwargs: bool = False,
        # pyrefly: ignore [bad-return]
    ) -> RemovableHandle:
        _raise_not_supported(self.register_forward_pre_hook.__name__)

    def register_forward_hook(  # type: ignore[return, override]
        self,
        hook: Callable[[T, tuple[Any, ...], Any], Any | None]
        | Callable[[T, tuple[Any, ...], dict[str, Any], Any], Any | None],
        prepend: bool = False,
        with_kwargs: bool = False,
        # pyrefly: ignore [bad-return]
    ) -> RemovableHandle:
        _raise_not_supported(self.register_forward_hook.__name__)

    def state_dict(self, *args, **kwargs):
        _raise_not_supported(self.state_dict.__name__)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        _raise_not_supported(self.load_state_dict.__name__)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        raise ValueError(
            "Method ``parameters`` not supported for RemoteModule. Please use ``remote_parameters`` instead."
        )

    def named_parameters(  # type: ignore[return]
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
        # pyrefly: ignore [bad-return]
    ) -> Iterator[tuple[str, Parameter]]:
        _raise_not_supported(self.named_parameters.__name__)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:  # type: ignore[return]
        _raise_not_supported(self.buffers.__name__)

    def named_buffers(  # type: ignore[return]
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
        # pyrefly: ignore [bad-return]
    ) -> Iterator[tuple[str, Tensor]]:
        _raise_not_supported(self.named_buffers.__name__)

    def children(self) -> Iterator[Module]:  # type: ignore[return]
        _raise_not_supported(self.children.__name__)

    def named_children(self) -> Iterator[tuple[str, Module]]:  # type: ignore[return]
        _raise_not_supported(self.named_children.__name__)

    def modules(self) -> Iterator[Module]:  # type: ignore[return]
        _raise_not_supported(self.modules.__name__)

    def named_modules(
        self,
        memo: set[Module] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        _raise_not_supported(self.named_modules.__name__)

    def train(self, mode: bool = True) -> Self:
        return self.module_rref.rpc_sync().train()  # type: ignore[operator, union-attr]

    def eval(self) -> Self:
        return self.module_rref.rpc_sync().eval()  # type: ignore[operator, union-attr]

    def requires_grad_(self, requires_grad: bool = True) -> Self:  # type: ignore[return]
        _raise_not_supported(self.requires_grad_.__name__)

    def zero_grad(self, set_to_none: bool = True) -> None:
        _raise_not_supported(self.zero_grad.__name__)

    def share_memory(self) -> Self:  # type: ignore[return]
        _raise_not_supported(self.share_memory.__name__)

    def extra_repr(self) -> str:  # type: ignore[return]
        _raise_not_supported(self.extra_repr.__name__)

    def _prepare_init(self, remote_device_str: str) -> bool:
        """Prepare the initialization and returns whether to enable automatically moving CPU tensors to CUDA devices."""
        # Sanity check.
        if not rpc._is_current_rpc_agent_set():
            raise AssertionError("RemoteModule only works in RPC.")

        remote_device = _remote_device(remote_device_str)
        self.on = (
            remote_device.worker_name()
            if remote_device.worker_name() is not None
            else remote_device.rank()
        )
        self.device = str(remote_device.device())
        agent = rpc._get_current_rpc_agent()
        # If the device map of the remote worker is set,
        # then enable moving any input CPU tensors to the same cuda device.
        self.is_device_map_set = bool(
            agent._get_device_map(agent.get_worker_info(self.on))  # type: ignore[arg-type]
        )
        # ``enable_moving_cpu_tensors_to_cuda`` is less strict than ``is_device_map_set``:
        # If ``enable_moving_cpu_tensors_to_cuda`` is true, but the device map is not set,
        # then any CPU tensors can still be moved to a cuda device to run forward,
        # but the output must be moved back to CPU before being sent over the wire.
        enable_moving_cpu_tensors_to_cuda = torch.device(self.device).type == "cuda"
        return enable_moving_cpu_tensors_to_cuda

    def _init_template(self, module_interface_cls, enable_moving_cpu_tensors_to_cuda):
        """Instantiate template on local side."""
        generated_module = instantiator.instantiate_scriptable_remote_module_template(
            module_interface_cls, enable_moving_cpu_tensors_to_cuda
        )
        self.generated_methods = generated_module._generated_methods

    def _check_attribute_picklability(self):
        """Check if all the attribute has explicitly defined whether to be pickled (i.e., picklability)."""
        for k in self.__dict__:
            if (
                k not in _REMOTE_MODULE_PICKLED_ATTRIBUTES
                and k not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING
            ):
                raise AttributeError(
                    f"Attribute {k} must be either in ``_REMOTE_MODULE_PICKLED_ATTRIBUTES`` or "
                    "``_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING``."
                )

    def _install_generated_methods(self):
        for method in self.generated_methods:
            method_name = method.__name__
            method = torch.jit.export(method)
            setattr(self, method_name, types.MethodType(method, self))

    @staticmethod
    def init_from_module_rref(
        remote_device: str,
        module_rref: rpc.RRef[nn.Module],
        _module_interface_cls: Any = None,
    ):
        """
        Besides the constructor, a RemoteModule instance can also be initialized given a module RRef.

        This alternate initialization method can be particularly useful if we want to create multiple
        RemoteModule instances that share the same underlying module and reduce memory consumption.

        Moreover, this also provides a workaround for passing script RemoteModule over RPC,
        which is not supported. The recommended way is as follows:

            1. the sender creates a RemoteModule;
            2. the sender sends its ``module_rref`` over RPC;
            3. the receiver calls this method to initialize another RemoteModule using the same ``module_rref``.

        Example::
            Run the following code in two different processes:

            >>> # xdoctest: +SKIP("distributed")
            >>> # On worker 0:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>> from torch import nn, Tensor
            >>> from torch.distributed.nn.api.remote_module import RemoteModule
            >>>
            >>> rpc.init_rpc("worker0", rank=0, world_size=2)
            >>> remote_module = RemoteModule(
            >>>     "worker1/cpu", nn.Linear, args=(20, 30),
            >>> )
            >>>
            >>> remote_module1 = rpc.rpc_sync(
            >>>     "worker1/cpu",
            >>>     RemoteModule.init_from_module_rref,
            >>>     ("worker1/cpu", remote_module1.get_module_rref()),
            >>> )
            >>> rpc.shutdown()

            >>> # On worker 1:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>>
            >>> rpc.init_rpc("worker1", rank=1, world_size=2)
            >>> rpc.shutdown()

        Args:
            remote_device (str): Device on the destination worker where we'd like to place this module.
                The device can be a local device or a remote device specified by one of the following remote
                formats:

                    1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
                    2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").

                In addition, the device field can be optional and the default value is "cpu".
            module_rref (RRef[nn.Module]): The module reference shared by both the caller and
                the created remote module.
            _module_interface_cls (type, optional): The TorchScript interface type for the module
                to be created. The type object should be decorated by @torch.jit.interface.
                If not provided, the generated RemoteModule is not torchscript-able.
                Warning, this is an experimental API and susceptible to frequent changes.

        Returns:
            A remote module instance which wraps the :class:`~nn.Module` created by the
            user-provided ``module_rref``, it has a blocking ``forward`` method and an
            asynchronous ``forward_async`` method that returns a future of the ``forward`` call
            on the user-provided module on the remote side.
        """
        # NOTE: if a new attribute is added to this class, also need to add it
        # to ``_REMOTE_MODULE_PICKLED_ATTRIBUTES`` for pickling/unpickling.

        remote_module = object.__new__(RemoteModule)

        enable_moving_cpu_tensors_to_cuda = remote_module._prepare_init(remote_device)

        if _module_interface_cls is not None:
            # Users reply on this field to know if this generated RemoteModule is TorchScript-able.
            remote_module.is_scriptable = True

            remote_module._init_template(
                _module_interface_cls, enable_moving_cpu_tensors_to_cuda
            )
        else:
            remote_module.is_scriptable = False
            remote_module.generated_methods = (
                _NON_SCRIPTABLE_REMOTE_MODULE_MODULE._generated_methods
            )
        remote_module.module_rref = module_rref

        remote_module._install_generated_methods()
        remote_module._check_attribute_picklability()

        return remote_module


class RemoteModule(_RemoteModule):
    """
        A RemoteModule instance can only be created after RPC initialization.

        It creates a user-specified module on a specified remote node.
        It behaves like a regular ``nn.Module`` except that the ``forward`` method is
        executed on the remote node.
        It takes care of autograd recording to ensure the backward pass propagates
        gradients back to the corresponding remote module.

        It generates two methods ``forward_async`` and ``forward`` based on the
        signature of the ``forward`` method of ``module_cls``. ``forward_async``
        runs asynchronously and returns a Future. The arguments of ``forward_async``
        and ``forward`` are the same as the ``forward`` method of the module
        returned by the ``module_cls``.

        For example, if ``module_cls`` returns an instance of ``nn.Linear``,
        that has ``forward`` method signature: ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods with the signatures:

        | ``def forward(input: Tensor) -> Tensor:``
        | ``def forward_async(input: Tensor) -> Future[Tensor]:``

    Args:
        remote_device (str): Device on the destination worker where we'd like to place this module.
            The format should be "<workername>/<device>", where the device field can be parsed as torch.device type.
            E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
            In addition, the device field can be optional and the default value is "cpu".
        module_cls (nn.Module): Class for the module to be created remotely. For example,

            >>> class MyModule(nn.Module):
            >>>     def forward(input):
            >>>         return input + 1
            >>>
            >>> module_cls = MyModule

        args (Sequence, optional): args to be passed to ``module_cls``.
        kwargs (Dict, optional): kwargs to be passed to ``module_cls``.

    Returns:
        A remote module instance which wraps the :class:`~nn.Module` created by the
        user-provided ``module_cls``, it has a blocking ``forward`` method and an
        asynchronous ``forward_async`` method that returns a future of the ``forward`` call
        on the user-provided module on the remote side.

    Example::
        Run the following code in two different processes:

        >>> # xdoctest: +SKIP("distributed")
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> from torch import nn, Tensor
        >>> from torch.distributed.nn.api.remote_module import RemoteModule
        >>>
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> remote_linear_module = RemoteModule(
        >>>     "worker1/cpu", nn.Linear, args=(20, 30),
        >>> )
        >>> input = torch.randn(128, 20)
        >>> ret_fut = remote_linear_module.forward_async(input)
        >>> ret = ret_fut.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>>
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Furthermore, a more practical example that is combined with
        `DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__ (DDP)
        can be found in this `tutorial <https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html>`__.
    """

    def __init__(
        self,
        remote_device: str,
        module_cls: type[nn.Module],
        args: tuple | None = None,
        kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(remote_device, module_cls, args, kwargs)


def _remote_module_receiver(
    *remote_module_pickled_attrs,
):
    """Deserializes a RemoteModule."""
    serialized_remote_module = _SerializedRemoteModule._make(
        remote_module_pickled_attrs
    )
    m = object.__new__(RemoteModule)
    m.__dict__.update(serialized_remote_module._asdict())

    # Unpickling the attribute `module_rref` must invoke RRef's `_deserialize()` method.
    m.module_rref = rpc.PyRRef._deserialize(m.module_rref)

    # Install generated methods when unpickled.
    for method in m.generated_methods:
        method_name = method.__name__
        method = torch.jit.export(method)
        setattr(m, method_name, types.MethodType(method, m))

    return m


def _remote_module_reducer(remote_module):
    """Serialize a RemoteModule."""
    pickled_attrs = {}
    for k, v in remote_module.__dict__.items():
        # Pickling the attribute `module_rref` must invoke RRef's `_serialize()` method.
        if k == "module_rref":
            pickled_attrs[k] = v._serialize()
        elif k in _REMOTE_MODULE_PICKLED_ATTRIBUTES:
            pickled_attrs[k] = v
        # Check if unpickled attributes are all in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING.
        elif k not in _REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING:
            print(
                f"The new attribute ``{k}`` of RemoteModule is ignored during RPC pickling. "
                "To pickle this attribute, please add it to ``_REMOTE_MODULE_PICKLED_ATTRIBUTES``. "
                "Otherwise, please explicitly add it to ``_REMOTE_MODULE_ATTRIBUTES_IGNORE_FOR_PICKLING``.",
                file=sys.stderr,
            )

    return (
        _remote_module_receiver,
        tuple(pickled_attrs.values()),
    )


def _recursive_script_module_receiver(
    recursive_script_module_serialized,
):
    """Deserializes a RecursiveScriptModule that does not contain a script RemoteModule."""
    f = io.BytesIO(recursive_script_module_serialized)
    m = torch.jit.load(f)
    return m


def _recursive_script_module_reducer(recursive_script_module):
    """Serialize a RecursiveScriptModule that does not contain a script RemoteModule, and raises an error otherwise."""
    if hasattr(recursive_script_module._c, "module_rref"):
        raise RuntimeError(
            "Passing a script RemoteModule over RPC is not supported. Please create a RemoteModule in the sender, "
            "send the `module_rref` to the receiver, and create a new instance on the receiver end by passing this `module_rref`."
        )

    f = io.BytesIO()
    torch.jit.save(recursive_script_module, f)
    return (_recursive_script_module_receiver, (f.getvalue(),))


_internal_rpc_pickler._register_reducer(RemoteModule, _remote_module_reducer)
_internal_rpc_pickler._register_reducer(
    torch.jit.RecursiveScriptModule, _recursive_script_module_reducer
)
<|endoftext|># mypy: allow-untyped-defs
import uuid
from collections import OrderedDict
from collections.abc import Callable
from functools import wraps
from typing import Concatenate, Generic, Protocol
from typing_extensions import ParamSpec, TypeVar

import torch
import torch.nn as nn
from torch.distributed._composable_state import _State
from torch.distributed.utils import _get_root_modules


_T = TypeVar("_T", covariant=True)
_P = ParamSpec("_P")


def generate_state_key(string="__composable_api_state_key"):
    return f"{string}_{str(uuid.uuid4())}"


STATE_KEY = generate_state_key()
REGISTRY_KEY = generate_state_key()


# TODO: we can add additional info to RegistryItem to share across APIs. E.g.,
# we can add args and kwargs here, and then we can detect whether fully_shard
# is combined with reentrant activation checkpointing and error out with a clear
# message.
class RegistryItem:
    pass


_TState = TypeVar("_TState", bound="_State", covariant=True)
_M = TypeVar("_M", nn.Module, list[nn.Module])


class _ContractFn(Protocol, Generic[_P, _T, _TState]):
    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _T: ...

    def state(self, module: nn.Module) -> _TState: ...


def contract(
    state_cls: type[_TState] = _State,  # type: ignore[assignment]
) -> Callable[
    [Callable[Concatenate[_M, _P], _M]],
    _ContractFn[Concatenate[_M, _P], _M, _TState],
]:
    r"""
    Decorate a function as a composable distributed API, where the first
    argument of the function must be an :class:`nn.Module` instance or sequence
    of :class:`nn.Module` instances.

    The decorator verifies that the decorated function does not modify
    fully-qualified names (FQNs) for parameters, buffers, or modules. The
    decorated function can return different module instances than the input
    modules; the FQN invariant will be enforced following the input order.

    When a function ``func`` is decorated by ``@contract()``, a
    ``.state(module: nn.Module)`` method will be installed to the decorated
    function. Then you can retrieve and modify the state on a module by calling
    ``func.state(module)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>>
        >>> class MyModel(nn.Module):
        >>>     def __init__(self) -> None:
        >>>         super().__init__()
        >>>         self.l1 = nn.Linear(10, 10)
        >>>         self.l2 = nn.Linear(10, 10)
        >>>
        >>>     def forward(self, x):
        >>>         return self.l2(self.l1(x))
        >>>
        >>> @contract()
        >>> def my_feature(module: nn.Module) -> nn.Module:
        >>>     my_feature.state(module).some_state = "any value"
        >>>     return module
        >>>
        >>> model = MyModel()
        >>> my_feature(model.l1)
        >>> assert my_feature.state(model.l1).some_state == "any value"
        >>> my_feature(model.l2)
        >>> model(torch.randn(2, 10)).sum().backward()
    """

    # wraps will make functions decorated with contract() pickleable - needed for integration with torch.package
    @wraps(state_cls)  # type: ignore[arg-type]
    def inner(
        func: Callable[Concatenate[_M, _P], _M],
    ) -> _ContractFn[Concatenate[_M, _P], _M, _TState]:
        @wraps(func)
        def wrapper(
            module: _M,
            *args: _P.args,
            **kwargs: _P.kwargs,
        ) -> _M:
            inp_module = module
            modules: list[nn.Module]
            if isinstance(module, nn.Module):
                modules = [module]
            else:
                # If the user passes a sequence of modules, then we assume that
                # we only need to insert the state object on the root modules
                # (i.e. those without a parent) among the passed-in modules.
                # pyrefly: ignore [no-matching-overload]
                modules = _get_root_modules(list(module))
            state = state_cls()  # shared across all modules
            registry_item = RegistryItem()  # shared across all modules

            # `func` is allowed to return different module instances than the
            # input modules as long as FQNs are preserved following the input
            # module order
            all_orig_named_params: list[dict[str, nn.Parameter]] = []
            all_orig_named_buffers: list[dict[str, torch.Tensor]] = []
            all_orig_named_modules: list[dict[str, nn.Module]] = []

            for module in modules:
                default_all_state: dict[Callable, _State] = OrderedDict()
                default_registry: dict[str, RegistryItem] = OrderedDict()
                all_state: dict[Callable, _State] = module.__dict__.setdefault(  # type: ignore[call-overload]
                    STATE_KEY, default_all_state
                )
                if not isinstance(all_state, dict):
                    raise AssertionError(
                        f"Distributed composable API states corrupted: {all_state}"
                    )
                registry: dict[str, RegistryItem] = module.__dict__.setdefault(  # type: ignore[call-overload]
                    REGISTRY_KEY, default_registry
                )
                if not isinstance(registry, dict):
                    raise AssertionError(
                        f"Distributed composable API registry corrupted: {registry}"
                    )
                if func in all_state or func.__name__ in registry:
                    raise AssertionError(
                        "Each distinct composable distributed API can only be applied to a "
                        f"module once. {func.__name__} has already been applied to the "
                        f"following module:\n{module}"
                    )
                all_state.setdefault(func, state)
                registry.setdefault(func.__name__, registry_item)

                all_orig_named_params.append(OrderedDict(module.named_parameters()))
                all_orig_named_buffers.append(OrderedDict(module.named_buffers()))
                all_orig_named_modules.append(OrderedDict(module.named_modules()))

            updated = func(inp_module, *args, **kwargs)
            if updated is None:
                updated = inp_module  # type: ignore[assignment]
            updated_modules: list[nn.Module]
            if isinstance(updated, nn.Module):
                updated_modules = [updated]
            else:
                updated_modules = _get_root_modules(list(inp_module))  # type: ignore[arg-type, call-overload]

            all_new_named_params: list[dict[str, nn.Parameter]] = []
            all_new_named_buffers: list[dict[str, torch.Tensor]] = []
            all_new_named_modules: list[dict[str, nn.Module]] = []
            for module in updated_modules:
                all_new_named_params.append(OrderedDict(module.named_parameters()))
                all_new_named_buffers.append(OrderedDict(module.named_buffers()))
                all_new_named_modules.append(OrderedDict(module.named_modules()))

            num_orig_modules = len(all_orig_named_modules)
            num_new_modules = len(all_new_named_modules)
            if num_orig_modules != num_new_modules:
                raise AssertionError(
                    f"{func.__name__} should return the same number of modules as input modules"
                    f"Inputs: {num_orig_modules} modules\n"
                    f"Outputs: {num_new_modules} modules"
                )

            def check_fqn(orig_fqns: list[str], new_fqns: list[str], check_key: str):
                if orig_fqns == new_fqns:
                    return

                orig_fqn_set, new_fqn_set = set(orig_fqns), set(new_fqns)
                orig_only = orig_fqn_set - new_fqn_set
                new_only = new_fqn_set - orig_fqn_set
                if len(orig_only) or len(new_only):
                    raise RuntimeError(
                        f"{check_key}"
                        "Composable distributed API implementations cannot modify FQNs.\n"
                        f"FQNs only in original: {orig_only}\n"
                        f"FQNs only in new: {new_only}"
                    )
                else:
                    raise RuntimeError(
                        f"{check_key}"
                        "Composable distributed API implementations cannot modify "
                        "the order of FQNs.\n"
                        f"Original FQNs: {orig_only}\n"
                        f"New FQNs: {new_only}"
                    )

            for orig_named_params, new_named_params in zip(
                all_orig_named_params, all_new_named_params
            ):
                check_fqn(
                    list(orig_named_params.keys()),
                    list(new_named_params.keys()),
                    "Checking parameters: ",
                )
            for orig_named_buffers, new_named_buffers in zip(
                all_orig_named_buffers, all_new_named_buffers
            ):
                check_fqn(
                    list(orig_named_buffers.keys()),
                    list(new_named_buffers.keys()),
                    "Checking buffers: ",
                )
            for orig_named_modules, new_named_modules in zip(
                all_orig_named_modules, all_new_named_modules
            ):
                check_fqn(
                    list(orig_named_modules.keys()),
                    list(new_named_modules.keys()),
                    "Checking modules: ",
                )

            # TODO: verify that installed distributed paradigms are compatible with
            # each other.

            return updated

        def get_state(module: nn.Module) -> _State:
            return module.__dict__.setdefault(  # type: ignore[call-overload]
                STATE_KEY,
                {},  # TODO(@yhcharles): this is a temporary fix, need a better way
            ).get(func)  # type: ignore[call-overload]

        wrapper.state = get_state  # type: ignore[attr-defined]

        return wrapper  # type: ignore[return-value]

    return inner  # type: ignore[return-value]


def _get_registry(module: nn.Module) -> dict[str, RegistryItem] | None:
    r"""
    Get an ``OrderedDict`` of composable APIs that have been applied to the
    ``module``, indexed by the API name. If no API has been applied, then this
    returns ``None``.
    """
    return getattr(module, REGISTRY_KEY, None)
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

