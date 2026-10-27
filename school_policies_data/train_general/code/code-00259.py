        List[torch.Tensor | BlockMask]: The sharded buffers, each corresponding to the local
            shard for the current rank.
    """
    # TODO: these global variables are going to bite us someday.
    # We will have to remove them soon.
    # For the new API, we only support the module wrapper mode.
    global _dispatch_mode
    _dispatch_mode = _DispatchMode.MODULE_WRAPPER
    global _cp_options
    if load_balancer is not None:
        _cp_options.enable_load_balance = True
    else:
        _cp_options.enable_load_balance = False

    if len(buffers) != len(seq_dims):
        raise ValueError(
            "`seq_dims` must have the same number of elements as `buffers`."
        )

    flat_buffers, spec = tree_flatten(buffers)
    flat_seq_dims, _ = tree_flatten(seq_dims)
    if len(flat_buffers) != len(flat_seq_dims):
        raise ValueError("`seq_dims` must have the pytree structure as `buffers`.")

    if isinstance(flat_buffers[0], torch.Tensor):
        device = flat_buffers[0].device
    else:
        device = flat_buffers[0].kv_num_blocks.device
    for buffer in flat_buffers:
        if isinstance(buffer, torch.Tensor):
            if device != buffer.device:
                raise AssertionError("All buffers must be on the same device")
        else:
            if device != buffer.kv_num_blocks.device:
                raise AssertionError("All buffers must be on the same device")

    flat_sharded_buffers = _context_parallel_buffers(
        mesh, flat_buffers, flat_seq_dims, load_balancer
    )

    return tree_unflatten(flat_sharded_buffers, spec)


def _enable_context_parallel_dispatcher() -> None:
    """
    Enable the context parallel dispatcher. This API is experimental and subject to change.
    """
    _enable_cp_dtensor_dispatcher()


def _disable_context_parallel_dispatcher() -> None:
    """
    Disable the context parallel dispatcher. This API is experimental and subject to change.
    """
    _disable_cp_dtensor_dispatcher()


#####################################################
# Current public APIs, but are also subject to change
#####################################################
@contextlib.contextmanager
@torch.no_grad()
def context_parallel(
    mesh: DeviceMesh,
    *,
    buffers: list[torch.Tensor] | None = None,
    buffer_seq_dims: list[int] | None = None,
    no_restore_buffers: set[torch.Tensor] | None = None,
) -> Generator[None, None, None]:
    """

    ``context_parallel`` is an experimental API to enable context
    parallelism (CP). This API performs two actions: 1) patch the SDPA
    (``torch.nn.functional.scaled_dot_product_attention``) with the CP-enabled
    one, 2) shard ``buffers`` along the sequence dimension and each rank will
    preserve the corresponding shard according ``mesh``.

    Args:
        mesh (:class:`DeviceMesh`): the device mesh for the context parallelism.
        buffers (Optional[List[torch.Tensor]]): buffers that the usage depend
            on the sequence dimension. Examples are input batch, labels and
            positional embedding buffers. These buffers must be sharded along
            the sequence dimension to ensure the accuracy. The sharding will
            happen in-place, the buffer's shape will change within the context.
            The buffers will be restored after the context finishes.
            ``no_restore_buffers`` can be used to specify which buffers don't
            need to be restored. Note that ``buffers`` should not contain any
            nn.Parameter.
        buffer_seq_dims (Optional[List[int]]): the sequence dimensions of ``buffers``.
        no_restore_buffers (Optional[Set[torch.Tensor]]): buffers in these set
            won't be restored after the context exits. This set must be a subset
            of ``buffers``. If the buffers won't be used after the context exits,
            these buffers can be put in this list to avoid extra restore time.

    .. warning::
        `torch.distributed.tensor.experimental.context_parallel` is a
        prototype feature in PyTorch. The API is subject to change.
    """
    # For the legacy API, we only support the monkey-patch mode.
    # We will deprecate this API once the new API is widely used.
    global _dispatch_mode
    _dispatch_mode = _DispatchMode.MONKEY_PATCH

    buffers = [] if buffers is None else buffers
    buffer_seq_dims = [] if buffer_seq_dims is None else buffer_seq_dims
    no_restore_buffers = set() if no_restore_buffers is None else no_restore_buffers

    if len(buffers) != len(buffer_seq_dims):
        raise ValueError(
            "`seq_dims` must have the same number of elements as `buffers`."
        )

    for buffer in no_restore_buffers:
        # Cannot use `if not buffer in buffers` which will incur tensor comparison.
        if not any(b is buffer for b in buffers):
            raise ValueError("`no_restore_buffers` must be a subset of `buffers`.")

    original_buffers = [None if b in no_restore_buffers else b.clone() for b in buffers]

    device = buffers[0].device
    seq_length = buffers[0].shape[buffer_seq_dims[0]]
    cp_world_size = mesh.size()

    # If `enable_load_balance` is True, the default Head-tail load balancer
    # (:class:`_HeadTailLoadBalancer`) is used to rearrange the buffers before
    # sharding. Otherwise, we don't do any load-balance rearrange by passing
    # `None` to `_context_parallel_shard()`.
    old_enable_load_balance = _cp_options.enable_load_balance
    load_balancer = _create_default_load_balancer(seq_length, cp_world_size, device)
    _cp_options.enable_load_balance = load_balancer is not None
    shards = _context_parallel_buffers(
        mesh,
        cast(list[torch.Tensor | BlockMask], buffers),
        buffer_seq_dims,
        load_balancer,
    )
    for buffer, shard in zip(buffers, shards):
        if not isinstance(shard, torch.Tensor):
            raise AssertionError("ContextParallel only supports Tensor")
        shard = shard.clone()
        buffer.resize_(shard.shape)
        buffer.copy_(shard)

    _enable_context_parallel_dispatcher_impl(seq_dim=2, mesh=mesh)
    try:
        yield
    finally:
        _disable_context_parallel_dispatcher_impl()
        _cp_options.enable_load_balance = old_enable_load_balance

        for buffer, original_buffer in zip(buffers, original_buffers, strict=True):
            if original_buffer is not None:
                buffer.resize_(original_buffer.shape)
                buffer.copy_(original_buffer)


@torch.no_grad()
def context_parallel_unshard(
    mesh: DeviceMesh,
    buffers: list[torch.Tensor],
    seq_dims: list[int],
    load_balancer: _LoadBalancer | None = None,
) -> list[torch.Tensor]:
    """
    Unshard the tensors (e.g., output) that are sharded due to context parallelism.

    Args:
        mesh (:class:`DeviceMesh`): the device mesh for the context parallelism.
        buffers (List[torch.Tensor]): the buffers to be unsharded.
        seq_dims (List[int]): the sequence dimensions of ``buffers``. This list
            must have the same length as ``buffers``.
        load_balancer (Optional[:class:`_Loadbalancer`]): an optional `_LoadBalancer`
            object. If this argument is `None`, it means the `buffers` were not
            rearranged when being sharded and there's no need to put it back to order
            after unsharding. If this argument is a `_LoadBalancer` object, call
            its `_generate_indices(restore=True)` to generate the restore indices such
            that `unsharded[restore_idx]` is the original buffer.

    Returns:
        List[torch.Tensor]: the unsharded buffers.

    Note:
        For `context_parallel_unshard` we require not-None `load_balancer` object be
        explicitly passed if flex_attention() is to be used and load-balancing is needed.
        This is different from the case of SDPA though we strongly suggest users follow
        the same convention.
    """
    device = buffers[0].device
    cp_world_size = mesh.size()
    seq_length = buffers[0].shape[seq_dims[0]] * cp_world_size

    # If users don't pass in a `load_balancer`:
    # - if `enable_load_balance` is True, we use the default round-robin
    #   load balancer.
    # - if `enable_load_balance` is False, we don't do any load balancing
    #   by passing in `None` as `restore_indices`.
    load_balancer = load_balancer or _create_default_load_balancer(
        seq_length, cp_world_size, device
    )
    restore_indices = (
        load_balancer._generate_indices(restore=True) if load_balancer else None
    )

    if not (restore_indices is None or restore_indices.ndim == 2):
        raise AssertionError(
            "load balance restore index expects shape (1, seq_len) or (B, seq_len) "
            f"but got {restore_indices.shape}."
        )
    unsharded_buffers = []
    for b, dim in zip(buffers, seq_dims):
        b = b.contiguous()
        unsharded_b = _maybe_wait(ft_c.all_gather_single(b, dim, mesh))

        if restore_indices is not None:
            # NOTE: assuming batch dim is 0
            idx_batch_size = restore_indices.size(0)
            data_batch_size = unsharded_b.size(0)
            if idx_batch_size != 1 and idx_batch_size != data_batch_size:
                raise ValueError(
                    "Cannot restore buffer: "
                    f"restore_indices has shape {restore_indices.shape}, "
                    f"but unsharded_b has shape {unsharded_b.shape}."
                )

            for i in range(data_batch_size):
                index = (
                    restore_indices[0]  # identical load-balance in batch
                    if idx_batch_size == 1
                    else restore_indices[i]
                )
                unsharded_b_batch_i = torch.index_select(
                    unsharded_b[i], dim=dim - 1, index=index
                )
                unsharded_b[i] = unsharded_b_batch_i

        unsharded_buffers.append(unsharded_b)

    return unsharded_buffers


def set_rotate_method(rotate_method: str) -> None:
    """
    Context Parallel SDPA requires the rotation of kv shards. Users can call this
    API to specify which rotation method to use. "alltoall" shuffles the kv shards
    using all-to-all collective. While "allgather" gathers the kv shards using
    all-gather collective after the first sub-SDPA computation. If this API has not
    been called, the default rotate method is "allgather".

    Args:
        rotate_method (str): the rotate method to use. Currently only supports
        "allgather" and "alltoall". If a different string other than these two
        is passed in, the function will raise an error.

    Returns:
        None
    """
    logger.info("Note that FlexAttention CP doesn't support alltoall yet.")
    if rotate_method == "allgather":
        _cp_options.rotate_method = _RotateMethod.ALL_GATHER
    elif rotate_method == "alltoall":
        _cp_options.rotate_method = _RotateMethod.ALL_TO_ALL
    else:
        raise NotImplementedError(
            "Context Parallel does not support "
            f"using {rotate_method} for kv shards rotation"
        )
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
# Context Parallel components

from ._attention import (
    _CausalBehavior,
    _context_parallel_shard,
    _ContextParallel,
    _cp_options,
    _disable_context_parallel_dispatcher,
    _enable_context_parallel_dispatcher,
    _is_causal_behavior,
    _RotateMethod,
    context_parallel,
    context_parallel_unshard,
    set_rotate_method,
)
from ._cp_custom_ops import flex_cp_allgather
from ._load_balancer import (
    _HeadTailLoadBalancer,
    _LoadBalancer,
    _PerDocumentHeadTailLoadBalancer,
    _PTRRLoadBalancer,
)


__all__ = [
    # From _attention
    "_CausalBehavior",
    "_context_parallel_shard",
    "_ContextParallel",
    "_cp_options",
    "_disable_context_parallel_dispatcher",
    "_enable_context_parallel_dispatcher",
    "_is_causal_behavior",
    "_RotateMethod",
    "context_parallel",
    "context_parallel_unshard",
    "set_rotate_method",
    # From _cp_custom_ops
    "flex_cp_allgather",
    # From _load_balancer
    "_HeadTailLoadBalancer",
    "_LoadBalancer",
    "_PerDocumentHeadTailLoadBalancer",
    "_PTRRLoadBalancer",
]
<|endoftext|># this file contains the `_LoadBalancer` class and its family of implementation
# for different load-balancing strategies in tensor sharding.
import functools
from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask


# make it private since it's still a prototype
class _LoadBalancer(ABC):
    @abstractmethod
    def _generate_indices(self, restore: bool = False) -> Tensor | None:
        """
        Generate indices for load balancing.
        Args:
            restore (bool):

        Returns:
            The generated indices of shape `(1, seq_len)` if the load-balancing is
            identical within the batch, or `(batch_size, seq_len)` if the load-balancing
            should vary within the batch.

        Warning:
            For Multi-Head Attention, we require the masks over the head dimension are identical
            (i.e. the return value of `_generate_indices()` does not have `heads` dimension).

        Example:
            Here is the causal mask for attention where q_len == kv_len == 8:
                            KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 0, 0, 0, 0, 0]
            Q_index [1, 1, 1, 1, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 0, 0]
                    [1, 1, 1, 1, 1, 1, 1, 0]
                    [1, 1, 1, 1, 1, 1, 1, 1]

            This mask matrix also represents the computation required to compute
            the masked Q @ K^T by:
            - mask[i, j] == 1: the computation of Q[i, :] dot K[j, :] is required
            - mask[i, j] == 0: the computation should be skipped

            Therefore the number of 1s in matrix represents the amount of computation
            required.

            Assume we want to distribute this Q @ K^T computation to 2 devices, then
            the matrix is also distributed as:
                            KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 0, 0, 0, 0, 0]    rank 0
                    [1, 1, 1, 1, 0, 0, 0, 0]
            Q_index ------------------------
                    [1, 1, 1, 1, 1, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 0, 0]    rank 1
                    [1, 1, 1, 1, 1, 1, 1, 0]
                    [1, 1, 1, 1, 1, 1, 1, 1]

            An imbalance of computation is observed on these 2 ranks and this could make
            rank 1 the straggler when performing Context Parallel. In order to balance
            the computation, we need to rearrange the QKV tensors before sharding in such a
            way that the result mask matrix is evenly distributed over devices and each
            rank has the number of 1s as close as possible.

            This method defines the strategy of how to rearrange the QKV tensor for better
            load-balance:
            - when `restore == False`, this method returns an indices tensor `rearrange_idx`
            such that Q[rearrange_idx] is the desired Q tensor after rearranging.
            - when `restore == True`, this method returns an indices tensor `restore_idx`
            such that Q[rearrange_idx][restore_idx] == Q, i.e. restoring the rearranged tensor
            back to the original status before rearranging.

        Sharding contract:
            After rearranging, Context Parallel shards `Q[rearrange_idx]` with
            `Shard(seq_dim)`, which follows `torch.chunk` semantics: the sequence is
            split into `world_size` contiguous, equal-sized chunks and chunk `r` is
            assigned to rank `r`. Implementations must therefore lay out
            `rearrange_idx` so that each such contiguous chunk is well-balanced --
            i.e. group each rank's positions together (rank-major), not interleaved
            by any other axis such as document.
        """


class _HeadTailLoadBalancer(_LoadBalancer):
    def __init__(self, seq_length: int, world_size: int, device: str | torch.device):
        self.seq_length = seq_length
        self.world_size = world_size
        self.device = device

    def _generate_indices(self, restore: bool = False) -> Tensor:
        """
        Generate head-and-tail load balancing indices or restore indices.
        Args:
            restore:
                If True, generate restore indices that map head-and-tail rearranged
                positions back to original positions. If False, generate load
                balance indices that rearrange original positions to head-and-tail pattern.

        Returns:
            The generated indices of shape `(1, seq_len)` because the load-balancing is
            identical within the batch.

        Warning:
            For Multi-Head Attention, we require the masks over the head dimension are identical
            (i.e. the return value of `_generate_indices()` does not have `heads` dimension).

        Example:
            Here is the causal mask for attention where q_len == kv_len == 8:
                            KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 0, 0, 0, 0, 0]
            Q_index [1, 1, 1, 1, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 0, 0]
                    [1, 1, 1, 1, 1, 1, 1, 0]
                    [1, 1, 1, 1, 1, 1, 1, 1]

            Head-tail load-balance strategy rearranges the Q tensor by combining
            Q[0:k] (on seq dim) and Q[-k:] for rank 0, Q[k:2k] and Q[-2k:-k] for
            rank 1, and so on. In python code it looks like:

                k = Q.size(0) // (2 * cp_world_size)
                for rank in range(cp_world_size):
                    reordered_Q[rank * 2 * k : (rank + 1) * 2 * k] = torch.cat(
                        (Q[rank * k : (rank + 1) * k], Q[-(rank + 1) * k : -rank * k])
                    )

            This can also be done by tensor slicing. For the above example, the indices
            tensor for slicing is:
                slice_indices = Tensor([0, 7, 1, 6, 2, 5, 3, 4])

            After reordering QKV using the `slice_indices`, the corresponding mask matrix
            distributing over 2 devices becomes well-balanced:
                            KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 1, 1]
                    [1, 1, 0, 0, 0, 0, 0, 0]    rank 0
                    [1, 1, 1, 1, 1, 1, 1, 0]
            Q_index ------------------------
                    [1, 1, 1, 0, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 1, 0, 0]    rank 1
                    [1, 1, 1, 1, 0, 0, 0, 0]
                    [1, 1, 1, 1, 1, 0, 0, 0]

            To restore the reordering and putting the tensor back, slicing op can do the
            trick with a `restore_indices` such that:
                slice_indices[restore_indices] == Tensor([0, 1, 2, ...])

            In this way, `reordered_Q[restore_indices]` will just be the original Q.
        """
        seq_length = self.seq_length
        world_size = self.world_size
        if seq_length % (world_size * 2) != 0:
            raise AssertionError
        chunk_size = seq_length // (world_size * 2)

        # Split sequence into 2*world_size chunks, then pair chunk r with
        # chunk (2*world_size - 1 - r) for each rank.
        indices = torch.arange(seq_length, dtype=torch.int, device=self.device)
        chunks = indices.view(world_size * 2, chunk_size)
        head_idx = torch.arange(world_size, device=self.device)
        tail_idx = 2 * world_size - 1 - head_idx
        paired = torch.stack([chunks[head_idx], chunks[tail_idx]], dim=1)
        all_indices_tensor = paired.reshape(-1)

        if restore:
            all_indices_tensor = torch.argsort(all_indices_tensor)

        return all_indices_tensor.unsqueeze(0)  # add batch dim


class _PerDocumentHeadTailLoadBalancer(_LoadBalancer):
    def __init__(
        self,
        seq_length_per_doc: list[list[int]],
        world_size: int,
        device: str | torch.device,
    ):
        """
        `seq_length_per_doc` has size (B, seq_len) if the load-balancing should vary
        within the batch. Otherwise `seq_length_per_doc` should have size (1, seq_len).
        """
        self.seq_length_per_doc = seq_length_per_doc
        self.world_size = world_size
        self.device = device

    def _generate_indices(self, restore: bool = False) -> Tensor:
        """
        Generate the per-document head-and-tail rearrange indices so that after rearranging
        the input is load-balanced in per-document head-and-tail style.

        Args:
            restore:
                If True, generate restore indices that map per-document head-and-tail
                rearranged positions back to original positions. If False, generate load
                balance indices that rearrange original positions to per-document
                head-and-tail pattern.

        Returns:
            The generated indices of shape `(batch_size, seq_len)` if the load-balancing
            should vary within the batch. Otherwise, it should have shape `(1, seq_len)`.

        Warning:
            For Multi-Head Attention, we require the masks over the head dimension are identical
            (i.e. `seq_length_per_doc` must have size (B, seq_len) or (1, seq_len)).

        Example:
            Here is the document causal mask for attention where q_len == kv_len == 16:
                                        KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            Q_index [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

            The per-document head-and-tail load-balancer will apply head-and-tail
            reordering within each document. After load-balancing for context-parallel
            on 2 devices, the above mask matrix will look like this:
                                        KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            Q_index [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
                    ------------------------------------------------
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
        """
        return torch.stack(
            [
                self._generate_indices_for_batch(seq_lengths, restore)
                for seq_lengths in self.seq_length_per_doc
            ]
        )

    def _generate_indices_for_batch(self, seq_length_per_doc, restore) -> Tensor:  # type: ignore[no-untyped-def]
        world_size = self.world_size
        device = self.device
        if not all(
            seq_length % (2 * world_size) == 0 for seq_length in seq_length_per_doc
        ):
            raise AssertionError

        # For each document, split it into 2 * world_size equal chunks and pair
        # chunk r with chunk (2 * world_size - 1 - r), so row r holds rank r's
        # head+tail chunks (the same strategy as _HeadTailLoadBalancer). Stacking
        # the per-document rows rank-wise (dim=1) and flattening lays the indices
        # out rank-major, so a contiguous per-rank shard gives each rank a
        # head+tail slice of every document. A document-major layout would instead
        # let the cut fall mid-document, imbalancing work for mixed-length docs.
        head_idx = torch.arange(world_size, device=device)
        tail_idx = 2 * world_size - 1 - head_idx
        per_doc_rank_chunks = []
        document_start_idx = 0
        for seq_length in seq_length_per_doc:
            chunk_length = seq_length // (2 * world_size)
            chunks = torch.arange(
                document_start_idx,
                document_start_idx + seq_length,
                device=device,
            ).view(2 * world_size, chunk_length)
            # (world_size, 2, chunk_length) -> (world_size, 2 * chunk_length):
            # row r is rank r's head chunk followed by its tail chunk.
            paired = torch.stack([chunks[head_idx], chunks[tail_idx]], dim=1)
            per_doc_rank_chunks.append(paired.reshape(world_size, -1))
            document_start_idx += seq_length

        # (world_size, seq_len / world_size) flattened rank-major.
        indices_tensor = torch.cat(per_doc_rank_chunks, dim=1).reshape(-1)
        if restore:
            indices_tensor = torch.argsort(indices_tensor)

        return indices_tensor


class _PTRRLoadBalancer(_LoadBalancer):
    """
    Processing-Time based Round-Robin (PTRR) load balancer. This load balancer should
    only be used for flex_attention() since it leverages `BlockMask`.
    """

    def __init__(
        self,
        block_mask: BlockMask,
        world_size: int,
    ):
        """
        `block_mask` must have shape (B, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len).
        """
        self.block_mask = block_mask
        self.world_size = world_size

    @staticmethod
    def ptrr_scheduling(process_time: Tensor, group_size: int) -> Tensor:
        """
        Separate the tasks into `group_size` groups using PTRR scheduling.
        process_time:
            1D tensor of size n, where n is the number of tasks. The value
            is the process time of the task. Size `n` must be divisible by
            `group_size`.
        group_size:
            the number of groups

        Returns:
        tasks_in_group (list[list[int]]):
            A collection of list[int] and each list should have size `n // group_size`
            (`group_size` lists in total). Each element is an index in the input
            `process_time` (i.e. [0, len(process_time) - 1]).

        Example:
            process_time = [9, 14, 2, 20, 10, 15, 8, 14, 16, 19, 15, 3, 12, 1, 12, 10]
            tasks_in_group = [
                [3, 12, 13, 14],    # values = [1, 12, 12, 20], sum = 45
                [2, 4, 7, 9],       # values = [2, 10, 14, 19], sum = 45
                [1, 8, 11, 15],     # values = [14, 16, 3, 10], sum = 43
                [0, 5, 6, 10]       # values = [9, 15, 8, 15], sum = 47
            ]
        """
        if process_time.ndim != 1:
            raise AssertionError

        num_tasks = process_time.size(0)

        if num_tasks % group_size != 0:
            raise NotImplementedError(
                f"num_tasks {num_tasks} must be divisible by group_size {group_size}"
            )

        device = process_time.device
        _, sorted_indices_descending = torch.sort(
            process_time, descending=True, stable=True
        )  # if process time is tied, the order is preserved
        sorted_indices_descending_reversed = torch.flip(
            sorted_indices_descending.view(-1, group_size), dims=[1]
        ).view(-1)
        tasks_in_group = torch.where(
            torch.arange(num_tasks, device=device) // group_size % 2 == 0,
            sorted_indices_descending,
            sorted_indices_descending_reversed,
        )
        tasks_in_group = tasks_in_group.view(-1, group_size).transpose(
            0, 1
        )  # (group_size, n // group_size)

        # sort each group. This step should not have impact on correctness
        # nor execution run time, but it helps users visualize the mask
        tasks_in_group, _ = torch.sort(tasks_in_group, dim=1)
        return tasks_in_group

    def _generate_indices(self, restore: bool = False) -> Tensor:
        """
        Generate the PTRR reorder indices of shape `(1, seq_len)` or `(batch_size, seq_len)`.

        Args:
            restore:
                If True, generate restore indices that map Processing-Time based Round-Robin
                (PTRR) rearranged positions back to original positions. If False, generate
                load balance indices that rearrange original positions to PTRR pattern.

            Returns:
                The generated indices of shape `(1, seq_len)` if the load-balancing is
                identical within the batch (i.e. `BlockMask.shape[0] == 1`), or
                `(batch_size, seq_len)` if the load-balancing should vary within the batch.

        Warning:
            For Multi-Head Attention, we require the masks over the head dimension are identical
            (i.e. `self.block_mask` must have shape (B, 1, seq_len, seq_len) or (1, 1, seq_len, seq_len)).

        Example:
            Here is the document causal mask for attention where q_len == kv_len == 16 * BLOCK_SIZE
            (each entry is a block):
                                        KV_index
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 1
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 2
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 3
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 4
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 1
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 2
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 3
            Q_index [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 4
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  -> row value = 5
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  -> row value = 6
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  -> row value = 7
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  -> row value = 8
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  -> row value = 1
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  -> row value = 2
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  -> row value = 3
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  -> row value = 4

            The reorder indices will be: [2, 3, 5, 6, 8, 11, 12, 13, 0, 1, 4, 7, 9, 10, 14, 15] and
            the mask matrix will look like:
                                        KV_index
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 3
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 4
                    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 2
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 3
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  -> row value = 5  rank 0 (sum=28)
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  -> row value = 8
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  -> row value = 1
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]  -> row value = 2
                    ------------------------------------------------
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 1
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 2
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 1
                    [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  -> row value = 4
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  -> row value = 6  rank 1 (sum=28)
                    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  -> row value = 7
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  -> row value = 3
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  -> row value = 4
        """
        block_mask = self.block_mask
        kv_num_blocks = block_mask.kv_num_blocks
        full_kv_num_blocks = block_mask.full_kv_num_blocks
        non_sparse_kv_num_blocks = (
            kv_num_blocks + full_kv_num_blocks
            if full_kv_num_blocks is not None
            else kv_num_blocks
        )
        B, H, Q = non_sparse_kv_num_blocks.shape
        # requirement: the masking is identical across heads (i.e. H == 1 in BlockMask)
        non_sparse_kv_num_blocks = non_sparse_kv_num_blocks.view(-1, Q)  # (B, Q_BLK)

        batch_ptrr = torch.vmap(
            functools.partial(
                _PTRRLoadBalancer.ptrr_scheduling,
                group_size=self.world_size,
            )
        )
        ptrr_indices = batch_ptrr(
            non_sparse_kv_num_blocks
        )  # (B, group_size, num_blks_in_group)
        ptrr_indices = ptrr_indices.reshape(B, -1)  # (B, num_blocks)

        # NOTE: only support the case where the qkv block size are equal
        q_blk_size, kv_blk_size = block_mask.BLOCK_SIZE
        if q_blk_size != kv_blk_size:
            raise AssertionError("for now only support q_blk_size == kv_blk_size")

        indices = torch.arange(
            q_blk_size * ptrr_indices.size(1), device=ptrr_indices.device
        ).view(-1, q_blk_size)  # (NUM_BLOCKS, BLOCK_SIZE)
        indices = indices[ptrr_indices].view(B, -1)  # (B, qkv_size)

        if restore:
            # pyrefly: ignore[missing-argument]
            indices = torch.vmap(torch.argsort)(indices)

        return indices


def _create_default_load_balancer(
    seq_length: int, world_size: int, device: str | torch.device
) -> _LoadBalancer | None:
    from ._attention import _cp_options

    if _cp_options.enable_load_balance and seq_length % (world_size * 2) == 0:
        return _HeadTailLoadBalancer(seq_length, world_size, device)
    else:
        return None
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates
"""
Context Parallelism sharding rules for scaled_dot_product attention operators.

The sharding rules for CP cannot be embedded by default because Shard(2) is not
a valid sharding for SDPA without CP enabled. This module provides utilities to
dynamically install Shard(2) sharding rules when CP is activated.
"""

from contextlib import contextmanager

import torch
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.distributed.tensor._op_schema import ArgsType, KwargsType, RuntimeSchemaInfo
from torch.distributed.tensor._ops.single_dim_strategy import (
    _ShardingPlaceholder,
    register_single_dim_strategy,
)
from torch.distributed.tensor.debug import (
    _clear_fast_path_sharding_prop_cache,
    _clear_python_sharding_prop_cache,
)
from torch.distributed.tensor.placement_types import Placement, Replicate


aten = torch.ops.aten

SEQ_DIM = 2
SingleDimPlacementList = list[Placement | _ShardingPlaceholder | None]


@contextmanager
def _single_dim_strategy_context(op_overload, strategy_func, schema_info=None):
    """Temporarily install a single-dim strategy for Context Parallelism."""
    from torch.distributed.tensor import DTensor

    propagator = DTensor._op_dispatcher.sharding_propagator
    _origin_single_dim_strategy = None
    _origin_op_strategy_schema = None
    try:
        if op_overload in propagator.op_single_dim_strategy_funcs:
            _origin_single_dim_strategy = propagator.op_single_dim_strategy_funcs[
                op_overload
            ]
        if op_overload in propagator.op_to_schema_info:
            _origin_op_strategy_schema = propagator.op_to_schema_info[op_overload]

        register_single_dim_strategy(op_overload, schema_info=schema_info)(
            strategy_func
        )
        yield (_origin_single_dim_strategy, _origin_op_strategy_schema)
    finally:
        if _origin_single_dim_strategy is None:
            if op_overload in propagator.op_single_dim_strategy_funcs:
                del propagator.op_single_dim_strategy_funcs[op_overload]
        else:
            propagator.op_single_dim_strategy_funcs[op_overload] = (
                _origin_single_dim_strategy
            )

        if _origin_op_strategy_schema is None:
            if op_overload in propagator.op_to_schema_info:
                del propagator.op_to_schema_info[op_overload]
        else:
            propagator.op_to_schema_info[op_overload] = _origin_op_strategy_schema

        # Ideally, we should clear the cache, but it is too expensive.
        # _clear_python_sharding_prop_cache()
        # _clear_fast_path_sharding_prop_cache()


def _cp_sharding() -> _ShardingPlaceholder:
    return _ShardingPlaceholder(SEQ_DIM)


# ==================== Flash Attention Strategies ====================


def _scaled_dot_product_flash_attention_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    return_debug_mask = len(args_schema) >= 6 and args_schema[5]
    debug_attn_mask_sharding: Placement | _ShardingPlaceholder = (
        _cp_sharding() if return_debug_mask else Replicate()
    )

    return [
        [
            _cp_sharding(),  # output
            _cp_sharding(),  # logsumexp
            None,  # cum_seq_q
            None,  # cum_seq_k
            None,  # max_q
            None,  # max_k
            Replicate(),  # rng_state
            None,  # unused
            debug_attn_mask_sharding,
            _cp_sharding(),  # q
            _cp_sharding(),  # k
            _cp_sharding(),  # v
        ]
    ]


def _scaled_dot_product_flash_attention_backward_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    num_tensor_inputs = sum(isinstance(arg, TensorMeta) for arg in args_schema)
    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(),  # grad_q
        _cp_sharding(),  # grad_k
        _cp_sharding(),  # grad_v
        _cp_sharding(),  # grad_output
        _cp_sharding(),  # q
        _cp_sharding(),  # k
        _cp_sharding(),  # v
        _cp_sharding(),  # output
        _cp_sharding(),  # logsumexp
    ]
    cp_strategy.extend([Replicate()] * (num_tensor_inputs - 6))
    return [cp_strategy]


# ==================== Efficient Attention Strategies ====================


def _scaled_dot_product_efficient_attention_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    has_attn_bias = args_schema[3] is not None

    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(),  # output
        _cp_sharding(),  # logsumexp
        None,  # philox_seed
        None,  # philox_offset
        _cp_sharding(),  # q
        _cp_sharding(),  # k
        _cp_sharding(),  # v
    ]
    if has_attn_bias:
        cp_strategy.append(Replicate())
    return [cp_strategy]


def _scaled_dot_product_efficient_attention_backward_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    has_attn_bias = args_schema[4] is not None

    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(),  # grad_q
        _cp_sharding(),  # grad_k
        _cp_sharding(),  # grad_v
        _ShardingPlaceholder(1) if has_attn_bias else None,  # grad_bias
        _cp_sharding(),  # grad_output
        _cp_sharding(),  # q
        _cp_sharding(),  # k
        _cp_sharding(),  # v
        _cp_sharding(),  # output
        _cp_sharding(),  # logsumexp
    ]
    if has_attn_bias:
        cp_strategy.insert(8, _ShardingPlaceholder(1))  # attn_bias
    cp_strategy.extend([Replicate(), Replicate()])
    return [cp_strategy]


# ==================== cuDNN Attention Strategies ====================


def _scaled_dot_product_cudnn_attention_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    attn_bias_meta = args_schema[3]
    compute_log_sumexp = args_schema[4]
    return_debug_mask = len(args_schema) >= 8 and args_schema[7]
    has_attn_bias = attn_bias_meta is not None

    logsumexp_sharding: Placement | _ShardingPlaceholder = (
        _cp_sharding() if compute_log_sumexp else Replicate()
    )
    debug_attn_mask_sharding = _cp_sharding() if return_debug_mask else None

    cp_strategy: SingleDimPlacementList = [
        _cp_sharding(),  # output
        logsumexp_sharding,
        None,  # cum_seq_q
        None,  # cum_seq_k
        None,  # max_q
        None,  # max_k
        None,  # philox_seed
        None,  # philox_offset
        debug_attn_mask_sharding,
        _cp_sharding(),  # q
        _cp_sharding(),  # k
        _cp_sharding(),  # v
    ]
    if has_attn_bias:
        cp_strategy.append(Replicate())
    return [cp_strategy]


def _scaled_dot_product_cudnn_attention_backward_cp_single_dim_strategy(
    op: torch._ops.OpOverload,
    args_schema: ArgsType,
    kwargs_schema: KwargsType,
) -> list[SingleDimPlacementList]:
    if len(args_schema) < 15:
        raise AssertionError(f"Expected at least 15 args, got {len(args_schema)}")

    for arg_index in range(6):
        arg = args_schema[arg_index]
        if not isinstance(arg, TensorMeta):
            raise AssertionError(f"Expected TensorMeta, got {type(arg)}")

    philox_placements: list[Placement] = []
    for arg_index in (6, 7):
        arg = args_schema[arg_index]
        if isinstance(arg, TensorMeta):
            philox_placements.append(Replicate())
        elif not isinstance(arg, torch.Tensor):
            raise AssertionError(f"Expected TensorMeta or Tensor, got {type(arg)}")

    has_attn_bias = args_schema[8] is not None
    if has_attn_bias and not isinstance(args_schema[8], (TensorMeta, torch.Tensor)):
        raise AssertionError(
            f"Expected TensorMeta or Tensor, got {type(args_schema[8])}"
        )

    cum_seq_placements: list[None] = []
    for arg_index in (9, 10):
        arg = args_schema[arg_index]
        if isinstance(arg, TensorMeta):
            cum_seq_placements.append(None)
        elif arg is None or isinstance(arg, torch.Tensor):
            pass
        else:
            raise AssertionError(f"Expected TensorMeta or Tensor, got {type(arg)}")

    cp_sharding: SingleDimPlacementList = [
        _cp_sharding(),  # grad_q
        _cp_sharding(),  # grad_k
        _cp_sharding(),  # grad_v
        _cp_sharding(),  # grad_out
        _cp_sharding(),  # q
        _cp_sharding(),  # k
        _cp_sharding(),  # v
        _cp_sharding(),  # output
        _cp_sharding(),  # logsumexp
    ]
    cp_sharding.extend(philox_placements)
    if has_attn_bias and isinstance(args_schema[8], TensorMeta):
        cp_sharding.append(_cp_sharding())
    cp_sharding.extend(cum_seq_placements)

    return [cp_sharding]


# Store context managers and original strategies
_cp_strategy_contexts = {}
_original_strategies = {}


def register_cp_sharding_rules():
    """Register Context Parallelism sharding rules for all scaled_dot_product ops."""
    global _cp_strategy_contexts, _original_strategies

    # If already registered, don't register again
    if _cp_strategy_contexts:
        return

    # Define ops and their corresponding CP strategy functions
    cp_strategies = [
        (
            aten._scaled_dot_product_flash_attention.default,
            _scaled_dot_product_flash_attention_cp_single_dim_strategy,
            RuntimeSchemaInfo(5),
        ),
        (
            aten._scaled_dot_product_flash_attention_backward.default,
            _scaled_dot_product_flash_attention_backward_cp_single_dim_strategy,
            None,
        ),
        (
            aten._scaled_dot_product_efficient_attention.default,
            _scaled_dot_product_efficient_attention_cp_single_dim_strategy,
            RuntimeSchemaInfo(4),
        ),
        (
            aten._scaled_dot_product_efficient_attention_backward.default,
            _scaled_dot_product_efficient_attention_backward_cp_single_dim_strategy,
            None,
        ),
        (
            aten._scaled_dot_product_cudnn_attention.default,
            _scaled_dot_product_cudnn_attention_cp_single_dim_strategy,
            RuntimeSchemaInfo(4),
        ),
        (
            aten._scaled_dot_product_cudnn_attention_backward.default,
            _scaled_dot_product_cudnn_attention_backward_cp_single_dim_strategy,
            None,
        ),
    ]

    # Register each strategy
    for op_overload, strategy_func, schema_info in cp_strategies:
        ctx = _single_dim_strategy_context(op_overload, strategy_func, schema_info)
        orig_funcs, orig_schema = ctx.__enter__()
        _cp_strategy_contexts[op_overload] = ctx
        _original_strategies[op_overload] = (orig_funcs, orig_schema)


def unregister_cp_sharding_rules(clear_the_cache=False):
    """Unregister Context Parallelism sharding rules and restore original strategies."""
    global _cp_strategy_contexts, _original_strategies

    # Exit all context managers
    for ctx in _cp_strategy_contexts.values():
        ctx.__exit__(None, None, None)

    if clear_the_cache:
        _clear_fast_path_sharding_prop_cache()
        _clear_python_sharding_prop_cache()

    _cp_strategy_contexts = {}
    _original_strategies = {}
<|endoftext|>from typing import Any

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d


@torch.library.custom_op("cplib::flex_cp_allgather", mutates_args=())
def flex_cp_allgather(
    k: torch.Tensor, v: torch.Tensor, seq_dim: int, pg_name: c10d.GroupName
) -> tuple[torch.Tensor, torch.Tensor]:
    k = k.contiguous()
    v = v.contiguous()
    k = funcol.all_gather_single(k, seq_dim, pg_name)
    v = funcol.all_gather_single(v, seq_dim, pg_name)
    if isinstance(k, funcol.AsyncCollectiveTensor):
        k = k.wait()
    if isinstance(v, funcol.AsyncCollectiveTensor):
        v = v.wait()
    return k, v


@flex_cp_allgather.register_fake
def _(
    k: torch.Tensor, v: torch.Tensor, seq_dim: int, pg_name: c10d.GroupName
) -> tuple[torch.Tensor, torch.Tensor]:
    shape_k = list(k.shape)
    shape_v = list(v.shape)
    shape_k[seq_dim] *= c10d._get_group_size_by_name(pg_name)
    shape_v[seq_dim] *= c10d._get_group_size_by_name(pg_name)
    new_k = torch.empty(shape_k, dtype=k.dtype, device=k.device)
    new_v = torch.empty(shape_v, dtype=v.dtype, device=v.device)
    return new_k, new_v


@torch.library.custom_op("cplib::flex_cp_allgather_backward", mutates_args=())
def flex_cp_allgather_backward(
    grad_full_k: torch.Tensor,
    grad_full_v: torch.Tensor,
    seq_dim: int,
    pg_name: c10d.GroupName,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_k = funcol.reduce_scatter_single(grad_full_k, "sum", seq_dim, pg_name)
    if isinstance(grad_k, funcol.AsyncCollectiveTensor):
        grad_k = grad_k.wait()
    grad_v = funcol.reduce_scatter_single(grad_full_v, "sum", seq_dim, pg_name)
    if isinstance(grad_v, funcol.AsyncCollectiveTensor):
        grad_v = grad_v.wait()

    return grad_k, grad_v


@flex_cp_allgather_backward.register_fake
def _(
    grad_full_k: torch.Tensor,
    grad_full_v: torch.Tensor,
    seq_dim: int,
    pg_name: c10d.GroupName,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape_k = list(grad_full_k.shape)
    shape_v = list(grad_full_v.shape)
    shape_k[seq_dim] //= c10d._get_group_size_by_name(pg_name)
    shape_v[seq_dim] //= c10d._get_group_size_by_name(pg_name)
    new_grad_k = torch.empty(
        shape_k, dtype=grad_full_k.dtype, device=grad_full_k.device
    )
    new_grad_v = torch.empty(
        shape_v, dtype=grad_full_v.dtype, device=grad_full_v.device
    )
    return new_grad_k, new_grad_v


def _flex_cp_allgather_backward(
    ctx: Any, grad_full_k: torch.Tensor, grad_full_v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None, None]:
    grad_k, grad_v = flex_cp_allgather_backward(
        grad_full_k, grad_full_v, ctx.seq_dim, ctx.pg_name
    )
    return grad_k, grad_v, None, None


def _flex_cp_setup_context(ctx: Any, inputs: Any, output: Any) -> None:
    _, _, ctx.seq_dim, ctx.pg_name = inputs


flex_cp_allgather.register_autograd(
    _flex_cp_allgather_backward, setup_context=_flex_cp_setup_context
)
<|endoftext|># mypy: allow-untyped-defs
import importlib.util

import numpy as np

from torch._prims_common import ShapeType
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset


__all__ = ["visualize_sharding"]

Color = tuple[float, float, float]


def _create_table(
    shards: list[tuple[tuple[int, int], tuple[int, int], int]], device_kind: str = ""
):
    """
    Creates a tabulate table given row and column ranges with device name
    """
    from tabulate import tabulate

    # Extract unique row and column ranges
    row_ranges = sorted({block[0] for block in shards})
    col_ranges = sorted({block[1] for block in shards})

    # Create a matrix initialized with empty strings
    matrix = [["" for _ in col_ranges] for _ in row_ranges]

    # Fill the matrix with values
    for block in shards:
        row_index = row_ranges.index(block[0])
        col_index = col_ranges.index(block[1])
        if matrix[row_index][col_index] == "":
            matrix[row_index][col_index] = device_kind + ":" + str(block[2])
        else:
            matrix[row_index][col_index] += "," + str(block[2])

    # Prepare headers
    row_headers = [f"Row {r[0]}-{r[1]}" for r in row_ranges]
    col_headers = [f"Col {c[0]}-{c[1]}" for c in col_ranges]

    return tabulate(matrix, headers=col_headers, showindex=row_headers)


def make_color_iter(color_map, num_rows, num_cols):
    num_colors = num_rows * num_cols
    for idx in range(num_colors):
        yield color_map(idx)


def _canonicalize_color(color: Color) -> str:
    if isinstance(color, str):
        return color
    r, g, b = (int(a * 255) for a in color)
    return f"#{r:02X}{g:02X}{b:02X}"


def _get_text_color(color: str) -> str:
    r, g, b = map(lambda x: int(x, 16), (color[1:3], color[3:5], color[5:7]))  # noqa: C417
    if (r * 0.299 + g * 0.587 + b * 0.114) > 186:
        return "#000000"
    return "#ffffff"


def _create_rich_table(
    shape: ShapeType,
    shards: list[tuple[tuple[int, int], tuple[int, int], int]],
    device_kind: str = "",
    scale: float = 1.0,
    min_width: int = 9,
    max_width: int = 80,
):
    import matplotlib
    import rich.align
    import rich.box
    import rich.console
    import rich.padding
    import rich.style
    import rich.table

    dtensor_height = shape[0]
    dtensor_width = shape[1] if len(shape) == 2 else 1

    row_ranges = sorted({s[0] for s in shards})
    col_ranges = sorted({s[1] for s in shards})
    num_rows, num_cols = len(row_ranges), len(col_ranges)

    console = rich.console.Console(width=max_width)
    use_color = console.color_system
    color_iter = make_color_iter(matplotlib.colormaps["tab20b"], num_rows, num_cols)

    base_height = int(10 * scale)
    aspect_ratio = (shape[1] if len(shape) == 2 else 1) / shape[0]
    base_width = int(base_height * aspect_ratio)
    height_to_width_ratio = 2.5

    table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None,
    )
    for row in range(num_rows):
        table_row = []
        for col in range(num_cols):
            entry = (
                device_kind
                + ":"
                + ",".join(
                    [
                        str(device_id)
                        for row_range, col_range, device_id in shards
                        if row_range == row_ranges[row] and col_range == col_ranges[col]
                    ]
                )
            )
            width = (col_ranges[col][1] - col_ranges[col][0]) / dtensor_width
            width = int(width * base_width * height_to_width_ratio)
            height = (row_ranges[row][1] - row_ranges[row][0]) / dtensor_height
            height = int(height * base_height)
            left_padding, remainder = divmod(width - len(entry) - 2, 2)
            right_padding = left_padding + remainder
            top_padding, remainder = divmod(height - 2, 2)
            bottom_padding = top_padding + remainder
            if use_color:
                color = _canonicalize_color(next(color_iter)[:3])
                text_color = _get_text_color(color)
                top_padding += 1
                bottom_padding += 1
                left_padding += 1
                right_padding += 1
            else:
                color = None
                text_color = None
            padding = (
                max(top_padding, 0),
                max(right_padding, 0),
                max(bottom_padding, 0),
                max(left_padding, 0),
            )
            table_row.append(
                rich.padding.Padding(
                    rich.align.Align(entry, "center", vertical="middle"),
                    padding,
                    style=rich.style.Style(bgcolor=color, color=text_color),
                )
            )
        table.add_row(*table_row)
    console.print(table, end="\n\n")


def visualize_sharding(dtensor, header="", use_rich: bool = False):
    """
    Visualizes sharding in the terminal for :class:`DTensor` that are 1D or 2D.

    .. note:: This requires the ``tabulate`` package, or ``rich`` and ``matplotlib``.
              No sharding info will be printed for empty tensors
    """
    if dtensor.numel() == 0:  # Do not print empty dtensors.
        return

    if len(dtensor.shape) >= 3:
        raise RuntimeError("visualize sharding supports only 1D or 2D DTensor")

    if dtensor.device_mesh.get_coordinate() is None:  # current rank is not in the mesh
        return

    # Only display the visualization once for each DTensor, on the rank whose
    # coordinate is 0 on all dimensions. For example, if the mesh is a full mesh,
    # we will only print on rank 0.
    local_rank_zero_on_all_dim = all(
        dtensor.device_mesh.get_local_rank(mesh_dim=dim) == 0
        for dim in range(dtensor.device_mesh.ndim)
    )
    if not local_rank_zero_on_all_dim:
        return

    device_coords = {
        int(device_index.item()): list(coord)
        for coord, device_index in np.ndenumerate(
            np.array(dtensor.device_mesh.mesh.tolist())
        )
    }

    device_shard_shape_and_offsets = {
        device_index: _compute_local_shape_and_global_offset(
            dtensor.shape,
            dtensor.device_mesh.shape,
            lambda i: device_coords[device_index][i],
            dtensor.placements,
        )
        for device_index in device_coords
    }

    # Extend shards in a 1D tensor to 2D
    device_shard_shape_and_offsets = {
        device_index: (
            shape if len(shape) == 2 else (shape[0], 1),
            offset if len(offset) == 2 else (offset[0], 0),
        )
        for device_index, (shape, offset) in device_shard_shape_and_offsets.items()
    }

    shards = [
        (
            (offset[0], offset[0] + shape[0] - 1),
            (offset[1], offset[1] + shape[1] - 1),
            device_index,
        )
        for device_index, (shape, offset) in device_shard_shape_and_offsets.items()
    ]

    if (
        importlib.util.find_spec("rich")
        and importlib.util.find_spec("matplotlib")
        and use_rich
    ):
        _create_rich_table(
            dtensor.shape, shards, device_kind=dtensor.device_mesh.device_type
        )
    elif importlib.util.find_spec("tabulate"):
        print(_create_table(shards, device_kind=dtensor.device_mesh.device_type))
    else:
        raise ValueError("`visualize_sharding` requires either `rich` or `tabulate`.")
<|endoftext|># mypy: allow-untyped-defs
import copy
import json
import re
import weakref
from collections import defaultdict
from typing import Any

import torch
import torch.nn
from torch._guards import detect_fake_mode
from torch.autograd.graph import register_multi_grad_hook
from torch.distributed._tools.mod_tracker import ModTracker
from torch.distributed.tensor._api import DTensor
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
    register_module_full_backward_pre_hook,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


__all__ = ["CommDebugMode"]

funcol_native = torch.ops._c10d_functional
funcol_py = torch.ops.c10d_functional
funcol_autograd = torch.ops._c10d_functional_autograd
c10d_ops = torch.ops.c10d

NATIVE_TO_PY_MAPPING = {
    funcol_native.all_gather_into_tensor: funcol_py.all_gather_into_tensor,
    funcol_native.all_gather_into_tensor_coalesced: funcol_py.all_gather_into_tensor_coalesced,
    funcol_native.all_reduce: funcol_py.all_reduce,
    funcol_native.all_reduce_coalesced: funcol_py.all_reduce_coalesced,
    funcol_native.all_to_all_single: funcol_py.all_to_all_single,
    funcol_native.broadcast: funcol_py.broadcast,
    funcol_native.reduce_scatter_tensor: funcol_py.reduce_scatter_tensor,
    funcol_native.reduce_scatter_tensor_coalesced: funcol_py.reduce_scatter_tensor_coalesced,
    # functional ops
    funcol_autograd.all_to_all_single: funcol_py.all_to_all_single,
}

c10d_collective_ops = {
    c10d_ops._allgather_base_,
    c10d_ops._reduce_scatter_base_,
    c10d_ops.allgather_,
    c10d_ops.allgather_coalesced_,
    c10d_ops.allgather_into_tensor_coalesced_,
    c10d_ops.allreduce_,
    c10d_ops.allreduce_coalesced_,
    c10d_ops.alltoall_,
    c10d_ops.alltoall_base_,
    c10d_ops.broadcast_,
    c10d_ops.gather_,
    c10d_ops.scatter_,
    c10d_ops.reduce_,
    c10d_ops.reduce_scatter_,
    c10d_ops.reduce_scatter_tensor_coalesced_,
}

trivial_ops = {
    "aten.detach.default",
    "aten.t.default",
    "aten.view.default",
    "aten._to_copy.default",
    "aten.as_strided.default",
    "aten.transpose.int",
}


class _CommModeModuleTracker(ModTracker):
    """
    Inherits ModuleTracker and expands on its functionality to track the
    parameters and sharding information of a model at a module-level
    """

    def __init__(self):
        super().__init__()
        self.module_helper_dict = {}
        self.module_parameters_dict = {}
        self.module_parents_dict = {}
        self.register_forward_hook_handles = {}
        self.parent_dict = {}
        self.parent_list = []
        self.sharding_dict = {}
        self.activation_checkpointing = False
        self.name = ""

    def _fw_set_module_hook(self, mod, input, output):
        """
        Updates the current module after module finishes running and
        all other hooks are resolved
        """

        if self.is_bw:
            self.activation_checkpointing = True
        else:
            self.activation_checkpointing = False

        if not self.activation_checkpointing:
            # module is no longer parent of next modules
            self.parent_list.pop()

            # set current module to previous parent module
            self.name = self.parent_list[-1]

    def _fw_pre_hook(self, mod, input):
        """
        This function is called before the forward pass of a module. It
        collects the parameters and sharding information of a module and
        stores it in a dictionary.
        """
        if self.is_bw:
            self.activation_checkpointing = True
        else:
            self.activation_checkpointing = False

        self.name = super()._get_mod_name(mod)
        w_mod = weakref.ref(mod)

        # adds current sub-module to module tracker parent class
        super()._get_append_fn(w_mod, self.name, False)()

        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if not self.is_bw and tensors:
            register_multi_grad_hook(
                tensors, super()._get_pop_fn(w_mod, self.name, True)
            )

        if not self.activation_checkpointing:
            # contains information about module ordering and depth in the module tree
            if self.name not in self.module_helper_dict:
                self.module_helper_dict[self.name] = {}

            self.module_helper_dict[self.name]["module_type"] = (
                str(type(mod)).replace("<", "").replace(">", "")
            )
            self.module_helper_dict[self.name]["depth"] = len(self.parents) - 1

            for param_name, param in mod.named_parameters(recurse=False):
                if self.name not in self.module_parameters_dict:
                    self.module_parameters_dict[self.name] = {}

                self.module_parameters_dict[self.name][param_name] = param.data

                if isinstance(param.data, DTensor):
                    key_name = self.name + "." + param_name
                    self.sharding_dict[key_name] = param.data.placements

                    if "parameters" not in self.module_helper_dict[self.name]:
                        self.module_helper_dict[self.name]["parameters"] = {}

                    self.module_helper_dict[self.name]["parameters"][param_name] = str(
                        param.data.placements
                    )

            # used to store module's parents to ensure correctness in backward pass/checkpointing
            if self.name not in self.module_parents_dict:
                self.module_parents_dict[self.name] = copy.deepcopy(self.parents)

            # used to create parent-child module associations for json dumps
            parent = self.parent_list[-1]
            if parent not in self.parent_dict:
                self.parent_dict[parent] = []

            self.parent_dict[parent].append(self.name)
            self.parent_list.append(self.name)

            if self.name in self.register_forward_hook_handles:
                self.register_forward_hook_handles[self.name].remove()
            self.register_forward_hook_handles[self.name] = mod.register_forward_hook(
                self._fw_set_module_hook
            )

    def _fw_post_hook(self, mod, input, output):  # pylint: disable=useless-parent-delegation
        """
        This function is called when the forward pass of a module is called.
        It updates the module tracker and removes the module from parent data
        """

        super()._fw_post_hook(mod, input, output)

    def _bw_hook(self, mod, output):
        """
        This function is called when the backward pass of a module is called. It
        updates the current module for backward passes
        """
        self.activation_checkpointing = False
        self.name = super()._get_mod_name(mod)

    def __enter__(self):
        self.activation_checkpointing = False
        self.module_parameters_dict.clear()
        self.sharding_dict.clear()
        self.parent_dict.clear()
        self.parent_list = ["Global"]
        self.module_helper_dict.clear()
        self.module_helper_dict["Global"] = {"depth": 0}
        self.module_parents_dict.clear()
        self.module_parents_dict["Global"] = set()
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(self._fw_post_hook)
        self.register_forward_hook_handles.clear()
        self._bw_handle = register_module_full_backward_pre_hook(self._bw_hook)
        self.name = "Global"

    def __exit__(self, *args):
        super().__exit__(*args)
        self._bw_handle.remove()

        # removes all forward_hook handles added in the pre-hook
        for handle in self.register_forward_hook_handles.values():
            handle.remove()

    def print_paramater_info(self):
        print(self.module_parameters_dict)

    def print_sharding_info(self):
        for key, value in self.sharding_dict.items():
            print(key + ": " + str(value))


class CommDebugMode(TorchDispatchMode):
    """
    :class:`CommDebugMode` is a context manager that counts the number of
    functional collectives within its context. It does this using a
    ``TorchDispatchMode``.

    .. note:: Not all collectives are supported yet.

    Example usage

    .. code-block:: python

        mod = ...
        comm_mode = CommDebugMode()
        with comm_mode:
            mod.sum().backward()
        print(comm_mode.get_comm_counts())
    """

    def __init__(self):
        super().__init__()
        self.supports_higher_order_operators = True
        self.comm_counts: dict[Any, int] = defaultdict(int)
        self.comm_module_counts = {}
        self.comm_module_operation_counts = {}
        self.comm_registry = set()
        for native_op, py_op in NATIVE_TO_PY_MAPPING.items():
            self.comm_registry.add(native_op)
            self.comm_registry.add(py_op)

        self.comm_registry.add(torch.ops._dtensor.shard_dim_alltoall)
        self.advanced_module_tracker = _CommModeModuleTracker()

    def generate_json_dump(self, file_name="comm_mode_log.json", noise_level=3):
        """
        Creates json file used to build browser visual
        0. prints module-level collective counts
        1. prints dTensor operations not included in trivial operations
        2. prints operations not included in trivial operations
        3. prints all operations
        """

        (
            include_DTensor_ops,
            include_module_data,
            include_ops,
            include_trivial_ops,
        ) = self._set_noise_parameters(noise_level)

        # recursively builds json data
        def add_json_information(json_dict, fqn):
            json_dict["fqn"] = fqn
            json_dict["module_type"] = ""
            json_dict["parameters"] = []
            json_dict["children"] = []
            json_dict["collectives_forward"] = []
            json_dict["collectives_backward"] = []
            json_dict["operations_forward"] = []
            json_dict["operations_backward"] = []

            # adds module layer type and parameters, and their sharding
            if (
                "module_type" in self.advanced_module_tracker.module_helper_dict[fqn]
                and include_module_data
            ):
                json_dict["module_type"] = (
                    self.advanced_module_tracker.module_helper_dict[fqn]["module_type"]
                )

                if "parameters" in self.advanced_module_tracker.module_helper_dict[fqn]:
                    for (
                        param_name,
                        placement,
                    ) in self.advanced_module_tracker.module_helper_dict[fqn][
                        "parameters"
                    ].items():
                        json_dict["parameters"].append((param_name, placement))

            # adds module collective information
            if fqn in self.comm_module_counts:
                for collective, count in self.comm_module_counts[fqn][
                    "forward"
                ].items():
                    json_dict["collectives_forward"].append((str(collective), count))

                for collective, count in self.comm_module_counts[fqn][
                    "backward"
                ].items():
                    json_dict["collectives_backward"].append((str(collective), count))

            # adds module operation information
            forward_operations = []
            backward_operations = []
            checkpointing_operations = []

            # only get operations if the minimum operation noise level is set to true
            if include_DTensor_ops:
                if fqn in self.comm_module_operation_counts:
                    (
                        forward_operations,
                        backward_operations,
                        checkpointing_operations,
                    ) = self._get_operations_list(
                        self.comm_module_operation_counts[fqn]
                    )

            # remove all operations who don't have DTensor inputs
            if not include_ops:
                forward_operations = [
                    op for op in forward_operations if len(op["input_sharding"])
                ]
                backward_operations = [
                    op for op in backward_operations if len(op["input_sharding"])
                ]
                checkpointing_operations = [
                    op for op in checkpointing_operations if len(op["input_sharding"])
                ]

            # remove all operations in trivial operations set
            if not include_trivial_ops:
                forward_operations = [
                    op
                    for op in forward_operations
                    if str(op["name"]) not in trivial_ops
                ]
                backward_operations = [
                    op
                    for op in backward_operations
                    if str(op["name"]) not in trivial_ops
                ]
                checkpointing_operations = [
                    op
                    for op in checkpointing_operations
                    if str(op["name"]) not in trivial_ops
                ]

            # converts operation information into string format for json.dumps()
            forward_operations = copy.deepcopy(forward_operations)
            for op in forward_operations:
                op["name"] = str(op["name"])

                for i in range(len(op["input_sharding"])):
                    op["input_sharding"][i] = str(op["input_sharding"][i])
                    op["input_shape"][i] = str(op["input_shape"][i])

            backward_operations = copy.deepcopy(backward_operations)
            for op in backward_operations:
                op["name"] = str(op["name"])

                for i in range(len(op["input_sharding"])):
                    op["input_sharding"][i] = str(op["input_sharding"][i])
                    op["input_shape"][i] = str(op["input_shape"][i])

            checkpointing_operations = copy.deepcopy(checkpointing_operations)
            for op in checkpointing_operations:
                op["name"] = str(op["name"])

                for i in range(len(op["input_sharding"])):
                    op["input_sharding"][i] = str(op["input_sharding"][i])
                    op["input_shape"][i] = str(op["input_shape"][i])

            json_dict["operations_forward"] = forward_operations
            json_dict["operations_backward"] = backward_operations
            json_dict["operations_checkpointing"] = checkpointing_operations

            if fqn not in self.advanced_module_tracker.parent_dict:
                return json_dict

            # recursively adds module's children
            for ele in self.advanced_module_tracker.parent_dict[fqn]:
                json_dict["children"].append(add_json_information({}, ele))

            return json_dict

        json_dict: dict[str, Any] = {}
        add_json_information(json_dict, "Global")

        # converts dictionary into json file
        with open(file_name, "w") as json_file:
            json.dump(json_dict, json_file, indent=4)

    def generate_comm_debug_tracing_table(self, noise_level=3):
        """
        Generates detailed table displaying operations and collective tracing information
        on a module level. Amount of information is dependent on noise_level

        0. prints module-level collective counts
        1. prints dTensor operations not included in trivial operations, module information
        2. prints operations not included in trivial operations
        3. prints all operations
        """

        (
            include_DTensor_ops,
            include_module_data,
            include_ops,
            include_trivial_ops,
        ) = self._set_noise_parameters(noise_level)

        table = ""
        for fqn in self.advanced_module_tracker.module_helper_dict:
            # setting up indentations for table formatting
            indent = "  " * (
                2 * self.advanced_module_tracker.module_helper_dict[fqn]["depth"]
            )
            table += f"{indent}{fqn}\n"

            if include_module_data:
                if (
                    "module_type"
                    in self.advanced_module_tracker.module_helper_dict[fqn]
                ):
                    module_type = self.advanced_module_tracker.module_helper_dict[fqn][
                        "module_type"
                    ]
                    table += f"{indent}*module type: {module_type}\n"

                if "parameters" in self.advanced_module_tracker.module_helper_dict[fqn]:
                    table += f"{indent}*Parameter List\n"
                    for (
                        param_name,
                        placement,
                    ) in self.advanced_module_tracker.module_helper_dict[fqn][
                        "parameters"
                    ].items():
                        table += f"{indent} *{param_name}: {placement}\n"

            indent += "  "
            collective_indent = "  " * (
                2 * self.advanced_module_tracker.module_helper_dict[fqn]["depth"] + 2
            )
            operation_indent = "  " * (
                2 * self.advanced_module_tracker.module_helper_dict[fqn]["depth"] + 3
            )

            # separate the module's collective and operations by forward and backward
            forward_collectives = {}
            backward_collectives = {}
            if fqn in self.comm_module_counts:
                forward_collectives = self.comm_module_counts[fqn]["forward"]
                backward_collectives = self.comm_module_counts[fqn]["backward"]

            forward_operations = []
            backward_operations = []
            checkpointing_operations = []

            if include_DTensor_ops:
                if fqn in self.comm_module_operation_counts:
                    (
                        forward_operations,
                        backward_operations,
                        checkpointing_operations,
                    ) = self._get_operations_list(
                        self.comm_module_operation_counts[fqn]
                    )

            def add_tracing_information(table, collectives_dict, operation_list):
                """
                adds tracing information for module's forward or backward
                """
                for collective, count in collectives_dict.items():
                    table += (
                        f"\033[1;33m{collective_indent}*{collective}: {count}\033[0m\n"
                    )

                def add_operations(
                    table, operation, collective_indent, operation_indent
                ):
                    """
                    adds operation information to the table
                    """
                    table += f"\033[1;33m{collective_indent}**{operation_name}\033[0m\n"

                    if len(operation["input_shape"]):
                        operation_shape = operation["input_shape"]
                        operation_sharding = operation["input_sharding"]
                        operation_device_mesh = operation["device_mesh"]

                        table += f"\033[1;31m{operation_indent}shape: {operation_shape}\033[0m\n"
                        table += f"\033[1;31m{operation_indent}sharding: {operation_sharding}\033[0m\n"
                        table += f"\033[1;31m{operation_indent}device mesh: {operation_device_mesh}\033[0m\n"

                    return table

                for operation in operation_list:
                    operation_name = str(operation["name"])

                    # include all operations
                    if include_trivial_ops:
                        table = add_operations(
                            table, operation, collective_indent, operation_indent
                        )

                    # include all operations not in trivial operations
                    elif include_ops and operation_name not in trivial_ops:
                        table = add_operations(
                            table, operation, collective_indent, operation_indent
                        )

                    # only include dTensor operations not in trivial set
                    elif (
                        include_DTensor_ops
                        and (operation_name not in trivial_ops)
                        and len(operation["input_shape"])
                    ):
                        table = add_operations(
                            table, operation, collective_indent, operation_indent
                        )

                return table

            if len(forward_collectives) or len(forward_operations):
                table += f"{indent}FORWARD PASS\n"
                table = add_tracing_information(
                    table, forward_collectives, forward_operations
                )

            if len(backward_collectives) or len(backward_operations):
                table += f"{indent}BACKWARD PASS\n"
                table = add_tracing_information(
                    table, backward_collectives, backward_operations
                )

            if len(checkpointing_operations):
                table += f"{indent}ACTIVATION CHECKPOINTING\n"
                table = add_tracing_information(table, {}, checkpointing_operations)

        return table

    def _get_operations_list(self, module_operation_counts):
        forward_operations = [
            op for op in module_operation_counts["operations_list"] if not op["is_bw"]
        ]
        backward_operations = [
            op
            for op in module_operation_counts["operations_list"]
            if op["is_bw"] and not op["is_activation_checkpointing"]
        ]
        checkpointing_operations = [
            op
            for op in module_operation_counts["operations_list"]
            if op["is_activation_checkpointing"]
        ]

        return forward_operations, backward_operations, checkpointing_operations

    def get_total_counts(self) -> int:
        return sum(self.comm_counts.values())

    def get_comm_counts(self) -> dict[Any, int]:
        """Returns the communication counts as a dictionary.

        Returns:
            Dict[Any, int]: The communication counts as a dictionary.
        """
        return self.comm_counts

    def get_parameter_info(self) -> dict[str, dict[str, Any]]:
        return self.advanced_module_tracker.module_parameters_dict

    def get_sharding_info(self) -> dict[str, dict[str, Any]]:
        return self.advanced_module_tracker.sharding_dict

    def __enter__(self):
        self.comm_counts.clear()
        self.comm_module_counts.clear()
        self.comm_module_counts["Global"] = {}
        self.comm_module_counts["Global"]["forward"] = defaultdict(int)
        self.comm_module_counts["Global"]["backward"] = defaultdict(int)

        self.comm_module_operation_counts.clear()

        super().__enter__()
        self.advanced_module_tracker.__enter__()
        return self

    # pyrefly: ignore [bad-override]
    def __exit__(self, *args):
        self.advanced_module_tracker.__exit__()
        super().__exit__(*args)

    def log_comm_debug_tracing_table_to_file(
        self, file_name="comm_mode_log.txt", noise_level=3
    ):
        """
        Alternative to console CommDebugMode output, writes to file specified by the user
        """
        ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
        table = ansi_escape.sub("", self.generate_comm_debug_tracing_table(noise_level))

        with open(file_name, "w") as log_file:
            log_file.write(table)

    def _set_noise_parameters(self, noise_level):
        """
        sets variables controlling what information displays based on noise level
        """
        include_DTensor_ops = False
        include_module_data = False
        include_ops = False
        include_trivial_ops = False

        if noise_level > 0:
            include_DTensor_ops = True
            include_module_data = True

        if noise_level > 1:
            include_ops = True

        if noise_level > 2:
            include_trivial_ops = True

        return (
            include_DTensor_ops,
            include_module_data,
            include_ops,
            include_trivial_ops,
        )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # When running this mode with DTensor, ordinarily all modes will
        # run **before** subclasses get a chance to run.
        # Returning NotImplemented here gives us a chance to let DTensor
        # run and desugar into comms ops, before CommDebugMode sees them.

        # Higher-order operators (e.g. run_dtensor_rng_op) don't have
        # _overloadpacket and aren't collectives — just redispatch.
        if isinstance(func, torch._ops.HigherOrderOperator):
            kwargs = kwargs if kwargs else {}
            return func(*args, **kwargs)

        # sets up operation-level collective count
        if self.advanced_module_tracker.name not in self.comm_module_operation_counts:
            # dictionary should hold module input and output shape, operations list and collective counter
            self.comm_module_operation_counts[self.advanced_module_tracker.name] = {
                "operations_list": []
            }
        operation_dict = {}
        operation_dict["name"] = func

        operation_dict["input_shape"] = []
        operation_dict["input_sharding"] = []
        operation_dict["device_mesh"] = ""

        # tracks if the operation is part of the backward pass
        operation_dict["is_bw"] = self.advanced_module_tracker.is_bw

        # tracks if the operation is part of activation checkpointing
        operation_dict["is_activation_checkpointing"] = (
            self.advanced_module_tracker.activation_checkpointing
        )

        if any(t == DTensor for t in types):
            for ele in args:
                if isinstance(ele, DTensor):
                    # saves shapes and placements of all DTensor args
                    operation_dict["input_shape"].append(ele.shape)
                    operation_dict["input_sharding"].append(ele.placements)
                    operation_dict["device_mesh"] = str(ele.device_mesh)

            self.comm_module_operation_counts[self.advanced_module_tracker.name][
                "operations_list"
            ].append(operation_dict)

            return NotImplemented

        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket

        # We have many tests that use CommDebugMode to verify the occurrence of
        # collectives. These tests do so by querying comm_counts with legacy
        # funcol ops as key. For the purpose of native funcol migration, we
        # need these tests to work for both legacy and native funcol. To avoid
        # the need to modify all tests to accommodate the two implementations,
        # we make CommDebugMode translate native funcol ops into legacy funcol
        # ops until the migration finishes.

        if func_packet in self.comm_registry or func_packet in c10d_collective_ops:
            if func_packet in NATIVE_TO_PY_MAPPING:
                func_packet = NATIVE_TO_PY_MAPPING[func_packet]
            self.comm_counts[func_packet] += 1

            key = "forward"
            if self.advanced_module_tracker.is_bw:
                key = "backward"

            # adds collective count to current module
            if self.advanced_module_tracker.name not in self.comm_module_counts:
                self.comm_module_counts[self.advanced_module_tracker.name] = {}
                self.comm_module_counts[self.advanced_module_tracker.name][
                    "forward"
                ] = defaultdict(int)
                self.comm_module_counts[self.advanced_module_tracker.name][
                    "backward"
                ] = defaultdict(int)
            self.comm_module_counts[self.advanced_module_tracker.name][key][
                func_packet
            ] += 1

            # adds collective count to parent modules
            for par in self.advanced_module_tracker.module_parents_dict[
                self.advanced_module_tracker.name
            ]:
                # makes sure we aren't double counting when current sub-module hasn't been removed from parents
                if par != self.advanced_module_tracker.name:
                    if par not in self.comm_module_counts:
                        self.comm_module_counts[par] = {}
                        self.comm_module_counts[par]["forward"] = defaultdict(int)
                        self.comm_module_counts[par]["backward"] = defaultdict(int)
                    self.comm_module_counts[par][key][func_packet] += 1

        # if tensor op uses fake tensors, return
        if detect_fake_mode(args):
            return out

        # add tensor operation to module operation list
        self.comm_module_operation_counts[self.advanced_module_tracker.name][
            "operations_list"
        ].append(operation_dict)

        return out

    def __repr__(self):
        return f"CommDebugMode(get_total_counts()={self.get_total_counts()})"
<|endoftext|># mypy: allow-untyped-defs
import torch._C
from torch.distributed.tensor.debug._comm_mode import CommDebugMode
from torch.distributed.tensor.debug._visualize_sharding import visualize_sharding


__all__ = ["CommDebugMode", "visualize_sharding"]


def _get_python_sharding_prop_cache_info():
    """
    Get the cache info for the Python sharding propagation cache, used for debugging purpose only.
    This would return a named tuple showing hits, misses, maxsize and cursize of the sharding
    propagator cache. Note that directly calling into the sharding propagator does not share cache
    state with the DTensor dispatch fast path!
    """
    from torch.distributed.tensor._api import DTensor

    return (
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_info()  # type:ignore[attr-defined]
    )


def _get_fast_path_sharding_prop_cache_stats():
    """
    Get a tuple (hits, misses) for the fast path sharding propagation cache, used for debugging
    only.
    """
    return torch._C._get_DTensor_sharding_propagator_cache_stats()


def _clear_python_sharding_prop_cache():
    """
    Clears the cache for the Python sharding propagation cache, used for debugging purpose only.
    """
    from torch.distributed.tensor._api import DTensor

    return (
        DTensor._op_dispatcher.sharding_propagator.propagate_op_sharding.cache_clear()  # type:ignore[attr-defined]
    )


def _clear_fast_path_sharding_prop_cache():
    """
    Clears the cache for the fast path sharding propagation cache, used for debugging purpose only.
    """
    torch._C._clear_DTensor_sharding_propagator_cache()


def _clear_sharding_prop_cache():
    """
    Clears both the Python and fast path sharding propagation caches, used for debugging purpose only.
    This is the recommended way to clear all sharding propagation caches.
    """
    _clear_python_sharding_prop_cache()
    _clear_fast_path_sharding_prop_cache()


def _reinit_dispatch_logger():
    """
    Resets the cached DTensor dispatch logger state so that the next DTensor
    dispatch re-checks whether debug logging is enabled. Call this after
    changing the log level on the ``torch.distributed.tensor._dispatch`` logger.
    """
    torch._C._reinit_DTensor_dispatch_logger()


# Set namespace for exposed private names
CommDebugMode.__module__ = "torch.distributed.tensor.debug"
visualize_sharding.__module__ = "torch.distributed.tensor.debug"
<|endoftext|># mypy: allow-untyped-defs
from operator import itemgetter

import torch
import torch.fx
import torch.nn as nn
from functorch.compile import make_boxed_func
from torch._functorch.compilers import aot_module
from torch._inductor.decomposition import select_decomp_table
from torch.distributed.tensor import DTensor


inductor_decomps = select_decomp_table()

graphs: list[torch.fx.GraphModule] = []


def fwd_bwd_compiler(fx_g, _):
    graphs.append(fx_g)
    return make_boxed_func(fx_g)


def get_inductor_decomp_graphs(model: nn.Module, args, kwargs):
    """
    Obtain forward and backward graphs of a model with inductor decompositions using tracing and aot_module.

    Convenient util to get the fwd and bwd graphs of an arbitrary model
    with inductor decompositions. Note that this would simply do tracing
    with aot_module and don't ensure correctness. This is useful to track
    the ops needed in DTensor.
    """
    compiled_mod = aot_module(
        model, fw_compiler=fwd_bwd_compiler, decompositions=inductor_decomps
    )
    output = compiled_mod(*args, **kwargs)

    if output.ndim != 0:
        # if output is not a scalar tensor, by default sum it in order to
        # run backward
        output = output.sum()

    output.backward()

    # one fwd, one bwd graph
    if len(graphs) != 2:
        raise AssertionError
    return graphs


def print_op_coverage_summary(model: nn.Module, args, kwargs, *, output_csv=False):
    """
    Util to print the operator coverage summary of a certain model with tabulate.

    Must have tabulate module installed.
    """
    # python module required for summary
    import csv

    from tabulate import tabulate

    fwd_graph, bwd_graph = get_inductor_decomp_graphs(model, args, kwargs)

    op_counts = {}

    for node in fwd_graph.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if node.target not in op_counts:
                op_counts[node.target] = 0

            op_counts[node.target] += 1

    for node in bwd_graph.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if node.target not in op_counts:
                op_counts[node.target] = 0

            op_counts[node.target] += 1

    op_infos = []

    for op, count in op_counts.items():
        supported = op in DTensor._op_dispatcher.sharding_propagator.op_to_rules
        op_infos.append([op, str(op._schema), count, supported])

    # sort the op info base on the total count index
    count_idx = 2
    op_infos.sort(key=itemgetter(count_idx), reverse=True)

    headers = ["Operator", "Schema", "Total Count", "Supported"]
    print(tabulate(op_infos, headers=headers))

    if output_csv:
        # Open a CSV file for writing
        with open("op_summary.csv", "w", newline="") as csv_file:
            # Create a CSV writer object
            csv_writer = csv.writer(csv_file)

            csv_writer.writerow(headers)
            # Write each table row to the CSV file
            for row in op_infos:
                csv_writer.writerow(row)
<|endoftext|>import logging
import math
from enum import IntEnum

from torch.distributed._tools.ilp_utils import Graph, is_submodule
from torch.distributed._tools.sac_estimator import SACStats


try:
    from pulp import (  # type: ignore[import-untyped,import-not-found]
        lpDot,
        LpInteger,
        LpMaximize,
        LpMinimize,
        LpProblem,
        LpStatus,
        lpSum,
        LpVariable,
        PULP_CBC_CMD,
        value,
    )
except ImportError as err:
    raise ImportError(
        "Please install pulp package. See: https://github.com/coin-or/pulp."
    ) from err

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level to INFO
logger.setLevel(logging.INFO)


def sac_milp(
    graph: Graph,
    memory_budget: float,
    world_size: int = 1,
    ac_units: list[str] | None = None,
    fsdp_units: list[str] | None = None,
) -> tuple[dict[str, float], float, int]:
    """
    MILP to decide which modules to AC and how much memory to discard.
    The objective is to minimize recomputation time.
    The constraint is to ensure peak memory is under budget.

    Args:
        graph: graph representation of the model as a module submodule tree
            where each node is a submodule with memory & runtime stats
        memory_budget: memory budget in GiB
        world_size: number of GPUs. In the case of FSDP, world_size will be
            used to compute the amount of parameter and gradient memory on each rank
        ac_units: a list of user-specified AC units.
        fsdp_units: a list of FSDP units. AC units cannot be supermodules of FSDP units.

    Returns:
        Dict[str, float]: the optimal SAC solution, mapping from module fqn to
            the percentage of activation memory to **discard**
        float: the recomputation time of the optimal SAC solution
        int: upper bound on the peak memory of the optimal SAC solution.
            note that value of -1 means that the ILP solver failed to find a solution.

    """
    num_nodes = len(graph.nodes)
    max_runtime = max(
        (graph.nodes[i]["sac_runtime"] for i in range(num_nodes)), default=1
    )
    M = max(10**2, 2 * max_runtime)
    MEM_MULTIPLIER = 2**30

    # Create a MILP problem
    prob = LpProblem("SAC", LpMinimize)

    # Create decision variables
    # y_i: indicator for if module i is AC'ed
    y = LpVariable.matrix("y", list(range(num_nodes)), 0, 1, LpInteger)
    # r_i: percentage of discarded activation memory
    r = LpVariable.matrix("r", list(range(num_nodes)), 0, 1)
    # d_i: discarded activation memory for module i
    d = LpVariable.matrix("d", list(range(num_nodes)), 0)
    # a_i: total activation memory at module i
    a = LpVariable.matrix("a", list(range(num_nodes)), 0)
    # m_i: memory at module i, combining parameters, gradients, and activations
    m = LpVariable.matrix("m", list(range(num_nodes)), 0)
    # rcp_i: percentage of recomputation time
    rcp = LpVariable.matrix("rcp", list(range(num_nodes)), 0)
    # rct_i: recomputation time for module i (in ms)
    rct = LpVariable.matrix("rct", list(range(num_nodes)), 0)
    # max_m: peak memory
    max_m = LpVariable("max_m", 0)

    # Add constraints
    # [Constraint] User specified AC units
    if ac_units:
        ac_units_set = set(ac_units)
        for i in range(num_nodes):
            if graph.nodes[i]["fqn"] not in ac_units_set:
                prob += y[i] == 0

    # [Constraint] AC units cannot be supmodules of user specified FSDP units
    if fsdp_units:
        for i in range(num_nodes):
            if any(
                is_submodule(fsdp_unit, graph.nodes[i]["fqn"])
                for fsdp_unit in fsdp_units
            ):
                prob += y[i] == 0

    # [Constraint] No nested AC units
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if graph.ad_matrix[i][j] == 1:
                prob += y[i] + y[j] <= 1

    # [Constraint] Do not AC leaf modules
    for i in range(num_nodes):
        if graph.nodes[i]["is_leaf"]:
            prob += y[i] == 0

    # [Constraint] Express amount of discarded activation memory
    for i in range(num_nodes):
        # There are two measures for activation memory: ACM and IA
        # 1. IA is the activation memory saved when not using AC
        # 2. ACM is the total activation memory, including those
        #    that are not typically saved when not using AC
        # Note: ACM >= IA
        if (not graph.nodes[i]["is_leaf"]) and graph.nodes[i][
            "sac_memory"
        ] < graph.nodes[i]["act_fw_per_module"]:
            logger.warning("For module {%s}: ", graph.nodes[i]["fqn"])
            logger.warning(
                "activation memory from memory tracker is {%d},",
                graph.nodes[i]["act_fw_per_module"],
            )
            logger.warning(
                "activation memory from SAC estimator is {%d}.",
                graph.nodes[i]["sac_memory"],
            )
            logger.warning("Something is wrong. Please check!")
            logger.warning("Overriding the latter with the former.")
            graph.nodes[i]["sac_memory"] = graph.nodes[i]["act_fw_per_module"]
        ACM_i = graph.nodes[i]["sac_memory"] / MEM_MULTIPLIER
        IA_i = graph.nodes[i]["act_fw_per_module"] / MEM_MULTIPLIER
        prob += d[i] == ACM_i * r[i] - (ACM_i - IA_i) * y[i]

    # [Constraint] Ensure correctness of r_i
    # There are two parts to its correctness
    # 1. r_i > 0 only if y_i == 1 (discard only if it is an AC unit)
    # 2. r_i needs to be large enough to cover the difference between
    #    ACM and IA. Otherwise, we are not saving any memory
    for i in range(num_nodes):
        prob += y[i] >= r[i]
        if graph.nodes[i]["is_leaf"]:
            continue
        ACM_i = graph.nodes[i]["sac_memory"] / MEM_MULTIPLIER
        IA_i = graph.nodes[i]["act_fw_per_module"] / MEM_MULTIPLIER
        prob += r[i] >= (ACM_i - IA_i) / ACM_i * y[i]

    # [Constraint] Express total activation memory in the backward pass
    for i in range(num_nodes):
        AG_i = graph.nodes[i]["act_grad_per_module"] / MEM_MULTIPLIER
        TA_i = graph.nodes[i]["act_total"] / MEM_MULTIPLIER
        # related to discarded amount of memory
        pos = graph.nodes[i]["pos_fw_post_order"]
        coeff = [0] * num_nodes
        for p in range(pos):
            j = graph.name2node[graph.fw_post_order[p]]["index"]
            coeff[j] = 1
        prob += a[i] == TA_i + AG_i - lpDot(coeff, d)

    # [Constraint] Express the total amount of memory at each module
    # Note that unsharded parameters and gradients are not included here
    P_1 = graph.nodes[0]["param_per_module"] / MEM_MULTIPLIER
    for i in range(num_nodes):
        TG_i = graph.nodes[i]["grad_total"] / MEM_MULTIPLIER
        prob += m[i] == a[i] + (P_1 + TG_i) / world_size

    # [Constraint] Express peak memory
    for i in range(num_nodes):
        prob += max_m >= m[i]

    # [Constraint] Express percentage of recomputation time
    for i in range(num_nodes):
        for s in range(graph.nodes[i]["n_segments"]):
            slope = graph.nodes[i]["slopes"][s]
            intercept = graph.nodes[i]["intercepts"][s]
            prob += rcp[i] >= slope * r[i] + intercept

    # [Constraint] Express recomputation time
    # rct_i = (rcp_i * ACT_i) if y_i == 1 else 0
    for i in range(num_nodes):
        ACT_i = graph.nodes[i]["sac_runtime"]
        prob += rct[i] <= M * y[i]
        prob += rct[i] <= ACT_i * rcp[i]
        prob += rct[i] >= ACT_i * rcp[i] - M * (1 - y[i])

    # [Constraint] Peak memory should be below budget
    prob += max_m <= memory_budget

    # Set Objeictive
    prob += lpSum(rct)

    # Solve
    solver = PULP_CBC_CMD(gapRel=0.05, timeLimit=180, msg=0)
    status = prob.solve(solver)

    # If solver fails, print status and return empty solution
    if status != 1:
        logger.error("Solver failed to find a solution: %s", LpStatus[status])
        return {}, 0, -1

    # Gather and return solution if optimal solution is found
    ac_decisions = {}
    for i in range(num_nodes):
        if round(y[i].varValue) == 1:
            ac_decisions[graph.nodes[i]["fqn"]] = round(r[i].varValue, 4)
    recomputation_time = round(value(prob.objective), 2)
    peak_mem = round(max_m.varValue * MEM_MULTIPLIER)

    return ac_decisions, recomputation_time, peak_mem


class SACDecision(IntEnum):
    RECOMPUTE = 0
    SAVE = 1


def get_optimal_checkpointing_policy_per_module(
    sac_stats: SACStats, memory_budget: float
) -> list[int]:
    """
    This is adapted from --
    https://github.com/facebookresearch/xformers/blob/c6c0ac31f1b08542a0bc27278c6ed10f825f6963/xformers/checkpoint.py#L375

    Given the SACStats of a module, including list of operators, their memory, runtimes, and metadata,
    decide via MILP an optimal set of operators to checkpoint under a given ``memory_budget``.

    Args:
        sac_stats: the SACStats object of the module
        memory_budget: a float between zero and one

    Returns:
        List[int]: the decision whether each operator should be saved (1) or recomptued (0).
    """
    if not (0 <= memory_budget <= 1):
        raise ValueError(
            f"`memory_budget` must be a float between 0 and 1. Got {memory_budget}."
        )
    num_ops = len(sac_stats.func_names)

    # Create a MILP problem
    prob = LpProblem("SAC-per-module", LpMaximize)

    # Create decision variables
    # x[i] = 1 means the i-th operator should be saved, otherwise it should be recomputed
    x = LpVariable.matrix("x", list(range(num_ops)), 0, 1, LpInteger)

    # Add constraints
    # [Constraint] random ops should be saved if ``force_store_random`` is True
    # otherwise, random ops should either be all recomputed or all saved
    if sac_stats.force_store_random:
        for i in sac_stats.rand_ops:
            prob += x[i] == SACDecision.SAVE.value
    else:
        for i1, i2 in zip(sac_stats.rand_ops[:-1], sac_stats.rand_ops[1:]):
            prob += x[i1] == x[i2]

    # [Constraint] view-like ops should always be recomputed
    for i in sac_stats.view_like_ops:
        prob += x[i] == SACDecision.RECOMPUTE.value

    # [Constraint] inplace ops should always be done in conjunction with its parent op
    for op, op_parent in sac_stats.inplace_ops:
        if op != op_parent:
            prob += x[op] == x[op_parent]
        else:
            prob += x[op] == SACDecision.SAVE.value

    # [Constraint] saved memory should be under the ``memory_budget``
    max_memory = math.ceil(memory_budget * sum(sac_stats.memory))
    prob += lpDot(x, sac_stats.memory) <= max_memory

    # [Objective] minimize recomputation time, note the ILP is a maximization problem
    # because x[i] == 1 means the op is saved (not recomputed), and thus recomputation
    # time is sum(sac_stats.runtimes) - lpDot(x, sac_stats.runtimes)
    prob += lpDot(x, sac_stats.runtimes)

    # Solve
    solver = PULP_CBC_CMD(gapRel=0.05, timeLimit=10, msg=0)
    status = prob.solve(solver)

    # If solver fails, print status and return empty solution
    if status != 1:
        logger.error("Solver failed to find a solution: %s", LpStatus[status])
        return []

    # Gather and return solution if optimal solution is found
    return [round(x[i].varValue) for i in range(num_ops)]
<|endoftext|># Owner(s): ["module: unknown"]
from collections import defaultdict
from typing import Any, TYPE_CHECKING
from typing_extensions import Self

import torch
import torch.utils._pytree as pytree
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.mod_tracker import ModTracker
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._runtime_estimation import (
    _FLOAT_TYPES,
    _IGNORE_OPS,
    _VIEW_OPS,
    get_compute_time,
    get_transfer_time,
)


if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["RuntimeEstimator"]


class RuntimeEstimator(TorchDispatchMode):
    """
    Estimates the GPU runtime in milliseconds using various estimation methods under the ``FakeTensorMode``.

    This class provides a ``TorchDispatchMode`` based context manager that can be used to estimate the eager
    runtime of PyTorch functions. It supports two estimation modes, benchmarking (`operator-level-benchmark`) and
    roofline cost modeling (`operator-level-cost-model`).
    For modules executed under this context manager, it aggregates the forward and backward operation runtimes
    and also records their execution orders.

    Attributes:
        mod_runtimes (Dict[str, Dict[str, float]]): A dictionary of module runtimes. The key to the outer dictionary
            is the fully qualified name (FQN) of the module. For each module the forward and backward runtimes of the
            operations are aggregated in the inner dictionary keyed by 'fw' and 'bw'.
        mod_fw_pre_order (List[str]): List of module FQNs in pre-forward execution order.
        mod_bw_pre_order (List[str]): List of module FQNs in pre-backward execution order.
        mod_fw_post_order (List[str]): List of module FQNs in post-forward execution order.
        mod_bw_post_order (List[str]): List of module FQNs in post-backward execution order.
        total_runtime (float): The total estimated runtime in milliseconds.

    Note:
        1) The benchmarking estimate mode will execute kernels on GPU and assumes that every operation can run in
            isolation without causing an OOM error. It is also designed to be used only under ``FakeTensorMode``.
        2) Currently wrapper tensor sub-classes such as ``DTensor`` won't produce correct estimates. We plan to support
            them in future PRs.
        3) We only estimate the compute time, if your code has communication, it will not be considered. Again, we will
            support this in future PRs.

    Example usage:

        .. code-block:: python

            runtime_estimator = RuntimeEstimator()
            with FakeTensorMode():
                module = ...
                optimizer = ...
                inp = ...
                with runtime_estimator(estimate_mode_type="operator-level-cost-model"):
                    loss = module(inp)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                runtime_estimator.display_modulewise_stats()
    """

    _no_fallback_kernel: set[torch._ops._OpNamespace] = set()
    fake_mode: FakeTensorMode
    gpu_type: str | None = None

    def __init__(self, gpu_type: str | None = None) -> None:
        """
        Args:
            gpu_type (str | None): Optional datasheet device name (e.g.
                ``"NVIDIA H100"``) to pin the roofline peak FLOPS and DRAM
                bandwidth to instead of querying the current device. This
                makes ``operator-level-cost-model`` estimates deterministic
                and hardware-independent. Only affects the roofline estimator.
        """
        super().__init__()
        self._gpu_type = gpu_type
        self._estimate: Callable
        self._estimate_mode_type: str
        self._mod_tracker = ModTracker()
        self.mod_runtimes: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: 0.0)
        )
        self.mod_fw_pre_order: list[str] = []
        self.mod_bw_pre_order: list[str] = []
        self.mod_fw_post_order: list[str] = []
        self.mod_bw_post_order: list[str] = []
        self.total_runtime: float = 0.0

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_subclasses/fake_tensor.py#L1969
    # NB: returns fake tensors
    @classmethod
    def _maybe_run_and_benchmark_fallback_kernel(  # type: ignore[no-untyped-def]
        cls,
        func,
        args,
        kwargs,
        orig_not_implemented_exception,
    ):
        """
        Runs and benchmarks a fallback kernel for a given function.

        Args:
            func (Callable): The function to benchmark.
            args (Tuple): The arguments to pass to the function.
            kwargs (Dict[str, Any]): The keyword arguments to pass to the function.
            orig_not_implemented_exception (Exception): The original exception to raise if the fallback kernel
                is not implemented.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        # these should all be supported, just to be safe
        # avoid fallback for operators which inplace modify metadata
        # because the input fake tensors would be unmodified
        if torch.Tag.inplace_view in func.tags:  # type: ignore[attr-defined]
            raise orig_not_implemented_exception

        inp_impls = {}
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        # Don't use in_kernel_invocation_manager(fake_mode) as we want to do
        # REAL compute (not with meta device)
        with no_dispatch():

            def to_real_tensor(e):  # type: ignore[no-untyped-def]
                if cls.fake_mode.is_our_fake(e):
                    if e.dtype in _FLOAT_TYPES:
                        out = torch.rand_like(e, device=e.fake_device)
                    else:
                        out = torch.ones_like(e, device=e.fake_device)
                    if e.is_sparse:
                        out._coalesced_(e.is_coalesced())
                    inp_impls[id(out)] = e
                    return out
                return e

            flat_args = [to_real_tensor(a) for a in flat_args]
            args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
            r = func(*args, **kwargs)
            warmup_iters, actual_iters = 2, 3
            for _ in range(warmup_iters):
                func(*args, **kwargs)
            start_event = torch.Event(enable_timing=True)
            end_event = torch.Event(enable_timing=True)
            start_event.record(torch.accelerator.current_stream())
            for _ in range(actual_iters):
                func(*args, **kwargs)
            end_event.record(torch.accelerator.current_stream())
            torch.accelerator.synchronize()
            cuda_time = start_event.elapsed_time(end_event)
            mean_op_time = cuda_time / actual_iters

        storages = set()

        for e in flat_args:
            if isinstance(e, torch.Tensor):
                if not e.is_sparse:
                    storages.add(e._typed_storage()._cdata)

        # TODO: also check metadata change on inputs
        # proper aliasing/metadata relationship between outputs and inputs will
        # not be set up, bc of conversion to device, unless we can reuse an
        # input impl

        def map_out(e):  # type: ignore[no-untyped-def]
            if id(e) not in inp_impls and (
                isinstance(e, torch.Tensor)
                and not e.is_sparse
                and e._typed_storage()._cdata in storages
            ):
                raise orig_not_implemented_exception

            if isinstance(e, torch.Tensor):
                if id(e) in inp_impls:
                    return inp_impls[id(e)]
                else:
                    return cls.fake_mode.fake_tensor_converter.from_real_tensor(
                        cls.fake_mode, e
                    )
            else:
                return e

        return (pytree.tree_map(map_out, r), mean_op_time)

    @classmethod
    def _benchmark_estimate(cls, func, args, kwargs) -> tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using benchmarking.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            res: The result of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        if not isinstance(cls.fake_mode, FakeTensorMode):
            raise AssertionError(
                "Initialize/Assign FakeTensorMode before using this function"
            )
        mean_op_time = 0.0
        if func._overloadpacket not in _VIEW_OPS:
            try:
                res, mean_op_time = cls._maybe_run_and_benchmark_fallback_kernel(
                    func,
                    args,
                    kwargs,
                    NotImplementedError,
                )
                return (res, mean_op_time)
            except NotImplementedError:
                cls._no_fallback_kernel.add(func._overloadpacket)
        res = func(*args, **kwargs or {})
        return (res, mean_op_time)

    # Adapted from: https://github.com/pytorch/pytorch/blob/9b902b3ee3bd608a19543362b66bf06c373dd374/torch/_inductor/scheduler.py#L589
    @classmethod
    def _roofline_estimate(cls, func, args, kwargs) -> tuple[Any, float]:  # type: ignore[no-untyped-def]
        """
        Estimates the runtime of a function using a roofline cost model.

        Args:
            func: The function to estimate.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            out: The output of the function.

        Returns:
            Tuple[Any, float]: A tuple containing the result of the function and
                the mean operation time in milliseconds.
        """
        if not torch.accelerator.is_available():
            raise AssertionError(
                "Roofline estimation needs to access CUDA capabilities to make estimations"
            )

        # Roofline Cost Model Explanation

        # The roofline cost model estimates the execution time of an operator based on
        # the device's empirical maximum FLOPs/sec (pi) and device DRAM bandwidth (beta).

        # Variables:
        # - pi: Maximum empirical FLOPs/sec of the device
        # - beta: Maximum empirical device DRAM bandwidth (bytes/sec) of the device
        # - I: Arithmetic intensity of the operator (FLOPs/bytes)
        # - op_flops: FLOPs required by the operator
        # - op_bytes: Bytes transferred to and from DRAM for the operator

        # Calculation Steps:
        # 1. Calculate arithmetic intensity: I = op_flops / op_bytes
        # 2. Calculate estimated FLOPs/sec: est_flops_sec = min(pi, beta * I)
        # 3. Calculate estimated operator time: estimated_op_time = op_flops / est_flops_sec
        #    This simplifies to: estimated_op_time = max(op_flops / pi, op_flops / (beta * I))
        #    Further simplifying: estimated_op_time = max(op_flops / pi, op_bytes / beta)

        # Simplified Formulas:
        # - compute_time = op_flops / pi
        # - transfer_time = op_bytes / beta
        # - estimated_op_time = max(compute_time, transfer_time)

        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        op_time = 0.0
        func_packet = func._overloadpacket
        if func_packet not in _IGNORE_OPS:
            flat_args_kwargs, args_spec = pytree.tree_flatten((args, kwargs))
            flat_outs, out_spec = pytree.tree_flatten(out)
            transfer_time = get_transfer_time(
                flat_args_kwargs, flat_outs, gpu_type=cls.gpu_type
            )

            out_dtypes = {
                t.dtype
                for t in flat_outs
                if isinstance(t, torch.Tensor) and t.dtype in _FLOAT_TYPES
            }

            args, kwargs = pytree.tree_unflatten(flat_args_kwargs, args_spec)
            out = pytree.tree_unflatten(flat_outs, out_spec)

            compute_time = get_compute_time(
                func_packet, args, kwargs, out, out_dtypes, gpu_type=cls.gpu_type
            )
            # We get the estimated time as the max of the transfer time and
            # compute time. We divide by 1e6 to get the time in ms
            op_time = max(transfer_time, compute_time) / 1e6

        return (out, op_time)

    def display_modulewise_stats(self, depth: int = 2) -> None:
        """
        Displays module-wise statistics collected by ``RuntimeEstimator``.

        Prints the pre-forward and pre-backward execution orders.
        Displays the module-wise forward and backward runtimes in milliseconds.

        Args:
            depth (int): The maximum depth of module hierarchy to display (default to 2).
        """
        print("Pre-Forward Execution Order: ")
        for mod_fqn in self.mod_fw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        print("Pre-Backward Execution Order: ")
        for mod_fqn in self.mod_bw_pre_order:
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(mod_fqn)
        for mod_fqn, runtimes in self.mod_runtimes.items():
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(
                f"{mod_fqn} fw: {runtimes.get('fw', 0.0):.3f}ms bw: {runtimes.get('bw', 0.0):.3f}ms"
            )

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):  # type: ignore[no-untyped-def]
        # TODO: @sanketpurandare: Flatten tensors by desugaring the tensor subclasses
        # TODO: @sanketpurandare: Add logic for incorporating communication time
        res, op_time = self._estimate(func, args, kwargs)
        for par in self._mod_tracker.parents:
            if self._mod_tracker.is_bw:
                self.mod_runtimes[par]["bw"] += op_time
            else:
                self.mod_runtimes[par]["fw"] += op_time
        self.total_runtime += op_time
        return res

    def __call__(self, estimate_mode_type: str) -> Self:
        """
        Sets the estimate mode type.

        Currently supported modes:
            - "operator-level-benchmark": Estimates runtime using operator benchmarking.
            - "operator-level-cost-model": Estimates runtime using roofline cost model.

        Args:
            estimate_mode_type (str): The type of estimate mode to use.

        Returns:
            RuntimeEstimator: The runtime estimator instance.

        Raises:
            NotImplementedError: If the estimate mode type is not supported.
        """
        if estimate_mode_type == "operator-level-benchmark":
            self._estimate = RuntimeEstimator._benchmark_estimate
        elif estimate_mode_type == "operator-level-cost-model":
            self._estimate = RuntimeEstimator._roofline_estimate
        else:
            raise NotImplementedError(
                f"estimate_mode_type {estimate_mode_type} not supported"
            )
        self._estimate_mode_type = estimate_mode_type
        return self

    def __enter__(self) -> Self:
        fake_mode = active_fake_mode()
        if not isinstance(fake_mode, FakeTensorMode):
            raise AssertionError(
                "No FakeTensorMode found, designed to be used under FakeTensorMode"
            )
        RuntimeEstimator.fake_mode = fake_mode
        RuntimeEstimator.gpu_type = self._gpu_type
        self.total_runtime = 0.0
        self.mod_runtimes = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.mod_fw_pre_order.clear()
        self.mod_bw_pre_order.clear()
        self.mod_fw_post_order.clear()
        self.mod_bw_post_order.clear()
        self._mod_tracker.register_user_hooks(
            pre_fw_hook=lambda mod, inp: self.mod_fw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            pre_bw_hook=lambda mod, g_out: self.mod_bw_pre_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            post_fw_hook=lambda mod, inp, out: self.mod_fw_post_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
            post_bw_hook=lambda mod, g_inp: self.mod_bw_post_order.append(
                self._mod_tracker.get_known_fqn(mod)
            ),
        )
        self._mod_tracker.__enter__()
        super().__enter__()
        return self

    # pyrefly: ignore [bad-override]
    def __exit__(self, *args: Any) -> None:
        print(
            f"Estimated ({self._estimate_mode_type})"
            f"total_time: {self.total_runtime:.3f} ms"
        )
        if len(self._no_fallback_kernel) > 0:
            print("no_fallback_kernel: ", list(self._no_fallback_kernel))
        super().__exit__(*args)
        self._mod_tracker.clear_user_hooks()
        self._mod_tracker.__exit__()
<|endoftext|>from .fsdp2_mem_tracker import FSDPMemTracker
from .mem_tracker import MemTracker
from .memory_tracker import MemoryTracker
from .mod_tracker import ModTracker
from .runtime_estimator import RuntimeEstimator
from .sac_estimator import (
    MSPS,
    SACEstimator,
    SACGreedyOrderMeta,
    SACStats,
    SACTradeOffStats,
)
<|endoftext|>from typing import Any

import torch
from torch._C._distributed_c10d import (
    _resolve_process_group,
    FakeWork,
    ProcessGroup,
    Work,
)
from torch.utils._pytree import tree_map_only


c10d = torch.ops.c10d
_c10d_functional = torch.ops._c10d_functional
_c10d_functional_autograd = torch.ops._c10d_functional_autograd
_dtensor = torch.ops._dtensor

# List of collective operation functions including functional collectives
# Note: The following collectives might be deprecated soon hence not adding them
# depcreated_non_functional_collectives = [
#     c10d.allreduce_coalesced_.default,
#     c10d.reduce_scatter_tensor_coalesced_.default,
#     c10d.allgather_into_tensor_coalesced_.default,
#     c10d.allgather_coalesced_.default,
# ]
non_functional_collectives: set[torch._ops.OpOverload] = {
    c10d.broadcast_.default,
    c10d.allreduce_.default,
    c10d.reduce_.default,
    c10d.send.default,
    c10d.recv_.default,
    c10d.recv_any_source_.default,
    c10d.allgather_.default,
    c10d.reduce_scatter_.default,
    c10d._reduce_scatter_base_.default,
    c10d._allgather_base_.default,
    c10d.gather_.default,
    c10d.scatter_.default,
    c10d.alltoall_.default,
    c10d.alltoall_base_.default,
    c10d.barrier.default,
    c10d.monitored_barrier_.default,
}
functional_collectives: set[torch._ops.OpOverload] = {
    _c10d_functional.broadcast.default,
    _c10d_functional.all_reduce.default,
    _c10d_functional.all_gather_into_tensor.default,
    _c10d_functional.reduce_scatter_tensor.default,
    _c10d_functional.reduce_scatter_tensor_out.default,
    _c10d_functional.all_to_all_single.default,
    _c10d_functional_autograd.all_to_all_single.default,
    _c10d_functional.wait_tensor.default,
    _c10d_functional.all_reduce_.default,
    _c10d_functional.all_reduce_coalesced.default,
    _c10d_functional.all_reduce_coalesced_.default,
    _c10d_functional.all_gather_into_tensor_out.default,
    _c10d_functional.all_gather_into_tensor_coalesced.default,
    _c10d_functional_autograd.all_gather_into_tensor.default,
    _c10d_functional.reduce_scatter_tensor_coalesced.default,
    _c10d_functional_autograd.reduce_scatter_tensor.default,
    _c10d_functional.broadcast_.default,
    _c10d_functional.isend.default,
    _c10d_functional.irecv.default,
    _c10d_functional.batch_p2p_ops.default,
    _dtensor.shard_dim_alltoall.default,
}

sync_ops: set[torch._ops.OpOverload] = {
    c10d.barrier.default,
    c10d.monitored_barrier_.default,
    _c10d_functional.wait_tensor.default,
}

collective_ops = set.union(functional_collectives, non_functional_collectives)


class CollectiveOp:
    # Static sets for performance optimization
    PG_ARG_1 = {
        c10d.broadcast_.default,
        c10d.allreduce_.default,
        c10d.reduce_.default,
        c10d.send.default,
        c10d.recv_.default,
        c10d.recv_any_source_.default,
        c10d.barrier.default,
        # c10d.allreduce_coalesced_.default
    }

    PG_ARG_2 = {
        c10d.allgather_.default,
        c10d._allgather_base_.default,
        c10d.reduce_scatter_.default,
        c10d._reduce_scatter_base_.default,
        c10d.gather_.default,
        c10d.scatter_.default,
        c10d.alltoall_.default,
        c10d.alltoall_base_.default,
        # c10d.allgather_coalesced_.default,
        # c10d.allgather_into_tensor_coalesced_.default
        # c10d.reduce_scatter_tensor_coalesced_.default
    }

    PG_ARG_3 = {
        _c10d_functional.broadcast.default,
        _c10d_functional.broadcast_.default,
        _c10d_functional.all_reduce.default,
        _c10d_functional.all_reduce_.default,
        _c10d_functional.all_reduce_coalesced.default,
        _c10d_functional.all_reduce_coalesced_.default,
        _c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor_out.default,
        _c10d_functional_autograd.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor_coalesced.default,
    }

    PG_ARG_4 = {
        _c10d_functional.reduce_scatter_tensor.default,
        _c10d_functional.reduce_scatter_tensor_coalesced.default,
        _c10d_functional_autograd.reduce_scatter_tensor.default,
        _c10d_functional.all_to_all_single.default,
        _c10d_functional_autograd.all_to_all_single.default,
        _c10d_functional.isend.default,
        _c10d_functional.irecv.default,
        _dtensor.shard_dim_alltoall.default,
    }

    PG_ARG_5 = {
        _c10d_functional.batch_p2p_ops.default,
    }

    WK_ARG_1 = {
        c10d.broadcast_.default,
        c10d.allreduce_.default,
        c10d.allgather_.default,
        c10d.reduce_scatter_.default,
        c10d._reduce_scatter_base_.default,
        c10d._allgather_base_.default,
        c10d.scatter_.default,
        c10d.alltoall_.default,
        _c10d_functional.isend.default,
        _c10d_functional.irecv.default,
    }

    WK = {
        c10d.send.default,
        c10d.recv_.default,
        c10d.recv_any_source_.default,
        c10d.reduce_.default,
        c10d.gather_.default,
        c10d.alltoall_base_.default,
        c10d.barrier.default,
    }

    COMM_TENSOR_ARG_0 = {
        c10d.allreduce_.default,
        c10d.send.default,
        c10d.recv_.default,
        c10d.recv_any_source_.default,
        c10d.allgather_.default,
        c10d.gather_.default,
        c10d.reduce_.default,
        c10d.broadcast_.default,
        _c10d_functional.all_reduce_coalesced.default,
        _c10d_functional.all_reduce_coalesced_.default,
        # c10d.allreduce_coalesced_.default
        # c10d.allgather_coalesced_.default
        # c10d.allgather_into_tensor_coalesced_.default,
    }

    COMM_TENSOR_ARG_1 = {
        c10d.reduce_scatter_.default,
        c10d.scatter_.default,
        # c10d.reduce_scatter_tensor_coalesced_.default,
    }

    COMM_TENSOR_ARG_RES = {
        _c10d_functional.all_gather_into_tensor.default,
        _c10d_functional_autograd.all_gather_into_tensor.default,
    }

    COMM_TENSOR_SINGLE_UNTYPED_STORAGE = {
        c10d._allgather_base_.default,
        _c10d_functional.broadcast.default,
        _c10d_functional.broadcast_.default,
        _c10d_functional.all_reduce.default,
        _c10d_functional.all_reduce_.default,
        _c10d_functional.reduce_scatter_tensor.default,
        _c10d_functional_autograd.reduce_scatter_tensor.default,
    }

    COMM_TENSOR_ARG_0_AND_RES = {
        _c10d_functional.all_to_all_single.default,
        _c10d_functional_autograd.all_to_all_single.default,
        _dtensor.shard_dim_alltoall.default,
    }

    COMM_TENSOR_RES_SUM = {
        _c10d_functional.all_gather_into_tensor_coalesced.default,
        _c10d_functional.reduce_scatter_tensor_coalesced.default,
    }

    @staticmethod
    def sum_tensors(arg: Any) -> int:
        """Calculate total memory consumed by the tensors in the argument."""
        total_memory = 0

        def sum_bytes(t: torch.Tensor) -> None:
            nonlocal total_memory
            total_memory += t.untyped_storage().nbytes()

        tree_map_only(torch.Tensor, sum_bytes, arg)
        return total_memory

    @staticmethod
    def get_process_group(func, args) -> ProcessGroup:  # type: ignore[no-untyped-def]
        """Retrieve the process group for collective operations, except `wait_tensor`."""
        if func in CollectiveOp.PG_ARG_1:
            return ProcessGroup.unbox(args[1])
        if func in CollectiveOp.PG_ARG_2:
            return ProcessGroup.unbox(args[2])
        if func in CollectiveOp.PG_ARG_3:
            return _resolve_process_group(args[2])
        if func in CollectiveOp.PG_ARG_4:
            return _resolve_process_group(args[3])
        if func in CollectiveOp.PG_ARG_5:
            return _resolve_process_group(args[4])
        raise TypeError(f"Func {func} not found in {collective_ops}")

    @staticmethod
    def get_comm_tensor_size(func, res, args, kwargs) -> int:  # type: ignore[no-untyped-def]
        """Compute the communication tensor size, except for `wait_tensor`, `barrier`, and `monitored_barrier`."""
        if func in CollectiveOp.COMM_TENSOR_ARG_0:
            return CollectiveOp.sum_tensors(args[0])
        if func in CollectiveOp.COMM_TENSOR_ARG_1:
            return CollectiveOp.sum_tensors(args[1])
        if func in CollectiveOp.COMM_TENSOR_ARG_RES:
            return res.untyped_storage().nbytes()
        if func in CollectiveOp.COMM_TENSOR_SINGLE_UNTYPED_STORAGE:
            return args[0].untyped_storage().nbytes()
        if func is c10d._reduce_scatter_base_.default:
            return args[1].untyped_storage().nbytes()
        if func is c10d.alltoall_.default:
            # TODO(@sanketpurandare) - Confirm size computation
            return max(
                CollectiveOp.sum_tensors(args[0]), CollectiveOp.sum_tensors(args[1])
            )
        if func is c10d.alltoall_base_.default:
            # TODO(@sanketpurandare) - Confirm size computation
            return max(
                args[0].untyped_storage().nbytes(), args[1].untyped_storage().nbytes()
            )
        if func == _c10d_functional.all_gather_into_tensor_out.default:
            return args[-1].untyped_storage().nbytes()
        if func in CollectiveOp.COMM_TENSOR_RES_SUM:
            return CollectiveOp.sum_tensors(res)
        if func in CollectiveOp.COMM_TENSOR_ARG_0_AND_RES:
            # TODO(@sanketpurandare) - Confirm size computation
            return args[0].untyped_storage().nbytes() + res.untyped_storage().nbytes()
        if func is _c10d_functional.batch_p2p_ops.default:
            return CollectiveOp.sum_tensors(args[3])
        raise TypeError(f"Unknown function: {func} in {collective_ops}")

    @staticmethod
    def get_work(func, res) -> Work:  # type: ignore[no-untyped-def]
        if func in CollectiveOp.WK:
            return FakeWork.unbox(res)
        elif func in CollectiveOp.WK_ARG_1:
            return FakeWork.unbox(res[1])
        raise TypeError(f"Func {func} not found in {collective_ops}")
<|endoftext|># mypy: allow-untyped-defs
import warnings
import weakref
from collections.abc import Callable

import torch
from torch.autograd.graph import register_multi_grad_hook
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.utils._pytree import tree_flatten


__all__ = ["ModTracker"]


class ModTracker:
    """
    ``ModTracker`` is a context manager that tracks the nn.Module hierarchy during execution
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

        with ModTracker() as tracker:
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

    def __init__(self):
        self.parents = {"Global"}
        self._active_module_cnt = {}
        self._known_modules: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._seen_modules: weakref.WeakSet = weakref.WeakSet()
        self._has_callback = False
        self._post_bw_callbacks_to_enqueue: list[Callable] = []
        self._user_pre_fw_hook = None
        self._user_post_fw_hook = None
        self._user_pre_bw_hook = None
        self._user_post_bw_hook = None

    def _maybe_set_engine_callback(self):
        # This assumes no concurrent calls to backward
        if self._has_callback:
            return

        for post_bw_callback in reversed(self._post_bw_callbacks_to_enqueue):
            torch.autograd.Variable._execution_engine.queue_callback(post_bw_callback)
        self._post_bw_callbacks_to_enqueue.clear()

        def callback():
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

    def get_known_fqn(self, mod):
        """
        Return the fqn for the given module if it is known to the ``ModTracker``, otherwise ``None``.
        """
        return self._known_modules.get(mod, None)

    def register_user_hooks(
        self,
        pre_fw_hook: Callable | None = None,
        post_fw_hook: Callable | None = None,
        pre_bw_hook: Callable | None = None,
        post_bw_hook: Callable | None = None,
    ):
        """
        Registers user-specified hooks to be called before/after the forward/backward pass for each
        module tracked by the ``ModTracker``. One or more can be ``None``.
        Args:
            pre_fw_hook (Callable, optional): A hook to be called before the forward pass for the
                module. It should have the following signature:
                pre_fw_hook (module, input) -> None
            post_fw_hook (Callable, optional): A hook to be called after the forward pass for the
                module. It should have the following signature:
                post_fw_hook (module, input, output) -> None
            pre_bw_hook (Callable, optional): A multi-grad hook to be called on all the outputs of
                the module that require gradients. It should have the following signature:
                pre_bw_hook (module, grad_output) -> None
            post_bw_hook (Callable, optional): A multi-grad hook to be called on all the inputs of
                the module that require gradients. It should have the following signature:
                post_bw_hook (module, grad_input) -> None
        Raises:
            AssertionError: If a new hook is provided when one is already registered.
        Note:
            If the module is not alive during the backward pass, the pre_bw_hook and post_bw_hook will
            receive None as the module argument.
            The module fqn will be present in the ``parents`` attribute when each of the hooks is called.
            Hooks are intended to be used as markers only not to modify the inputs/outputs.
        """

        def set_hook(hook, user_hook, hook_name):
            if hook is not None and user_hook is not None:
                raise AssertionError(
                    f"Only one {hook_name} can be registered at a time"
                    f" Clear the existing hook by calling ``clear_user_hooks`` before registering a new one"
                )
            return hook

        self._user_pre_fw_hook = set_hook(
            pre_fw_hook, self._user_pre_fw_hook, "pre_fw_hook"
        )
        self._user_post_fw_hook = set_hook(
            post_fw_hook, self._user_post_fw_hook, "post_fw_hook"
        )
        self._user_pre_bw_hook = set_hook(
            pre_bw_hook, self._user_pre_bw_hook, "pre_bw_hook"
        )
        self._user_post_bw_hook = set_hook(
            post_bw_hook, self._user_post_bw_hook, "post_bw_hook"
        )

    def clear_user_hooks(self):
        """
        Clears the user specified hooks registered with ``register_user_hooks``
        """
        self._user_pre_fw_hook = None
        self._user_post_fw_hook = None
        self._user_pre_bw_hook = None
        self._user_post_bw_hook = None

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

    def _get_append_fn(self, w_mod, name, is_bw):
        def fn(*args):
            if is_bw:
                self._maybe_set_engine_callback()
            if name in self.parents and not self.is_bw:

                def custom_formatwarning(msg, category, filename, lineno, line=None):
                    return f"{filename}:{lineno}: {category.__name__}: {msg} \n"

                # pyrefly: ignore [bad-assignment]
                warnings.formatwarning = custom_formatwarning
                warnings.warn(
                    "The module hierarchy tracking maybe be messed up."
                    " Please file a bug to PyTorch, if it is the case.",
                    stacklevel=2,
                )
            if name not in self.parents:
                self._active_module_cnt[name] = 1
                self.parents.add(name)
            else:
                self._active_module_cnt[name] += 1

            if self._user_pre_bw_hook is not None and is_bw:
                self._user_pre_bw_hook(w_mod(), args)

        return fn

    def _get_pop_fn(self, w_mod, name, is_bw):
        def fn(*args):
            if self._user_post_bw_hook is not None and is_bw:
                self._user_post_bw_hook(w_mod(), args)
            if name in self.parents:
                self._active_module_cnt[name] -= 1
                if self._active_module_cnt[name] == 0:
                    self.parents.remove(name)
            elif not self.is_bw:
                # Due to some input/output not requiring gradients, we cannot enforce
                # proper nesting in backward
                raise RuntimeError(
                    "The Module hierarchy tracking is wrong. Report a bug to PyTorch"
                )

        return fn

    def _fw_pre_hook(self, mod, input):
        if torch._dynamo.eval_frame._is_in_optimized_module():
            return

        name = self._get_mod_name(mod)
        w_mod = weakref.ref(mod)
        self._get_append_fn(w_mod, name, False)()
        if self._user_pre_fw_hook is not None:
            self._user_pre_fw_hook(mod, input)
        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if not self.is_bw:
            if tensors:
                register_multi_grad_hook(tensors, self._get_pop_fn(w_mod, name, True))
            else:
                self._post_bw_callbacks_to_enqueue.append(
                    self._get_pop_fn(w_mod, name, True)
                )

    def _fw_post_hook(self, mod, input, output):
        if torch._dynamo.eval_frame._is_in_optimized_module():
            return

        name = self._get_mod_name(mod)
        w_mod = weakref.ref(mod)
        if self._user_post_fw_hook is not None:
            self._user_post_fw_hook(mod, input, output)
        self._get_pop_fn(w_mod, name, False)()
        args, _ = tree_flatten(output)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if not self.is_bw and tensors:
            register_multi_grad_hook(
                tensors, self._get_append_fn(w_mod, name, True), mode="any"
            )

    def __enter__(self):
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(
            self._fw_post_hook, always_call=True
        )
        return self

    def __exit__(self, *args):
        self._fw_pre_handle.remove()
        self._fw_post_handle.remove()
<|endoftext|>import math
import os
import re
import warnings
from collections.abc import Callable
from copy import deepcopy
from enum import auto, Enum
from functools import partial, wraps
from typing import Any, TYPE_CHECKING
from typing_extensions import Self

import torch
import torch.distributed._tools.fake_collectives
from torch import nn, optim
from torch._guards import active_fake_mode
from torch.distributed._tools.common_utils import get_untyped_storages
from torch.distributed._tools.mod_tracker import ModTracker
from torch.distributed.tensor import DTensor
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map_only
from torch.utils.weak import WeakIdKeyDictionary, weakref


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)
_TOTAL_KEY = "Total"

__all__ = ["MemTracker"]


class _RefType(str, Enum):
    """Base Class for defining memory reference types, categorizing tensors based on their usage within a model."""


class _State(str, Enum):
    """Base Class for defining module state to capture snapshots ."""


class _MemRefType(_RefType):
    """
    An enum to define memory reference types, categorizing tensors based on their usage within a model.

        - PARAM: Tensors registered as nn.Parameter within modules.
        - BUFFER: Tensors registered as nn.Buffer within modules.
        - GRAD: Gradients associated with parameters.
        - ACT: Tensors produced during the forward pass and recomputation in activation checkpointing.
        - TMP: Temporary memory used during the backward pass, including gradients of activations.
        - OPT: Tensors holding optimizer states.
        - OTH: Tensors registered via `track_external` that do not fit the above categories.
    """

    PARAM = "Parameter"
    BUFFER = "Buffer"
    GRAD = "Gradient"
    ACT = "Activation"
    TEMP = "Temp"
    OPT = "Optstate"
    OTH = "Other"


class _ModState(_State):
    """
    An enum to define the state of a module.

        - PRE_FW: The module is about to run the forward pass.
        - POST_FW: The module has finished running the forward pass.
        - PEAK_FW: The module has reached the peak memory usage during the forward pass.
        - PRE_BW: The module is about to run the backward pass.
        - PRE_FW_AC: The module is about to run the forward pass with activation checkpointing.
        - POST_FW_AC: The module has finished running the forward pass with activation checkpointing.
        - POST_BW: The module has finished running the backward pass.
        - PEAK_BW: The module has reached the peak memory usage during the backward pass.
    """

    PRE_FW = "Pre-Forward"
    POST_FW = "Post-Forward"
    PEAK_FW = "Peak-Forward"
    PRE_BW = "Pre-Backward"
    PRE_FW_AC = "Pre-Forward-AC"
    POST_FW_AC = "Post-Forward-AC"
    POST_BW = "Post-Backward"
    PEAK_BW = "Peak-Backward"


class _ModMemStats:
    """
    A class to store the memory statistics of a module.

    Args:
        mod_fqn (str): The fully qualified name of the module.
    Attributes:
        mod_fqn (str): The fully qualified name of the module.
        parameter_mem (int): The memory usage of the parameters of the module.
        buffer_mem (int): The memory usage of the buffers of the module.
        input_mem (int): The memory usage of the inputs to the module.
        output_mem (int): The memory usage of the outputs from the module.
        snapshots (Dict[_ModState, Dict[torch.device, Dict[str, int]]]): A dictionary of memory snapshots
        of the module at different states defined by ``_ModState``.
    Note:
        The memory snapshot is stored as a dictionary - Dict[torch.device, Dict[str, int]], where each key is a device,
         and each value is another dictionary with keys as memory reference types defined by `_MemRefType` and
         values as the memory consumed in bytes.
    """

    def __init__(self, mod_fqn: str):
        self.mod_fqn = mod_fqn
        self.parameter_mem: int
        self.buffer_mem: int
        self.input_mem: int
        self.output_mem: int
        self.local_peak: dict[torch.device, int] = {}
        self.snapshots: dict[_ModState, list[dict[torch.device, dict[str, int]]]] = {}


class _WeakRefInfo:
    """
    Manages memory statistics and device attributes for tensor storages.
    """

    def __init__(
        self, size: int, element_size: int, device: torch.device, reftype: _RefType
    ) -> None:
        """
        Initializes the ``_WeakRefInfo`` object with tensor storage properties.

        Args:
            size (int): The number of elements in the tensor storage.
            element_size (int): The size of each element in the tensor storage.
            device (torch.device): The device on which the tensor is allocated.
            reftype (_RefType): The reference type of the tensor.
        """
        self.size = size
        self.element_size = element_size
        self.reftype = reftype
        self.device = device
        self.mem_consumed = self._calculate_mem_consumed()

    def _calculate_mem_consumed(self) -> int:
        """
        Calculates the memory consumed by the tensor storage, considering device-specific allocation rules.

        Returns:
            int: The memory consumed in bytes.
        """
        mem = self.size * self.element_size
        if self.device.type == "cuda":
            return math.ceil((mem) / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
        return mem

    def update_mem_consumed(self, st: torch.UntypedStorage) -> int:
        """
        Updates and returns the memory consumed if the storage size has changed.

        Args:
            st (torch.UntypedStorage): The tensor storage to check for size updates.

        Returns:
            int: The updated memory consumed in bytes.
        """
        if st.size() != self.size:
            self.size = st.size()
            self.mem_consumed = self._calculate_mem_consumed()
        return self.mem_consumed

    @classmethod
    def create_winfo(
        cls,
        st: torch.UntypedStorage,
        device: torch.device,
        reftype: _RefType,
        callback: Callable[[Self, weakref.ref], Any] | None = None,
    ) -> tuple[Self, weakref.ref]:
        """
        Creates a new ``_WeakRefInfo`` instance and a weak reference to a ``torch.UntypedStorage`` object,
        optionally attaching a callback to the weak reference.

        Args:
            st (torch.UntypedStorage): The storage object for which to create the weak reference info.
            device (torch.device): The device associated with the storage object.
            reftype (_RefType): The type of reference, used to categorize the storage.
            callback (Optional[Callable[[Self, weakref.ref]]]): A callback function that is called when
                the storage object is about to be finalized (garbage collected). The callback function
                should accept two arguments: the ``_WeakRefInfo`` instance and the weak reference to the storage.
        Returns:
            Tuple[Self, weakref.ref]: A tuple containing the newly created ``_WeakRefInfo`` instance and the
            weak reference to the storage object. The weak reference may have an attached callback if provided.
        """

        winfo = cls(st.size(), st.element_size(), device, reftype)
        w_st = weakref.ref(st, partial(callback, winfo) if callback else None)
        return winfo, w_st


def _get_mem_divisor(units: str) -> int:
    unit_dict = {"B": 1, "KiB": 2**10, "MiB": 2**20, "GiB": 2**30}
    if units in unit_dict:
        return unit_dict[units]
    else:
        raise ValueError(
            f"Unsupported unit: {units}. Supported units are: {', '.join(unit_dict.keys())}"
        )


def _rounding_fn(value: int, divisor: int, precision: int) -> float | int:
    return value if divisor == 1 else round(value / divisor, precision)


def _print_snapshot(snapshot: dict[torch.device, dict[str, int]], units: str) -> None:
    if len(snapshot) == 0:
        print("No memory tracked.")
        return
    divisor = _get_mem_divisor(units)
    for dev, dev_snap in snapshot.items():
        if _rounding_fn(dev_snap[_TOTAL_KEY], divisor, 2) <= 0:
            continue
        print(
            f"Device: {dev}",
            *(
                f"\t{k.value}: {_rounding_fn(v, divisor, 2)} {units}"
                if isinstance(k, _RefType)
                else f"\t{k}: {_rounding_fn(v, divisor, 2)} {units}"
                for k, v in dev_snap.items()
            ),
            sep="\n",
        )


def _print_snapshot_tabular(
    snapshot: dict[torch.device, dict[str, int]], units: str
) -> None:
    if len(snapshot) == 0:
        print("No memory tracked.")
        return
    try:
        from tabulate import tabulate
    except ImportError as err:
        raise ImportError(
            "Please install tabulate to use the tabulate option."
        ) from err
    divisor = _get_mem_divisor(units)
    table_data = []
    key_list = list(next(iter(snapshot.values())).keys())
    headers = ["Device"] + [
        f"{key.value}" if isinstance(key, _RefType) else f"{key}" for key in key_list
    ]

    for dev, dev_snap in snapshot.items():
        if _rounding_fn(dev_snap[_TOTAL_KEY], divisor, 2) <= 0:
            continue
        row = [str(dev)]
        row.extend(f"{_rounding_fn(v, divisor, 2)} {units}" for v in dev_snap.values())
        table_data.append(row)
    print(tabulate(table_data, headers=headers, tablefmt="rst"))


def _print_state_snapshots(
    snapshots: dict[_State, list[dict[torch.device, dict[str, int]]]], units: str
) -> None:
    for state, snapshot_list in snapshots.items():
        print(f"{state.value}")
        for i, snapshot in enumerate(snapshot_list):
            print(f"# {i + 1}:")
            _print_snapshot(snapshot, units)
    print()


def _print_state_snapshots_tabular(
    snapshots: dict[_State, list[dict[torch.device, dict[str, int]]]], units: str
) -> None:
    try:
        from tabulate import tabulate
    except ImportError as err:
        raise ImportError(
            "Please install tabulate to use the tabulate option."
        ) from err

    table_data = []
    last_state_call = None
    divisor = _get_mem_divisor(units)
    for state, snapshot_list in snapshots.items():
        for i, snapshot in enumerate(snapshot_list):
            state_call = f"{state.value} # {i + 1}"
            for dev, dev_snap in snapshot.items():
                if _rounding_fn(dev_snap[_TOTAL_KEY], divisor, 2) <= 0:
                    continue
                row = {
                    "State & Call": (
                        state_call if state_call != last_state_call else ""
                    ),
                    "Device": str(dev),
                }
                last_state_call = state_call
                for k, v in dev_snap.items():
                    row[f"{k.value}" if isinstance(k, _RefType) else f"{k}"] = (
                        f"{_rounding_fn(v, divisor, 2)} {units}"
                    )
                table_data.append(row)
    print(tabulate(table_data, headers="keys", tablefmt="rst"))


class _UpdateType(Enum):
    # These are used for tracking updates to the continuously maintained memory snapshot.
    # ADD - When a new tensor storage is tracked
    # DEL - When a tensor storage is about to be finalized (garbage collected).
    # REF - When a tensor reference is updated, for instance, the gradients are marked as
    #       generic backward reference types until the grad_hook categorizes them as gradients.
    # SIZE - When a tensor's storage is resized.
    ADD = auto()
    DEL = auto()
    REF = auto()
    SIZE = auto()


class MemTracker(TorchDispatchMode):
    """
    A TorchDispatchMode to track, categorize and attribute the tensor memory created or accessed within its context.

    It categorizes the tracked tensors as parameters, buffers, activations, gradients, temporary memory and optimizer states
    as defined by ``_MemRefType`` within its context. It captures memory `snapshots` for the modules, called within its context,
    at various states defined by ``_ModState``.

    Attributes:
        memory_tracking: A weakref key dictionary to store the memory statistics of each module. Each key
        is a reference to a module, and each value is a ``_ModMemStats`` object that stores the memory
        statistics of the module.

    Note:
        The MemTracker should be used as a context manager. The modules, optimizers, and any other tensors created within
        the context of MemTracker will be tracked by default. Any tensors or stateful objects such as modules, optimizers etc.
        that need to be tracked but are created outside the MemTracker should be registered using the `track_external` method.
        The `track_external` method should be called before the MemTracker is used. Any tensors created outside the ``MemTracker``
        and not supplied to the `track_external` method will not be tracked by the ``MemTracker``.

    Example usage:

        .. code-block:: python

            module = ...
            optimizer = ...
            inp = ...
            mem_tracker = MemTracker()
            mem_tracker.track_external(module, optimizer, inp)
            with mem_tracker as mt:
                loss = module(inp)
                print("After Forward:")
                mt.display_snapshot("current")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            mt.display_snapshot("peak")
            mt.display_modulewise_snapshots(depth=3, units="MiB")

    Known Limitations:
        - The ``MemTracker`` does not track memory for tensors that bypass the ``TorchDispatchMode`` ex. under ``no_dispatch``.
        - Resizing tensor storages directly by using non-Tensor methods other than using ``torch.Untyped_Storage.resize_``
          is not tracked. File a Github issue if you have use-cases for this.
        - If the tensors are not traceable or wrappable subclasses of ``torch.Tensor``, then the tracker does not know how to
            track their storages. File a Github issue if you have use-cases for this.
        - During AC in the backward pass there might be misattribution between activation and temp memory, but the peak memory
          will be tracked accurately. This will be fixed in the next update by hooking intricately with ``torch.utils.checkpoint``.
    """

    def __init__(self) -> None:
        self.memory_tracking = WeakIdKeyDictionary()
        self._curr_mem_snap: dict[torch.device, dict[str, int]] = {}
        self._peak_mem: dict[torch.device, int] = {}
        self._peak_mem_snap: dict[torch.device, dict[str, int]] = {}
        self._param_to_grad_hook_handles = WeakIdKeyDictionary()
        self._optimizer_hook_handles: tuple[RemovableHandle, RemovableHandle] | None = (
            None
        )
        # Dictionary to store the ``_WeakRefInfo`` instances corresponding to each tensor's storage.
        self._WINFO = WeakIdKeyDictionary()
        self._mod_tracker = ModTracker()
        # This is a general memory tracker which can be used with any ``_RefType`` subclass
        self._ref_class: type[_RefType] = _MemRefType
        # Flags to track if we are in the AC region or optimizer step region
        self._in_opt: bool = False
        self._in_ac: bool = False
        # Weak references to the topmost AC module currently active
        self._ac_mod: weakref.ref | None = None
        self._orig_resize = torch.UntypedStorage.resize_
        self._depth = 0

    def _update_snap(
        self,
        u_type: _UpdateType,
        winfo: _WeakRefInfo,
        old_mem_consumed: int | None = None,
        old_reftype: _RefType | None = None,
    ) -> None:
        # Initialize a flag to track if the total memory might drop to zero after updates.
        maybe_zero = False
        # Ensure the device entry exists in the current memory snapshot, initializing if necessary.
        # pyrefly: ignore [no-matching-overload]
        dev_snap = self._curr_mem_snap.setdefault(
            winfo.device, dict.fromkeys(self._ref_class, 0)
        )
        dev_snap.setdefault(_TOTAL_KEY, 0)
        # Handle different types of updates based on the update type (`u_type`).
        if u_type == _UpdateType.ADD:
            # Increase the memory consumed for the specific reference type and update the total.
            dev_snap[winfo.reftype] += winfo.mem_consumed
            dev_snap[_TOTAL_KEY] += winfo.mem_consumed
        elif u_type == _UpdateType.DEL:
            # Decrease the memory consumed for the specific reference type and reduce the total.
            dev_snap[winfo.reftype] -= winfo.mem_consumed
            dev_snap[_TOTAL_KEY] -= winfo.mem_consumed
            maybe_zero = True
        elif u_type == _UpdateType.REF:
            if old_reftype is None:
                raise AssertionError
            # Adjust memory consumption between two reference types within the same device.
            dev_snap[old_reftype] -= winfo.mem_consumed
            dev_snap[winfo.reftype] += winfo.mem_consumed
        elif u_type == _UpdateType.SIZE:
            if old_mem_consumed is None:
                raise AssertionError
            # Adjust the memory consumed for a reference type due to a change in size.
            change = winfo.mem_consumed - old_mem_consumed
            dev_snap[winfo.reftype] += change
            dev_snap[_TOTAL_KEY] += change
            maybe_zero = True
        else:
            raise ValueError(f"Invalid update type: {u_type}")
        # Check if the total memory for the device has dropped to zero.
        if maybe_zero:
            if self._curr_mem_snap[winfo.device][_TOTAL_KEY] == 0:
                # Remove the device entry from the memory snapshot if the total memory is zero.
                del self._curr_mem_snap[winfo.device]

    def _update_and_maybe_create_winfos(
        self,
        t: torch.Tensor,
        reftype: _RefType,
        update_existing: bool = False,
    ) -> set[_WeakRefInfo]:
        sts = get_untyped_storages(t)
        winfos = set()
        for st in sts:
            # Attempt to retrieve existing ``_WeakRefInfo`` and its weak reference from the tracking dictionary.
            winfo, _ = self._WINFO.get(st, (None, None))
            if winfo is not None:
                # If ``_WeakRefInfo`` exists, check if the reference type needs to be updated.
                old_reftype = winfo.reftype
                if old_reftype != reftype:
                    # Update the reference type and apply changes via ``_update_snap``.
                    winfo.reftype = reftype
                    self._update_snap(_UpdateType.REF, winfo, old_reftype=old_reftype)
                winfos.add(winfo)
            elif update_existing:
                # If no existing ``_WeakRefInfo`` is found and update_existing is True, raise an error.
                raise KeyError("No existing winfo found")
            else:
                # If no existing _WeakRefInfo is found and update_existing is False, create a new ``_WeakRefInfo``.
                winfo, w_st = _WeakRefInfo.create_winfo(
                    st, t.device, reftype, self._delete_callback
                )
                # Store the new ``_WeakRefInfo`` and its weak reference in the tracking dictionary.
                self._WINFO[st] = (winfo, w_st)
                # Update the snapshot for the newly added ``_WeakRefInfo``.
                if winfo.mem_consumed > 0:
                    self._update_snap(_UpdateType.ADD, winfo)
                winfos.add(winfo)
        return winfos

    def _delete_callback(self, winfo: _WeakRefInfo, w_st: weakref.ref) -> None:
        # Callback to be called when the storage object corresponding to the  ``_WeakRefInfo``
        # instance is about to be finalized.
        if winfo.mem_consumed > 0:
            self._update_snap(_UpdateType.DEL, winfo)

    def _track_resize(self) -> None:
        # Need to monkey-patch this because ``torch.UntypedStorage.resize_`` is not captured
        # by ``TorchDispatchMode``.
        @wraps(self._orig_resize)
        def resize_(st: torch.UntypedStorage, size: int) -> None:
            self._orig_resize(st, size)
            winfo, _ = self._WINFO.get(st, (None, None))
            if winfo is not None and winfo.size != st.size():
                old_mem_consumed = winfo.mem_consumed
                winfo.update_mem_consumed(st)
                self._update_snap(
                    _UpdateType.SIZE, winfo, old_mem_consumed=old_mem_consumed
                )

        torch.UntypedStorage.resize_ = resize_  # type: ignore[method-assign, assignment]

    def _restore_resize(self) -> None:
        torch.UntypedStorage.resize_ = self._orig_resize  # type: ignore[method-assign]

    def _update_peak_stats(self, peak_state: _State) -> None:
        # We first capture the current memory snapshot of the current tracker state then,
        # We step through each of the modules we have tracked so far in ``memory_tracking``
        #  and check if it is currently active by querying ``_mod_tracker.parents``
        # If it is active, we update the per device peak memory usage for the module
        #  corresponding to the ``_State`` which can be ``PEAK_FW`` or ``PEAK_BW``.
        curr_snap = self._curr_mem_snap

        for mod_stats in self.memory_tracking.values():
            if mod_stats.mod_fqn in self._mod_tracker.parents:
                if peak_state in mod_stats.snapshots:
                    for dev, dev_snap in curr_snap.items():
                        if mod_stats.local_peak.get(dev, 0) < dev_snap[_TOTAL_KEY]:
                            mod_stats.local_peak[dev] = dev_snap[_TOTAL_KEY]
                            mod_stats.snapshots[peak_state][-1][dev] = deepcopy(
                                dev_snap
                            )

        for dev, dev_snap in curr_snap.items():
            if self._peak_mem.get(dev, 0) < dev_snap[_TOTAL_KEY]:
                self._peak_mem[dev] = dev_snap[_TOTAL_KEY]
                self._peak_mem_snap[dev] = deepcopy(dev_snap)

    def _track(self, reftype: _RefType, t: torch.Tensor) -> None:
        # Get the storages of the tensor and check if we have already tracked them.
        # If yes, then check if the storage size has changed and update the current snapshot.
        # Else create a new ``_WeakRefInfo`` instance and add it to the dictionary.
        sts = get_untyped_storages(t)
        for st in sts:
            winfo, _ = self._WINFO.get(st, (None, None))
            if winfo is not None:
                if winfo.size != st.size():
                    old_mem_consumed = winfo.mem_consumed
                    winfo.update_mem_consumed(st)
                    self._update_snap(
                        _UpdateType.SIZE, winfo, old_mem_consumed=old_mem_consumed
                    )
                return
            else:
                winfo, w_st = _WeakRefInfo.create_winfo(
                    st, t.device, reftype, self._delete_callback
                )
                self._WINFO[st] = (winfo, w_st)
                # Update the current snapshot for the newly added ``_WeakRefInfo``.
                if winfo.mem_consumed > 0:
                    self._update_snap(_UpdateType.ADD, winfo)

    def get_tracker_snapshot(
        self, type: str = "current"
    ) -> dict[torch.device, dict[str, int]]:
        """
        Capture a snapshot of the memory usage breakdown per device, based on the specified type.

        Args:
            type (str): The type of snapshot to capture. Can be "current" for the current memory usage or "peak" for the
                        peak memory usage. Defaults to "current".
        Returns:
            Dict[torch.device, Dict[str, int]]: A dictionary where each key is a torch.device, and each value is another
                                                dictionary. This inner dictionary has keys representing memory reference
                                                types as defined in ``_MemRefType`` and values representing the amount of
                                                memory consumed in bytes.
        Raises:
            ValueError: If an invalid type is specified.
        """
        if type == "current":
            return deepcopy(self._curr_mem_snap)
        elif type == "peak":
            return deepcopy(self._peak_mem_snap)
        else:
            raise ValueError(f"Invalid type {type}")

    def _track_module_params_and_buffers(
        self, module: nn.Module, install_grad_hooks: bool = True
    ) -> tuple[int, int]:
        # Track the parameters and buffers of the module if not already tracked.
        # If the parameters have gradients, track the gradients as well.
        # If install_grad_hooks is True, install a gradient hook on the parameters
        #  to track the gradients, if it has not already been installed.
        # Return the total memory consumed by the parameters and buffers.
        def _grad_hook(grad: torch.Tensor) -> None:
            self._update_and_maybe_create_winfos(
                grad,
                _MemRefType.GRAD,
            )

        param_memory = 0
        for param in module.parameters():
            winfos = self._update_and_maybe_create_winfos(
                param,
                _MemRefType.PARAM,
            )
            param_memory += sum(winfo.mem_consumed for winfo in winfos)
            if param.grad is not None:
                self._update_and_maybe_create_winfos(
                    param.grad,
                    _MemRefType.GRAD,
                )
            if (
                self._param_to_grad_hook_handles.get(param, None) is None
                and install_grad_hooks
            ):
                grad_hook_handle = param.register_hook(_grad_hook)
                post_acc_grad_hook_handle = param.register_post_accumulate_grad_hook(
                    lambda p: (_grad_hook(p.grad))
                )
                self._param_to_grad_hook_handles[param] = (
                    grad_hook_handle,
                    post_acc_grad_hook_handle,
                )
        buffer_memory = 0
        for buffer in module.buffers():
            winfos = self._update_and_maybe_create_winfos(
                buffer,
                _MemRefType.BUFFER,
            )
            buffer_memory += sum(winfo.mem_consumed for winfo in winfos)
        return (param_memory, buffer_memory)

    def _track_inputs_or_outputs(self, args: Any) -> int:
        # Calculate the memory consumed by the inputs or outputs of the module.
        input_or_output_memory = 0

        def add_inps_or_outs(t: torch.Tensor) -> None:
            nonlocal input_or_output_memory
            sts = get_untyped_storages(t)
            for st in sts:
                winfo, _ = self._WINFO.get(st, (None, None))
                if winfo is not None:
                    input_or_output_memory += winfo.mem_consumed

        tree_map_only(torch.Tensor, add_inps_or_outs, args)
        return input_or_output_memory

    def _pre_fw_hook(self, module: nn.Module, inputs: Any) -> None:
        # This is installed as a pre-fwd user hook with ``ModTracker.`` Based on the following cases we
        # set the state and capture the memory snapshot for the module.
        # Case 1: If the module is not in the ``memory_tracking`` dictionary, we track the parameters, buffers,
        #         input and output memory of the module. Create a new ``_ModMemStats`` instance for the module
        #         and add it to the ``memory_tracking`` dictionary.
        # Case 2: If the module is already in the ``memory_tracking`` dictionary and we are in backward, this means
        #         we are in the AC region. We check if this is the top most module in the AC region. If it is,
        #         we store a weak reference and set the flag ``_in_ac`` to True.
        # Case 3: If the module is already in the ``memory_tracking`` dictionary and we are in forward, this means
        #         this module is called for the second time. If it is a root module, that means we are in the next
        #         iteration and we error out. If it is not a root module, that means it's a submodule that is being
        #         used multiple times in the same iteration, which we allow and track.
        # For Case 1 and 3, we also initialize the ``local_peak`` and ``PEAK_FW`` snapshot for the module.
        mod_name = self._mod_tracker.get_known_fqn(module)
        if mod_name is None:
            raise AssertionError
        if module not in self.memory_tracking:
            mod_stats = _ModMemStats(mod_name)
            param_mem, buffer_mem = self._track_module_params_and_buffers(
                module, install_grad_hooks=True
            )
            input_mem = self._track_inputs_or_outputs(inputs)
            mod_stats.parameter_mem = param_mem
            mod_stats.buffer_mem = buffer_mem
            mod_stats.input_mem = input_mem
            self.memory_tracking[module] = mod_stats
            state = _ModState.PRE_FW

        elif self._mod_tracker.is_bw:
            mod_stats = self.memory_tracking[module]
            state = _ModState.PRE_FW_AC
            if self._ac_mod is None:
                self._ac_mod = weakref.ref(module)
                self._in_ac = True
        else:
            parents = set(self._mod_tracker.parents) - {mod_name}
            if len(parents) == 1 and "Global" in parents:
                raise NotImplementedError(
                    "MemTracker does not support memory tracking for multiple iterative calls."
                    " Either use ``reset_mod_stats`` to clear module memory stats for the previous iteration"
                    " or file a github issue if you need this feature."
                )
            mod_stats = self.memory_tracking[module]
            state = _ModState.PRE_FW
            input_mem = self._track_inputs_or_outputs(inputs)
            mod_stats.mod_fqn = mod_name
            mod_stats.input_mem = input_mem

        mem_snapshot = self.get_tracker_snapshot()
        if state == _ModState.PRE_FW:
            mod_stats.local_peak = {
                dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in mem_snapshot.items()
            }
            mod_stats.snapshots.setdefault(_ModState.PEAK_FW, []).append(mem_snapshot)
        mod_stats.snapshots.setdefault(state, []).append(deepcopy(mem_snapshot))

    def _post_fw_hook(self, module: nn.Module, inputs: Any, outputs: Any) -> None:
        # This is installed as a post-fwd user hook with ``ModTracker``. Based on the following cases we
        # set the state and capture the memory snapshot for the module.
        # Case 1: This is called in backward, which means we are in the AC region. If this is the top most module
        #         in the AC region, we set the flag ``_in_ac`` to False.
        # Case 2: This is called in forward so we calculate the output memory
        #         of the module and update its mod_stats.
        mod_stats = self.memory_tracking[module]
        if self._mod_tracker.is_bw:
            state = _ModState.POST_FW_AC
            if self._ac_mod is not None and self._ac_mod() is module:
                self._ac_mod = None
                self._in_ac = False
        else:
            state = _ModState.POST_FW
            output_mem = self._track_inputs_or_outputs(outputs)
            mod_stats.output_mem = output_mem
        mod_stats.snapshots.setdefault(state, []).append(self.get_tracker_snapshot())

    def _pre_bw_hook(self, module: nn.Module, args: Any) -> None:
        # This is installed as a pre-bwd user hook with ``ModTracker``. We set the state and capture the
        # snapshot for the module. We also initialize the ``local_peak`` and ``PEAK_BW`` snapshot for it.
        # If the module is None, we skip the hook.
        # This can happen since this installed inside a multi-grad hook on the module's output tensors
        # and the module itself may not be alive during backward.
        if module is None:
            warnings.warn("Module is None. Skipping PRE_BW hook.", stacklevel=2)
            return
        mod_stats = self.memory_tracking[module]
        mem_snapshot = self.get_tracker_snapshot()
        mod_stats.local_peak = {
            dev: dev_snap[_TOTAL_KEY] for dev, dev_snap in mem_snapshot.items()
        }
        mod_stats.snapshots.setdefault(_ModState.PEAK_BW, []).append(mem_snapshot)
        mod_stats.snapshots.setdefault(_ModState.PRE_BW, []).append(
            deepcopy(mem_snapshot)
        )

    def _post_bw_hook(self, module: nn.Module, args: Any) -> None:
        # This is installed as a post-bwd user hook with ``ModTracker``. We set the state and capture the
        # snapshot for the module if it is not None.
        # This can happen since this installed inside a multi-grad hook on the module's input tensors
        # and the module itself may not be alive during backward.
        if module is None:
            warnings.warn("Module is None. Skipping POST_BW hook.", stacklevel=2)
            return
        mod_stats = self.memory_tracking[module]
        mod_stats.snapshots.setdefault(_ModState.POST_BW, []).append(
            self.get_tracker_snapshot()
        )

    def _track_optimizer_states(
        self, reftype: _RefType, optimizer: optim.Optimizer
    ) -> None:
        for states in optimizer.state.values():
            for val in states.values():
                if isinstance(val, torch.Tensor):
                    self._update_and_maybe_create_winfos(
                        val,
                        reftype,
                    )

    def _register_global_optimizer_hook(self) -> None:
        # Register a hook on the optimizer step to track the optimizer states.
        # The pre-hook is to set the flag ``_in_opt`` to True. The post-hook unsets the flag,
        # and also tracks any optimizer states that are created during the optimizer step.
        def _opt_step_pre_hook(
            optimizer: optim.Optimizer, args: Any, kwargs: Any
        ) -> None:
            self._in_opt = True

        def _opt_step_post_hook(
            optimizer: optim.Optimizer, args: Any, kwargs: Any
        ) -> None:
            self._track_optimizer_states(_MemRefType.OPT, optimizer)
            self._in_opt = False

        self._optimizer_hook_handles = (
            register_optimizer_step_pre_hook(_opt_step_pre_hook),
            register_optimizer_step_post_hook(_opt_step_post_hook),
        )

    def _deregister_param_and_optimizer_hooks(self) -> None:
        for (
            grad_hook_handle,
            post_acc_grad_hook_handle,
        ) in self._param_to_grad_hook_handles.values():
            grad_hook_handle.remove()
            post_acc_grad_hook_handle.remove()
        self._param_to_grad_hook_handles.clear()

        if self._optimizer_hook_handles is not None:
            for handle in self._optimizer_hook_handles:
                handle.remove()
            self._optimizer_hook_handles = None

    def track_external(
        self, *external: nn.Module | optim.Optimizer | torch.Tensor
    ) -> None:
        """
        Track tensors and stateful objects like modules, optimizers etc. that are created outside the MemTracker.

        This method should be called before the ``MemTracker`` is used. Any tensors that are not module parameters, buffers,
        gradients activations, or optimizer states will be categorized as ``Other``. If you want them categorized with a
        custom name, please file a GitHub issue. Any tensors created outside the MemTracker and not supplied to this
        method will not be tracked by ``MemTracker``.

        Args:
            *external (Union[nn.Module, optim.Optimizer, torch.Tensor]): The external modules, optimizers, and
                                                                         tensors to be tracked.
        """
        flat_external, _ = tree_flatten(external)
        for obj in flat_external:
            if isinstance(obj, torch.Tensor):
                self._update_and_maybe_create_winfos(
                    obj,
                    _MemRefType.OTH,
                )
            elif isinstance(obj, torch.nn.Module):
                self._track_module_params_and_buffers(obj, install_grad_hooks=False)
            elif isinstance(obj, optim.Optimizer):
                self._track_optimizer_states(_MemRefType.OPT, obj)
            elif obj is None:
                continue
            else:
                raise TypeError(
                    f"Object of type {type(obj)} is not supported for tracking. "
                    f"Only stateful objects like modules, optimizers, and tensors are supported."
                )

    def display_snapshot(
        self, type: str = "current", units: str = "B", tabulate: bool = False
    ) -> None:
        """
        Display the memory usage breakdown snapshot of the tracker based on the specified type and units.

        Keyword args:
            type (str): The type of snapshot to display. Can be "current" for the current memory usage or "peak" for the
                        peak memory usage. Defaults to "current".
            units (str): The units to use for displaying memory usage. Defaults to "B". Supports ["B", "KiB", "MiB", "GiB"].
            tabulate (bool): Whether to display the snapshot in a tabular format. Defaults to False.
        """
        snapshot = self.get_tracker_snapshot(type)
        if tabulate:
            _print_snapshot_tabular(snapshot, units)
        else:
            _print_snapshot(snapshot, units)

    def display_modulewise_snapshots(
        self, depth: int = 2, units: str = "B", tabulate: bool = False
    ) -> None:
        """
        Print per device memory breakdown snapshot for each module called within MemTracker.

        Snapshots are displayed for the states defined by ``_ModState``.
        The module hierarchy is displayed up to the specified depth.

        Keyword Args:
            depth (int, optional): The depth of the module hierarchy to display. Defaults to 2.
            units (str, optional): The units to use for memory tracking. Defaults to "B". Supports ["B", "KiB", "MiB", "GiB"].
            tabulate (bool, optional): Whether to display the snapshot in a tabular format. Defaults to False.
        """

        def natural_sort_key(s: str) -> list[int | str]:
            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split("([0-9]+)", s)
            ]

        for mod_stats in sorted(
            self.memory_tracking.values(),
            key=lambda m_stats: natural_sort_key(m_stats.mod_fqn),
        ):
            mod_fqn = mod_stats.mod_fqn
            mod_depth = mod_fqn.count(".") + 1
            if mod_depth > depth:
                continue
            print(f"Module:  {mod_fqn}")
            if tabulate:
                _print_state_snapshots_tabular(mod_stats.snapshots, units)
            else:
                _print_state_snapshots(mod_stats.snapshots, units)

    def reset_mod_stats(self) -> None:
        """
        Reset all the module memory stats. Clears ``memory_tracking`` dictionary.
        """
        self.memory_tracking.clear()

    def __enter__(self) -> "MemTracker":
        if self._depth == 0:
            # None in eager, a FakeTensorMode instance in SAC.  Used in
            # __torch_dispatch__ to skip DTensor propagation ops.
            self._fake_mode_on_entry = active_fake_mode()
            self._register_global_optimizer_hook()
            self._mod_tracker.register_user_hooks(
                self._pre_fw_hook,
                self._post_fw_hook,
                self._pre_bw_hook,
                self._post_bw_hook,
            )
            self._track_resize()
            self._peak_mem_snap = self.get_tracker_snapshot()
            self._peak_mem = {
                dev: dev_snap[_TOTAL_KEY]
                for dev, dev_snap in self._peak_mem_snap.items()
            }
            self._mod_tracker.__enter__()
        super().__enter__()
        self._depth += 1
        return self

    # pyrefly: ignore [bad-override]
    def __exit__(self, *args: Any) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._deregister_param_and_optimizer_hooks()
            self._mod_tracker.clear_user_hooks()
            self._restore_resize()
            self._mod_tracker.__exit__(*args)
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # type: ignore[no-untyped-def]
        # When running this mode with DTensor, ordinarily all modes will
        # run **before** subclasses get a chance to run.
        # Returning NotImplemented here gives us a chance to let DTensor
        # run and desugar into local tensor ops, before `MemTracker` sees them.
        if any(t == DTensor for t in types):
            return NotImplemented
        if (
            func is torch.ops._c10d_functional.wait_tensor.default
            and active_fake_mode()
        ):
            # N.B: This is a hacky way to override the Meta IMPL of wait_tensor. The original impl returns
            # a new tensor which does not happen in eager mode, when a wait_tensor is called.
            # pyrefly: ignore [bad-index]
            res = args[0]
        else:
            res = func(*args, **kwargs or {})
        # DTensor sharding propagation may use a nested FakeTensorMode to
        # compute output metadata (shapes, placements).  The real memory
        # usage comes from the local op on shard data, which
        # runs outside DTensor's fake mode.  We use identity comparison
        # against the fake mode at tracker entry to skip propagation ops
        # while still tracking local ops in both eager and SAC contexts.
        #   - Eager (entry mode is None): real ops have no fake mode
        #     (None is None), DTensor propagation ops are skipped (B is not None).
        #   - SAC (entry mode is A): SAC ops run under mode A (A is A),
        #     DTensor propagation ops are skipped (B is not A).
        if active_fake_mode() is self._fake_mode_on_entry:
            # If we are tracking an optimizer state, we use the optimizer reference type.
            # If we are in backward region and not in AC region, we use the backward reference type.
            # Else we use the forward reference type.
            if self._in_opt:
                reftype = _MemRefType.OPT
            elif self._mod_tracker.is_bw and not self._in_ac:
                reftype = _MemRefType.TEMP
            else:
                reftype = _MemRefType.ACT
            tree_map_only(torch.Tensor, partial(self._track, reftype), res)
            peak_state = (
                _ModState.PEAK_BW if self._mod_tracker.is_bw else _ModState.PEAK_FW
            )
            self._update_peak_stats(peak_state)
        return res
<|endoftext|># mypy: allow-untyped-defs
import operator
import pickle
from collections import defaultdict
from collections.abc import Callable, Sequence
from itertools import chain
from typing import Any, no_type_check, TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


BYTES_PER_MB = 1024 * 1024.0


class MemoryProfileDispatchMode(TorchDispatchMode):
    """Run in ``TorchDispatchMode`` to get memory stats at operator level."""

    def __init__(self, memory_tracker) -> None:
        self.memory_tracker = memory_tracker

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        rs = func(*args, **kwargs)
        if func is torch.ops.aten.detach.default:
            return rs
        func_name: str = (
            self.memory_tracker._cur_module_name
            + "."
            + func.__name__
            + "_"
            + str(self.memory_tracker._operator_names[func.__name__])
        )
        self.memory_tracker._operator_names[func.__name__] = (
            self.memory_tracker._operator_names[func.__name__] + 1
        )
        self.memory_tracker._record_memory_stats(func_name)

        return rs


class MemoryTracker:
    """
    Collect and plot the memory stats at operator level.

    Includes ``memories_allocated``, ``memories_active`` and ``memories_reserved``.
    It also prints a summary for the top 20 operators that generate the most memories.

    Example usage:

        >>> # xdoctest: +SKIP(failing)
        >>> net.cuda()
        >>> input = input.cuda()

        >>> mem_tracker = MemoryTracker()
        >>> mem_tracker.start_monitor(net)

        >>> net.zero_grad(True)
        >>> loss = net(input)
        >>> if isinstance(loss, dict):
        >>>    loss = loss['out']
        >>> loss.sum().backward()
        >>> net.zero_grad(set_to_none=True)

        >>> mem_tracker.stop()
        >>> mem_tracker.summary()
        >>> mem_tracker.show_traces()
    """

    def __init__(self) -> None:
        torch._C._log_api_usage_once("torch.distributed.memory_tracker")
        self._hooks: list[RemovableHandle] = []
        self._operator_names: dict[str, int] = defaultdict(int)
        self.memories_allocated: dict[int, dict[str, float]] = defaultdict()
        self.memories_active: dict[int, dict[str, float]] = defaultdict()
        self.memories_reserved: dict[int, dict[str, float]] = defaultdict()
        self._markers: dict[str, int] = defaultdict(int)
        self._cur_module_name: str = ""
        self._op_index: int = 0
        self._num_alloc_retries: int = 0
        self._device_module = torch.get_device_module()

    @no_type_check
    def start_monitor(self, root_module: nn.Module) -> None:
        """
        Register module hooks and entering ``MemoryProfileDispatchMode``.

        This enables operator level memory stats can be tracked during module runtime.
        """
        self._clear_state()
        root_module.__setattr__("_memory_tracker_is_root", True)
        for name, m in root_module.named_modules():
            if m is not root_module:
                m.__setattr__("_memory_tracker_is_root", False)
            # fused_proxy_group does not support hooks
            if ".fused_proxy_grouped_embedding_bag" in name:
                continue
            # hook ordering with other hooks added by users is not managed, so
            # the memory stats tracked here may not completely accurate.
            h1 = m.register_forward_pre_hook(self._create_pre_forward_hook(name))
            h2 = m.register_forward_hook(self._create_post_forward_hook(name))
            # it does not work well with jagged tensor somehow, the root cause is not
            # clear and remove it for now as it does not really capture important info.
            # h3 = m.register_backward_hook(self._create_backward_hook(name))
            self._hooks.extend([h1, h2])
        self._device_module.empty_cache()
        if getattr(self, "profile_mode", None) is not None:
            raise AssertionError
        self.profile_mode = MemoryProfileDispatchMode(self)
        self.profile_mode.__enter__()

    @no_type_check
    def stop(self) -> None:
        """
        Remove module hooks and exit ``MemoryProfileDispatchMode`` to stop tracking memory stats at operator level.

        Get some aggregated stats when the memory_tracker() is enabled, like ``num_alloc_retries``.
        """
        self._num_alloc_retries = self._device_module.memory_stats().get(
            "num_alloc_retries", 0
        )

        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if getattr(self, "profile_mode", None) is None:
            raise AssertionError
        self.profile_mode.__exit__(None, None, None)
        self.profile_mode = None

    @no_type_check
    def summary(self, top: int = 20) -> None:
        """
        Print out the top operators that generate the most memories.

        The number of the top operators can be configured.
        """
        op_diff: dict[str, float] = defaultdict(float)
        op_name, previous_allocated_memory = self.memories_allocated[0]
        for i in range(1, self._op_index):
            op_name, current_allocated_memory = self.memories_allocated[i]
            op_diff[op_name] = current_allocated_memory - previous_allocated_memory
            previous_allocated_memory = current_allocated_memory

        print("------------------------------------------------")
        print(f"The number of alloc retries are: {self._num_alloc_retries}")
        print(f"Top {top} ops that generates memory are:")
        for k, v in sorted(op_diff.items(), key=operator.itemgetter(1), reverse=True)[
            :top
        ]:
            print(f"{k}: {v}MB")
        print("------------------------------------------------")

    @no_type_check
    def show_traces(self, path: str = "") -> None:
        import matplotlib.pyplot as plt

        def _plot_figure(x, y_values, labels):
            min_val = min(chain.from_iterable(y_values)) * 0.999
            max_val = max(chain.from_iterable(y_values)) * 1.001
            plt.figure()
            for y, label in zip(y_values, labels):
                plt.plot(x, y, label=label)
            plt.xlabel("# Operator Calls")
            plt.ylabel("Memory (MB)")
            plt.legend()
            for marker_name, marker in self._markers.items():
                if marker_name == "fw_bw_boundary":
                    plt.plot(
                        [marker, marker],
                        [min_val, max_val],
                        "r",
                        lw=2,
                        label=marker_name,
                    )
                else:
                    plt.plot(
                        [marker, marker],
                        [min_val, max_val],
                        "k-",
                        lw=2,
                        label=marker_name,
                    )

        if path != "":
            self.load(path)

        y_1 = [gb for (name, gb) in self.memories_allocated.values()]
        y_2 = [gb for (name, gb) in self.memories_active.values()]
        y_3 = [gb for (name, gb) in self.memories_reserved.values()]
        x = list(range(len(y_1)))
        # Split figures when there is big difference between
        # "reserved_memory" and "allocated_memory" or "active_memory".
        _plot_figure(
            x,
            [list(y_1), list(y_2), list(y_3)],
            ["allocated_memory", "active_memory", "reserved_memory"],
        )
        _plot_figure(x, [list(y_1)], ["allocated_memory"])
        _plot_figure(x, [list(y_2)], ["active_memory"])
        _plot_figure(x, [list(y_3)], ["reserved_memory"])

    def save_stats(self, path: str) -> None:
        """Save the stats using pickle during runtime if users want to plot the traces in other places like notebook."""
        stats = {
            "memories_allocated": self.memories_allocated,
            "memories_active": self.memories_active,
            "memories_reserved": self.memories_reserved,
            "markers": self._markers,
            "num_alloc_retries": self._num_alloc_retries,
        }

        with open(path, "wb") as f:
            pickle.dump(stats, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """Load the pickled memory stats to plot the traces or print the summary."""
        with open(path, "rb") as f:
            stats = pickle.load(f)

        self.memories_allocated = stats["memories_allocated"]
        self.memories_active = stats["memories_active"]
        self.memories_reserved = stats["memories_reserved"]
        self._markers = stats["markers"]
        self._num_alloc_retries = stats["num_alloc_retries"]

    def _create_pre_forward_hook(self, name: str) -> Callable:
        """Prefix operator name with current module and 'forward', and insert 'fw_start' marker at forward pass start."""

        def _pre_forward_hook(module: nn.Module, inputs: Any) -> None:
            self._cur_module_name = f"{name}.forward"
            if (
                hasattr(module, "_memory_tracker_is_root")
                and module._memory_tracker_is_root
            ):
                self._add_marker("fw_start")

        return _pre_forward_hook

    def _create_post_forward_hook(self, name: str) -> Callable:
        """Insert the marker 'fw_bw_boundary' at the boundary of forward and backward pass."""

        def _post_forward_hook(
            module: nn.Module,
            inputs: Sequence[torch.Tensor],
            outputs: Sequence[torch.Tensor],
        ) -> None:
            if (
                hasattr(module, "_memory_tracker_is_root")
                and module._memory_tracker_is_root
            ):
                self._add_marker("fw_bw_boundary")

        return _post_forward_hook

    def _create_backward_hook(self, name: str) -> Callable:
        """Insert the current module name with backward prefix for the operator name."""

        def _backward_hook(
            module: nn.Module, grad_input: torch.Tensor, grad_output: torch.Tensor
        ) -> None:
            self._cur_module_name = f"{name}.backward"

        return _backward_hook

    @no_type_check
    def _record_memory_stats(self, fn_name: str) -> None:
        """
        Record current memory allocated, current memory active and current memory reserved.

        The memory stats dict is indexed with ``self._op_index``.
        """
        memory_allocated: float = self._device_module.memory_allocated() / BYTES_PER_MB
        memory_reserved: float = self._device_module.memory_reserved() / BYTES_PER_MB
        memory_active: float = (
            self._device_module.memory_stats().get("active_bytes.all.current", 0)
            / BYTES_PER_MB
        )
        self.memories_allocated[self._op_index] = (fn_name, memory_allocated)
        self.memories_reserved[self._op_index] = (fn_name, memory_reserved)
        self.memories_active[self._op_index] = (fn_name, memory_active)
        self._op_index += 1

    def _add_marker(self, marker_name: str) -> None:
        """Set the marker's x-axis value."""
        marker_val = len(self.memories_allocated.values())
        self._markers[marker_name] = marker_val

    def _clear_state(self) -> None:
        """Clear states when start_monitor() is called."""
        self._operator_names.clear()
        self.memories_allocated.clear()
        self.memories_active.clear()
        self.memories_reserved.clear()
        self._markers.clear()
        self._cur_module_name = ""
        self._op_index = 0
        self._num_alloc_retries = 0
<|endoftext|>from collections.abc import Callable
from copy import deepcopy
from enum import auto, Enum
from functools import partial, wraps
from typing import Any, NamedTuple, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec, TypeVarTuple, Unpack

import torch
import torch.distributed._tools.fake_collectives
from torch import nn, optim
from torch._guards import active_fake_mode
from torch.distributed._tools.mem_tracker import _RefType, _State, MemTracker
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.weak import WeakIdKeyDictionary, weakref


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

_TOTAL_KEY = "Total"

__all__ = ["FSDPMemTracker"]

_P = ParamSpec("_P")
_R = TypeVar("_R")
_Ts = TypeVarTuple("_Ts")

c10d = torch.ops.c10d


class _FSDPRefType(_RefType):
    """
    Enumerates categories of memory usage in FSDP modules, including parameters, gradients, activations,
    and optimizer states.

    Attributes:
        SHARDED_PARAM (str): Memory usage of sharded parameters.
        UNSHARDED_PARAM (str): Memory usage of unsharded parameters.
        SHARDED_GRAD (str): Memory usage of sharded gradients corresponding to the sharded parameters.
        UNSHARDED_GRAD (str): Memory usage of unsharded gradients corresponding to the unsharded parameters.
        ACT (str): Memory usage of activations and tensors from forward and AC recomputation.
        TEMP (str): Memory usage of temporary tensors during the backward pass including gradients of activations.
        ALL_GATHER (str): Memory usage of all_gather output tensor.
        REDUCE_SCATTER (str): Memory usage of reduce_scatter input tensor.
        OPT (str): Memory usage of tensors storing optimizer states.
        INP (str): Memory usage of input tensors.
    """

    SHARDED_PARAM = "Sharded Param"
    UNSHARDED_PARAM = "Unsharded Param"
    BUFFER = "Buffer"
    SHARDED_GRAD = "Sharded Grad"
    UNSHARDED_GRAD = "Unsharded Grad"
