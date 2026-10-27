                inp,
                local_shard,
                padding_idx=padding_idx,
            )
        else:
            result = op_func(inp, local_shard)
        results.append(torch.transpose(result, 0, col_dim))

    # Distribute results to each rank with col rearrangement.
    output = _result_distribute_with_col_rearrange(
        results, input, world_size, weight, pg
    )

    # transpose the output and return result.
    return torch.transpose(output, 0, col_dim)


def _result_distribute_with_col_rearrange(results, input, world_size, weight, pg):
    """
    For col-wise sharding of weight, we need to distribute
    results to each rank. We do them in this function.
    Note that, if the index in the Sharding Spec is not equal to
    the rank number, we need to do the rearrangement based on the
    order given by the Sharding Spec (placement).

    Args:
        results: results from ops applied to inputs from all ranks.
            We need to distribute them back to their original ranks.
        input: tensor to be applied op to.
        world_size: number of ranks.
        weight: sharded weight tensor.
        pg: process group.

    Return: column rearranged result.
    """
    # Process results and outputs for all2all.
    sharding_dim = weight._sharding_spec.dim
    sharding_dim_size = weight.size(sharding_dim)
    dims = list(results[0].size())
    dims[0] = sharding_dim_size
    combined_results = torch.cat(results)
    output = torch.empty(
        *dims, device=combined_results.device, dtype=combined_results.dtype
    )

    # Compute output splits
    split_size = get_split_size(sharding_dim_size, world_size)
    output_split_sizes = [0] * world_size
    for idx, placement in enumerate(weight._sharding_spec.placements):
        output_split_sizes[placement.rank()] = get_chunked_dim_size(
            sharding_dim_size, split_size, idx
        )

    # distribute the outputs using all2all.
    output = all_to_all_single(
        output, combined_results, output_split_sizes=output_split_sizes, group=pg
    )

    # Check if we need to rearrange columns appropriately for output.
    rearrange_columns = any(
        idx != placement.rank()
        for idx, placement in enumerate(weight._sharding_spec.placements)
    )
    if not rearrange_columns:
        return output

    indices = []
    for placement in weight._sharding_spec.placements:
        dim_size = output_split_sizes[placement.rank()]
        start = sum(
            split_size if i < placement.rank() else 0
            for i, split_size in enumerate(output_split_sizes)
        )
        indices += list(range(start, start + dim_size))

    return output.index_select(0, torch.tensor(indices, device=output.device))


def _handle_max_norm_col_wise(
    max_norm,
    norm_type,
    local_shard,
    input,
    world_size,
    gathered_inputs,
    pg,
):
    """
    For col-wise sharding of weight, we need to aggregate the
    norm across all ranks before we can perform the proper re-norm.
    Note that, the max_norm logic is only applied to the embedding
    indices that are looked up and not the whole shard.

    Args:
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        local_shard: col-wise shared local weight used for lookup.
        input: tensor to be applied op to.
        world_size: number of ranks.
        gathered_inputs: list of inputs from all ranks.
        pg: process group.

    Return:
        local_shard_norm_renormed: local_shard re-normed to max_norm if the norm is larger
            than it.

    """
    norm_type = norm_type if norm_type is not None else 2.0
    unique_inp = torch.unique(torch.cat(gathered_inputs))
    local_shard_sum = torch.sum(
        torch.pow(torch.abs(local_shard), norm_type), dim=1, dtype=local_shard.dtype
    )
    # For col-wise sharding, we need to first aggregate the powered sum
    # from each rank first and then calculate the norm.
    local_shard_sum = all_reduce(local_shard_sum, group=pg)
    local_shard_norm = torch.pow(local_shard_sum, 1.0 / norm_type)
    max_norm_tensor = torch.full(
        (local_shard.size(0),),
        float("inf"),
        dtype=local_shard.dtype,
        device=input.device,
    )
    max_norm_tensor[unique_inp] = max_norm
    local_shard_t = local_shard.t().contiguous()
    normalized_tensor = torch.where(
        local_shard_norm > max_norm_tensor, max_norm_tensor, local_shard_norm
    )
    # Make sure divisor is not zero.
    local_shard_norm[local_shard_norm == 0.0] = 1.0
    local_shard_norm_renormed = (
        torch.div(torch.mul(local_shard_t, normalized_tensor), local_shard_norm)
        .t()
        .contiguous()
    )
    return local_shard_norm_renormed


def _all_gather_base_input(input, pg):
    """
    Use _all_gather_base to get a concatenated input from each rank.

    Args:
        input: tensor to be applied op on.
        pg: process group.

    Returns:
        gathered_inputs: input gathered from each rank and concat by dim 0.
    """
    # allgather the inputs first.
    gather_inp_size = list(input.size())
    gather_inp_size[0] = input.size(0) * dist.get_world_size(pg)
    gather_inp = torch.empty(gather_inp_size, device=input.device, dtype=input.dtype)
    return _all_gather_base(gather_inp, input, group=pg)


def _handle_row_wise_mask(gather_inp, padding_idx, weight, world_size, rank):
    """
    Mask the input for embedding look-up for IDs which are not stored
    on the current rank. This function also adjust the ``padding_idx``
    so that it is only used on the rank where the corresponding row is
    stored.

    Note that, with ``max_norm`` flag on, only weights of rows being
    looked up will be re-normed. So we need an extra row for masked ID
    so that it does not affect the final result and ``max_norm``.

    Args:
        gather_inp: tensor to be applied op on gathered from all ranks.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.
        weight: weight tensor of Embedding look-up table.
        world_size: number of ranks.
        rank: # of cuda process.

    Returns:
        lookup_input: Tensor of masked input.
        padding_idx: adjusted padding_idx.
        padding_row: The extra row we used during lookup so that
            looking up does not affect ``max_norm``.
    """
    (start_pos, chunk_size) = get_chunk_sharding_params(
        weight.size(0), world_size, weight._sharding_spec, rank
    )
    mask = (gather_inp < start_pos) | (gather_inp >= start_pos + chunk_size)
    lookup_input = gather_inp.clone() - start_pos
    lookup_input[mask] = chunk_size
    if (
        padding_idx is not None
        and padding_idx >= start_pos
        and padding_idx < (start_pos + chunk_size)
    ):
        padding_idx = padding_idx - start_pos
    else:
        padding_idx = None

    # When max_norm is set, it will only re-norm the row being looked up.
    padding_row = torch.zeros(
        1, weight.size(1), device=gather_inp.device, dtype=weight.dtype
    )
    return lookup_input, padding_idx, padding_row
<|endoftext|># Keep old package for BC purposes, this file should be removed once
# everything moves to the `torch.distributed.checkpoint` package.
import sys
import warnings

import torch
from torch.distributed.checkpoint import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`torch.distributed._shard.checkpoint` will be deprecated, "
        "use `torch.distributed.checkpoint` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules["torch.distributed._shard.checkpoint"] = torch.distributed.checkpoint
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.optim._functional as F
from torch import Tensor
from torch.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional SGD Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalSGD:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        fused: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
        }
        self.nesterov = nesterov
        self.maximize = maximize
        self.foreach = foreach
        self.fused = fused
        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

    def step_param(self, param: Tensor, grad: Tensor | None):
        """Similar to self.step, but operates on a single parameter and
        its gradient.
        """
        # TODO: Once step_param interface is robust, refactor step to call
        # step param on each param.
        weight_decay = self.defaults["weight_decay"]
        momentum = self.defaults["momentum"]
        dampening = self.defaults["dampening"]
        lr = self.defaults["lr"]
        params = [param]
        momentum_buffer_list: list[Tensor | None] = []
        grads = []

        has_sparse_grad = False
        if grad is not None:
            grads.append(grad)
            if grad.is_sparse:
                has_sparse_grad = True
            if param not in self.state:
                self.state[param] = {}
            state = self.state[param]
            if "momentum_buffer" not in state:
                momentum_buffer_list.append(None)
            else:
                momentum_buffer_list.append(state["momentum_buffer"])

        with torch.no_grad():
            F.sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=self.nesterov,
                maximize=self.maximize,
                has_sparse_grad=has_sparse_grad,
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )
        # update momentum_buffer in state
        state = self.state[param]
        momentum_buffer = momentum_buffer_list[0]
        if momentum_buffer is not None:
            state["momentum_buffer"] = momentum_buffer

    def step(self, gradients: list[Tensor | None]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        momentum_buffer_list: list[Tensor | None] = []
        lr = self.defaults["lr"]
        weight_decay = self.defaults["weight_decay"]
        momentum = self.defaults["momentum"]
        dampening = self.defaults["dampening"]

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_sparse_grad = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                params_with_grad.append(param)
                grads.append(gradient)
                if gradient.is_sparse:
                    has_sparse_grad = True

                if param not in self.state:
                    self.state[param] = {}

                state = self.state[param]
                if "momentum_buffer" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["momentum_buffer"])

        with torch.no_grad():
            F.sgd(
                params_with_grad,
                grads,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=self.nesterov,
                maximize=self.maximize,
                has_sparse_grad=has_sparse_grad,
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )

        # update momentum_buffers in state
        for i, p in enumerate(params_with_grad):
            state = self.state[p]
            momentum_buffer = momentum_buffer_list[i]
            if momentum_buffer is not None:
                state["momentum_buffer"] = momentum_buffer
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

r"""Zero Redundancy Optimizer."""

import collections
import copy
import enum
import inspect
import io
import logging
from collections.abc import Callable
from itertools import chain
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.distributed.optim.utils import functional_optim_map
from torch.optim import Optimizer


__all__ = ["ZeroRedundancyOptimizer"]


logger = logging.getLogger(__name__)


# Credits:  classy_vision/generic/distributed_util.py
def _recursive_copy_to_device(
    value: Any,
    non_blocking: bool,
    device: torch.device,
) -> Any:
    r"""
    Recursively searches lists, tuples, dicts and copies tensors to device if possible.

    Non-tensor values are passed as-is in the result.

    .. note::
        These are all copies, so if there are two objects that reference
        the same object, then after this call, there will be two different objects
        referenced on the device.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)

    if isinstance(value, (list, tuple)):
        values = [
            _recursive_copy_to_device(val, non_blocking=non_blocking, device=device)
            for val in value
        ]
        return values if isinstance(value, list) else tuple(values)

    if isinstance(value, collections.abc.Mapping):
        return {
            key: _recursive_copy_to_device(
                val, non_blocking=non_blocking, device=device
            )
            for key, val in value.items()
        }

    return value


def _is_trainable(param: torch.Tensor) -> bool:
    r"""Return if a parameter is trainable, where trainability is equivalent to requiring a gradient."""
    return param.requires_grad


def _broadcast_object(
    obj: Any,
    src_rank: int,
    group: object = dist.group.WORLD,
    device: torch.device = torch.device("cpu"),
) -> Any:
    r"""
    Broadcasts an object to the given group.

    It will be sending the object if called from the source rank and receiving
    the object otherwise.

    Arguments:
        obj: object to broadcast; only used if called on the source rank.
        src_rank (int): source rank.
        group (``ProcessGroup``, optional): group used for the broadcast
            (default: ``dist.group.WORLD``).
        device (``torch.device``, optional): device to send from or receive
            to (default: ``torch.device("cpu")``).

    Returns:
        The broadcasted object.
    """
    if dist.get_rank() == src_rank:
        # Send the object
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = bytearray(buffer.getbuffer())
        length_tensor = torch.LongTensor([len(data)]).to(device)
        data_send_tensor = torch.ByteTensor(data).to(device)
        # pyrefly: ignore [bad-argument-type]
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        # pyrefly: ignore [bad-argument-type]
        dist.broadcast(data_send_tensor, src=src_rank, group=group, async_op=False)
    else:
        # Receive the object
        length_tensor = torch.LongTensor([0]).to(device)
        # pyrefly: ignore [bad-argument-type]
        dist.broadcast(length_tensor, src=src_rank, group=group, async_op=False)
        data_recv_tensor = torch.empty(
            [int(length_tensor.item())], dtype=torch.uint8, device=device
        )
        # pyrefly: ignore [bad-argument-type]
        dist.broadcast(data_recv_tensor, src=src_rank, group=group, async_op=False)
        buffer = io.BytesIO(data_recv_tensor.cpu().numpy())
        obj = torch.load(buffer, map_location=device, weights_only=False)
    return obj


class _ZeROJoinHook(JoinHook):
    def __init__(self, zero):
        if not isinstance(zero, ZeroRedundancyOptimizer):
            raise AssertionError(
                "ZeRO join hook requires passing in a "
                "ZeroRedundancyOptimizer instance as the state"
            )
        self.zero = zero
        super().__init__()

    def main_hook(self):
        """
        Perform an optimizer step.

        This step updates the joined process's shard of
        the parameters and broadcasts those parameters.
        """
        self.zero.step()


class _DDPBucketAssignment:
    r"""
    Represent a :class:`DistributedDataParallel` bucket assignment.

    This means that a (possibly non-strict) subset of the parameters corresponding to
    a DDP bucket assigned to a rank to update.

    Attributes:
        bucket_index (int): index of the bucket determined by the DDP gradient
            bucket all-reduce order.
        parameters (List[torch.Tensor]): model parameters in the bucket
            assigned to this rank.
        offset (int): offset into the :class:`GradBucket` 's :meth:`parameters`
            giving the index of the first element in the passed-in
            ``parameters``; this equivalently indexes into the
            :class:`GradBucket` 's :meth:`gradients`.
        device (torch.device): device on which the parameters are stored.
        tensor (torch.Tensor): flattened tensor giving the data of the
            parameter subset assigned to the rank.
    """

    def __init__(
        self,
        bucket_index: int,
        parameters: list[torch.Tensor],
        offset: int,
    ):
        self.bucket_index = bucket_index
        self.parameters = parameters
        self.offset = offset
        if len(self.parameters) == 0:
            raise ValueError("Empty bucket assignment")
        # DDP guarantees all parameters in the bucket have the same device
        self.device: torch.device = self.parameters[0].device
        self.tensor: torch.Tensor | None = None


class _OverlapStatus(enum.IntEnum):
    r"""
    Define possible statuses that :class:`ZeroRedundancyOptimizer` can be in when overlapping with :class:`DistributedDataParallel`.

    Attributes:
        ``UNINITIALIZED``: The ZeRO instance is effectively uninitialized and
            is waiting for DDP to finalize its bucketing.
        ``DDP_HAS_REBUILT_BUCKETS``: DDP has rebuilt its buckets, meaning that
            its bucketing is finalized. The ZeRO instance can now collect the
            necessary information about the DDP bucketing.
        ``INITIALIZED``: The ZeRO instance is fully initialized and can now
            optimize parameters.
    """

    UNINITIALIZED = 0
    DDP_HAS_REBUILT_BUCKETS = 1
    INITIALIZED = 2


class _OverlapInfo:
    r"""
    Information needed by :class:`ZeroRedundancyOptimizer` to overlap with :class:`DistributedDataParallel`.

    Arguments:
        world_size (int): world size of the process group being used.

    Attributes:
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity following
            a threshold given by the total parameter size divided by the world
            size; if ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank);
            this should be set to the value passed into the hook constructor.
        status (_OverlapStatus): current status; see :class:`_OverlapStatus`
            for more information.
        params_per_bucket (List[List[torch.Tensor]]): ``params_per_bucket[i]``
            gives the model parameters in the ``i``th bucket.
        params_per_rank (List[List[torch.Tensor]]): ``params_per_rank[i]``
            gives the model parameters assigned to the ``i``th rank, where the
            parameters are grouped by increasing bucket indices.
        offsets (Dict[int, int]): maps from bucket index to the offset in
            ``self.params_per_rank[rank]`` giving the index of the first
            parameter in that bucket, where ``rank`` is this process's own
            rank; the keys of this :class:`dict` are the bucket indices
            assigned to this rank.
        num_bucket_assignments (int): total number of bucket assignments across
            all ranks; this is equal to the number of
            :class:`DistributedDataParallel` gradient buckets if
            ``shard_buckets=False`` and possibly greater otherwise.
        total_size (int, optional): total size of all buckets (i.e. sum of
            ``param.numel()`` for all ``param`` across all buckets) if
            ``shard_buckets=True``; otherwise, ``None``.
        broadcast_handles (List[Work]): :class:`list` of async work handles for
            the parameter broadcasts.
        bucket_index_to_future (Dict[int, torch.futures.Future]):
            :class:`dict` mapping bucket index to the corresponding all-reduce
            future.
        bucket_index_to_bucket (Dict[int, dist.GradBucket]): :class:`dict`
            mapping bucket index to the corresponding bucket.
        bucket_indices_seen (List[int]): :class:`list` of the bucket indices
            seen on this iteration.
    """

    def __init__(self, world_size) -> None:
        self.status: _OverlapStatus = _OverlapStatus.UNINITIALIZED
        self.shard_buckets: bool = False

        # Modified per bucket reconstruction
        self.params_per_bucket: list[list[torch.Tensor]] = []
        self.params_per_rank: list[list[torch.Tensor]] = [[] for _ in range(world_size)]
        self.offsets: dict[int, int] = {}
        # Group Ranks
        self.assigned_ranks_per_bucket: list[set[int]] = []
        self.num_bucket_assignments: int = 0
        self.total_size: int | None = None

        # Modified per iteration
        self.broadcast_handles: list[Any] = []
        self.bucket_indices_seen: list[int] = []
        # Used by `hook_with_zero_step()`
        self.bucket_index_to_future: dict[int, torch.futures.Future] = {}
        self.bucket_index_to_bucket: dict[int, dist.GradBucket] = {}

    def wait_for_broadcasts(self) -> None:
        r"""
        Wait for all parameter broadcasts.

        This function should be called once all broadcasts have been scheduled,
        meaning ``self.broadcast_handles`` is filled. This clears ``self.broadcast_handles``
        in preparation for the next iteration.
        """
        if len(self.broadcast_handles) != self.num_bucket_assignments:
            raise AssertionError(
                f"Missing at least one broadcast handle on rank {dist.get_rank()}"
            )
        _ = [x.wait() for x in self.broadcast_handles]
        self.broadcast_handles.clear()

    def clear_per_iter_info(self) -> None:
        r"""
        Clear the data structures that are modified per-iteration.

        This function should be called at the end of an iteration.
        """
        self.bucket_indices_seen.clear()
        self.bucket_index_to_future.clear()
        self.bucket_index_to_bucket.clear()


class ZeroRedundancyOptimizer(Optimizer, Joinable):
    r"""
    Wrap an arbitrary :class:`optim.Optimizer <torch.optim.Optimizer>` and shards its states across ranks in the group.

    The sharing is done as described by `ZeRO <https://arxiv.org/abs/1910.02054>`_.

    The local optimizer instance in each rank is only
    responsible for updating approximately ``1 / world_size`` parameters and
    hence only needs to keep ``1 / world_size`` optimizer states. After
    parameters are updated locally, each rank will broadcast its parameters to
    all other peers to keep all model replicas in the same state.
    ``ZeroRedundancyOptimizer`` can be used in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel` to reduce per-rank peak
    memory consumption.

    ``ZeroRedundancyOptimizer`` uses a sorted-greedy algorithm to pack a number
    of parameters at each rank. Each parameter belongs to a single rank and is
    not divided among ranks. The partition is arbitrary and might not match the
    parameter registration or usage order.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.

    Keyword Args:
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
        process_group (``ProcessGroup``, optional): ``torch.distributed``
            ``ProcessGroup`` (default: ``dist.group.WORLD`` initialized by
            :meth:`torch.distributed.init_process_group`).
        parameters_as_bucket_view (bool, optional): if ``True``, parameters are
            packed into buckets to speed up communication, and ``param.data``
            fields point to bucket views at different offsets; if ``False``,
            each individual parameter is communicated separately, and each
            ``params.data`` stays intact (default: ``False``).
        overlap_with_ddp (bool, optional): if ``True``, :meth:`step` is
            overlapped with :class:`DistributedDataParallel` 's gradient
            synchronization; this requires (1) either a functional optimizer
            for the ``optimizer_class`` argument or one with a functional
            equivalent and (2) registering a DDP communication hook
            constructed from one of the functions in ``ddp_zero_hook.py``;
            parameters are packed into buckets matching those in
            :class:`DistributedDataParallel`, meaning that the
            ``parameters_as_bucket_view`` argument is ignored.
            If ``False``, :meth:`step` runs disjointly after the backward pass
            (per normal).
            (default: ``False``)
        **defaults: any trailing arguments, which are forwarded to the local
            optimizer.

    Example::

        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>> from torch.distributed.optim import ZeroRedundancyOptimizer
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
        >>> ddp = DDP(model, device_ids=[rank])
        >>> opt = ZeroRedundancyOptimizer(
        >>>     ddp.parameters(),
        >>>     optimizer_class=torch.optim.Adam,
        >>>     lr=0.01
        >>> )
        >>> ddp(inputs).sum().backward()
        >>> opt.step()

    .. warning::
        Currently, ``ZeroRedundancyOptimizer`` requires that all of the
        passed-in parameters are the same dense type.

    .. warning::
        If you pass ``overlap_with_ddp=True``, be wary of the following: Given
        the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``. To adjust for this, one option
        is to prepend dummy inputs.

    .. warning:: ZeroRedundancyOptimizer is experimental and subject to change.
    """

    def __init__(
        self,
        params,
        optimizer_class: type[Optimizer],
        process_group: Any | None = None,
        parameters_as_bucket_view: bool = False,
        overlap_with_ddp: bool = False,
        **defaults: Any,
    ):
        r"""Init."""
        # Perform type and assumption checks on the input parameters
        params = self._verify_and_init_params(params)
        self._verify_same_dense_param_type()

        # NOTE: The parent constructor uses `add_param_group()` which is
        # partially overloaded in ZeroRedundancyOptimizer, so we use the
        # `initialized` flag to dissociate the behaviour of `add_param_group()`
        # between the parent and child.
        self.initialized = False

        Optimizer.__init__(self, params, defaults)
        Joinable.__init__(self)
        # Now, all parameters are held in both `self._all_params` and
        # `self.param_groups`

        # Internal data structures (`_cache` indicates lazily evaluated)
        self._param_to_rank_cache: dict[torch.Tensor, int] = {}
        self._param_to_index_cache: dict[torch.Tensor, int] = {}
        self._partition_parameters_cache: list[list[dict]] = []
        self._index_to_param_cache: list[torch.Tensor] = []
        self._device_to_params_per_rank_cache: dict[
            torch.device, list[list[torch.Tensor]]
        ] = {}
        self._bucket_assignments_per_rank_cache: list[
            dict[int, _DDPBucketAssignment]
        ] = []
        self._is_trainable_mask = self._get_is_trainable_mask()

        # Default device for collective communication and buckets
        self._default_device = self._all_params[0].device

        self.process_group = (
            process_group if process_group is not None else dist.group.WORLD
        )
        self.world_size: int = dist.get_world_size(self.process_group)
        self.rank: int = dist.get_rank(self.process_group)
        self.global_rank: int = dist.distributed_c10d.get_global_rank(
            # pyrefly: ignore [bad-argument-type]
            self.process_group,
            self.rank,
        )

        self._overlap_with_ddp: bool = overlap_with_ddp
        self._optim_defaults = defaults
        self._optim_constructor = self._get_optimizer_constructor(optimizer_class)

        # If `overlap_with_ddp=True`, local optimizer initialization is delayed
        # to run time after the necessary information has been collected
        if not overlap_with_ddp:
            self._init_local_optimizer()
        else:
            self._overlap_info: _OverlapInfo = _OverlapInfo(self.world_size)
            if parameters_as_bucket_view:
                logger.warning(
                    "`parameters_as_bucket_view=True` will be ignored since "
                    "`overlap_with_ddp=True`; instead, a different bucketing "
                    "strategy will be used"
                )

        # `self._buckets` is used if `parameters_as_bucket_view=True`, in
        # which case parameter data is flattened into contiguous bucket tensors
        self.parameters_as_bucket_view = parameters_as_bucket_view
        self._buckets: list[list[torch.Tensor]] = []
        self._build_param_buckets()

        # Optional consolidated optimizer state, only populated if this rank
        # is the target in `consolidate_state_dict()`
        self._all_state_dicts: list[dict[str, Any]] = []

        self.initialized = True

    def _clear_cache(self) -> None:
        r"""Clear the cached data structures giving partition information."""
        self._partition_parameters_cache.clear()
        self._param_to_rank_cache.clear()
        self._index_to_param_cache.clear()
        self._param_to_index_cache.clear()
        self._device_to_params_per_rank_cache.clear()
        self._bucket_assignments_per_rank_cache.clear()

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        r"""
        Add a parameter group to the :class:`Optimizer` 's ``param_groups``.

        This can be useful when fine tuning a pre-trained network, as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.

        Arguments:
            param_group (dict): specifies the parameters to be optimized and
                group-specific optimization options.

        .. warning:: This method handles updating the shards on all partitions
            but needs to be called on all ranks. Calling this on a subset of
            the ranks will cause the training to hang because communication
            primitives are called depending on the managed parameters and
            expect all the ranks to participate on the same set of parameters.
        """
        if self.initialized and self._overlap_with_ddp:
            raise RuntimeError(
                "ZeroRedundancyOptimizer with `overlap_with_ddp=True` only "
                "supports a single parameter group"
            )

        super().add_param_group(param_group)
        # NOTE: The rest of the method assumes that the call to the parent's
        # `add_param_group()` appends the new parameter group and preserves
        # the previous parameter-group ordering

        if self.initialized:
            # Force a re-partitioning of the parameters
            self._clear_cache()
            param_groups = self._partition_parameters()[self.rank]
            # NOTE: All parameters in the old parameter groups should be
            # assigned to the same ranks so that the local optimizers do not
            # need to be reinitialized

            # Add the parameters assigned to this rank from the new parameter
            # group to the local optimizer, if any
            if len(param_groups) == len(self.optim.param_groups) + 1:
                self.optim.add_param_group(param_groups[-1])

            # Update the bucketing strategy accordingly
            if self.parameters_as_bucket_view:
                self._build_param_buckets()

    def consolidate_state_dict(self, to: int = 0) -> None:
        r"""
        Consolidate a list of ``state_dict`` s (one per rank) on the target rank.

        Arguments:
            to (int): the rank that receives the optimizer states (default: 0).

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt.

        .. warning:: This needs to be called on all ranks.
        """
        self._check_overlap_initialized()

        # Sync the exposed `param_groups` attributes to the local optimizer in
        # case they have been updated
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        # Pull the sharded state from all ranks and store them in rank order
        empty_messenger = torch.tensor(
            [0], dtype=torch.uint8, device=self._default_device
        )

        # NOTE: We wastefully use `broadcast()` (e.g. instead of `gather()`)
        # due to compatibility issues with NCCL backend; a possible follow-up
        # is to move all sharded state management to RPC RRef
        self._all_state_dicts = []
        for rank in range(self.world_size):
            global_rank = dist.distributed_c10d.get_global_rank(
                # pyrefly: ignore [bad-argument-type]
                self.process_group,
                rank,
            )
            if self.rank == to:
                # Consolidate all local `state_dict`s on this rank, storing on
                # CPU to save GPU memory
                if rank == self.rank:
                    # Directly append own optimizer state
                    self._all_state_dicts.append(
                        _recursive_copy_to_device(
                            self.optim.state_dict(),
                            non_blocking=True,
                            device=torch.device("cpu"),
                        )
                    )
                else:
                    # Receive the optimizer state from the source rank
                    local_state_dict = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )
                    self._all_state_dicts.append(
                        _recursive_copy_to_device(
                            local_state_dict,
                            non_blocking=True,
                            device=torch.device("cpu"),
                        )
                    )
            else:
                if rank == self.rank:
                    # Send the optimizer state to the target rank
                    _ = _broadcast_object(
                        self.optim.state_dict(),
                        src_rank=self.global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )
                elif rank != to:
                    # Discard the received object; `broadcast()` is used for
                    # compatibility reasons
                    _ = _broadcast_object(
                        empty_messenger,
                        src_rank=global_rank,
                        group=self.process_group,
                        device=self._default_device,
                    )

    def _verify_params_per_rank(
        self,
        params_per_rank: list[list[torch.Tensor]],
    ) -> None:
        r"""
        Verify ``params_per_rank`` for :meth:`_partition_parameters`.

        The verification is done by checking that ``params_per_rank`` has length equal
        to the world size and that it does not contain any parameters not passed into the
        :class:`ZeroRedundancyOptimizer` constructor.

        The parameters in ``params_per_rank`` being a strict subset of those
        passed into the constructor is valid since some parameters may be
        frozen.

        Raises:
            ValueError: if ``params_per_rank`` does not have length equal to
                the world size or if it contains a parameter that was not
                passed into the :class:`ZeroRedundancyOptimizer` constructor.
        """
        if len(params_per_rank) != self.world_size:
            raise ValueError(
                "`params_per_rank` must have length equal to the world size"
            )
        all_params_set = set(self._all_params)
        for params in params_per_rank:
            for param in params:
                if param not in all_params_set:
                    raise ValueError(
                        "Passing a new parameter in `params_per_rank` that "
                        "was not passed into the ZeroRedundancyOptimizer "
                        "constructor"
                    )

    def _partition_param_group(
        self, param_group: dict[str, Any], params_per_rank: list[list[torch.Tensor]]
    ) -> None:
        r"""
        Partition the parameter group ``param_group`` according to ``params_per_rank``.

        The partition will modify the ``self._partition_parameters_cache``. This method should
        only be used as a subroutine for :meth:`_partition_parameters`.

        Arguments:
            param_group (dict[str, Any]): a parameter group as normally defined
                in an optimizer state.
            params_per_rank (list[list[torch.Tensor]]): a :class:`list` of
                length world size containing :class:`list` s of parameters to
                assign to each rank.
        """
        for rank, params in enumerate(params_per_rank):
            rank_param_group = copy.copy(param_group)
            rank_param_group["params"] = params
            self._partition_parameters_cache[rank].append(rank_param_group)

    def _partition_parameters(
        self,
        params_per_rank: list[list[torch.Tensor]] | None = None,
    ) -> list[list[dict]]:
        r"""
        Partitions parameters across distributed data parallel ranks.

        Arguments:
            params_per_rank (list[list[torch.Tensor]], optional): a
                :class:`list` of length world size containing :class:`list` s
                of parameters to assign to each rank; this provides a way to
                specify a partition manually.
                If ``None``, the parameters are partitioned according to an
                internal algorithm.
                (default: ``None``)

        Returns:
            A :class:`list` where each element of the list contains the
            ``param_groups`` for a rank (which itself is a :class:`list` of
            :class:`dict`); element 0 corresponds to rank 0, etc.; each rank
            stores the ``param_groups`` for all ranks for the collective
            communication in :meth:`step`.

        Raises:
            ValueError: see :meth:`_validate_params_per_rank`.
            RuntimeError: if ``params_per_rank`` is not ``None`` and this
                :class:`ZeroRedundancyOptimizer` instance is using more than
                one parameter group.
        """
        if params_per_rank is None:
            # Partition the parameters optimizing for uniformity
            if len(self._partition_parameters_cache) == 0:
                self._partition_parameters_cache = [[] for _ in range(self.world_size)]
                sizes = [0] * self.world_size
                for param_group in self.param_groups:
                    param_group_params_per_rank: list[list] = [
                        [] for _ in range(self.world_size)
                    ]
                    # Sort the parameters by size (largest first)
                    params_sorted = sorted(
                        param_group["params"], key=lambda t: t.numel(), reverse=True
                    )
                    for param in params_sorted:
                        # Greedily add the parameter to rank with smallest size so far
                        rank = self._get_min_index(sizes)
                        param_group_params_per_rank[rank].append(param)
                        sizes[rank] += param.numel()
                    # Apply the constructed partition of the parameter group
                    self._partition_param_group(
                        param_group, param_group_params_per_rank
                    )

            return self._partition_parameters_cache

        # Partition the parameters according to `params_per_rank`
        if len(self._partition_parameters_cache) != 0:
            raise AssertionError(
                "Specifying `params_per_rank` should only be done when the "
                "parameters have not been partitioned yet"
            )
        if len(self.param_groups) != 1:
            raise RuntimeError(
                "Specifying `params_per_rank` only supports a single parameter group"
            )
        self._verify_params_per_rank(params_per_rank)
        self._partition_parameters_cache = [[] for _ in range(self.world_size)]

        # Apply the passed-in partition of the parameter group
        param_group = self.param_groups[0]
        self._partition_param_group(param_group, params_per_rank)

        return self._partition_parameters_cache

    @property
    def _param_to_rank(self) -> dict[torch.Tensor, int]:
        r""":class:`dict` mapping parameters to their assigned data parallel rank in the partition."""
        if len(self._param_to_rank_cache) == 0:
            for rank, param_groups in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        self._param_to_rank_cache[param] = rank
        return self._param_to_rank_cache

    @property
    def _param_to_index(self) -> dict[torch.Tensor, int]:
        r"""
        :class:`dict` mapping parameters to their indices in the global optimizer state.

        NOTE: This assumes that the global optimizer state's indexing (in
        ``state_dict``) follows a linear ordering over the parameter groups.
        """
        if len(self._param_to_index_cache) == 0:
            self._param_to_index_cache = {
                p: i
                for i, p in enumerate(
                    chain.from_iterable(g["params"] for g in self.param_groups)
                )
            }
        return self._param_to_index_cache

    @property
    def _index_to_param(self) -> list[torch.Tensor]:
        r"""List mapping parameter indices in the global optimizer scheme to the actual params."""
        if len(self._index_to_param_cache) == 0:
            self._index_to_param_cache = list(
                chain.from_iterable(g["params"] for g in self.param_groups)
            )
        return self._index_to_param_cache

    def _broadcast_params_from_rank(self, rank: int):
        r"""
        Broadcast the shard of parameters from a given rank to all other ranks asynchronously.

        Arguments:
            rank (int): the source rank.

        Returns:
            A :class:`list` of async work handles for the ``broadcast()`` s
            performed to synchronize the parameters.
        """
        if self._overlap_with_ddp:
            raise AssertionError(
                "`_broadcast_params_from_rank()` should not be used if "
                "`overlap_with_ddp=True`; instead, the broadcasting should "
                "happen in the DDP communication hook"
            )
        handles = []
        if self.parameters_as_bucket_view:
            for dev_i_buckets in self._buckets:
                bucket = dev_i_buckets[rank]
                global_rank = dist.distributed_c10d.get_global_rank(
                    # pyrefly: ignore [bad-argument-type]
                    self.process_group,
                    rank,
                )
                handles.append(
                    dist.broadcast(
                        tensor=bucket,
                        src=global_rank,
                        group=self.process_group,
                        async_op=True,
                    )
                )
        else:
            param_groups = self._partition_parameters()[rank]
            global_rank = dist.distributed_c10d.get_global_rank(
                # pyrefly: ignore [bad-argument-type]
                self.process_group,
                rank,
            )
            for param_group in param_groups:
                handles.extend(
                    dist.broadcast(
                        tensor=param.data,
                        src=global_rank,
                        group=self.process_group,
                        async_op=True,
                    )
                    for param in param_group["params"]
                )
        return handles

    def _sync_params(self):
        r"""
        Sync all parameter shards across the ranks.

        This rank sends its shard of the parameters to all other ranks and
        receives a shard from each other rank. This is done using
        ``broadcast()``. Parameters are sent bucket-by-bucket if
        ``parameters_as_bucket_view=True``and sent parameter-by-parameter
        otherwise.
        """
        handles = []
        for rank in range(self.world_size):
            handles.extend(self._broadcast_params_from_rank(rank))
        _ = [x.wait() for x in handles]

    @property
    def _device_to_params_per_rank(
        self,
    ) -> dict[torch.device, list[list[torch.Tensor]]]:
        r"""
        Return device parameters assigned per rank.

        :class:`dict` mapping each device to a :class:`list` of the per-rank parameter
        lists filtered to only include the parameters stored on that device.
        Each per-rank parameter list gives the parameters assigned to that rank
        to update.

        This is used for constructing the parameter buckets if
        ``parameters_as_bucket_view=True``.

        Let ``dev_i`` denote the ``i``th device for this rank. Then:
        ``dev_0`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_0``,
            rank 1's assigned parameters stored on ``dev_0``,
            ...
        ``dev_1`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_1``,
            rank 1's assigned parameters stored on ``dev_1``,
            ...
        ...
        """
        if not self.parameters_as_bucket_view:
            raise AssertionError(
                "`_device_to_params_per_rank` should only be used if "
                "`parameters_as_bucket_view=True`"
            )
        if len(self._device_to_params_per_rank_cache) == 0:
            for rank, param_groups in enumerate(self._partition_parameters()):
                for param_group in param_groups:
                    for param in param_group["params"]:
                        device = param.device
                        if device not in self._device_to_params_per_rank_cache:
                            self._device_to_params_per_rank_cache[device] = [
                                [] for _ in range(self.world_size)
                            ]
                        self._device_to_params_per_rank_cache[device][rank].append(
                            param
                        )
        return self._device_to_params_per_rank_cache

    def _get_min_index(
        self,
        values: list[int],
        disallowed_indices: set[int] | None = None,
    ) -> int:
        r"""
        Return ``values.index(min(values))``, except only uses one pass.

        It also excludes any indices in ``disallowed_indices`` if provided.

        Arguments:
            values: (List[int]): :class:`list` of values.
            disallowed_indices (Optional[set[int]]): indices that are
                disallowed from being the returned min index.
        """
        min_index = -1
        min_value = float("inf")
        for i, value in enumerate(values):
            if disallowed_indices and i in disallowed_indices:
                continue
            if value < min_value:
                min_value = value
                min_index = i
        if min_index < 0:
            raise AssertionError("All indices are disallowed")
        return min_index

    def _assign_bucket_subset_to_rank(
        self,
        bucket_index: int,
        bucket_params: list[torch.Tensor],
        bucket_offset: int,
        assigned_rank: int,
        assigned_ranks_per_bucket: list[set[int]],
    ) -> None:
        r"""
        Assign ``bucket_params`` to the rank with the least size assigned so far and collects relevant information.

        The model parameters given by ``bucket_params`` represents a (possibly non-strict)
        subset of the parameters corresponding to a :class:`DistributedDataParallel` bucket.

        Arguments:
            bucket_index (int): index of the :class:`DistributedDataParallel`
                gradient bucket.
            bucket_params (List[torch.Tensor]): subset of the parameters
                corresponding to the bucket to assign.
            bucket_offset (int): offset giving the index of the first element
                in ``bucket_params`` in the bucket's full parameter list.
            assigned_rank (int): group rank to assign to.
            assigned_ranks_per_bucket (list[set[int]]): :class:`set` of group ranks
                assigned to each bucket.
        """
        overlap_info = self._overlap_info
        if len(bucket_params) == 0:
            raise ValueError("Empty bucket assignment")
        params_per_rank = overlap_info.params_per_rank
        offsets = overlap_info.offsets

        self._bucket_assignments_per_rank_cache[assigned_rank][bucket_index] = (
            _DDPBucketAssignment(bucket_index, bucket_params, bucket_offset)
        )
        if self.global_rank == assigned_rank:
            offsets[bucket_index] = len(params_per_rank[assigned_rank])
        params_per_rank[assigned_rank].extend(bucket_params)
        assigned_ranks_per_bucket[bucket_index].add(assigned_rank)
        self._overlap_info.num_bucket_assignments += 1

    @property
    def _bucket_assignments_per_rank(self) -> list[dict[int, _DDPBucketAssignment]]:
        r"""
        Return DDP bucket parameters assigned per rank.

        :class:`list` of length world size consisting of :class:`dict` s
        mapping bucket indices to :class:`_DDPBucketAssignment` s for each
        rank.
        """
        if not self._overlap_with_ddp:
            raise AssertionError(
                "`_bucket_assignments_per_rank` only be used if `overlap_with_ddp=True`"
            )
        if len(self._bucket_assignments_per_rank_cache) > 0:
            return self._bucket_assignments_per_rank_cache

        overlap_info = self._overlap_info
        if overlap_info.status != _OverlapStatus.INITIALIZED:
            raise AssertionError

        self._bucket_assignments_per_rank_cache = [{} for _ in range(self.world_size)]
        params_per_bucket = overlap_info.params_per_bucket

        if overlap_info.shard_buckets:
            # Define the assignment threshold to approximate uniformity
            if overlap_info.total_size is None:
                raise AssertionError("`total_size` was not computed")
            threshold = overlap_info.total_size / self.world_size  # type: ignore[operator]
            size_per_rank = [0 for _ in range(self.world_size)]

        num_buckets = len(params_per_bucket)
        overlap_info.assigned_ranks_per_bucket = [set() for _ in range(num_buckets)]
        assigned_ranks_per_bucket = overlap_info.assigned_ranks_per_bucket
        if not overlap_info.shard_buckets:
            # Assign each DDP bucket entirely to a single rank
            for bucket_index, bucket_params in enumerate(params_per_bucket):
                if len(bucket_params) <= 0:
                    raise AssertionError("Empty bucket")
                assigned_rank = self._get_assigned_rank(bucket_index)
                self._assign_bucket_subset_to_rank(
                    bucket_index,
                    bucket_params,
                    0,
                    assigned_rank,
                    assigned_ranks_per_bucket,
                )
        else:
            # Assign each DDP bucket to possibly multiple ranks
            # Specifically, sort the DDP buckets by increasing size, and for
            # each bucket, iteratively assign the maximal unassigned subset
            # with size less than `threshold` to the rank with the least total
            # size so far -- each such assignment is represented by a
            # `_DDPBucketAssignment` instance and only contains parameters from
            # a single DDP bucket
            params_per_bucket_enum = sorted(
                enumerate(params_per_bucket), key=lambda x: sum(p.numel() for p in x[1])
            )
            for bucket_index, bucket_params in params_per_bucket_enum:
                if len(bucket_params) <= 0:
                    raise AssertionError("Empty bucket")
                bucket_offset = 0
                assignment_size = 0
                for param_index, param in enumerate(bucket_params):
                    param_numel = param.numel()
                    if (
                        # pyrefly: ignore [unbound-name]
                        assignment_size + param_numel >= threshold
                        and param_index > bucket_offset
                    ):
                        assigned_rank = self._get_min_index(
                            # pyrefly: ignore [unbound-name]
                            size_per_rank,
                            assigned_ranks_per_bucket[bucket_index],
                        )
                        # Include up to but not including the parameter that
                        # exceeded the threshold
                        self._assign_bucket_subset_to_rank(
                            bucket_index,
                            bucket_params[bucket_offset:param_index],
                            bucket_offset,
                            assigned_rank,
                            assigned_ranks_per_bucket,
                        )
                        # pyrefly: ignore [unbound-name]
                        size_per_rank[assigned_rank] += assignment_size
                        bucket_offset = param_index
                        assignment_size = 0
                    assignment_size += param_numel
                # Assign the remainder of the bucket so that no assignment
                # spans across two buckets
                assigned_rank = self._get_min_index(
                    # pyrefly: ignore [unbound-name]
                    size_per_rank,
                    assigned_ranks_per_bucket[bucket_index],
                )
                self._assign_bucket_subset_to_rank(
                    bucket_index,
                    bucket_params[bucket_offset:],
                    bucket_offset,
                    assigned_rank,
                    assigned_ranks_per_bucket,
                )
                # pyrefly: ignore [unbound-name]
                size_per_rank[assigned_rank] += assignment_size

        return self._bucket_assignments_per_rank_cache

    def _local_step(
        self,
        gradients: list[torch.Tensor | None] | None = None,
        closure: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> float | None:
        r"""
        Perform a single optimizer step without syncing parameters across ranks.

        Arguments:
            gradients (list[Optional[torch.Tensor]], optional): a :class:`list`
                of length equal to the number of parameters assigned to this
                rank containing gradient tensors or ``None`` as its elements;
                a ``None`` in the :class:`list` indicates that the
                corresponding parameter should not be updated.
                If the argument itself is ``None``, then all parameters are
                updated, and the gradients are assumed to be already populated.
                (default: ``None``)
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers and should be
                ``None`` if ``gradients`` is not ``None``; (default: ``None``)
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. warning::
            The argument ``gradients`` should only be specified (i.e. not
            ``None``) if ``overlap_with_ddp=True``, in which case
            :class:`ZeroRedundancyOptimizer` wraps a functional optimizer.
        """
        Join.notify_join_context(self)
        # Check if the model trainability has changed
        is_trainable_mask = self._get_is_trainable_mask()
        if is_trainable_mask != self._is_trainable_mask:
            if self._overlap_with_ddp:
                raise RuntimeError(
                    "ZeroRedundancyOptimizer with `overlap_with_ddp=True` "
                    "does not support changing parameter trainability at run "
                    "time"
                )
            logger.warning(
                "ZeroRedundancyOptimizer detected that the trainable "
                "parameters changed; rebuilding the parameter buckets if "
                "enabled"
            )
            self._build_param_buckets()
            self._is_trainable_mask = is_trainable_mask

        # Sync the exposed `param_groups` attributes to the local optimizer in
        # case they have been updated
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

        # Run the optimizer step on this shard only
        if gradients is None:
            loss = (
                self.optim.step(**kwargs)
                if closure is None
                else self.optim.step(closure=closure, **kwargs)
            )
        else:
            if not self._overlap_with_ddp:
                raise AssertionError(
                    "Specifying `gradients` should not "
                    "be used when `overlap_with_ddp=False`"
                )
            if closure is not None:
                raise AssertionError(
                    "`closure` is not supported when using a local functional optimizer"
                )
            loss = self.optim.step(gradients=gradients)

        # Sync any updated attributes in the local optimizer to the exposed
        # `param_groups`
        self._sync_param_groups(self.optim.param_groups, self.param_groups)

        return loss

    # pyrefly: ignore [bad-override]
    def step(
        self,
        closure: Callable[[], float] | None = None,
        **kwargs: Any,
    ) -> float | None:
        r"""
        Perform a single optimizer step and syncs parameters across all ranks.

        Arguments:
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers.
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. note:: Any extra parameters are passed to the base optimizer as-is.
        """
        if self._overlap_with_ddp:
            logger.warning(
                "`step()` should not be included in the training loop when "
                "`overlap_with_ddp=True`"
            )
            return None

        # Perform the local optimizer step
        loss = self._local_step(closure=closure, **kwargs)

        # Sync all of the updated parameter shards across the ranks
        self._sync_params()

        return loss

    def join_hook(self, **_kwargs: Any) -> JoinHook:
        r"""
        Return the ZeRO join hook.

        It enables training on uneven inputs by
        shadowing the collective communications in the optimizer step.

        Gradients must be properly set before this hook is called.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.

        This hook does not support any keyword arguments; i.e. ``kwargs`` is
        unused.
        """
        return _ZeROJoinHook(self)

    @property
    def join_device(self) -> torch.device:
        r"""Return default device."""
        return self._default_device

    @property
    def join_process_group(self) -> Any:
        r"""Return process group."""
        return self.process_group

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        r"""
        Load the state pertaining to the given rank from the input ``state_dict``, updating the local optimizer as needed.

        Arguments:
            state_dict (dict): optimizer state; should be an object returned
                from a call to :meth:`state_dict`.

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt.
        """
        self._check_overlap_initialized()

        for index, value in state_dict["state"].items():
            param = self._index_to_param[index]
            if self._param_to_rank[param] != self.rank:
                # Clear any state irrelevant to this rank
                state_dict["state"][index] = None
            else:
                # Load the parameter state to the local optimizer
                self.optim.state[param] = _recursive_copy_to_device(
                    value, non_blocking=True, device=param.device
                )
                # Force zero-dimensional tensors (like Adam "step") on CPU
                for state_name, state_value in self.optim.state[param].items():
                    if torch.is_tensor(state_value) and state_value.dim() == 0:
                        self.optim.state[param][state_name] = state_value.cpu()

        super().load_state_dict(state_dict)

        # Sync the input state with the exposed and local optimizer states
        self._sync_param_groups(state_dict["param_groups"], self.param_groups)
        self._sync_param_groups(self.param_groups, self.optim.param_groups)

    def state_dict(self) -> dict[str, Any]:
        r"""
        Return the last global optimizer state known to this rank.

        .. warning:
            If the state has not been consolidated to this rank, this raises a
            runtime error, and even if it has, the state may not be up-to-date,
            depending on when :meth:`consolidate_state_dict` was last called.

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt; or if this method is called without a preceding call
                to :meth:`consolidate_state_dict`.
        """
        self._check_overlap_initialized()

        if len(self._all_state_dicts) == 0:
            raise RuntimeError(
                "Optimizer state has not been consolidated on this rank. "
                f"Please call `consolidate_state_dict(to={self.rank})` on "
                "all ranks beforehand if you meant to save the global state."
            )

        # Get the possibly-stale global optimizer state that uses global
        # parameter indexing
        state_dict = super().state_dict()

        # Update the global optimizer state with local state information,
        # factoring in the translation from local to global indexing
        for rank, local_state_dict in enumerate(self._all_state_dicts):
            local_param_groups = local_state_dict["param_groups"]
            global_param_groups = self._partition_parameters()[rank]
            if len(local_param_groups) != len(global_param_groups):
                raise AssertionError(
                    "Mismatch between number of local and global parameter groups"
                )

            for local_param_group, global_param_group in zip(
                local_param_groups, global_param_groups
            ):
                # `local_param_group` stores local indices, while
                # `global_param_group` stores the tensors directly
                local_param_indices = local_param_group["params"]
                global_params = global_param_group["params"]

                if len(local_param_indices) != len(global_params):
                    raise AssertionError(
                        "Mismatch between number of local and global "
                        "parameters in parameter group"
                    )
                for local_param_index, global_param in zip(
                    local_param_indices, global_params
                ):
                    # Update the global parameter state, if any
                    if local_param_index in local_state_dict["state"]:
                        global_param_index = self._param_to_index[global_param]
                        state_dict["state"][global_param_index] = local_state_dict[
                            "state"
                        ][local_param_index]

        # Sort the parameters in the state
        state_dict["state"] = dict(sorted(state_dict["state"].items()))
        return state_dict

    @staticmethod
    def _sync_param_groups(
        src_param_groups: list[dict[Any, Any]],
        dst_param_groups: list[dict[Any, Any]],
    ) -> None:
        r"""
        Sync the attributes from the source parameter groups to the destination parameter groups.

        Example attributes include learning rate or scheduler attributes. The
        two parameter groups should have the same length (i.e. same number of
        parameter groups).

        Arguments:
            src_param_groups (list[dict]): parameter groups giving the
                attribute settings to copy.
            dst_param_groups (list[dict]): parameter groups giving the
                attribute settings to set.
        """
        if len(src_param_groups) != len(dst_param_groups):
            raise AssertionError(
                "Mismatch between number of source and destination parameter groups"
            )
        for src_param_group, dst_param_group in zip(src_param_groups, dst_param_groups):
            # Sync all attributes except the parameters
            for attr in filter(lambda x: x != "params", src_param_group.keys()):
                dst_param_group[attr] = src_param_group[attr]

    def _build_param_buckets(self) -> None:
        r"""
        Build parameter buckets if ``parameters_as_bucket_view=True``.

        For each device that stores this rank's parameters, there is a
        bucket (represented as a tensor) containing all of the parameters on
        that device that are assigned to a given rank in the parameter update
        partition.

        This method is called in the constructor and any time parameter
        trainability is changed.

        .. warning::
            The current implementation assumes that all of the parameters in a
            bucket are of the same dense type when allocating the bucket's
            tensor.

        .. warning::
            If the model parameters are stored across more than one device,
            then the storage partitioning must be the same across all
            processes in order for parameter synchronization to work.
        """
        if not self.parameters_as_bucket_view or self._overlap_with_ddp:
            return

        # `self._buckets[i][j]` are the parameters stored on device i and
        # assigned to rank j
        num_devices = len(self._device_to_params_per_rank)
        self._buckets = [[] for _ in range(num_devices)]  # type: ignore[assignment]

        for dev_i, (device, params_per_rank) in enumerate(
            self._device_to_params_per_rank.items()
        ):
            for params in params_per_rank:
                bucket_size = 0
                dtype = None
                trainable_params = []
                for param in params:
                    if not _is_trainable(param):
                        # Clone in case the parameter was previously part of
                        # a bucket to avoid the data from being destroyed
                        param.data = param.data.detach().clone()
                    else:
                        bucket_size += param.numel()
                        trainable_params.append(param)
                    dtype = param.dtype  # assumes all same dtype

                if bucket_size == 0:
                    # Create a dummy bucket if there are no parameters
                    bucket = torch.zeros(1, device=device)
                else:
                    # Construct the bucket (assuming all dense and same dtype)
                    bucket = torch.empty(bucket_size, dtype=dtype, device=device)
                    offset = 0
                    for param in trainable_params:
                        offset_next = offset + param.numel()
                        bucket[offset:offset_next].copy_(param.data.flatten())
                        param.data = bucket[offset:offset_next].view_as(param.data)
                        offset = offset_next
                self._buckets[dev_i].append(bucket)  # type: ignore[arg-type]

    def _build_ddp_param_buckets(self) -> None:
        r"""
        Build the DDP bucket with parameters assigned to this rank.

        For each DDP bucket with parameters assigned to this rank, flattens the
        data of those parameters into a single tensor and saves the tensor to
        the ``tensor`` attribute in the corresponding
        :class:`_DDPBucketAssignment` instance stored in
        ``self._bucket_assignments_per_rank``.

        :class:`DistributedDataParallel` guarantees that the parameters
        corresponding to a gradient bucket have the same device and the same
        dtype.
        """
        for bucket_assignments in self._bucket_assignments_per_rank:
            for bucket_assignment in bucket_assignments.values():
                params = bucket_assignment.parameters
                bucket_size = 0
                dtype = None
                for param in params:
                    if not _is_trainable(param):
                        raise AssertionError(
                            "Model parameter "
                            "corresponding to a gradient in a DDP bucket should "
                            "require a gradient"
                        )
                    bucket_size += param.numel()
                    dtype = param.dtype  # assumes all same dtype
                if bucket_size <= 0:
                    raise AssertionError("Empty bucket")

                # Construct the bucket tensor (assuming all dense and same dtype)
                tensor = torch.empty(
                    bucket_size, dtype=dtype, device=bucket_assignment.device
                )
                offset = 0
                for param in params:
                    offset_next = offset + param.numel()
                    tensor[offset:offset_next].copy_(param.data.flatten())
                    param.data = tensor[offset:offset_next].view_as(param.data)
                    offset = offset_next
                bucket_assignment.tensor = tensor

    def _verify_and_init_params(
        self,
        params: Any,
    ) -> list[torch.Tensor] | list[dict]:
        r"""
        Verify the type of ``params`` and initializes ``self._all_params`` as a :class:`list` of all parameters.

        The initialization will first make sure that provided ``params`` is valid.

        Arguments:
            params (Any): Candidate parameter list or parameter groups to verify.

        Raises:
            TypeError: ``params`` has an invalid type.
            ValueError: ``params`` is empty.

        Returns:
            The persistent form of ``params`` to be passed into the parent
            :class:`Optimizer` constructor -- i.e. returns ``params`` as a
            :class:`list` to ensure that it can be iterated over again.
        """
        if isinstance(params, torch.Tensor):
            raise TypeError(
                "`params` argument should be an iterable of "
                f"Tensors, but got {torch.typename(params)}"
            )
        try:
            all_params = list(params)
        except TypeError as e:
            raise TypeError(
                "`params` argument should be an iterable of Tensors"
                f" or dicts, but got {torch.typename(params)}"
            ) from e
        if len(all_params) == 0:
            raise ValueError("ZeroRedundancyOptimizer got an empty parameter list")
        all_tensors = True
        all_dicts = True
        for param in all_params:
            all_tensors &= isinstance(param, torch.Tensor)
            all_dicts &= isinstance(param, dict)
        if not all_tensors and not all_dicts:
            raise TypeError(
                "`params` argument should be an iterable of Tensors or dicts"
            )
        # Ensure that `self._all_params` contains a list of all parameters
        if all_tensors:
            self._all_params = all_params
        elif all_dicts:
            self._all_params = []
            # `all_params` contains parameter groups (not parameters)
            for param_group in all_params:
                if "params" not in param_group:
                    raise ValueError(
                        "Each parameter group passed-in via `params` must "
                        "have a 'params' key mapping to the parameters in "
                        "the group"
                    )
                self._all_params.extend(param_group["params"])
        return all_params

    def _verify_same_dense_param_type(self) -> None:
        r"""
        Verify that all parameters are of the same dense type.

        The method assumes that ``self._all_params`` has been initialized
        and is non-empty.

        Raises:
            ValueError: ``params`` contains sparse parameters or parameters
            of varying dense types.

        NOTE: This method can be removed once support for sparse parameters
        and varying parameter types is added.
        """
        typename = torch.typename(self._all_params[0])
        if self._all_params[0].is_sparse:
            raise ValueError(
                "ZeroRedundancyOptimizer only supports using "
                "the same dense type for all parameters but got "
                f"{typename}"
            )
        for param in self._all_params[1:]:
            other_typename = torch.typename(param)
            if other_typename != typename:
                raise ValueError(
                    "ZeroRedundancyOptimizer only supports "
                    "using the same dense type for all "
                    f"parameters but got both {typename} and "
                    f"{other_typename}"
                )

    def _get_is_trainable_mask(self) -> list[bool]:
        r"""Return a boolean mask indicating if each parameter is trainable (``requires_grad``) or not."""
        return list(map(_is_trainable, self._all_params))

    def _init_local_optimizer(self) -> None:
        r"""
        Initialize this rank's local optimizer, responsible for its subset of the parameters.

        The local optimizer is saved in ``self.optim``.
        """
        if self._optim_constructor is None:
            raise AssertionError("The local optimizer class has not been set")

        param_groups = self._partition_parameters()[self.rank]
        # `overlap_with_ddp=True` requires a local functional optimizer
        if self._overlap_with_ddp:
            # Functional optimizers only support a single parameter group and
            # require passing in the parameters as a list
            if len(param_groups) != 1:
                raise AssertionError(
                    "Initializing the local functional optimizer "
                    "with more than one parameter group"
                )
            params = param_groups[0]["params"]
            # Try to pass `_allow_empty_param_list=True` to avoid erroring
            if (
                "_allow_empty_param_list"
                in inspect.signature(self._optim_constructor).parameters
            ):
                self.optim: Any = self._optim_constructor(
                    params, **self._optim_defaults, _allow_empty_param_list=True
                )
            else:
                logger.warning(
                    "%s does not support the argument "
                    "`_allow_empty_param_list`; ZeroRedundancyOptimizer may "
                    "error due to an empty parameter list",
                    self._optim_constructor,
                )
                self.optim: Any = self._optim_constructor(
                    params, **self._optim_defaults
                )  # type: ignore[no-redef]

            # Log information about the DDP and ZeRO bucketing
            if dist.get_debug_level() != dist.DebugLevel.OFF:
                local_numel = sum(p.numel() for p in params)
                num_assigned_buckets = len(
                    self._bucket_assignments_per_rank[self.global_rank]
                )
                logger.info(
                    "rank %s with %s parameters across %s buckets",
                    self.global_rank,
                    local_numel,
                    num_assigned_buckets,
                )
                if self.global_rank == 0:
                    logger.info(
                        "%s DDP buckets and %s bucket assignments",
                        len(self._overlap_info.params_per_bucket),
                        self._overlap_info.num_bucket_assignments,
                    )
        else:
            # NOTE: Passing `param_groups` into the local optimizer constructor
            # bypasses the empty parameter list check
            self.optim: Optimizer = self._optim_constructor(
                param_groups, **self._optim_defaults
            )  # type: ignore[no-redef]

        # TODO: Manually add `self.param_groups` if using a functional
        # optimizer; remove this if/when the functional optimizers support
        # multiple parameter groups
        if self._overlap_with_ddp and not hasattr(self.optim, "param_groups"):
            if not hasattr(self.optim, "param_group"):
                raise AssertionError(
                    "The functional optimizer should set at least one of "
                    "the attributes `param_group` or `param_groups`"
                )
            self.optim.param_groups = [self.optim.param_group]  # type: ignore[attr-defined]

        self._sync_param_groups(self.optim.param_groups, self.param_groups)

    def _init_zero_for_overlap(self) -> None:
        r"""Perform a delayed initialization of the local optimizer and the supporting data structures."""
        if not self._overlap_with_ddp:
            raise AssertionError(
                "`_init_zero_for_overlap()` should only be called when "
                "`overlap_with_ddp=True`"
            )
        self._overlap_info.status = _OverlapStatus.INITIALIZED
        self._clear_cache()
        self._partition_parameters(self._overlap_info.params_per_rank)
        self._build_ddp_param_buckets()
        self._init_local_optimizer()

    def _get_assigned_rank(self, bucket_index: int) -> int:
        r"""
        Return the single rank assigned to a :class:`DistributedDataParallel` gradient bucket.

        Arguments:
            bucket_index (int): index of the :class:`DistributedDataParallel`
                bucket for which to get the assigned rank.
        """
        if self._overlap_info.shard_buckets:
            raise AssertionError(
                "The bucket assignment requires global bucket information "
                "and will be computed later; there should be no need to "
                "use this method"
            )
        return bucket_index % self.world_size

    def _check_overlap_initialized(self):
        r"""
        Check the delayed initialization depending on the value of ``overlap_with_ddp``.

        The delayed initialization has occurred (see
        :meth:`_init_zero_for_overlap`) if ``overlap_with_ddp=True``, and
        raises a ``RuntimeError`` if not. This should preface methods that
        should not be run before that delayed initialization.

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and
                :meth:`_init_zero_for_overlap` has not been called.
        """
        if (
            self._overlap_with_ddp
            and self._overlap_info.status != _OverlapStatus.INITIALIZED
        ):
            raise RuntimeError(
                "This method should not be called until this "
                "ZeroRedundancyOptimizer instance has been fully "
                "initialized"
            )

    def _get_optimizer_constructor(self, optimizer_class: Any) -> Any:
        r"""
        Return the optimizer constructor using validation and transformation depending on ``overlap_with_ddp``.

        Returns:
            - ``optimizer_class`` if ``overlap_with_ddp=False`` and
                ``optimizer_class`` is not a functional optimizer.
            - ``optimizer_class`` if ``overlap_with_ddp=True`` and
                ``optimizer_class`` is already a functional optimizer.
            - The functional equivalent of ``optimizer_class`` if
                ``overlap_with_ddp=True`` and ``optimizer_class`` is not
                already a functional optimizer (assuming the equivalent
                exists).

        Raises:
            ValueError:

                - if ``overlap_with_ddp=True`` but ``optimizer_class`` is
                    neither a functional optimizer nor translatable to a
                    functional optimizer.
                - if ``overlap_with_ddp=False`` and ``optimizer_class`` is a
                    functional optimizer.
        """
        functional_optims = functional_optim_map.values()
        if not self._overlap_with_ddp:
            if optimizer_class in functional_optims:
                # Using a functional optimizer is only supported when
                # `overlap_with_ddp=True`
                raise ValueError(
                    f"Passing in a functional optimizer {optimizer_class} "
                    "when `overlap_with_ddp=False`"
                )
            else:
                return optimizer_class
        else:
            if optimizer_class in functional_optims:
                # Already a functional optimizer
                return optimizer_class
            elif optimizer_class in functional_optim_map:
                # Translate the passed-in optimizer class to its functional
                # equivalent if `overlap_with_ddp=True`
                optim_constructor = functional_optim_map[optimizer_class]
                logger.info(
                    "Using the functional optimizer %s "
                    "instead of %s since "
                    "`overlap_with_ddp=True`",
                    optim_constructor,
                    optimizer_class,
                )
                return optim_constructor
            else:
                raise ValueError(
                    "Using `ddp_with_overlap=True` requires using a "
                    "functional optimizer, but there is no supported functional "
                    f"optimizer equivalent for {optimizer_class}"
                )
<|endoftext|>import logging
import warnings
from collections.abc import Callable, Collection, Mapping
from copy import deepcopy
from typing import Any, overload

import torch
import torch.nn as nn
from torch import optim
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


__all__: list[str] = []

logger = logging.getLogger(__name__)


class _NamedOptimizer(optim.Optimizer):
    """
    ``_NamedOptimizer`` takes a dict of parameters and exposes ``state_dict`` by parameter key.

    We replace the original key (number) in an optim to the
    fully qualified name (FQN) string. User can initialize the optim as they
    initialize a PyTorch optim, the only difference is that they also need to
    pass in the FQN of each parameters.

    Args:
        named_parameters (Mapping[str, Union[torch.Tensor, ShardedTensor]]):
            Mapping from FQN to parameter.
        optimizer_class (optim.Optimizer):
            The class of optimizer to instantiate.
        param_groups (Collection[Mapping[str, Any]]):
            `param_groups` to pass to optimizer if specified.
            The key of the inner map needs to be FQNs.
            Default: None
        module (nn.Module): the module whose parameters to be updated
            by the optimizer.
        args: arguments to pass to the optimizer constructor.
        kwargs: arguments to pass to the optimizer constructor.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch import optim
        >>> from torch.distributed.optim import _NamedOptimizer
        >>>
        >>> # Define the named optimizer.
        >>> m = Model(...)
        >>> named_optim = _NamedOptimizer(m.named_parameters(), optim.SGD)
        >>> # Forward pass + backward pass.
        >>> named_optim.step()
        >>> ...
        >>> # Call state_dict for the named optimizer returns an FQN state_dict.
        >>> named_optim.state_dict()

    Warning: This API is still in development and subject to change.

    TODO: Add tutorial for _NamedOptimizer.
    TODO: Add documentation in the docstring for the public attributes
          like self.param_groups and self.named_parameters.
    """

    def __init__(
        self,
        named_parameters: Mapping[str, torch.Tensor | ShardedTensor],
        optimizer_class: optim.Optimizer,
        param_groups: Collection[Mapping[str, Any]] | None = None,
        module: nn.Module | None = None,
        *args: tuple[Any, ...],
        **kwargs: dict[str, Any],
    ) -> None:
        torch._C._log_api_usage_once("torch.distributed.optim._NamedOptimizer")
        self.param_groups: Collection[Mapping[str, Any]] = param_groups  # type: ignore[assignment]
        self._param_groups_check()
        self.named_parameters = dict(named_parameters)
        params_for_optimizer = (
            self.named_parameters.values() if param_groups is None else param_groups
        )
        self._optimizer = optimizer_class(  # type: ignore[operator]
            params_for_optimizer,
            *args,
            **kwargs,
        )
        self.module = module
        if param_groups is None:
            self.ordered_param_keys = list(self.named_parameters.keys())
        else:
            warnings.warn(
                "Since we pass in param_groups, we will use param_groups to "
                "initialize the optimizer, not all parameters of the module.",
                stacklevel=2,
            )
            param_to_key = {param: key for key, param in self.named_parameters.items()}  # type: ignore[misc, has-type]
            ordered_param_keys = []
            for group in param_groups:
                for param in group["params"]:
                    if param not in param_to_key:
                        raise ValueError(
                            f"Expect param name {param} found in param group but is missing."
                        )
                    ordered_param_keys.append(param_to_key[param])
            self.ordered_param_keys = ordered_param_keys
        # Update param_groups from optimizer.
        self.param_groups = self._optimizer.param_groups

    def _param_groups_check(self) -> None:
        if self.param_groups is not None:
            for param_group in self.param_groups:
                if not isinstance(param_group, dict):
                    raise AssertionError("param group must be a dict")
                if "params" not in param_group:
                    raise AssertionError("param group must contain key params")
                params = param_group["params"]
                if isinstance(params, torch.Tensor):
                    params = [params]
                params = list(params)
                for param in params:
                    if not isinstance(param, torch.Tensor):
                        raise TypeError(
                            "optimizer can only optimize Tensors, "
                            "but one of the params is " + torch.typename(param)
                        )
                param_group["params"] = params

    def state_dict(self) -> dict[str, Any]:
        """
        Return the ``state_dict`` of the optimizer.

        Instead of using number to index
        parameters, we will use module fully qualified name (FQN) as the key.
        """
        state_dict = self._optimizer.state_dict()
        param_groups = state_dict["param_groups"]

        ret_state = {
            self.ordered_param_keys[st_key]: state_val
            for st_key, state_val in state_dict["state"].items()
        }

        ret_groups = []
        for group in param_groups:
            param_keys = [self.ordered_param_keys[param] for param in group["params"]]
            ret_group = {"params": sorted(param_keys)}
            for k, v in group.items():
                if k != "params":
                    ret_group[k] = deepcopy(v)
            ret_groups.append(ret_group)

        return self._post_state_dict({"state": ret_state, "param_groups": ret_groups})

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """
        Perform a single optimization step.

        This will call :meth:`torch.optim.Optimizer.step` on the wrapped
        optimizer.
        """
        return self._optimizer.step(closure=closure)

    @property
    def state(self) -> Mapping[torch.Tensor, Any]:  # type: ignore[override]
        return self._optimizer.state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Define the default behavior to load a state_dict for ``_NamedOptimizer``.

        Sample Code
        ```
            my_model = MyModule()
            optimizer = _NamedOptimizer(my_model.named_parameters(), Adagrad)
            ...

            optim_state_dict = optimizer.state_dict()
            ...
            ...

            optimizer.load_state_dict(optim_state_dict)
            ...
        ```
        Args:
            state_dict (dict[str, Any]) : A ``state_dict`` to load into the optimizer.
                Note that this state dict update is performed in place.

        .. note:: PyTorch is using lazy init to initialize the optim states.
            So it is possible that there is no optim state when user call
            ``load_state_dict`` and for ``_NamedOptimizer`` we make it stricter
            that users can only call ``load_state_dict`` after the state is initialized.
            By doing this, we can validate the optim ``state_dict`` to be loaded.
        """
        new_state_dict = self._optimizer.state_dict()
        state_dict = self._pre_load_state_dict(state_dict)
        state = state_dict["state"]
        new_state = new_state_dict["state"]
        if len(new_state) == 0:
            raise ValueError(
                "Expects the optim to be initialized before load but found not initialized."
            )

        for idx, param_key in enumerate(self.ordered_param_keys):
            # When the conditional training is performed, not all parameters are updated in the optim.
            if param_key not in state:
                continue
            if len(state[param_key]) != len(new_state[idx]):
                raise ValueError(
                    f"Expects equal length as {len(new_state[idx])} for parameter {param_key} but found: {len(state[param_key])}"
                )
            # Iterate through all optimizer states.
            for state_key, state_val in new_state[idx].items():
                if state_key not in state[param_key]:
                    raise ValueError(
                        f"Expects state {state_key} for parameter {param_key} but not found."
                    )

                src_state_val = state[param_key][state_key]
                if isinstance(state_val, ShardedTensor):
                    if not isinstance(src_state_val, ShardedTensor):
                        raise AssertionError
                    num_shards = len(state_val.local_shards())
                    num_new_shards = len(src_state_val.local_shards())
                    if num_shards != num_new_shards:
                        raise ValueError(
                            f"Expects equal number of shards as {num_new_shards} but found {num_shards} for {param_key}/{state_key}"
                        )
                    for shard, src_shard in zip(
                        state_val.local_shards(), src_state_val.local_shards()
                    ):
                        shard.tensor.detach().copy_(src_shard.tensor)
                elif isinstance(state_val, torch.Tensor):
                    if not isinstance(src_state_val, torch.Tensor):
                        raise AssertionError
                    state_val.detach().copy_(src_state_val)
                else:
                    new_state[idx][state_key] = deepcopy(src_state_val)

        # Load param_groups of state_dict
        src_param_groups = state_dict["param_groups"]
        new_param_groups = new_state_dict["param_groups"]

        src_group_map = {}
        for group in src_param_groups:
            param_keys = list(group["params"])
            src_group_map[_gen_param_group_key(param_keys)] = group
        new_group_map = {}
        for new_group in new_param_groups:
            param_keys = []
            for param_key in new_group["params"]:
                param_keys.append(self.ordered_param_keys[param_key])  # type: ignore[call-overload]
            new_group_map[_gen_param_group_key(param_keys)] = new_group
        for group_key, new_group in new_group_map.items():
            # When not all parameters are used in training or receive gradient, aka., not all parameters
            # would be in the param_group. Thus we skip the group_key here.
            if group_key not in src_group_map:
                continue
            src_group = src_group_map[group_key]
            if len(src_group) != len(new_group):
                raise ValueError(
                    f"Expects equal param_group size as {len(new_group)} for group {group_key} but found {len(src_group)}."
                )
            for k in src_group:
                if k not in new_group:
                    raise ValueError(
                        f"Expects group key {k} to be in group {group_key} in `state_dict` but is missing."
                    )
                if k != "params":
                    new_group[k] = deepcopy(src_group[k])

        self._optimizer.load_state_dict(new_state_dict)

    def add_param_group(self, param_group: Mapping[str, Any]) -> None:
        """
        Add a param group to the :class:`_NamedOptimizer` s `param_groups`.

        Warning: This API is still in development and subject to change.
        """
        if not isinstance(param_group, dict):
            raise AssertionError("param group must be a dict")

        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        else:
            param_group["params"] = list(params)

        param_to_key = {param: key for key, param in self.named_parameters.items()}  # type: ignore[misc, has-type]
        for param in param_group["params"]:
            if param not in param_to_key:
                raise ValueError("some parameters are not in the module")
            self.ordered_param_keys.append(param_to_key[param])

        self._optimizer.add_param_group(param_group)
        # Update param_groups from optimizer.
        self.param_groups = self._optimizer.param_groups

    def init_state(self) -> None:
        """
        Run a dummy optimizer step, which allows us to initialize optimizer state because we do lazy init for most optimizers.

        This allows doing in-place loading of optimizer state from a checkpoint.
        """
        for param in self.named_parameters.values():
            if param.requires_grad:
                t = torch.zeros_like(param)
                param.grad = torch.autograd.Variable(t)
        # Calling ``step`` will load the initial state for optimizer states.
        self.step(closure=None)

    def _pre_load_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        # TODO(chienchin): This API should be FSDP agnostic and should support
        # general user hooks.
        if isinstance(self.module, FSDP):
            return FSDP.optim_state_dict_to_load(
                self.module, self._optimizer, state_dict, is_named_optimizer=True
            )
        return state_dict

    def _post_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        # TODO(chienchin): This API should be FSDP agnostic and should support
        # general user hooks.
        if isinstance(self.module, FSDP):
            FSDP.optim_state_dict(self.module, self._optimizer, state_dict)
        return state_dict


def _gen_param_group_key(param_keys: list[str]) -> str:
    """Concatenate all param keys as a unique identifier for one param group."""
    return "/".join(sorted(param_keys))
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.optim._functional as F
from torch import Tensor
from torch.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional Adadelta Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalAdadelta:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
            "rho": rho,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        self.foreach = foreach
        self.maximize = maximize

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

    def step(self, gradients: list[Tensor | None]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        square_avgs = []
        acc_deltas = []
        state_steps = []
        lr = self.defaults["lr"]
        rho = self.defaults["rho"]
        eps = self.defaults["eps"]
        weight_decay = self.defaults["weight_decay"]

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )
        has_complex = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    state["square_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["acc_delta"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                state = self.state[param]
                square_avgs.append(state["square_avg"])
                acc_deltas.append(state["acc_delta"])
                state_steps.append(state["step"])

        with torch.no_grad():
            F.adadelta(
                params_with_grad,
                grads,
                square_avgs,
                acc_deltas,
                state_steps,
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.optim._functional as F
from torch import Tensor
from torch.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional Adamax Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalAdamax:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }
        self.foreach = foreach
        self.maximize = maximize
        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

    def step(self, gradients: list[Tensor | None]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_infs = []
        state_steps: list[Tensor] = []

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_complex = False
        for param, gradient in zip(self.param_group["params"], gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_inf"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )

                state = self.state[param]

                exp_avgs.append(state["exp_avg"])
                exp_infs.append(state["exp_inf"])
                state_steps.append(state["step"])

        with torch.no_grad():
            F.adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=self.defaults["eps"],
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.optim._functional as F
from torch import Tensor
from torch.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional Adam Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalAdam:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        fused: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.foreach = foreach
        self.fused = fused
        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

    def step_param(self, param: Tensor, grad: Tensor | None):
        """
        Similar to step, but operates on a single parameter and optionally a
        gradient tensor.
        """
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps: list[Tensor] = []
        has_complex = torch.is_complex(param)
        if grad is not None:
            params_with_grad.append(param)
            grads.append(grad)
        if param not in self.state:
            self.state[param] = {}
            state = self.state[param]
            state["step"] = torch.tensor(0.0)
            state["exp_avg"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            state["exp_avg_sq"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            if self.amsgrad:
                state["max_exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )

        state = self.state[param]
        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

        if self.amsgrad:
            max_exp_avg_sqs.append(state["max_exp_avg_sq"])

        state_steps.append(state["step"])
        with torch.no_grad():
            F.adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=self.amsgrad,
                has_complex=has_complex,
                maximize=self.maximize,
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                eps=self.defaults["eps"],
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )

    def step(self, gradients: list[Tensor | None]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps: list[Tensor] = []
        has_complex = False

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        for param, gradient in zip(self.param_group["params"], gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    if self.amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                state = self.state[param]

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if self.amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state_steps.append(state["step"])

        with torch.no_grad():
            F.adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=self.amsgrad,
                has_complex=has_complex,
                maximize=self.maximize,
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                eps=self.defaults["eps"],
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.optim._functional as F
from torch import Tensor
from torch.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional Rprop Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalRprop:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        etas: tuple[float, float] = (0.5, 1.2),
        step_sizes: tuple[float, float] = (1e-6, 50),
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
        }
        self.etas = etas
        self.step_sizes = step_sizes
        self.foreach = foreach
        self.maximize = maximize

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

    def step(self, gradients: list[Tensor | None]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        prevs = []
        step_sizes = []
        state_steps = []
        lr = self.defaults["lr"]
        etaminus, etaplus = self.etas
        step_size_min, step_size_max = self.step_sizes

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_complex = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    state["prev"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    state["step_size"] = torch.full_like(gradient, lr)

                state = self.state[param]
                prevs.append(state["prev"])
                step_sizes.append(state["step_size"])
                state_steps.append(state["step"])

        with torch.no_grad():
            F.rprop(
                params_with_grad,
                grads,
                prevs,
                step_sizes,
                state_steps,
                step_size_min=step_size_min,
                step_size_max=step_size_max,
                etaminus=etaminus,
                etaplus=etaplus,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
<|endoftext|># mypy: allow-untyped-defs
import logging
from collections import defaultdict
from threading import Lock

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.jit as jit
from torch import Tensor
from torch.distributed.rpc import RRef

from .utils import functional_optim_map


__all__ = ["DistributedOptimizer"]

logger = logging.getLogger(__name__)


class _FunctionalLocalOptimizer:
    def __init__(self, optim_cls, local_params_rref, *args, **kwargs):
        self._local_params = [rref.local_value() for rref in local_params_rref]
        self.optim = optim_cls(self._local_params, *args, **kwargs)

    def step(self, autograd_ctx_id: int):
        all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)
        # apply functional optimizer step with a list of gradients
        grads: list[Tensor | None] = [
            all_local_grads[p] if p in all_local_grads else None  # noqa: SIM401
            for p in self._local_params
        ]

        self.optim.step(grads)


# TODO (wanchaol): remove/merge this with _FunctionalLocalOptimizer once
# we have converted all to functional optimizer in distributed.optim
class _LocalOptimizer:
    # Ideally we would only need to share a lock for instances of
    # _LocalOptimizer that deal with the same parameters. We are
    # making a simplifying assumption here that if there is more
    # than one instance of _LocalOptimizer per worker, they will
    # be optimizing the same parameters (e.g. each data parallel
    # trainer will create its own instance of _LocalOptimizer but
    # they will all optimize the same parameters on each worker)
    global_lock = Lock()

    def __init__(self, optim_cls, local_params_rref, *args, **kwargs):
        self._local_params = [rref.local_value() for rref in local_params_rref]
        self.optim = optim_cls(self._local_params, *args, **kwargs)

    def step(self, autograd_ctx_id):
        all_local_grads = dist_autograd.get_gradients(autograd_ctx_id)

        with _LocalOptimizer.global_lock:
            for param, grad in all_local_grads.items():
                param.grad = grad
            self.optim.step()


def _new_local_optimizer(optim_cls, local_params_rref, *args, **kwargs):
    return rpc.RRef(_LocalOptimizer(optim_cls, local_params_rref, *args, **kwargs))


def _local_optimizer_step(local_optim_rref, autograd_ctx_id):
    local_optim = local_optim_rref.local_value()
    local_optim.step(autograd_ctx_id)


def _new_functional_local_optimizer(optim_cls, local_params_rref, *args, **kwargs):
    optim = _FunctionalLocalOptimizer(optim_cls, local_params_rref, *args, **kwargs)
    return rpc.RRef(optim)


def _functional_local_optimizer_step(
    local_optim_rref: RRef, autograd_ctx_id: int
) -> None:
    local_optim = local_optim_rref.local_value()
    local_optim.step(autograd_ctx_id)


def _wait_for_all(rpc_futs):
    # TODO: improve error propagation
    exception = None
    results = []
    for fut in rpc_futs:
        try:
            results.append(fut.wait())
        except Exception as e:
            results.append(e)
            exception = e
    if exception is not None:
        raise exception
    return results


class DistributedOptimizer:
    """
    DistributedOptimizer takes remote references to parameters scattered
    across workers and applies the given optimizer locally for each parameter.

    This class uses :meth:`~torch.distributed.autograd.get_gradients` in order
    to retrieve the gradients for specific parameters.

    Concurrent calls to
    :meth:`~torch.distributed.optim.DistributedOptimizer.step`,
    either from the same or different clients, will
    be serialized on each worker -- as each worker's optimizer can only work
    on one set of gradients at a time. However, there is no guarantee that
    the full forward-backward-optimizer sequence will execute for one client
    at a time. This means that the gradients being applied may not correspond
    to the latest forward pass executed on a given worker. Also, there is no
    guaranteed ordering across workers.

    `DistributedOptimizer` uses a functional optimizer internally when one is
    available for the given ``optimizer_class``, so that optimizer updates are
    not blocked by the Python Global Interpreter Lock (GIL) in the case of
    multithreaded training (e.g. Distributed Model Parallel). This feature is
    currently enabled for most optimizers.

    Args:
        optimizer_class (optim.Optimizer): the class of optimizer to
            instantiate on each worker.
        params_rref (list[RRef]): list of RRefs to local or remote parameters
            to optimize.
        args: arguments to pass to the optimizer constructor on each worker.
        kwargs: arguments to pass to the optimizer constructor on each worker.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> import torch.distributed.autograd as dist_autograd
        >>> import torch.distributed.rpc as rpc
        >>> from torch import optim
        >>> from torch.distributed.optim import DistributedOptimizer
        >>>
        >>> with dist_autograd.context() as context_id:
        >>>   # Forward pass.
        >>>   rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        >>>   rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
        >>>   loss = rref1.to_here() + rref2.to_here()
        >>>
        >>>   # Backward pass.
        >>>   dist_autograd.backward(context_id, [loss.sum()])
        >>>
        >>>   # Optimizer.
        >>>   dist_optim = DistributedOptimizer(
        >>>      optim.SGD,
        >>>      [rref1, rref2],
        >>>      lr=0.05,
        >>>   )
        >>>   dist_optim.step(context_id)
    """

    def __init__(self, optimizer_class, params_rref, *args, **kwargs):
        torch._C._log_api_usage_once("torch.distributed.optim.DistributedOptimizer")
        per_worker_params_rref = defaultdict(list)
        for param in params_rref:
            per_worker_params_rref[param.owner()].append(param)

        if optimizer_class in functional_optim_map and jit._state._enabled:
            optim_ctor = functional_optim_map.get(optimizer_class)
        else:
            optim_ctor = optimizer_class
        self.is_functional_optim = optim_ctor != optimizer_class

        if self.is_functional_optim:
            optimizer_new_func = _new_functional_local_optimizer
        else:
            logger.warning(
                "Creating the optimizer %s without a functional optimizer, "
                "this might result in slow computation time in multithreading environment"
                "(i.e. Distributed Model Parallel training on CPU) due to the Python's "
                "Global Interpreter Lock (GIL). Please file an issue if you need a "
                "functional optimizer for this optimizer. ",
                optimizer_class,
            )
            optimizer_new_func = _new_local_optimizer

        remote_optim_futs = []
        for worker, param_rrefs in per_worker_params_rref.items():
            remote_optim_rref_fut = rpc.rpc_async(
                worker,
                optimizer_new_func,
                args=(optim_ctor, param_rrefs) + args,
                kwargs=kwargs,
            )
            remote_optim_futs.append(remote_optim_rref_fut)

        self.remote_optimizers = _wait_for_all(remote_optim_futs)

    def step(self, context_id):
        """
        Performs a single optimization step.

        This will call :meth:`torch.optim.Optimizer.step` on each worker
        containing parameters to be optimized, and will block until all workers
        return. The provided ``context_id`` will be used to retrieve the
        corresponding :class:`~torch.distributed.autograd.context` that
        contains the gradients that should be applied to the parameters.

        Args:
            context_id: the autograd context id for which we should run the
                optimizer step.
        """
        dist_autograd._is_valid_context(context_id)

        optimizer_step_func = (
            _functional_local_optimizer_step
            if self.is_functional_optim
            else _local_optimizer_step
        )

        rpc_futs = [
            rpc.rpc_async(
                optimizer.owner(),
                optimizer_step_func,
                args=(optimizer, context_id),
            )
            for optimizer in self.remote_optimizers
        ]
        _wait_for_all(rpc_futs)
<|endoftext|># mypy: allow-untyped-defs

from torch import optim

from .functional_adadelta import _FunctionalAdadelta
from .functional_adagrad import _FunctionalAdagrad
from .functional_adam import _FunctionalAdam
from .functional_adamax import _FunctionalAdamax
from .functional_adamw import _FunctionalAdamW
from .functional_rmsprop import _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop
from .functional_sgd import _FunctionalSGD


# dict to map a user passed in optimizer_class to a functional
# optimizer class if we have already defined inside the
# distributed.optim package, this is so that we hide the
# functional optimizer to user and still provide the same API.
functional_optim_map = {
    optim.Adagrad: _FunctionalAdagrad,
    optim.Adam: _FunctionalAdam,
    optim.AdamW: _FunctionalAdamW,
    optim.SGD: _FunctionalSGD,
    optim.Adadelta: _FunctionalAdadelta,
    optim.RMSprop: _FunctionalRMSprop,
    optim.Rprop: _FunctionalRprop,
    optim.Adamax: _FunctionalAdamax,
}


def register_functional_optim(key, optim):
    """
    Interface to insert a new functional optimizer to functional_optim_map.
    ``fn_optim_key`` and ``fn_optimizer`` are user defined. The optimizer and key
    need not be of :class:`torch.optim.Optimizer` (e.g. for custom optimizers).

    Example::

        >>> # import the new functional optimizer
        >>> # xdoctest: +SKIP
        >>> from xyz import fn_optimizer
        >>> from torch.distributed.optim.utils import register_functional_optim
        >>> fn_optim_key = "XYZ_optim"
        >>> register_functional_optim(fn_optim_key, fn_optimizer)
    """
    if key not in functional_optim_map:
        functional_optim_map[key] = optim


def as_functional_optim(optim_cls: type, *args, **kwargs):
    try:
        functional_cls = functional_optim_map[optim_cls]
    except KeyError as e:
        raise ValueError(
            f"Optimizer {optim_cls} does not have a functional counterpart!"
        ) from e

    return _create_functional_optim(functional_cls, *args, **kwargs)


def _create_functional_optim(functional_optim_cls: type, *args, **kwargs):
    return functional_optim_cls(
        [],
        *args,
        **kwargs,
        _allow_empty_param_list=True,
    )
<|endoftext|>"""
:mod:`torch.distributed.optim` exposes DistributedOptimizer, which takes a list
of remote parameters (:class:`~torch.distributed.rpc.RRef`) and runs the
optimizer locally on the workers where the parameters live.  The distributed
optimizer can use any of the local optimizer :ref:`optimizer-algorithms` to
apply the gradients on each worker.
"""

import warnings

import torch
from torch import optim

from .apply_optimizer_in_backward import (
    _apply_optimizer_in_backward,
    _get_in_backward_optimizers,
)
from .functional_adadelta import _FunctionalAdadelta
from .functional_adagrad import _FunctionalAdagrad
from .functional_adam import _FunctionalAdam
from .functional_adamax import _FunctionalAdamax
from .functional_adamw import _FunctionalAdamW
from .functional_rmsprop import _FunctionalRMSprop
from .functional_rprop import _FunctionalRprop
from .functional_sgd import _FunctionalSGD
from .named_optimizer import _NamedOptimizer
from .utils import as_functional_optim


# DistributedOptimizer imports torch.distributed.rpc names, so gate availability
# based on RPC being available.
if hasattr(torch._C, "_rpc_init"):
    from .optimizer import DistributedOptimizer

from .post_localSGD_optimizer import PostLocalSGDOptimizer
from .zero_redundancy_optimizer import ZeroRedundancyOptimizer


__all__ = [
    "as_functional_optim",
    "DistributedOptimizer",
    "PostLocalSGDOptimizer",
    "ZeroRedundancyOptimizer",
]
<|endoftext|>import warnings

import torch


@torch.jit.ignore  # type: ignore[misc]
def _scripted_functional_optimizer_deprecation_warning(stacklevel: int = 0) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.warn(
            "`TorchScript` support for functional optimizers is deprecated "
            "and will be removed in a future PyTorch release. "
            "Consider using the `torch.compile` optimizer instead.",
            DeprecationWarning,
            stacklevel=stacklevel + 2,
        )
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.optim._functional as F
from torch import Tensor
from torch.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional AdamW Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalAdamW:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        fused: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }
        self.amsgrad = amsgrad
        self.maximize = maximize
        self.foreach = foreach
        self.fused = fused
        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

    def step_param(self, param: Tensor, grad: Tensor | None):
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps: list[Tensor] = []
        has_complex = torch.is_complex(param)
        if grad is not None:
            params_with_grad.append(param)
            grads.append(grad)
        # Lazy state initialization
        if param not in self.state:
            self.state[param] = {}
            state = self.state[param]
            state["step"] = torch.tensor(0.0)
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            # Exponential moving average of squared gradient values
            state["exp_avg_sq"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            if self.amsgrad:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state["max_exp_avg_sq"] = torch.zeros_like(
                    param, memory_format=torch.preserve_format
                )

        state = self.state[param]

        exp_avgs.append(state["exp_avg"])
        exp_avg_sqs.append(state["exp_avg_sq"])

        if self.amsgrad:
            max_exp_avg_sqs.append(state["max_exp_avg_sq"])

        state_steps.append(state["step"])
        with torch.no_grad():
            F.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=self.amsgrad,
                maximize=self.maximize,
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                eps=self.defaults["eps"],
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
                has_complex=has_complex,
            )

    def step(self, gradients: list[Tensor | None]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        max_exp_avg_sqs = []
        state_steps: list[Tensor] = []

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_complex = False
        for param, gradient in zip(self.param_group["params"], gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    if self.amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                state = self.state[param]

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if self.amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                state_steps.append(state["step"])

        with torch.no_grad():
            F.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=self.amsgrad,
                maximize=self.maximize,
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                eps=self.defaults["eps"],
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
                has_complex=has_complex,
            )
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.optim._functional as F
from torch import Tensor
from torch.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional Adagrad Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly let the user pass gradients to the `step` function
# this is so that we could separate the gradients and parameters
# and allow multithreaded trainer to update the parameters
# without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalAdagrad:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        warmup_lr_multiplier: float = 1.0,
        warmup_num_iters: float = 0.0,
        eps: float = 1e-10,
        coalesce_grad: bool = True,
        foreach: bool = False,
        fused: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "eps": eps,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "warmup_lr_multiplier": warmup_lr_multiplier,
            "warmup_num_iters": warmup_num_iters,
        }
        self.coalesce_grad = coalesce_grad
        self.foreach = foreach
        self.fused = fused
        self.maximize = maximize
        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

        # TODO: no union or any types in TorchScript, make step a scalar tensor instead
        # This is also needed by if we want to share_memory on the step across processes
        for p in self.param_group["params"]:
            self.state[p] = {
                "sum": torch.full_like(p.data, initial_accumulator_value),
                "step": torch.tensor(0.0),
            }

    def step(self, gradients: list[Tensor | None]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        state_sums = []
        state_steps: list[Tensor] = []

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_sparse_grad, has_complex = False, False
        for param, gradient in zip(self.param_group["params"], gradients):
            if gradient is not None:
                has_sparse_grad |= gradient.is_sparse
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                state = self.state[param]
                state_sums.append(state["sum"])
                state_steps.append(state["step"])

        with torch.no_grad():
            F.adagrad(
                params,
                grads,
                state_sums,
                state_steps,
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                lr_decay=self.defaults["lr_decay"],
                eps=self.defaults["eps"],
                has_sparse_grad=has_sparse_grad,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.optim._functional as F
from torch import Tensor
from torch.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional RMSprop Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalRMSprop:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum": momentum,
        }
        self.centered = centered
        self.foreach = foreach
        self.maximize = maximize

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

        self.state = torch.jit.annotate(dict[torch.Tensor, dict[str, torch.Tensor]], {})

    def step(self, gradients: list[Tensor | None]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        square_avgs = []
        grad_avgs = []
        momentum_buffer_list = []
        state_steps = []
        lr = self.defaults["lr"]
        alpha = self.defaults["alpha"]
        eps = self.defaults["eps"]
        momentum = self.defaults["momentum"]
        weight_decay = self.defaults["weight_decay"]

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_complex = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                has_complex |= torch.is_complex(param)
                params_with_grad.append(param)
                grads.append(gradient)
                # Lazy state initialization
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = torch.tensor(0.0)
                    state["square_avg"] = torch.zeros_like(
                        param, memory_format=torch.preserve_format
                    )
                    if momentum > 0:
                        state["momentum_buffer"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )
                    if self.centered:
                        state["grad_avg"] = torch.zeros_like(
                            param, memory_format=torch.preserve_format
                        )

                state = self.state[param]
                square_avgs.append(state["square_avg"])
                if momentum > 0:
                    momentum_buffer_list.append(state["momentum_buffer"])
                if self.centered:
                    grad_avgs.append(state["grad_avg"])

                state_steps.append(state["step"])

        with torch.no_grad():
            F.rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=self.centered,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
<|endoftext|># mypy: allow-untyped-defs
import warnings

import torch
import torch.distributed.algorithms.model_averaging.averagers as averagers


class PostLocalSGDOptimizer(torch.optim.Optimizer):
    r"""
    Wraps an arbitrary :class:`torch.optim.Optimizer` and runs `post-local SGD <https://arxiv.org/abs/1808.07217>`_,
    This optimizer runs local optimizer at every step.
    After the warm-up stage, it averages parameters periodically after the local optimizer is applied.

    Args:
        optim: The local optimizer.
        averager: A model averager instance to run post-localSGD algorithm.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import torch
        >>> import torch.distributed as dist
        >>> import torch.distributed.algorithms.model_averaging.averagers as averagers
        >>> import torch.nn as nn
        >>> from torch.distributed.optim import PostLocalSGDOptimizer
        >>> from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
        >>>   PostLocalSGDState,
        >>>   post_localSGD_hook,
        >>> )
        >>>
        >>> model = nn.parallel.DistributedDataParallel(
        >>>    module, device_ids=[rank], output_device=rank
        >>> )
        >>>
        >>> # Register a post-localSGD communication hook.
        >>> state = PostLocalSGDState(process_group=None, subgroup=None, start_localSGD_iter=100)
        >>> model.register_comm_hook(state, post_localSGD_hook)
        >>>
        >>> # Create a post-localSGD optimizer that wraps a local optimizer.
        >>> # Note that ``warmup_steps`` used in ``PostLocalSGDOptimizer`` must be the same as
        >>> # ``start_localSGD_iter`` used in ``PostLocalSGDState``.
        >>> local_optim = torch.optim.SGD(params=model.parameters(), lr=0.01)
        >>> opt = PostLocalSGDOptimizer(
        >>>     optim=local_optim,
        >>>     averager=averagers.PeriodicModelAverager(period=4, warmup_steps=100)
        >>> )
        >>>
        >>> # In the first 100 steps, DDP runs global gradient averaging at every step.
        >>> # After 100 steps, DDP runs gradient averaging within each subgroup (intra-node by default),
        >>> # and post-localSGD optimizer runs global model averaging every 4 steps after applying the local optimizer.
        >>> for step in range(0, 200):
        >>>    opt.zero_grad()
        >>>    loss = loss_fn(output, labels)
        >>>    loss.backward()
        >>>    opt.step()
    """

    def __init__(self, optim: torch.optim.Optimizer, averager: averagers.ModelAverager):
        self.optim = optim
        self.param_groups = self.optim.param_groups
        self.averager = averager

    @property
    def state(self):  # type: ignore[override]
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    def state_dict(self):
        r"""
        This is the same as :class:`torch.optim.Optimizer` :meth:`state_dict`,
        but adds an extra entry to record model averager's step to the checkpoint
        to ensure reload does not cause unnecessary warm up again.
        """
        optim_state_dict = self.optim.state_dict()
        optim_state_dict["step"] = self.averager.step
        return optim_state_dict

    def load_state_dict(self, state_dict):
        r"""
        This is the same as :class:`torch.optim.Optimizer` :meth:`load_state_dict`,
        but also restores model averager's step value to the one
        saved in the provided ``state_dict``.

        If there is no ``"step"`` entry in ``state_dict``,
        it will raise a warning and initialize the model averager's step to 0.
        """
        self.optim.load_state_dict(state_dict)
        if "step" in state_dict:
            self.averager.step = state_dict["step"]
        else:
            warnings.warn(
                "Loaded state dict does not contain a step counter for an averager. "
                "Setting step counter to 0.",
                stacklevel=2,
            )
            self.averager.step = 0

    def step(self):  # type: ignore[override]
        r"""
        Performs a single optimization step (parameter update).
        """
        self.optim.step()
        self.averager.average_parameters(params=self.param_groups)

    def zero_grad(self, set_to_none: bool = True):  # type: ignore[override]
        self.optim.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)
<|endoftext|>from collections.abc import Iterable
from typing import Any, no_type_check

import torch


__all__: list[str] = []

# WeakTensorKeyDictionary to store relevant meta-data for the Tensor/Parameter
# without changing it's life-time.
# NOTE: Alternative is to add the meta-data as an attribute to the tensor,
#       but that will serialize the meta-data if Tensor is serialized.
param_to_optim_hook_handle_map = torch.utils.weak.WeakTensorKeyDictionary()
param_to_acc_grad_map = torch.utils.weak.WeakTensorKeyDictionary()


@no_type_check
def _apply_optimizer_in_backward(
    optimizer_class: type[torch.optim.Optimizer],
    params: Iterable[torch.nn.Parameter],
    optimizer_kwargs: dict[str, Any],
    register_hook: bool = True,
) -> None:
    """
    Upon ``backward()``, the optimizer specified for each parameter will fire after
    the gradient has been accumulated into the parameter.

    Note - gradients for these parameters will be set to None after ``backward()``.
    This means that any other optimizer not specified via `_apply_optimizer_in_backward`
    over this parameter will be a no-op.

    Args:
        optimizer_class: (Type[torch.optim.Optimizer]): Optimizer to apply to parameter
        params: (Iterator[nn.Parameter]): parameters to apply optimizer state to
        optimizer_kwargs: (Dict[str, Any]): kwargs to pass to optimizer constructor
        register_hook: (bool): whether to register a hook that runs the optimizer
            after gradient for this parameter is accumulated. This is the default
            way that optimizer in backward is implemented, but specific use cases
            (such as DDP) may wish to override this to implement custom behavior.
            (Default = True)

    Example::
        params_generator = model.parameters()
        param_1 = next(params_generator)
        remainder_params = list(params_generator)

        apply_optimizer_in_backward(torch.optim.SGD, [param_1], {"lr": 0.02})
        apply_optimizer_in_backward(torch.optim.Adam, remainder_params, {"lr": 0.04})

        model(...).sum().backward()  # after backward, parameters will already
        # have their registered optimizer(s) applied.

    """
    torch._C._log_api_usage_once("torch.distributed.optim.apply_optimizer_in_backward")

    @no_type_check
    def _apply_optimizer_in_backward_to_param(param: torch.nn.Parameter) -> None:
        # view_as creates a node in autograd graph that allows us access to the
        # parameter's AccumulateGrad autograd function object. We register a
        # hook on this object to fire the optimizer when the gradient for
        # this parameter is ready (has been accumulated into .grad field)

        # Don't create a new acc_grad if we already have one
        # i.e. for shared parameters or attaching multiple optimizers to a param.
        if param not in param_to_acc_grad_map:
            param_to_acc_grad_map[param] = param.view_as(param).grad_fn.next_functions[
                0
            ][0]

        optimizer = optimizer_class([param], **optimizer_kwargs)

        if not hasattr(param, "_in_backward_optimizers"):
            param._in_backward_optimizers = []  # type: ignore[attr-defined]
            # TODO: Remove these attributes once we have a better way of accessing
            # optimizer classes and kwargs for a parameter.
            param._optimizer_classes = []  # type: ignore[attr-defined]
            param._optimizer_kwargs = []  # type: ignore[attr-defined]

        param._in_backward_optimizers.append(optimizer)  # type: ignore[attr-defined]
        param._optimizer_classes.append(optimizer_class)  # type: ignore[attr-defined]
        param._optimizer_kwargs.append(optimizer_kwargs)  # type: ignore[attr-defined]

        if not register_hook:
            return

        def optimizer_hook(*_unused) -> None:
            for opt in param._in_backward_optimizers:  # type: ignore[attr-defined]
                opt.step()

            param.grad = None

        handle = param_to_acc_grad_map[param].register_hook(optimizer_hook)  # type: ignore[attr-defined]
        if param not in param_to_optim_hook_handle_map:
            param_to_optim_hook_handle_map[param] = []
        param_to_optim_hook_handle_map[param].append(handle)

    for param in params:
        _apply_optimizer_in_backward_to_param(param)


def _get_in_backward_optimizers(module: torch.nn.Module) -> list[torch.optim.Optimizer]:
    """
    Return a list of in-backward optimizers applied to ``module``'s parameters. Note that these
    optimizers are not intended to directly have their ``step`` or ``zero_grad`` methods called
    by the user and are intended to be used for things like checkpointing.

    Args:
        module: (torch.nn.Module): model to retrieve in-backward optimizers for

    Returns:
        List[torch.optim.Optimizer]: the in-backward optimizers.

    Example::
        _apply_optimizer_in_backward(torch.optim.SGD, model.parameters(), {"lr": 0.01})
        optims = _get_optimizers_in_backward(model)
    """
    optims: list[torch.optim.Optimizer] = []
    for param in module.parameters():
        optims.extend(getattr(param, "_in_backward_optimizers", []))

    return optims
<|endoftext|>#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Functions for manipulating IntTuples
"""

from functools import reduce
from itertools import chain
from typing import TypeAlias
from typing_extensions import TypeIs

from .typing import Integer


# Type aliases for better readability
IntTuple: TypeAlias = int | tuple["IntTuple", ...]


def is_int(x: object) -> TypeIs[int]:
    return isinstance(x, Integer)


def is_tuple(x: object) -> TypeIs[tuple]:
    return isinstance(x, tuple)


def as_tuple(x: IntTuple) -> tuple[IntTuple, ...]:
    if is_int(x):
        return (x,)
    return x


def match_structure(a: IntTuple, b: IntTuple) -> bool:
    if is_int(a) and is_int(b):
        return True
    if is_tuple(a) and is_tuple(b):
        return len(a) == len(b) and all(match_structure(x, y) for x, y in zip(a, b))
    return False


def flatten(t: IntTuple) -> tuple[int, ...]:
    if is_tuple(t):
        if len(t) == 0:
            return ()
        else:
            return tuple(i for a in t for i in flatten(a))
    else:
        return (t,)


def signum(a: int) -> int:
    return bool(a > 0) - bool(a < 0)


def product(a: IntTuple) -> int:
    if is_tuple(a):
        return reduce(lambda val, elem: val * product(elem), a, 1)
    else:
        return a


def inner_product(a: IntTuple, b: IntTuple) -> int:
    if is_tuple(a) and is_tuple(b):  # tuple tuple
        if len(a) != len(b):
            raise AssertionError
        return sum(inner_product(x, y) for x, y in zip(a, b))
    else:  # "int" "int"
        if is_tuple(a) or is_tuple(b):
            raise AssertionError
        return a * b


def tuple_max(a: IntTuple) -> int:
    if is_tuple(a):
        return max(tuple_max(x) for x in a)
    else:
        return a


def elem_scale(a: IntTuple, b: IntTuple) -> IntTuple:
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            if len(a) != len(b):
                raise AssertionError
            return tuple(elem_scale(x, y) for x, y in zip(a, b))
        else:  # tuple "int"
            raise AssertionError("Invalid combination: tuple with int")
    else:
        if is_tuple(b):  # "int" tuple
            return elem_scale(a, product(b))
        else:  # "int" "int"
            return a * b


# Inclusive prefix ceil div with output congruent to input a
def shape_div(a: IntTuple, b: IntTuple) -> IntTuple:
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            if len(a) != len(b):
                raise AssertionError
            return tuple(shape_div(x, y) for x, y in zip(a, b))
        else:  # tuple "int"
            # r = [shape_div(a[0],b)] + [shape_div(a[i],b := shape_div(b, product(a[i-1]))) for i in range(1,len(a))]
            r = []
            for v in a:
                r.append(shape_div(v, b))
                b = shape_div(b, product(v))
            return tuple(r)
    else:
        if is_tuple(b):  # "int" tuple
            return shape_div(a, product(b))
        else:  # "int" "int"
            if not (a % b == 0 or b % a == 0):
                raise AssertionError
            return (a + b - 1) // b


# Exclusive suffix product with output congruent to input a (lexicographic)
def suffix_product(a: IntTuple, init: IntTuple = 1) -> IntTuple:
    # TODO: With all these length asserts, may want to create a zip_strict wrapper.
    if is_tuple(a):
        if is_tuple(init):  # tuple tuple
            if len(a) != len(init):
                raise AssertionError
            return tuple(suffix_product(x, i) for x, i in zip(a, init))
        else:  # tuple "int"
            # Process from right to left for lexicographic ordering
            # r = [prefix_product(a[len(a)-1],init)] +
            # [prefix_product(a[i],init := init * product(a[i+1])) for i in range(len(a)-1,0)].reverse()
            r = []

            # Calculate products from right to left, appending to list
            for i in range(len(a) - 1, -1, -1):
                r.append(suffix_product(a[i], init))
                init = init * product(a[i])

            # Reverse to get correct lexicographic order
            r.reverse()
            return tuple(r)
    else:
        if is_tuple(init):  # "int" tuple
            raise AssertionError("Invalid combination: int with tuple init")
        else:  # "int" "int"
            return init


def idx2crd(idx: IntTuple, shape: IntTuple, stride: IntTuple | None = None) -> IntTuple:
    if stride is None:
        stride = suffix_product(shape)

    if is_tuple(idx):
        if is_tuple(shape) and is_tuple(stride):  # tuple tuple tuple
            if not (len(idx) == len(shape) and len(stride) == len(shape)):
                raise AssertionError
            return tuple(idx2crd(i, s, d) for i, s, d in zip(idx, shape, stride))
        else:  # tuple "int" "int"
            raise AssertionError("Invalid combination: tuple with int stride")
    else:
        if is_tuple(shape) and is_tuple(stride):  # "int" tuple tuple
            if len(shape) != len(stride):
                raise AssertionError
            return tuple(idx2crd(idx, s, d) for s, d in zip(shape, stride))
        else:  # "int" "int" "int"
            if is_tuple(shape) or is_tuple(stride):
                raise AssertionError
            return (idx // stride) % shape  # all are ints after type checks


def crd2idx(
    crd: IntTuple | None, shape: IntTuple, stride: IntTuple | None = None
) -> int:
    if stride is None:
        stride = suffix_product(shape)

    if is_tuple(crd):
        if is_tuple(shape) and is_tuple(stride):  # tuple tuple tuple
            if not (len(crd) == len(shape) and len(stride) == len(shape)):
                raise AssertionError
            return sum(crd2idx(c, s, d) for c, s, d in zip(crd, shape, stride))
        else:  # tuple "int" "int"
            raise AssertionError(f"Invalid combination: crd={crd}, shape={shape}")
    else:
        if crd is None:
            crd = 0

        if is_tuple(shape) and is_tuple(stride):  # "int" tuple tuple
            if len(shape) != len(stride):
                raise AssertionError
            result = 0
            # Process from right to left for lexicographic ordering
            for i in range(len(shape) - 1, 0, -1):
                result += crd2idx(crd % product(shape[i]), shape[i], stride[i])
                crd = crd // product(shape[i])
            if len(shape) > 0:
                result += crd2idx(crd, shape[0], stride[0])
            return result
        else:  # "int" "int" "int"
            if is_tuple(shape) or is_tuple(stride):
                raise AssertionError
            return crd * stride  # all are ints after type checks


# Transform crd into the dst_shape's iteration space
def crd2crd(
    crd: IntTuple, dst_shape: IntTuple, src_shape: IntTuple | None = None
) -> IntTuple:
    if is_tuple(crd):
        if is_tuple(dst_shape):  # tuple tuple
            if len(crd) != len(dst_shape):
                raise AssertionError
            return tuple(crd2crd(x, y) for x, y in zip(crd, dst_shape))
        else:  # tuple "int"
            # Ambiguous unless we have src_shape
            if src_shape is None:
                raise AssertionError
            return crd2idx(crd, src_shape)
    else:
        if is_tuple(dst_shape):  # "int" tuple
            return idx2crd(crd, dst_shape)
        else:  # "int" "int"
            if crd >= dst_shape:
                raise AssertionError
            return crd


# Filter trg according to crd: keep only elements of trg that are paired with None
def slice_(crd: tuple | int | None, trg: tuple | int) -> tuple | int:
    if is_tuple(crd):
        if is_tuple(trg):  # tuple tuple
            if len(crd) != len(trg):
                raise AssertionError
            # match C++ behavior of `filter_tuple` using `tuple_cat(...)`
            return tuple(
                chain(
                    *filter(  # type: ignore[arg-type]  # filter returns Iterator which is compatible
                        lambda x: x != (),
                        [slice_(c, s) for c, s in zip(crd, trg)],
                    )
                )
            )
        else:
            raise AssertionError("Invalid combination: tuple crd with int trg")
    elif crd is None:
        # match C++ behavior `return cute::tuple<B>{b};`
        return (trg,)
    else:
        return ()


# Determine if None appears at any of an int_tuples' terminals
def has_none(a: tuple | int | None) -> bool:
    if is_tuple(a):
        return any(has_none(v) for v in a)
    else:
        return a is None
<|endoftext|>#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

from abc import ABC


class Integer(ABC):  # noqa: B024  # Uses __subclasshook__ instead of abstract methods
    @classmethod
    def __subclasshook__(cls, c: type) -> bool:
        if c in [bool, float]:
            return False

        return issubclass(c, int)
<|endoftext|>#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

from .int_tuple import (
    as_tuple,
    crd2crd,
    crd2idx,
    elem_scale,
    flatten,
    has_none,
    idx2crd,
    inner_product,
    IntTuple,
    is_int,
    is_tuple,
    match_structure,
    product,
    shape_div,
    signum,
    slice_,
    suffix_product,
    tuple_max,
)
from .layout import (
    coalesce,
    complement,
    composition,
    cosize,
    filter,
    is_layout,
    Layout,
    LayoutBase,
    left_inverse,
    logical_divide,
    logical_product,
    make_layout,
    right_inverse,
    size,
    slice_and_offset,
    tiled_divide,
    tiled_product,
    zipped_divide,
    zipped_product,
)
from .typing import Integer
<|endoftext|>#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Definition of CuTe Layouts and functions to manipulate them which works with the order
of lexicographic instead of co-lexicographic as implemented in the original layout.py
"""

from itertools import chain
from typing import TypeAlias
from typing_extensions import Self, TypeIs

from .int_tuple import (
    crd2idx,
    flatten,
    has_none,
    IntTuple,
    is_int,
    is_tuple,
    product,
    slice_,
    suffix_product,
)


# Type aliases
CoordinateType: TypeAlias = (
    int | IntTuple | tuple[object, ...] | None
)  # Input for slice_ and crd2idx functions


class LayoutBase:
    pass


def is_layout(x: object) -> TypeIs["Layout"]:
    return isinstance(x, LayoutBase)


class Layout(LayoutBase):
    def __init__(self, _shape: IntTuple, _stride: IntTuple | None = None) -> None:
        self.shape = _shape
        if _stride is None:
            self.stride = suffix_product(self.shape)
        else:
            self.stride = _stride

    # operator ==
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Layout):
            return False
        return self.shape == other.shape and self.stride == other.stride

    # operator len(L)  (len [rank] like tuples)
    def __len__(self) -> int:
        if is_tuple(self.shape):
            return len(self.shape)
        else:
            return 1

    # operator ()    (map coord to idx)
    def __call__(self, *args: CoordinateType) -> Self | int:
        """
        Map a logical coordinate to a linear index (Coord has no Underscore slice operators)
        OR
        Slice the layout and return the sublayout (Coord has an Underscore slice op)

        Follow the same behavior of `Layout::operator(Coord const&)` in cute C++
        """
        if has_none(args):
            if len(args) == 1:
                return Layout(slice_(args[0], self.shape), slice_(args[0], self.stride))
            else:
                return Layout(slice_(args, self.shape), slice_(args, self.stride))
        else:
            if len(args) == 1:
                return crd2idx(args[0], self.shape, self.stride)  # type: ignore[arg-type]
            else:
                return crd2idx(args, self.shape, self.stride)  # type: ignore[arg-type]

    # operator []    (get-i like tuples)
    def __getitem__(self, i: int) -> Self:
        if is_tuple(self.shape):
            return Layout(self.shape[i], self.stride[i])  # type: ignore[index]
        else:
            if i != 0:
                raise AssertionError
            return Layout(self.shape, self.stride)

    # size(layout)   Size of the domain
    def size(self) -> int:
        return product(self.shape)

    # cosize(layout)   Size of the codomain
    def cosize(self) -> int:
        return self(self.size() - 1) + 1  # type: ignore[operator]

    # print and str
    def __str__(self) -> str:
        return f"{self.shape}:{self.stride}"

    # error msgs and representation
    def __repr__(self) -> str:
        return f"Layout({self.shape},{self.stride})"


# Type aliases
LayoutOrIntTuple: TypeAlias = Layout | IntTuple
LayoutProfile: TypeAlias = tuple[object, ...] | Layout | None
LayoutInput: TypeAlias = Layout | IntTuple | tuple[object, ...] | None


# Make Layout from a list of layouts (each layout its own mode in the result)
def make_layout(*layouts: Layout | tuple[Layout, ...]) -> Layout:
    if len(layouts) == 1 and not is_layout(layouts[0]):
        layouts = layouts[0]

    shape, stride = zip(*((a.shape, a.stride) for a in layouts))  # type: ignore[union-attr]
    return Layout(shape, stride)


# Size of the domain
def size(layout: LayoutOrIntTuple) -> int:
    if is_layout(layout):
        return layout.size()
    return product(layout)


# Size of the codomain
def cosize(layout: Layout) -> int:
    return layout.cosize()


# Layout coalesce -- flatten and combine as many modes as possible while preserving the int-to-int function
def coalesce(layout: Layout, profile: LayoutProfile = None) -> Layout:
    if is_tuple(profile):
        if len(layout) < len(profile):
            raise AssertionError
        return make_layout(
            # pyrefly: ignore [bad-argument-type]
            chain(
                (coalesce(layout[i], profile[i]) for i in range(len(profile))),  # type: ignore[arg-type]
                (layout[i] for i in range(len(profile), len(layout))),
            )
        )

    result_shape = [1]
    result_stride = [0]
    # Since we now follow lexicographic order, we need to process from right to left.
    # And to make implementation more efficient, we append to the end of list and reverse it in the end.
    for shape, stride in zip(
        reversed(flatten(layout.shape)), reversed(flatten(layout.stride))
    ):
        # skip their shape-1s
        if shape == 1:
            continue
        # replace our shape-1 with anything
        elif result_shape[-1] == 1:
            result_shape[-1] = shape
            result_stride[-1] = stride
        # merge modes if the shape*stride match
        elif result_shape[-1] * result_stride[-1] == stride:
            result_shape[-1] = result_shape[-1] * shape
        # append a new mode
        else:
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    else:
        result_shape.reverse()
        result_stride.reverse()
        return Layout(tuple(result_shape), tuple(result_stride))


# Layout filter -- replace all stride-0 modes with size-1 and then coalesce to remove them
def filter(layout: Layout, profile: LayoutProfile = None) -> Layout:
    if is_tuple(profile):
        if len(layout) < len(profile):
            raise AssertionError
        return make_layout(
            # pyrefly: ignore [bad-argument-type]
            chain(
                (filter(layout[i], profile[i]) for i in range(len(profile))),  # type: ignore[arg-type]
                (layout[i] for i in range(len(profile), len(layout))),
            )
        )

    result_shape = []
    result_stride = []
    for shape, stride in zip(flatten(layout.shape), flatten(layout.stride)):
        # skip their shape-1s and stride-0s
        if not (shape == 1 or stride == 0):
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 0:
        return Layout(1, 0)
    else:
        return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


# Layout composition
# Use tuples-of-layouts to perform this operation by-mode and None as no-op
def composition(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    if layoutB is None:
        return layoutA
    elif is_int(layoutB):
        return composition(layoutA, Layout(layoutB))
    elif is_tuple(layoutB):
        if len(layoutA) < len(layoutB):
            raise AssertionError
        return make_layout(
            # pyrefly: ignore [bad-argument-type]
            chain(
                (composition(layoutA[i], layoutB[i]) for i in range(len(layoutB))),  # type: ignore[arg-type]
                (layoutA[i] for i in range(len(layoutB), len(layoutA))),
            )
        )
    elif is_tuple(layoutB.shape):
        return make_layout(composition(layoutA, layoutB_i) for layoutB_i in layoutB)  # type: ignore[arg-type, attr-defined]

    if layoutB.stride == 0:
        return Layout(layoutB.shape, 0)
    else:
        result_shape = []
        result_stride = []
        rest_shape = layoutB.shape
        rest_stride = layoutB.stride
        flat_A = coalesce(layoutA)
        # when left layout is multi-dimensional sublayout, aka, self = (a,b,...,c):(x,y,...,z), layout = s:d,
        # for integral s and d means that we want:
        # (1) “remove” the first d elements from left, starting from rightmost. (This will increase the stride.)
        # (2) “keep” the first s of those strided elements. (This does not affect the stride.)
        # For example, if self = (6,2):(2,1), layout = (3:2)
        # Step 1: remove the first 2 elements from self with stride increase, i.e., (6,2):(2,1) -> (6,1):(2,2)
        # Step 2: keep the first 3 of those strided elements, i.e., (6,1):(2,2) -> (3,1):(2,2)
        # Because we are going lexicographically, we go through left layout from right to left.
        for curr_shape, curr_stride in zip(
            reversed(flatten(flat_A.shape)[1:]), reversed(flatten(flat_A.stride)[1:])
        ):
            if not (curr_shape % rest_stride == 0 or rest_stride % curr_shape == 0):  # type: ignore[operator]
                raise AssertionError
            new_shape = min(max(1, curr_shape // rest_stride), rest_shape)  # type: ignore[operator]

            if new_shape != 1:
                result_shape.append(new_shape)  # Append to end, will reverse later
                result_stride.append(rest_stride * curr_stride)

            rest_shape = rest_shape // new_shape  # type: ignore[operator]
            rest_stride = -(
                -rest_stride // curr_shape  # type: ignore[operator]
            )  # Python exclusive impl: "//" is always floor div so == ceil_div(abs(rest_stride), curr_shape) * signum(rest_stride)

        # When left has single-size sublayout or reach the last sublayout, aka, left = a:b, layout = s:d,
        # the result is rather trivial: left o layout = a:b o s:d = s:(b*d).
        # For example, if self = (6:2), layout = (3:2), the result is (3:(2*2)) = (3:4).
        if rest_shape != 1 or len(result_shape) == 0:
            result_shape.append(rest_shape)  # Append to end, will reverse later
            result_stride.append(rest_stride * flatten(flat_A.stride)[0])

        # Reverse the lists because we build lists in reverse order (append to end), this way it is more efficient.
        result_shape.reverse()
        result_stride.reverse()

        if len(result_shape) == 1:
            return Layout(result_shape[0], result_stride[0])  # type: ignore[arg-type]
        else:
            return Layout(tuple(result_shape), tuple(result_stride))  # type: ignore[arg-type]


# Layout complement
def complement(layout: LayoutOrIntTuple, max_idx: int = 1) -> Layout:
    if is_int(layout):
        return complement(Layout(layout))

    result_shape = []
    result_stride = []
    current_idx = 1

    sorted_DS = sorted(zip(flatten(layout.stride), flatten(layout.shape)))  # type: ignore[union-attr]
    for stride, shape in sorted_DS:
        if stride == 0 or shape == 1:
            continue

        in_bound = current_idx <= shape * stride
        # To support symbolic value which can't be evaluated now
        if (type(in_bound) is bool) and not in_bound:
            raise AssertionError

        result_shape.append(stride // current_idx)
        result_stride.append(current_idx)
        current_idx = shape * stride

    result_shape.append((max_idx + current_idx - 1) // current_idx)  # ceil_div
    result_stride.append(current_idx)
    # This is different from original pycute implementation, because we want to follow the lexicographic order here
    # where the right-most dimension is the innermost dimension (smallest stride).
    result_shape.reverse()
    result_stride.reverse()

    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


# Layout right inverse
def right_inverse(layout: LayoutOrIntTuple | None) -> Layout | None:
    if layout is None:
        return None
    elif is_int(layout):
        return Layout(layout)

    result_shape = []
    result_stride = []
    current_idx = 1

    flat_shape = flatten(layout.shape)  # type: ignore[union-attr]
    flat_stride = flatten(layout.stride)  # type: ignore[union-attr]
    sorted_DSA = sorted(zip(flat_stride, flat_shape, suffix_product(flat_shape)))  # type: ignore[arg-type]
    for stride, shape, rstride in sorted_DSA:
        if shape == 1:
            continue
        if current_idx != stride:
            break

        result_shape.append(shape)
        result_stride.append(rstride)
        current_idx = shape * stride

    result_shape.reverse()
    result_stride.reverse()
    return coalesce(Layout(tuple(result_shape), tuple(result_stride)))


# Layout left inverse
def left_inverse(layout: LayoutOrIntTuple | None) -> Layout | None:
    if layout is None:
        return None
    elif is_int(layout):
        return Layout(layout)
    return right_inverse(make_layout(complement(layout), layout))  # type: ignore[arg-type]


# Split a layout by the composition of B and the "rest"
# Use tuples-of-layouts to perform this operation by-mode and None as no-op
def logical_divide(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    if layoutB is None:
        return layoutA
    elif is_int(layoutB):
        return logical_divide(layoutA, Layout(layoutB))
    elif is_tuple(layoutB):
        if len(layoutA) < len(layoutB):
            raise AssertionError
        return make_layout(
            # pyrefly: ignore [bad-argument-type]
            chain(
                (
                    logical_divide(layoutA[i], layoutB[i])  # type: ignore[arg-type]
                    for i in range(len(layoutB))
                ),
                (layoutA[i] for i in range(len(layoutB), len(layoutA))),
            )
        )

    return composition(
        layoutA,
        make_layout(layoutB, complement(layoutB, size(layoutA))),
    )


# Reproduce a layoutA over a layoutB
# Use tuples-of-layouts to perform this operation by-mode and None as no-op
def logical_product(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    if layoutB is None:
        return layoutA
    elif is_int(layoutB):
        return logical_divide(layoutA, Layout(layoutB))
    elif is_tuple(layoutB):
        if len(layoutA) < len(layoutB):
            raise AssertionError
        return make_layout(
            # pyrefly: ignore [bad-argument-type]
            chain(
                (
                    logical_product(layoutA[i], layoutB[i])  # type: ignore[arg-type]
                    for i in range(len(layoutB))
                ),
                (layoutA[i] for i in range(len(layoutB), len(layoutA))),
            )
        )

    return make_layout(
        layoutA,
        composition(complement(layoutA, size(layoutA) * cosize(layoutB)), layoutB),
    )


# Gather the modes from a hierarchical logical_divide or logical_product
def hier_unzip(
    splitter: object,
    layoutA: Layout,
    layoutB: LayoutInput,
) -> Layout:
    if layoutB is None:
        return make_layout(Layout(1, 0), layoutA)
    elif is_tuple(layoutB):
        if len(layoutA) < len(layoutB):
            raise AssertionError
        # A layout with shape ((A,a),(B,b),(C,c))
        split = make_layout(
            hier_unzip(splitter, layoutA[i], layoutB[i])  # type: ignore[arg-type]
            for i in range(len(layoutB))
        )
        # Gather to shape ((A,B,C,...),(a,b,c,...,y,z))
        return make_layout(
            make_layout(split[i][0] for i in range(len(layoutB))),  # type: ignore[arg-type]
            make_layout(
                chain(  # type: ignore[arg-type]
                    (split[i][1] for i in range(len(layoutB))),
                    (layoutA[i] for i in range(len(layoutB), len(layoutA))),
                )
            ),
        )

    # splitter must return a rank-2 layout
    return splitter(layoutA, layoutB)  # type: ignore[operator]


# Apply logical divide hierarchically and gather the split modes into two modes
def zipped_divide(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    return hier_unzip(logical_divide, layoutA, layoutB)


# Perform logical divide hierarchically and gather tiles (B-layouts) into a new mode
def tiled_divide(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    result = zipped_divide(layoutA, layoutB)
    return make_layout([result[0]] + [result[1][i] for i in range(len(result[1]))])  # type: ignore[arg-type]


# Apply logical product hierarchically and gather the split modes into two modes
def zipped_product(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    return hier_unzip(logical_product, layoutA, layoutB)


# Perform logical product hierarchically and gather tiles (B-layouts) into a new mode
def tiled_product(layoutA: Layout, layoutB: LayoutInput) -> Layout:
    result = zipped_product(layoutA, layoutB)
    return make_layout([result[0]] + [result[1][i] for i in range(len(result[1]))])  # type: ignore[arg-type]


def slice_and_offset(crd: tuple[object, ...], layout: Layout) -> tuple[Layout, int]:
    return (
        Layout(slice_(crd, layout.shape), slice_(crd, layout.stride)),
        crd2idx(crd, layout.shape, layout.stride),  # type: ignore[arg-type]
    )
<|endoftext|>#!/usr/bin/env python3
"""Flight Recorder Trace Analyzer

This script primarily merges data from individual flight recorder buffers from individual ranks in a
PyTorch Distributed program into a flattened database format that can be used for further analysis.

However as part of the merging process, it is necessary to perform some analysis in order to match operators
on one rank with corresponding operators on other ranks and register them as one 'collective' entry.  During this
process, a significant amount of useful information can already be extracted such as where the first mismatch occurs
in cases of desync (when not all ranks issue a compatible collective in a particular process group).


Not Yet Implemented
- TODO- tracebacks aren't implemented

Known Issues
- Flight Recorder buffer sequence_id information is not sufficient to match collectives and coalesced collectives
  unless we have the trace data from the beginning of the program.  To enable confident analysis of trace buffers that
  do not start from zero (and to simplify the script's matching logic) we need to add more information to the recorder.
- Currently, the script omits checking the 'status' of collectives.  We can look for the first 'non completed'
  collective easily enough and report that.

Usage
python fr_trace.py <dump dir containing trace files> [-o <output file>]

- Omitting the optional output file will still yield analysis information to stdout
- The output file is a pickle of the flat DB, which may change in format in the future.
- This script is versioned so that we can ensure our future changes to flight recorder are backwards compatible.
"""

import pickle
from collections.abc import Sequence

from torch.distributed.flight_recorder.components.builder import build_db, transform_ft
from torch.distributed.flight_recorder.components.config_manager import JobConfig
from torch.distributed.flight_recorder.components.loader import read_dir
from torch.distributed.flight_recorder.components.types import types


__all__ = ["main"]


def main(args: Sequence[str] | None = None) -> None:
    config = JobConfig()
    # pyrefly: ignore [bad-assignment]
    args = config.parse_args(args)
    # pyrefly: ignore [missing-attribute]
    if not args.trace_dir:
        raise AssertionError("Trace directory trace_dir is required")
    # pyrefly: ignore [bad-argument-type]
    details, version = read_dir(args)
    # pyrefly: ignore [missing-attribute]
    if args.transform_ft:
        # pyrefly: ignore [missing-attribute]
        if not args.group_world_size:
            raise AssertionError("World size is required for transform_ft")
        # pyrefly: ignore [bad-argument-type]
        details = transform_ft(details, args.group_world_size)
    # pyrefly: ignore [bad-argument-type]
    db = build_db(details, args, version)
    # pyrefly: ignore [missing-attribute]
    if args.output:
        # pyrefly: ignore [no-matching-overload]
        with open(args.output, "wb") as f:
            pickle.dump((types, db), f)


if __name__ == "__main__":
    main()
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import copy
import os
import sys
from typing import Any  # type: ignore[attr-defined]

from torch.distributed.flight_recorder.components.fr_logger import FlightRecorderLogger
from torch.distributed.flight_recorder.components.types import (
    Collective,
    Database,
    EntryState,
    Group,
    MatchStateRecord,
    Membership,
    NCCLCall,
    Op,
    Traceback,
)
from torch.distributed.flight_recorder.components.utils import (
    add_stack_id_in_entries,
    align_trace_from_beginning,
    check_current_entry_match,
    check_no_missing_dump_files,
    check_version,
    error_analysis,
    find_coalesced_group as find_coalesced_group_p2p_only,
    find_coalesced_group_with_non_p2p,
    get_version_detail,
    just_print_entries,
    match_coalesced_groups as match_coalesced_groups_p2p_only,
    match_coalesced_groups_with_non_p2p,
)


__all__ = [
    "build_groups_memberships",
    "build_collectives",
    "transform_ft",
    "build_db",
]

# Set up logging
logger: FlightRecorderLogger = FlightRecorderLogger()


try:
    from tabulate import tabulate as _tabulate
except ModuleNotFoundError:
    logger.warning("tabulate is not installed. Proceeding without it.")

    # Define a no-op tabulate function
    def _tabulate(data: Any, headers: Any = None) -> Any:  # type: ignore[misc]
        return data


"""
Flat DB builder
"""


def build_groups_memberships(
    pg_config: Any,
) -> tuple[
    list[Group],
    dict[Any, Group],
    list[Membership],
    dict[str, set[Any]],
    dict[tuple[str, int], str],
]:
    """
    pg_config: {
        global_rank: {
            (pg_guid, desc, ranks)
        }
    }

    `pg_guid` is a system generated id, but depending on the mode of PG creation it could be a globally incrementing int
          or a hash of the ranks.  See `_process_group_name` in distributed_c10d.py.
    `desc` is provided by the user (optionally) and should be 'meaningful' (e.g. TP/PP/DP group)
    `ranks` is a list of the 'global ranks' that are members of the PG.

    (pg_guid, desc, ranks) tuples are appended lazily to the flight buffer when `getNCCLComm` is called on a PG and
    the `enabled_` flag is true for that PG.
        - the order of calling (init_process_group, new_group, etc) does not affect the order of the tuples in the list

    Returns:
        `groups`: a groups table where each row is a Group namedtuple.
        `_groups`: a dict that is indexed by pg_guid with Group namedtuple as value.
        `memberships`: a membership table where each row is a Membership namedtuple.
        `_memberships`: a dict that is indexed by pg_guid with set of ranks (int) as value.
        `_pg_guids`: a dict that is indexed by (pg_uid, global_rank) with pg_guid as value.
    """
    # flat lists for return
    groups = []
    memberships = []

