                f"the same number of elements, found {len(self.shard_offsets)} "
                f"and {self.shard_sizes} respectively"
            )

        for i in range(len(self.shard_offsets)):
            if self.shard_offsets[i] < 0:
                raise ValueError("shard_offsets should be >=0")
            if self.shard_sizes[i] < 0:
                raise ValueError("shard_sizes should be >= 0")

    def __hash__(self):
        def _hash_reduce(a, b):
            return (a << 8) + hash(b)

        res = reduce(_hash_reduce, self.shard_offsets, 37)
        res = reduce(_hash_reduce, self.shard_sizes, res)
        res = _hash_reduce(res, self.placement)
        return res
<|endoftext|>import abc

import torch.nn as nn


class Sharder(abc.ABC):
    """
    This is an interface which allows user to create more advanced
    sharding strategies that are not easily be composed by the
    `ShardingSpec`.

    :class:`torch.distributed._shard.sharding_plan.ShardingPlan` could
    take an object of the `Sharder` and call `shard` to shard the module,
    then replace the original module with sharded module returned.
    """

    @abc.abstractmethod
    def shard(self, module: nn.Module) -> nn.Module:
        """
        Shard a module base on the implementation of this method, and
        return the sharded version of the module.

        Args:
            module (:class:`torch.nn.Module`):
                The module to apply sharding to.
        Returns:
            A :class:`torch.nn.Module` object that represents a module
            that's already been sharded.
        """
<|endoftext|># mypy: allow-untyped-defs
import copy

import torch
import torch.distributed as dist
import torch.distributed._shard.sharding_spec as shard_spec
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharding_spec._internals import (
    get_chunked_dim_size,
    get_split_size,
)
from torch.distributed.nn.functional import all_to_all, all_to_all_single

from .shard import Shard


def get_idx_from_placements(placements, current_rank) -> int:
    """
    Return the position of the current rank in the given placements.

    Args:
        placements(List[Union[_remote_device, str]]):
            Specifies the placement of each shard of the Tensor. The size of
            the list represents the number of shards to be created. This could
            be a list of
            :class:`torch.distributed._remote_device`'s. This list
            could also contain a string which represents remote
            device as accepted by
            :class:`torch.distributed._remote_device`
        current_rank (int): number of current device.

    Returns:
        An int which contains the position of current device in the placement list.
    """
    for idx, placement in enumerate(placements):  # type: ignore[attr-defined]
        if current_rank == placement.rank():  # type: ignore[union-attr]
            return idx
    raise RuntimeError("current_rank not in the placement.")


def build_reshard_metadata(
    st_size: torch.Size,
    sharding_spec: shard_spec.ShardingSpec,
    world_size: int,
) -> tuple[list[ShardMetadata], list[int]]:
    """
    Based the given sharding spec, we calculate the offset and local shard size.
    We then build a ShardMetadata on top of the calculation result.

    Args:
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded.
        world_size (int): number of ranks.

    Returns:
        A Tuple of the following:
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
            A List[int] which contains the ranks in the order of placement.
    """
    shard_dim = int(sharding_spec.dim)  # type: ignore[attr-defined]
    shards_metadata = [None] * world_size
    ranks = []
    offsets = [0] * len(st_size)
    split_size = get_split_size(st_size[shard_dim], world_size)
    for idx, placement in enumerate(sharding_spec.placements):  # type: ignore[attr-defined]
        ranks.append(placement.rank())
        sharded_dim_size = get_chunked_dim_size(st_size[shard_dim], split_size, idx)
        local_tensor_size = list(st_size)
        local_tensor_size[shard_dim] = sharded_dim_size
        shards_metadata[placement.rank()] = ShardMetadata(  # type: ignore[call-overload]
            shard_offsets=copy.deepcopy(offsets),
            shard_sizes=local_tensor_size,
            placement=placement,
        )
        offsets[shard_dim] += sharded_dim_size
    return shards_metadata, ranks  # type: ignore[return-value]


def reshuffle_local_shard(
    local_shard: torch.Tensor,
    st_size: torch.Size,
    sharding_spec: shard_spec.ShardingSpec,
    resharding_spec: shard_spec.ShardingSpec,
    pg: ProcessGroup,
) -> tuple[list[Shard], list[ShardMetadata]]:
    """
    Reshuffle the local shard directly when the reshard dim is same as the original
    sharding dim. Logically we do this in two step:
    1. To collect all shards based on original sharding spec.
    2. Reshard the tensor based on the given resharding spec.

    In reality, we consolidate the two steps into one by sending the local tensor to
    the new shard directly based on the resharding spec.

    Args:
        local_shard (Tensor): Local tensor stored in the current rank.
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded originally.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor will be resharded.
        pg (ProcessGroup): The process group to aggregate on.

    Returns:
        A Tuple of the following:
            A List[`Shard`] which contains the local tensor and its metadata.
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
    """
    current_rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    # Build shards_metadata first.
    shards_metadata, ranks = build_reshard_metadata(
        st_size, resharding_spec, world_size
    )
    # Get input split size for all2all.
    reshard_dim = int(resharding_spec.dim)  # type: ignore[attr-defined]
    split_size = get_split_size(st_size[reshard_dim], world_size)
    input_split_sizes = [0] * world_size
    idx = get_idx_from_placements(sharding_spec.placements, current_rank)  # type: ignore[attr-defined]
    new_rank = resharding_spec.placements[idx].rank()  # type: ignore[union-attr, attr-defined]
    input_split_sizes[new_rank] = local_shard.size(reshard_dim)
    # Get output split size for all2all.
    output_split_sizes = [0] * world_size
    new_idx = ranks.index(current_rank)
    sharded_dim_size = get_chunked_dim_size(st_size[reshard_dim], split_size, new_idx)
    output_split_sizes[new_rank] = sharded_dim_size
    # Get gathered_input for all2all.
    local_shard = local_shard.transpose(0, reshard_dim).contiguous()
    gathered_input_size = list(local_shard.size())
    gathered_input_size[0] = sharded_dim_size
    gathered_input = torch.empty(
        gathered_input_size, device=local_shard.device, dtype=local_shard.dtype
    )
    # all2all.
    local_shard = all_to_all_single(
        gathered_input,
        local_shard,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
        group=pg,
    )
    local_tensor = local_shard.transpose(0, reshard_dim).contiguous()
    local_shards = [Shard(local_tensor, shards_metadata[current_rank])]
    return local_shards, shards_metadata


def reshard_local_shard(
    local_tensor: torch.Tensor,
    st_size: torch.Size,
    sharding_spec: shard_spec.ShardingSpec,
    resharding_spec: shard_spec.ShardingSpec,
    pg: ProcessGroup,
) -> tuple[list[Shard], list[ShardMetadata]]:
    """
    Reshard a sharded tensor given the ``resharding_spec``. When the reshard dim is
    different from the original sharding dim, we need to do two steps logically:
    1. To collect all shards based on original sharding spec.
    2. Reshard the tensor based on the given resharding spec.

    In reality, we consolidate the two steps into one by sending each rank the new
    shard based on the resharding spec.

    Args:
        local_tensor (Tensor): Local tensor stored in the current rank.
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded originally.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor will be resharded.
        pg (ProcessGroup): The process group to aggregate on.

    Returns:
        A Tuple of the following:
            A List[`Shard`] which contains the local tensor and its metadata.
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
    """
    current_rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    current_sharding_dim = int(sharding_spec.dim)  # type: ignore[attr-defined]
    reshard_dim = int(resharding_spec.dim)  # type: ignore[attr-defined]

    # Build shards_metadata first.
    shards_metadata, ranks = build_reshard_metadata(
        st_size, resharding_spec, world_size
    )

    # Compute expected size
    input_split_sizes = [
        metadata.shard_sizes[reshard_dim] for metadata in shards_metadata
    ]
    rearrange_input = any(ranks[i] > ranks[i + 1] for i in range(len(ranks) - 1))

    if rearrange_input:
        # Need to re-arrange reshard_dim of local_tensor before all2all.
        indices: list[int] = []
        for metadata in shards_metadata:
            offset_start_idx = metadata.shard_offsets[reshard_dim]
            split_size = metadata.shard_sizes[reshard_dim]
            indices += range(offset_start_idx, offset_start_idx + split_size)
        local_tensor = local_tensor.index_select(
            reshard_dim, torch.tensor(indices, device=local_tensor.device)
        )

    # Because reshard_dim != original shard_dim. We need to compute the
    # size of tensor from each rank.
    output_tensor_list = [torch.tensor(1)] * world_size
    split_size = get_split_size(st_size[current_sharding_dim], world_size)
    rearrange_output_list = False
    indices = []
    for idx, placement in enumerate(sharding_spec.placements):  # type: ignore[attr-defined]
        sharded_dim_size = get_chunked_dim_size(
            st_size[current_sharding_dim], split_size, idx
        )
        output_tensor_size = list(st_size)
        output_tensor_size[current_sharding_dim] = sharded_dim_size
        output_tensor_size[reshard_dim] = input_split_sizes[current_rank]
        output_tensor_list[placement.rank()] = torch.empty(  # type: ignore[union-attr, index]
            output_tensor_size, device=local_tensor.device, dtype=local_tensor.dtype
        )
        indices.append(placement.rank())  # type: ignore[union-attr, index, arg-type]
        if idx != placement.rank():  # type: ignore[union-attr]
            rearrange_output_list = True

    # Perform autograd enabled all2all.
    input_tensor_tuple = torch.split(local_tensor, input_split_sizes, dim=reshard_dim)
    input_tensor_list = [tensor.contiguous() for tensor in input_tensor_tuple]
    output_tensor_list = all_to_all(
        output_tensor_list,
        input_tensor_list,
        group=pg,
    )

    if rearrange_output_list:
        # Need to re-arrange original shard_dim of output_tensor_list.
        output_tensor_list = [output_tensor_list[idx] for idx in indices]  # type: ignore[call-overload]
    local_tensor = torch.cat(output_tensor_list, dim=current_sharding_dim)
    local_shards = [Shard(local_tensor, shards_metadata[current_rank])]
    return local_shards, shards_metadata
<|endoftext|>#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torch.distributed._shard.sharded_tensor.logging_handlers import _log_handlers


__all__: list[str] = []


def _get_or_create_logger() -> logging.Logger:
    logging_handler, log_handler_name = _get_logging_handler()
    logger = logging.getLogger(f"sharding-spec-{log_handler_name}")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s %(filename)s:%(lineno)s %(levelname)s p:%(processName)s t:%(threadName)s: %(message)s"
    )
    logging_handler.setFormatter(formatter)
    logger.propagate = False
    logger.addHandler(logging_handler)
    return logger


def _get_logging_handler(
    destination: str = "default",
) -> tuple[logging.Handler, str]:
    log_handler = _log_handlers[destination]
    log_handler_name = type(log_handler).__name__
    return (log_handler, log_handler_name)
<|endoftext|># mypy: allow-untyped-defs
import collections.abc
import copy
import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch.distributed import distributed_c10d as c10d, rpc
from torch.distributed._shard.sharding_spec._internals import (
    check_tensor,
    validate_non_overlapping_shards_metadata,
)

from .metadata import ShardedTensorMetadata, TensorProperties
from .shard import Shard


if TYPE_CHECKING:
    from torch.distributed._shard.metadata import ShardMetadata


def _parse_and_validate_remote_device(pg, remote_device):
    if remote_device is None:
        raise ValueError("remote device is None")

    worker_name = remote_device.worker_name()
    rank = remote_device.rank()
    device = remote_device.device()

    # Validate rank, skip validation if rank is not part of process group.
    if rank is not None and not c10d._rank_not_in_group(pg):
        pg_global_ranks = c10d.get_process_group_ranks(pg)
        if rank not in pg_global_ranks:
            raise ValueError(
                f"Global rank {rank} does not exist in input process group: {pg_global_ranks}"
            )

    if worker_name is not None:
        if not rpc._is_current_rpc_agent_set():
            raise RuntimeError(
                f"RPC framework needs to be initialized for using worker names: {worker_name}"
            )

        workers = rpc._get_current_rpc_agent().get_worker_infos()
        for worker in workers:
            if worker.name == worker_name:
                return worker.id, device

        raise ValueError(f"Invalid worker name: {worker_name}")

    return rank, device


def _validate_output_tensor_for_gather(
    my_rank: int,
    dst_rank: int,
    size: torch.Size,
    dst_tensor: torch.Tensor | None,
) -> None:
    if dst_rank == my_rank:
        if dst_tensor is None:
            raise ValueError(
                f"Argument ``dst_tensor`` must be specified on destination rank {dst_rank}"
            )
        if tuple(size) != (dst_tensor.size()):
            raise ValueError(
                f"Argument ``dst_tensor`` have size {tuple(dst_tensor.size())},"
                f"but should be {tuple(size)}"
            )
    elif dst_tensor:
        raise ValueError(
            "Argument ``dst_tensor`` must NOT be specified on non-destination ranks."
        )


def _flatten_tensor_size(size) -> torch.Size:
    """
    Checks if tensor size is valid, then flatten/return a torch.Size object.
    """
    if len(size) == 1 and isinstance(size[0], collections.abc.Sequence):
        dims = list(*size)
    else:
        dims = list(size)

    for dim in dims:
        if not isinstance(dim, int):
            raise TypeError(f"size has to be a sequence of ints, found: {dims}")

    return torch.Size(dims)


def _raise_if_mismatch(expected, actual, prop_name, ranks, is_local=True):
    if is_local:
        if not isinstance(ranks, int):
            raise AssertionError
        if expected != actual:
            raise ValueError(
                f"Local shards' tensor {prop_name} property need to be the same on rank:{ranks}! "
                f"Found one local shard tensor {prop_name}={expected}, "
                f"the other local shard tensor {prop_name}={actual}."
            )
    else:
        # compare failure check across ranks, ranks list should have two rank
        if len(ranks) != 2:
            raise AssertionError
        if expected != actual:
            raise ValueError(
                f"ShardedTensor {prop_name} property does not match from different ranks! "
                f"Found {prop_name}={expected} on rank:{ranks[0]}, "
                f"and {prop_name}={actual} on rank:{ranks[1]}."
            )


def build_metadata_from_local_shards(
    local_shards: list[Shard],
    global_size: torch.Size,
    current_rank: int,
    pg: c10d.ProcessGroup,
) -> ShardedTensorMetadata:
    if len(local_shards) <= 0:
        raise AssertionError("must have local shards!")
    local_shard_metadatas: list[ShardMetadata] = []

    first_shard_dtype = local_shards[0].tensor.dtype
    first_shard_layout = local_shards[0].tensor.layout
    first_shard_requires_grad = local_shards[0].tensor.requires_grad
    first_shard_is_pinned = local_shards[0].tensor.is_pinned()

    # 1). Validate local tensors and associated metadata
    for local_shard in local_shards:
        local_shard_tensor = local_shard.tensor
        local_shard_meta = local_shard.metadata
        local_shard_metadatas.append(local_shard_meta)
        rank, local_device = _parse_and_validate_remote_device(
            pg, local_shard_meta.placement
        )

        if (
            local_shard_tensor.layout != torch.strided
            or local_shard_tensor.layout != first_shard_layout
        ):
            raise ValueError(
                f"Only torch.strided layout is currently supported, but found "
                f"{local_shard_tensor.layout} on rank:{current_rank}!"
            )

        if not local_shard_tensor.is_contiguous():
            raise ValueError(
                "Only torch.contiguous_format memory_format is currently supported!"
            )

        if rank != current_rank:
            raise ValueError(
                f"Local shard metadata's rank does not match with the rank in its process group! "
                f"Found current rank in the process group: {current_rank}, "
                f"local ShardMetadata placement's rank: {rank}"
            )
        if local_shard_tensor.device != local_device:
            raise ValueError(
                f"Local shard tensor device does not match with local Shard's placement! "
                f"Found local shard tensor device: {local_shard_tensor.device}, "
                f"local shard metadata placement device: {local_device}"
            )

        _raise_if_mismatch(
            local_shard_meta.shard_sizes,
            list(local_shard_tensor.size()),
            "size",
            current_rank,
        )
        _raise_if_mismatch(
            local_shard_tensor.is_pinned(),
            first_shard_is_pinned,
            "pin_memory",
            current_rank,
        )
        _raise_if_mismatch(
            local_shard_tensor.dtype, first_shard_dtype, "dtype", current_rank
        )
        _raise_if_mismatch(
            local_shard_tensor.requires_grad,
            first_shard_requires_grad,
            "requires_grad",
            current_rank,
        )

    # 2). Build a "local" ShardedTensorMetadata with all local shards on this rank, then
    #    do all_gather to collect local_sharded_tensor_metadata from all ranks
    local_tensor_properties = TensorProperties(
        dtype=first_shard_dtype,
        layout=first_shard_layout,
        requires_grad=first_shard_requires_grad,
        memory_format=torch.contiguous_format,
        pin_memory=first_shard_is_pinned,
    )

    local_sharded_tensor_metadata = ShardedTensorMetadata(
        shards_metadata=local_shard_metadatas,
        size=global_size,
        tensor_properties=local_tensor_properties,
    )

    return local_sharded_tensor_metadata


def build_global_metadata(
    gathered_metadatas: Sequence[ShardedTensorMetadata | None],
    recalc_metadata: bool = False,
):
    global_sharded_tensor_metadata = None
    global_metadata_rank = 0

    # pyrefly: ignore [bad-assignment]
    for rank, rank_metadata in enumerate(gathered_metadatas):
        if rank_metadata is None:
            continue

        if global_sharded_tensor_metadata is None:
            global_sharded_tensor_metadata = copy.deepcopy(rank_metadata)
            global_metadata_rank = rank
        else:
            _raise_if_mismatch(
                global_sharded_tensor_metadata.size,
                rank_metadata.size,
                "global_size",
                [global_metadata_rank, rank],
                is_local=False,
            )

            # don't need to check layout and memory format as we already checked in local shards validation stage
            _raise_if_mismatch(
                global_sharded_tensor_metadata.tensor_properties.dtype,
                rank_metadata.tensor_properties.dtype,
                "dtype",
                [global_metadata_rank, rank],
                is_local=False,
            )

            _raise_if_mismatch(
                global_sharded_tensor_metadata.tensor_properties.requires_grad,
                rank_metadata.tensor_properties.requires_grad,
                "requires_grad",
                [global_metadata_rank, rank],
                is_local=False,
            )

            _raise_if_mismatch(
                global_sharded_tensor_metadata.tensor_properties.pin_memory,
                rank_metadata.tensor_properties.pin_memory,
                "pin_memory",
                [global_metadata_rank, rank],
                is_local=False,
            )
            # pass all validations, extend shards metadata
            global_sharded_tensor_metadata.shards_metadata.extend(
                rank_metadata.shards_metadata
            )

    if global_sharded_tensor_metadata is not None:
        if recalc_metadata:
            recalc_global_sharded_tensor_metadata(
                global_sharded_tensor_metadata,
                0,  # sharded on 0th dim
            )

        # check if shards_metadata have overlap shards
        validate_non_overlapping_shards_metadata(
            global_sharded_tensor_metadata.shards_metadata
        )

        # check if the shards_metadata is compatible with global size of the sharded tensor.
        check_tensor(
            global_sharded_tensor_metadata.shards_metadata,
            global_sharded_tensor_metadata.size,
        )
    else:
        raise ValueError("ShardedTensor have no local shards on all ranks!")

    return global_sharded_tensor_metadata


def recalc_global_sharded_tensor_metadata(
    global_sharded_tensor_metadata: ShardedTensorMetadata, sharded_dim: int
) -> None:
    # recalculate global ShardedTensorMetadata

    # reorder here in case shard metadata is not sorted on sharded_dim
    placement_idx_pairs = []
    for i, shard_metadata in enumerate(global_sharded_tensor_metadata.shards_metadata):
        if shard_metadata.placement:
            placement_idx_pairs.append((shard_metadata.placement.rank(), i))
        else:
            raise AssertionError(
                "currently only support rw, it should always have valid rank info"
            )
    sorted_idx = sorted(placement_idx_pairs)
    shard_sizes = [
        global_sharded_tensor_metadata.shards_metadata[idx].shard_sizes[sharded_dim]
        for _, idx in sorted_idx
    ]
    cum_sum = [0] + list(itertools.accumulate(shard_sizes))

    for shard_id, shard_metadata in enumerate(
        global_sharded_tensor_metadata.shards_metadata
    ):
        # update shard offset for each shard on the sharded dimension
        shard_metadata.shard_offsets[sharded_dim] = cum_sum[shard_id]
        for other_dim in range(
            len(global_sharded_tensor_metadata.shards_metadata[0].shard_sizes)
        ):
            if other_dim != sharded_dim:
                # shard offset for each shard on the unsharded dimension
                shard_metadata.shard_offsets[other_dim] = 0

    # update global size for ShardedTensorMetadata
    global_size_list = []
    for other_dim in range(
        len(global_sharded_tensor_metadata.shards_metadata[0].shard_sizes)
    ):
        if other_dim != sharded_dim:
            global_size_list.append(
                global_sharded_tensor_metadata.shards_metadata[0].shard_sizes[other_dim]
            )
        else:
            global_size_list.append(cum_sum[-1])
    global_sharded_tensor_metadata.size = torch.Size(global_size_list)
<|endoftext|># mypy: allow-untyped-defs
import functools
from typing import TYPE_CHECKING

import torch
from torch.distributed._shard.op_registry_utils import _decorator_func

from .api import (
    _CUSTOM_SHARDED_OPS,
    _SHARDED_OPS,
    Shard,
    ShardedTensor,
    ShardedTensorBase,
    ShardedTensorMetadata,
    TensorProperties,
)
from .metadata import ShardMetadata


if TYPE_CHECKING:
    from torch.distributed._shard.sharding_spec import ShardingSpec
else:
    ShardingSpec = "ShardingSpec"


def empty(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    """
    Returns a :class:`ShardedTensor` filled with uninitialized data.
        Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.contiguous_format``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    return ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )


def ones(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    """
    Returns a :class:`ShardedTensor` with the scalar value 1.
        Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    return full(
        sharding_spec,
        size,
        fill_value=1,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )


def zeros(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    """
    Returns a :class:`ShardedTensor` filled with the scalar value 0.
        Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    return full(
        sharding_spec,
        size,
        fill_value=0,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )


def full(
    sharding_spec: ShardingSpec,
    size,
    fill_value,
    *,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    """
    Creates a :class:`ShardedTensor` filled with fill_value. The tensor's dtype
        is inferred from fill_value. If dtype is specified, it will override the
        inferred type from fill_value. Needs to be called on all ranks in an SPMD fashion.
    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...):  a list, tuple, or `torch.Size` of integers defining the shape of the
            output tensor.
        fill_value (Scalar) - the value to fill the output tensor with.
    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.
    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    sharded_tensor = ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )
    torch.nn.init.constant_(sharded_tensor, fill_value)  # type: ignore[arg-type]
    return sharded_tensor


def rand(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    """
    Creates a :class:`ShardedTensor` filled with random numbers from a uniform distribution
        on the interval :math:`[0, 1)`. The shape of the tensor is defined by the
        variable argument `size`. Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...):  a list, tuple, or `torch.Size` of integers defining the shape of the
            output tensor.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    sharded_tensor = ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )
    torch.nn.init.uniform_(sharded_tensor, 0, 1)  # type: ignore[arg-type]
    return sharded_tensor


def randn(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    """
    Creates a :class:`ShardedTensor` filled with random numbers from a uniform distribution
        with mean `0` and variance `1` (also called standard normal distribution). The shape
        of the tensor is defined by the variable argument `size`. Needs to be called on all ranks
        in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...):  a list, tuple, or `torch.Size` of integers defining the shape of the
            output tensor.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    sharded_tensor = ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )
    torch.nn.init.normal_(sharded_tensor, 0, 1)  # type: ignore[arg-type]
    return sharded_tensor


def init_from_local_shards(
    local_shards: list[Shard], *global_size, process_group=None, init_rrefs=False
) -> ShardedTensor:
    """
    Creates an :class:`ShardedTensor` from local shards and the global metadata.
    Needs to be called on all ranks in an SPMD fashion.

    Args:
        local_shards (List[:class `torch.distributed._shard.sharded_tensor.Shard`]): A list
            of shards that represent the local shards on this rank.
        global_size (int...):  a list, tuple, or `torch.Size` of integers defining the
            shape of the overall sharded tensor.

    Keyword args:
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object handle on this rank


    Examples:
        Suppose we want construct a sharded tensor on two ranks, global size = (10, 5),
        each shard have a (5, 5) local tensor, we can do it like below:

        on rank 0:
        >>> # xdoctest: +SKIP("not distributed")
        >>> local_shard_metadata = ShardMetadata(
        >>>     shard_offsets=[0, 0],
        >>>     shard_lengths=[5, 5],
        >>>     placement="rank:0/cuda:0"
        >>> )
        >>> local_shards = [Shard(torch.randn(5, 5), local_shard_metadata)]
        >>> sharded_tensor = init_from_local_shards(local_shards, [10, 5])

        on rank 1:
        >>> # xdoctest: +SKIP("not distributed")
        >>> local_shard_metadata = ShardMetadata(
        >>>     shard_offsets=[5, 0],
        >>>     shard_lengths=[5, 5],
        >>>     placement="rank:1/cuda:1"
        >>> )
        >>> local_shards = [Shard(torch.randn(5, 5), local_shard_metadata)]
        >>> sharded_tensor = init_from_local_shards(local_shards, [10, 5])
    """
    return ShardedTensor._init_from_local_shards(
        local_shards, *global_size, process_group=process_group, init_rrefs=init_rrefs
    )


def state_dict_hook(module, destination, prefix, local_metadata):
    """
    Hook to add ShardedTensor to Module's ``state_dict``. Needs to be
    registered to the Module using
    :meth:`torch.nn.Module._register_state_dict_hook`.
    """
    for submodule_name, submodule in module.named_modules():
        for attr_name, attr in submodule.__dict__.items():
            if isinstance(attr, ShardedTensor):
                mod_prefix = prefix + submodule_name
                key = mod_prefix + ("." if mod_prefix else "") + attr_name
                destination[key] = attr


def pre_load_state_dict_hook(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """
    Pre-load state dict hook to add ShardedTensor to the module.
    """
    for submodule_name, submodule in module.named_modules():
        for attr_name in submodule.__dict__:
            mod_prefix = prefix + submodule_name
            key = mod_prefix + ("." if mod_prefix else "") + attr_name
            if key in state_dict:
                if isinstance(state_dict[key], ShardedTensor):
                    setattr(submodule, attr_name, state_dict[key])


def custom_sharded_op_impl(func):
    """
    Provides a way for users to write their own custom sharded operator. This
    can be used to override existing ShardedTensor operators or write a new
    one not supported by ShardedTensor. If the operator in question is covered
    by ``__torch_function__`` dispatch and has a ShardedTensor as any of its
    parameters, the function provided will be invoked for that operator.

    Example::
        >>> # xdoctest: +SKIP
        >>> @custom_sharded_op_impl(torch.nn.functional.linear)
        >>> def my_custom_sharded_linear(types, args, kwargs, process_group):
        >>>     ...
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> input = torch.rand(10, 32)
        >>> weight = sharded_tensor.rand(32, 16)
        >>> bias = torch.rand(16)
        >>> # This will call 'my_custom_sharded_linear'
        >>> torch.nn.functional.linear(input, weight, bias)

    The types, args and kwargs parameters are the same parameters that are
    passed to ``__torch_function__`` dispatch API
    (https://pytorch.org/docs/stable/notes/extending.html#extending-torch).
    There is an additional ``process_group`` parameter which is the
    process_group used for the ShardedTensor and can be used by
    implementations for communications within a sharded implementation.

    Args:
        func(Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.nn.functional.linear)
    """
    return functools.partial(_decorator_func, op=func, op_table=_CUSTOM_SHARDED_OPS)


def _sharded_op_impl(func):
    """
    Decorator to register a default sharded op.
    """
    return functools.partial(_decorator_func, op=func, op_table=_SHARDED_OPS)


# Import all builtin sharded ops
from ._ops import *  # noqa: F403
<|endoftext|>from dataclasses import dataclass

import torch
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed.remote_device import _remote_device


@dataclass
class Shard:
    """
    Container which holds the data for a shard as a Tensor and also
    the associated metadata for that shard.

    Args:
        tensor(torch.Tensor): Local tensor for the shard.
        metadata(:class `torch.distributed._shard.sharded_tensor.ShardMetadata`):
            The metadata for the shard, including offsets, lengths and device placement.
    """

    __slots__ = ["tensor", "metadata"]
    tensor: torch.Tensor
    metadata: ShardMetadata

    def __post_init__(self) -> None:
        # verification between local tensor and metadata
        if list(self.tensor.size()) != self.metadata.shard_sizes:
            raise ValueError(
                "Shard tensor size does not match with metadata.shard_lengths! "
                f"Found shard tensor size: {list(self.tensor.size())}, "
                f"metadata.shard_lengths: {self.metadata.shard_sizes}, "
            )
        placement_device = self.metadata.placement
        if (
            placement_device is not None
            and placement_device.device() != self.tensor.device
        ):
            raise ValueError(
                f"Local shard tensor device does not match with local Shard's placement! "
                f"Found local shard tensor device: {self.tensor.device}, "
                f"local shard metadata placement device: {placement_device.device()}"
            )

    @classmethod
    def from_tensor_and_offsets(
        cls, tensor: torch.Tensor, shard_offsets: list[int], rank: int
    ) -> "Shard":
        """
        Creates a Shard of a ShardedTensor from a local torch.Tensor, shard_offsets and rank.

        Args:
            tensor(torch.Tensor): Local tensor for the shard.
            shard_offsets(List[int]): List of integers specify the offset
                of the shard on each dimension.
            rank(int): Specify the rank for the shard.
        """
        shard_sizes = list(tensor.size())
        placement = _remote_device(f"rank:{rank}/{str(tensor.device)}")
        shard_meta = ShardMetadata(
            shard_offsets=shard_offsets, shard_sizes=shard_sizes, placement=placement
        )
        return Shard(tensor, shard_meta)
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations  # type: ignore[attr-defined]

import copy
import operator
import threading
import warnings
import weakref
from dataclasses import dataclass
from functools import reduce
from typing import cast, TYPE_CHECKING
from typing_extensions import deprecated

import torch
import torch.distributed as dist
import torch.distributed._shard.sharding_spec as shard_spec
from torch._utils import _get_device_module
from torch.distributed import distributed_c10d, rpc
from torch.distributed._shard._utils import DEPRECATE_MSG
from torch.distributed._shard.sharding_spec._internals import (
    check_tensor,
    validate_non_overlapping_shards_metadata,
)
from torch.distributed._shard.sharding_spec.api import (
    _dispatch_custom_op,
    _has_custom_op,
)
from torch.distributed.remote_device import _remote_device
from torch.utils import _pytree as pytree

from .metadata import ShardedTensorMetadata, TensorProperties
from .reshard import reshard_local_shard, reshuffle_local_shard
from .shard import Shard
from .utils import (
    _flatten_tensor_size,
    _parse_and_validate_remote_device,
    _validate_output_tensor_for_gather,
    build_global_metadata,
    build_metadata_from_local_shards,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.distributed._shard.metadata import ShardMetadata


# Tracking for sharded tensor objects.
_sharded_tensor_lock = threading.Lock()
_sharded_tensor_current_id = 0
_sharded_tensor_map: dict[int, weakref.ReferenceType[ShardedTensor]] = {}

# Default sharded ops
_SHARDED_OPS: dict[Callable, Callable] = {}

# Customized user ops
_CUSTOM_SHARDED_OPS: dict[Callable, Callable] = {}


def _register_remote_shards(
    sharded_tensor_id: int, rrefs: list[rpc.RRef[Shard]], rpc_rank: int
):
    with _sharded_tensor_lock:
        if sharded_tensor_id not in _sharded_tensor_map:
            raise RuntimeError(
                f"Could not find sharded_tensor_id: {sharded_tensor_id} in map: {_sharded_tensor_map.keys()}"
            )

        sharded_tensor = _sharded_tensor_map[sharded_tensor_id]()
        if sharded_tensor is None:
            raise RuntimeError("ShardedTensor weakref has been deallocated")
        else:
            sharded_tensor._register_remote_shards(rrefs, rpc_rank)


class ShardedTensorBase(torch.Tensor):
    _sharding_spec: shard_spec.ShardingSpec
    _metadata: ShardedTensorMetadata
    _local_shards: list[Shard]

    def __new__(cls, sharding_spec: shard_spec.ShardingSpec, *size, **kwargs):
        # Use __new__ to construct a wrapper tensor, for recording tensor
        # properties and logging purposes.
        torch._C._log_api_usage_once("torch.distributed._shard.sharded_tensor")

        # check sharding spec and build sharded tensor metadata
        if not isinstance(sharding_spec, shard_spec.ShardingSpec):
            raise ValueError(f"Expecting ShardingSpec but got: {type(sharding_spec)}")

        sizes = _flatten_tensor_size(size)
        dtype = kwargs["dtype"]
        layout = kwargs["layout"]
        pin_memory = kwargs["pin_memory"]
        requires_grad = kwargs["requires_grad"]

        if dtype is None:
            dtype = torch.get_default_dtype()

        tensor_properties = TensorProperties(
            dtype, layout, requires_grad, pin_memory=pin_memory
        )
        sharded_tensor_metadata = sharding_spec.build_metadata(
            sizes, tensor_properties=tensor_properties
        )

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            sizes,
            dtype=dtype,
            layout=layout,
            pin_memory=pin_memory,
            requires_grad=requires_grad,
        )
        # set sharding spec
        r._sharding_spec = sharding_spec
        # set metadata
        r._metadata = sharded_tensor_metadata
        # set local shards
        r._local_shards = []
        return r

    def metadata(self) -> ShardedTensorMetadata:
        """
        Returns a :class:`ShardedTensorMetadata` object corresponding to the
        metadata for the entire tensor.
        """
        return self._metadata

    def local_shards(self) -> list[Shard]:
        """
        Returns a list of :class:`Shard' corresponding to the
        local shards for this rank. Returns an empty list if the current rank
        does not host any shards for this Tensor.
        """
        return self._local_shards

    @classmethod
    def _init_from_local_shards_and_global_metadata(
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        sharding_spec=None,
    ) -> ShardedTensorBase:
        """
        Initialize a ShardedTensorBase with local shards and a global
        ShardedTensorMetadata built on each rank.
        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if len(shards_metadata) == 0:
            raise ValueError("shards_metadata must not be empty!")

        if tensor_properties.layout != torch.strided:
            raise ValueError("Only torch.strided layout is currently supported")

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor_base = ShardedTensorBase.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # check if shards_metadata have overlap shards
        validate_non_overlapping_shards_metadata(shards_metadata)

        # check if the shards_metadata is compatible with overall size of the sharded tensor.
        check_tensor(shards_metadata, list(sharded_tensor_metadata.size))

        # done validation, add local_shards
        sharded_tensor_base._local_shards = local_shards
        return sharded_tensor_base

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        raise RuntimeError(
            f"A {cls.__name__} object is being used from c++ while calling {func.__module__}.{func.__name__} "
            "but the there is no custom __torch_dispatch__ implementation for it."
        )


class ShardedTensor(ShardedTensorBase):
    """
    ShardedTensor is an torch.Tensor subclass to represent Tensors that are sharded
    across multiple devices and multiple processes.

    ShardedTensor is initialized in an SPMD like fashion where each rank
    initializes the ShardedTensor. The ShardedTensor object on each rank
    then only stores the local shard for the Tensor and provides global
    metadata for all the shards.

    ShardedTensor doesn't provide any Tensor like operations but is a wrapper
    providing the Tensor representing the local shard and the global metadata.
    Using these, users can build their custom distributed._sharded computations
    on top of this primitive. The local shards are all initialized using the
    create_op specified by tensor_init_params.create_op, e.g., torch.ones, or
    torch.empty

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
                Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.contiguous_format``.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    .. note:: ShardedTensor uses collectives to do various operations, i.e. it
        uses all_gather to do cross rank validations. For NCCL-based process
        groups, internal tensor representations of objects must be moved to the
        GPU device before communication takes place. In this case, the device
        used is given by ``torch.cuda.current_device()`` and it is the user's
        responsibility to ensure that this is set so that each rank has an
        individual GPU, via ``torch.cuda.set_device()``

    """

    def __new__(cls, sharding_spec: shard_spec.ShardingSpec, *size, **kwargs):
        self = super().__new__(cls, sharding_spec, *size, **kwargs)
        return self

    def __init__(
        self,
        sharding_spec: shard_spec.ShardingSpec,
        *size,
        dtype=None,
        layout=torch.strided,
        requires_grad=False,
        pin_memory=False,
        memory_format=torch.contiguous_format,
        process_group=None,
        init_rrefs=False,
    ):
        # prepare initialization, initialize fields like
        # _process_group, _local_shards, etc.
        self._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        if layout != torch.strided:
            raise ValueError("Only torch.strided layout is currently supported")

        if memory_format != torch.contiguous_format:
            raise ValueError(
                "Only torch.contiguous_format memory_format is currently supported"
            )

        self._metadata.tensor_properties.memory_format = memory_format

        current_rank = dist.get_rank()  # global rank

        for shard_metadata in self._metadata.shards_metadata:
            rank, device = _parse_and_validate_remote_device(
                self._process_group, shard_metadata.placement
            )
            if rank == current_rank:
                local_tensor = _create_tensor_from_params(
                    shard_metadata.shard_sizes,
                    local_device=device,
                    tensor_properties=self._metadata.tensor_properties,
                )
                self._local_shards.append(Shard(local_tensor, shard_metadata))

        # do post initialization (i.e. register sharded_tensor_id, initialize_rpc)
        self._post_init()

    def _prepare_init(self, process_group=None, init_rrefs=False):
        self._init_rrefs = init_rrefs
        self._sharded_tensor_id = None

        self._process_group = self._normalize_pg(process_group)
        self._remote_shards: dict[int, list[rpc.RRef[Shard]]] = {}

    def _post_init(self):
        # Initialize RPC if available.
        if self._init_rrefs:
            with _sharded_tensor_lock:
                global _sharded_tensor_current_id, _sharded_tensor_map
                self._sharded_tensor_id = _sharded_tensor_current_id
                _sharded_tensor_map[self._sharded_tensor_id] = weakref.ref(self)
                _sharded_tensor_current_id += 1

            if not rpc._is_current_rpc_agent_set():
                raise RuntimeError(
                    "RPC Framework needs to be initialized using"
                    " torch.distributed.rpc.init_rpc if init_rrefs is set to True"
                )
            self._init_rpc()

    def __del__(self):
        # Clean up the global map.
        with _sharded_tensor_lock:
            global _sharded_tensor_current_id, _sharded_tensor_map
            if (
                hasattr(self, "_sharded_tensor_id")
                and self._sharded_tensor_id in _sharded_tensor_map
            ):
                _sharded_tensor_map.pop(self._sharded_tensor_id)  # type: ignore[call-overload]

    def _init_rpc(self):
        # Validate PG and RPC ranks match.
        pg_rank = dist.get_rank()
        rpc_rank = rpc.get_worker_info().id
        if pg_rank != rpc_rank:
            raise ValueError(
                f"Default ProcessGroup and RPC ranks must be "
                f"the same for ShardedTensor, found process group rank: "
                f"{pg_rank} and RPC rank: {rpc_rank}"
            )

        self._remote_shards = {}

        # Gather all the sharded tensor ids.
        worker_infos = rpc._get_current_rpc_agent().get_worker_infos()
        rank_to_name = {}
        name_to_rank = {}

        for worker_info in worker_infos:
            rank_to_name[worker_info.id] = worker_info.name
            name_to_rank[worker_info.name] = worker_info.id

        all_tensor_ids = rpc.api._all_gather(self._sharded_tensor_id)

        # Share the local shards to the entire world.
        futs = []
        rpc_rank = rpc.get_worker_info().id
        for rank in range(dist.get_world_size()):
            # Skip self.
            if rank == dist.get_rank():
                continue

            if len(self.local_shards()) != 0:
                rrefs: list[rpc.RRef[Shard]] = [
                    rpc.RRef(shard) for shard in self.local_shards()
                ]
                fut = rpc.rpc_async(
                    rank,
                    _register_remote_shards,
                    args=(all_tensor_ids[rank_to_name[rank]], rrefs, rpc_rank),
                )
                futs.append(fut)

        torch.futures.wait_all(futs)

        # Barrier for all RPCs to finish on all ranks.
        rpc.api._all_gather(None)

    def _get_preferred_device(self) -> torch.device:
        """
        Return the preferred device to be used when creating tensors for collectives.
        This method takes into account the associated process group
        """
        backend = dist.get_backend(self._process_group)
        if backend == dist.Backend.NCCL:
            return torch.device(torch.cuda.current_device())
        elif backend == dist.Backend.GLOO:
            return torch.device("cpu")
        else:
            backend_config = dist.BackendConfig(backend)
            for device, backend_str in backend_config.get_device_backend_map().items():
                if backend_str == backend and device != "cpu":
                    return torch.device(
                        device, _get_device_module(device).current_device()
                    )
        return torch.device("cpu")

    def gather(  # type: ignore[override]
        self,
        dst: int = 0,
        out: torch.Tensor | None = None,
        enforce_dtype: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Creates a full :class:`Tensor` on rank ``dst`` by gathering all shards of the
        sharded tensor.

        The API needs to be called on all ranks in SPMD fashion. All ranks should have
        the same ``dst``. ``out`` should be a tensor of the same size as the overall
        size of the sharded tensor on ``dst`` and ``None`` on all other ranks.

        Args:
            dst(int): The rank where full tensor is constructed.
                Default: 0
            out (:class `torch.Tensor`, optional): The output full tensor.
                Must be provided ONLY on ``dst`` rank.
                Default: ``None``
            enforce_dtype (bool): Deprecated, please use dtype instead.  Force the
                gathered tensors to be the same type as input and output.
            dtype (torch.dtype): Force the gathered tensors to be this dtype.
                Default: ``None``
        """

        def shard_size(shard_md):
            return reduce(operator.mul, shard_md.shard_sizes)  # type: ignore[attr-defined]

        if enforce_dtype:
            warnings.warn(
                "`enforce_dtype` is deprecated. Please use `dtype` instead.",
                FutureWarning,
                stacklevel=2,
            )

        rank = dist.get_rank(self._process_group)
        full_size = self.metadata().size
        _validate_output_tensor_for_gather(rank, dst, full_size, out)

        local_shards = self.local_shards()
        world_size = dist.get_world_size(self._process_group)
        rank_sizes = [0 for _ in range(world_size)]
        max_rank_size = 0
        shard_placement: dict[ShardMetadata, tuple[int, int]] = {}
        # collect sizes
        for shard_md in self.metadata().shards_metadata:
            shard_rank = cast(_remote_device, shard_md.placement).rank()
            if shard_rank is None:
                raise AssertionError

            shard_placement[shard_md] = (shard_rank, rank_sizes[shard_rank])
            rank_sizes[shard_rank] += shard_size(shard_md)
            max_rank_size = max(max_rank_size, rank_sizes[shard_rank])

        gather_list: list[torch.Tensor] | None
        if rank == dst:
            if out is None:
                raise AssertionError
            if enforce_dtype:
                # enforce_dtype is deprecated.  Do it for backward compatibility.
                dtype = out.dtype
            # TODO make it as a view of out tensor
            gather_list = [
                torch.empty((max_rank_size,), device=out.device, dtype=dtype)
                for _ in range(world_size)
            ]
        else:
            gather_list = None

        with torch.no_grad():
            if enforce_dtype and len(local_shards) > 0:
                # enforce_dtype is deprecated.  Do it for backward compatibility.
                dtype = local_shards[0].tensor.dtype
            data = torch.empty(
                max_rank_size, device=self._get_preferred_device(), dtype=dtype
            )

            for shard in local_shards:
                src = shard.tensor.flatten()
                if src.nelement() == 0:
                    warnings.warn(
                        "Gathering a tensor with zero elements on rank " + str(rank),
                        stacklevel=2,
                    )
                    continue
                shard_offset = shard_placement[shard.metadata][1]
                data[shard_offset : shard_offset + src.numel()].copy_(src)

        dist.gather(
            tensor=data,
            gather_list=gather_list,
            dst=dst,
            group=self._process_group,
        )
        if rank != dst:
            return
        # In _validate_output_tensor_for_gather, we raise if out == None and rank == dst
        out = cast(torch.Tensor, out)
        if gather_list is None:
            raise AssertionError

        full_size = self.metadata().size
        dims = len(full_size)
        for shard_md in self.metadata().shards_metadata:
            rank, rank_offset = shard_placement[shard_md]
            tensor = gather_list[rank]
            tensor = tensor[rank_offset : rank_offset + shard_size(shard_md)]
            tensor = tensor.view(shard_md.shard_sizes)

            out_narrow_view = out
            for dim in range(dims):
                out_narrow_view = out_narrow_view.narrow(
                    dim,
                    shard_md.shard_offsets[dim],
                    shard_md.shard_sizes[dim],
                )

            out_narrow_view.copy_(tensor)

    def cpu(
        self, memory_format=torch.preserve_format, process_group=None
    ) -> ShardedTensor:
        """
        Returns a copy of this object in CPU memory.

        If this ShardedTensor is already on CPU memory, then no copy is
        performed and original object is returned.

        .. note:: When moving a ShardedTensor from GPU to CPU, the ShardedTensor might
            need to be managed by a different type of ProcessGroup(i.e. ProcessGroupGloo),
            it is the user's responsibility to explicitly pass in a new process_group that
            is compatible with CPU.
        """
        # TODO: make this a __torch_function__ op once ShardedTensor becomes a
        # torch.Tensor subclass, see https://github.com/pytorch/pytorch/issues/75402
        if (
            memory_format != torch.preserve_format
            and memory_format != torch.contiguous_format
        ):
            raise RuntimeError(
                "Only `torch.contiguous_format` or "
                "`torch.preserve_format` is supported!"
            )
        all_on_cpu = True
        for meta in self.metadata().shards_metadata:
            all_on_cpu &= meta.placement.device().type == "cpu"  # type: ignore[union-attr]

        # if every shard is already on CPU, return the original object
        if all_on_cpu:
            return self

        # if not, returns a copy of this object on CPU
        list_shards: list[Shard] = []
        # move all local shards to cpu, and change metadata
        for shard in self._local_shards:
            cpu_tensor = shard.tensor.cpu(memory_format=memory_format)  # type: ignore[call-arg]
            metadata = copy.deepcopy(shard.metadata)
            metadata.placement._device = torch.device("cpu")  # type: ignore[union-attr]
            list_shards.append(Shard(cpu_tensor, metadata))

        st_meta = copy.deepcopy(self.metadata())
        for meta in st_meta.shards_metadata:
            if meta.placement.device().type != "cpu":  # type: ignore[union-attr]
                meta.placement._device = torch.device("cpu")  # type: ignore[union-attr]

        pg = self._process_group if process_group is None else process_group
        st_cpu = ShardedTensor._init_from_local_shards_and_global_metadata(
            list_shards,
            sharded_tensor_metadata=st_meta,
            process_group=pg,
            init_rrefs=self._init_rrefs,
        )
        return st_cpu

    def cuda(
        self,
        device=None,
        non_blocking=False,
        memory_format=torch.preserve_format,
        process_group=None,
    ) -> ShardedTensor:
        """
        Returns a copy of this object in CUDA memory, if the original ShardedTensor
        is on CPU, we will move the local shard to the current GPU device of each
        process in a SPMD fashion.
        If this ShardedTensor is already on CUDA memory and local shards on each rank are
        already on current device, we still returns a new ShardedTensor object with new
        metadata, but no underlying data movements are performed.
        .. note:: When moving a ShardedTensor from CPU to GPU, the ShardedTensor might
            need to be managed by a different type of ProcessGroup(i.e. ProcessGroupNCCL),
            it is the user's responsibility to explicitly pass in a new process_group that
            is compatible with GPU.
        """
        if (
            memory_format != torch.preserve_format
            and memory_format != torch.contiguous_format
        ):
            raise RuntimeError(
                "Only `torch.contiguous_format` or "
                "`torch.preserve_format` is supported!"
            )

        if device is not None:
            device = torch.device(device) if isinstance(device, str) else device
            if not (
                isinstance(device, torch.device)
                and device.index == torch.accelerator.current_device_index()
            ):
                raise AssertionError(
                    f"Only device without device id (e.g. 'cpu' or '{device.type}') "
                    f"is expected for ShardedTensor!"
                )
        current_device = torch.device(torch.accelerator.current_device_index())
        # returns a copy of ShardedTensor on accelerator current device
        list_shards: list[Shard] = []
        # move all local shards to current device, and change metadata
        # if local shards already on the current device, there's no
        # real data movement, only the metadata are copied.
        for shard in self._local_shards:
            new_tensor = shard.tensor.to(
                device=current_device,
                non_blocking=non_blocking,
                memory_format=memory_format,
            )  # type: ignore[call-arg]
            metadata = copy.deepcopy(shard.metadata)
            metadata.placement._device = current_device  # type: ignore[union-attr]

            list_shards.append(Shard(new_tensor, metadata))

        st_meta = copy.deepcopy(self.metadata())
        for meta in st_meta.shards_metadata:
            if meta.placement.device().type != current_device.type:  # type: ignore[union-attr]
                meta.placement._device = current_device  # type: ignore[union-attr]

        pg = self._process_group if process_group is None else process_group
        # we need to use `init_from_local_shards` to communicate between ranks
        # and update the sharding spec/shards metadata.
        return ShardedTensor._init_from_local_shards_and_global_metadata(
            list_shards,
            sharded_tensor_metadata=st_meta,
            process_group=pg,
            init_rrefs=self._init_rrefs,
        )

    def to(self, *args, **kwargs) -> ShardedTensor:
        current_device: torch.device
        if self._local_shards:
            current_device = self._local_shards[0].tensor.device
        elif self._process_group._get_backend_name() == "gloo":
            current_device = torch.device("cpu")
        else:
            current_device = torch.device(torch.accelerator.current_device_index())
        current_dtype = self.dtype
        device_to = current_device
        dtype_to = current_dtype
        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype_to = args[0]
            elif isinstance(args[0], torch.device):
                device_to = args[0]
            elif isinstance(args[0], (str, int)):
                device_to = torch.device(args[0])
            elif isinstance(args[0], torch.Tensor):
                dtype_to = args[0].dtype
                device_to = args[0].device
            else:
                raise RuntimeError(f"ShardedTensor.to() have wrong arguments: {args}")
        elif len(args) == 2:
            device_to, dtype_to = args
        else:
            dtype_to = kwargs.get("dtype", current_dtype)
            device_to = kwargs.get("device", current_device)

        device_to = (
            torch.device(device_to) if isinstance(device_to, (str, int)) else device_to
        )

        if device_to.type != "cpu":
            # if device_to set to accelerator, set to current device even
            # if user specifies the device index.
            current_idx = torch.accelerator.current_device_index()
            if device_to.index != current_idx:
                warnings.warn(
                    "ShardedTensor.to only move tensor to its current device"
                    "If you want to put to different device, use `reshard` instead.",
                    stacklevel=2,
                )
            device_to = torch.device(current_idx)

        copy_tensor = kwargs.get("copy", False)
        non_blocking = kwargs.get("non_blocking", False)
        memory_format = kwargs.get("memory_format", torch.preserve_format)
        process_group = kwargs.get("process_group")

        if (
            not copy_tensor
            and dtype_to == current_dtype
            and device_to == current_device
        ):
            # already have correct dtype and device, return itself
            return self

        # returns a copy of ShardedTensor on target device
        list_shards: list[Shard] = []

        for shard in self._local_shards:
            new_tensor = shard.tensor.to(  # type: ignore[call-overload]
                device=device_to,
                dtype=dtype_to,
                non_blocking=non_blocking,
                copy=copy_tensor,
                memory_format=memory_format,
            )
            metadata = copy.deepcopy(shard.metadata)
            if metadata.placement is not None:
                metadata.placement._device = device_to
            list_shards.append(Shard(new_tensor, metadata))

        # update metadata
        st_meta = copy.deepcopy(self.metadata())
        st_meta.tensor_properties.dtype = dtype_to
        for meta in st_meta.shards_metadata:
            meta.placement._device = device_to  # type: ignore[union-attr]

        pg = self._process_group if process_group is None else process_group
        # we need to use `init_from_local_shards` to communicate between ranks
        # and update the sharding spec/shards metadata.
        st_to = ShardedTensor._init_from_local_shards_and_global_metadata(
            list_shards,
            sharded_tensor_metadata=st_meta,
            process_group=pg,
            init_rrefs=self._init_rrefs,
        )
        return st_to

    @classmethod
    def _normalize_pg(
        cls, process_group: dist.ProcessGroup | None
    ) -> dist.ProcessGroup:
        if process_group is not None:
            return process_group
        return distributed_c10d._get_default_group()

    @classmethod
    def _init_from_local_shards(
        cls,
        local_shards: list[Shard],
        *global_size,
        process_group=None,
        init_rrefs=False,
    ):
        # recalc metadata handles special ST creation cases like each rank only has tensor available
        # caller need to provide None on the unknown dimension of the global size
        # We will change None into zeros and go through the same amount of checks as before to create ST
        # and use all_gather to calculate the offsets and global size for metadata
        # It is compatible with the current use case since, conventionally we don't pass None as global size
        # Therefore the old path won't trigger the new feature
        recalc_metadata = False
        for dim in global_size:
            if dim is None:
                recalc_metadata = True
        if recalc_metadata:
            global_size = tuple(
                0 if dim_size is None else dim_size for dim_size in global_size
            )
        # STEP 1: Validate the Shardmetadatas locally
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank
        world_size = dist.get_world_size(process_group)

        local_sharded_tensor_metadata: ShardedTensorMetadata | None = None
        global_tensor_size = _flatten_tensor_size(global_size)

        if len(local_shards) > 0:
            local_sharded_tensor_metadata = build_metadata_from_local_shards(
                local_shards, global_tensor_size, current_rank, process_group
            )

        # STEP 2. Validate metadata across ranks, and build a global sharded tensor
        # metadata by gathering local ShardedTensorMetadata
        gathered_metadatas: list[ShardedTensorMetadata | None] = []
        if world_size > 1:
            gathered_metadatas = [None for _ in range(world_size)]

            dist.all_gather_object(
                gathered_metadatas, local_sharded_tensor_metadata, group=process_group
            )
        else:
            gathered_metadatas = [local_sharded_tensor_metadata]

        global_sharded_tensor_metadata = build_global_metadata(
            gathered_metadatas, recalc_metadata=recalc_metadata
        )
        if recalc_metadata:
            # for recalc use cases, we only support rw for now, limit the blast radius
            # will modify here once we support more sharding type
            if not (
                len(local_shards) > 0
                and len(global_sharded_tensor_metadata.shards_metadata) > current_rank
            ):
                raise AssertionError(
                    f"# for metadata recalculation, local_shards must be larger than 0 "
                    f"actual:{len(local_shards)}, # glb metadata must be greater than any rank id, "
                    f"# metadata:{len(global_sharded_tensor_metadata.shards_metadata)}, rank id:{current_rank}"
                )
            local_md = [
                shard_md
                for shard_md in global_sharded_tensor_metadata.shards_metadata
                if shard_md.placement.rank() == current_rank
            ]
            if len(local_md) != 1:
                raise AssertionError(
                    f"should has and only has one metadata for local rank, actual:{local_md}"
                )
            local_shards[0].metadata = local_md[0]
        tensor_properties = global_sharded_tensor_metadata.tensor_properties

        # STEP 3: Validation done, create the actual ShardedTensor and populate fields
        # prepare initialization
        spec = shard_spec._infer_sharding_spec_from_shards_metadata(
            global_sharded_tensor_metadata.shards_metadata
        )
        sharded_tensor = cls.__new__(
            cls,
            spec,
            global_sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # attach local_shards to the ShardedTensor created
        sharded_tensor._local_shards = local_shards

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    @classmethod
    @deprecated(DEPRECATE_MSG, category=FutureWarning)
    def _init_from_local_tensor(
        cls,
        local_tensor: torch.Tensor,
        sharding_spec: shard_spec.ShardingSpec,
        *global_size: Sequence[int],
        process_group: dist.ProcessGroup | None = None,
        init_rrefs=False,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor given only one local tensor, global sharded tensor
        size and sharding spec on each rank.

        Args:
            local_tensor (Tensor): Single tensor of local shard stored in each rank.
            sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`):
                The specification describing how to shard the Tensor.
            global_size (Sequence[int]): Size of the sharded tensor.
            process_group (ProcessGroup, optional): The process group to aggregate on.
                Default: None
            init_rrefs (bool, optional): Whether or not to initialize
                :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
                Need to initialize the RPC Framework if specified as ``True``.
                Default: ``False``.

        Returns:
            A :class:`ShardedTensor` sharded based on the given sharding_spec with local
                tensor stored in the current rank.

        Examples:
            >>> # xdoctest: +SKIP
            >>> # All tensors below are of torch.int64 type.
            >>> # We have 2 process groups, 2 ranks.
            >>> tensor = torch.arange(2, dtype=torch.int64) + 1 + 2 * rank
            >>> local_tensor = torch.unsqueeze(torch.cat([tensor, tensor + 2]))
            >>> local_tensor
            tensor([[1, 2, 3, 4]]) # Rank 0
            tensor([[3, 4, 5, 6]]) # Rank 1
            >>> sharding_dim = 0
            >>> sharding_spec = ChunkShardingSpec(
                    dim=sharding_dim,
                    placements=[
                        "rank:0/cuda:0",
                        "rank:1/cuda:1",
                    ],
                )
            >>> st = ShardedTensor._init_from_local_tensor(
            ...     local_tensor, sharding_spec, [2, 4]
            ... )
            >>> st
            ShardedTensor(
                ShardedTensorMetadata(
                    shards_metadata=[
                        ShardMetadata(shard_offsets=[0, 0], shard_sizes=[1, 4], placement=rank:0/cuda:0),
                        ShardMetadata(shard_offsets=[1, 0], shard_sizes=[1, 4], placement=rank:1/cuda:1),
                    ],
                    size=torch.Size([2, 4])
            )
            >>> st.local_tensor()
            tensor([1, 2, 3, 4]) # Rank 0
            tensor([3, 4, 5, 6]) # Rank 1

        Warning: This API is experimental and subject to change. It lacks of a fully across
                 rank validations, and we only validate the local shard on the current rank.
                 We fully rely on the user to ensure local tensor is sharded based on the
                 sharding spec.
        """
        if not local_tensor.is_contiguous():
            raise ValueError("local_tensor is not a contiguous Tensor.")

        global_tensor_size = _flatten_tensor_size(global_size)
        tensor_properties = TensorProperties(
            dtype=local_tensor.dtype,
            layout=local_tensor.layout,
            requires_grad=local_tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=local_tensor.is_pinned(),
        )
        sharded_tensor_metadata = sharding_spec.build_metadata(
            global_tensor_size, tensor_properties
        )

        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        local_shards: list[Shard] = []
        for shard_metadata in sharded_tensor_metadata.shards_metadata:
            rank, _device = _parse_and_validate_remote_device(
                process_group, shard_metadata.placement
            )
            if rank == current_rank:
                local_shards.append(Shard(local_tensor, shard_metadata))

        # TODO: figure out what the API should behave when some rank have no shard
        # see https://github.com/pytorch/pytorch/issues/7313
        return ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards,
            sharded_tensor_metadata,
            process_group=process_group,
            init_rrefs=init_rrefs,
            sharding_spec=sharding_spec,
        )

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(
                process_group, shard_metadata.placement
            )

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        if len(local_shards) != len(local_shard_metadatas):
            raise RuntimeError(
                f"Number of local shards ({len(local_shards)}) does not match number of local "
                f"shards metadata in sharded_tensor_metadata ({len(local_shard_metadatas)}) "
                f"on rank ({current_rank}) "
            )

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if len(shards_metadata) == 0:
            raise ValueError("shards_metadata must not be empty!")

        if tensor_properties.layout != torch.strided:
            raise ValueError("Only torch.strided layout is currently supported")

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        def _raise_if_mismatch(expected, actual, prop_name, rank, is_property=False):
            tensor_property_or_metadata = (
                "tensor property" if is_property else "local ShardMetadata"
            )
            if expected != actual:
                raise ValueError(
                    f"Local shards' tensor {prop_name} property is incompatible with "
                    f"{tensor_property_or_metadata} on rank {rank}: "
                    f"{tensor_property_or_metadata} {prop_name}={expected}, "
                    f"local shard tensor {prop_name}={actual}."
                )

        for shard in local_shards:
            shard_meta = shard.metadata
            local_shard_tensor = shard.tensor
            placement = shard_meta.placement
            if placement is None:
                raise AssertionError("Must specify placement for `Shard`!")
            rank = placement.rank()
            local_device = placement.device()

            _raise_if_mismatch(
                tensor_properties.layout,
                local_shard_tensor.layout,
                "layout",
                rank,
                True,
            )
            if not local_shard_tensor.is_contiguous():
                raise ValueError(
                    "Only torch.contiguous_format memory_format is currently supported"
                )

            _raise_if_mismatch(
                shard_meta.shard_sizes,
                list(local_shard_tensor.size()),
                "size",
                rank,
            )
            _raise_if_mismatch(
                tensor_properties.pin_memory,
                local_shard_tensor.is_pinned(),
                "pin_memory",
                rank,
                True,
            )
            _raise_if_mismatch(local_device, local_shard_tensor.device, "device", rank)
            _raise_if_mismatch(
                tensor_properties.dtype,
                local_shard_tensor.dtype,
                "dtype",
                rank,
                True,
            )
            _raise_if_mismatch(
                tensor_properties.requires_grad,
                local_shard_tensor.requires_grad,
                "requires_grad",
                rank,
                True,
            )

        # check if shards_metadata have overlap shards
        validate_non_overlapping_shards_metadata(shards_metadata)

        # check if the shards_metadata is compatible with overall size of the sharded tensor.
        check_tensor(shards_metadata, list(sharded_tensor_metadata.size))

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    def sharding_spec(self) -> shard_spec.ShardingSpec:
        """
        Returns the ShardingSpec for the tensor.
        """
        return self._sharding_spec

    @deprecated(DEPRECATE_MSG, category=FutureWarning)
    def reshard(self, resharding_spec: shard_spec.ShardingSpec) -> ShardedTensor:
        """
        Reshard a sharded tensor given the ``resharding_spec``. For now, we only support
        single local shard.

        If ``resharding_spec`` is same as the original one, this becomes a no-op.
        If only ``resharding_spec`` shares the same sharding dim with the original one,
        we swap local shards directly.
        For more generic cases, we merge different shards across different ranks and split
        the local shards based on the ``resharding_spec`` via `all_to_all` collective API.

        Args:
            resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
                specification describing how the tensor is sharded.

        Returns:
            A :class:`ShardedTensor` object whose local shards are resharded.

        Examples:
            >>> # xdoctest: +SKIP
            >>> # We have 2 process groups, 2 ranks.
            >>> tensor = torch.arange(4, dtype=torch.int64) + 1 + 2 * rank
            >>> tensor = torch.stack([tensor, tensor])
            >>> tensor
            tensor([[1, 2, 3, 4], [1, 2, 3, 4]]) # Rank 0
            tensor([[3, 4, 5, 6], [3, 4, 5, 6]]) # Rank 1
            tensor([[5, 6, 7, 8], [5, 6, 7, 8]]) # Rank 2
            tensor([[7, 8, 9, 10], [7, 8, 9, 10]]) # Rank 3
            >>> sharding_dim = 0
            >>> spec = ChunkShardingSpec(
                    dim=sharding_dim,
                    placements=[
                        "rank:0/cuda:0",
                        "rank:1/cuda:1",
                        "rank:2/cuda:2",
                        "rank:3/cuda:3",
                    ],
                )
            >>> current_offsets = [0] * 2
            >>> current_offsets[0] = rank * 2
            >>> shard_metadata = ShardMetadata(
                    shard_offsets=copy.deepcopy(current_offsets),
                    shard_sizes=tensor.size(),
                    placement=spec.placements[rank],
                )
            >>> local_shards = [
                    Shard(
                        tensor=tensor,
                        metadata=shard_metadata,
                    )
                ]
            >>> st = ShardedTensor._init_from_local_shards(local_shards, tensor.size())
            >>> sharding_dim = 1
            >>> resharding_spec = ChunkShardingSpec(
                    dim=sharding_dim,
                    placements=[
                        "rank:0/cuda:0",
                        "rank:1/cuda:1",
                        "rank:2/cuda:2",
                        "rank:3/cuda:3",
                    ],
                )
            >>> st.reshard(resharding_spec)
            >>> tensor = st.local_shards()[0].tensor
            >>> tensor
            tensor([[1], [1], [3], [3], [5], [5], [7], [7]]) # Rank 0
            tensor([[2], [2], [4], [4], [6], [6], [8], [8]]) # Rank 1
            tensor([[3], [3], [5], [5], [7], [7], [9], [9]]) # Rank 2
            tensor([[4], [4], [6], [6], [8], [8], [10], [10]]) # Rank 3
        """
        if not isinstance(
            resharding_spec, shard_spec.ChunkShardingSpec
        ) or not isinstance(self._sharding_spec, shard_spec.ChunkShardingSpec):
            raise NotImplementedError("Only ChunkShardingSpec supported for reshard.")

        num_local_shards = len(self.local_shards())
        if num_local_shards != 1:
            raise NotImplementedError(
                f"Only single local shard supported for reshard. Number of shards: {num_local_shards}"
            )

        if self._sharding_spec.dim == resharding_spec.dim:  # type: ignore[attr-defined]
            if self._sharding_spec.placements == resharding_spec.placements:  # type: ignore[attr-defined]
                return self
            else:
                local_shards, shards_metadata = reshuffle_local_shard(
                    self.local_tensor(),
                    self.size(),  # type: ignore[arg-type]
                    self._sharding_spec,
                    resharding_spec,
                    self._process_group,
                )
        else:
            local_shards, shards_metadata = reshard_local_shard(
                self.local_tensor(),
                self.size(),  # type: ignore[arg-type]
                self._sharding_spec,
                resharding_spec,
                self._process_group,
            )
        self._local_shards = local_shards
        self._metadata.shards_metadata = shards_metadata
        self._sharding_spec = resharding_spec
        return self

    def local_tensor(self) -> torch.Tensor:
        """
        Return local tensor for a sharded_tensor. For now we only support single local shard.

        Returns:
            A :class:`torch.Tensor` of the local shard.
        """
        num_local_shards = len(self.local_shards())
        if num_local_shards != 1:
            raise NotImplementedError(
                f"Only single local shard is supported. Number of shards: {num_local_shards}"
            )
        return self.local_shards()[0].tensor

    @classmethod
    @deprecated(DEPRECATE_MSG, category=FutureWarning)
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        def dispatch(st: ShardedTensor, func: Callable):
            # Dispatch to custom user provided op first if it exists.
            if func in _CUSTOM_SHARDED_OPS:
                return _CUSTOM_SHARDED_OPS[func](types, args, kwargs, st._process_group)

            # Dispatch to custom sharding spec op if it has one.
            if _has_custom_op(st._sharding_spec, func):
                return _dispatch_custom_op(
                    st._sharding_spec, func, types, args, kwargs, st._process_group
                )

            if func in _SHARDED_OPS:
                return _SHARDED_OPS[func](types, args, kwargs, st._process_group)

            raise RuntimeError(
                f"torch function '{func.__name__}', with args: {args} and "
                f"kwargs: {kwargs} not supported for ShardedTensor!"
            )

        # Find ShardedTensor instance to get process_group and sharding_spec.
        st_instance = None

        def find_sharded_tensor(e):
            nonlocal st_instance
            if st_instance is None and isinstance(e, ShardedTensor):
                st_instance = e

        pytree.tree_map_(find_sharded_tensor, args)
        pytree.tree_map_(find_sharded_tensor, kwargs)

        if st_instance is not None:
            return dispatch(st_instance, func)

        raise RuntimeError(
            f"torch function '{func.__name__}', with args: {args} and "
            f"kwargs: {kwargs} not supported for ShardedTensor!"
        )

    def is_pinned(self) -> bool:  # type: ignore[override]
        """
        Returns True if the sharded tensor (each local shard) resides in pinned memory.
        """
        return self._metadata.tensor_properties.pin_memory

    def _register_remote_shards(
        self, remote_shards: list[rpc.RRef[Shard]], rpc_rank: int
    ):
        self._remote_shards[rpc_rank] = remote_shards

    def remote_shards(self) -> dict[int, list[rpc.RRef[Shard]]]:
        """
        Returns a Dict[int, RRef] with keys being the RPC rank and values
        being RRefs to shards on that rank. Need to initialize the
        RPC framework for this functionality.

        Raises an exception if ShardedTensor was created with ``init_rrefs=False``
        """
        if not self._init_rrefs:
            raise RuntimeError(
                "ShardedTensor created with init_rrefs=False, no RRefs to remote shards available"
            )
        return self._remote_shards

    def __hash__(self):
        return id(self)

    def __repr__(self) -> str:  # type: ignore[override]
        return f"ShardedTensor({self._metadata})"

    @dataclass
    class ProcessGroupState:
        """
        State for ser-de of process group
        """

        local_rank: int
        global_rank: int
        local_world_size: int
        global_world_size: int

    def __getstate__(self):
        pg_state = ShardedTensor.ProcessGroupState(
            distributed_c10d.get_rank(self._process_group),
            distributed_c10d.get_rank(),
            distributed_c10d.get_world_size(self._process_group),
            distributed_c10d.get_world_size(),
        )

        return (
            self._local_shards,
            self._metadata,
            pg_state,
            self._sharding_spec,
            self._init_rrefs,
        )

    def __setstate__(self, state):
        self._sharded_tensor_id = None
        if not distributed_c10d.is_initialized():
            raise RuntimeError(
                "Need to initialize default process group using "
                '"init_process_group" before loading ShardedTensor'
            )

        (
            self._local_shards,
            self._metadata,
            pg_state,
            self._sharding_spec,
            self._init_rrefs,
        ) = state

        # Setup process group
        from torch.distributed._shard.api import _get_current_process_group

        self._process_group = _get_current_process_group()

        # Validate process group.
        local_rank = distributed_c10d.get_rank(self._process_group)
        if pg_state.local_rank != local_rank:
            raise RuntimeError(
                f"Local rank at save time was {pg_state.local_rank}, but at "
                f"load time was {local_rank}"
            )

        global_rank = distributed_c10d.get_rank()
        if pg_state.global_rank != global_rank:
            raise RuntimeError(
                f"Global rank at save time was {pg_state.global_rank}, but at "
                f"load time was {global_rank}"
            )

        local_world_size = distributed_c10d.get_world_size(self._process_group)
        if pg_state.local_world_size != local_world_size:
            raise RuntimeError(
                f"Local world size at save time was {pg_state.local_world_size}, "
                f"but at load time was {local_world_size}"
            )

        global_world_size = distributed_c10d.get_world_size()
        if pg_state.global_world_size != global_world_size:
            raise RuntimeError(
                f"Global world size at save time was {pg_state.global_world_size}, "
                f"but at load time was {global_world_size}"
            )

        self._post_init()


def _create_tensor_from_params(
    *size, local_device, tensor_properties: TensorProperties
):
    """Helper to construct tensor from size, device and common params."""
    dtype = tensor_properties.dtype
    layout = tensor_properties.layout
    requires_grad = tensor_properties.requires_grad
    memory_format = tensor_properties.memory_format
    pin_memory = tensor_properties.pin_memory

    return torch.empty(
        *size,
        dtype=dtype,
        layout=layout,
        device=local_device,
        requires_grad=requires_grad,
        memory_format=memory_format,
        pin_memory=pin_memory,
    )
<|endoftext|># mypy: allow-untyped-defs
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch.distributed._shard.metadata import ShardMetadata


class MEM_FORMAT_ENCODING(Enum):
    TORCH_CONTIGUOUS_FORMAT = 0
    TORCH_CHANNELS_LAST = 1
    TORCH_PRESERVE_FORMAT = 2


@dataclass
class TensorProperties:
    """Properties used to create :class:`Tensor`"""

    # Regular tensor fields
    dtype: torch.dtype = field(default=torch.get_default_dtype())
    layout: torch.layout = field(default=torch.strided)
    requires_grad: bool = False
    memory_format: torch.memory_format = field(default=torch.contiguous_format)
    pin_memory: bool = False

    def __getstate__(self):
        # Since torch.memory_format cannot be pickled!
        memory_format = self.memory_format
        if memory_format == torch.contiguous_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT
        elif memory_format == torch.channels_last:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST
        elif memory_format == torch.preserve_format:
            mem_format_encoding = MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT
        else:
            raise RuntimeError(f"Invalid torch.memory_format: {memory_format}")

        return (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        )

    def __setstate__(
        self,
        state,
    ):
        (
            self.dtype,
            self.layout,
            self.requires_grad,
            mem_format_encoding,
            self.pin_memory,
        ) = state

        if mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CONTIGUOUS_FORMAT:
            memory_format = torch.contiguous_format
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_CHANNELS_LAST:
            memory_format = torch.channels_last
        elif mem_format_encoding == MEM_FORMAT_ENCODING.TORCH_PRESERVE_FORMAT:
            memory_format = torch.preserve_format
        else:
            raise RuntimeError(
                f"Invalid torch.memory_format encoding: {mem_format_encoding}"
            )

        self.memory_format = memory_format

    @staticmethod
    def create_from_tensor(tensor: torch.Tensor) -> "TensorProperties":
        return TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        )


@dataclass
class ShardedTensorMetadata:
    """
    Represents metadata for :class:`ShardedTensor`
    """

    # Metadata about each shard of the Tensor
    shards_metadata: list[ShardMetadata] = field(default_factory=list)

    # Size of each dim of the overall Tensor.
    size: torch.Size = field(default=torch.Size([]))

    tensor_properties: TensorProperties = field(default_factory=TensorProperties)
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.distributed._shard.sharded_tensor as sharded_tensor
from torch.distributed._shard.sharded_tensor import _sharded_op_impl


def validate_param(param, param_name):
    if param is None:
        raise ValueError(f"param: {param_name} shouldn't be None!")


@_sharded_op_impl(torch.nn.init.uniform_)
def uniform_(types, args=(), kwargs=None, pg=None):
    r"""
    Fills the Tensor in tensor.local_shards with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.
    Args:
        tensor: tensor sharded across devices
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    """
    validate_param(kwargs, "kwargs")
    # pyrefly: ignore [unsupported-operation]
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "tensor")
    # pyrefly: ignore [unsupported-operation]
    a = kwargs["a"]
    validate_param(a, "a")
    # pyrefly: ignore [unsupported-operation]
    b = kwargs["b"]
    validate_param(b, "b")

    for shard in sharded_tensor.local_shards():
        torch.nn.init.uniform_(shard.tensor, a=a, b=b)
    return sharded_tensor


@_sharded_op_impl(torch.nn.init.normal_)
def normal_(types, args=(), kwargs=None, pg=None):
    r"""
    Fills the Tensors in tensor.local_shards with values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.
    Args:
        tensor: tensor sharded across devices
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    """
    validate_param(kwargs, "kwargs")
    # pyrefly: ignore [unsupported-operation]
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "tensor")
    # pyrefly: ignore [unsupported-operation]
    mean = kwargs["mean"]
    validate_param(mean, "mean")
    # pyrefly: ignore [unsupported-operation]
    std = kwargs["std"]
    validate_param(std, "std")

    for shard in sharded_tensor.local_shards():
        torch.nn.init.normal_(shard.tensor, mean=mean, std=std)
    return sharded_tensor


@_sharded_op_impl(torch.nn.init.kaiming_uniform_)
def kaiming_uniform_(types, args=(), kwargs=None, pg=None):
    r"""
    Fills the Tensors in tensor.local_shards with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: tensor sharded across devices
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    """
    validate_param(kwargs, "kwargs")
    # pyrefly: ignore [unsupported-operation]
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "tensor")
    # pyrefly: ignore [unsupported-operation]
    a = kwargs["a"]
    validate_param(a, "a")
    # pyrefly: ignore [unsupported-operation]
    mode = kwargs["mode"]
    validate_param(mode, "mode")
    # pyrefly: ignore [unsupported-operation]
    nonlinearity = kwargs["nonlinearity"]
    validate_param(nonlinearity, "nonlinearity")

    for shard in sharded_tensor.local_shards():
        torch.nn.init.kaiming_uniform_(
            shard.tensor, a=a, mode=mode, nonlinearity=nonlinearity
        )
    return sharded_tensor


@_sharded_op_impl(torch.nn.init.constant_)
def constant_(types, args=(), kwargs=None, pg=None):
    r"""
    Fills the input ShardedTensor with the value \text{val}val.
    Args:
        tensor: tensor sharded across devices
        val: the value to fill the tensor with
    """
    validate_param(kwargs, "kwargs")
    # pyrefly: ignore [unsupported-operation]
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "tensor")
    # pyrefly: ignore [unsupported-operation]
    val = kwargs["val"]
    validate_param(val, "val")
    for shard in sharded_tensor.local_shards():
        torch.nn.init.constant_(shard.tensor, val=val)
    return sharded_tensor


tensor_like_creation_op_map = {
    torch.full_like: sharded_tensor.full,
    torch.empty_like: sharded_tensor.empty,
    torch.zeros_like: sharded_tensor.zeros,
    torch.ones_like: sharded_tensor.ones,
    torch.rand_like: sharded_tensor.rand,
    torch.randn_like: sharded_tensor.randn,
}


# tensor ops that behave the same as the default tensor
def register_tensor_creation_op(op):
    @_sharded_op_impl(op)
    def tensor_creation_op(types, args=(), kwargs=None, pg=None):
        """
        Handles ``__torch_function__`` dispatch for tensor creation ops that
        takes a ShardedTensor as argument, such as ``torch.zeros_like`` or
        ``torch.full_like``.
        """
        creation_op = tensor_like_creation_op_map.get(op)
        if creation_op is None:
            raise RuntimeError(f"Tensor creation {op} not supported!")
        if kwargs is None:
            kwargs = {}

        # pyrefly: ignore [bad-index]
        st = args[0]

        new_st = creation_op(st.sharding_spec(), st.size(), *args[1:], **kwargs)  # type: ignore[operator]
        return new_st


register_tensor_creation_op(torch.full_like)
register_tensor_creation_op(torch.empty_like)
register_tensor_creation_op(torch.zeros_like)
register_tensor_creation_op(torch.ones_like)
register_tensor_creation_op(torch.rand_like)
register_tensor_creation_op(torch.randn_like)
<|endoftext|>import torch.distributed._shard.sharded_tensor._ops.misc_ops
import torch.distributed._shard.sharded_tensor._ops.tensor_ops

# Import all ChunkShardingSpec ops
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding import (
    sharded_embedding,
)
from torch.distributed._shard.sharding_spec.chunk_sharding_spec_ops.embedding_bag import (
    sharded_embedding_bag,
)

from .binary_cmp import allclose, equal
from .init import constant_, kaiming_uniform_, normal_, uniform_
<|endoftext|># mypy: allow-untyped-defs
import copy

import torch
from torch.distributed._shard.common_op_utils import _register_default_op
from torch.distributed._shard.sharded_tensor import (
    _sharded_op_impl,
    Shard,
    ShardedTensor,
)

from ._common import _register_sharded_op_on_local_shards


# Tensor properties access
_register_default_op(torch.Tensor.shape.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.dtype.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.layout.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.size, _sharded_op_impl)
_register_default_op(torch.Tensor.dim, _sharded_op_impl)
_register_default_op(torch.Tensor.ndim.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
_register_default_op(torch.Tensor.is_contiguous, _sharded_op_impl)
_register_default_op(torch.Tensor.contiguous, _sharded_op_impl)
_register_default_op(torch.Tensor.is_floating_point, _sharded_op_impl)

# __reduce_ex__ to dispatch to get_state/set_state
_register_default_op(torch.Tensor.__reduce_ex__, _sharded_op_impl)

# autograd related properties
_register_default_op(torch.Tensor.requires_grad.__get__, _sharded_op_impl)  # type: ignore[attr-defined]
# TODO: set grad with a ShardedTensor that consists of all local grads
_register_default_op(torch.Tensor.grad.__get__, _sharded_op_impl)  # type: ignore[union-attr]
_register_default_op(torch.Tensor.grad_fn.__get__, _sharded_op_impl)  # type: ignore[union-attr]
_register_default_op(torch.Tensor.is_leaf.__get__, _sharded_op_impl)  # type: ignore[attr-defined]


# device property is ambiguous as from a global perspective,
# ShardedTensor.device consists of multiple devices (might even across hosts)
# We choose to return the current device of the local tensor to represent
# the device property on each rank
@_sharded_op_impl(torch.Tensor.device.__get__)
def tensor_device(types, args=(), kwargs=None, pg=None):
    # pyrefly: ignore [bad-index]
    self_st = args[0]
    # Validate types
    if not isinstance(self_st, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")
    dev: torch.device
    if self_st._local_shards:
        dev = self_st._local_shards[0].tensor.device
    elif pg and pg._get_backend_name() == "gloo":
        dev = torch.device("cpu")
    else:
        dev = torch.device(torch.cuda.current_device())
    return dev


@_sharded_op_impl(torch.Tensor.is_meta.__get__)  # type: ignore[attr-defined]
def st_is_meta(types, args=(), kwargs=None, pg=None):
    # pyrefly: ignore [bad-index]
    return args[0].local_tensor().is_meta


def sharded_type_as_check(*args, **kwargs):
    """
    Perform extra checks for the sharded_type_as op such as the input needs to
    be either a Tensor or ShardedTensor.

    Args: same as ``torch.Tensor.type_as``.

    Return: None
    """
    if len(args) < 2:
        raise ValueError("Needs to give a tensor to cast type as!")
    if not isinstance(args[1], torch.Tensor) and not isinstance(args[1], ShardedTensor):
        raise ValueError("Needs to give a Tensor or ShardedTensor to cast type as!")


def same_dtype(*args, **kwargs):
    """
    When the dtype is the same, return the original ShardedTensor.

    Args: same as ``torch.Tensor.type_as``.

    Return (bool): Whether to return early or not.
    """
    return args[0].dtype == args[1].dtype


def sharded_type_as(args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for the ``torch.Tensor.type_as`` op.

    Args: same as ``torch.Tensor.type_as``.

    Return:
        new_local_shards (List[Shard]): Local shards for the new sharded tensor.
        st_meta (ShardedTensorMetadata): Metadata of the new sharded tensor.
    """
    st = args[0]
    tensor = args[1]
    if isinstance(tensor, ShardedTensor):
        tensor = tensor.local_tensor()
    new_local_shards = [
        Shard(shard.tensor.type_as(tensor), shard.metadata)
        for shard in st.local_shards()
    ]
    st_meta = copy.deepcopy(st._metadata)
    st_meta.tensor_properties.dtype = tensor.dtype
    return new_local_shards, st_meta


_register_sharded_op_on_local_shards(
    torch.Tensor.type_as,
    early_stop_func=same_dtype,
    extra_check=sharded_type_as_check,
    customized_func=sharded_type_as,
)


def sharded_deepcopy(args, kwargs, pg):
    # NOTE: we directly implement deepcopy magic method
    # instead of using the default tensor.__deepcopy__
    # and implement clone(). This is because the default
    # tensor deepcopy copies every attribute, but the
    # process_group in ShardedTensor cannot be deep copied.
    self_st = args[0]
    new_local_shards = copy.deepcopy(self_st.local_shards())
    new_metadata = copy.deepcopy(self_st.metadata())
    return new_local_shards, new_metadata


_register_sharded_op_on_local_shards(
    torch.Tensor.__deepcopy__,
    customized_func=sharded_deepcopy,
)


@_sharded_op_impl(torch.Tensor.copy_)
def sharded_inplace_copy(types, args, kwargs, pg):
    # NOTE: inplace op don't need to rewrap
    kwargs = {} if kwargs is None else kwargs
    self_st = args[0]
    new_st = args[1]
    nonblocking = kwargs.get("non_blocking", False)
    for local_shard, new_shard in zip(self_st.local_shards(), new_st.local_shards()):
        if local_shard.metadata != new_shard.metadata:
            raise RuntimeError(
                "inplace copy can only happen between two ShardedTensor with same metadata!"
            )
    for local_shard, new_shard in zip(self_st.local_shards(), new_st.local_shards()):
        local_shard.tensor.copy_(new_shard.tensor, nonblocking)

    return self_st


def sharded_clone(args, kwargs, pg):
    self_st = args[0]
    desire_memory_format = kwargs.get("memory_format", None)
    if desire_memory_format and desire_memory_format != torch.preserve_format:
        raise RuntimeError("Only support torch.preserve_format for ShardedTensor!")
    cloned_local_shards = [
        Shard(
            local_shard.tensor.clone(memory_format=desire_memory_format),
            metadata=copy.deepcopy(local_shard.metadata),
        )
        for local_shard in self_st.local_shards()
    ]
    new_metadata = copy.deepcopy(self_st.metadata())
    return cloned_local_shards, new_metadata


_register_sharded_op_on_local_shards(
    torch.Tensor.clone,
    customized_func=sharded_clone,
)


def sharded_detach(args, kwargs, pg):
    self_st = args[0]
    detached_local_shards = [
        Shard(
            local_shard.tensor.detach(),
            metadata=copy.deepcopy(local_shard.metadata),
        )
        for local_shard in self_st.local_shards()
    ]
    new_metadata = copy.deepcopy(self_st.metadata())
    new_metadata.tensor_properties.requires_grad = False
    return detached_local_shards, new_metadata


_register_sharded_op_on_local_shards(
    torch.Tensor.detach,
    customized_func=sharded_detach,
)


@_sharded_op_impl(torch.Tensor.requires_grad_)
def tensor_requires_grad_set(types, args=(), kwargs=None, pg=None):
    # pyrefly: ignore [bad-index]
    self_st = args[0]
    # Validate types
    if not isinstance(self_st, ShardedTensor):
        raise TypeError("input needs to be a ShardedTensor")

    if kwargs is None:
        kwargs = {}

    requires_grad = args[1] if len(args) > 1 else kwargs.get("requires_grad", True)
    if requires_grad == self_st.requires_grad:
        return self_st

    for local_shard in self_st.local_shards():
        local_shard.tensor.requires_grad_(requires_grad)

        # update the wrapper class property
    with torch._C.DisableTorchFunctionSubclass():
        self_st.requires_grad_(requires_grad)
    # update the metadata in the meanwhile
    self_st._metadata.tensor_properties.requires_grad = requires_grad
    return self_st
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as distributed_c10d
from torch.distributed._shard.sharded_tensor import _sharded_op_impl, ShardedTensor


def _communicate_result(result, pg):
    # Gather results from all ranks.
    if result:
        result_tensor = torch.ones(
            1, device=torch.device(torch.accelerator.current_device_index())
        )
    else:
        result_tensor = torch.zeros(
            1, device=torch.device(torch.accelerator.current_device_index())
        )

    dist.all_reduce(result_tensor, group=pg)

    expected_result = torch.ones(
        1, device=torch.device(torch.accelerator.current_device_index())
    ) * dist.get_world_size(pg)

    return torch.equal(result_tensor, expected_result)


def binary_cmp(cmp_fun, types, args, kwargs=None, process_group=None):
    if len(args) != 2:
        raise ValueError(f"Expected two arguments for torch.{cmp_fun.__name__}")

    st1 = args[0]
    st2 = args[1]
    if not (isinstance(st1, ShardedTensor) and isinstance(st2, ShardedTensor)):
        raise TypeError(
            f"Both arguments to torch.{cmp_fun.__name__} need to be of type ShardedTensor"
        )

    # Verify same PG
    if st1._process_group != st2._process_group:
        return False

    if distributed_c10d._rank_not_in_group(
        st1._process_group
    ) or distributed_c10d._rank_not_in_group(st2._process_group):
        return distributed_c10d._rank_not_in_group(
            st1._process_group
        ) == distributed_c10d._rank_not_in_group(st2._process_group)

    # Verify metadata
    if st1.metadata() != st2.metadata():
        return _communicate_result(False, st1._process_group)

    # Verify number of local shards
    st1_local_shards = st1.local_shards()
    st2_local_shards = st2.local_shards()
    if len(st1_local_shards) != len(st2_local_shards):
        return _communicate_result(False, st1._process_group)

    # kwargs must be dict-like
    if kwargs is None:
        kwargs = {}
    # Verify each local shard
    for idx in range(len(st1_local_shards)):
        if st1_local_shards[idx].metadata != st2_local_shards[idx].metadata:
            return _communicate_result(False, st1._process_group)
        if not cmp_fun(
            st1_local_shards[idx].tensor, st2_local_shards[idx].tensor, **kwargs
        ):
            return _communicate_result(False, st1._process_group)

    return _communicate_result(True, st1._process_group)


@_sharded_op_impl(torch.equal)
def equal(types, args, kwargs, process_group):
    return binary_cmp(torch.equal, types, args, kwargs, process_group)


@_sharded_op_impl(torch.allclose)
def allclose(types, args, kwargs, process_group):
    return binary_cmp(torch.allclose, types, args, kwargs, process_group)
<|endoftext|># mypy: allow-untyped-defs
import torch
from torch.distributed._shard.sharded_tensor import _sharded_op_impl


# This is used by `_apply()` within module.py to set new
# parameters after apply a certain method, we should follow
# the future behavior of overwriting the existing tensor
# instead of doing in-place change using `.data = `.
@_sharded_op_impl(torch._has_compatible_shallow_copy_type)
def tensor_has_compatible_shallow_copy_type(types, args=(), kwargs=None, pg=None):
    return False
<|endoftext|># mypy: allow-untyped-defs
import functools

from torch.distributed._shard.common_op_utils import _basic_validation
from torch.distributed._shard.sharded_tensor import (
    _sharded_op_impl,
    Shard,
    ShardedTensor,
)


def _sharded_op_common(op, early_stop_func, extra_check):
    """
    Inject sharded tensor op registration with common logics executed before
    different behaviors are done on either local shards or a local tensor.

    Example::
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> op = torch.transpose
        >>> @_sharded_op_impl(op)
        >>> @_sharded_op_common(op, early_stop_func, extra_check)
        >>> def sharded_tensor_op(types, args, kwargs, process_group):
        >>>   ...
        >>>
        >>> st = sharded_tensor.rand(32, 16)
        >>> st.transpose(1, 2)
        >>> # This will call '_sharded_op_common'

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.

    Return:
        func (Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.transpose)
    """

    def decorator_sharded_func(wrapped_func):
        @functools.wraps(wrapped_func)
        def wrapper(types, args=(), kwargs=None, pg=None):
            _basic_validation(op, args, kwargs)

            # pyrefly: ignore [bad-index]
            st = args[0]
            if kwargs is None:
                kwargs = {}
            if extra_check:
                extra_check(*args, **kwargs)
            if early_stop_func:
                early_stop = early_stop_func(*args, **kwargs)
                if early_stop:
                    return st
            return wrapped_func(types, args, kwargs, pg)

        return wrapper

    return decorator_sharded_func


def _register_sharded_op_on_local_shards(
    op, early_stop_func=None, extra_check=None, customized_func=None
):
    """
    Handles ``__torch_function__`` dispatch for ops which are performed on
    each shard of the sharded tensor such as elementwise op like
    ``torch.nn.functional.gelu`` or ``torch.nn.functional.relu``.

    For more complicated ops, a customized func can be used to generate
    the new shards and sharded tensor size.

    This function expects that the original ShardingSpec for the ShardedTensor
    is preserved irrespective of whether or not a customized function is used.

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.
        customized_func (Callable, optional): the func for customized logic
            to generate new shards and sharded tensor size.
            Default: if ``None``, we simply lower to the real op call with
                all local shards of the st.

    Return:
        func (Callable): registered implementation for sharded op for
        ``__torch_function__`` dispatch.
    """

    @_sharded_op_impl(op)
    @_sharded_op_common(op, early_stop_func, extra_check)
    def sharded_tensor_op_on_local_shards(types, args=(), kwargs=None, pg=None):
        # pyrefly: ignore [bad-index]
        st = args[0]
        st_metadata = st.metadata()
        local_shards = st.local_shards()
        local_shards_new = []
        if customized_func:
            local_shards_new, st_metadata = customized_func(args, kwargs, pg)
        else:
            for local_shard in local_shards:
                args = (local_shard.tensor, *args[1:])
                local_shards_new.append(
                    Shard(op(*args, **kwargs), local_shard.metadata)
                )
        return ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards_new,
            st_metadata,
            process_group=pg,
            init_rrefs=st._init_rrefs,
            sharding_spec=st.sharding_spec(),
        )
<|endoftext|>import abc
from dataclasses import dataclass

import torch.nn as nn
from torch.distributed._shard.sharder import Sharder
from torch.distributed._shard.sharding_spec import ShardingSpec


@dataclass
class ShardingPlan:
    """
    Representation of a sharding plan, describes how to shard a module
    across hosts. `plan` is used to shard module parameters according to the spec provided,
    `output_plan` and `return_local_tensor` are optional, they are used to specify the output
    layout of a module with a spec, and when to convert back to data parallel fashion.

    Args:
        plan (Dict[str, Union[:class:`torch.distributed._shard.sharding_spec.ShardingSpec`,
              :class:`torch.distributed._shard.sharder.Sharder`]):
            a dict describes how to shard a module, there're currently two ways to shard a module:
                1. directly shard a module parameter by a `ShardingSpec`, keyed by the name of
                   a parameter to a `ShardingSpec`.
                2. shard a submodule by applying a `Sharder` on it, keyed by the name of a module
                   to a `Sharder` object.
        output_plan (Dict[str, :class:`torch.distributed._shard.sharding_spec.ShardingSpec`), optional):
            a dict specifies the layout of a module's output which produces a ShardedTensor,
            keyed by the name of module to ShardingSpec("" in key means the root module).
            Default: `None`
        return_local_tensor (List[str], optional): a list of string, each element enables
            a module's sharded output to be returned as a Tensor from its local shards to
            ensure further processing in a data parallel fashion. ("" in list means the
            root module).
            Default: None
    Example:
      Suppose we want to shard a module with two linear layers and then run it with DDP, we also
      want to convert the output of the second linear layer back to DDP, we can do it as follows:

        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> class MyModule(nn.Module):
        >>>     def __init__(self) -> None:
        >>>        super().__init__()
        >>>        self.fc1 = nn.Linear()
        >>>        self.gelu = nn.GELU()
        >>>        self.fc2 = nn.Linear()
        >>>        self.relu = nn.Linear()
        >>>
        >>>     def forward(self, input):
        >>>         return self.relu(self.fc2(self.gelu(self.fc1(input))))


        >>> # xdoctest: +SKIP("Undefined spec1, spec2)
        >>> sharding_plan = ShardingPlan(
        >>>    plan={
        >>>        "fc1.weight": spec1,
        >>>        "fc2.weight": spec2
        >>>    },
        >>>    output_plan={
        >>>        "fc2": output_spec
        >>>    },
        >>>    return_local_tensor=["fc2"]
        >>> )
    """

    plan: dict[str, ShardingSpec | Sharder]
    output_plan: dict[str, ShardingSpec] | None = None
    return_local_tensor: list[str] | None = None


class ShardingPlanner(abc.ABC):
    """
    Default ShardingPlanner interface, can be extended and
    implement advanced sharding strategies.
    """

    @abc.abstractmethod
    def build_plan(self, module: nn.Module) -> ShardingPlan:
        """
        Given an nn.Module, define how to shard the module across
        ranks, return a ShardingPlan
        Args:
            module (:class:`torch.nn.Module`):
                The module to apply sharding to.
        Returns:
            A :class:`torch.distributed._shard.sharding_plan.ShardingPlan` object that
            represents how to shard the module.
        """
<|endoftext|>from collections.abc import Iterator
from typing import Union

import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor

from .api import ShardedOptimizer


def named_params_with_sharded_tensor(
    module: nn.Module,
    prefix: str = "",
    recurse: bool = True,
) -> Iterator[tuple[str, nn.Parameter | ShardedTensor]]:
    r"""Returns an iterator over module parameters (together with the
    ShardedTensor parameters), yielding both the name of the parameter
    as well as the parameter itself. This is typically passed to a
    :class:torch.distributed._shard.sharded_optim.ShardedOptimizer

    Args:
        prefix (str): prefix to prepend to all parameter names.
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.

    Yields:
        (str, Union[Tensor, ShardedTensor]): Tuple containing
            the name and parameter (or ShardedTensor parameter)

    Example::

        >>> # xdoctest: +SKIP
        >>> model = torch.nn.Linear(*linear_size)
        >>> shard_parameter(model, "weight", spec)
        >>> for name, param in named_params_with_sharded_tensor(model):
        >>>    if name in ['weight']:
        >>>        print(param.size())

    """
    modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]

    memo = set()
    for mod_prefix, mod in modules:
        # find all sharded tensor params
        for name, val in vars(mod).items():
            if isinstance(val, ShardedTensor) and val not in memo:
                memo.add(val)
                name = mod_prefix + ("." if mod_prefix else "") + name
                yield name, val

    # find all nn.Parameters
    for name, val in module.named_parameters():
        yield name, val
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Mapping
from typing import Any

import torch.optim as optim
from torch import Tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor


class ShardedOptimizer(optim.Optimizer):
    def __init__(
        self,
        named_params: Mapping[str, Tensor | ShardedTensor],
        optimizer_class,
        *optimizer_args,
        **optimizer_kwargs,
    ):
        """
        ShardedOptimizer collects all tensors and local shard tensors of
        ShardedTensor, then use these tensors as ``params`` for optimizers

        Args:
            named_params (Dict[str, Union[Tensor, ShardedTensor]]) : a Dict
                of parameters, where key is the parameter key, value is either
                Tensor or ShardedTensor parameter.
            optimizer_class (torch.optim.Optimizer): the Optimizer to use
                locally, i.e. torch.optim.SGD, torch.optim.Adagrad, etc.
            *optimizer_args: the arguments to initialize the optimizer.
            **optimizer_kwargs: the key-word arguments to initialize the optimizer.

        """
        tensors: list[Tensor] = []
        for value in named_params.values():
            if isinstance(value, ShardedTensor):
                tensors.extend(
                    local_shard.tensor for local_shard in value.local_shards()
                )
            else:
                tensors.append(value)

        self.named_params = named_params
        self._optim = optimizer_class(tensors, *optimizer_args, **optimizer_kwargs)
        self.param_groups = self._optim.param_groups
        self.state = self._optim.state

    def zero_grad(self, set_to_none: bool = True):  # type: ignore[override]
        r"""Resets the gradients of all optimized :class:`torch.Tensor` s.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        self._optim.zero_grad(set_to_none)

    def step(self, closure=None):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        self._optim.step(closure)

    def state_dict(self) -> dict[str, Any]:
        """
        Returned state and param_groups will contain parameter keys
        instead of parameter indices like torch.optim.Optimizer.
        This allows for advanced functionality like optimizer re-sharding to be implemented.
        """
        # TODO: implement state_dict
        raise NotImplementedError("ShardedOptimizer state_dict not implemented yet!")

    def load_state_dict(self, state_dict: Mapping[str, Any]):
        r"""Loads the ShardedOptimizer state.

        Args:
            state_dict (dict): ShardedOptimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # TODO: implement load_state_dict
        raise NotImplementedError(
            "ShardedOptimizer load_state_dict not implemented yet!"
        )

    def add_param_group(self, param_group: Any):
        r"""Add a new param group"""
        # TODO: implement add_param_group
        raise NotImplementedError(
            "ShardedOptimizer add_param_group not implemented yet!"
        )
<|endoftext|>from torch.distributed._shard.metadata import ShardMetadata

from .api import (
    _infer_sharding_spec_from_shards_metadata,
    DevicePlacementSpec,
    EnumerableShardingSpec,
    PlacementSpec,
    ShardingSpec,
)
from .chunk_sharding_spec import ChunkShardingSpec as ChunkShardingSpec
<|endoftext|># mypy: allow-untyped-defs
import functools
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.op_registry_utils import _decorator_func

from ._internals import (
    check_tensor,
    get_chunked_dim_size,
    get_split_size,
    validate_non_overlapping_shards_metadata,
)


if TYPE_CHECKING:
    # Only include ShardedTensor when do type checking, exclude it
    # from run-time to resolve circular dependency.
    from torch.distributed._shard.sharded_tensor import ShardedTensor


class PlacementSpec(ABC):  # noqa: B024
    """
    Base class representing the placement of an entity. Subclasses of this
    class can be used to specify customized placements which might not be
    covered by existing APIs.
    """


@dataclass
class DevicePlacementSpec(PlacementSpec):
    """
    Associates placement of an entity with a single device.

    Args:
        device(:class:`torch.distributed._remote_device`): The device to place the entity on.
    """

    device: torch.distributed._remote_device

    def __post_init__(self):
        if not isinstance(self.device, torch.distributed._remote_device):
            self.device = torch.distributed._remote_device(self.device)


class ShardingSpec(ABC):
    """
    Base class representing sharding specifications.
    """

    @abstractmethod
    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: sharded_tensor_meta.TensorProperties,
    ) -> sharded_tensor_meta.ShardedTensorMetadata:
        """
        Given a global tensor size, define how to shard a tensor like this shape
        across ranks, return ShardedTensorMetadata
        Args:
            tensor_sizes (:class:`torch.Size`):
                The tensor shape to shard on, a `torch.Size` object that represents the
                tensor shape to be sharded according to the ShardingSpec.
            tensor_properties(:class:`torch.distributed._shard.sharded_tensor.TensorProperties):
                Tensor properties used to create a ShardedTensor.
        Returns:
            A :class:`ShardedTensorMetadata` object that encodes the information about
            the layout of the ShardedTensor and its properties.
        """

    @abstractmethod
    def shard(
        self, tensor: torch.Tensor, src_rank: int = 0, process_group=None
    ) -> "ShardedTensor":
        """
        Given a global tensor on src_rank, shard this tensor
        across ranks within the process group, return a ShardedTensor.
        Args:
            tensor (:class:`torch.Tensor`): Tensor needs to be sharded.
        Keyword args:
            src_rank (int, optional): The source rank which is used as the ground truth of
                the data for the parameter that would be sharded and scattered
                across the rest of the ranks.
                Default: 0.
            process_group (ProcessGroup, optional): The process group to work on. If None,
                the default process group will be used.
        Returns:
            A :class:`ShardedTensor` sharded from the given tensor.
        """


# Ops customized for a particular ShardingSpec.
_CUSTOM_SHARDING_SPEC_OPS: dict[str, dict[Callable, Callable]] = {}


def _has_custom_op(sharding_spec, op):
    """
    Returns whether or not the ShardingSpec has a custom op implementation.
    """
    class_name = type(sharding_spec).__qualname__
    return (
        class_name in _CUSTOM_SHARDING_SPEC_OPS
        and op in _CUSTOM_SHARDING_SPEC_OPS[class_name]
    )


def _dispatch_custom_op(
    sharding_spec, op: Callable, types, args, kwargs, process_group
):
    """
    Calls the custom op for this ShardingSpec if it exists.
    """
    class_name = type(sharding_spec).__qualname__
    if not _has_custom_op(sharding_spec, op):
        raise RuntimeError(f"Custom op: {op} not registered for {class_name}")
    func = _CUSTOM_SHARDING_SPEC_OPS[class_name][op]
    return func(types, args, kwargs, process_group)


def custom_sharding_spec_op(sharding_spec_class, func):
    """
    Decorator to allow custom registration of ops.
    Args:
        sharding_spec_class(type): The ShardingSpec for which we need to add this custom op.
        func(Callable): The op to override (ex: torch.bmm)
    """
    class_name = sharding_spec_class.__qualname__
    if class_name not in _CUSTOM_SHARDING_SPEC_OPS:
        _CUSTOM_SHARDING_SPEC_OPS[class_name] = {}
    return functools.partial(
        _decorator_func, op=func, op_table=_CUSTOM_SHARDING_SPEC_OPS[class_name]
    )


@dataclass
class EnumerableShardingSpec(ShardingSpec):
    """
    This is a type of PlacementSpec that allows users to specify a generic
    sharding scheme by enumerating exactly how each shard is laid out.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard. Note that none of the shards should overlap.
    """

    shards: list[ShardMetadata]

    def __post_init__(self):
        if len(self.shards) == 0:
            raise ValueError(f"Empty shard list provided: {self.shards}")

        # Validate each shard has same rank.
        rank = -1
        for shard in self.shards:
            if rank != -1 and rank != len(shard.shard_offsets):
                raise ValueError(
                    f"Found inconsistent ranks for shards: {rank} and {len(shard.shard_offsets)}"
                )
            rank = len(shard.shard_offsets)

        validate_non_overlapping_shards_metadata(self.shards)

    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: sharded_tensor_meta.TensorProperties,
    ) -> sharded_tensor_meta.ShardedTensorMetadata:
        # check if shards form a valid tensor
        check_tensor(self.shards, tensor_sizes)
        return sharded_tensor_meta.ShardedTensorMetadata(
            self.shards, tensor_sizes, tensor_properties
        )

    def shard(
        self, tensor: torch.Tensor, src_rank: int = 0, process_group=None
    ) -> "ShardedTensor":
        # TODO: figure out a generic and efficient way to scatter the shards for EnumerableShardingSpec
        raise NotImplementedError("EnumerableShardingSpec.shard not implemented yet!")


def _infer_sharding_spec_from_shards_metadata(shards_metadata):
    """
    Infer the sharding spec from the metadata of each shard of a ShardedTensor.
    If the tensor is sharded only on one dimension, we can then verify whether it's
    a ChunkShardingSpec or not. The way to verify it is to first get the total length
    and perform a chunk sharding with the given placements to see if we can have the
    same chunk size as the given shards_metadata. If not, we assume it's enum sharded.

    Args:
        shards_metadata (List[ShardMetadata]): List of Metadata of local shards.

    Returns:
        A :class:`torch.distributed._shard.sharding_spec.ShardingSpec` object of sharding
            spec for one sharded tensor.
    """
    placements = []
    chunk_sharding_dim = None
    chunk_offset_list = []
    shard_size_list = []
    shard_offset_list = []
    # collect local shard metadata from the global sharded_tensor_metadata
    for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
        placements.append(shard_metadata.placement)
        local_offsets = shard_metadata.shard_offsets
        chunk_offset_list.append(sum(local_offsets))
        shard_size_list.append(shard_metadata.shard_sizes)
        shard_offset_list.append(shard_metadata.shard_offsets)
        shard_dims = [idx for idx, e in enumerate(local_offsets) if e != 0]
        # If the offset is [0, 0, ..., 0] (all zeros),
        # we cannot decide whether how the tensor is sharded.
        if len(shard_dims) == 0:
            continue
        # If the offset is [0, N, .,0, M, 0, .., 0],
        # we are sure it's sharded by more than one dimension.
        if len(shard_dims) != 1:
            chunk_sharding_dim = None
            break
        # If the offset is [0, 0, .,0, M, 0, .., 0], aka, it's sharded by just
        # one dimension, we need to make sure all ranks share the same dimension.
        if not chunk_sharding_dim:
            chunk_sharding_dim = shard_dims[0]
        elif chunk_sharding_dim != shard_dims[0]:
            chunk_sharding_dim = None
            break

    if chunk_sharding_dim is not None:
        # Ensure we infer the correct placement order from offsets
        placements = [
            x
            for _, x in sorted(
                zip(chunk_offset_list, placements), key=operator.itemgetter(0)
            )
        ]

        from .chunk_sharding_spec import ChunkShardingSpec

        chunk_spec = ChunkShardingSpec(
            dim=chunk_sharding_dim,
            placements=placements,
        )

        shard_sizes = sorted([x[chunk_sharding_dim] for x in shard_size_list])
        shard_total_length = sum(shard_sizes)
        shard_offsets = sorted([x[chunk_sharding_dim] for x in shard_offset_list])

        chunks = len(placements)
        split_size = get_split_size(shard_total_length, chunks)
        chunk_shard_sizes = sorted(
            [
                get_chunked_dim_size(shard_total_length, split_size, idx)
                for idx in range(chunks)
            ]
        )
        # Should match ChunkShardingSpec offsets calculation
        chunk_shard_offsets = [split_size * idx for idx in range(chunks)]
        if shard_sizes == chunk_shard_sizes and shard_offsets == chunk_shard_offsets:
            return chunk_spec
    return EnumerableShardingSpec(shards_metadata)
<|endoftext|># mypy: allow-untyped-defs
import math
import sys
from bisect import bisect_right, insort

from torch.distributed._shard.metadata import ShardMetadata


def _check_shard_metadata_pair_overlap(shard1: ShardMetadata, shard2: ShardMetadata):
    """
    Checks if two shards overlap.
    """

    # For each dim of each shard, check if one shard resides on the other
    # end of second shard with respect to that dim. As an example for a 2D
    # shard, we would check if one shard is above or on the left of the
    # other shard.
    ndims = len(shard1.shard_offsets)
    for i in range(ndims):
        if shard1.shard_offsets[i] >= shard2.shard_offsets[i] + shard2.shard_sizes[i]:
            return False
        if shard2.shard_offsets[i] >= shard1.shard_offsets[i] + shard1.shard_sizes[i]:
            return False

    return True


def _find_nd_overlapping_shards(
    shards: list[ShardMetadata], sharded_dims: list[int]
) -> tuple[int, int] | None:
    """Find overlapping shards using sweep-line algorithm."""
    if len(shards) <= 1:
        return None

    dims = len(sharded_dims)
    if dims == 0:
        return None

    sweep_dim_idx = 0
    if dims > 1:
        max_size = 0
        for i, dim in enumerate(sharded_dims):
            dim_size = shards[0].shard_offsets[dim] + shards[0].shard_sizes[dim]
            if dim_size > max_size:
                max_size = dim_size
                sweep_dim_idx = i
    sweep_dim = sharded_dims[sweep_dim_idx]

    sorted_indices = sorted(
        range(len(shards)),
        key=lambda idx: (
            shards[idx].shard_offsets[sweep_dim],
            *(shards[idx].shard_offsets[d] for d in sharded_dims if d != sweep_dim),
        ),
    )
    active: list[tuple[int, int]] = []

    for idx in sorted_indices:
        current = shards[idx]
        start = current.shard_offsets[sweep_dim]
        end = start + current.shard_sizes[sweep_dim]

        cutoff = bisect_right(active, (start, sys.maxsize))
        if cutoff:
            del active[:cutoff]

        for _, other_idx in active:
            other = shards[other_idx]

            if _check_shard_metadata_pair_overlap(current, other):
                return (other_idx, idx)
        insort(active, (end, idx))
    return None


def _find_1d_overlapping_shards(
    shards: list[ShardMetadata], dim: int
) -> tuple[int, int] | None:
    # (begin, end, index_in_shards). Begin and end are inclusive.
    intervals = [
        (s.shard_offsets[dim], s.shard_offsets[dim] + s.shard_sizes[dim] - 1, i)
        for i, s in enumerate(shards)
    ]
    intervals.sort()
    for i in range(len(shards) - 1):
        if intervals[i][1] >= intervals[i + 1][0]:
            return (intervals[i][2], intervals[i + 1][2])
    return None


def validate_non_overlapping_shards_metadata(shards: list[ShardMetadata]):
    """
    Ensures none of the shards overlap with each other.

    Args:
        shards(List[ShardMetadata]): List of :class:`ShardMetadata` objects representing
            each shard.
    Raises:
        ``ValueError`` if there's overlap in any two shards.
    """
    if not shards or len(shards) == 1:
        return

    sharded_dims: list[int] = []
    for dim in range(len(shards[0].shard_offsets)):
        for i in range(1, len(shards)):
            if (
                shards[i].shard_offsets[dim] != shards[0].shard_offsets[dim]
                or shards[i].shard_sizes[dim] != shards[0].shard_sizes[dim]
            ):
                sharded_dims.append(dim)
                break

    pair: tuple[int, int] | None = None
    if len(sharded_dims) == 0:
        # if shard is all zeros, we should consider as pass
        all_zeros: bool = all(
            # strictly limited all offsets to be 0 to pass
            # could loosen it later on
            shard.shard_offsets == [0] * len(shards[0].shard_offsets)
            and math.prod(shard.shard_sizes) == 0  # one dimension is 0
            for shard in shards
        )
        if all_zeros:
            return
        # All shards are the same, all dims are not partitioned. Choose any 2.
        pair = (0, 1)
    elif len(sharded_dims) == 1:
        # Shards are partitioned over only one dimension. Overlap can be found
        # using a O(nlogn) overlapping interval algorithm.
        pair = _find_1d_overlapping_shards(shards, sharded_dims[0])
    else:
        # Shards are partitioned over more than one dimension.
        # Use sweep-line algorithm for O(n log n) complexity.
        pair = _find_nd_overlapping_shards(shards, sharded_dims)

    if pair:
        raise ValueError(f"Shards {shards[pair[0]]} and {shards[pair[1]]} overlap")


def check_tensor(shards_metadata, tensor_dims) -> None:
    """
    Checks if the shards_metadata is compatible with the provided tensor dims.

    Args:
        shards_metadata(List[ShardMetadata]): List of :class:`ShardMetadata`
            objects representing each shard of the tensor.
        tensor_dims(Sequence of int): Dimensions of tensor to verify
    Raises:
        ``ValueError`` if not compatible.
    """

    # If the tensor's volume matches the total volume of all shards and
    # all shard boundaries are within tensor dims, we have a compatible
    # sharding spec for this tensor. Note that we have already verified
    # we don't have overlapping shards.
    tensor_rank = len(tensor_dims)
    shards_rank = len(shards_metadata[0].shard_offsets)
    if tensor_rank != shards_rank:
        raise ValueError(
            f"Rank of tensor is {tensor_rank}, but shards rank is {shards_rank}"
        )

    total_shard_volume = 0
    for shard in shards_metadata:
        shard_volume = 1
        for i, shard_length in enumerate(shard.shard_sizes):
            shard_volume *= shard_length
            if shard.shard_offsets[i] + shard.shard_sizes[i] > tensor_dims[i]:
                raise ValueError(
                    f"Shard offset {shard.shard_offsets[i]} and length "
                    f"{shard.shard_sizes[i]} exceeds tensor dim: {tensor_dims[i]} for shard {shard}"
                )
        total_shard_volume += shard_volume

    tensor_volume = 1
    for size in tensor_dims:
        tensor_volume *= size

    if total_shard_volume != tensor_volume:
        # TODO: Can we improve this error message to point out the gaps?
        raise ValueError(
            f"Total volume of shards: {total_shard_volume} "
            f"does not match tensor volume: {tensor_volume}, in other words "
            f"all the individual shards do not cover the entire tensor"
        )


def get_split_size(dim_size, chunks):
    """
    Computes the split size inline with ``torch.chunk``

    Args:
        dim_size(int): Size of the dimension being chunked.
        chunks(int): Number of chunks to create for ``dim_size``.

    Returns:
        An int indicating the split size to use.
    """
    return (dim_size + chunks - 1) // chunks


def get_chunked_dim_size(dim_size, split_size, idx):
    """
    Computes the dim size of the chunk for provided ``idx`` given ``dim_size``
    and ``split_size``.

    Args:
        dim_size(int): Size of the dimension being chunked.
        split_size(int): The chunk size for each chunk of ``dim_size``.
        idx(int): The index of chunk whose dim size is being requested.

    Returns:
        An int indicating the dim size of the chunk.
    """
    return max(min(dim_size, split_size * (idx + 1)) - split_size * idx, 0)


def get_chunk_sharding_params(sharding_dim_size, world_size, spec, rank):
    """
    Generate the start pos and offset length for the current rank for
    chunk sharding.

    Args:
        sharding_dim_size(int): The dimension length which we shard on.
        world_size(int): number of ranks.
        spec (:class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec`):
            sharding spec.
        rank(int): # of cuda process.

    Returns:
        start_pos(int): start position of sharded tensor on the given rank.
        chunk_size(int): chunk size of sharded tensor on the given rank.
    """
    split_size = get_split_size(sharding_dim_size, world_size)
    current_offsets = 0
    start_pos = current_offsets
    for idx, placement in enumerate(spec.placements):
        chunk_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
        if rank == placement.rank():
            start_pos = current_offsets
            break
        current_offsets += chunk_size
    return start_pos, chunk_size  # type: ignore[possibly-undefined]
<|endoftext|># mypy: allow-untyped-defs
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
import torch.distributed.distributed_c10d as distributed_c10d
from torch.distributed._shard._utils import narrow_tensor
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharded_tensor.utils import (
    _parse_and_validate_remote_device,
)

from ._internals import get_chunked_dim_size, get_split_size
from .api import ShardingSpec


if TYPE_CHECKING:
    # Only include ShardedTensor when do type checking, exclude it
    # from run-time to resolve circular dependency.
    from torch.distributed._shard.sharded_tensor import ShardedTensor


@dataclass
class ChunkShardingSpec(ShardingSpec):
    """
    This is a type of PlacementSpec that defines the placement as being sharded
    across multiple devices. In particular, it represents sharding a Tensor
    along a single dimension into equal chunks (similar to :meth:`torch.chunk`).

    The semantics of how a tensor is partitioned is inline with
    :meth:`torch.chunk`, where ``dim`` in torch.chunk corresponds to the
    specified ``dim`` and ``chunks`` in torch.chunk is the number of elements
    in the placement specified.

    Args:
        dim (int):
            The dimension to shard on, an integer representing the dimension.
        placement(List[Union[_remote_device, str]]):
            Specifies the placement of each shard of the Tensor. The size of
            the list represents the number of shards to be created. This could
            be a list of
            :class:`torch.distributed._remote_device`'s. This list
            could also contain a string which represents remote
            device as accepted by
            :class:`torch.distributed._remote_device`
    """

    ShardingDim = int | str

    dim: ShardingDim
    placements: list[torch.distributed._remote_device | str]

    def __post_init__(self):
        self._verify_dim(self.dim)
        for i, remote_device in enumerate(self.placements):
            if not isinstance(remote_device, torch.distributed._remote_device):
                self.placements[i] = torch.distributed._remote_device(remote_device)

    @staticmethod
    def _verify_dim(dim):
        # Validate the sharding spec.
        # TODO: support named dimension
        if isinstance(dim, str):
            raise NotImplementedError(
                "ChunkShardingSpec does not support named dimension yet!"
            )

        if not isinstance(dim, int):
            raise ValueError(f"Sharding dim needs to be an integer, found: {dim}")

    def build_metadata(
        self,
        tensor_sizes: torch.Size,
        tensor_properties: sharded_tensor_meta.TensorProperties,
    ) -> sharded_tensor_meta.ShardedTensorMetadata:
        tensor_num_dim = len(tensor_sizes)

        self._verify_dim(self.dim)
        if self.dim >= tensor_num_dim or self.dim < -tensor_num_dim:  # type: ignore[operator]
            raise ValueError(f"Invalid sharding dim: {self.dim}")

        shards_metadata = []
        sharding_dim_size = tensor_sizes[self.dim]  # type: ignore[index]
        chunks = len(self.placements)
        split_size = get_split_size(sharding_dim_size, chunks)
        for idx, placement in enumerate(self.placements):
            # generate ShardMetadata for each placement device
            chunked_dim_size = get_chunked_dim_size(sharding_dim_size, split_size, idx)
            shard_size = list(tensor_sizes)
            current_offsets = [0] * tensor_num_dim
            current_offsets[self.dim] = split_size * idx  # type: ignore[index]
            shard_size[self.dim] = chunked_dim_size  # type: ignore[index]

            shard_metadata = ShardMetadata(
                shard_offsets=current_offsets,
                shard_sizes=shard_size,
                placement=placement,
            )
            shards_metadata.append(shard_metadata)

        return sharded_tensor_meta.ShardedTensorMetadata(
            shards_metadata, tensor_sizes, tensor_properties
        )

    def shard(
        self, tensor: torch.Tensor, src_rank: int = 0, process_group=None
    ) -> "ShardedTensor":
        """
        Args:
            src_rank: group rank relative to ``process_group``

            N.B. If ``process_group`` is None, ``src_rank`` is a global rank.
        """
        # relative imports to avoid circular dependency
        from torch.distributed._shard.sharded_tensor import ShardedTensor

        tensor_properties = sharded_tensor_meta.TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        )
        current_rank = dist.get_rank(process_group)
        current_global_rank = dist.get_rank()
        tensor_meta = self.build_metadata(tensor.size(), tensor_properties)
        local_shards = []
        local_tensor = None
        local_metadata = None

        tensors_to_scatter = cast(
            list[torch.Tensor | None],
            [None] * dist.get_world_size(process_group),
        )

        sharding_dim_size = tensor.size()[self.dim]  # type: ignore[index]
        chunks = len(self.placements)
        split_size = get_split_size(sharding_dim_size, chunks)
        scatter_shape = list(tensor.size())
        scatter_shape[self.dim] = split_size  # type: ignore[index]

        for shard_meta in tensor_meta.shards_metadata:
            remote_global_rank, device = _parse_and_validate_remote_device(
                process_group, shard_meta.placement
            )
            if current_rank == src_rank:
                # Reshape to get shard for this rank and we don't want autograd
                # recording here for the narrow op and 'local_shard' should be a
                # leaf variable in the autograd graph.
                narrowed_tensor = narrow_tensor(tensor, shard_meta)
                if shard_meta.shard_sizes[self.dim] < split_size:  # type: ignore[index]
                    # for the last shard that might be smaller to other shards
                    # resize the narrowed tensor to the same size and use it for
                    # the scatter collective as dist.scatter requires same size
                    # inputs on every rank
                    tensor_to_scatter = (
                        narrowed_tensor.detach().clone().resize_(scatter_shape)
                    )
                else:
                    tensor_to_scatter = narrowed_tensor.detach().clone(
                        memory_format=torch.contiguous_format
                    )

                tensors_to_scatter[
                    # pyrefly: ignore [bad-argument-type]
                    dist.get_group_rank(process_group, remote_global_rank)
                ] = tensor_to_scatter

            if current_global_rank == remote_global_rank:
                local_tensor = torch.empty(
                    scatter_shape,
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    device=device,
                )
                local_metadata = shard_meta

        # each rank should have local_tensor and local_metadata initialized if we build
        # the metadata list in a correct way.
        if local_tensor is None:
            raise AssertionError
        if local_metadata is None:
            raise AssertionError

        # Scatter the shards to all ranks in the pg
        # scatter takes the global rank as ``src``
        src_for_scatter = src_rank
        if (
            process_group is not None
            and process_group is not distributed_c10d._get_default_group()
        ):
            src_for_scatter = distributed_c10d.get_global_rank(
                process_group, src_for_scatter
            )

        tensors_to_scatter_: list[torch.Tensor] | None = None
        if current_rank == src_rank:
            tensors_to_scatter_ = []
            for t in tensors_to_scatter:
                if not isinstance(t, torch.Tensor):
                    raise AssertionError
                tensors_to_scatter_.append(t)

        dist.scatter(
            local_tensor,
            scatter_list=tensors_to_scatter_,
            src=src_for_scatter,
            group=process_group,
        )

        if list(local_tensor.size()) != local_metadata.shard_sizes:
            # detach again after receiving to ensure local shards remain a leaf node
            local_tensor = local_tensor.resize_(local_metadata.shard_sizes).detach()

        # Sync requires_grad to local_shard.
        local_tensor.requires_grad = tensor.requires_grad

        local_shards.append(Shard(tensor=local_tensor, metadata=local_metadata))

        st = ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards, tensor_meta, process_group=process_group
        )

        # Manually set sharding_spec
        st._sharding_spec = self

        return st
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter

from ._common import (
    _all_gather_base_input,
    _handle_col_wise_sharding_base,
    _handle_max_norm_col_wise,
    _handle_row_wise_mask,
)


@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.embedding)
def sharded_embedding(types, args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding``.
    This method computes a sharded embedding lookup and has the following limitations:

    1. Supports only sharding of ``weight``.
    2. Supports only ``ChunkShardingSpec``.
    3. Supports only a single local shard per rank.
    4. Supports all specs except for scale_grad_by_freq, sparse, etc.

    Based on the dimension that the weight is sharded on, there are two
    algorithms:

    ROWWISE SHARDING
    ================
    For row-wise sharding the weight is sharded on dimension 0.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (10 x 17) and W is sharded across
    4 GPUs creating 3 shard of (3 x 17) and 1 shard of (1 x 17).
    The algorithm is as follows:

    1. First the input is all gathered to all ranks, since this is SPMD and
       input is actually sharded across all ranks. The inputs then become a
       4 (4 x 6) tensor on each rank. For example if the given input is
       tensor([[6, 5, 2, 9, 6, 3],
               [3, 1, 2, 4, 7, 6],
               [4, 0, 4, 9, 8, 9],
               [8, 6, 6, 4, 6, 1]])
       on rank 0.
       Then on every rank, we will have this tensor.
       If input itself is already replicated, no all-gather will be done.
    2. Next, we mask the ID which are not stored on that rank.
       For example on rank 0, we store ID [0, 1, 2]. We only keep the ID
       inside the set of numbers. The rest of them will be masked to an extra row.
       The masked matrix will be used for embedding look up and is like:
       tensor([[4, 4, 2, 4, 4, 4],
               [4, 1, 2, 4, 4, 4],
               [4, 0, 4, 4, 4, 4],
               [4, 4, 4, 4, 4, 1]])
       The reason of having an extra row (aka, number 4 in the example) is
       because when max_norm is specified only weight which has looked will
       be re-normed so mask IDs whose embeddings are not stored in current
       rank will to an extra row will ensure max_norm still works as expected.
    3. If max_norm is specified, the extra row guarantees that the mask ID will
       not affect the behavior of weight re-norm.

    COLWISE SHARDING
    ================
    For col-wise sharding the weight is sharded on dimension 1.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across
    4 GPUs creating 3 shards of (16 x 5) and 1 shard of (16 x 2).
    The algorithm is as follows:

    1. First the input is broadcasted to all ranks, since this is SPMD we
       actually do an all_gather for all the inputs resulting in 4 (4 x 6)
       inputs on each rank.
    2. Next we perform local embedding lookup operation by apply each
       input (4 x 6) with the local shard (16 x 5) ((16 x 2) for the last).
       This results in 4 (5 x 6 x 4) ((2 x 6 x 4) for the last) matrices
       on each rank. We transpose dim 0 and dim 2.
    3. Next, we concat these 4 matrices and perform an all2all to share the
       appropriate (5 x 6 x 4) or (2 x 6 x 4) matrices to each rank.
    4. Now, each rank receives a (17 x 6 x 4) matrix which is basically the
       size of the result we need.
    5. If placements are not in order any appropriate rearrangement of columns
       are done for the (17 x 6 x 4) matrix and finally we transpose the
       dim 0 and dim 2 again.
    6. If max_norm is specified, we manually sum up the norm and renorm. Because
       the renorm must be in place, we need to override the local_shard to mimic
       this behavior.
    """
    # Validate input params
    _validate_embedding_param(args, kwargs)

    input = args[0]
    weight = args[1]
    max_norm = kwargs.get("max_norm")
    norm_type = kwargs.get("norm_type")
    padding_idx = kwargs.get("padding_idx")

    local_shard = weight.local_tensor().contiguous()
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)

    if sharding_dim == 1:
        output, local_shard = _handle_col_wise_sharding(
            input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, pg
        )
        weight.local_shards()[0].tensor = local_shard
        return output
    elif sharding_dim == 0:
        return _handle_row_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            max_norm,
            norm_type,
            padding_idx,
            rank,
            pg,
        )
    else:
        raise RuntimeError(
            f"nn.Embedding weight sharded on dim {sharding_dim} not supported!"
        )


def _validate_embedding_param(args, kwargs):
    """
    Validate input params of sharded embedding op.

    Args:
        input: list of ID used for lookup.
        weight: sharded weight tensor.
        kwargs: same as normal Embedding.

    Return: None.
    """

    input = args[0]
    weight = args[1]
    max_norm = kwargs.get("max_norm")
    scale_grad_by_freq = kwargs.get("scale_grad_by_freq")
    sparse = kwargs.get("sparse")

    # Validate types
    if not isinstance(input, torch.Tensor):
        raise TypeError("input need to be torch.Tensor")
    if not isinstance(weight, ShardedTensor):
        raise TypeError("weight needs to be ShardedTensor")
    weight_size = weight.size()
    if len(weight_size) != 2:
        raise ValueError("Weight needs to have exactly 2 dims")
    if int(torch.min(input).item()) < 0:
        raise ValueError(
            "Index out of range in Input %d %d",
            int(torch.min(input).item()),
            weight_size[1],
        )
    if int(torch.max(input).item()) >= weight_size[0]:
        raise ValueError(
            "Index out of range in Input %d %d",
            int(torch.max(input).item()),
            weight_size[1],
        )
    if scale_grad_by_freq:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "scale_grad_by_freq" not supported!'
        )
    if sparse:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "sparse" not supported!'
        )
    if max_norm and max_norm <= 0.0:
        raise ValueError('"max_norm" must be larger than zero!')

    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError("Only ChunkShardingSpec supported for ShardedTensor ops!")
    if len(weight.local_shards()) != 1:
        raise ValueError("Only one local shard supported!")


def _handle_col_wise_sharding(
    input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, pg
):
    """
    Entry-point function to handle the logic of col-wise sharding of weight
    for embedding. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: col-wise shared local weight used for lookup.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
        pg: process group.

    Returns: final result of lookup.
    """
    # allgather the inputs first for non Replicated Tensor.
    gathered_inputs = all_gather(input, group=pg)

    if max_norm is not None:
        # max_norm changes the weight in-place
        local_shard = _handle_max_norm_col_wise(
            max_norm, norm_type, local_shard, input, world_size, gathered_inputs, pg
        )

    output = _handle_col_wise_sharding_base(
        torch.nn.functional.embedding,
        len(input.size()),
        input,
        world_size,
        weight,
        local_shard,
        pg,
        gathered_inputs,
        padding_idx=padding_idx,
    )
    return (output, local_shard)


def _handle_row_wise_sharding(
    input, world_size, weight, local_shard, max_norm, norm_type, padding_idx, rank, pg
):
    """
    Entry-point function to handle the logic of row-wise sharding of weight
    for embedding. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: row-wise shared local weight used for lookup.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
        rank: # of cuda process.
        pg: process group.

    Returns: final result of lookup.
    """
    # allgather the inputs first for non Replicated Tensor.
    gather_inp = _all_gather_base_input(input, pg)

    # Mask the input according to sharding spec.
    lookup_input, padding_idx, padding_row = _handle_row_wise_mask(
        gather_inp, padding_idx, weight, world_size, rank
    )

    # When input is a large tensor, the value of weight is changed.
    # This is a walk-around for now. GH issue: #81717
    if max_norm is not None:
        torch.nn.functional.embedding(
            torch.unique(lookup_input)[:-1],
            local_shard,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
        )
        max_norm = None

    local_input_embeddings = torch.nn.functional.embedding(
        lookup_input,
        torch.cat([local_shard, padding_row]),
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
    )

    # TODO: Make the result a PartialTensor.
    local_shards = local_input_embeddings.chunk(pg.size())
    return reduce_scatter(
        torch.empty_like(local_shards[0]),
        list(local_shards),
        group=pg,
    )
<|endoftext|># mypy: allow-untyped-defs

from typing import cast

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import all_gather, reduce_scatter

from ._common import (
    _all_gather_base_input,
    _handle_col_wise_sharding_base,
    _handle_max_norm_col_wise,
    _handle_row_wise_mask,
)


@custom_sharding_spec_op(ChunkShardingSpec, torch.nn.functional.embedding_bag)
def sharded_embedding_bag(types, args, kwargs, pg):
    """
    Handles ``__torch_function__`` dispatch for ``torch.nn.functional.embedding_bag``.
    This method computes a sharded embedding bag aggregation and has the following limitations:

    1. Supports only sharding of ``weight``.
    2. Supports only ``ChunkShardingSpec``.
    3. Supports only a single local shard per rank.
    4. Supports all specs except for scale_grad_by_freq, sparse, etc.

    Based on the dimension that the weight is sharded on, there are two
    algorithms:

    ROWWISE SHARDING
    ================
    For row-wise sharding the weight is sharded on dimension 0.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across
    4 GPUs creating 4 shard of (4 x 17).
    The algorithm is as follows:

    1. First the input is all gathered to all ranks, since this is SPMD and
       input is actually sharded across all ranks. The inputs then become a
       4 (4 x 6) tensor on each rank. For example if the given input is
       tensor([[6, 5, 2, 9, 6, 3],
               [3, 1, 2, 4, 7, 6],
               [4, 0, 4, 9, 8, 9],
               [8, 6, 6, 4, 6, 1]])
       on rank 0.
       Then on every rank, we will have this tensor.
       If input itself is already replicated, no all-gather will be done.
    2. Next, we mask the ID which are not stored on that rank.
       For example on rank 0, we store ID [0, 1, 2]. We only keep the ID
       inside the set of numbers. The rest of them will be masked to an extra row.
       The masked matrix will be used for embedding look up and is like:
       tensor([[4, 4, 2, 4, 4, 4],
               [4, 1, 2, 4, 4, 4],
               [4, 0, 4, 4, 4, 4],
               [4, 4, 4, 4, 4, 1]])
    3. If ``max_norm`` is specified, the extra row guarantees that the mask ID will
       not affect the behavior of weight re-norm.
    4. The example above only happens in one rank and each rank does a very similar thing.
       For "Mean" mode we need to divide by either column size (2D) or the interval length
       defined by the offset (excluding the row specified in ``padding_idx``).
       We also need to mask the nonexistent row to neg Inf so that negative value does not
       gets wiped out in the "Max" mode.

    COLWISE SHARDING
    ================
    For col-wise sharding the weight is sharded on dimension 1.

    The overall algorithm can be best explained with an example. Let's assume
    the dims for input are (4 x 6) and W are (16 x 17) and W is sharded across
    4 GPUs creating 3 shards of (16 x 5) and 1 shard of (16 x 2).
    The algorithm is as follows:

    1. First the input is broadcasted to all ranks, since this is SPMD we
       actually do an all_gather for all the inputs resulting in 4 (4 x 6)
       inputs on each rank.
    2. Next we perform local embedding bag operation under the given mode by
       apply each input (4 x 6) with the local shard (16 x 5) ((16 x 2) for the last).
       This results in 4 (5 x 4) ((2 x 4) for the last) matrices on each rank.
       We transpose the aggregation result.
    3. Next, we concatenate these 4 matrices and perform an all2all to share the
       appropriate (5 x 4) or (2 x 4) matrices to each rank.
    4. Now, each rank receives a (17 x 4) matrix which is basically the
       size of the result we need.
    5. If placements are not in order any appropriate rearrangement of columns
       are done for the (17 x 4) matrix and finally we transpose the output again.
    6. If max_norm is specified, we manually sum up the norm and renorm. Because
       the renorm must be in place, we need to override the local_shard to mimic
       this behavior.
    """
    # Validate input params
    _validate_embedding_bag_param(args, kwargs)

    input = args[0]
    weight = args[1]
    offsets = kwargs.get("offsets")
    per_sample_weights = kwargs.get("per_sample_weights")
    mode = kwargs.get("mode")
    max_norm = kwargs.get("max_norm")
    norm_type = kwargs.get("norm_type")
    include_last_offset = kwargs.get("include_last_offset")
    padding_idx = kwargs.get("padding_idx")

    local_shard = weight.local_tensor().contiguous()
    sharding_dim = weight._sharding_spec.dim
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if include_last_offset:
        offsets = offsets[:-1]

    if sharding_dim == 1:
        output, local_shard = _handle_col_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            offsets,
            per_sample_weights,
            mode,
            max_norm,
            norm_type,
            padding_idx,
            pg,
        )
        weight.local_shards()[0].tensor = local_shard
        return output
    elif sharding_dim == 0:
        return _handle_row_wise_sharding(
            input,
            world_size,
            weight,
            local_shard,
            offsets,
            per_sample_weights,
            mode,
            max_norm,
            norm_type,
            padding_idx,
            rank,
            pg,
        )
    else:
        raise RuntimeError(
            f"nn.EmbeddingBag weight sharded on dim {sharding_dim} not supported!"
        )


def _validate_embedding_bag_param(args, kwargs):
    """
    Validate input params of sharded embeddingBag op.

    Args:
        input: list of ID used for lookup and aggregation.
        weight: sharded weight tensor.
        kwargs: same as normal EmbeddingBag.

    Return: None.
    """

    input = args[0]
    weight = args[1]
    offsets = kwargs.get("offsets")
    per_sample_weights = kwargs.get("per_sample_weights")
    mode = kwargs.get("mode")
    max_norm = kwargs.get("max_norm")
    scale_grad_by_freq = kwargs.get("scale_grad_by_freq")
    sparse = kwargs.get("sparse")
    include_last_offset = kwargs.get("include_last_offset")

    # Validate types
    if not isinstance(input, torch.Tensor):
        raise TypeError("input need to be torch.Tensor")
    if offsets is not None and not isinstance(offsets, torch.Tensor):
        raise TypeError("offsets need to be torch.Tensor")
    if per_sample_weights is not None and not isinstance(
        per_sample_weights, torch.Tensor
    ):
        raise TypeError("per_sample_weights need to be torch.Tensor")
    if not isinstance(weight, ShardedTensor):
        raise TypeError("weight needs to be ShardedTensor")
    if len(input.size()) > 2:
        raise ValueError("Input more than 2 dims not supported")
    weight_size = weight.size()
    if len(weight_size) != 2:
        raise ValueError("Weight needs to have exactly 2 dims")
    if int(torch.min(input).item()) < 0:
        raise ValueError(
            "Index out of range in Input %d %d",
            int(torch.min(input).item()),
            weight_size[1],
        )
    if int(torch.max(input).item()) >= weight_size[0]:
        raise ValueError(
            "Index out of range in Input %d %d",
            int(torch.max(input).item()),
            weight_size[1],
        )
    if offsets is not None and len(input.size()) != 1:
        raise ValueError("Input dimension needs to be exactly 1 dim")
    if len(input.size()) == 1 and offsets is None:
        raise ValueError("offsets is required for 1D input")
    if per_sample_weights is not None and per_sample_weights.size() != input.size():
        raise ValueError(
            f"per_sample_weights size {per_sample_weights.size()} not equal to input size {input.size()}"
        )
    if mode is None:
        mode = "mean"
    if mode not in ["sum", "mean", "max"]:
        raise ValueError(f"mode '{mode}' is not supported")
    if scale_grad_by_freq:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "scale_grad_by_freq" not supported!'
        )
    if sparse:
        raise RuntimeError(
            'nn.Embedding weight sharded with flag on "sparse" not supported!'
        )
    if include_last_offset and offsets is None:
        raise ValueError('offsets is required for flag "include_last_offset"!')
    if include_last_offset and cast(list[int], offsets)[-1] != input.size(0):
        raise ValueError(
            'offsets need to have the input size in the end when the flag "include_last_offset" is on!'
        )

    if max_norm and max_norm <= 0.0:
        raise ValueError('"max_norm" must be larger than zero!')

    if not isinstance(weight._sharding_spec, ChunkShardingSpec):
        raise ValueError("Only ChunkShardingSpec supported for ShardedTensor ops!")
    if len(weight.local_shards()) != 1:
        raise ValueError("Only one local shard supported!")


def _handle_col_wise_sharding(
    input,
    world_size,
    weight,
    local_shard,
    offsets,
    per_sample_weights,
    mode,
    max_norm,
    norm_type,
    padding_idx,
    pg,
):
    """
    Entry-point function to handle the logic of col-wise sharding of weight
    for embeddingBag. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding_bag.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: col-wise shared local weight used for lookup.
        offsets: list of start positions of each bag for 1D input.
        per_sample_weights: weights for weighted sum mode.
        mode: aggregation method of each bag.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.
        pg: process group.

    Return:
        output: final result of lookup and aggregation.
        local_shard: col-wise shared local weight used for lookup.
            If max_norm, this will be the renormed weight.
    """
    # allgather the special input of embedding bag first.
    (
        gathered_inputs,
        gathered_per_sample_weights,
        gathered_offsets,
    ) = _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg)

    if max_norm is not None:
        # max_norm changes the weight in-place
        local_shard = _handle_max_norm_col_wise(
            max_norm, norm_type, local_shard, input, world_size, gathered_inputs, pg
        )

    output = _handle_col_wise_sharding_base(
        torch.nn.functional.embedding_bag,
        1,
        input,
        world_size,
        weight,
        local_shard,
        pg,
        gathered_inputs,
        mode=mode,
        gathered_per_sample_weights=gathered_per_sample_weights,
        gathered_offsets=gathered_offsets,
        padding_idx=padding_idx,
    )
    return (output, local_shard)


def _handle_row_wise_sharding(
    input,
    world_size,
    weight,
    local_shard,
    offsets,
    per_sample_weights,
    mode,
    max_norm,
    norm_type,
    padding_idx,
    rank,
    pg,
):
    """
    Entry-point function to handle the logic of row-wise sharding of weight
    for embeddingBag. (Detailed explanations of the logic can be found in
    the comment for sharded_embedding_bag.)

    Args:
        input: list of ID used for lookup and aggregation.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: row-wise shared local weight used for lookup.
        offsets: list of start positions of each bag for 1D input.
        per_sample_weights: weights for weighted sum mode.
        mode: aggregation method of each bag.
        max_norm: If given, each embedding vector with norm larger
            than max_norm is renormalized to have norm max_norm.
            Note: this will modify weight in-place.
        norm_type: The p in the p-norm to compute for the max_norm option.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.
        rank: # of cuda process.
        pg: process group.

    Returns:
        gathered_output: final result of lookup and aggregation.
    """
    if input.dim() > 1 and per_sample_weights is None:
        # allgather the inputs first for non Replicated Tensor.
        gather_inp = _all_gather_base_input(input, pg)
    else:
        (
            gathered_inputs,
            gathered_per_sample_weights,
            gathered_offsets,
        ) = _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg)
        cat_dim = 0 if input.dim() != 1 else -1
        gather_inp = torch.cat(gathered_inputs, dim=cat_dim)
        if per_sample_weights is not None:
            per_sample_weights = torch.cat(gathered_per_sample_weights, dim=cat_dim)
        offset_add = 0 if input.dim() > 1 else input.size(0)
        if offsets is not None:
            offsets_list = torch.cat(
                [gathered_offsets[i] + (offset_add * i) for i in range(pg.size())],
                dim=cat_dim,
            )

    # Mask the input according to sharding spec.
    lookup_input, padding_local, padding_row = _handle_row_wise_mask(
        gather_inp, padding_idx, weight, world_size, rank
    )
    if mode == "max":
        padding_row[:] = -float("Inf")

    # When input is a large tensor, the value of weight is changed.
    # This is a walk-around for now. GH issue: #81717.
    if max_norm is not None:
        torch.nn.functional.embedding_bag(
            torch.unique(lookup_input)[:-1],
            local_shard,
            offsets=torch.tensor([0], device=local_shard.device, dtype=torch.long),
            mode=mode,
            per_sample_weights=None,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_local,
        )
        max_norm = None
    result = torch.nn.functional.embedding_bag(
        lookup_input,
        torch.cat([local_shard, padding_row]),
        offsets=offsets_list if offsets is not None else offsets,  # type: ignore[possibly-undefined]
        mode=mode if mode != "mean" else "sum",
        per_sample_weights=per_sample_weights,
        max_norm=max_norm,
        norm_type=norm_type,
        padding_idx=padding_local,
    )

    op = ReduceOp.SUM if mode != "max" else ReduceOp.MAX
    # TODO: Make the result a PartialTensor and move the logic below there.
    local_shards = result.chunk(pg.size())
    result = reduce_scatter(
        torch.empty_like(local_shards[0]),
        list(local_shards),
        op=op,
        group=pg,
    )

    # For Mean, we cannot do the division until very end because the sum of means
    # not equal to the mean of sum. (Divisor is different)
    if mode == "mean":
        if input.dim() > 1:
            padding_idx = padding_idx if padding_idx is not None else -1
            split_sizes = torch.sum(
                torch.ne(input, padding_idx), dim=-1, dtype=local_shard.dtype
            )
        else:
            split_sizes = torch.cat(
                (
                    offsets[1 : offsets.size(0)] - offsets[0:-1],
                    (input.size(0) - offsets[-1]).unsqueeze(0),
                ),
                dim=-1,
            )
        return torch.div(result, split_sizes.unsqueeze(1))

    # Return the appropriate local result.
    return result


def _all_gather_embedding_bag_input(input, per_sample_weights, offsets, pg):
    """
    In case we need to gather input and all other parameters of embeddingBag
    ops, we need to stack all input together to perform ``all_gather``
    collective communication just once.

    Note that since offsets does not share the same size as input and
    is always smaller than input, we resize it during the communication.

    Args:
        input: tensor to be applied op on.
        per_sample_weights: weights for weighted sum mode.
        offsets: when input is 1D. offsets determines the starting
            index position of each bag (sequence) in input.
        pg: process group.

    Returns:
        gathered_inputs: list of input tensor gathered from each rank.
        gathered_per_sample_weights: list of per_sample_weights from each rank.
        gathered_offsets: list of offsets from each rank.
    """
    input_to_gather = [input]
    if per_sample_weights is not None:
        input_to_gather.append(per_sample_weights)
    if offsets is not None:
        input_to_gather.append(offsets.clone().resize_(input.size()))
    gathered_inputs = all_gather(torch.stack(input_to_gather), group=pg)

    gathered_per_sample_weights = None
    if per_sample_weights is not None:
        gathered_per_sample_weights = [t[1] for t in gathered_inputs]
    gathered_offsets = None
    if offsets is not None:
        idx = 2 if per_sample_weights is not None else 1
        gathered_offsets = [
            t[idx].resize_(offsets.size()).to(offsets.dtype) for t in gathered_inputs
        ]
    gathered_inputs = [t[0].to(input.dtype) for t in gathered_inputs]
    return gathered_inputs, gathered_per_sample_weights, gathered_offsets
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor._ops._common import _sharded_op_common
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec._internals import (
    get_chunk_sharding_params,
    get_chunked_dim_size,
    get_split_size,
)
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import (
    _all_gather_base,
    all_reduce,
    all_to_all_single,
)


def _chunk_sharding_spec_check(spec, op):
    """
    For the given op implementation check if the sharding spec is ChunkShardingSpec.
    """
    if not isinstance(spec, ChunkShardingSpec):
        raise NotImplementedError(
            f"Only ChunkShardingSpec supported for '{op.__name__}'."
        )


def _register_sharded_op_on_local_tensor(
    op, early_stop_func=None, extra_check=None, customized_func=None
):
    """
    Handles ``__torch_function__`` dispatch for ops which are performed on
    the single local tensor of the sharded tensor such as op like
    ``torch.nn.functional.softmax`` or ``torch.Tensor.view``.

    For more complicated ops, a customized func can be used to generate
    the new local tensor, sharding spec and sharded tensor size.

    Args:
        op: The op to be registered and applied to all shards of the st.
        early_stop_func (Callable, optional): the func for early stop.
            Default: if ``None``, no early stop.
        extra_check (Callable, optional): the func for extra condition check.
            Default: if ``None``, no extra check.
        customized_func (Callable, optional): the func for customized logic
            to generate the new local tensor, sharding spec and sharded tensor size.
            Default: if ``None``, we simply lower to the real op call with
                the single local tensor of the st.

    Return:
        func (Callable): registered implementation for sharded op for
        ``__torch_function__`` dispatch.
    """

    @custom_sharding_spec_op(ChunkShardingSpec, op)
    @_sharded_op_common(op, early_stop_func, extra_check)
    def sharded_tensor_op_on_local_tensor(types, args=(), kwargs=None, pg=None):
        # pyrefly: ignore [bad-index]
        st = args[0]
        sharding_spec = st.sharding_spec()
        if len(st.local_shards()) != 1:
            raise TypeError(
                f"torch function '{op.__name__}', with args: {args} and "
                f"kwargs: {kwargs} only supported for single local tensor!"
            )
        st_size = st.size()
        if customized_func:
            local_tensor, sharding_spec, st_size = customized_func(args, kwargs, pg)
        else:
            args = (st.local_tensor(), *args[1:])
            local_tensor = op(*args, **kwargs)
        return ShardedTensor._init_from_local_tensor(
            local_tensor.contiguous(),
            sharding_spec,
            st_size,  # type: ignore[arg-type]
            process_group=pg,
            init_rrefs=st._init_rrefs,
        )


def _handle_col_wise_sharding_base(
    op_func,
    col_dim,
    input,
    world_size,
    weight,
    local_shard,
    pg,
    gathered_inputs,
    mode=None,
    gathered_per_sample_weights=None,
    gathered_offsets=None,
    padding_idx=None,
):
    """
    For col-wise sharding of weight, lots of logic are common.
    So we extract the common logic and put in this function:
    Step 1. To get input from each rank and
    Step 2. To perform the op on the concatenated tensor.
    Step 3. To distribute results to each rank with col rearrangement.
    Step 4. To concatenate all results from all ranks.

    Args:
        op_func: operator which is applied to the input tensor.
        col_dim: dim of result tensor after the operation.
        input: tensor to be applied op on.
        world_size: number of ranks.
        weight: sharded weight tensor.
        local_shard: col-wise sharded weight tensor.
        pg: process group.
        gathered_inputs: list of inputs from all ranks. If specified, we
            don't need to communicate with each rank any more.
        mode: aggregation mode of EmbeddingBag.
        gathered_per_sample_weights: per_sample_weights across all ranks.
        gathered_offsets: offsets across all ranks.
        padding_idx: If specified, the entries at padding_idx do
            not contribute to the gradient; therefore, the embedding
            vector at padding_idx is not updated during training,
            i.e. it remains as a fixed "pad".
            Note that the embedding vector at padding_idx is
            excluded from the reduction.

    Return: final result of input being applied with the op.
    """
    # run the operator's function for all the inputs.
    results = []
    for i, inp in enumerate(gathered_inputs):
        if op_func is torch.nn.functional.embedding_bag:
            result = op_func(
                inp,
                local_shard,
                offsets=gathered_offsets[i] if gathered_offsets is not None else None,
                # pyrefly: ignore [bad-argument-type]
                mode=mode,
                per_sample_weights=gathered_per_sample_weights[i]
                if gathered_per_sample_weights is not None
                else None,
                padding_idx=padding_idx,
            )
        elif op_func is torch.nn.functional.embedding:
            result = op_func(
