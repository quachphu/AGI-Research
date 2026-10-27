from torch._C._distributed_c10d import FakeWork, PythonCallbackWork
from torch.distributed._mesh_layout import _FlatLayout
from torch.distributed.distributed_c10d import (
    _check_op,
    _get_default_group,
    _resolve_process_group,
    GroupName,
    ProcessGroup,
    ReduceOp,
    Work,
)


# NOTE: Most of the c10d collectives often take a Tensor[] (or Tensor[][])
# when you would expect Tensor (or Tensor[]).  In fact, there will only ever
# be one Tensor in this case; the old signature was to support dispatching a
# collective on multiple devices (ala DataParallel) but we don't support that
# API anymore.  Note that we are not 100% consistent about this; some more
# modern collectives like _allgather_base_ got rid of the unnecessary list.
# When in doubt, consult the code that dispatches to the collective on the PG
# in distributed_c10d.py e.g., work = group.allgather([tensor_list], [tensor],
# opts) indicates it's always a list.


def _gcd_list(numbers: Sequence[int]) -> int:
    return 0 if not numbers else functools.reduce(math.gcd, numbers)


def _indices_to_layout(indices: list[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    # Base case: A single index represents a point, not a dimension.
    if len(indices) <= 1:
        return (), ()

    # The smallest stride is likely the GCD of the differences between consecutive indices.
    # For a sorted, unique list, all differences will be positive.
    diffs = [indices[i] - indices[i - 1] for i in range(1, len(indices))]
    last_stride = _gcd_list(diffs)

    # This case should not be reached if indices are unique and sorted.
    if last_stride == 0:
        raise AssertionError("Cannot determine stride; indices may not be unique.")

    # Identify the starting index of each "row" in the last dimension.
    # An index starts a new row if the preceding index (index - stride) is not present.
    indices_set = set(indices)
    higher_dim_indices = [indices[0]]
    for index in indices[1:]:
        if (index - last_stride) not in indices_set:
            higher_dim_indices.append(index)

    # From the number of rows, we can deduce the shape of the last dimension.
    if len(indices) % len(higher_dim_indices) != 0:
        raise AssertionError(
            "Indices do not form a regular grid. "
            f"Found {len(higher_dim_indices)} subgroups for {len(indices)} total elements."
        )
    last_shape = len(indices) // len(higher_dim_indices)

    # Recurse on the higher-dimensional indices (the start of each row).
    higher_shapes, higher_strides = _indices_to_layout(higher_dim_indices)

    # Combine the results from the recursion with the current dimension's results.
    final_shapes = higher_shapes + (last_shape,)
    final_strides = higher_strides + (last_stride,)

    return final_shapes, final_strides


def _resolve_pg_or_name(
    group: ScriptObject | ProcessGroup | str | GroupName,
) -> ScriptObject | ProcessGroup:
    """Accept any of the group reps that may reach a functional collective.

    Under ``compile_on_one_rank=True`` the functional collective ops receive
    the live ``ProcessGroup`` instead of a string group name, so we cannot
    unconditionally call ``_resolve_process_group``.
    """
    if isinstance(group, (ProcessGroup, ScriptObject)):
        return group
    # ``GroupName`` is ``NewType("GroupName", str)``, so wrapping ``group`` here
    # is a runtime no-op. The cast exists purely so pyrefly accepts the
    # narrowed ``str | GroupName`` argument against the declared parameter type.
    return _resolve_process_group(GroupName(group))


def _prepare_collective_groups(
    process_group_so: ScriptObject | ProcessGroup,
) -> tuple[list[int], list[int], int]:
    process_group = (
        ProcessGroup.unbox(process_group_so)
        if isinstance(process_group_so, ScriptObject)
        else process_group_so
    )

    ranks = torch.distributed.get_process_group_ranks(process_group)
    if not ranks:
        raise AssertionError
    # TODO: We can handle permutations but the layout inference algorithm will
    # lose the permutation so we will have to reapply it
    if ranks != sorted(ranks):
        raise AssertionError(ranks)
    offset = ranks[0]
    ranks = [r - offset for r in ranks]

    shape, strides = _indices_to_layout(ranks)
    layout = _FlatLayout(shape, strides)

    global_pg = _get_default_group()
    group_offsets = layout.complement(global_pg.size()).all_ranks_from_zero()

    return ranks, group_offsets, offset


# NB: There are two flavors of the collectives: regular and functional. Regular collectives
# allocate outputs to write the result to, accept process group and support async ops (return
# work object). Functional collectives expect the implementation to allocate outputs, accept
# process group name that must be resolved and do not support async ops (return output).
def _local_functional_all_gather_into_tensor(
    tensor: torch.Tensor,
    group_size: int,
    group_name: GroupName | ProcessGroup,
) -> torch.Tensor:
    # "all_gather_into_tensor(Tensor input, int group_size, str group_name) -> Tensor"
    from . import LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_pg_or_name(group_name)
    )

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")
    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        group_tensors = []
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue

        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        gathered_tensor = torch.cat(group_tensors, dim=0)

        for rank in group_ranks:
            output_local_tensors[rank] = gathered_tensor.clone()

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_functional_reduce_scatter_tensor(
    tensor: torch.Tensor,
    reduce_op: str,
    group_size: int,
    group_name: GroupName | ProcessGroup,
) -> torch.Tensor:
    #  "reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, str group_name) -> Tensor"
    from . import _zero_sized_like, LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_pg_or_name(group_name)
    )

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")
    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        group_tensors = []
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue

        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        reduced_tensor = _local_reduce(reduce_op, group_tensors)

        scattered_tensor = torch.split(
            reduced_tensor,
            reduced_tensor.size(0) // len(group_ranks),
            dim=0,
        )

        for i, rank in enumerate(group_ranks):
            if i < len(scattered_tensor):
                output_local_tensors[rank] = scattered_tensor[i].clone()
            else:
                output_local_tensors[rank] = _zero_sized_like(reduced_tensor, 0)

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_functional_shard_dim_alltoall(
    tensor: torch.Tensor,
    gather_dim: int,
    shard_dim: int,
    group_name: GroupName | ProcessGroup,
) -> torch.Tensor:
    # "shard_dim_alltoall(Tensor input, int gather_dim, int shard_dim, str group_name) -> Tensor"
    from . import _zero_sized_like, LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_pg_or_name(group_name)
    )

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")
    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        group_tensors = []
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue

        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        gathered_tensor = torch.cat(group_tensors, dim=gather_dim)

        split_tensor = torch.split(
            gathered_tensor,
            gathered_tensor.size(shard_dim) // len(group_ranks),
            dim=shard_dim,
        )

        for i, rank in enumerate(group_ranks):
            if i < len(split_tensor):
                output_local_tensors[rank] = split_tensor[i].clone()
            else:
                output_local_tensors[rank] = _zero_sized_like(
                    gathered_tensor, shard_dim
                )

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_functional_all_to_all_single(
    tensor: torch.Tensor,
    output_split_sizes: list[torch.SymInt],
    input_split_sizes: list[torch.SymInt],
    group_name: GroupName | ProcessGroup,
) -> torch.Tensor:
    # "all_to_all_single(Tensor input, SymInt[] output_split_sizes, SymInt[] input_split_sizes, str group_name) -> Tensor"
    from . import LocalIntNode, LocalTensor

    ranks, group_offsets, offset = _prepare_collective_groups(
        _resolve_pg_or_name(group_name)
    )

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")

    split_local_sizes: dict[int, list[int]] = {}
    for input_split_size in input_split_sizes:
        if isinstance(input_split_size, torch.SymInt) and isinstance(
            input_split_size.node, LocalIntNode
        ):
            local_ints = dict(input_split_size.node._local_ints.items())
        else:
            local_ints = {rank: int(input_split_size) for rank in tensor._local_tensors}
        for rank, split_size in local_ints.items():
            if rank not in split_local_sizes:
                split_local_sizes[rank] = []
            split_local_sizes[rank].append(split_size)

    split_local_tensors: dict[int, list[torch.Tensor]] = {}

    for rank, split_sizes in split_local_sizes.items():
        split_local_tensors[rank] = list(
            torch.split(tensor._local_tensors[rank], split_sizes)
        )

    output_local_tensors: dict[int, torch.Tensor] = {}

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        if not all(rank in split_local_tensors for rank in group_ranks):
            continue

        for i, dst in enumerate(group_ranks):
            splits = []
            for src in group_ranks:
                splits.append(split_local_tensors[src][i])
            output_local_tensors[dst] = torch.cat(splits)

    # pyrefly: ignore [bad-argument-type, bad-argument-count]
    output = LocalTensor(output_local_tensors)

    return output


def _local_broadcast_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    root_rank: int,
    root_tensor: int,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # "broadcast_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int root_rank, int root_tensor, bool async_op=True, int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)"
    from . import LocalTensor

    if len(tensors) != 1:
        raise AssertionError
    if root_tensor != 0:
        raise AssertionError
    tensor = tensors[0]

    ranks, group_offsets, offset = _prepare_collective_groups(process_group_so)

    # We're going to assume SPMD where for every rank group the root_rank is
    # the same relative to others
    relative_root_rank = root_rank - offset

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the broadcast on them
        group_ranks = [group_offset + r for r in ranks]
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue

        source_rank = group_offset + relative_root_rank
        source_tensor = tensor._local_tensors[source_rank]

        # Broadcast the source tensor to all ranks in this group
        for rank in group_ranks:
            if source_rank != rank:
                tensor._local_tensors[rank].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (tensors, work_so)


def _local_reduce(
    reduce_op: ReduceOp | str,
    tensors: list[torch.Tensor],
) -> torch.Tensor:
    if reduce_op == ReduceOp.SUM or reduce_op == "sum":
        op = operator.add
    elif reduce_op == ReduceOp.AVG or reduce_op == "avg":
        op = None
    elif reduce_op == ReduceOp.PRODUCT or reduce_op == "product":
        op = operator.mul
    elif reduce_op == ReduceOp.MIN or reduce_op == "min":
        op = torch.minimum
    elif reduce_op == ReduceOp.MAX or reduce_op == "max":
        op = torch.maximum
    elif reduce_op == ReduceOp.BAND or reduce_op == "band":
        op = torch.bitwise_and
    elif reduce_op == ReduceOp.BOR or reduce_op == "bor":
        op = torch.bitwise_or
    elif reduce_op == ReduceOp.BXOR or reduce_op == "bxor":
        op = torch.bitwise_xor
    elif reduce_op == ReduceOp.PREMUL_SUM or reduce_op == "premul_sum":
        raise NotImplementedError("PREMUL_SUM: need to add binding for scaling factor")
    else:
        raise NotImplementedError(f"ReduceOp {reduce_op} not implemented")

    if reduce_op == ReduceOp.AVG or reduce_op == "avg":
        return functools.reduce(operator.add, tensors) / len(tensors)
    else:
        if op is None:
            raise AssertionError
        return functools.reduce(op, tensors)


def _local_all_reduce_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    sparse_indices: torch.Tensor | None = None,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # "allreduce_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "__torch__.torch.classes.c10d.ReduceOp reduce_op, Tensor? sparse_indices, bool async_op=True, "
    # "int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)");
    from . import LocalTensor

    if len(tensors) != 1:
        raise AssertionError
    tensor = tensors[0]
    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on them
        group_ranks = [group_offset + r for r in ranks]
        if not all(rank in tensor._local_tensors for rank in group_ranks):
            continue

        # Collect tensors from the specified ranks in this group
        group_tensors = []
        for rank in group_ranks:
            group_tensors.append(tensor._local_tensors[rank])

        # Perform the reduction operation
        reduced_tensor = _local_reduce(reduce_op, group_tensors)

        # Update all tensors in the group with the reduced result
        for rank in group_ranks:
            tensor._local_tensors[rank].copy_(reduced_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (tensors, work_so)


def _local_allreduce_coalesced_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # "allreduce_coalesced_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "__torch__.torch.classes.c10d.ReduceOp reduce_op, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work"
    from . import LocalTensor

    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on all tensors together
        group_ranks = [group_offset + r for r in ranks]

        # For each tensor, perform the reduction operation
        for tensor in tensors:
            if not isinstance(tensor, LocalTensor):
                raise AssertionError("Input tensor must be a LocalTensor")
            if not all(rank in tensor._local_tensors for rank in group_ranks):
                continue
            # Collect tensors from the specified ranks in this group
            group_tensors = []
            for rank in group_ranks:
                group_tensors.append(tensor._local_tensors[rank])

            # Perform the reduction operation
            reduced_tensor = _local_reduce(reduce_op, group_tensors)

            # Update all tensors in the group with the reduced result
            for rank in group_ranks:
                tensor._local_tensors[rank].copy_(reduced_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_reduce_scatter_tensor_coalesced_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # "reduce_scatter_tensor_coalesced_(Tensor[] outputs, Tensor[] inputs, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "__torch__.torch.classes.c10d.ReduceOp reduce_op, bool async_op=True, "
    # "int timeout=-1) -> __torch__.torch.classes.c10d.Work"

    from . import LocalTensor

    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allreduce on all tensors together
        group_ranks = [group_offset + r for r in ranks]

        # For each tensor, perform the reduction operation
        for input_tensor, output_tensor in zip(input_tensors, output_tensors):
            if not isinstance(input_tensor, LocalTensor):
                raise AssertionError("Input tensor must be a LocalTensor")
            if not isinstance(output_tensor, LocalTensor):
                raise AssertionError("Output tensor must be a LocalTensor")
            if not all(rank in input_tensor._local_tensors for rank in group_ranks):
                continue
            if not all(rank in output_tensor._local_tensors for rank in group_ranks):
                continue

            # Collect tensors from the specified ranks in this group
            group_inputs = []
            for rank in group_ranks:
                group_inputs.append(input_tensor._local_tensors[rank])

            # Perform the reduction operation
            reduced_input = _local_reduce(reduce_op, group_inputs)

            reduced_input_splits = torch.split(
                reduced_input, reduced_input.size(0) // len(group_ranks), dim=0
            )

            # Update all tensors in the group with the reduced result
            for i, rank in enumerate(group_ranks):
                output_tensor._local_tensors[rank].copy_(reduced_input_splits[i])

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_allgather_base_(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    process_group_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[torch.Tensor, ScriptObject]:
    # "_allgather_base_(Tensor output_tensor, Tensor input_tensor, __torch__.torch.classes.c10d.ProcessGroup
    # process_group, bool async_op=True, int timeout=-1) -> (Tensor, __torch__.torch.classes.c10d.Work)");
    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    if not isinstance(output_tensor, LocalTensor):
        raise AssertionError("Output tensor must be a LocalTensor")
    if not isinstance(input_tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]

        if not all(rank in input_tensor._local_tensors for rank in group_ranks):
            continue
        if not all(rank in output_tensor._local_tensors for rank in group_ranks):
            continue

        gathered_tensors = []
        for rank_i in group_ranks:
            gathered_tensors.append(input_tensor._local_tensors[rank_i])

        gathered_tensor = torch.cat(gathered_tensors, dim=0)

        for rank_i in group_ranks:
            output_tensor._local_tensors[rank_i].copy_(gathered_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return output_tensor, work_so


def _local_reduce_scatter_base_(  # type: ignore[no-untyped-def]
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    process_group_so: ScriptObject,
    reduce_op_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[torch.Tensor, ScriptObject]:
    # "_reduce_scatter_base_(Tensor output_tensor, Tensor input_tensor,
    # __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op,
    # bool async_op=True, int timeout=-1) -> (Tensor, __torch__.torch.classes.c10d.Work)"

    from . import LocalTensor

    reduce_op = reduce_op_so.op()  # type: ignore[attr-defined]
    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    if not isinstance(output_tensor, LocalTensor):
        raise AssertionError("Output tensor must be a LocalTensor")
    if not isinstance(input_tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")

    for group_offset in group_offsets:
        group_ranks = [group_offset + r for r in ranks]
        if not all(rank in input_tensor._local_tensors for rank in group_ranks):
            continue
        if not all(rank in output_tensor._local_tensors for rank in group_ranks):
            continue

        gathered_tensors = []
        for rank_i in group_ranks:
            gathered_tensors.append(input_tensor._local_tensors[rank_i])

        reduced_tensor = _local_reduce(reduce_op, gathered_tensors)

        scattered_tensor = torch.split(
            reduced_tensor,
            reduced_tensor.size(0) // len(group_ranks),
            dim=0,
        )

        for i, rank_i in enumerate(group_ranks):
            output_tensor._local_tensors[rank_i].copy_(scattered_tensor[i].clone())

    work = FakeWork()
    work_so = Work.boxed(work)
    return output_tensor, work_so


def _local_all_gather_(
    output_tensors: list[list[torch.Tensor]],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[list[torch.Tensor]], ScriptObject]:
    # "allgather_(Tensor[][] output_tensors, Tensor[] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True, "
    # "int timeout=-1) -> (Tensor[][], __torch__.torch.classes.c10d.Work)");

    from . import LocalTensor

    if len(output_tensors) != 1:
        raise AssertionError
    if len(input_tensors) != 1:
        raise AssertionError

    input_tensor = input_tensors[0]
    # pyrefly: ignore [bad-assignment]
    output_tensors = output_tensors[0]

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    for i in range(len(output_tensors)):
        if not isinstance(output_tensors[i], LocalTensor):
            raise AssertionError("Output tensor must be a LocalTensor")

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the all_gather on them
        group_ranks = [group_offset + r for r in ranks]

        # For each rank in the group, gather from their input tensor
        for i, rank_i in enumerate(group_ranks):
            # allgather object happens to create pure tensor, so we special case it here
            source_tensor = input_tensor
            if isinstance(input_tensor, LocalTensor):
                source_tensor = input_tensor._local_tensors[rank_i]
            # pyrefly: ignore [missing-attribute]
            output_tensors[i].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    # pyrefly: ignore [bad-return]
    return ([output_tensors], work_so)


def _local_allgather_into_tensor_coalesced_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    async_op: bool = True,
) -> ScriptObject:
    # "allgather_into_tensor_coalesced_(Tensor[] outputs, Tensor[] inputs, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True) "
    # "-> __torch__.torch.classes.c10d.Work"
    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    # Each output tensor should be sized to hold all gathered inputs
    # outputs[i] will contain all inputs[i] from all ranks
    if len(output_tensors) != len(input_tensors):
        raise AssertionError(
            f"Number of outputs ({len(output_tensors)}) must match "
            f"number of inputs ({len(input_tensors)})"
        )

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the allgather_into_tensor on them
        group_ranks = [group_offset + r for r in ranks]

        # For each input/output pair
        for input_tensor, output_tensor in zip(input_tensors, output_tensors):
            if not isinstance(input_tensor, LocalTensor):
                raise AssertionError("Input tensor must be a LocalTensor")
            if not isinstance(output_tensor, LocalTensor):
                raise AssertionError("Output tensor must be a LocalTensor")

            if not all(rank in input_tensor._local_tensors for rank in group_ranks):
                continue
            if not all(rank in output_tensor._local_tensors for rank in group_ranks):
                continue

            # Gather input_tensor from all ranks into output_tensor
            # The output should be a concatenation of all inputs along the first dimension
            gathered_tensors = []
            for rank in group_ranks:
                gathered_tensors.append(input_tensor._local_tensors[rank])

            # Concatenate along first dimension and copy to output
            if gathered_tensors:
                concatenated = torch.cat(gathered_tensors, dim=0)
                for rank in group_ranks:
                    output_tensor._local_tensors[rank].copy_(concatenated)

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_gather_(
    output_tensors: list[list[torch.Tensor]],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    root_rank: int,
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # "gather_(Tensor[][] output_tensors, Tensor[] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, "
    # "bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work"
    raise NotImplementedError(
        "LocalTensor does not support MPMD operations like gather "
        "(only root rank receives data). Use SPMD collective operations like allgather instead."
    )


def _local_scatter_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[list[torch.Tensor]],
    process_group_so: ScriptObject,
    root_rank: int,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # "scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, "
    # "bool async_op=True, int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)");

    from . import LocalTensor

    if len(output_tensors) != 1:
        raise AssertionError
    if len(input_tensors) != 1:
        raise AssertionError
    output_tensor = output_tensors[0]
    # pyrefly: ignore [bad-assignment]
    input_tensors = input_tensors[0]

    ranks, group_offsets, offset = _prepare_collective_groups(process_group_so)

    # We're going to assume SPMD where for every rank group the root_rank is
    # the same relative to others
    relative_root_rank = root_rank - offset

    if not isinstance(output_tensor, LocalTensor):
        raise AssertionError("Output tensor must be a LocalTensor")
    if len(ranks) != len(input_tensors):
        raise AssertionError((ranks, input_tensors))

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the scatter on them
        group_ranks = [group_offset + r for r in ranks]
        if not all(rank in output_tensor._local_tensors for rank in group_ranks):
            continue

        # Root rank scatters its input tensors to all ranks in this group
        for i, rank in enumerate(group_ranks):
            input_tensor = input_tensors[i]
            if not isinstance(input_tensor, LocalTensor):
                raise AssertionError
            # Each rank i gets the i-th input tensor from the root
            source_tensor = input_tensor._local_tensors[
                group_offset + relative_root_rank
            ]
            output_tensor._local_tensors[rank].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (output_tensors, work_so)


def _local_alltoall_(
    output_tensors: list[torch.Tensor],
    input_tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    async_op: bool = True,
    timeout: int = -1,
) -> tuple[list[torch.Tensor], ScriptObject]:
    # "alltoall_(Tensor[] output_tensors, Tensor[] input_tensors, "
    # "__torch__.torch.classes.c10d.ProcessGroup process_group, bool async_op=True, "
    # "int timeout=-1) -> (Tensor[], __torch__.torch.classes.c10d.Work)";

    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    if not (len(input_tensors) == len(output_tensors) == len(ranks)):
        raise AssertionError(
            f"Number of input tensors ({len(input_tensors)}), "
            f"output tensors ({len(output_tensors)}), and ranks ({len(ranks)}) must match"
        )

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the alltoall on them
        group_ranks = [group_offset + r for r in ranks]

        # In alltoall, rank i sends input_tensors[j] to rank j and receives into output_tensors[i] from rank j
        for i, rank_i in enumerate(group_ranks):
            output_tensor = output_tensors[i]
            if not isinstance(output_tensor, LocalTensor):
                raise AssertionError("Output tensor must be a LocalTensor")

            if not all(rank in output_tensor._local_tensors for rank in group_ranks):
                continue

            for j, rank_j in enumerate(group_ranks):
                input_tensor = input_tensors[j]
                if not isinstance(input_tensor, LocalTensor):
                    raise AssertionError("Input tensor must be a LocalTensor")

                if not all(rank in input_tensor._local_tensors for rank in group_ranks):
                    continue

                # Rank i's j-th input tensor goes to rank j's i-th output tensor
                source_tensor = input_tensor._local_tensors[rank_i]
                output_tensor._local_tensors[rank_j].copy_(source_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return (output_tensors, work_so)


def _local_alltoall_base_(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    process_group_so: ScriptObject,
    output_split_sizes: list[int],
    input_split_sizes: list[int],
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # "alltoall_base_(Tensor output, Tensor input, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int[] output_split_sizes, int[] input_split_sizes, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work";

    from . import LocalTensor

    ranks, group_offsets, _offset = _prepare_collective_groups(process_group_so)

    if not isinstance(input_tensor, LocalTensor):
        raise AssertionError("Input tensor must be a LocalTensor")
    if not isinstance(output_tensor, LocalTensor):
        raise AssertionError("Output tensor must be a LocalTensor")
    # Convert split sizes to lists if they aren't already
    if output_split_sizes is not None:
        output_split_sizes = list(output_split_sizes)
    if input_split_sizes is not None:
        input_split_sizes = list(input_split_sizes)

    for group_offset in group_offsets:
        # For the tensors in this group [group_offset + r for r in ranks]
        # perform the alltoall_base on them
        group_ranks = [group_offset + r for r in ranks]

        if not all(rank in input_tensor._local_tensors for rank in group_ranks):
            continue
        if not all(rank in output_tensor._local_tensors for rank in group_ranks):
            continue

        for i, rank_i in enumerate(group_ranks):
            # Split input tensor from rank_i according to input_split_sizes
            rank_tensor = input_tensor._local_tensors[rank_i]

            if input_split_sizes is not None and len(input_split_sizes) > 0:
                # Split the input tensor
                input_splits = torch.split(rank_tensor, input_split_sizes, dim=0)
            else:
                # No split sizes specified, split evenly
                split_size = rank_tensor.size(0) // len(group_ranks)
                input_splits = torch.split(rank_tensor, split_size, dim=0)

            # Send each split to the corresponding rank
            for j, rank_j in enumerate(group_ranks):
                if j < len(input_splits):
                    split_tensor = input_splits[j]

                    # Determine where to place this split in the output tensor
                    if output_split_sizes is not None and len(output_split_sizes) > 0:
                        # Calculate offset based on output split sizes
                        output_offset = sum(output_split_sizes[:i]) if i > 0 else 0
                        end_offset = (
                            output_offset + output_split_sizes[i]
                            if i < len(output_split_sizes)
                            else output_tensor._local_tensors[rank_j].size(0)
                        )
                    else:
                        # No output split sizes, use even splits
                        split_size = output_tensor._local_tensors[rank_j].size(
                            0
                        ) // len(group_ranks)
                        output_offset = i * split_size
                        end_offset = min(
                            (i + 1) * split_size,
                            output_tensor._local_tensors[rank_j].size(0),
                        )

                    # Copy the split to the appropriate section of the output tensor
                    output_section = output_tensor._local_tensors[rank_j][
                        output_offset:end_offset
                    ]
                    if split_tensor.numel() != output_section.numel():
                        raise ValueError(
                            f"all_to_all_single: input split from rank {rank_i} to "
                            f"rank {rank_j} has {split_tensor.numel()} elements, but "
                            f"the corresponding output split has "
                            f"{output_section.numel()} elements"
                        )
                    if output_section.numel() > 0:
                        # Reshape split_tensor to match output_section if necessary
                        if split_tensor.size() != output_section.size():
                            split_tensor = split_tensor.view(output_section.size())
                        output_section.copy_(split_tensor)

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_barrier(
    tensor: torch.Tensor,
    process_group_so: ScriptObject,
    device_ids: list[int],
    async_op: bool = True,
    timeout: int = -1,
) -> ScriptObject:
    # "barrier(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int[] device_ids, bool async_op=True, int timeout=-1) -> __torch__.torch.classes.c10d.Work";

    from . import LocalTensor

    # Barrier is a synchronization primitive - in local simulation,
    # we don't need to do any actual work since all "ranks" are in the same process
    # Just validate that the tensor is a LocalTensor
    if not isinstance(tensor, LocalTensor):
        raise AssertionError

    # In a real distributed setting, barrier would synchronize all processes
    # In local simulation, this is essentially a no-op since all ranks are local
    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_monitored_barrier_(
    tensor: torch.Tensor,
    process_group_so: ScriptObject,
    device_ids: list[int],
    timeout: int,
    wait_all_ranks: bool,
) -> None:
    # "monitored_barrier_(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int[] device_ids, int timeout, bool wait_all_ranks) -> ()";

    from . import LocalTensor

    # Monitored barrier is a synchronization primitive with monitoring - in local simulation,
    # we don't need to do any actual work since all "ranks" are in the same process
    # Just validate that the tensor is a LocalTensor
    if not isinstance(tensor, LocalTensor):
        raise AssertionError

    # In a real distributed setting, monitored barrier would synchronize all processes
    # and provide monitoring capabilities. In local simulation, this is essentially a no-op
    # since all ranks are local and no actual synchronization is needed
    return


def _local_send(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    dst: int,
    tag: int,
) -> ScriptObject:
    # "send(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int dst, int tag) -> __torch__.torch.classes.c10d.Work";

    from . import LocalRunnerMode, LocalTensor

    if len(tensors) != 1:
        raise AssertionError
    tensor = tensors[0]

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a Tensor")
    src = int(tensor.__src_rank__)

    LocalRunnerMode.current()._signal_send(src, dst, tensor._local_tensors[src])

    work = FakeWork()
    work_so = Work.boxed(work)
    return work_so


def _local_recv_(
    tensors: list[torch.Tensor],
    process_group_so: ScriptObject,
    src: int,
    tag: int,
) -> ScriptObject:
    # "recv_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int src, int tag) -> __torch__.torch.classes.c10d.Work";
    from . import LocalRunnerMode, LocalTensor

    if len(tensors) != 1:
        raise AssertionError
    tensor = tensors[0]

    if not isinstance(tensor, LocalTensor):
        raise AssertionError("Input tensor must be a Tensor")
    dst = int(tensor.__src_rank__)

    def _recv_and_store(timeout: timedelta) -> bool:
        def _wait_and_store(obj: object) -> None:
            if not isinstance(obj, torch.Tensor):
                raise AssertionError("Expected to receive a Tensor")
            if not isinstance(tensor, LocalTensor):
                raise AssertionError("Input tensor must be a Tensor")
            tensor._local_tensors[dst] = obj

        LocalRunnerMode.current()._wait_recv(src, dst, _wait_and_store)
        return True

    work = PythonCallbackWork(_recv_and_store)
    work_so = Work.boxed(work)
    return work_so


def _local_recv_any_source_(
    tensors: list[torch.Tensor], process_group_so: ScriptObject, tag: int
) -> ScriptObject:
    # "recv_any_source_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, "
    # "int tag) -> __torch__.torch.classes.c10d.Work";

    return _local_recv_(tensors, process_group_so, -1, tag)


def _attach_rank(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Attaches rank as an attribute to given tensor so that the send or recv implementation
    knows which rank initiates the operation (note under local tensor mode ).
    """
    from torch.distributed.tensor import DTensor

    if isinstance(tensor, DTensor):
        tensor = tensor._local_tensor

    tensor.__src_rank__ = rank  # type: ignore[attr-defined]
    return tensor


def local_p2p_op(
    dst: torch.SymInt,
    tensor: torch.Tensor,
    op: Callable[[torch.Tensor, int], Work | None],
) -> Work | list[Work | None] | None:
    """
    Runs a point-to-point (P2P) operation for all combinations of source and destination ranks.
    """
    _check_op(op)

    from . import LocalIntNode

    if not isinstance(dst.node, LocalIntNode):
        raise AssertionError(
            "Expected 'dst' to be a LocalIntNode where the value is the "
            "destination rank and key is the source rank"
        )

    w = []
    for s, d in dst.node._local_ints.items():
        tensor = _attach_rank(tensor, s)
        w.append(op(tensor, d))
    return w


def wait_all(work: Work | list[Work | None] | None) -> None:
    """
    Waits for all work objects in the input to complete.

    A single Work object, None, or a list of Work objects (possibly containing None).
    If None, does nothing. If a single Work, waits for it to complete. If a list, waits
    for each non-None Work in the list to complete.
    """

    if work is None:
        return
    if isinstance(work, Work):
        work = [work]
    for w in work:
        if w is None:
            continue
        w.wait()
<|endoftext|>"""
A LocalTensor is a tensor subclass which simulates a tensor that is
distributed across SPMD ranks.  A LocalTensor might be size N, but in fact
there are world_size shards/replicas of it stored internally.  When you do a
plain PyTorch operation on it, we apply the operation to each shard; when you
do a collective, we do the mathematically equivalent operation on the local
shards.  A LocalTensor is associated with a list of ranks which specify
which ranks it holds local tensors for.

NB, this is NOT a DataParallel like abstraction where you can run operations
on multiple different GPUs. It is intended purely for *debugging* purposes,
the overhead is almost certainly too high to keep eight GPUs (even the C++
autograd needs multithreading to keep up!)  (It might potentially be possible
to trace through this with torch.compile and then compile it with CUDA graphs
but this is currently a non-goal.)

We do not directly handle MPMD. However in practice even in SPMD you may
encounter divergence in behavior per rank (for example, uneven sharding
across ranks). To support scenarios like this, we provide a helper decorator
that allows you to run a function with no side effects for each LocalTensor
shard and combine results back into LocalTensor or LocalIntNode.

NB: This is a torch dispatch Tensor subclass, as we want to assume that autograd
is SPMD, so we run it once, and dispatch the inner autograd calls to the individual
local shards.

NOTE ABOUT MESH:  This subclass requires collectives that are issued to it to
respect a DeviceMesh like abstraction.  The reason for this is that when
DTensor issues us a collective for a particular rank, you will be asked to do
this on a specific process group which involves some ranks.  However, this
will only be for the LOCAL PG that this particular rank is participating in;
there will be a bunch of other PGs for other nodes that you don't get to see.
We need to be able to reverse engineer all of the collectives that don't
involve the current local rank here to actually issue them.  This can be done
two ways: (1) looking at the participating local ranks in the PG and computing
the complement which specifies all the other collectives you have to run, or
(2) retrieving the device mesh axis corresponding to the PG for this rank, and
then running all the fibers for this.
"""

import contextlib
import copy
import functools
import importlib
import operator
import os
import sys
import threading
from ast import Call
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from types import TracebackType
from typing import Any, Optional, ParamSpec, TypeVar, Union


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

import torch
import torch.distributed as dist
from torch import Size, SymBool, SymInt, Tensor
from torch._C import DispatchKey, DispatchKeySet, ScriptObject
from torch._export.wrappers import mark_subclass_constructor_exportable_experimental
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.distributed.distributed_c10d import _get_default_group
from torch.fx.experimental._constant_symnode import ConstantIntNode
from torch.nested._internal.nested_int import NestedIntNode
from torch.utils import _pytree as pytree
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode_stack,
    return_and_correct_aliasing,
    TorchDispatchMode,
)
from torch.utils.checkpoint import get_device_states, set_device_states


_R = TypeVar("_R")
_P = ParamSpec("_P")

not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")


from . import _c10d


def _is_in_fake_tensor_mode() -> bool:
    return any(
        isinstance(mode, FakeTensorMode) for mode in _get_current_dispatch_mode_stack()
    )


def _reduce_multidim_lists(
    lists_to_reduce: list[Any], reduce_func: Callable[[list[Any]], Any]
) -> Any:
    """
    Reduces a list of multi-dimensional lists, assuming they all have
    the exact same shape.

    Args:
        lists_to_reduce (list): A list where each item is a multi-dimensional
                                list (e.g., [md_list_1, md_list_2, ...]).
                                All inner md_lists must have the same shape.
        reduce_func (callable): A function that takes an iterable (list) of
                                values and returns a single reduced value.
                                For example: sum, max, min, or
                                lambda x: sum(x) / len(x) for mean.

    Returns:
        A single multi-dimensional list of the same shape as the inputs,
        where each value is the result of the reduce_func.

    Raises:
        ValueError: If the input list is empty or if shapes are inconsistent
                    (which may also raise IndexError or TypeError).
    """
    if not lists_to_reduce:
        raise ValueError("Input 'lists_to_reduce' cannot be empty.")

    # Get the first list to inspect its structure (shape)
    first_list = lists_to_reduce[0]

    # Check if the first element of this list is *also* a list.
    # This determines if we are at the base case or need to recurse.
    if isinstance(first_list[0], list):
        # --- RECURSIVE STEP ---
        # The elements are lists, so we need to go one level deeper.

        # We find the number of sub-lists from the first list.
        # (e.g., for [[1,2], [3,4]], this is 2)
        num_sublists = len(first_list)

        result = []
        # Iterate by the index of the sub-lists (e.g., i = 0, then i = 1)
        for i in range(num_sublists):
            # Build a new list to pass to the recursive call.
            # This list will contain the i-th sublist from *each* of the
            # input lists.
            # e.g., if lists_to_reduce = [ L1, L2 ] and i = 0,
            # this creates [ L1[0], L2[0] ]
            sublists_to_reduce = [l[i] for l in lists_to_reduce]

            # Recurse and append the result
            result.append(_reduce_multidim_lists(sublists_to_reduce, reduce_func))
        return result
    else:
        # --- BASE CASE ---
        # The elements are values (int, float, etc.), not lists.
        # We are at the innermost dimension.

        # Find the number of values in the innermost list.
        # (e.g., for [1, 2], this is 2)
        num_values = len(first_list)

        result = []
        # Iterate by the index of the values (e.g., i = 0, then i = 1)
        for i in range(num_values):
            # Get the values at this specific position (i) from *all*
            # input lists.
            # e.g., if lists_to_reduce = [ [1,2], [10,20] ] and i = 0,
            # this creates [ 1, 10 ]
            values_at_pos = [l[i] for l in lists_to_reduce]

            # Apply the user-provided reduction function to this list of values
            # and append the single result.
            result.append(reduce_func(values_at_pos))
        return result


def _is_inplace_op(op: OpOverload | Callable[..., Any]) -> bool:
    return (
        isinstance(op, OpOverload)
        # Not precise heuristic to detect inplace operation
        and op._schema.name[-1] == "_"
        # Strengthen the heuristic to check that the first argument and return value are a write
        and len(op._schema.arguments) > 0
        and op._schema.arguments[0].is_write
        and len(op._schema.returns) > 0
        and op._schema.returns[0].is_write
    )


def _int_on_rank(i: "int | LocalIntNode | ConstantIntNode", r: int) -> int:
    if isinstance(i, LocalIntNode):
        return i._local_ints[r]
    elif isinstance(i, ConstantIntNode):
        return i.val
    elif isinstance(i, int):
        return i
    else:
        raise AssertionError(type(i))


def _check_for_subclass(flat_args: Sequence[object]) -> bool:
    return any(_check_for_subclass_arg(x) for x in flat_args)


def _check_for_subclass_arg(x: object) -> bool:
    return (
        not isinstance(x, LocalTensor)
        and isinstance(x, Tensor)
        and type(x)
        not in (
            Tensor,
            FakeTensor,
            torch.nn.Parameter,
            torch.nn.Buffer,
        )
    )


def _map_to_rank_local_val(val: Any, rank: int) -> Any:
    if isinstance(val, LocalTensor):
        return val._local_tensors[rank]
    if isinstance(val, SymInt):
        if isinstance(val.node, LocalIntNode):
            return val.node._local_ints[rank]
        if isinstance(val.node, ConstantIntNode):
            return val.node.val
    return val


def _collect_accelerator_rng_states() -> dict[int, torch.Tensor]:
    """
    Collects RNG state from all available accelerator devices.

    Returns:
        List of RNG state tensors, one for each accelerator device.
        Returns empty list if accelerator is not available.
    """
    if not torch.accelerator.is_available():
        return {}

    if torch.accelerator.is_available():
        device_idx = torch.accelerator.current_device_index()
        with torch.accelerator.device_index(device_idx):
            return {device_idx: torch.get_device_module().get_rng_state()}

    return {}


def _set_accelerator_rng_states(rng_states: dict[int, torch.Tensor]) -> None:
    """
    Sets RNG state for all accelerator devices from a list of states.

    Args:
        rng_states: List of RNG state tensors to restore.
    """
    if not torch.accelerator.is_available():
        return

    if torch.accelerator.is_available():
        for device_idx, device_rng_state in rng_states.items():
            with torch.accelerator.device_index(device_idx):
                torch.get_device_module().set_rng_state(device_rng_state)


def _get_rng_state() -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """
    Gets CPU and accelerator (e.g., CUDA, XPU device) rng states from all devices.
    """
    return (torch.get_rng_state(), _collect_accelerator_rng_states())


def _set_rng_state(
    cpu_state: torch.Tensor, accelerator_states: dict[int, torch.Tensor]
) -> None:
    """
    Sets CPU and accelerator (e.g., CUDA, XPU device) rng states for all devices. If
    the list of accelerator states is shorter than the number of devices only the
    first len(accelerator_states) devices will get their rng state set.
    """
    torch.set_rng_state(cpu_state)
    _set_accelerator_rng_states(accelerator_states)


def _combine_int_rank_results(rank_results: dict[int, int]) -> int | torch.SymInt:
    any_v = next(iter(rank_results.values()))

    if all(v == any_v for v in rank_results.values()):
        return any_v

    return torch.SymInt(LocalIntNode(rank_results))


def _combine_any_rank_results(rank_results: dict[int, Any]) -> Any:
    any_v = next(iter(rank_results.values()))

    if isinstance(any_v, Tensor):
        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        return LocalTensor(rank_results)

    if isinstance(any_v, int):
        return _combine_int_rank_results(rank_results)

    if isinstance(any_v, torch.device):
        if not all(v.type == any_v.type for v in rank_results.values()):
            raise AssertionError("device type should be the same")
        # Just use the first device - the device type is what matters,
        # and LocalTensorMode runs on a single physical device anyway
        return any_v

    if not all(v == any_v for v in rank_results.values()):
        raise AssertionError(
            "Non Tensor or int rank results must be equal for all ranks"
        )

    return any_v


def _combine_rank_results(rank_results: dict[int, Any], default: Any | None) -> Any:
    rank_ids = rank_results.keys()
    rank_value = rank_results[next(iter(rank_ids))]

    if isinstance(rank_value, (list, tuple)):
        max_rank_result_len = max(len(v) for v in rank_results.values())
        ret_list = []
        for i in range(max_rank_result_len):
            rank_col_results = {
                r: v[i] if i < len(v) else default for r, v in rank_results.items()
            }
            ret_list.append(_combine_any_rank_results(rank_col_results))
        return type(rank_value)(ret_list)
    else:
        return _combine_any_rank_results(rank_results)


def _zero_sized_like(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    tensor_size = list(tensor.size())
    tensor_size[dim] = 0
    empty_tensor = torch.empty(*tensor_size, dtype=tensor.dtype, device=tensor.device)
    return empty_tensor


def _for_each_rank_run_func(
    func: OpOverload | Callable[..., Any],
    ranks: frozenset[int],
    args: Sequence[Any],
    kwargs: dict[str, Any],
    *,
    alias: bool = True,
) -> Any:
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    flat_args = [
        a.wait() if isinstance(a, AsyncCollectiveTensor) else a for a in flat_args
    ]

    lm = enabled_local_tensor_mode()
    use_per_rank_rng = lm is not None and len(lm._per_rank_rng_states) > 0

    global_rng_state = None if use_per_rank_rng else _get_rng_state()

    flat_rank_rets = {}

    default_value: Tensor | None = None
    for r in sorted(ranks):
        if use_per_rank_rng:
            if lm is None:
                raise AssertionError
            if r in lm._per_rank_rng_states:
                _set_rng_state(*lm._per_rank_rng_states[r])
        else:
            if global_rng_state is None:
                raise AssertionError
            _set_rng_state(*global_rng_state)

        rank_flat_args = [_map_to_rank_local_val(a, r) for a in flat_args]
        rank_args, rank_kwargs = pytree.tree_unflatten(rank_flat_args, args_spec)
        if func is torch.ops.aten.hash_tensor.default and rank_args[0].numel() == 0:
            # Special case for empty tensors, hash_tensor returns an empty tensor
            rank_ret = torch.empty(0, dtype=torch.uint64, device=rank_args[0].device)
        else:
            rank_ret = func(*rank_args, **rank_kwargs)
        flat_rank_rets[r] = rank_ret

        if use_per_rank_rng:
            if lm is None:
                raise AssertionError
            lm._per_rank_rng_states[r] = _get_rng_state()

        if default_value is None and func is torch.ops.aten.split.Tensor:
            # If split happens over the dimension smaller than the number of chunks
            # it is possible that some ranks will produce shorter lists of chunks.
            # In order to make the result across all ranks of the same length we
            # append empty tensors (zero size on the split dimension).
            tensor = rank_flat_args[0]
            split_dim = 0 if len(rank_flat_args) < 3 else rank_flat_args[2]
            default_value = _zero_sized_like(tensor, split_dim)

    if _is_inplace_op(func):
        alias = False
        # For the in-place ops return self
        ret = args[0]
        if isinstance(func, OpOverload) and torch.Tag.inplace_view in func.tags:
            # Ensure that wrapper tensor size is synchronized with its local tensors
            ret._sync_meta()
    else:
        ret = _combine_rank_results(flat_rank_rets, default_value)

    if alias:
        return return_and_correct_aliasing(func, args, kwargs, ret)
    else:
        return ret


def _get_extra_dispatch_keys(t: torch.Tensor) -> DispatchKeySet:
    extra_dispatch_keys = torch._C.DispatchKeySet.from_raw_repr(0)
    if torch._C._dispatch_keys(t).has(torch._C.DispatchKey.Conjugate):
        extra_dispatch_keys = extra_dispatch_keys.add(torch._C.DispatchKey.Conjugate)
    if torch._C._dispatch_keys(t).has(torch._C.DispatchKey.Negative):
        extra_dispatch_keys = extra_dispatch_keys.add(torch._C.DispatchKey.Negative)
    return extra_dispatch_keys


class LocalIntNode:
    """
    Like a LocalTensor, but for an int.  We can't use a 0D tensor to represent this
    because often only a SymInt is accepted where we wish to use this.
    """

    def __new__(cls, local_ints: dict[int, int]) -> "ConstantIntNode | LocalIntNode":  # type: ignore[misc]
        if len(set(local_ints.values())) == 1:
            return ConstantIntNode(next(iter(local_ints.values())))
        return super().__new__(cls)

    def __init__(self, local_ints: dict[int, int]):
        self._local_ints = local_ints

    def maybe_as_int(self) -> int | None:
        return None

    def is_int(self) -> bool:
        return True

    def is_float(self) -> bool:
        return False

    def is_bool(self) -> bool:
        return False

    def is_nested_int(self) -> bool:
        return False

    def clone(self) -> "LocalIntNode":
        return self

    def _str(self) -> str:
        return f"LocalIntNode({self._local_ints})"

    def __str__(self) -> str:
        return self._str()

    def __repr__(self) -> str:
        return self._str()

    def _graph_repr(self) -> str:
        return self._str()

    def is_symbolic(self) -> bool:
        return False

    def is_constant(self) -> bool:
        return False

    def sym_max(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {
                r: max(self._local_ints[r], _int_on_rank(other, r))
                for r in self._local_ints
            }
        )

    def sym_min(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {
                r: min(self._local_ints[r], _int_on_rank(other, r))
                for r in self._local_ints
            }
        )

    def sym_sum(self, other: Sequence[Any]) -> "LocalIntNode | ConstantIntNode":
        t = LocalIntNode(dict.fromkeys(self._local_ints, 0))
        for o in other:
            t = t.add(o)
        return t

    def neg(self) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode({r: -self._local_ints[r] for r in self._local_ints})

    def add(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] + _int_on_rank(other, r) for r in self._local_ints}
        )

    def sub(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] - _int_on_rank(other, r) for r in self._local_ints}
        )

    def mul(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] * _int_on_rank(other, r) for r in self._local_ints}
        )

    def floordiv(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] // _int_on_rank(other, r) for r in self._local_ints}
        )

    def mod(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] % _int_on_rank(other, r) for r in self._local_ints}
        )

    def int_floordiv(
        self, other: "int | LocalIntNode | ConstantIntNode"
    ) -> "LocalIntNode | ConstantIntNode":
        return LocalIntNode(
            {r: self._local_ints[r] // _int_on_rank(other, r) for r in self._local_ints}
        )

    def eq(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] == _int_on_rank(other, r) for r in self._local_ints}
        return torch._C._get_constant_bool_symnode(len(r) == 1 and next(iter(r)))

    def ne(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] != _int_on_rank(other, r) for r in self._local_ints}
        return torch._C._get_constant_bool_symnode(len(r) > 1 or next(iter(r)))

    def ge(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] >= _int_on_rank(other, r) for r in self._local_ints}
        if len(r) != 1:
            raise AssertionError((self, other))
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def le(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] <= _int_on_rank(other, r) for r in self._local_ints}
        if len(r) != 1:
            raise AssertionError((self, other))
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def gt(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] > _int_on_rank(other, r) for r in self._local_ints}
        if len(r) != 1:
            raise AssertionError((self, other))
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def lt(self, other: "int | LocalIntNode | ConstantIntNode") -> bool | SymBool:
        r = {self._local_ints[r] < _int_on_rank(other, r) for r in self._local_ints}
        if len(r) != 1:
            raise AssertionError((self, other))
        return torch._C._get_constant_bool_symnode(next(iter(r)))

    def wrap_int(self, num: int) -> "LocalIntNode | ConstantIntNode":
        return ConstantIntNode(num)


class _LocalDeviceHandle:
    """
    Wrapper around device module (e.g., torch.cuda) with automatic LocalTensor semantics.

    This class wraps device modules and automatically handles per-rank operations in
    LocalTensor mode:
    - get_rng_state() returns a LocalTensor with per-rank states
    - set_rng_state(LocalTensor) sets per-rank states

    When not in LocalTensor mode, it delegates directly to the underlying device handle.
    """

    def __init__(self, device_handle, device_type: str):
        """
        Initialize the local device handle wrapper.

        Args:
            device_handle: The underlying device module (e.g., torch.cuda)
            device_type: Device type string (e.g., "cuda", "cpu")
        """
        self._device_handle = device_handle
        self._device_type = device_type

    def get_rng_state(self):
        """
        Get RNG state, automatically returning LocalTensor in LocalTensor mode.

        Returns:
            LocalTensor in LocalTensor mode, regular Tensor otherwise
        """
        lm = enabled_local_tensor_mode()
        if not lm:
            return self._device_handle.get_rng_state()

        original_state = _get_rng_state()
        per_rank_states = {}

        try:
            for rank in lm.ranks:
                # We need to set-then-get instead of directly copying lm._per_rank_rng_states[rank]
                # because they have different structures:
                # - lm._per_rank_rng_states[rank] is a tuple: (cpu_state, {device_idx: cuda_state})
                # - self._device_handle.get_rng_state() returns just the device-specific tensor
                # So we temporarily restore the full RNG state (CPU + all CUDA devices) for this rank,
                # then extract only the specific device's state tensor that we need.
                if rank in lm._per_rank_rng_states:
                    _set_rng_state(*lm._per_rank_rng_states[rank])

                per_rank_states[rank] = self._device_handle.get_rng_state()
        finally:
            _set_rng_state(*original_state)

        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        return LocalTensor(per_rank_states)

    def set_rng_state(self, state):
        """
        Set RNG state, automatically handling LocalTensor input.

        Args:
            state: Regular Tensor or LocalTensor with per-rank states
        """
        if isinstance(state, LocalTensor):
            lm = enabled_local_tensor_mode()
            if lm is None:
                raise AssertionError

            # Similar to get_rng_state but in reverse: we need to convert from
            # device-specific tensor format to full state tuple format.
            # - state._local_tensors[rank] contains just the device-specific RNG state tensor
            # - lm._per_rank_rng_states[rank] needs a tuple: (cpu_state, {device_idx: cuda_state})
            # So we set the device's state with the rank-specific tensor, then _get_rng_state()
            # captures both CPU and CUDA states into the tuple format that _per_rank_rng_states expects.
            for rank, rank_state in state._local_tensors.items():
                self._device_handle.set_rng_state(rank_state.to("cpu"))
                lm._per_rank_rng_states[rank] = _get_rng_state()
        else:
            self._device_handle.set_rng_state(state.to("cpu"))

    def __getattr__(self, name):
        """Delegate all other attributes to the underlying device module."""
        return getattr(self._device_handle, name)


class _LocalOffsetBasedRNGTracker:
    """
    LocalTensor-specific RNG tracker for DTensor random operations.

    This class manages per-rank RNG states when running in LocalTensor mode,
    using _LocalPhiloxState to track different offsets for each virtual rank.
    It is instantiated and used by OffsetBasedRNGTracker when in LocalTensor mode.

    Much of this is derived from OffsetBasedRNGTracker:
    https://github.com/pytorch/pytorch/blob/402c46503002f98ccfc023a733081fb0719223a1/torch/distributed/tensor/_random.py#L182
    """

    def __init__(self, device_type: str = "cuda"):
        """Initialize the LocalTensor RNG tracker."""
        from torch.distributed.device_mesh import _get_device_handle

        self._device_type = device_type
        self._device_handle = _LocalDeviceHandle(
            _get_device_handle(device_type), device_type
        )
        self.distribute_region_enabled = True
        self._device_mesh = None

    @property
    def _device(self):
        return torch.device(self._device_type, torch.accelerator.current_device_index())

    def _set_pre_op_offset(self, state, spec) -> None:
        """Compute and set per-rank offsets before the random operation."""
        from torch.distributed.tensor._ops.utils import prod
        from torch.distributed.tensor._utils import (
            _compute_local_shape_and_global_offset,
        )
        from torch.distributed.tensor.placement_types import Shard

        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError

        state._per_rank_offsets = {}

        for rank in lm.ranks:
            # compute this rank's coordinate in the mesh
            mesh_coords = []
            for mesh_dim_idx in range(spec.mesh.ndim):
                mesh_dim_size = spec.mesh.size(mesh_dim_idx)
                # calculate rank's coordinate in this mesh dimension
                num_chunks_after = 1
                for j in range(mesh_dim_idx + 1, spec.mesh.ndim):
                    num_chunks_after *= spec.mesh.size(j)
                coord = (rank // num_chunks_after) % mesh_dim_size
                mesh_coords.append(coord)

            # compute shard offset based on placements
            from torch.distributed.tensor._random import (
                _calc_first_shard_size,
                _calc_shard_info,
                _calc_shard_linear_idx,
            )

            # Compute shard index and total number of shards on each tensor dim
            shard_idx_by_dim, total_num_shards_by_dim = _calc_shard_info(
                mesh_coords, spec
            )

            # compute shard linear index
            shard_linear_idx = _calc_shard_linear_idx(
                shard_idx_by_dim, total_num_shards_by_dim
            )

            # get current offset for this rank
            current_offset = int(
                state._per_rank_states[rank][8:].view(dtype=torch.int64).item()
            )

            local_shape = _calc_first_shard_size(spec)
            # compute local size
            local_size = prod(local_shape)

            # compute new offset (must be multiple of 4)
            offset_incr = (shard_linear_idx * local_size + 3) // 4 * 4
            state._per_rank_offsets[rank] = current_offset + offset_incr

    def _set_post_op_offset(self, state, spec, old_offset) -> None:
        """Set per-rank offsets after the random operation."""
        from torch.distributed.tensor._ops.utils import prod

        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError

        dtensor_shape = spec.shape
        numel = prod(dtensor_shape)
        # offset must be multiple of 4
        numel = (numel + 3) // 4 * 4

        if not hasattr(state, "_per_rank_offsets"):
            state._per_rank_offsets = {}

        # handle LocalIntNode old_offset (different values per rank)
        if isinstance(old_offset, SymInt) and isinstance(old_offset.node, LocalIntNode):
            for rank in lm.ranks:
                rank_old_offset = old_offset.node._local_ints[rank]
                state._per_rank_offsets[rank] = rank_old_offset + numel
        else:
            # same old_offset for all ranks
            old_offset_int = (
                int(old_offset) if isinstance(old_offset, SymInt) else old_offset
            )
            for rank in lm.ranks:
                state._per_rank_offsets[rank] = old_offset_int + numel

    @contextlib.contextmanager
    def _distribute_region(self, spec, generator=None):
        """Context manager for LocalTensor mode distribute region."""
        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError

        # get base state
        if generator is not None:
            base_state_tensor = generator.get_state()
            per_rank_states = {rank: base_state_tensor.clone() for rank in lm.ranks}
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            base_state_tensor = LocalTensor(per_rank_states)
        else:
            base_state_tensor = self._device_handle.get_rng_state()

        state = _LocalPhiloxState(base_state_tensor)

        if self.distribute_region_enabled:
            # sync to rank 0's state if no explicit generator
            if generator is None:
                any_rank_state = lm._any_local_rng_state()
                any_rank_cpu, any_rank_cuda = any_rank_state

                if self._device.type in {"cuda", "xpu"}:
                    if self._device.index not in any_rank_cuda:
                        raise AssertionError
                    any_rank_device_state = any_rank_cuda[self._device.index]
                else:
                    any_rank_device_state = any_rank_cpu

                from torch.distributed.tensor._random import _PhiloxState

                any_rank_philox = _PhiloxState(any_rank_device_state)
                state.seed = int(any_rank_philox.seed.item())
                state.offset = int(any_rank_philox.offset.item())

            old_offset = state.offset
            self._set_pre_op_offset(state, spec)
            state.apply_to_local_tensor_mode(self._device_handle)

            try:
                yield
            finally:
                self._set_post_op_offset(state, spec, old_offset)
                state.apply_to_local_tensor_mode(self._device_handle)
        else:
            yield

        # maybe reset generator to rank 0's state
        if generator is not None:
            rank_0_state = state._per_rank_states[0]
            generator.set_state(rank_0_state)


_LOCAL_TENSOR_ATTR_PREFIX = "_local_tensor_"


def _is_local_tensor_attr(attr: str) -> bool:
    return attr.startswith(_LOCAL_TENSOR_ATTR_PREFIX)


def _to_local_tensor_attr(rank: int) -> str:
    return f"{_LOCAL_TENSOR_ATTR_PREFIX}{rank}"


def _from_local_tensor_attr(attr: str) -> int:
    if not _is_local_tensor_attr(attr):
        raise AssertionError(f"Invalid local tensor attr {attr}")
    return int(attr[len(_LOCAL_TENSOR_ATTR_PREFIX) :])


def _all_elements_same(values: list[Any]) -> bool:
    if not values:
        return True
    first_value = values[0]
    return all(value == first_value for value in values)


def _compute_local_tensor_meta(
    local_tensors: dict[int, torch.Tensor],
) -> tuple[
    list[torch.SymInt | int],
    list[torch.SymInt | int],
    torch.device,
    torch.dtype,
    torch.layout,
    DispatchKeySet,
]:
    """
    Computes the meta information for a LocalTensor from its local tensors.
    """
    it = iter(local_tensors.values())
    first_local_tensor = next(it)

    first_shape = first_local_tensor.shape
    first_stride = first_local_tensor.stride()
    dtype = first_local_tensor.dtype
    device = first_local_tensor.device
    layout = first_local_tensor.layout

    extra_dispatch_keys = _get_extra_dispatch_keys(first_local_tensor)

    # Assert that all tensors have the same dtype, layout and dispatch keys. Due
    # to uneven sharding, it is possible that tensors will have different shapes.
    for local_tensor in it:
        if dtype != local_tensor.dtype:
            raise AssertionError(
                "Tensors representing LocalTensor shards must have the same dtype"
            )
        if layout != local_tensor.layout:
            raise AssertionError(
                "Tensors representing LocalTensor shards must have the same layout"
            )
        if extra_dispatch_keys != _get_extra_dispatch_keys(local_tensor):
            raise AssertionError(
                "Tensors representing LocalTensor shards must have the "
                "same set of extra dispatch keys"
            )

    # Compute shape/stride.  We allow for non-SPMD'ness here
    local_shapes: dict[int, dict[int, int]] = defaultdict(dict)  # dim => rank => size
    local_strides: dict[int, dict[int, int]] = defaultdict(dict)  # dim => rank => size
    for r, local_tensor in local_tensors.items():
        for d, size in enumerate(local_tensor.shape):
            local_shapes[d][r] = size
            local_strides[d][r] = local_tensor.stride(d)
    shape = [
        (
            first_shape[d]
            if _all_elements_same(list(local_shapes[d].values()))
            else torch.SymInt(LocalIntNode(local_shapes[d]))
        )
        for d in range(len(first_shape))
    ]
    strides = [
        (
            first_stride[d]
            if _all_elements_same(list(local_strides[d].values()))
            else torch.SymInt(LocalIntNode(local_strides[d]))
        )
        for d in range(len(first_shape))
    ]
    return shape, strides, device, dtype, layout, extra_dispatch_keys


class LocalTensor(torch.Tensor):
    """
    LocalTensor is a Tensor subclass that simulates a tensor distributed across multiple SPMD
    (Single Program, Multiple Data) ranks. Each LocalTensor instance internally holds a mapping from
    global rank ids to their corresponding local Tensor shards.Operations performed on a LocalTensor
    are applied independently to each local shard, mimicking distributed computation. Collectives
    and other distributed operations are handled by mapping them to the local shards as appropriate.

    Note:
        This class is primarily intended for debugging and simulating distributed tensor computations
        on a single process.

    """

    # Map from global rank to the local tensor.
    _local_tensors: dict[int, torch.Tensor]
    # Precomputed for speed set of keys from the local tensor map.
    _ranks: frozenset[int]
    _size: list[torch.SymInt | int]
    __slots__ = ["_local_tensors", "_ranks", "_size"]

    @staticmethod
    @torch._disable_dynamo
    def __new__(
        cls,
        local_tensors: dict[int, torch.Tensor],
        requires_grad: bool = False,
    ) -> "LocalTensor":
        if any(t.requires_grad for t in local_tensors.values()):
            raise AssertionError(
                "Internal local_tensors require grad, but we will ignore those autograd graph. "
                "Make a custom autograd function and make sure you detach the inner tensors."
            )

        if len(local_tensors) == 0:
            raise ValueError("LocalTensor cannot be empty!")

        (shape, strides, device, dtype, layout, extra_dispatch_keys) = (
            _compute_local_tensor_meta(local_tensors)
        )

        r = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            strides=strides,
            dtype=dtype,
            device=device,
            layout=layout,
            # In place ops potentially change local tensor sizes (e.g. resize_). While
            # executing an in-place op the return value must be the same as "self" input
            # otherwise we can introduce errors due to tensor identity changes. Hence we
            # need to be able to update wrapper subclass sizes after in-place ops. This
            # dispatch policy allows us to do that.
            dispatch_sizes_strides_policy="sizes",
            requires_grad=requires_grad,
            _extra_dispatch_keys=extra_dispatch_keys,
        )
        # The wrapper has no real storage (data_ptr()=0). Prevent callers
        # (e.g. Triton kernels) from silently reading the null pointer —
        # turn it into a clear RuntimeError instead of a CUDA IMA.
        torch._C._set_throw_on_mutable_data_ptr(r)

        local_tensors = {
            r: v if not isinstance(v, AsyncCollectiveTensor) else v.wait()
            for r, v in local_tensors.items()
        }
        r._local_tensors = local_tensors
        r._ranks = frozenset(local_tensors.keys())
        r._size = shape
        return r

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental  # type: ignore[misc]
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()

    def __deepcopy__(self, memo: dict[Any, Any] | None) -> "LocalTensor":
        local_tensors_copy = {
            r: copy.deepcopy(t, memo) for r, t in self._local_tensors.items()
        }
        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        return LocalTensor(local_tensors_copy, self.requires_grad)

    def __repr__(self) -> str:  # type: ignore[override]
        parts = []
        for k, v in self._local_tensors.items():
            parts.append(f"  {k}: {v}")
        tensors_str = ",\n".join(parts)
        return f"LocalTensor(\n{tensors_str}\n)"

    def __getattr__(self, name: str) -> Any:
        if _is_local_tensor_attr(name):
            rank = _from_local_tensor_attr(name)
            if rank not in self._ranks:
                raise AttributeError(f"Local tensor has no knowledge of rank {rank}")
            return self._local_tensors[rank]
        return object.__getattribute__(self, name)

    def __tensor_flatten__(self) -> tuple[list[str], tuple[Any, ...]]:
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        local_tensor_attrs = [_to_local_tensor_attr(r) for r in self._ranks]
        return local_tensor_attrs, ()

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict[str, Any],
        flatten_spec: tuple[Any, ...],
        outer_size: torch.Size,
        outer_stride: tuple[int, ...],
    ) -> "LocalTensor":
        if flatten_spec is None:
            raise AssertionError(
                "Expecting spec to be not None from `__tensor_flatten__` return value!"
            )
        local_tensors = {
            _from_local_tensor_attr(a): t for a, t in inner_tensors.items()
        }
        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        return LocalTensor(local_tensors)

    @classmethod
    @torch._disable_dynamo
    def __torch_dispatch__(  # type: ignore[override]
        cls,
        func: Any,
        types: tuple[Any, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        # This is horribly inefficient
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        local_tensor = None
        for arg in flat_args:
            if isinstance(arg, LocalTensor):
                local_tensor = arg
                break

        if local_tensor is None:
            raise AssertionError("At least one of the arguments must be a LocalTensor")

        # Check for unrecognized tensor subclasses (but allow regular tensors and scalars)
        has_unrecognized_types = _check_for_subclass(flat_args)
        if has_unrecognized_types:
            unrecognized_types = [
                type(x) for x in flat_args if _check_for_subclass_arg(x)
            ]
            not_implemented_log.debug(
                "LocalTensor unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        with LocalTensorMode(local_tensor._ranks):
            return func(*args, **kwargs)

    def numpy(self, *, force: bool = False) -> Any:
        if HAS_NUMPY:
            return self.reconcile().numpy(force=force)
        else:
            raise RuntimeError("Numpy is not available")

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> torch.Tensor:
        return _LocalContiguous.apply(self, memory_format)

    def is_contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> bool:
        return all(
            t.is_contiguous(memory_format=memory_format)
            for t in self._local_tensors.values()
        )

    def tolist(self) -> list[Any]:
        """
        Try to reconcile, if successful convert to list, otherwise if dtype is integer,
        convert to list of local integers.
        """
        equal_obj = self._equal_local_tensors()
        if isinstance(equal_obj, torch.Tensor):
            return equal_obj.tolist()
        if isinstance(equal_obj, torch.Size):
            if not self.dtype.is_floating_point and not self.dtype.is_complex:
                ranks = sorted(self._ranks)
                local_lists = [self._local_tensors[r].tolist() for r in ranks]
                return _reduce_multidim_lists(
                    local_lists,
                    lambda values: torch.SymInt(
                        LocalIntNode(dict(zip(ranks, values, strict=True)))
                    ),
                )

        raise RuntimeError("Cannot convert local tensor to list")

    def reconcile(self) -> torch.Tensor:
        """
        Reconciles the LocalTensor into a single torch.Tensor by ensuring all local
        shards are identical and returning a detached clone of one of them.

        Note:
            This method is useful for extracting a representative tensor from a LocalTensor
            when all shards are expected to be the same, such as after a collective operation
            that synchronizes all ranks.
        """

        # Force all local tensor shards across ranks to be the same
        equal_obj = self._equal_local_tensors()
        if not isinstance(equal_obj, torch.Tensor):
            raise AssertionError("LocalTensor shards must be the same to reconcile")
        cl = equal_obj.clone().detach()
        cl.requires_grad_(self.requires_grad)
        return cl

    def _equal_local_tensors(self) -> torch.Tensor | torch.Size | None:
        it = iter(self._local_tensors.values())
        t1 = next(it)
        if all(t2.equal(t1) for t2 in it):
            return t1
        if all(t2.shape == t1.shape for t2 in it):
            return t1.shape
        return None

    def _sync_meta(self) -> None:
        with no_dispatch():
            (shape, strides, device, dtype, layout, extra_dispatch_keys) = (
                _compute_local_tensor_meta(self._local_tensors)
            )
            self._size = shape


class _LocalContiguous(torch.autograd.Function):
    """Autograd function for LocalTensor.contiguous() that preserves gradient flow."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        input: LocalTensor,
        memory_format: torch.memory_format,
    ) -> LocalTensor:
        # pyrefly: ignore [bad-argument-type]
        return LocalTensor(
            # pyrefly: ignore [bad-argument-count]
            {
                r: t.contiguous(memory_format=memory_format)
                for r, t in input._local_tensors.items()
            },
            input.requires_grad,
        )

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        return grad_output, None


# If set to `True` the LocalTensorMode stack will be created for the whole process,
# otherwise it will be created for each thread.
_PROCESS_MODE: bool = True
_PROCESS_LOCAL_TENSOR_MODE: list["LocalTensorMode"] = []
# When running under local runner each thread must create its own local tensor mode
# so that they do not interfere with each other.
_THREAD_LOCAL_TENSOR_MODE: threading.local = threading.local()


def get_local_tensor_mode_list() -> list["LocalTensorMode"]:
    global _PROCESS_MODE
    if _PROCESS_MODE:
        global _PROCESS_LOCAL_TENSOR_MODE
        return _PROCESS_LOCAL_TENSOR_MODE
    global _THREAD_LOCAL_TENSOR_MODE
    if not hasattr(_THREAD_LOCAL_TENSOR_MODE, "value"):
        _THREAD_LOCAL_TENSOR_MODE.value = []
    return _THREAD_LOCAL_TENSOR_MODE.value


# These methods are patched from DeviceMesh to the _LocalDeviceMesh versions.
_PATCHED_DEVICE_MESH_METHODS: Sequence[str] = (
    "get_coordinate",
    "get_local_rank",
    "get_rank",
    "_is_current_rank_part_of_mesh",
    "_sym_get_coordinate",
)

# These random functions are also patched.
_PATCHED_RANDOM_FUNCTIONS: Sequence[tuple[str, str]] = (
    ("torch.random.manual_seed", "torch_manual_seed"),
    ("torch.manual_seed", "torch_manual_seed"),
    ("torch.random.initial_seed", "torch_initial_seed"),
    ("torch.initial_seed", "torch_initial_seed"),
)

_PATCHED_DIST_FUNCTIONS: Sequence[str] = ("get_rank",)


class LocalTensorMode(TorchDispatchMode):
    """
    A TorchDispatchMode that simulates SPMD (Single Program, Multiple Data) execution
    for LocalTensor objects across a set of ranks.

    LocalTensorMode enables PyTorch operations to be transparently applied to each
    local shard of a LocalTensor, as if they were distributed across multiple ranks.
    When active, this mode intercepts tensor operations and dispatches them to each
    rank's local tensor, collecting and wrapping the results as LocalTensors. It also
    handles collective operations by mapping them to local implementations.

    This mode is primarily intended for debugging and simulating distributed tensor
    computations on a single process, rather than for high-performance distributed
    training. It maintains a stack of active modes, patches DeviceMesh coordinate
    resolution, and provides utilities for temporarily disabling the mode or mapping
    functions over ranks.
    """

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    # What ranks this local tensor mode is operating over
    def __init__(self, ranks: int | frozenset[int]):
        if isinstance(ranks, int):
            # assume is world size
            self.ranks = frozenset(range(ranks))
        else:
            if not isinstance(ranks, frozenset):
                raise AssertionError
            self.ranks = ranks
        self._disable = True
        # Used to store the patched DeviceMesh methods
        self._old_device_mesh_methods: dict[str, Callable[..., object]] | None = None
        # Used to store the patched "random" functions
        self._old_random_functions: dict[str, Callable[..., object]] = {}
        # Used to store the patched dist functions
        self._old_dist_functions: dict[str, Callable[..., object]] | None = None
        self._per_rank_rng_states: dict[
            int, tuple[torch.Tensor, dict[int, torch.Tensor]]
        ] = {}
        # Cache for get_coordinate results, keyed by the inputs used to compute them:
        # (ndim, flattened rank map, layout).
        # Protected by _coordinate_cache_lock for thread safety in MPMD contexts.
        self._coordinate_cache: dict[
            tuple[int, tuple, torch.distributed._mesh_layout._MeshLayout], list[SymInt]
        ] = {}
        self._coordinate_cache_lock = threading.Lock()

    def __enter__(self) -> "LocalTensorMode":
        get_local_tensor_mode_list().append(self)
        self.enable_()

        # _distribute_region will compute correct per-shard offsets
        # but we want all ranks to start with the same state
        if not _is_in_fake_tensor_mode():
            cpu_state, cuda_states = _get_rng_state()
            for rank in self.ranks:
                self._per_rank_rng_states[rank] = (
                    cpu_state.clone(),
                    {idx: state.clone() for idx, state in cuda_states.items()},
                )

        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        local_tensor_mode_list = get_local_tensor_mode_list()
        local_tensor_mode_list.pop()
        self.disable_()
        if len(local_tensor_mode_list) > 0:
            if local_tensor_mode_list[-1]._disable:
                local_tensor_mode_list[-1].disable_()
            else:
                local_tensor_mode_list[-1].enable_()
        super().__exit__(exc_type, exc_val, exc_tb)

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[Any, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        # Find all LocalTensor arguments to determine ranks
        local_tensors = [a for a in flat_args if isinstance(a, LocalTensor)]

        # Check for unrecognized tensor subclasses (but allow regular tensors and scalars)
        has_unrecognized_types = _check_for_subclass(flat_args)
        if has_unrecognized_types:
            unrecognized_types = [
                type(x) for x in flat_args if _check_for_subclass_arg(x)
            ]
            not_implemented_log.debug(
                "LocalTensorMode unrecognized subclass(es): %s", unrecognized_types
            )
            return NotImplemented

        # device_mesh::_runtime_compute_coordinate_on_dim is checked before the
        # no-LocalTensor early return because compile-on-one-rank calls it with
        # only a plain mesh-tensor argument, but the result must still vary per
        # simulated rank under LocalTensorMode.
        if (
            not self._disable
            and func.namespace == "device_mesh"
            and func is torch.ops.device_mesh._runtime_compute_coordinate_on_dim.default
        ):
            from torch.fx.operator_schemas import normalize_function

            normalized = normalize_function(
                func,
                args,
                kwargs,
                normalize_to_only_use_kwargs=True,
            )
            if normalized is None:
                raise AssertionError(
                    "LocalTensorMode could not normalize arguments for "
                    "device_mesh::_runtime_compute_coordinate_on_dim"
                )
            full_mesh = normalized.kwargs["full_mesh"]
            index = normalized.kwargs["index"]
            rank_results: dict[int, int] = {}
            for r in self.ranks:
                mesh_tensor = DeviceMesh._get_mesh_tensor_from_full_mesh(
                    full_mesh, current_rank=r
                )
                mesh_coords = DeviceMesh._compute_coordinates_from_mesh(mesh_tensor, r)
                if mesh_coords is None:
                    raise RuntimeError(
                        f"Rank {r} not found in mesh tensor for "
                        "_runtime_compute_coordinate_on_dim"
                    )
                rank_results[r] = mesh_coords[index]
            return _combine_int_rank_results(rank_results)

        # Factory functions convert into LocalTensor, so we don't have to
        # transmute a Tensor into a LocalTensor if mutation happens...
        # But if you do an operation on a Tensor, do NOT wrap it into a
        # LocalTensor.  This helps prevent accidents when you're doing Tensor
        # operations on the inner non-wrapped tensors.
        if not local_tensors:
            if self._disable or any(isinstance(a, Tensor) for a in flat_args):
                return func(*args, **kwargs)

        # For LocalTensors, verify they have compatible ranks
        for a in flat_args:
            if isinstance(a, LocalTensor):
                if a._ranks > self.ranks:
                    raise AssertionError(
                        f"Input LocalTensor {a} must be configured for a "
                        f"subset of the LocalTensorMode ranks {self.ranks}"
                    )

        if func.overloadpacket == torch.ops.aten.dim:
            return len(args[0]._size)
        if func.overloadpacket == torch.ops.aten.sym_size:
            return tuple(args[0]._size)

        if func.namespace == "c10d":
            if func is torch.ops.c10d.allreduce_.default:
                return _c10d._local_all_reduce_(*args, **kwargs)
            elif func is torch.ops.c10d.allreduce_coalesced_.default:
                return _c10d._local_allreduce_coalesced_(*args, **kwargs)
            elif func is torch.ops.c10d.reduce_scatter_tensor_coalesced_.default:
                return _c10d._local_reduce_scatter_tensor_coalesced_(*args, **kwargs)
            elif func is torch.ops.c10d.scatter_.default:
                return _c10d._local_scatter_(*args, **kwargs)
            elif func is torch.ops.c10d.broadcast_.default:
                return _c10d._local_broadcast_(*args, **kwargs)
            elif func is torch.ops.c10d.allgather_.default:
                return _c10d._local_all_gather_(*args, **kwargs)
            elif func is torch.ops.c10d.allgather_into_tensor_coalesced_.default:
                return _c10d._local_allgather_into_tensor_coalesced_(*args, **kwargs)
            elif func is torch.ops.c10d._allgather_base_.default:
                return _c10d._local_allgather_base_(*args, **kwargs)
            elif func is torch.ops.c10d._reduce_scatter_base_.default:
                return _c10d._local_reduce_scatter_base_(*args, **kwargs)
            elif func is torch.ops.c10d.gather_.default:
                return _c10d._local_gather_(*args, **kwargs)
            elif func is torch.ops.c10d.alltoall_.default:
                return _c10d._local_alltoall_(*args, **kwargs)
            elif func is torch.ops.c10d.alltoall_base_.default:
                return _c10d._local_alltoall_base_(*args, **kwargs)
            elif func is torch.ops.c10d.barrier.default:
                return _c10d._local_barrier(*args, **kwargs)
            elif func is torch.ops.c10d.monitored_barrier_.default:
                return _c10d._local_monitored_barrier_(*args, **kwargs)
            elif func is torch.ops.c10d.send.default:
                return _c10d._local_send(*args, **kwargs)
            elif func is torch.ops.c10d.recv_.default:
                return _c10d._local_recv_(*args, **kwargs)
            elif func is torch.ops.c10d.recv_any_source_.default:
                return _c10d._local_recv_any_source_(*args, **kwargs)
            raise NotImplementedError(f"{func} not implemented")

        if func.namespace == "_c10d_functional" or func.namespace == "_dtensor":
            if func is torch.ops._dtensor.shard_dim_alltoall.default:
                return _c10d._local_functional_shard_dim_alltoall(*args, **kwargs)
            elif func is torch.ops._c10d_functional.all_gather_into_tensor.default:
                return _c10d._local_functional_all_gather_into_tensor(*args, **kwargs)
            elif func is torch.ops._c10d_functional.reduce_scatter_tensor.default:
                return _c10d._local_functional_reduce_scatter_tensor(*args, **kwargs)
            elif func is torch.ops._c10d_functional.all_to_all_single.default:
                return _c10d._local_functional_all_to_all_single(*args, **kwargs)
            else:
                with LocalTensorMode(self.ranks):
                    return func._op_dk(
                        DispatchKey.CompositeExplicitAutograd, *args, **kwargs
                    )

        if func.namespace == "profiler":
            return func(*args, **kwargs)

        if func.namespace == "_c10d_functional_autograd":
            raise NotImplementedError(f"{func} not implemented")

        if func.namespace == "symm_mem":
            raise NotImplementedError(f"{func} not implemented")

        return _for_each_rank_run_func(func, self.ranks, args, kwargs, alias=True)

    def disable_(self):
        if self._disable:
            return

        self._unpatch_device_mesh()
        self._unpatch_random_functions()
        self._unpatch_dist()
        self._disable = True

    def enable_(self):
        if not self._disable:
            return

        self._patch_device_mesh()
        self._patch_random_functions()
        self._patch_dist()
        self._disable = False

    @contextlib.contextmanager
    def disable(self) -> Generator[None, None, None]:
        """
        Disables LocalTensorMode temporarily. Primarily is intended to be used to perform
        rank specific computations and merge results back before enabling LocalTensorMode back.
        """

        # don't unpatch again if already disabled
        if self._disable:
            try:
                yield
            finally:
                # re-disable if the yield messed
                # with the state
                self.disable_()
            return

        self.disable_()
        try:
            yield
        finally:
            self.enable_()

    def rank_map(self, cb: Callable[[int], Tensor]) -> LocalTensor:
        """
        Creates a LocalTensor instance by mapping rank id to its local shard.
        """

        with self.disable():
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            return LocalTensor({r: cb(r) for r in self.ranks})

    def tensor_map(
        self, tensor: LocalTensor, cb: Callable[[int, Tensor], Tensor | None]
    ) -> LocalTensor:
        """
        Creates a LocalTensor instance by mapping rank id to its local shard.
        """

        with self.disable():
            results = {}
            for r in self.ranks:
                if r in tensor._local_tensors:
                    m = cb(r, tensor._local_tensors[r])
                    if m is not None:
                        results[r] = m
            # pyrefly: ignore [bad-argument-type, bad-argument-count]
            return LocalTensor(results)

    def _any_local_rng_state(self) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        return self._per_rank_rng_states[next(iter(self.ranks))]

    def _patch_device_mesh(self) -> None:
        if self._old_device_mesh_methods is not None:
            raise AssertionError
        saved = {}
        for name in _PATCHED_DEVICE_MESH_METHODS:
            saved[name] = getattr(DeviceMesh, name)
            local = getattr(_LocalDeviceMesh, name)
            setattr(DeviceMesh, name, local)
        self._old_device_mesh_methods = saved

    def _unpatch_device_mesh(self) -> None:
        saved, self._old_device_mesh_methods = self._old_device_mesh_methods, None
        if saved is None:
            raise AssertionError
        for name, value in saved.items():
            setattr(DeviceMesh, name, value)

    def _patch_random_functions(self) -> None:
        # TODO: This should either be removed or documented why it's necessary.
        from torch.distributed.tensor import _random as dtensor_random

        for global_name, local_name in _PATCHED_RANDOM_FUNCTIONS:
            if global_name in self._old_random_functions:
                continue
            mod_name, attr_name = global_name.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            old = getattr(mod, attr_name)
            local = getattr(_LocalRandom, local_name)
            setattr(mod, attr_name, local)
            self._old_random_functions[global_name] = old

    def _unpatch_random_functions(self) -> None:
        # TODO: This should either be removed or documented why it's necessary.
        from torch.distributed.tensor import _random as dtensor_random

        for global_name, local_name in _PATCHED_RANDOM_FUNCTIONS:
            value = self._old_random_functions.pop(global_name, None)
            if value is not None:
                mod_name, attr_name = global_name.rsplit(".", 1)
                mod = importlib.import_module(mod_name)
                setattr(mod, attr_name, value)

    def _patch_dist(self) -> None:
        if self._old_dist_functions is not None:
            raise AssertionError
        import torch.distributed as dist

        saved = {}
        for name in _PATCHED_DIST_FUNCTIONS:
            saved[name] = getattr(dist, name)
            local = getattr(_LocalDist, name)
            setattr(dist, name, local)
        self._old_dist_functions = saved

    def _unpatch_dist(self) -> None:
        saved, self._old_dist_functions = self._old_dist_functions, None
        if saved is None:
            raise AssertionError
        import torch.distributed as dist

        for name, value in saved.items():
            setattr(dist, name, value)


class _LocalRandom:
    """
    Holds implementations of random functionality that must be patched while running
    under LocalTensorMode.
    """

    @staticmethod
    def torch_manual_seed(seed) -> torch._C.Generator:
        """LocalTensor-aware version of torch.random.manual_seed."""
        if (
            (lm := enabled_local_tensor_mode())
            and isinstance(seed, torch.SymInt)
            and isinstance(seed.node, LocalIntNode)
        ):
            from torch.random import _manual_seed_impl

            for rank in sorted(lm.ranks):
                rank_seed = seed.node._local_ints[rank]
                _manual_seed_impl(rank_seed)
                lm._per_rank_rng_states[rank] = _get_rng_state()
            return torch.random.default_generator
        from torch.random import _manual_seed_impl

        result = _manual_seed_impl(seed)

        if lm is not None and len(lm._per_rank_rng_states) > 0:
            cpu_state, cuda_states = _get_rng_state()
            for rank in lm.ranks:
                lm._per_rank_rng_states[rank] = (
                    cpu_state.clone(),
                    {idx: state.clone() for idx, state in cuda_states.items()},
                )

        return result

    @staticmethod
    def torch_initial_seed():
        """LocalTensor-aware version of torch.random.initial_seed."""
        if lm := enabled_local_tensor_mode():
            if len(lm._per_rank_rng_states) == 0:
                return torch.random.default_generator.initial_seed()
            rank_seeds = {}

            for rank in sorted(lm.ranks):
                _set_rng_state(*lm._per_rank_rng_states[rank])
                rank_seeds[rank] = torch.random.default_generator.initial_seed()

            local_int_node = LocalIntNode(rank_seeds)
            return torch.SymInt(local_int_node)

        return torch.random.default_generator.initial_seed()


# Save the original get_coordinate method before any patching


class _LocalDeviceMesh:
    """
    Holds implementations of DeviceMesh functionality that must be patched while running
    under LocalTensorMode.
    """

    @staticmethod
    def get_coordinate(self: DeviceMesh) -> list[SymInt] | None:
        # NB: In order to support submeshes the code below recreates for each
        # rank submesh with the same mesh dimensions as current mesh. We are
        # doing this because when submesh is created it is created for a particular
        # rank (therefore below we are patching get_rank method). We are trying to
        # limit the invasiveness of local tensor.
        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError("Unexpectedly not in LocalTensorMode")

        # Include all attributes used below in the cache key
        cache_key = (self.ndim, self._flatten_rank_map, self._layout)
        # Check cache first (fast path without lock)
        if cache_key in lm._coordinate_cache:
            return lm._coordinate_cache[cache_key]

        # Acquire lock for thread safety in MPMD contexts
        with lm._coordinate_cache_lock:
            # Double-check after acquiring lock
            if cache_key in lm._coordinate_cache:
                return lm._coordinate_cache[cache_key]

            coords: list[dict[int, int]] = [{} for _ in range(self.ndim)]
            # Clone rank_map to avoid "Cannot set version_counter for inference tensor"
            # error when running under torch.inference_mode()
            rank_map = self._rank_map.clone()
            for r in lm.ranks:
                rank_tensor = self._layout.remap_to_tensor(rank_map)
                rank_coords = (rank_tensor == r).nonzero().tolist()
                if len(rank_coords) != 1:
                    raise AssertionError
                for d, c in enumerate(rank_coords[0][1:]):
                    coords[d][r] = c

            out = [torch.SymInt(LocalIntNode(c)) for c in coords]
            # Cache the result
            lm._coordinate_cache[cache_key] = out
            # The output contains coordinates for each of the ranks with respect to
            # their meshes formed from root mesh and selecting the same dimensions
            # as the current mesh.
            return out  # type: ignore[return-value]

    @staticmethod
    def _is_current_rank_part_of_mesh(self: DeviceMesh) -> bool:
        my_coordinate = self.get_coordinate()
        return my_coordinate is not None

    @staticmethod
    def _sym_get_coordinate(self: DeviceMesh, index: int) -> int:
        my_coordinate = self.get_coordinate()
        if my_coordinate is None:
            raise AssertionError
        return my_coordinate[index]

    @staticmethod
    def get_rank(self) -> int | SymInt:
        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError("Unexpectedly not in LocalTensorMode")
        return torch.SymInt(LocalIntNode(local_ints={r: r for r in lm.ranks}))

    @staticmethod
    def get_local_rank(self, mesh_dim: int | str | None = None) -> int | SymInt:
        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError("Unexpectedly not in LocalTensorMode")

        if self.ndim > 1 and mesh_dim is None:
            raise RuntimeError(
                f"Found the DeviceMesh have {self.ndim} dimensions",
                "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
            )
        elif mesh_dim is None:
            mesh_dim = 0

        if isinstance(mesh_dim, str):
            mesh_dim = self._mesh_dim_names.index(mesh_dim)

        # Compute local rank for each global rank
        # get_coordinate returns a list of SymInt, one per mesh dimension
        # We need to extract the coordinate for the specified mesh_dim
        coords = _LocalDeviceMesh.get_coordinate(self)
        if coords is None:
            raise AssertionError
        return coords[mesh_dim]


class _LocalDist:
    """
    Holds implementations of torch.distributed functions that must be patched while
    running under LocalTensorMode.
    """

    @staticmethod
    def get_rank(group: "ProcessGroup | None" = None) -> "int | SymInt":
        import torch.distributed as dist

        lm = enabled_local_tensor_mode()
        if lm is None:
            raise AssertionError("Unexpectedly not in LocalTensorMode")
        if group is None or group is dist.GroupMember.WORLD:
            return torch.SymInt(LocalIntNode({r: r for r in lm.ranks}))
        from ._c10d import _prepare_collective_groups

        ranks, group_offsets, _ = _prepare_collective_groups(group)
        mapping: dict[int, int] = {}
        for group_offset in group_offsets:
            for local_rank, r in enumerate(ranks):
                mapping[group_offset + r] = local_rank
        return torch.SymInt(LocalIntNode(mapping))


def reconcile_args(args: Any, kwargs: dict[str, Any] | None = None) -> Any:
    """
    Reconciles arguments by converting any LocalTensor instances in the input
    arguments to their underlying torch.Tensor representation.

    This function is typically used to prepare arguments for functions that
    expect standard torch.Tensor objects, by flattening the input arguments,
    replacing LocalTensor instances with their reconciled (standard tensor)
    versions, and then reconstructing the original argument structure.

    Args:
        args: Positional arguments, possibly containing LocalTensor instances.
        kwargs: Keyword arguments, possibly containing LocalTensor instances.

    Returns:
        Any: The arguments with all LocalTensor instances replaced by their reconciled torch.Tensor equivalents,
             preserving the original structure.
    """
    if kwargs is None:
        kwargs = {}
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    reconciled_args = [
        a.reconcile() if isinstance(a, LocalTensor) else a for a in flat_args
    ]
    return pytree.tree_unflatten(reconciled_args, args_spec)


def local_tensor_mode() -> LocalTensorMode | None:
    """
    Returns the current active LocalTensorMode if one exists.

    This function checks the global stack of LocalTensorMode instance. If there
    is at least one LocalTensorMode active, it returns the most recently entered
    (top of the stack) LocalTensorMode. If no LocalTensorMode is active, it returns None.

    Returns:
        Optional[LocalTensorMode]: The current LocalTensorMode if active, else None.
    """
    local_tensor_mode_list = get_local_tensor_mode_list()
    if len(local_tensor_mode_list) > 0:
        return local_tensor_mode_list[-1]
    return None


def enabled_local_tensor_mode() -> LocalTensorMode | None:
    """
    Returns the current active LocalTensorMode only if it's enabled.

    This is a convenience function that combines the common pattern of checking
    if local_tensor_mode() is not None and not disabled.

    Returns:
        Optional[LocalTensorMode]: The current LocalTensorMode if active and enabled, else None.
    """
    lm = local_tensor_mode()
    if lm is not None and not lm._disable:
        return lm
    return None


def maybe_run_for_local_tensor(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Decorator that ensures a function is executed for each local tensor shard
    when running under LocalTensorMode. If not in LocalTensorMode, the function
    is executed normally. When in LocalTensorMode, the function is run for each
    rank, and the results are collected appropriately.

    This decorator is useful for functions that exhibit non-SPMD behavior, such
    as those requiring rank specific actions. For example, a function that computes
    offset into input tensor based on rank.

    Note that the function being decorated must not have any side effects and
    contain operations for a single rank only. For example, wrapping a function
    that performs a collective operation will not work.

    Args:
        func (Callable[..., Any]): The function to be decorated.

    Returns:
        Callable[..., Any]: The wrapped function that handles LocalTensorMode logic.
    """

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not (lm := enabled_local_tensor_mode()):
            return func(*args, **kwargs)
        ret = None
        with lm.disable():
            ret = _for_each_rank_run_func(func, lm.ranks, args, kwargs, alias=False)

        return ret

    return wrapper


def rank_map(cb: Callable[[int], Tensor]) -> Tensor:
    """
    Creates a tensor by mapping a callback over the current rank.

    Under LocalTensorMode, calls cb(rank) for each simulated rank and returns
    a LocalTensor. In real distributed (no LocalTensorMode), calls
    cb(dist.get_rank()) and returns a plain Tensor.
    """
    lm = enabled_local_tensor_mode()
    if lm is not None:
        return lm.rank_map(cb)
    else:
        return cb(dist.get_rank())


def tensor_map(tensor: Tensor, cb: Callable[[int, Tensor], Tensor | None]) -> Tensor:
    """
    Transforms a tensor by mapping a callback over the current rank and its
    local shard.

    Under LocalTensorMode, calls cb(rank, shard) for each simulated rank and
    returns a LocalTensor. In real distributed (no LocalTensorMode), calls
    cb(dist.get_rank(), tensor) and returns the result directly.
    """
    lm = enabled_local_tensor_mode()
    if lm is not None:
        if not isinstance(tensor, LocalTensor):
            raise AssertionError(f"Expected LocalTensor, got {type(tensor)}")
        return lm.tensor_map(tensor, cb)
    else:
        r = cb(dist.get_rank(), tensor)
        if r is None:
            raise AssertionError("callback returned None")
        return r


def maybe_disable_local_tensor_mode() -> contextlib.AbstractContextManager:
    """
    Context manager that disables LocalTensorMode for the duration of the context.
    """
    lm = local_tensor_mode()
    return lm.disable() if lm is not None else contextlib.nullcontext()


def maybe_enable_local_tracker(
    device_type: str, distribute_region_enabled: bool, spec, generator
):
    """
    Returns a context manager for LocalTensor-mode RNG tracking if local tensor mode is enabled.

    Args:
        device_type: The device type (e.g., "cuda", "cpu")
        distribute_region_enabled: Whether distribute region is enabled
        spec: The DTensorSpec
        generator: Optional torch.Generator

    Returns:
        Context manager from local_tracker._distribute_region if local tensor mode is enabled,
        otherwise None.
    """
    if enabled_local_tensor_mode():
        local_tracker = _LocalOffsetBasedRNGTracker(device_type)
        local_tracker.distribute_region_enabled = distribute_region_enabled
        return local_tracker._distribute_region(spec, generator)

    return None


def get_generator_seed_for_device_type(device_type: str):
    """
    Gets the generator seed for a specific device type, handling LocalTensor mode appropriately.

    Args:
        device_type: The device type (e.g., "cuda", "cpu")

    Returns:
        If in LocalTensor mode with per-rank RNG states:
            - Returns int if all ranks have the same seed
            - Returns SymInt(LocalIntNode) if ranks have different seeds
        Otherwise:
              - Returns int seed from the device's RNG state
    """
    if lm := enabled_local_tensor_mode():
        if len(lm._per_rank_rng_states) == 0:
            device_module = torch.get_device_module(device_type)
            return device_module.get_rng_state()[:8].view(torch.int64).item()
        device_module = torch.get_device_module(device_type)

        original_state = _get_rng_state()

        rank_seeds = {}
        try:
            for rank in sorted(lm.ranks):
                _set_rng_state(*lm._per_rank_rng_states[rank])
                rank_seeds[rank] = int(
                    device_module.get_rng_state()[:8].view(torch.int64).item()
                )
        finally:
            # restore original state
            _set_rng_state(*original_state)

        unique_seeds = set(rank_seeds.values())
        if len(unique_seeds) == 1:
            return next(iter(unique_seeds))
        local_int_node = LocalIntNode(rank_seeds)
        return torch.SymInt(local_int_node)
    else:
        device_module = torch.get_device_module(device_type)
        return device_module.get_rng_state()[:8].view(torch.int64).item()


import threading
from queue import Queue


_LOCAL_RUNNER_MODE: "LocalRunnerMode | None" = None


class _ExceptionRaisingThread(threading.Thread):
    def __init__(
        self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None
    ):
        super().__init__(
            target=target, name=name, args=args, kwargs=kwargs, daemon=daemon
        )
        self.exception: BaseException | None = None

    def run(self):
        try:
            super().run()
        except BaseException as e:
            self.exception = e

    def join(self, timeout=None):
        super().join(timeout=timeout)
        if self.exception:
            raise self.exception


class LocalRunnerMode:
    """
    A class for running multiple SPMD functions concurrently, however at any point
    in time only one function can be running. The main use case for the local runner
    mode is to enable SPMD functions to be able to use send and recv to communicate
    with each other. Without local runner mode send and recv are not supported.
    """

    runner_context = threading.local()

    def __init__(
        self, ranks: frozenset[int] | int, concurrency: int, fn: Callable[[int], None]
    ):
        if isinstance(ranks, int):
            ranks = frozenset(range(ranks))
        self._ranks = ranks
        self._fn = fn
        self._run_lock = threading.Lock()
        self._run_id = -1
        self._run_cond = threading.Condition(self._run_lock)

        self._recv_objects: dict[int, dict[int, Queue]] = {
            dst: {src: Queue() for src in ranks} for dst in ranks
        }
        self._runners = [
            _ExceptionRaisingThread(target=self._run, args=(i,), name="LocalRunnerMode")
            for i in range(concurrency)
        ]
        self._process_mode = True

    def __enter__(self) -> "LocalRunnerMode":
        global _LOCAL_RUNNER_MODE
        if _LOCAL_RUNNER_MODE is not None:
            raise AssertionError("LocalRunnerMode is already running")
        _LOCAL_RUNNER_MODE = self

        global _PROCESS_MODE
        self._process_mode = _PROCESS_MODE
        _PROCESS_MODE = False
        for r in self._runners:
            r.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        for r in self._runners:
            r.join()
        global _LOCAL_RUNNER_MODE
        _LOCAL_RUNNER_MODE = None

        global _PROCESS_MODE
        _PROCESS_MODE = self._process_mode

    def _run(self, id: int) -> None:
        LocalRunnerMode.runner_context.id = id
        # Only one thread can run at a time, hence must acquire the lock
        try:
            self._acquire_run_lock()
            self._fn(id)
        finally:
            self._release_run_lock()

    def _acquire_run_lock(self) -> None:
        self._run_lock.acquire()
        self._run_id = LocalRunnerMode.runner_context.id

    def _release_run_lock(self) -> None:
        self._run_id = -1
        self._run_lock.release()

    def _assert_holds_run_lock(self) -> None:
        if self._run_id != LocalRunnerMode.runner_context.id:
            raise AssertionError("Calling thread does not hold the run lock")

    def _get_recv_object(self, src: int, dst: int) -> object | None:
        peers = [src] if src != -1 else list(self._ranks)
        recv_objects = self._recv_objects[dst]

        for p in peers:
            if not recv_objects[p].empty():
                return recv_objects[p].get()

        return None

    def _signal_send(self, src: int, dst: int, obj: object) -> None:
        if obj is None:
            raise AssertionError("Cannot signal None")
        # Only a single thread a time executes so it is safe to mutate
        # read objects queue (executing thread is already holding the lock)
        self._recv_objects[dst][src].put(obj)
        # Signal directly condition variable since the calling thread is already
        # holding the lock
        self._run_cond.notify_all()

    def _wait_recv(self, src: int, dst: int, post: Callable[[object], None]) -> None:
        # Wait for the object to be available
        while True:
            obj = self._get_recv_object(src, dst)
            if obj is not None:
                post(obj)
                # Note that we are not releasing the lock here, since the thread
                # will continue to run and therefore must hold the lock
                return
            self._run_cond.wait()

    @staticmethod
    def current() -> "LocalRunnerMode":
        global _LOCAL_RUNNER_MODE
        if _LOCAL_RUNNER_MODE is None:
            raise AssertionError("LocalRunnerMode is not enabled")
        return _LOCAL_RUNNER_MODE


class _LocalPhiloxState:
    """
    LocalTensor-aware version of _PhiloxState that manages per-rank RNG states.
    This class handles the case where the generator state is a LocalTensor, allowing
    different offsets and seeds for different virtual ranks.

    Note: This is designed to be used as a drop-in replacement for _PhiloxState
    when working with LocalTensors in the DTensor random ops implementation.
    """

    def __init__(self, state: torch.Tensor):
        if not isinstance(state, LocalTensor):
            raise AssertionError("_LocalPhiloxState requires a LocalTensor")
        self._local_tensor = state
        self._per_rank_states = {
            rank: local_state.to("cpu")
            for rank, local_state in state._local_tensors.items()
        }

    @property
    def state(self):
        return LocalTensor(self._per_rank_states)  # type: ignore[name-defined]

    @property
    def offset(self) -> int | SymInt:
        from torch.distributed.tensor._random import _PhiloxState

        offsets = {}
        for rank, state in self._per_rank_states.items():
            rank_philox = _PhiloxState(state)
            offsets[rank] = int(rank_philox.offset.item())

        if len(set(offsets.values())) == 1:
            return next(iter(offsets.values()))

        return SymInt(LocalIntNode(offsets))

    @offset.setter
    def offset(self, offset: int | SymInt) -> None:
        from torch.distributed.tensor._random import _PhiloxState

        if isinstance(offset, SymInt) and isinstance(offset.node, LocalIntNode):
            for rank, state in self._per_rank_states.items():
                rank_offset = offset.node._local_ints[rank]
                rank_philox = _PhiloxState(state)
                rank_philox.offset = torch.tensor([rank_offset], dtype=torch.int64)
        else:
            offset_int = int(offset) if isinstance(offset, SymInt) else offset
            offset_tensor = torch.tensor([offset_int], dtype=torch.int64)
            for state in self._per_rank_states.values():
                rank_philox = _PhiloxState(state)
                rank_philox.offset = offset_tensor

    @property
    def seed(self) -> int | SymInt:
        from torch.distributed.tensor._random import _PhiloxState

        seeds = {}
        for rank, state in self._per_rank_states.items():
            rank_philox = _PhiloxState(state)
            seeds[rank] = int(rank_philox.seed.item())

        if len(set(seeds.values())) == 1:
            return next(iter(seeds.values()))
        return SymInt(LocalIntNode(seeds))

    @seed.setter
    def seed(self, seed: int | SymInt) -> None:
        from torch.distributed.tensor._random import _PhiloxState

        if isinstance(seed, SymInt) and isinstance(seed.node, LocalIntNode):
            for rank, state in self._per_rank_states.items():
                rank_seed = seed.node._local_ints[rank]
                rank_philox = _PhiloxState(state)
                rank_philox.seed = torch.tensor([rank_seed], dtype=torch.int64)
        else:
            seed_int = int(seed) if isinstance(seed, SymInt) else seed
            seed_tensor = torch.tensor([seed_int], dtype=torch.int64)
            for state in self._per_rank_states.values():
                rank_philox = _PhiloxState(state)
                rank_philox.seed = seed_tensor

    def apply_to_local_tensor_mode(self, device_handle) -> None:
        """
        Apply per-rank RNG states to the LocalTensorMode's tracked states.
        This updates both the device RNG state and the LocalTensorMode's _per_rank_rng_states.

        Args:
            device_handle: The device handle to use for setting RNG state (_LocalDeviceHandle)
        """
        if not enabled_local_tensor_mode():
            return

        if not hasattr(self, "_per_rank_offsets"):
            raise AssertionError

        for rank in sorted(self._per_rank_states.keys()):
            offset_value = self._per_rank_offsets[rank]
            if isinstance(offset_value, SymInt):
                if isinstance(offset_value.node, LocalIntNode):
                    offset_value = offset_value.node._local_ints[rank]
                else:
                    offset_value = int(offset_value)

            offset_tensor = torch.tensor(
                [offset_value], dtype=torch.uint64, device="cpu"
            ).view(torch.uint8)
            self._per_rank_states[rank][8:] = offset_tensor

        # pyrefly: ignore [bad-argument-type, bad-argument-count]
        device_handle.set_rng_state(LocalTensor(self._per_rank_states))
<|endoftext|>import inspect
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from functools import wraps
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec

import torch
import torch._custom_ops
from torch._C import DispatchKey
from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export
from torch._higher_order_ops.flat_apply import (
    _ConstantFunction,
    flat_apply,
    to_graphable,
)
from torch._higher_order_ops.strict_mode import strict_mode
from torch._higher_order_ops.utils import autograd_not_implemented, register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import (
    PreDispatchTorchFunctionMode,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type


if TYPE_CHECKING:
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI
    from torch.fx.experimental.proxy_tensor import _ProxyTracer
    from torch.fx.proxy import Proxy
    from torch.utils.hooks import RemovableHandle


_P = ParamSpec("_P")
_R = TypeVar("_R")


class ExportTracepoint(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("_export_tracepoint")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # pyrefly: ignore [missing-attribute]
        return super().__call__(*args, **kwargs)


_export_tracepoint = ExportTracepoint()


@_export_tracepoint.py_impl(ProxyTorchDispatchMode)
def export_tracepoint_dispatch_mode(
    mode: ProxyTorchDispatchMode, *args: Any, **kwargs: Any
) -> Any:
    # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
    p_args, p_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, (args, kwargs))
    proxy = mode.tracer.create_proxy(
        "call_function", _export_tracepoint, p_args, p_kwargs
    )
    return track_tensor_tree(args, proxy, constant=None, tracer=mode.tracer)


@register_fake(_export_tracepoint, skip_cache=True)
def export_tracepoint_fake_tensor_mode(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    return args


@_export_tracepoint.py_functionalize_impl
def export_tracepoint_functional(
    ctx: "BaseFunctionalizeAPI", *args: Any, **kwargs: Any
) -> tuple[Any, ...]:
    unwrapped_args = ctx.unwrap_tensors(args)
    # pyrefly: ignore[bad-argument-type]  # TODO unwrap_tensors accepts pytrees at runtime
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)

    with ctx.redispatch_to_next():
        _export_tracepoint(*unwrapped_args, **unwrapped_kwargs)
        return args


_export_tracepoint.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(_export_tracepoint, deferred_error=True)
)


@_export_tracepoint.py_impl(DispatchKey.CPU)
def export_tracepoint_cpu(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
    return args


def _wrap_submodule(
    mod: torch.nn.Module,
    path: str,
    module_call_specs: dict[str, dict[str, pytree.TreeSpec]],
) -> tuple["RemovableHandle", "RemovableHandle"]:
    if not isinstance(mod, torch.nn.Module):
        raise AssertionError(f"expected torch.nn.Module, got {type(mod)}")
    if path == "":
        raise AssertionError("path must not be empty")
    submodule = torch.fx.graph_module._get_attr(mod, path)

    def update_module_call_signatures(
        path: str, in_spec: pytree.TreeSpec, out_spec: pytree.TreeSpec
    ) -> None:
        if path in module_call_specs:
            if module_call_specs[path]["in_spec"] != in_spec:
                raise AssertionError(
                    f"in_spec mismatch for {path}: {module_call_specs[path]['in_spec']} != {in_spec}"
                )
            if module_call_specs[path]["out_spec"] != out_spec:
                raise AssertionError(
                    f"out_spec mismatch for {path}: {module_call_specs[path]['out_spec']} != {out_spec}"
                )
        module_call_specs[path] = {"in_spec": in_spec, "out_spec": out_spec}

    def check_flattened(flat_args: list[Any]) -> None:
        for a in flat_args:
            if not (isinstance(a, (torch.Tensor, str, int, float, bool)) or a is None):
                raise AssertionError(
                    f"Only Tensors or scalars are supported as pytree flattened inputs, got: {a}"
                )

    def pre_hook(
        module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        flat_args, in_spec = pytree.tree_flatten((args, kwargs))
        check_flattened(flat_args)
        flat_args = _export_tracepoint(*flat_args, kind="module_call_inputs", path=path)
        args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
        return args, kwargs

    def post_hook(
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        res: Any,
    ) -> Any:
        _, in_spec = pytree.tree_flatten((args, kwargs))
        flat_res, out_spec = pytree.tree_flatten(res)
        check_flattened(flat_res)
        flat_res = _export_tracepoint(*flat_res, kind="module_call_outputs", path=path)
        update_module_call_signatures(path, in_spec, out_spec)
        return pytree.tree_unflatten(flat_res, out_spec)

    pre_handle = submodule.register_forward_pre_hook(pre_hook, with_kwargs=True)
    post_handle = submodule.register_forward_hook(post_hook, with_kwargs=True)
    return pre_handle, post_handle


@contextmanager
def _wrap_submodules(
    f: torch.nn.Module | Callable[..., object],
    preserve_signature: Iterable[str],
    module_call_signatures: dict[str, dict[str, pytree.TreeSpec]],
) -> Generator[None, None, None]:
    handles: list[RemovableHandle] = []

    try:
        for path in preserve_signature:
            # pyrefly: ignore[bad-argument-type]  # TODO f is an nn.Module at runtime (asserted in _wrap_submodule)
            handles.extend(_wrap_submodule(f, path, module_call_signatures))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _mark_strict_experimental(cls: type[_R]) -> type[_R]:
    def call(self: Any, *args: Any) -> Any:
        return strict_mode(self, args)

    cls.__call__ = call  # type: ignore[attr-defined]
    return cls


def _register_func_spec_proxy_in_tracer(
    tracer: "_ProxyTracer", name: str, spec: pytree.TreeSpec
) -> "Proxy":
    """
    This is a wrapper utility method on top of tracer to cache the
    already registered subclass spec attribute. This is useful because
    Subclass.__init__ will be same for each subclass. By default, fx will
    create multiple attributes/proxies for given attribute.
    """
    fx_name = name + "0"
    # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
    if hasattr(tracer.root, fx_name):
        # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
        if getattr(tracer.root, fx_name) != spec:
            raise AssertionError(f"spec mismatch for {fx_name}")
        return tracer.create_proxy("get_attr", fx_name, (), {})

    # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
    qualname = tracer.get_fresh_qualname(name)
    # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
    setattr(tracer.root, qualname, spec)
    return tracer.create_proxy("get_attr", qualname, (), {})


def _emit_flat_apply_call(
    *,
    tracer: "_ProxyTracer",
    spec_name: str,
    const_target_for_apply: Callable[..., object],
    graphable_args: pytree.PyTree,
    track_value: object,
    call_spec_cache_key: str,
) -> None:
    # Flatten to graphable form and record the spec on the FX root
    flat_args, in_spec = to_graphable(graphable_args)
    qualname = tracer.get_fresh_qualname(spec_name)  # type: ignore[union-attr]
    setattr(tracer.root, qualname, in_spec)  # type: ignore[union-attr]
    spec_proxy = tracer.create_proxy("get_attr", qualname, (), {})

    # Reuse/cached ConstantFunction spec on the root
    _, func_spec = pytree.tree_flatten(_ConstantFunction(const_target_for_apply))
    func_spec_proxy = _register_func_spec_proxy_in_tracer(
        tracer, f"{call_spec_cache_key}_const_func_spec", func_spec
    )

    # Map runtime args -> proxies (always via tracer.unwrap_proxy now)
    # pyrefly: ignore[missing-attribute]  # TODO tracer is PythonKeyTracer here, not _GraphAppendingTracerEx
    flat_proxy_args = pytree.tree_map(tracer.unwrap_proxy, flat_args)

    # Emit flat_apply and track result structure
    out_proxy = tracer.create_proxy(
        "call_function", flat_apply, (func_spec_proxy, spec_proxy, *flat_proxy_args), {}
    )
    track_tensor_tree(track_value, out_proxy, constant=None, tracer=tracer)


def _is_init(fn: object) -> bool:
    return callable(fn) and fn.__name__ == "__init__"


def mark_subclass_constructor_exportable_experimental(
    constructor_subclass: Callable[_P, None],
) -> Callable[_P, None]:
    """
    Experimental decorator that makes subclass to be traceable in export
    with pre-dispatch IR. To make your subclass traceable in export, you need to:
        1. Implement __init__ method for your subclass (Look at DTensor implementation)
        2. Decorate your __init__ method with _mark_constructor_exportable_experimental
        3. Put torch._dynamo_disable decorator to prevent dynamo from peeking into its' impl

    Example:

    class FooTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, elem, *, requires_grad=False):
            # ...
            return torch.Tensor._make_subclass(cls, elem, requires_grad=requires_grad)

        @torch._dynamo_disable
        @mark_subclass_constructor_exportable_experimental
        def __init__(self, elem, ...):
            # ...
    """
    if not _is_init(constructor_subclass):
        raise RuntimeError(
            f"torch._export.wrappers.mark_constructor_exportable_experimental can only be applied on subclass tensor.__init__"
            f"But, you are adding it on {constructor_subclass.__name__} which is not supported. "
            f"If __init__ doesn't exist on your subclass, please add it. Look at DTensor.__init__ implementation for example"
        )

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        constructor_subclass(*args, **kwargs)

        if not torch.compiler.is_exporting():
            return

        if not is_traceable_wrapper_subclass_type(type(args[0])):
            if not constructor_subclass.__qualname__.endswith("__init__"):
                raise AssertionError(
                    f"expected __qualname__ to end with '__init__', got {constructor_subclass.__qualname__}"
                )
            obj_name = constructor_subclass.__qualname__[: -len("__init__")]
            raise RuntimeError(
                f"Can't intercept {obj_name} in export because this object is not a traceable "
                f"tensor subclass. Please look at DTensor.__init__ implementation as an example of proper usage of this API."
            )

        mode = _maybe_find_pre_dispatch_tf_mode_for_export()
        if mode is None:
            return

        if not isinstance(mode, PreDispatchTorchFunctionMode):
            raise AssertionError(
                f"expected PreDispatchTorchFunctionMode, got {type(mode)}"
            )

        tracer = mode.tracer
        subclass = args[0]
        graphable = (tuple(args[1:]), kwargs)

        spec_name = "_".join(constructor_subclass.__qualname__.lower().split("."))
        call_spec_cache_key = type(subclass).__name__.lower()

        _emit_flat_apply_call(
            tracer=tracer,
            spec_name=spec_name,
            const_target_for_apply=type(subclass),
            graphable_args=graphable,
            track_value=subclass,  # track the constructed subclass instance
            call_spec_cache_key=call_spec_cache_key,
        )
        return

    return wrapper


def allow_in_pre_dispatch_graph(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """
    Experimental decorator that adds user function to export pre-dispatch graph. Note that
    we only support custom autograd function/subclass constructors today. To use this function:
        1. For subclasses:
            1. refer to instructions in mark_subclass_constructor_exportable_experimental
        2. Define apply method on your custom autograd function and apply this decorator.

    Example:

    class MyCoolCustomAutogradFunc(autograd.Function):
        @classmethod
        @torch._export.wrappers.allow_in_pre_dispatch_graph
        def apply(cls, *args, **kwargs):
            return super(MyCoolCustomAutogradFunc, cls).apply(*args, **kwargs)

    """
    if _is_init(func):
        # pyrefly: ignore[bad-return, bad-argument-type]  # TODO _is_init(func) implies func returns None
        return mark_subclass_constructor_exportable_experimental(func)

    if not (_is_init(func) or func.__name__ == "apply"):
        raise RuntimeError(
            f"torch._export.wrappers.allow_in_pre_dispatch_graph can only be applied on subclass tensor.__init_ "
            f"or custom_autograd_function.apply. "
            f"But, you are adding it on {func.__name__} which is not supported. "
            f"If __init__ doesn't exist on your subclass, please add it. Look at DTensor.__init__ implementation for example. "
            f"If you are adding it on custom autograd function, please add it on apply method. "
            f"If anything else, file an issue on github and we may consider extending our support. "
        )

    @wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not torch.compiler.is_exporting():
            return func(*args, **kwargs)

        if not inspect.isclass(args[0]):
            return func(*args, **kwargs)

        if not issubclass(args[0], torch.autograd.Function):
            return func(*args, **kwargs)

        from torch._ops import _get_dispatch_mode_pre_dispatch

        mode = _get_dispatch_mode_pre_dispatch(torch._C._TorchDispatchModeKey.PROXY)
        if mode is None:
            return func(*args, **kwargs)

        # Sometimes custom autograd functions can call into HOPs that don't have proxy impl
        # at PreDispatch level, so we just dispatch it below to get the concrete result.
        include_to_set = torch._C._dispatch_tls_local_include_set().remove(
            torch._C.DispatchKey.PreDispatch
        )
        exclude_to_set = (
            torch._C._dispatch_tls_local_exclude_set()
            | torch._C.DispatchKeySet(torch._C.DispatchKey.PreDispatch)
        )

        with torch._C._ForceDispatchKeyGuard(include_to_set, exclude_to_set):
            out = func(*args, **kwargs)

        if not mode.pre_dispatch:
            raise AssertionError("Should only do this in predispatch")
        tracer = mode.tracer

        function_cls_name = f"{args[0].__module__}.{args[0].__qualname__}"
        graphable = ((function_cls_name, *args[1:]), kwargs)

        from torch.export.custom_ops import (
            _call_custom_autograd_function_in_pre_dispatch,
        )

        spec_name = "_".join(function_cls_name.split("."))
        call_spec_cache_key = type(
            _call_custom_autograd_function_in_pre_dispatch
        ).__name__.lower()
        _emit_flat_apply_call(
            tracer=tracer,
            spec_name=spec_name,
            const_target_for_apply=_call_custom_autograd_function_in_pre_dispatch,
            graphable_args=graphable,
            track_value=out,
            call_spec_cache_key=call_spec_cache_key,
        )
        return out

    return wrapper
<|endoftext|>import ast
import copy
import dataclasses
import functools
import inspect
import json
import math
import operator
import re
from collections import defaultdict
from collections.abc import Callable, Collection, Generator, Iterable, Sequence
from contextlib import contextmanager
from inspect import ismethod, Parameter
from typing import Any, TYPE_CHECKING

import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, is_fake_tensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx._utils import first_call_function_nn_module_stack
from torch.fx.experimental.proxy_tensor import PreDispatchTorchFunctionMode
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts


if TYPE_CHECKING:
    import sympy

    from torch._export.passes.lift_constants_pass import ConstantAttrMap
    from torch._functorch._aot_autograd.schemas import GraphSignature
    from torch._ops import OperatorBase
    from torch.export import ExportedProgram
    from torch.export.graph_signature import ExportGraphSignature
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.utils._pytree import TreeSpec

from torch.export.graph_signature import CustomObjArgument, InputKind, OutputKind
from torch.fx._pytree import (
    _deregister_pytree_flatten_spec,
    register_pytree_flatten_spec,
)
from torch.utils._pytree import (
    _deregister_pytree_node,
    _register_pytree_node,
    Context,
    FlattenFn,
    FromDumpableContextFn,
    GetAttrKey,
    KeyPath,
    keystr,
    MappingKey,
    SequenceKey,
    ToDumpableContextFn,
    tree_flatten_with_path,
    UnflattenFn,
)


placeholder_prefixes = {
    InputKind.USER_INPUT: "",
    InputKind.PARAMETER: "p_",
    InputKind.BUFFER: "b_",
    InputKind.CONSTANT_TENSOR: "c_",
    InputKind.CUSTOM_OBJ: "obj_",
    InputKind.TOKEN: "token",
}

_DISABLE_ATEN_TO_ASSERTION_PASS = False


def _collect_and_set_constant_attrs(
    graph_signature: "ExportGraphSignature",
    constants: dict[str, Any],
    mod: torch.nn.Module,
) -> "ConstantAttrMap":
    # the exported module will store constants & non-persistent buffers such that
    # retracing treats them as persistent buffers, so we inform the constants lifting pass
    # and overwrite the new graph signature using the previous program. This is intended to only be used
    # in run_decompositions where we still have access to original EP.
    from torch._export.passes.lift_constants_pass import ConstantAttrMap

    constant_attrs = ConstantAttrMap()
    non_persistent_buffers = {
        spec.target
        for spec in graph_signature.input_specs
        if spec.kind == InputKind.BUFFER and not spec.persistent
    }
    for name, value in constants.items():
        if name in non_persistent_buffers:
            continue
        # recursive getattr
        _mod = mod
        *atoms, attr = name.split(".")
        for atom in atoms:
            _mod = getattr(_mod, atom)
        # remove as buffer, reassign as constant/non-persistent buffer
        _mod._buffers.pop(attr, None)
        setattr(_mod, attr, value)
        constant_attrs.add(value, name)
    return constant_attrs


def _register_constants_as_buffers(
    mod: torch.fx.GraphModule,
    state_dict: dict[str, Any],
    non_persistent_buffers: Collection[str],
) -> set[str]:
    # TODO some annoying circular dependency issue
    from torch.export.unflatten import _assign_attr, _AttrKind

    temp_registered_constants = set()

    for node in mod.graph.nodes:
        if node.op == "get_attr":
            target = torch.fx.graph_module._get_attr(mod, node.target)
            if isinstance(target, torch.Tensor):
                # Make sure we also check if the original buffer is
                # non persistent as well.
                if (node.target not in state_dict) and (
                    node.target not in non_persistent_buffers
                ):
                    torch.fx.graph_module._del_attr(mod, node.target)
                    _assign_attr(target, mod, node.target, _AttrKind.BUFFER, False)
                    temp_registered_constants.add(node.target)

    mod.recompile()

    return temp_registered_constants


def _override_graph_signature_for_temp_registered_constants(
    sig: "ExportGraphSignature", temp_registered_constants: set[str]
) -> "ExportGraphSignature":
    for spec in sig.input_specs:
        if spec.target in temp_registered_constants:
            spec.kind = InputKind.CONSTANT_TENSOR
            spec.persistent = None

    for spec in sig.output_specs:
        if (
            spec.kind == OutputKind.BUFFER_MUTATION
            and spec.target in temp_registered_constants
        ):
            raise RuntimeError(
                f"Constant {spec.target} is mutated in the forward method. Pls register it as buffer"
            )

    return sig


def _overwrite_signature_for_non_persistent_buffers(
    old_sig: "ExportGraphSignature", new_sig: "ExportGraphSignature"
) -> "ExportGraphSignature":
    # overwrite signature for non-persistent buffers
    non_persistent_buffers = {
        spec.target
        for spec in old_sig.input_specs
        if spec.kind == InputKind.BUFFER and not spec.persistent
    }

    for spec in new_sig.input_specs:
        if spec.kind == InputKind.BUFFER and spec.target in non_persistent_buffers:
            spec.persistent = False
    return new_sig


def _collect_param_buffer_metadata(mod: torch.fx.GraphModule) -> dict[str, Any]:
    """
    Param/buffer metadata needs to be saved before lowering to aten IR
    because aten IR lifts them, as a result, automatic preservation doesn't work.
    This is intended to be called on the strict mode tracing right before lowering to
    aten IR OR run_decomposition pass.
    """
    params_buffers_to_node_meta = {}

    def _getattr(model: torch.fx.GraphModule, attr_name: str) -> Any:
        *prefix, field = attr_name.split(".")
        t = model
        for item in prefix:
            t = getattr(t, item, None)  # type: ignore[assignment]
            if t is None:
                raise AssertionError(f"attribute {item!r} not found in path")

        return getattr(t, field)

    for node in mod.graph.nodes:
        target = node.target
        meta = node.meta
        if node.op == "call_module":
            submodule = _getattr(mod, target)
            if isinstance(submodule, torch.nn.Module):
                for name, _ in submodule.named_parameters(
                    recurse=True, remove_duplicate=False
                ):
                    params_buffers_to_node_meta[target + "." + name] = meta

                for name, _ in submodule.named_buffers(
                    recurse=True, remove_duplicate=False
                ):
                    params_buffers_to_node_meta[target + "." + name] = meta

        if node.op == "get_attr":
            submodule = _getattr(mod, target)
            if not isinstance(submodule, torch.fx.GraphModule):
                params_buffers_to_node_meta[target] = meta

        # If the call_function uses param as input, we also need to update params' meta
        # with this call_function node's meta.
        # This is basically the same flow as torch.fx.traceback.preserve_meta()
        if node.op == "call_function" and not isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            for arg in node._input_nodes:
                if arg.op == "get_attr":
                    for entry in torch.fx.proxy._COPY_META_FIELDS:
                        #  the custom field should not be copied
                        if entry == "custom":
                            continue
                        if entry in meta:
                            params_buffers_to_node_meta[arg.target][entry] = meta[entry]

    return params_buffers_to_node_meta


def _maybe_find_pre_dispatch_tf_mode_for_export() -> (
    PreDispatchTorchFunctionMode | None
):
    if not torch.compiler.is_exporting():
        return None

    if not torch._C._is_torch_function_mode_enabled():
        return None

    torch_function_mode_stack = torch.overrides._get_current_function_mode_stack()

    pre_dispatch_tf_modes = [
        mode
        for mode in torch_function_mode_stack
        if isinstance(mode, PreDispatchTorchFunctionMode)
    ]

    if len(pre_dispatch_tf_modes) > 1:
        raise AssertionError(
            f"Expected only one PreDispatchTorchFunctionMode, found {len(pre_dispatch_tf_modes)}"
        )

    if len(pre_dispatch_tf_modes) == 0:
        return None

    mode = pre_dispatch_tf_modes[0]
    return mode


def _populate_param_buffer_metadata_to_new_gm(
    params_buffers_to_node_meta: dict[str, Any],
    gm: torch.fx.GraphModule,
    new_sig: "ExportGraphSignature",
) -> None:
    """
    Given that we collected param'buffer metadata before, we put them back in
    newly traced graph module
    """
    # Don't copy over nn_module_stack, stack_trace metadata for params/buffers nodes
    for metadata in params_buffers_to_node_meta.values():
        metadata.pop("nn_module_stack", None)
        metadata.pop("stack_trace", None)

    inputs_to_parameters = new_sig.inputs_to_parameters
    inputs_to_buffers = new_sig.inputs_to_buffers
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.target in inputs_to_parameters:
                param_name = inputs_to_parameters[node.target]
                if param_name in params_buffers_to_node_meta:
                    for k, v in params_buffers_to_node_meta[param_name].items():
                        node.meta[k] = v
            if node.target in inputs_to_buffers:
                buffer_name = inputs_to_buffers[node.target]
                if buffer_name in params_buffers_to_node_meta:
                    for k, v in params_buffers_to_node_meta[buffer_name].items():
                        node.meta[k] = v


def _get_shape_env_from_gm(gm: torch.fx.GraphModule) -> "ShapeEnv | None":
    vals = [
        node.meta["val"]
        for node in gm.graph.nodes
        if node.meta.get("val", None) is not None
    ]

    fake_mode = _detect_fake_mode_from_gm(gm)
    if fake_mode is not None:
        return fake_mode.shape_env
    for v in vals:
        if isinstance(v, torch.SymInt):
            return v.node.shape_env


def _rename_without_collisions(
    name_map: dict[str, str],
    find_available: dict[str, int],
    used_names: set[str],
    orig_name: str,
    name: str,
    is_placeholder: bool = False,
) -> str:
    """
    Renames nodes to avoid name collisions, with suffixing.
    name_map: map from original name to new name
    find_available: map prefix to available suffix
    used_names: cache of used names
    orig_name: mapping key
    name: candidate name (potentially suffixed, e.g. mul_2)
    is_placeholder: if the node is a placeholder, avoid detecting suffix
    """
    match = re.match(r"(.*)_(\d+)", name)
    key = name

    if match and not is_placeholder:
        prefix, n = match.group(1), match.group(2)
        key = prefix

    new_name = name
    if new_name in used_names:
        new_name = f"{key}_{find_available[key] + 1}"

    match = re.match(r"(.*)_(\d+)", new_name)
    if match:
        prefix, n = match.group(1), match.group(2)
        if int(n) > find_available[prefix]:
            find_available[prefix] = int(n)

    name_map[orig_name] = new_name
    used_names.add(new_name)

    return name_map[orig_name]


def get_keystr(key_path: KeyPath) -> str:
    """For a given index into the flat_args, return a human readable string
    describing how to access it, e.g. "*args["foo"][0].bar"
    """
    # Prefix the keypath with "*args" or "**kwargs" to make it clearer where
    # the arguments come from. Ultimately we ought to serialize the
    # original arg names for the best error message here.
    args_kwargs_key_path = key_path[0]
    if not isinstance(args_kwargs_key_path, SequenceKey):
        raise AssertionError(
            f"expected SequenceKey, got {type(args_kwargs_key_path).__name__}"
        )
    if args_kwargs_key_path.idx == 0:
        return f"*args{keystr(key_path[1:])}"
    else:
        kwarg_key = key_path[1]
        if not isinstance(kwarg_key, (GetAttrKey, MappingKey)):
            raise AssertionError(
                f"expected GetAttrKey or MappingKey, got {type(kwarg_key).__name__}"
            )
        name = str(kwarg_key)[1:-1]  # get rid of the enclosed []
        return f"{name}{keystr(key_path[2:])}"


def _check_symint(
    symint: int | torch.SymInt,
    arg: int,
    range_constraints: dict["sympy.Expr", Any],
    unification_map: dict["sympy.Symbol", Any],
    keypath: KeyPath,
    i: int | None = None,
) -> None:
    from torch.export.dynamic_shapes import _IntWrapper

    if (
        isinstance(arg, torch.SymInt)
        and not arg.node.expr.is_number
        or isinstance(arg, _IntWrapper)
    ):
        # This can happen when, say, arg is a fake tensor.
        # We do not run checks on symbolic shapes of fake inputs as
        # such checks can affect the shape env.
        return

    import sympy

    from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
        _convert_range_to_int,
    )
    from torch.utils._sympy.solve import try_solve

    if isinstance(symint, torch.SymInt) and len(symint.node.expr.free_symbols) == 1:
        symbol = next(iter(symint.node.expr.free_symbols))
        if symbol in unification_map:
            existing_dim = symint.node.expr.subs(unification_map)
            if arg != existing_dim:
                path = get_keystr(keypath)
                if i is not None:
                    path += f".shape[{i}]"
                raise RuntimeError(
                    f"Expected input at {path} to be equal to {existing_dim}, but got {arg}",
                )
        else:
            if isinstance(symint.node.expr, sympy.Symbol):
                # Short cut for try_solve below. Also useful in cases where
                # sympy.Eq(symint.node.expr, arg) would evaluate to False
                # purely because symbol is constrained to be size-like,
                # e.g., when symint.node.expr = symbol and arg = 0.
                unification_map[symbol] = int(arg)
            else:
                solution = try_solve(sympy.Eq(symint.node.expr, arg), symbol)
                if solution is None:
                    path = get_keystr(keypath)
                    if i is not None:
                        path += f".shape[{i}]"
                    raise RuntimeError(
                        f"Expected input {path} = {arg} to be "
                        f"of the form {symint.node.expr}, where {symbol} is an integer"
                    )
                else:
                    unification_map[symbol] = int(solution[1])

        if symint.node.expr in range_constraints:
            min_val, max_val = _convert_range_to_int(
                range_constraints[symint.node.expr]
            )
            # NOTE: we allow dimensions to be 0/1 at runtime
            if min_val > 2:
                if arg < min_val:
                    path = get_keystr(keypath)
                    if i is not None:
                        path += f".shape[{i}]"
                    raise RuntimeError(
                        f"Expected input at {path} to be >= {min_val}, but got {arg}",
                    )
            if max_val < math.inf:
                if arg > max_val:
                    path = get_keystr(keypath)
                    if i is not None:
                        path += f".shape[{i}]"
                    raise RuntimeError(
                        f"Expected input at {path} to be <= {max_val}, but got {arg}",
                    )
    elif isinstance(symint, torch.SymInt) and not symint.node.expr.is_number:
        # this means we deferred a guard from export analysis to runtime, let this pass
        # we'll add a runtime assert checking equality to this replacement expression
        pass
    elif arg != int(symint):
        path = get_keystr(keypath)
        if i is not None:
            path += f".shape[{i}]"
        raise RuntimeError(
            f"Expected input at {path} to be equal to {symint}, but got {arg}. "
            "If you meant for this dimension to be dynamic, please re-export and specify dynamic_shapes "
            "(e.g. with Dim.DYNAMIC)"
        )


def _check_input_constraints_for_graph(
    input_placeholders: list[torch.fx.Node],
    flat_args_with_path: Sequence[tuple[tuple[Any, ...], Any]],
    range_constraints: dict["sympy.Expr", Any],
) -> None:
    if len(flat_args_with_path) != len(input_placeholders):
        raise RuntimeError(
            "Unexpected number of inputs "
            f"(expected {len(input_placeholders)}, got {len(flat_args_with_path)})"
        )
    # NOTE: export already guarantees that the same symbol is used in metadata
    # for all InputDims related by equality constraints, so we can just unify
    # symbols with given input dimension values to check equality constraints.
    unification_map: dict[sympy.Symbol, Any] = {}
    for (key_path, arg), node in zip(flat_args_with_path, input_placeholders):
        node_val = node.meta.get("val")
        if is_fake_tensor(node_val):
            if not isinstance(arg, torch.Tensor):
                raise RuntimeError(
                    f"Expected input at {get_keystr(key_path)} to be a tensor, but got {type(arg)}",
                )

            if len(node_val.shape) != len(arg.shape):
                raise RuntimeError(
                    f"Unexpected number of dimensions in input at {get_keystr(key_path)}.shape "
                    f"(expected {node_val.shape}, got {arg.shape})"
                )

            for j, (arg_dim, node_dim) in enumerate(zip(arg.shape, node_val.shape)):
                _check_symint(
                    node_dim, arg_dim, range_constraints, unification_map, key_path, j
                )

        elif isinstance(node_val, (int, float, str)):
            if type(arg) is not type(node_val):
                raise RuntimeError(
                    f"Expected input at {get_keystr(key_path)} to be equal to {node_val}, but got {arg}",
                )
            # NaN != NaN in Python, so use math.isnan for NaN-to-NaN comparison
            if isinstance(node_val, float) and math.isnan(node_val):
                if not isinstance(arg, float) or not math.isnan(arg):
                    raise RuntimeError(
                        f"Expected input at {get_keystr(key_path)} to be nan, but got {arg}",
                    )
            elif arg != node_val:
                raise RuntimeError(
                    f"Expected input at {get_keystr(key_path)} to be equal to {node_val}, but got {arg}",
                )
        elif isinstance(node_val, torch.SymInt):
            _check_symint(
                node_val,
                arg,
                range_constraints,
                unification_map,
                key_path,
                None,
            )


def register_dataclass_as_pytree_node(
    cls: type[Any],
    flatten_fn: FlattenFn | None = None,
    unflatten_fn: UnflattenFn | None = None,
    *,
    serialized_type_name: str | None = None,
    to_dumpable_context: ToDumpableContextFn | None = None,
    from_dumpable_context: FromDumpableContextFn | None = None,
    return_none_fields: bool = False,
) -> None:
    if not dataclasses.is_dataclass(cls):
        raise AssertionError(
            f"Only dataclasses can be registered with this function: {cls}"
        )

    @torch._dynamo.dont_skip_tracing
    def default_flatten_fn(obj: Any) -> tuple[list[Any], Context]:
        flattened = []
        flat_names = []
        none_names = []
        for f in dataclasses.fields(obj):
            name, val = f.name, getattr(obj, f.name)
            if val is not None or return_none_fields:
                flattened.append(val)
                flat_names.append(name)
            else:
                none_names.append(name)
        return flattened, [flat_names, none_names]

    @torch._dynamo.dont_skip_tracing
    def default_unflatten_fn(values: Iterable[Any], context: Context) -> Any:
        flat_names, none_names = context
        return cls(**dict(zip(flat_names, values)), **dict.fromkeys(none_names))

    @torch._dynamo.dont_skip_tracing
    def default_flatten_fn_with_keys(obj: Any) -> tuple[list[Any], Context]:
        flattened, (flat_names, _none_names) = flatten_fn(obj)  # type: ignore[misc]
        return [(MappingKey(k), v) for k, v in zip(flat_names, flattened)], flat_names

    flatten_fn = flatten_fn if flatten_fn is not None else default_flatten_fn
    unflatten_fn = unflatten_fn if unflatten_fn is not None else default_unflatten_fn

    if (to_dumpable_context is None) ^ (from_dumpable_context is None):
        raise ValueError(
            f"Both to_dumpable_context and from_dumpable_context for {cls} must "
            "be None or registered."
        )

    _register_pytree_node(
        cls,
        flatten_fn,
        unflatten_fn,
        serialized_type_name=serialized_type_name,
        flatten_with_keys_fn=default_flatten_fn_with_keys,
        to_dumpable_context=to_dumpable_context,
        from_dumpable_context=from_dumpable_context,
    )


def is_param(program: "ExportedProgram", node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a parameter within the exported program
    """

    return node.name in program.graph_signature.inputs_to_parameters


def get_param(
    program: "ExportedProgram",
    node: torch.fx.Node,
) -> torch.nn.Parameter | None:
    """
    Returns the parameter associated with the given node in the exported program.
    Returns None if the node is not a parameter within the exported program
    """

    if is_param(program, node):
        parameter_name = program.graph_signature.inputs_to_parameters[node.name]
        return program.state_dict[parameter_name]

    return None


def is_buffer(program: "ExportedProgram", node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a buffer within the exported program
    """

    return node.name in program.graph_signature.inputs_to_buffers


def get_buffer(
    program: "ExportedProgram",
    node: torch.fx.Node,
) -> torch.Tensor | None:
    """
    Returns the buffer associated with the given node in the exported program.
    Returns None if the node is not a buffer within the exported program
    """

    if is_buffer(program, node):
        buffer_name = program.graph_signature.inputs_to_buffers[node.name]
        if buffer_name in program.graph_signature.non_persistent_buffers:
            return program.constants[buffer_name]
        else:
            return program.state_dict[buffer_name]

    return None


def is_lifted_tensor_constant(
    program: "ExportedProgram",
    node: torch.fx.Node,
) -> bool:
    """
    Checks if the given node is a lifted tensor constant within the exported program
    """

    return node.name in program.graph_signature.inputs_to_lifted_tensor_constants


def get_lifted_tensor_constant(
    program: "ExportedProgram",
    node: torch.fx.Node,
) -> torch.Tensor | None:
    """
    Returns the lifted tensor constant associated with the given node in the exported program.
    Returns None if the node is not a lifted tensor constant within the exported program
    """

    if is_lifted_tensor_constant(program, node):
        lifted_tensor_name = program.graph_signature.inputs_to_lifted_tensor_constants[
            node.name
        ]
        return program.constants[lifted_tensor_name]

    return None


def sequential_split(
    gm: torch.fx.GraphModule,
    node_call_back: Callable[[torch.fx.Node], torch.fx.Node | bool],
) -> torch.fx.GraphModule:
    """
    sequential_split creates a new graph module that splits the input graph module into multiple submodules
    based on the node_call_back. It doesn't mutate the input graph module. The node_call_back should return
    True if the node is a delimiter.  Delimiter will be the first node in the next submodule.
    """
    from torch.fx.passes.split_module import split_module

    split_map = {}
    split_id = 0
    for node in gm.graph.nodes:
        if node_call_back(node):
            split_id += 1
        split_map[node] = split_id

    new_gm = split_module(
        gm,
        gm,
        lambda node: split_map[node],
        keep_original_order=True,
        keep_original_node_name=True,
    )
    # Keep the codegen from original graph module to preserve e.g. pytree info.
    new_gm.graph._codegen = gm.graph._codegen
    new_gm.recompile()
    return new_gm


def nodes_filter(
    nodes: list[torch.fx.Node], node_call_back: Callable[[torch.fx.Node], object]
) -> list[torch.fx.Node]:
    """Returns the nodes that match the node_call_back as a list."""
    return [node for node in nodes if node_call_back(node)]


@contextmanager
def _disable_aten_to_metadata_assertions() -> Generator[None, None, None]:
    global _DISABLE_ATEN_TO_ASSERTION_PASS
    orig_val = _DISABLE_ATEN_TO_ASSERTION_PASS
    _DISABLE_ATEN_TO_ASSERTION_PASS = True
    try:
        yield
    finally:
        _DISABLE_ATEN_TO_ASSERTION_PASS = orig_val


def _insert_aten_to_metadata_assert_pass(gm: torch.fx.GraphModule) -> None:
    from torch._export.passes._node_metadata_hook import (
        _node_metadata_hook,
        _set_node_metadata_hook,
    )

    if _DISABLE_ATEN_TO_ASSERTION_PASS:
        return

    aten_to_variants = [
        torch.ops.aten.to.device,
        torch.ops.aten.to.dtype,
        torch.ops.aten.to.dtype_layout,
    ]
    for node in gm.graph.nodes:
        if node.target in aten_to_variants:
            if (
                node.prev.target is torch.ops.aten._assert_tensor_metadata.default
                and node.args[0] == node.prev.args[0]
            ):
                # skip if already guarded
                continue

            if (tensor_val := node.args[0].meta.get("val")) is not None:
                with (
                    gm.graph.inserting_before(node),
                    _set_node_metadata_hook(
                        gm,
                        functools.partial(
                            _node_metadata_hook,
                            metadata={
                                "stack_trace": node.meta.get("stack_trace"),
                                "nn_module_stack": node.meta.get("nn_module_stack"),
                            },
                        ),
                    ),
                ):
                    gm.graph.call_function(
                        torch.ops.aten._assert_tensor_metadata.default,
                        args=(node.args[0],),
                        kwargs={
                            "dtype": tensor_val.dtype,
                            "device": tensor_val.device,
                            "layout": tensor_val.layout,
                        },
                    )


def apply_runtime_assertion_pass(
    gm: torch.fx.GraphModule, graph_signature: "GraphSignature"
) -> tuple[torch.fx.GraphModule, "GraphSignature"]:
    from torch._export.passes._node_metadata_hook import (
        _node_metadata_hook,
        _set_node_metadata_hook,
    )
    from torch._functorch._aot_autograd.input_output_analysis import _graph_output_names

    if not torch._dynamo.config.do_not_emit_runtime_asserts:
        stack_trace = (
            'File "torch/fx/passes/runtime_assert.py", line 24, '
            "in insert_deferred_runtime_asserts"
        )
        with _set_node_metadata_hook(
            gm,
            functools.partial(
                _node_metadata_hook, metadata={"stack_trace": stack_trace}
            ),
        ):
            shape_env = _get_shape_env_from_gm(gm)
            if shape_env:
                insert_deferred_runtime_asserts(
                    gm,
                    shape_env,
                    f"exported program: {first_call_function_nn_module_stack(gm.graph)}",
                    export=True,
                )

        # insert runtime assertions for aten.to nodes
        _insert_aten_to_metadata_assert_pass(gm)

    # update output specs
    gm.recompile()
    graph_signature.user_outputs = _graph_output_names(gm)
    return gm, graph_signature


def nodes_first(
    nodes: list[torch.fx.Node],
    node_call_back: Callable[[torch.fx.Node], object] | None = None,
) -> torch.fx.Node | None:
    """
    Returns the first node that matches the node_call_back. If no node matches, returns None.
    When node_call_back is None, returns the first node in the node list.
    """
    ret = nodes_filter(nodes, node_call_back if node_call_back else lambda node: True)
    if len(ret) > 0:
        return ret[0]
    return None


def nodes_count(
    nodes: list[torch.fx.Node], node_call_back: Callable[[torch.fx.Node], object]
) -> int:
    """Returns the number of nodes that match the node_call_back."""
    return len(nodes_filter(nodes, node_call_back))


def nodes_map(
    nodes: list[torch.fx.Node], node_call_back: Callable[[torch.fx.Node], object]
) -> list[torch.fx.Node]:
    """
    Sequentially visit the nodes list and invoke node_call_back on each element.
    Returns the nodes list after the node_call_back is invoked on each element.
    """
    for node in nodes:
        node_call_back(node)
    return nodes


def node_replace_(old_node: torch.fx.Node, new_node: torch.fx.Node) -> None:
    """
    Replace all uses of old_node with new_node.
    """
    old_node.replace_all_uses_with(new_node)
    old_node.users.clear()
    old_node.graph.erase_node(old_node)


def _update_gm_meta_if_possible(gm: torch.fx.GraphModule, mod: torch.nn.Module) -> None:
    if (
        isinstance(mod, torch.fx.GraphModule)
        and hasattr(mod, "meta")
        and "custom" in mod.meta
    ):
        gm.meta.update({"custom": mod.meta["custom"]})


def node_inline_(call_mod_node: torch.fx.Node) -> torch.fx.GraphModule | None:
    """
    Inline the submodule of the given node into the parent module.
    Note: we only support the case where submodule takes tensors inputs.
    """
    if call_mod_node.op != "call_module":
        raise AssertionError(f"expected call_module op, got {call_mod_node.op}")
    gm = call_mod_node.graph.owning_module
    if gm is None:
        raise AssertionError("owning_module should not be None")

    if not isinstance(call_mod_node.target, str):
        raise AssertionError(
            f"expected target to be str, got {type(call_mod_node.target).__name__}"
        )
    sub_gm = getattr(gm, call_mod_node.target)

    phs = (node for node in sub_gm.graph.nodes if node.op == "placeholder")
    body = (
        node for node in sub_gm.graph.nodes if node.op not in ("placeholder", "output")
    )
    output = [node for node in sub_gm.graph.nodes if node.op == "output"]

    for ph, arg in zip(phs, call_mod_node.args):
        if not isinstance(arg, torch.fx.Node):
            raise AssertionError(f"expected fx.Node, got {type(arg)}")
        node_replace_(ph, arg)

    with gm.graph.inserting_before(call_mod_node):
        for node in body:
            new_node = gm.graph.node_copy(node)
            if node.op == "get_attr":
                if not isinstance(new_node.target, str):
                    raise AssertionError(
                        f"Expected str target for get_attr, got {type(new_node.target)}"
                    )
                if not isinstance(node.target, str):
                    raise AssertionError(
                        f"Expected str target for get_attr, got {type(node.target)}"
                    )
                new_target_name = new_node.target
                if hasattr(gm, new_target_name):
                    # Loop through and find the "submod_{i}" that have no name collision
                    i = 1
                    new_target_name = f"submod_{i}"
                    while hasattr(gm, new_target_name):
                        i += 1
                        new_target_name = f"submod_{i}"
                new_node.target = new_target_name
                setattr(gm, new_node.target, getattr(sub_gm, node.target))
            node_replace_(node, new_node)

        if len(output) > 0:
            if len(output) != 1 or len(output[0].args) != 1:
                raise AssertionError(
                    f"expected exactly 1 output with 1 arg, got {len(output)} outputs"
                )
            new_output = output[0].args[0]

            if isinstance(new_output, torch.fx.Node):
                # Clear the users of the output node and set
                # the users to be the users of original call_module node.
                new_output.users.clear()
                node_replace_(call_mod_node, new_output)
            elif isinstance(new_output, (list, tuple)):
                # Pop subgraph output node from users.
                for node in new_output:
                    node.users.pop(output[0])

                # Inline the get_item calls for the output node.
                get_item_users = nodes_filter(
                    list(call_mod_node.users.keys()),
                    lambda node: node.op == "call_function"
                    and node.target is operator.getitem,
                )
                # get_item_node.args[1] is the idx referring to new_output[idx]
                nodes_map(
                    get_item_users,
                    lambda get_item_node: node_replace_(
                        get_item_node,
                        # pyrefly: ignore[bad-index]  # TODO getitem idx is an int FX arg
                        new_output[get_item_node.args[1]],
                    ),
                )
                call_mod_node.graph.erase_node(call_mod_node)
            else:
                raise NotImplementedError(
                    f"Unsupported output type {type(new_output)}. Expect it to be a Node or a list/tuple of Nodes."
                )
        else:
            call_mod_node.graph.erase_node(call_mod_node)

    gm.delete_all_unused_submodules()
    gm.recompile()
    return gm


def _get_torch_jit_trace_forward_signature(mod: torch.nn.Module) -> inspect.Signature:
    """
    Get source code and parse argument names using AST. The function returns
    a signature of the forward() function.

    # TODO: Directly provide inspect.signature compatible TS-d module.
    """
    ast_mod = ast.parse(mod.code)  # type: ignore[call-overload]
    ast_func_def: ast.FunctionDef = ast_mod.body[0]

    # FIXME(jiashenc): TorchScript should only allow positional or keywords arguments.
    arg_type_map = {"args": Parameter.POSITIONAL_OR_KEYWORD}

    # Traverse all argument types in AST tree and create associated parameters.
    param_list = []
    for arg_type, param_type in arg_type_map.items():
        arg_name_list = [a.arg for a in getattr(ast_func_def.args, arg_type)]
        for arg_name in arg_name_list:
            if arg_name == "self":
                continue  # Skip self argument.
            param_list.append(inspect.Parameter(arg_name, param_type))

    return inspect.Signature(parameters=param_list)


def _bind_signature_to_inputs(
    mod: torch.nn.Module, fake_args: tuple[Any, ...], fake_kwargs: dict[str, Any]
) -> dict[str, Any]:
    if isinstance(mod, (torch.jit.ScriptModule, torch.jit.TracedModule)):
        sig = _get_torch_jit_trace_forward_signature(mod)

        # Sanity check for placeholder names coming from TorchScript.
        if len(sig.parameters) != len(fake_args) + len(fake_kwargs):
            raise AssertionError(
                "Arguments other than POSITIONAL_OR_KEYWORD kinds in forward() "
                "are not supported in _get_torch_jit_trace_forward_signature"
            )
    else:
        sig = inspect.signature(mod.forward)

    # Rather than binding both fake_args and fake_kwargs to sig names, we
    # (partially) bind only fake_args, while reusing fake_kwarg names. This
    # ensures that fake_kwargs do not get reordered, which is important to
    # match flattened user inputs.
    return {**sig.bind_partial(*fake_args).arguments, **fake_kwargs}


def _build_cache(
    name: str, find_available: dict[str, int], used_names: set[str]
) -> None:
    used_names.add(name)
    match = re.match(r"(.*)_(\d+)", name)
    if match:
        prefix, n = match.group(1), match.group(2)
        if int(n) > find_available[prefix]:
            find_available[prefix] = int(n)


def _name_hoo_subgraph_placeholders(gm: torch.fx.GraphModule) -> None:
    """
    Propagate placeholder names from the top-level graph into HigherOrderOp subgraphs,
    and handle collisions with non-placeholders by count suffixing.
    Different HOO subgraph types have different input schemas, so we first enumerate them
    and gather the top-level named placeholder nodes.
    """

    # gather all HOO subgraphs and their top-level named placeholder nodes
    subgraph_ph_tuples: list[tuple[torch.fx.GraphModule, list[torch.fx.Node]]] = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            # HOO subgraphs have varying input schemas, so we enumerate them there
            if node.target._name == "cond":
                _, true_graph, false_graph, cond_args = node._args
                subgraph_ph_tuples.append((getattr(gm, true_graph.target), cond_args))
                subgraph_ph_tuples.append((getattr(gm, false_graph.target), cond_args))
            elif node.target._name == "wrap_with_set_grad_enabled":
                subgraph, phs = node._args[1], node._args[2:]
                subgraph_ph_tuples.append((getattr(gm, subgraph.target), phs))
            elif node.target._name == "map_impl":
                body_graph, array, args = node._args
                subgraph_ph_tuples.append(
                    (getattr(gm, body_graph.target), array + args)
                )

    # propagate names
    for subgraph, hoo_phs in subgraph_ph_tuples:
        name_map: dict[str, str] = {}
        find_available: dict[str, int] = defaultdict(int)
        used_names: set[str] = set()
        for i, node in enumerate(subgraph.graph.nodes):
            if i < len(hoo_phs):  # placeholder, retain name
                name_map[node.name] = hoo_phs[i].name
                node.name = node.target = hoo_phs[i].name
                _build_cache(node.name, find_available, used_names)
            else:  # non-placeholder, check for collisions
                node.name = _rename_without_collisions(
                    name_map, find_available, used_names, node.name, node.name
                )

        # recurse and recompile
        _name_hoo_subgraph_placeholders(subgraph)
        subgraph.recompile()


def _assign_new_node_names(
    gm: torch.fx.GraphModule,
    name_map: dict[str, str],
    custom_meta: dict[str, Any],
) -> None:
    """
    Assign new names to all nodes, in the graph module, from name map.
    """
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.name not in name_map:
                raise AssertionError(f"placeholder node {node.name!r} not in name_map")
            node.name = node.target = name_map[node.name]
            if node.name in custom_meta:
                if node.meta.get("custom") is None:
                    empty_custom: dict[str, Any] = {}
                    node.meta["custom"] = empty_custom
                else:
                    # Assert if any existing key has different value
                    for k, v in node.meta["custom"].items():
                        if (
                            k in custom_meta[node.name]
                            and v != custom_meta[node.name][k]
                        ):
                            raise AssertionError(
                                f"Mismatch in custom metadata for key {k}. Value in "
                                f"node.meta is {v} and value in custom_meta is {custom_meta[node.name][k]}."
                            )
                node.meta["custom"].update(custom_meta[node.name])
            # if the constant obj is an input, we also need to update meta["val"]
            # because this is created before the placeholder naming pass
            if isinstance(node.meta["val"], CustomObjArgument):
                node.meta["val"].name = node.name
        elif node.name in name_map:
            node.name = name_map[node.name]


def placeholder_naming_pass(
    gm: torch.fx.GraphModule,
    export_graph_signature: "ExportGraphSignature",
    mod: torch.nn.Module,
    fake_args: tuple[Any, ...],
    fake_kwargs: dict[str, Any],
    fake_params_buffers: dict[str, torch.Tensor | torch.nn.Parameter],
    constants: dict[str, Any],
) -> None:
    """
    This pass is run at the end of _export_non_strict() to assign better placeholder node names:
        - User inputs:
            These follow the signature of mod.forward(), e.g. forward(x, y) produces nodes x, y.
            For nested inputs from dictionaries, lists, tuples, or dataclasses,
            the names are a concatenation of the path to the tensor.
                e.g. x = {
                    'a': torch.randn(),
                    'b': [torch.randn(), torch.randn()]
                }
            produces nodes x_a, x_b_0, x_b_1.
        - Parameters/buffers/constants/custom objects:
            These follow the FQN of the object, prefixed by "p", "b", "c", "obj" respectively.
                e.g. self.bar.l0.weight produces "p_bar_l0_weight".
        - Effect tokens:
            These are named token, token_1, ...
    """

    custom_meta: dict[str, Any] = {}
    if isinstance(mod, torch.fx.GraphModule):
        for node in mod.graph.nodes:
            if "custom" in node.meta:
                custom_meta[node.name] = node.meta["custom"]

    def _strip_name(x: str) -> str:
        if x.startswith("L__self___"):
            x = x[len("L__self___") :]
        elif x.startswith("self_"):
            x = x[len("self_") :]
        x = re.sub(r"[^a-zA-Z0-9]", "_", x)
        return x

    def _extract_pytree_key(x: object) -> str:
        if isinstance(x, MappingKey):
            x = re.sub(r"[^a-zA-Z0-9]", "_", str(x.key))
            return x
        elif isinstance(x, SequenceKey):
            return str(x.idx)
        elif isinstance(x, GetAttrKey):
            return x.name
        else:
            raise RuntimeError(f"Pytree key of type {type(x)} not handled for {x}")

    name_map: dict[str, str] = {}
    find_available: dict[str, int] = defaultdict(int)
    used_names: set[str] = set()

    # map user input names with mod.forward() signature
    combined_args = _bind_signature_to_inputs(mod, fake_args, fake_kwargs)

    flat_args_with_path, _ = tree_flatten_with_path(combined_args)
    user_input_names = [
        spec.arg.name
        for spec in export_graph_signature.input_specs
        if spec.kind == InputKind.USER_INPUT
    ]

    # use pytree path to name nested user inputs
    for (arg_path, _arg), user_input_name in zip(flat_args_with_path, user_input_names):
        if user_input_name:
            _rename_without_collisions(
                name_map,
                find_available,
                used_names,
                user_input_name,
                placeholder_prefixes[InputKind.USER_INPUT]
                + "_".join(_extract_pytree_key(x).lower() for x in arg_path),
                is_placeholder=True,
            )

    # use graph signature input specs to map param/buffer/constant names
    # name effect tokens as token, token_1, ... (these aren't visible to user)
    for spec in export_graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            continue
        if spec.kind == InputKind.TOKEN:
            base_name = ""
        else:
            # pyrefly: ignore[bad-argument-type]  # TODO non-token specs always have a str target
            base_name = _strip_name(spec.target).lower()
        base_name = re.sub(r"[^a-zA-Z0-9]", "_", base_name)

        _rename_without_collisions(
            name_map,
            find_available,
            used_names,
            spec.arg.name,
            placeholder_prefixes[spec.kind] + base_name,
            is_placeholder=True,
        )
        if base_name in custom_meta:
            # the keys in custom_meta are node names from `mod`,
            # which is the base_name here.
            # we need the re-mapped name for lookup later
            custom_meta[name_map[spec.arg.name]] = custom_meta[base_name]
            del custom_meta[base_name]

    # handle naming collisions with call_function/get_attr inputs.
    # here, we want to prioritize user input names over call_function names
    # e.g. not have forward(self, mul): lead to a placeholder node called mul_13,
    # so we increment the suffix of call_function nodes as needed
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        _rename_without_collisions(
            name_map, find_available, used_names, node.name, node.name
        )

    # assign new node names
    _assign_new_node_names(gm, name_map, custom_meta)

    # propagate names to higher order op subgraphs
    _name_hoo_subgraph_placeholders(gm)

    # re-generate graph module code
    gm.recompile()

    # modify graph signature (input specs, output specs, user input mutations)
    for spec in export_graph_signature.input_specs:
        if spec.arg.name not in name_map:
            raise AssertionError(f"input spec arg {spec.arg.name!r} not in name_map")
        spec.arg.name = name_map[spec.arg.name]
        if (  # handle targets for custom objects
            spec.kind == InputKind.CUSTOM_OBJ and spec.target in name_map
        ):
            # pyrefly: ignore [bad-index, index-error]
            spec.target = name_map[spec.target][4:]  # strip obj_ prefix

    for spec in export_graph_signature.output_specs:
        if spec.arg.name in name_map:
            spec.arg.name = name_map[spec.arg.name]
        if spec.kind == OutputKind.USER_INPUT_MUTATION and spec.target in name_map:
            # pyrefly: ignore [bad-index, index-error]
            spec.target = name_map[spec.target]

    # rename keys in constants dict for custom objects
    for name in list(constants.keys()):
        constant = constants[name]
        if name in name_map and not isinstance(
            constant, torch.Tensor
        ):  # rename custom objects with generic names
            new_name = name_map[name]
            if (
                new_name != name
                and re.match(r"arg(\d+)_1", name)
                and new_name != placeholder_prefixes[InputKind.CUSTOM_OBJ] + name
            ):
                constants[new_name] = constant
                del constants[name]


def remove_proxy_from_state_dict(
    state_dict: dict[str, Any], in_place: bool
) -> dict[str, Any]:
    """
    If `in_place` is false, return a new copy of `state_dict` with "proxy" removed from `v.__dict__`.
    `v` is the values in the dictionary.
    If `in_place` is true, modify `state_dict` in place.
    """
    if in_place:
        for k, v in state_dict.items():
            if hasattr(v, "proxy"):
                delattr(state_dict[k], "proxy")
        return state_dict
    else:
        new_state_dict = {}
        for k, v in state_dict.items():
            if hasattr(v, "proxy"):
                new_state_dict[k] = v.detach().clone()
            else:
                new_state_dict[k] = v
        return new_state_dict


def _detect_fake_mode_from_gm(
    gm: torch.fx.GraphModule,
) -> torch._subclasses.fake_tensor.FakeTensorMode | None:
    """
    For a given graph module, we look at the "val" of placeholder nodes to find the fake inputs.
    Additionally, if gm doesn't have placeholders, we further look at the "example_value" or "val" of other nodes.
    If no fake mode is found, we return None for fake_mode.
    """

    fake_inps: list[torch.Tensor] = []
    fake_vals: list[torch.Tensor] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder" and "val" in node.meta:
            fake_val = node.meta["val"]
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_inps.append(fake_val)
        elif len(fake_inps) == 0 and (
            "example_value" in node.meta or "val" in node.meta
        ):
            fake_val = None
            if "example_value" in node.meta:
                fake_val = node.meta["example_value"]
            elif "val" in node.meta:
                fake_val = node.meta["val"]
            if fake_val is not None and isinstance(fake_val, torch.Tensor):
                fake_vals.append(fake_val)

    return detect_fake_mode(fake_inps + fake_vals)


@contextmanager
def _disable_load_state_dict_hooks(
    mod: torch.nn.Module,
) -> Generator[None, None, None]:
    state_dict_hooks: dict[int, Callable[..., object]] = dict(mod._state_dict_hooks)
    state_dict_pre_hooks: dict[int, Callable[..., object]] = dict(
        mod._state_dict_pre_hooks
    )
    mod._state_dict_hooks.clear()
    mod._state_dict_pre_hooks.clear()
    try:
        yield
    finally:
        mod._state_dict_hooks = state_dict_hooks
        mod._state_dict_pre_hooks = state_dict_pre_hooks


def _is_cia_op(op: "OperatorBase") -> bool:
    return (
        torch._C._dispatch_has_kernel_for_dispatch_key(
            op.name(), torch._C.DispatchKey.CompositeImplicitAutograd
        )
        or torch._C.DispatchKey.CompositeImplicitAutograd in op.py_kernels
    )


def _is_preservable_cia_op(op: "OperatorBase") -> bool:
    return _check_valid_to_preserve(op) and _is_cia_op(op)


def _is_aten_op(op: "OperatorBase") -> bool:
    return op.name().split("::")[0] == "aten"


def _is_custom_op(op: "OperatorBase") -> bool:
    return not _is_aten_op(op)


# We can't cache this because custom op registry API in python can still
# add entries to the C++ dispatcher.
def _materialize_cpp_cia_ops() -> None:
    """
    Utility function to query C++ dispatcher to get the all
    possible CIA ops and populate them into torch.ops namespace
    """
    cia_ops = torch._C._dispatch_get_registrations_for_dispatch_key(
        "CompositeImplicitAutograd"
    )

    # Materialize all CIA ops
    for op in cia_ops:
        namespace, op_name = tuple(op.split("::"))
        split_list = op_name.split(".")
        # Sometime overload could be missing
        if len(split_list) not in (1, 2):
            raise AssertionError(f"expected 1 or 2 parts, got {len(split_list)}")
        op_name = split_list[0]
        op_overload_name = "default"
        if len(split_list) == 2:
            op_overload_name = split_list[1]

        _ = getattr(getattr(getattr(torch.ops, namespace), op_name), op_overload_name)


def _special_op_to_preserve_cia(*args: Any, **kwargs: Any) -> Any:
    """
    This is a special marker that tells our infra that we shouldn't decompose this op.
    """
    return NotImplemented


# Our strategy for deciding if we can preserve a op is following:
# 1. The op should be known statically that it is functional
# 2. If it is maybe aliasing, we decompose because we must know if an op
#    is mutating or aliasing.
def _check_valid_to_preserve(op_overload: "OperatorBase") -> bool:
    from torch._decomp import _should_decompose_because_unsafe_op

    if _should_decompose_because_unsafe_op(op_overload):
        return False
    if op_overload in FunctionalTensor.metadata_fns:
        return False

    if not hasattr(op_overload, "_schema"):
        return False

    alias_info = len(
        [i for i in op_overload._schema.arguments if i.alias_info is not None]
    )

    is_mutating_or_aliasing = alias_info != 0 or op_overload._schema.is_mutable

    if is_mutating_or_aliasing:
        return False

    if not torch._C._dispatch_has_kernel(op_overload.name()):
        return False

    return True


@functools.lru_cache(maxsize=1)
def _collect_all_valid_cia_ops_for_aten_namespace() -> set["OperatorBase"]:
    return _collect_all_valid_cia_ops_for_namespace(torch.ops.aten)


def _collect_all_valid_cia_ops_for_namespace(
    op_namespace: torch._ops._OpNamespace,
) -> set["OperatorBase"]:
    # Step 1: Materialize all ops from C++ dispatcher
    _materialize_cpp_cia_ops()

    # Step 2: Query all ops from python dispatcher
    cia_ops = set()
    for op in op_namespace:
        op_packet = getattr(op_namespace, op)
        for op_overload in op_packet.op_overloads():
            if _is_preservable_cia_op(op_overload):
                cia_ops.add(op_overload)
    return cia_ops


def _collect_all_valid_cia_ops() -> set["OperatorBase"]:
    """
    This is an util function that gets the all CIA functional ops.

    The algorithm is in 2 steps:
      1. We first query C++ dispatcher to get the list of CIA ops
         and then we call getattr on torch.ops.aten to lazily populate
         them.

      2. Sometimes, handful of ops have CIA registered in python dispatcher
         but not on the C++ side, these can't be caught at the first step.
         So we walk again to get the final list.

    Note that the output of this function should never be modified
    """
    cia_ops = set()
    for op_namespace_name in torch.ops._dir:
        # The reason we split here is because aten ops are safe to cache.
        if op_namespace_name != "aten":
            if not hasattr(torch.ops, op_namespace_name):
                raise AssertionError(
                    f"torch.ops does not have attribute {op_namespace_name!r}"
                )
            op_namespace = getattr(torch.ops, op_namespace_name)
            if isinstance(op_namespace, torch._ops._OpNamespace):
                cia_ops |= _collect_all_valid_cia_ops_for_namespace(op_namespace)
        else:
            cia_ops |= _collect_all_valid_cia_ops_for_aten_namespace()
    return cia_ops


def _get_decomp_for_cia(op: "OperatorBase") -> Callable[..., object]:
    # [NOTE] Separating out func.decompose
    # Ideally we should be able to just register func.decompose but
    # we can't as this decomp is gonna be registered to the py_impl.
    # As a result it will infinitely recurse. So we first check if the op
    # has py_impl entry for CIA and if it is we use that first. If not,
    # we register C++ query to py_impl.
    dk = torch._C.DispatchKey.CompositeImplicitAutograd
    if dk in op.py_kernels and not isinstance(op.py_kernels[dk], torch._C.DispatchKey):
        return op.py_kernels[dk]

    def _special_op_to_decompose_cia(*args: Any, **kwargs: Any) -> Any:
        kernel = kwargs["kernel"]
        del kwargs["kernel"]
        # Can't call kernel.decompose due to infinite recursion as
        # we register this kernel to py_impl directly
        dk = torch._C.DispatchKey.CompositeImplicitAutograd
        if torch._C._dispatch_has_kernel_for_dispatch_key(
            kernel.name(), torch._C.DispatchKey.CompositeImplicitAutograd
        ):
            return kernel._op_dk(dk, *args, **kwargs)
        else:
            raise AssertionError(
                f"Expected {kernel} to have CompositeImplicitAutograd kernel"
            )

    return functools.partial(_special_op_to_decompose_cia, kernel=op)


@contextmanager
def _compiling_state_context() -> Generator[None, None, None]:
    old_compiling_flag = torch.compiler._is_compiling_flag
    old_exporting_flag = torch.compiler._is_exporting_flag
    try:
        torch.compiler._is_compiling_flag = True
        torch.compiler._is_exporting_flag = True
        yield
    finally:
