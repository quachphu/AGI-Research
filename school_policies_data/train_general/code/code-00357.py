        return _ScaledMatmul(
            nodes=match,
            A_node=A_node,
            B_node=B_node,
            A_scale_node=A_scale_node,
            B_scale_node=B_scale_node,
            bias_node=get_arg(mm_node, 4, None),
            result_scale_node=get_arg(mm_node, 5, None),
            out_dtype=get_arg(mm_node, 6, None),
            use_fast_accum=get_arg(mm_node, 7, False),
            pre_mm_reshape=pre_mm_reshape,
            post_mm_reshape=post_mm_reshape,
        )


def _find_reshape_mm_reshape(node: torch.fx.Node) -> list[_Matmul]:
    if node.target != aten.reshape.default:
        return []

    matches = []
    for mm_node in node.users:
        if mm_node.target not in (aten.mm.default, aten._scaled_mm.default):
            continue
        for reshape_node in mm_node.users:
            if reshape_node.target != aten.reshape.default:
                continue

            # Since the reshape -> mm -> reshape pattern would be subsumed into
            # the fused op, we only match the patterns where the shape of the
            # second reshape is matches the mm result produced by the fused op.
            matmul_input_node = cast("torch.fx.Node", node.args[0])
            B_node = cast("torch.fx.Node", mm_node.args[1])
            matmul_out_shape = torch.Size(
                [
                    *_get_tensor(matmul_input_node).shape[:-1],
                    _get_tensor(B_node).shape[-1],
                ]
            )
            if _get_tensor(reshape_node).shape != matmul_out_shape:
                continue
            matches.append([node, mm_node, reshape_node])
            # If for some rare reason mm_node is being reshaped by two
            # different reshape nodes, we only include mm_node once in the
            # parsing result.
            break

    matmuls = []
    for match in matches:
        mm_node = match[1]
        if mm_node.target is aten.mm.default:
            matmul = _Matmul.from_match(match)
            matmuls.append(matmul)
        elif mm_node.target is aten._scaled_mm.default:
            matmul = _ScaledMatmul.from_match(match)
            matmuls.append(matmul)
        else:
            raise AssertionError(
                "Expect the node's target to be either aten.mm.default or "
                f"aten._scaled_mm.default. Got {mm_node.target}."
            )
    return matmuls


def _find_consumer_matmuls(node: torch.fx.Node) -> list[_Matmul]:
    """
    Find the matmuls that use `node` as the lhs argument.
    """
    matmuls = []
    for user in node.users:
        # ND matmuls
        if user.target is aten.reshape.default:
            matmuls.extend(_find_reshape_mm_reshape(user))
        # 2D matmuls
        elif user.target is aten.mm.default:
            matmul = _Matmul.from_match(match=[user])
            matmuls.append(matmul)
        elif user.target is aten._scaled_mm.default:
            matmul = _ScaledMatmul.from_match([user])
            matmuls.append(matmul)
    return matmuls


def _insert_fused_all_gather_matmul(
    graph: torch.fx.Graph,
    matmuls: list[_Matmul],
    shard_node: torch.fx.Node,
    gather_dim: int,
    group_name: "torch.distributed.distributed_c10d.GroupName",
) -> torch.fx.Node:
    mm_types = OrderedSet(map(type, matmuls))
    if len(mm_types) != 1:
        raise AssertionError(f"expected exactly one matmul type, got {len(mm_types)}")
    mm_type = next(iter(mm_types))
    if mm_type == _Matmul:
        B_nodes = [matmul.B_node for matmul in matmuls]
        return graph.call_function(
            torch.ops.symm_mem.fused_all_gather_matmul.default,
            args=(shard_node, B_nodes, gather_dim, group_name),
            kwargs={"return_A": True},
        )
    elif mm_type == _ScaledMatmul:
        scaled_matmuls = cast("list[_ScaledMatmul]", matmuls)
        return graph.call_function(
            torch.ops.symm_mem.fused_all_gather_scaled_matmul.default,
            args=(
                shard_node,
                [matmul.B_node for matmul in scaled_matmuls],
                scaled_matmuls[0].A_scale_node,
                [matmul.B_scale_node for matmul in scaled_matmuls],
                gather_dim,
                group_name,
                [matmul.bias_node for matmul in scaled_matmuls],
                [matmul.result_scale_node for matmul in scaled_matmuls],
                [matmul.out_dtype for matmul in scaled_matmuls],
                [matmul.use_fast_accum for matmul in scaled_matmuls],
            ),
        )
    else:
        raise AssertionError(f"Unexpected matmul match type: {mm_type}")


def fuse_all_gather_matmul(all_gather: _AllGatherMatch) -> None:
    """
    Fused the pattern

        A = all_gather_single(A_shard, gather_dim, group_name)
        C_0 = torch.matmul(A, B_0)
        C_1 = torch.matmul(A, B_1)
        C_2 = torch.matmul(A, B_2)
        ...

    into

        A, Cs = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, [B_0, B_1, B_2, ...], gather_dim, group_name,
        )
    """
    if not (
        torch.distributed.is_available()
        and (
            torch.distributed.is_nccl_available()
            or torch.distributed.is_xccl_available()
        )
    ):
        return

    from torch.distributed._symmetric_memory import (
        is_symm_mem_enabled_for_group,
        restride_A_shard_for_fused_all_gather_matmul,
    )

    shard_node, ag_node, ag_res_node, gather_dim, group_name = (
        all_gather.shard_node,
        all_gather.ag_node,
        all_gather.res_node,
        all_gather.gather_dim,
        all_gather.group_name,
    )

    if not is_symm_mem_enabled_for_group(group_name):
        return

    filter_matmul = None
    if _is_last_dim(_get_tensor(shard_node), gather_dim):
        # Decomposed mms should not be too small
        if _get_tensor(shard_node).shape[-1] < 1024:
            return

        # scaled_mm is not supported yet for last dim
        def _filter_out_scaled_matmul(matmul: _Matmul):
            return not isinstance(matmul, _ScaledMatmul)

        filter_matmul = _filter_out_scaled_matmul

    # Find consumer matmuls
    matmuls = _find_consumer_matmuls(ag_res_node)

    # The matmuls are only fusible if non-A args don't depend on the all-gather
    # result node
    matmuls = [
        matmul
        for matmul in matmuls
        if all_gather.res_node not in matmul.arg_ancestor_nodes
    ]

    if len(matmuls) == 0 or len(OrderedSet(map(type, matmuls))) != 1:
        return

    if _is_last_dim(_get_tensor(shard_node), gather_dim) and len(
        all_gather.res_node.users
    ) > len(matmuls):
        # The result of ag-split-cat is used not only in matmuls.
        # Then it has to be materialized, which can have overhead.
        return

    if filter_matmul and not filter_matmul(matmuls[0]):
        return

    # Fuse the all_gather_single with the eligible matmuls
    graph = ag_node.graph
    with graph.inserting_before(ag_node):
        if not _is_last_dim(_get_tensor(shard_node), gather_dim):
            if "val" in shard_node.meta:
                restrided = restride_A_shard_for_fused_all_gather_matmul(
                    _get_tensor(shard_node),
                    gather_dim,
                )
                shard_node = graph.call_function(
                    inductor_prims.force_stride_order,
                    args=(shard_node, restrided.stride()),
                )

        fused_node = _insert_fused_all_gather_matmul(
            graph, matmuls, shard_node, gather_dim, group_name
        )
        new_ag_node = graph.call_function(
            operator.getitem,
            args=(fused_node, 0),
        )
        new_out_nodes = graph.call_function(
            operator.getitem,
            args=(fused_node, 1),
        )
        for idx, matmul in enumerate(matmuls):
            new_out_node = graph.call_function(
                operator.getitem,
                args=(new_out_nodes, idx),
            )
            matmul.replace_with(new_out_node)
            matmul.erase()
        all_gather.replace_with(new_ag_node)
        all_gather.erase()

        # If the new_ag_node has no users, we tell the fused op to not return
        # it. This creates more optimization opportunities.
        if len(new_ag_node.users) == 0:
            graph.erase_node(new_ag_node)
            kwargs = dict(fused_node.kwargs)
            if "return_A" in kwargs:
                kwargs["return_A"] = False
                fused_node.kwargs = kwargs

    # Raise ancestors of non-A args that are topologically ordered between
    # ag_res_node and the matmul above fused_node.
    order = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes_to_raise = sorted(
        OrderedSet(x for matmul in matmuls for x in matmul.arg_ancestor_nodes),
        key=lambda x: order[x],
    )
    for node in nodes_to_raise:
        if order[node] > order[fused_node]:
            fused_node.prepend(node)


def _scatter_dim_after_reshape(
    reshape_node: torch.fx.Node, orig_scatter_dim: int
) -> int:
    """
    Given a reshape node and the original scatter dim for the target tensor,
    returns the new scatter dim for the reshaped tensor.
    """
    # if there was no pre-mm reshape, scatter dim will not change.
    if not reshape_node:
        return orig_scatter_dim

    reshape_op_output_tensor = _get_tensor(reshape_node)
    if reshape_op_output_tensor.ndim != 2:
        raise AssertionError("reshape must produce 2D tensor for scaled_mm")

    if len(reshape_node.args) < 1:
        raise AssertionError("reshape node must have at least 1 arg")
    input_tensor_node = cast(torch.fx.Node, reshape_node.args[0])
    reshape_op_input_tensor = _get_tensor(input_tensor_node)
    if reshape_op_input_tensor.ndim <= reshape_op_output_tensor.ndim:
        raise AssertionError("reshape must be from 3D+ to 2D")

    # Note: for a N-D tensor to be reshaped into 2D, either the leading dims or ending dims must
    # be collapsed to a single dim. First determine which of these happened.
    input_shape = reshape_op_input_tensor.shape
    output_shape = reshape_op_output_tensor.shape
    leading_dims_collapsed = output_shape[0] == prod(input_shape[:-1])

    # Case 1: scatter dim 0 always maps to 0 after any reshape from 3D+ to 2D, regardless if
    # leading dims or ending dims were collapsed.
    if orig_scatter_dim == 0:
        return 0

    # Case 2: scatter dim "ndim-1" always maps to 1 after any reshape from 3D+ to 2D, regardless if
    # leading dims or ending dims were collapsed.
    if orig_scatter_dim == reshape_op_input_tensor.ndim - 1:
        return 1

    # Case 3: scatter dim was one of the middle dims (between 0 and ndim-1).
    # if the leading dims were collapsed, the new scatter dim will be 0.
    # if the ending dims were collapsed, the new scatter dim will be 1.
    return 0 if leading_dims_collapsed else 1


def _find_producer_matmul(node: torch.fx.Node) -> _Matmul | None:
    """
    Returns producer matmul node if found, otherwise returns None.
    """
    if node.target is aten.mm.default:
        return _Matmul.from_match(match=[node])
    elif node.target is aten._scaled_mm.default:
        return _ScaledMatmul.from_match(match=[node])
    elif node.target is aten.reshape.default:
        reshape_node_1 = node

        mm_node = reshape_node_1.args[0]
        if not isinstance(mm_node, torch.fx.Node):
            raise AssertionError(f"expected an fx Node, got {type(mm_node)}")
        if mm_node.target not in (aten.mm.default, aten._scaled_mm.default):
            return None

        reshape_node_0 = mm_node.args[0]
        if not isinstance(reshape_node_0, torch.fx.Node):
            raise AssertionError(f"expected an fx Node, got {type(reshape_node_0)}")
        if reshape_node_0.target != aten.reshape.default:
            return None

        if mm_node.target is aten.mm.default:
            return _Matmul.from_match(match=[reshape_node_0, mm_node, reshape_node_1])
        elif mm_node.target is aten._scaled_mm.default:
            return _ScaledMatmul.from_match(
                match=[reshape_node_0, mm_node, reshape_node_1]
            )
    return None


def _insert_fused_matmul_reduce_scatter(
    graph: torch.fx.Graph,
    matmul: _Matmul,
    reduce_op: str,
    orig_scatter_dim: int,
    group_name: "torch.distributed.distributed_c10d.GroupName",
    scatter_dim_after_reshape: int,  # only used for reshape -> scaled_mm -> reshape pattern
    output_shape: list[int],  # only used for reshape -> scaled_mm -> reshape pattern
) -> torch.fx.Node:
    if type(matmul) is _Matmul:
        return graph.call_function(
            torch.ops.symm_mem.fused_matmul_reduce_scatter.default,
            args=(
                matmul.A_node,
                matmul.B_node,
                reduce_op,
                orig_scatter_dim,
                group_name,
            ),
        )
    elif type(matmul) is _ScaledMatmul:
        return graph.call_function(
            torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter.default,
            args=(
                matmul.A_node,
                matmul.B_node,
                matmul.A_scale_node,
                matmul.B_scale_node,
                reduce_op,
                orig_scatter_dim,
                scatter_dim_after_reshape,
                group_name,
                output_shape,
                matmul.bias_node,
                matmul.result_scale_node,
                matmul.out_dtype,
                matmul.use_fast_accum,
            ),
        )
    else:
        raise AssertionError(f"Unexpected matmul match type: {type(matmul)}")


def fuse_matmul_reduce_scatter(reduce_scatter: _ReduceScatterMatch) -> None:
    """
    Fused the pattern

        reduce_scatter_single(A @ B, scatter_dim, group_name)

    into

        torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, scatter_dim, group_name,
        )

    Returns boolean indicating if fusion was successful or not.
    """
    if not (
        torch.distributed.is_available()
        and (
            torch.distributed.is_nccl_available()
            or torch.distributed.is_xccl_available()
        )
    ):
        return

    from torch.distributed._symmetric_memory import (
        is_symm_mem_enabled_for_group,
        restride_A_for_fused_matmul_reduce_scatter,
    )

    (
        input_node,
        _reduce_scatter_node,
        rs_wait_tensor_node,
        reduce_op,
        orig_scatter_dim,
        group_name,
    ) = (
        reduce_scatter.input_node,
        reduce_scatter.reduce_scatter_node,
        reduce_scatter.wait_tensor_node,
        reduce_scatter.reduce_op,
        reduce_scatter.scatter_dim,
        reduce_scatter.group_name,
    )

    if not is_symm_mem_enabled_for_group(group_name):
        return

    filter_matmul = None
    if _is_last_dim(_get_tensor(input_node), orig_scatter_dim):
        # scaled_mm is not supported yet for last dim mm+rs
        def _filter_out_scaled_matmul(matmul: _Matmul):
            return not isinstance(matmul, _ScaledMatmul)

        filter_matmul = _filter_out_scaled_matmul

    matmul = _find_producer_matmul(input_node)
    if matmul is None:
        log.debug(
            "no producer matmul found for reduce scatter, skipping fuse_matmul_reduce_scatter fusion"
        )
        return

    # Currently fused_matmul_reduce_scatter doesn't return the matmul result,
    # so we can't apply the fusion if the matmul result is used by multiple
    # users. This is not a fundamental limitation of the fused op and can be
    # addressed if needed.
    match_nodes = (
        reduce_scatter.match.nodes
        if isinstance(reduce_scatter.match, Match)
        else reduce_scatter.match
    )
    if len(input_node.users) != 1 and not all(
        user in match_nodes for user in input_node.users
    ):
        log.warning(
            "matmul result has more than one user, skipping fused_matmul_reduce_scatter fusion."
        )
        return

    if filter_matmul and not filter_matmul(matmul):
        return

    if rs_wait_tensor_node in matmul.arg_ancestor_nodes:
        log.warning(
            "reduce-scatter result node is an ancestor of matmul, skipping fuse_matmul_reduce_scatter fusion"
        )
        return

    # We need to track 3 values for the fused scaled mm reduce scatter implementation:
    #   1. The scatter dim before the reshape, which was assigned using the original (a,b,c) @ (c,d) = (a,b,d) dims.
    #   2. The scatter dim after the reshape, to use when we are doing the 2D (a*b,c) @ (c,d) = (a,b,d) scaled mm op.
    #   3. Store expected potentially 3D+ mm output shape, so we can reshape the 2D mm output to the intended
    #      3D+ shape before applying reduce-scatter, and to prevent shape errors with subsequent ops.

    # If 'A' was reshaped from 3D+ -> 2D for the mm, we need to determine the new scattter dim after the reshape
    # for the fused matmul reduce scatter implementation to use.
    if matmul.pre_mm_reshape:
        scatter_dim_after_maybe_reshape = _scatter_dim_after_reshape(
            matmul.pre_mm_reshape, orig_scatter_dim
        )
    else:
        scatter_dim_after_maybe_reshape = orig_scatter_dim

    # If the 2D mm output was reshaped from 2D -> 3D+, we need to store the intended output shape for the
    # fused matmul reduce scatter implementation to use.
    if matmul.post_mm_reshape:
        output_shape = list(_get_tensor(matmul.post_mm_reshape).shape)
    else:
        A_orig_shape = list(_get_tensor(matmul.A_node).shape)
        B_shape = list(_get_tensor(matmul.B_node).shape)
        output_shape = [*A_orig_shape[:-1], B_shape[-1]]

    graph = rs_wait_tensor_node.graph
    with graph.inserting_before(rs_wait_tensor_node):
        # Restride A tensor before fused op, for optimal perf in fused matmul reduce scatter
        if "val" in matmul.A_node.meta:
            restrided = restride_A_for_fused_matmul_reduce_scatter(
                _get_tensor(matmul.A_node),
                scatter_dim_after_maybe_reshape,
            )
            matmul.A_node = graph.call_function(
                inductor_prims.force_stride_order,
                args=(matmul.A_node, restrided.stride()),
            )

        # Replace matched subgraph with fused matmul reduce scatter node
        fused_node = _insert_fused_matmul_reduce_scatter(
            graph,
            matmul,
            reduce_op,
            orig_scatter_dim,
            group_name,
            scatter_dim_after_maybe_reshape,
            output_shape,
        )
        reduce_scatter.replace_with(fused_node)
        reduce_scatter.erase()
        matmul.erase()

    order = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes_to_raise = sorted(
        matmul.arg_ancestor_nodes,
        key=lambda x: order[x],
    )
    for node in nodes_to_raise:
        if order[node] > order[fused_node]:
            fused_node.prepend(node)

    log.debug("successfully fused matmul reduce scatter")


def _get_node_to_ancestors(
    graph: torch.fx.Graph,
) -> dict[torch.fx.Node, OrderedSet[torch.fx.Node]]:
    """
    Compute the ancestors for all nodes in a graph.
    """
    node_to_ancestors = defaultdict(OrderedSet[torch.fx.Node])  # type: ignore[var-annotated]
    for node in graph.nodes:
        node_to_ancestors[node] = OrderedSet(node.all_input_nodes)
        for dep in node.all_input_nodes:
            node_to_ancestors[node] |= node_to_ancestors[dep]

    return node_to_ancestors


def _get_collective_to_overlappable_nodes(
    graph: torch.fx.Graph,
) -> dict[torch.fx.Node, list[torch.fx.Node]]:
    """
    For each collective in the graph, find nodes that are neither ancestors nor
    descendants of the collective.
    """

    def is_collective(node) -> bool:
        # Only consider all-gather and reduce-scatter in the context of
        # micro-pipeline TP.
        return node.target in [
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
        ]

    node_to_ancestors = _get_node_to_ancestors(graph)
    collective_to_overlappable_nodes = defaultdict(list)
    for node in graph.nodes:
        if not is_collective(node):
            continue
        for x in graph.nodes:
            if (
                node not in node_to_ancestors[x]
                and x not in node_to_ancestors[node]
                and x.op == "call_function"
            ):
                collective_to_overlappable_nodes[node].append(x)

    return collective_to_overlappable_nodes


def _get_unexposed_collectives(graph: torch.fx.Graph) -> list[torch.fx.Node]:
    """
    Find all unexposed collectives in the graph.

    Because we don't have the runtime estimate, this function is a rough
    estimation using the following strong/hand-wavy assumptions:

    - Only a predefined set of "compute intensive" operation can hide a collective.
    - Any "compute intensive" operation can hide exactly one collective.
    """

    def _is_compute_intensive(node: torch.fx.Node) -> bool:
        return node.target is torch.ops.aten.mm.default

    collective_to_overlapping_candidates = defaultdict(list)
    available_nodes = OrderedSet[torch.fx.Node]()
    collective_to_overlappable_nodes = _get_collective_to_overlappable_nodes(graph)
    for collective, overlappable_nodes in collective_to_overlappable_nodes.items():
        candidates = [x for x in overlappable_nodes if _is_compute_intensive(x)]
        collective_to_overlapping_candidates[collective] = candidates
        available_nodes.update(candidates)

    unexposed_collectives = []
    for (
        collective,
        overlapping_candidates,
    ) in collective_to_overlapping_candidates.items():
        # Each collective consumes exactly one overlapping candidate
        for x in overlapping_candidates:
            if x in available_nodes:
                unexposed_collectives.append(collective)
                available_nodes.remove(x)
                break
    return unexposed_collectives


def micro_pipeline_tp_pass(graph: torch.fx.Graph):
    all_gathers = find_all_gather_patterns(graph)
    reduce_scatters = find_reduce_scatter_patterns(graph)

    # When a collective can be hidden through either simple overlapping or
    # micro-pipeline TP, we prefer simple overlapping to avoid the overhead
    # associated with decomposition. If reorder_for_compute_comm_overlap is
    # enabled, we identify collectives that can be hidden through simple
    # overlapping and exclude them from micro-pipeline TP candidates.
    if config.reorder_for_compute_comm_overlap:
        unexposed_collectives = _get_unexposed_collectives(graph)
        all_gathers = [x for x in all_gathers if x.ag_node not in unexposed_collectives]
        reduce_scatters = [
            x
            for x in reduce_scatters
            if x.reduce_scatter_node not in unexposed_collectives
        ]

    if not all_gathers and not reduce_scatters:
        log.warning(
            "async TP found no matching all-gather/reduce-scatter patterns for fusion"
        )

    for all_gather in all_gathers:
        fuse_all_gather_matmul(all_gather)

    for reduce_scatter in reduce_scatters:
        fuse_matmul_reduce_scatter(reduce_scatter)
<|endoftext|>import json
import logging
import math
from collections.abc import Callable

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import (
    bucket_all_gather_by_mb,
    bucket_all_reduce_by_mb,
    bucket_reduce_scatter_by_mb,
    BucketMode,
    is_all_gather_into_tensor as is_all_gather,
    is_all_reduce_tensor,
    merge_all_gather,
    merge_all_reduce_bucket,
    merge_reduce_scatter,
)
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_fsdp_all_gather(n: torch.fx.Node) -> bool:
    """Check if an all_gather derives from exactly one placeholder (parameter).

    Uses backward BFS to count placeholder ancestors across all input branches.
    Handles multi-input chains (e.g. cat(param, zeros) for padding) that the old
    single-input-chain walk would miss.
    """
    if not is_all_gather(n):
        raise AssertionError(f"expected an all_gather node, got {n}")
    visited: OrderedSet[torch.fx.Node] = OrderedSet()
    queue = list(n.all_input_nodes)
    placeholders = 0
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        if node.op == "placeholder":
            placeholders += 1
            if placeholders > 1:
                return False
        else:
            queue.extend(node.all_input_nodes)
    return placeholders == 1


def is_fsdp_all_gather_wait(wait: torch.fx.Node) -> bool:
    ag_node = wait.args[0]  # type: ignore[arg-type, union-attr]
    return is_fsdp_all_gather(ag_node)  # type: ignore[arg-type]


def is_fsdp_reduce_scatter_wait(wait: torch.fx.Node) -> bool:
    """Check if a reduce_scatter wait flows only to graph outputs through unary ops.

    Uses forward walk to verify every path from *wait* reaches an output node and
    every intermediate node is unary (single input). This is conservative: it
    rejects multi-input nodes even if all inputs derive from the RS, but that is
    sufficient for current FSDP2 patterns where RS→output is always a unary chain
    (view/reshape/cast).

    Returns False for compiled multi-microbatch gradient accumulation where
    add(existing_grad, rs_result) appears in the chain. Current FSDP2 compile
    patterns don't produce this (each microbatch is compiled separately).
    """
    if not wait.users:
        return False
    visited: OrderedSet[torch.fx.Node] = OrderedSet()
    queue = [wait]
    while queue:
        node = queue.pop()
        if node in visited:
            continue
        visited.add(node)
        for user in node.users:
            if user.op == "output":
                continue
            if len(user.all_input_nodes) != 1:
                return False
            queue.append(user)
    return True


_LINEAR_REDUCE_OPS = OrderedSet(["sum", "avg"])

_dedup_rs_pass: PatternMatcherPass | None = None


def _get_dedup_rs_pass() -> PatternMatcherPass:
    global _dedup_rs_pass

    if _dedup_rs_pass is not None:
        return _dedup_rs_pass

    c10d = torch.ops._c10d_functional
    aten = torch.ops.aten
    dedup_rs_pass = PatternMatcherPass(pass_name="dedup_reduce_scatter")

    def wait_rs(name: str) -> CallFunction:
        return CallFunction(
            c10d.wait_tensor.default,
            CallFunction(
                c10d.reduce_scatter_tensor.default,
                KeywordArg(name),
                KeywordArg("reduce_op"),
                KeywordArg("group_size"),
                KeywordArg("group_name"),
            ),
        )

    def dedup_rs_extra_check(match: Match) -> bool:
        if match.kwargs["reduce_op"] not in _LINEAR_REDUCE_OPS:
            return False
        for node in match.nodes:
            if node.target is aten.add.Tensor:
                continue
            if node.target not in (
                c10d.wait_tensor.default,
                c10d.reduce_scatter_tensor.default,
            ):
                return False
            if len(node.users) != 1:
                return False
        input_a = match.kwargs["input_a"]
        input_b = match.kwargs["input_b"]
        if input_a.meta["val"].dtype != input_b.meta["val"].dtype:
            return False
        return True

    @register_graph_pattern(
        CallFunction(
            aten.add.Tensor,
            wait_rs("input_a"),
            wait_rs("input_b"),
        ),
        extra_check=dedup_rs_extra_check,
        # pyrefly: ignore[bad-argument-type]
        pass_dict=dedup_rs_pass,
    )
    def _(match: Match, input_a, input_b, reduce_op, group_size, group_name):
        def repl(input_a, input_b):
            combined = aten.add.Tensor(input_a, input_b)
            rs = c10d.reduce_scatter_tensor.default(
                combined, reduce_op, group_size, group_name
            )
            return c10d.wait_tensor.default(rs)

        # pyrefly: ignore[bad-argument-type]
        match.replace_by_example(repl, [input_a, input_b])

    _dedup_rs_pass = dedup_rs_pass
    return dedup_rs_pass


def dedup_fsdp_reduce_scatter(gm: torch.fx.GraphModule) -> None:
    """
    Fuse duplicate reduce_scatter ops whose waited results are summed.

    RS is linear, so RS(a) + RS(b) = RS(a + b). This pass rewrites
        rs_a = reduce_scatter(input_a, ...); wait_a = wait(rs_a)
        rs_b = reduce_scatter(input_b, ...); wait_b = wait(rs_b)
        result = add(wait_a, wait_b)
    into
        combined = add(input_a, input_b)
        rs = reduce_scatter(combined, ...)
        result = wait(rs)

    For N-way add trees (N > 2), the pattern is applied repeatedly
    until fixpoint — each iteration fuses one leaf pair.
    """
    dedup_rs_pass = _get_dedup_rs_pass()
    while dedup_rs_pass.apply(gm):
        pass
    gm.graph.lint()
    gm.recompile()


def bucket_fsdp_all_gather(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode = "default",
) -> None:
    """
    Bucketing pass for SimpleFSDP all_gather ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket id and returns size of a bucket in megabytes.
    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    if bucket_cap_mb_by_bucket_idx is None:
        raise AssertionError("expected bucket_cap_mb_by_bucket_idx to be set")
    ag_buckets = bucket_all_gather_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_all_gather_wait,
    )
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets, mode)


def bucket_fsdp_reduce_scatter(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode = "default",
) -> None:
    """
    Bucketing pass for SimpleFSDP reduce_scatter ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket idx and returns size of a bucket in megabytes. By default
            torch._inductor.fx_passes.bucketing.bucket_cap_mb_by_bucket_idx_default is used.

    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    # reduce_scatter bucketing does not support multidtype mode;
    # resolve None to the default and strip multidtype if present.
    rs_bucket_mode: BucketMode = mode or "default"
    if "multidtype" in rs_bucket_mode:
        rs_bucket_mode = rs_bucket_mode.replace("_multidtype", "")  # type: ignore[assignment]
    rs_buckets = bucket_reduce_scatter_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_reduce_scatter_wait,
        mode=rs_bucket_mode,
    )
    if len(rs_buckets) == 0:
        return
    merge_reduce_scatter(gm, rs_buckets, mode)


def bucket_fsdp_all_reduce(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    fsdp_groups: OrderedSet[str] | None = None,
) -> None:
    """Bucketing pass for FSDP all_reduce ops.

    For all_gather and reduce_scatter we use structural heuristics
    (single-placeholder ancestry for AG, unary-chain-to-output for RS)
    to identify FSDP collectives. For all_reduce there is no reliable
    structural pattern, so we identify FSDP groups via AG/RS first and
    filter by group name here.
    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default

    def is_fsdp_all_reduce_wait(wait: torch.fx.Node) -> bool:
        ar_node = wait.args[0]
        if not is_all_reduce_tensor(ar_node):  # type: ignore[arg-type]
            return False
        if fsdp_groups is None:
            return True
        return _get_group_name(ar_node) in fsdp_groups  # type: ignore[arg-type]

    ar_buckets = bucket_all_reduce_by_mb(
        gm, bucket_cap_mb_by_bucket_idx, filter_wait_node=is_fsdp_all_reduce_wait
    )
    for bucket in ar_buckets:
        merge_all_reduce_bucket(gm.graph, bucket)


def _get_collective_kwargs(n: fx.Node) -> dict[str, object]:
    """Normalize a collective node's args into keyword args."""
    from torch.fx.operator_schemas import normalize_function

    opt = normalize_function(
        n.target,  # type: ignore[arg-type]
        args=n.args,
        kwargs=n.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if opt is None:
        raise AssertionError(f"normalize_function returned None for {n.target}")
    _, kwargs = opt
    return kwargs


def _get_group_name(n: fx.Node) -> str:
    return _get_collective_kwargs(n)["group_name"]  # type: ignore[return-value]


def _get_group_size_from_node(n: fx.Node) -> int:
    return _get_collective_kwargs(n)["group_size"]  # type: ignore[return-value]


def _find_all_gathers(graph: torch.fx.Graph) -> list[torch.fx.Node]:
    """Return all all_gather nodes (both default and _out variants) via O(1) lookup."""
    return [
        *graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor.default,
        ),
        *graph.find_nodes(
            op="call_function",
            target=torch.ops._c10d_functional.all_gather_into_tensor_out.default,
        ),
    ]


def _find_reduce_scatters(graph: torch.fx.Graph) -> list[torch.fx.Node]:
    """Return all reduce_scatter nodes via O(1) lookup."""
    return graph.find_nodes(
        op="call_function",
        target=torch.ops._c10d_functional.reduce_scatter_tensor.default,
    )


def _find_all_reduces(graph: torch.fx.Graph) -> list[torch.fx.Node]:
    """Return all all_reduce nodes via O(1) lookup."""
    return graph.find_nodes(
        op="call_function",
        target=torch.ops._c10d_functional.all_reduce.default,
    )


def identify_fsdp_groups(
    gm: torch.fx.GraphModule,
) -> tuple[OrderedSet[str], int | None]:
    """Identify FSDP process groups and return (group_names, group_size).

    Uses is_fsdp_all_gather heuristic on all_gather nodes to find FSDP groups,
    then returns those group names plus the group_size from the first match.
    All collectives on these groups (AG, RS, AR) are considered FSDP via
    group-name transitivity.
    """
    fsdp_groups: OrderedSet[str] = OrderedSet()
    group_size: int | None = None
    for n in _find_all_gathers(gm.graph):
        if is_fsdp_all_gather(n):
            fsdp_groups.add(_get_group_name(n))
            if group_size is None:
                group_size = _get_group_size_from_node(n)
    return fsdp_groups, group_size


def compute_pre_bucket_cap_mb(
    group_size: int,
    bucket_cap_mb_override: float | None = None,
) -> float:
    """Compute the bucket cap for pre-bucketing based on bandwidth saturation.

    Returns a bucket size in MB that targets saturation of the process group's
    network bandwidth. Uses empirical per-interconnect profiles with auto-detection
    of GPU generation and intra/inter-node topology.

    If bucket_cap_mb_override is set, returns that directly.
    """
    if bucket_cap_mb_override is not None:
        return bucket_cap_mb_override

    import torch._inductor.config as inductor_config

    dist_opts = inductor_config.aten_distributed_optimizations
    cal_mult = (
        dist_opts.pre_bucketing_fsdp_collectives_saturation_calibration_multiplier
    )
    floor_mb = dist_opts.pre_bucketing_fsdp_collectives_min_bucket_cap_mb
    ceil_mb = dist_opts.pre_bucketing_fsdp_collectives_max_bucket_cap_mb

    min_bytes = 0
    try:
        from torch._inductor.comm_analysis import (
            compute_min_saturation_bytes,  # pyrefly: ignore [missing-module-attribute]
            NCCL_COLL,
        )

        min_bytes = compute_min_saturation_bytes(
            group_size, NCCL_COLL.ALL_GATHER, target_efficiency=0.9
        )
        cap_mb = cal_mult * min_bytes / (1024 * 1024)
    except ImportError:
        cap_mb = floor_mb
    cap_mb = max(floor_mb, min(ceil_mb, cap_mb))

    if dist_opts.pre_bucketing_fsdp_collectives_verbose:
        try:
            from torch._inductor.comm_analysis import (
                detect_interconnect,  # pyrefly: ignore [missing-module-attribute]
                get_inter_node_bw,  # pyrefly: ignore [missing-module-attribute]
                get_intra_node_bw,  # pyrefly: ignore [missing-module-attribute]
            )

            logger.info(
                "pre_bucket_cap: interconnect=%s intra_bw=%.0f inter_bw=%.0f "
                "saturation_bytes=%d (%.1f MB) cal_mult=%.2f cap_mb=%.1f",
                detect_interconnect(group_size).name,
                get_intra_node_bw(),
                get_inter_node_bw(),
                min_bytes,
                min_bytes / (1024 * 1024),
                cal_mult,
                cap_mb,
            )
        except ImportError:
            logger.info("pre_bucket_cap: cap_mb=%.1f (fallback)", cap_mb)

    return cap_mb


def _tensor_size_mb(val: torch.Tensor) -> float:
    """Return tensor size in MB, using size hints for dynamic shapes."""
    from torch.fx.experimental.symbolic_shapes import optimization_hint

    numel = optimization_hint(val.numel(), fallback=0)
    return numel * val.element_size() / (1024 * 1024)


def _collect_collective_sizes(
    gm: torch.fx.GraphModule, fsdp_groups: OrderedSet[str]
) -> list[dict[str, object]]:
    """Collect per-collective transfer sizes (MB) for FSDP collectives in graph order.

    AG: output tensor size (the gathered result, i.e. bytes on the wire).
    RS: input tensor size (the pre-scatter tensor, i.e. bytes on the wire).
    AR: input tensor size (in-place, same size in and out).
    """
    sizes: list[dict[str, object]] = []
    for n in _find_all_gathers(gm.graph):
        if _get_group_name(n) in fsdp_groups:
            size_mb = _tensor_size_mb(n.meta["val"])
            sizes.append({"type": "AG", "size_mb": round(size_mb, 3), "name": n.name})
    for n in _find_reduce_scatters(gm.graph):
        if _get_group_name(n) in fsdp_groups:
            size_mb = _tensor_size_mb(n.all_input_nodes[0].meta["val"])
            sizes.append({"type": "RS", "size_mb": round(size_mb, 3), "name": n.name})
    for n in _find_all_reduces(gm.graph):
        if _get_group_name(n) in fsdp_groups:
            size_mb = _tensor_size_mb(n.all_input_nodes[0].meta["val"])
            sizes.append({"type": "AR", "size_mb": round(size_mb, 3), "name": n.name})
    return sizes


def pre_bucket_fsdp_collectives(
    gm: torch.fx.GraphModule,
    mode: BucketMode | None = None,
    bucket_cap_mb: float | None = None,
) -> None:
    """Pre-bucket FSDP collectives before overlap scheduling.

    Identifies FSDP process groups via all_gather structural heuristics,
    then merges all_gather, reduce_scatter, and all_reduce ops on  those
    groups into bandwidth-saturating buckets.
    """
    import torch._inductor.config as inductor_config

    dist_opts = inductor_config.aten_distributed_optimizations
    verbose = dist_opts.pre_bucketing_fsdp_collectives_verbose

    fsdp_groups, group_size = identify_fsdp_groups(gm)
    if not fsdp_groups:
        return

    def _count_fsdp(nodes: list[fx.Node]) -> int:
        return sum(1 for n in nodes if _get_group_name(n) in fsdp_groups)

    ag_count = _count_fsdp(_find_all_gathers(gm.graph))
    rs_count = _count_fsdp(_find_reduce_scatters(gm.graph))
    ar_count = _count_fsdp(_find_all_reduces(gm.graph))

    if verbose:
        coll_sizes = _collect_collective_sizes(gm, fsdp_groups)
        logger.info(
            "pre_bucket_fsdp: %d collectives before bucketing, sizes (MB): %s",
            len(coll_sizes),
            ", ".join(f"{s['type']}({s['size_mb']})" for s in coll_sizes[:50]),
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "pre_bucketing_collective_sizes",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(coll_sizes),
        )

    if group_size is not None:
        cap_mb = compute_pre_bucket_cap_mb(group_size, bucket_cap_mb)
    else:
        # Reachable when fsdp_groups were identified from all-gathers that
        # were subsequently erased by an earlier pass, or when only RS/AR remain.
        logger.warning("pre_bucket_fsdp: no FSDP all_gather found for group_size")
        cap_mb = bucket_cap_mb if bucket_cap_mb is not None else 500.0

    def bucket_cap_fn(_idx: int) -> float:
        return cap_mb

    resolved_mode: BucketMode = mode or "default"
    bucket_fsdp_all_gather(
        gm, bucket_cap_mb_by_bucket_idx=bucket_cap_fn, mode=resolved_mode
    )
    bucket_fsdp_reduce_scatter(
        gm, bucket_cap_mb_by_bucket_idx=bucket_cap_fn, mode=resolved_mode
    )
    bucket_fsdp_all_reduce(
        gm, bucket_cap_mb_by_bucket_idx=bucket_cap_fn, fsdp_groups=fsdp_groups
    )

    ag_count_after = _count_fsdp(_find_all_gathers(gm.graph))
    rs_count_after = _count_fsdp(_find_reduce_scatters(gm.graph))
    ar_count_after = _count_fsdp(_find_all_reduces(gm.graph))

    gpus_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 8
    nNodes = math.ceil(group_size / gpus_per_node) if group_size is not None else 1

    # Verbose: log per-collective sizes after bucketing
    if verbose:
        coll_sizes_after = _collect_collective_sizes(gm, fsdp_groups)
        logger.info(
            "pre_bucket_fsdp: %d collectives after bucketing, sizes (MB): %s",
            len(coll_sizes_after),
            ", ".join(f"{s['type']}({s['size_mb']})" for s in coll_sizes_after[:50]),
        )
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "pre_bucketing_collective_sizes_after",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(coll_sizes_after),
        )

    logger.info(
        "pre_bucket_fsdp_collectives: fsdp_groups=%s, group_size=%s, nNodes=%d, "
        "bucket_cap_mb=%.1f, all_gather %d->%d, reduce_scatter %d->%d, "
        "all_reduce %d->%d",
        list(fsdp_groups),
        group_size,
        nNodes,
        cap_mb,
        ag_count,
        ag_count_after,
        rs_count,
        rs_count_after,
        ar_count,
        ar_count_after,
    )

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "pre_bucketing_fsdp_collectives",
            "encoding": "string",
        },
        payload_fn=lambda: (
            f"fsdp_groups={list(fsdp_groups)}, group_size={group_size}, "
            f"nNodes={nNodes}, bucket_cap_mb={cap_mb:.1f}, "
            f"all_gather {ag_count}->{ag_count_after}, "
            f"reduce_scatter {rs_count}->{rs_count_after}, "
            f"all_reduce {ar_count}->{ar_count_after}"
        ),
    )
<|endoftext|>"""Detect fusion regions for overlap scheduling."""

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


def is_view_node(n: fx.Node) -> bool:
    """Check if a node is a view operation (zero cost, no memory allocation)."""

    return isinstance(n.target, torch._ops.OpOverload) and (
        n.target.is_view and n.target.namespace in ("aten", "prims")
    )


def is_fusible_node(n: fx.Node) -> bool:
    """Check if a node is fusible based on whether it has an inductor lowering.

    A node is fusible if:
    - It has a lowering in torch._inductor.lowering.lowerings
    - It does NOT have a flop counter (expensive compute ops like mm/conv)
    - It is NOT a registered fallback (ops that fall back to eager)
    - It is NOT a collective or wait op
    - For aten.cat, it must have <= max_pointwise_cat_inputs inputs
    """
    if n.op != "call_function":
        return False

    target = n.target
    if not isinstance(target, torch._ops.OpOverload):
        return False

    # Exclude collectives and waits (they have their own scheduling)
    if target.namespace == "_c10d_functional":
        return False

    from torch._inductor.lowering import fallbacks, lowerings
    from torch.utils.flop_counter import flop_registry

    # Must have a lowering
    if target not in lowerings:
        return False

    # Exclude fallbacks (ops that fall back to eager execution)
    if target in fallbacks:
        return False

    # Exclude ops with flop counters (expensive compute ops like mm, conv, etc.)
    overload_packet = target.overloadpacket
    if overload_packet in flop_registry:
        return False

    # Special case: cat is only fusible if it has few enough inputs
    if target == torch.ops.aten.cat.default:
        inputs = n.args[0] if n.args else []
        if isinstance(inputs, (list, tuple)):
            import torch._inductor.config as inductor_config

            if len(inputs) > inductor_config.max_pointwise_cat_inputs:
                return False

    return True


def _get_contiguous_fusible_spans(gm: fx.GraphModule) -> list[list[fx.Node]]:
    """Get contiguous spans of fusible nodes from the graph.

    Walks the graph in topological order and groups consecutive fusible
    nodes into spans. Non-fusible nodes act as span boundaries.
    """
    spans: list[list[fx.Node]] = []
    current_span: list[fx.Node] = []

    for node in gm.graph.nodes:
        if is_fusible_node(node):
            current_span.append(node)
        else:
            # Non-fusible node ends the current span
            if current_span:
                spans.append(current_span)
                current_span = []

    if current_span:
        spans.append(current_span)

    return spans


def _find_connected_components(span: list[fx.Node]) -> list[list[fx.Node]]:
    """Find connected components within a span of fusible nodes.

    Two nodes are connected if one is an input to the other (direct data dependency).
    """
    if not span:
        return []

    from torch.fx.experimental.optimization import UnionFind

    span_set = OrderedSet(span)
    node_to_idx = {n: i for i, n in enumerate(span)}

    uf = UnionFind(len(span))
    for i in range(len(span)):
        uf.make_set(i)

    # Union nodes based on input edges
    for node in span:
        node_idx = node_to_idx[node]
        for inp in node.all_input_nodes:
            if inp in span_set:
                uf.join(node_idx, node_to_idx[inp])

    # Group by root
    root_to_nodes: dict[int, list[fx.Node]] = {}
    for node in span:
        root = uf.find(node_to_idx[node])
        if root not in root_to_nodes:
            root_to_nodes[root] = []
        root_to_nodes[root].append(node)

    return list(root_to_nodes.values())


def build_fusion_regions(
    gm: fx.GraphModule,
) -> dict[fx.Node, OrderedSet[fx.Node]]:
    """Build fusion regions from contiguous spans of fusible nodes.

    1. Identify contiguous spans of fusible nodes (separated by non-fusible nodes)
    2. Find connected components within each span
    3. Return regions that have 2+ non-view nodes

    This ensures fusion regions are strictly local - no reordering across
    non-fusible node boundaries.

    Returns a dict mapping each node to its fusion group (OrderedSet of nodes).
    """
    # Build node -> topo index map for sorting
    node_to_idx: dict[fx.Node, int] = {n: i for i, n in enumerate(gm.graph.nodes)}

    # Step 1: Get contiguous spans of fusible nodes
    spans = _get_contiguous_fusible_spans(gm)

    # Step 2: Find connected components within each span
    region_of: dict[fx.Node, OrderedSet[fx.Node]] = {}

    for span in spans:
        if len(span) < 2:
            continue

        components = _find_connected_components(span)

        for component in components:
            # Skip regions with fewer than 2 non-view nodes (views have no cost)
            non_view_count = sum(1 for n in component if not is_view_node(n))
            if non_view_count < 2:
                continue

            # Sort nodes in topological order to preserve original ordering
            sorted_component = sorted(component, key=lambda n: node_to_idx[n])
            node_set = OrderedSet(sorted_component)

            for node in sorted_component:
                region_of[node] = node_set

    return region_of


def estimate_fused_node_costs(
    region_of: dict[fx.Node, OrderedSet[fx.Node]],
) -> dict[fx.Node, float]:
    """Estimate per-node costs accounting for fusion (no graph collapse).

    For each node in a fusion region, only count I/O that crosses the
    region boundary (external reads/writes). Internal intermediates
    won't be materialized by inductor, so they shouldn't count.

    Nodes with only internal I/O get cost 0. Nodes with external I/O
    get the bandwidth cost of just those external tensors.
    """
    from torch.utils._pytree import tree_flatten
    from torch.utils._runtime_estimation import get_transfer_time

    costs: dict[fx.Node, float] = {}
    seen: OrderedSet[int] = OrderedSet()

    for node_set in region_of.values():
        rid = id(node_set)
        if rid in seen:
            continue
        seen.add(rid)

        for n in node_set:
            # External inputs: values from nodes outside this region
            ext_inputs = []
            for inp in n.all_input_nodes:
                if inp not in node_set:
                    val = inp.meta.get("val")
                    if val is not None:
                        ext_inputs.append(val)

            # External outputs: this node's value is used outside the region
            ext_outputs = []
            has_external_user = any(u not in node_set for u in n.users)
            if has_external_user:
                val = n.meta.get("val")
                if val is not None:
                    ext_outputs.append(val)

            if not ext_inputs and not ext_outputs:
                costs[n] = 0.0
            else:
                flat_in, _ = tree_flatten(ext_inputs)
                flat_out, _ = tree_flatten(ext_outputs)
                costs[n] = get_transfer_time(flat_in, flat_out) / 1e6

    return costs
<|endoftext|>from __future__ import annotations

import logging
import warnings

import torch
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


def _get_collective_info(node):
    """Return (is_ag, group_name) if node is an AG/RS collective, else None."""
    from torch._inductor.fx_passes.bucketing import (
        is_all_gather_into_tensor,
        is_reduce_scatter_tensor,
    )
    from torch._inductor.fx_passes.overlap_scheduling import get_group_name

    if is_all_gather_into_tensor(node):
        return True, get_group_name(node)
    if is_reduce_scatter_tensor(node):
        return False, get_group_name(node)
    return None


def replace_collectives_with_low_contention(
    graph: torch.fx.Graph,
) -> None:
    """Replace FSDP collectives with copy-engine symm_mem variants."""
    symm_mem = torch.ops.symm_mem

    collectives = []
    groups: OrderedSet[str] = OrderedSet()
    for node in list(graph.nodes):
        info = _get_collective_info(node)
        if info is None:
            continue
        is_ag, group_name = info
        collectives.append((node, is_ag, group_name))
        groups.add(group_name)

    if not collectives:
        return

    # Some group names can't be resolved at compile time — skip them.
    valid_groups: OrderedSet[str] = OrderedSet()
    for group_name in groups:
        if _enable_symm_mem(group_name):
            valid_groups.add(group_name)

    # Filter to collectives whose groups we can actually resolve
    collectives = [
        (node, is_ag, gn) for node, is_ag, gn in collectives if gn in valid_groups
    ]
    if not collectives:
        return

    from torch._inductor import config

    cfg = config.aten_distributed_optimizations
    min_bytes = cfg.low_contention_min_bytes_per_rank
    skip_overlap_check = cfg.low_contention_skip_overlap_check
    use_ag_ce_multicast = cfg.low_contention_all_gather_ce_multicast

    replacements = 0
    skipped_small = 0
    skipped_no_overlap = 0
    skipped_nvlink_contention = 0
    skipped_out_variant = 0
    for node, is_ag, group_name in collectives:
        coll_type = "AG" if is_ag else "RS"

        # Size filter: LC barrier overhead dominates for small messages
        if min_bytes > 0:
            per_rank_bytes = _get_per_rank_bytes(node, is_ag)
            if per_rank_bytes is not None and per_rank_bytes < min_bytes:
                skipped_small += 1
                log.debug(
                    "LC skip %s %s: size %d < min_bytes %d",
                    coll_type,
                    node.name,
                    per_rank_bytes,
                    min_bytes,
                )
                continue

        if not skip_overlap_check:
            # Skip collectives with no compute to hide behind
            if not _has_compute_bound_overlap(node, graph):
                skipped_no_overlap += 1
                log.debug(
                    "LC skip %s %s: no compute-bound overlap", coll_type, node.name
                )
                continue

            # Skip if other groups' NCCL collectives overlap on NVLink
            if _has_other_group_collectives(node, group_name, graph):
                skipped_nvlink_contention += 1
                log.debug(
                    "LC skip %s %s: overlaps other-group collectives (NVLink contention)",
                    coll_type,
                    node.name,
                )
                continue

        target = None
        if is_ag:
            target = _select_low_contention_all_gather_target(
                symm_mem,
                input_node=node.args[0],
                use_ag_ce_multicast=use_ag_ce_multicast,
            )

        if not _replace_collective(
            node,
            graph,
            symm_mem,
            is_ag,
            group_name,
            target=target,
        ):
            skipped_out_variant += 1
            continue
        replacements += 1

    log.info(
        "Replaced %d/%d FSDP collectives "
        "(skipped_small=%d, skipped_no_overlap=%d, "
        "skipped_nvlink_contention=%d, skipped_out_variant=%d, "
        "min_bytes=%d, skip_overlap_check=%s)",
        replacements,
        len(collectives),
        skipped_small,
        skipped_no_overlap,
        skipped_nvlink_contention,
        skipped_out_variant,
        min_bytes,
        skip_overlap_check,
    )


def _enable_symm_mem(group_name):
    """Try to enable symmetric memory for a group. Returns True on success."""
    from torch.distributed._symmetric_memory import (
        enable_symm_mem_for_group,
        is_symm_mem_enabled_for_group,
    )

    if is_symm_mem_enabled_for_group(group_name):
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            enable_symm_mem_for_group(group_name)
        return True
    except (TypeError, RuntimeError, KeyError) as e:
        log.debug("LC: cannot enable symm_mem for group %s: %s", group_name, e)
        return False


def _has_multicast_support(device_index: int) -> bool:
    try:
        from torch._C._autograd import DeviceType
        from torch._C._distributed_c10d import _SymmetricMemory

        return bool(
            _SymmetricMemory.has_multicast_support(DeviceType.CUDA, device_index)
        )
    except Exception:
        return False


def _select_low_contention_all_gather_target(
    symm_mem,
    input_node,
    use_ag_ce_multicast=False,
):
    if not use_ag_ce_multicast:
        return symm_mem._low_contention_all_gather.default

    device_index = None
    input_val = input_node.meta.get("val")
    if isinstance(input_val, torch.Tensor) and input_val.device.type == "cuda":
        device_index = input_val.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    if _has_multicast_support(device_index):
        return symm_mem._low_contention_all_gather_ce_multicast.default
    log.info(
        "low_contention_all_gather_ce_multicast requested but multicast is not "
        "supported on device %d; falling through.",
        device_index,
    )

    return symm_mem._low_contention_all_gather.default


def _replace_collective(
    node,
    graph,
    symm_mem,
    is_ag,
    group_name,
    target=None,
) -> bool:
    input_node = node.args[0]
    if is_ag:
        if target is None:
            return False
        if node.target is torch.ops._c10d_functional.all_gather_into_tensor_out.default:
            out = node.kwargs["out"]
            if target is symm_mem._low_contention_all_gather_ce_multicast.default:
                target = symm_mem._low_contention_all_gather_ce_multicast_out.default
            else:
                # The existing low-contention all-gather has no out variant.
                return False
            args = (input_node, group_name, out)
        else:
            args = (input_node, group_name)
    else:
        reduce_op = node.args[1]
        target = symm_mem._low_contention_reduce_scatter.default
        args = (input_node, reduce_op, group_name)

    with graph.inserting_before(node):
        new_node = graph.call_function(target, args=args)
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    graph.erase_node(node)
    return True


def _get_per_rank_bytes(node, is_ag):
    """Return per-rank message bytes for a collective, or None if unknown."""
    input_val = node.args[0].meta.get("val") if node.args else None
    if not isinstance(input_val, torch.Tensor):
        return None
    total_bytes = input_val.nelement() * input_val.element_size()
    if is_ag:
        return total_bytes
    # For RS, input is the full tensor; per-rank = total / group_size
    group_size = node.args[2] if len(node.args) > 2 else None
    if not isinstance(group_size, int) or group_size <= 0:
        return None
    return total_bytes // group_size


def _is_compute_or_contains_compute(node, graph_module):
    """Check if a node is compute, including inside control_deps subgraphs."""
    from torch._inductor.fx_passes.overlap_scheduling import is_compute_node

    if is_compute_node(node):
        return True
    from torch._inductor.fx_passes.control_dependencies import ControlDeps

    if node.op == "call_function" and isinstance(node.target, ControlDeps):
        subgraph_node = node.args[1] if len(node.args) > 1 else None
        if isinstance(subgraph_node, torch.fx.Node) and subgraph_node.op == "get_attr":
            if not isinstance(subgraph_node.target, str):
                raise AssertionError(
                    f"expected str get_attr target, got {type(subgraph_node.target)}"
                )
            subgraph = getattr(graph_module, subgraph_node.target, None)
            if isinstance(subgraph, torch.fx.GraphModule):
                for n in subgraph.graph.nodes:
                    if is_compute_node(n):
                        return True
    return False


def _has_compute_bound_overlap(start_node, graph):
    """Check if compute-bound ops exist between collective start and wait."""
    wait_node = _find_wait_for_collective(start_node)
    if wait_node is None:
        return False

    graph_module = graph.owning_module
    in_range = False
    for node in graph.nodes:
        if node is start_node:
            in_range = True
            continue
        if node is wait_node:
            break
        if in_range and _is_compute_or_contains_compute(node, graph_module):
            return True
    return False


def _has_other_group_collectives(start_node, group_name, graph):
    """Check if other groups' collectives overlap, competing for NVLink."""
    wait_node = _find_wait_for_collective(start_node)
    if wait_node is None:
        return False

    in_range = False
    for node in graph.nodes:
        if node is start_node:
            in_range = True
            continue
        if node is wait_node:
            break
        if not in_range:
            continue
        info = _get_collective_info(node)
        if info is not None:
            _, other_group = info
            if other_group != group_name:
                log.debug(
                    "LC contention %s: found %s (group %s) between start/wait",
                    start_node.name,
                    node.name,
                    other_group,
                )
                return True
    return False


def _is_wait_tensor(node):
    """Check if node is a wait_tensor op (direct or wrapped in ControlDeps)."""
    if node.op != "call_function":
        return False
    if node.target is torch.ops._c10d_functional.wait_tensor.default:
        return True
    # Handles public namespace (c10d_functional.wait_tensor) and
    # ControlDeps-wrapped wait_tensor (from TBB manual scheduling)
    return "wait_tensor" in node.name


def _find_wait_for_collective(start_node):
    """Find the wait_tensor node for a collective.

    Handles multiple graph patterns:
    1. Direct: start -> wait_tensor(start)
    2. _out variant: start(out=buf) -> wait_tensor(buf)
    3. ControlDeps-wrapped: start -> control_deps(wait_tensor_subgraph, start)
    """
    for user in start_node.users:
        if _is_wait_tensor(user):
            return user

    # For _out variants, check users of the out-buffer keyword argument.
    c10d = torch.ops._c10d_functional
    if start_node.target in (
        c10d.all_gather_into_tensor_out.default,
        c10d.reduce_scatter_tensor_out.default,
    ):
        out_buf = start_node.kwargs.get("out")
        if isinstance(out_buf, torch.fx.Node):
            for user in out_buf.users:
                if _is_wait_tensor(user):
                    return user

    return None
<|endoftext|>import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


class BitsetAncestors:
    """Precomputed transitive ancestor sets using Python arbitrary-precision ints.

    Each node gets an index i (0..N-1) from topological order. A node's
    ancestor set is a single Python ``int`` where bit j is set iff node j
    is a transitive ancestor. Python ``int`` is arbitrary-precision, so for
    N nodes each int is ~N/8 bytes internally (~N/64 machine words).

    The key advantage over ``dict[fx.Node, OrderedSet[fx.Node]]`` is that
    CPython implements ``int |= int`` as a C-level loop over machine words,
    making the transitive closure O(N^2 / 64) instead of O(N^2) in
    Python-level set operations.

    Example -- diamond graph (edges point downward)::

            a          index 0
           / \\
          b   c        index 1, 2
           \\ /
            d          index 3

    Ancestor bitsets::

        a: 0b0000 = 0   (no ancestors)
        b: 0b0001 = 1   (bit 0 -> a)
        c: 0b0001 = 1   (bit 0 -> a)
        d: 0b0111 = 7   (bits 0,1,2 -> a, b, c)

    How d's bitset is built (d has parents b and c)::

        b = 0
        # parent b (idx=1): set bit 1, merge b's ancestors
        b |= (1 << 1) | bits[1]   # b = 0b0010 | 0b0001 = 0b0011
        # parent c (idx=2): set bit 2, merge c's ancestors
        b |= (1 << 2) | bits[2]   # b = 0b0011 | 0b0100 | 0b0001 = 0b0111
        bits[3] = 0b0111

    Querying::

        ancestors = BitsetAncestors(nodes)
        ancestors.is_ancestor(a, d)    # (bits[3] >> 0) & 1 -> True
        ancestors.has_dep(a, d)  # True (a is ancestor of d)
        list(ancestors.iter_ancestors(d))  # [a, b, c] via bit-scan

    Iteration uses the x & -x trick to isolate the lowest set bit::

        bits = 0b0111
        bits & -bits = 0b0001  -> idx 0 -> yield a, clear: 0b0110
        bits & -bits = 0b0010  -> idx 1 -> yield b, clear: 0b0100
        bits & -bits = 0b0100  -> idx 2 -> yield c, clear: 0b0000

    Args:
        nodes: topologically sorted list of FX nodes.
        extra_inputs: optional additional edges beyond the FX graph
            (e.g. hiding-interval deps in the overlap bucketer).
    """

    def __init__(
        self,
        nodes: list[fx.Node],
        extra_inputs: dict[fx.Node, OrderedSet[fx.Node]] | None = None,
    ):
        n = len(nodes)
        node_to_idx: dict[fx.Node, int] = {nd: i for i, nd in enumerate(nodes)}
        bits = [0] * n
        extra = extra_inputs or {}

        for i, node in enumerate(nodes):
            b = 0
            for inp in node._input_nodes:
                j = node_to_idx.get(inp)
                if j is not None:
                    b |= (1 << j) | bits[j]
            for inp in extra.get(node, ()):
                j = node_to_idx.get(inp)
                if j is not None:
                    b |= (1 << j) | bits[j]
            bits[i] = b

        self._bits = bits
        self._node_to_idx = node_to_idx
        self._idx_to_node = nodes

    def is_ancestor(self, ancestor: fx.Node, descendant: fx.Node) -> bool:
        """O(1) test: is ``ancestor`` a transitive ancestor of ``descendant``?"""
        anc_idx = self._node_to_idx.get(ancestor)
        desc_idx = self._node_to_idx.get(descendant)
        if anc_idx is None or desc_idx is None:
            return False
        return bool((self._bits[desc_idx] >> anc_idx) & 1)

    def has_dep(self, n1: fx.Node, n2: fx.Node) -> bool:
        """Check if either node is an ancestor of the other."""
        return self.is_ancestor(n1, n2) or self.is_ancestor(n2, n1)

    def get_ancestor_bits(self, node: fx.Node) -> int:
        """Return the raw ancestor bitset for ``node``."""
        return self._bits[self._node_to_idx[node]]

    def node_bit(self, node: fx.Node) -> int:
        """Return the single-bit mask ``1 << idx`` for ``node``."""
        return 1 << self._node_to_idx[node]

    def ancestors_intersect(self, node: fx.Node, mask: int) -> bool:
        """Check if any ancestor of ``node`` is set in ``mask``."""
        idx = self._node_to_idx.get(node)
        if idx is None:
            return False
        return bool(self._bits[idx] & mask)

    def iter_ancestors(self, node: fx.Node):
        """Yield all ancestors of ``node`` via lowest-bit scan."""
        bits = self._bits[self._node_to_idx[node]]
        idx_to_node = self._idx_to_node
        while bits:
            idx = (bits & -bits).bit_length() - 1
            yield idx_to_node[idx]
            bits &= bits - 1
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import functools
import itertools
import logging
import operator
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from typing import Any, TypeVar
from typing_extensions import ParamSpec

import torch
import torch._inductor as inductor
import torch.utils._pytree as pytree
from torch import fx
from torch._decomp import register_decomposition
from torch._dynamo.utils import counters
from torch._higher_order_ops.flex_gemm import _PRESERVE_FLEX_GEMM_GEMM_OP
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_custom_graph_passes,
)
from torch._inductor.virtualized import ops  # noqa: F401
from torch._logging import trace_structured
from torch._prims_common import is_boolean_dtype, is_expandable_to, is_integer_dtype
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq
from torch.utils._ordered_set import OrderedSet

from .. import config, ir, pattern_matcher  # noqa: F401
from ..codegen.common import custom_backend_passes
from ..fx_utils import FakeTensorUpdater, get_fake_args_kwargs, get_node_storage
from ..lowering import lowerings as L
from ..pattern_matcher import (
    _return_true,
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    filter_nodes,
    fwd_only,
    get_arg_value,
    get_mutation_region_id,
    Ignored,
    init_once_fakemode,
    KeywordArg,
    ListOf,
    Match,
    MultiOutputPattern,
    MULTIPLE,
    PatternMatcherPass as PatternMatcherPassBase,
    register_graph_pattern,
    register_replacement,
    stable_topological_sort,
)
from ..utils import (
    decode_device,
    get_all_devices,
    get_gpu_type,
    is_gpu,
    is_pointwise_use,
    OPTIMUS_EXCLUDE_POST_GRAD,
)
from ..virtualized import V
from .b2b_gemm import B2B_GEMM_PASS
from .control_dependencies import control_deps, preserve_node_ordering
from .ddp_fusion import fuse_ddp_communication
from .group_batch_fusion import group_batch_fusion_passes, POST_GRAD_FUSIONS
from .micro_pipeline_tp import micro_pipeline_tp_pass
from .pre_grad import is_same_dict, save_inductor_dict
from .reduced_atomic_contention import partitioned_scatter_optimization_pass
from .reinplace import reinplace_inplaceable_ops
from .split_cat import POST_GRAD_PATTERNS


_T = TypeVar("_T")
_P = ParamSpec("_P")

PatternMatcherPass = functools.partial(
    PatternMatcherPassBase, subsystem="post_grad_passes"
)

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

# First pass_patterns[0] are applied, then [1], then [2]
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]


def _remove_profiler_ops(graph: torch.fx.Graph) -> None:
    """
    Remove profiler ops (record_function) from the graph.
    These ops are side-effectful but don't affect computation,
    and we don't want them to block fusion.
    """
    profiler_ops = OrderedSet(
        [
            torch.ops.profiler._record_function_enter.default,
            torch.ops.profiler._record_function_enter_new.default,
            torch.ops.profiler._record_function_exit._RecordFunction,
        ]
    )

    nodes_to_remove = []

    for node in graph.nodes:
        if node.op == "call_function" and node.target in profiler_ops:
            nodes_to_remove.append(node)

    for node in reversed(nodes_to_remove):
        graph.erase_node(node)


def _is_nondeterministic_seeded_node(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.nondeterministic_seeded in node.target.tags
    )


def _chain_random_ops_for_ordering(graph: torch.fx.Graph) -> None:
    """Chain nondeterministic_seeded ops with control_deps to preserve program order.

    When fallback_random=True, random ops use the global CUDA RNG and must
    execute in their original program order.
    """
    if not config.fallback_random:
        return

    random_nodes = [n for n in graph.nodes if _is_nondeterministic_seeded_node(n)]
    if len(random_nodes) < 2:
        return

    additional_deps_map: dict[torch.fx.Node, OrderedSet[torch.fx.Node]] = {}
    for i in range(1, len(random_nodes)):
        additional_deps_map[random_nodes[i]] = OrderedSet([random_nodes[i - 1]])

    preserve_node_ordering(graph, additional_deps_map)


def respecialize_current_device_nodes(graph: torch.fx.Graph) -> None:
    """[device-as-parameter] Re-specialize CooR _coor_current_device() device nodes.

    Under compile-on-one-rank, make_fx rewrites a baked accelerator device operand to a
    ``_coor_current_device()`` node so the FX graph is rank-agnostic. Inductor has no
    device-valued IR, so before lowering we replace each use of that node with the node's
    own runtime value -- the concrete current device (``_coor_current_device()``), the
    authoritative source, not the consumer's meta. Runs in post_grad, before GraphLowering,
    so the node never reaches the call_function OpOverload assertion. ``_coor_current_device``
    lives in core fx and only reads torch.accelerator, so this never imports
    torch.distributed for a non-distributed compile.
    """
    # Importing proxy_tensor registers the coor::current_device op (core fx, no
    # torch.distributed) and gives us its impl for the concrete value.
    from torch.fx.experimental.proxy_tensor import _coor_current_device

    target = torch.ops.coor.current_device.default
    nodes = graph.find_nodes(op="call_function", target=target)
    if not nodes:
        return
    device = _coor_current_device()
    for node in nodes:
        for user in list(node.users):
            user.args = torch.fx.map_arg(
                user.args, lambda n: device if n is node else n
            )
            user.kwargs = torch.fx.map_arg(
                user.kwargs, lambda n: device if n is node else n
            )
        graph.erase_node(node)


def post_grad_passes(gm: torch.fx.GraphModule, is_inference: bool):
    """
    Passes that run on after grad.  This is called once on the forwards
    graph and once on the backwards graph.

    The IR here has been normalized and functionalized.
    """
    GraphTransformObserver = functools.partial(
        torch.fx.passes.graph_transform_observer.GraphTransformObserver,
        subsystem="post_grad_passes",
    )

    if config.dce:
        # has some issues with mutation in inference mode
        gm.graph.eliminate_dead_code()

    if config.reorder_for_locality and (
        is_inference or config.reorder_for_locality_in_training
    ):
        GraphTransformObserver(gm, "reorder_for_locality").apply_graph_pass(
            reorder_for_locality
        )

    fake_tensor_updater = FakeTensorUpdater(gm)

    for post_grad_custom_pre_pass in get_custom_graph_passes(
        config.post_grad_custom_pre_pass
    ):
        if isinstance(post_grad_custom_pre_pass, CustomInferenceAwareGraphPass):
            post_grad_custom_pre_pass = functools.partial(
                post_grad_custom_pre_pass, is_inference=is_inference
            )
        GraphTransformObserver(gm, "post_grad_custom_pre_pass").apply_graph_pass(
            post_grad_custom_pre_pass
        )

    if torch._C._has_mkldnn:
        if (
            config.cpp.enable_grouped_gemm_template
            and config.max_autotune
            and "CPP" in config.max_autotune_gemm_backends
        ):
            from .mkldnn_fusion import grouped_gemm_pass

            grouped_gemm_pass(gm.graph)

        if config.cpp.enable_concat_linear:
            from .quantization import concat_linear_woq_int4

            # Concat linear optimization for WOQ int4
            concat_linear_woq_int4(gm)

    # Remove profiler ops (record_function) to prevent them blocking fusion
    GraphTransformObserver(gm, "remove_profiler_ops").apply_graph_pass(
        _remove_profiler_ops
    )

    # [device-as-parameter] Re-specialize CooR current_device() device nodes before
    # lowering (inductor has no device-valued IR).
    GraphTransformObserver(gm, "respecialize_current_device").apply_graph_pass(
        respecialize_current_device_nodes
    )

    if config.pattern_matcher:
        lazy_init()
        GraphTransformObserver(gm, "post_grad_custom_pre_pass").apply_graph_pass(
            functools.partial(group_batch_fusion_passes, pre_grad=False)
        )
        GraphTransformObserver(gm, "remove_noop_ops").apply_graph_pass(remove_noop_ops)
        GraphTransformObserver(gm, "remove_assert_ops").apply_graph_pass(
            remove_assert_ops
        )
        for i, patterns in enumerate(pass_patterns):
            GraphTransformObserver(gm, f"pass_pattern_{i}").apply_graph_pass(
                patterns.apply
            )
        if config.partitioned_scatter_enabled:
            GraphTransformObserver(
                gm, "partitioned_scatter_optimization"
            ).apply_graph_pass(partitioned_scatter_optimization_pass)
        for pass_name in config.post_grad_fusion_options:
            # skip all patterns for group batch fusions or quantization patterns
            if pass_name in POST_GRAD_FUSIONS or pass_name in OPTIMUS_EXCLUDE_POST_GRAD:
                continue
            pattern_matcher_pass = POST_GRAD_PATTERNS[pass_name]
            inductor_before_change = save_inductor_dict(
                [pattern_matcher_pass.pass_name]
            )
            GraphTransformObserver(gm, pass_name).apply_graph_pass(
                pattern_matcher_pass.apply
            )
            if not is_same_dict(counters["inductor"], inductor_before_change):
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": f"{pattern_matcher_pass.pass_name}_post_grad",
                        "encoding": "string",
                    },
                    payload_fn=lambda: gm.print_readable(
                        print_output=False, include_stride=True, include_device=True
                    ),
                )
        if config.b2b_gemm_pass:
            B2B_GEMM_PASS.apply(gm.graph)  # type: ignore[arg-type]

    if config._micro_pipeline_tp:
        micro_pipeline_tp_pass(gm.graph)

    if config._fuse_ddp_communication:
        GraphTransformObserver(gm, "fuse_ddp_communication").apply_graph_pass(
            lambda graph: fuse_ddp_communication(
                graph,
                config._fuse_ddp_communication_passes,
                config._fuse_ddp_bucket_size,
            )
        )

    for post_grad_custom_post_pass in get_custom_graph_passes(
        config.post_grad_custom_post_pass
    ):
        if isinstance(post_grad_custom_post_pass, CustomInferenceAwareGraphPass):
            post_grad_custom_post_pass = functools.partial(
                post_grad_custom_post_pass, is_inference=is_inference
            )
        GraphTransformObserver(gm, "post_grad_custom_post_pass").apply_graph_pass(
            post_grad_custom_post_pass
        )

    if config.fallback_random:
        GraphTransformObserver(gm, "chain_random_ops_ordering").apply_graph_pass(
            _chain_random_ops_for_ordering
        )

    GraphTransformObserver(gm, "stable_sort").apply_graph_pass(stable_topological_sort)

    GraphTransformObserver(gm, "move_constructors_to_cuda").apply_graph_pass(
        move_constructors_to_gpu
    )

    fake_tensor_updater.incremental_update()

    for device, custom_backend_pass in custom_backend_passes.items():
        if custom_backend_pass is not None:
            gm_devices = [d.type for d in get_all_devices(gm)]
            if device in gm_devices:
                pass_name = "custom_backend_passes_" + device
                GraphTransformObserver(gm, pass_name).apply_gm_pass(custom_backend_pass)

    # SPMD verification — before collective reordering passes.
    if (
        config.aten_distributed_optimizations.spmd_check
        and _needs_spmd_graph_preservation()
    ):
        from torch._inductor.fx_passes.spmd_check import spmd_check

        spmd_check(gm)

    if config.aten_distributed_optimizations.allow_comms_decompositions:
        from torch._inductor.fx_passes.decomp_comms import decomp_comms

        GraphTransformObserver(gm, "decomp_comms").apply_gm_pass(decomp_comms)

    collectives_bucketing: bool = False

    if config.dedup_reduce_scatters:
        from torch._inductor.fx_passes.fsdp import dedup_fsdp_reduce_scatter

        GraphTransformObserver(gm, "dedup_reduce_scatters").apply_gm_pass(
            dedup_fsdp_reduce_scatter
        )

    if config.bucket_reduce_scatters_fx != "none":
        from torch._inductor.fx_passes.bucketing import bucket_reduce_scatter
        from torch._inductor.fx_passes.fsdp import bucket_fsdp_reduce_scatter

        p = (
            bucket_fsdp_reduce_scatter
            if "fsdp" in config.bucket_reduce_scatters_fx
            else bucket_reduce_scatter
        )
        GraphTransformObserver(gm, "bucket_reduce_scatters").apply_graph_pass(
            lambda graph: p(
                graph.owning_module,  # pyrefly: ignore[bad-argument-type]
                config.bucket_reduce_scatters_fx_bucket_size_determinator,
                config.bucket_reduce_scatters_bucket_mode,  # type: ignore[arg-type]
            )
        )
        collectives_bucketing = True

    if config.bucket_all_reduces_fx != "none":
        from torch._inductor.fx_passes.bucketing import bucket_all_reduce

        GraphTransformObserver(gm, "bucket_all_reduce").apply_graph_pass(
            lambda graph: bucket_all_reduce(
                graph.owning_module,  # pyrefly: ignore[bad-argument-type]
                config.bucket_all_reduces_fx_bucket_size_determinator,
                config.bucket_all_reduces_fx,  # type: ignore[arg-type]
            )
        )
        collectives_bucketing = True

    # Fx all_gather bucketing introduces mutation op
    # Keeping it in the end to keep invariant of functional graph for previous passes.
    if config.bucket_all_gathers_fx != "none":
        from torch._inductor.fx_passes.bucketing import bucket_all_gather
        from torch._inductor.fx_passes.fsdp import bucket_fsdp_all_gather

        p = (
            bucket_fsdp_all_gather  # type: ignore[assignment]
            if "fsdp" in config.bucket_all_gathers_fx
            else bucket_all_gather
        )
        GraphTransformObserver(gm, "bucket_all_gathers").apply_graph_pass(
            lambda graph: p(
                graph.owning_module,  # pyrefly: ignore[bad-argument-type]
                config.bucket_all_gathers_fx_bucket_size_determinator,
                config.bucket_all_gathers_bucket_mode,  # type: ignore[arg-type]
            )
        )
        collectives_bucketing = True

    if collectives_bucketing:
        # Fx collectives bucketing passes require topological sort for the cases:
        # when bucketed collectives have users before the last collective in the bucket
        # AND when inputs of bucketed collective have ancestors after the first collective in the bucket.
        #
        # In this case we can not manually pick the place for bucketed collective insertion.
        # But we are guaranteed by the bucketing (independent collectives in the bucket),
        # that it is possible to reorder nodes to satisfy all ordering requirements.
        #
        # --- before bucketing ---
        # in0 = ...
        # wait_ag0 = ag(in0)
        # user0(wait_ag0)
        # ...
        # pre_in1 = ...
        # in1 = transform(pre_in1)
        # wait_ag1 = ag(in1)
        # user1(wait_ag1)
        #
        # --- after bucketing ---
        #
        # in0 = ...
        # user(wait_ag0) <--- wait_ag0 is defined only after bucketed collective.
        #
        # pre_in1 = ...
        # in1 = transform(pre_in1)
        # ag_bucket(in0+in1)
        # wait_bucket
        # wait_ag0 = wait_bucket[0]
        # wait_ag1 = wait_bucket[1]
        # user1(wait_ag1)
        stable_topological_sort(gm.graph)

    # Apply overlap scheduling if enabled
    if config.aten_distributed_optimizations.enable_overlap_scheduling:
        from torch._inductor.fx_passes.overlap_scheduling import (
            schedule_overlap_bucketing_from_inductor_configs,
        )

        overlap_deps = config.aten_distributed_optimizations.insert_overlap_deps

        # by default, insert overlap deps within inductor
        with config.patch(
            {
                "aten_distributed_optimizations.insert_overlap_deps": (
                    True if overlap_deps is None else overlap_deps
                ),
            }
        ):
            GraphTransformObserver(gm, "overlap_scheduling").apply_graph_pass(
                lambda graph: schedule_overlap_bucketing_from_inductor_configs(
                    graph.owning_module,  # pyrefly: ignore[bad-argument-type]
                )
            )

    if config.aten_distributed_optimizations.enable_low_contention_collectives:
        from torch._inductor.fx_passes.low_contention_collectives import (
            replace_collectives_with_low_contention,
        )

        GraphTransformObserver(
            gm, "replace_collectives_with_low_contention"
        ).apply_graph_pass(replace_collectives_with_low_contention)

    # Keep these last, since they introduce mutation. Look at
    # ./fx_passes/README.md for a discussion of mutation invariants.
    GraphTransformObserver(gm, "reinplace_inplaceable_ops").apply_graph_pass(
        functools.partial(reinplace_inplaceable_ops, fake_tensor_updater),
    )

    # Fix aliasing detection for dtype views AFTER reinplace determines cloning needs
    GraphTransformObserver(gm, "fix_auto_functionalized_dtype_views").apply_graph_pass(
        fix_auto_functionalized_dtype_views
    )

    GraphTransformObserver(
        gm, "decompose_triton_kernel_wrapper_functional"
    ).apply_graph_pass(decompose_triton_kernel_wrapper_functional)
    GraphTransformObserver(gm, "decompose_auto_functionalized").apply_graph_pass(
        decompose_auto_functionalized
    )
    GraphTransformObserver(gm, "decompose_scan_to_while_loop").apply_gm_pass(
        decompose_scan_to_while_loop
    )
    GraphTransformObserver(gm, "decompose_map_to_while_loop").apply_gm_pass(
        decompose_map_to_while_loop
    )

    gm.recompile()
    gm.graph.lint()


def prepare_softmax_pattern(x, dim):
    xmax = x.amax(dim=dim, keepdim=True)
    xsub = x - xmax
    xexp = xsub.exp()
    xsum = xexp.sum(dim=dim, keepdim=True)
    return xmax, xsum, xsub, xexp


def prepare_softmax_replacement(x, dim):
    """
    Return xsub since otherwise log-softmax can not be matched
    due to a use of this intermediate node. Same reason to return
    xsub.exp() for softmax.
    """
    from torch._inductor.inductor_prims import prepare_softmax_online

    xmax, xsum = prepare_softmax_online(x, dim)
    xsub = x - xmax
    return xmax, xsum, xsub, xsub.exp()


def prepare_softmax_extra_check(match):
    """
    We only have triton online softmax kernels currently.
    """
    device_type = match.kwargs["x"].meta["val"].device.type
    return (
        config.online_softmax
        and device_type in ["cuda", "xpu"]
        and getattr(config, f"{device_type}_backend") == "triton"
    )


def decompose_map_to_while_loop(gm: torch.fx.GraphModule):
    """This is similar to decompose_scan_to_while_loop."""
    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.map_impl),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        if len(kwargs) != 0:
            raise AssertionError(
                "kwargs of map are not merged into args before entering decompose_map_to_while_loop_pass"
            )
        subgraph, fx_xs, fx_additional_inputs = args
        sub_gm: torch.fx.GraphModule = getattr(gm, subgraph.target)
        cur_node = match.nodes[0]
        mapped_outputs = cur_node.meta["val"]

        def lower_to_while_loop(*args, **kwargs):
            if len(kwargs) != 0:
                raise AssertionError(f"expected no kwargs, got {kwargs}")
            xs, additional_inputs = pytree.tree_unflatten(args, tree_spec)
            if not (
                isinstance(xs, (tuple, list))
                and isinstance(additional_inputs, (tuple, list))
            ):
                raise AssertionError((xs, additional_inputs))
            map_length = xs[0].size(0)
            loop_idx = torch.zeros([], dtype=torch.int64, device=torch.device("cpu"))

            # Similar to NOTE [Pre-allocate scan's output buffer]
            bound_symbols = {
                arg.node.expr: arg
                for arg in pytree.tree_leaves((args, map_length))
                if isinstance(arg, torch.SymInt)
            }
            out_buffers = [
                torch.empty_strided(
                    resolve_shape_to_proxy(out.size(), bound_symbols),
                    resolve_shape_to_proxy(out.stride(), bound_symbols),
                    device=out.device,
                    dtype=out.dtype,
                    layout=out.layout,
                    requires_grad=out.requires_grad,
                )
                for out in mapped_outputs
            ]

            while_loop_operands = (loop_idx, out_buffers, xs)
            while_loop_flat_operands, operands_spec = pytree.tree_flatten(
                while_loop_operands
            )
            while_loop_additional_inputs = additional_inputs
            _, operands_and_additional_inputs_spec = pytree.tree_flatten(
                (*while_loop_operands, additional_inputs)
            )

            def cond_fn(*flat_args):
                loop_idx, _, _, _ = pytree.tree_unflatten(
                    flat_args,
                    operands_and_additional_inputs_spec,
                )
                return loop_idx < map_length

            def body_fn(*flat_args):
                loop_idx, out_bufs, xs, additional_inputs = pytree.tree_unflatten(
                    flat_args,
                    operands_and_additional_inputs_spec,
                )

                idx_int = loop_idx.item()
                torch.ops.aten._assert_scalar.default(idx_int >= 0, "")
                torch.ops.aten._assert_scalar.default(idx_int < map_length, "")
                sub_xs = [torch.ops.aten.select.int(x, 0, idx_int) for x in xs]
                outs = sub_gm(*sub_xs, *additional_inputs)

                for out, buffer in zip(outs, out_bufs):
                    buffer_slice = torch.ops.aten.select.int(buffer, 0, idx_int)
                    buffer_slice.copy_(out)
                return loop_idx + 1, *out_bufs, *xs

            _, final_out, _ = pytree.tree_unflatten(
                torch.ops.higher_order.while_loop(
                    cond_fn,
                    body_fn,
                    tuple(while_loop_flat_operands),
                    tuple(while_loop_additional_inputs),
                ),
                operands_spec,
            )
            return (final_out,)

        lower_to_while_loop_args, tree_spec = pytree.tree_flatten(
            (fx_xs, fx_additional_inputs)
        )
        match.replace_by_example(
            lower_to_while_loop, lower_to_while_loop_args, run_functional_passes=False
        )

    graph_pass.apply(gm)

    for _node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.map_impl
    ):
        raise AssertionError("map is not lowered to while_loop")


def resolve_shape_to_proxy(
    shape: list[int | torch.SymInt], bound_symbols: dict[Any, Any]
):
    """
    Given a list of symints/ints, this function returns a calculated expression of bound_symbols' values.
    When we trace this function, we'll get a graph with call_function nodes that describes how the shape expr is
    computed from bound_symbols' values.

    Suppose shape = (s1*s2, s1+s2) and bound_symbols = {s1: arg0, s2: arg1}, the result will be
    (arg0 * arg1, arg0 + arg1).
    """
    from torch.utils._sympy.interp import sympy_interp
    from torch.utils._sympy.reference import PythonReferenceAnalysis

    ret = []
    for s in shape:
        if isinstance(s, torch.SymInt):
            ret.append(
                sympy_interp(
                    PythonReferenceAnalysis,
                    bound_symbols,
                    s.node.expr,
                ),
            )
        else:
            if not isinstance(s, int):
                raise AssertionError(f"expected int, got {type(s).__name__}")
            ret.append(s)
    return ret


def decompose_scan_to_while_loop(gm: torch.fx.GraphModule):
    """
    NOTE [decompose scan to while_loop]
    This pass decomposes `scan` to  `while_loop` by replacing the scan fx_node with a while_loop hop.

    Suppose we have a function f:

        def f():
            init = torch.zeros([])
            xs = torch.arange(4)
            ys = []
            for i in range(xs.size(0)):
                init = xs[i] + init
                ys.append(init)

            # Return the final carry and stack the intermediates
            return init, torch.stack(ys)

    We could rewrite it with a scan with the benefits of reducing compilation time/binary size, reducing
    memory usage, supporting loops over unbacked shapes and cudagraph etc.

        def g():
            def step_fn(init: torch.Tensor, x: torch.Tensor):
                next_init = x + init
                return next_init, next_init

            init = torch.zeros([])
            xs = torch.arange(4)
            final_carry, ys = torch._higher_order.scan(step_fn, init, xs)
            return final_carry, ys

    This pass will rewrite scan into:

        def k():
            init = torch.zeros([])
            xs = torch.arange(4)

            # we create a loop_idx and loop through xs.shape[0]
            loop_idx = torch.zeros([])
            ys = torch.empty_strided(_shape_stride_of_ys)
            def cond_fn(loop_idx, ys, init, xs):
                return loop_idx < xs.shape[0]

            # we pre-allocate the output buffer ys and inplace
            # copy the y of each intermediate into a slice.
            # NOTE [Pre-allocate scan's output buffer].
            def body_fn(loop_idx, ys, init, xs):
                int_idx = loop_idx.item()
                next_init, y = step_fn(init, xs[int_idx])
                ys[int_idx].copy_(y)
                return loop_idx + 1, ys, next_init, xs

            final_carry, _, _, ys = torch._higher_order.while_loop(cond_fn, body_fn, (loop_idx, ys, init, xs))
            return final_carry, ys
    """

    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.scan),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._higher_order_ops.scan import _extract_carry_and_out

        if len(kwargs) != 0:
            raise AssertionError(
                "kwargs of scan are not merged into args before entering decompose_scan_to_while_loop_pass"
            )

        combine_subgraph, fx_init, fx_xs, fx_additional_inputs = args
        if combine_subgraph.op != "get_attr":
            raise AssertionError("first arg is not combine_subgraph")
        sub_gm: torch.fx.GraphModule = getattr(gm, combine_subgraph.target)
        cur_node = match.nodes[0]
        num_init_leaves = len(fx_init)
        _, ys_outputs = _extract_carry_and_out(cur_node.meta["val"], num_init_leaves)

        def lower_to_while_loop(*args, **kwargs):
            """
            The traced graph of this function will be used to replace the original scan fx_node.
            """
            if len(kwargs) != 0:
                raise AssertionError(f"expected no kwargs, got {kwargs}")

            # Step 1: construct necessary inputs to while_loop based on scan's input.
            (
                init,
                xs,
                additional_inputs,
            ) = pytree.tree_unflatten(args, tree_spec)
            scan_length = xs[0].size(0)
            if scan_length == 0:
                empty_ys = [
                    torch.empty(
                        [0] + list(ys_out.shape[1:]),
                        dtype=ys_out.dtype,
                        device=ys_out.device,
                    )
                    for ys_out in ys_outputs
                ]
                return list(init) + empty_ys

            loop_idx = torch.zeros([], dtype=torch.int64, device=torch.device("cpu"))

            # NOTE [Pre-allocate scan's output buffer]
            # In order to pre-allocate the output buffer for ys, we rely on the meta of scan's fx_node.
            # However, the meta consists of concrete symints, we need to bind those symints with
            # proxies in order to trace the torch.empty_strided call correctly.
            #
            # Also note that basic free symbols of tensor's shapes are guaranteed to be lifted as subgraph inputs
            # in dynamo so we can always re-construct the sym expression from placeholders.
            # See Note [Auto lift basic free symbols when create_graph_input] for how this is done.
            bound_symbols = {
                arg.node.expr: arg
                for arg in pytree.tree_leaves((args, scan_length))
                if isinstance(arg, torch.SymInt)
            }
            ys_outs = [
                torch.empty_strided(
                    resolve_shape_to_proxy(ys_out.size(), bound_symbols),
                    resolve_shape_to_proxy(ys_out.stride(), bound_symbols),
                    device=ys_out.device,
                    dtype=ys_out.dtype,
                    layout=ys_out.layout,
                    requires_grad=ys_out.requires_grad,
                )
                for ys_out in ys_outputs
            ]

            while_loop_operands = (loop_idx, ys_outs, init, xs)
            flat_operands, operands_spec = pytree.tree_flatten(while_loop_operands)
            _, operands_and_additional_inputs_spec = pytree.tree_flatten(
                (*while_loop_operands, additional_inputs)
            )

            # Step 2: create the cond_fn and body_fn for while_loop
            def cond_fn(*flat_args):
                loop_idx, _, _, _, _ = pytree.tree_unflatten(
                    flat_args, operands_and_additional_inputs_spec
                )  # type: ignore[has-type]
                return loop_idx < scan_length  # type: ignore[has-type]

            def body_fn(*flat_args):
                loop_idx, ys_outs, carry, xs, additional_inputs = pytree.tree_unflatten(
                    flat_args,
                    operands_and_additional_inputs_spec,  # type: ignore[has-type]
                )

                idx_int = loop_idx.item()
                torch.ops.aten._assert_scalar.default(idx_int >= 0, "")
                torch.ops.aten._assert_scalar.default(idx_int < scan_length, "")
                sub_xs = [torch.ops.aten.select.int(x, 0, idx_int) for x in xs]
                next_carry, ys = _extract_carry_and_out(
                    sub_gm(*(list(carry) + sub_xs + list(additional_inputs))),
                    num_init_leaves,
                )
                for y, y_out in zip(ys, ys_outs):
                    y_out_slice = torch.ops.aten.select.int(y_out, 0, idx_int)
                    y_out_slice.copy_(y)
                return loop_idx + 1, *ys_outs, *next_carry, *xs

            # Step 3: call the while_loop operator
            _, ys_outs, last_carry, _ = pytree.tree_unflatten(
                torch.ops.higher_order.while_loop(
                    cond_fn,
                    body_fn,
                    tuple(flat_operands),
                    tuple(additional_inputs),
                ),
                operands_spec,
            )
            return list(last_carry) + list(ys_outs)

        lower_to_while_loop_args, tree_spec = pytree.tree_flatten(
            (
                fx_init,
                fx_xs,
                fx_additional_inputs,
            )
        )
        match.replace_by_example(
            lower_to_while_loop,
            lower_to_while_loop_args,
            run_functional_passes=False,
        )

    graph_pass.apply(gm)

    for _node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.scan
    ):
        raise AssertionError("scan is not lowered to while_loop")


@init_once_fakemode
def lazy_init(input_device: torch.device | None = None):
    if torch._C._has_mkldnn:
        from . import decompose_mem_bound_mm  # noqa: F401
        from .mkldnn_fusion import _mkldnn_fusion_init

        _mkldnn_fusion_init()
    else:
        from .quantization import _register_woq_lowerings

        _register_woq_lowerings()

    # Put this patterns in post-grad pass rather than joint-graph
    # pass since otherwise there will be perf/peak-memory regression:
    # https://github.com/pytorch/pytorch/issues/148141
    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        prepare_softmax_pattern,
        # pyrefly: ignore [bad-argument-type]
        prepare_softmax_replacement,
        [torch.empty(4, 8)],
        scalar_workaround=dict(dim=-1),
        # pyrefly: ignore [bad-argument-type]
        trace_fn=fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_dicts=pass_patterns[1],
        extra_check=prepare_softmax_extra_check,
    )


def reorder_for_locality(graph: torch.fx.Graph):
    if torch.distributed.is_available():

        def check():
            # This is a wait node, and `other_node`` is some collective node.
            # Eager semantics allow waits to be issued in a different order than
            # the collectives. Reordering this wait node might reorder collectives
            # which cause hangs. Once we have SPMD mode, we can safely reorder them.
            # However, increasing the locality between a collective and its wait node
            # is generally worse for performance.
            return node.target != torch.ops._c10d_functional.wait_tensor.default
    else:

        def check():
            return True

    def consumes_rng_state(node: torch.fx.Node) -> bool:
        return (
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.nondeterministic_seeded in node.target.tags
        )

    def visit(other_node):
        if (
            other_node.op == "call_function"
            and other_node.target != operator.getitem
            and all((n in seen_nodes) for n in other_node.users)
            and get_mutation_region_id(graph, node)
            == get_mutation_region_id(graph, other_node)
            and check()
        ):
            # Ops that consume RNG state are order-sensitive and must not be
            # reordered during locality optimization.
            if consumes_rng_state(other_node):
                return

            # move node's producers right before it
            node.prepend(other_node)

    seen_nodes = OrderedSet[torch.fx.Node]()

    # only reorder nodes before the first copy_ in the graph.
    # copy_ will appear at the end of functionalized graphs when there is mutation on inputs,
    # and this reordering doesn't work well with mutation
    first_copy = next(
        iter(graph.find_nodes(op="call_function", target=torch.ops.aten.copy_.default)),
        None,
    )
    past_mutating_epilogue = first_copy is None

    for node in reversed(graph.nodes):
        seen_nodes.add(node)
        if not past_mutating_epilogue:
            past_mutating_epilogue = node is first_copy
            continue

        torch.fx.map_arg((node.args, node.kwargs), visit)


def register_lowering_pattern(
    pattern,
    extra_check=_return_true,
    pass_number=1,
    *,
    output_metadata_ignores_input_storage: bool = True,
    output_metadata_is_input: int | str | None = None,
    output_metadata_fn: Callable[..., Any] | None = None,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    Register an aten to inductor IR replacement pattern
    """
    return pattern_matcher.register_lowering_pattern(
        pattern,
        extra_check,
        # pyrefly: ignore [bad-argument-type]
        pass_dict=pass_patterns[pass_number],
        output_metadata_ignores_input_storage=output_metadata_ignores_input_storage,
        output_metadata_is_input=output_metadata_is_input,
        output_metadata_fn=output_metadata_fn,
    )


################################################################################
# Actual patterns below this point.
# Priority of patterns is:
#   - later output nodes first
#   - order patterns are defined in
################################################################################


def is_valid_mm_plus_mm(match: Match):
    if not (config.max_autotune or config.max_autotune_gemm):
        return False

    # Check if all required values exist
    mat1_val = match.kwargs["mat1"].meta.get("val")
    mat2_val = match.kwargs["mat2"].meta.get("val")
    mat3_val = match.kwargs["mat3"].meta.get("val")
    mat4_val = match.kwargs["mat4"].meta.get("val")

    if mat1_val is None or mat2_val is None or mat3_val is None or mat4_val is None:
        return False

    *_b1, m1, k1 = mat1_val.shape
    *_b2, k2, n1 = mat2_val.shape
    if k1 != k2:
        return False

    *_b1, m2, k3 = mat3_val.shape
    *_b2, k4, n2 = mat4_val.shape
    if k3 != k4:
        return False

    if m1 != m2 or n1 != n2:
        return False

    return True


@register_lowering_pattern(
    CallFunction(
        aten.add,
        CallFunction(aten.mm, KeywordArg("mat1"), KeywordArg("mat2")),
        CallFunction(aten.mm, KeywordArg("mat3"), KeywordArg("mat4")),
    ),
    extra_check=is_valid_mm_plus_mm,
)
def mm_plus_mm(match: Match, mat1, mat2, mat3, mat4):
    return inductor.kernel.mm_plus_mm.tuned_mm_plus_mm(mat1, mat2, mat3, mat4)


@register_graph_pattern(
    CallFunction(
        aten.cumsum.default,
        CallFunction(
            torch.ops.aten.full.default,
            KeywordArg("shape"),
            KeywordArg("fill_value"),
            dtype=KeywordArg("dtype"),
            layout=Ignored(),
            device=KeywordArg("device"),
            pin_memory=False,
            _users=MULTIPLE,
        ),
        KeywordArg("dim"),
        _users=MULTIPLE,
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[1],
)
def pointless_cumsum_replacement(match: Match, shape, fill_value, device, dtype, dim):
    """Based on a pattern in OPTForCausalLM"""

    if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
        # match full()'s fill_value cast
        fill_value = int(bool(fill_value) if is_boolean_dtype(dtype) else fill_value)
        # cumsum promotes all integral types to int64
        dtype = torch.int64

    out_dtype = match.output_node().kwargs.get("dtype") or dtype
    bool_out = is_boolean_dtype(out_dtype)  # pyrefly: ignore[bad-argument-type]
    # pyrefly: ignore[bad-argument-type]
    integral_out = bool_out or is_integer_dtype(out_dtype)
    if integral_out:
        fill_value = int(bool(fill_value) if bool_out else fill_value)
    acc_dtype = torch.int64 if integral_out else torch.float64

    def repl(*shape):
        dim_size = shape[dim]
        idx = torch.arange(1, dim_size + 1, device=device, dtype=acc_dtype)

        inter_shape = [1] * len(shape)
        inter_shape[dim] = dim_size
        return (idx * fill_value).view(inter_shape).expand(shape).to(out_dtype)

    # only replace the output node, not all nodes
    match.nodes = [match.output_node()]
    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, list(shape))


_cat_1 = CallFunction(aten.cat, Arg(), 1, _users=2)


@register_lowering_pattern(
    CallFunction(
        aten.cat,
        [
            _cat_1,
            CallFunction(
                aten.slice,
                _cat_1,
                1,
                0,
                KeywordArg("size"),
            ),
        ],
        1,
    )
)
def cat_slice_cat(match, cat_input, size, dim=1):
    """
    This is an example of a more complex pattern where cat_1 is used
    multiple times inside the pattern.  We fold 2 calls to cat into one.

    Matches:
        cat_1: f32[1024, 4077] = torch.ops.aten.cat.default([add_26, primals_217], 1)
        slice_1: f32[1024, 4077] = torch.ops.aten.slice.Tensor(cat_1, 0, 0, 9223372036854775807)
        slice_2: f32[1024, 19] = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 19)
        cat_2: f32[1024, 4096] = torch.ops.aten.cat.default([cat_1, slice_2], 1)


    Rewrite to:
        slice_2 = torch.ops.aten.slice.Tensor(add_26, 1, 0, 19)
        cat_2 = torch.ops.aten.cat.default([add_26, primals_217, slice2], 1)
    """
    first, *rest = cat_input
    # Optimization is optional, because we can just not fold the cat
    # size should be within first.get_size()[dim] such that the optimization is valid.
    # For negative `end`, we currently fallback to not optimizing.
    if size >= 0 and V.graph.sizevars.statically_known_leq(size, first.get_size()[dim]):
        # fold 2 cats into 1 cat
        return L[aten.cat](
            [
                first,
                *rest,
                L[aten.slice](first, dim, 0, size),
            ],
            dim,
        )
    else:
        # don't expect to hit this case, just fall back
        tmp = L[aten.cat](cat_input, dim)
        return L[aten.cat](
            [
                tmp,
                L[aten.slice](tmp, dim, 0, size),
            ],
            dim,
        )


def is_valid_splitwithsizes_cat(match):
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    get_item_nodes = filter_nodes(match.nodes, operator.getitem)
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    split_node, cat_node = split_nodes[0], cat_nodes[0]
    # The dim of split and cat should match for passthrough
    if get_arg_value(split_node, 2, "dim") != get_arg_value(cat_node, 1, "dim"):
        return False
    get_item_args = OrderedSet(
        get_arg_value(get_item_node, 1) for get_item_node in get_item_nodes
    )
    if None in get_item_args:
        raise AssertionError(f"expected no None in get_item_args, got {get_item_args}")
    split_sizes = get_arg_value(split_node, 1, "split_sizes")
    # All parts of split should be included in the cat
    if get_item_args != OrderedSet(range(len(split_sizes))):
        return False
    # The order of get_item_args should same with cat_node used.
    # For example, if the split_node like split_with_sizes(input, [2, 2, 3], 1),
    # the cat node should be like cat([get_item(0), get_item(1), get_item(2)], 1).
    cat_items_args_order = [
        get_arg_value(item_node, 1) for item_node in get_arg_value(cat_node, 0)
    ]
    if cat_items_args_order != list(range(len(split_sizes))):
        return False

    return True


def same_meta(node1: torch.fx.Node, node2: torch.fx.Node):
    """True if two nodes have the same metadata"""
    val1 = node1.meta.get("val")
    val2 = node2.meta.get("val")
    return (
        val1 is not None
        and val2 is not None
        and isinstance(val1, torch.Tensor)
        and isinstance(val2, torch.Tensor)
        and statically_known_true(sym_eq(val1.size(), val2.size()))
        and val1.layout == val2.layout
        and val1.dtype == val2.dtype
        and val1.device == val2.device
        and (
            val1.layout != torch.strided
            or statically_known_true(sym_eq(val1.stride(), val2.stride()))
        )
        # Check conjugate and negative bits - a clone that resolves these is not a no-op
        and val1.is_conj() == val2.is_conj()
        and val1.is_neg() == val2.is_neg()
    )


noop_registry: dict[Any, Any] = {}


def register_noop_decomp(targets, nop_arg=0):
    def register_fun(cond):
        register_decomposition(targets, registry=noop_registry, unsafe=True)(
            (cond, nop_arg)  # type: ignore[arg-type]
        )
        return cond

    return register_fun


def _needs_spmd_graph_preservation() -> bool:
    """Check if SPMD graph preservation is needed for distributed overlap."""
    return (
        config.aten_distributed_optimizations.enable_overlap_scheduling
        or config.reorder_for_compute_comm_overlap
    )


@register_noop_decomp(aten.slice)
def slice_noop(self, dim=0, start=None, end=None, step=1):
    if _needs_spmd_graph_preservation():
        # Keep no-op slices so all ranks produce identical FX graphs (SPMD)
        # with matching op counts and runtime estimations.
        return False
    if start is None or end is None:
        return False

    slice_dim_size = self.shape[dim]
    if (
        statically_known_true(sym_eq(start, 0))
        and (
            statically_known_true(end >= 2**63 - 1)
            or statically_known_true(end >= slice_dim_size)
        )
        and statically_known_true(sym_eq(step, 1))
    ):
        return True
    return False


@register_noop_decomp(aten.slice_scatter, 1)
def slice_scatter_noop(self, src, dim=0, start=None, end=None, step=1):
    if start is None:
        start = 0
    if end is None:
        end = 2**63 - 1
    slice_scatter_dim_size = self.shape[dim]
    if (
        self.shape == src.shape
        and start == 0
        and (
            statically_known_true(end >= 2**63 - 1)
            or statically_known_true(end >= slice_scatter_dim_size)
        )
        and step == 1
    ):
        return True
    return False


@register_noop_decomp(aten.repeat)
def repeat_noop(self, repeats):
    return all(r == 1 for r in repeats)


@register_noop_decomp(aten.constant_pad_nd)
def constant_pad_nd(x, padding, fill_value=0):
    if _needs_spmd_graph_preservation():
        # Keep no-op pads so all ranks produce identical FX graphs (SPMD)
        # with matching op counts and runtime estimations.
        return False
    return all(p == 0 for p in padding)


@register_noop_decomp(torch.ops.prims.convert_element_type)
def convert_element_type_noop(x, dtype: torch.dtype):
    return x.dtype == dtype


@register_noop_decomp(torch.ops.prims.device_put)
def device_put_noop(x, device, non_blocking=True):
    return x.device == decode_device(device)


@register_noop_decomp([aten.ceil, aten.floor, aten.round, aten.trunc])
def int_noop(x):
    return is_integer_dtype(x.dtype)


@register_noop_decomp([aten.pow])
def pow_noop(a, b):
    return isinstance(b, int) and b == 1


@register_noop_decomp([aten.cat], lambda args: args[0][0])
def cat_noop(inputs, dim=0):
    return len(inputs) == 1


@register_noop_decomp(aten.view.default)
def view_default_noop(arg, size):
    return statically_known_true(sym_eq(arg.shape, tuple(size)))


@register_noop_decomp(aten.view.dtype)
def view_dtype_noop(arg, dtype):
    return arg.dtype == dtype


# Note, we also always have a check for identical metadata, which is why these
# are safe
@register_noop_decomp([aten.copy], nop_arg=1)
@register_noop_decomp([aten.alias, aten.clone])
def true_noop(*args, **kwargs):
    return True


def remove_noop_ops(graph: torch.fx.Graph):
    """
    Removes both operations that are essentially aten.clone and operations that are essentially aten.alias from the graph.
    """
    inputs = OrderedSet[torch.fx.Node]()
    input_storages = OrderedSet[int | None]()
    output_storages = OrderedSet[int | None]()

    for node in graph.find_nodes(op="placeholder"):
        inputs.add(node)
        input_storages.add(get_node_storage(node))

    output_node = next(iter(reversed(graph.nodes)))
    if output_node.op != "output":
        raise AssertionError(f"expected output node, got {output_node.op}")
    outputs = output_node.args[0]
    if not isinstance(outputs, (list, tuple)):
        # nested subgraphs can have singleton outputs
        outputs = (outputs,)
    for out in outputs:
        if isinstance(out, torch.fx.Node):
            output_storages.add(get_node_storage(out))

    for node in graph.nodes:
        if node.target in noop_registry:
            cond, src_index = noop_registry[node.target]
            if isinstance(src_index, int):
                src = node.args[src_index]
            else:
                src = src_index(node.args)
            if not isinstance(src, torch.fx.Node):
                continue

            if node.target is torch.ops.aten.copy.default:
                dst = node.args[0]
                if (
                    isinstance(dst, torch.fx.Node)
                    and dst.op == "call_function"
                    and dst.kwargs.get("pin_memory") is True
                ):
                    continue

            # Don't introduce new aliasing between inputs and outputs.
            # See fx_passes/README.md for a discussion of why this is
            # necessary.
            node_storage = get_node_storage(node)
            src_storage = get_node_storage(src)
            node_is_view = node_storage == src_storage
            if (
                not node_is_view
                and node_storage in output_storages
                and (src_storage in input_storages or src_storage in output_storages)
            ):
                continue

            # Even if input and outputs are expected to alias,
            # don't make "node is src" True
            if (
                node_is_view
                and node in output_node.args
                and (src in inputs or src in output_node.args)
            ):
                continue

            is_valid, args, kwargs = get_fake_args_kwargs(node)
            if not is_valid:
                continue
            if same_meta(node, src) and cond(*args, **kwargs):
                node.replace_all_uses_with(src)
                graph.erase_node(node)


def remove_assert_ops(graph: torch.fx.Graph):
    """
    Removes aten._assert_tensor_metadata.default op because
    1) it will be lowered to a no-op in inductor
    2) it can block fusion, such as unfuse_bias_add_to_pointwise fusion.

    This op could come from aten.to functionalization in export.

    For example, if we have a graph like below

    %addmm = aten.addmm.default(%linear_bias, %arg3_1, %permute)
    %_assert_tensor_metadata = aten._assert_tensor_metadata.default(%addmm, None, None, torch.float16)
    %convert_element_type_3 = prims.convert_element_type.default(%addmm, torch.float32)
    %pow_1 = aten.pow.Tensor_Scalar(%convert_element_type_3, 2)

    We still want to fuse add from addmm with pow, instead of fusing add with mm, according to unfuse_bias_add_to_pointwise fusion.

    However, aten._assert_tensor_metadata.default is not a pointwise op, and would fail the should_prefer_unfused_addmm check.

    We remove this op so it doesn't block fusion decisions. It's safe because this op is lowered to a no-op with @register_lowering.

    """
    for node in graph.find_nodes(
        op="call_function", target=torch.ops.aten._assert_tensor_metadata.default
    ):
        graph.erase_node(node)


def apply_pass_to_subgraphs(pass_fn: Callable[[fx.Graph], None], graph: fx.Graph):
    """Recursively apply a pass function to all subgraphs referenced by get_attr nodes."""
    gm = graph.owning_module
    if gm is None:
        return
    subgraph_names: OrderedSet[str] = OrderedSet(
        x.target for x in graph.find_nodes(op="get_attr")
    )
    for child_name, child_mod in gm.named_children():
        if child_name in subgraph_names and isinstance(child_mod, torch.fx.GraphModule):
            pass_fn(child_mod.graph)


def apply_pass_to_control_deps_subgraphs(
    pass_fn: Callable[[fx.Graph], None], graph: fx.Graph
):
    """Recursively apply a pass function to subgraphs referenced by control_deps."""
    gm = graph.owning_module
    if gm is None:
        return

    for node in graph.find_nodes(op="call_function", target=control_deps):
        if len(node.args) < 2:
            continue
        subgraph_attr = node.args[1]
        if (
            not isinstance(subgraph_attr, torch.fx.Node)
            or subgraph_attr.op != "get_attr"
            or not isinstance(subgraph_attr.target, str)
        ):
            continue
        subgraph = getattr(gm, subgraph_attr.target, None)
        if isinstance(subgraph, torch.fx.GraphModule):
            pass_fn(subgraph.graph)


def _get_single_replacement_node(
    replacement_nodes: Sequence[torch.fx.Node], target: torch.fx.node.Target
) -> torch.fx.Node:
    nodes = [
        node
        for node in replacement_nodes
        if node.op == "call_function" and node.target is target
    ]
    if len(nodes) != 1:
        raise AssertionError(f"Expected exactly one replacement node for {target}")
    return nodes[0]


def _propagate_triton_eager_input_vals(
    replacement_nodes: Sequence[torch.fx.Node],
    hop_node: torch.fx.Node,
) -> None:
    eager_input_vals = hop_node.meta.get("eager_input_vals")
    if eager_input_vals is None:
        return

    _, eager_kwargs = eager_input_vals
    mutation_eager_kwargs = {
        key: value for key, value in eager_kwargs.items() if key != "tensors_to_clone"
    }
    # The dense decomposition introduces clones plus the mutation HOP, but only
    # the mutation HOP should receive the eager-mode tensor metadata.
    mutation_node = _get_single_replacement_node(
        replacement_nodes, torch.ops.higher_order.triton_kernel_wrapper_mutation
    )
    mutation_node.meta["eager_input_vals"] = ((), mutation_eager_kwargs)


def decompose_triton_kernel_wrapper_functional(graph):
    """Decomposes triton_kernel_wrapper_functional nodes into clones and the underlying
    mutation node.

    We assume that the reinplacing pass runs before this; the reinplacing pass
    tells us (via rewriting the arguments or .meta to those nodes) which
    Tensors we should clone and which Tensors are safe to reinplace.
    """
    # First, recursively process any subgraphs
    apply_pass_to_subgraphs(decompose_triton_kernel_wrapper_functional, graph)

    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.triton_kernel_wrapper_functional),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._higher_order_ops.triton_kernel_wrap import (
            triton_kernel_wrapper_functional_dense,
        )

        hop_node = match.nodes[0]
        flat_args, spec = pytree.tree_flatten((args, kwargs))

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args):
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            return (triton_kernel_wrapper_functional_dense(*args, **kwargs),)

        # pyrefly: ignore [bad-argument-type]
        replacement_nodes = match.replace_by_example(
            decomp, flat_args, run_functional_passes=False
        )
        _propagate_triton_eager_input_vals(replacement_nodes, hop_node)

    graph_pass.apply(graph)

    for _ in graph.find_nodes(
        op="call_function",
        target=torch.ops.higher_order.triton_kernel_wrapper_functional,
    ):
        raise AssertionError("triton_kernel_wrapper_functional was not removed")


def fix_auto_functionalized_dtype_views(graph: torch.fx.Graph) -> None:
    """
    Fix aliasing detection for dtype views in auto_functionalized_v2.

    When a dtype view shares storage with a graph input, we can skip cloning
    it during decomposition because the view can be safely regenerated.

    This pass identifies such cases and updates the reinplace metadata to
    indicate no clone is needed.
    """
    storage_to_input: dict[int, torch.fx.Node] = {}
    for node in graph.find_nodes(op="placeholder"):
        storage = get_node_storage(node)
        if storage is not None:
            storage_to_input[storage] = node

    for node in graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.auto_functionalized_v2
    ):
        all_bases = node.kwargs.get("_all_bases")
        only_clone_these = node.meta.get("only_clone_these_tensors")
        if all_bases is None or only_clone_these is None:
            continue

        keep = []
        for idx in only_clone_these:
            if idx >= len(all_bases):
                keep.append(idx)
                continue
            base = all_bases[idx]
            if not isinstance(base, torch.fx.Node):
                keep.append(idx)
                continue
            base_storage = get_node_storage(base)
            if base_storage is None or base_storage not in storage_to_input:
                keep.append(idx)
                continue
            input_val = storage_to_input[base_storage].meta["val"]
            if input_val.dtype == base.meta["val"].dtype:
                keep.append(idx)
                continue
            counters["inductor"]["fix_auto_functionalized_dtype_views"] += 1

        if len(keep) != len(only_clone_these):
            node.meta["only_clone_these_tensors"] = keep


def decompose_auto_functionalized(graph):
    """Decomposes auto_functionalized nodes into clones and the underlying
    mutation node.

    We assume that the reinplacing pass runs before this; the reinplacing pass
    tells us (via rewriting the arguments or .meta to those nodes) which
    Tensors we should clone and which Tensors are safe to reinplace.
    """
    apply_pass_to_control_deps_subgraphs(decompose_auto_functionalized, graph)

    graph_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.auto_functionalized),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._higher_order_ops.auto_functionalize import auto_functionalized_dense

        only_clone_these_tensors = match.nodes[0].meta.get("only_clone_these_tensors")
        if only_clone_these_tensors is not None:
            only_clone_these_tensors = tuple(only_clone_these_tensors)

        flat_args, spec = pytree.tree_flatten((args, kwargs))

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args):
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            if len(args) != 1:
                raise AssertionError(f"expected 1 arg, got {len(args)}")
            mode = args[0]
            return auto_functionalized_dense(mode, only_clone_these_tensors, **kwargs)

        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.higher_order.auto_functionalized_v2),
        # pyrefly: ignore [bad-argument-type]
        pass_dict=graph_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._higher_order_ops.auto_functionalize import (
            auto_functionalized_v2_dense,
        )

        only_clone_these_bases = match.nodes[0].meta.get("only_clone_these_tensors")
        if only_clone_these_bases is not None:
            only_clone_these_bases = tuple(only_clone_these_bases)

        flat_args, spec = pytree.tree_flatten((args, kwargs))

        def _maybe_resolve_constant_get_attr(node):
            # Resolve getattr node to its value because they don't always have meta["val"]
            if (
                isinstance(node, torch.fx.Node)
                and node.op == "get_attr"
                and "val" not in node.meta
            ):
                const_attr = getattr(graph.owning_module, node.target)  # type: ignore[arg-type]
                if not isinstance(const_attr, (torch.fx.GraphModule, pytree.TreeSpec)):
                    raise AssertionError((type(const_attr), const_attr))
                return const_attr
            return node

        flat_args = [_maybe_resolve_constant_get_attr(arg) for arg in flat_args]

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args):
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            if len(args) != 1:
                raise AssertionError(f"expected 1 arg, got {len(args)}")
            mutable_op = args[0]
            return auto_functionalized_v2_dense(
                mutable_op, only_clone_these_bases, **kwargs
            )

        # pyrefly: ignore [bad-argument-type]
        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    graph_pass.apply(graph)

    # Remove unused get_attr nodes and their corresponding attributes from the graph module.
    # When auto_functionalizing a hop, we need to clean up get_attr nodes for _constant_schema
    # and the auto_functionalized graph module that are no longer referenced.
    unused_get_attr_nodes = []
    removable_attrs: OrderedSet[torch.fx.node.Target] = OrderedSet()
    protected_attrs: OrderedSet[torch.fx.node.Target] = OrderedSet()

    # First pass: identify unused get_attr nodes and track attribute usage
    for node in graph.nodes:
        if node.op != "get_attr":
            continue

        if len(node.users) == 0:
            # Node is unused, mark for removal
            unused_get_attr_nodes.append(node)

            # Check if the attribute can be removed from the module
            if (
                hasattr(graph.owning_module, node.target)
                and isinstance(
                    getattr(graph.owning_module, node.target), torch.fx.GraphModule
                )
                and node.target not in protected_attrs
            ):
                removable_attrs.add(node.target)
        else:
            # Node is used, protect its attribute from removal
            if node.target in removable_attrs:
                removable_attrs.remove(node.target)
            protected_attrs.add(node.target)

    # Second pass: clean up unused nodes and attributes
    for node in unused_get_attr_nodes:
        graph.erase_node(node)

    for attr_name in removable_attrs:
        if not isinstance(attr_name, str):
            raise AssertionError(f"expected str, got {type(attr_name).__name__}")
        delattr(graph.owning_module, attr_name)

    graph.lint()

    for _ in graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.auto_functionalized
    ):
        raise AssertionError("auto_functionalized was not removed")

    for _ in graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.auto_functionalized_v2
    ):
        raise AssertionError("auto_functionalized_v2 was not removed")


@register_lowering_pattern(
    CallFunction(
        aten.cat,
        ListOf(
            CallFunction(
                operator.getitem,
                CallFunction(
                    aten.split_with_sizes,
                    KeywordArg("input_"),
                    Ignored(),
                    Ignored(),
                    _users=MULTIPLE,
                ),
                Ignored(),
            ),
        ),
        Ignored(),
    ),
    pass_number=2,
    extra_check=is_valid_splitwithsizes_cat,
    output_metadata_is_input="input_",
)
def splitwithsizes_cat_replace(match, input_):
    return input_


def is_valid_cat_splitwithsizes(match):
    cat_nodes = filter_nodes(match.nodes, aten.cat)
    split_nodes = filter_nodes(match.nodes, aten.split_with_sizes)
    if len(split_nodes) != 1 or len(cat_nodes) != 1:
        return False
    split_node, cat_node = split_nodes[0], cat_nodes[0]

    # the cat node has other users: can't eliminate
    if len(cat_node.users) > 1:
        return False

    # the dim of the cat and split should match
    dim = get_arg_value(split_node, 2, "dim")
    if dim != get_arg_value(cat_node, 1, "dim"):
        return False

    cat_inputs = list(get_arg_value(cat_node, 0))
    split_sizes = get_arg_value(split_node, 1, "split_sizes")
    # the number of input tensors in cat and the
    # length of the split sizes should match
    if len(cat_inputs) != len(split_sizes):
        return False

    for cat_input, split_size in zip(cat_inputs, split_sizes):
        # each cat input tensor's size along dim
        # should match the corresponding split size
        if "val" not in cat_input.meta:
            return False
        cat_input_size = cat_input.meta["val"].size(dim)
        if cat_input_size != split_size:
            return False

    return True


@register_lowering_pattern(
    CallFunction(
        aten.split_with_sizes,
        CallFunction(
            aten.cat,
            KeywordArg("input_"),
            Ignored(),
            _users=MULTIPLE,
        ),
        Ignored(),
        Ignored(),
    ),
    pass_number=2,
    extra_check=is_valid_cat_splitwithsizes,
    output_metadata_is_input="input_",
)
def cat_splitwithsizes_replace(match, input_):
    return input_


# reciprocal(sqrt(x)) -> rsqrt(x): an unconditional algebraic identity
# (1 / sqrt(x) == rsqrt(x)) that saves one op per element in the generated kernel.
@register_graph_pattern(
    CallFunction(
        aten.reciprocal.default,
        CallFunction(aten.sqrt.default, KeywordArg("x")),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[1],
)
def reciprocal_sqrt_to_rsqrt(match: Match, x):
    """reciprocal(sqrt(x)) -> rsqrt(x)"""

    def repl(x):
        return aten.rsqrt(x)

    match.replace_by_example(repl, [x])


def view_to_reshape(gm):
    """
    Replace view ops in the GraphModule to reshape ops.
    """

    def _view_to_reshape_graph(graph):
        for nd in graph.find_nodes(
            op="call_function", target=torch.ops.aten.view.default
        ):
            nd.target = torch.ops.aten.reshape.default

    def _recursive_view_to_reshape(graph):
        apply_pass_to_subgraphs(_recursive_view_to_reshape, graph)
        _view_to_reshape_graph(graph)

    _recursive_view_to_reshape(gm.graph)


def _is_bias_like_addmm_input(inp: torch.fx.Node, output: torch.fx.Node) -> bool:
    if inp.op in ("placeholder", "get_attr"):
        return True

    inp_val = inp.meta.get("val")
    output_val = output.meta.get("val")
    if not (isinstance(inp_val, torch.Tensor) and isinstance(output_val, torch.Tensor)):
        return False

    if len(inp_val.shape) != len(output_val.shape):
        return True

    same_shape = statically_known_true(sym_eq(inp_val.shape, output_val.shape))
    if not same_shape:
        for inp_dim, output_dim in zip(inp_val.shape, output_val.shape):
            if statically_known_true(sym_eq(inp_dim, 1)) and not statically_known_true(
                sym_eq(output_dim, 1)
            ):
                return True
        return False

    return inp_val.layout == torch.strided and any(
        statically_known_true(sym_eq(stride, 0)) for stride in inp_val.stride()
    )


def should_prefer_unfused_addmm(match):
    inp = match.kwargs["inp"]
    if not is_gpu(inp.meta["val"].device.type):
        return False
    if match.output_node().meta.get(_PRESERVE_FLEX_GEMM_GEMM_OP):
        return False
    mat1, mat2 = match.args
    inp_val = inp.meta["val"]
    mat1_val = mat1.meta["val"]
    mat2_val = mat2.meta["val"]
    if inp_val.dtype != mat1_val.dtype or inp_val.dtype != mat2_val.dtype:
        return False
    if inp_val.device != mat1_val.device or inp_val.device != mat2_val.device:
        return False
    beta = match.kwargs.get("beta", 1)
    if inp_val.device.type != "cuda" and beta == 0:
        mm_shape = mat1_val.shape[0], mat2_val.shape[1]
        if not is_expandable_to(inp_val.shape, mm_shape):
            return False

    output = match.output_node()
    if not _is_bias_like_addmm_input(inp, output):
        return False
    return all(is_pointwise_use(use) for use in output.users)


def should_prefer_unfused_baddbmm(match):
    inp = match.kwargs["inp"]
    if not is_gpu(inp.meta["val"].device.type):
        return False
    if match.output_node().meta.get(_PRESERVE_FLEX_GEMM_GEMM_OP):
        return False

    output = match.output_node()
    if not _is_bias_like_addmm_input(inp, output):
        return False
    return all(is_pointwise_use(use) for use in output.users)


@register_graph_pattern(
    CallFunction(
        aten.addmm,
        KeywordArg("inp"),
        Arg(),
        Arg(),
        beta=KeywordArg("beta"),
        alpha=KeywordArg("alpha"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[2],
    extra_check=should_prefer_unfused_addmm,
)
def unfuse_bias_add_to_pointwise(match: Match, mat1, mat2, *, inp, alpha, beta):
    if config.keep_addmm_fused_for_half_dtypes and inp.meta["val"].dtype in (
        torch.bfloat16,
        torch.float16,
    ):
        # Narrowing-cast unfuse (PR #183680) is XPU-only: it preserves
        # precision on XPU pointwise but regresses accuracy on ROCm
        # (basic_gnn_edgecnn training+amp fails on gfx950 otherwise).
        if inp.meta["val"].device.type != "xpu":
            return
        if not (
            inp.op == "call_function"
            and inp.target is torch.ops.prims.convert_element_type.default
            and inp.args[0].meta["val"].dtype.is_floating_point
            and torch.finfo(inp.args[0].meta["val"].dtype).bits
            > torch.finfo(inp.meta["val"].dtype).bits
        ):
            return

    drop_input_for_beta_zero = inp.meta["val"].device.type == "cuda"

    def repl(inp, x1, x2, alpha, beta):
        if alpha == 0 and beta == 0 and drop_input_for_beta_zero:
            return x1.new_zeros((x1.shape[0], x2.shape[1]))
        mm_result = x1 @ x2
        if alpha != 1:
            mm_result = alpha * mm_result
        if beta == 0 and drop_input_for_beta_zero:
            return mm_result
        if beta != 1:
            inp = beta * inp
        return inp + mm_result

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [inp, mat1, mat2, alpha, beta])


@register_graph_pattern(
    CallFunction(
        aten.baddbmm,
        KeywordArg("inp"),
        Arg(),
        Arg(),
        beta=KeywordArg("beta"),
        alpha=KeywordArg("alpha"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[2],
    extra_check=should_prefer_unfused_baddbmm,
)
def unfuse_bias_baddbmm_to_pointwise(match: Match, mat1, mat2, *, inp, alpha, beta):
    if config.keep_addmm_fused_for_half_dtypes and inp.meta["val"].dtype in (
        torch.bfloat16,
        torch.float16,
    ):
        if inp.meta["val"].device.type != "xpu":
            return
        if not (
            inp.op == "call_function"
            and inp.target is torch.ops.prims.convert_element_type.default
            and inp.args[0].meta["val"].dtype.is_floating_point
            and torch.finfo(inp.args[0].meta["val"].dtype).bits
            > torch.finfo(inp.meta["val"].dtype).bits
        ):
            return

    def repl(inp, x1, x2, alpha, beta):
        bmm_result = torch.bmm(x1, x2)
        if alpha != 1:
            bmm_result = alpha * bmm_result
        if beta != 1:
            inp = beta * inp
        return inp + bmm_result

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [inp, mat1, mat2, alpha, beta])


def is_valid_addmm_fusion(match):
    if any(
        node.target is aten.mm.default and node.meta.get(_PRESERVE_FLEX_GEMM_GEMM_OP)
        for node in match.nodes
    ):
        return False

    mat1, mat2 = match.args
    inp = match.kwargs["inp"]

    if not (
        isinstance(inp, torch.fx.Node) and isinstance(inp.meta["val"], torch.Tensor)
    ):
        return False  # Input is a number

    in_shape = inp.meta["val"].shape
    mm_shape = mat1.meta["val"].shape[0], mat2.meta["val"].shape[1]
    matched = is_expandable_to(in_shape, mm_shape)
    if not matched:
        return False  # Shape mismatch

    inp_dtype = inp.meta["val"].dtype

    # aten cublas integration assumes equal dtypes
    if inp_dtype != mat1.meta["val"].dtype or inp_dtype != mat2.meta["val"].dtype:
        return False

    return not should_prefer_unfused_addmm(match)


@register_graph_pattern(
    CallFunction(
        aten.add,
        CallFunction(aten.mm, Arg(), Arg()),
        KeywordArg("inp"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[2],
    extra_check=is_valid_addmm_fusion,
)
@register_graph_pattern(
    CallFunction(
        aten.add,
        KeywordArg("inp"),
        CallFunction(aten.mm, Arg(), Arg()),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[2],
    extra_check=is_valid_addmm_fusion,
)
def addmm(match, mat1, mat2, *, inp):
    def repl(inp, mat1, mat2):
        return aten.addmm(inp, mat1, mat2)

    match.replace_by_example(repl, [inp, mat1, mat2])


def _is_addcdiv_fma_eligible(match: Match) -> bool:
    """Guards for the addcdiv FMA re-fusion pass."""
    # aten.addcdiv requires floating-point self; check inp, not output, because
    # aten.div promotes integers to float so the output is float even for int inp.
    inp_val = match.kwargs["inp"].meta.get("val")
    if not (isinstance(inp_val, torch.Tensor) and inp_val.dtype.is_floating_point):
        return False
    # tl.fma / div_rn are Triton GPU-only
    out_val = match.output_node().meta.get("val")
    if not (
        isinstance(out_val, torch.Tensor) and out_val.device.type in ("cuda", "xpu")
    ):
        return False
    # aten.addcdiv requires all tensor args to be floating-point; integer
    # constants and SymInts can appear as t1/t2 in decomposed graphs.
    for key in ("t1", "t2"):
        node = match.kwargs.get(key)
        val = node.meta.get("val") if isinstance(node, torch.fx.Node) else node
        if not (isinstance(val, torch.Tensor) and val.dtype.is_floating_point):
            return False
    # aten.addcdiv requires a scalar value, not a tensor
    return not isinstance(match.kwargs.get("value"), torch.fx.Node)


@register_graph_pattern(
    CallFunction(
        aten.add.Tensor,
        KeywordArg("inp"),
        CallFunction(
            aten.mul.Tensor,
            CallFunction(aten.div.Tensor, KeywordArg("t1"), KeywordArg("t2")),
            KeywordArg("value"),
        ),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[2],
    extra_check=_is_addcdiv_fma_eligible,
)
def _fuse_addcdiv_to_fma(match: Match, inp, t1, t2, value) -> None:
    """Re-fuse ``inp + (t1/t2)*value`` back into ``aten.addcdiv``.

    torch.addcdiv is CompositeImplicitAutograd: it decomposes into
    aten.div + aten.mul + aten.add before Inductor sees the graph, making the
    FMA-aware lowering in lowering.py unreachable.  Re-inserting a single
    aten.addcdiv node lets that lowering fire (tl.fma + triton.language.div_rn).
    """

    def repl(
        inp: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, value
    ) -> torch.Tensor:
        return torch.ops.aten.addcdiv(inp, t1, t2, value=value)

    counters["inductor"]["addcdiv_fma_fused"] += 1
    match.replace_by_example(repl, [inp, t1, t2, value])


def register_partial_reduction_pattern():
    "Reuse partial reductions in complete reductions"

    # post grad equivalents
    equiv_red = {
        aten.amax.default: aten.max.default,
        aten.amin.default: aten.min.default,
    }

    # TODO: to support other reductions like sum, would need to skip
    # lower precision reductions since partial output would need to be kept at fp32.
    for red_op in (aten.amax.default, aten.amin.default):
        inp = KeywordArg("input")
        partial_reduc = CallFunction(
            red_op, inp, KeywordArg("reduced_dims"), KeywordArg("keepdim")
        )
        full_reduc = CallFunction([red_op, equiv_red[red_op]], inp)

        @register_graph_pattern(
            MultiOutputPattern([partial_reduc, full_reduc]),
            # pyrefly: ignore [bad-argument-type]
            pass_dict=pass_patterns[2],
        )
        def reuse_partial(match, input, reduced_dims, keepdim):
            partial_red, full_red = match.output_nodes()

            # if they're small, reuse not worth it
            if not statically_known_true(input.meta["val"].numel() >= 4096):
                return True

            def replacement(inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                partial = partial_red.target(inp, reduced_dims, keepdim)
                complete = full_red.target(partial)
                return (partial, complete)

            counters["inductor"]["partial_reduction_reuse"] += 1
            match.replace_by_example(replacement, [input])


register_partial_reduction_pattern()


def check_shape_cuda_and_fused_int_mm_mul_enabled(match):
    return (
        config.force_fuse_int_mm_with_mul
        and len(getattr(match.args[2].meta.get("val"), "shape", [])) == 2
        and getattr(match.args[2].meta.get("val"), "is_cuda", False)
    )


def is_index_put_and_requires_h2d_sync_for_gpu_value(node):
    from torch.fx.operator_schemas import normalize_function

    if node.target not in [
        torch.ops.aten.index_put.default,
        torch.ops.aten.index_put_.default,
    ]:
        return False
    # Inductor falls back to aten.index_put_.
    # index_put_ will call nonzero() and perform a H2D sync if
    # any of its indices are bool/byte tensors
    # However, it will short-circuit this H2D sync and run mask_fill_
    # if the value we are putting is a cpu scalar.
    # Therefore, when inductor sees an index_put_ with byte tensor indices,
    # it should *not* convert the cpu scalar value into a gpu tensor.
    args_, _kwargs = normalize_function(node.target, node.args, node.kwargs)  # type: ignore[misc]
    any_byte_bool_indices = False
    indices = args_[1]
    for i in indices:
        if i is not None and i.meta["val"].dtype in [torch.bool, torch.int8]:
            any_byte_bool_indices = True

    val = args_[2].meta["val"]
    val_is_cpu_scalar = val.device.type == "cpu" and val.numel() == 1
    # If both these conditions hold, then converting the val
    # to a gpu tensor will incur a H2D sync when inductor calls aten.index_put_
    return any_byte_bool_indices and val_is_cpu_scalar


class ConstructorMoverPass:
    def __init__(
        self, target: str, allow_outputs: bool = False, allow_inputs: bool = False
    ) -> None:
        """
        Move constructors from cpu to the target_device.

        Sweeps through the module, looking for constructor nodes that can be moved
        to the target_device.

        A constructor node can be moved to the target_device iff all of its users
        can also be moved (tested by cannot_be_moved). Otherwise, all dependent
        constructor nodes won't be moved.

        - target: target device type
        - allow_outputs: allow outputs to be moved
        - allow_inputs: allow inputs to be moved
        """

        self.target = target
        self.allow_inputs = allow_inputs
        self.allow_outputs = allow_outputs

        if not isinstance(target, str):
            raise AssertionError(
                "target should be a string representing the device type. "
                f"Got: {type(target).__name__}"
            )

    def allow_cpu_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node that returns a tensor on the target device may have
        cpu tensors as input.
        """
        return node.target in (
            torch.ops.aten.index.Tensor,
            torch.ops.aten.index_put.default,
            torch.ops.aten.index_put_.default,
            torch.ops.aten.copy.default,
            torch.ops.aten.copy_.default,
            torch.ops.aten.slice_scatter.default,
        )

    def is_on_target_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node is on the target device.
        """
        node_device = self.get_node_device(node)
        return node_device is not None and node_device.type == self.target

    def is_cpu_scalar_tensor(self, node: fx.Node) -> bool:
        """
        Returns whether a node is a cpu scalar tensor.
        """
        device = self.get_node_device(node)
        is_cpu = device is not None and device.type == "cpu"
        ten = node.meta.get("val")
        is_scalar = isinstance(ten, torch.Tensor) and len(ten.size()) == 0
        return is_cpu and is_scalar

    def all_inputs_are_cpu_scalar_or_on_target_device(self, node: fx.Node) -> bool:
        """
        Returns whether a node's inputs are either cpu scalar tensors or
        on the target device.
        """
        inputs = (
            inp
            for inp in itertools.chain(node.args, node.kwargs.values())
            if isinstance(inp, fx.Node)
        )
        return all(
            self.is_cpu_scalar_tensor(inp) or self.is_on_target_device(inp)
            for inp in inputs
        )

    def cannot_be_moved(self, node: fx.Node) -> bool:
        """
        Returns whether a node can be moved to the target device.

        If this function returns False, it means that this node and all of its users
        won't be moved into the target device.
        """
        if node.target == "output":
            return not self.allow_outputs

        if not (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.namespace in ("prims", "aten")
        ):
            return True

        if is_index_put_and_requires_h2d_sync_for_gpu_value(node):
            return True

        return False

    def get_node_device(self, node: fx.Node) -> torch.device | None:
        """
        Get the device of a node.
        """
        ten = node.meta.get("val")
        return None if not isinstance(ten, torch.Tensor) else ten.device

    def get_cpu_indeg_count(self, graph: fx.Graph) -> dict[fx.Node, int]:
        """
        Get the number of cpu inputs to a node
        """
        cpu_indeg: dict[fx.Node, int] = Counter()

        for node in graph.nodes:
            cpu_count = 0

            def add_cpu_inp(node):
                nonlocal cpu_count
                device = self.get_node_device(node)
                cpu_count += device is not None and device.type == "cpu"

            pytree.tree_map_only(fx.Node, add_cpu_inp, (node.args, node.kwargs))

            # pyrefly: ignore [redundant-condition]
            if cpu_count:
                cpu_indeg[node] = cpu_count

        return cpu_indeg

    def __call__(self, graph: fx.Graph) -> None:
        target_devices = OrderedSet[torch.device]()
        constructors = []
        cpu_placeholders: OrderedSet[fx.Node] = OrderedSet()

        for node in graph.nodes:
            device = self.get_node_device(node)
            if device and device.type == self.target:
                target_devices.add(device)

            if (
                self.allow_inputs
                and node.op == "placeholder"
                and self.is_cpu_scalar_tensor(node)
            ):
                cpu_placeholders.add(node)
                constructors.append(node)
                continue

            if not (
                isinstance(node.target, torch._ops.OpOverload)
                and node.target.namespace in ("prims", "aten")
            ):
                continue

            if not torch._subclasses.fake_tensor._is_tensor_constructor(node.target):
                continue

            if node.kwargs.get("device") != torch.device("cpu"):
                continue

            if (
                torch._inductor.config.fallback_random
                and torch.Tag.nondeterministic_seeded in node.target.tags
            ):
                continue

            constructors.append(node)

        # not handling multiple target devices initially
        if not constructors or len(target_devices) != 1:
            return

        movable_constructors = self.find_movable_constructors(graph, constructors)

        target_device = next(iter(target_devices))
        movable_cpu_placeholders = movable_constructors & cpu_placeholders
        if movable_cpu_placeholders:
            node = next(iter(reversed(movable_cpu_placeholders)))
            last_node = node
            unsqueezed_nodes = []
            for elem in movable_cpu_placeholders:
                with graph.inserting_after(last_node):
                    unsqueezed_nodes.append(
                        graph.call_function(torch.ops.aten.unsqueeze.default, (elem, 0))
                    )
                    last_node = unsqueezed_nodes[-1]
            with graph.inserting_after(last_node):
                cpu_concat = graph.call_function(
                    torch.ops.aten.cat.default, (unsqueezed_nodes,)
                )
                last_node = cpu_concat
            with graph.inserting_after(last_node):
                gpu_concat = graph.call_function(
                    torch.ops.prims.device_put.default,
                    (cpu_concat, target_device, True),
                )
                last_node = gpu_concat
            with graph.inserting_after(last_node):
                gpu_split = graph.call_function(
                    torch.ops.aten.unbind.int, (gpu_concat,)
                )
                last_node = gpu_split
            for idx, node in enumerate(movable_cpu_placeholders):
                with graph.inserting_after(last_node):
                    gpu_node = graph.call_function(operator.getitem, (gpu_split, idx))
                    node.replace_all_uses_with(
                        gpu_node,
                        lambda x: x
                        not in [cpu_concat, gpu_concat, gpu_split, gpu_node]
                        + unsqueezed_nodes
                        and x.target != torch.ops.aten.copy_.default
                        and x.target != "output",
                    )
                    last_node = gpu_node

                # noop elimination if there are other device_put for gpu_node to
                # target device. Alternatively, we could just move the other device_put
                # earlier in the graph, but that is not supported in fx graph yet.
                noop_device_puts = [
                    user
                    for user in gpu_node.users
                    if user.target is torch.ops.prims.device_put.default
                    and user.args[1] == target_device
                ]
                for noop in noop_device_puts:
                    noop.replace_all_uses_with(gpu_node)
                    graph.erase_node(noop)

        movable_constructors -= movable_cpu_placeholders
        for node in movable_constructors:
            kwargs = node.kwargs.copy()
            kwargs["device"] = target_device
            node.kwargs = kwargs

    def find_movable_constructors(
        self, graph: fx.Graph, constructors: list[fx.Node]
    ) -> OrderedSet[fx.Node]:
        """
        Starting from the cpu constructors, iterate through the graph and test that all of their
        downstream uses can safely be moved to cpu.
        """
        cpu_indeg: dict[fx.Node, int] = self.get_cpu_indeg_count(graph)

        # which constructors cannot be moved to gpu
        cannot_move_to_gpu = OrderedSet[fx.Node]()

        # For any node in the graph, which constructors does it have a dependency on
        constructor_dependencies: dict[fx.Node, OrderedSet[fx.Node]] = defaultdict(
            OrderedSet
        )

        # if a cpu node has a dependency on two different cpu constructors,
        # then if either constructor cannot be moved to gpu, the other cannot as well.
        # In this case any node with a dependency on one will have a dependency on the other
        equal_constructor_sets: dict[fx.Node, OrderedSet[fx.Node]] = {
            c: OrderedSet([c]) for c in constructors
        }

        def make_dependencies_equivalent(
            set1: OrderedSet[fx.Node], set2: OrderedSet[fx.Node]
        ) -> OrderedSet[fx.Node]:
            # could use union find but not worth complexity here
            set1.update(set2)
            for obj in set1:
                equal_constructor_sets[obj] = set1
            return set1

        queue: list[fx.Node] = list(constructors)

        for c in queue:
            constructor_dependencies[c].add(c)

        while queue:
            node = queue.pop()
            dependencies = constructor_dependencies[node]

            for user in node.users:
                if self.cannot_be_moved(user):
                    cannot_move_to_gpu.update(dependencies)
                    break

                # this node was used on an op which takes in multiple devices and output a gpu
                # tensor. we can convert its cpu input to gpu without making further changes
                if self.allow_cpu_device(user) and self.is_on_target_device(user):
                    del cpu_indeg[user]
                elif (
                    self.allow_inputs
                    and self.is_on_target_device(user)
                    and self.all_inputs_are_cpu_scalar_or_on_target_device(user)
                ):
                    # this node takes only cpu scalar tensors or gpu tensors as inputs
                    # and outputs a gpu tensor. we can convert its cpu scalar inputs to gpu
                    # without making further changes
                    del cpu_indeg[user]
                else:
                    # otherwise, we should continue look at its downstream uses
                    cpu_indeg[user] -= 1
                    if cpu_indeg[user] == 0:
                        del cpu_indeg[user]
                        queue.append(user)

                unioned_set = make_dependencies_equivalent(
                    dependencies, constructor_dependencies[user]
                )
                constructor_dependencies[user] = unioned_set

        for node in cpu_indeg:
            if constructor_dependencies[node]:
                cannot_move_to_gpu.update(constructor_dependencies[node])

        all_cannot_move_to_gpu = cannot_move_to_gpu.copy()
        for constructor in cannot_move_to_gpu:
            all_cannot_move_to_gpu.update(equal_constructor_sets[constructor])

        return OrderedSet(constructors) - all_cannot_move_to_gpu


def move_constructors_to_gpu(graph: fx.Graph) -> None:
    """
    Moves intermediary tensors which are constructed on the cpu to gpu when safe
    """

    # cudagraph does not support cpu tensors. In this pass, we update the graph
    # by explicitly moving cpu scalar tensors to gpu when profitable, relying on
    # graph partition to split off this data copy, and cudagraphifying
    # the remaining gpu ops.
    allow_inputs_outputs = bool(
        torch._inductor.config.triton.cudagraphs
        and torch._inductor.config.graph_partition
    )
    ConstructorMoverPass(
        get_gpu_type(),
        allow_inputs=allow_inputs_outputs,
        allow_outputs=allow_inputs_outputs,
    )(graph)
<|endoftext|># mypy: allow-untyped-defs
"""
Effect ordering pass for inductor.

This pass adds ordering dependencies to FX graphs using the control_deps HOP
for precise control over scheduling constraints. When you need exact ordering between
operations (e.g., collective_start -> mm -> wait), this pass wraps operations
with control_deps to make dependencies explicit.
"""

from operator import attrgetter
from typing import Any

import torch.fx as fx
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx._lazy_graph_module import _LazyGraphModule
from torch.utils._ordered_set import OrderedSet


class ControlDeps(HigherOrderOperator):
    """
    Higher-order operator that enforces ordering by making dependencies explicit.

    Schema: control_deps(additional_deps, subgraph, *args, **kwargs) -> result
    where:
    - additional_deps: tuple of tensors that must be computed before this op
      (ordering-only, not a real data use)
    - subgraph: GraphModule containing the exact operation to execute
    - *args: pass-through arguments forwarded to the subgraph

    Semantics:
    - All tensors in additional_deps are computed before the subgraph executes.
    - Pass-through args (inputs returned unchanged by the subgraph) are
      versioned: future readers are ordered after all subgraph operations
      via a rename chain (OrderingOutput).  This ensures that consumers of
      a pass-through value cannot be scheduled before the subgraph's sync
      ops (e.g. wait_event) complete.
    """

    def __init__(self) -> None:
        super().__init__("control_deps")

    def __call__(self, additional_deps, subgraph, *args, **kwargs):
        """Call the operator with dependencies and subgraph.

        Args:
            additional_deps: Tuple of tensors that must be computed first
            subgraph: GraphModule containing the exact operation to execute
            *args: Arguments to pass to the subgraph
        """
        if not isinstance(additional_deps, (tuple, list)):
            raise TypeError(
                f"additional_deps must be tuple/list, got {type(additional_deps).__name__}"
            )
        if not (isinstance(subgraph, fx.GraphModule) or callable(subgraph)):
            raise TypeError(
                f"subgraph must be GraphModule or callable, got {type(subgraph).__name__}"
            )
        # pyrefly: ignore [missing-attribute]
        return super().__call__(additional_deps, subgraph, *args, **kwargs)


control_deps = ControlDeps()

# control_deps wraps side-effecting ops (e.g. record_event, wait_event)
# and must not be eliminated by DCE even when its outputs are unused.
from torch.fx.node import has_side_effect


has_side_effect(control_deps)


# Register fake implementation for tracing
@register_fake(control_deps)
def _(additional_deps, subgraph, *args, **kwargs):
    """Fake tensor implementation - execute the subgraph."""
    return subgraph(*args, **kwargs)


# Register eager execution implementation
@control_deps.py_impl(DispatchKey.CompositeExplicitAutograd)
def control_deps_eager(additional_deps, subgraph, *args, **kwargs):
    """Eager implementation - just execute the subgraph."""
    return subgraph(*args, **kwargs)


# Autograd impl needed because additional_deps tensors may have autograd state,
# causing dispatch through AutogradCUDA even in post-autograd graphs.
@control_deps.py_impl(DispatchKey.Autograd)
def control_deps_autograd(additional_deps, subgraph, *args, **kwargs):
    return subgraph(*args, **kwargs)


def get_subgraph_name(gm: fx.GraphModule, name):
    name = f"subgraph_{name}"

    if not hasattr(gm, name):
        return name

    i = 0
    while hasattr(gm, f"{name}_{i}"):
        i += 1

    return f"{name}_{i}"


def _extract_unique_nodes(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[list[fx.Node], list[Any], Any]:
    """Extract unique fx.Node instances from args/kwargs using pytree.

    Args:
        args: The positional arguments (may contain nested structures with fx.Node)
        kwargs: The keyword arguments (may contain nested structures with fx.Node)

    Returns:
        - Ordered list of unique fx.Node instances (preserves first occurrence order)
        - Flattened list of all items from args/kwargs
        - The pytree spec for reconstructing the original structure
    """
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))
    unique_nodes: list[fx.Node] = []
    seen: OrderedSet[fx.Node] = OrderedSet()
    for item in flat_args_kwargs:
        if isinstance(item, fx.Node) and item not in seen:
            unique_nodes.append(item)
            seen.add(item)
    return unique_nodes, flat_args_kwargs, spec


def preserve_node_ordering(
    graph: fx.Graph,
    additional_deps_map: dict[fx.Node, OrderedSet[fx.Node]],
    verbose: bool = False,
) -> None:
    """
    Preserve node ordering using control_deps HOP with subgraph.

    This function wraps operations with control_deps that:
    1. Makes additional dependencies explicit (first argument)
    2. Creates a subgraph internally to preserve the exact original operation
    3. Preserves the original node names

    Args:
        graph: The FX graph to modify
        additional_deps_map: Mapping from dependent nodes to their dependencies
        verbose: If True, print debug information
    """
    if not additional_deps_map:
        return

    # Track replacements so we can update dependencies
    replacements: dict[fx.Node, fx.Node] = {}

    # Process each node that needs additional dependencies
    for dependent_node, dep_nodes in additional_deps_map.items():
        if dependent_node.op != "call_function":
            raise AssertionError(dependent_node.op)

        original_name = dependent_node.name
        original_args = dependent_node.args
        original_kwargs = dependent_node.kwargs
        original_meta = dependent_node.meta.copy()

        updated_dep_nodes = [replacements.get(dep, dep) for dep in dep_nodes]

        # Create a subgraph that preserves the exact original operation
        subgraph_module = _create_subgraph_for_node(graph, dependent_node)

        owning_mod = graph.owning_module
        if owning_mod is None:
            raise AssertionError("expected graph to have an owning_module")
        subgraph_attr_name = get_subgraph_name(owning_mod, original_name)
        setattr(graph.owning_module, subgraph_attr_name, subgraph_module)

        # Create control_deps call with:
        # 1. Additional dependencies as first arg (explicit)
        # 2. Subgraph via get_attr (like b2b gemm pass)
        # 3. Original arguments (only fx.Node args and kwargs are passed)
        with graph.inserting_before(dependent_node):
            # Create get_attr node for the subgraph
            get_subgraph = graph.get_attr(subgraph_attr_name)

            # Extract unique nodes from nested args/kwargs
            node_args, _, _ = _extract_unique_nodes(original_args, original_kwargs)

            # Create with temporary name first
            ordered_node = graph.call_function(
                control_deps,
                args=(
                    tuple(updated_dep_nodes),  # additional_deps
                    get_subgraph,  # subgraph via get_attr (like b2b gemm)
                    *node_args,  # original node arguments (from both args and kwargs)
                ),
                kwargs={},
                name=f"__temp_{original_name}",  # Temporary name to avoid conflict
            )

        # Copy metadata from original node
        ordered_node.meta = original_meta
        # this will be constrained on the target node in subgraph if it exists
        ordered_node.meta.pop("eager_input_vals", None)

        # Replace all uses of the original node with the ordered version
        dependent_node.replace_all_uses_with(ordered_node)

        # Remove the original node from the graph
        graph.erase_node(dependent_node)

        # Now rename the ordered node to the original name
        ordered_node.name = original_name  # PRESERVE ORIGINAL NAME

        # Track the replacement for future dependencies
        replacements[dependent_node] = ordered_node


def _create_subgraph_for_node(
    graph: fx.Graph, node: fx.Node, additional_deps=None
) -> fx.GraphModule:
    """
    Create a subgraph that exactly recreates a node's operation optionally passing through additional dependencies.

    The subgraph takes only the fx.Node arguments and recreates the operation
    with the exact target, args structure, and kwargs.

    Args:
        graph: The parent graph
        node: The node to wrap in a subgraph
        additional_deps: Additional dependencies to pass through the subgraph

    Returns:
        A GraphModule containing the subgraph
    """
    # Get the owning module
    owning_module = graph.owning_module
    if owning_module is None:
        raise AssertionError("graph.owning_module must not be None")

    # Create a new graph for the subgraph
    subgraph = fx.Graph(owning_module)

    # Extract unique nodes and get flattened structure + spec
    unique_nodes, flat_args_kwargs, spec = _extract_unique_nodes(node.args, node.kwargs)

    # Create placeholders for each unique node
    node_to_placeholder: dict[fx.Node, fx.Node] = {}
    for idx, orig_node in enumerate(unique_nodes):
        placeholder = subgraph.placeholder(f"arg_{idx}")
        if "val" in orig_node.meta:
            placeholder.meta.update(orig_node.meta)
        elif orig_node.op == "get_attr" and isinstance(orig_node.target, str):
            placeholder.meta["val"] = attrgetter(orig_node.target)(owning_module)
        node_to_placeholder[orig_node] = placeholder

    # Replace fx.Node instances with their placeholders
    def replace_nodes(item: Any) -> Any:
        if isinstance(item, fx.Node):
            return node_to_placeholder[item]
        return item

    additional_deps_placeholders = []
    for idx, dep in enumerate(additional_deps or ()):
        placeholder = subgraph.placeholder(f"dep_{idx}")
        if "val" in dep.meta:
            placeholder.meta.update(dep.meta)
        additional_deps_placeholders.append(placeholder)

    new_flat = [replace_nodes(item) for item in flat_args_kwargs]
    new_args, new_kwargs = pytree.tree_unflatten(new_flat, spec)

    # Recreate the exact original operation in the subgraph
    if not callable(node.target):
        raise AssertionError(f"expected node.target to be callable, got {node.target}")
    result = subgraph.call_function(
        node.target,
        tuple(new_args),
        new_kwargs,  # type: ignore[arg-type]
    )

    # Copy metadata from the original node
    result.meta.update(node.meta)

    if additional_deps_placeholders:
        outputs = tuple([result] + additional_deps_placeholders)
        out = subgraph.output(outputs)
        out.meta["val"] = tuple(output.meta.get("val") for output in outputs)
    else:
        out = subgraph.output(result)
        if "val" in result.meta:
            out.meta["val"] = result.meta["val"]

    return _LazyGraphModule(owning_module, subgraph)
<|endoftext|># mypy: allow-untyped-defs
"""
Partitioned scatter optimization for high-contention index_put operations.

Algorithm:
  1. Assign each write operation a partition: partition_id = op_id & (P - 1)
  2. Scatter into an expanded buffer of size P * dim_size along scatter_dim
  3. Reshape to [..., P, dim_size, ...] and sum across partitions
  4. Add result to the original input
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.fx as fx
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.fx_passes.memory_estimator import build_memory_profile
from torch._inductor.pattern_matcher import (
    Arg,
    CallFunction,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._logging import getArtifactLogger
from torch.fx.experimental.symbolic_shapes import optimization_hint


log = logging.getLogger(__name__)
_artifact_log = getArtifactLogger(__name__, "partitioned_scatter")
aten = torch.ops.aten
prims = torch.ops.prims

_INDEX_PUT_TARGETS = (aten.index_put.default, aten.index_put_.default)


@dataclass
class ScatterCandidate:
    """An index_put(accumulate=True) op that passed the cheap (non-memory) gates."""

    output_node: fx.Node
    index_node: fx.Node
    scatter_dim: int
    output_size: int
    index_size: int
    scatter_dim_size: int
    element_bytes: int
    contention_ratio: float
    dtype: torch.dtype


@dataclass
class ScatterMemoryState:
    # Peak live GPU bytes at each compute node in the original (un-transformed)
    # graph, taken at the allocation phase (before this node's last-use inputs
    # are freed) so the expanded scatter buffer is charged against the true peak.
    peak_mem_by_node: list[int]

    node_to_idx: dict[fx.Node, int]

    # total_gpu_memory - non_model_floor_bytes
    allowed_peak_bytes: int

    total_gpu_bytes: int
    non_model_floor_bytes: int

    # Expanded-buffer bytes already granted to scatters transformed earlier in this
    # invocation. peak_mem_by_node profiles the *original* graph, so without this every
    # candidate is sized against the same baseline and each claims the whole budget.
    # A running sum rather than a per-live-range charge because scatters sharing an
    # input fuse into one kernel, which needs all their expanded buffers live at once.
    committed_overhead_bytes: int = 0


@dataclass
class ScatterPassContext:
    """Per-invocation state, passed to the pattern callbacks via closures."""

    # index_put nodes that survived the cheap pre-scan, keyed by output node.
    candidates: dict[fx.Node, ScatterCandidate] = field(default_factory=dict)

    # Built lazily, only when at least one candidate survives the pre-scan.
    memory: "ScatterMemoryState | None" = None

    n_candidates: int = 0
    n_applied: int = 0
    skip_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    applied_partitions: list[int] = field(default_factory=list)


def _record_skip(
    ctx: ScatterPassContext, reason: str, node_name: str, *args: Any
) -> None:
    ctx.skip_reasons[reason] += 1
    counters["inductor"][f"partitioned_scatter_skipped_{reason}"] += 1
    if _artifact_log.isEnabledFor(logging.DEBUG):
        fmt = f"partitioned_scatter: SKIP node=%s reason={reason}"
        if args:
            fmt += " " + " ".join(str(a) for a in args)
        _artifact_log.debug(fmt, node_name)


def _evaluate_candidate(
    output_node: fx.Node, force: bool, ctx: ScatterPassContext
) -> "ScatterCandidate | None":
    """
    Cheap (non-memory) gates, in order:
      1. Single non-None index (multi-axis not supported)
      2. Valid tensor metadata
      3. Runs on a CUDA device (this optimization targets GPU atomic contention)
      4. Non-bool dtype
      5. scatter_dim in bounds
      6. Resolvable, non-zero sizes
      7. index_size >= min_index_size
      8. contention_ratio >= threshold  (uses scatter_dim_size, not output_numel)
    """
    node_name = output_node.name

    input_node = output_node.args[0]
    indices_arg = output_node.args[1]

    if not isinstance(input_node, fx.Node):
        _record_skip(ctx, "input_not_node", node_name)
        return None

    scatter_dim, index_node = _extract_scatter_dim_and_index(indices_arg)
    if scatter_dim is None or index_node is None:
        _record_skip(ctx, "multi_index", node_name)
        return None

    input_meta = _get_tensor_meta(input_node)
    index_meta = _get_tensor_meta(index_node)
    if not input_meta or not index_meta:
        _record_skip(ctx, "no_meta", node_name)
        return None

    # Only rewrite accelerator ops: the expanded-buffer trade-off targets GPU
    # atomic contention and is a pessimization on CPU. Matches the non-CPU
    # device_filter that the memory estimator uses.
    if input_meta["device"].type == "cpu":
        _record_skip(ctx, "cpu_device", node_name)
        return None

    if input_meta["dtype"] == torch.bool or index_meta["dtype"] == torch.bool:
        _record_skip(ctx, "bool_dtype", node_name)
        return None

    if scatter_dim >= len(input_meta["shape"]):
        _record_skip(ctx, "dim_out_of_bounds", node_name)
        return None

    output_size = _resolve_numel(input_meta["numel"])
    index_size = _resolve_numel(index_meta["numel"])

    if output_size is None or index_size is None:
        _record_skip(ctx, "dynamic_no_hint", node_name)
        return None

    if output_size == 0 or index_size == 0:
        _record_skip(ctx, "zero_size", node_name)
        return None

    # Contention is per scatter-dim slot, so use scatter_dim_size as denominator.
    # For a [vocab, dim] output, ratio = N/vocab, not N/(vocab*dim).
    scatter_dim_size = _resolve_numel(input_meta["shape"][scatter_dim])
    if scatter_dim_size is None or scatter_dim_size == 0:
        _record_skip(ctx, "zero_size", node_name)
        return None

    min_index_size: int = config.partitioned_scatter_min_index_size
    if not force and index_size < min_index_size:
        _record_skip(
            ctx,
            "index_too_small",
            node_name,
            f"index_size={index_size} min={min_index_size}",
        )
        return None

    contention_ratio = index_size / scatter_dim_size
    min_contention: float = config.partitioned_scatter_min_contention_ratio
    if not force and contention_ratio < min_contention:
        _record_skip(
            ctx,
            "low_contention",
            node_name,
            f"contention_ratio={contention_ratio:.3f} threshold={min_contention:.3f}",
        )
        return None

    return ScatterCandidate(
        output_node=output_node,
        index_node=index_node,
        scatter_dim=scatter_dim,
        output_size=output_size,
        index_size=index_size,
        scatter_dim_size=scatter_dim_size,
        element_bytes=input_meta["dtype"].itemsize,
        contention_ratio=contention_ratio,
        dtype=input_meta["dtype"],
    )


def _scan_candidates(graph: fx.Graph, ctx: ScatterPassContext) -> None:
    """
    Cheap pre-scan for index_put(accumulate=True) ops that could be rewritten.

    Applies every gate that does not require the memory profile (device, dtype,
    shape, min index size, contention ratio) so we can skip building the
    memory profile entirely when no op would qualify.
    """
    force: bool = config.partitioned_scatter_force

    for node in graph.nodes:
        if node.op != "call_function" or node.target not in _INDEX_PUT_TARGETS:
            continue
        args = node.args
        if len(args) < 4 or args[3] is not True:
            continue

        ctx.n_candidates += 1
        candidate = _evaluate_candidate(node, force, ctx)
        if candidate is not None:
            ctx.candidates[node] = candidate


def _build_scatter_memory_state(graph: fx.Graph) -> "ScatterMemoryState | None":
    """
    Build a per-node peak-memory profile of the original graph.
    Returns None when CUDA is unavailable or the profile can't be built; the
    pass then runs unconstrained by the memory budget.
    """
    if not torch.cuda.is_available():
        return None

    _, total_gpu = torch.cuda.mem_get_info()

    floor_bytes: int = config.partitioned_scatter_non_model_floor_bytes
    allowed_peak = max(0, total_gpu - floor_bytes)

    def is_releasable(n: fx.Node) -> bool:
        return not n.name.startswith("primals")

    # build_memory_profile emits two entries per compute node: the first after
    # its allocations (the node-local peak, while last-use inputs are still
    # live) and the second after its deallocations. We key on the allocation
    # entry so the expanded scatter buffer is charged against the real peak.
    # Unbacked symbolic sizes have no optimization_hint, so sizing the profile
    # can raise. Memory gating is a heuristic, so fall back to running the pass
    # unconstrained rather than failing the compile.
    try:
        profile = build_memory_profile(graph, is_releasable)
    except Exception as e:
        _artifact_log.debug(
            "partitioned_scatter: build_memory_profile failed (%s), "
            "running without memory gating",
            e,
        )
        return None

    compute_nodes = [
        n for n in graph.nodes if n.op not in ("placeholder", "get_attr", "output")
    ]
    node_to_idx: dict[fx.Node, int] = {}
    peak_mem: list[int] = []
    for i, node in enumerate(compute_nodes):
        alloc_idx = 1 + 2 * i
        if alloc_idx >= len(profile):
            break
        node_to_idx[node] = i
        peak_mem.append(profile[alloc_idx])

    _artifact_log.debug(
        "partitioned_scatter: memory state built — "
        "graph_nodes=%d compute_nodes=%d "
        "total_gpu=%d MB floor=%d MB allowed_peak=%d MB",
        sum(1 for _ in graph.nodes),
        len(compute_nodes),
        total_gpu // 1_000_000,
        floor_bytes // 1_000_000,
        allowed_peak // 1_000_000,
    )

    return ScatterMemoryState(
        peak_mem_by_node=peak_mem,
        node_to_idx=node_to_idx,
        allowed_peak_bytes=allowed_peak,
        total_gpu_bytes=total_gpu,
        non_model_floor_bytes=floor_bytes,
    )


def _compute_num_partitions(
    available_bytes: int,
    output_size: int,
    element_bytes: int,
    min_p: int,
    max_p: int,
    index_size: int = 0,
    scatter_dim_size: int = 0,
    force: bool = False,
) -> int:
    """
    Return the largest power-of-2 P in [min_p, max_p] satisfying:
      1. Memory: output_size * element_bytes * (P - 1) <= available_bytes
      2. Diminishing-returns cap (skipped when force=True):
         P <= 4 * writes_per_slot, where writes_per_slot = index_size / scatter_dim_size.
         Past ~4W partitions for a slot taking W writes most partition slots are
         never written, so the zero-fill and reduce over them buy nothing.

    Returns 0 if min_p doesn't fit. Power-of-2 is required by the bitwise-AND
    partition assignment.
    """
    if available_bytes <= 0 or output_size == 0 or element_bytes == 0:
        return 0

    max_raw = available_bytes / (output_size * element_bytes) + 1
    if max_raw < min_p:
        return 0

    p = 2 ** int(math.log2(max_raw))
    p = min(p, max_p)

    if not force and index_size > 0 and scatter_dim_size > 0:
        writes_per_slot = index_size / scatter_dim_size
        contention_cap = max(min_p, 2 ** int(math.log2(max(1, 4 * writes_per_slot))))
        p = min(p, contention_cap)

    return p


def _check_memory(
    state: ScatterMemoryState,
    candidate: ScatterCandidate,
    force: bool = False,
) -> int:
    """
    Compute num_partitions for this candidate given the peak-memory profile.
    Returns num_partitions >= min_p, or 0 if the memory constraint cannot be met.
    """
    min_p = config.partitioned_scatter_min_partitions
    max_p = config.partitioned_scatter_max_partitions

    idx = state.node_to_idx.get(candidate.output_node)
    if idx is None:
        # No profile entry means we cannot bound this scatter's peak, so fail
        # closed: granting max_p here would allocate max_p-1 extra copies of the
        # output with no memory check at all.
        return 0

    baseline = state.peak_mem_by_node[idx]
    available = state.allowed_peak_bytes - baseline - state.committed_overhead_bytes

    num_partitions = _compute_num_partitions(
        available,
        candidate.output_size,
        candidate.element_bytes,
        min_p,
        max_p,
        index_size=candidate.index_size,
        scatter_dim_size=candidate.scatter_dim_size,
        force=force,
    )

    if _artifact_log.isEnabledFor(logging.DEBUG):
        overhead = (
            candidate.output_size * candidate.element_bytes * max(0, num_partitions - 1)
        )
        _artifact_log.debug(
            "partitioned_scatter: memory check node=%s "
            "baseline_peak=%d MB committed=%d MB available=%d MB "
            "expanded_buffer_cost=%d MB num_partitions=%d "
            "total_gpu=%d MB floor=%d MB allowed_peak=%d MB",
            candidate.output_node.name,
            baseline // 1_000_000,
            state.committed_overhead_bytes // 1_000_000,
            available // 1_000_000,
            overhead // 1_000_000,
            num_partitions,
            state.total_gpu_bytes // 1_000_000,
            state.non_model_floor_bytes // 1_000_000,
            state.allowed_peak_bytes // 1_000_000,
        )

    return num_partitions


def _resolve_numel(numel: Any) -> int | None:
    """Resolve numel to a concrete int, handling SymInt via optimization_hint."""
    if isinstance(numel, torch.SymInt):
        hint = optimization_hint(numel)
        if hint is None:
            return None
        return hint * 2  # 2× safety margin for dynamic shapes
    return int(numel)


def _validate_memory(match: Match, ctx: ScatterPassContext, force: bool) -> bool:
    """
    Second-stage gate (the memory budget) for a matched index_put node.

    The cheap gates already ran in the pre-scan; here we only look up the
    surviving candidate and size the partition count against the memory budget.
    """
    output_node = match.output_node()
    candidate = ctx.candidates.get(output_node)
    if candidate is None:
        # Node failed a cheap gate in the pre-scan (already recorded).
        return False

    min_p: int = config.partitioned_scatter_min_partitions
    max_p: int = config.partitioned_scatter_max_partitions

    if ctx.memory is not None:
        num_partitions = _check_memory(ctx.memory, candidate, force=force)
    else:
        num_partitions = _compute_num_partitions(
            available_bytes=2**62,
            output_size=candidate.output_size,
            element_bytes=candidate.element_bytes,
            min_p=min_p,
            max_p=max_p,
            index_size=candidate.index_size,
            scatter_dim_size=candidate.scatter_dim_size,
            force=force,
        )

    if num_partitions < min_p:
        _record_skip(
            ctx,
            "memory_budget",
            output_node.name,
            f"num_partitions={num_partitions} min={min_p}",
        )
        return False

    match._num_partitions = num_partitions  # type: ignore[attr-defined]
    match._scatter_dim = candidate.scatter_dim  # type: ignore[attr-defined]
    match._index_node = candidate.index_node  # type: ignore[attr-defined]
    match._output_size = candidate.output_size  # type: ignore[attr-defined]
    match._element_bytes = candidate.element_bytes  # type: ignore[attr-defined]

    if _artifact_log.isEnabledFor(logging.DEBUG):
        _artifact_log.debug(
            "partitioned_scatter: APPLY node=%s "
            "num_partitions=%d scatter_dim=%d "
            "contention_ratio=%.1f (index_size=%d / scatter_dim_size=%d) "
            "output_size=%d dtype=%s%s",
            output_node.name,
            num_partitions,
            candidate.scatter_dim,
            candidate.contention_ratio,
            candidate.index_size,
            candidate.scatter_dim_size,
            candidate.output_size,
            candidate.dtype,
            " [force]" if force else "",
        )

    return True


def _create_replacement(
    match: Match, ctx: ScatterPassContext, input_tensor, indices, values
) -> None:
    """Replace high-contention index_put with partitioned scatter."""
    num_partitions: int = match._num_partitions  # type: ignore[attr-defined]
    scatter_dim: int = match._scatter_dim  # type: ignore[attr-defined]
    index_node = match._index_node  # type: ignore[attr-defined]

    def repl(input_tensor, index_node, values):
        dim_size = input_tensor.shape[scatter_dim]
        num_operations = index_node.numel()

        # Flatten multi-dimensional indices to 1-D
        if len(index_node.shape) > 1:
            flat_index = index_node.reshape(num_operations)
            values_ndim = len(index_node.shape)
            flat_values = values.reshape(
                [num_operations] + list(values.shape[values_ndim:])
            )
        else:
            flat_index = index_node
            flat_values = values

        # partition_id = op_id & (num_partitions - 1), requires power-of-2
        operation_ids = torch.ops.prims.iota.default(
            num_operations,
            start=0,
            step=1,
            dtype=flat_index.dtype,
            device=flat_index.device,
            requires_grad=False,
        )
        partition_ids = torch.ops.aten.bitwise_and.Scalar(
            operation_ids, num_partitions - 1
        )

        # Expanded buffer: one copy per partition along scatter_dim
        expanded_shape = list(input_tensor.shape)
        expanded_shape[scatter_dim] *= num_partitions
        expanded_buffer = torch.ops.aten.full.default(
            expanded_shape,
            0,
            dtype=flat_values.dtype,
            layout=torch.strided,
            device=flat_values.device,
            pin_memory=False,
        )

        # Shift each write into its partition's slice
        partition_offsets = partition_ids * dim_size
        adjusted_index = flat_index + partition_offsets

        if isinstance(indices, (list, tuple)):
            adjusted_indices = [
                adjusted_index if i == scatter_dim else idx
                for i, idx in enumerate(indices)
            ]
        else:
            adjusted_indices = [adjusted_index]

        scattered_buffer = torch.ops.aten.index_put.default(
            expanded_buffer, adjusted_indices, flat_values, True
        )

        # Reshape to [..., num_partitions, dim_size, ...] then sum partitions
        reduce_shape = list(expanded_shape)
        reduce_shape[scatter_dim] = num_partitions
        reduce_shape.insert(scatter_dim + 1, dim_size)
        reshaped = torch.ops.aten.view.default(scattered_buffer, reduce_shape)

        # Preserve dtype for integer types that don't promote during sum
        if flat_values.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            reduced = torch.ops.aten.sum.dim_IntList(
                reshaped, [scatter_dim], dtype=flat_values.dtype
            )
        else:
            reduced = torch.ops.aten.sum.dim_IntList(reshaped, [scatter_dim])

        return input_tensor + reduced

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(repl, [input_tensor, index_node, values])

    # Charge this scatter's expanded buffer so later candidates in the same
    # invocation see a correspondingly smaller budget.
    if ctx.memory is not None:
        output_size: int = match._output_size  # type: ignore[attr-defined]
        element_bytes: int = match._element_bytes  # type: ignore[attr-defined]
        ctx.memory.committed_overhead_bytes += (
            output_size * element_bytes * (num_partitions - 1)
        )

    ctx.n_applied += 1
    ctx.applied_partitions.append(num_partitions)
    counters["inductor"]["partitioned_scatter_applied"] += 1


def _build_pattern_pass(ctx: ScatterPassContext) -> PatternMatcherPass:
    """
    Construct a per-invocation PatternMatcherPass whose callbacks close over
    `ctx`, avoiding module-level global state (which is unsafe under concurrent
    compilation).
    """
    patterns = PatternMatcherPass(pass_name="partitioned_scatter_optimization")
    force: bool = config.partitioned_scatter_force

    def extra_check(match: Match) -> bool:
        return _validate_memory(match, ctx, force)

    def replacement(match: Match, input_tensor, indices, values) -> None:
        _create_replacement(match, ctx, input_tensor, indices, values)

    for target in _INDEX_PUT_TARGETS:
        register_graph_pattern(
            CallFunction(target, Arg(), Arg(), Arg(), True),
            extra_check=extra_check,
            pass_dict=patterns,  # type: ignore[arg-type]
        )(replacement)

    return patterns


def _log_summary(ctx: ScatterPassContext, num_matches: int) -> None:
    if ctx.n_candidates == 0:
        return

    if ctx.memory is not None:
        log.info(
            "partitioned_scatter: candidates=%d applied=%d skipped=%d "
            "partitions_per_op=%s "
            "skip_breakdown=%s "
            "total_gpu=%d MB floor=%d MB allowed_peak=%d MB",
            ctx.n_candidates,
            ctx.n_applied,
            ctx.n_candidates - ctx.n_applied,
            ctx.applied_partitions,
            dict(ctx.skip_reasons),
            ctx.memory.total_gpu_bytes // 1_000_000,
            ctx.memory.non_model_floor_bytes // 1_000_000,
            ctx.memory.allowed_peak_bytes // 1_000_000,
        )
    else:
        log.info(
            "partitioned_scatter: candidates=%d applied=%d skipped=%d "
            "partitions_per_op=%s skip_breakdown=%s (no memory state)",
            ctx.n_candidates,
            ctx.n_applied,
            ctx.n_candidates - ctx.n_applied,
            ctx.applied_partitions,
            dict(ctx.skip_reasons),
        )


def partitioned_scatter_optimization_pass(graph: fx.Graph) -> fx.Graph:
    """
    Apply partitioned scatter optimization to high-contention index_put operations.
    Controlled by config.partitioned_scatter_enabled.
    """
    if not config.partitioned_scatter_enabled:
        return graph

    ctx = ScatterPassContext()

    # Stage 1: cheap pre-scan. If nothing qualifies we avoid building the
    # (relatively expensive) memory profile entirely.
    _scan_candidates(graph, ctx)
    if not ctx.candidates:
        _log_summary(ctx, 0)
        return graph

    # Stage 2: build the memory profile and run the pattern matcher.
    ctx.memory = _build_scatter_memory_state(graph)
    patterns = _build_pattern_pass(ctx)
    num_matches = patterns.apply(graph)

    _log_summary(ctx, num_matches)

    if num_matches > 0:
        graph.lint()

    return graph


def _extract_scatter_dim_and_index(
    indices_arg: Any,
) -> tuple[int | None, fx.Node | None]:
    """Extract scatter dimension and index node from indices argument."""
    if not isinstance(indices_arg, (list, tuple)):
        return 0, indices_arg

    index_node = None
    scatter_dim = None

    for dim, idx in enumerate(indices_arg):
        if idx is not None:
            if index_node is not None:
                return None, None
            index_node = idx
            scatter_dim = dim

    return scatter_dim, index_node


def _get_tensor_meta(node: fx.Node) -> dict[str, Any] | None:
    """Extract tensor metadata from an FX node's meta['val'] FakeTensor."""
    if not hasattr(node, "meta") or "val" not in node.meta:
        return None

    val = node.meta["val"]
    if not hasattr(val, "shape") or not hasattr(val, "dtype"):
        return None

    return {
        "shape": tuple(val.shape),
        "dtype": val.dtype,
        "device": val.device,
        "numel": val.numel(),
    }


__all__ = ["partitioned_scatter_optimization_pass"]
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import copy
import itertools
import math
import operator

import torch
from torch._dynamo.utils import counters
from torch.fx.node import map_arg

from .. import config
from ..lowering import lowerings as L, require_channels_last
from ..pattern_matcher import (
    Arg,
    CallFunction,
    filter_nodes,
    KeywordArg,
    ListOf,
    Match,
    stable_topological_sort,
)
from ..utils import pad_listlike
from .freezing_patterns import register_freezing_graph_pattern
from .post_grad import register_lowering_pattern


aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed
quantized = torch.ops.quantized

# Only for per tensor quant since permute may change the channel idx
_PER_TENSOR_QUANTIZE_OPS = [
    quantized_decomposed.quantize_per_tensor.default,
    quantized_decomposed.quantize_per_tensor.tensor,
]

_VIEW_OPS = [
    aten.transpose.int,
    aten.permute.default,
    aten.view.default,
    aten.reshape.default,
]

"""
The quantization.py file primarily incorporates passes related to quantization fusion
in inductor, includes:
1. Lowering of qconv/qlinear (unary and binary post-op variants) from the
   `onednn::qconv_pointwise` / `onednn::qlinear_pointwise` ops to Inductor IR.
2. Pointwise operators' quantization fusion like: qmaxpool2d, qcat, qreshape and more.
3. Weight-only quantization (WOQ) int8 mm / concat-linear patterns lowered to
   `aten._weight_int8pack_mm` and the int4 concat-linear pass for
   `aten._weight_int4pack_mm_for_cpu`.

Note: the dequant promotion, qconv/qlinear weight prepack and qconv/qlinear
fusion patterns that used to live here have been moved to TorchAO.
"""


def _get_pattern_output_dtype(match: Match):
    """
    Get the pattern's output dtype from node's meta
    Assume only 1 output node in this matched pattern.
    """
    pattern_output_nodes = match.output_nodes()
    if len(pattern_output_nodes) != 1:
        raise AssertionError(
            f"expected exactly 1 output node, got {len(pattern_output_nodes)}"
        )
    output_node = pattern_output_nodes[0]
    if not isinstance(output_node, torch.fx.Node):
        raise AssertionError(f"expected torch.fx.Node, got {type(output_node)}")
    output_dtype = output_node.meta["val"].dtype
    if output_dtype not in [
        torch.int8,
        torch.uint8,
        torch.float32,
        torch.bfloat16,
        torch.float8_e4m3fn,
    ]:
        raise AssertionError(f"unexpected output dtype {output_dtype}")
    return output_dtype


def _may_generate_pattern_with_dtype_convert(
    pattern, dtype=Arg(), with_dtype_convert=True, users=1
):
    if with_dtype_convert:
        return CallFunction(
            prims.convert_element_type.default,
            pattern,
            dtype,
            _users=users,
        )
    else:
        return pattern


def _may_generate_pattern_with_reshape(pattern, reshape_size=Arg(), with_reshape=True):
    if with_reshape:
        return CallFunction(
            torch.ops.aten.reshape.default,
            pattern,
            reshape_size,
        )
    else:
        return pattern


def _generate_linear_t_pattern(
    _dequant_per_channel_pattern,
    dtype,
):
    if dtype not in [torch.float32, torch.bfloat16]:
        raise AssertionError(f"expected float32 or bfloat16, got {dtype}")
    t_pattern = CallFunction(
        aten.permute.default,
        _may_generate_pattern_with_dtype_convert(
            _dequant_per_channel_pattern,
            KeywordArg("autocast_wgt_dtype"),
            dtype == torch.bfloat16,
        ),
        KeywordArg("permute_axes"),
    )
    return t_pattern


def _unary_fusion_pattern(unary_fusion, call_fn, users, is_bf16):
    # only insert to_dtype if is_bf16 is True
    computation_call = _may_generate_pattern_with_dtype_convert(
        call_fn, dtype=KeywordArg("to_float"), with_dtype_convert=is_bf16, users=users
    )
    return unary_fusion(computation_call)


def get_dequantize_per_tensor_activation_pattern(is_tensor_overload=False):
    dequantize_per_tensor_activation_pattern = CallFunction(
        quantized_decomposed.dequantize_per_tensor.tensor
        if is_tensor_overload
        else quantized_decomposed.dequantize_per_tensor.default,
        KeywordArg("x"),
        KeywordArg("x_scale"),
        KeywordArg("x_zp"),
        KeywordArg("x_quant_min"),
        KeywordArg("x_quant_max"),
        KeywordArg("x_dq_dtype"),
    )
    return dequantize_per_tensor_activation_pattern


dequantize_per_channel_weight_pattern = CallFunction(
    quantized_decomposed.dequantize_per_channel.default,
    KeywordArg("q_weight"),
    KeywordArg("w_scale"),
    KeywordArg("w_zp"),
    KeywordArg("w_axis"),
    KeywordArg("w_quant_min"),
    KeywordArg("w_quant_max"),
    KeywordArg("w_dtype"),
)

dequantize_per_channel_to_bf16_weight_pattern = (
    _may_generate_pattern_with_dtype_convert(
        dequantize_per_channel_weight_pattern,
        KeywordArg("autocast_wgt_dtype"),
    )
)

dequantize_per_channel_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_channel_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)

dequantize_per_channel_to_bf16_clone_weight_pattern = CallFunction(
    aten.clone.default,
    dequantize_per_channel_to_bf16_weight_pattern,
    memory_format=KeywordArg("memory_format"),
)


def get_qconv_pt2e_pattern(x_scale_zp_are_tensors=False, users=1):
    qconv_op = (
        torch.ops.onednn.qconv_pointwise.tensor
        if x_scale_zp_are_tensors
        else torch.ops.onednn.qconv_pointwise.default
    )
    return CallFunction(
        qconv_op,
        KeywordArg("x"),
