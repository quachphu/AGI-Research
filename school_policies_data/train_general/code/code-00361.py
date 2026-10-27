    register_graph_pattern,
)
from ..virtualized import V


log = logging.getLogger(__name__)
patterns = PatternMatcherPass(subsystem="joint_graph_passes")
aten = torch.ops.aten


def _shape_to_offset(shape, device: torch.device):
    # Modified from torch/_prims/rng_prims.py:philox_rand_offset
    nelem = 1
    for s in shape:
        nelem *= s

    # Empty tensor: no random numbers are generated/consumed.
    is_empty = nelem == 0
    if statically_known_true(is_empty) or guard_or_false(is_empty):
        return 0

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if device.type != "cuda":
        return 0

    block_size = 256
    unroll = 4
    curand4_engine_calls = 4

    device_property = torch.cuda.get_device_properties(device)

    blocks_per_sm = device_property.max_threads_per_multi_processor // block_size
    max_grid = device_property.multi_processor_count * blocks_per_sm
    grid_size = (nelem + block_size - 1) // block_size
    grid_size = -torch.sym_min(-grid_size, -1)
    grid_size = torch.sym_min(grid_size, max_grid)

    return ((nelem - 1) // (block_size * grid_size * unroll) + 1) * curand4_engine_calls


def replace_random_passes(gm: torch.fx.GraphModule):
    """Modify the given FX graph to use backend-native random ops"""
    if config.fallback_random:
        return 0

    count = patterns.apply(gm)
    with GraphTransformObserver(gm, "fuse_seed_creation_pass", "joint_graph_passes"):
        count += fuse_seed_creation_pass(gm.graph)
    if config.align_random_eager:
        with GraphTransformObserver(gm, "fuse_offset_creation_pass"):
            count += fuse_offset_creation_pass(gm.graph)

    return count


def fuse_offset_creation_pass(graph: torch.fx.Graph) -> int:
    """
    Here offset node means seed << 32 + offset, will unpacked in lowering.py:inductor_random()
    Horizontally fuse all the seed generation on each device
        a = inductor_prims.rand_eager_offset(offset, dev)
        b = inductor_prims.rand_eager_offset(offset, dev)
    Becomes:
        offsets = inductor_prims.rand_eager_offsets([offset1, offset2...], dev)
        a = torch.ops.aten.select.int(offsets, 0, 0)
        b = torch.ops.aten.select.int(offsets, 0, 1)
    We do this because seed creation is entirely launch overhead bound.
    """
    device_offsets = collections.defaultdict(list)
    for node in graph.nodes:
        if CallFunctionVarArgs(inductor_prims.rand_eager_offset).match(node):
            device_offsets[node.args[1]].append(node)

    if not device_offsets:
        return 0

    for device, offsets in device_offsets.items():
        with graph.inserting_before(offsets[0]):
            offs = [n.args[0] for n in offsets]
            combined = graph.call_function(
                inductor_prims.rand_eager_offsets, (offs, device)
            )
            combined.meta.update(offsets[0].meta)
            with V.fake_mode:
                combined.meta["val"] = torch.empty(
                    [len(offsets), 2], device=device, dtype=torch.int64
                )
                combined.meta["tensor_meta"] = _extract_tensor_metadata(
                    combined.meta["val"]
                )

        for idx, offset in enumerate(offsets):
            with graph.inserting_before(offset):
                new_state = graph.call_function(
                    torch.ops.aten.select.int, (combined, 0, idx)
                )
            offset.replace_all_uses_with(new_state)
            new_state.meta.update(offset.meta)
            graph.erase_node(offset)

    return len(device_offsets)


def fuse_seed_creation_pass(graph: torch.fx.Graph):
    """
    Horizontally fuse all the seed generation on each device

        a = inductor_seed(dev)
        b = inductor_seed(dev)

    Becomes:
        seeds = inductor_seeds(2, dev)
        a = inductor_lookup_seed(seeds, 0)
        b = inductor_lookup_seed(seeds, 1)

    We do this because seed creation is entirely launch overhead bound.
    """
    device_seeds = collections.defaultdict(list)
    for node in graph.nodes:
        if CallFunctionVarArgs(inductor_prims.seed).match(node):
            device_seeds[node.args[0]].append(node)

    if not device_seeds:
        return 0

    for device, seeds in device_seeds.items():
        with graph.inserting_before(seeds[0]):
            combined = graph.call_function(inductor_prims.seeds, (len(seeds), device))
            combined.meta.update(seeds[0].meta)
            with V.fake_mode:
                combined.meta["val"] = torch.empty(
                    [len(seeds)], device=device, dtype=torch.int64
                )
                combined.meta["tensor_meta"] = _extract_tensor_metadata(
                    combined.meta["val"]
                )

        for idx, seed in enumerate(seeds):
            with graph.inserting_before(seed):
                new_seed = graph.call_function(
                    inductor_prims.lookup_seed, (combined, idx)
                )
            seed.replace_all_uses_with(new_seed)
            new_seed.meta.update(seed.meta)
            graph.erase_node(seed)

    return len(device_seeds)


def default_kwargs(device):
    return {}


def get_device(device):
    if device is not None:
        return device
    return torch.empty([]).device  # default device


# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.rand.default), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.rand.generator), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randn.default), pass_dict=patterns)
# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randn.generator), pass_dict=patterns)
def replace_random(
    match: Match,
    size,
    *,
    generator=None,
    dtype=None,
    device=None,
    layout=None,
    pin_memory=None,
):
    if generator is not None:
        return
    if pin_memory:
        return

    def replacement(size):
        result = inductor_prims.random(
            size, inductor_prims.seed(device), mode, **default_kwargs(device)
        )
        if dtype is not None:
            result = result.to(dtype)
        return result

    mode = {
        aten.rand: "rand",
        aten.randn: "randn",
    }[
        match.output_node().target.overloadpacket  # type: ignore[union-attr]
    ]  # type: ignore[union-attr]
    device = get_device(device)
    replacement_fn = replacement

    if mode == "rand" and config.align_random_eager and device.type == "cuda":
        # Only enable when align_random_eager is on.
        def replacement_align(size):
            offset = _shape_to_offset(size, device)

            align_dtype = dtype
            if isinstance(align_dtype, (tuple, list)):
                align_dtype = align_dtype[0] if len(align_dtype) else None

            result = inductor_prims.random(
                size,
                inductor_prims.rand_eager_offset(offset, device),
                mode,
                **default_kwargs(device),
                align_dtype=align_dtype,
            )
            if dtype is not None:
                result = result.to(dtype)
            return result

        replacement_fn = replacement_align

    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement_fn, [size])


# pyrefly: ignore [bad-argument-type]
@register_graph_pattern(CallFunctionVarArgs(aten.randint.low), pass_dict=patterns)
def replace_randint(
    match: Match,
    low,
    high,
    size,
    *,
    dtype=torch.int64,
    device=None,
    layout=None,
    pin_memory=None,
):
    if pin_memory:
        return

    def replacement(low, high, size):
        result = inductor_prims.randint(low, high, size, inductor_prims.seed(device))
        return result.to(dtype)

    device = get_device(device)
    # pyrefly: ignore [bad-argument-type]
    match.replace_by_example(replacement, [low, high, size])
<|endoftext|>import functools
import logging
import math
from collections.abc import Callable, Sequence
from enum import Enum
from queue import Queue
from typing import Any, TypeAlias

import torch
from torch.fx import Graph, Node

from .common import CantChunk
from .core import (
    ChunkingMeta,
    copy_chunking_meta,
    get_chunking_meta,
    has_nop_chunking_meta,
    set_chunking_meta,
    set_chunking_meta_if_none,
    update_chunking_meta,
)
from .propagate_scale_by import propagate_scale_by
from .utils import (
    format_node_with_chunking_meta,
    get_args_of_node_type,
    get_fake_tensor_from_node_arg,
    get_first_chunking_meta,
    get_node_is_scalar,
    get_node_ndim,
    get_nodes_with_chunking_meta,
    has_any_chunking_meta,
    is_chunked_by_dim,
    is_tangent_node,
)


log = torch._logging.getArtifactLogger(__name__, "auto_chunker")
aten = torch.ops.aten
prims = torch.ops.prims


"""
NOTE [Why we need both fwd and bwd chunking metadata propagation?]
The starting point of chunking is we found an op that creates much larger outputs
than the inputs size. We attach chunking metadata upon the op and propagate it forward.

But for backward rules like NLLLossBackward, we do a scatter upon a zero matrix. That
zero matrix is created by torch.full. We will only know we should chunk that tensor
by propagating chunking metadata backward.
"""

"""
NOTE [Why we need a separate pass to propagate ChunkingMeta.scale_by?]

ChunkingMeta.scale_by only need to be propagated forward from the tangent placeholder nodes.
If we do this together with propagating other metadata, we can not fully control the propagating
order and end up with cases like:
    out = aten.sub(lhs, rhs)
where `lhs` has scale_by set, while `rhs` and `out` don't.
For sub op, we could propagate `scale_by` to `rhs` and `out` since that's the
only way to make sense. But overall this is unsafe since maybe this is a case
that chunking does not work and we should bail out.
Another example is, we can not really propagate `scale_by` backward for
aten.mul since we don't know which of the input should have this `scale_by` metadata.

But it's safer that we only propagate `scale_by` metadata in the topological order.

Have the `scale_by` handled in a separate pass also makes the fwd/bwd
chunking metadata propagation much simpler. We don't need special rules
for mul/div/where etc due to the special handling of scale_by:
https://gist.github.com/shunting314/324e57881f168009784991300acab852
"""


class PropagateStatus(Enum):
    SUCCEED_NO_CHANGE = 0
    SUCCEED_WITH_CHANGE = 1
    FAIL = 2


_HandlerRetType: TypeAlias = PropagateStatus | tuple[PropagateStatus, PropagateStatus]

_HandlerType: TypeAlias = Callable[[Node], _HandlerRetType]

# Rules to propagate chunking metadata from inputs to the current node
# or from the current node back to its inputs
propagate_rules: dict[torch._ops.OpOverload, _HandlerType] = {}


def _register_propagate_rule(
    aten_op: torch._ops.OpOverload | Sequence[torch._ops.OpOverload],
    handler: _HandlerType,
) -> _HandlerType:
    if not isinstance(aten_op, (list, tuple)):
        aten_op = [aten_op]  # type: ignore[assignment, list-item]

    @functools.wraps(handler)
    def wrapper(node: Node, *args: Any, **kwargs: Any) -> PropagateStatus:
        fwd_bwd_status = handler(node, *args, **kwargs)
        if isinstance(fwd_bwd_status, PropagateStatus):
            return fwd_bwd_status
        if not (isinstance(fwd_bwd_status, (list, tuple)) and len(fwd_bwd_status) == 2):
            raise AssertionError(
                f"expected (list, tuple) of len 2, got {type(fwd_bwd_status)}"
            )
        fwd_status, bwd_status = fwd_bwd_status
        log.debug(
            "Chunking metadata propagation for %s: Fwd status %s, bwd status %s",
            node,
            fwd_status,
            bwd_status,
        )
        if fwd_status == PropagateStatus.FAIL or bwd_status == PropagateStatus.FAIL:
            return PropagateStatus.FAIL
        if (
            fwd_status == PropagateStatus.SUCCEED_WITH_CHANGE
            or bwd_status == PropagateStatus.SUCCEED_WITH_CHANGE
        ):
            return PropagateStatus.SUCCEED_WITH_CHANGE
        return PropagateStatus.SUCCEED_NO_CHANGE

    if not isinstance(aten_op, (list, tuple)):
        raise AssertionError(f"{type(aten_op)=}")
    for op in aten_op:
        if not isinstance(op, torch._ops.OpOverload):
            raise AssertionError(f"expected OpOverload, got {type(op)}")
        propagate_rules[op] = wrapper
    return wrapper


def register_propagate_rule(
    aten_op: torch._ops.OpOverload | Sequence[torch._ops.OpOverload],
) -> Callable[[_HandlerType], _HandlerType]:
    return functools.partial(_register_propagate_rule, aten_op)


def _is_success(*statuslist: PropagateStatus) -> bool:
    return not any(status == PropagateStatus.FAIL for status in statuslist)


def _enqueue(queue: Queue, item: Node) -> None:  # type: ignore[type-arg]
    """
    Have a function to make it easier to do debug logging in a central place
    """
    queue.put(item)
    log.debug("Enqueue: %s", item)


def can_reach_amplified_node(
    graph: Graph, amplifier_node: Node, is_fwd: bool
) -> dict[Node, bool]:
    """
    A amplified node means a node with the same numel as `amplifier_node`
    """
    filter_obj: dict[Node, bool] = {}
    nodelist = reversed(graph.nodes) if is_fwd else graph.nodes
    target_numel = get_fake_tensor_from_node_arg(amplifier_node).numel()  # type: ignore[union-attr]

    for node in nodelist:
        reach = False
        if node.op == "output":
            # output node does not have a meta['val']
            reach = False

        elif get_fake_tensor_from_node_arg(node) is None:
            reach = False

        # for the back propagation, we should continue propagate if we can
        # reach a tangent node
        elif get_fake_tensor_from_node_arg(node).numel() == target_numel or (  # type: ignore[union-attr]
            not is_fwd and is_tangent_node(node)
        ):
            reach = True
        else:
            neighbors = node.users if is_fwd else get_args_of_node_type(node)
            reach = any(filter_obj[neighbor] for neighbor in neighbors)
        filter_obj[node] = reach
    return filter_obj


def propagate(amplifier_node: Node) -> None:
    log.debug("amplifier_node is %s", amplifier_node.format_node())
    # Chunk the batch dimension (dim 0) of the amplifier_node
    graph = amplifier_node.graph

    fwd_filter = can_reach_amplified_node(graph, amplifier_node, True)
    bwd_filter = can_reach_amplified_node(graph, amplifier_node, False)

    log.debug("fwd_filter %s", fwd_filter)
    log.debug("bwd_filter %s", bwd_filter)

    if len(get_nodes_with_chunking_meta(graph)) != 0:
        raise AssertionError("Expect no nodes with chunking meta yet")

    set_chunking_meta(amplifier_node, chunk_dim=0)

    queue: Queue[Node] = Queue()
    _enqueue(queue, amplifier_node)

    while not queue.empty():
        propagate_single_node(queue, fwd_filter, bwd_filter, queue.get())

    nodes_with_chunking_meta = get_nodes_with_chunking_meta(graph)
    propagate_scale_by(nodes_with_chunking_meta)

    if log.isEnabledFor(logging.DEBUG):
        print("All nodes with chunking metadata set:")
        for node in nodes_with_chunking_meta:
            format_node_with_chunking_meta(node)


def propagate_single_node(
    queue: Queue, fwd_filter: dict[Node, bool], bwd_filter: dict[Node, bool], node: Node
) -> None:  # type: ignore[type-arg]
    log.debug("Propagate_single_node: %s", node.format_node())

    if node.op != "call_function":
        raise CantChunk(
            "Chunker can only propagate chunking metadata thru call_function nodes"
        )

    target = node.target
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Before propagation, the node has the following chunking meta:")
        format_node_with_chunking_meta(node, True)

    if not isinstance(target, torch._ops.OpOverload) or target not in propagate_rules:
        raise CantChunk(
            f"Missing propagation rule for target {target}: {node.format_node()}"
        )

    status = propagate_rules[target](node)

    if log.isEnabledFor(logging.DEBUG):
        log.debug("After propagation, the node has the following chunking meta:")
        format_node_with_chunking_meta(node, True)

    if status == PropagateStatus.FAIL:
        raise CantChunk(f"Propagate rule for {target} fail: {node.format_node()}")
    elif status == PropagateStatus.SUCCEED_WITH_CHANGE:
        # propagate to used nodes
        for arg in get_args_of_node_type(node):
            # don't propagate back thru a placeholder node
            if arg.op == "placeholder":
                if "tangent" in arg.target:  # type: ignore[operator]
                    # we have a separate pass to propagate scale_by information fwd.
                    set_chunking_meta(arg, scale_by=arg)
            elif bwd_filter[arg]:
                _enqueue(queue, arg)

        # propagate to user nodes
        if fwd_filter[node]:
            for user in node.users:
                _enqueue(queue, user)
    else:
        if status != PropagateStatus.SUCCEED_NO_CHANGE:
            raise AssertionError(f"status type {type(status)}, value {status}")


def _bool_to_status(changed: bool) -> PropagateStatus:
    """
    Return the variant of the success flag depending on whether there is any change.
    """
    return (
        PropagateStatus.SUCCEED_WITH_CHANGE
        if changed
        else PropagateStatus.SUCCEED_NO_CHANGE
    )


@register_propagate_rule(aten.addmm.default)
def propagate_addmm(addmm_node: Node) -> _HandlerRetType:
    bias_node, input_node, weight_node = addmm_node.args

    def propagate_addmm_fwd() -> PropagateStatus:
        if not isinstance(bias_node, Node):
            raise AssertionError(
                f"expected bias_node to be a Node, got {type(bias_node)}"
            )
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        if not isinstance(weight_node, Node):
            raise AssertionError(
                f"expected weight_node to be a Node, got {type(weight_node)}"
            )
        if not has_any_chunking_meta(bias_node, input_node, weight_node):
            return PropagateStatus.SUCCEED_NO_CHANGE

        # only input is chunked by dim 0
        if (
            has_nop_chunking_meta(bias_node)
            and has_nop_chunking_meta(weight_node)
            and is_chunked_by_dim(input_node, 0)
        ):
            # set a nop chunking metadata on bias_node & weight_node
            # to indicate that they should be a part of the chunking
            # subgraph. (i.e. we pass in them as placeholder node)
            return _bool_to_status(
                copy_chunking_meta(addmm_node, input_node)
                | set_chunking_meta(bias_node)
                | set_chunking_meta(weight_node)
            )
        return PropagateStatus.FAIL

    def propagate_addmm_bwd() -> PropagateStatus:
        if not isinstance(bias_node, Node):
            raise AssertionError(
                f"expected bias_node to be a Node, got {type(bias_node)}"
            )
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        if not isinstance(weight_node, Node):
            raise AssertionError(
                f"expected weight_node to be a Node, got {type(weight_node)}"
            )

        if not (meta := get_chunking_meta(addmm_node)):
            return PropagateStatus.SUCCEED_NO_CHANGE

        if meta.chunked_by_dim(0):
            # if the output is chunked by the batch dimension, then
            # bias and input should as well
            changed = set_chunking_meta(input_node, meta) | set_chunking_meta(
                weight_node
            )

            # We should chunk the bias only if it's not broadcasted
            bias_node_ft = get_fake_tensor_from_node_arg(bias_node)
            input_node_ft = get_fake_tensor_from_node_arg(input_node)
            if bias_node_ft is None:
                raise AssertionError("expected bias_node_ft to not be None")
            if input_node_ft is None:
                raise AssertionError("expected input_node_ft to not be None")
            if bias_node_ft.ndim < input_node_ft.ndim:
                changed |= set_chunking_meta(bias_node)
            else:
                changed |= set_chunking_meta(bias_node, meta)
            return _bool_to_status(changed)

        return PropagateStatus.FAIL

    return propagate_addmm_fwd(), propagate_addmm_bwd()


@register_propagate_rule(aten.mm.default)
def propagate_mm(mm_node: Node) -> _HandlerRetType:
    lhs_node, rhs_node = mm_node.args[:2]

    def fwd() -> PropagateStatus:
        if not isinstance(lhs_node, Node):
            raise AssertionError(
                f"expected lhs_node to be a Node, got {type(lhs_node)}"
            )
        if not isinstance(rhs_node, Node):
            raise AssertionError(
                f"expected rhs_node to be a Node, got {type(rhs_node)}"
            )
        lhs_meta = get_chunking_meta(lhs_node)
        rhs_meta = get_chunking_meta(rhs_node)

        if has_nop_chunking_meta(lhs_node) and has_nop_chunking_meta(rhs_node):
            return _bool_to_status(False)

        # only lhs is chunked
        if not has_nop_chunking_meta(lhs_node) and has_nop_chunking_meta(rhs_node):
            if lhs_meta is None:
                raise AssertionError("expected lhs_meta to not be None")
            return _bool_to_status(
                copy_chunking_meta(mm_node, lhs_meta) | set_chunking_meta(rhs_node)
            )

        # either lhs or rhs is chunked at the reduction dimension
        if (lhs_meta is not None and lhs_meta.chunk_dim == 1) or (
            rhs_meta is not None and rhs_meta.chunk_dim == 0
        ):
            # The output is not chunked, but need to be sum'ed up!
            return _bool_to_status(
                set_chunking_meta(mm_node, chunk_dim=None, need_sum=True)
                | update_chunking_meta(lhs_node, chunk_dim=1)
                | update_chunking_meta(rhs_node, chunk_dim=0)
            )

        return PropagateStatus.FAIL

    def bwd() -> PropagateStatus:
        if not isinstance(lhs_node, Node):
            raise AssertionError(
                f"expected lhs_node to be a Node, got {type(lhs_node)}"
            )
        if not isinstance(rhs_node, Node):
            raise AssertionError(
                f"expected rhs_node to be a Node, got {type(rhs_node)}"
            )
        out_meta = get_chunking_meta(mm_node)
        if out_meta is None:
            return _bool_to_status(False)

        # first dim of a 2D output is chunked
        ft = get_fake_tensor_from_node_arg(mm_node)
        if ft is None:
            raise AssertionError("expected ft to not be None")
        if ft.ndim == 2 and out_meta.chunk_dim == 0:
            rhs_meta = get_chunking_meta(rhs_node)
            if not ChunkingMeta.is_nop(rhs_meta):
                raise AssertionError("expected rhs_meta to be a nop ChunkingMeta")
            return _bool_to_status(
                copy_chunking_meta(lhs_node, mm_node) | set_chunking_meta(rhs_node)
            )

        if out_meta.need_sum:
            changed = set_chunking_meta(lhs_node, chunk_dim=1) | set_chunking_meta(
                rhs_node, chunk_dim=0
            )
            return _bool_to_status(changed)

        return PropagateStatus.FAIL

    return fwd(), bwd()


@register_propagate_rule(
    [
        prims.convert_element_type.default,
        aten.exp.default,
        aten.log.default,
        aten.tanh.default,
        aten.add.Tensor,
        aten.sub.Tensor,
        aten.div.Tensor,
        aten.mul.Tensor,
        prims.fma.default,
        aten.where.self,
        aten.neg.default,
        aten.eq.Tensor,
    ]
)
def propagate_general_copy_metadata(
    out_node: Node, ignore_broadcast: bool = False
) -> _HandlerRetType:
    """
    A general propagation rules that basically copy around the chunking
    metadata.
    """
    node_args = get_args_of_node_type(out_node)
    node_is_scalar = get_node_is_scalar(node_args)
    node_ndim = get_node_ndim(node_args)

    scalar_args = [node for node in node_args if node_is_scalar[node]]
    non_scalar_args = [node for node in node_args if not node_is_scalar[node]]

    out_ndim = out_node.meta["val"].ndim

    # This general rule only allow scalar tensors without chunking meta
    if scalar_args and not all(
        ChunkingMeta.is_nop(get_chunking_meta(arg)) for arg in scalar_args
    ):
        return PropagateStatus.FAIL

    def _chunk_broadcasted_tensor(chunk_dim: int) -> bool:
        for node in non_scalar_args:
            if node_ndim[node] != out_ndim and chunk_dim >= out_ndim - node_ndim[node]:
                return True
        return False

    def propagate_fwd() -> PropagateStatus:
        if len(node_args) == 0:
            return PropagateStatus.FAIL

        first_meta = get_first_chunking_meta(*non_scalar_args)
        if first_meta is None:
            return _bool_to_status(False)

        need_handle_broadcast = (
            not ignore_broadcast and first_meta.chunk_dim is not None
        )
        if (
            need_handle_broadcast
            and first_meta.chunk_dim is not None
            and _chunk_broadcasted_tensor(first_meta.chunk_dim)
        ):
            # We don't chunk a broadcasted tensor for now.
            # Can add the rule if such a use case comes up
            return PropagateStatus.FAIL

        changed = set_chunking_meta_if_none(
            non_scalar_args, first_meta, lambda node: node_ndim[node] != out_ndim
        )

        for other_node in non_scalar_args:
            other_meta = get_chunking_meta(other_node)

            if need_handle_broadcast and node_ndim[other_node] != out_ndim:
                if not ChunkingMeta.is_nop(other_meta):
                    return PropagateStatus.FAIL
            else:
                if other_meta != first_meta:
                    return PropagateStatus.FAIL

        changed |= copy_chunking_meta(out_node, first_meta)
        return _bool_to_status(changed)

    def propagate_bwd() -> PropagateStatus:
        if not (meta := get_chunking_meta(out_node)):
            return PropagateStatus.SUCCEED_NO_CHANGE

        need_handle_broadcast = not ignore_broadcast and meta.chunk_dim is not None
        if (
            need_handle_broadcast
            and meta.chunk_dim is not None
            and _chunk_broadcasted_tensor(meta.chunk_dim)
        ):
            return PropagateStatus.FAIL

        # apply any to a list to avoid short-circuit
        changed = any(
            [  # noqa: C419
                copy_chunking_meta(node, meta)
                if not need_handle_broadcast or node_ndim[node] == out_ndim
                else set_chunking_meta(node)
                for node in non_scalar_args
            ]
        )

        # [NOTE: NOP Chunking metadata]
        # For scalar node arguments, we add a nop ChunkingMeta so the
        # propagation continues. This is mainly needed to reach the point
        # where we attach chunking metadata to tangents that need to be
        # included in the chunking subgraph.
        # This is different to having a None ChunkingMeta
        changed |= any(
            [  # noqa: C419
                set_chunking_meta(node)
                for node in scalar_args
                if get_chunking_meta(node) is None
            ]
        )

        return _bool_to_status(changed)

    return propagate_fwd(), propagate_bwd()


@register_propagate_rule(
    [
        aten.squeeze.dim,
        aten.gather.default,
        aten.scatter.value,
        aten.scatter_add.default,
    ]
)
def propagate_general_copy_metadata_ignore_broadcast(out_node: Node) -> _HandlerRetType:
    return propagate_general_copy_metadata(out_node, ignore_broadcast=True)  # type: ignore[call-arg]


@register_propagate_rule(
    [
        aten.amax.default,
        aten.sum.dim_IntList,
    ]
)
def propagate_reduce(reduce_node: Node) -> _HandlerRetType:
    arg_node, reduce_dims = reduce_node.args[0:2]

    def propagate_fwd() -> PropagateStatus:
        if not isinstance(arg_node, Node):
            raise AssertionError(
                f"expected arg_node to be a Node, got {type(arg_node)}"
            )
        if not isinstance(reduce_dims, (tuple, list)):
            raise AssertionError(
                f"expected reduce_dims to be a tuple or list, got {type(reduce_dims)}"
            )
        arg_meta = get_chunking_meta(arg_node)
        if arg_meta is None:
            return PropagateStatus.SUCCEED_NO_CHANGE
        if arg_meta.chunk_dim not in reduce_dims:
            # Reduce across the non chunked dimension
            return _bool_to_status(copy_chunking_meta(reduce_node, arg_meta))

        # sum across the chunked dimension. E.g. happens for computing
        # the gradient of bias for an addmm
        if reduce_node.target == aten.sum.dim_IntList and list(reduce_dims) == [
            arg_meta.chunk_dim
        ]:
            return _bool_to_status(
                set_chunking_meta(reduce_node, arg_meta, chunk_dim=None, need_sum=True)
            )

        return PropagateStatus.FAIL

    def propagate_bwd() -> PropagateStatus:
        if not isinstance(arg_node, Node):
            raise AssertionError(
                f"expected arg_node to be a Node, got {type(arg_node)}"
            )
        if not isinstance(reduce_dims, (tuple, list)):
            raise AssertionError(
                f"expected reduce_dims to be a tuple or list, got {type(reduce_dims)}"
            )
        out_meta = get_chunking_meta(reduce_node)
        if out_meta is None:
            return PropagateStatus.SUCCEED_NO_CHANGE
        if out_meta.chunk_dim is not None:
            if out_meta.chunk_dim in reduce_dims:
                raise AssertionError(
                    f"expected chunk_dim {out_meta.chunk_dim} not in {reduce_dims}"
                )
            return _bool_to_status(copy_chunking_meta(arg_node, out_meta))

        if out_meta.chunk_dim is None and out_meta.need_sum and len(reduce_dims) == 1:
            if reduce_node.target != aten.sum.dim_IntList:
                raise AssertionError(
                    f"expected target aten.sum.dim_IntList, got {reduce_node.target}"
                )
            return _bool_to_status(
                set_chunking_meta(
                    arg_node, out_meta, chunk_dim=reduce_dims[0], need_sum=False
                )
            )

        return PropagateStatus.FAIL

    return propagate_fwd(), propagate_bwd()


@register_propagate_rule(aten.permute.default)
def propagate_permute(permute_node: Node) -> _HandlerRetType:
    input_node, order = permute_node.args[:2]

    def propagate_fwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        if not isinstance(order, (tuple, list)):
            raise AssertionError(
                f"expected order to be a tuple or list, got {type(order)}"
            )
        input_meta = get_chunking_meta(input_node)
        output_meta = get_chunking_meta(permute_node)
        if input_meta is None:
            return _bool_to_status(False)

        if input_meta.chunk_dim is None:
            return PropagateStatus.FAIL

        orig_chunk_dim = input_meta.chunk_dim
        # pyrefly: ignore [bad-argument-type, bad-assignment]
        reverse_lookup: dict[int, int] = {v: k for k, v in enumerate(order)}
        new_chunk_dim = reverse_lookup[orig_chunk_dim]

        # sanity check
        if output_meta is not None:
            if output_meta.chunk_dim != new_chunk_dim:
                raise AssertionError(
                    f"expected chunk_dim {new_chunk_dim}, got {output_meta.chunk_dim}"
                )
        return _bool_to_status(
            set_chunking_meta(permute_node, meta=input_meta, chunk_dim=new_chunk_dim)
        )

    def propagate_bwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        if not isinstance(order, (tuple, list)):
            raise AssertionError(
                f"expected order to be a tuple or list, got {type(order)}"
            )

        input_meta = get_chunking_meta(input_node)
        output_meta = get_chunking_meta(permute_node)
        if output_meta is None:
            return _bool_to_status(False)

        if output_meta.chunk_dim is None:
            return PropagateStatus.FAIL

        orig_chunk_dim = output_meta.chunk_dim

        new_chunk_dim = order[orig_chunk_dim]

        # sanity check
        if input_meta is not None:
            if input_meta.chunk_dim != new_chunk_dim:
                raise AssertionError(
                    f"expected chunk_dim {new_chunk_dim}, got {input_meta.chunk_dim}"
                )
        return _bool_to_status(
            set_chunking_meta(input_node, meta=output_meta, chunk_dim=new_chunk_dim)
        )

    return propagate_fwd(), propagate_bwd()


@register_propagate_rule(
    [
        aten.full.default,  # nop since there is no inputs for fwd/bwd metadata propagation
    ]
)
def propagate_nop(node: Node) -> _HandlerRetType:
    def fwd() -> PropagateStatus:
        return _bool_to_status(False)

    def bwd() -> PropagateStatus:
        return _bool_to_status(False)

    return fwd(), bwd()


@register_propagate_rule(aten.unsqueeze.default)
def propagate_unsqueeze(unsqueeze_node: Node) -> _HandlerRetType:
    input_node, unsqueeze_dim = unsqueeze_node.args[:2]
    if not isinstance(input_node, Node):
        raise AssertionError(
            f"expected input_node to be a Node, got {type(input_node)}"
        )
    if not isinstance(unsqueeze_dim, int):
        raise AssertionError(
            f"expected unsqueeze_dim to be an int, got {type(unsqueeze_dim)}"
        )
    input_ndim = get_fake_tensor_from_node_arg(input_node).ndim  # type: ignore[union-attr]
    # Normalize negative dim: unsqueeze valid range is [-(ndim+1), ndim]
    normalized_dim = (
        unsqueeze_dim + input_ndim + 1 if unsqueeze_dim < 0 else unsqueeze_dim
    )

    def fwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        input_meta = get_chunking_meta(input_node)
        if input_meta is None:
            return _bool_to_status(False)
        if input_meta.chunk_dim is None:
            return _bool_to_status(copy_chunking_meta(unsqueeze_node, input_meta))

        # pyrefly: ignore[unsupported-operation]
        new_dim = input_meta.chunk_dim + (
            1 if input_meta.chunk_dim >= normalized_dim else 0
        )
        return _bool_to_status(
            set_chunking_meta(unsqueeze_node, meta=input_meta, chunk_dim=new_dim)
        )

    def bwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        output_meta = get_chunking_meta(unsqueeze_node)
        if output_meta is None:
            return _bool_to_status(False)
        if output_meta.chunk_dim is None:
            return _bool_to_status(copy_chunking_meta(input_node, output_meta))
        # pyrefly: ignore[unsupported-operation]
        new_dim = output_meta.chunk_dim - (
            1 if output_meta.chunk_dim > normalized_dim else 0
        )
        return _bool_to_status(
            set_chunking_meta(input_node, meta=output_meta, chunk_dim=new_dim)
        )

    return fwd(), bwd()


def _find_chunk_dim_after_reshape(
    old_shape: Sequence[int], new_shape: Sequence[int], chunk_dim: int
) -> int | None:
    """
    Find the equivalent chunk_dim position after a reshape by matching
    the prefix product (number of elements before the dimension) and
    the dimension size. Returns None if the chunk dimension is merged
    or split by the reshape, making it unsafe to propagate.

    Examples:
      [M, N] -> [M, N, 1], chunk_dim=0: returns 0 (trailing dim added)
      [M]    -> [M, 1],     chunk_dim=0: returns 0
      [M, N] -> [M1, M2, N] where M1*M2=M, chunk_dim=0: returns None (split)
      [M, N] -> [M*N],      chunk_dim=0: returns None (merged)
    """
    chunk_size = old_shape[chunk_dim]
    old_offset = math.prod(old_shape[:chunk_dim])
    new_offset = 1
    for new_dim in range(len(new_shape)):
        if new_offset == old_offset and new_shape[new_dim] == chunk_size:
            return new_dim
        new_offset *= new_shape[new_dim]
    return None


@register_propagate_rule(aten.view.default)
def propagate_view(view_node: Node) -> _HandlerRetType:
    input_node = view_node.args[0]
    if not isinstance(input_node, Node):
        raise AssertionError(
            f"expected input_node to be a Node, got {type(input_node)}"
        )
    input_shape = list(get_fake_tensor_from_node_arg(input_node).shape)  # type: ignore[union-attr]
    output_shape = list(get_fake_tensor_from_node_arg(view_node).shape)  # type: ignore[union-attr]

    def fwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        input_meta = get_chunking_meta(input_node)
        if input_meta is None:
            return _bool_to_status(False)
        if input_meta.chunk_dim is None:
            return _bool_to_status(copy_chunking_meta(view_node, input_meta))
        new_dim = _find_chunk_dim_after_reshape(
            input_shape, output_shape, input_meta.chunk_dim
        )
        if new_dim is None:
            return PropagateStatus.FAIL
        return _bool_to_status(
            set_chunking_meta(view_node, meta=input_meta, chunk_dim=new_dim)
        )

    def bwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        output_meta = get_chunking_meta(view_node)
        if output_meta is None:
            return _bool_to_status(False)
        if output_meta.chunk_dim is None:
            return _bool_to_status(copy_chunking_meta(input_node, output_meta))
        new_dim = _find_chunk_dim_after_reshape(
            output_shape, input_shape, output_meta.chunk_dim
        )
        if new_dim is None:
            return PropagateStatus.FAIL
        return _bool_to_status(
            set_chunking_meta(input_node, meta=output_meta, chunk_dim=new_dim)
        )

    return fwd(), bwd()


@register_propagate_rule(
    [
        aten.expand.default,
    ]
)
def propagate_expand(expand_node: Node) -> _HandlerRetType:
    input_node = expand_node.args[0]
    if not isinstance(input_node, Node):
        raise AssertionError(
            f"expected input_node to be a Node, got {type(input_node)}"
        )

    input_ft = get_fake_tensor_from_node_arg(input_node)
    if input_ft is None:
        raise AssertionError("expected input_ft to not be None")
    output_ft = get_fake_tensor_from_node_arg(expand_node)
    if output_ft is None:
        raise AssertionError("expected output_ft to not be None")
    input_shape = list(input_ft.shape)
    output_shape = list(output_ft.shape)

    if input_ft.numel() == 1:
        # Scalar input: combined fwd/bwd rule
        output_meta = get_chunking_meta(expand_node)
        if output_meta is None:
            return _bool_to_status(False)
        return _bool_to_status(set_chunking_meta(input_node))

    # How many leading dims are added by expand
    dim_offset = len(output_shape) - len(input_shape)

    def is_expand_dim(out_dim: int) -> bool:
        """Check if out_dim is a broadcast dimension (newly added or size 1 in input)."""
        return out_dim < dim_offset or input_shape[out_dim - dim_offset] == 1

    def fwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        input_meta = get_chunking_meta(input_node)
        if input_meta is None:
            return _bool_to_status(False)
        # Fail if chunk_dim is an expand dimension (input size 1 broadcast to larger size)
        if input_meta.chunk_dim is not None and is_expand_dim(
            input_meta.chunk_dim + dim_offset
        ):
            return PropagateStatus.FAIL
        return _bool_to_status(copy_chunking_meta(expand_node, input_meta))

    def bwd() -> PropagateStatus:
        if not isinstance(input_node, Node):
            raise AssertionError(
                f"expected input_node to be a Node, got {type(input_node)}"
            )
        output_meta = get_chunking_meta(expand_node)
        if output_meta is None:
            return _bool_to_status(False)
        # Fail if chunk_dim is an expand dimension (input size 1 broadcast to larger size)
        if output_meta.chunk_dim is not None and is_expand_dim(output_meta.chunk_dim):
            return PropagateStatus.FAIL
        return _bool_to_status(copy_chunking_meta(input_node, output_meta))

    return fwd(), bwd()


@register_propagate_rule(
    [
        aten.sum.default,
    ]
)
def propagate_sum_to_scalar(sum_node: Node) -> _HandlerRetType:
    input_node = sum_node.args[0]
    if not isinstance(input_node, Node):
        raise AssertionError(
            f"expected input_node to be a Node, got {type(input_node)}"
        )

    def fwd() -> PropagateStatus:
        input_meta = get_chunking_meta(input_node)  # pyrefly: ignore[bad-argument-type]
        if has_nop_chunking_meta(input_node):  # pyrefly: ignore[bad-argument-type]
            return _bool_to_status(False)

        if input_meta is None:
            raise AssertionError("expected input_meta to not be None")
        if input_meta.chunk_dim is not None:
            changed = update_chunking_meta(sum_node, need_sum=True, chunk_by=None)
            return _bool_to_status(changed)
        return PropagateStatus.FAIL

    def bwd() -> PropagateStatus:
        input_meta = get_chunking_meta(input_node)  # pyrefly: ignore[bad-argument-type]
        if has_nop_chunking_meta(sum_node):
            return _bool_to_status(False)

        # We won't know how the input is chunked if sum_node.meta.need_sum is True.
        # On the other hand, sum_node.meta.need_sum is True can only happen
        # by propagating from input_node. Doing sanity check here is fine
        # since the input_node.meta should have already been properly
        # setup.
        output_meta = get_chunking_meta(sum_node)
        if (
            output_meta is not None
            and output_meta.need_sum
            and input_meta is not None
            and input_meta.chunk_dim is not None
        ):
            return _bool_to_status(False)

        return PropagateStatus.FAIL

    return fwd(), bwd()
<|endoftext|>import functools
import logging
from collections.abc import Callable, Sequence
from typing import TypeAlias

import torch
from torch.fx import Node

from .common import CantChunk
from .core import get_chunking_meta, get_chunking_metas, log, update_chunking_meta
from .utils import (
    format_node_with_chunking_meta,
    get_args_of_node_type,
    get_scale_by_from_metas,
    get_scale_by_from_node,
)


aten = torch.ops.aten
prims = torch.ops.prims

_HandlerType: TypeAlias = Callable[[Node], bool]
propagate_rules: dict[torch._ops.OpOverload, _HandlerType] = {}


def _register_propagate_rule(
    aten_op: torch._ops.OpOverload | Sequence[torch._ops.OpOverload],
    handler: _HandlerType,
) -> _HandlerType:
    if not isinstance(aten_op, (list, tuple)):
        aten_op = [aten_op]  # type: ignore[assignment, list-item]

    if not isinstance(aten_op, (list, tuple)):
        raise AssertionError(f"{type(aten_op)=}")
    for op in aten_op:
        if not isinstance(op, torch._ops.OpOverload):
            raise AssertionError(f"expected OpOverload, got {type(op)}")
        propagate_rules[op] = handler
    return handler


def register_propagate_rule(
    aten_op: torch._ops.OpOverload | Sequence[torch._ops.OpOverload],
) -> Callable[[_HandlerType], _HandlerType]:
    return functools.partial(_register_propagate_rule, aten_op)


def propagate_scale_by(nodes_with_chunking_meta: Sequence[Node]) -> None:
    """
    The input is a list of nodes that have chunking metadata.
    The nodes are already sorted in topological order.
    """
    for node in nodes_with_chunking_meta:
        arg_nodes = get_args_of_node_type(node)
        arg_metas = get_chunking_metas(arg_nodes)

        if all(arg_meta is None for arg_meta in arg_metas):
            # should be graph input of the chunking subgraph
            continue

        if log.isEnabledFor(logging.DEBUG):
            print("Propagate scale_by:")
            format_node_with_chunking_meta(node, True)

        if not all(arg_meta is not None for arg_meta in arg_metas):
            raise AssertionError(node.format_node())

        # None of the input has scale_by set
        if all(arg_meta.scale_by is None for arg_meta in arg_metas):  # type: ignore[union-attr]
            continue

        target = node.target
        if (
            not isinstance(target, torch._ops.OpOverload)
            or target not in propagate_rules
        ):
            raise CantChunk(
                f"Missing scale_by propagation rule for target {target}: {node.format_node()}"
            )

        if not propagate_rules[target](node):
            raise CantChunk(
                f"scale_by propagate rule for {target} fail: {node.format_node()}"
            )


@register_propagate_rule(
    [
        aten.div.Tensor,
    ]
)
def propagate_div(div_node: Node) -> bool:
    lhs_node, rhs_node = div_node.args[:2]
    if not isinstance(lhs_node, Node):
        raise AssertionError(f"expected lhs_node to be Node, got {type(lhs_node)}")
    lhs_scale_by = get_scale_by_from_node(lhs_node)

    # When gradient accumulation is enabled, rhs_node can be a constant
    # representing the gradient accumulation steps
    rhs_scale_by = (
        get_scale_by_from_node(rhs_node) if isinstance(rhs_node, Node) else None
    )
    if lhs_scale_by and rhs_scale_by is None:
        update_chunking_meta(div_node, scale_by=lhs_scale_by)
        return True
    return False


@register_propagate_rule(
    [
        aten.where.self,
    ]
)
def propagate_where(where_node: Node) -> bool:
    cond_node, true_node, false_node = where_node.args
    if not isinstance(cond_node, Node):
        raise AssertionError(f"expected cond_node to be Node, got {type(cond_node)}")
    if not isinstance(true_node, Node):
        raise AssertionError(f"expected true_node to be Node, got {type(true_node)}")
    if not isinstance(false_node, Node):
        raise AssertionError(f"expected false_node to be Node, got {type(false_node)}")
    cond_meta, true_meta, false_meta = get_chunking_metas(
        [cond_node, true_node, false_node]
    )
    out_meta = get_chunking_meta(where_node)

    if true_meta is None:
        raise AssertionError("expected true_meta to be not None")
    if false_meta is None:
        raise AssertionError("expected false_meta to be not None")
    if true_meta.scale_by and not false_meta.scale_by:
        # the false_node must be all zero
        if false_node.target != aten.full.default:
            return False
        if false_node.args[1] != 0.0:
            return False
        if out_meta is None:
            raise AssertionError("expected out_meta to be not None")
        out_meta.scale_by = true_meta.scale_by
        return True
    return False


@register_propagate_rule(
    [
        aten.exp.default,
        aten.log.default,
        aten.tanh.default,
        aten.eq.Tensor,
    ]
)
def propagate_requires_no_scaling(out_node: Node) -> bool:
    """
    For nonlinear ops (exp, log, tanh) scale_by cannot be propagated
    through since f(S*x) != S*f(x). For boolean-output ops (eq) scale_by
    is meaningless. These ops only appear in the chunking subgraph when
    scale_by is None (e.g. the final gradient is 1).
    """
    args_node = get_args_of_node_type(out_node)
    args_meta = get_chunking_metas(args_node)
    out_meta = get_chunking_meta(out_node)

    scale_by = get_scale_by_from_metas(*args_meta)  # type: ignore[arg-type]
    if scale_by is not None:
        raise AssertionError(
            f"Nonlinear op {out_node.target} requires scale_by=None, got {scale_by}"
        )
    if out_meta is None:
        raise AssertionError("expected out_meta to be not None")
    out_meta.scale_by = None
    return True


@register_propagate_rule(
    [
        aten.mul.Tensor,
        prims.convert_element_type.default,
        aten.neg.default,
        aten.sum.dim_IntList,
        aten.sum.default,  # sum to scalar
        aten.amax.default,
        aten.mm.default,
        aten.permute.default,
        aten.expand.default,
        aten.squeeze.dim,
        aten.unsqueeze.default,
        aten.gather.default,
        aten.scatter_add.default,
        aten.view.default,
    ]
)
def propagate_general_copy(out_node: Node) -> bool:
    """
    A rule that holds for multiple ops: the scale_by of the output is
    set to the only scale_by of input nodes or None if no input has scale_by
    set.
    """
    args_node = get_args_of_node_type(out_node)
    args_meta = get_chunking_metas(args_node)
    out_meta = get_chunking_meta(out_node)

    scale_by = get_scale_by_from_metas(*args_meta)  # type: ignore[arg-type]
    if out_meta is None:
        raise AssertionError("expected out_meta to be not None")
    out_meta.scale_by = scale_by
    return True


@register_propagate_rule(aten.scatter.value)
def propagate_scatter_value(out_node: Node) -> bool:
    # The backward of scatter.value always has value=0 (gradient of a constant),
    # so S * scatter(x, idx, 0) = scatter(S*x, idx, 0) holds.
    value = out_node.args[3]
    if value != 0:
        return False
    return propagate_general_copy(out_node)


@register_propagate_rule(
    [
        aten.add.Tensor,
        aten.sub.Tensor,
    ]
)
def propagate_add_sub(out_node: Node) -> bool:
    """
    The scale_by node of the two arguments must be the same.
    """
    lhs_node, rhs_node = get_args_of_node_type(out_node)
    if not isinstance(lhs_node, Node):
        raise AssertionError(f"expected lhs_node to be Node, got {type(lhs_node)}")
    if not isinstance(rhs_node, Node):
        raise AssertionError(f"expected rhs_node to be Node, got {type(rhs_node)}")
    lhs_meta, rhs_meta = get_chunking_metas([lhs_node, rhs_node])
    if lhs_meta is None:
        raise AssertionError("expected lhs_meta to be not None")
    if rhs_meta is None:
        raise AssertionError("expected rhs_meta to be not None")
    if lhs_meta.scale_by is rhs_meta.scale_by:
        update_chunking_meta(out_node, scale_by=lhs_meta.scale_by)
        return True
    return False


@register_propagate_rule(
    [
        prims.fma.default,
    ]
)
def propagate_fma(out_node: Node) -> bool:
    mul_lhs, mul_rhs, add_rhs = out_node.args[:3]
    if not isinstance(mul_lhs, Node):
        raise AssertionError(f"expected mul_lhs to be Node, got {type(mul_lhs)}")
    if not isinstance(mul_rhs, Node):
        raise AssertionError(f"expected mul_rhs to be Node, got {type(mul_rhs)}")
    if not isinstance(add_rhs, Node):
        raise AssertionError(f"expected add_rhs to be Node, got {type(add_rhs)}")
    mul_lhs_meta, mul_rhs_meta, add_rhs_meta = get_chunking_metas(
        [mul_lhs, mul_rhs, add_rhs]
    )
    if mul_lhs_meta is None:
        raise AssertionError("expected mul_lhs_meta to be not None")
    if mul_rhs_meta is None:
        raise AssertionError("expected mul_rhs_meta to be not None")
    add_lhs_scale_by = get_scale_by_from_metas(mul_lhs_meta, mul_rhs_meta)
    if add_rhs_meta is None:
        raise AssertionError("expected add_rhs_meta to be not None")
    if add_lhs_scale_by is add_rhs_meta.scale_by:
        update_chunking_meta(out_node, scale_by=add_lhs_scale_by)
        return True
    return False
<|endoftext|>from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from torch.fx import Node


class CantChunk(RuntimeError):
    pass


@dataclasses.dataclass
class ChunkingMeta:
    # The value of the current node having this ChunkingMeta should be scaled by the specified scalar
    # tensor in the `scale_by` field. Need this since we pretend tangent to be 1 first and
    # scale the affected tensor later. Need propagate such information
    # to downstream.
    scale_by: Node | None = None

    # The dimension of the current tensor that get chunked.
    # Can be None if the current tensor is not chunked. E.g., when
    # the current tensor is a scalar tensor generated by summing a chunked tensor.
    #
    # To recover a tensor with non-None chunk_dim, we need concat each
    # chunk at the 'chunk_dim' dimension.
    chunk_dim: int | None = None

    # The original tensor is the sum of each tensor in the chunking subgraph.
    # `chunk_dim` and `need_sum` are exclusive! A node can not have both a non
    # None chunk_dim and a need_sum with True value.
    #
    # Note for some special cases like the tangent placeholder node, both
    # chunk_dim can be None and need_sum can be false, but scale_by
    # in that case is the tangent node itself.
    need_sum: bool = False

    def __post_init__(self) -> None:
        if self.chunk_dim is not None and self.need_sum:
            raise AssertionError(
                f"Can not have both a non-None chunk_dim and a true need_sum: {self.chunk_dim}"
            )

    def copy(self, **kwargs: Any) -> ChunkingMeta:
        meta = ChunkingMeta(**self.__dict__)
        for k, v in kwargs.items():
            setattr(meta, k, v)
        return meta

    @staticmethod
    def equal(
        lhs_meta: ChunkingMeta, rhs_meta: ChunkingMeta, skip_scale_by: bool = False
    ) -> bool:
        if skip_scale_by:
            lhs_meta = lhs_meta.copy(scale_by=None)
            rhs_meta = rhs_meta.copy(scale_by=None)
        return lhs_meta == rhs_meta

    def chunked_by_dim(self, dim: int) -> bool:
        return self.chunk_dim == dim

    @staticmethod
    def is_nop(meta: ChunkingMeta | None) -> bool:
        return meta is None or meta == ChunkingMeta()
<|endoftext|>from collections.abc import Sequence
from typing import Any

import torch
from torch.fx import Graph, GraphModule, Node
from torch.utils._pytree import tree_flatten

from .common import CantChunk, ChunkingMeta


def is_tangent_node(node: Node) -> bool:
    return node.op == "placeholder" and "tangent" in node.name


def get_args_of_node_type(node: Node) -> Sequence[Node]:
    return [x for x in tree_flatten((node.args, node.kwargs))[0] if isinstance(x, Node)]


def use_tangent(node: Node) -> bool:
    """
    Whether the fx node uses tangent input.
    """

    return any(
        is_tangent_node(arg)  # type: ignore[operator]
        for arg in get_args_of_node_type(node)
    )


def compute_tensor_size(*args: Any, count_bytes: bool = True, **kwargs: Any) -> int:
    """
    Compute total tensor sizes from fx.Node in args & kwargs.
    """
    flat_args, _ = tree_flatten((args, kwargs))
    tot = 0
    for arg in flat_args:
        if (fake_tensor := get_fake_tensor_from_node_arg(arg)) is None:
            continue
        tot += fake_tensor.numel() * (fake_tensor.dtype.itemsize if count_bytes else 1)
    return tot


def get_fake_tensor_from_node_arg(
    node: torch.fx.node.Argument,
) -> torch.Tensor | None:
    if (
        not hasattr(node, "meta")
        or ("val" not in node.meta)  # type: ignore[union-attr]
        or not isinstance(node.meta["val"], torch.Tensor)  # type: ignore[union-attr]
    ):
        return None
    return node.meta["val"]  # type: ignore[union-attr]


def get_nodes_with_chunking_meta(graph: torch.fx.Graph) -> Sequence[Node]:
    from .core import get_chunking_meta

    output = []
    for node in graph.nodes:
        if get_chunking_meta(node):
            output.append(node)
    return output


def format_node_with_chunking_meta(
    node: torch.fx.Node, include_args: bool = False
) -> None:
    """
    Print the node with chunking metadata for the current node if exists.

    If include_args is True, also print chunking metadata for Node arguments.
    """
    from torch._inductor.runtime.runtime_utils import green_text

    from .core import get_chunking_meta

    fake_tensor = get_fake_tensor_from_node_arg(node)
    shape = list(fake_tensor.shape) if fake_tensor is not None else "?"
    print(f"  {shape} {node.format_node()}")

    if meta := get_chunking_meta(node):
        print(f"    {green_text(str(meta))}")

    if include_args:
        for arg in get_args_of_node_type(node):
            if arg_meta := get_chunking_meta(arg):
                print(f"    {arg}: {green_text(str(arg_meta))}")


def has_any_chunking_meta(*node_list: Node) -> bool:
    from .core import get_chunking_meta

    return any(get_chunking_meta(node) for node in node_list)


def get_first_chunking_meta(*node_list: Node) -> ChunkingMeta | None:
    """
    Get the first non-none chunking metadata if there is any.
    """
    from .core import get_chunking_meta

    for node in node_list:
        if (meta := get_chunking_meta(node)) is not None:
            return meta

    return None


def get_scale_by_from_metas(*metas: ChunkingMeta) -> Node | None:
    """
    If there are multiple ChunkingMeta having the scale_by field,
    raise a CantChunk exception.

    If no ChunkingMeta has scale_by field, return None.
    Otherwise return the only scale_by field.
    """

    scale_by_list = []

    # don't do dedup on the scale_by field on purpose for this API
    for meta in metas:
        if meta.scale_by is not None:
            scale_by_list.append(meta.scale_by)

    if len(scale_by_list) > 1:
        raise CantChunk("Multiple scale_by")

    return scale_by_list[0] if len(scale_by_list) == 1 else None


def get_scale_by_from_node(node: Node) -> Node | None:
    from .core import get_chunking_meta

    meta = get_chunking_meta(node)
    return meta.scale_by if meta is not None else None


def get_node_is_scalar(nodes: Sequence[Node]) -> dict[Node, bool]:
    """
    Returns a dict map a node to 'is_scalar'.
    """
    node_is_scalar = {}
    for node in nodes:
        ft = get_fake_tensor_from_node_arg(node)
        if ft is None:
            raise AssertionError(f"expected a fake tensor for node {node}, got None")
        node_is_scalar[node] = ft.numel() == 1
    return node_is_scalar


def get_node_ndim(nodes: Sequence[Node]) -> dict[Node, int]:
    """
    Returns a dict map a node to 'ndim'.
    """
    node_ndim = {}
    for node in nodes:
        ft = get_fake_tensor_from_node_arg(node)
        if ft is None:
            raise AssertionError(f"expected a fake tensor for node {node}, got None")
        node_ndim[node] = ft.ndim
    return node_ndim


def is_chunked_by_dim(node: Node, dim: int) -> bool:
    from .core import get_chunking_meta

    meta = get_chunking_meta(node)
    return meta is not None and meta.chunked_by_dim(dim)


def tangent_has_chunking_meta(gm: GraphModule) -> bool:
    from .core import get_chunking_meta

    return any(
        is_tangent_node(node) and get_chunking_meta(node) is not None
        for node in gm.graph.find_nodes(op="placeholder", sort=False)
    )


def get_tangent_nodes(graph: Graph) -> Sequence[Node]:
    tangents = []
    for node in graph.find_nodes(op="placeholder", sort=False):
        if is_tangent_node(node):
            tangents.append(node)
    return tangents
<|endoftext|>import dataclasses
import functools
import itertools
import logging
import operator
import os
from collections.abc import Sequence
from typing import Optional

import torch
from torch._dynamo.utils import detect_fake_mode
from torch._inductor import config, metrics
from torch._inductor.utils import cache_on_self
from torch.fx import GraphModule
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.utils import _pytree
from torch.utils._ordered_set import OrderedSet

from .applier import ChunkingApplier
from .common import CantChunk
from .core import find_amplifier_node
from .propagator import propagate
from .utils import get_tangent_nodes, tangent_has_chunking_meta


aten = torch.ops.aten
prims = torch.ops.prims
log = torch._logging.getArtifactLogger(__name__, "auto_chunker")


def decide_num_chunks(gm: GraphModule) -> int:
    # TODO: this is just a placeholder for now.
    # We should either create heuristics or do autotuning to
    # decide the number of chunks.
    return 8


def chunk(gm: GraphModule) -> GraphModule:
    """
    Chunk input tensors for operations that amplify the tensor size significantly.
    The chunking operation is propagated thru the fx graph until a point we should
    re-generate non-chunked tensors.

    Only chunk across the batch dimension of the tensor for now.
    """
    graph = gm.graph

    if torch._inductor.config.cpp_wrapper:
        raise CantChunk("cpp wrapper does not support codegening invoke_subgraph")

    if gm.meta.get("produced_by_chunker", False):
        # Don't chunk a graph produced by the chunker
        return gm

    if len(get_tangent_nodes(gm.graph)) == 0:
        # no tangents. Can be the optimizer graph. Skip chunking
        return gm

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Joint graph before chunking:\n%s", gm.print_readable(False))

    amplifier_node = find_amplifier_node(graph)
    if amplifier_node is None:
        raise CantChunk("Skip chunking due to no amplifier node found")

    if amplifier_node.meta["val"]._has_symbolic_sizes_strides:
        raise CantChunk("Can't chunk due to dynamic shape")

    propagate(amplifier_node)
    if not tangent_has_chunking_meta(gm):
        raise CantChunk(
            "Skip chunking either because the graph is for inference only or "
            "because the chunking metadata does not propagate to the backward "
            "(e.g. due to too trivial loss function)"
        )

    num_chunks = config.auto_chunker.num_chunk or decide_num_chunks(gm)
    out_gm = ChunkingApplier(gm, num_chunks).apply()
    metrics.num_auto_chunking += 1
    log.debug("AutoChunker being applied with %s chunks", num_chunks)
    return out_gm
<|endoftext|>import logging
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch._inductor import config
from torch.fx import Graph, Node
from torch.utils._ordered_set import OrderedSet

from .common import ChunkingMeta
from .utils import (
    compute_tensor_size,
    get_args_of_node_type,
    get_fake_tensor_from_node_arg,
    use_tangent,
)


aten = torch.ops.aten
log = torch._logging.getArtifactLogger(__name__, "auto_chunker")


def set_chunking_meta(
    node: Node, meta: ChunkingMeta | None = None, **kwargs: Any
) -> bool:
    """
    kwargs can override fields in the passed in `meta`
    """
    if meta is None:
        meta = ChunkingMeta(**kwargs)
    else:
        # make a copy to avoid override the passed in instance
        meta = meta.copy()
        for k, v in kwargs.items():
            setattr(meta, k, v)

    old_meta = get_chunking_meta(node)
    node.meta["chunking"] = meta
    return old_meta is None or old_meta != meta


def update_chunking_meta(node: Node, **kwargs: Any) -> bool:
    """
    Unlike set_chunking_meta, this function keeps the existing chunking
    metadata if it's not overridden.
    """
    changed = False
    meta = get_chunking_meta(node)
    if meta is None:
        meta = ChunkingMeta()
        changed = True
    for k, v in kwargs.items():
        if getattr(meta, k, None) != v:
            changed = True
            setattr(meta, k, v)

    node.meta["chunking"] = meta
    return changed


def set_chunking_meta_if_none(
    nodes: Sequence[Node],
    meta: ChunkingMeta,
    filter_for_nop: Callable[[Node], bool] | None = None,
) -> bool:
    """
    If filter_for_nop returns true for a node, we set the chunking
    meta to nop instead.
    """
    changed = False
    for node in nodes:
        if get_chunking_meta(node) is None:
            changed = True
            if filter_for_nop and filter_for_nop(node):
                set_chunking_meta(node)
            else:
                set_chunking_meta(node, meta)
    return changed


def copy_chunking_meta(dst_node: Node, src_node: Node | ChunkingMeta) -> bool:
    if isinstance(src_node, torch.fx.Node):
        src_meta = get_chunking_meta(src_node)
    else:
        if not isinstance(src_node, ChunkingMeta):
            raise AssertionError(
                f"expected src_node to be ChunkingMeta, got {type(src_node)}"
            )
        src_meta = src_node
    if not src_meta:
        raise AssertionError("expected src_meta to be truthy")
    return set_chunking_meta(dst_node, src_meta)


def get_chunking_meta(node: Node) -> ChunkingMeta | None:
    return node.meta.get("chunking")


def has_nop_chunking_meta(node: Node) -> bool:
    return ChunkingMeta.is_nop(get_chunking_meta(node))


def get_chunking_metas(
    nodes: Sequence[Node], skip_none: bool = False
) -> Sequence[ChunkingMeta | None]:
    return [
        get_chunking_meta(node)
        for node in nodes
        if not skip_none or get_chunking_meta(node) is not None
    ]


eligible_amplifier_node = OrderedSet(
    [
        aten.mm.default,
        aten.addmm.default,
    ]
)


def find_amplifier_node(graph: Graph) -> Node | None:
    r"""
    Find the 'amplifier' node which is a node that generates large
    output with small/medium input.

    If there are multiple amplifier nodes, return the one with the largest
    amplification ratio.
    """

    amplifier_nodes_ratio = []

    for node in graph.nodes:
        if use_tangent(node):
            # enter backward part of the graph
            break

        # Only trigger chunking for a small set of nodes like matmul for now
        if node.op != "call_function" or node.target not in eligible_amplifier_node:
            continue

        input_size = compute_tensor_size(node.args, node.kwargs)
        output_size = compute_tensor_size(node)

        if input_size == 0:
            continue

        ratio = output_size / input_size
        if (
            output_size > config.auto_chunker.output_size_threshold
            and ratio > config.auto_chunker.amplify_ratio_threshold
        ):
            amplifier_nodes_ratio.append((node, ratio))

    amplifier_nodes_ratio = sorted(
        amplifier_nodes_ratio, key=lambda x: x[1], reverse=True
    )
    return amplifier_nodes_ratio[0][0] if len(amplifier_nodes_ratio) > 0 else None


def reorder_nodes(graph: Graph) -> Graph:
    """
    Create a new graph to be like:
    1. all nodes run before `chunking_subgraph_nodes`
    2. all nodes in `chunking_subgraph_nodes`
    3. all nodes run after `chunking_subgraph_nodes`

    Return a new graph so it's easier to fallback.
    """
    from .applier import is_chunking_subgraph_input

    # `pre_chunking_nodes` are all nodes that only depends on
    # nodes inside `pre_chuning_nodes`
    pre_chunking_nodes: OrderedSet[Node] = OrderedSet()

    for node in graph.nodes:
        if node.op == "placeholder" or is_chunking_subgraph_input(node):
            # these nodes have chunking meta but they should be placed
            # before we do chunking
            pre_chunking_nodes.add(node)

        if get_chunking_meta(node) is not None:
            continue
        if all(arg in pre_chunking_nodes for arg in get_args_of_node_type(node)):
            pre_chunking_nodes.add(node)

    post_chunking_nodes = []

    def _copy_node(typestr: str, node: Node) -> None:
        if log.isEnabledFor(logging.DEBUG):
            fake_tensor = get_fake_tensor_from_node_arg(node)
            shape = list(fake_tensor.shape) if fake_tensor is not None else "?"
            log.debug(" - %s: %s %s", typestr, shape, node.format_node())
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    # add pre_chunking_nodes
    new_graph = Graph()
    env: dict[Node, Node] = {}
    for node in pre_chunking_nodes:
        _copy_node("prechunking", node)

    # add nodes in the chunking subgraph
    for node in graph.nodes:
        if node in pre_chunking_nodes:
            continue
        elif get_chunking_meta(node):
            _copy_node("chunking", node)
        else:
            post_chunking_nodes.append(node)

    for node in post_chunking_nodes:
        _copy_node("postchuking", node)

    if graph._len != new_graph._len:
        raise AssertionError(
            f"expected graph._len == new_graph._len, got {graph._len} and {new_graph._len}"
        )
    new_graph.eliminate_dead_code()
    new_graph.lint()

    # Need replace the scale_by node in the chunking meta with the new node
    for node in new_graph.nodes:
        meta = get_chunking_meta(node)
        if meta and meta.scale_by is not None:
            meta.scale_by = env[meta.scale_by]

    return new_graph
<|endoftext|>import copy
import logging
import operator
from typing import Any

import torch
from torch import Tensor
from torch._dynamo.utils import detect_fake_mode
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

from .core import get_chunking_meta, reorder_nodes
from .utils import get_args_of_node_type, get_fake_tensor_from_node_arg, is_tangent_node


log = torch._logging.getArtifactLogger(__name__, "auto_chunker")
aten = torch.ops.aten
prims = torch.ops.prims


def _factory_args(fake_tensor: Tensor) -> dict[str, Any]:
    return {
        "device": fake_tensor.device,
        "dtype": fake_tensor.dtype,
    }


def fake_tensor_prop(gm: GraphModule) -> None:
    inputs = []
    for node in gm.graph.find_nodes(op="placeholder", sort=False):
        fake_tensor = get_fake_tensor_from_node_arg(node)
        if fake_tensor is not None:
            inputs.append(fake_tensor)
        else:
            inputs.append(node)

    fake_mode = detect_fake_mode(inputs)
    FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*inputs)


def is_chunking_subgraph_input(node: Node) -> bool:
    meta = get_chunking_meta(node)
    if meta is None or is_tangent_node(node):
        return False
    arg_nodes = get_args_of_node_type(node)
    arg_nodes_no_meta = [node for node in arg_nodes if get_chunking_meta(node) is None]
    return len(arg_nodes_no_meta) > 0 or node.op == "placeholder"


class ChunkingApplier:
    """
    A class that chunks the graph assuming chunking metadata has already
    been attached to the nodes in the chunking subgraph.
    """

    def __init__(self, parent_gm: GraphModule, num_chunk: int):
        self.gm = parent_gm
        self.parent_graph = reorder_nodes(self.gm.graph)
        # From this point on self.parent_graph is not equal to self.gm.graph
        # due to reordering. We create a new copy of the graph so it's
        # easier to fallback if chunking fails.
        self.num_chunk = num_chunk

        # tangent node to the all-one tensor
        self.overriden_tangent: dict[Node, Node | None] = {}

        self.subgraph_input: list[Node] = []
        self.subgraph_body: list[Node] = []
        self.subgraph_output: list[Node] = []
        self._categorize_subgraph_nodes()

        self.chunk_sizes: list[int] | None = None

        # First index is node index,
        # Second index is chunk index.
        # Node index may be different to the index for self.subgraph_input
        # since not every subgraph_input may be chunked.
        # check self.chunk_subgraph_input for more details
        self.chunked_subgraph_input: list[list[Node]] = []

        self.accumulators: dict[Node, Node | None] = {}
        self.chunks_for_recovering: dict[Node, list[Node]] = {}
        for node in self.subgraph_output:
            meta = get_chunking_meta(node)
            if not meta:
                raise AssertionError("expected chunking meta for subgraph output node")
            if meta.chunk_dim is not None:
                self.chunks_for_recovering[node] = []
            else:
                # the accumulator nodes is created later
                self.accumulators[node] = None

        self.chunk_size_to_gm_attr: dict[int, str] = {}

    def _categorize_subgraph_nodes(self) -> None:
        """
        For each chunked node, decide if it's an input/body/output node of the
        chunking subgraph.
        """
        for node in self.parent_graph.nodes:
            meta = get_chunking_meta(node)

            # The node does not have chunking metadata, skip
            if meta is None:
                continue

            # skip tangent nodes since they are overridden as 1 and
            # reapplied later in the bwd graph
            if is_tangent_node(node):
                self.overriden_tangent[node] = None  # will update the value later
                continue

            # To check if a node is a placeholder for the chunking
            # subgraph there are 2 alternatives
            # 1. decide it's a placeholder if it's using any non-chunked node as argument
            # 2. decide it's a placeholder if it's not using any chunked nodes as argument
            # These 2 rules are almost equivalent since we don't mix
            # chunked and non-chunked arguments for a fx.Node.
            # But one subtle difference is for aten.full node.
            # We don't need to pass in aten.full as a placeholder since
            # we can just add the chunked aten.full in the subgraph
            # directly.
            # Alternative 2 does not work for this case, so we implement
            # alternative 1.

            user_nodes = node.users
            user_nodes_no_meta = [
                node for node in user_nodes if get_chunking_meta(node) is None
            ]

            if is_chunking_subgraph_input(node):
                # None of the node's arguments are chunked. It's a placeholder
                self.subgraph_input.append(node)
            elif len(user_nodes_no_meta) > 0:
                self.subgraph_output.append(node)
            else:
                self.subgraph_body.append(node)

        if len(self.subgraph_body) <= 0:
            raise AssertionError("expected non-empty subgraph body")

    def replace_tangent_to_one(self) -> None:
        for idx, node in enumerate(self.overriden_tangent):
            if not is_tangent_node(node):
                raise AssertionError(f"expected tangent node, got {node}")

            fake_tensor = node.meta["val"]
            one = self.parent_graph.call_function(
                aten.full.default, (fake_tensor.shape, 1), _factory_args(fake_tensor)
            )

            name = "tangent_overriden_as_one_{idx}"
            one._rename(name)
            one.meta = copy.copy(node.meta)
            node.replace_all_uses_with(one)
            self.overriden_tangent[node] = one

    def chunk_subgraph_input(self) -> None:
        """
        Chunk subgraph inputs if necessary. Note that not every subgraph
        input needs to be chunked.
        E.g. the weight for matmul does not need to be chunked.
        """
        for node_idx, subgraph_input in enumerate(self.subgraph_input):
            meta = get_chunking_meta(subgraph_input)
            if meta is None:
                raise AssertionError("expected chunking meta for subgraph input")

            # not chunked
            if meta.chunk_dim is None:
                continue

            chunk_node = self.parent_graph.call_function(
                aten.chunk.default, (subgraph_input, self.num_chunk, meta.chunk_dim)
            )
            chunks = [
                self.parent_graph.call_function(operator.getitem, (chunk_node, i))
                for i in range(self.num_chunk)
            ]

            self.chunked_subgraph_input.append(chunks)

            # let's do a small meta propagation to get the size of each chunk
            if self.chunk_sizes is None:
                input_tensor = subgraph_input.meta["val"]
                tensors = torch.chunk(input_tensor, self.num_chunk)
                self.chunk_sizes = [t.size(0) for t in tensors]

    def create_accumulators(self) -> None:
        for node in self.accumulators:
            # use fp32 for accumulators
            fake_tensor = node.meta["val"]
            if fake_tensor.numel() == 1:
                # TODO(shunting) revisit
                # This tensor may be fused with mm and becomes a addmm
                # if we upcast here, addmm may fail due to incompatible
                # dtypes for input arguments.
                override_dtype = torch.float32
            else:
                override_dtype = fake_tensor.dtype

            kwargs = _factory_args(fake_tensor)
            kwargs["dtype"] = override_dtype
            accum = self.parent_graph.call_function(
                aten.full.default, (fake_tensor.shape, 0), kwargs
            )
            # reuse the meta['val']
            accum.meta = {"val": fake_tensor.to(override_dtype)}
            self.accumulators[node] = accum

    def build_subgraph(self, chunk_size: int) -> GraphModule:
        """
        Build a subgraph for the given chunk size.
        The last chunk can be smaller and a new subgraph will be created
        to avoid involving dynamic shapes.
        """
        new_graph = Graph()
        env: dict[Node, Node] = {}

        def _create_placeholder_node(input_node: Node) -> Node:
            new_node = new_graph.placeholder(input_node.name)
            fake_tensor = input_node.meta["val"]
            chunking_meta = get_chunking_meta(input_node)
            if chunking_meta is not None and chunking_meta.chunk_dim is not None:
                # the node is chunked and we need update the
                # fake tensor
                # TODO any better way to do this?
                new_tensor = aten.slice.Tensor(
                    fake_tensor, chunking_meta.chunk_dim, 0, chunk_size
                )
                fake_tensor = new_tensor
            new_node.meta = {"val": fake_tensor}
            return new_node

        for node_idx, input_node in enumerate(self.subgraph_input):
            env[input_node] = _create_placeholder_node(input_node)

        for overriden_tangent_node in self.overriden_tangent.values():
            if overriden_tangent_node is None:
                raise AssertionError("expected tangent node to be set")
            env[overriden_tangent_node] = _create_placeholder_node(
                overriden_tangent_node
            )

        for accum in self.accumulators.values():
            if accum is None:
                raise AssertionError("expected accumulator to be set")
            env[accum] = _create_placeholder_node(accum)

        for original_node in self.subgraph_body + self.subgraph_output:
            if original_node.op == "placeholder":
                raise AssertionError(
                    f"unexpected placeholder node in subgraph body/output: {original_node}"
                )

            # Chunk aten.full
            if (
                original_node.target == aten.full.default
                and (meta := get_chunking_meta(original_node)) is not None
                and meta.chunk_dim is not None
            ):
                shape = list(original_node.args[0])  # type: ignore[arg-type]
                # pyrefly: ignore [unsupported-operation]
                shape[meta.chunk_dim] = chunk_size
                env[original_node] = new_graph.call_function(
                    aten.full.default,
                    (shape, original_node.args[1]),
                    original_node.kwargs,
                )
                continue
            # Chunk aten.expand: adjust the target shape at the chunk dimension
            if (
                original_node.target == aten.expand.default
                and isinstance(original_node.args[0], torch.fx.Node)
                and (meta := get_chunking_meta(original_node)) is not None
                and meta.chunk_dim is not None
            ):
                shape = list(original_node.args[1])  # type: ignore[arg-type]
                # pyrefly: ignore [unsupported-operation]
                shape[meta.chunk_dim] = chunk_size
                env[original_node] = new_graph.call_function(
                    aten.expand.default,
                    (env.get(original_node.args[0], original_node.args[0]), shape),  # type: ignore[arg-type]
                    original_node.kwargs,
                )
                continue

            # Chunk aten.view: adjust the target shape at the chunk dimension
            if (
                original_node.target == aten.view.default
                and isinstance(original_node.args[0], torch.fx.Node)
                and (meta := get_chunking_meta(original_node)) is not None
                and meta.chunk_dim is not None
            ):
                shape = list(original_node.args[1])  # type: ignore[arg-type]
                # pyrefly: ignore [unsupported-operation]
                shape[meta.chunk_dim] = chunk_size
                env[original_node] = new_graph.call_function(
                    aten.view.default,
                    (env[original_node.args[0]], shape),  # type: ignore[arg-type]
                    original_node.kwargs,
                )
                continue

            # create the node with chunked inputs
            env[original_node] = new_graph.node_copy(original_node, lambda x: env[x])

        # Do the accumulation inside this subgraph
        for node, accum in self.accumulators.items():
            lhs = env[node]
            if accum is None:
                raise AssertionError("expected accumulator to be set")
            rhs = env[accum]

            # add `addend` and `accum`
            add_out = new_graph.call_function(aten.add.Tensor, (lhs, rhs))

            # override the chunk value
            env[node] = add_out

        out_values = []
        for node in self.subgraph_output:
            out_values.append(env[node])

        new_graph.output(tuple(out_values))
        new_graph.eliminate_dead_code()
        new_graph.lint()

        sub_gm = torch.fx._lazy_graph_module._make_graph_module(self.gm, new_graph)
        fake_tensor_prop(sub_gm)
        return sub_gm

    def build_subgraphs(self) -> None:
        if self.chunk_sizes is None:
            raise AssertionError("expected chunk_sizes to be set")
        for chunk_size in self.chunk_sizes:
            if chunk_size in self.chunk_size_to_gm_attr:
                continue

            sub_gm = self.build_subgraph(chunk_size)
            gm_attr = f"chunking_subgraph_{len(self.chunk_size_to_gm_attr)}"
            self.chunk_size_to_gm_attr[chunk_size] = gm_attr
            setattr(self.gm, gm_attr, sub_gm)

            # Mark this sub graph module so we don't recursively chunking
            # it.
            sub_gm.meta["produced_by_chunker"] = True

    def call_subgraph_for_each_chunk(self) -> None:
        for chunk_id in range(self.num_chunk):
            if self.chunk_sizes is None:
                raise AssertionError("expected chunk_sizes to be set")
            chunk_size = self.chunk_sizes[chunk_id]
            subgraph_id = self.chunk_size_to_gm_attr[chunk_size]
            sub_gm = self.parent_graph.get_attr(subgraph_id)

            args = []
            chunks_iter = iter(self.chunked_subgraph_input)
            for node in self.subgraph_input:
                chunking_meta = get_chunking_meta(node)
                if chunking_meta is None:
                    raise AssertionError("expected chunking meta for subgraph input")
                if chunking_meta.chunk_dim is not None:
                    args.append(next(chunks_iter)[chunk_id])
                else:
                    # not chunked
                    args.append(node)

            args += list(self.overriden_tangent.values())  # type: ignore[arg-type]

            for accum in self.accumulators.values():
                if accum is None:
                    raise AssertionError("expected accumulator to be set")
                args.append(accum)

            output_node = self.parent_graph.call_function(
                torch.ops.higher_order.invoke_subgraph, (sub_gm, subgraph_id, *args), {}
            )

            output_node_dict = {}
            for i, orig_node in enumerate(self.subgraph_output):
                output_node_dict[orig_node] = self.parent_graph.call_function(
                    operator.getitem, (output_node, i)
                )

            for orig_node, node_list in self.chunks_for_recovering.items():
                chunk = output_node_dict[orig_node]
                node_list.append(chunk)

            for orig_node in self.accumulators:
                self.accumulators[orig_node] = output_node_dict[orig_node]

    def recover_to_unchunked_nodes(self) -> None:
        """
        Recover the node from chunks and do the replacement.
        """
        for node in self.subgraph_output:
            meta = get_chunking_meta(node)
            if meta is None:
                raise AssertionError("expected chunking meta for subgraph output")

            recovered: torch.fx.Node = node

            if meta.chunk_dim is not None:
                chunks = self.chunks_for_recovering[node]
                recovered = self.parent_graph.call_function(
                    aten.cat.default, (chunks, meta.chunk_dim)
                )
            elif meta.need_sum:
                recovered = self.accumulators[node]  # type: ignore[assignment]

            # do scaling last
            if meta.scale_by is not None:
                recovered = self.parent_graph.call_function(
                    aten.mul.Tensor, (recovered, meta.scale_by)
                )

            # convert back to the original dtype
            if meta.need_sum:
                original_dtype = node.meta["val"].dtype
                # TODO(shunting): do we always uses a fp32 accumulator?
                if original_dtype != torch.float32:
                    recovered = self.parent_graph.call_function(
                        prims.convert_element_type.default, (recovered, original_dtype)
                    )

            if recovered is node:
                raise AssertionError("expected recovered node to differ from original")
            node.replace_all_uses_with(recovered)

    def erase_original_nodes(self) -> None:
        # Traverse reversely to erase user first
        for node in reversed(tuple(self.subgraph_body + self.subgraph_output)):
            if node.op == "placeholder":
                continue

            self.parent_graph.erase_node(node)

    def apply(self) -> GraphModule:
        with self.parent_graph.inserting_before(self.subgraph_body[0]):
            self.replace_tangent_to_one()
            self.chunk_subgraph_input()
            self.create_accumulators()
            self.build_subgraphs()
            self.call_subgraph_for_each_chunk()
            self.recover_to_unchunked_nodes()
            self.erase_original_nodes()

        newgm = torch.fx._lazy_graph_module._make_graph_module(
            self.gm,
            self.parent_graph,
        )
        fake_tensor_prop(newgm)
        if log.isEnabledFor(logging.DEBUG):
            print("Graph module after chunking:")
            newgm.print_readable()
        return newgm
<|endoftext|>import contextlib
from collections.abc import Generator, Sequence

import torch
from torch.utils._content_store import ContentStoreReader


LOAD_TENSOR_READER: ContentStoreReader | None = None


@contextlib.contextmanager
def load_tensor_reader(loc: str) -> Generator[None, None, None]:
    global LOAD_TENSOR_READER
    if LOAD_TENSOR_READER is not None:
        raise AssertionError("LOAD_TENSOR_READER is already set")
    # load_tensor is an "op", and we will play merry hell on
    # Inductor's memory planning if we return a tensor that
    # aliases another tensor that we previously returned from
    # an operator.  So unlike standard ContentStoreReader use,
    # we disable the cache so that you always get fresh storages
    # (no aliasing for you!)
    LOAD_TENSOR_READER = ContentStoreReader(loc, cache=False)
    try:
        yield
    finally:
        LOAD_TENSOR_READER = None


def register_debug_prims() -> None:
    torch.library.define(
        "debugprims::load_tensor",
        "(str name, int[] size, int[] stride, *, ScalarType dtype, Device device) -> Tensor",
    )

    @torch.library.impl("debugprims::load_tensor", "BackendSelect")
    def load_tensor_factory(
        name: str,
        size: Sequence[int],
        stride: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if LOAD_TENSOR_READER is None:
            from torch._dynamo.testing import rand_strided

            return rand_strided(size, stride, dtype, device)
        else:
            from torch._dynamo.utils import clone_input

            # device argument here takes care of coercion
            r = LOAD_TENSOR_READER.read_tensor(name, device=device)
            if list(r.size()) != size:
                raise AssertionError(f"{r.size()} != {size}")
            if list(r.stride()) != stride:
                raise AssertionError(f"{r.stride()} != {stride}")
            if r.device != device:
                raise AssertionError(f"{r.device} != {device}")

            # Unlike the other properties, we will do coercions for dtype
            # mismatch
            if r.dtype != dtype:
                r = clone_input(r, dtype=dtype)  # type: ignore[no-untyped-call]
            return r
<|endoftext|>from __future__ import annotations

import functools
from contextlib import nullcontext
from typing import Any, TYPE_CHECKING, TypeVar
from typing_extensions import ParamSpec


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import torch
import torch._decomp
import torch._prims
import torch._refs
import torch._refs.nn
import torch._refs.nn.functional
import torch._refs.special
import torch.overrides
from torch._prims_common import torch_function_passthrough


_P = ParamSpec("_P")
_R = TypeVar("_R")


@functools.cache
def torch_to_refs_map() -> dict[Any, Any]:
    """
    Mapping of torch API functions to torch._refs functions.
    E.g. torch_to_refs_map()[torch.add] == torch._refs.add
    """
    modules = [
        (torch, torch._refs),
        (torch.nn, torch._refs.nn),
        (torch.nn.functional, torch._refs.nn.functional),
        (torch.special, torch._refs.special),
        (torch.fft, torch._refs.fft),
        (torch.linalg, torch._refs.linalg),
    ]
    r: dict[Any, Any] = {
        torch.Tensor.__invert__: torch._refs.bitwise_not,
        torch.Tensor.__xor__: torch._refs.bitwise_xor,
        torch.Tensor.__and__: torch._refs.bitwise_and,
        torch.Tensor.__or__: torch._refs.bitwise_or,
        torch.Tensor.__eq__: torch._refs.eq,
        torch.Tensor.__rsub__: torch._refs.rsub,
        torch.Tensor.__rtruediv__: torch._refs.rtruediv,
        torch.Tensor.__floordiv__: torch._refs.floor_divide,
        torch.Tensor.__rfloordiv__: torch._refs.rfloordiv,
        torch.Tensor.__pow__: torch._refs.pow,
        torch.Tensor.__rpow__: torch._refs.rpow,
        torch.Tensor.new_empty: torch._refs.new_empty,
        torch.Tensor.new_full: torch._refs.new_full,
        torch.Tensor.new_zeros: torch._refs.new_zeros,
        torch.Tensor.new_ones: torch._refs.new_ones,
        torch.Tensor.fill_: torch._refs.fill_,
        torch.Tensor.zero_: torch._refs.zero_,
        torch.Tensor.to: torch._refs.to,
        torch.Tensor.sum_to_size: torch._refs.sum_to_size,
        # TODO: Should these methods be mapped some other way?
        torch.Tensor.copy_: torch._prims.copy_to,
        torch.Tensor.resize: torch._prims.resize,
    }
    for mod_torch, mod_refs in modules:
        for s in mod_refs.__all__:  # type: ignore[attr-defined]
            r[mod_torch.__dict__.get(s)] = mod_refs.__dict__.get(s)

    # Support remapping torch.Tensor.foo to _refs.foo
    for s in dir(torch.Tensor):
        if s in torch._refs.__all__:
            r[getattr(torch.Tensor, s)] = torch._refs.__dict__.get(s)

    # Support conversions
    for s in torch._refs._conversions.__all__:
        tensor_attr = getattr(torch.Tensor, s, None) or getattr(torch, s)
        r[tensor_attr] = torch._refs._conversions.__dict__.get(s)

    return r


@functools.cache
def all_prims() -> set[Any]:
    """
    Set of all prim functions, e.g., torch._prims.add in all_prims()
    """
    return {torch._prims.__dict__.get(s) for s in torch._prims.__all__}


class TorchRefsMode(torch.overrides.TorchFunctionMode):
    """
    Switches the interpretation of torch.* functions and Tensor methods to
    use PrimTorch refs in torch._refs.  (Direct calls to _refs are unaffected.)

    >>> # xdoctest: +SKIP
    >>> with TorchRefsMode():
    ...     torch.add(x, y)  # calls torch._refs.add(x, y)

    By default, this context manager will fall back on the torch.* if the
    ref does not exist; set strict=True to error if this occurs.
    If the ref exists we still would like to fall back on the torch.* sometimes,
    this behavior can be customized by passing a function to should_fallback_fn.
    """

    def __init__(
        self,
        strict: bool = False,
        should_fallback_fn: Callable[..., bool] = lambda *_: False,
        prims_mode_cls: type = nullcontext,
    ) -> None:
        self.strict = strict
        self.should_fallback_fn = should_fallback_fn
        self.prims_mode_cls = prims_mode_cls

    def __torch_function__(
        self,
        orig_func: Callable[_P, _R],
        types: Sequence[type],
        args: Sequence[Any] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        # For primitive operations, run them as is without interception
        # Unless we are in prims_mode, in which case we want to use nvprims
        if orig_func in torch_function_passthrough or orig_func in all_prims():
            with self.prims_mode_cls():
                # pyrefly: ignore [invalid-param-spec]
                return orig_func(*args, **kwargs)
        mapping = torch_to_refs_map()
        func = mapping.get(orig_func, None)

        # For torch.ops.aten.*, use registered decompositions from torch._decomp
        # torch._decomp.decomposition_table provides a mapping from
        # torch.ops.aten.* to torch._refs or torch._decomp.decompositions
        # implementations.
        # There're other ways to implement this functionality,
        # see https://github.com/pytorch/pytorch/pull/82657#discussion_r939776417
        if func is None and isinstance(orig_func, torch._ops.OpOverload):
            func = torch._decomp.decomposition_table.get(orig_func, None)
        elif func is None and isinstance(orig_func, torch._ops.OpOverloadPacket):
            default = getattr(orig_func, "default", None)
            if default is None and orig_func._dir:
                default = getattr(orig_func, orig_func._dir[0], None)
            if default is not None:
                func = torch._decomp.decomposition_table.get(default, None)

        if func is not None:
            # If the ref exists query whether we should use it or not
            if self.should_fallback_fn(self, orig_func, func, args, kwargs):
                # pyrefly: ignore [invalid-param-spec]
                return orig_func(*args, **kwargs)
            # torch calls inside func should be interpreted as refs calls
            with self:
                return func(*args, **kwargs)
        if self.strict:
            raise RuntimeError(
                f"no _refs support for {torch.overrides.resolve_name(orig_func)}"
            )
        # pyrefly: ignore [invalid-param-spec]
        return orig_func(*args, **kwargs)
<|endoftext|># mypy: allow-untyped-defs
import operator
from collections.abc import Callable, Sequence
from enum import Enum
from functools import partial, reduce
from typing import Optional, Union

import torch
import torch._prims_common as utils
import torch.library
from torch import sym_float, Tensor
from torch._C import _get_default_device
from torch._higher_order_ops.effects import new_token_tensor
from torch._library.utils import is_functional_schema
from torch._prims.debug_prims import register_debug_prims
from torch._prims.rng_prims import register_rng_prims
from torch._prims_common import (
    Dim,
    DimsSequenceType,
    DimsType,
    IntLike,
    Number,
    NumberType,
    RETURN_TYPE,
    ShapeType,
    StrideType,
    TensorLike,
    TensorLikeType,
    type_to_dtype,
)
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.overrides import handle_torch_function, has_torch_function
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


prim = torch.library.Library("prims", "DEF")
prim_impl = torch.library.Library("prims", "IMPL", "CompositeExplicitAutograd")
prim_backend_select_impl = torch.library.Library("prims", "IMPL", "BackendSelect")
prim_autograd_impl = torch.library.Library("prims", "IMPL", "Autograd")
prim_meta_impl = torch.library.Library("prims", "IMPL", "Meta")

# Experimental module containing prototype "primitive" operations.

__all__ = [
    #
    # Common datastructures and helpers
    #
    "RETURN_TYPE",
    #
    # Elementwise unary prims
    #
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "bessel_i0",
    "bessel_i0e",
    "bessel_i1",
    "bessel_i1e",
    "bessel_j0",
    "bessel_j1",
    "bitwise_not",
    "cbrt",
    "ceil",
    "conj_physical",
    "digamma",
    "erf",
    "erf_inv",
    "erfc",
    "erfcx",
    "exp",
    "expm1",
    "exp2",
    "fill",
    "floor",
    "imag",
    "isfinite",
    "lgamma",
    "log",
    "log1p",
    "log2",
    "log10",
    "ndtri",
    "neg",
    "real",
    "reciprocal",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "spherical_bessel_j0",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
    #
    # Elementwise binary prims
    #
    "add",
    "atan2",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    # 'complex',  # needs custom meta
    "div",
    "eq",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "gcd",
    "ge",
    "gt",
    "hypot",
    "igamma",
    "igammac",
    "le",
    "lt",
    "maximum",
    "minimum",
    "mul",
    "ne",
    "nextafter",
    "pow",
    "remainder",
    "rsqrt",
    "shift_left",
    "shift_right_arithmetic",
    "shift_right_logical",  # not implemented
    "sub",
    "zeta",
    #
    # View prims
    #
    "as_strided",
    "broadcast_in_dim",
    "collapse_view",
    "conj",
    "expand_dims",
    # pyrefly: ignore [bad-dunder-all]
    "slice",
    "split_dim",
    "squeeze",
    "transpose",
    "view_of",
    "view_element_type",
    #
    # Functionalized view mutations
    #
    "as_strided_scatter",
    #
    # Shape prims
    #
    "collapse",
    "cat",
    "reshape",
    "rev",
    #
    # Conditional prims
    #
    "where",
    #
    # Data conversion and movement prims
    #
    "clone",
    "convert_element_type",
    "device_put",
    "item",
    "maximum_value",
    "minimum_value",
    "copy_strided",
    #
    # Inplace prims
    #
    "copy_to",
    "resize",
    # "_set",  # Commented out, see note below
    #
    # Reduction prims
    #
    "amax",
    "amin",
    "prod",
    "sum",
    "xor_sum",
    "var",
    #
    # Tensor Creation Prims
    #
    "empty_strided",
    "empty_permuted",
    "scalar_tensor",
    "iota",
    #
    # Linear algebra (linalg) Prims
    #
    "svd",
    #
    # Randomness Prims
    #
    "normal",
    "_uniform_helper",
    #
    # FFT prims
    #
    "fft_r2c",
    "fft_c2c",
    "fft_c2r",
    #
    # prims for making/sinking tokens
    #
    "_make_token",
    "_sink_tokens",
]


def TensorMeta(
    tensorlike: NumberType | torch.Tensor | None = None,
    *,
    shape: ShapeType | None = None,
    strides: StrideType | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
):
    if isinstance(tensorlike, Number):
        if shape and not isinstance(shape, Sequence):
            raise AssertionError(
                f"shape must be None or a Sequence for Number input, got {type(shape)}"
            )
        if strides and not isinstance(strides, Sequence):
            raise AssertionError(
                f"strides must be None or a Sequence for Number input, got {type(strides)}"
            )
        inferred_shape: tuple[int, ...] = ()
        inferred_strides: tuple[int, ...] = ()
        inferred_dtype = type_to_dtype(type(tensorlike))
        inferred_device = torch.device("cpu")
        # TODO: This looks wrong, a number that is wrapped into a tensor
        # needs to behave differently than a scalar tensor for type
        # promotion purposes
    elif tensorlike is not None:
        if not isinstance(tensorlike, torch.Tensor):
            raise AssertionError(
                f"tensorlike must be torch.Tensor, got {type(tensorlike)}"
            )  # mypy
        inferred_shape = tuple(tensorlike.shape)
        inferred_strides = tuple(tensorlike.stride())
        inferred_dtype = tensorlike.dtype
        inferred_device = tensorlike.device
    else:
        # If no tensorlike "example" is given then all metadata
        # must be provided explicitly
        if shape is None:
            raise AssertionError("shape must be provided when tensorlike is None")
        if strides is None:
            raise AssertionError("strides must be provided when tensorlike is None")
        if dtype is None:
            raise AssertionError("dtype must be provided when tensorlike is None")
        if device is None:
            raise AssertionError("device must be provided when tensorlike is None")

    shape = inferred_shape if shape is None else tuple(shape)  # type: ignore[possibly-undefined]
    strides = inferred_strides if strides is None else tuple(strides)  # type: ignore[possibly-undefined]
    dtype = inferred_dtype if dtype is None else dtype  # type: ignore[possibly-undefined]
    device = inferred_device if device is None else device  # type: ignore[possibly-undefined]

    if isinstance(device, str):
        device = torch.device(device)

    return torch.empty_strided(shape, strides, dtype=dtype, device=device)


def _make_prim(
    *,
    schema: str,
    return_type: RETURN_TYPE | tuple[RETURN_TYPE, ...],
    meta: Callable,
    impl_aten: Callable,
    doc: str,
    tags: Sequence[torch.Tag] | None = None,
    use_old_custom_ops_api: bool = False,
    register_conj_neg_fallthrough: bool = False,
):
    """
    Creates a primitive operation.

    """

    def _prim_impl(*args, **kwargs):
        # always run the meta function because aten implementation will
        # typically accept more inputs (e.g., it will do promotion and
        # broadcasting) which we want to reject
        meta(*args, **kwargs)
        return impl_aten(*args, **kwargs)

    # Right now prims don't support autograd (we can and should add an
    # argument that provides an implementation for backward here.)  Because we
    # don't have derivative formulas, we must setup a custom autograd function
    # that raises an error if backwards is invoked
    def _autograd_impl(*args, **kwargs):
        return backwards_not_supported(_prim)(*args, **kwargs)

    def _backend_select_impl(*args, **kwargs):
        if kwargs.get("device") and kwargs["device"].type == "meta":
            return meta(*args, **kwargs)
        if any(isinstance(x, torch.device) and x.type == "meta" for x in args):
            return meta(*args, **kwargs)
        else:
            return _prim_impl(*args, **kwargs)

    name = schema.split("(", maxsplit=1)[0]
    schema = schema[len(name) :]

    # register non-functional ops with old custom ops API
    cpp_schema = torch._C.parse_schema(name + schema)
    if use_old_custom_ops_api or not is_functional_schema(cpp_schema):
        prim.define(name + schema, tags=torch.Tag.pt2_compliant_tag)
        prim_impl.impl(name, _prim_impl)
        prim_autograd_impl.impl(name, _autograd_impl)
        prim_meta_impl.impl(name, meta)
    else:
        mutates_args = [
            arg.name
            for arg in cpp_schema.arguments
            if arg.alias_info is not None and arg.alias_info.is_write
        ]
        prim_def = torch.library.custom_op(
            "prims::" + name,
            _prim_impl,
            mutates_args=tuple(mutates_args),
            schema=schema,
        )
        prim_def.register_fake(meta)

        # all view ops get conj/neg fallthroughs
        if return_type == RETURN_TYPE.VIEW or register_conj_neg_fallthrough:
            prim_def._lib.impl(name, torch.library.fallthrough_kernel, "Conjugate")
            prim_def._lib.impl(name, torch.library.fallthrough_kernel, "Negative")

    _prim_packet = getattr(torch._ops.ops.prims, name)
    _prim = _prim_packet.default
    if tags:
        _prim._tags = tags
    elif aten_packet := getattr(torch.ops.aten, name, None):
        overload_tags = [overload.tags for overload in aten_packet.op_overloads()]
        tags_intersection = set(overload_tags[0])
        tags_intersection.intersection_update(*overload_tags[1:])

        # don't inadvertently add to prim ops
        tags_intersection.discard(torch.Tag.core)
        # causes errors with python ref executor tests, none of the
        # data dependent pytorch ops actually decompose to prims
        tags_intersection.discard(torch.Tag.data_dependent_output)

        # iter over first tags for determinism
        _prim._tags = tuple(t for t in overload_tags[0] if t in tags_intersection)

    from torch._subclasses.fake_tensor import contains_tensor_types

    if (
        not any(contains_tensor_types(a.type) for a in _prim._schema.arguments)
        or str(
            _prim
            # See https://github.com/pytorch/pytorch/issues/103532
        )
        == "prims.device_put.default"
    ):
        prim_backend_select_impl.impl(name, _backend_select_impl)

    for p in (_prim_packet, _prim):
        p.__doc__ = doc
        p.return_type = return_type  # type: ignore[attr-defined]

        p.schema = schema
        p.prim_impl = _prim_impl
        p.prim_meta_impl = meta
        p.impl_aten = impl_aten

    torch._C._fake_dispatch_register_prim_meta(
        _prim._schema.name, _prim._schema.overload_name
    )

    return _prim


class ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND(Enum):
    DEFAULT = (0,)
    INT_TO_FLOAT = (2,)
    ALWAYS_BOOL = (3,)
    COMPLEX_TO_FLOAT = (4,)


# TODO: implement dtype validation here, too, or on the corresponding refs
def _prim_elementwise_meta(
    *args,
    type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND,
    args_with_fixed_dtypes: tuple[TensorLikeType, ...] | None = None,
) -> FakeTensor:
    """
    Meta function for elementwise operations that produce outputs in the same dtype
    as their inputs.

    Stride logic is currently incorrect.
    """

    if len(args) == 0:
        raise AssertionError("elementwise operation requires at least one argument")

    utils.check_same_dtype(*args)

    args_ = list(args)
    if args_with_fixed_dtypes is not None:
        args_ = list(args_with_fixed_dtypes) + args_

    utils.check_same_device(*args_, allow_cpu_scalar_tensors=True)
    utils.check_same_shape(*args_, allow_cpu_scalar_tensors=True)

    l2p_perm, _ = utils.compute_elementwise_output_logical_to_physical_perm(*args_)
    shape = utils.extract_shape(*args_, allow_cpu_scalar_tensors=True)

    # Acquires the dtype
    dtype = None
    scalar_type = None
    for arg in args:
        if isinstance(arg, TensorLike):
            if not utils.is_cpu_scalar_tensor(arg):
                dtype = arg.dtype
                break
            else:
                dtype = arg.dtype
        elif isinstance(arg, Number):
            scalar_type = type(arg)

    if dtype is None and scalar_type is not None:
        dtype = utils.type_to_dtype(scalar_type)

    # Acquires the device (if it exists) or number
    device = None
    number = None
    # pyrefly: ignore [bad-assignment]
    for arg in args_:
        if isinstance(arg, TensorLike):
            if utils.is_cpu_scalar_tensor(arg):
                if device is None:
                    device = arg.device
                # keep going, in case there is a cuda tensor later
            else:
                device = arg.device
                break

        elif isinstance(arg, Number):
            if number is None:
                number = arg

    # NOTE: type promotion behavior here is mostly hidden from tests because
    # references will typically handle the type promotion properly even if this doesn't
    # (but getting it wrong will cause too many casts to be inserted in traces!)
    if device is not None:
        if dtype is None:
            raise AssertionError("dtype must not be None when device is not None")
        if type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL:
            dtype = torch.bool
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.INT_TO_FLOAT:
            if utils.is_integer_dtype(dtype) or utils.is_boolean_dtype(dtype):
                dtype = torch.get_default_dtype()
        elif type_promotion == ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT:
            if utils.is_complex_dtype(dtype):
                dtype = utils.corresponding_real_dtype(dtype)

        if shape is None:
            raise AssertionError("shape must not be None when device is not None")
        return torch.empty_permuted(shape, l2p_perm, device=device, dtype=dtype)  # type: ignore[return-value]

    # Number case
    # TODO: fix number type promotion (bool, complex->float)

    # For now for symint/float, just implementing the common / simple cases of (int,float,symint,symfloat)
    seen_float = False
    if isinstance(number, (torch.SymInt, torch.SymFloat)):
        for a in args:
            if not isinstance(a, (int, float, torch.SymInt, torch.SymFloat)):
                raise AssertionError(
                    f"Expected int, float, SymInt, or SymFloat, got {type(a)}"
                )
            seen_float = seen_float or isinstance(a, (float, torch.SymFloat))
        if seen_float:
            number = sym_float(number)

    return TensorMeta(number)  # type: ignore[arg-type]


def _complex_only_elementwise_meta(*args, **kwargs):
    torch._check(
        utils.is_complex_dtype(args[0].dtype), lambda: "Only complex dtype is supported"
    )
    return _prim_elementwise_meta(*args, **kwargs)


def _make_elementwise_unary_prim(
    name: str, *, type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND, **kwargs
):
    """
    Creates an elementwise unary prim.
    """

    return _make_prim(
        schema=f"{name}(Tensor self) -> Tensor",
        meta=partial(_prim_elementwise_meta, type_promotion=type_promotion),
        return_type=RETURN_TYPE.NEW,
        **kwargs,
    )


def _make_elementwise_binary_prim(
    name: str, *, type_promotion: ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND, **kwargs
):
    """
    Creates an elementwise binary prim.
    """

    return _make_prim(
        schema=f"{name}(Tensor self, Tensor other) -> Tensor",
        meta=partial(_prim_elementwise_meta, type_promotion=type_promotion),
        return_type=RETURN_TYPE.NEW,
        **kwargs,
    )


def _not_impl(*args, **kwargs):
    raise NotImplementedError


#
# Elementwise unary operations
#


abs = _make_elementwise_unary_prim(
    "abs",
    impl_aten=torch.abs,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
)

acos = _make_elementwise_unary_prim(
    "acos",
    impl_aten=torch.acos,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

acosh = _make_elementwise_unary_prim(
    "acosh",
    impl_aten=torch.acosh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

asin = _make_elementwise_unary_prim(
    "asin",
    impl_aten=torch.asin,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

asinh = _make_elementwise_unary_prim(
    "asinh",
    impl_aten=torch.asinh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atan = _make_elementwise_unary_prim(
    "atan",
    impl_aten=torch.atan,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atanh = _make_elementwise_unary_prim(
    "atanh",
    impl_aten=torch.atanh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cos = _make_elementwise_unary_prim(
    "cos",
    impl_aten=torch.cos,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

cosh = _make_elementwise_unary_prim(
    "cosh",
    impl_aten=torch.cosh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_j0 = _make_elementwise_unary_prim(
    "bessel_j0",
    impl_aten=torch.special.bessel_j0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_j1 = _make_elementwise_unary_prim(
    "bessel_j1",
    impl_aten=torch.special.bessel_j1,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i0 = _make_elementwise_unary_prim(
    "bessel_i0",
    impl_aten=torch.i0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i0e = _make_elementwise_unary_prim(
    "bessel_i0e",
    impl_aten=torch.special.i0e,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i1 = _make_elementwise_unary_prim(
    "bessel_i1",
    impl_aten=torch.special.i1,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bessel_i1e = _make_elementwise_unary_prim(
    "bessel_i1e",
    impl_aten=torch.special.i1e,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_not = _make_elementwise_unary_prim(
    "bitwise_not",
    impl_aten=torch.bitwise_not,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _cbrt_aten(a: torch.Tensor) -> Tensor:
    torch._check(
        not a.is_complex(),
        lambda: "cbrt: Complex inputs not supported. Consider calling torch.pow(a, 1.0/3.0)",
    )
    # Returns the real cubic root of the number.
    # Note that if a < 0, pow(a, (1. / 3.)) returns the complex number
    # exp(1/3 * log(a)) = exp(1/3 * (log(abs(a)) + pi*i)) = cbrt(abs(a)) * e^{pi/3*i}
    # which is a complex number.
    # For more info see the section Note in
    # https://en.cppreference.com/w/cpp/numeric/math/cbrt
    return torch.copysign(torch.pow(a.abs(), 1 / 3), a)


cbrt = _make_elementwise_unary_prim(
    "cbrt",
    impl_aten=_cbrt_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ceil = _make_elementwise_unary_prim(
    "ceil",
    impl_aten=torch.ceil,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _conj_physical_meta(input: TensorLikeType) -> TensorLikeType:
    if not input.dtype.is_complex:
        raise RuntimeError("prims.conj_physical is only defined for complex dtypes")

    strides = utils.compute_elementwise_output_strides(input)
    return TensorMeta(input, strides=strides)


conj_physical = _make_prim(
    schema="conj_physical(Tensor self) -> Tensor",
    meta=_conj_physical_meta,
    impl_aten=torch._conj_physical,
    doc="Returns the physical conjugation of a complex tensor",
    return_type=RETURN_TYPE.NEW,
)


def _clone_meta(
    input: TensorLikeType, *, memory_format: torch.memory_format = torch.preserve_format
) -> TensorLikeType:
    if memory_format != torch.preserve_format:
        return torch.empty(
            input.shape,
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
            memory_format=memory_format,
        )
    else:
        # Match eager behavior by preserving strides for non_overlapping_and_dense tensors
        # If not, eager clone creates contiguous strides
        computed_stride = utils.compute_elementwise_output_strides(input)
        return torch.empty_strided(
            input.shape,
            computed_stride,
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
        )


clone = _make_prim(
    schema="clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor",
    meta=_clone_meta,
    impl_aten=torch.clone,
    doc="Returns the copy of a tensor",
    return_type=RETURN_TYPE.NEW,
    register_conj_neg_fallthrough=True,
)

digamma = _make_elementwise_unary_prim(
    "digamma",
    impl_aten=torch.digamma,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erf = _make_elementwise_unary_prim(
    "erf",
    impl_aten=torch.erf,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erf_inv = _make_elementwise_unary_prim(
    "erf_inv",
    impl_aten=torch.special.erfinv,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erfc = _make_elementwise_unary_prim(
    "erfc",
    impl_aten=torch.special.erfc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

erfcx = _make_elementwise_unary_prim(
    "erfcx",
    impl_aten=torch.special.erfcx,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

exp = _make_elementwise_unary_prim(
    "exp",
    impl_aten=torch.exp,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

expm1 = _make_elementwise_unary_prim(
    "expm1",
    impl_aten=torch.special.expm1,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

exp2 = _make_elementwise_unary_prim(
    "exp2",
    impl_aten=torch.special.exp2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _fill_meta(a: TensorLikeType, value: NumberType) -> TensorLikeType:
    return _prim_elementwise_meta(
        a, type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT
    )


# NOTE: fill uses _make_prim directly because it has a value parameter
fill = _make_prim(
    schema="fill(Tensor self, Scalar value) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_fill_meta,
    impl_aten=torch.fill,
    doc="",
)

floor = _make_elementwise_unary_prim(
    "floor",
    impl_aten=torch.floor,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

imag = _make_prim(
    schema="imag(Tensor(a) self) -> Tensor(a)",
    meta=partial(
        _complex_only_elementwise_meta,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    return_type=RETURN_TYPE.VIEW,
    impl_aten=torch.imag,
    doc="",
)

isfinite = _make_elementwise_unary_prim(
    "isfinite",
    impl_aten=torch.isfinite,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

lgamma = _make_elementwise_unary_prim(
    "lgamma",
    impl_aten=torch.lgamma,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log = _make_elementwise_unary_prim(
    "log",
    impl_aten=torch.log,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log1p = _make_elementwise_unary_prim(
    "log1p",
    impl_aten=torch.log1p,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log2 = _make_elementwise_unary_prim(
    "log2",
    impl_aten=torch.log2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

log10 = _make_elementwise_unary_prim(
    "log10",
    impl_aten=torch.log10,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

real = _make_prim(
    schema="real(Tensor(a) self) -> Tensor(a)",
    meta=partial(
        _complex_only_elementwise_meta,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    return_type=RETURN_TYPE.VIEW,
    impl_aten=torch.real,
    doc="",
)

reciprocal = _make_elementwise_unary_prim(
    "reciprocal",
    impl_aten=torch.reciprocal,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ndtri = _make_elementwise_unary_prim(
    "ndtri",
    impl_aten=torch.special.ndtri,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

neg = _make_elementwise_unary_prim(
    "neg",
    impl_aten=torch.neg,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

round = _make_elementwise_unary_prim(
    "round",
    impl_aten=torch.round,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

rsqrt = _make_elementwise_unary_prim(
    "rsqrt",
    impl_aten=torch.rsqrt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sign = _make_elementwise_unary_prim(
    "sign",
    impl_aten=torch.sign,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

signbit = _make_elementwise_unary_prim(
    "signbit",
    impl_aten=torch.signbit,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sin = _make_elementwise_unary_prim(
    "sin",
    impl_aten=torch.sin,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sinh = _make_elementwise_unary_prim(
    "sinh",
    impl_aten=torch.sinh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

spherical_bessel_j0 = _make_elementwise_unary_prim(
    "spherical_bessel_j0",
    impl_aten=torch.special.spherical_bessel_j0,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

sqrt = _make_elementwise_unary_prim(
    "sqrt",
    impl_aten=torch.sqrt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

tan = _make_elementwise_unary_prim(
    "tan",
    impl_aten=torch.tan,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

tanh = _make_elementwise_unary_prim(
    "tanh",
    impl_aten=torch.tanh,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

trunc = _make_elementwise_unary_prim(
    "trunc",
    impl_aten=torch.trunc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

#
# Elementwise binary operations
#

add = _make_elementwise_binary_prim(
    name="add",
    impl_aten=torch.add,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

atan2 = _make_elementwise_binary_prim(
    name="atan2",
    impl_aten=torch.atan2,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_and = _make_elementwise_binary_prim(
    "bitwise_and",
    impl_aten=torch.bitwise_and,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_or = _make_elementwise_binary_prim(
    "bitwise_or",
    impl_aten=torch.bitwise_or,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

bitwise_xor = _make_elementwise_binary_prim(
    "bitwise_xor",
    impl_aten=torch.bitwise_xor,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

# TODO: complex needs a special meta to account for its float -> complex behavior
# complex = _make_elementwise_binary_prim(
#   impl_aten=torch.complex,
#   doc="",
# )


# div prim performs truncation division on integer inputs
#   and true division for floating and complex inputs
def _div_aten(a, b):
    is_integral = isinstance(a, (bool, int, torch.SymInt)) or (
        isinstance(a, torch.Tensor) and utils.is_integer_dtype(a.dtype)
    )

    if is_integral:
        return torch.div(a, b, rounding_mode="trunc")
    else:
        return torch.true_divide(a, b)


div = _make_elementwise_binary_prim(
    "div",
    impl_aten=_div_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

eq = _make_elementwise_binary_prim(
    "eq",
    impl_aten=torch.eq,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

fmax = _make_elementwise_binary_prim(
    "fmax",
    impl_aten=torch.fmax,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

fmin = _make_elementwise_binary_prim(
    "fmin",
    impl_aten=torch.fmin,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

fmod = _make_elementwise_binary_prim(
    "fmod",
    impl_aten=torch.fmod,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


gcd = _make_elementwise_binary_prim(
    "gcd",
    impl_aten=torch.gcd,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


ge = _make_elementwise_binary_prim(
    "ge",
    impl_aten=torch.ge,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

gt = _make_elementwise_binary_prim(
    "gt",
    impl_aten=torch.gt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

hypot = _make_elementwise_binary_prim(
    "hypot",
    impl_aten=torch.hypot,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

igamma = _make_elementwise_binary_prim(
    "igamma",
    impl_aten=torch.special.gammainc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

igammac = _make_elementwise_binary_prim(
    "igammac",
    impl_aten=torch.special.gammaincc,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

le = _make_elementwise_binary_prim(
    "le",
    impl_aten=torch.le,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

lt = _make_elementwise_binary_prim(
    "lt",
    impl_aten=torch.lt,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)


# Note: the following impls are because torch.maximum and torch.minimum do not support scalar inputs
def _maximum_aten(
    a: TensorLikeType | NumberType, b: TensorLikeType | NumberType
) -> TensorLikeType:
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)

    return torch.maximum(a, b)  # type: ignore[arg-type]


maximum = _make_elementwise_binary_prim(
    "maximum",
    impl_aten=_maximum_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


def _minimum_aten(
    a: TensorLikeType | NumberType, b: TensorLikeType | NumberType
) -> TensorLikeType:
    if isinstance(a, TensorLike) and isinstance(b, Number):
        b = scalar_tensor(b, dtype=a.dtype, device=a.device)
    elif isinstance(b, TensorLike) and isinstance(a, Number):
        a = scalar_tensor(a, dtype=b.dtype, device=b.device)

    return torch.minimum(a, b)  # type: ignore[arg-type]


minimum = _make_elementwise_binary_prim(
    "minimum",
    impl_aten=_minimum_aten,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

mul = _make_elementwise_binary_prim(
    "mul",
    impl_aten=torch.mul,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

ne = _make_elementwise_binary_prim(
    "ne",
    impl_aten=torch.ne,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.ALWAYS_BOOL,
)

nextafter = _make_elementwise_binary_prim(
    "nextafter",
    impl_aten=torch.nextafter,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

pow = _make_elementwise_binary_prim(
    "pow",
    impl_aten=torch.pow,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

remainder = _make_elementwise_binary_prim(
    "remainder",
    impl_aten=torch.remainder,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


shift_left = _make_elementwise_binary_prim(
    "shift_left",
    impl_aten=torch.bitwise_left_shift,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

shift_right_arithmetic = _make_elementwise_binary_prim(
    "shift_right_arithmetic",
    impl_aten=torch.bitwise_right_shift,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

shift_right_logical = _not_impl

sub = _make_elementwise_binary_prim(
    "sub",
    impl_aten=torch.sub,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)

zeta = _make_elementwise_binary_prim(
    "zeta",
    impl_aten=torch.special.zeta,
    doc="",
    type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
)


#
# View operations
def _as_strided_meta(
    a: TensorLikeType, size: ShapeType, stride: StrideType, storage_offset: int
) -> TensorLikeType:
    if len(size) != len(stride):
        raise AssertionError(f"len(size)={len(size)} != len(stride)={len(stride)}")
    if storage_offset < 0:
        raise AssertionError(f"storage_offset must be >= 0, got {storage_offset}")
    utils.validate_strides(stride)
    utils.validate_shape(size)

    if reduce(operator.mul, size) == 0:
        # NOTE: This special case is to avoid having to acquire the storage below
        # as_strided to shapes with no elements are trivially valid, so it's OK
        pass
    elif isinstance(a, torch.Tensor):
        utils.check_in_bounds_for_storage(
            a._typed_storage(), size, stride, storage_offset
        )

    return torch.as_strided(a, size, stride, storage_offset)


def _as_strided_aten(
    a: Tensor, size: ShapeType, stride: StrideType, storage_offset: int
) -> Tensor:
    return torch.as_strided(a, size, stride, storage_offset)


_as_strided_doc = """
    Creates a view of the tensor with the given shape (size), strides (stride) and
    storage offset (storage_offset).
"""

as_strided = _make_prim(
    schema="as_strided(Tensor(a!) a, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor(a!)",
    meta=_as_strided_meta,
    impl_aten=_as_strided_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_as_strided_doc,
)


def _broadcast_in_dim_meta(
    a: TensorLikeType, shape: ShapeType, broadcast_dimensions: Sequence[int]
):
    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        sym_or,
    )

    # Type checks
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    if not isinstance(shape, Sequence):
        raise AssertionError(f"shape must be a Sequence, got {type(shape)}")
    if not isinstance(broadcast_dimensions, Sequence):
        raise AssertionError(
            f"broadcast_dimensions must be a Sequence, got {type(broadcast_dimensions)}"
        )

    # every dimension must be accounted for
    if a.ndim != len(broadcast_dimensions):
        raise AssertionError(
            f"a.ndim ({a.ndim}) != len(broadcast_dimensions) ({len(broadcast_dimensions)})"
        )

    # broadcast shape must have weakly more dimensions
    if len(shape) < a.ndim:
        raise AssertionError(f"len(shape) ({len(shape)}) must be >= a.ndim ({a.ndim})")

    # broadcast_dimensions must be an ascending sequence
    # (no relative reordering of dims) of integers and
    # each dimension must be within the new shape
    def _greater_than_reduce(acc, x):
        if not isinstance(x, Dim):
            raise AssertionError(
                f"broadcast_dimensions element must be Dim, got {type(x)}"
            )
        if x <= acc:
            raise AssertionError(
                f"broadcast_dimensions must be strictly ascending: {x} <= {acc}"
            )
        if x >= len(shape):
            raise AssertionError(
                f"broadcast_dimension {x} out of bounds for shape of length {len(shape)}"
            )

        return x

    reduce(_greater_than_reduce, broadcast_dimensions, -1)

    # shape must be broadcastable to
    for idx, new_idx in enumerate(broadcast_dimensions):
        torch._check(
            sym_or(a.shape[idx] == 1, shape[new_idx] == a.shape[idx]),
            lambda: f"{a.shape[idx]} must be broadcastable to {shape[new_idx]}",
        )

    new_strides = []
    original_idx = 0
    for idx in range(len(shape)):
        if idx in broadcast_dimensions:
            # Assigns a stride of zero to dimensions
            # which were actually broadcast
            if guard_or_false(a.shape[original_idx] == 1):
                if guard_or_false(a.shape[original_idx] == shape[idx]):
                    new_strides.append(a.stride()[original_idx])
                else:
                    new_strides.append(0)
            else:
                torch._check(
                    a.shape[original_idx] == shape[idx],
                    lambda: f"non-broadcasting semantics require {a.shape[original_idx]} == {shape[idx]}",
                )
                new_strides.append(a.stride()[original_idx])
            original_idx = original_idx + 1
        else:
            if guard_or_true(shape[idx] != 1):
                # consistent with previous use of guard_size_oblivious
                new_strides.append(0)
            elif original_idx == a.ndim:
                new_strides.append(1)
            else:
                new_strides.append(a.stride()[original_idx] * a.size()[original_idx])

    return a.as_strided(shape, new_strides, a.storage_offset())


def _broadcast_in_dim_aten(a, shape, broadcast_dimensions):
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1

    v = a
    for idx, x in enumerate(s):
        if x != -1:
            v = v.unsqueeze(idx)

    return v.expand(shape)


_broadcast_in_dim_doc = """
  Creates a view of a with the specified shape.

  Allows adding dimensions of any length and broadcasting
  dimensions of length one in a to any length.

  The location of the broadcast dimensions must be specified
  using the broadcast_dimensions argument. Changing the
  relative order of dimensions is not supported.
  """

broadcast_in_dim = _make_prim(
    schema="broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)",
    meta=_broadcast_in_dim_meta,
    impl_aten=_broadcast_in_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_broadcast_in_dim_doc,
)


def _validate_collapse_args(a: Tensor, start: int, end: int) -> None:
    # Special-case for zero dimensional tensors
    ndim = max(1, a.dim())
    utils.validate_idx(ndim, start)
    utils.validate_idx(ndim, end)

    # Verifies end is strictly greater than start
    # (Collapse requires a non-empty interval)
    torch._check_value(
        end >= start,
        lambda: f"Attempting to collapse but end, {end}, is less than start, {start}!",
    )


def _collapsed_shape(shape: ShapeType, start: int, end: int) -> tuple[int, ...]:
    """
    Returns the shape of a with dims in [start, end) merged into a single dimension.
    """
    # Special-case for zero dimensional tensors
    shape = (1,) if len(shape) == 0 else tuple(shape)

    dim_length = 1
    for s in shape[start : end + 1]:
        dim_length = dim_length * s

    return shape[0:start] + (dim_length,) + shape[end + 1 :]


# If the collapse is invalid or cannot be determined (because of unbacked data)
# then `must_be_valid` determines the behavior:
#   None: return None, None.
#   str: Do a torch._check() to ensure the collapse is valid and if it isn't
#   then fail with the provided string.
def _collapse_view_helper(
    a: TensorLikeType, start: int, end: int, must_be_valid: str | None
) -> tuple[ShapeType | None, StrideType | None]:
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy

    from torch.fx.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
        sym_and,
        sym_or,
    )

    _validate_collapse_args(a, start, end)

    # Special-case for zero dimensional tensors
    if a.ndim == 0:
        shape = (1,)
        strides = (1,)
    else:
        shape = a.shape  # type: ignore[assignment]
        strides = a.stride()  # type: ignore[assignment]

    if a.ndim == 0 or (end == start):
        return shape, strides

    valid_op = True
    if guard_or_false(a.numel() != 0):
        for idx in range(end - 1, start - 1, -1):
            valid_op = sym_and(
                valid_op,
                sym_or(
                    shape[idx] == 1,
                    shape[idx + 1] == 1,
                    strides[idx] == strides[idx + 1] * shape[idx + 1],
                ),
            )  # type: ignore[assignment]

            # early exit if we already know its invalid.
            if guard_or_false(valid_op is False):
                break

    # for unbacked this becomes a runtime assertion.
    valid_op = sym_or(valid_op, a.numel() == 0)

    if must_be_valid:
        torch._check(valid_op, lambda: must_be_valid)
    else:
        if not guard_or_false(valid_op):
            return None, None

    # compute stride
    stride = strides[end]
    for idx in range(end - 1, start - 1, -1):
        if shape[idx] != 1:
            # TODO with unbacked we should really exclude when shape[idx] == 1
            # something like
            # min(stride[end], torch.ite(shape[x]!=1,stride[idx], inf), ...)
            stride = min(stride, strides[idx])

    # compute length
    length = shape[end]
    if guard_or_true(length != 0):
        for idx in range(end - 1, start - 1, -1):
            if guard_or_false(shape[idx] == 0):
                length = 0
                stride = 0
                break
            length = length * shape[idx]
    else:
        stride = 0

    new_shape = shape[:start] + (length,) + shape[end + 1 :]
    new_strides = strides[:start] + (stride,) + strides[end + 1 :]

    # NOTE: when the input has no elements it's restrided as if it were contiguous
    # except for unbacked.
    if guard_or_false(a.numel() == 0):
        new_strides = utils.make_contiguous_strides_for(new_shape)

    return new_shape, new_strides


def _collapse_view_meta(a: TensorLikeType, start: int, end: int) -> TensorLikeType:
    new_shape, new_strides = _collapse_view_helper(
        a, start, end, "Attempting to view a collapsed tensor, but no such view exists!"
    )
    if new_strides is None:
        raise AssertionError(
            "new_strides should not be None after _collapse_view_helper"
        )
    if new_shape is None:
        raise AssertionError("new_shape should not be None after _collapse_view_helper")
    return a.as_strided(new_shape, new_strides, a.storage_offset())


def _collapse_view_aten(a: Tensor, start: int, end: int) -> Tensor:
    new_shape = _collapsed_shape(a.shape, start, end)
    return a.view(new_shape)


_collapse_view_doc = """
  Creates a view of a with the dimensions between
  start (inclusive) and end (exclusive) merged into a
  single dimension.

  If it's not possible to take such a view then an error
  is thrown. See collapse instead.

  The dimensions can be merged if and only if
  they are all "nested" with each other. That is, they all
  have the property that

  stride[i] = stride[i+1] * shape[i+1]

  for all i in [start, end - 1).
  """

collapse_view = _make_prim(
    schema="collapse_view(Tensor(a) a, int start, int end) -> Tensor(a)",
    meta=_collapse_view_meta,
    impl_aten=_collapse_view_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_collapse_view_doc,
)


def _conj_meta(a: TensorLikeType) -> TensorLikeType:
    if not a.dtype.is_complex:
        raise RuntimeError("Expected complex dtype in prims.conj")
    out = a.as_strided(a.shape, a.stride(), a.storage_offset())
    torch._C._set_conj(out, not a.is_conj())
    return out


_conj_doc = """
Returns a conjugated view of the original tensor
"""

conj = _make_prim(
    schema="conj(Tensor(a) a) -> Tensor(a)",
    meta=_conj_meta,
    impl_aten=torch.conj,
    return_type=RETURN_TYPE.VIEW,
    doc=_conj_doc,
)


def expand_dims(
    a: TensorLikeType, dimensions: DimsSequenceType, ndim=None
) -> TensorLikeType:
    """
    Creates a view of a with a.ndim + len(dimensions) dimensions, with new
    dimensions of length one at the dimensions specified by dimensions.
    """
    if ndim is not None:
        # TODO: this is only here to support the unsqueeze ref
        dims = sorted(utils.canonicalize_dims(ndim, dimensions))  # type: ignore[arg-type]
    else:
        dims = sorted(utils.canonicalize_dims(a.ndim, dimensions))  # type: ignore[arg-type]
    if len(set(dims)) != len(dims):
        msg = f"Received duplicate dimensions to expand in {str(dimensions)}"
        raise ValueError(msg)

    new_shape = list(a.shape)
    for idx in dims:
        new_shape.insert(idx, 1)

    broadcast_dimensions = [
        idx for idx in range(len(new_shape)) if idx not in dimensions
    ]
    return broadcast_in_dim(a, new_shape, broadcast_dimensions)


def _split_dim_meta(a: TensorLikeType, dim: int, outer_length: int) -> TensorLikeType:
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    utils.validate_idx(a.ndim, dim)
    utils.validate_dim_length(outer_length)

    # Verifies the dim can be split with the specified lhs_length
    inner_length = a.shape[dim] // outer_length

    if (a.shape[dim] % outer_length) != 0:
        msg = (
            f"Attempting to split dimension of length {a.shape[dim]}, "
            f"but outer length of {outer_length} divides it with a remainder!"
        )
        raise ValueError(msg)

    new_shape: list[int] = []
    new_strides: list[int] = []
    for idx in range(a.ndim):
        if idx == dim:
            new_shape.extend((outer_length, inner_length))
            new_strides.extend((a.stride()[idx] * inner_length, a.stride()[idx]))
        else:
            new_shape.append(a.shape[idx])
            new_strides.append(a.stride()[idx])

    return a.as_strided(new_shape, new_strides, a.storage_offset())


def _split_dim_aten(a: Tensor, dim: int, outer_length: int) -> Tensor:
    inner_length = a.shape[dim] // outer_length
    new_shape = a.shape[0:dim] + (outer_length, inner_length) + a.shape[dim + 1 :]

    return a.view(new_shape)


_split_dim_doc = """
  Creates a view of a with the given dimension (of length l) split
  into two dimensions, with the outer of the two having
  length outer_length and the inner of the two having computed
  length inner_length such outer_length * inner_length = l.
  """

# TODO: consider renaming split_dim_view
split_dim = _make_prim(
    schema="split_dim(Tensor(a) a, int dim, SymInt outer_length) -> Tensor(a)",
    meta=_split_dim_meta,
    impl_aten=_split_dim_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_split_dim_doc,
)


# Note: allows dimensions to be specified redundantly
def _squeeze_meta(a: TensorLikeType, dimensions: Sequence) -> TensorLikeType:
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy

    for idx in dimensions:
        utils.validate_idx(a.ndim, idx)
        if a.shape[idx] != 1:
            raise AssertionError(
                f"Cannot squeeze dimension {idx} with size {a.shape[idx]} (must be 1)"
            )

    new_shape = []
    new_strides = []
    for idx in range(len(a.shape)):
        if idx in dimensions:
            continue

        new_shape.append(a.shape[idx])
        new_strides.append(a.stride()[idx])

    return a.as_strided(new_shape, new_strides, a.storage_offset())


_squeeze_doc = """
  Creates a view of the tensor with the specified dimensions removed.

  The removed dimensions must each have length one.
  """

squeeze = _make_prim(
    schema="squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)",
    meta=_squeeze_meta,
    impl_aten=torch.squeeze,
    return_type=RETURN_TYPE.VIEW,
    doc=_squeeze_doc,
)


def _transpose_meta(a: TensorLikeType, permutation: DimsSequenceType) -> TensorLikeType:
    if a.ndim != len(permutation):
        msg = f"Attempting to permute a tensor of rank {a.ndim}, but received a permutation of length {len(permutation)}!"
        raise ValueError(msg)

    if not utils.is_valid_permutation(a.ndim, permutation):
        msg = f"Received an invalid permutation, {permutation}!"
        raise ValueError(msg)

    new_shape = [0] * a.ndim
    new_strides = [0] * a.ndim
    for idx, dim in enumerate(permutation):
        new_shape[idx] = a.shape[dim]
        new_strides[idx] = a.stride()[dim]

    return a.as_strided(tuple(new_shape), tuple(new_strides), a.storage_offset())


def _transpose_aten(a: Tensor, permutation: DimsSequenceType) -> Tensor:
    return torch.permute(a, permutation)


_transpose_doc = """
    Creates a view of the tensor with its dimensions permuted.

    The length of the permutation must be the rank of the tensor,
    and each element of the permutation specifies the new order
    for the corresponding dimension.
    """

transpose = _make_prim(
    schema="transpose(Tensor(a) a, int[] permutation) -> Tensor(a)",
    meta=_transpose_meta,
    impl_aten=_transpose_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_transpose_doc,
)


def _view_of_meta(a: TensorLikeType) -> TensorLikeType:
    return a.as_strided(a.shape, a.stride(), a.storage_offset())


def _view_of_aten(a: Tensor) -> Tensor:
    return a.view(a.shape)


_view_of_doc = """
    Creates a view of the tensor.
    """

view_of = _make_prim(
    schema="view_of(Tensor(a) a) -> Tensor(a)",
    meta=_view_of_meta,
    impl_aten=_view_of_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_view_of_doc,
)


def _view_element_type_meta(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    return a.view(dtype)


def _view_element_type_aten(a: Tensor, dtype: torch.dtype) -> Tensor:
    return a.view(dtype)


_view_element_type_doc = """
    Creates a view of the tensor with a different dtype.
    """

view_element_type = _make_prim(
    schema="view_of_dtype(Tensor(a) a, ScalarType dtype) -> Tensor(a)",
    meta=_view_element_type_meta,
    impl_aten=_view_element_type_aten,
    return_type=RETURN_TYPE.VIEW,
    doc=_view_element_type_doc,
)

#
# Functionalized view mutations
#


def _as_strided_scatter_meta(
    input: TensorLikeType,
    src: TensorLikeType,
    size: ShapeType,
    stride: StrideType,
    storage_offset: int,
) -> TensorLikeType:
    utils.validate_shape(size)
    utils.validate_strides(stride)

    required_size = utils.compute_required_storage_length(size, stride, storage_offset)
    torch._check(
        input.numel() >= required_size,
        lambda: (
            f"as_strided_scatter: sizes {size}, strides {stride}, storage offset {storage_offset} "
            f" and itemsize {input.element_size()} requiring a storage size of "
            f"{required_size * input.element_size()} are out of bounds "
            f"for storage of size {input.numel() * input.element_size()}"
        ),
    )
    torch._check(
        utils.is_same_shape(src.shape, size),
        lambda: f"expected src to have a size equal to the slice of self. src size = {src.shape}, slice size = {size}",
    )

    return utils.clone_preserve_strides(input)


_as_strided_scatter_doc = """
    Creates a new tensor equivalent to ``out = input.clone()`` after mutation by
    ``out.as_strided(size, stride, storage_offset).copy_(src)``.
"""

as_strided_scatter = _make_prim(
    schema="as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor",
    meta=_as_strided_scatter_meta,
    impl_aten=torch.as_strided_scatter,
    return_type=RETURN_TYPE.NEW,
    doc=_as_strided_scatter_doc,
)


#
# Shape operations
#


def _collapse_meta(a: Tensor, start: int, end: int) -> Tensor:
    # Special-case for zero dimensional tensors
    _validate_collapse_args(a, start, end)
    new_shape = _collapsed_shape(a.shape, start, end)
    return a.new_empty(new_shape)


def _collapse_aten(a: Tensor, start: int, end: int) -> Tensor:
    new_shape = _collapsed_shape(a.shape, start, end)
    out = a.new_empty(new_shape)
    with torch.no_grad():
        out.view_as(a).copy_(a)
    return out


_collapse_doc = """
Collapse a span of neighboring dimensions into one.

See collapse_view for the corresponding view operation.
"""
collapse = _make_prim(
    schema="collapse(Tensor a, int start, int end) -> Tensor",
    meta=_collapse_meta,
    impl_aten=_collapse_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_collapse_doc,
)


# TODO: review stride logic
# NB: unlike torch.cat, this is more strict about empty tensors and dim is
# never negative
def _cat_meta(tensors: Sequence[TensorLikeType], dim: int) -> TensorLikeType:
    # Verifies same shape (except in the concat dimension)
    if dim < 0:
        raise AssertionError(f"dim must be non-negative, got {dim}")
    shape = tensors[0].shape
    sym_sum_args = []
    for tensor_idx, tensor in enumerate(tensors):
        if len(shape) != len(tensor.shape):
            raise AssertionError(
                f"All tensors must have the same number of dimensions. "
                f"Expected {len(shape)} but tensor {tensor_idx} has {len(tensor.shape)}"
            )
        for idx, (common_length, length) in enumerate(zip(shape, tensor.shape)):
            if idx == dim:
                sym_sum_args.append(length)
            else:
                torch._check(
                    length == common_length,
                    lambda: f"Sizes of tensors must match except in dimension {dim}. "
                    f"Expected {common_length} in dimension {idx} but got {length} for tensor number "
                    f"{tensor_idx} in the list",
                )

    new_shape = list(tensors[0].shape).copy()
    new_shape[dim] = torch.sym_sum(sym_sum_args)  # type: ignore[call-overload]
    return TensorMeta(
        tensors[0],
        shape=new_shape,
        strides=utils.make_contiguous_strides_for(new_shape),
    )


def _cat_aten(tensors: tuple[Tensor, ...] | list[Tensor], dim: int) -> Tensor:
    return torch.cat(tensors, dim)


_cat_doc = """
  Concatenates tensors along the specified dimension.

  The tensors' shapes must have the same rank and same length for other dimensions.
  """

cat = _make_prim(
    schema="cat(Tensor[] tensors, int dim) -> Tensor",
    meta=_cat_meta,
    impl_aten=_cat_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_cat_doc,
)


def _reshape_meta(a: TensorLikeType, shape: ShapeType):
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    utils.validate_shape(shape)

    # Validates the tensor and the requested shape have the
    # same number of elements
    numel = reduce(operator.mul, shape)
    if numel != a.numel():
        msg = f"Attempting to reshape a tensor with {a.numel()} elements to a shape with {numel} elements!"
        raise ValueError(msg)

    return TensorMeta(a, shape=shape, strides=utils.make_contiguous_strides_for(shape))


def _reshape_aten(a: Tensor, shape: ShapeType) -> Tensor:
    return a.reshape(shape).clone(memory_format=torch.contiguous_format)


_reshape_doc = """
  Creates a contiguous tensor with the specified shape
  containing a copy of the data in a.
  """
reshape = _make_prim(
    schema="reshape(Tensor a, SymInt[] shape) -> Tensor",
    meta=_reshape_meta,
    impl_aten=_reshape_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_reshape_doc,
)


def _rev_meta(a: TensorLikeType, dims: DimsSequenceType) -> TensorLikeType:
    utils.validate_dimension_indices(a.ndim, dims)
    return torch.empty_like(a, memory_format=torch.preserve_format)


_rev_doc = """
    Reverses the order of elements along the given dimensions.
    """

rev = _make_prim(
    schema="rev(Tensor a, int[] dims) -> Tensor",
    meta=_rev_meta,
    impl_aten=torch.flip,
    return_type=RETURN_TYPE.NEW,
    doc=_rev_doc,
)

#
# Conditional prims
#


def _where_meta(
    pred: TensorLikeType, a: TensorLikeType, b: TensorLikeType
) -> TensorLikeType:
    return _prim_elementwise_meta(
        a,
        b,
        type_promotion=ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND.DEFAULT,
        args_with_fixed_dtypes=(pred,),
    )


_where_doc = """
  Selects elements from a and b according to pred.

  Where pred is true the result contains the element from a, and
  where pred is false the result contains the element from b.
  """

where = _make_prim(
    schema="where(Tensor pred, Tensor a, Tensor b) -> Tensor",
    meta=_where_meta,
    impl_aten=torch.where,
    return_type=RETURN_TYPE.NEW,
    doc=_where_doc,
)


#
# Type conversions
#
def _convert_element_type_meta(a: TensorLikeType, dtype: torch.dtype) -> TensorLikeType:
    # Type checks
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    if not isinstance(dtype, torch.dtype):
        raise AssertionError(f"dtype must be torch.dtype, got {type(dtype)}")

    # dtype conversion preserves dense strides
    if torch._prims_common.is_non_overlapping_and_dense_or_false(a):
        strides = a.stride()
    else:
        strides = utils.compute_elementwise_output_strides(a)

    return TensorMeta(a, strides=strides, dtype=dtype)


def _convert_element_type_aten(a: Tensor, dtype: torch.dtype) -> Tensor:
    # Propagates requires grad when possible
    if not utils.is_grad_dtype(dtype):
        requires_grad = False
    else:
        # TODO: update meta objects so this can be acquired directly
        try:
            requires_grad = a.requires_grad
        except Exception:
            requires_grad = False

    result = torch.empty_like(
        a, device=a.device, dtype=dtype, requires_grad=requires_grad
    )
    with torch.no_grad():
        return copy_to(result, a)


_convert_element_type_doc = """
  Creates a copy of a tensor with the given dtype.
  """

convert_element_type = _make_prim(
    schema="convert_element_type(Tensor a, ScalarType dtype) -> Tensor",
    meta=_convert_element_type_meta,
    impl_aten=_convert_element_type_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_convert_element_type_doc,
    tags=(torch.Tag.pointwise,),
)


def _device_put_meta(
    a: TensorLikeType, device: str | torch.device, non_blocking=False
) -> TensorLikeType:
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    if not isinstance(device, (str, torch.device)):
        raise AssertionError(f"device must be str or torch.device, got {type(device)}")
    if not isinstance(non_blocking, bool):
        raise AssertionError(f"non_blocking must be bool, got {type(non_blocking)}")

    return TensorMeta(a, device=utils.canonicalize_device(device))


def _device_put_aten(
    a: Tensor, device: str | torch.device, non_blocking=False
) -> Tensor:
    return a.to(device, non_blocking=non_blocking)


_device_put_doc = """
  Creates a copy of a tensor on the given device.
  """

device_put = _make_prim(
    schema="device_put(Tensor a, Device device, bool non_blocking=False) -> Tensor",
    meta=_device_put_meta,
    impl_aten=_device_put_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_device_put_doc,
)


# NOTE: need to model meta scalars
# See https://github.com/pytorch/pytorch/issues/78070
def _item_meta(a: TensorLikeType) -> FakeTensor:
    number_type = utils.dtype_to_type(a.dtype)
    return TensorMeta(number_type(-1))


_item_doc = """
    Converts a tensor with one element to a Python number.
"""


# We can't call into python dispatcher for item again
# because the current prim decomp calls into python dispatcher
# again. https://github.com/pytorch/pytorch/issues/136050
def _item_aten_no_python_dispatcher(*args, **kwargs):
    with torch._dispatch.python.no_python_dispatcher():
        return torch.Tensor.item(*args, **kwargs)


# TODO: create a new return type for scalars?
# FIXME: currently returns integers for boolean tensors
# https://github.com/pytorch/pytorch/issues/78071
item = _make_prim(
    schema="item(Tensor a) -> Scalar",
    meta=_item_meta,
    impl_aten=_item_aten_no_python_dispatcher,
    return_type=RETURN_TYPE.NEW,
    doc=_item_doc,
)


# NOTE: need to model meta scalars
# See https://github.com/pytorch/pytorch/issues/78070
def _maximum_value_meta(dtype: torch.dtype) -> FakeTensor:
    number_type = utils.dtype_to_type(dtype)
    return TensorMeta(number_type(-1))


def _maximum_value_aten(dtype: torch.dtype):
    if dtype == torch.bool:
        return True
    elif dtype.is_complex or dtype.is_floating_point:
        return torch.finfo(dtype).max
    else:
        return torch.iinfo(dtype).max


_maximum_value_doc = """
    Return the maximum finite value for a dtype.
"""

# TODO: create a new return type for scalars?
# FIXME: currently returns integers for boolean tensors
# https://github.com/pytorch/pytorch/issues/78071
maximum_value = _make_prim(
    schema="maximum_value(ScalarType dtype) -> Scalar",
    meta=_maximum_value_meta,
    impl_aten=_maximum_value_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_maximum_value_doc,
)


# NOTE: need to model meta scalars
# See https://github.com/pytorch/pytorch/issues/78070
def _minimum_value_meta(dtype: torch.dtype) -> FakeTensor:
    number_type = utils.dtype_to_type(dtype)
    return TensorMeta(number_type(-1))


def _minimum_value_aten(dtype: torch.dtype):
    if dtype == torch.bool:
        return False
    elif dtype.is_complex or dtype.is_floating_point:
        return torch.finfo(dtype).min
    else:
        return torch.iinfo(dtype).min


_minimum_value_doc = """
    Return the minimum finite value for a dtype.
"""

# TODO: create a new return type for scalars?
# FIXME: currently returns integers for boolean tensors
# https://github.com/pytorch/pytorch/issues/78071
minimum_value = _make_prim(
    schema="minimum_value(ScalarType dtype) -> Scalar",
    meta=_minimum_value_meta,
    impl_aten=_minimum_value_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_minimum_value_doc,
)

#
# Inplace operators
#


def _copy_to_meta(a: TensorLikeType, b: TensorLikeType):
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    if not isinstance(b, TensorLike):
        raise AssertionError(f"b must be TensorLike, got {type(b)}")  # mypy

    # Validates the cast is safe
    # TODO: move this as an option on the reference
    # a_typ = utils.dtype_to_type(a.dtype)
    # b_typ = utils.dtype_to_type(b.dtype)
    # if a_typ is not utils.get_higher_type(a_typ, b_typ):
    #     raise RuntimeError(str(b.dtype), " can't be cast safely to ", str(a.dtype), "!")

    # Validates the tensors have the same number of elements
    if a.numel() != b.numel():
        msg = f"Attempting to copy {b.numel()} elements to a tensor with {a.numel()} elements!"
        raise RuntimeError(msg)

    return a


def _copy_to_aten(a: Tensor, b: Tensor) -> Tensor:
    return a.copy_(b)


_copy_to_doc = """
  Copies the data in b to a and returns the modified a.
  """

# TODO: Remove safe casting and implement on reference instead
copy_to = _make_prim(
    schema="copy_to(Tensor(a!) a, Tensor b) -> Tensor(a!)",
    meta=_copy_to_meta,
    impl_aten=_copy_to_aten,
    return_type=RETURN_TYPE.INPLACE,
    doc=_copy_to_doc,
    register_conj_neg_fallthrough=True,
)


def _copy_strided_meta(a: TensorLikeType, stride: ShapeType):
    if not isinstance(a, TensorLike):
        raise AssertionError(f"a must be TensorLike, got {type(a)}")  # mypy
    return torch.empty_strided(
        a.shape,
        stride,
        dtype=a.dtype,
        layout=a.layout,
        device=a.device,
        requires_grad=a.requires_grad,
    )


def _copy_strided_aten(a: Tensor, stride: ShapeType) -> Tensor:
    out = torch.empty_strided(
        a.size(),
        stride=stride,
        dtype=a.dtype,
        layout=a.layout,
        device=a.device,
        requires_grad=a.requires_grad,
    )
    out.copy_(a)
    return out


_copy_strided_doc = """
  Copies the data in a to a new tensor, the new tensor has same shape with a size, but has different stride.
  """


copy_strided = _make_prim(
    schema="copy_strided(Tensor a, SymInt[] stride) -> Tensor",
    meta=_copy_strided_meta,
    impl_aten=_copy_strided_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_copy_strided_doc,
)


def _resize_meta(a: TensorLikeType, shape: ShapeType):
    return a.resize_(shape)


def _resize_aten(a: Tensor, shape: ShapeType) -> Tensor:
    return a.resize_(shape)


_resize_doc = """
  Gives a tensor with no elements a new shape, returning the modified tensor.

  The tensor's strides are contiguous and its values are uninitialized.
  """

# TODO: review support arbitrary resizes
resize = _make_prim(
    schema="resize(Tensor(a!) a, SymInt[] shape) -> Tensor(a!)",
    meta=_resize_meta,
    impl_aten=_resize_aten,
    return_type=RETURN_TYPE.INPLACE,
    doc=_resize_doc,
)


def _reduction_meta(inp, dims, *, output_dtype=None):
    """
    Meta function for single output reduction operations
    Stride logic is incorrect
    """
    if not isinstance(inp, TensorLike):
        raise AssertionError(f"inp must be TensorLike, got {type(inp)}")  # mypy
    if output_dtype is None:
        output_dtype = inp.dtype
    output_shape = utils.compute_reduction_output_shape(inp.shape, dims)
    return TensorMeta(
        shape=output_shape,
        strides=utils.make_contiguous_strides_for(output_shape),
        dtype=output_dtype,
        device=inp.device,
    )


def _var_reduction_meta(inp, dims, correction):
    if utils.is_complex_dtype(inp.dtype):
        output_dtype = utils.corresponding_real_dtype(inp.dtype)
    else:
        output_dtype = inp.dtype
    return _reduction_meta(inp, dims, output_dtype=output_dtype)


_sum_doc = """
    Computes the sum of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_xor_sum_doc = """
    Computes the xor sum of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_prod_doc = """
    Computes the product of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_amax_doc = """
    Computes the maximum value of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_amin_doc = """
    Computes the minimum value of elements in the input tensor over the list of dimensions
    specified in the dim argument
    """
_var_doc = """
    Computes the biased variance of x over the list of dimensions specified in the dim argument
    """


def _make_reduction_prim(name: str, impl_aten, doc):
    """Creates a reduction prim."""
    return _make_prim(
        schema=f"{name}(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor",
        meta=_reduction_meta,
        impl_aten=impl_aten,
        return_type=RETURN_TYPE.NEW,
        doc=doc,
    )


def _make_var_reduction_prim(name: str, impl_aten, doc):
    """Creates a reduction prim."""
    return _make_prim(
        schema=f"{name}(Tensor inp, int[]? dims, float? correction=1, *, ScalarType? output_dtype=None) -> Tensor",
        meta=_var_reduction_meta,
        impl_aten=impl_aten,
        return_type=RETURN_TYPE.NEW,
        doc=doc,
    )


sum = _make_reduction_prim(
    name="sum",
    impl_aten=torch.sum,
    doc=_sum_doc,
)


def _xor_sum_aten(
    inp: TensorLikeType,
    dims: DimsSequenceType | None,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    raise NotImplementedError("xor_sum only implemented with inductor")


xor_sum = _make_reduction_prim(
    name="xor_sum",
    impl_aten=_xor_sum_aten,
    doc=_xor_sum_doc,
)


def _prod_aten(
    inp: TensorLikeType,
    dims: DimsSequenceType | None,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if dims is not None:
        if len(dims) == 0:
            return inp.clone()
        for d in sorted(dims, reverse=True):
            if d < 0:
                raise AssertionError(f"dimension must be non-negative, got {d}")
            inp = torch.prod(inp, d, dtype=dtype)
        return inp
    else:
        return torch.prod(inp, dtype=dtype)


prod = _make_reduction_prim(
    name="prod",
    impl_aten=_prod_aten,
    doc=_prod_doc,
)


# torch.var, but correction is not kwarg-only
def torch_var(input, dim=None, correction=1, **kwargs):
    return torch.var(input, dim=dim, correction=correction, **kwargs)


var = _make_var_reduction_prim(
    name="var",
    impl_aten=torch_var,
    doc=_var_doc,
)

amax = _make_reduction_prim(
    name="amax",
    impl_aten=torch.amax,
    doc=_amax_doc,
)

amin = _make_reduction_prim(
    name="amin",
    impl_aten=torch.amin,
    doc=_amin_doc,
)


_iota_doc = """
    Constructs a 1-D tensor t where ``t[i] == start + i * step``.
"""


# TODO: layout, pin_memory, memory_format
# TODO: model requires_grad on TensorMeta
def _iota_meta(
    length: int,
    *,
    start: int,
    step: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    torch._check(
        utils.is_integer_dtype(dtype),
        lambda: "prims.iota only supports integer dtypes",
    )
    torch._check(step != 0, lambda: "step must be nonzero")
    return torch.empty(
        length,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def _iota_aten(
    length: int,
    *,
    start: int,
    step: int,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    end = start + length * step
    return torch.arange(
        start, end, step, dtype=dtype, device=device, requires_grad=requires_grad
    )


iota = _make_prim(
    schema="iota(SymInt length, *, SymInt start, SymInt step, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_iota_meta,
    impl_aten=_iota_aten,
    doc=_iota_doc,
)


# TODO: layout, pin_memory, memory_format
# TODO: model requires_grad on TensorMeta
def _empty_meta(
    shape: ShapeType, *, dtype: torch.dtype, device: torch.device, requires_grad: bool
) -> TensorLikeType:
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _empty_aten(
    shape: ShapeType, *, dtype: torch.dtype, device: torch.device, requires_grad: bool
) -> Tensor:
    return torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)


_empty_doc = """
    Creates a tensor with uninitialized values and the specified shape, dtype, and device.
"""

empty = _make_prim(
    schema="empty(SymInt[] shape, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_empty_meta,
    impl_aten=_empty_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_empty_doc,
)


def _empty_strided_meta(
    shape: ShapeType,
    strides: StrideType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


_empty_strided_doc = """
    Creates a tensor with uninitialized values.
"""

# TODO: add layout, pin_memory
empty_strided = _make_prim(
    schema="empty_strided(SymInt[] shape, SymInt[] strides, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_empty_strided_meta,
    impl_aten=torch.empty_strided,
    doc=_empty_strided_doc,
)


def _empty_permuted_meta(
    shape: ShapeType,
    physical_layout: DimsSequenceType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    p_strides = utils.make_contiguous_strides_for([shape[l] for l in physical_layout])
    dim = len(shape)
    torch._check(
        len(physical_layout) == dim,
        lambda: (
            "Number of dimensions in the tensor input does not match the "
            f"length of the physical layout; i.e. len(size) = {dim} "
            f"is not equal to len(physical_layout) = {len(physical_layout)}"
        ),
    )
    strides = [0] * len(shape)
    seen_dims = set()
    for p, l in enumerate(physical_layout):
        torch._check(
            0 <= l < dim,
            lambda: (
                f"Dimension out of range (expected to be between 0 and {dim - 1}, but got "
                f"{l} at index {p}).  NB: negative dims "
                "not currently supported; file an issue if you want it."
            ),
        )
        torch._check(l not in seen_dims, lambda: "Duplicate dim not allowed")
        strides[l] = p_strides[p]
        seen_dims.add(l)
    return TensorMeta(
        shape=shape,
        strides=strides,
        dtype=dtype,
        device=device,
    )


_empty_permuted_doc = """
    Creates a tensor with uninitialized values according to some physical layout,
    that is guaranteed to be non-overlapping and dense.
"""

# TODO: add layout, pin_memory
empty_permuted = _make_prim(
    schema="empty_permuted(SymInt[] shape, int[] physical_layout, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    return_type=RETURN_TYPE.NEW,
    meta=_empty_permuted_meta,
    impl_aten=torch.empty_permuted,
    doc=_empty_permuted_doc,
)


def _full_meta(
    shape: ShapeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _full_aten(
    shape: ShapeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> Tensor:
    # Note that Mypy thinks torch.full can't accept a complex fill_value
    return torch.full(
        shape,
        fill_value,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,  # type: ignore[arg-type]
    )


_full_doc = """
    Creates a tensor filled with the given fill value, and with the specified shape, dtype, and device.
"""

# TODO: add layout
full = _make_prim(
    schema="full(SymInt[] shape, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_full_meta,
    impl_aten=_full_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_full_doc,
)


def _full_like_meta(
    a: TensorLikeType,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> TensorLikeType:
    strides = utils.compute_elementwise_output_strides(a)
    if a.numel() == 0:
        strides = a.stride()

    return TensorMeta(a, strides=strides, dtype=dtype, device=device)


def _full_like_aten(
    a: Tensor,
    fill_value: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> Tensor:
    # Note that Mypy thinks torch.full can't accept a complex fill_value
    return torch.full_like(
        a,
        fill_value,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,  # type: ignore[arg-type]
    )


_full_like_doc = """
    Creates a tensor filled with the given fill value, and the same shape, dtype, and device as the
    given tensor by default. The dtype and device settings can be overridden
    by specifying them explicitly.
"""

full_like = _make_prim(
    schema="full_like(Tensor a, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor",
    meta=_full_like_meta,
    impl_aten=_full_like_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_full_like_doc,
)


def _scalar_tensor_meta(
    scalar: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> TensorLikeType:
    shape: ShapeType = []
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(scalar, shape=shape, strides=strides, dtype=dtype, device=device)


def _scalar_tensor_aten(
    scalar: NumberType,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if isinstance(scalar, complex) and (
        dtype is None or not utils.is_complex_dtype(dtype)
    ):
        raise TypeError("Complex scalar requires complex tensor dtype.")
    # Note that Mypy thinks torch.scalar can't accept a complex scalar
    return torch.scalar_tensor(scalar, dtype=dtype, device=device)  # type: ignore[arg-type]


_scalar_tensor_doc = """
    Wraps a Number into a Tensor with the specified dtype and device.
"""

# TODO: add layout and pin_memory support
scalar_tensor = _make_prim(
    schema="scalar_tensor(Scalar s, *, ScalarType? dtype=None, Device? device=None) -> Tensor",
    meta=_scalar_tensor_meta,
    impl_aten=_scalar_tensor_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_scalar_tensor_doc,
)


#
# Linear algebra (linalg) prims
#


def _svd_meta(
    A: TensorLikeType, *, full_matrices: bool
) -> tuple[TensorLikeType, TensorLikeType, TensorLikeType]:
    utils.check_is_matrix(A, "linalg.svd")
    utils.check_fp_or_complex(A.dtype, "linalg.svd", allow_low_precision_dtypes=False)

    A_shape = A.shape
    batch = A_shape[:-2]
    m, n = A_shape[-2:]
    k = min(m, n)

    shape_U = batch + (m, m if full_matrices else k)
    strides_U = utils.make_contiguous_strides_for(shape_U, row_major=False)
    U = TensorMeta(shape=shape_U, strides=strides_U, dtype=A.dtype, device=A.device)

    shape_S = batch + (k,)
    strides_S = utils.make_contiguous_strides_for(shape_S)
    S = TensorMeta(
        shape=shape_S,
        strides=strides_S,
        dtype=utils.corresponding_real_dtype(A.dtype) if A.is_complex() else A.dtype,
        device=A.device,
    )

    shape_Vh = batch + (n if full_matrices else k, n)
    # The CPU backend returns V, but the cuSolver backend returns V^H
    # TODO The MAGMA backend returns V, so this is wrong if used with the MAGMA backend
    is_cuda = A.device.type == "cuda"
    strides_Vh = utils.make_contiguous_strides_for(shape_Vh, row_major=is_cuda)
    Vh = TensorMeta(shape=shape_Vh, strides=strides_Vh, dtype=A.dtype, device=A.device)
    # Also makes sure this is CUDA or HIP:
    # https://pytorch.org/docs/stable/notes/hip.html#checking-for-hip
    if A.numel() != 0 and Vh.is_complex() and torch.cuda.is_available():
        Vh = Vh.conj()
    return U, S, Vh


def _svd_aten(
    A: TensorLikeType, *, full_matrices: bool
) -> tuple[Tensor, Tensor, Tensor]:
    return torch.linalg.svd(A, full_matrices=full_matrices)


_svd_doc = """
    Returns the SVD of a matrix or batch of matrices.

    The `full_matrices` flag controls whether the full or reduced SVD decomposition is returned.
"""

svd = _make_prim(
    schema="svd(Tensor A, *, bool full_matrices) -> (Tensor U, Tensor S, Tensor Vh)",
    meta=_svd_meta,
    impl_aten=_svd_aten,
    return_type=(RETURN_TYPE.NEW, RETURN_TYPE.NEW, RETURN_TYPE.NEW),
    doc=_svd_doc,
)


#
# Randomness Prims
#


def _normal_meta(
    shape: ShapeType,
    *,
    mean: float | complex,
    std: float,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    generator: torch.Generator | None = None,
) -> TensorLikeType:
    torch._check(
        std >= 0.0,
        lambda: f"expected non-negative standard deviation, but got std={std}",
    )

    torch._check(
        utils.is_float_dtype(dtype) or utils.is_complex_dtype(dtype),
        lambda: f"expected a floating-point or complex dtype, but got dtype={dtype}",
    )

    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=device)


def _normal_aten(
    shape: ShapeType,
    *,
    mean: float | complex,
    std: float,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
    generator: torch.Generator | None = None,
) -> Tensor:
    a = torch.empty(shape, dtype=dtype, device=device, requires_grad=requires_grad)
    with torch.no_grad():
        # NOTE: normal_ is incorrectly annotated to expect mean to be a float
        a.normal_(mean, std, generator=generator)  # type: ignore[arg-type]
    return a


_normal_doc = """
    Constructs a tensor filled with values drawn from a normal distribution with the specified mean
    and standard deviation.

    Only supports floating-point types.
"""

normal = _make_prim(
    schema=(
        "normal(SymInt[] shape, *, Scalar mean, Scalar std, ScalarType dtype, Device device, bool requires_grad, Generator? generator=None) -> Tensor"
    ),
    return_type=RETURN_TYPE.NEW,
    meta=_normal_meta,
    impl_aten=_normal_aten,
    doc=_normal_doc,
)


def _uniform_meta(
    shape: ShapeType,
    *,
    low: float,
    high: float,
    dtype: torch.dtype,
    device: torch.device,
    stride: ShapeType,
    generator: torch.Generator | None = None,
) -> TensorLikeType:
    return TensorMeta(shape=shape, strides=stride, dtype=dtype, device=device)


def _uniform_aten(
    shape: ShapeType,
    *,
    low: float,
    high: float,
    dtype: torch.dtype,
    device: torch.device,
    stride: ShapeType,
    generator: torch.Generator | None = None,
) -> Tensor:
    a = torch.empty_strided(shape, stride=stride, dtype=dtype, device=device)
    a.uniform_(low, high, generator=generator)
    return a


_uniform_doc = """
    Constructs a tensor filled with values drawn uniformly from low to high.
"""

# TODO: we should more seriously review randomness modeling and prims
_uniform_helper = _make_prim(
    schema=(
        "uniform(SymInt[] shape, *, Scalar low, Scalar high, ScalarType dtype, "
        "Device device, SymInt[] stride, Generator? generator=None) -> Tensor"
    ),
    return_type=RETURN_TYPE.NEW,
    meta=_uniform_meta,
    impl_aten=_uniform_aten,
    doc=_uniform_doc,
)

#
# FFT prims
#


def _fft_r2c_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    onesided: bool,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)

    shape = list(input.shape)
    if onesided:
        last_dim = dim[-1]
        shape[last_dim] = shape[last_dim] // 2 + 1

    dtype = utils.corresponding_complex_dtype(input.dtype)
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=input.device)


def _fft_r2c_aten(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    onesided: bool,
) -> TensorLikeType:
    normalization = 0  # No normalization
    return torch._fft_r2c(input, dim, normalization, onesided)


_fft_r2c_doc = """
    Performs a real to complex Fast Fourier Transform
"""


fft_r2c = _make_prim(
    schema="fft_r2c(Tensor self, *, int[] dim, bool onesided) -> Tensor",
    meta=_fft_r2c_meta,
    impl_aten=_fft_r2c_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_fft_r2c_doc,
)


def _fft_c2c_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    forward: bool,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)

    shape = input.shape
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(
        shape=shape, strides=strides, dtype=input.dtype, device=input.device
    )


def _fft_c2c_aten(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    forward: bool,
) -> TensorLikeType:
    normalization = 0  # No normalization
    return torch._fft_c2c(input, dim, normalization, forward)


_fft_c2c_doc = """
    Performs either a Fast Fourier Transform, or its inverse
"""


fft_c2c = _make_prim(
    schema="fft_c2c(Tensor self, *, int[] dim, bool forward) -> Tensor",
    meta=_fft_c2c_meta,
    impl_aten=_fft_c2c_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_fft_c2c_doc,
)


def _fft_c2r_meta(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    last_dim_size: int,
) -> TensorLikeType:
    dim = utils.canonicalize_dims(input.ndim, dim)
    utils.validate_no_repeating_dims(dim)

    shape = list(input.shape)
    shape[dim[-1]] = last_dim_size
    dtype = utils.corresponding_real_dtype(input.dtype)
    strides = utils.make_contiguous_strides_for(shape)
    return TensorMeta(shape=shape, strides=strides, dtype=dtype, device=input.device)


def _fft_c2r_aten(
    input: TensorLike,
    *,
    dim: DimsSequenceType,
    last_dim_size: int,
) -> TensorLikeType:
    normalization = 0  # No normalization
    return torch._fft_c2r(input, dim, normalization, last_dim_size)


_fft_c2r_doc = """
    Performs a complex to real Inverse Fast Fourier Transform
"""


fft_c2r = _make_prim(
    schema="fft_c2r(Tensor self, *, int[] dim, SymInt last_dim_size) -> Tensor",
    meta=_fft_c2r_meta,
    impl_aten=_fft_c2r_aten,
    return_type=RETURN_TYPE.NEW,
    doc=_fft_c2r_doc,
)


def _frexp_meta(self: TensorLikeType) -> tuple[TensorLikeType, TensorLikeType]:
    torch._check(
        self.dtype.is_floating_point,
        lambda: "torch.frexp() only supports floating-point dtypes",
    )
    return torch.empty_like(self), torch.empty_like(self, dtype=torch.int32)


frexp = _make_prim(
    schema="frexp(Tensor self) -> (Tensor mantissa, Tensor exponent)",
    meta=_frexp_meta,
    return_type=(RETURN_TYPE.NEW, RETURN_TYPE.NEW),
    impl_aten=torch.frexp,
    doc="",
)


def _make_token_aten() -> TensorLikeType:
    return new_token_tensor()


_make_token = _make_prim(
    schema="_make_token() -> Tensor",
    meta=_make_token_aten,
    return_type=RETURN_TYPE.NEW,
    impl_aten=_make_token_aten,
    doc="Creates a token used for keeping track of side effects.",
)


def _sink_tokens_aten(tokens) -> None:
    pass


_sink_tokens = _make_prim(
    schema="_sink_tokens(Tensor[] tokens) -> ()",
    meta=_sink_tokens_aten,
    return_type=RETURN_TYPE.NONE,
    impl_aten=_sink_tokens_aten,
    doc="Sink all of the tokens which were previously used for keeping track of side effects.",
)

torch.fx.node.has_side_effect(_sink_tokens)


register_rng_prims()
register_debug_prims()
<|endoftext|># mypy: allow-untyped-defs
from typing import cast

import torch
import torch.utils._pytree as pytree
from torch import _prims
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented, register_fake
from torch._ops import HigherOrderOperator
from torch._prims_common import CUDARngStateHelper, make_contiguous_strides_for
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.types import _device, _dtype


def throw_on_non_cuda(device):
    raise RuntimeError(
        f"You are trying to functionalize a {device.type} RNG operator but {device.type} does not "
        f"use Philox/counter-based RNG. Therefore, functionalizing a {device.type} RNG operator is "
        "not supported. We are discussing the possibility of a Philox-based RNG implementation for CPU."
    )


def register_rng_prim(name, schema, impl_aten, impl_meta, doc, tags=None):
    rngprim_def = torch.library.custom_op(
        "rngprims::" + name, impl_aten, mutates_args=(), schema=schema
    )

    rngprim_def.register_fake(impl_meta)

    prim_packet = getattr(torch._ops.ops.rngprims, name)
    prim = prim_packet.default
    if tags:
        prim._tags = tags

    for p in (prim_packet, prim):
        p.__doc__ = doc
        p.return_type = torch._prims_common.RETURN_TYPE.NEW  # type: ignore[attr-defined]

        p.schema = name + schema
        p.impl_aten = impl_aten
        p.prim_meta_impl = impl_meta


# Philox rand offsets could be shared in future with other philox ops, so
# keeping these functions in global scope.
def philox_rand_offset_meta(
    shape: torch.Size,
):
    return _prims.TensorLike(torch.tensor(0, dtype=torch.int64))


def philox_rand_offset(
    shape: torch.Size,
):
    # For impl, look at the function calc_execution_policy in the file
    # aten/src/ATen/native/cuda/DistributionTemplates.h. The impl was copied at
    # commit hash 72aa0667bd16707d50eb8fa337092a1f5d11dfb6
    numel_scalar = 1
    for dim_size in shape:
        numel_scalar *= dim_size
    numel = torch.scalar_tensor(numel_scalar, dtype=torch.int64)

    block_size = 256
    unroll = 4
    curand4_engine_calls = 4
    device_property = torch.cuda.get_device_properties(torch.cuda.current_device())
    blocks_per_sm = device_property.max_threads_per_multi_processor // block_size
    num = cast(int, numel)
    grid_size = (num + block_size - 1) // block_size
    grid_size = min(grid_size, device_property.multi_processor_count * blocks_per_sm)
    return ((num - 1) // (block_size * grid_size * unroll) + 1) * curand4_engine_calls


def register_philox_rand():
    name = "philox_rand"
    schema = "(SymInt[] size, Tensor seed, Tensor offset, int[]? stride, Device? device=None, ScalarType? dtype=None) -> (Tensor, Tensor)"

    def _philox_rand_meta(
        shape: torch.Size,
        seed: torch.Tensor,
        offset: torch.Tensor,
        stride: tuple[int, ...] | None,
        device: _device,
        dtype: _dtype,
    ):
        # stride arg will be useful for distributed usecase. Currently, it's unused.
        if stride is not None:
            raise AssertionError(f"stride must be None, got {stride}")
        stride = make_contiguous_strides_for(shape)
        random_values = _prims.TensorMeta(
            shape=shape, strides=stride, dtype=dtype, device=device
        )
        offset = philox_rand_offset_meta(shape)
        return (random_values, offset)

    def _philox_rand(
        shape: torch.Size,
        seed: torch.Tensor,
        offset: torch.Tensor,
        stride: tuple[int, ...] | None,
        device: _device,
        dtype: _dtype,
    ):
        # stride arg will be useful for distributed usecase. Currently, it's unused.
        if stride is not None:
            raise AssertionError(f"stride must be None, got {stride}")
        if device.type == "cpu":
            devices = []
        else:
            devices = [device]

        if device.type != "cuda":
            raise throw_on_non_cuda(device)

        with torch.random.fork_rng(devices):
            CUDARngStateHelper.set_torch_state_tensor(seed, offset)
            random_values = torch.rand(shape, device=device, dtype=dtype)

        return random_values, philox_rand_offset(shape)

    register_rng_prim(
        name=name,
        schema=schema,
        impl_aten=_philox_rand,
        impl_meta=_philox_rand_meta,
        doc="Philox based stateless rand operator",
        tags=(torch.Tag.nondeterministic_seeded,),
    )


def get_device(args, kwargs):
    if kwargs.get("device"):
        device = kwargs.get("device")
        if isinstance(device, str):
            device = torch.device(device)
        return device.type

    for arg in args:
        if isinstance(arg, torch.Tensor):
            return arg.device.type
        if isinstance(arg, torch.device):
            return arg.type
    return None


def register_run_and_save_rng_state_op():
    class RunAndSaveRngState(HigherOrderOperator):
        def __init__(self):
            super().__init__("run_and_save_rng_state", cacheable=True)

        def __call__(self, op, *args, **kwargs):
            # pyrefly: ignore [missing-attribute]
            return super().__call__(op, *args, **kwargs)

    run_and_save_rng_state = RunAndSaveRngState()

    run_and_save_rng_state.py_impl(DispatchKey.Autograd)(
        autograd_not_implemented(run_and_save_rng_state, deferred_error=True)
    )

    @run_and_save_rng_state.py_impl(DispatchKey.CUDA)
    def impl_cuda(op, *args, **kwargs):
        return torch.cuda.get_rng_state(), op(*args, **kwargs)

    @run_and_save_rng_state.py_impl(DispatchKey.CPU)
    def impl_cpu(op, *args, **kwargs):
        return torch.get_rng_state(), op(*args, **kwargs)

    @run_and_save_rng_state.py_impl(DispatchKey.HPU)
    def impl_hpu(op, *args, **kwargs):
        if hasattr(torch, "hpu"):
            return torch.hpu.get_rng_state(), op(*args, **kwargs)
        raise RuntimeError("functionalize a hpu RNG operator is not supported.")

    @run_and_save_rng_state.py_impl(DispatchKey.XPU)
    def impl_xpu(op, *args, **kwargs):
        return torch.xpu.get_rng_state(), op(*args, **kwargs)

    @run_and_save_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(op, *args, **kwargs):
        impl_map = {
            "cuda": impl_cuda,
            "cpu": impl_cpu,
            "hpu": impl_hpu,
            "xpu": impl_xpu,
        }
        device = get_device(args, kwargs)
        if device not in impl_map:
            raise AssertionError(f"Backend not supported for {device}")
        impl = impl_map[device]
        return impl(op, *args, **kwargs)

    @register_fake(run_and_save_rng_state, skip_cache=True)
    def impl_fake_tensor_mode(op, *args, **kwargs):
        # Check device to call the right impl
        return impl_backend_select(op, *args, **kwargs)

    @run_and_save_rng_state.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(mode, op, *args, **kwargs):
        out = impl_backend_select(op, *args, **kwargs)
        proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (op, *args))
        proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
        out_proxy = mode.tracer.create_proxy(
            "call_function", run_and_save_rng_state, proxy_args, proxy_kwargs
        )
        return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)

    return run_and_save_rng_state


def register_run_with_rng_state_op():
    class RunWithRngState(HigherOrderOperator):
        def __init__(self):
            super().__init__("run_with_rng_state", cacheable=True)

        def __call__(self, rng_state, op, *args, **kwargs):
            # pyrefly: ignore [missing-attribute]
            return super().__call__(rng_state, op, *args, **kwargs)

    run_with_rng_state = RunWithRngState()

    run_with_rng_state.py_impl(DispatchKey.Autograd)(
        autograd_not_implemented(run_with_rng_state, deferred_error=True)
    )

    @run_with_rng_state.py_impl(DispatchKey.CUDA)
    def impl_cuda(rng_state, op, *args, **kwargs):
        current_state = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(rng_state.cpu())
        out = op(*args, **kwargs)
        torch.cuda.set_rng_state(current_state)
        return out

    @run_with_rng_state.py_impl(DispatchKey.CPU)
    def impl_cpu(rng_state, op, *args, **kwargs):
        current_state = torch.get_rng_state()
        torch.set_rng_state(rng_state)
        out = op(*args, **kwargs)
        torch.set_rng_state(current_state)
        return out

    @run_with_rng_state.py_impl(DispatchKey.HPU)
    def impl_hpu(rng_state, op, *args, **kwargs):
        if hasattr(torch, "hpu"):
            current_state = torch.hpu.get_rng_state()
            torch.hpu.set_rng_state(rng_state)
            out = op(*args, **kwargs)
            torch.hpu.set_rng_state(current_state)
            return out
        raise RuntimeError("functionalize a hpu RNG operator is not supported.")

    @run_with_rng_state.py_impl(DispatchKey.XPU)
    def impl_xpu(rng_state, op, *args, **kwargs):
        current_state = torch.xpu.get_rng_state()
        torch.xpu.set_rng_state(rng_state)
        out = op(*args, **kwargs)
        torch.xpu.set_rng_state(current_state)
        return out

    @run_with_rng_state.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(mode, rng_state, op, *args, **kwargs):
        # TODO: you don't need to do this, the dispatch here already disabled
        # it
        with disable_proxy_modes_tracing():
            out = run_with_rng_state(rng_state, op, *args, **kwargs)
        proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (rng_state, op, *args))
        proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
        out_proxy = mode.tracer.create_proxy(
            "call_function", run_with_rng_state, proxy_args, proxy_kwargs
        )
        return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)

    @run_with_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(rng_state, op, *args, **kwargs):
        impl_map = {
            "cuda": impl_cuda,
            "cpu": impl_cpu,
            "hpu": impl_hpu,
            "xpu": impl_xpu,
        }
        device = get_device(args, kwargs)
        if device not in impl_map:
            raise AssertionError(f"Backend not supported for {device}")
        impl = impl_map[device]
        return impl(rng_state, op, *args, **kwargs)

    @register_fake(run_with_rng_state, skip_cache=True)
    def impl_fake_tensor_mode(rng_state, op, *args, **kwargs):
        # Skip setting the set_rng_state as it does not work well with fake tensors.
        # And it does not matter for the fake tensor mode.
        return op(*args, **kwargs)

    @run_with_rng_state.py_functionalize_impl
    def impl_functional(ctx, rng_state, op, *args, **kwargs):
        unwrapped_rng_state = ctx.unwrap_tensors(rng_state)
        unwrapped_args = ctx.unwrap_tensors(args)
        unwrapped_kwargs = ctx.unwrap_tensors(kwargs)

        with ctx.redispatch_to_next():
            out = run_with_rng_state(
                unwrapped_rng_state, op, *unwrapped_args, **unwrapped_kwargs
            )
            return ctx.wrap_tensors(out)

    return run_with_rng_state


run_and_save_rng_state = register_run_and_save_rng_state_op()
run_with_rng_state = register_run_with_rng_state_op()


def _impl_graphsafe_rng(op, *args, rng_state=None, **kwargs):
    # pyrefly: ignore [missing-attribute]
    device_idx = rng_state.device.index
    device_type = rng_state.device.type  # pyrefly: ignore [missing-attribute]
    device_mod = getattr(torch, device_type, None)
    if device_mod is None or not hasattr(device_mod, "default_generators"):
        raise AssertionError(
            f"GraphSafe RNG operations not supported for device '{device_type}' "
            f"(no torch.{device_type}.default_generators)"
        )
    generator = device_mod.default_generators[device_idx]
    current_state = generator.graphsafe_get_state()

    generator.graphsafe_set_state(rng_state)
    out = op(*args, **kwargs)
    generator.graphsafe_set_state(current_state)
    return out


def register_graphsafe_run_with_rng_state_op():
    class GraphSafeRunWithRngState(HigherOrderOperator):
        def __init__(self):
            super().__init__("graphsafe_run_with_rng_state", cacheable=True)

        def __call__(self, op, *args, rng_state=None, **kwargs):
            # pyrefly: ignore [missing-attribute]
            return super().__call__(op, *args, rng_state=rng_state, **kwargs)

    graphsafe_run_with_rng_state = GraphSafeRunWithRngState()

    graphsafe_run_with_rng_state.py_impl(DispatchKey.Autograd)(
        autograd_not_implemented(graphsafe_run_with_rng_state, deferred_error=True)
    )

    @graphsafe_run_with_rng_state.py_impl(DispatchKey.BackendSelect)
    def impl_backend_select(op, *args, rng_state=None, **kwargs):
        from torch._functorch._aot_autograd.utils import supports_graphsafe_rng

        device = get_device(args, kwargs)
        if device is None or not supports_graphsafe_rng(torch.device(device)):
            raise AssertionError(
                f"GraphSafe RNG operations not supported for device '{device}'. "
                f"Call register_graphsafe_rng_device_type('{device}') to register."
            )
        return _impl_graphsafe_rng(op, *args, rng_state=rng_state, **kwargs)

    @register_fake(graphsafe_run_with_rng_state, skip_cache=True)
    def impl_fake_tensor_mode(op, *args, rng_state=None, **kwargs):
        return op(*args, **kwargs)

    @graphsafe_run_with_rng_state.py_impl(ProxyTorchDispatchMode)
    def impl_proxy_dispatch_mode(mode, op, *args, rng_state=None, **kwargs):
        with disable_proxy_modes_tracing():
            out = graphsafe_run_with_rng_state(op, *args, rng_state=rng_state, **kwargs)
        proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, (op, *args))
        proxy_kwargs = pytree.tree_map(
            mode.tracer.unwrap_proxy, {"rng_state": rng_state, **kwargs}
        )
        out_proxy = mode.tracer.create_proxy(
            "call_function", graphsafe_run_with_rng_state, proxy_args, proxy_kwargs
        )
        return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)

    @graphsafe_run_with_rng_state.py_functionalize_impl
    def impl_functional(ctx, op, *args, rng_state=None, **kwargs):
        unwrapped_rng_state = (
            ctx.unwrap_tensors(rng_state) if rng_state is not None else None
        )
        unwrapped_args = ctx.unwrap_tensors(args)
        unwrapped_kwargs = ctx.unwrap_tensors(kwargs)

        with ctx.redispatch_to_next():
            out = graphsafe_run_with_rng_state(
                op, *unwrapped_args, rng_state=unwrapped_rng_state, **unwrapped_kwargs
            )
            return ctx.wrap_tensors(out)

    return graphsafe_run_with_rng_state


graphsafe_run_with_rng_state = register_graphsafe_run_with_rng_state_op()

_registered_graphsafe_dispatch_keys: set["DispatchKey"] = set()


def register_graphsafe_rng_dispatch(dispatch_key: "DispatchKey") -> None:
    """Register graphsafe_run_with_rng_state py_impl for a dispatch key."""
    if dispatch_key in _registered_graphsafe_dispatch_keys:
        return
    _registered_graphsafe_dispatch_keys.add(dispatch_key)
    graphsafe_run_with_rng_state.py_impl(dispatch_key)(_impl_graphsafe_rng)


# Register CUDA as a graphsafe RNG device by default
register_graphsafe_rng_dispatch(DispatchKey.CUDA)


# Late-bind CustomClassBaseMeta as Generator's metaclass. This is done here
# rather than in THPGenerator_init (C++) to avoid making torch._C depend
# on torch._custom_class_base at init time.
from torch._custom_class_base import CustomClassBaseMeta


torch._C._set_generator_metaclass(CustomClassBaseMeta)

torch._library.opaque_object.register_custom_class(
    torch._C.Generator,
    typ="symbolic",
    guard_fn=lambda gen: [gen.device],
    members={
        "device": torch._library.opaque_object.MemberType.USE_REAL,
        "__eq__": torch._library.opaque_object.MemberType.USE_REAL,
        "__ne__": torch._library.opaque_object.MemberType.USE_REAL,
    },
)


def register_run_dtensor_rng_op():
    """
    Register a higher-order operator for DTensor distributed random operations.
    Takes pre-computed integer offsets (start_offset_incr, end_offset_incr), an op,
    and args. Internally adjusts the RNG state using the offsets before/after
    running the op.

    The offsets are computed at trace time from the DTensorSpec, so the compiled graph
