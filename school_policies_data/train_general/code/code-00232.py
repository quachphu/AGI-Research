                    replacement_proxy = replacement_op(*args, **node.kwargs)

                    if (
                        compute_dtype != node.meta["val"].dtype
                        and not is_inplace_method
                    ):
                        replacement_proxy = (
                            torch.ops.prims.convert_element_type.default(
                                replacement_proxy,
                                node.meta["val"].dtype,
                            )
                        )

                    node.replace_all_uses_with(replacement_proxy.node)

                    graph.erase_node(node)

                    metrics_context = get_metrics_context()
                    if metrics_context.in_progress():
                        metrics_context.set(
                            "tensorify_float_success", True, overwrite=True
                        )
            else:
                for a in node.args:
                    if (
                        isinstance(a, fx.Node)
                        and "val" in a.meta
                        and isinstance(zf := a.meta["val"], torch.SymFloat)
                    ):
                        failed_tensorify_ops.update(str(node.target))

                        log.info("Failed to tensorify %s", node.target)

    # Now do one more pass that specializes all symfloats we didn't manage
    # to tensorify away.
    for node in reversed(graph.nodes):
        if node.op == "output" or node.op == "placeholder":
            continue

        with graph.inserting_before(node):
            if len(node.users) == 0 and not node.is_impure():
                graph.erase_node(node)
                continue

            if isinstance(
                (val := node.meta.get("val")),
                (torch.SymFloat, torch.SymInt, torch.SymBool),
            ):
                if has_free_symbols(val.node.expr) and all(
                    symbol_is_type(s, SymT.FLOAT) for s in val.node.expr.free_symbols
                ):
                    # If all symbols are backed symfloats, we can just specialize the whole node
                    # and get more precise guards. eg.
                    #
                    # zf = a.item()
                    # zf2 = zf // 2
                    # op(.. zf2 ..)
                    #
                    # It's better to guard on zf // 2 == 2.0 than zf == 5.0

                    node.replace_all_uses_with(guard_scalar(val))
                    graph.erase_node(node)

    # Sometimes by the time we get to tensorify, there have already been
    # specializations, eg. in python_arg_parser.h. In these cases,
    # placeholder nodes no longer have a reference to their original
    # symfloat and thus we need to deduce specializations have happened
    # via shape_env.replacements. NB: there's an important invariant here
    # that symfloats keep consistent names across restarts.
    for k, v in shape_env.backed_var_to_val.items():
        if symbol_is_type(k, SymT.FLOAT) and isinstance(v, sympy.core.numbers.Float):
            name = str(k)
            if (
                not TensorifyState.should_specialize(name)
                and k not in tensorified_symbols
            ):
                TensorifyState.specialize(name)
                should_restart = True

    if should_restart:
        # Sledgehammer time. Restart dynamo analysis, keeping track of which input sources
        # are no longer needed and should be specialized. Restarting analysis is necessary
        # because we need to instruct Dynamo to NOT make these as inputs.
        metrics_context = get_metrics_context()
        if metrics_context.in_progress():
            metrics_context.set(
                "tensorify_float_failure", failed_tensorify_ops, overwrite=True
            )
            metrics_context.set("tensorify_float_success", True, overwrite=True)
        raise TensorifyScalarRestartAnalysis

    graph_code_log.debug(
        "%s", lazy_format_graph_code("tensorify_python_scalars", gm, colored=True)
    )
<|endoftext|>import collections
import heapq
import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.node import _get_qualified_name


__all__ = [
    "get_acc_ops_name",
    "get_node_target",
    "is_node_output_tensor",
    "FxNetAccFusionsFinder",
    "legalize_graph",
    "stable_topological_sort",
]

Tensors = tuple[torch.Tensor] | list[torch.Tensor]
TensorOrTensors = torch.Tensor | Tensors
NodeList = list[torch.fx.Node]
NodeSet = set[torch.fx.Node]
Names = list[str]
CALLABLE_NODE_OPS = {"call_module", "call_function", "call_method"}


@compatibility(is_backward_compatible=False)
def get_acc_ops_name(k: str | type) -> str:
    if isinstance(k, str):
        return k
    elif k.__module__ and "acc_ops" in k.__module__:
        return f"acc_ops.{k.__name__}"
    else:
        module = k.__module__.replace(
            "torch._ops", "torch.ops"
        )  # WAR for bug in how torch.ops assigns module
        return f"{module if module else ''}.{k.__name__}"


@compatibility(is_backward_compatible=False)
def get_node_target(
    submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
) -> str:
    """
    Given a `node` returns its target typename.

    For "call_method" node, return node.target which is the name of that method being called.
    This could potential lead to conflict but should be okay because normally it's on a tensor.

    For "call_function" node, return typename of node.target.

    For "call_module" node, return typename of the module that node.target point to.

    If seeing "_VariableFunctionsClass" in the target name string, it will be replaced by
    "torch". e.g. _VariableFunctionsClass.relu would become torch.relu.
    """

    if node.op not in CALLABLE_NODE_OPS:
        raise AssertionError(
            "Expect op types of "
            + ", ".join(CALLABLE_NODE_OPS)
            + f", but found {node.op}"
        )

    if node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError(f"Expected str target, got {type(node.target)}")
        submod = submodules[node.target]
        submod_type = getattr(submod, "_base_class_origin", type(submod))
        return get_acc_ops_name(submod_type)
    elif node.op == "call_function":
        target: Any = node.target
        return (
            f"acc_ops.{target.__name__}"
            if target.__module__ is not None and "acc_ops" in target.__module__
            else _get_qualified_name(target)
        )
    else:
        if not isinstance(node.target, str):
            raise AssertionError(f"Expected str target, got {type(node.target)}")
        return node.target


@compatibility(is_backward_compatible=False)
def is_node_output_tensor(node: torch.fx.Node) -> bool:
    """Checks if the node output produces a Tensor or not.

    NOTE: This requires to run `ShapeProp` on the containing fx graph before
    calling this function. This is because it works by checking the `type`
    metadata on the node. This metadata is produced by the `ShapeProp`.
    """
    type_ = node.meta.get("type", None)
    return type_ is not None and issubclass(type_, torch.Tensor)


@compatibility(is_backward_compatible=False)
class FxNetAccFusionsFinder:
    """
    Finds groups of connected ACC nodes that pass non-tensor data between each other.
    Such groups are called fusion groups.
    """

    def __init__(self, module: torch.fx.GraphModule, acc_nodes: NodeSet) -> None:
        self.module = module
        self.nodes = list(module.graph.nodes)
        self.acc_nodes = acc_nodes
        self.node_index = {node: i for i, node in enumerate(self.nodes)}

    @dataclass
    class FusionGroup:
        # The smallest idx of nodes in the fusion group after topological sorting all the nodes in the model.
        top_node_idx: int

        # Nodes in this fusion group.
        nodes: NodeSet

        # Inputs to this fusion group.
        inputs: NodeSet

        # Nodes that in the fusion group that haven't been processed yet.
        nodes_need_process: NodeSet

        def add_node(self, node: torch.fx.Node) -> None:
            """
            Add a node to fusion group.
            """
            if node in self.nodes:
                return

            self.nodes_need_process.add(node)
            self.nodes.add(node)
            self.inputs.discard(node)
            self.inputs.update(
                {
                    n
                    for n in node.all_input_nodes
                    if n.op in CALLABLE_NODE_OPS and n not in self.nodes
                }
            )

    def recursive_add_node(
        self,
        fusion_group: "FxNetAccFusionsFinder.FusionGroup",
        inputs: NodeSet | NodeList,
        visited: NodeSet | None = None,
    ) -> bool:
        """
        Start from inputs and going reverse topological order. If any upstream node
        is in the fusion group, add all the nodes in this path to fusion group.
        """
        for arg in inputs:
            # skip the node if already seen
            if visited is not None:
                if arg in visited:
                    continue
                visited.add(arg)

            # Skip placeholder and get_attr because they won't be in the fusion group.
            if arg.op not in CALLABLE_NODE_OPS:
                continue

            # If the node has smaller idx, it's already an upstream node of the fusion
            # group. We don't need to check it anymore.
            if self.node_index[arg] < fusion_group.top_node_idx:
                continue

            # If the node is in the fusion group, return True.
            if arg in fusion_group.nodes:
                return True

            # Check the upstream nodes of the node, if any of them is in the fusion group
            # we'll add this node to fusion group and return True.
            if self.recursive_add_node(fusion_group, arg.all_input_nodes, visited):
                fusion_group.add_node(arg)
                return True

        return False

    def __call__(self) -> dict[torch.fx.Node, NodeSet]:
        result: dict[torch.fx.Node, NodeSet] = {}
        acc_nodes = list(self.acc_nodes)

        for node in acc_nodes:
            if node in result:
                continue
            if node.op not in CALLABLE_NODE_OPS:
                continue
            if "tensor_meta" in node.meta:
                continue
            if node not in self.acc_nodes:
                continue

            fusion_group: FxNetAccFusionsFinder.FusionGroup = self.FusionGroup(
                top_node_idx=self.node_index[node],
                nodes={node},
                inputs=set(node.all_input_nodes),
                nodes_need_process={node},
            )
            while fusion_group.nodes_need_process:
                node = fusion_group.nodes_need_process.pop()
                self.recursive_add_node(
                    fusion_group,
                    fusion_group.inputs,
                    visited=set(),
                )

                # Optionally add downstream nodes
                if "tensor_meta" not in node.meta:
                    for user in node.users:
                        if user.op not in CALLABLE_NODE_OPS:
                            continue
                        if user in fusion_group.nodes:
                            continue

                        fusion_group.add_node(user)
                        self.recursive_add_node(
                            fusion_group,
                            fusion_group.inputs,
                            visited=set(),
                        )

                # Add some upstream nodes
                for arg in node.all_input_nodes:
                    if arg.op not in CALLABLE_NODE_OPS:
                        continue
                    if "tensor_meta" in arg.meta:
                        continue
                    if arg in fusion_group.nodes:
                        continue

                    fusion_group.add_node(arg)
                    fusion_group.top_node_idx = min(
                        fusion_group.top_node_idx, self.node_index[arg]
                    )
                    self.recursive_add_node(
                        fusion_group,
                        fusion_group.inputs,
                        visited=set(),
                    )

            if not (set(fusion_group.nodes) <= self.acc_nodes):
                self.acc_nodes -= fusion_group.nodes
            else:
                for n in fusion_group.nodes:
                    result[n] = fusion_group.nodes

        return result


@compatibility(is_backward_compatible=False)
def legalize_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Replace the graph of the given GraphModule with one that contains the same nodes as the
    original, but in topologically sorted order.

    This is used by the merge_matmul transformation below, which disturbs the topologically sorted
    order of its input GraphModule, so that this order is restored before further transformation.

    Arguments:
        gm: The graph module to topologically sort. It is modified in-place.

    Returns:
        The graph module in-place sorted

    Warning:
        This topological sort is NOT stable, it will NOT preserve the original node order.
        If you need a stable topological sort, use stable_topological_sort instead.
    """

    # These operators are used for making runtime assertions before any
    # data-dependent operators occur. We want to prioritize sorting these to
    # ensure that these assertions appear before any data-dependent operations
    # in the graph.
    PRIORITIZED_OPS = [
        operator.add,
        operator.mul,
        operator.sub,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.le,
        operator.lt,
        operator.ge,
        operator.gt,
        operator.eq,
        operator.ne,
        torch.ops.aten.sym_constrain_range.default,
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch.ops.aten._assert_async.msg,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten._assert_scalar.default,
    ]

    indeg = dict.fromkeys(gm.graph.nodes, 0)
    new_graph = torch.fx.Graph()
    # Track how many unfulfilled dependencies each node has
    for node in gm.graph.nodes:
        for user in node.users:
            indeg[user] += 1
    queue: collections.deque[torch.fx.Node] = collections.deque()
    # Add all nodes with no dependencies to the queue
    for node in gm.graph.nodes:
        if indeg[node] == 0:
            queue.append(node)
    env: dict[torch.fx.Node, torch.fx.Node] = {}
    # Pop nodes from the queue, and add nodes that have had all their
    # dependencies fulfilled
    while len(queue) > 0:
        cur = queue.popleft()
        env[cur] = new_graph.node_copy(cur, lambda x: env[x])
        for user in cur.users:
            indeg[user] -= 1
            if indeg[user] == 0:
                if user.op == "call_function" and user.target in PRIORITIZED_OPS:
                    queue.appendleft(user)
                else:
                    queue.append(user)
    # If the new graph's size is not as large as the old one, then there must be
    # a cycle (i.e. some node's dependencies were not satisfied.)
    if len(new_graph.nodes) < len(gm.graph.nodes):
        raise RuntimeError(
            f"Input graph has cycles, unable to add {[node for node in indeg if indeg[node] != 0]}"
        )
    new_graph._codegen = gm.graph._codegen
    gm.graph = new_graph
    return gm


@compatibility(is_backward_compatible=False)
def stable_topological_sort(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Replace the graph of the given GraphModule with one that contains the same nodes as the
    original, but in topologically sorted order while preserving the original node order
    as much as possible.

    This function performs a stable topological sort where nodes appear in an order that:
    1. Respects data dependencies (topological ordering)
    2. Preserves the original node order when there are no dependency constraints

    The algorithm uses Kahn's algorithm with a priority queue: nodes with all dependencies
    satisfied are added to a min-heap, ordered by their original position. This ensures
    we always process the earliest node in the original order among ready nodes.

    Arguments:
        gm: The graph module to topologically sort. It is modified in-place.

    Returns:
        The graph module in-place sorted
    """
    indeg = dict.fromkeys(gm.graph.nodes, 0)
    new_graph = torch.fx.Graph()

    # Build node to original index mapping
    node_to_id: dict[torch.fx.Node, int] = {
        node: idx for idx, node in enumerate(gm.graph.nodes)
    }

    # Track how many unfulfilled dependencies each node has
    for node in gm.graph.nodes:
        for user in node.users:
            indeg[user] += 1

    # Priority queue: (original_index, node)
    # Use min-heap to always process the node with smallest original index
    ready_queue: list[tuple[int, torch.fx.Node]] = []
    for node in gm.graph.nodes:
        if indeg[node] == 0:
            heapq.heappush(ready_queue, (node_to_id[node], node))

    env: dict[torch.fx.Node, torch.fx.Node] = {}

    # Process nodes
    while ready_queue:
        # Pop node with smallest original index
        _, cur = heapq.heappop(ready_queue)
        env[cur] = new_graph.node_copy(cur, lambda x: env[x])

        # Update in-degrees and add newly ready nodes
        for user in cur.users:
            indeg[user] -= 1
            if indeg[user] == 0:
                heapq.heappush(ready_queue, (node_to_id[user], user))

    # Check if all nodes were processed
    if len(new_graph.nodes) != len(gm.graph.nodes):
        raise AssertionError(
            f"Input graph has cycles, unable to add {[node for node in indeg if indeg[node] != 0]}"
        )

    new_graph._codegen = gm.graph._codegen
    gm.graph = new_graph
    return gm
<|endoftext|>import argparse
import copy
import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, IO, Literal, NamedTuple

import torch
from torch._logging import trace_structured
from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg
from torch.fx.passes.graph_manipulation import get_size_of_node

from .graph_drawer import FxGraphDrawer
from .operator_support import get_node_target, OperatorSupportBase
from .shape_prop import ShapeProp
from .split_utils import move_non_tensor_nodes_on_boundary, split_by_tags
from .tools_common import (
    CALLABLE_NODE_OPS,
    FxNetAccFusionsFinder,
    is_node_output_tensor,
    NodeList,
    NodeSet,
    Tensors,
)


__all__ = [
    "FxNetAccNodesFinder",
    "FxNetSplitterInternalError",
    "Subgraph",
    "SplitResult",
    "generate_inputs_for_submodules",
    "NodeEvent",
    "NodeEventTracker",
]

DEFAULT_MIN_ACC_MODULE_SIZE = 1
DEFAULT_SKIP_FUSION = False
DEFAULT_ALLOW_NON_TENSOR = False

# ENV var and constants for node tracker

TRACKER_DUMP_PATH = "_fx_net_tracker"
NODES_SUFFIX = "_nodes.txt"
ALL_SUFFIX = "_all.txt"

ENV_FX_NET_ACC_SPLITTER_TRACKER_MODE = "FX_NET_ACC_SPLITTER_TRACKER_MODE"
ENV_FX_NET_ACC_SPLITTER_TRACKER_DUMP_PATH = "FX_NET_ACC_SPLITTER_TRACKER_DUMP_PATH"
ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES = (
    "FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES"
)

DUMP_PREFIX = os.environ.get(
    ENV_FX_NET_ACC_SPLITTER_TRACKER_DUMP_PATH, TRACKER_DUMP_PATH
)

"""
Different modes of the event tracker for local debugging:
"0": No local dumps. Information available by setting breakpoints and visually inspect in pdb.
"1": Dump all events to DUMP_PREFIX_all.txt
"2": In addition to events dump, track nodes specified by ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES
     recursively and dump to DUMP_PREFIX_nodex.txt
"3": In addition to events dump, track all nodes with more than 1 event recursively and dump to DUMP_PREFIX_nodex.txt
In addition to the above local dumps, tracker is always enabled and dumps via trace_structured.
"""
# pyrefly: ignore [bad-assignment]
TRACKER_MODE: Literal["0", "1", "2", "3"] = os.environ.get(
    ENV_FX_NET_ACC_SPLITTER_TRACKER_MODE, "0"
)  # type: ignore[assignment]


class _SplitterSettingBase:
    def __init__(
        self,
        min_acc_module_size: int = DEFAULT_MIN_ACC_MODULE_SIZE,
        skip_fusion: bool = DEFAULT_SKIP_FUSION,
        allow_non_tensor: bool = DEFAULT_ALLOW_NON_TENSOR,
        max_acc_splits: int = -1,
        move_non_tensor_nodes_on_boundary: bool = False,
    ) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--min-acc-module-size",
            "--min_acc_module_size",
            required=False,
            type=int,
            help="Minimum size limit of an accelerator subgraph.",
        )
        parser.add_argument(
            "--max-acc-splits",
            "--max_acc_splits",
            required=False,
            type=int,
            help="Enforce a maximum number of split subgraphs.",
        )
        parser.add_argument(
            "--skip-fusion",
            "--skip_fusion",
            default=False,
            action="store_true",
            help="If true then no fusion groups. Fusion group is used to "
            "enforce no non-tensor data flow between submodules. If we don't "
            "have this constraint, setting this to false is recommended as it "
            "can reduce overhead.",
        )
        parser.add_argument(
            "--allow-non-tensor",
            "--allow_non_tensor",
            default=False,
            action="store_true",
            help="For some backends non-tensor data flow between cpu and them "
            "are not allowed. Therefore, if a node supported by accelerator but "
            "it has non-tensor inputs or outputs to a cpu node we would want to "
            "consider it as a cpu node during splitting. However, for some backends "
            "we might not care about non-tensor data flow and we can set this option "
            "to true to disable the functionality that prevent non-tensor data flow.",
        )
        parser.add_argument(
            "--move-non-tensor-nodes-on-boundary",
            "--move_non_tensor_nodes_on_boundary",
            required=False,
            action="store_true",
            help="AOTI does not support non-tensor nodes on acc->acc, acc->gpu and gpu->acc boundary. "
            "For non-tensor nodes on acc->acc boundary and acc->gpu, we move the nodes from upstream to downstream. "
            "For non-tensor nodes on gpu->acc boundary, it is handled by the pre-split process. "
            "(by method reduce_acc_nodes_non_tensor_input). ",
        )
        args, _unknown = parser.parse_known_args()

        self.min_acc_module_size: int = (
            args.min_acc_module_size
            if args.min_acc_module_size
            else min_acc_module_size
        )
        self.skip_fusion: bool = args.skip_fusion if args.skip_fusion else skip_fusion
        self.allow_non_tensor: bool = (
            args.allow_non_tensor if args.allow_non_tensor else allow_non_tensor
        )
        self.max_acc_splits: int = max_acc_splits
        self.move_non_tensor_nodes_on_boundary: bool = (
            args.move_non_tensor_nodes_on_boundary
            if args.move_non_tensor_nodes_on_boundary
            else move_non_tensor_nodes_on_boundary
        )


@compatibility(is_backward_compatible=False)
class NodeEvent:
    """
    An event in graph split that happened on a node.
    source: Subject of the event
    desc: readable description
    dep: Optional dependency, usually the node that caused the event.
    """

    def __init__(
        self, source: torch.fx.Node, desc: str, dep: torch.fx.Node | None = None
    ) -> None:
        self.source = source
        self.desc = desc
        self.dep = dep

    def to_str(self) -> str:
        # source: The name of the subject of the event.
        # desc: description of the event, in the format of <event_type>|<explanation>
        # dep: The name of the cause of this event, which is another node, or #
        # if it's caused by the subject node
        return f"{self.source.name}: {self.desc} {self.dep.name if self.dep else '#'}"


@compatibility(is_backward_compatible=False)
class NodeEventTracker:
    """
    Tracks node events during the splitter execution.
    """

    def __init__(self, tracker_mode: int, dump_prefix: str) -> None:
        self.tracker_mode = tracker_mode
        self.dump_prefix = dump_prefix
        # list of events
        self.events = []
        # dict from node name to event index
        self.node_events = {}
        self.writer = print

    def add(
        self, node: torch.fx.Node, desc: str, dep: torch.fx.Node | None = None
    ) -> None:
        """
        Add a new event to the tracker.
        """
        event = NodeEvent(node, desc, dep)
        self.events.append(event)
        if node.name not in self.node_events:
            self.node_events[node.name] = []
        self.node_events[node.name].append(len(self.events) - 1)

    def print_node(
        self,
        node_name: str,
        recursive: bool = False,
        tab: str = "",
        writer: Callable[[str], object] | None = None,
    ) -> None:
        """
        Print a node and its events.
        @param recursive: if True, print nodes that caused the events on this current node.
        @param tab: Indentation for dependencies.
        @param writer: function to write to file. If None, use print.
        """
        if not writer:
            writer = self.writer
        for idx in self.node_events.get(node_name, []):
            event = self.events[idx]
            writer(tab + event.to_str())
            if recursive and event.dep is not None:
                self.print_node(
                    event.dep.name, recursive=True, tab="| " + tab, writer=writer
                )

    def to_dict(self) -> dict[str, list[str]]:
        """
        Create dict dump on all events.
        """
        ret: dict[str, list[str]] = {}
        for name in self.node_events:
            ret[name] = []
            for idx in self.node_events.get(name, []):
                event = self.events[idx]
                ret[name].append(event.to_str())
        return ret

    def print_all(self, writer: Callable[[str], object] | None = None) -> None:
        """
        Print all nodes in a list.
        @param writer: function to write to file. If None, use print.
        """
        if not writer:
            writer = self.writer
        for name in self.node_events:
            writer(f"Node: {name}:")
            self.print_node(name, recursive=False, tab="  ", writer=writer)

    def dump(self) -> None:
        """
        Function to be invoked at the end of the finder execution to printout tracked events specified by the mode.
        """
        # dump via trace_structured
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_net_acc_splitter_finder_events",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(self.to_dict()),
        )

        def make_writer(f: IO[str]) -> Callable[[str], int]:
            def fn(x: str) -> int:
                return f.write(x + "\n")

            return fn

        # Mode 0: no local dump
        # Mode >=1: Dump all events to file
        if self.tracker_mode >= 1:
            with open(self.dump_prefix + ALL_SUFFIX, "w") as f:
                self.print_all(make_writer(f))

        def dump_selected_nodes(nodes: list[str]) -> None:
            with open(self.dump_prefix + NODES_SUFFIX, "w") as f:
                writer = make_writer(f)
                for node_name in nodes:
                    writer(f"===== Tracking node {node_name} =====")
                    self.print_node(node_name, recursive=True, tab="|-", writer=writer)
                    writer(f"===== End of tracking node {node_name} =====")

        # Mode 2: Dump specific nodes in recursive manner.
        # Mode 3: Dump all nodes with more than 1 event in recursive manner.
        if self.tracker_mode == 2 or self.tracker_mode == 3:
            nodes = (
                os.environ.get(ENV_FX_NET_ACC_SPLITTER_TRACKER_TRACKED_NODES, "").split(
                    ","
                )
                if self.tracker_mode == 2
                else [
                    name for name, events in self.node_events.items() if len(events) > 1
                ]
            )
            dump_selected_nodes(nodes)


@compatibility(is_backward_compatible=False)
class FxNetAccNodesFinder:
    """
    Finds a set of nodes that can be supported on ACC, excluding nodes that have non-tensor
    input/output to cpu nodes to prevent non-tensor data flow between backends and cpu.

    I.e. if we have a chain:

    ACC_NODE_1 -> ACC_NODE_2 -> ACC_NODE_3 -> CPU_NODE_1

    where every ACC node produces non-tensor output, then they all should be treated as CPU nodes.

    This behavior can be turned off by passing allow_non_tensor=True.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        operator_support: OperatorSupportBase,
        allow_non_tensor: bool,
    ) -> None:
        self.module = module
        self.operator_support = operator_support
        self.allow_non_tensor = allow_non_tensor
        self.acc_nodes: NodeSet = set()

        self.tracker = NodeEventTracker(int(TRACKER_MODE), DUMP_PREFIX)

    def reduce_acc_nodes_non_tensor_input_helper(self, cpu_worklist: NodeList) -> None:
        """
        Transitively excludes nodes from ACC supported set.
        For every node in the worklist:
        - removes its downstream ACC nodes from ACC supported set,
        - if any downstream ACC node produces non-tensor output,
          then it gets added into the worklist.
        """
        while cpu_worklist:
            node = cpu_worklist.pop(0)

            for user in node.users:
                if user in self.acc_nodes:
                    self.acc_nodes.remove(user)
                    self.tracker.add(user, "acc_del|user_of_new_cpu_node", node)
                    if not is_node_output_tensor(user):
                        self.tracker.add(user, "new_cpu_node|non_tensor_output")
                        cpu_worklist.append(user)

    def reduce_acc_nodes_non_tensor_input(self) -> None:
        """
        Excludes nodes from ACC supported set that have direct
        upstream CPU nodes that produce non-tensor outputs.
        """
        non_tensor_cpu_nodes: NodeList = []

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            if node in self.acc_nodes:
                continue
            if is_node_output_tensor(node):
                continue
            self.tracker.add(node, "new_cpu_node|callable_non_tensor_input")
            non_tensor_cpu_nodes.append(node)

        self.reduce_acc_nodes_non_tensor_input_helper(non_tensor_cpu_nodes)

    def reduce_acc_nodes_non_tensor_output(self) -> None:
        """
        Excludes nodes from ACC supported set that produce non-tensor
        outputs and have downstream CPU nodes.
        """
        while True:
            new_cpu_nodes: NodeList = []

            for acc_node in self.acc_nodes:
                if is_node_output_tensor(acc_node):
                    continue
                for user in acc_node.users:
                    if user not in self.acc_nodes:
                        new_cpu_nodes.append(acc_node)
                        self.tracker.add(
                            acc_node, "acc_del|non_tensor_output_with_cpu_user", user
                        )
                        break

            if not new_cpu_nodes:
                break

            for new_cpu_node in new_cpu_nodes:
                self.acc_nodes.remove(new_cpu_node)

            self.reduce_acc_nodes_non_tensor_input_helper(new_cpu_nodes)

    def __call__(self) -> NodeSet:
        submodules = dict(self.module.named_modules())
        self.acc_nodes = set()
        for n in self.module.graph.nodes:
            if n.op not in CALLABLE_NODE_OPS:
                self.tracker.add(n, "init_cpu|not_callable")
                continue
            if not self.operator_support.is_node_supported(submodules, n):
                self.tracker.add(n, "init_cpu|operator_support")
                continue

            self.tracker.add(n, "init_acc|callable_and_operator_supported")
            self.acc_nodes.add(n)

        if not self.allow_non_tensor:
            self.reduce_acc_nodes_non_tensor_input()
            self.reduce_acc_nodes_non_tensor_output()
        self.tracker.dump()
        return self.acc_nodes


@compatibility(is_backward_compatible=False)
class FxNetSplitterInternalError(Exception):
    pass


@compatibility(is_backward_compatible=False)
@dataclass
class Subgraph:
    is_acc: bool
    nodes: NodeList
    device_ordinal: int | None = None


@compatibility(is_backward_compatible=False)
class SplitResult(NamedTuple):
    """
    Stores the results of the splitter.

    Attributes:
        split_module: root module after splitting.
        submodule_inputs: a dict that maps submodule name to its inputs.
        non_acc_submodule_prefix: the prefix for non acc submodules. For
            acc submodule the prefix is always "_run_on_acc_".
    """

    split_module: torch.fx.GraphModule
    submodule_inputs: dict[str, Any]
    non_acc_submodule_prefix: str


@compatibility(is_backward_compatible=False)
def generate_inputs_for_submodules(
    model: torch.nn.Module,
    inputs: Sequence[Any],
    target_submodules: Iterable[str],
    deepcopy: bool = False,
) -> dict[str, Any]:
    """
    Generate inputs for targeting submodules in the given model. Note that if two submodules refer to the same obj, this
    function doesn't work.

    Args:
        model: root model.
        inputs: inputs to the root model.
        target_submodules: submodules that we want to generate inputs for.

    Returns:
        A dict that maps from submodule name to its inputs.
    """

    handles = []
    results: dict[str, Any] = {}
    submodule_to_names = {mod: name for name, mod in model.named_modules()}

    def pre_forward(module: torch.nn.Module, module_inputs: tuple[Any, ...]) -> None:
        results[submodule_to_names[module]] = (
            copy.deepcopy(module_inputs) if deepcopy else module_inputs
        )

    for name, mod in model.named_modules():
        if name in target_submodules:
            if not isinstance(mod, torch.jit.ScriptModule):
                handles.append(mod.register_forward_pre_hook(pre_forward))

    def clean_up_handles() -> None:
        for h in handles:
            h.remove()

    try:
        with torch.no_grad():
            model(*inputs)
    except Exception as e:
        clean_up_handles()
        raise e

    clean_up_handles()
    return results


class _SplitterBase:
    """
    Splits a GraphModule into sub-GraphModules for execution on CPU or the accelerator.
    Output is a GraphModule with supported and unsupported operators grouped into as few sub-GraphModules as possible.
    Assumes that only "call_module", "call_function" and "call_method" from FX IR can potentially be executed on the accelerator.

    Given the following graph:
          ==> b ==>
        //         \\
       a             d
        \\         //
          ==> c ==>

    class SimpleModule(torch.nn.Module):
        def forward(self, a):
            b = torch.sin(a)
            c = torch.cos(a)
            d = b + c
            return d

    and providing "operator_support" that indicates that 'b' and 'c' can be executed on the accelerator,
    we will get the following split result:

    main:
    def forward(self, a):
        run_on_acc_0_0 = self._run_on_acc_0_0(a)
        getitem = run_on_acc_0_0[0]
        getitem_1 = run_on_acc_0_0[1]
        run_on_cpu_1_1 = self._run_on_cpu_1_1(getitem, getitem_1)
        return run_on_cpu_1_1

    _run_on_acc_0_0:
    def forward(self, a):
        sin_1 = torch.sin(a)
        cos_1 = torch.cos(a)
        return (sin_1, cos_1)

    _run_on_cpu_1_1:
    def forward(self, sin_1, cos_1):
        add_1 = sin_1 + cos_1
        return add_1
    """

    # PCIe bandwidth for the backend, default to 100 GB/s
    PCIe_BW = 100 * 2**30

    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Sequence[Any],
        operator_support: OperatorSupportBase,
        settings: _SplitterSettingBase,
        non_acc_submodule_name: str = "_run_on_cpu_",
        return_tuple: bool = False,
        nodes_finder: FxNetAccNodesFinder | None = None,
    ) -> None:
        """
        Preprocesses graph before splitting:
        - finds nodes supported by ACC,
        - finds fusion groups for ACC nodes having non-tensor IO,
        - builds a graph of direct dependencies,
        - builds a map of fused nodes to their fusions.
        As a result we get self.acc_nodes, self.deps and self.fusions.
        """
        if not isinstance(module, torch.fx.GraphModule):
            raise AssertionError(f"Expected GraphModule, got {type(module)}")

        self.module = module
        ShapeProp(self.module).propagate(*sample_input)

        self.settings = settings
        self.operator_support = operator_support
        self.sample_input = sample_input
        if nodes_finder is None:
            nodes_finder = FxNetAccNodesFinder(
                self.module, self.operator_support, self.settings.allow_non_tensor
            )
        self.acc_nodes = nodes_finder()

        if self.settings.skip_fusion:
            self.fusions = {}
        else:
            self.fusions = FxNetAccFusionsFinder(module, self.acc_nodes)()
            self._merge_overlapping_fusions()

        # Modify deps to add more deps for fused nodes
        self.deps = self.find_deps()
        self.update_deps_for_fusions()

        self.non_acc_submodule_name = non_acc_submodule_name
        self._node_submodule_map: dict[str, str] = {}
        self._return_tuple = return_tuple

        self.tags: list[str] = []

    # ===============================================================
    # Helpers for ctor and initial state
    # ===============================================================

    def get_node_submodule_map(self) -> dict[str, str]:
        """Returns a map from node name to submodule name, e.g.
        node: main_module_impl_impl_over_arch_unary_multiple_embedding
          _pooling_embedding_pooling_sparse_entity_equivalence_key
          _proxy_embedding_bag
        maps to submodule name of: _run_on_acc_1
        """
        return self._node_submodule_map

    def find_deps(self) -> dict[torch.fx.Node, NodeSet]:
        """
        Builds a graph of node dependencies. Leaf nodes don't have any
        dependencies and the "output" node doesn't have nodes depending on it.

        Resulting graph has only direct dependencies, i.e. there are no
        transitive dependencies.
        """
        deps: dict[torch.fx.Node, NodeSet] = defaultdict(set)
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            for user in node.users:
                if user.op != "output":
                    deps[user].add(node)
        return deps

    def update_deps_for_fusions(self) -> None:
        """
        Updates graph of dependencies so that:
        - nodes from the same fusion depend on the same set of outer nodes,
        - outer nodes depending on a fusion depend on all nodes in that fusion.
        """
        for node in self.fusions:
            fusion = self.fusions[node]
            for fused_neighbor in fusion:
                self.deps[node].update(self.deps[fused_neighbor] - fusion)

                for user in fused_neighbor.users:
                    if user not in fusion:
                        self.deps[user].add(node)

    def _merge_overlapping_fusions(self) -> None:
        """
        Merge fusion groups that share nodes.

        FxNetAccFusionsFinder can produce overlapping fusion groups when a
        node is absorbed into multiple groups during expansion. When
        update_deps_for_fusions() later propagates deps, shared nodes act
        as bridges between groups, creating bidirectional dependency cycles
        that crash the splitter with "Subgraph can't be empty".

        This method detects overlapping groups via union-find and merges
        them into single groups before dep propagation.
        """
        if os.environ.get("_SPLITTER_MERGE_OVERLAPPING_FUSIONS", "0") != "1":
            return

        if not self.fusions:
            return

        # Collect unique groups by identity.
        unique_groups: dict[int, NodeSet] = {}
        for group in self.fusions.values():
            unique_groups[id(group)] = group

        if len(unique_groups) <= 1:
            return

        # Map each node to all group IDs it belongs to.
        node_to_gids: dict[torch.fx.Node, list[int]] = defaultdict(list)
        for gid, group in unique_groups.items():
            for node in group:
                node_to_gids[node].append(gid)

        # Union-find: merge groups that share nodes.
        parent: dict[int, int] = {gid: gid for gid in unique_groups}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        needs_merge = False
        for gids in node_to_gids.values():
            if len(gids) > 1:
                root = find(gids[0])
                for i in range(1, len(gids)):
                    other = find(gids[i])
                    if other != root:
                        parent[other] = root
                        needs_merge = True

        if not needs_merge:
            return

        # Build merged groups.
        merged: dict[int, NodeSet] = defaultdict(set)
        for gid, group in unique_groups.items():
            merged[find(gid)].update(group)

        # Rebuild self.fusions so every node points to its merged group.
        for merged_group in merged.values():
            for node in merged_group:
                self.fusions[node] = merged_group

    # ===============================================================
    # Helpers for preview
    # ===============================================================

    def _lower_model_to_backend(
        self, mod: torch.fx.GraphModule, inputs: Tensors
    ) -> torch.nn.Module:
        """
        Lower the model to a backend.
        """

        return mod

    def _find_culprit(self, mod: torch.fx.GraphModule, inputs: Tensors) -> str:
        """
        When an error occurs during lowering or running the lowered mod, we use this
        function to find culprits in the `mod` that causes the error.
        """

        return "Unable to find a culprit because _find_culprit() function is not implemented."

    def _draw_graph_based_on_node_support(
        self, mod: torch.fx.GraphModule, supported_nodes: NodeList
    ) -> None:
        color_map = {
            "default": "AliceBlue",
            "supported": "chartreuse1",
            "unsupported": "crimson",
        }

        class CustomDrawer(FxGraphDrawer):
            def _get_node_style(self, node: torch.fx.Node) -> dict[str, str]:
                template = super()._get_node_style(node)
                if node in supported_nodes:
                    template["fillcolor"] = color_map["supported"]
                elif node.op in CALLABLE_NODE_OPS:
                    template["fillcolor"] = color_map["unsupported"]
                else:
                    template["fillcolor"] = color_map["default"]

                return template

        drawer = CustomDrawer(mod, "node_support", ignore_getattr=True)
        dot_graph = drawer.get_main_dot_graph()
        # pyre-fixme[16]: `pydot.Dot` has no attribute `write_raw`.
        dot_graph.write_raw("node_support.dot")  # type: ignore[attr-defined]

    def node_support_preview(self, dump_graph: bool = False) -> str:
        submodules = dict(self.module.named_modules())

        supported_nodes: NodeList = []
        supported_node_types = defaultdict(set)
        unsupported_node_types = defaultdict(set)

        def get_dtype(arg: torch.fx.Node) -> torch.dtype | None:
            tensor_meta = arg.meta.get("tensor_meta")
            return getattr(tensor_meta, "dtype", None)

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            target = get_node_target(submodules, node)

            # Store dtype of arg in node.args. If arg doesn't have dtype, i.e. not a tensor, we'll store None.
            arg_dtypes = [
                get_dtype(arg) if isinstance(arg, torch.fx.Node) else None
                for arg in node.args
            ]

            # Find last non-None element. If all elements are None, return max_len.
            last_index = len(arg_dtypes) - next(
                (
                    i
                    for i, dtype in enumerate(reversed(arg_dtypes))
                    if dtype is not None
                ),
                len(arg_dtypes),
            )

            # Strip None elements at the end.
            arg_dtypes_tuple = tuple(arg_dtypes[:last_index])
            kwarg_dtypes_tuple = tuple(
                (k, get_dtype(arg))
                for k, arg in node.kwargs.items()
                if isinstance(arg, torch.fx.Node)
            )

            if self.operator_support.is_node_supported(submodules, node):
                supported_nodes.append(node)
                supported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))
            else:
                unsupported_node_types[target].add(
                    (arg_dtypes_tuple, kwarg_dtypes_tuple)
                )

        if dump_graph:
            self._draw_graph_based_on_node_support(self.module, supported_nodes)

        reports = "\nSupported node types in the model:\n"
        for t, dtypes in supported_node_types.items():
            for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
                reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

        reports += "\nUnsupported node types in the model:\n"
        for t, dtypes in unsupported_node_types.items():
            for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
                reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

        print(reports)

        # Return reports for testing purpose
        return reports

    def split_preview(self, dump_graph: bool = False) -> str:
        reports = ""
        subgraphs = self.put_nodes_into_subgraphs()
        acc_subgraphs_num = len([g for g in subgraphs if g.is_acc])
        cpu_subgraphs_num = len(subgraphs) - acc_subgraphs_num
        reports += f"Before removing small acc subgraphs, total {len(subgraphs)} subgraphs are created:"
        reports += f" {acc_subgraphs_num} acc subgraphs and {cpu_subgraphs_num} cpu subgraphs.\n"

        subgraphs = self.remove_small_acc_subgraphs(subgraphs)
        acc_subgraphs_num = len([g for g in subgraphs if g.is_acc])
        cpu_subgraphs_num = len(subgraphs) - acc_subgraphs_num
        reports += f"After removing small acc subgraphs, total {len(subgraphs)} subgraphs are created:"
        reports += f" {acc_subgraphs_num} acc subgraphs and {cpu_subgraphs_num} cpu subgraphs.\n"

        for i, subgraph in enumerate(subgraphs):
            reports += (
                f"_run_on_acc_{i}: "
                if subgraph.is_acc
                else f"{self.non_acc_submodule_name}{i}: "
            )
            reports += f"{len(subgraph.nodes)} node(s)\n"

        self.tag(subgraphs)
        split_mod = self.split(remove_tag=True)
        split_mod.eval()

        if dump_graph:
            drawer = FxGraphDrawer(split_mod, "preview", ignore_getattr=True)
            dot_graphs = drawer.get_all_dot_graphs()
            for name, dot_graph in dot_graphs.items():
                # pyre-fixme[16]: `pydot.Dot` has no attribute `write_raw`.
                dot_graph.write_raw(f"{name}.dot")  # type: ignore[attr-defined]

        max_qps: float = self.PCIe_BW
        bottleneck_module = ""

        for node in split_mod.graph.nodes:
            if node.op == "call_module" and "acc" in node.target:
                reports += f"\nProcessing acc submodule {node.target}\n"

                submod = getattr(split_mod, node.target)

                def get_submod_inputs(
                    main_mod: torch.fx.GraphModule,
                    submod: torch.fx.GraphModule,
                    example_inputs: Sequence[Any],
                ) -> tuple[Any, ...]:
                    sub_inputs: tuple[Any, ...] | None = None

                    def get_inputs(
                        self: torch.nn.Module, inputs: tuple[Any, ...]
                    ) -> None:
                        nonlocal sub_inputs
                        sub_inputs = inputs

                    handle = submod.register_forward_pre_hook(get_inputs)
                    main_mod(*example_inputs)
                    handle.remove()
                    if sub_inputs is None:
                        raise AssertionError("Forward pre-hook did not capture inputs")
                    return sub_inputs

                submod_inputs = get_submod_inputs(split_mod, submod, self.sample_input)
                ShapeProp(submod).propagate(*submod_inputs)

                total_input_bytes = 0
                total_output_bytes = 0

                reports += "Checking inputs...\n"
                for n in submod.graph.nodes:
                    if n.op == "placeholder":
                        if not is_node_output_tensor(n):
                            reports += f"Input {n.name} is not a tensor, this might cause problems during lowering!\n"
                        else:
                            total_input_bytes += get_size_of_node(submod, n)[0]
                    if n.op == "output":
                        output_node = n

                reports += "Checking outputs...\n"

                def get_bytes(node: torch.fx.Node) -> None:
                    nonlocal total_output_bytes
                    nonlocal reports
                    if not is_node_output_tensor(node):
                        reports += f"Output {node.name} is not a tensor, this might cause problems during lowering!\n"
                    else:
                        total_output_bytes += get_size_of_node(submod, node)[0]

                map_arg(output_node.args, get_bytes)  # type: ignore[possibly-undefined]
                qps = self.PCIe_BW / max(total_input_bytes, total_output_bytes)
                reports += f"Total input size in bytes is {total_input_bytes}, total output size in bytes is {total_output_bytes},"
                reports += f" theoretical max qps (bounds by PCIe bandwidth) for this submodule is {qps}.\n"

                if qps < max_qps:
                    max_qps = qps
                    bottleneck_module = node.target

                try:
                    lowered_submod = self._lower_model_to_backend(submod, submod_inputs)
                except RuntimeError:
                    reports += "Run into an error during lowering!\n"
                    reports += self._find_culprit(submod, submod_inputs)
                    continue

                try:
                    lowered_submod(*submod_inputs)
                except RuntimeError:
                    reports += "Run into an error during inference!\n"
                    reports += self._find_culprit(submod, submod_inputs)
                else:
                    reports += "Lowering and running succeed!\n"

        reports += f"\nTheoretical max qps (bounds by PCIe bandwidth) for this model is {max_qps},"
        reports += f" bottleneck is submodule {bottleneck_module}."
        print(reports)

        # return the reports for testing purposes
        return reports

    # ===============================================================
    # Helpers for extend_acc_subgraph() method
    # ===============================================================

    def find_reverse_deps(
        self, tag_id: int | None = None
    ) -> dict[torch.fx.Node, NodeSet]:
        """
        Builds reversed topological node dependencies, if tag_id is specified,
        we ignore nodes that are in later subgraph i.e. nodes have greater tag_id.
        """
        result: dict[torch.fx.Node, NodeSet] = defaultdict(set)

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue

                if tag_id is None or (int(user.tag.split("_")[-1]) < tag_id):
                    result[node].add(user)

        return result

    def update_reverse_deps_for_fusions(
        self, deps: dict[torch.fx.Node, NodeSet]
    ) -> None:
        processed_node = set()

        for node, fusion in self.fusions.items():
            if node in processed_node:
                continue

            new_dep = set()

            # Create a new dependency set which include all the
            # dependencies of the nodes in the fusion group
            for n in fusion:
                new_dep.update(deps[n])

            # Exclude nodes in the fusion
            new_dep.difference_update(fusion)

            # Update dependency
            for n in fusion:
                deps[n] = new_dep

                for arg in n.all_input_nodes:
                    if arg not in fusion:
                        deps[arg].update(fusion)

                processed_node.add(n)

    def find_parent_nodes_of_subgraph(self, tag: str) -> NodeSet:
        """
        Finds parent nodes of the `tag` subgraph.

        Traverse the inputs of nodes in the subgraph, if input doesn't belong to the subgraph
        and is not a placeholder, we consider it as the parent node of the subgraph.
        """
        parent_nodes = set()

        for node in self.module.graph.nodes:
            if node.op in CALLABLE_NODE_OPS and node.tag == tag:
                for arg in node.all_input_nodes:
                    if arg.op in CALLABLE_NODE_OPS and arg.tag != tag:
                        parent_nodes.add(arg)

        return parent_nodes

    def extend_acc_subgraph(self, tag: str) -> None:
        """
        Extend the acc subgraph with `tag` going the reversed topological direction.
        """
        # Dict that maps node to its users and ignore users that
        # are in the subgraph that has greater tag
        deps = self.find_reverse_deps(tag_id=int(tag.rsplit("_", maxsplit=1)[-1]))
        self.update_reverse_deps_for_fusions(deps)

        # Parent nodes of the subgraph
        parent_nodes = self.find_parent_nodes_of_subgraph(tag)

        visited_nodes: NodeSet = set()

        while parent_nodes:
            node = None

            # Find a acc node that depends on visited nodes only
            for n in parent_nodes:
                if deps[n] <= visited_nodes and n in self.acc_nodes:
                    node = n
                    break

            if node is None:
                break

            # Put the node into `tag` subgraph
            node.tag = tag  # type: ignore[attr-defined]
            parent_nodes.remove(node)
            visited_nodes.add(node)

            # If node is in a fusion group, add all fusion buddies to parent nodes
            if node in self.fusions:
                for fusion_node in self.fusions[node]:
                    if fusion_node not in visited_nodes:
                        parent_nodes.add(fusion_node)

            # Add inputs of the node to parent nodes
            for arg in node.all_input_nodes:
                if arg.op in CALLABLE_NODE_OPS and arg not in visited_nodes:
                    parent_nodes.add(arg)

    # ===============================================================
    # Helpers for split() method
    # ===============================================================

    def starter_nodes(self) -> tuple[NodeSet, NodeSet]:
        """
        Finds nodes that consume module inputs or get_attr nodes.
        """
        starter_cpu_nodes: NodeSet = set()
        starter_acc_nodes: NodeSet = set()
        for node in self.module.graph.nodes:
            # edge case, call_function, but with no dependencies
            if node.op == "call_function" and len(node.all_input_nodes) == 0:
                if node in self.acc_nodes:
                    starter_acc_nodes.add(node)
                else:
                    starter_cpu_nodes.add(node)

            if node.op not in {"placeholder", "get_attr"}:
                continue

            for user in node.users:
                if user in self.acc_nodes:
                    starter_acc_nodes.add(user)
                else:
                    starter_cpu_nodes.add(user)

        return starter_cpu_nodes, starter_acc_nodes

    def put_nodes_into_subgraphs(self) -> list[Subgraph]:
        # We start graph traversal from leaf nodes
        current_cpu_nodes, current_acc_nodes = self.starter_nodes()
        visited_nodes: NodeSet = set()

        # Determine which subgraph to start from based on which subgraph has
        # 0-dep node
        acc_subgraph: bool = not any(len(self.deps[n]) == 0 for n in current_cpu_nodes)

        current_subgraph_nodes: NodeList = []

        # Result accumulator
        subgraphs: list[Subgraph] = []
        while current_cpu_nodes or current_acc_nodes:
            # Find the first node that should belong to the current subgraph and has all dependencies resolved
            current_nodes = current_acc_nodes if acc_subgraph else current_cpu_nodes
            node = next(
                (n for n in current_nodes if self.deps[n] <= visited_nodes),
                None,
            )

            # If nothing was found, then it's time to flip the mode and start a new subgraph
            if node is None:
                if not current_subgraph_nodes:
                    raise FxNetSplitterInternalError("Subgraph can't be empty")

                subgraphs.append(
                    Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes)
                )
                acc_subgraph = not acc_subgraph
                current_subgraph_nodes = []
                continue

            current_nodes.remove(node)
            visited_nodes.add(node)
            current_subgraph_nodes.append(node)

            # Add fusion buddies
            if node in self.fusions:
                if node in self.acc_nodes:
                    current_acc_nodes.update(self.fusions[node] - visited_nodes)
                else:
                    current_cpu_nodes.update(self.fusions[node] - visited_nodes)

            # Put depending nodes into the queue
            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue

                # Add downstream nodes
                if user in self.acc_nodes:
                    current_acc_nodes.add(user)
                else:
                    current_cpu_nodes.add(user)

        # Check if the last subgraph was not created
        if current_subgraph_nodes:
            subgraphs.append(
                Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes)
            )

        if not subgraphs:
            raise FxNetSplitterInternalError("Couldn't create subgraphs")

        return subgraphs

    def remove_small_acc_subgraphs(self, subgraphs: list[Subgraph]) -> list[Subgraph]:
        """
        This pass finds ACC submodules with less than specified size and merges
        them with adjacent CPU submodules.
        """
        result: list[Subgraph] = []
        for subgraph in subgraphs:
            if subgraph.is_acc:
                if len(subgraph.nodes) >= self.settings.min_acc_module_size:
                    result.append(subgraph)
                else:
                    print(
                        "Eliminating acc subgraph because it's smaller than the threshold: "
                        f"{len(subgraph.nodes)} < {self.settings.min_acc_module_size}"
                    )
                    if result:
                        result[-1].nodes.extend(subgraph.nodes)
                    else:
                        subgraph.is_acc = False
                        result.append(subgraph)
            else:
                if result and not result[-1].is_acc:
                    result[-1].nodes.extend(subgraph.nodes)
                else:
                    result.append(subgraph)
        return result

    def tag(self, subgraphs: list[Subgraph]) -> None:
        self.tags = []
        for subgraph in subgraphs:
            tag = (
                f"_run_on_acc_{len(self.tags)}"
                if subgraph.is_acc
                else f"{self.non_acc_submodule_name}{len(self.tags)}"
            )
            self.tags.append(tag)
            for node in subgraph.nodes:
                if hasattr(node, "tag"):
                    raise FxNetSplitterInternalError(f"Node {node} was already tagged")

                node.tag = tag  # type: ignore[attr-defined]
                self._node_submodule_map[node.name] = tag

    def split(self, remove_tag: bool = False) -> torch.fx.GraphModule:
        split_module = split_by_tags(
            self.module, self.tags, return_tuple=self._return_tuple
        )
        if remove_tag:
            for node in self.module.graph.nodes:
                if hasattr(node, "tag"):
                    del node.tag
        return split_module  # type: ignore[return-value]

    def __call__(self) -> torch.fx.GraphModule:
        subgraphs = self.put_nodes_into_subgraphs()
        if self.settings.move_non_tensor_nodes_on_boundary:
            move_non_tensor_nodes_on_boundary(subgraphs)
        subgraphs = self.remove_small_acc_subgraphs(subgraphs)
        acc_subgraphs_count = len([s for s in subgraphs if s.is_acc])
        non_acc_subgraphs_count = len(subgraphs) - acc_subgraphs_count
        print(
            f"Got {acc_subgraphs_count} acc subgraphs and {non_acc_subgraphs_count} non-acc subgraphs"
        )
        self.tag(subgraphs)
        return self.split()

    def generate_split_results(self) -> SplitResult:
        split_module = self()
        submodule_names: list[str] = []
        for name, _mod in split_module.named_children():
            submodule_names.append(name)
        if (
            self.settings.max_acc_splits > 0
            and len(submodule_names) > self.settings.max_acc_splits
        ):
            raise ValueError(
                "Cannot fulfill max_acc_splits limit. "
                "This may cause split fragmentation and "
                "result in performance issues."
            )

        submodule_inputs = generate_inputs_for_submodules(
            split_module, self.sample_input, submodule_names
        )
        return SplitResult(split_module, submodule_inputs, self.non_acc_submodule_name)
<|endoftext|>from typing import Any

import torch
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import Target
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._pytree import tree_flatten


aten = torch.ops.aten


# stateful ops are banned from CSE
rand_ops = {
    aten.dropout,
    aten._fused_dropout,
    aten._standard_gamma,
    aten.bernoulli,
    aten.multinomial,
    aten.native_dropout,
    aten.normal,
    aten.poisson,
    aten.binomial,
    aten.rrelu,
    aten.rand_like,
    aten.rand,
    aten.randint,
    aten.randn,
    aten.randperm,
}

inplace_ops = {
    aten.add_,
    aten.sub_,
    aten.mul_,
    aten.div_,
    aten.pow_,
    aten.lerp_,
    aten.relu_,
    aten.sigmoid_,
    aten.tanh_,
}


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def get_CSE_banned_ops() -> set[torch._ops.OpOverloadPacket]:
    return rand_ops.union(inplace_ops)


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
class CSEPass(PassBase):
    def __init__(
        self, banned_ops: set[torch._ops.OpOverloadPacket] | None = None
    ) -> None:
        """
        This version of CSE Pass aims to be dialect agnostic, and it's implemented purely based on the connectivity between fx.Node.

        For functional dialects, user would only need to specify the random ops in ban list.

        Warning: CSE Pass cannot be safely applied on a FX graph in non-functional dialects.
        If your dialect contains stateful operators, please customized the banned_ops.

        """
        if banned_ops is None:
            banned_ops = set()
        self.banned_ops = banned_ops
        super().__init__()

    def call(self, graph_module: GraphModule) -> PassResult:
        """
        Return a new copy of torch.fx.GraphModule with CSE applied to the input graph

        Example usage:

        from torch.fx.experimental.proxy_tensor import make_fx
        def f(a):
            b = a * a
            c = a * a
            return b+c

        p = CSEPass()
        traced_graph = make_fx(f)(torch.tensor(1))
        print(traced_graph)
        result = p(traced_graph)
        print(result.graph_module)
        """

        def get_aten_target(node: Node) -> Target:
            if hasattr(node.target, "overloadpacket"):
                return node.target.overloadpacket
            return node.target

        modified = False
        new_graph = Graph()
        env: dict[
            Node, Node
        ] = {}  # map from node in the old graph to node in the new graph
        hash_env: dict[
            tuple[Target, int], Node
        ] = {}  # map from hash to a node in the new graph
        token_map: dict[
            tuple[Target, int], dict[str, Any]
        ] = {}  # map from hash to token
        for n in graph_module.graph.nodes:
            # The placeholder, output, and get_attr nodes are copied to the new graph without change
            # do not CSE away random operations
            if (
                n.op == "placeholder"
                or n.op == "output"
                or n.op == "get_attr"
                or get_aten_target(n) in self.banned_ops
            ):
                new_node = new_graph.node_copy(n, lambda x: env[x])
                env[n] = new_node
            else:  # n.op == 'call_function', should never see n.op == 'call_module' or 'call_method'
                # substitute args and kwargs members to their mapping in env if exists
                # specs can be used to reconstruct nested list/dictionaries
                def substitute(arg_list: Any) -> tuple[tuple[Any, ...], Any]:
                    arg_list, spec = tree_flatten(arg_list)
                    for i in range(len(arg_list)):
                        v = arg_list[i]
                        if isinstance(v, Node) and v in env:
                            arg_list[i] = env[v]
                    return tuple(arg_list), spec

                args, args_spec = substitute(n.args)
                kwargs, kwargs_spec = substitute(n.kwargs)

                # each token corresponds to a unique node
                # nodes with the same token can be substituted
                token = {
                    "target": n.target,
                    "args": args,
                    "args_spec": args_spec,
                    "kwargs": kwargs,
                    "kwargs_spec": kwargs_spec,
                }

                # hash substituted args to a number, do not hash specs because specs are not hashable
                hash_arg = hash((args, kwargs))
                hash_val = (n.target, hash_arg)

                # check if a node has a substitute and can be eliminated
                hash_val_in_hash_env = hash_val in hash_env
                if hash_val_in_hash_env and token_map[hash_val] == token:
                    modified = True  # substitution happens and the graph is modified
                    env[n] = hash_env[hash_val]
                    continue

                new_node = new_graph.node_copy(n, lambda x: env[x])
                env[n] = new_node
                if not hash_val_in_hash_env:
                    hash_env[hash_val] = new_node
                    token_map[hash_val] = token

        if not modified:
            return PassResult(graph_module, False)
        csed_gm = GraphModule(graph_module, new_graph)
        return PassResult(csed_gm, True)
<|endoftext|>import collections
import itertools
import logging
import operator
from collections.abc import Iterable, Sequence

from torch.fx.graph_module import GraphModule
from torch.fx.node import _get_qualified_name, Node
from torch.fx.passes.operator_support import OperatorSupportBase
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Partition:
    def __init__(
        self,
        id: int | None = None,
        nodes: Iterable[Node] | None = None,
        node_orders: Iterable[int] | None = None,
    ) -> None:
        self.id = id
        self.nodes: dict[Node, int | None] = {}
        if nodes is not None:
            if node_orders is None:
                self.nodes = dict.fromkeys(nodes, None)
            else:
                nodes_list = list(nodes)
                node_orders_list = list(node_orders)
                if len(nodes_list) != len(node_orders_list):
                    raise AssertionError(
                        "nodes and node_orders must have the same length"
                    )
                self.nodes = dict(zip(nodes_list, node_orders_list))

    def __repr__(self) -> str:
        return str(self.nodes)

    def add_node(self, node: Node, node_order: int | None = None) -> None:
        self.nodes.update({node: node_order})

    def remove_node(self, node: Node) -> None:
        del self.nodes[node]

    def size(self) -> int:
        return len(self.nodes)


class _DependencyViewer:
    def __init__(self, graph_module: GraphModule) -> None:
        self.downstreams = collections.defaultdict(set)

        for node in reversed(graph_module.graph.nodes):
            for output_node in node.users:
                # add output_node and output_node's downstream dependency
                self.downstreams[node].add(output_node)
                self.downstreams[node].update(self.downstreams[output_node])

    def downstreams_of(self, node: Node) -> set[Node]:
        return self.downstreams[node]


class CapabilityBasedPartitioner:
    def __init__(
        self,
        graph_module: GraphModule,
        operator_support: OperatorSupportBase,
        allows_single_node_partition: bool = False,
        non_compute_ops: Sequence[str] | None = None,
        allowed_single_node_partition_ops: Sequence[str] | None = None,
        skip_horizontal_fusion: bool = False,
    ) -> None:
        self.graph_module = graph_module
        self.operator_support = operator_support
        self.allows_single_node_partition = allows_single_node_partition
        self.non_compute_ops = non_compute_ops if non_compute_ops is not None else []
        self.allowed_single_node_partition_ops = (
            allowed_single_node_partition_ops
            if allowed_single_node_partition_ops is not None
            else []
        )
        self.skip_horizontal_fusion = skip_horizontal_fusion
        self.dependency_viewer = _DependencyViewer(graph_module)

    def _is_node_supported(self, node: Node) -> bool:
        return self.operator_support.is_node_supported(
            dict(self.graph_module.named_modules()), node
        )

    def propose_partitions(self) -> list[Partition]:
        # partition_map is a mapping from partition id to a set of partition id's.
        # The value set contains all the partition ids that can be reached by doing a
        # DFS starting from the partition id in the key.
        partition_map: dict[int, set[int]] = collections.defaultdict(set)

        # assumptions: nodes in candidate list is sorted in topological order
        assignment: dict[Node, int] = {}  # mapping from node to partition_id
        partitions_by_id: dict[
            int, Partition
        ] = {}  # mapping from partition_id to partition
        nodes_order: dict[
            Node, int
        ] = {}  # mapping from nodes to reversed topological order
        partitions_order: dict[
            int, int
        ] = {}  # mapping from partition_id to minimum topo order of nodes in partition
        partition_users: dict[
            int, set[Node]
        ] = {}  # mapping from partition_id to partition users
        new_partition_id = itertools.count()

        def downstream_partitions(
            user_nodes: Iterable[Node], source_ids: Iterable[int]
        ) -> set[int]:
            source_ids_set = set(source_ids)
            downstream_partition_ids: set[int] = set()
            for user_node in user_nodes:
                for path_node in itertools.chain(
                    (user_node,), self.dependency_viewer.downstreams_of(user_node)
                ):
                    target_id = assignment.get(path_node)
                    if target_id is None:
                        continue
                    downstream_partition_ids.add(target_id)
                    downstream_partition_ids.update(partition_map[target_id])

            downstream_partition_ids.difference_update(source_ids_set)
            return downstream_partition_ids

        # try to merge partition other_id into partition self_id
        # merge only happens if the end graph doesn't contain cyclic dependency
        # returns `True` when merge happens, `False` otherwise.
        def maybe_merge_partition(self_id: int, other_id: int) -> tuple[int, bool]:
            # merged_nodes is the union of nodes in two partition to-be-merged
            self_nodes = partitions_by_id[self_id].nodes
            other_nodes = partitions_by_id[other_id].nodes

            def dfs_iter_find_cycle(all_user_nodes: set[Node]) -> bool:
                for user_node in all_user_nodes:
                    visited_partition_ids = set()

                    for path_node in itertools.chain(
                        (user_node,), self.dependency_viewer.downstreams_of(user_node)
                    ):
                        # If any of the nodes in the dfs path of this node are in the merged_nodes
                        # list then there is a cycle in the graph.
                        if path_node in self_nodes or path_node in other_nodes:
                            return True

                        # If any of the nodes in the dfs path of this node are in the assignment
                        # map then we have to make sure that the partitions that these nodes belong
                        # to do not form a cycle with the current partitions being merged. This means
                        # iterating through all the nodes in all the partitions that are traversed in
                        # the dfs path and checking if they are in the merged_nodes list.
                        if path_node in assignment:
                            partition_id = assignment[path_node]
                            # If the partition id has already been visited then we know that it doesn't
                            # form a cycle with the current partitions being merged.
                            if partition_id in visited_partition_ids:
                                continue
                            p_map = partition_map[partition_id]
                            if self_id in p_map or other_id in p_map:
                                return True

                            visited_partition_ids.add(partition_id)

                return False

            # find new partition users if merge.
            all_user_nodes = partition_users[self_id] | partition_users[other_id]
            all_user_nodes.difference_update(other_nodes, self_nodes)

            # check if merge would create cyclic dependency.
            if dfs_iter_find_cycle(all_user_nodes):
                # return false indicating cyclic dependency found and
                # merge is aborted
                return self_id, False

            # merge the smaller partition into the larger.
            merge_id, removed_id = self_id, other_id
            if len(self_nodes) < len(other_nodes):
                merge_id, removed_id = removed_id, merge_id
            # no cyclic dependency found, move forward with the merge
            # updating partition nodes
            partitions_by_id[merge_id].nodes.update(partitions_by_id[removed_id].nodes)
            # updating assignment map
            for node in partitions_by_id[removed_id].nodes:
                assignment[node] = merge_id
            # delete other partition
            del partitions_by_id[removed_id]

            partitions_order[merge_id] = min(
                partitions_order[merge_id], partitions_order[removed_id]
            )
            del partitions_order[removed_id]

            partition_map[merge_id] = partition_map[merge_id].union(
                partition_map[removed_id]
            )
            del partition_map[removed_id]

            partition_users[merge_id] = all_user_nodes
            del partition_users[removed_id]

            partition_map[merge_id].update(
                downstream_partitions(all_user_nodes, (merge_id, removed_id))
            )
            partition_map[merge_id].discard(merge_id)
            partition_map[merge_id].discard(removed_id)

            return merge_id, True

        def merge_single_node(
            node: Node, node_order: int | None, id: int | None
        ) -> None:
            def _update_partition_map(node: Node, id: int) -> None:
                # Update reachability through both assigned and unsupported users.
                partition_map[id].update(downstream_partitions(node.users, (id,)))

            if node in assignment:
                partitions_by_id[assignment[node]].remove_node(node)

            if id is None:
                assignment.pop(node)
            elif id not in partitions_by_id:
                assignment[node] = id
                if node_order is None:
                    raise AssertionError("node_order is required for new partitions")
                partitions_by_id[id] = Partition(
                    id=id, nodes=[node], node_orders=[node_order]
                )
                partition_users[id] = set(node.users)
                _update_partition_map(node, id)
            else:
                assignment[node] = id
                partitions_by_id[id].add_node(node, node_order)

        logger.debug("Proposing partitions...")

        for node_order, node in enumerate(reversed(self.graph_module.graph.nodes)):
            # use Dict as an ordered set to ensure deterministic partitioning result, don't care value
            merge_candidates: dict[int, None] = {}
            created_partition = False

            # Note a limited horizontal fusion is enabled:
            #   when `node` is not supported, the code below attempts to fuse consumer of `node`.
            #
            # When skip_horizontal_fusion is True, only merge the newly-created
            # partition for this supported node with partitions for its direct users.
            # This preserves data-dependent vertical fusion while avoiding fusion of
            # independent consumer partitions around an unsupported node.
            if self._is_node_supported(node) and node not in assignment:
                partition_id = next(new_partition_id)
                nodes_order[node] = partition_id
                partitions_order[partition_id] = partition_id
                merge_single_node(node, node_order, partition_id)
                merge_candidates[partition_id] = None
                created_partition = True

            if self.skip_horizontal_fusion:
                if created_partition:
                    for user in node.users:
                        if user in assignment:
                            merge_candidates[assignment[user]] = None
            else:
                # merge all possible partitions
                for partition_id, _ in sorted(
                    partitions_order.items(), key=operator.itemgetter(1)
                ):
                    merge_candidates[partition_id] = None

            merge_candidates_list = list(merge_candidates.keys())
            if len(merge_candidates_list) > 1:
                self_id = merge_candidates_list[0]
                for other_id in merge_candidates_list[1:]:
                    # note: merge partitions if it doesn't create cyclic dependency
                    # in the graph, otherwise, this is a no-op
                    self_id, _ = maybe_merge_partition(self_id, other_id)

        # sort partition nodes based on descending node order
        for partition in partitions_by_id.values():
            partition.nodes = dict(
                sorted(
                    partition.nodes.items(), key=operator.itemgetter(1), reverse=True
                )
            )

        # post processing to re-assign "getitem" nodes into upstream partition
        # Run iteratively until no more changes, to handle nested getitem chains
        # (e.g., getitem_619 = getitem_618[0] where getitem_618 = with_effects_167[1])
        logger.debug("Reassigning getitem nodes to its producer node's partition...")
        while True:
            nodes_reassignment: dict[Node, int] = {}
            for node in self.graph_module.graph.nodes:
                is_tuple_output = True
                for user in node.users:
                    if (
                        user.op != "call_function"
                        or _get_qualified_name(user.target) != "_operator.getitem"
                    ):  # type: ignore[arg-type]
                        is_tuple_output = False
                        break

                # node has tuple outputs, re-assign all following getitem node into node's partition
                if is_tuple_output:
                    id = assignment.get(node)  # type: ignore[arg-type]
                    for user in node.users:
                        if assignment.get(user) != id:  # type: ignore[arg-type]
                            nodes_reassignment[user] = id  # type: ignore[assignment]

            # no more re-assignments
            if not nodes_reassignment:
                break

            for node, id in nodes_reassignment.items():
                merge_single_node(node, None, id)

        # filter out single node partitions
        if not self.allows_single_node_partition:
            logger.debug("Filtering out single node partitions...")
            default_non_compute_ops = {"torch.ops.aten.view", "_operator.getitem"}
            non_compute_ops = default_non_compute_ops.union(set(self.non_compute_ops))
            partitions_to_remove: list[int] = []
            for id, partition in partitions_by_id.items():
                compute_node_count = 0
                for node in partition.nodes:
                    if node.op == "call_function":
                        if not callable(node.target):
                            raise AssertionError(
                                f"Expected callable target, got {type(node.target)}"
                            )
                        if _get_qualified_name(node.target) not in non_compute_ops:
                            compute_node_count += 1
                        if (
                            _get_qualified_name(node.target)
                            in self.allowed_single_node_partition_ops
                        ):
                            compute_node_count += 1
                if compute_node_count <= 1:
                    partitions_to_remove.append(id)
            for id in partitions_to_remove:
                del partitions_by_id[id]

        logger.debug("Partitions proposed:")
        for id, partition in partitions_by_id.items():
            logger.debug(
                "partition #%s: %s", id, [node.name for node in partition.nodes]
            )

        return [
            partition for partition in partitions_by_id.values() if partition.size() > 0
        ]

    def fuse_partitions(
        self, partitions: list[Partition], prefix: str = "fused_"
    ) -> GraphModule:
        logger.debug("Fusing partitions...")
        # fuse_by_partitions expects partitions in List[Dict[Node, None]]: [ {node0 : None}, {node1 : None} ]
        return fuse_by_partitions(
            self.graph_module,
            [partition.nodes for partition in partitions],
            prefix=prefix,
        )

    # remove non-compute-ops that sits at the boundary of a partition.
    def remove_bookend_non_compute_ops(self, partitions: list[Partition]) -> None:
        non_compute_ops = set(self.non_compute_ops)

        def is_non_compute_node(node: Node) -> bool:
            return (
                node.op == "call_function"
                and _get_qualified_name(node.target) in non_compute_ops  # type: ignore[arg-type]
            )

        # cache transparent nodes
        transparent_input_nodes: dict[Node, bool] = {}
        transparent_output_nodes: dict[Node, bool] = {}

        def is_transparent_input_node(
            node: Node, partition: set[Node], removed_nodes: set[Node]
        ) -> bool:
            if (
                node.op == "placeholder"
                or (node not in partition)
                or (node in removed_nodes)
            ):
                return True
            if node in transparent_input_nodes:
                return transparent_input_nodes[node]
            if is_non_compute_node(node):
                for input_n in node.all_input_nodes:
                    if not is_transparent_input_node(input_n, partition, removed_nodes):
                        transparent_input_nodes[node] = False
                        return False
                transparent_input_nodes[node] = True
                return True
            transparent_input_nodes[node] = False
            return False

        def is_transparent_output_node(
            node: Node, partition: set[Node], removed_nodes: set[Node]
        ) -> bool:
            if (
                node.op == "placeholder"
                or (node not in partition)
                or (node in removed_nodes)
            ):
                return True
            if node in transparent_output_nodes:
                return transparent_output_nodes[node]
            if is_non_compute_node(node):
                for output_n in node.users:
                    if not is_transparent_output_node(
                        output_n, partition, removed_nodes
                    ):
                        transparent_output_nodes[node] = False
                        return False
                transparent_output_nodes[node] = True
                return True
            transparent_output_nodes[node] = False
            return False

        for partition in partitions:
            # Note it's ok to use `set` here, since we are only query if a node
            # has been removed. We are NEVER going to iterate on nodes inside
            # the set.
            remove_node: set[Node] = set()
            for node in partition.nodes:
                if is_non_compute_node(node) and (
                    is_transparent_input_node(node, set(partition.nodes), remove_node)
                    or is_transparent_output_node(
                        node, set(partition.nodes), remove_node
                    )
                ):
                    remove_node.add(node)

            if len(remove_node) != 0:
                for node in remove_node:
                    partition.nodes.pop(node, None)

    def partition_and_fuse(self, prefix: str = "fused_") -> GraphModule:
        partitions = self.propose_partitions()
        fused_gm = self.fuse_partitions(partitions, prefix=prefix)
        return fused_gm
<|endoftext|>import abc
from collections import namedtuple

import torch.nn as nn
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule


__all__ = ["PassResult", "PassBase"]


@compatibility(is_backward_compatible=False)
# pyrefly: ignore [invalid-inheritance]
class PassResult(namedtuple("PassResult", ["graph_module", "modified"])):
    """
    Result of a pass:
        graph_module: The modified graph module
        modified: A flag for if the pass has modified the graph module
    """

    __slots__: tuple[str, ...] = ()

    def __new__(cls, graph_module: nn.Module, modified: bool) -> "PassResult":
        return super().__new__(cls, graph_module, modified)


@compatibility(is_backward_compatible=False)
class PassBase(abc.ABC):
    """
    Base interface for implementing passes.

    It is required to implement the `call` function so that we can directly
    pass instances of the Pass directly to the PassManager and call them as a
    function.

    We can directly pass an instance of a class implementing this interface into
    the PassManager's `passes` attribute.
    """

    def __call__(self, graph_module: GraphModule) -> PassResult | None:
        """
        Runs the precondition check, the pass itself, and the postcondition check.
        """

        self.requires(graph_module)
        res = self.call(graph_module)
        self.ensures(graph_module)
        return res

    @abc.abstractmethod
    def call(self, graph_module: GraphModule) -> PassResult | None:
        """
        The pass that is run through the given graph module. To implement a
        pass, it is required to implement this function.

        Args:
            graph_module: The graph module we will run a pass on
        """

    def requires(self, graph_module: GraphModule) -> None:  # noqa: B027
        """
        This function will be called before the pass is run and will check that
        the given graph module contains the preconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """

    def ensures(self, graph_module: GraphModule) -> None:  # noqa: B027
        """
        This function will be called after the pass is run and will check that
        the given graph module contains the postconditions needed to run the
        pass. It is not required to implement this function.

        Args:
            graph_module: The graph module we will run checks on
        """
<|endoftext|>import inspect
import logging
from collections.abc import Callable
from functools import wraps
from queue import Queue
from typing import Any

import torch.nn as nn
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.passes.infra.pass_base import PassResult


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

__all__ = ["pass_result_wrapper", "this_before_that_pass_constraint", "PassManager"]


@compatibility(is_backward_compatible=False)
def pass_result_wrapper(fn: Callable[..., Any]) -> Callable[..., PassResult | None]:
    """
    Wrapper for passes which currently do not return a PassResult.
    This wrapper makes them return a PassResult containing the modified object
    and True for the "modified" flag.

    Args:
        fn (Callable[Module, Any])

    Returns:
        wrapped_fn (Callable[Module, PassResult])
    """
    if fn is None:
        # pyrefly: ignore [bad-return]
        return None

    @wraps(fn)
    def wrapped_fn(gm: nn.Module) -> PassResult | None:
        res = fn(gm)
        if res is None:
            return PassResult(gm, True)
        if isinstance(res, PassResult):
            return res
        elif isinstance(res, nn.Module):
            return PassResult(res, True)

    if not inspect.isfunction(fn):
        wrapped_fn.__name__ = type(fn).__name__

    return wrapped_fn


def _validate_pass_schedule_constraint(
    constraint: Callable[[Callable[..., Any], Callable[..., Any]], bool],
    passes: list[Callable[..., Any]],
) -> None:
    for i, a in enumerate(passes):
        for j, b in enumerate(passes[i + 1 :]):
            if constraint(a, b):
                continue
            raise RuntimeError(
                f"pass schedule constraint violated. Expected {a} before {b}"
                f" but found {a} at index {i} and {b} at index {j} in pass"
                f" list."
            )


def _topological_sort_passes(
    passes: list[Callable[..., Any]], constraints: list[Callable[..., Any]]
) -> list[Callable[..., Any]]:
    """
    Args
        passes: Passes that we are ordering
        constraints: Constraints applied on these passes

    Returns
        A sorted list of callables and a boolean of if a circular dependency
        existed
    """
    if len(constraints) == 0:
        return passes

    # Construct a graph mapping nodes to a list of their users
    graph: dict[Callable[..., Any], list[Callable[..., Any]]] = {p: [] for p in passes}
    indegree_map: dict[Callable[..., Any], int] = dict.fromkeys(passes, 0)
    candidates: Queue[Callable[..., Any]] = Queue()
    for a in passes:
        for b in passes:
            if a == b:
                continue

            for constraint in constraints:
                if not constraint(a, b):
                    graph[b].append(a)
                    indegree_map[a] += 1

        if indegree_map[a] == 0:
            candidates.put(a)

    visited: dict[Callable[..., Any], bool] = dict.fromkeys(passes, False)
    sorted_passes: list[Callable[..., Any]] = []

    while not candidates.empty():
        p = candidates.get()
        sorted_passes.append(p)
        visited[p] = True

        for n in graph[p]:
            if not visited[n]:
                indegree_map[n] -= 1
                if indegree_map[n] == 0:
                    candidates.put(n)

    # Check if there are unvisited nodes (aka cycles in the graph)
    cycle_passes = list(filter(lambda p: indegree_map[p] != 0, indegree_map.keys()))
    if len(cycle_passes) != 0:
        error = (
            f"Circular dependency detected within the following passes: {cycle_passes}"
        )
        raise RuntimeError(error)

    return sorted_passes


@compatibility(is_backward_compatible=False)
def this_before_that_pass_constraint(
    this: Callable[..., Any], that: Callable[..., Any]
) -> Callable[[Callable[..., Any], Callable[..., Any]], bool]:
    """
    Defines a partial order ('depends on' function) where ``this`` must occur
    before ``that``.

    For example, the following pass list and constraint list would be invalid::

        passes = [pass_b, pass_a]

        constraints = [this_before_that_pass_constraint(pass_a, pass_b)]

    Args:
        this (Callable): pass which should occur first
        that (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool])
    """

    def depends_on(a: Callable[..., Any], b: Callable[..., Any]) -> bool:
        return a != that or b != this

    return depends_on


@compatibility(is_backward_compatible=False)
class PassManager:
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): List of passes. A pass is a
            callable which modifies an object and returns a PassResult
        constraint (Optional[List[Callable]]): List of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
        steps (int): Max number of times we run the passes (default = 1).
        run_checks_after_each_pass (bool): Whether to run checks and linting
            after each pass
        suppress_check_failures (bool): Whether to raise errors when running
            checks
    """

    passes: list[Callable[..., PassResult | None]]
    constraints: list[Callable[[Callable[..., Any], Callable[..., Any]], bool]]
    _validated: bool = False
    steps: int = 1

    def __init__(
        self,
        passes: list[Callable[..., PassResult | None]] | None = None,
        constraints: list[Callable[[Callable[..., Any], Callable[..., Any]], bool]]
        | None = None,
        steps: int | None = None,
        run_checks_after_each_pass: bool = False,
        suppress_check_failures: bool = False,
    ) -> None:
        self.passes = passes or []
        self.constraints = constraints or []
        if steps:
            self.steps = steps

        self.run_checks_after_each_pass = run_checks_after_each_pass
        self.suppress_check_failures = suppress_check_failures

    def add_pass(self, _pass: Callable[..., PassResult | None]) -> None:
        """
        Adds a pass into the current list of passes.
        """
        self.passes.append(_pass)
        self._validated = False

    def add_constraint(
        self, constraint: Callable[[Callable[..., Any], Callable[..., Any]], bool]
    ) -> None:
        """
        Adds a constraint into the current list of constraints.
        """
        self.constraints.append(constraint)
        self._validated = False

    def validate_constraints(self) -> None:
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
        if self._validated:
            return
        for constraint in self.constraints:
            _validate_pass_schedule_constraint(constraint, self.passes)
        self._validated = True

    def solve_constraints(self) -> None:
        """
        Finds a valid traversal order based on the given constraints and orders
        the passes based on this order.

        If a circular dependency exists between the constraints and steps = 1,
        then we will raise an error because if steps != 1 this means that we
        will re-run the passes, allowing for circular dependencies.
        """
        self.passes = _topological_sort_passes(self.passes, self.constraints)
        self._validated = True

    def add_checks(self, check: Callable[[nn.Module], None]) -> None:
        """
        Adds a function which takes runs various checks on a given graph module.
        This function is run before and after each pass if the
        `run_checks_after_each_pass` flag is enabled.
        """
        sig = inspect.signature(check)

        if len(list(sig.parameters.values())) != 1:
            raise TypeError(
                "PassManager check function should only take in one variable, a module"
            )

        setattr(self, "check", check)  # noqa: B010

    def check(self, module: nn.Module) -> None:
        pass

    def __call__(self, module: nn.Module) -> PassResult:
        """
        Runs a list of passes in the order based on `self.passes` on the given
        graph module. Each time a pass is run, checks and linting will be run on
        the graph module if `run_checks_after_each_pass` is set.

        If the module is a graph module, we will run the list of passes until
        the graph stops changing, or until `steps` number of times.
        """
        # Order the passes based on the constraints
        if not self._validated:
            self.solve_constraints()

        # Check graph invariants
        self.check(module)

        # Run the set of passes `steps` number of times or until the graph stops
        # changing
        overall_modified = False
        for _ in range(self.steps):
            modified = False

            # Run the set of passes on the graph module
            for i, fn in enumerate(self.passes):
                fn_name = fn.__name__ if inspect.isfunction(fn) else type(fn).__name__
                logger.debug("Running pass '%s'", fn_name)

                try:
                    res = fn(module)

                    if not isinstance(res, PassResult) and not hasattr(
                        res, "graph_module"
                    ):
                        raise TypeError(
                            f"The result of the pass {fn_name} should be type PassResult."
                            + "Please wrap it with pass_result_wrapper()"
                        )
                    # pyrefly: ignore[missing-attribute]
                    module = res.graph_module
                    # pyrefly: ignore[missing-attribute]
                    modified = modified or res.modified

                    if isinstance(module, GraphModule):
                        logger.debug("Graph after pass '%s': %s", fn_name, module.graph)
                        module.recompile()

                    # Check graph invariants
                    if self.run_checks_after_each_pass:
                        self.check(module)

                except Exception as e:
                    prev_pass_names = [
                        p.__name__ if inspect.isfunction(p) else type(p).__name__
                        for p in self.passes[:i]
                    ]
                    msg = f"An error occurred when running the '{fn_name}' pass after the following passes: {prev_pass_names}"
                    raise Exception(msg) from e  # noqa: TRY002

            # If the graph no longer changes, then we can stop running these passes
            overall_modified = overall_modified or modified
            if not modified:
                break

        return PassResult(module, overall_modified)
<|endoftext|>import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.node import Node


__all__ = ["get_source_partitions", "check_subgraphs_connected", "SourcePartition"]


# Set`PYTORCH_MATCHER_LOGLEVEL=INFO` to see debug logs
def _init_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)

    level = os.environ.get("PYTORCH_MATCHER_LOGLEVEL", "WARNING").upper()
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(filename)s > %(message)s")
    console.setFormatter(formatter)
    console.setLevel(level)
    # add the handlers to the logger
    logger.addHandler(console)
    logger.propagate = False
    return logger


logger = _init_logger()


@compatibility(is_backward_compatible=False)
@dataclass
class SourcePartition:
    # Nodes in a particular partition
    nodes: list[Node]

    # The source these nodes decomposed from
    source: Any

    # Nodes in the graph that are needed as inputs to the partition
    # These do not include the params of the partition
    input_nodes: list[Node] = field(default_factory=list)

    # Nodes in the partition that are being used by nodes outside of the
    # partition
    output_nodes: list[Node] = field(default_factory=list)

    # Parameters that are being used
    params: list[Node] = field(default_factory=list)


@compatibility(is_backward_compatible=False)  # type: ignore[misc]
def get_source_partitions(
    graph: Graph,
    wanted_sources: list[Any],
    filter_fn: Callable[[Node], bool] | None = None,
) -> dict[Any, list[SourcePartition]]:
    """
    Args:
        graph: The graph we want to partition
        wanted_sources: List of sources of nodes that were decomposed from this
            source. This can be a function (ex. torch.nn.functional.linear) or a
            leaf module type (ex. torch.nn.Linear).

    Returns:
        Dictionary mapping sources that were given to a list of SourcePartitions
        that correspond to the list of nodes that were decomposed from the given
        source.
    """
    modules: dict[type, dict[str, list[Node]]] = {}

    def add_to_partition(src: Any, fqn: str, node: Node) -> None:
        diff_modules = modules.setdefault(src, {})
        partition = diff_modules.setdefault(fqn, [])
        partition.append(node)

    for node in graph.nodes:
        # The metadata source_fn should contain a tuple of a unique name for the
        # source, and the source function if the node is decomposed from a
        # function, or the type of module if the node is decomposed from a leaf
        # module

        # TODO: Bypass "torch_fn" when "source_fn_stack" because now "torch_fn" can
        # be different from "source_fn_stack", for example for the add_ node
        # decomposed from batch norm. We should remove the check on "source_fn_stack"
        # after we fix "torch_fn". T199561090
        source_fn_st = node.meta.get("source_fn_stack", None)
        if source_fn_st is None:
            matched = False
            torch_fn = node.meta.get("torch_fn", None)
            if torch_fn is not None:
                node_fqn, source_fn = torch_fn
                source_fn_name = source_fn.split(".")[1]
                if source_fn_name in wanted_sources:
                    add_to_partition(source_fn_name, node_fqn, node)
                    matched = True
            # Fallback: when source_fn_stack is not populated (e.g. strict=False export),
            # use nn_module_stack to resolve the originating module type.
            # Only apply to call_function nodes to avoid incorrectly including
            # placeholder, get_attr, or output nodes in partitions.
            if not matched and node.op == "call_function":
                nn_module_stack = node.meta.get("nn_module_stack", None)
                if nn_module_stack:
                    # Get the innermost module (last entry in the ordered dict)
                    innermost_fqn, innermost_cls = list(nn_module_stack.values())[-1]
                    for src in wanted_sources:
                        if isinstance(src, type):
                            if isinstance(innermost_cls, type) and issubclass(
                                innermost_cls, src
                            ):
                                add_to_partition(src, innermost_fqn, node)
                                break
                            elif isinstance(innermost_cls, str):
                                src_str = src.__module__ + "." + src.__qualname__
                                if innermost_cls == src_str:
                                    add_to_partition(src, innermost_fqn, node)
                                    break
                        elif innermost_cls == src:
                            add_to_partition(src, innermost_fqn, node)
                            break

        if source_fn_st is not None:
            source_fn = source_fn_st[-1]
            if source_fn[1] in wanted_sources:
                add_to_partition(source_fn[1], source_fn[0], node)

    def make_partition(nodes: list[Node], module_type: type) -> SourcePartition:
        input_nodes = set()
        output_nodes = set()
        params = set()
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, Node) and arg not in nodes and arg.op != "get_attr":
                    input_nodes.add(arg)

            if node.op == "get_attr":
                params.add(node)
                # get_attr nodes won't be output nodes
                continue

            for user in node.users:
                if user not in nodes:
                    output_nodes.add(node)

        return SourcePartition(
            nodes,
            module_type,
            sorted(input_nodes),
            sorted(output_nodes),
            sorted(params),  # type: ignore[arg-type]
        )

    ret: dict[type[Any], list[SourcePartition]] = {}

    if filter_fn:
        # for each partition, we apply filter_fn to filter out all partitions that doesn't satisfy the
        # filter condition
        filtered_modules = {}
        for tp, name_to_partition in modules.items():
            filtered_name_to_partition = {
                name: partition
                for name, partition in name_to_partition.items()
                if all(map(filter_fn, partition))
            }
            filtered_modules[tp] = filtered_name_to_partition
        modules = filtered_modules

    for k, v in modules.items():
        ret[k] = [make_partition(partition, k) for partition in v.values()]

    return ret


@compatibility(is_backward_compatible=False)  # type: ignore[misc]
def check_subgraphs_connected(
    subgraph1: SourcePartition, subgraph2: SourcePartition
) -> bool:
    """
    Given two subgraphs A and B (in the form of a list of nodes), checks if
    A has nodes connecting to at least one node in B -- aka there exists a node
    in B that uses a node in A (not the other way around).
    """

    for node in reversed(subgraph1.nodes):
        for user in node.users:
            if user in subgraph2.nodes:
                return True
    return False
<|endoftext|>from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.nn import Module


__all__ = ["HolderModule", "lift_subgraph_as_module", "compare_graphs"]


@compatibility(is_backward_compatible=False)
class HolderModule(Module):
    """
    HolderModule is used to copy all the attributes from original module to submodules
    that uses the attributes
    """

    def __init__(self, d: dict[str, Module | None]) -> None:
        super().__init__()
        for k, v in d.items():
            self.add_module(k, v)


@compatibility(is_backward_compatible=False)
def lift_subgraph_as_module(
    gm: GraphModule,
    subgraph: Graph,
    comp_name: str = "",
    class_name: str = "GraphModule",
) -> tuple[GraphModule, dict[str, str]]:
    """
    Create a GraphModule for subgraph, which copies the necessary attributes
    from the original parent graph_module.

    Args:
        gm (GraphModule): parent graph module

        subgraph (:class:`torch.fx.Graph`): a valid subgraph that contains copied nodes from the
            parent graph

        comp_name (str): name for the new component

        class_name (str): name for the submodule

    """

    # Loop through all module calls (call_module) and param fetches (get_attr)
    # in this component, creating HolderModules as necessary to match the path.
    # e.g. if in the original module there's a get_attr node fetches "conv.weight".
    # We create a HolderModule as root -> add a HolderModule named "conv" ->
    # make "weight" an attribute of "conv" HolderModule and point to conv.weight in
    # the original module.
    submodule = HolderModule({})
    orig_to_split_fqn_mapping: dict[str, str] = {}
    for n in subgraph.nodes:
        if n.op not in ("call_module", "get_attr"):
            continue

        target = n.target
        if not isinstance(target, str):
            raise AssertionError(f"Expected str target, got {type(target)}")
        target_name_parts = target.split(".")
        curr = submodule
        orig_gm = gm

        for name in target_name_parts[:-1]:
            if not hasattr(curr, name):
                curr.add_module(name, HolderModule({}))

            curr = getattr(curr, name)
            orig_gm = getattr(orig_gm, name)

        leaf_node_name = target_name_parts[-1]
        leaf_node = getattr(orig_gm, leaf_node_name)

        orig_to_split_fqn_mapping[target] = f"{comp_name}.{target}"
        # Relies on custom __setattr__ magic.
        setattr(curr, leaf_node_name, leaf_node)

    return GraphModule(submodule, subgraph, class_name), orig_to_split_fqn_mapping


@compatibility(is_backward_compatible=False)
def compare_graphs(left: Graph, right: Graph) -> bool:
    """
    Return True if two graphs are identical, i.e they
        - have the same number of outputs in the same order
        - have the same number of inputs in the same order
        - have the same set of nodes, and identical connectivity
    """

    matcher = SubgraphMatcher(left, match_output=True, match_placeholder=True)
    matches = matcher.match(right)

    return len(matches) > 0
<|endoftext|>import copy
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.fx import Graph, Node
from torch.fx._compatibility import compatibility


__all__ = ["SubgraphMatcher", "InternalMatch"]


# Set`PYTORCH_MATCHER_LOGLEVEL=INFO` to see debug logs
def _init_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)

    level = os.environ.get("PYTORCH_MATCHER_LOGLEVEL", "WARNING").upper()
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter("%(filename)s > %(message)s")
    console.setFormatter(formatter)
    console.setLevel(level)
    # add the handlers to the logger
    logger.addHandler(console)
    logger.propagate = False
    return logger


logger = _init_logger()


@compatibility(is_backward_compatible=False)
@dataclass
class InternalMatch:
    # Nodes from which the match was found
    anchors: list[Node]
    # Maps nodes in the pattern subgraph to nodes in the larger graph
    nodes_map: dict[Node, Node] = field(default_factory=dict)

    # nodes in target graph that are matched placeholder in pattern
    placeholder_nodes: list[Node] = field(default_factory=list)

    # nodes in matched subgraph returned by output
    returning_nodes: list[Node] = field(default_factory=list)

    # map from a string name to a node in the target graph
    # only available if the matcher is `SubgraphMatcherWithNameNodesMap`
    name_node_map: dict[str, Node] = field(default_factory=dict)

    def __copy__(self) -> "InternalMatch":
        return InternalMatch(
            anchors=self.anchors,
            nodes_map=self.nodes_map.copy(),
            placeholder_nodes=self.placeholder_nodes.copy(),
            returning_nodes=self.returning_nodes.copy(),
        )


@compatibility(is_backward_compatible=False)
class SubgraphMatcher:
    def __init__(
        self,
        pattern: Graph,
        match_output: bool = False,
        match_placeholder: bool = False,
        remove_overlapping_matches: bool = True,
        ignore_literals: bool = False,
    ) -> None:
        """
        Args:
            pattern: the targeted matching pattern, represented in fx.Graph.
            match_output: If True, output node in the pattern graph will be treated as a part of the targeted pattern.
                If False, output node is ignored during match.
            match_placeholder: If True, placeholder node in the pattern graph will be treated as a part of
                the targeted pattern. If False, placeholder nodes will be used a wildcard.
            remove_overlapping_matches: If True, in the case of overlapping matches, only the first match
                will be returned.
            ignore_literals: If True, will not check if literals are equal and
                will instead treat them as wildcards.
        """

        self.pattern = pattern
        self.match_output = match_output
        self.match_placeholder = match_placeholder
        self.remove_overlapping_matches = remove_overlapping_matches
        self.ignore_literals = ignore_literals

        if len(pattern.nodes) == 0:
            raise ValueError(
                "SubgraphMatcher cannot be initialized with an empty pattern"
            )

        for node in pattern.nodes:
            if node.op != "output" and not node.is_impure():
                if len(node.users) == 0:
                    raise AssertionError(
                        "SubgraphMatcher cannot be initialized with a pattern with dead code"
                    )

        # TODO: assert pattern is a connected graph

        self.pattern_placeholder_nodes = [
            n for n in pattern.nodes if n.op == "placeholder"
        ]
        output_node = next(iter(reversed(pattern.nodes)))
        # nodes returned by outputs
        self.pattern_returning_nodes: list[Node] = output_node.all_input_nodes

        self.pattern_anchors: list[Node] = []
        if match_output:
            self.pattern_anchors = [output_node]
        else:
            # If a node has output_node as the ONLY user, then this node is a graph sink,
            # and should be matched against as an anchor
            self.pattern_anchors = [
                n for n in output_node.all_input_nodes if len(n.users) == 1
            ]

    def _match_attributes(self, pn: Node, gn: Node) -> bool:
        # Attributes matching is complicated. Right now we only support matching constant tensor
        if not isinstance(pn.target, str):
            raise AssertionError(f"pn.target {pn.target} must be a string.")
        if not isinstance(gn.target, str):
            raise AssertionError(f"gn.target {gn.target} must be a string.")

        if pn.graph.owning_module is None:
            raise AssertionError("pn.graph.owning_module must not be None")
        if gn.graph.owning_module is None:
            raise AssertionError("gn.graph.owning_module must not be None")
        pn_value = torch.fx.graph_module._get_attr(pn.graph.owning_module, pn.target)
        gn_value = torch.fx.graph_module._get_attr(gn.graph.owning_module, gn.target)

        if type(pn_value) is not type(gn_value):
            return False

        # Don't require exact match on tensor values.
        if isinstance(pn_value, torch.Tensor):
            return isinstance(gn_value, torch.Tensor)
        else:
            raise RuntimeError(f"Unsupported type {pn_value} when matching attributes")
        # pyrefly: ignore [unreachable]
        return False

    def _nodes_are_equal(self, pn: Node, gn: Node, node_name_match: str = "") -> bool:
        # if exact match for placeholder is not required, then use placeholder as a wildcard
        if not self.match_placeholder and pn.op == "placeholder":
            return True

        if node_name_match and node_name_match in gn.name:
            return True

        if pn.op == gn.op:
            if pn.op == "placeholder" or pn.op == "output":
                return True
            elif pn.op == "get_attr":
                return self._match_attributes(pn, gn)
            return pn.target == gn.target
        return False

    def _is_contained(self, nodes_map: dict[Node, Node]) -> bool:
        # `lookup` represents all the nodes in `original_graph`
        # that are part of `pattern`

        # Placeholders can be used by other nodes in the graphs
        lookup: dict[Node, Node] = {
            gn: pn for pn, gn in nodes_map.items() if pn.op != "placeholder"
        }

        for gn, pn in lookup.items():
            # nodes returned by output are allowed to be used in other areas of the graph
            if pn in self.pattern_returning_nodes:
                continue

            for user in gn.users:
                # If this node has users that were not in `lookup`, then it must leak out of the
                # pattern subgraph
                if user not in lookup:
                    return False
        return True

    def _remove_overlapping_matches(
        self, matches: list[InternalMatch]
    ) -> list[InternalMatch]:
        non_overlapping_matches: list[InternalMatch] = []
        nodes_matched: set[Node] = set()

        for match in matches:
            found_overlap = False
            for pn, gn in match.nodes_map.items():
                if pn.op not in {"placeholder", "output"} and gn in nodes_matched:
                    found_overlap = True
                    break

            if not found_overlap:
                non_overlapping_matches.append(match)
                for pn, gn in match.nodes_map.items():
                    if pn.op not in {"placeholder", "output"}:
                        nodes_matched.add(gn)
        return non_overlapping_matches

    def _match_literals(self, pn: Any, gn: Any, match: InternalMatch) -> bool:
        if isinstance(pn, Node) and isinstance(gn, Node):
            raise AssertionError("pn and gn cannot both be Node")

        if isinstance(pn, Node) and not isinstance(gn, Node):
            if pn.op == "placeholder":
                # Check if we've already matched these nodes in the current
                # traversal
                if pn in match.nodes_map:
                    return match.nodes_map[pn] == gn

                match.nodes_map[pn] = gn
                return True
            else:
                return False
        elif not isinstance(pn, Node) and isinstance(gn, Node):
            return False
        else:
            return type(gn) is type(pn) and gn == pn

    def _match_nodes(
        self, pn: Node, gn: Node, match: InternalMatch, node_name_match: str = ""
    ) -> bool:
        logger.info("  matching %s to %s", pn, gn)

        if not (isinstance(pn, Node) and isinstance(gn, Node)):
            raise AssertionError(f"pn and gn must be Node, pn: {pn}, gn: {gn}")

        # Check if we've already matched these nodes in the current
        # traversal
        if pn in match.nodes_map:
            return match.nodes_map[pn] == gn

        # TODO: use a more efficient way to check if gn is matched before: two-way dict
        if gn in match.nodes_map.values():
            return False

        if not self._nodes_are_equal(pn, gn, node_name_match):
            return False

        # Optimistically mark `pn` as a match for `gn`, and save a local copy of match
        saved_match = copy.copy(match)
        match.nodes_map[pn] = gn

        # Placeholder is a wildcard and can be matched with any python object
        # (including list/tuple)
        if pn.op == "placeholder":
            return True

        # Recursively traverse upwards to check if `pn` is a true
        # match for `gn`
        match_found = True

        def _match_args(
            args1: list[Any] | tuple[Any, ...], args2: list[Any] | tuple[Any, ...]
        ) -> bool:
            if len(args1) != len(args2):
                return False

            for a1, a2 in zip(args1, args2):
                if isinstance(a1, Node) and isinstance(a2, Node):
                    matched = self._match_nodes(a1, a2, match)
                elif isinstance(a1, (list, tuple)) and isinstance(a2, (list, tuple)):
                    matched = _match_args(a1, a2)
                else:
                    matched = (
                        self._match_literals(a1, a2, match) or self.ignore_literals
                    )

                if not matched:
                    return False

            return True

        # Flatten all args/kwargs into 1 list of args
        pn_args: list[Any] | None = None
        gn_args: list[Any] | None = None
        if (
            (
                len(pn.args) != len(gn.args)
                or list(pn.kwargs.keys()) != list(gn.kwargs.keys())
            )
            and pn.op == "call_function"
            and isinstance(pn.target, torch._ops.OpOverload)
        ):
            args_schema = pn.target._schema.arguments

            def get_all_arguments(
                orig_args: tuple[Any, ...], orig_kwargs: dict[str, Any]
            ) -> list[Any]:
                all_args = []
                for i, schema in enumerate(args_schema):
                    if schema.name in orig_kwargs:
                        all_args.append(orig_kwargs[schema.name])
                    elif not schema.kwarg_only and i < len(orig_args):
                        all_args.append(orig_args[i])
                    else:
                        all_args.append(schema.default_value)
                return all_args

            pn_args = get_all_arguments(pn.args, pn.kwargs)
            gn_args = get_all_arguments(gn.args, gn.kwargs)

        elif len(pn.args) == len(gn.args) and list(pn.kwargs.keys()) == list(
            gn.kwargs.keys()
        ):
            pn_args = list(pn.args)
            gn_args = list(gn.args)
            pn_args.extend(list(pn.kwargs.values()))
            gn_args.extend(list(gn.kwargs.values()))
        else:
            match_found = False

        match_found = (
            match_found
            and pn_args is not None
            and gn_args is not None
            and _match_args(pn_args, gn_args)
        )

        if not match_found:
            # revert to saved_match before matching with current node
            match = copy.copy(saved_match)
            return False

        return True

    def match(self, graph: Graph, node_name_match: str = "") -> list[InternalMatch]:
        """
        Returns:
            The matched subgraphs.
            The returned subgraph would be fully self-contained, meaning the nodes (except placeholder
            and nodes returned by output) can only be consumed by nodes within the matched subgraph.

        Subgraph pattern matcher is implemented with the backtracking style in the following steps:

        1. We first identify all the anchor nodes in the pattern graph. The anchor nodes
        are the "sinks" (nodes with no user other than the output node) of the pattern graph.
        One pattern graph could have multiple anchors if it has multiple return values.

        2. In the target graph, we identify the potential candidate nodes that can be matched
        with each anchor. These anchor-candidate pairs are the starting points for
        pairwise per-node matching.

        3. For each anchor-candidate pair, we simultaneously traverse backwards (DFS) in both
        pattern and target graphs. For every pattern nodes along traversal path, we compare it
        against the target nodes. In case any comparison failed, the match for this anchor-candidate
        pair fails. A match is found when DFS completes traversing the graph. See `self._match_nodes`
        for more details.

        4. In the case of multiple anchors, every anchor will need to find a match using step 3.
        In addition, the matches found between anchors need to have a common intersection node
        in order for the match to be valid. This is implemented with backtracking. See `backtracking`
        for more details.

        Notice: graph traversal must be done in the reverse order because a tensor can have multiple
        consumers, but can only have a single producer. Only with reverse order can we jointly
        traverse the pattern and target graph in a deterministic path.

        Warning: In theory, this backtracking algorithm have an **exponential** time complexity. However,
        in practice, it's unlikely to blow up.

        """
        from torch.fx.passes.utils.fuser_utils import validate_partition

        # find candidate nodes to match with pattern anchors
        match_candidates: dict[Node, list[Node]] = defaultdict(list)
        for pattern_anchor in self.pattern_anchors:
            for node in graph.nodes:
                if self._nodes_are_equal(pattern_anchor, node, node_name_match):
                    match_candidates[pattern_anchor].append(node)
        match_candidates_list = list(match_candidates.items())

        logger.info("Initial match_candidates_list: %s\n", match_candidates_list)

        matches: list[InternalMatch] = []

        def backtracking(anchor_index: int, match: InternalMatch) -> None:
            if anchor_index == len(match_candidates_list):
                match.placeholder_nodes = [
                    match.nodes_map[pn] for pn in self.pattern_placeholder_nodes
                ]
                match.returning_nodes = [
                    match.nodes_map[pn] for pn in self.pattern_returning_nodes
                ]
                matches.append(match)

                logger.info("Found a match: %s\n", match)
                return

            pattern_anchor, candidate_nodes = match_candidates_list[anchor_index]
            saved_match = copy.copy(match)

            for node in candidate_nodes:
                logger.info("Trying to match anchor %s to %s", pattern_anchor, node)

                match_found = self._match_nodes(
                    pattern_anchor, node, match, node_name_match
                )
                if match_found:
                    # match next anchor
                    backtracking(anchor_index + 1, match)
                else:
                    logger.info(
                        "Failed to match anchor %s to %s\n", pattern_anchor, node
                    )

                # revert to saved_match before matching with current anchor
                match = copy.copy(saved_match)

        match = InternalMatch(anchors=self.pattern_anchors)
        if match_candidates_list:
            backtracking(0, match)

        # filter out the matches where the subgraph is not fully_contained
        before = len(matches)
        matches = [match for match in matches if self._is_contained(match.nodes_map)]
        after = len(matches)
        if before != after:
            logger.info(
                "Filtered out %s matches because they are not fully contained",
                before - after,
            )

        # filter out the matches that form a cycle if the subgraph is fused
        valid_matches: list[InternalMatch] = []
        for match in matches:
            matched_compute_nodes = [
                gn
                for pn, gn in match.nodes_map.items()
                if pn.op not in {"placeholder", "output"}
            ]
            if validate_partition(matched_compute_nodes):
                valid_matches.append(match)
        if len(valid_matches) != len(matches):
            logger.info(
                "Filtered out %s matches because \
                          matched subgraph would form a cycle if fused",
                len(matches) - len(valid_matches),
            )

        if self.remove_overlapping_matches:
            before = len(valid_matches)
            matches = self._remove_overlapping_matches(valid_matches)
            after = len(matches)
            if before != after:
                logger.info(
                    "Filtered out %s matches because matched subgraphs are overlapping",
                    before - after,
                )

        logger.info("Matches returned: %s", matches)

        return matches
<|endoftext|>from torch.fx import Graph, GraphModule, Node
from torch.fx._compatibility import compatibility

from .matcher_utils import InternalMatch, SubgraphMatcher


__all__ = ["SubgraphMatcherWithNameNodeMap"]


def _split_to_graph_and_name_node_map(
    gm: GraphModule,
) -> tuple[GraphModule, dict[str, Node]]:
    from torch.fx.graph import _PyTreeInfo
    from torch.utils._pytree import tree_flatten, tree_unflatten

    name_node_map: dict[str, Node] = {}
    for n in gm.graph.nodes:
        if n.op == "output":
            if gm._out_spec is None:
                raise AssertionError("gm._out_spec is None")
            output = tree_unflatten(n.args[0], gm._out_spec)
            if not isinstance(output, tuple):
                raise AssertionError("Expecting the pattern graph to return a tuple")
            if len(output) < 2:
                raise AssertionError(
                    "Expecting the pattern graph to have at least two outputs"
                )
            *out, name_node_map = output
            flattened, out_spec = tree_flatten(out)
            if not isinstance(name_node_map, dict):
                raise AssertionError(
                    "Expecting the input graph to have a dict output as the last element"
                )
            n.args = (flattened,)
            orig_pytree_info = gm._graph._codegen.pytree_info  # type: ignore[attr-defined]
            gm._graph._codegen.pytree_info = _PyTreeInfo(  # type: ignore[attr-defined]
                orig_pytree_info.orig_args, orig_pytree_info.in_spec, out_spec
            )
    gm.recompile()
    return gm, name_node_map


@compatibility(is_backward_compatible=False)
class SubgraphMatcherWithNameNodeMap(SubgraphMatcher):
    """Extends SubgraphMatcher to support querying the matched subgraph nodes through node name,
    this requires pattern to have specific format (returning and additional dictionary at the output,
    that has node name as key, and the node in the pattern graph as value, see Example for more details)

    Difference with SubgraphMatcher is that it takes a `pattern_gm` GraphModule as input during
    initialization since we need to modify the graph (which requires `recompile` the GraphModule)

    Example::

        def pattern(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            return relu, {"conv": conv, "relu": relu}


        def target_graph(x, weight):
            conv = F.conv2d(x, weight)
            relu = F.relu(conv)
            relu *= 2
            return relu


        pattern_gm = export(pattern, example_inputs).module()
        target_gm = export(target_graph, example_inputs).module()
        matcher = SubgraphMatcherWithNameNodeMap(pattern_gm)
        matches = matcher.match(target_gm)
        for match in matches:
            match.name_node_map["conv"].meta["annotation"] = ...

    """

    def __init__(
        self,
        pattern_gm: GraphModule,
        match_output: bool = False,
        match_placeholder: bool = False,
        remove_overlapping_matches: bool = True,
        ignore_literals: bool = False,
    ) -> None:
        pattern_gm, name_node_map = _split_to_graph_and_name_node_map(pattern_gm)
        self.name_node_map = name_node_map
        super().__init__(
            pattern_gm.graph,
            match_output,
            match_placeholder,
            remove_overlapping_matches,
            ignore_literals,
        )

    def match(self, graph: Graph, node_name_match: str = "") -> list[InternalMatch]:
        """The returned InternalMatch will have name_node_map populated with a map
        from node name (str) to the target node, e.g.
        ``{"conv": target_conv_node, "relu": target_relu_node}``

        This requires the pattern graph returns an additional
        output of node name to node, e.g. instead of::

            def pattern(...):
                ...
                return relu

        we should do::

            def pattern(...):
                ...
                return relu, {"conv": conv, "relu": relu}
        """
        internal_matches = super().match(graph, node_name_match)
        for internal_match in internal_matches:
            for k, n in self.name_node_map.items():
                internal_match.name_node_map[k] = internal_match.nodes_map[n]
        return internal_matches
<|endoftext|>import copy
import heapq

import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.passes.tools_common import legalize_graph, NodeList, NodeSet  # noqa: F401
from torch.fx.passes.utils import lift_subgraph_as_module  # type: ignore[attr-defined]


@compatibility(is_backward_compatible=False)
def topo_sort(nodes: NodeList) -> NodeList:
    # Stable topological sort: among nodes with no dependency between them,
    # preserve their relative order in the input list. This uses a min-heap
    # keyed by original position instead of a FIFO queue.
    indegree_map = dict.fromkeys(nodes, 0)
    position = {node: i for i, node in enumerate(nodes)}
    candidates: list[tuple[int, Node]] = []

    for node in nodes:
        for n in node.all_input_nodes:
            if n in indegree_map:
                indegree_map[node] += 1
        if indegree_map[node] == 0:
            heapq.heappush(candidates, (position[node], node))

    sorted_nodes: NodeList = []
    while candidates:
        _, node = heapq.heappop(candidates)
        sorted_nodes.append(node)

        for n in node.users:
            if n in indegree_map:
                indegree_map[n] -= 1
                if indegree_map[n] == 0:
                    heapq.heappush(candidates, (position[n], n))

    if len(nodes) != len(sorted_nodes):
        raise AssertionError(
            "topological sorted nodes doesn't have same length as input nodes"
        )

    return sorted_nodes


@compatibility(is_backward_compatible=False)
def validate_partition(partition: NodeList) -> bool:
    # verify the partition doesn't form a dependency cycle in the original graph
    # returns True for valid partition, False for invalid

    partition_set = set(partition)

    outputs: NodeList = []
    for node in partition_set:
        for user_node in node.users:
            if user_node not in partition_set:
                # external user node, need to expose as an output
                outputs.append(user_node)

    # Perform BFS on the partition outputs.
    # If it reaches a node within the partition, then it found a cycle.
    # This function takes the ownership of `root_nodes` and may modify it.
    def bfs_find_cycle(root_nodes: NodeList) -> bool:
        # Set used to exclude nodes that have already been visited.
        # If a node has been visited, that node and all its children have
        # been checked for cycles.
        visited: NodeSet = set()

        # Start with `root_nodes` and traverse through (toward child nodes)
        # their connected sub-graph. Nodes in `visited` won't be added
        # to `queue` again.
        queue: NodeList = root_nodes
        while queue:
            current = queue.pop()
            visited.add(current)
            if current in partition_set:
                # Started from partition's `output` nodes, and reached
                # another node in partition. Cycle!
                return True
            for user_node in current.users:
                if user_node in visited:
                    continue
                queue.append(user_node)
        # `root_nodes` don't cause cycle.
        return False

    # Use all output nodes as roots to traverse
    # the graph to check cycles.
    if bfs_find_cycle(outputs):
        return False

    return True


@compatibility(is_backward_compatible=False)
def fuse_as_graphmodule(
    gm: GraphModule,
    nodes: NodeList,
    module_name: str,
    partition_lookup_table: dict[Node, int | None] | None = None,
    *,
    always_return_tuple: bool = False,
) -> tuple[GraphModule, tuple[Node, ...], tuple[Node, ...]]:
    """
    Fuse nodes in graph_module into a GraphModule.

    Args:
        gm (GraphModule): target graph_module

        nodes (List[Node]): list of nodes in `gm` to fuse, where the node must be topologically sorted

        module_name: class name for the fused GraphModule

        partition_lookup_table (Optional[Dict[Node, None]]): optional dict of nodes to speed up lookup

        always_return_tuple (bool): whether to always return a tuple, even if there is only one output

    Returns:
        fused_gm (GraphModule): fused graph module, where its node is a copy of `nodes` in `gm`

        original_inputs (Tuple[Node, ...]): input nodes to `nodes` in original `gm`

        original_outputs (Tuple[Node, ...]): consumer nodes of `nodes` in original `gm`

    """

    # assumption: nodes are already sorted in topo order

    for node in nodes:
        if node.graph.owning_module is not gm:
            raise AssertionError(
                f"{node} doesn't belong to passed in graph module {gm._get_name()}"
            )
        if node._erased:
            raise AssertionError(f"{node} has been removed from owning graph")
        if node not in gm.graph._find_nodes_lookup_table:
            raise AssertionError(
                f"{node} is not found in graph module {gm._get_name()}"
            )

    # validates partition doesn't introduce dependency circles in the graph
    if not validate_partition(nodes):
        raise AssertionError("Invalid partition, found dependency cycles")

    # if no dict of partition nodes is provided, reconstruct it by nodes list to reduce lookup time
    if partition_lookup_table is None:
        partition_lookup_table = dict.fromkeys(nodes)

    subgraph = Graph()

    node_to_placeholder: dict[
        Node, Node
    ] = {}  # mapping of nodes from old graph to placeholder in new graph
    node_map: dict[Node, Node] = {}  # mapping of nodes from old graph to new graph

    def create_placeholder(input_node: Node) -> None:
        placeholder_node = subgraph.placeholder(
            input_node.name, type_expr=input_node.type
        )
        # copy all meta fields, even if some fields might be irrelevant for the placeholder node
        placeholder_node.meta = copy.copy(input_node.meta)
        node_to_placeholder[input_node] = placeholder_node

    external_inputs: list[Node] = []
    external_inputs_set: set[Node] = set()
    for node in nodes:
        for input_node in node.all_input_nodes:
            if (
                input_node not in partition_lookup_table
                and input_node not in external_inputs_set
            ):
                external_inputs.append(input_node)
                external_inputs_set.add(input_node)

    if external_inputs and all(
        input_node.op == "placeholder" for input_node in external_inputs
    ):
        # True graph placeholders have a stable original input order. Keep the
        # historical encounter order for mixed or intermediate boundaries,
        # because downstream lowering may derive its own call args from the
        # submodule placeholder order.
        graph_order = {node: i for i, node in enumerate(gm.graph.nodes)}
        external_inputs.sort(key=lambda node: graph_order[node])

    for input_node in external_inputs:
        create_placeholder(input_node)

    # handles inputs through graph.node_copy's arg_transform functions
    def remap_inputs(x: Node) -> Node:
        if x.op == "get_attr":
            # TODO: do we really need copy the get_attr node into the graph?
            # do something here
            pass

        if x in partition_lookup_table:
            # x is inside subgraph, return the copied node
            # the node should have been copied already, as we are copying graph in the topological order
            return node_map[x]

        return node_to_placeholder[x]

    # copy nodes in topological order
    for node in nodes:
        new_node = subgraph.node_copy(node, remap_inputs)
        node_map[node] = new_node

    # handles outputs
    output_mapping: dict[Node, Node] = {}  # mapping from old output to new outputs

    for node in nodes:
        for user_node in node.users:
            if user_node not in partition_lookup_table:
                # external user node, need to expose as an output
                output_mapping[node] = node_map[node]

    # outs contain nodes in the new subgraph
    outs = tuple(output_mapping.values())

    if always_return_tuple:
        # always return a tuple, even if there is only one output
        subgraph.output(outs)
    else:
        # If there's a single output then return it directly, otherwise return a tuple.
        subgraph.output(outs[0] if len(outs) == 1 else outs)

    # lint to ensure correctness
    subgraph.lint()  # type: ignore[no-untyped-call]
    fused_gm: GraphModule
    fused_gm, _ = lift_subgraph_as_module(
        gm, subgraph, comp_name="", class_name=module_name
    )

    # sub_gm's input nodes in the original module
    original_inputs: tuple[Node, ...] = tuple(node_to_placeholder.keys())

    # sub_gm's outputs node in the original module
    original_outputs: tuple[Node, ...] = tuple(output_mapping.keys())

    return fused_gm, original_inputs, original_outputs


@compatibility(is_backward_compatible=False)
def insert_subgm(
    gm: GraphModule,
    sub_gm: GraphModule,
    orig_inputs: tuple[Node, ...],
    orig_outputs: tuple[Node, ...],
    insertion_point: Node | None = None,
) -> GraphModule:
    # add sub_gm into gm
    submodule_name = sub_gm.__class__.__name__
    gm.add_submodule(submodule_name, sub_gm)

    # Use provided insertion point, or fall back to last output node for backwards compat
    if insertion_point is None:
        for node in reversed(gm.graph.nodes):
            if node in orig_outputs:
                insertion_point = node
                break
        if insertion_point is None:
            raise AssertionError(
                "Cannot determine insertion point: no insertion_point provided and "
                "orig_outputs is empty. Pass the last partition node as insertion_point."
            )

    # Create a call_module node in main graph.
    with gm.graph.inserting_after(insertion_point):
        module_node = gm.graph.call_module(
            submodule_name, args=orig_inputs, kwargs=None
        )
        output_node = sub_gm.graph.output_node()

    # Replace uses of original outputs with the fused module outputs.
    # If there are no external outputs, skip replacement (nothing to replace).
    if orig_outputs:
        next_node = module_node.next
        with gm.graph.inserting_before(next_node):
            if len(orig_outputs) == 1 and not isinstance(output_node.args[0], tuple):
                # main_remapping[comp.orig_outputs[0]] = module_node
                orig_outputs[0].replace_all_uses_with(module_node, propagate_meta=True)
            else:
                for i, orig_output in enumerate(orig_outputs):
                    # Use Proxy to record getitem access.
                    proxy_out = torch.fx.Proxy(module_node)[i].node  # type: ignore[index]
                    orig_output.replace_all_uses_with(proxy_out, propagate_meta=True)

                module_node.meta["val"] = tuple(
                    orig_output.meta.get("val", None) for orig_output in orig_outputs
                )
    return gm


@compatibility(is_backward_compatible=False)
def erase_nodes(gm: GraphModule, nodes: NodeList) -> None:
    # erase original nodes in inverse topological order
    for node in reversed(nodes):
        gm.graph.erase_node(node)


@compatibility(is_backward_compatible=False)
def fuse_by_partitions(
    gm: GraphModule,
    partitions: list[dict[Node, int | None]],
    prefix: str = "fused_",
    always_return_tuple: bool = False,
) -> GraphModule:
    for partition_id, partition in enumerate(partitions):
        sorted_nodes = topo_sort(list(partition))

        submodule_name = prefix + str(partition_id)
        sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(
            gm,
            sorted_nodes,
            submodule_name,
            partition,
            always_return_tuple=always_return_tuple,
        )

        insert_subgm(gm, sub_gm, orig_inputs, orig_outputs, sorted_nodes[-1])

        erase_nodes(gm, sorted_nodes)

    torch.fx.passes.tools_common.stable_topological_sort(gm)
    gm.graph.lint()

    return gm
<|endoftext|>import operator
from collections.abc import Mapping, Sequence
from typing import Any

import torch


__all__ = ["CudaGraphsSupport", "partition_cudagraphs"]
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree as pytree


class CudaGraphsSupport(OperatorSupport):
    # TODO: why is submodules passed here
    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        if node.op not in CALLABLE_NODE_OPS:
            return False

        if node.target is torch.ops.aten.embedding_dense_backward.default:
            return False

        if node.target is operator.getitem:
            return True

        found_not_cuda = False

        def meta_fk(meta: dict[str, Any]) -> torch.Tensor:
            return meta["val"] if "val" in meta else meta["fake_result"]

        def find_not_cuda(t: object) -> None:
            nonlocal found_not_cuda
            if isinstance(t, torch.Tensor) and t.device.type != "cuda":
                found_not_cuda = True

        for n in node.all_input_nodes:
            pytree.tree_map_(find_not_cuda, meta_fk(n.meta))

        pytree.tree_map_(find_not_cuda, meta_fk(node.meta))

        # NB: factory function is accounted for because the result would be
        # cpu or cuda

        return not found_not_cuda


def partition_cudagraphs(
    gm: torch.fx.GraphModule, inputs: Sequence[object]
) -> torch.fx.GraphModule:
    """
    Partition an FX graph into sub-GraphModules that can be validly run under
    CUDA graphs.  For a subgraph to be runnable under CUDA, all of the operations
    must involve CUDA tensors only.
    """

    FakeTensorProp(gm).propagate(*inputs)
    supported_ops = CudaGraphsSupport()
    # TODO: single node partition may be wrong due to the pessimization
    # from copying in and out the data.  Check in benchmarks, perhaps
    partitioner = CapabilityBasedPartitioner(
        gm, supported_ops, allows_single_node_partition=True
    )
    partitions = partitioner.propose_partitions()
    fused_graph = partitioner.fuse_partitions(partitions)
    return fused_graph
<|endoftext|>from collections.abc import Sequence

import torch.fx as fx


__all__ = ["set_trace"]


def set_trace(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Sets a breakpoint in `gm`'s generated python code. It drops into pdb when
    `gm` gets run.

    Args:
        gm: graph module to insert breakpoint. It is then recompiled for it to
            take effect.

    Returns:
        the `gm` with breakpoint inserted.
    """

    def insert_pdb(body: Sequence[str]) -> list[str]:
        return ["import pdb; pdb.set_trace()\n", *body]

    with gm.graph.on_generate_code(
        make_transformer=lambda cur_transform: (
            # new code transformer to register
            lambda body: (insert_pdb(cur_transform(body) if cur_transform else body))
        )
    ):
        gm.recompile()

    return gm
<|endoftext|>import builtins
import functools
import logging
import math
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, overload, TYPE_CHECKING, TypeVar

import sympy

import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch._dynamo.exc import TorchDynamoException
from torch._dynamo.utils import dynamo_timed
from torch.fx.node import Argument, Target
from torch.utils._sympy.interp import sympy_interp


if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, TrackedFake


log = logging.getLogger(__name__)

_R = TypeVar("_R")

try:
    import z3  # type: ignore[import]

    # Translation Validation for Dynamo guards
    # ========================================
    #
    # Checks whether optimizations applied to the collected guards are
    # valid. In other words, whether the guard function we actually run
    # does not have false positives (unsound).
    #
    # In order to do so, we build the guards using 2 different information
    # attached to each 'SymNode':
    #   1. SymPy expressions
    #   2. FX nodes
    #
    # SymPy expressions have implicit optimizations baked within itself,
    # which may have a few bugs. On the other hand, we build the FX graph
    # manually, with no optimizations enabled. This gives us access to
    # the "ground truth".
    #
    # We then convert into Z3 expressions both the SymPy expressions
    # (see [Note: SympyToZ3]) that reach 'ShapeEnv.produce_guards' function
    # and the FX nodes (see [Note: PopulateValidator]) that go through
    # 'ShapeEnv.evaluate_expr' function. Finally, we run the validation.
    # (see [Note: TranslationValidator])
    # Better Z3 to string implementation (for a small fraction of Z3).
    #
    # Here are the things we clean before showing the Z3 expression:
    #   - Rename a few ops (e.g. "Distinct" ==> "!=")
    #
    #   - Ignore ToInt and ToReal operations:
    #     usually they don't really matter
    #
    #   - Transform (ToInt (/ ...)) into (idiv ...):
    #     this is the pattern for floor division
    #
    #   - Collect a chain of the same operations into one
    def z3str(e: z3.ExprRef) -> str:
        if not z3.is_expr(e):
            raise AssertionError(f"unsupported expression type: {e}")

        def get_args_str(e: z3.ExprRef) -> list[str]:
            return [z3str(e.arg(i)) for i in range(e.num_args())]

        # First, we simplify the given expression.
        # This is done using rewriting rules, so shouldn't take long.
        e = z3.simplify(e)

        # Only support function applications.
        # Even Z3 "variables" are, in fact, function applications.
        if not z3.is_app(e):
            raise ValueError(f"can't print Z3 expression: {e}")

        if z3.is_int_value(e) or z3.is_rational_value(e):
            return e.as_string()  # type: ignore[attr-defined]

        decl = e.decl()
        kind = decl.kind()
        op = str(decl)
        args = get_args_str(e)

        if kind == z3.Z3_OP_POWER:
            op = "pow"

        elif kind in (z3.Z3_OP_ADD, z3.Z3_OP_MUL):
            # Collect the arguments of chains of ADD and MUL.
            # This is safe, since they are associative.

            def collect_str_args(e: z3.ExprRef) -> list[str]:
                if not (z3.is_app(e) and e.decl().kind() == kind):
                    return [z3str(e)]
                else:
                    return [
                        x
                        for i in range(e.num_args())
                        for x in collect_str_args(e.arg(i))
                    ]

            args = collect_str_args(e)

        elif kind == z3.Z3_OP_NOT:
            # Revert some conversions that z3.simplify applies:
            #   - a != b ==> (Not (== a b)) ==> (!= a b)
            #   - a < b ==> (Not (<= b a)) ==> (> b a)
            #   - a > b ==> (Not (<= a b)) ==> (> a b)

            if e.num_args() != 1:
                raise AssertionError(f"Expected 1 arg, got {e.num_args()}")
            arg = e.arg(0)

            if not z3.is_app(arg):
                raise AssertionError("Expected z3 app")
            argkind = arg.decl().kind()

            logic_inverse = {
                z3.Z3_OP_EQ: "!=",
                z3.Z3_OP_LE: ">",
                z3.Z3_OP_GE: "<",
            }

            if argkind in logic_inverse:
                op = logic_inverse[argkind]
                args = get_args_str(arg)

        elif kind in (z3.Z3_OP_TO_INT, z3.Z3_OP_TO_REAL):
            if e.num_args() != 1:
                raise AssertionError(f"Expected 1 arg, got {e.num_args()}")
            argstr = z3str(e.arg(0))

            # Check if it's the floor division pattern.
            if argstr.startswith("(/"):
                return "(idiv" + argstr[2:]

            # Otherwise, just ignore it.
            return argstr

        elif kind == z3.Z3_OP_UNINTERPRETED:
            if e.num_args() != 0:
                raise AssertionError(f"Expected 0 args, got {e.num_args()}")
            return str(decl)

        string = op + " " + " ".join(args)
        return f"({string.rstrip()})"

    # We need to convert to/from BitVec in order to use z3 bitwise ops.
    # We assume that integers are 64 bit.
    # If all args are boolean, then use the boolean bitwise op implementation instead, if provided.
    def _bitwise_op(
        bitwise_func: Callable[..., Any], bool_func: Callable[..., Any] | None
    ) -> Callable[..., Any]:
        @functools.wraps(bitwise_func)
        def wrapper(self: "_Z3Ops", *args: z3.ExprRef) -> Any:
            if bool_func is not None and all(
                isinstance(arg, z3.BoolRef) for arg in args
            ):
                return bool_func(*args)

            wrapped_args = tuple(z3.Int2BV(a, 64) for a in args)
            return z3.BV2Int(bitwise_func(*wrapped_args))

        return wrapper

    # Implementation of Python semantics as Z3 expressions.
    #
    # Z3 Real-Int theory has operators with semantics that differ from that of
    # Python. Therefore, in order to get it right, we need to implement
    # the (Python) semantics we are relying on in Z3.
    @dataclass
    class _Z3Ops:
        # Validator used for adding assertions as needed.
        # e.g. div(a, b) requires b != 0.
        validator: "TranslationValidator"

        # The 2 functions below are used for conditionally casting between
        # integer and reals.
        #
        # Returns a real expression from 'x'.
        @staticmethod
        def to_real(x: z3.ArithRef) -> z3.ArithRef:
            return x if x.is_real() else z3.ToReal(x)

        # Returns an integer expression from 'x'.
        @staticmethod
        def to_int(x: z3.ArithRef) -> z3.ArithRef:
            return x if x.is_int() else z3.ToInt(x)

        def sym_sum(self, args: z3.ArithRef) -> z3.ArithRef:
            return sum(args)  # pyrefly: ignore [no-matching-overload]

        # Implements Python division semantics.
        def div(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
            self.validator.add_assertion(denominator != 0)  # type: ignore[arg-type]
            return _Z3Ops.to_real(numerator) / _Z3Ops.to_real(denominator)

        def floor(self, number: z3.ArithRef) -> z3.ArithRef:
            # Z3 ToInt function rounds a real number towards negative infinity.
            return _Z3Ops.to_int(number)

        # Python semantics for 'FloorDiv' states that before applying the floor
        # function, the operands are converted to their common type.
        def floordiv(
            self, numerator: z3.ArithRef, denominator: z3.ArithRef
        ) -> z3.ArithRef:
            cast_result_to_real = numerator.is_real() or denominator.is_real()
            result = _Z3Ops.to_int(self.div(numerator, denominator))
            # Since the 'result' is already an integer, we just have to check
            # whether we should cast it to real.
            return _Z3Ops.to_real(result) if cast_result_to_real else result

        def ceil(self, number: z3.ArithRef) -> z3.ArithRef:
            return z3.If(self.floor(number) < number, self.floor(number + 1), number)  # type: ignore[return-value]

        def trunc(self, number: z3.ArithRef) -> z3.ArithRef:
            return z3.If(number >= 0, self.floor(number), self.ceil(number))  # type: ignore[return-value]

        def max(self, a: z3.ArithRef, b: z3.ArithRef) -> z3.ArithRef:
            return z3.If(a > b, a, b)  # type: ignore[return-value]

        def min(self, a: z3.ArithRef, b: z3.ArithRef) -> z3.ArithRef:
            return z3.If(a < b, a, b)  # type: ignore[return-value]

        # Python semantics for 'Mod' is defined as: p % q = p - floordiv(p, q) * q
        # It should work with both integer and reals.
        def mod(self, p: z3.ArithRef, q: z3.ArithRef) -> z3.ArithRef:
            return p - self.floordiv(p, q) * q

        def pow(self, base: z3.ArithRef, exp: z3.ArithRef) -> z3.ArithRef:
            # Z3 can't handle complex numbers very well.
            self.validator.add_assertion(z3.Or(base != 0, exp > 0))  # type: ignore[arg-type]
            return base**exp

        def sqrt(self, number: z3.ArithRef) -> z3.ArithRef:
            # Square-root:
            # 1. Only work with reals
            number = _Z3Ops.to_real(number)
            # 2. The number should be positive or zero.
            #    Otherwise, Z3 returns 'unknown'.
            self.validator.add_assertion(number >= 0)
            return number**0.5

        def abs(self, number: z3.ArithRef) -> z3.ArithRef:
            return z3.Abs(number)

        def round_to_int(self, number: z3.ArithRef) -> z3.ArithRef:
            # Python's builtin 'round' implements the 'round half to even' strategy
            # See https://en.wikipedia.org/wiki/Rounding#Rounding_half_to_even
            # z3 has an equivalent z3.fpRoundToIntegral(z3.RoundNearestTiesToEven(), ...), but this only applies to
            # floating point numbers, which is different from real numbers that we are dealing with here.
            # Instead, we implement 'round half to even' in terms of 'round half up' (floor(x + 0.5)) and
            # 'round half down' (ceil(x - 0.5)).
            # Assuming 'round half up' is the default case, we need to correct ..., -3.5, -1.5, 0.5, 2.5, 4.5, ...
            # to round down, i.e. use the 'round half down' strategy
            return z3.If(
                self.mod(number, z3.IntVal(2)) == 0.5,
                self.ceil(number - 0.5),
                self.floor(number + 0.5),
            )

        bitwise_and = _bitwise_op(operator.and_, z3.And)
        bitwise_or = _bitwise_op(operator.or_, z3.Or)
        lshift = _bitwise_op(operator.lshift, None)
        rshift = _bitwise_op(operator.rshift, None)

    # Lifts a callable to be used in Z3.
    #
    # This function replaces the given 'op' by a function that:
    #
    #   1. Lifts the arguments into Z3 (i.e. make them inhabitants of Z3)
    #
    #   2. Calls an operation that corresponds to 'op', but works with Z3
    #      inhabitants (left as is if it works as is)
    def z3op(
        op: Callable[..., Any], validator: "TranslationValidator"
    ) -> Callable[..., Any]:
        # Operations that have booleans as their argument.
        # This is needed because the argument of some FX nodes were
        # literal integers, instead of booleans. So, whenever this flag
        # is set, we also convert ints to booleans.
        boolean_ops = {operator.not_, torch.sym_not}
        as_bool = op in boolean_ops

        # Lifts the function into 'z3.ExprRef' domain.
        def lift(func: Callable[..., _R]) -> Callable[..., _R]:
            def wrap(a: object) -> z3.ExprRef:
                if isinstance(a, (z3.ArithRef, z3.BoolRef)):
                    return a
                # Convert it into a Z3 value, if it is some of the supported
                # types below.
                if isinstance(a, bool) or (as_bool and isinstance(a, int)):
                    return z3.BoolVal(bool(a))
                if isinstance(a, (int, sympy.Integer)):
                    return z3.IntVal(int(a))
                if isinstance(a, (float, sympy.Float)):
                    return z3.RealVal(float(a))
                raise ValueError(f"can't lift type: {type(a)}")

            @functools.wraps(func)
            def wrapper(*args: object) -> Any:
                # Lifts the arguments into a list of Z3 inhabitants.
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    wrapped_args = (tuple(wrap(a) for a in args[0]),)
                else:
                    wrapped_args = tuple(wrap(a) for a in args)
                # Run the function on the Z3 expressions.
                return func(*wrapped_args)

            return wrapper

        ops = _Z3Ops(validator)
        replacement_map = {
            # Operator module.
            operator.not_: lift(z3.Not),
            operator.and_: lift(ops.bitwise_and),
            operator.or_: lift(ops.bitwise_or),
            operator.lshift: lift(ops.lshift),
            operator.rshift: lift(ops.rshift),
            operator.floordiv: lift(ops.floordiv),
            operator.truediv: lift(ops.div),
            operator.mod: lift(ops.mod),
            operator.abs: lift(ops.abs),
            builtins.round: lift(ops.round_to_int),
            # Math module.
            math.ceil: lift(ops.ceil),
            math.floor: lift(ops.floor),
            math.trunc: lift(ops.trunc),
            # Torch module.
            torch.sym_float: lift(ops.to_real),
            torch.sym_max: lift(ops.max),
            torch.sym_min: lift(ops.min),
            torch.sym_not: lift(z3.Not),
            torch.sym_sum: lift(ops.sym_sum),
            torch.sym_ite: lift(lambda b, t, f: z3.If(b, t, f)),
            torch._sym_sqrt: lift(ops.sqrt),  # type: ignore[attr-defined]
            # Not lifted because we only use this function as a
            # marker for adding the expression as validator input.
            torch._assert: torch._assert,
        }
        return replacement_map[op] if op in replacement_map else lift(op)

    # Processes an FX graph, populating the given validator.
    #
    # [Note: PopulateValidator]
    # This class walks through each node in the FX graph, translating
    # them into the Z3 world.
    #
    # Then, whenever it finds an 'torch._assert' call_function operation,
    # it adds the Z3 expression corresponding to the argument as validator
    # input.
    class PopulateValidator(torch.fx.Interpreter):
        def __init__(
            self, graph: torch.fx.Graph, validator: "TranslationValidator"
        ) -> None:
            # Reference to the translation validator.
            self.validator = validator

            # Build the graph module and call `Interpreter` constructor.
            module = torch.fx.GraphModule(root={}, graph=graph)
            super().__init__(module, garbage_collect_values=True)

        def placeholder(
            self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
        ) -> Any:
            symbol = fx_traceback.get_current_meta()["symbol"]
            return self.validator.z3var(symbol)

        def call_function(
            self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
        ) -> Any:
            if target is not torch._assert:
                # Lift and runs the node target function
                return super().call_function(z3op(target, self.validator), args, kwargs)  # type: ignore[arg-type]
            # Adds the Z3 expression corresponding to the first argument
            # as a validator input.
            if len(args) != 1:
                raise AssertionError(
                    f"expected 1 argument on assertion. Got: {len(args)} "
                )
            self.validator.add_source_expr(args[0])  # type: ignore[arg-type]

    # Translates SymPy expressions into Z3 expressions.
    #
    # [Note: SympyToZ3]
    # At the time of the translation, all free variables present in the
    # SymPy expression being translated must be already mapped to a Z3
    # integer variable.
    class SympyToZ3:
        OPERATOR_HANDLES = {"add", "mul", "eq", "ne", "lt", "gt", "le", "ge"}

        def __init__(
            self,
            validator: "TranslationValidator",
        ) -> None:
            self._validator = validator
            self._ops = _Z3Ops(self._validator)

        def constant(self, value: int | float | bool, dtype: torch.dtype) -> z3.ExprRef:
            # TODO: Probably OK to relax this and allow lower precision
            if dtype is torch.int64:
                return z3.IntVal(int(value))
            if dtype is torch.double:
                return z3.RealVal(float(value))
            if dtype is torch.bool:
                return z3.BoolVal(bool(value))
            raise ValueError(f"unsupported dtype (SympyToZ3): {dtype}")

        def to_dtype(self, x: z3.ArithRef, dtype: torch.dtype) -> z3.ArithRef:
            if dtype == torch.float64:
                return z3.ToReal(x)
            raise NotImplementedError(f"to_dtype {dtype} NYI")

        def trunc_to_int(self, x: z3.ArithRef, dtype: torch.dtype) -> z3.ArithRef:
            return z3.ToInt(x)

        def round_to_int(self, x: z3.ArithRef, dtype: torch.dtype) -> z3.ArithRef:
            return self._ops.round_to_int(x)

        def int_truediv(
            self, numerator: z3.ArithRef, denominator: z3.ArithRef
        ) -> z3.ArithRef:
            return self._ops.div(numerator, denominator)

        def truediv(
            self, numerator: z3.ArithRef, denominator: z3.ArithRef
        ) -> z3.ArithRef:
            return self._ops.div(numerator, denominator)

        def floordiv(
            self, numerator: z3.ArithRef, denominator: z3.ArithRef
        ) -> z3.ArithRef:
            return self._ops.floordiv(numerator, denominator)

        def div(self, numerator: z3.ArithRef, denominator: z3.ArithRef) -> z3.ArithRef:
            return self._ops.floordiv(numerator, denominator)

        def pow(self, base: z3.ArithRef, exp: z3.ArithRef) -> z3.ArithRef:
            return self._ops.pow(base, exp)

        def pow_by_natural(self, base: z3.ArithRef, exp: z3.ArithRef) -> z3.ArithRef:
            return self._ops.pow(base, exp)

        def mod(self, p: z3.ArithRef, q: z3.ArithRef) -> z3.ArithRef:
            return self._ops.mod(p, q)

        def python_mod(self, p: z3.ArithRef, q: z3.ArithRef) -> z3.ArithRef:
            return self._ops.mod(p, q)

        def ceil_to_int(self, x: z3.ArithRef, dtype: torch.dtype) -> z3.ArithRef:
            return self._ops.ceil(x)

        def floor_to_int(self, x: z3.ArithRef, dtype: torch.dtype) -> z3.ArithRef:
            return self._ops.floor(x)

        def __getattr__(self, name: str) -> Any:
            REPLACEMENT = {
                "and_": z3.And,
                "or_": z3.Or,
                "not_": z3.Not,
                "bitwise_and": self._ops.bitwise_and,
                "bitwise_or": self._ops.bitwise_or,
                "lshift": self._ops.lshift,
                "rshift": self._ops.rshift,
                "floor": self._ops.floor,
                "ceil": self._ops.ceil,
                "minimum": self._ops.min,
                "maximum": self._ops.max,
            }

            if name in REPLACEMENT:
                return REPLACEMENT[name]
            if name in self.OPERATOR_HANDLES:
                return getattr(operator, name)
            raise AttributeError(f"unhandled operator: {name}")

        def run(self, expr: sympy.Basic) -> z3.ExprRef:
            return sympy_interp(self, self._validator.symbols, expr)  # type: ignore[arg-type]

    # Dynamo guards translation validator.
    #
    # [Note: TranslationValidator]
    # Verifies whether the guards issued by 'ShapeEnv.produce_guards' are sound.
    # That is: whether those (target) guards only yield TRUE whenever the original,
    # unoptimized, (source) guards yield TRUE.
    #
    # More concretely, given 'source' and 'target' guard expressions, we wish to
    # check whether the following expression holds:
    #
    # Not(And(source)) AND And(target)
    #
    # i.e. whether there is an assignment of the free variables where the opposite
    # happens: target is TRUE, but source is FALSE.
    class TranslationValidator:
        def __init__(self) -> None:
            log.debug("new instance")

            # Mapping of SymPy symbols to Z3 variables.
            self.symbols: dict[sympy.Symbol, z3.ExprRef] = {}

            # Set of source Z3 expressions.
            # They represent the generated guards without any kind of
            # simplification or transformation.
            self._source_exprs: set[z3.BoolRef] = set()

            # Set of target Z3 expressions.
            # They represent the actual checked guards at runtime. They might
            # be simplified or transformed versions of the source guards.
            self._target_exprs: set[z3.BoolRef] = set()

            # Set of Z3 expressions representing assertions over both the
            # source and target expressions.
            self._assertions: set[z3.BoolRef] = set()

        # Retrieves the corresponding Z3 variable.
        def z3var(self, symbol: sympy.Symbol) -> z3.ExprRef:
            if symbol not in self.symbols:
                raise AssertionError(f"Z3 variable not found for: {symbol}")
            return self.symbols[symbol]

        # Create a variable in Z3 of 'type' for 'symbol', if it doesn't already exist.
        def add_var(self, symbol: sympy.Symbol, type: type) -> z3.ExprRef:
            if symbol in self.symbols:
                return self.symbols[symbol]

            log.debug("new variable: %s (%s)", symbol.name, type.__name__)

            if type is int:
                var = z3.Int(symbol.name)

                # If 'symbol' is positive (SymPy assumption), we have to
                # convey it to Z3 as well.
                if symbol.is_positive:  # type: ignore[attr-defined]
                    self._target_exprs.add(var > 0)
            elif type is float:
                var = z3.Real(symbol.name)
            elif type is bool:
                var = z3.Bool(symbol.name)
            else:
                raise RuntimeError(f"unsupported type for Z3 variable: {type}")

            self.symbols[symbol] = var
            return var

        # Checks whether all symbols were already added.
        def _check_freesymbols(self, e: sympy.Basic) -> None:
            for s in e.free_symbols:
                if not isinstance(s, sympy.Symbol):
                    raise AssertionError(f"Expected sympy.Symbol, got {type(s)}")
                # Call 'z3var' just to check whether there's already a
                # Z3 variable corresponding to 's'.
                self.z3var(s)

        def to_z3_boolean_expr(self, e: sympy.Basic) -> z3.BoolRef:
            z3expr = SympyToZ3(self).run(e)
            if not isinstance(z3expr, z3.BoolRef):
                raise AssertionError(f"expected boolean expression. Got: {z3expr}")
            return z3expr

        def add_source_expr(self, e: z3.BoolRef) -> None:
            if e not in self._source_exprs:
                log.debug("add source guard: %s", z3str(e))
            self._source_exprs.add(e)

        def add_target_expr(self, e: "sympy.logic.boolalg.Boolean") -> None:
            self._check_freesymbols(e)
            z3expr = self.to_z3_boolean_expr(e)
            if e not in self._target_exprs:
                log.debug("add target guard: %s", z3str(z3expr))
            self._target_exprs.add(z3expr)

        def add_assertion(self, e: z3.BoolRef | sympy.Basic) -> None:
            if isinstance(e, sympy.Basic):
                self._check_freesymbols(e)
                ref = self.to_z3_boolean_expr(e)
            else:
                ref = e
            if not isinstance(ref, z3.BoolRef):
                raise AssertionError(f"Expected z3.BoolRef, got {type(ref)}")
            if ref not in self._assertions:
                log.debug("add assertion: %s", z3str(ref))
            self._assertions.add(ref)

        def validate(self) -> None:
            with dynamo_timed("TranslationValidator.validate"):
                return self._validate()

        def _validate(self) -> None:
            if len(self._source_exprs) == 0 or len(self._target_exprs) == 0:
                # If there are no source/target expressions, there's nothing we really
                # wish to prove. So, we just return.
                return None

            # Here, we use "QF_NRA" logic for the solver:
            #   "Quantifier-free Non-linear Real Arithmetic".
            #
            # Most of the guards expressions have:
            #   1. arithmetic between integer and reals
            #   2. no quantifiers
            #   3. potentially non-linear.
            #
            # Although there's also "QF_NIRA" (mixed integer-real arithmetic),
            # "QF_NRA" seems to work better on 'dynamo/test_dynamic_shapes.py'.
            solver = z3.SolverFor("QF_NRA")
            # Set a timeout for finding a solution.
            solver.set(timeout=translation_validation_timeout())

            # Add all the assertions to the solver.
            for assertion in self._assertions:
                solver.add(assertion)

            # "Is there any case where it's TRUE for the target expressions,
            #  but FALSE for the source expressions?"
            solver.add(z3.Not(z3.And(*self._source_exprs)))
            solver.add(*self._target_exprs)

            log.debug("translation validation: start")
            r = solver.check()
            if r == z3.sat:
                # Target expressions are unsound.
                # Log the found model and the source expressions that failed.
                model = solver.model()
                raise ValidationException(
                    model,
                    self._assertions,
                    self._target_exprs,
                    failed_source_exprs=[
                        inp for inp in self._source_exprs if not model.evaluate(inp)
                    ],
                )
            else:
                if r == z3.unknown:
                    # Could not find a solution. It didn't fail, but it also
                    # didn't succeed. Canceling the validation execution (keyboard
                    # interrupt) also gets to this branch.
                    log.warning(
                        "translation validation: could not validate: got z3.unknown"
                    )
                else:
                    # Target expressions are sound.
                    if r != z3.unsat:
                        raise AssertionError(f"Expected z3.unsat, got {r}")
                    log.debug("translation validation: success")

except ImportError:
    _HAS_Z3 = False

    __all__ = [
        "translation_validation_enabled",
        "translation_validation_timeout",
        "ValidationException",
        "BisectValidationException",
    ]

else:
    _HAS_Z3 = True

    __all__ = [
        "z3str",
        "z3op",
        "PopulateValidator",
        "SympyToZ3",
        "TranslationValidator",
        "translation_validation_enabled",
        "translation_validation_timeout",
        "ValidationException",
        "BisectValidationException",
    ]

from torch.fx.experimental import _config as config


def translation_validation_enabled() -> bool:
    # Checks every time this function is called, in case the Dynamo
    # option is set, but Z3 is not installed.
    _assert_z3_installed_if_tv_set()
    return _HAS_Z3 and config.translation_validation


def translation_validation_timeout() -> int:
    return config.translation_validation_timeout


def _assert_z3_installed_if_tv_set() -> None:
    if not (_HAS_Z3 or not config.translation_validation):
        raise AssertionError(
            "translation validation requires Z3 package. Please, either install "
            "z3-solver or disable translation validation."
        )


class ValidationException(TorchDynamoException):
    def __init__(
        self, model: Any, assertions: Any, target_exprs: Any, failed_source_exprs: Any
    ) -> None:
        if not _HAS_Z3:
            raise AssertionError("Z3 is required")

        def symbolstr(sym: Any) -> str:
            return f"{sym}: {model[sym]}"

        def joinlines(xs: Any) -> str:
            return "\n".join(f"  ==> {x}" for x in xs)

        model_str = joinlines(sorted(map(symbolstr, model)))
        assertions_str = joinlines(sorted(map(z3str, assertions)))
        target_exprs_str = joinlines(sorted(map(z3str, target_exprs)))
        failed_source_exprs_str = joinlines(sorted(map(z3str, failed_source_exprs)))

        self.msg = "translation validation failed."
        self.details = f"""\
Model:
{model_str}

Assertions:
{assertions_str}

Target Expressions:
{target_exprs_str}

Failed Source Expressions:
{failed_source_exprs_str}"""

    def __str__(self) -> str:
        return f"{self.msg}\n\n{self.details}"


class BisectValidationException(TorchDynamoException):
    def __init__(
        self,
        validation_exc: ValidationException,
        expr: sympy.Basic,
        failed_action: str,
        traced_node: torch.fx.Node,
    ) -> None:
        self.msg = f"translation validation failed when {failed_action}: {expr}"
        self.details = f"""\
Failure occurred while running node:
    {traced_node.format_node()}

{validation_exc.details}"""

    def __str__(self) -> str:
        return f"{self.msg}\n\n{self.details}"


# Checks when this module is loaded.
_assert_z3_installed_if_tv_set()


# Translation validation bisection.
#
# Bisect into the torch._assert nodes recorded in the shape_env FX graph, and raise
# the earliest ValidationException.
#
# As guards are added by ShapeEnv.evaluate_expr calls, some simplification errors
# might be silently happening. This function tries to nail down exactly at which
# point things went wrong from a validation perspective.
def bisect(shape_env: "ShapeEnv") -> None:
    from torch.fx.experimental.recording import (
        FakeTensorMeta,
        replay_shape_env_events,
        ShapeEnvEvent,
    )
    from torch.fx.experimental.symbolic_shapes import (
        CURRENT_NODE_KEY,
        ShapeEnv,
        SHAPEENV_EVENT_KEY,
    )

    events = shape_env.events

    # Retrieves the ShapeEnvEvent associated with node.
    def get_node_event(node: torch.fx.Node) -> ShapeEnvEvent:
        if SHAPEENV_EVENT_KEY not in node.meta:
            raise AssertionError("SHAPEENV_EVENT_KEY not in node.meta")
        return events[node.meta[SHAPEENV_EVENT_KEY]]

    # Creates a new instance of fake, but updating every symbolic value's ShapeEnv
    # reference to the one given as argument.
    #
    # This is needed so as not to simplify a symbolic expression using a ShapeEnv
    # "from the future", where it may have a different set of replacements.
    @overload
    def new_with_shape_env(shape_env: ShapeEnv, fake: int) -> int: ...

    @overload
    def new_with_shape_env(shape_env: ShapeEnv, fake: torch.SymInt) -> torch.SymInt: ...

    @overload
    def new_with_shape_env(
        shape_env: ShapeEnv, fake: torch.SymFloat
    ) -> torch.SymFloat: ...

    @overload
    def new_with_shape_env(
        shape_env: ShapeEnv, fake: FakeTensorMeta
    ) -> FakeTensorMeta: ...

    def new_with_shape_env(
        shape_env: ShapeEnv, fake: int | torch.SymInt | torch.SymFloat | FakeTensorMeta
    ) -> int | torch.SymInt | torch.SymFloat | FakeTensorMeta:
        if isinstance(fake, int):
            return fake
        if isinstance(fake, torch.SymInt):
            return torch.SymInt(fake.node.with_shape_env(shape_env))
        if isinstance(fake, torch.SymFloat):
            return torch.SymFloat(fake.node.with_shape_env(shape_env))
        if not isinstance(fake, FakeTensorMeta):
            raise AssertionError(f"Expected FakeTensorMeta, got {type(fake)}")
        return FakeTensorMeta(
            tuple(new_with_shape_env(shape_env, s) for s in fake.size()),
            tuple(new_with_shape_env(shape_env, s) for s in fake.stride()),
            new_with_shape_env(shape_env, fake.storage_offset()),
            fake.is_nested,
        )

    # Checks whether the given shape_env fails when produce_guards is called.
    def check_shapeenv_fails(
        shape_env: ShapeEnv, tracked_fakes: list["TrackedFake"] | None
    ) -> ValidationException | None:
        if tracked_fakes is None:
            raise AssertionError("tracked_fakes is None")
        try:
            # This produce_guards call is a best-effort replication, since we
            # don't populate EqualityConstraint list. Reason: we would also have
            # to save OutputGraph.tracked_fakes_id_to_source.
            shape_env.produce_guards(
                # pyrefly: ignore [no-matching-overload]  # TrackedFake.fake includes FakeTensor
                [new_with_shape_env(shape_env, a.fake) for a in tracked_fakes],
                [a.source for a in tracked_fakes],
                input_contexts=[a.symbolic_context for a in tracked_fakes],
            )
            return None
        except ValidationException as e:
            return e

    # Checks whether the ShapeEnv reconstructed by replaying the events until
    # node is created fails when produce_guards is called.
    def check_node_fails(node: torch.fx.Node) -> ValidationException | None:
        number = node.meta[SHAPEENV_EVENT_KEY]
        # Reconstruct shape_env until the event at event_number.
        shape_env = replay_shape_env_events(events[: number + 1])
        shape_env.graph.lint()
        return check_shapeenv_fails(shape_env, events[number].tracked_fakes)

    last_exception = check_shapeenv_fails(
        shape_env, shape_env._snapshot_tracked_fakes()
    )

    if not last_exception:
        # We don't actually fail due to a produce_guards call.
        # Stop and don't bisect.
        log.info("translation validation succeeded: no errors found.")
        return

    if not shape_env.should_record_events or config.translation_validation_no_bisect:
        # Bisection is off.
        # Return the last ValidationException we got.
        raise last_exception

    # Cache the raised exception (if any) at each bisection point.
    exception = {}

    # Bisection happens on the assertion nodes of the recorded FX graph for
    # dynamic shapes.
    assert_nodes = [
        node for node in shape_env.graph.nodes if node.target is torch._assert
    ]

    # Preparing the indices for binary search.
    # The overall invariants are
    # - for all i < left, assert_node[i] doesn't fail
    # - for all i >= right, assert_node[i] fails
    # - `right in exception` always holds
    # - `left <= right` always holds
    left, mid, right = 0, 0, len(assert_nodes) - 1
    exception[right] = check_node_fails(assert_nodes[right])

    while left < right:
        mid = (left + right) // 2

        node = assert_nodes[mid]
        log.debug("bisecting at %s: %s", mid, get_node_event(node))

        # Check whether the new shape_env raises a ValidationException or not.
        exception[mid] = check_node_fails(node)

        if exception[mid]:
            right = mid
        else:
            left = mid + 1

    if not (left in exception and isinstance(exception[left], ValidationException)):
        raise AssertionError("Expected ValidationException at bisect result")
    left_exception: ValidationException = exception[left]  # type: ignore[assignment]

    node = assert_nodes[left]
    event = get_node_event(node)

    if event.is_evaluate_expr():
        failed_action = "evaluating"
    else:
        if not event.is_defer_runtime_assert():
            raise AssertionError(f"unexpected event type: {event}")
        failed_action = "adding runtime assert"

    args = event.args
    if args is None:
        raise AssertionError("event.args is None")
    if len(args) < 2:
        raise AssertionError(
            f"bisecting expects {event.name} to have at least 2 positional arguments. "
            f"Got: {len(args)}"
        )
    if not isinstance(args[1], sympy.Basic):
        raise AssertionError(
            f"bisecting expects {event.name} to have a SymPy expression as its second "
            f"argument. Got: {type(args[1])}"
        )

    raise BisectValidationException(
        left_exception,
        expr=args[1],
        failed_action=failed_action,
        traced_node=node.meta[CURRENT_NODE_KEY],
    )
<|endoftext|>"""Spec types for controlling what is dynamic in compiled/exported code.
Currently only supports unbacked dynamic shapes.

"""

from __future__ import annotations

import itertools
import logging
import threading
from typing import Any, cast, TYPE_CHECKING, TypeAlias

from torch import SymBool, SymInt


log = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import torch.nn
    from torch.fx.experimental.symbolic_shapes import ShapeEnv


__all__ = [
    "ShapeVar",
    "IntVar",
    "STATIC",
    "TensorSpec",
    "ObjectSpec",
    "DictSpec",
    "SeqSpec",
    "ParamsSpec",
    "ShapesSpec",
    "LeafSpec",
    "LeafIntSpec",
    "IntermediateSpec",
    "dynamic_spec",
]


# Indent unit for nested repr output. Two spaces per level.
_INDENT = "  "


# Sentinel meaning "this dim / scalar arg is static" (taken from the
# actual input at runtime, with recompiles on different values).
# Alias for ``None`` so users can write either ``TensorSpec([A, STATIC])``
# (more readable) or ``TensorSpec([A, None])``.
STATIC = None


# ---------------------------------------------------------------------------
# Spec ShapeEnv
#
# IntVar/ShapeVar wrap a SymNode, which requires a ShapeEnv. We use a singleton
# global env for shape specs. This shape env is special and only used for
# expression construction and will fail on any other calls.
# ---------------------------------------------------------------------------


_SPEC_SHAPE_ENV: ShapeEnv | None = None
# Lock guarding the lazy init of _SPEC_SHAPE_ENV so concurrent IntVar()
# calls observe the same singleton. We cannot construct eagerly at module
# import time because ShapeEnv.__init__ pulls in modules that
# transitively import this one.
_SPEC_SHAPE_ENV_LOCK = threading.Lock()


def _get_spec_shape_env() -> ShapeEnv:
    """Lazily build the singleton spec ShapeEnv (thread-safe)."""
    global _SPEC_SHAPE_ENV
    if _SPEC_SHAPE_ENV is not None:
        return _SPEC_SHAPE_ENV
    with _SPEC_SHAPE_ENV_LOCK:
        if _SPEC_SHAPE_ENV is not None:
            return _SPEC_SHAPE_ENV
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        class _SpecShapeEnv(ShapeEnv):
            """Special ShapeEnv used only at spec-definition time.

            Attribute access is blocked by default via ``__getattribute__``;
            only private (``_``-prefix) names and the explicit allowlist in
            ``_ALLOWED_PUBLIC`` pass through. SymInt arithmetic paths read
            a handful of public fields transparently (via the
            ``@record_shapeenv_event`` decorator and FX-cache machinery),
            so those are allowlisted. Everything else — evaluation, guard
            recording, deferred asserts, etc. — is blocked.

            During ``ShapeEnv.__init__`` itself, access is unrestricted
            because the base class touches its own public fields while
            bootstrapping.
            """

            # Public fields read by internal ShapeEnv infra during arithmetic.
            # Grow as new accesses are discovered by tests.
            _ALLOWED_PUBLIC: frozenset[str] = frozenset(
                {
                    "should_record_events",
                    "is_recording",
                    "fx_node_cache",
                    # bound_sympy is read-only (no guards/asserts/mutation).
                    # Needed for `%` (mod) which inspects symbol ranges to
                    # pick between Mod and PythonMod. var_to_range is the
                    # field bound_sympy reads to compute ranges.
                    "bound_sympy",
                    "var_to_range",
                }
            )

            def __init__(self) -> None:
                # Allow everything during super().__init__.
                object.__setattr__(self, "_init_done", False)
                super().__init__()
                object.__setattr__(self, "_init_done", True)

            def __getattribute__(self, name: str) -> Any:
                if name.startswith("_"):
                    return super().__getattribute__(name)
                if not super().__getattribute__("_init_done"):
                    return super().__getattribute__(name)
                if name in type(self)._ALLOWED_PUBLIC:
                    return super().__getattribute__(name)
                raise TypeError(
                    f"_SpecShapeEnv: '{name}' is not allowed at "
                    f"spec-definition time. Use IntVar / ShapeVar only "
                    f"inside TensorSpec / ShapesSpec."
                )

        _SPEC_SHAPE_ENV = _SpecShapeEnv()
        return _SPEC_SHAPE_ENV


class IntVar(SymInt):
    """Indicates that a scalar integer argument is dynamic (no implicit range).

    Unbacked dynamic shapes will represent the underlying value in the
    compiled graph.

    IntVar is a ``SymInt`` subclass backed by ``_SpecShapeEnv``,
    so arithmetic and comparisons compose naturally::

        A = IntVar("a")
        B = IntVar("b")
        A + 1  # SymInt expr=a#0 + 1
        A * B + 1  # SymInt expr=a#0*b#1 + 1
        A > 0  # SymBool

    Such derived ``SymInt`` values may be used directly as leaf specs in
    ``TensorSpec`` / ``ParamsSpec``

    Repr always includes a per-instance uid so two IntVars with the same name
    (or two anonymous ones) are distinguishable in logs::

        IntVar()           -> "IntVar(anon#0)"
        IntVar("offset")   -> "IntVar(offset#1)"

    Example::

        IntVar("num_heads")
        IntVar("offset", min=-100, max=100)
        IntVar("size", min=1, max=2048, optimization_hint=512)
    """

    _uid_counter = itertools.count()

    def __init__(
        self,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        import sympy

        from torch.fx.experimental.sym_node import _NO_HINT, SymNode
        from torch.utils._sympy.numbers import int_oo
        from torch.utils._sympy.value_ranges import ValueRanges

        self.name = name if name is not None else "anon"
        self._uid = next(IntVar._uid_counter)
        self.min = min
        self.max = max
        self.optimization_hint = optimization_hint
        self.sympy_sym = sympy.Symbol(f"{self.name}#{self._uid}")
        env = _get_spec_shape_env()
        node = SymNode(
            self.sympy_sym,
            shape_env=env,
            pytype=int,
            hint=_NO_HINT,
        )
        super().__init__(node)
        # Register the user-supplied (or unbounded) range so
        # range-dependent ops like `%` get correct ValueRanges info.
        env.var_to_range[self.sympy_sym] = ValueRanges(
            min if min is not None else -int_oo,
            max if max is not None else int_oo,
        )

    def __repr__(self) -> str:
        # Always include uid so two same-named (or anonymous) instances
        # are distinguishable in logs.
        parts = [f"{self.name}#{self._uid}"]
        if self.min is not None:
            parts.append(f"min={self.min}")
        if self.max is not None:
            parts.append(f"max={self.max}")
        if self.optimization_hint is not None:
            parts.append(f"optimization_hint={self.optimization_hint}")
        return f"{type(self).__name__}({', '.join(parts)})"

    # Hashable by identity — IntVars are unique per construction, and inheriting
    # SymInt.__hash__ would route through node hashing which we don't need here.
    def __hash__(self) -> int:  # type: ignore[override]
        return id(self)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "optimization_hint": self.optimization_hint,
        }


def _validate_spec_sym(v: Any, *, where: str) -> None:
    """If ``v`` is a SymInt or SymBool, validate it originates from the
    spec ShapeEnv.
    """
    if isinstance(v, SymInt):
        kind = "SymInt"
    elif isinstance(v, SymBool):
        kind = "SymBool"
    else:
        return
    if v.node.shape_env is not _get_spec_shape_env():
        raise TypeError(
            f"{where}: {kind} spec values must originate from spec IntVar / "
            f"ShapeVar; got {v!r} backed by a different ShapeEnv."
        )


class ShapeVar(IntVar):
    """Indicates that a dimension size is dynamic and is >= 0.

    Subclass of `IntVar` with ``min=0`` by default.

    Example::

        ShapeVar("batch")
        ShapeVar("batch", max=64)
        ShapeVar("batch", max=64, optimization_hint=32)
    """

    def __init__(
        self,
        name: str | None = None,
        *,
        min: int = 0,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        if min < 0:
            raise ValueError(
                f"ShapeVar requires min >= 0 (a shape dim is non-negative); "
                f"got min={min}. Use IntVar(...) for unrestricted scalars."
            )
        super().__init__(name, min=min, max=max, optimization_hint=optimization_hint)


class TensorSpec:
    """Per-dimension shape specification for a tensor.

    A list-like container of ``LeafIntSpec`` with length
    equal to the tensor's dim.

    Example::

        B = ShapeVar("batch")
        TensorSpec([B, STATIC])  # rank 2, dim 0 dynamic
        TensorSpec([B, 10])  # rank 2, dim 0 dynamic, dim 1 static=10
        TensorSpec([B * 2 + 1, STATIC])  # rank 2, dim 0 derived from B
    """

    def __init__(self, dims: Sequence[LeafIntSpec]) -> None:
        for i, d in enumerate(dims):
            if d is not None and not isinstance(d, (int, SymInt)):
                raise TypeError(
                    f"TensorSpec dim {i}: expected LeafIntSpec, got "
                    f"{type(d).__name__}: {d!r}"
                )
            _validate_spec_sym(d, where=f"TensorSpec dim {i}")
        self._specs: list[LeafIntSpec] = list(dims)

    def __getitem__(self, index: int) -> LeafIntSpec:
        return self._specs[index]

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[LeafIntSpec]:
        return iter(self._specs)

    def __repr__(self) -> str:
        lines = ["Tensor:"]
        for i, spec in enumerate(self._specs):
            lines.append(f"{_INDENT}{i}: {spec!r}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "TensorSpec",
            "dims": [
                spec.to_jsonable() if hasattr(spec, "to_jsonable") else spec
                for spec in self._specs
            ],
        }


class ObjectSpec:
    """Spec for any Python object's attributes.

    Constructor::

        ObjectSpec({name: IntermediateSpec, ...})

    Values may be leaves (``TensorSpec`` / ``IntVar`` / ``int`` /
    ``None``) or another ``ObjectSpec`` for recursion.

    Example::

        ObjectSpec({"weight": TensorSpec([ShapeVar("h"), STATIC])})
        ObjectSpec({"inner": ObjectSpec({"weight": TensorSpec([ShapeVar("h")])})})
    """

    def __init__(self, fields: dict[str, IntermediateSpec] | None = None) -> None:
        self._fields: dict[str, IntermediateSpec] = dict(fields) if fields else {}

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def items(self) -> Any:
        return self._fields.items()

    def __repr__(self) -> str:
        lines = ["object_spec:"]
        for name, spec in self._fields.items():
            spec_repr = repr(spec)
            if "\n" in spec_repr:
                lines.append(f"{_INDENT}.{name}:")
                for line in spec_repr.splitlines():
                    lines.append(_INDENT * 2 + line)
            else:
                lines.append(f"{_INDENT}.{name}: {spec_repr}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
