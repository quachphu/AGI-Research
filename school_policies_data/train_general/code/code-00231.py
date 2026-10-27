    # Stores all components.
    all_components: list[Component] = []

    # Stores nodes that will be used in main graph.
    used_in_main: dict[torch.fx.Node, None] = {}

    # Main graph after split.
    main_g = torch.fx.Graph()

    # Mapping from node in original module to node in main graph after split.
    main_remapping: dict[torch.fx.Node, torch.fx.Node] = {}

    # Output node of original module.
    output_node: torch.fx.Node | None = None

    # Create a component for each tag, we don't expect to create other components afterwards.
    for tag in tags:
        comp = Component(torch.fx.Graph(), len(all_components), f"{tag}")
        all_components.append(comp)
        tag_to_component[tag] = comp

    # Traverse the nodes in original graph and take care of them.
    for node in gm.graph.nodes:
        if node.op == "output":
            if output_node is not None:
                raise RuntimeError("Multiple output nodes in graph!")
            output_node = node
            continue

        # Placeholders in the original graph get copied to main graph.
        if node.op == "placeholder":
            main_remapping[node] = main_g.placeholder(node.name, type_expr=node.type)
            main_remapping[node].meta = copy.copy(node.meta)
            continue

        # Get_attr nodes are ignored because we are not tagging them.
        # Instead, we copy them directly to the submodules use them afterwards.
        if node.op == "get_attr":
            continue

        # Now we process callable nodes which are nodes with op of call_module,
        # call_function or call_method. Every callable nodes should be tagged.
        if not hasattr(node, "tag"):
            raise AssertionError(f"Node does not have tag: {node.format_node()}")

        upstream_components = [
            node_to_component[x]
            for x in flatten(node.args) + flatten(node.kwargs)
            if x.op not in {"placeholder", "get_attr"}
        ]

        comp = tag_to_component[node.tag]
        node_to_component[node] = comp

        # Max order of upperstream components.
        mx = max((c.order for c in upstream_components), default=0)

        # Expect the component for `node` has higher order then its upstream components.
        if comp.order < mx:
            raise AssertionError(
                f"Component {comp.name} order must be >= max of its upstream components, "
                f"order={comp.order} and max={mx}"
            )

        # Map an input of `node` to nodes in the component's graph.
        def remap_func(x: torch.fx.Node) -> torch.fx.Node:
            # If input is a get_attr node, copy it to current component's graph.
            # Returns the get_attr node in current component's graph.
            if x.op == "get_attr":
                if not isinstance(x.target, str):
                    raise RuntimeError(
                        f"Expected get_attr node target to be a str, got {type(x.target)}"
                    )
                if x not in comp.getattr_maps:
                    comp.getattr_maps[x] = comp.graph.get_attr(
                        x.target, type_expr=x.type
                    )
                    comp.getattr_maps[x].meta = copy.copy(x.meta)
                return comp.getattr_maps[x]

            # If input is not a placeholder, it should have been put into a component
            # already. If it's the current component then we return the corresponding
            # node in the component.
            if x.op != "placeholder" and node_to_component[x] == comp:
                return node_remapping[x]

            # If input is a placeholder or it's in other components, we want to make it
            # as a placeholder in current component's graph.
            if x not in comp.orig_inputs:
                comp.orig_inputs.append(x)
                placeholder = comp.graph.placeholder(x.name, type_expr=x.type)
                placeholder.meta = copy.copy(x.meta)
                comp.input_placeholders.append(placeholder)
                used_in_main[x] = None

            return comp.input_placeholders[comp.orig_inputs.index(x)]

        n = comp.graph.node_copy(node, remap_func)
        n.tag = node.tag  # type: ignore[attr-defined]
        node_remapping[node] = n
        node_to_component[n] = comp

    if output_node is None:
        raise RuntimeError("Graph had no output node!")

    for x in flatten(output_node.args[0]):
        if x.op == "get_attr":
            # We don't need components mapping for nodes of type "get_attr"
            # that are consumed by the output. Only need to make sure we create
            # corresponding counterparts in the resulting graph.
            main_remapping[x] = main_g.get_attr(x.name, type_expr=x.type)
        else:
            # All component results consumed by the output node should be
            # marked as "used in main".
            used_in_main[x] = None

    # If a node is used in main graph then we mark it as an output in the component
    # it belongs to.
    for n in used_in_main:
        if n.op != "placeholder":
            node_to_component[n].orig_outputs.append(n)

    # Now we create a graphmodule for each component.
    orig_to_split_fqn_mapping: dict[str, str] = {}
    for comp in all_components:
        outs = tuple(map(node_remapping.__getitem__, comp.orig_outputs))

        if return_tuple:
            comp.graph.output(outs)
        else:
            # Take care of the args of FX output node. If there's a single
            # output then the output node args is like (output_single), else
            # if there're multiple outputs then the output node args is like
            # ((output_0, output_1, ...)).
            comp.graph.output(outs[0] if len(outs) == 1 else outs)

        comp.gm, comp_orig_to_split_fqn_mapping = lift_subgraph_as_module(
            gm, subgraph=comp.graph, comp_name=comp.name
        )
        orig_to_split_fqn_mapping.update(comp_orig_to_split_fqn_mapping)

        # Create a call_module node in main graph.
        main_node = main_g.call_module(
            comp.name,
            args=tuple(map(main_remapping.__getitem__, comp.orig_inputs)),
            kwargs=None,
        )

        if len(outs) == 1 and not return_tuple:
            main_remapping[comp.orig_outputs[0]] = main_node
        else:
            for i, o in enumerate(comp.orig_outputs):
                # Use Proxy to record getitem access.
                main_remapping[o] = torch.fx.Proxy(main_node)[i].node  # type: ignore[index]

    main_g.output(map_arg(output_node.args[0], main_remapping.__getitem__))
    main_root = HolderModule({comp.name: comp.gm for comp in all_components})
    main_g._codegen = gm.graph._codegen

    # If the output nodes consumes get_attr directly in the original graph,
    # then we need to make sure get_attr is copied to the new graph.
    for x in flatten(output_node.args[0]):
        if x.op == "get_attr":
            setattr(main_root, x.name, getattr_recursive(gm, x.target))  # type: ignore[arg-type]

    result_gm = GraphModuleCls(main_root, main_g)
    if return_fqn_mapping:
        return result_gm, orig_to_split_fqn_mapping

    return result_gm


@compatibility(is_backward_compatible=False)
def move_non_tensor_nodes_on_boundary(subgraphs: list["Subgraph"]) -> None:
    """
    Move non-tensor nodes on the boundary between subgraphs.

    For each subgraph:

    1. Find nodes whose type is not tensor and any of its children is in another
       subgraph, put them in a queue for next step

    2. Do a BFS on those nodes in the queue,  and run a DFS for each node, let's say node X and it is in subgraph A:

       a. if it is in to_subgraph, return (continue DFS)
       b. if it is in from_subgraph, collect the nodes to nodes_to_move, and continue DFS
       c. otherwise, this means it cannot be moved
       d. also check if node X's parent should be put into the queue. (The queue may
          have duplicated nodes, just process the node once)

    Args:
        subgraphs: List of subgraphs containing nodes to be processed
    """
    # Create a mapping from node to subgraph for quick lookup
    node_to_subgraph: dict[torch.fx.Node, int] = {}
    for i, subgraph in enumerate(subgraphs):
        for node in subgraph.nodes:
            node_to_subgraph[node] = i

    def get_children_in_graph(node: torch.fx.Node) -> list[torch.fx.Node]:
        """Get children nodes that are in callable ops and in some subgraph"""
        return [
            user
            for user in node.users
            if user.op in CALLABLE_NODE_OPS and user in node_to_subgraph
        ]

    def get_parents_in_graph(node: torch.fx.Node) -> list[torch.fx.Node]:
        """Get parent nodes that are in callable ops and in some subgraph"""
        return [
            arg
            for arg in node.all_input_nodes
            if arg.op in CALLABLE_NODE_OPS and arg in node_to_subgraph
        ]

    def has_children_in_other_subgraph(
        node: torch.fx.Node, current_subgraph_idx: int
    ) -> bool:
        """
        Check if the node has any children in a subgraph different from current_subgraph_idx.
        This is the requirement used in both step 1 and step d.
        """
        children = get_children_in_graph(node)
        return any(
            node_to_subgraph[child] != current_subgraph_idx for child in children
        )

    def can_move_node_and_dependencies(
        node: torch.fx.Node, from_subgraph: int, to_subgraph: int
    ) -> tuple[bool, set[torch.fx.Node]]:
        """
        Check if node and its dependencies can be moved from from_subgraph to to_subgraph.
        Returns (can_move, nodes_to_move)

        For node X, do a DFS on its descendants, for each node:
        - if it is in to_subgraph, return (continue DFS)
        - if it is in from_subgraph, collect the nodes to nodes_to_move, and continue DFS
        - otherwise, this means it cannot be moved
        """
        nodes_to_move = set()
        visited = set()
        can_move = True

        def dfs(current_node: torch.fx.Node) -> None:
            nonlocal can_move, nodes_to_move

            if current_node in visited:
                return
            visited.add(current_node)

            # Check current node's subgraph
            if current_node not in node_to_subgraph:
                return  # Skip nodes not in any subgraph

            current_subgraph = node_to_subgraph[current_node]

            if current_subgraph == to_subgraph:
                # If it is in to_subgraph, just end DFS
                return
            elif current_subgraph == from_subgraph:
                # If it is in from_subgraph, collect it and continue DFS
                nodes_to_move.add(current_node)
            else:
                # Otherwise, this means it cannot be moved
                can_move = False
                return

            # Continue DFS on children
            children = get_children_in_graph(current_node)
            for child in children:
                if can_move:  # Only continue if we haven't already failed
                    dfs(child)

        # Start DFS from the original node
        dfs(node)

        return can_move, nodes_to_move

    # For each subgraph, find non-tensor nodes with children in other subgraphs
    for subgraph_idx, subgraph in enumerate(subgraphs):
        # non acc nodes cannot be moved to downstream acc graph, so skip
        if not subgraph.is_acc:
            continue
        # Step 1: Find non-tensor nodes with children in other subgraphs
        queue: list[torch.fx.Node] = []
        processed: set[torch.fx.Node] = set()

        for node in subgraph.nodes:
            # Check if node is non-tensor
            if is_node_output_tensor(node):
                continue

            # Check if node meets step 1 requirement: any children in another subgraph
            if has_children_in_other_subgraph(node, subgraph_idx):
                queue.append(node)

        # Step 2: BFS to move nodes that meet the criteria
        while queue:
            current_node = queue.pop(0)

            # Skip if already processed (queue may have duplicates)
            if current_node in processed:
                continue
            processed.add(current_node)

            # Skip if node is no longer in this subgraph (may have been moved)
            if (
                current_node not in node_to_subgraph
                or node_to_subgraph[current_node] != subgraph_idx
            ):
                continue

            children = get_children_in_graph(current_node)
            if len(children) == 0:
                raise AssertionError(
                    "Only node that has children in other subgraph can be moved"
                )

            # Find target subgraph. The children should all be in the same subgraph except current subgraph
            target_subgraph_candidates = set()
            for child in children:
                child_subgraph = node_to_subgraph[child]
                if child_subgraph != subgraph_idx:
                    target_subgraph_candidates.add(child_subgraph)
            # If multiple children live in different subgraphs, the node cannot be moved. User needs to find other ways to move it.
            if len(target_subgraph_candidates) != 1:
                print(
                    f"Cannot move non-tensor node {current_node.name} on boundary because it has children in multiple subgraphs"
                )
                continue

            target_subgraph = target_subgraph_candidates.pop()

            # Check if we can move this node and its dependencies
            can_move, nodes_to_move = can_move_node_and_dependencies(
                current_node, subgraph_idx, target_subgraph
            )

            if can_move:
                # Move all nodes in nodes_to_move to target subgraph
                for node_to_move in nodes_to_move:
                    # Remove from current subgraph
                    subgraph.nodes.remove(node_to_move)
                    # Add to target subgraph
                    subgraphs[target_subgraph].nodes.append(node_to_move)
                    # Update mapping
                    node_to_subgraph[node_to_move] = target_subgraph
                    print(
                        f"In order to move the non-tensor node {current_node.name} on boundary, "
                        f"moved node {node_to_move.name} from {'acc' if subgraph.is_acc else 'gpu'}_{subgraph_idx} "
                        f"to {'acc' if subgraphs[target_subgraph].is_acc else 'gpu'}_{target_subgraph}"
                    )

                # Add parents to the queue if they're non-tensor and not already processed
                # and meet the requirement from step 1 (any children in another subgraph)
                parents = get_parents_in_graph(current_node)
                for parent in parents:
                    if (
                        not is_node_output_tensor(parent)
                        and parent not in processed
                        and parent in node_to_subgraph
                        and node_to_subgraph[parent] == subgraph_idx
                    ):
                        # Check if parent meets step 1 requirement: any children in another subgraph
                        if not has_children_in_other_subgraph(parent, subgraph_idx):
                            raise AssertionError(
                                f"Parent {parent.name} should have children in another subgraph"
                            )
                        queue.append(parent)
<|endoftext|>import _operator
import itertools
from collections import defaultdict
from enum import Enum
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, is_fake_tensor
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map_only


__all__ = ["reinplace"]


class _ViewType(Enum):
    NonView = 0
    SingleOutputView = 1
    MultiOutputView = 2


def _is_view_op(tgt: object) -> bool | None:
    if tgt is not None and isinstance(tgt, torch._ops.OpOverload):
        schema = tgt._schema
        if len(schema.arguments) > 0:
            first_arg = schema.arguments[0]
            # check if op is a view
            return (
                first_arg.alias_info is not None and not first_arg.alias_info.is_write
            )


def _get_view_type(tgt: object) -> _ViewType:
    if tgt is not None and isinstance(tgt, torch._ops.OpOverload):
        schema = tgt._schema
        if len(schema.arguments) > 0:
            first_arg = schema.arguments[0]
            # check if op is a view
            if first_arg.alias_info is not None and not first_arg.alias_info.is_write:
                # check if op is a multi-output view
                if "*" in first_arg.alias_info.after_set:
                    return _ViewType.MultiOutputView
                else:
                    return _ViewType.SingleOutputView
    return _ViewType.NonView


# Stores a bunch of metadata related to functionalization each node.
# Relevant metadata:
# n.meta['fake_result']: FakeTensor (same type as the output of the node, but with FakeTensors instead of Tensors)
#   The fake tensor output from running the current node
# n.meta['view_of']: Node
#   If the current node n is a view of some base tensor, the 'view_of' field tells us which
#   view node was used to generate the current node (a view tensor).
#   This information actually makes `fake_result` redundant, but we can use `fake_result`
#   to sanity check that our aliasing information is correct.
@compatibility(is_backward_compatible=False)
class _FunctionalizationMetadataProp(torch.fx.Interpreter):
    def run_node(self, node: Node) -> Any:
        self.node_counter += 1
        result = super().run_node(node)
        node.meta["fake_result"] = result
        node.meta["node_idx"] = self.node_counter

        # (1) Update metadata with the list of nodes that are used by this node
        # copy_() doesn't read from its first argument; it writes to it, overwriting previous data.
        # We don't want to treat it as "being used as an input".
        node_args = node.args
        if node.target is torch.ops.aten.copy_.default:
            node_args = node_args[1:]

        # (2) Update metadata to track aliasing information about view tensor nodes.
        if node.op == "call_function":
            view_type = _get_view_type(node.target)
            if view_type == _ViewType.SingleOutputView:
                if not isinstance(node.args[0], Node):
                    raise AssertionError(f"Expected Node, got {type(node.args[0])}")
                node.meta["view_of"] = node.args[0]
            elif view_type == _ViewType.MultiOutputView:
                self.multi_output_view_nodes[node] = node.args[0]

            # Check if we returned a multi-output view,
            # and we're now grabbing the individual views from the output.
            #
            # For multi-output views, we want to map each output view to the base,
            # but this mapping involves two separate nodes in FX IR.
            # e.g. "a, b = x_1.split(...)" becomes:
            #    %split_tensor : [num_users=2] = call_function[target=torch.ops.aten.split.Tensor](args = (%x_1, 2), kwargs = {})
            #    %getitem : [num_users=1] = call_function[target=operator.getitem](args = (%split_tensor, 0), kwargs = {})
            #    %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%split_tensor, 1), kwargs = {})
            # And we'd like to set:
            #    getitem1.meta['view_of'] = x_1
            elif node.target is _operator.getitem:
                list_arg = node.args[0]
                maybe_base_of_view = self.multi_output_view_nodes.get(list_arg, None)
                if maybe_base_of_view is not None:
                    # Note: we could also track indexing info here for multi-output views.
                    # I don't think this metadata is strictly needed for de-functionalization.
                    if not isinstance(maybe_base_of_view, Node):
                        raise AssertionError(
                            f"Expected Node, got {type(maybe_base_of_view)}"
                        )
                    node.meta["view_of"] = maybe_base_of_view

        if "view_of" in node.meta:
            # We're linking the current node with its first argument as views.
            # Assert here that this is actually the case, and their storages are the same.
            if not is_fake_tensor(node.meta["fake_result"]):
                raise AssertionError("Expected FakeTensor in fake_result")
            if not is_fake_tensor(node.meta["view_of"].meta["fake_result"]):
                raise AssertionError("Expected FakeTensor in view_of fake_result")
            view_storage = StorageWeakRef(node.meta["fake_result"]._typed_storage())
            base_storage = StorageWeakRef(
                node.meta["view_of"].meta["fake_result"]._typed_storage()
            )
            if view_storage != base_storage:
                raise AssertionError("view_storage != base_storage")
        return result

    def propagate(self, *args: object) -> Any:
        self.multi_output_view_nodes = {}
        self.node_counter = -1

        with FakeTensorMode() as mode:
            fake_args = [
                mode.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args
            ]
            return super().run(*fake_args)


def _schemas_match(
    functional_schema: torch._C.FunctionSchema, inplace_schema: torch._C.FunctionSchema
) -> bool:
    names_match = (
        inplace_schema.name.endswith("_")
        and inplace_schema.name[:-1] == functional_schema.name
    )
    arg_types_match = len(functional_schema.arguments) == len(
        inplace_schema.arguments
    ) and all(
        a1.type == a2.type
        for a1, a2 in zip(functional_schema.arguments, inplace_schema.arguments)
    )
    # for the inplace op, its first argument should be mutable
    if not (
        inplace_schema.arguments[0].alias_info is not None
        and inplace_schema.arguments[0].alias_info.is_write
    ):
        raise AssertionError("First argument of inplace op must be mutable")
    # and its remaining arguments shouldn't be.
    if not all(a.alias_info is None for a in inplace_schema.arguments[1:]):
        raise AssertionError("Remaining arguments of inplace op must not be mutable")
    return names_match and arg_types_match


# TODO: this should be beefed up to be able to properly re-inplace with:
# - mutating ops (e.g. _fused_moving_avg_obs_fq_helper)
# - out= ops (e.g. angle -> angle.out)
# TODO: we should also figure this info out using torchgen.
def _maybe_get_inplace_op(op: object) -> torch._ops.OpOverload | None:
    # __module__ seems broken; it returns torch._ops.aten which doesn't exist
    if not isinstance(op, torch._ops.OpOverload):
        return None
    # Some view ops have inplace variants (as_strided_, etc),
    # but we do NOT want the reinplacing pass to directly add these into the program.
    # (they'll require extra special handling, aren't aren't really useful for perf anyway)
    if _is_view_op(op):
        return None
    op_namespace = op.__module__.split(".")[-1]
    op_base_name = op.overloadpacket.__name__
    maybe_namespace_module = getattr(torch.ops, op_namespace)
    maybe_inplace_op = (
        None
        if maybe_namespace_module is None
        else getattr(maybe_namespace_module, f"{op_base_name}_", None)
    )
    if maybe_inplace_op is None:
        return None

    inplace_overloads = maybe_inplace_op.op_overloads()
    inplace_overloads_with_matching_schemas = [
        f for f in inplace_overloads if _schemas_match(op._schema, f._schema)
    ]
    # Just because foo() and foo_() are both existing operators,
    # They aren't guaranteed to have compatible schemas.
    # For example, pow.Scalar(Scalar self, Tensor exponent) has no valid inplace variant,
    # Even though several overloads of pow_ exist.
    if len(inplace_overloads_with_matching_schemas) == 0:
        return None
    if len(inplace_overloads_with_matching_schemas) != 1:
        raise AssertionError(
            f"Expected exactly 1 matching inplace overload, got "
            f"{len(inplace_overloads_with_matching_schemas)}"
        )
    inplace_op = inplace_overloads_with_matching_schemas[0]
    return inplace_op


_VIEW_INVERSE_MAP: dict[torch._ops.OpOverload, torch._ops.OpOverload] = {
    torch.ops.aten.diagonal_scatter.default: torch.ops.aten.diagonal.default,
    torch.ops.aten.select_scatter.default: torch.ops.aten.select.int,
    torch.ops.aten.slice_scatter.default: torch.ops.aten.slice.Tensor,
    torch.ops.aten.as_strided_scatter.default: torch.ops.aten.as_strided.default,
}


# This function, given a set of set of (aliased) tensor nodes,
# Returns any nodes in the graph that *use* any of the aliases, that occur *after* op_index
# in the node ordering.
def _get_all_later_node_usages(tensor_aliases: set[Node], op_index: int) -> set[Node]:
    def _add_if_tensor(x: object, set_: set[StorageWeakRef]) -> None:
        if is_fake_tensor(x):
            set_.add(StorageWeakRef(x._typed_storage()))

    nodes_used_after = set()
    for t in tensor_aliases:
        # get all nodes that use the current alias
        usage_nodes = t.users
        for n in usage_nodes:
            # We only care about usages after the current node
            if "node_idx" not in n.meta or n.meta["node_idx"] <= op_index:
                continue
            # We also don't care about intermediate view ops.
            # They only matter if their output is then used elsewhere
            # (either in an out-of-place op, or as an output to the function).
            if n in tensor_aliases:
                if (
                    isinstance(n.target, torch._ops.OpOverload)
                    or n.target is _operator.getitem
                ):
                    continue
            nodes_used_after.add(n)
    return nodes_used_after


# Given an op that we're trying to re-inplace, "b = foo(a)",
# And given a {view}_scatter op that shows up later in the graph, "y = {view}_scatter(base, x, args...)"
# Then re-inplacing `foo()` would allow us to remove the `{view}_scatter` op entirely, IF:
# If there are any aliases in the alias_set(a) that satisfy:
# (1) The base of "alias", "alias_base", has the same size/stride/offset metadata as "base"
# (2) The output of running {view}(alias, args...) gives you the same size/stride/offset metadata
#     as "alias"
def _get_view_inverse_node_usages(
    later_node_usages: set[Node], self_aliases: set[Node]
) -> set[Node]:
    def matching_view_metadata(a: FakeTensor, b: FakeTensor) -> bool:
        return (
            a.size() == b.size()  # type: ignore[bad-return]
            and a.stride() == b.stride()
            and a.storage_offset() == b.storage_offset()
        )

    view_inverse_nodes = set()
    # Go through them in node order, so we can see chains of view_scatter ops.
    for n in sorted(later_node_usages, key=lambda x: x.meta["node_idx"]):
        if n.target not in _VIEW_INVERSE_MAP:
            continue
        base = n.args[0]
        mutated_view = n.args[1]
        if not isinstance(base, Node):
            raise AssertionError(f"Expected Node for base, got {type(base)}")
        if not isinstance(  # noqa: ISINSTANCE_FAKE_TENSOR
            base.meta["fake_result"], FakeTensor
        ):
            raise AssertionError("Expected FakeTensor in base.meta['fake_result']")
        if not isinstance(mutated_view, Node):
            raise AssertionError(
                f"Expected Node for mutated_view, got {type(mutated_view)}"
            )
        if not is_fake_tensor(mutated_view.meta["fake_result"]):
            raise AssertionError(
                "Expected FakeTensor in mutated_view.meta['fake_result']"
            )
        if isinstance(n.target, str):
            raise AssertionError("n.target should not be a string")
        # Check that this view_inverse op actually corresponds to taking doing the inverse
        # of one of our existing self_alias nodes.
        original_view = _VIEW_INVERSE_MAP[n.target]
        for self_alias in self_aliases:
            # We're looking for some alias of the self arg, "alias",
            # that was created from some op `alias = foo(base, args...)`
            # such that the current _scatter op "inverts" that foo call.
            # We can check that by running the original op again, and checking that the strides match.
            if "view_of" not in self_alias.meta:
                continue
            self_alias_base = self_alias.meta["view_of"]
            try:
                # We're trying to reuse the args from the view_scatter call inside of the corresponding
                # view op, which might throw. This just indicates that view_scatter op isn't a valid inverse
                # of the current alias we're looking at.
                view_replay_metadata = original_view(
                    self_alias_base.meta["fake_result"], *n.args[2:], **n.kwargs
                )
                expected_metadata = self_alias.meta["fake_result"]
                # If the alias and its base both have matching metadata, then this view_scatter op is valid to re-inplace.
                if matching_view_metadata(
                    self_alias_base.meta["fake_result"], base.meta["fake_result"]
                ) and matching_view_metadata(view_replay_metadata, expected_metadata):
                    view_inverse_nodes.add(n)
            except Exception:
                continue

    return view_inverse_nodes


@compatibility(is_backward_compatible=True)
def reinplace(gm: torch.fx.GraphModule, *sample_args: Any) -> torch.fx.GraphModule:
    r"""
    Given an fx.GraphModule, modifies it to perform "reinplacing",
    mutating the nodes of the graph.
    We look for out-of-place op call sites like ``b = a.add(...)``,
    and convert them to be inplace (``b = a.add_(...)``),
    as long as the input to the current operator ("a") isn't reused
    anywhere later in the graph.

    This pass currently expects to operate on a **functional, ATen** graph.
    This can be obtained by running ``make_fx(functionalize(f))``.

    Sample inputs are needed to determine aliasing relationships of the inputs.
    In general, we can't reinplace node ``b = a.add(...)`` if "a" aliases any of the
    inputs to the program.

    Given a node ``b = foo(a, args...)`` the algorithm for re-inplacing is as follows:

    **(1)** Perform some initial checks on the metadata of "a" and "args..."
    that can disqualify them from being reinplaced.

    - **(1a)** Check that the self argument we're attempting to reinplace
      has acceptable dtype/size metadata to reinplace with.

      For example, if we have::

        a = torch.ones(1)
        b = torch.ones(10)
        out = torch.add(a, b)

      We can't turn that into ``a.add_(b)`` because that would require resizing "a".

      Similarly, we can't convert ``torch.ge(a, b)`` into ``a.ge_(b)``,
      because that would require changing a's dtype (from e.g. float32 to bool).
      Note that in this specific example, we could technically do better..

      If we see the pattern::

        a_1 = a.ge(b)
        a_2 = aten._to_copy(a_1, a.dtype)

      Then this should be valid to completely re-inplace
      (this is exactly what functionalization will emit when it sees ``a.ge_(b)``).

      This optimization is only really important for user programs
      that directly use inplace comparison ops though.

      We also cannot re-inplace on tensors that have overlapping memory,
      e.g. ``torch.ones(1).expand(4, 4).add_(1)``.

    - **(1b)** Check if "a" is an alias of any of the program inputs.

      If it is, skip and move to the next node.
      Inplace'ing an op that would cause it to mutate a program is not sound,
      because that would be a side effect visible to the user.

      NOTE: there's a future optimization that we should make:
      if "a" is a (alias of a)  program input, but later in the program
      there is a node that looks like ``a.copy_(...)``,
      then re-inplacing is ok to do - we are temporarily reusing a's buffer,
      which will later be overwritten by the ``copy_()`` call.

      This will be an important optimization to have for programs that mutate
      their inputs. It currently isn't implemented though.

    - **(1c)** Check if "a" and "args..." alias.

      For example, re-inplacing to create code like the below
      isn't guaranteed to be sound::

        aten.mul_(a, a)

    **(2)** Check that "a" and all of its outstanding aliases are not used anywhere
    later in the graph. If this is the case, then it's safe to re-inplace
    to ``b = foo_(a)``.

    There are a few caveats to this, explained in more detail below:

    - (a) If "a" is used later as an argument to a view op, that is okay.
      It's only a problem if "a" (or that view) is later passed
      into a normal operator, or if it is returned as the program output.
    - (b) If "a" is a repeat argument in ``foo()``, then don't reinplace.
      Most ATen kernels don't make any guarantees that this is sound,
      e.g. if you do ``aten.mul_(a, a)``.
      So we'll just ban re-inplacing in this case.
    - (c) If "a" is used as an input into a view "inverse" / "scatter"
      operator, it is potentially fine to re-inplace
      (and remove that scatter operator from the graph).
      See below for a more detailed example.

    NOTE: there is an optimization in this step that is crucial
    to fully recovering performance from functionalization.

    Given this program::

        def f(x):
            a = torch.ops.aten.add(x, x)
            b = torch.ops.aten.diagonal(a)
            torch.ops.aten.fill_(b, 0)
            return d

    Functionalization will emit the following::

        def f(x):
            a = torch.ops.aten.add(x, x)
            b = torch.ops.aten.diagonal(a, 0, 1)
            b_updated = torch.ops.aten.fill(b, 0)
            a_updated = torch.ops.aten.diagonal_scatter(a, b_updated, 0, 1)
            return a_updated

    Ordinarily, we would not be able to reinplace the fill,
    because "b" aliases with "a" which is used by the diagonal_scatter call.

    "re-inplacing" is on the hook for figuring out that it is ok to
    completely remove the expensive diagonal_scatter call, if we re-inplace
    the add().

    So, for every ``alias in alias_set(a)``, instead of checking
    that "alias" is not used anywhere later in the graph,
    we check that EITHER:

    - (a) alias is not used anywhere later in the graph, OR
    - (b) alias is used exactly once later on in the graph,
      in the following op::

        out = foo_scatter(alias, x, args...)

      where the following must hold:

      - (i) ``foo_scatter`` is the "inverse" operator for foo.
        This only applies to "foo" ops that are view operators,
        which view into a subset of the original tensor's memory.
        In practice, there are ~4 operators where this applies::

          diagonal -> diagonal_scatter
          slice -> slice_scatter
          select -> select_scatter
          as_strided -> as_strided_scatter

      - (ii) "args..." are the same between the ``foo()`` and
        ``foo_scatter()`` calls.

    **(3)** Perform the actual re-inplacing on foo!

    (3b) is the common case, but special care is needed for
    ``{view}_scatter`` (3a).

    - **(3a)** ``{view}_scatter`` ops.

      Consider this program::

        a = torch.zeros(2, 2)
        b = torch.ones(2)
        a[0] = b

      Post functionalization, that will look like::

        a = torch.zeros(2)
        b = torch.ones(1)
        a_updated = torch.select_scatter(a, b, 0, 0)

      In this case though, there is no "functional" op to re-inplace!
      Instead, we'd like to directly remove the select_scatter call.
      We already know from (3) that this is valid,
      because "a" has no later usages in the graph.

      We perform the re-inplacing on the ``{view}_scatter`` op like so.

      Before::

        a_updated = torch.select_scatter(a, b, args...)

      After::

        a_slice = a.select(a, args...)
        a_slice.copy_(b)

    - **(3b)** Otherwise, replace the functional op with its inplace variant.

      Before::

        b = foo(a, args...)

      After::

        a.foo_(args...)

    **(4)** Finally, after converting either::

        # Before:                              # After:
        b = foo(a)                             foo_(a)

    or::

        # Before:
        b = {slice}_scatter(a, mutated_slice, args...)
        # After:
        slice = {slice}(a, args...)
        slice.copy_(mutated_slice)

    We now need to find all later nodes that use "b" as an argument
    and update them to take in "a" instead.

    Note that for the majority of inplace ops, this isn't actually necessary
    (because most inplace ops return "self" as their output).
    This isn't generally true for all mutable ops though, which is why
    we need to actually replace all of the arguments.

    We also need to update our metadata of ``Dict[StorageWeakRef, Set[Node]]``,
    that maps a given tensor storage to the set of all nodes that take in that
    storage as an input.
    Specifically, re-inplacing ``b = foo(a)`` causes "a" and "b"'s sets to get
    fused together.

    **(5)** Any ``view_inverse/scatter`` nodes that were identified as
    "it's ok to ignore them" during step (3) get manually deleted from the graph.
    Their outputs are no longer used, so technically standard DCE would be able
    to do this, but we can no longer run FX's DCE pass now that we have mutable
    ops in the graph.
    """
    _FunctionalizationMetadataProp(gm).propagate(*sample_args)

    # Useful debug printing
    # def _print(x):
    # if is_fake_tensor(x):
    # print(f'fake_result: {StorageWeakRef(x._typed_storage()).cdata}')

    # for n in gm.graph.nodes:
    # print(n.format_node())
    # if hasattr(n, 'meta'):
    # print(f'node_idx: {n.meta["node_idx"]}')
    # if 'fake_result' in n.meta:
    # tree_map(_print, n.meta['fake_result'])
    # if 'view_of' in n.meta:
    # print(f'view_of: {str(n.meta["view_of"])}')
    # print()

    # We need to know which nodes correspond to inputs (or their aliases)
    # so we know not to re-inplace them.
    # NOTE: later, we'll need to add an optimization for fully recovering performance
    # on programs that mutate inputs.
    input_storages = {
        StorageWeakRef(node.meta["fake_result"]._typed_storage())
        for node in gm.graph.nodes
        if (
            node.op == "placeholder"
            and isinstance(node.meta["fake_result"], torch.Tensor)
        )
    }

    # We also need to know for a given node, what are all of its aliasing nodes.
    storage_to_nodes: dict[StorageWeakRef, set[Node]] = defaultdict(set)
    for n in gm.graph.nodes:
        if "fake_result" in n.meta:
            # Tree-mapping because some ops can return lists of tensors.
            def _add_to_map(x: object) -> None:
                if is_fake_tensor(x):
                    storage_to_nodes[StorageWeakRef(x._typed_storage())].add(n)

            pytree.tree_map_(_add_to_map, n.meta["fake_result"])

    # inplace-ify functional ops, subject to the constraints written below.
    all_later_view_inverse_nodes_to_delete = set()
    for node in gm.graph.nodes:
        if node.op == "call_function":
            # Today, the re-inplace pass on directly acts on:
            # - functional ops with an inplace variant
            # - {view}_scatter ops that can be potentially removed from the graph.
            # Both of these ops take in tensor first args, so filtering on this condition
            # makes the later code simpler.
            # We should revisit this at some point though, particularly when we also want
            # the reinplacer to be able to handle out= and mutable operators
            # and tensorlist first args (like `_foreach_` ops).
            if not isinstance(node.target, torch._ops.OpOverload):
                continue
            if len(node.target._schema.arguments) < 1:
                continue
            if type(node.target._schema.arguments[0].type) is not torch.TensorType:
                continue

            # Step 1a: Check that the self argument we're attempting to reinplace
            # has the same size/stride as the output.
            # For example, we shouldn't try to reinplace torch.add(scalar_tensor, larger_tensor)
            # As it would require resizing scalar_tensor.
            # (We could potentially swizzle this into larger_tensor.add_(scalar_tensor),
            # this is probably an optimization to revisit later).
            self_arg = node.args[0]
            self_flattened = pytree.tree_leaves(self_arg.meta["fake_result"])
            node_flattened = pytree.tree_leaves(node.meta["fake_result"])
            self_has_wrong_metadata = False
            if len(self_flattened) == len(node_flattened):
                for self_meta, node_meta in zip(self_flattened, node_flattened):
                    if self_meta.numel() != node_meta.numel():
                        self_has_wrong_metadata = True
                    if self_meta.dtype != node_meta.dtype:
                        self_has_wrong_metadata = True
                    # We also cannot re-inplace on tensors that have internal memory overlap.
                    # e.g. torch.ones(1).expand(4, 4).add_(1)
                    if torch._debug_has_internal_overlap(self_meta) == 1:
                        self_has_wrong_metadata = True
            # Here, we (optimistically) assume that a.resize(b) is valid to re-inplace,
            # Since users should never really be calling the functional "torch.ops.aten.resize"
            # op directly in their programs.
            if self_has_wrong_metadata and node.target != torch.ops.aten.resize.default:
                continue

            # Step 1b: ensure that the op we're trying to re-inplace isn't a program input
            self_arg_storage = StorageWeakRef(
                self_arg.meta["fake_result"]._typed_storage()
            )
            if self_arg_storage in input_storages:
                # TODO: later, add the optimization for handling `copy_()` calls in the graph.
                continue
            if len([x for x in node.args if x is self_arg]) > 1:
                # Step 1c:
                # Calling stuff like aten.mul_(a, a) isn't guaranteed to be sound,
                # so we prevent re-inplacing in this case.
                continue

            self_arg_storage = StorageWeakRef(
                self_arg.meta["fake_result"]._typed_storage()
            )
            self_aliases = storage_to_nodes[self_arg_storage]

            # First, we find all later usages of any of the aliases of self_arg.
            later_node_usages = _get_all_later_node_usages(
                self_aliases, node.meta["node_idx"]
            )
            # Then, we check if any of those later usages are actually view_scatter ops
            # that are safe to fully remove.
            later_view_inverse_node_usages = _get_view_inverse_node_usages(
                later_node_usages, self_aliases
            )

            # Step 2: Check to see if the input to the op is reused later in the graph.
            # If not (same goes for its aliases), then this op is safe to re-in place.
            # This is a slightly roundabout way to check that there are no later usages of the current self argument.
            # (later_view_inverse_node_usages corresponds to "view_scatter" nodes that we are allowed to delete)
            can_reinplace = len(later_node_usages - later_view_inverse_node_usages) == 0
            if not can_reinplace:
                continue

            # Step 3a: Special handling for when we see *_scatter operators.
            # When we see an operator like `b = torch.slice_scatter(a, ...)`,
            # instead of trying to "inplace" it into a.slice_scatter_(..._),
            # we would prefer to remove it from the graph entirely,
            # and instead copy_() the slice directly into the larger tensor.
            # See the description of the algorithm for a full example.
            if (
                node.target in _VIEW_INVERSE_MAP
                and node not in all_later_view_inverse_nodes_to_delete
            ):
                view_op = _VIEW_INVERSE_MAP[node.target]
                # Before:
                #   base_updated = torch.ops.aten.slice_scatter.default(base, mutated_slice, args...)
                # After:
                #   slice = torch.ops.aten.slice.default(base, args...)
                #   slice.copy_(mutated_slice)
                with gm.graph.inserting_before(node):
                    mutated_slice_node = node.args[1]
                    remaining_slice_args = node.args[2:]
                    slice_node = gm.graph.create_node(
                        "call_function",
                        view_op,
                        (self_arg,) + tuple(remaining_slice_args),
                        node.kwargs,
                    )
                    gm.graph.create_node(
                        "call_function",
                        torch.ops.aten.copy_.default,
                        (
                            slice_node,
                            mutated_slice_node,
                        ),
                        {},
                    )
                # Add the slice_scatter node to our "nodes to delete" list.
                all_later_view_inverse_nodes_to_delete.add(node)

            else:
                # Step 3b: Check to see if this operator has an inplace variant.
                maybe_inplace_op = _maybe_get_inplace_op(node.target)
                if maybe_inplace_op is None:
                    continue
                # And if so, replace it with its inplace variant.
                node.target = maybe_inplace_op

            # At this point, 'storage_to_nodes' will be stale.
            # Now that we're inplacing `b = foo(a)`, we need to effectively
            # union together the dict values for b and a's storage.
            # Hmm... morally I think we also want to keep the `fake_result` metadata
            # up to date here, but I'm not sure how easy it is to do.
            # Maybe it's fine to wait until the end of the pass to update it.
            curr_node_storage = StorageWeakRef(
                node.meta["fake_result"]._typed_storage()
            )
            storage_to_nodes[self_arg_storage].update(
                storage_to_nodes[curr_node_storage]
            )
            storage_to_nodes[curr_node_storage].update(
                storage_to_nodes[self_arg_storage]
            )

            # Need to remember the view_scatter view nodes we found so we can remove them alter.
            all_later_view_inverse_nodes_to_delete.update(
                later_view_inverse_node_usages
            )

            # Step 4:
            # Now that we've replaced b = a.foo() with a.foo_(),
            # We need to replace any later usages of "b" with "a"
            for old in itertools.chain([node], later_view_inverse_node_usages):
                new = old.args[0]
                if not isinstance(new, Node):
                    raise AssertionError(f"Expected Node, got {type(new)}")
                nodes_to_update = [
                    n for n in old.users if n.meta["node_idx"] > node.meta["node_idx"]
                ]
                for node_to_update in nodes_to_update:

                    def replace_arg(a: Node) -> Node:
                        if a == old:
                            return new
                        return a

                    # First, replace usages of "b" with "a"
                    node_to_update.args = tree_map_only(
                        Node, replace_arg, node_to_update.args
                    )
                    node_to_update.kwargs = tree_map_only(
                        Node, replace_arg, node_to_update.kwargs
                    )

                    # Second, update our storage_to_nodes data structure.
                    old_flattened_res = pytree.tree_leaves(old.meta["fake_result"])
                    node_flattened_res = pytree.tree_leaves(
                        node_to_update.meta["fake_result"]
                    )

                    old_res_storage = {
                        StorageWeakRef(x._typed_storage())
                        for x in old_flattened_res
                        if is_fake_tensor(x)
                    }
                    node_res_storage = {
                        StorageWeakRef(x._typed_storage())
                        for x in node_flattened_res
                        if is_fake_tensor(x)
                    }

                    # This will happen if we're updating a view op, e.g.
                    # e.g. replacing
                    #     x = view(old)
                    #     x = view(new)
                    # When that happens, we need to make sure to keep our
                    # storage mapping up to date.
                    #
                    # We're checking for len(...) == 1 here because all view ops are guaranteed to return either a single tensor,
                    # or multiple tensors that all share the same storage.
                    # We can't just check equality because we might encounter FX nodes that return zero tensor outputs.
                    if (
                        len(old_res_storage) == 1
                        and len(node_res_storage) == 1
                        and old_res_storage == node_res_storage
                    ):
                        # pyrefly: ignore [missing-attribute]
                        new_flattened_res = pytree.tree_leaves(new.meta["fake_result"])
                        new_res_storage = {
                            StorageWeakRef(x._typed_storage())
                            for x in new_flattened_res
                            if is_fake_tensor(x)
                        }
                        if len(new_res_storage) != 1:
                            raise AssertionError(
                                f"Expected 1 storage, got {len(new_res_storage)}"
                            )
                        (new_ref,) = new_res_storage
                        (node_ref,) = node_res_storage
                        # Technically, "old_ref" and all its aliases will remain
                        # in our mapping.
                        # That should be fine though, since we deleted "old"
                        # from the graph at this point.
                        storage_to_nodes[node_ref].update(storage_to_nodes[new_ref])
                        storage_to_nodes[new_ref].update(storage_to_nodes[node_ref])

    # Step 4: delete any _scatter nodes that we de-functionalized
    # Need to take care not to delete any of these nodes until after *all* modifications
    # to the graph are finished.
    for to_delete in all_later_view_inverse_nodes_to_delete:
        gm.graph.erase_node(to_delete)

    gm.recompile()
    return gm
<|endoftext|>from typing import Any

import torch.fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, is_fake_tensor
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import py_sym_types, snapshot_fake
from torch.fx.node import map_aggregate
from torch.utils._ordered_set import OrderedSet


__all__ = ["FakeTensorProp"]


@compatibility(is_backward_compatible=False)
class FakeTensorProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and record a fake tensor representing
    the metadata for the node.  Unlike ShapeProp, (1) this propagation
    is cheap--it does the propagation with meta tensors which do not actually
    store data, and (2) the fake tensors have much more fine grained information,
    e.g., they have accurate alias information that can be consulted by looking
    at the storages.

    Args:
         module (GraphModule): The module to be executed
         mode (Optional[FakeTensorMode]): The dispatch mode used to execute computation indicated by each FX Node.
    """

    def __init__(
        self, module: torch.fx.GraphModule, mode: FakeTensorMode | None = None
    ) -> None:
        super().__init__(module)
        if mode is None:
            mode = FakeTensorMode()
        self._mode = mode
        mode.epoch += 1
        mode.reset_nt_tensor_id_counter()
        self.seen_subgraphs: OrderedSet[str] = OrderedSet()

    def run_node(self, n: Node) -> Any:
        from torch.fx.experimental.symbolic_shapes import (
            compute_unbacked_bindings,
            rebind_unbacked,
        )

        if (
            n.op == "call_function"
            and n.target is torch.ops.higher_order.invoke_subgraph
            and n.args[1] not in self.seen_subgraphs
        ):
            # Prevent redundant fake tensor prop for invoke_subgraphs. Note that
            # there is also fake tensor caching for the entire subgraph. This
            # happens the next time we call `run_node` for the same subgraph,
            # which goes through super.run_node and caches the fake tensor prop.
            # Therefore, we are propagating fake tensor through the subgraphs
            # twice.
            if not isinstance(n.args[1], str):
                raise AssertionError(f"Expected str, got {type(n.args[1])}")
            if not (
                isinstance(n.args[0], torch.fx.Node)
                and n.args[0].op == "get_attr"
                and isinstance(n.args[0].target, str)
            ):
                raise AssertionError(
                    "Expected n.args[0] to be a get_attr Node with str target"
                )
            self.seen_subgraphs.add(n.args[1])
            operands = n.args[2:]
            example_inputs = []
            for operand in operands:
                if not (isinstance(operand, torch.fx.Node) and "val" in operand.meta):
                    raise AssertionError("Expected Node with 'val' in meta")
                example_inputs.append(operand.meta["val"])
            return FakeTensorProp(
                getattr(self.module, n.args[0].target), mode=self._mode
            ).propagate(*example_inputs)

        result = super().run_node(n)
        rebind_unbacked(self._mode.shape_env, n, result)

        def extract_val(obj: Any) -> Any:
            if is_fake_tensor(obj):
                return snapshot_fake(obj)
            elif isinstance(obj, torch.Tensor):
                # TODO: How is it possible that we get a non fake tensor?  We
                # should be running under the mode...
                return snapshot_fake(self._mode.from_tensor(obj, static_shapes=True))
            elif isinstance(obj, (*py_sym_types, int, float, bool)):
                return obj
            else:
                return None

        meta = map_aggregate(result, extract_val)
        if meta is not None:
            n.meta["val"] = meta
            if (shape_env := self._mode.shape_env) and (
                symbol_to_path := compute_unbacked_bindings(shape_env, result)
            ):
                n.meta["unbacked_bindings"] = symbol_to_path

        return result

    def propagate(self, *args: object) -> Any:
        fake_args = [
            self._mode.from_tensor(a) if isinstance(a, torch.Tensor) else a
            for a in args
        ]
        return self.propagate_dont_convert_inputs(*fake_args)

    def propagate_dont_convert_inputs(self, *args: object) -> Any:
        # In-place ops like shallow_copy_data_ can mutate fake_device on
        # input FakeTensors during propagation. Save and restore so the
        # caller's inputs are not permanently corrupted.
        saved_devices = [
            (a, a.fake_device)
            for a in args
            if isinstance(a, FakeTensor)  # noqa: ISINSTANCE_FAKE_TENSOR
        ]
        with self._mode:
            try:
                return super().run(*args)
            finally:
                for fake, device in saved_devices:
                    fake.fake_device = device
<|endoftext|>import hashlib
from itertools import chain
from types import ModuleType
from typing import Any, TYPE_CHECKING

import torch
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import _parse_stack_trace
from torch.fx.node import _format_arg, _get_qualified_name
from torch.fx.operator_schemas import normalize_function
from torch.fx.passes.shape_prop import TensorMetadata


if TYPE_CHECKING:
    import pydot

    HAS_PYDOT = True
else:
    pydot: ModuleType | None
    try:
        import pydot

        HAS_PYDOT = True
    except ModuleNotFoundError:
        HAS_PYDOT = False
        pydot = None


__all__ = ["FxGraphDrawer"]

_COLOR_MAP = {
    "placeholder": '"AliceBlue"',
    "call_module": "LemonChiffon1",
    "get_param": "Yellow2",
    "get_attr": "LightGrey",
    "output": "PowderBlue",
}

_HASH_COLOR_MAP = [
    "CadetBlue1",
    "Coral",
    "DarkOliveGreen1",
    "DarkSeaGreen1",
    "GhostWhite",
    "Khaki1",
    "LavenderBlush1",
    "LightSkyBlue",
    "MistyRose1",
    "MistyRose2",
    "PaleTurquoise2",
    "PeachPuff1",
    "Salmon",
    "Thistle1",
    "Thistle3",
    "Wheat1",
]

_WEIGHT_TEMPLATE = {
    "fillcolor": "Salmon",
    "style": '"filled,rounded"',
    "fontcolor": "#000000",
}

if HAS_PYDOT:

    @compatibility(is_backward_compatible=False)
    class FxGraphDrawer:
        """
        Visualize a torch.fx.Graph with graphviz
        Basic usage:
            g = FxGraphDrawer(symbolic_traced, "resnet18")
            g.get_dot_graph().write_svg("a.svg")
        """

        def __init__(
            self,
            graph_module: torch.fx.GraphModule,
            name: str,
            ignore_getattr: bool = False,
            ignore_parameters_and_buffers: bool = False,
            skip_node_names_in_args: bool = True,
            parse_stack_trace: bool = False,
            dot_graph_shape: str | None = None,
            normalize_args: bool = False,
        ) -> None:
            self._name = name
            self.dot_graph_shape = (
                dot_graph_shape if dot_graph_shape is not None else "record"
            )
            self.normalize_args = normalize_args
            _WEIGHT_TEMPLATE["shape"] = self.dot_graph_shape

            self._dot_graphs = {
                name: self._to_dot(
                    graph_module,
                    name,
                    ignore_getattr,
                    ignore_parameters_and_buffers,
                    skip_node_names_in_args,
                    parse_stack_trace,
                )
            }

            for node in graph_module.graph.nodes:
                if node.op != "call_module":
                    continue

                leaf_node = self._get_leaf_node(graph_module, node)

                if not isinstance(leaf_node, torch.fx.GraphModule):
                    continue

                self._dot_graphs[f"{name}_{node.target}"] = self._to_dot(
                    leaf_node,
                    f"{name}_{node.target}",
                    ignore_getattr,
                    ignore_parameters_and_buffers,
                    skip_node_names_in_args,
                    parse_stack_trace,
                )

        def get_dot_graph(self, submod_name: str | None = None) -> pydot.Dot:
            """
            Visualize a torch.fx.Graph with graphviz
            Example:
                >>> # xdoctest: +REQUIRES(module:pydot)
                >>> # xdoctest: +REQUIRES(module:ubelt)
                >>> # define module
                >>> class MyModule(torch.nn.Module):
                >>>     def __init__(self) -> None:
                >>>         super().__init__()
                >>>         self.linear = torch.nn.Linear(4, 5)
                >>>     def forward(self, x):
                >>>         return self.linear(x).clamp(min=0.0, max=1.0)
                >>> module = MyModule()
                >>> # trace the module
                >>> symbolic_traced = torch.fx.symbolic_trace(module)
                >>> # setup output file
                >>> import ubelt as ub
                >>> dpath = ub.Path.appdir("torch/tests/FxGraphDrawer").ensuredir()
                >>> fpath = dpath / "linear.svg"
                >>> # draw the graph
                >>> g = FxGraphDrawer(symbolic_traced, "linear")
                >>> g.get_dot_graph().write_svg(fpath)
            """
            if submod_name is None:
                return self.get_main_dot_graph()
            else:
                return self.get_submod_dot_graph(submod_name)

        def get_main_dot_graph(self) -> pydot.Dot:
            return self._dot_graphs[self._name]

        def get_submod_dot_graph(self, submod_name: str) -> pydot.Dot:
            return self._dot_graphs[f"{self._name}_{submod_name}"]

        def get_all_dot_graphs(self) -> dict[str, pydot.Dot]:
            return self._dot_graphs

        def _get_node_style(self, node: torch.fx.Node) -> dict[str, str]:
            template = {
                "shape": self.dot_graph_shape,
                "fillcolor": "#CAFFE3",
                "style": '"filled,rounded"',
                "fontcolor": "#000000",
            }
            if node.op in _COLOR_MAP:
                template["fillcolor"] = _COLOR_MAP[node.op]
            else:
                # Use a random color for each node; based on its name so it's stable.
                target_name = node._pretty_print_target(node.target)
                target_hash = int(
                    hashlib.md5(
                        target_name.encode(), usedforsecurity=False
                    ).hexdigest()[:8],
                    16,
                )
                template["fillcolor"] = _HASH_COLOR_MAP[
                    target_hash % len(_HASH_COLOR_MAP)
                ]
            return template

        def _get_leaf_node(
            self, module: torch.nn.Module, node: torch.fx.Node
        ) -> torch.nn.Module:
            py_obj = module
            if not isinstance(node.target, str):
                raise AssertionError(f"Expected str target, got {type(node.target)}")
            atoms = node.target.split(".")
            for atom in atoms:
                if not hasattr(py_obj, atom):
                    raise RuntimeError(
                        str(py_obj) + " does not have attribute " + atom + "!"
                    )
                py_obj = getattr(py_obj, atom)
            return py_obj

        def _typename(self, target: torch.fx.node.Target | torch.nn.Module) -> str:
            if isinstance(target, torch.nn.Module):
                ret = torch.typename(target)
            elif isinstance(target, str):
                ret = target
            else:
                ret = _get_qualified_name(target)

            # Escape "{" and "}" to prevent dot files like:
            # https://gist.github.com/SungMinCho/1a017aab662c75d805c5954d62c5aabc
            # which triggers `Error: bad label format (...)` from dot
            return ret.replace("{", r"\{").replace("}", r"\}")

        # shorten path to avoid drawing long boxes
        # for full path = '/home/weif/pytorch/test.py'
        # return short path = 'pytorch/test.py'
        def _shorten_file_name(
            self,
            full_file_name: str,
            truncate_to_last_n: int = 2,
        ) -> str:
            splits = full_file_name.split("/")
            if len(splits) >= truncate_to_last_n:
                return "/".join(splits[-truncate_to_last_n:])
            return full_file_name

        def _get_node_label(
            self,
            module: torch.fx.GraphModule,
            node: torch.fx.Node,
            skip_node_names_in_args: bool,
            parse_stack_trace: bool,
        ) -> str:
            def _get_str_for_args_kwargs(arg: tuple[Any, ...] | dict[str, Any]) -> str:
                if isinstance(arg, tuple):
                    prefix, suffix = r"|args=(\l", r",\n)\l"
                    arg_strs_list = [_format_arg(a, max_list_len=8) for a in arg]
                elif isinstance(arg, dict):
                    prefix, suffix = r"|kwargs={\l", r",\n}\l"
                    arg_strs_list = [
                        f"{k}: {_format_arg(v, max_list_len=8)}" for k, v in arg.items()
                    ]
                else:  # Fall back to nothing in unexpected case.
                    return ""

                # Strip out node names if requested.
                if skip_node_names_in_args:
                    arg_strs_list = [a for a in arg_strs_list if "%" not in a]
                if len(arg_strs_list) == 0:
                    return ""
                arg_strs = prefix + r",\n".join(arg_strs_list) + suffix
                if len(arg_strs_list) == 1:
                    arg_strs = arg_strs.replace(r"\l", "").replace(r"\n", "")
                return arg_strs.replace("{", r"\{").replace("}", r"\}")

            label = "{" + f"name=%{node.name}|op_code={node.op}\n"

            if node.op == "call_module":
                leaf_module = self._get_leaf_node(module, node)
                label += r"\n" + self._typename(leaf_module) + r"\n|"
                extra = ""
                if hasattr(leaf_module, "__constants__"):
                    extra = r"\n".join(
                        [
                            f"{c}: {getattr(leaf_module, c)}"
                            for c in leaf_module.__constants__  # type: ignore[union-attr]
                        ]  # type: ignore[union-attr]
                    )
                label += extra + r"\n"
            else:
                label += f"|target={self._typename(node.target)}" + r"\n"
                if self.normalize_args:
                    try:
                        args, kwargs = normalize_function(  # type: ignore[misc]
                            node.target,  # type: ignore[arg-type]
                            node.args,  # type: ignore[arg-type]
                            node.kwargs,
                            normalize_to_only_use_kwargs=True,
                        )
                    except Exception:
                        # Fallback to not normalizing if there's an exception.
                        # Some functions need overloads specified to normalize.
                        args, kwargs = node.args, node.kwargs
                else:
                    args, kwargs = node.args, node.kwargs
                if len(args) > 0:
                    label += _get_str_for_args_kwargs(args)
                if len(kwargs) > 0:
                    label += _get_str_for_args_kwargs(kwargs)
                label += f"|num_users={len(node.users)}" + r"\n"

            tensor_meta = node.meta.get("tensor_meta")
            label += self._tensor_meta_to_label(tensor_meta)

            # for original fx graph
            # print buf=buf0, n_origin=6
            buf_meta = node.meta.get("buf_meta", None)
            if buf_meta is not None:
                label += f"|buf={buf_meta.name}" + r"\n"
                label += f"|n_origin={buf_meta.n_origin}" + r"\n"

            # for original fx graph
            # print file:lineno code
            if parse_stack_trace and node.stack_trace is not None:
                parsed_stack_trace = _parse_stack_trace(node.stack_trace)
                if parsed_stack_trace is not None:
                    fname = self._shorten_file_name(parsed_stack_trace.file)
                    label += (
                        f"|file={fname}:{parsed_stack_trace.lineno} {parsed_stack_trace.code}"
                        + r"\n"
                    )

            return label + "}"

        def _tensor_meta_to_label(self, tm: object) -> str:
            if tm is None:
                return ""
            elif isinstance(tm, TensorMetadata):
                return self._stringify_tensor_meta(tm)
            elif isinstance(tm, list):
                result = ""
                for item in tm:
                    result += self._tensor_meta_to_label(item)
                return result
            elif isinstance(tm, dict):
                result = ""
                for v in tm.values():
                    result += self._tensor_meta_to_label(v)
                return result
            elif isinstance(tm, tuple):
                result = ""
                for item in tm:
                    result += self._tensor_meta_to_label(item)
                return result
            else:
                raise RuntimeError(f"Unsupported tensor meta type {type(tm)}")

        def _stringify_tensor_meta(self, tm: TensorMetadata) -> str:
            result = ""
            if not hasattr(tm, "dtype"):
                print("tm", tm)
            result += "|" + "dtype" + "=" + str(tm.dtype) + r"\n"
            result += "|" + "shape" + "=" + str(tuple(tm.shape)) + r"\n"
            result += "|" + "requires_grad" + "=" + str(tm.requires_grad) + r"\n"
            result += "|" + "stride" + "=" + str(tm.stride) + r"\n"
            if tm.is_quantized:
                if tm.qparams is None:
                    raise AssertionError("qparams is None for quantized tensor")
                if "qscheme" not in tm.qparams:
                    raise AssertionError("qscheme not in qparams")
                qscheme = tm.qparams["qscheme"]
                if qscheme in {
                    torch.per_tensor_affine,
                    torch.per_tensor_symmetric,
                }:
                    result += "|" + "q_scale" + "=" + str(tm.qparams["scale"]) + r"\n"
                    result += (
                        "|"
                        + "q_zero_point"
                        + "="
                        + str(tm.qparams["zero_point"])
                        + r"\n"
                    )
                elif qscheme in {
                    torch.per_channel_affine,
                    torch.per_channel_symmetric,
                    torch.per_channel_affine_float_qparams,
                }:
                    result += (
                        "|"
                        + "q_per_channel_scale"
                        + "="
                        + str(tm.qparams["scale"])
                        + r"\n"
                    )
                    result += (
                        "|"
                        + "q_per_channel_zero_point"
                        + "="
                        + str(tm.qparams["zero_point"])
                        + r"\n"
                    )
                    result += (
                        "|"
                        + "q_per_channel_axis"
                        + "="
                        + str(tm.qparams["axis"])
                        + r"\n"
                    )
                else:
                    raise RuntimeError(f"Unsupported qscheme: {qscheme}")
                result += "|" + "qscheme" + "=" + str(tm.qparams["qscheme"]) + r"\n"
            return result

        def _get_tensor_label(self, t: torch.Tensor) -> str:
            return str(t.dtype) + str(list(t.shape)) + r"\n"

        # when parse_stack_trace=True
        # print file:lineno code
        def _to_dot(
            self,
            graph_module: torch.fx.GraphModule,
            name: str,
            ignore_getattr: bool,
            ignore_parameters_and_buffers: bool,
            skip_node_names_in_args: bool,
            parse_stack_trace: bool,
        ) -> pydot.Dot:
            """
            Actual interface to visualize a fx.Graph. Note that it takes in the GraphModule instead of the Graph.
            If ignore_parameters_and_buffers is True, the parameters and buffers
            created with the module will not be added as nodes and edges.
            """

            # "TB" means top-to-bottom rank direction in layout
            dot_graph = pydot.Dot(name, rankdir="TB")

            buf_name_to_subgraph: dict[str, pydot.Cluster] = {}

            for node in graph_module.graph.nodes:
                if ignore_getattr and node.op == "get_attr":
                    continue

                style = self._get_node_style(node)
                dot_node = pydot.Node(
                    node.name,
                    label=self._get_node_label(
                        graph_module, node, skip_node_names_in_args, parse_stack_trace
                    ),
                    **style,  # type: ignore[arg-type]
                )

                current_graph = dot_graph

                buf_meta = node.meta.get("buf_meta", None)
                if buf_meta is not None and buf_meta.n_origin > 1:
                    buf_name = buf_meta.name
                    if buf_name not in buf_name_to_subgraph:
                        buf_name_to_subgraph[buf_name] = pydot.Cluster(
                            buf_name, label=buf_name
                        )
                    current_graph = buf_name_to_subgraph.get(buf_name)  # type: ignore[assignment]

                # pyrefly: ignore [missing-attribute]
                current_graph.add_node(dot_node)

                def get_module_params_or_buffers() -> None:
                    for pname, ptensor in chain(
                        leaf_module.named_parameters(),
                        # pyrefly: ignore [bad-argument-type]
                        leaf_module.named_buffers(),
                    ):
                        pname1 = node.name + "." + pname
                        label1 = (
                            pname1 + "|op_code=get_" + "parameter"
                            if isinstance(ptensor, torch.nn.Parameter)
                            else "buffer" + r"\l"
                        )
                        dot_w_node = pydot.Node(
                            pname1,
                            label="{" + label1 + self._get_tensor_label(ptensor) + "}",
                            **_WEIGHT_TEMPLATE,  # type: ignore[arg-type]
                        )
                        dot_graph.add_node(dot_w_node)
                        dot_graph.add_edge(pydot.Edge(pname1, node.name))

                if node.op == "call_module":
                    leaf_module = self._get_leaf_node(graph_module, node)

                    if not ignore_parameters_and_buffers and not isinstance(
                        leaf_module, torch.fx.GraphModule
                    ):
                        get_module_params_or_buffers()

            for subgraph in buf_name_to_subgraph.values():
                subgraph.set("color", "royalblue")
                subgraph.set("penwidth", "2")
                dot_graph.add_subgraph(subgraph)  # type: ignore[arg-type]

            for node in graph_module.graph.nodes:
                if ignore_getattr and node.op == "get_attr":
                    continue

                for user in node.users:
                    dot_graph.add_edge(pydot.Edge(node.name, user.name))

            return dot_graph

else:
    if not TYPE_CHECKING:

        @compatibility(is_backward_compatible=False)
        class FxGraphDrawer:
            def __init__(
                self,
                graph_module: torch.fx.GraphModule,
                name: str,
                ignore_getattr: bool = False,
                ignore_parameters_and_buffers: bool = False,
                skip_node_names_in_args: bool = True,
                parse_stack_trace: bool = False,
                dot_graph_shape: str | None = None,
                normalize_args: bool = False,
            ):
                raise RuntimeError(
                    "FXGraphDrawer requires the pydot package to be installed. Please install "
                    "pydot through your favorite Python package manager."
                )
<|endoftext|>import inspect
import logging
from collections import defaultdict, OrderedDict
from collections.abc import Callable
from typing import Any

import torch
from torch.fx._compatibility import compatibility
from torch.fx._lazy_graph_module import _LazyGraphModule, _make_graph_module
from torch.fx._utils import lazy_format_graph_code
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node


__all__ = ["Partition", "split_module", "split_module_simple"]
log = _LOGGER = logging.getLogger(__name__)


@compatibility(is_backward_compatible=True)
class Partition:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.submod_name = f"submod_{name}"
        self.node_names: list[str] = []
        self.inputs: dict[str, None] = {}
        self.outputs: dict[str, None] = {}
        self.dependencies: dict[str, None] = {}
        self.dependents: dict[str, None] = {}
        self.graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
        self.environment: dict[Node, Node] = {}
        self.targets: dict[str, Any] = {}

    def __repr__(self) -> str:
        return (
            f"name: {self.name},\n"
            f" nodes: {self.node_names},\n"
            f" inputs: {self.inputs},\n"
            f" outputs: {self.outputs},\n"
            f" partitions depended on: {self.dependencies},\n"
            f" partition dependents: {self.dependents}"
        )


def _get_attr_from_qualname(mod: torch.nn.Module, qualname: str) -> Any:
    attr_val = mod
    for atom in qualname.split("."):  # type: ignore[union-attr]
        if not hasattr(attr_val, atom):
            raise AttributeError(f"Node target {qualname} not found!")
        attr_val = getattr(attr_val, atom)
    return attr_val


# Creates subgraphs out of main graph
@compatibility(is_backward_compatible=True)
def split_module(
    m: GraphModule,
    root_m: torch.nn.Module,
    split_callback: Callable[[Node], int],
    qualname_map: dict[str, str] | None = None,
    keep_original_order: bool | None = False,
    keep_original_node_name: bool | None = False,
    keep_original_input_name: bool = True,
    *,
    partition_affix: str | None = None,
    tuple_return: bool = False,
) -> GraphModule:
    """
    Creates subgraphs out of main graph

    Args:
        m (GraphModule): Graph module to split
        root_m (torch.nn.Module): root nn module. Not currently used. Included
            because the root nn module is usually transformed via
            torch.fx._symbolic_trace.symbolic_trace (see example below)
        split_callback (Callable[[Node], int]): Callable function
            that maps a given Node instance to a numeric partition identifier.
            split_module will use this function as the policy for which operations
            appear in which partitions in the output Module.
        qualname_map: Optional[Dict[str, str]]: optional output parameter that returns a
            mapping from new target names in the module after split to old target
            names in the original module.
        keep_original_order: Optional[bool]: keep the original order of the GraphModule
            or use the Topological order of the new constructed GraphModule
        keep_original_node_name: Optional[bool]: If the partitioned graphs should
            have the same node names as the original graph.
        keep_original_input_name: bool: If the partitioned graphs should
            have the same input names as the original graph.
        partition_affix: Optional[str]: If specified, the submodules' names will contain
            the affix, e.g. "submod_<affix>_<idx>".
        tuple_return: bool: If True, submodule outputs are always wrapped in a tuple,
            even when there is only a single output value.  This makes all subgraphs
            conform to the convention expected by ``torch._inductor.compile_fx``.

    Returns:
        GraphModule: the module after split.

    Example:

        This is a sample setup:

            import torch
            from torch.fx._symbolic_trace import symbolic_trace
            from torch.fx.graph_module import GraphModule
            from torch.fx.node import Node
            from torch.fx.passes.split_module import split_module

            class MyModule(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.param = torch.nn.Parameter(torch.rand(3, 4))
                    self.linear = torch.nn.Linear(4, 5)

                def forward(self, x, y):
                    z = self.linear(x + self.param).clamp(min=0.0, max=1.0)
                    w = self.linear(y).clamp(min=0.0, max=1.0)
                    return z + w

            # symbolically trace model
            my_module = MyModule()
            my_module_traced = symbolic_trace(my_module)

            # random mod partitioning
            partition_counter = 0
            NPARTITIONS = 3

            def mod_partition(node: Node):
                global partition_counter
                partition = partition_counter % NPARTITIONS
                partition_counter = (partition_counter + 1) % NPARTITIONS
                return partition

            # split module in module with submodules
            module_with_submodules = split_module(
                my_module_traced, my_module, mod_partition
            )

        Output looks like this. Original graph is broken into partitions

            > print(module_with_submodules)
            GraphModule(
                (submod_0): GraphModule(
                    (linear): Linear(in_features=4, out_features=5, bias=True)
                )
                (submod_1): GraphModule(
                    (linear): Linear(in_features=4, out_features=5, bias=True)
                )
                (submod_2): GraphModule()
            )

            def forward(self, x, y):
                param = self.param
                submod_0 = self.submod_0(x, param, y);  x = param = y = None
                getitem = submod_0[0]
                getitem_1 = submod_0[1];  submod_0 = None
                submod_1 = self.submod_1(getitem, getitem_1);  getitem = getitem_1 = None
                getitem_2 = submod_1[0]
                getitem_3 = submod_1[1];  submod_1 = None
                submod_2 = self.submod_2(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
                return submod_2

        Output of split module is the same as output of input traced module.
        This is an example within a test setting:

            > orig_out = my_module_traced(x, y)
            > submodules_out = module_with_submodules(x, y)
            > self.assertEqual(orig_out, submodules_out)
            True
    """

    log.debug(
        "%s",
        lazy_format_graph_code("pre split_module", m, colored=True),
    )

    def construct_graph(
        node: Node,
        base_mod_env: dict[str, Node],
        base_mod_attrs: dict[str, torch.fx.graph_module.GraphModule],
    ) -> tuple[dict[str, Node], dict[str, torch.fx.graph_module.GraphModule]]:
        if node.op == "placeholder":
            default_value = (
                node.args[0] if len(node.args) > 0 else inspect.Signature.empty
            )
            if keep_original_node_name:
                args = (
                    () if default_value is inspect.Signature.empty else (default_value,)
                )
                base_mod_env[node.name] = base_mod_graph.create_node(
                    "placeholder",
                    node.name,
                    args=args,  # type: ignore[arg-type]
                    type_expr=node.type,
                )
            else:
                base_mod_env[node.name] = base_mod_graph.placeholder(
                    node.target,  # type: ignore[arg-type]
                    type_expr=node.type,
                    default_value=default_value,
                )
            base_mod_env[node.name].meta = node.meta.copy()
        elif node.op == "get_attr":
            base_mod_env[node.name] = base_mod_graph.get_attr(node.target)  # type: ignore[arg-type]
            base_mod_env[node.name].meta = node.meta.copy()
            if not isinstance(node.target, str):
                raise AssertionError(f"Expected str target, got {type(node.target)}")
            attr_val = _get_attr_from_qualname(m, node.target)
            base_mod_attrs[node.target] = attr_val  # type: ignore[index]
        return base_mod_env, base_mod_attrs

    import sympy

    partitions: dict[str, Partition] = {}
    orig_nodes: dict[str, Node] = {}
    symbol_to_node: dict[sympy.Symbol, Node] = {}

    def record_cross_partition_use(def_node: Node, use_node: Node | None) -> None:
        from torch.fx.experimental.symbolic_shapes import free_symbols

        defined = getattr(def_node, "_fx_partition", None)
        used = getattr(use_node, "_fx_partition", None)

        log.debug(
            "record_cross_partition_use %s (%s) %s (%s)",
            def_node.name,
            defined,
            use_node.name if use_node is not None else "-",
            used,
        )

        if defined != used:
            if defined is not None:
                def_partition = partitions[defined]
                def_partition.outputs.setdefault(def_node.name)
                if used is not None:
                    def_partition.dependents.setdefault(used)

            if used is not None:
                use_partition = partitions[used]
                use_partition.inputs.setdefault(def_node.name)
                # We have made def_node an input to the use_partition.  If
                # this input has symbolic symbols in its size, those also must
                # be made as inputs to the partition
                if (def_val := def_node.meta.get("example_value")) is not None:
                    for s in sorted(free_symbols(def_val), key=str):
                        s_node = symbol_to_node[s]
                        use_partition.inputs.setdefault(s_node.name)
                        if symbol_to_node[s].op != "placeholder":
                            # If the node that defines the symbol is not a
                            # placeholder, we must make it an output of the
                            # partition.  Note that this may be in a different
                            # partition than defined!  Although, this doesn't
                            # really make a difference for correctness, since
                            # defined is guaranteed to have the symbol in
                            # scope and can return it; you just get less
                            # optimal codegen in this case.
                            s_defined = getattr(s_node, "_fx_partition", None)
                            if s_defined is not None:
                                s_def_partition = partitions[s_defined]
                                s_def_partition.outputs.setdefault(s_node.name)
                                s_def_partition.dependents.setdefault(used)
                                use_partition.dependencies.setdefault(s_defined)
                if defined is not None:
                    use_partition.dependencies.setdefault(defined)

    def instantiate_node_partition_mapping(node: Node) -> None:
        partition_idx = split_callback(node)
        partition_name = str(partition_idx)
        if partition_affix is not None:
            # For example, if user specifies partition_affix = "pp", then the
            # partition name will be "pp_0", "pp_1", etc
            partition_name = "_".join([partition_affix, partition_name])

        log.debug(
            "instantiate_node_partition_mapping %s (%s)", node.name, partition_name
        )

        # add node to partitions
        partition = partitions.get(partition_name)
        if partition is None:
            partitions[partition_name] = partition = Partition(partition_name)

        partition.node_names.append(node.name)
        node._fx_partition = partition_name

    # Global State Nodes are nodes which by their global state effects,
    # "taint" all downstream nodes while they are active.
    GLOBAL_STATE_NODES = [
        torch.amp._enter_autocast,
        torch.amp._exit_autocast,
        torch._C._set_grad_enabled,
    ]

    # For grad regions:
    # ------------------------
    # 1. first region: we do nothing
    # 2. subsequent regions: we insert the set_grad at the beginning
    grad_regions: OrderedDict[Node, set[int]] = OrderedDict()

    # For autocast regions:
    # ------------------------
    # 1. first region: we will only insert the _exit at the end
    # 2. intermediate regions: we will insert both the
    #    _enter at the beginning and _exit at the end
    # 3. last region: we will only insert _enter at the beginning
    # We will do so in the order in which the autocasts were instantiated.
    autocast_regions: OrderedDict[Node, set[int]] = OrderedDict()
    autocast_exits: dict[Node, Node | None] = {}

    active_grad = None
    active_autocasts = set()

    for node in m.graph.nodes:
        # This will prefer placeholder bindings, because those come first.
        # This is a little dangerous though: it is possible that an unbacked
        # symbol is used without any binding site for it, in which case we
        # will get a KeyError not able to find it.  I'd like to fix this by
        # having passes.runtime_assert establish some invariants that I can
        # rely on later, but this needs some extra work.  Quick fix first.
        # See https://github.com/pytorch/pytorch/issues/130534
        if (
            (val := node.meta.get("example_value")) is not None
            and isinstance(val, (torch.SymInt, torch.SymFloat))
            and isinstance(s0 := val.node.expr, sympy.Symbol)
            and s0 not in symbol_to_node
        ):
            symbol_to_node[val.node.expr] = node

        if node.op in ["placeholder", "get_attr", "output"]:
            continue

        instantiate_node_partition_mapping(node)

        if node.op == "call_function" and node.target in GLOBAL_STATE_NODES:
            if node.target is torch._C._set_grad_enabled:
                if len(node.args) != 1:
                    raise AssertionError(
                        f"Expected 1 arg for _set_grad_enabled, got {len(node.args)}"
                    )
                if not isinstance(node.args[0], bool):
                    raise AssertionError(f"Expected bool arg, got {type(node.args[0])}")
                active_grad = node
                grad_regions[active_grad] = set({split_callback(node)})
            elif node.target is torch.amp._enter_autocast:
                # Should all be python constants
                if not all(not isinstance(arg, Node) for arg in node.args):
                    raise AssertionError(
                        "Expected all args to be python constants, not Nodes"
                    )
                active_autocasts.add(node)
                autocast_regions[node] = set({split_callback(node)})
                autocast_exits[node] = None
            elif node.target is torch.amp._exit_autocast:
                if len(node.args) != 1:
                    raise AssertionError(
                        f"Expected 1 arg for _exit_autocast, got {len(node.args)}"
                    )
                autocast_regions[node.args[0]].add(split_callback(node))
                active_autocasts.remove(node.args[0])
                autocast_exits[node.args[0]] = node

        if active_grad is not None:
            grad_regions[active_grad].add(split_callback(node))

        for a in active_autocasts:
            autocast_regions[a].add(split_callback(node))

    if not all(v is not None for v in autocast_exits.values()):
        raise AssertionError("autocast must exit")

    # pyrefly: ignore [bad-assignment]
    autocast_regions = {k: sorted(v) for k, v in autocast_regions.items()}
    # pyrefly: ignore [bad-assignment]
    grad_regions = {k: sorted(v) for k, v in grad_regions.items()}

    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug("autocast_regions: %s", autocast_regions)
        _LOGGER.debug("grad_regions: %s", grad_regions)

    assert_monotonically_increasing = bool(autocast_regions) or bool(grad_regions)

    # split nodes into partitions
    highest_partition = -1
    for node in m.graph.nodes:
        orig_nodes[node.name] = node

        # TODO currently placeholders/parameters aren't put into random partitions,
        # rather they're added to the graphs where they are used down below
        if node.op in ["placeholder", "get_attr"]:
            continue
        if node.op == "output":
            torch.fx.graph.map_arg(
                node.args[0], lambda n: record_cross_partition_use(n, None)
            )
            continue

        if assert_monotonically_increasing:
            pid = split_callback(node)
            if highest_partition > pid:
                raise AssertionError(
                    "autocast or set_grad_enabled require monotonically increasing "
                    f"partitions: highest: {highest_partition}, this node's: {pid}"
                )
            highest_partition = pid

        # do not capture cross-partition dependencies for global state nodes as they will be
        # self-contained - their setup and unwind will be isolated to each partition submodule.
        if node.target not in GLOBAL_STATE_NODES:
            torch.fx.graph.map_arg(
                node.args, lambda def_node: record_cross_partition_use(def_node, node)
            )
            torch.fx.graph.map_arg(
                node.kwargs, lambda def_node: record_cross_partition_use(def_node, node)
            )

    original_partition_order = list(partitions.keys())
    # find partitions with no dependencies
    root_partitions: list[str] = []
    for partition_name, partition in partitions.items():
        if not len(partition.dependencies):
            root_partitions.append(partition_name)

    # check partitions for circular dependencies and create topological partition ordering
    sorted_partitions: list[str] = []
    while root_partitions:
        root_partition = root_partitions.pop()
        sorted_partitions.append(root_partition)
        for dependent in partitions[root_partition].dependents:
            partitions[dependent].dependencies.pop(root_partition)  # noqa: B909
            if not partitions[dependent].dependencies:
                root_partitions.append(dependent)
    if len(sorted_partitions) != len(partitions):
        raise RuntimeError("cycle exists between partitions!")

    # Enter prelude
    for regions_mapping in [autocast_regions, grad_regions]:
        for node, regions in regions_mapping.items():
            if len(regions) == 0:
                raise AssertionError("Expected at least one region for node")
            # pyrefly: ignore [bad-index]
            partitions[str(regions[0])].environment[node] = node
            # pyrefly: ignore [bad-index, index-error]
            # pyrefly: ignore [bad-index, index-error]
            for r in regions[1:]:
                partition = partitions[str(r)]
                new_node = partition.graph.create_node(
                    op=node.op,
                    target=node.target,
                    args=tuple(arg for arg in node.args),
                    kwargs={},
                    type_expr=node.type,
                )
                new_node.meta = (
                    node.meta.copy()
                )  # is it really a good idea to copy this?
                partition.environment[node] = new_node

    # add placeholders to partition inputs
    for partition_name in sorted_partitions:
        partition = partitions[partition_name]
        new_inputs: dict[str, None] = {}

        counter = 0

        # Process inputs that become placeholders before inputs that become
        # get_attr nodes, so that all placeholders precede get_attr in the
        # submodule graph.
        placeholder_inputs: list[str] = []
        get_attr_inputs: list[str] = []
        for inp in partition.inputs:
            orig_node = orig_nodes[inp]
            if (
                orig_node.op == "get_attr"
                and isinstance(orig_node.target, str)
                and isinstance(
                    _get_attr_from_qualname(m, orig_node.target), torch.nn.Module
                )
            ):
                get_attr_inputs.append(inp)
            else:
                placeholder_inputs.append(inp)

        for inp in placeholder_inputs + get_attr_inputs:
            orig_node = orig_nodes[inp]
            # We don't pass in get_attr nodes as inputs to the partition, but
            # instead set them as targets and use getattr within the module

            def add_placeholder() -> Node:
                if keep_original_input_name:
                    name = inp
                else:
                    nonlocal counter
                    name = f"arg_{counter}"
                    counter += 1
                placeholder = partition.graph.placeholder(
                    name,
                    type_expr=orig_nodes[inp].type,
                )
                new_inputs[inp] = None
                return placeholder

            if orig_node.op == "get_attr":
                if not isinstance(orig_node.target, str):
                    raise AssertionError(
                        f"Expected str target, got {type(orig_node.target)}"
                    )

                orig_attr = _get_attr_from_qualname(m, orig_node.target)
                if isinstance(orig_attr, torch.nn.Module):
                    placeholder = partition.graph.get_attr(orig_node.target)
                    partition.targets[orig_node.target] = orig_attr
                else:
                    placeholder = add_placeholder()
            else:
                placeholder = add_placeholder()
            placeholder.meta = orig_nodes[inp].meta.copy()
            partition.environment[orig_nodes[inp]] = placeholder
        partition.inputs = new_inputs

    # Transform nodes and collect targets for partition's submodule
    for node in m.graph.nodes:
        if hasattr(node, "_fx_partition"):
            partition = partitions[node._fx_partition]

            # swap out old graph nodes in kw/args with references to new nodes in this submodule
            environment = partition.environment
            gathered_args = torch.fx.graph.map_arg(node.args, lambda n: environment[n])
            gathered_kwargs = torch.fx.graph.map_arg(
                node.kwargs, lambda n: environment[n]
            )

            if node.op not in ["call_module", "get_attr"]:
                target = node.target
            else:
                target_attr = _get_attr_from_qualname(m, node.target)
                target = node.target.replace(".", "_")
                partition.targets[target] = target_attr
                # Fill in the passed-in mapping from new qualname to old qualname
                if qualname_map is not None:
                    # When creating the split module later, the submodules will have
                    # path prefix matching the corresponding partition's submod_name
                    qualname = f"{partition.submod_name}.{target}"
                    qualname_map[qualname] = node.target

            if not isinstance(gathered_args, tuple):
                raise AssertionError(
                    f"Expected tuple for gathered_args, got {type(gathered_args)}"
                )
            if not isinstance(gathered_kwargs, dict):
                raise AssertionError(
                    f"Expected dict for gathered_kwargs, got {type(gathered_kwargs)}"
                )
            name = node.name if keep_original_node_name else None
            new_node = partition.graph.create_node(
                op=node.op,
                target=target,
                args=gathered_args,
                kwargs=gathered_kwargs,
                type_expr=node.type,
                name=name,
            )
            new_node.meta = node.meta.copy()
            partition.environment[node] = new_node

    # Exit epilogue
    for regions_mapping in [autocast_regions]:
        for node in reversed(regions_mapping):
            regions = regions_mapping[node]
            if len(regions) == 0:
                raise AssertionError("Expected at least one region")
            # pyrefly: ignore [bad-index, index-error]
            for r in regions[:-1]:
                partition = partitions[str(r)]
                exit_node = autocast_exits[node]
                if exit_node is None:
                    raise AssertionError("Missing exit node")
                new_node = partition.graph.create_node(
                    op=exit_node.op,
                    target=exit_node.target,
                    args=(partition.environment[node],),
                    kwargs={},
                    type_expr=exit_node.type,
                )
                new_node.meta = (
                    exit_node.meta.copy()
                )  # is it really a good idea to copy this?

    # original module environment dict mapping node names to nodes
    orig_mod_env: dict[str, Node] = {}
    # Set up values to construct base module
    base_mod_env: dict[str, Node] = {}
    base_mod_graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
    base_mod_attrs: dict[str, torch.fx.graph_module.GraphModule] = {}
    if not keep_original_order:
        for node in m.graph.nodes:
            base_mod_env, base_mod_attrs = construct_graph(
                node, base_mod_env, base_mod_attrs
            )

    else:
        # Go through the graph to construct the mapping dict
        for node in m.graph.nodes:
            orig_mod_env[node.name] = node

    # Do some things iterating over the partitions in topological order again:
    # 1) Finish off submodule Graphs by setting corresponding outputs
    # 2) Construct GraphModules for each submodule
    # 3) Construct the base graph by emitting calls to those submodules in
    #    topological order or original order specified by keep_original_order

    construct_order_partitions = (
        sorted_partitions if not keep_original_order else original_partition_order
    )

    already_constructed_attr_nodes = set()

    # We actually need to insert the placeholder nodes in the original order
    # otherwise graph signature will be wrong.
    original_order = [node for node in m.graph.nodes if node.op == "placeholder"]

    for partition_name in construct_order_partitions:
        partition = partitions[partition_name]

        # Set correct output values
        output_vals = tuple(
            partition.environment[orig_nodes[name]] for name in partition.outputs
        )

        if len(output_vals) == 1 and not tuple_return:
            partition.graph.output(output_vals[0])
        else:
            partition.graph.output(output_vals)

        if keep_original_order:
            # first get the attr nodes required by this partition
            orig_mod_attr_nodes: list[Node] = [
                orig_mod_env[key]
                for key in partition.inputs
                if key not in original_order
            ]

            for node in original_order:
                if node in already_constructed_attr_nodes:
                    continue  # already added this attr to the base graph
                base_mod_env, _based_mod_attrs = construct_graph(
                    node, base_mod_env, base_mod_attrs
                )
                already_constructed_attr_nodes.add(node)

            # Construct GraphModule for this partition
            for node in orig_mod_attr_nodes:  # type: ignore[attr-defined]
                if node in already_constructed_attr_nodes:
                    continue
                base_mod_env, base_mod_attrs = construct_graph(
                    node, base_mod_env, base_mod_attrs
                )
                already_constructed_attr_nodes.add(node)

        base_mod_attrs[partition.submod_name] = _make_graph_module(
            partition.targets, partition.graph
        )

        # Emit call in base graph to this submodule
        output_val = base_mod_graph.call_module(
            partition.submod_name,
            tuple(base_mod_env[name] for name in partition.inputs),
        )

        num_outputs = len(partition.outputs)
        if num_outputs > 1 or (num_outputs == 1 and tuple_return):
            # Unpack return values from submodule
            output_val_proxy = torch.fx.proxy.Proxy(output_val)
            for i, output_name in enumerate(partition.outputs):
                base_mod_env[output_name] = output_val_proxy[i].node  # type: ignore[index]
        elif num_outputs == 1:
            base_mod_env[next(iter(partition.outputs))] = output_val

    # When keep_original_order=True and if the graph doesn't have any
    # `call_function` node then `base_mod_graph`, `base_mod_env` and `base_mod_attrs`
    # are never populated.
    # For this case, we call `construct_graph` here which takes care of updating them.
    if keep_original_order and not base_mod_env:
        for node in m.graph.nodes:
            base_mod_env, base_mod_attrs = construct_graph(
                node, base_mod_env, base_mod_attrs
            )

    # Add output node to `base_mod_graph` (i.e. the split graph) which will be returned.
    for node in m.graph.nodes:
        if node.op == "output":
            base_mod_graph.output(
                torch.fx.graph.map_arg(node.args[0], lambda n: base_mod_env[n.name])
            )

    ret = _make_graph_module(base_mod_attrs, base_mod_graph)
    # Record which children are partitions (vs hoisted attributes like HOP
    # body subgraphs) so callers can distinguish them.
    ret.meta["partition_names"] = {p.submod_name for p in partitions.values()}
    log.debug(
        "%s",
        lazy_format_graph_code("post split_module", ret, colored=True),
    )
    return ret


def _decompose_size_nodes(m: GraphModule) -> None:
    """Decompose x.size() into per-dim sym_size.int calls.

    torch.Size objects cannot cross split boundaries because aot_autograd
    cannot handle them as submodule outputs. This replaces each size() call
    with individual sym_size.int(x, dim) nodes:
      - Dynamic dims (SymInt) -> new sym_size.int node
      - Static dims (plain int) -> inlined as literal constant
    """
    # Collect upfront since we mutate the graph during iteration.
    size_nodes = list(m.graph.find_nodes(op="call_method", target="size"))

    for node in size_nodes:
        # Skip size(dim) calls -- only decompose full size() with no dim argument.
        if len(node.args) > 1 or node.kwargs:
            continue
        tensor_node = node.args[0]
        ev = tensor_node.meta.get("example_value")
        if ev is None:
            continue

        dims: list[Node | int] = []
        with m.graph.inserting_after(tensor_node):
            for i, dim_val in enumerate(ev.shape):
                if isinstance(dim_val, torch.SymInt):
                    dn = m.graph.call_function(
                        torch.ops.aten.sym_size.int, args=(tensor_node, i)
                    )
                    dn.meta["example_value"] = dim_val
                    dims.append(dn)
                elif isinstance(dim_val, int):
                    dims.append(dim_val)

        for user in list(node.users):
            new_args = []
            for arg in user.args:
                if arg is node:
                    new_args.extend(dims)
                else:
                    new_args.append(arg)
            user.args = tuple(new_args)
        m.graph.erase_node(node)


def _make_lite_graph_module(
    graph: torch.fx.graph.Graph,
    modules: dict[str, GraphModule] | None = None,
) -> GraphModule:
    """Construct a lightweight GraphModule that bypasses expensive init overhead.

    Creates a ``_LazyGraphModule`` instance without going through
    ``GraphModule.__new__`` (MRO traversal + per-instance class creation),
    ``nn.Module.__init__`` (17+ container allocations), or
    ``GraphModule.__init__`` (node iteration + ``recompile()`` codegen).
    Per-instance classes are still created for lazy forward dispatch.

    Only suitable for parameterless graph containers (no parameters, buffers,
    or ``get_attr``/``call_module`` targets to copy from a root module).

    The dict entries below mirror nn.Module.__init__ (torch/nn/modules/module.py).
    Update in lockstep if nn.Module adds or removes instance attributes.
    """
    cls = type("GraphModuleImpl", (_LazyGraphModule,), {})
    inst = object.__new__(cls)
    d = inst.__dict__
    d["training"] = False
    d["_parameters"] = dict[str, Any]()
    d["_buffers"] = dict[str, Any]()
    d["_modules"] = modules if modules is not None else dict[str, Any]()
    d["_non_persistent_buffers_set"] = set[str]()
    d["_backward_pre_hooks"] = OrderedDict()
    d["_backward_hooks"] = OrderedDict()
    d["_is_full_backward_hook"] = None
    d["_forward_hooks"] = OrderedDict()
    d["_forward_hooks_with_kwargs"] = OrderedDict()
    d["_forward_hooks_always_called"] = OrderedDict()
    d["_forward_pre_hooks"] = OrderedDict()
    d["_forward_pre_hooks_with_kwargs"] = OrderedDict()
    d["_state_dict_hooks"] = OrderedDict()
    d["_state_dict_pre_hooks"] = OrderedDict()
    d["_load_state_dict_pre_hooks"] = OrderedDict()
    d["_load_state_dict_post_hooks"] = OrderedDict()
    d["_graph"] = graph
    d["meta"] = dict[str, Any]()
    graph.owning_module = inst
    cls.forward = _LazyGraphModule._lazy_forward  # type: ignore[attr-defined]
    return inst


def _detect_dependencies(
    m: GraphModule,
    node_to_partition: dict[Node, int],
) -> tuple[
    dict[int, torch.fx.graph.Graph],
    dict[int, dict[str, Node]],
    dict[int, dict[str, Node]],
    list[int],
]:
    """Assign nodes to partitions and detect cross-partition dependencies.

    Returns:
        partition_graphs: Maps partition ID to its Graph.
        partition_inputs: Maps partition ID to ordered dict of input names -> source nodes.
        partition_outputs: Maps partition ID to ordered dict of output names -> source nodes.
        seen_partitions: Partition IDs in order of first appearance.
    """
    import sympy

    from torch.fx.experimental.symbolic_shapes import free_symbols

    partition_graphs: dict[int, torch.fx.graph.Graph] = {}
    partition_inputs: dict[int, dict[str, Node]] = defaultdict(dict)
    partition_outputs: dict[int, dict[str, Node]] = defaultdict(dict)
    symbol_to_node: dict[sympy.Symbol, Node] = {}
    seen_partitions: list[int] = []
    seen_partitions_set: set[int] = set()

    def _record_output_dep(
        n: Node,
        _outputs: dict[int, dict[str, Node]] = partition_outputs,
    ) -> Node:
        if hasattr(n, "_fx_partition"):
            _outputs[n._fx_partition].setdefault(n.name, n)
        return n

    for node in m.graph.nodes:
        # Track SymInt symbol bindings (prefer earlier bindings).
        val = node.meta.get("example_value")
        if val is not None and hasattr(val, "node") and hasattr(val.node, "expr"):
            s0 = val.node.expr
            if isinstance(s0, sympy.Symbol) and s0 not in symbol_to_node:
                symbol_to_node[s0] = node

        if node.op == "placeholder":
            continue

        if node.op == "output":
            torch.fx.graph.map_arg(node.args[0], _record_output_dep)
            continue

        pid = node_to_partition[node]
        node._fx_partition = pid

        if pid not in seen_partitions_set:
            seen_partitions.append(pid)
            seen_partitions_set.add(pid)
            partition_graphs[pid] = torch.fx.graph.Graph()

        def _record_cross_dep(
            def_node: Node,
            use_pid: int = pid,
            _inputs: dict[int, dict[str, Node]] = partition_inputs,
            _outputs: dict[int, dict[str, Node]] = partition_outputs,
            _sym: dict[sympy.Symbol, Node] = symbol_to_node,
        ) -> Node:
            def_pid = getattr(def_node, "_fx_partition", None)
            if def_pid != use_pid:
                if def_pid is not None:
                    _outputs[def_pid].setdefault(def_node.name, def_node)
                _inputs[use_pid].setdefault(def_node.name, def_node)

                def_val = def_node.meta.get("example_value")
                if def_val is not None and _sym:
                    for s in sorted(free_symbols(def_val), key=str):
                        s_node = _sym.get(s)
                        if s_node is None:
                            continue
                        _inputs[use_pid].setdefault(s_node.name, s_node)
                        if s_node.op != "placeholder":
                            s_pid = getattr(s_node, "_fx_partition", None)
                            if s_pid is not None:
                                _outputs[s_pid].setdefault(s_node.name, s_node)
            return def_node

        torch.fx.graph.map_arg(node.args, _record_cross_dep)
        torch.fx.graph.map_arg(node.kwargs, _record_cross_dep)

    return partition_graphs, partition_inputs, partition_outputs, seen_partitions


def _clone_nodes_into_partitions(
    m: GraphModule,
    partition_graphs: dict[int, torch.fx.graph.Graph],
    partition_inputs: dict[int, dict[str, Node]],
    partition_outputs: dict[int, dict[str, Node]],
    seen_partitions: list[int],
) -> dict[int, dict[Node, Node]]:
    """Create placeholders, clone nodes, and set outputs for each partition.

    Returns:
        partition_env: Maps partition ID to {original_node: cloned_node} dict.
    """
    partition_env: dict[int, dict[Node, Node]] = defaultdict(dict)

    # Create placeholders for cross-partition inputs
    for pid in seen_partitions:
        g = partition_graphs[pid]
        env = partition_env[pid]
        for inp_name, orig_node in partition_inputs[pid].items():
            placeholder = g.placeholder(inp_name, type_expr=orig_node.type)
            placeholder.meta = orig_node.meta.copy()
            env[orig_node] = placeholder

    # Clone operational nodes
    for node in m.graph.nodes:
        if not hasattr(node, "_fx_partition"):
            continue
        pid = node._fx_partition
        g = partition_graphs[pid]
        env = partition_env[pid]

        gathered_args = torch.fx.graph.map_arg(node.args, env.__getitem__)
        gathered_kwargs = torch.fx.graph.map_arg(node.kwargs, env.__getitem__)

        new_node = g.create_node(
            op=node.op,
            target=node.target,
            args=gathered_args,
            kwargs=gathered_kwargs,
            type_expr=node.type,
        )
        new_node.meta = node.meta.copy()
        env[node] = new_node

    # Set output nodes
    for pid in seen_partitions:
        g = partition_graphs[pid]
        env = partition_env[pid]
        out_nodes = partition_outputs[pid]
        output_vals = tuple(env[orig_node] for orig_node in out_nodes.values())
        if len(output_vals) == 1:
            g.output(output_vals[0])
        elif len(output_vals) > 1:
            g.output(output_vals)
        else:
            g.output(())

    return partition_env


def _build_stitching_graph(
    m: GraphModule,
    partition_graphs: dict[int, torch.fx.graph.Graph],
    partition_inputs: dict[int, dict[str, Node]],
    partition_outputs: dict[int, dict[str, Node]],
    seen_partitions: list[int],
    partition_affix: str | None = None,
) -> GraphModule:
    """Build the outer stitching graph that calls each partition submodule."""
    base_graph = torch.fx.graph.Graph()
    base_env: dict[str, Node] = {}
    base_modules: dict[str, GraphModule] = {}

    for node in m.graph.nodes:
        if node.op == "placeholder":
            p = base_graph.placeholder(
                node.target,
                type_expr=node.type,
            )
            p.meta = node.meta.copy()
            base_env[node.name] = p

    for pid in seen_partitions:
        if partition_affix is not None:
            submod_name = f"submod_{partition_affix}_{pid}"
        else:
            submod_name = f"submod_{pid}"
        base_modules[submod_name] = _make_lite_graph_module(partition_graphs[pid])

        input_nodes = tuple(base_env[name] for name in partition_inputs[pid])
        output_val = base_graph.call_module(submod_name, input_nodes)

        out_names = list(partition_outputs[pid].keys())
        if len(out_names) > 1:
            output_val_proxy = torch.fx.proxy.Proxy(output_val)
            for i, name in enumerate(out_names):
                base_env[name] = output_val_proxy[i].node  # type: ignore[index]
        elif len(out_names) == 1:
            base_env[out_names[0]] = output_val

    for node in m.graph.nodes:
        if node.op == "output":
            base_graph.output(
                torch.fx.graph.map_arg(node.args[0], lambda n: base_env[n.name])
            )

    return _make_lite_graph_module(base_graph, modules=base_modules)


@compatibility(is_backward_compatible=False)
def split_module_simple(
    m: GraphModule,
    node_to_partition: dict[Node, int],
    *,
    partition_affix: str | None = None,
) -> GraphModule:
    """Lightweight graph splitter for simple partition patterns.

    A faster alternative to :func:`split_module` for inference-only graphs
    from ``torch.compile``/Dynamo. Because these graphs have no autocast/grad
    regions, no ``get_attr`` nodes, and no non-linear partition dependencies,
    we can skip the topological sort, autocast tracking, and ``get_attr``
    special-casing that ``split_module`` performs. More importantly, we
    construct partition submodules as lightweight ``_LazyGraphModule``
    instances that bypass ``nn.Module.__init__`` and defer ``recompile()``
    codegen — this eliminates the dominant cost when creating 70-100+
    partition submodules for large models.

    Args:
        m: Graph module to split.
        node_to_partition: Maps each operational node to a partition ID.
            Placeholders, get_attr, and output nodes should NOT be included.
        partition_affix: If set, submodule names become
            ``submod_{affix}_{idx}`` instead of ``submod_{idx}``.
    """
    partition_graphs, partition_inputs, partition_outputs, seen_partitions = (
        _detect_dependencies(m, node_to_partition)
    )

    _clone_nodes_into_partitions(
        m, partition_graphs, partition_inputs, partition_outputs, seen_partitions
    )

    return _build_stitching_graph(
        m,
        partition_graphs,
        partition_inputs,
        partition_outputs,
        seen_partitions,
        partition_affix,
    )
<|endoftext|>import traceback
from typing import Any, NamedTuple

import torch
import torch.fx
from torch._dispatch.python import enable_python_dispatcher
from torch._guards import detect_fake_mode
from torch._prims_common import is_contiguous_for_memory_format_or_false
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx._compatibility import compatibility
from torch.fx.node import map_aggregate, Node


__all__ = ["TensorMetadata", "ShapeProp"]


@compatibility(is_backward_compatible=True)
class TensorMetadata(NamedTuple):
    """A structure containing pertinent information about a tensor within a PyTorch program."""

    # General Tensor metadata
    shape: torch.Size
    dtype: torch.dtype
    requires_grad: bool
    stride: tuple[int, ...]
    memory_format: torch.memory_format | None

    # Quantization metadata
    is_quantized: bool
    qparams: dict[str, Any]


# When include_contiguity is True, we will set contiguity when it's always true for the tensor.
# Some tensors can represent both contiguous and non-contiguous tensors. e.g: (u0, u1) with (u2, u3).
# In such situation contiguity is not set. We could also make it a tri-state i.e: (def_contiguous,
# def_not_contiguous and unknown).
def _extract_tensor_metadata(
    result: torch.Tensor, include_contiguity: bool = True
) -> TensorMetadata:
    """
    Extract a TensorMetadata NamedTuple describing `result`.
    """
    shape = result.shape
    dtype = result.dtype
    requires_grad = result.requires_grad
    stride = result.stride() if not is_sparse_any(result) else ()

    memory_format = None

    if include_contiguity and not is_sparse_any(result):
        memory_formats = (
            torch.contiguous_format,
            torch.channels_last,
            torch.channels_last_3d,
        )
        for query_format in memory_formats:
            if is_contiguous_for_memory_format_or_false(
                result, memory_format=query_format
            ):
                memory_format = query_format
                break

    is_quantized = result.is_quantized
    qparams: dict[str, Any] = {}
    if is_quantized:
        qscheme = result.qscheme()
        qparams["qscheme"] = qscheme
        if qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            qparams["scale"] = result.q_scale()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_zero_point()  # type: ignore[assignment]
        elif qscheme in (
            torch.per_channel_affine,
            torch.per_channel_affine_float_qparams,
            torch.per_channel_symmetric,
        ):
            # In this branch, scale and zero_point are expected to be tensors,
            # we store the values as immutable_list in TensorMetadata for
            # easier serialization downstream
            qparams["scale"] = result.q_per_channel_scales().tolist()  # type: ignore[assignment]
            qparams["zero_point"] = result.q_per_channel_zero_points().tolist()  # type: ignore[assignment]
            qparams["axis"] = result.q_per_channel_axis()  # type: ignore[assignment]

    return TensorMetadata(
        shape, dtype, requires_grad, stride, memory_format, is_quantized, qparams
    )


@compatibility(is_backward_compatible=True)
class ShapeProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and
    record the shape and type of the result
    into the corresponding node.

    Example:
         In this example, we record the shape
         and data type of a module given
         an example input ``torch.randn(50, D_in)``.
         We print the name, shape and dtype of each node.

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super().__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)
            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred
        N, D_in, H, D_out = 64, 1000, 100, 10
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        model = TwoLayerNet(D_in, H, D_out)
        gm = torch.fx.symbolic_trace(model)
        sample_input = torch.randn(50, D_in)
        ShapeProp(gm).propagate(sample_input)

        for node in gm.graph.nodes:
            print(node.name, node.meta['tensor_meta'].dtype,
                node.meta['tensor_meta'].shape)

        The output of this code is:

        x torch.float32 torch.Size([50, 1000])
        linear1 torch.float32 torch.Size([50, 100])
        clamp_1 torch.float32 torch.Size([50, 100])
        linear2 torch.float32 torch.Size([50, 10])
        output torch.float32 torch.Size([50, 10])

    Args:
         module (GraphModule): The module to be executed
         fake_mode (FakeTensorMode): A fake mode for copying the gm

    """

    def __init__(self, gm: torch.fx.GraphModule, fake_mode: Any = None) -> None:
        super().__init__(gm)
        if fake_mode is None:
            fake_mode = detect_fake_mode()
        if fake_mode is not None:
            from torch._dynamo.utils import deepcopy_to_fake_tensor

            # Note:
            # We need fake execution cause the inputs are fake, however, we cannot fakify the module
            # - because we need to write to the tensor_meta of the real module. So we fakify to
            # produce a result (L131 below), to extract tensor meta, and then keep going.
            #
            # If we were to fakify, we would write to the wrong node, and then downstream fusion
            # would be missing the tensor_meta.
            #
            # See torch/_inductor/overrides.py for where this is called upstream of fusion.
            self.fake_module = deepcopy_to_fake_tensor(self.module, fake_mode)
            self.fake_mode = fake_mode
        else:
            self.fake_module = None
            self.fake_mode = None

        self.real_module = self.module

    def run_node(self, n: Node) -> Any:
        from torch.fx.experimental.symbolic_shapes import (
            compute_unbacked_bindings,
            rebind_unbacked,
        )

        try:
            if self.fake_module is not None:
                # Hacky swap. Alternatively, we could do this with overriding
                # call_module and get_attr.
                self.module = self.fake_module
            try:
                if self.fake_mode is not None:
                    with self.fake_mode, enable_python_dispatcher():
                        result = super().run_node(n)
                        rebind_unbacked(self.fake_mode.shape_env, n, result)
                else:
                    result = super().run_node(n)
            finally:
                self.module = self.real_module
        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(
                f"ShapeProp error for: node={n.format_node()} with meta={n.meta}"
            ) from e

        found_tensor = False

        def extract_tensor_meta(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                nonlocal found_tensor
                found_tensor = True
                return _extract_tensor_metadata(obj)
            else:
                return obj

        meta = map_aggregate(result, extract_tensor_meta)
        if found_tensor:
            n.meta["tensor_meta"] = meta

        if self.fake_mode:
            if (shape_env := self.fake_mode.shape_env) and (
                symbol_to_path := compute_unbacked_bindings(shape_env, result)
            ):
                n.meta["unbacked_bindings"] = symbol_to_path

        n.meta["type"] = type(result)
        return result

    def propagate(self, *args: Any) -> Any:
        """
        Run `module` via interpretation and return the result and
        record the shape and type of each node.

        Args:
            *args (Tensor): the sample input.

        Returns:
            Any: The value returned from executing the Module
        """
        if self.fake_mode is not None:
            fake_args = [
                self.fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
                for t in args
            ]
        else:
            fake_args = args
        return super().run(*fake_args)
<|endoftext|>"""Canonicalize an FX graph's node order and names.

Provides two passes:

- ``canonicalize_graph``: reorders nodes into a deterministic topological order
  using Kahn's algorithm with a caller-supplied canonical key function and
  barrier predicate.
- ``rename_nodes_to_canonical``: renames all nodes to canonical names derived
  from their target, using the same naming scheme as ``Graph.create_node``.
"""

import collections
import heapq
import itertools
import operator
from collections.abc import Callable

import torch
import torch.fx as fx


__all__ = ["canonicalize_graph", "rename_nodes_to_canonical"]


_IN_PLACE_OPERATORS = frozenset(
    {
        "iadd",
        "iand",
        "iconcat",
        "ifloordiv",
        "ilshift",
        "imatmul",
        "imod",
        "imul",
        "ior",
        "ipow",
        "irshift",
        "isub",
        "itruediv",
        "ixor",
    }
)


def _computation_node_key(
    node: fx.Node, canonical_idx: dict[fx.Node, int]
) -> tuple[int, str, tuple[int, ...]]:
    """Canonical heap key for a computation node (call_function / call_method / call_module)."""
    input_indices = tuple(canonical_idx[n] for n in node.all_input_nodes)
    return (2, node.graph._target_to_str(node.target), input_indices)


def _canonical_node_key(node: fx.Node, canonical_idx: dict[fx.Node, int]) -> object:
    """Canonical heap key for get_attr, output, and computation nodes.

    Callers must handle placeholder nodes themselves (the ordering strategy
    differs between Dynamo and export) and never pass them here.
    """
    if node.op == "placeholder":
        raise AssertionError("callers must handle placeholder nodes themselves")
    if node.op == "get_attr":
        return (1, str(node.target))
    elif node.op == "output":
        return (3,)
    else:
        return _computation_node_key(node, canonical_idx)


def _is_safe_to_reorder(node: fx.Node) -> bool:
    """Check if a node is safe to reorder during graph canonicalization.

    Builds on Node.is_impure() (used by DCE) with two additional checks for
    cases it doesn't cover: in-place call_method nodes and non-OpOverload
    state-changing functions detected by a no-node-arguments heuristic.
    """
    if node.op == "call_method":
        return not node.target.endswith("_")  # pyrefly: ignore[missing-attribute]
    if node.op == "call_module":
        return not node.is_impure()
    if node.op != "call_function":
        return True
    if node.is_impure():
        return False
    if not isinstance(node.target, torch._ops.OpOverload):
        name = getattr(node.target, "__name__", "")
        if name.endswith("_"):
            return False
        if (
            getattr(node.target, "__module__", "") == "_operator"
            and name in _IN_PLACE_OPERATORS
        ):
            return False
        if isinstance(node.kwargs.get("out"), fx.Node):
            return False
        # triton_kernel_wrapper_mutation mutates tensors via kwargs but
        # is not detected by is_impure() or trailing-underscore checks.
        if name == "triton_kernel_wrapper_mutation":
            return False
        # Non-OpOverload targets with no FX Node arguments are likely
        # state-changing (e.g., _vmap_increment_nesting,
        # _set_fwd_grad_enabled). This is intentionally conservative:
        # pure constant-producing ops would also be treated as barriers,
        # but those are rare in Dynamo output graphs (constants are
        # typically lifted as placeholders or get_attr nodes).
        if not node.all_input_nodes:
            return False
        # functorch batch dim ops modify the vmap interpreter stack.
        if name in ("_add_batch_dim", "_remove_batch_dim"):
            return False
    return True


def rename_nodes_to_canonical(
    graph: fx.Graph,
    skip_ops: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Rename all nodes in the graph to canonical names based on their target.

    Uses the same naming scheme as FX ``Graph.create_node`` (auto-generated
    names from the target string). After renaming, replaces the graph's
    namespace so future node creation stays consistent.

    Args:
        graph: The FX graph whose nodes to rename.
        skip_ops: Node ops to skip renaming (their names are reserved in the
            new namespace but left unchanged).

    Returns a mapping from old name to new name for nodes that were renamed.
    """
    from torch.fx.graph import _Namespace

    renamed: dict[str, str] = {}
    ns = _Namespace()
    for node in graph.nodes:
        if node.op in skip_ops:
            ns.create_name(node.name, node)
            continue
        old_name = node.name
        candidate = graph._target_to_str(node.target)
        new_name = ns.create_name(candidate, node)
        if new_name != old_name:
            renamed[old_name] = new_name
        node.name = new_name
    graph._graph_namespace = ns
    return renamed


def _sink_get_attr_nodes(order: list[fx.Node]) -> None:
    """Move each get_attr node to right before its earliest consumer.

    By default, Kahn's algorithm places get_attr nodes (which have no data
    dependencies) at the top of the graph.  This post-processing step sinks
    each one to just before its first consumer, keeping definitions close to
    their uses.
    """
    non_ga = [n for n in order if n.op != "get_attr"]
    gas = [n for n in order if n.op == "get_attr"]
    if not gas:
        return

    pos = {n: i for i, n in enumerate(non_ga)}
    inserts: dict[int, list[fx.Node]] = collections.defaultdict(list)
    for ga in gas:
        if ga.users:
            target = min(pos.get(u, len(non_ga)) for u in ga.users)
        else:
            target = (
                len(non_ga) - 1 if non_ga and non_ga[-1].op == "output" else len(non_ga)
            )
        inserts[target].append(ga)

    order.clear()
    for i, node in enumerate(non_ga):
        order.extend(inserts.pop(i, ()))
        order.append(node)
    for remaining in inserts.values():
        order.extend(remaining)


def _group_getitem_nodes(order: list[fx.Node]) -> None:
    """Move each getitem node to immediately after its producer, in index order.

    Serialization does not store getitem nodes as free-standing nodes; the
    deserializer reconstructs them right after their producer, index-ordered
    (see ``generate_getitems`` in ``torch/_export/serde/serialize.py``).  If
    canonicalization leaves getitems interleaved with other computation, the
    serialize/deserialize roundtrip produces a different node order.  This pass
    normalizes getitem placement to match the deserializer so the canonical
    form is roundtrip-stable.  Nested getitems (from list-typed outputs) are
    emitted depth-first, matching the recursive reconstruction.

    Relies on each getitem's producer (``args[0]``) being a node present in
    ``order``, and on a getitem sharing its producer's ``nn_module_stack``
    scope (so pulling it adjacent stays within that scope).  Both hold for
    graphs FX/export produce; the count assertion below guards against silent
    node loss if they ever don't.
    """
    children: dict[fx.Node, list[fx.Node]] = collections.defaultdict(list)
    getitems: set[fx.Node] = set()
    for node in order:
        if node.op == "call_function" and node.target is operator.getitem:
            children[node.args[0]].append(node)  # type: ignore[index]
            getitems.add(node)
    if not getitems:
        return

    def _index(node: fx.Node) -> tuple[int, int, str]:
        idx = node.args[1]
        return (0, idx, "") if isinstance(idx, int) else (1, 0, str(idx))

    for group in children.values():
        group.sort(key=_index)

    new_order: list[fx.Node] = []

    def _emit(node: fx.Node) -> None:
        new_order.append(node)
        for child in children.get(node, ()):
            _emit(child)

    for node in order:
        if node not in getitems:
            _emit(node)
    if len(new_order) != len(order):
        raise AssertionError(
            f"getitem grouping lost nodes: {len(new_order)} != {len(order)} "
            "(a getitem's producer is missing from the order)"
        )
    order[:] = new_order


def canonicalize_graph(
    graph: fx.Graph,
    canonical_key_fn: Callable[[fx.Node, dict[fx.Node, int]], object],
    is_safe_to_reorder: Callable[[fx.Node], bool],
    *,
    skip_rename_ops: frozenset[str] = frozenset(),
    group_getitems: bool = False,
) -> dict[str, str]:
    """Reorder graph nodes into a canonical topological order and rename them.

    This ensures that structurally equivalent graphs produce identical node
    names and ordering, regardless of the order in which nodes were originally
    traced.

    Uses Kahn's algorithm with a canonical tiebreaker provided by
    ``canonical_key_fn``.

    Args:
        graph: The FX graph to canonicalize. Modified in-place.
        canonical_key_fn: ``(node, canonical_idx) -> comparable tuple``.
            Called when a node becomes ready.  ``canonical_idx`` maps already-
            ordered nodes to their position.  The returned tuple is used as the
            primary heap key.
        is_safe_to_reorder: ``(node) -> bool``.  Nodes for which this returns
            ``False`` act as barriers: they are chained in original order, and
            pure nodes are confined to their barrier segment.
        skip_rename_ops: Node ops to skip renaming. Skipped nodes keep their
            original names.
        group_getitems: If True, place getitem nodes immediately after their
            producer (index-ordered) so the order matches what serialization's
            deserializer reconstructs. Enable for graphs that get serialized
            (e.g. export) to keep the serialize/deserialize roundtrip stable.

    Returns:
        A mapping from old node name to new node name for nodes that were
        renamed.
    """
    indeg: dict[fx.Node, int] = {
        node: len(node.all_input_nodes) for node in graph.nodes
    }

    # Nodes that aren't provably pure act as barriers. We partition the graph
    # into segments separated by barrier nodes and add synthetic edges:
    #   prev_barrier -> reorderable_nodes_in_segment -> next_barrier
    extra_users: dict[fx.Node, list[fx.Node]] = collections.defaultdict(list)
    prev_barrier: fx.Node | None = None
    segment_reorderable: list[fx.Node] = []
    for node in graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        is_barrier = not is_safe_to_reorder(node)
        if is_barrier:
            for reorderable in segment_reorderable:
                extra_users[reorderable].append(node)
                indeg[node] += 1
            segment_reorderable = []
        if prev_barrier is not None:
            extra_users[prev_barrier].append(node)
            indeg[node] += 1
        if is_barrier:
            prev_barrier = node
        else:
            segment_reorderable.append(node)

    canonical_idx: dict[fx.Node, int] = {}

    # The counter is a tiebreaker that prevents heapq from comparing
    # fx.Node objects (which have no __lt__). It only affects nodes with
    # identical canonical keys -- i.e., structurally equivalent operations
    # (same target, same input indices). These are usually CSE-equivalent and
    # value-interchangeable, so the fallback to trace order is fine. This is
    # best-effort canonical labeling: it does not refine equal-key nodes by
    # their consumers' argument positions, so two graphs that are isomorphic
    # only under swapping such nodes may still get different (but value-equal)
    # labelings.
    counter = 0
    ready: list[tuple[object, int, fx.Node]] = []
    for node in graph.nodes:
        if indeg[node] == 0:
            ready.append((canonical_key_fn(node, canonical_idx), counter, node))
            counter += 1
    heapq.heapify(ready)

    canonical_order: list[fx.Node] = []

    while ready:
        _, _, cur = heapq.heappop(ready)
        canonical_order.append(cur)
        canonical_idx[cur] = len(canonical_idx)

        for user in itertools.chain(cur.users, extra_users.get(cur, ())):
            indeg[user] -= 1
            if indeg[user] == 0:
                heapq.heappush(
                    ready,
                    (canonical_key_fn(user, canonical_idx), counter, user),
                )
                counter += 1

    if len(canonical_order) != len(graph.nodes):
        remaining = [n for n in indeg if indeg[n] != 0]
        raise RuntimeError(
            f"Canonicalization failed: processed {len(canonical_order)} of "
            f"{len(graph.nodes)} nodes. Remaining: {remaining}"
        )

    _sink_get_attr_nodes(canonical_order)

    if group_getitems:
        _group_getitem_nodes(canonical_order)

    # Purge erased nodes that are still physically in the linked list.
    # erase_node() sets _erased=True and unlinks the node, but stale
    # _prev/_next pointers on the erased node can leave it reachable from
    # neighbors that were inserted later.  If such a ghost node sits between
    # `cursor` and the node being appended, cursor.append() (which goes via
    # cursor._next._prepend()) corrupts the chain.
    root = graph._root  # type: ignore[attr-defined]
    node = root._next
    while node is not root:
        nxt = node._next
        if node._erased:
            node._remove_from_list()
        node = nxt

    # Reorder nodes in-place to preserve node object identity.
    cursor = root
    for node in canonical_order:
        cursor.append(node)
        cursor = node

    return rename_nodes_to_canonical(graph, skip_ops=skip_rename_ops)
<|endoftext|>import logging
from collections.abc import Callable
from functools import wraps
from inspect import unwrap
from typing import Any, Concatenate, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")


logger = logging.getLogger(__name__)

__all__ = [
    "PassManager",
    "inplace_wrapper",
    "log_hook",
    "loop_pass",
    "this_before_that_pass_constraint",
    "these_before_those_pass_constraint",
]


# for callables which modify object inplace and return something other than
# the object on which they act
def inplace_wrapper(
    fn: Callable[Concatenate[_T, _P], Any],
) -> Callable[Concatenate[_T, _P], _T]:
    """
    Convenience wrapper for passes which modify an object inplace. This
    wrapper makes them return the modified object instead.

    Args:
        fn (Callable[Object, Any])

    Returns:
        wrapped_fn (Callable[Object, Object])
    """

    @wraps(fn)
    def wrapped_fn(gm: _T, *args: _P.args, **kwargs: _P.kwargs) -> _T:
        fn(gm, *args, **kwargs)
        return gm

    return wrapped_fn


def log_hook(fn: Callable[_P, _R], level: int = logging.INFO) -> Callable[_P, _R]:
    """
    Logs callable output.

    This is useful for logging output of passes. Note ``inplace_wrapper`` replaces
    the pass output with the modified object. If we want to log the original
    output, apply this wrapper before ``inplace_wrapper``.

    Example::

        def my_pass(d: Dict) -> bool:
            changed = False
            if "foo" in d:
                d["foo"] = "bar"
                changed = True
            return changed


        pm = PassManager(passes=[inplace_wrapper(log_hook(my_pass))])

    Args:
        fn (Callable[Type1, Type2])
        level: logging level (e.g. logging.INFO)

    Returns:
        wrapped_fn (Callable[Type1, Type2])
    """

    @wraps(fn)
    def wrapped_fn(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        val = fn(*args, **kwargs)
        logger.log(level, "Ran pass %s\t Return value: %s", fn, val)
        return val

    return wrapped_fn


def loop_pass(
    base_pass: Callable[[_T], _T],
    n_iter: int | None = None,
    predicate: Callable[[_T], bool] | None = None,
) -> Callable[[_T], _T]:
    """
    Convenience wrapper for passes which need to be applied multiple times.

    Exactly one of `n_iter`or `predicate` must be specified.

    Args:
        base_pass (Callable[Object, Object]): pass to be applied in loop
        n_iter (int, optional): number of times to loop pass
        predicate (Callable[Object, bool], optional):

    """
    if not ((n_iter is not None) ^ (predicate is not None)):
        raise AssertionError("Exactly one of `n_iter`or `predicate` must be specified.")

    @wraps(base_pass)
    def new_pass(source: _T) -> _T:
        output = source
        if n_iter is not None and n_iter > 0:
            for _ in range(n_iter):
                output = base_pass(output)
        elif predicate is not None:
            while predicate(output):
                output = base_pass(output)
        else:
            raise RuntimeError(
                f"loop_pass must be given positive int n_iter (given "
                f"{n_iter}) xor predicate (given {predicate})"
            )
        return output

    return new_pass


# Pass Schedule Constraints:
#
# Implemented as 'depends on' operators. A constraint is satisfied iff a list
# has a valid partial ordering according to this comparison operator.
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
                f" but found {a} at index {i} and {b} at index{j} in pass"
                f" list."
            )


def this_before_that_pass_constraint(
    this: Callable[..., Any], that: Callable[..., Any]
) -> Callable[[Callable[..., Any], Callable[..., Any]], bool]:
    """
    Defines a partial order ('depends on' function) where `this` must occur
    before `that`.
    """

    def depends_on(a: Callable[..., Any], b: Callable[..., Any]) -> bool:
        return a != that or b != this

    return depends_on


def these_before_those_pass_constraint(
    these: Callable[..., Any], those: Callable[..., Any]
) -> Callable[[Callable[..., Any], Callable[..., Any]], bool]:
    """
    Defines a partial order ('depends on' function) where ``these`` must occur
    before ``those``. Where the inputs are 'unwrapped' before comparison.

    For example, the following pass list and constraint list would be invalid::

        passes = [
            loop_pass(pass_b, 3),
            loop_pass(pass_a, 5),
        ]

        constraints = [these_before_those_pass_constraint(pass_a, pass_b)]

    Args:
        these (Callable): pass which should occur first
        those (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool])
    """

    def depends_on(a: Callable[..., Any], b: Callable[..., Any]) -> bool:
        return unwrap(a) != those or unwrap(b) != these

    return depends_on


class PassManager:
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): list of passes. A pass is a
            callable which modifies an object and returns modified object
        constraint (Optional[List[Callable]]): list of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
    """

    passes: list[Callable[..., Any]]
    constraints: list[Callable[..., Any]]
    _validated: bool = False

    def __init__(
        self,
        passes: list[Callable[..., Any]] | None = None,
        constraints: list[Callable[..., Any]] | None = None,
    ) -> None:
        self.passes = passes or []
        self.constraints = constraints or []

    @classmethod
    def build_from_passlist(cls, passes: list[Callable[..., Any]]) -> "PassManager":
        pm = PassManager(passes)
        # TODO(alexbeloi): add constraint management/validation
        return pm

    def add_pass(self, _pass: Callable[..., Any]) -> None:
        self.passes.append(_pass)
        self._validated = False

    def add_constraint(self, constraint: Callable[..., Any]) -> None:
        self.constraints.append(constraint)
        self._validated = False

    def remove_pass(self, _passes: list[str]) -> None:
        if _passes is None:
            return
        passes_left = [ps for ps in self.passes if ps.__name__ not in _passes]
        self.passes = passes_left
        self._validated = False

    def replace_pass(
        self, _target: Callable[..., Any], _replacement: Callable[..., Any]
    ) -> None:
        passes_left = []
        for ps in self.passes:
            if ps.__name__ == _target.__name__:
                passes_left.append(_replacement)
            else:
                passes_left.append(ps)
        self.passes = passes_left
        self._validated = False

    def validate(self) -> None:
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
        if self._validated:
            return
        for constraint in self.constraints:
            _validate_pass_schedule_constraint(constraint, self.passes)
        self._validated = True

    def __call__(self, source: Any) -> Any:
        self.validate()
        out = source
        for _pass in self.passes:
            out = _pass(out)
        return out
<|endoftext|>import os
from collections.abc import Callable
from types import TracebackType
from typing import TypeVar

from torch.fx import Graph, Node
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.traceback import NodeSource, NodeSourceAction


T = TypeVar("T")


from .graph_drawer import FxGraphDrawer


__all__ = ["GraphTransformObserver"]


@compatibility(is_backward_compatible=False)
class GraphTransformObserver:
    __pass_count = 0

    def __init__(
        self,
        gm: GraphModule,
        passname: str,
        subsystem: str | None = None,
        log_url: str | None = None,
    ) -> None:
        """
        log_url is inferred to be torch._inductor.config.trace.log_url_for_graph_xform unless otherwise specified
        """
        from torch._inductor import config as inductor_config

        self.gm = gm
        self.passname = passname
        self.subsystem = subsystem

        if log_url is None:
            log_url = inductor_config.trace.log_url_for_graph_xform

        self.log_url = log_url

        self.active = (
            self.log_url is not None
            or inductor_config.effective_provenance_tracking_level() == 1
        )

        if self.active:
            self.erased_nodes: set[str] = set()
            self.created_nodes: set[str] = set()
            self.name_to_node: dict[str, Node] = {}
            # record graph modules deepcopied from self.gm, so we can remove hooks on them when exiting the context
            self.copied_gms: list[GraphModule] = []

            self._node_creation_hook = self.get_node_creation_hook()
            self._node_erase_hook = self.get_node_erase_hook()
            self._node_replace_hook = self.get_node_replace_hook()
            self._deepcopy_hook = self.get_deepcopy_hook()

        # If log_url is None, we don't log anything
        if self.log_url is None:
            return
        GraphTransformObserver.__pass_count += 1

        self.input_dot_graph = FxGraphDrawer(
            self.gm,
            self.passname,
            ignore_getattr=True,
            ignore_parameters_and_buffers=True,
        ).get_dot_graph()

    @classmethod
    def get_current_pass_count(cls) -> int:
        return cls.__pass_count

    def apply_gm_pass(self, pass_fn: Callable[[GraphModule], T]) -> T | None:
        from torch._dynamo.utils import dynamo_timed

        with self:
            if self._check_disable_pass():
                return None
            with dynamo_timed(
                f"pass.{self.subsystem}.{self.passname}"
                if self.subsystem
                else f"pass.{self.passname}"
            ):
                return pass_fn(self.gm)

    def apply_graph_pass(self, pass_fn: Callable[[Graph], T]) -> T | None:
        from torch._dynamo.utils import dynamo_timed

        with self:
            if self._check_disable_pass():
                return None
            with dynamo_timed(
                f"pass.{self.subsystem}.{self.passname}"
                if self.subsystem
                else f"pass.{self.passname}"
            ):
                return pass_fn(self.gm.graph)

    def _check_disable_pass(self) -> bool:
        from torch._inductor import config as inductor_config

        if self.passname.upper() in inductor_config.disabled_passes.upper():
            return True

        if self.subsystem is None:
            return False

        from torch._inductor.compiler_bisector import CompilerBisector

        return CompilerBisector.disable_subsystem(
            "inductor", self.subsystem, lambda: self.passname
        )

    def __enter__(self) -> "GraphTransformObserver":
        if not self.active:
            return self
        self.gm._register_create_node_hook(self._node_creation_hook)
        self.gm._register_erase_node_hook(self._node_erase_hook)
        self.gm._register_replace_node_hook(self._node_replace_hook)
        self.gm._register_deepcopy_hook(self._deepcopy_hook)

        self.erased_nodes.clear()
        self.created_nodes.clear()
        self.name_to_node.clear()
        self.copied_gms.clear()

        for node in self.gm.graph.nodes:
            self.name_to_node[node.name] = node

        return self

    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if not self.active:
            return
        for gm in self.copied_gms + [self.gm]:
            gm._unregister_create_node_hook(self._node_creation_hook)
            gm._unregister_erase_node_hook(self._node_erase_hook)
            gm._unregister_replace_node_hook(self._node_replace_hook)
            gm._unregister_deepcopy_hook(self._deepcopy_hook)

        if self.log_url is None:
            return

        if len(self.created_nodes) > 0 or len(self.erased_nodes) > 0:
            for e in self.input_dot_graph.get_node_list():
                if e.get_name() in self.erased_nodes:
                    e.obj_dict["attributes"]["fillcolor"] = "yellow"
                else:
                    e.obj_dict["attributes"]["fillcolor"] = "grey"
            if self.log_url is None:
                raise AssertionError("log_url is not set")
            self.input_dot_graph.write(
                os.path.join(
                    self.log_url,
                    f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_input_graph.dot",
                )
            )

            output_dot_graph = FxGraphDrawer(
                self.gm,
                self.passname,
                ignore_getattr=True,
                ignore_parameters_and_buffers=True,
            ).get_dot_graph()
            for e in output_dot_graph.get_node_list():
                if e.get_name() in self.created_nodes:
                    e.obj_dict["attributes"]["fillcolor"] = "yellow"
                else:
                    e.obj_dict["attributes"]["fillcolor"] = "grey"
            output_dot_graph.write(
                os.path.join(
                    self.log_url,
                    f"pass_{GraphTransformObserver.__pass_count}_{self.passname}_output_graph.dot",
                )
            )

    def get_node_creation_hook(self) -> Callable[[Node], None]:
        # We have to return a function instead of using a class method directly
        # to avoid max recursion issue when deepcopy a graph module within the context manager.
        def on_node_creation(node: Node) -> None:
            self.created_nodes.add(node.name)
            self.name_to_node[node.name] = node
            source = NodeSource(None, self.passname, NodeSourceAction.CREATE)
            if "from_node" not in node.meta:
                node.meta["from_node"] = [source]
            else:
                node.meta["from_node"].append(source)

        return on_node_creation

    def get_node_erase_hook(self) -> Callable[[Node], None]:
        def on_node_erase(node: Node) -> None:
            self.erased_nodes.add(node.name)
            self.name_to_node.pop(node.name, None)

        return on_node_erase

    def get_node_replace_hook(self) -> Callable[[Node, str, Node], None]:
        def on_node_replace(old: Node, new: str, user: Node) -> None:
            # Update node meta when replacing old node with new node
            new_node = self.name_to_node.get(new, None)

            if not new_node:
                return

            if not isinstance(new_node, Node):
                raise AssertionError(f"Expected Node, got {type(new_node)}")

            # replace hook is called once for each user of old
            # this avoids adding duplicated source nodes
            added_nodes = {s.name for s in new_node.meta.get("from_node", [])}
            if old.name in added_nodes:
                return

            action = [NodeSourceAction.REPLACE]
            if new_node.name in self.created_nodes:
                action.append(NodeSourceAction.CREATE)

            def created_this_pass(source: NodeSource) -> bool:
                return source.pass_name == self.passname and source.action == [
                    NodeSourceAction.CREATE
                ]

            # remove redundant source added on node creation
            new_from_node = new_node.meta.get("from_node", [])
            new_from_node = [
                source for source in new_from_node if not created_this_pass(source)
            ]

            # add new source
            new_node_source = NodeSource(old, self.passname, action)
            new_from_node.append(new_node_source)
            new_node.meta["from_node"] = new_from_node

        return on_node_replace

    def get_deepcopy_hook(self) -> Callable[[GraphModule], None]:
        def on_deepcopy(gm: GraphModule) -> None:
            self.copied_gms.append(gm)

        return on_deepcopy
<|endoftext|>import contextlib
import functools
import logging
from collections.abc import Callable, Iterator, Mapping
from typing import Any, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")

import torch
from torch.fx._compatibility import compatibility


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor"]


# standalone_inductor returns a callable class object - this does not sit well
# with Fx graph node op call_function which expects a function. So this is just
# a wrapper function to make Fx graph codegen happy.
def _dummy_wrapper(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @functools.wraps(fn)
    def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        return fn(*args, **kwargs)

    return inner


@contextlib.contextmanager
def _disable_remat_for_regional_subcompile() -> Iterator[None]:
    # In torch.compile, regional_inductor subcompiles run after the enclosing
    # non-strict full graph has already been partitioned, so any graph-SAC
    # remat pass has already run before we reach this nested compile.
    # Rerunning remat here can see stage-2-reordered backward nodes that
    # violate remat's contiguous-backward-region assumption.
    with torch._functorch.config.patch(remat_using_tags_for_fwd_loss_bwd_graph=False):
        yield


def _compile_submod(gm: torch.fx.GraphModule, prefix: str) -> torch.fx.GraphModule:
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target.startswith(prefix):
            fake_inputs = []
            for inp_node in node.all_input_nodes:
                if hasattr(inp_node, "meta") and "val" in inp_node.meta:
                    fake_inputs.append(inp_node.meta["val"])
                else:
                    raise RuntimeError(
                        f"Partition is bad because non fake tensor value is seen {inp_node}"
                    )

            submod = getattr(gm, node.target)

            # Get inductor configs from annotation
            # TODO we should change partition when there are multiple differently
            # annotated regions.
            inductor_options: dict[str, Any] = {}
            for sub_node in submod.graph.nodes:
                if hasattr(sub_node, "meta") and sub_node.meta.get("custom", None):
                    custom = sub_node.meta["custom"]
                    if isinstance(custom, dict) and "compile_with_inductor" in custom:
                        compile_value = custom["compile_with_inductor"]
                        if (
                            isinstance(compile_value, dict)
                            and "inductor_configs" in compile_value
                        ):
                            inductor_options = compile_value["inductor_configs"]
                            break

            # Log the options being used
            logger.info(
                "Compiling submodule %s with inductor options: %s",
                node.target,
                inductor_options,
            )

            # Apply config patches before compilation
            import torch._inductor.config as inductor_config

            # Validate that all config keys exist
            for key in inductor_options:
                if not hasattr(inductor_config, key):
                    raise ValueError(
                        f"Invalid inductor config key '{key}' in regional_inductor annotation. "
                        f"Available config keys can be found in torch._inductor.config"
                    )

            with (
                inductor_config.patch(inductor_options),
                _disable_remat_for_regional_subcompile(),
                torch.fx.traceback._set_regional_inductor_subgraph_name(node.target),
            ):
                compiled_fn = torch._inductor.standalone_compile(
                    submod,
                    fake_inputs,
                    dynamic_shapes="from_tracing_context",
                    aot=True,
                    donate_graph_module=True,
                )
            if not isinstance(compiled_fn, AOTCompiledArtifact):
                raise AssertionError(
                    f"Expected AOTCompiledArtifact, got {type(compiled_fn)}"
                )
            # _dummy_wrapper is to make call_function happy
            compiled_submod = _dummy_wrapper(compiled_fn)
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    compiled_submod, args=node.args, kwargs=node.kwargs
                )
                new_node.meta = node.meta
                node.replace_all_uses_with(new_node)
                gm.graph.erase_node(node)
                del gm._modules[node.target]

    gm.recompile()
    return gm


def _needs_inductor_compile(node: torch.fx.Node) -> bool:
    return bool(
        node.op not in ("placeholder", "output")
        and hasattr(node, "meta")
        and node.meta.get("custom", None)
        and "compile_with_inductor" in node.meta["custom"]
    )


class _RegionScooper:
    """
    Scoops out the inductor marked regions. It does NOT compile them.
    """

    @staticmethod
    def scoop_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
        from torch.fx.passes.operator_support import create_op_support
        from torch.fx.passes.utils.fuser_utils import fuse_by_partitions

        # Group tagged nodes by region ID.  The region ID comes from the
        # optional "inductor_region" key inside the compile_with_inductor
        # annotation. When absent, all tagged nodes share a single default region
        _DEFAULT_REGION = object()
        regions: dict[object, set[torch.fx.Node]] = {}
        for node in gm.graph.nodes:
            if _needs_inductor_compile(node):
                compile_value = node.meta["custom"]["compile_with_inductor"]
                if (
                    isinstance(compile_value, dict)
                    and "inductor_region" in compile_value
                ):
                    rid = compile_value["inductor_region"]
                else:
                    rid = _DEFAULT_REGION
                regions.setdefault(rid, set()).add(node)

        if not regions:
            logger.info("No inductor marked nodes found")
            return gm

        # Run CapabilityBasedPartitioner per region to get cycle-safe partitions
        # without merging across region boundaries.
        def _is_in_region(
            region_nodes: set[torch.fx.Node],
        ) -> Callable[[Mapping[str, torch.nn.Module], torch.fx.Node], bool]:
            def is_node_supported(
                _submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
            ) -> bool:
                return node in region_nodes

            return is_node_supported

        all_partitions: list[dict[torch.fx.Node, int | None]] = []
        for region_nodes in regions.values():
            support = create_op_support(_is_in_region(region_nodes))
            partitioner = CapabilityBasedPartitioner(
                gm, support, allows_single_node_partition=True
            )
            for partition in partitioner.propose_partitions():
                all_partitions.append(partition.nodes)

        return fuse_by_partitions(
            gm,
            all_partitions,
            prefix="__marked_inductor_submod",
            always_return_tuple=True,
        )

    @staticmethod
    def recursively_scoop_regions(
        gm: torch.fx.GraphModule, _processed: set[int] | None = None
    ) -> torch.fx.GraphModule:
        if _processed is None:
            _processed = set()
        for node in gm.graph.find_nodes(op="get_attr"):
            if _needs_inductor_compile(node):
                # If the get_attr itself is marked for compile, the outer graph will
                # take care of it. If we don't do that, we end up with nested
                # regional inductor compiles that do not work well.
                continue
            submod = getattr(gm, node.target)
            # Track by id: multiple get_attr nodes may reference the same GraphModule
            if (
                isinstance(submod, torch.fx.GraphModule)
                and id(submod) not in _processed
            ):
                _processed.add(id(submod))
                _RegionScooper.recursively_scoop_regions(submod, _processed)

        return _RegionScooper.scoop_regions(gm)

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        with torch.fx.traceback.preserve_node_meta(enable=False):
            return _RegionScooper.recursively_scoop_regions(gm)


class _RegionCompiler:
    """
    Compiles the scooped out regions.
    """

    @staticmethod
    def compile_region(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        from torch.fx.graph import _BoxedCodeGen

        gm = _compile_submod(gm, "__marked_inductor_submod")
        gm.graph.set_codegen(_BoxedCodeGen())
        gm.recompile()
        return gm

    @staticmethod
    def recursively_compile_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        # Find if the graph module has a scooped out region
        found_region = False
        for node in gm.graph.find_nodes(op="call_module"):
            submod = getattr(gm, node.target)
            if isinstance(submod, torch.fx.GraphModule):
                if node.target.startswith("__marked_inductor_submod"):
                    found_region = True

        # Recurse through the subgraphs
        for node in gm.graph.find_nodes(op="get_attr"):
            submod = getattr(gm, node.target)
            if isinstance(submod, torch.fx.GraphModule):
                _RegionCompiler.recursively_compile_regions(submod)

        if found_region:
            return _RegionCompiler.compile_region(gm)
        return gm

    def __call__(self, gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
        with torch.fx.traceback.preserve_node_meta(enable=False):
            return _RegionCompiler.recursively_compile_regions(gm)


def _create_inductor_marked_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    with torch.fx.traceback.preserve_node_meta(enable=False):
        return _RegionScooper()(gm)


def _compile_inductor_marked_regions(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    with torch.fx.traceback.preserve_node_meta(enable=False):
        return _RegionCompiler()(gm)


@compatibility(is_backward_compatible=False)
def regional_inductor(
    gm: torch.fx.GraphModule, *example_args: object
) -> torch.fx.GraphModule:
    """
    Scoops out inductor marked regions and compiles them with inductor.

    Inductor options should be provided via the annotation API::

        with fx_traceback.annotate(
            {
                "compile_with_inductor": {
                    "inductor_configs": {
                        "max_autotune": True,
                        "triton.cudagraphs": False,
                    }
                }
            }
        ):
            ...
    """

    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues

    with torch.fx.traceback.preserve_node_meta(enable=False):
        gm = _create_inductor_marked_regions(gm)
        gm = _compile_inductor_marked_regions(gm)
        if torch._functorch.config.force_autograd_cache:
            from torch._inductor.output_code import RegionalOutputCode

            return RegionalOutputCode(gm)  # type: ignore[return-value]
        return gm
<|endoftext|>import copy
import logging
from collections import defaultdict

import torch
from torch._inductor.standalone_compile import AOTCompiledArtifact
from torch.compiler._cache import CacheArtifactManager
from torch.fx._compatibility import compatibility
from torch.fx.passes.regional_inductor import (
    _disable_remat_for_regional_subcompile,
    _dummy_wrapper,
)


logger = logging.getLogger(__name__)

__all__ = ["regional_inductor_invoke_subgraph"]


def _compile_submod(
    gm: torch.fx.GraphModule, subgraph: str, subgraph_users: list[torch.fx.Node]
) -> torch.fx.GraphModule:
    """
    Compiles subgraph submodule in gm. subgraph is used by subgraph_users.
    subgraph_users must all be  torch.ops.higher_order.invoke_subgraph HOP.
    """

    submod = getattr(gm, subgraph)

    compile_config = None
    fake_inputs = []

    # We use the first user for compile configs and inputs
    sub_node = subgraph_users[0]
    if not _needs_inductor_compile(sub_node):
        raise AssertionError("sub_node does not need inductor compile")
    compile_config = sub_node.meta["custom"]["nested_region_config"]
    if sub_node.meta.get("partitioner_tag") == "is_forward":
        compile_fn = compile_config.fw_compiler
    else:
        compile_fn = compile_config.bw_compiler

    for inp_node in sub_node.all_input_nodes[
        1:
    ]:  # exclude the graph module input to torch.ops.higher_order.invoke_subgraph
        if hasattr(inp_node, "meta") and "val" in inp_node.meta:
            fake_inputs.append(inp_node.meta["val"])
        else:
            raise RuntimeError(
                f"Partition is bad because non fake tensor value is seen {inp_node}"
            )

    # Log the options being used
    logger.info(
        "Compiling submodule %s with inductor options: %s",
        subgraph,
        compile_config,
    )

    def get_compiled_fn() -> AOTCompiledArtifact:
        context = torch._guards.TracingContext.get()
        if context.fake_mode is None:
            raise AssertionError("context.fake_mode is None")

        context = torch._guards.TracingContext(context.fake_mode)

        with (
            torch._guards.tracing(context),
            CacheArtifactManager.with_fresh_cache(),
            torch._functorch.config.patch("bundled_autograd_cache", True),
            _disable_remat_for_regional_subcompile(),
            torch.fx.traceback._set_regional_inductor_subgraph_name(subgraph),
        ):
            # compile_fx can mutate gm
            gm = copy.deepcopy(submod)

            compiled_fn = compile_fn(gm, fake_inputs)
            return compiled_fn

    compiled_fn = get_compiled_fn()
    if not isinstance(compiled_fn, AOTCompiledArtifact):
        raise AssertionError(f"Expected AOTCompiledArtifact, got {type(compiled_fn)}")

    # _dummy_wrapper is to make call_function happy
    compiled_submod = _dummy_wrapper(compiled_fn)
    for node in subgraph_users:
        with gm.graph.inserting_after(node):
            new_node = gm.graph.call_function(
                # exclude graph nodes input args
                compiled_submod,
                args=node.args[2:],
                kwargs=node.kwargs,
            )
            new_node.meta = node.meta
            node.replace_all_uses_with(new_node)
            gm.graph.erase_node(node)

    gm.recompile()
    return gm


def _needs_inductor_compile(node: torch.fx.Node) -> bool:
    # TODO: maybe we could change to check
    # node.meta.get("partitioner_tag") != "is_forward"
    # if the tag is relibable
    return bool(
        (
            node.op not in ("placeholder", "output")
            and hasattr(node, "meta")
            and node.meta.get("custom", None)
            and node.meta["custom"].get("nested_region_config", None)
            and node.meta["custom"]["nested_region_config"].fw_compiler
            and node.meta.get("partitioner_tag") != "is_backward"
        )
        or (
            node.op not in ("placeholder", "output")
            and hasattr(node, "meta")
            and node.meta.get("custom", None)
            and node.meta["custom"].get("nested_region_config", None)
            and node.meta["custom"]["nested_region_config"].bw_compiler
            and node.meta.get("partitioner_tag") == "is_backward"
        )
    )


def _compile_invoke_subgraph_nodes_with_inductor(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    map_subgraph_to_nodes = defaultdict(list)
    subgraphs: set[str] = set()

    for node in gm.graph.find_nodes(
        op="call_function", target=torch.ops.higher_order.invoke_subgraph
    ):
        if not _needs_inductor_compile(node):
            continue
        if node.args[0].op != "get_attr":
            raise AssertionError(f"Expected get_attr, got {node.args[0].op}")
        subgraph_name = node.args[0].target
        if not isinstance(subgraph_name, str):
            raise AssertionError(f"Expected str, got {type(subgraph_name)}")
        subgraphs.add(subgraph_name)
        map_subgraph_to_nodes[subgraph_name].append(node)

    for subgraph in subgraphs:
        gm = _compile_submod(gm, subgraph, map_subgraph_to_nodes[subgraph])

    return gm


def _recursive_compile_invoke_subgraph_nodes(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    for node in gm.graph.find_nodes(op="get_attr"):
        if _needs_inductor_compile(node):
            # If the get_attr itself is marked for compile, the outer graph will
            # take care of it. If we don't do that, we end up with nested
            # regional inductor compiles that do not work well.
            continue
        submod = getattr(gm, node.target)
        if isinstance(submod, torch.fx.GraphModule):
            _recursive_compile_invoke_subgraph_nodes(submod)

    return _compile_invoke_subgraph_nodes_with_inductor(gm)


@compatibility(is_backward_compatible=False)
def regional_inductor_invoke_subgraph(
    gm: torch.fx.GraphModule, *example_args: object
) -> torch.fx.GraphModule:
    """
    Compile invoke_subgraph nodes if they have custom compiler specified
    in node.meta["nested_region_config"].bw_compiler or fw_compiler
    """
    # fuser utils create new nodes using create_proxy which retains the seq_nr
    # metadata and cause issues
    with torch.fx.traceback.preserve_node_meta(enable=False):
        compiled_gm = _recursive_compile_invoke_subgraph_nodes(gm)
        # TODO: might not need this boxed_nop after we switch to _RegionCompiler
        # pyrefly: ignore [bad-return]
        return torch._dynamo.backends.debugging.boxed_nop(
            compiled_gm, example_inputs=[]
        )
<|endoftext|>import abc
import typing as t

import torch
import torch.fx
from torch.fx._compatibility import compatibility

from .shape_prop import TensorMetadata
from .tools_common import CALLABLE_NODE_OPS, get_node_target


__all__ = [
    "OperatorSupportBase",
    "OperatorSupport",
    "create_op_support",
    "chain",
    "OpSupports",
    "any_chain",
]

# fx.Node.target typename, as returned by `get_node_target()`
TargetTypeName = str

# Arguments' dtypes for a given node, see `OperatorSupport`
SupportedArgumentDTypes = (
    tuple[
        t.Sequence[t.Sequence[torch.dtype]],
        dict[str, t.Sequence[torch.dtype]],
    ]
    | None
)

SupportDict = t.Mapping[TargetTypeName, SupportedArgumentDTypes]


@compatibility(is_backward_compatible=False)
class OperatorSupportBase(abc.ABC):
    """Interface for determining if a fx.Node is supported by a backend"""

    @abc.abstractmethod
    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        raise NotImplementedError


@compatibility(is_backward_compatible=False)
class OperatorSupport(OperatorSupportBase):
    """
    `_support_dict` maps node.target typename to supported inputs dtypes.

    node.target typename is retrieved using helper function `get_node_target()`

    If supported inputs dtypes is None, it means any dtype is supported, else
    we should see a tuple like (([dtypes], ...), {"name":[dtypes], ...}).

    The first tuple ([dtypes], ...) indicates what dtypes are supported for
    inputs in node.args and the second dict {"name": [dtypes], ...} indicates
    what dtypes are supported for inputs in node.kwargs.

    For inputs in args, if we don't want to check it, we can put None there,
    e.g. (None, [torch.float]) indicates that we don't care about the type of
    the first input in args. And for inputs in kwargs, if not listed, will not
    be checked.
    """

    _support_dict: SupportDict

    def __init__(self, support_dict: SupportDict | None = None) -> None:
        self._support_dict = support_dict or {}

    def is_node_supported(
        self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        """
        Args:
            `submodules`: mapping from module name to the module. This can be
                          retrieved by calling model.named_modules().

            `node`: a Fx node that we want to determine whether it's supported.

        Returns:
            `is_supported`: whether the arg `node` is supported.
        """
        if node.op not in CALLABLE_NODE_OPS:
            return True

        target = get_node_target(submodules, node)

        # Target not found in _support_dict meaning that we don't support this op at all
        if target not in self._support_dict:
            return False

        # The rule for target is None meaning that we accept any dtype
        if self._support_dict[target] is None:
            return True

        args_dtypes, kwargs_dtypes = self._support_dict[target]  # type: ignore[misc]

        # Check args dtypes
        for i, dtypes in enumerate(args_dtypes):
            if len(node.args) <= i:
                break

            # None indicates we don't care about the dtype of args[i]
            if dtypes is None:
                continue

            # If arg is not a node then we don't check it
            if not isinstance(node.args[i], torch.fx.Node):
                continue

            arg_dtype = _get_arg_dtype(node.args[i])  # type: ignore[arg-type]
            if arg_dtype not in dtypes:
                return False

        # Check kwargs dtypes
        for k, dtypes in kwargs_dtypes.items():
            if k not in node.kwargs:
                continue

            # If arg is not a node then we don't check it
            if not isinstance(node.kwargs[k], torch.fx.Node):
                continue

            kwarg_dtype = _get_arg_dtype(node.kwargs[k])  # type: ignore[arg-type]
            if kwarg_dtype not in dtypes:
                return False

        return True


# ======================================================================
# Functional interfaces and utils for defining basic operator support logic
# and composing them into more complex ones
# ======================================================================

IsNodeSupported = t.Callable[[t.Mapping[str, torch.nn.Module], torch.fx.Node], bool]


@compatibility(is_backward_compatible=False)
def create_op_support(is_node_supported: IsNodeSupported) -> OperatorSupportBase:
    """Wraps a `IsNodeSupported` function into an `OperatorSupportBase` instance

    `IsNodeSupported` has the same call signature as
    `OperatorSupportBase.is_node_supported`
    """

    class FunctionalOperatorSupport(OperatorSupportBase):
        def is_node_supported(
            self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
        ) -> bool:
            return is_node_supported(submodules, node)

    return FunctionalOperatorSupport()


@compatibility(is_backward_compatible=False)
def chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`
    instance by evaluating each input `OperatorSupportBase` instance, and returns False if
    any of it reports False.
    """

    def _chain(submods: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        return all(x.is_node_supported(submods, node) for x in op_support)

    return create_op_support(_chain)


@compatibility(is_backward_compatible=False)
def any_chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`
    instance by evaluating each input `OperatorSupportBase` instance, and returns True if
    any of it reports True.
    """

    def _any_chain(
        submods: t.Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        return any(x.is_node_supported(submods, node) for x in op_support)

    return create_op_support(_any_chain)


@compatibility(is_backward_compatible=False)
class OpSupports:
    """A set of atomic `OperatorSupportBase` instances that can be combined together
    to form more complex operator support logic.
    """

    @classmethod
    def decline_if_input_dtype(cls, dtype: torch.dtype) -> OperatorSupportBase:
        """Report a node as non-supported, if any of its arguments is of dtype"""

        def _decline_if_input_dtype(
            submodules: t.Mapping[str, torch.nn.Module],
            node: torch.fx.Node,
        ) -> bool:
            for arg in node.all_input_nodes:
                arg_dtype = _get_arg_dtype(arg)
                if arg_dtype == dtype:
                    return False
            return True

        return create_op_support(_decline_if_input_dtype)

    @classmethod
    def decline_if_node_in_names(cls, disallow_set: set[str]) -> OperatorSupportBase:
        """
        If a node has a name that is in the disallow set, reported it as non-supported.
        """

        def _decline_if_node_in_names(
            submodules: t.Mapping[str, torch.nn.Module],
            node: torch.fx.Node,
        ) -> bool:
            return node.name not in disallow_set

        return create_op_support(_decline_if_node_in_names)


def _get_arg_dtype(arg: torch.fx.Node) -> torch.dtype:
    if not isinstance(arg, torch.fx.Node):
        raise AssertionError(f"Expected torch.fx.Node, got {type(arg)}")
    tensor_meta = arg.meta.get("tensor_meta")  # type: ignore[union-attr]
    dtype = (
        tensor_meta.dtype
        if isinstance(tensor_meta, TensorMetadata)
        else arg.meta["type"]
    )
    return dtype
<|endoftext|>from __future__ import annotations

import logging
import operator
from typing import Any, TYPE_CHECKING

from sympy import Integer, Number, Symbol
from sympy.logic.boolalg import BooleanAtom

import torch
import torch.fx as fx
from torch._dynamo.exc import TensorifyScalarRestartAnalysis
from torch._dynamo.symbolic_convert import TensorifyState
from torch._dynamo.utils import _is_tensorify_enabled, get_metrics_context
from torch._prims_common import get_computation_dtype
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.fx._utils import lazy_format_graph_code
from torch.fx.experimental.symbolic_shapes import (
    guard_scalar,
    has_free_symbols,
    ShapeEnv,
)

# TODO: refactor
from torch.fx.passes.runtime_assert import _get_sym_val
from torch.fx.proxy import MetaProxy
from torch.utils._sympy.interp import _run_sympy_handler, sympy_interp
from torch.utils._sympy.reference import TensorReferenceAnalysis
from torch.utils._sympy.symbol import symbol_is_type, SymT


if TYPE_CHECKING:
    from torch._subclasses import fake_tensor
    from torch.fx.graph_module import GraphModule


__all__: list[str] = []

log = logging.getLogger(__name__)
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code_verbose")

# The general shape of this transformation is to look for Tensor operations
# that take a backed SymFloat as an argument, and then redo them as tensor
# compute (with ints and tensors as inputs). For example, add(Tensor, Scalar)
# can be translated into add(Tensor, Tensor). Because Dynamo has already
# arranged for floats to be Tensor inputs to the graph, for typical float
# compute you can entirely translate the Python float operations into Tensor
# operations with only Tensor inputs.
#
# This pass is also responsible for doing CSE on the fly as we do this, since
# you don't want to keep recomputing the same quantity over and over again if
# it's used multiple times.
#
# This pass runs on the JOINT graph produced by AOT Autograd, prior to partitioning.
# The primary goal of this pass is to eliminate floats by replacing TensorScalar
# operations with TensorTensor operations and then Dead Code Elimination (DCE) of
# the item calls, which effectively removes the floats.
#
# This needs to happen before partitioning because it influences partitioning decisions,
# specifically by ensuring that we don't need to save floats across partitions.
# Additionally, there is a separate pass that changes which device computations
# occur on. That pass must be run after this one, but still before partitioning.
#
# HISTORY NOTE: Originally, I wanted to formulate this pass as pushing item()
# calls down, transforming float compute into int compute as we went. If you
# manage to eliminate all float compute, this ends up being equivalent, but
# there is a critical difference when some floats cannot be eliminated: when
# we call item() on them, what should it's SymFloat be? Ideally, it would
# be the same backed SymFloat we had before. But without symbolic expression
# propagation on tensor quantities, repropagating would instead give you an
# unbacked SymFloat. Maybe it is a good idea to implement symbolic propagation
# on 0d scalar tensors, but I decided to go for something simpler to start.
#
# The boring stuff:
#
# * What operators can I Tensor-ify? (Anything with a Scalar argument)
# * How do I Tensor-ify a SymFloat sympy expression (Sympy -> Op Handler -> Tensor)
#
# TODO: make sure this runs before CPU->CUDA pass for cudagraph friendliness


SUPPORTED_OPS = {
    torch.ops.aten.mul.Tensor: torch.ops.aten.mul.Tensor,
    torch.ops.aten.add.Tensor: torch.ops.aten.add.Tensor,
    torch.ops.aten.sub.Tensor: torch.ops.aten.sub.Tensor,
    torch.ops.aten.div.Tensor: torch.ops.aten.div.Tensor,
    torch.ops.aten.pow.Tensor_Tensor: torch.ops.aten.pow.Tensor_Tensor,
    operator.mul: torch.ops.aten.mul.Tensor,
    operator.add: torch.ops.aten.add.Tensor,
    operator.sub: torch.ops.aten.sub.Tensor,
    operator.truediv: torch.ops.aten.div.Tensor,
    operator.pow: torch.ops.aten.pow.Tensor_Tensor,
    torch.ops.aten.gt.Scalar: torch.ops.aten.gt.Tensor,
    torch.ops.aten.lt.Scalar: torch.ops.aten.lt.Tensor,
    torch.ops.aten.ge.Scalar: torch.ops.aten.ge.Tensor,
    torch.ops.aten.le.Scalar: torch.ops.aten.le.Tensor,
    torch.ops.aten.eq.Scalar: torch.ops.aten.eq.Tensor,
    torch.ops.aten.ne.Scalar: torch.ops.aten.ne.Tensor,
    operator.gt: torch.ops.aten.gt.Tensor,
    operator.lt: torch.ops.aten.lt.Tensor,
    operator.ge: torch.ops.aten.ge.Tensor,
    operator.le: torch.ops.aten.le.Tensor,
    operator.eq: torch.ops.aten.eq.Tensor,
    operator.ne: torch.ops.aten.ne.Tensor,
}

SUPPORTED_METHOD_OPS = {
    "mul_": torch.ops.aten.mul_.Tensor,
    "add_": torch.ops.aten.add_.Tensor,
    "sub_": torch.ops.aten.sub_.Tensor,
    "div_": torch.ops.aten.div_.Tensor,
}


@torch.fx._compatibility.compatibility(is_backward_compatible=False)
def tensorify_python_scalars(
    gm: GraphModule, shape_env: ShapeEnv, fake_mode: fake_tensor.FakeTensorMode
) -> None:
    """
    Converts Python scalar operations into Tensor operations within the graph. This pass looks for
    Tensor operations that involve SymFloat arguments and transforms them into equivalent operations
    that use only Tensor inputs.

    Args:
        gm: The FX graph module representing the computation graph.
        shape_env: The shape environment responsible for symbolic shape tracking and propagation
        during graph transformations.

    Returns:
        None
    """

    if not _is_tensorify_enabled():
        return None

    # This pass uses MetaProxy which relies on __torch_function__.
    # DisableTorchFunctionSubclass may be active here (see #177088),
    # so re-enable dispatch for MetaProxy ops.
    with torch._C._EnableTorchFunction():
        return _tensorify_impl(gm, shape_env, fake_mode)


def _tensorify_impl(
    gm: GraphModule,
    shape_env: ShapeEnv,
    fake_mode: fake_tensor.FakeTensorMode,
) -> None:
    """Helper fn in tensorify_python_scalars so the caller can wrap
    with _EnableTorchFunction (#180906).
    """
    import sympy

    graph = gm.graph
    tracer = fx.proxy.GraphAppendingTracer(graph)
    expr_to_sym_proxy: dict[sympy.Expr, MetaProxy] = {}
    expr_to_tensor_proxy: dict[sympy.Expr, MetaProxy] = {}
    tensorified_symbols: set[sympy.Symbol] = set()
    should_restart = False

    first_non_placeholder = None
    placeholders = set()
    for node in graph.nodes:
        if node.op != "placeholder":
            first_non_placeholder = node
            break
        else:
            placeholders.add(node)

    Analysis = TensorReferenceAnalysis

    def _sympy_interp(expr: sympy.Expr) -> MetaProxy:
        # sympy_interp() with hash consing, and special handling for
        # generating constants correctly

        # hash cons
        if isinstance(expr, Symbol) and expr not in expr_to_tensor_proxy:
            # This is guaranteed to be populated by invariant established by
            # insert_deferred_runtime_asserts
            expr_to_tensor_proxy[expr] = torch.ops.aten.scalar_tensor.default(
                expr_to_sym_proxy[expr]
            )

        # cache constants, why not
        if isinstance(expr, (Integer, Number, BooleanAtom)):
            dtype = None
            c: bool | int | float
            if isinstance(expr, BooleanAtom):
                dtype = torch.bool
                c = bool(expr)
            elif isinstance(expr, sympy.Integer):
                dtype = torch.int64
                c = int(expr)
            elif isinstance(expr, sympy.Number):
                dtype = torch.float64
                c = float(expr)

            node = graph.call_function(
                torch.ops.aten.scalar_tensor.default,
                # pyrefly: ignore [unbound-name]
                (c,),
                {"dtype": dtype},
            )
            with fake_mode:
                # pyrefly: ignore [unbound-name]
                node.meta["val"] = torch.ops.aten.scalar_tensor.default(c, dtype=dtype)
            expr_to_tensor_proxy[expr] = MetaProxy(
                node,
                tracer=tracer,
                fake_mode=fake_mode,
            )

        if expr in expr_to_tensor_proxy:
            return expr_to_tensor_proxy[expr]

        # don't cache
        if isinstance(expr, Symbol):
            return sympy_interp(Analysis, expr_to_tensor_proxy, expr)  # type: ignore[arg-type]

        # hash cons on arguments, run expr handler
        expr_to_tensor_proxy[expr] = _run_sympy_handler(
            Analysis,
            [_sympy_interp(arg) for arg in expr.args],  # type: ignore[arg-type]
            expr,
        )

        return expr_to_tensor_proxy[expr]

    failed_tensorify_ops: set[str] = set()
    nodes = list(graph.nodes)
    for i, node in enumerate(nodes[:-1]):
        with graph.inserting_before(
            nodes[i + 1] if node not in placeholders else first_non_placeholder
        ):
            # Look for tensor.item() calls on placeholders
            if (
                node is not None
                and node.op == "call_function"
                and node.target is torch.ops.aten._local_scalar_dense.default
            ):
                source_tensor = node.args[0].meta["val"]
                dtype = source_tensor.dtype

                if not isinstance(node.args[0], fx.Node):
                    raise AssertionError(f"Expected fx.Node, got {node.args[0]}")

                s = node.meta["val"].node.expr

                expr_to_sym_proxy[s] = MetaProxy(
                    node, tracer=tracer, fake_mode=fake_mode
                )

                # only tensorify if the dtype is floating point
                if not dtype.is_floating_point:
                    continue

                expr_to_tensor_proxy[s] = MetaProxy(
                    node.args[0], tracer=tracer, fake_mode=fake_mode
                )
                if len(source_tensor.shape) != 0:
                    # .item() always produces a scalar value, even when it is
                    # called on a size-1 tensor with rank > 0. Preserve that 0-d
                    # semantics before tensorifying the scalar expression so
                    # later tensor math and autograd tangents do not keep an
                    # accidental length-1 dimension.
                    expr_to_tensor_proxy[s] = torch.ops.aten.reshape.default(
                        expr_to_tensor_proxy[s], []
                    )
                # Upcast the float tensor to torch.float64 to avoid precision problem
                expr_to_tensor_proxy[s] = torch.ops.prims.convert_element_type.default(
                    expr_to_tensor_proxy[s], torch.float64
                )

            # pyrefly: ignore [bad-argument-type]
            elif (sym_expr := _get_sym_val(node)) is not None:
                if sym_expr not in expr_to_sym_proxy and not isinstance(
                    sym_expr, (sympy.Number, sympy.logic.boolalg.BooleanAtom)
                ):
                    expr_to_sym_proxy[sym_expr] = MetaProxy(
                        # pyrefly: ignore [bad-argument-type]
                        node,
                        tracer=tracer,
                        fake_mode=fake_mode,
                    )

            # Specialize all dimensions that contain symfloats. Here's
            # an example test that requires this:
            # PYTORCH_OPINFO_SAMPLE_INPUT_INDEX=4 python test/inductor/test_torchinductor_opinfo.py TestInductorOpInfoCUDA.test_comprehensive_nn_functional_interpolate_bicubic_cuda_float32

            val = node.meta.get("val")
            if is_fake_tensor(val):
                for dim in val.shape:
                    if isinstance(dim, torch.SymInt):
                        for s in dim.node.expr.free_symbols:
                            name = str(s)
                            if symbol_is_type(
                                s, SymT.FLOAT
                            ) and not TensorifyState.should_specialize(name):
                                # In principle, we could support float input that
                                # is used to do size compute. The problem is that
                                # we don't actually want to tensorify the compute
                                # in this case, which means we need codegen support for
                                # all symfloats.
                                TensorifyState.specialize(name)
                                should_restart = True

            # Look for functions to convert

            replacement_op = None
            is_inplace_method = False
            if node.op == "call_function":
                replacement_op = SUPPORTED_OPS.get(node.target)
            elif node.op == "call_method":
                replacement_op = SUPPORTED_METHOD_OPS.get(node.target)
                is_inplace_method = replacement_op is not None

            if replacement_op is not None:
                # Pure SymFloat/SymBool expression nodes are scalar intermediates,
                # not tensor-valued ops. Let later tensor uses consume their
                # symbolic expression instead of treating them as a tensorify
                # failure here.
                if not hasattr(node.meta.get("val"), "dtype"):
                    continue

                args: list[Any] = []
                transform = False

                compute_dtype = get_computation_dtype(node.meta["val"].dtype)

                for a in node.args:
                    if (
                        isinstance(a, fx.Node)
                        and "val" in a.meta
                        and isinstance(zf := a.meta["val"], torch.SymFloat)
                    ):
                        transform = True
                        try:
                            proxy = _sympy_interp(zf.node.expr)
                        except NotImplementedError:
                            transform = False
                            break

                        # Track the original backed float symbols that flowed into
                        # this tensorified expression so the later specialization
                        # sweep does not incorrectly restart analysis for them.
                        tensorified_symbols.update(
                            s
                            # pyrefly: ignore [missing-attribute]
                            for s in a.meta["val"].node._expr.free_symbols
                            if symbol_is_type(s, SymT.FLOAT)
                        )

                        # The upcasting is irrelevant when the compute dtype is bool. This happens
                        # in cases where we are tensorifying a comparison operator such as
                        # torch.ops.aten.gt.Tensor
                        if (
                            compute_dtype != torch.bool
                            and proxy.node.meta["val"].dtype != compute_dtype
                        ):
                            proxy = torch.ops.prims.convert_element_type.default(
                                proxy, compute_dtype
                            )

                        args.append(proxy)
                    elif isinstance(a, fx.Node):
                        args.append(MetaProxy(a, tracer=tracer, fake_mode=fake_mode))
                    else:
                        args.append(a)

                if transform:
                    # Preserve keyword-only args (e.g. add/sub `alpha`) which
                    # live in node.kwargs; the loop above only rebuilds
                    # positional args, so without this they are dropped.
