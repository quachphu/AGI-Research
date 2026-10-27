                         / / /
    (input_2)?----------/ / /
                         / /
    weight -> weight_obs  /
                         /
    bias ----------------

    graph C (derived from B):
    =========================

    input_node_c --> node_a_copy
                     / / /
    (input_node_c_2)? / /
                     / /
    weight_copy ----/ /
                     /
    bias_copy ------/
    """
    if isinstance(input_node_c, Node):
        graph_c = input_node_c.graph
    else:
        if not isinstance(input_node_c, list):
            raise AssertionError(f"Expected list, got {type(input_node_c)}")
        graph_c = input_node_c[0].graph

    norm_args_kwargs = node_a.normalized_arguments(
        gm_a, normalize_to_only_use_kwargs=True
    )
    if norm_args_kwargs is not None:
        norm_args, norm_kwargs = norm_args_kwargs
    else:
        norm_args, norm_kwargs = node_a.args, node_a.kwargs

    new_args = []
    new_kwargs = {}

    def _copy_arg(arg):
        # copy the other inputs from the other graph
        if isinstance(arg, Node):
            arg = return_first_non_observer_node(arg, gm_a)
            arg = _copy_node_from_a_to_c(arg, gm_a, gm_b, graph_c)
            return arg
        elif isinstance(arg, (int, float, torch.dtype)):
            return arg
        elif isinstance(kwarg_val, (list, tuple)):
            for el in kwarg_val:
                if isinstance(el, Node):
                    raise AssertionError(
                        "handling of Node inside list is not implemented"
                    )
            return arg
        else:
            raise AssertionError(
                f"handling for kwarg of type {type(kwarg_val)} is not implemented"
            )

    cur_idx = 0

    while cur_idx < len(norm_args):
        if cur_idx == 0:
            new_arg = input_node_c
        elif cur_idx == 1 and input_node_c_2 is not None:
            new_arg = input_node_c_2
        else:
            new_arg = _copy_arg(norm_args[cur_idx])
        new_args.append(new_arg)
        cur_idx += 1

    for kwarg_name, kwarg_val in norm_kwargs.items():
        # stitch the inputs from base graph
        if cur_idx == 0:
            new_kwargs[kwarg_name] = input_node_c
        elif cur_idx == 1 and input_node_c_2 is not None:
            new_kwargs[kwarg_name] = input_node_c_2
        else:
            new_kwargs[kwarg_name] = _copy_arg(kwarg_val)
        cur_idx += 1

    new_args = tuple(new_args)  # type: ignore[assignment]

    node_a_shadows_c_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)

    if node_a.op == "call_module":
        # if target is a module, we point to the module from gm_b
        new_mod_copy_name = get_new_attr_name_with_prefix(node_name_prefix)(gm_b)
        # fetch the corresponding module from gm_a
        if not isinstance(node_a.target, str):
            raise AssertionError(f"Expected str, got {type(node_a.target)}")
        mod_a = getattr_from_fqn(gm_a, node_a.target)
        setattr(gm_b, new_mod_copy_name, mod_a)
        node_a_shadows_c = graph_c.create_node(
            node_a.op,
            new_mod_copy_name,
            new_args,  # type: ignore[arg-type]
            new_kwargs,  # type: ignore[arg-type]
            node_a_shadows_c_name,
        )
        return node_a_shadows_c
    else:
        if node_a.op not in ("call_function", "call_method"):
            raise AssertionError(f"Unexpected op: {node_a.op}")
        node_a_shadows_c = graph_c.create_node(
            node_a.op,
            node_a.target,
            new_args,  # type: ignore[arg-type]
            new_kwargs,  # type: ignore[arg-type]
            node_a_shadows_c_name,
        )
        return node_a_shadows_c


def create_a_shadows_b(
    name_a: str,
    gm_a: GraphModule,
    name_b: str,
    gm_b: GraphModule,
    matched_subgraph_pairs: dict[str, tuple[NSSubgraph, NSSubgraph]],
    logger_cls: Callable,
    should_log_inputs: bool,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]] | None = None,
) -> GraphModule:
    """
    Creates a new GraphModule consisting of the graph of C, with the meaningful
    nodes of A shadowing the corresponding nodes of B.  For example,

    Graph A:
    a0 -> op0_fp32 -> a1 -> op1_fp32 -> a2

    Graph B:
    b0 -> op0_int8 -> b1 -> op1_int8 -> b2

    matched_node_pairs: {'op0': (op0_fp32, op0_int8), 'op1': (op1_fp32, op1_int8)}

    Graph C (A shadows B):

        / dequant0 -> op0_fp32 -> logger_a_0  / dequant_1 -> op1_fp32 -> logger_a_1
       /                                     /
    b0 -------------> op0_int8 -> logger_b_0 --------------> op1_int8 -> logger_b_1

    In a nutshell, this function does the following for each node pair:
    * copies the necessary attributes and modules from gm_a to gm_b,
      keeping names unique
    * adds a dtype cast op (dequant, quant, etc)
    * adds a copy of node_a in gm_b's graph
    * adds loggers to the outputs of node_a and node_b
    """

    if node_type_to_io_type_map is None:
        node_type_to_io_type_map = get_node_type_to_io_type_map()

    # graph_c is the graph created from copying the nodes of graph_b and inserting
    # the shadows with the nodes copied from graph_a
    graph_c = Graph()
    env_c: dict[str, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env_c[node.name])

    start_node_b_to_matched_subgraph_a_and_name = {}
    end_node_b_to_matched_subgraph_a_and_name = {}
    for match_name, match in matched_subgraph_pairs.items():
        subgraph_a, subgraph_b = match
        ref_node_type_a = get_target_type_str(subgraph_a.base_op_node, gm_a)
        ref_node_type_b = get_target_type_str(subgraph_b.base_op_node, gm_b)
        start_node_b_to_matched_subgraph_a_and_name[subgraph_b.start_node] = (
            subgraph_a,
            match_name,
            ref_node_type_a,
            ref_node_type_b,
        )
        end_node_b_to_matched_subgraph_a_and_name[subgraph_b.end_node] = (
            subgraph_a,
            match_name,
            ref_node_type_a,
            ref_node_type_b,
        )

    for node_b in gm_b.graph.nodes:
        if node_b.op == "output":
            graph_c.output(map_arg(node_b.args[0], load_arg))
            continue

        # calculate the flags to determine what to do with this node
        node_b_is_start_node = node_b in start_node_b_to_matched_subgraph_a_and_name
        node_b_is_end_node = node_b in end_node_b_to_matched_subgraph_a_and_name

        if node_b_is_start_node or node_b_is_end_node:
            if node_b_is_start_node:
                (
                    subgraph_a,
                    ref_name,
                    ref_node_type_a,
                    ref_node_type_b,
                ) = start_node_b_to_matched_subgraph_a_and_name[node_b]
            else:
                if not node_b_is_end_node:
                    raise AssertionError("Expected node_b_is_end_node to be not false")
                (
                    subgraph_a,
                    ref_name,
                    ref_node_type_a,
                    ref_node_type_b,
                ) = end_node_b_to_matched_subgraph_a_and_name[node_b]

            all_op_types_support_shadowing = op_type_supports_shadowing(
                subgraph_a.start_node
            ) and op_type_supports_shadowing(node_b)
            if not all_op_types_support_shadowing:
                print(
                    f"skipping shadow loggers for node_b: {get_target_type_str(node_b, gm_b)}"
                    + f", start_node_a: {get_target_type_str(subgraph_a.start_node, gm_a)}"
                    + ", unsupported"
                )
                env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
                continue

            # For both start_node and end_node verify that we know how to do
            # the dtype cast. If we do not, skip.
            (
                node_input_type_a,
                node_output_type_a,
            ) = get_node_first_input_and_output_type(
                subgraph_a.start_node, gm_a, logger_cls, node_type_to_io_type_map
            )
            (
                node_input_type_b,
                node_output_type_b,
            ) = get_node_first_input_and_output_type(
                node_b, gm_b, logger_cls, node_type_to_io_type_map
            )
            node_io_types_known_a_and_b = (
                node_input_type_a != NodeInputOrOutputType.UNKNOWN
                and node_output_type_a != NodeInputOrOutputType.UNKNOWN
                and node_input_type_b != NodeInputOrOutputType.UNKNOWN
                and node_output_type_b != NodeInputOrOutputType.UNKNOWN
            )
            if not node_io_types_known_a_and_b:
                print(
                    f"skipping shadow loggers for node_b: {get_target_type_str(node_b, gm_b)}"
                    + f", start_node_a: {get_target_type_str(subgraph_a.start_node, gm_a)}"
                    + ", unknown dtype cast"
                )
                env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
                continue

            # If we are shadowing from fp32 to int8, we need to insert
            # quantize_per_tensor call with qparams from the previous node.
            # Only do this if we are able to infer these qparams from the graph.
            if (
                node_input_type_a == NodeInputOrOutputType.INT8
                and node_input_type_b == NodeInputOrOutputType.FP32
            ):
                node_a_input_qparams = get_node_input_qparams(
                    subgraph_a.start_node, gm_a, node_type_to_io_type_map
                )
                if not node_a_input_qparams:
                    print(
                        f"skipping shadow loggers for node_b: {get_target_type_str(node_b, gm_b)}"
                        + f", start_node_a: {get_target_type_str(subgraph_a.start_node, gm_a)}"
                        + ", unknown input qparams"
                    )
                    env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
                    continue

            num_non_param_args_node_a = get_number_of_non_param_args(
                subgraph_a.start_node, gm_a
            )
            if not _can_insert_copy_of_subgraph_a(
                subgraph_a, gm_a, num_non_param_args_node_a
            ):
                print(
                    f"skipping shadow loggers for node_b: {get_target_type_str(node_b, gm_b)}"
                    + f", start_node_a: {get_target_type_str(subgraph_a.start_node, gm_a)}"
                    + ", unhandled logic in subgraph copy"
                )
                env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
                continue

            fqn_base_a = _maybe_get_fqn(subgraph_a.base_op_node, gm_a)
            fqn_base_b = _maybe_get_fqn(subgraph_b.base_op_node, gm_b)  # type: ignore[possibly-undefined]

            if node_b_is_start_node:
                # if necessary, log the input of node_c
                if should_log_inputs:
                    prev_node_b = get_normalized_nth_input(node_b, gm_b, 0)
                    if isinstance(prev_node_b, Node):
                        prev_node_c = env_c[prev_node_b.name]
                        env_c[prev_node_c.name] = _insert_logger_after_node(
                            prev_node_c,
                            gm_b,
                            logger_cls,
                            "_ns_logger_b_inp_",
                            node_b.name,
                            name_b,
                            ref_name,
                            ref_node_type_b,
                            NSSingleResultValuesType.NODE_INPUT.value,
                            index_within_arg=0,
                            index_of_arg=0,
                            fqn=fqn_base_b,
                        )
                    elif isinstance(prev_node_b, list):
                        # first, save the prev_node instances, because they
                        # will be overwritten in the env after the first logger
                        # is added
                        prev_node_c_list = [env_c[arg.name] for arg in prev_node_b]

                        for arg_idx, prev_node_c in enumerate(prev_node_c_list):
                            env_c[prev_node_c.name] = _insert_logger_after_node(
                                prev_node_c,
                                gm_b,
                                logger_cls,
                                "_ns_logger_b_inp_",
                                node_b.name,
                                name_b,
                                ref_name,
                                ref_node_type_b,
                                NSSingleResultValuesType.NODE_INPUT.value,
                                index_within_arg=arg_idx,
                                index_of_arg=0,
                                fqn=fqn_base_b,
                            )
                    else:
                        # logging of inputs which are not lists is not supported yet
                        raise AssertionError(
                            f"type {type(prev_node_b)} is not handled yet"
                        )
                # subgraph so far:
                #
                # (prev_node_c)+ -> (logger_c_input)?

            # Note: this if statement is always True, spelling it out to clarify code
            # intent.
            if node_b_is_start_node or node_b_is_end_node:
                # ensure env_c is populated with base node
                env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)
                node_c = env_c[node_b.name]

                # after this point,
                #
                # node_a is the original node from graph_a, with parent module gm_a
                # node_b is the original node from graph_b, with parent module gm_b
                # node_c is the copy of node_b in graph_c
                #
                # subgraph so far:
                #
                # (prev_node_c)+ -> (logger_c_input)? -> node_start_c

            if node_b_is_start_node:
                # cast dtype from the dtype of node_c's input to the dtype of
                # node_a's input (dequant, etc)
                # prev_node_c = node_c.args[0]
                prev_node_c = get_normalized_nth_input(node_c, gm_b, 0)  # type: ignore[possibly-undefined]
                if should_log_inputs:
                    # skip the input logger when inserting a dtype cast
                    if isinstance(prev_node_c, Node):
                        # pyrefly: ignore [unbound-name]
                        prev_node_c = get_normalized_nth_input(node_c, gm_b, 0)
                    elif isinstance(prev_node_c, list):
                        prev_node_c = [
                            get_normalized_nth_input(arg, gm_b, 0)
                            for arg in prev_node_c
                        ]
                dtype_cast_node = _insert_dtype_cast_after_node(
                    subgraph_a.start_node,
                    # pyrefly: ignore [unbound-name]
                    node_c,
                    prev_node_c,
                    gm_a,
                    gm_b,
                    graph_c,
                    node_b.name + "_dtype_cast_",
                    logger_cls,
                    node_type_to_io_type_map,
                )
                # note: not inserting to env_c because all nodes which use the dtype
                #   casts are copied from graph_a
                #
                # subgraph so far:
                #
                #           (dtype_cast_node)+
                #                  /
                # (prev_node_c)+ -> (logger_c_input)? -> node_start_c

                # if input logging is enabled, log the input to the subgraph
                if should_log_inputs:
                    # TODO: explain this
                    ref_node_name = ""
                    if isinstance(dtype_cast_node, Node):
                        dtype_cast_node = _insert_logger_after_node(
                            dtype_cast_node,
                            gm_b,
                            logger_cls,
                            "_ns_logger_a_inp_",
                            ref_node_name,
                            name_a,
                            ref_name,
                            ref_node_type_a,
                            NSSingleResultValuesType.NODE_INPUT.value,
                            index_within_arg=0,
                            index_of_arg=0,
                            fqn=fqn_base_a,
                        )
                        input_logger: Node | list[Node] = dtype_cast_node
                    else:
                        if not isinstance(dtype_cast_node, list):
                            raise AssertionError(
                                f"Expected list, got {type(dtype_cast_node)}"
                            )
                        new_loggers = []
                        for dtype_cast_idx, dtype_cast_node_inner in enumerate(
                            dtype_cast_node
                        ):
                            dtype_cast_logger = _insert_logger_after_node(
                                dtype_cast_node_inner,
                                gm_b,
                                logger_cls,
                                "_ns_logger_a_inp_",
                                ref_node_name,
                                name_a,
                                ref_name,
                                ref_node_type_a,
                                NSSingleResultValuesType.NODE_INPUT.value,
                                index_within_arg=dtype_cast_idx,
                                index_of_arg=0,
                                fqn=fqn_base_a,
                            )
                            new_loggers.append(dtype_cast_logger)
                        dtype_cast_node = new_loggers
                        input_logger = dtype_cast_node
                    # subgraph so far:
                    #
                    #       (dtype_cast_node)+ -> (logger_a_input)?
                    #                  /
                    # prev_node_c -> (logger_c_input)? -> node_start_c

                # hook up the new mod_a copy to be in the graph, receiving the
                # same inputs as mod_b does, with dtype cast to match a
                # Some ops, such as LSTMs, have two non-param inputs. If we have
                # such an op, pass the second param as well. Note: dtype casting
                # for the second param is not implemented yet, it can be added
                # later if there is a use case.
                node_c_second_non_param_arg = None
                num_non_param_args_node_a = get_number_of_non_param_args(
                    subgraph_a.start_node, gm_a
                )
                if num_non_param_args_node_a == 2:
                    # node_c_second_non_param_arg = node_c.args[1]
                    node_c_second_non_param_arg = get_normalized_nth_input(
                        # pyrefly: ignore [unbound-name]
                        node_c,
                        gm_b,
                        1,
                    )
                node_a_shadows_c = _insert_copy_of_subgraph_a_after_input_node_c(
                    dtype_cast_node,
                    node_c_second_non_param_arg,
                    subgraph_a,
                    gm_a,
                    gm_b,
                    # pyrefly: ignore [unbound-name]
                    node_c.name + "_shadow_copy_",
                )
                env_c[node_a_shadows_c.name] = node_a_shadows_c
                # subgraph so far:
                #
                #       dtype_cast_node -> (logger_a_input)? -> subgraph_a_copy(args/kwargs not shown)
                #                  /
                # (prev_node_c)+ -> (logger_c_input)? -> node_start_c

                if should_log_inputs:
                    # When we created the input logger, we left the ref_node_name
                    # as an empty string, because the subgraph copy did not exist
                    # yet. Now that the subgraph copy exists, we modify this name
                    # to its true value.
                    # Note: the alternative to this is to create the input logger
                    # after creating the subgraph, which is slightly more
                    # complicated. This is the lesser of two evils.
                    # input_logger = env_c[dtype_cast_node.name]
                    # Find the first node in the subgraph
                    cur_node = node_a_shadows_c
                    while get_normalized_nth_input(cur_node, gm_b, 0) != input_logger:  # type: ignore[possibly-undefined]
                        cur_node = get_normalized_nth_input(cur_node, gm_b, 0)  # type: ignore[assignment]
                    # pyrefly: ignore [unbound-name]
                    if isinstance(input_logger, Node):
                        # pyrefly: ignore [unbound-name]
                        input_logger_mod = getattr(gm_b, input_logger.name)
                        input_logger_mod.ref_node_name = cur_node.name
                    else:
                        # pyrefly: ignore [unbound-name]
                        if not isinstance(input_logger, list):
                            raise AssertionError(
                                # pyrefly: ignore [unbound-name]
                                f"Expected list, got {type(input_logger)}"
                            )
                        # pyrefly: ignore [unbound-name]
                        for input_logger_inner in input_logger:
                            input_logger_mod = getattr(gm_b, input_logger_inner.name)
                            input_logger_mod.ref_node_name = cur_node.name

                # hook up a logger to the mod_a copy
                env_c[node_a_shadows_c.name] = _insert_logger_after_node(
                    env_c[node_a_shadows_c.name],
                    gm_b,
                    logger_cls,
                    "_ns_logger_a_",
                    node_a_shadows_c.name,
                    name_a,
                    ref_name,
                    ref_node_type_a,
                    NSSingleResultValuesType.NODE_OUTPUT.value,
                    index_within_arg=0,
                    index_of_arg=0,
                    fqn=fqn_base_a,
                )
                # subgraph so far:
                #
                #       dtype_cast_node -> (logger_a_input)? -> subgraph_a_copy -> logger_a
                #                  /
                # (prev_node_c)+ -> (logger_c_input)? -> node_start_c

            if node_b_is_end_node:
                # hook up a logger to the mod_b copy
                env_c[node_b.name] = _insert_logger_after_node(
                    env_c[node_b.name],
                    gm_b,
                    logger_cls,
                    "_ns_logger_b_",
                    node_b.name,
                    name_b,
                    ref_name,
                    ref_node_type_b,
                    NSSingleResultValuesType.NODE_OUTPUT.value,
                    index_within_arg=0,
                    index_of_arg=0,
                    fqn=fqn_base_b,
                )
                # subgraph so far:
                #
                #       dtype_cast_node -> (logger_a_input)? -> subgraph_a_copy -> logger_a
                #                  /
                # (prev_node_c+) -> (logger_c_input)? -> node_start_c -> ... -> node_end_c -> logger_c
                #
                # Note: node_start_c may be the same node as node_end_c, or they
                # may have nodes in between.

        else:
            env_c[node_b.name] = graph_c.node_copy(node_b, load_arg)

    gm_c = GraphModule(gm_b, graph_c)
    return gm_c
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import copy
from typing import Any, TYPE_CHECKING

import torch
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.qconfig_mapping import _QCONFIG_STYLE_ORDER


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.ao.quantization.qconfig import QConfigAny

__all__ = ["QConfigMultiMapping"]

_QCONFIG_STYLE_TO_METHOD: dict[str, str] = {
    "global_qconfig": "set_global",
    "object_type_qconfigs": "set_object_type",
    "module_name_regex_qconfigs": "set_module_name_regex",
    "module_name_qconfigs": "set_module_name",
    "module_name_object_type_order_qconfigs": "set_module_name_object_type_order",
}


def _remove_duplicates_and_none(qconfig_list: list[QConfigAny]) -> None:
    to_remove = []
    for index, cur_qconfig in enumerate(qconfig_list):
        if cur_qconfig is None:
            to_remove.append(index)
            break
        for checked_qconfig in qconfig_list[:index]:
            if torch.ao.quantization.qconfig_equals(cur_qconfig, checked_qconfig):
                to_remove.append(index)
                break
    for index in to_remove[::-1]:
        qconfig_list.pop(index)


class QConfigMultiMapping:
    """
    This class, used with the prepare_n_shadows_model API, stores a list of :class:`torch.ao.quantization.QConfigMapping`s
    so that multiple QConfigs can be specified for each QConfig matching style.

    The user can specify QConfigs using the following methods (in increasing match priority):

        ``set_global`` : sets the global (default) QConfigs

        ``set_object_type`` : sets the QConfigs for a given module type, function, or method name

        ``set_module_name_regex`` : sets the QConfigs for modules matching the given regex string

        ``set_module_name`` : sets the QConfigs for modules matching the given module name

        ``set_module_name_object_type_order`` : sets the QConfigs for modules matching a combination
        of the given module name, object type, and the index at which the module appears

    Note: Usage of set methods is the same as in QConfigMapping except with a passed in list of QConfigs rather than a
    single QConfig.

    Example usage::

        qconfig_mapping = QConfigMultiMapping()
            .set_global([qconfig1, qconfig2])
            .set_object_type(torch.nn.Linear, [qconfig2, qconfig3])
            .set_object_type(torch.nn.ReLU, [qconfig1])
            .set_module_name_regex("foo.*bar.*conv[0-9]+", [qconfig2])
            .set_module_name_regex("foo.*", [qconfig1, qconfig2, qconfig3])
            .set_module_name("module1", [None])
            .set_module_name("module2", [qconfig2])
            .set_module_name_object_type_order("foo.bar", torch.nn.functional.linear, 0, [qconfig3])

    """

    def __init__(self) -> None:
        # initialize this with 1 QConfigMapping to avoid corner cases
        self.qconfig_mappings_list: list[QConfigMapping] = [QConfigMapping()]

    def _handle_list_size_mismatch(
        self, qconfig_list: list[QConfigAny], style: str
    ) -> None:
        # this method handles cases where the size of qconfig_list does not match
        # the size of qconfig_mappings_list.
        # Issue: Consider a user inserting global_qconfig A and B first, then inserting
        # qconfig C as an object_type_qconfig for conv ops. If we internally store
        # 1 QConfigMapping with A and C and another with just B, then the
        # second QConfigMapping will match B to conv ops (which is not wanted), since B is global.

        # we avoid this by maintaining the invariant that if any QConfigMapping
        # has a qconfig style+key with a qconfig in it, all QConfigMappings must
        # have either a qconfig or None for that same style+key. In the above
        # example, a None qconfig would prevent the unwanted match in the
        # second QConfigMapping

        if len(qconfig_list) > len(self.qconfig_mappings_list):
            # Case: we have more qconfigs (in qconfig_list) than QConfigMappings

            # Add new QConfigMappings (initialized so we maintain the `invariant`)

            new_qconfig_mapping = QConfigMapping()
            # searches other QConfigMappings for qconfig style+keys
            # that need to be inserted as `None` into the new QConfigMapping
            for qconfig_mapping in self.qconfig_mappings_list:
                # global_qconfig has None by default
                for check_style in _QCONFIG_STYLE_ORDER[1:]:
                    qconfigs_dict = getattr(qconfig_mapping, check_style)
                    target_qconfigs_dict = getattr(new_qconfig_mapping, check_style)
                    for key in qconfigs_dict:
                        target_qconfigs_dict[key] = None
                break

            # insert copies of this new QConfigMapping until all entries
            # in qconfig_list can fit among the QConfigMappings
            while len(qconfig_list) > len(self.qconfig_mappings_list):
                self.qconfig_mappings_list.append(copy.deepcopy(new_qconfig_mapping))
        else:
            # Case: we have fewer qconfigs in qconfig_list than QConfigMappings

            # pad qconfig_list with `None` until length is same
            while len(qconfig_list) < len(self.qconfig_mappings_list):
                qconfig_list.append(None)

    # this function applies the insertion method across each QConfigMapping
    def _insert_qconfig_list(
        self,
        style: str,
        args: list[str | int | Callable],
        qconfig_list: list[QConfigAny],
    ) -> None:
        # we remove duplicates and None to make the ordering of qconfigs
        # deterministic upon insertion.
        _remove_duplicates_and_none(qconfig_list)

        self._handle_list_size_mismatch(qconfig_list, style)
        method_name = _QCONFIG_STYLE_TO_METHOD[style]
        for qconfig_mapping, qconfig in zip(self.qconfig_mappings_list, qconfig_list):
            # uses QConfigMapping set method to insert qconfig
            set_method = getattr(qconfig_mapping, method_name)
            set_method(*args, qconfig)

    def set_global(self, global_qconfig_list: list[QConfigAny]) -> QConfigMultiMapping:
        """
        Set global QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_global()` for more info
        """
        self._insert_qconfig_list("global_qconfig", [], global_qconfig_list)
        return self

    def set_object_type(
        self, object_type: Callable | str, qconfig_list: list[QConfigAny]
    ) -> QConfigMultiMapping:
        """
        Set object type QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_object_type()` for more info
        """
        self._insert_qconfig_list("object_type_qconfigs", [object_type], qconfig_list)
        return self

    def set_module_name_regex(
        self, module_name_regex: str, qconfig_list: list[QConfigAny]
    ) -> QConfigMultiMapping:
        """
        Set module_name_regex QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name_regex()` for more info
        """
        self._insert_qconfig_list(
            "module_name_regex_qconfigs", [module_name_regex], qconfig_list
        )
        return self

    def set_module_name(
        self, module_name: str, qconfig_list: list[QConfigAny]
    ) -> QConfigMultiMapping:
        """
        Set module_name QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name()` for more info
        """
        self._insert_qconfig_list("module_name_qconfigs", [module_name], qconfig_list)
        return self

    def set_module_name_object_type_order(
        self,
        module_name: str,
        object_type: Callable,
        index: int,
        qconfig_list: list[QConfigAny],
    ) -> QConfigMultiMapping:
        """
        Set module_name QConfigs
        see :func:`~torch.ao.quantization.QConfigMapping.set_module_name_object_type_order()` for more info
        """
        self._insert_qconfig_list(
            "module_name_object_type_order_qconfigs",
            [module_name, object_type, index],
            qconfig_list,
        )
        return self

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ["
            + "".join(
                f"\n{qconfig_mapping.__repr__()},"
                for qconfig_mapping in self.qconfig_mappings_list
            )
            + "\n]"
        )

    @classmethod
    def from_list_qconfig_mapping(
        cls, qconfig_mapping_list: list[QConfigMapping]
    ) -> QConfigMultiMapping:
        """
        Creates a QConfigMultiMapping from a list of QConfigMappings
        """
        new_qconfig_multi_mapping = cls()

        new_qconfig_multi_mapping.qconfig_mappings_list = copy.deepcopy(
            qconfig_mapping_list
        )

        # we need to avoid the issue described in _handle_list_size_mismatch,
        # so we reinsert all the qconfigs using the QConfigMultiMapping
        # set methods

        # go through all qconfig styles
        # note: global can be ignored since it is None by default
        for style in _QCONFIG_STYLE_ORDER[1:]:
            # gather all key+qconfigs for current style
            # into qconfig_dict_list
            qconfig_dict_list: dict[Any, list[QConfigAny]] = {}
            for qconfig_mapping in qconfig_mapping_list:
                qconfig_dict = getattr(qconfig_mapping, style)
                for key, qconfig in qconfig_dict.items():
                    if key not in qconfig_dict_list:
                        qconfig_dict_list[key] = []
                    qconfig_dict_list[key].append(qconfig)

            # reinsert all gathered key+qconfigs
            set_method_name = _QCONFIG_STYLE_TO_METHOD[style]
            set_method = getattr(new_qconfig_multi_mapping, set_method_name)
            for key, qconfig_list in qconfig_dict_list.items():
                if isinstance(key, tuple):
                    set_method(*key, qconfig_list)
                else:
                    set_method(key, qconfig_list)

        return new_qconfig_multi_mapping
<|endoftext|># mypy: allow-untyped-defs
import collections
import enum
from typing import Any

import torch
from torch.ao.quantization import FakeQuantizeBase, ObserverBase
from torch.ao.quantization.utils import getattr_from_fqn
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node

from .mappings import get_base_name_to_sets_of_related_ops, get_unmatchable_types_map
from .ns_types import NSNodeTargetType, NSSubgraph
from .pattern_utils import (
    end_node_matches_reversed_fusion,
    get_reversed_fusions,
    get_type_a_related_to_b,
)


toq = torch.ops.quantized


def _get_output_nodes(g: Graph) -> list[Node]:
    return [n for n in g.nodes if n.op == "output"]


class _NSGraphMatchableSubgraphsIterator:
    """
    Iterates through the graph of gm, starting with the output nodes
    and continuing backwards.
    1. Returns matchable subgraphs, in order. A subgraph is defined by
       (start_node, end_node).
    2. Skips over non-matchable subgraphs
    """

    def __init__(
        self,
        gm: GraphModule,
        non_matchable_functions: set[NSNodeTargetType],
        non_matchable_modules: set[NSNodeTargetType],
        non_matchable_methods: set[NSNodeTargetType],
    ):
        self.gm: GraphModule = gm
        self.non_matchable_functions: set[NSNodeTargetType] = non_matchable_functions
        self.non_matchable_modules: set[NSNodeTargetType] = non_matchable_modules
        self.non_matchable_methods: set[NSNodeTargetType] = non_matchable_methods
        self.seen_nodes: set[Node] = set()
        self.stack: list[Node] = []
        for start_node in _get_output_nodes(self.gm.graph):
            self.stack.append(start_node)

    def __iter__(self):
        return self

    def __next__(self) -> NSSubgraph:
        """
        Returns the next matchable subgraph.
        """
        while len(self.stack) > 0:
            cur_end_node = self.stack.pop()
            if cur_end_node in self.seen_nodes:
                continue

            # for subgraphs which are single nodes, start_node == end_node
            # for subgraphs with more than one node, start node != end_node
            cur_start_node = cur_end_node
            # Subgraphs like linear-relu have the base node as the start node.
            # Subgraphs like dequantize-linear-relu-to(torch.float16) have the
            #   base node as the second node.
            # The cur_base_op_node var will move to the actual node during
            #   the fusion matching later in this code block.
            cur_base_op_node = cur_end_node

            # Check for potential fusions. For now, we are greedy
            # and always skip all non-base nodes of a fusion.  For example,
            # if we match linear-relu backwards, we will always skip the
            # relu node and attempt to match the linear node.  This can
            # be made configurable later if needed.
            for _reverse_fusion_ops, base_op_idx in get_reversed_fusions():
                is_match = end_node_matches_reversed_fusion(
                    cur_end_node, _reverse_fusion_ops, self.gm, self.seen_nodes
                )
                if is_match:
                    # navigate to the base node
                    # pyrefly: ignore [bad-assignment, non-convergent-recursion]
                    for rev_fusion_idx in range(len(_reverse_fusion_ops) - 1):
                        # pyrefly: ignore [bad-argument-type]
                        self.seen_nodes.add(cur_start_node)
                        # for now, assume that there are no other nodes
                        # which need to be added to the stack
                        cur_start_node = cur_start_node.args[0]  # type: ignore[assignment]
                        # if the base op index matches the current node, set it
                        rev_base_op_idx = len(_reverse_fusion_ops) - 2 - base_op_idx
                        if rev_fusion_idx == rev_base_op_idx:
                            cur_base_op_node = cur_start_node
                    break

            # pyrefly: ignore [bad-argument-type]
            self.seen_nodes.add(cur_start_node)
            # add args of previous nodes to stack
            # pyrefly: ignore [missing-attribute]
            for arg in cur_start_node.all_input_nodes:
                self._recursively_add_node_arg_to_stack(arg)

            # skip unmatchable nodes
            # note: this check is done on the start_node, i.e.
            # if we are matching linear-relu in reverse, this would do the matchable
            # check on the linear
            # pyrefly: ignore [bad-argument-type]
            if not self._is_matchable(cur_base_op_node):
                continue

            # If an observer or a fake_quant was not matched as a part of
            # a pattern of multiple nodes, ignore it. One case where this is
            # relevant is an observer on a graph input, which was added because
            # it is necessary for the next node.
            if cur_end_node.op == "call_module" and cur_start_node is cur_end_node:
                maybe_obs = getattr_from_fqn(self.gm, cur_end_node.target)  # type: ignore[arg-type]
                if isinstance(maybe_obs, (ObserverBase, FakeQuantizeBase)):
                    continue

            return NSSubgraph(
                # pyrefly: ignore [bad-argument-type]
                start_node=cur_start_node,
                end_node=cur_end_node,
                # pyrefly: ignore [bad-argument-type]
                base_op_node=cur_base_op_node,
            )

        raise StopIteration

    def _recursively_add_node_arg_to_stack(self, arg: Any) -> None:
        """
        Adds all of the nodes in this arg to the stack, properly navigating
        through list, dicts and tuples.
        """
        if isinstance(arg, Node):
            self.stack.append(arg)
        elif (
            isinstance(arg, torch.fx.immutable_collections.immutable_list)
            or type(arg) is tuple
        ):
            for inner_arg in arg:
                self._recursively_add_node_arg_to_stack(inner_arg)
        elif isinstance(arg, torch.fx.immutable_collections.immutable_dict):
            for value in arg.values():
                self._recursively_add_node_arg_to_stack(value)

    def _is_matchable(self, node: Node) -> bool:
        if node.op == "call_function":
            return node.target not in self.non_matchable_functions
        elif node.op == "call_module":
            if not isinstance(node.target, str):
                raise AssertionError(f"Expected str, got {type(node.target)}")
            target_mod = getattr_from_fqn(self.gm, node.target)
            return not any(
                isinstance(target_mod, t)  # type: ignore[arg-type]
                for t in self.non_matchable_modules
            )
        elif node.op == "call_method":
            return node.target not in self.non_matchable_methods
        else:
            return False


class GraphMatchingException(Exception):
    """
    Exception raised when two graphs cannot be matched.
    """


class SubgraphTypeRelationship(enum.Enum):
    # same type, known
    # example: F.linear and F.linear, or nn.Conv2d and nn.Conv2d
    EQUAL = enum.auto()
    # same type, but the type is not known to Numerical Suite
    # (user defined type, etc).
    EQUAL_BUT_UKNOWN = enum.auto()
    # known, same subgraph_relationship set, but not the same type
    # example: F.linear and toq.linear
    RELATED_BUT_NOT_EQUAL = enum.auto()
    # not related
    NOT_RELATED = enum.auto()


def _get_subgraph_relationship_type(
    subgraph_a: NSSubgraph,
    subgraph_b: NSSubgraph,
    gm_a: GraphModule,
    gm_b: GraphModule,
    type_a_related_to_b: set[tuple[NSNodeTargetType, NSNodeTargetType]],
) -> SubgraphTypeRelationship:
    node_a = subgraph_a.base_op_node
    node_b = subgraph_b.base_op_node

    # TODO(next): make this code handle matching by what is before the base op
    if node_a.op != node_b.op:
        if not (
            node_a.op in ("call_function", "call_method")
            and node_b.op in ("call_function", "call_method")
        ):
            return SubgraphTypeRelationship.NOT_RELATED

    if node_a.op in ("call_function", "call_method"):
        key = (node_a.target, node_b.target)

        if key not in type_a_related_to_b:
            if node_a.target == node_b.target:
                return SubgraphTypeRelationship.EQUAL_BUT_UKNOWN
            else:
                return SubgraphTypeRelationship.NOT_RELATED
        # after this point, we are dealing with known types

        if node_a.target == node_b.target:
            node_a_has_prev = subgraph_a.base_op_node == subgraph_a.start_node
            node_b_has_prev = subgraph_b.base_op_node == subgraph_b.start_node
            if node_a_has_prev and (not node_b_has_prev):
                return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL
            elif (not node_a_has_prev) and node_b_has_prev:
                return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL
            elif (not node_a_has_prev) and (not node_b_has_prev):
                return SubgraphTypeRelationship.EQUAL
            else:
                # TODO(future PR): check for matches start_op_node and base_op_node
                return SubgraphTypeRelationship.EQUAL

        if key in type_a_related_to_b:
            return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL
        else:
            return SubgraphTypeRelationship.NOT_RELATED
    elif node_a.op == "call_module":
        if (
            subgraph_a.base_op_node != subgraph_a.start_node
            or subgraph_b.base_op_node != subgraph_b.start_node
        ):
            raise AssertionError(
                "Matching call_module patterns where base_op_node != start_node is not supported yet"
            )
        # for call_module, we need to look up the modules to do the type check
        if not isinstance(node_a.target, str):
            raise AssertionError(f"Expected str, got {type(node_a.target)}")
        mod_a = getattr_from_fqn(gm_a, node_a.target)
        if not isinstance(node_b.target, str):
            raise AssertionError(f"Expected str, got {type(node_b.target)}")
        mod_b = getattr_from_fqn(gm_b, node_b.target)

        key = (type(mod_a), type(mod_b))

        if key not in type_a_related_to_b:
            if type(mod_a) is type(mod_b):
                return SubgraphTypeRelationship.EQUAL_BUT_UKNOWN
            else:
                return SubgraphTypeRelationship.NOT_RELATED
        elif type(mod_a) is type(mod_b):
            return SubgraphTypeRelationship.EQUAL
        else:
            return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL

    return SubgraphTypeRelationship.NOT_RELATED


def _get_name_for_subgraph(
    subgraph_a: NSSubgraph,
    gm_a: GraphModule,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]],
    existing_names: set[str],
) -> str:
    """
    Returns a unique name for a subgraph. This name is based on two things:
    1. the name of the set containing the underlying type of the base op in the
       subgraph (i.e. 'torch.nn.functional.linear' if this is related to a linear op)
    2. the number of previous subgraphs with related underlying type of the base op

    For example, in the graph

    linear0 -> relu0 -> linear1 -> relu1

    The subgraphs are (linear0, relu0) and (linear1, relu1).  If we iterate
    from the output node backwards, the name given to (linear1, relu1) will be
    `base_op_torch.nn.functional.linear_0`, and the name given to (linear0, relu0)
    will be `base_op_torch.nn.functional.linear_1`.

    Why are we not just using the node name? Answer: because of two requirements:
    A. fusions must be supported
    B. some Numeric Suite APIs can be called without having all of the models in memory

    For example, let's say we need to match nodes of

    (1) ... -> linear0 -> relu0 -> ...

    And

    (2) ... -> linear_relu0 -> ...

    Without being able to inspect them together. With the current naming scheme, if
    we iterate through both of these graphs in the same order, and assuming the rest
    of the graphs match, both of these subgraphs will get the same name without
    (1) and (2) knowing anything about each other.
    """
    target_type = _get_node_target_type(subgraph_a.base_op_node, gm_a)
    target_base_type = None
    for base_name, sets_of_related_ops in base_name_to_sets_of_related_ops.items():
        if target_type in sets_of_related_ops:
            target_base_type = base_name
    target_base_name = "base_op_" + str(target_base_type)
    counter = 0
    proposed_name = target_base_name + "_" + str(counter)
    while proposed_name in existing_names:
        counter += 1
        proposed_name = target_base_name + "_" + str(counter)
    existing_names.add(proposed_name)
    return proposed_name


def _get_node_target_type(node: Node, gm: GraphModule) -> NSNodeTargetType | None:
    if node.op in ("call_function", "call_method"):
        return node.target
    elif node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError(f"Expected str, got {type(node.target)}")
        mod = getattr_from_fqn(gm, node.target)
        return type(mod)
    return None


def get_matching_subgraph_pairs(
    gm_a: GraphModule,
    gm_b: GraphModule,
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None,
    unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None,
) -> dict[str, tuple[NSSubgraph, NSSubgraph]]:
    """
    Matches matchable subgraphs of graph_a to graph_b.

    For a node, "matchable" is defined as a node which is not an observer,
    fake_quants, quant or dequant.

    A subgraph can contain one or more nodes.  A subgraph is matchable if
    at least one node inside of it is matchable.  Currently, all nodes in
    a subgraph must be matchable (because we assume no observers will be
    inserted in the middle of a fusion).

    A subgraph is defined by (start_node, end_node).  We assume that only
    start_node and end_node are linked with the surrounding graph, all other
    nodes in a subgraph are self-contained.

    A pair of nodes is "related" if both nodes represent the same mathematical
    operation across different quantization flavors. For example,
    `F.linear` and `torch.ops.quantized.linear` are related, and
    `F.linear` and `torch.nn.Conv` are not related.

    For each matchable pair of nodes node_a and node_b, they will match
    if node_a and node_b are related.

    For graphs A and B, they will match iff:
    1. the number of matchable subgraphs in A and B is equivalent
    2. when iterating through the matchable subgraphs of A and B in the same order, each
       corresponding pair of base nodes is related.

    This enables us to find the corresponding subgraphs between
    graphs of related models.  For example, if we had two graphs such as:

    graph_a: x0 -> conv_0 (type: nn.Conv2d) -> obs_0 -> x1
             w  -/
             b  -/

    graph_b: x0 -> quant_0 -> qconv_0 (type: nnq.Conv2d) -> dequant_0 -> x1
           packed_params_0 -/

    This function will return the following result:
    {
        'conv_0': (  # the name of the node in graph_b
          (conv_0, conv_0),  # (start_node_a, end_node_a)
          (qconv_0, qconv_0),  # (start_node_b, end_node_b)
        ),
    }

    Or, if we have a fusion pattern,

    graph_a: x0 -> linear_0 -> relu_0 -> obs_0 -> x1
             w  -/
             b  -/

    graph_b: x0 -> quant_0 -> linear_relu_0 -> dequant_0 -> x1
           packed_params_0 -/

    This function will return the following result:
    {
        'linear_relu_0': (  # the name of the node in graph_b
          (linear_0, relu_0),  # (start_node_a, end_node_a)
          (linear_relu_0, linear_relu_0),  # (start_node_b, end_node_b)
        ),
    }
    """
    if unmatchable_types_map is None:
        unmatchable_types_map = get_unmatchable_types_map()
    non_matchable_functions = unmatchable_types_map["funs_unmatchable"]
    non_matchable_modules = unmatchable_types_map["mods_unmatchable"]
    non_matchable_methods = unmatchable_types_map["meths_unmatchable"]

    graph_a_iterator = _NSGraphMatchableSubgraphsIterator(
        gm_a, non_matchable_functions, non_matchable_modules, non_matchable_methods
    )
    graph_b_iterator = _NSGraphMatchableSubgraphsIterator(
        gm_b, non_matchable_functions, non_matchable_modules, non_matchable_methods
    )
    results = collections.OrderedDict()
    if base_name_to_sets_of_related_ops is None:
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    type_a_related_to_b = get_type_a_related_to_b(base_name_to_sets_of_related_ops)

    existing_names_a: set[str] = set()
    existing_names_b: set[str] = set()

    while True:
        # fetch the next subgraphs from a and b
        cur_subgraph_a, cur_subgraph_b = None, None
        try:
            cur_subgraph_a = next(graph_a_iterator)
        except StopIteration:
            pass
        try:
            cur_subgraph_b = next(graph_b_iterator)
        except StopIteration:
            pass

        # look up types of a and b for useful error messages
        type_start_a, type_start_b = None, None
        if cur_subgraph_a is not None:
            type_start_a = _get_node_target_type(cur_subgraph_a.start_node, gm_a)
        if cur_subgraph_b is not None:
            type_start_b = _get_node_target_type(cur_subgraph_b.start_node, gm_b)

        # check for results and determine what to do next
        if cur_subgraph_a is not None and cur_subgraph_b is not None:
            # both nodes were fetched, check for subgraph_relationship
            # note: subgraph_relationship is checked on the start node, i.e.
            # if a linear-relu pattern is checked, we would check for subgraph_relationship
            # of the linear
            subgraph_relationship = _get_subgraph_relationship_type(
                cur_subgraph_a, cur_subgraph_b, gm_a, gm_b, type_a_related_to_b
            )
            if subgraph_relationship == SubgraphTypeRelationship.NOT_RELATED:
                msg = f"""
The subgraphs
({cur_subgraph_a}, {type_start_a}) and
({cur_subgraph_b}, {type_start_b})
are not related. Please ensure that the two models you pass in have the same number
of subgraphs, and each pair of subgraphs is related to each other."""
                raise GraphMatchingException(msg)
            elif subgraph_relationship == SubgraphTypeRelationship.EQUAL_BUT_UKNOWN:
                # skip matching but unknown types
                continue
            key_name_a = _get_name_for_subgraph(
                cur_subgraph_a, gm_a, base_name_to_sets_of_related_ops, existing_names_a
            )
            key_name_b = _get_name_for_subgraph(
                cur_subgraph_b, gm_b, base_name_to_sets_of_related_ops, existing_names_b
            )
            if key_name_a != key_name_b:
                raise AssertionError(
                    f"Subgraph names {key_name_a} and {key_name_b} do not match"
                )
            results[key_name_a] = (cur_subgraph_a, cur_subgraph_b)
            continue
        elif cur_subgraph_a is None and cur_subgraph_b is None:
            # we reached the end of both graphs
            break
        else:
            # only one node was fetched, no match possible, throw error
            msg = f"""
Attempting to match
({cur_subgraph_a}, {type_start_a}) and
({cur_subgraph_b}, {type_start_b}),
one of which is empty. Please ensure that the two models you pass in have the same number
of subgraphs."""
            raise GraphMatchingException(msg)

    # The subgraph pairs are originally created by traversing the two graphs
    # from the outputs to the inputs. Reverse the results to return the
    # subgraphs in their order of execution.
    results = collections.OrderedDict(reversed(results.items()))

    return results
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import enum
import operator
from collections.abc import Callable

import torch
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.quantized as nnq
import torch.nn as nn
from torch.ao.quantization import FakeQuantizeBase, ObserverBase
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.utils import getattr_from_fqn
from torch.fx import GraphModule
from torch.fx.graph import Node

from .ns_types import NSNodeTargetType, NSResultsType


toq = torch.ops.quantized


# TODO(future PR): consider deleting this enum and using the torch types
# directly.  This might be tricky because it is not a one to one mapping.
class NodeInputOrOutputType(enum.Enum):
    FP32 = enum.auto()  # torch.float
    INT8 = enum.auto()  # torch.qint8 or torch.quint8
    FP16 = enum.auto()  # torch.float16
    UNKNOWN = enum.auto()  # we cannot determine input/output dtype
    # TODO(future PR): while these functions can support multiple dtypes,
    #   for the purposes of numerical debugging we want to get the actual
    #   dtype used in the model. We will likely need some kind of dtype
    #   propagation to estimate this.
    FP32_OR_INT8 = enum.auto()  # either torch.float or torch.quint8 or torch.qint8
    # TODO(future PRs): dynamic quant, fake quant, etc


def get_node_first_input_and_output_type(
    node: Node,
    gm: GraphModule,
    logger_cls: Callable,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]],
) -> tuple[NodeInputOrOutputType, NodeInputOrOutputType]:
    # TODO(future PR): clean this up
    FUNS_IO_TYPE_FP32 = node_type_to_io_type_map["funs_io_type_fp32"]
    FUNS_IO_TYPE_FP16 = node_type_to_io_type_map["funs_io_type_fp16"]
    FUNS_IO_TYPE_INT8 = node_type_to_io_type_map["funs_io_type_int8"]
    FUNS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["funs_io_type_fp32_or_int8"]
    MODS_IO_TYPE_FP32 = node_type_to_io_type_map["mods_io_type_fp32"]
    MODS_IO_TYPE_INT8 = node_type_to_io_type_map["mods_io_type_int8"]
    MODS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["mods_io_type_fp32_or_int8"]
    METHS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["meths_io_type_fp32_or_int8"]

    if node.op == "call_function":
        if node.target in FUNS_IO_TYPE_FP32:
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        if node.target in FUNS_IO_TYPE_FP16:
            return (NodeInputOrOutputType.FP16, NodeInputOrOutputType.FP16)
        elif node.target in FUNS_IO_TYPE_INT8:
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
        elif node.target in FUNS_IO_TYPE_FP32_OR_INT8:
            first_arg = get_normalized_nth_input(node, gm, 0)
            if not isinstance(first_arg, Node):
                raise AssertionError(f"Expected Node, got {type(first_arg)}")
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                first_arg, gm, logger_cls, node_type_to_io_type_map
            )
            return (prev_node_output_type, prev_node_output_type)
        else:
            return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)

    elif node.op == "call_module":
        if node.op != "call_module":
            raise AssertionError(f"Expected call_module, got '{node.op}'")
        if not isinstance(node.target, str):
            raise AssertionError(f"Expected str, but got {type(node.target)}")

        mod = getattr_from_fqn(gm, node.target)
        is_known_fp32_or_int8_input_module = any(
            isinstance(mod, target_type)  # type: ignore[arg-type]
            for target_type in MODS_IO_TYPE_FP32_OR_INT8
        )
        if (
            isinstance(mod, (logger_cls, ObserverBase, FakeQuantizeBase))  # type: ignore[arg-type]
            or is_known_fp32_or_int8_input_module
        ):
            # A logger or observer's input and output type is the output
            # type of the preceding node.
            first_arg = get_normalized_nth_input(node, gm, 0)
            if not isinstance(first_arg, Node):
                raise AssertionError(f"Expected Node, got {type(first_arg)}")
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                first_arg, gm, logger_cls, node_type_to_io_type_map
            )
            return (prev_node_output_type, prev_node_output_type)
        is_known_fp32_input_module = any(
            isinstance(mod, target_type)  # type: ignore[arg-type]
            for target_type in MODS_IO_TYPE_FP32
        )
        is_known_int8_input_module = any(
            isinstance(mod, target_type)  # type: ignore[arg-type]
            for target_type in MODS_IO_TYPE_INT8
        )
        if is_known_fp32_input_module:
            return (NodeInputOrOutputType.FP32, NodeInputOrOutputType.FP32)
        elif is_known_int8_input_module:
            return (NodeInputOrOutputType.INT8, NodeInputOrOutputType.INT8)
        else:
            return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)

    elif node.op == "call_method":
        if node.target == "dequantize":
            # Dequantize is a special node because it allows multiple input types.
            # So, we look up the output type of the previous node and return that
            # as the input type of this node instance.
            prev_node = get_normalized_nth_input(node, gm, 0)
            if not isinstance(prev_node, Node):
                raise AssertionError(f"Expected Node, got {type(prev_node)}")
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                prev_node, gm, logger_cls, node_type_to_io_type_map
            )
            return (prev_node_output_type, NodeInputOrOutputType.FP32)

        elif node.target == "to":
            # to is a special node because it allows multiple input types.
            # So, we look up the output type of the previous node and return that
            # as the input type of this node instance. We also look up the target
            # of to and return the correct output type.
            prev_node = get_normalized_nth_input(node, gm, 0)
            if not isinstance(prev_node, Node):
                raise AssertionError(f"Expected Node, got {type(prev_node)}")
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                prev_node, gm, logger_cls, node_type_to_io_type_map
            )

            cur_node_dtype_target = get_normalized_nth_input(node, gm, 1)
            if cur_node_dtype_target is not torch.float16:
                raise AssertionError(
                    f"{cur_node_dtype_target} handling needs to be added"
                )

            return (prev_node_output_type, NodeInputOrOutputType.FP16)

        elif node.target in METHS_IO_TYPE_FP32_OR_INT8:
            first_arg = get_normalized_nth_input(node, gm, 0)
            if not isinstance(first_arg, Node):
                raise AssertionError(f"Expected Node, got {type(first_arg)}")
            (
                _prev_node_input_type,
                prev_node_output_type,
            ) = get_node_first_input_and_output_type(
                first_arg, gm, logger_cls, node_type_to_io_type_map
            )
            return (prev_node_output_type, prev_node_output_type)

        return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)
    else:
        return (NodeInputOrOutputType.UNKNOWN, NodeInputOrOutputType.UNKNOWN)


def get_node_input_qparams(
    node: Node,
    gm: GraphModule,
    node_type_to_io_type_map: dict[str, set[NSNodeTargetType]],
) -> tuple[torch.Tensor | float, torch.Tensor | int] | None:
    """
    Returns the qparams (scale, zero_point) of the first input to `node`,
    if they can be inferred from the graph.
    """
    prev_node = get_normalized_nth_input(node, gm, 0)

    if not isinstance(prev_node, Node):
        return None

    MODS_IO_TYPE_FP32_OR_INT8 = node_type_to_io_type_map["mods_io_type_fp32_or_int8"]

    def _get_scale_zp_from_function_args(node, gm, scale_arg_idx, zp_arg_idx):
        scale_node = get_normalized_nth_input(node, gm, scale_arg_idx)
        zp_node = get_normalized_nth_input(node, gm, zp_arg_idx)
        if not isinstance(scale_node, Node):
            raise AssertionError(f"Expected Node, got {type(scale_node)}")
        if not isinstance(scale_node.target, str):
            raise AssertionError(f"Expected str, got {type(scale_node.target)}")
        if not isinstance(zp_node, Node):
            raise AssertionError(f"Expected Node, got {type(zp_node)}")
        if not isinstance(zp_node.target, str):
            raise AssertionError(f"Expected str, got {type(zp_node.target)}")
        scale_obj = getattr_from_fqn(gm, scale_node.target)
        zp_obj = getattr_from_fqn(gm, zp_node.target)
        return (scale_obj, zp_obj)

    if prev_node.op == "call_function":
        # quantize - read the args directly
        if prev_node.target is torch.quantize_per_tensor:
            return _get_scale_zp_from_function_args(prev_node, gm, 1, 2)
        elif prev_node.target in (toq.add, toq.add_relu, toq.mul, toq.mul_relu):
            return _get_scale_zp_from_function_args(prev_node, gm, 2, 3)

        return None
        # TODO(future PR): handle more functionals
        # TODO(future PR): handle functional ops which inherit qparams from input

    elif prev_node.op == "call_module":
        # get type of the module
        if not isinstance(prev_node.target, str):
            raise AssertionError(f"Expected str, got {type(prev_node.target)}")
        module_obj = getattr_from_fqn(gm, prev_node.target)
        if isinstance(
            module_obj,
            (
                nnq.Linear,
                nnq.Conv1d,
                nnq.Conv2d,
                nniq.ConvReLU2d,
                nnq.Conv3d,
                nnq.BatchNorm2d,
                nnq.BatchNorm3d,
                nnq.ConvTranspose1d,
                nnq.ConvTranspose2d,
                nnq.ELU,
                nnq.GroupNorm,
                nnq.InstanceNorm1d,
                nnq.InstanceNorm2d,
                nnq.InstanceNorm3d,
                nnq.LayerNorm,
                nnq.Hardswish,
                nnq.LeakyReLU,
                nnq.ReLU6,
                nniq.BNReLU2d,
                nniq.BNReLU3d,
                nniq.ConvReLU1d,
                nniq.ConvReLU2d,
                nniq.ConvReLU3d,
                nniq.LinearReLU,
            ),
        ):
            return (module_obj.scale, module_obj.zero_point)  # type: ignore[return-value]

        is_known_fp32_or_int8_input_module = any(
            isinstance(module_obj, target_type)  # type: ignore[arg-type]
            for target_type in MODS_IO_TYPE_FP32_OR_INT8
        )
        if is_known_fp32_or_int8_input_module:
            return get_node_input_qparams(prev_node, gm, node_type_to_io_type_map)

    return None


def return_first_non_observer_node(
    node: Node,
    gm: GraphModule,
) -> Node:
    """
    If node is not an observer, returns it.  If node is an observer,
    navigates up the graph and returns the first parent which is not an
    observer.  For example,

    graph: (node_non_obs), node = node_non_obs : returns node_non_obs
    graph: (node_non_obs -> obs0), node = obs0 : returns node_non_obs
    graph: (node_non_obs -> obs0 -> fq0), node = fq0 : returns node_non_obs
    """
    if node.op == "call_module":
        node_obj = getattr_from_fqn(gm, node.target)  # type: ignore[arg-type]
        if _is_activation_post_process(node_obj):
            if len(node.args) != 1:
                raise AssertionError(
                    f"Expected node.args to have length 1, got {len(node.args)}"
                )
            if not isinstance(node.args[0], Node):
                raise AssertionError(f"Expected Node, got {type(node.args[0])}")
            node = node.args[0]
            # code duplication intended, not worth refactoring
            if not isinstance(node.target, str):
                raise AssertionError(f"Expected str, got {type(node.target)}")
            node_obj = getattr_from_fqn(gm, node.target)
            if _is_activation_post_process(node_obj):
                if len(node.args) != 1:
                    raise AssertionError(
                        f"Expected node.args to have length 1, got {len(node.args)}"
                    )
                if not isinstance(node.args[0], Node):
                    raise AssertionError(f"Expected Node, got {type(node.args[0])}")
                node = node.args[0]
    return node


def get_number_of_non_param_args(
    node: Node,
    gm: GraphModule,
) -> int:
    """
    Assumes that all non-param args occur first. Returns the number of
    non-param args expected for a node.  For example, for

      F.linear(x, weight, bias)

    Returns 1, because x is a non-param arg and weight and bias are params.
    For

      lstm_mod(x, hid)

    Returns 2, because both x and hid are non-param args.
    """
    if node.op == "call_module":
        node_obj = getattr_from_fqn(gm, node.target)  # type: ignore[arg-type]
        if isinstance(node_obj, nn.LSTM):
            return 2

    # default is 1
    return 1


def get_arg_indices_of_inputs_to_log(node: Node) -> list[int]:
    """
    Returns the indices of args of the node which we should attach
    loggers to, if input logging is enabled.

    For example,
    * for (x + y), returns [0, 1]
    * for (1 + y), returns [1]
    * for (x + 1), returns [0]
    * for (linear(x, w, b)) returns [0]
    * by default, returns [0]
    """
    if len(node.args) == 0:
        return []
    if node.op == "call_function" and (
        # TODO(future PR): use relationship map instead of hardcoding
        node.target in (torch.add, torch.ops.quantized.add, operator.add)
        or node.target in (torch.mul, torch.ops.quantized.mul, operator.mul)
    ):
        result = [i for i in range(2) if type(node.args[i]) is Node]
        return result
    return [0]


def get_target_type_str(node: Node, gm: GraphModule) -> str:
    """
    Returns a string representation of the type of the function or module
    pointed to by this node, or '' for other node types.
    """
    target_type = ""
    if node.op in ("call_function", "call_method"):
        target_type = torch.typename(node.target)
    elif node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError(f"Expected str, got {type(node.target)}")
        target_mod = getattr_from_fqn(gm, node.target)
        target_type = torch.typename(target_mod)
    return target_type


def rekey_logger_info_on_node_name_of_model(
    results: NSResultsType,
    model_name: str,
) -> NSResultsType:
    """
    Rekeys the layer name of a results dictionary to use node names
    from `model_name`.

    For example, transforms

        {'base_op_1_0': {'node_output': {'model_a':
          [{'ref_node_name': 'linear1', ...}]}}}

    into

        {'linear1': {'node_output': {'model_a':
          [{'ref_node_name': 'linear1', ...}]}}}

    Note: we cannot use these node names directly because they are not
    guaranteed to be consistent across models. This is why we extract
    the results first and rekey afterwards.
    """
    new_results = {}
    for old_layer_name, result_type_to_results in results.items():
        new_layer_name = None
        for model_name_to_results in result_type_to_results.values():
            for cur_model_name, list_of_results in model_name_to_results.items():
                if cur_model_name == model_name:
                    if len(list_of_results) == 0:
                        raise AssertionError("Expected list_of_results to be not empty")
                    new_layer_name = list_of_results[0]["ref_node_name"]
                else:
                    continue
        if new_layer_name is not None:
            new_results[new_layer_name] = result_type_to_results
        else:
            new_results[old_layer_name] = result_type_to_results
    return new_results


def maybe_add_missing_fqns(results: NSResultsType) -> None:
    """
    If `fqn` entries are filled in for one of the models in `results`, copies
    them over to any models which do not have them filled out.

    A common use case benefitting from this is comparing a model prepared by
    quantization to a quantized model. In this case, the model prepared by
    quantization would have `fqn` entries, and the quantized model would not.
    """

    # Check in the first result to find any model with fqn entries defined.
    model_name_with_fqns = None
    for result_type_to_results in results.values():
        for model_name_to_results in result_type_to_results.values():
            for model_name, model_results in model_name_to_results.items():
                if len(model_results) > 0:
                    if model_results[0]["fqn"] is not None:
                        model_name_with_fqns = model_name
                        break
            break
        break

    if model_name_with_fqns:
        for result_type_to_results in results.values():
            for model_name_to_results in result_type_to_results.values():
                ref_model_results = model_name_to_results[model_name_with_fqns]
                for model_name, model_results in model_name_to_results.items():
                    if model_name == model_name_with_fqns:
                        continue

                    for i in range(len(model_results)):
                        fqn = ref_model_results[i]["fqn"]
                        model_results[i]["fqn"] = fqn


def maybe_dequantize_first_two_tensor_args_and_handle_tuples(f):
    def inner(*args, **kwargs):
        a0, a1, *a_other = args

        if (isinstance(a0, tuple) and isinstance(a1, tuple)) or (
            isinstance(a0, list) and isinstance(a1, list)
        ):
            results = []
            for el0, el1 in zip(a0, a1):
                new_args = (el0, el1, *a_other)
                results.append(inner(*new_args, **kwargs))
            return results

        elif isinstance(a0, torch.Tensor) and isinstance(a1, torch.Tensor):
            if a0.is_quantized:
                a0 = a0.dequantize()
            if a1.is_quantized:
                a1 = a1.dequantize()

        # for the purposes of this util, only handle floats
        if a0.dtype != torch.float or a1.dtype != torch.float:
            return None

        new_args = (a0, a1, *a_other)
        return f(*new_args, **kwargs)

    return inner


@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_sqnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the SQNR between `x` and `y`.

    Args:
        x: Tensor or tuple of tensors
        y: Tensor or tuple of tensors

    Return:
        float or tuple of floats
    """
    Ps = torch.linalg.vector_norm(x)
    Pn = torch.linalg.vector_norm(x - y)
    return 20 * torch.log10(Ps / Pn)


@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_normalized_l2_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized L2 error between `x` and `y`.

    Args:
        x: Tensor or tuple of tensors
        y: Tensor or tuple of tensors

    Return:
        float or tuple of floats
    """

    return torch.sqrt(((x - y) ** 2).sum() / (x**2).sum())


@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between `x` and `y`.

    Args:
        x: Tensor or tuple of tensors
        y: Tensor or tuple of tensors

    Return:
        float or tuple of floats
    """
    # For convolutions, the shape of the quantized weight has one additional
    # dimension compared to the shape of the fp32 weight. Match the shapes
    # to enable cosine similarity comparison.
    x = x.reshape(1, -1)
    y = y.reshape(1, -1)
    return torch.nn.functional.cosine_similarity(x, y)


def op_type_supports_shadowing(node: Node) -> bool:
    if node.op == "call_function":
        if node.target in (
            torch.add,
            torch.mul,
            operator.add,
            operator.mul,
            torch.cat,
            torch.stack,
        ):
            # shadowing for ops with multiple tensor inputs is not implemented yet
            return False
    return True


def get_normalized_nth_input(node: Node, gm: GraphModule, idx: int) -> Node:
    """
    Given a node, gets the n'th input to that node, normalizing
    args and kwargs to the best of its ability.
    """
    try:
        norm_args_and_kwargs = node.normalized_arguments(
            gm, normalize_to_only_use_kwargs=True
        )
        if norm_args_and_kwargs is not None:
            norm_args, norm_kwargs = norm_args_and_kwargs
            if len(norm_args) + len(norm_kwargs) <= idx:
                raise AssertionError(
                    f"Index {idx} out of range: total = {len(norm_args) + len(norm_kwargs)}"
                )
            if idx < len(norm_args):
                return norm_args[idx]
            else:
                # note: in Python 3.7+ dicts are ordered
                return list(norm_kwargs.values())[idx]
        else:
            if len(node.args) + len(node.kwargs) <= idx:
                raise AssertionError(
                    f"Index {idx} out of range: total = {len(node.args) + len(node.kwargs)}"
                )
            if idx < len(node.args):
                return node.args[idx]  # type: ignore[return-value]
            else:
                kwargs_idx = idx + len(node.args)
                return list(node.kwargs.values())[kwargs_idx]  # type: ignore[return-value]
    except RuntimeError:
        # this RuntimeError happens when node argument normalization
        # requires typehints to proceed, such as for torch.add where
        # either the first, second or both arguments could be tensors
        if len(node.args) + len(node.kwargs) <= idx:
            raise AssertionError(
                f"Index {idx} out of range: total = {len(node.args) + len(node.kwargs)}"
            ) from None
        if idx < len(node.args):
            return node.args[idx]  # type: ignore[return-value]
        else:
            kwargs_idx = idx + len(node.args)
            return list(node.kwargs.values())[kwargs_idx]  # type: ignore[return-value]
<|endoftext|>from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization import FakeQuantizeBase, ObserverBase
from torch.ao.quantization.backend_config import get_native_backend_config
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers
from torch.ao.quantization.utils import getattr_from_fqn
from torch.fx import GraphModule
from torch.fx.graph import Node

from .ns_types import NSNodeTargetType


toq = torch.ops.quantized


def get_type_a_related_to_b(
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]],
) -> set[tuple[NSNodeTargetType, NSNodeTargetType]]:
    # TODO(future PR): allow customizations
    # TODO(future PR): reuse existing quantization mappings
    # TODO(future PR): add the rest of modules and ops here
    type_a_related_to_b: set[tuple[NSNodeTargetType, NSNodeTargetType]] = set()

    for s in base_name_to_sets_of_related_ops.values():
        s_list = list(s)
        # add every bidirectional pair
        for idx_0 in range(len(s_list)):
            for idx_1 in range(idx_0, len(s_list)):
                type_a_related_to_b.add((s_list[idx_0], s_list[idx_1]))
                type_a_related_to_b.add((s_list[idx_1], s_list[idx_0]))

    return type_a_related_to_b


NSFusionElType = (
    Callable  # call_function or call_module type, example: F.linear or nn.Conv2d
    | str  # call_method name, example: "dequantize"
    | tuple[
        str, Any
    ]  # call_method name and first argument, example: ("to", torch.float16)
)
NSFusionType = (
    tuple[NSFusionElType, NSFusionElType]
    | tuple[NSFusionElType, NSFusionElType, NSFusionElType, NSFusionElType]
)


def get_reversed_fusions() -> list[tuple[NSFusionType, int]]:
    """
    Set of potential fusions, in reverse order.  The order is reversed
    to match how fusion patterns are defined in quantization code.

    Fusion format:
    ((fusion_op_0, fusion_op_1), base_op_idx)

    Where base_op_idx is the idx of the op we should use to match other related
    ops. Note: base_op_idx is specified in non-reverse order, i.e. a base_op_idx
    of 0 represents the first op in regular (non-reverse) order, 1 represents the
    second op, etc.
    """
    results: list[tuple[NSFusionType, int]] = []

    # Possible syntaxes:
    # * single op: torch.nn.Conv2d
    # * multiple ops: (torch.nn.ReLU, torch.nn.Conv2d)
    # For fusions, we only care about patterns composed of multiple ops.
    # TODO(future PR): allow customizations from default patterns.
    all_quant_patterns = _get_pattern_to_quantize_handlers(get_native_backend_config())

    default_base_op_idx = 0
    for quant_pattern in all_quant_patterns:
        # TODO: this is a temporary hack to flatten the patterns from quantization so
        # that it works with the ns matcher function, maybe we should use `_is_match`
        # in torch.ao.quantization.fx.match_utils to match the patterns
        if (
            isinstance(quant_pattern, tuple)
            and len(quant_pattern) == 2
            and isinstance(quant_pattern[1], tuple)
            and len(quant_pattern[1]) == 2
        ):
            # flatten the pattern with form (nn.ReLU, (nn.BatchNorm2d, nn.Conv2d))
            quant_pattern = (quant_pattern[0], quant_pattern[1][0], quant_pattern[1][1])

        # Only patterns of multiple ops are fusions, ignore
        # patterns which contain a single op (they get matched
        # without caring about fusions).
        if isinstance(quant_pattern, tuple):
            results.append((quant_pattern, default_base_op_idx))  # type: ignore[arg-type]

        # For each pattern, add additional patterns with observers and
        # fake quants at the end.
        # TODO(future PR): if needed, implement matching for a node
        #   having multiple output observers.
        for cls in (ObserverBase, FakeQuantizeBase):
            if isinstance(quant_pattern, tuple):
                # pyrefly: ignore [not-iterable]
                new_pattern = (cls, *quant_pattern)
            else:
                new_pattern = (cls, quant_pattern)
            results.append((new_pattern, default_base_op_idx))  # type: ignore[arg-type]

    # After this point, results contains values such as
    # [..., ((torch.nn.Relu, torch.nn.Conv2d), 0), ...]

    # Patterns for matching fp16 emulation are not specified in the quantization
    # fusion mappings.  For now, define them here.
    fp16_em_base_op_idx = 1
    patterns_to_add = [
        # linear-relu fp16 emulation:
        # fp16_to_fp32 -> linear -> relu -> fp32_to_fp16
        (
            (("to", torch.float16), F.relu, F.linear, "dequantize"),
            fp16_em_base_op_idx,
        ),
        # Conv-BN fusion (this happens outside of quantization patterns,
        # which is why it is defined separately here).
        ((nn.BatchNorm1d, nn.Conv1d), default_base_op_idx),
        ((nn.BatchNorm2d, nn.Conv2d), default_base_op_idx),
        ((nn.BatchNorm3d, nn.Conv3d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm1d, nn.Conv1d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm2d, nn.Conv2d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm3d, nn.Conv3d), default_base_op_idx),
    ]
    for p in patterns_to_add:
        results.append(p)  # type: ignore[arg-type]
        results.append(((ObserverBase, *p[0]), p[1]))  # type: ignore[arg-type]
        results.append(((FakeQuantizeBase, *p[0]), p[1]))  # type: ignore[arg-type]

    return results


def end_node_matches_reversed_fusion(
    end_node: Node,
    reversed_fusion: NSFusionType,
    gm: GraphModule,
    seen_nodes: set[Node],
) -> bool:
    """
    Returns true if a pattern ending with `end_node` matches
    the fusion pattern.
    """
    cur_node = end_node
    for fusion_idx in range(len(reversed_fusion)):
        # each node can only belong to one matched pattern
        if cur_node in seen_nodes:
            return False

        cur_fusion_el = reversed_fusion[fusion_idx]

        if cur_node.op == "call_function":
            fusion_el_is_fun = (not isinstance(cur_fusion_el, str)) and (
                not isinstance(cur_fusion_el, type)
            )
            if fusion_el_is_fun:
                if cur_node.target != cur_fusion_el:
                    return False
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        elif cur_node.op == "call_module":
            fusion_el_is_mod = isinstance(cur_fusion_el, type)
            if fusion_el_is_mod:
                if not isinstance(cur_node.target, str):
                    raise AssertionError(f"Expected str, got {type(cur_node.target)}")
                target_mod = getattr_from_fqn(gm, cur_node.target)
                if not isinstance(cur_fusion_el, type):
                    return False
                if not isinstance(target_mod, cur_fusion_el):
                    return False
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        elif cur_node.op == "call_method":
            fusion_el_is_meth_with_second_arg = (
                isinstance(cur_fusion_el, tuple) and len(cur_fusion_el) == 2
            )
            fusion_el_is_meth_without_args = isinstance(cur_fusion_el, str)
            if fusion_el_is_meth_without_args or fusion_el_is_meth_with_second_arg:
                if fusion_el_is_meth_without_args:
                    if cur_node.target != cur_fusion_el:
                        return False
                else:
                    if not isinstance(cur_fusion_el, tuple):
                        raise AssertionError(
                            f"Expected tuple, got {type(cur_fusion_el)}"
                        )
                    if cur_node.target != cur_fusion_el[0]:
                        return False
                    elif len(cur_node.args) < 2:
                        return False
                    elif cur_node.args[1] != cur_fusion_el[1]:
                        return False

                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False
        else:
            return False

    return True
<|endoftext|>from collections.abc import Callable

import torch
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.qat as nnqat
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule
from torch.fx.graph import Node

from .ns_types import NSSingleResultType, NSSingleResultValuesType
from .utils import get_target_type_str, getattr_from_fqn, return_first_non_observer_node


toq = torch.ops.quantized


def mod_weight_detach(mod: nn.Module) -> torch.Tensor:
    return mod.weight.detach()  # type: ignore[operator]


def mod_0_weight_detach(mod: nn.Module) -> torch.Tensor:
    return mod[0].weight.detach()  # type: ignore[index]


def mod_weight_bias_0(mod: nn.Module) -> torch.Tensor:
    return mod._weight_bias()[0]  # type: ignore[operator]


def get_lstm_weight(mod: nn.Module) -> list[torch.Tensor]:
    res = []
    for idx, param_name in enumerate(mod._flat_weights_names):  # type: ignore[arg-type]
        if "weight_ih_l" in param_name or "weight_hh_l" in param_name:
            param_value = mod._flat_weights[idx].detach()  # type: ignore[index,union-attr]
            res.append(param_value)
    return res


def get_qlstm_weight(mod: nn.Module) -> list[torch.Tensor]:
    res = []
    for weight_value in mod._all_weight_values:  # type: ignore[union-attr]
        res.append(weight_value.param.__getstate__()[0][4][0].__getstate__()[0][0])
        res.append(weight_value.param.__getstate__()[0][4][1].__getstate__()[0][0])
    return res


def get_conv_mod_weight(mod: nn.Module) -> torch.Tensor:
    if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return mod.weight.detach()
    elif isinstance(mod, (nni.ConvReLU1d, nni.ConvReLU2d, nni.ConvReLU3d)):
        return mod[0].weight.detach()  # type: ignore[operator]
    else:
        return mod._weight_bias()[0]  # type: ignore[operator]


def get_linear_mod_weight(mod: nn.Module) -> torch.Tensor:
    if isinstance(mod, nn.Linear):
        return mod.weight.detach()
    elif isinstance(mod, nni.LinearReLU):
        return mod[0].weight.detach()  # type: ignore[operator]
    else:
        return mod._weight_bias()[0]  # type: ignore[operator]


def get_lstm_mod_weights(mod: nn.Module) -> list[torch.Tensor]:
    # TODO(future PR): make more generic, handle everything
    if isinstance(mod, nn.LSTM):
        res = []
        for idx, param_name in enumerate(mod._flat_weights_names):
            if "weight_ih_l" in param_name or "weight_hh_l" in param_name:
                param_value = mod._flat_weights[idx].detach()  # type: ignore[index,union-attr]
                res.append(param_value)
        return res
    else:
        if not isinstance(mod, nnqd.LSTM):
            raise AssertionError(f"type {type(mod)} not handled yet")
        res = []
        for weight_value in mod._all_weight_values:
            res.append(
                weight_value.param.__getstate__()[0][4][0].__getstate__()[0][0]  # type: ignore[index]
            )
            res.append(
                weight_value.param.__getstate__()[0][4][1].__getstate__()[0][0]  # type: ignore[index]
            )
        return res


def get_conv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # traverse backwards from the weight arg, accounting for any observers
    weight_arg_node = node.args[1]
    if not isinstance(weight_arg_node, Node):
        raise AssertionError(f"Expected Node, got {type(weight_arg_node)}")
    weight_node = return_first_non_observer_node(weight_arg_node, gm)
    if not isinstance(weight_node, Node):
        raise AssertionError(f"Expected Node, got {type(weight_node)}")
    if weight_node.op != "get_attr":
        raise AssertionError(f"Expected get_attr, got {weight_node.op}")
    weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
    return weight.detach()


def get_qconv_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # qconv state is arg 1
    qconv_state_node = node.args[1]
    if not isinstance(qconv_state_node, Node):
        raise AssertionError(f"Expected Node, got {type(qconv_state_node)}")
    if qconv_state_node.op != "get_attr":
        raise AssertionError(f"Expected get_attr, got {qconv_state_node.op}")
    qconv_state_obj = getattr_from_fqn(gm, qconv_state_node.target)  # type: ignore[arg-type]
    return qconv_state_obj.weight()


def get_linear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # traverse backwards from the weight arg, accounting for any observers
    # supported patterns:
    # weight -> obs -> linear
    # weight -> to(torch.float16) -> dequantize -> linear
    linear_second_arg = node.args[1]
    if not isinstance(linear_second_arg, Node):
        raise AssertionError(f"Expected Node, got {type(linear_second_arg)}")

    if linear_second_arg.op == "call_module":
        # weight -> obs -> linear
        weight_arg_node = node.args[1]
        if not isinstance(weight_arg_node, Node):
            raise AssertionError(f"Expected Node, got {type(weight_arg_node)}")
        weight_node = weight_arg_node.args[0]
        if not isinstance(weight_node, Node):
            raise AssertionError(f"Expected Node, got {type(weight_node)}")
        if weight_node.op != "get_attr":
            raise AssertionError(f"Expected get_attr, got {weight_node.op}")
        weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
        return weight.detach()
    elif linear_second_arg.op == "call_method":
        # weight -> to(torch.float16) -> dequantize -> linear
        if linear_second_arg.op != "call_method":
            raise AssertionError(f"Expected call_method, got {linear_second_arg.op}")
        dequant_node = node.args[1]
        if not isinstance(dequant_node, Node):
            raise AssertionError(f"Expected Node, got {type(dequant_node)}")
        to_fp16_node = dequant_node.args[0]
        if not isinstance(to_fp16_node, Node):
            raise AssertionError(f"Expected Node, got {type(to_fp16_node)}")
        # extract the dtype, so we can cast to it before returning
        target_dtype = to_fp16_node.args[1]
        weight_node = to_fp16_node.args[0]
        if not isinstance(weight_node, Node):
            raise AssertionError(f"Expected Node, got {type(weight_node)}")
        if weight_node.op != "get_attr":
            raise AssertionError(f"Expected get_attr, got {weight_node.op}")
        weight = getattr_from_fqn(gm, weight_node.target)  # type: ignore[arg-type]
        # return the weight with fp16 cast
        return weight.detach().to(target_dtype)
    else:
        if linear_second_arg.op != "get_attr":
            raise AssertionError(f"Expected get_attr, got {linear_second_arg.op}")
        weight = getattr_from_fqn(gm, linear_second_arg.target)  # type: ignore[arg-type]
        return weight.detach()


def get_qlinear_fun_weight(node: Node, gm: GraphModule) -> torch.Tensor:
    # packed weight is arg 1
    packed_weight_node = node.args[1]
    if not isinstance(packed_weight_node, Node):
        raise AssertionError(f"Expected Node, got {type(packed_weight_node)}")
    if packed_weight_node.op != "get_attr":
        raise AssertionError(f"Expected get_attr, got {packed_weight_node.op}")
    packed_weight = getattr_from_fqn(gm, packed_weight_node.target)  # type: ignore[arg-type]
    # TODO(future PR): why does packed_weight.unpack() not work?
    (weight, _bias), _name = packed_weight.__getstate__()
    return weight


def get_op_to_type_to_weight_extraction_fn() -> dict[str, dict[Callable, Callable]]:
    op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]] = {
        "call_module": {
            # Conv1d
            nn.Conv1d: mod_weight_detach,
            nni.ConvReLU1d: mod_0_weight_detach,
            nnq.Conv1d: mod_weight_bias_0,
            nnqat.Conv1d: mod_weight_detach,
            nniqat.ConvBn1d: mod_weight_detach,
            nniqat.ConvBnReLU1d: mod_weight_detach,
            nniqat.ConvReLU1d: mod_weight_detach,
            nniq.ConvReLU1d: mod_weight_bias_0,
            # Conv2d
            nn.Conv2d: mod_weight_detach,
            nni.ConvReLU2d: mod_0_weight_detach,
            nnq.Conv2d: mod_weight_bias_0,
            nnqat.Conv2d: mod_weight_detach,
            nniqat.ConvBn2d: mod_weight_detach,
            nniqat.ConvBnReLU2d: mod_weight_detach,
            nniqat.ConvReLU2d: mod_weight_detach,
            nniq.ConvReLU2d: mod_weight_bias_0,
            # Conv3d
            nn.Conv3d: mod_weight_detach,
            nni.ConvReLU3d: mod_0_weight_detach,
            nnq.Conv3d: mod_weight_bias_0,
            nnqat.Conv3d: mod_weight_detach,
            nniqat.ConvBn3d: mod_weight_detach,
            nniqat.ConvBnReLU3d: mod_weight_detach,
            nniqat.ConvReLU3d: mod_weight_detach,
            nniq.ConvReLU3d: mod_weight_bias_0,
            # Linear
            nn.Linear: mod_weight_detach,
            nnq.Linear: mod_weight_bias_0,
            nni.LinearReLU: mod_0_weight_detach,
            nniq.LinearReLU: mod_weight_bias_0,
            nnqat.Linear: mod_weight_detach,
            nnqd.Linear: mod_weight_bias_0,
            nniqat.LinearReLU: mod_weight_detach,
            nniqat.LinearBn1d: mod_weight_detach,
            nn.modules.linear.NonDynamicallyQuantizableLinear: mod_weight_detach,
            # LSTM
            nn.LSTM: get_lstm_weight,
            nnqd.LSTM: get_qlstm_weight,
        },
        "call_function": {
            # Conv
            F.conv1d: get_conv_fun_weight,
            F.conv2d: get_conv_fun_weight,
            F.conv3d: get_conv_fun_weight,
            toq.conv1d: get_qconv_fun_weight,
            toq.conv2d: get_qconv_fun_weight,
            toq.conv3d: get_qconv_fun_weight,
            toq.conv1d_relu: get_qconv_fun_weight,
            toq.conv2d_relu: get_qconv_fun_weight,
            toq.conv3d_relu: get_qconv_fun_weight,
            # Linear
            F.linear: get_linear_fun_weight,
            toq.linear: get_qlinear_fun_weight,
            toq.linear_relu: get_qlinear_fun_weight,
        },
    }

    return op_to_type_to_weight_extraction_fn


def extract_weight_from_node(
    node: Node,
    gm: GraphModule,
    op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]]
    | None = None,
) -> NSSingleResultType | None:
    res_type = NSSingleResultValuesType.WEIGHT.value

    # Not all graphmodules have _node_name_to_scope, so only fill it
    # out if it exists.
    fqn = None
    if hasattr(gm, "_node_name_to_scope"):
        fqn = gm._node_name_to_scope[node.name][0]  # type: ignore[index]

    if op_to_type_to_weight_extraction_fn is None:
        op_to_type_to_weight_extraction_fn = get_op_to_type_to_weight_extraction_fn()

    ref_node_type = get_target_type_str(node, gm)
    # for extracting weights, these are always the same
    prev_node_type = ref_node_type

    if node.op == "call_function":
        function_mapping = op_to_type_to_weight_extraction_fn["call_function"]
        for target_fn_type, weight_extraction_fn in function_mapping.items():
            if node.target == target_fn_type:
                weight = weight_extraction_fn(node, gm)
                return {
                    "type": res_type,
                    "values": [weight],
                    "prev_node_name": node.name,
                    "prev_node_target_type": prev_node_type,
                    "ref_node_name": node.name,
                    "ref_node_target_type": ref_node_type,
                    "index_within_arg": 0,
                    "index_of_arg": 0,
                    "fqn": fqn,
                }

    elif node.op == "call_module":
        # for call_module, we need to look up the modules to do the type check
        if not isinstance(node.target, str):
            raise AssertionError(f"Expected str, got {type(node.target)}")
        mod = getattr_from_fqn(gm, node.target)
        module_mapping = op_to_type_to_weight_extraction_fn["call_module"]
        for target_mod_type, weight_extraction_fn in module_mapping.items():
            if type(mod) is target_mod_type:
                weight = weight_extraction_fn(mod)
                return {
                    "type": res_type,
                    "values": [weight],
                    "prev_node_name": node.name,
                    "prev_node_target_type": prev_node_type,
                    "ref_node_name": node.name,
                    "ref_node_target_type": ref_node_type,
                    "index_within_arg": 0,
                    "index_of_arg": 0,
                    "fqn": fqn,
                }

    return None
<|endoftext|># mypy: allow-untyped-defs
import collections
import copy
import operator
from collections.abc import Callable
from typing import Any

import torch
import torch.fx
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn
from torch.ao.ns.fx.ns_types import NSResultsType, NSSingleResultValuesType
from torch.ao.ns.fx.utils import (  # TODO(future PR): make this work correctly for methods
    get_normalized_nth_input,
    get_target_type_str,
)
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.fx.match_utils import _MatchResult
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn
from torch.fx import Graph, GraphModule, Node
from torch.utils._pytree import tree_map


SHADOW_NODE_NAME_PREFIX = "shadow"
SHADOW_WRAPPER_NODE_NAME_PREFIX = "shadow_wrapper"

# TODO(future PR): reuse existing mapping instead of creating a new one
BINARY_FUNCTIONS = {
    torch.add,
    torch.Tensor.add,
    operator.add,
    torch.mul,
    torch.Tensor.mul,
    operator.mul,
}


def _get_attr_name(subgraph_idx, subgraph_candidate_idx):
    return f"{SHADOW_NODE_NAME_PREFIX}_{subgraph_idx}_{subgraph_candidate_idx}"


def _get_attr_wrapper_name(subgraph_idx, subgraph_candidate_idx):
    return f"{SHADOW_WRAPPER_NODE_NAME_PREFIX}_{subgraph_idx}_{subgraph_candidate_idx}"


class OutputProp:
    """
    Output propagation (modeled from shape propagation).

    Given a GraphModule and an example input, saves the output flowing
    through each node on `node.traced_result`.

    Code based on the example from
    https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
    """

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env: dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target: str):
            target_atoms = target.split(".")
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
                    )
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == "placeholder":
                result = next(args_iter)
            elif node.op == "get_attr":
                result = fetch_attr(node.target)
            elif node.op == "call_function":
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == "call_method":
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == "call_module":
                result = self.modules[node.target](
                    *load_arg(node.args), **load_arg(node.kwargs)
                )

            if isinstance(result, torch.Tensor):  # type: ignore[possibly-undefined]
                # pyrefly: ignore [unbound-name]
                node.traced_result = result

            # pyrefly: ignore [unbound-name]
            env[node.name] = result

        return None


def _get_dedup_subgraphs(matches: dict[str, _MatchResult]) -> dict[str, list[Node]]:
    # the original matches variable is unique by node, make it unique by subgraph
    # instead
    seen_nodes = set()
    subgraphs_dedup = {}

    # Dict items are not reversible until Python 3.8, so we hack it
    # to be compatible with previous Python versions
    # TODO(future PR): try reversed(list(matches.items()))
    matches_items_reversed: list[tuple[str, _MatchResult]] = list(
        reversed(matches.items())
    )

    # Note: the order is important.  `matches` currently provides the matches
    # in reverse order.  We would like to process the matches in non-reverse
    # order, so that we can create an intuitive naming scheme, such as
    # naming the first op's submodules `shadow_0_0` through `shadow_0_(n-1)`
    for name, cur_match in matches_items_reversed:  # type: ignore[call-overload]
        was_seen = False
        for node_or_tuple in cur_match[1]:
            # Cur_match[1] has an unusual type. It says that it's a `List[Node]`,
            # but it is really not. Furthermore, the contents of this field
            # can change from match results of multiple nodes of the same pattern
            #
            # For example, for conv -> bn -> relu, we see
            # match_results = {
            #   'conv': (relu, [(bn, conv), relu], ...),
            #   'bn': (relu, [(bn, conv), relu], ...),
            #   'relu': (relu, [(bn, conv), relu], ...),
            # }
            #
            # Ideally we should clean up the `find_matches` function to make
            # this more intuitive. For the purposes of this prototype, we hack
            # around it.

            if isinstance(node_or_tuple, Node):
                if node_or_tuple in seen_nodes:
                    was_seen = True
                seen_nodes.add(node_or_tuple)

            else:
                if not isinstance(node_or_tuple, tuple):
                    raise AssertionError(f"Expected tuple, got {type(node_or_tuple)}")
                for node in node_or_tuple:
                    if not isinstance(node, Node):
                        raise AssertionError(f"Expected Node, got {type(node)}")
                    if node in seen_nodes:
                        was_seen = True
                    seen_nodes.add(node)

        if was_seen:
            continue

        # Start with the unusual type, convert it to [op_0, ..., op_n]
        list_of_nodes = []

        if len(cur_match[1]) == 1:
            list_of_nodes = cur_match[1]
        else:
            if len(cur_match[1]) != 2:
                raise ValueError(
                    f"Expected cur_match[1] to have length 2, got {len(cur_match[1])}"
                )
            # either (a, b), or ((a, b), c) or (c, (a, b))
            # cannot make any assumptions on order, not clear what the
            # _find_matches function is doing to populate this
            # TODO(future PR): make this code less confusing,  see discussion
            # in https://github.com/pytorch/pytorch/pull/80521/files#r975918836

            def _order_nodes(node_a, node_b, node_c) -> list[Node]:
                nodes = [node_a, node_b, node_c]
                first_node = None
                mid_node = None
                last_node = None
                for n in nodes:
                    prev_n = n.args[0]
                    next_n = next(iter(n.users))
                    if prev_n not in nodes:
                        first_node = n
                    elif next_n not in nodes:
                        last_node = n
                    else:
                        mid_node = n
                if first_node is None or mid_node is None or last_node is None:
                    raise AssertionError("Expected all nodes to be non-None")
                if mid_node.args[0] is not first_node:
                    raise AssertionError("Expected mid_node.args[0] to be first_node")
                if last_node.args[0] is not mid_node:
                    raise AssertionError("Expected last_node.args[0] to be mid_node")
                return [last_node, mid_node, first_node]

            if isinstance(cur_match[1][0], Node) and isinstance(cur_match[1][1], Node):
                # (a, b)
                list_of_nodes = cur_match[1]
            elif isinstance(cur_match[1][0], tuple):
                # ((a, b), c)
                node_a, node_b = cur_match[1][0]
                node_c = cur_match[1][1]
                list_of_nodes = _order_nodes(node_a, node_b, node_c)
            elif isinstance(cur_match[1][1], tuple):
                # (a, (b, c))
                node_a, node_b = cur_match[1][1]
                node_c = cur_match[1][0]
                list_of_nodes = _order_nodes(node_a, node_b, node_c)

        # [node_n, ..., node_0], note that the order is reversed
        # to make it chronological for simple subgraphs
        list_of_nodes.reverse()
        subgraphs_dedup[name] = list_of_nodes

    return subgraphs_dedup


def _get_logger_for_subgraph(
    model: GraphModule,
    first_node: Node,
    last_node: Node,
    subgraph_idx: int,
    subgraph_candidate_idx: int,
    qconfig_str: str,
    logger_cls: Callable,
    fqn: str | None,
) -> torch.nn.Module:
    """
    Given a model and a linear subgraph starting from `first_node` and
    ending with `last_node`, creates a logger for the end of this
    subgraph.
    """
    if fqn is None:
        fqn = ""
    logger_mod_orig = logger_cls(
        first_node.name,  # ref_node_name
        last_node.name,  # prev_node_name
        f"subgraph_{subgraph_idx}_{subgraph_candidate_idx}",  # model_name
        "model",  # ref_name
        get_target_type_str(last_node, model),  # prev_node_target_type
        get_target_type_str(first_node, model),  # ref_node_target_type
        NSSingleResultValuesType.NODE_OUTPUT.value,  # results_type
        0,  # index_within_arg
        0,  # index_of_arg
        fqn,  # fqn
        qconfig_str,
    )
    # Usually we expect the user to add loggers, then calibrate, then convert,
    # and then populate loggers.  This is why the loggers start disabled.
    # TODO(future PR): reconsider the design to make this more intuitive.
    logger_mod_orig.enabled = False
    return logger_mod_orig


def create_submodule_from_subgraph(
    model: torch.nn.Module,
    first_node: Node,
    last_node: Node,
) -> GraphModule:
    """
    Input: a model, and a linear subgraph within the model from first_node to
      last_node.

    Output: a new submodule containing a copy of the subgraph, with the inputs
      to the first node becoming the inputs to the submodule, and all other
      nodes in the subgraph being copied.

    Example inputs:

    `model`: a module with graph

      x0 -> op1 -> x1 -> op2 -> x2
             |
            arg1

    `first_node`: op1
    `last_node`: op2

    Example output: a new module with graph

      input1 -> op1_copy -> x1 -> op2_copy -> output1
                   |
                  arg1
    """

    #
    # create a blank GraphModule with an empty graph
    #

    class M(torch.nn.Module):
        def forward(self, x):
            pass

    m = M()
    gm = torch.fx.symbolic_trace(m)
    g = gm.graph
    for node in reversed(gm.graph.nodes):
        g.erase_node(node)

    #
    # modify the graph to have a copy of our subgraph
    #

    cur_node_orig = first_node

    cur_name_idx = 0

    iteration_limit = 100
    cur_iteration = 0

    while True:
        if cur_node_orig is first_node:
            # we are at the first node, we need to set up graph inputs
            # TODO(future): some graphs could have placeholders which are unrelated
            # to the first node, need to handle this
            cur_args_copy = []
            cur_kwargs_copy = {}
            seen_names: set[str] = set()
            old_name_to_new_node: dict[str, Node] = {}

            def _add_placeholder(
                g: Graph, node: Node, seen_names, old_name_to_new_node
            ):
                # note: for graphs starting with patterns such as `y = x + x`, we
                # need to ensure we do not add multiple placeholders with the
                # same name
                counter = 0
                while node.name + "_" + str(counter) in seen_names:
                    counter += 1
                cur_name = node.name + "_" + str(counter)
                seen_names.add(cur_name)
                placeholder = g.placeholder(cur_name)
                old_name_to_new_node[node.name] = placeholder
                return placeholder

            for arg in cur_node_orig.args:
                if isinstance(arg, Node):
                    p = _add_placeholder(g, arg, seen_names, old_name_to_new_node)
                    cur_args_copy.append(p)
                elif isinstance(arg, (list, tuple)):
                    new_arg = []
                    for inner_arg in arg:
                        if isinstance(inner_arg, Node):
                            new_arg.append(
                                _add_placeholder(
                                    g, inner_arg, seen_names, old_name_to_new_node
                                )
                            )
                        else:
                            new_arg.append(inner_arg)
                    cur_args_copy.append(new_arg)
                else:
                    cur_args_copy.append(arg)

            # TODO(future PR): handle non-normalized kwargs
            for kwarg_name, kwarg in cur_node_orig.kwargs.items():
                if isinstance(kwarg, Node):
                    cur_kwargs_copy[kwarg_name] = _add_placeholder(
                        g, kwarg, seen_names, old_name_to_new_node
                    )
                elif isinstance(kwarg, (list, tuple)):
                    new_kwarg = []
                    for inner_kwarg in kwarg:
                        p = _add_placeholder(
                            g,
                            inner_kwarg,  # type: ignore[arg-type]
                            seen_names,
                            old_name_to_new_node,
                        )
                        new_kwarg.append(p)
                    cur_kwargs_copy[kwarg_name] = new_kwarg
                else:
                    cur_kwargs_copy[kwarg_name] = kwarg

            cur_args_copy = tuple(cur_args_copy)  # type: ignore[assignment]
        else:
            # we are not at first node, first arg is from the previous node,
            # and all other args are copied

            # the current implementation is simplistic and cannot handle
            # ops with two or more arguments which need to be passed from
            # the previous op, so we assert them out
            if cur_node_orig.target in BINARY_FUNCTIONS:
                raise AssertionError(
                    f"Unexpected binary function target: {cur_node_orig.target}"
                )

            # at this point in the code, cur_node_copy is pointing to the copy
            # of the previous node
            # TODO(future PR): this is not handling complicated graphs correctly, need to
            # look at actual relationships instead of assuming sequential graph
            # TODO(future PR): this is ignoring kwargs, will need to support kwargs
            # for any fusion pattern which has them for a node that is not the
            # first node.
            cur_args_copy = [cur_node_copy]  # type: ignore[has-type, possibly-undefined]  # noqa: F821

            if len(cur_node_orig.args) > 1:
                for arg in cur_node_orig.args[1:]:
                    if isinstance(arg, torch.nn.Parameter):
                        new_arg = arg.detach().clone()  # type: ignore[assignment]
                        mod_name = f"mod_{cur_name_idx}"
                        cur_name_idx += 1
                        setattr(gm, mod_name, new_arg)
                        new_arg_placeholder = gm.placeholder(mod_name)  # type: ignore[operator]
                        cur_args_copy.append(new_arg_placeholder)
                    elif isinstance(arg, (float, int, torch.dtype)):
                        # pyrefly: ignore [bad-argument-type]
                        cur_args_copy.append(arg)
                    else:
                        raise AssertionError(f"arg of type {type(arg)} not handled yet")
            cur_args_copy = tuple(cur_args_copy)  # type: ignore[assignment]

        # copy the node
        if cur_node_orig.op == "call_module":
            orig_mod = getattr_from_fqn(model, cur_node_orig.target)  # type: ignore[arg-type]
            orig_mod_copy = copy.deepcopy(orig_mod)
            mod_name = f"mod_{cur_name_idx}"
            setattr(gm, mod_name, orig_mod_copy)
            cur_name_idx += 1
            cur_node_copy = g.call_module(mod_name, cur_args_copy, cur_kwargs_copy)  # type: ignore[possibly-undefined,arg-type]

        elif cur_node_orig.op == "call_function":
            cur_node_copy = g.call_function(
                cur_node_orig.target,  # type: ignore[arg-type]
                cur_args_copy,  # type: ignore[arg-type]
                cur_kwargs_copy,  # type: ignore[possibly-undefined]
            )

        elif cur_node_orig.op == "call_method":
            cur_node_copy = g.call_method(
                cur_node_orig.target,  # type: ignore[arg-type]
                cur_args_copy,  # type: ignore[arg-type]
                cur_kwargs_copy,  # type: ignore[possibly-undefined]
            )

        else:
            raise AssertionError(f"{cur_node_orig.op} not supported yet")

        if cur_node_orig is last_node:
            break

        # go to next node
        if len(cur_node_orig.users.keys()) != 1:
            raise AssertionError(
                f"{cur_node_orig} has more than 1 users, not supported yet"
            )
        cur_node_orig = next(iter(cur_node_orig.users.keys()))
        cur_iteration += 1
        if cur_iteration > iteration_limit:
            raise AssertionError("iteration limit exceeded")

    # set up outputs
    g.output(cur_node_copy)

    gm.recompile()
    return gm


def create_one_transformed_and_logged_copy_of_subgraph(
    mt: GraphModule,
    subgraph_idx: int,
    subgraph_candidate_idx: int,
    first_node: Node,
    last_node: Node,
    fqn: str | None,
    list_of_node_name_to_qconfig: list[dict[str, QConfigAny]],
    example_inputs: Any,
    last_added_shadow_node_list: list[Node | None],
    custom_prepare_fn: Callable | None = None,
    custom_prepare_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Given a subgraph in `mt` and a subgraph candidate idx, inserts the
    subgraph candidate copy and instruments it with loggers.

    If subgraph_candidate_idx is 0, this is the baseline fp32 subgraph and we just
    add a logger to the end.

    If subgraph_candidate_idx is not 0, we create a copy of the subgraph and
    prepare it with `prepare_fx`.
    """

    # TODO(future PR): move logger classes to utils to remove circular dependency
    from torch.ao.ns._numeric_suite_fx import OutputComparisonLogger, OutputLogger

    if subgraph_candidate_idx == 0:
        # idx = 0 is the floating point (original) version of the subgraph
        # We keep the subgraph as is, and add a logger at the end

        qconfig_str = ""
        logger_mod_orig = _get_logger_for_subgraph(
            mt,
            first_node,
            last_node,
            subgraph_idx,
            subgraph_candidate_idx,
            qconfig_str,
            OutputLogger,
            fqn,
        )

        attr_name = _get_attr_name(subgraph_idx, subgraph_candidate_idx)
        if hasattr(mt, attr_name):
            raise AssertionError(f"Unexpected attribute '{attr_name}' found in {mt}")
        setattr(mt, attr_name, logger_mod_orig)
        with mt.graph.inserting_after(last_node):
            new_node = mt.graph.call_module(attr_name, args=(last_node,), kwargs={})
            last_added_shadow_node_list[0] = new_node

    else:
        # idx > 0 means we have a candidate qconfig to try, so we need
        # to make a copy of the subgraph, feed it with the right inputs,
        # and add a logger at the end

        # get the qconfig
        # subtract one because the first candidate is the floating point
        # version of the subgraph
        node_name_to_qconfig = list_of_node_name_to_qconfig[subgraph_candidate_idx - 1]
        qconfig = node_name_to_qconfig[first_node.name]

        # if no quantization is requested, skip
        # TODO(future PR): deduplicate equivalent qconfigs that come from
        #   different qconfig mapping objects
        if qconfig is None:
            return

        qconfig_mapping = QConfigMapping().set_global(qconfig)

        # create a copy of the submodule, wrapped in a separate module
        orig_mod_copy_wrapped = create_submodule_from_subgraph(
            mt, first_node, last_node
        )

        # add a call to prepare_fx on the wrapper module
        if custom_prepare_fn is None:
            orig_mod_copy_wrapped = torch.ao.quantization.quantize_fx.prepare_fx(
                orig_mod_copy_wrapped, qconfig_mapping, example_inputs=example_inputs
            )
        else:
            if custom_prepare_kwargs is None:
                custom_prepare_kwargs = {}
            for kwarg_name in [
                "example_inputs",
                "prepare_custom_config",
                "qconfig_mapping",
            ]:
                if kwarg_name in custom_prepare_kwargs:
                    raise AssertionError(
                        f"cannot specify {kwarg_name} in custom_prepare_kwargs"
                    )
            prepare_kwargs: dict[str, Any] = {
                "example_inputs": example_inputs,
                "qconfig_mapping": qconfig_mapping,
            }
            prepare_kwargs.update(custom_prepare_kwargs)
            orig_mod_copy_wrapped = custom_prepare_fn(
                orig_mod_copy_wrapped, **prepare_kwargs
            )

        # attach the wrapper to the model
        attr_name = _get_attr_wrapper_name(subgraph_idx, subgraph_candidate_idx)
        if hasattr(mt, attr_name):
            raise AssertionError(f"Unexpected attribute '{attr_name}' found in {mt}")
        setattr(mt, attr_name, orig_mod_copy_wrapped)

        # add a call to the wrapper module from the parent graph
        insert_after_node = last_added_shadow_node_list[0]
        with mt.graph.inserting_after(insert_after_node):
            # TODO(future PR): handle fusion patterns where non-first nodes
            # need inputs

            # pass in all node args and kwargs

            new_args = []
            for arg in first_node.args:
                if isinstance(arg, Node):
                    new_args.append(arg)
                elif (
                    isinstance(arg, (list, tuple))
                    and len(arg)
                    and isinstance(arg[0], Node)
                ):
                    new_args.extend(
                        inner_arg for inner_arg in arg if isinstance(inner_arg, Node)
                    )

            new_kwargs = {}
            for name, old_kwarg in first_node.kwargs.items():
                if isinstance(old_kwarg, Node):
                    new_kwargs[name] = old_kwarg
                elif isinstance(old_kwarg, (list, tuple)) and len(old_kwarg):
                    # TODO(future PR): clarify why we are adding kwargs to args
                    new_args.extend(old_kwarg)  # type: ignore[arg-type]

            new_args = tuple(new_args)  # type: ignore[assignment]

            new_node = mt.graph.call_module(attr_name, args=new_args, kwargs=new_kwargs)  # type: ignore[arg-type]

        # add a logger to parent graph to observe the shadow wrapper
        logger_mod_orig = _get_logger_for_subgraph(
            mt,
            first_node,
            last_node,
            subgraph_idx,
            subgraph_candidate_idx,
            str(qconfig),
            OutputComparisonLogger,
            fqn,
        )

        attr_name = _get_attr_name(subgraph_idx, subgraph_candidate_idx)
        if hasattr(mt, attr_name):
            raise AssertionError(f"Unexpected attribute '{attr_name}' found in {mt}")
        setattr(mt, attr_name, logger_mod_orig)
        with mt.graph.inserting_after(new_node):
            logger = mt.graph.call_module(
                attr_name, args=(new_node, last_node), kwargs={}
            )
            last_added_shadow_node_list[0] = logger

    mt.recompile()


def create_n_transformed_and_logged_copies_of_subgraph(
    mt: GraphModule,
    subgraph_idx: int,
    match_name: str,
    nodes_in_this_subgraph: list[Any],
    qconfig_mappings: list[QConfigMapping],
    list_of_node_name_to_qconfig: list[dict[str, QConfigAny]],
    custom_prepare_fn: Callable | None = None,
    custom_prepare_kwargs: dict[str, Any] | None = None,
) -> None:
    """
    Given a model `mt` and a subgraph_idx, creates the needed copies
    of the subgraph for all qconfigs, and instruments them with loggers.
    """
    # for now, assume that
    # 1. the first node has one input
    # 2. the last node has one output

    # for now, ignore all subgraphs that contain non-nodes (tuples, etc)
    # TODO(future PR): implement this
    if any(not isinstance(node, Node) for node in nodes_in_this_subgraph):
        return

    first_node = nodes_in_this_subgraph[0]
    last_node = nodes_in_this_subgraph[-1]
    # We used output propagation to populate example values on each
    # node. Use the example values from the previous node as the input
    # to the current node.
    prev_node = get_normalized_nth_input(first_node, mt, 0)
    if isinstance(prev_node, list):
        example_inputs = [x.traced_result for x in prev_node]
    elif isinstance(prev_node, tuple):
        example_inputs = (x.traced_result for x in prev_node)  # type: ignore[assignment]
    else:
        # currently some customer models do not have a traced_result in
        # every node, so we have to guard for this case since we cannot
        # quantize without an example input
        # TODO(future PR): add a test case for this once we have an easy
        # repro, see https://github.com/pytorch/pytorch/pull/80521/files#r975940489
        # for additional context
        if hasattr(prev_node, "traced_result"):
            example_inputs = (prev_node.traced_result,)  # type: ignore[attr-defined, assignment]
        else:
            print(
                "unable to get example input for node "
                + f"{first_node.format_node()}, skipping"
            )
            return

    # If there are no quantization configs for this subgraph, skip adding
    # loggers. This reduces memory usage for models where not all layers are
    # quantized.
    # TODO(future): consider making this configurable
    found_at_least_one_qconfig = False
    for subgraph_candidate_idx in range(len(qconfig_mappings) + 1):
        if subgraph_candidate_idx == 0:
            # fp32 baseline does not need a qconfig
            continue

        # a. we have N shadows, so len(qconfig_mappings) is N
        # b. we will have the fp32 layer + N shadows, so overall number of
        #    (original_op) + (*shadows) will be N+1
        # c. since `subgraph_candidate_idx` represents (b), we need
        #    to subtract 1 to query from (a)
        node_name_to_qconfig = list_of_node_name_to_qconfig[subgraph_candidate_idx - 1]
        qconfig = node_name_to_qconfig[first_node.name]
        if qconfig is not None:
            found_at_least_one_qconfig = True
            break
    if not found_at_least_one_qconfig:
        print(
            "unable to find at least one qconfig for node "
            + f"{first_node.format_node()}, skipping"
        )
        return

    fqn = _maybe_get_fqn(first_node, mt)

    # We want the results to contain the subgraphs in natural order,
    # and the graph to also contain shadow wrappers and shadow loggers
    # in natural order.
    # If we just iterate in reverse, the graph will be in natural
    # order but the eventual results will be in reverse order.
    # So, we keep track of the last shadow logger we added and
    # always insert after it.
    last_added_shadow_node_list: list[Node | None] = [None]
    for subgraph_candidate_idx in range(len(qconfig_mappings) + 1):
        create_one_transformed_and_logged_copy_of_subgraph(
            mt,
            subgraph_idx,
            subgraph_candidate_idx,
            first_node,
            last_node,
            fqn,
            list_of_node_name_to_qconfig,
            example_inputs,
            last_added_shadow_node_list,
            custom_prepare_fn,
            custom_prepare_kwargs,
        )


def create_add_loggers_graph(
    model: GraphModule,
    subgraphs_dedup: dict[str, list[Node]],
    qconfig_mapping: QConfigMapping,
    node_name_to_qconfig: dict[str, QConfigAny],
) -> None:
    r"""
    Given a model, a model graph partition (currently a set of matched
    subgraphs) and instructions how to transform each subgraph
    (currently quantizing it according to qconfig_mapping), modifies
    the model graph to create an alternate path through the original graph,
    with each of the subgraphs quantized.  This is useful to compare
    propagation error of a transformation such as quantization.

    For example, given layer op0 and op1, there are four cases when handling op1:
    1. op0 and op1 quantized
    2. op0 and op1 unquantized
    3. op0 quantized, op1 unquantized
    4. op0 unquantized, op1 quantized

    Example input, case 1:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \                        \          \                 \       # noqa: W605
         ---> op0_1 -> x1_1 ----> clog    op1_1 -> x2_1 ----> clog

    Example output, case 1:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \                        \                           \        # noqa: W605
         ---> op0_1 -> x1_1 ----> clog -> op1_1 -> x2_1 ----> clog

    """
    # TODO(future PR): move logger classes to utils to remove circular dependency
    from torch.ao.ns._numeric_suite_fx import OutputComparisonLogger, OutputLogger

    def _get_subgraph_containing_node(node, subgraphs_dedup):
        for subgraph in subgraphs_dedup.values():
            if node in subgraph:
                return subgraph
        return None

    # First, we need to create shadow branches, going from
    #
    #   x0 -> op0 -> x1 -> ...
    #
    #
    # to
    #
    #   x0 -> op0_0 -> x1_0 -> log -> ...
    #    \                     \
    #      -> op0_1 -> x1_1 -> clog
    #
    # Later, the outputs of each shadow will be rerouted to calculate
    # propagation error.

    # Note: we cannot iterate over matched subgraphs because some nodes
    # may not be matched. So, we iterate over nodes in the graph, and
    # associate them to matched subgraphs if possible.

    nodes_to_skip = set()
    # for each subgraph, save a mapping from first node of subgraph
    # to first and last node of the shadow of this subgraph
    orig_first_node_to_shadow_in_node = {}
    orig_first_node_to_shadow_out_node = {}
    # need to record original list because we will mutate the graph as we go
    orig_nodes = list(model.graph.nodes)  # type: ignore[union-attr, arg-type]
    cur_subgraph_idx = 0
    for n in orig_nodes:
        if n.op in ("placeholder", "get_attr", "output") or n in nodes_to_skip:
            continue

        maybe_subgraph = _get_subgraph_containing_node(n, subgraphs_dedup)
        insert_submodule_copy = False
        if maybe_subgraph is not None:
            first_node, last_node = maybe_subgraph[0], maybe_subgraph[-1]
            nodes_to_skip.update(maybe_subgraph)
            qconfig = node_name_to_qconfig[first_node.name]
            if qconfig is not None:
                insert_submodule_copy = True
        else:
            first_node, last_node = n, n

        if insert_submodule_copy:
            match_name = first_node.name
            create_n_transformed_and_logged_copies_of_subgraph(
                model,
                cur_subgraph_idx,
                match_name,
                maybe_subgraph,
                [qconfig_mapping],
                [node_name_to_qconfig],
                None,
                None,  # type: ignore[arg-type]
            )
            # find the created shadow module and record it so we
            # can find it easily in step 2
            expected_shadow_target = f"shadow_wrapper_{cur_subgraph_idx}_1"
            new_shadow_mod = None
            for maybe_shadow_mod in model.graph.nodes:
                if (
                    maybe_shadow_mod.op == "call_module"
                    and maybe_shadow_mod.target == expected_shadow_target
                ):
                    new_shadow_mod = maybe_shadow_mod
                    break
            if new_shadow_mod is None:
                raise AssertionError("Expected new_shadow_mod to be non-None")
            orig_first_node_to_shadow_in_node[first_node] = new_shadow_mod
            orig_first_node_to_shadow_out_node[first_node] = new_shadow_mod

        else:
            # create a copy of the subgraph by only copying FX nodes
            # but not copying any parameters, to minimize memory usage
            subgraph_to_use = (
                maybe_subgraph if maybe_subgraph is not None else [first_node]
            )

            # add a regular logger after last_node
            qconfig_str = ""
            subgraph_candidate_idx = 0
            fqn = _maybe_get_fqn(first_node, model)
            logger_mod_orig = _get_logger_for_subgraph(
                model,
                first_node,
                last_node,
                cur_subgraph_idx,
                subgraph_candidate_idx,
                qconfig_str,
                OutputLogger,
                fqn,
            )
            attr_name = _get_attr_name(cur_subgraph_idx, subgraph_candidate_idx)
            if hasattr(model, attr_name):
                raise AssertionError(
                    f"Unexpected attribute '{attr_name}' found in {model}"
                )
            setattr(model, attr_name, logger_mod_orig)
            insertion_point = last_node
            with model.graph.inserting_after(insertion_point):
                logger = model.graph.call_module(
                    attr_name, args=(last_node,), kwargs={}
                )
                insertion_point = logger

            # create a copy of the subgraph
            cur_node_orig = first_node
            cur_node_copy = None
            first_node_copy = None
            # pyrefly: ignore [bad-assignment]
            while cur_node_orig in subgraph_to_use:
                # TODO(future PR): make this support all possible args/kwargs
                if cur_node_orig is first_node:
                    new_args = cur_node_orig.args
                    new_kwargs = cur_node_orig.kwargs
                else:
                    first_arg_for_copy: Node | None = cur_node_copy
                    new_args = (first_arg_for_copy, *cur_node_orig.args[1:])
                    new_kwargs = cur_node_orig.kwargs
                # make a copy of cur_node_orig
                with model.graph.inserting_after(insertion_point):
                    cur_node_copy = model.graph.create_node(
                        cur_node_orig.op,
                        cur_node_orig.target,
                        new_args,
                        new_kwargs,
                        # cur_node_orig.name,  # TODO(future PR): set name explicitly
                    )
                    if first_node_copy is None:
                        first_node_copy = cur_node_copy
                # since now only linear subgraphs are supported, all nodes
                # except the last one must have only one user
                if cur_node_orig != last_node:
                    if len(cur_node_orig.users.keys()) != 1:
                        raise AssertionError(
                            f"Expected exactly 1, but got {len(cur_node_orig.users)}"
                        )
                cur_node_orig = next(iter(cur_node_orig.users.keys()))
                if cur_node_orig.name.startswith(SHADOW_NODE_NAME_PREFIX):
                    raise AssertionError(
                        "cur_node_orig should not start with SHADOW_NODE_NAME_PREFIX"
                    )
                insertion_point = cur_node_copy

            # add a comparison logger after last_node's copy
            subgraph_candidate_idx = 1
            logger_mod_orig = _get_logger_for_subgraph(
                model,
                first_node,
                last_node,
                cur_subgraph_idx,
                subgraph_candidate_idx,
                qconfig_str,
                OutputComparisonLogger,
                fqn,
            )
            attr_name = _get_attr_name(cur_subgraph_idx, subgraph_candidate_idx)
            if hasattr(model, attr_name):
                raise AssertionError(
                    f"Unexpected attribute '{attr_name}' found in {model}"
                )
            setattr(model, attr_name, logger_mod_orig)
            with model.graph.inserting_after(insertion_point):
                logger = model.graph.call_module(
                    attr_name, args=(cur_node_copy, last_node), kwargs={}
                )

            # save the final node so we can use it in step 2
            orig_first_node_to_shadow_in_node[first_node] = first_node_copy
            orig_first_node_to_shadow_out_node[first_node] = cur_node_copy

        cur_subgraph_idx += 1

    model.recompile()

    # Now, we go from
    #
    #   x0 -> op0_0 -> x1_0 -> log -> x1 -> op1_0 -> ...
    #    \                     \       \
    #      -> op0_1 -> x1_1 -> clog      -> op1_1 -> ...
    #
    # to
    #
    #   x0 -> op0_0 -> x1_0 -> log --> x1_0 -> op1_0 -> ...
    #    \                     \
    #      -> op0_1 -> x1_1 -> clog -> x1_1 -> op1_1 -> ...
    #
    # sample values of key internal variables for the example above:
    #
    #   orig_first_node_to_shadow_in_node = {op0_0: op0_1, op1_0: op1_1}
    #   orig_first_node_to_shadow_out_node = {op0_0: op0_1, op1_0: op1_1}
    #
    # note: for subgraphs with more than one node, in_node will be different
    # compared to out_node

    nodes_to_skip = set()
    for n in orig_nodes:
        if n.op in ("placeholder", "get_attr", "output") or n in nodes_to_skip:
            continue

        maybe_subgraph = _get_subgraph_containing_node(n, subgraphs_dedup)
        if maybe_subgraph is not None:
            first_node, last_node = maybe_subgraph[0], maybe_subgraph[-1]
            nodes_to_skip.update(maybe_subgraph)
        else:
            first_node, last_node = n, n

        def maybe_remap_node_to_shadow(node):
            """
            If unshadowed `node` has a shadow version, return that. If not,
            return `node`.
            """
            if not isinstance(node, Node):
                # handle scalars
                return node

            if node.op in ("placeholder", "get_attr"):
                return node

            # Find the shadowed version of this arg from the previous
            # subgraph. For this, we need to:
            # 1. navigate to the first node of the previous subgraph
            # 2. get the output of the shadow wrapper which has (1) as an input

            # For now, assume the arg is in matched subgraphs. In the
            # future we may have to handle the case where this is not true.
            prev_subgraph = _get_subgraph_containing_node(node, subgraphs_dedup)
            if prev_subgraph is None:
                prev_subgraph = [node]
            prev_first_node = prev_subgraph[0]
            prev_shadow_output = orig_first_node_to_shadow_out_node[prev_first_node]
            return prev_shadow_output

        cur_shadow_input = orig_first_node_to_shadow_in_node[first_node]
        if cur_shadow_input is None:
            raise AssertionError("Expected cur_shadow_input to be non-None")
        cur_shadow_input.args = tree_map(
            maybe_remap_node_to_shadow, cur_shadow_input.args
        )
        cur_shadow_input.kwargs = tree_map(
            maybe_remap_node_to_shadow, cur_shadow_input.kwargs
        )

        model.recompile()


def _get_weight_info_from_shadow_wrapper(shadow_wrapper: torch.nn.Module):
    # input: shadow wrapper module
    # output if shadow wrapper module has a weighted op:
    #   (quantize_fn, (quantize_fn_args))
    # output if shadow wrapper module doesn't have a weighted op:
    #   None

    # For now, assume that the weight is the second input
    # to the shadow module. If that changes, we can fix it later.
    placeholders_seen = 0
    for shadow_n in shadow_wrapper.graph.nodes:  # type: ignore[union-attr]
        if shadow_n.op != "placeholder":
            continue

        placeholders_seen += 1
        if placeholders_seen != 2:
            continue

        # the subgraph looks like
        #
        #   _input_scale_1 = self._input_scale_1
        #   _input_zero_point_1 = self._input_zero_point_1
        #   quantize_per_channel = torch.quantize_per_channel(
        #       w2_0, _input_scale_1, _input_zero_point_1,
        #       0, torch.qint8)
        #
        #  we have `w2_0`, and are navigating this subgraph
        #  to get `_input_scale_1` and `_input_zero_point_1`

        if len(shadow_n.users) != 1:
            raise AssertionError(f"Expected exactly 1, got {len(shadow_n.users)}")
        quant_node = next(iter(shadow_n.users.keys()))
        new_args: Any = None
        if quant_node.target is torch.quantize_per_channel:
            _weight, scale_node, zp_node, axis, dtype = quant_node.args
            scale_val = getattr_from_fqn(shadow_wrapper, scale_node.target)
            zp_val = getattr_from_fqn(shadow_wrapper, zp_node.target)
            new_args = (scale_val, zp_val, axis, dtype)
        else:
            if quant_node.target != torch.quantize_per_tensor:
                raise AssertionError(
                    f"Expected torch.quantize_per_tensor, but got {quant_node.target}"
                )
            _weight, scale_node, zp_node, dtype = quant_node.args
            scale_val = getattr_from_fqn(shadow_wrapper, scale_node.target)
            zp_val = getattr_from_fqn(shadow_wrapper, zp_node.target)
            new_args = (scale_val, zp_val, dtype)
        return (quant_node.target, new_args)

    return None


def extract_weight_comparison(m: GraphModule) -> NSResultsType:
    # example graph:
    #
    #   w1 = self.w1
    #   b1 = self.b1
    #   linear = torch._C._nn.linear(x, w1, b1)
    #   shadow_0_0 = self.shadow_0_0(linear)
    #   shadow_wrapper_0_1 = self.shadow_wrapper_0_1(x, w1, b1)
    #   shadow_0_1 = self.shadow_0_1(shadow_wrapper_0_1, linear)
    #
    # algorithm:
    # 1. for each call_function node matching our allowlist:
    # 2.   if corresponding shadow wrapper exists, extract the weight pair
    #
    # Note: this is not super robust, but that's ok because this is
    # just for legacy customers who depend on the previous two-model version
    # of this API. TBD if we need to make this robust.
    # Note: modules are not supported, since existing customers only
    # use functions.

    # TODO(future PR): move this to config
    weighted_ops = {
        torch.nn.functional.linear,
    }

    results: NSResultsType = {"model": {NSSingleResultValuesType.WEIGHT.value: {}}}

    for n in m.graph.nodes:  # type: ignore[union-attr]
        if not (n.op == "call_function" and n.target in weighted_ops):
            continue

        # Check if we have a corresponding shadow wrapper
        # TODO(future PR, if needed): support kwargs
        # TODO(future PR, if needed): support multiple shadow users
        first_arg = n.args[0]
        shadow_wrapper_node = None
        for user in first_arg.users:
            # TODO(before land): fix string match
            if user.op == "call_module" and user.target.startswith("shadow_wrapper"):
                shadow_wrapper_node = user
                break

        if shadow_wrapper_node is None:
            continue

        shadow_wrapper = getattr_from_fqn(m, shadow_wrapper_node.target)  # type: ignore[arg-type]
        weight_info = _get_weight_info_from_shadow_wrapper(shadow_wrapper)
        if weight_info is None:
            continue

        # get weight
        w_node = n.args[1]
        w_obj = getattr_from_fqn(m, w_node.target).detach()

        # get a quantized version of weight
        quant_fn, quant_fn_args_except_first = weight_info
        new_args = (w_obj, *quant_fn_args_except_first)
        w_obj_q = quant_fn(*new_args)

        # add a comparison
        ref_node_name = n.name
        prev_node_name = n.name
        ref_node_type = get_target_type_str(n, m)
        prev_node_type = ref_node_type
        fqn = None
        if hasattr(m, "_node_name_to_scope"):
            fqn = m._node_name_to_scope[n.name][0]  # type: ignore[index]
        comparison = torch.ao.ns.fx.utils.compute_sqnr(w_obj, w_obj_q)
        result_fp32 = {
            "res_type": NSSingleResultValuesType.WEIGHT.value,
            "values": [w_obj],
            "prev_node_name": prev_node_name,
            "prev_node_target_type": prev_node_type,
            "ref_node_name": ref_node_name,
            "ref_node_target_type": ref_node_type,
            "index_within_arg": 0,
            "index_of_arg": 0,
            "fqn": fqn,
            "qconfig_str": "",
            "comparisons": [comparison],
            "comparison_fn_name": "sqnr",
        }
        result_q = {
            "res_type": NSSingleResultValuesType.WEIGHT.value,
            "values": [w_obj_q],
            "prev_node_name": prev_node_name,
            "prev_node_target_type": prev_node_type,
            "ref_node_name": ref_node_name,
            "ref_node_target_type": ref_node_type,
            "index_within_arg": 0,
            "index_of_arg": 0,
            "fqn": fqn,
            "qconfig_str": "",
            "comparisons": [comparison],
            "comparison_fn_name": "sqnr",
        }

        # go from subgraph_n_1 to subgraph_n_0
        _1, _2, node_idx, _3 = shadow_wrapper_node.target.split("_")
        name_fp32 = f"subgraph_{node_idx}_0"
        name_q = f"subgraph_{node_idx}_1"

        results["model"][NSSingleResultValuesType.WEIGHT.value][name_fp32] = [
            result_fp32
        ]
        results["model"][NSSingleResultValuesType.WEIGHT.value][name_q] = [result_q]

    return results


# TODO(future PR): redesign this to make it easier to consume outputs
def group_results_by_subgraph(results: NSResultsType) -> Any:
    """
    Creates a comparison of results

    Input:

    {
      'model': {
        'node_output': {
          'subgraph_0_0': [
            'values': [torch.tensor(...), ...], ...
            'ref_node_name': ...,
            'ref_node_target_type': ...,
            'qconfig_str': ...,
            'comparisons': [], ...
            'comparison_fn_name': '',
            'fqn': '...',
          ],
          'subgraph_0_1': [
            'values': [torch.tensor(...), ...], ...
            'ref_node_name': ...,
            'ref_node_target_type': ...,
            'qconfig_str': ...,
            'comparisons': [torch.tensor(...), ...], ...
            'comparison_fn_name': '...',
            'fqn': '...',
          ],
          ...
        },
      },
    }

    Output:
    {
      'subgraph_0': {
        '0': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': None,
          'comparisons': [torch.tensor(...), ...], ...
          'comparison_fn_name': '...',
          'fqn': '...',
        },
        '1': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '...',
          'comparisons': [torch.tensor(...), ...], ...
          'comparison_fn_name': '...',
          'fqn': '...',
        },
      },
    }

    """
    subgraph_name_to_subgraph_results: Any = collections.defaultdict(dict)

    # node_output or weight
    key_to_use = next(iter(results["model"].keys()))

    for subgraph_name_with_idx, subgraph_candidate_results in results["model"][
        key_to_use
    ].items():
        # convert from `subgraph_m_n` to `subgraph_m` and `n`
        (
            subgraph_str,
            subgraph_idx,
            subgraph_candidate_idx,
        ) = subgraph_name_with_idx.split("_")
        subgraph_name = f"{subgraph_str}_{subgraph_idx}"

        subgraph_results = {
            "ref_node_name": subgraph_candidate_results[0]["ref_node_name"],
            "ref_node_target_type": subgraph_candidate_results[0][
                "ref_node_target_type"
            ],
            "fqn": subgraph_candidate_results[0]["fqn"],
            "values": subgraph_candidate_results[0]["values"],
            "qconfig_str": subgraph_candidate_results[0]["qconfig_str"],
            "comparisons": subgraph_candidate_results[0]["comparisons"],
            "comparison_fn_name": subgraph_candidate_results[0]["comparison_fn_name"],
        }

        subgraph_name_to_subgraph_results[subgraph_name][subgraph_candidate_idx] = (
            subgraph_results
        )

    return dict(subgraph_name_to_subgraph_results)


# TODO(future PR): redesign this to make it easier to consume outputs
def create_results_comparison(
    results_grouped,
) -> Any:
    """
    Input:

    {
      'subgraph_0': {
        '0': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '',
          'comparisons': [],
          'comparison_fn_name': '',
          'fqn': '...',
        },
        '1': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '...',
          'comparisons': [torch.tensor(...), ...],
          'comparison_fn_name': 'sqnr',
          'fqn': '...',
        },
      },
    }

    Output:
    {
      'subgraph_0': {
        'ref_node_name': '...',
        'ref_node_target_type': '...',
        'fqn': '...',
        'candidates': {
          '1': {
            'qconfig_str': ...,
            'comparison_fn_name': 'sqnr',
            'cmp_raw': [..., ...],
            'cmp_mean': ...,
          },
          ...,
        },
      },
    }
    """

    results_comparison = {}

    for subgraph_name, subgraph_results in results_grouped.items():
        candidates = {}
        for subgraph_inner_name, subgraph_inner_result in subgraph_results.items():
            # skip comparing baseline to baseline
            if subgraph_inner_name == "0":
                continue

            # we expect the comparisons to be precalculated from
            # calibration, so we just fetch them here
            cmp_raw = subgraph_inner_result["comparisons"]
            cmp_raw_tensor = torch.stack(cmp_raw)

            candidates[subgraph_inner_name] = {
                "qconfig_str": subgraph_inner_result["qconfig_str"],
                "comparison_fn_name": subgraph_inner_result["comparison_fn_name"],
                "cmp_raw": cmp_raw_tensor,
                "cmp_mean": torch.mean(cmp_raw_tensor),
            }

        results_comparison[subgraph_name] = {
            "ref_node_name": subgraph_results["0"]["ref_node_name"],
            "ref_node_target_type": subgraph_results["0"]["ref_node_target_type"],
            "fqn": subgraph_results["0"]["fqn"],
            "candidates": candidates,
        }

    return results_comparison


# TODO(future PR): redesign this to make it easier to consume outputs
def print_n_shadows_summary(
    results_comparison,
) -> None:
    """
    Input:

    {
      'subgraph_0': {
        'ref_node_name': 'linear1',
        'ref_node_target_type': '...',
        'fqn': '...',
        'candidates': {
          '1': {
            'qconfig_str': ...,
            'comparison_fn_name': ...,
            'cmp_raw': [45.0, 55.0],
            'cmp_mean': 50.0,
          },
          ...,
        },
      },
    }

    Prints:

    node_name | node_type | fqn | 0    | 1    | ...
    linear1   | ...       | ... | 45.0 | 50.0 | ...
    """

    try:
        from tabulate import tabulate
    except ImportError:
        print(
            "`print_tabular` relies on the library `tabulate`, "
            "which could not be found on this machine. Run `pip "
            "install tabulate` to install the library."
        )
        return

    results = []
    for subgraph_data in results_comparison.values():
        mean_all_candidates = [
            candidate["cmp_mean"] for candidate in subgraph_data["candidates"].values()
        ]

        data_row = [
            subgraph_data["ref_node_name"],
            subgraph_data["ref_node_target_type"],
            subgraph_data["fqn"],
            *mean_all_candidates,
        ]
        results.append(data_row)

    max_candidate_idx_len = -1
    for data_row in results:
        max_candidate_idx_len = max(max_candidate_idx_len, len(data_row[1]))
    candidate_idx_headers = [str(x) for x in range(max_candidate_idx_len)]

    headers = ["node_name", "node_type", "fqn", *candidate_idx_headers]
    print(tabulate(results, headers=headers))
<|endoftext|>import enum
from collections.abc import Callable
from typing import Any, NamedTuple

from torch.fx.graph import Node


class NSSingleResultValuesType(str, enum.Enum):
    WEIGHT = "weight"
    NODE_OUTPUT = "node_output"
    NODE_INPUT = "node_input"


class NSSubgraph(NamedTuple):
    start_node: Node
    end_node: Node
    base_op_node: Node


# TODO(future PR): see if we can use typing_extensions's TypedDict instead
# to properly type the various keys
# {
#   # one of NSSingleResultValuesType
#   'type': 'weight',
#   # the values of type specified above
#   'values': [torch.tensor(...), ...],
#   # name of the node directly before the logger
#   'prev_node_name': 'linear1',
#   # type of the underlying function or module
#   'prev_node_target_type': torch.nn.functional.linear  # or torch.nn.Linear, etc
#   # name of the node responsible for adding this logger
#   # Note: this may differ from prev_node_name if we are logging inputs
#   'ref_node_name': 'linear1',
#   # index of this node within the arg of the input/output node
#   # for example, in cat([x1, x2, x3], dim=0), x2 would have index_within_arg == 1
#   'index_within_arg': 0,
#   # index of this node within the args of the input/output node
#   # for example, in add(x1, x2), x2 would have index_of_arg == 1
#   'index_of_arg': 0,
#   # precomputed comparisons of logger values to reference values
#   'comparisons': [torch.tensor(...), ...]
#   # name of function used for precomputed comparisons
#   'comparison_fn_name': 'sqnr',
#   # string representation of qconfig responsible for creating this logger
#   'qconfig_str': 'QConfig(...)',
# }
NSSingleResultType = dict[str, Any]

# {
#   'layer_name_1': {  # subgraph name
#     'node_output': {  # results type (node_output, node_input, weight)
#       'model_name_a':  # model name
#          [NSSingleResultType, ...],  # results, ordered by index_within_arg
#       'model_name_b':
#          [NSSingleResultType, ...],
#     },
#   },
# }
#
NSResultsType = dict[str, dict[str, dict[str, list[NSSingleResultType]]]]

# Defines the underlying target type of a node, for example:
# `F.conv1d` for a `call_function` conv node
# `nn.Conv1d` for a `call_module` node calling the forward of a `nn.Conv1d` module
# `'sigmoid'` for a `call_method` node calling `x.sigmoid()`
NSNodeTargetType = Callable | str
<|endoftext|># We are exposing all subpackages to the end-user.
# Because of possible inter-dependency, we want to avoid
# the cyclic imports, thus implementing lazy version
# as per https://peps.python.org/pep-0562/

from typing import TYPE_CHECKING as _TYPE_CHECKING


if _TYPE_CHECKING:
    from types import ModuleType

    from torch.ao.nn import (
        intrinsic as intrinsic,
        qat as qat,
        quantizable as quantizable,
        quantized as quantized,
        sparse as sparse,
    )


__all__ = [
    "intrinsic",
    "qat",
    "quantizable",
    "quantized",
    "sparse",
]


def __getattr__(name: str) -> "ModuleType":
    if name in __all__:
        import importlib

        return importlib.import_module("." + name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
<|endoftext|>from . import functional
from .modules import *  # noqa: F403
from .modules import MaxPool2d


__all__ = [
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DeQuantize",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "GroupNorm",
    "Hardswish",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "LSTM",
    "MultiheadAttention",
    "Quantize",
    "ReLU6",
    "Sigmoid",
    "Softmax",
    "Dropout",
    "PReLU",
    # Wrapper modules
    "FloatFunctional",
    "FXFloatFunctional",
    "QFunctional",
]
<|endoftext|># mypy: allow-untyped-defs
r"""Functional interface (quantized)."""

import warnings

import torch
from torch import Tensor
from torch.jit.annotations import BroadcastingList2
from torch.nn.modules.utils import _pair, _triple

from .modules.utils import _pair_from_first


# Although some of the functions and docstrings are mirrored from the torch.nn,
# we want to have them here for future changes.

__all__ = [
    "avg_pool2d",
    "avg_pool3d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool3d",
    "conv1d",
    "conv2d",
    "conv3d",
    "interpolate",
    "linear",
    "max_pool1d",
    "max_pool2d",
    "celu",
    "leaky_relu",
    "hardtanh",
    "hardswish",
    "threshold",
    "elu",
    "hardsigmoid",
    "clamp",
    "upsample",
    "upsample_bilinear",
    "upsample_nearest",
]


def avg_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    r"""
    Applies 2D average-pooling operation in :math:`kH \times kW` regions by step size
    :math:`sH \times sW` steps. The number of output features is equal to the number of
    input planes.

    .. note:: The input quantization parameters propagate to the output.

    See :class:`~torch.ao.nn.quantized.AvgPool2d` for details and output shape.

    Args:
        input: quantized input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a
          tuple `(kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
          tuple `(sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula
            to compute the output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation. Default: ``True``
        divisor_override: if specified, it will be used as divisor, otherwise
             size of the pooling region will be used. Default: None
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.avg_pool2d' must be quantized!")
    return torch.nn.functional.avg_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def avg_pool3d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    ceil_mode=False,
    count_include_pad=True,
    divisor_override=None,
):
    r"""
    Applies 3D average-pooling operation in :math:`kD \ times kH \times kW` regions by step size
    :math:`sD \times sH \times sW` steps. The number of output features is equal to the number of
    input planes.

    .. note:: The input quantization parameters propagate to the output.

    Args:
        input: quantized input tensor :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a
          tuple `(kD, kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
          tuple `(sD, sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple `(padD, padH, padW)`. Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula
            to compute the output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation. Default: ``True``
        divisor_override: if specified, it will be used as divisor, otherwise
             size of the pooling region will be used. Default: None
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.avg_pool3d' must be quantized!")
    return torch.nn.functional.avg_pool3d(
        input,
        kernel_size,
        stride,
        padding,
        ceil_mode,
        count_include_pad,
        divisor_override,
    )


def adaptive_avg_pool2d(input: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    r"""
    Applies a 2D adaptive average pooling over a quantized input signal composed
    of several quantized input planes.

    .. note:: The input quantization parameters propagate to the output.

    See :class:`~torch.ao.nn.quantized.AdaptiveAvgPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
                     double-integer tuple)
    """
    if not input.is_quantized:
        raise ValueError(
            "Input to 'quantized.functional.adaptive_avg_pool2d' must be quantized!"
        )
    return torch.nn.functional.adaptive_avg_pool2d(input, output_size)


def adaptive_avg_pool3d(input: Tensor, output_size: BroadcastingList2[int]) -> Tensor:
    r"""
    Applies a 3D adaptive average pooling over a quantized input signal composed
    of several quantized input planes.

    .. note:: The input quantization parameters propagate to the output.

    See :class:`~torch.ao.nn.quantized.AdaptiveAvgPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
                     double-integer tuple)
    """
    if not input.is_quantized:
        raise ValueError(
            "Input to 'quantized.functional.adaptive_avg_pool3d' must be quantized!"
        )
    return torch.nn.functional.adaptive_avg_pool3d(input, output_size)


def conv1d(
    input,
    weight,
    bias,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    scale=1.0,
    zero_point=0,
    dtype=torch.quint8,
):
    r"""
    Applies a 1D convolution over a quantized 1D input composed of several input
    planes.

    See :class:`~torch.ao.nn.quantized.Conv1d` for details and output shape.

    Args:
        input: quantized input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iW)`
        weight: quantized filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , iW)`
        bias: **non-quantized** bias tensor of shape :math:`(\text{out\_channels})`. The tensor type must be `torch.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sW,)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padW,)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dW,)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
          number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> warnings.filterwarnings("ignore", message=".*quantized tensor creation functions")  # docs: hide
        >>> from torch.ao.nn.quantized import functional as qF
        >>> filters = torch.randn(33, 16, 3, dtype=torch.float)
        >>> inputs = torch.randn(20, 16, 50, dtype=torch.float)
        >>> bias = torch.randn(33, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype_inputs = torch.quint8
        >>> dtype_filters = torch.qint8
        >>>
        >>> q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype_filters)
        >>> q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype_inputs)
        >>> qF.conv1d(q_inputs, q_filters, bias, padding=1, scale=scale, zero_point=zero_point)
    """
    if padding_mode != "zeros":
        raise NotImplementedError("Only zero-padding is supported!")
    if input.dtype != torch.quint8:
        raise NotImplementedError(
            "Only torch.quint8 is supported for activation tensor!"
        )
    if weight.dtype != torch.qint8:
        raise NotImplementedError("Only torch.qint8 is supported for weight tensor!")
    if input.ndim != 3:
        raise ValueError("Input shape must be `(N, C, L)`!")
    stride = _pair_from_first(stride)
    padding = _pair_from_first(padding)
    dilation = _pair_from_first(dilation)

    packed_params = torch.ops.quantized.conv1d_prepack(
        weight, bias, stride, padding, dilation, groups
    )
    return torch.ops.quantized.conv1d(input, packed_params, scale, zero_point)


def conv2d(
    input,
    weight,
    bias,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    scale=1.0,
    zero_point=0,
    dtype=torch.quint8,
):
    r"""
    Applies a 2D convolution over a quantized 2D input composed of several input
    planes.

    See :class:`~torch.ao.nn.quantized.Conv2d` for details and output shape.

    Args:
        input: quantized input tensor of shape :math:`(\text{minibatch} , \text{in\_channels} , iH , iW)`
        weight: quantized filters of shape :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW)`
        bias: **non-quantized** bias tensor of shape :math:`(\text{out\_channels})`. The tensor type must be `torch.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dH, dW)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be divisible by the
          number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> warnings.filterwarnings("ignore", message=".*quantized tensor creation functions")  # docs: hide
        >>> from torch.ao.nn.quantized import functional as qF
        >>> filters = torch.randn(8, 4, 3, 3, dtype=torch.float)
        >>> inputs = torch.randn(1, 4, 5, 5, dtype=torch.float)
        >>> bias = torch.randn(8, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype_inputs = torch.quint8
        >>> dtype_filters = torch.qint8
        >>>
        >>> q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype_filters)
        >>> q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype_inputs)
        >>> qF.conv2d(q_inputs, q_filters, bias, padding=1, scale=scale, zero_point=zero_point)
    """
    if padding_mode != "zeros":
        raise NotImplementedError("Only zero-padding is supported!")
    if input.dtype != torch.quint8:
        raise NotImplementedError(
            "Only torch.quint8 is supported for activation tensor!"
        )
    if weight.dtype != torch.qint8:
        raise NotImplementedError("Only torch.qint8 is supported for weight tensor!")
    if input.ndim != 4:
        raise ValueError("Input shape must be `(N, C, H, W)`!")
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    packed_params = torch.ops.quantized.conv2d_prepack(
        weight, bias, stride, padding, dilation, groups
    )
    return torch.ops.quantized.conv2d(input, packed_params, scale, zero_point)


def conv3d(
    input,
    weight,
    bias,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    padding_mode="zeros",
    scale=1.0,
    zero_point=0,
    dtype=torch.quint8,
):
    r"""
    Applies a 3D convolution over a quantized 3D input composed of several input
    planes.

    See :class:`~torch.ao.nn.quantized.Conv3d` for details and output shape.

    Args:
        input: quantized input tensor of shape
          :math:`(\text{minibatch} , \text{in\_channels} , iD , iH , iW)`
        weight: quantized filters of shape
          :math:`(\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kD , kH , kW)`
        bias: **non-quantized** bias tensor of shape
          :math:`(\text{out\_channels})`. The tensor type must be `torch.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sD, sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padD, padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dD, dH, dW)`. Default: 1
        groups: split input into groups, :math:`\text{in\_channels}` should be
          divisible by the number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for
          quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_QENGINE)
        >>> warnings.filterwarnings("ignore", message=".*quantized tensor creation functions")  # docs: hide
        >>> from torch.ao.nn.quantized import functional as qF
        >>> filters = torch.randn(8, 4, 3, 3, 3, dtype=torch.float)
        >>> inputs = torch.randn(1, 4, 5, 5, 5, dtype=torch.float)
        >>> bias = torch.randn(8, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype_inputs = torch.quint8
        >>> dtype_filters = torch.qint8
        >>>
        >>> q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype_filters)
        >>> q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype_inputs)
        >>> qF.conv3d(q_inputs, q_filters, bias, padding=1, scale=scale, zero_point=zero_point)
    """
    if padding_mode != "zeros":
        raise NotImplementedError("Only zero-padding is supported!")
    if input.dtype != torch.quint8:
        raise NotImplementedError(
            "Only torch.quint8 is supported for activation tensor!"
        )
    if weight.dtype != torch.qint8:
        raise NotImplementedError("Only torch.qint8 is supported for weight tensor!")
    if input.ndim != 5:
        raise ValueError("Input shape must be `(N, C, D, H, W)`!")
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)

    packed_params = torch.ops.quantized.conv3d_prepack(
        weight, bias, stride, padding, dilation, groups
    )
    return torch.ops.quantized.conv3d(input, packed_params, scale, zero_point)


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    r"""Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    See :func:`torch.nn.functional.interpolate` for implementation details.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D/3D input is supported for quantized inputs

    .. note:: Only the following modes are supported for the quantized inputs:

        - `bilinear`
        - `nearest`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'bilinear'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'bilinear'``.
            Default: ``False``
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.interpolate' must be quantized!")
    return torch.nn.functional.interpolate(
        input, size, scale_factor, mode, align_corners
    )


def linear(
    input: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    scale: float | None = None,
    zero_point: int | None = None,
) -> Tensor:
    r"""
    Applies a linear transformation to the incoming quantized data:
    :math:`y = xA^T + b`.
    See :class:`~torch.ao.nn.quantized.Linear`

    .. note::

      Current implementation packs weights on every call, which has penalty on performance.
      If you want to avoid the overhead, use :class:`~torch.ao.nn.quantized.Linear`.

    Args:
      input (Tensor): Quantized input of type `torch.quint8`
      weight (Tensor): Quantized weight of type `torch.qint8`
      bias (Tensor): None or fp32 bias of type `torch.float`
      scale (double): output scale. If None, derived from the input scale
      zero_point (long): output zero point. If None, derived from the input zero_point

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if scale is None:
        scale = input.q_scale()
    if zero_point is None:
        zero_point = input.q_zero_point()
    _packed_params = torch.ops.quantized.linear_prepack(weight, bias)
    return torch.ops.quantized.linear(input, _packed_params, scale, zero_point)


def max_pool1d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    r"""Applies a 1D max pooling over a quantized input signal composed of
    several quantized input planes.

    .. note:: The input quantization parameters are propagated to the output.

    See :class:`~torch.ao.nn.quantized.MaxPool1d` for details.
    """
    if return_indices:
        raise NotImplementedError("return_indices is not yet implemented!")
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch.nn.functional.max_pool1d(
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )


def max_pool2d(
    input,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    r"""Applies a 2D max pooling over a quantized input signal composed of
    several quantized input planes.

    .. note:: The input quantization parameters are propagated to the output.

    See :class:`~torch.ao.nn.quantized.MaxPool2d` for details.
    """
    if return_indices:
        raise NotImplementedError("return_indices is not yet implemented!")
    if stride is None:
        stride = torch.jit.annotate(list[int], [])
    return torch.nn.functional.max_pool2d(
        input,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )


def celu(input: Tensor, scale: float, zero_point: int, alpha: float = 1.0) -> Tensor:
    r"""celu(input, scale, zero_point, alpha=1.) -> Tensor

    Applies the quantized CELU function element-wise.

    .. math::
        \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x / \alpha) - 1))

    Args:
        input: quantized input
        alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.celu' must be quantized!")
    return torch.ops.quantized.celu(input, scale, zero_point, alpha)


def leaky_relu(
    input: Tensor,
    negative_slope: float = 0.01,
    inplace: bool = False,
    scale: float | None = None,
    zero_point: int | None = None,
):
    r"""
    Quantized version of the.
    leaky_relu(input, negative_slope=0.01, inplace=False, scale, zero_point) -> Tensor

    Applies element-wise,
    :math:`\text{LeakyReLU}(x) = \max(0, x) + \text{negative\_slope} * \min(0, x)`

    Args:
        input: Quantized input
        negative_slope: The slope of the negative input
        inplace: Inplace modification of the input tensor
        scale, zero_point: Scale and zero point of the output tensor.

    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    if scale is not None and zero_point is not None:
        if inplace:
            raise AssertionError("Cannot rescale with `inplace`")
        output = torch._empty_affine_quantized(
            input.shape, scale=scale, zero_point=int(zero_point), dtype=input.dtype
        )
        torch._C._nn.leaky_relu(input, negative_slope, out=output)
        return output
    if inplace:
        result = torch._C._nn.leaky_relu_(input, negative_slope)
    else:
        result = torch._C._nn.leaky_relu(input, negative_slope)
    return result


def hardtanh(
    input: Tensor, min_val: float = -1.0, max_val: float = 1.0, inplace: bool = False
) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.hardtanh`."""
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardtanh' must be quantized!")
    if inplace:
        return torch._C._nn.hardtanh_(input, min_val, max_val)
    return torch._C._nn.hardtanh(input, min_val, max_val)


def hardswish(input: Tensor, scale: float, zero_point: int) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.hardswish`.

    Args:
        input: quantized input
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardswish' must be quantized!")
    return torch._ops.ops.quantized.hardswish(input, scale, zero_point)


def threshold(input: Tensor, threshold: float, value: float) -> Tensor:
    r"""Applies the quantized version of the threshold function element-wise:

    .. math::
        x = \begin{cases}
                x & \text{if~} x > \text{threshold} \\
                \text{value} & \text{otherwise}
            \end{cases}

    See :class:`~torch.nn.Threshold` for more details.
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.threshold' must be quantized!")
    if threshold is None:
        raise ValueError("Input to 'threshold' must be specified!")
    if value is None:
        raise ValueError("Input to 'value' must be specified!")
    return torch._ops.ops.quantized.threshold(input, threshold, value)


def elu(input: Tensor, scale: float, zero_point: int, alpha: float = 1.0) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.elu`.

    Args:
        input: quantized input
        scale: quantization scale of the output tensor
        zero_point: quantization zero point of the output tensor
        alpha: the alpha constant
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.elu' must be quantized!")
    return torch.ops.quantized.elu(input, scale, zero_point, alpha)


def hardsigmoid(input: Tensor, inplace: bool = False) -> Tensor:
    r"""This is the quantized version of :func:`~torch.nn.functional.hardsigmoid`."""
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.hardsigmoid' must be quantized!")
    if inplace:
        return torch._C._nn.hardsigmoid_(input)  # type: ignore[attr-defined]
    return torch._C._nn.hardsigmoid(input)


def clamp(input: Tensor, min_: float, max_: float) -> Tensor:
    r"""float(input, min\_, max\_) -> Tensor

    Applies the clamp function element-wise.
    See :class:`~torch.ao.nn.quantized.clamp` for more details.

    Args:
        input: quantized input
        min_: minimum value for clamping
        max_: maximum value for clamping
    """
    if not input.is_quantized:
        raise ValueError("Input to 'quantized.clamp' must be quantized!")
    return torch.clamp(input, min_, max_)


def upsample(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    r"""Upsamples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    .. warning::
        This function is deprecated in favor of
        :func:`torch.ao.nn.quantized.functional.interpolate`.
        This is equivalent with ``nn.quantized.functional.interpolate(...)``.

    See :func:`torch.nn.functional.interpolate` for implementation details.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D input is supported for quantized inputs

    .. note:: Only the following modes are supported for the quantized inputs:

        - `bilinear`
        - `nearest`

    Args:
        input (Tensor): quantized input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to be an integer.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'bilinear'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'bilinear'``.
            Default: ``False``

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`bilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.
    """
    warnings.warn(
        "nn.quantized.functional.upsample is deprecated. Use nn.quantized.functional.interpolate instead.",
        stacklevel=2,
    )
    return interpolate(input, size, scale_factor, mode, align_corners)


def upsample_bilinear(input, size=None, scale_factor=None):
    r"""Upsamples the input, using bilinear upsampling.

    .. warning::
        This function is deprecated in favor of
        :func:`torch.ao.nn.quantized.functional.interpolate`.
        This is equivalent with
        ``nn.quantized.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D inputs are supported

    Args:
        input (Tensor): quantized input
        size (int or Tuple[int, int]): output spatial size.
        scale_factor (int or Tuple[int, int]): multiplier for spatial size
    """
    # DeprecationWarning is ignored by default
    warnings.warn(
        "nn.quantized.functional.upsample_bilinear is deprecated. Use nn.quantized.functional.interpolate instead.",
        stacklevel=2,
    )
    return interpolate(input, size, scale_factor, mode="bilinear", align_corners=True)


def upsample_nearest(input, size=None, scale_factor=None):
    r"""Upsamples the input, using nearest neighbours' pixel values.

    .. warning::
        This function is deprecated in favor of
        :func:`torch.ao.nn.quantized.functional.interpolate`.
        This is equivalent with ``nn.quantized.functional.interpolate(..., mode='nearest')``.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D inputs are supported

    Args:
        input (Tensor): quantized input
        size (int or Tuple[int, int] or Tuple[int, int, int]): output spatial
            size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
    """
    # DeprecationWarning is ignored by default
    warnings.warn(
        "nn.quantized.functional.upsample_nearest is deprecated. Use nn.quantized.functional.interpolate instead.",
        stacklevel=2,
    )
    return interpolate(input, size, scale_factor, mode="nearest")
<|endoftext|># mypy: allow-untyped-defs
import torch


__all__ = [
    "LayerNorm",
    "GroupNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
]


class LayerNorm(torch.nn.LayerNorm):
    r"""This is the quantized version of :class:`~torch.nn.LayerNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        normalized_shape,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            **factory_kwargs,
        )
        self.weight = weight
        self.bias = bias
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.layer_norm(
            input,
            self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
            output_scale=self.scale,
            output_zero_point=self.zero_point,
        )

    def _get_name(self):
        return "QuantizedLayerNorm"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.normalized_shape,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.elementwise_affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(
            mod.normalized_shape,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.elementwise_affine,
        )


class GroupNorm(torch.nn.GroupNorm):
    r"""This is the quantized version of :class:`~torch.nn.GroupNorm`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    __constants__ = ["num_groups", "num_channels", "eps", "affine"]

    def __init__(
        self,
        num_groups,
        num_channels,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(num_groups, num_channels, eps, affine, **factory_kwargs)
        self.weight = weight
        self.bias = bias
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.group_norm(
            input,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps,
            self.scale,
            self.zero_point,
        )

    def _get_name(self):
        return "QuantizedGroupNorm"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_groups,
            mod.num_channels,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod


class InstanceNorm1d(torch.nn.InstanceNorm1d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm1d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.weight = weight
        self.bias = bias
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedInstanceNorm1d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )


class InstanceNorm2d(torch.nn.InstanceNorm2d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm2d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.weight = weight
        self.bias = bias
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedInstanceNorm2d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )


class InstanceNorm3d(torch.nn.InstanceNorm3d):
    r"""This is the quantized version of :class:`~torch.nn.InstanceNorm3d`.

    Additional args:
        * **scale** - quantization scale of the output, type: double.
        * **zero_point** - quantization zero point of the output, type: long.

    """

    def __init__(
        self,
        num_features,
        weight,
        bias,
        scale,
        zero_point,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.weight = weight
        self.bias = bias
        self.register_buffer("scale", torch.tensor(scale, **factory_kwargs))
        self.register_buffer("zero_point", torch.tensor(zero_point, **factory_kwargs))

    def forward(self, input):
        return torch.ops.quantized.instance_norm(
            input, self.weight, self.bias, self.eps, self.scale, self.zero_point
        )

    def _get_name(self):
        return "QuantizedInstanceNorm3d"

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        scale, zero_point = mod.activation_post_process.calculate_qparams()
        new_mod = cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
        return new_mod

    @classmethod
    def from_reference(cls, mod, scale, zero_point):
        return cls(
            mod.num_features,
            mod.weight,
            mod.bias,
            float(scale),
            int(zero_point),
            mod.eps,
            mod.affine,
        )
<|endoftext|># mypy: allow-untyped-defs

import torch
from torch import Tensor
from torch._ops import ops


__all__ = ["FloatFunctional", "FXFloatFunctional", "QFunctional"]


class FloatFunctional(torch.nn.Module):
    r"""State collector class for float operations.

    The instance of this class can be used instead of the ``torch.`` prefix for
    some operations. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

        >>> f_add = FloatFunctional()
        >>> a = torch.tensor(3.0)
        >>> b = torch.tensor(4.0)
        >>> f_add.add(a, b)  # Equivalent to ``torch.add(a, b)``

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def __init__(self) -> None:
        super().__init__()
        self.activation_post_process = torch.nn.Identity()

    def forward(self, x):
        raise RuntimeError(
            "FloatFunctional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

    r"""Operation equivalent to ``torch.add(Tensor, Tensor)``"""

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.add(Tensor, float)``"""

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.add(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, Tensor)``"""

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.mul(x, y)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, float)``"""

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.mul(x, y)
        # Note: this operation is not observed because the observation is not
        # needed for the quantized op.
        return r

    r"""Operation equivalent to ``torch.cat``"""

    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor:
        r = torch.cat(x, dim=dim)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``relu(torch.add(x,y))``"""

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        r = self.activation_post_process(r)
        return r

    r"""Operation equivalent to ``torch.matmul(Tensor, Tensor)``"""

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.matmul(x, y)
        r = self.activation_post_process(r)
        return r


class FXFloatFunctional(torch.nn.Module):
    r"""module to replace FloatFunctional module before FX graph mode quantization,
    since activation_post_process will be inserted in top level module directly

    Valid operation names:
        - add
        - cat
        - mul
        - add_relu
        - add_scalar
        - mul_scalar
    """

    def forward(self, x):
        raise RuntimeError(
            "FloatFunctional is not intended to use the "
            + "'forward'. Please use the underlying operation"
        )

    r"""Operation equivalent to ``torch.add(Tensor, Tensor)``"""

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        return r

    r"""Operation equivalent to ``torch.add(Tensor, float)``"""

    def add_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.add(x, y)
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, Tensor)``"""

    def mul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.mul(x, y)
        return r

    r"""Operation equivalent to ``torch.mul(Tensor, float)``"""

    def mul_scalar(self, x: Tensor, y: float) -> Tensor:
        r = torch.mul(x, y)
        return r

    r"""Operation equivalent to ``torch.cat``"""

    def cat(self, x: list[Tensor], dim: int = 0) -> Tensor:
        r = torch.cat(x, dim=dim)
        return r

    r"""Operation equivalent to ``relu(torch.add(x,y))``"""

    def add_relu(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.add(x, y)
        r = torch.nn.functional.relu(r)
        return r

    r"""Operation equivalent to ``torch.matmul(Tensor, Tensor)``"""

    def matmul(self, x: Tensor, y: Tensor) -> Tensor:
        r = torch.matmul(x, y)
        return r


class QFunctional(torch.nn.Module):
    r"""Wrapper class for quantized operations.

    The instance of this class can be used instead of the
    ``torch.ops.quantized`` prefix. See example usage below.

    .. note::

        This class does not provide a ``forward`` hook. Instead, you must use
        one of the underlying functions (e.g. ``add``).

    Examples::

