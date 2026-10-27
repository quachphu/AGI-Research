    "NON_OBSERVABLE_ARG_DICT",
    "NON_QUANTIZABLE_WEIGHT_OPS",
    "return_arg_list",
    "ObservedGraphModuleAttrs",
]

NON_QUANTIZABLE_WEIGHT_OPS = {
    torch.nn.functional.layer_norm,
    torch.nn.functional.group_norm,
    torch.nn.functional.instance_norm,
}


@dataclass
class ObservedGraphModuleAttrs:
    node_name_to_qconfig: dict[str, QConfigAny]
    node_name_to_scope: dict[str, tuple[str, type]]
    prepare_custom_config: PrepareCustomConfig
    equalization_node_name_to_qconfig: dict[str, Any]
    qconfig_mapping: QConfigMapping
    is_qat: bool
    observed_node_names: set[str]
    is_observed_standalone_module: bool = False
    standalone_module_input_quantized_idxs: list[int] | None = None
    standalone_module_output_quantized_idxs: list[int] | None = None


def node_arg_is_weight(node: Node, arg: Any) -> bool:
    """Returns if node arg is weight"""
    weight_index = None
    if "target_dtype_info" in node.meta:
        weight_index = node.meta["target_dtype_info"].get("weight_index", None)
    if (
        weight_index is not None
        and weight_index < len(node.args)
        and node.args[weight_index] is arg
    ):
        return True
    return node.kwargs.get("weight") is arg


def node_arg_is_bias(node: Node, arg: Any) -> bool:
    """Returns if node arg is bias"""
    bias_index = None
    if "target_dtype_info" in node.meta:
        bias_index = node.meta["target_dtype_info"].get("bias_index", None)
    if (
        bias_index is not None
        and bias_index < len(node.args)
        and node.args[bias_index] is arg
    ):
        return True
    return node.kwargs.get("bias") is arg


def get_custom_module_class_keys(
    custom_module_mapping: dict[QuantType, dict[type, type]],
) -> list[Any]:
    r"""Get all the unique custom module keys in the custom config dict.

    Example input::

        {
            QuantType.STATIC: {CustomModule1: ObservedCustomModule},
            QuantType.DYNAMIC: {CustomModule2: DynamicObservedCustomModule},
            QuantType.WEIGHT_ONLY: {CustomModule3: WeightOnlyObservedCustomModule},
        }

    Example output::

        # extract the keys across all inner STATIC, DYNAMIC, and WEIGHT_ONLY dicts
        [CustomModule1, CustomModule2, CustomModule3]
    """
    # using set to dedup
    float_custom_module_classes: set[Any] = set()
    for quant_mode in [QuantType.STATIC, QuantType.DYNAMIC, QuantType.WEIGHT_ONLY]:
        quant_mode_custom_module_config = custom_module_mapping.get(quant_mode, {})
        quant_mode_custom_module_classes = set(quant_mode_custom_module_config.keys())
        float_custom_module_classes |= quant_mode_custom_module_classes
    return list(float_custom_module_classes)


def get_linear_prepack_op_for_dtype(dtype):
    if dtype == torch.float16:
        return torch.ops.quantized.linear_prepack_fp16
    elif dtype == torch.qint8:
        return torch.ops.quantized.linear_prepack
    else:
        raise Exception("can't get linear prepack op for dtype:", dtype)  # noqa: TRY002


def get_qconv_prepack_op(conv_op: Callable) -> Callable:
    prepack_ops = {
        torch.nn.functional.conv1d: torch.ops.quantized.conv1d_prepack,
        torch.nn.functional.conv2d: torch.ops.quantized.conv2d_prepack,
        torch.nn.functional.conv3d: torch.ops.quantized.conv3d_prepack,
        torch.nn.functional.conv_transpose1d: torch.ops.quantized.conv_transpose1d_prepack,
        torch.nn.functional.conv_transpose2d: torch.ops.quantized.conv_transpose2d_prepack,
        torch.nn.functional.conv_transpose3d: torch.ops.quantized.conv_transpose3d_prepack,
    }
    prepack_op = prepack_ops.get(conv_op)
    if prepack_op is None:
        raise AssertionError(f"Didn't find prepack op for {conv_op}")
    return prepack_op


# Returns a function that can get a new attribute name for module with given
# prefix, for example,
# >> get_new_observer_name = get_new_attr_name_with_prefix('_observer')
# >> new_name = get_new_observer_name(module)
# new_name will be an unused attribute name on module, e.g. `_observer_1`
def get_new_attr_name_with_prefix(prefix: str) -> Callable:
    prefix = prefix.replace(".", "_")

    def get_new_attr_name(module: torch.nn.Module):
        def get_attr_name(i: int):
            return prefix + str(i)

        i = 0
        attr_name = get_attr_name(i)
        while hasattr(module, attr_name):
            i += 1
            attr_name = get_attr_name(i)
        return attr_name

    return get_new_attr_name


def collect_producer_nodes(node: Node) -> list[Node] | None:
    r"""Starting from a target node, trace back until we hit input or
    getattr node. This is used to extract the chain of operators
    starting from getattr to the target node, for example::

        def forward(self, x):
            observed = self.observer(self.weight)
            return F.linear(x, observed)

    collect_producer_nodes(observed) will either return a list of nodes that
    produces the observed node or None if we can't extract a self contained
    graph without free variables(inputs of the forward function).
    """
    nodes = [node]
    frontier = [node]
    while frontier:
        node = frontier.pop()
        all_args = list(node.args) + list(node.kwargs.values())
        for arg in all_args:
            if not isinstance(arg, Node):
                continue
            if arg.op == "placeholder":
                # hit input, can't fold in this case
                return None
            nodes.append(arg)
            if not (arg.op == "call_function" and arg.target is getattr):
                frontier.append(arg)
    return nodes


def graph_module_from_producer_nodes(
    root: GraphModule, producer_nodes: list[Node]
) -> GraphModule:
    r"""Construct a graph module from extracted producer nodes
    from `collect_producer_nodes` function
    Args:
      root: the root module for the original graph
      producer_nodes: a list of nodes we use to construct the graph
    Return:
      A graph module constructed from the producer nodes
    """
    if len(producer_nodes) == 0:
        raise AssertionError("list of producer nodes can not be empty")
    # since we traced back from node to getattr
    producer_nodes.reverse()
    graph = Graph()
    env: dict[Any, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node])

    for producer_node in producer_nodes:
        env[producer_node] = graph.node_copy(producer_node, load_arg)
    graph.output(load_arg(producer_nodes[-1]))
    graph_module = GraphModule(root, graph)
    return graph_module


# TODO: delete
@functools.cache
def assert_and_get_unique_device(module: torch.nn.Module) -> Any:
    """
    Returns the unique device for a module, or None if no device is found.
    Throws an error if multiple devices are detected.
    """
    return _assert_and_get_unique_device(module)


def create_getattr_from_value(
    module: torch.nn.Module,
    graph: Graph,
    prefix: str,
    value: Any,
    device: torch.device | None = None,
) -> Node:
    """
    Given a value of any type, creates a getattr node corresponding to the value and
    registers the value as a buffer to the module.
    """
    get_new_attr_name = get_new_attr_name_with_prefix(prefix)
    attr_name = get_new_attr_name(module)
    if device is None:
        device = assert_and_get_unique_device(module)
    new_value = (
        value.detach().clone()
        if isinstance(value, torch.Tensor)
        else torch.tensor(value, device=device)
    )
    module.register_buffer(attr_name, new_value)
    # Create get_attr with value
    attr_node = graph.create_node("get_attr", attr_name)
    return attr_node


def all_node_args_have_no_tensors(
    node: Node, modules: dict[str, torch.nn.Module], cache: dict[Node, bool]
) -> bool:
    """
    If we know for sure that all of this node's args have no
    tensors (are primitives), return True.  If we either
    find a tensor or are not sure, return False. Note: this
    function is not exact.
    """
    if cache and node in cache:
        return cache[node]

    result = False  # will be overwritten
    if not isinstance(node, Node):
        result = True
    elif node.op == "placeholder":
        result = False
    elif node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError("node.target must be a string for call_module nodes")
        if _is_activation_post_process(modules[node.target]):
            result = all_node_args_have_no_tensors(node.args[0], modules, cache)  # type: ignore[arg-type]
    elif node.op == "call_module":
        result = False
    elif node.op == "call_function" and node.target is operator.getitem:
        result = all_node_args_have_no_tensors(node.args[0], modules, cache)  # type: ignore[arg-type]
    elif node.op == "get_attr":
        result = False
    elif node.target is getattr and node.args[1] in ["ndim", "shape"]:
        # x1 = x0.ndim
        result = True
    elif node.op == "call_method" and node.target == "size":
        # x1 = x0.size(0)
        result = True
    else:
        found_one_tensor = False
        for arg in node.args:
            if isinstance(arg, list):
                for list_el in arg:
                    if isinstance(list_el, Node):
                        this_list_el_args_have_no_tensors = (
                            all_node_args_have_no_tensors(list_el, modules, cache)
                        )
                        found_one_tensor = found_one_tensor or (
                            not this_list_el_args_have_no_tensors
                        )
                        # If found_one_tensor is True, there is no point in
                        # recursing further as the end result will always
                        # be True.
                        # TODO(future PR): remove this entire function  and
                        # change to dtype inference without recursion.
                        if found_one_tensor:
                            result = not found_one_tensor
                            if cache:
                                cache[node] = result
                            return result
            elif isinstance(arg, int):
                pass
            else:
                if isinstance(arg, Node):
                    this_arg_args_have_no_tensors = all_node_args_have_no_tensors(
                        arg, modules, cache
                    )
                    found_one_tensor = found_one_tensor or (
                        not this_arg_args_have_no_tensors
                    )
                    # If found_one_tensor is True, there is no point in
                    # recursing further as the end result will always
                    # be True.
                    # TODO(future PR): remove this entire function  and
                    # change to dtype inference without recursion.
                    if found_one_tensor:
                        result = not found_one_tensor
                        if cache:
                            cache[node] = result
                        return result
                else:
                    found_one_tensor = True
            result = not found_one_tensor
    if cache:
        cache[node] = result
    return result


def all_node_args_except_first(node: Node) -> list[int]:
    """
    Returns all node arg indices after first
    """
    return list(range(1, len(node.args)))


def return_arg_list(arg_indices: list[int]) -> Callable[[Node], list[int]]:
    """
    Constructs a function that takes a node as arg and returns the arg_indices
    that are valid for node.args
    """

    def arg_indices_func(node: Node) -> list[int]:
        return [i for i in arg_indices if i < len(node.args)]

    return arg_indices_func


NodeInfo = namedtuple("NodeInfo", "op target")

# this dict identifies which indices of a node are non tensors
# so that they can be propagated correctly since inserting observers
# for them would cause errors

NON_OBSERVABLE_ARG_DICT: dict[
    NodeInfo, dict[type | torch.dtype, Callable[[Node], list[int]]]
] = {
    NodeInfo("call_method", "masked_fill"): {
        torch.bool: return_arg_list([1]),
        float: return_arg_list([2]),
    },
    NodeInfo("call_method", "permute"): {int: all_node_args_except_first},
    NodeInfo("call_method", "repeat"): {int: all_node_args_except_first},
    NodeInfo("call_method", "reshape"): {int: all_node_args_except_first},
    NodeInfo("call_method", "size"): {int: return_arg_list([1])},
    NodeInfo("call_method", "transpose"): {int: all_node_args_except_first},
    NodeInfo("call_method", torch.transpose): {int: all_node_args_except_first},
    NodeInfo("call_method", "unsqueeze"): {int: return_arg_list([1])},
    NodeInfo("call_method", "unsqueeze_"): {int: return_arg_list([1])},
    NodeInfo("call_method", torch.unsqueeze): {int: return_arg_list([1])},
    NodeInfo("call_method", "view"): {int: all_node_args_except_first},
}

EMPTY_ARG_DICT: dict[type | torch.dtype, Callable[[Node], list[int]]] = {}


def get_non_observable_arg_indexes_and_types(
    node: Node,
) -> dict[type | torch.dtype, Callable[[Node], list[int]]]:
    """
    Returns a dict with of non float tensor types as keys and values which correspond to a
    function to retrieve the list (which takes the node as an argument)
    """
    info = NodeInfo(node.op, node.target)

    return NON_OBSERVABLE_ARG_DICT.get(info, EMPTY_ARG_DICT)


def maybe_get_next_module(
    node: Node,
    modules: dict[str, nn.Module],
    target_module_type: type[nn.Module] | None = None,
    target_functional_type: Any = None,
) -> Node | None:
    """Gets the next module that matches what is needed in
    is_target_module_type if it exists

    Args:
        node: The node whose users we want to look at
        target_module_type: Module type that we want to check
        target_functional_type: Functional type that we want to check
    """

    for user in node.users:
        if (
            user.op == "call_module"
            and target_module_type is not None
            and isinstance(modules[str(user.target)], target_module_type)
        ):
            return user
        elif (
            user.op == "call_function"
            and target_functional_type is not None
            and user.target == target_functional_type
        ):
            return user

    return None


def create_node_from_old_node_preserve_meta(
    quantized_graph: Graph,
    create_node_args: tuple[Any, ...],
    old_node: Node,
) -> Node:
    """
    Creates `new_node` and copies the necessary metadata to it from `old_node`.
    """
    new_node = quantized_graph.create_node(*create_node_args)
    new_node.stack_trace = old_node.stack_trace
    return new_node


def get_skipped_module_name_and_classes(
    prepare_custom_config: PrepareCustomConfig, is_standalone_module: bool
) -> tuple[list[str], list[type[Any]]]:
    skipped_module_names = copy.copy(prepare_custom_config.non_traceable_module_names)
    skipped_module_classes = copy.copy(
        prepare_custom_config.non_traceable_module_classes
    )
    if not is_standalone_module:
        # standalone module and custom module config are applied in top level module
        skipped_module_names += list(
            prepare_custom_config.standalone_module_names.keys()
        )
        skipped_module_classes += list(
            prepare_custom_config.standalone_module_classes.keys()
        )
        skipped_module_classes += get_custom_module_class_keys(
            prepare_custom_config.float_to_observed_mapping
        )

    return skipped_module_names, skipped_module_classes


def _is_custom_module_lstm(
    node: Node,
    named_modules: dict[str, torch.nn.Module],
    qconfig: QConfigAny = None,
    # QuantizeHandler, but we cannot include the type here due to circular imports
    qhandler: Any | None = None,
) -> bool:
    """
    Return whether this refers to the custom module LSTM flow.
    """
    mod = _get_module(node, named_modules)
    if qconfig is not None and qhandler is not None:
        if not isinstance(
            qhandler, torch.ao.quantization.fx.quantize_handler.QuantizeHandler
        ):  # type: ignore[attr-defined]
            raise AssertionError("qhandler must be a QuantizeHandler when provided")
        return (
            isinstance(mod, torch.nn.LSTM)
            and activation_is_statically_quantized(qconfig)
            and qhandler.is_custom_module()
        )
    else:
        return isinstance(mod, torch.ao.nn.quantizable.LSTM)


def _is_custom_module_mha(
    node: Node,
    named_modules: dict[str, torch.nn.Module],
    qconfig: QConfigAny = None,
    # QuantizeHandler, but we cannot include the type here due to circular imports
    qhandler: Any | None = None,
) -> bool:
    """
    Return whether this refers to the custom module MultiheadAttention flow.
    """
    mod = _get_module(node, named_modules)
    if qconfig is not None and qhandler is not None:
        if not isinstance(
            qhandler, torch.ao.quantization.fx.quantize_handler.QuantizeHandler
        ):  # type: ignore[attr-defined]
            raise AssertionError("qhandler must be a QuantizeHandler when provided")
        return (
            isinstance(mod, torch.nn.MultiheadAttention)
            and activation_is_statically_quantized(qconfig)
            and qhandler.is_custom_module()
        )
    else:
        return isinstance(mod, torch.ao.nn.quantizable.MultiheadAttention)


def _get_module(
    node: Node, named_modules: dict[str, torch.nn.Module]
) -> torch.nn.Module | None:
    """
    If `node` refers to a call_module node, return the module, else None.
    """
    if node.op == "call_module" and str(node.target) in named_modules:
        return named_modules[str(node.target)]
    else:
        return None


def _insert_dequant_stub(
    node: Node,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    """
    Attach a `DeQuantStub` to the model and create a node that calls this
    `DeQuantStub` on the output of `node`, similar to how observers are inserted.
    """
    prefix = "dequant_stub_"
    get_new_dequant_stub_name = get_new_attr_name_with_prefix(prefix)
    dequant_stub_name = get_new_dequant_stub_name(model)
    dequant_stub = DeQuantStub()
    setattr(model, dequant_stub_name, dequant_stub)
    named_modules[dequant_stub_name] = dequant_stub
    with graph.inserting_after(node):
        return graph.call_module(dequant_stub_name, (node,))


def _insert_dequant_stubs_for_custom_module_lstm_output(
    node: Node,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
) -> Node:
    """
    Insert DeQuantStubs after each internal output node of custom module LSTM.

    Custom module LSTM outputs are nested tuples of the structure (output, (hidden0, hidden1)),
    Since we cannot dequantize a tuple as a whole, we must first break down the tuple into its
    components through `getitem`. This function transforms the graph as follows:

      (1) Split the LSTM node into (output, (hidden0, hidden1))
      (2) Insert a DeQuantStub after each internal node
      (3) Recombine the DeQuantStubs into the same structure as before
      (4) Reroute all consumers of the original LSTM node and its sub-nodes
          (e.g. lstm[0])

    Before:
                   lstm_output
                        |
                        v
                  original_user(s)
    After:
                   lstm_output
                  /           \\
                 /  (getitem)  \\
                /               \\
               v                 v
             output            hidden
               |               /   \\
         (DeQuantStub)        (getitem)
               |             /       \\
               v            v         v
           output_dq     hidden0    hidden1
               |            |         |
               |    (DeQuantStub) (DeQuantStub)
               |            |         |
               |            v         v
               |      hidden0_dq  hidden1_dq
               |            \\       /
               |              (tuple)
               |              \\   /
               |               v  v
               |             hidden_dq
               \\               /
                \\   (tuple)   /
                 v            v
                 lstm_output_dq
                       |
                       v
                original_user(s)

    For step (4), reroute all users of the original LSTM node(s) as follows:
      lstm_output -> lstm_output_dq
      lstm_output[0] -> output_dq
      lstm_output[1] -> hidden_dq
      lstm_output[1][0] -> hidden0_dq
      lstm_output[1][1] -> hidden1_dq

    Return the node `lstm_output_dq`.
    """
    # (1) Split the LSTM node into (output, (hidden0, hidden1))
    # (2) Insert a DeQuantStub after each internal node
    with graph.inserting_after(node):
        output = graph.call_function(operator.getitem, (node, 0))
        output_dq = _insert_dequant_stub(output, model, named_modules, graph)
    with graph.inserting_after(output_dq):
        hidden = graph.call_function(operator.getitem, (node, 1))
    with graph.inserting_after(hidden):
        hidden0 = graph.call_function(operator.getitem, (hidden, 0))
        hidden0_dq = _insert_dequant_stub(hidden0, model, named_modules, graph)
    with graph.inserting_after(hidden0_dq):
        hidden1 = graph.call_function(operator.getitem, (hidden, 1))
        hidden1_dq = _insert_dequant_stub(hidden1, model, named_modules, graph)

    # (3) Recombine the DeQuantStubs into the same structure as before
    with graph.inserting_after(hidden1_dq):
        hidden_dq = graph.call_function(tuple, ([hidden0_dq, hidden1_dq],))
    with graph.inserting_after(hidden_dq):
        lstm_output_dq = graph.call_function(tuple, ([output_dq, hidden_dq],))

    # (4) Reroute all consumers of the original LSTM node and its sub-nodes
    for user in list(node.users.keys()):
        if user != output and user != hidden:
            user.replace_input_with(node, lstm_output_dq)
    # The getitem and tuple nodes we added here may interfere with reference quantized
    # pattern matching, so we need to redirect the consumers of internal nodes to the
    # corresponding nodes with DeQuantStubs (e.g. lstm_output_dq[0] -> output_dq) attached,
    # in order to preserve reference patterns like "dequantize - consumer - quantize".
    _reroute_tuple_getitem_pattern(graph)
    return lstm_output_dq


def _maybe_get_custom_module_lstm_from_node_arg(
    arg: Node,
    named_modules: dict[str, torch.nn.Module],
) -> Node | None:
    """
    Given an argument of a node, if the argument refers to the path through which the node
    is a consumer of custom module LSTM, return the custom module LSTM node, or None otherwise.

    This is used to determine whether a node is a consumer of custom module LSTM, and, if so,
    skip inserting input observers for this node. This is because custom module LSTM produces
    quantized outputs, so inserting an input observer for the consumer of custom module LSTM
    would unnecessarily quantize the outputs again.

      lstm -> consumer

    In practice, however, custom module LSTM outputs a tuple (output, (hidden0, hidden1)) with
    DeQuantStubs attached to each internal node (see `_insert_dequant_stubs_for_custom_module_lstm_output`).
    This tuple can be consumed in one of four ways:

      lstm -> getitem -> DeQuantStub -> consumer                       # consume lstm[0]
      lstm -> getitem -> getitem -> DeQuantStub -> tuple -> consumer   # consume lstm[1]
      lstm -> getitem -> getitem -> DeQuantStub -> consumer            # consume lstm[1][0] or lstm[1][1]
      lstm -> getitem -> DeQuantStub -> tuple -> consumer              # consume lstm

    Thus, we must match against the above patterns instead of simply checking the parent node
    to determine whether this node is a consumer of a custom module LSTM.
    """

    def match_dq(a):
        return isinstance(_get_module(a, named_modules), DeQuantStub)

    def match_lstm(a):
        return _is_custom_module_lstm(a, named_modules)

    def match_getitem(a):
        return a.op == "call_function" and a.target is operator.getitem

    def match_tuple(a):
        return a.op == "call_function" and a.target is tuple

    def _match_pattern(match_pattern: list[Callable]) -> Node | None:
        """
        Traverse up the graph and match the args one by one.
        If there is a match, return the last matched node, or None otherwise.
        """
        a = arg
        # pyrefly: ignore [bad-assignment]
        for i, match in enumerate(match_pattern):
            if not match(a):
                return None
            # Match next arg, for tuple the arg is a tuple of a list, e.g. ([dq_1, other_node],)
            if i < len(match_pattern) - 1:
                if match is match_tuple:
                    a = a.args[0][0]  # type: ignore[assignment,index]
                else:
                    a = a.args[0]  # type: ignore[assignment]
        # pyrefly: ignore [bad-return]
        return a

    all_match_patterns = [
        [match_dq, match_getitem, match_lstm],
        [match_tuple, match_dq, match_getitem, match_getitem, match_lstm],
        [match_dq, match_getitem, match_getitem, match_lstm],
        [match_tuple, match_dq, match_getitem, match_lstm],
    ]

    for p in all_match_patterns:
        matched_node = _match_pattern(p)
        if matched_node is not None:
            return matched_node
    return None


def _reroute_tuple_getitem_pattern(graph: Graph):
    """
    Search for patterns where N consecutive `tuple` call_function nodes are followed by
    N consecutive `getitem` call_function nodes that are "reverses" of the `tuple` nodes.
    If we find this pattern, reroute the consumers of the last `getitem` to skip these
    N `tuple` and `getitem` nodes.

    Before:

        a   b     c
        |   \\   /
        \\   tuple
         \\   /
          tuple
            |
        getitem(1)
            |
        getitem(0)
            |
            d

    After:

        b
        |
        d
    """

    def find_patterns(
        node: Node,
        index_stack: list[int],
        current_pattern: list[Node],
        matched_patterns: list[list[Node]],
        seen: set[tuple[Node, tuple[int, ...]]],
    ):
        """
        Traverse the graph recursively to match for the N-tuple - N-getitem patterns,
        starting at the given node.

        We use a stack to keep track of the expected `getitem` indices, since these are
        reversed from the `tuple` indices. In the above example, the stack after
        (b -> tuple -> tuple) will be [0, 1], which will be popped by getitem(1) first
        and then by getitem(0).

        TODO: traverse upwards from the output and handle the case when tuple is not a
        separate node, e.g. graph.call_function(operator.getitem, args=(a, (b, c)))
        """
        if len(index_stack) == 0 and len(current_pattern) > 0:
            matched_patterns.append(copy.copy(current_pattern))
            current_pattern.clear()

        # Avoid duplicating work
        state = (node, tuple(index_stack))
        if state in seen:
            return
        seen.add(state)

        # Iterate through users of this node to find tuple/getitem nodes to match
        for user in node.users:
            if user.op == "call_function" and user.target is tuple:
                for i, user_arg in enumerate(user.args[0]):  # type: ignore[arg-type]
                    if user_arg == node:
                        index_stack.append(i)
                        current_pattern.append(user)
                        find_patterns(
                            user, index_stack, current_pattern, matched_patterns, seen
                        )
            elif user.op == "call_function" and user.target is operator.getitem:
                if len(index_stack) > 0:
                    if user.args[1] == index_stack[-1]:
                        index_stack.pop()
                        current_pattern.append(user)
                        find_patterns(
                            user, index_stack, current_pattern, matched_patterns, seen
                        )
        return matched_patterns

    # Collect all matched patterns
    matched_patterns: list[list[Node]] = []
    seen: set[tuple[Node, tuple[int, ...]]] = set()  # (node, index_stack)
    for node in graph.nodes:
        find_patterns(node, [], [], matched_patterns, seen)

    # For each pattern, redirect all consumers of the last getitem node to the correct input
    # of the first tuple node
    for pattern in matched_patterns:
        first_tuple = pattern[0]
        last_getitem = pattern[-1]
        if not (first_tuple.op == "call_function" and first_tuple.target is tuple):
            raise AssertionError(
                "first tuple node must be a call_function with target tuple"
            )
        if not (
            last_getitem.op == "call_function"
            and last_getitem.target is operator.getitem
        ):
            raise AssertionError(
                "last getitem node must be a call_function with target operator.getitem"
            )
        last_getitem_index = last_getitem.args[1]
        new_input = first_tuple.args[0][last_getitem_index]  # type: ignore[index]
        for user in list(last_getitem.users.keys()):
            user.replace_input_with(last_getitem, new_input)  # type: ignore[arg-type]


def _get_observer_from_activation_post_process(
    activation_post_process: ObserverBase | FakeQuantizeBase,
) -> ObserverBase:
    """
    If `activation_post_process` is an observer, return the observer.
    If `activation_post_process` is a fake quantize, return the internal observer.
    """
    if isinstance(activation_post_process, ObserverBase):
        return activation_post_process
    else:
        if not isinstance(activation_post_process, FakeQuantizeBase):
            raise AssertionError(
                "activation_post_process must be an ObserverBase or FakeQuantizeBase"
            )
        return activation_post_process.activation_post_process  # type: ignore[return-value]


def _qconfig_satisfies_dtype_config_constraints(
    qconfig: QConfigAny,
    dtype_with_constraints: DTypeWithConstraints,
    is_activation: bool = True,
) -> bool:
    """
    Return whether `qconfig` satisfies the following constraints from the backend,
    specified through the activation and weight DTypeWithConstraints.

        1. QConfig specified a quantization range that falls within the backend's, if any
        2. QConfig specified a min scale value that is >= the backend's, if any
        3. QConfig specified a FixedQParamsObserver or FixedQParamsFakeQuantize that has
           scale and zero point that match the backend's, if any

    If `is_activation` is True, we check `qconfig.activation`, else we check `qconfig.weight`.
    If `qconfig` or `dtype_with_constraints.dtype` is None, or the dtypes do not match, return True.
    """

    # TODO: log warnings only when the user enabled a debug flag
    def _activation_post_process_satisfies_dtype_config_constraints(
        activation_post_process: ObserverBase | FakeQuantizeBase,
        dtype_with_constraints: DTypeWithConstraints,
        debug_string: str,
    ) -> bool:
        observer = _get_observer_from_activation_post_process(activation_post_process)
        app_quant_min = getattr(observer, "quant_min", None)
        app_quant_max = getattr(observer, "quant_max", None)
        # TODO: for now, just use the existing eps value as scale_min. In the future, we should
        # resolve the differences between the two, either by renaming eps or some other way
        app_scale_min = getattr(observer, "eps", None)
        backend_quant_min = dtype_with_constraints.quant_min_lower_bound
        backend_quant_max = dtype_with_constraints.quant_max_upper_bound
        backend_scale_min = dtype_with_constraints.scale_min_lower_bound
        backend_scale_exact_match = dtype_with_constraints.scale_exact_match
        backend_zero_point_exact_match = dtype_with_constraints.zero_point_exact_match
        # check quantization ranges
        if backend_quant_min is not None and backend_quant_max is not None:
            if app_quant_min is None or app_quant_max is None:
                warnings.warn(
                    f"QConfig {debug_string} must specify 'quant_min' and 'quant_max', ignoring {qconfig}",
                    stacklevel=2,
                )
                return False
            elif app_quant_min < backend_quant_min or app_quant_max > backend_quant_max:
                warnings.warn(
                    f"QConfig {debug_string} quantization range must fall within the backend's:\n"
                    f"QConfig range = ({app_quant_min}, {app_quant_max}), "
                    f"BackendConfig range = ({backend_quant_min}, {backend_quant_max}), "
                    f"ignoring {qconfig}",
                    stacklevel=2,
                )
                return False
        # check scale min
        if backend_scale_min is not None:
            if app_scale_min is None:
                warnings.warn(
                    f"QConfig {debug_string} must specify 'eps', ignoring {qconfig}",
                    stacklevel=2,
                )
                return False
            if app_scale_min < backend_scale_min:
                warnings.warn(
                    f"QConfig {debug_string} eps ({app_scale_min}) must be greater than or equal to "
                    f"the backend's min scale value ({backend_scale_min}), ignoring {qconfig}",
                    stacklevel=2,
                )
                return False
        # check fixed scale and zero point
        if (
            backend_scale_exact_match is not None
            and backend_zero_point_exact_match is not None
        ):
            # For tests only, accept the following qconfigs for now
            # TODO: handle fp16 qconfigs properly
            for accepted_qconfig in [float16_static_qconfig, float16_dynamic_qconfig]:
                if qconfig_equals(qconfig, accepted_qconfig):
                    return True
            suggestion_str = (
                "Please use torch.ao.quantization.get_default_qconfig_mapping or "
                "torch.ao.quantization.get_default_qat_qconfig_mapping. Example:\n"
                '    qconfig_mapping = get_default_qconfig_mapping("fbgemm")\n'
                "    model = prepare_fx(model, qconfig_mapping, example_inputs)"
            )
            if not isinstance(
                activation_post_process, FixedQParamsObserver
            ) and not isinstance(activation_post_process, FixedQParamsFakeQuantize):
                warnings.warn(
                    f"QConfig must specify a FixedQParamsObserver or a FixedQParamsFakeQuantize "
                    f"for fixed qparams ops, ignoring {qconfig}.\n{suggestion_str}",
                    stacklevel=2,
                )
                return False
            if (
                observer.scale != backend_scale_exact_match
                or observer.zero_point != backend_zero_point_exact_match
            ):
                warnings.warn(
                    f"QConfig fixed scale ({observer.scale}) and zero point ({observer.zero_point}) "
                    f"do not match the backend's ({backend_scale_exact_match} and {backend_zero_point_exact_match}), "
                    f"ignoring {qconfig}.\n{suggestion_str}",
                    stacklevel=2,
                )
                return False
        return True

    if qconfig is None or dtype_with_constraints.dtype is None:
        return True

    activation_post_process_ctr = (
        qconfig.activation if is_activation else qconfig.weight
    )
    debug_string = "activation" if is_activation else "weight"
    satisfies_constraints = True
    if activation_post_process_ctr is not None:
        activation_post_process = activation_post_process_ctr()
        if not _is_activation_post_process(activation_post_process):
            raise AssertionError(
                "activation_post_process must be an activation post process"
            )
        # If dtypes don't match, don't check the activation_post_process and return True early
        if activation_post_process.dtype != dtype_with_constraints.dtype:
            return True
        satisfies_constraints = (
            _activation_post_process_satisfies_dtype_config_constraints(
                activation_post_process, dtype_with_constraints, debug_string
            )
        )
    return satisfies_constraints
<|endoftext|># mypy: allow-untyped-defs
import copy
from collections import OrderedDict
from typing import Any

from torch.ao.quantization.fake_quantize import FixedQParamsFakeQuantize
from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.utils import Pattern


__all__ = [
    "get_default_fusion_patterns",
    "get_default_quant_patterns",
    "get_default_output_activation_post_process_map",
]

# TODO(future PR): fix the typing on QuantizeHandler (currently a circular dependency)
QuantizeHandler = Any

# pattern for conv bn fusion
_DEFAULT_FUSION_PATTERNS: dict[Pattern, QuantizeHandler] = OrderedDict()


def _register_fusion_pattern(pattern):
    def insert(fn):
        _DEFAULT_FUSION_PATTERNS[pattern] = fn
        return fn

    return insert


def get_default_fusion_patterns() -> dict[Pattern, QuantizeHandler]:
    return copy.copy(_DEFAULT_FUSION_PATTERNS)


_DEFAULT_QUANTIZATION_PATTERNS: dict[Pattern, QuantizeHandler] = OrderedDict()

# Mapping from pattern to activation_post_process(observer/fake_quant) constructor for output activation
# e.g. pattern: torch.sigmoid,
#      output_activation_post_process: default_fixed_qparams_range_0to1_fake_quant
_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP: dict[Pattern, QuantizeHandler] = {}
_DEFAULT_OUTPUT_OBSERVER_MAP: dict[Pattern, QuantizeHandler] = {}


# Register pattern for both static quantization and qat
def _register_quant_pattern(pattern, fixed_qparams_observer=None):
    def insert(fn):
        _DEFAULT_QUANTIZATION_PATTERNS[pattern] = fn
        if fixed_qparams_observer is not None:
            _DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP[pattern] = (
                FixedQParamsFakeQuantize.with_args(observer=fixed_qparams_observer)
            )
            _DEFAULT_OUTPUT_OBSERVER_MAP[pattern] = fixed_qparams_observer
        return fn

    return insert


# Get patterns for both static quantization and qat
def get_default_quant_patterns() -> dict[Pattern, QuantizeHandler]:
    return copy.copy(_DEFAULT_QUANTIZATION_PATTERNS)


# a map from pattern to output activation post process constructor
# e.g. torch.sigmoid -> default_affine_fixed_qparam_fake_quant
def get_default_output_activation_post_process_map(
    is_training,
) -> dict[Pattern, ObserverBase]:
    if is_training:
        return copy.copy(_DEFAULT_OUTPUT_FAKE_QUANTIZE_MAP)
    else:
        return copy.copy(_DEFAULT_OUTPUT_OBSERVER_MAP)


# Example use of register pattern function:
# @_register_fusion_pattern(torch.nn.ReLU, (torch.nn.BatchNorm2d, torch.nn.Conv2d)))
# class ConvOrLinearBNReLUFusion():
#     def __init__(...):
#         ...
#


def _sorted_patterns_dict(
    patterns_dict: dict[Pattern, QuantizeHandler],
) -> dict[Pattern, QuantizeHandler]:
    """
    Return a sorted version of the patterns dictionary such that longer patterns are matched first,
    e.g. match (F.relu, F.linear) before F.relu.
    This works for current use cases, but we may need to have a more clever way to sort
    things to address more complex patterns
    """

    def get_len(pattern):
        """this will calculate the length of the pattern by counting all the entries
        in the pattern.
        this will make sure (nn.ReLU, (nn.BatchNorm, nn.Conv2d)) comes before
        (nn.BatchNorm, nn.Conv2d) so that we can match the former first
        """
        len = 0
        if isinstance(pattern, tuple):
            for item in pattern:
                len += get_len(item)
        else:
            len += 1
        return len

    return OrderedDict(
        sorted(
            patterns_dict.items(),
            key=lambda kv: -get_len(kv[0]) if isinstance(kv[0], tuple) else 1,
        )
    )
<|endoftext|># mypy: allow-untyped-defs
import copy
import warnings
from typing import Any

import torch
from torch._subclasses import FakeTensor
from torch.ao.quantization import (
    ObserverBase,
    ObserverOrFakeQuantize,
    PlaceholderObserver,
)
from torch.ao.quantization.backend_config import (
    BackendConfig,
    DTypeConfig,
    get_native_backend_config,
)
from torch.ao.quantization.backend_config.utils import (
    get_fusion_pattern_to_root_node_getter,
    get_module_to_qat_module,
    get_pattern_to_dtype_configs,
)
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig import _is_reuse_input_qconfig, QConfigAny
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quantize import convert, propagate_qconfig_
from torch.ao.quantization.utils import (
    _parent_name,
    get_qconfig_dtypes,
    get_swapped_custom_module_class,
    NodePattern,
    Pattern,
)
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from torch.fx.node import Argument

from ._equalize import is_equalization_observer, node_supports_equalization
from .custom_config import PrepareCustomConfig, StandaloneModuleConfigEntry
from .match_utils import _find_matches, _MatchResultWithQConfig
from .pattern_utils import _sorted_patterns_dict
from .qconfig_mapping_utils import (
    _generate_node_name_to_qconfig,
    _get_flattened_qconfig_dict,
    _update_qconfig_for_fusion,
    _update_qconfig_for_qat,
)
from .quantize_handler import (
    _default_root_node_getter,
    _get_pattern_to_quantize_handlers,
    QuantizeHandler,
)
from .utils import (
    _insert_dequant_stubs_for_custom_module_lstm_output,
    _is_custom_module_lstm,
    _maybe_get_custom_module_lstm_from_node_arg,
    _qconfig_satisfies_dtype_config_constraints,
    all_node_args_have_no_tensors,
    assert_and_get_unique_device,
    get_custom_module_class_keys,
    get_new_attr_name_with_prefix,
    get_non_observable_arg_indexes_and_types,
    node_arg_is_bias,
    node_arg_is_weight,
    NON_QUANTIZABLE_WEIGHT_OPS,
    ObservedGraphModuleAttrs,
)


__all__ = [
    "insert_observers_for_model",
    "prepare",
    "propagate_dtypes_for_known_nodes",
]


# list of dtypes to not add observers to
_DO_NOT_OBS_DTYPE_LIST = [int, float, torch.bool, None]
_OBS_DTYPE_LIST = [
    torch.quint8,
    torch.qint8,
    torch.qint32,
    torch.float16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.float8_e5m2,
    torch.float8_e4m3fn,
]

_DEFAULT_FP32_OBS_OR_FQ_CTR = PlaceholderObserver.with_args(dtype=torch.float)

# note: the following default target dtype info dicts are temporary,
# should be moved to the new programmable API class soon
_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO = {
    "input_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig.activation,
    "output_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig.activation,
}

_DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO = {
    "input_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_quint8_placeholder_qconfig.activation,
    "output_act_obs_or_fq_ctr": torch.ao.quantization.qconfig._default_quint8_placeholder_qconfig.activation,
}


def _needs_obs_or_fq(
    prev_output_dtype: Any,
    prev_output_is_dynamic: bool,
    cur_target_dtype: Any,
    cur_target_is_dynamic: bool,
    reuse_input_obs_or_fq: bool,
    is_zeroth_arg: bool = False,
) -> bool:
    """
    note: we will treat "not specified" as torch.float for now
    utility function that checks if we should insert an observer or fake quant node
    base on the requested dtype for the nodes from user

    is_zeroth_arg: we only dynamically quantize the first arg of the node right now
      this should be removed when we enable configuring dynamic quantization
      for a specific argument, this can be removed if we deprecate fx graph mode
      quantization

    """

    # need to insert placeholder observer for dynamic quantization so that it can
    # be converted to choose_qparams -> q -> dq in convert step
    if cur_target_is_dynamic:
        if cur_target_dtype not in _OBS_DTYPE_LIST:
            raise AssertionError(
                f"Expected cur_target_dtype to be torch.float, but got: {cur_target_dtype}"
            )
        if prev_output_dtype in _DO_NOT_OBS_DTYPE_LIST:
            raise AssertionError(
                "prev_output_dtype must not be in _DO_NOT_OBS_DTYPE_LIST"
            )
        return is_zeroth_arg
    if reuse_input_obs_or_fq:
        return False
    # non dynamic quantization
    if cur_target_dtype in _OBS_DTYPE_LIST:
        return (
            prev_output_dtype in _OBS_DTYPE_LIST + [torch.float]
            and cur_target_dtype != prev_output_dtype
        )

    # lots of error checking are skipped here for now
    return False


def _is_activation_post_process_node(
    node: Node, named_modules: dict[str, torch.nn.Module]
) -> bool:
    return (
        isinstance(node, torch.fx.Node)
        and node.op == "call_module"
        and _is_activation_post_process(named_modules[str(node.target)])
    )


def _get_dtype_and_is_dynamic(
    obs_or_fq: ObserverOrFakeQuantize | None,
) -> tuple[torch.dtype | None, bool]:
    """Given a constructor for observer or fake quant module, returns
    a Tuple of dtype and is_dynamic
    """
    # TODO: instead of instantiating the instance, we can use inspect to get the default args
    if obs_or_fq is None:
        return None, False
    else:
        return obs_or_fq.dtype, getattr(obs_or_fq, "is_dynamic", False)  # type: ignore[return-value]


def _is_input_arg_dtype_supported_by_backend(
    arg: Argument,
    node: Node,
    qconfig: QConfigAny,
    dtype_config: DTypeConfig,
    backend_config: BackendConfig,
) -> bool:
    """Check if the configured qconfig for the argument
    is supported by the backend or not
    """
    if isinstance(arg, (list, tuple)):
        return all(
            _is_input_arg_dtype_supported_by_backend(
                a,
                node,
                qconfig,
                dtype_config,
                backend_config,
            )
            for a in arg
        )
    if not isinstance(arg, Node):
        return True
    # TODO: support check for standalone module
    is_weight = node_arg_is_weight(node, arg)
    is_bias = node_arg_is_bias(node, arg)
    is_activation = not is_weight and not is_bias
    if is_activation:
        input_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get(
            "input_act_obs_or_fq_ctr"
        )
        input_act_obs_or_fq = (
            input_act_obs_or_fq_ctr() if input_act_obs_or_fq_ctr else None
        )
        qconfig_dtype, qconfig_is_dynamic = _get_dtype_and_is_dynamic(
            input_act_obs_or_fq
        )
        # TODO(future PR): remove the cast to bool below after figuring
        # out why backend_config has is_dynamic set to None in some cases.
        return (dtype_config.input_dtype is None) or (
            dtype_config.input_dtype == qconfig_dtype
            and bool(dtype_config.is_dynamic) == bool(qconfig_is_dynamic)
            and _qconfig_satisfies_dtype_config_constraints(
                qconfig, dtype_config.input_dtype_with_constraints
            )
        )
    elif is_weight:
        # TODO: move dtype check into `_qconfig_satisfies_dtype_config_constraints` as well
        weight_obs_or_fq_ctr = node.meta["target_dtype_info"].get(
            "weight_obs_or_fq_ctr", None
        )
        weight_obs_or_fq = weight_obs_or_fq_ctr() if weight_obs_or_fq_ctr else None
        qconfig_weight_dtype, _ = _get_dtype_and_is_dynamic(weight_obs_or_fq)
        backend_config_weight_dtype = dtype_config.weight_dtype
        dtype_matches = qconfig_weight_dtype == backend_config_weight_dtype
        qconfig_satisfies_constraints = _qconfig_satisfies_dtype_config_constraints(
            qconfig, dtype_config.weight_dtype_with_constraints, is_activation=False
        )
        return backend_config_weight_dtype is None or (
            dtype_matches and qconfig_satisfies_constraints
        )
    else:  # bias
        # TODO: move dtype check into `_qconfig_satisfies_dtype_config_constraints` as well
        bias_obs_or_fq_ctr = node.meta["target_dtype_info"].get(
            "bias_obs_or_fq_ctr", None
        )
        bias_obs_or_fq = bias_obs_or_fq_ctr() if bias_obs_or_fq_ctr else None
        qconfig_bias_dtype, _ = _get_dtype_and_is_dynamic(bias_obs_or_fq)
        backend_config_bias_dtype = dtype_config.bias_dtype
        return (
            backend_config_bias_dtype is None
            or qconfig_bias_dtype == backend_config_bias_dtype
        )


def _is_output_dtype_supported_by_backend(
    node: Node,
    qconfig: QConfigAny,
    dtype_config: DTypeConfig,
) -> bool:
    """Check if the configured qconfig for the output
    is supported by the backend or not
    """
    # TODO: move dtype check into `_qconfig_satisfies_dtype_config_constraints` as well
    backend_config_output_dtype = dtype_config.output_dtype
    # TODO: we should check is_dynamic here as well, the code from _is_input_arg_dtype_supported_by_backend
    # from input activation check can be reused here
    qconfig_output_dtype = None
    output_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get(
        "output_act_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR
    )
    output_act_obs_or_fq = (
        output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
    )
    qconfig_output_dtype, qconfig_output_is_dynamic = _get_dtype_and_is_dynamic(
        output_act_obs_or_fq
    )
    # TODO: this is a hack because we can only specify one activation_obs_or_fq for
    # qconfig (qconfig.activation), and we are only supporting dynamically quantized
    # linear op which has fp32 output dtype, this should be removed if we generalize
    # the structure of qconfig in the future
    if qconfig_output_is_dynamic:
        qconfig_output_dtype = torch.float32
    dtype_matches = qconfig_output_dtype == backend_config_output_dtype
    qconfig_satisfies_constraints = _qconfig_satisfies_dtype_config_constraints(
        qconfig, dtype_config.output_dtype_with_constraints
    )
    return backend_config_output_dtype is None or (
        dtype_matches and qconfig_satisfies_constraints
    )


from typing import Annotated

from torch.fx import Node


EdgeOrNode = Annotated[tuple[Node, Node] | Node, None]


def _is_observer_in_same_graph(
    node: Node,
    named_modules: dict[str, torch.nn.Module],
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat,
):
    """Check if observer in same graph
    when the node output is not fp32 and input is 'placeholder'
    the input is assumed to be quantized, so it is observed
    in a different place rather than not observed.
    """
    node_output_dtype = _get_arg_target_dtype_as_output(
        node, named_modules, obs_or_fq_map, is_qat
    )
    if len(node.args) > 0 and isinstance(node.args[0], Node):
        if (
            node_output_dtype in [torch.quint8, torch.uint8]
            and node.args[0].op == "placeholder"
        ):
            return False
    return True


def _is_pattern_dtype_config_and_qconfig_supported_by_backend(
    pattern: Pattern | None,
    matched_node_pattern: list[Node] | None,
    qconfig: QConfigAny,
    backend_config: BackendConfig,
) -> bool:
    """Check if the dtype configuration of a pattern is supported by
    the backend or not, and whether the qconfig satisfies constraints
    specified in the corresponding dtype config.
    """
    if backend_config is None or pattern is None:
        return True
    if matched_node_pattern is None or len(matched_node_pattern) < 1:
        raise AssertionError("matched_node_pattern must be non-empty")
    pattern_to_dtype_configs = get_pattern_to_dtype_configs(backend_config)
    dtype_configs: list[DTypeConfig] = pattern_to_dtype_configs.get(pattern, [])
    pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(backend_config)

    root_node_getter = pattern_to_root_node_getter.get(
        pattern, _default_root_node_getter
    )
    root_node = root_node_getter(matched_node_pattern)
    input_node = root_node
    output_node = matched_node_pattern[0]
    for dtype_config in dtype_configs:
        # check if arg dtype are supported
        supported = True
        for arg in list(input_node.args) + list(input_node.kwargs.values()):
            supported = supported and _is_input_arg_dtype_supported_by_backend(
                arg, input_node, qconfig, dtype_config, backend_config
            )
        # check if output dtype is supported
        supported = supported and _is_output_dtype_supported_by_backend(
            output_node, qconfig, dtype_config
        )
        if supported:
            return True
    return False


def _get_standalone_module_configs(
    node: Node,
    named_modules: dict[str, torch.nn.Module],
    prepare_custom_config: PrepareCustomConfig,
    parent_qconfig: QConfigAny,
    parent_backend_config: BackendConfig | None,
) -> tuple[QConfigMapping, tuple[Any, ...], PrepareCustomConfig, BackendConfig | None]:
    """
    Returns the standalone module QConfigMapping and PrepareCustomConfig
    for `node`, assuming that the module pointed to by `node` is
    a standalone modules.
    """
    module_name = str(node.target)
    module_type = type(named_modules[module_name])  # type: ignore[index]
    # name config has precedence over type config
    config_entry = StandaloneModuleConfigEntry(None, (), None, None)
    config_entry = prepare_custom_config.standalone_module_classes.get(
        module_type, config_entry
    )
    config_entry = prepare_custom_config.standalone_module_names.get(
        module_name, config_entry
    )
    # fallback to use parent module's qconfig if user didn't specify qconfig dict
    qconfig_mapping = config_entry.qconfig_mapping or QConfigMapping().set_global(
        parent_qconfig
    )
    example_inputs = config_entry.example_inputs
    prepare_custom_config = config_entry.prepare_custom_config or PrepareCustomConfig()
    backend_config = config_entry.backend_config or parent_backend_config
    return (qconfig_mapping, example_inputs, prepare_custom_config, backend_config)


def _qat_swap_modules(
    root: torch.nn.Module, module_to_qat_module: dict[Pattern, type[torch.nn.Module]]
) -> None:
    convert(root, mapping=module_to_qat_module, inplace=True, remove_qconfig=False)


def _add_matched_node_name_to_set(matched_node_pattern: NodePattern, s: set[str]):
    if isinstance(matched_node_pattern, Node):
        s.add(matched_node_pattern.name)
    elif isinstance(matched_node_pattern, (list, tuple)):
        for maybe_node in matched_node_pattern:
            _add_matched_node_name_to_set(maybe_node, s)


def _insert_obs_or_fq(
    node: Node,
    obs_or_fq: ObserverOrFakeQuantize,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    model_device: torch.device | None = None,
) -> Node:
    """
    Attaches `obs_or_fq` to `model`, and creates a node which calls
    `obs_or_fq` on the output of `node`.

    obs_or_fq: an instance of Observer or FakeQuantize module
    """
    if model_device is None:
        model_device = assert_and_get_unique_device(model)
    if model_device:
        obs_or_fq.to(model_device)
    # add obs_or_fq module as attribute
    if is_equalization_observer(obs_or_fq):
        prefix = node.name + "_equalization_process_"
    else:
        prefix = "activation_post_process_"
    get_new_obs_or_fq_name = get_new_attr_name_with_prefix(prefix)
    obs_or_fq_name = get_new_obs_or_fq_name(model)
    setattr(model, obs_or_fq_name, obs_or_fq)
    named_modules[obs_or_fq_name] = obs_or_fq
    with graph.inserting_after(node):
        new_obs = graph.create_node("call_module", obs_or_fq_name, (node,), {})
    return new_obs


def _set_target_dtype_info_for_matched_node_pattern(
    matched_node_pattern: NodePattern,
    last_node: Node,
    qconfig: QConfigAny,
    qhandler: QuantizeHandler | None,
    backend_config: BackendConfig,
    named_modules: dict[str, torch.nn.Module],
    cache_for_no_tensor_check: dict[Node, bool],
    processed_nodes: set[Node],
) -> None:
    """Sets the target_dtype_info for each node in matched_node_pattern
    Note: processed_nodes is used to ensure we only process each node once
    """
    if isinstance(matched_node_pattern, (list, tuple)):
        for node_pattern in matched_node_pattern:
            _set_target_dtype_info_for_matched_node_pattern(
                node_pattern,
                last_node,
                qconfig,
                qhandler,
                backend_config,
                named_modules,
                cache_for_no_tensor_check,
                processed_nodes,
            )

    # set target_dtype_info if matched_node_pattern is a Node
    # other types of matched object, e.g. int, float literals, are ignored
    elif isinstance(matched_node_pattern, Node):
        # for pyre
        if not isinstance(matched_node_pattern, Node):
            raise AssertionError("matched_node_pattern must be a Node")
        node = matched_node_pattern
        if node in processed_nodes:
            return
        processed_nodes.add(node)

        if qconfig is None:
            return
        # TODO: refactor the following code in terms of apply a qconfig to a pattern
        # e.g. for a pattern with op1 -> op2 -> op3, and qconfig = QConfig(input_act=obs0, output_act=obs1)
        # we set the input_obs_or_fq_ctr for the arguments of op1 to based on qconfig.input_act,
        # and set output_obs_or_fq_ctr based on qconfig.output_act
        # this also requires we extend the structure of QConfig to support more fine
        # grained configurations
        target_dtype_info: dict[str, Any] = _get_target_activation_dtype_for_node(
            node,
            qconfig,
            qhandler,
            named_modules,
            backend_config,
            cache_for_no_tensor_check,
        )
        node.meta["target_dtype_info"] = target_dtype_info


def _get_target_activation_dtype_for_node(
    node: Node,
    qconfig: QConfigAny,
    qhandler: QuantizeHandler | None,
    named_modules: dict[str, torch.nn.Module],
    backend_config: BackendConfig,
    cache_for_no_tensor_check: dict[Node, bool],
) -> dict[str, Any]:
    """
    For each op attribute in the op's input activation, output activation,
    weight, bias - returns the settings of dtype and is_dynamic we expect
    for the `quantize` call in the reference model representation, or None
    if there is no `quantize` call needed.

    For example, if we have a node corresponding to `op0` in

      x0 -> op0 -> x1

    And we want a reference quantized representation to be

      x0 -> quant_static -> dequant -> op0 -> quant_dynamic -> dequant -> x1

    Then this function will return

      {
        "input_act_obs_or_fq_ctr": MinMaxObserver.with_args(dtype=torch.quint8, is_dynamic=False),
        "output_act_obs_or_fq_ctr": MinMaxObserver.with_args(dtype=torch.quint8, is_dynamic=False),
      }

    TODO(future PR, if needed): explicitly spell out the non-Tensor
    dtypes.
    """
    args_have_no_tensors = all_node_args_have_no_tensors(
        node, named_modules, cache_for_no_tensor_check
    )
    if args_have_no_tensors:
        return {
            "input_act_obs_or_fq_ctr": None,
            "output_act_obs_or_fq_ctr": None,
        }
    # get qconfig to determine the eventual dtype of this node
    if qconfig is not None:
        act_dtype, weight_dtype, input_act_is_dynamic = get_qconfig_dtypes(qconfig)

        # Currently `QConfig` only has one `activation` field.
        # For static quantization, it is reused for both input
        # and output activation. For dynamic quantization, this
        # field is currently only used for the input activation,
        # with the output activation being in fp32.
        # In the future this may change as we add more fields
        # to the `QConfig` object.
        bias_dtype = (
            torch.float16
            if (
                act_dtype == torch.float16
                and weight_dtype == torch.float16
                and (not input_act_is_dynamic)
            )
            else torch.float
        )

        is_general_tensor_value_op = (
            qhandler is not None and qhandler.is_general_tensor_value_op()
        )

        _is_standalone_module = qhandler is not None and qhandler.is_standalone_module()

        weight_index = None
        if (
            isinstance(node, Node)
            and node.op == "call_function"
            and node.target in backend_config._pattern_complex_format_to_config
        ):
            weight_index = backend_config._pattern_complex_format_to_config[
                node.target
            ]._input_type_to_index.get("weight")

        bias_index = None
        if (
            isinstance(node, Node)
            and node.op == "call_function"
            and node.target in backend_config._pattern_complex_format_to_config
        ):
            bias_index = backend_config._pattern_complex_format_to_config[
                node.target
            ]._input_type_to_index.get("bias")

        return {
            "input_act_obs_or_fq_ctr": qconfig.activation,
            "weight_obs_or_fq_ctr": qconfig.weight,
            "bias_obs_or_fq_ctr": PlaceholderObserver.with_args(dtype=bias_dtype),
            "weight_index": weight_index,
            "bias_index": bias_index,
            "output_act_obs_or_fq_ctr": qconfig.activation,
            "reuse_input_obs_or_fq": _is_reuse_input_qconfig(qconfig),
            "input_output_share_observers": is_general_tensor_value_op,
            "_is_standalone_module": _is_standalone_module,
        }
    return copy.copy(_DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO)


def _get_output_act_obs_or_fq(
    arg: Node,
    named_modules: dict[str, torch.nn.Module],
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> ObserverOrFakeQuantize | None:
    """Get the constructor for observer or fake quant object for
    the argument in the original graph as the output of previous node,
    skipping inserted observers

    We are assuming that the observers are inserted correctly, and the dtype for
    argument in quantized graph will match what is specified by the qconfig
    """
    if not isinstance(arg, Node):
        raise AssertionError("arg must be a Node")

    if "quantization_annotation" in arg.meta:
        raise NotImplementedError(
            "Please use torchao (https://github.com/pytorch/ao) for pt2e quantization flow"
        )

    # Custom module LSTM output is a tuple that we broke down into the internal nodes in order
    # to insert DeQuantStubs (see `_insert_dequant_stubs_for_custom_module_lstm_output`).
    # Since we modified the graph in this case, we must trace back from the args through
    # the specific nodes we added in order to reach the original LSTM node. Otherwise, we would
    # not be able to accurately detect whether this node is a consumer of custom module LSTM.
    custom_module_lstm_node = _maybe_get_custom_module_lstm_from_node_arg(
        arg, named_modules
    )
    output_act_obs_or_fq_ctr = None
    if custom_module_lstm_node is not None:
        output_act_obs_or_fq_ctr = custom_module_lstm_node.meta["target_dtype_info"][
            "output_act_obs_or_fq_ctr"
        ]
        output_act_obs_or_fq = (
            output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
        )
    elif _is_activation_post_process_node(arg, named_modules):
        observed_arg = arg.args[0]
        if not isinstance(observed_arg, Node):
            raise AssertionError("Currently we only support observing Node")

        if "quantization_annotation" in observed_arg.meta:
            raise NotImplementedError(
                "Please use torchao (https://github.com/pytorch/ao) for pt2e quantization flow"
            )

        if "target_dtype_info" not in observed_arg.meta:
            raise AssertionError("expected 'target_dtype_info' in observed_arg.meta")
        output_act_obs_or_fq_ctr = observed_arg.meta["target_dtype_info"][
            "output_act_obs_or_fq_ctr"
        ]
        output_act_obs_or_fq = (
            output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
        )
    else:
        if "target_dtype_info" in arg.meta:
            output_act_obs_or_fq_ctr = arg.meta["target_dtype_info"].get(
                "output_act_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR
            )
        else:
            output_act_obs_or_fq_ctr = _DEFAULT_FP32_OBS_OR_FQ_CTR
        output_act_obs_or_fq = (
            output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
        )

    return output_act_obs_or_fq


def _get_arg_target_dtype_as_output(
    arg: Node,
    named_modules: dict[str, torch.nn.Module],
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> torch.dtype | None:
    arg_as_output_act_obs_or_fq = _get_output_act_obs_or_fq(
        arg, named_modules, obs_or_fq_map, is_qat
    )
    arg_as_output_target_dtype, _ = _get_dtype_and_is_dynamic(
        arg_as_output_act_obs_or_fq
    )
    return arg_as_output_target_dtype


def _get_arg_as_input_act_obs_or_fq(
    arg: Node,
    node: Node,
    named_modules: dict[str, torch.nn.Module],
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> ObserverOrFakeQuantize | None:
    """Get the observer or fake quant constructor for the Argument `arg`, as input
    to Node `node`
    """
    if not isinstance(arg, Node):
        raise AssertionError("arg must be a Node")

    if "quantization_annotation" in node.meta:
        raise NotImplementedError(
            "Please use torchao (https://github.com/pytorch/ao) for pt2e quantization flow"
        )

    # we can remove the following path in the future if fx graph mode quantization is
    # no longer used
    is_weight = node_arg_is_weight(node, arg)
    is_bias = node_arg_is_bias(node, arg)
    is_activation = not is_weight and not is_bias
    obs_or_fq_ctr = None
    if is_activation:
        obs_or_fq_ctr = node.meta["target_dtype_info"].get(
            "input_act_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR
        )
    elif is_weight:
        if node.target not in NON_QUANTIZABLE_WEIGHT_OPS:
            obs_or_fq_ctr = node.meta["target_dtype_info"].get(
                "weight_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR
            )
    else:
        obs_or_fq_ctr = node.meta["target_dtype_info"].get(
            "bias_obs_or_fq_ctr", _DEFAULT_FP32_OBS_OR_FQ_CTR
        )
    return obs_or_fq_ctr() if obs_or_fq_ctr else None


def _maybe_insert_input_observer_for_arg_or_kwarg(
    node: Node | Any,
    arg: Argument,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    qhandler: QuantizeHandler | None,
    prepare_custom_config: PrepareCustomConfig,
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
    backend_config: BackendConfig | None = None,
    model_device: torch.device | None = None,
) -> Argument:
    """
    Given a `node` and an `arg`, inserts an input observer between
    `node` and `arg` if necessary.
    """
    # for ops such as torch.cat([x0, x1]),
    # traverse through the list
    if isinstance(arg, (list, tuple)):
        new_arg_to_return = []
        for inner_arg in arg:
            new_inner_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
                node,
                inner_arg,
                qconfig,
                model,
                named_modules,
                graph,
                qhandler,
                prepare_custom_config,
                obs_or_fq_map,
                is_qat,
                backend_config,
                model_device,
            )
            new_arg_to_return.append(new_inner_arg)
        return type(arg)(new_arg_to_return)

    if not isinstance(arg, Node):
        return arg
    if not isinstance(arg, Node):
        raise AssertionError("arg must be a Node")
    # default (no observer)
    new_arg = arg

    is_standalone_module = qhandler is not None and qhandler.is_standalone_module()
    # TODO: move this to a separate function
    if not is_standalone_module:
        # Note: qconfig can be None in this branch this we are getting act/fq from
        # node.meta now
        # regular flow for most nodes, except standalone modules

        if "quantization_annotation" in node.meta:
            raise NotImplementedError(
                "Please use torchao (https://github.com/pytorch/ao) for pt2e quantization flow"
            )

        if "target_dtype_info" not in node.meta:
            raise AssertionError("expected 'target_dtype_info' in node.meta")
        # TODO: we are assuming "target_dtype_info" exists here, maybe
        # a default value also need to be provided here
        target_dtype_info = node.meta["target_dtype_info"]
        # for nodes that doesn't have `reuse_input_obs_or_fq` configured,
        # we'll default to False, this makes configuring this field optional for users
        reuse_input_obs_or_fq = target_dtype_info.get("reuse_input_obs_or_fq", False)
        arg_as_input_act_obs_or_fq = _get_arg_as_input_act_obs_or_fq(
            arg, node, named_modules, obs_or_fq_map, is_qat
        )
        (
            arg_as_input_target_dtype,
            arg_as_input_target_is_dynamic,
        ) = _get_dtype_and_is_dynamic(arg_as_input_act_obs_or_fq)

        arg_as_output_act_obs_or_fq = _get_output_act_obs_or_fq(
            arg, named_modules, obs_or_fq_map, is_qat
        )
        (
            arg_as_output_target_dtype,
            arg_as_output_target_is_dynamic,
        ) = _get_dtype_and_is_dynamic(arg_as_output_act_obs_or_fq)

        needs_obs_or_fq = _needs_obs_or_fq(
            arg_as_output_target_dtype,
            arg_as_output_target_is_dynamic,
            arg_as_input_target_dtype,
            arg_as_input_target_is_dynamic,
            reuse_input_obs_or_fq,
            is_zeroth_arg=len(node.args) > 0 and arg is node.args[0],
        )

    else:
        if qconfig is None:
            raise AssertionError("qconfig must not be None")
        # custom flow for standalone modules
        _, _, sm_prepare_custom_config, _ = _get_standalone_module_configs(
            node, named_modules, prepare_custom_config, qconfig, backend_config
        )
        sm_input_quantized_idxs = sm_prepare_custom_config.input_quantized_indexes

        # for args, this is set to the index of the current arg
        # for kwargs, this is left at None
        cur_input_idx = None
        for arg_idx, arg_to_check in enumerate(node.args):
            if arg_to_check is arg:
                cur_input_idx = arg_idx
                break

        if cur_input_idx is None:
            needs_obs_or_fq = False
        else:
            arg_as_output_target_dtype = _get_arg_target_dtype_as_output(
                arg, named_modules, obs_or_fq_map, is_qat
            )
            arg_as_input_target_dtype = (
                torch.quint8
                if cur_input_idx in sm_input_quantized_idxs
                else torch.float
            )
            needs_obs_or_fq = (
                arg_as_output_target_dtype != arg_as_input_target_dtype
            ) and (arg_as_input_target_dtype != torch.float)

        act_post_process_ctr = qconfig.activation
        arg_as_input_act_obs_or_fq = (
            act_post_process_ctr() if act_post_process_ctr else None
        )

    if needs_obs_or_fq:
        existing_obs_node = None

        # Before using the new observer, check if an observer
        # of the correct type already exists. If it does, use it.
        # This prevents duplicate observer insertions if a node is
        # used by multiple nodes.
        # TODO: this is looking into how the value is used in the future
        # we should remove this
        # removing this means we insert one observer for each use, even if they
        # have the same dtype, we can have an extra pass that removes the extra observers
        for maybe_obs_node in arg.users:
            if maybe_obs_node.op == "call_module":
                maybe_obs_mod = named_modules[maybe_obs_node.target]  # type: ignore[index]
                if (
                    type(maybe_obs_mod) is type(arg_as_input_act_obs_or_fq)
                    and maybe_obs_mod.dtype == arg_as_input_target_dtype  # type: ignore[possibly-undefined]
                ):
                    arg_as_input_act_obs_or_fq = maybe_obs_mod  # type: ignore[assignment]
                    existing_obs_node = maybe_obs_node
                    break

        if arg_as_input_act_obs_or_fq is None:
            raise AssertionError("arg_as_input_act_obs_or_fq must not be None")
        obs_or_fq_map[(arg, node)] = arg_as_input_act_obs_or_fq
        if existing_obs_node is None:
            new_obs_node = _insert_obs_or_fq(
                arg,
                arg_as_input_act_obs_or_fq,
                model,
                named_modules,
                graph,
                model_device,
            )
            # override this arg to be the observed arg
            new_arg = new_obs_node
        else:
            new_arg = existing_obs_node

    return new_arg


def _maybe_insert_input_observers_for_node(
    node: Node,
    qconfig: QConfigAny,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    qhandler: QuantizeHandler | None,
    prepare_custom_config: PrepareCustomConfig,
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
    backend_config: BackendConfig | None = None,
    model_device: torch.device | None = None,
) -> None:
    """
    If needed, inserts observers to the input args and kwargs of `node`.
    Note: modifies `node` inplace.

    For example, if cur_node needs an observer after prev_node, we change from

      prev_node -> cur_node

    To

      prev_node -> obs -> cur_node

    Note: backend_config only needed for standalone_module node
    """
    # Look through every input arg.  If that arg's target dtype does not
    # match the current node's target dtype, insert an observer.
    new_args = []
    for arg in node.args:
        new_arg = _maybe_insert_input_observer_for_arg_or_kwarg(
            node,
            arg,
            qconfig,
            model,
            named_modules,
            graph,
            qhandler,
            prepare_custom_config,
            obs_or_fq_map,
            is_qat,
            backend_config,
            model_device,
        )
        new_args.append(new_arg)

    new_kwargs = {}
    for k, kwarg in node.kwargs.items():
        new_kwarg = _maybe_insert_input_observer_for_arg_or_kwarg(
            node,
            kwarg,
            qconfig,
            model,
            named_modules,
            graph,
            qhandler,
            prepare_custom_config,
            obs_or_fq_map,
            is_qat,
            backend_config,
            model_device,
        )
        new_kwargs[k] = new_kwarg

    # assign the new args and kwargs to the node, inplace
    node.args = tuple(new_args)
    node.kwargs = new_kwargs


def _maybe_insert_input_equalization_observers_for_node(
    node: Node,
    equalization_qconfig: Any,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    is_branch: bool,
) -> None:
    """
    If `node` needs to be equalized, find the input/weight observers it needs in
    `equalization_qconfig`, creates them, and inserts it into `graph`.

    If `node` does not need an equalization observer, returns None.
    """
    if equalization_qconfig is None or not node_supports_equalization(
        node, named_modules
    ):
        return

    if is_branch:
        warnings.warn(
            f"Cannot equalize {node} because it is part of a branch.", stacklevel=2
        )
        return

    new_args = []
    for arg in node.args:
        if not isinstance(arg, Node) or node_arg_is_bias(node, arg):
            new_args.append(arg)
            continue

        is_weight = node_arg_is_weight(node, arg)

        act_eq_process_ctr = (
            equalization_qconfig.weight
            if is_weight
            else equalization_qconfig.input_activation
        )

        new_eq_obs_mod = act_eq_process_ctr()
        new_eq_obs_node = _insert_obs_or_fq(
            arg, new_eq_obs_mod, model, named_modules, graph
        )

        new_args.append(new_eq_obs_node)

    # assign the new args and kwargs to the node, inplace
    node.args = tuple(new_args)


def _maybe_insert_output_observer_for_node(
    node: Node,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> Node | None:
    """
    If `node` needs an output observer, creates it, inserts it into `graph`
    and returns it.

    If `node` does not need an output observer, returns None.

    Note: inserting dynamic quantization ops for output is not supported in fx graph mode
    quantization code path right now
    """
    if node.op == "output":
        raise AssertionError("observer insertion for outputs is handled elsewhere")

    is_standalone_module = False
    if "quantization_annotation" in node.meta:
        raise NotImplementedError(
            "Please use torchao (https://github.com/pytorch/ao) for pt2e quantization flow"
        )

    if "target_dtype_info" not in node.meta:
        raise AssertionError("expected 'target_dtype_info' in node.meta")
    is_standalone_module = node.meta["target_dtype_info"].get(
        "_is_standalone_module", False
    )
    output_act_obs_or_fq_ctr = node.meta["target_dtype_info"].get(
        "output_act_obs_or_fq_ctr"
    )
    output_act_obs_or_fq = (
        output_act_obs_or_fq_ctr() if output_act_obs_or_fq_ctr else None
    )
    target_dtype, target_is_dynamic = _get_dtype_and_is_dynamic(output_act_obs_or_fq)
    # uncomment after we support reuse_input_obs_or_fq properly by having separate
    # implementations for this key instead of reusing the input_output_share_observers
    # code
    # reuse_input_obs_or_fq = node.meta["target_dtype_info"].get("reuse_input_obs_or_fq", False)
    # for now we set this to False since reuse_input_obs_or_fq for
    # the output of a node is implemented in the same code path as observer sharing,
    # we should refactor this part to make it clearer in the future
    # and we would be able to read this from config directly
    reuse_input_obs_or_fq = False

    # Note: prev_output_dtype = torch.float and prev_output_is_dynamic=False
    # because the prev_output is the output of an fp32 op, although technically
    # we should get the dtype of the output from node.meta["val"] in the future
    # if we deprecate fx graph mode quantization
    needs_obs_or_fq = _needs_obs_or_fq(
        torch.float, False, target_dtype, target_is_dynamic, reuse_input_obs_or_fq
    )
    # currently the activation in QConfig(activation=...,) is for both input
    # and output, and when the activation is configured to be dynamic quantization
    # e.g. PlaceholderObserver(dtype=torch.quint8, is_dynamic=True, ...), it means
    # the input should by dynamically quantized, but output should not be quantized
    #
    # there is no way we can specify different observer/fq for input and output
    # activation through QConfig today, this limitation is lifted in the
    # quantizer/annotation API in pytorch 2.0 export quantization code path,
    # but since this code is reused, annotating output to be dynamically quantized
    # would not work either for that.
    # we can change QConfig to support input/output activation if we want
    # to remove the following check, or if we can deprecate fx graph mode quantization
    if target_is_dynamic:
        needs_obs_or_fq = False

    # we never insert observers to output of standalone module, we assume
    # if needed, they are inserted inside the standalone module
    needs_obs_or_fq = needs_obs_or_fq and (not is_standalone_module)

    if needs_obs_or_fq:
        obs_or_fq_map[node] = output_act_obs_or_fq
        return _insert_obs_or_fq(
            node, output_act_obs_or_fq, model, named_modules, graph
        )
    else:
        return None


def _maybe_insert_observers_before_graph_output(
    graph_output_node: Node,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
    graph: Graph,
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize],
    is_qat: bool,
) -> None:
    """
    If the output needs to be quantized and there are any nodes
    in the output which are not already observed, inserts observers
    for those nodes.
    """

    def _recursive_maybe_replace_node_with_obs(
        maybe_node: Argument,
        model: torch.nn.Module,
        named_modules: dict[str, torch.nn.Module],
        graph: Graph,
    ) -> Argument:
        """
        Navigate an arbitrary data structure of lists, tuples, dicts.
        For each container type, recurse on all inputs. Once any Node
        is found, insert an observer if needed and do not recurse further.

        For example, given a structure of

          {'foo1': [[bar1]], 'foo2': {'foo3': [[[bar3]]]}}

        we recurse down to bar1 and bar3, observe them if necessary,
        and if we inserted an observer then replace the original node
        with its observer.

        Returns the data structure with all nodes needing observation being
        replaced by their observers.
        """
        if isinstance(maybe_node, Node):
            # check dtype of this node
            arg_as_output_target_dtype = _get_arg_target_dtype_as_output(
                maybe_node, named_modules, obs_or_fq_map, is_qat
            )
            observer_mod = None
            arg_as_input_target_dtype = torch.float
            if "target_dtype_info" in maybe_node.meta:
                observer_cls = maybe_node.meta["target_dtype_info"].get(
                    "input_act_obs_or_fq_ctr", None
                )
                if observer_cls is not None:
                    observer_mod = observer_cls()
                    arg_as_input_target_dtype = observer_mod.dtype
            # TODO: this does not handle dynamic quantization yet
            need_obs = (
                arg_as_output_target_dtype != arg_as_input_target_dtype
                and arg_as_input_target_dtype != torch.float
            )
            if need_obs:
                if observer_mod is None:
                    raise AssertionError(
                        "observer_mod must not be None when need_obs is True"
                    )
                # insert observer
                observer_node = _insert_obs_or_fq(
                    maybe_node, observer_mod, model, named_modules, graph
                )
                return observer_node
            else:
                return maybe_node
        elif isinstance(maybe_node, (list, tuple)):
            results = [
                _recursive_maybe_replace_node_with_obs(
                    inner_node,
                    model,
                    named_modules,
                    graph,
                )
                for inner_node in maybe_node
            ]
            if isinstance(maybe_node, list):
                return results
            else:
                return tuple(results)
        elif isinstance(maybe_node, dict):
            results_dict = {}
            for k, inner_v in maybe_node.items():
                results_dict[k] = _recursive_maybe_replace_node_with_obs(
                    inner_v, model, named_modules, graph
                )
            return results_dict
        elif maybe_node is None:
            return None
        else:
            raise Exception(  # noqa: TRY002
                "Unhandled type for returned node:", maybe_node
            )

    new_args = [
        _recursive_maybe_replace_node_with_obs(old_arg, model, named_modules, graph)
        for old_arg in graph_output_node.args
    ]

    graph_output_node.args = tuple(new_args)  # type: ignore[assignment]


def _maybe_propagate_dtype_for_node(
    node: Node,
    target_dtype: torch.dtype | type,
    node_name_to_match_result_with_qconfig: dict[str, _MatchResultWithQConfig],
) -> None:
    """
    Assigns `target_dtype` to `node`, setting `is_dynamic` to False. If `node`
    is a general tensor shape op, also call this function recursively on
    the first argument, to propagate the dtype to the caller.
    """
    node.meta["target_dtype_info"]["input_act_obs_or_fq_ctr"] = None
    node.meta["target_dtype_info"]["output_act_obs_or_fq_ctr"] = None
    # if this is a copy node, propagate to first arg
    (
        _root_node,
        _,
        _pattern,
        qhandler,
        _qconfig,
    ) = node_name_to_match_result_with_qconfig.get(
        node.name, (None, None, None, None, None)
    )
    # TODO: probably need to remove `is_general_tensor_value_op`
    if qhandler is not None and qhandler.is_general_tensor_value_op():
        prev_node = node.args[0]
        if isinstance(prev_node, Node):
            _maybe_propagate_dtype_for_node(
                prev_node, target_dtype, node_name_to_match_result_with_qconfig
            )


def propagate_dtypes_for_known_nodes(
    graph: Graph,
    node_name_to_match_result_with_qconfig: dict[str, _MatchResultWithQConfig],
) -> None:
    """
    Currently we assume that inputs to the graph are either `torch.float` or
    `torch.quint8`, which is not always correct. For ops such as
    `x.masked_fill(mask, value)`, we know that the dtype of  `mask` is a
    `BoolTensor`. Propagate this information throughout the graph.

    Note: not all dtypes in the graph will be correct after this pass, but a
    higher percentage of them will be correct. Hopefully in the future we can
    replace this with a better way to reason about dtypes of tensors.
    """
    for node in graph.nodes:
        non_observable_arg_dict = get_non_observable_arg_indexes_and_types(node)

        for arg_type in non_observable_arg_dict:
            non_observable_indices = non_observable_arg_dict[arg_type](node)

            for index in non_observable_indices:
                arg = node.args[index]

                # when an argument is a tuple, it does not show up as another node so we need to go through
                # all elements of the tuple manually
                if isinstance(arg, (tuple, list)):
                    arg_list = list(arg)
                else:
                    arg_list = [arg]

                for cur_arg in arg_list:
                    # hard coded arguments show up but aren't `Node` typed and do not need dtype propagated
                    if isinstance(cur_arg, torch.fx.node.Node):
                        _maybe_propagate_dtype_for_node(
                            cur_arg, arg_type, node_name_to_match_result_with_qconfig
                        )


def _maybe_make_input_output_share_observers(
    node: Node,
    model: torch.nn.Module,
    named_modules: dict[str, torch.nn.Module],
) -> bool:
    """
    Ensures that we share an observer
    for all input arguments as well as the output argument. In detail, given
    a graph of

      x0 -> obs0 -> op -> x2
                  /
      x1 -> obs1 /

    where node obs0 points to observer instance observer0,
    obs1 points to observer1 and obs2 points to observer2, we make nodes obs1
    and ob2 point to observer0.
    Returns: whether the operation succeeded or not
    """
    first_arg = None
    # find the first non-Tensor arg
    for i in range(len(node.args)):
        if isinstance(node.args[i], (Node, list, tuple)):
            first_arg = node.args[i]
            break

    # if there is no non-Tensor arg, return directly
    if first_arg is None:
        return False

    if isinstance(first_arg, (list, tuple)):
        first_arg_arg = first_arg[0]
    elif isinstance(first_arg, Node):
        first_arg_arg = first_arg
    else:
        return False

    # if we have a graph such as
    #   observed_node -> non_observed_node -> cat
    # we need to navigate up to the first observer
    iteration_guard = 0
    # pyrefly: ignore [bad-argument-type]
    while not _is_activation_post_process_node(first_arg_arg, named_modules):
        if not isinstance(first_arg_arg, Node):
            return False
        # did not find an activation_post_process for the op
        if first_arg_arg.op == "placeholder":
            return False
        # trace back the args until we found the first Tensor/Node
        trace_back_node = None
        for i in range(len(first_arg_arg.args)):
            trace_back_node = first_arg_arg.args[i]
            if isinstance(trace_back_node, Node):
                break
        if trace_back_node is None:
            return False
        first_arg_arg = trace_back_node

        iteration_guard += 1
        if iteration_guard > 10000:
            raise AssertionError("Unable to find observer of previous node")

    if not isinstance(first_arg_arg, Node):
        raise AssertionError("first_arg_arg must be a Node")
    target_to_use = first_arg_arg.target
    if not isinstance(target_to_use, str):
        raise AssertionError("target_to_use must be a string")
    obs_mod_to_use = named_modules[target_to_use]

    if isinstance(first_arg, (list, tuple)):
        # set all other input observer nodes to use that module
        for input_idx, input_arg in enumerate(first_arg):
            if input_idx == 0:
                continue
            iteration_guard = 0
            # pyrefly: ignore [bad-argument-type]
            while not _is_activation_post_process_node(input_arg, named_modules):
                # failed to trace back since no input arg for the current node
                # pyrefly: ignore [missing-attribute]
                if len(input_arg.args) < 1:
                    return False
                # pyrefly: ignore [bad-index, unsupported-operation]
                input_arg = input_arg.args[0]
                iteration_guard += 1
                if iteration_guard > 10000:
                    raise AssertionError("Unable to find observer of previous node")

            # pyrefly: ignore [missing-attribute]
            parent_name, name = _parent_name(input_arg.target)
            setattr(named_modules[parent_name], name, obs_mod_to_use)

    # set the output observer node to use that module
    for output_obs_node in node.users:
        if not _is_activation_post_process_node(output_obs_node, named_modules):
            raise AssertionError(
                "output_obs_node must be an activation post process node"
            )
        parent_name, name = _parent_name(output_obs_node.target)
        setattr(named_modules[parent_name], name, obs_mod_to_use)

    # TODO(future PR): delete the orphaned observer modules
    return True


def _remove_output_observer(
    node: Node, model: torch.nn.Module, named_modules: dict[str, torch.nn.Module]
):
    items = list(node.users.items())
    for output_obs_node, _ in items:
        if not _is_activation_post_process_node(output_obs_node, named_modules):
            raise AssertionError(
                "output_obs_node must be an activation post process node"
            )
        output_obs_node.replace_all_uses_with(node)
        model.graph.erase_node(output_obs_node)  # type: ignore[union-attr, operator]


def _swap_custom_module_to_observed(
    node: Node,
    qconfig: QConfigAny,
    named_modules: dict[str, torch.nn.Module],
    prepare_custom_config: PrepareCustomConfig,
):
    custom_module = named_modules[node.target]  # type: ignore[index]
    custom_module_class_mapping = prepare_custom_config.float_to_observed_mapping
    observed_custom_module_class = get_swapped_custom_module_class(
        custom_module, custom_module_class_mapping, qconfig
    )
    observed_custom_module = observed_custom_module_class.from_float(custom_module)
    parent_name, name = _parent_name(node.target)
    setattr(named_modules[parent_name], name, observed_custom_module)


def insert_observers_for_model(
    model: GraphModule,
    node_name_to_match_result_with_qconfig: dict[str, _MatchResultWithQConfig],
    node_name_to_qconfig: dict[str, QConfigAny],
    prepare_custom_config: PrepareCustomConfig,
    equalization_config_map: dict[str, Any],
    backend_config: BackendConfig,
    observed_node_names: set[str],
    is_qat: bool,
) -> Node | None:
    """
    Inserts observers, using the following high level algorithm:

    For each node in the graph:
      1. determine the target dtype of this node in the quantized graph, and save
           it for future steps
      2. determine the target dtype or all args and kwargs of this node
      3. if any arg or kwarg's target dtype does not match the current node's
           dtype, insert an observer
      4. if the current node needs an output observer, insert it

    For example:

    - starting graph:
        x0 -> linear -> x1

    - observed graph after processing x0:
        x0(fp32)

    - observed graph after processing linear:
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8)

    - observed graph after processing x1:
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8) -> x1

    After a node is processed, the naive observer placement is guaranteed to be
    complete for that node and all of its predecessors. There can be future
    passes which optimize the graph by deduplicating observers, etc.
    """

    # node.meta["target_dtype_info"] stores the target dtype information
    # that's derived from qconfig for the Node, for example, if we have
    # a conv2d node that has a qconfig
    # qconfig = QConfig(activation=..., weight=...)
    # # information for input and bias node omitted
    # # for getattr node
    # # weight = getattr(self, 'weight')
    # weight.meta["target_dtype_info"] = {
    #    'output_act_obs_or_fq_ctr': qconfig.weight,
    # }
    # # for conv2d node
    # # conv2d = call_function[target=torch.nn.functional.conv2d](
    # #            args=(input, weight, bias))
    # conv2d.meta["target_dtype_info"] = {
    #   'input_act_obs_or_fq_ctr': qconfig.activation
    #   'weight_obs_or_fq_ctr': qconfig.weight,
    #   'bias_obs_or_fq_ctr': PlaceholderObserver.with_args(dtype=torch.float32),
    #   'output_act_obs_or_fq_ctr': qconfig.activation,
    # }
    #
    cache_for_no_tensor_check: dict[Node, bool] = {}

    # first, populate the dtype map based only on qconfig and qhandler
    # this assumes:
    # graph inputs are fp32 by default, and int8 where overridden
    # other nodes output dtype is specified by the qconfig
    named_modules = dict(model.named_modules(remove_duplicate=False))

    input_quantized_idxs: list[int] = prepare_custom_config.input_quantized_indexes
    output_quantized_idxs: list[int] = prepare_custom_config.output_quantized_indexes
    processed_nodes: set[Node] = set()
    # initialize target_dtype_info
    for node in model.graph.nodes:
        node.meta["target_dtype_info"] = copy.copy(
            _DEFAULT_FP32_QCONFIG_FOR_TARGET_DTYPE_INFO
        )

    inputs_seen_counter = 0
    outputs_seen_counter = 0
    placeholder_node_to_input_index: dict[Node, int] = {}
    # TODO: we probably don't need this counter since each graph will only have
    # one output node?
    output_node_to_output_index: dict[Node, int] = {}
    for node in model.graph.nodes:
        if node.op == "placeholder":
            placeholder_node_to_input_index[node] = inputs_seen_counter
            inputs_seen_counter += 1
        if node.op == "output":
            output_node_to_output_index[node] = outputs_seen_counter
            outputs_seen_counter += 1

    # Step 1, set the observer or fake quantize module constructor for each node in the
    # matched_node_pattern

    for match_res_with_qconfig in node_name_to_match_result_with_qconfig.values():
        (
            last_node,
            matched_node_pattern,
            pattern,
            qhandler,
            qconfig,
        ) = match_res_with_qconfig
        if qhandler is None:
            raise AssertionError("qhandler must not be None")
        _set_target_dtype_info_for_matched_node_pattern(
            matched_node_pattern,
            last_node,
            qconfig,
            qhandler,
            backend_config,
            named_modules,
            cache_for_no_tensor_check,
            processed_nodes,
        )

    # Step 2. Special cases for some operators, we might be able to remove them
    # in the future if we know dtype information of each node better

    # Step 2.1. some settings are not based on patterns, we need to process each node
    # instead
    for node in model.graph.nodes:
        if (
            node.op == "placeholder"
            and placeholder_node_to_input_index[node] in input_quantized_idxs
        ):
            # users are not supposed to call calculate_qparams on PlaceholderObserver, and
            # this is OK because we are using this as a way to encode the dtypes of input
            # tensor, we won't actually insert these observers in the graph and won't
            # actually call calculate_qparams
            node.meta["target_dtype_info"] = copy.copy(
                _DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO
            )
        elif node.op in ("call_module", "call_method", "call_function"):
            args_have_no_tensors = all_node_args_have_no_tensors(
                node, named_modules, cache_for_no_tensor_check
            )
            if args_have_no_tensors:
                node.meta["target_dtype_info"] = {
                    "input_act_obs_or_fq_ctr": None,
                    "output_act_obs_or_fq_ctr": None,
                }
        elif (
            node.op == "output"
            and output_node_to_output_index[node] in output_quantized_idxs
        ):
            # TODO(future PR): update the output_quantized_idxs API to match
            # arbitrary data structures. There is always a single output, and
            # that output can have arbitrary nesting of values. List[int] is
            # not the right data type for this.

            # TODO(future PR): support more dtypes in model outputs, if necessary
            node.meta["target_dtype_info"] = copy.copy(
                _DEFAULT_QUINT8_QCONFIG_FOR_TARGET_DTYPE_INFO
            )

    # Step 2.2, for nodes with known input dtypes, propagate them throughout the
    # graph. For example, if there is a call such as
    #   x1 = x0.masked_fill(mask, 1)
    # we propagate the type of mask to be torch.bool
    propagate_dtypes_for_known_nodes(
        model.graph, node_name_to_match_result_with_qconfig
    )

    # Step 3, check if the requested target_dtype_info is supported by backend or not
    # if not, we'll reset the target_dtye_info to use the default (float Tensor)

    # reset the counters and set of processed_nodes
    processed_nodes: set[Node] = set()
    for match_res_with_qconfig in node_name_to_match_result_with_qconfig.values():
        (
            last_node,
            matched_node_pattern,
            pattern,
            qhandler,
            qconfig,
        ) = match_res_with_qconfig
        is_supported_by_backend = (
            _is_pattern_dtype_config_and_qconfig_supported_by_backend(
                pattern, matched_node_pattern, qconfig, backend_config
            )
        )
        if qhandler is None:
            raise AssertionError("qhandler must not be None")

        # get output_act_dtype so that we don't also reset the special typed nodes
        # TODO: we might want to handle these more uniformly with the default path
        # this can be improved if we can use node.meta["val"]
        output_act_or_fq_ctr = node.meta["target_dtype_info"][
            "output_act_obs_or_fq_ctr"
        ]
        output_act_or_fq = output_act_or_fq_ctr() if output_act_or_fq_ctr else None
        output_act_dtype, _ = _get_dtype_and_is_dynamic(output_act_or_fq)
        if not is_supported_by_backend and output_act_dtype not in [
            None,
            int,
            float,
            torch.bool,
        ]:
            # restore target_dtype_info to default if it is not supported by backend
            _set_target_dtype_info_for_matched_node_pattern(
                matched_node_pattern,
                last_node,
                torch.ao.quantization.qconfig._default_fp32_placeholder_qconfig,
                None,
                backend_config,
                named_modules,
                cache_for_no_tensor_check,
                processed_nodes,
            )

    # After this point, the current node and all of its arguments
    # have a target_dtype_info assigned. Now, we insert observers for inputs
    # of this node (if needed for this node), and the output of this node
    # (if needed for this node).

    # Since we are mutating the graph as we go, we iterate over the original
    # nodes before observer insertion, instead of model.graph.nodes.
    nodes_before_observation = list(model.graph.nodes)

    # Avoid duplicates custom module swaps for multiple nodes with same target.
    custom_module_names_already_swapped: set[str] = set()

    # TODO: reuse placeholder_node_to_input_index and output_node_to_output_index
    # reset inputs/outputs counters
    inputs_seen_counter = 0
    outputs_seen_counter = 0
    results_node = None
    obs_or_fq_map: dict[EdgeOrNode, ObserverOrFakeQuantize] = {}
    model_device = assert_and_get_unique_device(model)

    # TODO: change this to insert obs/fq by pattern instead of by node
    for node in nodes_before_observation:
        if node.op == "placeholder":
            # if a graph input is in fp32, it does not need observation
            # if a graph input is in int8, we assume the observation happens
            #   outside of the graph, and no additional observation is needed
            pass

        elif node.op in ("call_module", "call_method", "call_function", "output"):
            # check for matches
            (
                last_node,
                matched_node_pattern,
                pattern,
                qhandler,
                qconfig,
            ) = node_name_to_match_result_with_qconfig.get(  # type: ignore[assignment]
                node.name, (None, None, None, None, None)
            )
            equalization_qconfig = equalization_config_map.get(node.name)

            this_node_dtype_info = node.meta["target_dtype_info"]
            if "val" in node.meta:
                output_is_a_tensor = this_node_dtype_info is not None and isinstance(
                    node.meta["val"], FakeTensor
                )
            else:
                output_is_a_tensor = this_node_dtype_info is not None

            skip_inserting_observers = (
                (qconfig is None) or not output_is_a_tensor
            ) and (node.op != "output")

            # TODO: take a closer look to see if we can remove this check
            # right now it is here because of `observed_node_names`, we are using
            # it as an indicator for swapping the modules to reference modules in
            # convert
            is_supported_by_backend = (
                _is_pattern_dtype_config_and_qconfig_supported_by_backend(
                    pattern, matched_node_pattern, qconfig, backend_config
                )
            )

            if not skip_inserting_observers and is_supported_by_backend:
                named_modules = dict(model.named_modules(remove_duplicate=False))
                if node.op != "output":
                    if matched_node_pattern is None:
                        raise AssertionError("matched_node_pattern must not be None")
                    # add matched nodes to the observed node name set
                    _add_matched_node_name_to_set(
                        matched_node_pattern, observed_node_names
                    )

                    # This is currently only used for equalization.
                    # Checks if the current node is in a branch in which the two
                    # first layers are both being quantized.
                    #
                    # ex.       conv2
                    #         /
                    #      x -> conv1
                    #
                    # If this is the case, we will not apply equalization to the
                    # initial two layers.
                    is_quantized_branch = False
                    if (
                        len(node.args) > 0
                        and isinstance(node.args[0], Node)
                        and len(node.args[0].users) > 1
                    ):
                        for user in node.args[0].users:
                            # Checks if there exists another user being quantized
                            is_user_quantized = node_name_to_qconfig.get(
                                user.name
                            ) is not None or (
                                user.op == "call_module"
                                and isinstance(
                                    named_modules[str(user.target)], ObserverBase
                                )
                            )
                            if user != node and is_user_quantized:
                                is_quantized_branch = True

                    pattern_to_root_node_getter = (
                        get_fusion_pattern_to_root_node_getter(backend_config)
                    )
                    root_node_getter = pattern_to_root_node_getter.get(
                        pattern, _default_root_node_getter
                    )
                    root_node = root_node_getter(matched_node_pattern)
                    is_input_node_of_the_pattern = node is root_node
                    if is_input_node_of_the_pattern:
                        # this modifies node inplace
                        _maybe_insert_input_observers_for_node(
                            node,
                            qconfig,
                            model,
                            named_modules,
                            model.graph,
                            qhandler,
                            prepare_custom_config,
                            obs_or_fq_map,
                            is_qat,
                            backend_config,
                            model_device,
                        )

                        # insert equalization input observers if needed
                        _maybe_insert_input_equalization_observers_for_node(
                            node,
                            equalization_qconfig,
                            model,
                            named_modules,
                            model.graph,
                            is_quantized_branch,
                        )

                    is_last_node_of_pattern = node is last_node
                    input_output_share_observers = node.meta["target_dtype_info"].get(
                        "input_output_share_observers", False
                    )
                    reuse_input_obs_or_fq = node.meta["target_dtype_info"].get(
                        "reuse_input_obs_or_fq", False
                    )

                    if is_last_node_of_pattern:
                        if _is_custom_module_lstm(
                            # pyrefly: ignore [bad-argument-type]
                            node,
                            named_modules,
                            qconfig,
                            qhandler,
                        ):
                            # Currently custom module outputs are assumed to be already quantized,
                            # so we need to insert a DeQuantStub after the output. For custom module
                            # LSTM specifically, the outputs are also a nested tuple, so we must first
                            # break down the tuple to insert DeQuantStubs after the internal nodes.

                            # TODO: This currently diverges from how custom modules are handled today,
                            # where we insert observers after the output instead of DeQuantStubs, and
                            # replace these observers with "dequantize" nodes during convert. Conceptually,
                            # these output observers are the same as DeQuantStubs. In the future, we
                            # should resolve this inconsistency by inserting DeQuantStubs for all custom
                            # modules, not just for LSTM.
                            _insert_dequant_stubs_for_custom_module_lstm_output(
                                # pyrefly: ignore [bad-argument-type]
                                node,
                                model,
                                named_modules,
                                model.graph,
                            )
                            # pyrefly: ignore [missing-attribute]
                            if node.target not in custom_module_names_already_swapped:
                                # pyrefly: ignore [bad-argument-type]
                                custom_module_names_already_swapped.add(node.target)
                                _swap_custom_module_to_observed(
                                    # pyrefly: ignore [bad-argument-type]
                                    node,
                                    qconfig,
                                    named_modules,
                                    prepare_custom_config,
                                )
                        else:
                            # this returns the new observer node if it was needed
                            maybe_output_obs_node = (
                                _maybe_insert_output_observer_for_node(
                                    # pyrefly: ignore [bad-argument-type]
                                    node,
                                    model,
                                    named_modules,
                                    model.graph,
                                    obs_or_fq_map,
                                    is_qat,
                                )
                            )

                            if maybe_output_obs_node is not None:
                                # Update users of original node to use the output observer
                                # instead. For example, change
                                #
                                #           next_node
                                #          /
                                #   cur_node -> obs
                                #
                                # to
                                #
                                #                 next_node
                                #                 /
                                #   cur_node -> obs
                                #
                                # We need to save orig users before updating uses because
                                # the list of users will change as we update uses
                                # pyrefly: ignore [missing-attribute]
                                orig_users = list(node.users.keys())
                                for user_node in orig_users:
                                    if user_node is maybe_output_obs_node:
                                        continue
                                    user_node.replace_input_with(
                                        node, maybe_output_obs_node
                                    )

                                _is_observer_in_same_graph_ = (
                                    _is_observer_in_same_graph(
                                        # pyrefly: ignore [bad-argument-type]
                                        node,
                                        named_modules,
                                        obs_or_fq_map,
                                        is_qat,
                                    )
                                )

                                # for ops whose inputs and outputs share observer/fqs, we modify the graph
                                # to make all inputs and outputs use the first input's
                                # observer/fq
                                if (
                                    input_output_share_observers
                                    and _is_observer_in_same_graph_
                                ) or reuse_input_obs_or_fq:
                                    if not _maybe_make_input_output_share_observers(
                                        # pyrefly: ignore [bad-argument-type]
                                        node,
                                        model,
                                        named_modules,
                                    ):
                                        _remove_output_observer(
                                            # pyrefly: ignore [bad-argument-type]
                                            node,
                                            model,
                                            named_modules,
                                        )

                                if qhandler is not None and qhandler.is_custom_module():
                                    if (
                                        # pyrefly: ignore [missing-attribute]
                                        node.target
                                        not in custom_module_names_already_swapped
                                    ):
                                        custom_module_names_already_swapped.add(
                                            # pyrefly: ignore [bad-argument-type]
                                            node.target
                                        )
                                        _swap_custom_module_to_observed(
                                            # pyrefly: ignore [bad-argument-type]
                                            node,
                                            qconfig,
                                            named_modules,
                                            prepare_custom_config,
                                        )

                else:  # output
                    _maybe_insert_observers_before_graph_output(
                        node, model, named_modules, model.graph, obs_or_fq_map, is_qat
                    )

        #
        # After this point, the current node has input and output observers
        # that it needs for itself inserted.
        #

        # increment the counters, so future inputs and outputs are assigned
        # correct dtypes
        if node.op == "placeholder":
            inputs_seen_counter += 1
        elif node.op == "output":
            outputs_seen_counter += 1
            results_node = node

    return results_node


def _run_prepare_fx_on_standalone_modules(
    model: torch.nn.Module,
    is_qat: bool,
    named_modules: dict[str, torch.nn.Module],
    node_name_to_match_result_with_qconfig: Any,
    prepare_custom_config: PrepareCustomConfig,
    backend_config: BackendConfig,
) -> None:
    """
    Runs prepare_fx on each standalone module. Note: this does
    not modify the graph, it just replaces the unobserved modules with
    their observed versions.
    """
    for (
        root_node,
        _,
        _pattern,
        qhandler,
        qconfig,
    ) in node_name_to_match_result_with_qconfig.values():
        if qhandler is None:
            continue
        elif not qhandler.is_standalone_module():
            continue

        (
            sm_qconfig_mapping,
            sm_example_inputs,
            sm_prepare_custom_config,
            sm_backend_config,
        ) = _get_standalone_module_configs(
            root_node, named_modules, prepare_custom_config, qconfig, backend_config
        )

        standalone_module = named_modules[root_node.target]
        prepare = torch.ao.quantization.quantize_fx._prepare_standalone_module_fx  # type: ignore[attr-defined]
        observed_standalone_module = prepare(
            standalone_module,
            sm_qconfig_mapping,
            is_qat,
            example_inputs=sm_example_inputs,
            prepare_custom_config=sm_prepare_custom_config,
            backend_config=sm_backend_config,
        )
        parent_name, name = _parent_name(root_node.target)
        setattr(named_modules[parent_name], name, observed_standalone_module)
        named_modules[root_node.target] = observed_standalone_module


def _save_state(
    observed: GraphModule,
    node_name_to_qconfig: dict[str, QConfigAny],
    node_name_to_scope: dict[str, tuple[str, type]],
    prepare_custom_config: PrepareCustomConfig,
    equalization_node_name_to_qconfig: dict[str, Any],
    qconfig_mapping: QConfigMapping,
    is_qat: bool,
    observed_node_names: set[str],
) -> None:
    observed.meta["_observed_graph_module_attrs"] = ObservedGraphModuleAttrs(
        node_name_to_qconfig=node_name_to_qconfig,
        node_name_to_scope=node_name_to_scope,
        prepare_custom_config=prepare_custom_config,
        equalization_node_name_to_qconfig=equalization_node_name_to_qconfig,
        qconfig_mapping=qconfig_mapping,
        is_qat=is_qat,
        observed_node_names=observed_node_names,
    )


def prepare(
    model: GraphModule,
    qconfig_mapping: QConfigMapping | dict[str, Any],
    is_qat: bool,
    node_name_to_scope: dict[str, tuple[str, type]],
    example_inputs: tuple[Any, ...],
    prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = None,
    _equalization_config: QConfigMapping | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
    is_standalone_module: bool = False,
) -> GraphModule:
    """standalone_module means it a submodule that is not inlined in
    parent module, and will be quantized separately as one unit.

    How the standalone module is observed is specified by `input_quantized_idxs` and
    `output_quantized_idxs` in the prepare_custom_config for the standalone module
    Args:
        node_name_to_scope: mapping from node name to the scope of the module which contains the node.
        The scope is a tuple of fully qualified path of the module and the type of the module
    Returns:
        model(GraphModule): prepared standalone module
        attributes related to standalone module
        in model.meta["_observed_graph_module_attrs"]:
            is_observed_standalone_module (bool): boolean value that shows whether the
            current model is a observed standalone module or not
            standalone_module_input_quantized_idxs(List[Int]): a list of
                indexes for the graph input that is expected to be quantized,
                same as input_quantized_idxs configuration provided
                for the standalone module
            standalone_module_output_quantized_idxs(List[Int]): a list of
                indices for the graph output that is quantized
                same as input_quantized_idxs configuration provided
                for the standalone module
    """
    if prepare_custom_config is None:
        prepare_custom_config = PrepareCustomConfig()
    if _equalization_config is None:
        _equalization_config = QConfigMapping()

    if isinstance(qconfig_mapping, dict):
        warnings.warn(
            "Passing a QConfig dictionary to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a QConfigMapping instead.",
            FutureWarning,
            stacklevel=2,
        )
        qconfig_mapping = QConfigMapping.from_dict(qconfig_mapping)

    if isinstance(_equalization_config, dict):
        warnings.warn(
            "Passing a QConfig dictionary to prepare for equalization is deprecated and will not "
            "be supported in a future version. Please pass in a QConfigMapping instead.",
            FutureWarning,
            stacklevel=2,
        )
        _equalization_config = QConfigMapping.from_dict(_equalization_config)

    if isinstance(prepare_custom_config, dict):
        warnings.warn(
            "Passing a prepare_custom_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a PrepareCustomConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        prepare_custom_config = PrepareCustomConfig.from_dict(prepare_custom_config)

    if isinstance(backend_config, dict):
        warnings.warn(
            "Passing a backend_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a BackendConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        backend_config = BackendConfig.from_dict(backend_config)

    if not isinstance(qconfig_mapping, QConfigMapping):
        raise AssertionError("qconfig_mapping must be a QConfigMapping")
    if not isinstance(_equalization_config, QConfigMapping):
        raise AssertionError("_equalization_config must be a QConfigMapping")
    qconfig_mapping = copy.deepcopy(qconfig_mapping)
    _equalization_config = copy.deepcopy(_equalization_config)

    # mapping from a tuple of nodes in reverse order to uninitialized
    #   QuantizeHandler subclass. For example,
    # {
    #   # match a single node
    #   (<class 'torch.nn.modules.conv.Conv3d'>:
    #     <class 'torch.ao.quantization.fx.quantize.ConvRelu'>),
    #   # match multiple nodes in reverse order
    #   ((<function relu at 0x7f766a7360d0>, <built-in function add>):
    #     <class 'torch.ao.quantization.fx.quantize.Add'>),
    # }

    pattern_to_quantize_handler: dict[Pattern, QuantizeHandler] = {}
    if backend_config is None:
        backend_config = get_native_backend_config()
    pattern_to_quantize_handler = _get_pattern_to_quantize_handlers(backend_config)
    pattern_to_quantize_handler = _sorted_patterns_dict(pattern_to_quantize_handler)

    root_node_getter_mapping = get_fusion_pattern_to_root_node_getter(backend_config)

    # pyrefly: ignore [bad-argument-type]
    _update_qconfig_for_fusion(model, qconfig_mapping)
    # pyrefly: ignore [bad-argument-type]
    _update_qconfig_for_fusion(model, _equalization_config)
    # pyrefly: ignore [bad-argument-type]
    flattened_qconfig_dict = _get_flattened_qconfig_dict(qconfig_mapping)
    # TODO: support regex as well
    propagate_qconfig_(model, flattened_qconfig_dict, prepare_custom_config.to_dict())

    if is_qat:
        module_to_qat_module = get_module_to_qat_module(backend_config)
        _qat_swap_modules(model, module_to_qat_module)
        # pyrefly: ignore [bad-argument-type]
        _update_qconfig_for_qat(qconfig_mapping, backend_config)

    # mapping from fully qualified module name to module instance
    # for example,
    # {
    #   '': Model(...),
    #   'linear': Linear(...),
    #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
    # }
    named_modules = dict(model.named_modules(remove_duplicate=False))

    # fill node_name_to_qconfig, a map from node name to qconfig, used in _find_matches
    equalization_node_name_to_qconfig = _generate_node_name_to_qconfig(
        model,
        named_modules,
        model.graph,
        # pyrefly: ignore [bad-argument-type]
        _equalization_config,
        node_name_to_scope,
    )
    node_name_to_qconfig = _generate_node_name_to_qconfig(
        model,
        named_modules,
        model.graph,
        # pyrefly: ignore [bad-argument-type]
        qconfig_mapping,
        node_name_to_scope,
    )

    # match the patterns that will get quantized
    standalone_module_names = list(prepare_custom_config.standalone_module_names.keys())
    standalone_module_classes = list(
        prepare_custom_config.standalone_module_classes.keys()
    )

    custom_module_classes = get_custom_module_class_keys(
        prepare_custom_config.float_to_observed_mapping
    )
    matches_without_qconfig = _find_matches(
        model.graph,
        named_modules,
        pattern_to_quantize_handler,
        root_node_getter_mapping,
        standalone_module_names,
        standalone_module_classes,
        custom_module_classes,
    )

    # map qconfig instances to matches
    node_name_to_match_result_with_qconfig = {}
    for node_name, match_without_qconfig in matches_without_qconfig.items():
        match_with_qconfig = (*match_without_qconfig, node_name_to_qconfig[node_name])
        node_name_to_match_result_with_qconfig[node_name] = match_with_qconfig

    _run_prepare_fx_on_standalone_modules(
        model,
        is_qat,
        named_modules,
        node_name_to_match_result_with_qconfig,
        prepare_custom_config,
        backend_config,
    )

    # record names for the set of observed node, so that in convert step
    # we know whether we need to convert a floating point module to reference
    # quantized module or not
    observed_node_names: set[str] = set()

    result_node = insert_observers_for_model(
        model,
        node_name_to_match_result_with_qconfig,
        node_name_to_qconfig,
        prepare_custom_config,
        equalization_node_name_to_qconfig,
        backend_config,
        observed_node_names,
        is_qat,
    )
    model = GraphModule(model, model.graph)

    _save_state(
        model,
        node_name_to_qconfig,
        node_name_to_scope,
        prepare_custom_config,
        equalization_node_name_to_qconfig,
        # pyrefly: ignore [bad-argument-type]
        qconfig_mapping,
        is_qat,
        observed_node_names,
    )

    if is_standalone_module:
        if result_node is None:
            raise AssertionError("result_node must not be None for standalone modules")
        if not isinstance(result_node.args[0], Node):
            raise AssertionError(
                "standalone module only supports returning simple value currently (not tuple, dict etc.)"
            )
        # these inputs are observed in parent
        # converting List[int] to Tensor since module attribute is
        # Union[Tensor, Module]
        input_quantized_idxs: list[int] = prepare_custom_config.input_quantized_indexes
        output_quantized_idxs: list[int] = (
            prepare_custom_config.output_quantized_indexes
        )
        observed_graph_module_attrs = model.meta["_observed_graph_module_attrs"]
        # inplace modification
        observed_graph_module_attrs.is_observed_standalone_module = True
        observed_graph_module_attrs.standalone_module_input_quantized_idxs = (
            input_quantized_idxs
        )
        observed_graph_module_attrs.standalone_module_output_quantized_idxs = (
            output_quantized_idxs
        )
    return model
<|endoftext|># mypy: ignore-errors

import copy
import operator
import warnings
from typing import Any, TYPE_CHECKING

import torch
from torch.ao.quantization.backend_config import (
    BackendConfig,
    get_native_backend_config,
)
from torch.ao.quantization.backend_config.utils import (
    get_fused_module_classes,
    get_pattern_to_dtype_configs,
    get_qat_module_classes,
    get_root_module_to_quantized_reference_module,
)
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.qconfig import qconfig_equals, QConfigAny
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.ao.quantization.quant_type import QuantType
from torch.ao.quantization.quantize import _remove_qconfig
from torch.ao.quantization.stubs import DeQuantStub
from torch.ao.quantization.utils import (
    _parent_name,
    activation_is_statically_quantized,
    get_qparam_dict,
    get_swapped_custom_module_class,
    is_per_channel,
    to_underlying_dtype,
    weight_is_quantized,
)
from torch.fx import GraphModule
from torch.fx.graph import Argument, Graph, Node
from torch.nn.utils.parametrize import type_before_parametrizations

# importing the lib so that the quantized_decomposed ops are registered
from ._decomposed import quantized_decomposed_lib  # noqa: F401
from ._equalize import convert_eq_obs, update_obs_for_equalization
from .custom_config import ConvertCustomConfig, PrepareCustomConfig
from .graph_module import _is_observed_module, _is_observed_standalone_module
from .lower_to_fbgemm import lower_to_fbgemm
from .qconfig_mapping_utils import (
    _compare_prepare_convert_qconfig_mappings,
    _generate_node_name_to_qconfig,
    _is_qconfig_supported_by_dtype_configs,
    _update_qconfig_for_fusion,
    _update_qconfig_for_qat,
)
from .utils import (
    _get_module,
    _is_custom_module_lstm,
    _is_custom_module_mha,
    assert_and_get_unique_device,
    collect_producer_nodes,
    create_getattr_from_value,
    get_custom_module_class_keys,
    graph_module_from_producer_nodes,
    node_arg_is_weight,
)


if TYPE_CHECKING:
    from collections.abc import Callable


NUMERIC_DEBUG_HANDLE_KEY = "numeric_debug_handle"
CUSTOM_KEY = "custom"

__all__ = [
    "convert",
    "convert_custom_module",
    "convert_standalone_module",
    "convert_weighted_module",
]

SUPPORTED_QDTYPES = [
    torch.quint8,
    torch.qint8,
    torch.qint32,
    torch.uint8,
    torch.int8,
    torch.uint16,
    torch.int16,
    torch.int32,
    torch.float8_e5m2,
    torch.float8_e4m3fn,
]

_QSCHEME_TO_CHOOSE_QPARAMS_OP = {
    torch.per_tensor_affine: torch.ops.quantized_decomposed.choose_qparams.tensor,
    torch.per_tensor_symmetric: torch.ops.quantized_decomposed.choose_qparams_symmetric.tensor,
}


def _replace_observer_with_quantize_dequantize_node_decomposed(
    model: torch.fx.GraphModule,
    node: Node,
    modules: dict[str, torch.nn.Module],
    node_name_to_scope: dict[str, tuple[str, type]],
    node_name_to_qconfig: dict[str, QConfigAny],
    model_device: torch.device | None = None,
) -> None:
    """Replace activation_post_process module call node with quantize and
    dequantize node working with decomposed Tensor

    Before:
    ... -> observer_0(x) -> ...
    After:
    ... -> torch.ops.quantized_decomposed.quantize_per_tensor(x, ...) ->
    torch.ops.quantized_decomposed.dequantize_per_tensor() -> ...

    or quantize_per_channel and dequantize_per_channel
    """
    graph = model.graph
    if modules is None:
        raise AssertionError("modules must not be None")
    if not isinstance(node.target, str):
        raise AssertionError(
            f"Expected node.target to be a str, but got {type(node.target)}"
        )
    module_path, prefix = _get_module_path_and_prefix(
        node, node_name_to_scope, node_name_to_qconfig
    )
    activation_post_process = modules[node.target]
    if hasattr(activation_post_process, "convert"):
        activation_post_process.convert(model, node)
        return
    # skip replacing observers to quant/dequant nodes if the qconfigs of all
    # consumers and producers of this observer are None
    skip_replacement = all(
        _has_none_qconfig(n, node_name_to_qconfig)
        for n in list(node.args) + list(node.users.keys())
    )
    if skip_replacement or not _is_conversion_supported(activation_post_process):
        # didn't find corresponding quantize op and info for the activation_post_process
        # so we just remove the observer
        with graph.inserting_before(node):
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
        return

    # otherwise, we can convert the activation_post_process module call to quantize/dequantize node

    # 1. extract the information from activation_post_process module for generating
    # the quantize and dequantize operator
    dtype = activation_post_process.dtype  # type: ignore[attr-defined]

    is_dynamic = False
    if hasattr(activation_post_process, "is_dynamic"):
        is_dynamic = activation_post_process.is_dynamic  # type: ignore[assignment]

    def add_dequantize_op_kwargs(dequantize_op, input_node):
        dequantize_op_kwargs = {}
        if "val" in input_node.meta:
            dq_out_dtype = input_node.meta["val"].dtype
            if dq_out_dtype != torch.float32:
                dequantize_op_kwargs = {"out_dtype": dq_out_dtype}
        return dequantize_op_kwargs

    if dtype in SUPPORTED_QDTYPES and (not is_dynamic):
        # TODO: probably should cleanup this condition check, it's hard
        # to reason about this if and the following elif

        # uint8/int8/int32 static quantization branch

        # 1. extract information for inserting q/dq node from activation_post_process
        node_type = "call_function"
        quantize_op: Callable | None = None
        scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[attr-defined, operator]
        if is_per_channel(activation_post_process.qscheme):  # type: ignore[attr-defined]
            ch_axis = int(activation_post_process.ch_axis)  # type: ignore[attr-defined, arg-type]
            quantize_op = torch.ops.quantized_decomposed.quantize_per_channel.default
            dequantize_op = (
                torch.ops.quantized_decomposed.dequantize_per_channel.default
            )
            quant_min = activation_post_process.quant_min
            quant_max = activation_post_process.quant_max
            dtype_ = to_underlying_dtype(dtype)
            qparams = {
                "_scale_": scale,
                "_zero_point_": zero_point,
                "_axis_": ch_axis,
                "_quant_min_": quant_min,
                "_quant_max_": quant_max,
                "_dtype_": dtype_,
            }
        else:
            quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.default
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.default
            scale = float(scale)
            zero_point = int(zero_point)
            quant_min = activation_post_process.quant_min  # type: ignore[attr-defined]
            quant_max = activation_post_process.quant_max  # type: ignore[attr-defined]
            dtype_ = to_underlying_dtype(dtype)
            qparams = {
                "_scale_": scale,
                "_zero_point_": zero_point,
                "_quant_min_": quant_min,
                "_quant_max_": quant_max,
                "_dtype_": dtype_,
            }

        # 2. replace activation_post_process node with quantize and dequantize
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value_or_node in qparams.items():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                if key in ["_scale_", "_zero_point_"] and (
                    not isinstance(value_or_node, (float, int))
                ):
                    # For scale and zero_point values we register them as buffers in the root module.
                    # However, note that when the values are not tensors, as in the case of
                    # per_tensor quantization, they will be treated as literals.
                    # However, registering them as a node seems to cause issue with dynamo
                    # tracing where it may consider tensor overload as opposed to default.
                    # With extra check of scale and zero_point being scalar, it makes
                    # sure that the default overload can be used.
                    # TODO: maybe need more complex attr name here
                    qparam_node = create_getattr_from_value(
                        model,
                        graph,
                        module_path + prefix + key,
                        value_or_node,
                        model_device,
                    )
                    quantize_op_inputs.append(qparam_node)
                else:
                    # for qparams that are not scale/zero_point (like axis, dtype) we store them as literals in the graph.
                    quantize_op_inputs.append(value_or_node)

            quantized_node = graph.create_node(
                node_type, quantize_op, tuple(quantize_op_inputs), {}
            )
            # use the same qparams from quantize op
            dq_inputs = [quantized_node] + quantize_op_inputs[1:]
            dequantized_node = graph.call_function(
                dequantize_op,
                tuple(dq_inputs),
                add_dequantize_op_kwargs(dequantize_op, input_node),
            )

            node.replace_all_uses_with(dequantized_node)
            # propagate numeric debug handle from observer/fake_quant node to dequantize node
            if (
                CUSTOM_KEY in node.meta
                and NUMERIC_DEBUG_HANDLE_KEY in node.meta[CUSTOM_KEY]
            ):
                raise NotImplementedError(
                    "pt2e numeric suite has been migrated to torchao (https://github.com/pytorch/ao)"
                )
            graph.erase_node(node)
    elif is_dynamic:
        # uint8/int8/fp16 dynamic quantization

        # 1. extract information for inserting q/dq node from activation_post_process
        node_type = "call_function"
        quantize_op = torch.ops.quantized_decomposed.quantize_per_tensor.tensor
        # we only use choose_qparams for is_decomposed now,
        # but we should probably align the non-decomposed path with this as well,
        # and that can be done after we remove reduce_range flag
        # 1. extract qparams from activation_post_process module
        dtype_ = to_underlying_dtype(dtype)
        if dtype_ not in [torch.uint8, torch.int8]:
            raise AssertionError(
                "only uint8 and int8 are supported in reference flow for dynamic quantization right now"
            )
        quant_min = activation_post_process.quant_min  # type: ignore[attr-defined]
        quant_max = activation_post_process.quant_max  # type: ignore[attr-defined]
        qscheme = getattr(activation_post_process, "qscheme", torch.per_tensor_affine)  # type: ignore[attr-defined]
        eps = getattr(activation_post_process, "eps", torch.finfo(torch.float32).eps)  # type: ignore[attr-defined]
        # note: scale and zero_point are missing for quantize_per_tensor op
        # we'll need to get this from choose_qparams op, which we'll add after
        # this step
        qparams = {
            "_quant_min_": quant_min,
            "_quant_max_": quant_max,
            "_eps_": eps,
            "_dtype_": dtype_,
        }

        choose_qparams_op = _QSCHEME_TO_CHOOSE_QPARAMS_OP[qscheme]
        # 2. insert choose_qparams op and update the qparams list
        with graph.inserting_before(node):
            input_node = node.args[0]
            choose_qparams_op_inputs = [node.args[0]] + list(qparams.values())
            choose_qparams_node = graph.create_node(
                "call_function", choose_qparams_op, tuple(choose_qparams_op_inputs), {}
            )
            # choose_qparms returns (scale, zero_point)
            scale_node = graph.create_node(
                "call_function", operator.getitem, (choose_qparams_node, 0), {}
            )
            zero_point_node = graph.create_node(
                "call_function", operator.getitem, (choose_qparams_node, 1), {}
            )
            # we have quant_min, quant_max and dtype, all should be stored
            # as literals
            quant_min = qparams["_quant_min_"]
            quant_max = qparams["_quant_max_"]
            dtype = qparams["_dtype_"]
            qparams = {
                "_scale_": scale_node,
                "_zero_point_": zero_point_node,
                "_quant_min_": quant_min,
                "_quant_max_": quant_max,
                "_dtype_": dtype,
            }

        # 3. replace activation_post_process node to quantize and dequantize node
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value_or_node in qparams.items():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                if key in ["_scale_", "_zero_point_"]:
                    # in this case we have a node in the graph since it's dynamically
                    # computed from the input, with choose_qparams op
                    qparam_node = value_or_node
                    quantize_op_inputs.append(qparam_node)
                else:
                    # for qparams that are not scale/zero_point (like axis, dtype) we
                    # store them as literals in the graph.
                    quantize_op_inputs.append(value_or_node)

            quantized_node = graph.create_node(
                node_type, quantize_op, tuple(quantize_op_inputs), {}
            )
            # use the same qparams from quantize op
            dq_inputs = [quantized_node] + quantize_op_inputs[1:]
            # need to use the tensor variant of this op, since scale and zero_point
            # from choose_qparam are Tensors, instead of float/int, this is to
            # prevent these nodes being traced away by downstream systems
            dequantize_op = torch.ops.quantized_decomposed.dequantize_per_tensor.tensor
            dequantized_node = graph.call_function(
                dequantize_op,
                tuple(dq_inputs),
                add_dequantize_op_kwargs(dequantize_op, input_node),
            )

            node.replace_all_uses_with(dequantized_node)
            # propagate numeric debug handle from observer/fake_quant node to dequantize node
            if NUMERIC_DEBUG_HANDLE_KEY in node.meta:
                raise NotImplementedError(
                    "pt2e numeric suite has been migrated to torchao (https://github.com/pytorch/ao)"
                )
            graph.erase_node(node)
    elif dtype == torch.float16:
        # Insert to_fp16 -> to_fp32 node
        dtype_convert_op = torch.ops.quantized_decomposed.convert_element_type.no_fuse
        with graph.inserting_before(node):
            input_node = node.args[0]
            convert_fp16_node = graph.create_node(
                "call_function", dtype_convert_op, (input_node, torch.float16), {}
            )
            convert_fp32_node = graph.create_node(
                "call_function", dtype_convert_op, (convert_fp16_node, torch.float), {}
            )
            node.replace_all_uses_with(convert_fp32_node)
            graph.erase_node(node)

    # should not reach since we have checks in the beginning to make sure the
    # activation_post_process is supported


def _replace_observer_with_quantize_dequantize_node(
    model: torch.fx.GraphModule,
    node: Node,
    modules: dict[str, torch.nn.Module],
    node_name_to_scope: dict[str, tuple[str, type]],
    node_name_to_qconfig: dict[str, QConfigAny],
    model_device: torch.device | None = None,
) -> None:
    """Replace activation_post_process module call node with quantize and
    dequantize node

    Before:
    ... -> observer_0(x) -> ...
    After:
    ... -> torch.quantize_per_tensor(x, ...) -> x.dequantize() -> ...
    """
    if modules is None:
        raise AssertionError("modules must not be None")
    if not isinstance(node.target, str):
        raise AssertionError(
            f"Expected node.target to be a str, but got {type(node.target)}"
        )
    graph = model.graph
    module_path, prefix = _get_module_path_and_prefix(
        node, node_name_to_scope, node_name_to_qconfig
    )
    activation_post_process = modules[node.target]
    # skip replacing observers to quant/dequant nodes if the qconfigs of all
    # consumers and producers of this observer are None
    skip_replacement = all(
        _has_none_qconfig(n, node_name_to_qconfig)
        for n in list(node.args) + list(node.users.keys())
    )
    if skip_replacement or not _is_conversion_supported(activation_post_process):
        # didn't find corresponding quantize op and info for the activation_post_process
        # so we just remove the observer
        with graph.inserting_before(node):
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
        return

    # otherwise, we can convert the activation_post_process module call to quantize/dequantize node
    dtype = activation_post_process.dtype  # type: ignore[attr-defined]

    is_dynamic = False
    if hasattr(activation_post_process, "is_dynamic"):
        is_dynamic = activation_post_process.is_dynamic  # type: ignore[attr-defined, assignment]

    if dtype in [
        torch.quint8,
        torch.qint8,
        torch.qint32,
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    ] and (not is_dynamic):
        # TODO: probably should cleanup this condition check, it's hard
        # to reason about this if and the following elif

        # uint8/int8/int32 static quantization branch

        # 1. extract the information from activation_post_process module for generating
        # the quantize and dequantize operator
        node_type = "call_function"
        quantize_op: Callable | None = None
        scale, zero_point = activation_post_process.calculate_qparams()  # type: ignore[attr-defined, operator]
        if is_per_channel(activation_post_process.qscheme):  # type: ignore[attr-defined]
            ch_axis = int(activation_post_process.ch_axis)  # type: ignore[attr-defined, arg-type]
            qparams = {
                "_scale_": scale,
                "_zero_point_": zero_point,
                "_axis_": ch_axis,
                "_dtype_": dtype,
            }
            quantize_op = torch.quantize_per_channel
        else:
            scale = float(scale)
            zero_point = int(zero_point)
            qparams = {"_scale_": scale, "_zero_point_": zero_point, "_dtype_": dtype}
            quantize_op = torch.quantize_per_tensor

        # 2. replace activation_post_process node with quantize and dequantize
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for key, value_or_node in qparams.items():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                if key in ["_scale_", "_zero_point_"]:
                    # For scale and zero_point values we register them as buffers in the root module.
                    # TODO: maybe need more complex attr name here
                    qparam_node = create_getattr_from_value(
                        model,
                        graph,
                        module_path + prefix + key,
                        value_or_node,
                        model_device,
                    )
                    quantize_op_inputs.append(qparam_node)
                else:
                    # for qparams that are not scale/zero_point (like axis, dtype) we store them as literals in the graph.
                    quantize_op_inputs.append(value_or_node)

            quantized_node = graph.create_node(
                node_type, quantize_op, tuple(quantize_op_inputs), {}
            )
            dequantized_node = graph.call_method("dequantize", args=(quantized_node,))
            node.replace_all_uses_with(dequantized_node)
            graph.erase_node(node)
    elif is_dynamic:
        # uint8/int8/fp16 dynamic quantization branch

        node_type = "call_function"
        quantize_op = torch.quantize_per_tensor_dynamic
        # TODO: get reduce range from observer
        # reduce_range = activation_post_process.reduce_range
        reduce_range = torch.backends.quantized.engine in ("fbgemm", "x86")
        qparams = {"_dtype_": dtype, "_reduce_range_": reduce_range}

        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for value in qparams.values():
                quantize_op_inputs.append(value)

            quantized_node = graph.create_node(
                node_type, quantize_op, tuple(quantize_op_inputs), {}
            )
            dequantized_node = graph.call_method("dequantize", args=(quantized_node,))
            node.replace_all_uses_with(dequantized_node)
            graph.erase_node(node)
    elif dtype == torch.float16:
        node_type = "call_method"
        quantize_op = "to"  # type: ignore[assignment]
        qparams = {"_dtype_": dtype}
        with graph.inserting_before(node):
            input_node = node.args[0]
            quantize_op_inputs = [input_node]
            for value in qparams.values():
                # TODO: we can add the information of whether a value needs to
                # be registered as an attribute in qparams dict itself
                quantize_op_inputs.append(value)

            quantized_node = graph.create_node(
                node_type, quantize_op, tuple(quantize_op_inputs), {}
            )
            dequantized_node = graph.call_method("dequantize", args=(quantized_node,))
            node.replace_all_uses_with(dequantized_node)
            graph.erase_node(node)

    # should not reach since we have checks in the beginning to make sure the
    # activation_post_process is supported


# this is a temporary hack for custom module, we may want to implement
# this properly after the custom module class design is finalized
# TODO: DeQuantStubs are currently inserted only after custom module LSTM, while observers are inserted
# after all other custom modules. In the future, we should simply insert QuantStubs before and DeQuantStubs
# after custom modules in general, and replace these with "quantize" and "dequantize" nodes respectively.
def _replace_observer_or_dequant_stub_with_dequantize_node(
    node: Node, graph: Graph
) -> None:
    call_custom_module_node = node.args[0]
    if not isinstance(call_custom_module_node, Node):
        raise AssertionError(
            f"Expecting the for call custom module node to be a Node, but got {call_custom_module_node}"
        )
    node.replace_all_uses_with(call_custom_module_node)
    graph.erase_node(node)
    _insert_dequantize_node(call_custom_module_node, graph)


def _is_conversion_supported(activation_post_process: torch.nn.Module) -> bool:
    dtype = activation_post_process.dtype  # type: ignore[attr-defined]

    is_dynamic = False
    if hasattr(activation_post_process, "is_dynamic"):
        is_dynamic = activation_post_process.is_dynamic  # type: ignore[attr-defined, assignment]

    return (
        (dtype in SUPPORTED_QDTYPES and (not is_dynamic))
        or is_dynamic  # type: ignore[return-value]
        or dtype == torch.float16
    )


def _has_none_qconfig(
    node: Argument, node_name_to_qconfig: dict[str, QConfigAny]
) -> bool:
    """Check if a node has a qconfig of None, i.e. user requested to not quantize
    the node
    """
    return (
        isinstance(node, Node)
        and node.name in node_name_to_qconfig
        and node_name_to_qconfig[node.name] is None
    )


def _run_weight_observers(observed: GraphModule, backend_config: BackendConfig) -> None:
    """Extract the subgraph that produces the weight for dynamic quant
    or weight only quant node and run the subgraph to observe the weight.
    Note that the observers of dynamic quant or weight only quant ops are
    run during the convert step.
    """
    for node in observed.graph.nodes:
        if node.op != "call_function":
            continue
        for node_arg in node.args:
            # node_arg is weight
            if node_arg and node_arg_is_weight(node, node_arg):
                weight_observer_nodes = collect_producer_nodes(node_arg)
                if weight_observer_nodes is None:
                    continue
                weight_observer_module = graph_module_from_producer_nodes(
                    observed, weight_observer_nodes
                )
                # run the weight observer
                weight_observer_module()


def _maybe_recursive_remove_dequantize(arg: Any, node: Node, graph: Graph) -> None:
    """If the arg is a dequantize Node, or a list/tuple/dict of dequantize Node,
    we'll recursively remove the dequantize Node
    """
    if isinstance(arg, Node) and arg.op == "call_method" and arg.target == "dequantize":
        quantize_node = arg.args[0]
        # we only replace the specific use since dequantize could be used by other nodes
        # as well
        node.replace_input_with(arg, quantize_node)
    elif isinstance(arg, (list, tuple)):
        for arg_element in arg:
            _maybe_recursive_remove_dequantize(arg_element, node, graph)
    elif isinstance(arg, dict):
        for arg_element in arg.values():
            _maybe_recursive_remove_dequantize(arg_element, node, graph)
    else:
        warnings.warn(
            f"Unsupported node type in recursive remove dequantize: {type(arg)}",
            stacklevel=2,
        )


def _get_module_path_and_prefix(
    obs_node: Node,
    node_name_to_scope: dict[str, tuple[str, type]],
    node_name_to_qconfig: dict[str, QConfigAny],
) -> tuple[str, str]:
    """Given an observer node, get the `Scope` or the fully qualified name for
    the submodule containing the observed node, also return a prefix of "_input"
    when the observed node is an input of a F.linear op, and not the output of another
    quantized op.
    TODO: this logic is hacky, we should think about how to remove it or make it more
    general
    """
    observed_node = obs_node.args[0]
    # an observer can be inserted for both input of the next operator or output of the previous
    # operator (they can be the same)
    # this flag identifies if the observer is inserted only because the observed node is
    # the input of the next operator
    if not isinstance(observed_node, Node):
        raise AssertionError(
            f"Expecting observed node to be a Node, but got {observed_node}"
        )
    is_input_observer_only = (
        node_name_to_qconfig[observed_node.name] is None
        if observed_node.name in node_name_to_qconfig
        else None
    )
    if is_input_observer_only:
        # if the quantize function is at the input of op, then we find the first user of the observer_node
        # to get the path. If a linear call_function is in the user list, we return the first instance
        # of linear node to get the FQN.
        users = list(obs_node.users)
        first_linear_use_or_first_use = users[0] if users else None
        linear_node = None
        for n in users:
            if n.op == "call_function" and n.target is torch.nn.functional.linear:
                linear_node = n
                break
        if linear_node:
            first_linear_use_or_first_use = linear_node
        prefix = "_input"
    else:
        # if the quantize function is at the output of the op, we use the observer input node to get the path
        first_linear_use_or_first_use = observed_node
        prefix = ""

    if (
        first_linear_use_or_first_use
        and first_linear_use_or_first_use.name in node_name_to_scope
    ):
        module_path, _ = node_name_to_scope[first_linear_use_or_first_use.name]
    else:
        # TODO: it's not used, so actually we can skip quantization
        # but this requires changing return type of quantize_node
        # we can fix it later if needed
        module_path = ""
    return module_path, prefix


def _insert_dequantize_node(node: Node, graph: Graph) -> None:
    """Inserts dequantize node for `node` in `graph`"""
    with graph.inserting_after(node):
        dequantize_node = graph.call_method("dequantize", (node,))
        for user_node in dict(node.users):
            if user_node is not dequantize_node:
                user_node.replace_input_with(node, dequantize_node)


def _maybe_get_observer_for_node(
    node: Node, modules: dict[str, torch.nn.Module]
) -> torch.nn.Module | None:
    """
    If the node is observed, return the observer
    instance. Otherwise, return None.
    """
    for maybe_obs_node in node.users:
        if maybe_obs_node.op == "call_module":
            maybe_obs = modules[str(maybe_obs_node.target)]
            if _is_activation_post_process(maybe_obs):
                return maybe_obs
    return None


def convert_standalone_module(
    node: Node,
    modules: dict[str, torch.nn.Module],
    model: torch.fx.GraphModule,
    is_reference: bool,
    backend_config: BackendConfig | None,
) -> None:
    """Converts an observed standalone module to a quantized standalone module by calling
    the fx convert api, currently using the same `is_reference` flag as parent, but we may
    changing this behavior in the future (e.g. separating quantization and lowering for
    standalone module as well)

    Args:
      - node: The call_module node of the observed standalone module
      - modules: named_module of original model
      - model: original model
      - is_reference: a flag from parent provided by user to decide if we want to
        produce a reference model or a fbgemm/qnnpack model
      - backend_config: backend configuration of the target backend of quantization
    """
    # TODO: remove is_reference flag
    if is_reference:
        convert_fn = torch.ao.quantization.quantize_fx.convert_to_reference_fx
    else:
        convert_fn = torch.ao.quantization.quantize_fx.convert_fx  # type: ignore[attr-defined]
    # We know that observed standalone module is a GraphModule since
    # it's produced by us
    observed_standalone_module: GraphModule = modules[str(node.target)]  # type: ignore[assignment]
    sm_input_quantized_idxs = observed_standalone_module.meta[
        "_observed_graph_module_attrs"
    ].standalone_module_input_quantized_idxs
    # remove the dequantize nodes for inputs
    args = list(node.args)
    for idx in range(len(args)):
        if idx in sm_input_quantized_idxs:
            arg = args[idx]
            if arg.op == "call_method" and arg.target == "dequantize":  # type: ignore[union-attr]
                quantize_node = arg.args[0]  # type: ignore[union-attr]
                node.replace_input_with(arg, quantize_node)
                if len(arg.users) == 0:  # type: ignore[union-attr]
                    model.graph.erase_node(arg)
    # add dequantize node for output
    sm_output_quantized_idxs = observed_standalone_module.meta[
        "_observed_graph_module_attrs"
    ].standalone_module_output_quantized_idxs
    if len(sm_output_quantized_idxs) > 0:
        if sm_output_quantized_idxs[0] != 0:
            raise AssertionError(
                "Currently only quantized output idxs = [0] is supported"
            )

        # if it's non-empty, then it means the output is kept in quantized form
        # we'll just add a dequantize node after this node
        _insert_dequantize_node(node, model.graph)

    # TODO: allow convert_custom_config to override backend_config
    # for standalone module
    quantized_standalone_module = convert_fn(
        observed_standalone_module, backend_config=backend_config
    )
    parent_name, name = _parent_name(node.target)
    # update the modules dict
    setattr(modules[parent_name], name, quantized_standalone_module)
    modules[str(node.target)] = quantized_standalone_module


def convert_weighted_module(
    node: Node,
    modules: dict[str, torch.nn.Module],
    observed_node_names: set[str],
    node_name_to_qconfig: dict[str, QConfigAny],
    backend_config: BackendConfig,
    is_decomposed: bool = False,
    is_reference: bool = False,
    model_device: torch.device | None = None,
) -> None:
    """Convert a weighted module to reference quantized module in the model
    If the QConfig of a QAT module is not set, the module will still be converted to
    a float module.

    Args:
      - node: The call_module node of the observed standalone module
      - modules: named_module of original model
      - observed_node_names: names for the set of observed fx node, we can skip
        this conversion if the node is not observed
    """
    original_module = modules[str(node.target)]
    qconfig: QConfigAny = original_module.qconfig  # type: ignore[assignment]
    weight_post_process = None
    qat_module_classes = get_qat_module_classes(backend_config)

    if isinstance(original_module, qat_module_classes):
        # Converting qat module to a float module, we need to attach
        # weight fake_quant to the module, weight fake_quant is assumed to be run during
        # QAT so we don't need to run it again here
        weight_post_process = original_module.weight_fake_quant
        original_module = original_module.to_float()  # type: ignore[operator]
        # change qat module to float module
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, original_module)

    is_observed = node.name in observed_node_names
    # If a qconfig is not defined for this node, then skip converting to a reference module
    if (
        qconfig is None
        or _has_none_qconfig(node, node_name_to_qconfig)
        or not is_observed
    ):
        return

    # skip converting to reference quantized module if the qconfig is not supported
    pattern_to_dtype_configs = get_pattern_to_dtype_configs(backend_config)
    dtype_configs = pattern_to_dtype_configs.get(type(original_module), [])
    if not _is_qconfig_supported_by_dtype_configs(qconfig, dtype_configs):
        return

    # TODO: rename weight_is_statically_quantized to weight_is_int8_quantized
    is_weight_quantized = weight_is_quantized(qconfig)

    # the condition for swapping the module to reference quantized module is:
    # weights need to be quantized
    if not is_weight_quantized:
        return

    fused_module = None
    float_module = original_module
    # extract the individual float_module and fused module
    if isinstance(original_module, torch.ao.nn.intrinsic._FusedModule):
        fused_module = float_module
        float_module = fused_module[0]  # type: ignore[index]

    # TODO: move this to the reference quantized module
    # weight_qparams or weight_qparams dict
    wq_or_wq_dict = {"is_decomposed": is_decomposed}
    if isinstance(float_module, torch.nn.RNNCellBase):
        weight_post_process_ih = qconfig.weight()  # type: ignore[union-attr, operator]
        weight_post_process_hh = qconfig.weight()  # type: ignore[union-attr, operator]
        weight_post_process_ih(float_module.weight_ih)
        weight_post_process_hh(float_module.weight_hh)
        weight_qparams_ih = get_qparam_dict(weight_post_process_ih)
        weight_qparams_hh = get_qparam_dict(weight_post_process_hh)
        wq_or_wq_dict.update(
            {
                "weight_ih": weight_qparams_ih,
                "weight_hh": weight_qparams_hh,
            }
        )
    elif isinstance(float_module, (torch.nn.LSTM, torch.nn.GRU)):
        # format for wq_or_wq_dict (flattened attributes):
        # {"weight_ih_l0_scale": ..., "weight_ih_l0_qscheme": ..., ...}
        for wn in float_module._flat_weights_names:
            if hasattr(float_module, wn) and wn.startswith("weight"):
                weight = getattr(float_module, wn)
                weight_post_process = qconfig.weight()  # type: ignore[union-attr, operator]
                if weight_post_process.dtype == torch.qint8:  # type: ignore[union-attr]
                    weight_post_process(weight)  # type: ignore[operator, misc]
                wq_or_wq_dict[wn] = get_qparam_dict(weight_post_process)
    else:
        # weight_post_process is None means the original module is not a QAT module
        # we need to get weight_post_process from qconfig in this case
        is_ptq = weight_post_process is None
        if is_ptq:
            weight_post_process = qconfig.weight()  # type: ignore[union-attr, operator]
            if model_device is not None:
                device = model_device
            else:
                device = assert_and_get_unique_device(float_module)
            if device:
                weight_post_process.to(device)

        # Call weight observer/fake_quant at least once to ensure the scales and zero points
        # have the right shapes. Note: there are two cases where we don't have to do this:
        #
        # (1) QAT: The model's forward method already calls the weight observer/fake_quant,
        #     and this typically happens during training, so we don't need to do it here.
        #
        # (2) Non-reference (lowered) case: The quantized module's from_float method already
        #     calls the weight observer/fake_quant, so we don't have to do it here.
        #
        # Currently we ignore both cases and call the weight observer/fake_quant here
        # regardless, which is technically incorrect. For (1), this is mainly to preserve BC
        # in test code, which may not always train before convert. In the future, we should
        # break BC for these two cases. See https://github.com/pytorch/pytorch/issues/73941.
        #
        # For PT2, however, we don't need to preserve BC here, so we can skip this hack
        # for QAT. We identify this case as (is_decomposed + is_reference + is_qat).
        # Note that we still need it for PTQ in the PT2 flow since the model's forward
        # method doesn't call the weight observer.
        is_qat = not is_ptq
        if not (is_decomposed and is_reference and is_qat):
            weight_post_process(float_module.weight)  # type: ignore[operator]

        wq_or_wq_dict.update(get_qparam_dict(weight_post_process))

    # We use the same reference module for all modes of quantization: static, dynamic, weight_only
    # root_module_to_quantized_reference_module: module mapping from root (floating point) module class
    # to quantized reference module class, e.g. nn.Conv2d to nn.quantized._reference.Conv2d
    root_module_to_quantized_reference_module = (
        get_root_module_to_quantized_reference_module(backend_config)
    )
    ref_qmodule_cls = root_module_to_quantized_reference_module.get(
        type_before_parametrizations(float_module), None
    )
    if ref_qmodule_cls is None:
        raise AssertionError(
            f"No reference quantized module class configured for {type_before_parametrizations(float_module)}"
        )
    ref_qmodule = ref_qmodule_cls.from_float(float_module, wq_or_wq_dict)  # type: ignore[attr-defined]
    if fused_module is not None:
        fused_module[0] = ref_qmodule  # type: ignore[operator]
    else:
        parent_name, name = _parent_name(node.target)
        setattr(modules[parent_name], name, ref_qmodule)


def _remove_previous_dequantize_in_custom_module(
    node: Node, prev_node: Node, graph: Graph
) -> None:
    """
    Given a custom module `node`, if the previous node is a dequantize, reroute the custom as follows:

    Before: quantize - dequantize - custom_module
    After: quantize - custom_module
                 \\ - dequantize
    """
    # expecting the input node for a custom module node to be a Node
    if not isinstance(prev_node, Node):
        raise AssertionError(
            f"Expecting the argument for custom module node to be a Node, but got {prev_node}"
        )
    if prev_node.op == "call_method" and prev_node.target == "dequantize":
        node.replace_input_with(prev_node, prev_node.args[0])
        # Remove the dequantize node if it doesn't have other users
        if len(prev_node.users) == 0:
            graph.erase_node(prev_node)


def convert_custom_module(
    node: Node,
    graph: Graph,
    modules: dict[str, torch.nn.Module],
    custom_module_class_mapping: dict[QuantType, dict[type, type]],
    statically_quantized_custom_module_nodes: set[Node],
) -> None:
    """Converts an observed custom module to a quantized custom module based on
    `custom_module_class_mapping`
    For static quantization, we'll also remove the previous `dequantize` node and
    attach the observer node for output to the module, the observer for the node
    will be converted to a dequantize node instead of quantize-dequantize pairs
    later in the graph. In the end we would have a quantized custom module that
    has the same interface as a default quantized module in nn.quantized namespace,
    i.e. quantized input and quantized output.

    Args:
      - node: The call_module node of the observed standalone module
      - graph: The graph containing the node
      - modules: named_module of original model
      - custom_module_class_mapping: mapping from observed custom module class to
        quantized custom module class, used to swap custom modules
      - statically_quantized_custom_module_nodes: we'll add the custom module node
        if we find it is statically quantized, this will be used later when converting
        observers to quant/dequant node pairs, if the observed node is a statically
        quantized custom module nodes, we'll convert the observer to a dequantize node,
        this is to keep the interface the same as the default quantized module.
        TODO: maybe we want to redesign this part to align with reference model design
        as well, but there has been some discussions around the interface, so we can do
        it later.
    """
    observed_custom_module = modules[str(node.target)]
    qconfig = observed_custom_module.qconfig
    if activation_is_statically_quantized(qconfig):
        statically_quantized_custom_module_nodes.add(node)
        if _is_custom_module_lstm(node, modules):
            # The inputs are tuples in the form (input, (hidden0, hidden1))
            # Ensure all three input nodes are quantized
            if not (
                len(node.args) == 2
                and isinstance(node.args[1], tuple)
                and len(node.args[1]) == 2
            ):
                raise AssertionError(
                    "Expected LSTM custom module inputs to be (input, (hidden0, hidden1))"
                )
            (inputs, (hidden0, hidden1)) = node.args  # type: ignore[misc]
            if not isinstance(inputs, Node):
                raise AssertionError("Expected inputs to be a Node")
            if not isinstance(hidden0, Node):
                raise AssertionError("Expected hidden0 to be a Node")
            if not isinstance(hidden1, Node):
                raise AssertionError("Expected hidden1 to be a Node")
            _remove_previous_dequantize_in_custom_module(node, inputs, graph)
            _remove_previous_dequantize_in_custom_module(node, hidden0, graph)
            _remove_previous_dequantize_in_custom_module(node, hidden1, graph)
        elif _is_custom_module_mha(node, modules):
            # Inputs are in the form (query, key, value)
            # TODO: This is the first step in enabling the full fx custom module
            # quantization path for MultiheadAttention, and only covers the inputs
            # to the module.
            # Additional handling is yet to be implemented for the outputs, similar
            # to LSTM custom module
            if len(node.args) != 3:
                raise AssertionError(
                    "Expected MHA custom module inputs to be (query, key, value)"
                )
            query, key, value = node.args
            if not isinstance(query, Node):
                raise AssertionError("Expected query to be a Node")
            if not isinstance(key, Node):
                raise AssertionError("Expected key to be a Node")
            if not isinstance(value, Node):
                raise AssertionError("Expected value to be a Node")
            _remove_previous_dequantize_in_custom_module(node, query, graph)
            _remove_previous_dequantize_in_custom_module(node, key, graph)
            _remove_previous_dequantize_in_custom_module(node, value, graph)
        else:
            # remove the previous dequant node to ensure the inputs are quantized
            arg = node.args[0]
            if not isinstance(arg, Node):
                raise AssertionError("Expected arg to be a Node")
            _remove_previous_dequantize_in_custom_module(node, arg, graph)
            # absorb the following observer into the module conversion
            activation_post_process = _maybe_get_observer_for_node(node, modules)
            if activation_post_process is None:
                raise AssertionError(
                    "Expected activation_post_process to be present for observed custom module"
                )
            observed_custom_module.activation_post_process = activation_post_process

    # swap the observed custom module to quantized custom module
    quantized_custom_module_class = get_swapped_custom_module_class(
        observed_custom_module, custom_module_class_mapping, qconfig
    )
    quantized_custom_module = quantized_custom_module_class.from_observed(
        observed_custom_module
    )
    parent_name, name = _parent_name(node.target)
    setattr(modules[parent_name], name, quantized_custom_module)


def convert(
    model: GraphModule,
    is_reference: bool = False,
    convert_custom_config: ConvertCustomConfig | dict[str, Any] | None = None,
    is_standalone_module: bool = False,
    _remove_qconfig_flag: bool = True,
    qconfig_mapping: QConfigMapping | dict[str, Any] | None = None,
    backend_config: BackendConfig | dict[str, Any] | None = None,
    is_decomposed: bool = False,
    keep_original_weights: bool = False,
) -> GraphModule:
    """
    We will convert an observed model (a module with observer calls) to a reference
    quantized model, the rule is simple:
    1. for each observer module call in the graph, we'll convert it to calls to
       quantize and dequantize functions based on the observer instance
    2. for weighted operations like linear/conv, we need to convert them to reference
       quantized module, this requires us to know whether the dtype configured for the
       weight is supported in the backend, this is done in prepare step and the result
       is stored in observed_node_names, we can decide whether we need to swap the
       module based on this set

    Args:
       * `is_standalone_module`: when this flag is True, it means we are quantizing
       a submodule that is not inlined in parent module, and will be quantized
       separately as one unit.

       * `is_decomposed`: a boolean flag to indicate whether we want to use the
        quantize operator for decomposed quantized tensor
        (torch.ops.quantized_decomposed.quantize_per_tensor) or default/standalone
        quantized tensor (torch.quantize_per_tensor)

    Returns:
         a quantized standalone module, whether input/output is quantized is
         specified by prepare_custom_config, with
         input_quantized_idxs, output_quantized_idxs, please
         see docs for :func:`~torch.ao.quantization.prepare_fx` for details
    """
    if convert_custom_config is None:
        convert_custom_config = ConvertCustomConfig()

    if isinstance(convert_custom_config, dict):
        warnings.warn(
            "Passing a convert_custom_config_dict to convert is deprecated and will not be supported "
            "in a future version. Please pass in a ConvertCustomConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        convert_custom_config = ConvertCustomConfig.from_dict(convert_custom_config)

    if isinstance(qconfig_mapping, dict):
        warnings.warn(
            "Passing a QConfig dictionary to convert is deprecated and will not be supported "
            "in a future version. Please pass in a QConfigMapping instead.",
            FutureWarning,
            stacklevel=2,
        )
        qconfig_mapping = (
            QConfigMapping.from_dict(qconfig_mapping) if qconfig_mapping else None
        )
    qconfig_mapping = copy.deepcopy(qconfig_mapping)
    if not (qconfig_mapping is None or isinstance(qconfig_mapping, QConfigMapping)):
        raise AssertionError("qconfig_mapping must be None or a QConfigMapping")

    if isinstance(backend_config, dict):
        warnings.warn(
            "Passing a backend_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a BackendConfig instead.",
            FutureWarning,
            stacklevel=2,
        )
        backend_config = BackendConfig.from_dict(backend_config)

    if backend_config is None:
        backend_config = get_native_backend_config()

    if not _is_observed_module(model):
        raise AssertionError("incoming model must be produced by prepare_fx")
    observed_graph_module_attrs = model.meta["_observed_graph_module_attrs"]
    node_name_to_scope: dict[str, tuple[str, type]] = (
        observed_graph_module_attrs.node_name_to_scope
    )
    prepare_custom_config: PrepareCustomConfig = (
        observed_graph_module_attrs.prepare_custom_config
    )
    observed_node_names: set[str] = observed_graph_module_attrs.observed_node_names
    node_name_to_qconfig: dict[str, QConfigAny] = (
        observed_graph_module_attrs.node_name_to_qconfig
    )  # type: ignore[assignment]

    # mapping from fully qualified module name to module instance
    # for example,
    # {
    #   '': Model(...),
    #   'linear': Linear(...),
    #   'linear.weight_fake_quant': PerChannelMinMaxObserver(...),
    # }
    # We use remove_duplicate=False here because torch.cat uses
    # the same activation_post_process module instance but different names
    modules = dict(model.named_modules(remove_duplicate=False))

    # TODO refactor this code once we update the prepare logic to have additional information on
    # which graph nodes have been observed and share that with convert to decide which observers to ignore.
    if qconfig_mapping:
        prepare_qconfig_mapping: QConfigMapping = (
            observed_graph_module_attrs.qconfig_mapping
        )  # type: ignore[assignment]
        modules_copy = copy.deepcopy(modules)

        if observed_graph_module_attrs.is_qat:
            _update_qconfig_for_qat(qconfig_mapping, backend_config)
        _update_qconfig_for_fusion(model, qconfig_mapping)

        _compare_prepare_convert_qconfig_mappings(
            prepare_qconfig_mapping, qconfig_mapping
        )  # type: ignore[arg-type]
        convert_node_name_to_qconfig = _generate_node_name_to_qconfig(
            model, modules_copy, model.graph, qconfig_mapping, node_name_to_scope
        )
        # check the convert_node_name_to_qconfig generated and ensure that
        # all the values either match what was set in prepare node_name_to_qconfig
        # or are set to None in the convert_node_name_to_qconfig.
        for k, v in node_name_to_qconfig.items():
            if k not in convert_node_name_to_qconfig:
                raise AssertionError(
                    f"Expected key {k} in convert node_name_to_qconfig"
                )
            if convert_node_name_to_qconfig[k] is not None:
                if not qconfig_equals(v, convert_node_name_to_qconfig[k]):
                    raise AssertionError(
                        f"Expected k {k} to have the same value in prepare and convert QConfigMappings, "
                        f"but {v} was updated to {convert_node_name_to_qconfig[k]}"
                    )
        node_name_to_qconfig = convert_node_name_to_qconfig

    custom_module_classes = get_custom_module_class_keys(
        convert_custom_config.observed_to_quantized_mapping
    )
    custom_module_class_mapping = convert_custom_config.observed_to_quantized_mapping

    if observed_graph_module_attrs.equalization_node_name_to_qconfig is not None:
        # If we want to do equalization then do the following:
        # Calculate the equalization scale, update the observers with the scaled
        # inputs, and scale the weight
        weight_eq_obs_dict = update_obs_for_equalization(model, modules)
        convert_eq_obs(model, modules, weight_eq_obs_dict)

    # always run weight observers in the top level forward method
    # for dynamic quant ops or weight only quant ops
    _run_weight_observers(model, backend_config)

    # additional state to override inputs to be quantized, if specified
    # by the user
    placeholder_node_seen_cnt = 0
    input_quantized_idxs: list[int] = prepare_custom_config.input_quantized_indexes
    output_quantized_idxs: list[int] = prepare_custom_config.output_quantized_indexes

    root_module_to_quantized_reference_module = (
        get_root_module_to_quantized_reference_module(backend_config)
    )
    # convert tuples so that it can work with isinstance(module, tuple_of_classes)
    root_module_classes = tuple(root_module_to_quantized_reference_module.keys())
    qat_module_classes = get_qat_module_classes(backend_config)
    fused_module_classes = get_fused_module_classes(backend_config)
    statically_quantized_custom_module_nodes: set[Node] = set()
    model_device = assert_and_get_unique_device(model)

    for node in list(model.graph.nodes):
        if node.op == "placeholder":
            cur_placeholder_node_idx = placeholder_node_seen_cnt
            placeholder_node_seen_cnt += 1
            if cur_placeholder_node_idx in input_quantized_idxs:
                # Inputs are assumed to be quantized if the user specified the
                # input_quantized_idxs override.
                # we need to dequantize the inputs since all operators took
                # floating point inputs in reference quantized models
                _insert_dequantize_node(node, model.graph)
        elif node.op == "output":
            # If the argument is empty we don't need to do anything
            if len(output_quantized_idxs) == 0:
                continue
            # Result are kept quantized if the user specified the
            # output_quantized_idxs override.
            # Remove the dequantize operator for the node in the end if any
            return_node = node
            output = node.args[0]
            # outputs can be Node, list, tuple, dict, other cases are not supported yet
            if isinstance(output, (list, tuple)):
                for idx in output_quantized_idxs:
                    _maybe_recursive_remove_dequantize(
                        output[idx], return_node, model.graph
                    )
            elif isinstance(output, (Node, dict)):
                # we treat dict as a single argument currently, but it can be extended
                # to support {"key": dtype} after we change output_quantized_idxs to
                # dict
                if 0 in output_quantized_idxs:
                    _maybe_recursive_remove_dequantize(output, return_node, model.graph)
            else:
                warnings.warn(
                    f"Unsupported node type for output_quantized_idxs: {type(output)}",
                    stacklevel=2,
                )
        elif node.op == "call_module":
            mod = _get_module(node, modules)
            if mod is None:
                raise AssertionError(
                    "Expected module for call_module node to be present in modules mapping"
                )
            if _is_activation_post_process(mod):
                observed_node = node.args[0]
                if observed_node in statically_quantized_custom_module_nodes:
                    _replace_observer_or_dequant_stub_with_dequantize_node(
                        node, model.graph
                    )
                else:
                    if is_decomposed:
                        _replace_observer_with_quantize_dequantize_node_decomposed(
                            model,
                            node,
                            modules,
                            node_name_to_scope,
                            node_name_to_qconfig,
                            model_device,
                        )
                    else:
                        _replace_observer_with_quantize_dequantize_node(
                            model,
                            node,
                            modules,
                            node_name_to_scope,
                            node_name_to_qconfig,
                            model_device,
                        )
            elif isinstance(mod, DeQuantStub):
                _replace_observer_or_dequant_stub_with_dequantize_node(
                    node, model.graph
                )
            elif _is_observed_standalone_module(mod):
                convert_standalone_module(
                    node, modules, model, is_reference, backend_config
                )
            # below this point `type_before_parametrizations` is used
            # instead of `type` to handle situations with fx quant + sparsity
            elif type_before_parametrizations(mod) in set(root_module_classes).union(
                qat_module_classes
            ).union(fused_module_classes):
                # extra check for fused module classes to make sure they are fused module classes
                # of target modules
                if (
                    type_before_parametrizations(mod) in fused_module_classes
                    and type_before_parametrizations(mod[0]) not in root_module_classes
                ):  # type: ignore[index]
                    continue
                convert_weighted_module(
                    node,
                    modules,
                    observed_node_names,
                    node_name_to_qconfig,
                    backend_config,
                    is_decomposed,
                    is_reference,
                    model_device,
                )
            elif type_before_parametrizations(mod) in custom_module_classes:
                convert_custom_module(
                    node,
                    model.graph,
                    modules,
                    custom_module_class_mapping,
                    statically_quantized_custom_module_nodes,
                )

    # remove deadcode after converting observers to quant/dequant ops
    model.graph.eliminate_dead_code()
    model = GraphModule(model, model.graph)

    # TODO: maybe move this to quantize_fx.py
    if not is_reference:
        model = lower_to_fbgemm(
            model, node_name_to_qconfig, node_name_to_scope, keep_original_weights
        )

    # TODO: this looks hacky, we want to check why we need this and see if we can
    # remove this
    # removes qconfig and activation_post_process modules
    if _remove_qconfig_flag:
        _remove_qconfig(model)
    model.delete_all_unused_submodules()
    model.meta.pop("_observed_graph_module_attrs", None)
    return model
<|endoftext|># mypy: allow-untyped-defs
import math

import torch
from torch._refs import _unsqueeze_multiple
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from torch.library import impl, Library


# Note: decomposed means decomposed quantized tensor, using decomposed so that the
# name is not too long
quantized_decomposed_lib = Library("quantized_decomposed", "DEF")

_INTEGER_DTYPES = [torch.uint8, torch.int8, torch.uint16, torch.int16, torch.int32]
_FLOAT_DTYPES = [torch.float8_e5m2, torch.float8_e4m3fn]

_DTYPE_TO_QVALUE_BOUNDS = {
    k: (torch.iinfo(k).min, torch.iinfo(k).max) for k in _INTEGER_DTYPES
}
_DTYPE_TO_QVALUE_BOUNDS.update(
    {k: (int(torch.finfo(k).min), int(torch.finfo(k).max)) for k in _FLOAT_DTYPES}
)


# Helper to check the passed in quant min and max are valid for the dtype
def _quant_min_max_bounds_check(quant_min, quant_max, dtype):
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f"Unsupported dtype: {dtype}")
    quant_min_lower_bound, quant_max_upper_bound = _DTYPE_TO_QVALUE_BOUNDS[dtype]

    if quant_min < quant_min_lower_bound:
        raise AssertionError(
            "quant_min out of bounds for dtype, "
            f"quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}"
        )

    if quant_max > quant_max_upper_bound:
        raise AssertionError(
            "quant_max out of bounds for dtype, "
            f"quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}"
        )


quantized_decomposed_lib.define(
    "quantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(quantized_decomposed_lib, "quantize_per_tensor", "CompositeExplicitAutograd")
def quantize_per_tensor(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values

    Args:
       input (torch.Tensor): original float32 or bfloat16 Tensor
       scale (float): quantization parameter for affine quantization
       zero_point (int): quantization parameter for affine quantization
       quant_min (int): minimum quantized value for output Tensor
       quant_max (int): maximum quantized value for output Tensor
       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor

    Returns:
       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters
       are not stored in the Tensor, we are storing them in function arguments instead
    """
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    if input.dtype != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        )
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)

    inv_scale = 1.0 / scale
    return torch.clamp(
        torch.round(input * inv_scale) + zero_point, quant_min, quant_max
    ).to(dtype)


@impl(quantized_decomposed_lib, "quantize_per_tensor", "Meta")
def quantize_per_tensor_meta(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    if input.dtype != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        )
    return torch.empty_like(input, dtype=dtype)


quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib, "quantize_per_tensor.tensor", "CompositeExplicitAutograd"
)
def quantize_per_tensor_tensor(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return quantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min,  # type: ignore[arg-type]
        quant_max,  # type: ignore[arg-type]
        dtype,
    )


@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor", "Meta")
def quantize_per_tensor_tensor_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if input.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(torch.float32)
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    if input.dtype != torch.float32:
        raise AssertionError(
            f"Expecting input to have dtype torch.float32, but got dtype: {input.dtype}"
        )
    return torch.empty_like(input, dtype=dtype)


# TODO: remove other variants and keep this one
quantized_decomposed_lib.define(
    "quantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, "
    "Tensor quant_min, Tensor quant_max, ScalarType dtype) -> Tensor"
)


@impl(
    quantized_decomposed_lib, "quantize_per_tensor.tensor2", "CompositeExplicitAutograd"
)
def quantize_per_tensor_tensor2(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: torch.Tensor,
    quant_max: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Affine quantization for the Tensor using the same quantization parameters to map
    from floating point to quantized values
    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return quantize_per_tensor(
        input,
        scale.item(),
        zero_point.item(),  # type: ignore[arg-type]
        quant_min.item(),  # type: ignore[arg-type]
        quant_max.item(),  # type: ignore[arg-type]
        dtype,
    )


@impl(quantized_decomposed_lib, "quantize_per_tensor.tensor2", "Meta")
def quantize_per_tensor_tensor2_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: torch.Tensor,
    quant_max: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return quantize_per_tensor_tensor_meta(
        input,
        scale,
        zero_point,  # type: ignore[arg-type]
        quant_min,  # type: ignore[arg-type]
        quant_max,  # type: ignore[arg-type]
        dtype,
    )


# Note: quant_min/quant_max/dtype are not used in the operator, but for now it's kept in
# the signature as metadata for the input Tensor, this might be useful for pattern
# matching in the future
# We will revisit this later if we found there are no use cases for it
quantized_decomposed_lib.define(
    "dequantize_per_tensor(Tensor input, float scale, int zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(quantized_decomposed_lib, "dequantize_per_tensor", "CompositeExplicitAutograd")
def dequantize_per_tensor(
    input: torch.Tensor,
    scale: float,
    zero_point: int,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values

    Args:
       input (torch.Tensor): Tensor with dtype matching `dtype` argument,
       e.g. (`torch.uint8`), it is a per tensor quantized Tensor if combined with
       quantization parameters in the argument of this function (scale/zero_point)

       scale (float): quantization parameter for affine quantization

       zero_point (int): quantization parameter for affine quantization

       quant_min (int): minimum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       quant_max (int): maximum quantized value for input Tensor (not used in computation,
       reserved for pattern matching)

       dtype (torch.dtype): dtype for input Tensor (not used in computation,
       reserved for pattern matching)

       out_dtype (torch.dtype?): optional dtype for output Tensor

    Returns:
       dequantized float32 Tensor
    """
    if input.dtype != dtype:
        raise AssertionError(
            f"Expecting input to have dtype: {dtype}, but got {input.dtype}"
        )
    if out_dtype is None:
        out_dtype = torch.float32
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        # TODO: investigate why
        # (input - zero_point).to(torch.float32) * scale
        # failed the test
        return (input.to(out_dtype) - zero_point) * scale
    else:
        raise ValueError(f"Unsupported dtype in dequantize_per_tensor: {dtype}")


@impl(quantized_decomposed_lib, "dequantize_per_tensor", "Meta")
def dequantize_per_tensor_meta(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if out_dtype is None:
        out_dtype = torch.float32
    return torch.empty_like(input, dtype=out_dtype)


quantized_decomposed_lib.define(
    "dequantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, "
    "int quant_min, int quant_max, ScalarType dtype, *, ScalarType? out_dtype=None) -> Tensor"
)


@impl(
    quantized_decomposed_lib,
    "dequantize_per_tensor.tensor",
    "CompositeExplicitAutograd",
)
def dequantize_per_tensor_tensor(
    input: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: torch.dtype,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Affine dequantization for the Tensor using the same quantization parameters to map
    from quantized values to floating point values
    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of
    scalar values
    """
    if zero_point.numel() != 1:
        raise AssertionError(
            f"Expecting zero_point tensor to be one element, but received : {zero_point.numel()}"
        )
    if scale.numel() != 1:
        raise AssertionError(
            f"Expecting scale tensor to be one element, but received : {scale.numel()}"
        )
    return dequantize_per_tensor(
