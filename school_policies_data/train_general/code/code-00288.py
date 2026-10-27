                for g in new_guards_code
            ]

    return new_guards_code


def _convert_guards_code_to_fn(
    guards_code: list[str],
    paths_of_placeholders: list[pytree.KeyPath],
):
    """
    Generates Python code given guards code and paths of placeholders.
    We assume that, based on source information,
    - the tracer generates the guards code
    - the input spec generates the paths of placeholders.

    Example:

    Suppose we are given the guards code "L['z']['k'].size()[1] == 3"
    and we are given that ['z']['k'] is the path of placeholder #2.
    Then we will generate:
    ```
    torch._assert(
        args[2].size()[0] == 3,
        "Guard failed: z['k'].size()[0] == 3",
    )
    ```

    FAQ: Why do we generate code based on (flattened) args instead of
    the original (unflattened) inputs? Because this would require
    inserting an additional pytree.unflatten call in our graph.

    FAQ: Why do we not emit RuntimeError on guard failure as we used to?
    Because it is inconvenient :/, get used to AssertionError instead.
    """

    import ast

    from torch.fx.experimental.symbolic_shapes import SYMPY_INTERP

    actual_guards_code = []
    shadow_guards_code = []
    for c in guards_code:
        a, s = c, c
        for idx, path in enumerate(paths_of_placeholders):
            # e.g., replace L['z']['k'] with args[2] for Python code (actual)
            a = a.replace("L" + pytree.keystr(path), f"args[{idx}]")
            # e.g., replace L['z']['k'] with z['k'] for error message (shadow)
            s = s.replace(
                "L" + pytree.keystr(path),
                path[0].key + pytree.keystr(path[1:]),  # type: ignore[attr-defined]
            )
        actual_guards_code.append(a)
        shadow_guards_code.append(s.replace("\n", ""))

    # generate function code as str
    code_str = "\ndef _(*args):\n"
    for actual, shadow in zip(actual_guards_code, shadow_guards_code):
        # printing guards code may potentially introduce redundant parens;
        # we can normalize them out for readability by parsing/unparsing
        # NOTE: this is not necessary for correctness, just deemed desirable
        try:
            _shadow = ast.unparse(ast.parse(shadow, mode="eval"))
        except RecursionError as e:
            # A deeply nested guard expression (e.g. a sum over many symbolic
            # sizes) can exceed the recursion limit in ast.parse/ast.unparse.
            # This normalization only affects the assert error message, so fall
            # back to the un-normalized guard string instead of crashing.
            warnings.warn(f"ast.unparse failed for guard expression: {e}", stacklevel=2)
            _shadow = shadow
        # actual code and shadow error message
        code_str += f'  torch._assert({actual}, "Guard failed: {_shadow}")\n'
    code_str += "  return\n"

    # populate namespace with sympy globals, materialize function (named `_`)
    namespace = {
        **SYMPY_INTERP,
        "math": math,
        "inf": float("inf"),
    }
    exec(code_str, namespace)

    # create and return a module whose forward is the materialized function
    # NOTE: we want Dynamo to trace through this module, to repopulate guards:
    # otherwise we would lose them when retracing
    # NOTE: calling this module will be a side effect (no users): so it must
    # be marked impure to avoid being not cleaned up by DCE
    guards_fn = GuardsFn()
    guards_fn.forward = torch._dynamo.dont_skip_tracing(namespace["_"])  # type: ignore[call-overload, method-assign]
    guards_fn._is_impure = True  # type: ignore[assignment]
    return guards_fn


@torch._dynamo.disable
def _check_input_constraints_for_module(self, args, kwargs):
    flat_args_with_path = _check_inputs_match(args, kwargs, self._in_spec)
    _check_input_constraints_for_graph(
        self.graph.find_nodes(op="placeholder"),
        flat_args_with_path,
        self.range_constraints,
    )


def _check_input_constraints_pre_hook(self, args, kwargs):
    # preserve current behavior for clients that do not want any validation
    if not self.validate_inputs:
        return

    # when a guards function exists, assume that the graph does calls it!
    # so we do not need to check input constraints...but we still want
    # to check inputs match, otherwise we'd get obscure pytree errors
    if hasattr(self, "_guards_fn"):
        _check_inputs_match(args, kwargs, self._in_spec)
        return

    # NOTE: for some reason, Dynamo is tracing into this, we should see why and
    # put compile at the right place. Until then, we can skip the input
    # constraint checks.
    if not torch.compiler.is_dynamo_compiling():
        _check_input_constraints_for_module(self, args, kwargs)


def _unlift_inputs_as_getattr(
    gm: torch.fx.GraphModule,
    lifted_inputs: Sequence[str | None],
) -> tuple[dict[str, torch.fx.Node], dict[str, torch.fx.Node]]:
    """
    Unlift inputs referring to params/buffers/constants as getattr nodes in the
    graph
    """
    unlifted_name_to_node = {}
    input_name_to_node = {}

    placeholder_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(lifted_inputs) != len(placeholder_nodes):
        raise AssertionError(
            f"Number of lifted inputs ({len(lifted_inputs)}) does not match "
            f"placeholder nodes ({len(placeholder_nodes)})"
        )
    for input_node, lifted_node in zip(placeholder_nodes, lifted_inputs):
        if lifted_node is None:
            input_name_to_node[input_node.name] = input_node

        else:
            with gm.graph.inserting_after(input_node):
                # It is fine to ignore this warning because
                # it is guaranteed that we will populate this
                # attr later.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    getattr_node = gm.graph.get_attr(lifted_node)
                input_node.replace_all_uses_with(getattr_node)
                metadata = input_node.meta
                gm.graph.erase_node(input_node)
                getattr_node.meta = metadata
                getattr_node.meta["from_node"] = [
                    NodeSource(
                        input_node,
                        "ExportedProgram.module().unlift()",
                        [NodeSourceAction.CREATE, NodeSourceAction.REPLACE],
                    )
                ]
                unlifted_name_to_node[lifted_node] = getattr_node

    return unlifted_name_to_node, input_name_to_node


def _insert_copy_for_mutations(
    gm: torch.fx.GraphModule,
    mutated_outputs: Sequence[str | None],
    unlifted_name_to_node: dict[str, torch.fx.Node],
    input_name_to_node: dict[str, torch.fx.Node],
) -> None:
    """
    Find the all the buffers and inputs that were mutated and insert copy_
    operators to reflect mutations.
    """
    output_node = gm.graph.output_node()
    outputs = pytree.tree_flatten(output_node.args)[0]
    if len(outputs) != len(mutated_outputs):
        raise AssertionError(
            f"Number of outputs ({len(outputs)}) does not match "
            f"mutated outputs ({len(mutated_outputs)})"
        )

    user_output_nodes = []
    return_nodes_to_copy = {}
    for return_node, mutated_node_name in zip(outputs, mutated_outputs):
        if mutated_node_name is None:
            user_output_nodes.append(return_node)
            continue

        if mutated_node_name in unlifted_name_to_node:
            mutated_node = unlifted_name_to_node[mutated_node_name]
        elif mutated_node_name in input_name_to_node:
            mutated_node = input_name_to_node[mutated_node_name]
        else:
            raise RuntimeError(
                f"Could not find {mutated_node_name} in either buffer or input nodes"
            )

        with gm.graph.inserting_before(output_node):
            copy_node = gm.graph.call_function(
                torch.ops.aten.copy_.default, (mutated_node, return_node)
            )
            return_nodes_to_copy[return_node] = copy_node

    output_args = tuple(
        return_nodes_to_copy.get(node, node) for node in user_output_nodes
    )
    with gm.graph.inserting_before(output_node):
        # Only return user outputs
        new_output = gm.graph.output(output_args)
        output_node.replace_all_uses_with(new_output)
        gm.graph.erase_node(output_node)
        new_output.name = output_node.name
        new_output.meta.update(output_node.meta)
        new_output.meta["from_node"] = [
            NodeSource(
                output_node,
                "ExportedProgram.module().unlift()",
                [NodeSourceAction.CREATE, NodeSourceAction.REPLACE],
            )
        ]


def _get_codegen(
    in_spec: pytree.TreeSpec,
    out_spec: pytree.TreeSpec | None,
    forward_arg_names: list[str] | None = None,
) -> _PyTreeCodeGen:
    """
    Create the codegen for the graph module based on the in/out specs
    """
    if forward_arg_names:
        names = forward_arg_names
    elif (
        in_spec.type is tuple
        and in_spec.num_children == 2
        and in_spec.child(0).type is tuple
        and in_spec.child(1).type is dict
    ):
        # if in_spec contains the args (tuple) and kwargs (dict)
        names = [f"arg_{i}" for i in range(in_spec.child(0).num_children)]
        # add kwarg names
        names.extend(in_spec.child(1).context)
    else:
        names = [f"arg_{i}" for i in range(in_spec.num_children)]

    return _PyTreeCodeGen(
        _PyTreeInfo(
            names,
            in_spec,
            out_spec,
        )
    )


def _unlift(
    gm: torch.fx.GraphModule,
    lifted_inputs: Sequence[str | None],
    mutated_outputs: Sequence[str | None],
    in_spec: pytree.TreeSpec,
    out_spec: pytree.TreeSpec | None,
    forward_arg_names: list[str] | None = None,
):
    """
    Args:
        lifted_inputs: A list matching the graph module's input nodes. For
        an input node that is referring to a lifted parameter/buffer, this
        list will contain the fqn the corresponding attribute. Otherwise, this
        list will contain None. This is used to unlift the lifted parameters as
        get_attr nodes.

        mutated_outputs: A list matching the graph module's output nodes. For
        an output node that is referring to a mutated buffer or user input, this
        list will contain the name of the corresponding buffer or user input
        that needs to be mutated. Otherwise, this list will contain None. This
        is used to re-insert an inplace copy_ operator to copy the mutated
        values back to the original node.
    """
    unlifted_name_to_node, input_name_to_node = _unlift_inputs_as_getattr(
        gm, lifted_inputs
    )
    _insert_copy_for_mutations(
        gm, mutated_outputs, unlifted_name_to_node, input_name_to_node
    )
    gm.graph._codegen = _get_codegen(in_spec, out_spec, forward_arg_names)
    gm.graph.lint()
    gm.recompile()
    return gm


def _register_attrs_to_new_gm(
    new_gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    state_dict: dict[str, Any],
    constants: dict[str, Any],
) -> None:
    non_persistent_buffers = set(graph_signature.non_persistent_buffers)
    for name in graph_signature.buffers:
        if name in non_persistent_buffers:
            persistent = False
            value = constants[name]
        else:
            persistent = True
            value = state_dict[name]
        _assign_attr(
            value, new_gm, name, attr_kind=_AttrKind.BUFFER, persistent=persistent
        )
    for name in graph_signature.parameters:
        value = state_dict[name]
        _assign_attr(
            value,
            new_gm,
            name,
            attr_kind=_AttrKind.PARAMETER,
        )

    # Technically this doesn't account for the aliased multiple constants but
    # it is ok because we have a separate pass later in the stack that populates
    # the final gm.
    for name in chain(
        graph_signature.lifted_custom_objs, graph_signature.lifted_tensor_constants
    ):
        value = constants[name]
        _assign_attr(
            value,
            new_gm,
            name,
            attr_kind=_AttrKind.CONSTANT,
        )


class _StatefulGraphModuleFactory(type):
    """
    Metaclass that ensures a private constructor for _StatefulGraphModule
    """

    def __call__(cls, *args, **kwargs):
        raise TypeError(
            f"{cls.__module__}.{cls.__qualname__} has no public constructor. "
        )

    def _create(cls, root, graph, range_constraints=None):
        return super().__call__(
            root,
            graph,
            range_constraints=range_constraints,
        )


class _StatefulGraphModule(torch.fx.GraphModule, metaclass=_StatefulGraphModuleFactory):
    def __init__(self, root, graph, range_constraints=None):
        super().__init__(root, graph)
        # Need to fix up non-persistent buffers.
        self.range_constraints = range_constraints or []
        self.validate_inputs = True


def _create_stateful_graph_module(
    plain_graph_module: torch.fx.GraphModule,
    range_constraints,
    ep: ExportedProgram,
) -> _StatefulGraphModule:
    stateful_gm = _StatefulGraphModule._create(
        plain_graph_module,
        plain_graph_module.graph,
        range_constraints=range_constraints,
    )

    module_types = _get_graph_inputs_of_type_nn_module(ep.example_inputs)
    stateful_gm.register_forward_pre_hook(
        lambda *args, **kwargs: _enter_enable_graph_inputs_of_type_nn_module(
            module_types
        )
    )
    stateful_gm.register_forward_pre_hook(
        _check_input_constraints_pre_hook, with_kwargs=True
    )

    stateful_gm.register_forward_hook(
        lambda *args, **kwargs: _exit_enable_graph_inputs_of_type_nn_module(
            module_types
        ),
        always_call=True,
    )

    # When we have a constant that has requires_grad=True, we need to detach it
    # when we unlift as the tensors that require gradients should be registered
    # via parameters. But this is problematic when we have aliasing two constants
    # because when we call detach, they will become different tensors. This dict
    # keeps track of this logic.
    original_tensor_to_detached_tensor = {}

    # Fix up lifted tensor constants.
    # fx.GraphModule() constructor silently turns a constant attribute of plain_graph_module
    # into a buffer in stateful_gm and creates an inconsistency with graph_signature.
    # We fix this by de-registering these buffers in lifted_tensor_constants
    # and call _assign_attr(attr_kind=CONSTANT) to register them as constants.
    for constant_fqn in ep.graph_signature.lifted_tensor_constants:
        # Sometimes, the constant can require gradient, this is probably a bug in user code,
        # e.g. `self.const = torch.randn(2, 2, requires_grad=True)`.
        # We call detach on the constant_val since they're tensor constants and we don't need to
        # compute their gradients anyway.
        # Users should properly register it as parameter if they want it to require gradient.
        buffer = stateful_gm.get_buffer(constant_fqn)
        if buffer.requires_grad:
            warnings.warn(
                f"A model attribute `{constant_fqn}` requires gradient. "
                f"but it's not properly registered as a parameter. "
                f"torch.export will detach it and treat it as a constant tensor "
                f"but please register it as parameter instead.",
                stacklevel=2,
            )
            detached_buffer = buffer.detach()
            original_tensor_to_detached_tensor[buffer] = detached_buffer
            buffer = detached_buffer
        *prefix, field = constant_fqn.rsplit(".")
        submod = torch.fx.graph_module._get_attr_via_attr_list(stateful_gm, prefix)
        delattr(submod, field)
        _assign_attr(buffer, stateful_gm, constant_fqn, attr_kind=_AttrKind.CONSTANT)

    # Constants are not preserved well when we create a new GraphModule unlike param/buffers
    for const_name, value in ep.constants.items():
        if not torch.fx.graph_module._has_attr(stateful_gm, const_name):
            if isinstance(value, torch.Tensor):
                if value.requires_grad:
                    warnings.warn(
                        f"A model attribute `{const_name}` requires gradient "
                        f"but it's not properly registered as a parameter. "
                        f"torch.export will detach it and treat it as a constant tensor "
                        f"but please register it as parameter instead.",
                        stacklevel=2,
                    )
                    if value in original_tensor_to_detached_tensor:
                        value = original_tensor_to_detached_tensor[value]
                    else:
                        detached_value = value.detach()
                        original_tensor_to_detached_tensor[value] = detached_value
                        value = detached_value
            _assign_attr(
                value,
                stateful_gm,
                const_name,
                attr_kind=_AttrKind.CONSTANT,
            )

    # Fix up non-persistent buffers. torch.fx does not distinguish between
    # persistent and non-persistent buffers, so we must restore that distinction
    # here.
    for buffer in ep.graph_signature.non_persistent_buffers:
        _assign_attr(
            plain_graph_module.get_buffer(buffer),
            stateful_gm,
            buffer,
            attr_kind=_AttrKind.BUFFER,
            persistent=False,
        )

    return stateful_gm


def _get_input_paths(example_inputs, signature):
    """
    Generate paths of placeholders, needed for generating the guards function.

    NOTE: Here we make use of the example inputs used for export as well as
    the signature of the unlifted graph module (not preserved by export).
    """

    args, kwargs = example_inputs
    binded = signature.bind(*args, **kwargs)
    binded.apply_defaults()
    ctx = binded.arguments
    flat_example_inputs_with_paths = pytree.tree_leaves_with_path(ctx)
    return [path for path, _ in flat_example_inputs_with_paths]


def _replace_sources(result_str: str, flat_input_paths: list[Any]):
    """
    Given user specified input paths, maybe fix up the guard string
    to reflect user path instead of tracer path.
    """
    name_mapping = {}
    for idx, path in enumerate(flat_input_paths):
        name_mapping[f"L['flat_args'][{idx}]"] = f"L{pytree.keystr(path)}"

    replace = result_str
    for key, val in name_mapping.items():
        replace = replace.replace(key, val)
    return replace


def _get_input_guards_for_graph(
    placeholders: list[torch.fx.Node],
    range_constraints: dict[sympy.Symbol, ValueRanges],
    paths_for_placeholders: list[pytree.KeyPath],
):
    """
    Guards generated by the tracer include conditions observed in code, but
    do not include some additional checks we typically do in export.
    For example, when dynamic shapes get specialized, are specified to be
    within a range, or are specified to be in some equational relation,
    corresponding input invalidation is done within a pre_hook, specifically,
    `_check_input_constraints_for_graph`.

    Here we generate guards corresponding to the checks that happen in
    `_check_input_constraints_for_graph`, and add them to the guards already
    generated by the tracer. In the future, it may be worthwhile to separate
    them so that we can allow clients to turn off one but not the other.
    (Looking at you, AOTI.)

    NOTE: We should eventually reconcile this logic with `build_guards` that
    is used by AOT Precompile.
    """

    deferred_expressions = []
    new_guards_code = []
    sources: dict[sympy.Expr, str] = {}

    def handle_symint(expr, src):
        if len(expr.free_symbols) == 1:
            # complex equations (e.g., involving derived dims) need to
            # handled later, since we may not have enough information
            # just as we are passing through the placeholders in order
            deferred_expressions.append((src, expr))
        if expr in sources:
            # expressions that appear in multiple sources should force
            # inputs corresponding to those sources to be equal
            # e.g., x.shape[0] == y.shape[1]
            orig_src = sources[expr]
            new_guards_code.append(f"{src} == {orig_src}")
        else:
            sources[expr] = src
            # process value ranges as elsewhere in export
            min_val, max_val = _convert_range_to_int(range_constraints[expr])
            if min_val > 2:
                new_guards_code.append(f"{src} >= {min_val}")
            if max_val < math.inf:
                new_guards_code.append(f"{src} <= {max_val}")

    for placeholder, path in zip(placeholders, paths_for_placeholders):
        src = "L" + pytree.keystr(path)
        meta = placeholder.meta["val"]
        # specializations
        if isinstance(meta, int):
            new_guards_code.append(f"{src} == {meta}")
        if isinstance(meta, float):
            if math.isnan(meta):
                new_guards_code.append(f"math.isnan({src})")
            elif meta == math.inf:
                new_guards_code.append(f"{src} == math.inf")
            elif meta == -math.inf:
                new_guards_code.append(f"{src} == -math.inf")
            else:
                new_guards_code.append(f"{src} == {meta}")
        elif isinstance(meta, str):
            new_guards_code.append(f"{src} == '{meta}'")
        # range constraints and equalities
        elif isinstance(meta, torch.SymInt) and meta.node.expr in range_constraints:
            handle_symint(meta.node.expr, src)
        elif isinstance(meta, torch.Tensor):
            for i, dim in enumerate(meta.shape):
                src = "L" + pytree.keystr(path) + f".size()[{i}]"
                if isinstance(dim, int):
                    # specializations
                    new_guards_code.append(f"{src} == {dim}")
                elif (
                    isinstance(dim, torch.SymInt) and dim.node.expr in range_constraints
                ):
                    # range constraints and equalities
                    handle_symint(dim.node.expr, src)

    unification_map: dict[sympy.Symbol, sympy.Expr] = {}
    py_printer = torch.utils._sympy.printers.PythonPrinter()

    # process complex equations (e.g., involving derived dims)
    for src, expr in deferred_expressions:
        # we know this is the only symbol in expr (see check above)
        symbol = next(iter(expr.free_symbols))
        if symbol in sources:
            # if s0 is already known to be directly sourced from inputs,
            # e.g., z.shape[2], we do not need to do anything further
            # (assume we have already processed constraints on s0 above)
            continue

        # otherwise s0 has some "hidden" source like 'dim'
        # example: src = y.shape[1], expr = s0 + 1
        if symbol in unification_map:
            # suppose that we already know that s0 = x.shape[0] * 2
            # so we can emit the guard: x.shape[0] * 2 + 1 = y.shape[1]
            substitution = expr.subs(unification_map)
            new_guards_code.append(
                py_printer.doprint(sympy.Eq(substitution, sympy.Symbol(src)))
            )
        else:
            # we do not yet know what s0 is, but given s0 + 1 = y.shape[1],
            # we can solve for s0...now knowing that s0 = y.shape[1] - 1
            solution = try_solve(sympy.Eq(expr, sympy.Symbol(src)), symbol)
            if solution is not None:
                definition = solution[1]
                unification_map[symbol] = definition

    return new_guards_code


def _ok_to_generate_guards_fn():
    patterns = [
        "executorch",
        "modai",
        "on_device_ai",
        "torchao",
    ]
    # force check_guards=False for files matching `patterns`
    # because they have too many calls to .module() and
    # do not like any call modules in the graph
    # TODO: fix these files to handle guard fns
    frame = inspect.currentframe()
    while frame is not None:
        if any(path in frame.f_code.co_filename for path in patterns):
            return False
        frame = frame.f_back

    return True


def _unlift_exported_program_lifted_states(
    ep: ExportedProgram, check_guards=True
) -> torch.fx.GraphModule:
    check_guards = check_guards and _ok_to_generate_guards_fn()

    source_node_dict = {
        node.name: node for node in ep.graph.nodes if node.op != "placeholder"
    }
    # placeholder node name might change after deepcopy
    placeholder_source_node_dict = {
        node.target: node for node in ep.graph.nodes if node.op == "placeholder"
    }

    new_gm = torch.fx.GraphModule(ep.graph_module, copy.deepcopy(ep.graph))
    new_gm.meta.update(ep.graph_module.meta)
    ep = copy.copy(ep)
    ep._graph_signature = ExportGraphSignature(
        ep._graph_signature.input_specs, ep._graph_signature.output_specs
    )
    ep._graph_module = new_gm

    # TODO T206340015
    if ep.verifiers[0].dialect != "TRAINING":
        ep = _remove_effect_tokens(ep)

    _register_attrs_to_new_gm(new_gm, ep.graph_signature, ep.state_dict, ep.constants)
    forward_arg_names = (
        sig.forward_arg_names if (sig := ep.module_call_graph[0].signature) else None
    )
    lifted_inputs: list[str | None] = [
        (
            in_spec.target
            if in_spec.kind
            in (
                InputKind.BUFFER,
                InputKind.CONSTANT_TENSOR,
                InputKind.PARAMETER,
                InputKind.CUSTOM_OBJ,
            )
            else None
        )
        for in_spec in ep.graph_signature.input_specs
    ]

    mutated_outputs: list[str | None] = [
        (
            out_spec.target
            if out_spec.kind
            in (
                OutputKind.BUFFER_MUTATION,
                OutputKind.USER_INPUT_MUTATION,
                OutputKind.PARAMETER_MUTATION,
            )
            else None
        )
        for out_spec in ep.graph_signature.output_specs
    ]

    for node in new_gm.graph.nodes:
        source_node = None
        if node.op == "placeholder":
            source_node = placeholder_source_node_dict.get(node.target)
        else:
            if node.name in source_node_dict:
                source_node = source_node_dict.get(node.name)
        node.meta["from_node"] = [
            NodeSource(
                source_node,
                "ExportedProgram.module()",
                NodeSourceAction.CREATE,
            )
        ]

    if ep.call_spec.in_spec is None:
        raise AssertionError("ep.call_spec.in_spec cannot be None")
    new_gm = _unlift(
        new_gm,
        lifted_inputs,
        mutated_outputs,
        ep.call_spec.in_spec,
        ep.call_spec.out_spec,
        forward_arg_names=forward_arg_names,
    )
    unlift_gm = _create_stateful_graph_module(new_gm, ep.range_constraints, ep)
    unlift_gm.meta.update(ep.graph_module.meta)

    # create a _guards_fn submodule and insert a call to it after placeholders
    graph = unlift_gm.graph
    placeholders = graph.find_nodes(op="placeholder")
    if check_guards and placeholders and ep.example_inputs:
        sig = inspect.signature(unlift_gm.forward)
        input_paths = _get_input_paths(
            ep.example_inputs,
            sig,
        )

        # TODO (tmanlaibaatar)
        # This is band-aid solution to export new tracer replacing
        # shape env sources to flat_args. The real fix should be replacing
        # shape env sources to original user sources but this is quite
        # involved because you need to carefully construct new sources using
        # dynamo and replace all instances of it inside shape env. But it is
        # lot easier to manipulate after we turn them into strings and only
        # time we use these guards is during retracing or running exported program,
        # so it is probably ok to have "not useful" guards on ep for now.
        ep_guards = []
        for guard in ep._guards_code:
            ep_guards.append(_replace_sources(guard, input_paths))

        guards_code = _get_input_guards_for_graph(
            placeholders, ep.range_constraints, input_paths
        )

        ep_guards_code = _force_ep_signature_match(ep._guards_code, input_paths)
        ep_guards_code = _force_gm_signature_match(ep_guards_code, sig)
        guards_code.extend(ep_guards_code)
        unlift_gm._guards_fn = _convert_guards_code_to_fn(guards_code, input_paths)

        root_nn_module_stack = torch.fx._utils.first_call_function_nn_module_stack(
            graph
        )
        with graph.inserting_after(placeholders[-1]):
            node = graph.call_module("_guards_fn", tuple(placeholders))
            node.meta["nn_module_stack"] = root_nn_module_stack

        unlift_gm.recompile()

    return unlift_gm


class GuardsFn(torch.nn.Module):
    """
    Module class for guard functions.
    """

    def forward(self, *args):
        pass
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import contextlib
import copy
import dataclasses
import functools
import operator
import types
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, final, NamedTuple, TYPE_CHECKING

from torch._guards import tracing, TracingContext
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_impls import (
    _deregister_op_impl,
    _is_op_registered_to_fake_rule,
    register_op_impl,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx._symbolic_trace import _ConstantAttributeType
from torch.fx._utils import first_call_function_nn_module_stack
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts


if TYPE_CHECKING:
    # Import the following modules during type checking to enable code intelligence features,
    # such as auto-completion in tools like pylance, even when these modules are not explicitly
    # imported in user code.

    import sympy

    from torch.utils._sympy.value_ranges import ValueRanges

import torch
import torch.utils._pytree as pytree
from torch._export.utils import (
    _build_cache,
    _collect_all_valid_cia_ops,
    _collect_and_set_constant_attrs,
    _collect_param_buffer_metadata,
    _detect_fake_mode_from_gm,
    _fakify_params_buffers,
    _get_decomp_for_cia,
    _is_preservable_cia_op,
    _name_hoo_subgraph_placeholders,
    _override_graph_signature_for_temp_registered_constants,
    _overwrite_signature_for_non_persistent_buffers,
    _populate_param_buffer_metadata_to_new_gm,
    _register_constants_as_buffers,
    _rename_without_collisions,
    _special_op_to_preserve_cia,
    placeholder_naming_pass,
)
from torch._export.verifier import Verifier
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.export._tree_utils import is_equivalent, reorder_kwargs
from torch.export.decomp_utils import CustomDecompTable
from torch.fx._compatibility import compatibility
from torch.fx.passes.infra.pass_base import PassResult
from torch.fx.passes.infra.pass_manager import PassManager

from .graph_signature import (  # noqa: F401
    ArgumentSpec,
    ConstantArgument,
    CustomObjArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    SymBoolArgument,
    SymFloatArgument,
    SymIntArgument,
    TensorArgument,
    TokenArgument,
)


__all__ = [
    "ExportedProgram",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "default_decompositions",
]


PassType = Callable[[torch.fx.GraphModule], PassResult | None]


@dataclasses.dataclass
class ModuleCallSignature:
    inputs: list[ArgumentSpec]
    outputs: list[ArgumentSpec]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec
    forward_arg_names: list[str] | None = None

    def replace_all_uses_with(self, original_node, new_node):
        for i in self.inputs:
            if i.name == original_node.name:
                i.name = new_node.name
        for o in self.outputs:
            if o.name == original_node.name:
                o.name = new_node.name


@dataclasses.dataclass
class ModuleCallEntry:
    fqn: str
    signature: ModuleCallSignature | None = None


def _disable_prexisiting_fake_mode(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with unset_fake_temporarily():
            return fn(*args, **kwargs)

    return wrapper


def _fx_collection_equivalence_fn(
    spec1_type: type | None,
    spec1_context: pytree.Context,
    spec2_type: type | None,
    spec2_context: pytree.Context,
) -> bool:
    """Treat containers and their immutable variants as the same type. Otherwise
    compare as normal.
    """
    if spec1_type is None or spec2_type is None:
        return spec1_type is spec2_type and spec1_context == spec2_context

    if issubclass(spec1_type, (dict, immutable_dict)) and issubclass(
        spec2_type, (dict, immutable_dict)
    ):
        return spec1_context == spec2_context

    if issubclass(spec1_type, (list, immutable_list)) and issubclass(
        spec2_type, (list, immutable_list)
    ):
        return spec1_context == spec2_context

    return spec1_type is spec2_type and spec1_context == spec2_context


# This list is compiled from DispatchKey.cpp.
# The idea is that we use these keys to override
# CIA decomp in export
_AUTOGRAD_ALIAS_BACKEND_KEYS_TO_OVERRIDE = [
    torch._C.DispatchKey.AutogradCPU,
    torch._C.DispatchKey.AutogradCUDA,
    torch._C.DispatchKey.AutogradMeta,
    torch._C.DispatchKey.AutogradXLA,
    torch._C.DispatchKey.AutogradLazy,
    torch._C.DispatchKey.AutogradIPU,
    torch._C.DispatchKey.AutogradXPU,
    torch._C.DispatchKey.AutogradMPS,
    torch._C.DispatchKey.AutogradHPU,
    torch._C.DispatchKey.AutogradPrivateUse1,
    torch._C.DispatchKey.AutogradPrivateUse2,
    torch._C.DispatchKey.AutogradPrivateUse3,
]


# This list is compiled from DispatchKey.cpp.
# The idea is that we use these keys to add
# python kernels that directly uses default
# CIA decomp
# See NOTE Registering old CIA to Backend kernel
_BACKEND_KEYS_TO_OVERRIDE = [
    torch._C.DispatchKey.CPU,
    torch._C.DispatchKey.CUDA,
    torch._C.DispatchKey.Meta,
    torch._C.DispatchKey.XLA,
    torch._C.DispatchKey.Lazy,
    torch._C.DispatchKey.IPU,
    torch._C.DispatchKey.XPU,
    torch._C.DispatchKey.MPS,
    torch._C.DispatchKey.HPU,
]


@contextmanager
def _override_composite_implicit_decomp(cia_ops_to_callable):
    # This function overrides CompositeImplicitAutograd decomp for
    # functional composite ops that user specified. Ideally we want to not-decompose
    # ALL composite ops but today's C++ functionalization relies on
    # the fact that it is working with the opset after decomp is run.
    # Hence we can only do it for functional ops. One caveat is that
    # there are some composite ops that lie about their schema (claimed to be
    # functional but not really aka dropout), for these cases, we just decompose.
    saved_tables = {}
    patched_ops = set()
    for op_overload, decomp_callable in cia_ops_to_callable.items():
        saved_tables[op_overload] = op_overload.py_kernels.copy()
        patched_ops.add(op_overload)
        for override_dispatch_key in _AUTOGRAD_ALIAS_BACKEND_KEYS_TO_OVERRIDE:
            if override_dispatch_key not in op_overload.py_kernels:
                # TODO (tmanlaibaatar)https://github.com/pytorch/pytorch/issues/129430
                op_overload.py_impl(override_dispatch_key)(
                    autograd_not_implemented(op_overload, deferred_error=True)
                )
        # See NOTE: Registering old CIA to Backend kernel
        # It is important that we cache this before we override py_kernels.
        orig_cia_callable = _get_decomp_for_cia(op_overload)
        if torch._C.DispatchKey.CompositeImplicitAutograd in op_overload.py_kernels:
            del op_overload.py_kernels[torch._C.DispatchKey.CompositeImplicitAutograd]

        op_overload.py_impl(torch._C.DispatchKey.CompositeImplicitAutograd)(
            decomp_callable
        )

        # [NOTE] Directly registering fake tensor rule to CIA ops
        # The problem we are facing here is if your CIA custom rule
        # says we want to preserve the op, we will return NotImplemented.
        # Unfortunately, this will invoke meta device tracing in fake tensor
        # resulting in divergent behaviour for CIA kernels that has device based
        # branching (one case is torch.ops.aten.scaled_dot_product.attention)
        # To get around this issue, we register direct fake impl so that we
        # run the kernel before we actually try to decompose the op in FakeTensorMode.
        # Note that is a no-op in most cases, because:
        #   1) In post dispatch tracing, CIA would have already decomposed
        #   2) Most CIA impl are device agnostic.
        def _force_dispatch_to_orig_cia_callable(fake_tensor_mode, op, *args, **kwargs):
            orig_cia_callable = kwargs["original_callable"]
            del kwargs["original_callable"]
            with fake_tensor_mode:
                return orig_cia_callable(*args, **kwargs)

        if not _is_op_registered_to_fake_rule(op_overload):
            register_op_impl(op_overload)(
                functools.partial(
                    _force_dispatch_to_orig_cia_callable,
                    original_callable=orig_cia_callable,
                )
            )

        for key in _BACKEND_KEYS_TO_OVERRIDE:
            if key not in op_overload.py_kernels:
                # [NOTE] Registering old CIA to Backend kernel
                # We always register original CIA behavior to the backend keys kernel
                # The reason is when we are fake tensor prop-ing or executing real kernel,
                # we end up calling an operator on respective backend, which in python dispatcher,
                # will resolve into CIA key. (see resolve_key in torch/_ops.py)
                # As a result, this CIA now will call into the custom user defined
                # CIA which can cause a problem.
                # To make it more concrete, the case we are handling is:
                #  (1) there is a tensor constant we are performing constant propagation
                #      on during tracing
                #  (2) we invoke an op underneath autograd (either because we are below autograd,
                #      or we are tracing in inference mode), so one of the backend keys gets hit
                #  (3) the op we are invoking has a CIA impl that normally runs in eager mode
                #      (and the user wants to tweak this CIA impl during tracing, but during
                #      const-prop we want the original CIA to run
                op_overload.py_impl(key)(orig_cia_callable)

    try:
        yield
    finally:
        for op in patched_ops:
            op.py_kernels.clear()
            op.py_kernels.update(saved_tables[op])
            op._dispatch_cache.clear()
            _deregister_op_impl(op)


def _split_decomp_table_to_cia_and_python_decomp(
    decomp_table: dict[torch._ops.OperatorBase, Callable],
) -> tuple[dict[torch._ops.OperatorBase, Callable], ...]:
    all_preservable_cia_ops = set(_collect_all_valid_cia_ops())
    cia_ops_to_callable = {}

    for op in list(decomp_table.keys()):
        # TODO we are silently allowing non-safe(non-functional) ops through a crack
        # due to core aten decomp table having non-functional entries. Once we have
        # a tighter check around core aten decomp, we should warn users about them.
        # Tracking issue: (https://github.com/pytorch/pytorch/issues/135759)

        # if it is a valid CIA op we can mess with in export, we check if it is:
        #  1. Has been marked as to be decomposed. Example:
        #        decomp_table = decomp_table_to_core_aten()
        #        del decomp_table[aten.linear]
        #     In this case, user says decompose everything except for aten.linear
        #  2. Has been marked with custom decomp behaviour. Example:
        #        decomp_table = {aten.linear: some_op}
        # For (1), we want to remove all the CIA ops that weren't handled by user as
        # it suggests they are safe to decompose, so we should remove from preservable_list.
        # for (2), we just plumb the custom decomp to AOTDIspatcher.
        # In both cases, we want to remove this CIA op from the decomp_table as it is special
        # handled.
        if op in all_preservable_cia_ops:
            cia_ops_to_callable[op] = decomp_table[op]
            all_preservable_cia_ops.remove(op)
            del decomp_table[op]
        # If it is a custom op, we want to still preserve or do whatever
        # with it if it is a functional CIA. The reason we don't remove
        # from CIA list is because we don't query custom ops.
        elif _is_preservable_cia_op(op):
            op_name = op.name()
            if op_name.startswith("aten"):
                raise AssertionError(
                    f"This should be a custom op, got aten op: {op_name}"
                )
            cia_ops_to_callable[op] = decomp_table[op]

    # If we reached here, it means user intentionally deleted these CIA ops from
    # decomp table.
    for k in all_preservable_cia_ops:
        cia_ops_to_callable[k] = _special_op_to_preserve_cia

    return cia_ops_to_callable, decomp_table


def default_decompositions() -> "CustomDecompTable":
    """
    This is the default decomposition table which contains decomposition of
    all ATEN operators to core aten opset. Use this API together with
    :func:`run_decompositions()`
    """
    return CustomDecompTable()


def _decompose_and_get_gm_with_new_signature_constants(
    ep: "ExportedProgram",
    *,
    cia_to_decomp: dict[torch._ops.OperatorBase, Callable],
    python_decomp_table: dict[torch._ops.OperatorBase, Callable],
    joint_loss_index: int | None,
    decompose_custom_triton_ops,
):
    from torch._export.passes.lift_constants_pass import _materialize_and_lift_constants
    from torch._functorch.aot_autograd import aot_export_module
    from torch.export._trace import (
        _disable_custom_triton_op_functional_decomposition,
        _export_to_aten_ir,
        _ignore_backend_decomps,
        _verify_nn_module_stack,
        _verify_placeholder_names,
        _verify_stack_trace,
    )
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    def _is_joint_ir_decomp(ep, joint_loss_index):
        return (
            joint_loss_index is not None
            or ep.graph_signature.backward_signature is not None
        )

    if not _is_joint_ir_decomp(ep, joint_loss_index):
        mod = ep.module()

        wrapped_params_buffers = {
            **dict(mod.named_parameters(remove_duplicate=False)),
            **dict(mod.named_buffers(remove_duplicate=False)),
        }

        from torch._functorch._aot_autograd.subclass_parametrization import (
            unwrap_tensor_subclass_parameters,
        )

        # [NOTE] Unwrapping subclasses AOT
        # In torch.compile, the subclass unwrapping/wrapping happen at runtime
        # but at export, this is impossible as it is intended to be run on
        # C++ environment. As a result, we unwrap subclass parameters AOT. After this,
        # ExportedProgram state_dict won't be same as eager model because eager model
        # could have subclass weights while ExportedProgram will have desugared versions.
        # This is fine because run_decompositions is supposed to specialize to post-autograd
        # graph where the subclass desugaring is supposed to happen.
        unwrap_tensor_subclass_parameters(mod)
        unwrapped_params_buffers = {
            **dict(mod.named_parameters(remove_duplicate=False)),
            **dict(mod.named_buffers(remove_duplicate=False)),
        }

        # TODO T204030333
        fake_mode = _detect_fake_mode_from_gm(ep.graph_module)
        if fake_mode is None:
            fake_mode = FakeTensorMode(shape_env=ShapeEnv(), export=True)

        # Fix the graph output signature to be tuple if scalar
        out_spec = mod._out_spec

        if not isinstance(mod.graph._codegen, _PyTreeCodeGen):
            raise AssertionError(
                f"expected mod.graph._codegen to be _PyTreeCodeGen, got {type(mod.graph._codegen)}"
            )
        orig_arg_names = mod.graph._codegen.pytree_info.orig_args

        # aot_export expect the return type to always be a tuple.
        if out_spec is None:
            raise AssertionError("out_spec must not be None")
        if out_spec.type not in (list, tuple):
            out_spec = pytree.treespec_tuple([out_spec])

        mod.graph._codegen = _PyTreeCodeGen(
            _PyTreeInfo(
                orig_arg_names,
                mod._in_spec,
                out_spec,
            )
        )

        mod.recompile()

        # the exported module will store constants & non-persistent buffers such that
        # retracing treats them as persistent buffers, so we inform the constants lifting pass
        # and overwrite the new graph signature using the previous program.
        _collect_and_set_constant_attrs(ep.graph_signature, ep.constants, mod)

        # When we have a module with constant attributes, AotDispatcher doesn't actually
        # wrap them as functional tensors, because dynamo would have already made it buffer.
        # In non-strict case, however, AotDispatcher can intercept constants, causing it to not
        # functionalize the operators that are operating on constant tensors. Since dynamo already
        # wraps constants as buffers, we temporarily register the constants as buffers and undo this
        # operation after AOTDispatcher is done.
        temp_registered_constants = _register_constants_as_buffers(
            mod, ep.state_dict, ep.graph_signature.non_persistent_buffers
        )

        # get params & buffers after excluding constants
        fake_params_buffers = _fakify_params_buffers(fake_mode, mod)

        params_buffers_to_node_meta = _collect_param_buffer_metadata(mod)

        # TODO (tmanlaibaatar) Ideally run_decomp should just call _non_strict_export
        # but due to special handling of constants as non-persistent buffers make it little
        # difficult. But we should unify this code path together. T206837815
        from torch._export.non_strict_utils import (
            _enable_graph_inputs_of_type_nn_module,
            _fakify_script_objects,
        )

        retracing_args = []
        for node in mod.graph.nodes:
            if node.op == "placeholder":
                if isinstance(node.meta["val"], CustomObjArgument):
                    real_script_obj = None
                    if node.meta["val"].fake_val is None:
                        real_script_obj = ep.constants[node.meta["val"].name]
                    else:
                        real_script_obj = node.meta["val"].fake_val.real_obj
                    retracing_args.append(real_script_obj)
                else:
                    retracing_args.append(node.meta["val"])

        tx = TracingContext(fake_mode)

        with (
            fake_mode,
            _override_composite_implicit_decomp(
                cia_to_decomp,
            ),
            _enable_graph_inputs_of_type_nn_module(ep.example_inputs),
            tracing(tx),
        ):
            retracing_args_unwrapped = pytree.tree_unflatten(
                retracing_args, mod._in_spec
            )
            # this requires empty kwargs, but not in pytree.flattened format
            with _fakify_script_objects(
                mod,
                (
                    *retracing_args_unwrapped[0],
                    *retracing_args_unwrapped[1].values(),
                ),
                {},
                fake_mode,
            ) as (
                patched_mod,
                new_fake_args,
                new_fake_kwargs,
                new_fake_constant_attrs,
                map_fake_to_real,
            ):
                aten_export_artifact = _export_to_aten_ir(
                    patched_mod,
                    new_fake_args,
                    new_fake_kwargs,
                    fake_params_buffers,
                    new_fake_constant_attrs,
                    decomp_table=python_decomp_table,
                    _prettify_placeholder_names=False,
                    decompose_custom_triton_ops=decompose_custom_triton_ops,
                )

                # aten_export_artifact.constants contains only fake script objects, we need to map them back
                aten_export_artifact.constants = {
                    fqn: (
                        map_fake_to_real[obj]
                        if isinstance(obj, FakeScriptObject)
                        else obj
                    )
                    for fqn, obj in aten_export_artifact.constants.items()
                }

                gm = aten_export_artifact.gm
                new_graph_signature = aten_export_artifact.sig

                # In the previous step, we assume constants as buffers for AOTDispatcher to
                # functionalize properly, so undo that here
                new_graph_signature = (
                    _override_graph_signature_for_temp_registered_constants(
                        new_graph_signature, temp_registered_constants
                    )
                )

                _populate_param_buffer_metadata_to_new_gm(
                    params_buffers_to_node_meta, gm, new_graph_signature
                )

                # overwrite signature for non-persistent buffers
                new_graph_signature = _overwrite_signature_for_non_persistent_buffers(
                    ep.graph_signature, new_graph_signature
                )

                constants = {
                    **ep.constants,
                    **aten_export_artifact.constants,
                }
                constants.update(
                    _materialize_and_lift_constants(
                        gm, new_graph_signature, new_fake_constant_attrs
                    )
                )

                placeholder_naming_pass(
                    gm,
                    new_graph_signature,
                    patched_mod,
                    new_fake_args,
                    new_fake_kwargs,
                    fake_params_buffers,
                    constants,
                )

        _verify_nn_module_stack(gm)
        _verify_stack_trace(gm)
        _verify_placeholder_names(gm, new_graph_signature)

        gm, new_graph_signature = _remove_unnecessary_copy_op_pass(
            gm, new_graph_signature
        )

        # When we apply parameterization rule to unwrap
        # subclasses, the state dict will now have different
        # desugared parameters. We need to manually filter those
        # and update the ep.state_dict. Ideally, we should just return
        # the state dict of ep.module but ep.module only stores params
        # buffers that participate in forward. If we undo this behavior,
        # it would break some downstream users.
        new_state_dict = {
            **ep.state_dict,
            **{
                name: p
                for name, p in unwrapped_params_buffers.items()
                if name not in wrapped_params_buffers
            },
        }

        for name, p in wrapped_params_buffers.items():
            # Buffers can be persistent/non-persistent
            if name not in new_state_dict:
                if isinstance(p, torch.nn.Parameter):
                    raise AssertionError(
                        f"expected {name!r} not to be a torch.nn.Parameter when not in state_dict"
                    )

            if name in new_state_dict:
                if name not in unwrapped_params_buffers:
                    new_state_dict.pop(name)

        return gm, new_graph_signature, new_state_dict, constants

    old_placeholders = [
        node for node in ep.graph_module.graph.nodes if node.op == "placeholder"
    ]
    fake_args = [node.meta["val"] for node in old_placeholders]

    buffers_to_remove = [name for name, _ in ep.graph_module.named_buffers()]
    for name in buffers_to_remove:
        delattr(ep.graph_module, name)

    # TODO(zhxhchen17) Return the new graph_signature directly.
    fake_mode_det = detect_fake_mode(fake_args)
    fake_mode_ctx = contextlib.nullcontext() if fake_mode_det is None else fake_mode_det  # type: ignore[assignment]
    custom_triton_ops_decomposition_ctx = (
        contextlib.nullcontext
        if decompose_custom_triton_ops
        else _disable_custom_triton_op_functional_decomposition
    )
    with (
        _ignore_backend_decomps(),
        fake_mode_ctx,
        _override_composite_implicit_decomp(cia_to_decomp),
        custom_triton_ops_decomposition_ctx(),
    ):
        gm, graph_signature = aot_export_module(
            ep.graph_module,
            fake_args,
            # pyrefly: ignore[bad-argument-type]
            decompositions=python_decomp_table,
            trace_joint=joint_loss_index is not None,
            output_loss_index=(
                joint_loss_index if joint_loss_index is not None else None
            ),
        )
        if not isinstance(gm, torch.fx.GraphModule):
            raise AssertionError(f"expected torch.fx.GraphModule, got {type(gm)}")
        gm.graph.eliminate_dead_code()

    # Update the signatures with the new placeholder names in case they
    # changed when calling aot_export
    def update_arg(old_arg, new_ph):
        if isinstance(old_arg, ConstantArgument):
            return old_arg
        elif isinstance(old_arg, TensorArgument):
            return TensorArgument(name=new_ph.name)
        elif isinstance(old_arg, SymIntArgument):
            return SymIntArgument(name=new_ph.name)
        elif isinstance(old_arg, SymFloatArgument):
            return SymFloatArgument(name=new_ph.name)
        elif isinstance(old_arg, SymBoolArgument):
            return SymBoolArgument(name=new_ph.name)
        raise RuntimeError(f"Type of old_arg not supported: {type(old_arg)}")

    new_placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    new_outputs: tuple[torch.fx.Node, ...] = tuple(gm.graph.output_node().args[0])  # type: ignore[arg-type]

    # rename the placeholders
    if len(new_placeholders) != len(old_placeholders):
        raise AssertionError(
            f"new_placeholders length {len(new_placeholders)} does not match old_placeholders length {len(old_placeholders)}"
        )
    for old_ph, new_ph in zip(old_placeholders, new_placeholders):
        new_ph.name = new_ph.target = old_ph.name

    # handle name collisions with newly decomposed graph nodes
    name_map = {}
    find_available: dict[str, int] = defaultdict(int)
    used_names: set[str] = set()
    for ph in new_placeholders:
        name_map[ph.name] = ph.name
        _build_cache(ph.name, find_available, used_names)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        node.name = _rename_without_collisions(
            name_map, find_available, used_names, node.name, node.name
        )

    # propagate names to higher order op subgraphs
    _name_hoo_subgraph_placeholders(gm)

    # Run this pass before creating input/output specs, since size-related CSE/DCE might affect output signature.
    # Overwrite output specs afterwards.
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
        shape_env = _get_shape_env(gm)
        if shape_env is not None:
            with _set_node_metadata_hook(
                gm,
                functools.partial(
                    _node_metadata_hook, metadata={"stack_trace": stack_trace}
                ),
            ):
                insert_deferred_runtime_asserts(
                    gm,
                    shape_env,
                    f"exported program: {first_call_function_nn_module_stack(gm.graph)}",
                    export=True,
                )

    # update output specs
    gm.recompile()
    for output, name in zip(new_outputs, _graph_output_names(gm)):
        if name is not None:
            output.name = name

    # To match the output target with correct input for input mutations
    # need to find the old to new placeholder map
    old_new_placeholder_map = {
        spec.arg.name: new_placeholders[i].name
        for i, spec in enumerate(ep.graph_signature.input_specs)
        if not isinstance(spec.arg, ConstantArgument)
    }

    input_specs = [
        InputSpec(
            spec.kind,
            update_arg(spec.arg, new_placeholders[i]),
            spec.target,
            spec.persistent,
        )
        for i, spec in enumerate(ep.graph_signature.input_specs)
    ]

    output_specs = []

    # handle buffer & input mutations; these appear before loss output & gradients
    # (1) ep.graph_signature.input_specs tells us types of inputs
    # (2) graph_signature.user_inputs tells us node input names in order
    # (3) graph_signature.user_inputs_to_mutate tells us buffer & input mutations
    # map (3) -> (2) for input order, -> (1) for input type
    user_inputs_index = {name: i for i, name in enumerate(graph_signature.user_inputs)}
    mutation_names = list(graph_signature.user_inputs_to_mutate.keys())
    expected_names = [node.name for node in new_outputs[: len(mutation_names)]]
    if mutation_names != expected_names:
        raise AssertionError(
            f"mutation_names {mutation_names} does not match expected {expected_names}"
        )
    for output_name, input_name in graph_signature.user_inputs_to_mutate.items():
        i = user_inputs_index[input_name]
        input_spec = ep.graph_signature.input_specs[i]
        if input_spec.kind not in (InputKind.USER_INPUT, InputKind.BUFFER):
            raise AssertionError(
                f"expected input_spec.kind to be USER_INPUT or BUFFER, got {input_spec.kind}"
            )
        output_kind = (
            OutputKind.BUFFER_MUTATION
            if input_spec.kind == InputKind.BUFFER
            else OutputKind.USER_INPUT_MUTATION
        )
        target = (
            input_spec.target
            if input_spec.kind == InputKind.BUFFER
            else input_spec.arg.name
        )
        output_specs.append(
            OutputSpec(
                kind=output_kind,
                arg=TensorArgument(name=output_name),
                target=target,
            )
        )

    # handle actual user outputs
    for i, spec in enumerate(ep.graph_signature.output_specs):
        output_specs.append(
            OutputSpec(
                OutputKind.LOSS_OUTPUT if i == joint_loss_index else spec.kind,
                update_arg(spec.arg, new_outputs[len(mutation_names) + i]),
                old_new_placeholder_map.get(spec.target, spec.target),
            )
        )

    if joint_loss_index is not None:
        if graph_signature.backward_signature is None:
            raise AssertionError(
                "graph_signature.backward_signature must not be None when joint_loss_index is set"
            )
        gradients = graph_signature.backward_signature.gradients_to_user_inputs
        if len(graph_signature.user_inputs) != len(ep.graph_signature.input_specs):
            raise AssertionError(
                f"graph_signature.user_inputs length {len(graph_signature.user_inputs)} does not match "
                f"input_specs length {len(ep.graph_signature.input_specs)}"
            )
        specs = {
            graph_signature.user_inputs[i]: spec
            for i, spec in enumerate(ep.graph_signature.input_specs)
            if isinstance(spec.arg, TensorArgument)
        }
        for node in new_outputs[len(output_specs) :]:
            source = gradients[node.name]
            spec = specs[source]  # type: ignore[index]
            if spec.kind == InputKind.PARAMETER:
                kind = OutputKind.GRADIENT_TO_PARAMETER
                target = spec.target
            elif spec.kind == InputKind.USER_INPUT:
                kind = OutputKind.GRADIENT_TO_USER_INPUT
                target = source
            else:
                raise AssertionError(f"Unknown input kind: {spec.kind}")
            output_specs.append(
                OutputSpec(
                    kind,
                    TensorArgument(name=node.name),
                    target,
                )
            )

    if len(new_placeholders) != len(old_placeholders):
        raise AssertionError(
            f"new_placeholders length {len(new_placeholders)} does not match old_placeholders length {len(old_placeholders)}"
        )

    new_graph_signature = ExportGraphSignature(
        input_specs=input_specs, output_specs=output_specs
    )
    # NOTE: aot_export adds symint metadata for placeholders with int
    # values; since these become specialized, we replace such metadata with
    # the original values.
    # Also, set the param/buffer metadata back to the placeholders.
    inputs_to_parameters = new_graph_signature.inputs_to_parameters
    inputs_to_buffers = new_graph_signature.inputs_to_buffers
    for old_node, new_node in zip(old_placeholders, new_placeholders):
        if not isinstance(old_node.meta["val"], torch.Tensor):
            new_node.meta["val"] = old_node.meta["val"]

        if (
            new_node.target in inputs_to_parameters
            or new_node.target in inputs_to_buffers
        ):
            for k, v in old_node.meta.items():
                new_node.meta[k] = v
    return gm, new_graph_signature, ep.state_dict, ep.constants


def _remove_unnecessary_copy_op_pass(
    gm: torch.fx.GraphModule, new_graph_signature: ExportGraphSignature
) -> tuple[torch.fx.GraphModule, ExportGraphSignature]:
    """
    Removes redundant copy_ node that was introduced due to mutated buffer.
    """
    buffers_to_mutate = new_graph_signature.buffers_to_mutate
    parameters_to_mutate = new_graph_signature.parameters_to_mutate
    with gm._set_replace_hook(new_graph_signature.get_replace_hook()):
        for node in gm.graph.nodes:
            if node.op == "output":
                args, _ = pytree.tree_flatten(node.args)
                for out in args:
                    if isinstance(out, torch.fx.Node) and (
                        out.name in buffers_to_mutate
                        or out.name in parameters_to_mutate
                    ):
                        if (
                            out.op == "call_function"
                            and out.target is torch.ops.aten.copy.default
                        ):
                            out.replace_all_uses_with(out.args[1])  # type: ignore[arg-type]
                            gm.graph.erase_node(out)
        gm.recompile()
    return gm, new_graph_signature


def _common_getitem_elimination_pass(
    gm: torch.fx.GraphModule, graph_signature, module_call_graph
):
    with gm._set_replace_hook(graph_signature.get_replace_hook()):
        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue

            node_id: dict[torch.fx.Node, str] = {}
            getitems: dict[str, torch.fx.Node] = {}
            for node in list(module.graph.nodes):
                if node.op == "call_function" and node.target is operator.getitem:
                    source, idx = node.args
                    new_id = f"{node_id[source]}.{idx}"
                    if new_id in getitems:
                        node.replace_all_uses_with(getitems[new_id])
                        for entry in module_call_graph:
                            if entry.signature is not None:
                                entry.signature.replace_all_uses_with(
                                    node, getitems[new_id]
                                )
                        module.graph.erase_node(node)
                    else:
                        getitems[new_id] = node
                        node_id[node] = new_id
                else:
                    node_id[node] = node.name


def _get_updated_module_call_graph(
    old_gm: torch.fx.GraphModule,
    old_graph_signature: ExportGraphSignature,
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    old_module_call_graph: list[ModuleCallEntry],
):
    new_module_call_graph = copy.deepcopy(old_module_call_graph)

    old_nodes = {node.name: node for node in old_gm.graph.nodes}

    old_graph_params_buffers = {
        **old_graph_signature.inputs_to_parameters,
        **old_graph_signature.inputs_to_buffers,
    }
    new_graph_params_buffers = {
        **graph_signature.inputs_to_parameters,
        **graph_signature.inputs_to_buffers,
    }
    old_graph_non_user_inputs = {
        **old_graph_params_buffers,
        **old_graph_signature.inputs_to_lifted_tensor_constants,
        **old_graph_signature.inputs_to_lifted_custom_objs,
    }
    new_graph_non_user_inputs = {
        **new_graph_params_buffers,
        **graph_signature.inputs_to_lifted_tensor_constants,
        **graph_signature.inputs_to_lifted_custom_objs,
    }

    # use node-level provenance metadata to create a map
    # from old node names to new node names
    provenance: dict[str, str] = {}

    user_input_counter = 0
    old_user_input_names = [
        node.target for node in old_gm.graph.nodes if node.op == "placeholder"
    ]
    old_user_input_names = list(
        filter(
            lambda x: x not in old_graph_non_user_inputs
            and x not in old_graph_signature.input_tokens,
            old_user_input_names,
        )
    )
    new_user_input_names = [
        node.target for node in gm.graph.nodes if node.op == "placeholder"
    ]

    for node in gm.graph.nodes:
        if history := node.meta.get("from_node", []):
            provenance[history[-1].name] = node.name

        # For params and buffers, we might have applied parameterization rule
        # so that the names might have changed. But for user inputs, we know we
        # must preserve the old name.
        elif node.op == "placeholder":
            if not (
                node.target in new_graph_non_user_inputs
                or node.target in graph_signature.input_tokens
            ):
                if node.target in new_user_input_names:
                    if not isinstance(node.name, str):
                        raise AssertionError(
                            f"expected node.name to be str, got {type(node.name)}"
                        )
                    old_name = old_user_input_names[user_input_counter]
                    if not isinstance(old_name, str):
                        raise AssertionError(
                            f"expected old_name to be str, got {type(old_name)}"
                        )
                    provenance[old_name] = node.name
                    user_input_counter += 1

    # For all the parameters and buffers, we first see
    # if they are result of parametrizations and if they
    # are, we log them and error later
    old_param_to_desugared = defaultdict(list)
    for name, target in new_graph_params_buffers.items():
        # if the parameters are not parametrized, the naming won't change.
        if not target.startswith("parametrizations."):
            # If we are in strict mode, we can't just reuse the param names
            if name in old_graph_params_buffers:
                provenance[name] = name
        else:
            old_target = ".".join(target.split(".")[1:-1])
            old_param_to_desugared[old_target].append(name)

    # map old names to new names in module call signatures
    for entry in new_module_call_graph:
        signature = entry.signature
        if signature is None:
            continue
        for x in [*signature.inputs, *signature.outputs]:
            # We noticed that submodule is taking subclass as input. we can't
            # preserve signature here.
            if x.name in old_param_to_desugared:
                raise ValueError(
                    f"It looks like {x.name} is a tensor subclass. "
                    f"Preserving submodule that takes subclass parameter is not supported"
                    f" in inference IR because we desugar them, resulting in more tensors"
                )

            if x.name in provenance:
                x.name = provenance[x.name]

            # This can happen when aten.to is called at graph boundaries.
            # Basically aten.to at post-dispatch level can either be copy
            # or alias. In the alias case, we will no-op it so it will
            # disappear from the graph. If we detect such case, we should
            # reuse the input to aten.to as the new input to the submodule.
            # Technically this can happen for other maybe aliasing ops,
            # but aten.to is probably the most common one.
            elif x.name in old_nodes:
                old_node = old_nodes[x.name]
                if old_node.op == "call_function" and old_node.target in [
                    torch.ops.aten.to.dtype_layout,
                    torch.ops.aten.to.device,
                    torch.ops.aten.to.dtype,
                ]:
                    old_target = old_node.args[0].name
                    if old_target not in provenance:
                        raise ValueError(
                            f"It looks like {old_target} is a tensor subclass. "
                            f"Preserving submodule that takes subclass parameter is not supported"
                            f" in inference IR because we desugar them, resulting in more tensors"
                        )

                    x.name = provenance[old_target]

    return new_module_call_graph


def _decompose_exported_program(
    ep,
    *,
    cia_to_decomp: dict[torch._ops.OperatorBase, Callable],
    python_decomp_table: dict[torch._ops.OperatorBase, Callable],
    joint_loss_index: int | None,
    decompose_custom_triton_ops: bool,
):
    (
        gm,
        new_graph_signature,
        state_dict,
        constants,
    ) = _decompose_and_get_gm_with_new_signature_constants(
        ep,
        cia_to_decomp=cia_to_decomp,
        python_decomp_table=python_decomp_table,
        joint_loss_index=joint_loss_index,
        decompose_custom_triton_ops=decompose_custom_triton_ops,
    )

    # The signatures of ep.module_call_graph refer to input / output nodes of
    # the original graph module. However, the new graph module may have
    # new nodes due to decompositions. So we need to update these signatures
    # in the decomposed exported program's module_call_graph.
    new_module_call_graph = _get_updated_module_call_graph(
        ep.graph_module,
        ep.graph_signature,
        gm,
        new_graph_signature,
        ep.module_call_graph,
    )

    # TODO unfortunately preserving graph-level metadata is not
    # working well with aot_export. So we manually copy it.
    # (The node-level meta is addressed above.)
    gm.meta.update(ep.graph_module.meta)

    new_range_constraints = _get_updated_range_constraints(
        gm,
        ep.range_constraints,
    )

    exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=new_graph_signature,
        state_dict=state_dict,
        range_constraints=new_range_constraints,
        module_call_graph=new_module_call_graph,
        example_inputs=ep.example_inputs,
        constants=constants,
    )
    return exported_program


class ExportedProgram:
    """
    Package of a program from :func:`export`. It contains
    an :class:`torch.fx.Graph` that represents Tensor computation, a state_dict containing
    tensor values of all lifted parameters and buffers, and various metadata.

    You can call an ExportedProgram like the original callable traced by
    :func:`export` with the same calling convention.

    To perform transformations on the graph, use ``.module`` property to access
    an :class:`torch.fx.GraphModule`. You can then use
    `FX transformation <https://pytorch.org/docs/stable/fx.html#writing-transformations>`_
    to rewrite the graph. Afterwards, you can simply use :func:`export`
    again to construct a correct ExportedProgram.
    """

    _graph_module: torch.fx.GraphModule
    """The underlying GraphModule containing the exported computation graph."""

    _graph_signature: ExportGraphSignature
    """The signature containing input/output specifications for the graph."""

    _state_dict: dict[str, Any]
    """Dictionary containing parameter and buffer values from the original module."""

    _range_constraints: "dict[sympy.Symbol, ValueRanges]"
    """Symbolic shape constraints for dynamic shapes in the graph."""

    _module_call_graph: list[ModuleCallEntry]
    """Call graph information tracking module hierarchy and signatures."""

    _example_inputs: tuple[tuple[Any, ...], dict[str, Any]] | None
    """Example inputs used during export, stored as (args, kwargs) tuple."""

    _constants: dict[str, _ConstantAttributeType]
    """Dictionary of constant values used in the graph."""

    _verifiers: list[type[Verifier]]
    """List of verifier classes used to validate the exported program."""

    _guards_code: list[str]

    def __init__(
        self,
        root: torch.nn.Module | dict[str, Any],
        graph: torch.fx.Graph,
        graph_signature: ExportGraphSignature,
        state_dict: dict[str, torch.Tensor | torch.nn.Parameter],
        range_constraints: "dict[sympy.Symbol, Any]",
        module_call_graph: list[ModuleCallEntry],
        example_inputs: tuple[tuple[Any, ...], dict[str, Any]] | None = None,
        constants: dict[str, _ConstantAttributeType] | None = None,
        *,
        verifiers: list[type[Verifier]] | None = None,
    ):
        # Remove codegen related things from the graph. It should just be a flat graph.
        graph._codegen = torch.fx.graph.CodeGen()
        self._graph_module = _create_graph_module_for_export(root, graph)
        if isinstance(root, torch.fx.GraphModule):
            self._graph_module.meta.update(root.meta)

        _common_getitem_elimination_pass(
            self._graph_module, graph_signature, module_call_graph
        )
        self._graph_signature: ExportGraphSignature = graph_signature
        self._state_dict: dict[str, Any] = state_dict
        self._range_constraints: dict[sympy.Symbol, ValueRanges] = range_constraints
        if module_call_graph is None:
            raise AssertionError("module_call_graph must not be None")
        self._module_call_graph: list[ModuleCallEntry] = module_call_graph
        self._example_inputs = example_inputs

        self._constants = constants or {}

        verifiers = verifiers or [Verifier]
        if not all(issubclass(v, Verifier) for v in verifiers):
            raise AssertionError(
                f"all verifiers must be subclasses of Verifier, got {verifiers}"
            )
        self._verifiers = verifiers
        # Validate should be always the last step of the constructor.
        self.validate()

        self._guards_code = _convert_guards_to_code(self._graph_module)

    @property
    @compatibility(is_backward_compatible=False)
    def graph_module(self):
        return self._graph_module

    @graph_module.setter
    @compatibility(is_backward_compatible=False)
    def graph_module(self, value):
        raise RuntimeError("Unable to set ExportedProgram's graph_module attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def graph(self):
        return self.graph_module.graph

    @graph.setter
    @compatibility(is_backward_compatible=False)
    def graph(self, value):
        raise RuntimeError("Unable to set ExportedProgram's graph attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def graph_signature(self):
        return self._graph_signature

    @graph_signature.setter
    @compatibility(is_backward_compatible=False)
    def graph_signature(self, value):
        raise RuntimeError("Unable to set ExportedProgram's graph_signature attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def state_dict(self):
        return self._state_dict

    @state_dict.setter
    @compatibility(is_backward_compatible=False)
    def state_dict(self, value):
        raise RuntimeError("Unable to set ExportedProgram's state_dict attribute.")

    @compatibility(is_backward_compatible=False)
    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """
        Returns an iterator over original module's parameters.
        """
        for _, param in self.named_parameters():
            yield param

    @compatibility(is_backward_compatible=False)
    def named_parameters(self) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """
        Returns an iterator over original module parameters, yielding
        both the name of the parameter as well as the parameter itself.
        """
        for param_name in self.graph_signature.parameters:
            yield param_name, self.state_dict[param_name]

    @compatibility(is_backward_compatible=False)
    def buffers(self) -> Iterator[torch.Tensor]:
        """
        Returns an iterator over original module buffers.
        """
        for _, buf in self.named_buffers():
            yield buf

    @compatibility(is_backward_compatible=False)
    def named_buffers(self) -> Iterator[tuple[str, torch.Tensor]]:
        """
        Returns an iterator over original module buffers, yielding
        both the name of the buffer as well as the buffer itself.
        """
        non_persistent_buffers = set(self.graph_signature.non_persistent_buffers)
        # Lazily computed on first use to avoid calling graph_module.state_dict()
        # unless at least one buffer is absent from state_dict and constants.
        gm_state_dict = None
        for buffer_name in self.graph_signature.buffers:
            if buffer_name in non_persistent_buffers:
                yield buffer_name, self.constants[buffer_name]
            elif buffer_name in self.state_dict:
                yield buffer_name, self.state_dict[buffer_name]
            elif buffer_name in self.constants:
                yield buffer_name, self.constants[buffer_name]
            else:
                # Nested invoke_subgraph tracing can surface lifted tensor
                # constants owned by an inner subgraph as persistent buffers
                # in the top-level graph signature; the tensor lives in the
                # subgraph submodule rather than state_dict or constants.
                if gm_state_dict is None:
                    gm_state_dict = self.graph_module.state_dict()
                yield buffer_name, gm_state_dict[buffer_name]

    @property
    @compatibility(is_backward_compatible=False)
    def range_constraints(self):
        return self._range_constraints

    @range_constraints.setter
    @compatibility(is_backward_compatible=False)
    def range_constraints(self, value):
        raise RuntimeError(
            "Unable to set ExportedProgram's range_constraints attribute."
        )

    @property
    @compatibility(is_backward_compatible=False)
    def module_call_graph(self):
        return self._module_call_graph

    @module_call_graph.setter
    @compatibility(is_backward_compatible=False)
    def module_call_graph(self, value):
        raise RuntimeError(
            "Unable to set ExportedProgram's module_call_graph attribute."
        )

    @property
    @compatibility(is_backward_compatible=False)
    def example_inputs(self):
        return self._example_inputs

    @example_inputs.setter
    @compatibility(is_backward_compatible=False)
    def example_inputs(self, value):
        # This is allowed

        if value is None:
            self._example_inputs = value
            return

        if not (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], tuple)
            and isinstance(value[1], dict)
        ):
            raise ValueError(
                "Example inputs should be a tuple containing example arguments (as "
                "a tuple), and example kwargs (as a dictionary)."
            )

        args, kwargs = value
        from ._unlift import _check_inputs_match

        _check_inputs_match(args, kwargs, self.call_spec.in_spec)

        self._example_inputs = value

    @property
    @compatibility(is_backward_compatible=False)
    def call_spec(self):
        class CallSpec(NamedTuple):
            in_spec: pytree.TreeSpec | None
            out_spec: pytree.TreeSpec | None

        if len(self.module_call_graph) == 0:
            return CallSpec(in_spec=None, out_spec=None)
        if self.module_call_graph[0].fqn != "":
            raise AssertionError(
                f"expected first module_call_graph fqn to be empty string, got {self.module_call_graph[0].fqn!r}"
            )
        return CallSpec(
            in_spec=self.module_call_graph[0].signature.in_spec,
            out_spec=self.module_call_graph[0].signature.out_spec,
        )

    @call_spec.setter
    @compatibility(is_backward_compatible=False)
    def call_spec(self, value):
        raise RuntimeError("Unable to set ExportedProgram's call_spec attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def verifier(self) -> Any:
        return self._verifiers[0]

    @verifier.setter
    @compatibility(is_backward_compatible=False)
    def verifier(self, value):
        raise RuntimeError("Unable to set ExportedProgram's verifier attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def dialect(self) -> str:
        if self._verifiers is None:
            raise AssertionError("_verifiers must not be None")
        return self._verifiers[0].dialect

    @dialect.setter
    @compatibility(is_backward_compatible=False)
    def dialect(self, value):
        raise RuntimeError("Unable to set ExportedProgram's dialect attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def verifiers(self):
        return self._verifiers

    @verifiers.setter
    @compatibility(is_backward_compatible=False)
    def verifiers(self, value):
        raise RuntimeError("Unable to set ExportedProgram's verifiers attribute.")

    @property
    @compatibility(is_backward_compatible=False)
    def tensor_constants(self):
        return self._constants

    @tensor_constants.setter
    @compatibility(is_backward_compatible=False)
    def tensor_constants(self, value):
        raise RuntimeError(
            "Unable to set ExportedProgram's tensor_constants attribute."
        )

    @property
    @compatibility(is_backward_compatible=False)
    def constants(self):
        return self._constants

    @constants.setter
    @compatibility(is_backward_compatible=False)
    def constants(self, value):
        raise RuntimeError("Unable to set ExportedProgram's constants attribute.")

    def _get_flat_args_with_check(self, args, kwargs):
        """Flatten args, kwargs using pytree, then, check specs.

        Args:
            args: List[Any] original args passed to __call__
            kwargs: Dict[str, Any] original kwargs passed to __call

        Returns:
            A tuple of (flat_args, received_spec)
            flat_args is flattened args / kwargs
            received_spec is the pytree spec produced while flattening the
            tuple (args, kwargs)
        """
        in_spec = self.call_spec.in_spec
        if in_spec is not None:
            kwargs = reorder_kwargs(kwargs, in_spec)
        flat_args_with_path, received_spec = pytree.tree_flatten_with_path(
            (args, kwargs)
        )
        self._check_input_constraints(flat_args_with_path)
        flat_args = tuple(x[1] for x in flat_args_with_path)
        return flat_args, received_spec

    def _graph_module_flat_inputs(self, args: Any, kwargs: Any) -> Any:
        """Transform args, kwargs of __call__ to args for graph_module.

        self.graph_module takes stuff from state dict as inputs.
        The invariant is for ep: ExportedProgram is
        ep(args, kwargs) ==
          ep.postprocess(ep.graph_module(ep.graph_module_flat_inputs(args, kwargs)))
        """

        in_spec = self.call_spec.in_spec
        flat_args, received_spec = self._get_flat_args_with_check(args, kwargs)
        if in_spec is not None and not is_equivalent(
            received_spec, in_spec, _fx_collection_equivalence_fn
        ):
            raise ValueError(
                "Trying to flatten user inputs with exported input tree spec: \n"
                f"{in_spec}\n"
                "but actually got inputs with tree spec of: \n"
                f"{received_spec}"
            )

        additional_inputs = []
        gm_state_dict = None
        for input_ in self.graph_signature.input_specs:
            if input_.kind == InputKind.USER_INPUT:
                continue
            elif input_.kind in (
                InputKind.PARAMETER,
                InputKind.BUFFER,
            ):
                if input_.persistent is False:
                    # This is a non-persistent buffer, grab it from our
                    # constants instead of the state dict.
                    additional_inputs.append(self.constants[input_.target])
                elif input_.target in self.state_dict:
                    additional_inputs.append(self.state_dict[input_.target])
                elif input_.target in self.constants:
                    additional_inputs.append(self.constants[input_.target])
                else:
                    # Nested invoke_subgraph tracing can leave a lifted tensor
                    # constant in the subgraph submodule's state rather than
                    # the top-level state_dict or constants.
                    if gm_state_dict is None:
                        gm_state_dict = self.graph_module.state_dict()
                    additional_inputs.append(gm_state_dict[input_.target])
            elif input_.kind in (
                InputKind.CONSTANT_TENSOR,
                InputKind.CUSTOM_OBJ,
            ):
                additional_inputs.append(self.constants[input_.target])
        additional_inputs = tuple(additional_inputs)

        # NOTE: calling convention is first params, then buffers, then args as user supplied them.
        # See: torch/_functorch/aot_autograd.py#L1034
        return additional_inputs + flat_args

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "Unable to call ExportedProgram directly. "
            "You should use `exported_program.module()` instead."
        )

    def __str__(self) -> str:
        graph_module = self.graph_module.print_readable(
            print_output=False, colored=False
        ).replace("\n", "\n    ")
        graph_signature = str(self.graph_signature).replace("\n", "\n    ")
        # No space after "Graph signature:" — graph_signature starts with
        # a newline; trailing whitespace breaks expecttest snapshots.
        string = (
            "ExportedProgram:\n"
            f"    {graph_module}\n"
            f"Graph signature:{graph_signature}\n"
            f"Range constraints: {self.range_constraints}\n"
        )
        return string

    def module(self, check_guards=True) -> torch.fx.GraphModule:
        """
        Returns a self contained GraphModule with all the parameters/buffers inlined.

        - When `check_guards=True` (default), a `_guards_fn` submodule is generated
          and a call to a `_guards_fn` submodule is inserted right after placeholders
          in the graph. This module checks guards on inputs.
        - When `check_guards=False`, a subset of these checks are performed by a
          forward pre-hook on the graph module. No `_guards_fn` submodule is generated.

        """
        from ._unlift import _unlift_exported_program_lifted_states

        module = _unlift_exported_program_lifted_states(self, check_guards=check_guards)

        def _train(self, mode: bool = True):
            raise NotImplementedError("Calling train() is not supported yet.")

        def _eval(self, mode: bool = True):
            raise NotImplementedError("Calling eval() is not supported yet.")

        module.train = types.MethodType(_train, module)  # type: ignore[method-assign]
        module.eval = types.MethodType(_eval, module)  # type: ignore[method-assign]
        return module

    def _num_lifted_params_buffers(self):
        return next(
            (
                i
                for i, s in enumerate(self._graph_signature.input_specs)
                if s.kind == InputKind.USER_INPUT
            ),
            len(self._graph_signature.input_specs),
        )

    @_disable_prexisiting_fake_mode
    def run_decompositions(
        self,
        decomp_table: dict[torch._ops.OperatorBase, Callable] | None = None,
        decompose_custom_triton_ops: bool = False,
    ) -> "ExportedProgram":
        """
        Run a set of decompositions on the exported program and returns a new
        exported program. By default we will run the Core ATen decompositions to
        get operators in the
        `Core ATen Operator Set <https://pytorch.org/docs/stable/torch.compiler_ir.html>`_.

        For now, we do not decompose joint graphs.

        Args:
            decomp_table:
             An optional argument that specifies decomp behaviour for Aten ops
             (1) If None, we decompose to core aten decompositions
             (2) If empty, we don't decompose any operator


        Some examples:

        If you don't want to decompose anything

        .. code-block:: python

            ep = torch.export.export(model, ...)
            ep = ep.run_decompositions(decomp_table={})

        If you want to get a core aten operator set except for certain operator, you can do following:

        .. code-block:: python

            ep = torch.export.export(model, ...)
            decomp_table = torch.export.default_decompositions()
            decomp_table[your_op] = your_custom_decomp
            ep = ep.run_decompositions(decomp_table=decomp_table)
        """
        _decomp_table = (
            default_decompositions() if decomp_table is None else dict(decomp_table)
        )

        if isinstance(_decomp_table, CustomDecompTable):
            _decomp_table = _decomp_table.materialize()

        # Note [Separating decomp_table into CIA decomps and non-CIA decomps]
        # At this point, we have a decomp_table that contains decomp behaviour for
        # both CIA and post-autograd ops.
        # We need to separate the op into two categories:
        # 1. CIA op: These are the ops that we want to override
        #    CompositeImplicitAutograd decomp for. For them, we need to use _override_composite_implicit_decomp
        #    context manager to plumb it through AOTDispatcher
        # 2. Non-CIA op: These ops are only relevant after AOTDIspatcher runs, so just
        #    checking if they are statically functional is enough.
        # For joint IR case tho, we need to use the old path because we can't register
        # custom decomps this way because we can't use context manager as it installs
        # autograd_error node.
        (
            cia_to_decomp,
            python_decomp_table,
        ) = _split_decomp_table_to_cia_and_python_decomp(_decomp_table)

        return _decompose_exported_program(
            self,
            cia_to_decomp=cia_to_decomp,
            python_decomp_table=python_decomp_table,
            joint_loss_index=None,
            decompose_custom_triton_ops=decompose_custom_triton_ops,
        )

    def _transform_do_not_use(self, *passes: PassType) -> "ExportedProgram":
        pm = PassManager(list(passes))
        # Since we abstractly run the passes, we need to disable backend decomp here
        # again.
        from torch.export._trace import _ignore_backend_decomps

        with _ignore_backend_decomps():
            res = pm(self.graph_module)
        transformed_gm = res.graph_module if res is not None else self.graph_module
        if transformed_gm is None:
            raise AssertionError("transformed_gm must not be None")

        # pyrefly: ignore [missing-attribute]
        if transformed_gm is self.graph_module and not res.modified:
            return self

        # TODO(zhxchen17) Remove this.
        def _get_updated_graph_signature(
            old_signature: ExportGraphSignature,
            new_gm: torch.fx.GraphModule,
        ) -> ExportGraphSignature:
            """
            Update the graph signature's user_input/user_outputs.
            """
            new_input_specs = []
            for i, node in enumerate(new_gm.graph.nodes):
                if node.op != "placeholder":
                    break

                if i >= len(old_signature.input_specs):
                    raise AssertionError(
                        f"Number of inputs changed after transformation: got index {i} "
                        f"but only {len(old_signature.input_specs)} input_specs"
                    )
                old_input_spec = old_signature.input_specs[i]
                arg = (
                    old_input_spec.arg
                    if isinstance(
                        old_input_spec.arg, (ConstantArgument, CustomObjArgument)
                    )
                    else type(old_input_spec.arg)(node.name)
                )
                new_input_specs.append(
                    InputSpec(
                        old_input_spec.kind,
                        arg,
                        old_input_spec.target,
                        old_input_spec.persistent,
                    )
                )

            output_node = list(new_gm.graph.nodes)[-1]
            if output_node.op != "output":
                raise AssertionError(
                    f"expected last node to have op='output', got {output_node.op!r}"
                )

            new_output_specs = []
            for i, node in enumerate(output_node.args[0]):
                if i >= len(old_signature.output_specs):
                    raise AssertionError(
                        f"Number of outputs changed after transformation: got index {i} "
                        f"but only {len(old_signature.output_specs)} output_specs"
                    )
                old_output_spec = old_signature.output_specs[i]
                arg = (
                    old_output_spec.arg
                    if isinstance(
                        old_output_spec.arg, (ConstantArgument, CustomObjArgument)
                    )
                    else type(old_output_spec.arg)(node.name)
                )
                new_output_specs.append(
                    OutputSpec(old_output_spec.kind, arg, old_output_spec.target)
                )

            new_signature = ExportGraphSignature(
                input_specs=new_input_specs, output_specs=new_output_specs
            )
            return new_signature

        transformed_ep = ExportedProgram(
            root=transformed_gm,
            graph=transformed_gm.graph,
            graph_signature=_get_updated_graph_signature(
                self.graph_signature, transformed_gm
            ),
            state_dict=self.state_dict,
            range_constraints=_get_updated_range_constraints(
                transformed_gm,
                self.range_constraints,
            ),
            module_call_graph=copy.deepcopy(self._module_call_graph),
            example_inputs=self.example_inputs,
            constants=self.constants,
            verifiers=self.verifiers,
        )
        transformed_ep.graph_module.meta.update(self.graph_module.meta)
        # pyrefly: ignore [missing-attribute]
        transformed_ep.graph_module.meta.update(res.graph_module.meta)
        return transformed_ep

    def _check_input_constraints(self, flat_args_with_path):
        from torch._export.utils import _check_input_constraints_for_graph

        placeholders = [p for p in self.graph.nodes if p.op == "placeholder"]
        input_placeholders = [
            p
            for p, s in zip(placeholders, self.graph_signature.input_specs)
            if s.kind == InputKind.USER_INPUT
        ]
        _check_input_constraints_for_graph(
            input_placeholders, flat_args_with_path, self.range_constraints
        )

    @compatibility(is_backward_compatible=False)
    def validate(self):
        self._validate()

    # TODO: remove this
    @final
    def _validate(self):
        if len(self.verifiers) == 0:
            raise AssertionError("ExportedProgram must have at least one verifier.")
        for v in self.verifiers:
            v().check(self)

    # TODO(zhxchen17) Formalize this.
    def _update(
        self,
        graph_module,
        graph_signature,
        *,
        state_dict=None,
        constants=None,
        verifiers=None,
    ) -> "ExportedProgram":
        return ExportedProgram(
            root=graph_module,
            graph=graph_module.graph,
            graph_signature=graph_signature,
            state_dict=state_dict if state_dict is not None else self.state_dict,
            range_constraints=copy.deepcopy(self.range_constraints),
            module_call_graph=copy.deepcopy(self._module_call_graph),
            example_inputs=self.example_inputs,
            constants=constants if constants is not None else self.constants,
            verifiers=verifiers if verifiers is not None else self.verifiers,
        )


def _get_shape_env(gm):
    vals = [
        node.meta["val"]
        for node in gm.graph.nodes
        if node.meta.get("val", None) is not None
    ]
    from torch._guards import detect_fake_mode

    fake_mode = detect_fake_mode(vals)
    if fake_mode is not None:
        return fake_mode.shape_env
    for v in vals:
        if isinstance(v, torch.SymInt):
            return v.node.shape_env


def _get_updated_range_constraints(
    gm: torch.fx.GraphModule,
    old_range_constraints: "dict[sympy.Symbol, Any] | None" = None,
) -> "dict[sympy.Symbol, Any]":
    if old_range_constraints is None:
        raise AssertionError("old_range_constraints must not be None")

    shape_env = _get_shape_env(gm)
    if shape_env is None:
        return {}

    range_constraints = copy.copy(old_range_constraints)
    range_constraints = {
        k: v for k, v in range_constraints.items() if k not in shape_env.replacements
    }
    # Only when we have an unbacked symint, and it's used as constructor inputs,
    # runtime_var_to_range will make a difference compated to var_to_range.
    # e.g. [2, oo) -> [0, oo)
    for k, v in shape_env.var_to_range.items():
        if k not in shape_env.replacements and k not in range_constraints:
            range_constraints[k] = v
    return range_constraints


def _create_graph_module_for_export(root, graph):
    try:
        gm = torch.fx.GraphModule(root, graph)
    except SyntaxError:
        # If custom objects stored in memory are being used in the graph,
        # the generated python code will result in a syntax error on the custom
        # object, since it is unable to parse the in-memory object. However
        # we can still run the graph eagerly through torch.fx.Interpreter,
        # so we will bypass this error.
        warnings.warn(
            "Unable to execute the generated python source code from "
            "the graph. The graph module will no longer be directly callable, "
            "but you can still run the ExportedProgram, and if needed, you can "
            "run the graph module eagerly using torch.fx.Interpreter.",
            stacklevel=2,
        )
        gm = torch.fx.GraphModule(root, torch.fx.Graph())
        gm._graph = graph

    return gm


def _convert_guards_to_code(graph_module):
    shape_env = _get_shape_env(graph_module)
    if shape_env is None:
        return []

    local_vars = {
        var
        for var, sources in shape_env.var_to_sources.items()
        if all(
            not isinstance(source, torch._dynamo.source.ConstantSource)
            for source in sources
        )
    }
    py_printer = torch.fx.experimental.symbolic_shapes.ShapeGuardPythonPrinter(
        shape_env.var_to_sources, lambda s: s.name, shape_env.var_to_sources
    )
    ret = [
        py_printer.doprint(guard.expr)
        for guard in shape_env.guards
        if guard.expr.free_symbols.issubset(local_vars)
    ]
    # TODO Figure out how to resolve guards containing weight sizes.
    # This is not a big deal as _guards_code is mostly empty today.
    return [guard for guard in ret if "L['self']" not in guard]
<|endoftext|>import logging
import os
import warnings
import zipfile
from collections.abc import Callable, Mapping
from typing import Any
from typing_extensions import deprecated

import torch
import torch.utils._pytree as pytree
from torch.fx.passes.infra.pass_base import PassResult
from torch.types import FileLike


__all__ = [
    "AdditionalInputs",
    "Constraint",
    "CustomDecompTable",
    "default_decompositions",
    "Dim",
    "dims",
    "draft_export",
    "export",
    "ExportBackwardSignature",
    "ExportedProgram",
    "ExportGraphSignature",
    "FlatArgsAdapter",
    "load",
    "ModuleCallEntry",
    "ModuleCallSignature",
    "register_dataclass",
    "save",
    "ShapesCollection",
    "unflatten",
    "UnflattenedModule",
]

# To make sure export specific custom ops are loaded
import torch.export.custom_ops

from ._state_dict_utils import _restore_state_dict
from .decomp_utils import CustomDecompTable
from .dynamic_shapes import AdditionalInputs, Constraint, Dim, dims, ShapesCollection
from .exported_program import (
    default_decompositions,
    ExportedProgram,
    ModuleCallEntry,
    ModuleCallSignature,
)
from .graph_signature import ExportBackwardSignature, ExportGraphSignature
from .unflatten import FlatArgsAdapter, unflatten, UnflattenedModule


PassType = Callable[[torch.fx.GraphModule], PassResult | None]

log: logging.Logger = logging.getLogger(__name__)


def export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = None,
    *,
    # dynamic_shapes: _DynamicShapesInput | AdditionalInputs | ShapesCollection
    dynamic_shapes: Any = None,
    strict: bool = False,
    preserve_module_call_signature: tuple[str, ...] = (),
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> ExportedProgram:
    """
    :func:`export` takes any nn.Module along with example inputs, and produces a traced graph representing
    only the Tensor computation of the function in an Ahead-of-Time (AOT) fashion,
    which can subsequently be executed with different inputs or serialized.  The
    traced graph (1) produces normalized operators in the functional ATen operator set
    (as well as any user-specified custom operators), (2) has eliminated all Python control
    flow and data structures (with certain exceptions), and (3) records the set of
    shape constraints needed to show that this normalization and control-flow elimination
    is sound for future inputs.

    **Soundness Guarantee**

    While tracing, :func:`export()` takes note of shape-related assumptions
    made by the user program and the underlying PyTorch operator kernels.
    The output :class:`ExportedProgram` is considered valid only when these
    assumptions hold true.

    Tracing makes assumptions on the shapes (not values) of input tensors.
    Such assumptions must be validated at graph capture time for :func:`export`
    to succeed. Specifically:

    - Assumptions on static shapes of input tensors are automatically validated without additional effort.
    - Assumptions on dynamic shape of input tensors require explicit specification
      by using the :func:`Dim` API to construct dynamic dimensions and by associating
      them with example inputs through the ``dynamic_shapes`` argument.

    If any assumption can not be validated, a fatal error will be raised. When that happens,
    the error message will include suggested fixes to the specification that are needed
    to validate the assumptions. For example :func:`export` might suggest the
    following fix to the definition of a dynamic dimension ``dim0_x``, say appearing in the
    shape associated with input ``x``, that was previously defined as ``Dim("dim0_x")``::

        dim = Dim("dim0_x", max=5)

    This example means the generated code requires dimension 0 of input ``x`` to be less
    than or equal to 5 to be valid. You can inspect the suggested fixes to dynamic dimension
    definitions and then copy them verbatim into your code without needing to change the
    ``dynamic_shapes`` argument to your :func:`export` call.

    Args:
        mod: We will trace the forward method of this module.

        args: Example positional inputs.

        kwargs: Optional example keyword inputs.

        dynamic_shapes:
         An optional argument where the type should either be:
         1) a dict from argument names of ``f`` to their dynamic shape specifications.
         For a variadic ``**kwargs`` argument, the dict may use the actual keyword names
         passed to ``f``, unless a keyword name collides with another input name.
         2) a tuple that specifies dynamic shape specifications for each input in original order.
         If you are specifying dynamism on keyword args, you will need to pass them in the order that
         is defined in the original function signature.

         The dynamic shape of a tensor argument can be specified as either
         (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
         not required to include static dimension indices in this dict, but when they are,
         they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
         where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
         are denoted by None. Arguments that are dicts or tuples / lists of tensors are
         recursively specified by using mappings or sequences of contained specifications.

         **ShapesSpec API.** ``dynamic_shapes`` may also be a
         :class:`torch.fx.experimental.dynamic_spec.ShapesSpec` (or its
         shorthand :class:`torch.fx.experimental.dynamic_spec.ParamsSpec`).
         This is a newer unbacked unified API across compile, pre-compile,
         export, etc., and is the recommended way to specify dynamic
         shapes for export going forward. It is the same spec API exposed
         via ``dynamic_shapes=`` in :func:`torch.compile`.

         The keys of ``ParamsSpec`` are **parameter names of the callable
         being traced** (for an ``nn.Module``, the parameters of
         ``forward``); keyword and ``**kwargs`` arguments are addressed by
         name too.

         Key properties (see :mod:`torch.fx.experimental.dynamic_spec` for
         full details):

         * **Unbacked-only.** Dims / scalars marked dynamic become unbacked
           SymInts (``u`` symbols) and are never specialized (including no
           0/1 specialization).
         * **Assumptions and derived expressions.** A dim can be an
           expression over the spec's symbols (e.g. ``TensorSpec([B * 2,
           ...])``), and you can pass relational ``assumptions`` between
           symbols (e.g. ``[B % 2 == 0]``) that export validates.
         * **No silent specialization.** The exported graph is guaranteed valid
           for every assumption provided, otherwise export fails. (By
           contrast, ``Dim.DYNAMIC`` / ``Dim.AUTO`` may silently specialize a
           dynamic dim, yielding a graph that is not valid for all the inputs).

         Example::

            batch = ShapeVar("batch", min=2, max=128)
            ep = torch.export.export(
                mod,
                (torch.randn(8, 3), torch.randn(16, 3)),
                dynamic_shapes=ShapesSpec(
                    params=ParamsSpec(
                        {
                            "x": TensorSpec([batch, 3]),
                            "y": TensorSpec([batch * 2, 3]),  # derived expression
                        }
                    ),
                    assumptions=[batch % 2 == 0],
                ),
                strict=True,
            )

        strict: When disabled (default), the export function will trace the program through
         Python runtime, which by itself will not validate some of the implicit assumptions
         baked into the graph. It will still validate most critical assumptions like shape
         safety. When enabled (by setting ``strict=True``), the export function will trace
         the program through TorchDynamo which will ensure the soundness of the resulting
         graph. TorchDynamo has limited Python feature coverage, thus you may experience more
         errors. Note that toggling this argument does not affect the resulting IR spec to be
         different and the model will be serialized in the same way regardless of what value
         is passed here.

        preserve_module_call_signature: A list of submodule paths for which the original
         calling conventions are preserved as metadata. The metadata will be used when calling
         torch.export.unflatten to preserve the original calling conventions of modules.

    Returns:
        An :class:`ExportedProgram` containing the traced callable.

    **Acceptable input/output types**

    Acceptable types of inputs (for ``args`` and ``kwargs``) and outputs include:

    - Primitive types, i.e. ``torch.Tensor``, ``int``, ``float``, ``bool`` and ``str``.
    - Dataclasses, but they must be registered by calling :func:`register_dataclass` first.
    - (Nested) Data structures comprising of ``dict``, ``list``, ``tuple``, ``namedtuple`` and
      ``OrderedDict`` containing all above types.

    """
    from ._trace import _export

    if not isinstance(mod, torch.nn.Module):
        raise ValueError(
            f"Expected `mod` to be an instance of `torch.nn.Module`, got {type(mod)}."
        )
    if isinstance(mod, torch.jit.ScriptModule):
        raise ValueError(
            "Exporting a ScriptModule is not supported. "
            "Maybe try converting your ScriptModule to an ExportedProgram "
            "using `TS2EPConverter(mod, args, kwargs).convert()` instead."
        )

    # If ``mod.forward`` carries an ``@dynamic_spec(...)`` decorator, the
    # attached ``ShapesSpec`` is used as ``dynamic_shapes``. Passing both raises.
    from torch.fx.experimental.dynamic_spec import _resolve_dynamic_shapes

    dynamic_shapes = _resolve_dynamic_shapes(mod, dynamic_shapes)

    try:
        return _export(
            mod,
            args,
            kwargs,
            dynamic_shapes,
            strict=strict,
            preserve_module_call_signature=preserve_module_call_signature,
            pre_dispatch=True,
            prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
        )
    except Exception as e:
        draft_export_msg = (
            "The error above occurred when calling torch.export.export. If you would "
            "like to view some more information about this error, and get a list "
            "of all other errors that may occur in your export call, you can "
            "replace your `export()` call with `draft_export()`."
        )

        # For errors that we know can be caught by draft-export, add the message
        # to ask users to try out draft-export
        if isinstance(
            e,
            (
                torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode,
                torch._subclasses.fake_tensor.UnsupportedOperatorException,
                torch._dynamo.exc.UserError,
                torch.fx.experimental.symbolic_shapes.ConstraintViolationError,
            ),
        ):
            new_msg = str(e) + "\n\n" + draft_export_msg
            e.args = (new_msg,)
        elif isinstance(e, RuntimeError) and "no fake impl registered" in str(e):
            new_msg = str(e) + "\n\n" + draft_export_msg
            e.args = (new_msg,)
        raise e


DEFAULT_PICKLE_PROTOCOL = 2


def save(
    ep: ExportedProgram,
    f: FileLike,
    *,
    extra_files: dict[str, Any] | None = None,
    opset_version: dict[str, int] | None = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> None:
    """

    .. warning::
        Under active development, saved files may not be usable in newer versions
        of PyTorch.

    Saves an :class:`ExportedProgram` to a file-like object. It can then be
    loaded using the Python API :func:`torch.export.load <torch.export.load>`.

    Args:
        ep (ExportedProgram): The exported program to save.

        f (str | os.PathLike[str] | IO[bytes]) A file-like object (has to
         implement write and flush) or a string containing a file name.

        extra_files (Optional[Dict[str, Any]]): Map from filename to contents
         which will be stored as part of f.

        opset_version (Optional[Dict[str, int]]): A map of opset names
         to the version of this opset

        pickle_protocol: can be specified to override the default protocol

    Example::

        import torch
        import io


        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x + 10


        ep = torch.export.export(MyModule(), (torch.randn(5),))

        # Save to file
        torch.export.save(ep, "exported_program.pt2")

        # Save to io.BytesIO buffer
        buffer = io.BytesIO()
        torch.export.save(ep, buffer)

        # Save with extra files
        extra_files = {"foo.txt": b"bar".decode("utf-8")}
        torch.export.save(ep, "exported_program.pt2", extra_files=extra_files)

    """
    if not isinstance(ep, ExportedProgram):
        raise TypeError(
            f"The 'ep' parameter must be an instance of 'ExportedProgram', got '{type(ep).__name__}' instead."
        )

    from torch.export.pt2_archive._package import package_pt2

    package_pt2(
        f,
        exported_programs={"model": ep},
        extra_files=extra_files,
        pickle_protocol=pickle_protocol,
        opset_version=opset_version,
    )


def load(
    f: FileLike,
    *,
    extra_files: dict[str, Any] | None = None,
    expected_opset_version: dict[str, int] | None = None,
) -> ExportedProgram:
    """

    .. warning::
        Under active development, saved files may not be usable in newer versions
        of PyTorch.

    .. warning::
        :func:`torch.export.load()` uses pickle under the hood to load models. **Never load data from an untrusted source.**

    Loads an :class:`ExportedProgram` previously saved with
    :func:`torch.export.save <torch.export.save>`.

    Args:
        f (str | os.PathLike[str] | IO[bytes]): A file-like object (has to
         implement write and flush) or a string containing a file name.

        extra_files (Optional[Dict[str, Any]]): The extra filenames given in
         this map would be loaded and their content would be stored in the
         provided map.

        expected_opset_version (Optional[Dict[str, int]]): A map of opset names
         to expected opset versions

    Returns:
        An :class:`ExportedProgram` object

    Example::

        import torch
        import io

        # Load ExportedProgram from file
        ep = torch.export.load("exported_program.pt2")

        # Load ExportedProgram from io.BytesIO object
        with open("exported_program.pt2", "rb") as f:
            buffer = io.BytesIO(f.read())
        buffer.seek(0)
        ep = torch.export.load(buffer)

        # Load with extra files.
        extra_files = {"foo.txt": ""}  # values will be replaced with data
        ep = torch.export.load("exported_program.pt2", extra_files=extra_files)
        print(extra_files["foo.txt"])
        print(ep(torch.randn(5)))
    """
    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    extra_files = extra_files or {}

    from torch.export.pt2_archive._package import load_pt2, PT2ArchiveContents

    try:
        pt2_contents = load_pt2(
            f,
            expected_opset_version=expected_opset_version,
        )
    except RuntimeError:
        log.warning("Ran into the following error when deserializing", exc_info=True)
        pt2_contents = PT2ArchiveContents({}, {}, {})

    if len(pt2_contents.exported_programs) > 0 or len(pt2_contents.extra_files) > 0:
        for k, v in pt2_contents.extra_files.items():
            extra_files[k] = v

        return pt2_contents.exported_programs["model"]

    # TODO: For backward compatibility, we support loading a zip file from 2.7. Delete this path in 2.9(?)
    with zipfile.ZipFile(f, "r") as zipf:
        if "version" not in zipf.namelist():
            raise RuntimeError(
                "We ran into an error when deserializing the saved file. "
                "Please check the warnings above for possible errors. "
            )

        log.warning(
            "Trying to deserialize for the older format. This version of file is "
            "deprecated. Please generate a new pt2 saved file."
        )

        # Check the version
        version = zipf.read("version").decode().split(".")
        from torch._export.serde.schema import (
            SCHEMA_VERSION,  # todo change archive version to schema version
        )

        if len(version) != len(SCHEMA_VERSION):
            raise AssertionError(
                "Version in the saved file has incorrect length, double check if the file is generated by torch.export.save()"
            )
        if version[0] != str(SCHEMA_VERSION[0]):
            raise RuntimeError(
                f"Serialized version {version} does not match our current "
                f"schema version {SCHEMA_VERSION}."
            )

        from torch._export.serde.serialize import deserialize, SerializedArtifact

        # Load serialized_ep and serialized_state_dict from the zip file

        serialized_exported_program: bytes | None = None
        serialized_state_dict: bytes | None = None
        serialized_constants: bytes | None = None
        serialized_example_inputs: bytes | None = None

        for file_info in zipf.infolist():
            file_content = zipf.read(file_info.filename)

            if file_info.filename == "serialized_exported_program.json":
                serialized_exported_program = file_content
            elif file_info.filename == "serialized_state_dict.json":
                warnings.warn("This version of file is deprecated", stacklevel=2)
                serialized_state_dict = file_content
            elif file_info.filename == "serialized_constants.json":
                warnings.warn("This version of file is deprecated", stacklevel=2)
                serialized_constants = file_content
            elif file_info.filename == "serialized_state_dict.pt":
                serialized_state_dict = file_content
            elif file_info.filename == "serialized_constants.pt":
                serialized_constants = file_content
            elif file_info.filename == "serialized_example_inputs.pt":
                serialized_example_inputs = file_content
            elif file_info.filename.startswith("extra_files"):
                filename = file_info.filename.split("/", 1)[1]
                extra_files[filename] = file_content.decode("utf-8")

        if serialized_exported_program is None:
            raise AssertionError("serialized_exported_program is None")
        if serialized_state_dict is None:
            raise AssertionError("serialized_state_dict is None")
        if serialized_constants is None:
            raise AssertionError("serialized_constants is None")
        if serialized_example_inputs is None:
            raise AssertionError("serialized_example_inputs is None")
        artifact: SerializedArtifact = SerializedArtifact(
            serialized_exported_program,
            serialized_state_dict,
            serialized_constants,
            serialized_example_inputs,
        )

        # Deserialize ExportedProgram
        ep = deserialize(artifact, expected_opset_version)

        return ep


def draft_export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = None,
    *,
    dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None,
    preserve_module_call_signature: tuple[str, ...] = (),
    strict: bool = False,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> ExportedProgram:
    """
    A version of torch.export.export which is designed to consistently produce
    an ExportedProgram, even if there are potential soundness issues, and to
    generate a report listing the issues found.
    """
    from torch.fx.experimental.dynamic_spec import ParamsSpec, ShapesSpec

    from ._draft_export import draft_export

    if isinstance(dynamic_shapes, (ShapesSpec, ParamsSpec)):
        raise NotImplementedError(
            f"draft_export does not support the new dynamic shapes API "
            f"({type(dynamic_shapes).__name__}); use torch.export.export instead."
        )

    return draft_export(
        mod=mod,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        strict=strict,
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
    )


def register_dataclass(
    cls: type[Any],
    *,
    serialized_type_name: str | None = None,
) -> None:
    """
    Registers a dataclass as a valid input/output type for :func:`torch.export.export`.

    Args:
        cls: the dataclass type to register
        serialized_type_name: The serialized name for the dataclass. This is
        required if you want to serialize the pytree TreeSpec containing this
        dataclass.

    Example::

        import torch
        from dataclasses import dataclass


        @dataclass
        class InputDataClass:
            feature: torch.Tensor
            bias: int


        @dataclass
        class OutputDataClass:
            res: torch.Tensor


        torch.export.register_dataclass(InputDataClass)
        torch.export.register_dataclass(OutputDataClass)


        class Mod(torch.nn.Module):
            def forward(self, x: InputDataClass) -> OutputDataClass:
                res = x.feature + x.bias
                return OutputDataClass(res=res)


        ep = torch.export.export(Mod(), (InputDataClass(torch.ones(2, 2), 1),))
        print(ep)

    """
    pytree.register_dataclass(cls, serialized_type_name=serialized_type_name)
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable

import torch
from torch._export.utils import (
    _collect_all_valid_cia_ops,
    _collect_all_valid_cia_ops_for_aten_namespace,
    _get_decomp_for_cia,
    _is_aten_op,
)


__all__ = ["CustomDecompTable"]


"""
Core ATen ops with Composite Implicit Autograd dispatch that should be excluded from decomposition
by default. The decomposition logic should eventually exclude all core-tagged CIA ops, but until all
backends are ready, this list allows opt-in one at a time.
"""
PRESERVED_ATEN_CIA_OPS = {
    torch.ops.aten.upsample_bilinear2d.vec,
    torch.ops.aten.upsample_nearest2d.vec,
    # NB: don't use the C++ decomp, because it is not functional!
    torch.ops.aten.silu_backward.default,
    torch.ops.aten.mish_backward.default,
    torch.ops.aten._fused_rms_norm.default,
}


class CustomDecompTable(dict[torch._ops.OperatorBase, Callable]):
    """
    This is a custom dictionary that is specifically used for handling decomp_table in export.
    The reason we need this is because in the new world, you can only *delete* an op from decomp
    table to preserve it. This is problematic for custom ops because we don't know when the custom
    op will actually be loaded to the dispatcher. As a result, we need to record the custom ops operations
    until we really need to materialize it (which is when we run decomposition pass.)

    Invariants we hold are:
     1. All aten decomp is loaded at the init time
     2. We materialize ALL ops when user ever reads from the table to make it more likely
        that dispatcher picks up the custom op.
     3. If it is write operation, we don't necessarily materialize
     4. We load the final time during export, right before calling run_decompositions()

    """

    def __init__(self):
        super().__init__()
        from torch._decomp import _core_aten_decompositions_post_autograd

        # For aten ops, we load them up in the beginning
        self.decomp_table = _core_aten_decompositions_post_autograd()

        for op in _collect_all_valid_cia_ops_for_aten_namespace():
            if op not in PRESERVED_ATEN_CIA_OPS and op not in self.decomp_table:
                self.decomp_table[op] = _get_decomp_for_cia(op)

        # This is to track the *pending* deleted custom ops that haven't been materialized yet
        self.deleted_custom_ops = set()
        # When this is true, there shouldn't be any pending operations in the table.
        self.has_materialized = False

    def __getitem__(self, key):
        self._materialize_if_needed()
        return self.decomp_table.__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self.decomp_table.__setitem__(key, value)

        if key in self.deleted_custom_ops:
            self.deleted_custom_ops.remove(key)

    def keys(self):
        self._materialize_if_needed()
        return self.decomp_table.keys()

    def __delitem__(self, key) -> None:
        self.pop(key)

    def update(self, other_dict):  # type: ignore[override]
        for k, v in other_dict.items():
            self.decomp_table.__setitem__(k, v)

    def __missing__(self, key) -> bool:
        return not self.__contains__(key)

    def __contains__(self, key) -> bool:
        self._materialize_if_needed()
        return self.decomp_table.__contains__(key)

    def __len__(self) -> int:
        self._materialize_if_needed()
        return self.decomp_table.__len__()

    def __iter__(self):
        self._materialize_if_needed()
        return self.decomp_table.__iter__()

    def __reversed__(self):
        self._materialize_if_needed()
        return self.decomp_table.__reversed__()

    def copy(self) -> "CustomDecompTable":
        new_dict = CustomDecompTable()
        new_dict.decomp_table = self.decomp_table.copy()
        new_dict.deleted_custom_ops = self.deleted_custom_ops.copy()
        new_dict.has_materialized = self.has_materialized
        return new_dict

    def pop(self, *args):
        def _pop_if_can(key):
            if _is_aten_op(key):
                return self.decomp_table.pop(key)

            if key in self.decomp_table:
                # Even if we materialized it, we should add it to the deleted
                # custom ops list so that when we materialize next time,
                # we should respect user's intention.
                self.deleted_custom_ops.add(key)
                return self.decomp_table.pop(key)

            if key in self.deleted_custom_ops:
                raise KeyError(f"{key} doesn't exist in the table")

            self.deleted_custom_ops.add(key)
            # We would come here when user pops off something that is
            # not in the table. In this case, we just pretend that it
            # was in the table.
            return _get_decomp_for_cia(key)

        if len(args) == 1:
            return _pop_if_can(args[0])

        if len(args) == 2:
            try:
                return _pop_if_can(args[0])
            except KeyError:
                return args[1]

    def items(self):
        self._materialize_if_needed()
        return self.decomp_table.items()

    def materialize(self) -> dict[torch._ops.OperatorBase, Callable]:
        for op in _collect_all_valid_cia_ops():
            if _is_aten_op(op):
                continue
            elif op in self.decomp_table:
                continue
            elif op not in self.deleted_custom_ops:
                self.decomp_table[op] = _get_decomp_for_cia(op)

        self.has_materialized = True
        self.deleted_custom_ops = set()
        return {**self.decomp_table}

    def _materialize_if_needed(self) -> None:
        if not self.has_materialized:
            self.materialize()
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import dataclasses
import functools
import inspect
import itertools
import logging
import re
import sys
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager, ExitStack, nullcontext
from itertools import chain
from typing import Any, cast, TYPE_CHECKING, TypeAlias
from unittest import mock


if TYPE_CHECKING:
    import weakref

import torch
import torch._dynamo
import torch.fx
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.exc import UserError, UserErrorType
from torch._export.db.logging import (
    exportdb_error_message,
    get_class_if_classified_error,
)
from torch._export.non_strict_utils import (
    _fakify_module_inputs,
    _fakify_script_objects,
    _gather_constant_attrs,
    _NonStrictTorchFunctionHandler,
    _override_builtin_ops,
    make_constraints,
    make_fake_inputs,
    produce_guards_and_solve_constraints,
)
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass
from torch._export.passes.lift_constants_pass import (
    _materialize_and_lift_constants,
    ConstantAttrMap,
)
from torch._export.utils import (
    _collect_param_buffer_metadata,
    _compiling_state_context,
    _fakify_params_buffers,
    _populate_param_buffer_metadata_to_new_gm,
    _update_gm_meta_if_possible,
    apply_runtime_assertion_pass,
    placeholder_naming_pass,
    placeholder_prefixes,
)
from torch._export.verifier import SpecViolationError
from torch._export.wrappers import _wrap_submodules
from torch._functorch._aot_autograd.graph_capture_wrappers import create_functional_call
from torch._functorch._aot_autograd.input_output_analysis import (
    _graph_input_names,
    _graph_output_names,
)
from torch._functorch._aot_autograd.schemas import GraphSignature
from torch._functorch._aot_autograd.subclass_utils import get_subclass_typing_container
from torch._functorch._aot_autograd.utils import (
    create_tree_flattened_fn,
    register_buffer_assignment_hook,
)
from torch._functorch.aot_autograd import (
    _detect_attribute_assignment,
    aot_export_joint_with_descriptors,
)
from torch._guards import detect_fake_mode, tracing, TracingContext
from torch._library.fake_class_registry import FakeScriptObject, maybe_to_fake_obj
from torch._library.opaque_object import is_custom_class
from torch._logging import dtrace_structured
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._utils_internal import compile_time_strobelight_meta, log_export_usage
from torch.export._leakage_detection_utils import find_legit_leaks_from_referrers
from torch.export._unlift import _check_input_constraints_pre_hook
from torch.export.dynamic_shapes import (
    _check_dynamic_shapes,
    _combine_args,
    _combine_args_for_tracing,
    _DimHintType,
    _IntWrapper,
    _normalize_dynamic_shapes,
    _process_dynamic_shapes,
)
from torch.export.exported_program import OutputKind
from torch.fx._symbolic_trace import _ConstantAttributeType
from torch.fx.experimental.dynamic_spec import (
    _SHAPES_SPEC_VS_DEFERRED_RUNTIME_ASSERTS_MSG,
    ParamsSpec,
    ShapesSpec,
)
from torch.fx.experimental.proxy_tensor import (
    get_proxy_slot,
    make_fx,
    PreDispatchTorchFunctionMode,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    ShapeEnv,
)
from torch.fx.graph import _PyTreeInfo
from torch.utils._pytree import TreeSpec
from torch.utils._sympy.value_ranges import ValueRangeError

from .exported_program import (
    _disable_prexisiting_fake_mode,
    ExportedProgram,
    InputKind,
    ModuleCallEntry,
    ModuleCallSignature,
)
from .graph_signature import (
    _convert_to_export_graph_signature,
    ArgumentSpec,
    ExportGraphSignature,
)


log = logging.getLogger(__name__)

# Dim-based dynamic shapes spec: the raw container forms (dict / tuple / list of
# ``Dim`` / ``None``) -- i.e. anything accepted as ``dynamic_shapes`` that is
# not the structured ShapesSpec / ParamsSpec API. Dims marked dynamic this way
# become backed symbols.
_DimDynamicShapesSpec: TypeAlias = dict[str, Any] | tuple[Any, ...] | list[Any]

# Full set of accepted ``dynamic_shapes`` inputs across export entry points.
_DynamicShapesInput: TypeAlias = _DimDynamicShapesSpec | ShapesSpec | ParamsSpec | None


@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """

    allow_rnn: bool = True
    reorderable_logging_functions: set[Callable] = dataclasses.field(
        default_factory=set
    )
    # Emit runtime asserts after AOTAutograd instead.
    # This isn't really necessary, and isn't much more efficient since the runtime asserts pass does CSE,
    # but if we want to reason more about what guards/runtime asserts to emit,
    # this makes it a bit cleaner to do from the export side. Also no real point in running this twice.
    do_not_emit_runtime_asserts: bool = True
    specialize_int: bool = True
    specialize_float: bool = True
    assume_static_by_default: bool = False
    automatic_dynamic_shapes: bool = False
    capture_dynamic_output_shape_ops: bool = True
    capture_scalar_outputs: bool = True
    prefer_deferred_runtime_asserts_over_guards: bool = False
    replay_side_effects: bool = False
    side_effect_replay_policy: str = "warn"


@dataclasses.dataclass
class ATenExportArtifact:
    gm: torch.fx.GraphModule
    sig: ExportGraphSignature
    constants: dict[str, _ConstantAttributeType]
    inferred_out_spec: TreeSpec


@dataclasses.dataclass(frozen=True)
class ExportArtifact:
    aten: ATenExportArtifact
    in_spec: TreeSpec
    out_spec: TreeSpec
    fake_mode: FakeTensorMode
    module_call_specs: dict[str, dict[str, pytree.TreeSpec]]


DEFAULT_EXPORT_DYNAMO_CONFIG = ExportDynamoConfig()
DEFAULT_EXPORT_DYNAMO_CONFIG.reorderable_logging_functions = {
    logging.critical,
    logging.debug,
    logging.error,
    logging.exception,
    logging.info,
    logging.log,
    logging.warning,
    print,
    warnings.warn,
}


@contextmanager
def _ignore_backend_decomps():
    orig_mkldnn_flag = torch.backends.mkldnn.set_flags(False)
    orig_nnpack_flag = torch.backends.nnpack.set_flags(False)
    orig_cudnn_flag = torch.backends.cudnn.set_flags(False)

    try:
        yield
    finally:
        torch.backends.mkldnn.set_flags(*orig_mkldnn_flag)
        torch.backends.nnpack.set_flags(*orig_nnpack_flag)
        torch.backends.cudnn.set_flags(*orig_cudnn_flag)


@contextmanager
def _disable_custom_triton_op_functional_decomposition():
    old = torch._functorch.config.decompose_custom_triton_ops
    try:
        # pyrefly: ignore [bad-assignment]
        torch._functorch.config.decompose_custom_triton_ops = False
        yield torch._functorch.config.decompose_custom_triton_ops
    finally:
        torch._functorch.config.decompose_custom_triton_ops = old


def custom_triton_ops_decomposition_disabled():
    return not torch._functorch.config.decompose_custom_triton_ops


def _fixup_key(x):
    return "L__self__" + _strip_root(x)


def _strip_root(x):
    if isinstance(x, str) and x.startswith("_export_root"):
        stripped = x[len("_export_root") :]
        return stripped.removeprefix(".")
    return x


def _is_bogus_const_name(name: str):
    splitted_names = name.split(".")
    if len(splitted_names) < 1:
        return True

    return splitted_names[-1].startswith("lifted_tensor")


def _rewrite_tracepoint_node(gm: torch.fx.GraphModule):
    """
    In-place modify input graph module by replacing the export tracepoint with a new node
    that has the same target and args, but with the _export_root stripped from path.
    """
    for node in gm.graph.nodes:
        if node.target is torch.ops.higher_order._export_tracepoint:
            if "path" in node.kwargs:
                path = _strip_root(node.kwargs["path"])
                with gm.graph.inserting_before(node):
                    new_node = gm.graph.create_node(
                        "call_function",
                        torch.ops.higher_order._export_tracepoint,
                        args=node.args,
                        kwargs={
                            "path": path,
                            "kind": node.kwargs["kind"],
                        },
                    )
                    new_node.meta = node.meta
                    node.replace_all_uses_with(new_node)
                    gm.graph.erase_node(node)


def detect_shape_env(inputs: Any = None):
    shape_envs = []

    for i, flat_input in enumerate(inputs):
        if isinstance(flat_input, torch.SymInt):
            shape_envs.append((flat_input.node.shape_env, "symint input", i))

    if shape_envs:
        shape_env, desc1, i1 = shape_envs[0]
        for m, desc2, i2 in shape_envs[1:]:
            if shape_env is not m:
                raise AssertionError(
                    f"shape env ({shape_env}) from {desc1} {i1} doesn't match mode ({m}) from {desc2} {i2}\n\n"
                    f"shape env from {desc1} {i1} allocated at:\n{shape_env.stack}\n"
                    f"shape env from {desc2} {i2} allocated at:\n{m.stack}"
                )
        return shape_env
    else:
        return None


def _extract_fake_inputs(gm, args, kwargs):
    """
    Given a graph module, extract fakified input tensors from the metadata of
    its placeholders, and map them to the structure of given args and kwargs.
    Also return the fake mode used to fakify those inputs.
    """
    fake_inps: list[Any] = []
    fake_vals: list[Any] = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            fake_inps.append(node.meta.get("val"))
        else:
            fake_vals.append(node.meta.get("example_value"))

    if dynamo_bytecode_flatten := getattr(gm, "_dynamo_bytecode_flatten", None):
        # In _extract_fake_inputs, the goal is to make real inputs into
        # fake (and symbolic) inputs. The way currently it's implemented
        # is by looking at the node.meta["val"] of the placeholder nodes.
        # This doesn't work when the graph is Dynamo flattened, because now
        # placeholder nodes doesn't have the ordering like pytree inputs do.
        # Instead, we need to look at how the inputs are shuffled, and map
        # the inputs to their actual fake inputs and symbolic inputs.
        # Since inputs can also contain symints, we cannot simply use the
        # FakeTensorMode memo to look up tensors only there.

        fake_inps = []
        positions = {}
        idx = 0

        def mark_inputs(x):
            # x can be a tensor or symbolic integer or a normal constant.
            nonlocal idx
            fake_inps.append(x)
            if isinstance(x, torch.Tensor):
                ret = x
            else:
                ret = object()
            if id(ret) not in positions:
                positions[id(ret)] = idx
            idx += 1
            return ret

        dummy_args = pytree.tree_map(mark_inputs, args + tuple(kwargs.values()))
        shuffled_args = dynamo_bytecode_flatten(*dummy_args)

        for node, shuffled_arg in zip(
            gm.graph.find_nodes(op="placeholder"), shuffled_args
        ):
            if id(shuffled_arg) in positions:
                fake_inps[positions[id(shuffled_arg)]] = node.meta.get("val")

    # We get both because now we might have a combination of symint and tensor
    # inputs, and we want to check that the shape env is consistent between
    # both. Unfortunately we can't see what fake mode is attached to the shape
    # env, then we can just compare fake modes.
    detected_fake_mode = detect_fake_mode(fake_inps + fake_vals)
    detected_shape_env = detect_shape_env(fake_inps + fake_vals)

    if detected_fake_mode:
        if detected_shape_env:
            if detected_shape_env is not detected_fake_mode.shape_env:
                raise AssertionError(
                    "Detected shape env does not match fake mode's shape env"
                )
        fake_mode = detected_fake_mode
    elif detected_shape_env:
        fake_mode = FakeTensorMode(shape_env=detected_shape_env, export=True)
    else:
        fake_mode = FakeTensorMode(shape_env=ShapeEnv(), export=True)

    count = 0

    def lookup_fake(x):
        nonlocal count
        val = fake_inps[count] if isinstance(x, (int, torch.Tensor)) else x
        count += 1
        return val

    fake_args = pytree.tree_map(lookup_fake, args)
    fake_kwargs = pytree.tree_map(lookup_fake, kwargs)

    return fake_args, fake_kwargs, fake_mode


def _replace_param_buffer_names(param_buffer_table, sig):
    for spec in sig.input_specs:
        if spec.kind in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
        ):
            spec.target = param_buffer_table[spec.target]
    for spec in sig.output_specs:
        if spec.kind in (
            OutputKind.BUFFER_MUTATION,
            OutputKind.GRADIENT_TO_PARAMETER,
        ):
            spec.target = param_buffer_table[spec.target]


def _convert_to_positional_args(orig_arg_names, args, kwargs):
    if len(orig_arg_names) != len(args) + len(kwargs):
        raise AssertionError(
            f"Total number of arg names is expected to be {len(orig_arg_names)} "
            f"but got {len(args)} positional args, {len(kwargs)} kwargs."
        )
    reordered_kwargs = [kwargs[kw_name] for kw_name in orig_arg_names[len(args) :]]
    return (
        *args,
        *reordered_kwargs,
    )


def _normalize_nn_module_stack(gm_torch_level, root_cls):
    # Append a root module to every nn_module_stack.
    root = "L['self']"
    root_key = re.sub(r"[^a-zA-Z0-9]", "_", root)
    for gm in gm_torch_level.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in gm.graph.nodes:
            if node.op in ["placeholder", "output"]:
                continue
            add_root = True
            if nn_module_stack := node.meta.get("nn_module_stack", {}):
                path, ty = next(iter(nn_module_stack.values()))
                # After deserializing the class `ty` might not exist anymore so
                # it could be a string
                if inspect.isclass(ty) and issubclass(ty, torch.nn.Module):
                    # TODO Figure out why sometimes we have root sometimes we don't.
                    if path == root and ty is root_cls:
                        add_root = False
                else:
                    if not isinstance(ty, str):
                        raise AssertionError(f"expected ty to be str, got {type(ty)}")
            if add_root:

                def normalize_path(path):
                    if path == "L['self']":
                        return ""
                    if path.startswith("L['self']."):
                        return path[len("L['self'].") :]
                    return path

                nn_module_stack = {
                    root_key: (root, root_cls.__module__ + "." + root_cls.__qualname__),
                    # pyrefly: ignore [unbound-name]
                    **nn_module_stack,
                }
                node.meta["nn_module_stack"] = {
                    key: (normalize_path(path), ty)
                    for key, (path, ty) in nn_module_stack.items()
                }


def _get_param_buffer_mapping(
    original_module: torch.nn.Module,
    traced_module: torch.nn.Module,
) -> dict[str, str]:
    """
    Returns a mapping of parameter/buffer names from the new module to the
    original model. This is to help with restoring the FQN for parameter/buffers
    of a traced module to what the original module contains.
    """

    param_lookup: dict[int, str] = {}
    buffer_lookup: dict[int, str] = {}
    for name, param in original_module.named_parameters(remove_duplicate=False):
        if param_lookup.get(id(param)) is None:
            # we only want to keep the first occurrence of a parameter to guarantee parity of original and traced module.
            param_lookup[id(param)] = name
    for name, buffer in original_module.named_buffers(remove_duplicate=False):
        buffer_lookup[id(buffer)] = name

    param_buffer_table: dict[str, str] = {}
    for dynamo_name, dynamo_param in traced_module.named_parameters(
        remove_duplicate=False
    ):
        if dynamo_name in param_buffer_table:
            raise AssertionError(
                f"dynamo_name {dynamo_name!r} already exists in param_buffer_table"
            )
        if id(dynamo_param) in param_lookup:
            param_buffer_table[dynamo_name] = param_lookup[id(dynamo_param)]

    for dynamo_name, dynamo_buffer in traced_module.named_buffers(
        remove_duplicate=False
    ):
        if dynamo_name in param_buffer_table:
            raise AssertionError(
                f"dynamo_name {dynamo_name!r} already exists in param_buffer_table for buffer"
            )
        if id(dynamo_buffer) in buffer_lookup:
            param_buffer_table[dynamo_name] = buffer_lookup[id(dynamo_buffer)]

    return param_buffer_table


def _preserve_requires_grad_pass(
    gm: torch.fx.GraphModule,
    sig: ExportGraphSignature,
    fake_params_buffers: dict[str, torch.Tensor],
    constants: dict[str, _ConstantAttributeType],
    flat_fake_args: list[Any],
):
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(sig.input_specs) != len(placeholders):
        raise AssertionError(
            f"input_specs length {len(sig.input_specs)} does not match placeholders length {len(placeholders)}"
        )
    i = 0
    for node, spec in zip(placeholders, sig.input_specs):
        if spec.kind in (
            InputKind.PARAMETER,
            InputKind.BUFFER,
        ):
            if spec.target is None:
                raise AssertionError(
                    f"spec.target must not be None for kind {spec.kind}"
                )
            node.meta["val"].requires_grad = fake_params_buffers[
                spec.target
            ].requires_grad
        elif spec.kind == InputKind.USER_INPUT:
            fake_arg = flat_fake_args[i]
            if isinstance(fake_arg, torch.Tensor):
                node.meta["val"].requires_grad = fake_arg.requires_grad
            i += 1
        elif spec.kind == InputKind.CONSTANT_TENSOR:
            if spec.target is None:
                raise AssertionError(
                    "spec.target must not be None for CONSTANT_TENSOR kind"
                )
            constant = constants[spec.target]
            if isinstance(constant, torch.Tensor):
                _set_constant_requires_grad(node, constant)
        elif spec.kind in (InputKind.CUSTOM_OBJ, InputKind.TOKEN):
            continue
        else:
            raise AssertionError(spec.kind)


def _set_constant_requires_grad(node: torch.fx.Node, constant: torch.Tensor) -> None:
    meta_val = node.meta.get("val")
    if isinstance(meta_val, torch.Tensor):
        # Non-leaf tensors should already have the correct requires_grad field.
        if meta_val.is_leaf:
            meta_val.requires_grad = constant.requires_grad
        elif meta_val.requires_grad != constant.requires_grad:
            raise AssertionError(
                f"node requires_grad {meta_val.requires_grad} does not match "
                f"constant requires_grad {constant.requires_grad}"
            )


def _remap_constants(
    orig_constant_attrs: ConstantAttrMap,
    graph_signature: ExportGraphSignature,
    constants: dict[str, _ConstantAttributeType],
) -> None:
    """Rewrite the graph signature and constants table to use the FQN from the original module."""
    remap_table: dict[str, list[str]] = {}
    for name, value in constants.items():
        if value in orig_constant_attrs:
            remap_table[name] = orig_constant_attrs[value]

    for spec in graph_signature.input_specs:
        if spec.kind in (
            InputKind.CONSTANT_TENSOR,
            InputKind.CUSTOM_OBJ,
        ):
            orig_target = spec.target
            if orig_target is None:
                raise AssertionError(
                    f"spec.target must not be None for kind {spec.kind}"
                )
            targets = remap_table.get(orig_target, [orig_target])
            spec.target = targets[0]

            constant = constants[orig_target]
            del constants[orig_target]
            for target in targets:
                constants[target] = constant


def _replace_unbacked_bindings(gm: torch.fx.GraphModule) -> None:
    """
    When we run an interpreter-based pass over a GraphModule, execution of data-dependent operators
    will produce example values with new unbacked symbols. To track that the new/old symbols are equivalent,
    we used to rely on the unbacked_renamings mapping. This led to problematic metadata where the unbacked_bindings
    keys mapped new symbols (u2) to paths containing old symbols (u0) in the example values, or worse, backed symbols
    or constants (e.g. if the original unbacked was replaced/specialized). Additionally this created problems with
    de/serialized programs, since we didn't comprehensively serialize ShapeEnv/unbacked renamings/node bindings.

    This pass attempts a simpler way of handling these for export, by throwing away the previously computed bindings, and re-running
    the pattern match used in compute_unbacked_bindings. This ensures we keep the original symbols contained in the example values,
    or delete bindings if they've been replaced/specialized.
    """
    from torch._export.utils import _get_shape_env_from_gm
    from torch.fx.experimental.symbolic_shapes import _free_unbacked_symbols_with_path
    from torch.utils._sympy.symbol import symbol_is_type, SymT

    if (shape_env := _get_shape_env_from_gm(gm)) is None:
        return

    base_unbacked_symbols = {
        symbol
        for symbol in shape_env.var_to_range
        if symbol_is_type(symbol, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT))
        and symbol not in shape_env.unbacked_renamings
    }
    for node in gm.graph.nodes:
        node.meta.pop("unbacked_bindings", None)
        if (val := node.meta.get("val")) is not None and (
            unbacked_bindings := _free_unbacked_symbols_with_path(
                val,
                (),
                shape_env=shape_env,
                pending=base_unbacked_symbols,
                simplify=True,
            )
        ):
            node.meta["unbacked_bindings"] = unbacked_bindings


def _add_input_unbacked_bindings(gm: torch.fx.GraphModule) -> None:
    from torch._export.utils import _get_shape_env_from_gm
    from torch.fx.experimental.symbolic_shapes import (
        _free_unbacked_symbols_with_path,
        free_unbacked_symbols,
    )

    if (shape_env := _get_shape_env_from_gm(gm)) is None:
        return

    for node in gm.graph.find_nodes(op="placeholder"):
        if node.meta.get("unbacked_bindings"):
            continue
        if (val := node.meta.get("val")) is None:
            continue
        pending: set = set()
        if isinstance(val, torch.Tensor):
            dims = [
                d for d in val.shape if isinstance(d, (torch.SymInt, torch.SymFloat))
            ]
        elif isinstance(val, (torch.SymInt, torch.SymFloat)):
            dims = [val]
        else:
            dims = []
        for dim in dims:
            pending |= set(free_unbacked_symbols(dim.node._expr))
        if unbacked_bindings := _free_unbacked_symbols_with_path(
            val,
            (),
            shape_env=shape_env,
            pending=pending,
            simplify=False,
        ):
            node.meta["unbacked_bindings"] = unbacked_bindings
        # Each input placeholder defines its own unbacked symbol(s); they must
        # all be bound here, otherwise their deferred runtime asserts would be
        # silently dropped by the runtime-assert pass.
        if pending:
            raise AssertionError(
                f"input placeholder {node.name!r} has unbound unbacked symbols "
                f"{pending}; their deferred runtime asserts would be dropped"
            )


def _apply_renames_to_signature(
    signature: ExportGraphSignature,
    renamed: dict[str, str],
) -> None:
    """Apply a batch of old-name-to-new-name renames to the signature."""
    for spec in [*signature.input_specs, *signature.output_specs]:
        arg = spec.arg
        if isinstance(arg, ArgumentSpec) and arg.name in renamed:
            arg.name = renamed[arg.name]


class _ExportSafeToReorder:
    """Barrier predicate for export canonicalization.

    Wraps ``_is_safe_to_reorder`` with an additional constraint: nodes from
    different ``nn_module_stack`` scopes are never reordered past each other,
    because ``unflatten`` expects nodes within the same module scope to be
    contiguous.
    """

    def __init__(self) -> None:
        from torch.fx.passes.canonicalize import _is_safe_to_reorder

        self._is_safe_to_reorder = _is_safe_to_reorder
        self._prev_stack: tuple[str, ...] | None = None

    def __call__(self, node: torch.fx.Node) -> bool:
        if not self._is_safe_to_reorder(node):
            self._prev_stack = tuple(node.meta.get("nn_module_stack", {}).keys())
            return False
        stack = tuple(node.meta.get("nn_module_stack", {}).keys())
        if stack != self._prev_stack:
            self._prev_stack = stack
            return False
        return True


def _canonicalize_export_graph(
    gm: torch.fx.GraphModule,
    signature: ExportGraphSignature,
) -> None:
    """Canonicalize node order and names in an export graph and all subgraphs.

    Reorders nodes into a deterministic topological order and renames them to
    canonical names so that strict and non-strict export produce identical
    graphs.  Updates ``signature`` to reflect the new node names.
    """
    from torch.fx.passes.canonicalize import _canonical_node_key, canonicalize_graph

    for mod in gm.modules():
        if isinstance(mod, torch.fx.GraphModule):
            placeholder_ord = itertools.count()

            def _key(
                node: torch.fx.Node,
                canonical_idx: dict[torch.fx.Node, int],
                _ord: itertools.count = placeholder_ord,
            ) -> object:
                if node.op == "placeholder":
                    return (0, next(_ord))
                return _canonical_node_key(node, canonical_idx)

            renamed = canonicalize_graph(
                mod.graph,
                _key,
                _ExportSafeToReorder(),
                skip_rename_ops=frozenset({"placeholder"}),
                group_getitems=True,
            )
            if mod is gm and renamed:
                _apply_renames_to_signature(signature, renamed)


def _produce_aten_artifact(
    *,
    gm: torch.fx.GraphModule,
    mod,
    constant_attrs,
    graph_signature,
    pre_dispatch,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    _prettify_placeholder_names=True,
) -> ATenExportArtifact:
    """
    This is a helper function that is shared between export_to_aten_ir and export_to_aten_ir_make_fx
    to produce the aten artifact. (export compatible graph module + signature)

    It does:
    1. Applies runtime assertion pass
    2. Recompute unbacked_bindings pass
    3. Populate meta val when missing
    4. Lift constants as placeholders
    5. Replace raw autograd and autocast ops with HOPs
    6. Prettify names for placeholders
    7. Preserve requires_grad value on node meta val
    """
    # Run runtime asserts pass before creating input/output specs, since size-related CSE/DCE might affect output signature.
    # Overwrite output specs afterwards.
    flat_fake_args = pytree.tree_leaves((fake_args, fake_kwargs))

    # Attach ``unbacked_bindings`` to *input placeholders* whose dims are
    # input-bound unbacked symbols, so the runtime-assert pass can emit their
    # deferred asserts.
    _add_input_unbacked_bindings(gm)

    gm, graph_signature = apply_runtime_assertion_pass(gm, graph_signature)

    # Simplify unbacked_bindings by recomputing them.
    # Useful for any pass that's interpreter-based and might call rebind_unbacked(),
    # e.g. AOTAutograd in this case.
    _replace_unbacked_bindings(gm)

    total_non_user_inputs = (
        len(graph_signature.parameters)
        + len(graph_signature.buffers)
        + len(graph_signature.input_tokens)
    )
    set_missing_meta_vals(gm, flat_fake_args, total_non_user_inputs)

    export_graph_signature: ExportGraphSignature | None
    export_graph_signature = _convert_to_export_graph_signature(
        graph_signature, gm, _get_non_persistent_buffers(mod)
    )

    # script objects are always stored in constants no matter whether they're initial inputs or
    # they're lifted in aot" before rewrite_script_object_meta
    constants = _materialize_and_lift_constants(
        gm, export_graph_signature, constant_attrs
    )

    if pre_dispatch:
        from torch._export.passes.replace_autocast_with_hop_pass import (
            replace_autocast_with_hop_pass,
        )
        from torch._export.passes.replace_set_grad_with_hop_pass import (
            replace_set_grad_with_hop_pass,
        )

        # Note: replace_set_grad_with_hop_pass need to be after lift_constant_pass because
        # a getattr of a constant tensor doesn't have meta["val"] until after lift_constant_pass.
        # If replace_set_grad_with_hop_pass is before lift_constant_pass,
        # and the constant_tensor is passed as input of the set grad hop, the placeholder's
        # meta["val"] will be None and fails our verifier for placeholder.
        gm, export_graph_signature = replace_set_grad_with_hop_pass(
            gm, export_graph_signature
        )

        gm, export_graph_signature = replace_autocast_with_hop_pass(
            gm, export_graph_signature
        )

    # Remove nn_module_stack, stack_trace metadata from all placeholders/inputs nodes.
    for _mod in gm.modules():
        if not isinstance(_mod, torch.fx.GraphModule):
            continue
        for node in _mod.graph.nodes:
            if node.op in ["placeholder", "output"]:
                node.meta.pop("nn_module_stack", None)
                node.meta.pop("stack_trace", None)

    # Prettify names for placeholder nodes.
    if export_graph_signature is None:
        raise AssertionError("export_graph_signature must not be None")
    if _prettify_placeholder_names:
        placeholder_naming_pass(
            gm,
            export_graph_signature,
            mod,
            fake_args,
            fake_kwargs,
            fake_params_buffers,
            constants,
        )

    _preserve_requires_grad_pass(
        gm, export_graph_signature, fake_params_buffers, constants, flat_fake_args
    )

    if torch._dynamo.config.canonicalize_output_graph_node_order:
        _canonicalize_export_graph(gm, export_graph_signature)

    return ATenExportArtifact(
        gm,
        export_graph_signature,
        constants,
        inferred_out_spec=graph_signature.out_spec,
    )


def _rename_constants_nodes(
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
) -> None:
    """
    For strict mode, rename constants nodes that were previously annotated as buffers.
    """
    # handle name collisions with existing constants
    node_names = {node.name for node in gm.graph.nodes}

    def rename_constant(name):
        if name in node_names:
            n = 1
            while (dup_name := f"{name}_{n}") in node_names:
                n += 1
            # pyrefly: ignore [unbound-name]
            name = dup_name
        node_names.add(name)
        return name

    # use input specs to map names from buffers to constants
    buffer_prefix = placeholder_prefixes[InputKind.BUFFER]
    const_prefix = placeholder_prefixes[InputKind.CONSTANT_TENSOR]
    buffer_to_constant = {}
    for spec in graph_signature.input_specs:
        if spec.kind == InputKind.CONSTANT_TENSOR and not spec.arg.name.startswith(
            const_prefix
        ):
            if spec.arg.name.startswith(buffer_prefix):  # map from buffer to constants
                c_name = rename_constant(
                    const_prefix + spec.arg.name[len(buffer_prefix) :]
                )
            else:  # lifted constant
                c_name = rename_constant(const_prefix + spec.arg.name)
            buffer_to_constant[spec.arg.name] = c_name
            spec.arg.name = c_name
    for spec in graph_signature.output_specs:
        if spec.arg.name in buffer_to_constant:
            spec.arg.name = buffer_to_constant[spec.arg.name]

    # Rename constants nodes for all modules
    for mod in gm.modules():
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in mod.graph.nodes:
            if node.name in buffer_to_constant:
                node.name = node.target = buffer_to_constant[node.name]
        mod.recompile()


def _restore_state_dict(
    original_module: torch.nn.Module, traced_module: torch.fx.GraphModule
) -> None:
    """
    Restores the state dict of the traced module to that of the original module.
    """
    param_buffer_table = _get_param_buffer_mapping(original_module, traced_module)
    # Don't want to change the convention of previous call.
    param_buffer_table_reverse = {v: k for k, v in param_buffer_table.items()}

    # Replace state dict attr names with the fqn
    for name, _ in list(
        chain(
            original_module.named_parameters(remove_duplicate=False),
            # pyrefly: ignore [bad-argument-type]
            original_module.named_buffers(remove_duplicate=False),
        )
    ):
        if name in param_buffer_table_reverse:
            dynamo_name = param_buffer_table_reverse[name]
            param = torch.fx.graph_module._get_attr(traced_module, dynamo_name)
            torch.fx.graph_module._assign_attr(param, traced_module, name)
            torch.fx.graph_module._del_attr(traced_module, dynamo_name)

    # Replace graph getattr nodes with the correct name
    for node in traced_module.graph.nodes:
        if node.op == "get_attr":
            attr_name = node.target
            if attr_name in param_buffer_table:
                node.target = param_buffer_table[attr_name]

    traced_module.recompile()


def _get_module_hierarchy(mod: torch.nn.Module) -> dict[str, str]:
    return {
        name: type(m).__name__ for name, m in mod.named_modules(remove_duplicate=False)
    }


def _make_module_call_graph(
    in_spec: TreeSpec,
    out_spec: TreeSpec,
    module_call_signatures: dict[str, ModuleCallSignature],
    forward_arg_names: list[str] | None = None,
) -> list[ModuleCallEntry]:
    original = [
        ModuleCallEntry(fqn=fqn, signature=module_call_signatures.get(fqn))
        for fqn in _EXPORT_MODULE_HIERARCHY  # type: ignore[union-attr]
    ]
    if original[0].fqn != "":
        raise AssertionError(
            f"expected first fqn to be empty string, got {original[0].fqn!r}"
        )
    original[0].signature = ModuleCallSignature(
        inputs=[],
        outputs=[],
        in_spec=in_spec,
        out_spec=out_spec,
        forward_arg_names=forward_arg_names,
    )
    additional = [
        ModuleCallEntry(fqn=fqn, signature=signature)
        for fqn, signature in module_call_signatures.items()
        if fqn not in _EXPORT_MODULE_HIERARCHY  # type: ignore[operator]
    ]
    return [*original, *additional]


class _ExportModuleSpecTrackerDict(dict):
    pass


def _export_to_torch_ir(
    f: Callable,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    dynamic_shapes: _DynamicShapesInput = None,
    *,
    preserve_module_call_signature: tuple[str, ...] = (),
    disable_constraint_solver: bool = False,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
    restore_fqn: bool = True,
    _log_export_usage: bool = True,
    same_signature: bool = True,
) -> torch.fx.GraphModule:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a torch.fx.GraphModule in torch IR.
    """

    if _log_export_usage:
        log_export_usage(event="export.private_api", flags={"_export_to_torch_ir"})

    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )

    kwargs = kwargs or {}

    # Map ints to a wrapper structure to help us mark it as dynamic, if it is
    # dynamic. We will unwrap ints in fakify later.
    args, kwargs = pytree.tree_map_only(int, _IntWrapper, (args, kwargs))

    is_shapes_spec = isinstance(dynamic_shapes, (ShapesSpec, ParamsSpec))

    if not is_shapes_spec:
        # Dim-based dynamic shapes.
        dynamic_shapes = _normalize_dynamic_shapes(dynamic_shapes, f, args, kwargs)
        combined_args = _combine_args(f, args, kwargs)
        _check_dynamic_shapes(combined_args, dynamic_shapes)
        constraints = _process_dynamic_shapes(combined_args, dynamic_shapes)
    else:
        constraints = []

    # Unwrap static ints -- in the case where we have an empty graph
    # containing just integer computation, dynamo will run its generated
    # bytecode with these args/kwargs, which will error because we cannot
    # directly apply int operations on IntWrapper. So we will just unwrap
    # them here.
    args, kwargs = pytree.tree_map_only(
        _IntWrapper,
        lambda a: a.val
        if a.dynamism is None or a.dynamism.type == _DimHintType.STATIC
        else a,
        (args, kwargs),
    )

    dynamo_cfg = dataclasses.replace(
        DEFAULT_EXPORT_DYNAMO_CONFIG,
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
    )

    def use_legacy_dynamo_graph_capture() -> bool:
        return bool(
            constraints  # dynamic shape
            or dynamic_shapes  # dynamic shape
            or isinstance(f, torch.fx.GraphModule)  # retracing
            or preserve_module_call_signature  # unflatten
            or torch._functorch.config.fake_tensor_propagate_real_tensors  # draft
            or torch._export.config.use_legacy_dynamo_graph_capture
        )

    with torch._dynamo.config.patch(dataclasses.asdict(dynamo_cfg)):
        try:
            module_call_specs: dict[str, dict[str, pytree.TreeSpec]] = (
                _ExportModuleSpecTrackerDict()
            )
            ctx = nullcontext()
            if not isinstance(f, torch.fx.GraphModule):
                ctx = _wrap_submodules(  # type: ignore[assignment]
                    f, preserve_module_call_signature, module_call_specs
                )
            with ctx, _ignore_backend_decomps():
                if torch._export.config.use_new_tracer_experimental:
                    from torch._dynamo.functional_export import (
                        _dynamo_graph_capture_for_export,
                        dynamo_graph_capture_for_export,
                    )

                    if use_legacy_dynamo_graph_capture():
                        dynamo_graph_capture = _dynamo_graph_capture_for_export(
                            f, constraints=constraints, dynamic_shapes=dynamic_shapes
                        )
                    else:
                        dynamo_graph_capture = torch._dynamo.config.patch(
                            replay_side_effects=False
                        )(dynamo_graph_capture_for_export(f))
                    # We can't serialize entire fake mode yet, so this is to make sure
                    # things like copy.deepcopy(ep.graph_module) not crash.
                    # see test_export.py::test_custom_tag_metadata_re_export
                    # Once we delete the old strict export, we can use
                    gm_torch_level = dynamo_graph_capture(*args, **kwargs)
                    # We can't serialize entire fake mode yet, so this is to make sure
                    # things like copy.deepcopy(ep.graph_module) not crash.
                    # see test_export.py::test_custom_tag_metadata_re_export
                    # Once we delete the old strict export, we can use this fake mode in the
                    # subsequent logic when lowering to aten IR.
                    del gm_torch_level.meta["fake_mode"]

                else:
                    if is_shapes_spec:
                        raise NotImplementedError(
                            "ShapesSpec is not supported on the legacy v1 "
                            "`torch._dynamo.export` path "
                            "(`use_new_tracer_experimental=False`)."
                        )
                    gm_torch_level, _ = torch._dynamo.export(
                        f,
                        dynamic_shapes=dynamic_shapes,  # type: ignore[arg-type]
                        constraints=constraints,  # type: ignore[arg-type]
                        assume_static_by_default=True,
                        tracing_mode="symbolic",
                        disable_constraint_solver=disable_constraint_solver,
                        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
                        _log_export_usage=_log_export_usage,
                        same_signature=same_signature,
                    )(
                        *args,
                        **kwargs,
                    )
                    gm_torch_level.meta["module_call_specs"] = module_call_specs
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: B904
        except GuardOnDataDependentSymNode as e:
            raise UserError(  # noqa: B904
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using torch._check*(). {str(e)}",
                case_name="constrain_as_size_example",
            )

    if isinstance(f, torch.nn.Module) and restore_fqn:
        _restore_state_dict(f, gm_torch_level)

    return gm_torch_level


def _aot_export_joint_with_descriptors(
    stack,
    mod,
    args,
    *,
    kwargs,
    decompositions,
    fake_params_buffers,
    _record_nn_module_stack=True,
):
    from torch._functorch._aot_autograd.graph_compile import aot_stage2_export
    from torch._functorch._aot_autograd.input_output_analysis import (
        create_graph_signature,
    )

    joint_with_descriptors = aot_export_joint_with_descriptors(
        stack,
        mod,
        args,
        kwargs=kwargs,
        decompositions=decompositions,
        _record_nn_module_stack=_record_nn_module_stack,
    )
    # Convert JointWithDescriptors to graph module and ViewAndMutationMeta
    gm, fw_metadata = aot_stage2_export(
        joint_with_descriptors._aot_state,
        joint_with_descriptors._aot_graph_capture,
    )

    if not isinstance(gm, torch.fx.GraphModule):
        raise AssertionError(f"expected gm to be torch.fx.GraphModule, got {type(gm)}")

    # Create GraphSignature from the metadata
    graph_signature = create_graph_signature(
        gm,
        fw_metadata,
        joint_with_descriptors.in_spec,
        joint_with_descriptors.out_spec,
        user_args_flat=pytree.tree_leaves((args, kwargs)),
        params_and_buffers_flat=list(fake_params_buffers.values()),
        param_names=joint_with_descriptors.params_spec,
        buffer_names=joint_with_descriptors.buffers_spec,
        trace_joint=False,
        num_user_fw_outs=None,
        loss_index=None,
    )
    return gm, graph_signature


def _export_to_aten_ir(
    mod: torch.nn.Module,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    constant_attrs: ConstantAttrMap,
    produce_guards_callback=None,
    *,
    transform=lambda x: x,  # TODO(zhxchen17) Revisit if this is needed later.
    pre_dispatch=False,
    decomp_table=None,
    _prettify_placeholder_names: bool = True,
    decompose_custom_triton_ops: bool = False,
) -> ATenExportArtifact:
    custom_triton_ops_decomposition_ctx = (
        nullcontext
        if decompose_custom_triton_ops
        else _disable_custom_triton_op_functional_decomposition
    )
    # This _reparameterize_module makes sure inputs and module.params/buffers have the same fake_mode,
    # otherwise aot_export_module will error out because it sees a mix of fake_modes.
    # And we want aot_export_module to use the fake_tensor mode in dynamo to keep the pipeline easy to reason about.
    with ExitStack() as stack:
        stack.enter_context(
            torch.nn.utils.stateless._reparametrize_module(
                mod,
                fake_params_buffers,
                tie_weights=True,
                strict=True,
                stack_weights=True,
            )
        )
        stack.enter_context(_ignore_backend_decomps())
        stack.enter_context(_compiling_state_context())
        stack.enter_context(custom_triton_ops_decomposition_ctx())
        stack.enter_context(torch.no_grad())

        gm, graph_signature = transform(_aot_export_joint_with_descriptors)(
            stack,
            mod,
            fake_args,
            kwargs=fake_kwargs,
            decompositions=decomp_table,
            fake_params_buffers=fake_params_buffers,
            _record_nn_module_stack=True,
        )

    def _maybe_fixup_gm_and_output_node_meta(old_gm, new_gm):
        if isinstance(old_gm, torch.fx.GraphModule):
            if hasattr(old_gm, "meta"):
                new_gm.meta.update(old_gm.meta)
            old_output_node = list(old_gm.graph.nodes)[-1]
            new_output_node = list(new_gm.graph.nodes)[-1]
            if old_output_node.op != "output" or new_output_node.op != "output":
                raise AssertionError(
                    f"expected both output nodes to have op='output', got old={old_output_node.op!r}, new={new_output_node.op!r}"
                )
            # make sure we don't override any meta
            if "desc" in new_output_node.meta:
                del new_output_node.meta["desc"]
            new_output_node.meta.update(old_output_node.meta)

    # TODO unfortunately preserving graph-level metadata and output node's meta
    # is not working well with aot_export. So we manually copy it.
    # (The node-level meta is addressed above.)
    _maybe_fixup_gm_and_output_node_meta(mod, gm)

    # Run produce guards before we handle runtime asserts.
    # This means we run the export solver before the runtime asserts pass.
    # Right now this doesn't mean much - the export solver is only there for suggested fixes,
    # and we won't even get to constraint solving if that's needed.
    # But if in future we want to control what runtime asserts are emitted for export,
    # or rely on produce_guards + solver for some simplification on runtime asserts, this probably makes sense.
    if produce_guards_callback:
        try:
            produce_guards_callback(gm)
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAINT_VIOLATION, str(e))  # noqa: B904

    return _produce_aten_artifact(
        gm=gm,
        mod=mod,
        constant_attrs=constant_attrs,
        graph_signature=graph_signature,
        pre_dispatch=pre_dispatch,
        fake_args=fake_args,
        fake_kwargs=fake_kwargs,
        fake_params_buffers=fake_params_buffers,
        _prettify_placeholder_names=_prettify_placeholder_names,
    )


def _get_forward_arg_names(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> list[str]:
    """
    Gets the argument names to forward that are used, for restoring the
    original signature when unlifting the exported program module.
    - Positional args: retain the original argument names, and enumerate
        *args as args_0, args_1, ...
    - Keyword args: retain the original kwarg names in the order specified
        by the user. This order seems to matter for the current state of
        export lifted modules.
    """
    sig = inspect.signature(mod.forward)
    _args = sig.bind_partial(*args).arguments

    names: list[str] = []
    for name, value in _args.items():
        # handle variable number of positional args
        if sig.parameters[name].kind == inspect._ParameterKind.VAR_POSITIONAL:
            names.extend([f"{name}_{i}" for i, _ in enumerate(value)])
        else:
            names.append(name)
    # order of kwargs matters for input spec
    if kwargs:
        names.extend(kwargs.keys())

    return names


def _get_non_persistent_buffers(mod: torch.nn.Module) -> set[str]:
    """
    Returns set of non-persistent buffers in a module and its submodules.
    """
    result: set[str] = set()
    for name, m in mod.named_modules(remove_duplicate=False):
        if name:
            result.update(f"{name}.{b}" for b in m._non_persistent_buffers_set)
        else:
            result.update(m._non_persistent_buffers_set)
    return result


def _rewrite_dynamo_tensor_constants(
    orig_mod_state: set[torch.Tensor],
    traced_mod_state: dict[str, torch.Tensor],
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    constants: dict[str, _ConstantAttributeType],
) -> None:
    """
    Dynamo can mark tensor attributes on modules as parameters or buffers.
    Rewrite them to be tensor constants.
    """
    placeholder_lookup = {
        node.name: node for node in gm.graph.find_nodes(op="placeholder")
    }

    for spec in graph_signature.input_specs:
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
            if spec.target is None:
                raise AssertionError(
                    f"spec.target must not be None for {spec.kind} kind"
                )
            value = traced_mod_state[spec.target]
            if value not in orig_mod_state:
                # This was a tensor constant erroneously marked as state.
                # Convert it into a constant in the graph signature, and add its
                # value to the constants table.
                spec.kind = InputKind.CONSTANT_TENSOR
                spec.persistent = None
                if isinstance(value, torch.nn.Parameter):
                    value = value.detach()
                constants[spec.target] = value  # type: ignore[arg-type]

                node = placeholder_lookup[spec.arg.name]
                _set_constant_requires_grad(node, value)


def _move_non_persistent_buffers_to_tensor_constants(
    orig_mod: torch.nn.Module,
    graph_signature: ExportGraphSignature,
    constants: dict[str, _ConstantAttributeType],
) -> None:
    """
    Moves non-persistent buffers to tensor constants.
    """
    for spec in graph_signature.input_specs:
        if spec.kind == InputKind.BUFFER and not spec.persistent:
            if spec.target is None:
                raise AssertionError(
                    "spec.target must not be None for non-persistent BUFFER kind"
                )
            if spec.target in constants:
                raise AssertionError(
                    f"spec.target {spec.target!r} should not already be in constants"
                )
            constants[spec.target] = orig_mod.get_buffer(spec.target)  # type: ignore[arg-type]


def _verify_nn_module_stack(graph_module: torch.fx.GraphModule) -> None:
    """
    Perform nn_module_stack checks on the graph.
    Current constraints:
        For the top level graph:
        - populated for 'call_function', 'get_attr'
        - None for 'placeholder', 'output'
        For submodule graphs:
        - None for 'placeholder', output'

    TODO(pianpwk): make this a consistent node-level check once nn_module_stack is populated for cond submodules.
    """
    # Check top-level graph for all nodes, all graphs for placeholder & output nodes
    for i, mod in enumerate([graph_module] + list(graph_module.modules())):
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in mod.graph.nodes:
            if node.op in ["call_function", "get_attr"]:
                if i == 0:
                    if (
                        nn_module_stack := node.meta.get("nn_module_stack", None)
                    ) is None:
                        raise SpecViolationError(
                            f"Node {node} of type {node.op} is missing nn_module_stack metadata"
                        )
                    if not all(
                        isinstance(k, str)
                        and isinstance(v, tuple)
                        and len(v) == 2
                        and all(isinstance(x, str) for x in v)
                        for k, v in nn_module_stack.items()
                    ):
                        raise SpecViolationError(
                            f"Node {node} of type {node.op} has incorrect nn_module_stack metadata format"
                            f"expected Dict[str, Tuple[str, str]], but got {nn_module_stack}"
                        )
            elif node.op in ["placeholder", "output"]:
                if node.meta.get("nn_module_stack", None):
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} contains nn_module_stack metadata, this should be None"
                    )


def _verify_stack_trace(graph_module: torch.fx.GraphModule) -> None:
    """
    Perform stack trace checks on the graph.
    Constraints:
        - None or non-empty str for 'call_function', 'get_attr'
        - None for 'placeholder', 'output'
    """
    for mod in [graph_module, *graph_module.modules()]:
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in graph_module.graph.nodes:
            stack_trace = node.meta.get("stack_trace", None)
            if node.op in ["call_function", "get_attr"]:
                if not (stack_trace is None or isinstance(stack_trace, str)):
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} has invalid stack_trace metadata, "
                        f"expected a string or None but instead found: {stack_trace}"
                    )
            elif node.op in ["placeholder", "output"]:
                if stack_trace:
                    raise SpecViolationError(
                        f"Node {node} of type {node.op} contains stack_trace metadata, "
                        f"expected None but instead found: {stack_trace}"
                    )


def _verify_placeholder_names(
    gm: torch.fx.GraphModule, sig: ExportGraphSignature
) -> None:
    """
    Performs a sanity check on the placeholder node names.
    - User input nodes: no restrictions, should match the original forward() signature
    - Params/buffers/constants/custom_obj/token nodes: should start with prefixes defined in <placeholder_prefixes>
    """
    name_to_kind = {spec.arg.name: spec.kind for spec in sig.input_specs}
    for mod in gm.modules():
        if not isinstance(mod, torch.fx.GraphModule):
            continue
        for node in mod.graph.nodes:
            if node.op == "placeholder":
                if node.name not in name_to_kind:
                    continue
                node_kind = name_to_kind[node.name]
                prefix = placeholder_prefixes[node_kind]
                if not node.name.startswith(prefix):
                    raise SpecViolationError(
                        f"Placeholder node name {node.name} does not follow spec for {node_kind}, name should have prefix: {prefix}"
                    )


def get_ep_stats(ep: ExportedProgram) -> dict[str, Any]:
    op_count = 0
    op_set = set()
    for m in ep.graph_module.modules():
        if not isinstance(m, torch.fx.GraphModule):
            continue
        for node in m.graph.nodes:
            if node.op != "call_function":
                continue
            op_count += 1
            if not hasattr(node.target, "__module__"):
                raise AssertionError(
                    f"node.target {node.target} must have __module__ attribute"
                )
            if not hasattr(node.target, "__name__"):
                raise AssertionError(
                    f"node.target {node.target} must have __name__ attribute"
                )
            op_set.add(f"{node.target.__module__}.{node.target.__name__}")
    return {"op_count": op_count, "op_set": op_set}


_EXPORT_FLAGS: set[str] | None = None
_EXPORT_MODULE_HIERARCHY: dict[str, str] | None = None


def _log_export_wrapper(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _EXPORT_FLAGS, _EXPORT_MODULE_HIERARCHY
        try:
            start = time.time()
            ep = fn(*args, **kwargs)
            end = time.time()
            log_export_usage(
                event="export.time",
                metrics=end - start,
                flags=_EXPORT_FLAGS,
                **get_ep_stats(ep),
            )
        except Exception as e:
            t = type(e)
            error_type = t.__module__ + "." + t.__qualname__
            case_name = get_class_if_classified_error(e)
            if case_name is not None:
                log.error(exportdb_error_message(case_name))
                log_export_usage(
                    event="export.error.classified",
                    type=error_type,
                    message=str(e),
                    flags=_EXPORT_FLAGS,
                )
            else:
                log_export_usage(
                    event="export.error.unclassified",
                    type=error_type,
                    message=str(e),
                    flags=_EXPORT_FLAGS,
                )

            if hasattr(e, "partial_fx_graph"):
                print(
                    e.partial_fx_graph,
                    file=sys.stderr,
                )

            raise e
        finally:
            _EXPORT_FLAGS = None
            _EXPORT_MODULE_HIERARCHY = None

        return ep

    return wrapper


def _process_jit_trace_inputs_for_export(example_inputs, example_kwarg_inputs):
    if not isinstance(example_inputs, (tuple, list, dict)):
        example_inputs = (example_inputs,)

    elif isinstance(example_inputs, list):
        example_inputs = tuple(example_inputs)

    elif (
        isinstance(example_inputs, (torch.Tensor, dict))
        and example_kwarg_inputs is None
    ):
        example_inputs = (example_inputs,)

    if example_kwarg_inputs is None:
        example_kwarg_inputs = {}
    return example_inputs, example_kwarg_inputs


def _get_original_state_dict(mod: torch.nn.Module) -> dict[str, Any]:
    # Explicitly not calling mod.state_dict() as we do not want the module state for serialization
    # but the running module state so we can always match by id() the entries here with the graph inputs
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    original_state_dict = named_parameters | named_buffers

    non_persistent_buffers = _get_non_persistent_buffers(mod)
    for k in non_persistent_buffers:
        original_state_dict.pop(k, None)

    return original_state_dict


def _process_export_inputs(
    mod: torch.nn.Module,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None,
    dynamic_shapes: _DynamicShapesInput
    | torch.export.AdditionalInputs
    | torch.export.ShapesCollection,
) -> tuple[
    tuple[object, ...],
    dict[str, object],
    TreeSpec,
    _DynamicShapesInput,
    Callable[[ExportedProgram], None],
]:
    """
    Process and validate export inputs for the torch.export API.

    This function validates the input arguments, normalizes kwargs, computes input tree specs,
    and handles special dynamic shapes cases like AdditionalInputs and ShapesCollection.

    Args:
        mod: The PyTorch module to be exported.
        args: Tuple of example positional inputs for the module.
        kwargs: Optional dictionary of example keyword inputs.
        dynamic_shapes: Optional specification for dynamic shapes. Can be:
            - dict mapping argument names to dynamic shape specifications
            - tuple/list specifying dynamic shapes for each input in order
            - torch.export.AdditionalInputs object with verification callback
            - torch.export.ShapesCollection object

    Returns:
        A tuple containing:
        - args: Validated tuple of positional inputs
        - kwargs: Normalized dictionary of keyword inputs (empty dict if None was passed)
        - original_in_spec: TreeSpec representing the flattened input structure
        - dynamic_shapes: Processed dynamic shapes specification
        - verify_additional_inputs: Callback function for additional input verification

    Raises:
        UserError: If args is not a tuple.
    """
    if not isinstance(args, tuple):
        raise UserError(
            UserErrorType.INVALID_INPUT,
            f"Expecting `args` to be a tuple of example positional inputs, got {type(args)}",
        )
    kwargs = kwargs if kwargs is not None else {}
    if pytree.is_namedtuple_instance(args):
        args = tuple(args)

    _, original_in_spec = pytree.tree_flatten((args, kwargs))

    verify_additional_inputs: Callable[[ExportedProgram], None]
    out_dynamic_shapes: _DynamicShapesInput
    if isinstance(dynamic_shapes, torch.export.AdditionalInputs):
        verify_additional_inputs = dynamic_shapes.verify  # type: ignore[assignment]
        out_dynamic_shapes = dynamic_shapes.dynamic_shapes(mod, args, kwargs)  # type: ignore[assignment]
    else:
        verify_additional_inputs = lambda ep: None  # noqa: E731
        if isinstance(dynamic_shapes, torch.export.ShapesCollection):
            out_dynamic_shapes = dynamic_shapes.dynamic_shapes(mod, args, kwargs)  # type: ignore[assignment]
        else:
            out_dynamic_shapes = dynamic_shapes

    if not isinstance(out_dynamic_shapes, (ShapesSpec, ParamsSpec)):
        out_dynamic_shapes = _normalize_dynamic_shapes(
            cast(_DimDynamicShapesSpec | None, out_dynamic_shapes), mod, args, kwargs
        )

    return args, kwargs, original_in_spec, out_dynamic_shapes, verify_additional_inputs


def _get_module_call_graph(
    export_artifact: ExportArtifact,
    preserve_module_call_signature: tuple[str, ...],
    strict_mode_export: bool,
    forward_arg_names: list[str] | None = None,
) -> tuple[torch.fx.GraphModule, list[ModuleCallEntry]]:
    """
    In-place modify the graph module in export_artifact, remove _export_tracepoint nodes and
    return module_call_graph.
    """
    gm: torch.fx.GraphModule = export_artifact.aten.gm
    export_graph_signature: ExportGraphSignature = export_artifact.aten.sig
    module_call_specs: dict[str, dict[str, TreeSpec]] = (
        export_artifact.module_call_specs
    )
    in_spec: TreeSpec = export_artifact.in_spec
    out_spec: TreeSpec = export_artifact.out_spec

    # Make module signatures.
    module_call_signatures: dict[str, ModuleCallSignature] = {}
    for fqn, specs in module_call_specs.items():
        mod_fqn = _strip_root(fqn) if not strict_mode_export else fqn
        module_call_signatures[mod_fqn] = ModuleCallSignature(
            inputs=[],
            outputs=[],
            in_spec=specs["in_spec"],
            out_spec=specs["out_spec"],
            forward_arg_names=None,  # we only propagate forward_arg_names for the top level module
        )

    if len(preserve_module_call_signature) > 0:
        if not strict_mode_export:
            _rewrite_tracepoint_node(gm)
        res = CollectTracepointsPass(module_call_signatures, export_graph_signature)(gm)
        if res is None:
            raise AssertionError("CollectTracepointsPass returned None")
        gm = res.graph_module

    if _EXPORT_MODULE_HIERARCHY is None:
        raise AssertionError("_EXPORT_MODULE_HIERARCHY must not be None")
    module_call_graph = _make_module_call_graph(
        in_spec,
        out_spec,
        module_call_signatures,
        forward_arg_names,
    )
    return gm, module_call_graph


def _get_range_constraints(
    mod: torch.nn.Module,
    export_artifact: ExportArtifact,
    args,
    kwargs,
    dynamic_shapes,
):
    gm: torch.fx.GraphModule = export_artifact.aten.gm
    export_graph_signature: ExportGraphSignature = export_artifact.aten.sig
    fake_mode: FakeTensorMode = export_artifact.fake_mode
    num_lifted = next(
        (
            i
            for i, s in enumerate(export_graph_signature.input_specs)
            if s.kind == InputKind.USER_INPUT
        ),
        len(export_graph_signature.input_specs),
    )
    combined_args, dynamic_shapes = _combine_args_for_tracing(
        mod, args, kwargs, dynamic_shapes
    )

    range_constraints = make_constraints(
        fake_mode,
        gm,
        combined_args,
        dynamic_shapes,
        num_lifted,
    )
    return range_constraints


def _get_inline_constraints(fake_mode: FakeTensorMode):
    if fake_mode.shape_env is None:
        raise AssertionError("fake_mode.shape_env must not be None")
    return {
        k: v
        for k, v in fake_mode.shape_env.var_to_range.items()
        if free_unbacked_symbols(k)
    }


@contextmanager
def patch_forward(obj: torch.nn.Module, new_method):
    """Helper method to make it easier to cleanly torch.export() a method on a
    module that is not `forward`.
    """
    # Save the original method
    original_method = obj.forward

    # Patch the method
    obj.forward = new_method.__get__(obj, obj.__class__)

    try:
        yield
    finally:
        # Restore the original method
        obj.forward = original_method


@contextmanager
def _temp_disable_texpr_fuser():
    original_state = torch._C._jit_texpr_fuser_enabled()
    torch._C._jit_set_texpr_fuser_enabled(False)
    try:
        yield
    finally:
        torch._C._jit_set_texpr_fuser_enabled(original_state)


def _strict_export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    dynamic_shapes: _DynamicShapesInput,
    preserve_module_call_signature: tuple[str, ...],
    orig_in_spec: TreeSpec,
    prefer_deferred_runtime_asserts_over_guards: bool,
    _to_aten_func: Callable,
) -> ExportArtifact:
    """
    _to_aten_func can either be `_export_to_aten_ir_make_fx` or `_export_to_aten_ir`
    """

    gm_torch_level = _export_to_torch_ir(
        # pyrefly: ignore [bad-argument-type]
        mod,
        args,
        kwargs,
        dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        restore_fqn=False,  # don't need to restore because we will do it later
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
        _log_export_usage=False,
    )

    # We detect the fake_mode by looking at gm_torch_level's placeholders, this is the fake_mode created in dynamo.
    (
        fake_args,
        fake_kwargs,
        dynamo_fake_mode,
    ) = _extract_fake_inputs(gm_torch_level, args, kwargs)

    fake_params_buffers = _fakify_params_buffers(dynamo_fake_mode, gm_torch_level)

    # First, we want to pass through the graph to try populating
    # val field for getattr if there is anything missing.
    # This can happen when quantization adds extra params and forgets
    # to update "val"
    for node in gm_torch_level.graph.nodes:
        if node.op == "get_attr" and "val" not in node.meta:
            attr = getattr(gm_torch_level, node.target)
            # Checks if it is not a HigherOrderOp branch or a module
            if not isinstance(attr, torch.nn.Module):
                if dynamo_fake_mode is None:
                    raise AssertionError(
                        "Cannot find dynamo_fake_mode. This could be due to the exported graph module have no placeholders."
                    )
                if is_custom_class(type(attr)):
                    node.meta["val"] = maybe_to_fake_obj(dynamo_fake_mode, attr)
                else:
                    node.meta["val"] = dynamo_fake_mode.from_tensor(
                        attr, static_shapes=True
                    )

    # Fix the graph output signature to be tuple if scalar
    wrap_tuple = False
    # Calling gm_torch_level._out_spec is not safe because gm_torch_level might be
    # a _LazyGraphModule, which does not populate _out_spec when calling recompile().
    # TODO: Fix recompile() in  _LazyGraphModule. T207713214
    if isinstance(gm_torch_level.graph._codegen, torch.fx.graph._PyTreeCodeGen):
        out_spec = orig_out_spec = gm_torch_level.graph._codegen.pytree_info.out_spec
        orig_arg_names = gm_torch_level.graph._codegen.pytree_info.orig_args  # type: ignore[attr-defined]

        # Used to get rid of lint type error.
        if out_spec is None:
            raise AssertionError("out_spec must not be None")
        if out_spec.type not in (list, tuple):
            # aot_export expect the return type to always be a tuple.
            out_spec = pytree.treespec_tuple([out_spec])
            wrap_tuple = True
        gm_torch_level.graph._codegen.pytree_info = _PyTreeInfo(
            orig_arg_names,
            gm_torch_level._in_spec,
            out_spec,
        )
    elif isinstance(
        gm_torch_level.graph._codegen,
        torch._dynamo.functional_export._DynamoBytecodeCodeGen,
    ):
        # Since we're using bytecode codegen, we need to separately apply tuple
        # output instead of modifying pytree spec inplace.
        orig_arg_names = gm_torch_level.graph._codegen.orig_arg_names
        out_spec = orig_out_spec = None
        wrap_tuple = gm_torch_level.graph._codegen.wrap_tuple = True
    else:
