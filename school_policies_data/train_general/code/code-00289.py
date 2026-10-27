        raise RuntimeError(f"Unknown codegen type: {gm_torch_level.graph._codegen}")

    gm_torch_level.recompile()

    _normalize_nn_module_stack(gm_torch_level, type(mod))

    params_buffers_to_node_meta = _collect_param_buffer_metadata(gm_torch_level)

    # When aot_export lifts the params, we lose metadata (e.g. source_fn_stack, stack_trace)
    # from the param nodes as they are treated as fresh inputs
    # Therefore, we manually extract them before calling into aot_export
    # params_buffers_to_node_meta = _collect_param_buffer_metadata(gm_torch_level)

    constant_attrs = _gather_constant_attrs(mod)
    param_buffer_table: dict[str, str] = _get_param_buffer_mapping(mod, gm_torch_level)

    # Dynamo does not track which buffers were registered as non-persistent. This info
    # is available in the original module, so we transfer it to the traced module. Also,
    # since we didn't restore original param/buffer names yet, we must use traced names.
    non_persistent_buffers = _get_non_persistent_buffers(mod)
    reverse_name_lookup = {orig: traced for traced, orig in param_buffer_table.items()}
    gm_torch_level._non_persistent_buffers_set = {
        reverse_name_lookup[name]
        for name in non_persistent_buffers
        if name in reverse_name_lookup
    }

    tx = TracingContext(dynamo_fake_mode)
    with (
        dynamo_fake_mode,
        tracing(tx),
        mock.patch.object(dynamo_fake_mode, "allow_non_fake_inputs", True),
    ):
        aten_export_artifact = _to_aten_func(
            gm_torch_level,
            # NOTE: graph module expects only positional args
            _convert_to_positional_args(orig_arg_names, fake_args, fake_kwargs),
            {},
            fake_params_buffers,
            constant_attrs,
        )

    # Decompose for readability.
    gm = aten_export_artifact.gm
    export_graph_signature = aten_export_artifact.sig
    constants = aten_export_artifact.constants

    _populate_param_buffer_metadata_to_new_gm(
        params_buffers_to_node_meta, gm, export_graph_signature
    )

    # Do some cleanups on the graph module to restore the state dict to the
    # expected form. Each of these steps should probably get fixed upstream.
    # 1. Remove tensor constants that were added as parameters/buffers.
    _rewrite_dynamo_tensor_constants(
        orig_mod_state=set(chain(mod.parameters(), mod.buffers())),
        traced_mod_state=dict(
            chain(
                gm_torch_level.named_parameters(remove_duplicate=False),
                gm_torch_level.named_buffers(remove_duplicate=False),
            )
        ),
        gm=gm,
        graph_signature=export_graph_signature,
        constants=constants,
    )
    # 2. Restore FQN of param/buffers
    _replace_param_buffer_names(param_buffer_table, export_graph_signature)

    # 3. Move non-persistent buffers to tensor constants
    _move_non_persistent_buffers_to_tensor_constants(
        mod, export_graph_signature, constants
    )

    # 4. Rewrite constants to have the same FQN as the original module.
    _remap_constants(constant_attrs, export_graph_signature, constants)

    # 5. Rename constants nodes in graph module from buffers to constants
    _rename_constants_nodes(gm, export_graph_signature)

    if orig_out_spec is None:
        out_spec = aten_export_artifact.inferred_out_spec
        if wrap_tuple:
            out_spec = out_spec.children()[0]
    else:
        out_spec = orig_out_spec
    return ExportArtifact(
        aten=aten_export_artifact,
        in_spec=orig_in_spec,
        out_spec=out_spec,
        fake_mode=dynamo_fake_mode,
        module_call_specs=gm_torch_level.meta["module_call_specs"],
    )


def _export_to_aten_ir_make_fx(
    mod: torch.nn.Module,
    fake_args,
    fake_kwargs,
    fake_params_buffers,
    constant_attrs: ConstantAttrMap,
    produce_guards_callback=None,
    transform=lambda x: x,
) -> ATenExportArtifact:
    def _make_fx_helper(stack, mod, args, kwargs, **flags):
        kwargs = kwargs or {}

        named_parameters = dict(mod.named_parameters(remove_duplicate=False))
        named_buffers = dict(mod.named_buffers(remove_duplicate=False))

        params_and_buffers = {**named_parameters, **named_buffers}
        params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
        params_and_buffers_flat = tuple(params_and_buffers_flat)

        param_len = len(named_parameters)
        buffer_len = len(named_buffers)
        params_len = len(params_and_buffers)

        functional_call = create_functional_call(
            mod, params_spec, params_len, store_orig_mod=True
        )

        params_buffers_args: list[Any] = []
        params_buffers_args.extend(params_and_buffers_flat)
        params_buffers_args.extend(args)

        flat_fn, out_spec = create_tree_flattened_fn(
            functional_call, params_buffers_args, kwargs
        )
        flat_args, in_spec = pytree.tree_flatten((params_buffers_args, kwargs))

        @functools.wraps(flat_fn)
        def wrapped_fn(*args):
            return tuple(flat_fn(*args))

        with enable_python_dispatcher():
            ctx = nullcontext()
            non_strict_root = getattr(mod, "_export_root", None)
            assigned_buffers: dict[str, str] = {}
            hook = None
            if non_strict_root is not None:
                ctx = _detect_attribute_assignment(non_strict_root)  # type: ignore[assignment]

                # For any buffer that is assigned, we want to associate it to the final proxy node
                # that it is assigned to. This node can then be copied into the buffer.
                hook = register_buffer_assignment_hook(
                    non_strict_root, assigned_buffers
                )

            def custom_getattribute(self, attr, *, original_getattr, attrs_to_proxy):
                """
                The idea here is that we override subclass getattr methods to proxy
                inner tensors and metadata. Because of infinite loop shenanigans, we have
                to manually construct the getattr proxy nodes without relying on torch function
                system.
                """
                out = original_getattr(self, attr)
                if attr in attrs_to_proxy:
                    if torch._C._is_torch_function_mode_enabled():
                        if isinstance(out, torch.Tensor):
                            # When we get here there is no guarantee that we will hit the
                            # PreDispatchTorchFunctionMode, so we manually peak into the torch
                            # function mode list and tweak the PreDispatchTorchFunctionMode.
                            # This has side effect of proxying stuff like
                            # proxy.node.meta["val"] = extract_val(val) because at that time, torch function
                            # mode is still active. It seems bad to turn it off inside proxy_tensor.py, so
                            # I guess we will just rely on DCE for now to remove extra stuff like detach
                            torch_function_mode_stack = (
                                torch.overrides._get_current_function_mode_stack()
                            )
                            for mode in torch_function_mode_stack:
                                if isinstance(mode, PreDispatchTorchFunctionMode):
                                    tracer = mode.tracer
                                    proxy = get_proxy_slot(self, tracer).proxy
                                    inner_proxy = tracer.create_proxy(
                                        "call_function",
                                        torch.ops.export.access_subclass_inner_tensor.default,
                                        (proxy, attr),
                                        {},
                                    )
                                    track_tensor_tree(
                                        out, inner_proxy, constant=None, tracer=tracer
                                    )
                return out

            @contextmanager
            def override_getattribute_for_subclasses(args):
                """
                Context manager that temporarily monkey patches
                tensor.__getattribute__ so that we can intercept it at
                torch_function layer.
                """

                # Dictionary that tracks subclass type to original getattr function
                # and the attributes we can proxy.
                tensor_type_to_old_getattribute: dict[
                    type[torch.Tensor], tuple[Callable, set[str]]
                ] = {}
                for arg in args:
                    subclass_types_to_instances: dict[
                        type[torch.Tensor], list[type[torch.Tensor]]
                    ] = get_subclass_typing_container(arg)
                    for subclass_type in subclass_types_to_instances:
                        if subclass_type not in tensor_type_to_old_getattribute:
                            if len(subclass_types_to_instances[subclass_type]) == 0:
                                raise AssertionError(
                                    f"subclass_types_to_instances[{subclass_type}] must not be empty"
                                )
                            instance = subclass_types_to_instances[subclass_type][0]
                            # Query subclass specific attrs
                            attrs_to_proxy = set(dir(instance)) - set(dir(torch.Tensor))
                            tensor_type_to_old_getattribute[subclass_type] = (
                                subclass_type.__getattribute__,  # type: ignore[attr-defined]
                                attrs_to_proxy,
                            )

                try:
                    for k, (
                        old_getattr,
                        attrs_to_proxy,
                    ) in tensor_type_to_old_getattribute.items():
                        custom = functools.partialmethod(
                            custom_getattribute,
                            original_getattr=old_getattr,
                            attrs_to_proxy=attrs_to_proxy,
                        )
                        k.__getattribute__ = custom  # type: ignore[assignment, attr-defined]
                    yield
                finally:
                    for k, (old_getattr, _) in tensor_type_to_old_getattribute.items():
                        k.__getattribute__ = old_getattr  # type: ignore[method-assign, attr-defined]

            @contextmanager
            def _maybe_restore_grad_state():
                """
                When pre-dispatch export accidentally change grad state, we restore it back.
                This can happen when we are calling torch._C._set_grad_enabled directly in the
                forward.
                """
                old_state = torch.is_grad_enabled()
                try:
                    yield
                finally:
                    torch._C._set_grad_enabled(old_state)

            try:
                with (
                    ctx,
                    override_getattribute_for_subclasses(flat_args),
                    _maybe_restore_grad_state(),
                ):
                    gm = make_fx(
                        wrapped_fn,
                        record_module_stack=True,
                        pre_dispatch=True,
                    )(*flat_args)
            finally:
                if hook is not None:
                    hook.remove()
                    hook = None

            if non_strict_root is not None:
                input_names = _graph_input_names(gm)
                buffer_input_names = {
                    name: input_names[param_len + i]
                    for i, (name, buf) in enumerate(non_strict_root._buffers.items())
                    if buf is not None
                }
                output_node = list(gm.graph.nodes)[-1]
                # We copy nodes corresponding to buffer assignments to buffers in the graph.
                for buf, name in assigned_buffers.items():
                    buf_node = _find_node(gm, buffer_input_names[buf])
                    name_node = _find_node(gm, name)
                    with gm.graph.inserting_before(output_node):
                        new_node = gm.graph.create_node(
                            "call_function",
                            torch.ops.aten.copy_.default,
                            args=(buf_node, name_node),
                        )
                        new_node.meta = name_node.meta

            def _is_impure(node):
                if node.op == "call_function" and node.target in (
                    # In export, we ignore any op that is related to
                    # eager mode profiling call. The expectation is
                    # that either runtimes provide their own profiling
                    # OR user wrap the compiled region on a profiling in
                    # later stage.
                    torch.ops.profiler._record_function_enter.default,
                    torch.ops.profiler._record_function_enter_new.default,
                    torch.ops.profiler._record_function_exit._RecordFunction,
                    # In theory, we could fix this dead detach and getattr nodes
                    # from subclass tensors if we carefully rewrite track_tensor_tree
                    # in a way that it doesn't do any tensor methods.
                    torch.ops.aten.detach.default,
                    torch.ops.export.access_subclass_inner_tensor.default,
                ):
                    return False
                return True

            gm.graph.eliminate_dead_code(_is_impure)

        # create graph signature
        if out_spec.spec is None:
            raise AssertionError("out_spec.spec is None!")
        input_names = _graph_input_names(gm)
        output_names = _graph_output_names(gm)
        sig = GraphSignature(
            parameters=list(named_parameters),
            buffers=list(named_buffers),
            # pyrefly: ignore[bad-argument-type]
            user_inputs=input_names[params_len:],
            user_outputs=output_names,
            # pyrefly: ignore[no-matching-overload]
            inputs_to_parameters=dict(zip(input_names[0:param_len], named_parameters)),
            # pyrefly: ignore[no-matching-overload]
            inputs_to_buffers=dict(
                zip(input_names[param_len : param_len + buffer_len], named_buffers)
            ),
            buffers_to_mutate={},
            parameters_to_mutate={},
            user_inputs_to_mutate={},
            in_spec=in_spec,
            out_spec=out_spec.spec,
            backward_signature=None,
            input_tokens=[],
            output_tokens=[],
        )
        return gm, sig

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
        gm, graph_signature = transform(_make_fx_helper)(
            stack,
            mod,
            fake_args,
            trace_joint=False,
            kwargs=fake_kwargs,
        )

        # [NOTE] In training IR, we don't run
        # any DCE as a result we preserve constant
        # nodes in the graph. make_fx invariant is that
        # they don't guarantee every node gets a meta['val']
        # field. Since the actual value is already hardcoded in
        # graph, the node.meta here actually doesn't matter. But
        # we do this to make spec verifier happy.
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and len(node.users) == 0
                and "val" not in node.meta
            ):
                node.meta["val"] = None

        if isinstance(mod, torch.fx.GraphModule) and hasattr(mod, "meta"):
            gm.meta.update(mod.meta)

    # See comment in _export_to_aten_ir()
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
        pre_dispatch=True,
        fake_args=fake_args,
        fake_kwargs=fake_kwargs,
        fake_params_buffers=fake_params_buffers,
    )


def set_missing_meta_vals(gm, flat_args, num_params_buffers):
    # Sets missing metadata to address two problems:
    # 1. aot_export adds symint metadata for placeholders with int values; since
    #    these become specialized, we replace such metadata with the original values.
    # 2. any tensor attributes that are not params / buffers, i.e., are constants
    #    need to have their metadata set before lifting them because it is needed
    #    for computing the exported program's signature.
    index = 0
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if index >= num_params_buffers:
                user_arg = flat_args[index - num_params_buffers]
                if not isinstance(user_arg, torch.Tensor):
                    node.meta["val"] = user_arg
            index += 1


def _find_node(gm: torch.fx.GraphModule, name: str) -> torch.fx.Node:
    return next(iter(node for node in gm.graph.nodes if node.name == name))


def _non_strict_export(
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

    out_spec: TreeSpec | None = None
    in_spec: TreeSpec | None = None

    module_call_specs: dict[str, dict[str, pytree.TreeSpec]] = {}

    def _tuplify_outputs(aot_export):
        def _aot_export_non_strict(stack, mod, args, *, kwargs=None, **flags):
            kwargs = kwargs or {}

            class Wrapper(torch.nn.Module):
                def __init__(self, mod):
                    super().__init__()
                    self._export_root = mod

                def forward(self, *args, **kwargs):
                    nonlocal out_spec
                    nonlocal in_spec
                    mod = self._export_root
                    _, in_spec = pytree.tree_flatten((args, kwargs))
                    if isinstance(mod, torch.fx.GraphModule):
                        # NOTE: We're going to run this graph module with an fx interpreter,
                        # which will not run any forward hooks. Thus, ideally, we should run
                        # all forward hooks here. But the general logic for running them is
                        # complicated (see nn/module.py), and probably not worth duplicating.
                        # Instead we only look for, and run, an export-specific forward hook.
                        if (
                            _check_input_constraints_pre_hook
                            in mod._forward_pre_hooks.values()
                        ):
                            _check_input_constraints_pre_hook(mod, args, kwargs)
                        with torch.fx.traceback.preserve_node_meta():
                            args = (*args, *kwargs.values())
                            tree_out = torch.fx.Interpreter(mod).run(*args)
                    else:
                        tree_out = mod(*args, **kwargs)
                    flat_outs, out_spec = pytree.tree_flatten(tree_out)
                    return tuple(flat_outs)

            wrapped_mod = Wrapper(mod)
            # Patch export_root to the signatures so that wrapper module correctly populates the
            # in/out spec
            new_preserved_call_signatures = [
                "_export_root." + i for i in preserve_module_call_signature
            ]
            ctx = nullcontext()
            if not isinstance(mod, torch.fx.GraphModule):
                ctx = _wrap_submodules(  # type: ignore[assignment]
                    wrapped_mod, new_preserved_call_signatures, module_call_specs
                )
            with ctx:
                gm, sig = aot_export(stack, wrapped_mod, args, kwargs=kwargs, **flags)
            log.debug("Exported program from AOTAutograd:\n%s", gm)

            sig.parameters = pytree.tree_map(_strip_root, sig.parameters)
            sig.buffers = pytree.tree_map(_strip_root, sig.buffers)
            sig.inputs_to_buffers = pytree.tree_map(_strip_root, sig.inputs_to_buffers)
            sig.inputs_to_parameters = pytree.tree_map(
                _strip_root, sig.inputs_to_parameters
            )
            sig.buffers_to_mutate = pytree.tree_map(_strip_root, sig.buffers_to_mutate)
            sig.parameters_to_mutate = pytree.tree_map(
                _strip_root, sig.parameters_to_mutate
            )

            for node in gm.graph.nodes:
                if "nn_module_stack" in node.meta:
                    nn_module_stack = node.meta["nn_module_stack"]
                    node.meta["nn_module_stack"] = {
                        _fixup_key(key): val
                        for key, val in pytree.tree_map(
                            _strip_root, nn_module_stack
                        ).items()
                    }

            return gm, sig

        return _aot_export_non_strict

    # NOTE: We need to enter _compiling_state_context() here so that FakeTensors
    # created for params/buffers are properly tracked for leak detection.
    # See detect_non_strict_fake_tensor_leaks config.
    # We only enter the context if leak detection is enabled to avoid changing
    # behavior when the config is OFF.
    _fakify_ctx = (
        _compiling_state_context()
        if torch._export.config.detect_non_strict_fake_tensor_leaks
        else nullcontext()
    )
    with _fakify_ctx:
        (
            fake_mode,
            fake_args,
            fake_kwargs,
            equalities_inputs,
            original_signature,
            dynamic_shapes,
        ) = make_fake_inputs(
            mod,
            args,
            kwargs,
            dynamic_shapes,
            prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,  # for shape env initialization
        )

        fake_params_buffers = _fakify_params_buffers(fake_mode, mod)

    def _produce_guards_callback(gm):
        return produce_guards_and_solve_constraints(
            fake_mode=fake_mode,
            gm=gm,
            dynamic_shapes=dynamic_shapes,
            equalities_inputs=equalities_inputs,
            original_signature=original_signature,
        )

    tx = TracingContext(fake_mode)

    # We also need to attach dynamo configs as these will be used in HOOs that
    # use torch.compile, like cond
    dynamo_config = dataclasses.asdict(DEFAULT_EXPORT_DYNAMO_CONFIG)
    dynamo_config["do_not_emit_runtime_asserts"] = (
        False  # We want to emit runtime asserts
    )

    with (
        fake_mode,
        _NonStrictTorchFunctionHandler(),
        tracing(tx),
        torch._dynamo.config.patch(dynamo_config),
    ):
        with (
            _fakify_script_objects(mod, fake_args, fake_kwargs, fake_mode) as (
                patched_mod,
                new_fake_args,
                new_fake_kwargs,
                new_fake_constant_attrs,
                map_fake_to_real,
            ),
            _fakify_module_inputs(fake_args, fake_kwargs, fake_mode),
            _override_builtin_ops(),
        ):
            # _to_aten_func is _export_to_aten_ir when using the default non-strict export
            # We need to pass positional args correctly
            aten_export_artifact = _to_aten_func(
                patched_mod,
                new_fake_args,
                new_fake_kwargs,
                fake_params_buffers,
                new_fake_constant_attrs,
                produce_guards_callback=_produce_guards_callback,
                transform=_tuplify_outputs,
            )
            # aten_export_artifact.constants contains only fake script objects, we need to map them back
            aten_export_artifact.constants = {
                fqn: map_fake_to_real[obj] if isinstance(obj, FakeScriptObject) else obj
                for fqn, obj in aten_export_artifact.constants.items()
            }

    _move_non_persistent_buffers_to_tensor_constants(
        mod, aten_export_artifact.sig, aten_export_artifact.constants
    )

    if out_spec is None:
        raise AssertionError("out_spec must not be None")
    if in_spec is None:
        raise AssertionError("in_spec must not be None")

    return ExportArtifact(
        aten=aten_export_artifact,
        in_spec=in_spec,
        out_spec=out_spec,
        fake_mode=fake_mode,
        module_call_specs=module_call_specs,
    )


@_log_export_wrapper
@_disable_prexisiting_fake_mode
def _export_for_training(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    dynamic_shapes: _DynamicShapesInput = None,
    *,
    strict: bool = True,
    preserve_module_call_signature: tuple[str, ...] = (),
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> ExportedProgram:
    global _EXPORT_MODULE_HIERARCHY
    _EXPORT_MODULE_HIERARCHY = _get_module_hierarchy(mod)

    is_shapes_spec = isinstance(dynamic_shapes, (ShapesSpec, ParamsSpec))

    if is_shapes_spec and prefer_deferred_runtime_asserts_over_guards:
        raise ValueError(_SHAPES_SPEC_VS_DEFERRED_RUNTIME_ASSERTS_MSG)

    (
        args,
        kwargs,
        orig_in_spec,
        dynamic_shapes,
        verify_additional_inputs,
    ) = _process_export_inputs(mod, args, kwargs, dynamic_shapes)

    original_state_dict = _get_original_state_dict(mod)

    has_ambient_mode = False
    if not strict:
        flat_args, _ = pytree.tree_flatten((args, kwargs))
        has_ambient_mode = torch._guards.detect_fake_mode(flat_args) is not None

    # Call the appropriate export function based on the strictness of tracing.
    export_func = _strict_export if strict else _non_strict_export

    if not strict and torch._export.config.detect_non_strict_fake_tensor_leaks:
        from torch._subclasses.fake_tensor import fake_tensor_tls

        fake_tensor_tls.non_strict_export_fake_tensor_tracker.clear()

    export_artifact = export_func(
        mod=mod,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        orig_in_spec=orig_in_spec,
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
        _to_aten_func=_export_to_aten_ir_make_fx,
    )

    # If we are tracing with fake inputs, it is expected to
    # see fake tensor constants.
    if not strict and not has_ambient_mode:
        for const, val in export_artifact.aten.constants.items():
            if isinstance(
                val, torch._subclasses.fake_tensor.FakeTensor
            ) and _is_bogus_const_name(const):
                error_msg = (
                    f"We found a fake tensor in the exported program constant's list. "
                    f"This typically means our tracing system encountered an op that "
                    f"we can't trace through. For the potential source, you can refer to "
                    f"following model attribute: {const}. "
                    f"Please file an issue on github. "
                )
                if torch._export.config.error_on_lifted_constant_tensors:
                    raise RuntimeError(error_msg)
                else:
                    warnings.warn(error_msg, stacklevel=2)

    export_graph_signature = export_artifact.aten.sig

    forward_arg_names = _get_forward_arg_names(mod, args, kwargs)
    inline_constraints = _get_inline_constraints(export_artifact.fake_mode)
    # The unbacked symint symbols are updated in aot_export
    # so we serialize them here instead of inside dynamo.
    # Note: _get_range_constraints depends on "inline_constraints" to be set.
    export_artifact.aten.gm.meta["inline_constraints"] = inline_constraints
    range_constraints = _get_range_constraints(
        mod,
        export_artifact,
        args,
        kwargs,
        None if is_shapes_spec else dynamic_shapes,
    )
    # The returned the gm is in-place modified
    gm, module_call_graph = _get_module_call_graph(
        export_artifact,
        preserve_module_call_signature,
        strict,
        forward_arg_names,
    )

    _verify_nn_module_stack(gm)
    _verify_stack_trace(gm)
    _verify_placeholder_names(gm, export_graph_signature)

    _update_gm_meta_if_possible(gm, mod)

    from torch._export.verifier import TrainingIRVerifier

    exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=export_graph_signature,
        state_dict=original_state_dict,
        range_constraints=range_constraints,
        module_call_graph=module_call_graph,
        example_inputs=(args, kwargs),
        constants=export_artifact.aten.constants,
        verifiers=[TrainingIRVerifier],
    )

    verify_additional_inputs(exported_program)

    if not strict and torch._export.config.detect_non_strict_fake_tensor_leaks:
        # See NOTE [export non-strict fake tensor leak detection]
        from torch._subclasses.fake_tensor import fake_tensor_tls
        from torch.fx.experimental.proxy_tensor import (
            _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT,
        )

        active_fakes = fake_tensor_tls.non_strict_export_fake_tensor_tracker
        legit_leak: weakref.WeakSet = find_legit_leaks_from_referrers(active_fakes)
        leak_sources: list[str] = []
        if len(legit_leak) > 0:
            for fake_val in legit_leak:
                if id(fake_val) in _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT:
                    node = _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT[id(fake_val)]
                    stack_trace = node.meta.get("stack_trace")
                    node_name = node.name

                    # If no stack trace on this node (e.g., placeholder), look at users
                    if stack_trace is None:
                        for user in node.users:
                            user_stack = user.meta.get("stack_trace")
                            if user_stack is not None:
                                stack_trace = f"Used by '{user.name}':\n{user_stack}"
                                break

                    stack_trace = (
                        "<no stack trace available>"
                        if stack_trace is None
                        else stack_trace
                    )

                    # Get shape and dtype info
                    shape_info = f"shape={fake_val.shape}, dtype={fake_val.dtype}"
                    leak_info = f"FakeTensor({shape_info}) from node '{node_name}':\n{stack_trace}"
                    leak_sources.append(leak_info)
                else:
                    # Fallback: no proxy mapping found, show basic info
                    shape_info = f"shape={fake_val.shape}, dtype={fake_val.dtype}"
                    leak_info = f"FakeTensor({shape_info}): <no proxy mapping found>"
                    leak_sources.append(leak_info)

            # Format the warning message more nicely
            leak_details = "\n  ".join(leak_sources)
            warnings.warn(
                f"Detected {len(legit_leak)} fake tensors that are still alive after export.\n"
                f"This is likely result of torch.export.export not being able to track side effects "
                f"that is happening outside of model scope.\n\n"
                f"Leaked tensors:\n  {leak_details}\n\n"
                f"Alternatively, please file a bug report to PyTorch team for further debugging help.",
                stacklevel=2,
            )

            del legit_leak

    return exported_program


@_log_export_wrapper
@_disable_prexisiting_fake_mode
@compile_time_strobelight_meta(phase_name="export")
def _export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    dynamic_shapes: _DynamicShapesInput = None,
    *,
    strict: bool = True,
    preserve_module_call_signature: tuple[str, ...] = (),
    pre_dispatch: bool = False,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> ExportedProgram:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a ExportedProgram.

    Args:
        mod: the `nn.Module` to trace.

        args: example positional inputs.

        kwargs: optional example keyword inputs.

        dynamic_shapes:
         An optional argument where the type should either be:
         1) a dict from argument names of ``f`` to their dynamic shape specifications,
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

        preserve_module_call_signature: A list of submodule paths for which the original
            calling conventions are preserved as metadata.

        prefer_deferred_runtime_asserts_over_guards:
         With the current dynamic shapes language for dims and derived dims, we can run into constraints
         that are not expressible with the language. For example, flattening a matrix and adding to a vector,
         both fully dynamic (i.e. x.reshape([-1]) + y) emits a guard s0 * s1 = s2, which is not expressible.
         By default, we either raise a constraint violation error or specialize to static values.
         If this flag is set to True, we avoid erroring out and instead allow complex constraints to exist as runtime
         assertions in the graph. The sympy interpreter (torch/utils/_sympy/interp.py) will produce the math ops
         required to compute and assert the value of the guard (e.g. sym_size_int, eq, _assert_scalar).
         Additionally, if TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS=1 is specified, we will allow complex constraints
         while not emitting runtime asserts, returning a cleaner graph with lesser guarantees around dynamic shapes.

    Returns:
        An ExportedProgram containing the traced module.
    """

    from torch._utils_internal import export_training_ir_rollout_check

    is_shapes_spec = isinstance(dynamic_shapes, (ShapesSpec, ParamsSpec))

    if is_shapes_spec and prefer_deferred_runtime_asserts_over_guards:
        raise ValueError(_SHAPES_SPEC_VS_DEFERRED_RUNTIME_ASSERTS_MSG)

    global _EXPORT_FLAGS, _EXPORT_MODULE_HIERARCHY
    _EXPORT_MODULE_HIERARCHY = _get_module_hierarchy(mod)

    flags = set()
    flags.add("strict" if strict else "non_strict")
    flags.add("pre_dispatch" if pre_dispatch else "aot_dispatch")
    _EXPORT_FLAGS = flags

    log_export_usage(event="export.enter", flags=_EXPORT_FLAGS)

    dtrace_structured("export", payload_fn=lambda: "start!")

    # NOTE Export training IR rollout
    # Old export calls export._trace(pre_dispatch=True)
    # and there are still lot of internal/OSS callsites that
    # use export._trace(pre_dispatch=True) directly. Therefore,
    # it makes more sense to do the switch here.
    # export_training_ir_rollout_check returns True in OSS
    # while internally it returns False UNLESS otherwise specified.
    if pre_dispatch and export_training_ir_rollout_check():
        ep = _export_for_training(
            mod,
            args,
            kwargs,
            dynamic_shapes,
            strict=strict,
            preserve_module_call_signature=preserve_module_call_signature,
            prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
        )
        dtrace_structured("exported_program", payload_fn=lambda: str(ep))
        return ep

    (
        args,
        kwargs,
        original_in_spec,
        dynamic_shapes,
        verify_additional_inputs,
    ) = _process_export_inputs(mod, args, kwargs, dynamic_shapes)

    original_state_dict = _get_original_state_dict(mod)

    # Call the appropriate export function based on the strictness of tracing.
    export_func = _strict_export if strict else _non_strict_export

    export_artifact = export_func(  # type: ignore[operator]
        mod=mod,
        args=args,
        kwargs=kwargs,
        dynamic_shapes=dynamic_shapes,
        preserve_module_call_signature=preserve_module_call_signature,
        orig_in_spec=original_in_spec,
        prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
        _to_aten_func=functools.partial(
            _export_to_aten_ir,
            pre_dispatch=pre_dispatch,
        ),
    )
    export_graph_signature: ExportGraphSignature = export_artifact.aten.sig

    forward_arg_names = _get_forward_arg_names(mod, args, kwargs)
    inline_constraints = _get_inline_constraints(export_artifact.fake_mode)
    # The unbacked symint symbols are updated in aot_export
    # so we serialize them here instead of inside dynamo.
    # Note: this step must be before _get_range_constraints.
    export_artifact.aten.gm.meta["inline_constraints"] = inline_constraints
    range_constraints = _get_range_constraints(
        mod,
        export_artifact,
        args,
        kwargs,
        None if is_shapes_spec else dynamic_shapes,
    )
    gm, module_call_graph = _get_module_call_graph(
        export_artifact,
        preserve_module_call_signature,
        strict,
        forward_arg_names,
    )

    _verify_nn_module_stack(gm)
    _verify_stack_trace(gm)
    _verify_placeholder_names(gm, export_graph_signature)

    # Remove Proxy because they cannot be deepcopied or pickled.
    torch._export.utils.remove_proxy_from_state_dict(original_state_dict, in_place=True)

    from torch._export.verifier import Verifier

    _update_gm_meta_if_possible(gm, mod)

    exported_program = ExportedProgram(
        root=gm,
        graph=gm.graph,
        graph_signature=export_graph_signature,
        state_dict=original_state_dict,
        range_constraints=range_constraints,
        module_call_graph=module_call_graph,
        example_inputs=(args, kwargs),
        constants=export_artifact.aten.constants,
        verifiers=[Verifier],
    )

    dtrace_structured("exported_program", payload_fn=lambda: str(exported_program))

    verify_additional_inputs(exported_program)
    return exported_program
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
from collections.abc import Collection, Mapping
from enum import auto, Enum
from typing import TYPE_CHECKING

from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import get_opaque_type_name, is_custom_class
from torch._subclasses.fake_tensor import is_fake


if TYPE_CHECKING:
    import torch
    from torch._functorch._aot_autograd.schemas import GraphSignature

__all__ = [
    "ConstantArgument",
    "CustomObjArgument",
    "ExportBackwardSignature",
    "ExportGraphSignature",
    "InputKind",
    "InputSpec",
    "OutputKind",
    "OutputSpec",
    "SymIntArgument",
    "SymFloatArgument",
    "SymBoolArgument",
    "TensorArgument",
]


@dataclasses.dataclass
class TensorArgument:
    name: str


@dataclasses.dataclass
class TokenArgument:
    name: str


@dataclasses.dataclass
class SymIntArgument:
    name: str


@dataclasses.dataclass
class SymFloatArgument:
    name: str


@dataclasses.dataclass
class SymBoolArgument:
    name: str


@dataclasses.dataclass
class CustomObjArgument:
    name: str
    class_fqn: str
    fake_val: FakeScriptObject | None = None


@dataclasses.dataclass
class ConstantArgument:
    name: str
    value: int | float | bool | str | None


ArgumentSpec = (
    TensorArgument
    | SymIntArgument
    | SymFloatArgument
    | SymBoolArgument
    | ConstantArgument
    | CustomObjArgument
    | TokenArgument
)


class InputKind(Enum):
    USER_INPUT = auto()
    PARAMETER = auto()
    BUFFER = auto()
    CONSTANT_TENSOR = auto()
    CUSTOM_OBJ = auto()
    TOKEN = auto()


@dataclasses.dataclass
class InputSpec:
    kind: InputKind
    arg: ArgumentSpec
    target: str | None
    persistent: bool | None = None

    def __post_init__(self):
        if self.kind == InputKind.BUFFER:
            if self.persistent is None:
                raise AssertionError("Failed to specify persistent flag on BUFFER.")
        if not isinstance(
            self.arg,
            (
                TensorArgument,
                SymIntArgument,
                SymFloatArgument,
                SymBoolArgument,
                ConstantArgument,
                CustomObjArgument,
                TokenArgument,
            ),
        ):
            raise AssertionError(f"expected valid arg type, got {type(self.arg)}")

    def __str__(self):
        target = "" if self.target is None else f" target='{self.target}'"
        persistent = "" if self.persistent is None else f" persistent={self.persistent}"
        return f"{str(self.arg.name)}: {str(self.kind.name)}{target}{persistent}"


class OutputKind(Enum):
    USER_OUTPUT = auto()
    LOSS_OUTPUT = auto()
    BUFFER_MUTATION = auto()
    PARAMETER_MUTATION = auto()
    GRADIENT_TO_PARAMETER = auto()
    GRADIENT_TO_USER_INPUT = auto()
    USER_INPUT_MUTATION = auto()
    TOKEN = auto()


@dataclasses.dataclass
class OutputSpec:
    kind: OutputKind
    arg: ArgumentSpec
    target: str | None

    def __post_init__(self):
        if not isinstance(
            self.arg,
            (
                TensorArgument,
                SymIntArgument,
                SymFloatArgument,
                SymBoolArgument,
                ConstantArgument,
                TokenArgument,
                CustomObjArgument,
            ),
        ):
            raise AssertionError(f"expected valid arg type, got {self.arg}")

    def __str__(self):
        target = "" if self.target is None else f" target='{self.target}'"
        return f"{str(self.arg.name)}: {str(self.kind.name)}{target}"


@dataclasses.dataclass
class ExportBackwardSignature:
    gradients_to_parameters: dict[str, str]
    gradients_to_user_inputs: dict[str, str]
    loss_output: str


@dataclasses.dataclass
class ExportGraphSignature:
    """
    :class:`ExportGraphSignature` models the input/output signature of Export Graph,
    which is a fx.Graph with stronger invariants guarantees.

    Export Graph is functional and does not access "states" like parameters
    or buffers within the graph via ``getattr`` nodes. Instead, :func:`export`
    guarantees that parameters, buffers, and constant tensors are lifted out of
    the graph as inputs.  Similarly, any mutations to buffers are not included
    in the graph either, instead the updated values of mutated buffers are
    modeled as additional outputs of Export Graph.

    The ordering of all inputs and outputs are::

        Inputs = [*parameters_buffers_constant_tensors, *flattened_user_inputs]
        Outputs = [*mutated_inputs, *flattened_user_outputs]

    e.g. If following module is exported::

        class CustomModule(nn.Module):
            def __init__(self) -> None:
                super(CustomModule, self).__init__()

                # Define a parameter
                self.my_parameter = nn.Parameter(torch.tensor(2.0))

                # Define two buffers
                self.register_buffer("my_buffer1", torch.tensor(3.0))
                self.register_buffer("my_buffer2", torch.tensor(4.0))

            def forward(self, x1, x2):
                # Use the parameter, buffers, and both inputs in the forward method
                output = (
                    x1 + self.my_parameter
                ) * self.my_buffer1 + x2 * self.my_buffer2

                # Mutate one of the buffers (e.g., increment it by 1)
                self.my_buffer2.add_(1.0)  # In-place addition

                return output


        mod = CustomModule()
        ep = torch.export.export(mod, (torch.tensor(1.0), torch.tensor(2.0)))

    Resulting Graph is non-functional::

        graph():
            %p_my_parameter : [num_users=1] = placeholder[target=p_my_parameter]
            %b_my_buffer1 : [num_users=1] = placeholder[target=b_my_buffer1]
            %b_my_buffer2 : [num_users=2] = placeholder[target=b_my_buffer2]
            %x1 : [num_users=1] = placeholder[target=x1]
            %x2 : [num_users=1] = placeholder[target=x2]
            %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x1, %p_my_parameter), kwargs = {})
            %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %b_my_buffer1), kwargs = {})
            %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x2, %b_my_buffer2), kwargs = {})
            %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
            %add_ : [num_users=0] = call_function[target=torch.ops.aten.add_.Tensor](args = (%b_my_buffer2, 1.0), kwargs = {})
            return (add_1,)

    Resulting ExportGraphSignature of the non-functional Graph would be::

        # inputs
        p_my_parameter: PARAMETER target='my_parameter'
        b_my_buffer1: BUFFER target='my_buffer1' persistent=True
        b_my_buffer2: BUFFER target='my_buffer2' persistent=True
        x1: USER_INPUT
        x2: USER_INPUT

        # outputs
        add_1: USER_OUTPUT

    To get a functional Graph, you can use :func:`run_decompositions`::

        mod = CustomModule()
        ep = torch.export.export(mod, (torch.tensor(1.0), torch.tensor(2.0)))
        ep = ep.run_decompositions()

    Resulting Graph is functional::

        graph():
            %p_my_parameter : [num_users=1] = placeholder[target=p_my_parameter]
            %b_my_buffer1 : [num_users=1] = placeholder[target=b_my_buffer1]
            %b_my_buffer2 : [num_users=2] = placeholder[target=b_my_buffer2]
            %x1 : [num_users=1] = placeholder[target=x1]
            %x2 : [num_users=1] = placeholder[target=x2]
            %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%x1, %p_my_parameter), kwargs = {})
            %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %b_my_buffer1), kwargs = {})
            %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%x2, %b_my_buffer2), kwargs = {})
            %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %mul_1), kwargs = {})
            %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%b_my_buffer2, 1.0), kwargs = {})
            return (add_2, add_1)

    Resulting ExportGraphSignature of the functional Graph would be::

        # inputs
        p_my_parameter: PARAMETER target='my_parameter'
        b_my_buffer1: BUFFER target='my_buffer1' persistent=True
        b_my_buffer2: BUFFER target='my_buffer2' persistent=True
        x1: USER_INPUT
        x2: USER_INPUT

        # outputs
        add_2: BUFFER_MUTATION target='my_buffer2'
        add_1: USER_OUTPUT

    """

    input_specs: list[InputSpec]
    output_specs: list[OutputSpec]

    # A list of parameters uniquely identified by mangled fully qualified name
    @property
    def parameters(self) -> Collection[str]:
        return tuple(
            s.target
            for s in self.input_specs
            if s.kind == InputKind.PARAMETER
            if isinstance(s.target, str)
        )

    # A list of buffers uniquely identified by mangled fully qualified name
    @property
    def buffers(self) -> Collection[str]:
        return tuple(
            s.target
            for s in self.input_specs
            if s.kind == InputKind.BUFFER
            if isinstance(s.target, str)
        )

    @property
    def non_persistent_buffers(self) -> Collection[str]:
        return tuple(
            s.target
            for s in self.input_specs
            if s.kind == InputKind.BUFFER
            if s.persistent is False
            if isinstance(s.target, str)
        )

    # A list of lifted constant tensors
    @property
    def lifted_tensor_constants(self) -> Collection[str]:
        return tuple(
            s.target
            for s in self.input_specs
            if s.kind == InputKind.CONSTANT_TENSOR
            if isinstance(s.target, str)
        )

    @property
    def lifted_custom_objs(self) -> Collection[str]:
        return tuple(
            s.target
            for s in self.input_specs
            if s.kind == InputKind.CUSTOM_OBJ
            if isinstance(s.target, str)
        )

    # Graph node names of pytree-flattened inputs of original program
    @property
    def user_inputs(self) -> Collection[int | float | bool | str | None]:
        user_inputs: list[int | float | bool | str | None] = []
        for s in self.input_specs:
            if s.kind != InputKind.USER_INPUT:
                continue

            if isinstance(
                s.arg,
                (
                    TensorArgument,
                    SymIntArgument,
                    SymFloatArgument,
                    SymBoolArgument,
                    CustomObjArgument,
                ),
            ):
                user_inputs.append(s.arg.name)
            elif isinstance(s.arg, ConstantArgument):
                user_inputs.append(s.arg.value)
            else:
                raise RuntimeError(f"{s.arg} is not a valid user input")
        return tuple(user_inputs)

    # Graph node names of pytree-flattened outputs of original program
    # For joint-graph purposes, will include the loss output.
    @property
    def user_outputs(self) -> Collection[int | float | bool | str | None]:
        user_outputs: list[int | float | bool | str | None] = []
        for s in self.output_specs:
            if s.kind not in [
                OutputKind.USER_OUTPUT,
                OutputKind.LOSS_OUTPUT,
            ]:
                continue

            if isinstance(
                s.arg,
                (TensorArgument, SymIntArgument, SymFloatArgument, SymBoolArgument),
            ):
                user_outputs.append(s.arg.name)
            elif isinstance(s.arg, ConstantArgument):
                user_outputs.append(s.arg.value)
            elif isinstance(s.arg, CustomObjArgument):
                user_outputs.append(s.arg.name)
            else:
                raise RuntimeError(f"{s.arg} is not a valid user output")
        return tuple(user_outputs)

    # A dictionary mapping graph input node names to parameters. If a graph input
    # name is found in this dictionary, it is guaranteed to be a lifted parameter.
    @property
    def inputs_to_parameters(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.input_specs
            if s.kind == InputKind.PARAMETER
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    # A dictionary mapping graph input node names to buffers. If a graph input
    # name is found in this dictionary, it is guaranteed to be a lifted buffer.
    @property
    def inputs_to_buffers(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)  # type: ignore[union-attr, misc]
            for s in self.input_specs
            if s.kind == InputKind.BUFFER
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    # A dictionary mapping graph output node names to buffers that are mutated in the
    # original program. Buffers that are not mutated will not be found in this dictionary.
    @property
    def buffers_to_mutate(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.output_specs
            if s.kind == OutputKind.BUFFER_MUTATION
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    @property
    def parameters_to_mutate(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.output_specs
            if s.kind == OutputKind.PARAMETER_MUTATION
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    @property
    def user_inputs_to_mutate(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.output_specs
            if s.kind == OutputKind.USER_INPUT_MUTATION
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    # A dictionary mapping graph input node names to lifted tensor constants.
    @property
    def inputs_to_lifted_tensor_constants(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.input_specs
            if s.kind == InputKind.CONSTANT_TENSOR
            and isinstance(s.arg, TensorArgument)
            and isinstance(s.target, str)
        )

    @property
    def inputs_to_lifted_custom_objs(self) -> Mapping[str, str]:
        return _immutable_dict(
            (s.arg.name, s.target)
            for s in self.input_specs
            if s.kind == InputKind.CUSTOM_OBJ
            and isinstance(s.arg, CustomObjArgument)
            and isinstance(s.target, str)
        )

    @property
    def backward_signature(self) -> ExportBackwardSignature | None:
        loss_output = None
        gradients_to_parameters: dict[str, str] = {}
        gradients_to_user_inputs: dict[str, str] = {}
        for spec in self.output_specs:
            if spec.kind == OutputKind.LOSS_OUTPUT:
                if loss_output is not None:
                    raise AssertionError("multiple LOSS_OUTPUT specs found")
                if not isinstance(spec.arg, TensorArgument):
                    raise AssertionError(
                        f"expected TensorArgument for LOSS_OUTPUT, got {type(spec.arg)}"
                    )
                loss_output = spec.arg.name
            elif spec.kind == OutputKind.GRADIENT_TO_PARAMETER:
                if not isinstance(spec.target, str):
                    raise AssertionError(
                        f"expected str target for GRADIENT_TO_PARAMETER, got {type(spec.target)}"
                    )
                if not isinstance(spec.arg, TensorArgument):
                    raise AssertionError(
                        f"expected TensorArgument for GRADIENT_TO_PARAMETER, got {type(spec.arg)}"
                    )
                gradients_to_parameters[spec.arg.name] = spec.target
            elif spec.kind == OutputKind.GRADIENT_TO_USER_INPUT:
                if not isinstance(spec.target, str):
                    raise AssertionError(
                        f"expected str target for GRADIENT_TO_USER_INPUT, got {type(spec.target)}"
                    )
                if not isinstance(spec.arg, TensorArgument):
                    raise AssertionError(
                        f"expected TensorArgument for GRADIENT_TO_USER_INPUT, got {type(spec.arg)}"
                    )
                gradients_to_user_inputs[spec.arg.name] = spec.target

        if loss_output is None:
            return None

        return ExportBackwardSignature(
            loss_output=loss_output,
            gradients_to_parameters=gradients_to_parameters,
            gradients_to_user_inputs=gradients_to_user_inputs,
        )

    # Map from assertion dependency token index to assertion dep token output
    # name in output. The shape of output after aot_autograd will be like:
    # (updated_inputs, user_outputs, dep_token).
    @property
    def assertion_dep_token(self) -> Mapping[int, str] | None:
        return None

    @property
    def input_tokens(self) -> Collection[str]:
        input_tokens = []
        for s in self.input_specs:
            if s.kind == InputKind.TOKEN:
                if not isinstance(s.arg, TokenArgument):
                    raise AssertionError(
                        f"expected TokenArgument for TOKEN kind, got {type(s.arg)}"
                    )
                input_tokens.append(s.arg.name)
        return tuple(input_tokens)

    @property
    def output_tokens(self) -> Collection[str]:
        output_tokens = []
        for s in self.output_specs:
            if s.kind == OutputKind.TOKEN:
                if not isinstance(s.arg, TokenArgument):
                    raise AssertionError(
                        f"expected TokenArgument for TOKEN kind, got {type(s.arg)}"
                    )
                output_tokens.append(s.arg.name)
        return tuple(output_tokens)

    def __post_init__(self) -> None:
        assertion_dep_token = self.assertion_dep_token
        if assertion_dep_token is None:
            return
        if len(assertion_dep_token) != 1:
            raise AssertionError(
                f"expected exactly 1 assertion_dep_token, got {len(assertion_dep_token)}"
            )
        assertion_dep_token_index = next(iter(assertion_dep_token.keys()))
        expected_index = len(self.user_outputs) + len(self.buffers_to_mutate)
        if expected_index != assertion_dep_token_index:
            raise AssertionError(
                f"expected assertion_dep_token_index to be {expected_index}, got {assertion_dep_token_index}"
            )

    def replace_all_uses(self, old: str, new: str):
        """
        Replace all uses of the old name with new name in the signature.
        """
        if not isinstance(old, str):
            raise AssertionError(f"expected old to be str, got {type(old)}")
        if not isinstance(new, str):
            raise AssertionError(f"expected new to be str, got {type(new)}")
        arg_types = (
            TensorArgument,
            SymIntArgument,
            SymFloatArgument,
            SymBoolArgument,
            CustomObjArgument,
            TokenArgument,
        )
        for o in self.output_specs:
            if isinstance(o.arg, arg_types):
                if o.arg.name == old:
                    o.arg.name = new
        for i in self.input_specs:
            if isinstance(i.arg, arg_types):
                if i.arg.name == old:
                    i.arg.name = new

    def get_replace_hook(self, replace_inputs=False):
        def _(old, new, user):
            if user.op == "output":
                self.replace_all_uses(old.name, new)
            if replace_inputs and old.op == "placeholder":
                self.replace_all_uses(old.name, new)

        return _

    def __str__(self):
        input_specs = "\n".join(str(s) for s in self.input_specs)
        output_specs = "\n".join(str(s) for s in self.output_specs)
        return f"\n# inputs\n{input_specs}\n\n# outputs\n{output_specs}\n"


def _immutable_dict(items):
    """
    Creates a mapping where items cannot be added, deleted, or updated.
    NOTE: The immutability is shallow (like tuple is an immutable collection).
    """
    from types import MappingProxyType

    return MappingProxyType(dict(items))


def _make_argument_spec(node, token_names) -> ArgumentSpec:
    from torch import ScriptObject, SymBool, SymFloat, SymInt
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(node, (int, bool, float, type(None), str)):
        # For const outputs we just directly return this
        return ConstantArgument(name="", value=node)

    if "val" not in node.meta:
        raise AssertionError(
            f"{node} is not a constant or a node with a 'val' metadata field"
        )
    val = node.meta["val"]
    if node.name in token_names:
        return TokenArgument(name=node.name)
    elif is_fake(val):
        return TensorArgument(name=node.name)
    elif isinstance(val, SymInt):
        return SymIntArgument(name=node.name)
    elif isinstance(val, SymFloat):
        return SymFloatArgument(name=node.name)
    elif isinstance(val, SymBool):
        return SymBoolArgument(name=node.name)
    elif isinstance(val, ScriptObject):
        return CustomObjArgument(name=node.name, class_fqn=val._type().qualified_name())  # type: ignore[attr-defined]
    elif isinstance(val, FakeScriptObject):
        return CustomObjArgument(
            name=node.name, class_fqn=val.script_class_name, fake_val=val
        )
    elif is_custom_class(type(val)):
        return CustomObjArgument(
            name=node.name, class_fqn=get_opaque_type_name(type(val)), fake_val=val
        )
    elif isinstance(val, (int, bool, str, float, type(None))):
        return ConstantArgument(name=node.name, value=val)
    else:
        raise AssertionError(
            f"Encountered an unsupported object of type {type(val)} "
            f"while writing the metadata for exported program"
        )


def _convert_to_export_graph_signature(
    graph_signature: "GraphSignature",
    gm: "torch.fx.GraphModule",
    non_persistent_buffers: set[str],
) -> "ExportGraphSignature":
    from torch.utils import _pytree as pytree

    is_joint = graph_signature.backward_signature is not None

    # unpack objects
    user_inputs = set(graph_signature.user_inputs)
    inputs_to_parameters = graph_signature.inputs_to_parameters
    inputs_to_buffers = graph_signature.inputs_to_buffers
    user_outputs = set(graph_signature.user_outputs)
    buffer_mutations = graph_signature.buffers_to_mutate
    parameter_mutations = graph_signature.parameters_to_mutate
    user_input_mutations = graph_signature.user_inputs_to_mutate
    grad_params = (
        graph_signature.backward_signature.gradients_to_parameter  # type: ignore[union-attr]
        if is_joint
        else {}
    )
    grad_user_inputs = (
        graph_signature.backward_signature.gradients_to_user_inputs  # type: ignore[union-attr]
        if is_joint
        else {}
    )
    loss_output = (
        graph_signature.backward_signature.loss_output  # type: ignore[union-attr]
        if is_joint
        else None
    )
    input_tokens = graph_signature.input_tokens
    output_tokens = graph_signature.output_tokens

    inputs = [
        _make_argument_spec(node, input_tokens)
        for node in gm.graph.nodes
        if node.op == "placeholder"
    ]
    outputs = [
        _make_argument_spec(node, output_tokens)
        for node in pytree.tree_leaves(next(iter(reversed(gm.graph.nodes))).args)
    ]

    def to_input_spec(inp: ArgumentSpec) -> InputSpec:
        if isinstance(inp, TokenArgument):
            return InputSpec(kind=InputKind.TOKEN, arg=inp, target=None)

        if not isinstance(inp, TensorArgument):
            return InputSpec(kind=InputKind.USER_INPUT, arg=inp, target=None)
        name = inp.name
        if name in user_inputs:
            return InputSpec(kind=InputKind.USER_INPUT, arg=inp, target=None)
        elif name in inputs_to_parameters:
            return InputSpec(
                kind=InputKind.PARAMETER,
                arg=inp,
                target=inputs_to_parameters[name],  # type: ignore[index]
            )
        elif name in inputs_to_buffers:
            return InputSpec(
                kind=InputKind.BUFFER,
                arg=inp,
                target=inputs_to_buffers[name],  # type: ignore[index]
                persistent=(inputs_to_buffers[name] not in non_persistent_buffers),  # type: ignore[index]
            )
        else:
            raise AssertionError(f"Unknown tensor input kind: {name}")

    def to_output_spec(idx: int, o: ArgumentSpec) -> OutputSpec:
        if isinstance(o, TokenArgument):
            return OutputSpec(kind=OutputKind.TOKEN, arg=o, target=None)

        if not isinstance(o, TensorArgument):
            return OutputSpec(kind=OutputKind.USER_OUTPUT, arg=o, target=None)
        name = o.name
        if idx < len(buffer_mutations) + len(parameter_mutations) + len(
            user_input_mutations
        ) + len(output_tokens):
            if name in buffer_mutations:
                return OutputSpec(
                    kind=OutputKind.BUFFER_MUTATION,
                    arg=o,
                    target=buffer_mutations[name],  # type: ignore[index]
                )
            elif name in parameter_mutations:
                return OutputSpec(
                    kind=OutputKind.PARAMETER_MUTATION,
                    arg=o,
                    target=parameter_mutations[name],  # type: ignore[index]
                )
            elif name in user_input_mutations:
                return OutputSpec(
                    kind=OutputKind.USER_INPUT_MUTATION,
                    arg=o,
                    target=user_input_mutations[name],  # type: ignore[index]
                )
            else:
                raise AssertionError(f"Unknown tensor mutation kind: {name}")
        else:
            if name in user_outputs:
                return OutputSpec(kind=OutputKind.USER_OUTPUT, arg=o, target=None)

            elif name in grad_params:
                return OutputSpec(
                    kind=OutputKind.GRADIENT_TO_PARAMETER,
                    arg=o,
                    target=grad_params[name],
                )
            elif name in grad_user_inputs:
                return OutputSpec(
                    kind=OutputKind.GRADIENT_TO_USER_INPUT,
                    arg=o,
                    target=grad_user_inputs[name],
                )
            elif name == loss_output:
                return OutputSpec(kind=OutputKind.LOSS_OUTPUT, arg=o, target=None)

            else:
                raise AssertionError(f"Unknown tensor output kind: {name}")

    input_specs = [to_input_spec(inp) for inp in inputs]
    output_specs = [to_output_spec(idx, o) for idx, o in enumerate(outputs)]
    return ExportGraphSignature(input_specs=input_specs, output_specs=output_specs)
<|endoftext|>from dataclasses import dataclass


__all__ = ["ScriptObjectMeta"]


@dataclass
class ScriptObjectMeta:
    """
    Metadata which is stored on nodes representing ScriptObjects.
    """

    # Key into constants table to retrieve the real ScriptObject.
    constant_name: str

    class_fqn: str
<|endoftext|># mypy: allow-untyped-defs
import torch
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode
from torch.overrides import TorchFunctionMode


class AutogradStateOpsFailSafeguard(TorchFunctionMode):
    """
    Detect grad state ops during exporting the graph and fail the process by
    raising an error, to avoid unexpected behavior. Those grad mode ops could be:
    `torch.no_grad`
    `torch.enable_grad`
    `torch.set_grad_enabled`

    Export with predispatch mode is exempted.
    """

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        unsupported_grad_mode_ops = [
            torch._C._set_grad_enabled,
        ]
        # It's only enabled while tracing, by confirming the torch dispatch mode is
        # any active PROXY. This is to allow the autograd ops out of tracing.
        current_state = torch._C.is_grad_enabled()
        if func in unsupported_grad_mode_ops:
            if len(args) != 1:
                raise AssertionError(
                    f"Expected exactly 1 argument for grad mode op, but got {len(args)}"
                )
            changed_state = args[0]
            mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
            # Intend to check if it's not the pre_dispatch mode. It's allowed to use
            # autograd ops in pre_dispatch mode, e.g. `torch.no_grad`
            if (
                mode
                and isinstance(mode, ProxyTorchDispatchMode)
                and not mode.pre_dispatch
                and changed_state != current_state
            ):
                raise RuntimeError(
                    f"Encountered autograd state manager op {func} trying to change global autograd state "
                    "while exporting. This is unsafe because we don't capture this op in torch.export "
                    "today, hence we can't reflect the user intention soundly. You can fix this by "
                    "adding a torch.no_grad() context around the export call."
                )
        return func(*args, **kwargs)
<|endoftext|>import logging
import operator
import types
from collections import defaultdict

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch.export.exported_program import (
    ConstantArgument,
    ExportedProgram,
    ModuleCallSignature,
)
from torch.fx.passes.tools_common import legalize_graph, NodeList
from torch.fx.passes.utils.fuser_utils import erase_nodes, fuse_as_graphmodule


log = logging.getLogger(__name__)


def _get_getitem_users(node: torch.fx.Node) -> set[torch.fx.Node]:
    node_users = list(node.users.keys())
    getitem_users = set()
    for user in node_users:
        if user.op == "output":
            continue

        if not (user.op == "call_function" and user.target is operator.getitem):
            raise AssertionError(
                f"Expected getitem node as user for {node}, instead got {user}"
            )
        getitem_users.update(list(user.users.keys()))
    return getitem_users


def _try_remove_connecting_pytrees(curr_module_node: torch.fx.Node) -> None:
    """
    We want to try to remove extraneous pytree flatten/unflatten calls between modules
    calls. Instead of having the following:
    graph():
        ...
        %foo : [num_users=1] = call_module[target=foo](args = (%getitem_1, %getitem_2), kwargs = {})
        %tree_flatten_spec : [num_users=1] = call_function[target=torch.fx._pytree.tree_flatten_spec](args = (%foo, %_spec_1), kwargs = {})
        %getitem_4 : [num_users=1] = call_function[target=operator.getitem](args = (%tree_flatten_spec, 0), kwargs = {})
        %tree_unflatten_1 : [num_users=2] = call_function[target=torch.utils._pytree.tree_unflatten](args = ([%getitem_4], %_spec_2), kwargs = {})
        %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%tree_unflatten_1, 0), kwargs = {})
        %getitem_7 : [num_users=0] = call_function[target=operator.getitem](args = (%tree_unflatten_1, 1), kwargs = {})
        %getitem_6 : [num_users=1] = call_function[target=operator.getitem](args = (%getitem_5, 0), kwargs = {})
        %bar : [num_users=1] = call_module[target=bar](args = (%getitem_6,), kwargs = {})
        ...

    We could do the following, if we know that all the outputs of `foo` feed into `bar`:
    graph():
        ...
        %foo : [num_users=1] = call_module[target=foo](args = (%getitem_1, %getitem_2), kwargs = {})
        %bar : [num_users=1] = call_module[target=bar](args = (%getitem_6,), kwargs = {})
        ...

    Currently this optimization only works for the case where all of the outputs
    of `foo` go directly into `bar`, and `bar` has no other inputs.
    """

    log.debug("Trying to remove pytrees for module call %s", curr_module_node)

    curr_module_users = list(curr_module_node.users.keys())
    if len(curr_module_users) != 1:
        raise AssertionError(
            f"Expected only one user for module node, instead got {list(curr_module_users)}"
        )
    flatten_node = curr_module_users[0]
    if not (
        flatten_node.op == "call_function"
        and flatten_node.target is fx_pytree.tree_flatten_spec
    ):
        raise AssertionError(
            f"Expected flatten_node to be a call_function with target tree_flatten_spec, "
            f"but got op={flatten_node.op}, target={flatten_node.target}"
        )

    flatten_getitem_users = _get_getitem_users(flatten_node)
    if len(flatten_getitem_users) != 1:
        log.debug(
            "More than one user found for flatten node, %s: %s. "
            "Unable to fuse it with another unflatten call.",
            flatten_node,
            flatten_getitem_users,
        )
        return

    unflatten_node = next(iter(flatten_getitem_users))
    if not (
        unflatten_node.op == "call_function"
        and unflatten_node.target is pytree.tree_unflatten
    ):
        log.debug(
            "Flatten node %s's user is not a pytree.tree_unflatten. "
            "Instead it is: %s. Passing...",
            flatten_node,
            unflatten_node,
        )
        return

    for i, arg in enumerate(unflatten_node.args[0]):  # type: ignore[union-attr,arg-type]
        if arg not in flatten_node.users:
            log.debug(
                "Module %s's outputs are not all directly used as inputs to "
                "the subsequent module. Unable to fuse the connecting "
                "flatten/unflatten. The inputs to the subsequent module are: %s. ",
                curr_module_node,
                unflatten_node.args[0],
            )
            return

        if not (
            # pyrefly: ignore [missing-attribute]
            arg.op == "call_function"
            # pyrefly: ignore [missing-attribute]
            and arg.target is operator.getitem
            # pyrefly: ignore [missing-attribute]
            and arg.args[1] == i
        ):
            log.debug(
                "Module %s's outputs are not all directly used in the same "
                "order as outputted. Unable to fuse the connecting "
                "flatten/unflatten. The inputs to the "
                "subsequent module are: %s. ",
                curr_module_node,
                unflatten_node.args[0],
            )
            return

    # Unflatten has two levels of getitem, because it gets the args and kwargs
    unflatten_getitem_getitem_users = set()
    unflatten_getitem_users = _get_getitem_users(unflatten_node)
    for unflatten_getitem_user in unflatten_getitem_users:
        unflatten_getitem_getitem_users.update(
            list(unflatten_getitem_user.users.keys())
        )

    if len(unflatten_getitem_getitem_users) != 1:
        log.debug(
            "More than one user found for unflatten node, %s: %s. "
            "Unable to fuse it with another flatten call.",
            unflatten_node,
            unflatten_getitem_getitem_users,
        )
        return

    next_module_node = next(iter(unflatten_getitem_getitem_users))
    if next_module_node.op != "call_module":
        log.debug(
            "Unflatten node %s's user is not a call_module. "
            "Instead it is: %s. Passing...",
            unflatten_node,
            next_module_node,
        )
        return

    # Directly put the outputs of the current module into the next module
    next_module_node.args = (curr_module_node,)


def _remove_extraneous_pytrees(gm: torch.fx.GraphModule) -> None:
    """
    Remove extraneous pytree flatten/unflatten calls.

    We try a couple of optimizations here:
        1. Remove pytree flatten/unflatten calls between modules
        2. TODO: Remove module's in_spec + initial unflatten call
        3. TODO: Remove module's out_spec + final flatten call
    """

    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target != "_guards_fn":
            _try_remove_connecting_pytrees(node)

    gm.graph.eliminate_dead_code()


def _construct_inputs(
    gm: torch.fx.GraphModule,
    signature: ModuleCallSignature,
    node_name_map: dict[str, torch.fx.Node],
) -> tuple[list[torch.fx.Node], dict[str, torch.fx.Node]]:
    tree_unflatten_args: list[torch.fx.Node | None] = []
    for input_ in signature.inputs:
        if isinstance(input_, ConstantArgument) and input_.value is None:
            # Constants should be directly embedded into the graph and not used
            # as inputs
            tree_unflatten_args.append(None)
        elif input_.name not in node_name_map:
            # For unused inputs
            tree_unflatten_args.append(None)
        else:
            tree_unflatten_args.append(node_name_map[input_.name])

    # Insert unflatten call
    from .unflatten import _generate_unflatten

    unflatten_node = _generate_unflatten(gm, tree_unflatten_args, signature.in_spec)

    if signature.in_spec.num_children != 2:
        raise AssertionError(
            f"Expected in_spec to have 2 children, but got {signature.in_spec.num_children}"
        )
    if signature.in_spec.type is not tuple:
        raise AssertionError(
            f"Expected in_spec type to be tuple, but got {signature.in_spec.type}"
        )
    args_spec, kwargs_spec = signature.in_spec.children()
    if args_spec.type is not tuple:
        raise AssertionError(
            f"Expected args_spec type to be tuple, but got {args_spec.type}"
        )
    if kwargs_spec.type is not dict:
        raise AssertionError(
            f"Expected kwargs_spec type to be dict, but got {kwargs_spec.type}"
        )

    args_node = gm.graph.call_function(operator.getitem, (unflatten_node, 0))
    args_nodes = [
        gm.graph.call_function(operator.getitem, (args_node, i))
        for i in range(args_spec.num_children)
    ]
    kwargs_node = gm.graph.call_function(operator.getitem, (unflatten_node, 1))
    kwargs_nodes = {
        k: gm.graph.call_function(operator.getitem, (kwargs_node, k))
        for k in kwargs_spec.context
    }
    return args_nodes, kwargs_nodes


def _insert_call_module(
    gm: torch.fx.GraphModule,
    args_nodes: list[torch.fx.Node],
    kwargs_nodes: dict[str, torch.fx.Node],
    module_to_swap: torch.nn.Module,
    name: str,
) -> torch.fx.Node:
    from .unflatten import _assign_attr, _AttrKind

    _assign_attr(module_to_swap, gm, name, _AttrKind.MODULE)
    module_node = gm.graph.call_module(name, tuple(args_nodes), kwargs_nodes)  # type: ignore[arg-type]
    return module_node


def _deconstruct_outputs(
    gm: torch.fx.GraphModule,
    signature: ModuleCallSignature,
    module_node: torch.fx.Node,
    node_name_map: dict[str, torch.fx.Node],
    orig_outputs: tuple[torch.fx.Node, ...],
) -> None:
    from .unflatten import _generate_flatten_spec

    flatten_node = _generate_flatten_spec(gm, module_node, signature.out_spec)

    for i, orig_output in enumerate(orig_outputs):
        # Use Proxy to record getitem access.
        proxy_out = torch.fx.Proxy(flatten_node)[i].node  # type: ignore[index]
        orig_output.replace_all_uses_with(proxy_out, propagate_meta=True)

        node_name_map[orig_output.name] = proxy_out


def _swap_module_helper(
    gm: torch.fx.GraphModule,
    modules_to_swap: dict[str, torch.nn.Module],
    module_call_graph: dict[str, ModuleCallSignature],
) -> torch.fx.GraphModule:
    log.debug("Starting graph:")
    log.debug(gm.graph)

    legalize_graph(gm)

    partitions: dict[str, NodeList] = defaultdict(list)

    node_name_map: dict[str, torch.fx.Node] = {
        node.name: node for node in gm.graph.nodes
    }

    # TODO: Handle the duplicate module case
    for node in gm.graph.nodes:
        if nn_module_stack := node.meta.get("nn_module_stack"):
            for path, _ in nn_module_stack.values():
                if path in modules_to_swap:
                    partitions[path].append(node)
                    break

    for name, nodes in partitions.items():
        """
        Given a graph like the following, and we want to swap out the submodule "foo":
        graph():
            %x : [num_users=1] = placeholder[target=x]
            %y : [num_users=2] = placeholder[target=y]
            %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%y, %x), kwargs = {}), nn_module_stack = {"foo": ("foo", torch.nn.Module)}
            %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%y, %add), kwargs = {}), nn_module_stack = {"bar": ("bar", torch.nn.Module)}
            return (sub,)

        We will first partition out foo's subgraph:
        graph():
            %x : [num_users=1] = placeholder[target=x]
            %y : [num_users=2] = placeholder[target=y]
            %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%y, %x), kwargs = {})
            return add

        And then insert an unflatten + call_module + flatten to replace the subgraph:
        graph():
            %x : [num_users=1] = placeholder[target=x]
            %y : [num_users=1] = placeholder[target=y]

            %_spec_0 : [num_users=1] = get_attr[target=_spec_0]
            %tree_unflatten : [num_users=2] = call_function[target=torch.utils._pytree.tree_unflatten](args = ([%x, %y], %_spec_0), kwargs = {})
            %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%tree_unflatten, 0), kwargs = {})
            %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%getitem, 0), kwargs = {})
            %getitem_2 : [num_users=1] = call_function[target=operator.getitem](args = (%getitem, 1), kwargs = {})
            %getitem_3 : [num_users=0] = call_function[target=operator.getitem](args = (%tree_unflatten, 1), kwargs = {})
            %foo : [num_users=0] = call_module[target=foo](args = (%getitem_1, %getitem_2), kwargs = {})
            %_spec_1 : [num_users=1] = get_attr[target=_spec_1]
            %tree_flatten_spec : [num_users=1] = call_function[target=torch.fx._pytree.tree_flatten_spec](args = (None, %_spec_1), kwargs = {})
            %getitem_4 : [num_users=1] = call_function[target=operator.getitem](args = (%tree_flatten_spec, 0), kwargs = {})

            %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%y, %getitem_4), kwargs = {})
            return (%sub,)

        The `tree_unflatten` call will construct tensor inputs into the input
        format needed by the swapped eager module.
        The `call_module` node should now reference the swapped torch.nn.Module.
        The `tree_flatten_spec` call will deconstruct the eager outputs of the
        swapped module into tensors.
        """

        submod_name = name.replace(".", "_")
        sub_gm, orig_inputs, orig_outputs = fuse_as_graphmodule(
            gm, nodes, f"fused_{submod_name}"
        )

        log.debug("Fused subgraph nodes:")
        log.debug(sub_gm.graph)

        signature: ModuleCallSignature = module_call_graph[name]

        args_nodes, kwargs_nodes = _construct_inputs(gm, signature, node_name_map)
        module_node = _insert_call_module(
            gm, args_nodes, kwargs_nodes, modules_to_swap[name], name
        )
        _deconstruct_outputs(gm, signature, module_node, node_name_map, orig_outputs)

        erase_nodes(gm, nodes)

        log.debug("Swapped graph:")
        log.debug(gm.graph)

    legalize_graph(gm)

    log.debug("Before removing extraneous pytrees:")
    log.debug(gm.graph)

    _remove_extraneous_pytrees(gm)
    log.debug("After removing extraneous pytrees:")
    log.debug(gm.graph)

    gm.recompile()

    return gm


def _fix_input_output_signature(
    gm: torch.fx.GraphModule, signature: ModuleCallSignature
) -> None:
    """
    Given the unlifted module from calling ep.module(), we want to remove the
    pytree processing from the graph module's PyTreeCodeGen and instead make it
    nodes inside of the graph. This allows us to do some optimizations, like
    remove these pytree calls if it is unnecessary, and makes the PyTree part
    more obvious to graph passes.
    """
    from torch.export.unflatten import _generate_flatten, _generate_unflatten

    # Remove the registered pytree codegen because we will take care of it
    # through inserting pytree nodes into the graph
    gm.graph._codegen = torch.fx.graph.CodeGen()

    old_placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]

    new_placeholders = []
    forward_arg_names = signature.forward_arg_names
    if forward_arg_names is None:
        forward_arg_names = []
        if signature.in_spec.num_children != 2:
            raise AssertionError(
                f"Expected in_spec to have 2 children, but got {signature.in_spec.num_children}"
            )
        arg_spec = signature.in_spec.child(0)
        kwarg_spec = signature.in_spec.child(1)
        if arg_spec.type is not tuple:
            raise AssertionError(
                f"Expected arg_spec type to be tuple, but got {arg_spec.type}"
            )
        if kwarg_spec.type is not dict:
            raise AssertionError(
                f"Expected kwarg_spec type to be dict, but got {kwarg_spec.type}"
            )
        for i in range(arg_spec.num_children):
            forward_arg_names.append(f"arg_{i}")
        forward_arg_names.extend(kwarg_spec.context)

    for arg in forward_arg_names:
        with gm.graph.inserting_before(old_placeholders[0]):
            new_placeholders.append(gm.graph.placeholder(arg))

    # Insert flatten call for the inputs
    with gm.graph.inserting_before(old_placeholders[0]):
        flat_node = _generate_flatten(gm, tuple(new_placeholders))
        for i, old_placeholder in enumerate(old_placeholders):
            old_placeholder.op = "call_function"
            old_placeholder.target = operator.getitem
            old_placeholder.args = (flat_node, i)

    # Insert unflatten call for the outputs
    output_node = next(node for node in gm.graph.nodes if node.op == "output")
    with gm.graph.inserting_before(output_node):
        unflat = _generate_unflatten(gm, output_node.args[0], signature.out_spec)
        output_node.args = (unflat,)

    gm.recompile()


def _swap_modules(
    ep: ExportedProgram, modules_to_swap: dict[str, torch.nn.Module]
) -> torch.fx.GraphModule:
    """
    Unlifts the given ExportedProgram into a fx.GraphModule, and then swaps
    previously traced modules with new eager modules specified. Returns a
    fx.GraphModule with a custom forward function.

    Args:
        ep (ExportedProgram): Exported program to modify
        modules_to_swap (Dict[str, torch.nn.Module]): Mapping from module fqn to
            eager module to swap with. The specified module fqn should have also
            been specified in the `preserve_module_call_signature` argument to
            torch.export so that we know how to restore the calling convention
            to this argument.
        run_with_interpreter: Whether or not to run the graph using
            fx.Interpreter. Setting to true will help result in better error
            messages and easier debugging, but it has found to result in a QPS
            drop.
    """
    module_call_graph = {
        entry.fqn: entry.signature for entry in ep.module_call_graph if entry.signature
    }

    gm = ep.module()
    gm.validate_inputs = False  # type: ignore[assignment]
    gm.graph.eliminate_dead_code()  # type: ignore[operator, union-attr]
    if not isinstance(gm, torch.fx.GraphModule):
        raise AssertionError(
            f"Expected gm to be a torch.fx.GraphModule, but got {type(gm)}"
        )
    _fix_input_output_signature(gm, ep.module_call_graph[0].signature)

    gm.module_call_graph = ep.module_call_graph
    gm.train = types.MethodType(type(gm).train, gm)  # type: ignore[assignment]
    gm.eval = types.MethodType(type(gm).eval, gm)  # type: ignore[assignment]

    if not isinstance(gm, torch.fx.GraphModule):
        raise AssertionError(
            f"Expected gm to be a torch.fx.GraphModule, but got {type(gm)}"
        )
    gm = _swap_module_helper(gm, modules_to_swap, module_call_graph)

    return gm
<|endoftext|># mypy: allow-untyped-defs
import abc
import copy
import logging
import operator
import re
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import torch
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class_obj
from torch.export import ExportedProgram
from torch.export._tree_utils import reorder_kwargs
from torch.export.exported_program import (
    ConstantArgument,
    ExportGraphSignature,
    InputKind,
    ModuleCallSignature,
    SymBoolArgument,
    SymFloatArgument,
    SymIntArgument,
    TensorArgument,
)
from torch.fx._symbolic_trace import is_fx_symbolic_tracing
from torch.fx.graph_module import _get_attr, _get_attr_via_attr_list, _print_readable
from torch.utils._pytree import GetAttrKey, SequenceKey

from ._remove_effect_tokens_pass import _remove_effect_tokens


log = logging.getLogger(__name__)


__all__ = [
    "FlatArgsAdapter",
    "InterpreterModule",
    "InterpreterModuleDispatcher",
    "UnflattenedModule",
    "unflatten",
]


class _AttrKind(Enum):
    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT = "constant"
    MODULE = "module"


@dataclass(frozen=True)
class _TensorID:
    """Custom tensor identifier containing storage, stride, and size information."""

    untyped_storage: torch.UntypedStorage
    stride: tuple
    size: tuple
    storage_offset: int


RUN_WITH_INTERPRETER = True


@contextmanager
def _disable_interpreter():
    global RUN_WITH_INTERPRETER
    old_flag = RUN_WITH_INTERPRETER
    RUN_WITH_INTERPRETER = False
    try:
        yield
    finally:
        RUN_WITH_INTERPRETER = old_flag


# Assign attribute 'from_obj' to the qualified name 'target' on 'to_module
# This installs empty Modules where none exist yet if they are subpaths of target
def _assign_attr(
    from_obj: Any,
    to_module: torch.nn.Module,
    target: str,
    attr_kind: _AttrKind,
    persistent: bool = True,
):
    *prefix, field = target.split(".")
    # We need to generate all submodules of `to_module` that are at `prefix` and
    # variants of `prefix` that differ only by call name. All of these submodules
    # will then be assigned `from_obj` at `field` so that they can share this attribute.
    # For example, if target is foo.bar.f, foo has another call name foo@1,
    # and bar has other call names bar@1, bar@2, then we will assign f to
    # foo.bar, foo.bar@1, foo.bar@2, foo@1.bar, foo@1.bar@1, foo@1.bar@2.
    to_modules = {to_module}
    for item in prefix:
        ts: set[torch.nn.Module] = set()
        for to_module in to_modules:
            if not hasattr(to_module, item):
                setattr(to_module, item, torch.nn.Module())
            # Collect `item` and every call-name variant `item@N` present in
            # _modules, regardless of contiguity. Export passes (e.g.
            # replace_set_grad_with_hop_pass) can relocate intermediate calls
            # into HOP subgraphs, leaving non-contiguous indices (e.g. base + @3
            # with @1/@2 absent); scanning contiguously from @1 would stop at the
            # gap and never populate the surviving higher-index copies.
            ts.update(
                t_call  # type: ignore[misc]
                for k, t_call in to_module._modules.items()
                if _is_call_name(k, item)
            )
        to_modules = ts

    for to_module in to_modules:
        if attr_kind == _AttrKind.PARAMETER:
            if not isinstance(from_obj, torch.nn.Parameter):
                raise AssertionError(
                    f"expected torch.nn.Parameter for PARAMETER attr_kind, got {type(from_obj)}"
                )
            to_module.register_parameter(field, from_obj)
        elif attr_kind == _AttrKind.BUFFER:
            if not isinstance(from_obj, torch.Tensor):
                raise AssertionError(
                    f"expected torch.Tensor for BUFFER attr_kind, got {type(from_obj)}"
                )
            to_module.register_buffer(field, from_obj, persistent=persistent)
        elif attr_kind == _AttrKind.CONSTANT:
            if isinstance(from_obj, FakeScriptObject):
                raise AssertionError(
                    "FakeScriptObject should only exist during tracing."
                )
            if not isinstance(
                from_obj,
                (
                    torch.Tensor,
                    torch.ScriptObject,
                ),
            ) and not is_custom_class_obj(from_obj):
                raise AssertionError(
                    f"expected torch.Tensor, torch.ScriptObject, or opaque type for CONSTANT attr_kind, got {type(from_obj)}"
                )
            setattr(to_module, field, from_obj)
        elif attr_kind == _AttrKind.MODULE:
            if not isinstance(from_obj, torch.nn.Module):
                raise AssertionError(
                    f"expected torch.nn.Module for MODULE attr_kind, got {type(from_obj)}"
                )
            setattr(to_module, field, from_obj)


class _SubmoduleBase:
    _ty: str | None

    def type_name(self) -> str | None:
        """
        Subclass of this class - InterpreterModule, InterpreterModuleDispatcher, represents
        corresponding model in eager model. To get this type information for those modules
        in eager model we need to use this method.
        """
        return self._ty


class InterpreterModule(_SubmoduleBase, torch.nn.Module):
    """A module that uses torch.fx.Interpreter to execute instead of the usual
    codegen that GraphModule uses. This provides better stack trace information
    and makes it easier to debug execution.
    """

    graph_module: torch.fx.GraphModule | None

    def __init__(
        self,
        graph: torch.fx.Graph,
        ty: str | None = None,
    ):
        super().__init__()
        self.graph = graph
        self._ty = ty
        self.graph.owning_module = self  # type: ignore[assignment]
        self._run_with_interpreter = RUN_WITH_INTERPRETER

    def forward(self, *args, **kwargs):
        if self.graph_module is None:
            raise AssertionError("Didn't finalize this InterpreterModule")
        if not is_fx_symbolic_tracing() and (
            torch.compiler.is_dynamo_compiling() or not self._run_with_interpreter
        ):
            # Dynamo cannot trace through torch.fx.Interpreter, so fall back to
            # GraphModule codegen in this instance.
            # Patch the codegened forward to run with this InterpreterModule,
            # so attribute accesses, etc. are on this module instead.
            return type(self.graph_module).forward(self, *args, **kwargs)
        else:
            if kwargs:
                # Handle **kwargs. FX only natively supports positional
                # arguments (through placeholders). So in order to pass in
                # kwargs, we must correspond the names of the placeholders with
                # the keys in the kwarg dict.
                arg_list = list(args)
                kwarg_names = self.arg_names[len(arg_list) :]
                arg_list.extend(
                    kwargs[kwarg_name]
                    for kwarg_name in kwarg_names
                    if kwarg_name in kwargs
                )

                # Assert that the kwargs passed in exactly match the positional
                # arguments specified by the GraphModule. This should be
                # guaranteed by the unflattening process.
                if len(kwarg_names) != len(kwargs):
                    raise AssertionError(
                        f"kwarg_names length {len(kwarg_names)} does not match kwargs length {len(kwargs)}"
                    )
                if len(arg_list) != len(self.arg_names):
                    raise AssertionError(
                        f"arg_list length {len(arg_list)} does not match arg_names length {len(self.arg_names)}"
                    )
                args = tuple(arg_list)

            return torch.fx.Interpreter(self, graph=self.graph).run(
                *args, enable_io_processing=False
            )

    def finalize(self):
        # We need to "finalize" because GraphModule populates its own state_dict
        # based on the get_attrs observed in the graph. So we need to fully
        # construct the graph and call _sink_params before generating this
        # GraphModule.

        # need to set `graph_module` directly on the dict to avoid it getting
        # registered as a submodule.
        self.__dict__["graph_module"] = torch.fx.GraphModule(self, self.graph)
        self.graph.lint()

        # Cache arg names for kwarg handling (see forward())
        self.arg_names = []
        for node in self.graph.nodes:
            if node.op == "placeholder":
                self.arg_names.append(node.target)

    def print_readable(
        self,
        print_output=True,
        include_stride=False,
        include_device=False,
        colored=False,
    ):
        return _print_readable(
            self,
            "InterpreterModule",
            print_output,
            include_stride,
            include_device,
            colored,
        )


class InterpreterModuleDispatcher(_SubmoduleBase, torch.nn.Module):
    """
    A module that carries a sequence of InterpreterModules corresponding to
    a sequence of calls of that module. Each call to the module dispatches
    to the next InterpreterModule, and wraps back around after the last.
    """

    def __init__(self, attrs: set[str], call_modules: list[InterpreterModule]):
        super().__init__()
        if not call_modules:
            raise AssertionError("call_modules must not be empty")
        self._modules = call_modules[0]._modules
        for accessor in attrs:
            setattr(self, accessor, getattr(call_modules[0], accessor))
        self._ty = call_modules[0]._ty
        self._call_modules = call_modules
        self._num_calls = 0

    def forward(self, *args, **kwargs):
        call_module = self._call_modules[self._num_calls]
        self._num_calls = (self._num_calls + 1) % len(self._call_modules)
        try:
            return call_module(*args, **kwargs)
        except Exception:
            self._num_calls = 0
            raise

    def call_modules(self):
        return self._call_modules

    def print_readable(
        self,
        print_output=True,
        include_stride=False,
        include_device=False,
        colored=False,
    ):
        outputs = [
            mod.print_readable(
                print_output,
                include_stride,
                include_device,
                colored,
            )
            for mod in self._call_modules
        ]
        return "\n".join(outputs)


class FlatArgsAdapter(abc.ABC):
    """
    Adapts input arguments with ``input_spec`` to align ``target_spec``.
    """

    @abc.abstractmethod
    def adapt(
        self,
        target_spec: pytree.TreeSpec,
        input_spec: pytree.TreeSpec,
        input_args: list[Any],
        metadata: dict[str, Any] | None = None,
        obj: Any | None = None,
    ) -> list[Any]:
        """NOTE: This adapter may mutate given ``input_args_with_path``."""
        ...

    def get_flat_arg_paths(self) -> list[str]:
        """Returns a list of paths that are used to access the flat args."""
        return []


class UnflattenedModule(_SubmoduleBase, torch.nn.Module):
    def __init__(
        self,
        export_module: ExportedProgram,
        flat_args_adapter: FlatArgsAdapter | None = None,
    ):
        super().__init__()
        if export_module.graph_signature.backward_signature is not None:
            raise ValueError("Unflattening on JointExportModule NYI")

        def _id(obj):
            """Returns _TensorID dataclass for tensors, otherwise id()."""
            if isinstance(obj, torch.Tensor):
                return _TensorID(
                    untyped_storage=obj.untyped_storage(),
                    stride=obj.stride(),
                    size=obj.size(),
                    storage_offset=obj.storage_offset(),  # type: ignore[arg-type]
                )
            return id(obj)

        fqn_list = [entry.fqn for entry in export_module.module_call_graph]
        if fqn_list[0] != "":
            raise AssertionError(
                f"expected first fqn to be empty string, got {fqn_list[0]!r}"
            )
        export_graph = deepcopy(export_module.graph)
        self.graph_signature = deepcopy(export_module.graph_signature)
        self.graph = torch.fx.Graph()
        self.graph.owning_module = self  # type: ignore[assignment]
        self.module_call_graph = deepcopy(export_module.module_call_graph)
        self.flat_args_adapter = flat_args_adapter

        self.meta = export_module.graph_module.meta
        self.meta["unflattened_module"] = self

        # Flag to indicate whether args have been adapted.
        self.adapted = False
        self._run_with_interpreter = RUN_WITH_INTERPRETER

        _inplace_buffer_and_input_mutations(export_graph, self.graph_signature)
        _fix_nn_module_stacks(export_graph)
        self._ty = _root_module_type(export_graph)

        self.ivals = _IVals()
        # for any intermediate value of a mutation that is read, track the mutation
        seen_modules, seen_attrs = _outline_submodules(export_graph, self)
        # for each read intermediate value of a mutation, find where it was created,
        # and perform the mutation
        self.ivals.update(seen_modules.values())
        # move attributes that correspond to graph arguments for HOPs
        # from exported program to unflattened submodules
        _copy_graph_attrs(export_module._graph_module, self, seen_attrs)

        self.range_constraints = export_module.range_constraints
        self.equality_constraints: list = []

        # aliasing/unused param or buffer issues:
        # in strict-mode export, dynamo export will deduplicate aliased tensors,
        # and ignore unused tensors. For aliasing, this causes issues when some aliases
        # are unused, and we're unable to match the placeholder node to the correct FQN.
        # This leads to the graph signature potentially having the wrong target FQN,
        # and downstream issues where parameters are assigned to the wrong target attribute,
        # mismatching the relevant placeholder node in the unflattened module.
        # To resolve this we restore (_assign_attr) all aliased/unused tensors in
        # the state_dict as module attributes, but only keep the used tensors in the
        # graph's forward pass (_sink_params).
        state_dict = export_module.state_dict
        assigned_params: set[str] = set()  # tracking unused params
        id_to_param: dict[
            int | _TensorID, torch.nn.Parameter
        ] = {}  # handling weight-sharing
        for name in self.graph_signature.parameters:  # this loop adds used params
            param = state_dict[name]
            if _id(param) not in id_to_param:
                id_to_param[_id(param)] = torch.nn.Parameter(
                    param.clone(), requires_grad=param.requires_grad
                )

            _assign_attr(
                id_to_param[_id(param)],
                self,
                name,
                attr_kind=_AttrKind.PARAMETER,
            )
            assigned_params.add(name)

        non_persistent_buffers = set(self.graph_signature.non_persistent_buffers)
        assigned_buffers: set[str] = set()  # tracking unused buffers
        id_to_buffer: dict[int | _TensorID, tuple[torch.nn.Parameter, bool]] = {}
        for name in self.graph_signature.buffers:  # this loop adds used buffers
            if name in non_persistent_buffers:
                persistent = False
                buffer = export_module.constants[name]
            else:
                persistent = True
                buffer = state_dict[name]

            if _id(buffer) not in id_to_buffer:
                id_to_buffer[_id(buffer)] = (buffer.clone(), persistent)

            _assign_attr(
                id_to_buffer[_id(buffer)][0],
                self,
                name,
                attr_kind=_AttrKind.BUFFER,
                persistent=persistent,
            )
            assigned_buffers.add(name)

        # restore aliased/unused params and buffers
        # these appear in state dict but not graph signature
        for name, tensor in state_dict.items():
            if name in assigned_params or name in assigned_buffers:  # already assigned
                continue

            is_buffer = False
            if _id(tensor) in id_to_buffer or not isinstance(
                tensor, torch.nn.Parameter
            ):  # aliased buffer
                is_buffer = True

            if is_buffer:
                if (
                    _id(tensor) not in id_to_buffer
                ):  # this is completely unused (not weight-sharing)
                    id_to_buffer[_id(tensor)] = (
                        tensor,
                        True,
                    )  # assign to respect original model
                _assign_attr(
                    id_to_buffer[_id(tensor)][0],
                    self,
                    name,
                    attr_kind=_AttrKind.BUFFER,
                    persistent=True,
                )
            else:
                if _id(tensor) not in id_to_param:  # this is unused
                    id_to_param[_id(tensor)] = tensor
                _assign_attr(
                    id_to_param[_id(tensor)],
                    self,
                    name,
                    attr_kind=_AttrKind.PARAMETER,
                )

        # use id map so we don't double-clone aliased constants
        id_to_const: dict[int | _TensorID, torch.Tensor | torch._C.ScriptObject] = {}
        for fqn, constant in export_module.constants.items():
            if _id(constant) not in id_to_const:
                if isinstance(constant, torch.Tensor):
                    constant = constant.clone()
                id_to_const[_id(constant)] = constant
            _constant = id_to_const[_id(constant)]
            _assign_attr(
                _constant,
                self,
                fqn,
                attr_kind=_AttrKind.CONSTANT,
            )

        # This is to handle parameters/buffers that point to the same tensor
        # object id -> list of (node_name, target_name)
        consts_map: dict[int | _TensorID, list[tuple[str, str]]] = defaultdict(list)
        consts_targets: set[str] = set()

        def add_to_consts_map(obj_id, node_name, target_name):
            name_list = consts_map[obj_id]
            name_list.append((node_name, target_name))

        # track aliased/unused params, buffers
        # prefer using untyped_storage() over id() when it's available
        added_params_buffers: set[str] = set()
        for s in self.graph_signature.input_specs:
            if s.kind == InputKind.PARAMETER or (
                s.kind == InputKind.BUFFER and s.persistent
            ):
                if not hasattr(s.arg, "name"):
                    raise AssertionError(
                        f"expected s.arg to have 'name' attribute, got {type(s.arg)}"
                    )
                if not isinstance(s.target, str):
                    raise AssertionError(
                        f"expected s.target to be str, got {type(s.target)}"
                    )
                add_to_consts_map(
                    _id(export_module.state_dict[s.target]),
                    s.arg.name,
                    s.target,
                )
                consts_targets.add(s.target)
                added_params_buffers.add(s.target)
            elif (
                s.kind == InputKind.BUFFER
                and not s.persistent
                or s.kind == InputKind.CONSTANT_TENSOR
                or s.kind == InputKind.CUSTOM_OBJ
            ):
                if not hasattr(s.arg, "name"):
                    raise AssertionError(
                        f"expected s.arg to have 'name' attribute for kind {s.kind}, got {type(s.arg)}"
                    )
                if not isinstance(s.target, str):
                    raise AssertionError(
                        f"expected s.target to be str for kind {s.kind}, got {type(s.target)}"
                    )
                add_to_consts_map(
                    _id(export_module.constants[s.target]),
                    s.arg.name,
                    s.target,
                )
                consts_targets.add(s.target)

        # add constants that are aliased and don't appear in graph signature
        for const_name, const in export_module.constants.items():
            if const_name not in consts_targets:
                const_id = _id(const)
                if const_id not in consts_map:
                    raise AssertionError(
                        f"constant {const_name!r} id not found in consts_map"
                    )
                ph_name, _ = consts_map[const_id][0]
                add_to_consts_map(const_id, ph_name, const_name)
                added_params_buffers.add(s.target)

        # add aliased/unused params and buffers that don't appear in graph signature
        for fqn, tensor in export_module.state_dict.items():
            if fqn not in added_params_buffers:
                tensor_id = _id(tensor)
                if tensor_id not in consts_map:
                    # completely unused (no weight-sharing), ignore.
                    # this weight doesn't appear in graph module,
                    # so won't cause FQN assignment issues
                    continue
                ph_name, _ = consts_map[tensor_id][0]
                add_to_consts_map(tensor_id, ph_name, fqn)

        # node name -> list of possible targets
        inputs_to_state: dict[str, list[str]] = {}
        for node_target in consts_map.values():
            targets = [t[1] for t in node_target]
            for n, _ in node_target:
                inputs_to_state[n] = targets

        _sink_params(self, inputs_to_state, [])

        redirected_call_indices = _deduplicate_modules(seen_modules.values())
        fqn_list = [fqn for fqn in fqn_list if fqn not in redirected_call_indices]

        self._dispatch_modules(redirected_call_indices, consts_targets)
        fqn_list = [fqn for fqn in fqn_list if "@" not in fqn]

        # Cache so we don't have to compute this every time.
        # NOTE: this needs to be kept in sync with the placeholders in
        # self.graph, but currently we have no way to guarantee that.
        self.input_placeholders = [
            node for node in self.graph.nodes if node.op == "placeholder"
        ]
        self.check_input_constraints = True
        # TODO(zhxchen17) We can register modules ahead of time instead of reorder later.
        fqn_order = {fqn: i for i, fqn in enumerate(fqn_list)}
        # In the case of legacy IR, we might be missing some modules from metadata.
        for name, _ in self.named_modules(remove_duplicate=False):
            if name not in fqn_order:
                fqn_order[name] = len(fqn_order)
        _reorder_submodules(self, fqn_order)
        self.graph.lint()
        self.finalize()

    def _print_graph(self):
        for fqn, mod in self.named_modules():
            print(fqn + ":")
            if hasattr(mod, "graph") and isinstance(mod.graph, torch.fx.Graph):
                print(mod.graph)

    def _adapt_flat_args(self, flat_args, in_spec, input):
        signature = self.module_call_graph[0].signature
        if in_spec == signature.in_spec:
            return flat_args

        if self.flat_args_adapter is None:
            raise TypeError(
                "There is no flat args adapter specified. "
                "Are you sure you are calling this with the right arguments? "
            )
        else:
            flat_args = self.flat_args_adapter.adapt(
                target_spec=signature.in_spec,
                input_spec=in_spec,
                input_args=flat_args,
                metadata=self.meta,
                obj=input,
            )

            if len(flat_args) != signature.in_spec.num_leaves:
                raise TypeError(
                    f"Flat args adaption failed, number of args mismatch "
                    f"Adapted: {len(flat_args)} \n"
                    f"Exported module: {signature.in_spec.num_leaves}"
                )
            return flat_args

    def process_forward_inputs(self, *args, **kwargs):
        signature = self.module_call_graph[0].signature

        reordered_kwargs = kwargs
        if kwargs:
            reordered_kwargs = reorder_kwargs(kwargs, signature.in_spec)

        flat_args_with_path, in_spec = pytree.tree_flatten_with_path(
            (args, reordered_kwargs)
        )
        flat_args = [x[1] for x in flat_args_with_path]

        if is_fx_symbolic_tracing():
            return flat_args

        if in_spec != signature.in_spec:
            if not self.adapted:
                print(
                    "Input treespec does not match with exported module's: \n"
                    f"Input treespec: {in_spec}. ",
                    f"Exported module treespec: {signature.in_spec}",
                )
                print("Adapting flat arg to match exported module's treespec")
            flat_args = self._adapt_flat_args(flat_args, in_spec, args)
            self.adapted = True

        if self.check_input_constraints:
            # Import here to avoid an unfortunate circular dependency.
            # TODO(suo): untangle this.
            from torch._export.utils import _check_input_constraints_for_graph

            if self.adapted is True:
                flat_arg_paths = (
                    self.flat_args_adapter.get_flat_arg_paths()
                    if self.flat_args_adapter
                    else []
                )
                if flat_arg_paths and len(flat_arg_paths) != len(flat_args):
                    raise AssertionError(
                        f"flat_arg_paths length {len(flat_arg_paths)} does not match flat_args length {len(flat_args)}"
                    )
                new_flat_args_with_path = [  # type: ignore[var-annotated]
                    (
                        (
                            SequenceKey(idx=idx),
                            GetAttrKey(
                                name=flat_arg_paths[idx]
                                if flat_arg_paths
                                else "<unknown location>"
                            ),
                        ),
                        arg,
                    )
                    for idx, arg in enumerate(flat_args)
                ]
            else:
                new_flat_args_with_path = flat_args_with_path  # type: ignore[assignment]

            _check_input_constraints_for_graph(
                self.input_placeholders, new_flat_args_with_path, self.range_constraints
            )

        return flat_args

    def forward(self, *args, **kwargs):
        flat_args = self.process_forward_inputs(*args, **kwargs)
        signature = self.module_call_graph[0].signature

        if is_fx_symbolic_tracing():
            return_val = torch.fx.Interpreter(self, graph=self.graph).run(
                *flat_args, enable_io_processing=False
            )
            # For scalar return value, fx.Graph wraps in a tuple
            if isinstance(return_val, tuple) and len(return_val) == 1:
                return return_val[0]
            return return_val

        if torch.compiler.is_dynamo_compiling() or not self._run_with_interpreter:
            tree_out = type(self.graph_module).forward(self, *flat_args)  # type: ignore[union-attr]
        else:
            tree_out = torch.fx.Interpreter(self, graph=self.graph).run(
                *flat_args, enable_io_processing=False
            )
        return pytree.tree_unflatten(tree_out, signature.out_spec)

    def finalize(self):
        self.__dict__["graph_module"] = torch.fx.GraphModule(self, self.graph)
        self.graph.lint()

    def _dispatch_modules(self, redirected_call_indices, consts_targets):
        """For a module whose call signatures are preserved, replace
        multiple modules corresponding to multiple calls to that module
        with a single dispatcher module that tracks which module to call.
        """

        # for each fqn whose module call signature is preserved,
        # map that fqn to a list of called modules
        called_modules = defaultdict(list)
        for entry in self.module_call_graph:
            if entry.fqn and entry.signature:
                # some modules were removed and their fqns redirected to other
                # fqns during deduplication
                fqn = entry.fqn
                mod = _get_attr(self, redirected_call_indices.get(fqn, fqn))
                base, idx = fqn.split("@") if "@" in fqn else [fqn, "0"]
                called_modules[base].append((int(idx), mod))

        attrs_map = defaultdict(set)
        for target in consts_targets:
            if "." in target:
                orig_fqn, name = target.rsplit(".", 1)
                attrs_map[orig_fqn].add(name)
            else:
                attrs_map[""].add(target)

        # replace multiple call modules with a single dispatcher module
        for orig_fqn, indexed_call_modules in called_modules.items():
            call_modules = [mod for _, mod in sorted(indexed_call_modules)]
            if len(call_modules) > 1:
                for i in range(len(call_modules)):
                    fqn = _call_name(orig_fqn, i + 1)
                    if fqn not in redirected_call_indices:
                        *prefix, name = fqn.split(".")
                        _get_attr_via_attr_list(self, prefix)._modules.pop(name)
                self.set_submodule(
                    orig_fqn,
                    InterpreterModuleDispatcher(attrs_map[orig_fqn], call_modules),
                )

        # elide call indices in call modules because they are
        # tracked automatically inside the dispatcher module
        def elide_call_indices(prefix, graph):
            for node in graph.nodes:
                if node.op == "call_module":
                    fqn = node.target.split("@")[0]
                    path = f"{prefix}.{fqn}" if prefix else fqn
                    if path in called_modules:
                        node.target = fqn

        for fqn, mod in self.named_modules(remove_duplicate=False):
            if hasattr(mod, "graph"):
                elide_call_indices(fqn, mod.graph)
            elif hasattr(mod, "_call_modules"):
                for mod_ in mod._call_modules:
                    if not hasattr(mod_, "graph"):
                        raise AssertionError(
                            f"expected mod_ to have 'graph' attribute, got {type(mod_)}"
                        )
                    elide_call_indices(fqn, mod_.graph)

    def print_readable(
        self,
        print_output=True,
        include_stride=False,
        include_device=False,
        colored=False,
    ):
        return _print_readable(
            self,
            "UnflattenedModule",
            print_output,
            include_stride,
            include_device,
            colored,
        )


def unflatten(
    module: ExportedProgram, flat_args_adapter: FlatArgsAdapter | None = None
) -> UnflattenedModule:
    """Unflatten an ExportedProgram, producing a module with the same module
    hierarchy as the original eager module. This can be useful if you are trying
    to use :mod:`torch.export` with another system that expects a module
    hierarchy instead of the flat graph that :mod:`torch.export` usually produces.

    .. note:: The args/kwargs of unflattened modules will not necessarily match
        the eager module, so doing a module swap (e.g. :code:`self.submod =
        new_mod`) will not necessarily work. If you need to swap a module out, you
        need to set the :code:`preserve_module_call_signature` parameter of
        :func:`torch.export.export`.

    Args:
        module (ExportedProgram): The ExportedProgram to unflatten.
        flat_args_adapter (Optional[FlatArgsAdapter]): Adapt flat args if input TreeSpec does not match with exported module's.

    Returns:
        An instance of :class:`UnflattenedModule`, which has the same module
        hierarchy as the original eager module pre-export.
    """
    module = _remove_effect_tokens(module)
    m = UnflattenedModule(module, flat_args_adapter)

    # Disable process_forward_inputs as the adapter has many
    # non-dynamo-traceable behavior.
    m.process_forward_inputs = torch._dynamo.disable(  # type: ignore[method-assign]
        m.process_forward_inputs,
        reason="do not trace into preprocessing the inputs",
        recursive=True,
    )

    return m


def _inplace_buffer_and_input_mutations(
    graph: torch.fx.Graph,
    graph_signature: ExportGraphSignature,
) -> None:
    """Transform buffer and input mutations from their functionalized form
    into copy_ nodes in the graph.

    Functionalization represents a buffer mutation by passing the buffer as
    an input and output. For example, consider the eager code:
        def forward(self, x):
            self.buffer += x
            return x * x

    This corresponds to a graph that looks like:
        def forward(self, buffer, x):
            mutated_buffer = aten.add(buffer, x)
            mul = aten.mul(x, x)
            return (mutated_buffer, mul)

    We want to inplace this into something that looks like the original
    eager code:
        def forward(self, buffer, x):
            mutated_buffer = aten.add(buffer, x)
            buffer.copy_(mutated_buffer)
            mul = aten.mul(x, x)
            return (mul,)

    Input mutations are handled similarly.
    """
    output_node = next(iter(reversed(graph.nodes)))
    if output_node.op != "output" or len(output_node.args) != 1:
        raise AssertionError(
            f"expected output node with op='output' and 1 arg, got op={output_node.op!r} with {len(output_node.args)} args"
        )
    return_args = output_node.args[0]

    input_name_to_node = {
        node.name: node for node in graph.nodes if node.op == "placeholder"
    }
    mutation_name_to_input_name = {}

    # Collect mutated buffers.
    buffer_fqn_to_input_name = {
        buffer_fqn: k for k, buffer_fqn in graph_signature.inputs_to_buffers.items()
    }
    mutation_name_to_input_name = {
        k: buffer_fqn_to_input_name[buffer_fqn]
        for k, buffer_fqn in graph_signature.buffers_to_mutate.items()
    }
    # Collect mutated user inputs.
    mutation_name_to_input_name.update(graph_signature.user_inputs_to_mutate)

    num_mutations = len(mutation_name_to_input_name)

    for mutation in return_args[:num_mutations]:
        input_name = mutation_name_to_input_name[mutation.name]
        input_node = input_name_to_node[input_name]

        with graph.inserting_after(mutation):
            # Create a copy_ node that inplaces the mutation.
            new_node = graph.create_node(
                "call_function", torch.ops.aten.copy_.default, (input_node, mutation)
            )
            for k, v in mutation.meta.items():
                new_node.meta[k] = v
        # Replace all uses of the previously functional mutation with
        # our copy_ node.
        mutation.replace_all_uses_with(new_node, lambda x: x is not new_node)

    # Remove the mutated buffer / input from the graph outputs, since we don't
    # need to thread it through anymore.
    user_outputs = tuple(return_args[num_mutations:])
    output_node.args = ((user_outputs),)


def _root_module_type(graph: torch.fx.Graph) -> str | None:
    for node in graph.nodes:
        if "nn_module_stack" not in node.meta:
            continue

        for path, ty in node.meta["nn_module_stack"].values():
            if not path:
                return ty
    return None


def _fix_nn_module_stacks(graph):
    # For each nn module stack in the graph, check if the fqns in it represent a stack:
    # 1. Each fqn must be a prefix of the next fqn.
    # 2. If not, remove the entries starting from the next fqn, emitting a warning.
    for node in graph.nodes:
        if "nn_module_stack" not in node.meta:
            continue

        nn_module_stack = node.meta["nn_module_stack"]
        fqns = [
            fqn.split("@")[0] if "@" in fqn else fqn
            for fqn, _t in nn_module_stack.values()
        ]

        # Check if each FQN is a prefix of the next one
        prev_fqn, *next_fqns = fqns
        num_valid_indices = 1  # root FQN
        for curr_fqn in next_fqns:
            # Check if the previous FQN is a prefix of the current one
            if _is_prefix(prev_fqn, curr_fqn):
                num_valid_indices += 1
                prev_fqn = curr_fqn
            else:
                # Found a non-prefix FQN, stop here
                break

        # If we need to remove entries, create a new stack with only valid entries
        if num_valid_indices < len(nn_module_stack):
            log.warning(
                "nn_module_stack fqns %s at node %s do not form a stack! dropping last %d entries",
                fqns,
                node,
                len(nn_module_stack) - num_valid_indices,
            )
            node.meta["nn_module_stack"] = dict(
                list(nn_module_stack.items())[:num_valid_indices]
            )


def _is_prefix(candidate, target):
    """Check whether `candidate` is a prefix of `target`."""
    return len(candidate) < len(target) and target[: len(candidate)] == candidate


def _compute_accessor(parent_fqn: str, child_fqn: str) -> str:
    if parent_fqn == "":
        # Handle the root module correctly.
        return child_fqn

    parent_split = parent_fqn.split(".")
    child_split = child_fqn.split(".")

    # TODO: support skip connection by inlining the child module.
    if child_split[: len(parent_split)] != parent_split:
        raise RuntimeError(
            f"Child module '{child_fqn}' is not a descendant of parent module '{parent_fqn}'."
            "This is currently unsupported."
            "Please try to make child module attach to parent module directly."
        )
    return ".".join(child_split[len(parent_split) :])


def _check_graph_equivalence(x: torch.nn.Module, y: torch.nn.Module):
    def graph_dump(graph: torch.fx.Graph) -> str:
        ret = []
        nodes_idx: dict[int, int] = {}

        def arg_dump(arg) -> str:
            if isinstance(arg, torch.fx.Node):
                return "%" + str(nodes_idx[id(arg)])
            return str(arg)

        for i, node in enumerate(graph.nodes):
            args_dump = [str(arg) for arg in pytree.tree_map(arg_dump, node.args)]
            args_dump += [
                f"{key}={value}"
                for key, value in pytree.tree_map(arg_dump, node.kwargs).items()
            ]
            target = node.target if node.op in ("call_function", "get_attr") else ""
            # pyrefly: ignore [bad-argument-type]
            ret.append(f"{i}: {node.op}[{target}]({', '.join(args_dump)})")
            nodes_idx[id(node)] = i
        return "\n".join(ret)

    if not isinstance(x.graph, torch.fx.Graph):
        raise AssertionError(
            f"expected x.graph to be torch.fx.Graph, got {type(x.graph)}"
        )
    if not isinstance(y.graph, torch.fx.Graph):
        raise AssertionError(
            f"expected y.graph to be torch.fx.Graph, got {type(y.graph)}"
        )
    return graph_dump(x.graph) == graph_dump(y.graph)


def _add_spec(gm: torch.nn.Module, spec) -> str:
    i = 0
    while hasattr(gm, f"_spec_{i}"):
        i += 1
    name = f"_spec_{i}"
    setattr(gm, name, spec)
    return name


def _generate_flatten(gm: torch.fx.GraphModule, node) -> torch.fx.Node:
    flatten = gm.graph.call_function(pytree.tree_flatten, (node,))
    getitem_0 = gm.graph.call_function(operator.getitem, (flatten, 0))
    return getitem_0


def _generate_flatten_spec(
    gm: torch.fx.GraphModule | InterpreterModule | UnflattenedModule, node, spec
) -> torch.fx.Node:
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(fx_pytree.tree_flatten_spec, (node, spec_node))


def _generate_unflatten(
    gm: torch.fx.GraphModule | InterpreterModule | UnflattenedModule, nodes, spec
) -> torch.fx.Node:
    name = _add_spec(gm, spec)
    spec_node = gm.graph.get_attr(name)
    return gm.graph.call_function(pytree.tree_unflatten, (nodes, spec_node))


def _get_submodule(mod: torch.nn.Module, target: str):
    *prefix, field = target.split(".")

    for item in prefix:
        submod = getattr(mod, item, None)

        if submod is None:
            return None

        if not isinstance(submod, torch.nn.Module):
            return None

        mod = submod

    return getattr(mod, field, None)


def _add_submodule(
    mod: torch.nn.Module,
    target: str,
    module_to_add: torch.nn.Module,
    create_module: Callable[[str], torch.nn.Module] | None = None,
):
    *prefix, field = target.split(".")

    for i, item in enumerate(prefix):
        submod = getattr(mod, item, None)

        if submod is None:
            if create_module is not None:
                submod = create_module(".".join(prefix[: i + 1]))
            else:
                submod = torch.nn.Module()
            setattr(mod, item, submod)

        if not isinstance(submod, torch.nn.Module):
            return False

        mod = submod

    mod.add_module(field, module_to_add)


def _call_name(base: str, n: int) -> str:
    # Given n >= 0, generate call names to a submodule `base` of the form
    # `base`, `base@1`, `base@2`, etc.
    return base if n == 1 else f"{base}@{n - 1}"


def _is_call_name(call_name: str, base: str) -> bool:
    # Recognize when call_name = _call_name(base, n) for some n >= 0.
    return re.match(re.escape(base) + r"(@\d+)?$", call_name) is not None


class _ModuleFrame:
    def __init__(
        self,
        flat_graph: torch.fx.Graph,
        nodes: tuple[torch.fx.Node, ...],
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        parent,
        module_stack: list[tuple[str, str | None, int]],
        module_id,
        module_call_graph: dict[str, ModuleCallSignature],
        module: torch.fx.GraphModule | UnflattenedModule | None = None,
    ):
        self.flat_graph = flat_graph
        self.nodes = nodes
        self.seen_nodes = seen_nodes
        self.seen_modules = seen_modules
        self.seen_attrs = seen_attrs
        self.created_modules = created_modules
        self.parent = parent
        self.module_stack = module_stack
        self.module_id = module_id

        self.module_call_graph = module_call_graph
        self.verbose = False

        self.fqn, ty, num_calls = self.module_stack[-1]
        # generate call name for self.fqn
        self.child_fqn = _call_name(self.fqn, num_calls + 1)

        self.module: torch.fx.GraphModule | UnflattenedModule | InterpreterModule
        if module is not None:
            self.module = module
            self.ivals = module.ivals if hasattr(module, "ivals") else {}  # type: ignore[var-annotated]
        else:
            self.module = self.created_modules.get(
                self.fqn,
                InterpreterModule(torch.fx.Graph(), ty=ty),
            )
            self.ivals = parent.ivals

        self.graph = self.module.graph

        # Mapping of nodes in the flat graph to nodes in this graph.
        self.node_map: dict[torch.fx.Node, torch.fx.Node] = {}
        self.node_to_placeholder = {}

        self.parent_call_module: torch.fx.Node | None = None
        if parent is not None:
            accessor = _compute_accessor(parent.fqn, self.child_fqn)

            def create_module(fqn):
                path = f"{parent.fqn}.{fqn}" if parent.fqn else fqn
                if path in self.created_modules:
                    return self.created_modules[path]
                submod = InterpreterModule(torch.fx.Graph(), ty=ty)
                self.created_modules[path] = submod
                return submod

            _add_submodule(parent.module, accessor, self.module, create_module)
            self.parent_call_module = parent.graph.call_module(accessor)
            if self.seen_modules[self.module_id]:
                base_module_frame = self.seen_modules[self.module_id][0]
                self.module._modules = base_module_frame.module._modules
            self.seen_modules[self.module_id].append(
                _SubmoduleEntry(
                    parent_fqn=self.parent.fqn,
                    parent_module=self.parent.module,
                    parent_call_module=self.parent_call_module,
                    fqn=self.fqn,
                    call_idx=num_calls + 1,
                    module=self.module,
                )
            )

        signature = module_call_graph.get(self.child_fqn)
        if signature is not None and self.parent is not None:
            if signature.in_spec.num_children != 2:
                raise AssertionError(
                    f"expected in_spec to have 2 children, got {signature.in_spec.num_children}"
                )
            if signature.in_spec.type is not tuple:
                raise AssertionError(
                    f"expected in_spec.type to be tuple, got {signature.in_spec.type}"
                )
            args_spec, kwargs_spec = signature.in_spec.children()
            if args_spec.type is not tuple:
                raise AssertionError(
                    f"expected args_spec.type to be tuple, got {args_spec.type}"
                )
            if kwargs_spec.type is not dict:
                raise AssertionError(
                    f"expected kwargs_spec.type to be dict, got {kwargs_spec.type}"
                )

            with self.graph.inserting_after(None):
                arg_nodes = [
                    self.graph.placeholder(f"_positional_arg_{idx}")
                    for idx in range(args_spec.num_children)
                ]
                kwarg_nodes = {}
                for name in kwargs_spec.context:
                    kwarg_nodes[name] = self.graph.placeholder(name)
                flat_args = _generate_flatten_spec(
                    self.module,
                    (tuple(arg_nodes), kwarg_nodes),
                    signature.in_spec,
                )
                for idx, arg in enumerate(signature.inputs):
                    flat_arg_node = self.graph.create_node(
                        op="call_function",
                        target=operator.getitem,
                        args=(flat_args, idx),
                        name=(
                            arg.name
                            if not isinstance(arg, ConstantArgument)
                            else f"_constant_{idx}"
                        ),
                    )
                    if isinstance(arg, ConstantArgument):
                        continue

                    if arg.name in self.seen_nodes:
                        flat_arg_node.meta = copy.copy(self.seen_nodes[arg.name].meta)
                        self.node_to_placeholder[self.seen_nodes[arg.name]] = (
                            flat_arg_node
                        )

            with self.parent.graph.inserting_before(self.parent_call_module):
                input_nodes: list[torch.fx.Node | None] = []
                for input in signature.inputs:
                    if isinstance(input, ConstantArgument):
                        input_nodes.append(input.value)  # type: ignore[arg-type]
                    elif input.name not in self.seen_nodes:
                        input_nodes.append(None)
                    else:
                        if not isinstance(
                            input,
                            (
                                TensorArgument,
                                SymIntArgument,
                                SymBoolArgument,
                                SymFloatArgument,
                            ),
                        ):
                            raise AssertionError(
                                f"expected input to be TensorArgument, SymIntArgument, "
                                f"SymBoolArgument, or SymFloatArgument, got {type(input)}"
                            )
                        input_nodes.append(
                            self.parent.remap_input(self.seen_nodes[input.name])
                        )

                inputs_node = _generate_unflatten(
                    self.parent.module,
                    input_nodes,
                    signature.in_spec,
                )

                args_node = self.parent.graph.call_function(
                    operator.getitem, (inputs_node, 0)
                )
                kwargs_node = self.parent.graph.call_function(
                    operator.getitem, (inputs_node, 1)
                )
                arg_nodes = [
                    self.parent.graph.call_function(operator.getitem, (args_node, i))
                    for i in range(args_spec.num_children)
                ]
                kwarg_nodes = {
                    k: self.parent.graph.call_function(
                        operator.getitem, (kwargs_node, k)
                    )
                    for k in kwargs_spec.context
                }
            if self.parent_call_module is None:
                raise AssertionError("parent_call_module must not be None")
            # pyrefly: ignore [bad-assignment]
            self.parent_call_module.args = tuple(arg_nodes)
            self.parent_call_module.kwargs = kwarg_nodes  # type: ignore[assignment]

    def add_placeholder(self, x):
        if self.fqn == "":
            raise AssertionError(f"Cannot add placeholder {x} to root module")
        if x.graph is not self.flat_graph:
            raise AssertionError(
                "expected x.graph to be flat_graph, got different graph"
            )
        # x is not in subgraph, create a new placeholder for subgraph
        with self.graph.inserting_before(None):
            placeholder_node = self.graph.placeholder(x.name, type_expr=x.type)
        # copy all meta fields, even if some fields might be irrelevant for
        # the placeholder node
        placeholder_node.meta = copy.copy(x.meta)
        self.node_to_placeholder[x] = placeholder_node

    def copy_sym_call_function(self, x):
        # This only exists because we deduplicate sym_size nodes in the flat export graph,
        # and if preserve_module_call_signature is set, we may not be able to pass sym_size
        # nodes, or their downstream users, as inputs to submodule calls.
        # To avoid this we copy these call_function nodes with sym_type results.
        # This should however only be done for sym_type nodes - call_function nodes on tensors
        # should not be deduplicated in the first place.
        args = pytree.tree_map_only(torch.fx.Node, self.remap_input, x.args)
        kwargs = pytree.tree_map_only(torch.fx.Node, self.remap_input, x.kwargs)
        node = self.graph.call_function(x.target, args, kwargs)
        node.meta = copy.copy(x.meta)
        self.node_map[x] = node
        return node

    def remap_input(self, x):
        if x.graph is not self.flat_graph:
            raise AssertionError(
                "expected x.graph to be flat_graph, got different graph"
            )
        if x in self.node_map:
            return self.node_map[x]
        self.print(f"remap_input({x})")
        if x in self.node_to_placeholder:
            return self.node_to_placeholder[x]
        elif (
            x.op == "placeholder" or self.module_call_graph.get(self.fqn) is None
            # allow placeholder creation if we are not preserving module call signature
        ):
            self.add_placeholder(x)
            if self.parent_call_module is not None:
                # Important to *prepend* the output to match how we are
                # inserting placeholder nodes.
                with self.parent.graph.inserting_before(self.parent_call_module):
                    self.parent_call_module.insert_arg(0, self.parent.remap_input(x))
            return self.node_to_placeholder[x]
        elif x.op == "call_function" and (
            x.target
            in (
                torch.ops.aten.sym_size.int,
                torch.ops.aten.item.default,
                torch.ops.aten.unbind.int,
                torch.ops.aten.sum.dim_IntList,
                torch.ops.aten.view.default,
                torch.ops.aten.diff.default,
            )
            or (hasattr(x.target, "__module__") and x.target.__module__ == "_operator")
        ):
            # export deduplicates sym_size nodes, and may need to re-copy them
            # if module call signature needs to be preserved
            self.copy_sym_call_function(x)
            return self.node_map[x]
        elif self.module_call_graph.get(self.fqn) is not None:
            # x is reading the intermediate value of a mutation, so record it;
            # later we will find where it was created and perform the update
            return self.ivals.read(self, x)  # type: ignore[operator, union-attr]
        else:
            raise RuntimeError(
                f"Could not run remap_input() on op type: {x.op} for node {x}"
            )

    def uplift_common_custom_metadata(self) -> None:
        # Copy custom metadata if all nodes have same custom metadata
        custom_meta = None
        for node in self.node_map.values():
            curr_meta = node.meta.get("custom", {})
            if custom_meta is None:
                # first node
                custom_meta = curr_meta
                continue

            if curr_meta != custom_meta:
                custom_meta = {}
                break

        if custom_meta:
            # Lift common custom metadata to parent node and clear children node's custom metadata
            if self.parent_call_module is None:
                raise AssertionError(
                    "parent_call_module must not be None when uplifting custom metadata"
                )
            self.parent_call_module.meta["custom"] = custom_meta
            for node in self.node_map.values():
                del node.meta["custom"]

    def finalize_outputs(self):
        self.created_modules.pop(self.fqn, None)

        orig_outputs = []

        signature = self.module_call_graph.get(self.child_fqn)
        if signature is not None and self.parent is not None:
            for output in signature.outputs:
                if isinstance(
                    output,
                    (
                        TensorArgument,
                        SymIntArgument,
                        SymBoolArgument,
                        SymFloatArgument,
                        ConstantArgument,
                    ),
                ):
                    if output.name in self.seen_nodes:
                        orig_outputs.append(self.seen_nodes[output.name])
                    else:
                        orig_outputs.append(None)
                else:
                    raise RuntimeError(
                        f"Unsupported data type for output node: {output}"
                    )

            def get_actual_output_node(output):
                if output is None:
                    return None

                seen_node = self.seen_nodes[output.name]
                if seen_node in self.node_map:
                    return self.node_map[seen_node]
                elif seen_node in self.node_to_placeholder:
                    return self.node_to_placeholder[seen_node]
                else:
                    raise RuntimeError(
                        f"Could not find output node {output}. Graph: {self.graph}"
                    )

            tree_out_node = _generate_unflatten(
                self.module,
                tuple(get_actual_output_node(output) for output in orig_outputs),
                signature.out_spec,
            )
            parent_out: torch.fx.Node | None = _generate_flatten_spec(
                self.parent.module, self.parent_call_module, signature.out_spec
            )
            graph_outputs: torch.fx.Node | list[torch.fx.Node] = tree_out_node
        else:
            graph_outputs = []
            # Iterate through nodes we have copied into self.graph.
            for orig_node in self.node_map:
                for user_node in orig_node.users:
                    if user_node.name not in self.seen_nodes:
                        # external user node, need to expose as an output
                        orig_outputs.append(orig_node)
                        graph_outputs.append(self.node_map[orig_node])
                        break

            parent_out = self.parent_call_module
            if len(graph_outputs) == 1:
                graph_outputs = graph_outputs[0]

        if not isinstance(graph_outputs, (list, torch.fx.Node)):
            raise AssertionError(
                f"expected graph_outputs to be list or torch.fx.Node, got {type(graph_outputs)}"
            )

        self.graph.output(graph_outputs)

        # Rewrite outputs in parent module
        if parent_out is None:
            return

        parent_out.meta["val"] = (
            graph_outputs.meta.get("val")
            if isinstance(graph_outputs, torch.fx.Node)
            else [o.meta.get("val") for o in graph_outputs]
        )
        self.uplift_common_custom_metadata()

        if len(orig_outputs) == 1 and signature is None:
            self.parent.node_map[orig_outputs[0]] = parent_out
        else:
            for i, orig_output in enumerate(orig_outputs):
                if orig_output is None:
                    continue
                # Use Proxy to record getitem access.
                proxy_out = torch.fx.Proxy(parent_out)[i].node  # type: ignore[index]
                proxy_out.meta["val"] = orig_output.meta.get("val")
                self.parent.node_map[orig_output] = proxy_out

    def copy_node(self, node):
        self.print("copying", node.format_node())
        self.node_map[node] = self.graph.node_copy(node, self.remap_input)
        self.seen_nodes[node.name] = node

    def run_outer(self):
        for i, node in enumerate(self.flat_graph.nodes):
            self.print(i, node.meta.get("nn_module_stack"), node.format_node())

        # Copy all graph inputs
        node_idx: int = 0
        node = self.nodes[node_idx]
        while node.op == "placeholder":
            self.copy_node(node)
            node_idx += 1
            node = self.nodes[node_idx]

        self.run_from(node_idx)

        # Copy graph outputs
        for node in self.flat_graph.nodes:
            if node.op == "output":
                self.copy_node(node)

    def print(self, *args, **kwargs):
        if self.verbose:
            # pyrefly: ignore [not-iterable]
            print(*args, **kwargs)

    def run_from(self, node_idx):
        module_idx = 0
        # Walk through the graph, building up a new graph with the right submodules
        while node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            if node.op == "placeholder":
                raise AssertionError(f"unexpected placeholder node at index {node_idx}")

            self.print()
            self.print("STEP", node_idx, node.format_node())
            self.print(self.module_stack)
            depth = len(self.module_stack)
            if node.op == "output":
                if depth == 1:
                    # We want the output node of the original graph to be handled
                    # specially by the outermost stack frame (in run_outer). So
                    # skip finalization here.
                    return node_idx

                # We've reached the end of the graph. Wrap up all the existing stack frames.
                self.finalize_outputs()
                return node_idx

            if len(node.meta.get("nn_module_stack", {})) == 0:
                raise RuntimeError(f"Unable to find nn_module_stack for node {node}")

            nn_module_stack = node.meta["nn_module_stack"]
            from torch._export.passes._node_metadata_hook import (
                _EMPTY_NN_MODULE_STACK_KEY,
            )

            if (
                len(nn_module_stack) == 1
                and _EMPTY_NN_MODULE_STACK_KEY in nn_module_stack
            ):
                # Empty case from the node_metadata_hook
                node_module_stack = self.module_stack
            else:
                node_module_stack = [
                    (
                        path,
                        ty if path else None,
                        int(k.split("@")[-1]) if "@" in k else 0,
                    )
                    for k, (path, ty) in node.meta["nn_module_stack"].items()
                ]

            if node_module_stack[:depth] != self.module_stack:
                # This means that the current module is done executing and the
                # current node is the beginning of a new module.
                #
                # In this case, we should finalize this module and return without
                # incrementing the node counter.
                self.finalize_outputs()
                self.print("outlining", self.fqn)
                self.print(self.graph)
                return node_idx

            if node_module_stack is None:
                raise AssertionError("node_module_stack must not be None")

            if _is_prefix(self.module_stack, node_module_stack):
                # This means that the current node represents the execution of a new
                # module.
                next_module = node_module_stack[depth]
                self.print("Creating new stack frame for", next_module)
                # Run a nested version of module outliner from the current node
                # counter. Once it is complete, continue from that point.
                next_module_key = list(node.meta["nn_module_stack"].keys())[depth]
                node_idx = _ModuleFrame(
                    self.flat_graph,
                    self.nodes,
                    self.seen_nodes,
                    self.seen_modules,
                    self.seen_attrs,
                    self.created_modules,
                    self,
                    self.module_stack + [next_module],
                    next_module_key.split("@")[0],
                    self.module_call_graph,
                ).run_from(node_idx)
                module_idx += 1
                continue

            # The only remaining possibility is that we are in the right stack
            # frame. Copy the node into this frame's graph and increment the node counter.
            if node_module_stack != self.module_stack:
                raise AssertionError(
                    f"expected node_module_stack {node_module_stack} to equal module_stack {self.module_stack}"
                )

            if node.op == "get_attr":
                # this must be a graph argument for a HOP
                self.seen_attrs[self.child_fqn].add(node.target)

            self.copy_node(node)
            # pyrefly: ignore [unsupported-operation]
            node_idx += 1


@dataclass
class _SubmoduleEntry:
    parent_fqn: str
    parent_module: torch.nn.Module
    parent_call_module: torch.fx.Node
    fqn: str
    call_idx: int
    module: torch.nn.Module


def _outline_submodules(orig_graph: torch.fx.Graph, root_module: UnflattenedModule):
    seen_nodes: dict[str, torch.fx.Node] = {}
    seen_modules: dict[int, list[_SubmoduleEntry]] = defaultdict(list)
    seen_attrs: dict[str, set[str]] = defaultdict(set)
    created_modules: dict[str, torch.nn.Module] = {}
    _ModuleFrame(
        orig_graph,
        tuple(orig_graph.nodes),
        seen_nodes,
        seen_modules,
        seen_attrs,
        created_modules,
        None,
        [("", None, 0)],
        "",
        {
            entry.fqn: entry.signature
            for entry in root_module.module_call_graph
            if entry.signature
        },
        module=root_module,
    ).run_outer()
    return seen_modules, seen_attrs


def _reorder_submodules(
    parent: torch.nn.Module, fqn_order: dict[str, int], prefix: str = ""
):
    # TODO Can be optimized by adding submodules ahead of time.
    if prefix == "":
        for fqn in list(fqn_order.keys())[1:]:
            if _get_submodule(parent, fqn) is None:
                _add_submodule(parent, fqn, torch.nn.Module())

    children = []
    for name, child in list(parent._modules.items()):
        if child is None:
            continue
        fqn = prefix + name
        _reorder_submodules(child, fqn_order, prefix=fqn.split("@")[0] + ".")
        delattr(parent, name)
        base_fqn = fqn.split("@")[0]
        children.append(
            (fqn_order.get(fqn, fqn_order.get(base_fqn, len(fqn_order))), name, child)
        )
    children.sort(key=operator.itemgetter(0))
    for _, name, child in children:
        parent.register_module(name, child)


class _IVals:
    """
    Collect the intermediate values of mutations in a graph.

    Example: in the following graph, suppose that buf_in and buf_out
    are the input and output values of a buffer.

        buf_in = placeholder()
        ...
        ival1 = f0(buf_in, ...)  # inside self.n0(...)
        ...
        ival2 = f1(ival1, ...)  # inside self.n1(...)
        ...
        buf_out = f2(ival2, ...)  # inside self.n2(...)
        return buf_out, ...

    Here ival1 and ival2 are intermediate values created inside
    calls to n0 and n1 respectively, and used inside calls to
    n1 and n2 respectively.
    """

    def __init__(self):
        # for each fqn, set of node names corresponding to intermediate values
        self.node_names_by_fqn = defaultdict(set)

    def _is_mutable(self, target):
        if isinstance(target, torch._ops.OpOverload):
            return target._schema.is_mutable
        return False

    def read(self, mf, node):
        """
        Read state corresponding to a given intermediate value.
        """
        # we can assume that the node must be from a mutation
        if node.op != "call_function":
            raise AssertionError(
                f"expected node.op to be 'call_function', got {node.op!r}"
            )
        b = self._is_mutable(node.target)
        print("Checking mutability", node.target, b)
        if not b:
            # so the mutation was functionalized;
            # we will apply the original mutation later (see below)
            fqn, _ = next(reversed(node.meta["nn_module_stack"].values()))
            self.node_names_by_fqn[fqn].add(node.name)
        return mf.remap_input(node.args[0])

    def update(self, partitions):
        """
        Update states corresponding to intermediate values that were read.
        """
        for shared_submodules in partitions:
            for entry in shared_submodules:
                graph = entry.module.graph
                node_names = self.node_names_by_fqn[entry.fqn]
                nodes = [n for n in graph.nodes if n.name in node_names]
                for node in nodes:
                    # so node must be from a functionalized mutation;
                    # we perform the original mutation now
                    with graph.inserting_after(node):
                        new_node = graph.create_node(
                            "call_function",
                            torch.ops.aten.copy_.default,
                            (node.args[0], node),
                        )
                        new_node.meta = copy.copy(node.meta)


def _copy_graph_attrs(
    gm: torch.fx.GraphModule,
    root_module: UnflattenedModule,
    seen_attrs: dict[str, set[str]],
):
    for child_fqn, names in seen_attrs.items():
        module = _get_attr(root_module, child_fqn) if child_fqn else root_module
        for name in names:
            val = getattr(gm, name)
            setattr(module, name, val)


def _deduplicate_modules(partitions):
    redirected_call_indices = {}
    for shared_submodules in partitions:
        for i, entry in enumerate(shared_submodules):
            child_fqn = _call_name(entry.fqn, entry.call_idx)
            target = _compute_accessor(entry.parent_fqn, child_fqn)
            deduplicated = False
            # Iterate over all previously seen modules, and deduplicate if possible
            for seen in shared_submodules[:i]:
                if _check_graph_equivalence(seen.module, entry.module):
                    parent = entry.parent_module
                    # Since graphs are equivalent, we can deduplicate.
                    # There are two cases.
                    if seen.fqn == entry.fqn:
                        # Case 1: The current module has the same fqn as the seen module.
                        # In this case we have generated a call name that can be optimized away.
                        # So we remove the current module from the hierarchy and replace
                        # the current call name with the seen call name in the parent graph.
                        *prefix, name = target.split(".")
                        _get_attr_via_attr_list(parent, prefix)._modules.pop(name)
                        seen_child_fqn = _call_name(seen.fqn, seen.call_idx)
                        seen_target = _compute_accessor(
                            entry.parent_fqn, seen_child_fqn
                        )
                        entry.parent_call_module.target = seen_target
                        redirected_call_indices[child_fqn] = seen_child_fqn
                        break
                    elif not deduplicated:
                        # Case 2: The current module has a different fqn than the seen module.
                        # In this case we replace the current module with the seen module.
                        # There should be nothing pointing to the current module any more,
                        # so it can be garbage collected.
                        # NOTE: We *do not* replace the current call name with the seen call name
                        # in the parent graph, because this will lose information on which fqn
                        # was actually called. However, it is possible that the current call name
                        # will be optimized away when we find another seen module with the same fqn,
                        # so we do not break out of the loop yet.
                        parent.set_submodule(target, seen.module)
                        deduplicated = True

    return redirected_call_indices


def _sink_params(
    module: torch.nn.Module,
    inputs_to_state: dict[str, list[str]],
    scope: list[str],
    module_id_to_inputs_removed: dict[int, set[str]] | None = None,
):
    """Sink params, buffers, and constants from graph inputs into get_attr nodes.

    Exported modules are purely functional, so they pass their parameters and
    buffers in as inputs to the graph.

    To replicate eager's semantics, we need to get them from the module state
    via get_attr instead.

    module: GraphModule, potentially containing nested submodules.
    inputs_to_state: mapping graph input names to the corresponding key in the state_dict.
    scope: tracks where we are in the module hierarchy, so that we can emit the
        right `getattr(self, "foo.bar")` calls, etc.
    module_id_to_inputs_removed: records inputs removed by child modules, mapping
        the module object id to the list of placeholder node names in the child module
        that were removed.
    """
    if module_id_to_inputs_removed is None:
        module_id_to_inputs_removed = defaultdict(set)

    if id(module) in module_id_to_inputs_removed:
        return {id(module): module_id_to_inputs_removed[id(module)]}

    # We need to use _modules here instead of named_children(), because we
    # explicitly want duplicate modules to show up in the traversal.
    for name, submodule in module._modules.items():
        submod_id_to_inputs_removed = _sink_params(
            cast("torch.nn.Module", submodule),
            inputs_to_state,
            scope + [name],
            module_id_to_inputs_removed,
        )
        for k, v in submod_id_to_inputs_removed.items():
            module_id_to_inputs_removed[k].update(v)

    graph = getattr(module, "graph", None)
    if graph is None or len(graph.nodes) == 0:
        # Not all modules have graphs defined, if they are empty modules with no operations (like ParameterList)
        return module_id_to_inputs_removed

    if not isinstance(graph, torch.fx.Graph):
        raise AssertionError(f"expected graph to be torch.fx.Graph, got {type(graph)}")

    inputs = list(filter(lambda n: n.op == "placeholder", graph.nodes))
    the_last_input = None if len(inputs) == 0 else inputs[-1]

    # Also remove from call_module nodes
    call_module_nodes = filter(lambda n: n.op == "call_module", graph.nodes)
    for node in call_module_nodes:
        submodule = _get_attr(module, node.target)
        # remove placeholder from call_module node arguments, only if we've
        # erased the placeholder node in the corresponding _sink_params() call
        if submodule is not None and id(submodule) in module_id_to_inputs_removed:
            node.args = tuple(
                filter(
                    lambda n: n.name not in module_id_to_inputs_removed[id(submodule)],
                    node.args,
                )
            )

    # Filter out inputs_to_state corresponding to current scope.
    inputs_to_state_of_scope: dict[torch.fx.Node, list[str]] = {}
    for node in inputs:
        if node.name not in inputs_to_state:
            continue

        state_name = None
        for sn in inputs_to_state[node.name]:
            sn_split = sn.split(".")
            if sn_split[: len(scope)] == [x.split("@")[0] for x in scope]:
                state_name = sn_split
                break

        # If there's a mismatch between scope name and state name, then
        # there must be multiple scopes pointing to the same state name,
        # meaning some modules are shared. In such case, we can simply skip
        # updating the current node because another later iteration will
        # take care of this input node when the unique match between scope
        # and state name occurs.  To make sure this always happen, we should
        # enforce the invariant that no placeholder node in the unflattened
        # graph appears in inputs_to_state dict, which means all the extra
        # input nodes have been handled.
        if state_name is None:
            continue

        inputs_to_state_of_scope[node] = state_name

    # Record name of removed inputs for return purpose.
    inputs_removed: set[str] = set()

    for node, state_name in inputs_to_state_of_scope.items():
        if len(node.users) > 0:
            attr_path = state_name[len(scope) :]
            state_attr = _get_attr_via_attr_list(module, attr_path)
            if not isinstance(state_attr, (torch.Tensor, torch.ScriptObject)):
                raise AssertionError(
                    f"expected state_attr to be torch.Tensor or torch.ScriptObject, got {type(state_attr)}"
                )

            # Make sure the newly created get_attr node is placed after the last placeholder node
            with graph.inserting_after(the_last_input):
                new_node = graph.create_node("get_attr", ".".join(attr_path))

            node.replace_all_uses_with(new_node, propagate_meta=True)

        graph.erase_node(node)
        inputs_removed.add(node.name)

    if isinstance(module, InterpreterModule):
        module.finalize()

    return {id(module): inputs_removed}
<|endoftext|>import getpass
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
import torch._logging._internal
import torch.utils._pytree as pytree
from torch._dynamo.exc import UserError, UserErrorType
from torch._export.passes.insert_custom_op_guards import (
    get_op_profiles,
    insert_custom_op_guards,
    OpProfile,
)
from torch._utils_internal import log_draft_export_usage

from ._trace import _export, get_ep_stats
from .dynamic_shapes import _DimHint, _DimHintType, Dim
from .exported_program import ExportedProgram


log = logging.getLogger(__name__)


_DRAFT_EXPORT_WARNING_COLOR = "\033[31m"
_DRAFT_EXPORT_SUCCESS_COLOR = "\033[32m"
_DRAFT_EXPORT_END_COLOR = "\033[0m"


def _format_colored(text: str, color: str) -> str:
    return f"{color}{text}{_DRAFT_EXPORT_END_COLOR}"


class FailureType(IntEnum):
    MISSING_FAKE_KERNEL = 1
    DATA_DEPENDENT_ERROR = 2
    GUARD_ADDED = 3
    MISMATCHED_FAKE_KERNEL = 4

    def __str__(self) -> str:
        return self.name


def prettify_stack(stack: list[dict[str, str]], str_to_filename: dict[int, str]) -> str:
    res = ""
    for frame in stack:
        if frame["filename"] not in str_to_filename:
            continue

        res += f"""
        File {str_to_filename[frame["filename"]]}, lineno {frame["line"]}, in {frame["name"]}"""  # type: ignore[index]

    res += f"\n            {stack[-1]['loc']}"
    return res


def prettify_frame_locals(
    loc: str, locals: dict[str, Any], symbols: dict[str, Any]
) -> str:
    local_str = "\n".join(f"            {k}: {v}" for k, v in locals.items())
    res = f"""
        Locals:
{local_str}
"""
    if any(v is not None for v in symbols.values()):
        symbol_str = "\n".join(
            f"           {k}: {v}" for k, v in symbols.items() if v is not None
        )
        res += f"""
        Symbols:
{symbol_str}
"""
    return res


def get_loc(filename: str, lineno: int) -> str | None:
    try:
        with open(filename) as f:
            for i, line in enumerate(f):
                if i == lineno - 1:
                    return line.strip()
    except FileNotFoundError:
        pass
    return None


class FailureReport:
    def __init__(
        self, failure_type: FailureType, data: dict[str, Any], xfail: bool = False
    ) -> None:
        self.failure_type: FailureType = failure_type
        self.data: dict[str, Any] = data
        self.xfail: bool = xfail

    def __repr__(self) -> str:
        return f"FailureReport(failure_type={self.failure_type}, xfail={self.xfail}, data={self.data})"

    def print(self, str_to_filename: dict[int, str]) -> str:
        if self.failure_type == FailureType.MISSING_FAKE_KERNEL:
            op = self.data["op"]

            return f"""Missing fake kernel.
    torch.ops.{op} is missing a fake kernel implementation.

    Please refer to https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz for more detailed instructions on how to write a meta implementation.
"""

        elif self.failure_type == FailureType.GUARD_ADDED:
            locals_info = (
                prettify_frame_locals(**self.data["frame_locals"])
                if self.data["frame_locals"]
                else ""
            )
            return f"""Guard Added.
    A guard was added during tracing, which might've resulted in some incorrect
    tracing or constraint violation error.
    Specifically, this guard was added: {self.data["expr"]}, where {self.data["symbol_to_sources"]}.
    This occurred at the following stacktrace: {prettify_stack(self.data["user_stack"], str_to_filename)}:
        {locals_info}
    And the following framework stacktrace: {prettify_stack(self.data["stack"], str_to_filename)}\n
    Because of this, we have modified the dynamic shapes structure to be the
    following. You can also use torch.export.Dim.AUTO instead to specify your
    dynamic shapes, and we will automatically infer the dynamism for you.
    ```
    dynamic_shapes = {self.data["new_dynamic_shapes"]}
    ```
"""

        elif self.failure_type == FailureType.DATA_DEPENDENT_ERROR:
            locals_info = (
                prettify_frame_locals(**self.data["frame_locals"])
                if self.data["frame_locals"]
                else ""
            )
            return f"""Data dependent error.
    When exporting, we were unable to evaluate the value of `{self.data["expr"]}`.
    This was encountered {self.data["occurrences"]} times.
    This occurred at the following user stacktrace: {prettify_stack(self.data["user_stack"], str_to_filename)}
        {locals_info}
    And the following framework stacktrace: {prettify_stack(self.data["stack"], str_to_filename)}\n
    As a result, it was specialized to a constant (e.g. `{self.data["result"]}` in the 1st occurrence), and asserts were inserted into the graph.

    Please add `torch._check(...)` to the original code to assert this data-dependent assumption.
    Please refer to https://docs.google.com/document/d/1kZ_BbB3JnoLbUZleDT6635dHs88ZVYId8jT-yTFgf3A/edit#heading=h.boi2xurpqa0o for more details.
"""

        elif self.failure_type == FailureType.MISMATCHED_FAKE_KERNEL:
            op = self.data["op"]
            reason = self.data["reason"]
            return f"""Mismatched fake kernel.
    torch.ops.{op} has a fake kernel implementation, but it has incorrect behavior, based on the real kernel.
    The reason for the mismatch is: {reason}.

    Please refer to https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ahugy69p2jmz for more detailed instructions on how to write a fake implementation.
"""

        else:
            raise ValueError(f"Unknown failure type: {self.failure_type}")


class DraftExportReport:
    def __init__(
        self,
        failures: list[FailureReport],
        str_to_filename: dict[int, str],
        expressions_created: dict[int, dict[str, Any]],
        op_profiles: dict[str, set[OpProfile]],
    ):
        self.failures: list[FailureReport] = failures
        self.str_to_filename = str_to_filename
        self.expressions_created: dict[int, dict[str, Any]] = expressions_created
        self.op_profiles = op_profiles

    def successful(self) -> bool:
        return len(self.failures) == 0 or all(
            failure.xfail for failure in self.failures
        )

    def __repr__(self) -> str:
        return f"DraftExportReport({self.failures})"

    def __str__(self) -> str:
        if self.successful():
            return _format_colored(
                """
##############################################################################################
Congratulations: No issues are found during export, and it was able to soundly produce a graph.
You can now change back to torch.export.export()
##############################################################################################
""",
                _DRAFT_EXPORT_SUCCESS_COLOR,
            )

        error = _format_colored(
            f"""
###################################################################################################
WARNING: {len(self.failures)} issue(s) found during export, and it was not able to soundly produce a graph.
Please follow the instructions to fix the errors.
###################################################################################################

""",
            _DRAFT_EXPORT_WARNING_COLOR,
        )
        error += "\n"

        for i, failure in enumerate(self.failures):
            error += f"{i + 1}. {failure.print(self.str_to_filename)}\n"
        return error

    def apply_suggested_fixes(self) -> None:
        raise NotImplementedError("Not implemented yet")


@dataclass
class ExpressionCreatedNode:
    result_id: int
    argument_ids: list[int]
    record: dict[str, object]
    visited: bool = False


class LogRecord:
    def __init__(self) -> None:
        self.log_count: dict[int, int] = {}
        self.logs: list[tuple[str, dict[str, Any]]] = []

    def _hash(self, element: tuple[str, dict[str, Any]]) -> int:
        key, data = element

        if key == "missing_fake_kernel":
            return hash((key, data["op"]))
        elif key == "mismatched_fake_kernel":
            return hash((key, data["op"], data["reason"]))
        elif key == "propagate_real_tensors_provenance":
            return hash((key, json.dumps(data["user_stack"])))
        elif key == "guard_added":
            return hash((key, json.dumps(data["user_stack"])))
        elif key == "create_unbacked_symbol":
            return hash((key, json.dumps(data["user_stack"])))

        return hash((key, json.dumps(data)))

    def try_add(self, element: tuple[str, dict[str, str]]) -> bool:
        hash_value = self._hash(element)
        if hash_value in self.log_count:
            self.log_count[hash_value] += 1
            return False

        self.log_count[hash_value] = 1
        self.logs.append(element)
        return True

    def get_log_count(self, element: tuple[str, dict[str, Any]]) -> int:
        return self.log_count[self._hash(element)]


class CaptureStructuredTrace(torch._logging._internal.LazyTraceHandler):
    def __init__(self) -> None:
        self.specific_log_keys = [
            "str",
            "exported_program",
            "propagate_real_tensors_provenance",
            "guard_added",
            "missing_fake_kernel",
            "mismatched_fake_kernel",
            "expression_created",
            "create_unbacked_symbol",
        ]
        self.log_record: LogRecord = LogRecord()
        self.expression_created_logs: dict[int, ExpressionCreatedNode] = {}
        self.symbol_to_expressions: dict[str, list[dict[str, Any]]] = {}
        self.logger = logging.getLogger("torch.__trace")
        self.prev_get_dtrace = False

        if root_dir := os.environ.get(torch._logging._internal.DTRACE_ENV_VAR):
            super().__init__(root_dir)
        else:
            sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
            root_dir = os.path.join(
                tempfile.gettempdir(),
                "export_" + sanitized_username,
            )
            super().__init__(root_dir)

        self.setFormatter(torch._logging._internal.TorchLogsFormatter(trace=True))

    def __enter__(self) -> "CaptureStructuredTrace":
        self.log_record = LogRecord()
        self.expression_created_logs = {}

        # Remove the lazy trace handler if it exists
        possible_lazy_trace_handlers = [
            handler
            for handler in self.logger.handlers
            if isinstance(handler, torch._logging._internal.LazyTraceHandler)
        ]
        for handler in possible_lazy_trace_handlers:
            self.logger.removeHandler(handler)

        self.logger.addHandler(self)
        self.prev_get_dtrace = torch._logging._internal.GET_DTRACE_STRUCTURED
        # pyrefly: ignore [bad-assignment]
        torch._logging._internal.GET_DTRACE_STRUCTURED = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self.log_record = LogRecord()
        self.expression_created_logs = {}
        self.logger.removeHandler(self)
        # pyrefly: ignore [bad-assignment]
        torch._logging._internal.GET_DTRACE_STRUCTURED = self.prev_get_dtrace
        self.prev_get_dtrace = False

    def emit(self, record: Any) -> None:
        def _log_expression_created(
            emit_func: Callable[[Any], None], sym_node_id: int
        ) -> None:
            # Log all the relevant expression_created logs
            if sym_node_id is None:
                return
            if res := self.expression_created_logs.get(sym_node_id, None):
                # Don't log the expression if we have already
                # printed it beforehand
                if not res.visited:
                    res.visited = True
                    for arg in res.argument_ids:
                        _log_expression_created(emit_func, arg)

                emit_func(res.record)

        metadata = record.metadata
        for key in self.specific_log_keys:
            if key in metadata:
                if self.log_record.try_add((key, metadata[key])):
                    if key == "expression_created":
                        # We don't want to log all expression_created logs, only
                        # the ones that are relevant to the
                        # guards/propagate_real_tensor
                        self.expression_created_logs[metadata[key]["result_id"]] = (
                            ExpressionCreatedNode(
                                metadata[key]["result_id"],
                                metadata[key].get("argument_ids", []),
                                record,
                            )
                        )
                        return

                    elif key == "propagate_real_tensors_provenance":
                        _log_expression_created(
                            super().emit, metadata[key].get("expr_node_id")
                        )

                    elif key == "guard_added":
                        if len(metadata[key]["symbol_to_sources"]) == 0:
                            # We only want to include guards added that are relevant to
                            # the symbolic shapes corresponding to the inputs which were
                            # specified in the dynamic_shapes arg. These have a source.
                            return
                        elif metadata[key]["prefix"] == "runtime_assert":
                            # This should've been captured by a
                            # propagate_real_tensors log
                            return

                        _log_expression_created(
                            super().emit, metadata[key].get("expr_node_id")
                        )

                    super().emit(record)


def draft_export(
    mod: torch.nn.Module,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any] | None = None,
    *,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None,
    preserve_module_call_signature: tuple[str, ...] = (),
    strict: bool = False,
    pre_dispatch: bool = True,
    prefer_deferred_runtime_asserts_over_guards: bool = False,
) -> ExportedProgram:
    start_time = time.time()
    kwargs = kwargs or {}
    dynamic_shapes = dynamic_shapes or {}

    constraint_violation_msg = None
    capture_structured_log = CaptureStructuredTrace()

    with (
        torch._functorch.config.patch(
            fake_tensor_propagate_real_tensors=True,
            generate_fake_kernels_from_real_mismatches=True,
        ),
        capture_structured_log,
    ):
        try:
            new_shapes = None
            ep = _export(
                mod,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                strict=strict,
                pre_dispatch=pre_dispatch,
                preserve_module_call_signature=preserve_module_call_signature,
                prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
            )
        except Exception as exc:
            if (
                isinstance(exc, UserError)
                and exc.error_type == UserErrorType.CONSTRAINT_VIOLATION
            ):
                constraint_violation_msg = exc.msg

                def convert_dim_to_auto(dim: Any) -> Any:
                    if isinstance(dim, Dim):
                        return Dim.AUTO(min=dim.min, max=dim.max)
                    elif isinstance(dim, _DimHint) and dim.type == _DimHintType.DYNAMIC:
                        return Dim.AUTO(min=dim.min, max=dim.max)
                    return dim

                new_shapes = pytree.tree_map(convert_dim_to_auto, dynamic_shapes)
                ep = _export(
                    mod,
                    args,
                    kwargs,
                    dynamic_shapes=new_shapes,
                    strict=strict,
                    pre_dispatch=pre_dispatch,
                    preserve_module_call_signature=preserve_module_call_signature,
                    prefer_deferred_runtime_asserts_over_guards=prefer_deferred_runtime_asserts_over_guards,
                )
            else:
                log_draft_export_usage(
                    error=True,
                    export_time=time.time() - start_time,
                    strict=strict,
                    message=str(exc),
                    type=f"{type(exc).__name__}.{type(exc).__qualname__}",
                )
                raise exc

        torch._logging.dtrace_structured("exported_program", payload_fn=lambda: str(ep))

        str_to_filename: dict[int, str] = {}
        failures: list[FailureReport] = []
        incorrect_custom_ops: set[str] = set()
        expressions_created: dict[int, dict[str, Any]] = {}

        for log_name, log_contents in capture_structured_log.log_record.logs:
            failure_type = None

            if log_name == "str":
                str_to_filename[log_contents[1]] = log_contents[0]  # type: ignore[index]
                continue

            elif log_name == "propagate_real_tensors_provenance":
                log_contents["occurrences"] = (
                    capture_structured_log.log_record.get_log_count(
                        (log_name, log_contents)
                    )
                )

                failure_type = FailureType.DATA_DEPENDENT_ERROR

            elif log_name == "guard_added":
                if new_shapes is None:
                    continue

                failure_type = FailureType.GUARD_ADDED
                log_contents["new_dynamic_shapes"] = new_shapes
            elif log_name == "missing_fake_kernel":
                failure_type = FailureType.MISSING_FAKE_KERNEL
                incorrect_custom_ops.add(log_contents["op"])

            elif log_name == "mismatched_fake_kernel":
                failure_type = FailureType.MISMATCHED_FAKE_KERNEL
                incorrect_custom_ops.add(log_contents["op"])

            else:
                continue

            if failure_type is None:
                raise AssertionError("failure_type cannot be None at this point")
            failures.append(
                FailureReport(
                    failure_type,
                    log_contents,
                )
            )

        for k, v in capture_structured_log.expression_created_logs.items():
            if v.visited:
                expressions_created[k] = v.record

        op_profiles = get_op_profiles(ep.graph_module, incorrect_custom_ops)
        report = DraftExportReport(
            failures, str_to_filename, expressions_created, op_profiles
        )

        # Add asserts around custom ops
        insert_custom_op_guards(ep.graph_module, incorrect_custom_ops)

    ep._report = report
    if not report.successful():
        log_filename = capture_structured_log.stream.name

        warning_msg = f"""
###################################################################################################
WARNING: {len(report.failures)} issue(s) found during export, and it was not able to soundly produce a graph.
To view the report of failures in an html page, please run the command:
    `tlparse {log_filename} --export`
Or, you can view the errors in python by inspecting `print(ep._report)`.
"""

        if len(report.op_profiles) > 0:
            warning_msg += f"""
While tracing we found {len(report.op_profiles)} operator(s) which do not have a fake kernel registered.
If you intend to retrace the exported graph or run it with fake tensors, please run it under the
following context manager, which will register a fake kernel for those operators.
```
with torch._library.fake_profile.unsafe_generate_fake_kernels(ep._report.op_profiles):
    # run with fake tensors
```
"""

        warning_msg += """#################################################################################################"""

        log.warning(warning_msg)

    else:
        log.info(
            """
##############################################################################################
Congratulations: No issues are found during export, and it was able to soundly produce a graph.
You can now change back to torch.export.export()
##############################################################################################
    """
        )

    log_draft_export_usage(
        error=False,
        export_time=time.time() - start_time,
        strict=strict,
        constraint_violations=constraint_violation_msg,
        report=ep._report,
        **get_ep_stats(ep),
    )
    return ep
<|endoftext|>import contextlib
from collections.abc import Generator

import torch
from torch._decomp import global_decomposition_table
from torch._decomp.decompositions import _rnn_helper, gather_params, gru_cell, lstm_cell
from torch._higher_order_ops.while_loop import while_loop


def _clone_tensors_that_alias(tensors):
    # Flattened RNN weights can share storage. while_loop rejects aliased
    # closure inputs, and these weights are read-only inside the loop.
    cloned_tensors = []
    seen_tensors = []
    for tensor in tensors:
        if tensor is not None and any(
            torch._C._overlaps(tensor, seen_tensor) for seen_tensor in seen_tensors
        ):
            tensor = tensor.clone()
        cloned_tensors.append(tensor)
        if tensor is not None:
            seen_tensors.append(tensor)
    return tuple(cloned_tensors)


def one_layer_while_loop_lstm(inp, hidden, params, has_biases, reverse=False):
    """
    1 layer fn for while loop LSTM

    Args:
        inp: Input tensor of shape (seq_len, batch, input_size)
        hidden: Tuple of (hx, cx) hidden states
        params: List of weight and bias tensors
        has_biases: Whether biases are included
        reverse: Whether to process sequence in reverse

    Returns:
        Tuple of (output, (final_hx, final_cx))
    """
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None
    hr_weight = (
        params[4] if len(params) == 5 else params[2] if len(params) == 3 else None
    )

    hx = hidden[0].unsqueeze(0)
    cx = hidden[1].unsqueeze(0)

    precomputed_input = torch.nn.functional.linear(inp, ih_weight, ih_bias)
    precomputed_input = precomputed_input.flip(0) if reverse else precomputed_input
    hh_weight, hh_bias, hr_weight = _clone_tensors_that_alias(
        (hh_weight, hh_bias, hr_weight)
    )

    # while loop rewrite
    step_output = torch.empty(
        precomputed_input.size(0),
        *tuple(hx.shape[1:]),
        dtype=hx.dtype,
        device=hx.device,
    )

    def cond_fn(i, out, hx, cx):
        return i < precomputed_input.size(0)

    def body_fn(idx, out, hx, cx):
        # Extract the integer value from idx and constrain it for data-dependent indexing
        i = idx.item()
        torch._check_is_size(i)
        torch._check_is_size(i, max=precomputed_input.size(0) - 1)
        hx, cx = lstm_cell(
            precomputed_input[i], hx, cx, hh_weight, hh_bias, hr_weight, chunk_dim=2
        )
        out = out.clone()
        # Squeeze the first dimension before storing (lstm_cell preserves the unsqueezed dim)
        out[i] = hx.squeeze(0)
        return idx + 1, out, hx, cx

    cnt = torch.tensor(0, dtype=torch.int64)
    _, out, final_hx, final_cx = while_loop(
        cond_fn, body_fn, [cnt, step_output, hx, cx]
    )
    if reverse:
        out = out.flip(0)

    return out, (final_hx.squeeze(0), final_cx.squeeze(0))


def lstm_while_loop_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """
    LSTM implementation using while_loop for export compatibility.

    This is a drop-in replacement for the default LSTM decomposition that uses
    while_loop instead of Python loops, making it more suitable for torch.export.

    Args:
        input: Input tensor
        hx: Tuple of (h0, c0) hidden states
        params: List of weight and bias tensors
        has_biases: Whether biases are included
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        train: Training mode
        bidirectional: Whether to use bidirectional LSTM
        batch_first: Whether batch dimension is first

    Returns:
        Tuple of (output, h_n, c_n)
    """
    if len(hx) != 2:
        raise AssertionError("lstm expects two hidden states")
    params = gather_params(params, has_biases, hx[0].size(2) != hx[1].size(2))
    hidden = list(zip(hx[0], hx[1]))
    layer_fn = one_layer_while_loop_lstm
    out, final_hiddens = _rnn_helper(
        input,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        layer_fn,
    )
    final_hiddens = list(zip(*final_hiddens))
    return out, torch.stack(final_hiddens[0], 0), torch.stack(final_hiddens[1], 0)


def one_layer_while_loop_gru(inp, hidden, params, has_biases, reverse=False):
    """
    1 layer fn for while loop GRU

    Args:
        inp: Input tensor of shape (seq_len, batch, input_size)
        hidden: Hidden state tensor
        params: List of weight and bias tensors
        has_biases: Whether biases are included
        reverse: Whether to process sequence in reverse

    Returns:
        Tuple of (output, final_hidden)
    """
    ih_weight = params[0]
    hh_weight = params[1]
    ih_bias = params[2] if has_biases else None
    hh_bias = params[3] if has_biases else None

    precomputed_input = torch.nn.functional.linear(inp, ih_weight, ih_bias)
    precomputed_input = precomputed_input.flip(0) if reverse else precomputed_input
    hh_weight, hh_bias = _clone_tensors_that_alias((hh_weight, hh_bias))
    cur_hidden = hidden.unsqueeze(0)

    # while loop rewrite
    step_output = torch.empty(
        precomputed_input.size(0),
        *tuple(cur_hidden.shape[1:]),
        dtype=cur_hidden.dtype,
        device=cur_hidden.device,
    )

    def cond_fn(i, out, cur_hidden):
        return i < precomputed_input.size(0)

    def body_fn(idx, out, cur_hidden):
        # Extract the integer value from idx and constrain it for data-dependent indexing
        i = idx.item()
        torch._check_is_size(i)
        torch._check_is_size(i, max=precomputed_input.size(0) - 1)
        cur_hidden = gru_cell(
            precomputed_input[i], cur_hidden, None, None, hh_weight, hh_bias
        )
        out = out.clone()
        out[i] = cur_hidden.squeeze(0)
        return idx + 1, out, cur_hidden

    cnt = torch.tensor(0, dtype=torch.int64)
    _, out, final_hidden = while_loop(cond_fn, body_fn, [cnt, step_output, cur_hidden])
    if reverse:
        out = out.flip(0)

    return out, final_hidden.squeeze(0)


def gru_while_loop_impl(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    """
    GRU implementation using while_loop for export compatibility.

    This is a drop-in replacement for the default GRU decomposition that uses
    while_loop instead of Python loops, making it more suitable for torch.export.

    Args:
        input: Input tensor
        hx: Hidden state tensor
        params: List of weight and bias tensors
        has_biases: Whether biases are included
        num_layers: Number of GRU layers
        dropout: Dropout probability
        train: Training mode
        bidirectional: Whether to use bidirectional GRU
        batch_first: Whether batch dimension is first

    Returns:
        Tuple of (output, h_n)
    """
    params = gather_params(params, has_biases, False)
    hidden = list(hx.unbind(0))
    layer_fn = one_layer_while_loop_gru
    out, final_hiddens = _rnn_helper(
        input,
        hidden,
        params,
        has_biases,
        num_layers,
        dropout,
        train,
        bidirectional,
        batch_first,
        layer_fn,
    )
    return out, torch.stack(final_hiddens, 0)


@contextlib.contextmanager
def _register_rnn_while_loop_decomposition(
    rnn_op, rnn_impl
) -> Generator[None, None, None]:
    """
    Generic context manager for registering while_loop-based RNN decompositions.

    Args:
        rnn_op: The aten operation to patch (e.g., torch.ops.aten.lstm.input)
        rnn_impl: The while_loop-based implementation function

    Note:
        This is an internal helper. Use register_lstm_while_loop_decomposition()
        or register_gru_while_loop_decomposition() instead.
    """
    registry = global_decomposition_table["post_autograd"]

    # Save the original decomposition if it exists
    original_decomp = registry.get(rnn_op, None)

    # Save the original py_kernel if it exists
    original_py_kernel = rnn_op.py_kernels.get(
        torch._C.DispatchKey.CompositeImplicitAutograd, None
    )

    try:
        # Register our while_loop-based implementation
        registry[rnn_op] = rnn_impl
        rnn_op.py_kernels[torch._C.DispatchKey.CompositeImplicitAutograd] = rnn_impl
        yield
    finally:
        # Restore the original decomposition
        if original_decomp is not None:
            registry[rnn_op] = original_decomp
        else:
            # If there was no original, remove our registration
            registry.pop(rnn_op, None)

        # Restore the original py_kernel
        if original_py_kernel is not None:
            rnn_op.py_kernels[torch._C.DispatchKey.CompositeImplicitAutograd] = (
                original_py_kernel
            )
        else:
            # If there was no original, remove our registration
            rnn_op.py_kernels.pop(torch._C.DispatchKey.CompositeImplicitAutograd, None)


@contextlib.contextmanager
def register_lstm_while_loop_decomposition() -> Generator[None, None, None]:
    """
    Context manager that temporarily registers the while_loop-based LSTM decomposition.

    The while_loop-based decomposition is more suitable for export and graph-based
    execution, as it avoids Python control flow that cannot be captured in the graph.
    This should support dynamic sequence lengths, however as while_loop does not
    support Autograd yet, an ExportedProgram created with this will not be trainable.

    Usage::

        from torch.export._patches import register_lstm_while_loop_decomposition
        from torch.export import export

        with register_lstm_while_loop_decomposition():
            # Export your model with LSTM
            ep = export(model, (x, h0, c0))

    Note:
        This context manager temporarily modifies the global decomposition table
        and py_kernels registration. The original registrations are restored when
        exiting the context.
    """
    with _register_rnn_while_loop_decomposition(
        torch.ops.aten.lstm.input, lstm_while_loop_impl
    ):
        yield


@contextlib.contextmanager
def register_gru_while_loop_decomposition() -> Generator[None, None, None]:
    """
    Context manager that temporarily registers the while_loop-based GRU decomposition.

    The while_loop-based decomposition is more suitable for export and graph-based
    execution, as it avoids Python control flow that cannot be captured in the graph.
    This should support dynamic sequence lengths, however as while_loop does not
    support Autograd yet, an ExportedProgram created with this will not be trainable.

    Usage::

        from torch.export._patches import register_gru_while_loop_decomposition
        from torch.export import export

        with register_gru_while_loop_decomposition():
            # Export your model with GRU
            ep = export(model, (x, h0))

    Note:
        This context manager temporarily modifies the global decomposition table
        and py_kernels registration. The original registrations are restored when
        exiting the context.
    """
    with _register_rnn_while_loop_decomposition(
        torch.ops.aten.gru.input, gru_while_loop_impl
    ):
        yield
<|endoftext|># mypy: allow-untyped-defs
import operator

import torch
from torch._higher_order_ops.effects import _get_schema, with_effects

from .exported_program import ExportedProgram
from .graph_signature import (
    CustomObjArgument,
    InputKind,
    InputSpec,
    OutputKind,
    OutputSpec,
    TokenArgument,
)


def _get_custom_obj_for_node(node, inputs_to_lifted_custom_objs, constants):
    """Extract the custom object from a node's arguments."""
    custom_obj_node = node
    custom_obj_meta = custom_obj_node.meta["val"]  # type: ignore[union-attr]
    if not isinstance(custom_obj_meta, CustomObjArgument):
        raise AssertionError(
            f"Expected custom_obj_meta to be a CustomObjArgument, but got {type(custom_obj_meta)}"
        )

    if custom_obj_meta.fake_val:
        return custom_obj_meta.fake_val
    elif custom_obj_node.name in inputs_to_lifted_custom_objs:  # type: ignore[union-attr]
        return constants[inputs_to_lifted_custom_objs[custom_obj_node.name]]  # type: ignore[union-attr]
    else:
        raise RuntimeError(f"Unable to find custom obj for node {node}")


def _replace_with_effects_node(
    node, ep, inputs_to_lifted_custom_objs, output_tokens, input_tokens, module
):
    """Replace a with_effects node with the underlying function call."""
    # Get the input nodes
    token_node, func, *node_args = node.args
    if token_node.op == "placeholder":
        input_tokens.append(token_node)

    if not isinstance(func, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        raise AssertionError(
            f"Expected func to be an OpOverload or HigherOrderOperator, but got {type(func)}"
        )

    # Get the schema for the function
    if func is torch.ops.higher_order.call_torchbind:
        custom_obj = _get_custom_obj_for_node(
            node_args[0], inputs_to_lifted_custom_objs, ep.constants
        )
        schema = _get_schema(func, [custom_obj] + node_args[1:])
    else:
