        # nonsensical. This does not affect the collection of metadata.
        with patch("torch.cuda.set_rng_state", lambda *args: None):
            mod = root_module_when_exporting_non_strict(flat_fn)
            if mod is not None:
                ctx = _detect_attribute_assignment(mod)
            else:
                ctx = nullcontext()

            if torch._functorch.config.fake_tensor_propagate_real_tensors:
                # Running dynamo_timed causes fake tensor issues when
                # propagate real tensor is switched on.
                dynamo_timed_ctx = nullcontext()
            else:
                dynamo_timed_ctx = dynamo_timed(
                    "aot_collect_metadata", log_pt2_compile_event=True
                )

            with (
                dynamo_timed_ctx,
                ctx,
                torch._dynamo.eval_frame._use_eager_on_nested_compile(),
            ):
                fw_metadata = run_functionalized_fw_and_collect_metadata(
                    flat_fn,
                    flat_args_descs=flat_args_descs,
                    static_input_indices=aot_config.static_input_indices,
                    keep_input_mutations=aot_config.keep_inference_input_mutations,
                    pre_dispatch=aot_config.pre_dispatch,
                )(*_dup_fake_script_obj(fake_flat_args))

            req_subclass_dispatch = requires_subclass_dispatch(
                fake_flat_args, fw_metadata
            )
            CompileEventLogger.try_add_pt2_compile(
                "backend_compile", requires_subclass_dispatch=req_subclass_dispatch
            )

            output_and_mutation_safe = not any(
                x.requires_grad
                # view-type operations preserve requires_grad even in no_grad.
                # Do not count aliases of inputs with requires_grad as reason to make a training graph,
                # as AOTAutograd will perform view-replay to regenerate the view outputs at runtime,
                # setting their grad_fn properly.
                and not (
                    x.output_type in (OutputType.alias_of_input, OutputType.is_input)
                    and x.base_idx is not None
                    and fw_metadata.input_info[x.base_idx].requires_grad
                )
                for x in fw_metadata.output_info
            ) and not any(
                x.requires_grad
                and x.mutates_data
                and not x.mutations_under_no_grad_or_inference_mode
                and not x.mutations_hidden_from_autograd
                for x in fw_metadata.input_info
            )

            if needs_autograd and output_and_mutation_safe:
                # We realized that none of the outputs require grad,
                # and none of the inputs that require grad are mutated.
                # so we actually have an inference graph.
                needs_autograd = False

    if fw_metadata.num_intermediate_bases > 0:
        if req_subclass_dispatch:
            raise AssertionError(f"""\
torch.compile is currently being used with tensor subclass inputs.
We are attempting to a compile a graph with two graph outputs
that alias one another, specifically output indices:

    {[i for i, x in enumerate(fw_metadata.output_info) if x.output_type == OutputType.alias_of_intermediate]}

ANY output aliasing (even for regular tensors) is currently unsupported if
there are any subclass outputs. If you run into this, please file a github
issue""")

    if aot_config.is_export:
        # aot_export: ban input metadata mutations for now to keep shared code paths simpler.
        # Keeping .resize_() in the graph will require some work
        # Allowing it but keeping the graph functional will require some calling convention changes.
        if len([x for x in fw_metadata.input_info if x.mutates_metadata]) != 0:
            raise RuntimeError(
                f"""\
Found an input that received a metadata mutation, through e.g. a call to `.resize_()` or `.transpose_()`.
This is currently banned in the aot_export workflow. If you need this functionality, please file a github issue.

fw_metadata={str(fw_metadata)}"""
            )
        # In export, banning data mutations on inputs that require grad for now.
        # This should be rare, and is tricky to get right. When we trace the backward,
        # we currently trace with autograd.grad instead of .backward(), which makes it difficult
        # to ensure that we run autograd all the way through the input **before** it saw the mutation.
        if (
            len(
                [
                    x
                    for x in fw_metadata.input_info
                    if x.requires_grad and x.mutates_data
                ]
            )
            != 0
            and aot_config.export_trace_joint
        ):
            raise RuntimeError(
                f"""\
Found a graph input that requires gradients, and received a mutation.
This is currently banned in the aot_export workflow. If you need this functionality, please file a github issue.

fw_metadata={str(fw_metadata)}"""
            )
        if req_subclass_dispatch:
            raise RuntimeError(
                """\
aot_export is not currently supported with traceable tensor subclass.
If you need this feature, please comment on <CREATE_ISSUE_LINK>"""
            )

        # Need to decide on a strategy for functionalized RNG: toggling via global config seems bad,
        # and turning it on will require a non-trivial calling convention change for any export runtime.
        if config.functionalize_rng_ops:
            raise RuntimeError(
                """\
Functionalized RNG is not currently supported in the aot_export workflow. Please file a github issue,
or otherwise set torch._functorch.config.functionalize_rng_ops = False."""
            )

    return AOTState(
        needs_autograd=needs_autograd,
        flat_args=_dup_fake_script_obj(fake_flat_args),
        flat_args_descs=flat_args_descs,
        fw_metadata=fw_metadata,
        # Packaging this just for later use
        aot_config=aot_config,
        stack=stack,
        fake_mode=fake_mode,
    )


def aot_function(
    fn: Callable[_P, _R],
    fw_compiler: AOTDispatchCompiler,
    bw_compiler: AOTDispatchCompiler | None = None,
    partition_fn: Callable[..., Any] = default_partition,
    decompositions: dict[OpOverload, Callable[..., Any]] | None = None,
    num_params_buffers: int = 0,
    keep_inference_input_mutations: bool = False,
    inference_compiler: AOTDispatchCompiler | None = None,
    *,
    # Whether or not to trace with dynamic shapes
    dynamic: bool = False,
    enable_log: bool = True,
    disable_functionalization: bool = False,
    _disable_torch_fn_metadata_mode: bool = False,
) -> Callable[_P, Any]:
    """
    Traces the forward and backward graph of :attr:`fn` using torch dispatch
    mechanism, and then compiles the generated forward and backward graphs
    through :attr:`fw_compiler` and :attr:`bw_compiler`.

    :func:`aot_function` traces the forward and backward graph ahead of time,
    and generates a joint forward and backward graph.  :attr:`partition_fn` is
    then used to separate out forward and backward graphs. The partitioner
    function can be used to perform optimizations such as recomputation. One can
    set `decompositions` dictionary to decompose the operators into a sequence
    of core or simpler operators supported by the backend compilers.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fn (Callable): A Python function that takes one or more arguments. Must
            return one or more Tensors.
        fw_compiler (Callable): A Python function that accepts an Fx graph with
            Aten ops and input args, and returns a Callable that semantically is
            equivalent to the input Fx graph.
        bw_compiler (Optional[Callable]): A Python function that accepts an
            Fx graph with Aten ops and input args, and returns a Callable that
            semantically is equivalent to the input Fx graph.  Default: None
            (when None, it defaults to the :attr:`fw_compiler`)
        partition_fn (Callable): A Python function that takes a joint forward
            and backward graph, and partitions it into separate forward and
            backward graphs.
        decompositions (Dict): A dictionary to define the decomposition of
            larger Aten ops into simpler or core Aten ops.
        inference_compiler (Optional[Callable]): A Python function that accepts an
            Fx graph with Aten ops and input args, and returns a Callable that
            semantically is equivalent to the input Fx graph. inference_compiler is invoked
            if no autograd is needed. Default: None
            (when None, it defaults to the :attr:`fw_compiler`)
    Returns:
        Returns a ``Callable`` that retains the eager behavior of the original
        :attr:`fn`, but with forward and backward graph compiled via
        :attr:`fw_compile` and :attr:`bw_compile`.

    A simple example usage of :func:`aot_function` is as follows. This example
    will print the forward and backward graphs of the function ``fn``

        >>> fn = lambda x: x.sin().cos()
        >>> def print_compile_fn(fx_module, args):
        >>>     print(fx_module)
        >>>     return fx_module
        >>> aot_fn = aot_function(fn, print_compile_fn)
        >>> x = torch.randn(4, 5, requires_grad=True)
        >>> aot_fn(x)
    """

    aot_config = AOTConfig(
        fw_compiler=None,
        bw_compiler=None,
        inference_compiler=None,
        partition_fn=None,
        decompositions=decompositions,
        num_params_buffers=num_params_buffers,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=keep_inference_input_mutations,
        dynamic_shapes=dynamic,
        aot_autograd_arg_pos_to_source=None,
        is_export=False,
        no_tangents=False,
        enable_log=enable_log,
        disable_functionalization=disable_functionalization,
        _disable_torch_fn_metadata_mode=_disable_torch_fn_metadata_mode,
    )
    cached_res = None

    @wraps(fn)
    def returned_function(*args: _P.args, **kwargs: _P.kwargs) -> Any:
        nonlocal cached_res
        # Now flatten the tensor args
        flat_args = pytree.arg_tree_leaves(*args, **kwargs)

        # Compile the function and save it in the cache
        if cached_res is None:
            flat_fn, out_spec = create_tree_flattened_fn(fn, args, kwargs)
            (fake_mode, shape_env) = construct_fake_mode(flat_args, aot_config)
            fake_flat_args: FakifiedFlatArgs
            fake_flat_args, act_input_paths = process_inputs(
                flat_args, aot_config, fake_mode, shape_env
            )
            # TODO: We actually could use the pytree path to make better descs.
            # Also, the descs here are bad if you do aot_module.
            fake_flat_args_descs: list[AOTInput] = [
                PlainAOTInput(i) for i in range(len(fake_flat_args))
            ]
            with contextlib.ExitStack() as stack:
                aot_state = create_aot_state(
                    stack,
                    flat_fn,
                    fake_flat_args,
                    fake_flat_args_descs,
                    aot_config,
                    fake_mode,
                    shape_env,
                )
                aot_state.fw_metadata.act_input_paths = act_input_paths
                aot_graph_capture = aot_stage1_graph_capture(aot_state, flat_fn)
                compiled_fn, _ = aot_stage2_compile(
                    aot_state,
                    aot_graph_capture,
                    partition_fn,
                    fw_compiler,
                    bw_compiler,
                    inference_compiler,
                )
            cached_res = (compiled_fn, out_spec)

        cached_fn, out_spec = cached_res
        out = cached_fn(flat_args)
        return out_spec.unflatten(out)

    return returned_function


def aot_module(mod: nn.Module, *args: Any, **kwargs: Any) -> nn.Module:
    """
    Traces the forward and backward graph of :attr:`mod` using torch dispatch
    tracing mechanism. It is wrapper function, that underneath uses
    :func:`aot_function` to perform tracing and compilation.

    :func:`aot_module` lifts the parameters and buffers of ``nn.Module`` as inputs
    to a new callable which is then compiled through :func:`aot_function`.

    .. warning::
        This API is experimental and likely to change.

    Args:
        mod (Callable): A ``nn.Module`` module.
        args : args to be passed to :func:`aot_function`
        kwargs : kwargs to be passed to :func:`aot_function`

    Returns:
        Returns a ``nn.Module`` that retains the eager behavior of the original
        :attr:`mod`, but with forward and backward graph compiled.

    """
    # See Note: [Fake Modules and AOTAutograd]
    torch._dynamo.utils.assert_no_fake_params_or_buffers(mod)

    def functional_call(
        named_params: dict[str, torch.nn.Parameter],
        named_buffers: dict[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        params_and_buffers = {**named_params, **named_buffers}
        return torch.func.functional_call(mod, params_and_buffers, args, kwargs)

    named_params = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))
    num_params_buffers = len(named_params) + len(named_buffers)
    compiled_f = aot_function(
        functional_call,
        *args,
        # pyrefly: ignore[bad-keyword-argument]
        num_params_buffers=num_params_buffers,
        **kwargs,
    )

    class AOTModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.orig_module = mod

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return compiled_f(
                named_params,
                named_buffers,
                *args,
                **kwargs,
            )

    return AOTModule()


def autograd_cache_key(
    graph,
    example_inputs,
    ignore_shape_env: bool,
    decompositions,
    compiler_config_extra: CompilerConfigExtra,
    keep_inference_input_mutations: bool = False,
    disable_functionalization: bool = False,
):
    (
        _params_buffers_flat,
        _params_spec,
        _buffers_spec,
        full_args,
        _full_args_descs,
        aot_config,
    ) = prepare_aot_config(
        graph,
        example_inputs,
        decompositions,
        keep_inference_input_mutations,
        ignore_shape_env,
        force_non_lazy_backward_lowering=config.force_non_lazy_backward_lowering,
        disable_functionalization=disable_functionalization,
    )

    fake_mode, shape_env = construct_fake_mode(full_args, aot_config)
    fake_flat_args, act_input_paths = process_inputs(
        full_args, aot_config, fake_mode, shape_env, ignore_shape_env
    )

    return autograd_cache.autograd_cache_key(
        graph, fake_flat_args, aot_config, compiler_config_extra, act_input_paths
    )


def prepare_aot_config(
    mod: nn.Module,
    args: Iterable[Any],
    decompositions: dict[OpOverload, Callable[..., Any]] | None,
    keep_inference_input_mutations: bool,
    ignore_shape_env: bool,
    *,
    force_non_lazy_backward_lowering: bool = False,
    disable_functionalization: bool = False,
    _disable_torch_fn_metadata_mode: bool = False,
) -> tuple[
    list[torch.nn.Parameter | Tensor],
    list[str],
    list[str],
    list[Any],
    list[Any],
    AOTConfig,
]:
    # TODO: There's something a bit suspicious here; typically simplified
    # module shouldn't actually have any parameters...
    params = dict(mod.named_parameters(remove_duplicate=False))
    buffers = dict(dict(mod.named_buffers(remove_duplicate=False)))

    params_flat, params_spec = list(params.values()), list(params.keys())
    params_len = len(params_flat)

    buffers_flat, buffers_spec = list(buffers.values()), list(buffers.keys())
    buffers_len = len(buffers_flat)

    params_buffers = {**params, **buffers}
    params_buffers_flat = params_flat + buffers_flat

    full_args = [*params_flat, *buffers_flat, *args]

    # OK, set up the descs

    full_args_descs: list[DifferentiableAOTInput] = []
    full_args_descs.extend(ParamAOTInput(fqn) for fqn in params_spec)
    full_args_descs.extend(BufferAOTInput(fqn) for fqn in buffers_spec)

    # TODO: These tracing_context fields should become unnecessary once we
    # always maintain sources on all arguments
    if tracing_context := torch._guards.TracingContext.try_get():
        # NB: TracingContext misnames this, the "params" here also contains
        # buffers
        tracing_context.params_flat = params_buffers_flat
        (
            tracing_context.params_flat_unwrap_subclasses,
            tracing_context.params_unwrapped_to_flat_index,
        ) = unwrap_tensor_subclasses_with_indices_to_original(params_buffers_flat)

    # TODO: Might be nice to hold on to the Dynamo source here in full_args_descs!
    (
        aot_autograd_arg_pos_to_source,
        static_input_indices,
    ) = _try_get_metadata_from_dynamo(
        mod, params_buffers.keys(), len(full_args), full_args_descs
    )

    dynamic_shapes = False
    for x in full_args:
        fake_mode = maybe_get_fake_mode(x)
        if fake_mode is not None:
            dynamic_shapes = fake_mode.shape_env is not None
            break

    aot_config = AOTConfig(
        fw_compiler=None,
        bw_compiler=None,
        inference_compiler=None,
        partition_fn=None,
        decompositions=decompositions,
        num_params_buffers=params_len + buffers_len,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=keep_inference_input_mutations,
        dynamic_shapes=dynamic_shapes,
        # pyrefly: ignore[bad-argument-type]
        aot_autograd_arg_pos_to_source=aot_autograd_arg_pos_to_source,
        static_input_indices=static_input_indices,
        is_export=False,
        no_tangents=False,
        cache_info=None,
        ignore_shape_env=ignore_shape_env,
        precompile_backend_id=getattr(mod, "_backend_id", None),
        force_non_lazy_backward_lowering=force_non_lazy_backward_lowering,
        disable_functionalization=disable_functionalization,
        _disable_torch_fn_metadata_mode=_disable_torch_fn_metadata_mode,
    )

    return (
        params_buffers_flat,
        params_spec,
        buffers_spec,
        full_args,
        full_args_descs,
        aot_config,
    )


def prepare_aot_module_simplified(
    mod: nn.Module,
    args: Iterable[Any],
    kwargs: dict[str, Any] | None,
    decompositions: dict[OpOverload, Callable[..., Any]] | None,
    keep_inference_input_mutations: bool,
    ignore_shape_env: bool,
    flatten: bool,
    *,
    force_non_lazy_backward_lowering: bool = False,
    disable_functionalization: bool = False,
    _record_nn_module_stack: bool = False,
    _disable_torch_fn_metadata_mode: bool = False,
) -> tuple[
    Any,
    list[torch.nn.Parameter | Tensor],
    list[str],
    list[str],
    FakifiedFlatArgs,
    list[Any],
    AOTConfig,
    FakeTensorMode,
    ShapeEnv | None,
    pytree.TreeSpec | None,
    PytreeThunk | None,
    list[ActInputPath],
]:
    if not flatten:
        if kwargs is not None:
            raise AssertionError("kwargs must be None when flatten=False")
    elif kwargs is None:
        kwargs = {}

    (
        params_buffers_flat,
        params_spec,
        buffers_spec,
        full_args,
        full_args_descs,
        aot_config,
    ) = prepare_aot_config(
        mod,
        args,
        decompositions,
        keep_inference_input_mutations,
        ignore_shape_env,
        force_non_lazy_backward_lowering=force_non_lazy_backward_lowering,
        disable_functionalization=disable_functionalization,
        _disable_torch_fn_metadata_mode=_disable_torch_fn_metadata_mode,
    )

    params_buffers_spec = params_spec + buffers_spec

    # NB: This doesn't change the in/out convention, except adding the
    # parameters as explicit arguments
    functional_call = create_functional_call(
        mod,
        params_buffers_spec,
        aot_config.num_params_buffers,
        strict_out_tuple=not flatten,
        # We need this for export to run ModuleStackTracer
        # instead of PythonKeyTracer
        store_orig_mod=_record_nn_module_stack,
    )

    in_spec, out_spec = None, None
    if flatten:
        functional_call, out_spec = create_tree_flattened_fn(
            functional_call, full_args, kwargs
        )
        full_args, in_spec = pytree.tree_flatten((full_args, kwargs))

    # TODO: it would be better to put pytree information in here
    full_args_descs.extend(
        PlainAOTInput(i) for i in range(len(full_args) - len(full_args_descs))
    )

    fake_mode, shape_env = construct_fake_mode(full_args, aot_config)
    # NB: full_args_descs not needed here, fake_flat_args is 1:1 with full_args
    fake_flat_args, act_input_paths = process_inputs(
        full_args, aot_config, fake_mode, shape_env, ignore_shape_env
    )

    return (
        functional_call,
        params_buffers_flat,
        params_spec,
        buffers_spec,
        fake_flat_args,
        full_args_descs,
        aot_config,
        fake_mode,
        shape_env,
        in_spec,
        out_spec,
        act_input_paths,
    )


def aot_module_simplified(
    mod: torch.fx.GraphModule | torch._dynamo.utils.GmWrapper,
    args: Sequence[Any],
    fw_compiler: AOTDispatchCompiler,
    bw_compiler: AOTDispatchCompiler | None = None,
    partition_fn: Callable[..., Any] = default_partition,
    decompositions: dict[OpOverload, Callable[..., Any]] | None = None,
    keep_inference_input_mutations: bool = False,
    inference_compiler: AOTDispatchCompiler | None = None,
    # TODO: This doesn't seem to be used in any nontrivial way, check if it's
    # actually needed
    compiler_config_extra: CompilerConfigExtra | None = None,
    ignore_shape_env: bool = False,
    disable_functionalization: bool = False,
    # Optional callback to run passes on the module at the start of AOT autograd.
    pre_grad_passes: Callable[
        [torch.fx.GraphModule, Sequence[InputType]], torch.fx.GraphModule
    ]
    | None = None,
    compile_region_name: str | None = None,
) -> Callable[..., Any]:
    """
    This is the simplified or low overhead version of aot_module. For frontends
    like TorchDynamo, the input functions/modules to AOT are static and have
    unpacked inputs/outputs. This gives us an opportunity to remove the
        (1) pytree overhead to parse inputs/outputs,
        (2) AOT Autograd cache,
        (3) Reading of params/buffers in every forward call

    :func:`aot_module_simplified` removes these overheads.
    """

    pre_grad_pass_timing: Literal["early", "late"] = resolve_pre_grad_pass_timing()

    if (
        pre_grad_pass_timing == "early"
        and pre_grad_passes
        and isinstance(mod, torch.fx.GraphModule)
    ):
        mod = pre_grad_passes(mod, args)

    with contextlib.ExitStack() as stack:
        (
            functional_call,
            params_buffers_flat,
            _params_spec,
            _buffers_spec,
            fake_flat_args,
            full_args_descs,
            aot_config,
            fake_mode,
            shape_env,
            _in_spec,
            _out_spec,
            act_input_paths,
        ) = prepare_aot_module_simplified(
            mod,
            args,
            None,
            decompositions,
            keep_inference_input_mutations,
            ignore_shape_env,
            flatten=False,
            force_non_lazy_backward_lowering=config.force_non_lazy_backward_lowering,
            disable_functionalization=disable_functionalization,
        )

        compiled_fn = None

        if (
            isinstance(fw_compiler, SerializableAOTDispatchCompiler)
            or torch._functorch.config.force_autograd_cache
        ):
            local = should_use_local_autograd_cache()
            remote = should_use_remote_autograd_cache()
            if local or remote:
                set_feature_use("aot_autograd_remote_cache", remote)
                compiled_fn, aot_config = AOTAutogradCache.try_load(
                    mod,
                    fake_flat_args,
                    aot_config,
                    compiler_config_extra,
                    local,
                    remote,
                    act_input_paths,
                    compile_region_name=compile_region_name,
                )

        if compiled_fn is None:
            if (
                pre_grad_pass_timing == "late"
                and pre_grad_passes
                and isinstance(mod, torch.fx.GraphModule)
            ):
                mod = pre_grad_passes(mod, args)

            stack.enter_context(compiled_autograd._disable())
            aot_state = create_aot_state(
                stack,
                functional_call,
                fake_flat_args,
                full_args_descs,
                aot_config,
                fake_mode,
                shape_env,
            )
            aot_state.fw_metadata.act_input_paths = act_input_paths
            aot_graph_capture = aot_stage1_graph_capture(aot_state, functional_call)
            compiled_fn, _ = aot_stage2_compile(
                aot_state,
                aot_graph_capture,
                partition_fn,
                fw_compiler,
                bw_compiler,
                inference_compiler,
            )
    if compiled_fn is None:
        raise AssertionError("compiled_fn must not be None")
    if isinstance(mod, torch._dynamo.utils.GmWrapper):
        # This function is called by the flatten_graph_inputs wrapper, which boxes
        # the inputs so that they can be freed before the end of this scope.
        # For overhead reasons, this is not the default wrapper, see comment:
        # https://github.com/pytorch/pytorch/pull/122535/files#r1560096481
        @simple_wraps(compiled_fn)
        def forward(runtime_args: list[Any]) -> Any:
            flat_args = []
            flat_args.extend(params_buffers_flat)
            flat_args.extend(runtime_args)
            runtime_args.clear()
            if compiled_fn is None:
                raise AssertionError("compiled_fn must not be None")
            return compiled_fn(flat_args)

    else:
        # TODO: There is something deeply wrong here; compiled_fn running with
        # the boxed calling convention, but aot_module_simplified somehow
        # historically returned a function that was not the boxed calling
        # convention.  This should get fixed...
        # NB: GraphModule/nn.Module rely on the non-boxed calling convention here
        @simple_wraps(compiled_fn)
        def forward(*runtime_args: tuple[Any]) -> Any:
            full_args = []
            full_args.extend(params_buffers_flat)
            # pyrefly: ignore[bad-argument-type]
            full_args.extend(runtime_args)
            if compiled_fn is None:
                raise AssertionError("compiled_fn must not be None")
            return compiled_fn(full_args)

    # Just for convenience
    forward.zero_grad = mod.zero_grad  # type: ignore[attr-defined]
    forward.named_parameters = mod.named_parameters  # type: ignore[attr-defined]
    forward.named_buffers = mod.named_buffers  # type: ignore[attr-defined]

    # Add a serialize function
    def grab_serialize_fn(fn: Any) -> Callable[..., Any] | None:
        if isinstance(fn, SerializableCompiledFunction):
            return fn.serialize_fn
        elif hasattr(fn, "__wrapped__"):
            return grab_serialize_fn(fn.__wrapped__)
        else:
            return None

    forward.serialize = grab_serialize_fn(forward)  # type: ignore[attr-defined]
    return forward


def boxed_nop_preserve_node_meta(
    gm: torch.fx.GraphModule, example_inputs: Sequence[InputType]
) -> Any:
    def run(args: list[Any]) -> OutputCode:
        with torch.fx.traceback.preserve_node_meta():
            return torch.fx.Interpreter(gm).boxed_run(args)

    run._boxed_call = True  # type: ignore[attr-defined]
    return run


def aot_export_joint_with_descriptors(
    stack: contextlib.ExitStack,
    mod: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    decompositions: dict[OpOverload, Callable[..., Any]] | None = None,
    keep_inference_input_mutations: bool = False,
    ignore_shape_env: bool = False,
    disable_functionalization: bool = False,
    _record_nn_module_stack: bool = False,
    _disable_torch_fn_metadata_mode: bool = False,
) -> JointWithDescriptors:
    """
    This API captures the joint graph for an nn.Module.  However, unlike
    aot_export_joint_simple or aot_export_module(trace_joint=True), the
    calling convention of the produced joint graph follows no fixed positional
    schema; for example, you cannot rely on the second argument of the traced
    joint graph to correspond to the second argument of the module you traced.
    However, the inputs and outputs of the traced graph are schematized
    with **descriptors**, annotated on meta['desc'] on the placeholder and
    return FX nodes, which you can use to determine the meaning of arguments.

    The major benefit of using this export rather than aot_export_joint_simple
    is that we have feature parity with all situations that torch.compile
    supports (via aot_module_simplified), including handling for more
    complicated cases such as multiple differentiable outputs, input mutations
    that must be handled outside of the graph, tensor subclasses, etc.

    What can you do with one of these joint graphs with descriptors?  The
    motivating use case (autoparallel) involves taking the joint graph, doing
    optimizations on it, and then turning it back into a callable so it can be
    torch.compile'd at a later point in time.  This cannot be done as a
    traditional torch.compile joint graph pass for two reasons:

        1. The sharding of parameters must be decided before parameter
           initialization / checkpoint load, far before torch.compile would
           ordinarily run.

        2. We need to change the meaning of parameters (e.g., we might replace
           a replicated parameter with a sharded version of it, changing its
           input size).  torch.compile is ordinarily semantics preserving, and
           not allowed to change the meaning of inputs.

    Some descriptors can be quite exotic, so we recommend thinking carefully
    if there is a safe fallback you can apply to descriptors you don't understand.
    For example, you should have some way to handle not finding a particular
    input exactly as is in the final FX graph inputs.

    Note: When using this API, you must create and enter an ExitStack context
    manager, which will be passed into this function.  This context manager
    must remain active if you call the compile function to finish compilation.
    (TODO: We may relax this requirement by having AOTAutograd keep track of
    how to reconstruct all the context managers at a later point in time.)

    NB: You're not obligated to do a /full/ compile in stage2; instead you can
    leave the forward/backward compilers unspecified in which case the
    partitioned FX graphs will directly run.  The overall autograd Function
    can be allowed in graph so you can reprocess it in the context of a
    (potentially larger) compiled region later.

    NB: These APIs do NOT hit cache, as we only ever cache the final compile results,
    not the intermediate export result.

    NB: If the passed nn.Module has parameters and buffers on it, we will
    generate extra implicit parameter/buffer arguments and assign ParamAOTInput
    and BufferAOTInput descriptors to them.  However, if you generate the input
    nn.Module from a mechanism like Dynamo, you will NOT get these descriptors
    (because Dynamo will already have taken care of lifting the parameters/buffers
    into arguments!)  In that case, it would be necessary to analyze the Sources
    of the inputs to determine if inputs are parameters and their FQNs.
    """

    (
        functional_call,
        _params_buffers_flat,
        params_spec,
        buffers_spec,
        fake_flat_args,
        full_args_descs,
        aot_config,
        fake_mode,
        shape_env,
        in_spec,
        out_spec,
        act_input_paths,
    ) = prepare_aot_module_simplified(
        mod,
        args,
        kwargs,
        # In contrast, decompositions are needed at this stage.
        decompositions,
        keep_inference_input_mutations,
        ignore_shape_env,
        flatten=True,
        # Without this, we will attempt to "compile" the backward lazily
        # at runtime, but this is pointless because it's just boxed_nop,
        # it's trivial.  But this will get Inductor confused about scoping
        # Metric(s) {'is_forward'} have already been set in the current
        # context.
        force_non_lazy_backward_lowering=True,
        disable_functionalization=disable_functionalization,
        _record_nn_module_stack=_record_nn_module_stack,
        _disable_torch_fn_metadata_mode=_disable_torch_fn_metadata_mode,
    )

    # TODO: Maybe this should be in create_aot_state?  Not sure, that would
    # increase its scope
    stack.enter_context(compiled_autograd._disable())

    aot_state = create_aot_state(
        stack,
        functional_call,
        fake_flat_args,
        full_args_descs,
        aot_config,
        fake_mode,
        shape_env,
    )
    aot_state.fw_metadata.act_input_paths = act_input_paths

    # NB: no cache lookup!
    aot_graph_capture = aot_stage1_graph_capture(aot_state, functional_call)

    if out_spec is None or out_spec.spec is None:
        raise AssertionError("out_spec and out_spec.spec must not be None")
    if in_spec is None:
        raise AssertionError("in_spec must not be None")
    return JointWithDescriptors(
        _aot_state=aot_state,
        _aot_graph_capture=aot_graph_capture,
        params_spec=params_spec,
        buffers_spec=buffers_spec,
        in_spec=in_spec,
        out_spec=out_spec.spec,
    )


def aot_compile_joint_with_descriptors(
    jd: JointWithDescriptors,
    *,
    partition_fn: Callable[..., Any] = default_partition,
    fw_compiler: AOTDispatchCompiler = boxed_nop_preserve_node_meta,
    bw_compiler: AOTDispatchCompiler | None = boxed_nop_preserve_node_meta,
    serializable: bool = False,
) -> Callable[..., Any]:
    """
    Companion function for aot_export_joint_with_descriptors which compiles the joint
    graph into a callable function that follows a standard calling convention.
    params_flat all are arguments.

    Note: We do NOT instantiate the module; this gives you the flexibility to subclass it and
    customize its behavior without having to worry about FQN rebinding.

    Args:
        serializable: If True, configures the compilation to produce a serializable
            callable by leveraging AOTAutogradCache machinery. This sets up the
            necessary cache_info and patches config options required for serialization.
            When True, this function will always return a BundledAOTAutogradSerializableCallable.
    """
    # TODO: Consider if we should allow_in_graph the result by default.
    from torch._dynamo.aot_compile_types import (
        BundledAOTAutogradSerializableCallable,
        SerializableCallable,
    )
    from torch._functorch._aot_autograd.schemas import AOTAutogradCacheInfo
    from torch._guards import detect_fake_mode
    from torch._inductor.output_code import OutputCode

    fw_compiler = SerializableAOTDispatchCompiler(OutputCode, fw_compiler)

    cache_ctx = nullcontext()
    if serializable:
        jd._aot_state.aot_config = dataclasses.replace(
            jd._aot_state.aot_config,
            cache_info=AOTAutogradCacheInfo(
                jd.cache_hash(),
                time.time_ns(),
                forward_symints=[],
            ),
        )
        cache_ctx = torch._functorch.config.patch(
            {
                "strict_autograd_cache": True,
                "bundled_autograd_cache": True,
                "force_non_lazy_backward_lowering": True,
                "bypass_autograd_cache_key": True,
            }
        )

    # Set up a TracingContext if one isn't already available.
    # This is needed for compilers (like regional_inductor) that call
    # standalone_compile with dynamic_shapes="from_tracing_context".
    tracing_context = torch._guards.TracingContext.try_get()
    if tracing_context is None:
        fake_mode = detect_fake_mode(jd._aot_state.flat_args)
        if fake_mode is not None:
            tracing_context = torch._guards.TracingContext(fake_mode)
            tracing_ctx = torch._guards.tracing(tracing_context)
        else:
            tracing_ctx = nullcontext()
    else:
        tracing_ctx = nullcontext()

    with cache_ctx, tracing_ctx:
        compiled_fn, _ = aot_stage2_compile(
            jd._aot_state,
            jd._aot_graph_capture,
            partition_fn,
            fw_compiler,
            bw_compiler,
        )

    if not isinstance(compiled_fn, SerializableCallable) and hasattr(
        compiled_fn, "serialize"
    ):
        compiled_fn = BundledAOTAutogradSerializableCallable(compiled_fn)

    # Cribbed from torch/export/pt2_archive/_package.py
    @simple_wraps(compiled_fn)
    @torch._dynamo.nonstrict_trace  # allow recursive compilation
    def unflattened_compiled_fn(*args: Any, **kwargs: Any) -> Any:
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, jd.in_spec)))[0]
        # TODO: do I need to filter? I hope not!
        flat_outputs = compiled_fn(flat_inputs)
        return pytree.tree_unflatten(flat_outputs, jd.out_spec)

    return unflattened_compiled_fn


def aot_export_module(
    mod: nn.Module,
    args: Iterable[Any],
    *,
    decompositions: dict[OpOverload, Callable[..., Any]] | None = None,
    # If true, we'll return a joint forward-backward graph,
    # As well as metadata on the loss + gradients in the backward.
    trace_joint: bool,
    # If trace_joint is True, we expect your module to return a scalar loss.
    # Your module can return multiple outputs, so you must specify which output the loss is.
    output_loss_index: int | None = None,
    pre_dispatch: bool = False,
    # If None, will be inferred from inputs and mod.graph.nodes if mod is a graph module, but the inferred result might be wrong.
    dynamic_shapes: bool | None = None,
    kwargs: dict[str, Any] | None = None,
) -> tuple[Callable[..., Any] | torch.fx.GraphModule, GraphSignature]:
    """
    This function takes in a module, and returns:
    (1) an FX graph that can be exported
    (2) some metadata about the graph

    If `trace_joint=True` we will return a joint graph of the forward + backward.

    The traced FX graph will have the following properties compared to the original module:
    (1) Inputs and outputs to the module will be pytree-flattened
    (2) Parameters and buffers on the module will be lifted into graph inputs,
        graph_inputs = (*parameters, *buffers, *user_inputs)
    (3) The graph will be fully functionalized
    (4) Any input mutations will be converted into additional outputs in the graph,
        meaning whoever calls this graph is responsible for applying the mutations
        back to the original inputs.
    (5) If is_joint is provided the graph will return parameter gradients in addition to user outputs.
        The graph output will look like:
        graph_outputs = (*updated_inputs, *user_outputs, *param_gradients)

    There are also several restrictions on what modules can use this API. In particular:
    (1) If trace_joint is specified, we expect the loss function to be **fused**
        into the module forward. One of the outputs to the forward must be a scalar loss,
        which is specified with `output_loss_index`.
        All other outputs to the forward are presumed to not require gradients.
    (2) This API cannot capture optimizers (although in theory we could build an API for this).
    (3) Metadata mutations on params/buffers/inputs are banned.
    (4) Data mutations on anything that requires gradients are banned (parameters)
    (5) If an input is mutated, it is not allowed to alias any other inputs.
    (6) Parameters must not be duplicated.
    """
    if pre_dispatch and trace_joint:
        raise RuntimeError("pre_dispatch is not supported when trace_joint is True.")
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))

    params_and_buffers = {
        **dict(named_parameters),
        **dict(named_buffers),
    }
    params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
    params_and_buffers_flat = tuple(params_and_buffers_flat)
    params_len = len(params_and_buffers_flat)

    kwargs = kwargs or {}

    functional_call = create_functional_call(
        mod, params_spec, params_len, store_orig_mod=True
    )

    num_fw_outs = None

    if trace_joint:
        # This helper effectively just adds some extra asserts about what the backward will look like:
        # Outputs must include a scalar loss, that we compute gradients w.r.t.
        # We don't compute gradients w.r.t. anything else: so just in case we detach()
        # and other output tensors.
        def fn_to_trace(*args: Any) -> Any:
            nonlocal num_fw_outs
            out = functional_call(*args)
            if output_loss_index is None:
                raise RuntimeError(
                    """\
If trace_joint=Trueit is required that one of your forward outputs must be a scalar loss.
You must specify the which (index) output is the loss with output_loss_index."""
                )
            if isinstance(out, (torch.Tensor)):
                out = (out,)
            if not isinstance(out, (tuple, list)):
                raise RuntimeError(
                    f"Expected forward output to be either a tensor or a list/tuple of tensors. found {type(out)}"
                )

            for i, o in enumerate(out):
                # We only want to create a backward graph w.r.t. the loss that the user passed in.
                # This implies that every other output should not require gradients.
                # Instead of making this an error (and forcing the user to detach all other outputs
                # of their forward),
                # we'll automatically detach them here.
                if o.requires_grad and i != output_loss_index:
                    raise RuntimeError(
                        f"""\
Found an output of the forward that requires gradients, that was not the scalar loss.
We require all outputs to the forward that are not the scalar loss to not require gradient,
because we will only compute a backward graph against the scalar loss.
You can fix this by calling .detach() on each of your forward outputs that is not the loss.
You specified that output index {output_loss_index} is the loss, but we found that
the output at index {i} requires gradients."""
                    )
            out_loss = out[output_loss_index]
            num_fw_outs = len(out)
            if not out_loss.requires_grad:
                raise RuntimeError(
                    f"""\
The output at index {output_loss_index} was marked as the loss, but it does not require gradients"""
                )
            if out_loss.numel() != 1:
                raise RuntimeError(
                    f"""\
We require the output marked as the loss (at index {output_loss_index}) to be a scalar, but it has shape {out_loss.shape}"""
                )
            return out

        ctx = nullcontext
    else:
        # Run under no_grad, so our tracing machinery only traces an inference graph.
        # However if pre_dispatch=True, we want to correctly trace set_grad_enabled calls for training.
        ctx = nullcontext if pre_dispatch else torch.no_grad
        fn_to_trace = functional_call

    full_args = []
    # First, the params
    # NB: It is REQUIRED that parameters come first, Inductor infers "fixed"
    # parameters by looking at the difference in parameter count outside
    # and inside AOTAutograd, and assumes the prefix of arguments are fixed
    # arguments
    full_args.extend(params_and_buffers_flat)
    # Next, the input args
    full_args.extend(args)

    with ctx():
        fx_g, metadata, in_spec, out_spec = _aot_export_function(
            fn_to_trace,
            tuple(full_args),
            decompositions=decompositions,
            num_params_buffers=params_len,
            no_tangents=True,
            pre_dispatch=pre_dispatch,
            dynamic_shapes=dynamic_shapes,
            trace_joint=trace_joint,
            kwargs=kwargs,
        )

    # TODO: subsume this path with the aot_stage2_graph_capture path
    if trace_joint:

        @wraps(functional_call)
        def flattened_joint(*args: Any) -> Any:
            # The idea here is that the joint graph that AOTAutograd creates has some strict properties:
            # (1) It accepts two arguments (primals, tangents), and pytree_flattens them
            # (2) It returns a tuple of (fw_outs, gradients)
            # This is a very useful convention for anyone who wants to partition the joint graph
            # into a separate forward and backward graph.
            # However,
            # (1) for people exporting a single joint graph, it would be preferable not to have
            #     any pytrees in the graph.
            # (2) We are guaranteed in the aot_export_module case that the forward outputs a loss,
            #     and there are therefore no tangents that are needed to run the joint graph.
            # (3) AOTAutograd creates a grad_input for every input in the forward,
            #     including None's for inputs that are not grad-requiring tensors.
            #     we don't want these in our export graph.
            #     and there are therefore no tangents that are needed to run the joint graph.
            # This function "fixes" both of the above by removing any tangent inputs,
            # and removing pytrees from the original FX graph.
            fake_tangents = [
                None
                for _ in range(
                    metadata.num_outputs + metadata.num_mutated_inp_runtime_indices
                )
            ]
            fw_outs, gradients = fx_g(args, fake_tangents)
            if len(gradients) != len(args):
                raise AssertionError(
                    f"len(gradients)={len(gradients)} != len(args)={len(args)}"
                )
            output_gradients = []
            for a, grad in zip(args, gradients):
                if isinstance(a, torch.Tensor) and a.requires_grad:
                    if grad is None:
                        raise AssertionError("""\
Found a parameter that did not receive a gradient.
"This is most likely a bug, but if this needs to be supported please comment on this Github issue:
https://github.com/pytorch/pytorch/issues/101192
""")
                    output_gradients.append(grad)
                else:
                    if grad is not None:
                        raise AssertionError(
                            f"expected grad to be None for non-tensor or non-requires_grad input, got {type(grad)}"
                        )
            return *fw_outs, *output_gradients

        fx_g = make_fx(flattened_joint, record_module_stack=True)(*full_args)

    user_args_flat = pytree.arg_tree_leaves(*args, **kwargs)
    if out_spec is None:
        raise AssertionError("out_spec must not be None")
    return fx_g, create_graph_signature(
        # type: ignore[bad-argument-type]
        fx_g,
        metadata,
        in_spec,
        out_spec,
        user_args_flat=user_args_flat,
        params_and_buffers_flat=list(params_and_buffers_flat),
        param_names=list(named_parameters.keys()),
        buffer_names=list(named_buffers.keys()),
        trace_joint=trace_joint,
        num_user_fw_outs=num_fw_outs,
        loss_index=output_loss_index,
    )


def aot_export_joint_simple(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    trace_joint: bool,
    # It looks like the main consequence of this API is that for dynamic shapes,
    # it will assume that params/buffers are static.
    # With the new inferred dynamic shapes API, maybe this doesn't matter?
    num_params_buffers: int = 0,
    decompositions: dict[OpOverload, Callable[..., Any]] | None = None,
) -> Callable[..., Any]:
    """
    A simplified version of export. Used by higher order operators.

    This function makes a high-level "no calling convention changes" guarantee:
    - If no inputs require grad (so we export an inference graph),
      there are *no* calling convention change between the exported graph, and "func".
    - If at least one input requires grad (so we trace out and export a joint fw-bw graph),
      Then if you were partition the graph into a separate forward and backward graph,
      The forward graph will have no calling convention changes compared to "func".

    The above also relies on some strong restrictions around which functions this API accepts:
    (1) `args` cannot contain any pytrees (they must have been pytree_flattened already)
    (2) `func` cannot mutate any inputs
    (3) The outputs of `func` cannot alias any inputs.

    Note: this function is only lightly tested today. It will probably be tested more heavily by higher order ops.
    """
    if trace_joint:
        ctx = nullcontext
    else:
        # Run under no_grad, so our tracing machinery only traces an inference graph.
        ctx = torch.no_grad

    with ctx():
        fx_g, metadata, in_spec, out_spec = _aot_export_function(
            func,
            args,
            decompositions=decompositions,
            trace_joint=trace_joint,
        )
        in_spec, _kw_in_spec = in_spec.children()
    # At this point, we can just directly return the (joint or inference graph) that we traced.
    # First though: a bunch of assertions to make sure that our graph doesn't require
    # any calling convention changes compared to the original function.
    # These restrictions are *in addition to* the general restrictions on export.

    # No input mutations
    if (
        len([x for x in metadata.input_info if x.mutates_data or x.mutates_metadata])
        != 0
    ):
        raise RuntimeError(
            f"aot_export_joint_simple does not support input mutations. {str(metadata)}"
        )
    # No output aliasing
    if (
        len([x for x in metadata.output_info if x.output_type != OutputType.non_alias])
        != 0
    ):
        raise RuntimeError(
            f"aot_export_joint_simple does not support outputs that alias inputs. {str(metadata)}"
        )
    # No pytrees
    if in_spec.is_leaf():
        raise RuntimeError(
            f"aot_export_joint_simple requires inputs to be a single list/tuple. in_spec={str(in_spec)}"
        )
    if not all(child.is_leaf() for child in in_spec.children()):
        raise RuntimeError(
            f"aot_export_joint_simple requires individual inputs not to be pytrees. in_spec={str(in_spec)}"
        )

    if out_spec is None:
        raise AssertionError("out_spec must not be None")
    if out_spec.is_leaf():
        raise RuntimeError(
            f"aot_export_joint_simple requires outputs to be a single list/tuple. out_spec={str(out_spec)}"
        )
    if not all(child.is_leaf() for child in out_spec.children()):
        raise RuntimeError(
            f"aot_export_joint_simple requires individual outputs not to be pytrees. out_spec={str(out_spec)}"
        )
    # TODO: we might have to temporarily patch config.functionalize_rng
    # so that it doesn't run when we're exporting a higher order op.

    if config.debug_assert:
        # Smoke test that after partitioning, we can run the forward without any calling convention changes.
        fw_module, _bw_module = default_partition(
            # type: ignore[bad-argument-type]
            fx_g,
            args,
            # type: ignore[unknown-name]
            num_fwd_outputs=len(fw_metadata.output_infos),  # noqa: F821
        )
        # Attempt to run the fw_module with the original user inputs
        fake_mode = detect_fake_mode(args)
        if fake_mode is None:
            fake_mode = FakeTensorMode()
        with fake_mode:
            fw_module(*args)
    return fx_g


# Private for now because we aren't providing a contract on what to return
# for joint graphs (we could when there's a clearer use case)
# In the future, we may need to add more export API's that provide their own strong guarantees.
# This is meant as a general helper function for handling various export-y use cases.
def _aot_export_function(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    num_params_buffers: int = 0,
    decompositions: dict[OpOverload, Callable[..., Any]] | None = None,
    # If we're exporting a joint graph and we don't want any tangent inputs in the graph
    # (because we are backpropping through a scalar 1 loss),
    # we need to explicitly specify not to include tangents in the graph.
    # It's not enough just to check that our tangent is a scalar, since we also
    # need to know if it is a 1 (no need to make it a graph input), or something else
    # (requiring it to be a graph input).
    # We don't know this info at trace time though, so we need to make it an explicit config.
    no_tangents: bool = False,
    pre_dispatch: bool = False,
    # If None, `dynamic_shapes` will be inferred from inputs, but the inferred result might be wrong.
    dynamic_shapes: bool | None = None,
    keep_input_mutations: bool = False,
    # Under export, configures whether we are getting inference or training IR
    trace_joint: bool = False,
    kwargs: dict[str, Any] | None = None,
) -> tuple[
    Callable[..., Any], ViewAndMutationMeta, pytree.TreeSpec, pytree.TreeSpec | None
]:
    kwargs = kwargs or {}

    flat_fn, out_spec = create_tree_flattened_fn(func, args, kwargs)
    flat_args, in_spec = pytree.tree_flatten((args, kwargs))

    fake_mode = None
    if dynamic_shapes is None:
        # Try to infer `dynamic_shapes from inputs and graph nodes
        fake_mode = detect_fake_mode(flat_args)
        if (
            fake_mode is None
            and hasattr(func, "_orig_mod")
            and isinstance(func._orig_mod, torch.fx.GraphModule)
        ):
            vals = [
                node.meta["val"]
                for node in func._orig_mod.graph.nodes
                if "val" in node.meta
            ]
            fake_mode = detect_fake_mode(vals)
        dynamic_shapes = fake_mode is not None and fake_mode.shape_env is not None

    # The export use case doesn't care about several bits of AOTConfig
    # (1) compilers (we just export the graph)
    # (2) partitioners (export is only full graph, user can partition themselves)
    aot_config = AOTConfig(  # type: ignore[arg-type]
        fw_compiler=None,
        bw_compiler=None,
        inference_compiler=None,
        partition_fn=None,
        decompositions=decompositions,
        num_params_buffers=num_params_buffers,
        aot_id=next(AOT_COUNTER),
        keep_inference_input_mutations=keep_input_mutations,
        dynamic_shapes=dynamic_shapes,
        aot_autograd_arg_pos_to_source=None,
        is_export=True,
        no_tangents=no_tangents,
        pre_dispatch=pre_dispatch,
        export_trace_joint=trace_joint,
    )
    if fake_mode is None:
        fake_mode, shape_env = construct_fake_mode(flat_args, aot_config)
    else:
        shape_env = fake_mode.shape_env
    fake_flat_args, act_input_paths = process_inputs(
        flat_args, aot_config, fake_mode, shape_env
    )
    # TODO: Improve the descs here with pytree information
    fake_flat_args_descs: list[AOTInput] = [
        PlainAOTInput(i) for i in range(len(fake_flat_args))
    ]

    with contextlib.ExitStack() as stack:
        aot_state = create_aot_state(
            stack,
            flat_fn,
            fake_flat_args,
            fake_flat_args_descs,
            aot_config,
            fake_mode,
            shape_env,
        )
        aot_state.fw_metadata.act_input_paths = act_input_paths
        aot_graph_capture = aot_stage1_graph_capture(aot_state, flat_fn)
        fx_g, meta = aot_stage2_export(aot_state, aot_graph_capture)

    return fx_g, meta, in_spec, out_spec.spec


compiled_function = aot_function
compiled_module = aot_module
<|endoftext|>"""
The APIs in this file are exposed as `functorch.*`. They are thin wrappers
around the torch.func.* APIs that have deprecation warnings -- we're trying
to move people to the torch.func.* equivalents.

NB: We don't use *args, **kwargs in the signatures because that changes the
documentation.
"""

from __future__ import annotations

import textwrap
import warnings
from typing import Any, TYPE_CHECKING

import torch._functorch.apis as apis
import torch._functorch.eager_transforms as _impl
import torch._functorch.make_functional as _nn_impl
import torch.nn as nn


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._functorch.eager_transforms import argnums_t
    from torch._functorch.vmap import in_dims_t, out_dims_t


def get_warning(
    api: str, new_api: str | None = None, replace_newlines: bool = False
) -> str:
    if new_api is None:
        new_api = f"torch.func.{api}"
    warning = (
        f"We've integrated functorch into PyTorch. As the final step of the \n"
        f"integration, `functorch.{api}` is deprecated as of PyTorch \n"
        f"2.0 and will be deleted in a future version of PyTorch >= 2.3. \n"
        f"Please use `{new_api}` instead; see the PyTorch 2.0 release notes \n"
        f"and/or the `torch.func` migration guide for more details \n"
        f"https://pytorch.org/docs/main/func.migrating.html"
    )
    if replace_newlines:
        warning = warning.replace("\n", "")
    return warning


def warn_deprecated(api: str, new_api: str | None = None) -> None:
    warning = get_warning(api, new_api, replace_newlines=True)
    warnings.warn(warning, FutureWarning, stacklevel=3)


def setup_docs(
    functorch_api: Callable[..., Any],
    torch_func_api: Callable[..., Any] | None = None,
    new_api_name: str | None = None,
) -> None:
    api_name = functorch_api.__name__
    if torch_func_api is None:
        torch_func_api = getattr(_impl, api_name)
    # See https://docs.python.org/3/using/cmdline.html#cmdoption-OO
    if torch_func_api.__doc__ is None:
        return

    warning = get_warning(api_name, new_api_name)
    warning_note = "\n.. warning::\n\n" + textwrap.indent(warning, "    ")
    warning_note = textwrap.indent(warning_note, "    ")
    functorch_api.__doc__ = torch_func_api.__doc__ + warning_note


def vmap(
    func: Callable[..., Any],
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    randomness: str = "error",
    *,
    chunk_size: int | None = None,
) -> Callable[..., Any]:
    warn_deprecated("vmap", "torch.vmap")
    return apis.vmap(func, in_dims, out_dims, randomness, chunk_size=chunk_size)


def grad(
    func: Callable[..., Any], argnums: argnums_t = 0, has_aux: bool = False
) -> Callable[..., Any]:
    warn_deprecated("grad")
    return apis.grad(func, argnums, has_aux)


def grad_and_value(
    func: Callable[..., Any], argnums: argnums_t = 0, has_aux: bool = False
) -> Callable[..., Any]:
    warn_deprecated("grad_and_value")
    return apis.grad_and_value(func, argnums, has_aux)


def vjp(func: Callable[..., Any], *primals: Any, has_aux: bool = False) -> Any:
    warn_deprecated("vjp")
    return _impl.vjp(func, *primals, has_aux=has_aux)


def jvp(
    func: Callable[..., Any],
    primals: Any,
    tangents: Any,
    *,
    strict: bool = False,
    has_aux: bool = False,
) -> Any:
    warn_deprecated("jvp")
    return _impl.jvp(func, primals, tangents, strict=strict, has_aux=has_aux)


def jacrev(
    func: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
    *,
    has_aux: bool = False,
    chunk_size: int | None = None,
    _preallocate_and_copy: bool = False,
) -> Callable[..., Any]:
    warn_deprecated("jacrev")
    return _impl.jacrev(
        func,
        argnums,
        has_aux=has_aux,
        chunk_size=chunk_size,
        _preallocate_and_copy=_preallocate_and_copy,
    )


def jacfwd(
    func: Callable[..., Any],
    argnums: argnums_t = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
) -> Callable[..., Any]:
    warn_deprecated("jacfwd")
    return _impl.jacfwd(func, argnums, has_aux, randomness=randomness)


def hessian(func: Callable[..., Any], argnums: int = 0) -> Callable[..., Any]:
    warn_deprecated("hessian")
    return _impl.hessian(func, argnums=argnums)


def functionalize(
    func: Callable[..., Any], *, remove: str = "mutations"
) -> Callable[..., Any]:
    warn_deprecated("functionalize")
    return _impl.functionalize(func, remove=remove)


def make_functional(model: nn.Module, disable_autograd_tracking: bool = False) -> Any:
    warn_deprecated("make_functional", "torch.func.functional_call")
    return _nn_impl.make_functional(model, disable_autograd_tracking)


def make_functional_with_buffers(
    model: nn.Module, disable_autograd_tracking: bool = False
) -> Any:
    warn_deprecated("make_functional_with_buffers", "torch.func.functional_call")
    return _nn_impl.make_functional_with_buffers(model, disable_autograd_tracking)


def combine_state_for_ensemble(models: list[nn.Module]) -> Any:
    warn_deprecated("combine_state_for_ensemble", "torch.func.stack_module_state")
    return _nn_impl.combine_state_for_ensemble(models)


setup_docs(vmap, apis.vmap, "torch.vmap")
setup_docs(grad, apis.grad)
setup_docs(grad_and_value, apis.grad_and_value)
setup_docs(vjp)
setup_docs(jvp)
setup_docs(jacrev)
setup_docs(jacfwd)
setup_docs(hessian)
setup_docs(functionalize)
setup_docs(make_functional, _nn_impl.make_functional, "torch.func.functional_call")
setup_docs(
    make_functional_with_buffers, _nn_impl.make_functional, "torch.func.functional_call"
)
setup_docs(
    combine_state_for_ensemble,
    _nn_impl.combine_state_for_ensemble,
    "torch.func.stack_module_state",
)
<|endoftext|>"""AC rematerialize pass: Duplicates recompute nodes for backward, then DCE removes unused forward versions."""

import itertools
import logging
from typing import Any, overload

import torch
import torch.fx as fx
from torch._functorch.compile_utils import raise_getitems
from torch._functorch.partitioners import (
    cleanup_recompute_tags,
    force_save_bw_mutation_src,
    has_recomputable_ops,
    has_recomputable_rng_ops,
    is_not_collective,
    must_recompute,
)


log = logging.getLogger(__name__)
_EMPTY_CUSTOM_META: dict[str, object] = {}


def is_impure_node_for_dce(node: fx.Node) -> bool:
    # Check for special collectives that should be treated as pure
    if not is_not_collective(node):
        # It's a collective (wait_tensor, all_gather_into_tensor, etc.)
        # Treat as pure - can be eliminated if unused
        return False

    # For everything else, fall back to the DEFAULT logic
    # This is what eliminate_dead_code() calls when is_impure_node=None
    impure_random = True
    if torch._guards.TracingContext.try_get():
        impure_random = torch._inductor.config.fallback_random
    return node.is_impure(impure_random)


def _is_backward_node(node: fx.Node, use_phase: bool = False) -> bool:
    """Check if node is in backward region.

    If use_phase is True, only checks custom["phase"] == "backward"
    (user annotation). Otherwise falls back to node.meta["autograd_backward"],
    which Dynamo adds when tracing torch.autograd.grad.
    """
    custom = node.meta.get("custom", _EMPTY_CUSTOM_META)
    if use_phase:
        return custom.get("phase") == "backward"
    return node.meta.get("autograd_backward", False)


def _has_user_phase_annotation(gm: fx.GraphModule) -> bool:
    """Check if any node has the user-level phase: backward annotation."""
    return any(
        node.meta.get("custom", _EMPTY_CUSTOM_META).get("phase") == "backward"
        for node in gm.graph.nodes
    )


def _collect_backward_regions(
    gm: fx.GraphModule, use_phase: bool
) -> list[tuple[int, int, bool]]:
    """Returns (bwd_start, bwd_end, needs_remat) for each backward region.

    Regions are maximal contiguous runs of backward nodes, as [start, end)
    indices into the graph node list.
    """
    regions: list[tuple[int, int, bool]] = []
    bwd_start: int | None = None
    needs_remat = False

    for idx, node in enumerate(gm.graph.nodes):
        if _is_backward_node(node, use_phase=use_phase):
            if bwd_start is None:
                bwd_start = idx
                needs_remat = False
            if not needs_remat and any(
                must_recompute(inp) for inp in node.all_input_nodes
            ):
                needs_remat = True
        elif bwd_start is not None:
            regions.append((bwd_start, idx, needs_remat))
            bwd_start = None

    if bwd_start is not None:
        regions.append((bwd_start, idx + 1, needs_remat))

    return regions


def remat_using_tags_for_fwd_loss_bwd_graph(gm: fx.GraphModule) -> fx.GraphModule:
    """
    Duplicate recompute nodes for backward use. DCE removes unused forward versions.

    Backward regions are identified by custom["phase"] == "backward" (user
    annotation) or node.meta["autograd_backward"] == True (set automatically when
    Dynamo traces torch.autograd.grad). When the user provides phase
    annotations, only those annotated regions are used.

    The graph may contain multiple disjoint backward regions (e.g. chunked
    loss). Regions that do not depend on recomputable forward nodes are
    skipped. Only one region may require remat; if multiple do, we error
    and ask the user to annotate which region to rematerialize.
    """
    if not has_recomputable_ops(gm):
        return gm

    if has_recomputable_rng_ops(gm):
        raise RuntimeError(
            "Activation checkpoint rematerialization in `forward-loss-backward` graph does not support RNG ops "
            "in recompute regions. Please move RNG operations outside "
            "of recompute regions, or use joint graph mode (where partitioner handles RNG)."
        )

    # Use partitioner pass to normalize AC node tags.
    gm = cleanup_recompute_tags(gm, is_default_partition=True)

    force_save_bw_mutation_src(gm)

    # must_recompute (used inside _collect_backward_regions) requires
    # cleanup_recompute_tags to have run first.
    use_phase = _has_user_phase_annotation(gm)
    regions = _collect_backward_regions(gm, use_phase)
    if not regions:
        return gm

    # User-annotated phase regions: multiple annotations is always an error.
    if use_phase and len(regions) > 1:
        raise RuntimeError(
            f"Detected {len(regions)} disjoint backward regions annotated with "
            'phase: "backward" but remat only supports a single backward region. '
            "Please ensure only one contiguous region is annotated."
        )

    remat_regions = [(s, e) for s, e, needs in regions if needs]

    if len(remat_regions) > 1:
        raise RuntimeError(
            f"Detected {len(remat_regions)} disjoint backward regions that require recomputation, "
            "but remat only supports one such region in a forward-loss-backward graph."
        )

    if not remat_regions:
        return gm

    bwd_start, bwd_end = remat_regions[0]

    order = {node: idx for idx, node in enumerate(gm.graph.nodes)}
    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    recomputed_nodes: dict[fx.Node, fx.Node] = {}

    # Insert forward nodes
    for node in itertools.islice(gm.graph.nodes, 0, bwd_start):
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    def _ensure_in_env(node: fx.Node) -> fx.Node:
        """Eagerly copy a backward node and its dependencies into the new graph.

        The CPU offloading pass must run before remat. Offloaded forward nodes
        carry ``cpu_offload_reload_node`` metadata pointing to their backward
        reload chain (ao.reload -> ao.wait_tensor). Recomputed nodes need the
        reloaded tensor, but the reload chain may not have been copied into
        ``env`` yet (e.g. layer-boundary nodes whose reload is in a later
        backward region). This helper copies the chain on demand.
        """
        if node in env:
            return env[node]
        for dep in node.all_input_nodes:
            _ensure_in_env(dep)
        env[node] = new_graph.node_copy(node, lambda x: env[x])
        return env[node]

    @overload
    def remat_input(x: fx.Node) -> fx.Node: ...
    @overload
    def remat_input(x: Any) -> Any: ...

    def remat_input(x: object) -> object:
        if not isinstance(x, fx.Node):
            return x
        if x in recomputed_nodes:
            return recomputed_nodes[x]
        reload_node = x.meta.get("cpu_offload_reload_node")
        if reload_node is not None:
            return _ensure_in_env(reload_node)
        return env[x]

    def gather_recompute_deps(node: fx.Node) -> set[fx.Node]:
        deps: set[fx.Node] = set()

        def _gather(n: fx.Node) -> None:
            if n in deps or n in recomputed_nodes or not must_recompute(n):
                return
            deps.add(n)
            for inp in n.all_input_nodes:
                _gather(inp)

        # Can't call _gather(node) directly: node itself may not be must_recompute
        # (e.g. backward nodes), so _gather would return early without visiting inputs.
        for inp in node.all_input_nodes:
            _gather(inp)
        return deps

    # Insert backward nodes
    for node in itertools.islice(gm.graph.nodes, bwd_start, bwd_end):
        # Gather all deps that need to be recomputed for this node
        deps = gather_recompute_deps(node)

        # Insert deps in forward order (guaranteed disjoint from already-inserted)
        # This is not as inefficient as it looks, because we only add fresh dependencies
        # when they are not yet processed as recomputed nodes.
        new_deps = sorted(deps, key=lambda n: order[n])
        if new_deps:
            log.debug(
                "To compute backward node %s, recomputing [%s]",
                node.name,
                ", ".join(dep.name for dep in new_deps),
            )
        for dep in new_deps:
            dup = new_graph.node_copy(dep, remat_input)
            dup.name = dep.name + "_recomputed"
            dup.meta["autograd_backward"] = True
            recomputed_nodes[dep] = dup

        # Skip nodes already copied by _ensure_in_env (offload reload chains)
        # to avoid duplicating H2D transfers.
        if node not in env:
            env[node] = new_graph.node_copy(node, remat_input)

    for node in itertools.islice(gm.graph.nodes, bwd_end, None):
        if node not in env:
            env[node] = new_graph.node_copy(node, lambda x: env[x])

    new_gm = torch.fx.GraphModule(gm, new_graph)

    # DCE with custom is_impure_node (like default_partition)
    # Treats certain collectives as pure while delegating to default impurity logic
    new_gm.graph.eliminate_dead_code(is_impure_node=is_impure_node_for_dce)

    # raise_getitems pass for better memory (like default_partition)
    new_gm = raise_getitems(new_gm)

    new_gm.recompile()

    return new_gm
<|endoftext|>import json
import logging
from typing import Any

from torch._logging import trace_structured
from torch.fx import Graph, Node


log: logging.Logger = logging.getLogger(__name__)


def create_joint_graph_node_information(
    joint_graph: Graph,
    recomputable_node_info: dict[str, int],
) -> dict[str, Any]:
    joint_graph_node_information: dict[str, Any] = {}

    for i, joint_graph_node in enumerate(joint_graph.nodes):
        is_recomputable_candidate: bool = (
            joint_graph_node.name in recomputable_node_info
        )
        tensor_meta = joint_graph_node.meta.get("tensor_meta")
        # pyrefly: ignore [implicit-any]
        shape = getattr(tensor_meta, "shape", []) if tensor_meta else []

        node_info: dict[str, Any] = {
            "index": i,
            "name": joint_graph_node.name,
            "is_recomputable_candidate": is_recomputable_candidate,
            "target": str(joint_graph_node.target),
            "shape": str(shape),
            "input_arguments": [inp.name for inp in joint_graph_node.all_input_nodes],
            "stack_trace": joint_graph_node.meta.get("stack_trace", ""),
        }

        if is_recomputable_candidate:
            idx: int = recomputable_node_info[joint_graph_node.name]
            node_info["recomputable_candidate_info"] = {
                "recomputable_node_idx": idx,
            }

        joint_graph_node_information[joint_graph_node.name] = node_info

    return joint_graph_node_information


def create_joint_graph_edges(joint_graph: Graph) -> list[tuple[str, str]]:
    joint_graph_edges: list[tuple[str, str]] = [
        (inp.name, node.name)
        for node in joint_graph.nodes
        for inp in node.all_input_nodes
    ]
    return joint_graph_edges


def create_activation_checkpointing_logging_structure_payload(
    joint_graph: Graph,
    joint_graph_node_information: dict[str, Any],
    joint_graph_edges: list[tuple[str, str]],
    all_recomputable_banned_nodes: list[Node],
    expected_runtime: float,
    saved_node_idxs: list[int],
    recomputable_node_idxs: list[int],
    memories_banned_nodes: list[int],
    normalized_memories_banned_nodes: list[float],
    runtimes_banned_nodes: list[float],
    min_cut_saved_values: list[Node],
    memory_budget: float,
    min_act_size: float,
    max_act_size: float,
    saved_values_act_size: float,
    more_aggressive_saved_values_mem_ratio: float,
    aggressive_recomputation_saved_values_mem_ratio: float,
) -> dict[str, Any]:
    """
    Creates a structured payload for logging activation checkpointing information.

    Args:
        joint_graph: The computational graph representing operations.
        joint_graph_node_information: Dictionary containing information about nodes in the joint graph.
        joint_graph_edges: List of edges in the joint graph represented as tuples of node names.
        all_recomputable_banned_nodes: List of nodes that are banned from recomputation.
        expected_runtime: Expected runtime of the computation.
        saved_node_idxs: Indices of nodes that are saved (not recomputed).
        recomputable_node_idxs: Indices of nodes that can be recomputed.
        memories_banned_nodes: Memory usage values (in absolute units) for banned nodes.
        normalized_memories_banned_nodes: Normalized memory usage values for banned nodes,
            used as input to the knapsack algorithm.
        runtimes_banned_nodes: Runtime values for banned nodes, used as input to the
            knapsack algorithm.
        min_cut_saved_values: List of nodes saved by the min-cut algorithm.
        memory_budget: The memory budget ratio (0 to 1) used for activation checkpointing.
        min_act_size: Minimum activation size in GB (size when saving only inputs).
        max_act_size: Maximum activation size in GB (size when saving all runtime-optimized values).
        saved_values_act_size: Estimated activation size in GB for the saved values.
        more_aggressive_saved_values_mem_ratio: Memory ratio for more aggressive recomputation strategy.
        aggressive_recomputation_saved_values_mem_ratio: Memory ratio for fully aggressive recomputation strategy.

    Returns:
        A dictionary containing structured logging information for activation checkpointing.
    """
    activation_checkpointing_logging_structure_payload: dict[str, Any] = {
        "Joint Graph Size": len(joint_graph.nodes),
        "Joint Graph Edges": {
            "Total": len(joint_graph_edges),
            "Edges": joint_graph_edges,
        },
        "Joint Graph Node Information": joint_graph_node_information,
        "Recomputable Banned Nodes Order": [
            node.name for node in all_recomputable_banned_nodes
        ],
        "Expected Runtime": expected_runtime,
        "Knapsack Saved Nodes": saved_node_idxs,
        "Knapsack Recomputed Nodes": recomputable_node_idxs,
        "Absolute Memories": memories_banned_nodes,
        "Knapsack Input Memories": normalized_memories_banned_nodes,
        "Knapsack Input Runtimes": runtimes_banned_nodes,
        "Min Cut Solution Saved Values": [node.name for node in min_cut_saved_values],
        "Memory Budget": memory_budget,
        "Min Activation Size (GB)": min_act_size,
        "Max Activation Size (GB)": max_act_size,
        "Saved Values Activation Size (GB)": saved_values_act_size,
        "More Aggressive Saved Values Mem Ratio": more_aggressive_saved_values_mem_ratio,
        "Aggressive Recomputation Saved Values Mem Ratio": aggressive_recomputation_saved_values_mem_ratio,
    }
    return activation_checkpointing_logging_structure_payload


def create_structured_trace_for_min_cut_info(
    joint_graph: Graph,
    all_recomputable_banned_nodes: list[Node],
    saved_node_idxs: list[int],
    recomputable_node_idxs: list[int],
    expected_runtime: float,
    memories_banned_nodes: list[int],
    normalized_memories_banned_nodes: list[float],
    runtimes_banned_nodes: list[float],
    min_cut_saved_values: list[Node],
    memory_budget: float,
    min_act_size: float,
    max_act_size: float,
    saved_values_act_size: float,
    more_aggressive_saved_values_mem_ratio: float,
    aggressive_recomputation_saved_values_mem_ratio: float,
) -> None:
    """
    Creates a structured trace for minimum cut information in the graph.

    Args:
        joint_graph: The computational graph representation.
        all_recomputable_banned_nodes: List of nodes that can be recomputed.
        saved_node_idxs: Indices of nodes that are saved in memory.
        recomputable_node_idxs: Indices of nodes that are recomputed.
        expected_runtime: Expected runtime for the computation.
        memories_banned_nodes: Memory requirements for each banned node in bytes.
        normalized_memories_banned_nodes: Normalized memory requirements for each banned node
            (typically scaled between 0 and 1 for relative comparison).
        runtimes_banned_nodes: Runtime costs associated with each banned node.
        min_cut_saved_values: Nodes that are saved as part of the minimum cut solution.
        memory_budget: The memory budget ratio (0 to 1) used for activation checkpointing.
        min_act_size: Minimum activation size in GB (size when saving only inputs).
        max_act_size: Maximum activation size in GB (size when saving all runtime-optimized values).
        saved_values_act_size: Estimated activation size in GB for the saved values.
        more_aggressive_saved_values_mem_ratio: Memory ratio for more aggressive recomputation strategy.
        aggressive_recomputation_saved_values_mem_ratio: Memory ratio for fully aggressive recomputation strategy.
    """
    # Create a dictionary to store recomputable node information
    recomputable_node_info: dict[str, int] = {
        node.name: idx for idx, node in enumerate(all_recomputable_banned_nodes)
    }

    # Create joint graph node information
    joint_graph_node_information = create_joint_graph_node_information(
        joint_graph, recomputable_node_info
    )

    # Update node information with recomputable candidate details
    for node_name, node_info in joint_graph_node_information.items():
        if node_info["is_recomputable_candidate"]:
            idx = recomputable_node_info[node_name]
            node_info["recomputable_candidate_info"]["memory"] = memories_banned_nodes[
                idx
            ]
            node_info["recomputable_candidate_info"]["runtime"] = runtimes_banned_nodes[
                idx
            ]
            node_info["recomputable_candidate_info"]["is_saved"] = (
                idx in saved_node_idxs
            )
            node_info["recomputable_candidate_info"]["is_recomputed"] = (
                idx in recomputable_node_idxs
            )

    # Create joint graph edges
    joint_graph_edges = create_joint_graph_edges(joint_graph)

    # Create activation checkpointing logging structure payload
    activation_checkpointing_logging_structure_payload = create_activation_checkpointing_logging_structure_payload(
        joint_graph=joint_graph,
        joint_graph_node_information=joint_graph_node_information,
        joint_graph_edges=joint_graph_edges,
        all_recomputable_banned_nodes=all_recomputable_banned_nodes,
        expected_runtime=expected_runtime,
        saved_node_idxs=saved_node_idxs,
        recomputable_node_idxs=recomputable_node_idxs,
        memories_banned_nodes=memories_banned_nodes,
        normalized_memories_banned_nodes=normalized_memories_banned_nodes,
        runtimes_banned_nodes=runtimes_banned_nodes,
        min_cut_saved_values=min_cut_saved_values,
        memory_budget=memory_budget,
        min_act_size=min_act_size,
        max_act_size=max_act_size,
        saved_values_act_size=saved_values_act_size,
        more_aggressive_saved_values_mem_ratio=more_aggressive_saved_values_mem_ratio,
        aggressive_recomputation_saved_values_mem_ratio=aggressive_recomputation_saved_values_mem_ratio,
    )

    # Create structured trace
    payload_str = json.dumps(activation_checkpointing_logging_structure_payload)

    # Store on graph for AOTAutograd cache replay
    joint_graph._min_cut_info_str = payload_str  # type: ignore[attr-defined]

    trace_structured(
        "artifact",
        metadata_fn=lambda: {"name": "min_cut_information", "encoding": "json"},
        payload_fn=lambda: payload_str,
    )
<|endoftext|>import operator
from collections import deque
from collections.abc import Callable

import networkx as nx

from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)


class KnapsackEvaluator:
    """
    This class evaluates the theoretical runtime and peak memory usage of a given checkpointing strategy.
    It takes in a graph and a list of nodes that are saved and recomputed, and then simulates the
    backward pass to calculate the peak memory usage.
    """

    def __init__(
        self,
        graph_info_provider: GraphInfoProvider,
    ) -> None:
        self._graph_info_provider = graph_info_provider

    def _get_backward_memory_from_topologically_sorted_graph(
        self,
        node_graph: nx.DiGraph,
        node_memories: dict[str, float],
        saved_nodes_set: set[str],
        peak_memory_after_forward_pass: float,
    ) -> list[tuple[float, str]]:
        """
        Simulates the backward pass and keeps track of the peak memory usage.

        High Level Steps:
            1. Set Initial Peak/Current Memory
                Allows you to set the peak memory after the forward pass, but typically this is
                the sum of the estimated memory of the saved nodes.
            2. Perform a reverse topological sort of the node_graph.
                If full graph is defined then will sort the full graph and only process the subset
                of nodes in the node_graph.
            3. Iterate through the sorted graph nodes.
                If the node is saved then just drop its memory from current memory.
                If the node is not saved then add its memory to current memory and then traverse its
                predecessors to simulate recomputation chain. Will check if new peak memory after all
                predecessors are processed.

        Args:
            node_graph (nx.DiGraph): A directed graph representing the recomputable forward nodes.
            saved_nodes_set (Set[str]): A set of node names that are saved.
            peak_memory_after_forward_pass (float): The peak memory usage after the forward pass.
        """
        current_memory = [
            (peak_memory_after_forward_pass, "Initial Peak/Current Memory")
        ]
        already_computed = set()
        sorted_nodes = list(reversed(list(nx.topological_sort(node_graph))))
        dependencies_computed = set()

        for node in sorted_nodes:
            if node in saved_nodes_set or node in already_computed:
                current_memory.append(
                    (
                        current_memory[-1][0] - node_memories[node],
                        f"Dropping Node(already saved): {node}",
                    )
                )
                continue

            already_computed.add(node)
            current_memory.append(
                (
                    current_memory[-1][0] + node_memories[node],
                    f"Recomputing Node: {node}",
                )
            )
            # Create a queue of dependencies required for recomputation
            predecessor_queue = deque(
                [
                    dependency
                    # pyrefly: ignore [bad-unpacking]
                    for dependency, v in node_graph.in_edges(node)
                    if dependency not in already_computed
                ]
            )
            while predecessor_queue:
                dep = predecessor_queue.popleft()
                already_computed.add(dep)
                dependencies_computed.add(dep)
                current_memory.append(
                    (
                        current_memory[-1][0] + node_memories[dep],
                        f"Recomputing Predecessor of {node}: {dep}",
                    )
                )
                # Add predecessors of the predecessor to the queue if they haven't been recomputed yet
                # pyrefly: ignore [bad-unpacking]
                for dependency_of_dependency, _ in node_graph.in_edges(dep):
                    if (
                        dependency_of_dependency in already_computed
                        or dependency_of_dependency in saved_nodes_set
                        or dependency_of_dependency in predecessor_queue
                    ):
                        continue
                    predecessor_queue.append(dependency_of_dependency)
            dependencies_computed.clear()
            current_memory.append(
                (current_memory[-1][0] - node_memories[node], f"Dropping Node: {node}")
            )
        return current_memory

    def _validate_all_indexes_accounted_for_in_provided_output(
        self, saved_nodes_idxs: list[int], recomputable_node_idxs: list[int]
    ) -> None:
        """
        Validate that all indexes are accounted for in the provided output.
        This function checks that the union of saved nodes and recomputable nodes
        covers all candidate nodes without any overlaps.
        """
        recomputable_node_idxs_set = set(recomputable_node_idxs)
        saved_nodes_idxs_set = set(saved_nodes_idxs)
        all_candidate_nodes_idxs = set(
            range(len(self._graph_info_provider.all_recomputable_banned_nodes))
        )
        # Check that there are no overlaps between saved nodes and recomputable nodes
        if len(recomputable_node_idxs_set.intersection(saved_nodes_idxs_set)) != 0:
            raise AssertionError(
                f"Saved nodes and recomputable nodes cannot have any overlaps, "
                f"but found overlap: {recomputable_node_idxs_set.intersection(saved_nodes_idxs_set)}"
            )
        # Check that all candidate nodes are accounted for
        if (
            recomputable_node_idxs_set.union(saved_nodes_idxs_set)
            != all_candidate_nodes_idxs
        ):
            raise AssertionError(
                f"All candidate nodes must be accounted for in the provided output, "
                f"got union={recomputable_node_idxs_set.union(saved_nodes_idxs_set)}, "
                f"expected={all_candidate_nodes_idxs}"
            )

    def evaluate_knapsack_output(
        self,
        saved_nodes_idxs: list[int],
        recomputable_node_idxs: list[int],
        account_for_backward_pass: bool = False,
    ) -> dict[str, float]:
        """
        Evaluate the theoretical runtime and peak memory usage of a given checkpointing strategy.
        Args:
        - saved_nodes_idxs (List[int]): The indices of nodes that are saved.
        - recomputable_node_idxs (List[int]): The indices of nodes that need to be recomputed.
        """
        self._validate_all_indexes_accounted_for_in_provided_output(
            saved_nodes_idxs, recomputable_node_idxs
        )
        recomputation_runtime = sum(
            self._graph_info_provider.all_node_runtimes[
                self._graph_info_provider.all_recomputable_banned_nodes[node]
            ]
            for node in recomputable_node_idxs
        )
        if account_for_backward_pass:
            memory_list = self._get_backward_memory_from_topologically_sorted_graph(
                node_graph=self._graph_info_provider.recomputable_node_only_graph_with_larger_graph_context,
                saved_nodes_set={
                    self._graph_info_provider.all_recomputable_banned_nodes[i]
                    for i in saved_nodes_idxs
                },
                node_memories=self._graph_info_provider.all_node_memories,
                peak_memory_after_forward_pass=sum(
                    self._graph_info_provider.all_node_memories[
                        self._graph_info_provider.all_recomputable_banned_nodes[i]
                    ]
                    for i in saved_nodes_idxs
                ),
            )
            peak_memory = max(memory_list, key=operator.itemgetter(0))[0]
        else:
            peak_memory = sum(
                self._graph_info_provider.all_node_memories[
                    self._graph_info_provider.all_recomputable_banned_nodes[node]
                ]
                for node in saved_nodes_idxs
            )
        return {
            "peak_memory": peak_memory,
            "recomputation_runtime": recomputation_runtime,
            "non_ac_peak_memory": self._graph_info_provider.get_non_ac_peak_memory(),
            "theoretical_max_runtime": self._graph_info_provider.get_theoretical_max_runtime(),
            "percentage_of_theoretical_peak_memory": peak_memory
            / self._graph_info_provider.get_non_ac_peak_memory(),
            "percentage_of_theoretical_peak_runtime": recomputation_runtime
            / self._graph_info_provider.get_theoretical_max_runtime(),
        }

    def evaluate_distribution_of_results_for_knapsack_algo(
        self,
        knapsack_algo: Callable[
            [list[float], list[float], float], tuple[float, list[int], list[int]]
        ],
        memory_budget_values: list[float],
    ) -> list[dict[str, float]]:
        """
        Evaluates the distribution of results for a given knapsack algorithm.
        Args:
            knapsack_algo (Callable): The knapsack algorithm to use for evaluation.
            memory_budget_values (List[float]): A list of memory budgets to evaluate.
        """
        results = []
        for memory_budget in memory_budget_values:
            _, saved_nodes, recomputed_nodes = knapsack_algo(
                self._graph_info_provider.get_knapsack_memory_input(),
                self._graph_info_provider.get_knapsack_runtime_input(),
                memory_budget,
            )
            result = self.evaluate_knapsack_output(
                saved_nodes_idxs=saved_nodes,
                recomputable_node_idxs=recomputed_nodes,
            )
            result["memory_budget"] = memory_budget
            results.append(result)
        return results

    def get_knee_point_memory_budget(
        self,
        knapsack_algo: Callable[
            [list[float], list[float], float], tuple[float, list[int], list[int]]
        ],
        max_mem_budget: float = 0.1,
        min_mem_budget: float = 0.001,
        iterations: int = 100,
    ) -> float:
        """
        Finds the memory budget at the knee point in the Pareto frontier.

        The knee point is defined as the point where the trade-off between
        runtime and memory usage is optimal.

        Args:
            knapsack_algo (callable): Knapsack algorithm to use for evaluation.
            max_mem_budget (float, optional): Maximum memory budget. Defaults to 0.1.
            min_mem_budget (float, optional): Minimum memory budget. Defaults to 0.001.
            iterations (int, optional): Number of memory budgets to evaluate. Defaults to 100.

        Returns:
            float: Memory budget at the knee point.
        """
        results = self.evaluate_distribution_of_results_for_knapsack_algo(
            knapsack_algo=knapsack_algo,
            memory_budget_values=[
                min_mem_budget
                + i * (max_mem_budget - min_mem_budget) / (iterations - 1)
                for i in range(iterations)
            ],
        )
        runtime_values = [
            result["percentage_of_theoretical_peak_runtime"] for result in results
        ]
        memory_values = [
            result["percentage_of_theoretical_peak_memory"] for result in results
        ]
        runtime_range = max(runtime_values) - min(runtime_values)
        memory_range = max(memory_values) - min(memory_values)
        if runtime_range == 0 or memory_range == 0:
            return max_mem_budget

        # Normalize values
        runtime_min = min(runtime_values)
        memory_min = min(memory_values)
        runtime_norm = [
            (value - runtime_min) / runtime_range for value in runtime_values
        ]
        memory_norm = [(value - memory_min) / memory_range for value in memory_values]
        # Calculate Euclidean distance
        distances = [
            (runtime_norm[i] ** 2 + memory_norm[i] ** 2) ** 0.5
            for i in range(len(runtime_norm))
        ]
        # Find the knee point(shortest distance from the origin)
        knee_index = distances.index(min(distances))
        return results[knee_index]["memory_budget"]
<|endoftext|>from typing import Any

import networkx as nx

from torch.fx import Graph, Node


class GraphInfoProvider:
    """
    This class provides information about the graph, such as the nodes, edges, and their runtime and memory requirements.
    It also provides methods to create graphs from the information provided.
    """

    __RECOMPUTABLE_NODE_ONLY_GRAPH = "recomputable_node_only_graph"
    __RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT = (
        "recomputable_node_only_graph_with_larger_graph_context"
    )
    __FULL_NX_JOINT_GRAPH = "full_nx_joint_graph"
    __SIMPLIFIED_FX_JOINT_GRAPH = "fx_joint_graph"

    def __init__(
        self,
        graph_nodes_in_order: list[str],
        graph_edges: list[tuple[str, str]],
        all_recomputable_banned_nodes: list[str],
        all_node_runtimes: dict[str, float] | None = None,
        all_node_memories: dict[str, float] | None = None,
        recorded_knapsack_input_memories: list[float] | None = None,
        recorded_knapsack_input_runtimes: list[float] | None = None,
        joint_graph: Graph | None = None,
    ) -> None:
        self.graph_nodes_in_order = graph_nodes_in_order
        self.graph_edges = graph_edges
        self.all_node_runtimes: dict[str, float] = dict()
        if all_node_runtimes is None:
            if recorded_knapsack_input_runtimes is None:
                raise ValueError(
                    "Either all_node_runtimes or recorded_knapsack_input_runtimes must be provided."
                )
            self.all_node_runtimes = {
                node: recorded_knapsack_input_runtimes[i]
                for i, node in enumerate(all_recomputable_banned_nodes)
            }
        else:
            self.all_node_runtimes.update(all_node_runtimes)
        self.all_node_memories: dict[str, float] = dict()
        if all_node_memories is None:
            if recorded_knapsack_input_memories is None:
                raise ValueError(
                    "Either all_node_memories or recorded_knapsack_input_memories must be provided."
                )
            self.all_node_memories = {
                node: recorded_knapsack_input_memories[i]
                for i, node in enumerate(all_recomputable_banned_nodes)
            }
        else:
            self.all_node_memories.update(all_node_memories)
        self.all_recomputable_banned_nodes = all_recomputable_banned_nodes
        self.all_recomputable_banned_nodes_set = set(all_recomputable_banned_nodes)
        self.recorded_knapsack_input_memories = recorded_knapsack_input_memories
        self.recorded_knapsack_input_runtimes = recorded_knapsack_input_runtimes
        self._lazily_initialized_graphs: dict[str, Any] = {
            self.__RECOMPUTABLE_NODE_ONLY_GRAPH: None,
            self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT: None,
            self.__FULL_NX_JOINT_GRAPH: None,
            self.__SIMPLIFIED_FX_JOINT_GRAPH: None,
        }

    @classmethod
    def inialize_from_graph(
        cls,
        joint_graph: Graph,
        all_recomputable_banned_nodes: list[Node],
        recorded_knapsack_input_memories: list[float],
        recorded_knapsack_input_runtimes: list[float],
    ) -> "GraphInfoProvider":
        """
        Enables initialization from a joint graph.
        """
        graph_nodes_in_order = [node.name for node in joint_graph.nodes]
        graph_edges = [
            (node.name, user.name) for node in joint_graph.nodes for user in node.users
        ]
        all_recomputable_banned_node_names = [
            node.name for node in all_recomputable_banned_nodes
        ]
        return cls(
            graph_nodes_in_order=graph_nodes_in_order,
            graph_edges=graph_edges,
            all_recomputable_banned_nodes=all_recomputable_banned_node_names,
            recorded_knapsack_input_memories=recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=recorded_knapsack_input_runtimes,
            joint_graph=joint_graph,
        )

    @property
    def recomputable_node_only_graph(self) -> nx.DiGraph:
        if self._lazily_initialized_graphs[self.__RECOMPUTABLE_NODE_ONLY_GRAPH] is None:
            self._lazily_initialized_graphs[self.__RECOMPUTABLE_NODE_ONLY_GRAPH] = (
                self._create_recomputable_node_only_graph()
            )
        return self._lazily_initialized_graphs[self.__RECOMPUTABLE_NODE_ONLY_GRAPH]

    @property
    def recomputable_node_only_graph_with_larger_graph_context(self) -> nx.DiGraph:
        if (
            self._lazily_initialized_graphs[
                self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT
            ]
            is None
        ):
            self._lazily_initialized_graphs[
                self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT
            ] = self._create_recomputable_node_only_graph_with_larger_graph_context()
        return self._lazily_initialized_graphs[
            self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT
        ]

    @property
    def full_joint_nx_graph(self) -> nx.DiGraph:
        if self._lazily_initialized_graphs[self.__FULL_NX_JOINT_GRAPH] is None:
            self._lazily_initialized_graphs[self.__FULL_NX_JOINT_GRAPH] = (
                self._create_full_joint_graph()
            )
        return self._lazily_initialized_graphs[self.__FULL_NX_JOINT_GRAPH]

    @property
    def simplified_fx_joint_graph(self) -> Graph:
        if self._lazily_initialized_graphs[self.__SIMPLIFIED_FX_JOINT_GRAPH] is None:
            self._lazily_initialized_graphs[self.__SIMPLIFIED_FX_JOINT_GRAPH] = (
                self._recreate_psuedo_joint_graph()
            )
        return self._lazily_initialized_graphs[self.__SIMPLIFIED_FX_JOINT_GRAPH]

    def get_non_ac_peak_memory(self) -> float:
        return sum(
            self.all_node_memories[node_name]
            for node_name in self.all_recomputable_banned_nodes_set
        )

    def get_theoretical_max_runtime(self) -> float:
        return sum(
            self.all_node_runtimes[node_name]
            for node_name in self.all_recomputable_banned_nodes_set
        )

    def get_knapsack_memory_input(self) -> list[float]:
        return (
            self.recorded_knapsack_input_memories
            if self.recorded_knapsack_input_memories
            else [
                self.all_node_memories[node_name]
                for node_name in self.all_recomputable_banned_nodes
            ]
        )

    def get_knapsack_runtime_input(self) -> list[float]:
        return (
            self.recorded_knapsack_input_runtimes
            if self.recorded_knapsack_input_runtimes
            else [
                self.all_node_runtimes[node_name]
                for node_name in self.all_recomputable_banned_nodes
            ]
        )

    def _create_recomputable_node_only_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for recomputable_node in self.all_recomputable_banned_nodes:
            graph.add_node(recomputable_node)

        for a, b in self.graph_edges:
            if (
                a in self.all_recomputable_banned_nodes_set
                and b in self.all_recomputable_banned_nodes_set
            ):
                graph.add_edge(a, b)
        return graph

    def _create_recomputable_node_only_graph_with_larger_graph_context(
        self,
    ) -> nx.DiGraph:
        # Create a dictionary to store the reachable nodes for each node
        all_recomputable_banned_nodes_set = set(self.all_recomputable_banned_nodes)

        reachable_nodes = {}
        for node in all_recomputable_banned_nodes_set:
            # Use BFS to find all reachable nodes
            predecessors = dict(nx.bfs_predecessors(self.full_joint_nx_graph, node))
            reachable_recomputable_nodes = set(predecessors.keys()).intersection(
                all_recomputable_banned_nodes_set
            )
            reachable_nodes[node] = reachable_recomputable_nodes
        # Create the candidate graph
        candidate_graph = nx.DiGraph()
        candidate_graph.add_nodes_from(all_recomputable_banned_nodes_set)
        for node1 in all_recomputable_banned_nodes_set:
            for node2 in reachable_nodes[node1]:
                # Check if there is an overlapping path
                overlapping_path = False
                for intermediate_node in reachable_nodes[node1]:
                    if (
                        intermediate_node != node2
                        and node2 in reachable_nodes[intermediate_node]
                    ):
                        overlapping_path = True
                        break
                if not overlapping_path:
                    candidate_graph.add_edge(node1, node2)
        return candidate_graph

    def _create_full_joint_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node in self.graph_nodes_in_order:
            if node == "output":
                continue
            graph.add_node(node)

        for a, b in self.graph_edges:
            if a == "output" or b == "output":
                continue
            graph.add_edge(a, b)
        return graph

    def _recreate_psuedo_joint_graph(self) -> Graph:
        # Create a dictionary to store the dependencies of each node
        node_dependencies: dict[str, list[str]] = {
            node: [] for node in self.graph_nodes_in_order
        }
        for a, b in self.graph_edges:
            if a not in node_dependencies or b not in node_dependencies:
                raise ValueError(f"Edge ({a}, {b}) references a non-existent node.")
            node_dependencies[b].append(a)

        joint_graph = Graph()
        # Create nodes in the graph
        nodes: dict[str, Node] = {}
        for node_name in self.graph_nodes_in_order:
            input_nodes = [nodes[dep] for dep in node_dependencies[node_name]]
            if input_nodes:
                node = joint_graph.call_function(lambda *x: x, tuple(input_nodes))
                node.name = node_name
            else:
                node = joint_graph.placeholder(node_name)
            nodes[node_name] = node
        return joint_graph

    def _visualize_recomputable_candidate_graph_with_larger_context(
        self,
        layout_k: float = 0.5,
        layout_iterations: int = 30,
    ) -> None:
        """
        Visualize the recomputable candidate graph with larger context.
        """
        from matplotlib import cm, colors as mcolors, pyplot as plt

        pos = nx.spring_layout(
            self.recomputable_node_only_graph_with_larger_graph_context,
            k=layout_k,
            iterations=layout_iterations,
        )
        # pos = nx.spectral_layout(graph_with_indirect_edges)
        plt.figure(figsize=(20, 15))

        # Create a dictionary for node labels using the index
        labels = {
            node: self.recomputable_node_only_graph_with_larger_graph_context.nodes[
                node
            ].get("index", node)
            for node in self.recomputable_node_only_graph_with_larger_graph_context.nodes
        }

        # Extract memory values and normalize them
        norm = mcolors.Normalize(
            vmin=min(self.get_knapsack_memory_input()),
            vmax=max(self.get_knapsack_memory_input()),
        )
        cmap = cm.viridis  # type: ignore[attr-defined]

        # Assign colors based on memory
        node_colors = [
            cmap(
                norm(
                    float(
                        self.recomputable_node_only_graph_with_larger_graph_context.nodes[
                            node
                        ]["memory"]
                    )
                )
            )
            for node in self.recomputable_node_only_graph_with_larger_graph_context.nodes
        ]

        # Draw the graph with parsed nodes only
        nx.draw_networkx_nodes(
            self.recomputable_node_only_graph_with_larger_graph_context,
            pos,
            node_color=node_colors,
            node_size=300,
            label="Parsed Nodes",
        )
        nx.draw_networkx_edges(
            self.recomputable_node_only_graph_with_larger_graph_context,
            pos,
            arrows=True,
            arrowsize=10,
        )
        nx.draw_networkx_labels(
            self.recomputable_node_only_graph_with_larger_graph_context,
            pos,
            labels=labels,
            font_size=8,
            font_weight="bold",
        )

        plt.title("Memory Colour Coded Dependency Graph for Recomputable Nodes")
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label="Memory")
        plt.show()
<|endoftext|>import torch


def greedy_knapsack(
    memory: list[float], runtimes: list[float], max_memory: float
) -> tuple[float, list[int], list[int]]:
    n = len(runtimes)
    items = list(range(n))

    # Sort items based on the ratio of runtime to memory in descending order
    items = sorted(items, key=lambda i: runtimes[i] / memory[i], reverse=True)

    total_memory = 0.0
    total_runtime = 0.0
    items_to_save = []
    items_to_allow_recomputing = []

    for i in items:
        if total_memory + memory[i] <= max_memory:
            total_memory += memory[i]
            total_runtime += runtimes[i]
            items_to_save.append(i)
        else:
            items_to_allow_recomputing.append(i)
    return total_runtime, items_to_save, items_to_allow_recomputing


def ilp_knapsack(
    memory: list[float], runtimes: list[float], max_memory: float
) -> tuple[float, list[int], list[int]]:
    import numpy as np

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError:
        raise RuntimeError(
            "To use the ILP for memory budget checkpointing you need to install scipy"
        ) from None

    np_memory = np.array(memory)
    np_runtimes = np.array(runtimes)
    c = -np_runtimes  # type: ignore[operator]

    memory_constraint = LinearConstraint(A=np_memory, ub=np.array(max_memory))
    constraints = [memory_constraint]

    integrality = np.ones_like(c)
    res = milp(
        c=c, constraints=constraints, integrality=integrality, bounds=Bounds(0, 1)
    )
    if not res.success:
        raise RuntimeError("Somehow scipy solving failed")

    items_to_save = []
    items_to_allow_recomputing = []
    for idx, i in enumerate(res.x):
        if i == 1:
            items_to_save.append(idx)
        else:
            items_to_allow_recomputing.append(idx)
    return -res.fun, items_to_save, items_to_allow_recomputing


def dp_knapsack(
    memory: list[float], runtime: list[float], max_memory: float
) -> tuple[float, list[int], list[int]]:
    # Scaling factor to convert floating point weights to integers
    S = 10000

    # Quantize the memory weights
    quantized_memory = torch.tensor(
        [round(m * S) for m in memory], dtype=torch.long, device="cpu"
    )
    runtimes = torch.tensor(runtime, dtype=torch.float32, device="cpu")

    # Quantized pseudopolynomial DP for 0-1 Knapsack
    quantized_max_memory = round(max_memory * S)

    n = len(memory)

    # Initialize the DP table
    # TODO(chilli): I think if needed, this memory can be optimized with sliding
    # window trick + Hirschberg trick:
    # https://codeforces.com/blog/entry/47247?#comment-316200
    dp = torch.zeros(
        (n + 1, quantized_max_memory + 1), dtype=torch.float32, device="cpu"
    )

    for i in range(1, n + 1):
        current_memory = quantized_memory[i - 1]
        current_runtime = runtimes[i - 1]

        # Copy the previous row
        dp[i, :] = dp[i - 1, :]

        # Update dp[i, j] for all j >= current_memory
        if current_memory == 0:
            dp[i, :] = dp[i - 1, :] + current_runtime
        else:
            dp[i, current_memory:] = torch.maximum(
                dp[i - 1, current_memory:],
                dp[i - 1, :-current_memory] + current_runtime,
            )

    # Backtrack to find the items included in the knapsack
    saved_items = []
    recomputable_items = []
    j: int = quantized_max_memory
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            saved_items.append(i - 1)  # Include this item (indexing from 0)
            j -= int(quantized_memory[i - 1].item())
        else:
            recomputable_items.append(i - 1)

    saved_items.reverse()  # To get items in the order they were added

    # The maximum runtime that can be achieved within the max_memory constraint
    max_runtime = dp[n][quantized_max_memory].item()

    return max_runtime, saved_items, recomputable_items


def dp_knapsack_sliding_hirschberg(
    memory: list[float], runtime: list[float], max_memory: float
) -> tuple[float, list[int], list[int]]:
    # Scaling factor to convert floating point weights to integers
    S = 10000

    # q_ prefix stands for quantized
    q_memory = [int(round(m * S)) for m in memory]
    runtimes = [float(v) for v in runtime]

    q_max_memory = int(round(max_memory * S))

    q_memory_length = len(q_memory)
    if q_memory_length == 0:
        return 0.0, [], []

    item_indices = list(range(q_memory_length))
    dp_profile_size = q_max_memory + 1

    # Current DP profile (row)
    dp_profile = torch.zeros(dp_profile_size, dtype=torch.float32, device="cpu")
    # Store a candidate for next dp_profile - current dp row + item
    candidate_profile = torch.empty(dp_profile_size, dtype=torch.float32, device="cpu")
    left_profile = torch.empty(dp_profile_size, dtype=torch.float32, device="cpu")
    right_profile = torch.empty(dp_profile_size, dtype=torch.float32, device="cpu")

    saved_items: list[int] = []
    recomputable_items: list[int] = []

    # Explicit stack to optimize memory and avoid recursion
    # Stack stores segments as (start index, end index, capacity for segment)
    stack: list[tuple[int, int, int]] = [(0, q_memory_length, q_max_memory)]

    # LIFO
    while stack:
        start, end, capacity = stack.pop()
        length = end - start
        if length == 0:
            continue

        # Leaf
        if length == 1:
            index = item_indices[start]
            memory_item = q_memory[index]
            runtime_item = runtimes[index]
            if memory_item <= capacity and runtime_item > 0.0:
                saved_items.append(index)
            else:
                recomputable_items.append(index)
            continue

        # Split the segment into two halves
        middle = start + (length // 2)
        left_start, left_end = middle, end
        right_start, right_end = start, middle

        # Assign items to both halves
        left_items = item_indices[left_start:left_end]
        right_items = item_indices[right_start:right_end]

        # Working only on items allowed by segment's capacity
        capacity = capacity + 1
        dp_view = dp_profile[:capacity]
        candidate_view = candidate_profile[:capacity]
        left_dp_local = left_profile[:capacity]
        right_dp_local = right_profile[:capacity]

        # Left part
        dp_view.zero_()
        for index in left_items:
            memory_item = q_memory[index]
            runtime_item = runtimes[index]

            if memory_item == 0:
                # Weight is 0, so add it to all capacities; a "free lunch", essentially
                dp_view.add_(runtime_item)
                continue

            # If item is too heavy, we skip it
            if memory_item >= capacity:
                continue

            # Add the current item so we can then pick the highest value
            dp_view_candidate = candidate_view[: capacity - memory_item]
            torch.add(dp_view[:-memory_item], runtime_item, out=dp_view_candidate)
            # Take the highest - either previous (without current) or with current
            torch.maximum(
                dp_view[memory_item:], dp_view_candidate, out=dp_view[memory_item:]
            )

        # Store the left profile
        left_dp_local.copy_(dp_view)

        # Right part
        dp_view.zero_()
        for index in right_items:
            memory_item = q_memory[index]
            runtime_item = runtimes[index]

            if memory_item == 0:
                dp_view.add_(runtime_item)
                continue

            if memory_item >= capacity:
                continue

            dp_view_candidate = candidate_view[: capacity - memory_item]
            torch.add(dp_view[:-memory_item], runtime_item, out=dp_view_candidate)
            torch.maximum(
                dp_view[memory_item:], dp_view_candidate, out=dp_view[memory_item:]
            )

        # Store the reversed right profile
        right_dp_local.copy_(dp_view.flip(-1))

        # In-place compute item-wise sum of left and right to pick the split point where the sum is highest
        left_dp_local.add_(right_dp_local)

        # Pick the index of highest value of a pair, which we then use as a split point
        best_split = int(torch.argmax(left_dp_local).item())

        left_capacity = best_split
        right_capacity = capacity - best_split

        # Clamp (might be removed if we're 100% sure that there is no edge case that will mess up the indices math)
        if left_capacity < 0:
            left_capacity = 0
        if right_capacity < 0:
            right_capacity = 0
        if left_capacity > q_max_memory:
            left_capacity = q_max_memory
        if right_capacity > q_max_memory:
            right_capacity = q_max_memory

        # Push right then left, so left is processed next
        stack.append((right_start, right_end, right_capacity))
        stack.append((left_start, left_end, left_capacity))

    saved_items = sorted(saved_items)
    recomputable_items = sorted(recomputable_items)

    max_runtime = sum(runtime[i] for i in saved_items)
    recomputable_items.reverse()
    return max_runtime, saved_items, recomputable_items
<|endoftext|>"""
Contains utils for logging in AOTAutograd, including managing the names of the graphs under
compilation, capturing user-friendly tracebacks, and debug messages.
"""

import collections
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.fx.traceback as fx_traceback

from .schemas import AOTConfig


# This is a list since looking forward, we can have this arbitrarily nested.
graph_being_compiled: list[str] = []
# TODO: It would be nice to reset the numbering every time aot_id goes
# up, but this is annoying to do right now (because we don't know if
# an aot_id will come back from the dead), so right now this also happens
# to be a globally unique number too (at the cost of wobbling if you change
# how the graphs compile)
nth_graph: int = 0
model_name: str = "model"


def set_model_name(name: str) -> None:
    global model_name
    model_name = name


def get_aot_compilation_context() -> tuple[list[str], str, int]:
    return list(graph_being_compiled), model_name, nth_graph


def get_aot_graph_name() -> str:
    """
    Returns the name of the graph being compiled.
    """
    global model_name, graph_being_compiled, nth_graph
    return f"{model_name}__{'_'.join(graph_being_compiled)}_{nth_graph}"


get_graph_being_compiled = get_aot_graph_name


@contextmanager
def track_graph_compiling(
    aot_config: AOTConfig, graph_name: str
) -> Generator[None, None, None]:
    global graph_being_compiled
    # TODO: Don't shove the aot_id in here; set it in the context
    graph_being_compiled = [f"{aot_config.aot_id}_{graph_name}"]
    old_name = None
    if tracing_context := torch._guards.TracingContext.try_get():
        old_name = tracing_context.aot_graph_name
        tracing_context.aot_graph_name = graph_being_compiled
        has_tracing_context = True
    else:
        has_tracing_context = False
    try:
        yield
    finally:
        global nth_graph
        nth_graph += 1
        graph_being_compiled = []
        if has_tracing_context:
            if tracing_context := torch._guards.TracingContext.try_get():
                tracing_context.aot_graph_name = old_name


# Set up hooks so that during backward the fx's stack_trace is properly set
callback_set = False


def setup_stacktrace_preservation_hooks(roots: list[torch.autograd.graph.Node]) -> None:
    def iter_graph(
        roots: list[torch.autograd.graph.Node],
    ) -> Iterator[torch.autograd.graph.Node]:
        if not roots:
            return
        seen = set()
        q = collections.deque()
        for node in roots:
            if node is not None and node not in seen:
                seen.add(node)
                q.append(node)

        while q:
            node = q.popleft()
            for fn, _idx in node.next_functions:
                if fn in seen or fn is None:
                    continue
                seen.add(fn)
                q.append(fn)

            yield node

    def get_callback(saved_stack_: list[str]) -> Callable[[], None]:
        def callback() -> None:
            global callback_set
            fx_traceback.set_stack_trace(saved_stack_)
            callback_set = False

        return callback

    def get_prehook(stack_: list[str], seq_nr: int) -> Callable[[Any], None]:
        def prehook(grad_output: Any) -> None:
            global callback_set

            if not callback_set:
                torch.autograd.variable.Variable._execution_engine.queue_callback(  # type: ignore[attr-defined]
                    get_callback(fx_traceback.format_stack())
                )
                callback_set = True

            fx_traceback.set_stack_trace(stack_)
            fx_traceback.set_grad_fn_seq_nr(seq_nr)
            fx_traceback._mark_autograd_backward()

        return prehook

    def get_posthook(
        special_stack_: list[str], seq_nr: int
    ) -> Callable[[Any, Any], None]:
        def posthook(grad_input: Any, grad_output: Any) -> None:
            fx_traceback.set_stack_trace(special_stack_)
            fx_traceback.reset_grad_fn_seq_nr()
            fx_traceback._reset_autograd_backward()

        return posthook

    for node in iter_graph(roots):
        # pyrefly: ignore[missing-attribute]
        forward_node_stack = node.metadata.get("traceback_", [])
        node.register_prehook(get_prehook(forward_node_stack, node._sequence_nr()))

        special_stack = forward_node_stack.copy()
        special_stack.append(fx_traceback.GRADIENT_ACC_SPECIAL_STACK)
        node.register_hook(get_posthook(special_stack, node._sequence_nr()))


def setup_stacktrace_preservation_hooks_from_tensors(outputs: Any) -> None:
    roots = [
        t.grad_fn
        for t in (outputs if isinstance(outputs, (list, tuple)) else (outputs,))
        if isinstance(t, torch.Tensor) and t.grad_fn is not None
    ]
    if roots:
        setup_stacktrace_preservation_hooks(roots)


def describe_input(i: int, aot_config: AOTConfig) -> str:
    if i < aot_config.num_params_buffers:
        return f"parameter/buffer {i}"
    else:
        return f"input {i - aot_config.num_params_buffers}"


def format_guard_bug_msg(aot_config: AOTConfig, expected: str) -> str:
    return (
        f"At compilation time, graph {aot_config.aot_id} was compiled under the "
        f"assumption that {expected}, but at runtime this was not the case.  "
        "This indicates a guard bug in AOTAutograd or Dynamo, please file a bug to PyTorch."
    )
<|endoftext|>"""
Graph pass to decompose complex-valued operations into real-valued operations.

This pass runs AFTER functionalization, so the graph is purely functional (no
mutations, no aliasing). It retraces the graph with ComplexTensor dispatch active,
which decomposes each complex op into real-valued sub-ops. The graph signature
(input/output count and dtypes) is preserved: complex inputs get unpacked via
aten.real/aten.imag at the top, and complex outputs get repacked via aten.complex
at the bottom.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
import torch.fx as fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import Tensor
from torch._subclasses.complex_tensor import ComplexTensor, WrapComplexMode
from torch.fx.experimental.proxy_tensor import make_fx


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch._ops import OpOverload


def _can_wrap(value: Any) -> bool:
    return (
        isinstance(value, Tensor)
        and value.dtype.is_complex
        and not isinstance(value, ComplexTensor)
    )


def _maybe_wrap(value: Any) -> Any:
    if _can_wrap(value):
        return ComplexTensor(torch.real(value), torch.imag(value))
    return value


def _maybe_unwrap(value: Any) -> Any:
    if isinstance(value, ComplexTensor):
        return value.as_interleaved()
    return value


def _has_complex(gm: fx.GraphModule, flat_args: list[Any]) -> bool:
    for arg in flat_args:
        if _can_wrap(arg):
            return True
    for node in gm.graph.nodes:
        if node.op == "call_function":
            val = node.meta.get("val")
            if _can_wrap(val):
                return True
    return False


def _aliased_input_indices(op: torch._ops.OpOverload) -> list[int]:
    schema = op._schema
    ret_aliases = set()
    idxs = []

    for ret in schema.returns:
        ret_alias = ret.alias_info
        if ret_alias is None:
            continue
        ret_aliases.update(ret_alias.before_set)

    if len(ret_aliases) == 0:
        return idxs

    for i, arg in enumerate(schema.arguments):
        if arg.alias_info is None:
            continue
        if not arg.alias_info.before_set.isdisjoint(ret_aliases):
            idxs.append(i)

    return idxs


def _collect_storage_aliases(node: fx.Node) -> set[fx.Node]:
    seen: set[fx.Node] = set()
    stack: list[fx.Node] = [node]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if not isinstance(n.target, torch._ops.OpOverload):
            continue
        idxs = _aliased_input_indices(n.target)
        for idx in idxs:
            src = n.args[idx]
            if isinstance(src, fx.Node):
                stack.append(src)
    return seen


def _assert_no_incorrect_aliasing_mutation(gm: fx.GraphModule) -> None:
    """
    In this function, we check that the compiled graph respects aliasing semantics.

    There are three cases in which this check fires:
        1. A complex tensor is mutated in place via an incorrect alias.
           The ops that produce incorrect aliases are listed in `INCORRECT_ALIASING_OPS`.
        2. A complex input is mutated in place. This is not allowed because complex inputs
           decompose to real and imaginary parts, which are made contiguous for performance.
           Therefore, any complex input passed in does not alias the original tensor.
        3. A complex input aliases an output. This is not allowed as the input and output
           tensors can't alias each other because of 2.

    In all three cases, we raise an error if possible incorrectness is detected.
    """
    for n in gm.graph.nodes:
        if (
            isinstance(n.target, torch._ops.OpOverload)
            and n.target.overloadpacket == torch.ops.aten.copy_
            and n.args[0].meta["val"].dtype.is_complex
        ):
            raise RuntimeError(
                "Mutating a complex tensor in place is not allowed in a `torch.compile` region."
            )


def decompose_complex_in_graph(
    gm: fx.GraphModule,
    flat_args: list[Any],
    decompositions: Mapping[OpOverload, Callable[..., Any]] | None = None,
) -> fx.GraphModule:
    if not _has_complex(gm, flat_args):
        return gm

    _assert_no_incorrect_aliasing_mutation(gm)

    def wrapper(*args: Any) -> Any:
        wrapped = tuple(_maybe_wrap(a) for a in args)
        with WrapComplexMode():
            result = fx.Interpreter(gm).run(*wrapped)
        return pytree.tree_map(_maybe_unwrap, result)

    with fx_traceback.preserve_node_meta():
        return make_fx(wrapper, decomposition_table=decompositions)(*flat_args)
<|endoftext|>from collections.abc import Iterator, MutableMapping
from typing import Generic, TypeVar


K = TypeVar("K")
V = TypeVar("V")


# Used for fast next key access (using the fact that the dict is ordered)
# Note: doesn't support deletion but we don't need it!
class IndexedDict(MutableMapping[K, V], Generic[K, V]):
    """A dict that maintains insertion order with O(1) index access."""

    __slots__ = ("_dict", "_keys", "_key_to_index")

    def __init__(self) -> None:
        self._dict: dict[K, V] = {}
        self._keys: list[K] = []  # typing: ignore[bad-override]
        self._key_to_index: dict[K, int] = {}

    def __setitem__(self, key: K, value: V) -> None:
        if key not in self._dict:
            self._key_to_index[key] = len(self._keys)
            self._keys.append(key)
        self._dict[key] = value

    def __getitem__(self, key: K) -> V:
        return self._dict[key]

    def __delitem__(self, key: K) -> None:
        raise NotImplementedError("Deletion not supported for IndexedDict")

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[K]:
        return iter(self._keys)

    def __contains__(self, key: object) -> bool:
        return key in self._dict

    def next_key(self, key: K) -> K | None:
        """Get the next key in insertion order. O(1)."""
        idx = self._key_to_index.get(key)
        if idx is not None and idx + 1 < len(self._keys):
            return self._keys[idx + 1]
        return None

    def prev_key(self, key: K) -> K | None:
        """Get the previous key in insertion order. O(1)."""
        idx = self._key_to_index.get(key)
        if idx is not None and idx > 0:
            return self._keys[idx - 1]
        return None
<|endoftext|>"""
Generic codegen for AOTAutograd runtime wrappers.

PySourceBuilder accumulates indented Python source together with the globals the
generated code closes over and a fresh-name counter; _compile_and_exec_source is
the single chokepoint that turns generated source into a live function (logging it
as a structured-trace artifact for tlparse and giving it a stable traceback
filename). Every runtime-wrapper codegen routes through it -- the subclass wrappers
in subclass_codegen.py and the orchestration / alias / mutation epilogues in
runtime_wrappers.py -- so it knows nothing about any particular wrapper kind. This
module also hosts the optional thread-local capture sink that records each codegen'd
wrapper's source, which AOT-to-Python lowering (to_standalone_python.py) uses to
compose the wrappers into one standalone module. Kept as a leaf module (stdlib +
torch only) so it is safe to import anywhere.
"""

import contextlib
import functools
import logging
import threading
from collections.abc import Callable, Iterator

import torch
from torch.utils._indented_buffer import IndentedBuffer


log = logging.getLogger(__name__)


class PySourceBuilder(IndentedBuffer):
    """Builds indented Python source for compile/exec, along with the globals
    the generated code closes over and a monotonic fresh-name counter.

    Inherits the indentation primitive (``writeline``/``indent()``) from
    IndentedBuffer: body lines are written WITHOUT leading whitespace and
    indentation is managed by the ``indent()`` context manager so call sites
    read as plain code. This subclass adds the exec layer (globals binding,
    fresh-name generation, and ``build()`` via _compile_and_exec_source). Pass
    ``fn_name``/``artifact_name`` to emit a ``def`` header and enable
    ``build()``.
    """

    def __init__(
        self,
        fn_name: str | None = None,
        *,
        args: str = "args",
        artifact_name: str | None = None,
    ) -> None:
        super().__init__()
        self.globals: dict[str, object] = {}
        self._name_counter: int = 0
        self._fn_name = fn_name
        self._artifact_name = artifact_name
        if fn_name is not None:
            self.writeline(f"def {fn_name}({args}):")

    def emit(self, line: str, indent: int = 1) -> None:
        """Append one line at an explicit absolute indent level.

        Convenient for recursive generators that thread an indent depth instead
        of nesting indent() context managers.
        """
        saved, self._indent = self._indent, indent
        try:
            self.writeline(line)
        finally:
            self._indent = saved

    def fresh_name(self, prefix: str) -> str:
        name = f"{prefix}_{self._name_counter}"
        self._name_counter += 1
        return name

    def add_global(self, name: str, value: object) -> str:
        """Bind a live object into the exec globals under ``name``, by reference."""
        self.globals[name] = value
        return name

    def bind(self, **values: object) -> None:
        """Bind live objects into the exec globals by keyword, by reference."""
        self.globals.update(values)

    def bind_value(self, prefix: str, value: object) -> str:
        """Bind a value under a fresh unique name and return that name."""
        return self.add_global(self.fresh_name(prefix), value)

    def getvalue(self) -> str:
        # Join without a trailing newline (unlike IndentedBuffer.getvalue).
        if not all(isinstance(li, str) for li in self._lines):
            raise AssertionError(f"expected all lines to be str, got {self._lines}")
        return "\n".join(self._lines)  # type: ignore[arg-type]

    def build(
        self, *, wrapped_fn: Callable[..., object] | None = None
    ) -> Callable[..., object]:
        if self._fn_name is None or self._artifact_name is None:
            raise AssertionError("build() requires fn_name and artifact_name")
        return _compile_and_exec_source(
            self.getvalue(),
            self.globals,
            self._fn_name,
            self._artifact_name,
            wrapped_fn=wrapped_fn,
        )


# Optional sink for the source of every runtime-wrapper function codegen'd via
# ``_compile_and_exec_source``. When a sink is installed (see
# ``capture_generated_sources``), each codegen'd wrapper appends a
# ``GeneratedSource`` to it; this lets AOT-to-Python lowering compose the wrappers
# into one standalone module. Thread-local (NOT a process global) so a concurrent
# compile on another thread cannot splice its wrappers into this capture -- and,
# conversely, so an unrelated concurrent compile on another thread is never mistaken for
# offloaded capture work and aborted (``_compile_and_exec_source`` runs on EVERY
# AOTAutograd compile, not just this one). Absent (zero overhead, no behavior change)
# during ordinary compilation. Mirrors the threading.local used by
# _saved_tensor_hook_context in graph_compile.py.
_capture_tls = threading.local()


def _current_capture_sink() -> "list[GeneratedSource] | None":
    return getattr(_capture_tls, "sink", None)


class GeneratedSource:
    """One codegen'd runtime-wrapper function: its source, the exec'd function
    object, and the globals it closes over (which include the inner ``compiled_fn``
    it chains to, plus any baked metadata). The function object lets a composer wire
    cross-wrapper references by identity. Recorded by ``_compile_and_exec_source``
    when capture is active.

    ``globals_dict`` is a PRE-EXEC snapshot of the declared closure globals: it does
    NOT contain the interpreter ``__builtins__`` that exec() would otherwise inject,
    so the standalone composer can reconstruct every entry as source without having
    to special-case ``__builtins__``."""

    def __init__(
        self,
        artifact_name: str,
        fn_name: str,
        source: str,
        globals_dict: dict[str, object],
        fn: object,
        origin_id: int | None = None,
    ) -> None:
        self.artifact_name = artifact_name
        self.fn_name = fn_name
        self.source = source
        self.globals_dict = globals_dict
        self.fn = fn
        # Identity of the TracingContext this wrapper was codegen'd under. The capture
        # sink is duration-scoped over one inductor compile, so a re-entrant on-thread
        # lowering that codegen's into THIS sink during that window would append its
        # wrappers too; the composer filters by this id, which separates such a foreign
        # lowering when it ran under a DISTINCT TracingContext. A same-context re-entrant
        # lowering shares this id and is caught instead by the composer's
        # orchestration-count guard.
        self.origin_id = origin_id


@contextlib.contextmanager
def capture_generated_sources(into: "list[GeneratedSource]") -> "Iterator[None]":
    """Within this context, record every codegen'd runtime-wrapper function's source
    into ``into`` (in codegen order). A no-op when not entered.

    THREADING: the sink is thread-local, so every wrapper captured into ``into`` must be
    codegen'd ON THIS THREAD. There is deliberately NO cross-thread tripwire here: a
    process-global owner check would abort an unrelated concurrent compile on another
    thread, since ``_compile_and_exec_source`` runs on every AOTAutograd compile and is not
    serialized against this capture. If a future change offloads wrapper codegen to a
    worker thread, those wrappers are simply not captured; a forward that loses its
    orchestration wrapper this way is rejected by the composer (it requires exactly one),
    so the common case fails loudly rather than emitting an empty module.
    """
    prev = _current_capture_sink()
    _capture_tls.sink = into
    try:
        yield
    finally:
        _capture_tls.sink = prev


def _compile_and_exec_source(
    source: str,
    globals_dict: dict[str, object],
    fn_name: str,
    artifact_name: str,
    wrapped_fn: Callable[..., object] | None = None,
) -> Callable[..., object]:
    """Compile generated source, exec it, and return the named function.

    If wrapped_fn is provided, applies functools.update_wrapper so that
    __wrapped__ and __dict__ (e.g. _fx_graph_cache_key) propagate to the
    generated function.
    """
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Generated %s:\n%s", artifact_name, source)

    torch._logging.trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": artifact_name,
            "encoding": "string",
        },
        payload_fn=lambda: source,
    )

    # Use a path under torch/_functorch/ so the code object is recognized by
    # dynamo's MOD_SKIPLIST. The eval frame hook stays active during the entire
    # torch.compile(fn)(*args) call (to handle graph breaks and resume functions),
    # so codegen'd functions called during backward get intercepted even though
    # no tracing is active. A real path makes them skip automatically.
    code = compile(source, f"{__file__}:codegen({artifact_name})", "exec")
    local_dict: dict[str, object] = {}
    # exec() mutates ``globals_dict`` in place, injecting ``__builtins__`` (and any
    # name the source binds at module scope). When capturing, snapshot the declared
    # closed-over names BEFORE that happens so the captured GeneratedSource holds only
    # the intended closure globals -- not the post-exec dict with the interpreter's
    # ``__builtins__`` -- which is what the standalone composer reconstructs as source.
    sink = _current_capture_sink()
    captured_globals = dict(globals_dict) if sink is not None else globals_dict
    exec(code, globals_dict, local_dict)
    fn = local_dict[fn_name]
    if wrapped_fn is not None:
        functools.update_wrapper(fn, wrapped_fn)  # type: ignore[arg-type]

    if sink is not None:
        # Tag with the current TracingContext identity so the composer can drop any
        # wrapper a re-entrant lowering appended during the capture window (see the
        # origin_id note on GeneratedSource).
        ctx = torch._guards.TracingContext.try_get()
        origin_id = id(ctx) if ctx is not None else None
        sink.append(
            GeneratedSource(
                artifact_name, fn_name, source, captured_globals, fn, origin_id
            )
        )

    return fn  # type: ignore[return-value]
<|endoftext|>"""
This module dispatches the graphs to either the forward-only or joint compilation
pathways, taking into account the AOTConfig and the collected ViewAndMutationMetadata.
"""

import contextlib
import dataclasses
from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as pytree
import torch.utils.dlpack
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import detect_fake_mode, lazy_format_graph_code
from torch._logging import getArtifactLogger, trace_structured
from torch._subclasses.functional_tensor import FunctionalTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torchgen.utils import dataclass_repr

from .. import config
from .descriptors import AOTInput, BackwardTokenAOTInput
from .functional_utils import (
    assert_functional_graph,
    propagate_input_mutation_stacktraces,
)
from .graph_capture_wrappers import (
    aot_dispatch_subclass,
    create_functionalized_fn,
    create_joint,
    fn_input_mutations_to_outputs,
    fn_prepped_for_autograd,
    handle_effect_tokens_fn,
)
from .schemas import AOTConfig, FxValue, SubclassMeta, TraceFn, ViewAndMutationMeta
from .streams import (
    assign_backward_streams,
    assign_epilogue_copy_streams,
    insert_backward_syncs,
    populate_fw_metadata_with_stream_indices,
    sync_deallocations,
    wrap_all_sync_nodes_with_control_deps,
)
from .utils import (
    call_and_expect_output_descs,
    copy_fwd_metadata_to_bw_nodes,
    fn_wrappers,
    register_buffer_assignment_hook,
    root_module_when_exporting_non_strict,
    simple_wraps,
    unlift_tokens,
)


aot_graphs_log = getArtifactLogger(__name__, "aot_graphs")


def _extract_tangent_source_stack_traces(
    fx_g: torch.fx.GraphModule,
    fw_metadata: ViewAndMutationMeta,
) -> None:
    from .descriptors import PlainAOTOutput, TangentAOTInput

    if not fw_metadata.traced_tangents_descs:
        return

    output_node = list(fx_g.graph.nodes)[-1]
    all_outputs = output_node.args[0]

    stack_traces: list[str | None] = []
    got_one = False

    for desc in fw_metadata.traced_tangents_descs:
        stack_trace = None

        if isinstance(desc, TangentAOTInput):
            output_desc = desc.output
            if isinstance(output_desc, PlainAOTOutput) and output_desc.idx < len(
                all_outputs
            ):
                output_arg = all_outputs[output_desc.idx]
                if isinstance(output_arg, torch.fx.Node):
                    stack_trace = output_arg.meta.get("stack_trace", None)
                    got_one = True

        stack_traces.append(stack_trace)

    if got_one:
        fw_metadata.tangent_source_stack_traces = stack_traces


def _create_graph(
    f: Callable[..., Any],
    args: list[torch.Tensor],
    args_descs: list[AOTInput]
    | None = None,  # keep compat with old clients; maybe we should split into two impls
    *,
    aot_config: AOTConfig,
) -> torch.fx.GraphModule:
    # FunctionalTensorMode must be enabled here.
    # See Note [Accessing .grad_fn on FunctionalTensor]
    out_descs = None

    if args_descs is None:
        inner_f = f
    else:

        @simple_wraps(f)
        def inner_f(*args: Any) -> Any:
            nonlocal out_descs
            if out_descs is not None:
                raise AssertionError("out_descs must be None")
            out, out_descs = call_and_expect_output_descs(f, args)
            return out

    if aot_config.disable_functionalization:
        ctx = contextlib.nullcontext()
    else:
        ctx = FunctionalTensorMode(  # type: ignore[assignment]
            pre_dispatch=aot_config.pre_dispatch,
            export=aot_config.is_export,
            # Allow token discovery for joint fn tracing as tokens can be used in backward.
            _allow_token_discovery=True,
            _keep_input_mutations=aot_config.keep_inference_input_mutations,
        )

    with (
        enable_python_dispatcher(),
        ctx,
        torch._dynamo.eval_frame._use_eager_on_nested_compile(),
    ):
        fx_g = make_fx(
            inner_f,
            decomposition_table=aot_config.decompositions,
            record_module_stack=True,
            pre_dispatch=aot_config.pre_dispatch,
            _disable_torch_fn_metadata_mode=aot_config._disable_torch_fn_metadata_mode,
        )(*args)

        if args_descs is not None:
            flat_args_descs, _ = pytree.tree_flatten(args_descs)
            flat_out_descs, _ = pytree.tree_flatten(out_descs)

            # Unfortunately, flat_args_descs is not guaranteed to match the
            # number of actual arguments that show up on the FX graph.
            # Specifically, allow_token_discovery=True means that we will
            # silently add extra token arguments to the backwards graph.
            #
            # Although there are a few ways to detect what these tokens are,
            # we are going to settle for something dodgy but simple to
            # implement: match tangents_token placeholders specifically,
            # as these are the only placeholders that are created by token
            # discovery (NB: there is NO other code that treats this name
            # as load bearing, so this is a bit naughty!)
            #
            # I originally wanted to detect tokens in exactly the same way
            # that they are detected at normal runtime, but to be honest
            # the normal runtime detection is pretty strange: it seems the
            # backward tokens are not reliably at the end of the argument list
            # but *precede* the RNG arguments (I don't understand why this is
            # the case).  And in unlift_tokens, token arguments are detected
            # by seeing if they feed into an effects call!  Dastardly.  Why
            # didn't we just introduce a new type.

            i = 0
            j = 0
            for n in fx_g.graph.nodes:
                if n.op == "placeholder":
                    if n.name.startswith("tangents_token"):
                        n.meta["desc"] = BackwardTokenAOTInput(j)
                        j += 1
                    else:
                        if i >= len(flat_args_descs):
                            raise AssertionError(
                                f"i={i} >= len(flat_args_descs)={len(flat_args_descs)}: "
                                f"fn_wrappers={fn_wrappers(inner_f)}, "
                                f"placeholders={[n for n in fx_g.graph.nodes if n.op == 'placeholder']}"
                            )
                        n.meta["desc"] = flat_args_descs[i]
                        i += 1
                elif n.op == "output":
                    n.meta["desc"] = flat_out_descs

    return fx_g


# TODO: Refactor the following code so detach() persists item_memo
def _detach_and_copy_item_memo(t: torch.Tensor) -> torch.Tensor:
    detached_t = t.detach()
    if hasattr(t, "item_memo"):
        # pyrefly: ignore[missing-attribute]
        detached_t.item_memo = t.item_memo
    return detached_t


@dataclasses.dataclass
class _GraphCaptureTracingResult:
    fn_to_trace: Callable[..., Any]
    flat_args: Any
    flat_args_descs: Any
    maybe_subclass_meta: SubclassMeta | None


def _detach_traced_inputs(flat_args: Any) -> Any:
    if detect_fake_mode():
        detach_tensor = _detach_and_copy_item_memo
    else:

        def detach_tensor(t: torch.Tensor) -> torch.Tensor:
            return t.detach()

    return pytree.tree_map_only(torch.Tensor, detach_tensor, flat_args)


def _prepare_graph_capture_tracing(
    fn_to_trace: Callable[..., Any],
    flat_args: Any,
    flat_args_descs: Any,
    flat_fn: TraceFn,
    *,
    fw_metadata: ViewAndMutationMeta,
    aot_config: AOTConfig,
    trace_joint: bool,
    joint_fn_handle: Any | None = None,
) -> _GraphCaptureTracingResult:
    if aot_config.disable_functionalization:
        updated_flat_args, updated_flat_args_descs = flat_args, flat_args_descs
    else:
        fn_to_trace, updated_flat_args, updated_flat_args_descs = (
            create_functionalized_fn(
                fn_to_trace,
                flat_args,
                flat_args_descs,
                meta=fw_metadata,
                aot_config=aot_config,
                trace_joint=trace_joint,
                joint_fn_handle=joint_fn_handle,
            )
        )

    subclass_tracing_info = aot_dispatch_subclass(
        fn_to_trace,
        updated_flat_args,
        updated_flat_args_descs,
        is_joint_structure=trace_joint,
        meta=fw_metadata,
        fw_only=flat_fn,
    )
    fn_to_trace = subclass_tracing_info.plain_tensor_trace_fn
    updated_flat_args = subclass_tracing_info.plain_tensor_args
    updated_flat_args_descs = subclass_tracing_info.plain_tensor_args_descs

    if not aot_config.disable_functionalization:
        fn_to_trace, updated_flat_args, updated_flat_args_descs = (
            handle_effect_tokens_fn(
                fn_to_trace,
                updated_flat_args,
                updated_flat_args_descs,
                meta=fw_metadata,
                trace_joint=trace_joint,
            )
        )
    return _GraphCaptureTracingResult(
        fn_to_trace=fn_to_trace,
        flat_args=updated_flat_args,
        flat_args_descs=updated_flat_args_descs,
        maybe_subclass_meta=subclass_tracing_info.maybe_subclass_meta,
    )


def _create_graph_and_save_traced_inputs(
    fn_to_trace: Callable[..., Any],
    flat_args: Any,
    flat_args_descs: Any,
    *,
    aot_config: AOTConfig,
) -> tuple[torch.fx.GraphModule, Any]:
    saved_flat_args = _detach_traced_inputs(flat_args)
    return (
        _create_graph(fn_to_trace, flat_args, flat_args_descs, aot_config=aot_config),
        saved_flat_args,
    )


def aot_dispatch_base_graph(
    flat_fn: TraceFn,
    flat_args: list[FxValue],
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> tuple[torch.fx.GraphModule, list[FxValue], list[AOTInput], SubclassMeta | None]:
    # aot_dispatch_base requires functionalization, but doesn't need to handle as many cases as the autograd case.
    # The cases that aot_dispatch_base doesn't need to handle include:
    # - outputs that are aliases of graph intermediates
    # - outputs that are aliases of graph inputs
    # While cases that it does need to handle include:
    # - input mutations (including when inputs are aliases of each other)
    # - input metadata mutations
    fn_to_trace = fn_input_mutations_to_outputs(
        flat_fn,
        flat_args_descs,
        fw_metadata,
        keep_data_input_mutations=aot_config.keep_inference_input_mutations,
    )
    # TODO: replace with AOTDispatchSubclassWrapper once we refactor
    # fn_input_mutations_to_outputs and create_functionalized_fn
    # into CompilerWrappers.
    tracing_state = _prepare_graph_capture_tracing(
        fn_to_trace,
        flat_args,
        flat_args_descs,
        flat_fn,
        fw_metadata=fw_metadata,
        aot_config=aot_config,
        trace_joint=False,
    )
    fn_to_trace = tracing_state.fn_to_trace
    updated_flat_args_subclasses_desugared = tracing_state.flat_args
    updated_flat_args_subclasses_desugared_descs = tracing_state.flat_args_descs
    maybe_subclass_meta = tracing_state.maybe_subclass_meta

    aot_graphs_log.debug(
        "aot_config id: %s, fw_metadata=%s,subclass_metadata=%s",
        aot_config.aot_id,
        fw_metadata,
        maybe_subclass_meta,
    )

    # We track buffer assignments when exporting in non-strict mode.
    # (In contrast, strict mode errors on any attribute assignment.)
    mod_when_exporting_non_strict = root_module_when_exporting_non_strict(flat_fn)
    assigned_buffers: dict[str, str] = {}
    hook = None
    if aot_config.is_export and mod_when_exporting_non_strict is not None:
        # For any buffer that is assigned, we want to associate it to the final proxy node
        # that it is assigned to. This node can then be added as a buffer mutation output.
        hook = register_buffer_assignment_hook(
            mod_when_exporting_non_strict, assigned_buffers
        )

    try:
        (
            fw_module,
            saved_updated_flat_args_subclasses_desugared,
        ) = _create_graph_and_save_traced_inputs(
            fn_to_trace,
            updated_flat_args_subclasses_desugared,
            updated_flat_args_subclasses_desugared_descs,
            aot_config=aot_config,
        )
    finally:
        if hook is not None:
            hook.remove()
            hook = None
    saved_updated_flat_args_subclasses_desugared_descs = (
        updated_flat_args_subclasses_desugared_descs
    )

    if aot_config.is_export and mod_when_exporting_non_strict is not None:
        # We update metadata to consider any assigned buffers as buffer mutations.
        i = len(dict(mod_when_exporting_non_strict.named_parameters()))
        for name, _ in mod_when_exporting_non_strict.named_buffers():
            if name in assigned_buffers and not fw_metadata.input_info[i].mutates_data:
                fw_metadata.input_info[i] = dataclasses.replace(
                    fw_metadata.input_info[i], mutates_data=True
                )
                fw_metadata.num_mutated_inp_runtime_indices += 1
            i += 1

        # We add nodes corresponding to buffer assignments as output nodes in the graph.
        add_nodes = []
        output_node = list(fw_module.graph.nodes)[-1]
        for name in assigned_buffers.values():
            for node in fw_module.graph.nodes:
                if node.name == name:
                    add_nodes.append(node)
                    node.users[output_node] = None
        output_node.args = ((*add_nodes, *output_node.args[0]),)

    # As long as we opted to remove input mutations, then
    # there should be *NO* mutating ops in the graph at this point.
    if not aot_config.disable_functionalization:
        copy_count = assert_functional_graph(fw_module.graph)
        assign_epilogue_copy_streams(fw_module)
        # Wrap sync nodes with control_deps to prevent reordering
        wrap_all_sync_nodes_with_control_deps(fw_module)
        # Populate fw_metadata with stream indices from the compiled graph
        populate_fw_metadata_with_stream_indices(fw_module, fw_metadata)
        fw_module.graph.eliminate_dead_code()
        fw_module.recompile()
        copy_count2 = assert_functional_graph(fw_module.graph)
        propagate_input_mutation_stacktraces(fw_module.graph)
        if copy_count != copy_count2:
            raise AssertionError(
                f"copy_count={copy_count} != copy_count2={copy_count2}"
            )
    else:
        fw_module.graph.eliminate_dead_code()

    # See Note [Side-Effectful Tokens in AOTAutograd]
    num_tokens = len(fw_metadata.tokens)
    if num_tokens != 0 and config.unlift_effect_tokens:
        unlift_tokens(fw_module, fw_metadata, aot_config)
        saved_updated_flat_args_subclasses_desugared = (
            saved_updated_flat_args_subclasses_desugared[num_tokens:]
        )
        saved_updated_flat_args_subclasses_desugared_descs = (
            saved_updated_flat_args_subclasses_desugared_descs[num_tokens:]
        )

    if aot_config.enable_log:
        aot_graphs_log.info(
            "%s",
            lazy_format_graph_code(
                "Forward graph",
                fw_module,
                aot_config.aot_id,
                include_stride=True,
                include_device=True,
                colored=True,
                # For more expanded output set this to True (but can't default
                # to this because it affects tests):
                expanded_def=False,
            ),
        )

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "aot_forward_graph_fw_metadata",
                "encoding": "string",
            },
            payload_fn=lambda: dataclass_repr(fw_metadata),
        )
        if maybe_subclass_meta is not None:
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "aot_forward_graph_fw_subclass_metadata",
                    "encoding": "string",
                },
                payload_fn=lambda: dataclass_repr(maybe_subclass_meta),
            )

        trace_structured(
            "aot_inference_graph",
            payload_fn=lambda: fw_module.print_readable(
                print_output=False,
                include_stride=True,
                include_device=True,
                expanded_def=True,
            ),
        )

    # TODO: should factor this into a separate function for export that always only returns just the graph.
    if aot_config.is_export and maybe_subclass_meta is not None:
        raise AssertionError(
            "aot_export_module does not support tensor subclass inputs for now."
        )
    return (
        fw_module,
        saved_updated_flat_args_subclasses_desugared,
        saved_updated_flat_args_subclasses_desugared_descs,
        maybe_subclass_meta,
    )


# Has the precondition that there
# are no duplicate arguments in flat_args (e.g., the same Tensor
# object never shows up twice.  However, two tensor inputs MAY alias
# the same storage, so long as they have separate TensorImpls.)
def aot_dispatch_autograd_graph(
    flat_fn: TraceFn,
    flat_args: list[Any],
    flat_args_descs: list[AOTInput],
    aot_config: AOTConfig,
    *,
    fw_metadata: ViewAndMutationMeta,
) -> tuple[
    torch.fx.GraphModule,
    tuple[list[Any], list[Any]],
    tuple[list[AOTInput], list[AOTInput]],
    SubclassMeta | None,
]:
    # NB: flat_fn here is the original user function (as far as
    # aot_module_simplified is concerned)

    # traced_tangents corresponds to the set of outputs in the traced forward that should get grad_outputs in the traced backward.
    # It includes outputs of the original forward, *and* any updated inputs due to input mutations.
    # However, it does *not* include any outputs that are aliases of inputs or intermediates, or any metadata-only input mutations.
    joint_inputs = (flat_args, fw_metadata.traced_tangents)
    joint_inputs_descs = (flat_args_descs, fw_metadata.traced_tangents_descs)

    fn_prepared_for_autograd = fn_prepped_for_autograd(
        flat_fn,
        flat_args_descs,
        fw_metadata,
        aot_config,
    )
    joint_fn_to_trace = create_joint(
        fn_prepared_for_autograd, flat_args_descs, aot_config=aot_config
    )
    # pyrefly: ignore[missing-attribute]
    joint_fn_handle = joint_fn_to_trace.handle
    # TODO: replace with AOTDispatchSubclassWrapper once we refactor
    # fn_input_mutations_to_outputs and create_functionalized_fn
    # into CompilerWrappers.
    tracing_state = _prepare_graph_capture_tracing(
        joint_fn_to_trace,
        joint_inputs,
        joint_inputs_descs,
        flat_fn,
        fw_metadata=fw_metadata,
        aot_config=aot_config,
        trace_joint=True,
        joint_fn_handle=joint_fn_handle,
    )
    joint_fn_to_trace = tracing_state.fn_to_trace
    updated_joint_inputs = tracing_state.flat_args
    updated_joint_inputs_descs = tracing_state.flat_args_descs
    maybe_subclass_meta = tracing_state.maybe_subclass_meta

    # When we call _create_graph, this may mutate the metadata of joint
    # inputs.  But callers are expecting to get the original joint inputs.  So
    # we make aliases of all the inputs to make sure we have a copy that
    # doesn't get modified.
    #
    # This destroys requires_grad/grad_fn information.  However, backends
    # beneath AOTAutograd are indifferent to this information, so it doesn't
    # matter.
    fx_g, saved_updated_joint_inputs = _create_graph_and_save_traced_inputs(
        joint_fn_to_trace,
        updated_joint_inputs,
        updated_joint_inputs_descs,
        aot_config=aot_config,
    )

    # Redundant with the check above, but worth having in case tracing introduced
    # a fake tensor. Unlikely.
    # See Note: [Fake Modules and AOTAutograd]
    torch._dynamo.utils.assert_no_fake_params_or_buffers(fx_g)

    # Have to copy before eliminate_dead_code otherwise the
    # fw node match might be erased
    copy_fwd_metadata_to_bw_nodes(fx_g)

    # After copying metadata, assign streams to gradient accumulation nodes
    assign_backward_streams(fx_g)

    assign_epilogue_copy_streams(fx_g)

    # Insert syncs for newly assigned backward streams
    insert_backward_syncs(fx_g)

    # Sync deallocations for tensors where the stream w/ their last usage
    # is distinct from their allocation stream
    sync_deallocations(fx_g)

    # Wrap sync nodes with control_deps to prevent reordering
    # (must be after sync_deallocations which inserts additional sync nodes)
    wrap_all_sync_nodes_with_control_deps(fx_g)

    # Populate fw_metadata with stream indices from the compiled graph
    # NB: This needs to be done after the above stream assignments
    populate_fw_metadata_with_stream_indices(fx_g, fw_metadata)

    # this helps users identify which forward output to call .detach() on.
    _extract_tangent_source_stack_traces(fx_g, fw_metadata)

    fx_g.graph.eliminate_dead_code()
    if not aot_config.disable_functionalization:
        # There should be *NO* mutating ops in the graph at this point.
        assert_functional_graph(fx_g.graph)

    fx_g.recompile()

    # TODO: in AOTAutograd, we create metadata like _indices_of_inps_to_detach to detect
    # when we need to manually detach() some inputs in the forward.
    # Higher order ops might eventually need to do the same.
    if aot_config.is_export and maybe_subclass_meta is not None:
        raise AssertionError(
            "aot_export_module does not support tensor subclass inputs for now."
        )
    return (
        fx_g,
        saved_updated_joint_inputs,
        updated_joint_inputs_descs,
        maybe_subclass_meta,
    )
<|endoftext|>"""Runtime-support surface for standalone artifacts.

Modules emitted by ``torch._functorch.aot_autograd.compile_to_python`` inline
AOTAutograd's codegen'd prelude/epilogue, which closes over a few small runtime
helpers (output-alias regeneration, etc.). Rather than have the generated code
reach into scattered AOTAutograd internals -- whose exact locations are not a
stable contract -- it imports those helpers from this one module. This is the
intentional, single dependency surface of a standalone artifact: keep it small and
stable, and update generated-artifact compatibility deliberately if it changes.
"""

# Importing ``runtime_wrappers`` directly in a fresh process pulls in a name
# (e.g. AutogradLazyBackwardCompileInfo) that is only bound once the dynamo init
# chain has run, so importing it first triggers a circular ImportError. Force the
# dynamo/aot chain to fully initialize before the ``from .runtime_wrappers``
# import below so a bare ``import torch`` artifact stays self-contained.
import torch._dynamo  # noqa: F401
from torch._prims_common import CUDARngStateHelper

# IDENTITY CONTRACT: these names MUST be plain re-exports that preserve the original
# object identity -- never wrap, decorate, or alias them (e.g. functools.wraps, a thin
# forwarding lambda, a partial). to_standalone_python._known_helper_table keys on
# id() of these exact objects to recognize a global the codegen'd wrappers close over.
# A wrapper would change id(), so the table lookup would silently miss and that global
# would route to its internal AOTAutograd location instead of this stable surface.
# The same contract covers ``CUDARngStateHelper`` (imported above for circular-import
# ordering): the table keys on id() of its ``get_torch_state_as_tuple`` /
# ``set_new_offset`` staticmethods, so it too must not be wrapped or aliased.
from .functional_utils import gen_alias_from_base
from .runtime_wrappers import (
    _unwrap_tensoralias,
    mark_dynamo_propagated_dynamic_indices,
)
from .utils import normalize_as_list


__all__ = [
    "gen_alias_from_base",
    "_unwrap_tensoralias",
    "mark_dynamo_propagated_dynamic_indices",
    "normalize_as_list",
    "CUDARngStateHelper",
]
<|endoftext|>"""
Utils for caching the outputs of AOTAutograd
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import time
import traceback
from copy import copy
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import torch
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.trace_rules import torch_non_c_binding_in_graph_functions
from torch._dynamo.utils import (
    chromium_event_log_active,
    CompileEventLogger,
    counters,
    is_safe_numpy_wrapper,
    numpy_wrapper_cache_key,
    warn_once,
)
from torch._functorch import config
from torch._inductor.codecache import (
    _ident,
    add_ephemeral_timeout_increase_for_distributed,
    AOTAUTOGRAD_CACHE_PREFIX,
    BypassFxGraphCache,
    create_cache,
    extract_tensor_metadata_for_cache_key,
    FxGraphCache,
    FxGraphCachePickler,
    FxGraphHashDetails,
    GuardedCache,
    sha256_hash,
    write_atomic,
)
from torch._inductor.custom_graph_pass import (
    CustomKnapsackSolver,
    CustomRuntimeEstimator,
)
from torch._inductor.output_code import OutputCode
from torch._inductor.remote_cache import cache_stats
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.utils import BoxedBool, should_use_remote_fx_graph_cache
from torch._library.fake_class_registry import FakeScriptObject
from torch._logging import LazyString
from torch._utils_internal import log_cache_bypass
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactRecorder,
)
from torch.fx.experimental.symbolic_shapes import guarding_hint_or_throw
from torch.fx.node import Node
from torch.fx.traceback import _get_memory_budget_annotation
from torch.utils._triton import has_triton_package

from .aot_autograd_result import (
    AOTAutogradResult,
    BundledAOTAutogradCacheArtifact,
    BundledAOTAutogradResult,
    BundledCompiledBackward,
    BundledCompiledForward,
    CompiledBackward,
    CompiledForward,
    GenericAOTAutogradResult,
    SerializedGraphModule,
)
from .runtime_wrappers import (
    CompilerWrapper,
    SerializableCompiledFunction,
    SubclassMeta,
)
from .schemas import (
    ActInputPaths,
    AOTAutogradCacheInfo,
    AOTConfig,
    CacheableAOTConfig,
    ViewAndMutationMeta,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from torch._inductor.compile_fx import _CompileFxKwargs, CompilerConfigExtra
    from torch._inductor.remote_cache import JsonDataTy, RemoteCache


log = logging.getLogger(__name__)


class BypassAOTAutogradCache(Exception):
    pass


# Used to signify when FXGraphCache missed when AOTAutogradCache uses it
class FXGraphCacheMiss(BypassAOTAutogradCache):
    pass


def should_use_remote_autograd_cache() -> bool:
    if torch.compiler.config.force_disable_caches:
        return False
    if config.enable_remote_autograd_cache is not None:
        return config.enable_remote_autograd_cache
    if not config.is_fbcode():
        return False

    if torch._utils_internal.is_fb_unit_test():
        return False

    try:
        from torch._inductor.fb.remote_cache import REMOTE_CACHE_VERSION
    except ModuleNotFoundError:
        return False

    jk_name = "pytorch/remote_cache:aot_autograd_cache_version"

    return REMOTE_CACHE_VERSION >= torch._utils_internal.justknobs_getval_int(jk_name)


def should_use_local_autograd_cache() -> bool:
    if torch.compiler.config.force_disable_caches:
        return False
    return config.enable_autograd_cache


def should_bundle_autograd_cache() -> bool:
    return config.bundled_autograd_cache or torch._dynamo.config.caching_precompile


def check_node_safe(node: Node) -> None:
    """
    Checks that the node only uses supported operators. We are starting with very
    conservative cacheability constraints, and incrementally adding more support as we expand.

    [Note: AOTAutograd Cacheability checks]
    - Our cache key is computed from the FX graph produced by Dynamo and the input example values
    - A node is "safe" if the same cache key results in a compiled artifact that has the same behavior
        (i.e, the set of inputs that go into our cache key is sufficient to distinguish its behavior)

    To accomplish this safety check, we consider the following functions to be safe:
        - Public functions under modules torch, torch.functional, and torch.nn.functional: these are
        allowed in the graph by dynamo, so we can assume they are safe to cache.
        - method calls on base tensor types
        - Any call_module that dynamo deemed safe to allow AOTAutograd to trace
        - Non callable nodes, such as placeholder, output, get_attr

    The test suite test_aot_autograd_cache.py::AOTAutogradCachePicklerTests tries its best to fully cover/specify this behavior.
    """
    SAFE_TORCH_MODULES = ("torch.functional", "torch.nn.functional")
    SAFE_TORCH_FUNCTIONS = (
        "torch.Size",
        "torch.Tensor",
        "torch.sym_int",
        "torch._sym_sqrt",
        "torch.sym_float",
        "torch.sym_sum",
        "torch.autograd.grad",
        "torch.distributed.tensor._api.from_local",
        # An autocast context manager *inside* a compiled region is traced into
        # these calls. What they do is fully determined by their arguments
        # (device type, dtype, enabled, cache_enabled), which are constants in
        # the graph and therefore part of the cache key, so a graph compiled
        # under one autocast setting can never be reused for another.
        "torch.amp.autocast_mode._enter_autocast",
        "torch.amp.autocast_mode._exit_autocast",
    )
    SAFE_NON_TORCH_FUNCTIONS = (
        "einops.einops.rearrange",
        "einops.einops.repeat",
    )

    def is_public_torch_api(target: Callable[..., Any]) -> bool:
        # Don't blindly allow private functions in the torch namespace
        is_private = target.__name__.startswith("_")

        return (
            getattr(target, "__module__", None) in SAFE_TORCH_MODULES and not is_private
        )

    def is_safe_torch_function(target: Callable[..., Any]) -> bool:
        """Allowlisted torch functions"""
        function_name = f"{target.__module__}.{target.__name__}"
        # Allow torch.autograd.function.FunctionCtx if custom autograd functions are allowed
        if function_name == "torch.autograd.function.FunctionCtx":
            return (
                torch._functorch.config.autograd_cache_allow_custom_autograd_functions
            )

        # Functions in torch_non_c_binding_in_graph_functions
        # are guaranteed to be cache safe.
        # See NOTE: [Cacheability of in-graph torch functions]
        return (
            function_name in torch_non_c_binding_in_graph_functions
            or function_name in SAFE_TORCH_FUNCTIONS
            or function_name in torch._inductor.config.unsafe_marked_cacheable_functions
        )

    def is_cacheable_function(target: Callable[..., Any]) -> bool:
        if isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
            return True
        if is_safe_numpy_wrapper(target):
            return True
        if is_public_torch_api(target):
            return True
        # Technically, FXGraphCache._check_for_hop already checks this,
        # but better to error earlier anyway
        if isinstance(target, torch._ops.HigherOrderOperator):
            return target.cacheable()
        is_builtin_fun_or_type = type(target).__name__ == "builtin_function_or_method"
        if is_builtin_fun_or_type:
            return True
        if is_safe_torch_function(target):
            return True
        function_name = f"{target.__module__}.{target.__name__}"
        if function_name in SAFE_NON_TORCH_FUNCTIONS:
            return True
        return False

    def is_tensor(target: Node) -> bool:
        # Tensors always have example values in meta field
        return "example_value" in target.meta

    # I'd love to use a match statement here, but it wasn't introduced until py3.10
    if node.op == "call_function":
        if node.meta and node.meta.get("is_wrapped", False):
            # This is fx.wrap function
            # By default we BypassAOTAutogradCache for unknown functions,
            # But if user explicitly specified cache hash - allow to cache it.
            if node.meta.get("user_cache_hash", None):
                return
        if isinstance(node.target, str):
            raise AssertionError(
                f"expected node.target to not be a string, got {node.target}"
            )
        if not is_cacheable_function(node.target):
            module = getattr(node.target, "__module__", None)
            name = getattr(node.target, "__name__", None)
            raise BypassAOTAutogradCache(
                f"Unsupported call_function target {node.target}. \n Function module: {module}, \nFunction name: {name}"
            )
    elif node.op == "call_method":
        method_name = node.target
        method_target = node.args[0]
        # Only support method calls on base tensors
        if not isinstance(method_target, Node):
            raise AssertionError(
                f"expected method_target to be Node, got {type(method_target)}"
            )
        if not is_tensor(method_target):
            module = getattr(method_target, "__module__", None)
            name = getattr(method_target, "__name__", None)
            raise BypassAOTAutogradCache(
                f"Unsupported call_method target {method_target}. \nMethod module: {module}, \nMethod name: {name}"
            )
        if (
            type(method_name) is not str
            and type(method_name).__name__ != "method_descriptor"
        ):
            raise BypassAOTAutogradCache(
                f"Unsupported call_method method {node.target}: {method_name}"
            )
    # Cache safe
    elif node.op in ("placeholder", "get_attr", "call_module", "output"):
        # Assumption today for call_module being a safe op:
        # (1) today the only call_module ops that can show up in a graph come from "built-in-nn-modules"
        # that dynamo assumes are safe to trace. If dynamo assumes they are safely to blindly trace, then
        # they should be safe to cache as well.
        # (2) in the steady-state (some time in H2?) we shouldn't see these anymore, once inline builtin nn modules by default
        # (3) We do not allow user made nn modules in the graph today, only function calls.
        pass
    else:
        raise BypassAOTAutogradCache(f"Unsupported node op {node.op}")


def check_cacheable(gm: torch.fx.GraphModule) -> None:
    """
    Checks that the graph module only uses supported operators
    """
    nodes = gm.graph.nodes
    if torch._inductor.config.freezing:
        raise BypassAOTAutogradCache("Cannot cache a graph with freezing enabled")

    if not (
        torch._inductor.config.fx_graph_cache or should_use_remote_fx_graph_cache()
    ):
        raise BypassAOTAutogradCache("FX graph cache is not enabled")

    tracing_context = torch._guards.TracingContext.try_get()
    if tracing_context and tracing_context.fakify_first_call:
        raise BypassAOTAutogradCache(
            "Won't cache a graph with fakify_first_call enabled"
        )
    for node in nodes:
        check_node_safe(node)

    # Saved tensors hooks are globally set subgraphs,
    # that are not used explicitly in the main graph.
    # They are inlined in aot_autograd graphs.
    # Subgraphs are only used for caching logic.
    if hasattr(gm, "saved_tensors_hooks_pack_0"):
        check_cacheable(gm.saved_tensors_hooks_pack_0)  # type: ignore[arg-type]
        # We have guarantee of unpack subgraph existence if pack subgraph exists
        check_cacheable(gm.saved_tensors_hooks_unpack_0)  # type: ignore[arg-type]


def _get_context_fn_cache_hash(context_fn: Callable[..., Any]) -> str | None:
    """
    Extract a cache hash from a context_fn used for selective activation checkpointing (SAC).

    The context_fn determines which ops are saved vs recomputed in the SAC region.
    Since context_fn can be an arbitrary Python function, we cannot reliably pickle
    it for cache key generation (pickle only captures the function name, not the code).

    Users must provide a stable hash by setting a `cache_hash` attribute on the context_fn.
    For functools.partial objects, set the cache_hash on the partial object itself, not on
    the underlying function.

    Returns:
        The cache hash if found
        None: If no hash is provided (caller should bypass caching)
    """
    if hasattr(context_fn, "cache_hash"):
        return context_fn.cache_hash

    return None


def _iter_graph_modules(
    gm: torch.fx.GraphModule,
) -> Generator[torch.fx.GraphModule, None, None]:
    for module in gm.modules():
        if isinstance(module, torch.fx.GraphModule):
            yield module


def _collect_context_fn_hashes(gm: torch.fx.GraphModule) -> list[str]:
    """
    Collect cache hashes from all context_fn used in SAC HOPs within the graph module.

    Returns a list of hashes. Raises BypassAOTAutogradCache if any context_fn
    lacks a cache_hash attribute.
    """
    hashes = []
    for module in _iter_graph_modules(gm):
        context_fn = module.meta.get("_checkpoint_context_fn")
        if context_fn is not None:
            cache_hash = _get_context_fn_cache_hash(context_fn)
            if cache_hash is None:
                raise BypassAOTAutogradCache(
                    "SAC context_fn does not have a cache_hash attribute. "
                    "To enable caching with selective activation checkpointing, "
                    "add a 'cache_hash' attribute to your context_fn. This can be "
                    "a string or any hashable value that uniquely identifies the checkpointing "
                    "behavior (e.g., based on source code hash and closed-over globals). "
                    "For functools.partial objects, set cache_hash on the partial itself."
                )
            hashes.append(cache_hash)
    return hashes


def _collect_wrapped_user_cache_hashes(gm: torch.fx.GraphModule) -> list[str]:
    wrapped_user_cache_hashes = []
    for node in gm.graph.nodes:
        if node.meta and node.meta.get("is_wrapped", False):
            wrapped_user_cache_hashes.append(node.meta["user_cache_hash"])
    return wrapped_user_cache_hashes


def _collect_saved_tensors_hooks_fx_wrap_cache_hashes(
    gm: torch.fx.GraphModule,
) -> tuple[list[str], list[str]]:
    if not hasattr(gm, "saved_tensors_hooks_pack_0"):
        return ([], [])

    return (
        _collect_wrapped_user_cache_hashes(
            # pyrefly: ignore[bad-argument-type]
            gm.saved_tensors_hooks_pack_0
        ),
        _collect_wrapped_user_cache_hashes(
            # pyrefly: ignore[bad-argument-type]
            gm.saved_tensors_hooks_unpack_0
        ),
    )


def _get_custom_estimator_solver_uuids(
    autograd_config: Any,
) -> tuple[object | None, object | None]:
    """
    Extract uuid values from custom runtime estimator and solver configs if they have uuid() methods.

    Returns a tuple of (runtime_estimator_uuid, solver_uuid).

    Returns None for each component if:
    - The config field value is None
    - The config field value is a string (built-in option like "flops", "greedy")

    Raises BypassAOTAutogradCache if:
    - The config field value is a raw callable without uuid() method (caching not supported)
    - The CustomRuntimeEstimator/CustomKnapsackSolver's uuid() method returns None
    (caching explicitly disabled by implementation)
    """

    runtime_estimator = getattr(
        autograd_config, "activation_memory_budget_runtime_estimator", None
    )
    solver = getattr(autograd_config, "activation_memory_budget_solver", None)

    runtime_estimator_uuid = None
    solver_uuid = None

    if isinstance(runtime_estimator, CustomRuntimeEstimator):
        runtime_estimator_uuid = runtime_estimator.uuid()
        if runtime_estimator_uuid is None:
            raise BypassAOTAutogradCache(
                "CustomRuntimeEstimator.uuid() returned None, bypassing cache"
            )
    elif callable(runtime_estimator) and not isinstance(runtime_estimator, str):
        raise BypassAOTAutogradCache(
            "activation_memory_budget_runtime_estimator is a raw callable without uuid() method, "
            "bypassing cache. Use CustomRuntimeEstimator for cache support."
        )

    if isinstance(solver, CustomKnapsackSolver):
        solver_uuid = solver.uuid()
        if solver_uuid is None:
            raise BypassAOTAutogradCache(
                "CustomKnapsackSolver.uuid() returned None, bypassing cache"
            )
    elif callable(solver) and not isinstance(solver, str):
        raise BypassAOTAutogradCache(
            "activation_memory_budget_solver is a raw callable without uuid() method, "
            "bypassing cache. Use CustomKnapsackSolver for cache support."
        )

    return runtime_estimator_uuid, solver_uuid


class AOTAutogradCacheDetails(FxGraphHashDetails):
    """
    Object to capture all the details for a dynamo graph module relevant to computing
    a safe and stable cache key for AOTAutograd.
    """

    def _iter_triton_kernels_from_node(self, node: Node) -> Generator[Any, None, None]:
        if isinstance(node.target, torch._ops.OpOverloadPacket):
            for attr in node.target._dir:
                custom_op = getattr(node.target, attr, None)
                if custom_op is not None:
                    yield from torch._library.triton.get_triton_kernels_for_op(
                        custom_op._name
                    )
            return
        if isinstance(node.target, torch._ops.OpOverload):
            yield from torch._library.triton.get_triton_kernels_for_op(
                node.target._name
            )

    def get_triton_source_codes_from_gm(
        self,
        gm: torch.fx.GraphModule,
    ) -> list[str]:
        if not has_triton_package():
            raise AssertionError("Triton is not available")

        triton_kernels = []
        for module in _iter_graph_modules(gm):
            for node in module.graph.nodes:
                triton_kernels.extend(self._iter_triton_kernels_from_node(node))

        triton_kernel_source_codes = []
        from torch._inductor.codegen.wrapper import (
            user_defined_triton_kernel_transitive_closure_source_code,
        )

        for kernel in triton_kernels:
            from triton.runtime.autotuner import Autotuner

            if isinstance(kernel, Autotuner):
                # Grab the Inner JITFunction
                kernel = kernel.fn
            source_codes = user_defined_triton_kernel_transitive_closure_source_code(
                kernel
            )
            triton_kernel_source_codes.append(source_codes)

        return triton_kernel_source_codes

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        aot_config: AOTConfig,
        fx_config: _CompileFxKwargs,
        act_input_paths: ActInputPaths = (),
    ) -> None:
        # FxGraphHashDetails contains all the keys related to inductor. Also
        # includes some system info.
        self.aot_config = aot_config
        self.act_input_paths = tuple(act_input_paths)
        self._record_runtime_state(gm)
        self.saved_tensors_hooks_fx_wrap_cache_hashes = (
            _collect_saved_tensors_hooks_fx_wrap_cache_hashes(gm)
        )
        self.sac_context_fn_hashes = _collect_context_fn_hashes(gm)

        # region_activation_memory_budget is graph-wide (the partitioner enforces
        # a single value across the graph) and propagates to every node, so the
        # cache key only needs the value off the first node. node.meta is stripped
        # by GraphModule.__reduce__, so without recording it here a budget change
        # would not invalidate the cache.
        first_node = next(iter(gm.graph.nodes), None)
        self.region_activation_memory_budget: float | None = (
            _get_memory_budget_annotation(first_node)
            if first_node is not None
            else None
        )

        # Note: We use the live config module, not self.autograd_config (the
        # saved config), because activation_memory_budget_runtime_estimator and
        # activation_memory_budget_solver are excluded from save_config (in
        # _save_config_ignore) since they're not serializable. We must access the
        # config module directly to get the patched runtime values.
        self.custom_estimator_solver_uuids = _get_custom_estimator_solver_uuids(config)
        self._init_fx_graph_hash_details(gm, example_inputs, fx_config)

    def _record_runtime_state(self, gm: torch.fx.GraphModule) -> None:
        self.grad_enabled = torch.is_grad_enabled()
        # Include per-device autocast dtype in cache key to avoid reusing
        # a graph compiled for one autocast dtype (e.g. bfloat16) when
        # running under a different autocast dtype (e.g. float16).
        self.autocast_state: dict[str, torch.dtype] = {}
        for device_type in torch._C._autocast_supported_devices():
            if torch.is_autocast_enabled(device_type):
                self.autocast_state[device_type] = torch.get_autocast_dtype(device_type)
        self.deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
        self.autograd_config = config.save_config()
        if has_triton_package():
            self.triton_kernel_source_codes = self.get_triton_source_codes_from_gm(gm)

    def _init_fx_graph_hash_details(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[Any],
        fx_config: _CompileFxKwargs,
    ) -> None:
        try:
            # FXGraphCache has constraints on what can be pickled in its inductor
            # config. Check that the gm is cacheable by inductor first,
            # and if it raises an exception, also bypass on our end.
            FxGraphCache._check_can_cache(gm, example_inputs, fx_config)
            super().__init__(gm, example_inputs, fx_config, [])
        except BypassFxGraphCache as e:
            # Sometimes inductor configs are unpickleable and can fail
            raise BypassAOTAutogradCache(str(e)) from e


class AOTAutogradCachePickler(FxGraphCachePickler):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        symint_name_map: dict[str, str] | None = None,
        graph_symint_names: set[str] | None = None,
    ) -> None:
        super().__init__(gm)
        self._symint_name_map: dict[str, str] = symint_name_map or {}
        self._graph_symint_names: set[str] = graph_symint_names or set()
        # pyrefly: ignore[missing-attribute]
        self.dispatch_table.update(
            {
                AOTConfig: functools.partial(self._reduce_aot_config),
                torch.Tensor: functools.partial(self._reduce_tensor),
                FakeScriptObject: functools.partial(self._reduce_fake_script_object),
            }
        )

    def _reduce_symint(
        self, s: torch.SymInt
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        name = str(s)
        normalized = self._symint_name_map.get(name)
        if normalized is not None:
            if name in self._graph_symint_names:
                return (_ident, (normalized,))
            # Non-graph SymInt with non-deterministic name (from MetaConverter).
            # Include the hint to distinguish different specializations.
            hint = s.node.hint
            if hint is not None:
                return (_ident, (normalized, hint))
        return (_ident, (name,))

    # pyrefly: ignore [bad-override]
    def reducer_override(self, obj: Any) -> Any:
        """
        Override to handle tensor subclasses (like DTensor) that aren't caught
        by the dispatch_table's exact type matching.

        The dispatch_table only matches exact types, so subclasses like DTensor
        fall through to the default __reduce_ex__ which includes non-deterministic
        storage addresses. This method catches those cases using isinstance checks.
        """
        if isinstance(obj, torch.fx.GraphModule):
            return self._reduce_graph_module_for_cache_key(obj)
        if (numpy_key := numpy_wrapper_cache_key(obj)) is not None:
            return (_ident, (numpy_key,))
        # Handle tensor subclasses that aren't exactly torch.Tensor
        # dispatch_table already handles torch.Tensor exactly
        if isinstance(obj, torch.Tensor) and type(obj) is not torch.Tensor:
            return self._reduce_tensor_subclass(obj)
        # Fall back to parent class reducer which handles unpicklable types
        return super().reducer_override(obj)

    @staticmethod
    def _format_global_for_cache_key(name: str, obj: Any) -> str:
        if (numpy_key := numpy_wrapper_cache_key(obj)) is not None:
            return f"# cache_key_numpy_wrapper {name}: {numpy_key!r}"

        from torch.fx.graph_module import _format_import_statement
        from torch.package import sys_importer

        return _format_import_statement(name, obj, sys_importer)

    def _reduce_graph_module_for_cache_key(
        self, gm: torch.fx.GraphModule
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        recompile = getattr(gm, "_real_recompile", gm.recompile)
        python_code = recompile()
        # FxGraphCache strips user-defined Triton side-table indices from
        # Inductor wrapper code. This reducer handles arbitrary AOT GraphModule
        # code, so keep the generated code exact and let AOTAutogradCacheDetails
        # carry Triton source/constant-arg metadata separately.
        dict_without_graph = gm.__dict__.copy()
        dict_without_graph.pop("_graph", None)

        import_block = "\n".join(
            sorted(
                self._format_global_for_cache_key(name, obj)
                for name, obj in python_code.globals.items()
            )
        )
        return (_ident, ((dict_without_graph, import_block),))

    # [NOTE] Tensor subclass stable hashing for AOT autograd cache
    # Python's hash() varies with PYTHONHASHSEED, making cache keys unstable
    # across processes. We use blake2b for cross-process determinism.
    #
    # EXTENSION POINT: Traceable wrapper subclasses can override cache key
    # generation by implementing _stable_hash_for_caching(self) -> str.
    # This method should return a deterministic string that uniquely identifies
    # the tensor's metadata for caching purposes. See DTensor for an example.
    #
    # We can't define a default method on subclasses because there is no abstract
    # base subclass, and we don't want to pollute torch.Tensor. Instead, we provide
    # a default implementation here that uses __tensor_flatten__ to recursively
    # hash inner tensors and metadata.

    def _hash_bytes_for_cache(self, data: bytes) -> str:
        return hashlib.blake2b(data, digest_size=16).hexdigest()

    def _hash_pickled_value_for_cache(self, value: Any) -> str:
        return self._hash_bytes_for_cache(pickle.dumps(value))

    def _stable_hash_for_cache_value(self, obj: Any) -> str:
        """Get a stable hash for an object used inside tensor subclass metadata."""
        from torch._custom_class_base import CustomClassBase
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if hasattr(obj, "_stable_hash_for_caching"):
            return obj._stable_hash_for_caching()
        if isinstance(obj, torch.Tensor) and is_traceable_wrapper_subclass(obj):
            return self._default_stable_hash_for_caching(obj)
        if isinstance(obj, CustomClassBase):
            # Opaque objects are runtime pass-throughs; only the type matters
            # for cache key purposes, not the instance identity or value.
            return self._hash_bytes_for_cache(type(obj).__qualname__.encode())
        if isinstance(obj, torch.Tensor):
            return self._hash_pickled_value_for_cache(
                extract_tensor_metadata_for_cache_key(obj)
            )
        return self._hash_pickled_value_for_cache(obj)

    def _reduce_tensor_subclass(
        self, tensor: torch.Tensor
    ) -> tuple[Callable[..., Any], tuple[Any]]:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if hasattr(tensor, "_stable_hash_for_caching"):
            return (_ident, (tensor._stable_hash_for_caching(),))
        if is_traceable_wrapper_subclass(tensor):
            warn_once(
                f"{type(tensor).__name__} does not implement _stable_hash_for_caching. "
                "For PT2-compatible tensor subclasses, it is recommended to implement "
                "_stable_hash_for_caching(self) -> str for stable AOT autograd caching."
            )
            return (_ident, (self._default_stable_hash_for_caching(tensor),))
        return self._reduce_tensor(tensor)

    def _collect_inner_tensor_hashes(
        self, tensor: torch.Tensor, inner_tensor_names: Sequence[str]
    ) -> dict[str, str]:
        inner_hashes = {}
        for name in inner_tensor_names:
            inner_hashes[name] = self._stable_hash_for_cache_value(
                getattr(tensor, name)
            )
        return inner_hashes

    def _stabilize_tensor_subclass_metadata(self, obj: Any) -> Any:
        from torch._custom_class_base import CustomClassBase

        if isinstance(obj, CustomClassBase):
            return type(obj).__qualname__
        if isinstance(obj, tuple):
            return tuple(self._stabilize_tensor_subclass_metadata(x) for x in obj)
        if isinstance(obj, list):
            return [self._stabilize_tensor_subclass_metadata(x) for x in obj]
        if isinstance(obj, dict):
            return {
                k: self._stabilize_tensor_subclass_metadata(v) for k, v in obj.items()
            }
        return obj

    def _default_stable_hash_for_caching(self, tensor: torch.Tensor) -> str:
        """
        Default stable hash implementation for traceable wrapper subclasses.
        """
        inner_tensor_names, subclass_metadata = tensor.__tensor_flatten__()  # type: ignore[attr-defined]
        cache_payload = (
            tensor.shape,
            tensor.requires_grad,
            self._stabilize_tensor_subclass_metadata(subclass_metadata),
            self._collect_inner_tensor_hashes(tensor, inner_tensor_names),
        )
        return self._hash_pickled_value_for_cache(cache_payload)

    def _reduce_aot_config(
        self, aot_config: AOTConfig
    ) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        """
        Reduce the config to a stable key for caching.
        """
        cacheable = aot_config.to_cacheable()
        return (
            # Cache entries persist the full CacheableAOTConfig, but cache keys
            # intentionally ignore the debug-only aot_id so equivalent graphs can hit.
            _ident,
            (
                cacheable.num_params_buffers,
                cacheable.keep_inference_input_mutations,
                cacheable.is_export,
                cacheable.no_tangents,
                cacheable.dynamic_shapes,
                cacheable.aot_autograd_arg_pos_to_source,
                cacheable.static_input_indices,
                cacheable.enable_log,
                cacheable.pre_dispatch,
                cacheable.precompile_backend_id,
            ),
        )

    def _reduce_tensor(self, t: torch.Tensor) -> tuple[Callable[..., Any], tuple[Any]]:
        """
        Reduce the tensor to a stable key for caching.
        """
        metadata = extract_tensor_metadata_for_cache_key(t)
        return (_ident, (metadata,))


@contextlib.contextmanager
def normalize_placeholder_names(
    gm: torch.fx.GraphModule,
) -> Generator[None, None, None]:
    """
    Context manager that normalizes the placeholder names in the graph module.
    This is used while generating a cache key for AOTAutogradCache, so that two graphs
    that are isomorphic when normalizing names can hit the same cache entry.
    This is safe because nothing underneath AOTAutograd uses the node names on the
    original dynamo graph: AOTAutograd re-traces with its own nodes, and guards are
    in terms of original sources rather than placeholder names.
    """
    # Standalone inductor: we're bypassing AOTAutogradCache anyway, so return the graph
    # as-is
    if not config.autograd_cache_normalize_inputs or not hasattr(gm, "graph"):
        yield
        return

    # Track all the old state of placeholders
    old_placeholder_names = []
    old_symint_names = []
    old_used_names = copy(gm.graph._graph_namespace._used_names)
    i = 0
    j = 0
    for n in gm.graph.find_nodes(op="placeholder", sort=True):
        if n.type == torch.SymInt:
            new_name = f"s_{j}"
            old_symint_names.append((n.name, n.target))
            n.target = new_name
            n._rename(new_name)
            j += 1
        else:
            # _rename renames the node in the body of the function,
            # but it doesn't change the raw name from node.target
            # So we also set the raw_name of node.target to a new placeholder name
            new_placeholder_name = f"p_{i}"
            old_placeholder_names.append((n.name, n.target))
            n.target = new_placeholder_name
            n._rename(new_placeholder_name)
            i += 1
    gm.recompile()
    try:
        yield
    finally:
        # Used_names contains all our old placeholder names,
        # so we clear it temporarily when we put them back
        gm.graph._graph_namespace._used_names = set()
        # Restore the placeholder names
        i = 0
        j = 0
        for n in gm.graph.find_nodes(op="placeholder", sort=True):
            if n.type == torch.SymInt:
                (name, target) = old_symint_names[j]
                n.target = target
                n._rename(name)
                j += 1
            else:
                (name, target) = old_placeholder_names[i]
                n.target = target
                n._rename(name)
                i += 1
        if i != len(old_placeholder_names):
            raise AssertionError(
                f"i={i} != len(old_placeholder_names)={len(old_placeholder_names)}"
            )
        if j != len(old_symint_names):
            raise AssertionError(
                f"j={j} != len(old_symint_names)={len(old_symint_names)}"
            )
        # Now restore the old namespace's used names
        gm.graph._graph_namespace._used_names = old_used_names
        gm.recompile()


def create_fx_config(
    compiler_config_extra: CompilerConfigExtra | None = None,
    compile_region_name: str | None = None,
) -> _CompileFxKwargs:
    if compiler_config_extra is None:
        cudagraphs = BoxedBool(torch._inductor.config.triton.cudagraphs)
        boxed_forward_device_index = None
    else:
        cudagraphs = compiler_config_extra.cudagraphs
        boxed_forward_device_index = compiler_config_extra.forward_device
    return {
        "cudagraphs": cudagraphs,
        "boxed_forward_device_index": boxed_forward_device_index,
        "compile_region_name": compile_region_name,  # pyrefly: ignore[bad-typed-dict-key]
    }


def _check_triton_cache_version() -> None:
    if not has_triton_package():
        return

    # Due to https://github.com/triton-lang/triton/issues/3729, if triton is <
    # 3.2.0, AOTAutogradCache may cause us to attempt to load a cache entry
    # without initializing the CUDA context on the autograd thread.
    #
    # Without caching, we naturally do this initialization when tracing through
    # the graph with the autograd engine.
    import triton

    if triton.__version__ < "3.2.0":
        raise BypassAOTAutogradCache("AOTAutogradCache requires triton 3.2.0")


def _get_debug_lines_for_cache_key(
    pickler: AOTAutogradCachePickler,
    details: AOTAutogradCacheDetails,
    key: str,
) -> list[str]:
    # debug_lines re-hashes every attribute individually and is expensive. Only
    # compute when debug logging is enabled.
    if not log.isEnabledFor(logging.DEBUG):
        return []

    debug_lines = pickler.debug_lines(details)
