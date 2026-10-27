    # aot_inductor codegens a call that takes in just the inputs, so we don't return a wrapper
    # that drops constant-ified params
    if V.aot_compilation:
        return optimized_function

    def wrapper(args: list[object]) -> Sequence[torch.Tensor]:
        args_new = [
            args[i - unwrapped_args_offsets[min(i, max_offset_idx)]]
            for i in preserved_arg_indices
        ]
        args.clear()
        return optimized_function(args_new)

    wrapper._boxed_call = True  # type: ignore[attr-defined]

    return wrapper


def get_cpp_wrapper_config(log_cudagraph_skip: bool = True) -> dict[str, object]:
    if log_cudagraph_skip and config.triton.cudagraphs and config.graph_partition:
        log_cudagraph_skip_and_bump_counter(
            format_default_skip_message(
                "cpp-wrapper does not support graph partition yet"
            )
        )

    autotune_at_compile_time = (
        config.triton.autotune_at_compile_time
        if config.triton.autotune_at_compile_time is not None
        # Default to True for AOTI. Subject to change in future.
        else has_triton() and V.aot_compilation
    )
    return {
        "triton.autotune_at_compile_time": autotune_at_compile_time,
        "triton.autotune_cublasLt": not autotune_at_compile_time,
        "triton.cudagraphs": (
            config.triton.cudagraphs
            and not V.aot_compilation
            and not config.graph_partition
        ),
        "triton.store_cubin": True,
    }


def get_cuda_device_context(gm: torch.fx.GraphModule) -> AbstractContextManager[None]:
    """
    Returns a cuda device context manager if there is a single device in the graph
    """
    if not torch.cuda.is_available():
        return contextlib.nullcontext()

    cuda_devices: OrderedSet[torch.device] = OrderedSet(
        device for device in get_all_devices(gm) if device.type == "cuda"
    )

    return (
        torch.cuda.device(next(iter(cuda_devices)))  # type: ignore[return-value]
        if len(cuda_devices) == 1
        else contextlib.nullcontext()
    )


def partition_fn(
    gm: GraphModule,
    joint_inputs: Sequence[object],
    *,
    # When set, replaces the default partitioner (`config.custom_partitioner_fn`
    # or `min_cut_rematerialization_partition`) while still running Inductor's
    # joint-graph passes first. Used by `_get_partition_fn` so HOP subgraphs
    # go through the same joint-pass + raw-partitioner pipeline as the outer
    # Inductor compile.
    partitioner_fn_override: Callable[..., Any] | None = None,
    **kwargs: object,
) -> tuple[GraphModule, GraphModule]:
    cuda_context = get_cuda_device_context(gm)
    with cuda_context:
        # We can skip the invoke_subgraph because the
        # entire_partition_fn is called recursively for invoke_subgraph
        # in partitioning.
        inputs_devices = get_inputs_devices(joint_inputs, gm)
        gm = _recursive_joint_graph_passes(
            gm,
            skip_invoke_subgraph=True,
            input_device=next(iter(inputs_devices)),
        )

    static_lifetime_input_indices: list[int] | None = kwargs.pop(  # type: ignore[assignment]
        "static_lifetime_input_indices", None
    )

    if partitioner_fn_override is not None:
        return partitioner_fn_override(
            gm,
            joint_inputs,
            static_lifetime_input_indices=static_lifetime_input_indices,
            **kwargs,
        )

    if config.custom_partitioner_fn is None:
        with dynamo_utils.dynamo_timed(
            "min_cut_rematerialization_partition", log_pt2_compile_event=True
        ):
            return min_cut_rematerialization_partition(
                gm,
                joint_inputs,
                compiler="inductor",
                static_lifetime_input_indices=static_lifetime_input_indices,
                # pyrefly: ignore[bad-argument-type]
                **kwargs,
            )
    else:
        if not isinstance(config.custom_partitioner_fn, CustomPartitionerFn):
            raise AssertionError(
                f"Expected custom_partitioner_fn to be CustomPartitionerFn, "
                f"got {type(config.custom_partitioner_fn)}"
            )
        with dynamo_utils.dynamo_timed(
            config.custom_partitioner_fn.__class__.__name__,
            log_pt2_compile_event=True,
        ):
            return config.custom_partitioner_fn(
                gm,
                joint_inputs,
                compiler="inductor",
                static_lifetime_input_indices=static_lifetime_input_indices,
                **kwargs,
            )


def get_num_model_outputs(model: GraphModule) -> int:
    model_outputs_node = output_node(model)
    model_outputs = pytree.arg_tree_leaves(*model_outputs_node.args)
    return len(model_outputs)


def cudagraph_annotation_context(
    cudagraphs: BoxedBool,
) -> contextlib.AbstractContextManager[None]:
    # When an annotation force-enables cudagraphs but the global config has them
    # off, patch config.triton.cudagraphs for the duration of compilation,
    # so existing codepaths that access config.triton.cudagraphs work
    if cudagraphs.value and not config.triton.cudagraphs:
        return config.patch({"triton.cudagraphs": True})
    return contextlib.nullcontext()


@dataclass(frozen=True)
class CompilerConfigExtra:
    cudagraphs: BoxedBool
    graph_id: int
    forward_device: BoxedDeviceIndex
    forward_is_partitioned: BoxedBool
    cudagraphs_bwd_override: bool | None = None


def create_compiler_config_extra(
    gm: GraphModule | GmWrapper,
) -> CompilerConfigExtra:
    gm_meta = gm.meta if isinstance(gm, GraphModule) else None

    # Although cudagraphs may have been enabled via config, various
    # conditions (which are tested within the bowels of Inductor) may
    # force cudagraphs to be disabled.  This mutable box lets us retrieve
    # the final determination if cudagraphs actually can be used or not.
    cudagraphs = BoxedBool(config.triton.cudagraphs)

    cudagraphs_bwd_override: bool | None = None

    # Override cudagraphs BoxedBool based on override_cudagraphs annotation.
    # Disabling fwd disables bwd (copying activations isn't profitable),
    # so cudagraphs_bwd_override is only needed for fwd=True / bwd=False.
    if (
        gm_meta is not None
        and (annotation := gm_meta.get("cudagraph_annotation")) is not None
    ):
        if annotation.fwd is not None and annotation.fwd != config.triton.cudagraphs:
            cudagraphs = BoxedBool(annotation.fwd)
            if annotation.fwd:
                cudagraphs_log.info(
                    "enabling cudagraphs due to override_cudagraphs annotation"
                )
            else:
                log_cudagraph_skip_and_bump_counter(
                    "disabling cudagraphs due to override_cudagraphs annotation"
                )

        # bwd override only matters when fwd enables cudagraphs but bwd
        # explicitly disables them.
        if cudagraphs.value and annotation.bwd is not None and not annotation.bwd:
            cudagraphs_bwd_override = annotation.bwd
            log_cudagraph_skip_and_bump_counter(
                "disabling cudagraphs for backward due to override_cudagraphs annotation"
            )

    # TODO: The modern style is to use CompileId from TracingContext to
    # identify Inductor compilation.  However, this CompileId cannot
    # uniquely identify multiple Inductor compilations that arise from
    # DDPOptimizer
    graph_id = next(_graph_counter)

    # See [Backward Generation Handling]
    forward_device = BoxedDeviceIndex(None)

    # Set by the forward compilation when it is partitioned for CUDA graphs.
    # The backward reads this to decide whether saved tensors can be assumed
    # to have fixed addresses.
    forward_is_partitioned = BoxedBool(False)

    return CompilerConfigExtra(
        cudagraphs=cudagraphs,
        graph_id=graph_id,
        forward_device=forward_device,
        cudagraphs_bwd_override=cudagraphs_bwd_override,
        forward_is_partitioned=forward_is_partitioned,
    )


def compile_fx_forward(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    num_orig_model_outputs: int,
    num_example_inputs: int,
    compiler_config_extra: CompilerConfigExtra,
    inner_compile: Callable[..., OutputCode] = compile_fx_inner,
    is_inference: bool = False,
) -> OutputCode:
    """
    Compile the forward graph of the given graph module.

    Args:
        gm: The graph module to compile.
        example_inputs: The example inputs to use for compilation.
        num_orig_model_outputs: The number of model outputs from the original dynamo graph.
        num_example_inputs: The number of example inputs from the original dynamo graph.
        compiler_config_extra: Extra configuration for the compiler.
        inner_compile: The inner compile function to use.
        is_inference: Whether this is an inference graph.
    """

    if is_inference:
        # partition_fn won't be called
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "before_joint_graph",
                "encoding": "string",
            },
            payload_fn=lambda: gm.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )

        # Snapshot stack traces on the output node before passes run,
        # as later passes may strip stack_trace from individual nodes.
        output = output_node(gm)
        output.meta["output_stack_traces"] = [
            (
                arg.meta.get("stack_trace")
                if isinstance(arg, torch.fx.node.Node)
                else None
            )
            for arg in output.args[0]  # type: ignore[union-attr]
        ]

        # Record original output strides BEFORE joint_graph_passes, because
        # pad_mm (run as part of joint_graph_passes) can introduce views with
        # padded strides that would be incorrectly captured as "original".
        _recursive_record_original_output_strides(gm)

        inputs_devices = get_inputs_devices(example_inputs, gm)
        gm = _recursive_joint_graph_passes(gm, input_device=next(iter(inputs_devices)))

        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "after_joint_graph",
                "encoding": "string",
            },
            payload_fn=lambda: gm.print_readable(
                print_output=False, include_stride=True, include_device=True
            ),
        )

    fixed = torch._inductor.utils.num_fw_fixed_arguments(
        num_example_inputs, len(example_inputs)
    )

    model_outputs_node = output_node(gm)
    clone_live_user_outputs = _cudagraph_trees_clone_live_user_outputs()
    model_outputs = None
    user_visible_output_idxs: list[int] = []
    if config.keep_output_stride or clone_live_user_outputs:
        model_outputs = pytree.arg_tree_leaves(*model_outputs_node.args)
        num_model_outputs = len(model_outputs)

        context = torch._guards.TracingContext.try_get()
        # See Note [User Outputs in the inductor graph]
        if context is not None and context.fw_metadata and not is_inference:
            original_output_start_index = (
                context.fw_metadata.num_mutated_inp_runtime_indices
            )
        else:
            original_output_start_index = 0

        if num_orig_model_outputs > num_model_outputs:
            raise AssertionError(
                f"Expected num_orig_model_outputs ({num_orig_model_outputs}) "
                f"<= num_model_outputs ({num_model_outputs})"
            )

        # Note [User Outputs in the inductor graph]
        # We makes the following assumption
        # For inference
        #   len(orig_model_outputs) == len(model_outputs)
        # For training
        #   len(orig_model_outputs) <= len(model_outputs)
        # During training, most of the time the model_outputs starts with
        # original module's outputs followed by saved activations.
        # But this can be not true if the model have inplace updated tensors.
        # AOTAutograd will make those tensors being returned before the original
        # module's output.
        # To make things safe, we'll use original_output_start_index field
        # set by AOTAutograd to decide where the original module outputs start.
        orig_output_end_idx = original_output_start_index + num_orig_model_outputs
        # Sanity check: we are about to splice out the "user" outputs from the full set
        # of "graph" outputs. Make sure we're within bounds.
        if orig_output_end_idx > num_model_outputs:
            raise AssertionError(
                f"Expected orig_output_end_idx ({orig_output_end_idx}) "
                f"<= num_model_outputs ({num_model_outputs})"
            )

        user_visible_output_idxs = [
            idx
            for idx in range(original_output_start_index, orig_output_end_idx)
            if isinstance(model_outputs[idx], torch.fx.Node)
        ]

    if config.keep_output_stride or clone_live_user_outputs:
        model_outputs_node.meta["user_visible_output_idxs"] = user_visible_output_idxs
    else:
        model_outputs_node.meta["user_visible_output_idxs"] = []

    # We also mark the invoke_subgraph outputs as user_visible to
    # force the outputs of invoke_subgraph subgraph to follow the
    # original strides
    _recursive_record_user_visible_output_idxs(gm)

    with cudagraph_annotation_context(compiler_config_extra.cudagraphs):
        result = inner_compile(
            gm,
            example_inputs,
            static_input_idxs=get_static_input_idxs(fixed),
            cudagraphs=compiler_config_extra.cudagraphs,
            graph_id=compiler_config_extra.graph_id,
            is_inference=is_inference,
            boxed_forward_device_index=compiler_config_extra.forward_device,
        )

        if (
            not is_inference
            and isinstance(result, CompiledFxGraph)
            and result.partition_maps
            and len(result.partition_maps) > 1
        ):
            compiler_config_extra.forward_is_partitioned.value = True

        return result


def compile_fx_backward(
    gm: GraphModule,
    example_inputs: Sequence[InputType],
    compiler_config_extra: CompilerConfigExtra,
    inner_compile: Callable[..., OutputCode] = compile_fx_inner,
) -> OutputCode:
    """
    Compile the backward graph of the given graph module.

    Args:
        gm: The graph module to compile.
        example_inputs: The example inputs to use for compilation.
        compiler_config_extra: Extra configuration for the compiler.
        inner_compile: The inner compile function to use.
    """
    from torch._dynamo.convert_frame import compile_lock

    with compile_lock:
        model_outputs_node = output_node(gm)
        if config.bw_outputs_user_visible:
            model_outputs = pytree.arg_tree_leaves(*model_outputs_node.args)
            model_outputs_node.meta["user_visible_output_idxs"] = [
                idx
                for idx, n in enumerate(model_outputs)
                if isinstance(n, torch.fx.Node)
            ]
        else:
            model_outputs_node.meta["user_visible_output_idxs"] = []

        fixed = count_tangents(gm)

        # Check if cudagraphs should be overridden for backward via annotation
        cudagraphs = compiler_config_extra.cudagraphs
        if compiler_config_extra.cudagraphs_bwd_override is not None:
            cudagraphs = BoxedBool(compiler_config_extra.cudagraphs_bwd_override)

        # When the forward was partitioned, saved activations from inline
        # code between partitions are NOT at fixed addresses. Only mark
        # primals (params/buffers) as static.
        if compiler_config_extra.forward_is_partitioned.value:
            static_input_idxs: Sequence[int] = get_static_bw_input_idxs(gm)
        else:
            static_input_idxs = list(range(fixed))
        with (
            (
                config.patch(get_cpp_wrapper_config())
                if config.cpp_wrapper
                else contextlib.nullcontext()
            ),
            cudagraph_annotation_context(cudagraphs),
        ):
            return inner_compile(
                gm,
                example_inputs,
                static_input_idxs=static_input_idxs,
                cudagraphs=cudagraphs,
                is_backward=True,
                graph_id=compiler_config_extra.graph_id,
                boxed_forward_device_index=compiler_config_extra.forward_device,
            )


def run_pre_grad_passes(
    model_: GraphModule, example_inputs_: Sequence[InputType]
) -> GraphModule:
    # "before_pre_grad_graph" is used in inductor provenance
    # tracking highlighter front-end.
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "before_pre_grad_graph",
            "encoding": "string",
        },
        payload_fn=lambda: model_.print_readable(
            print_output=False, include_stride=True, include_device=True
        )
        + f"\n\n # graph id: {id(model_.graph)}",
    )
    pre_grad_graphs_log.debug(
        "%s",
        lazy_format_graph_code(
            "BEFORE PRE GRAD",
            model_,
            include_stride=True,
            include_device=True,
            colored=True,
        ),
    )
    torch._inductor.debug._pre_grad_graph_id = id(model_.graph)

    if config.effective_provenance_tracking_level() == 1:
        for node in model_.graph.nodes:
            if node.stack_trace:
                torch._inductor.debug._inductor_pre_grad_node_stack_trace[node.name] = (
                    node.stack_trace
                )

    model_ = _recursive_pre_grad_passes(model_, example_inputs_)
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "after_pre_grad_graph",
            "encoding": "string",
        },
        payload_fn=lambda: model_.print_readable(
            print_output=False, include_stride=True, include_device=True
        )
        + f"\n\n # graph id: {id(model_.graph)}",
    )
    return model_


@dataclass(frozen=True)
class _ConstantDecompTable(Generic[_P, _T]):
    """
    Picklable wrapper around a user-supplied decomposition dict.

    This module-level class is addressable as (module, qualname), so pickle can
    serialize an instance by reducing its single field.
    """

    table: dict[OpOverload, Callable[_P, _T]]

    def __call__(self) -> dict[OpOverload, Callable[_P, _T]]:
        return self.table


def compile_fx(
    model_: GraphModule,
    example_inputs_: Sequence[InputType],
    inner_compile: Callable[..., OutputCode] = compile_fx_inner,
    config_patches: dict[str, Any] | None = None,
    decompositions: dict[OpOverload, Callable[..., Any]] | None = None,
    ignore_shape_env: bool = False,
    compile_region_name: str | None = None,
) -> CompileFxOutput:
    """
    Main entry point for compiling given FX graph.  Despite the fact that this
    lives in :mod:`torch._inductor`, this function is responsible for calling
    into AOT Autograd (and we will eventually get a callback to
    ``inner_compile`` to perform actual compilation.  In other words, this
    function orchestrates end-to-end compilation for the inductor backend when
    you use :func:`torch.compile`.

    NB: This function TAKES OWNERSHIP of the input ``model_`` and can potentially
    mutate it!  Make a copy if you need to preserve the original GraphModule.
    """
    if decompositions is not None:
        get_decomp_fn: Callable[..., dict[Any, Callable[..., Any]]] = (
            _ConstantDecompTable(decompositions)
        )
    else:
        get_decomp_fn = select_decomp_table

    # Some arguments trigger a recursive call to compile_fx.  Handle these
    # short circuits first, before anything else

    from torch._inductor.compiler_bisector import CompilerBisector

    if CompilerBisector.disable_subsystem("inductor", "pre_grad_graph"):
        # pyrefly: ignore [bad-return]
        return model_

    if config_patches:
        with config.patch(config_patches):
            return compile_fx(
                model_,
                example_inputs_,
                # need extra layer of patching as backwards is compiled out of scope
                inner_compile=config.patch(config_patches)(inner_compile),
                decompositions=decompositions,
                ignore_shape_env=ignore_shape_env,
                compile_region_name=compile_region_name,
            )

    # Keep region names out of graph_kwargs so they don't perturb FX cache keys.
    inner_compile = functools.partial(
        inner_compile,
        compile_region_name=compile_region_name,
    )

    # Wake up the AsyncCompile subproc pool as early as possible (if there's cuda).
    if any(
        isinstance(e, torch.Tensor) and e.device.type in ("cuda", "xpu")
        for e in example_inputs_
    ):
        torch._inductor.async_compile.AsyncCompile.wakeup()

    if config.cpp_wrapper or config.fx_wrapper:
        from torch.fx.experimental.proxy_tensor import _coor_enabled

        if _coor_enabled():
            # cpp_wrapper/AOTInductor bakes the compile-time device index into the C++
            # device guard, and fx_wrapper's device-context codegen is a no-op (see
            # wrapper_fxir.py); neither is rank-portable, so refuse rather than silently
            # emit a non-portable artifact.
            #
            # NB cudagraphs is deliberately NOT refused here. Its device dependence
            # (CompiledFxGraph.device_idxs, which cudagraph_post_compile passes to
            # cudagraphify as device_index) lives in the wrapper-level artifact, and that
            # artifact is not shared across ranks today: the FX graph cache key embeds the
            # device, so each rank builds its own. Only the kernel/cubin layer is shared,
            # and that is what the device-agnostic launcher handles. Whoever makes the FX
            # graph cache key device-agnostic must revisit device_idxs (and re-check that
            # graph-partition functions still define _coor_device_idx in their own scope)
            # before cudagraphs can ride on a shared artifact.
            raise RuntimeError(
                "compile-on-one-rank (device-as-parameter) is not supported with "
                "cpp_wrapper/AOTInductor or fx_wrapper: the device guard bakes the "
                "compile-time device index, which is not rank-portable."
            )
        from torch._export.non_strict_utils import _fakify_script_objects

        cpp_wrapper_config = config.cpp_wrapper
        fx_wrapper_config = config.fx_wrapper

        with (
            config.patch(get_cpp_wrapper_config()),
            V.set_real_inputs(example_inputs_),
        ):
            inputs_: Sequence[InputType] = (
                _extract_inputs_from_exported_gm(model_, example_inputs_)
                if isinstance(model_, GraphModule)
                else example_inputs_
            )
            fake_mode = detect_fake_mode(inputs_)
            with _fakify_script_objects(model_, inputs_, {}, fake_mode) as (
                patched_mod,
                fake_args,
                _,
                _,
                _,
            ):
                return _maybe_wrap_and_compile_fx_main(
                    patched_mod,
                    fake_args,
                    inner_compile=functools.partial(
                        inner_compile,
                        cpp_wrapper=cpp_wrapper_config,
                        fx_wrapper=fx_wrapper_config,
                    ),
                    ignore_shape_env=ignore_shape_env,
                    get_decomp_fn=get_decomp_fn,
                    compile_region_name=compile_region_name,
                )

    return _maybe_wrap_and_compile_fx_main(
        model_,
        example_inputs_,
        inner_compile,
        ignore_shape_env,
        get_decomp_fn=get_decomp_fn,
        compile_region_name=compile_region_name,
    )


def _extract_inputs_from_exported_gm(
    gm: GraphModule, example_inputs_: Sequence[InputType]
) -> Sequence[InputType]:
    fake_inputs = [
        node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
    ]

    if not config.fx_wrapper:
        # Replace non-tensor inputs with Nones
        # constant scalars embedded in the graph
        # symbolic scalars (symint) are not supported in non-fx_wrapper mode
        fake_inputs = [
            inp if isinstance(inp, torch.Tensor) else None for inp in fake_inputs
        ]

    if any(v is not None for v in fake_inputs):
        # Validate devices before switching to fake tensors.
        for idx, fi, i in zip(count(), fake_inputs, example_inputs_):
            if fi is not None and isinstance(fi, torch.Tensor):
                if not isinstance(i, torch.Tensor):
                    raise AssertionError(
                        f"Expected example input at position #{idx} to be a Tensor, got {type(i)}"
                    )
                if fi.device != i.device:
                    raise ValueError(
                        f"Device mismatch between fake input and example input at position #{idx}: "
                        f"{fi.device} vs {i.device}. If the model was exported via torch.export(), "
                        "make sure torch.export() and torch.aot_compile() run on the same device."
                    )
        return fake_inputs

    return example_inputs_


def _maybe_wrap_and_compile_fx_main(
    model_: GraphModule,
    example_inputs_: Sequence[InputType],
    inner_compile: Callable[..., OutputCode],
    ignore_shape_env: bool,
    *,
    get_decomp_fn: Callable[..., dict[Any, Callable[..., Any]]] = select_decomp_table,
    compile_region_name: str | None = None,
) -> CompileFxOutput:
    """
    Part of compile_fx, called after patching configs.

    Ultimately we want to call _compile_fx_main, where the actual work happens.
    But under various conditions, various forms of wrapping might be needed
    around _compile_fx_main.
    """
    # Each wrapper below takes a self-contained compile_gm function which is
    # called inside the wrapper. This just recursively calls this function.
    compile_gm = functools.partial(
        _maybe_wrap_and_compile_fx_main,
        inner_compile=inner_compile,
        ignore_shape_env=ignore_shape_env,
        get_decomp_fn=get_decomp_fn,
        compile_region_name=compile_region_name,
    )
    if not graph_returns_tuple(model_):
        return make_graph_return_tuple(model_, example_inputs_, compile_gm)

    if isinstance(model_, GraphModule) and isinstance(
        model_.graph._codegen, _PyTreeCodeGen
    ):
        # this graph is the result of dynamo.export()
        return handle_dynamo_export_graph(model_, example_inputs_, compile_gm)

    if any(isinstance(x, (list, tuple, dict)) for x in example_inputs_):
        # NB: this short circuit never occurs for Dynamo produced graphs
        # (which are pre-flattened)
        return flatten_graph_inputs(model_, example_inputs_, compile_gm)

    # Finally do the actual work!
    return _compile_fx_main(
        model_,
        example_inputs_,
        inner_compile,
        ignore_shape_env,
        get_decomp_fn=get_decomp_fn,
        compile_region_name=compile_region_name,
    )


def _compile_fx_main(
    model_: GraphModule,
    example_inputs_: Sequence[InputType],
    inner_compile: Callable[..., OutputCode],
    ignore_shape_env: bool,
    *,
    get_decomp_fn: Callable[..., dict[Any, Callable[..., Any]]] = select_decomp_table,
    compile_region_name: str | None = None,
) -> CompileFxOutput:
    """
    Main part of compile_fx, called after wrapping is done.

    Roughly speaking, here the steps will be:
    (1) apply pre-grad passes
    (2) create `fw_compiler` and `bw_compiler` functions out of `inner_compile`
    (3) call aot_autograd, which:
    - (3a) creates a joint graph with `decompositions`,
    - (3b) partitions it with `partition_fn` into fw and bw graphs (applying joint-graph passes),
    - (3c) calls `fw_compiler` and `bw_compiler` on those graphs (applying post-grad passes)
    - (3d) finally, assembles the fw and bw compiled functions back together and returns.
    """
    with (
        _use_lazy_graph_module(dynamo_config.use_lazy_graph_module),
        enable_python_dispatcher(),
        torch.fx.traceback.preserve_node_meta(
            config.effective_provenance_tracking_level() == 1
        ),
        torch._inductor.debug.reset_provenance_globals(),
    ):
        # Note: Pre-grad passes are now run inside aot_module_simplified (via the
        # pre_grad_passes callback) after the cache lookup.

        if config._raise_error_for_testing:
            raise AssertionError("config._raise_error_for_testing is set")

        num_example_inputs = len(example_inputs_)

        compiler_config_extra = create_compiler_config_extra(model_)

        decompositions = get_decomp_fn()
        inner_compile = functools.partial(inner_compile, get_decomp_fn=get_decomp_fn)

        def fw_compiler_base(
            gm: GraphModule,
            example_inputs: Sequence[InputType],
            is_inference: bool,
        ) -> OutputCode:
            with dynamo_utils.dynamo_timed("compile_fx.<locals>.fw_compiler_base"):
                if isinstance(model_, GraphModule):
                    num_orig_model_outputs = get_num_model_outputs(model_)
                else:
                    num_orig_model_outputs = get_num_model_outputs(gm)
                return compile_fx_forward(
                    gm,
                    example_inputs,
                    num_orig_model_outputs=num_orig_model_outputs,
                    num_example_inputs=num_example_inputs,
                    compiler_config_extra=compiler_config_extra,
                    inner_compile=inner_compile,
                    is_inference=is_inference,
                )

        fw_compiler: Callable[[GraphModule, Sequence[InputType]], OutputCode] = (
            functools.partial(fw_compiler_base, is_inference=False)
        )
        fw_compiler = SerializableAOTDispatchCompiler(OutputCode, fw_compiler)

        if config.freezing and not torch.is_grad_enabled():
            inference_compiler: Callable[..., Any] = functools.partial(
                fw_compiler_freezing,
                dynamo_model=model_,
                num_example_inputs=num_example_inputs,
                inner_compile=inner_compile,
                cudagraphs=compiler_config_extra.cudagraphs,
                graph_id=compiler_config_extra.graph_id,
                forward_device=compiler_config_extra.forward_device,
            )
        else:
            inference_compiler = functools.partial(fw_compiler_base, is_inference=True)
            inference_compiler = SerializableAOTDispatchCompiler(
                OutputCode, inference_compiler
            )

        @compile_time_strobelight_meta(phase_name="backward")
        def bw_compiler(
            gm: GraphModule, example_inputs: Sequence[InputType]
        ) -> OutputCode:
            with (
                dynamo_utils.dynamo_timed("compile_fx.<locals>.bw_compiler"),
            ):
                return compile_fx_backward(
                    gm,
                    example_inputs,
                    compiler_config_extra=compiler_config_extra,
                    inner_compile=inner_compile,
                )

        bw_compiler = SerializableAOTDispatchCompiler(OutputCode, bw_compiler)

        fake_mode = detect_fake_mode(
            example_inputs_
        ) or torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        tracing_context = (
            torch._guards.TracingContext.try_get()
            or torch._guards.TracingContext(fake_mode)
        )

        if V.aot_compilation and not config.enable_autograd_for_aot:
            from .utils import is_valid_aoti_model_name

            is_valid_aoti_model_name()

            # Pre-grad passes for the aot_autograd path are run inside
            # aot_module_simplified after the cache lookup.  The
            # aot_export_module path doesn't use that cache so run them here.
            if isinstance(model_, GraphModule):
                model_ = run_pre_grad_passes(model_, example_inputs_)

            with functorch_config.patch(
                unlift_effect_tokens=True,
                selective_decompose=config.selective_decompose,
            ):
                gm, graph_signature = aot_export_module(
                    model_,
                    example_inputs_,
                    trace_joint=False,
                    decompositions=decompositions,
                )
                if not isinstance(gm, GraphModule):
                    raise AssertionError(
                        f"Expected GraphModule from aot_export_module, got {type(gm)}"
                    )
                from torch._export.utils import _detect_fake_mode_from_gm

                fake_mode = _detect_fake_mode_from_gm(gm)  # type: ignore[assignment]
                # aot_export_module doesn't account for constant tensor attributes
                # so we end up having tensors that don't have fake vals attached.
                # This can happen when upstream export is non-strict where we
                # preserve the original module params/buffers. Once AOTI switches
                # to ep.run_decompositions() flow to lower to post-autograd opset
                # this will go away.
                for node in gm.graph.nodes:
                    if node.op == "get_attr" and "val" not in node.meta:
                        target = attrgetter(node.target)(gm)
                        if isinstance(target, torch.Tensor):
                            if fake_mode is None:
                                raise AssertionError(
                                    "Expected fake_mode to be set for get_attr tensor node"
                                )
                            node.meta["val"] = fake_mode.from_tensor(
                                target, static_shapes=True
                            )
                        elif isinstance(target, torch.ScriptObject) or is_custom_class(
                            type(target)
                        ):
                            node.meta["val"] = (
                                torch._library.fake_class_registry.maybe_to_fake_obj(
                                    fake_mode, target
                                )
                            )
                        elif isinstance(target, FakeScriptObject):
                            node.meta["val"] = target

            unlifted_gm = _unlift_graph(model_, gm, graph_signature)
            if "dynamo_flat_name_to_original_fqn" in model_.meta:
                unlifted_gm.meta["dynamo_flat_name_to_original_fqn"] = model_.meta[
                    "dynamo_flat_name_to_original_fqn"
                ]

            if "dynamo_compile_id" in model_.meta:
                unlifted_gm.meta["dynamo_compile_id"] = model_.meta["dynamo_compile_id"]

            # Disable amp as in aot_dispatch_autograd (https://github.com/pytorch/pytorch/pull/86515)
            # In inference_compiler (fw_compiler_base), _recursive_joint_graph_passes will call into
            # _sfdp_init() to register patterns.
            # When fallback_random is set to True, the sdpa patterns will be traced during runtime.
            # If amp is turned on, the traced FP32 patterns will have prims.convert_element_type which
            # will be the same as the generated FP16 patterns.
            disable_amp = torch._C._is_any_autocast_enabled()
            context = (
                torch._C._DisableAutocast if disable_amp else contextlib.nullcontext
            )
            with V.set_fake_mode(fake_mode), compiled_autograd._disable(), context():
                return inference_compiler(unlifted_gm, example_inputs_)

        with (
            V.set_fake_mode(fake_mode),
            torch._guards.tracing(tracing_context),
            compiled_autograd._disable(),
            functorch_config.patch(
                unlift_effect_tokens=True,
                selective_decompose=config.selective_decompose,
            ),
        ):
            try:
                return dynamo_common.aot_autograd(
                    fw_compiler=fw_compiler,
                    bw_compiler=bw_compiler,
                    inference_compiler=inference_compiler,
                    decompositions=decompositions,
                    partition_fn=partition_fn,
                    keep_inference_input_mutations=True,
                    compiler_config_extra=compiler_config_extra,
                    ignore_shape_env=ignore_shape_env,
                    pre_grad_passes=run_pre_grad_passes,
                    compile_region_name=compile_region_name,
                )(model_, example_inputs_)
            except ShortenTraceback as e:
                # We will also shorten the traceback inside dynamo.
                # This is only useful if inductor is called directly with an FX graph.
                raise e.remove_dynamo_frames() from None  # see TORCHDYNAMO_VERBOSE=1


def graph_returns_tuple(gm: GraphModule) -> bool:
    """True if a FX graph returns a tuple"""
    if not isinstance(gm, GraphModule):
        return True  # can't check this, assume true
    (rv,) = output_node(gm).args
    if isinstance(rv, (list, tuple)):
        return True
    if (
        isinstance(rv, torch.fx.node.Node)
        and hasattr(rv.target, "_schema")
        and len(rv.target._schema.returns) > 1
        and all(str(ret.type) == "Tensor" for ret in rv.target._schema.returns)
    ):
        # for graphs whose result is one node with multiple outputs
        return True
    return False


def make_graph_return_tuple(
    gm: GraphModule,
    inputs: Sequence[InputType],
    compile_gm: Callable[..., Any],
) -> Callable[..., Any]:
    """
    Mutate gm so it returns a tuple.  This is only needed for graphs
    not created by torchdynamo that return non-tuples.
    """
    node = output_node(gm)
    (rv,) = node.args
    rv, spec = pytree.tree_flatten(rv)
    with gm.graph.inserting_before(node):
        gm.graph.output(rv)
    gm.graph.erase_node(node)
    gm.recompile()
    if not graph_returns_tuple(gm):
        raise AssertionError("Expected graph to return a tuple")

    compiled_fn = compile_gm(gm, inputs)

    @functools.wraps(compiled_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return pytree.tree_unflatten(compiled_fn(*args, **kwargs), spec)

    return wrapper


def handle_dynamo_export_graph(
    gm: GraphModule,
    inputs: Sequence[InputType],
    compile_gm: Callable[..., Any],
) -> Callable[..., Any]:
    """
    `torch._dynamo.export` embeds pytrees in the FX graph codegen object,
    convert that to a normal FX graph so inductor can compile it.
    """
    codegen = gm.graph._codegen
    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.recompile()

    compiled_fn = compile_gm(gm, codegen.process_inputs(*inputs))

    @functools.wraps(compiled_fn)  # type: ignore[misc]
    def wrapper(*args: Any) -> Any:
        return codegen.process_outputs(compiled_fn(*codegen.process_inputs(*args)))

    return wrapper


def _check_triton_bf16_support(graph: GraphLowering) -> None:
    def warn_and_skip(device: torch.device | None) -> Never:
        from torch._dynamo.exc import SkipFrame

        if device is None:
            raise AssertionError("Expected device to be set for bf16 support check")
        device_interface = get_interface_for_device(device.type)
        device_props = device_interface.get_device_properties(device)
        warnings.warn(
            f"{device_props.name} does not support bfloat16 compilation natively, skipping"
        )
        raise SkipFrame("BF16 is not supported")

    for node in itertools.chain(graph.graph_inputs.values(), graph.graph_outputs):
        if not isinstance(node, IRNode):
            continue
        device_type = get_device_type(node)
        if (
            not device_type
            or not is_gpu(device_type)
            or node.get_dtype() != torch.bfloat16
        ):
            continue
        device_interface = get_interface_for_device(device_type)
        if device_interface.is_bf16_supported(including_emulation=False):
            return
        warn_and_skip(node.get_device())


def _aoti_flatten_inputs(
    gm: torch.fx.GraphModule,
    args: list[Any] | tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    options: dict[str, Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Flatten the inputs to the graph module and return the flat inputs and options.
    Add "aot_inductor.serialized_in_spec" and "aot_inductor.serialized_out_spec" to the options.
    """
    # pyrefly: ignore [missing-module-attribute]
    from .compile_fx import graph_returns_tuple

    if not graph_returns_tuple(gm):
        raise AssertionError(
            "Graph output must be a tuple(). This is so that we can avoid "
            "pytree processing of the outputs. Please change the module to "
            "have tuple outputs."
        )

    # We will serialize the pytree info into the .so as constant strings
    in_spec = None
    out_spec = None
    if isinstance(gm.graph._codegen, torch.fx.graph._PyTreeCodeGen):
        codegen = gm.graph._codegen
        gm.graph._codegen = torch.fx.graph.CodeGen()
        gm.recompile()

        if codegen.pytree_info.in_spec is not None:
            in_spec = codegen.pytree_info.in_spec
        if codegen.pytree_info.out_spec is not None:
            out_spec = codegen.pytree_info.out_spec

    else:
        if hasattr(gm, "_in_spec"):
            in_spec = gm._in_spec
        if hasattr(gm, "_out_spec"):
            out_spec = gm._out_spec

    serialized_in_spec = pytree.treespec_dumps(in_spec) if in_spec is not None else ""
    serialized_out_spec = (
        pytree.treespec_dumps(out_spec) if out_spec is not None else ""
    )

    flat_args_with_path, received_spec = pytree.tree_flatten_with_path(
        (args, kwargs or {})
    )

    if any(isinstance(x[1], torch.ScriptObject) for x in flat_args_with_path):
        from torch._dynamo.exc import UserError, UserErrorType

        raise UserError(
            UserErrorType.INVALID_INPUT,
            "TorchBind objects found in inputs. TorchBind object inputs are not supported in AOTInductor. "
            "TorchBind objects can only be attributes.",
        )

    # Replace non-tensor (constant) inputs with Nones, since these are not being
    # used anyways by the graph
    flat_example_inputs = [
        x[1] if isinstance(x[1], torch.Tensor) else None for x in flat_args_with_path
    ]

    if in_spec is not None and received_spec != in_spec:
        raise ValueError(
            "Trying to flatten user inputs with exported input tree spec: \n"
            f"{in_spec}\n"
            "but actually got inputs with tree spec of: \n"
            f"{received_spec}"
        )

    options = (
        {
            "aot_inductor.serialized_in_spec": serialized_in_spec,
            "aot_inductor.serialized_out_spec": serialized_out_spec,
        }
        if options is None
        else {
            **options,
            "aot_inductor.serialized_in_spec": serialized_in_spec,
            "aot_inductor.serialized_out_spec": serialized_out_spec,
        }
    )
    return flat_example_inputs, options


def autograd_cache_key(
    graph,
    example_inputs,
    ignore_shape_env: bool,
    decompositions=None,
):
    if config.cpp_wrapper or config.fx_wrapper:
        raise RuntimeError(
            "autograd_cache_key is not supported with cpp_wrapper or fx_wrapper"
        )

    decompositions = (
        decompositions if decompositions is not None else select_decomp_table()
    )
    # compile_fx applies these graph transforms before reaching _compile_fx_main.
    # Neither occurs on the torch.compile/Dynamo path (which always produces
    # tuple-returning, pre-flattened graphs). Not supported by this API.
    if isinstance(graph, GraphModule) and not graph_returns_tuple(graph):
        raise NotImplementedError(
            "autograd_cache_key does not support graphs that don't return a tuple"
        )
    if any(isinstance(x, (list, tuple, dict)) for x in example_inputs):
        raise NotImplementedError(
            "autograd_cache_key does not support nested container inputs"
        )

    compiler_config_extra = create_compiler_config_extra(graph)

    # These context managers replicate the ones that _compile_fx_main sets up
    # before calling aot_autograd, so that the config snapshot captured by
    # autograd_cache_key is identical to a real compile_fx run:
    #   _compile_fx_main outer with-block: _use_lazy_graph_module,
    #       enable_python_dispatcher, preserve_node_meta,
    #       reset_provenance_globals
    #   _compile_fx_main aot_autograd with-block: V.set_fake_mode,
    #       torch._guards.tracing, compiled_autograd._disable,
    #       functorch_config.patch

    fake_mode = detect_fake_mode(example_inputs) or torch._subclasses.FakeTensorMode(
        allow_non_fake_inputs=True
    )
    tracing_context = (
        torch._guards.TracingContext.try_get()
        or torch._guards.TracingContext(fake_mode)
    )

    with (
        functorch_config.patch(
            unlift_effect_tokens=True, selective_decompose=config.selective_decompose
        ),
        _use_lazy_graph_module(dynamo_config.use_lazy_graph_module),
        enable_python_dispatcher(),
        torch.fx.traceback.preserve_node_meta(
            config.effective_provenance_tracking_level() == 1
        ),
        torch._inductor.debug.reset_provenance_globals(),
        V.set_fake_mode(fake_mode),
        torch._guards.tracing(tracing_context),
        compiled_autograd._disable(),
    ):
        return aot_autograd.autograd_cache_key(
            graph,
            example_inputs,
            ignore_shape_env=ignore_shape_env,
            decompositions=decompositions,
            compiler_config_extra=compiler_config_extra,
            keep_inference_input_mutations=True,
        )
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
import functools
import itertools
import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import sympy
from sympy import Expr

from torch import SymInt
from torch.fx.experimental._size_hinting import (
    _guarding_hint_or_throw_base,
    _maybe_realize_expr,
    _optimization_hint_base,
)
from torch.fx.experimental.symbolic_shapes import (
    free_symbols,
    free_unbacked_symbols,
    IterateExprs,
    ShapeEnv,
    SymNode,
)
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import (
    FloorDiv,
    Max,
    Min,
    Mod,
    ModularIndexing,
    safe_gcd,
    simple_floordiv_gcd,
)
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.symbol import symbol_is_type, SymT
from torch.utils._sympy.value_ranges import IntInfinity, ValueRanges

from . import config
from .runtime.runtime_utils import is_power_of_2
from .utils import (
    has_free_symbols,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    sympy_subs,
    VarRanges,
)
from .virtualized import V


log = logging.getLogger(__name__)


Width = int | IntInfinity


@dataclasses.dataclass(frozen=True)
class LaneContiguity:
    """How an index expression varies across lanes.

    This describes only the per-lane relationship, not full memory safety.
    Callers still need to prove the relevant tensor dimension, stride, and base
    alignment make a vector load valid. ``int_oo`` means the expression does not
    impose a finite limit on that width.
    """

    contiguous_width: Width | None = None
    uniform_width: Width | None = None
    stride: int | Expr | None = None
    unknown: bool = False

    @property
    def uniform(self) -> bool:
        return self.uniform_width is not None

    def is_contiguous_for(self, width: int) -> bool:
        return _width_covers(self.contiguous_width, width)

    def is_uniform_for(self, width: int) -> bool:
        return _width_covers(self.uniform_width, width)


def _width_covers(max_width: Width | None, width: int) -> bool:
    return max_width is not None and (max_width == int_oo or max_width >= width)


def _min_width(lhs: Width | None, rhs: Width | None) -> Width | None:
    if lhs is None or rhs is None:
        return None
    if lhs == int_oo:
        return rhs
    if rhs == int_oo:
        return lhs
    return min(lhs, rhs)


def _largest_power_of_2_factor(n: int) -> int:
    return n & -n


# Symbols created by CppTemplateKernel.slice_nd → parse_expr_with_index_symbols
# and are internal to C++ GEMM codegen (not tracked by ShapeEnv).
_GEMM_TEMPLATE_SYMBOL_NAMES = OrderedSet(
    [
        "m_start",
        "m_end",
        "n_start",
        "n_end",
        "k_start",
        "k_end",
        "m_size",
        "n_size",
        "m_offset",
        "m_start_unsliced",
        "m_end_unsliced",
        "m_size_unsliced",
    ]
)


# Threshold above which we skip sympy reasoning that scales poorly in the
# number of symbols. Past this point, polynomial-domain conversions inside
# sympy (e.g. gcd, Mod) dominate compile time on wide concat/sum expressions
# (see https://github.com/sympy/sympy/issues/28200). Chosen empirically from
# AOT-partitioned bwd graphs with ~60-variable shape expressions.
_MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS = 20


@functools.lru_cache
def stride_at(index: sympy.Expr, var: sympy.Symbol):
    if not index.has(var):
        # see test_torchinductor_dynamic_shapes.py::test_full_boolean_dynamic_shapes_cpu
        # which has tmp0 = ops.index_expr(s0 >= 1024, torch.bool) and fails below calculation.
        # in this case, there is no dependencies between index and var.
        return sympy.S.Zero
    replacement = {var: var + 1}
    new_index = sympy_subs(index, replacement)  # type: ignore[arg-type]
    return sympy.simplify(new_index - index)


@functools.lru_cache
def simplify_index_in_vec_range(index: sympy.Expr, var: sympy.Expr, vec_length: int):
    """
    Simplifies the index expression within the range of a vectorized loop.
    Given a vectorized loop variable `var` in the range of a loop with `vec_length`,
    this function transforms the `index` into an equivalent form. It handles
    simplifications for cases where `var` can be expressed as `vec_length * a + b`,
    where `b` ranges from 0 to `vec_length - 1`. The function reduces occurrences
    of `FloorDiv` and `ModularIndexing` in the `index` with best-effort optimizations.

    NOTE:
    The simplified index expression is intended for analysis purposes only, not
    for code generation. It replaces `FloorDiv` and `ModularIndexing` with free variables
    which are not dependent on the loop variable `var` in the vectorized range. Check
    https://github.com/pytorch/pytorch/pull/117221#discussion_r1449746217 for more details.

    Examples:
    1. If `var` is `x3` and `vec_length` is 16, and `x3 = 16*a + b`, then
       `FloorDiv(x3, div)` or `ModularIndexing(x3, div, mod)` becomes a free variable
       when `div` is divisible by 16.
    2. `ModularIndexing(x3, 1, mod)` can be simplified to `x3 + c` where `c` is a free
       variable when `mod` is divisible by 16.
    """

    div_freevar_id = 0
    mod_freevar_id = 0

    def visit_indexing_div(divisor):
        nonlocal div_freevar_id
        result = FloorDiv(var, divisor)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_div_c{div_freevar_id}")
            div_freevar_id += 1
        return result

    def visit_modular_indexing(divisor, modulus):
        nonlocal mod_freevar_id
        result = ModularIndexing(var, divisor, modulus)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        elif divisor == 1 and sympy.gcd(modulus, vec_length) == vec_length:
            result = var + sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        return result

    original_index = index

    div = sympy.Wild("divisor", integer=True)
    if index.has(FloorDiv):
        index = index.replace(FloorDiv(var, div), visit_indexing_div)

    mod = sympy.Wild("modulus", integer=True)
    if index.has(ModularIndexing):
        index = index.replace(ModularIndexing(var, div, mod), visit_modular_indexing)

    if not index.has(sympy.Rel):
        index = sympy.simplify(index)
    if index != original_index:
        return simplify_index_in_vec_range(index, var, vec_length)

    return index


@functools.lru_cache
def stride_at_vec_range(
    index: sympy.Expr, var: sympy.Symbol, vec_length: int | None = None
):
    if vec_length:
        index = simplify_index_in_vec_range(index, var, vec_length)
    return stride_at(index, var)


def statically_known_true(
    shape_env: ShapeEnv,
    expr: sympy.Basic | bool,
    axioms: tuple[sympy.Expr] | None = None,
    var_to_range: tuple[tuple[sympy.Symbol, ValueRanges[Any]]] | None = None,
) -> bool:
    if expr in (True, False):
        return bool(expr)

    # Hint fast-path: if the current concrete hints make `expr` evaluate to
    # False, it cannot be universally True, so bail out without running the
    # much more expensive `_maybe_evaluate_static`. (A True hint doesn't let
    # us conclude universal truth, so we still fall through in that case.)
    try:
        hinted = expr.xreplace(shape_env.backed_var_to_val)
        if hinted is sympy.S.false:
            return False
    except Exception:
        pass

    try:
        simplified = shape_env._maybe_evaluate_static(
            expr,
            axioms=axioms,
            var_to_range=var_to_range,
        )
        if simplified is not None:
            return bool(simplified)
    except Exception:
        log.debug("Could not simplify  %s", expr, exc_info=True)

    return False


# This class is a little awkward, because ShapeEnv is doing most of the heavy
# lifting and in some cases we should be directly passing through to ShapeEnv,
# but there is some extra inductor logic that needs to be handled here
class SizeVarAllocator:
    """
    A class that manages symbolic size variables and their relationships.

    This class works with the ShapeEnv to handle symbolic shape expressions,
    simplify them, and provide utilities for guarding, checking, and evaluating
    symbolic expressions. It also manages precomputed replacements and stride
    calculations for tensor operations.
    """

    def __init__(self, shape_env=None) -> None:
        super().__init__()
        # Note: this can lead to bugs. Reasoning APIs depends on existing information in
        # in the shape_env. For example! var_to_ranges can't be empty!
        if shape_env is None:
            shape_env = ShapeEnv()
        self.shape_env = shape_env
        self.backed_var_to_val = self.shape_env.backed_var_to_val
        self.var_to_hint_override = self.shape_env.var_to_hint_override
        self.replacements: dict[sympy.Symbol, Expr] = self.shape_env.replacements
        self.unbacked_replacements: dict[Expr, Expr] | None = None
        # Maps of dynamic sizes that have to be precomputed on the host to the kernel args.
        # The basic idea is if we have some complicated sympy expression
        # f(s0), we may choose to precompute it on the host and then replace
        # all occurrences of that sympy expression with ps0, so that when we
        # codegen we simply reference ps0 directly without repeating
        # f(s0).  Unlike regular size variables, ps variables cannot be
        # guarded upon; so if we are asked to guard on a Sympy expression
        # which potentially could have already had a precomputed replacement
        # on it, we are obligated to invert the precomputed replacements
        # (inv_precomputed_replacements).
        self.precomputed_replacements: dict[Expr, sympy.Symbol] = {}
        self.inv_precomputed_replacements: dict[sympy.Symbol, Expr] = {}
        self.stride_vars = self.make_stride_vars_cache()
        self.simplify_with_ranges = self.make_simplify_with_ranges_cache()
        self._simplify_loops = self.make_simplify_loops_cache()

    def simplify(self, expr: Expr):
        return sympy.expand(expr).xreplace(self.replacements)

    def make_simplify_with_ranges_cache(self) -> Callable[[Expr, VarRanges], Expr]:
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
        cache: dict[tuple[Any, ...], Expr] = {}
        replacement_count = len(self.replacements)

        def simplify_with_ranges(expr: Expr, var_ranges: VarRanges) -> Expr:
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                # new replacements invalidates cached results
                cache.clear()
                replacement_count = len(self.replacements)
            key = (expr, *var_ranges.items())
            result = cache.get(key)
            if result is None:
                result = self._simplify_with_ranges(expr, var_ranges)
                cache[key] = result
                if result != expr:
                    cache[(result, *var_ranges.items())] = result
            return result

        return simplify_with_ranges

    def make_simplify_loops_cache(self):
        """
        self._simplify_with_ranges() can be expensive, cache its results
        """
        cache: dict[tuple[Any, ...], Any] = {}
        replacement_count = len(self.replacements)

        def simplify_loops(index_vars, sizes, index_formulas):
            nonlocal replacement_count
            if replacement_count != len(self.replacements):
                # new replacements invalidates cached results
                cache.clear()
                replacement_count = len(self.replacements)
            key = (*index_vars, *sizes, *index_formulas)
            result = cache.get(key)
            if result is None:
                result = self._simplify_loops_impl(index_vars, sizes, index_formulas)
                cache[key] = result
            return result

        return simplify_loops

    def _simplify_with_ranges(self, expr: Expr, var_ranges: VarRanges) -> Expr:
        """
        Simplify indexing expression with knowledge of the ranges of
        iteration variables.
        """

        expr = join_dimensions(self.simplify(expr))
        original_expr = expr

        var_to_range = dict(self.shape_env.var_to_range)
        var_to_range.update(
            {
                k: ValueRanges(
                    0, max(0, v - 1) if not has_free_symbols([v]) else IntInfinity()
                )
                for k, v in var_ranges.items()
            }
        )
        for var in expr.free_symbols:
            if var not in var_to_range:
                var_to_range[var] = ValueRanges(0, IntInfinity())

        var_to_range_tuple = cast(
            tuple[tuple[sympy.Symbol, ValueRanges[sympy.Expr]]],
            tuple(var_to_range.items()),
        )

        axioms = []
        for var, upper_bound in var_ranges.items():
            axioms.append(0 <= var)
            axioms.append(var < upper_bound)
        axioms = tuple(axioms) + self.shape_env.get_axioms()

        def statically_known(expr):
            evaluated = self.shape_env._maybe_evaluate_static(
                expr,
                # pyrefly: ignore [bad-argument-type]
                axioms=axioms,
                var_to_range=var_to_range_tuple,
            )
            return bool(evaluated)

        def remove_zero_terms(base, divisor):
            """Symbols smaller than the divisor are zero"""
            if not statically_known(base >= 0):
                return base

            for v in base.free_symbols:
                if v in var_ranges:
                    rest = sympy.Wild("_rest", exclude=[v])
                    m = base.match(v + rest)
                    if m and v not in m[rest].free_symbols:
                        gcd = safe_gcd(m[rest], divisor)
                        if statically_known(v < gcd):
                            base = m[rest]
            return base

        def visit_indexing_div(base, divisor):
            base = remove_zero_terms(base, divisor)
            if statically_known(base >= 0) and statically_known(base < divisor):
                return sympy.S.Zero
            # FloorDiv(ModularIndexing(b, d1, m), d2) = ModularIndexing(b, d1*d2, m//d2)
            if isinstance(base, ModularIndexing) and isinstance(divisor, sympy.Integer):
                b, d1, m = base.args
                if m % divisor == 0:
                    return ModularIndexing(b, d1 * divisor, FloorDiv(m, divisor))
            return FloorDiv(base, divisor)

        def visit_modular_indexing(base, divisor, modulus):
            base = remove_zero_terms(base, divisor)

            if isinstance(base, ModularIndexing):
                inner_base, inner_divisor, inner_modulus = base.args
                period = divisor * modulus
                # Target use cases have constant indices; avoid symbolic compile-time analysis.
                if all(
                    isinstance(value, sympy.Integer) and value > 0
                    for value in (inner_modulus, divisor, modulus)
                ) and self.statically_known_multiple_of(inner_modulus, period):
                    return ModularIndexing(inner_base, inner_divisor * divisor, modulus)

            can_remove_mod = statically_known(base >= 0) and statically_known(
                base < modulus * divisor
            )

            if can_remove_mod:
                return FloorDiv(base, divisor)
            return ModularIndexing(base, divisor, modulus)

        if expr.has(ModularIndexing):
            expr = expr.replace(
                ModularIndexing(
                    sympy.Wild("base", integer=True),
                    sympy.Wild("divisor", integer=True),
                    sympy.Wild("modulus", integer=True),
                ),
                visit_modular_indexing,
            )

        if expr.has(FloorDiv):
            expr = expr.replace(
                FloorDiv(
                    sympy.Wild("base", integer=True),
                    sympy.Wild("divisor", integer=True),
                ),
                visit_indexing_div,
            )

        if expr != original_expr:
            return self._simplify_with_ranges(expr, var_ranges)
        return expr

    def _simplify_loops_impl(
        self, index_vars: list[sympy.Symbol], sizes, index_formulas
    ):
        """
        Try to remove as many axis from loop iterations as possible, by:
            1) removing size==1 dimensions
            2) fuse contiguous dimensions into a single loop
            If channel_last = True, we will prevent the last dim fused with other dims
        """
        sizes = list(map(self.simplify, sizes))

        strides = [
            # index_formulas may contain boolean expressions (e.g. s0 < 10),
            # for which "strides" don't make sense so we ignore them here.
            # NOTE: These expressions may still block merging dims in the sound
            # substitution test performed in can_merge_dims.
            (
                self.stride_vars(x, index_vars)
                if isinstance(x, sympy.Expr)
                else [0] * len(index_vars)
            )
            for x in index_formulas
        ]
        if len(sizes) != len(strides[0]):
            raise AssertionError((len(sizes), len(strides[0])))

        for i in range(len(sizes)):
            if sizes[i] == 1:
                # remove dim
                sizes[i] = None

        def can_merge_dims(a, b):
            for k in range(len(strides)):
                if self.simplify(strides[k][a] * sizes[a]) == self.simplify(
                    strides[k][b]
                ):
                    # approximate test passed, try sound version
                    va = index_vars[a]
                    vb = index_vars[b]
                    m1 = sympy_index_symbol("_merge_tester1")
                    m2 = sympy_index_symbol("_merge_tester2")
                    # NOTE: can't sub vb=0 here in case va * vb appears in the expression,
                    # in which case both expr1 and expr2 would be zero!
                    expr1 = sympy_subs(index_formulas[k], {va: m1 * sizes[a], vb: m2})
                    expr2 = sympy_subs(index_formulas[k], {va: 0, vb: (m1 + m2)})
                    if self.simplify(expr1) == self.simplify(expr2):
                        continue
                return False
            return True

        changed = True
        while changed:
            changed = False
            for i, j in itertools.product(
                reversed(range(len(sizes))), reversed(range(len(sizes)))
            ):
                if i == j or sizes[i] is None or sizes[j] is None:
                    continue
                if can_merge_dims(i, j):
                    changed = True
                    sizes[i] = sizes[i] * sizes[j]
                    sizes[j] = None

        def reindex(index):
            it = list(reversed(index))
            new_index = []
            for size in sizes:
                if size is None:
                    new_index.append(sympy.S.Zero)
                else:
                    new_index.append(it.pop())
            if it:
                raise AssertionError(f"expected all entries consumed, got {it}")
            return new_index

        def prune(index):
            if len(index) != len(sizes):
                raise AssertionError(
                    f"expected len(index) == len(sizes), got {len(index)} != {len(sizes)}"
                )
            return [i for i, s in zip(index, sizes) if s is not None]

        return [x for x in sizes if x is not None], reindex, prune

    # Note - [On Statically Known]
    # The statically_known_* family of functions below NEVER guard, they could return True if the
    # asked questions can be answered without guarding otherwise they return False.
    # Those are similar to statically_known_true in symbolic_shapes.py but operate on sympy
    # expressions instead of symnodes.
    def statically_known_true(self, expr: sympy.Basic | bool) -> bool:
        """
        Returns true if an expression is always true (symbolically or via guards),
        false otherwise. Never add guards, or throw data dependent errors.
        """
        return statically_known_true(self.shape_env, expr)

    def statically_known_equals(self, left: Expr | int, right: Expr | int) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left and right are equal.
        """
        return self.statically_known_true(sympy.Eq(left, right))  # type: ignore[arg-type]

    def statically_known_list_equals(
        self, left: Sequence[Expr], right: Sequence[Expr]
    ) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left and right lists are equal.
        """
        return len(left) == len(right) and all(
            self.statically_known_equals(l, r) for l, r in zip(left, right)
        )

    def statically_known_leq(self, left: Expr, right: Expr | int) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is less than or equal to right.
        """
        expr = left <= right
        return self.statically_known_true(expr)

    def statically_known_geq(self, left: Expr, right: Expr | int) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is greater than or equal to right.
        """
        expr = left >= right
        return self.statically_known_true(expr)

    def statically_known_lt(self, left: Expr, right: Expr | int) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is less than right.
        """
        expr = left < right
        return self.statically_known_true(expr)

    def statically_known_gt(self, left: Expr, right: Expr | int) -> bool:
        """
        Returns a bool indicating if it is sound to optimize as if left is greater than right.
        """
        expr = left > right
        return self.statically_known_true(expr)

    def _is_multiple_of(self, numerator: Expr, denominator: int) -> bool:
        """
        Structural divisibility check: returns True only if numerator is
        provably a multiple of denominator.  Recurses over sympy expression
        structure before falling back to statically_known_true.
        """
        # Rule 1 — concrete value
        if isinstance(numerator, (int, sympy.Integer)):
            return int(numerator) % denominator == 0

        # Rule 2 — product: any factor divisible → product divisible
        if isinstance(numerator, sympy.Mul):
            for factor in numerator.args:
                if self._is_multiple_of(factor, denominator):
                    return True
            const = 1
            for factor in numerator.args:
                if isinstance(factor, (int, sympy.Integer)):
                    const *= int(factor)
            if const != 1 and const % denominator == 0:
                return True

        # Rule 3 — sum: all terms divisible → sum divisible
        if isinstance(numerator, sympy.Add):
            if all(self._is_multiple_of(term, denominator) for term in numerator.args):
                return True

        # Rule 4 — FloorDiv(a, b): if a is multiple of b*n
        if isinstance(numerator, FloorDiv):
            a, b = numerator.args
            if isinstance(b, (int, sympy.Integer)):
                if self._is_multiple_of(a, int(b) * denominator):
                    return True

        # Rule 5 — Mod(a, b): Mod(a,b) = a - b*floor(a/b), so if both a and b
        # are multiples of n, then Mod(a,b) is too.
        if isinstance(numerator, (Mod, sympy.Mod)):
            a, b = numerator.args
            if self._is_multiple_of(a, denominator) and self._is_multiple_of(
                b, denominator
            ):
                return True

        # Rule 6 — cheap gcd check before expensive sympy fallback.
        gcd = simple_floordiv_gcd(numerator, sympy.Integer(denominator))
        if isinstance(gcd, (int, sympy.Integer)) and int(gcd) % denominator == 0:
            return True

        # Rule 7 — full sympy fallback (expensive on many-variable exprs).
        if len(free_symbols(numerator)) > _MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS:
            return False
        expr = sympy.Eq(Mod(numerator, denominator), 0)
        return self.statically_known_true(expr)

    def statically_known_multiple_of(
        self, numerator: Expr, denominator: Expr | int
    ) -> bool:
        """
        Return a bool indicating if it is sound to optimize for the numerator being a multiple of the denominator.
        """
        # Use cheap structural equality, not statically_known_equals —
        # the latter constructs sympy.Eq(wide, wide) which has measurable
        # overhead (~30s on 64-variable expressions).
        if numerator is denominator or numerator == denominator:
            return True

        if isinstance(denominator, (int, sympy.Integer)):
            return self._is_multiple_of(numerator, int(denominator))

        if numerator == 0:
            return True

        # Symbolic denominator: Mod-based reasoning can miss factorable Add
        # denominators even for small expressions, so prove cancellation
        # directly under the same limit used for the sympy fallback below.
        if (
            len(free_symbols([numerator, denominator]))
            > _MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS
        ):
            return False

        def gcd_covers_denominator(gcd: sympy.Basic) -> bool:
            return gcd != 1 and (
                gcd == denominator or self.simplify(gcd - denominator) == 0
            )

        try:
            if gcd_covers_denominator(simple_floordiv_gcd(numerator, denominator)):
                return True
            if gcd_covers_denominator(safe_gcd(numerator, denominator)):
                return True
        except sympy.PolynomialError:
            pass

        expr = sympy.Eq(Mod(numerator, denominator), 0)
        return self.statically_known_true(expr)  # type: ignore[arg-type]

    def statically_known_power_of_2(self, expr: Expr) -> bool:
        """
        Returns a bool indicating if x is known to be a power of 2.
        """
        return isinstance(expr, sympy.Integer) and is_power_of_2(int(expr))

    def analyze_lane_contiguity(
        self, expr: Expr, lane_var: sympy.Symbol
    ) -> LaneContiguity:
        """Analyze whether `expr` is uniform or contiguous over vector lanes.

        The analysis is conservative: unknown cases must fall back to scalar or
        gather-style lowering. It only describes the lane pattern of the index
        expression; callers must still prove tensor stride, offset, dtype, and
        alignment before emitting a vector memory load.
        """
        expr = self.simplify(expr)
        if lane_var not in expr.free_symbols:
            return LaneContiguity(stride=0, uniform_width=int_oo)
        stride = stride_at(expr, lane_var)
        match expr:
            case _ if self.statically_known_equals(stride, 1):
                return LaneContiguity(contiguous_width=int_oo, stride=1)
            case _ if self.statically_known_equals(stride, 0):
                return LaneContiguity(stride=0, uniform_width=int_oo)
            case sympy.Add():
                return self._analyze_lane_contiguity_add(expr.args, lane_var)
            case sympy.Mul():
                return self._analyze_lane_contiguity_mul(expr.args, lane_var)
            case FloorDiv():
                return self._analyze_lane_contiguity_floor_div(
                    expr.args[0], expr.args[1], lane_var
                )
            case ModularIndexing():
                return self._analyze_lane_contiguity_modular_indexing(expr, lane_var)
            case _ if isinstance(expr, (Mod, sympy.Mod)):
                return self._analyze_lane_contiguity_mod(
                    expr.args[0], expr.args[1], lane_var
                )
            case _:
                return LaneContiguity(unknown=True)

    def _analyze_lane_contiguity_add(
        self, args: tuple[Expr, ...], lane_var: sympy.Symbol
    ) -> LaneContiguity:
        """Combine additive terms when at most one term varies across lanes."""
        uniform_width: Width | None = int_oo
        varying_result: LaneContiguity | None = None
        for arg in args:
            arg_result = self.analyze_lane_contiguity(arg, lane_var)
            if arg_result.unknown:
                return LaneContiguity(unknown=True)
            if arg_result.uniform:
                uniform_width = _min_width(uniform_width, arg_result.uniform_width)
            elif varying_result is None:
                varying_result = arg_result
            else:
                return LaneContiguity(unknown=True)
        if varying_result is None:
            return LaneContiguity(stride=0, uniform_width=uniform_width)
        return LaneContiguity(
            contiguous_width=_min_width(varying_result.contiguous_width, uniform_width),
            stride=varying_result.stride,
        )

    def _analyze_lane_contiguity_mul(
        self, args: tuple[Expr, ...], lane_var: sympy.Symbol
    ) -> LaneContiguity:
        """Propagate lane stride through lane-uniform multiplication."""
        uniform_factor: Expr = sympy.S.One
        lane_factors = []
        for arg in args:
            if lane_var in arg.free_symbols:
                lane_factors.append(arg)
            else:
                uniform_factor *= arg
        if len(lane_factors) != 1:
            return LaneContiguity(unknown=True)
        child_result = self.analyze_lane_contiguity(lane_factors[0], lane_var)
        if child_result.unknown or child_result.stride is None:
            return LaneContiguity(unknown=True)
        stride = self.simplify(uniform_factor * child_result.stride)
        if self.statically_known_equals(stride, 0):
            return LaneContiguity(
                stride=0,
                uniform_width=child_result.uniform_width,
            )
        return LaneContiguity(
            contiguous_width=child_result.contiguous_width
            if self.statically_known_equals(stride, 1)
            else None,
            stride=stride,
        )

    def _analyze_lane_contiguity_modular_indexing(
        self, expr: Expr, lane_var: sympy.Symbol
    ) -> LaneContiguity:
        """Analyze ModularIndexing(base, divisor, modulus)."""
        base, divisor, modulus = expr.args
        if not isinstance(divisor, (int, sympy.Integer)):
            return LaneContiguity(unknown=True)
        if int(divisor) != 1:
            return self._analyze_lane_contiguity_floor_div(base, divisor, lane_var)
        return self._analyze_lane_contiguity_mod(base, modulus, lane_var)

    def _analyze_lane_contiguity_floor_div(
        self, base: Expr, divisor: Expr, lane_var: sympy.Symbol
    ) -> LaneContiguity:
        """Return the largest width where ``base // divisor`` is lane-uniform."""
        if not isinstance(divisor, (int, sympy.Integer)):
            return LaneContiguity(unknown=True)
        divisor_int = int(divisor)
        if divisor_int <= 0:
            return LaneContiguity(unknown=True)
        base_result = self.analyze_lane_contiguity(base, lane_var)
        if not base_result.is_contiguous_for(2) or not self.statically_known_equals(
            base_result.stride, 1
        ):
            return LaneContiguity(unknown=True)
        group_start = self.simplify(base.xreplace({lane_var: sympy.Integer(0)}))
        width = _min_width(
            base_result.contiguous_width,
            _largest_power_of_2_factor(divisor_int),
        )
        while isinstance(width, int) and width >= 2:
            if self.statically_known_multiple_of(group_start, width):
                return LaneContiguity(stride=0, uniform_width=width)
            width //= 2
        return LaneContiguity(unknown=True)

    def _analyze_lane_contiguity_mod(
        self,
        base: Expr,
        modulus: Expr,
        lane_var: sympy.Symbol,
    ) -> LaneContiguity:
        """Return the largest aligned no-wrap modulo span.

        `base % modulus` is contiguous for a vector group only when both the
        group start and modulus are multiples of the chosen width. For example,
        lanes 0..3 under `% 4` are contiguous, but lanes 2..5 wrap to 2,3,0,1.
        """
        if not isinstance(modulus, (int, sympy.Integer)):
            return LaneContiguity(unknown=True)
        modulus_int = int(modulus)
        if modulus_int <= 0:
            return LaneContiguity(unknown=True)
        base_result = self.analyze_lane_contiguity(base, lane_var)
        if not base_result.is_contiguous_for(2) or not self.statically_known_equals(
            base_result.stride, 1
        ):
            return LaneContiguity(unknown=True)
        group_start = self.simplify(base.xreplace({lane_var: sympy.Integer(0)}))
        width = _min_width(
            base_result.contiguous_width,
            _largest_power_of_2_factor(modulus_int),
        )
        while isinstance(width, int) and width >= 2:
            if self.statically_known_multiple_of(group_start, width):
                return LaneContiguity(contiguous_width=width, stride=1)
            width //= 2
        return LaneContiguity(unknown=True)

    # The expect/check functions require you to ALREADY KNOW that a particular
    # condition holds. They are similar to expect_true in symbolic_shapes.py and
    # torch.check but operates on sympy expressions instead of symnodes.
    def expect_true(self, expr: Expr) -> bool:
        """
        Use it when you already know that expr is true or should be true and want to
        ensure that guards/runtime assertions are in place to ensure this in compiled
        function. Unlike check, this WON'T raise an error if expr isn't actually true.
        check Note [expect_true].
        """
        if not self.statically_known_true(expr):
            return self.shape_env.guard_or_defer_runtime_assert(
                expr, "sizevars.expect_true"
            )
        return True

    def check(self, expr: Expr) -> None:
        """
        Use it when you already know that expr is true or should be true and want to
        ensure that guards/runtime assertions are in place to ensure this in compiled
        function. Unlike expect_true, this WILL raise an error if expr isn't actually true.
        check Note [expect_true].
        """
        expr = sympy_subs(expr, self.inv_precomputed_replacements)
        if not self.expect_true(expr):
            raise AssertionError(f"expect_true failed for {expr}")

    def check_equals(self, left: Expr, right: Expr) -> None:
        """
        check(sympy.Eq(left, right)).

        """
        self.check(sympy.Eq(left, right))
        return left

    def check_equals_and_simplify(self, left: Expr, right: Expr) -> Expr:
        """
        check(sympy.Eq(left, right)) and returns left after applying
        inv_precomputed_replacements.
        """
        self.check(sympy.Eq(left, right))
        return sympy_subs(left, self.inv_precomputed_replacements)

    def check_leq(self, left: Expr, right: Expr) -> None:
        self.check(sympy.Le(left, right))

    def check_lt(self, left: Expr, right: Expr) -> None:
        self.check(sympy.Lt(left, right))

    # Similar to the functions guard_or_false/guard_or_true in symbolic_shapes.py
    # but operates on sympy expressions instead of symnodes. see Note [guard_or_].
    def guard_or_false(self, left):
        import torch.fx.experimental._config as exp_config

        if exp_config.backed_size_oblivious:
            static_val = self.shape_env._maybe_evaluate_static(left)
            if static_val is not None:
                return static_val
            return False
        return self.evaluate_expr(left, fallback_value=False)

    def guard_or_true(self, left):
        import torch.fx.experimental._config as exp_config

        if exp_config.backed_size_oblivious:
            static_val = self.shape_env._maybe_evaluate_static(left)
            if static_val is not None:
                return static_val
            return True
        return self.evaluate_expr(left, fallback_value=True)

    # The evaluate functions evaluate some symbolic sympy expression
    # (NB: not necessarily an Expr) and return what the concrete result
    # is, guarding on the expression being that result

    # NB: write evaluate_expr(sympy.Lt(a, b)) rather than evaluate_expr(a < b)
    # as this will ensure that you actually have a sympy'ified expression,
    # and will prevent you from incorrectly writing evaluate_expr(a == b)
    # which does the wrong thing if a or b is a sympy expression
    def evaluate_expr(
        self,
        left: Expr | sympy.logic.boolalg.Boolean,
        size_oblivious: bool = False,
        fallback_value: bool | None = None,
    ) -> bool:
        if not isinstance(left, (Expr, sympy.logic.boolalg.Boolean)):
            raise AssertionError(type(left))
        return self.shape_env.evaluate_expr(
            sympy.sympify(left),
            size_oblivious=size_oblivious,
            fallback_value=fallback_value,
        )

    def is_size_one_or_false(self, size: Expr) -> bool:
        """Return True if size equals 1.

        Unbacked symbolic sizes return False without introducing a guard.
        """
        return self.guard_or_false(sympy.Eq(size, 1))

    def evaluate_min(self, left: Expr, right: Expr) -> Expr:
        """Return the smaller of left and right, and guard on that choice."""
        if isinstance(left, Expr):
            left = sympy_subs(left, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        if isinstance(right, Expr):
            right = sympy_subs(right, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        if self.guard_or_false(sympy.Le(left, right)):
            return left
        if self.guard_or_false(sympy.Le(right, left)):
            return right

        # GCD fallback: if gcd(a, b) == a then a divides b, implying a <= b.
        #
        # TODO: This is NOT always sound for unbacked symints.  It can
        # produce wrong results when:
        #   - inputs can be negative: gcd(u0, 10*u0) = u0, returns u0,
        #     but if u0 < 0 then u0 > 10*u0 (e.g. u0=-1: min(-1,-10) = -10)
        #   - a factor can be zero: gcd(u0, u0*u1) = u0, returns u0,
        #     but if u1=0 then u0*u1=0 < u0 (e.g. u0=5,u1=0: min(5,0) = 0)
        # TODO shall we add a runtime assertion at least.
        gcd = sympy.gcd(left, right)
        if left == gcd:
            return left
        if right == gcd:
            return right

        # Min/Max fallback: we can prove Min(a, b) <= c when any arg <= c, but
        # sympy doesn't simplify this yet. So, evaluate it here. Same for Max.
        for lhs, rhs in [(left, right), (right, left)]:

            def le_rhs(a: Expr) -> bool:
                return self.guard_or_false(sympy.Le(a, rhs))

            # Min(Min(a, b), c) ==> Min(a, b) if (a <= c) or (b <= c).
            if isinstance(lhs, (sympy.Min, Min)) and any(le_rhs(a) for a in lhs.args):
                return lhs
            # Min(Max(a, b), c) ==> Max(a, b) if (a <= c) and (b <= c).
            if isinstance(lhs, (sympy.Max, Max)) and all(le_rhs(a) for a in lhs.args):
                return lhs

        raise TypeError(
            f"evaluate_min({left}, {right}) with unbacked symints"
        ) from None

    def evaluate_max(self, left: Expr, right: Expr) -> Expr:
        """Return the larger of left and right, and guard on that choice."""
        # Always choose the opposite of eval min for consistency
        # This means min(a, b) and max(a, b) produce the same guards
        min_val = self.evaluate_min(left, right)
        return right if min_val is left else left

    def guard_int(self, expr: Expr | int) -> int:
        """
        Similar to guard_int in symbolic_shapes.py, except this function works with SymPy
        expressions instead of SymNodes. It extracts the value represented by expr from shapeEnv
        and specialize the compiled graph on it. Raises an error if the result cannot be
        determined due to unhinted or unbacked symbols.
        """
        if isinstance(expr, int):
            return expr
        val = self.guarding_hint_or_throw(expr)
        self.check_equals(expr, sympy.Integer(val))
        return int(val)

    def guard_int_seq(self, left: Sequence[Expr | int]) -> list[int]:
        """
        Apply guard_int on a sequence of inputs.
        """
        return [self.guard_int(x) for x in left]

    def remove_precomputed_replacements(self, expr: Expr) -> Expr:
        if any(symbol_is_type(s, SymT.PRECOMPUTED_SIZE) for s in expr.free_symbols):  # type: ignore[attr-defined]
            return sympy_subs(expr, self.inv_precomputed_replacements)  # type: ignore[arg-type]
        return expr

    def replace_backed_symbols_with_hints(
        self,
        expr: Expr | int,
    ) -> Expr | int:
        """
        Replace all backed symbols in an expression with their concrete hint values.

        This function substitutes backed symbolic variables with their known concrete
        values from the shape environment, while leaving unbacked symbols (data-dependent
        values) untouched. Not recommended to be used except as last resort!
        was added for existing usages compatibility.
        """
        if isinstance(expr, int):
            return expr
        # Substitute all hints into expr, but leave unbacked symints alone
        expr = self.simplify(expr)
        if not isinstance(expr, Expr):
            if not isinstance(expr, int):
                raise AssertionError(f"expected int, got {type(expr)}")
            return expr

        expr = self.remove_precomputed_replacements(expr)
        expr = sympy_subs(expr, self.backed_var_to_val)
        if isinstance(expr, Expr):
            expr = expr.expand(identity=True)

        free_symbols = expr.free_symbols
        if not free_symbols:
            try:
                return int(expr)
            except TypeError:
                return expr  # inf/nan/I

        return expr

    def to_symint_or_int(self, expr: Expr | int) -> SymInt | int:
        """Convert a sympy expression to SymInt, or return int as is."""
        if isinstance(expr, int):
            return expr

        from torch.fx.experimental.sym_node import SymNode

        expr = self.simplify(expr)
        if isinstance(expr, (int, sympy.Integer)):
            return int(expr)

        # Remove precomputed replacements to get canonical form
        # (consistent with symbolic_hint behavior)
        expr = self.remove_precomputed_replacements(expr)

        try:
            hint = self.guarding_hint_or_throw(expr)
        except Exception:
            hint = None
        node = SymNode(expr, self.shape_env, int, hint)
        return SymInt(node)

    def to_symints_or_ints(self, exprs: Sequence[Expr | int]) -> list[SymInt | int]:
        """Convert a sequence of sympy expressions to SymInts, or return ints as is."""
        return [self.to_symint_or_int(e) for e in exprs]

    def guarding_hint_or_throw(self, expr: Expr | int) -> int:
        """
        Return a concrete integer hint for an expression that is safe to use for guarding.

        This method evaluates the expression using only backed-symbols hints. Unlike
        optimization_hint(), this method does NOT use heuristics or fallback values
        for unbacked symbols.

        Use this method when you need a hint value that will be used for guarding decision.

        Args:
            expr: A sympy expression or integer to evaluate.

        Returns:
            The concrete integer value of the expression based on backed symbol hints.

        Raises:
            GuardOnDataDependentSymNode: If the expression contains unbacked symbols
            (data-dependent values) that cannot be resolved to concrete values.

        See Also:
            optimization_hint: For cases where fallback/heuristic values are acceptable
                for unbacked symbols.
        """
        if isinstance(expr, SymNode):
            raise TypeError(
                f"guarding_hint_or_throw expects a sympy Expr or int, not {type(expr)}. "
                "Use expr.expr to extract the sympy expression from a SymNode."
            )
        return _guarding_hint_or_throw_base(
            self.shape_env, expr, self.inv_precomputed_replacements
        )

    def optimization_hint(self, expr: Expr | int, fallback: int | None = None) -> int:
        """
        Return a concrete integer hint for an expression.

        This function should be used for non-guarding based optimizations. If you
        want a hint that you can guard on, use the guarding_hint API instead.

        This function will hint unbacked symbols using user provided optimization
        hints. If not provided, fallback will be used along with some heuristics
        that try to maximize consistency with the shape environment.

        Special cases:
        - Complex numbers (containing sympy.I): raises an error since tensor
          dimensions cannot be complex.
        - Infinity (int_oo, sympy.oo): returns sys.maxsize.
        - NaN (sympy.nan): returns the fallback value.
        """
        return _optimization_hint_base(
            self.shape_env, expr, self.inv_precomputed_replacements, fallback
        )

    def optimization_hints(
        self,
        exprs: Iterable[Expr | int],
        fallback: int | None = None,
    ) -> tuple[int, ...]:
        """
        Like optimization_hint but for a sequence of expressions.
        Returns a tuple of concrete integer hints.
        """
        # Read config at call time to respect runtime patches (e.g., in tests)
        if fallback is None:
            fallback = config.unbacked_symint_fallback
        return tuple(self.optimization_hint(x, fallback=fallback) for x in exprs)

    def all_unbacked_explicitly_hinted(self, exprs: IterateExprs) -> bool:
        """
        Return True if every unbacked symbol in *exprs* has an explicit
        user-provided hint in var_to_hint_override.  If there are no
        unbacked symbols at all, returns True (vacuously).
        *exprs* can be a single expression or any iterable accepted by
        free_unbacked_symbols (list, tuple, etc.).
        """
        unbacked = free_unbacked_symbols(exprs)
        return unbacked.issubset(self.var_to_hint_override.keys())

    def optimization_hint_with_override(
        self,
        expr: Expr | int,
        hint_override: int | None,
    ) -> int:
        r"""Return a concrete integer hint for an expression, with optional override.
        This is used in dynamic dispatch scenarios where callers may want to
        provide a specific hint value rather than computing one from the expression.
        The resolution order is:
        1. If ``expr`` simplifies to a static integer, return that value
           (``hint_override`` is ignored for static shapes).
        2. If ``expr`` is dynamic and ``hint_override`` is not ``None``,
           return ``hint_override``.
        3. Otherwise, compute a hint via :meth:`optimization_hint`
        Args:
            expr (Expr or int): The expression to get a hint for.
            hint_override (int, optional): If provided and ``expr`` is dynamic,
                this value is returned instead of computing a hint.
        Returns:
            int: A concrete integer hint for the expression.
        """
        simplified = _maybe_realize_expr(self.simplify(expr), None)

        if simplified is not None:
            return simplified

        # Dynamic shape: use hint_override if set,
        # else return optimization_hint
        if hint_override is not None:
            return hint_override

        return self.optimization_hint(expr)

    def optimization_hints_with_override(
        self,
        exprs: Iterable[Expr | int],
        hint_override: int | None,
    ) -> tuple[int, ...]:
        """
        Like optimization_hint_with_override but for a sequence of expressions.
        Returns a tuple of concrete integer hints.
        """
        return tuple(
            self.optimization_hint_with_override(e, hint_override) for e in exprs
        )

    def guarding_hints_or_throw(
        self,
        exprs: Iterable[Expr | int],
    ) -> tuple[int, ...]:
        return tuple(self.guarding_hint_or_throw(x) for x in exprs)

    def _lru_cache(self, fn, maxsize=None):
        """
        Wrapper around functools.lru_cache that clears when replacements
        has been invalidated.
        """
        fn_cache = functools.lru_cache(maxsize)(fn)
        prior_len = len(self.replacements)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal prior_len
            if prior_len != len(self.replacements):
                prior_len = len(self.replacements)
                fn_cache.cache_clear()
            return fn_cache(*args, **kwargs)

        return wrapper

    def make_stride_vars_cache(self):
        cache = self._lru_cache(self._stride_vars)

        def stride_vars(
            index: Expr,
            vars: Sequence[sympy.Symbol],
            support_vars: Sequence[sympy.Symbol] | None = None,
        ) -> list[Expr]:
            if not support_vars:
                support_vars = vars
            return cache(index, tuple(vars), tuple(support_vars))

        return stride_vars

    def _stride_vars(
        self,
        index: Expr,
        vars: Sequence[sympy.Symbol],
        support_vars: Sequence[sympy.Symbol],
    ) -> list[Expr]:
        """Convert an indexing expression back into strides

        NOTE: This is only valid if the index is a standard strided offset
        calculation. e.g. 10 * ModularIndexing(i0 + 1, 1, 2) would give a
        stride of -10 because the index wraps around after the first element

        """
        strides = []
        index = self.simplify(index)
        # remove any offset
        index = index - sympy_subs(
            index, {v: sympy.S.Zero for v in support_vars if v != 0}
        )
        for i in range(len(vars)):
            # drop all the other dims
            index_dim = sympy_subs(
                index,
                {
                    support_vars[j]: sympy.S.Zero
                    for j in range(len(support_vars))
                    if vars[i] != support_vars[j] and support_vars[j] != 0
                },
            )
            v = vars[i]
            if v == 0:
                strides.append(sympy.S.Zero)
            else:
                # TODO(jansel): should we use sympy.diff here?
                strides.append(
                    sympy_subs(index_dim, {v: sympy.S.One})
                    - sympy_subs(index_dim, {v: sympy.S.Zero})
                )
        return strides

    def offset_var(self, index: Expr, vars: Sequence[sympy.Symbol]) -> Expr:
        """Extract offset part of an indexing expression"""
        index = self.simplify(index)
        return sympy_subs(index, {v: sympy.S.Zero for v in vars if v != 0})

    # Return stride optimizaitons hints,
    # only used for optimizations.
    def stride_hints(
        self,
        index: Expr,
        vars: Sequence[sympy.Symbol],
        support_vars: Sequence[sympy.Symbol] | None = None,
    ) -> list[int]:
        for v in index.free_symbols:
            if symbol_is_type(v, SymT.INDIRECT):  # type: ignore[attr-defined]
                index = sympy_subs(index, {v: 0})  # type: ignore[dict-item]
        result = []
        for s in self.stride_vars(index, vars, support_vars):
            result.append(self.optimization_hint(s, fallback=0))
        return result

    def stride_order(self, index: Expr, vars: list[sympy.Symbol]) -> list[int]:
        strides = tuple(map(abs, self.stride_hints(index, vars)))
        order = list(range(len(strides)))
        order.sort(key=lambda x: (strides[x] == 0, strides[x]))
        return order

    def lookup_precomputed_size(self, expr: Expr) -> Expr:
        if (
            isinstance(expr, (int, sympy.Symbol, sympy.Number))
            or expr.is_number
            or expr.is_symbol
        ):
            return expr
        expr = self.remove_precomputed_replacements(expr)
        if expr not in self.precomputed_replacements:
            sym = sympy_index_symbol_with_prefix(
                SymT.PRECOMPUTED_SIZE, len(self.precomputed_replacements)
            )
            self.precomputed_replacements[expr] = sym
            self.inv_precomputed_replacements[sym] = expr
        return self.precomputed_replacements[expr]

    def free_symbols(self) -> OrderedSet[sympy.Symbol]:
        return OrderedSet(self.backed_var_to_val.keys()) - OrderedSet(
            self.replacements.keys()
        )

    def combine_modular_indexing_pairs(self, index: sympy.Expr) -> sympy.Expr:
        """
        A pair of special ModularIndexing can be combined.

        E.g. ModularIndexing(ModularIndexing(x, 1, a), 1, b)
        We can simplify this to ModuleIndexing(x, 1, b), if
        1. x is non negative integer
        2. a and b are positive integers
        3. a is a multiple of b.
        """

        def _check_args(x, div, mod, is_first):
            if not isinstance(div, sympy.Integer) or not isinstance(mod, sympy.Integer):
                return False
            if div != 1:
                return False
            if mod <= 0:
                return False

            if is_first:
                # first ModularIndexing should contains a nested ModularIndex
                if not isinstance(x, ModularIndexing):
                    return False
            else:
                # second ModularIndexing should contains a non-negative
                # symbol
                if not isinstance(x, sympy.Symbol) or not self.statically_known_geq(
                    x, 0
                ):
                    return False
            return True

        if isinstance(index, ModularIndexing):
            x, div, mod = index.args

            if not _check_args(x, div, mod, True):
                return index

            x2, div2, mod2 = x.args

            if not _check_args(x2, div2, mod2, False):
                return index

            if Mod(mod2, mod) != 0:
                return index

            return ModularIndexing(x2, 1, mod)

        return index

    def expand_floor_div(
        self,
        index: sympy.Expr,
        candidate_vars: Iterable[sympy.Symbol] | None = None,
    ) -> bool | tuple[sympy.Expr, sympy.Expr]:
        """
        Expand the FloorDiv to the entire expression so that the expression may
        be simplified.

        E.g., for a 2D contiguous tensor with shape [a, 2 * b], and index variables
        x1, x2, index expression 'x1 * 2b + x2' can be easily combined.
        But index expression 'x1 * b + x2 // 2' can not.
        By expanding the FloorDiv to the entire expression, we get
        '(x1 * 2b + x2) // 2'. This transformation allows us to merge loops
        for the numerator!

        Return false if this optimization cannot be applied;
        Return the new expression and the denominator otherwise.
        The original expression will be equivalent to 'new_expression // denominator'
        """
        candidate_vars_set = (
            OrderedSet(candidate_vars) if candidate_vars is not None else None
        )

        def parse_static_term(
            term: sympy.Expr,
        ) -> tuple[sympy.Expr, sympy.Symbol] | None:
            if not isinstance(term, sympy.Mul):
                return None
            # For dynamic shape, term like '2*s1*x1' has 3 child nodes.
            # Without candidate loop vars we cannot tell shape and loop
            # symbols apart, so preserve the historical conservative path.
            if len(term.args) != 2:
                return None
            factor, var = term.args
            if not isinstance(factor, sympy.Integer) or not isinstance(
                var, sympy.Symbol
            ):
                return None
            return factor, var

        def parse_term(term: sympy.Expr) -> tuple[sympy.Expr, sympy.Symbol] | None:
            if candidate_vars_set is None:
                parsed = parse_static_term(term)
                if parsed is None:
                    return None
                factor, var = parsed
            else:
                if isinstance(term, sympy.Symbol):
                    factor = sympy.S.One
                    var = term
                elif isinstance(term, sympy.Mul):
                    vars_in_term = [
                        symbol
                        for symbol in term.free_symbols
                        if symbol in candidate_vars_set
                    ]
                    if len(vars_in_term) == 0:
                        parsed = parse_static_term(term)
                        if parsed is None:
                            return None
                        factor, var = parsed
                    elif len(vars_in_term) == 1:
                        var = vars_in_term[0]
                        factor = sympy.cancel(term / var)
                        if factor.has(*candidate_vars_set):
                            return None
                        if factor.is_integer is not True:
                            return None
                    else:
                        return None
                else:
                    return None

                if var not in candidate_vars_set:
                    return None

            # It's easier to reason about the correctness of the transformation
            # for non-negative integers.
            if not self.statically_known_geq(var, 0):
                return None
            return factor, var

        if not isinstance(index, sympy.Add):
            return False
        terms = index.args

        if len(terms) < 2:
            return False
        floor_div_index = -1
        varlist = []
        factorlist = []
        for idx, term in enumerate(terms):
            if parsed := parse_term(term):
                factor, var = parsed
                varlist.append(var)
                factorlist.append(factor)
            elif isinstance(term, FloorDiv):
                var, factor = term.args
                if not isinstance(factor, sympy.Integer) or not isinstance(
                    var, sympy.Symbol
                ):
                    return False
                if candidate_vars_set is not None and var not in candidate_vars_set:
                    return False
                if not self.statically_known_geq(var, 0):
                    return False
                if floor_div_index >= 0:
                    # can not handle multi FloorDiv yet
                    return False

                floor_div_index = idx
                varlist.append(var)
                # this factor is denominator
                factorlist.append(factor)
            else:
                return False

        if floor_div_index < 0:
            return False

        # Construct the new expression and remember the denominator
        denominator = factorlist[floor_div_index]
        new_index = sympy.S.Zero

        for var, factor, idx in zip(varlist, factorlist, itertools.count()):
            if idx == floor_div_index:
                new_index += var
            else:
                new_index += (factor * denominator) * var

        return new_index, denominator


def join_dimensions(expr: Expr) -> Expr:
    if not isinstance(expr, sympy.Add) or not expr.has(ModularIndexing):
        return expr  # fast exit path
    return _join_dimensions_cached(expr)


@functools.lru_cache(256)
def _join_dimensions_cached(expr: Expr) -> Expr:
    """
    ModularIndexing(i0, 1, 32) + 32 * ModularIndexing(i0, 32, 4)
    becomes
    ModularIndexing(i0, 1, 128)
    ModularIndexing(i0, 1, 32) + 32 * FloorDiv(i0, 32)
    becomes i0


    This type of pattern can come from view operations
    """
    if not isinstance(expr, sympy.Add):
        raise AssertionError(f"expected sympy.Add, got {type(expr)}")

    scale = sympy.Wild("scale", exclude=[0], integer=True)
    base = sympy.Wild("base", integer=True)
    divisor = sympy.Wild("divisor", integer=True)
    mod1 = sympy.Wild("modulus", integer=True)
    mod2 = sympy.Wild("modulus2", integer=True)
    for term1 in expr.args:
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            for term2 in expr.args:
                m2 = term2.match(
                    m1[scale]
                    * m1[mod1]
                    * ModularIndexing(m1[base], m1[divisor] * m1[mod1], mod2)
                )
                if m2 and term1 != term2:
                    expr = join_dimensions(
                        expr
                        - term1
                        - term2
                        + m1[scale]
                        * ModularIndexing(m1[base], m1[divisor], m1[mod1] * m2[mod2])
                    )
                    return expr
    for term1 in expr.args:
        m1 = term1.match(scale * ModularIndexing(base, divisor, mod1))
        if m1:
            for term2 in expr.args:
                m2 = term2.match(
                    m1[scale] * m1[mod1] * FloorDiv(m1[base], m1[divisor] * m1[mod1])
                )
                if m2 is not None:  # in case of success we get an empty dict here
                    expr = join_dimensions(
                        expr
                        - term1
                        - term2
                        + m1[scale] * FloorDiv(m1[base], m1[divisor])
                    )
                    return expr
    return expr


class SimplifyIndexing(V.WrapperHandler):  # type: ignore[name-defined]
    """
    A wrapper around .virtualize.ops that uses var range information to
    simplify ModularIndexing/FloorDiv.
    """

    def __init__(self, inner, var_ranges: VarRanges) -> None:
        super().__init__(inner)
        self.name = "SimplifyIndexing"
        self._simplify: Callable[[Expr], Expr] = (
            lambda index: V.graph.sizevars.simplify_with_ranges(index, var_ranges)
        )

    def load(self, name: str, index: sympy.Expr):
        return self._inner.load(name, self._simplify(index))

    def store(self, name, index, value, mode=None):
        return self._inner.store(name, self._simplify(index), value, mode=mode)

    def store_reduction(self, name, index, value):
        return self._inner.store_reduction(name, self._simplify(index), value)

    def index_expr(self, index, dtype):
        return self._inner.index_expr(self._simplify(index), dtype)

    def value_expr(self, index, dtype):
        return self._inner.value_expr(self._simplify(index), dtype)

    def check_bounds(self, index, size, lower, upper):
        return self._inner.check_bounds(self._simplify(index), size, lower, upper)
<|endoftext|># Backward-compatibility shim.
# This package has been moved to torch._inductor.heuristics.template.
# Keep every old import path working -- both
#   from torch._inductor.template_heuristics import <sub>
# and
#   import torch._inductor.template_heuristics.<sub>
# -- by importing the moved submodules once and aliasing them into this
# package via sys.modules so the same module objects are reused (this avoids
# duplicate imports and double registration of heuristics).
# These re-exports will be removed in a future release.
import sys

from torch._inductor.heuristics.template import (
    aten,
    base,
    contiguous_mm,
    cutedsl,
    decompose_k,
    flex_gemm,
    gemm,
    nv_universal_gemm,
    params,
    registry,
    tlx,
    triton,
    triton_addmm,
)
from torch._inductor.heuristics.template.registry import get_template_heuristic


for _name in (
    "aten",
    "base",
    "contiguous_mm",
    "cutedsl",
    "decompose_k",
    "flex_gemm",
    "gemm",
    "nv_universal_gemm",
    "params",
    "registry",
    "tlx",
    "triton",
    "triton_addmm",
):
    sys.modules[f"{__name__}.{_name}"] = globals()[_name]

del sys, _name
<|endoftext|># mypy: allow-untyped-defs
"""Backend-neutral FX graph and runtime argument helpers for GEMM epilogues."""

import dataclasses
import operator
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

import torch
from torch._inductor import inductor_prims
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from torch.utils._ordered_set import OrderedSet

from .gemm_epilogue_utils import statically_known_shape_equal


@dataclasses.dataclass(frozen=True)
class GemmReductionGeometry:
    """Grouped M/N reduction geometry shared by frontend and backend plans.

    Attributes:
        group: Number of adjacent GEMM output elements in each reduction group.
        axis: GEMM output axis grouped by the reduction, either M (0) or N (1).
    """

    group: int
    axis: int

    def __post_init__(self) -> None:
        if self.group <= 0:
            raise RuntimeError("local_reduce_group must be positive")
        if self.axis not in (0, 1):
            raise RuntimeError("local_reduce_axis must be 0 or 1")

    @property
    def needs_physical_callbacks(self) -> bool:
        return self.axis == 0 or self.group > 32

    @property
    def group_size(self) -> int:
        return self.group

    @classmethod
    def from_output_shape(
        cls, output_shape: Sequence[Any], gemm_shape: Sequence[Any]
    ) -> "GemmReductionGeometry | None":
        if len(output_shape) != 3 or len(gemm_shape) != 2:
            return None
        for axis, group_dim in ((0, 1), (1, 2)):
            try:
                group = V.graph.sizevars.optimization_hint(output_shape[group_dim])
            except (GuardOnDataDependentSymNode, TypeError, ValueError):
                continue
            geometry = cls(group=group, axis=axis)
            if geometry.matches_output_shape(output_shape, gemm_shape):
                return geometry
        return None

    @property
    def reduce_dims(self) -> tuple[int, ...]:
        return (-1, 2) if self.axis == 1 else (-2, 1)

    def matches_reduction_dim(self, dim: Any) -> bool:
        dims = tuple(dim) if isinstance(dim, (list, tuple)) else (dim,)
        return len(dims) == 1 and dims[0] in self.reduce_dims

    def matches_output_shape(
        self, output_shape: Sequence[Any], gemm_shape: Sequence[Any]
    ) -> bool:
        if len(gemm_shape) != 2:
            return False
        m, n = gemm_shape
        grouped = (
            (m, n // self.group, self.group)
            if self.axis == 1
            else (m // self.group, self.group, n)
        )
        return statically_known_shape_equal(
            output_shape, (m, n)
        ) or statically_known_shape_equal(output_shape, grouped)


@dataclasses.dataclass(frozen=True)
class GemmReductionDescriptor:
    """Backend lowering descriptor for a recognized reduction expression.

    Attributes:
        kind: Canonical reduction or normalized-consumer expression name.
        parameters: Compile-time scalar parameters encoded by that expression.
    """

    kind: str
    parameters: tuple[float, ...] = ()

    @classmethod
    def parse(cls, value: str) -> "GemmReductionDescriptor":
        kind, *parameters = value.split(":")
        return cls(kind, tuple(float(parameter) for parameter in parameters))

    def serialize(self) -> str:
        if not self.parameters:
            return self.kind
        return (
            self.kind
            + ":"
            + ":".join(format(parameter, ".17g") for parameter in self.parameters)
        )


@dataclasses.dataclass(frozen=True)
class GemmReductionConfig:
    """Reduction recognized from frontend graph or scheduler loop IR.

    This is an analysis result, before output ownership and feed-main behavior
    are finalized into a :class:`GemmReductionPlan`.

    Attributes:
        output_name: Buffer produced by the recognized reduction.
        group: Number of adjacent GEMM output elements in each reduction group.
        axis: GEMM output axis grouped by the reduction, either M (0) or N (1).
        reduction_type: Reduction or normalized consumer expression to compute.
        source_type: Transformation applied to GEMM accumulator values.
    """

    output_name: str
    group: int
    axis: int
    reduction_type: str
    source_type: str

    @property
    def geometry(self) -> GemmReductionGeometry:
        return GemmReductionGeometry(self.group, self.axis)

    @property
    def contract(self) -> tuple[int, int, str, str]:
        return self.group, self.axis, self.reduction_type, self.source_type


@dataclasses.dataclass(frozen=True)
class GemmReductionPlan:
    """Backend-neutral reduction outputs passed from analysis to codegen.

    Attributes:
        reduction_output: Optional compressed reduction output buffer.
        group: Number of adjacent GEMM output elements in each reduction group.
        axis: GEMM output axis grouped by the reduction, either M (0) or N (1).
        reduction_type: Reduction or normalized consumer expression to compute.
        source_type: Transformation applied to GEMM accumulator values.
        primary_output: Buffer receiving the primary GEMM result.
        feeds_main: Whether the reduction participates in the primary output.
        feed_output: Optional full-shape output consuming the reduction.
        secondary_feed_output: Optional second full-shape reduction consumer.
        secondary_feed_type: Expression implemented by the secondary consumer.
    """

    reduction_output: str | None
    group: int
    axis: int
    reduction_type: str
    source_type: str
    primary_output: str
    feeds_main: bool = False
    feed_output: str | None = None
    secondary_feed_output: str | None = None
    secondary_feed_type: str | None = None

    @property
    def geometry(self) -> GemmReductionGeometry:
        return GemmReductionGeometry(self.group, self.axis)

    @property
    def auxiliary_outputs(self) -> tuple[str, ...]:
        return tuple(
            OrderedSet(
                output
                for output in (
                    self.reduction_output,
                    self.feed_output,
                    self.secondary_feed_output,
                )
                if output is not None and output != self.primary_output
            )
        )


@dataclasses.dataclass(frozen=True)
class GemmReductionArguments:
    """Runtime tensors and compile-time parameters for a grouped GEMM reduction.

    Attributes:
        output: Optional tensor receiving the compressed reduction.
        feed_output: Optional full-shape tensor receiving the reduction consumer.
        secondary_feed_output: Optional second full-shape reduction consumer.
        secondary_feed_type: Expression implemented by ``secondary_feed_output``.
        group: Number of adjacent GEMM output elements in each reduction group.
        axis: GEMM output axis grouped by the reduction, either M (0) or N (1).
        reduction_type: Reduction or normalized consumer expression to compute.
        source_type: Transformation applied to GEMM accumulator values.
        feeds_main: Whether the reduction also produces the primary GEMM output.
    """

    output: Any | None = None
    feed_output: Any | None = None
    secondary_feed_output: Any | None = None
    secondary_feed_type: str | None = None
    group: int = 0
    axis: int = 1
    reduction_type: str = "sum"
    source_type: str = "identity"
    feeds_main: bool = False

    SPECIALIZATION_FIELDS: ClassVar[tuple[str, ...]] = (
        "group",
        "axis",
        "reduction_type",
        "source_type",
        "feeds_main",
        "secondary_feed_type",
    )

    @property
    def enabled(self) -> bool:
        return (
            any(
                value is not None
                for value in (self.output, self.feed_output, self.secondary_feed_output)
            )
            or self.feeds_main
        )

    @property
    def primary_enabled(self) -> bool:
        return (
            self.output is not None or self.feed_output is not None or self.feeds_main
        )

    @property
    def descriptor(self) -> GemmReductionDescriptor:
        return GemmReductionDescriptor.parse(self.reduction_type)

    def tensors(self, attr: str) -> tuple[Any | None, Any | None, Any | None]:
        def tensor(value: Any | None) -> Any | None:
            return getattr(value, attr) if value is not None else None

        return (
            tensor(self.output),
            tensor(self.feed_output),
            tensor(self.secondary_feed_output),
        )


@dataclasses.dataclass(frozen=True)
class NormalizedView:
    """Canonical source and shape for an FX view or reshape."""

    source: torch.fx.Node
    shape: tuple[Any, ...]


@dataclasses.dataclass(frozen=True)
class NormalizedDtypeView:
    """Canonical source and target dtype for a storage reinterpretation."""

    source: torch.fx.Node
    dtype: torch.dtype


@dataclasses.dataclass(frozen=True)
class NormalizedReduction:
    """Canonical arguments for a supported FX reduction."""

    source: torch.fx.Node
    dim: Any
    keepdim: Any
    dtype: Any
    reduction_type: str


@dataclasses.dataclass(frozen=True)
class NormalizedPrepareSoftmax:
    """Canonical source and dimension for online softmax preparation."""

    source: torch.fx.Node
    dim: Any


@dataclasses.dataclass(frozen=True)
class NormalizedSqueeze:
    """Canonical source for an FX squeeze alias."""

    source: torch.fx.Node


@dataclasses.dataclass(frozen=True)
class NormalizedGetItem:
    """Canonical aggregate source and literal FX getitem index."""

    source: torch.fx.Node
    index: int


@dataclasses.dataclass(frozen=True)
class NormalizedSplit:
    """Canonical source, width, and dimension for an FX tensor split."""

    source: torch.fx.Node
    split_size: Any
    dim: int


@dataclasses.dataclass(frozen=True)
class NormalizedSelect:
    """Canonical source, dimension, and index for an FX tensor select."""

    source: torch.fx.Node
    dim: int
    index: Any


@dataclasses.dataclass(frozen=True)
class NormalizedUnsupportedReduction:
    """Canonical source and target for an unsupported FX reduction."""

    source: torch.fx.Node
    target: str


NormalizedNode = (
    NormalizedView
    | NormalizedDtypeView
    | NormalizedReduction
    | NormalizedPrepareSoftmax
    | NormalizedSqueeze
    | NormalizedGetItem
    | NormalizedSplit
    | NormalizedSelect
    | NormalizedUnsupportedReduction
)


FUNCTION_REDUCTION_TYPES = {
    torch.ops.aten.sum.dim_IntList: ("sum", True),
    torch.ops.aten.mean.dim: ("mean", True),
    torch.ops.aten.prod.dim_int: ("prod", True),
    torch.ops.aten.amax.default: ("max", False),
    torch.ops.aten.amin.default: ("min", False),
}

FUNCTION_UNSUPPORTED_REDUCTIONS = frozenset(
    (
        torch.ops.aten.all.dim,
        torch.ops.aten.all.dims,
        torch.ops.aten.all.default,
        torch.ops.aten.any.dim,
        torch.ops.aten.any.dims,
        torch.ops.aten.any.default,
        torch.ops.aten.argmax.default,
        torch.ops.aten.argmin.default,
        torch.ops.aten.std.correction,
        torch.ops.aten.std.dim,
        torch.ops.aten.var.correction,
        torch.ops.aten.var.dim,
    )
)


def normalize_gemm_epilogue_fx_node(node: torch.fx.Node) -> NormalizedNode | None:
    """Return canonical arguments for a selected epilogue FX node."""
    if node.op != "call_function":
        return None
    if node.target is torch.ops.aten.view.dtype:
        source, dtype = node.args
        if not isinstance(source, torch.fx.Node) or not isinstance(dtype, torch.dtype):
            raise AssertionError(
                f"malformed GEMM epilogue dtype view: {node.format_node()}"
            )
        return NormalizedDtypeView(source, dtype)
    if node.target in (
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
    ):
        source = node.args[0]
        shape = node.args[1]
        if not isinstance(source, torch.fx.Node) or not isinstance(
            shape, (tuple, list, torch.Size)
        ):
            raise AssertionError(f"malformed GEMM epilogue view: {node.format_node()}")
        return NormalizedView(
            source,
            tuple(
                arg.meta.get("val", arg) if isinstance(arg, torch.fx.Node) else arg
                for arg in shape
            ),
        )
    if node.target in FUNCTION_REDUCTION_TYPES:
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed GEMM epilogue reduction: {node.format_node()}"
            )
        reduction_type, has_dtype = FUNCTION_REDUCTION_TYPES[node.target]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        keepdim = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        )
        dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
        return NormalizedReduction(
            source,
            dim,
            keepdim,
            dtype if has_dtype else None,
            reduction_type,
        )
    if node.target is inductor_prims.prepare_softmax_online:
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed GEMM epilogue softmax: {node.format_node()}"
            )
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
        return NormalizedPrepareSoftmax(source, dim)
    if node.target is torch.ops.aten.split.Tensor:
        source = node.args[0]
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        if not isinstance(source, torch.fx.Node) or not isinstance(dim, int):
            raise AssertionError(f"malformed GEMM epilogue split: {node.format_node()}")
        return NormalizedSplit(source, node.args[1], dim)
    if node.target is torch.ops.aten.select.int:
        source = node.args[0]
        dim = node.args[1]
        if not isinstance(source, torch.fx.Node) or not isinstance(dim, int):
            raise AssertionError(
                f"malformed GEMM epilogue select: {node.format_node()}"
            )
        return NormalizedSelect(source, dim, node.args[2])
    if node.target in (
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze.dims,
        torch.ops.aten.squeeze.default,
    ):
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed GEMM epilogue squeeze: {node.format_node()}"
            )
        return NormalizedSqueeze(source)
    if node.target is operator.getitem:
        source, index = node.args
        if isinstance(source, torch.fx.Node) and isinstance(index, int):
            return NormalizedGetItem(source, index)
        return None
    if node.target in FUNCTION_UNSUPPORTED_REDUCTIONS:
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            raise AssertionError(
                f"malformed GEMM epilogue reduction: {node.format_node()}"
            )
        return NormalizedUnsupportedReduction(source, str(node.target))
    return None


def iter_fx_node_inputs(value: Any) -> Iterator[torch.fx.Node]:
    """Yield FX node inputs nested in args/kwargs-style containers."""
    result: list[torch.fx.Node] = []
    torch.fx.map_arg(value, lambda node: result.append(node))
    yield from result


@dataclasses.dataclass(frozen=True)
class GemmEpilogueGraph:
    """Index dependencies and canonical interpretations of epilogue FX nodes."""

    dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]]
    normalized_nodes: dict[torch.fx.Node, NormalizedNode]

    @classmethod
    def from_nodes(cls, nodes: Sequence[torch.fx.Node]) -> "GemmEpilogueGraph":
        """Build the FX dependency and normalization indexes in one graph walk."""
        dependencies: dict[torch.fx.Node, frozenset[torch.fx.Node]] = {}
        normalized_nodes: dict[torch.fx.Node, NormalizedNode] = {}
        for node in nodes:
            node_dependencies: OrderedSet[torch.fx.Node] = OrderedSet()
            for input_node in iter_fx_node_inputs((node.args, node.kwargs)):
                node_dependencies.add(input_node)
                node_dependencies.update(dependencies.get(input_node, ()))
            dependencies[node] = frozenset(node_dependencies)
            if (normalized := normalize_gemm_epilogue_fx_node(node)) is not None:
                normalized_nodes[node] = normalized
        return cls(dependencies, normalized_nodes)

    def depends_on(self, value: Any, target: torch.fx.Node) -> bool:
        """Return whether a value is or transitively depends on the target node."""
        return any(
            node is target or target in self.dependencies.get(node, ())
            for node in iter_fx_node_inputs(value)
        )
<|endoftext|># mypy: allow-untyped-defs
"""Lower GEMM epilogue loop IR to shared epilogue contracts."""

import dataclasses
import math
from collections.abc import Sequence
from typing import Any

import sympy

import torch
from torch._inductor.ir import ComputedBuffer
from torch._inductor.kernel.gemm_epilogue import GemmReductionConfig
from torch._inductor.ops_handler import DefaultHandler
from torch._inductor.utils import OrderedSet
from torch._inductor.virtualized import V


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRExpression:
    """Operation captured from a lowered GEMM epilogue loop body.

    Attributes:
        op: Virtualized Inductor operation name.
        args: Captured positional arguments.
        kwargs: Captured keyword arguments in deterministic key order.
    """

    op: str
    args: tuple[Any, ...]
    kwargs: tuple[tuple[str, Any], ...] = ()
    loads: frozenset[str] = frozenset()
    reductions: tuple["GemmEpilogueIRReduction", ...] = ()


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRReduction:
    reduction_type: str
    source: GemmEpilogueIRExpression
    result: int | None = None
    source_type: str | None = None


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRRegion:
    output_name: str
    reductions: tuple[GemmEpilogueIRReduction, ...]
    expression: GemmEpilogueIRExpression

    @property
    def algorithm(self) -> str:
        reduction_types = OrderedSet(
            reduction.reduction_type for reduction in self.reductions
        )
        if "online_softmax_reduce" in reduction_types:
            return "online_softmax"
        if "welford_reduce" in reduction_types:
            return "welford"
        return "generic"


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRStore:
    """Symbolic store produced by replaying a lowered epilogue loop body.

    Attributes:
        index: Symbolic destination index used by the lowered store.
        value: Captured expression written at that index.
    """

    index: sympy.Expr
    value: GemmEpilogueIRExpression


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIROutputRole:
    """Transitive inputs that determine one captured epilogue output.

    Attributes:
        transitive_inputs: All buffers loaded directly or through stored values.
        reduction_inputs: Loaded buffers whose captured values contain reductions.
    """

    transitive_inputs: frozenset[str]
    reduction_inputs: frozenset[str]


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRFinalizer:
    """Normalized operation applied to a completed grouped reduction."""

    output_name: str
    source_name: str
    kind: str


def _expression_values(value: Any):
    if isinstance(value, GemmEpilogueIRExpression):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _expression_values(item)


def _unique_reductions(
    values: Sequence[GemmEpilogueIRExpression],
) -> tuple[GemmEpilogueIRReduction, ...]:
    reductions = []
    seen: OrderedSet[int] = OrderedSet()
    for value in values:
        for reduction in value.reductions:
            if id(reduction) not in seen:
                seen.add(id(reduction))
                reductions.append(reduction)
    return tuple(reductions)


class _GemmEpilogueIRHandler(DefaultHandler):
    def __init__(self) -> None:
        self.stores: dict[str, GemmEpilogueIRStore] = {}

    def _default(
        self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> GemmEpilogueIRExpression:
        values = tuple(_expression_values((*args, *tuple(kwargs.values()))))
        return GemmEpilogueIRExpression(
            name,
            args,
            tuple(sorted(kwargs.items())),
            frozenset().union(*(value.loads for value in values)),
            _unique_reductions(values),
        )

    def indirect_indexing(self, x, size, check=True, wrap_neg=True):
        return sympy.Symbol(f"indirect_{len(self.stores)}", integer=True)

    def load(self, name: str, index: sympy.Expr) -> GemmEpilogueIRExpression:
        stored = self.stores.get(name)
        if stored is None:
            return GemmEpilogueIRExpression(
                "load", (name, index, None), loads=frozenset((name,))
            )
        return GemmEpilogueIRExpression(
            "load",
            (name, index, stored.value),
            loads=frozenset((name,)) | stored.value.loads,
            reductions=stored.value.reductions,
        )

    def reduction(self, dtype, src_dtype, reduction_type, value):
        args = (dtype, src_dtype, reduction_type, value)
        if reduction_type in ("online_softmax_reduce", "welford_reduce"):
            count = 2 if reduction_type == "online_softmax_reduce" else 3
            return tuple(
                GemmEpilogueIRExpression(
                    "reduction",
                    (*args, index),
                    loads=value.loads,
                    reductions=(
                        *value.reductions,
                        GemmEpilogueIRReduction(reduction_type, value, index),
                    ),
                )
                for index in range(count)
            )
        return GemmEpilogueIRExpression(
            "reduction",
            args,
            loads=value.loads,
            reductions=(
                *value.reductions,
                GemmEpilogueIRReduction(reduction_type, value),
            ),
        )

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: GemmEpilogueIRExpression,
        mode=None,
    ) -> None:
        self.stores[name] = GemmEpilogueIRStore(index, value)

    def store_reduction(
        self, name: str, index: sympy.Expr, value: GemmEpilogueIRExpression
    ) -> None:
        self.stores[name] = GemmEpilogueIRStore(index, value)


def _loaded_names(expr: Any) -> frozenset[str]:
    if not isinstance(expr, GemmEpilogueIRExpression):
        return frozenset()
    if expr.op == "load":
        name, _, stored = expr.args
        return frozenset((name,)) | _loaded_names(stored)
    return frozenset().union(*(_loaded_names(arg) for arg in expr.args))


@dataclasses.dataclass(frozen=True)
class GemmEpilogueIRAnalysis:
    """Candidate-wide semantic view of existing lowered loop bodies."""

    stores: dict[str, GemmEpilogueIRStore]

    @classmethod
    def from_buffers(
        cls, buffers: Sequence[ComputedBuffer]
    ) -> "GemmEpilogueIRAnalysis":
        handler = _GemmEpilogueIRHandler()
        with V.set_ops_handler(handler):
            for buffer in buffers:
                buffer.get_store_function()(*buffer.data.inner_fn_args())
        return cls(handler.stores)

    @classmethod
    def store_from_buffer(cls, buffer: ComputedBuffer) -> GemmEpilogueIRStore | None:
        return cls.from_buffers((buffer,)).store(buffer.get_name())

    def store(self, name: str) -> GemmEpilogueIRStore | None:
        return self.stores.get(name)

    def output_role(self, name: str) -> GemmEpilogueIROutputRole | None:
        store = self.store(name)
        if store is None:
            return None
        transitive_inputs = store.value.loads or _loaded_names(store.value)
        return GemmEpilogueIROutputRole(
            transitive_inputs,
            transitive_inputs & self.reduction_stores,
        )

    def grouped_reduction(
        self,
        output_name: str,
        source_name: str,
        group: int,
        axis: int,
        source_dtype: torch.dtype,
    ) -> GemmReductionConfig | None:
        store = self.store(output_name)
        classified = (
            grouped_reduction_ir(store, source_name, group, source_dtype)
            if store is not None
            else None
        )
        if classified is None:
            return None
        reduction_type, source_type = classified
        return GemmReductionConfig(
            output_name, group, axis, reduction_type, source_type
        )

    def reduction_region(
        self,
        output_name: str,
        source_name: str,
        group: int,
        source_dtype: torch.dtype,
    ) -> GemmEpilogueIRRegion | None:
        store = self.store(output_name)
        if store is None:
            return None
        reductions = store.value.reductions or _synthetic_reductions_ir(
            store.value, store.index, source_name, group, source_dtype
        )
        if not reductions:
            return None
        if any(
            (reduction.source.loads or _loaded_names(reduction.source))
            != frozenset((source_name,))
            for reduction in reductions
        ):
            return None
        return GemmEpilogueIRRegion(output_name, reductions, store.value)

    def reduction_finalizer(
        self,
        output_name: str,
        source_name: str,
        group: int | None = None,
    ) -> GemmEpilogueIRFinalizer | None:
        store = self.store(output_name)
        if store is None:
            return None
        if operation_names_ir(store).issubset(
            ("load", "to_dtype", "to_dtype_bitcast", "identity")
        ):
            kind = "identity"
        elif group is not None and single_source_affine_ir(store, source_name) == (
            1.0 / group,
            0.0,
        ):
            kind = "mean"
        elif is_absmax_scale_finalizer_ir(store, source_name):
            kind = "absmax_scale"
        elif (
            store.value.loads == frozenset((source_name,))
            and not store.value.reductions
        ):
            kind = "generic"
        else:
            return None
        return GemmEpilogueIRFinalizer(output_name, source_name, kind)

    @property
    def reduction_stores(self) -> frozenset[str]:
        return frozenset(
            name
            for name, store in self.stores.items()
            if store.value.reductions or _contains_reduction(store.value)
        )


def _constant_value(expr: Any) -> Any | None:
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op in ("constant", "index_expr") and expr.args:
        return expr.args[0]
    if expr.op in ("to_dtype", "to_dtype_bitcast") and expr.args:
        return _constant_value(expr.args[0])
    return None


def _strip_conversions(expr: Any) -> Any:
    while (
        isinstance(expr, GemmEpilogueIRExpression)
        and expr.op in ("to_dtype", "to_dtype_bitcast", "identity")
        and expr.args
    ):
        expr = expr.args[0]
    return expr


def _walk(expr: Any):
    if not isinstance(expr, GemmEpilogueIRExpression):
        return
    yield expr
    for arg in expr.args:
        yield from _walk(arg)


def grouped_reduction_axis_ir(
    reduction: GemmEpilogueIRReduction, group: int, n: int
) -> int | None:
    """Infer grouped M/N geometry from the reduction source load strides."""
    indices = [
        expr.args[1]
        for expr in _walk(reduction.source)
        if expr.op == "load" and len(expr.args) > 1
    ]
    unique_indices = []
    for index in indices:
        if not any(sympy.simplify(index - other) == 0 for other in unique_indices):
            unique_indices.append(index)
    if len(unique_indices) != group:
        return None

    def is_progression(step: int) -> bool:
        return any(
            all(
                any(
                    sympy.simplify(index - (base + offset * step)) == 0
                    for index in unique_indices
                )
                for offset in range(group)
            )
            for base in unique_indices
        )

    axes = [axis for axis, step in ((1, 1), (0, n)) if is_progression(step)]
    return axes[0] if len(axes) == 1 else None


def operation_names_ir(store: GemmEpilogueIRStore) -> frozenset[str]:
    return frozenset(expr.op for expr in _walk(store.value))


def _source_transform(
    expr: Any,
    source_name: str,
    allowed_conversion_dtypes: frozenset[torch.dtype] | None = None,
) -> str | None:
    if allowed_conversion_dtypes is None:
        expr = _strip_conversions(expr)
    else:
        while isinstance(expr, GemmEpilogueIRExpression) and expr.args:
            if expr.op == "identity":
                expr = expr.args[0]
            elif expr.op == "to_dtype":
                if len(expr.args) < 2 or expr.args[1] not in allowed_conversion_dtypes:
                    return None
                expr = expr.args[0]
            elif expr.op == "to_dtype_bitcast":
                return None
            else:
                break
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op == "load":
        return "identity" if expr.args[0] == source_name else None
    if expr.op == "abs":
        return (
            "abs"
            if _source_transform(expr.args[0], source_name, allowed_conversion_dtypes)
            == "identity"
            else None
        )
    if expr.op == "mul" and expr.args[0] == expr.args[1]:
        return (
            "square"
            if _source_transform(expr.args[0], source_name, allowed_conversion_dtypes)
            == "identity"
            else None
        )
    if expr.op == "pow":
        exponent = _constant_value(expr.args[1])
        return (
            "square"
            if exponent == 2
            and _source_transform(expr.args[0], source_name, allowed_conversion_dtypes)
            == "identity"
            else None
        )
    return None


def _flatten_associative(expr: Any, op: str) -> list[Any]:
    stripped = _strip_conversions(expr)
    if isinstance(stripped, GemmEpilogueIRExpression) and stripped.op == op:
        return _flatten_associative(stripped.args[0], op) + _flatten_associative(
            stripped.args[1], op
        )
    return [expr]


def grouped_reduction_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    group: int,
    source_dtype: torch.dtype,
) -> tuple[str, str] | None:
    """Classify a primitive or unrolled grouped reduction loop body."""
    allowed_conversion_dtypes = frozenset((source_dtype, torch.float32))
    candidates = [expr for expr in _walk(store.value) if expr.op == "reduction"]
    matches = []
    for reduction in candidates:
        reduction_type = str(reduction.args[2])
        source_type = _source_transform(
            reduction.args[3], source_name, allowed_conversion_dtypes
        )
        if source_type is not None:
            matches.append((reduction_type, source_type))
    if candidates:
        return matches[0] if len(candidates) == len(matches) == 1 else None

    root = _strip_conversions(store.value)
    while isinstance(root, GemmEpilogueIRExpression) and root.op in (
        "add",
        "mul",
        "truediv",
    ):
        if root.op == "truediv" and _constant_value(root.args[1]) == group:
            terms = _flatten_associative(root.args[0], "add")
            transforms = [
                _source_transform(term, source_name, allowed_conversion_dtypes)
                for term in terms
            ]
            if (
                len(terms) == group
                and len(frozenset(transforms)) == 1
                and transforms[0]
            ):
                return "mean", transforms[0]
        terms = _flatten_associative(root, root.op)
        transforms = [
            _source_transform(term, source_name, allowed_conversion_dtypes)
            for term in terms
        ]
        if len(terms) == group and len(frozenset(transforms)) == 1 and transforms[0]:
            return ("sum" if root.op == "add" else "prod"), transforms[0]
        break
    for op, reduction_type in (("maximum", "max"), ("minimum", "min")):
        terms = _flatten_associative(root, op)
        transforms = [
            _source_transform(term, source_name, allowed_conversion_dtypes)
            for term in terms
        ]
        if len(terms) == group and len(frozenset(transforms)) == 1 and transforms[0]:
            return reduction_type, transforms[0]
    return None


def _synthetic_reductions_ir(
    expr: Any,
    index: sympy.Expr,
    source_name: str,
    group: int,
    source_dtype: torch.dtype,
) -> tuple[GemmEpilogueIRReduction, ...]:
    if not isinstance(expr, GemmEpilogueIRExpression):
        return ()
    classified = grouped_reduction_ir(
        GemmEpilogueIRStore(index, expr), source_name, group, source_dtype
    )
    if classified is not None:
        reduction_type, source_type = classified
        return (GemmEpilogueIRReduction(reduction_type, expr, source_type=source_type),)
    reductions = []
    seen: OrderedSet[int] = OrderedSet()
    for arg in expr.args:
        for value in _expression_values(arg):
            for reduction in _synthetic_reductions_ir(
                value, index, source_name, group, source_dtype
            ):
                if id(reduction.source) not in seen:
                    seen.add(id(reduction.source))
                    reductions.append(reduction)
    return tuple(reductions)


def is_direct_bool_gt_zero_ir(store: GemmEpilogueIRStore, source_name: str) -> bool:
    """Match a direct comparison of one lowered buffer load against zero."""
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op not in ("gt", "ge"):
        return False
    lhs, rhs = map(_strip_conversions, expr.args[:2])
    return (
        isinstance(lhs, GemmEpilogueIRExpression)
        and lhs.op == "load"
        and lhs.args[0] == source_name
        and _constant_value(rhs) == 0
    )


def is_absmax_scale_finalizer_ir(store: GemmEpilogueIRStore, source_name: str) -> bool:
    """Match clamp(source, 1e-12) / 448 used by FP8 absmax scaling."""
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression):
        return False
    if expr.op == "truediv" and _constant_value(expr.args[1]) == 448.0:
        clamped = expr.args[0]
    elif expr.op == "mul":
        lhs, rhs = expr.args[:2]
        if _constant_value(lhs) == 1.0 / 448.0:
            clamped = rhs
        elif _constant_value(rhs) == 1.0 / 448.0:
            clamped = lhs
        else:
            return False
    else:
        return False

    clamped = _strip_conversions(clamped)
    if not isinstance(clamped, GemmEpilogueIRExpression):
        return False
    if clamped.op in ("maximum", "clamp_min"):
        lhs, rhs = clamped.args[:2]
        return (
            _source_transform(lhs, source_name) == "identity"
            and _constant_value(rhs) == 1e-12
        ) or (
            _source_transform(rhs, source_name) == "identity"
            and _constant_value(lhs) == 1e-12
        )
    if clamped.op == "clamp" and len(clamped.args) >= 2:
        return (
            _source_transform(clamped.args[0], source_name) == "identity"
            and _constant_value(clamped.args[1]) == 1e-12
            and (len(clamped.args) < 3 or clamped.args[2] is None)
        )
    return False


def _contains_reduction(expr: Any) -> bool:
    if not isinstance(expr, GemmEpilogueIRExpression):
        return False
    return expr.op == "reduction" or any(_contains_reduction(arg) for arg in expr.args)


def _affine_scale(
    value: tuple[float, float, float], scale: float
) -> tuple[float, float, float]:
    return value[0] * scale, value[1] * scale, value[2] * scale


def _affine_add(
    lhs: tuple[float, float, float], rhs: tuple[float, float, float]
) -> tuple[float, float, float]:
    return lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]


def _affine_coefficients(
    expr: Any, source_name: str, reduction_names: frozenset[str]
) -> tuple[float, float, float] | None:
    expr = _strip_conversions(expr)
    if isinstance(expr, (int, float, sympy.Number)) and not isinstance(expr, bool):
        return 0.0, 0.0, float(expr)
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op == "load":
        name, _, stored = expr.args
        if name == source_name:
            return 1.0, 0.0, 0.0
        if name in reduction_names or _contains_reduction(stored):
            return 0.0, 1.0, 0.0
        return _affine_coefficients(stored, source_name, reduction_names)
    if expr.op == "reduction":
        return 0.0, 1.0, 0.0
    constant = _constant_value(expr)
    if isinstance(constant, (int, float, sympy.Number)) and not isinstance(
        constant, bool
    ):
        return 0.0, 0.0, float(constant)
    if expr.op == "neg":
        value = _affine_coefficients(expr.args[0], source_name, reduction_names)
        return None if value is None else _affine_scale(value, -1.0)
    if expr.op in ("add", "sub"):
        lhs = _affine_coefficients(expr.args[0], source_name, reduction_names)
        rhs = _affine_coefficients(expr.args[1], source_name, reduction_names)
        if lhs is None or rhs is None:
            return None
        scale = 1.0 if expr.op == "add" else -1.0
        return _affine_add(lhs, _affine_scale(rhs, scale))
    if expr.op in ("mul", "truediv"):
        lhs = _affine_coefficients(expr.args[0], source_name, reduction_names)
        rhs = _affine_coefficients(expr.args[1], source_name, reduction_names)
        if lhs is None or rhs is None:
            return None
        if expr.op == "truediv":
            if rhs[:2] != (0.0, 0.0) or rhs[2] == 0.0:
                return None
            return _affine_scale(lhs, 1.0 / rhs[2])
        if lhs[:2] == (0.0, 0.0):
            return _affine_scale(rhs, lhs[2])
        if rhs[:2] == (0.0, 0.0):
            return _affine_scale(lhs, rhs[2])
    if expr.op == "fma":
        lhs = _affine_coefficients(expr.args[0], source_name, reduction_names)
        rhs = _affine_coefficients(expr.args[1], source_name, reduction_names)
        addend = _affine_coefficients(expr.args[2], source_name, reduction_names)
        if lhs is None or rhs is None or addend is None:
            return None
        if lhs[:2] == (0.0, 0.0):
            product = _affine_scale(rhs, lhs[2])
        elif rhs[:2] == (0.0, 0.0):
            product = _affine_scale(lhs, rhs[2])
        else:
            return None
        return _affine_add(product, addend)
    return None


def centered_mean_consumer_type_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    reduction_names: frozenset[str],
    reduction_scale: float = 1.0,
) -> str | None:
    """Classify an affine combination of a lowered value and grouped reduction."""
    coefficients = _affine_coefficients(store.value, source_name, reduction_names)
    if coefficients is not None:
        coefficients = (
            coefficients[0],
            coefficients[1] * reduction_scale,
            coefficients[2],
        )
    if (
        coefficients is None
        or coefficients[0] == 0.0
        or coefficients[1] == 0.0
        or not all(math.isfinite(value) for value in coefficients)
    ):
        return None
    return "mean_linear:" + ":".join(format(value, ".17g") for value in coefficients)


def single_source_affine_ir(
    store: GemmEpilogueIRStore, source_name: str
) -> tuple[float, float] | None:
    """Return scale and bias for an affine expression of one lowered buffer."""
    coefficients = _affine_coefficients(store.value, source_name, frozenset())
    if coefficients is None or coefficients[1] != 0.0:
        return None
    return coefficients[0], coefficients[2]


def _affine_around(expr: Any, basis) -> tuple[float, float] | None:
    expr = _strip_conversions(expr)
    if basis(expr):
        return 1.0, 0.0
    constant = _constant_value(expr)
    if isinstance(constant, (int, float, sympy.Number)) and not isinstance(
        constant, bool
    ):
        return 0.0, float(constant)
    if not isinstance(expr, GemmEpilogueIRExpression):
        return None
    if expr.op == "neg":
        value = _affine_around(expr.args[0], basis)
        return None if value is None else (-value[0], -value[1])
    if expr.op in ("add", "sub"):
        lhs = _affine_around(expr.args[0], basis)
        rhs = _affine_around(expr.args[1], basis)
        if lhs is None or rhs is None:
            return None
        sign = 1.0 if expr.op == "add" else -1.0
        return lhs[0] + sign * rhs[0], lhs[1] + sign * rhs[1]
    if expr.op == "mul":
        lhs = _affine_around(expr.args[0], basis)
        rhs = _affine_around(expr.args[1], basis)
        if lhs is None or rhs is None:
            return None
        if lhs[0] == 0.0:
            return rhs[0] * lhs[1], rhs[1] * lhs[1]
        if rhs[0] == 0.0:
            return lhs[0] * rhs[1], lhs[1] * rhs[1]
    return None


def _is_source(expr: Any, source_name: str) -> bool:
    return _source_transform(expr, source_name) == "identity"


def _sum_terms(expr: Any, source_name: str, group: int) -> bool:
    terms = _flatten_associative(expr, "add")
    return len(terms) == group and all(_is_source(term, source_name) for term in terms)


def _group_max(expr: Any, source_name: str, group: int) -> bool:
    terms = _flatten_associative(expr, "maximum")
    return len(terms) == group and all(_is_source(term, source_name) for term in terms)


def _stable_group_max(expr: Any, source_name: str, group: int) -> bool:
    """Match max(xs), including logsumexp's where(isinf(max), 0, max)."""
    expr = _strip_conversions(expr)
    if _group_max(expr, source_name, group):
        return True
    if not (
        isinstance(expr, GemmEpilogueIRExpression)
        and expr.op == "where"
        and len(expr.args) >= 3
        and _constant_value(expr.args[1]) == 0.0
    ):
        return False
    condition = _strip_conversions(expr.args[0])
    maximum = _strip_conversions(expr.args[2])
    if not (
        isinstance(condition, GemmEpilogueIRExpression)
        and condition.op == "eq"
        and _group_max(maximum, source_name, group)
    ):
        return False
    for absolute, infinity in (condition.args[:2], reversed(condition.args[:2])):
        absolute = _strip_conversions(absolute)
        if (
            isinstance(absolute, GemmEpilogueIRExpression)
            and absolute.op == "abs"
            and _strip_conversions(absolute.args[0]) == maximum
            and _constant_value(infinity) == math.inf
        ):
            return True
    return False


def _shifted_exp(expr: Any, source_name: str) -> Any | None:
    expr = _strip_conversions(expr)
    if not (
        isinstance(expr, GemmEpilogueIRExpression) and expr.op == "exp" and expr.args
    ):
        return None
    shifted = _strip_conversions(expr.args[0])
    if not (
        isinstance(shifted, GemmEpilogueIRExpression)
        and shifted.op == "sub"
        and _is_source(shifted.args[0], source_name)
    ):
        return None
    return _strip_conversions(shifted.args[1])


def is_softmax_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    group: int,
    reduction_names: frozenset[str] = frozenset(),
) -> bool:
    """Match stable grouped softmax, either unrolled or reduction-backed."""
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "truediv":
        return False
    numerator, denominator = expr.args[:2]
    maximum = _shifted_exp(numerator, source_name)
    if maximum is None:
        return False
    denominator = _strip_conversions(denominator)
    if reduction_names:
        return (
            isinstance(maximum, GemmEpilogueIRExpression)
            and maximum.op == "load"
            and maximum.args[0] in reduction_names
            and isinstance(denominator, GemmEpilogueIRExpression)
            and denominator.op == "load"
            and denominator.args[0] in reduction_names
            and maximum.args[0] != denominator.args[0]
        )
    terms = _flatten_associative(denominator, "add")
    return (
        _group_max(maximum, source_name, group)
        and len(terms) == group
        and all(_shifted_exp(term, source_name) == maximum for term in terms)
    )


def is_absmax_normalize_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    scale_names: str | frozenset[str],
) -> bool:
    """Match source * reciprocal(clamp(absmax, 1e-12) / 448)."""
    if isinstance(scale_names, str):
        scale_names = frozenset((scale_names,))
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "mul":
        return False
    for source, reciprocal in (expr.args[:2], reversed(expr.args[:2])):
        reciprocal = _strip_conversions(reciprocal)
        if (
            _is_source(source, source_name)
            and isinstance(reciprocal, GemmEpilogueIRExpression)
            and reciprocal.op == "reciprocal"
            and any(
                is_absmax_scale_finalizer_ir(
                    GemmEpilogueIRStore(store.index, reciprocal.args[0]), scale_name
                )
                for scale_name in scale_names
            )
        ):
            return True
    return False


def _sum_affine_ir(
    expr: Any,
    source_name: str,
    reduction_names: frozenset[str],
    group: int,
) -> tuple[float, float] | None:
    def is_sum(candidate: Any) -> bool:
        candidate = _strip_conversions(candidate)
        return (
            isinstance(candidate, GemmEpilogueIRExpression)
            and candidate.op == "load"
            and candidate.args[0] in reduction_names
        ) or _sum_terms(candidate, source_name, group)

    return _affine_around(expr, is_sum)


def sum_normalize_consumer_type_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    reduction_names: frozenset[str],
    group: int,
) -> str | None:
    """Classify normalization by a primitive or unrolled grouped sum."""
    parameters: tuple[str, float, float] | None = None

    def is_normalization(expr: Any) -> bool:
        nonlocal parameters
        expr = _strip_conversions(expr)
        if not isinstance(expr, GemmEpilogueIRExpression):
            return False
        if expr.op == "truediv":
            lhs, rhs = expr.args[:2]
            if _is_source(lhs, source_name):
                affine = _sum_affine_ir(rhs, source_name, reduction_names, group)
                if affine is not None:
                    parameters = "forward", *affine
                    return True
            if _is_source(rhs, source_name):
                affine = _sum_affine_ir(lhs, source_name, reduction_names, group)
                if affine is not None:
                    parameters = "reverse", *affine
                    return True
        if expr.op == "mul":
            for source, reciprocal in (expr.args[:2], reversed(expr.args[:2])):
                reciprocal = _strip_conversions(reciprocal)
                if (
                    _is_source(source, source_name)
                    and isinstance(reciprocal, GemmEpilogueIRExpression)
                    and reciprocal.op == "reciprocal"
                ):
                    affine = _sum_affine_ir(
                        reciprocal.args[0], source_name, reduction_names, group
                    )
                    if affine is not None:
                        parameters = "forward", *affine
                        return True
        return False

    affine = _affine_around(store.value, is_normalization)
    if (
        affine is None
        or parameters is None
        or affine[0] == 0.0
        or not all(math.isfinite(value) for value in (*affine, *parameters[1:]))
    ):
        return None
    kind = (
        "normalize_sum_affine"
        if parameters[0] == "forward"
        else "normalize_sum_reverse_affine"
    )
    values = (*affine, *parameters[1:])
    return kind + ":" + ":".join(format(value, ".17g") for value in values)


def sum_multiply_consumer_type_ir(
    store: GemmEpilogueIRStore,
    source_name: str,
    reduction_names: frozenset[str],
    group: int,
) -> str | None:
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "mul":
        return None
    for source, reduction in (expr.args[:2], reversed(expr.args[:2])):
        if not _is_source(source, source_name):
            continue
        affine = _sum_affine_ir(reduction, source_name, reduction_names, group)
        if affine is not None and all(math.isfinite(value) for value in affine):
            return "sum_mul_affine:" + ":".join(
                format(value, ".17g") for value in affine
            )
    return None


def variance_parameters_ir(
    store: GemmEpilogueIRStore, source_name: str, group: int
) -> tuple[float, float] | None:
    """Match an unrolled grouped variance followed by an affine transform."""

    def is_variance(expr: Any) -> bool:
        expr = _strip_conversions(expr)
        if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "truediv":
            return False
        if _constant_value(expr.args[1]) != group:
            return False
        squares = _flatten_associative(expr.args[0], "add")
        if len(squares) != group:
            return False
        for square in squares:
            square = _strip_conversions(square)
            if not (
                isinstance(square, GemmEpilogueIRExpression)
                and square.op == "mul"
                and square.args[0] == square.args[1]
            ):
                return False
            centered = _strip_conversions(square.args[0])
            if not (
                isinstance(centered, GemmEpilogueIRExpression)
                and centered.op == "sub"
                and _is_source(centered.args[0], source_name)
            ):
                return False
            mean = _strip_conversions(centered.args[1])
            if not (
                isinstance(mean, GemmEpilogueIRExpression)
                and mean.op == "truediv"
                and _constant_value(mean.args[1]) == group
                and _sum_terms(mean.args[0], source_name, group)
            ):
                return False
        return True

    affine = _affine_around(store.value, is_variance)
    return affine if affine is not None and affine[0] != 0.0 else None


def centered_mean_consumer_type_unrolled_ir(
    store: GemmEpilogueIRStore, source_name: str, group: int
) -> str | None:
    """Classify an affine source/mean expression after a small reduction unroll."""

    def coefficients(expr: Any) -> tuple[float, float, float] | None:
        expr = _strip_conversions(expr)
        if isinstance(expr, GemmEpilogueIRExpression) and expr.op == "truediv":
            if _constant_value(expr.args[1]) == group and _sum_terms(
                expr.args[0], source_name, group
            ):
                return 0.0, 1.0, 0.0
        if _is_source(expr, source_name):
            return 1.0, 0.0, 0.0
        constant = _constant_value(expr)
        if isinstance(constant, (int, float, sympy.Number)) and not isinstance(
            constant, bool
        ):
            return 0.0, 0.0, float(constant)
        if not isinstance(expr, GemmEpilogueIRExpression):
            return None
        if expr.op in ("add", "sub"):
            lhs, rhs = coefficients(expr.args[0]), coefficients(expr.args[1])
            if lhs is None or rhs is None:
                return None
            return _affine_add(
                lhs, _affine_scale(rhs, 1.0 if expr.op == "add" else -1.0)
            )
        if expr.op == "mul":
            lhs, rhs = coefficients(expr.args[0]), coefficients(expr.args[1])
            if lhs is None or rhs is None:
                return None
            if lhs[:2] == (0.0, 0.0):
                return _affine_scale(rhs, lhs[2])
            if rhs[:2] == (0.0, 0.0):
                return _affine_scale(lhs, rhs[2])
        return None

    values = coefficients(store.value)
    if (
        values is None
        or values[0] == 0.0
        or values[1] == 0.0
        or not all(math.isfinite(value) for value in values)
    ):
        return None
    return "mean_linear:" + ":".join(format(value, ".17g") for value in values)


def is_logsumexp_ir(store: GemmEpilogueIRStore, source_name: str, group: int) -> bool:
    """Match max(xs) + log(sum(exp(xs - max(xs))))."""
    expr = _strip_conversions(store.value)
    if not isinstance(expr, GemmEpilogueIRExpression) or expr.op != "add":
        return False
    for maximum, logarithm in (expr.args[:2], reversed(expr.args[:2])):
        maximum = _strip_conversions(maximum)
        logarithm = _strip_conversions(logarithm)
        if not (
            _stable_group_max(maximum, source_name, group)
            and isinstance(logarithm, GemmEpilogueIRExpression)
            and logarithm.op == "log"
        ):
            continue
        terms = _flatten_associative(logarithm.args[0], "add")
        if len(terms) == group and all(
            _shifted_exp(term, source_name) == maximum for term in terms
        ):
            return True
    return False
<|endoftext|># Owner(s): ["module: inductor"]

import contextlib
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast, Protocol
from typing_extensions import NotRequired, TypedDict

import torch
from torch._dynamo.utils import counters
from torch._inductor.codegen.subgraph import SubgraphTemplate
from torch._inductor.ir import (
    Buffer,
    FixedLayout,
    ir_node_to_tensor,
    Layout,
    StorageBox,
    TensorBox,
)
from torch._inductor.lowering import user_lowerings, validate_ir
from torch._inductor.select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
)
from torch._inductor.utils import convert_symint_to_expr
from torch._inductor.virtualized import V


log = logging.getLogger(__name__)

DEFAULT_RANGE_UPPER_BOUND = 65536


class DispatchOnConfig(TypedDict):
    """Range-based dispatch spec for register_custom_op_autotuning(dispatch_on=...)."""

    tensor_name: str
    dim: int
    range_upper_bound: NotRequired[int]


class _AutotuneChoice(Protocol):
    """Winning-choice surface consumed after autotuning.

    Structurally satisfied by ChoiceCaller and its SubgraphChoiceCaller subclass;
    lets callers read choice metadata without getattr fallbacks.
    """

    name: str
    gm: torch.fx.GraphModule | None
    config_patches: dict[str, Any]
    decomposition: Callable[..., Any] | None
    decomposition_kwargs: dict[str, Any]
    layout: Layout

    def output_node(self) -> TensorBox: ...


@dataclass(frozen=True)
class RangeBounds:
    """Inclusive range [start, end] for dimension-based dispatch."""

    start: int
    end: int | float  # float('inf') for unbounded

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError(f"Range start must be >= 1, got {self.start}")
        if self.end != float("inf") and self.start > self.end:
            raise ValueError(f"Invalid range: start={self.start} > end={self.end}")

    def contains(self, value: int) -> bool:
        if self.end == float("inf"):
            return value >= self.start
        return self.start <= value <= int(self.end)

    def __str__(self) -> str:
        end_str = "inf" if self.end == float("inf") else str(int(self.end))
        return f"[{self.start}, {end_str}]"


@dataclass(frozen=True)
class ImplConfig:
    """Implementation config with semantic identity (name + kwargs) for hashing."""

    impl_name: str
    impl_func: Callable[..., Any] = field(compare=False, hash=False, repr=False)
    kwargs: dict[str, Any] = field(default_factory=dict)
    config_patches: dict[str, Any] = field(
        default_factory=dict, compare=False, hash=False, repr=False
    )

    def __hash__(self) -> int:
        return hash((self.impl_name, tuple(sorted(self.kwargs.items()))))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ImplConfig):
            return False
        return self.impl_name == other.impl_name and self.kwargs == other.kwargs

    def __str__(self) -> str:
        if self.kwargs:
            kwargs_str = ", ".join(f"{k}={v}" for k, v in sorted(self.kwargs.items()))
            return f"{self.impl_name}({kwargs_str})"
        return self.impl_name


@dataclass
class RangeImplGroup:
    """Groups non-adjacent ranges using the same implementation."""

    impl_config: ImplConfig
    ranges: list[RangeBounds] = field(default_factory=list)

    def add_range(self, range_bounds: RangeBounds) -> None:
        self.ranges.append(range_bounds)
        self.ranges.sort(key=lambda r: r.start)

    def __str__(self) -> str:
        ranges_str = ", ".join(str(r) for r in self.ranges)
        return f"{self.impl_config.impl_name}: {ranges_str}"

    @property
    def impl_name(self) -> str:
        return self.impl_config.impl_name

    @property
    def impl_func(self) -> Callable[..., Any]:
        return self.impl_config.impl_func

    @property
    def impl_kwargs(self) -> dict[str, Any]:
        return self.impl_config.kwargs

    @property
    def config_patches(self) -> dict[str, Any]:
        return self.impl_config.config_patches


def _detect_collective_ops(choices: list) -> bool:
    """
    Detect if choices contain collective operations.
    """
    from torch._inductor.utils import is_collective_op

    for choice in choices:
        if not hasattr(choice, "gm") or choice.gm is None:
            continue

        for node in choice.gm.graph.nodes:
            if node.op == "call_function" and node.target is not None:
                op_name = str(node.target)

                if is_collective_op(op_name) or is_collective_op(
                    f"torch.ops.{op_name}"
                ):
                    return True

    return False


class CustomOpConfig:
    """Config for custom op autotuning.

    Specifies optional decomposition function with parameter values.
    Each config creates exactly one variant.

    Args:
        decomposition: Optional functions to autotune. If not provided, default will be used.
        config_patches: Optional dict of config patches to apply during kernel codegen
            (e.g., {"coordinate_descent_tuning": True})
        **params: Parameters passed to the function

    Examples:
        CustomOpConfig(attention_impl, head_dim=32, method='chunked')
        CustomOpConfig(head_dim=32, method='chunked')
        CustomOpConfig(decomposition, config_patches={"coordinate_descent_tuning": True})
    """

    def __init__(
        self,
        decomposition: Callable[..., Any] | None = None,
        config_patches: dict[str, Any] | None = None,
        **params: Any,
    ):
        if decomposition is not None and not callable(decomposition):
            raise TypeError(
                f"decomposition must be callable, got {type(decomposition)}"
            )

        self.decomposition = decomposition
        self.config_patches = config_patches or {}
        self.params = params

    def get_decomposition(
        self, default_impl: Callable[..., Any] | None = None
    ) -> Callable[..., Any]:
        """Return the decomposition function for this config.
        When decomposition is not specified, return the default implementation.
        """
        if self.decomposition is not None:
            return self.decomposition

        if default_impl is not None and callable(default_impl):
            return default_impl

        raise TypeError(
            "No decomposition specified in config and no default implementation provided. "
            "Please provide a decomposition function in CustomOpConfig."
        )

    def __repr__(self) -> str:
        decomp_name = self.decomposition.__name__ if self.decomposition else "default"
        if self.params:
            params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
            return f"CustomOpConfig({decomp_name}, {params_str})"
        return f"CustomOpConfig({decomp_name})"


__all__ = [
    "autotune_custom_op",
    "register_custom_op_autotuning",
    "CustomOpConfig",
]


def _extract_tensor_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    op_overload: torch._ops.OpOverload,
) -> tuple[list[Any], dict[str, Any]]:
    """Extract tensor inputs from mixed args/kwargs.
    Separates tensors (for autotuning input_nodes) from non-tensor parameters.

    Args:
        args: Positional arguments (mix of tensors and scalars)
        kwargs: Keyword arguments (mix of tensors and scalars)
        op_overload: Custom Op overload to get parameter names from schema.

    Returns:
        Tuple of (tensor_inputs_list, non_tensor_kwargs)
    """
    tensor_inputs = []
    non_tensor_kwargs = {}

    # Get schema names and extend with fallback names for any extra args
    schema_names = [arg.name for arg in op_overload._schema.arguments]
    param_names = schema_names + [
        f"arg_{i}" for i in range(len(schema_names), len(args))
    ]

    for i, arg in enumerate(args):
        if isinstance(arg, (TensorBox, Buffer, StorageBox)):
            tensor_inputs.append(arg)
        else:
            non_tensor_kwargs[param_names[i]] = arg

    for key, value in kwargs.items():
        if isinstance(value, (TensorBox, Buffer, StorageBox)):
            tensor_inputs.append(value)
        else:
            non_tensor_kwargs[key] = value

    return tensor_inputs, non_tensor_kwargs


def _merge_config_and_runtime_kwargs(
    config_params: dict[str, Any],
    runtime_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Merge config parameters with runtime kwargs. Config params take precedence,
    since they represent the values being autotuned.

    Args:
        config_params: Parameters from CustomOpConfig (autotuning knobs)
        runtime_kwargs: Runtime non-tensor kwargs from _extract_tensor_inputs

    Returns:
        Merged kwargs dictionary with config values taking precedence
    """
    merged_kwargs = runtime_kwargs.copy()
    merged_kwargs.update(config_params)
    return merged_kwargs


def _adapt_user_input_gen_fns(
    inputs: list[Any],
    op_overload: torch._ops.OpOverload,
    user_input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]],
) -> dict[int, Callable[[Any], torch.Tensor]]:
    """Convert user input generators from name-based to index-based format.
    Inductor autotune's input_gen_fns expects index of arg_names as key.
    """
    arg_names = [arg.name for arg in op_overload._schema.arguments]
    name_to_index = {name: i for i, name in enumerate(arg_names)}
    index_based_fns = {}

    for name, gen_fn in user_input_gen_fns.items():
        if name in name_to_index:
            index_based_fns[name_to_index[name]] = gen_fn
        else:
            raise ValueError(
                f"Unknown argument name '{name}' in input_gen_fns. "
                f"Available argument names: {list(name_to_index.keys())}"
            )

    def create_internal_input_gen_fn(
        user_function: Callable[[torch.Tensor], torch.Tensor], arg_name: str
    ) -> Callable[[Any], torch.Tensor]:
        """Create internal input generator that converts IR buffer to user's fake tensor."""

        def internal_input_gen_fn(ir_buffer: Any) -> torch.Tensor:
            fake_tensor = ir_node_to_tensor(ir_buffer, replace_symbols_with_hints=True)
            if fake_tensor is None:
                raise AssertionError("ir_node_to_tensor returned None")
            return user_function(fake_tensor)

        return internal_input_gen_fn

    return {
        i: create_internal_input_gen_fn(
            user_gen_fn, arg_names[i] if i < len(arg_names) else f"arg_{i}"
        )
        for i, user_gen_fn in index_based_fns.items()
        if i < len(inputs)
    }


def _group_ranges_by_impl(
    range_to_best_impl: dict[RangeBounds, ImplConfig],
) -> list[RangeImplGroup]:
    """Group ranges by implementation using semantic identity (name + kwargs)."""
    from torch._inductor import config

    if not range_to_best_impl:
        return []

    # Test mode: skip grouping to force torch.cond dispatch path
    if config.test_configs.force_no_impl_grouping:
        log.info("Test mode: skipping impl grouping, each range is separate group")
        groups = []
        for range_bounds, impl_config in sorted(
            range_to_best_impl.items(), key=lambda x: x[0].start
        ):
            group = RangeImplGroup(impl_config)
            group.add_range(range_bounds)
            groups.append(group)
        return groups

    # Group ranges by impl_config (uses __hash__ and __eq__ based on semantic identity)
    impl_to_group: dict[ImplConfig, RangeImplGroup] = {}

    for range_bounds, impl_config in range_to_best_impl.items():
        if impl_config not in impl_to_group:
            impl_to_group[impl_config] = RangeImplGroup(impl_config)
        impl_to_group[impl_config].add_range(range_bounds)

    # Sort groups by first range start for deterministic codegen
    groups = sorted(impl_to_group.values(), key=lambda g: g.ranges[0].start)

    # Log grouping info
    original_count = len(range_to_best_impl)
    grouped_count = len(groups)

    if grouped_count < original_count:
        log.info(
            "Implementation grouping: reduced from %d ranges to %d impl groups",
            original_count,
            grouped_count,
        )

    return groups


def _create_ranges_from_split_points(
    split_points: list[int],
) -> list[tuple[int, int] | tuple[int, float]]:
    """Convert split points into ranges for autotuning dispatch.

    Example:
        split_points=[512, 2048]
        returns:
               [(1, 512), (513, 2048), (2049, float('inf'))]
    """
    ranges: list[tuple[int, int] | tuple[int, float]] = []
    start = 1

    for split_point in split_points:
        ranges.append((start, split_point))
        start = split_point + 1

    ranges.append((start, float("inf")))

    return ranges


def _create_range_input_gen_fn(
    base_gen_fn: Callable[[torch.Tensor], torch.Tensor],
    dim_index: int,
    range_start: int,
    range_end: int | float,
    range_upper_bound: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create input generator that modifies target dimension to top of range.
    range_upper_bound: Size to use for benchmarking when range_end is unbounded.
    Default is DEFAULT_RANGE_UPPER_BOUND = 65536
    """
    from torch._inductor.ir import get_fill_order
    from torch._inductor.kernel.flex.common import construct_strides

    target_dim = range_upper_bound if range_end == float("inf") else int(range_end)

    def constrained_gen_fn(fake_tensor: torch.Tensor) -> torch.Tensor:
        result = base_gen_fn(fake_tensor)
        shape = list(result.shape)
        shape[dim_index] = target_dim

        # We modified the shape of the result, so we need to recalculate the strides
        # TODO: Refine this to a better way to more directly preserve strides
        fill_order = get_fill_order(result.stride(), shape_env=None)
        new_stride = construct_strides(shape, fill_order)

        storage_size = sum((s - 1) * st for s, st in zip(shape, new_stride)) + 1
        storage = torch.randn(storage_size, dtype=result.dtype, device=result.device)
        return storage.as_strided(shape, tuple(new_stride))

    return constrained_gen_fn


def _default_input_gen_fn(fake_tensor: torch.Tensor) -> torch.Tensor:
    """Default input generator that creates a real tensor matching the fake tensor's shape."""
    return torch.randn(
        fake_tensor.shape, dtype=fake_tensor.dtype, device=fake_tensor.device
    )


def _create_fallback_choice(
    op_overload: torch._ops.OpOverload,
) -> ExternKernelChoice:
    """Create or reuse fallback choice that calls the op eagerly.

    Since kwargs are passed at bind time via maybe_append_choice rather than
    baked into the kernel, the same ExternKernelChoice is reused across
    compilations for the same op_overload.
    """
    fallback_name = (
        f"{op_overload.name().replace('::', '_').replace('.', '_')}_fallback"
    )

    existing = ExternKernelChoice.lookup(fallback_name)
    if existing is not None:
        return existing

    return ExternKernelChoice(
        kernel=op_overload,
        name=fallback_name,
        has_out_variant=False,
        op_overload=op_overload,
        use_fallback_kernel=True,
    )


def autotune_custom_op(
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[torch.fx.Node],
    non_tensor_args: list[dict[str, Any]],
    op_overload: torch._ops.OpOverload,
    user_input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    config_patches_list: list[dict[str, Any]] | None = None,
    min_speedup_threshold: float = 1.0,
    benchmark_with_cudagraphs: bool = False,
) -> tuple[TensorBox, _AutotuneChoice]:
    """Autotune custom operations by comparing multiple decomposition implementations.

    Currently supports SINGLE OUTPUT custom ops only.
    TODO: Add support for multiple output custom ops (tuple/list returns).

    This function generates multiple implementation choices for a custom operation and
    uses Inductor's autotuning system to select the best performing variant at runtime.
    After selecting the best choice, applies inline fusion if the winning choice has a graph.

    Args:
        name: Unique identifier for the autotuning operation
        decompositions: List of alternative implementation functions to benchmark
        inputs: Input tensor IR nodes from compilation (TensorBox/Buffer objects)
        non_tensor_args: List of kwargs dicts, paired with corresponding decompositions arg
        op_overload: OpOverload of the custom op, used as fallback implementation
        user_input_gen_fns: Optional custom input generators for benchmarking.
                           Maps input indices to functions that take fake tensors
                           and return real tensors for performance measurement.

    Returns:
        Tuple of (IR node representing the optimized operation result, winning choice)

    Raises:
        TypeError: If decompositions is not a list/tuple
        RuntimeError: If no inputs or no valid choices generated
    """
    if not isinstance(decompositions, (list, tuple)):
        raise TypeError(
            f"decompositions must be a list or tuple of callables, got {type(decompositions)}"
        )

    if not inputs:
        raise RuntimeError(f"Custom op '{name}' requires tensor inputs for autotuning")

    if len(decompositions) != len(non_tensor_args):
        raise ValueError(
            f"decompositions and non_tensor_args must have same length, "
            f"got {len(decompositions)} decompositions and {len(non_tensor_args)} kwargs"
        )

    # Convert user input generation functions BEFORE creating choices
    input_gen_fns: dict[int, Callable[[Any], torch.Tensor]] = {}
    if user_input_gen_fns:
        input_gen_fns = _adapt_user_input_gen_fns(
            inputs, op_overload, user_input_gen_fns
        )

    template = SubgraphTemplate(name=name)
    choices = template.generate_custom_op_choices(
        name=name,
        decompositions=decompositions,
        # pyrefly: ignore [bad-argument-type, no-matching-overload]
        input_nodes=list(inputs),
        non_tensor_args=non_tensor_args,
        input_gen_fns=input_gen_fns if input_gen_fns else None,
        config_patches_list=config_patches_list,
    )

    # Add fallback choice that calls the op eagerly (not through inductor lowering)
    # This provides a baseline to compare decompositions against
    from torch._inductor import config

    fallback_kwargs = non_tensor_args[0] if non_tensor_args else {}

    with V.fake_mode:
        # pyrefly: ignore [no-matching-overload]
        fake_inputs = [ir_node_to_tensor(inp) for inp in inputs]
        fake_output = op_overload(*fake_inputs, **fallback_kwargs)

    output_size = tuple(convert_symint_to_expr(s) for s in fake_output.shape)
    output_stride = tuple(convert_symint_to_expr(s) for s in fake_output.stride())

    fallback_choice = _create_fallback_choice(op_overload)
    fallback_choice.maybe_append_choice(
        choices=choices,
        input_nodes=list(inputs),
        layout=FixedLayout(
            device=fake_output.device,
            dtype=fake_output.dtype,
            size=output_size,
            stride=output_stride,
        ),
        **fallback_kwargs,
    )

    if not choices:
        raise RuntimeError(f"No valid choices generated for {name}")

    is_collective = _detect_collective_ops(choices)

    # Run autotuning and get both result and winning choice. The selector returns
    # the base ChoiceCaller type, so cast to the structural winning-choice surface.
    selected_result, winning_choice_base = autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=list(inputs),
        layout=choices[0].layout,
        input_gen_fns=input_gen_fns,
        is_collective=is_collective,
        min_speedup_threshold=min_speedup_threshold,
        benchmark_with_cudagraphs=benchmark_with_cudagraphs,
    )
    winning_choice: _AutotuneChoice = cast(_AutotuneChoice, winning_choice_base)

    # Test mode: force specific choice to win
    force_choice = config.test_configs.force_custom_op_decomposition
    if force_choice is True and winning_choice.gm is None:
        # Force decomposition: pick first choice with a graph
        for choice in choices:
            if choice.gm is not None:
                log.info(
                    "Test mode: forcing decomposition %s over fallback",
                    choice.name,
                )
                winning_choice = choice
                selected_result = choice.output_node()
                break
    elif force_choice is False and winning_choice.gm is not None:
        # Force fallback: pick first choice without a graph
        for choice in choices:
            if choice.gm is None:
                log.info(
                    "Test mode: forcing fallback %s over decomposition",
                    choice.name,
                )
                winning_choice = choice
                selected_result = choice.output_node()
                break

        # Always inline when winning_choice has a graph; callers extract choice metadata separately
    if winning_choice.gm is not None:
        log.debug(
            "Inlining winning choice: %s (name=%s)",
            winning_choice.name,
            name,
        )
        from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes

        ops_before = len(V.graph.operations)
        result = inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)

        # Tag inlined operations with config_patches from the winning choice
        config_patches = winning_choice.config_patches
        if config_patches:
            for op in V.graph.operations[ops_before:]:
                op.set_config_patches(config_patches.copy())

        return result, winning_choice

    log.debug(
        "Winning choice does not support inlining: %s (name=%s)",
        winning_choice.name,
        name,
    )
    return selected_result, winning_choice


def _generate_dynamic_configs(
    tensor_inputs: list[Buffer],
    config_generator: Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]],
    op_overload: torch._ops.OpOverload,
    operation_name: str,
) -> list[CustomOpConfig]:
    """Generate configs dynamically based on input tensors at lowering time."""
    # Get parameter names from op schema instead of impl signature
    schema = op_overload._schema
    param_names = [arg.name for arg in schema.arguments if not arg.kwarg_only]

    with V.fake_mode:
        fake_tensors = [ir_node_to_tensor(inp) for inp in tensor_inputs]

    fake_tensors_dict = dict(zip(param_names, fake_tensors))

    configs = config_generator(fake_tensors_dict)

    if not isinstance(configs, (list, tuple)):
        raise TypeError(
            f"config_generator must return a list or tuple of CustomOpConfig, "
            f"got {type(configs)}"
        )
    if not configs:
        log.info(
            "config_generator returned empty list for %s, will use default lowering",
            operation_name,
        )
        return []

    return list(configs)


def _prepare_configs_and_decompositions(
    processed_configs: list[CustomOpConfig] | None,
    config_generator: Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]] | None,
    tensor_inputs: list[Any],
    default_impl: Callable[..., Any],
    op_overload: torch._ops.OpOverload,
    runtime_kwargs: dict[str, Any],
    name: str,
) -> tuple[list[Callable], list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare decompositions and merged kwargs from configs.

    Handles both static configs and dynamic config generation.
    Merges config params with runtime kwargs (runtime takes precedence).
    """
    # Get configs: either generate dynamically or use static configs
    if config_generator is not None:
        configs_to_use = _generate_dynamic_configs(
            tensor_inputs, config_generator, op_overload, name
        )
    else:
        if processed_configs is None:
            raise AssertionError("processed_configs must not be None")
        configs_to_use = processed_configs

    # Prepare decompositions and kwargs for autotuning
    decompositions = []
    non_tensor_args = []
    config_patches_list = []

    for cfg in configs_to_use:
        decomp = cfg.get_decomposition(default_impl=default_impl)
        decompositions.append(decomp)

        # Merge config params with runtime kwargs (runtime takes precedence)
        merged_kwargs = _merge_config_and_runtime_kwargs(cfg.params, runtime_kwargs)
        non_tensor_args.append(merged_kwargs)

        # Collect config_patches for each config
        config_patches_list.append(cfg.config_patches)

    return decompositions, non_tensor_args, config_patches_list


def _standard_lowering_fn(
    processed_configs: list[CustomOpConfig],
    default_impl: Callable[..., Any],
    name: str,
    op_overload: torch._ops.OpOverload,
    input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None,
    tensor_inputs: list[Any],
    runtime_kwargs: dict[str, Any],
    config_generator: Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    | None = None,
    min_speedup_threshold: float = 1.0,
    benchmark_with_cudagraphs: bool = False,
) -> Any:
    """Standard autotuning lowering function.

    Returns None if no configs/decompositions available, signaling caller to
    use normal lowering.
    """
    decompositions, non_tensor_args, config_patches_list = (
        _prepare_configs_and_decompositions(
            processed_configs,
            config_generator,
            tensor_inputs,
            default_impl,
            op_overload,
            runtime_kwargs,
            name,
        )
    )

    # If no decompositions, signal caller to use normal lowering
    if not decompositions:
        return None

    result, _ = autotune_custom_op(
        name=name,
        decompositions=decompositions,
        inputs=tensor_inputs,
        non_tensor_args=non_tensor_args,
        config_patches_list=config_patches_list,
        op_overload=op_overload,
        user_input_gen_fns=input_gen_fns,
        min_speedup_threshold=min_speedup_threshold,
        benchmark_with_cudagraphs=benchmark_with_cudagraphs,
    )

    validate_ir(result)
    return result


def _apply_config_patches_recursive(
    operations: list,
    config_patches: dict[str, Any],
) -> None:
    """Apply config_patches to operations, including those inside subgraphs."""
    for op in operations:
        if hasattr(op, "set_config_patches"):
            op.set_config_patches(config_patches.copy())

        # Recurse into any subgraphs (Switch, WhileLoop, InvokeSubgraph, etc.)
        for subgraph in op.get_subgraphs():
            if subgraph.graph:
                _apply_config_patches_recursive(
                    subgraph.graph.operations, config_patches
                )


def _lower_single_impl(
    impl: Callable[..., Any],
    impl_kwargs: dict[str, Any],
    runtime_kwargs: dict[str, Any],
    tensor_inputs: list[Any],
    name: str,
    config_patches: dict[str, Any] | None = None,
) -> Any:
    """Lower a single implementation by tracing and inlining it.

    Uses error_on_new_guards() during tracing to detect if the impl adds guards.
    Returns None if the impl adds guards, signaling caller to skip this choice.
    """
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.fx.experimental.symbolic_shapes import _ShapeEnvGuardError

    from ..decomposition import select_decomp_table

    merged_kwargs = _merge_config_and_runtime_kwargs(impl_kwargs, runtime_kwargs)

    def impl_wrapper(*tensors):
        return impl(*tensors, **merged_kwargs)

    shape_env = V.fake_mode.shape_env

    with V.fake_mode:
        fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
        decomposition_table = select_decomp_table()

        context = (
            shape_env.error_on_new_guards
            if shape_env is not None
            else contextlib.nullcontext
        )
        try:
            with context():
                impl_gm = make_fx(
                    impl_wrapper,
                    decomposition_table=decomposition_table,
                    tracing_mode="symbolic",
                )(*fake_inputs)
        except (_ShapeEnvGuardError, AssertionError) as e:
            is_guard_error = isinstance(e, _ShapeEnvGuardError) or (
                isinstance(e, AssertionError)
                and "Guard attempted while ShapeEnv guards are frozen" in str(e)
            )
            if not is_guard_error:
                raise
            log.info(
                "Implementation %s adds guards, skipping custom op lowering",
                impl.__name__,
            )
            counters["inductor"]["custom_op_decomp_guard_skips"] += 1
            return None

    log.info("Inlining implementation: %s", impl.__name__)
    ops_before = len(V.graph.operations)
    result = inline_subgraph_to_ir_nodes(impl_gm, tensor_inputs, name)

    if config_patches:
        _apply_config_patches_recursive(V.graph.operations[ops_before:], config_patches)

    validate_ir(result)
    return result


def _range_based_lowering_fn(
    processed_configs: list[CustomOpConfig],
    default_impl: Callable[..., Any],
    name: str,
    op_overload: torch._ops.OpOverload,
    input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None,
    tensor_name: str,
    dim_index: int,
    ranges: list[tuple[int, int | float]],
    tensor_inputs: list[Any],
    runtime_kwargs: dict[str, Any],
    range_upper_bound: int,
    config_generator: Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    | None = None,
    min_speedup_threshold: float = 1.0,
    benchmark_with_cudagraphs: bool = False,
) -> Any:
    """Range-based autotuning lowering function."""
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes
    from torch.fx.experimental.proxy_tensor import make_fx

    from ..decomposition import select_decomp_table

    log.info("=== Range-based Autotuning for %s ===", name)
    log.info("Dispatch on: %s[%d], Ranges: %s", tensor_name, dim_index, ranges)

    decompositions, non_tensor_args, config_patches_list = (
        _prepare_configs_and_decompositions(
            processed_configs,
            config_generator,
            tensor_inputs,
            default_impl,
            op_overload,
            runtime_kwargs,
            name,
        )
    )

    range_to_best_impl_map: dict[RangeBounds, ImplConfig] = {}

    # Benchmark each range and collect winning implementations
    for range_start, range_end in ranges:
        if input_gen_fns and tensor_name in input_gen_fns:
            base_gen_fn = input_gen_fns[tensor_name]
        else:
            base_gen_fn = _default_input_gen_fn

        range_gen_fn = _create_range_input_gen_fn(
            base_gen_fn, dim_index, range_start, range_end, range_upper_bound
        )
        range_input_gen_fns = {**(input_gen_fns or {}), tensor_name: range_gen_fn}

        range_name = f"{name}_range_{int(range_start)}_{int(range_end) if range_end != float('inf') else 'inf'}"

        # pyrefly: ignore [not-iterable]
        autotuned_result, winning_choice = autotune_custom_op(
            name=range_name,
            decompositions=decompositions,
            inputs=tensor_inputs,
            non_tensor_args=non_tensor_args,
            op_overload=op_overload,
            user_input_gen_fns=range_input_gen_fns,
            min_speedup_threshold=min_speedup_threshold,
            benchmark_with_cudagraphs=benchmark_with_cudagraphs,
            config_patches_list=config_patches_list,
        )

        if winning_choice.decomposition is not None:
            winning_impl = winning_choice.decomposition
            winning_kwargs = winning_choice.decomposition_kwargs
        else:
            # Fallback was selected (ExternKernelCaller)
            winning_impl = default_impl
            winning_kwargs = non_tensor_args[0] if non_tensor_args else {}
            log.info(
                "   Range [%s, %s]: Fallback (default_impl) selected",
                range_start,
                range_end if range_end != float("inf") else "inf",
            )

        winning_config_patches = winning_choice.config_patches or {}

        # Create dataclass instances for cleaner code
        range_bounds = RangeBounds(range_start, range_end)
        impl_config = ImplConfig(
            impl_name=winning_impl.__name__,
            impl_func=winning_impl,
            kwargs=winning_kwargs,
            config_patches=winning_config_patches,
        )
        range_to_best_impl_map[range_bounds] = impl_config

        log.info(
            "   Range %s -> %s",
            range_bounds,
            impl_config.impl_name,
        )

    # Group ranges by implementation
    from torch.fx.experimental.symbolic_shapes import _ShapeEnvGuardError

    impl_groups = _group_ranges_by_impl(range_to_best_impl_map)

    log.info("After grouping by implementation: %d impl groups", len(impl_groups))
    for group in impl_groups:
        log.info("   %s", group)

    # If only one impl group remains, just inline that implementation
    if len(impl_groups) == 1:
        group = impl_groups[0]
        log.info("Only one implementation after grouping, directly inlining")
        return _lower_single_impl(
            group.impl_func,
            group.impl_kwargs,
            runtime_kwargs,
            tensor_inputs,
            name,
            config_patches=group.config_patches,
        )

    def dispatch_fn(*fake_tensors):
        """Build nested torch.cond dispatch: cond(pred1, impl1, cond(pred2, impl2, ...))."""
        num_impl_groups = len(impl_groups)
        if num_impl_groups < 2:
            raise RuntimeError(
                f"dispatch_fn requires at least 2 impl groups, got {num_impl_groups}"
            )

        dim_value = fake_tensors[0].size(dim_index)

        def build_range_predicate(ranges_list: list[RangeBounds]) -> torch.Tensor:
            predicates = []
            for rb in ranges_list:
                end = int(rb.end) if rb.end != float("inf") else None
                if end is None:
                    predicates.append(dim_value >= rb.start)
                else:
                    predicates.append((dim_value >= rb.start) & (dim_value <= end))

            result = predicates[0]
            for pred in predicates[1:]:
                result = result | pred
            return result  # pyrefly: ignore [bad-return]

        def build_nested_cond(idx: int):
            if idx >= num_impl_groups:
