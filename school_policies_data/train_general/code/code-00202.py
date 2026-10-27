    Saves the subgraph, i.e. original callable, in the forward method. And then
    traces out a joint graph in the backward. This delaying of tracing in
    backward, also called as lazy backward, ensures that the assumptions about
    the grad_out strides and tensor-subclass-ness are already accounted for.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        subgraph,
        identifier,
        output_metadata,
        *operands,
    ):
        # We want to delay the backward graph construction until the backward.
        # So in forward, we just run the fw callable as is. And save all the
        # information necessary to construct the backward graph in the ctx.
        ctx._subgraph = subgraph
        ctx._identifier = identifier
        ctx._output_metadata = output_metadata
        ctx._call_id = _next_invoke_subgraph_call_id()
        # We snapshot the dispatch keys in forward for materializing
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()

        save_values_for_backward(ctx, operands)

        with (
            torch._C._AutoDispatchBelowAutograd(),
            _set_invoke_subgraph_call_id(ctx._call_id),
        ):
            out = invoke_subgraph(
                subgraph,
                f"fw_{identifier}",
                *operands,
            )

        # Check that int (coming from symint) is at expected indexes.
        for idx, o in enumerate(out):
            if isinstance(o, int):
                if idx not in output_metadata.indexes_with_symint:
                    raise AssertionError(
                        f"unexpected int output at index {idx}, not in indexes_with_symint"
                    )

        # Propagate non-differentiable status from the inner subgraph.
        # See get_non_differentiable_indices for details on why this is needed.
        non_differentiable_indices = get_non_differentiable_indices(subgraph)
        if non_differentiable_indices:
            non_differentiable = [
                out[i]
                for i in non_differentiable_indices
                if isinstance(out[i], torch.Tensor)
            ]
            if non_differentiable:
                ctx.mark_non_differentiable(*non_differentiable)
        ctx._non_differentiable_indices = non_differentiable_indices

        return out

    @staticmethod
    def backward(
        ctx,
        *grad_outs,
    ):
        from torch._dynamo.utils import dynamo_timed

        subgraph = ctx._subgraph
        identifier = ctx._identifier
        output_metadata = ctx._output_metadata
        primals = saved_values(ctx)

        # Filter out grads that are None or do not require_grad. This was
        # the assumption we made during the tracing of joint_graph.
        non_differentiable_indices = ctx._non_differentiable_indices
        filtered_grad_outs = []
        for idx, o in enumerate(grad_outs):
            if o is None:
                if (
                    idx not in output_metadata.indexes_with_symint
                    and idx not in non_differentiable_indices
                ):
                    raise AssertionError(
                        f"unexpected None grad_out at index {idx}, not in indexes_with_symint"
                    )
            elif (
                idx in output_metadata.indexes_with_no_grad
                or idx in non_differentiable_indices
            ):
                # Deliberately skip over the grad_outs which we know should be
                # None because the corresponding fwd_out does not require_grad.
                pass
            else:
                filtered_grad_outs.append(o)
        filtered_grad_outs = tuple(filtered_grad_outs)

        # Important note - Even though the forward graph can be same for
        # different invoke_subgraphs, the backward graph can be different
        # because the tangent strides can be different. So, here we cache on
        # tangent_metadata in addition to identifier
        from torch._guards import detect_fake_mode
        from torch._subclasses._fake_tensor_utils import _CacheKeyState
        from torch._subclasses.fake_tensor import extract_tensor_metadata

        fake_mode = detect_fake_mode(primals + filtered_grad_outs)
        if fake_mode is None:
            raise AssertionError("fake_mode should be enabled for HOPs")
        state = _CacheKeyState(fake_mode.shape_env)

        tangent_metadata: list[object] = []
        for tangent in filtered_grad_outs:
            metadata = extract_tensor_metadata(tangent)
            metadata._flatten_into(tangent_metadata, fake_mode, state)

        # Add aliasing information to tangent_metadata
        # Two tangents are aliased if they are the same tensor object (using id())
        # We create a tuple of tuples where each inner tuple contains indices of aliased tensors
        # e.g. ((0, 1),) would mean there is one aliasing group, and the first and second tangents are aliased
        # e.g. () would mean there is no aliasing between tangents
        tensor_to_indices: dict[int, list[int]] = defaultdict(list)
        for i, tangent in enumerate(filtered_grad_outs):
            if isinstance(tangent, torch.Tensor):
                tensor_to_indices[id(tangent)].append(i)

        aliasing_groups = tuple(
            sorted(
                tuple(indices)
                for indices in tensor_to_indices.values()
                if len(indices) > 1
            )
        )
        tangent_metadata.append(aliasing_groups)

        # pyrefly: ignore [bad-assignment]
        tangent_metadata = tuple(tangent_metadata)

        # bw_graph is a joint graph with signature (*primals_and_tangents) and
        # returns (*grads_and_fw_outs). To get the grads, we use the num_fw_outs
        # to extract the grads.
        primals_and_tangents = primals + filtered_grad_outs

        # Check if we have already traced the bwd subgraph.
        bw_graph = None
        suffix = None
        invoke_subgraph_cache = get_invoke_subgraph_cache()
        cache_hit = False
        if invoke_subgraph_cache:
            bw_graph, suffix = invoke_subgraph_cache.get_lazy_bwd_entry(
                identifier, tangent_metadata
            )
            cache_hit = bw_graph is not None

        if bw_graph is None:
            if suffix is not None:
                raise AssertionError(
                    f"suffix should be None when bw_graph is None, got {suffix}"
                )
            with dynamo_timed(
                "invoke_subgraph_trace_joint_graph", log_pt2_compile_event=True
            ):
                bw_graph = trace_joint_graph_as_bwd(
                    subgraph,
                    len(primals),
                    primals_and_tangents,
                    ctx._fw_include_key_set,
                    ctx._fw_exclude_key_set,
                )
                if (
                    hasattr(subgraph, "meta")
                    and "nested_region_config" in subgraph.meta
                ):
                    bw_graph.meta["nested_region_config"] = subgraph.meta[
                        "nested_region_config"
                    ]

        if invoke_subgraph_cache and not cache_hit:
            suffix = invoke_subgraph_cache.add_lazy_bwd_entry(
                identifier, tangent_metadata, bw_graph
            )
            warn_and_trace_duplicate_backward(
                identifier, suffix, bw_graph, len(primals), filtered_grad_outs
            )

        with _set_invoke_subgraph_call_id(ctx._call_id):
            grads = invoke_subgraph(
                bw_graph, f"bw_{identifier}_{suffix}", *primals_and_tangents
            )[: -output_metadata.num_fw_outs]
        return None, None, None, *grads


@invoke_subgraph.py_autograd_impl
def _(subgraph, identifier, *operands):
    from torch._guards import detect_fake_mode

    # Eager backends run the captured HOP with real tensors and no AOTAutograd
    # fake mode. Let regular autograd record through the subgraph in that case.
    if detect_fake_mode(operands) is None:
        if getattr(subgraph, "_boxed_call", False):
            return subgraph(list(operands))
        return subgraph(*operands)

    # Check if we have already traced the subgraph.
    invoke_subgraph_cache = get_invoke_subgraph_cache()
    if invoke_subgraph_cache:
        if saved_autograd_fn := invoke_subgraph_cache.get_autograd_key_entry(
            identifier
        ):
            return saved_autograd_fn(*operands)

    output_metadata = get_output_metadata(subgraph, *operands)

    def autograd_fn_callable(*args):
        return InvokeSubgraphAutogradOp.apply(
            subgraph, identifier, output_metadata, *args
        )

    # Save the autograd_fn_callable in the dispatch set cache.
    if invoke_subgraph_cache:
        invoke_subgraph_cache.add_autograd_key_entry(identifier, autograd_fn_callable)

    return autograd_fn_callable(*operands)


@invoke_subgraph.py_impl(DebugMode)
def _(debug_mode, subgraph, identifier, *operands):
    # record HOP call
    call = torch.utils._debug_mode._OpCall(
        invoke_subgraph,
        (identifier, *operands),
        kwargs={},
        call_depth=debug_mode.call_depth + 1,
        stack=debug_mode.record_stack_trace,
    )
    debug_mode._record_call(call)

    debug_mode.call_depth += 1
    debug_mode._handle_annotate(f"[enter InvokeSubgraph HOP] {identifier}")

    # If the HOP is dispatched from DebugMode, we should enable debug_mode
    # for the subgraph call.
    with debug_mode:
        if getattr(subgraph, "_boxed_call", False):
            result = subgraph(list(operands))
        else:
            result = subgraph(*operands)
    debug_mode._handle_annotate(f"[exit InvokeSubgraph HOP] {identifier}")
    debug_mode.call_depth -= 1
    # record output of HOP
    debug_mode._record_call_output(call, result)
    return result


@invoke_subgraph.py_impl(DispatchKey.CompositeExplicitAutograd)
def _(subgraph, identifier, *operands):
    from torch.utils._python_dispatch import _get_current_dispatch_mode

    mode = _get_current_dispatch_mode()

    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")

    if getattr(subgraph, "_boxed_call", False):
        return subgraph(list(operands))
    else:
        return subgraph(*operands)


@invoke_subgraph.py_functionalize_impl
def _(ctx, subgraph, identifier, *operands):
    from torch._higher_order_ops.auto_functionalize import (
        can_auto_functionalize,
        do_auto_functionalize_v2,
    )

    # (in the functionalization metadata phase) Capture tokens before
    tokens_before = dict(ctx.mode._tokens)

    # Check if this subgraph has effects stored in the cache
    invoke_subgraph_cache = get_invoke_subgraph_cache()
    effects = None
    if invoke_subgraph_cache:
        effects = invoke_subgraph_cache.get_effects(identifier)

    if effects:
        if len(effects) != 1:
            raise AssertionError(
                f"Multiple effects within a subgraph NYI, got {len(effects)} effects"
            )
        tokens = ctx.mode._tokens
        effects = next(iter(effects))
        token_input = tokens[effects]

        operands = (token_input, *operands)

        def wrap_subgraph(subgraph):
            def wrapped_subgraph(token, *args):
                res = subgraph(*args)
                return ctx.unwrap_tensors(ctx.mode._tokens[effects]), *res

            return wrapped_subgraph

        subgraph = wrap_subgraph(subgraph)

    unwrapped_operands = ctx.unwrap_tensors(operands)

    functionalize_schema = None
    schema_cache_key = None
    if invoke_subgraph_cache:
        schema_cache_key = (
            id(subgraph),
            identifier,
            tuple(type(operand) for operand in operands),
        )
        functionalize_schema = invoke_subgraph_cache.get_functionalize_schema_entry(
            schema_cache_key
        )

    if functionalize_schema is None:
        hop_instance = HopInstance.create(
            invoke_subgraph, subgraph, identifier, *operands
        )
        if invoke_subgraph_cache and schema_cache_key is not None:
            invoke_subgraph_cache.add_functionalize_schema_entry(
                schema_cache_key, hop_instance._schema
            )
    else:
        hop_instance = HopInstance(invoke_subgraph, functionalize_schema)

    functionalized_identifier = None
    if can_auto_functionalize(hop_instance):
        # NOTE: [auto_functionalize x invoke_subgraph caching]
        # invoke_subgraph is special because of its identifier based caching mechanism.
        # In invoke_subgraph's functionalization key implementation, we create a new
        # identifier because the subgraph is replaced by FunctionWithNoFreeVars in a
        # functional + epilogue form.
        if not isinstance(identifier, str):
            raise AssertionError(
                f"identifier must be a string for auto_functionalize, got {type(identifier)}"
            )

        if ctx.mode._keep_input_mutations:
            # With keep_input_mutations=True, wrap invoke_subgraph in
            # auto_functionalized_v2. This allows copy_ epilogues in the subgraph for
            # Inductor to fuse where useful.
            # See NOTE [Support input mutation of hops] for the overall design.
            return do_auto_functionalize_v2(
                ctx.mode,
                hop_instance,
                (subgraph, "auto_functionalized_" + identifier, *operands),
                {},
            )

        # With keep_input_mutations=False, the subgraph must not mutate its inputs.
        # Return updated inputs so AOTAutograd can generate the copy_ epilogue.
        mutated_operand_indices = tuple(
            idx
            for idx, arg in enumerate(hop_instance._schema.arguments[2:])
            if arg.alias_info is not None and arg.alias_info.is_write
        )
        functionalized_subgraph = FunctionalizeCtxWrapper(
            ctx, subgraph, mutated_input_indices=mutated_operand_indices
        )
        functionalized_identifier = "functionalized_" + identifier
        with ctx.redispatch_to_next():
            out = invoke_subgraph(
                functionalized_subgraph,
                functionalized_identifier,
                *unwrapped_operands,
            )

        num_outputs = len(hop_instance._schema.returns)
        actual_out = out[:num_outputs]
        mutated_out = out[num_outputs:]
        for operand_idx, updated_operand in zip(mutated_operand_indices, mutated_out):
            operand = operands[operand_idx]
            ctx.replace(operand, updated_operand)
            ctx.commit_update(operand)
            ctx.sync(operand)
        out = actual_out
    else:
        with ctx.redispatch_to_next():
            # NB: There is an assumption that subgraph does not mutate inputs and
            # there is no aliasing. It's Dynamo's responsibility to prevent formation
            # of invoke_subgraph ops if input aliasing/mutation is detected.
            functionalized_subgraph = FunctionalizeCtxWrapper(ctx, subgraph)
            out = invoke_subgraph(
                functionalized_subgraph, identifier, *unwrapped_operands
            )

    if effects:
        (new_token, *out) = out
        ctx.mode._tokens[effects] = new_token

    # (in the functionalization metadata phase) Capture tokens after and see if
    # there are any differences (there are new effects or the token value for an
    # effect type has changed)
    tokens_after = dict(ctx.mode._tokens)
    discovered_effects = set()
    for effect_type, token in tokens_after.items():
        if effect_type not in tokens_before or tokens_before[effect_type] is not token:
            discovered_effects.add(effect_type)

    if discovered_effects:
        if not ctx.mode._allow_token_discovery:
            raise AssertionError(
                f"Number of tokens changed by {len(discovered_effects)} when tracing subgraph {subgraph}."
            )
        # Later passes look up effects using the identifier on the emitted HOP.
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_effects(identifier, discovered_effects)
            if functionalized_identifier is not None:
                invoke_subgraph_cache.add_effects(
                    functionalized_identifier, discovered_effects
                )

    return ctx.wrap_tensors(out)


# Register the hop fake fn. This will be called in the fake_tensor _dispatch_impl.
@register_fake(invoke_subgraph)
def _(subgraph, identifier, *operands):
    from torch._dynamo.utils import dynamo_timed

    with dynamo_timed("invoke_subgraph_fake_tensor", log_pt2_compile_event=True):
        if getattr(subgraph, "_boxed_call", False):
            return subgraph(list(operands))
        return subgraph(*operands)


@invoke_subgraph.py_impl(ProxyTorchDispatchMode)
def _(proxy_mode: ProxyTorchDispatchMode, subgraph, identifier, *operands):
    # Check if we have already traced the subgraph.
    graph = None
    invoke_subgraph_cache = get_invoke_subgraph_cache()
    if invoke_subgraph_cache:
        graph = invoke_subgraph_cache.get_proxy_dispatch_entry(identifier)

    if graph is None:
        from torch._dynamo.utils import dynamo_timed

        with dynamo_timed("invoke_subgraph_proxy_tensor", log_pt2_compile_event=True):
            subgraph_decomp_table = _extract_nested_region_config(subgraph)

            # NB: invoke_subgraph subgraph re-trace seq_nr
            # The joint graph seq_nr will get wrong in the subsequent re-trace (all nodes will have the same seq_nr),
            # so we preserve the original graph's seq_nr here.
            with torch.fx.traceback._preserve_node_seq_nr():
                graph = reenter_make_fx(
                    subgraph, subgraph_decomp_table=subgraph_decomp_table
                )(*operands)

        # Propagate nested_region_config from the original subgraph to the
        # re-traced graph so it's available for regional compilation.
        if (
            isinstance(subgraph, torch.fx.GraphModule)
            and hasattr(subgraph, "meta")
            and "nested_region_config" in subgraph.meta
            and "nested_region_config" not in graph.meta
        ):
            graph.meta["nested_region_config"] = subgraph.meta["nested_region_config"]

        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode(operands)
        # Only insert deferred runtime asserts when we have dynamic shapes.
        # When shape_env is None (static shapes), there are no deferred asserts to insert.
        if fake_mode is not None and fake_mode.shape_env is not None:
            insert_deferred_runtime_asserts(
                graph,
                fake_mode.shape_env,
                "invoke_subgraph_proxy_torch_dispatch_mode",
                export=True,
            )
            graph.recompile()

        if not isinstance(proxy_mode.tracer, torch.fx.Tracer):
            raise AssertionError(
                f"expected proxy_mode.tracer to be torch.fx.Tracer, got {type(proxy_mode.tracer)}"
            )
        if invoke_subgraph_cache:
            invoke_subgraph_cache.add_proxy_dispatch_entry(identifier, graph)

    node_args = (graph, identifier, *operands)

    def _unwrap_proxy(arg):
        if isinstance(arg, torch.fx.GraphModule):
            # NOTE: [invoke_subgraph proxy_mode x auto_functionalize]
            # Previously, we assumed that `invoke_subgraph` would always be traced with the same tracer.
            # This allowed us to cache modules by their identifiers, assuming they were already registered.
            #
            # However, this assumption no longer holds when we auto-functionalize `invoke_subgraph`.
            # auto_functionalize functionalizes the subgraph and wrap it with `FunctionWithNoFreeVars`.
            # In the proxy mode implementation of `auto_functionalized_v2`, we need to materialize `FunctionWithNoFreeVars`
            # input as a graph module. To do this, we re-trace the `invoke_subgraph` hop, which starts a new sub-tracer
            # (see NOTE [materialize callable inputs as graph]). # When the new sub-tracer traces the `invoke_subgraph`
            # with a previously cached identifier, the corresponding graph module might not
            # exist as a submodule in the new tracer's root. Therefore, we register it as a submodule below.
            #
            # The alternative is to give a new identifier when we re-trace the invoke_subgraph but this will increase
            # the compilation time, which defeats the purpose of caching.
            registered_before = False
            for (
                _,
                submod,
            ) in proxy_mode.tracer.root.named_modules():  # type: ignore[union-attr]
                if arg is submod:
                    registered_before = True

            if not registered_before:
                qualname = proxy_mode.tracer.get_fresh_qualname("repeated_subgraph")  # type: ignore[union-attr]
                proxy_mode.tracer.root.register_module(qualname, arg)  # type: ignore[union-attr]
        return proxy_mode.tracer.unwrap_proxy(arg)  # type: ignore[union-attr]

    proxy_args = pytree.tree_map(_unwrap_proxy, node_args)  # type: ignore[union-attr]
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", invoke_subgraph, proxy_args, {}
    )

    # Propagate nested_region_config to the node metadata so the regional
    # inductor compiler can compile backward invoke_subgraph nodes.
    # Dynamo sets this for forward nodes during tracing.  For backward
    # nodes created during joint tracing, the config lives on the subgraph
    # module (set by InvokeSubgraphAutogradOp.backward) but may be wrapped
    # by FunctionalizeCtxWrapper — unwrap to find it.
    nested_config = None
    orig_subgraph = (
        subgraph.subgraph if isinstance(subgraph, FunctionalizeCtxWrapper) else subgraph
    )
    for gm in (graph, orig_subgraph):
        if (
            isinstance(gm, torch.fx.GraphModule)
            and hasattr(gm, "meta")
            and "nested_region_config" in gm.meta
        ):
            nested_config = gm.meta["nested_region_config"]
            break

    call_id = _current_invoke_subgraph_call_id()

    if nested_config is not None or call_id is not None:
        node = out_proxy.node
        if "custom" not in node.meta:
            node.meta["custom"] = {}
        if nested_config is not None:
            node.meta["custom"]["nested_region_config"] = nested_config
        if call_id is not None:
            node.meta["custom"]["call_id"] = call_id

    example_out = invoke_subgraph(graph, identifier, *operands)
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


def invoke_subgraph_inductor_compile(
    gm, example_inputs, inductor_config_patches=None, **kwargs
):
    from torch._functorch._aot_autograd.runtime_wrappers import (
        SerializableCompiledFunction,
    )
    from torch._functorch._aot_autograd.utils import simple_wraps
    from torch._inductor import config
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.standalone_compile import AOTCompiledArtifact

    # Used for testing only, should only be changed via _testing_capture_invoke_subgraph_inductor_compile_gms()
    if (
        torch._dynamo.testing._testing_invoke_subgraph_inductor_compile_captured_gms
        is not None
    ):
        torch._dynamo.testing._testing_invoke_subgraph_inductor_compile_captured_gms.append(
            copy.deepcopy(gm)
        )

    if inductor_config_patches is None:
        inductor_config_patches = {}

    # Saved tensors flow across the HOP boundary into a separately-compiled bw
    # subgraph whose IR was traced with natural (unpadded) strides. Mark every
    # output of this subgraph as user-visible so Inductor's comprehensive_padding
    # leaves their strides alone — otherwise the bw's assert_size_stride on the
    # incoming saved tensor will fire (e.g. F.linear output padded 200008 -> 200064).
    from torch._inductor.compile_fx import _recursive_record_user_visible_output_idxs

    output_node = next(iter(gm.graph.find_nodes(op="output")))
    output_node.meta["user_visible_output_idxs"] = [
        idx
        for idx in range(len(output_node.args[0]))
        if isinstance(output_node.args[0][idx], torch.fx.Node)
    ]
    _recursive_record_user_visible_output_idxs(gm)

    compile_fn = config.patch(inductor_config_patches)(compile_fx_inner)
    compiled_fn_inner = compile_fn(gm, example_inputs)
    if not compiled_fn_inner._boxed_call:
        raise AssertionError(
            "compiled_fn_inner must have _boxed_call attribute set to True"
        )

    # Follow boxed calling convention
    @simple_wraps(compiled_fn_inner)
    def forward(*runtime_args: tuple[Any]):
        full_args = []
        full_args.extend(runtime_args)
        return compiled_fn_inner(full_args)

    # Just for convenience
    forward.zero_grad = gm.zero_grad  # type: ignore[attr-defined]
    forward.named_parameters = gm.named_parameters  # type: ignore[attr-defined]
    forward.named_buffers = gm.named_buffers  # type: ignore[attr-defined]

    # TODO: Do we need the post compile passes in _aot_stage2b_compile_forward_or_inference?
    # TODO: add a real serialize function for SerializableCompiledFunction like _cache_inference_info
    forward.serialize = SerializableCompiledFunction(forward, lambda: None)  # type: ignore[attr-defined]
    return AOTCompiledArtifact(forward)


def get_invoke_subgraph_compile_options(
    fw_inductor_config_patches=None,
    decompositions=None,
    partitioner="min_cut_rematerialization_partition",
    *,
    bw_inductor_config_patches=None,
):
    if fw_inductor_config_patches is None:
        fw_inductor_config_patches = {"triton.autotune_at_compile_time": True}

    # The backward uses bw_inductor_config_patches when set (independently of the
    # forward), otherwise it reuses the forward config.
    bw_patches = (
        bw_inductor_config_patches
        if bw_inductor_config_patches is not None
        else fw_inductor_config_patches
    )

    from torch._inductor import config as inductor_config

    # Validate that all config keys exist
    for patches in (fw_inductor_config_patches, bw_inductor_config_patches):
        for key in patches or {}:
            if not hasattr(inductor_config, key):
                raise ValueError(
                    f"Invalid inductor config key '{key}' in get_invoke_subgraph_compile_options. "
                    f"Available config keys can be found in torch._inductor.config"
                )

    fw_compiler = functools.partial(
        invoke_subgraph_inductor_compile,
        inductor_config_patches=fw_inductor_config_patches,
    )
    bw_compiler = functools.partial(
        invoke_subgraph_inductor_compile,
        inductor_config_patches=bw_patches,
    )

    return NestedCompileRegionOptions(
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        partitioner=partitioner,
        decompositions=decompositions,
        inductor_config_patches=fw_inductor_config_patches,
        bw_inductor_config_patches=bw_inductor_config_patches,
    )
<|endoftext|># mypy: allow-untyped-defs
import logging
from contextlib import contextmanager

import torch
from torch._C import DispatchKey  # @manual
from torch._functorch._aot_autograd.utils import KNOWN_TYPES
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._library.fake_class_registry import (
    _is_script_object,
    _ns_and_class_name,
    FakeScriptObject,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.fx.node import has_side_effect
from torch.utils import _pytree as pytree


log = logging.getLogger(__name__)


# The call_torchbind operator represents a method invocation on a torchbind
# object. The calling convention is:
#   call_torchbind(self: ScriptObject, method_name: str, *method_args, **method_kwargs)
# We do not expect users to write this operator directly. Instead it will be
# emitted by Dynamo when tracing encounters a torchbind object.
class CallTorchBind(HigherOrderOperator):
    def __init__(self):
        super().__init__("call_torchbind")

    def __call__(self, obj, method, *args, **kwargs):
        # pyrefly: ignore [missing-attribute]
        return super().__call__(obj, method, *args, **kwargs)

    @staticmethod
    def schema(obj, method) -> torch.FunctionSchema:
        """
        Returns the schema of ``CallTorchbind.__call__``.
        """
        if not isinstance(obj, torch._inductor.ir.TorchBindObject):
            raise AssertionError(f"expected obj to be TorchBindObject, got {type(obj)}")
        val = obj.get_real_obj()
        schema = val._get_method(method).schema
        schema_str = str(schema)
        new_schema_str = f"call_torchbind({str(schema.arguments[0].real_type)} {schema.arguments[0].name},"
        first_comma_index = schema_str.find(",")
        if first_comma_index == -1:
            # If no comma is found, find the last closing parenthesis
            first_comma_index = schema_str.rfind(") ->")
        new_schema_str = new_schema_str + " str method" + schema_str[first_comma_index:]
        new_schema = torch._C.parse_schema(new_schema_str)
        return new_schema


call_torchbind = CallTorchBind()

# Register this operator as side-effectful with FX.
# TODO: this is not really sufficient. While passes (hopefully) check
# Node.is_impure() and make good decisions, we also assume we can execute the
# graph as many times as we want without changing behavior, which is NOT true of
# ops that mutate torchbind object state.
has_side_effect(call_torchbind)

_orig_scriptmethod_call = torch.ScriptMethod.__call__


def torchbind_method_redispatch(self, *args, **kwargs):
    if _is_script_object(self.raw_owner):
        return call_torchbind(self.raw_owner, self.name, *args, **kwargs)
    return _orig_scriptmethod_call(self, *args, **kwargs)


@contextmanager
def enable_torchbind_tracing():
    """Context manager that acts as a feature flag to enable torchbind tracing
    behavior. Once torchbind tracing has been stabilized, we can remove this and
    turn it always on.
    """
    try:
        KNOWN_TYPES.append(torch.ScriptObject)
        torch.ScriptMethod.__call__ = torchbind_method_redispatch  # type: ignore[method-assign]
        yield
    finally:
        if KNOWN_TYPES.pop() is not torch.ScriptObject:
            raise AssertionError(
                "Someone else messed with KNOWN_TYPES during tracing, exploding."
            )
        torch.ScriptMethod.__call__ = _orig_scriptmethod_call  # type: ignore[method-assign]


@call_torchbind.py_impl(DispatchKey.CompositeExplicitAutograd)
def call_torchbind_impl(obj, method, *args, **kwargs):
    if isinstance(obj, torch.ScriptObject):
        return _orig_scriptmethod_call(getattr(obj, method), *args, **kwargs)
    elif isinstance(obj, FakeScriptObject):
        return getattr(obj.wrapped_obj, method)(*args, **kwargs)
    else:
        raise RuntimeError(f"Unsupported first arg type {type(obj)} for call_torchbind")


@call_torchbind.py_impl(ProxyTorchDispatchMode)
def inner(mode, *args, **kwargs):
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)

    out_proxy = mode.tracer.create_proxy(
        "call_function",
        call_torchbind,
        proxy_args,
        proxy_kwargs,
    )
    out = call_torchbind(*args, **kwargs)

    obj, method, *_rest_args = args
    if isinstance(obj, torch.ScriptObject):
        ns, class_name = _ns_and_class_name(
            obj._type().qualified_name()  # type: ignore[attr-defined]
        )
        log.warning(
            "Tracing torchbind method %s.%s with real ScriptObject. This may"
            " cause the original object being mutated. If this is not intended,"
            ' You can register a fake class with torch._library.register_fake_class("%s::%s").',
            class_name,
            method,
            ns,
            class_name,
        )

    ret = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    if "val" not in out_proxy.node.meta:
        if out is not None and not isinstance(out, (int, float, bool)):
            raise AssertionError(
                f"Currently, only these constant dtypes are supported to be returned from torchbind methods, got {type(out)}"
            )
        out_proxy.node.meta["val"] = out
    return ret


# When tracing with fake script object, the call_torchbind op will return a fake tensor
# When tracing with real script object, the call_torchbind op may return a real tensor,
# we need to convert it to fake tensor manually. Dynamic shape is supported.
@call_torchbind.py_impl(FakeTensorMode)
def call_torchbind_fake(mode, *args, **kwargs):
    with mode:
        out = call_torchbind_impl(*args, **kwargs)
        return pytree.tree_map_only(
            torch.Tensor,
            lambda x: mode.from_tensor(x, static_shapes=True)
            if not torch._subclasses.fake_tensor.is_fake_tensor(x)
            else x,
            out,
        )


call_torchbind.py_autograd_impl(
    autograd_not_implemented(call_torchbind, deferred_error=True)
)


@call_torchbind.py_functionalize_impl
def call_torchbind_func(ctx, *args, **kwargs):
    from torch._higher_order_ops.effects import handle_effects

    return handle_effects(
        ctx.mode._allow_token_discovery, ctx.mode._tokens, call_torchbind, args, kwargs
    )
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented, register_fake
from torch._ops import HigherOrderOperator
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    maybe_handle_decomp,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


# TODO to figure out a more generic approach
ALLOWABLE_OPS = [
    torch.ops.aten.linear.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.div.Scalar,
]


class OutDtypeOperator(HigherOrderOperator):
    """
    The out_dtype operator takes an existing ATen functional operator, an
    `out_dtype` argument, and arguments to the original operator, and executes
    the original operator and returns a Tensor with the `out_dtype` precision.
    This operator does not mandate a compute precision so it allows the
    representation to not be opinionated about the exact implementation.

    The general implementation for all operators will be the following:
        1. Promote inputs dtypes based on default PyTorch dtype promotion rules,
            using the dtypes of all input Tensors/Scalars and the `out_dtype`
            argument.
        2. Execute the operator
        3. Cast the output to `out_dtype`
    """

    def __init__(self) -> None:
        super().__init__("out_dtype")

    def __call__(self, op, output_dtype, *args):
        if not isinstance(op, torch._ops.OpOverload):
            raise ValueError("out_dtype's first argument must be an OpOverload")
        if op._schema.is_mutable:
            raise ValueError(
                "out_dtype's first argument needs to be a functional operator"
            )
        if not (
            len(op._schema.returns) == 1
            and isinstance(op._schema.returns[0].type, torch.TensorType)
        ):
            raise ValueError(
                "out_dtype's can only apply to ops that return a single tensor"
                f"Instead got {[r.type for r in op._schema.returns]}"
            )

        if op not in ALLOWABLE_OPS:
            raise ValueError(
                f"out_dtype only allows the following operators: {ALLOWABLE_OPS}."
            )

        # pyrefly: ignore [missing-attribute]
        res = super().__call__(op, output_dtype, *args)

        return res


out_dtype = OutDtypeOperator()


def trace_out_dtype(proxy_mode, func_overload, op, output_dtype, *args):
    # NB: Long-term we should put the decomposition logic into
    # ProxyTorchDispatchMode so that people do not need to call maybe_handle_decomp
    # in all HigherOrderOp proxy implementations.
    r = maybe_handle_decomp(proxy_mode, func_overload, (op, output_dtype, *args), {})
    if r is not NotImplemented:
        return r

    with disable_proxy_modes_tracing():
        # This is a simplified implementation of this operator just for tracing.
        # Actual implementation may also first promote the arguments
        out = op(*args).to(dtype=output_dtype)

    node_args = (op, output_dtype, *args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="out_dtype"
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@out_dtype.py_impl(DispatchKey.CompositeExplicitAutograd)
def out_dtype_dense(op: torch._ops.OpOverload, output_dtype: torch.dtype, *args):
    if is_int_mm(op, output_dtype, args):
        return torch._int_mm(*args)
    return out_dtype_fallback(op, output_dtype, *args)


def is_int_mm(op, output_dtype, args):
    return (
        op is torch.ops.aten.mm.default
        and output_dtype == torch.int32
        and len(args) == 2
        and args[0].dtype == torch.int8
        and args[1].dtype == torch.int8
        and (args[0].is_cuda or args[0].is_xpu)
        and (args[1].is_cuda or args[1].is_xpu)
    )


def out_dtype_fallback(op, output_dtype, *args):
    flat_inputs = pytree.arg_tree_leaves(*args) + [torch.ones(1, dtype=output_dtype)]
    promote_dtype: torch.dtype = elementwise_dtypes(
        *flat_inputs,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )[0]

    casted_args = pytree.tree_map_only(
        torch.Tensor, lambda arg: arg.to(dtype=promote_dtype), args
    )
    res = op(*casted_args).to(dtype=output_dtype)
    return res


out_dtype.py_autograd_impl(autograd_not_implemented(out_dtype, deferred_error=True))


@out_dtype.py_impl(ProxyTorchDispatchMode)
def out_dtype_proxy(
    mode: ProxyTorchDispatchMode,
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
):
    return trace_out_dtype(mode, out_dtype, op, output_dtype, *args)


@register_fake(out_dtype, skip_cache=True)
def out_dtype_fake_tensor_mode(
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
):
    return out_dtype_dense(op, output_dtype, *args)


@out_dtype.py_functionalize_impl
def out_dtype_func(ctx, op, output_dtype, *args):
    unwrapped_args = tuple(ctx.unwrap_tensors(arg) for arg in args)

    with ctx.redispatch_to_next():
        res = out_dtype(op, output_dtype, *unwrapped_args)
    return ctx.wrap_tensors(res)
<|endoftext|># mypy: allow-untyped-defs

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import is_fake_tensor
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


AOTI_LOWERED_MODULE = "AOTInductorEPModule/AOTInductorRunnerWrapper"


class AOTICallDelegate(HigherOrderOperator):
    """aoti_call_delegate is a HOP for calling AOTInductor lowered submodule in ExportedProgram.

    It has the following signature:
    aoti_call_delegate(
        lowered_module: Union[AOTInductorEPModule, AOTInductorRunnerWrapper]
        original_gm:fx.GraphModule,
        weight_args: List[Tensor],
        input_args: List[Tensor],
    ) -> outputs: List[Tensor]

    where,
    - lowered_module is the AOTInductor lowered submodule, backed by compiled .so file, supporting real tensor inputs
    - original_gm is the stateless version of the original GraphModule before lowering, allowing FakeTensor propagation
    - weight_args is the list of weights in original GraphModule, including parameters and buffers
    - input_args is the list of flatten inputs
    """

    def __init__(self) -> None:
        super().__init__("aoti_call_delegate")

    def __call__(
        self,
        lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
        original_gm: torch.fx.GraphModule,
        weight_args: list[torch.Tensor],
        input_args: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        # pyrefly: ignore [missing-attribute]
        return super().__call__(lowered_module, original_gm, weight_args, input_args)


aoti_call_delegate = AOTICallDelegate()
aoti_call_delegate.fallthrough(torch._C.DispatchKey.PythonDispatcher)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.PythonTLSSnapshot)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.ADInplaceOrView)
aoti_call_delegate.fallthrough(torch._C.DispatchKey.AutocastCPU)


@aoti_call_delegate.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
def call_delegate_cpu(
    lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
) -> list[torch.Tensor]:
    # FX creates this immutable_dict/list concept. Get rid of this.
    map_types: dict[type, type] = {
        torch.fx.immutable_collections.immutable_dict: dict,
        torch.fx.immutable_collections.immutable_list: list,
    }
    new_args = pytree.tree_map_only(
        tuple(map_types.keys()),
        lambda a: map_types[type(a)](a),
        weight_args + input_args,
        lambda a: isinstance(a, tuple(map_types.keys())),
    )
    has_fake_args = any(is_fake_tensor(arg) for arg in new_args)
    if has_fake_args:
        # use stateless original_gm for tracing with fake tensors
        fake_out = original_gm(*new_args)
        return fake_out
    else:
        # use AOTI Runner for real tensors
        new_input_args = new_args[len(weight_args) :]
        if type(lowered_module).__name__ == "AOTInductorRunnerWrapper":
            return lowered_module(*new_input_args)  # type: ignore[misc]
        elif type(lowered_module).__name__ == "AOTInductorEPModule":
            return lowered_module(new_input_args)  # type: ignore[misc]
        else:
            raise RuntimeError(
                f"Unexpected lowered_module type: {type(lowered_module)}."
            )


def trace_aoti_call_delegate(
    proxy_mode, func_overload, lowered_module, original_gm, weight_args, input_args
):
    proxy_mode.tracer.root.register_module("lowered_module", lowered_module)
    proxy_mode.tracer.root.register_module("original_gm", original_gm)

    node_args = (lowered_module, original_gm, weight_args, input_args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="aoti_call_delegate"
    )
    with disable_proxy_modes_tracing():
        out = call_delegate_cpu(lowered_module, original_gm, weight_args, input_args)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@aoti_call_delegate.py_impl(ProxyTorchDispatchMode)
def call_delegate_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
):
    res = trace_aoti_call_delegate(
        mode, aoti_call_delegate, lowered_module, original_gm, weight_args, input_args
    )
    return res


@register_fake(aoti_call_delegate, skip_cache=True)
def call_delegate_fake_tensor_mode(
    lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
) -> list[torch.Tensor]:
    return call_delegate_cpu(lowered_module, original_gm, weight_args, input_args)


@aoti_call_delegate.py_functionalize_impl
def call_delegate_functionalize(
    ctx,
    lowered_module: AOTI_LOWERED_MODULE,  # type: ignore[valid-type]
    original_gm: torch.fx.GraphModule,
    weight_args: list[torch.Tensor],
    input_args: list[torch.Tensor],
):
    unwrapped_weight_args = tuple(
        ctx.unwrap_tensors(weight_arg) for weight_arg in weight_args
    )
    unwrapped_input_args = tuple(
        ctx.unwrap_tensors(input_arg) for input_arg in input_args
    )
    with ctx.redispatch_to_next():
        res = aoti_call_delegate(
            lowered_module,
            original_gm,
            unwrapped_weight_args,  # type: ignore[arg-type]
            unwrapped_input_args,  # type: ignore[arg-type]
        )
        return ctx.wrap_tensors(res)
<|endoftext|># mypy: allow-untyped-defs
# need to fix prim_hop_base type annotations first

import dataclasses

import torch
from torch._higher_order_ops.base_hop import BaseHOP, FunctionWithNoFreeVars


class InvokeQuantTracer(BaseHOP):
    def __init__(self) -> None:
        super().__init__("invoke_quant_packed")

    def __call__(self, subgraph, *operands, scheme=None, quant_options=None):
        subgraph = FunctionWithNoFreeVars(subgraph)
        return super().__call__(
            subgraph, *operands, scheme=scheme, quant_options=quant_options
        )


invoke_quant_packed = InvokeQuantTracer()


class InvokeQuantUnpacked(BaseHOP):
    def __init__(self) -> None:
        super().__init__("invoke_quant")


invoke_quant = InvokeQuantUnpacked()


@dataclasses.dataclass(frozen=True, repr=True)
class InvokeQuant:
    """
    Invoke a quantization function that will be preserved as a single operator. Preservation
    as a single operator aids in pattern matching and custom lowerings.

    The operation appears as:
        torch.ops.higher_order.invoke_quant(subgraph, *args, scheme=scheme)

    Args:
        codegen_low_precision: Use observed subgraph dtypes for codegen instead of
            upcasting to fp32. Can improve performance for prologue fusion but
            requires careful testing of numerics.
    """

    codegen_low_precision: bool = True

    def __call__(
        self,
        *args,
        scheme: str | None = None,
        **kwargs,
    ):
        if not torch.compiler.is_compiling():
            return args[0](*args[1:], **kwargs)

        if scheme is not None:
            kwargs["scheme"] = scheme

        return invoke_quant_packed(*args, **kwargs, quant_options=self)  # type: ignore[call-arg]
<|endoftext|># mypy: allow-untyped-defs
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, get_args

import torch
import torch._library.utils as library_utils
import torch.utils._pytree as pytree
from torch import Tensor
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    _has_gen_schema,
    call_op,
    HopInstance,
    HopSchema,
    materialize_callable_in_args,
    register_fake,
    unique_graph_id,
)
from torch._ops import HigherOrderOperator, OperatorBase, OpOverload
from torch._prims_common import clone_preserve_strides
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class SchemaHolder:
    def __init__(self, schema: torch.FunctionSchema):
        self.schema = schema

    def __eq__(self, other):
        return self.schema == other.schema

    def __hash__(self) -> int:
        return hash(self.schema)

    @classmethod
    def from_tree_spec(cls, tree_spec: pytree.TreeSpec):
        if tree_spec is None:
            raise AssertionError("tree_spec cannot be None")
        return cls(pytree.tree_unflatten([], tree_spec).schema)


# register_constant allows us to get a tree_spec from pytree.tree_flatten(SchemaHolder(FunctionSchema)).
# The tree_spec is proxable in the graph and we can get back the schema via
# schema = pytree.tree_unflatten([], tree_spec).schema
pytree.register_constant(SchemaHolder)


def get_base(tensor):
    if torch.is_inference_mode_enabled():
        return tensor._inference_mode_base
    else:
        return tensor._base


class ViewInfo(ABC):
    base_index: int

    def __init__(self, base_index):
        self.base_index = base_index

    @abstractmethod
    def regenerate_view(self, bases_list: list[Tensor]):
        pass


@dataclass
class AsStridedViewInfo(ViewInfo):
    size: Sequence[int | torch.SymInt]
    stride: Sequence[int | torch.SymInt]
    storage_offset: int

    def __init__(self, base_index, size, stride, storage_offset):
        super().__init__(base_index)
        self.size = size
        self.stride = stride
        self.storage_offset = storage_offset

    def regenerate_view(self, bases_list: list[Tensor]):
        return torch.as_strided(
            bases_list[self.base_index],
            self.size,
            self.stride,
            self.storage_offset,
        )


@dataclass
class SliceViewInfo(ViewInfo):
    dim: int | torch.SymInt
    start: int | torch.SymInt
    end: int | torch.SymInt

    def __init__(self, base_index, dim, start, end):
        super().__init__(base_index)
        self.dim = dim
        self.start = start
        self.end = end

    def regenerate_view(self, bases_list: list[Tensor]):
        return torch.ops.aten.slice.Tensor(
            bases_list[self.base_index], self.dim, self.start, self.end
        )


@dataclass
class AliasViewInfo(ViewInfo):
    def __init__(self, base_index):
        super().__init__(base_index)

    def regenerate_view(self, bases_list: list[Tensor]):
        return torch.ops.aten.alias.default(bases_list[self.base_index])


@dataclass
class NotView(ViewInfo):
    def __init__(self, base_index):
        super().__init__(base_index)

    def regenerate_view(self, bases_list: list[Tensor]):
        return bases_list[self.base_index]


def is_alias(base, tensor):
    from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

    return all(
        statically_known_true(a)
        for a in [
            sym_eq(base.storage_offset(), tensor.storage_offset()),
            sym_eq(base.stride(), tensor.stride()),
            sym_eq(base.size(), tensor.size()),
        ]
    )


# return None or (dim, start, end)
def try_use_slice(base, tensor):
    from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

    # This condition should never be triggered.
    if is_alias(base, tensor):
        return (0, 0, base.size()[0])

    # TODO is there cases can we use slice even if stride or len(sizes) are not equal?
    if not statically_known_true(sym_eq(tensor.stride(), base.stride())):
        return None
    if not statically_known_true(sym_eq(len(tensor.size()), len(base.size()))):
        return None

    dim = None
    count = 0
    for i in range(len(tensor.size())):
        if base.size()[i] != tensor.size()[i]:
            dim = i
            count = count + 1
    if count != 1:
        return None

    if tensor.storage_offset() % tensor.stride()[dim] != 0:
        return None
    start = tensor.storage_offset() // tensor.stride()[dim]
    end = start + tensor.size()[dim]
    return (dim, start, end)


def write_view_information_to_args(
    mutable_arg_names: list[str],
    mutable_arg_types: list[torch.Type],
    kwargs: dict[str, Any],
    arg_to_base_index: dict[str, Any],
):
    """
    This function writes the view information into kwargs. It reads mutable_args from kwargs.
    and uses arg_to_base_index and tensor information to write ViewInfo into kwargs.
    mutable_arg_names: mutable custom operator arg names.
    mutable_arg_types: mutable custom operator arg types.
    kwargs: the original custom operator args.
    arg_to_base_index: maps mutable_arg_name to int | [int] that refers to the base tensor that
                       corresponds to the input tensor
    """

    def write_single_view(prefix: str, tensor: Tensor, base_index: int):
        if f"{prefix}_base_index" in kwargs:
            raise AssertionError(f"{prefix}_base_index already in kwargs")
        if f"{prefix}_size" in kwargs:
            raise AssertionError(f"{prefix}_size already in kwargs")
        if f"{prefix}_stride" in kwargs:
            raise AssertionError(f"{prefix}_stride already in kwargs")
        if f"{prefix}_storage_offset" in kwargs:
            raise AssertionError(f"{prefix}_storage_offset already in kwargs")

        if f"{prefix}_slice_dim" in kwargs:
            raise AssertionError(f"{prefix}_slice_dim already in kwargs")
        if f"{prefix}_slice_start" in kwargs:
            raise AssertionError(f"{prefix}_slice_start already in kwargs")
        if f"{prefix}_slice_end" in kwargs:
            raise AssertionError(f"{prefix}_slice_end already in kwargs")

        def use_as_strided(tensor):
            kwargs[f"{prefix}_size"] = tensor.size()
            kwargs[f"{prefix}_stride"] = tensor.stride()
            kwargs[f"{prefix}_storage_offset"] = tensor.storage_offset()

        def use_slice(dim, start, end):
            kwargs[f"{prefix}_slice_dim"] = dim
            kwargs[f"{prefix}_slice_start"] = start
            kwargs[f"{prefix}_slice_end"] = end

        def use_alias():
            kwargs[f"{prefix}_alias"] = True

        # The start if the function
        if tensor is None:
            kwargs[f"{prefix}_base_index"] = None
        else:
            base = get_base(tensor)
            kwargs[f"{prefix}_base_index"] = base_index
            if base is None:
                # no need to add anything else other than _base_index
                return
            elif is_alias(base, tensor):
                use_alias()
            elif (slice_info := try_use_slice(base, tensor)) is not None:
                use_slice(*slice_info)
            else:
                use_as_strided(tensor)

    for arg_name, arg_type in zip(mutable_arg_names, mutable_arg_types):
        arg = kwargs[arg_name]
        if library_utils.is_tensorlist_like_type(arg_type):
            if arg is None:
                kwargs[f"_{arg_name}_length"] = None
            else:
                kwargs[f"_{arg_name}_length"] = len(arg)
                for i, elem in enumerate(arg):
                    write_single_view(
                        f"_{arg_name}_{i}", elem, arg_to_base_index[arg_name][i]
                    )

        elif library_utils.is_tensor_like_type(arg_type):
            write_single_view(
                f"_{arg_name}",
                kwargs[arg_name],
                arg_to_base_index.get(arg_name),  # type: ignore[arg-type]
            )
        else:
            raise RuntimeError(f"Unsupported type {arg_type}")


# Returns a dict of arg_name -> ViewInfo | [ViewInfo]
def read_view_information_from_args(
    mutable_arg_names: list[str],
    mutable_arg_types: list[torch.Type],
    kwargs: dict[str, Any],
    all_bases: list[Tensor],
):
    """
    This reads the view information added by `write_view_information_to_args` from kwargs, pop them,
    and returns a dict arg_name -> ViewInfo | [ViewInfo](if the input is list). that maps each mutable arg
    to its view information.
    mutable_arg_names: mutable custom operator arg names.
    mutable_arg_types: mutable custom operator arg types.
    kwargs : args of auto_functionalize(custom_op, kwargs)
    """

    def get_arg(name):
        return kwargs.pop(name)

    def read_single_view(prefix):
        base_index = get_arg(f"{prefix}_base_index")
        if base_index is None:
            return None
        elif f"{prefix}_alias" in kwargs:
            get_arg(f"{prefix}_alias")
            return AliasViewInfo(base_index)
        elif f"{prefix}_storage_offset" in kwargs:
            # The view is regenerated using as_strided.
            size = get_arg(f"{prefix}_size")
            stride = get_arg(f"{prefix}_stride")
            storage_offset = get_arg(f"{prefix}_storage_offset")
            return AsStridedViewInfo(base_index, size, stride, storage_offset)
        elif f"{prefix}_slice_dim" in kwargs:
            dim = get_arg(f"{prefix}_slice_dim")
            start = get_arg(f"{prefix}_slice_start")
            end = get_arg(f"{prefix}_slice_end")
            return SliceViewInfo(base_index, dim, start, end)
        else:
            # This means that the argument is the base tensor
            return NotView(base_index)

    args_view_info: dict[str, Any] = {}
    for arg_name, arg_type in zip(mutable_arg_names, mutable_arg_types):
        if library_utils.is_tensorlist_like_type(arg_type):
            length = get_arg(f"_{arg_name}_length")
            if length is None:
                # The whole list is None.
                args_view_info[arg_name] = None
            else:
                args_view_info[arg_name] = [
                    read_single_view(f"_{arg_name}_{i}") for i in range(length)
                ]

        elif library_utils.is_tensor_like_type(arg_type):
            args_view_info[arg_name] = read_single_view(f"_{arg_name}")
        else:
            raise RuntimeError(f"Unsupported type {arg_type}")
    return args_view_info


# NOTE: [auto-functionalizing custom ops]
# Users may wish to torch.compile custom ops that mutate their inputs.
# torch.compile will automatically support this op without anyone needing
# to provide a functionalization kernel for it. Here's how.
#
# Let's say we have a hypothetical mylib::sin_(Tensor(a!) x) -> ()
# op. First, when FakeTensor sees this op:
# - If the schema says it returns nothing, we can generate a trivial
#   FakeTensor rule for it (that returns nothing).
# - Otherwise, the user needs to provide a FakeTensor impl (fake impl)
#
# Next, when Python FunctionalTensor sees the op, it will functionalize
# it by emitting a call to an auto_functionalize(op, ["x"], {"x": ...})
# HOP and replacing the mutated inputs with corresponding outputs of this HOP.
# This HOP effectively runs the functional version of the op when
# called: it clones inputs that will be mutated, runs the op, and
# then returns (output, Tensors with the new values)
#
# auto_functionalize_v2 is an improved version of auto_functionalize that better handle
# re-inplacing views.


class AutoFunctionalized(HigherOrderOperator):
    """auto_functionalized(_mutable_op, **kwargs)

    This HOP runs a "functional" version of _mutable_op.

    Concretely, it looks at all the arguments that are mutable through
    _mutable_op's operator schema, clones those kwargs, runs
    `out = _mutable_op(**kwargs)` with the cloned values, and then returns the
    operator output concatenated with the cloned values that were mutated.

    We have some restrictions on `_mutable_op`.
    See `can_auto_functionalize` for the restrictions. We can likely lift
    many of these if users request it.

    The reason why _mutable_op is prefixed with an
    underscore is to prevent collisions with kwarg names in **kwargs.
    """

    def __init__(self) -> None:
        super().__init__("auto_functionalized", cacheable=True)

    def __call__(
        self,
        /,
        _mutable_op: OpOverload,
        **kwargs: Any,
    ) -> tuple[Any, tuple[Tensor, ...]]:
        if not can_auto_functionalize(_mutable_op):
            raise AssertionError(f"Cannot auto-functionalize op {_mutable_op}")
        if not isinstance(kwargs, dict):
            raise AssertionError(f"kwargs must be a dict, got {type(kwargs)}")
        # pyrefly: ignore [missing-attribute]
        return super().__call__(_mutable_op, **kwargs)


auto_functionalized = AutoFunctionalized()
auto_functionalized.__module__ = "torch.ops.higher_order"

auto_functionalized.fallthrough(DispatchKey.AutogradCPU)
auto_functionalized.fallthrough(DispatchKey.AutogradCUDA)


_MutableOpType = OpOverload | HigherOrderOperator


class AutoFunctionalizedV2(HigherOrderOperator):
    """auto_functionalized_v2(_mutable_op, **kwargs)

    This HOP runs a "functional" version of _mutable_op.
    Unlike AutoFunctionalized, this version is improved to better handle
    view tensors. This version is only used in non export mode.
    """

    def __init__(self) -> None:
        super().__init__("auto_functionalized_v2", cacheable=True)

    def __call__(
        self,
        /,
        _mutable_op: _MutableOpType,
        **kwargs: Any,
    ) -> tuple[Any, tuple[Tensor, ...]]:
        _op_to_check: OpOverload | HopInstance | None = None
        if isinstance(_mutable_op, HigherOrderOperator):
            _op_to_check = HopInstance(
                _mutable_op,
                SchemaHolder.from_tree_spec(kwargs.get("_op_schema")).schema,  # type: ignore[arg-type]
            )
        else:
            _op_to_check = _mutable_op

        if _op_to_check is None:
            raise AssertionError("_op_to_check cannot be None")
        if not can_auto_functionalize(_op_to_check):
            raise AssertionError(f"Cannot auto-functionalize op {_op_to_check}")
        if not isinstance(kwargs, dict):
            raise AssertionError(f"kwargs must be a dict, got {type(kwargs)}")
        # pyrefly: ignore [missing-attribute]
        return super().__call__(_mutable_op, **kwargs)


auto_functionalized_v2 = AutoFunctionalizedV2()
auto_functionalized_v2.__module__ = "torch.ops.higher_order"

auto_functionalized_v2.fallthrough(DispatchKey.AutogradCPU)
auto_functionalized_v2.fallthrough(DispatchKey.AutogradCUDA)


def can_auto_functionalize(
    op: OperatorBase | HopInstance,
) -> bool:
    if isinstance(op, HopInstance):
        # HOPs that implement gen_schema and schema is not functional are auto_functionalizable.
        if not _has_gen_schema(op._op):
            return False

    else:
        if not isinstance(op, OpOverload):
            return False

        if torch._library.utils.is_builtin(op):
            # We control the built-ins. These may (in rare cases)
            # do input metadata mutation (which we have banned on custom ops)
            return False

    schema = op._schema
    if not schema.is_mutable:
        return False
    schema = op._schema

    for arg in schema.arguments:
        if arg.alias_info is None:
            continue
        if not arg.alias_info.is_write:
            continue
        if torch._library.utils.is_tensor_like_type(arg.type):
            continue
        if torch._library.utils.is_tensorlist_like_type(arg.type):
            continue
        return False

    if len(schema.returns) == 1 and isinstance(schema.returns[0].type, torch.NoneType):
        # Skip schema returns -> None
        return True
    if isinstance(op, OpOverload):
        if torch._library.utils.is_out(op):
            # Out ops have aliased returns (returns alias the mutable args).
            # This is fine because the mutable args are write-only output buffers.
            pass
        elif torch._library.utils.is_inplace(op):
            # Inplace ops have aliased returns (return aliases the first mutable arg).
            pass
        else:
            # The returns of OpOverload must not alias anything
            for ret in schema.returns:
                if ret.alias_info is None and type(ret.type) is torch.TensorType:
                    continue
                # Not yet supported: List[Tensor] return.
                return False
        if torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), "Functionalize"):
            return False
    return True


def get_mutable_args_from_schema(
    schema: torch.FunctionSchema,
) -> tuple[list[str], list[torch.Type]]:
    """
    Returns the list of argument names that get mutated according to the
    schema and their types.
    """
    mutable_args_names = [
        arg.name
        for arg in schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]

    mutable_args_types = [
        arg.type
        for arg in schema.arguments
        if arg.alias_info is not None and arg.alias_info.is_write
    ]
    return mutable_args_names, mutable_args_types  # type: ignore[return-value]


def get_mutable_args(op: OpOverload) -> tuple[list[str], list[torch.Type]]:
    return get_mutable_args_from_schema(op._schema)


def do_auto_functionalize(
    mode: "torch._subclasses.functional_tensor.FunctionalTensorMode",
    op: OpOverload,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Functionalizes a call to op(*args, **kwargs) by emitting a call to
    `outs = auto_functionalized(op, normalized_kwargs)`
    and replacing the mutated (args, kwargs) with the corresponding outputs.

    The normalized_kwargs are just the (args, kwargs), but all in kwarg form.
    This makes handling easier for the auto_functionalized HOP.
    """
    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI(mode=mode)

    # All of the (args, kwargs), but all as kwargs. The names for the
    # args come from the schema. This makes it easier for us to work with them.
    normalized_kwargs = {}
    schema = op._schema
    for idx, arg in enumerate(schema.arguments):
        # NB: torch_dispatch kwargs are the args defined as kwarg-only in the schema
        if arg.name in kwargs:
            normalized_kwargs[arg.name] = kwargs[arg.name]
        elif idx < len(args):
            # if its out of bounds we don't need to do anything
            # as it means the optional arg was passed with its default
            # value
            normalized_kwargs[arg.name] = args[idx]
        else:
            normalized_kwargs[arg.name] = arg.default_value

    unwrapped_kwargs = ctx.unwrap_tensors(normalized_kwargs)  # type: ignore[arg-type]
    if "self" in unwrapped_kwargs or "self_" in unwrapped_kwargs:
        warnings.warn(
            "Using `self` or `self_` as an argument in the definition of custom ops may lead to ambiguous parsing. "
            "Please consider using a different name for this argument to avoid potential issues.",
            stacklevel=2,
        )
    with ctx.redispatch_to_next():
        unwrapped_outs = auto_functionalized(
            op,
            **unwrapped_kwargs,  # type: ignore[arg-type]
        )

    # List of the name of args that get mutated (according to the schema)
    mutable_args_names, _ = get_mutable_args(op)

    unwrapped_actual_out: Any | tuple[Any] = unwrapped_outs[: -len(mutable_args_names)]
    unwrapped_mutable_out = unwrapped_outs[-len(mutable_args_names) :]

    if len(op._schema.returns) == 0:
        if unwrapped_actual_out[0] is not None:
            raise AssertionError(
                f"Expected None for op with no returns, got {unwrapped_actual_out[0]}"
            )
        unwrapped_actual_out = None
    elif len(op._schema.returns) == 1:
        if len(unwrapped_actual_out) != 1:
            raise AssertionError(f"Expected 1 output, got {len(unwrapped_actual_out)}")
        unwrapped_actual_out = unwrapped_actual_out[0]
    else:
        if len(unwrapped_actual_out) != len(op._schema.returns):
            raise AssertionError(
                f"Expected {len(op._schema.returns)} outputs, got {len(unwrapped_actual_out)}"
            )

    for name, unwrapped_out in zip(mutable_args_names, unwrapped_mutable_out):
        # Can be None if input was `Tensor(a!)?`
        if unwrapped_out is None:
            continue

        # We only handle Tensor or List[Tensor] here for now.
        def sync_update(o, orig_arg):
            ctx.replace(orig_arg, o)
            ctx.commit_update(orig_arg)
            ctx.sync(orig_arg)

        orig_arg = normalized_kwargs[name]

        if isinstance(unwrapped_out, torch.Tensor):
            sync_update(unwrapped_out, orig_arg)
        elif isinstance(unwrapped_out, list) and all(
            isinstance(o, torch.Tensor) for o in unwrapped_out
        ):
            if len(orig_arg) != len(unwrapped_out):
                raise AssertionError(
                    f"orig_arg length ({len(orig_arg)}) != unwrapped_out length ({len(unwrapped_out)})"
                )
            for orig_a, o in zip(orig_arg, unwrapped_out):
                sync_update(o, orig_a)
        else:
            raise RuntimeError(
                f"unsupported type for auto-functionalization: {unwrapped_out}"
            )

    return ctx.wrap_tensors(unwrapped_actual_out)  # type: ignore[arg-type]


# Wrapper for GraphModule that applies functionalization during execution to enable
# epilogue graph inlining and better fusion opportunities in subgraphs
# When tracing this wrapper, we'll get a graph module with epilogue.
#
# We want to hash it according to the original graph module, so that when we go
# from Functional mode -> fake mode for multiple invoke_subgraph calls that share,
# the same inner graph module, we can hit the cache.
class FunctionalCallableWithEpilogue:
    def __init__(self, orig_callable: Callable):
        self.orig_callable = orig_callable
        # Propagate so callers pass inputs as a list, enabling input deallocation.
        self._boxed_call = getattr(orig_callable, "_boxed_call", False)

    def __call__(self, *args, **kwargs):
        # Functionalize to inline the epilogue graph (copy_ ops) for better fusion.
        # Python functionalization so mutable custom ops reach auto_functionalized_v2.
        from torch._subclasses.functional_tensor import (
            dispatch_functionalize,
            FunctionalTensorMode,
        )

        functionalized = dispatch_functionalize(
            self.orig_callable,
            FunctionalTensorMode(),
            propagate_input_mutations=True,
        )
        if self._boxed_call:
            # Not all callers respect _boxed_call (e.g. reenter_make_fx
            # always calls f(*args)). Detect which convention was used.
            if len(args) == 1 and isinstance(args[0], list):
                return tuple(functionalized(args[0]))
            return tuple(functionalized(list(args)))
        return functionalized(*args, **kwargs)

    def __hash__(self):
        return id(self.orig_callable)


def do_auto_functionalize_v2(
    mode: "torch._subclasses.functional_tensor.FunctionalTensorMode",
    op: OpOverload | HopInstance,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    from torch._subclasses.functional_tensor import PythonFunctionalizeAPI

    ctx = PythonFunctionalizeAPI(mode=mode)

    # All of the (args, kwargs), but all as kwargs. The names for the
    # args come from the schema. This makes it easier for us to work with them.
    normalized_kwargs = {}

    schema = op._schema
    # pyrefly: ignore [bad-assignment]
    op = op._op if isinstance(op, HopInstance) else op
    if not isinstance(op, get_args(_MutableOpType)):
        raise AssertionError(f"Expected _MutableOpType, got {type(op)}")

    subgraph_arg_names = {
        arg_info.name
        for arg_info in schema.arguments
        if isinstance(arg_info.type, torch._C.AnyType)
    }

    def _maybe_functionalize(name: str, arg: Any) -> Any:
        if name in subgraph_arg_names and callable(arg):
            return FunctionalCallableWithEpilogue(arg)
        return arg

    args = tuple(
        _maybe_functionalize(schema.arguments[i].name, a)
        if i < len(schema.arguments)
        else a
        for i, a in enumerate(args)
    )
    kwargs = {k: _maybe_functionalize(k, v) for k, v in kwargs.items()}

    for idx, arg in enumerate(schema.arguments):
        # NB: torch_dispatch kwargs are the args defined as kwarg-only in the schema
        if arg.name in kwargs:
            normalized_kwargs[arg.name] = kwargs[arg.name]
        elif idx < len(args):
            # if its out of bounds we don't need to do anything
            # as it means the optional arg was passed with its default
            # value
            normalized_kwargs[arg.name] = args[idx]
        else:
            normalized_kwargs[arg.name] = arg.default_value

    if isinstance(op, OpOverload) and torch._library.utils.is_out(op):
        return _do_auto_functionalize_v2_for_out_operator(
            ctx, op, schema, normalized_kwargs
        )
    if isinstance(op, OpOverload) and torch._library.utils.is_inplace(op):
        return _do_auto_functionalize_v2_for_inplace_operator(
            ctx, op, schema, normalized_kwargs
        )
    return _do_auto_functionalize_v2_for_generic_mutable_operator(
        ctx, op, schema, normalized_kwargs
    )


def _do_auto_functionalize_v2_for_out_operator(
    ctx: Any,
    op: OpOverload,
    schema: Any,
    normalized_kwargs: dict[str, Any],
) -> Any:
    """Handle functionalization for out= operators.

    Out= operators have write-only mutable args. These are not inputs to
    auto_functionalized_v2; we encode their tensor properties so the dense
    impl can create empty tensors.
    """
    mutable_args_names, _ = get_mutable_args_from_schema(schema)

    # Save references to the original FunctionalTensors for post-call sync
    out_arg_originals = [normalized_kwargs[name] for name in mutable_args_names]

    # Encode tensor properties and remove the actual out tensors from kwargs
    for arg_name in mutable_args_names:
        arg = normalized_kwargs[arg_name]
        normalized_kwargs[f"_{arg_name}_size"] = arg.size()
        normalized_kwargs[f"_{arg_name}_stride"] = arg.stride()
        normalized_kwargs[f"_{arg_name}_dtype"] = arg.dtype
        normalized_kwargs[f"_{arg_name}_device"] = arg.device
        del normalized_kwargs[arg_name]

    unwrapped_kwargs = ctx.unwrap_tensors(normalized_kwargs)  # type: ignore[arg-type]
    auto_func_kwargs = dict(unwrapped_kwargs, _all_bases=[])

    with ctx.redispatch_to_next():
        unwrapped_outs = auto_functionalized_v2(
            op,
            **auto_func_kwargs,  # type: ignore[arg-type]
        )

    # For out= ops, the functional HOP returns exactly the out tensors.
    # auto_functionalized_v2 returns a bare tensor for single return,
    # tuple for multiple.
    results = unwrapped_outs if isinstance(unwrapped_outs, tuple) else (unwrapped_outs,)
    for orig_arg, result in zip(out_arg_originals, results, strict=True):
        ctx.replace(orig_arg, result)
        ctx.commit_update(orig_arg)
        ctx.sync(orig_arg)

    return ctx.wrap_tensors(unwrapped_outs)  # type: ignore[arg-type]


def _do_auto_functionalize_v2_for_inplace_operator(
    ctx: Any,
    op: OpOverload,
    schema: Any,
    normalized_kwargs: dict[str, Any],
) -> Any:
    """Handle functionalization for inplace operators.

    Inplace operators mutate their first positional argument (self) and return
    it. Unlike out= operators (write-only), self is read by the operator, so
    it must be passed as a base for clone analysis during reinplacing.
    """
    mutable_args_names, mutable_args_types = get_mutable_args_from_schema(schema)

    self_arg_name = mutable_args_names[0]
    self_arg = normalized_kwargs[self_arg_name]

    # self is both read and written; pass its base for clone analysis
    base = self_arg if get_base(self_arg) is None else get_base(self_arg)

    arg_to_base_index: dict[str, Any] = {self_arg_name: 0}
    write_view_information_to_args(
        mutable_args_names, mutable_args_types, normalized_kwargs, arg_to_base_index
    )
    del normalized_kwargs[self_arg_name]

    unwrapped_kwargs = ctx.unwrap_tensors(normalized_kwargs)  # type: ignore[arg-type]
    all_bases_unwrapped = ctx.unwrap_tensors([base])
    auto_func_kwargs = dict(unwrapped_kwargs, _all_bases=all_bases_unwrapped)

    with ctx.redispatch_to_next():
        unwrapped_outs = auto_functionalized_v2(
            op,
            **auto_func_kwargs,  # type: ignore[arg-type]
        )

    # HOP returns (op_result, mutated_self_base).
    # op_result aliases mutated_self_base (inplace semantics).
    unwrapped_result = unwrapped_outs[0]
    unwrapped_mutated_base = unwrapped_outs[1]

    ctx.replace(base, unwrapped_mutated_base)
    ctx.commit_update(base)
    ctx.sync(base)

    return ctx.wrap_tensors(unwrapped_result)  # type: ignore[arg-type]


def _do_auto_functionalize_v2_for_generic_mutable_operator(
    ctx: Any,
    op: Any,
    schema: Any,
    normalized_kwargs: dict[str, Any],
) -> Any:
    """Handle functionalization for generic mutable operators via the
    auto_functionalized_v2 HOP."""
    mutable_args_names, mutable_args_types = get_mutable_args_from_schema(schema)

    # A list of all bases of mutable args without duplication
    all_bases: list[Any] = []
    all_bases_addresses: list[int] = []

    # Map arg_name to the index of its base in all_bases.
    arg_to_base_index: dict[str, Any] = {}

    def update_dict(tensor, arg_name, index=None):
        base = tensor if get_base(tensor) is None else get_base(tensor)

        def set_result(base_index):
            if index is None:
                arg_to_base_index[arg_name] = base_index
            else:
                arg_to_base_index[arg_name][index] = base_index

        if not all_bases_addresses.__contains__(base._cdata):
            all_bases_addresses.append(base._cdata)
            all_bases.append(base)
            set_result(len(all_bases) - 1)
        else:
            set_result(all_bases_addresses.index(base._cdata))

    for arg_name in mutable_args_names:
        arg = normalized_kwargs[arg_name]
        if arg is None:
            continue

        if isinstance(arg, list):
            arg_to_base_index[arg_name] = {}
            for i, tensor in enumerate(arg):
                if tensor is None:
                    arg_to_base_index[arg_name].append(None)
                    continue

                update_dict(tensor, arg_name, i)

        else:
            update_dict(arg, arg_name)

    # add view_meta for each args into unwrapped_kwargs.
    write_view_information_to_args(
        mutable_args_names,
        mutable_args_types,
        normalized_kwargs,
        arg_to_base_index,
    )

    # remove mutated args from the kwargs (its a function of _all_bases now)
    for arg_name in mutable_args_names:
        del normalized_kwargs[arg_name]  # type: ignore[arg-type]

    unwrapped_kwargs = ctx.unwrap_tensors(normalized_kwargs)  # type: ignore[arg-type]
    if "self" in unwrapped_kwargs or "self_" in unwrapped_kwargs:
        warnings.warn(
            "Using `self` or `self_` as an argument in the definition of custom ops may lead to ambiguous parsing. "
            "Please consider using a different name for this argument to avoid potential issues.",
            stacklevel=2,
        )
    all_basis_unwrapped = ctx.unwrap_tensors(all_bases)

    if "_all_bases" in unwrapped_kwargs:
        raise AssertionError(f"_all_bases already in unwrapped_kwargs for {op}")
    auto_func_kwargs = dict(unwrapped_kwargs, _all_bases=all_basis_unwrapped)
    if isinstance(op, HigherOrderOperator):
        if "_ops_schema" in unwrapped_kwargs:
            raise AssertionError(f"_ops_schema already in unwrapped_kwargs for {op}")
        # We pass in the tree_spec of tree_flatten(SchemaHolder) to make it proxable
        auto_func_kwargs.update(
            {"_op_schema": pytree.tree_flatten(SchemaHolder(schema))[1]}
        )

    with ctx.redispatch_to_next():
        unwrapped_outs = auto_functionalized_v2(
            op,
            **auto_func_kwargs,  # type: ignore[arg-type]
        )

    unwrapped_actual_out: Any | tuple[Any] = (
        unwrapped_outs if len(all_bases) == 0 else unwrapped_outs[: -len(all_bases)]
    )

    unwrapped_mutable_out = (
        [] if len(all_bases) == 0 else unwrapped_outs[-len(all_bases) :]
    )

    if isinstance(op, HigherOrderOperator):
        if len(schema.returns) <= 0:
            raise AssertionError(
                f"hop is expected to return at least one output {schema}."
            )
        if len(unwrapped_actual_out) != len(schema.returns):
            raise AssertionError(
                f"Expected {len(schema.returns)} outputs, got {len(unwrapped_actual_out)}"
            )
    else:
        if len(schema.returns) == 0:
            if unwrapped_actual_out[0] is not None:
                raise AssertionError(
                    f"Expected None for op with no returns, got {unwrapped_actual_out[0]}"
                )
            unwrapped_actual_out = None
        elif len(schema.returns) == 1:
            if len(unwrapped_actual_out) != 1:
                raise AssertionError(
                    f"Expected 1 output, got {len(unwrapped_actual_out)}"
                )
            unwrapped_actual_out = unwrapped_actual_out[0]
        else:
            if len(unwrapped_actual_out) != len(schema.returns):
                raise AssertionError(
                    f"Expected {len(schema.returns)} outputs, got {len(unwrapped_actual_out)}"
                )

    for orig_arg, unwrapped_out in zip(all_bases, unwrapped_mutable_out):
        # Can be None if input was `Tensor(a!)?`
        if unwrapped_out is None:
            continue

        # We only handle Tensor or List[Tensor] here for now.
        def sync_update(o, orig_arg):
            ctx.replace(orig_arg, o)
            ctx.commit_update(orig_arg)
            ctx.sync(orig_arg)

        if isinstance(unwrapped_out, torch.Tensor):
            sync_update(unwrapped_out, orig_arg)
        elif isinstance(unwrapped_out, list) and all(
            isinstance(o, torch.Tensor) for o in unwrapped_out
        ):
            if len(orig_arg) != len(unwrapped_out):
                raise AssertionError(
                    f"orig_arg length ({len(orig_arg)}) != unwrapped_out length ({len(unwrapped_out)})"
                )
            for orig_a, o in zip(orig_arg, unwrapped_out):
                sync_update(o, orig_a)
        else:
            raise RuntimeError(
                f"unsupported type for auto-functionalization: {unwrapped_out}"
            )

    return ctx.wrap_tensors(unwrapped_actual_out)  # type: ignore[arg-type]


# auto_functionalize functions
@auto_functionalized.py_impl(DispatchKey.CompositeExplicitAutograd)
def auto_functionalized_dense(
    _mutable_op: OpOverload,
    _only_clone_these_tensors: tuple[str, ...] | None = None,
    **kwargs: Any,
) -> tuple[Any, tuple[Tensor, ...]]:
    new_kwargs = dict(**kwargs)
    result = []

    _mutable_args_names, _ = get_mutable_args(_mutable_op)
    for name in _mutable_args_names:
        if (
            _only_clone_these_tensors is not None
            and name not in _only_clone_these_tensors
        ):
            new_kwargs[name] = kwargs[name]
        else:
            new_kwargs[name] = (
                [clone_preserve_strides(x) for x in kwargs[name]]
                if kwargs[name] is not None and isinstance(kwargs[name], list)
                else (
                    clone_preserve_strides(kwargs[name])
                    if kwargs[name] is not None
                    else None
                )
            )
        result.append(new_kwargs[name])
    out = _mutable_op(**new_kwargs)

    if isinstance(out, tuple):
        return (*out, *result)  # type: ignore[return-value]
    else:
        return (out, *result)  # type: ignore[return-value]


@register_fake(auto_functionalized, skip_cache=True)
def auto_functionalized_fake(
    _mutable_op: OpOverload,
    **kwargs: Any,
) -> tuple[Any, tuple[Tensor, ...]]:
    result = auto_functionalized_dense(
        _mutable_op, _only_clone_these_tensors=None, **kwargs
    )
    return result


@auto_functionalized.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_proxy(
    mode,
    _mutable_op: OpOverload,
    **kwargs: Any,
) -> tuple[Any, tuple[Tensor, ...]]:
    with disable_proxy_modes_tracing():
        out = auto_functionalized(_mutable_op, **kwargs)

    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)
    out_proxy = mode.tracer.create_proxy(
        "call_function",
        auto_functionalized,
        (_mutable_op,),
        proxy_kwargs,
    )
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


@auto_functionalized.py_functionalize_impl
def auto_functionalized_func(ctx, _mutable_op, **kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        result = auto_functionalized(_mutable_op, **unwrapped_kwargs)
    return ctx.wrap_tensors(result)


# auto_functionalized_v2 functions
@auto_functionalized_v2.py_impl(DispatchKey.CompositeExplicitAutograd)
def auto_functionalized_v2_dense(
    _mutable_op: _MutableOpType,
    _only_clone_these_bases: tuple[int, ...] | None = None,
    **kwargs: Any,
) -> tuple[Any, tuple[Tensor, ...]]:
    _all_bases: list[Tensor] = kwargs.pop("_all_bases", [])
    if _only_clone_these_bases is None:
        _only_clone_these_bases = tuple(range(len(_all_bases)))

    if isinstance(_mutable_op, OpOverload):
        schema: torch._C.FunctionSchema = _mutable_op._schema
    else:
        schema = pytree.tree_unflatten([], kwargs.pop("_op_schema")).schema

    if isinstance(_mutable_op, OpOverload):
        _callable_op: HopInstance | OpOverload = _mutable_op
    else:
        if not isinstance(schema, HopSchema):
            raise AssertionError(f"Expected HopSchema, got {type(schema)}")
        _callable_op = HopInstance(_mutable_op, schema)

    _is_out = isinstance(_mutable_op, OpOverload) and torch._library.utils.is_out(
        _mutable_op
    )

    op_kwargs_new, all_bases_new = _generate_new_op_kwargs_from_bases(
        schema,
        kwargs,
        _all_bases,
        _only_clone_these_bases,
        _is_out,
    )

    out = call_op(
        _callable_op,
        tuple(),
        op_kwargs_new,
    )

    if _is_out:
        return out  # type: ignore[return-value]

    if isinstance(out, tuple):
        return (*out, *all_bases_new)  # type: ignore[return-value]
    else:
        return (out, *all_bases_new)  # type: ignore[return-value]


def _generate_new_op_kwargs_from_bases(
    schema, kwargs, all_bases, _only_clone_these_bases, _is_out
):
    mutable_args_names, mutable_args_types = get_mutable_args_from_schema(schema)

    if _is_out:
        # For out= ops, _all_bases is empty. Create empty tensors from the
        # metadata that was encoded by _do_auto_functionalize_v2_for_out_operator.
        new_kwargs = dict(**kwargs)
        created_out_tensors = []
        for arg_name in mutable_args_names:
            size = new_kwargs.pop(f"_{arg_name}_size")
            stride = new_kwargs.pop(f"_{arg_name}_stride")
            dtype = new_kwargs.pop(f"_{arg_name}_dtype")
            device = new_kwargs.pop(f"_{arg_name}_device")
            t = torch.empty_strided(size, stride, dtype=dtype, device=device)
            new_kwargs[arg_name] = t
            created_out_tensors.append(t)
        return new_kwargs, created_out_tensors

    args_view_info = read_view_information_from_args(
        mutable_args_names, mutable_args_types, kwargs, all_bases
    )

    def maybe_copy(i, t):
        if t is None:
            return None
        if i in _only_clone_these_bases:
            return clone_preserve_strides(t)
        else:
            return t

    all_bases_new = [maybe_copy(i, t) for i, t in enumerate(all_bases)]

    # create new args
    new_kwargs = dict(**kwargs)

    # re-generate all inputs from all_bases_new using args_view_info and add them to new_kwargs.
    for arg_name in mutable_args_names:
        if args_view_info[arg_name] is None:
            new_kwargs[arg_name] = None
        elif isinstance(args_view_info[arg_name], list):
            new_kwargs[arg_name] = []
            for i, elem in enumerate(args_view_info[arg_name]):
                if elem is None:
                    new_kwargs[arg_name].append(None)
                else:
                    view_info = args_view_info[arg_name][i]
                    new_kwargs[arg_name].append(
                        view_info.regenerate_view(all_bases_new)
                    )
        else:
            new_kwargs[arg_name] = args_view_info[arg_name].regenerate_view(
                all_bases_new
            )

    return new_kwargs, all_bases_new


@register_fake(auto_functionalized_v2, skip_cache=True)
def auto_functionalized_v2_fake(
    _mutable_op: _MutableOpType,
    **kwargs: dict[str, Any],
) -> tuple[Any, tuple[Tensor, ...]]:
    result = auto_functionalized_v2_dense(
        _mutable_op, _only_clone_these_bases=None, **kwargs
    )
    return result


@auto_functionalized_v2.py_impl(ProxyTorchDispatchMode)
def auto_functionalized_v2_proxy(
    mode,
    _mutable_op: _MutableOpType,
    **kwargs: Any,
) -> tuple[Any, tuple[Tensor, ...]]:
    if isinstance(_mutable_op, HigherOrderOperator):
        # Note [materialize callable inputs as graph]
        # Below code materializes the callable inputs to the hop as graph modules.
        # kwargs may contain general callables, that are not proxable e.g. FunctionWithNoFreeVars
        # this could happen when we auto_functionalize the backward of the hop,
        # where backward fn is a callable that wraps forward graph module.
        # This function materialize the callable args according to the schema of the hop.

        # We cannot materialize the callables in kwargs directly because the inputs to callable
        # vary from hops to hop. To make the materialiation process generic to all hops,
        # we trace a function that wraps the hop and let each hop itself figure out how to trace
        # its callable inputs. Then we look at the schema of the traced hop node and replace the
        # callable in original kwarg with the traced subgraphs.
        #
        # Specifically, we first trace a wrapped_fn that calls into the hop. Then we look for the
        # hop node in the traced graph and graph module inputs to the hop. Finally, we replace the
        # original kwarg's callable with the graph module.
        all_bases = kwargs.get("_all_bases", [])
        _only_clone_these_bases = kwargs.get("_only_clone_these_bases")
        if _only_clone_these_bases is None:
            _only_clone_these_bases = tuple(range(len(all_bases)))

        schema = pytree.tree_unflatten([], kwargs.get("_op_schema")).schema  # type: ignore[arg-type]
        new_kwargs, _ = _generate_new_op_kwargs_from_bases(
            schema,
            {k: v for k, v in kwargs.items() if k not in ("_all_bases", "_op_schema")},
            all_bases,
            _only_clone_these_bases,
            _is_out=False,
        )

        _, materialized_kwargs = materialize_callable_in_args(
            HopInstance(_mutable_op, schema), tuple(), new_kwargs
        )

        # Only replace the callables in kwargs with the materialized subgraphs.
        # The rest of the kwargs are kept unchanged.
        subgraph_arg_names = {
            arg_info.name
            for arg_info in schema.arguments
            if isinstance(arg_info.type, torch._C.AnyType)
        }
        for k, v in kwargs.items():
            if k in subgraph_arg_names and callable(v):
                if k not in materialized_kwargs or not isinstance(
                    materialized_kwargs[k], torch.fx.GraphModule
                ):
                    raise AssertionError(
                        f"Expected {k} to be in materialized_kwargs as a GraphModule"
                    )
                kwargs[k] = materialized_kwargs[k]

    with disable_proxy_modes_tracing():
        out = auto_functionalized_v2(_mutable_op, **kwargs)

    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)

    if isinstance(_mutable_op, HigherOrderOperator):

        def _maybe_register_subgraph(val: Any):
            if isinstance(val, torch.fx.GraphModule):
                _, graph_name = unique_graph_id(
                    mode, prefix="auto_functionalized_subgraph"
                )
                mode.tracer.root.register_module(graph_name, val)
                return val
            return val

        proxy_kwargs = pytree.tree_map(_maybe_register_subgraph, proxy_kwargs)

    out_proxy = mode.tracer.create_proxy(
        "call_function",
        auto_functionalized_v2,
        (_mutable_op,),
        proxy_kwargs,
    )
    result = track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)
    return result


@auto_functionalized_v2.py_functionalize_impl
def auto_functionalized_v2_func(ctx, _mutable_op, **kwargs):
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)
    with ctx.redispatch_to_next():
        result = auto_functionalized_v2(_mutable_op, **unwrapped_kwargs)
    return ctx.wrap_tensors(result)
<|endoftext|># mypy: allow-untyped-defs
import functools
import re

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented, register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree


__all__ = ["inline_asm_elementwise"]


class InlineAsmElementwiseOp(HigherOrderOperator):
    """Execute inline PTX assembly elementwise over tensors.

    This is an elementwise map where the function body is inline assembly.
    Input tensors are implicitly broadcast to the same shape.

    Each invocation of the inline asm processes ``pack`` elements at a time.
    Exactly which set of inputs a given invocation receives is unspecified.

    Output strides follow PyTorch's standard pointwise striding propagation
    rules.

    In eager mode, the assembly is executed via the CUDA Jiterator.  Under
    ``torch.compile`` the assembly is lowered to Triton's
    ``tl.inline_asm_elementwise`` via Inductor, which allows fusion with
    surrounding operators.

    Args:
        *inputs: Input tensors whose values are passed to the asm block.
        asm_str: PTX assembly string. Operands use ``$N`` syntax
            (e.g. ``$0`` for the first output, ``$1`` for the first input).
        constraints: Inline-asm constraints in LLVM format. Output constraints
            are prefixed with ``=`` (e.g. ``"=f,f,f"`` for one float output
            and two float inputs).
        dtype: Element type of the returned tensor.
        is_pure: Must be ``True``. If true, the compiler may assume the asm
            block has no side-effects.
        pack: Number of elements processed per asm invocation.  When
            ``pack > 1``, the constraint string must list ``pack`` outputs
            and ``pack`` copies of each input.  Requires ``torch.compile``.

    Returns:
        A tensor with the broadcast shape of the inputs and the given dtype.

    Example::

        >>> # xdoctest: +SKIP(requires CUDA)
        >>> # Float32 fused multiply-add via PTX
        >>> result = inline_asm_elementwise(
        ...     a, b, c,
        ...     asm_str="fma.rn.f32 $0, $1, $2, $3;",
        ...     constraints="=f,f,f,f",
        ...     dtype=torch.float32,
        ... )

        >>> # xdoctest: +SKIP(requires CUDA)
        >>> # pack=2: each asm invocation processes two elements
        >>> result = inline_asm_elementwise(
        ...     x,
        ...     asm_str="mov.b32 $0, $2; mov.b32 $1, $3;",
        ...     constraints="=r,=r,r,r",
        ...     dtype=torch.float32,
        ...     pack=2,
        ... )
    """

    def __init__(self):
        super().__init__("inline_asm_elementwise")

    def __call__(
        self,
        *inputs: torch.Tensor,
        asm_str: str,
        constraints: str,
        dtype: torch.dtype,
        is_pure: bool = True,
        pack: int = 1,
    ) -> torch.Tensor:
        if not is_pure:
            raise ValueError("inline_asm_elementwise only supports is_pure=True")
        # pyrefly: ignore [missing-attribute]
        return super().__call__(
            *inputs,
            asm_str=asm_str,
            constraints=constraints,
            dtype=dtype,
            is_pure=True,
            pack=pack,
        )


inline_asm_elementwise = InlineAsmElementwiseOp()


def _parse_constraints(constraints: str) -> tuple[int, int]:
    parts = [p.strip() for p in constraints.split(",")]
    n_outputs = sum(1 for p in parts if p.startswith("="))
    n_inputs = len(parts) - n_outputs
    return n_outputs, n_inputs


_DTYPE_TO_CUDA_TYPE = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "__half",
    torch.bfloat16: "__nv_bfloat16",
    torch.int32: "int",
    torch.int64: "long long",
    torch.int16: "short",
    torch.int8: "signed char",
    torch.uint8: "unsigned char",
    torch.uint16: "unsigned short",
    torch.uint32: "unsigned int",
    torch.bool: "bool",
}


_TRITON_ARG_RE = re.compile(r"\$(\d+)")


def _triton_asm_to_cuda_asm(asm_str: str) -> str:
    return _TRITON_ARG_RE.sub(r"%\1", asm_str)


@functools.lru_cache
def _get_jiterator_fn(
    asm_str: str,
    constraints: str,
    n_inputs: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
):
    from torch.cuda.jiterator import _create_jit_fn

    cuda_asm = _triton_asm_to_cuda_asm(asm_str)

    constraint_parts = [p.strip() for p in constraints.split(",")]
    output_constraints = [p.lstrip("=") for p in constraint_parts if p.startswith("=")]
    input_constraints = [p for p in constraint_parts if not p.startswith("=")]

    if input_dtype not in _DTYPE_TO_CUDA_TYPE:
        raise ValueError(f"Unsupported input dtype for inline asm: {input_dtype}")
    if output_dtype not in _DTYPE_TO_CUDA_TYPE:
        raise ValueError(f"Unsupported output dtype for inline asm: {output_dtype}")

    input_type = _DTYPE_TO_CUDA_TYPE[input_dtype]
    output_type = _DTYPE_TO_CUDA_TYPE[output_dtype]

    input_params = ", ".join(f"{input_type} in{i}" for i in range(n_inputs))
    out_constraints_str = ", ".join(f'"={c}"(result)' for c in output_constraints)
    in_constraints_str = ", ".join(
        f'"{c}"(in{i})' for i, c in enumerate(input_constraints)
    )
    escaped_asm = (
        cuda_asm.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    )

    code = f"""
template <typename T>
{output_type} inline_asm_kernel({input_params}) {{
    {output_type} result;
    asm volatile (
        "{escaped_asm}"
        : {out_constraints_str}
        : {in_constraints_str}
    );
    return result;
}}
"""

    return _create_jit_fn(code)


def _inline_asm_dense(*inputs, asm_str, constraints, dtype, is_pure, pack):
    if not inputs:
        raise ValueError("inline_asm_elementwise requires at least one input tensor")

    inputs = torch.broadcast_tensors(*inputs)

    if not inputs[0].is_cuda:
        raise RuntimeError("inline_asm_elementwise only supports CUDA tensors")

    if pack > 1:
        raise RuntimeError(
            "inline_asm_elementwise with pack > 1 requires torch.compile"
        )

    n_outputs, n_inputs = _parse_constraints(constraints)

    if n_outputs != 1:
        raise ValueError(f"Expected 1 output constraint, got {n_outputs}")

    if n_inputs != len(inputs):
        raise ValueError(
            f"Constraint string specifies {n_inputs} inputs but got "
            f"{len(inputs)} tensor(s)"
        )

    # Jiterator generates a single input type for all inputs — mixed dtypes
    # would produce incorrect CUDA code.
    input_dtypes = {inp.dtype for inp in inputs}
    if len(input_dtypes) > 1:
        raise ValueError(
            f"All inputs must have the same dtype for eager execution, "
            f"got {sorted(str(d) for d in input_dtypes)}"
        )

    jit_fn = _get_jiterator_fn(
        asm_str=asm_str,
        constraints=constraints,
        n_inputs=len(inputs),
        input_dtype=inputs[0].dtype,
        output_dtype=dtype,
    )

    return jit_fn(*inputs)


@inline_asm_elementwise.py_impl(DispatchKey.CompositeExplicitAutograd)
def _(*inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    return _inline_asm_dense(
        *inputs,
        asm_str=asm_str,
        constraints=constraints,
        dtype=dtype,
        is_pure=is_pure,
        pack=pack,
    )


inline_asm_elementwise.py_autograd_impl(
    autograd_not_implemented(inline_asm_elementwise, deferred_error=True)
)


def _elementwise_output_like(*inputs, dtype):
    from torch._prims_common import compute_elementwise_output_logical_to_physical_perm

    broadcasted = torch.broadcast_tensors(*inputs)
    l2p_perm, _ = compute_elementwise_output_logical_to_physical_perm(*broadcasted)
    return torch.empty_permuted(
        broadcasted[0].shape, l2p_perm, dtype=dtype, device=broadcasted[0].device
    )


@register_fake(inline_asm_elementwise, skip_cache=True)
def _(*inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    return _elementwise_output_like(*inputs, dtype=dtype)


@inline_asm_elementwise.py_impl(ProxyTorchDispatchMode)
def _(mode, *inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, inputs)

    out_proxy = mode.tracer.create_proxy(
        "call_function",
        inline_asm_elementwise,
        proxy_args,
        {
            "asm_str": asm_str,
            "constraints": constraints,
            "dtype": dtype,
            "is_pure": is_pure,
            "pack": pack,
        },
        name="inline_asm_elementwise",
    )

    out = inline_asm_elementwise(
        *inputs,
        asm_str=asm_str,
        constraints=constraints,
        dtype=dtype,
        is_pure=is_pure,
        pack=pack,
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@inline_asm_elementwise.py_functionalize_impl
def _(ctx, *inputs, asm_str, constraints, dtype, is_pure=True, pack=1):
    unwrapped_inputs = ctx.unwrap_tensors(inputs)

    with ctx.redispatch_to_next():
        res = inline_asm_elementwise(
            *unwrapped_inputs,
            asm_str=asm_str,
            constraints=constraints,
            dtype=dtype,
            pack=pack,
        )
    return ctx.wrap_tensors(res)
<|endoftext|>import ast
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import logging
import operator
import threading
import typing
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional, Protocol, TYPE_CHECKING, Union
from typing_extensions import Never

import sympy

import torch.fx as fx
import torch.utils._pytree as pytree
from torch import SymBool, SymFloat, SymInt, Tensor
from torch._C import _dispatch_keys, DispatchKey
from torch._higher_order_ops.utils import redirect_to_mode, register_fake
from torch._ops import HigherOrderOperator
from torch._prims_common import clone_preserve_strides
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import guard_scalar
from torch.types import IntLikeType
from torch.utils._ordered_set import OrderedSet
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


if TYPE_CHECKING:
    from triton._C.libtriton.ir import (
        module as TritonIRModule,
        operation as TritonIROperation,
    )

    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._dynamo.variables.constant import ConstantVariable
    from torch._dynamo.variables.functions import TritonKernelVariable
    from torch._guards import Source
    from torch._inductor.dependencies import ReadWrites
    from torch._subclasses.functional_tensor import BaseFunctionalizeAPI
    from torch.fx.proxy import Proxy
    from torch.utils._triton import has_triton

    TritonMetaParamsType = dict[str, int]
    TritonGridTupleType = tuple[int | sympy.Expr | SymInt, ...]
    TritonGridCallableType = Callable[[TritonMetaParamsType], tuple[int, ...]]
    TritonGridType = TritonGridTupleType | TritonGridCallableType

    if has_triton():
        from triton.runtime.autotuner import Autotuner, Config as TritonConfig
        from triton.runtime.jit import JITFunction
    else:

        class Autotuner:  # type: ignore[no-redef]
            pass

        class JITFunction:  # type: ignore[no-redef]
            pass

    TritonKernelType = Autotuner | JITFunction
    # mypy specifically complains that TritonAutotunerType is not a valid type if Autotuner is not inside of a Union.
    TritonAutotunerType = Union[Autotuner]  # noqa: UP007

log = logging.getLogger("torch._dynamo")

# e.g. for a host-side Triton TMA API call ``create_2d_tma_descriptor(ptr, 50, 60, 32, 15, 4)``,
# the metadata will look like ``("experimental", ([50, 60], [32, 15], 4))``
TMAExperimentalMetadata = tuple[
    str,  # type of TMA (should be "experimental")
    tuple[
        list[IntLikeType],  # dims
        list[IntLikeType],  # block_dims
        IntLikeType,  # element_size
    ],
]

# e.g. for host-side Triton TMA API call ``TensorDescriptor.from_tensor(ptr, [32, 64])``
# the metadata will look like ``("stable", ([32, 64],))``
TMAStableMetadata = tuple[
    str,  # type of TMA ("experimental" or "stable")
    tuple[list[IntLikeType],],  # block_shape
]


def create_tma_experimental_metadata(
    dims: list[IntLikeType],
    block_dims: list[IntLikeType],
    element_size: IntLikeType,
) -> TMAExperimentalMetadata:
    return ("experimental", (dims, block_dims, element_size))


def maybe_unpack_tma_experimental_metadata(
    tma_meta: TMAExperimentalMetadata | TMAStableMetadata,
) -> tuple[list[IntLikeType], list[IntLikeType], IntLikeType] | None:
    if not tma_meta or len(tma_meta) != 2:
        return None
    if tma_meta[0] == "experimental":
        return tma_meta[1]  # type: ignore[return-value]
    return None


def create_tma_stable_metadata(
    block_shape: list[IntLikeType],
) -> TMAStableMetadata:
    return ("stable", (block_shape,))


def maybe_unpack_tma_stable_metadata(
    tma_meta: TMAExperimentalMetadata | TMAStableMetadata,
) -> tuple[list[IntLikeType]] | None:
    if not tma_meta or len(tma_meta) != 2:
        return None
    if tma_meta[0] == "stable":
        return tma_meta[1]  # type: ignore[return-value]
    return None


# TMADescriptorMetadata maps kernel parameter names to the metadata that allows
# reconstructing TMA descriptors from the underlying tensors (passed as kernel
# arguments in the fx graph, instead of the TMA descriptors).
#
# Since there are two TMA APIs (the old "experimental" API and the new "stable" API),
# each entry in the dict is a tuple that starts with a string, either "experimental"
# or "stable". The second entry in the tuple is another tuple, with data that depends
# on the API type (see TMAExperimentalMetadata and TMAStableMetadata above).
#
# These are stored as raw tuples (instead of classes) for ease of serialization.
TMADescriptorMetadata = dict[
    str,  # kernel parameter name
    TMAExperimentalMetadata | TMAStableMetadata,
]


###############################################################################
# Kernel Side Table


# We cannot put Triton Kernels into the FX graph as the graph nodes
# do not support arbitrary functions.
# Use a side table.
# We use two dicts so that fetching both the kernel and id are O(1)
class KernelSideTable:
    id_to_kernel: dict[int, "TritonKernelType"] = {}
    kernel_to_id: dict["TritonKernelType", int] = {}
    constant_args: dict[int, dict[str, Any]] = {}
    lock = threading.Lock()

    # Returns index on the table
    def add_kernel(self, kernel: "TritonKernelType") -> int:
        with self.lock:
            if kernel in self.kernel_to_id:
                return self.kernel_to_id[kernel]

            idx = len(self.id_to_kernel)
            self.id_to_kernel[idx] = kernel
            self.kernel_to_id[kernel] = idx
            return idx

    # Returns the triton kernel at the given index
    def get_kernel(self, idx: int) -> "TritonKernelType":
        # No need to lock here as fetching from dict is atomic
        if idx not in self.id_to_kernel:
            raise AssertionError(f"Kernel index {idx} not found in id_to_kernel")
        return self.id_to_kernel[idx]

    # Not every constant arg can be added to the graph. Use this side table
    # for constant args.
    def add_constant_args(self, args: dict[str, Any]) -> int:
        with self.lock:
            idx = len(self.constant_args)
            self.constant_args[idx] = args
            return idx

    # Returns the constant args
    def get_constant_args(self, idx: int) -> dict[str, Any]:
        # No need to lock here as fetching from dict is atomic
        if idx not in self.constant_args:
            raise AssertionError(
                f"Constant args index {idx} not found in constant_args"
            )
        return self.constant_args[idx]

    # Resets the table (only meant to be used in unit tests)
    # This is only safe assuming single threaded execution
    def reset_table(self) -> None:
        self.id_to_kernel = {}
        self.kernel_to_id = {}
        self.constant_args = {}


kernel_side_table = KernelSideTable()


###############################################################################
# Mutation Tracker


@dataclasses.dataclass(frozen=True, slots=True)
class Param:
    idx: int


@dataclasses.dataclass(frozen=True, slots=True)
class Intermediate:
    idx: int

    def fake(self) -> bool:
        return self.idx < 0


@dataclasses.dataclass(frozen=True, slots=True)
class Op:
    name: str
    fn_call_name: str | None
    args: list[Param | Intermediate]
    ret: Intermediate = dataclasses.field(repr=False)
    # used for scf.yield: see [Note: scf.yield fix-up]
    sub_idx: int | None = None
    # used for tt.elementwise_inline_asm
    # `is_pure = True` assumes the asm block has no side-effects
    is_pure: bool = False

    def __post_init__(self) -> None:
        if self.name == "tt.call":
            if self.fn_call_name is None:
                raise AssertionError("fn_call_name must not be None for tt.call op")
        else:
            if self.fn_call_name is not None:
                raise AssertionError(
                    f"fn_call_name must be None for non-tt.call op, got {self.fn_call_name}"
                )


def generate_ttir(
    kernel: "TritonKernelType",
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> tuple["TritonIRModule", list[str]]:
    """
    Uses Triton's internal code generation to create TTIR
    """
    import sympy
    import triton
    import triton.runtime.jit
    from triton.compiler.compiler import ASTSource
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    from torch._inductor.utils import (
        get_triton_attrs_descriptor_version,
        triton_version_uses_attrs_dict,
        TritonAttrsDescriptorVersion,
    )
    from torch.utils._triton import has_triton_tensor_descriptor_host_tma

    triton_version = get_triton_attrs_descriptor_version()

    import torch._inductor.ir
    from torch._subclasses.fake_tensor import is_fake_tensor

    if isinstance(kernel, Autotuner):
        if len(kernel.configs) > 0:
            # If we are autotuning, then it doesn't matter which version gets
            # picked for tracing purposes, so lets pick the first one
            kwargs = {**kwargs, **kernel.configs[0].kwargs}
        kernel = kernel.fn

    if not isinstance(kernel, JITFunction):
        raise AssertionError(f"Expected kernel to be a JITFunction, got {type(kernel)}")

    context = triton._C.libtriton.ir.context()
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.compiler.make_backend(target)
    options = backend.parse_options({})

    # ignore backend-specific kwargs same way as in the native Triton code
    # https://github.com/triton-lang/triton/blob/a6bb57d6285e723c58e87dd7cba263db6efff789/python/triton/runtime/jit.py#L594-L596
    # why this is important for user-defined Triton kernels on AMD: https://github.com/pytorch/pytorch/issues/140800
    for name in list(kwargs):
        if name not in kernel.arg_names and name in options.__dict__:
            kwargs.pop(name)

    if len(kwargs) != len(kernel.arg_names):
        raise ValueError(
            "Incorrect number of arguments passed to kernel: "
            f"passed {list(kwargs.keys())}, expected {kernel.arg_names}."
        )

    # Replace all SymExprs with a regular value for TTIR generation
    # Replace all FakeTensor/TensorBox with real tensors
    # These replacements are needed for triton's type, key and config functions
    ordered_args: dict[str, Any] = {}
    for name in kernel.arg_names:
        a = kwargs[name]
        if isinstance(a, (torch.SymInt, torch.SymFloat, torch.SymBool, sympy.Expr)):
            ordered_args[name] = 2
        elif (
            stable_meta := maybe_unpack_tma_stable_metadata(
                # pyrefly: ignore [bad-argument-type]
                tma_descriptor_metadata.get(name, None)
            )
        ) is not None:
            from triton.tools.tensor_descriptor import TensorDescriptor

            block_shape = stable_meta[0]
            with torch._C._DisableTorchDispatch():
                # need 16-byte aligned strides
                elements_per_dim = max(1, 16 // a.dtype.itemsize)
                base_tensor = torch.empty(
                    [elements_per_dim] * len(block_shape), dtype=a.dtype
                )

            ordered_args[name] = TensorDescriptor.from_tensor(base_tensor, block_shape)
        elif is_fake_tensor(a) or isinstance(a, torch._inductor.ir.TensorBox):
            with torch._C._DisableTorchDispatch():
                ordered_args[name] = torch.empty(2, dtype=a.dtype)
        else:
            ordered_args[name] = a

    def is_stable_tensor_descriptor_arg(arg: Any) -> bool:
        if has_triton_tensor_descriptor_host_tma():
            from triton.tools.tensor_descriptor import TensorDescriptor

            if isinstance(arg, TensorDescriptor):
                return True
        return False

    def _is_constexpr_or_none(name: str, arg: Any) -> bool:
        param_idx = kernel.arg_names.index(name)
        return kernel.params[param_idx].is_constexpr or arg is None

    # Note: one would expect that each input to the triton kernel maps to
    # one input parameter in the TTIR. This is _not_ true for TMA descriptors:
    # one TMA descriptor gets converted into:
    #   * one TMA descriptor input
    #   * N strides, for a rank-N tensor
    #   * N sizes, for a rank-N tensor
    # To account for this, we inject some fake arg names as placeholders for
    # the stride and size parameters.
    #
    # Additionally, tensors and scalars are both included as TTIR parameters,
    # whereas `constexpr` are inlined, and None are excluded. We both preserve
    # scalars and tensors as this matters for "odd" ordering,
    # eg. [tensor, scalar, tensor].
    def get_arg_names(name: str, arg: Any) -> list[str]:
        if _is_constexpr_or_none(name, arg):
            return []

        if is_stable_tensor_descriptor_arg(arg):
            stable_meta = maybe_unpack_tma_stable_metadata(
                tma_descriptor_metadata[name]
            )
            if stable_meta is None:
                raise AssertionError(f"Failed to unpack stable TMA metadata for {name}")
            block_shape = stable_meta[0]
            tensor_rank = len(block_shape)
            names = [name]
            names.extend(name + f" STRIDE PLACEHOLDER {i}" for i in range(tensor_rank))
            names.extend(name + f" SIZE PLACEHOLDER {i}" for i in range(tensor_rank))
            return names

        return [name]

    ordered_arg_names = list(
        itertools.chain.from_iterable(
            get_arg_names(name, arg) for name, arg in ordered_args.items()
        )
    )

    def _get_specialization(args):  # type: ignore[no-untyped-def]
        # Support multiple triton versions.
        # This code basically copies JITFunction.run() logic to get the attrs to construct an ASTSource.
        if triton_version == TritonAttrsDescriptorVersion.V1_COMPILER:
            return kernel._get_config(*args)
        elif triton_version in {
            TritonAttrsDescriptorVersion.V2_BACKENDS,
            TritonAttrsDescriptorVersion.V3_BACKENDS_TUPLE,
        }:
            from triton.backends.compiler import AttrsDescriptor  # noqa: F401

            target = triton.runtime.driver.active.get_current_target()
            backend_ = triton.compiler.compiler.make_backend(target)

            return backend_.get_attrs_descriptor(args, kernel.params)
        else:
            if (
                get_triton_attrs_descriptor_version()
                != TritonAttrsDescriptorVersion.V4_DICT
            ):
                raise AssertionError(
                    f"Expected Triton attrs descriptor version V4_DICT, "
                    f"got {get_triton_attrs_descriptor_version()}"
                )
            # specialize_impl switched to create_specialize_impl in https://github.com/triton-lang/triton/pull/6099
            if hasattr(triton.runtime.jit, "create_specialize_impl"):
                try:
                    # Latest versions of Triton take specialize_extra as an arg to create_specialize_impl
                    specialize_impl = triton.runtime.jit.create_specialize_impl(
                        specialize_extra=backend.get_arg_specialization
                    )
                except TypeError:  # Unknown arg `specialize_extra`
                    # Older versions of Triton take specialize_extra as an arg to specialize_impl
                    specialize_impl = functools.partial(
                        triton.runtime.jit.create_specialize_impl(),
                        specialize_extra=backend.get_arg_specialization,
                    )
            # create_specialize_impl is removed in https://github.com/triton-lang/triton/pull/7771
            # switch to native_specialize_impl instead
            elif hasattr(triton.runtime.jit, "native_specialize_impl"):
                from triton.backends import BaseBackend
                from triton.runtime.jit import native_specialize_impl

                def _native_specialize_impl(
                    arg: Any,
                    is_const: bool = False,
                    specialize_value: bool = True,
                    align: bool = True,
                ) -> Callable:
                    return native_specialize_impl(
                        BaseBackend, arg, is_const, specialize_value, align
                    )

                specialize_impl = _native_specialize_impl
            else:
                from triton.runtime.jit import specialize_impl as specialize_impl_orig

                specialize_impl = functools.partial(
                    specialize_impl_orig,
                    specialize_extra=backend.get_arg_specialization,
                )

            from triton._utils import find_paths_if, get_iterable_path

            # logic is copied from: binder = create_function_from_signature(self.signature, self.params, backend)
            attrvals = []
            for arg, kp in zip(args, kernel.params):
                if kp.is_constexpr:
                    attrvals.append(arg)
                else:
                    spec = specialize_impl(
                        arg,
                        is_const=kp.is_const,
                        specialize_value=not kp.do_not_specialize,
                        align=not kp.do_not_specialize_on_alignment,
                    )
                    # pyrefly: ignore [unsupported-operation]
                    attrvals.append(spec[1])

            attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
            attrs = {
                k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs
            }
            return attrs

    specialization = _get_specialization(ordered_args.values())
    # Triton explicitly interprets ASTSource.constants entries as constexpr
    # Thus, only None and arguments marked `is_constexpr` should be treated as such.
    constants = {
        name: arg
        for name, arg in ordered_args.items()
        if _is_constexpr_or_none(name, arg)
    }

    if (mangle_type := getattr(triton.runtime.jit, "mangle_type", None)) is not None:

        def get_signature_value(idx: int, arg: Any) -> str:
            if kernel.params[idx].is_constexpr:
                return "constexpr"
            # pyrefly: ignore [not-callable]
            result = mangle_type(arg)
            # Workaround for Triton i1/u1 AOTI bug: PyTorch stores bool
            # tensors as uint8 (1 byte per element), but *i1/*u1 causes
            # the compiled kernel to generate bit-packed loads. Use *u8
            # so loads correctly read 1 byte per element.
            if result in ("*i1", "*u1"):
                result = "*u8"
            return result

    else:

        def get_signature_value(idx: int, arg: Any) -> str:
            return kernel._type_of(kernel.key_of(arg))

    if triton_version_uses_attrs_dict():
        # In newer versions of Triton, the signature includes constexpr args
        signature = {
            name: get_signature_value(i, arg)
            for i, (name, arg) in enumerate(ordered_args.items())
        }
    else:
        # In older versions of Triton, the signature does not include constexpr args
        constexprs = [p.num for p in kernel.params if p.is_constexpr]
        signature = {
            name: get_signature_value(i, arg)
            for i, (name, arg) in enumerate(ordered_args.items())
            if i not in constexprs
        }

    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)

    src = ASTSource(kernel, signature, constants, specialization)

    # Triton changes ASTSource.make_ir to take 3/4 arguments. Handle
    # backward compatibility here.
    make_ir_sig_params = len(inspect.signature(src.make_ir).parameters)
    get_codegen_implementation_sig_params = len(
        inspect.signature(backend.get_codegen_implementation).parameters
    )
    if make_ir_sig_params == 2:
        ttir_module = src.make_ir(options, context)
    elif make_ir_sig_params == 3:
        codegen_fns = backend.get_codegen_implementation()

        ttir_module = src.make_ir(options, codegen_fns, context)
    elif make_ir_sig_params == 4:
        codegen_args = [options] if get_codegen_implementation_sig_params == 1 else []

        codegen_fns = backend.get_codegen_implementation(*codegen_args)
        module_map = backend.get_module_map()

        ttir_module = src.make_ir(options, codegen_fns, module_map, context)
    else:
        codegen_args = [options] if get_codegen_implementation_sig_params == 1 else []

        codegen_fns = backend.get_codegen_implementation(*codegen_args)
        module_map = backend.get_module_map()

        ttir_module = src.make_ir(target, options, codegen_fns, module_map, context)
    if not ttir_module.verify():
        raise RuntimeError("Verification for TTIR module has failed")

    return ttir_module, ordered_arg_names


def ttir_to_functions(
    ttir_module: "TritonIRModule",
) -> dict[str, dict[Intermediate, list[Op]]]:
    """
    Walk the `ttir_module` bottom up to mine the `functions` from
    the structured MLIR entities representing the Triton kernel
    (mlir::Operation, mlir::Block, mlir::Region).
    """
    functions: dict[str, dict[Intermediate, list[Op]]] = {}

    # block id --> op result (Intermediate) --> one or more ops
    op_stack: dict[int, dict[Intermediate, list[Op]]] = defaultdict(
        lambda: defaultdict(list)
    )
    region_id_to_block_ids: dict[int, list[int]] = defaultdict(list)
    block_id_to_block_arg_ids: dict[int, list[int]] = {}
    replacements: dict[int, Intermediate | Param] = {}
    reindex_map: dict[int, int] = {}
    next_fake_intermediate = 0

    def reindex(idx: int) -> int:
        if idx not in reindex_map:
            reindex_map[idx] = len(reindex_map)
        return reindex_map[idx]

    def mlir_to_functions(op: "TritonIROperation") -> None:
        name: str = op.get_name()
        if name == "builtin.module":
            # this wraps all tt.func ops
            return

        operand_ids: list[int] = [
            reindex(op.get_operand(i).id()) for i in range(op.get_num_operands())
        ]
        result_ids: list[int] = [
            reindex(op.get_result(i).id()) for i in range(op.get_num_results())
        ]

        child_block_ids: list[int] = []
        for i in [op.get_region(i).id() for i in range(op.get_num_regions())]:
            # as the walk is bottom-up, the region_id_to_block_ids[i]
            # must be populated by the time we process the enclosing op
            child_block_ids.extend(region_id_to_block_ids[i])

        parent_block_id = -1
        parent_block = op.get_block()
        if parent_block is not None:
            parent_block_id = parent_block.id()
            if parent_block_id not in block_id_to_block_arg_ids:
                block_id_to_block_arg_ids[parent_block_id] = []
                for i in range(parent_block.get_num_arguments()):
                    block_id_to_block_arg_ids[parent_block_id].append(
                        reindex(parent_block.get_argument(i).id()),
                    )
                # the region info is collected via ops' parent blocks to be
                # used later when the region's enclosing op is traversed
                parent_region = parent_block.get_parent()
                if parent_region is not None:
                    region_id_to_block_ids[parent_region.id()].append(parent_block_id)

        nonlocal next_fake_intermediate

        if name == "tt.func":
            # for function ops: gather and inline
            # the ops from all child blocks
            fn_ops = defaultdict(list)
            for child_block_id in child_block_ids:
                for result, block_fn_ops in op_stack.pop(child_block_id).items():
                    for block_fn_op in block_fn_ops:
                        fn_ops[result].append(block_fn_op)

            # replace the corresponding Intermediates in the
            # child op args with the function args (Params)
            for i, idx in enumerate(block_id_to_block_arg_ids[child_block_ids[0]]):
                replacements[idx] = Param(i)

            for fn_op_list in fn_ops.values():
                for fn_op in fn_op_list:
                    for i in range(len(fn_op.args)):
                        arg = fn_op.args[i]
                        seen = set()  # to break cycles
                        # there can be transitive replacements, but likely
                        # no cycles (we keep the `seen` set just in case)
                        while (
                            isinstance(arg, Intermediate)
                            and arg.idx in replacements
                            and arg.idx not in seen
                        ):
                            seen.add(arg.idx)
                            arg = fn_op.args[i] = replacements[arg.idx]

            # next function capture starts
            # with empty replacements
            replacements.clear()

            fn_name = op.get_str_attr("sym_name")
            functions[fn_name] = fn_ops
        elif child_block_ids:
            if name in {"scf.if", "scf.for", "scf.while", "tt.reduce", "tt.scan"}:
                # for blocked ops: inline the enclosed ops into
                # the parent block + rewire the last op in each
                # child block to return the block result
                return_ops = []
                for block_id in child_block_ids:
                    if name == "scf.for":
                        # example:
                        # %result = scf.for %iv = %lb to %ub step %step iter_args(%arg = %init) -> (i32) ...
                        # block args: 2 (%iv, %arg)
                        # op operands: 4 (%lb, %ub, %step, %init)
                        # `%arg` is mapping to `%init`
                        for i, idx in enumerate(block_id_to_block_arg_ids[block_id]):
                            if i == 0:
                                next_fake_intermediate -= 1
                                replacements[idx] = Intermediate(next_fake_intermediate)
                            else:
                                replacements[idx] = Intermediate(operand_ids[i + 2])
                    elif name == "scf.while":
                        # example:
                        # %3:3 = scf.while (%arg2 = %1, %arg3 = %2, %arg4 = %c0_i32_8) ...
                        # block args: 3 (%arg2, %arg3, %arg4)
                        # op operands: 3 (%1, %2, %c0_i32_8)
                        # `%arg2` is mapping to `%1`, `%arg3` is mapping to `%2`, ...
                        for i, idx in enumerate(block_id_to_block_arg_ids[block_id]):
                            replacements[idx] = Intermediate(operand_ids[i])
                    elif name == "scf.if":
                        # the scf block args are ignored by the pass. but, as they
                        # may be used as operands of the ops inside the block
                        # (and nested blocks inlined in the current block by now),
                        # they are replaced by new fake Intermediates to avoid "this
                        # operand is not returned by any other op in the fn" error
                        # in the downstream analysis
                        for idx in block_id_to_block_arg_ids[block_id]:
                            next_fake_intermediate -= 1
                            replacements[idx] = Intermediate(next_fake_intermediate)
                    else:
                        if name not in ("tt.reduce", "tt.scan"):
                            raise AssertionError(
                                f"Expected op name to be 'tt.reduce' or 'tt.scan', got {name}"
                            )
                        # wire the block arguments to the op arguments
                        num_operands = len(operand_ids)
                        block_arg_ids = block_id_to_block_arg_ids[block_id]
                        if len(block_arg_ids) != 2 * num_operands:
                            raise AssertionError(
                                f"{name} is expected to have twice as "
                                "many block arguments as op arguments: "
                                f"{operand_ids=}, {block_arg_ids=}."
                            )
                        for i, idx in enumerate(block_arg_ids):
                            # for a tt.reduce/tt.scan op with N arguments, the block
                            # arguments comprise N reduced values followed by
                            # N current values corresponding to the N op args
                            replacements[idx] = Intermediate(
                                operand_ids[i % num_operands]
                            )

                    if block_id in op_stack:
                        block_ops = op_stack.pop(block_id)
                        if not block_ops:
                            continue
                        last_ret, last_ops = block_ops.popitem()
                        if all(
                            op.name
                            in ("scf.yield", "tt.reduce.return", "tt.scan.return")
                            for op in last_ops
                        ):
                            # if last_ops are all return ops, treat them separately
                            return_ops.extend(last_ops)
                        else:
                            # otherwise, return last_ops to the block
                            block_ops[last_ret] = last_ops
                        for op_result, child_ops in block_ops.items():
                            op_stack[parent_block_id][op_result].extend(child_ops)

                scf_results = [Intermediate(idx) for idx in result_ids]

                if return_ops and all(
                    (op.name == "scf.yield" and len(result_ids) == len(op.args))
                    for op in return_ops
                ):
                    # [Note: scf.yield fix-up]
                    #
                    # TL;DR: if our scf.yield takes N args, then we'll create N scf.yield ops to handle each of the
                    # args.
                    #
                    #      **Context**:
                    # During mutation analysis, the analysis pass will identify mutating ops (e.g. tt.store)
                    # and then DFS upwards towards the parameters of the function. Specifically, the analysis pass
                    # looks at the mutated arg in tt.store; then looks for its source ops; and then recurses on the
                    # arguments to each of the source ops.
                    #
                    # In the case of scf.if/scf.for, we may have multiple return ops, each passed as an arg
                    # to scf.yield:
                    #
                    # %18:2 = scf.if %... -> (!tt.ptr<f32>, !tt.ptr<f32>) {
                    #   ...
                    #   scf.yield %1, %2
                    # } else {
                    #   scf.yield %3, %4
                    # }
                    #
                    # And for each of the returns of the scf.if, we'd naively assign the source op of each of the
                    # return values to be the scf.yields. But the scf.yields take _all_ the returns as arguments.
                    # Therefore, if _any_ of the return values of the scf.if are mutated, then the analysis pass
                    # would mark _all_ of the yield args as mutated.
                    #
                    #      **Solution**:
                    # For the purposes of this analysis pass, we create N yield ops - one for each
                    # return-val/yield-arg. In the example above, we'll have two scf.yield's for each branch of the
                    # scf.if.

                    for return_op in return_ops:
                        for i, (scf_result, yield_arg) in enumerate(
                            zip(scf_results, return_op.args)
                        ):
                            sub_yield_op = Op(
                                return_op.name,
                                return_op.fn_call_name,
                                [yield_arg],
                                return_op.ret,
                                sub_idx=i,
                            )
                            op_stack[parent_block_id][scf_result].append(sub_yield_op)

                else:
                    for scf_result in scf_results:
                        for return_op in return_ops:
                            op_stack[parent_block_id][scf_result].append(return_op)
            else:
                raise RuntimeError(
                    f"Unknown blocked function: {name}. Can't capture the TTIR."
                )
        else:
            callee = None
            if name == "tt.call":
                callee = op.get_flat_symbol_ref_attr("callee")
            args: list[Param | Intermediate] = [
                Intermediate(operand) for operand in operand_ids
            ]
            block_ops = op_stack[parent_block_id]

            is_pure = False
            # Handle the case for tt.elementwise_inline_asm to set `is_pure` for mutation analysis
            if name == "tt.elementwise_inline_asm":
                is_pure = op.get_bool_attr("pure")

            if result_ids:
                for result_id in result_ids:
                    res = Intermediate(result_id)
                    block_ops[res].append(Op(name, callee, args, res, is_pure=is_pure))
            else:
                next_fake_intermediate -= 1
                fake_res = Intermediate(next_fake_intermediate)
                block_ops[fake_res].append(
                    Op(name, callee, args, fake_res, is_pure=is_pure)
                )

    ttir_module.walk(mlir_to_functions)

    return functions


class MemoizeWithCycleCheck:
    fn: Callable[..., Any]
    cache: dict[tuple[Any], Any]

    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn
        self.reset()

    def __call__(
        self,
        functions: dict[str, dict[Intermediate, list[Op]]],
        fn_name: str,
        *args: Any,
    ) -> Any:
        key: tuple[Any, ...] = (fn_name, *args)
        if key not in self.cache:
            self.cache[key] = None
            self.cache[key] = self.fn(functions, fn_name, *args)
        if self.cache[key] is None:
            raise RuntimeError("Recursion is not supported")
        return self.cache[key]

    def reset(self) -> None:
        self.cache = {}


@MemoizeWithCycleCheck
def get_tma_stores(
    functions: dict[str, dict[Intermediate, list[Op]]], fn_name: str
) -> set[Intermediate | Param]:
    """
    Identifies all intermediates and parameters that are written to by a
    `tt.experimental_descriptor_store`. It tracks only the specific values
    written to via experimental_descriptor_store and the input values to
    `tt.reinterpret_tensor_descriptor` used to construct the direct inputs
    to tt.experimental_descriptor_store - not any recursive values
    used to construct those values.

    For example: for
      tt.reinterpret_tensor_descriptor(Intermediate(idx=0), ...)
      Intermediate(idx=1) = tt.experimental_descriptor_store(Intermediate(idx=0), ...)
    this function will return [Intermediate(idx=0), Intermediate(idx=1)],

    However
      Intermediate(idx=4) = arith.addptr(Intermediate(idx=2), Intermediate(idx=3))
      Intermediate(idx=5) = tt.experimental_descriptor_store(Intermediate(idx=4), ...)
      tt.experimental_descriptor_store(Intermediate(idx=5), ...)
    this function will mark only idx=4 and idx=5 (but not idx=2 or idx=3)

    If an intermediate/parameter is passed into a function and is written to
    via experimental_descriptor_store within that function, the argument to the
    function will also be marked.
    """

    result: set[Intermediate | Param] = set()

    ops = functions[fn_name]
    for op_list in ops.values():
        for op in op_list:
            if op.name == "tt.call":
                if op.fn_call_name not in functions:
                    raise AssertionError(
                        f"Function {op.fn_call_name} not found in functions for TMA stores"
                    )
                # pyrefly: ignore [bad-argument-type]
                tma_stores = get_tma_stores(functions, op.fn_call_name)
                for i, inp in enumerate(op.args):
                    if Param(idx=i) in tma_stores:
                        result.add(inp)
            else:
                op_info = _KERNEL_ACCESS_OPS.get(op.name)
                if (
                    op_info is not None
                    and op_info.is_tma_store
                    and op_info.write_indexes is not None
                ):
                    result.update(op.args[idx] for idx in op_info.write_indexes(op))

    for val in list(result):
        if val in ops:
            if not isinstance(val, Intermediate):
                continue
            for op in ops[val]:
                if op.name == "tt.reinterpret_tensor_descriptor":
                    if len(op.args) < 1:
                        raise AssertionError(
                            "tt.reinterpret_tensor_descriptor expected at least 1 arg, "
                            f"got {len(op.args)}"
                        )
                    result.add(op.args[0])

    return result


@dataclasses.dataclass
class TensorAccesses:
    read_writes: "ReadWrites"
    can_fuse_epilogue: bool


class IgnoreUnknownOp(Protocol):
    """Returns True if the op should be skipped; otherwise the caller raises."""

    def __call__(self, op: Op) -> bool: ...


class ReadWriteIndexes(Protocol):
    """Return the argument indexes read / written."""

    def __call__(self, op: Op) -> Sequence[int]: ...


def first_arg(op: Op) -> list[int]:
    """Return the first argument index after checking that it exists."""
    if len(op.args) < 1:
        raise AssertionError(f"{op.name} expected at least 1 arg, got {len(op.args)}")
    return [0]


@dataclasses.dataclass(frozen=True)
class _KernelAccessOpInfo:
    read_indexes: ReadWriteIndexes | None = None
    write_indexes: ReadWriteIndexes | None = None
    is_tma_store: bool = False
    ignore_if: IgnoreUnknownOp | None = None


# List from Triton Github include/triton/Dialect/Triton/IR/TritonOps.td.
# What if Triton exposed this?
_KERNEL_ACCESS_OPS: dict[str, _KernelAccessOpInfo] = {
    "tt.experimental_descriptor_store": _KernelAccessOpInfo(
        write_indexes=first_arg, is_tma_store=True
    ),
    "tt.descriptor_store": _KernelAccessOpInfo(
        write_indexes=first_arg, is_tma_store=True
    ),
    "tt.store": _KernelAccessOpInfo(write_indexes=first_arg),
    "tt.atomic_cas": _KernelAccessOpInfo(write_indexes=first_arg),
    "tt.atomic_rmw": _KernelAccessOpInfo(
        read_indexes=first_arg, write_indexes=first_arg
    ),
    "tt.experimental_tensormap_create": _KernelAccessOpInfo(write_indexes=first_arg),
    "tt.load": _KernelAccessOpInfo(read_indexes=first_arg),
    "tt.load_tensor_descriptor": _KernelAccessOpInfo(read_indexes=first_arg),
    "tt.descriptor_load": _KernelAccessOpInfo(read_indexes=first_arg),
    "tt.elementwise_inline_asm": _KernelAccessOpInfo(ignore_if=lambda op: op.is_pure),
}


def _read_write_indexes(indexes: Sequence[int] | ReadWriteIndexes) -> ReadWriteIndexes:
    if callable(indexes):
        return indexes
    index_tuple = tuple(indexes)
    return lambda op: index_tuple


def register_kernel_access_op(
    name: str,
    *,
    read_indexes: Sequence[int] | ReadWriteIndexes | None = None,
    write_indexes: Sequence[int] | ReadWriteIndexes | None = None,
    is_tma_store: bool = False,
    ignore_if: IgnoreUnknownOp | None = None,
) -> None:
    """Register a Triton op for kernel read/write analysis."""
    if is_tma_store and write_indexes is None:
        raise AssertionError(
            f"{name} is marked as a TMA store but has no write indexes"
        )
    has_indexes = read_indexes is not None or write_indexes is not None
    if ignore_if is not None and has_indexes:
        raise AssertionError(
            f"{name}: ignore_if cannot be combined with read/write indexes"
        )
    _KERNEL_ACCESS_OPS[name] = _KernelAccessOpInfo(
        read_indexes=(
            None if read_indexes is None else _read_write_indexes(read_indexes)
        ),
        write_indexes=(
            None if write_indexes is None else _read_write_indexes(write_indexes)
        ),
        is_tma_store=is_tma_store,
        ignore_if=ignore_if,
    )


def unregister_kernel_access_op(name: str) -> None:
    """
    Unregister a Triton op from kernel read/write analysis.

    This is mainly useful for tests and temporary backend registrations. It
    does not reset memoized analysis, so direct analyze_kernel_access callers
    must reset their own caches.
    """
    _KERNEL_ACCESS_OPS.pop(name, None)


@MemoizeWithCycleCheck
def analyze_kernel_access(
    functions: dict[str, dict[Intermediate, list[Op]]],
    fn_name: str,
    num_args: int,
    tensor_names: tuple[str, ...],
    tensor_arg_indices: frozenset[int] | None,
) -> TensorAccesses:
    """
    Analyzes the graph to detect which arguments are written to and which are read.

    For writes: traverses from write sinks (tt.store, tt.atomic_cas, etc.) backwards
    to identify input pointers that are written to.

    For reads: traverses from read operations (tt.load) backwards to identify
    input pointers that are read from.

    Returns ReadWrites with StarDep objects for each accessed tensor.
    """
    from torch._inductor.dependencies import Dep, ReadWrites, StarDep

    write_stack: list[Param | Intermediate] = []
    read_stack: list[Param | Intermediate] = []

    ops = functions[fn_name]
    tma_stores = get_tma_stores(functions, fn_name)

    for op_list in ops.values():
        for op in op_list:
            # If we encounter an operation with effects that cannot be reliably analyzed
            # (e.g. `tt.elementwise_inline_asm`), we assume it does not mutate any input parameters.
            op_info = _KERNEL_ACCESS_OPS.get(op.name)
            if op_info is not None and op_info.ignore_if is not None:
                if op_info.ignore_if(op):
                    continue
                raise RuntimeError(
                    f"ttir analysis hit an op we do not know how to analyze: {op.name}"
                )

            if op.name == "tt.experimental_tensormap_create":
                # Note: this is how we implement experimental_descriptor_store mutation analysis.
                # for on-device TMA.
                # experimental_tensormap_store(a, b, ...) stores b to the location specified
                # by descriptor in the memory of a.
                # To track this, we first find all the intermediates/params to which we store via
                # experimental_tensormap_store (get_tma_stores, called above). Then, during this
                # analysis we wait to find the corresponding experimental_tensormap_create (if it
                # exists), at which point we will mark the global_ptr as mutated (as done below).
                if len(op.args) < 2:
                    raise AssertionError(
                        f"tt.experimental_tensormap_create expected at least 2 args, "
                        f"got {len(op.args)}"
                    )
                if op.args[0] in tma_stores:
                    write_stack.append(op.args[1])

            if op.name == "tt.call":
                if op.fn_call_name not in functions:
                    raise AssertionError(
                        f"Function {op.fn_call_name} not found in functions dict"
                    )
                # Create placeholder names for nested function arguments
                nested_names = tuple(f"_arg{i}" for i in range(len(op.args)))

                # Do not pass tensor_arg_indices, most outer call of
                # analyze_kernel_access will filter Param nodes.
                accesses = analyze_kernel_access(
                    functions,
                    # pyrefly: ignore [bad-argument-type]
                    op.fn_call_name,
                    len(op.args),
                    nested_names,
                    None,
                )
                # Map back from StarDep names to args
                written_set = {dep.name for dep in accesses.read_writes.writes}
                read_set = {dep.name for dep in accesses.read_writes.reads}
                for arg, name in zip(op.args, nested_names):
                    if name in written_set:
                        write_stack.append(arg)
                    if name in read_set:
                        read_stack.append(arg)
            else:
                if op_info is not None and op_info.write_indexes is not None:
                    write_stack.extend(
                        op.args[idx] for idx in op_info.write_indexes(op)
                    )
                if op_info is not None and op_info.read_indexes is not None:
                    read_stack.extend(op.args[idx] for idx in op_info.read_indexes(op))

    # For these ops, only the first argument (base pointer) refers to actual
    # memory. The remaining arguments are shape/stride/offset metadata and
    # should not be traced during mutation analysis.
    POINTER_ONLY_OPS = {
        "tt.make_tensor_ptr",
        "tt.advance",
        "tt.make_tensor_descriptor",
    }

    def _find_arg_access_count(
        initial_stack: list[Param | Intermediate],
        skip_loads: bool,
    ) -> dict[int, int]:
        """DFS traversal to find argument indices that are accessed (and how many times they are accessed)."""
        access_count = dict()
        stack = initial_stack[:]

        while stack:
            arg = stack.pop()

            if isinstance(arg, Param):
                if arg.idx >= num_args:
                    continue
                if tensor_arg_indices is not None and arg.idx not in tensor_arg_indices:
                    continue
                if arg.idx not in access_count:
                    access_count[arg.idx] = 1
                else:
                    access_count[arg.idx] += 1
            elif isinstance(arg, Intermediate) and not arg.fake():
                for op in ops[arg]:
                    if skip_loads and op.name == "tt.load":
                        continue
                    if op.name in POINTER_ONLY_OPS:
                        stack.append(op.args[0])
                    else:
                        stack.extend(op.args)

        return access_count

    write_count = _find_arg_access_count(write_stack, skip_loads=True)
    read_count = _find_arg_access_count(read_stack, skip_loads=False)

    writes: OrderedSet[Dep] = OrderedSet(
        StarDep(tensor_names[i]) for i in sorted(write_count.keys())
    )
    reads: OrderedSet[Dep] = OrderedSet(
        StarDep(tensor_names[i]) for i in sorted(read_count.keys())
    )

    read_writes = ReadWrites(
        reads=reads,
        writes=writes,
        index_exprs=OrderedSet(),
    )

    def _decide_can_fuse_epilogue():
        # only do epilogue fusion if the kernel has a single output tensor
        if len(write_count) != 1:
            return False

        written_arg_index = next(iter(write_count))
        # only do epilogue fusion if the written tensor is written exactly once
        if write_count[written_arg_index] != 1:
            return False

        written_arg_name = next(iter(writes)).name
        #  cannot fuse if the kernel also reads from the output buffer
        if any(read_dep.name == written_arg_name for read_dep in reads):
            return False

        return True

    can_fuse_epilogue = _decide_can_fuse_epilogue()

    return TensorAccesses(read_writes=read_writes, can_fuse_epilogue=can_fuse_epilogue)


def identify_accessed_tensors(
    kernel: "TritonKernelType",
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> TensorAccesses:
    """
    Given a triton kernel and the arguments for this kernel, this function
    1) Retrieves the TTIR converted version of the kernel from Triton's API.
    2) Parses the TTIR and creates a control flow graph
    3) Analyzes the graph to detect which input tensors are read and/or written
    """

    from torch._inductor.dependencies import Dep, ReadWrites, StarDep
    from torch._inductor.ir import TensorBox

    ttir_module = None
    functions = None
    try:
        ttir_module, ordered_arg_names = generate_ttir(
            kernel, kwargs, tma_descriptor_metadata
        )

        # extract functions from TTIR using MLIR bindings exposed by Triton code
        functions = ttir_to_functions(ttir_module)

        if functions is None:
            raise AssertionError("ttir_to_functions returned None")
        kernel_name = next(iter(functions.keys()))
        # Triton codegen modifies the name
        # pyrefly: ignore [missing-attribute]
        kernel_fn_name = kernel.fn.__name__
        if kernel_fn_name not in kernel_name:
            raise AssertionError(
                f"Kernel name {kernel_fn_name} not found in TTIR kernel name {kernel_name}"
            )
        # Reset the cache between top level invocations
        # The cache for analyze kernel access is mainly used for cycle
        # detection, so each top level invocation needs a clean cache
        analyze_kernel_access.reset()
        get_tma_stores.reset()

        # Build frozenset of indices corresponding to tensor args only.
        # Used to filter out scalars which are transitively captured as mutated
        # during traversal.
        tensor_arg_indices = frozenset(
            i
            for i, name in enumerate(ordered_arg_names)
            if isinstance(kwargs.get(name), (Tensor, TensorBox))
        )

        return analyze_kernel_access(
            functions,
            kernel_name,
            len(ordered_arg_names),
            tuple(ordered_arg_names),
            tensor_arg_indices,
        )
    except Exception:
        log.warning(
            "Encountered an exception in identify_accessed_tensors, assuming every input is mutated",
            exc_info=True,
        )
        if ttir_module is not None:
            log.debug("TTIR:\n%s", ttir_module)
        if functions is not None:
            log.debug("functions:")
            for name, fn in functions.items():
                log.debug("===\t%s\t===", name)
                for ret, ops in fn.items():
                    log.debug("%s\t=>\t%s", ret, ops)

        all_tensor_names = [
            key
            for key, value in kwargs.items()
            if isinstance(value, (Tensor, TensorBox))
        ]
        all_deps = OrderedSet(StarDep(name) for name in all_tensor_names)
        all_deps = typing.cast(OrderedSet[Dep], all_deps)
        return TensorAccesses(
            ReadWrites(
                reads=all_deps,
                writes=all_deps,
                index_exprs=OrderedSet(),
            ),
            can_fuse_epilogue=False,
        )


@dataclasses.dataclass
class TritonStore:
    store_node: ast.Call
    store_pointer_node: ast.Expr
    store_value_node: ast.Expr


@dataclasses.dataclass
class TritonStores:
    stores: list[TritonStore]


@functools.cache
def identify_triton_stores(source_code: str) -> TritonStores:
    """
    Parse Python source code of triton kernel and find all tl.store calls.
    Returns a TritonStores object containing information about pointer, value, and mask.

    tl.store signature: store(pointer, value, mask=None, boundary_check=(), ...)
    """
    return identify_triton_stores_from_ast(ast.parse(source_code))


def identify_triton_stores_from_ast(tree: ast.Module) -> TritonStores:
    stores = []

    def _extract_arg(node, arg_name, positional_index):
        """
        Extract an argument from a Call node, checking both positional and keyword args.
        Returns the AST node for the argument, or None if not found.
        """
        # Check positional args first
        if len(node.args) > positional_index:
            return node.args[positional_index]

        # Check keyword args
        for keyword in node.keywords:
            if keyword.arg == arg_name:
                return keyword.value

        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Check if this is a tl.store call
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "tl"
                and node.func.attr == "store"
            ):
                # Extract required arguments
                pointer_node = _extract_arg(node, "pointer", 0)
                value_node = _extract_arg(node, "value", 1)

                if pointer_node is None or value_node is None:
                    continue

                stores.append(TritonStore(node, pointer_node, value_node))

    return TritonStores(stores=stores)


###############################################################################
# Triton Kernel Wrappers


class _TritonKernelWrapper(HigherOrderOperator):
    def _get_overloaded_args(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Tensor, ...]:
        overloaded_args = list(super()._get_overloaded_args(args, kwargs))
        triton_kwargs = kwargs.get("kwargs")
        if isinstance(triton_kwargs, dict):
            overloaded_args.extend(
                arg
                for arg in triton_kwargs.values()
                if isinstance(arg, Tensor) and _dispatch_keys(arg).has("Python")
            )
        return tuple(overloaded_args)


# Used for wrapping a Triton Kernel
class TritonKernelWrapperMutation(_TritonKernelWrapper):
    def __init__(self) -> None:
        super().__init__("triton_kernel_wrapper_mutation", cacheable=True)

    def __call__(
        self,
        kernel_idx: int,
        constant_args_idx: int,
        grid: list["TritonGridType"],
        tma_descriptor_metadata: TMADescriptorMetadata,
        kwargs: dict[str, Any],
        launch_kwargs: tuple[str, ...] | None = None,
    ) -> Any:
        hop_kwargs: dict[str, Any] = {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": grid,
            "tma_descriptor_metadata": tma_descriptor_metadata,
            "kwargs": kwargs,
        }
        if launch_kwargs:
            hop_kwargs["launch_kwargs"] = launch_kwargs

        # pyrefly: ignore [missing-attribute]
        return super().__call__(**hop_kwargs)


triton_kernel_wrapper_mutation = TritonKernelWrapperMutation()


# Used for wrapping a Triton Kernel in a functional manner
class TritonKernelWrapperFunctional(_TritonKernelWrapper):
    def __init__(self) -> None:
        super().__init__("triton_kernel_wrapper_functional", cacheable=True)

    def __call__(
        self,
        kernel_idx: int,
        constant_args_idx: int,
        grid: list["TritonGridType"],
        tma_descriptor_metadata: TMADescriptorMetadata,
        kwargs: dict[str, Any],
        tensors_to_clone: list[str],
        launch_kwargs: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        hop_kwargs: dict[str, Any] = {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": grid,
            "tma_descriptor_metadata": tma_descriptor_metadata,
            "kwargs": kwargs,
            "tensors_to_clone": tensors_to_clone,
        }
        if launch_kwargs:
            hop_kwargs["launch_kwargs"] = launch_kwargs

        # pyrefly: ignore [missing-attribute]
        return super().__call__(**hop_kwargs)


triton_kernel_wrapper_functional = TritonKernelWrapperFunctional()


def get_kernel(kernel_idx: int) -> "TritonKernelType":
    return kernel_side_table.get_kernel(kernel_idx)


@triton_kernel_wrapper_mutation.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_mutation_dense(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    launch_kwargs: tuple[str, ...] | None = None,
) -> None:
    from torch._inductor.codegen.wrapper import user_defined_kernel_grid_fn_code

    kernel = kernel_side_table.get_kernel(kernel_idx)
    constant_args = kernel_side_table.get_constant_args(constant_args_idx)

    if len(grid) == 1:
        grid_fn = grid[0]
    else:
        fn_name, code = user_defined_kernel_grid_fn_code(
            # pyrefly: ignore [missing-attribute]
            kernel.fn.__name__,
            # pyrefly: ignore [missing-attribute]
            kernel.configs,
            grid,
        )
        namespace: dict[str, Any] = {}
        exec(code, namespace)
        grid_fn = namespace[fn_name]

    if tma_descriptor_metadata:
        # as we need to launch the kernel here, we "unwrap" the
        # tma_descriptor_metadata, create the TMA descriptors
        # from it, and replace the tensors in the kwargs by the
        # corresponding TMA descriptors before launching
        kwargs = kwargs.copy()
        for k, v in tma_descriptor_metadata.items():
            tensor = kwargs[k]
            if (exp_meta := maybe_unpack_tma_experimental_metadata(v)) is not None:
                from triton.tools.experimental_descriptor import (
                    create_1d_tma_descriptor,
                    create_2d_tma_descriptor,
                )

                dims, block_dims, element_size = exp_meta
                create_tma_descriptor = (
                    create_1d_tma_descriptor
                    if len(dims) == 1
                    else create_2d_tma_descriptor
                )
                kwargs[k] = create_tma_descriptor(
                    tensor.data_ptr(),
                    *dims,
                    *block_dims,
                    element_size,
                )
            else:
                stable_meta = maybe_unpack_tma_stable_metadata(v)
                if stable_meta is None:
                    raise AssertionError(
                        f"Failed to unpack stable TMA metadata for key {k}"
                    )
                from triton.tools.tensor_descriptor import TensorDescriptor

                block_shape = stable_meta[0]

                kwargs[k] = TensorDescriptor.from_tensor(tensor, block_shape)

    # move as many positional arguments from dicts to args as we
    # can to circumvent the bug with the kwargs and pre_/post_hook:
    # https://github.com/triton-lang/triton/issues/5082
    # TODO: remove this when the Triton issue above is fixed
    args = []
    # copy kwargs and constant_args here to
    # avoid mutating the original inputs
    kwargs = kwargs.copy()
    constant_args = constant_args.copy()
    launch_kwargs = () if launch_kwargs is None else launch_kwargs
    # pyrefly: ignore [missing-attribute]
    for name in kernel.arg_names:
        if name in launch_kwargs:
            # Preserve the original kwarg form for launch kwargs. Triton's
            # binder uses the kwarg to bind the kernel parameter, and
            # _pack_args still exposes launch kwargs to backend.parse_options()
            # so the same name may also be parsed as a compiler option.
            #
            # Stop positional packing entirely at this point. A later kernel
            # parameter cannot be safely packed positionally after an earlier
            # parameter is intentionally left in kwargs/constant_args; doing so
            # would shift it into the wrong argument slot. Triton accepts the
            # remaining parameters by keyword, so the only trade-off is that
            # the triton#5082 workaround no longer applies after this first
            # backend-option/kernel-parameter overlap.
            break
        if name in kwargs:
            args.append(kwargs.pop(name))
        elif name in constant_args:
            args.append(constant_args.pop(name))
        else:
            break

    # pyrefly: ignore [bad-index, index-error]
    kernel[grid_fn](*args, **kwargs, **constant_args)


@register_fake(triton_kernel_wrapper_mutation, skip_cache=True)
def triton_kernel_wrapper_mutation_fake_tensor_mode(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    launch_kwargs: tuple[str, ...] | None = None,
) -> None:
    return None


@triton_kernel_wrapper_mutation.py_impl(DispatchKey.Meta)
def _(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    launch_kwargs: tuple[str, ...] | None = None,
) -> None:
    return None


def trace_triton_kernel_wrapper(
    proxy_mode: ProxyTorchDispatchMode,
    func_overload: Callable[..., Any],
    node_args: dict[str, Any],
) -> dict[str, Any] | None:
    with disable_proxy_modes_tracing():
        out = func_overload(**node_args)

    proxy_args = pytree.tree_map(
        proxy_mode.tracer.unwrap_proxy,  # type: ignore[union-attr]
        node_args,
    )
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        func_overload,
        (),
        proxy_args,
        name=func_overload.__name__ + "_proxy",
    )

    ret = track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)
    return ret


@triton_kernel_wrapper_mutation.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    launch_kwargs: tuple[str, ...] | None = None,
) -> None:
    node_args: dict[str, Any] = {
        "kernel_idx": kernel_idx,
        "constant_args_idx": constant_args_idx,
        "grid": grid,
        "tma_descriptor_metadata": tma_descriptor_metadata,
        "kwargs": kwargs,
    }
    if launch_kwargs:
        node_args["launch_kwargs"] = launch_kwargs

    trace_triton_kernel_wrapper(
        mode,
        triton_kernel_wrapper_mutation,
        node_args,
    )

    return None


def get_mutated_tensors(
    kernel_idx: int,
    constant_args_idx: int,
    kwargs: dict[str, Any],
    tma_descriptor_metadata: TMADescriptorMetadata,
) -> list[str]:
    kernel = kernel_side_table.get_kernel(kernel_idx)
    constant_args = kernel_side_table.get_constant_args(constant_args_idx)
    tensor_accesses = identify_accessed_tensors(
        kernel, {**kwargs, **constant_args}, tma_descriptor_metadata
    )
    # Filter to only tensor kwargs: with Triton 3.7+, ordered_arg_names
    # includes scalars, so writes may reference non-tensor args like SymInts.
    return [
        dep.name
        for dep in tensor_accesses.read_writes.writes
        if isinstance(kwargs.get(dep.name), Tensor)
    ]


@triton_kernel_wrapper_mutation.py_functionalize_impl
def triton_kernel_wrapper_mutation_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    launch_kwargs: tuple[str, ...] | None = None,
) -> None:
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    # TODO(oulgen): Preexisting bug, if two kernel inputs are views of each
    # other, and one gets mutated in kernel, and later another gets mutated,
    # they are no longer equal. Fix this by graph breaking on this condition
    # earlier in dynamo.
    tensors_to_clone = get_mutated_tensors(
        kernel_idx, constant_args_idx, unwrapped_kwargs, tma_descriptor_metadata
    )
    with ctx.redispatch_to_next():
        functional_kwargs: dict[str, Any] = {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": grid,
            "tma_descriptor_metadata": tma_descriptor_metadata,
            "kwargs": unwrapped_kwargs,
            "tensors_to_clone": tensors_to_clone,
        }
        if launch_kwargs:
            functional_kwargs["launch_kwargs"] = launch_kwargs
        unwrapped_outputs = triton_kernel_wrapper_functional(**functional_kwargs)

    if not set(unwrapped_outputs.keys()).issubset(set(kwargs.keys())):
        raise AssertionError(
            f"Output keys {set(unwrapped_outputs.keys())} not subset of input keys {set(kwargs.keys())}"
        )
    for key, output_arg in unwrapped_outputs.items():
        if not isinstance(output_arg, Tensor):
            continue
        input_arg = kwargs[key]
        if not isinstance(input_arg, Tensor):
            raise AssertionError(
                f"Expected input_arg for key {key} to be a Tensor, got {type(input_arg)}"
            )

        ctx.replace(input_arg, output_arg)
        # indicate that above replace is hidden from autograd
        ctx.mark_mutation_hidden_from_autograd(input_arg)
        ctx.commit_update(input_arg)
        ctx.sync(input_arg)
    return None


@triton_kernel_wrapper_functional.py_impl(DispatchKey.CompositeExplicitAutograd)
def triton_kernel_wrapper_functional_dense(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
    launch_kwargs: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    kwargs = {
        key: (clone_preserve_strides(val) if key in tensors_to_clone else val)
        for key, val in kwargs.items()
    }
    mutation_kwargs: dict[str, Any] = {
        "kernel_idx": kernel_idx,
        "constant_args_idx": constant_args_idx,
        "grid": grid,
        "tma_descriptor_metadata": tma_descriptor_metadata,
        "kwargs": kwargs,
    }
    if launch_kwargs:
        mutation_kwargs["launch_kwargs"] = launch_kwargs
    triton_kernel_wrapper_mutation(**mutation_kwargs)
    return {key: val for key, val in kwargs.items() if key in tensors_to_clone}


@register_fake(triton_kernel_wrapper_functional, skip_cache=True)
def triton_kernel_wrapper_functional_fake_tensor_mode(
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
    launch_kwargs: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    # TODO(oulgen): For performance reasons, we want to ensure that these
    # `clone_preserve_strides` calls are never executed at runtime
    # (inductor should always optimize them away).
    # Requires https://github.com/pytorch/pytorch/issues/109240
    return {
        key: clone_preserve_strides(val)
        for key, val in kwargs.items()
        if key in tensors_to_clone
    }


@triton_kernel_wrapper_functional.py_impl(ProxyTorchDispatchMode)
def triton_kernel_wrapper_functional_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
    launch_kwargs: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    node_args: dict[str, Any] = {
        "kernel_idx": kernel_idx,
        "constant_args_idx": constant_args_idx,
        "grid": grid,
        "tma_descriptor_metadata": tma_descriptor_metadata,
        "kwargs": kwargs,
        "tensors_to_clone": tensors_to_clone,
    }
    if launch_kwargs:
        node_args["launch_kwargs"] = launch_kwargs

    ret = trace_triton_kernel_wrapper(
        mode,
        triton_kernel_wrapper_functional,
        node_args,
    )
    if ret is None:
        raise AssertionError("trace_triton_kernel_wrapper returned None")
    return ret


@triton_kernel_wrapper_functional.py_functionalize_impl
def triton_kernel_wrapper_functional_functionalize(
    ctx: "BaseFunctionalizeAPI",
    kernel_idx: int,
    constant_args_idx: int,
    grid: list["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: dict[str, Any],
    tensors_to_clone: list[str],
    launch_kwargs: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    unwrapped_kwargs = ctx.unwrap_tensors(kwargs)  # type: ignore[arg-type]
    with ctx.redispatch_to_next():
        functional_kwargs: dict[str, Any] = {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": grid,
            "tma_descriptor_metadata": tma_descriptor_metadata,
            "kwargs": unwrapped_kwargs,
            "tensors_to_clone": tensors_to_clone,
        }
        if launch_kwargs:
            functional_kwargs["launch_kwargs"] = launch_kwargs
        outputs = triton_kernel_wrapper_functional(**functional_kwargs)
        return ctx.wrap_tensors(outputs)  # type: ignore[return-value,arg-type]


triton_kernel_wrapper_mutation.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.ADInplaceOrView)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.BackendSelect)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_mutation.fallthrough(DispatchKey.AutogradCPU)

triton_kernel_wrapper_functional.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.ADInplaceOrView)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.BackendSelect)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCUDA)
triton_kernel_wrapper_functional.fallthrough(DispatchKey.AutogradCPU)

# Adds SAC support for triton ops
redirect_to_mode(triton_kernel_wrapper_mutation, _CachingTorchDispatchMode)
redirect_to_mode(triton_kernel_wrapper_mutation, _CachedTorchDispatchMode)

###############################################################################
# The "TritonHOPifier": a class that transforms a call to a triton kernel into
# a call to the triton_kernel_wrapper_mutation HOP.


class TritonHOPifier:
    """Orchestrator for converting a user-defined triton kernel into a call
    to the triton_kernel_wrapper_mutation HOP.

    It has two main use cases.

    1. When Dynamo sees a triton kernel, it wraps it into a TritonKernelVariable
    and uses the TritonHOPifier to convert calls to the TritonKernelVariable
    into a call to the HOP.

    2. In order to capture a user-defined triton kernel while performing
    tracing (via make_fx or non-strict export), a user must annotate their
    triton kernel with the `wrap_triton` decorator. The decorator uses
    TritonHOPifier to convert calls to the triton kernel into a call
    to the HOP (which can then be traced).

    Because Dynamo has its own calling conventions for e.g. invoking a user-defined function
    TritonHOPifier is an abstract class that can be overridden by its subclasses.
    """

    def raise_unsupported(self, msg: str) -> Never:
        raise NotImplementedError("abstract method")

    def is_callable(self, maybe_callable: Any) -> bool:
        raise NotImplementedError("abstract method")

    def get_value(self, val: Any) -> Any:
        raise NotImplementedError("abstract method")

    def call_grid(  # type: ignore[no-untyped-def]
        self,
        grid,
        meta,
        tx,
    ) -> tuple[int | sympy.Expr | SymInt, ...] | tuple["Proxy", ...]:
        raise NotImplementedError("abstract method")

    def wrap_user_defined_obj(
        self,
        user_obj: Any,
        tx: Optional["InstructionTranslatorBase"],
        variable: Union["TritonKernelVariable", "TraceableTritonKernelWrapper"] | None,
        name: str,
    ) -> Any:
        raise NotImplementedError("abstract method")

    def call_user_defined_fn(
        self,
        user_fn: Callable[..., Any],
        args: list,
        kwargs: dict,
        tx: Optional["InstructionTranslatorBase"],
        variable: Union["TritonKernelVariable", "TraceableTritonKernelWrapper"] | None,
    ) -> Any:
        raise NotImplementedError("abstract method")

    def maybe_unpack_configs(
        self, configs: list["TritonConfig"], tx: Optional["InstructionTranslatorBase"]
    ) -> list["TritonConfig"]:
        raise NotImplementedError("abstract method")

    def maybe_unpack_heuristic_result(self, result: Any) -> Any:
        raise NotImplementedError("abstract method")

    @staticmethod
    def do_prune_configs(  # type: ignore[no-untyped-def]
        autotuner: "TritonAutotunerType",
        early_config_prune: Callable | None,
        perf_model: Callable | None,
        top_k: float,
        configs: list,
        named_args: dict,
        kwargs: dict,
    ) -> list["TritonConfig"]:
        # Reimplement autotuner.prune_configs(...) here
        # see: https://github.com/triton-lang/triton/blob/e57b46897191b3b3061c78d0d60e58e94be565b6/python/triton/runtime/autotuner.py
        # We do this to avoid calling prune_configs, which in turn calls early_config_prune and perf_model
        # These are both user-defined functions which can contain side effects, so we want to sandbox them in Dynamo

        if early_config_prune:
            configs = early_config_prune(configs, named_args, **kwargs)

        if perf_model:
            # we assert top_k is a float before calling this
            if isinstance(top_k, float) and top_k <= 1.0:
                top_k = int(len(configs) * top_k)
            elif not isinstance(top_k, int):
                """
                Slice index must be an integer, SupportsIndex or None
                """
                raise TypeError(
                    "Error while pruning configs, top_k must be either 1) a float <= 1.0 or 2) an int"
                )
            if len(configs) > top_k:
                est_timing = [
                    (
                        config,
                        float(
                            perf_model(**named_args, **kwargs, **config.all_kwargs())
                        ),
                    )
                    for config in configs
                ]
                configs = [
                    config[0]
                    for config in sorted(est_timing, key=operator.itemgetter(1))[:top_k]
                ]
        return configs

    def call_HOP(  # type: ignore[no-untyped-def]
        self,
        variable,
        grids,
        combined_args: dict[str, Any],
        launch_kwargs: tuple[str, ...],
        kernel_arg_names: set[str],
        tx,
    ) -> Optional["ConstantVariable"]:
        raise NotImplementedError("abstract method")

    def check_grid(  # type: ignore[no-untyped-def]
        self, grid
    ) -> tuple[int | sympy.Expr | SymInt, ...] | tuple["Proxy", ...]:
        raise NotImplementedError("abstract method")

    def init_variable(
        self,
        variable: Union["TraceableTritonKernelWrapper", "TritonKernelVariable"],
        kernel: "TritonKernelType",
        kernel_idx: int | None,
        grid: Optional["TritonGridType"],
    ) -> None:
        from triton.runtime.autotuner import Autotuner

        if kernel is None:
            raise AssertionError("kernel cannot be None")

        variable.kernel = kernel
        variable.kernel_idx = kernel_side_table.add_kernel(kernel)

        if kernel_idx is not None and variable.kernel_idx != kernel_idx:
            raise AssertionError(
                f"kernel_idx mismatch: expected {kernel_idx}, got {variable.kernel_idx}"
            )

        # pyrefly: ignore [bad-assignment]
        variable.grid = grid

        if isinstance(kernel, Autotuner):
            import torch
            import torch._dynamo

            # We only support configs, keys, and restore_value arguments
            # of triton.autotune. Make sure other arguments are defaulted.
            defaults = inspect.signature(Autotuner.__init__).parameters
            # Newer version of triton change attribute name from warmup to num_warmup and rep to num_rep.
            # The call to get_first_attr is to maintain backward-compatibility.

            def defaults_ok(
                attr: str, alternates: tuple[str, ...], values: tuple[Any, ...]
            ) -> bool:
                if attr not in defaults:
                    return True
                value = torch._dynamo.utils.get_first_attr(kernel, attr, *alternates)
                if value == defaults[attr].default:
                    return True
                return value in values

            if (
                not torch._inductor.config.unsafe_ignore_unsupported_triton_autotune_args
                and (
                    not defaults_ok("num_warmups", ("warmup",), (25, None))
                    or not defaults_ok("num_reps", ("rep",), (100, None))
                    or not defaults_ok("use_cuda_graph", (), (False,))
                )
            ):
                self.raise_unsupported(
                    "Only configs, keys, restore_value, and reset_to_zero are supported for triton.autotune"
                )
            if (
                not torch._inductor.config.unsafe_ignore_unsupported_triton_autotune_args
                and (
                    # pre_hook requires running arbitrary code at runtime, which we cannot handle at this time
                    # https://github.com/pytorch/pytorch/issues/139059
                    # we can't support pre_hook or post_hook in user defined triton kernels at the moment,
                    # as they require the ability to execute code at runtime (AOTI can't support this)
                    (
                        hasattr(kernel, "user_defined_pre_hook")
                        and kernel.user_defined_pre_hook is not False
                    )
                    or (
                        hasattr(kernel, "user_defined_post_hook")
                        and kernel.user_defined_post_hook is not False
                    )
                    or (
                        # Check Config passed to autotuner in configs
                        any(cfg.pre_hook is not None for cfg in kernel.configs)
                    )
                )
            ):
                self.raise_unsupported(
                    "pre_hook and post_hook are not supported in triton.Autotune or triton.Config"
                )

    @staticmethod
    def get_kernel_source(
        variable: "TritonKernelVariable | TraceableTritonKernelWrapper",
    ) -> "Source | None":
        kernel_source = getattr(variable, "kernel_source", None)
        if kernel_source is None:
            kernel_source = getattr(variable, "source", None)
        return kernel_source

    def recreate_variable(
        self,
        variable: "TritonKernelVariable | TraceableTritonKernelWrapper",
        *,
        kernel: "TritonKernelType",
        kernel_idx: int | None,
        grid: "TritonGridType | None",
    ) -> "TritonKernelVariable | TraceableTritonKernelWrapper":
        return type(variable)(
            kernel=kernel,
            kernel_idx=kernel_idx,
            grid=grid,
            kernel_source=self.get_kernel_source(variable),
        )

    def call_getitem(
        self,
        variable: Union["TritonKernelVariable", "TraceableTritonKernelWrapper"],
        args: Sequence[Any],
    ) -> Union["TritonKernelVariable", "TraceableTritonKernelWrapper"]:
        # __getitem__ should only be called if we don't already have a grid
        # Only grid needs to be passed
        if variable.grid is not None or len(args) != 1:
            self.raise_unsupported(
                "Triton kernels should be called with only a single grid"
            )

        return self.recreate_variable(
            variable,
            kernel=variable.kernel,
            kernel_idx=variable.kernel_idx,
            grid=args[0],
        )

    def call_run(
        self,
        variable: Union["TritonKernelVariable", "TraceableTritonKernelWrapper"],
        args: Sequence[Any],
        kwargs: dict[str, Any],
        tx: Optional["InstructionTranslatorBase"],
    ) -> Optional["ConstantVariable"]:
        if "grid" not in kwargs:
            self.raise_unsupported("Triton kernel requires to be called with a grid")
        grid = kwargs.pop("grid")
        kwargs.pop("warmup", None)
        # rewrite kernel.run(*args, grid=grid) to kernel[grid](*args)
        return self.call_triton_kernel(
            self.recreate_variable(
                variable,
                kernel=variable.kernel,
                kernel_idx=variable.kernel_idx,
                grid=grid,
            ),
            args,
            kwargs,
            tx,
        )

    def call_triton_kernel(
        self,
        variable: Union["TritonKernelVariable", "TraceableTritonKernelWrapper"],
        args: Sequence[Any],
        kwargs: dict[str, Any],
        tx: Optional["InstructionTranslatorBase"],
    ) -> Optional["ConstantVariable"]:
        from triton import JITFunction
        from triton.runtime.autotuner import autotune, Autotuner, Config, Heuristics

        # Check if num_ctas is in kwargs
        if "num_ctas" in kwargs:
            self.raise_unsupported(
                "Passing num_ctas directly to the Triton kernel is not supported. "
                "Please use a Config in @triton.autotune instead."
            )

        # Make sure the kernel has a grid
        if variable.grid is None:
            self.raise_unsupported("Triton kernels should always be called with a grid")

        # raise an exception if there are multiple @triton.autotune decorators
        iter_kernel = variable.kernel
        autotuner_count = 0
        while not isinstance(iter_kernel, JITFunction):
            if isinstance(iter_kernel, Autotuner):
                autotuner_count += 1
            if autotuner_count > 1:
                self.raise_unsupported(
                    "Passing multiple @triton.autotune decorators is not supported. "
                    "Please use a single @triton.autotune decorator instead."
                )

            iter_kernel = iter_kernel.fn

        jit_arg_names = set(iter_kernel.arg_names)

        # Process the @triton.heuristics decorator:
        # - We know there is only 1 autotuner decorator here
        # - We can apply the heuristic to all triton.Configs in the order that the decorators appear
        #   This way, when the config is selected, the heuristics have already been applied.
        # - Decorators that appear *before* the autotuner are already processed correctly
        if isinstance(variable.kernel, Autotuner) and isinstance(
            variable.kernel.fn, Heuristics
        ):
            # unwrap the heuristics decorator, we don't need it anymore
            # variable.kernel ==> Autotuner
            # variable.kernel.fn ==> Heuristics
            # ...
            # There can be arbitrarily many heuristics wrappers here!
            # ...
            # variable.kernel.fn ==> JITFunction

            # Copy the configs, we are going to be modifying them
            new_configs = copy.deepcopy(variable.kernel.configs)

            named_args = dict(zip(variable.kernel.arg_names, args))

            # Iterate through all of the heuristics wrappers that come after the autotune wrapper
            iter_kernel = variable.kernel.fn
            while isinstance(iter_kernel, Heuristics):
                # For each config, apply the heuristic fn(s)
                for config_idx in range(len(new_configs)):
                    for kwarg_key, heuristic_fn in iter_kernel.values.items():
                        # Run heuristics on the combined configs + kwargs
                        heuristic_result = self.call_user_defined_fn(
                            heuristic_fn,
                            [
                                {
                                    **named_args,
                                    **kwargs,
                                    **new_configs[config_idx].__dict__["kwargs"],
                                },
                            ],
                            {},
                            tx,
                            variable,
                        )

                        # Update the kwargs in each config
                        # maybe_unpack_heuristic_result raises unsupported if the value is non-constant
                        new_configs[config_idx].__dict__["kwargs"][kwarg_key] = (
                            self.maybe_unpack_heuristic_result(heuristic_result)
                        )

                iter_kernel = iter_kernel.fn
            if not isinstance(iter_kernel, JITFunction):
                raise AssertionError(
                    f"Expected iter_kernel to be a JITFunction, got {type(iter_kernel)}"
                )
            prune_configs_by = {
                "perf_model": variable.kernel.perf_model,
                "early_config_prune": variable.kernel.early_config_prune,
                "configs_top_k": variable.kernel.configs_top_k,
            }
            new_kernel = autotune(
                configs=new_configs, key=[], prune_configs_by=prune_configs_by
            )(iter_kernel)
            # create a new variable to contain the new (wrapped) kernel;
            # skip kernel_idx to get a new record in the kernel side table
            new_var = self.recreate_variable(
                variable,
                kernel=new_kernel,
                kernel_idx=None,
                grid=variable.grid,
            )
            return self.call_triton_kernel(new_var, args, kwargs, tx)

        SPECIAL_CONFIG_NAMES = {
            "num_warps",
            "num_stages",
            "num_ctas",
            "num_consumer_groups",
            "num_buffers_warp_spec",
            "num_cpu_threads",
        }

        # move special config names to configs out of kwargs
        special_kwargs = {}
        for name in SPECIAL_CONFIG_NAMES:
            if name in kwargs:
                # remove special kwargs from `kwargs`
                val = kwargs.pop(name)
                special_kwargs[name] = self.get_value(val)

        if special_kwargs:
            if isinstance(variable.kernel, Autotuner):
                # if there is Autotuner already, set
                # special kwargs to each of its configs
                new_configs = copy.deepcopy(variable.kernel.configs)
                for config in new_configs:
                    config.__dict__.update(special_kwargs)
                prune_configs_by = {
                    "perf_model": variable.kernel.perf_model,
                    "early_config_prune": variable.kernel.early_config_prune,
                    "configs_top_k": variable.kernel.configs_top_k,
                }

                new_kernel = autotune(
                    configs=new_configs, key=[], prune_configs_by=prune_configs_by
                )(variable.kernel.fn)
            else:
                # if there is no Autotuner, wrap the kernel into a
                # new one with a single config with special kwargs
                new_config = Config(kwargs={}, **special_kwargs)

                new_kernel = autotune(configs=[new_config], key=[])(variable.kernel)

            # create a new variable to contain the new (wrapped) kernel;
            # skip kernel_idx to get a new record in the kernel side table
            new_var = self.recreate_variable(
                variable,
                kernel=new_kernel,
                kernel_idx=None,
                grid=variable.grid,
            )
            return self.call_triton_kernel(new_var, args, kwargs, tx)

        if isinstance(variable.kernel, Autotuner):
            special_param_names = []
            for name in SPECIAL_CONFIG_NAMES:
                if name in jit_arg_names:
                    special_param_names.append(name)

            if special_param_names:
                # If the Triton kernel has SPECIAL_CONFIG_NAMES in parameters, those should
                # be passed from the kernel configs: the behavior of Triton runtime is that
                # those values get folded into the kernel arguments iff there are parameters
                # with the same name. Normally the values of those parameters are defined
                # outside the `kwargs` part of the autotuning configs. Here we move them to
                # the `kwargs` part (if they're absent there) to facilitate passing them as
                # arguments to the kernel downstream.
                updated = False
                new_configs = copy.deepcopy(variable.kernel.configs)
                for config in new_configs:
                    for name in special_param_names:
                        if name not in config.__dict__["kwargs"]:
                            if name not in config.__dict__:
                                raise AssertionError(
                                    f"{name} must be in autotuning configs to be used "
                                    "as a kernel parameter"
                                )
                            config.__dict__["kwargs"][name] = config.__dict__[name]
                            updated = True

                if updated:
                    prune_configs_by = {
                        "perf_model": variable.kernel.perf_model,
                        "early_config_prune": variable.kernel.early_config_prune,
                        "configs_top_k": variable.kernel.configs_top_k,
                    }

                    new_kernel = autotune(
                        configs=new_configs, prune_configs_by=prune_configs_by, key=[]
                    )(variable.kernel.fn)
                    new_var = self.recreate_variable(
                        variable,
                        kernel=new_kernel,
                        kernel_idx=None,
                        grid=variable.grid,
                    )
                    return self.call_triton_kernel(new_var, args, kwargs, tx)

        # These are the default values in upstream Triton
        # see: https://github.com/triton-lang/triton/blob/e57b46897191b3b3061c78d0d60e58e94be565b6/python/triton/runtime/autotuner.py
        default_perf_model = None
        default_early_config_prune = None

        # run prune_configs_by
        if isinstance(variable.kernel, Autotuner) and (
            variable.kernel.perf_model != default_perf_model
            or variable.kernel.early_config_prune != default_early_config_prune
        ):
            # Prune the configs
            named_args = dict(zip(variable.kernel.arg_names, args))

            # The source information is important here so the guards are installed correctly

            wrapped_early_configs_prune = self.wrap_user_defined_obj(
                variable.kernel.early_config_prune,
                tx,
                variable,
                "early_config_prune",
            )

            wrapped_perf_model = self.wrap_user_defined_obj(
                variable.kernel.perf_model, tx, variable, "perf_model"
            )

            wrapped_configs_top_k = self.wrap_user_defined_obj(
                variable.kernel.configs_top_k, tx, variable, "configs_top_k"
            )

            wrapped_configs = self.wrap_user_defined_obj(
                variable.kernel.configs, tx, variable, "configs"
            )

            pruned_configs = self.call_user_defined_fn(
                self.do_prune_configs,
                [
                    variable,
                    wrapped_early_configs_prune,
                    wrapped_perf_model,
                    wrapped_configs_top_k,
                    wrapped_configs,
                    named_args,
                    kwargs,
                ],
                {},
                tx,
                variable,
            )

            pruned_configs = self.maybe_unpack_configs(pruned_configs, tx)

            # after pruning the configs, create a new autotuner object with
            # these configs and recurse.
            new_kernel = autotune(configs=pruned_configs, key=[])(variable.kernel.fn)
            # create a new variable to contain the new (wrapped) kernel;
            # skip kernel_idx to get a new record in the kernel side table
            new_var = self.recreate_variable(
                variable,
                kernel=new_kernel,
                kernel_idx=None,
                grid=variable.grid,
            )
            return self.call_triton_kernel(new_var, args, kwargs, tx)

        # Triton passes launch kwargs to backend.parse_options(), even when a
        # kwarg also binds a kernel parameter. Preserve the launch-site names;
        # values stay in combined_args_raw so Inductor materializes backend
        # options only after the target backend is known.
        launch_kwargs = tuple(kwargs)

        if isinstance(variable.kernel, Autotuner):
            config_kwarg_names = set().union(
                *(set(config.kwargs) for config in variable.kernel.configs)
            )
            config_conflicts = sorted(set(launch_kwargs) & config_kwarg_names)
            if config_conflicts:
                self.raise_unsupported(
                    "Triton launch kwargs conflict with autotune config kwargs: "
                    f"{config_conflicts!r}."
                )

        positional_arg_names = iter_kernel.arg_names[: len(args)]
        if len(args) > len(iter_kernel.arg_names):
            self.raise_unsupported(
                f"{iter_kernel.__name__}() takes {len(iter_kernel.arg_names)} "
                f"positional arguments but {len(args)} were given"
            )

        duplicate_kwargs = sorted(set(positional_arg_names) & set(kwargs))
        if duplicate_kwargs:
            self.raise_unsupported(
                f"{iter_kernel.__name__}() got multiple values for argument "
                f"{duplicate_kwargs[0]!r}"
            )

        # Both for grid's meta as well as for the kernel, we need combined
        # args and launch kwargs combined and normalized. Non-signature launch
        # kwargs are included here so Inductor can later recover their values
        # when materializing target-specific backend options.

        combined_args_raw = {**dict(zip(iter_kernel.arg_names, args)), **kwargs}

        # precompute the grid for the kernel
        configs = (
            [config.kwargs for config in variable.kernel.configs]
            if isinstance(variable.kernel, Autotuner)
            else [{}]
        )
        grids = []
        for config_args in configs:
            # If the grid is a function, then lets execute it and convert it to
            # a list
            grid = variable.grid
            if grid is None:
                raise AssertionError("grid cannot be None at this point")
            if self.is_callable(grid):
                # Populate the special "meta" argument to call the grid function
                meta = {**combined_args_raw, **config_args}
                grid = self.call_grid(grid, meta, tx)  # type: ignore[arg-type]
            grids.append(self.check_grid(grid))

        for i in range(len(grids)):
            if not isinstance(grids[i], tuple):
                self.raise_unsupported("Only tuple grids are supported")
            # inductor expects all grids to be 3-tuple so lets make it
            if len(grids[i]) == 1:
                grids[i] = (grids[i][0], 1, 1)
            elif len(grids[i]) == 2:
