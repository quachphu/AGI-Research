                value,
                out,
                logsumexp,
                max_scores,
                *block_mask[:-1],
                *score_mod_other_buffers,
                *mask_mod_other_buffers,
            ),
        )
        return out, logsumexp, max_scores

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_out: Tensor,
        grad_logsumexp: Tensor,
        grad_max_scores: Tensor,
    ) -> tuple[Tensor | None, ...]:
        fw_args = saved_values(ctx)
        (
            query,
            key,
            value,
            out,
            logsumexp,
            max_scores,
            query_lengths,
            kv_lengths,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
            dq_write_order,
            dq_write_order_full,
            dq_kv_order,
            dq_kv_order_spt,
            Q_BLOCK_SIZE,
            KV_BLOCK_SIZE,
            *other_buffers,
        ) = fw_args
        fw_graph = ctx._fw_graph
        joint_graph = ctx._joint_graph
        mask_graph = ctx._mask_graph
        scale = ctx.scale
        kernel_options = ctx.kernel_options
        score_mod_other_buffers = tuple(
            other_buffers[: ctx._score_mod_other_buffers_len]
        )
        mask_mod_other_buffers = tuple(
            other_buffers[ctx._score_mod_other_buffers_len :]
        )
        # We have asserted that mask_mod_other_buffers do not require grad,
        # but score_mod_other_buffers can require grad.
        none_grads = [None] * 6
        _sparsity_ctx = (
            annotate({"sparsity_hint": ctx._sparsity_hint})
            if ctx._sparsity_hint > 0
            else contextlib.nullcontext()
        )
        with _sparsity_ctx:
            (
                grad_query,
                grad_key,
                grad_value,
                grad_score_mod_captured,
            ) = flex_attention_backward(
                query,
                key,
                value,
                out,
                logsumexp,
                grad_out,
                grad_logsumexp,
                fw_graph,
                joint_graph,
                (
                    query_lengths,
                    kv_lengths,
                    kv_num_blocks,
                    kv_indices,
                    full_kv_num_blocks,
                    full_kv_indices,
                    q_num_blocks,
                    q_indices,
                    full_q_num_blocks,
                    full_q_indices,
                    dq_write_order,
                    dq_write_order_full,
                    dq_kv_order,
                    dq_kv_order_spt,
                    Q_BLOCK_SIZE,
                    KV_BLOCK_SIZE,
                    mask_graph,
                ),
                scale,
                kernel_options,
                score_mod_other_buffers,
                mask_mod_other_buffers,
            )
        return grad_query, grad_key, grad_value, *none_grads, *grad_score_mod_captured


# TODO: Rework DispatchKey.Autograd to py_autograd_impl
@flex_attention.py_impl(DispatchKey.Autograd)
def flex_attention_autograd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    score_mod: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple[Tensor, ...] = (),
    mask_mod_other_buffers: tuple[Tensor, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    with TransformGetItemToIndex():
        input_requires_grad = any(
            isinstance(t, torch.Tensor) and t.requires_grad
            for t in (query, key, value, *score_mod_other_buffers)
        )
        if torch.is_grad_enabled() and input_requires_grad:
            if block_mask[7] is None:
                raise RuntimeError(
                    "BlockMask q_indices is None. Backward pass requires q_indices to be computed. "
                    "Please create the BlockMask with compute_q_blocks=True"
                )
            example_vals = (
                query.new_zeros((), requires_grad=input_requires_grad),
                query.new_zeros((), dtype=torch.int),
                query.new_zeros((), dtype=torch.int),
                query.new_zeros((), dtype=torch.int),
                query.new_zeros((), dtype=torch.int),
            )
            fw_graph, bw_graph = create_fw_bw_graph(
                score_mod, example_vals, score_mod_other_buffers
            )
        else:
            fw_graph, bw_graph = score_mod, None
        out, logsumexp, max_scores = FlexAttentionAutogradOp.apply(
            query,
            key,
            value,
            fw_graph,
            bw_graph,
            block_mask,
            scale,
            kernel_options,
            mask_mod_other_buffers,
            *score_mod_other_buffers,
        )
    return out, logsumexp, max_scores


# ---------------------------- Backward HOP Implementation ----------------------------


@flex_attention_backward.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor | None,
    grad_logsumexp: torch.Tensor | None,
    fw_graph: Callable,  # GraphModule type hint?
    joint_graph: Callable,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple,
    mask_mod_other_buffers: tuple,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            f"Backward pass with mixed query, key, and value dtype is not supported, "
            f"got query.dtype={query.dtype}, key.dtype={key.dtype}, "
            f"and value.dtype={value.dtype}"
        )
    if joint_graph is None:
        example_vals = (
            query.new_zeros((), requires_grad=True),
            query.new_zeros((), dtype=torch.int),
            query.new_zeros((), dtype=torch.int),
            query.new_zeros((), dtype=torch.int),
            query.new_zeros((), dtype=torch.int),
        )
        _, joint_graph = create_fw_bw_graph(
            fw_graph, example_vals, score_mod_other_buffers
        )
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    Bq, Hq, seq_len_q, qk_head_dim = query.shape
    Bkv, Hkv, seq_len_kv, v_head_dim = value.shape

    # Get outputs before calling repeat interleave and permute to input stride orders
    actual_grad_query = query.new_empty((Bq, Hq, seq_len_q, qk_head_dim))
    actual_grad_query = _permute_strides(actual_grad_query, query.stride())

    actual_grad_key = key.new_empty((Bq, Hkv, seq_len_kv, qk_head_dim))
    actual_grad_key = _permute_strides(actual_grad_key, key.stride())

    actual_grad_value = value.new_empty((Bq, Hkv, seq_len_kv, v_head_dim))
    actual_grad_value = _permute_strides(actual_grad_value, value.stride())

    actual_grad_score_mod_captured = _new_captured_grad_buffers(
        score_mod_other_buffers, joint_graph
    )

    Bq, Bkv = query.size(0), key.size(0)
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.size()[1:]))
    value = value.expand((Bq, *value.size()[1:]))

    G = query.size(1) // key.size(1)
    key = torch.repeat_interleave(key, G, dim=1)
    value = torch.repeat_interleave(value, G, dim=1)

    if grad_out is None:
        grad_out = torch.zeros_like(out)
    if grad_logsumexp is None:
        grad_logsumexp = torch.zeros_like(logsumexp)

    # logsumexp is expected in log2 scale (as returned by the forward HOP).
    # The public flex_attention API converts lse to natural log before returning,
    # so callers using the public API must not pass that value here directly.
    logsumexp = logsumexp * math.log(2)
    # The backwards formula for the log -> log2 change of base in the forwards
    grad_logsumexp = grad_logsumexp / math.log(2)
    scores, post_mod_scores = _math_attention_inner(
        query,
        key,
        value,
        fw_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    masked_out_rows = logsumexp == -float("inf")
    softmax_scores = torch.exp(post_mod_scores - logsumexp.unsqueeze(-1))
    softmax_scores = torch.where(masked_out_rows.unsqueeze(-1), 0, softmax_scores)

    grad_value = softmax_scores.to(query.dtype).transpose(-2, -1) @ grad_out

    grad_softmax_scores = grad_out.to(dtype=softmax_scores.dtype) @ value.to(
        dtype=softmax_scores.dtype
    ).transpose(-2, -1)

    sum_scores = torch.sum(
        out.to(dtype=softmax_scores.dtype) * grad_out.to(dtype=softmax_scores.dtype),
        -1,
        keepdim=True,
    )
    grad_score_mod = softmax_scores * (
        grad_softmax_scores - sum_scores + grad_logsumexp.unsqueeze(-1)
    )

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    mask_graph = block_mask[-1]
    # Gradient of the inline score_mod function, with respect to the scores
    captured_buffers_in_dim = (None,) * len(score_mod_other_buffers)
    out_dims = [0, None, None, None, None] + [None] * len(score_mod_other_buffers)
    from torch.nn.attention.flex_attention import _vmap_for_bhqkv

    # inputs are [score, b, h, q_idx, kv_idx, gradOut, ...]
    # score and gradOut are "fully" batched
    joint_score_mod = _vmap_for_bhqkv(
        joint_graph,
        prefix=(0,),
        suffix=(0,) + captured_buffers_in_dim,
        out_dims=out_dims,
    )
    with TransformGetItemToIndex():
        grad_scores, _, _, _, _, *grad_score_mod_captured = joint_score_mod(
            scores, b, h, m, n, grad_score_mod, *score_mod_other_buffers
        )
    grad_scores = grad_scores * scale
    grad_scores = grad_scores.to(query.dtype)

    mask_mod = _vmap_for_bhqkv(
        mask_graph, prefix=(), suffix=(None,) * len(mask_mod_other_buffers)
    )
    with TransformGetItemToIndex():
        mask_scores = mask_mod(b, h, m, n, *mask_mod_other_buffers)
        grad_scores = torch.where(
            mask_scores, grad_scores, torch.tensor(0, dtype=query.dtype)
        )

    grad_query = grad_scores @ key
    grad_key = grad_scores.transpose(-2, -1) @ query

    # Reduce DK, DV along broadcasted heads.
    grad_key = grad_key.view(
        grad_key.size(0), -1, G, grad_key.size(-2), grad_key.size(-1)
    )
    grad_value = grad_value.view(
        grad_value.size(0), -1, G, grad_value.size(-2), grad_value.size(-1)
    )

    grad_key = torch.sum(grad_key, 2, keepdim=False)
    grad_value = torch.sum(grad_value, 2, keepdim=False)

    # Fill to correctly strided outputs
    actual_grad_query.copy_(grad_query)
    actual_grad_key.copy_(grad_key)
    actual_grad_value.copy_(grad_value)

    if Bq != Bkv:
        if not (Bq > 1 and Bkv == 1):
            raise AssertionError(
                f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}"
            )

        actual_grad_key = torch.sum(actual_grad_key, 0, keepdim=True)
        actual_grad_value = torch.sum(actual_grad_value, 0, keepdim=True)

    score_mod_other_buffer_grads = []
    for actual_grad, grad in zip(
        actual_grad_score_mod_captured, grad_score_mod_captured
    ):
        if not isinstance(grad, torch.Tensor):
            score_mod_other_buffer_grads.append(None)
            continue
        if actual_grad is None:
            raise AssertionError("Expected a captured grad buffer for a tensor grad")
        score_mod_other_buffer_grads.append(actual_grad.copy_(grad))

    return (
        actual_grad_query,
        actual_grad_key,
        actual_grad_value,
        tuple(score_mod_other_buffer_grads),
    )


def _captured_grad_buffer_mask(
    joint_graph: Callable | GraphModule, num_captures: int
) -> tuple[bool, ...]:
    """Return which score_mod captures need concrete grad buffers.

    A capture can require grad but still be dead in score_mod, for example when
    the function reads a closed-over tensor but returns the original score. AOT
    represents that as a None grad in the joint graph, and that None must stay a
    None instead of being copied into a materialized buffer.

    Conversely, this is independent of the saved capture's requires_grad flag:
    compiled backward may save an intermediate whose grad is needed to propagate
    to earlier user inputs.
    """
    if not isinstance(joint_graph, GraphModule):
        # The @register_fake path for direct flex_attention_backward HOP calls
        # can see callable subgraphs before make_fx materializes them. In that
        # metadata-only phase, no capture is provably dead.
        return (True,) * num_captures

    output_node = next(node for node in joint_graph.graph.nodes if node.op == "output")
    score_mod_arg_grads = 5
    captured_grad_outputs = output_node.args[0][score_mod_arg_grads:]
    if len(captured_grad_outputs) != num_captures:
        raise AssertionError(
            f"Expected {num_captures} captured grad outputs, "
            f"got {len(captured_grad_outputs)}"
        )
    return tuple(grad is not None for grad in captured_grad_outputs)


def _new_captured_grad_buffers(
    score_mod_other_buffers: tuple, joint_graph: Callable | GraphModule
) -> tuple[torch.Tensor | None, ...]:
    """Allocate only the captured grad buffers that the joint graph can produce."""
    needs_grad_buffer = _captured_grad_buffer_mask(
        joint_graph, len(score_mod_other_buffers)
    )
    # The lowering returns captured grads as contiguous buffers, so match that here.
    return tuple(
        torch.empty_like(buffer, memory_format=torch.contiguous_format)
        if isinstance(buffer, torch.Tensor) and needs_buffer
        else None
        for buffer, needs_buffer in zip(score_mod_other_buffers, needs_grad_buffer)
    )


def trace_flex_attention_backward(
    proxy_mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable | GraphModule,
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """We already have the forward graph and joint graph from the forward pass, so we create a proxy attach both graphs"""
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    requires_grad = any(pytree.tree_map(lambda x: x.requires_grad, (query, key)))
    fw_example_vals = [query.new_zeros((), requires_grad=requires_grad)] + [
        query.new_zeros((), dtype=torch.int) for _ in range(4)
    ]
    bw_example_vals = fw_example_vals + [query.new_zeros(())]
    mask_example_vals = [query.new_zeros((), dtype=torch.int) for _ in range(4)]
    mask_graph = block_mask[-1]
    with TransformGetItemToIndex():
        # There's no active make_fx during the compiled autograd graph's initial capture
        fw_graph = _maybe_reenter_make_fx(fw_graph)(
            *fw_example_vals, *score_mod_other_buffers
        )
        joint_graph = _maybe_reenter_make_fx(joint_graph)(
            *bw_example_vals, *score_mod_other_buffers
        )
        mask_graph = _maybe_reenter_make_fx(mask_graph)(
            *mask_example_vals, *mask_mod_other_buffers
        )
    block_mask = block_mask[:-1] + (mask_graph,)
    example_out = flex_attention_backward(
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        joint_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )

    if not isinstance(proxy_mode.tracer, torch.fx.Tracer):
        raise AssertionError(
            f"expected proxy_mode.tracer to be torch.fx.Tracer, got {type(proxy_mode.tracer)}"
        )

    qualname = proxy_mode.tracer.get_fresh_qualname("fw_graph")
    proxy_mode.tracer.root.register_module(qualname, fw_graph)  # type: ignore[arg-type]
    qualname = proxy_mode.tracer.get_fresh_qualname("joint_graph")
    proxy_mode.tracer.root.register_module(qualname, joint_graph)
    qualname = proxy_mode.tracer.get_fresh_qualname("mask_graph")
    proxy_mode.tracer.root.register_module(qualname, mask_graph)

    node_args = (
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        fw_graph,
        joint_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    # pyrefly: ignore [missing-attribute]
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function",
        flex_attention_backward,
        proxy_args,
        {},
        name="flex_attention_backward",
    )
    return track_tensor_tree(
        example_out,
        out_proxy,
        constant=None,
        tracer=proxy_mode.tracer,
    )


@flex_attention_backward.py_impl(ProxyTorchDispatchMode)
def flex_attention_backward_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable | GraphModule,
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    if mode is None:
        raise AssertionError("Mode should always be enabled for python fallback key")
    with torch.fx.experimental.proxy_tensor.set_original_aten_op(
        flex_attention_backward
    ):
        return trace_flex_attention_backward(
            mode,
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            fw_graph,
            joint_graph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        )


@flex_attention_backward.py_functionalize_impl
def flex_attention_backward_functionalize(
    ctx: torch._subclasses.functional_tensor.BaseFunctionalizeAPI,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable | GraphModule,
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    """Defines the functionalization rules for the flex_attention operator.

    Right now we are unwrapping each tensor and then redispatching to the next,
    since we know that the forward score mod function is assured to be free of mutations
    to the other_buffers, we skip that mutate check and go straight to redispatching.
    """

    query_unwrapped = ctx.unwrap_tensors(query)
    key_unwrapped = ctx.unwrap_tensors(key)
    value_unwrapped = ctx.unwrap_tensors(value)
    out_unwrapped = ctx.unwrap_tensors(out)
    logsumexp_unwrapped = ctx.unwrap_tensors(logsumexp)
    grad_out_unwrapped = ctx.unwrap_tensors(grad_out)
    grad_logsumexp_unwrapped = ctx.unwrap_tensors(grad_logsumexp)
    block_mask_unwrapped = ctx.unwrap_tensors(block_mask)
    score_mod_other_buffers_unwrapped = ctx.unwrap_tensors(score_mod_other_buffers)
    mask_mod_other_buffers_unwrapped = ctx.unwrap_tensors(mask_mod_other_buffers)

    # Appease the mypy overlords
    if not isinstance(query_unwrapped, torch.Tensor):
        raise AssertionError(
            f"expected query_unwrapped to be torch.Tensor, got {type(query_unwrapped)}"
        )
    if not isinstance(key_unwrapped, torch.Tensor):
        raise AssertionError(
            f"expected key_unwrapped to be torch.Tensor, got {type(key_unwrapped)}"
        )
    if not isinstance(value_unwrapped, torch.Tensor):
        raise AssertionError(
            f"expected value_unwrapped to be torch.Tensor, got {type(value_unwrapped)}"
        )
    if not isinstance(out_unwrapped, torch.Tensor):
        raise AssertionError(
            f"expected out_unwrapped to be torch.Tensor, got {type(out_unwrapped)}"
        )
    if not isinstance(logsumexp_unwrapped, torch.Tensor):
        raise AssertionError(
            f"expected logsumexp_unwrapped to be torch.Tensor, got {type(logsumexp_unwrapped)}"
        )
    if grad_out_unwrapped is not None and not isinstance(
        grad_out_unwrapped, torch.Tensor
    ):
        raise AssertionError(
            f"expected grad_out_unwrapped to be torch.Tensor or None, got {type(grad_out_unwrapped)}"
        )
    if grad_logsumexp_unwrapped is not None and not isinstance(
        grad_logsumexp_unwrapped, torch.Tensor
    ):
        raise AssertionError(
            f"expected grad_logsumexp_unwrapped to be torch.Tensor or None, got {type(grad_logsumexp_unwrapped)}"
        )
    if not isinstance(block_mask_unwrapped, tuple):
        raise AssertionError(
            f"expected block_mask_unwrapped to be tuple, got {type(block_mask_unwrapped)}"
        )
    if not isinstance(score_mod_other_buffers_unwrapped, tuple):
        raise AssertionError(
            f"expected score_mod_other_buffers_unwrapped to be tuple, got {type(score_mod_other_buffers_unwrapped)}"
        )
    if not isinstance(mask_mod_other_buffers_unwrapped, tuple):
        raise AssertionError(
            f"expected mask_mod_other_buffers_unwrapped to be tuple, got {type(mask_mod_other_buffers_unwrapped)}"
        )

    with ctx.redispatch_to_next():
        # pyrefly: ignore [bad-argument-type]
        functional_fw_graph = ctx.functionalize(fw_graph)
        # pyrefly: ignore [bad-argument-type]
        functional_joint_graph = ctx.functionalize(joint_graph)

        (
            grad_query,
            grad_key,
            grad_value,
            grad_score_mod_captured,
        ) = flex_attention_backward(
            query_unwrapped,
            key_unwrapped,
            value_unwrapped,
            out_unwrapped,
            logsumexp_unwrapped,
            grad_out_unwrapped,
            grad_logsumexp_unwrapped,
            functional_fw_graph,  # type: ignore[arg-type]
            functional_joint_graph,  # type: ignore[arg-type]
            block_mask_unwrapped,
            scale,
            kernel_options,
            score_mod_other_buffers_unwrapped,
            mask_mod_other_buffers_unwrapped,
        )

    return ctx.wrap_tensors((grad_query, grad_key, grad_value, grad_score_mod_captured))  # type: ignore[return-value,arg-type]


@register_fake(flex_attention_backward)
def flex_attention_backward_fake_tensor_mode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable | GraphModule,
    joint_graph: GraphModule,
    block_mask: tuple,
    scale: float,
    kernel_options: dict[str, Any],
    score_mod_other_buffers: tuple = (),
    mask_mod_other_buffers: tuple = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]:
    if has_user_subclass(
        (
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        ),
        allowed_subclasses=(FakeTensor,),
    ):
        return NotImplemented
    Bq, _, _, qk_head_dim = query.shape
    Bkv, Hkv, seq_len_kv, v_head_dim = value.shape

    grad_query = query.new_empty(query.shape)
    grad_query = _permute_strides(grad_query, query.stride())

    grad_score_mod_captured = _new_captured_grad_buffers(
        score_mod_other_buffers, joint_graph
    )

    broadcasted_grad_key = key.new_empty((Bq, Hkv, seq_len_kv, qk_head_dim))
    broadcasted_grad_key = _permute_strides(broadcasted_grad_key, key.stride())

    broadcasted_grad_value = value.new_empty((Bq, Hkv, seq_len_kv, v_head_dim))
    broadcasted_grad_value = _permute_strides(broadcasted_grad_value, value.stride())

    from torch.fx.experimental.symbolic_shapes import guard_or_false, sym_and

    # This branch chooses fake tensor metadata, including strides. Guarding is
    # valid for backed symbolic sizes, while unbacked/data-dependent predicates
    # conservatively fall through to the non-broadcast contract below.
    if guard_or_false(sym_and(Bkv == 1, Bq > 1)):
        grad_key = torch.sum(broadcasted_grad_key, dim=0, keepdim=True)
        grad_value = torch.sum(broadcasted_grad_value, dim=0, keepdim=True)
    else:
        torch._check(
            Bq == Bkv,
            lambda: "grad_key/grad_value batch must match key/value batch.",
        )
        grad_key = broadcasted_grad_key
        grad_value = broadcasted_grad_value

    return grad_query, grad_key, grad_value, grad_score_mod_captured


flex_attention_backward.py_autograd_impl(
    autograd_not_implemented(flex_attention_backward, deferred_error=True)
)
<|endoftext|>import builtins

import torch
import torch.utils._pytree as pytree
from torch._higher_order_ops.utils import register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


class Print(HigherOrderOperator):
    """
    print(format_str, *args, **kwargs) -> None

    This Higher Order Operator (HOP) provides a functional version of print for use in PyTorch graphs.
    It supports the calling conventions of print(format_str.format(*args, **kwargs)):

    1. Format string with keyword arguments (named placeholders):
       torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
       Output: "moo 1 2"

    2. Format string with positional arguments (positional placeholders):
       torch._higher_order_ops.print("moo {} {}", 1, 2)
       Output: "moo 1 2"

    3. Mixed positional and keyword arguments:
       torch._higher_order_ops.print("moo {} {y}", 1, y=2)
       Output: "moo 1 2"

    4. DTensor support:
       DTensor args are unwrapped to local tensors via to_local() (no collective).
       Each rank prints its own local view, prefixed with [rank N].
       For the global view of a sharded tensor, call full_tensor() before
       passing to print.

       dt = DTensor.from_local(local_shard, mesh, [Shard(0)])
       torch._higher_order_ops.print("activations: {}", dt)
       # Output: [rank 0] activations: tensor([0., 1.])

    This HOP enables printing without causing graph break.
    """

    def __init__(self) -> None:
        super().__init__("print")

    def __call__(self, format_str: str, *args: object, **kwargs: object) -> None:
        if not isinstance(format_str, str):
            raise AssertionError(f"format_str must be a string, got {type(format_str)}")
        # pyrefly: ignore [missing-attribute]
        return super().__call__(format_str, *args, **kwargs)

    # pyrefly: ignore [bad-override]
    def gen_schema(
        self, format_str: str, *args: object, **kwargs: object
    ) -> torch.FunctionSchema:
        from torch._higher_order_ops.schema import HopSchemaGenerator

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("format_str", format_str[0])

        # Add each positional arg
        for i, value in enumerate(args):
            schema_gen.add_arg(f"arg{i}", value)

        # Add each kwarg as a keyword-only argument
        for key, value in kwargs.items():
            schema_gen.add_arg(key, value, kw_only=True)

        schema_gen.add_schema_tree_spec(format_str, *args, **kwargs)

        return schema_gen.gen_schema()


print = Print()


@print.py_impl(ProxyTorchDispatchMode)
# pyre-ignore
def print_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode, format_str: str, *args: object, **kwargs: object
) -> None:
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)  # type: ignore[union-attr]
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)  # type: ignore[union-attr]
    mode.tracer.create_proxy(
        "call_function", print, (format_str, *proxy_args), proxy_kwargs
    )


@register_fake(print, skip_cache=True)
# pyre-ignore
def print_fake_tensor_mode(format_str: str, *args: object, **kwargs: object):
    return None


@print.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
# pyre-ignore
def print_impl(format_str: str, *args: object, **kwargs: object) -> None:
    # Ensure all immutable_dict/list in args and kwargs are converted to regular dict/list
    map_types: dict[type, type] = {
        torch.fx.immutable_collections.immutable_dict: dict,
        torch.fx.immutable_collections.immutable_list: list,
    }
    new_args, new_kwargs = pytree.tree_map_only(
        tuple(map_types.keys()),
        lambda a: map_types[type(a)](a),
        (args, kwargs),
        lambda a: isinstance(a, tuple(map_types.keys())),
    )
    #  Use built-in print to avoid recursion with the HOP print
    builtins.print(format_str.format(*new_args, **new_kwargs))


print.fallthrough(torch._C.DispatchKey.AutogradCPU)
print.fallthrough(torch._C.DispatchKey.AutogradCUDA)


def _register_dtensor_impl() -> None:
    from torch.distributed.tensor import DTensor

    @print.py_impl(DTensor)  # pyrefly: ignore [missing-attribute]
    # pyre-ignore
    def print_dtensor(format_str: str, *args: object, **kwargs: object) -> None:
        # Unwrap DTensors to local tensors via to_local() — no collective is
        # introduced so there is no OOM or performance risk.  Every rank prints
        # its own local view (including Replicate, where to_local() already
        # holds the full tensor).
        #
        # The output is prefixed with [rank N] so users can identify which rank
        # produced each line.
        #
        # If the user needs the global view of a sharded tensor, they can call
        # full_tensor() explicitly before passing it to print.
        import torch.distributed as dist

        local_args = pytree.tree_map_only(DTensor, DTensor.to_local, args)
        local_kwargs = pytree.tree_map_only(DTensor, DTensor.to_local, kwargs)
        if dist.is_initialized() and dist.get_world_size() > 1:
            format_str = f"[rank {dist.get_rank()}] {format_str}"
        print(  # pyrefly: ignore [no-matching-overload]
            format_str, *local_args, **local_kwargs
        )


@print.py_functionalize_impl
def print_func(ctx, format_str: str, *args: object, **kwargs: object):
    from torch._higher_order_ops.effects import handle_effects

    return handle_effects(
        ctx.mode._allow_token_discovery,
        ctx.mode._tokens,
        print,  # type: ignore[arg-type]
        (format_str, *args),
        kwargs,  # type: ignore[arg-type]
    )
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


class CUDAGraphCaptureControlFlowOpDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(
        self,
    ) -> None:
        self.supports_higher_order_operators = True
        super().__init__()

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        kwargs = {} if kwargs is None else kwargs
        if func is torch.ops.higher_order.cond:
            # Re-enter the mode to support nested conditionals
            _check_no_cond_kwargs(kwargs)
            with self:
                return if_else_node(*args)
        if func is torch.ops.higher_order.while_loop:
            # Re-enter the mode to support nested control flow
            _check_no_while_loop_kwargs(kwargs)
            with self:
                return while_loop_node(*args)
        return func(*args, **kwargs)


class ControlFlowOpWarmupDispatchMode(TorchDispatchMode):
    """Warm up control-flow subgraphs before CUDA graph capture.

    Data-dependent control flow does not necessarily execute every subgraph, so
    operations in an untaken torch.cond branch or a torch.while_loop body may
    not have been warmed up. This mode uses a relaxed stream capture, whose
    final CUDA graph is discarded, to warm up both cond branches and execute a
    while_loop body once. This works because stream capture does not execute GPU
    code and the branch and body functions are FX graphs without CPU side
    effects.
    """

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.supports_higher_order_operators = True
        self.capture_stream = torch.cuda.Stream()

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        kwargs = {} if kwargs is None else kwargs

        # Warm up both sides of this torch.cond()
        if func is torch.ops.higher_order.cond:
            if torch.cuda.is_current_stream_capturing():
                # This is a call to torch.cond() nested within another
                # control-flow function.
                _check_no_cond_kwargs(kwargs)
                with self:
                    # We re-enter the mode in case of nested calls to torch.cond()
                    return if_else_node(*args)
            else:
                _check_no_cond_kwargs(kwargs)
                with (
                    torch.cuda.graph(
                        torch.cuda.CUDAGraph(),
                        pool=None,
                        stream=self.capture_stream,
                        capture_error_mode="relaxed",
                    ),
                    self,
                ):
                    if_else_node(*args)

                return func(*args, **kwargs)
        elif func is torch.ops.higher_order.while_loop:
            _check_no_while_loop_kwargs(kwargs)
            if torch.cuda.is_current_stream_capturing():
                # This is a call to torch.while_loop() nested within another
                # control-flow function.
                with self:
                    return while_loop_node(*args)
            else:
                with (
                    torch.cuda.graph(
                        torch.cuda.CUDAGraph(),
                        pool=None,
                        stream=self.capture_stream,
                        capture_error_mode="relaxed",
                    ),
                    self,
                ):
                    while_loop_node(*args)

                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)


def _check_no_cond_kwargs(kwargs) -> None:
    if kwargs:
        raise RuntimeError("CUDA graph conditional torch.cond does not support kwargs")


def _check_no_while_loop_kwargs(kwargs) -> None:
    if kwargs:
        raise RuntimeError(
            "CUDA graph conditional torch.while_loop does not support kwargs"
        )


def _is_boolean_scalar_cuda_tensor(pred: object) -> bool:
    return (
        isinstance(pred, torch.Tensor)
        and pred.size() == torch.Size([])
        and pred.dtype == torch.bool
        and pred.is_cuda
    )


@contextmanager
def _if_body(pred: torch.Tensor) -> Generator[None, None, None]:
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    current_cuda_graph.begin_capture_to_if_node(pred)
    try:
        yield
    finally:
        current_cuda_graph.end_capture_to_conditional_node()


def if_else_node(pred: torch.Tensor, true_fn, false_fn, operands):
    if not pred.is_cuda:
        raise ValueError(
            "Conditions must be on a cuda device to use conditional node in cuda graphs"
        )
    # if-else is not supported until CUDA 12.8. Therefore, we use two
    # if conditions, where one evaluates !pred
    outs = []

    for lazy_pred, fn in [
        (lambda: pred, true_fn),
        (lambda: torch.logical_not(pred), false_fn),
    ]:
        with _if_body(lazy_pred()):
            outs.append(fn(*operands))

            # The output of the else branch gets copied into the
            # output of the if branch. This is done because the rest
            # of the cudagraph after the conditional node has fixed
            # inputs, so we need to merge the two outputs into a
            # single output.
            if len(outs) == 2:
                for if_out, else_out in zip(
                    pytree.tree_iter(outs[0]), pytree.tree_iter(outs[1])
                ):
                    if_out.copy_(else_out)
    return outs[0]


@contextmanager
def _while_body(pred: torch.Tensor):
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    current_cuda_graph.begin_capture_to_while_node(pred)
    try:
        yield current_cuda_graph
    finally:
        current_cuda_graph.end_capture_to_conditional_node()


def while_loop_node(
    cond_fn,
    body_fn,
    carried_inputs,
    additional_inputs,
):
    flat_carried_inputs, carried_spec = pytree.tree_flatten(carried_inputs)
    if not all(isinstance(inp, torch.Tensor) for inp in flat_carried_inputs):
        raise RuntimeError(
            "CUDA graph while_loop conditional nodes only support tensor carried_inputs"
        )

    loop_carried = pytree.tree_map_only(
        torch.Tensor, lambda inp: inp.clone(), carried_inputs
    )
    flat_loop_carried = pytree.tree_leaves(loop_carried)

    pred = cond_fn(*loop_carried, *additional_inputs)
    if not _is_boolean_scalar_cuda_tensor(pred):
        raise RuntimeError(
            f"cond_fn must return a boolean scalar CUDA tensor but got {pred}"
        )

    with _while_body(pred) as current_cuda_graph:
        body_out = body_fn(*loop_carried, *additional_inputs)
        flat_body_out, body_out_spec = pytree.tree_flatten(body_out)
        if body_out_spec != carried_spec:
            raise RuntimeError(
                "body_fn should return the same pytree structure as carried_inputs"
            )
        if not all(isinstance(out, torch.Tensor) for out in flat_body_out):
            raise RuntimeError(
                "CUDA graph while_loop conditional nodes only support tensor "
                "body_fn outputs"
            )

        for carried, out in zip(flat_loop_carried, flat_body_out):
            if carried.data_ptr() != out.data_ptr():
                carried.copy_(out)

        pred = cond_fn(*loop_carried, *additional_inputs)
        if not _is_boolean_scalar_cuda_tensor(pred):
            raise RuntimeError(
                f"cond_fn must return a boolean scalar CUDA tensor but got {pred}"
            )
        current_cuda_graph.set_conditional_handle_for_current_node(pred)

    return loop_carried
<|endoftext|>import contextlib
import functools
import logging
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import (
    _maybe_compile_and_run_fn,
    _maybe_run_with_interpreter,
    check_input_alias_and_mutation_return_outputs,
    create_bw_fn,
    create_fn_remove_none,
    fill_none_with_masks,
    materialize_as_graph,
    reenter_make_fx,
    save_values_for_backward,
    saved_values,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode


log = logging.getLogger(__name__)


class SwitchOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("switch")

    def __call__(self, index, branches, operands):
        validate_subgraph_args_types(operands)
        # pyrefly: ignore [missing-attribute]
        return super().__call__(index, branches, operands)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, index, branches, operands):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import materialize_as_graph

        branch_gms: list[torch.fx.GraphModule] = []
        branch0_outputs: list[Any] | tuple[Any, ...] = []
        mutated_inputs: set[int] = set()
        for branch in branches:
            gm = materialize_as_graph(branch, operands)
            (
                _,
                _,
                _,
                branch_mutated_inputs,
                branch_outputs,
            ) = check_input_alias_and_mutation_return_outputs(gm)
            branch_gms.append(gm)
            if not branch0_outputs:
                branch0_outputs = branch_outputs
            mutated_inputs |= set(branch_mutated_inputs)

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("index", index)
        for i, gm in enumerate(branch_gms):
            schema_gen.add_arg(f"branch{i}_fn", gm)
        for idx, arg in enumerate(operands):
            schema_gen.add_arg(f"operand{idx}", arg, is_mutated=idx in mutated_inputs)

        for out in branch0_outputs:
            schema_gen.add_output(out)
        schema_gen.add_schema_tree_spec(index, branches, operands)
        return schema_gen.gen_schema()


switch_op = SwitchOp()


def wrap_branch_fn_flat(*args, branch_fn, spec_operands):
    operands = pytree.tree_unflatten(args, spec_operands)
    return branch_fn(*operands)


def _get_branch(branches, idx):
    if not 0 <= idx < len(branches):
        raise AssertionError(
            f"switch index {idx} out of range for {len(branches)} branches"
        )
    return branches[idx]


@exposed_in("torch")
def switch(
    index: int | torch.SymInt | torch.Tensor,
    branches: tuple[Callable, ...] | list[Callable],
    operands: tuple | list = (),
) -> Any:
    r"""
    Selects and runs one of N branch functions by index.

    .. warning::

        `torch.switch` is a prototype feature in PyTorch. It has limited support for input and
        output types. Please look forward to a more stable implementation in a future version of
        PyTorch. Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    Equivalent to: ``branches[index](*operands)`` with index in ``[0, len(branches))``.

    Args:
        index (Union[int, torch.Tensor]): An int or single-element integer tensor
          indicating which branch to run. Out-of-range values are clamped into
          ``[0, len(branches))``.

        branches (Union[tuple[Callable, ...], list[Callable]]): Non-empty sequence of
          callables. Each must accept operands and return the same structure of outputs.

        operands (Tuple of possibly nested dict/list/tuple of torch.Tensor): Inputs to
          the branch functions. Defaults to ().

    Restrictions:
        - Each branch must have the same signature as operands and return the same
          output structure (shape, dtype, etc.). Constant ``int`` and ``None`` leaves
          are also permitted in branch outputs and are merged across branches (an
          unbacked SymInt is introduced when ``int`` leaves differ between branches).
        - Branches cannot have in-place mutations on inputs or global variables.
    """

    # Flatten operands so the HOP only sees a flat list of tensors.
    # Each branch is wrapped to unflatten the leaves
    # back to the user-facing pytree before invocation.
    leaves_operands, spec_operands = pytree.tree_flatten(operands)

    def _validate_input(index, branches, operands):
        if not isinstance(index, (int, torch.SymInt, torch.Tensor)):
            raise RuntimeError(
                f"Expected index to be int or single-element tensor, but got {index}."
            )

        if isinstance(index, torch.Tensor) and index.numel() != 1:
            raise RuntimeError(
                f"Expected index to be int or single-element tensor, but got {index}."
            )

        if not isinstance(branches, (tuple, list)) or len(branches) == 0:
            raise RuntimeError(
                "Expected branches to be a non-empty tuple or list of callables."
            )

        for i, branch in enumerate(branches):
            if not callable(branch):
                raise RuntimeError(
                    f"Expected all branches to be callable. branch{i} is not callable."
                )

        for x in operands:
            if not isinstance(x, torch.Tensor):
                raise RuntimeError(f"All operand leaves must be a Tensor but got {x}")

    _validate_input(index, branches, leaves_operands)

    wrapped_branches = tuple(
        functools.partial(
            wrap_branch_fn_flat,
            branch_fn=branch_fn,
            spec_operands=spec_operands,
        )
        for branch_fn in branches
    )

    # Early shortcut: single-branch switch degenerates to a plain call
    num_branches = len(wrapped_branches)
    if num_branches == 1:
        return wrapped_branches[0](*leaves_operands)

    # Clamp out-of-range indices to [0, len(branches) - 1]
    if isinstance(index, torch.Tensor):
        index = index.clamp(0, num_branches - 1)
    elif isinstance(index, torch.SymInt):
        index = torch.sym_max(0, torch.sym_min(index, num_branches - 1))
    elif isinstance(index, int):
        index = max(0, min(index, num_branches - 1))

    # Constant index shortcut for eager mode.
    if not torch.compiler.is_dynamo_compiling() and isinstance(index, int):
        # This is the non-strict export case. Strict export and torch.compile are
        # handled below via _maybe_compile_and_run_fn.
        if torch.compiler.is_compiling():
            warnings.warn(
                "Index is a Python constant. When used with torch.switch, it specializes on one of the branches."
                " If you want torch.switch to preserve the branches, please make the index an int tensor or a SymInt.",
                UserWarning,
                stacklevel=2,
            )

        return _get_branch(wrapped_branches, index)(*leaves_operands)

    # Use _maybe_compile_and_run_fn pattern from scan/associative_scan
    def run_switch(index, wrapped_branches, leaves_operands):
        return switch_op(index, wrapped_branches, leaves_operands)

    return _maybe_compile_and_run_fn(
        run_switch, index, wrapped_branches, leaves_operands
    )


def trace_switch(proxy_mode, func_overload, index, branches, operands):
    if not isinstance(operands, (list, tuple)):
        raise AssertionError(
            f"Switch operands must be a list or tuple of tensors and SymInts {operands}"
        )
    if not isinstance(branches, (list, tuple)) or len(branches) == 0:
        raise AssertionError(
            "Switch branches must be a non-empty list or tuple of callables"
        )

    branch_graphs = [reenter_make_fx(branch)(*operands) for branch in branches]

    branch_outs = [list() for _ in branches]
    for i, branch_graph in enumerate(branch_graphs):
        for node in branch_graph.graph.nodes:
            if node.op == "output":
                branch_outs[i].extend(node.args)

    branch_out_spec = [pytree.tree_flatten(outs)[1] for outs in branch_outs]
    for i, spec in enumerate(branch_out_spec):
        if branch_out_spec[0] != spec:
            raise RuntimeError(
                "Unmatched output spec from torch.switch branches: "
                f"branch0 tree_spec {branch_out_spec[0]} vs branch{i} tree_spec {spec}"
            )

    uid, _ = unique_graph_id(proxy_mode, prefix="branch0_graph")
    for i, branch_graph in enumerate(branch_graphs):
        proxy_mode.tracer.root.register_module(f"branch{i}_graph_{uid}", branch_graph)

    args = (index, branch_graphs, operands)

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}
    )

    out = func_overload(index, branch_graphs, operands)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@switch_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def switch_op_dense(index, branches, operands):
    if not all(isinstance(o, (torch.Tensor, int)) for o in operands):
        raise RuntimeError(
            f"Dense implementation operands must be a list of tensors and ints {operands}"
        )
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    idx: int = int(index.item()) if isinstance(index, torch.Tensor) else int(index)
    return _get_branch(branches, idx)(*operands)


class SwitchAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        index,
        branches,
        *operands,
    ):
        ctx._index = index
        # Build one bw fn per branch.
        ctx._branch_bw_fns = [
            create_bw_fn(create_fn_remove_none(branch)[0], operands)
            for branch in branches
        ]

        # We snapshot the dispatch keys in forward for materializing the
        # bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()
        save_values_for_backward(ctx, operands)

        with torch._C._AutoDispatchBelowAutograd():
            outs = switch_op(index, branches, operands)

        # Record which output slots are Tensors. Non-Tensor slots (None,
        # int/SymInt) carry no tangent in backward and the joint is built
        # to omit them.
        ctx._fw_output_is_tensor = [
            isinstance(o, torch.Tensor) for o in pytree.tree_leaves(outs)
        ]
        return outs

    @staticmethod
    def backward(ctx, *flat_grads):
        operands = saved_values(ctx)
        tensor_grads = tuple(
            g for g, keep in zip(flat_grads, ctx._fw_output_is_tensor) if keep
        )
        args = operands + tensor_grads
        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.

        branches_bw_gm: list[torch.fx.GraphModule] = []
        grads_tensor_masks: list[bool] = []
        # All branches share the same input signature (see _validate_input)
        for bw_fn in ctx._branch_bw_fns:
            wrapped_bw, mask = create_fn_remove_none(bw_fn)
            bw_gm = materialize_as_graph(
                wrapped_bw,
                args,
                ctx._fw_include_key_set,
                ctx._fw_exclude_key_set,
                force_enable_grad=True,
            )
            branches_bw_gm.append(bw_gm)
            if not grads_tensor_masks:
                grads_tensor_masks = mask

        grads = switch_op(
            ctx._index,
            branches_bw_gm,
            args,
        )
        return None, None, *fill_none_with_masks(grads, grads_tensor_masks)


@switch_op.py_autograd_impl
def switch_autograd(index, branches, operands):
    return SwitchAutogradOp.apply(
        index,
        branches,
        *operands,
    )


@switch_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, index, branches, operands):
    return trace_switch(mode, switch_op, index, branches, operands)


@switch_op.py_impl(FakeTensorMode)
def switch_fake_tensor_mode(mode, index, branches, operands):
    # Ignore here, because if you've gotten here but you're not manually
    # tracing the inner graphs, that means that you intend to reuse the graph
    # directly.  Which means the old unbacked symbol bindings are appropriate.
    # This strategy will not work if unbacked symbols can escape.
    ignore_fresh_unbacked = contextlib.nullcontext()
    if mode.shape_env:
        ignore_fresh_unbacked = mode.shape_env.ignore_fresh_unbacked_symbols()

    with mode, ignore_fresh_unbacked:
        flat_branch_outs, branch_out_spec = zip(
            *[pytree.tree_flatten(branch(*operands)) for branch in branches]
        )
        for i, spec in enumerate(branch_out_spec):
            if branch_out_spec[0] != spec:
                raise RuntimeError(
                    "Unmatched output spec from torch.switch branches: "
                    f"branch0 tree_spec {branch_out_spec[0]} vs branch{i} tree_spec {spec}"
                )

    merged_outs = []
    for branches_out in zip(*flat_branch_outs):
        merged_outs.append(_merge_output(branches_out, mode))
    return pytree.tree_unflatten(merged_outs, branch_out_spec[0])


def _merge_output(xs: tuple[torch.Tensor | int | None, ...], mode: FakeTensorMode):
    from torch._higher_order_ops.cond import (
        _merge_ints_to_symint,
        _merge_output as cond_merge_output,
    )

    # Shortcut if a branch produces None outputs; then all branches need to produce None
    if any(x is None for x in xs):
        if not all(x is None for x in xs):
            raise AssertionError(f"expected all leaves to be None, got {xs}")
        return None

    # In case all branches return an int, use an unbacked symbol as the merge result
    if all(type(x) is int for x in xs):
        return _merge_ints_to_symint(list(xs), mode)  # type: ignore[arg-type]

    return functools.reduce(lambda a, b: cond_merge_output(a, b, mode), xs)


@switch_op.py_functionalize_impl
def switch_func(ctx, index, branches, inputs):
    from torch._higher_order_ops.auto_functionalize import (
        can_auto_functionalize,
        do_auto_functionalize_v2,
    )
    from torch._higher_order_ops.utils import _check_alias_and_mutation, HopInstance

    hop_instance = HopInstance.create(switch_op, index, branches, inputs)
    if can_auto_functionalize(hop_instance) and hasattr(ctx, "mode"):
        return do_auto_functionalize_v2(
            ctx.mode,
            hop_instance,
            tuple(pytree.tree_flatten((index, branches, inputs))[0]),
            {},
        )

    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    unwrapped_index = ctx.unwrap_tensors(index)
    with ctx.redispatch_to_next():
        functional_branches = [
            ctx.functionalize(_maybe_run_with_interpreter(fn)) for fn in branches
        ]
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for i, branch in enumerate(branches):
            _check_alias_and_mutation(
                branch, unwrapped_inputs, f"switch_branch{i}", pre_dispatch
            )
        switch_return = switch_op(
            unwrapped_index, functional_branches, unwrapped_inputs
        )
        return ctx.wrap_tensors(switch_return)
<|endoftext|># mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import contextlib
import logging
import warnings
from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._C._functorch import (
    _add_batch_dim,
    get_unwrapped,
    is_batchedtensor,
    maybe_get_bdim,
)
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import (
    _maybe_run_with_interpreter,
    check_input_alias_and_mutation_return_outputs,
    create_bw_fn,
    create_fn_remove_none,
    fill_none_with_masks,
    materialize_as_graph,
    reenter_make_fx,
    save_values_for_backward,
    saved_values,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode


log = logging.getLogger(__name__)

"""
We're going to define a `cond_op` operation.
In order to do this, we need implementations for each of the dispatch keys.
"""


class CondOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("cond")

    def __call__(self, pred, true_fn, false_fn, operands):
        validate_subgraph_args_types(operands)
        # pyrefly: ignore [missing-attribute]
        return super().__call__(pred, true_fn, false_fn, operands)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, pred, true_fn, false_fn, operands):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import materialize_as_graph

        then_gm: torch.fx.GraphModule = materialize_as_graph(true_fn, operands)
        else_gm: torch.fx.GraphModule = materialize_as_graph(false_fn, operands)
        (
            _,
            _,
            _,
            then_mutated_inputs,
            then_outputs,
        ) = check_input_alias_and_mutation_return_outputs(then_gm)
        (
            _,
            _,
            _,
            else_mutated_inputs,
            else_outputs,
        ) = check_input_alias_and_mutation_return_outputs(else_gm)
        mutated_inputs = set(then_mutated_inputs) | set(else_mutated_inputs)

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("pred", pred)
        schema_gen.add_arg("true_fn", then_gm)
        schema_gen.add_arg("false_fn", else_gm)
        for idx, arg in enumerate(operands):
            schema_gen.add_arg(f"operand{idx}", arg, is_mutated=idx in mutated_inputs)

        for out in then_outputs:
            schema_gen.add_output(out)
        schema_gen.add_schema_tree_spec(pred, true_fn, false_fn, operands)
        return schema_gen.gen_schema()


cond_op = CondOp()


@exposed_in("torch")
def cond(
    pred: bool | int | float | torch.Tensor,
    true_fn: Callable,
    false_fn: Callable,
    operands: tuple | list = (),
) -> Any:
    r"""
    Conditionally applies `true_fn` or `false_fn`.

    .. warning::

        `torch.cond` is a prototype feature in PyTorch. It has limited support for input and output types.
        Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `cond` is structured control flow operator. That is, it is like a Python if-statement,
    but has restrictions on `true_fn`, `false_fn`, and `operands` that enable it to be
    capturable using torch.compile and torch.export.

    Assuming the constraints on `cond`'s arguments are met, `cond` is equivalent to the following::

        def cond(pred, true_branch, false_branch, operands):
            if pred:
                return true_branch(*operands)
            else:
                return false_branch(*operands)

    Args:
        pred (Union[bool, torch.Tensor]): A boolean expression or a tensor with one element,
          indicating which branch function to apply.

        true_fn (Callable): A callable function (a -> b) that is within the
          scope that is being traced.

        false_fn (Callable): A callable function (a -> b) that is within the
          scope that is being traced. The true branch and false branch must
          have consistent input and outputs, meaning the inputs have to be
          the same, and the outputs have to be the same type and shape. Int
          output is also allowed. We'll make the output dynamic by turning it
          into a symint.

        operands (Tuple of possibly nested dict/list/tuple of torch.Tensor): A tuple of inputs to the
          true/false functions. It can be empty if true_fn/false_fn doesn't require input. Defaults to ().

    Example::

        def true_fn(x: torch.Tensor):
            return x.cos()


        def false_fn(x: torch.Tensor):
            return x.sin()


        return cond(x.shape[0] > 4, true_fn, false_fn, (x,))

    Restrictions:
        - The conditional statement (aka `pred`) must meet one of the following constraints:

          - It's a `torch.Tensor` with only one element, and torch.bool dtype

          - It's a boolean expression, e.g. `x.shape[0] > 10` or `x.dim() > 1 and x.shape[1] > 10`

        - The branch function (aka `true_fn`/`false_fn`) must meet all of the following constraints:

          - The function signature must match with operands.

          - The function must return a tensor with the same metadata, e.g. shape,
            dtype, etc.

          - The function cannot have in-place mutations on global variables.
            (Note: in-place tensor operations such as `add_` for intermediate results
            are allowed in a branch)

          - The function can perform in-place mutations on its input tensors during inference (i.e.,
            when `torch.is_grad_enabled()` is False).
            Note: When using `torch.compile()` with a non-constant predicate, the outputs will always
            be new tensors that do not share object identity with the original inputs.

            Example::

                def true_fn(x):
                    return x.sin_()


                def false_fn(x):
                    return x + 1


                def f(x):
                    return cond(x.sum() > 0, true_fn, false_fn, (x,))


                x = torch.ones(4)
                with torch.no_grad():
                    result = torch.compile(f)(x)
                assert result is not x  # result is a new tensor, not the original x

    """
    if torch.compiler.is_dynamo_compiling():
        return cond_op(pred, true_fn, false_fn, operands)

    if isinstance(pred, (bool, int, float)):
        # This is the non-strict export case. Strict export and torch.compile are
        # handled above in dynamo.
        if torch.compiler.is_exporting():
            warnings.warn(
                "Pred is a Python constant. When used with torch.cond, it specializes on one of the branches."
                " If you want torch.cond to preserve two branches, please make the predicate a boolean tensor or a SymBool.",
                UserWarning,
                stacklevel=2,
            )
        # This is the eager case. We can just run the true or false branch.
        if pred:
            return true_fn(*operands)
        else:
            return false_fn(*operands)

    def _validate_input(pred, true_fn, false_fn, operands):
        if not isinstance(pred, (bool, torch.Tensor, torch.SymBool)):
            raise RuntimeError(f"Expected pred to be bool or tensor, but got {pred}.")

        if isinstance(pred, torch.Tensor) and pred.numel() != 1:
            raise RuntimeError(
                f"Expected pred to be bool or single-element tensor, but got {pred}."
            )

        if not callable(true_fn) or not callable(false_fn):
            raise RuntimeError("Expect both branches to be callable.")

        if not isinstance(operands, (tuple, list)) or pytree.tree_any(
            lambda t: not isinstance(t, torch.Tensor), operands
        ):
            raise RuntimeError(
                "Expect operands to be a tuple of possibly nested dict/list/tuple that only "
                f"consists of tensor leaves, but got {operands}."
            )

    _validate_input(pred, true_fn, false_fn, operands)

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("torch.cond requires dynamo support.")

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass cond_op to it. So we wrap it in a dummy function.
    def _cond_op_wrapper(*args, **kwargs):
        return cond_op(*args, **kwargs)

    from torch._higher_order_ops.utils import _hop_compile_and_call

    return _hop_compile_and_call(_cond_op_wrapper, (pred, true_fn, false_fn, operands))


def trace_cond(proxy_mode, func_overload, pred, true_fn, false_fn, operands):
    if not isinstance(operands, (list, tuple)):
        raise AssertionError(
            f"Cond operands must be a list or tuple of tensors and SymInts {operands}"
        )

    true_graph = reenter_make_fx(true_fn)(*operands)
    false_graph = reenter_make_fx(false_fn)(*operands)

    true_outs = []
    false_outs = []
    for node in true_graph.graph.nodes:
        if node.op == "output":
            true_outs.extend(node.args)

    for node in false_graph.graph.nodes:
        if node.op == "output":
            false_outs.extend(node.args)

    flat_true_outs = pytree.arg_tree_leaves(*true_outs)
    flat_false_outs = pytree.arg_tree_leaves(*false_outs)
    if len(flat_true_outs) != len(flat_false_outs):
        raise torch._dynamo.exc.CondOpArgsMismatchError(
            f"Expected to return same number of outputs but got:"
            f"\n  true branch returns {len(flat_true_outs)} item(s)"
            f"\n  false branch returns {len(flat_false_outs)} item(s)"
        )

    i, true_name = unique_graph_id(proxy_mode, prefix="true_graph")

    false_name = f"false_graph_{i}"
    if hasattr(proxy_mode.tracer.root, false_name):
        raise AssertionError(
            f"proxy_mode.tracer.root already has attribute {false_name}"
        )

    proxy_mode.tracer.root.register_module(true_name, true_graph)
    proxy_mode.tracer.root.register_module(false_name, false_graph)

    args = (pred, true_graph, false_graph, operands)

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}
    )

    out = func_overload(pred, true_graph, false_graph, operands)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@cond_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def cond_op_dense(pred, true_fn, false_fn, operands):
    if not all(isinstance(o, (torch.Tensor, int)) for o in operands):
        raise AssertionError(
            f"Dense implementation operands must be a list of tensors and ints {operands}"
        )
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    if pred:
        return true_fn(*operands)
    else:
        return false_fn(*operands)


class CondAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        pred,
        true_fn,
        false_fn,
        *operands,
    ):
        ctx._pred = pred
        ctx._true_bw_fn = create_bw_fn(
            true_fn,
            operands,
        )
        ctx._false_bw_fn = create_bw_fn(
            false_fn,
            operands,
        )
        # We snapshot the dispatch keys in forward for materializing
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()
        save_values_for_backward(ctx, operands)

        with torch._C._AutoDispatchBelowAutograd():
            return cond_op(pred, true_fn, false_fn, operands)

    @staticmethod
    def backward(ctx, *flat_grads):
        operands = saved_values(ctx)
        args = operands + flat_grads
        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.
        true_wrapped, grads_tensor_masks = create_fn_remove_none(ctx._true_bw_fn)
        true_bw_gm = materialize_as_graph(
            true_wrapped,
            args,
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )
        false_wrapped, _ = create_fn_remove_none(ctx._false_bw_fn)
        false_bw_gm = materialize_as_graph(
            false_wrapped,
            args,
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )
        grads = cond_op(
            ctx._pred,
            true_bw_gm,
            false_bw_gm,
            args,
        )
        return None, None, None, *fill_none_with_masks(grads, grads_tensor_masks)


# Note:
# As long as one of the tensors in pred or operands requires grad,
# all the output would require grad with backward fn set to be the CondAutogradOp.
# This is consistent with autograd.Function's semantic.
@cond_op.py_autograd_impl
def cond_autograd(pred, true_fn, false_fn, operands):
    return CondAutogradOp.apply(
        pred,
        true_fn,
        false_fn,
        *operands,
    )


@cond_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, pred, true_fn, false_fn, operands):
    return trace_cond(mode, cond_op, pred, true_fn, false_fn, operands)


@cond_op.py_impl(FakeTensorMode)
def cond_fake_tensor_mode(mode, pred, true_fn, false_fn, operands):
    # Ignore here, because if you've gotten here but you're not manually
    # tracing the inner graphs, that means that you intend to reuse the graph
    # directly.  Which means the old unbacked symbol bindings are appropriate.
    # This strategy will not work if unbacked symbols can escape.
    ignore_fresh_unbacked = contextlib.nullcontext()
    if mode.shape_env:
        ignore_fresh_unbacked = mode.shape_env.ignore_fresh_unbacked_symbols()

    with mode, ignore_fresh_unbacked:
        flat_true_outs, true_out_spec = pytree.tree_flatten(true_fn(*operands))
        flat_false_outs, false_out_spec = pytree.tree_flatten(false_fn(*operands))
        if true_out_spec != false_out_spec:
            raise RuntimeError(
                "Unmatched output spec from torch.cond branches: "
                f"true branch tree_spec {true_out_spec} vs false branch tree_spec {false_out_spec}."
            )

    merged_outs = []
    for true_out, false_out in zip(flat_true_outs, flat_false_outs):
        merged_outs.append(_merge_output(true_out, false_out, mode))
    return pytree.tree_unflatten(merged_outs, true_out_spec)


def check_tensor_meta_match(
    t1: torch.Tensor, t2: torch.Tensor, attr_names: tuple[str, ...], msg_prefix: str
) -> None:
    def _get_attr_maybe_call(t: torch.Tensor, attr_name: str) -> Any:
        attr = getattr(t, attr_name)
        if callable(attr):
            return attr()
        return attr

    for attr_name in attr_names:
        lattr = _get_attr_maybe_call(t1, attr_name)
        rattr = _get_attr_maybe_call(t2, attr_name)
        torch._check(
            lattr == rattr,
            lambda: f"{msg_prefix} expected same {attr_name} but got {lattr} and {rattr}.",
        )


def _merge_ints_to_symint(
    values: list[int], mode: FakeTensorMode
) -> int | torch.SymInt:
    """Merge N concrete ints into a single unbacked SymInt bounded [min, max].

    Returns the value unchanged when all values are equal.
    """
    if all(v == values[0] for v in values):
        return values[0]
    if mode.shape_env is None:
        raise AssertionError("mode.shape_env is None")
    merged = mode.shape_env.create_unbacked_symint()
    mode.shape_env.constrain_symbol_range(merged.node.expr, min(values), max(values))
    return merged


def _merge_output(
    a: torch.Tensor | int | None,
    b: torch.Tensor | int | None,
    mode: FakeTensorMode,
):
    from torch.fx.experimental.symbolic_shapes import (
        has_free_unbacked_symbols,
        SymIntEqByExpr,
    )

    if a is None or b is None:
        if not (a is None and b is None):
            raise AssertionError(f"expected both a and b to be None, got a={a}, b={b}")
        return None

    def min_max(s0, s1):
        def _bound(s0, lower_bound: bool):
            if isinstance(s0, int):
                return s0
            r = mode.shape_env.var_to_range.get(  # type: ignore[union-attr]
                s0.node.expr,
                torch.utils._sympy.value_ranges.ValueRanges.unknown(),
            )
            return r.lower if lower_bound else r.upper

        return min(_bound(s0, True), _bound(s1, True)), max(
            _bound(s0, False), _bound(s1, False)
        )

    if type(a) is int and type(b) is int:
        return _merge_ints_to_symint([a, b], mode)

    if not (type(a) is FakeTensor and type(b) is FakeTensor):
        raise AssertionError(
            f"expected both a and b to be FakeTensor, got a={type(a)}, b={type(b)}"
        )

    # Note: we don't check size, stride because
    # they'll be merged with unbacked symints if they differ.
    _meta_to_check = {
        "dtype",
        "device",
        "layout",
        "dim",
        "is_quantized",
        "is_conj",
        "is_sparse",
        "storage_offset",
    }
    check_tensor_meta_match(
        a,
        b,
        tuple(_meta_to_check),
        msg_prefix="When merging two branches' output in torch.cond, ",
    )
    # NYI
    if a.is_quantized or b.is_quantized:
        raise AssertionError("quantized tensors not yet implemented")
    if a.is_sparse or b.is_sparse:
        raise AssertionError("sparse tensors not yet implemented")
    if a.is_conj() or b.is_conj():
        raise AssertionError("conjugate tensors not yet implemented")

    """
    Step 1: create unbacked symints for sizes that are different
    along the same axis. For example:
        a.size is [s0, 4, s0, 5, 4, 5]
        b.size is [s1, 4, s2, 8, 4, 7]
        merged_size will be [u0, 4, u1, u2, 4, u3], where
        u0 has range [min(s0, s1), max(s0, s1)]
        u1 has range [min(s0, s2), max(s0, s2)]
        u2 has range [5, 8]
        u3 has range [5, 7]
    """
    merged_size: list[int | torch.SymInt] = []

    def _has_unbacked_symbols(s: int | torch.SymInt) -> bool:
        if isinstance(s, int):
            return False
        else:
            return has_free_unbacked_symbols(s.node.expr)

    for s0, s1 in zip(a.size(), b.size()):
        # If there are unbacked symbols leaked out of true_branch or false_branch
        # we need to merge them with a new unbacked symbol and track in parent graph.
        if (
            not _has_unbacked_symbols(s0)
            and not _has_unbacked_symbols(s1)
            and SymIntEqByExpr(s0) == SymIntEqByExpr(s1)
        ):
            merged_size.append(s0)
        else:
            if mode.shape_env is None:
                raise AssertionError("mode.shape_env is None")
            new_size = mode.shape_env.create_unbacked_symint()
            mode.shape_env.constrain_symbol_range(new_size.node.expr, *min_max(s0, s1))
            merged_size.append(new_size)

    """
    This follows the logic in symbolic_shapes._compute_symbolic_stride
    Step 2: Since tensor stride is an accumulative multiplication of the sizes, which is a permutated
        (due to view ops) non-descending sequence.

        Case 1: No size is 1. In this case, strides have unique values.
            For example, suppose we have a tensor with:
            size [3, 4, 3, 5, 4, 5],
            stride (1200, 300, 1, 12, 3, 60),
            merged_size [u0, u1, u2, u3, u4, u5].

            We visit the strides in ascending order: 1, 3, 12, 60, 300, 1200. In each step, we check whether
            the current stride is bounded or not and bound next stride by setting.
                stride_expr[next_stride] = current_stride_expr * current_size_expr
            1st round:
                current_stride is 1, current_size is 3, so next_stride is 1 * 3 = 3,
                current_stride_expr is set to 1, current_size_expr is u2, so stride_expr[3] is therefore 1 * u2 = u2
            2nd round:
                current_stride is 3, current_size is 4, so next_stride is 3 * 4 = 12,
                current_stride_expr is stride_expr[3] i.e. u2, current_size_expr is u4, so stride_expr[12] = u2 * u4
                ...

        Case 2: At least one dimension has size 1, which can produce duplicates in strides.
            In this case, theoretically, we cannot uniquely determine the expr of strides because
            the accessing stride_expr with same key in different order causes the final stride expression
            to be different.

            Suppose we have:
                size: (3, 1)
                stride: (1, 1)
                merged_size: (u0, u1)

            The stride expr could either be (u1, 1) or (1, u0) depending on whether we start with u1 or u0.
            For this reason, we try to break tie by sorting via descending index so we always get (u1, 1).

            Note that backend might optimize the strides anyway so this is usually not a problem as long
            as two branches matches. See relevant discussions in https://github.com/pytorch/pytorch/issues/142024.

        Case 3: Dim has 0 stride. 0 stride doesn't participate in the accumulative multiplication of
            sizes. So they're always treated as constant even if their corresponding size is turned into unbacked symint.

            Suppose we have:
                size: (3, 3)
                stride: (0, 1)
                merged_size: (u0, u1)

            The merged stride would be (0, 1)
    """

    def _bound_stride(
        a_ex_size: torch.Size,
        b_ex_size: torch.Size,
        a_ex_stride: tuple[int, ...],
        b_ex_stride: tuple[int, ...],
        merged_size: list[int | torch.SymInt],
    ) -> list[int | torch.SymInt]:
        from torch._inductor.ir import get_stride_order

        a_sorted_stride_idx = get_stride_order(a_ex_stride, mode.shape_env)
        b_sorted_stride_idx = get_stride_order(b_ex_stride, mode.shape_env)

        a_stride_li: list[tuple[int | torch.SymInt, int] | None] = [None] * len(
            a_ex_stride
        )
        b_stride_li: list[tuple[int | torch.SymInt, int] | None] = [None] * len(
            b_ex_stride
        )
        for i, idx in enumerate(a_sorted_stride_idx):
            a_stride_li[idx] = (a_ex_stride[i], -i)
        for i, idx in enumerate(b_sorted_stride_idx):
            b_stride_li[idx] = (b_ex_stride[i], -i)

        for a_pair, b_pair in zip(a_stride_li, b_stride_li):
            if a_pair is None or b_pair is None:
                raise AssertionError(
                    f"expected a_pair and b_pair to be non-None, got a_pair={a_pair}, b_pair={b_pair}"
                )
            _, a_idx = a_pair
            _, b_idx = b_pair

            if a_idx != b_idx:
                raise RuntimeError(
                    f"The sorted order of strides of the two branches' output doesn't match."
                    f"this indicates the contiguousness of the two branches are different. "
                    f"True branch has stride {a_ex_stride} but false branch has stride {b_ex_stride}."
                    f"Consider using contiguous() to make the two branches have the same contiguousness."
                )

        def _maybe_expr(s: int | torch.SymInt):
            if isinstance(s, int):
                return s
            return s.node.expr

        a_stride_expr: dict[Any, int | torch.SymInt] = {}
        b_stride_expr: dict[Any, int | torch.SymInt] = {}
        merged_strides: list[int | torch.SymInt] = [None] * len(a_ex_stride)  # type: ignore[list-item]
        for a_pair, b_pair in zip(a_stride_li, b_stride_li):
            if a_pair is None or b_pair is None:
                raise AssertionError(
                    f"expected a_pair and b_pair to be non-None, got a_pair={a_pair}, b_pair={b_pair}"
                )
            a_val, neg_i = a_pair
            b_val, _ = b_pair

            i = -neg_i
            if a_val == 0:
                if b_val != 0:
                    raise AssertionError(
                        f"expected b_val == 0 when a_val == 0, got a_val={a_val}, b_val={b_val}"
                    )
                merged_strides[i] = 0
                continue

            if _maybe_expr(a_val) in a_stride_expr:
                a_expr = a_stride_expr[_maybe_expr(a_val)]
                if b_stride_expr[_maybe_expr(b_val)] != a_expr:
                    raise AssertionError(
                        f"a_stride_expr:{a_stride_expr}, b_stride_expr:{b_stride_expr}"
                    )
                merged_strides[i] = a_expr
            else:
                if a_val == 1:
                    if b_val != 1:
                        raise AssertionError(
                            f"expected b_val == 1 when a_val == 1, got b_val={b_val}"
                        )
                    a_stride_expr[_maybe_expr(a_val)] = 1
                    b_stride_expr[_maybe_expr(b_val)] = 1
                    merged_strides[i] = 1
                else:
                    # If we cannot find the expr of a_val in a_stride_expr, it means
                    # the strides is not a simple accumulative multiplication of sizes.
                    # In this case, we cannot determine the expr of strides from the new
                    # shapes so we error out and hint users to call contiguous().
                    raise RuntimeError(
                        f"It seems one of cond's output stride is not a simple accumulative multiplication of sizes. "
                        f"This could be because cond returns a slice of a tensor, which is not dense in memory. "
                        f"True branch has size {a_ex_size}, stride {a_ex_stride} and false branch has size {b_ex_size} "
                        f"stride {b_ex_stride}. Hint: can call t.contiguous(). "
                    )
            nxt_merged_stride_expr = merged_strides[i] * merged_size[i]
            a_stride_expr[_maybe_expr(a_val * a_ex_size[i])] = nxt_merged_stride_expr
            b_stride_expr[_maybe_expr(b_val * b_ex_size[i])] = nxt_merged_stride_expr
            # fake tensors accumulate contiguous strides as stride * max(size, 1)
            a_max_key = _maybe_expr(a_val * torch.sym_max(a_ex_size[i], 1))
            b_max_key = _maybe_expr(b_val * torch.sym_max(b_ex_size[i], 1))
            a_stride_expr.setdefault(a_max_key, nxt_merged_stride_expr)
            b_stride_expr.setdefault(b_max_key, nxt_merged_stride_expr)
        return merged_strides

    merged_stride: list[int | torch.SymInt] = _bound_stride(
        a.size(), b.size(), a.stride(), b.stride(), merged_size
    )

    with mode:
        return torch.empty_strided(
            merged_size, merged_stride, dtype=a.dtype, device=a.device
        )


@cond_op.py_functionalize_impl
def cond_func(ctx, pred, true_fn, false_fn, inputs):
    from torch._higher_order_ops.auto_functionalize import (
        can_auto_functionalize,
        do_auto_functionalize_v2,
    )
    from torch._higher_order_ops.utils import _check_alias_and_mutation, HopInstance

    hop_instance = HopInstance.create(cond_op, pred, true_fn, false_fn, inputs)
    # For now, we only support auto-functionalization for cond when using python
    # functionalization mode
    if can_auto_functionalize(hop_instance) and hasattr(ctx, "mode"):
        return do_auto_functionalize_v2(
            ctx.mode,
            hop_instance,
            tuple(pytree.tree_flatten((pred, true_fn, false_fn, inputs))[0]),
            {},
        )

    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    unwrapped_pred = ctx.unwrap_tensors(pred)
    with ctx.redispatch_to_next():
        functional_true = ctx.functionalize(_maybe_run_with_interpreter(true_fn))
        functional_false = ctx.functionalize(_maybe_run_with_interpreter(false_fn))
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for branch, branch_name in [(true_fn, "cond_true"), (false_fn, "cond_false")]:
            _check_alias_and_mutation(
                branch, unwrapped_inputs, branch_name, pre_dispatch
            )

        cond_return = cond_op(
            unwrapped_pred, functional_true, functional_false, unwrapped_inputs
        )
        return ctx.wrap_tensors(cond_return)


@cond_op.py_impl(torch._C._functorch.TransformType.Vmap)
def cond_batch_rule(interpreter, pred, true_fn, false_fn, inputs):
    if not isinstance(inputs, (list, tuple)):
        raise AssertionError(
            f"Cond inputs must be a list or tuple of tensors, got {type(inputs)}"
        )
    if not all(isinstance(i, torch.Tensor) for i in inputs):
        raise AssertionError(
            f"Cond inputs must be a list of tensors, got {[type(i) for i in inputs]}"
        )

    pred_is_batched = isinstance(pred, torch.Tensor) and is_batchedtensor(pred)
    pred_ = get_unwrapped(pred) if pred_is_batched else pred

    # unbatched tensors are not vmapped
    tensors, in_dims = zip(
        *[
            (get_unwrapped(t), maybe_get_bdim(t)) if is_batchedtensor(t) else (t, None)
            for t in inputs
        ]
    )

    if pred_is_batched:
        # prepend "pred" and vmap everything
        tensors = (pred_,) + tensors
        in_dims = (0,) + in_dims

        def fn(p, *args):
            t = true_fn(*args)
            f = false_fn(*args)
            return torch.where(p, t[0], f[0])

        with interpreter.lower():
            result = torch.vmap(fn, in_dims=in_dims)(*tensors)

    else:
        # predicate is known at this stage and it is a boolean expression or a
        # tensor with one element.
        true_fn = torch.vmap(true_fn, in_dims=in_dims)
        false_fn = torch.vmap(false_fn, in_dims=in_dims)

        with interpreter.lower():
            result = cond_op(pred, true_fn, false_fn, tensors)

    if not isinstance(result, tuple):
        result = (result,)
    lvl = interpreter.level()
    return tuple(_add_batch_dim(r, 0, lvl) for r in result)
<|endoftext|>import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_custom_class
from torch.fx.node import Target


# Below is an implementation of generating FunctionSchema from example values.
# This is helpful for generating FunctionSchema for HigherOrderOperator, where
# we don't have a function to inspect and each call of the higher order operator
# would have different schema.
@dataclass(frozen=True, slots=True)
class HopArgumentInfo:
    # Could give a name to the operand by default it's empty string.
    name: str
    example_value: Any
    # Provide a default_value
    default_value: Any
    # Whether this argument gets mutated in the hop subgraph.
    # For output, this should always be False
    is_mutated: bool
    kw_only: bool


class HopArgumentInfoGen:
    @staticmethod
    def from_example(
        example_value: Any,
        *,
        name: str = "",
        default_value: Any | None = None,
        is_mutated: bool = False,
        kw_only: bool = False,
    ) -> HopArgumentInfo:
        if default_value is not None:
            if type(example_value) is not type(default_value):
                raise AssertionError(
                    f"example_value type {type(example_value)} doesn't match default_value type: {type(default_value)}"
                )

        return HopArgumentInfo(
            name=name,
            example_value=example_value,
            default_value=default_value,
            is_mutated=is_mutated,
            kw_only=kw_only,
        )


class CTypeGen:
    convert_to_base_ty = {
        int: torch._C.IntType.get(),
        float: torch._C.FloatType.get(),
        str: torch._C.StringType.get(),
        bool: torch._C.BoolType.get(),
    }

    # should return torch._C.JitType but that annotation is busted
    @staticmethod
    def from_example(obj: Any) -> Any:
        import torch

        if isinstance(obj, torch.fx.GraphModule):
            return torch._C.AnyType.get()
        elif isinstance(obj, torch.SymInt):
            return torch._C.SymIntType.get()
        elif isinstance(obj, torch.SymBool):
            return torch._C.SymBoolType.get()
        elif isinstance(obj, torch.SymFloat):
            return torch._C.FloatType.get()
        elif isinstance(obj, (FakeScriptObject, pytree.TreeSpec)) or is_custom_class(
            type(obj)
        ):
            return torch._C.PyObjectType.get()  # pyrefly: ignore[missing-attribute]
        return torch._C._jit_try_infer_type(obj).type()


class CArgumentGen:
    @staticmethod
    def from_hop_argument_info(
        arg_idx: int, arg_info: HopArgumentInfo, is_output: bool = False
    ) -> Any:
        typ = CTypeGen.from_example(arg_info.example_value)
        if is_output:
            return torch._C.Argument("", typ, None, None, False, None)

        alias_set = set({f"alias::a{arg_idx}"}) if arg_info.is_mutated else set()
        alias_info = torch._C._AliasInfo(arg_info.is_mutated, alias_set, alias_set)  # type: ignore[attr-defined]
        return torch._C.Argument(
            arg_info.name,
            typ,
            None,
            arg_info.default_value,
            arg_info.kw_only,
            alias_info,
        )


class HopSchemaGenerator:
    def __init__(self, hop: torch._ops.HigherOrderOperator):
        self.arg_infos: list[HopArgumentInfo] = []
        self.example_outputs: list[Any] = []
        self.schema_tree_spec: pytree.TreeSpec | None = None
        self.hop = hop

    def add_arg(
        self,
        name: str,
        example_value: Any,
        default_value: Any | None = None,
        is_mutated: bool = False,
        kw_only: bool = False,
    ) -> None:
        if callable(example_value) and not is_custom_class(type(example_value)):
            if not isinstance(
                example_value, (torch.fx.GraphModule, torch._ops.OperatorBase)
            ):
                raise AssertionError(
                    "Expect callable to be a GraphModule or an OperatorBase. Please call materialize_as_graph first "
                    f"to turn callable arguments {example_value} into a GraphModule."
                )
        _, flat_spec = pytree.tree_flatten(example_value)
        if not flat_spec.is_leaf():
            raise RuntimeError(
                f"example_value {example_value} is not a leaf node. "
                "Please only add flattened inputs to the hop schema. "
                "If you need some structure in the arguments, please"
                "add_arg for flattened args one by one then "
                "call add_schema_tree_spec to register the original pytree "
                " spec of the args."
            )

        arg_info = HopArgumentInfoGen.from_example(
            example_value=example_value,
            name=name,
            default_value=default_value,
            is_mutated=is_mutated,
            kw_only=kw_only,
        )
        self.arg_infos.append(arg_info)

    def add_output(self, output: Any) -> None:
        self.example_outputs.append(output)

    def add_schema_tree_spec(self, *args: Any, **kwargs: Any) -> None:
        """schema tree spec is the tree spec from flattening all inputs to the hop with pytree.tree_flatten
        Since torch.FunctionSchema only have proper mutation/alias support for flattened inputs, we need
        to store the tree spec in order to reconstruct the inputs to the hop.
        """
        self.schema_tree_spec = pytree.tree_flatten((args, kwargs))[1]

    def gen_schema(self) -> torch._C.FunctionSchema:
        for i, arg_info in enumerate(self.arg_infos):
            arg_spec = pytree.tree_flatten(arg_info.example_value)[1]
            if not arg_spec.is_leaf() and self.schema_tree_spec is None:
                raise RuntimeError(
                    f"example_value of arg_infos[{i}] is {arg_info.example_value}, which is not a leaf node. "
                    "Please call add_schema_tree_spec to add a schema tree spec first. "
                    "Or consider changing the hop's signature to only take flattened arguments."
                )

        return CFunctionSchemaGen.from_hop_argument_info(
            str(self.hop),
            self.arg_infos,
            HopArgumentInfoGen.from_example(tuple(self.example_outputs), name="out"),
            self.schema_tree_spec,
        )


class CFunctionSchemaGen:
    """
    Note: [HigherOrderOperator schema generation]
    Each invocation of a HigherOrderOperator will have a different schema.
    For example, the schema of torch.cond varies depending on the true_fn and
    false_fn. So we need a way to generate the schema for each invocation of a HOP.

    We want to enforce the following invariants for HOP's schema:
        1. Flattened inputs. There should be no pytree structure in it.
        2. Flattened outputs. Note even if the hop returns a single value, it should be wrapped as a tuple.
        3. No aliasing. This includes inp-inp aliasing, inp-out aliasing and out-out aliasing.

    By enforcing these invariants, we could make HOP's schema meets the requirement of schema parser
    and makes hop easier to handle downstream. For example, suppose we have an invoke_quant_test HOP:

    class GraphModule(torch.nn.Module):
        def forward(self, l_x_, l_y_):
            subgraph_0 = self.subgraph_0
            invoke_quant_test = torch.ops.higher_order.invoke_quant_test(subgraph_0, l_x_, l_y_, scheme = 'nf4');

        class subgraph_0(torch.nn.Module):
            def forward(self, l_x_, l_y_):
                add_ = l_x_.add_(1)
                matmul = l_x_ @ l_y_
                sin = matmul.sin()
                child = sin.cos()
                child_1 = l_x_ + l_y_
                child_2 = l_x_ - l_y_
                child_3 = l_x_ @ l_y_
                return (child, child_1, child_2, child_3)

    By encoding the inputs of hop into a list of HopArgumentInfo and output as a single HopArgumentInfo,
    we would get the following schema:
        invoke_quant_test(Any arg0, Tensor(!) arg1, Tensor arg2, str scheme="\\"nf4\\"") -> (Tensor, Tensor, Tensor, Tensor)
    """

    @staticmethod
    def from_hop_argument_info(
        op_name: str,
        inp_argument_info: list[HopArgumentInfo],
        out_argument_info: HopArgumentInfo,
        schema_tree_spec: pytree.TreeSpec | None,
    ) -> Any:
        args = []
        for i, arg_info in enumerate(inp_argument_info):
            args.append(CArgumentGen.from_hop_argument_info(i, arg_info))

        # NOTE: we want the output to always be a single argument with torch._C.TupleType.
        if not isinstance(out_argument_info.example_value, tuple):
            raise AssertionError(
                f"expect out_argument_info's example_value to be a tuple but got {out_argument_info.example_value}"
            )
        if out_argument_info.is_mutated:
            raise AssertionError(
                "out_argument_info.is_mutated should always be set to False."
            )
        rets = None
        if len(out_argument_info.example_value) == 1:
            rets = [CArgumentGen.from_hop_argument_info(0, out_argument_info, True)]
        else:
            rets = [
                CArgumentGen.from_hop_argument_info(
                    i,
                    HopArgumentInfoGen.from_example(
                        name=f"out{i}",
                        example_value=val,
                        default_value=None,
                        is_mutated=False,
                    ),
                    is_output=True,
                )
                for i, val in enumerate(out_argument_info.example_value)
            ]

        return HopSchema(
            op_name,
            "",
            args,
            rets,
            False,
            False,
            schema_tree_spec,
        )


class HopSchema(torch._C.FunctionSchema):
    def __init__(
        self,
        name: str,
        overload_name: str,
        arguments: list[torch._C.Argument],
        returns: list[torch._C.Argument],
        is_vararg: bool,
        is_varret: bool,
        schema_tree_spec: pytree.TreeSpec | None,
    ):
        self.tree_spec = schema_tree_spec
        self.is_vararg = is_vararg
        self.is_varret = is_varret
        super().__init__(
            name,
            overload_name,
            arguments,
            returns,
            self.is_vararg,
            self.is_varret,
        )

    def __deepcopy__(self, memo: Any) -> "HopSchema":
        # Need to additionally copy the tree_spec since
        # it's not a member of torch._C.FunctionSchema
        return HopSchema(
            self.name,
            self.overload_name,
            self.arguments,
            self.returns,
            self.is_vararg,
            self.is_varret,
            copy.deepcopy(self.tree_spec),
        )


def find_hop_schema(
    gm: torch.fx.GraphModule, target: Target
) -> list[torch._C.FunctionSchema]:
    schemas = []
    for node in gm.graph.find_nodes(op="call_function", target=target):

        def _get_example_value(node: torch.fx.Node) -> Any:
            if node.op == "get_attr":
                if not isinstance(node.target, str):
                    raise AssertionError(
                        f"expected node.target to be str for get_attr, got {type(node.target)}"
                    )
                return getattr(gm, node.target)
            else:
                return (
                    node.meta["example_value"]
                    if "example_value" in node.meta
                    else node.meta["val"]
                )

        fake_args, fake_kwargs = pytree.tree_map_only(
            torch.fx.Node,
            _get_example_value,
            (node.args, node.kwargs),
        )
        schema = node.target.gen_schema(*fake_args, **fake_kwargs)
        schemas.append(schema)
    return schemas
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import functools
from collections.abc import Callable

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.auto_functionalize import (
    can_auto_functionalize,
    do_auto_functionalize_v2,
)
from torch._higher_order_ops.utils import (
    _check_alias_and_mutation,
    _maybe_run_with_interpreter,
    autograd_not_implemented,
    check_meta_consistency,
    fill_none_with_masks,
    filter_with_masks,
    get_graph_output_example_values,
    HopInstance,
    materialize_as_graph,
    reenter_make_fx,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)


class WhileLoopOp(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("while_loop")

    def __call__(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: tuple[torch.Tensor | int | float | bool],
        additional_inputs: tuple[torch.Tensor | torch.SymInt | int, ...],
        /,
        *,
        mutated_arg_indices: str = "",
    ):
        if not isinstance(carried_inputs, (tuple, list)):
            raise RuntimeError(
                f"carried_inputs must be a tuple or list, got {type(carried_inputs)}"
            )
        if not isinstance(additional_inputs, (tuple, list)):
            raise RuntimeError(
                f"additional_inputs must be a tuple or list, got {type(additional_inputs)}"
            )

        validate_subgraph_args_types(carried_inputs)
        validate_subgraph_args_types(additional_inputs)
        kwargs = {}
        if mutated_arg_indices:
            kwargs["mutated_arg_indices"] = mutated_arg_indices
        # pyrefly: ignore [missing-attribute]
        return super().__call__(
            cond_fn,
            body_fn,
            carried_inputs,
            additional_inputs,
            **kwargs,
        )

    # pyrefly: ignore [bad-override]
    def gen_schema(
        self,
        cond_fn,
        body_fn,
        carried_inputs,
        additional_inputs,
        mutated_arg_indices="",
    ):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import materialize_as_graph

        all_inputs = carried_inputs + additional_inputs

        cond_gm: torch.fx.GraphModule = (
            cond_fn
            if isinstance(cond_fn, torch.fx.GraphModule)
            else materialize_as_graph(cond_fn, all_inputs)
        )
        body_gm: torch.fx.GraphModule = (
            body_fn
            if isinstance(body_fn, torch.fx.GraphModule)
            else materialize_as_graph(body_fn, all_inputs)
        )

        body_outputs = get_graph_output_example_values(body_gm)
        mutated_inputs = (
            {int(i) for i in mutated_arg_indices.split(",") if i}
            if mutated_arg_indices
            else set()
        )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("cond_fn", cond_gm)
        schema_gen.add_arg("body_fn", body_gm)

        for idx, arg in enumerate(carried_inputs):
            schema_gen.add_arg(
                f"carried_input{idx}", arg, is_mutated=idx in mutated_inputs
            )

        for idx, arg in enumerate(additional_inputs):
            additional_idx = len(carried_inputs) + idx
            schema_gen.add_arg(
                f"additional_input{idx}",
                arg,
                is_mutated=additional_idx in mutated_inputs,
            )

        for out in body_outputs:
            schema_gen.add_output(out)

        schema_gen.add_schema_tree_spec(
            cond_fn,
            body_fn,
            carried_inputs,
            additional_inputs,
        )
        return schema_gen.gen_schema()


while_loop_op = WhileLoopOp()


def while_loop(cond_fn, body_fn, carried_inputs):
    r"""
    Run ``body_fn(*carried_inputs)`` while ``cond_fn(*carried_inputs)`` returns
    a True scalar tensor. Returns the output of body_fn or initial
    carried_inputs.

    .. warning::

        `torch.while_loop` is a prototype feature in PyTorch. It has limited support for input and output types and
        doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `while_loop` is a structured control flow operator. It preserves the loop semantic across the torch.compile and torch.export.

    `while_loop` is equivalent to the following::

        def while_loop(cond_fn, body_fn, carried_inputs):
            val = carried_inputs
            while cond_fn(*val):
                val = body_fn(*val)
            return val

    Args:
        cond_fn (Callable): A callable function that returns a boolean Scalar tensor or a python boolean.

        body_fn (Callable): A callable function that takes the same inputs as `cond_fn` and returns a tuple of tensors or ints

        carried_inputs (Tuple of possibly nested dict/list/tuple of tensors or ints): A tuple of inputs to cond_fn and body_fn.
            It's also the initial value of states that are carried across iterations. Note that when pass an integer as carry,
            the corresponding return of while_loop will be another int with unknown values because we don't know how many
            iterations while_loop will run.

    Example 1::

        def cond_fn(iter, x):
            return iter.sum() < 10


        def body_fn(iter, x):
            return iter + 1, x.sin()


        while_loop(cond_fn, body_fn, (torch.zeros(1), torch.randn(3, 4)))

    Example 2::

        def cond_fn(int_iter, x):
            return 2 * int_iter < x.shape[0]


        def body_fn(int_iter, x):
            return int_iter + 1, x + int_iter


        while_loop(cond_fn, body_fn, (0, torch.randn(3, 4)))

    Restrictions:

        - body_fn must return tensors or int with the same metadata (e.g.shape, dtype) as inputs.

        - body_fn and cond_fn must not in-place mutate the carried_inputs. A clone before the mutation is required.

        - body_fn and cond_fn must not mutate python variables (e.g. list/dict) created outside of the body_fn.

        - body_fn and cond_fn's output cannot alias any of the inputs. A clone is required.

        - During inference, body_fn and cond_fn can in-place mutate tensors that are not
          carried_inputs, such as module buffers and captured tensors from the enclosing scope.

    """

    # Currently, additional_inputs is not a user-facing input. It will be automatically set in dynamo.
    # parameters and buffers accessed in cond_fn or body_fn or tensor closures will become additional_inputs.
    additional_inputs: tuple = ()

    # The reason we flatten the output before calling into dynamo is that
    # we want to create a consistent input ordering for cond_fn and body_fn.
    # and we also want to the input ordering matches the output ordering.
    # Also see NOTE: [why we cannot use "automatic" for while_loop]
    # Construct flat cond_fn and flat_body_fn, which takes flattened inputs
    flat_inputs, in_spec = pytree.tree_flatten((carried_inputs, additional_inputs))

    def flat_cond_fn(*flat_args):
        carried, additional = pytree.tree_unflatten(flat_args, in_spec)
        return cond_fn(*carried, *additional)

    def flat_body_fn(*flat_args):
        carried, additional = pytree.tree_unflatten(flat_args, in_spec)
        return body_fn(*carried, *additional)

    if torch.compiler.is_dynamo_compiling():
        return while_loop_op(flat_cond_fn, flat_body_fn, tuple(flat_inputs), tuple())

    def _validate_input(cond_fn, body_fn, carried_inputs):
        from torch._higher_order_ops.utils import validate_subgraph_args_types

        if not callable(cond_fn) or not callable(body_fn):
            raise RuntimeError("Expect cond_fn and body_fn to be callable.")

        validate_subgraph_args_types(flat_inputs)

        if not pytree.tree_all(
            lambda t: isinstance(t, (torch.Tensor, torch.SymInt, int)), carried_inputs
        ):
            raise RuntimeError(
                "Expect carried_inputs to be a tuple of possibly nested dict/list/tuple that only"
                f"consists of tensor or int leaves, but got {carried_inputs}."
            )

    _validate_input(cond_fn, body_fn, carried_inputs)

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass cond_op to it. So we wrap it in a dummy function.
    def _while_loop_op_wrapper(*args, **kwargs):
        return while_loop_op(*args, **kwargs)

    from torch._higher_order_ops.utils import _hop_compile_and_call

    return _hop_compile_and_call(
        _while_loop_op_wrapper,
        (flat_cond_fn, flat_body_fn, tuple(flat_inputs), tuple()),
    )


@while_loop_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def while_loop_dense(
    cond_fn,
    body_fn,
    carried_inputs,
    additional_inputs,
    stack_output=False,
    mutated_arg_indices="",
):
    carried_vals = carried_inputs

    def _validate_cond_output(pred):
        if (
            isinstance(pred, torch.Tensor)
            and pred.size() == torch.Size([])
            and pred.dtype == torch.bool
        ) or isinstance(pred, bool):
            return
        else:
            raise RuntimeError(
                f"cond_fn must return a boolean scalar tensor or a boolean but got {pred}"
            )

    if not isinstance(carried_inputs, (tuple, list)):
        raise RuntimeError(
            f"carried_inputs must be a tuple or list but got {type(carried_inputs)}"
        )

    # Check condition and set up flag
    should_loop = cond_fn(*carried_vals, *additional_inputs)
    _validate_cond_output(should_loop)

    if not should_loop:
        if stack_output:
            return tuple(
                val.unsqueeze(0).clone() if isinstance(val, torch.Tensor) else val
                for val in carried_vals
            )
        else:
            return tuple(
                val.clone() if isinstance(val, torch.Tensor) else val
                for val in carried_vals
            )

    outputs: list[list[torch.Tensor]] = [[] for _ in carried_vals]

    while should_loop:
        out = body_fn(*carried_vals, *additional_inputs)
        if stack_output:
            for i, o in enumerate(out):
                outputs[i].append(o)

        if not isinstance(out, tuple):
            raise AssertionError(f"body_fn should return a tuple but got {type(out)}")
        if len(out) != len(carried_inputs):
            raise AssertionError(
                f"body_fn should return the same number of elements as carried_inputs, got {len(out)} vs {len(carried_inputs)}"
            )
        carried_vals = out

        should_loop = cond_fn(*carried_vals, *additional_inputs)

    if stack_output:
        outs: list[torch.Tensor] = []
        for out in outputs:
            outs.append(torch.stack(out, dim=0))
        return tuple(outs)

    return carried_vals


@while_loop_op.py_autograd_impl
def while_loop_autograd(
    cond_fn, body_fn, operands, additional_inputs, mutated_arg_indices=""
):
    return WhileLoopAutogradOp.apply(
        cond_fn,
        body_fn,
        len(operands),
        len(additional_inputs),
        *operands,
        *additional_inputs,
    )


def _find_or_create_fake_mode() -> FakeTensorMode:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    fake_mode = torch._guards.detect_fake_mode()
    if fake_mode is None:
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())

    return fake_mode


def _create_unbacked_symint(
    fake_mode: FakeTensorMode, ignore_fresh_unbacked_symbols: bool
) -> torch.SymInt:
    if fake_mode is None or fake_mode.shape_env is None:
        raise AssertionError("Must provide a fake_mode with shape_env.")
    ctx = (
        contextlib.nullcontext()
        if not ignore_fresh_unbacked_symbols
        else fake_mode.shape_env.ignore_fresh_unbacked_symbols()
    )
    with ctx:
        return fake_mode.shape_env.create_unbacked_symint()


@while_loop_op.py_impl(ProxyTorchDispatchMode)
def while_loop_tracing(
    mode,
    cond_fn,
    body_fn,
    carried_inputs,
    additional_inputs,
    stack_output=False,
    mutated_arg_indices="",
):
    op = while_loop_stack_output_op if stack_output else while_loop_op

    def _trace_while_loop(
        proxy_mode,
        op,
        cond_fn,
        body_fn,
        carried_inputs,
        additional_inputs,
        mutated_arg_indices,
    ):
        # NOTE [unspecialize int carry with unbacked symints]
        # When we support int carry, we'll also need to support int output of body_fn because.
        # previous iteration's output is next iteration's input and they must match.
        # For carries, when we start tracing while_loop, they can be
        #   - constants e.g. (0, [1, 3])
        #   - backed symints (x.shape[0], [x.shape[1] + x.stride[1], x.shape[2]])
        #   - unbacked symints e.g. (u0, [u0 + u1, u2])
        #   We choose the most conservative design: in all cases, we create new unbacked symints to trace the
        #   subgraph. It's possible to do some analysis on initial carry and the output of first
        #   iteration to determine a better range for the output unbacked symbol e.g. when input is an unbacked
        #   symint >= 0 before the while_loop but in general this is difficult because we don't know
        #   the number of iterations. Users would have to re-constrain the unbacked symint in subgraph if needed.
        #
        # For output of fake cond_fn, it could be constant bool or SymBool (e.g. return x.shape[0] < 4,
        #   where x.shape[0] can be either static of dynamic). In the case of constant bool, we should do a
        #   specialization (NYI).

        # For output of fake body_fn, it could be all three types though from user's point of view,
        # they're all integers e.g.

        #   init_carry = (0, s0, u1, t)
        #   def body_fn(u0, s0, u1, t):
        #     ...
        #     return (t.shape[0], t.shape[1], t.shape[2], y + 1)
        #
        #   It may seem that a constant output isn't possible: users shouldn't write a while_loop
        #   that always return 0. But it could be that a shape is not set as dynamic properly (e.g.
        #   automatic dynamic hasn't been triggered).
        #
        #   For this reason, we treat int, symint outputs in the same way:
        #   - they can match against any of int, symint carry
        #   - we unspecialize them with new unbacked symints in fake while_loop
        #   Similarly, we could do some analysis to refine the output ranges but it's easier to start with
        #   fresh unbacked symints. One surprising case can be: an input unbacked symint is constrained by
        #   users to be >= 0 (either before while_loop or inside body_fn) and it increments by 1 in each
        #   iteration. Ideally, we should know that the final output is >= 0 but we didn't constrain the
        #   unbacked symint output of subgraph as of today because this requires a smart range analysis.
        fake_mode: FakeTensorMode = _find_or_create_fake_mode()

        def _unspecialize_carried_inputs(x):
            if isinstance(x, (int, torch.SymInt)):
                return _create_unbacked_symint(
                    fake_mode, ignore_fresh_unbacked_symbols=True
                )
            # Note: [unspecialize constant tensor carry]
            # We need to disable constant specialization for tensor inputs that become loop carries.
            # Here's the problem: when a user creates a constant tensor e.g. torch.tensor(0), PyTorch calls aten.lift_fresh_copy
            # to create a safe copy (avoiding aliasing issues), which creates a FakeTensor with constant=True.
            # But when this FakeTensor becomes a loop carry, we have a problem:
            # - Operations like .item() will read the constant value and bake it into the traced code
            # - This is incorrect because carry variables change between loop iterations
            # - The traced code would use the wrong constant value for all iterations
            # Solution: We clone the constant tensors and mark the cloned tensor as non-constant so they won't
            # be specialized to fixed values during tracing body_fn or cond_fn.
            elif isinstance(x, torch.Tensor):
                x = x.clone()
                if hasattr(x, "constant") and x.constant is not None:
                    # pyrefly: ignore [missing-attribute]
                    x.constant = None
            return x

        with disable_proxy_modes_tracing():
            unspecialized_carried_inputs = pytree.tree_map_only(
                (int, torch.SymInt, torch.Tensor),
                # For temporarily created unbacked symints, we don't need to bind them to any proxy
                lambda x: _unspecialize_carried_inputs(x),
                carried_inputs,
            )

            def produce_graph(fn):
                cloned_carried_inputs = pytree.tree_map_only(
                    torch.Tensor, lambda x: x.clone(), unspecialized_carried_inputs
                )
                return reenter_make_fx(fn)(*cloned_carried_inputs, *additional_inputs)

            cond_graph = produce_graph(cond_fn)
            body_graph = produce_graph(body_fn)

        next_name = None
        i = 0
        # pyrefly: ignore [bad-assignment]
        while not next_name:
            candidate = f"while_loop_cond_graph_{i}"
            if hasattr(proxy_mode.tracer.root, candidate):
                i += 1
            else:
                next_name = candidate
        cond_graph_name = next_name
        body_graph_name = f"while_loop_body_graph_{i}"
        if hasattr(proxy_mode.tracer.root, body_graph_name):
            raise AssertionError(
                f"proxy_mode.tracer.root already has attribute {body_graph_name}"
            )

        proxy_mode.tracer.root.register_module(cond_graph_name, cond_graph)
        proxy_mode.tracer.root.register_module(body_graph_name, body_graph)

        args = (cond_graph, body_graph, carried_inputs, additional_inputs)
        kwargs = {}
        if not stack_output and mutated_arg_indices:
            kwargs["mutated_arg_indices"] = mutated_arg_indices

        proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

        out_proxy = proxy_mode.tracer.create_proxy(
            "call_function", op, proxy_args, kwargs, name=op._name
        )

        out = op(
            cond_graph,
            body_graph,
            unspecialized_carried_inputs,
            additional_inputs,
            **kwargs,
        )
        return track_tensor_tree(
            out, out_proxy, constant=None, tracer=proxy_mode.tracer
        )

    return _trace_while_loop(
        mode,
        op,
        cond_fn,
        body_fn,
        carried_inputs,
        additional_inputs,
        mutated_arg_indices,
    )


@while_loop_op.py_impl(FakeTensorMode)
def while_loop_fake_tensor_mode(
    mode,
    cond_fn,
    body_fn,
    carried_inputs,
    additional_inputs,
    stack_output=False,
    mutated_arg_indices="",
):
    with mode:
        # NOTE: [Handling unback symints in subgraph of while_loop]
        # The idea is that the scope of unbacked symints are limited to the subgraph.
        #
        # We're implementing the fake tensor mode of while_loop operator.
        # and we run body_fn once to get an fake output.
        # Let's first consider the case that unbacked symints are tensor shapes:
        #
        # Case 1:
        # if the unbacked symints is local to the subgraph e.g.
        #   def body_fn(it, x):
        #     nz = x.nonzero()
        #     return it+1. nz.sum()
        # we can just ignore the newly created unbacked symints because it has
        # no effect on the output of while_loop and it's tracked when we tracing.
        # the subgraph.
        #
        # Case 2:
        # if the unbacked symints are shape of output of while_loop e.g.
        #   def body_fn(it, x):
        #     nz = x.nonzero()
        #     return it+1, nz
        # This will fail the shape check because in each iteration, the carried_input's shape
        # must match the output shape as nz.shape contains newly allocated unbacked symint, this
        # won't match the carried_input's shape.
        #
        # Case 3:
        # if the unbacked symints are shape of carried_inputs e.g.
        #   nz = a.nonzero()
        #   body_fn(it, nz):
        #     return it+1. nz.sin() + 1,
        # There's no new unbacked symints allocated in subgraph, so we're safe.
        with mode.shape_env.ignore_fresh_unbacked_symbols():
            # body_fn return output with the same pytree and tensor meta data as carried_inputs
            # so we could just return the output after one iteration.
            body_outs = body_fn(*carried_inputs, *additional_inputs)
            check_meta_consistency(
                carried_inputs,
                body_outs,
                "carried_inputs",
                "body_output",
                include_contiguity=False,
            )

        if stack_output:
            n_iter = _create_unbacked_symint(mode, ignore_fresh_unbacked_symbols=False)
            if not all(isinstance(x, torch.Tensor) for x in carried_inputs):
                raise AssertionError(
                    f"all carried_inputs must be tensors for stack_output, got {[type(x) for x in carried_inputs]}"
                )
            fake_outputs = tuple(
                out.clone()
                .unsqueeze(0)
                .repeat((n_iter,) + tuple(1 for _ in range(out.dim())))
                for out in body_outs
            )
            return pytree.tree_map_only(
                (int, torch.SymInt),
                # For while_loop's unbacked symint output, we want them to be bound
                # to the proxy of while_loop's output.
                lambda _: _create_unbacked_symint(
                    mode, ignore_fresh_unbacked_symbols=False
                ),
                fake_outputs,
            )

        # See NOTE [unspecialize int carry with unbacked symints]
        return pytree.tree_map_only(
            (int, torch.SymInt),
            # For while_loop's unbacked symint output, we want them to be bound
            # to the proxy of while_loop's output.
            lambda _: _create_unbacked_symint(
                mode, ignore_fresh_unbacked_symbols=False
            ),
            body_outs,
        )


@while_loop_op.py_functionalize_impl
def while_loop_func(
    ctx,
    cond_fn,
    body_fn,
    carried_inputs,
    additional_inputs,
    stack_output=False,
    mutated_arg_indices="",
):
    op = while_loop_stack_output_op if stack_output else while_loop_op
    # For now, we only support auto-functionalization for while_loop when using python
    # functionalization mode
    if not stack_output and hasattr(ctx, "mode"):
        hop_instance = HopInstance.create(
            op,
            cond_fn,
            body_fn,
            carried_inputs,
            additional_inputs,
            mutated_arg_indices=mutated_arg_indices,
        )
        if can_auto_functionalize(hop_instance):
            return do_auto_functionalize_v2(
                ctx.mode,
                hop_instance,
                tuple(
                    pytree.tree_flatten(
                        (cond_fn, body_fn, carried_inputs, additional_inputs)
                    )[0]
                ),
                {},
            )

    unwrapped_carried_inputs = ctx.unwrap_tensors(carried_inputs)
    unwrapped_additional_inputs = ctx.unwrap_tensors(additional_inputs)
    unwrapped_inputs = unwrapped_carried_inputs + unwrapped_additional_inputs
    with ctx.redispatch_to_next():
        functional_cond_fn = ctx.functionalize(_maybe_run_with_interpreter(cond_fn))
        functional_body_fn = ctx.functionalize(_maybe_run_with_interpreter(body_fn))
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for fn, fn_name in [
            (cond_fn, "cond_fn"),
            (body_fn, "body_fn"),
        ]:
            _check_alias_and_mutation(fn, unwrapped_inputs, fn_name, pre_dispatch)
        op_kwargs = {}
        if not stack_output and mutated_arg_indices:
            op_kwargs["mutated_arg_indices"] = mutated_arg_indices
        ret = op(
            functional_cond_fn,
            functional_body_fn,
            unwrapped_carried_inputs,
            unwrapped_additional_inputs,
            **op_kwargs,
        )
        return ctx.wrap_tensors(ret)


class WhileLoopStackOutputOp(HigherOrderOperator):
    """
    while_loop_stack_output is a variant of while_loop that returns a stack of outputs.
    Its semantic can be illurated using python code as:
    def while_loop_stack_output(cond_fn, body_fn, carried_inputs, additional_inputs):
        outs = []
        while cond_fn(*carried_inputs, *additional_inputs):
            out = body_fn(*carried_inputs, *additional_inputs)
            outs.append(out)
        return torch.stack(outs)

    It's useful for supporting autograd of while_loop.
    """

    def __init__(self) -> None:
        super().__init__("while_loop_stack_output")

    def __call__(
        self,
        cond_fn: Callable,
        body_fn: Callable,
        carried_inputs: tuple[torch.Tensor | int | float | bool],
        additional_inputs: tuple[torch.Tensor | torch.SymInt | int, ...],
        /,
    ):
        if not isinstance(carried_inputs, (tuple, list)):
            raise RuntimeError(
                f"carried_inputs must be a tuple or list, got {type(carried_inputs)}"
            )
        if not isinstance(additional_inputs, (tuple, list)):
            raise RuntimeError(
                f"additional_inputs must be a tuple or list, got {type(additional_inputs)}"
            )

        validate_subgraph_args_types(carried_inputs)
        validate_subgraph_args_types(additional_inputs)
        # pyrefly: ignore [missing-attribute]
        return super().__call__(cond_fn, body_fn, carried_inputs, additional_inputs)


# Note [while_loop autograd]
# Consider wthe following while_loop that can be visualized as:
#           additional_inputs
#       ┌─────┬─────┼─────┬─────┐
#       |     |     |     |     |
#       ↓     ↓     ↓     ↓     ↓
# x ──→ y0 ─→ y1 ─→ y2 ─→ y3 ─→ y4
#
# The backward can be visualized as follows:
#
#             g_additional_inputs
#         ┌──────┬──────┼──────┬──────┐
#         |      |      |      |      |
#         |      |      |      |      |
# gx <── gy0 <─ gy1 <─ gy2 <─ gy3 <─ gy4
#
# We can compute gx using chain rule:
#
#     gx = gy0 * bw(y0, x),
#
# where gy0 denotes the gradient of loss with respect to y0, and bw(y0, x) denotes the gradient of y0 with
# respect to x. Note that bw can be computed from forward body_fn easily using torch.autograd.grad.
# We could substitute the unknowns gy0, gy1, ..., with chain rule until gy4:
#
#     gx = gy1 * bw(y1, y0) * bw(y0, x)
#        = gy2 * bw(y2, y1) * bw(y1, y0) * bw(y0, x)
#        = ...
#        = gy4 * bw(y4, y3) * bw(y3, y2) * bw(y2, y1) * bw(y1, y0) * bw(y0, x)
#
# since gy4 is the gradient of the final output, which is given as the backward input, we've got a formula
# to compute gx. A abbr for the formula is: gy4 * bw43210x
#
# In a similar way, we can compute g_additional_inputs using chain rule:
#
# g_additional_inputs = gy0 * bw(y0, addi) + gy1 * bw(y1, addi) + gy2 * bw(y2, addi) + ... + gy4 * bw(y4, addi)
#
# Notice that gy0 = gy4 * bw43210, gy1 = gy4 * bw4321 etc, we now also get a formula for g_additional_inputs.
#
# Implementation:
# The idea of implementation is to construct a while_loop to calculate both gx and g_additional_inputs.
# Specifically, we can implement the backward of while_loop with as follows:
#
# def cond_fn(idx, grad_carries, grad_additional_inputs, fw_additional_inputs, fw_inps):
#     return idx < fw_inps.size(0)
#
# def body_fn(idx, grad_carries, grad_additional_inputs, fw_additional_inputs, fw_inps):
#     reversed_idx = fw_inps.size(0) - 1 - idx
#     next_grad_carry, next_grad_additional_inputs  = bw(fw_inps[reversed_idx], fw_additional_inputs, grad_carries)
#     return idx + 1, next_grad_carry, next_grad_additional_inputs + grad_additional_inputs
#
# idx = 0
# init_grad_carries = grads
# init_grad_additional_inputs = torch.zeros_like(g_additional_inputs)
# fw_inps = torch.cat([ctx.fw_carried_inputs, fw_outputs[:-1]])
# while_loop(cond_fn, body_fn, (idx, init_grad_carries, init_grad_additional_inputs,), (fw_additional_inputs, fw_inps))


class WhileLoopAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        cond_fn,
        body_fn,
        num_carried_inputs,
        num_additional_inputs,
        *carries_and_inputs,
    ):
        from torch._higher_order_ops.scan import split_into_chunks

        carries, additional_inputs = split_into_chunks(
            carries_and_inputs, [num_carried_inputs, num_additional_inputs]
        )
        with torch._C._AutoDispatchBelowAutograd():
            fw_outputs = while_loop_stack_output_op(
                cond_fn, body_fn, carries, additional_inputs
            )

        if hasattr(ctx, "fw_cond_fn"):
            raise AssertionError("ctx already has fw_cond_fn attribute")
        if hasattr(ctx, "fw_body_fn"):
            raise AssertionError("ctx already has fw_body_fn attribute")
        if hasattr(ctx, "carries"):
            raise AssertionError("ctx already has carries attribute")
        if hasattr(ctx, "additional_inputs"):
            raise AssertionError("ctx already has additional_inputs attribute")
        if hasattr(ctx, "fw_outputs"):
            raise AssertionError("ctx already has fw_outputs attribute")
        ctx.fw_cond_fn = cond_fn
        ctx.fw_body_fn = body_fn
        ctx.carries = carries
        ctx.additional_inputs = additional_inputs
        ctx.fw_outputs = fw_outputs
        loop_count = None
        # pyrefly: ignore [bad-assignment]
        for out in fw_outputs:
            if isinstance(out, torch.Tensor):
                if loop_count is not None:
                    if out.size(0) != loop_count:
                        raise AssertionError(
                            f"inconsistent loop_count: expected {loop_count}, got {out.size(0)}"
                        )
                else:
                    loop_count = out.size(0)
        if loop_count is None:
            raise AssertionError(
                "loop_count must not be None after processing fw_outputs"
            )

        # Remove the loop_count from pending_fresh_unbacked_symbols
        # because it's not part of forward output and it's impossible
        # to bind it to a proxy in forward graph anyways.
        if (
            isinstance(loop_count, torch.SymInt)
            and (shape_env := loop_count.node.shape_env)
            and loop_count in shape_env.pending_fresh_unbacked_symbols
        ):
            shape_env.pending_fresh_unbacked_symbols.remove(loop_count)

        # Even when body function is not executed, we clone and unsqueeze the input
        # to avoid the aliasing, therefore loop_count is always >= 1
        torch._check(loop_count >= 1)
        # We snapshot the dispatch keys in forward for materializing the
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()
        if len(fw_outputs) <= 0:
            raise AssertionError("fw_outputs shouldn't be empty")
        # Only the last of the output fw_outputs need to be returned
        return tuple(ckp[-1] for ckp in fw_outputs)

    @staticmethod
    def backward(ctx, *grads):
        from torch._higher_order_ops.cond import create_bw_fn
        from torch._higher_order_ops.scan import split_into_chunks

        # set up single step bw fn
        bw_body_fn = create_bw_fn(ctx.fw_body_fn, ctx.carries + ctx.additional_inputs)
        # Note [Handle inputs that're not differentiable]
        # When a forward input is non-differentiable e.g. a symint or an integer tensor, their gradients
        # will be None. However, we don't want to return None in the subgraph because this complicates the
        # inductor codegen, where we need to do a non-uniform treatment for None and tensors.
        # So we set up masks and filter the None gradients so that only tensors are returned from each step.
        carries_tensor_masks = [
            bool(isinstance(t, torch.Tensor) and t.dtype.is_floating_point)
            for t in ctx.carries
        ]
        additional_inputs_tensor_masks = [
            bool(isinstance(t, torch.Tensor) and t.dtype.is_floating_point)
            for t in ctx.additional_inputs
        ]

        init_idx = torch.zeros((), dtype=torch.int64)
        # Autograd can pass view gradients. The generated backward while_loop
        # needs stable carry metadata across iterations, so canonicalize tensor
        # gradients before they become carries.
        init_grad_carries = tuple(
            grad.clone(memory_format=torch.contiguous_format)
            if isinstance(grad, torch.Tensor)
            else grad
            for grad in filter_with_masks(grads, carries_tensor_masks)  # type: ignore[arg-type]
        )
        init_grad_additional_inputs = tuple(
            torch.zeros_like(t)
            for need_keep, t in zip(
                additional_inputs_tensor_masks, ctx.additional_inputs
            )
            if need_keep
        )
        # We need to the forward inputs to each iteration to compute the backward
        # which is the concatenation of first iteration input i.e. ctx.carries and all iterations'
        # output except the last iteration.
        fw_carries = [
            torch.cat([carry.unsqueeze(0), carries[:-1]])
            for carry, carries in zip(ctx.carries, ctx.fw_outputs)
        ]
        for fw_carry, carry in zip(fw_carries, ctx.carries):
            fw_carry.requires_grad_(carry.requires_grad)

        _, spec = pytree.tree_flatten(
            (
                init_idx,
                init_grad_carries,
                init_grad_additional_inputs,
                ctx.fw_outputs,
                ctx.additional_inputs,
            )
        )

        def cond_fn(*flat_args):
            (
                idx,
                grad_carries,
                grad_additional_inputs,
                fw_carries,
                additional_inputs,
            ) = pytree.tree_unflatten(flat_args, spec)
            if not isinstance(fw_carries[0], torch.Tensor):
                raise AssertionError(
                    f"expected fw_carries[0] to be torch.Tensor, got {type(fw_carries[0])}"
                )
            # excluding the last iteration's output
            return idx < fw_carries[0].size(0)

        def body_fn(*flat_args):
            (
                idx,
                grad_carries,
                grad_additional_inputs,
                fw_carries,
                additional_inputs,
            ) = pytree.tree_unflatten(flat_args, spec)
            reversed_idx = fw_carries[0].size(0) - idx - 1
            selected_fw_carries = [
                ckp.select(0, reversed_idx.item()) for ckp in fw_carries
            ]
            cur_grad_carries, cur_grad_additional_inputs = split_into_chunks(
                bw_body_fn(*selected_fw_carries, *additional_inputs, *grad_carries),
                [len(ctx.carries), len(ctx.additional_inputs)],
            )
            if not all(isinstance(t, torch.Tensor) for t in cur_grad_carries):
                raise AssertionError(
                    f"all cur_grad_carries must be tensors, got {[type(t) for t in cur_grad_carries]}"
                )
            cur_grad_carries_tensors = filter_with_masks(
                cur_grad_carries, carries_tensor_masks
            )
            cur_grad_additional_inputs_tensors = filter_with_masks(
                cur_grad_additional_inputs, additional_inputs_tensor_masks
            )
            return (
                idx + 1,
                *cur_grad_carries_tensors,
                *(
                    cur_grad + grad
                    for cur_grad, grad in zip(
                        cur_grad_additional_inputs_tensors, grad_additional_inputs
                    )
                ),
            )

        args_single_step_bw = (
            init_idx,
            *init_grad_carries,
            *init_grad_additional_inputs,
            *fw_carries,
            *ctx.additional_inputs,
        )

        cond_gm = materialize_as_graph(
            cond_fn,
            args_single_step_bw,
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )

        body_gm = materialize_as_graph(
            body_fn,
            args_single_step_bw,
            ctx._fw_include_key_set,
            ctx._fw_exclude_key_set,
            force_enable_grad=True,
        )

        _, final_grad_carries, final_grad_additional_inputs = split_into_chunks(
            while_loop_op(
                # pyrefly: ignore [bad-argument-type]
                cond_gm,
                # pyrefly: ignore [bad-argument-type]
                body_gm,
                # pyrefly: ignore [bad-argument-type]
                (
                    init_idx,
                    *init_grad_carries,
                    *init_grad_additional_inputs,
                ),
                (*fw_carries, *ctx.additional_inputs),
            ),
            [1, len(init_grad_carries), len(init_grad_additional_inputs)],
        )
        return (
            None,
            None,
            None,
            None,
            *fill_none_with_masks(final_grad_carries, carries_tensor_masks),
            *fill_none_with_masks(
                final_grad_additional_inputs, additional_inputs_tensor_masks
            ),
        )


while_loop_stack_output_op = WhileLoopStackOutputOp()

while_loop_stack_output_op.py_impl(DispatchKey.CompositeExplicitAutograd)(
    functools.partial(while_loop_dense, stack_output=True)
)

while_loop_stack_output_op.py_impl(ProxyTorchDispatchMode)(
    functools.partial(while_loop_tracing, stack_output=True)
)

while_loop_stack_output_op.py_impl(FakeTensorMode)(
    functools.partial(while_loop_fake_tensor_mode, stack_output=True)
)

while_loop_stack_output_op.py_functionalize_impl(
    functools.partial(while_loop_func, stack_output=True)
)

while_loop_stack_output_op.py_autograd_impl(
    autograd_not_implemented(while_loop_stack_output_op, deferred_error=True)
)
<|endoftext|># mypy: allow-untyped-defs

import torch
import torch._subclasses.functional_tensor
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import autograd_not_implemented, register_fake
from torch._ops import HigherOrderOperator
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode


@exposed_in("torch")
def strict_mode(callable, operands):
    if torch.compiler.is_dynamo_compiling():
        return strict_mode_op(callable, operands)

    from torch._higher_order_ops.utils import _hop_compile_and_call

    return _hop_compile_and_call(strict_mode_op, (callable, operands))


class StrictMode(HigherOrderOperator):
    def __init__(self):
        super().__init__("strict_mode")

    def __call__(self, callable, operands):
        # pyrefly: ignore [missing-attribute]
        return super().__call__(callable, operands)


strict_mode_op = StrictMode()


@strict_mode_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def strict_mode_op_dense(callable, operands):
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    return callable(*operands)


strict_mode_op.py_autograd_impl(
    autograd_not_implemented(strict_mode_op, deferred_error=True)
)


@strict_mode_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, callable, operands):
    return trace_strict_mode(mode, strict_mode_op, callable, operands)


def trace_strict_mode(mode, strict_mode_op, callable, operands):
    pre_dispatch = getattr(mode, "pre_dispatch", False)

    with disable_proxy_modes_tracing():
        graph = make_fx(callable, pre_dispatch=pre_dispatch)(*operands)

    graph_name = mode.tracer.get_fresh_qualname("strict_graph_")
    mode.tracer.root.register_module(graph_name, graph)

    args = (graph, operands)

    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)

    out_proxy = mode.tracer.create_proxy(
        "call_function", strict_mode_op, proxy_args, {}, name="strict_mode"
    )

    out = graph(*operands)
    return track_tensor_tree(out, out_proxy, constant=None, tracer=mode.tracer)


@register_fake(strict_mode_op, skip_cache=True)
def strict_mode_fake_tensor_mode(callable, operands):
    return callable(*operands)


@strict_mode_op.py_functionalize_impl
def strict_mode_func(ctx, callable, inputs):
    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    with ctx.redispatch_to_next():
        functional_callable = ctx.functionalize(callable)

        cond_return = strict_mode_op(functional_callable, unwrapped_inputs)
        return ctx.wrap_tensors(cond_return)
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: this file may be removed once we move to a dynamo frontend

import contextlib
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any, TypeAlias

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import (
    clone_outputs_aliasing_inputs,
    redirect_to_mode,
    register_fake,
    save_values_for_backward,
    saved_values,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensor, is_fake_tensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils.checkpoint import _CachedTorchDispatchMode, _CachingTorchDispatchMode


# Proxy the HOP instead of inlining into it
# And trace it with local shapes for AP
_DEFER_INLINING = False

GraphArg: TypeAlias = tuple[torch.Tensor, int, torch.SymInt, None]


@contextmanager
def defer_inlining() -> Generator[None, None, None]:
    global _DEFER_INLINING
    prior = _DEFER_INLINING
    try:
        _DEFER_INLINING = True
        yield
    finally:
        _DEFER_INLINING = prior


# Used to unwrap tensors classes like FunctionalTensor and Parameter
def _new_tensor(
    t: Any,
    new_shape: Sequence[int] | None = None,
    new_stride: Sequence[int] | None = None,
) -> Any:
    if isinstance(t, torch.Tensor):
        if type(t) not in (FunctionalTensor, FakeTensor, torch.Tensor):
            raise AssertionError(f"No subclasses support for now, found {type(t)}")
        return torch.empty_strided(
            t.size() if new_shape is None else new_shape,
            t.stride() if new_stride is None else new_stride,
            device=t.device,
            dtype=t.dtype,
            requires_grad=t.requires_grad,
        )
    return t


# Autoparallel specific, we want to treat plain tensors as DTensors
def _redistribute(
    args: Any,
    all_placements: tuple[Any],
    mesh: Any,
    shape_stride_fn: Callable[[torch.Tensor, Any, Any], tuple[list[int], list[int]]],
) -> GraphArg:
    from torch._dispatch.python import suspend_functionalization
    from torch._guards import detect_fake_mode
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

    with (
        suspend_functionalization(),
        disable_functional_mode(),
        disable_proxy_modes_tracing(),
    ):
        fake_mode = detect_fake_mode(args)
        if fake_mode is None:
            raise AssertionError("defer_inlining() is only supported for FakeTensors")

        with fake_mode:
            new_args = list(pytree.tree_map(_new_tensor, args))
            for i, (tensor, placements) in enumerate(zip(new_args, all_placements)):
                if tensor is None:
                    # Sometimes gradients can be None
                    continue

                new_shape, new_stride = shape_stride_fn(
                    tensor,
                    mesh,
                    placements,
                )
                new_args[i] = _new_tensor(
                    tensor, new_shape=new_shape, new_stride=new_stride
                )

            new_args = tuple(new_args)
            if not all(
                (is_fake_tensor(t) or isinstance(t, (int, torch.SymInt, type(None))))
                for t in new_args
            ):
                raise AssertionError(f"Unexpected element in {args=}")

    return new_args


def redistribute_fw_inputs(
    global_args: Any, all_placements: Any, mesh: Any, _: int | None = None
) -> GraphArg:
    if len(global_args) != len(all_placements):
        raise AssertionError(
            f"global_args length ({len(global_args)}) != all_placements length ({len(all_placements)})"
        )
    return _redistribute(
        global_args,
        all_placements,
        mesh,
        torch.distributed.tensor._utils.compute_local_tensor_info,
    )


def redistribute_fw_outputs(
    local_outs: Any, all_placements: Any, mesh: Any, num_activations: int
) -> GraphArg:
    if len(local_outs) != len(all_placements) + num_activations:
        raise AssertionError(
            f"local_outs length ({len(local_outs)}) != "
            f"all_placements length ({len(all_placements)}) + num_activations ({num_activations})"
        )
    num_fw_outs = len(local_outs) - num_activations
    if num_fw_outs <= 0:
        raise AssertionError(f"num_fw_outs must be > 0, got {num_fw_outs}")
    outs, activations = local_outs[:num_fw_outs], local_outs[num_fw_outs:]
    return (
        *_redistribute(
            outs,
            all_placements,
            mesh,
            torch.distributed.tensor._utils.compute_global_tensor_info,
        ),
        *activations,
    )


def redistribute_bw_inputs(
    global_args: Any, all_placements: Any, mesh: Any, num_activations: int
) -> GraphArg:
    if len(global_args) != len(all_placements) + num_activations:
        raise AssertionError(
            f"global_args length ({len(global_args)}) != "
            f"all_placements length ({len(all_placements)}) + num_activations ({num_activations})"
        )
    activations, inputs = global_args[:num_activations], global_args[num_activations:]
    if len(inputs) <= 0:
        raise AssertionError("inputs must not be empty")
    local_inputs = _redistribute(
        inputs,
        all_placements,
        mesh,
        torch.distributed.tensor._utils.compute_local_tensor_info,
    )
    return (
        *activations,
        *local_inputs,
    )


def redistribute_bw_outputs(
    local_outs: Any, all_placements: Any, mesh: Any, _: int | None = None
) -> GraphArg:
    if len(local_outs) != len(all_placements):
        raise AssertionError(
            f"local_outs length ({len(local_outs)}) != all_placements length ({len(all_placements)})"
        )
    return _redistribute(
        local_outs,
        all_placements,
        mesh,
        torch.distributed.tensor._utils.compute_global_tensor_info,
    )


class LocalMapHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("local_map_hop")

    def __call__(self, gm: GraphModule, *args: Any, **kwargs: Any) -> Any:
        # pyrefly: ignore [missing-attribute]
        return super().__call__(gm, *args, **kwargs)


local_map_hop = LocalMapHOP()

# Registers dispatches for SAC
redirect_to_mode(local_map_hop, _CachingTorchDispatchMode)
redirect_to_mode(local_map_hop, _CachedTorchDispatchMode)


def create_hop_fw_bw(
    fw_gm: GraphModule,
    *_args: Any,
) -> tuple[GraphModule, GraphModule, int, int, set[int]]:
    """
    Traces a joint, applies passes and partitions it
    """
    # Keeping these imports here
    # Avoid circular dependencies once we upstream with dynamo frontend
    from torch._dispatch.python import suspend_functionalization
    from torch._functorch.aot_autograd import AOTConfig, create_joint
    from torch._guards import detect_fake_mode
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._subclasses.functional_tensor import disable_functional_mode
    from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing, make_fx

    local_map_kwargs = fw_gm.meta["local_map_kwargs"]  # type: ignore[attr-defined]
    if "in_placements" not in local_map_kwargs:
        raise AssertionError("'in_placements' not found in local_map_kwargs")
    if "out_placements" not in local_map_kwargs:
        raise AssertionError("'out_placements' not found in local_map_kwargs")
    if "device_mesh" not in local_map_kwargs:
        raise AssertionError("'device_mesh' not found in local_map_kwargs")
    if len(local_map_kwargs["in_placements"]) != len(_args):
        raise AssertionError(
            f"in_placements length ({len(local_map_kwargs['in_placements'])}) != _args length ({len(_args)})"
        )

    dummy_aot_config = AOTConfig(
        fw_compiler=None,  # type: ignore[arg-type]
        bw_compiler=None,  # type: ignore[arg-type]
        partition_fn=None,  # type: ignore[arg-type]
        decompositions={},
        num_params_buffers=0,
        aot_id=0,
        keep_inference_input_mutations=False,
    )

    with suspend_functionalization(), disable_functional_mode():
        with disable_proxy_modes_tracing():
            # If someone runs this hop under the default compiler backend ("eager")
            # Then this path will be run with the actual user inputs. We convert them
            # to fake tensors in order to not perform any actual compute.

            fake_mode = detect_fake_mode(_args)
            if fake_mode is None:
                fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

            with fake_mode:
                fw_inputs = redistribute_fw_inputs(
                    _args,
                    local_map_kwargs["in_placements"],
                    local_map_kwargs["device_mesh"],
                )
                if len(fw_inputs) != len(local_map_kwargs["in_placements"]):
                    raise AssertionError(
                        f"fw_inputs length ({len(fw_inputs)}) != "
                        f"in_placements length ({len(local_map_kwargs['in_placements'])})"
                    )

            if not all(
                (is_fake_tensor(t) or isinstance(t, (int, torch.SymInt)))
                for t in fw_inputs
            ):
                raise AssertionError(f"Unexpected element in {fw_inputs=}")

            ctx = (
                fake_mode.shape_env.ignore_fresh_unbacked_symbols
                if fake_mode.shape_env is not None
                else contextlib.nullcontext
            )
            with ctx():
                fw_outs = fw_gm(*fw_inputs)

            example_grads = pytree.tree_map(
                _new_tensor,
                fw_outs,
            )
            if not isinstance(example_grads, (list, tuple)):
                example_grads = [example_grads]

            num_fw_inputs = len(fw_inputs)
            num_fw_outputs = len(example_grads)

        def joint_f(
            *primals_and_tangents: list[torch.Tensor],
        ) -> Any:
            primals = primals_and_tangents[:num_fw_inputs]
            tangents = primals_and_tangents[num_fw_inputs:]

            def prepare_fw_with_masks(
                fw_gm: torch.fx.GraphModule,
            ) -> Callable[..., Any]:
                def fw_with_masks(*args: Any) -> tuple[tuple[Any], list[bool]]:
                    # The Interpreter here is required to propagate metadata
                    # from the dynamo graph body to the local_map graph body.
                    # This is required for fx_traceback.annotate for work.
                    fw_out = torch.fx.Interpreter(fw_gm).run(*args)
                    if not isinstance(fw_out, tuple):
                        raise AssertionError(
                            "Dynamo traced submodule should return tuple"
                        )
                    return fw_out, [
                        bool(isinstance(ret, torch.Tensor) and ret.requires_grad)
                        for ret in fw_out
                    ]

                return fw_with_masks

            fw_outs, grads = create_joint(
                prepare_fw_with_masks(fw_gm), aot_config=dummy_aot_config
            )(primals, tangents)
            from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

            if has_free_unbacked_symbols((*fw_outs, *grads)):
                raise AssertionError(
                    "Unbacked symints leaking outside of the joint graph is not yet supported."
                )

            maybe_clone = clone_outputs_aliasing_inputs(primals_and_tangents)
            # put grads first to work with existing hop utils
            return pytree.tree_map(maybe_clone, (*grads, *fw_outs))

        filtered_grads_idx = set()
        for i, example_grad in enumerate(example_grads):
            # Filter out grads that are None or do not require_grad.
            # The AOTAutograd utils we rely on force this assumption.
            # We must also filter the runtime tangents too.
            if example_grad is not None and (
                isinstance(example_grad, torch.Tensor) and example_grad.requires_grad
            ):
                filtered_grads_idx.add(i)

        primals_and_tangents = [
            *fw_inputs,
            *[example_grads[i] for i in filtered_grads_idx],
        ]
        joint_hop_gm = make_fx(joint_f)(*primals_and_tangents)
        from torch._functorch._aot_autograd.graph_capture import (
            copy_fwd_metadata_to_bw_nodes,
        )

        copy_fwd_metadata_to_bw_nodes(joint_hop_gm)

        from torch._functorch._aot_autograd.graph_compile import prepare_for_partitioner
        from torch._inductor.compile_fx import partition_fn

        # Match partitioner convention
        prepped_joint_hop_gm = prepare_for_partitioner(
            joint_hop_gm, num_fw_inputs, num_fw_outputs
        )
        with disable_proxy_modes_tracing():
            # Also runs joint passes
            new_fw_gm, new_bw_gm = partition_fn(
                prepped_joint_hop_gm,
                [],
                num_fwd_outputs=num_fw_outputs,
                static_lifetime_input_indices=[],
            )

        # Fix tags because min-cut does not respect fw/bw boundary, breaking
        # default partitioner's assumptions.
        for node in new_fw_gm.graph.nodes:
            node.meta["partitioner_tag"] = "is_forward"
            node.meta.pop("autograd_backward", None)
        for node in new_bw_gm.graph.nodes:
            node.meta["partitioner_tag"] = "is_backward"
            node.meta["autograd_backward"] = True

        # Propagate meta onto fw/bw graphs, later will be set on proxied nodes
        new_fw_gm.meta["local_map_kwargs"] = local_map_kwargs
        new_bw_gm.meta["local_map_kwargs"] = {**local_map_kwargs}
        # Okay because Autoparallel assumes same sharding between param and grads
        new_bw_gm.meta["local_map_kwargs"]["in_placements"] = tuple(
            [local_map_kwargs["out_placements"][i] for i in filtered_grads_idx]
        )
        new_bw_gm.meta["local_map_kwargs"]["out_placements"] = local_map_kwargs[
            "in_placements"
        ]

        # Validate Forward
        fw_kwargs = new_fw_gm.meta["local_map_kwargs"]
        expected_fw_inputs = len(fw_kwargs["in_placements"])
        expected_fw_outputs = len(fw_kwargs["out_placements"])
        actual_fw_inputs = len(new_fw_gm.graph.find_nodes(op="placeholder"))
        actual_fw_outputs = num_fw_outputs
        if expected_fw_inputs != actual_fw_inputs:
            raise AssertionError(
                f"expected_fw_inputs ({expected_fw_inputs}) != actual_fw_inputs ({actual_fw_inputs})"
            )
        if expected_fw_outputs != actual_fw_outputs:
            raise AssertionError(
                f"expected_fw_outputs ({expected_fw_outputs}) != actual_fw_outputs ({actual_fw_outputs})"
            )

        # Validate Activations
        if len(new_fw_gm.graph.find_nodes(op="output")) != 1:
            raise AssertionError(
                f"Expected exactly 1 output node, got {len(new_fw_gm.graph.find_nodes(op='output'))}"
            )
        num_activations = (
            len(new_fw_gm.graph.find_nodes(op="output")[0].args[0]) - num_fw_outputs
        )
        # tensors first, then symints
        if num_activations < 0:
            raise AssertionError(f"num_activations must be >= 0, got {num_activations}")

        # Validate Backward
        bw_kwargs = new_bw_gm.meta["local_map_kwargs"]
        expected_bw_inputs = len(bw_kwargs["in_placements"])
        expected_bw_outputs = len(bw_kwargs["out_placements"])
        actual_bw_inputs = (
            len(new_bw_gm.graph.find_nodes(op="placeholder")) - num_activations
        )
        if actual_bw_inputs <= 0:
            raise AssertionError(
                f"actual_bw_inputs must be > 0, got {actual_bw_inputs}"
            )
        if expected_fw_inputs + expected_bw_inputs != len(primals_and_tangents):
            raise AssertionError(
                f"expected_fw_inputs ({expected_fw_inputs}) + expected_bw_inputs ({expected_bw_inputs}) "
                f"!= primals_and_tangents length ({len(primals_and_tangents)})"
            )
        if actual_fw_inputs + actual_bw_inputs != len(primals_and_tangents):
            raise AssertionError(
                f"actual_fw_inputs ({actual_fw_inputs}) + actual_bw_inputs ({actual_bw_inputs}) "
                f"!= primals_and_tangents length ({len(primals_and_tangents)})"
            )
        if len(new_bw_gm.graph.find_nodes(op="output")) != 1:
            raise AssertionError(
                f"Expected exactly 1 bw output node, got {len(new_bw_gm.graph.find_nodes(op='output'))}"
            )
        actual_bw_outputs = len(new_bw_gm.graph.find_nodes(op="output")[0].args[0])
        if expected_bw_inputs != actual_bw_inputs:
            raise AssertionError(
                f"expected_bw_inputs ({expected_bw_inputs}) != actual_bw_inputs ({actual_bw_inputs})"
            )
        if expected_bw_outputs != actual_bw_outputs:
            raise AssertionError(
                f"expected_bw_outputs ({expected_bw_outputs}) != actual_bw_outputs ({actual_bw_outputs})"
            )

        new_fw_gm.meta["num_activations"] = num_activations
        new_fw_gm.meta["is_backward"] = False
        new_bw_gm.meta["num_activations"] = num_activations
        new_bw_gm.meta["is_backward"] = True

        return new_fw_gm, new_bw_gm, num_fw_inputs, num_fw_outputs, filtered_grads_idx


class LocalMapAutogradOp(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx: Any,
        fw_gm: GraphModule,
        bw_gm: GraphModule,
        num_fw_ins: int,
        num_fw_outs: int,
        filtered_grads_idx: set[int],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor | None, ...]:
        from torch._functorch._aot_autograd.schemas import MemoryFormatMeta

        ctx.bw_gm = bw_gm
        ctx.num_fw_ins = num_fw_ins
        ctx.filtered_grads_idx = filtered_grads_idx

        with torch._C._AutoDispatchBelowAutograd():
            fw_outs_with_saved_activations = local_map_hop(fw_gm, *args, **kwargs)

        fw_outs = fw_outs_with_saved_activations[:num_fw_outs]
        saved_activations = fw_outs_with_saved_activations[num_fw_outs:]
        save_values_for_backward(ctx, saved_activations)

        # Force memory_format path (not exact size/stride) because local_map forward
        # operates on local shapes but backward receives global-shaped tangents.
        # TODO(ivankobzarev): Support exact size/stride by converting between local/global shapes.
        ctx.expected_tangent_metadata = {
            i: MemoryFormatMeta.from_tensor(fw_outs[i], force_use_memory_format=True)
            for i in filtered_grads_idx
        }
        return fw_outs

    @staticmethod
    def backward(
        ctx: Any, *_grads: tuple[torch.Tensor]
    ) -> tuple[torch.Tensor | None, ...]:
        from torch._functorch._aot_autograd.runtime_wrappers import (
            coerce_to_expected_memory_format,
        )

        if ctx.pos != sorted(ctx.pos):
            raise AssertionError(
                "Interleaving saved tensor activations and symints is not expected from min-cut partitioner."
            )
        ctx.pos = list(reversed(ctx.pos))  # make saved_values return symints first
        saved_activations = saved_values(ctx)
        with torch._C._AutoDispatchBelowAutograd():
            # Filter out grads that are None or do not require_grad.
            # The AOTAutograd utils we rely on force this assumption.
            grads = [_grads[i] for i in ctx.filtered_grads_idx]
            if len(grads) != len(ctx.expected_tangent_metadata):
                raise AssertionError(
                    f"{len(grads)=} vs {len(ctx.expected_tangent_metadata)}"
                )

            for i, meta in ctx.expected_tangent_metadata.items():
                grads[i] = coerce_to_expected_memory_format(grads[i], meta)

            grad_ins = local_map_hop(ctx.bw_gm, *saved_activations, *grads)
            if len(grad_ins) != ctx.num_fw_ins:
                raise RuntimeError(
                    f"Expected {ctx.num_fw_ins} grad_ins, got {len(grad_ins)}"
                )
        return None, None, None, None, None, *grad_ins


@local_map_hop.py_impl(torch._C.DispatchKey.Autograd)
def autograd_key(
    fw_gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> Any:
    local_map_kwargs = fw_gm.meta["local_map_kwargs"]  # type: ignore[attr-defined]
    if local_map_kwargs.get("in_grad_placements", None) is not None:
        raise AssertionError("local_map in_grad_placements are not yet supported.")
    if _DEFER_INLINING:
        fw_gm, bw_gm, num_fw_ins, num_fw_outs, filtered_grads_idx = create_hop_fw_bw(
            fw_gm, *args
        )
        return LocalMapAutogradOp.apply(
            fw_gm, bw_gm, num_fw_ins, num_fw_outs, filtered_grads_idx, *args, **kwargs
        )

    # TODO: get rid of this when we can install as a subgraph
    return torch.fx.Interpreter(fw_gm).run(*args, **kwargs)


@local_map_hop.py_functionalize_impl
def functional_mode_key(
    ctx: Any, gm: GraphModule, *args: Any, **kwargs: Any
) -> tuple[torch.Tensor]:
    if kwargs:
        raise AssertionError(f"kwargs must be empty, got {kwargs}")

    unwrapped_inputs = ctx.unwrap_tensors(args)
    with ctx.redispatch_to_next():
        out = local_map_hop(gm, *unwrapped_inputs)
        return ctx.wrap_tensors(out)


@register_fake(local_map_hop, skip_cache=True)
def fake_mode_key(
    gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> GraphArg:
    if not _DEFER_INLINING:
        return gm(*args, **kwargs)

    # otherwise, we need to convert to local shapes for AP
    # invoke_subgraph runs its body below Autograd, so local_map_hop's Autograd
    # key may not have partitioned the body or set fw/bw metadata.
    is_backward = gm.meta.get("is_backward", False)
    num_activations = gm.meta.get("num_activations", 0)
    redistribute_inputs = (
        redistribute_bw_inputs if is_backward else redistribute_fw_inputs
    )
    local_args = redistribute_inputs(
        args,
        gm.meta["local_map_kwargs"]["in_placements"],
        gm.meta["local_map_kwargs"]["device_mesh"],
        num_activations,
    )
    local_outs = gm(*local_args)
    redistribute_outputs = (
        redistribute_bw_outputs if is_backward else redistribute_fw_outputs
    )
    global_outs = redistribute_outputs(
        local_outs,
        gm.meta["local_map_kwargs"]["out_placements"],
        gm.meta["local_map_kwargs"]["device_mesh"],
        num_activations,
    )
    return global_outs


def _proxy_mode_arg(
    proxy_mode: ProxyTorchDispatchMode,
    arg: Any,
) -> Any:
    if isinstance(arg, torch.fx.GraphModule):
        # Install local_map bodies as real submodules so FX source emission can
        # reload them through self instead of closing over a local Python target.
        registered = any(
            arg is submod
            for _, submod in proxy_mode.tracer.root.named_modules()  # type: ignore[union-attr]
        )
        if not registered:
            qualname = proxy_mode.tracer.get_fresh_qualname("local_map_body")  # type: ignore[union-attr]
            proxy_mode.tracer.root.register_module(qualname, arg)  # type: ignore[union-attr]
    return proxy_mode.tracer.unwrap_proxy(arg)  # type: ignore[union-attr]


def proxy_mode_key_common(
    proxy_mode: ProxyTorchDispatchMode,
    gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    if proxy_mode is None:
        raise AssertionError("Mode should always be enabled for python fallback key")
    if len(kwargs) != 0:
        raise AssertionError(f"kwargs must be empty, got {kwargs}")

    example_out = local_map_hop(gm, *args, **kwargs)
    proxy_args = pytree.tree_map(
        lambda arg: _proxy_mode_arg(proxy_mode, arg),
        (gm, *args),
    )

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", local_map_hop, proxy_args, {}
    )

    # extract local_map args, post-dispatch operates on GraphModules
    if not gm.meta["local_map_kwargs"]:
        raise AssertionError("gm.meta['local_map_kwargs'] must be set")
    local_map_kwargs = gm.meta["local_map_kwargs"]

    # propagate local_map args to the call_function node
    out_proxy.node.meta["local_map_kwargs"] = local_map_kwargs

    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@local_map_hop.py_impl(ProxyTorchDispatchMode)
def proxy_mode_key(
    proxy_mode: ProxyTorchDispatchMode,
    gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    return proxy_mode_key_common(proxy_mode, gm, *args, **kwargs)


# Running HOP in eager with real tensors
@local_map_hop.py_impl(DispatchKey.CompositeExplicitAutograd)
def real_impl(
    gm: GraphModule,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor]:
    return gm(*args, **kwargs)
<|endoftext|># mypy: allow-untyped-defs
import enum
import functools
import itertools
import logging
from collections.abc import Callable
from typing import Any

import torch
import torch._prims_common as utils
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._functorch.vmap import restore_vmap, unwrap_batched, wrap_batched
from torch._guards import detect_fake_mode
from torch._higher_order_ops.auto_functionalize import (
    can_auto_functionalize,
    do_auto_functionalize_v2,
)
from torch._higher_order_ops.partitioner import (
    _find_hop_subgraph_outputs,
    HopGraphMinCutPartitioner,
    HopPartitionedGraph,
)
from torch._higher_order_ops.utils import (
    _maybe_compile_and_run_fn,
    check_meta_consistency,
    fill_none_with_masks,
    filter_with_masks,
    first_slice_copy,
    get_graph_output_example_values,
    get_tensor_mask,
    HopInstance,
    mask_list,
    materialize_as_graph,
    reenter_make_fx,
    split_into_chunks,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake_tensor
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import _get_current_dispatch_mode


logger: logging.Logger = logging.getLogger(__name__)
aten = torch._ops.ops.aten


def wrap_combine_fn_flat(
    *args, combine_fn, spec_init, spec_xs, num_init_leaves, num_inp_leaves
):
    if len(args) != (num_init_leaves + num_inp_leaves):
        raise AssertionError(
            f"combine_fn received wrong number of arguments, expected {num_init_leaves + num_inp_leaves}, but got {len(args)}"
        )
    carry = pytree.tree_unflatten(args[:num_init_leaves], spec_init)
    xs = pytree.tree_unflatten(args[num_init_leaves:], spec_xs)
    return combine_fn(carry, xs)


def _extract_carry_and_out(flat_out: list[Any], num_carry: int):
    return split_into_chunks(flat_out, [num_carry, len(flat_out) - num_carry])


# We also do a clone with contiguous_format. This is to be consistent with
# eager semantic of scan, which stacks the outputs. The result is contiguous
# as a result of the stack operation.
def stack_y(y: torch.Tensor, scan_length: int) -> torch.Tensor:
    return (
        y.unsqueeze(0)
        .repeat(*([scan_length] + [1] * y.ndim))
        .clone(memory_format=torch.contiguous_format)
    )


def call_operator(operator, *args):
    return pytree.tree_leaves(operator(*args))


def _build_empty_output_for_length_zero(
    combine_fn: Callable, init: pytree.PyTree
) -> pytree.PyTree:
    """Probe combine_fn(init, None) to learn the output structure for length==0."""
    fake_mode = detect_fake_mode(pytree.tree_leaves(init)) or FakeTensorMode()
    with fake_mode:
        fake_init = pytree.tree_map(
            lambda t: fake_mode.from_tensor(t)
            if isinstance(t, torch.Tensor) and not is_fake_tensor(t)
            else t,
            init,
        )
        _, sample_y = combine_fn(fake_init, None)

    def _empty_or_passthrough(l):
        if isinstance(l, torch.Tensor):
            return torch.empty([0] + list(l.shape), dtype=l.dtype, device=l.device)
        if l is not None:
            raise AssertionError(f"Expected leaf to be a Tensor or None, got {type(l)}")
        return l

    return pytree.tree_map(_empty_or_passthrough, sample_y)


def scan(
    combine_fn: Callable[
        [pytree.PyTree, pytree.PyTree], tuple[pytree.PyTree, pytree.PyTree]
    ],
    init: pytree.PyTree,
    xs: pytree.PyTree,
    *,
    dim: int = 0,
    reverse: bool = False,
    length: int | None = None,
) -> tuple[pytree.PyTree, pytree.PyTree]:
    r"""
    Performs an inclusive scan with a combine function.

    .. warning::

        ``torch.scan`` is a prototype feature in PyTorch. You may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> (Tensor, Tensor)``,
            or if xs is a pytree ``(pytree, pytree) -> (pytree, pytree)``.
            The first input to ``combine_fn`` is the previous or initial scan carry
            and the second input element to ``combine_fn`` is a slice of the input along dim.
            The first output element of ``combine_fn`` is the next scan carry
            and the second output  of ``combine_fn`` represents a slice of the output.
            This function must be pure, i.e., no lifted arguments are supported at the moment
            and may not have any side effects.
        init (torch.Tensor or pytree with tensor leaves): The initial scan carry, a tensor, or nested pytree of tensors.
            The ``init`` is expected to have the same pytree structure as the first output element (i.e. carry)
            of ``combine_fn``.
        xs (torch.Tensor or pytree with tensor leaves or None): The input tensor, or nested pytree of tensors.
            May be ``None`` when ``length`` is provided, in which case ``combine_fn`` receives ``None`` as ``x``
            each step (counter-loop mode).

    Keyword Args:
        dim (int): the dimension to scan over, default 0.
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        length (int or None): Optional number of scan iterations, default ``None``.
            When ``xs`` has tensor leaves, ``length`` is optional; if given it must equal
            ``xs.shape[dim]`` and serves only as a consistency check (no constraint when
            ``length`` is ``None``). When ``xs`` has no leaves (``None`` or empty pytree),
            ``length`` drives the number of iterations and ``combine_fn`` receives
            ``x=None`` each step. ``length=0`` with no xs tensors is supported in
            eager mode only; it is not supported under ``torch.compile``.

    Returns:
        final_carry (torch.Tensor or pytree with tensor leaves),
            the final carry of the scan operation with same pytree structure as init.
        out (torch.Tensor or pytree with tensor leaves),
            each tensor leaf is a stacked output along first dim, where each slice is the output of a scan iteration.
            If the scan dimension has size 0, ``final_carry`` equals ``init`` unchanged and each output leaf has
            size 0 along ``dim``. The gradient of ``final_carry`` with respect to ``init`` is the identity
            (not zero), since the body is never called and the carry passes through untouched.

    Restrictions:
        - The combine_fn shouldn't have any aliasing between input-input, input-output, and output-output. E.g. return a view
            or the same tensor as input is not supported. As a workaround, can clone the output to avoid aliasing.

        - The combine_fn shouldn't mutate any inputs. We'll remove the mutation restriction for inference soon. Please file an issue
            if you input mutation support for training is needed.

        - The combine_fn's init carry should match the next_carry in pytree structure and in tensor metadata.

    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            next_carry = y = x + y
            # clone the output to avoid output-output aliasing
            return next_carry, y.clone()


        i0 = torch.zeros(1)
        xs = torch.arange(5)
        # returns torch.tensor([10.]), torch.tensor([[0], [1.], [3.], [6.], [10.]])
        last_carry, cumsum = scan(add, init=i0, xs=xs)


    """
    # The reason we flatten init and xs before calling into dynamo is that
    # we want to create a consistent input ordering for combine_fn
    # and we also want to the input ordering matches the output ordering.
    leaves_init, spec_init = pytree.tree_flatten(init)
    leaves_xs_orig, spec_xs = pytree.tree_flatten(xs)

    # Determine whether xs carries any tensor data.
    xs_has_tensors = any(isinstance(l, torch.Tensor) for l in leaves_xs_orig)

    # short-cuts
    if length is not None:
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise RuntimeError(
                f"scan() length must be a non-negative integer, got {length!r}"
            )

        if not xs_has_tensors:
            if length == 0:
                if torch.compiler.is_dynamo_compiling():
                    # TODO: Resolve this in a follow-up PR
                    raise RuntimeError(
                        "scan() with length=0 and no xs tensors is not supported under torch.compile"
                    )
                return init, _build_empty_output_for_length_zero(combine_fn, init)

            # No real xs: fabricate a length-N dummy purely as an iteration counter.
            # the wrapped combine_fn below discards each slice and passes x=None to the body.
            leaves_xs_orig = [torch.zeros(length, dtype=torch.int64)]
            spec_xs = pytree.tree_structure(None)
            _user_combine_fn = combine_fn

            def combine_fn(carry, _ignored):  # noqa: E306
                return _user_combine_fn(carry, None)
    elif not xs_has_tensors:
        return init, []

    def _validate_input(cfn, lxs, linit, d, r, l):
        # Basic arguments check
        if not callable(cfn):
            raise RuntimeError(f"Combine_fn must be a callable, but got {cfn}")
        if not isinstance(d, int):
            raise RuntimeError("Dim must be an int, but got " + str(type(d)))
        if not isinstance(r, bool):
            raise RuntimeError("Reverse must be a bool, but got " + str(type(r)))

        if l is not None and xs_has_tensors and lxs[0].shape[d] != l:
            raise RuntimeError(
                f"scan() length={l} does not match xs size along dim={d}: "
                f"{lxs[0].shape[d]}"
            )

        # Checks for init
        if len(linit) == 0:
            raise RuntimeError("scan() operator requires init leaves.")
        for x in linit:
            if not isinstance(x, torch.Tensor):
                raise RuntimeError(f"All init leaves must be a Tensor but got {x}")

        # Checks for xs
        for x in lxs:
            if not isinstance(x, torch.Tensor):
                raise RuntimeError(f"All xs leaves must be a Tensor but got {x}")
        if any(x.ndim <= d for x in lxs):
            raise RuntimeError("All xs leaves must have at least 'dim + 1' dimensions")
        if any(x.shape[d] != lxs[0].shape[d] for x in lxs[1:]):
            raise RuntimeError("All xs leaves must have the same scan dimension size")

    ndim = leaves_xs_orig[0].ndim
    dim = utils.canonicalize_dim(ndim, dim)

    _validate_input(combine_fn, leaves_xs_orig, leaves_init, dim, reverse, length)

    # Move scan dim to 0 and always perform scan on dim 0
    leaves_xs = []
    for elem in leaves_xs_orig:
        leaves_xs.append(torch.movedim(elem, dim, 0) if dim != 0 else elem)

    if reverse:
        leaves_xs = [torch.flip(elem, [0]) for elem in leaves_xs]

    # TODO: Support _inductor lowering
    # TODO: Unify handling of pytrees for control flow ops, such as cond, while_loop, etc.

    combine_fn = functools.partial(
        wrap_combine_fn_flat,
        combine_fn=combine_fn,
        spec_init=spec_init,
        spec_xs=spec_xs,
        num_init_leaves=len(leaves_init),
        num_inp_leaves=len(leaves_xs),
    )

    def run_flattened_scan(combine_fn, leaves_init, leaves_xs):
        return scan_op(combine_fn, leaves_init, leaves_xs, ())

    carry, out = _maybe_compile_and_run_fn(
        run_flattened_scan,
        combine_fn,
        leaves_init,
        leaves_xs,
    )

    if reverse:
        out = pytree.tree_map(lambda elem: elem.flip([0]), out)

    # Move the scan dimension from 0 back to the user-specified `dim`.
    if dim != 0:
        out = pytree.tree_map(
            lambda elem: torch.movedim(elem, 0, dim) if dim < elem.ndim else elem,
            out,
        )

    return carry, out


class ScanOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("scan")

    def __call__(
        self,
        combine_fn,
        init,
        xs,
        additional_inputs,
        *,
        mutated_arg_indices: str = "",
    ):
        # There is currently an issue that the ScanOp is sometimes called with
        # the additional_inputs being a list. See https://github.com/pytorch/pytorch/issues/145785
        # Once this issue is resolved, the assertion should only allow tuples
        # and the tuple cast should be removed
        if not isinstance(additional_inputs, (tuple, list)):
            raise AssertionError(
                f"additional_inputs must be a tuple or list, got {type(additional_inputs)}"
            )
        additional_inputs = (
            tuple(additional_inputs)
            if isinstance(additional_inputs, list)
            else additional_inputs
        )
        validate_subgraph_args_types(additional_inputs)
        kwargs = {}
        if mutated_arg_indices:
            kwargs["mutated_arg_indices"] = mutated_arg_indices
        # pyrefly: ignore [missing-attribute]
        return super().__call__(combine_fn, init, xs, additional_inputs, **kwargs)

    # pyrefly: ignore [bad-override]
    def gen_schema(
        self, combine_fn, init, xs, additional_inputs, mutated_arg_indices=""
    ):
        from torch._higher_order_ops.schema import HopSchemaGenerator

        all_inputs = tuple(
            list(init)
            + [
                torch.empty_strided(
                    x.shape[1:],
                    x.stride()[1:],
                    dtype=x.dtype,
                    device=x.device,
                    requires_grad=x.requires_grad,
                )
                for x in xs
            ]
            + list(additional_inputs)
        )
        combine_gm = materialize_as_graph(combine_fn, all_inputs)

        # Mutation semantics for scan:
        # - additional_inputs is mutable: loop-invariant tensor identity
        #   across sequential iterations, same semantics as while_loop's
        #   additional_inputs (CUDA-graph-friendly lifted / pre-allocated
        #   buffers such as KV caches or workspace scratch).
        # - init is NOT mutable: init is only the *initial* carry,
        #   thus an in-place update to init only affects step 0.
        # - xs is NOT mutable: each iteration sees a fresh, storage-disjoint
        #   slice (xs[t] and xs[t+1] share no storage), so a mutation on
        #   xs[t] cannot be observed by iteration t+1. The only externally-
        #   observable effect is "write-back to xs's t-th slice", which is
        #   already expressible via the ys output path at no extra cost.
        #   If xs-like in-place updates are required, pass the buffer via
        #   additional_inputs and index into it inside combine_fn.
        outputs = get_graph_output_example_values(combine_gm)
        mutated_set = (
            {int(i) for i in mutated_arg_indices.split(",") if i}
            if mutated_arg_indices
            else set()
        )

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("combine_fn", combine_gm)

        for idx, arg in enumerate(init):
            schema_gen.add_arg(f"init{idx}", arg)

        for idx, arg in enumerate(xs):
            schema_gen.add_arg(f"xs{idx}", arg)

        offset = len(init) + len(xs)
        for idx, arg in enumerate(additional_inputs):
            schema_gen.add_arg(
                f"additional_input{idx}",
                arg,
                is_mutated=(offset + idx) in mutated_set,
            )

        for out in outputs:
            schema_gen.add_output(out)

        schema_gen.add_schema_tree_spec(combine_fn, init, xs, additional_inputs)
        return schema_gen.gen_schema()


scan_op = ScanOp()


def generic_scan(operator, init, xs, dim=0, additional_inputs=()):
    def _scan(init, xs):
        """Perform scan on `elems` using `elems_init."""
        carry = init
        if len(xs) == 0:
            return carry, []

        num_elems = xs[0].shape[dim]
        num_init_leaves = len(init)

        proto_xs = [first_slice_copy(x, dim) for x in xs]

        # Process element 0 to infer output shapes for pre-allocation
        # AND produce the first real result in a single call.  The previous
        # approach used first_slice_copy() for shape inference and then
        # re-processed element 0 in the main loop, calling the operator
        # num_elems+1 times.  That extra invocation is incorrect for
        # operators with side effects.
        carry, out_0 = _extract_carry_and_out(
            call_operator(
                operator,
                *carry,
                *proto_xs,
                *additional_inputs,
            ),
            num_init_leaves,
        )

        out_tensor_mask = get_tensor_mask(out_0)
        out_0_masked = mask_list(out_tensor_mask, out_0)

        # Pre-allocate
        # outs -> Output matrix
        # idxs -> Index matrix for scatter_
        # out: (num_elems, M, N, ...)
        # idx: (1, M, N)
        outs = [
            torch.empty(
                [num_elems] + list(e.size()),
                dtype=e.dtype,
                device=e.device,
            )
            for e in out_0_masked
        ]
        idxs = [
            torch.ones_like(e, dtype=torch.int64).unsqueeze(0) for e in out_0_masked
        ]

        if num_elems == 0:
            outs_expanded = [
                outs.pop(0) if out_m else None for out_m in out_tensor_mask
            ]
            return (*init, *outs_expanded)

        def store_out_in_outs(out, ind):
            # Store the intermediate out in the outs matrix
            for o, x, idx in zip(outs, out, idxs):
                # o: (num_elems, M, N ...)
                # x: (M, N, ...) -> (1, M, N)
                # ind * idx: (1, M, N,) with values to be ind
                # essentially: o[ind][n][k] = x[0][n][k]
                o.scatter_(0, ind * idx, x.unsqueeze(0))

        # Store element 0's result, then continue from element 1.
        store_out_in_outs(out_0_masked, 0)

        for i in range(1, num_elems):
            carry, out = _extract_carry_and_out(
                call_operator(
                    operator,
                    *carry,
                    *[elem.select(dim, i) for elem in xs],
                    *additional_inputs,
                ),
                num_init_leaves,
            )

            store_out_in_outs(mask_list(out_tensor_mask, out), i)

        # Expand outs with None depending on the tensor mask of the output
        outs_expanded = [outs.pop(0) if out_m else None for out_m in out_tensor_mask]

        return (*carry, *outs_expanded)

    scans = _scan(init, xs)
    return scans


def trace_scan(
    proxy_mode,
    func_overload,
    combine_fn: Callable,
    init: list[torch.Tensor],
    xs: list[torch.Tensor],
    additional_inputs: tuple[torch.Tensor],
    mutated_arg_indices: str = "",
):
    from torch._dynamo.utils import clone_input

    with disable_proxy_modes_tracing():
        sample_inits = [clone_input(x_init) for x_init in init]
        sample_inputs = [first_slice_copy(x) for x in xs]
        sample_additional_inputs = [
            clone_input(x) if isinstance(x, torch.Tensor) else x
            for x in additional_inputs
        ]
        combine_graph = reenter_make_fx(combine_fn)(
            *sample_inits, *sample_inputs, *sample_additional_inputs
        )

    outputs = None
    for node in combine_graph.graph.nodes:
        if node.op == "output":
            if outputs is not None:
                raise AssertionError("found multiple output nodes in combine_graph")
            if len(node.args) != 1:
                raise AssertionError(
                    f"expected output node to have 1 arg, got {len(node.args)}"
                )
            outputs = node.args[0]

    if outputs is None:
        raise AssertionError("no output node found in combine_graph")

    carry, output = _extract_carry_and_out(outputs, len(init))
    init_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        i.clone() for i in init
    ]
    carry_fake_tensors: list[torch.Tensor | torch.SymInt | int] = [
        c.meta["val"] for c in carry
    ]
    check_meta_consistency(
        init_fake_tensors, carry_fake_tensors, "init", "carry", include_contiguity=False
    )

    _, combine_graph_name = unique_graph_id(proxy_mode, prefix="scan_combine_graph")

    proxy_mode.tracer.root.register_module(combine_graph_name, combine_graph)

    args = (combine_graph, init, xs, additional_inputs)
    kwargs = {}
    if mutated_arg_indices:
        kwargs["mutated_arg_indices"] = mutated_arg_indices

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, kwargs, name="scan"
    )

    with disable_proxy_modes_tracing():
        scan_length = xs[0].shape[0]
        fake_carry, fake_outputs = _extract_carry_and_out(
            [o.meta["val"] if o is not None else None for o in outputs], len(init)
        )
        for t in fake_outputs:
            if not isinstance(t, torch.Tensor) and t is not None:
                raise AssertionError(
                    f"Expected leaf to be a Tensor or None, got {type(t)}"
                )
        out = (
            *fake_carry,
            *(
                stack_y(t, scan_length) if isinstance(t, torch.Tensor) else t
                for t in fake_outputs
            ),
        )

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@scan_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def scan_op_dense(combine_fn, init, xs, additional_inputs, mutated_arg_indices=""):
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    return generic_scan(combine_fn, init, xs, additional_inputs=additional_inputs)


class ScanAutogradOp(torch.autograd.Function):
    """
    NOTE: [scan partial grad handling]
    If any element of init, of xs, of the outputs or of the additional_inputs does not require gradients,
    i.e., requires_grad=False, there will be still gradients returned for those elements,
    but those gradients will be a tensor filled with zeros of the same shape as the element itself.

    A special case are additional_inputs that are not tensors. Such inputs can occur for example with symbolic tracing,
    where the shape symbol (SymInt) becomes an additional_input.
    For such cases, we compute a ``additional_inputs_tensor_mask``, which is True for elements of additional_inputs
    that are tensors and False otherwise. Gradients of additional_inputs are only accumulated if this mask is True,
    otherwise, the value of initial_g_additional_inputs is passed, which is None for non-Tensor values.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        hop_partitioned_graph,
        n_init,
        n_xs,
        n_additional_inputs,
        *operands,
    ):
        init, xs, additional_inputs = split_into_chunks(
            operands, [n_init, n_xs, n_additional_inputs]
        )
        ctx._scan_impl = ScanAutogradImpl(
            hop_partitioned_graph, init, xs, additional_inputs
        )
        with torch._C._AutoDispatchBelowAutograd():
            return ctx._scan_impl.call_forward()

    @staticmethod
    def backward(ctx, *grad_fw_outputs):
        return (
            None,
            None,
            None,
            None,
            *ctx._scan_impl.call_backward(*grad_fw_outputs),
        )


class ScanForwardIntermediatesHandlingPolicy(enum.Enum):
    """
    Partitioner can add intermediates to the output of original graph.
    These intermediates fall into 4 categories and we want to have different policies for handling them by
    modifying the graph:

    CLONE: we clone the intermediate when it is a carried input (i.e. init). In this case, this carry will be
        replaced with new values at each forward step so we need to clone the carry as part of return (i.e. ys)
        so as to remove the aliasing and that each step's intermediate will be stacked together and saved in backward.

    REMOVE_XS: we remove the intermediate from output when it is part of xs. Since xs is read-only, in this case,
        we can directly save them for backward to use.

    REMOVE_ADDITIONAL_INPUTS: we remove the intermediate from output when it is part of additional_inputs. additional_inputs
        are also read-only in each step, we can directly save them for backward to use. We differentiate XS and ADDITIONAL_INPUTS
        so that we could have different treatment for them in backward. In backward, we need to put xs intermediates in carry but
        put additional_inputs as backward scan's additional_inputs.

    KEEP: this corresponds to a real intermediate tensor operations' output. It varies at each forward step, we could just keep
        it as part of ys.

    """

    KEEP = 0
    CLONE = 1
    REMOVE_XS = 2
    REMOVE_ADDITIONAL_INPUTS = 3


class ScanAutogradImpl:
    """
    Wraps over partitioned graph and encapsulates scan-specific implementation details
    """

    def __init__(
        self, hop_partitioned_graph: HopPartitionedGraph, init, xs, additional_inputs
    ):
        self.hop_partitioned_graph = hop_partitioned_graph
        self.init = init
        self.xs = xs
        self.additional_inputs = additional_inputs
        self.forward_intermediates_handling_policies: list[
            ScanForwardIntermediatesHandlingPolicy
        ] = []
        self.saved_fw_xs: list[Any] = []
        self.saved_fw_additional_inputs: list[Any] = []
        self.saved_intermediates: list[Any] = []
        self.fw_spec = pytree.tree_flatten((init, xs, additional_inputs))[1]
        self._optimize_forward_intermediates()
        self._break_bw_input_output_aliasing()

    def _insert_clone(
        self, need_copy_node: torch.fx.Node, output_node: torch.fx.Node
    ) -> torch.fx.Node:
        graph: torch.fx.Graph = output_node.graph
        with graph.inserting_before(output_node):
            clone_node = graph.call_function(
                torch.ops.aten.clone.default,
                args=(need_copy_node,),
            )
            clone_node.meta = (
                need_copy_node.meta.copy() if hasattr(need_copy_node, "meta") else {}
            )
        return clone_node

    def _break_bw_input_output_aliasing(self) -> None:
        """
        The min-cut partitioner can produce a ``bw_gm`` whose output aliases
        an input placeholder. The most common case is a direct placeholder
        return (e.g. when an input doesn't require grad and its gradient is
        ``zeros_like(input)`` saved as a forward intermediate), but transitive
        aliases (views, ``_unsafe_view``, slices, ...) can hit the same path.

        When ``bw_gm`` is wrapped as the per-step backward inside ``scan_op``,
        any input returned as an output violates ``scan_op``'s aliasing-free
        invariant and surfaces as ``UncapturedHigherOrderOpError`` under
        dynamo. Clone bw_gm outputs that are direct placeholders or whose
        ``meta['val']`` shares storage with any placeholder.
        """
        from torch.multiprocessing.reductions import StorageWeakRef

        bw_gm = self.hop_partitioned_graph.bw_gm
        bw_output_node = next(iter(bw_gm.graph.find_nodes(op="output")))
        if len(bw_output_node.args) != 1:
            raise AssertionError(
                f"expected bw_gm output to have 1 arg, got {len(bw_output_node.args)}"
            )
        bw_outputs = bw_output_node.args[0]

        ph_storages: set = set()
        for ph in bw_gm.graph.find_nodes(op="placeholder"):
            val = ph.meta.get("val", None) if hasattr(ph, "meta") else None
            if isinstance(val, torch.Tensor):
                ph_storages.add(StorageWeakRef(val._typed_storage()))

        def _aliases_placeholder(node: torch.fx.Node) -> bool:
            if node.op == "placeholder":
                return True
            val = node.meta.get("val", None) if hasattr(node, "meta") else None
            if isinstance(val, torch.Tensor):
                return StorageWeakRef(val._typed_storage()) in ph_storages
            return False

        new_bw_outputs = []
        rewrote = False
        for out in bw_outputs:
            if isinstance(out, torch.fx.Node) and _aliases_placeholder(out):
                new_bw_outputs.append(self._insert_clone(out, bw_output_node))
                rewrote = True
            else:
                new_bw_outputs.append(out)
        if rewrote:
            bw_output_node.args = (tuple(new_bw_outputs),)
            bw_gm.graph.lint()
            bw_gm.recompile()

    def _optimize_forward_intermediates(self):
        """
        We optimize the forward intermediates by categorize forward intermediates into categories
        and construct a ScanForwardIntermediatesHandlingPolicy for them

        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Need remove aliasing in fw_gm:\n%s",
                self.hop_partitioned_graph.fw_gm.print_readable(print_output=False),
            )

        fw_gm = self.hop_partitioned_graph.fw_gm
        fw_all_outputs = _find_hop_subgraph_outputs(fw_gm)
        phs = list(fw_gm.graph.find_nodes(op="placeholder"))
        fw_outputs = fw_all_outputs[: self.hop_partitioned_graph.n_fw_outputs]
        fw_intermediates = fw_all_outputs[self.hop_partitioned_graph.n_fw_outputs :]

        init_phs, xs_phs, additional_inputs_phs = pytree.tree_unflatten(
            phs, self.fw_spec
        )
        init_node_set, xs_node_set, addi_node_set = (
            set(init_phs),
            set(xs_phs),
            set(additional_inputs_phs),
        )

        if len(self.forward_intermediates_handling_policies) != 0:
            raise AssertionError(
                "forward_intermediates_handling_policies should be empty"
            )
        if len(self.saved_fw_xs) != 0:
            raise AssertionError("saved_fw_xs should be empty")
        if len(self.saved_fw_additional_inputs) != 0:
            raise AssertionError("saved_fw_additional_inputs should be empty")
        intermediate_idx_to_ph_idx = {}
        ph_idx = {ph: i for i, ph in enumerate(phs)}
        for i, out in enumerate(fw_intermediates):
            if out in init_node_set:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.CLONE
                )
                intermediate_idx_to_ph_idx[i] = ph_idx[out]
            elif out in xs_node_set:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.REMOVE_XS
                )
                intermediate_idx_to_ph_idx[i] = ph_idx[out]
            elif out in addi_node_set:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.REMOVE_ADDITIONAL_INPUTS
                )
                intermediate_idx_to_ph_idx[i] = ph_idx[out]
            else:
                self.forward_intermediates_handling_policies.append(
                    ScanForwardIntermediatesHandlingPolicy.KEEP
                )

        new_output_node = []
        real_graph_inputs = (
            list(self.init) + list(self.xs) + list(self.additional_inputs)
        )
        fw_output_node = next(iter(fw_gm.graph.find_nodes(op="output")))
        for intermediate_idx, (node, policy) in enumerate(
            zip(fw_intermediates, self.forward_intermediates_handling_policies)
        ):
            if policy == ScanForwardIntermediatesHandlingPolicy.CLONE:
                new_output_node.append(self._insert_clone(node, fw_output_node))
            elif policy == ScanForwardIntermediatesHandlingPolicy.REMOVE_XS:
                if intermediate_idx not in intermediate_idx_to_ph_idx:
                    raise AssertionError(
                        f"intermediate_idx {intermediate_idx} not in intermediate_idx_to_ph_idx"
                    )
                inp_idx = intermediate_idx_to_ph_idx[intermediate_idx]
                self.saved_fw_xs.append(real_graph_inputs[inp_idx])
            elif (
                policy
                == ScanForwardIntermediatesHandlingPolicy.REMOVE_ADDITIONAL_INPUTS
            ):
                if intermediate_idx not in intermediate_idx_to_ph_idx:
                    raise AssertionError(
                        f"intermediate_idx {intermediate_idx} not in intermediate_idx_to_ph_idx for REMOVE_ADDITIONAL_INPUTS"
                    )
                inp_idx = intermediate_idx_to_ph_idx[intermediate_idx]
                self.saved_fw_additional_inputs.append(real_graph_inputs[inp_idx])
            else:
                new_output_node.append(node)

        fw_output_node.args = (tuple(fw_outputs) + tuple(new_output_node),)
        fw_gm.graph.lint()
        fw_gm.recompile()

