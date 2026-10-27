    name = f"{autograd_function.__name__}Generated"
    Generated = type(
        name,
        (torch.autograd.function._SingleLevelFunction,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
            "jvp": staticmethod(jvp),
            "setup_context": staticmethod(setup_context),
        },
    )
    return Generated


# wrap_outputs_maintaining_identity handles outputs from the vmap,
# backward (vjp), and jvp staticmethod. The way it distinguishes
# between the vmap case and the {backward, jvp} case is if the out_dims
# are specified or not.
#
# NB: we cannot use out_dims=None as the deciding factor. This because
# out_dims=None can still happen in the vmap staticmethod! What the
# user is saying in that case is that their output does not have a
# dimension that is being vmapped over, which is valid.
NO_OUT_DIMS = "not specified"


# NOTE [mark_dirty object identity check]
# autograd.Function's ctx.mark_dirty expect a returned input
# to have the same object identity as the input.
# Mode-only functorch will greatly simplify this logic.
def wrap_outputs_maintaining_identity(
    outputs: Any,
    unwrapped_inputs: Any,
    orig_inputs: Any,
    wrap_fn: Callable[..., Any],
    out_dims: Any = NO_OUT_DIMS,
) -> Any:
    flat_unwrapped_inputs = pytree.arg_tree_leaves(*unwrapped_inputs)
    flat_orig_inputs = pytree.arg_tree_leaves(*orig_inputs)

    unwrapped_input_to_orig_input = {
        id(unwrapped): orig
        for unwrapped, orig in zip(flat_unwrapped_inputs, flat_orig_inputs)
    }

    flat_outputs, spec = pytree.tree_flatten(outputs)
    result = []

    out_dims_specified = out_dims != NO_OUT_DIMS

    flat_out_dims = None
    if out_dims_specified:
        flat_out_dims = _broadcast_to_and_flatten(out_dims, spec)
        # _broadcast_to_and_flatten returns None if it is unable to broadcast.
        # TODO: update following link from master to stable once that's out
        if flat_out_dims is None:
            raise RuntimeError(
                f"The autograd.Function's vmap staticmethod returned an "
                f"incompatible (output, out_dims) tuple. "
                f"Expected out_dims={out_dims} "
                f"to be compatible with the structure of `output`. "
                f"out_dims has structure {pytree.tree_flatten(out_dims)[1]} "
                f"but output has structure {spec}. "
                f"For more details, please see "
                f"https://pytorch.org/docs/main/notes/extending.func.html"
            )

    for i, output in enumerate(flat_outputs):
        if not isinstance(output, torch.Tensor):
            result.append(output)
            continue
        if id(output) in unwrapped_input_to_orig_input:
            result.append(unwrapped_input_to_orig_input[id(output)])
            continue
        if out_dims_specified:
            if flat_out_dims is None:
                raise AssertionError(
                    "flat_out_dims must not be None when out_dims is specified"
                )
            result.append(wrap_fn(output, flat_out_dims[i]))
        else:
            result.append(wrap_fn(output))

    return pytree.tree_unflatten(result, spec)


# NOTE: [functorch vjp and autograd interaction]
# There's an edge case with the functorch vjp and autograd interaction
# that will eventually be fixed by mode-only functorch.
# The TL;DR is that there's no way to unwrap a dead GradTensorWrapper,
# so we (the framework) need to do it manually. Regular PyTorch operators
# automatically do so this is consistent.
#
# class MyExp(torch.autograd.Function):
#     @staticmethod
#     def forward(x):
#         return x.exp()
#
#     @staticmethod
#     def setup_context(ctx, inputs, output):
#         y = output
#         ctx.save_for_backward(y)
#
#     @staticmethod
#     def backward(gy):
#         y, = ctx.saved_tensors()
#         return MyMul.apply(gy, y)
#
# x = torch.randn([], requires_grad=True)
# gy = torch.randn([], requires_grad=True)
# _, vjp_fn = vjp(MySin.apply, x)
# result = vjp_fn(gy)
#
# MyMul is an autograd.Function that is not shown here.
# It saves a `y` for backward (since gy requires grad).
#
# in vjp_fn(gy), we get:
# > MyMul.apply(gy, GradTensorWrapper(y, level=dead))
# Because the y that is saved for backward by MyExp is a GradTensorWrapper
# but is now dead since we are outside the vjp context.
#
# PyTorch dispatcher operations, upon seeing a dead GradTensorWrapper,
# will automatically unwrap the GradTensorWrapper when applied.
# But since autograd.Function technically sits above the regular PyTorch
# dispatcher, it doesn't get this treatment. So we manually do
# the unwrapping to be consistent with regular PyTorch dispatcher operations.


class VmapInfo(NamedTuple):
    batch_size: int
    randomness: str


def has_overridden_vmap_rule(
    autograd_function: type[torch.autograd.Function],
) -> bool:
    return autograd_function.vmap is not torch.autograd.Function.vmap


def validate_vmap_returns_tuple_of_two_elements(result: Any) -> None:
    base_error_msg = (
        "Expected the vmap staticmethod to have two returns, an output "
        "and out_dims with pytree structure compatible with the output. "
    )
    if not isinstance(result, tuple):
        raise RuntimeError(base_error_msg + f"Got a {type(result)} instead")
    if not len(result) == 2:
        raise RuntimeError(base_error_msg + f"Got {len(result)} returns instead")


@custom_function_call.py_impl(TransformType.Vmap)
def custom_function_call_vmap(
    interpreter: VmapInterpreter,
    autograd_function: type[torch.autograd.Function],
    *operands: Any,
    **kwargs: Any,
) -> Any:
    if any(
        isinstance(val, torch.Tensor)
        for val in torch.utils._pytree.tree_flatten(kwargs)[0]
    ):
        raise NotImplementedError(
            f"Run vmap on autograd.Function with kwarg-only Tensor args. "
            f"Please do not pass kwarg-only Tensors to autograd.Function. "
            f"Got: {kwargs}"
        )

    if autograd_function.generate_vmap_rule:
        if has_overridden_vmap_rule(autograd_function):
            # TODO: Update link to stable once that's out
            # https://github.com/pytorch/pytorch/issues/92029
            raise RuntimeError(
                f"You tried to vmap over {autograd_function.__name__}, but "
                f"it has both generate_vmap_rule=True and an overridden vmap "
                f"staticmethod. Please set generate_vmap_rule=False or delete "
                f"the overridden vmap staticmethod to avoid ambiguity. "
                f"For more details, please see "
                f"https://pytorch.org/docs/main/notes/extending.func.html"
            )
        return custom_function_call_vmap_generate_rule(
            interpreter, autograd_function, *operands
        )

    if not has_overridden_vmap_rule(autograd_function):
        # TODO: Update link to stable once that's out
        # https://github.com/pytorch/pytorch/issues/92029
        raise RuntimeError(
            f"You tried to vmap over {autograd_function.__name__}, but "
            f"it does not have vmap support. Please override and implement the "
            f"vmap staticmethod or set generate_vmap_rule=True. "
            f"For more details, please see "
            f"https://pytorch.org/docs/main/notes/extending.func.html"
        )

    return custom_function_call_vmap_helper(
        interpreter, autograd_function.vmap, autograd_function, *operands, **kwargs
    )


def custom_function_call_vmap_helper(
    interpreter: VmapInterpreter,
    vmap_function: Callable[..., Any],
    op: Any,
    *operands: Any,
    **kwargs: Any,
) -> Any:
    current_level = interpreter.level()
    info = VmapInfo(
        batch_size=interpreter.batch_size(),
        randomness=interpreter.randomness(),
    )
    # We're either in the autograd.Function case (vmap staticmethod)
    # or the torch.library.register_vmap case.
    autograd_function_case = isinstance(op, torch.autograd.function.FunctionMeta)

    def lower_to_next() -> Any:
        if autograd_function_case:
            return interpreter.lower()
        else:
            return torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.FuncTorchBatched)
            )

    unwrapped_operands, in_dims = unwrap_batched(operands, current_level)
    # If none of the tensors are batched at the current level, then we skip the
    # current level. This saves the user from needing to handle this case in
    # their vmap staticmethod (and is consistent with our C++ batching rule API)
    if pytree.tree_all(lambda dim: dim is None, in_dims):
        with lower_to_next():
            if autograd_function_case:
                return custom_function_call(op, *operands)
            else:
                return op(*operands, **kwargs)

    with lower_to_next():
        result = vmap_function(info, in_dims, *unwrapped_operands, **kwargs)
    validate_vmap_returns_tuple_of_two_elements(result)
    unwrapped_output, out_dims = result

    # See NOTE [mark_dirty object identity check]
    def wrap_fn(output: torch.Tensor, out_dim: int | None) -> torch.Tensor:
        return (
            output
            if out_dim is None
            else _add_batch_dim(output, out_dim, current_level)
        )

    return wrap_outputs_maintaining_identity(
        unwrapped_output, unwrapped_operands, operands, wrap_fn, out_dims=out_dims
    )


def unpack_outputs(outputs: tuple[Any, ...]) -> tuple[Any, Any]:
    out_dims = outputs[-1]
    if isinstance(out_dims, tuple):
        outputs = outputs[:-1]
    else:
        outputs = outputs[0]
    return outputs, out_dims


def custom_function_call_vmap_generate_rule(
    interpreter: VmapInterpreter,
    autograd_function: type[torch.autograd.Function],
    *operands: Any,
) -> Any:
    unwrapped_operands, in_dims = unwrap_batched(operands, interpreter.level())
    vmapped_function = vmapify_autograd_function(
        autograd_function,
        in_dims,
        interpreter.batch_size(),
        interpreter.randomness(),
    )
    with interpreter.lower():
        outputs = custom_function_call(vmapped_function, *unwrapped_operands)

    if not isinstance(outputs, tuple):
        raise AssertionError(f"expected outputs to be a tuple, got {type(outputs)}")
    outputs, out_dims = unpack_outputs(outputs)
    return wrap_batched(outputs, out_dims, interpreter.level())


@custom_function_call.py_impl(TransformType.Functionalize)
def custom_function_call_functionalize(
    interpreter: FuncTorchInterpreter,
    autograd_function: type[torch.autograd.Function],
    generate_vmap_rule: bool,
    *operands: Any,
) -> Any:
    raise RuntimeError("NYI: Functionalize rule for custom_function_call")


def vmapify_autograd_function(
    autograd_function: type[torch.autograd.Function],
    in_dims: Any,
    batch_size: int,
    randomness: str,
) -> type[torch.autograd.Function]:
    def forward(*operands: Any) -> Any:
        outputs, out_dims = restore_vmap(
            autograd_function.forward, in_dims, batch_size, randomness
        )(*operands)
        if isinstance(outputs, torch.Tensor):
            return outputs, out_dims
        else:
            return *outputs, out_dims

    def setup_context(ctx: Any, inputs: Any, outputs: Any) -> None:
        outputs, out_dims = unpack_outputs(outputs)
        key = id(Generated)

        def inner(inputs: Any, outputs: Any) -> None:
            # wrapped_ctx.save_for_backward will:
            # - unwrap batchedtensors into (tensor, bdim)
            # - save_for_backward(*unwrapped_tensors)
            # - assign the bdims to wrapped_ctx._pt_saved_tensors_bdims
            wrapped_ctx = CtxCustomSave(ctx, current_level())
            autograd_function.setup_context(wrapped_ctx, inputs, outputs)

            # input_shapes are used for reductify later to reduce expanded gradients
            # to the correct shape.
            # See NOTE: [Why can't we rely on autograd to reduce expanded gradients?]
            # for more details
            input_shapes = tuple(
                inp.shape if isinstance(inp, torch.Tensor) else None for inp in inputs
            )
            if not hasattr(ctx, "_pt_input_shapes"):
                # pyrefly: ignore [implicit-any]
                ctx._pt_input_shapes = {}
            ctx._pt_input_shapes.update({key: input_shapes})

            if not hasattr(ctx, "_pt_saved_tensors_bdims_stack"):
                # pyrefly: ignore [implicit-any]
                ctx._pt_saved_tensors_bdims_stack = {}
            ctx._pt_saved_tensors_bdims_stack.update(
                {key: (wrapped_ctx._pt_saved_tensors_bdims)}
            )

        # See NOTE: [Why do we need to run setup_context under a vmap?]
        restore_vmap(
            inner,
            (in_dims, out_dims),
            batch_size,
            randomness,
        )(inputs, outputs)

        if not hasattr(ctx, "_pt_out_dims"):
            # pyrefly: ignore [implicit-any]
            ctx._pt_out_dims = {}
        ctx._pt_out_dims.update({key: out_dims})

    def jvp(ctx: Any, *tangents: Any) -> Any:
        key = id(Generated)

        def jvp_no_context(saved_tensors: Any, tangents: Any) -> Any:
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.jvp(wrapped_ctx, *tangents)

        tangent_in_dims = get_tangents_in_dims(in_dims, tangents)
        out_tangents, out_tangents_dims = restore_vmap(
            jvp_no_context,
            (ctx._pt_saved_tensors_bdims_stack[key], tangent_in_dims),
            batch_size,
            randomness,
        )(ctx.saved_tensors, tangents)

        result = reductify(
            out_tangents, out_tangents_dims, ctx._pt_out_dims[key], batch_size
        )
        if isinstance(result, torch.Tensor):
            return result, None
        else:
            return *result, None

    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        key = id(Generated)
        grad_outputs_ = grad_outputs[:-1]
        grad_outputs_in_dims = ctx._pt_out_dims[key]

        if not isinstance(grad_outputs_in_dims, tuple):
            grad_outputs_in_dims = (grad_outputs_in_dims,)

        grad_outputs_in_dims = tuple(
            in_dim if grad_output is not None else None
            for grad_output, in_dim in zip(grad_outputs_, grad_outputs_in_dims)
        )

        def backward_no_context(inputs: Any) -> Any:
            saved_tensors, grad_outputs = inputs
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.backward(wrapped_ctx, *grad_outputs)

        grad_ins, grad_ins_dims = restore_vmap(
            backward_no_context,
            ((ctx._pt_saved_tensors_bdims_stack[key], grad_outputs_in_dims),),
            batch_size,
            randomness,
        )((ctx.saved_tensors, grad_outputs_))
        result = reductify(
            grad_ins, grad_ins_dims, in_dims, batch_size, ctx._pt_input_shapes[key]
        )
        return result

    name = f"Vmapped{autograd_function.__name__}"
    Generated = type(
        name,
        (torch.autograd.Function,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
            "jvp": staticmethod(jvp),
            "setup_context": staticmethod(setup_context),
            "generate_vmap_rule": True,
        },
    )

    return Generated


# tangents might be None, so we need to replace
# the corresponding in_dims with None.
def get_tangents_in_dims(input_dims: Any, tangents: tuple[Any, ...]) -> Any:
    flat_in_dims, spec = pytree.tree_flatten(input_dims)
    flat_tangents = pytree.arg_tree_leaves(*tangents)
    result = [
        None if tangent is None else in_dim
        for in_dim, tangent in zip(flat_in_dims, flat_tangents)
    ]
    return pytree.tree_unflatten(result, spec)


# NOTE: [Why do we need to run setup_context under a vmap?]
# Consider the following autograd.Function
#
# class Sum(torch.autograd.Function):
#    @staticmethod
#    def forward(x):
#        return x.sum()
#    @staticmethod
#    def setup_context(ctx, inputs, outputs):
#        ctx.x_shape = inputs[0]
#    @staticmethod
#    def backward(ctx, gy):
#        return gy.expand(ctx.x_shape)
#
# x = torch.randn(B, 4)
# in_dims = 0
# vmap(Sum.apply, in_dims)(x)
#
# Let's assume for a moment that we didn't vmap setup_context in VmappedSum:
#
# class VmappedSum(torch.autograd.Function):
#    @staticmethod
#    def forward(x):
#        return vmap(Sum.forward, in_dims)(x)
#
#    @staticmethod
#    def setup_context(ctx, inputs, outputs):
#        Sum.setup_context(ctx, inputs, outputs)
#
#    @staticmethod
#    def backward(ctx, gy):
#        def backward_no_context(gy):
#            return gy.expand(ctx.x_shape)
#
#        dims = (0,)
#        gx = vmap(backward_no_context, dims)(gy)
#        return gx
#
# We end up saving [B, 4] as x_shape. In the backward, gy has shape [B],
# and we're doing:
#
# def backward_no_context(gy):
#     return gy.expand([B, 4])
#
# gx = vmap(backward_no_context, dims)(gy: "Tensor[B]")
#
# This gives us the wrong result (gx has shape [B, B, 4], but it should
# have shape [4]). Performing vmap over setup_context means the shape
# saved has shape [4] and leads to a correct result shape for gx.


# Wraps a ctx object. Forwards all attr accesses to the underlying object
# except for the attrs in _pt_attrs
class WrappedCtx:
    _pt_reserved_attrs: tuple[str, ...] = ("_pt_reserved_attrs", "_pt_inner_ctx")

    def __init__(self, ctx: Any) -> None:
        if not isinstance(ctx, WrappedCtx):
            reserved_attrs = type(self)._pt_reserved_attrs
            for name in reserved_attrs:
                if not hasattr(ctx, name):
                    continue
                raise RuntimeError(
                    f"PyTorch reserves the {reserved_attrs} field on ctx. "
                    "Please name your fields on ctx something else to avoid name "
                    "collision."
                )
        self._pt_inner_ctx = ctx

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pt_inner_ctx, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self)._pt_reserved_attrs:
            self.__dict__[name] = value
            return
        return setattr(self._pt_inner_ctx, name, value)


# Wraps ctx to create a new ctx object that overrides saved_tensors.
class CtxWithSavedTensors(WrappedCtx):
    _pt_reserved_attrs = ("_pt_new_saved_tensors", *WrappedCtx._pt_reserved_attrs)

    def __init__(self, ctx: Any, new_saved_tensors: Sequence[torch.Tensor]) -> None:
        super().__init__(ctx)
        self._pt_new_saved_tensors = new_saved_tensors

    @property
    def saved_tensors(self) -> Sequence[torch.Tensor]:
        return self._pt_new_saved_tensors


class CtxCustomSave(WrappedCtx):
    _pt_reserved_attrs = (
        "_pt_saved_tensors_bdims",
        "_pt_current_level",
        *WrappedCtx._pt_reserved_attrs,
    )

    def __init__(self, ctx: Any, current_level: int) -> None:
        super().__init__(ctx)
        self._pt_saved_tensors_bdims: tuple[Any, ...] = ()
        self._pt_current_level = current_level

    def save_for_backward(self, *tensors: torch.Tensor) -> None:
        unwrapped_tensors, bdims = unwrap_batched(tensors, self._pt_current_level)
        self._pt_inner_ctx.save_for_backward(*unwrapped_tensors)
        self._pt_saved_tensors_bdims = bdims

    def save_for_forward(self, *tensors: torch.Tensor) -> None:
        unwrapped_tensors, bdims = unwrap_batched(tensors, self._pt_current_level)
        self._pt_inner_ctx.save_for_forward(*unwrapped_tensors)
        self._pt_saved_tensors_bdims = bdims


def reductify(
    grad_input: torch.Tensor | tuple[torch.Tensor, ...],
    grad_input_bdim: int | tuple[int, ...],
    input_bdim: int | tuple[int, ...],
    batch_size: int,
    target_shape_without_bdim_to_reduce_to: Any = None,
) -> tuple[Any, ...]:
    if not isinstance(grad_input, tuple):
        grad_input = (grad_input,)
    if not isinstance(grad_input_bdim, tuple):
        grad_input_bdim = (grad_input_bdim,)
    if not isinstance(input_bdim, tuple):
        input_bdim = (input_bdim,)

    if target_shape_without_bdim_to_reduce_to is None:
        target_shape_without_bdim_to_reduce_to = len(grad_input) * (None,)
    result = tuple(
        reductify_leaf(gi, gi_bdim, i_bdim, batch_size, maybe_ishape)
        for gi, gi_bdim, i_bdim, maybe_ishape in zip(
            grad_input,
            grad_input_bdim,
            input_bdim,
            target_shape_without_bdim_to_reduce_to,
        )
    )
    return result


def reductify_leaf(
    grad_input: torch.Tensor | None,
    grad_input_bdim: int | None,
    input_bdim: int | None,
    batch_size: int,
    target_shape_without_bdim_to_reduce_to: Any = None,
) -> torch.Tensor | None:
    if grad_input is None:
        return None

    if grad_input_bdim is None and input_bdim is None:
        return grad_input

    if grad_input_bdim is not None and input_bdim is None:
        return grad_input.sum(grad_input_bdim)

    # NOTE: [Why can't we rely on autograd to reduce expanded gradients?]
    # For reverse-mode AD,
    # given a grad_input and input, it is valid for the user to return a
    # grad_input that has a broadcasted shape when compared to the input.
    # In this situation, autograd automatically reduces the grad_input to
    # the shape of the input.
    #
    # However, when input_bdim is not None, we have problems.
    #
    # [example 1]
    # grad_input: Tensor[3, 4], input: Tensor[B, 4]
    # We can expand grad_input to Tensor[B, 3, 4], but that isn't broadcastable
    # from [B, 4].
    #
    # [example 2]
    # grad_input: Tensor[3, B, 4], input: Tensor[B, 4]
    # We can swizzle grad_input to Tensor[B, 3, 4], but that isn't broadcastable
    # from [B, 4].
    #
    # This means that we need to also reduce the grad_input to the shape of the
    # input. This behavior is controlled by the `target_shape_without_bdim_to_reduce_to` flag;
    # if not-None then we do the reducing manually, otherwise, we do not do a reduction.
    if input_bdim is None:
        raise AssertionError("input_bdim must not be None")

    if grad_input_bdim is None:
        grad_input = grad_input.unsqueeze(input_bdim)
        new_shape = list(grad_input.shape)
        new_shape[input_bdim] = batch_size
        grad_input = grad_input.expand(new_shape)
        grad_input_bdim = input_bdim

    if target_shape_without_bdim_to_reduce_to is not None:
        return vmap(
            torch.Tensor.sum_to_size,
            in_dims=(grad_input_bdim, None),
            out_dims=input_bdim,
        )(grad_input, target_shape_without_bdim_to_reduce_to)

    if input_bdim != grad_input_bdim:
        grad_input = grad_input.movedim(grad_input_bdim, input_bdim)
    return grad_input


def autograd_function_forward_rewritten(
    original_forward: Callable[_P, _R],
    original_setup_context: Callable[..., Any],
) -> Callable[..., _R]:
    def new_forward(ctx: Any, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        output = original_forward(*args, **kwargs)
        original_setup_context(ctx, args, output)
        return output

    return new_forward


class AutogradFunctionApply(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("autograd_function_apply")

    def __call__(
        self,
        fwd: torch.fx.GraphModule,
        bwd: torch.fx.GraphModule,
        *fwd_args: Any,
        **fwd_kwargs: Any,
    ) -> Any:
        saved_values: Iterable[Any] | None = None
        dirty_idx = fwd_kwargs.get("dirty_idx", [])
        dirty_idx_set = set(dirty_idx)
        non_differentiable_idx = fwd_kwargs["non_differentiable_idx"]
        saved_for_backward_idx = fwd_kwargs["saved_for_backward_idx"]

        class ApplyTemplate(torch.autograd.Function):
            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                nonlocal saved_values

                # The Interpreter here is required to propagate metadata
                # from the dynamo graph body to the local_map graph body.
                # This is required for fx_traceback.annotate for work.
                output, saved_values = torch.fx.Interpreter(fwd).run(*args)

                # See Note [Activations with no version counter checks in eager]
                # Mark tensors that came from ctx.save_for_backward with metadata.
                # This allows AOT autograd to distinguish between tensors saved via
                # save_for_backward vs those stashed directly on ctx (e.g., ctx.x = x).
                from torch.fx.experimental.proxy_tensor import _get_proxies

                for idx, t in enumerate(saved_values):
                    if idx not in saved_for_backward_idx:
                        for proxy in _get_proxies(t):
                            proxy.node.meta["saved_tensor_with_no_vc_check"] = True

                return output

            @staticmethod
            def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
                def selected_outputs(idx: Iterable[int]) -> list[Any]:
                    idx_set = set(idx)
                    if isinstance(output, (tuple, list)):
                        return [x for i, x in enumerate(output) if i in idx_set]
                    return [output] if 0 in idx_set else []

                # If users call ctx.mark_dirty() in the original fwd function.
                if len(dirty_idx) > 0:
                    ctx.mark_dirty(*selected_outputs(dirty_idx_set))

                # If users call ctx.mark_non_differentiable() in the original fwd function.
                if len(non_differentiable_idx) > 0:
                    ctx.mark_non_differentiable(
                        *selected_outputs(non_differentiable_idx)
                    )

            @staticmethod
            def backward(ctx: Any, *grad: Any) -> Any:
                # The Interpreter here is required to propagate metadata
                # from the dynamo graph body to the local_map graph body.
                # This is required for fx_traceback.annotate for work.

                if saved_values is None:
                    raise AssertionError("saved_values must not be None")
                return torch.fx.Interpreter(bwd).run(*grad, *saved_values)

        return ApplyTemplate.apply(*fwd_args)


autograd_function_apply = AutogradFunctionApply()


class DynamoAutogradFunctionTraceHelper:
    @staticmethod
    def fwd_trace_helper(orig_fwd: Callable[_P, Any]) -> Callable[_P, Any]:
        # autograd.Function forward does more than just running the forward method. Most
        # of this logic is in C++. Here, we rewrite that functionality in python and let
        # Dynamo trace it.
        def inner(*args: _P.args, **kwargs: _P.kwargs) -> Any:
            with torch.no_grad():
                outs = orig_fwd(*args, **kwargs)

            ctx = args[0]
            dirty_tensors = getattr(ctx, "dirty_tensors", None)

            def is_dirty_tensor(tensor: torch.Tensor) -> bool:
                if dirty_tensors is None:
                    return False
                return any(
                    tensor is dirty_tensor
                    for dirty_tensor in dirty_tensors
                    if isinstance(dirty_tensor, torch.Tensor)
                )

            def is_tensor_arg(tensor: torch.Tensor) -> bool:
                return any(
                    arg is tensor for arg in args if isinstance(arg, torch.Tensor)
                )

            # Handle the case where if the input is passed on directly to the output, we call view_as
            # Refer to https://github.com/pytorch/pytorch/blob/main/torch/csrc/autograd/custom_function.cpp#L254
            if isinstance(outs, torch.Tensor):
                if is_tensor_arg(outs) and not is_dirty_tensor(outs):
                    return outs.view_as(outs)
                else:
                    return outs

            new_outs = []
            for out in outs:
                if isinstance(out, torch.Tensor):
                    if is_tensor_arg(out) and not is_dirty_tensor(out):
                        new_outs.append(out.view_as(out))
                    else:
                        new_outs.append(out)
                else:
                    new_outs.append(out)
            return tuple(new_outs)

            # TODO - there is missing functionality here, where
            # autograd.Function overwrites the requires_grad_ of the output
            # tensors depending on the `mark_non_differentiable`. Currently,
            # this is handled hackily in Dynamo, where we just overwrite the
            # variable trackers requires_grad. Refer to the function -
            # overwrite_tensor_vt_requires_grad

        return inner
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable


"""
Global flags for aot autograd
"""

import os
import sys
from typing import Literal, TYPE_CHECKING

from torch.utils._config_module import Config, install_config_module


# [@compile_ignored: debug]
_save_config_ignore = [
    # callable not serializable
    "joint_custom_pass",
    # callable configs with uuid() for caching, or raw callables
    "activation_memory_budget_runtime_estimator",
    "activation_memory_budget_solver",
]


# Converts torch rng ops to their functional philox rng equivalents. Note that
# we functionalize only CUDA rng ops today.
functionalize_rng_ops = False

# can be useful for debugging if we are incorrectly creating meta fake tensors
fake_tensor_allow_meta = os.environ.get("FAKE_ALLOW_META", "1") != "0"

# Enables optional asserts in hotpath code to check for errors. If
# you are seeing weird accuracy problems, try turning this on.
# This is currently off by default as it will harm tracing time,
# but it is on by default for aot_eager.
debug_assert = False

debug_partitioner = os.environ.get("AOT_PARTITIONER_DEBUG", "0") != "0"

# See # NOTE [Export custom triton op]
decompose_custom_triton_ops = True

static_weight_shapes = True

# See https://github.com/pytorch/pytorch/issues/141881
# Tells partitioner that parameters are free to save for backward.
treat_parameters_as_free_to_save = True

# Applies CSE to the graph before partitioning
cse = True

# When the partitioner's _size_of encounters a (Fake)ScriptObject in node
# metadata, it has no general way to know the object's true memory footprint:
# a ScriptObject may hold tensors internally. By default we raise rather than
# guess. Enabling this flag makes _size_of assume such objects are zero size,
# which unblocks dynamic-shapes compilation of models containing ScriptObject
# parameters (e.g. embedding-table configs from FBGEMM) at the cost of
# soundness. This is a temporary escape hatch until we have a principled way to
# probe the size of a (Fake)ScriptObject.
unsafe_treat_script_objects_as_zero_size = False

from torch._environment import is_fbcode


enable_autograd_cache: bool = Config(
    justknob="pytorch/remote_cache:enable_local_autograd_cache",
    env_name_force="TORCHINDUCTOR_AUTOGRAD_CACHE",
    default=True,
)

autograd_cache_allow_custom_autograd_functions: bool = Config(
    env_name_force="TORCHINDUCTOR_AUTOGRAD_CACHE_ALLOW_CUSTOM_AUTOGRAD", default=False
)

# For now, this is just for enabling unit testing in test_aot_autograd_cache.py
# We will either make this the default with AOTAutogradCache, or
# we'll just use it in the precompile flow. So there's no
# need to add env vars or make it configurable
bundled_autograd_cache: bool = False

bypass_autograd_cache_key: bool = False

# Whether or not to normalize placeholder names in graphs
# from dynamo in AOTAutogradCache
autograd_cache_normalize_inputs = not is_fbcode()

# Enable debug mode at first invocation to check if custom ops are valid.
# When enabled, this checks that custom operators don't violate aliasing constraints.
#
# check_custom_op_aliasing: Controls whether to run the custom op aliasing check at all.
#   - When True: The check runs on first invocation of compiled functions.
#   - When False: The check is skipped entirely.
#
# error_on_custom_op_aliasing: Controls behavior when a violation is detected.
#   Only has effect when check_custom_op_aliasing is True.
#   - When True: Raises RuntimeError on aliasing violations.
#   - When False: Emits UserWarning on aliasing violations.
#
# Deprecated: Custom ops returning aliased outputs is deprecated and will
# become an error in a future version of PyTorch. Currently error_on_custom_op_aliasing
# is True in CI unless explicitly overridden.
check_custom_op_aliasing = True
error_on_custom_op_aliasing: bool = Config(
    env_name_force="TORCHINDUCTOR_ERROR_ON_CUSTOM_OP_ALIASING",
    default=bool(os.getenv("CI")),
)


def remote_autograd_cache_default() -> bool | None:
    if os.environ.get("TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE") == "1":
        return True
    if os.environ.get("TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE") == "0":
        return False
    return None


enable_remote_autograd_cache = remote_autograd_cache_default()


# When AOTAutograd regenerates aliased graph outputs,
# attempt to use functionalization's view-replay logic
# before falling back to the autograd engine's view replay or as_strided.
# This can have some perf implications
# (although for many models this will not matter).
# (1) If you have many view ops chained together, replaying all of them
#     at runtime can have more overhead compared to a single as_strided call
# (2) If you are doing training, AsStridedBackward is quite slow,
#     and the individual view op backward formulas will likely be faster.
# (3) Some backends like XLA do not support as_strided

# Temporary hack: disable this flag for internal
# (needed to fix an internal issue while avoiding bumping XLA pin)
# eventually: either default this config to false completely
# once XLA pin update works,
# or default config to true and fix relevant bugs


# View replay is currently not compatible with AOTAutogradCache, since
# FunctionalTensors are not serializable. We'll need to make them
# serializable before enabling warm cache with this config turned on.
view_replay_for_aliased_outputs = not is_fbcode()

# Restricts the amount of computation AOTAutograd can do.
# NB: We have essentially disabled this heuristic now. However, this is kept
# here for now in case it's useful. Setting it low can artificially reduce the
# amount of recomputation AOTAutograd performs, although not in any kind of
# principled way.
max_dist_from_bw = 1000


# Bans recomputation of nodes that are reading from nodes that are far before
# the current node
ban_recompute_used_far_apart = True
# Breaks up long chain of fusible ops, as otherwise we can have an arbitrarily
# long chain of recomputation in the backwards pass.
ban_recompute_long_fusible_chains = True
# Bans recomputation of nodes that must be materialized in the backwards pass
# (used by a non-fusible node)
ban_recompute_materialized_backward = True
# Chooses to ban recomputation of nodes based off an allowlist. Setting it to
# False changes it to use a denylist. Main change is on operators like
# sort/pool/stuff that isn't cheap enough to be fusible for free but also isn't
# that expensive
ban_recompute_not_in_allowlist = True
# Chooses to ban recomputation of reductions. This is generally a good idea, as
# the result of reductions is generally very small but recomputing reductions in
# a fusion can be expensive.
ban_recompute_reductions = True
# Prevents the partitioner from ever saving views (i.e. always recompute them).
# Generally a good idea since views are free to recompute.
recompute_views = False
# Set this flag to enable considering non-built-in ops, including triton and custom
# ops, for recomputation during the knapsack optimization solver.
is_non_builtin_to_include = False

# Rematerialize AC nodes for graphs with forward+loss+backward in one graph.
# This optimization minimizes activation checkpoint node lifetimes by computing them
# just-in-time. For AC nodes only used in backward, they are deferred to backward region
# instead of being computed and saved in forward. This reduces peak memory usage.
# Note: This only applies to forward+loss+backward graphs where torch.autograd.grad is allowed
# in the graph. Joint graphs (standard AOTAutograd) use the partitioner instead.
remat_using_tags_for_fwd_loss_bwd_graph = True

# By default, the partitioner is purely trying to optimize for runtime (although
# it should always use less memory than eager)
# This knob controls the partitioner to make that tradeoff for you, choosing the
# fastest option that saves less activations than the memory budget.
# Specifically, 0.0 corresponds to the activation memory from applying
# activation checkpointing to the full compiled region, and 1.0 corresponds to
# the activation memory from the default runtime-optimized strategy.  So, 0.4
# would result in a strategy that saves 40% of the activations compared to the
# default strategy.
# It solves a 0-1 knapsack to find the minimum recompute necessary to stay below
# the activation memory budget.
# NOTE: This *cannot* be treated as
activation_memory_budget = 1.0

# This controls how we estimate the runtime when deciding what the cheapest
# operators to recompute are. The 3 options are
# "flops": Bases it off of the flop count provided by torch.utils.flop_counter
# "profile": Benchmarks each operator to come up with a runtime
# "testing": Returns 1 for everything
activation_memory_budget_runtime_estimator = "flops"

# This controls the solver used for the 0-1 knapsack. By default we use a
# quantized DP solution ("dp"). The other approaches are a "greedy", an "ilp"
# (which has a scipy dependency) and "dp_knapsack_sliding_hirschberg", which
# used memory-efficient quantized DP solution
activation_memory_budget_solver = "dp"

# This dumps out an SVG visualization of the expected runtime vs. activation
# memory tradeoffs for all memory budget values from 0 to 1 in increments of
# 0.5. See an example here:
# https://github.com/pytorch/pytorch/pull/126320#discussion_r1625104015
visualize_memory_budget_pareto = (
    os.environ.get("PARTITIONER_MEMORY_BUDGET_PARETO", "0") == "1"
)

# This controls the directory in which to dump the SVG plot with the pareto
# frontier of the activation checkpointing memory-vs-runtime tradeoffs.
memory_budget_pareto_dir = os.environ.get("PARTITIONER_MEMORY_BUDGET_PARETO_DIR")

# Sets all of the ban_recompute heuristics to False except ban_recompute_reductions
# Generally, this will probably result in some memory improvement, but at the
# cost of some performance
aggressive_recomputation = False

# activation offloading enablement (testing purpose)
enable_activation_offloading = False

# activation offloading with separate CUDA stream
activation_offload_separate_stream = False

# activation offloading wait sinking when using separate stream (fwd graph)
activation_offload_sink_wait = False

# activation reloading with prefetching when using separate streams (bwd graph)
activation_reload_prefetch = False

# CPU ↔ GPU bandwidth in GB/s, used to estimate transfer times for prefetch
# scheduling. This is hardware-specific and should be set by the user.
activation_offload_cpu_gpu_bw: float = 50.0

# If FakeTensor.data_ptr() should error.
# This option is independent of AOTAutograd and torch.compile, but our policy
# is to turn it off during torch.compile.
fake_tensor_allow_unsafe_data_ptr_access = True

# Unlifts effect tokens from the inputs/outputs in the traced graph and instead
# inserts make_token/sink_token calls in the graph to create tokens and then
# sink them at the end. Note that this means the graph is no longer functional
# which may lead to silent errors unless the backend knows how to handle the
# tokens.
unlift_effect_tokens = False

# NOTE: [The default layout constraint for custom operators.]
# This must be the name of one of the layout constraint tags
# (that is, one of {"needs_fixed_stride_order", "flexible_layout"}),
# If the custom op does not have a layout constraint tag already
# then we assume the following applies.
#
# This config is respected by Inductor and we recommend other backends also
# respect it.
# This config is in torch._functorch and not torch._inductor because it affects
# ProxyTensor tracing.
custom_op_default_layout_constraint: Literal[
    "needs_exact_strides", "needs_fixed_stride_order", "flexible_layout"
] = "needs_exact_strides"


# Run aot eager decomp partition with CrossRefFakeMode
# options = False, "all", "custom_ops"
fake_tensor_crossref = False

# This mode specifies that we should also keep track of the real
# tensor along with the fake tensor, and do real compute.  While
# seemingly this eliminates the whole point of fake tensors, there are
# two obvious use cases for it:
#
#   1. When users call item()/other data dependent operations,
#      if we propagate_real_tensors we are able to determine what
#      the true value is and keep going.
#
#   2. It can be useful for testing, when you want to see if the fake
#      and real tensors agree with each other.  (Note that there are
#      currently known inaccuracies in how we clone real tensors, that
#      would have to be tightened up for this to be useful in this
#      case.)
#
# Note that fake tensors are typically understood to be cheap to store
# indefinitely, so we tend to hold on to them longer than we would
# hold onto the real tensors.  So we also support you explicitly
# deallocating the real tensor associated with a fake tensor, at which
# point we will stop propagating real tensors.
#
# One more thing: when you provide a real tensor to fakeify, we will
# clone it, so that we can safely perform mutations on it if necessary.
# This will increase live memory usage.  This could potentially be
# optimized by using COW.  We also currently do not faithfully
# maintain autograd metadata on the real tensor; this is fine because
# AOTAutograd will only use the fake tensor to determine leafness/etc
# of tensors in question.
fake_tensor_propagate_real_tensors = False

# AOTDispatcher traces out a backward graph at the time of the forward pass.
# This flag controls whether or not that backward graph gets autocast behavior
# applied to it.
#
# The options are either:
# - "same_as_forward". We assume that the backward of the torch.compile'ed region
#   will be run under the same autocast context manager that the region was run
#   under. This is equivalent to running the following code in eager:
#
#   with torch.amp.autocast(...):
#       y = region(x)
#       ...
#       z.backward()
#
# - "off". We assume that the backward of the torch.compile'd region will
#   not be run under any autocast context managers.
#   This is equivalent to running the following code in eager:
#
#   with torch.amp.autocast(...):
#       y = region(x)
#       ...
#   z.backward()
#
# - or a list of kwargs dicts that represent an autocast context manager to turn
#   on during the backward pass.
#
#   e.g. [{"device_type": "cuda"}] is equivalent to running the following code in eager:
#
#   y = region(x)
#   ...
#   with torch.amp.autocast(device="cuda"):
#       z.backward()
backward_pass_autocast = "same_as_forward"

# This controls whether we collect donated buffers. This flag must be set
# False if a user wants to retain_graph=True for backward.
donated_buffer = not is_fbcode()

# Controls the default graph output format used by draw_graph.
# Most supported formats are defined here https://graphviz.org/docs/outputs/.
# The "dot" and "raw" formats write raw DOT text without invoking Graphviz.
torch_compile_graph_format = os.environ.get("TORCH_COMPILE_GRAPH_FORMAT", "svg")

# Valid only if fake_tensor_propagate_real_tensors = True; if a fake-real
# kernel mismatch is detected, bypasses by making a fake kernel from the
# real tensor outputs.
generate_fake_kernels_from_real_mismatches = False

# When there are device mismatches in FakeTensor device propagation,
# prefer a specific device type over others. This is particularly useful
# in full compiled mode where intermediate tensors with device mismatches
# represent only logical differences during compilation - these intermediate
# tensors will never physically materialize in the binary execution, so the
# device mismatch is not a real runtime concern. Enabling this allows the
# compiler to proceed with compilation by choosing the preferred device type
# for consistency. For example, set to "mtia" to prefer MTIA devices over
# CPU, or "cuda" to prefer CUDA devices over CPU.
fake_tensor_prefer_device_type: str | None = None

# CUDAGraph safe run_with_rng functionalization.
graphsafe_rng_functionalization = True

# Whether or not to eagerly compile the backward
# used by AOT compile and other settings
# TODO: once AOT compile calls aot autograd directly instead of
# through compile_fx, we can remove this
force_non_lazy_backward_lowering = False

# only for testing, used to turn functionalization off in AOTDispatcher
_test_disable_functionalization = True

# Error on BypassAOTAutogradCache instead of just a warning
# Used for tests
strict_autograd_cache = False

# Note [Recomputing collectives in the partitioner]
# The purpose of this config is as follows:
# - We have many passes in the compiler (min-cut partitioning, DCE, etc)
#   which can reorder or delete duplicate nodes in the graph
# - If any of these passes reorder/delete/duplicate a collective
#   in a setting where the compiler is being run independently on multiple
#   ranks, we run the risk that the compiler will make a different decision on
#   different ranks, resulting in a NCCL hang when using torch.compile
# To handle this, we will (by default) ensure that collectives are not modified
# by the compiler.
#
# A few examples:
# - don't dead-code-eliminate collectives
#   (in case they are dead on rank i but not rank j)
# - don't recompute collectives in partitioning
#   (in case we recompute on rank i but not rank j)
#
# Today this flag **must** be set to false, but eventually
# we want the option to set it to true.
# In order to potentially optimize collectives, we'll need the compiler
# to broadcast information across ranks at compile time to ensure
# that any decisions on collectives are made consistently.
unsafe_allow_optimization_of_collectives = False

# See Note [AOTAutograd Tangent Subclassness for mutated inputs]
# TODO(ivankobzarev): Remove this config, being able to deduce it compile time.
disable_guess_zero_tangent_for_mutated_input_subclass = False

# See Note [Tangents memory format]
# By default tangents strideness is guessed to be contiguous,
# At runtime non contiguous tangents will be coerced to be contiguous.
# This config changes this guess for tangents strides to be the same as outputs.
# TODO(ivankobzarev): Remove this config once extra memory usage is investigated.
guess_tangent_strides_as_outputs = not is_fbcode()


# This is a temporary config to ensure all ranks take the same decision in the partitioner
# it will ultimately be removed once we share size_hints across ranks through compiler collectives
_sync_decision_cross_ranks = False

# By default apply inlined saved_tensors_hooks only for "donated" buffers.
# "donated" buffers are invisible to the user, they are intermediates of the forward graph.
# Applying saved tensors hooks for memory optimizations only for intermediates
# guarantees that original saved tensors could be deallocated.
# This config enables saved_tensors_hooks are applied for **all** saved tensors,
# that could include inputs, parameters, outputs.
# "donated" - applied only to saved intermediates of the graph
# "no_static" - applied to all saved but not "static"
# (this includes parameters and user marked as static)
# "all" - no filtering, everything saved for backward.
saved_tensors_hooks_filtering_mode = "donated"


# This callback is invoked on the joint graph before partitioning
joint_custom_pass: Callable = None  # type: ignore[assignment]

force_autograd_cache = False

# Note [Selective Decomposition]
# This config allows selective decomposition of certain operators in the graph.
# When True, it does NOT decompose any nodes, except those nodes that users explicitly
# annotated with regional inductor compile. Please read torch.fx.passes.regional_inductor
# on to explicitly annotate. This is currently only used by inductor lite mode.
selective_decompose: bool = False

# Complex Support
# This config disallows decomposition of complex-valued Tensors using
# `torch._subclasses.complex_tensor.ComplexTensor` by decomposing everything into
# real-valued operations, passing through the regular pipeline as necessary,
# then converting back to a regular tensor.
enable_complex_wrapper: bool = False


if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F403


# adds patch, save_config, invalid config checks, etc
install_config_module(sys.modules[__name__])
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains pre-dispatch wrappers for functorch operations
that enable proper tracing in PT2 non-strict export/compile fx graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._C import (
    _enter_dual_level as _enter_dual_level_impl,
    _exit_dual_level as _exit_dual_level_impl,
)
from torch._C._functorch import (
    _add_batch_dim as _add_batch_dim_impl,
    _jvp_decrement_nesting as _jvp_decrement_nesting_impl,
    _jvp_increment_nesting as _jvp_increment_nesting_impl,
    _remove_batch_dim as _remove_batch_dim_impl,
    _unwrap_for_grad as _unwrap_for_grad_impl,
    _vmap_decrement_nesting as _vmap_decrement_nesting_impl,
    _vmap_increment_nesting as _vmap_increment_nesting_impl,
)
from torch._VF import (  # type: ignore[attr-defined]
    _make_dual as _make_dual_impl,
    _unpack_dual as _unpack_dual_impl,
)


if TYPE_CHECKING:
    import threading


def _add_batch_dim(self: torch.Tensor, batch_dim: int, level: int) -> torch.Tensor:
    """
    Thin wrapper around torch._C._add_batch_dim that is used to proxy in
    PT2 export/compile fx graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()
    batch_dim = self.ndim + batch_dim if batch_dim < 0 else batch_dim

    if mode:
        return torch.overrides.handle_torch_function(
            _add_batch_dim, (self,), self, batch_dim, level
        )

    res = _add_batch_dim_impl(self, batch_dim, level)
    return res


def _remove_batch_dim(
    self: torch.Tensor, level: int, batch_size: int, out_dim: int
) -> torch.Tensor:
    """
    Thin wrapper around torch._C._remove_batch_dim that is used to proxy in
    PT2 export/compile fx graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _remove_batch_dim, (self,), self, level, batch_size, out_dim
        )

    res = _remove_batch_dim_impl(self, level, batch_size, out_dim)
    return res


def _vmap_increment_nesting(batch_size: int, randomness: str) -> int:
    """
    Thin wrapper around torch._C._vmap_increment_nesting that is used
    to proxy in export/compile graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _vmap_increment_nesting, (batch_size,), batch_size, randomness
        )
    res = _vmap_increment_nesting_impl(batch_size, randomness)
    return res


def _vmap_decrement_nesting() -> int:
    """
    Thin wrapper around torch._C._vmap_increment_nesting that is used
    to proxy in export/compile graph
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _vmap_decrement_nesting,
            (),
        )
    return _vmap_decrement_nesting_impl()


# Global variables for lazy_load_decompositions
DECOMPOSITIONS_LOADED: bool = False
DECOMPOSITIONS_LOCK: threading.Lock | None = None
VMAP_DECOMPOSITIONS_LIB: torch.library.Library | None = None


def lazy_load_decompositions() -> None:
    """
    Lazy loading of vmap decompositions with pre-dispatch support.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(lazy_load_decompositions, ())

    global DECOMPOSITIONS_LOADED, DECOMPOSITIONS_LOCK, VMAP_DECOMPOSITIONS_LIB

    if DECOMPOSITIONS_LOADED:
        return

    # Initialize lock if needed
    if DECOMPOSITIONS_LOCK is None:
        import threading

        DECOMPOSITIONS_LOCK = threading.Lock()

    with DECOMPOSITIONS_LOCK:
        if DECOMPOSITIONS_LOADED:
            return

        import os

        if not (os.environ.get("PYTORCH_JIT", "1") == "1" and __debug__):
            DECOMPOSITIONS_LOADED = True
            return

        # use an alternate way to register an operator into the decomposition table
        # _register_jit_decomposition doesn't work for some operators, e.g. addr,
        #  because the Tensor types generated cannot be unioned by torchscript
        # decomp should be type OpOverload
        VMAP_DECOMPOSITIONS_LIB = torch.library.Library(
            "aten", "IMPL", "FuncTorchBatched"
        )

        from torch._decomp import decomposition_table

        def _register_python_decomposition_vmap(decomp: torch._ops.OpOverload) -> None:
            if VMAP_DECOMPOSITIONS_LIB is None:
                raise AssertionError("VMAP_DECOMPOSITIONS_LIB must not be None")
            if decomp in decomposition_table:
                VMAP_DECOMPOSITIONS_LIB.impl(decomp, decomposition_table[decomp])
            else:
                raise RuntimeError(f"could not find decomposition for {decomp}")

        _register_python_decomposition_vmap(torch.ops.aten.mse_loss_backward.default)
        _register_python_decomposition_vmap(
            torch.ops.aten.smooth_l1_loss_backward.default
        )
        _register_python_decomposition_vmap(torch.ops.aten.huber_loss_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss_forward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_forward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_backward.default)
        _register_python_decomposition_vmap(torch.ops.aten.addr.default)

        DECOMPOSITIONS_LOADED = True


def _make_dual(
    tensor: torch.Tensor, tangent: torch.Tensor, *, level: int = 0
) -> torch.Tensor:
    """
    Thin wrapper around torch._VF._make_dual that is used to proxy in
    PT2 export/compile fx graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _make_dual, (tensor, tangent), tensor, tangent, level=level
        )

    return _make_dual_impl(tensor, tangent, level=level)


def _unpack_dual(
    tensor: torch.Tensor, *, level: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Thin wrapper around torch._VF._unpack_dual that is used to proxy in
    PT2 export/compile fx graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _unpack_dual, (tensor,), tensor, level=level
        )

    return _unpack_dual_impl(tensor, level=level)


def _jvp_increment_nesting() -> int:
    """
    Thin wrapper around torch._C._functorch._jvp_increment_nesting that is
    used to proxy in export/compile graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _jvp_increment_nesting,
            (),
        )
    return _jvp_increment_nesting_impl()


def _jvp_decrement_nesting() -> int:
    """
    Thin wrapper around torch._C._functorch._jvp_decrement_nesting that is
    used to proxy in export/compile graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _jvp_decrement_nesting,
            (),
        )
    return _jvp_decrement_nesting_impl()


def _unwrap_for_grad(tensor: torch.Tensor, level: int) -> torch.Tensor:
    """
    Thin wrapper around torch._C._functorch._unwrap_for_grad that is used
    to proxy in PT2 export/compile fx graph for functorch transforms (grad, vjp, jvp).
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(
            _unwrap_for_grad, (tensor,), tensor, level
        )

    return _unwrap_for_grad_impl(tensor, level)


def _enter_dual_level() -> int:
    """
    Thin wrapper around torch._C._enter_dual_level that is used to proxy in
    PT2 export/compile fx graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(_enter_dual_level, ())
    return _enter_dual_level_impl()


def _exit_dual_level(*, level: int) -> None:
    """
    Thin wrapper around torch._C._exit_dual_level that is used to proxy in
    PT2 export/compile fx graph for forward-mode AD.
    """
    from torch._export.utils import _maybe_find_pre_dispatch_tf_mode_for_export

    mode = _maybe_find_pre_dispatch_tf_mode_for_export()

    if mode:
        return torch.overrides.handle_torch_function(_exit_dual_level, (), level=level)
    return _exit_dual_level_impl(level=level)
<|endoftext|>from __future__ import annotations

import copy
import functools
import hashlib
import heapq
import itertools
import logging
import math
import operator
import os
import os.path
import re
import warnings
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any, cast, TYPE_CHECKING

import sympy

import torch
import torch._inductor.inductor_prims
import torch.distributed
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters, is_node_meta_valid
from torch._functorch._activation_checkpointing.ac_logging_utils import (
    create_structured_trace_for_min_cut_info,
)
from torch._functorch._aot_autograd.streams import _SYNC_OPS
from torch._functorch._aot_autograd.utils import is_with_effects
from torch._inductor import config as inductor_config
from torch._inductor.custom_graph_pass import (
    CustomKnapsackSolver,
    CustomRuntimeEstimator,
)
from torch._inductor.fx_passes.control_dependencies import (
    control_deps as control_deps_hop,
    get_subgraph_name,
)
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import is_opaque_value
from torch._library.utils import is_builtin
from torch._logging import LazyString, trace_structured
from torch._logging._internal import trace_log
from torch._subclasses.fake_tensor import extract_tensor_metadata
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import is_sym_node, py_sym_types
from torch.fx.experimental.sym_node import magic_methods, method_to_operator, SymNode
from torch.fx.experimental.symbolic_shapes import (
    _get_placeholder_expr,
    find_symbol_binding_fx_nodes,
    free_symbols,
    is_symbol_binding_fx_node,
    is_symbolic,
    optimization_hint,
    statically_known_false,
    statically_known_true,
)
from torch.fx.passes import graph_drawer
from torch.fx.traceback import _get_memory_budget_annotation
from torch.utils._ordered_set import OrderedSet
from torch.utils.checkpoint import CheckpointPolicy

from . import config
from ._activation_checkpointing.graph_info_provider import GraphInfoProvider
from ._activation_checkpointing.knapsack import (
    dp_knapsack,
    dp_knapsack_sliding_hirschberg,
    greedy_knapsack,
    ilp_knapsack,
)
from ._activation_checkpointing.knapsack_evaluator import KnapsackEvaluator
from ._aot_autograd.descriptors import (
    AOTOutput,
    SavedForBackwardsAOTOutput,
    SavedForBackwardsNoVcCheckAOTOutput,
)
from ._aot_autograd.functional_utils import _is_functional_graph
from ._aot_autograd.graph_compile import is_opaque_node
from ._aot_autograd.logging_utils import get_aot_graph_name
from ._aot_autograd.utils import (
    _is_bwd_seed_offset,
    _is_fwd_seed_offset,
    _is_primal,
    _is_tangent,
    get_default_generator,
    get_device_rng_state,
    supports_graphsafe_rng,
)
from .compile_utils import fx_graph_cse, get_aten_target, raise_getitems


if TYPE_CHECKING:
    import networkx as nx


AOT_PARTITIONER_DEBUG: bool = config.debug_partitioner
log: logging.Logger = logging.getLogger(__name__)

aten = torch.ops.aten
prims = torch.ops.prims


@dataclass
class OpTypes:
    """Class for keeping track of different operator categories"""

    fusible_ops: OrderedSet[Callable[..., Any]]
    compute_intensive_ops: OrderedSet[Callable[..., Any]]
    random_ops: OrderedSet[Callable[..., Any]]
    view_ops: OrderedSet[Callable[..., Any]]
    recomputable_ops: OrderedSet[Callable[..., Any]]

    def is_fusible(self, node: fx.Node) -> bool:
        return get_aten_target(node) in self.fusible_ops

    def is_compute_intensive(self, node: fx.Node) -> bool:
        return get_aten_target(node) in self.compute_intensive_ops

    def is_random(self, node: fx.Node) -> bool:
        return get_aten_target(node) in self.random_ops

    def is_view(self, node: fx.Node) -> bool:
        return get_aten_target(node) in self.view_ops

    def is_recomputable(self, node: fx.Node) -> bool:
        return get_aten_target(node) in self.recomputable_ops


@dataclass
class NodeInfo:
    # Be careful about iterating over these explicitly, as their order may not
    # be deterministic
    inputs: list[fx.Node]
    _required_fw_nodes: OrderedSet[fx.Node]
    required_bw_nodes: OrderedSet[fx.Node]
    tangents_closure: OrderedSet[fx.Node]
    unclaimed_nodes: OrderedSet[fx.Node]
    fw_order: dict[fx.Node, int]
    # Effectively maps to which of our primals are parameters
    static_lifetime_input_nodes: OrderedSet[fx.Node]

    @functools.cached_property
    def required_fw_nodes(self) -> list[fx.Node]:
        return sorted(
            (n for n in self._required_fw_nodes), key=lambda n: self.fw_order[n]
        )

    def is_required_fw(self, n: fx.Node) -> bool:
        return n in self._required_fw_nodes

    def is_required_bw(self, n: fx.Node) -> bool:
        return n in self.required_bw_nodes

    def is_unclaimed(self, n: fx.Node) -> bool:
        return n in self.unclaimed_nodes

    def get_fw_order(self, n: fx.Node) -> int:
        if n not in self._required_fw_nodes:
            raise AssertionError(f"Node {n} not in fw nodes!")
        return self.fw_order[n]


@dataclass
class MinCutOptions:
    ban_if_used_far_apart: bool
    ban_if_long_fusible_chains: bool
    ban_if_materialized_backward: bool
    ban_if_not_in_allowlist: bool
    ban_if_reduction: bool


def must_recompute(node: fx.Node) -> bool:
    return node.meta.get("recompute", None) in [
        CheckpointPolicy.MUST_RECOMPUTE,
        CheckpointPolicy.PREFER_RECOMPUTE,
    ]


def get_schema_arg_idx(target: object, arg_name: str) -> int | None:
    """Return the positional schema index for an operator argument name."""
    if not isinstance(target, torch._ops.OpOverload):
        return None
    for idx, arg in enumerate(target._schema.arguments):
        if arg.name == arg_name:
            return idx
    return None


def is_nonzero_dropout_sdpa(node: fx.Node) -> bool:
    """Return True unless SDPA-family dropout is statically known to be disabled."""
    target = node.target
    if not isinstance(target, torch._ops.OpOverload):
        return True

    dropout_arg_idx = get_schema_arg_idx(target, "dropout_p")
    if dropout_arg_idx is None:
        return True

    dropout_arg = target._schema.arguments[dropout_arg_idx]
    if len(node.args) > dropout_arg_idx:
        dropout_p = node.args[dropout_arg_idx]
    elif "dropout_p" in node.kwargs:
        dropout_p = node.kwargs["dropout_p"]
    elif dropout_arg.default_value is not None:
        dropout_p = dropout_arg.default_value
    else:
        return True

    return not statically_known_true(cast(bool | torch.SymBool, dropout_p == 0.0))


def is_rng_op(node: fx.Node) -> bool:
    """Return True when a node's seeded nondeterminism cannot be statically ruled out."""
    if not (
        hasattr(node.target, "tags")
        and torch.Tag.nondeterministic_seeded in node.target.tags
    ):
        return False

    if get_schema_arg_idx(node.target, "dropout_p") is not None:
        return is_nonzero_dropout_sdpa(node)

    return True


def _is_assert_only_symbool(node: fx.Node) -> bool:
    return (
        isinstance(node.meta.get("val"), torch.SymBool)
        and len(node.users) > 0
        and all(
            user.target is torch.ops.aten._assert_scalar.default for user in node.users
        )
    )


def has_recomputable_ops(fx_g: fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if must_recompute(node):
            return True
    return False


def has_recomputable_rng_ops(fx_g: fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if must_recompute(node) and is_rng_op(node):
            return True
    return False


def sym_node_size(node: fx.Node) -> int:
    if isinstance(node.meta["val"], (torch.SymInt, torch.SymBool)):
        return 1
    if not isinstance(node.meta["val"], torch.SymFloat):
        raise AssertionError(
            f"expected node.meta['val'] to be SymFloat, got {type(node.meta['val'])}"
        )
    return 4


class InvalidNodeBase:
    def __repr__(self) -> str:
        return "Invalid Node"


# Run DCE while overriding the definition of is_impure_node
def is_not_collective(node: fx.Node) -> bool:
    return getattr(node.target, "namespace", None) != "_c10d_functional"


InvalidNode = InvalidNodeBase()


def _get_ho_op_original_input(getitem_node: fx.Node) -> fx.Node | None:
    """Given a getitem node, check if it extracts from a higher-order op
    that has kwargs mapping the key back to an original input.

    Returns the original input node if found, None otherwise.
    """
    if getitem_node.target != operator.getitem:
        return None
    ho_result = getitem_node.args[0]
    key = getitem_node.args[1]
    if not isinstance(ho_result, fx.Node) or ho_result.op != "call_function":
        return None
    if "kwargs" not in ho_result.kwargs:
        return None
    kwargs = ho_result.kwargs["kwargs"]
    # pyrefly: ignore [not-iterable, unsupported-operation]
    if key not in kwargs:
        return None
    # pyrefly: ignore [bad-index, unsupported-operation]
    original_input = kwargs[key]
    if isinstance(original_input, fx.Node):
        return original_input
    return None


def _is_copy_node_bw_only(node: fx.Node) -> fx.Node | None:
    """Check if node is a view/reshape of a higher-order op output that aliases an input.

    Returns the original input node from the higher-order op's kwargs if the pattern
    matches, None otherwise.
    """
    if node.target not in (torch.ops.aten.view.default, torch.ops.aten.reshape.default):
        return None
    source = node.args[0]
    if not isinstance(source, fx.Node):
        return None
    return _get_ho_op_original_input(source)


def _find_input_for_invalid_output(
    node: fx.Node,
    env: dict[fx.Node, Any],
) -> fx.Node | None:
    """Try to find a valid input replacement for an invalid forward output.

    This handles cases where a forward output depends on backward nodes but
    semantically aliases an input. For example, a view of a getitem from a
    triton kernel that mutates a buffer in backward, or a direct getitem from
    such a higher-order op. The original input may be a primal or a valid
    intermediate node already present in the forward graph.
    """
    # Pattern 1: view/reshape(getitem(ho_op, key)) -> ho_op.kwargs["kwargs"][key]
    original_input = _is_copy_node_bw_only(node)
    if (
        original_input is not None
        and original_input in env
        and not isinstance(env[original_input], InvalidNodeBase)
    ):
        return env[original_input]
    # Pattern 2: getitem(ho_op, key) -> ho_op.kwargs["kwargs"][key]
    original_input = _get_ho_op_original_input(node)
    if (
        original_input is not None
        and original_input in env
        and not isinstance(env[original_input], InvalidNodeBase)
    ):
        return env[original_input]
    return None


def _control_deps_has_sync_op(node: fx.Node) -> bool:
    """Check if a control_deps node's subgraph has a sync operation."""
    sg_node = node.args[1]
    if not (
        isinstance(sg_node, fx.Node)
        and sg_node.op == "get_attr"
        and isinstance(sg_node.target, str)
    ):
        raise AssertionError(f"expected get_attr subgraph node, got {sg_node}")
    owning_mod = node.graph.owning_module
    if owning_mod is None:
        raise AssertionError("expected graph to have an owning_module")
    sg_mod = getattr(owning_mod, sg_node.target)
    if not isinstance(sg_mod, fx.GraphModule):
        raise AssertionError(f"expected GraphModule, got {type(sg_mod)}")
    return any(
        n.op == "call_function" and n.target in _SYNC_OPS for n in sg_mod.graph.nodes
    )


def _build_see_through_map(cd_node: fx.Node) -> dict[int, fx.Node]:
    """Build a mapping from getitem indices to underlying dep nodes.

    For a control_deps(additional_deps, subgraph, *deps), getitem index k
    (k >= 1) corresponds to deps[k-1] which is args[k+1].
    """
    deps = cd_node.args[2:]
    return {i + 1: dep for i, dep in enumerate(deps) if isinstance(dep, fx.Node)}


def _try_clone_subgraph_filtering_deps(
    original_sg_mod: fx.GraphModule,
    valid_dep_indices: list[int],
) -> tuple[fx.Graph, dict[int, int]] | None:
    """Clone a control_deps subgraph, keeping only valid-dep placeholders.

    Preserves any call_function nodes (the side-effecting operation) as long
    as all placeholders they reference are in the valid set.  Returns
    (new_graph, idx_remap) where idx_remap maps old getitem indices to new
    ones, or None if the operation uses an invalid dep.
    """
    orig_phs = original_sg_mod.graph.find_nodes(op="placeholder")
    valid_dep_set = OrderedSet(valid_dep_indices)
    dropped_phs = OrderedSet(
        [ph for i, ph in enumerate(orig_phs) if i not in valid_dep_set]
    )

    for n in original_sg_mod.graph.nodes:
        if n.op == "call_function":
            if any(inp in dropped_phs for inp in n.all_input_nodes):
                return None

    sg = fx.Graph()
    sg_env: dict[fx.Node, fx.Node] = {}
    for n in original_sg_mod.graph.nodes:
        if n.op == "output" or n in dropped_phs:
            continue
        sg_env[n] = sg.node_copy(n, lambda x, _env=sg_env: _env[x])

    orig_out_arg = original_sg_mod.graph.output_node().args[0]
    if isinstance(orig_out_arg, tuple):
        sg.output(
            tuple(
                sg_env[item] if isinstance(item, fx.Node) else item
                for item in orig_out_arg
                if not isinstance(item, fx.Node) or item in sg_env
            )
        )
    elif isinstance(orig_out_arg, fx.Node) and orig_out_arg in sg_env:
        sg.output(sg_env[orig_out_arg])
    else:
        sg.output(orig_out_arg)

    preserved_op = any(n.op == "call_function" for n in sg.nodes)
    idx_remap: dict[int, int] = {
        old_i + 1: new_j + 1 for new_j, old_i in enumerate(valid_dep_indices)
    }
    if preserved_op:
        idx_remap[0] = 0
    return sg, idx_remap


def _rebuild_control_deps_for_extract(
    node: fx.Node,
    env: dict[fx.Node, Any],
    new_graph: fx.Graph,
) -> tuple[fx.Node, dict[int, int]] | None:
    """Rebuild a control_deps node with only valid deps for graph extraction.

    When a control_deps node mixes valid (forward) and invalid (backward) args,
    this creates a new control_deps in new_graph with only the valid deps and a
    matching subgraph.  Returns (new_node, idx_remap) where idx_remap maps old
    getitem indices to new getitem indices, or None if no valid deps remain.

    additional_deps (args[0]) and deps (args[2:]) are filtered independently --
    additional_deps are ordering-only constraints and may reference nodes that
    are not in the deps set.
    """
    deps = node.args[2:]
    valid_dep_indices = [
        i
        for i, dep in enumerate(deps)
        if isinstance(dep, fx.Node) and not isinstance(env.get(dep), InvalidNodeBase)
    ]
    if not valid_dep_indices:
        return None

    # pyrefly: ignore [bad-index]
    new_deps = tuple(env[deps[i]] for i in valid_dep_indices)

    additional_deps = node.args[0]
    if not isinstance(additional_deps, (tuple, list)):
        raise AssertionError(
            f"expected tuple/list additional_deps, got {type(additional_deps)}"
        )
    valid_additional_deps = tuple(
        env[d]
        for d in additional_deps
        if isinstance(d, fx.Node) and not isinstance(env.get(d), InvalidNodeBase)
    )

    owning_mod = node.graph.owning_module
    if owning_mod is None:
        raise AssertionError("expected graph to have an owning_module")

    sg_node = node.args[1]
    if not (
        isinstance(sg_node, fx.Node)
        and sg_node.op == "get_attr"
        and isinstance(sg_node.target, str)
    ):
        raise AssertionError(f"expected get_attr subgraph node, got {sg_node}")
    original_sg_mod = getattr(owning_mod, sg_node.target)
    if not isinstance(original_sg_mod, fx.GraphModule):
        raise AssertionError(f"expected GraphModule, got {type(original_sg_mod)}")

    result = _try_clone_subgraph_filtering_deps(original_sg_mod, valid_dep_indices)
    if result is not None:
        sg, idx_remap = result
    else:
        sg = fx.Graph()
        sg_placeholders = [sg.placeholder(f"dep_{i}") for i in range(len(new_deps))]
        sg.output((None, *sg_placeholders))
        idx_remap = {
            old_i + 1: new_j + 1 for new_j, old_i in enumerate(valid_dep_indices)
        }

    sg_name = get_subgraph_name(owning_mod, "cd_extract")
    setattr(owning_mod, sg_name, fx.GraphModule(torch.nn.Module(), sg))
    new_sg_node = new_graph.get_attr(sg_name)

    new_cd = new_graph.create_node(
        "call_function",
        control_deps_hop,
        args=(valid_additional_deps, new_sg_node, *new_deps),
        name=node.name,
    )
    new_cd.meta = node.meta.copy()
    if "val" in node.meta and isinstance(node.meta["val"], tuple):
        orig_val = node.meta["val"]
        new_cd.meta["val"] = (orig_val[0],) + tuple(
            orig_val[i + 1] for i in valid_dep_indices if i + 1 < len(orig_val)
        )

    return new_cd, idx_remap


def _extract_graph_with_inputs_outputs(
    joint_graph: fx.Graph,
    inputs: list[fx.Node],
    outputs: list[fx.Node],
    outputs_descs: list[AOTOutput],
    subgraph: str | None = None,
    ignore_must_be_in_fw_bw: bool = False,
) -> fx.Graph:
    """
    Given a graph, extracts out a subgraph that takes the specified nodes as
    inputs and returns the specified outputs.

    This includes specifying non-placeholder nodes as inputs.

    The general strategy is to initialize all inputs with proxies as we
    encounter them, and trace through the graph, only keeping values which take
    in valid proxies. Then, all dead code is eliminated.
    """
    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}
    # For control_deps nodes with mixed valid/invalid args, we create a new
    # control_deps with only the valid deps.  This dict maps the original
    # control_deps node to a {old_getitem_idx: new_getitem_idx} remapping so
    # downstream getitem nodes can be rewritten to the correct indices.
    _control_deps_idx_remap: dict[fx.Node, dict[int, int]] = {}
    # For forward sync control_deps eliminated during backward extraction,
    # maps the control_deps node to {getitem_idx: joint_graph_dep_node}.
    # Getitems resolve directly to env[dep_node], bypassing the control_deps.
    _control_deps_see_through: dict[fx.Node, dict[int, fx.Node]] = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in inputs:
        new_node = new_graph.placeholder(node.name)
        # Can't use node_copy here as we may be turning previous call_function into placeholders
        new_node.meta = node.meta
        # pyrefly: ignore [unsupported-operation]
        env[node] = new_node

    for node in joint_graph.nodes:
        if not ignore_must_be_in_fw_bw:
            if (
                _must_be_in_backward(node)
                and subgraph != "backward"
                and node not in inputs
            ):
                env[node] = InvalidNode  # type: ignore[assignment]
                continue

            if (
                _must_be_in_forward(node)
                and subgraph != "forward"
                and node not in inputs
            ):
                env[node] = InvalidNode  # type: ignore[assignment]
                continue

        if node in env:
            # Node must be one of our inputs. (Any member of env which wasn't an
            # input to start must have been created by this loop and won't be in
            # joint_graph.nodes).
            continue
        elif node.op == "placeholder":
            env[node] = InvalidNode  # type: ignore[assignment]
        elif node.op == "call_function":
            if (
                node.target is operator.getitem
                and node.args[0] in _control_deps_see_through
            ):
                dep_map = _control_deps_see_through[node.args[0]]
                idx = node.args[1]
                if idx in dep_map:
                    env[node] = env[dep_map[idx]]
                else:
                    env[node] = InvalidNode  # type: ignore[assignment]
                continue
            if (
                node.target is operator.getitem
                and node.args[0] in _control_deps_idx_remap
            ):
                idx_map = _control_deps_idx_remap[node.args[0]]
                idx = node.args[1]
                if idx in idx_map:
                    new_idx = idx_map[idx]
                    new_getitem = new_graph.create_node(
                        "call_function",
                        operator.getitem,
                        args=(env[node.args[0]], new_idx),
                        name=node.name,
                    )
                    new_getitem.meta = node.meta
                    env[node] = new_getitem
                else:
                    env[node] = InvalidNode  # type: ignore[assignment]
                continue

            all_args = pytree.arg_tree_leaves(*node.args, **node.kwargs)
            all_args = [
                isinstance(env[x], InvalidNodeBase)
                for x in all_args
                if isinstance(x, fx.Node)
            ]
            if any(all_args):
                if node.target is control_deps_hop:
                    # During backward extraction, eliminate forward sync
                    # control_deps entirely -- their subgraphs reference
                    # forward-only runtime state (streams/events) that is
                    # dead at backward time.  Getitems are resolved directly
                    # to the underlying deps via _control_deps_see_through.
                    if (
                        subgraph == "backward"
                        and not (
                            _has_tag_is_backward(node)
                            or _has_tag_must_be_in_backward(node)
                        )
                        and _control_deps_has_sync_op(node)
                    ):
                        _control_deps_see_through[node] = _build_see_through_map(node)
                        env[node] = InvalidNode  # type: ignore[assignment]
                        continue
                    new_cd_node = _rebuild_control_deps_for_extract(
                        node, env, new_graph
                    )
                    if new_cd_node is not None:
                        env[node] = new_cd_node[0]
                        _control_deps_idx_remap[node] = new_cd_node[1]
                        continue
                env[node] = InvalidNode  # type: ignore[assignment]
                continue
            # During backward extraction, eliminate forward sync control_deps
            # entirely.  See above for rationale.
            if (
                node.target is control_deps_hop
                and subgraph == "backward"
                and not (
                    _has_tag_is_backward(node) or _has_tag_must_be_in_backward(node)
                )
                and _control_deps_has_sync_op(node)
            ):
                _control_deps_see_through[node] = _build_see_through_map(node)
                env[node] = InvalidNode  # type: ignore[assignment]
                continue
            # pyrefly: ignore [unsupported-operation, bad-argument-type]
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "get_attr":
            # pyrefly: ignore [unsupported-operation, bad-argument-type]
            env[node] = new_graph.node_copy(node, lambda x: env[x])
        elif node.op == "output":
            pass
    output_values = []
    for x, x_desc in zip(outputs, outputs_descs):
        if isinstance(x, fx.Node):
            if x not in env:
                raise RuntimeError(f"Node {x} couldn't be found in env")
            if isinstance(env[x], InvalidNodeBase):
                # For forward outputs that are invalid (depend on backward), try
                # to find a valid replacement.
                replacement = None
                # For copy_ nodes that are backward-only, use the destination
                # (first arg) which is the original input.
                if (
                    x.target is torch.ops.aten.copy_.default
                    and _must_be_in_backward(x)
                    and len(x.args) >= 1
                    and isinstance(x.args[0], fx.Node)
                    and x.args[0] in env
                    and not isinstance(env[x.args[0]], InvalidNodeBase)
                ):
                    replacement = env[x.args[0]]
                # For view/reshape outputs that trace back to a getitem of a
                # higher-order op that mutates an input, find that input.
                # This handles custom_function_view outputs from triton kernels.
                if replacement is None:
                    replacement = _find_input_for_invalid_output(x, env)
                if replacement is not None:
                    output_values.append(replacement)
                    continue
                raise AssertionError(f"Node {x} was invalid, but is output")
            output_values.append(env[x])
        else:
            output_values.append(x)
    out = new_graph.output(tuple(output_values))
    out.meta["desc"] = outputs_descs
    # Snapshot stack traces on the output node before passes run,
    # as later passes may strip stack_trace from individual nodes.
    out.meta["output_stack_traces"] = [
        v.meta.get("stack_trace") if isinstance(v, fx.Node) else None
        for v in output_values
    ]

    new_graph.eliminate_dead_code()
    new_graph.lint()
    return new_graph


def is_non_builtin_to_include(node: fx.Node) -> bool:
    return config.is_non_builtin_to_include and (
        (isinstance(node.target, torch._ops.OpOverload) and not is_builtin(node.target))
        or node.target == torch.ops.higher_order.triton_kernel_wrapper_functional
    )


def _is_backward_state(node: fx.Node) -> bool:
    return node.op == "placeholder" and isinstance(node.meta.get("val"), BackwardState)


def _has_tag_is_backward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "is_backward"


def _has_tag_is_forward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "is_forward"


def _has_tag_must_be_in_forward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "must_be_in_forward"


def _has_tag_must_be_in_backward(node: fx.Node) -> bool:
    return node.meta.get("partitioner_tag", None) == "must_be_in_backward"


def _must_be_in_forward(node: fx.Node) -> bool:
    if _has_tag_must_be_in_forward(node):
        return True

    is_mutable = (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target._schema.is_mutable
    )
    return (
        not _has_tag_is_backward(node)
        and not _has_tag_must_be_in_backward(node)
        and is_mutable
    )


def _must_be_in_backward(node: fx.Node) -> bool:
    if _has_tag_must_be_in_backward(node):
        return True
    is_mutable = (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target._schema.is_mutable
    )
    return _has_tag_is_backward(node) and is_mutable


def _iter_input_exprs_without_replacements(val: Any) -> Iterator[sympy.Basic]:
    # Keep this traversal in sync with symbolic_shapes._iterate_exprs.
    if isinstance(val, py_sym_types):
        if is_symbolic(val):
            yield _get_placeholder_expr(val.node)
    elif isinstance(val, SymNode):
        yield _get_placeholder_expr(val)
    elif isinstance(val, sympy.Basic):
        yield val
    elif isinstance(val, (int, float, bool, str)):
        pass
    elif isinstance(val, (tuple, list)):
        for s in val:
            yield from _iter_input_exprs_without_replacements(s)
    elif isinstance(val, dict):
        for s in itertools.chain(val.keys(), val.values()):
            yield from _iter_input_exprs_without_replacements(s)
    elif is_sparse_any(val):
        yield from _iter_input_exprs_without_replacements(val.size())
    elif isinstance(val, torch.Tensor):
        yield from _iter_input_exprs_without_replacements(val.size())
        yield from _iter_input_exprs_without_replacements(val.stride())
        yield from _iter_input_exprs_without_replacements(val.storage_offset())
    elif val is None:
        pass
    elif isinstance(val, torch.Generator) or is_opaque_value(val):
        pass
    elif isinstance(val, FakeScriptObject):
        pass
    else:
        raise AssertionError(f"cannot extract sympy expressions from {val} {type(val)}")


def _free_symbols_without_replacements(val: Any) -> OrderedSet[sympy.Symbol]:
    symbols: OrderedSet[sympy.Symbol] = OrderedSet()
    for expr in _iter_input_exprs_without_replacements(val):
        symbols.update(expr.free_symbols)
    return symbols


def _extract_fwd_bwd_outputs(
    joint_module: fx.GraphModule, *, num_fwd_outputs: int
) -> tuple[list[fx.Node], list[fx.Node], list[AOTOutput], list[AOTOutput]]:
    outputs = pytree.arg_tree_leaves(
        *(node.args for node in joint_module.graph.find_nodes(op="output"))
    )
    outputs_descs = pytree.arg_tree_leaves(
        next(iter(joint_module.graph.find_nodes(op="output"))).meta.get(
            "desc", [None] * len(outputs)
        )
    )
    fwd_outputs = outputs[:num_fwd_outputs]
    bwd_outputs = outputs[num_fwd_outputs:]
    fwd_outputs_descs = outputs_descs[:num_fwd_outputs]
    bwd_outputs_descs = outputs_descs[num_fwd_outputs:]
    return fwd_outputs, bwd_outputs, fwd_outputs_descs, bwd_outputs_descs


def _remove_by_name(saved_values: list[fx.Node], name: str) -> None:
    for saved_value in saved_values:
        if saved_value.name == name:
            saved_values.remove(saved_value)
            break


def find_first_sym_node(
    fwd_module_outputs: list[fx.Node] | tuple[fx.Node, ...],
) -> int:
    idx = len(fwd_module_outputs)
    for i in range(len(fwd_module_outputs) - 1, -1, -1):
        if not is_sym_node(fwd_module_outputs[i]):
            idx = i + 1
            break
    return idx


def calculate_quantization_scaling(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    max: float = 57344.0,
    min: float = 1e-12,
    position: int = 0,
) -> torch.fx.Node:
    with graph.inserting_after(node):
        abs_node = graph.call_function(
            torch.ops.aten.abs.default,
            args=(node,),
        )
        abs_node.meta["val"] = torch.ops.aten.abs.default(node.meta["val"])
        abs_node.meta["tensor_meta"] = extract_tensor_metadata(abs_node.meta["val"])
    with graph.inserting_after(abs_node):
        amax_node = graph.call_function(
            torch.ops.aten.amax.default,
            args=(abs_node, [-1], True),
        )
        amax_node.meta["val"] = torch.ops.aten.amax.default(
            abs_node.meta["val"], [-1], True
        )
        amax_node.meta["tensor_meta"] = extract_tensor_metadata(amax_node.meta["val"])
    with graph.inserting_after(amax_node):
        amax_64_node = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(amax_node, torch.float64),
        )
        amax_64_node.meta["val"] = torch.ops.prims.convert_element_type.default(
            amax_node.meta["val"], torch.float64
        )
        amax_64_node.meta["tensor_meta"] = extract_tensor_metadata(
            amax_64_node.meta["val"]
        )
    with graph.inserting_after(amax_64_node):
        clamp_min_node = graph.call_function(
            torch.ops.aten.clamp_min.default,
            args=(amax_64_node, min),
        )
        clamp_min_node.meta["val"] = torch.ops.aten.clamp_min.default(
            amax_64_node.meta["val"], min
        )
        clamp_min_node.meta["tensor_meta"] = extract_tensor_metadata(
            clamp_min_node.meta["val"]
        )
    with graph.inserting_after(clamp_min_node):
        reciprocal_node = graph.call_function(
            torch.ops.aten.reciprocal.default,
            args=(clamp_min_node,),
        )
        reciprocal_node.meta["val"] = torch.ops.aten.reciprocal.default(
            clamp_min_node.meta["val"]
        )
        reciprocal_node.meta["tensor_meta"] = extract_tensor_metadata(
            reciprocal_node.meta["val"]
        )
    with graph.inserting_after(reciprocal_node):
        mul_node = graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(reciprocal_node, max),
        )
        mul_node.meta["val"] = torch.ops.aten.mul.Tensor(
            reciprocal_node.meta["val"], max
        )
        mul_node.meta["tensor_meta"] = extract_tensor_metadata(mul_node.meta["val"])
    with graph.inserting_after(mul_node):
        scale_node = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(mul_node, torch.float32),
            name=f"fp8_scale_pos_{position}_{node.name}",
        )
        scale_node.meta["val"] = torch.ops.prims.convert_element_type.default(
            mul_node.meta["val"], torch.float32
        )
        scale_node.meta["tensor_meta"] = extract_tensor_metadata(scale_node.meta["val"])
    return scale_node


def perform_quantization(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    scale_node: torch.fx.Node,
    quant_type: torch.dtype,
    clamp_min: float,
    clamp_max: float,
    position: int,
) -> torch.fx.Node:
    with graph.inserting_after(scale_node):
        target_node_32 = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(node, torch.float32),
        )
        target_node_32.meta["val"] = torch.ops.prims.convert_element_type.default(
            node.meta["val"], torch.float32
        )
        target_node_32.meta["tensor_meta"] = extract_tensor_metadata(
            target_node_32.meta["val"]
        )
    with graph.inserting_after(target_node_32):
        scaled_target_node = graph.call_function(
            torch.ops.aten.mul.Tensor,
            args=(target_node_32, scale_node),
        )
        scaled_target_node.meta["val"] = torch.ops.aten.mul.Tensor(
            target_node_32.meta["val"], scale_node.meta["val"]
        )
        scaled_target_node.meta["tensor_meta"] = extract_tensor_metadata(
            scaled_target_node.meta["val"]
        )
    with graph.inserting_after(scaled_target_node):
        clamp_min_scaled_node = graph.call_function(
            torch.ops.aten.clamp_min.default,
            args=(scaled_target_node, clamp_min),
        )
        clamp_min_scaled_node.meta["val"] = torch.ops.aten.clamp_min.default(
            scaled_target_node.meta["val"], clamp_min
        )
        clamp_min_scaled_node.meta["tensor_meta"] = extract_tensor_metadata(
            clamp_min_scaled_node.meta["val"]
        )
    with graph.inserting_after(clamp_min_scaled_node):
        clamp_max_scaled_node = graph.call_function(
            torch.ops.aten.clamp_max.default,
            args=(clamp_min_scaled_node, clamp_max),
        )
        clamp_max_scaled_node.meta["val"] = torch.ops.aten.clamp_max.default(
            clamp_min_scaled_node.meta["val"], clamp_max
        )
        clamp_max_scaled_node.meta["tensor_meta"] = extract_tensor_metadata(
            clamp_max_scaled_node.meta["val"]
        )
    with graph.inserting_after(clamp_max_scaled_node):
        quant_activation_node = graph.call_function(
            torch.ops.prims.convert_element_type.default,
            args=(clamp_max_scaled_node, quant_type),
            name=f"fp8_quant_pos_{position}_{node.name}",
        )
        quant_activation_node.meta["val"] = (
            torch.ops.prims.convert_element_type.default(
                clamp_max_scaled_node.meta["val"], quant_type
            )
        )
        quant_activation_node.meta["tensor_meta"] = extract_tensor_metadata(
            quant_activation_node.meta["val"]
        )
    return quant_activation_node


def calculate_tensor_size(tensor: torch.Tensor) -> float:
    """
    Calculate the size of a PyTorch tensor in megabytes (MB).

    Args:
        tensor (torch.Tensor): Input tensor

    Returns:
        float: Memory size in MB
    """
    # Get number of elements and size per element
    num_elements = tensor.numel()
    element_size = tensor.element_size()

    return (num_elements * element_size) / (1024 * 1024)


def get_allowed_dtypes() -> list[torch.dtype]:
    allowed_dtypes = torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("allowed_dtypes", "torch.bfloat16")
    allowed_dtypes = [
        getattr(torch, dtype.split(".")[-1]) for dtype in allowed_dtypes.split(";")
    ]
    return allowed_dtypes


def should_quantize(node: torch.fx.Node) -> bool:
    allowed_dtypes = get_allowed_dtypes()
    if not is_node_meta_valid(node) or node.meta["val"].dtype not in allowed_dtypes:
        return False
    size_threshold = torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("size_in_mb", 100)
    # calculate the size of the node
    size_in_mb = calculate_tensor_size(node.meta["val"])
    if not torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("skip_dynamo_guards", False):
        return size_in_mb >= size_threshold
    else:
        # case 1: we always quantize tensors with dynamic shapes
        if torch._inductor.config.post_grad_fusion_options[
            "activation_quantization_aten_pass"
        ].get("quantize_dynamic_shape", False):
            return statically_known_true(
                size_in_mb >= size_threshold
            ) or not statically_known_false(size_in_mb >= size_threshold)
        else:
            # case 2: we always not quantize tensors with dynamic shapes
            return statically_known_true(size_in_mb >= size_threshold)


def get_quant_type() -> torch.dtype:
    quant_type = torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("quant_type", "torch.float8_e5m2")

    return getattr(torch, quant_type.split(".")[-1])


def calculate_range(dtype: torch.dtype) -> tuple[float, float]:
    """
    Calculate the range of values for a given torch.dtype.
    Args:
        dtype (torch.dtype): The input dtype.
    Returns:
        tuple: A tuple containing the minimum and maximum values.
    """
    info = torch.finfo(dtype)
    return info.min, info.max


def quantize_activation_fw(graph: torch.fx.Graph, num_fwd_outputs: int = 0) -> None:
    output = graph.find_nodes(op="output")[0]
    fwd_outputs = output.args[0]
    quant_type = get_quant_type()
    clamp_min, clamp_max = calculate_range(quant_type)
    position_to_quant = dict()
    tensor_scale_nodes: list[fx.Node] = []
    sym_scale_nodes: list[fx.Node] = []
    for position, node in enumerate(fwd_outputs):
        # Don't quantize user-visible forward outputs. A tensor may appear as
        # both a user output and a saved-for-backward activation (same FX node
        # at two positions). Quantizing the user output position would:
        # 1. Return fp8 to the user instead of the original precision
        # 2. Create duplicate fp8_quant/fp8_scale backward placeholders that
        #    shift the stride mapping in _aot_stage2b_bw_compile (T264303372)
        if position < num_fwd_outputs:
            continue
        # check if the activation node is the node saved for quantization
        if node.meta.get("saved_for_quantization", False):
            # case: use scaling
            if torch._inductor.config.post_grad_fusion_options[
                "activation_quantization_aten_pass"
            ].get("use_scaling", True):
                # calculating the scale
                scale_node = calculate_quantization_scaling(
                    graph, node, clamp_max, 1e-12, position
                )

                # converting to fp8
                quant_node = perform_quantization(
                    graph, node, scale_node, quant_type, clamp_min, clamp_max, position
                )
                if not is_sym_node(scale_node):
                    tensor_scale_nodes.append(scale_node)
                else:
                    sym_scale_nodes.append(scale_node)
            else:
                # case: do not use scaling
                with graph.inserting_after(node):
                    quant_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(node, quant_type),
                        name=f"fp8_quant_pos_{position}_{node.name}",
                    )
                    quant_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            node.meta["val"], quant_type
                        )
                    )
                    quant_node.meta["tensor_meta"] = extract_tensor_metadata(
                        quant_node.meta["val"]
                    )

            position_to_quant[position] = quant_node

    # Use position-based lookup for building output
    # only update the return node args, and remain all other users unchanged
    output_updated_args = [
        position_to_quant.get(i, node) for i, node in enumerate(fwd_outputs)
    ]
    # add the scale nodes to the output find the first sym_node in the output
    # pyrefly: ignore [bad-argument-type]
    idx = find_first_sym_node(output_updated_args)
    scale_nodes = tensor_scale_nodes + sym_scale_nodes
    if scale_nodes:
        output_updated_args = (
            output_updated_args[:idx] + scale_nodes + output_updated_args[idx:]
        )

    output.update_arg(0, tuple(output_updated_args))
    counters["inductor"]["activation_quantization_fwd_aten_pass"] += 1


def quantize_activation_bw(graph: torch.fx.Graph) -> None:
    bw_inputs = [node for node in graph.nodes if node.op == "placeholder"]
    activation_node = None
    for node in bw_inputs:
        if node.meta.get("saved_for_quantization", False):
            node.meta.pop("saved_for_quantization")
            dequant_type = node.meta.pop("dequant_type")
            # dequantize the node
            if torch._inductor.config.post_grad_fusion_options[
                "activation_quantization_aten_pass"
            ].get("use_scaling", False):
                # case: use scaling
                with graph.inserting_after(node):
                    # find corresponding scale node
                    scale_name = "fp8_scale_" + node.name.replace("fp8_quant_", "")
                    scale_node = next(
                        bwd_input
                        for bwd_input in bw_inputs
                        if bwd_input.name == scale_name
                    )
                with graph.inserting_after(scale_node):
                    activation_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(node, dequant_type),
                    )
                    activation_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            node.meta["val"], dequant_type
                        )
                    )
                    activation_node.meta["tensor_meta"] = extract_tensor_metadata(
                        activation_node.meta["val"]
                    )
                with graph.inserting_after(activation_node):
                    divided_target_node_32 = graph.call_function(
                        torch.ops.aten.div.Tensor,
                        args=(activation_node, scale_node),
                    )
                    divided_target_node_32.meta["val"] = torch.ops.aten.div.Tensor(
                        activation_node.meta["val"], scale_node.meta["val"]
                    )
                    divided_target_node_32.meta["tensor_meta"] = (
                        extract_tensor_metadata(divided_target_node_32.meta["val"])
                    )
                with graph.inserting_after(divided_target_node_32):
                    dequant_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(divided_target_node_32, dequant_type),
                    )
                    dequant_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            divided_target_node_32.meta["val"], dequant_type
                        )
                    )
                    dequant_node.meta["tensor_meta"] = extract_tensor_metadata(
                        dequant_node.meta["val"]
                    )
            else:
                with graph.inserting_after(node):
                    dequant_node = graph.call_function(
                        torch.ops.prims.convert_element_type.default,
                        args=(node, dequant_type),
                        name="dequant_" + str(node.name),
                    )
                    dequant_node.meta["val"] = (
                        torch.ops.prims.convert_element_type.default(
                            node.meta["val"], dequant_type
                        )
                    )
                    dequant_node.meta["tensor_meta"] = extract_tensor_metadata(
                        dequant_node.meta["val"]
                    )
            # find the users of the node and replace them with the new node except the dequant_node
            for user in list(node.users.keys()):
                if user != dequant_node and user != activation_node:
                    user.replace_input_with(node, dequant_node)

    counters["inductor"]["activation_quantization_bwd_aten_pass"] += 1


def perform_fp8_activation_quantization(
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    bwd_module_inputs: dict[str, fx.Node],
    num_fwd_outputs: int = 0,
) -> None:
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "before_activation_quantization_fwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: fwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )

    quantize_activation_fw(fwd_module.graph, num_fwd_outputs)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "after_activation_quantization_fwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: fwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "before_activation_quantization_bwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: bwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )

    quant_fwd_module_outputs = fwd_module.graph.find_nodes(op="output")[0].args[0]
    # update the corresponding bwd_inputs due to the fwd_outputs quantization
    for fwd_node in quant_fwd_module_outputs:
        if "fp8_quant_" in fwd_node.name:
            bwd_input = bwd_module_inputs[
                re.sub(r"^fp8_quant_pos_\d+_", "", fwd_node.name)
            ]
            with bwd_module.graph.inserting_after(bwd_input):
                quant_bwd_input = bwd_module.graph.placeholder(name=fwd_node.name)
            dequant_type = bwd_input.meta["dequant_type"]
            quant_bwd_input.meta.update(fwd_node.meta)
            quant_bwd_input.meta["saved_for_quantization"] = True
            quant_bwd_input.meta["dequant_type"] = dequant_type
            bwd_input.replace_all_uses_with(quant_bwd_input)
            bwd_module.graph.erase_node(bwd_input)
    # update the bwd_inputs if quantization with scaling is used
    if torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("use_scaling", True):
        quant_bwd_module_inputs = list(bwd_module.graph.find_nodes(op="placeholder"))
        # update the corresponding bwd input nodes find the last non-tangent node
        bwd_input_loc = quant_bwd_module_inputs[-1]
        for bw_input in reversed(quant_bwd_module_inputs):
            if not _is_tangent(bw_input):
                bwd_input_loc = bw_input
                break

        scaled_fwd_module_outputs = fwd_module.graph.find_nodes(op="output")[0].args[0]
        for fwd_node in scaled_fwd_module_outputs:
            if "fp8_scale_" in fwd_node.name:
                # fwd node is a scale node
                with bwd_module.graph.inserting_after(bwd_input_loc):
                    scale_bwd_input = bwd_module.graph.placeholder(name=fwd_node.name)
                scale_bwd_input.meta.update(fwd_node.meta)
                bwd_input_loc = scale_bwd_input

    quantize_activation_bw(bwd_module.graph)

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "after_activation_quantization_bwd_aten_pass",
            "encoding": "string",
        },
        payload_fn=lambda: bwd_module.print_readable(
            print_output=False, include_stride=True, include_device=True
        ),
    )


def enable_activation_quantization(
    saved_values: list[fx.Node],
    fwd_module: fx.GraphModule,
    bwd_module: fx.GraphModule,
    static_lifetime_input_nodes: OrderedSet[fx.Node] | None = None,
    num_fwd_outputs: int = 0,
) -> None:
    static_input_names: list[str] = (
        [node.name for node in static_lifetime_input_nodes]
        if static_lifetime_input_nodes
        else []
    )
    saved_values_names = {node.name: node for node in saved_values}
    if torch._inductor.config.post_grad_fusion_options[
        "activation_quantization_aten_pass"
    ].get("exclude_primals", False):
        saved_values_names = {
            node.name: node for node in saved_values if "primals" not in node.name
        }
    fwd_module_outputs = fwd_module.graph.find_nodes(op="output")[0].args[0]
    bwd_module_inputs = {
        node.name: node for node in bwd_module.graph.find_nodes(op="placeholder")
    }
    should_perform_fp8_quant = False
    for node in fwd_module_outputs:
        if node.name in saved_values_names and should_quantize(node):
            if node.name in static_input_names:
                log.debug("Skipping quantization of static input %s: ", node.name)
                continue
            node.meta["saved_for_quantization"] = True
            node.meta["dequant_type"] = node.meta["val"].dtype
            # some of the fwd outputs and bwd inputs are not share the same object
            bwd_module_inputs[node.name].meta["saved_for_quantization"] = True
            bwd_module_inputs[node.name].meta["dequant_type"] = node.meta["val"].dtype
            should_perform_fp8_quant = True

    if should_perform_fp8_quant:
        perform_fp8_activation_quantization(
            fwd_module, bwd_module, bwd_module_inputs, num_fwd_outputs
        )


def _extract_fwd_bwd_modules(
    joint_module: fx.GraphModule,
    saved_values: list[fx.Node],
    saved_sym_nodes: list[fx.Node],
    saved_opaque_nodes: list[fx.Node] | None = None,
    *,
    num_fwd_outputs: int,
    static_lifetime_input_nodes: OrderedSet[fx.Node] | None = None,
    ignore_must_be_in_fw_bw: bool = False,
    omit_aot_autograd_runtime: bool = False,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    """Extract forward and backward graph modules from a joint graph.

    Args:
        ignore_must_be_in_fw_bw: When True, disables forward/backward placement
            enforcement in _extract_graph_with_inputs_outputs. Needed when the
            joint_module is not an original fwd+bwd joint graph (e.g. a backward
            graph being re-partitioned for dI/dW splitting).
        omit_aot_autograd_runtime: When True, skips postprocessing that is
            only needed when the resulting modules will be wrapped in a custom
            autograd.Function (the AOTAutograd path). This includes: tangent input
            handling, version-counter check sorting of saved tensors, opaque object
            (FakeScriptObject) separation, and fp8 activation quantization. Set this
            to True when the fwd/bwd modules will be executed directly without autograd.
    """
    fwd_outputs, bwd_outputs, fwd_outputs_descs, bwd_outputs_descs = (
        _extract_fwd_bwd_outputs(joint_module, num_fwd_outputs=num_fwd_outputs)
    )
    placeholders = joint_module.graph.find_nodes(op="placeholder")
    primal_inputs = [*filter(_is_primal, placeholders)]
    tangent_inputs = (
        [] if omit_aot_autograd_runtime else [*filter(_is_tangent, placeholders)]
    )
    fwd_seed_offset_inputs = [*filter(_is_fwd_seed_offset, placeholders)]
    bwd_seed_offset_inputs = [*filter(_is_bwd_seed_offset, placeholders)]
    backward_state_inputs = [*filter(_is_backward_state, placeholders)]

    if saved_opaque_nodes is None:
        saved_opaque_nodes = []

    bwd_graph = _extract_graph_with_inputs_outputs(
        joint_module.graph,
        saved_sym_nodes
        + saved_opaque_nodes
        + saved_values
        + tangent_inputs
        + bwd_seed_offset_inputs,
        bwd_outputs,
        bwd_outputs_descs,
        "backward",
        ignore_must_be_in_fw_bw=ignore_must_be_in_fw_bw,
    )

    distributed_enabled = torch.distributed.is_available()

    for node in bwd_graph.find_nodes(op="placeholder"):
        # This is to filter out saved values that don't actually end up being used by the backwards pass
        if not node.users:
            _remove_by_name(saved_values, node.name)
            _remove_by_name(saved_sym_nodes, node.name)
            _remove_by_name(saved_opaque_nodes, node.name)
        # wait_tensor is a bit special: if we have a "dead activation" that is not used in the bw,
        # but this dead activation is actually a collective,
        # then the collective will generally by followed by a wait_tensor() call.
        # we need to peek one node further to see if this wait_tensor is dead as well.
        elif distributed_enabled and all(
            n.target is torch.ops._c10d_functional.wait_tensor.default
            and len(n.users) == 0
            for n in node.users
        ):
            _remove_by_name(saved_values, node.name)
            _remove_by_name(saved_sym_nodes, node.name)
            _remove_by_name(saved_opaque_nodes, node.name)
        elif _is_backward_state(node):
            # BackwardState is saved directly
            _remove_by_name(saved_values, node.name)
            if not backward_state_inputs:
                raise AssertionError("backward_state_inputs must not be empty")

    # Now that we have the finalized list of saved values, we need to ensure
    # we propagate all symbols which are referenced by backwards inputs.
    # These are not directly used in the graph but are required for downstream
    # sizevar assignment
    saved_symbols: OrderedSet[sympy.Symbol] = OrderedSet()
    saved_sym_nodes_binding = []
    saved_sym_nodes_derived = []

    # Some symbols may already be bound in the directly saved_sym_nodes,
    # keep track of them so we don't re-bind them
    for node in saved_sym_nodes:
        symbol = is_symbol_binding_fx_node(node)
        if symbol:
            saved_symbols.add(symbol)
            saved_sym_nodes_binding.append(node)
        else:
            saved_sym_nodes_derived.append(node)

    # Now go through all of the prospective backward inputs and track any
    # other symbols we need to bind
    symbol_bindings = find_symbol_binding_fx_nodes(joint_module.graph)
    for node in itertools.chain(saved_sym_nodes_derived, saved_values, tangent_inputs):
        if "val" not in node.meta:
            continue
        # Bind both the unreplaced symbols (needed by runtime assertions, which
        # preserve raw placeholder expressions -- see #155468) and the replaced
        # symbols (needed by sizevar codegen and FxGraphCache guards, which use
        # ShapeEnv replacements). Binding only one side can leave the other's
        # symbols undefined in the backward: e.g. an offsets size that is a
        # symbol replaced to `s + 1` leaves the base symbol `s` unbound, which
        # surfaces as a KeyError during backward FxGraphCache guard evaluation.
        new_symbols = (
            _free_symbols_without_replacements(node.meta["val"])
            | free_symbols(node.meta["val"])
        ) - saved_symbols
        # NB: Deterministic order please!
        for s in sorted(new_symbols, key=lambda s: s.name):
            # NB: For well formed graphs, the symbol should always be present,
            # but we also have ways to produce ill-formed graphs, e.g., direct
            # make_fx usages, so don't choke in this case
            if s not in symbol_bindings:
                continue
            saved_sym_nodes_binding.append(symbol_bindings[s])
        saved_symbols |= new_symbols

    # Update saved_sym_nodes that are now reordered to have all bindings at
    # front. This can also be used later on to figure out the position of saved
    # sym nodes in the output of fwd graph.
    saved_sym_nodes.clear()
    saved_sym_nodes.extend(saved_sym_nodes_binding + saved_sym_nodes_derived)

    if not omit_aot_autograd_runtime:
        # See Note [Activations with no version counter checks in eager]
        # Sort saved_values so that tensors with saved_tensor_with_no_vc_check=True
        # are at the end. This allows us to have two consecutive slices:
        # 1. tensors_saved_with_vc_check_slice - tensors saved via save_for_backward
        # 2. tensors_saved_with_no_vc_check_slice - tensors stashed on ctx without save_for_backward
        # The sort is stable, so the relative order within each group is preserved.
        #
        # Additionally, separate out opaque objects (FakeScriptObject) from tensors.
        # Opaque objects should be placed after tensors in the forward outputs.
        saved_values_with_vc_check = []
        saved_values_no_vc_check = []
        saved_opaque_objects = []
        for node in saved_values:
            # Check if this is an opaque object
            if isinstance(node.meta.get("val"), FakeScriptObject):
                saved_opaque_objects.append(node)
            elif node.meta.get("saved_tensor_with_no_vc_check", False):
                saved_values_no_vc_check.append(node)
            else:
                saved_values_with_vc_check.append(node)
        saved_values.clear()
        saved_values.extend(saved_values_with_vc_check + saved_values_no_vc_check)
        no_vc_check_start_idx = len(saved_values_with_vc_check)

        # debug assert: given saved_values where the last k of them are expected to not
        # require VC checks, they should all have node metadata indicating so.
        for i, node in enumerate(saved_values):
            if i >= no_vc_check_start_idx:
                if not node.meta.get("saved_tensor_with_no_vc_check", False):
                    raise AssertionError(
                        f"i={i}, no_vc_check_start_idx={no_vc_check_start_idx}, len(saved_values)={len(saved_values)}"
                    )

        # Now, we re-generate the fwd/bwd graphs.
        # NB: This might increase compilation time, but I doubt it matters
        # Convention for saved acts is (tensors_with_vc_check, tensors_no_vc_check, opaque_objects, symints, opaque_nodes)
        fwd_graph = _extract_graph_with_inputs_outputs(
            joint_module.graph,
            primal_inputs + fwd_seed_offset_inputs,
            fwd_outputs
            + saved_values
            + saved_opaque_objects
            + saved_opaque_nodes
            + saved_sym_nodes,
            fwd_outputs_descs
            + [
                SavedForBackwardsNoVcCheckAOTOutput(i)
                if i >= no_vc_check_start_idx and i < len(saved_values)
                else SavedForBackwardsAOTOutput(i)
                for i in range(
                    len(saved_values)
                    + len(saved_opaque_objects)
                    + len(saved_opaque_nodes)
                    + len(saved_sym_nodes)
                )
            ],
            "forward",
            ignore_must_be_in_fw_bw=ignore_must_be_in_fw_bw,
        )
        bwd_graph = _extract_graph_with_inputs_outputs(
            joint_module.graph,
            saved_sym_nodes
            + saved_values
            + saved_opaque_objects
            + saved_opaque_nodes
            + tangent_inputs
            + bwd_seed_offset_inputs
            + backward_state_inputs,
            bwd_outputs,
            bwd_outputs_descs,
            "backward",
            ignore_must_be_in_fw_bw=ignore_must_be_in_fw_bw,
        )
    else:
        # Raw fwd/bwd split for direct execution without autograd
        fwd_graph = _extract_graph_with_inputs_outputs(
            joint_module.graph,
            primal_inputs + fwd_seed_offset_inputs,
            fwd_outputs + saved_values + saved_sym_nodes,
            fwd_outputs_descs
            + [
                SavedForBackwardsAOTOutput(i)
                for i in range(len(saved_values) + len(saved_sym_nodes))
            ],
            "forward",
            ignore_must_be_in_fw_bw=ignore_must_be_in_fw_bw,
        )
        bwd_graph = _extract_graph_with_inputs_outputs(
            joint_module.graph,
            saved_values
            + saved_sym_nodes
            + bwd_seed_offset_inputs
            + backward_state_inputs,
            bwd_outputs,
            bwd_outputs_descs,
            "backward",
            ignore_must_be_in_fw_bw=ignore_must_be_in_fw_bw,
        )

    for node in bwd_graph.nodes:
        if node.op in ("call_function", "get_attr"):
            node.meta["autograd_backward"] = True

    fwd_module = fx._lazy_graph_module._make_graph_module(joint_module, fwd_graph)
    bwd_module = fx._lazy_graph_module._make_graph_module(joint_module, bwd_graph)
    if (
        inductor_config.post_grad_fusion_options.get(
            "activation_quantization_aten_pass", None
        )
        is not None
    ):
        enable_activation_quantization(
            saved_values,
            fwd_module,
            bwd_module,
            static_lifetime_input_nodes,
            num_fwd_outputs,
        )
    return fwd_module, bwd_module


def default_partition(
    joint_module: fx.GraphModule,
    _joint_inputs: Any,
    *,
    num_fwd_outputs: int,
    static_lifetime_input_indices: list[int] | None = None,
    static_lifetime_input_nodes: OrderedSet[fx.Node] | None = None,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    """
    Partitions the :attr:`joint_module` in a manner that closely resembles the
    behavior observed in the original ``.forward()`` and ``.backward()`` of the
    callable, i.e., the resulting forward graph contains those operators that
    are executed in the original ``.forward()`` callable passed to
    :func:`aot_function`.

    The default partitioner collects the operators that are between the forward
    inputs and the forward outputs. This helps in finding the tensors which have
    to be stashed for the backward pass. These stashed tensors become the output
    of the generated forward graph. The remaining operators are then placed in
    the backward graph.

    .. warning::
        This API is experimental and likely to change.

    Args:
        joint_module(fx.GraphModule): The joint forward and backward graph. This
            is the result of AOT Autograd tracing.

    Returns:
        Returns the generated forward and backward Fx graph modules.
    """
    # Respect the original placement of ops rather than rely on dataflow.
    forward_nodes = []
    last_node = None
    for node in joint_module.graph.nodes:
        if _has_tag_is_forward(node) or _is_primal(node) or _is_fwd_seed_offset(node):
            last_node = node
    if last_node is None:
        raise AssertionError("last_node must not be None")
    for node in joint_module.graph.nodes:
        if not _is_tangent(node):
            forward_nodes.append(node)
        if node is last_node:
            break
    forward_node_names = OrderedSet(
        node.name for node in forward_nodes if node.op != "output"
    )
    graph_has_recomputable_ops = has_recomputable_ops(joint_module)
    graph_has_recomputable_rng_ops = has_recomputable_rng_ops(joint_module)
    if graph_has_recomputable_ops:
        if _is_functional_graph(joint_module.graph)[0] is not None:
            # Fall-back to previous behavior to avoid bc-breaking, although can
            # eventually flip the switch to make this a hard error.
            warnings.warn(
                "Trying to unsafely apply AC to a non-functional graph with the "
                "default partitioner. Falling back to min-cut partitioner."
            )
            return min_cut_rematerialization_partition(
                joint_module,
                _joint_inputs,
                num_fwd_outputs=num_fwd_outputs,
                static_lifetime_input_indices=static_lifetime_input_indices,
            )

        joint_module = cleanup_recompute_tags(joint_module, is_default_partition=True)

    if not config.unsafe_allow_optimization_of_collectives:
        force_save_collectives(joint_module)

    force_save_effectful_ops(joint_module)
    force_save_bw_mutation_src(joint_module)

    if static_lifetime_input_indices is None:
        static_lifetime_input_indices = []
    node_info = classify_nodes(
        joint_module, static_lifetime_input_indices, num_fwd_outputs
    )

    saved_values = []
    saved_sym_nodes = []
    saved_opaque_nodes = []

    distributed_enabled = torch.distributed.is_available()

    def is_tensor(node: fx.Node) -> bool:
        return "tensor_meta" in node.meta or isinstance(
            node.meta.get("val"), torch._subclasses.FakeTensor
        )

    def is_multi_output(node: fx.Node) -> bool:
        return (
            all(user.target == operator.getitem for user in node.users)
            and len(node.users) > 0
        )

    def is_impure(node: fx.Node) -> bool:
        # wait tensor is an "impure" op according to DCE's definition of impure
        # (see is_impure in torch/fx/node.py), but it survives past
        # functionalization and can be safely dup'd and reordered under the
        # assumption SPMD.
        return (
            node.is_impure(impure_random=False)
            and node.op
            not in (
                "placeholder",
                "output",
            )
            and (
                not distributed_enabled
                or node.target is not torch.ops._c10d_functional.wait_tensor.default
            )
        )

    for node in joint_module.graph.nodes:
        if node.name not in forward_node_names:
            continue
        if node.op == "get_attr" and node.name in (
            k for k, v in joint_module.named_modules()
        ):
            continue
        if node.target in (
            torch.ops.aten._assert_scalar.default,
            torch.ops.aten._assert_async.default,
            torch.ops.aten._assert_async.msg,
            # Profiler record_function ops are technically impure (they set up
            # profiling spans), but they're safe to duplicate during AC recompute.
            # We skip both enter and exit to keep profiling spans balanced.
            torch.ops.profiler._record_function_enter_new.default,
            torch.ops.profiler._record_function_enter.default,
            torch.ops.profiler._record_function_exit.default,
            torch.ops.profiler._record_function_exit._RecordFunction,
        ):
            continue
        if is_sym_node(node):
            # Symints must be kept separate from tensors so that PythonFunction only calls
            # save_for_backward on tensors and stashes symints in autograd .ctx
            saved_sym_nodes.append(node)
            continue
        if is_multi_output(node):
            # Must be ordered before MUST_SAVE tags to avoid saving tuples marked MUST_SAVE.
            continue
        if node.meta.get("recompute") == CheckpointPolicy.MUST_SAVE:
            if is_opaque_node(node):
                saved_opaque_nodes.append(node)
            else:
                saved_values.append(node)
            continue
        if is_impure(node):
            if graph_has_recomputable_ops:
                raise AssertionError(
                    f"Trying to apply AC on a graph with impure op: {node}, {node.target}"
                )
            if is_opaque_node(node):
                saved_opaque_nodes.append(node)
            else:
                saved_values.append(node)
            continue
        if is_opaque_node(node):
            saved_opaque_nodes.append(node)
            continue
        if not is_tensor(node) and node.op == "call_function":
            raise AssertionError(f"Expected {node} to be a tensor")
        backward_usages = [n for n in node.users if n.name not in forward_node_names]
        if all(is_sym_node(n) for n in backward_usages):
            # If we have a tensor in the forward, where only its sizes/strides are needed in the backward,
            # and not the actual tensor data,
            # then it will be a lot cheaper to save only the sizes/strides, and not the actual tensor.
            #
            # Note that saving the tensor could also cause compilation problems:
            # If the user mutated an input in the forward and uses its sizes/strides in the backward,
            # then we would be obligated to clone the input before saving it to appease autograd.
            # (This is how we originally found this bug).
            saved_sym_nodes.extend(backward_usages)
            continue
        if not must_recompute(node):
            saved_values.append(node)

    saved_values = list(dict.fromkeys(saved_values).keys())
    saved_sym_nodes = list(dict.fromkeys(saved_sym_nodes).keys())
    # Skip SymBool nodes whose only consumers are _assert_scalar calls.
    # These are runtime assertion intermediates and are not needed in backward
    # for any real computation.
    saved_sym_nodes = [
        node for node in saved_sym_nodes if not _is_assert_only_symbool(node)
    ]
    saved_opaque_nodes = list(dict.fromkeys(saved_opaque_nodes).keys())

    if config._sync_decision_cross_ranks:
        saved_values = _sync_decision_cross_ranks(joint_module.graph, saved_values)

    if static_lifetime_input_nodes is None:
        static_lifetime_input_nodes = node_info.static_lifetime_input_nodes
    fw_module, bw_module = _extract_fwd_bwd_modules(
        joint_module,
        saved_values,
        saved_sym_nodes=saved_sym_nodes,
        saved_opaque_nodes=saved_opaque_nodes,
        num_fwd_outputs=num_fwd_outputs,
        static_lifetime_input_nodes=static_lifetime_input_nodes,
    )

    # Run DCE while overriding the definition of is_impure_node
    fw_module.graph.eliminate_dead_code(is_impure_node=is_not_collective)
    bw_module.graph.eliminate_dead_code(is_impure_node=is_not_collective)

    if graph_has_recomputable_ops:
        if graph_has_recomputable_rng_ops:
            fw_module, bw_module = functionalize_rng_ops(
                joint_module, fw_module, bw_module, len(saved_sym_nodes)
            )
        bw_module = reordering_to_mimic_autograd_engine(bw_module)

    # pyrefly: ignore [unbound-name]
    if config.enable_activation_offloading:
        from ._activation_offloading.activation_offloading import (
            enable_activation_offloading,
        )

        enable_activation_offloading(
            fw_module,
            bw_module,
            num_fwd_outputs,
            static_lifetime_input_nodes,
        )

    # raise all getitem ops to as early as possible
    # this is helpful for memory, especially in the case of aot_eager backend
    fw_module = raise_getitems(fw_module)
    bw_module = raise_getitems(bw_module)

    fw_module = thread_graphsafe_rng_from_hops(fw_module, is_backward=False)
    if len(node_info.required_bw_nodes) > 0:
        bw_module = thread_graphsafe_rng_from_hops(bw_module, is_backward=True)

    return fw_module, bw_module


INT_INF = int(1e6)


def _tensor_nbytes(numel: int, dtype: torch.dtype) -> int:
    return numel * dtype.itemsize


def _size_of(node: fx.Node) -> int:
    def object_nbytes(x: object) -> int:
        if not isinstance(x, torch.Tensor):
            return 0
        return _tensor_nbytes(optimization_hint(x.numel(), fallback=4096), x.dtype)

    if "val" in node.meta:
        val = node.meta["val"]
        if isinstance(val, py_sym_types):
            return 1
        # NB: The fallback values here are meaningless, maybe we should respect
        # torch._inductor.config.unbacked_symint_fallback (but this is a
        # layering violation)
        elif isinstance(val, (list, tuple)):
            return sum(object_nbytes(n) for n in val)
        elif isinstance(val, dict):
            return sum(object_nbytes(n) for n in val.values())
        elif isinstance(val, torch.Tensor):
            return object_nbytes(val)
        elif isinstance(val, (torch.ScriptObject, FakeScriptObject)):
            # A (Fake)ScriptObject may hold tensors internally, so we cannot
            # soundly compute its size here. Only treat it as zero size when the
            # user has explicitly opted in via this escape hatch.
            if config.unsafe_treat_script_objects_as_zero_size:
                return 0
            raise RuntimeError(
                f"Cannot compute the size of {type(val)} on node {node}. A "
                "ScriptObject may hold tensors internally and the partitioner "
                "has no general way to measure its memory footprint. Set "
                "torch._functorch.config.unsafe_treat_script_objects_as_zero_size"
                " = True to assume such objects are zero size (unsound)."
            )

        raise RuntimeError(f"Unknown metadata type {type(val)} on node {node}")
    if node.op == "get_attr" or (
        isinstance(node.target, torch._ops.OpOverload)
        and len(node.target._schema.returns) == 0
    ):
        return 0
    raise RuntimeError(
        f"Node {node} didn't have `val` metadata; we should always have `val` metadata on the nodes."
    )


# Used for some investigative purposes
def _count_ops(graph: fx.Graph) -> None:
    from collections import defaultdict

    cnt: dict[str, int] = defaultdict(int)
    for node in graph.nodes:
        if node.op == "call_function":
            cnt[node.target.__name__] += 1
    log.info("%s", sorted(cnt.items(), key=operator.itemgetter(1), reverse=True))


@functools.cache
def pointwise_ops() -> list[torch._ops.OpOverloadPacket]:
    ops: list[torch._ops.OpOverloadPacket] = []
    for attr_name in dir(torch.ops.aten):
        opoverloadpacket = getattr(torch.ops.aten, attr_name)
        if not isinstance(opoverloadpacket, torch._ops.OpOverloadPacket):
            continue

        for op_overload in opoverloadpacket.op_overloads():
            if torch.Tag.pointwise in op_overload.tags:
                # currently aot autograd uses packet not overload
                ops.append(opoverloadpacket)
                break

    return ops


def sort_depths(
    args: tuple[Any, ...], depth_map: dict[fx.Node, int]
) -> list[tuple[fx.Node, int]]:
    arg_depths = {
        arg: depth_map[arg] for arg in args if isinstance(arg, torch.fx.node.Node)
    }
    return sorted(arg_depths.items(), key=operator.itemgetter(1), reverse=True)


def reordering_to_mimic_autograd_engine(gm: fx.GraphModule) -> fx.GraphModule:
    """
    This pass finds the first bwd node in the graph (by looking at users of
    tangents) and then reorders the graph by walking from this node to all the
    way to the end of the graph. At each op in this traversal, we insert this op
    in a new graph and try to bring only the relevant subgraph from the other
    non-bwd edges relevant for this op. This closely mimics the behavior of
    autograd engine.

    Why is this pass required in the first place?

    This is an artifact of how partitioners work today. The starting point of
    partitioner is a joint graph, which is fwd and then bwd graph. In the case
    of checkpointing, we keep portions of fwd graph in their original place in
    the joint graph, while obtaining a bwd graph. As a result, the resulting bwd
    graph has copies of recomputed fwd subgraphs followed by the original bwd
    graph. If we run this naively, this leads to bad memory footprint, because
    the fwd subgraphs are live for way longer duration than necessary. This pass
    reorders the operations such that we prioritize the ops for the original bwd
    graph while only realizing those ops from the fwd graph that are necessary
    at any given point in the graph.
    """

    new_graph = fx.Graph()
    env: dict[fx.Node, fx.Node] = {}

    # Add new placeholder nodes in the order specified by the inputs
    for node in gm.graph.find_nodes(op="placeholder"):
        env[node] = new_graph.node_copy(node, lambda x: env[x])

    order = {node: idx for idx, node in enumerate(gm.graph.nodes)}

    def insert_node_in_graph(node: fx.Node) -> None:
        cur_nodes = [node]
        insertable_nodes: OrderedSet[fx.Node] = OrderedSet()
        while len(cur_nodes) > 0:
            node = cur_nodes.pop()
            if node in insertable_nodes or node in env:
                continue
            insertable_nodes.add(node)

            # Bias traversal towards the nodes that have higher depth - prioritizes
            # critical path first.
            cur_nodes += node.all_input_nodes

        # pyrefly: ignore [bad-assignment]
        insertable_nodes = sorted(insertable_nodes, key=lambda n: order[n])
        for node in insertable_nodes:
            env[node] = new_graph.node_copy(node, lambda x: env[x])

    # Find first bwd node in the graph
    tangent_inputs = list(filter(_is_tangent, gm.graph.nodes))
    first_node_in_bwd = None
    minimum_order = math.inf
    for tangent in tangent_inputs:
        for user in tangent.users:
            if order[user] < minimum_order:
                minimum_order = order[user]
                first_node_in_bwd = user

    # If gradInp does not depend upon gradOut, we may not find any nodes in the "backwards pass"
    if first_node_in_bwd is None:
        return gm

    # Build the graph op-by-op by starting from the node all the way to the end
    # copy_ can be not using tangents at all, we must copy it.
    for node in list(gm.graph.nodes)[: order[first_node_in_bwd]]:
        if node.op == "call_function" and node.target is torch.ops.aten.copy_.default:
            insert_node_in_graph(node)

    for node in list(gm.graph.nodes)[order[first_node_in_bwd] :]:
        insert_node_in_graph(node)

    # The output node is already built by the traversal.
    new_gm = torch.fx.GraphModule(gm, new_graph)
    return new_gm


def apply_graphsafe_rng_functionalization(
    fw_module: torch.fx.GraphModule,
    bw_module: torch.fx.GraphModule,
    fw_node: torch.fx.Node,
    bw_node: torch.fx.Node,
    device: torch.device,
    rng_count: int,
    last_fwd_input: torch.fx.Node,
    last_bwd_input: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node]:
    """
    Note [CUDA Graph Safe RNG Functionalization]

    CUDA Graph capture doesn't work with get_rng_state and set_rng_state because these functions operate on CPU values,
    while CUDA Graph RNG capture uses on-device CUDA tensors. To solve this, we use graphsafe_set_state with a
    CUDA Generator registered to the CUDA Graph before capture begins. graphsafe_set_state updates the generator's pointer
    to reference a different GeneratorImpl, ensuring subsequent calls are correctly forwarded to the desired generator
    (and its cuda-tensor RNG state during graph capture).

    For each RNG operation's forward/backward pair:

    - We create two generators initialized with identical values
    - Each forward and backward call advances its respective generator equally
    - This keeps generators synchronized so forward and backward operations use matching RNG values

    When forward is called multiple times before backward (causing desynchronization):

    - We save the forward RNG state
    - We update the backward Generator's state before executing backward

    This function modifies both forward and backward computation graphs by:

    Creating RNG state placeholders for both passes
    Updating the forward node to use graph-safe RNG state
    Updating the backward node to use graph-safe RNG state

    For more details: https://github.com/pytorch/pytorch/issues/113541
    """
    device_idx = device.index
    if device_idx is None:
        raise AssertionError("device_idx must not be None")
    fw_graph = fw_module.graph
    bw_graph = bw_module.graph
    graphsafe_run_with_rng_state = torch._prims.rng_prims.graphsafe_run_with_rng_state

    # Handle forward pass

    # Note: [Generator arguments in AOTDispatcher]
    # Generator arguments in AOTDispatcher are added to support graphsafe rng
    # functionalization. See note above [CUDA Graph Safe RNG Functionalization]
    with fw_module.graph.inserting_after(last_fwd_input):
        fwd_rng_state = fw_module.graph.placeholder(f"fwd_rng_state_{rng_count}")
        fwd_rng_state.meta["val"] = get_default_generator(device).clone_state()
        last_fwd_input = fwd_rng_state

    # Handle backward pass
    with bw_module.graph.inserting_after(last_bwd_input):
        bwd_rng_state = bw_module.graph.placeholder(f"bwd_rng_state_{rng_count}")
        # as above, clone so that meta val generator will not contain tensors
        bwd_rng_state.meta["val"] = get_default_generator(device).clone_state()
        last_bwd_input = bwd_rng_state

    # Update forward node
    fw_kwargs = dict(fw_node.kwargs)
    fw_kwargs["rng_state"] = fwd_rng_state
    with fw_module.graph.inserting_after(fw_node):
        functional_fw_node = fw_graph.create_node(
            "call_function",
            graphsafe_run_with_rng_state,
            args=(fw_node.target, *fw_node.args),  # type: ignore[arg-type]
            kwargs=fw_kwargs,
        )
    fw_node.replace_all_uses_with(functional_fw_node)
    fw_graph.erase_node(fw_node)

    # Update backward node
    bwd_kwargs = dict(bw_node.kwargs)
    bwd_kwargs["rng_state"] = bwd_rng_state
    with bw_graph.inserting_before(bw_node):
        rng_output = bw_graph.create_node(
            "call_function",
            graphsafe_run_with_rng_state,
            args=(bw_node.target, *bw_node.args),  # type: ignore[arg-type]
            kwargs=bwd_kwargs,
        )
        bw_node.replace_all_uses_with(rng_output)
        bw_graph.erase_node(bw_node)

    return last_fwd_input, last_bwd_input


def functionalize_rng_ops(
    joint_module: fx.GraphModule,
    fw_module: fx.GraphModule,
    bw_module: fx.GraphModule,
    num_sym_nodes: int,
) -> tuple[fx.GraphModule, fx.GraphModule]:
    # During user-driven activation checkpointing, we have to ensure that a rng
    # op in fwd yields the same output as the recomputed rng op in the bwd.  To
    # do this, we use functionalize wrappers to wrap the random ops and share
    # rng state between the fwd and bwd graphs.

    # There are 3 main steps to do this
    # Step 1 - Construct a mapping of rng node between the fwd and its counterpart in bwd.
    # Step 2 - Modify the fwd pass such that
    #   1) Replace rand with run_and_save_rng_state wrapper
    #   2) Replace the users of the original op with the output[1] of this op.
    #   3) Collect all the rng_state - output[0] of each op, and make them
    #   output nodes. Special care needs to be taken here because fwd outputs
    #   has symints at the very end.
    # Step 3 - Modify the bwd pass such that
    #   1) Add the input nodes just before the tangents for the stashed rng states
    #   2) Replace rand with run_with_save_rng_state wrappers
    #   3) Use the stashed states as inputs to these ops

    # Unique id to generate name
    uid = itertools.count()

    def get_rng_ops(gmod: fx.GraphModule) -> dict[str, fx.Node]:
        random_nodes: dict[str, fx.Node] = {}
        for node in gmod.graph.nodes:
            if node.op == "call_function" and is_rng_op(node):
                random_nodes[node.name] = node
        return random_nodes

    def get_device(node: fx.Node) -> torch.device | None:
        """
        Check the example value of the node outputs to find the device type.
        """
        if "val" not in node.meta:
            return None

        candidates = node.meta["val"]
        if not isinstance(candidates, tuple):
            candidates = (candidates,)

        for candidate in candidates:
            if isinstance(candidate, torch.Tensor):
                if supports_graphsafe_rng(candidate.device):
                    return candidate.device

        return torch.device("cpu")

    def get_sample_rng_state(device: torch.device | None) -> torch.Tensor:
        from torch._guards import detect_fake_mode

        fake_mode = detect_fake_mode()
        if fake_mode is None:
            raise AssertionError("fake_mode must not be None")
        with fake_mode:
            if device is not None and supports_graphsafe_rng(device):
                return fake_mode.from_tensor(get_device_rng_state(device))
            return fake_mode.from_tensor(torch.get_rng_state())

    # Step 1 - Construct a mapping of rng node between the fwd and its counterpart in bwd.
    joint_graph_rng_ops = get_rng_ops(joint_module)
    fw_graph_rng_ops = get_rng_ops(fw_module)
    bw_graph_rng_ops = get_rng_ops(bw_module)
    recomputable_rng_ops_map = {}
    for node in joint_module.graph.nodes:
        if must_recompute(node) and is_rng_op(node):
            # Skip if the node doesn't exist in both forward and backward graphs.
            # This can happen when the RNG op's output is not needed for gradient
            # computation and gets eliminated by dead code elimination.
            if node.name not in fw_graph_rng_ops or node.name not in bw_graph_rng_ops:
                continue
            base_node = joint_graph_rng_ops[node.name]
            fw_node = fw_graph_rng_ops[node.name]
            bw_node = bw_graph_rng_ops[node.name]
            recomputable_rng_ops_map[base_node] = {"fwd": fw_node, "bwd": bw_node}

    run_and_save_rng = torch._prims.rng_prims.run_and_save_rng_state
    run_with_rng_state = torch._prims.rng_prims.run_with_rng_state

    bw_tangent_start_node = None
    for node in bw_module.graph.find_nodes(op="placeholder"):
        if "tangent" in node.name:
            bw_tangent_start_node = node
            break
    if bw_tangent_start_node is None:
        raise RuntimeError(
            "Couldn't find tangent node in graph inputs. This is unexpected, please file a bug if you see this"
        )

    fw_rng_state_outputs: list[fx.Node] = []

    last_fwd_input = next(reversed(fw_module.graph.find_nodes(op="placeholder")))
    last_bwd_input = next(reversed(bw_module.graph.find_nodes(op="placeholder")))

    devices = OrderedSet(
        get_device(node_pair["fwd"]) for node_pair in recomputable_rng_ops_map.values()
    )
    # pyrefly: ignore [unbound-name]
    devices.discard(torch.device("cpu"))
    # multiple graphsafe devices won't work with cudagraphs anyway,
    # fallback to non graphsafe rng checkpointing
    multi_graphsafe_devices = len(devices) > 1

    # this changes numerics, so if fallback_random is set we will not use it
    # pyrefly: ignore [unbound-name]
    ind_config = torch._inductor.config
    use_rng_graphsafe_rng_functionalization = (
        config.graphsafe_rng_functionalization
        and not multi_graphsafe_devices
        and (
            not ind_config.fallback_random
            or ind_config.test_configs.graphsafe_rng_func_ignores_fallback_random
        )
    )

    for rng_count, node_pair in enumerate(recomputable_rng_ops_map.values()):
        # Step 2 - Modify the fwd pass such that
        fw_node = node_pair["fwd"]
        bw_node = node_pair["bwd"]
        device = get_device(fw_node)

        fw_graph = fw_module.graph
        bw_graph = bw_module.graph

        if (
            use_rng_graphsafe_rng_functionalization
            and device is not None
            and supports_graphsafe_rng(device)
        ):
            last_fwd_input, last_bwd_input = apply_graphsafe_rng_functionalization(
                fw_module,
                bw_module,
                fw_node,
                bw_node,
                device,
                rng_count,
                last_fwd_input,
                last_bwd_input,
            )
        else:
            with fw_graph.inserting_before(fw_node):
                functional_fw_node = fw_graph.create_node(
                    "call_function",
                    run_and_save_rng,
                    # pyrefly: ignore [bad-argument-type]
                    args=(
                        fw_node.target,
                        *fw_node.args,
                    ),  # pyrefly: ignore[bad-argument-type]
                    kwargs=fw_node.kwargs,
                )
                state = fw_graph.create_node(
                    "call_function",
                    operator.getitem,
                    args=(functional_fw_node, 0),
                    kwargs={},
                )
                state.meta["val"] = get_sample_rng_state(device)

                rng_output = fw_graph.create_node(
                    "call_function",
                    operator.getitem,
                    args=(
                        functional_fw_node,
                        1,
                    ),
                    kwargs={},
                )
                # Copy the meta data from the original node
                rng_output.meta = copy.copy(fw_node.meta)

                fw_node.replace_all_uses_with(rng_output)
                fw_graph.erase_node(fw_node)
                fw_rng_state_outputs.append(state)

            # Step 3 - Modify the bwd pass such that
            with bw_graph.inserting_before(bw_tangent_start_node):
                state_name = f"rng_state_output_{next(uid)}"
                bw_rng_state_node = bw_graph.placeholder(state_name)
                bw_rng_state_node.meta["val"] = get_sample_rng_state(device)

            with bw_graph.inserting_before(bw_node):
                rng_output = bw_graph.create_node(
                    "call_function",
                    run_with_rng_state,
                    # pyrefly: ignore [bad-argument-type]
                    args=(
                        bw_rng_state_node,
                        bw_node.target,
                        *bw_node.args,
                    ),  # pyrefly: ignore[bad-argument-type]
                    kwargs=bw_node.kwargs,
                )

                bw_node.replace_all_uses_with(rng_output)
                bw_graph.erase_node(bw_node)

    # Add the rng states in the output of the fwd graph. AOT Autograd assumes
    # that symints are at the end of forward graph outputs. So, insert the new
    # rng states accordingly.
    if fw_rng_state_outputs:
        fw_output_node = next(iter(fw_module.graph.find_nodes(op="output")))
        fw_outputs = fw_output_node.args[0]
        sym_node_start_idx = len(fw_outputs) - num_sym_nodes
        outputs = (
            fw_outputs[:sym_node_start_idx]
            + tuple(fw_rng_state_outputs)
            + fw_outputs[sym_node_start_idx:]
        )
        fw_module.graph.output(outputs)
        fw_module.graph.erase_node(fw_output_node)
    fw_module.recompile()
    bw_module.recompile()
    return fw_module, bw_module


def force_save_collectives(joint_module: fx.GraphModule) -> None:
    """
    By default, the partitioner is not allowed to recompute collectives
    unless they come from a user-annotated AC region.
    See Note [Recomputing collectives in the partitioner]
    """
    for node in joint_module.graph.nodes:
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and node.target.namespace == "_c10d_functional"
            and not must_recompute(node)
        ):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE


def force_save_effectful_ops(joint_module: fx.GraphModule) -> None:
    """
    Force save outputs from with_effects nodes wrapping effectful ops.

    Effectful ops (registered via _register_effectful_op) should not be recomputed
    because they may have arbitrary global side effects (I/O, RNG state, collectives,
    etc.). We mark the tensor outputs of with_effects as MUST_SAVE to prevent
    recomputation of the effectful op.

    The with_effects node returns a tuple (token, result). We recursively find all
    leaf outputs extracted via getitem and mark them as MUST_SAVE. Since these are
    saved, the with_effects op doesn't need to be recomputed in backward.
    """

    def mark_getitem_outputs(node: fx.Node) -> None:
        for user in node.users:
            if user.target is operator.getitem:
                mark_getitem_outputs(user)
                if not isinstance(user.meta.get("val"), (tuple, list)):
                    user.meta["recompute"] = CheckpointPolicy.MUST_SAVE

    for node in joint_module.graph.nodes:
        if (
            is_with_effects(node)
            and not must_recompute(node)
            and not _has_tag_is_backward(node)
        ):
            mark_getitem_outputs(node)


def force_save_bw_mutation_src(joint_module: fx.GraphModule) -> None:
    # If we have mutations of the same primal in forward and backward,
    # We must not recompute the source of mutation to not apply twice.
    has_mutation_in_bw: OrderedSet[torch.fx.Node] = OrderedSet()
    for node in reversed(joint_module.graph.nodes):
        if node.op == "output":
            continue

        is_copy_ = node.target is torch.ops.aten.copy_.default
        if is_copy_:
            if _has_tag_must_be_in_backward(node):
                has_mutation_in_bw.add(node.args[0])

            if _has_tag_must_be_in_forward(node) and node.args[0] in has_mutation_in_bw:
                node.args[1].meta["recompute"] = CheckpointPolicy.MUST_SAVE
        else:
            # We use invariant of aotdispatch joint graph,
            # That we emit copy_ only in the end of it.
            # We do not want to iterate through all the joint graph,
            # so break at the first non-output, non-copy_ node.
            break


def is_getitem_of_multi_output(node: fx.Node) -> bool:
    if node.target != operator.getitem:
        return False
    parent = node.args[0]
    if type(parent) is not fx.Node:
        raise AssertionError(f"expected parent to be fx.Node, got {type(parent)}")
    return "tensor_meta" not in parent.meta and node.op == "call_function"


def cleanup_recompute_tags(
    joint_module: fx.GraphModule, *, is_default_partition: bool
) -> fx.GraphModule:
    """
    If there are two consecutive checkpointed blocks with no operator in
    between, we would still want to stash the tensor at the boundary of
    checkpointed blocks. The following pass makes the last output node
    non-recomputable to allow for that.
    """
    for node in joint_module.graph.nodes:
        if must_recompute(node):
            for user in node.users:
                if (
                    must_recompute(user)
                    and "ac_graph_id" in user.meta
                    and "ac_graph_id" in node.meta
                    and user.meta["ac_graph_id"] > node.meta["ac_graph_id"]
                ):
                    node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            if node.meta.get("has_backward_hook", False) and not any(
                must_recompute(user) for user in node.users
            ):
                # If node is AC region output and has a backward hook on it, we intentionally choose to save it.
                # This is to work around circular dependencies with backward hooks + AC.
                # Example:
                # ```
                # out = fully_shard(utils.checkpoint(module))(x)
                # norm_out = layer_norm(out)
                # ```
                # Here there is a circular dependency:
                # 1. In backward, grad_input of layer_norm aka. `out_grad` is actually dependent on `out`.
                # 2. `out` depends on `out`'s backward hook created by FSDP2 (which does all-gather for `module` weights)
                #    in order to be recomputed.
                # 3. `out`'s backward hook, as is the case for all eager backward hooks, depends on `out_grad`
                #    -> circular dependency with (1)!
                #
                # Solution: check whether `out` has a backward hook, and if so, intentionally save `out`
                # in forward graph outputs. With this, we can break the above circular dependency.
                node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        elif (
            "ac_graph_id" not in node.meta
            and any(must_recompute(user) for user in node.users)
            and not (
                # Avoid saving getitem nodes which are not labeled with "ac_graph_id"
                is_getitem_of_multi_output(node) and "ac_graph_id" in node.args[0].meta
            )
            and is_default_partition
        ):
            # This node is not part of the AC region and a user is marked as recompute.
            # This means it's an input to the AC region and we should save it.
            # For ease of landing, gate this to default partitioner only, but we should think
            # about flipping the switch in general as well.
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    return joint_module


def solve_min_cut(
    joint_graph: fx.Graph,
    node_info: NodeInfo,
    min_cut_options: MinCutOptions,
    dont_ban: OrderedSet[fx.Node] | None = None,
) -> tuple[list[fx.Node], OrderedSet[fx.Node]]:
    if dont_ban is None:
        dont_ban = OrderedSet()
    op_types = get_default_op_list()

    if AOT_PARTITIONER_DEBUG:
        joint_module_ops = OrderedSet(
            str(node.target._overloadpacket)
            for node in joint_graph.nodes
            if node.op == "call_function" and hasattr(node.target, "_overloadpacket")
        )
        ops_ignored = joint_module_ops - OrderedSet(
            str(i) for i in op_types.recomputable_ops
        )
        log.info("Ops banned from re-materialization: %s", ops_ignored)

    def can_fuse_into_auto_functionalized(a: fx.Node, b: fx.Node) -> bool:
        if b.target != torch.ops.higher_order.auto_functionalized:
            return False
        mutable_op = b.args[0]
        (
            mutable_arg_names,
            _,
        ) = torch._higher_order_ops.auto_functionalize.get_mutable_args(
            # pyrefly: ignore[bad-argument-type]
            mutable_op
        )
        for name in mutable_arg_names:  # pyrefly: ignore [not-iterable]
            arg = b.kwargs[name]
            if a is arg:
                return True
            if isinstance(arg, list):
                if a in arg:
                    return True
        return False

    def can_fuse_into_triton_kernel_wrapper_functional(a: fx.Node, b: fx.Node) -> bool:
        if b.target != torch.ops.higher_order.triton_kernel_wrapper_functional:
            return False
        mutable_arg_names = b.kwargs["tensors_to_clone"]
        for name in mutable_arg_names:  # pyrefly: ignore [not-iterable]
            kwargs: Any = b.kwargs["kwargs"]
            if kwargs is None:
                raise AssertionError("kwargs must not be None")
            arg = kwargs[name]
            if a is arg:
                return True
        return False

    def is_fusible(a: fx.Node, b: fx.Node) -> bool:
        # We can perform "memory fusion" into a cat, but cat cannot be a
        # producer to a fusion
        if get_aten_target(b) == aten.cat:
            return True
        if can_fuse_into_auto_functionalized(a, b):
            return True
        if can_fuse_into_triton_kernel_wrapper_functional(a, b):
            return True
        if (
            a.target is operator.getitem
            and a.args[0].target  # pyrefly: ignore [missing-attribute]
            is torch.ops.higher_order.triton_kernel_wrapper_functional
        ):
            # if a is the output of a user triton kernel,
            # then (by default) we will not be able to fuse b into it
            return False
        return op_types.is_fusible(a) and op_types.is_fusible(b)

    try:
        import networkx as nx
    except ImportError as e:
        raise RuntimeError(
            "Need networkx installed to perform smart recomputation heuristics"
        ) from e

    def is_materialized_backwards(node: fx.Node) -> bool:
        if op_types.is_view(node):
            return False
        cur_nodes = OrderedSet([node])
        while len(cur_nodes) > 0:
            cur = cur_nodes.pop()
            for user in cur.users:
                if not node_info.is_required_fw(user) and not is_fusible(cur, user):
                    return True
                if op_types.is_view(user):
                    cur_nodes.add(user)

        return False

    def should_ban_recomputation(node: fx.Node) -> str | None:
        """Returns reason string if node should be banned from recomputation, None otherwise."""
        if node.op != "call_function":
            return None
        if node.target is operator.getitem:
            return None
        if node.meta.get("recompute", None) == CheckpointPolicy.MUST_SAVE:
            return "marked MUST_SAVE"
        if config.recompute_views and op_types.is_view(node):
            return None
        if node.target in [aten.lift_fresh_copy.default, aten.lift_fresh.default]:
            return None

        if min_cut_options.ban_if_not_in_allowlist:
            if not op_types.is_recomputable(node):
                return "not in recomputable allowlist"
        else:
            if op_types.is_random(node):
                return "random op"
            if op_types.is_compute_intensive(node):
                return "compute intensive op"
            if is_non_builtin_to_include(node):
                return "non-builtin op"

        # If a node *must* be materialized in the backwards pass, then we
        # should never recompute it. This is a pretty subtle point.  In
        # general, the assumption we make is that recomputing a node in the
        # backwards pass is "free". However, if a node must be materialized
        # in the backwards pass, then recomputing it is never free.
        if min_cut_options.ban_if_materialized_backward and is_materialized_backwards(
            node
        ):
            log.debug("materialized backwards: %s %s", node, tuple(node.users))
            return "materialized in backward"

        # Arbitrary hack that sometimes seems to help things. The above
        # modification appears to have made this heuristic a lot less critical
        # for performance.
        # NB: As of PR #121692, this hack no longer seems necessary.
        if (
            # pyrefly: ignore [missing-attribute]
            node.dist_from_bw < 1000 and node.dist_from_bw > config.max_dist_from_bw
        ):
            return "too far from backward"

        # If the output of an op is 4x smaller (arbitrary choice),
        # then we don't allow recomputation. The idea here is that for
        # things like reductions, saving the output of the reduction is very
        # cheap/small, and it makes sure we don't do things like recompute
        # normalizations in the backwards.
        if min_cut_options.ban_if_reduction:
            input_tensors_size = sum(
                _size_of(i) for i in node.args if isinstance(i, fx.Node)
            )
            output_size = _size_of(node)
            if output_size * 4 < input_tensors_size:
                return "reduction op"
        return None

    def is_materialized(node: fx.Node) -> bool:
        if node.op == "placeholder":
            return True

        return not all(is_fusible(node, user) for user in node.users)

    def get_node_weight(
        node: fx.Node, static_lifetime_input_nodes: OrderedSet[fx.Node]
    ) -> tuple[float, str | None]:
        """Returns (weight, cannot_save_reason).

        cannot_save_reason is None for finite weights, or a string explaining
        why the node cannot be saved for infinite weights.
        """
        if (
            config.treat_parameters_as_free_to_save
            and node in static_lifetime_input_nodes
        ):
            return 0, None
        mem_sz = _size_of(node)
        if config.recompute_views and op_types.is_view(node):
            # If `config.recompute_views=True`, we don't save views. This is generally
            # a good idea since views are free to recompute, and it makes it a bit simpler
            # to analyze.
            # NB: If they're not free to recompute (e.g. nested tensors)... I
            # think we should modify checks for view_ops to `is_view` and check
            # that. Basically, with nested tensors, `aten.view` is not a "view
            # op".
            return math.inf, "view op (recompute_views=True)"

        if isinstance(node.meta["val"], py_sym_types):
            # We never want to save symfloats
            if not isinstance(node.meta["val"], torch.SymInt):
                return INT_INF, "SymFloat (non-SymInt symbolic value)"

        # Heuristic to bias towards nodes closer to the backwards pass
        # Complete guess about current value
        mem_sz = int(
            # pyrefly: ignore [missing-attribute]
            mem_sz * (1.1 ** max(min(node.dist_from_bw, 100), 1))
        )
        if is_materialized(node):
            return mem_sz, None
        else:
            return mem_sz * 2, None

    nx_graph = nx.DiGraph()
    banned_nodes: OrderedSet[fx.Node] = OrderedSet()

    def ban_recomputation_if_allowed(node: fx.Node, reason: str = "") -> bool:
        if op_types.is_view(node):
            return False
        if node in dont_ban:
            # collectives are *always* banned from recompute, overriding `dont_ban`
            # (in particular, the activation memory budget logic is not allowed to recompute collectives)
            is_collective = (
                isinstance(node.target, torch._ops.OpOverload)
                and node.target.namespace == "_c10d_functional"
            )
            if config.unsafe_allow_optimization_of_collectives or not is_collective:
                return False
        # This bans recomputation of the node unless we've been forced not to by
        # user annotation
        if must_recompute(node):
            return False

        if "val" in node.meta and isinstance(node.meta["val"], torch.SymFloat):
            return False
        banned_nodes.add(node)
        # A node will only ever be recomputed if there is a path from an
        # ancestor of this node to the backwards path through this node that
        # doesn't go through any saved value. If this node is saved, then that
        # condition is not possible.
        nx_graph.add_edge(
            "source",
            node.name + "_in",
            capacity=math.inf,
            reason=f"cannot recompute: {reason}" if reason else "cannot recompute",
        )
        return True

    for node in joint_graph.nodes:
        if node.op == "output":
            continue

        if node in node_info.required_bw_nodes:
            # See Note: [tangents_closure vs required_bw_nodes]
            if node not in node_info.tangents_closure:
                nx_graph.add_edge(
                    node.name + "_out",
                    "sink",
                    capacity=math.inf,
                    reason="must be available for backward: input required for gradient",
                )
            else:
                nx_graph.add_edge(
                    node.name + "_in",
                    "sink",
                    capacity=math.inf,
                    reason="must be computed in backward: required for gradient",
                )
                continue

        if must_recompute(node):
            # If user explicitly says they want to recompute a node, we honor it
            # by adding an inf-capacity edge from X_in to the sink.
            # This way, X_in node is guaranteed to be part of the subgraph that contains "sink"
            # after the cut, thus guaranteeing that X op will be recomputed.
            nx_graph.add_edge(
                node.name + "_in",
                "sink",
                capacity=math.inf,
                reason="must recompute: marked by checkpoint policy",
            )
            continue

        if _is_primal(node):
            ban_recomputation_if_allowed(node, "primal input")
        elif _is_fwd_seed_offset(node):
            ban_recomputation_if_allowed(node, "forward RNG seed")

        # If a node can't be recomputed (too expensive or involves randomness),
        # we prevent it from being recomputed by adding an inf edge to the source
        # We only need to ban nodes in the fw pass, as those are the only ones that would be recomputed.
        ban_reason = should_ban_recomputation(node)
        if node_info.is_required_fw(node) and ban_reason:
            ban_recomputation_if_allowed(node, ban_reason)

        # Checks if a node is actually a tuple. Can be simplified to just an isinstance check if we always use faketensors.
        is_non_tensor_node = (
            "val" not in node.meta and "tensor_meta" not in node.meta
        ) or ("val" in node.meta and not isinstance(node.meta["val"], torch.Tensor))

        if is_sym_node(node):
            weight = float(sym_node_size(node))
            cannot_save_reason = None
        elif is_non_tensor_node:
            # FakeScriptObjects and opaque objects should have weight 0.0
            # so they can be properly partitioned between forward and
            # backward, like BackwardState.
            if isinstance(
                node.meta.get("val"), (BackwardState, FakeScriptObject)
            ) or is_opaque_node(node):
                weight = 0.0
                cannot_save_reason = None
            else:
                weight = math.inf
                cannot_save_reason = "non-tensor output"
        else:
            weight, cannot_save_reason = get_node_weight(
                node, node_info.static_lifetime_input_nodes
            )

        # Creates the weights on the "node" edge
        if cannot_save_reason and (weight == math.inf or weight == INT_INF):
            nx_graph.add_edge(
                node.name + "_in",
                node.name + "_out",
                capacity=weight,
                reason=f"cannot save: {cannot_save_reason}",
            )
        else:
            nx_graph.add_edge(node.name + "_in", node.name + "_out", capacity=weight)

        for user in node.users:
            nx_graph.add_edge(
                node.name + "_out",
                user.name + "_in",
                capacity=math.inf,
                reason="data dependency",
            )

    # todo(chilli): This is the most questionable of the 3 heuristics for banning recompute.
    # Some example models to look at where this helps perf: poolformer_m36,
    # mixer_b16_224, cait_m36_384

    # The "rough" idea here is that if you have some node that is used by both a
    # node nearby downstream as well as a node far downstream, if we recompute
    # both of the downstream nodes, we're unlikely to be able to fuse both
    # downstream nodes together.

    # Thus, we shouldn't aim to recompute far downstream nodes that depend on
    # this node. That intuition of "far downstream" is captured by whether
    # there's an unfusible op along the chain somewhere

    # It could probably be improved by properly analyzing what's going on in the
    # backwards pass instead of only relying on whether it's unfusible in the
    # forwards.

    def find_first_unfusible(start_nodes: list[fx.Node], max_range: int) -> int:
        """
        Finds the first unfusible node in the chain of nodes starting from
        `start_nodes` and returns its position.
        """
        sorted_nodes: list[tuple[int, fx.Node, bool]] = []
        for n in start_nodes:
            heapq.heappush(sorted_nodes, (node_info.get_fw_order(n), n, True))

        while len(sorted_nodes) > 0:
            _, node, node_is_fusible = heapq.heappop(sorted_nodes)
            if not node_is_fusible:
                return node_info.get_fw_order(node)
            for user in node.users:
                if node_info.is_required_fw(user):
                    if node_info.get_fw_order(user) > max_range:
                        continue
                    val: tuple[int, fx.Node, bool] = (
                        node_info.get_fw_order(user),
                        user,
                        is_fusible(node, user),
                    )
                    if val not in sorted_nodes:
                        heapq.heappush(sorted_nodes, val)
        return max_range

    if min_cut_options.ban_if_used_far_apart:
        for used_node in node_info.required_fw_nodes:
            orders = [
                node_info.get_fw_order(user)
                for user in used_node.users
                if node_info.is_required_fw(user)
            ]
            fw_users = [
                user for user in used_node.users if node_info.is_required_fw(user)
            ]
            if len(orders) > 0:
                first_unfusible_use = find_first_unfusible(fw_users, max(orders))
                for user in tuple(used_node.users):
                    if (
                        node_info.is_required_fw(user)
                        and node_info.get_fw_order(user) > first_unfusible_use
                        and is_fusible(used_node, user)
                    ):
                        if user in banned_nodes:
                            continue
                        log.info(
                            "used above/below fusible %s:(%s) -> %s -> %s:(%s)",
                            used_node,
                            node_info.get_fw_order(used_node),
                            first_unfusible_use,
                            user,
                            node_info.get_fw_order(user),
                        )
                        ban_recomputation_if_allowed(user)

    # This heuristic is fairly straightforward. The idea is that although it is
    # cheap to recompute bandwidth-bound ops, we don't want to end up in a situation
    # where we have a long chain of pointwise ops from the beginning to the end
    # of the model (like say, residual connections)

    # todo: I'm not totally sure why this heuristic matters. It's possible that this is
    # working around Inductor fusion decisions, or that it's a patch over
    # suboptimal partitioning decisions

    # Some models it improves perf on are cait_m36_384, mixer_b16_224, poolformer_m36

    if min_cut_options.ban_if_long_fusible_chains:
        visited: OrderedSet[fx.Node] = OrderedSet()
        for start_node in joint_graph.nodes:
            if not node_info.is_required_fw(start_node):
                continue
            fusible: list[tuple[int, fx.Node]] = [
                (node_info.get_fw_order(start_node), start_node)
            ]
            start_order = node_info.get_fw_order(start_node)
            while len(fusible) > 0:
                _, cur = heapq.heappop(fusible)
                if cur in visited:
                    continue
                visited.add(cur)
                # 100 is arbitrary choice to try and prevent degenerate cases
                if (
                    node_info.get_fw_order(cur) > start_order + 100
                    and len(fusible) == 0
                ):
                    log.info(
                        "too long %s %s %s %s",
                        cur,
                        start_node,
                        node_info.get_fw_order(cur),
                        node_info.get_fw_order(start_node),
                    )
                    ban_recomputation_if_allowed(cur)
                    break

                for user in cur.users:
                    if (
                        node_info.is_required_fw(user)
                        and is_fusible(cur, user)
                        and user not in banned_nodes
                    ):
                        heapq.heappush(fusible, (node_info.get_fw_order(user), user))

    try:
        cut_value, partition = nx.minimum_cut(nx_graph, "source", "sink")
    except nx.NetworkXUnbounded as unbounded_exc:
        # Check if structured tracing is enabled (for production job debugging via tlparse)
        structured_tracing_enabled = bool(trace_log.handlers)

        # Dump the FX graph for debugging
        fx_graph_file: str | None = None
        fx_graph_str: str | None = None
        joint_module = joint_graph.owning_module
        try:
            fx_graph_str = (
                joint_module.print_readable(
                    print_output=False, include_stride=True, include_device=True
                )
                if joint_module is not None
                else str(joint_graph)
            )
            # Always log to structured trace for production debugging
            trace_structured(
                "artifact",
                metadata_fn=lambda: {
                    "name": "min_cut_failed_fx_graph",
                    "encoding": "string",
                },
                payload_fn=lambda: fx_graph_str,
            )
            # Also write to local file for local debugging
            fx_graph_file = _get_unique_path("min_cut_failed_graph", ".txt")
            with open(fx_graph_file, "w") as f:
                f.write(fx_graph_str)
        except Exception as e:
            fx_graph_file = f"(failed to write: {e})"

        # Dump the min-cut edge list to structured trace
        edge_list_str = "\n".join(nx.readwrite.edgelist.generate_edgelist(nx_graph))
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "min_cut_failed_edge_list",
                "encoding": "string",
            },
            payload_fn=lambda: edge_list_str,
        )

        # Find and report the infinite-capacity path
        inf_path = _find_infinite_capacity_path(nx_graph)
        if inf_path:
            # Group edges by FX node and format for user understanding
            # inf_path is a list of (from_node, to_node, reason) tuples
            #
            # Edge types and what they mean:
            # - source -> X_in: X cannot be recomputed
            # - X_in -> X_out (inf): X's output cannot be saved
            # - X_out -> Y_in: Y depends on X (data flow)
            # - X_in -> sink: X must be computed in backward
            # - X_out -> sink: X's output must be available for backward

            # Build a user-friendly explanation grouped by FX node
            node_constraints: dict[str, list[str]] = {}
            raw_path_nodes = ["source"]

            def get_base_name(node_name: str) -> str:
                for suffix in ("_in", "_out"):
                    if node_name.endswith(suffix):
                        return node_name[: -len(suffix)]
                return node_name

            for from_node, to_node, reason in inf_path:
                raw_path_nodes.append(to_node)

                # Skip source/sink, focus on FX nodes
                if from_node == "source":
                    base = get_base_name(to_node)
                    node_constraints.setdefault(base, []).append(reason)
                elif to_node == "sink":
                    base = get_base_name(from_node)
                    node_constraints.setdefault(base, []).append(reason)
                elif get_base_name(from_node) == get_base_name(to_node):
                    # Internal edge (X_in -> X_out)
                    base = get_base_name(from_node)
                    node_constraints.setdefault(base, []).append(reason)
                else:
                    # Data dependency edge (X_out -> Y_in)
                    from_base = get_base_name(from_node)
                    to_base = get_base_name(to_node)
                    node_constraints.setdefault(to_base, []).append(
                        f"depends on {from_base}"
                    )

            # Format the constraints nicely
            constraint_lines: list[str] = []
            for node_name, constraints in node_constraints.items():
                constraint_lines.append(f"  {node_name}:")
                for c in constraints:
                    constraint_lines.append(f"    - {c}")

            constraints_str = "\n".join(constraint_lines)
            raw_path_str = " -> ".join(raw_path_nodes)

            # Try to visualize (logs to structured trace and writes local file)
            svg_path, svg_content = visualize_min_cut_graph(nx_graph)
            if svg_content:
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "min_cut_failed_svg",
                        "encoding": "string",
                    },
                    payload_fn=lambda: svg_content,
                )

            # Build file location messages
            local_files_msg = (
                f"FX graph dump: {fx_graph_file}\n" if fx_graph_file else ""
            )
            if svg_path:
                local_files_msg += f"Min-cut graph visualization: {svg_path}\n"

            # Suggest tlparse if structured tracing is enabled
            tlparse_msg = ""
            if structured_tracing_enabled:
                tlparse_msg = (
                    "[Production debugging: Use tlparse to extract debug artifacts "
                    "(min_cut_failed_fx_graph, min_cut_failed_edge_list, min_cut_failed_svg)]\n"
                )

            raise RuntimeError(
                f"AOT Autograd failed to partition the joint forward-backward graph.\n\n"
                f"The partitioner determines which intermediate values to save from the "
                f"forward pass vs recompute in the backward pass. This error means a value "
                f"is required for backward, but cannot be saved AND cannot be recomputed.\n\n"
                f"This is a bug in PyTorch. Please file an issue at "
                f"https://github.com/pytorch/pytorch/issues\n\n"
                f"Nodes involved in the conflict:\n"
                f"{constraints_str}\n\n"
                f"[For PyTorch developers: one of the above constraints is wrong. "
                f"Either the node should be recomputable, saveable, or not required for backward.]\n\n"
                f"[Debug: min-cut path] {raw_path_str}\n"
                f"{local_files_msg}"
                f"{tlparse_msg}"
            ) from unbounded_exc

        # Fallback if we couldn't find the path
        log.info("Failed to compute min-cut on following graph:")
        log.info(
            "%s",
            LazyString(
                lambda: "\n".join(nx.readwrite.edgelist.generate_edgelist(nx_graph))
            ),
        )
        visualize_min_cut_graph(nx_graph)
        raise
    except Exception:
        log.info("Failed to compute min-cut on following graph:")
        log.info(
            "%s",
            LazyString(
                lambda: "\n".join(nx.readwrite.edgelist.generate_edgelist(nx_graph))
            ),
        )
        visualize_min_cut_graph(nx_graph)
        raise

    reachable, non_reachable = partition
    cutset: OrderedSet[tuple[str, str]] = OrderedSet()
    for u, nbrs in ((n, nx_graph[n]) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    cut_nodes: OrderedSet[str] = OrderedSet()
    for node_in, node_out in cutset:
        if node_in[:-3] != node_out[:-4]:
            raise AssertionError(
                f"node_in[:-3]={node_in[:-3]} != node_out[:-4]={node_out[:-4]}"
            )
        node_name = node_in[:-3]
        cut_nodes.add(node_name)

    name_to_node = get_name_to_node(joint_graph)
    # To make this stuff deterministic
    node_idx = {node: idx for idx, node in enumerate(joint_graph.nodes)}
    saved_values = sorted(
        (name_to_node[node] for node in cut_nodes), key=lambda x: node_idx[x]
    )
    return saved_values, banned_nodes


def _find_infinite_capacity_path(
    nx_graph: nx.DiGraph[str, dict[str, Any]],
) -> list[tuple[str, str, str]] | None:
    """BFS from source to sink following only infinite-capacity edges.

    Returns a list of (from_node, to_node, reason) tuples representing the path,
    or None if no such path exists.
    """

    visited = OrderedSet(["source"])
    # Each queue item: (current_node, path_of_edges)
    # where path_of_edges is a list of (from_node, to_node, reason) tuples
    queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque([("source", [])])

    while queue:
        node, edge_path = queue.popleft()
        for neighbor in nx_graph.successors(node):
            if neighbor in visited:
                continue
            edge_data = nx_graph[node][neighbor]
            capacity = edge_data.get("capacity", 0)
            # Check for infinite capacity (either math.inf or INT_INF)
            if capacity == math.inf or capacity == INT_INF:
                reason = edge_data.get("reason", "unknown")
                new_edge = (node, neighbor, reason)
                new_path = edge_path + [new_edge]
                if neighbor == "sink":
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))
    return None


def _get_unique_path(base_name: str, extension: str) -> str:
    """Get a unique file path, appending a counter if the file already exists.

    For example, if "min_cut_failed.svg" exists, returns "min_cut_failed_1.svg".
    """
    path = f"{base_name}{extension}"
    if not os.path.exists(path):
        return path

    counter = 1
    while os.path.exists(f"{base_name}_{counter}{extension}"):
        counter += 1
    return f"{base_name}_{counter}{extension}"


def visualize_min_cut_graph(
    nx_graph: nx.DiGraph[str, dict[str, Any]],
) -> tuple[str | None, str | None]:
    """Visualize the min-cut graph to an SVG file.

    Returns (path_to_svg, svg_content) tuple. Both are None if pydot is unavailable.
    """
    import networkx as nx

    try:
        import pydot
    except ImportError:
        log.info(
            "Install pydot to visualize the min-cut graph for debugging: pip install pydot",
            exc_info=True,
        )
        return None, None

    dot_format = nx.nx_pydot.to_pydot(nx_graph).to_string()
    dot_graph = pydot.graph_from_dot_data(dot_format)[0]  # type: ignore[index]
    for edge in dot_graph.get_edges():
        weight = nx_graph[edge.get_source()][edge.get_destination()]["capacity"]
        # Set edge label to weight
        edge.set_label(str(weight))  # type: ignore[union-attr]
        # Color edges with weight 'inf' as red
        if weight == float("inf"):
            edge.set_color("red")  # type: ignore[union-attr]

    # Generate SVG content
    svg_content = dot_graph.create_svg().decode("utf-8")  # type: ignore[union-attr]

    # Write to local file
    svg_path = _get_unique_path("min_cut_failed", ".svg")
    with open(svg_path, "w") as f:
        f.write(svg_content)

    return svg_path, svg_content


def get_default_op_list() -> OpTypes:
    default_recomputable_ops: list[Callable[..., Any]] = [
        aten.add,
        aten.sub,
        aten.div,
        aten.atan2,
        aten.mul,
        aten.max,
        aten.min,
        aten.pow,
        aten.remainder,
        aten.fmod,
        aten.__and__,
        aten.__or__,
        aten.__xor__,
        aten.__lshift__,
        aten.__rshift__,
        aten.eq,
        aten.ne,
        aten.ge,
        aten.gt,
        aten.le,
        aten.lt,
        aten.abs,
        aten.bitwise_not,
        aten.ceil,
        aten.floor,
        aten.frac,
        aten.neg,
        aten.relu,
        aten.round,
        aten.silu,
        aten.trunc,
        aten.log,
        aten.log10,
        aten.log1p,
        aten.log2,
        aten.lgamma,
        aten.exp,
        aten.expm1,
        aten.erf,
        aten.erfc,
        aten.cos,
        aten.acos,
        aten.cosh,
        aten.sin,
        aten.asin,
        aten.sinh,
        aten.tan,
        aten.atan,
        aten.tanh,
        aten.atanh,
        aten.sqrt,
        aten.rsqrt,
        aten.reciprocal,
        aten.sigmoid,
        aten.softplus,
        aten.threshold,
        aten.threshold_backward,
        aten.clamp,
        aten.where,
        aten.lerp,
        aten.addcmul,
        aten.gelu,
        aten.gelu_backward,
        aten.sum,
        aten.mean,
        aten._grad_sum_to_size,
        aten.sum_to_size,
        aten.amax,
        aten.to,
        aten.type_as,
        operator.getitem,
        aten.squeeze,
        aten.unsqueeze,
        aten.rsub,
        aten._to_copy,
    ]
    recomputable_view_ops = [aten.squeeze, aten.unsqueeze, aten.alias]
    recomputable_view_ops += [
        aten.view,
        aten.slice,
        aten.t,
        prims.broadcast_in_dim,
        aten.expand,
        aten.as_strided,
        aten.permute,
        aten.select,
        aten.split,
    ]
    view_ops = recomputable_view_ops
    default_recomputable_ops += [
        prims.div,
        prims.convert_element_type,
        prims.prepare_softmax_online,
        aten.clone,
        aten._to_copy,
        aten.full_like,
        prims.var,
        prims.sum,
        aten.var,
        aten.std,
        prims.broadcast_in_dim,
        aten.select,
        aten._unsafe_view,
        aten.view,
        aten.expand,
        aten.slice,
        aten.reshape,
        aten.broadcast_tensors,
        aten.scalar_tensor,
        aten.ones,
        aten.new_zeros,
        aten.lift_fresh_copy,
        aten.arange,
        aten.triu,
        aten.var_mean,
        aten.isinf,
        aten.any,
        aten.full,
        aten.as_strided,
        aten.zeros,
        aten.empty,
        aten.empty_like,
        aten.argmax,
        aten.maximum,
        prims.iota,
        prims._low_memory_max_pool_offsets_to_indices,
    ]
    # Natalia said that we should allow recomputing indexing :)
    default_recomputable_ops += [aten.index, aten.gather]
    default_recomputable_ops += view_ops

    default_recomputable_ops += pointwise_ops()

    default_recomputable_ops += [
        aten.zeros_like,
    ]

    default_recomputable_ops += [method_to_operator(m) for m in magic_methods]
    recomputable_ops = OrderedSet(default_recomputable_ops)

    random_ops = OrderedSet[Callable[..., Any]](
        [aten.native_dropout, aten.rand_like, aten.randn_like]
    )
    compute_intensive_ops = [
        aten.mm,
        aten.convolution,
        aten.convolution_backward,
        aten.bmm,
        aten.addmm,
        aten._scaled_dot_product_flash_attention,
        aten._scaled_dot_product_efficient_attention,
        aten._flash_attention_forward,
        aten._efficient_attention_forward,
        aten.upsample_bilinear2d,
        aten._scaled_mm,
    ]

    fusible_ops = recomputable_ops | random_ops
    return OpTypes(
        fusible_ops,
        OrderedSet(compute_intensive_ops),
        random_ops,
        OrderedSet(view_ops),
        recomputable_ops,
    )


def get_name_to_node(graph: fx.Graph) -> dict[str, fx.Node]:
    name_to_node: dict[str, fx.Node] = {}
    for node in graph.nodes:
        name_to_node[node.name] = node
    return name_to_node


def _optimize_runtime_with_given_memory(
    joint_graph: fx.Graph,
    memory: list[float],
    runtimes: list[float],
    max_memory: float,
    node_info: NodeInfo,
    all_recomputable_banned_nodes: list[fx.Node],
) -> tuple[float, list[int], list[int]]:
    SOLVER = config.activation_memory_budget_solver
    if SOLVER == "greedy":
        return greedy_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "ilp":
        return ilp_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "dp":
        return dp_knapsack(memory, runtimes, max_memory)
    elif SOLVER == "dp_knapsack_sliding_hirschberg":
        return dp_knapsack_sliding_hirschberg(memory, runtimes, max_memory)
    elif SOLVER == "dynamic_memory_budget_dp":
        log.warning(
            "dynamic_memory_budget_dp is an experimental solver. "
            "It does not guarantee performance improvements. "
            "Additionally, it is not guaranteed to be stable."
        )
        graph_info_provider = GraphInfoProvider.inialize_from_graph(
            joint_graph=joint_graph,
            all_recomputable_banned_nodes=all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=memory,
            recorded_knapsack_input_runtimes=runtimes,
        )
        return dp_knapsack(
            memory,
            runtimes,
            KnapsackEvaluator(
                graph_info_provider=graph_info_provider,
            ).get_knee_point_memory_budget(
                knapsack_algo=dp_knapsack,
                max_mem_budget=max_memory,
            ),
        )
    elif isinstance(SOLVER, CustomKnapsackSolver):
        saved_node_idx, recomp_node_idx = SOLVER(
            memory, joint_graph, max_memory, node_info, all_recomputable_banned_nodes
        )
        return (0.0, saved_node_idx, recomp_node_idx)
    else:
        raise RuntimeError(f"Not aware of memory budget knapsack solver: {SOLVER}")


from torch.utils._mode_utils import no_dispatch


# replace symbols in size and strides with their hints without guarding.
def _remove_symbols_without_guarding(x: torch.Tensor, fallback: int) -> torch.Tensor:
    shape = list(x.shape)

    def realize_symbol(d: torch.SymInt | int) -> int:
        return optimization_hint(d, fallback=fallback)

    shape = [realize_symbol(s) for s in shape]
    stride = [realize_symbol(s) for s in x.stride()]
    return x.new_empty_strided(shape, stride=stride)


def estimate_runtime(node: fx.Node) -> float:
    RUNTIME_MODE = config.activation_memory_budget_runtime_estimator

    def materialize_arg(x: Any) -> Any:
        if isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.Tensor):
            return _remove_symbols_without_guarding(x.meta["val"], fallback=4096)
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymInt):
            return optimization_hint(x.meta["val"], fallback=4096)
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymFloat):
            return 1.0
        elif isinstance(x, fx.Node) and isinstance(x.meta["val"], torch.SymBool):
            return True
        else:
            return x

    if RUNTIME_MODE == "testing":
        return 1

    elif RUNTIME_MODE == "profile":
        with no_dispatch():
            from torch._inductor.runtime.benchmarking import benchmarker

            args, kwargs = pytree.tree_map(materialize_arg, (node.args, node.kwargs))
            # pyrefly: ignore[not-callable]
            ms = benchmarker.benchmark_gpu(lambda: node.target(*args, **kwargs))
            return ms

    elif RUNTIME_MODE == "flops":
        # todo(chilli): Normalize this to also return ms
        from torch.utils.flop_counter import FlopCounterMode

        args, kwargs = pytree.tree_map(materialize_arg, (node.args, node.kwargs))
        with FlopCounterMode(display=False) as mode:
            # pyrefly: ignore[not-callable]
            node.target(*args, **kwargs)
        counted_flops = mode.get_total_flops()
        return max(counted_flops, 1)

    elif isinstance(RUNTIME_MODE, CustomRuntimeEstimator):
        return RUNTIME_MODE(node)

    else:
        raise RuntimeError(f"Not aware of runtime estimator: {RUNTIME_MODE}")


def choose_saved_values_set(
    joint_graph: fx.Graph,
    node_info: NodeInfo,
    memory_budget: float = 1,
) -> list[fx.Node]:
    if memory_budget > 1 or memory_budget < 0:
        raise RuntimeError(
            f"The valid ranges for memory budget are 0 <= m <= 1. The provided value is {memory_budget}"
        )
    min_cut_options = MinCutOptions(
        ban_if_used_far_apart=config.ban_recompute_used_far_apart,
        ban_if_long_fusible_chains=config.ban_recompute_long_fusible_chains,
        ban_if_materialized_backward=config.ban_recompute_materialized_backward,
        ban_if_not_in_allowlist=config.ban_recompute_not_in_allowlist,
        ban_if_reduction=config.ban_recompute_reductions,
    )

    if config.aggressive_recomputation:
        min_cut_options = replace(
            min_cut_options,
            ban_if_used_far_apart=False,
            ban_if_long_fusible_chains=False,
            ban_if_materialized_backward=False,
            ban_if_not_in_allowlist=False,
        )
    if memory_budget == 0:
        return node_info.inputs

    runtime_optimized_saved_values, _ = solve_min_cut(
        joint_graph,
        node_info,
        min_cut_options,
    )
    # return runtime_optimized_saved_values
    if memory_budget == 1:
        return runtime_optimized_saved_values

    def estimate_activations_size(saved_values: list[fx.Node]) -> float:
        return sum(map(_size_of, saved_values)) / 1e9

    min_act_size = estimate_activations_size(node_info.inputs)
    max_act_size = estimate_activations_size(runtime_optimized_saved_values)
    # The optimized choice is smaller than the inputs anyways
    if max_act_size <= min_act_size:
        return runtime_optimized_saved_values

    def get_normalized_size(sz: float) -> float:
        return (sz / 1e9) / (max_act_size - min_act_size)

    def get_mem_ratio(activations: list[fx.Node]) -> float:
        return (estimate_activations_size(activations) - min_act_size) / (
            max_act_size - min_act_size
        )

    more_aggressive_options = replace(
        min_cut_options,
        ban_if_used_far_apart=False,
        ban_if_long_fusible_chains=False,
        ban_if_materialized_backward=False,
    )
    more_aggressive_saved_values, _ = solve_min_cut(
        joint_graph, node_info, more_aggressive_options
    )
    more_aggressive_saved_values_mem_ratio = get_mem_ratio(more_aggressive_saved_values)
    if more_aggressive_saved_values_mem_ratio < memory_budget:
        return more_aggressive_saved_values

    aggressive_options = replace(
        more_aggressive_options,
        ban_if_not_in_allowlist=False,
    )
    aggressive_recomputation_saved_values, banned_nodes = solve_min_cut(
        joint_graph, node_info, aggressive_options
    )

    aggressive_recomputation_saved_values_mem_ratio = get_mem_ratio(
        aggressive_recomputation_saved_values
    )
    if aggressive_recomputation_saved_values_mem_ratio < memory_budget:
        return aggressive_recomputation_saved_values

    from torch._inductor.fx_utils import get_node_storage

    input_storages = OrderedSet(get_node_storage(node) for node in node_info.inputs)

    def get_recomputable_banned_nodes(
        banned_nodes: OrderedSet[fx.Node],
    ) -> list[fx.Node]:
        return [
            i
            for i in banned_nodes
            if (
                # Only allow recomputing nodes that are actually required for BW
                i.dist_from_bw < int(1e9)  # type: ignore[attr-defined]
                and (
                    get_node_storage(i) not in input_storages
                    or is_non_builtin_to_include(i)
                )
            )
        ]

    recomputable_banned_nodes = get_recomputable_banned_nodes(banned_nodes)
    must_save_nodes = [
        i
        for i in recomputable_banned_nodes
        if i.meta.get("recompute", False) == CheckpointPolicy.MUST_SAVE
    ]
    recomputable_banned_nodes = [
        i for i in recomputable_banned_nodes if i not in must_save_nodes
    ]

