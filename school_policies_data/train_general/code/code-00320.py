            # (to detect returning tensors whose grad_fn was consumed by autograd.grad)
            # Skip if retain_graph=True or create_graph=True since the graph is not
            # consumed in those cases and can be traversed again.
            retain_graph = kwargs.get("retain_graph")
            create_graph = kwargs.get("create_graph")
            graph_preserved = (
                isinstance(retain_graph, ConstantVariable)
                and retain_graph.value is True
            ) or (
                isinstance(create_graph, ConstantVariable)
                and create_graph.value is True
            )
            if not graph_preserved:
                # Filter out AccumulateGrad nodes - they're never actually "consumed"
                # by autograd. They just accumulate gradients into leaf.grad and can
                # be traversed multiple times without issues.
                non_leaf_consumed = {
                    gf
                    for gf in consumed_grad_fns
                    if type(gf).__name__ != "AccumulateGrad"
                }

                # Check for double-consumption: if any grad_fn was already consumed
                # by a previous autograd.grad, that's an error.
                already_consumed = tx.output.autograd_grad_consumed_grad_fns
                double_consumed = non_leaf_consumed & already_consumed
                if double_consumed:
                    unimplemented(
                        gb_type="autograd.grad with already consumed grad_fn",
                        context=f"double consumed grad_fns: {len(double_consumed)}",
                        explanation=(
                            "torch.autograd.grad() is trying to consume grad_fns that were "
                            "already consumed by a previous autograd.grad() call. This would "
                            "cause 'backward through graph a second time' errors at runtime."
                        ),
                        hints=[
                            "Use retain_graph=True in the first autograd.grad() call if you "
                            "need to compute gradients through the same graph multiple times.",
                        ],
                    )
                tx.output.autograd_grad_consumed_grad_fns.update(non_leaf_consumed)

            # Convert dict inputs to tuple for the FX graph. The engine
            # always operates on flat tuples; we reconstruct the dict after.
            inputs_var = args[1] if len(args) >= 2 else kwargs.get("inputs")
            if isinstance(inputs_var, ConstDictVariable):
                inputs_as_tuple = TupleVariable(list(inputs_var.items.values()))
                if len(args) >= 2:
                    args = (args[0], inputs_as_tuple, *args[2:])
                else:
                    kwargs = {**kwargs, "inputs": inputs_as_tuple}

            with (
                torch.fx.traceback.preserve_node_meta(),
                torch.fx.traceback._set_autograd_backward(),
            ):
                proxy = tx.output.create_proxy(
                    "call_function",
                    torch.autograd.grad,
                    *proxy_args_kwargs(args, kwargs),
                )
            result = wrap_fx_proxy(tx=tx, proxy=proxy)

            if isinstance(inputs_var, ConstDictVariable):
                if not isinstance(result, BaseListVariable):
                    raise AssertionError(
                        f"Expected BaseListVariable from autograd.grad with dict inputs, "
                        f"got {type(result)}"
                    )
                items: dict[VariableTracker, VariableTracker] = dict(
                    zip(
                        inputs_var.items.keys(),
                        result.items,
                        strict=True,
                    )
                )
                return ConstDictVariable(items)
            return result

        @register(torch._functorch.eager_transforms._autograd_grad)
        def handle_functorch_autograd_grad(
            self,
            tx: "InstructionTranslatorBase",
            *args: VariableTracker,
            **kwargs: VariableTracker,
        ) -> VariableTracker | None:
            """Graph-break when closure-captured tensors lose their grad_fn.

            NOTE [Detecting lost autograd linkage in closure-captured tensors]

            Functorch transforms (vjp, grad, jacrev) return closures that capture
            tensors with grad_fn. When such a closure is compiled separately, those
            tensors become graph placeholders whose FakeTensors lose grad_fn,
            causing _autograd_grad to silently return zeros.

            _collect_placeholder_nodes gathers placeholder nodes from the
            outputs/inputs args. For each, we compare grapharg.example (the real
            tensor, retains grad_fn) against example_value (FakeTensor, grad_fn
            is None). A mismatch means autograd linkage was lost, so we
            graph-break.

            This is a pre-check only: kwargs (retain_graph, create_graph,
            grad_outputs) don't affect linkage detection and are handled by
            the default proxy path when this returns None.
            """
            outputs_var = args[0] if len(args) >= 1 else None
            inputs_var = args[1] if len(args) >= 2 else None

            if outputs_var is None or inputs_var is None:
                return None

            output_placeholder_nodes = _collect_placeholder_nodes(outputs_var)
            input_placeholder_nodes = _collect_placeholder_nodes(inputs_var)

            if output_placeholder_nodes and input_placeholder_nodes:
                for node in output_placeholder_nodes:
                    fake = node.meta.get("example_value")
                    grapharg = node.meta.get("grapharg")
                    if (
                        grapharg is not None
                        and isinstance(fake, torch.Tensor)
                        and fake.grad_fn is None
                    ):
                        real = grapharg.example
                        if isinstance(real, torch.Tensor) and real.grad_fn is not None:
                            unimplemented(
                                gb_type="_autograd_grad with lost grad_fn linkage",
                                context="outputs lost autograd linkage during tracing",
                                explanation=(
                                    "_autograd_grad() received tensors whose grad_fn "
                                    "was lost during tracing - this silently produces "
                                    "zero gradients."
                                ),
                                hints=[
                                    "Compile the full transform instead of the returned "
                                    "closure: torch.compile(lambda x: torch.func.vjp(f, x))",
                                    *graph_break_hints.SUPPORTABLE,
                                ],
                            )
            return None

        return handlers

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> "VariableTracker":
        source = self.source and AttrSource(self.source, name)
        try:
            member = getattr(self.value, name)
        except AttributeError:
            raise_observed_exception(AttributeError, tx)
            raise

        if isinstance(
            member, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)
        ) and torch._dynamo.trace_rules.is_aten_op_or_tensor_method(member):
            return TorchInGraphFunctionVariable(member, source=source)
        return variables.GetAttrVariable(self, name, source=source)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import SymNodeVariable
        from .builder import wrap_fx_proxy

        if self.kind == AllowInGraphKind.NONSTRICT_TRACE:
            return self._call_nonstrict_traceable_function(tx, args, kwargs)

        if self.kind == AllowInGraphKind.LEAF_FUNCTION:
            return self._call_leaf_function(tx, args, kwargs)

        if self.torch_function_override_enabled(tx, args, kwargs):
            return dispatch_torch_function(tx, self, args, kwargs)

        if self.can_constant_fold_through() and check_unspec_or_constant_args(
            args, kwargs
        ):
            # constant fold functions need to be guarded.
            if self.value in constant_fold_functions_need_guards:
                if self.source is None:
                    raise AssertionError(
                        "Expected source to be set for constant fold function needing guards"
                    )
                source = CallFunctionNoArgsSource(self.source)
                install_guard(source.make_guard(GuardBuilder.EQUALS_MATCH))
            # constant fold
            try:
                return VariableTracker.build(
                    tx,
                    self.as_python_constant()(
                        *[x.as_python_constant() for x in args],
                        **{k: v.as_python_constant() for k, v in kwargs.items()},
                    ),
                )
            except (OverflowError, TypeError, ValueError) as exc:
                raise_observed_exception(
                    type(exc),
                    tx,
                    args=list(exc.args),
                )

        if self.is_tensor_method():
            name = self.value.__name__
            # Guard against inplace view op on input tensor (not supported)
            if args and args[0].is_tensor():
                tensor_var = args[0]
                # Check if input tensor and inplace_view op specifically
                if tensor_var.source is not None and hasattr(torch.ops.aten, name):
                    fn = getattr(torch.ops.aten, name)
                    if (
                        hasattr(fn, "overloads")
                        and hasattr(fn, fn.overloads()[0])
                        and torch.Tag.inplace_view
                        in getattr(fn, fn.overloads()[0]).tags
                    ):
                        unimplemented(
                            gb_type="Inplace op on input tensor",
                            context="",
                            explanation=f"Attempted to trace an inplace view op on input tensor {typestr(self.value)}.",
                            hints=[
                                *graph_break_hints.SUPPORTABLE,
                                "Ensure you do not modify input tensor in place.",
                            ],
                        )
            return self.call_tensor_method(tx, list(args), kwargs)

        special_handler = self._get_handlers().get(self.value)
        if special_handler:
            result = special_handler(self, tx, *args, **kwargs)
            if result:
                return result

        any_symints_or_symfloats = any(isinstance(x, SymNodeVariable) for x in args)

        all_ints_or_floats = all(
            isinstance(x, SymNodeVariable) or x.is_python_constant() for x in args
        )
        # Intentional abstraction violation: when tensorify is enabled,
        # skip this graph break. Scalar-only bin_ops on SymInt/SymFloat
        # args are valid symbolic arithmetic that Dynamo can trace.
        # The tensorify pass (in AOTAutograd) will convert SymFloat
        # scalar ops to tensor ops downstream; SymInt ops flow through
        # as symbolic expressions.
        if (
            getattr(self.value, "__module__", "") == "torch"
            and self.value.__name__ in bin_ops
            and any_symints_or_symfloats
            and all_ints_or_floats
            and not _is_tensorify_enabled()
        ):
            msg = f"""\
Calling {str(self.value)} on only torch.SymInt arguments is not yet supported.
To support this behavior, we need to allow const-propping tensors that store symint data.
For now, dynamo will explicitly graph break when it encounters user code with this behavior.
"""
            log.warning(msg)
            unimplemented(
                gb_type="Attempted to call torch in-graph function on only torch.SymInt arguments",
                context=f"fn={self.value}, args={args}, kwargs={kwargs}",
                explanation=(
                    f"Attempted to call {str(self.value)} (that should be put in the FX graph) on only torch.SymInt arguments. "
                    "Dynamo does not support this."
                ),
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # TODO(voz): Replace w/ dynamic shape rewrite table.
        # Ideally, we would be able to do this at ctor time, but alas we need a combination
        # of value + args to determine this.
        fn_ = self.value
        if any_symints_or_symfloats:
            torch_sym_op = f"_sym_{self.value.__name__}"
            if getattr(self.value, "__module__", None) == "math" and hasattr(
                torch, torch_sym_op
            ):
                fn_ = getattr(torch, torch_sym_op)

        # TODO for each of the following check on `out=` or `requires_grad=`
        # variant torch ops, the original function could come from a user
        # defined `@allow_in_graph` function as well, which doesn't have the
        # same semantics as the torch ops.

        # Calling fake tensor propagation can mutate the out= tensor in
        # tx.output.tracked_fakes. tracked_fakes are used to apply
        # symbolic_shape guards. Mutating them destroys the information
        # prior to tracing, which is essential for creating right
        # guards. So save the shape now, and check later if it has
        # changed. If it has, graph break.
        saved_out_shapes = None
        out_kwarg_vt = None
        if "out" in kwargs:
            out_kwarg_vt = kwargs["out"]

            # e.g., out=(t1, t2, ...)
            if isinstance(out_kwarg_vt, (TupleVariable, ListVariable)):
                saved_out_shapes = []
                for vt in out_kwarg_vt.items:
                    if vt.is_tensor():
                        shape = vt.as_proxy().node.meta["example_value"].shape
                    else:
                        shape = None
                    saved_out_shapes.append(shape)

            # e.g., out=output_tensor
            if out_kwarg_vt.is_tensor():
                saved_out_shapes = (
                    out_kwarg_vt.as_proxy().node.meta["example_value"].shape
                )

        ctx = nullcontext
        if fn_ in ops_consuming_unbacked_scalars:
            if tx.fake_mode and tx.fake_mode.shape_env:
                ctx = tx.fake_mode.shape_env.ignore_fresh_unbacked_symbols

        with ctx():
            tensor_variable = wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    fn_,
                    *proxy_args_kwargs(args, kwargs),
                ),
            )

        # Handle e.g., `torch.ones(10, requires_grad=True)`
        if (
            tensor_variable.is_tensor()
            and "requires_grad" in kwargs
            and kwargs["requires_grad"].as_python_constant()
        ):
            unimplemented(
                gb_type="Attempted to use tensor creation function with requires_grad=True",
                context=f"fn={self.value}, args={args}, kwargs={kwargs}",
                explanation="Dynamo does not support this.",
                hints=[
                    "Create the tensor outside the compiled region.",
                    "Do not set `requires_grad=True`.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # Handle e.g., `torch.add(a, b, out=result)`
        if saved_out_shapes is not None:
            # out variants of torch operators like torch.sort and torch.sigmoid
            # mutate the tensors in the out field.
            #
            # However, it's non-trivial to update all references of the old
            # `TensorVariable` to the new one returned (`result_var`), so we
            # take the conservative approach to graph break on size changes, and
            # assume other cases can fall through soundly.
            #
            # Note that although these tensor variables would hold different
            # proxies, the in-place mutation semantics is preserved in the FX
            # graph, so we won't have correctness issues.
            if isinstance(saved_out_shapes, list):
                for out_tensor_vt, saved_out_shape in zip(
                    out_kwarg_vt.items,  # type: ignore[union-attr]
                    saved_out_shapes,
                ):
                    if saved_out_shape is None:
                        # This should be extremely rare, but it's kept for now
                        # until we invest in enforcing the `out=` kwarg for only
                        # torch methods.
                        continue

                    if not out_tensor_vt.is_tensor():
                        raise AssertionError(
                            "Expected out= list element to be a tensor"
                        )
                    fake_out = out_tensor_vt.proxy.node.meta["example_value"]
                    if saved_out_shape != fake_out.shape:
                        # It's hard to get out variants with resizing on graph inputs work
                        # properly across dynamo/aot/inductor, just fall back.
                        unimplemented(
                            gb_type="Shape mismatch with out= list of tensor variants",
                            context=f"fn={self.value}, args={args}, kwargs={kwargs}",
                            explanation=(
                                f"Shape mismatch when calling {self.value} with `out=`. "
                                f"Provided `out=` shape: {saved_out_shape}. Actual shape: {fake_out.shape}."
                            ),
                            hints=[
                                *graph_break_hints.SUPPORTABLE,
                            ],
                        )
                    if not _is_supported_out_tensor_layout(
                        fake_out, false_if_dde=False
                    ):
                        # It's difficult to handle strides correctly in functionalization
                        # when calling an out= op with a non-contiguous out argument
                        unimplemented(
                            gb_type="Attempted to call op with non-contiguous `out=` list of tensors",
                            context=f"self.value={self.value}, args={args}, kwargs={kwargs}",
                            explanation="Dynamo does not support this.",
                            hints=[
                                *graph_break_hints.SUPPORTABLE,
                            ],
                        )
            else:
                if out_kwarg_vt is None:
                    raise AssertionError("Expected out= kwarg to be set")
                if not out_kwarg_vt.is_tensor():
                    raise AssertionError("Expected out= kwarg to be a tensor")
                if "example_value" not in out_kwarg_vt.as_proxy().node.meta:
                    raise AssertionError(
                        "Expected out= kwarg proxy node to have example_value metadata"
                    )
                fake_out = out_kwarg_vt.as_proxy().node.meta["example_value"]
                if saved_out_shapes != fake_out.shape:
                    # It's hard to get out variants with resizing on graph inputs work
                    # properly across dynamo/aot/inductor, just fall back.
                    unimplemented(
                        gb_type="Shape mismatch with out= tensor variant",
                        context=f"fn={self.value}, args={args}, kwargs={kwargs}",
                        explanation=(
                            f"Shape mismatch when calling {self.value} with `out=`. "
                            f"Provided `out=` shape: {saved_out_shapes}. Actual shape: {fake_out.shape}."
                        ),
                        hints=[
                            *graph_break_hints.SUPPORTABLE,
                        ],
                    )
                if not _is_supported_out_tensor_layout(fake_out, false_if_dde=True):
                    # It's difficult to handle strides correctly in functionalization
                    # when calling an out= op with a non-contiguous out argument
                    unimplemented(
                        gb_type="Attempted to call op with non-contiguous `out=` tensor",
                        context=f"self.value={self.value}, args={args}, kwargs={kwargs}",
                        explanation="Dynamo does not support this.",
                        hints=[
                            *graph_break_hints.SUPPORTABLE,
                        ],
                    )

        return tensor_variable

    def _call_nonstrict_traceable_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: "dict[str, VariableTracker]",
    ) -> VariableTracker:
        from torch._dynamo.utils import _make_inlined
        from torch._higher_order_ops.flat_apply import (
            flat_apply,
            is_graphable_type,
            is_valid_output,
            to_graphable,
        )
        from torch._higher_order_ops.invoke_leaf_function import _LeafCallable
        from torch._subclasses.fake_tensor import fake_tensor_tls
        from torch.utils._pytree import tree_flatten

        from .base import AsPythonConstantNotImplementedError
        from .builder import SourcelessBuilder, wrap_fx_proxy

        # 1. Convert `args, kwargs` into pytree-flattened proxy forms.
        #
        # Rather than reconstructing `args, kwargs` into python objects and
        # then tree_flatten them, we just let Dynamo symbolically interpret
        # `tree_flatten((args, kwargs))`. This saves us from having to
        # worry about the reconstruction logic, side effects, and guards.
        args_with_states, kwargs_with_states = self._extract_nn_module_states(
            tx, args, kwargs
        )
        flat_args_vts, input_spec_vt = unpack_iterable(
            tx,
            _make_inlined(tx, tree_flatten)(
                VariableTracker.build(tx, (args_with_states, kwargs_with_states))
            ),
        )
        if not isinstance(flat_args_vts, ListVariable):
            raise AssertionError(
                f"Expected flat_args_vts to be a ListVariable, got {type(flat_args_vts)}"
            )

        # Handle the case when the input contains a non-graphable type.
        for flat_arg_vt in flat_args_vts.items:
            arg_type = flat_arg_vt.python_type()
            if not is_graphable_type(arg_type):
                type_name = flat_arg_vt.python_type().__qualname__
                unimplemented(
                    gb_type="Invalid input type for nonstrict_trace-ed function",
                    context=f"Encountered input of type <{type_name}>.",
                    explanation=(
                        "For `nonstrict_trace`-ed functions, only basic types (e.g., torch.Tensor, int, float) "
                        "or pytree containers of those are allowed as inputs. The provided argument contains "
                        "an unsupported type."
                    ),
                    hints=[
                        "Use one of the following to register the type with pytree:\n"
                        "* `torch.utils._pytree.register_constant`\n"
                        "* `torch.utils._pytree.register_dataclass`\n"
                        "* `torch.utils._pytree.register_pytree_node`",
                    ],
                )

        # Since we checked with `is_graphable` above, `as_proxy` on the
        # flat_arg VT should always work.
        proxified_flat_args = [
            flat_arg_vt.as_proxy() for flat_arg_vt in flat_args_vts.items
        ]

        # The downstream `flat_apply` call requires the input spec; however,
        # the spec not a graphable type, so we still have to reconstruct it
        # into a python object, and store it as a constant attribute on the
        # fx graph.
        try:
            input_spec = input_spec_vt.as_python_constant()
        except AsPythonConstantNotImplementedError as e:
            typ = e.vt.python_type()
            type_name = typ.__qualname__
            import torch.utils._pytree as pytree

            if pytree.is_constant_class(typ):
                unimplemented(
                    gb_type="Input marked with `pytree.register_constant` constructed in the `torch.compile` region",
                    context=f"Input={input_spec_vt}, offending type <{type_name}>.",
                    explanation=(
                        "Calling a `nonstrict_trace`-ed function with an input that contains an object "
                        f"of type <{type_name}>, which was marked with `pytree.register_constant`. However, the object "
                        "was constructed _inside_ the `torch.compile` region. This is not supported."
                    ),
                    hints=[
                        "Construct the object _outside_ the `torch.compile` region, or submit an issue to GitHub.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                    from_exc=e,
                )
            else:
                unimplemented(
                    gb_type="Invalid use of pytree_flatten with nonstrict_trace-ed function",
                    context=f"Input={input_spec_vt}, offending type <{type_name}>.",
                    explanation=(
                        "Calling a `nonstrict_trace`-ed function where one of the inputs has been registered "
                        f"with a `pytree_flatten` that places an object of type <{type_name}> into the context."
                    ),
                    hints=[
                        "Modifying the `pytree_flatten` to avoid placing the object into the context.",
                        f"Apply one of the following to <{type_name}>:\n"
                        "* `torch.utils._pytree.register_constant`\n"
                        "* `torch.utils._pytree.register_dataclass`\n"
                        "* `torch.utils._pytree.register_pytree_node`",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                    from_exc=e,
                )

        fn = self.value

        def patched_fn(
            *args: VariableTracker, **kwargs: VariableTracker
        ) -> VariableTracker:
            # This enables reads to global/captured tensors, and we'll just
            # treat them as constants in the graph. Note that after
            # AOTDispatcher, this logic would disappear.
            old_val = fake_tensor_tls.allow_non_fake_inputs_override
            fake_tensor_tls.allow_non_fake_inputs_override = True
            try:
                res = fn(*args, **kwargs)
            finally:  # reset even when `fn` raises
                fake_tensor_tls.allow_non_fake_inputs_override = old_val
            return res

        f_callable = _LeafCallable(patched_fn)

        f_callable_proxy = tx.output.register_static_attr_and_return_proxy(
            f"{fn.__name__}_callable", f_callable
        )
        input_spec_proxy = tx.output.register_static_attr_and_return_proxy(
            fn.__name__ + "_input_spec",
            # pyrefly: ignore [unbound-name]
            input_spec,
        )
        f_callable_proxy.node.type = type(f_callable)
        # pyrefly: ignore [unbound-name]
        input_spec_proxy.node.type = type(input_spec)
        all_args = (f_callable_proxy, input_spec_proxy, *proxified_flat_args)

        # 2. Create a proxy call to `flat_apply`, then fake-tensor propagate
        # the call and wrap output into a VariableTracker.

        # What's going on here? The output of the nonstrict-traced function must
        # be something we can put into the graph. This means it has to be Tuple,
        # int, str, etc or lists/tuples of those (or lists of lists of those,
        # etc). So by default we don't handle PyTree-able outputs.

        # To handle PyTree-able outputs we flatten the output to a flattened
        # list of graph types and then trace the unflattening into the graph.
        captured_spec: TreeSpec | None = None

        def flat_apply_capture(*args: Any) -> list[object]:
            nonlocal captured_spec
            out = flat_apply(*args, checked_output=False)
            # Output is handled similar to flat_apply input but reverse by
            # tree_flattening the output and trace the unflattening. Note that
            # wrapped functions must return the same pytree structure every time
            # they're called.
            flat_out, spec = to_graphable(out)
            if captured_spec is None:
                captured_spec = spec
            else:
                if captured_spec != spec:
                    raise AssertionError(
                        "Error: nonstrict-traced functions must return the same "
                        f"output shape every time. got {spec!r} vs but expected {captured_spec!r}"
                    )
            if not is_valid_output(flat_out):
                raise AssertionError(
                    "Nonstrict-traced function returned invalid output"
                )
            return flat_out

        proxy = tx.output.create_proxy(
            "call_function", flat_apply_capture, all_args, {}
        )

        # Instead of calling tree_unflatten at runtime, symbolically trace it
        # just like we did for tree_flatten on inputs. This lets Dynamo
        # capture the unflatten into the FX graph as well.

        # Build VTs representing (flat_output_list, out_spec)
        try:
            proxy_list_vt = wrap_fx_proxy(tx, proxy)
        except (
            # From `handle_traced_output`.
            torch._dynamo.exc.Unsupported,
            # From `flat_apply` assert on output type.
            torch._dynamo.exc.TorchRuntimeError,
            # From fake tensor eval in _get_fake_value_impl.
            torch._dynamo.exc.FakeTensorObservedException,
        ):
            unimplemented(
                gb_type="Unsupported output type for nonstrict_trace-ed function",
                context=f"Function: {fn.__name__}",
                explanation=(
                    "For `nonstrict_trace`-ed functions, only basic types (e.g., torch.Tensor, int, list)"
                    " are allowed as output. The result of this call contains an unsupported type."
                ),
                hints=[*graph_break_hints.SUPPORTABLE],
            )
            # pyrefly error: why doesn't it recognize unimplemented() as NoReturn?
            raise AssertionError("unreachable")  # noqa: B904

        if captured_spec is None:
            raise AssertionError("captured_spec was not set during nonstrict trace")
        out_spec_vt = VariableTracker.build(tx, captured_spec)

        # Reuse the same pattern used above for tree_flatten: call the python
        # function through Dynamo so it symbolically interprets it.
        out_vt = SourcelessBuilder.create(tx, _pytree.tree_unflatten).call_function(
            tx, [proxy_list_vt, out_spec_vt], {}
        )

        return out_vt

    def _extract_nn_module_states(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> tuple[VariableTracker, VariableTracker]:
        """
        Extract nn.Module states from arguments for leaf function invocation.

        Replaces nn.Module arguments with LeafModuleState objects containing
        the module's index (for later retrieval), parameters, and buffers.
        """
        import torch.utils._pytree as pytree
        from torch._dynamo.graph_bytecode_inputs import register_user_object
        from torch._dynamo.utils import _make_inlined
        from torch._higher_order_ops.invoke_leaf_function import (
            convert_modules_to_states,
        )

        from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable

        def is_module_variable(
            var: VariableTracker,
        ) -> TypeIs[Union["NNModuleVariable", "UnspecializedNNModuleVariable"]]:
            return isinstance(var, (NNModuleVariable, UnspecializedNNModuleVariable))

        flat_args_var, tree_spec_var = unpack_iterable(
            tx,
            _make_inlined(tx, pytree.tree_flatten)(
                VariableTracker.build(tx, (args, kwargs))
            ),
        )

        module_to_index: dict[int, int] = {}
        for arg in unpack_iterable(tx, flat_args_var):
            if is_module_variable(arg):
                if arg.source is None:
                    unimplemented(
                        gb_type="leaf_function: nn.Module argument without source",
                        context=f"module type: {type(arg.value).__name__}",
                        explanation=(
                            "leaf_function received an nn.Module argument that cannot be "
                            "traced back to its origin. This typically happens when the "
                            "module is created dynamically inside the compiled region."
                        ),
                        hints=[
                            "Ensure the nn.Module is created outside the compiled function "
                            "and passed as an argument.",
                            "If the module is a class attribute, access it via self.module_name.",
                        ],
                    )
                if arg.source is None:  # make linter happy
                    raise AssertionError("Expected arg.source to be set")
                module_to_index[id(arg.value)] = register_user_object(
                    arg.value, arg.source
                )

        if not module_to_index:
            args_var = VariableTracker.build(tx, tuple(args))
            kwargs_var = VariableTracker.build(tx, kwargs)
            return args_var, kwargs_var

        module_to_index_var = VariableTracker.build(tx, module_to_index)

        result_var = _make_inlined(tx, convert_modules_to_states)(
            VariableTracker.build(tx, (args, kwargs)), module_to_index_var
        )
        return unpack_iterable(tx, result_var)  # pyrefly: ignore [bad-return]

    def _call_leaf_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        import torch.utils._pytree as pytree
        from torch._dynamo.utils import _make_inlined
        from torch._higher_order_ops.invoke_leaf_function import (
            _LeafCallable,
            invoke_leaf_function,
            make_leaf_function_wrappers,
        )

        from .builder import wrap_fx_proxy

        decorated_fn = self.value
        real_impl = decorated_fn._torchdynamo_leaf_real_fn
        fake_impl = decorated_fn._torchdynamo_leaf_fake_fn
        mutates_args = decorated_fn._torchdynamo_leaf_mutates_args

        if fake_impl is None:
            raise ValueError(
                f"leaf_function '{getattr(decorated_fn, '__name__', decorated_fn)}' "
                "requires a fake implementation. Please provide one using the @<func>.register_fake "
                "decorator. See the leaf_function docstring for details."
            )

        args_with_states, kwargs_with_states = self._extract_nn_module_states(
            tx, args, kwargs
        )

        flat_args_var, input_spec_var = unpack_iterable(
            tx,
            _make_inlined(tx, pytree.tree_flatten)(
                VariableTracker.build(tx, (args_with_states, kwargs_with_states))
            ),
        )
        flat_arg_proxies = [
            arg.as_proxy() for arg in unpack_iterable(tx, flat_args_var)
        ]
        input_spec = input_spec_var.as_python_constant()

        mutated_flat_indices = ""
        if mutates_args:
            from torch._higher_order_ops.invoke_leaf_function import (
                _resolve_mutated_flat_indices,
            )

            try:
                mutated_flat_indices = _resolve_mutated_flat_indices(
                    real_impl, mutates_args, len(flat_arg_proxies), input_spec
                )
            except ValueError as e:
                raise UserError(UserErrorType.INVALID_INPUT, str(e)) from e

        # Single-element mutable list so the wrappers can write back the output
        # TreeSpec. Read captured_out_spec[0] after the wrappers have been called.
        captured_out_spec: list[pytree.TreeSpec | None] = [None]
        wrapped_real_impl, wrapped_fake_impl = make_leaf_function_wrappers(
            real_impl, fake_impl, captured_out_spec
        )

        real_impl_callable = _LeafCallable(wrapped_real_impl)
        fake_impl_callable = _LeafCallable(wrapped_fake_impl)
        hook_fn = getattr(decorated_fn, "_torchdynamo_leaf_hook_fn", None)
        if hook_fn is not None:
            hook_fake_fn = getattr(decorated_fn, "_torchdynamo_leaf_hook_fake_fn", None)
            real_impl_callable._leaf_hook_real_fn = hook_fn  # type: ignore[attr-defined]
            real_impl_callable._leaf_hook_fake_fn = hook_fake_fn  # type: ignore[attr-defined]

        def make_callable_proxy(name: str, spec: Any) -> Any:
            proxy = tx.output.register_static_attr_and_return_proxy(name, spec)
            proxy.node.type = type(spec)
            return proxy

        real_impl_proxy = make_callable_proxy("real_fn", real_impl_callable)
        fake_impl_proxy = make_callable_proxy("fake_fn", fake_impl_callable)
        input_spec_proxy = make_callable_proxy("input_spec", input_spec)

        invoke_args = (
            real_impl_proxy,
            fake_impl_proxy,
            input_spec_proxy,
            mutated_flat_indices,
            *flat_arg_proxies,
        )
        result_proxy = tx.output.create_proxy(
            "call_function", invoke_leaf_function, invoke_args, {}
        )

        flat_output_vt = wrap_fx_proxy(tx, result_proxy)

        if captured_out_spec[0] is None:
            raise AssertionError(
                "Output spec was not captured during fake tensor propagation. "
                "This should not happen - please report a bug."
            )
        out_spec_vt = VariableTracker.build(tx, captured_out_spec[0])
        return _make_inlined(tx, _pytree.tree_unflatten)(flat_output_vt, out_spec_vt)

    def _call_ntuple(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        """inline behavior of torch.nn.modules.utils._ntuple"""
        if self.value is torch.nn.modules.utils._ntuple:
            count = args[0].as_python_constant()
        else:
            count = self.value.__closure__[0].cell_contents
        if not isinstance(count, int):
            raise AssertionError(
                f"Expected count to be an int in _ntuple, got {type(count)}"
            )
        if kwargs:
            raise AssertionError(f"_ntuple expects no kwargs, got {len(kwargs)}")

        def handle_ntuple(value: VariableTracker) -> VariableTracker:
            if vt_is_iterable(value):
                return variables.TupleVariable(unpack_iterable(tx, value))
            elif value.is_python_constant():
                # constant prop through it
                return VariableTracker.build(
                    tx,
                    torch.nn.modules.utils._ntuple(count)(value.as_python_constant()),
                )
            else:
                unimplemented(
                    gb_type="Attempted to use `torch.nn.modules.utils._ntuple` with unsupported argument type",
                    context=f"value={value}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Change use of _ntuple with argument as constant or tensor.",
                    ],
                )

        if self.value is torch.nn.modules.utils._ntuple:
            return variables.LambdaVariable(handle_ntuple)
        else:
            return handle_ntuple(args[0])

    @classmethod
    def call_nn_parameter(
        cls,
        tx: "InstructionTranslatorBase",
        data: Any | None = None,
        requires_grad: bool = True,
    ) -> VariableTracker:
        """A call to torch.nn.Parameter() gets lifted to before the graph"""
        if tx.export:
            unimplemented(
                gb_type="Attempted to use `torch.nn.Parameter()` with export",
                context="",
                explanation="Dynamo does not support this.",
                hints=[
                    "Do not use `torch.nn.Parameter()` with export.",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        if isinstance(requires_grad, variables.VariableTracker):
            try:
                requires_grad = requires_grad.as_python_constant()
            except NotImplementedError:
                unimplemented(
                    gb_type="non-constant `requires_grad` argument to `torch.nn.Parameter`",
                    context=f"requires_grad={requires_grad}",
                    explanation="Dynamo does not support this.",
                    hints=[
                        "Change `requires_grad` to be a bool.",
                        *graph_break_hints.USER_ERROR,
                    ],
                )

        if data is None or not data.is_tensor():
            unimplemented(
                gb_type="`torch.nn.Parameter()` with unsupported data type",
                context=f"data={data}",
                explanation="Called `torch.nn.Parameter()` with non-Tensor argument.",
                hints=[
                    "Ensure the argument to `torch.nn.Parameter()` is a `torch.Tensor`.",
                    *graph_break_hints.USER_ERROR,
                ],
            )

        # this results in cleaner graphs, but only works for inputs
        if data.source:
            return cls._nn_param_via_prefix_insert(tx, data, requires_grad)

        if config.graph_break_on_nn_param_ctor:
            # Need user to manually move since we cannot
            unimplemented(
                gb_type="Attempted to use `torch.nn.Parameter()` constructor with Dynamo",
                context="",
                explanation="Dynamo does not support this",
                hints=[
                    "Try to construct `torch.nn.Parameter()` outside the compiled region.",
                    "If this is not possible, turn `graph_break_on_nn_param_ctor` off",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # TODO[@lucaskabela]: Remove the behavior below since it is deprecated
        if isinstance(
            data,
            TensorWithTFOverrideVariable,
        ) or is_traceable_wrapper_subclass_type(data.class_type):
            unimplemented(
                gb_type="Attempted to use torch.nn.Parameter constructor with tensor subclass",
                context=str(data),
                explanation="Dynamo does not support this.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        if not can_convert_to_tracable_parameter():
            unimplemented(
                gb_type="`torch.nn.Parameter`: cannot convert to traceable tracable",
                context="",
                explanation="convert_tracable_parameter is set to False.",
                hints=[
                    "Check usage of context manager: do_not_convert_to_tracable_parameter",
                    *graph_break_hints.DIFFICULT,
                ],
            )

        try:
            shape = tuple(data.tp_getattro_impl(tx, "shape").as_python_constant())
            dtype = data.tp_getattro_impl(tx, "dtype").as_python_constant()
            device = data.tp_getattro_impl(tx, "device").as_python_constant()
        except NotImplementedError as e:
            unimplemented(
                gb_type="`torch.nn.Parameter` with non-constant Tensor attributes",
                context=f"data={data}",
                explanation="Dynamo does not support this.",
                hints=[
                    "Ensure the Tensor argument's shape, dtype, and device are correct.",
                    *graph_break_hints.USER_ERROR,
                ],
                from_exc=e,
            )

        placeholder = tx.output.synthetic_graph_input(
            new_parameter_placeholder,
            (shape, dtype, device, requires_grad),
        )
        if data.requires_grad:
            data = data.call_method(tx, "detach", [], {})

        from .builder import wrap_fx_proxy

        result = wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_function",
                tracable_create_parameter,
                (data.as_proxy(), placeholder.as_proxy()),
                {},
            ),
            # In reconstruct() we should use the original parameter. The one
            # returned by the graph will be an alias.
            source=placeholder.source,
        )
        if not result.is_tensor():
            raise AssertionError(
                "Expected result of traceable_create_parameter to be a tensor"
            )
        result.class_type = torch.nn.Parameter  # type: ignore[union-attr]

        # TODO(jansel/bdhirsh) - There is some issue with
        # tracable_create_parameter. It does not seem to use the right
        # grad_enabled. Since this is parameter, we can just override the
        # has_grad_fn field to False to workaround the issue.
        result.has_grad_fn = False  # type: ignore[union-attr]

        # Register this parameter as a leaf tensor for backward() auto-detection.
        # When backward() is called without inputs, we need to find all leaf tensors,
        # including those created in-graph like nn.Parameter.
        tx.output.leaf_var_creation_order.append(result)

        # TODO(jansel): if the new param falls out of scope, currently it won't get freed until
        # the end of the graph.  We should fix this.
        return result

    @staticmethod
    def _nn_param_via_prefix_insert(
        tx: "InstructionTranslatorBase", data: Any, requires_grad: bool
    ) -> VariableTracker:
        # Alternate version if we have a .source
        varname = tx.output.new_var()

        # construct the nn.Parameter before the graph save it to varname
        if tx.output.root_tx is None:
            raise AssertionError("Expected tx.output.root_tx to be set")
        cg = PyCodegen(tx.output.root_tx)
        cg.add_push_null(lambda: cg.load_import_from("torch.nn", "Parameter"))
        cg(data.source)
        cg(VariableTracker.build(tx, requires_grad))
        cg.call_function(2, False)
        cg.store(varname)
        tx.output.pregraph_bytecode.extend(cg.get_instructions())

        data_node = data.as_proxy().node
        if data_node.op not in ("placeholder", "get_attr"):
            unimplemented(
                gb_type="Unexpected type of data placeholder op for parameter construction",
                context=f"data_node.op={data_node.op}",
                explanation="Data node op should be placeholder or get_attr.",
                hints=[
                    *graph_break_hints.DIFFICULT,
                ],
            )

        # add the newly constructed nn.Parameter as a graph input
        source = SyntheticLocalSource(varname)
        example_value = torch.nn.Parameter(
            tx.output.example_value_from_input_node(data.as_proxy().node),
            requires_grad=requires_grad,
        )
        result = VariableTracker.build(tx, example_value, source)
        # Realize the VT because we will delete the guards on it in the next line.
        result = result.realize()
        # No need to guard on this since we already guarded on `data`.
        # These guards would fail since varname doesn't exist until after the function starts
        TracingContext.get().guards_context.dynamo_guards.remove_guards_with_source(
            source
        )
        return result

    def call_tensor_method(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return args[0].call_method(tx, self.get_function().__name__, args[1:], kwargs)

    def is_tensor_method(self) -> bool:
        from ..trace_rules import get_tensor_method

        return (
            inspect.ismethoddescriptor(self.get_function())
            and hasattr(self.get_function(), "__objclass__")
            # pyrefly: ignore[missing-attribute]
            and self.get_function().__objclass__ == torch._C.TensorBase
        ) or self.get_function() in get_tensor_method()

    def torch_function_override_enabled(
        self,
        tx: "InstructionTranslatorBase",
        args: Iterable[Any],
        kwargs: dict[str, Any],
    ) -> bool:
        return (
            self.get_function() in get_overridable_functions()
            or isinstance(
                self.get_function(),
                (torch._ops.OpOverload, torch._ops.OpOverloadPacket),
            )
        ) and can_dispatch_torch_function(tx, args, kwargs)


class DispatchKeySetVariable(BaseTorchVariable):
    """represents torch.DispatchKeySet"""

    @staticmethod
    def create(value: DispatchKeySet, **kwargs: Any) -> "DispatchKeySetVariable":
        return DispatchKeySetVariable(value, **kwargs)

    @classmethod
    def create_with_source(
        cls, value: DispatchKeySet, source: Source
    ) -> "DispatchKeySetVariable":
        install_guard(source.make_guard(GuardBuilder.DISPATCH_KEY_SET_MATCH))
        return cls(value, source=source)

    def tp_richcompare_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker, op: str
    ) -> VariableTracker:
        from .object_protocol import python_constant_richcompare_impl

        return python_constant_richcompare_impl(self, tx, other, op)

    def is_constant_fold_method(self, name: str) -> bool:
        return name == "has"

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        if self.is_constant_fold_method(name) and check_unspec_or_constant_args(
            args, kwargs
        ):
            method = getattr(self.value, name)
            return VariableTracker.build(
                tx,
                method(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
            )
        elif name == "highestPriorityTypeId":
            return VariableTracker.build(tx, self.value.highestPriorityTypeId())
        return super().call_method(tx, name, args, kwargs)


class FuncTorchInterpreterVariable(BaseTorchVariable):
    """represents torch._functorch.pyfunctorch.FuncTorchInterpreter"""

    @classmethod
    def create_with_source(
        cls, value: Any, source: Source
    ) -> "FuncTorchInterpreterVariable":
        install_guard(source.make_guard(GuardBuilder.ID_MATCH))
        return cls(value, source=source)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        if name == "key":
            return VariableTracker.build(tx, self.value.key())
        elif name == "process":
            return tx.inline_user_function_return(
                VariableTracker.build(tx, self.value.process.__func__),
                [self] + args,
                kwargs,
            )
        elif name in ["level", "batch_size", "randomness"]:
            return VariableTracker.build(tx, getattr(self.value, name)())
        elif name == "lower":
            if args:
                raise AssertionError(f"lower() expects no args, got {len(args)}")
            if kwargs:
                raise AssertionError(f"lower() expects no kwargs, got {len(kwargs)}")
            return variables.TemporarilyPopInterpreterStackCtxManagerVariable.create(
                tx, None
            )
        return super().call_method(tx, name, args, kwargs)
<|endoftext|>"""
Set-related variable tracking classes for PyTorch Dynamo.

This module implements variable tracking for different types of set-like objects:
- Regular Python sets (set)
- Frozen sets (frozenset)
- Ordered sets (torch.utils._ordered_set.OrderedSet)
- Dictionary key sets (dict_keys views used as sets)

These classes are responsible for tracking set operations during graph compilation,
maintaining proper guards for set mutations and element existence checks.

The implementation uses a special HashableTracker wrapper to handle set elements
while preserving proper aliasing semantics. Sets are modeled internally as
dictionaries with None values.
"""

import functools
import operator
from collections.abc import Iterable, Iterator
from typing import Any, TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import raise_observed_exception, raise_type_error
from ..guards import GuardBuilder, install_guard
from ..source import is_constant_source, is_from_local_source
from ..utils import (
    _item_debug_repr,
    cmp_name_to_op_mapping,
    istype,
    lazily_unpack,
    raise_args_mismatch,
    set_methods,
    tracked_repr,
    unpack_iterable,
)
from .base import Member, Method, ValueMutationNew, VariableTracker
from .constant import ConstantVariable
from .hashable import HashableTracker, is_hashable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


# [Adding a new supported class within the keys of SetVariable]
# see steps outlined for ConstDictVariable


def pyanyset_check(obj: VariableTracker) -> bool:
    return issubclass(obj.python_type(), (set, frozenset))


def pyset_check(obj: VariableTracker) -> bool:
    # ref: https://github.com/python/cpython/blob/v3.13.0/Include/setobject.h#L36-L38
    return issubclass(obj.python_type(), set)


def set_copy(obj: VariableTracker) -> VariableTracker:
    """Mirrors CPython's internal `set_copy` (Objects/setobject.c).

    Always allocates a fresh set/frozenset with a shallow-copied items dict.
    Distinct from the user-visible `.copy()` method, which preserves identity
    for exact frozenset (`frozenset_copy`).  Use this for binary-op scratch
    storage so mutations don't bleed into the input.
    """
    base = obj._base_vt if isinstance(obj, variables.UserDefinedSetVariable) else obj
    if base is None:
        raise AssertionError("_base_vt must not be None")
    return base.clone(
        items=base.items.copy(),  # type: ignore[missing-attribute]
        mutation_type=ValueMutationNew(),
        source=None,
    )


class SetVariable(VariableTracker):
    """Represents a Python set during symbolic execution."""

    # PySet_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2436
    _cpython_type = set

    CONTAINS_GUARD = GuardBuilder.SET_CONTAINS
    NOT_CONTAINS_GUARD = GuardBuilder.SET_NOT_CONTAINS

    def __init__(
        self,
        items: Iterable[VariableTracker | HashableTracker],
        **kwargs: Any,
    ) -> None:
        # .clone() passes these arguments in kwargs but they're recreated below
        if "original_items" in kwargs:
            kwargs.pop("original_items")
        if "should_reconstruct_all" in kwargs:
            kwargs.pop("should_reconstruct_all")

        super().__init__(**kwargs)

        # Items can be either VariableTrackers or HashableTrackers (from set ops).
        # For VariableTrackers, realize them to ensure aliasing guards are installed
        # when the same object appears multiple times.
        hashable_items = []
        for item in items:
            if isinstance(item, HashableTracker):
                # Already a HashableTracker from a set operation
                hashable_items.append(item)
            else:
                # VariableTracker - realize to install guards, then wrap
                # pyrefly: ignore [bad-argument-type]
                hashable_items.append(HashableTracker(item.realize()))
        # Internal representation as dict allows for simple integration with
        # OrderedSet, notably polyfills. Using set moves complexity to OrderedSet
        self.items = dict.fromkeys(hashable_items, SetVariable._default_value())
        self.should_reconstruct_all = (
            not is_from_local_source(self.source) if self.source else True
        )
        self.original_items = dict.fromkeys(
            hashable_items, SetVariable._default_value()
        )

    def debug_repr(self) -> str:
        if not self.items:
            return "set()"
        else:
            items: list[str] = []
            for v in self.items:
                vt = v.vt if isinstance(v, HashableTracker) else v
                val_str = _item_debug_repr(vt)
                items.append(val_str)
            return "{" + ", ".join(items) + "}"

    @property
    def set_items(self) -> set["HashableTracker"]:
        return set(self.items.keys())

    @staticmethod
    def _default_value() -> VariableTracker:
        # Variable to fill in the keys of the dictionary
        return ConstantVariable.create(None)

    def as_proxy(self) -> Any:
        return {k.vt.as_proxy() for k in self.set_items}

    def python_type(self) -> type:
        return set

    def is_python_constant(self) -> bool:
        # Avoid the base implementation, which probes as_python_constant() and
        # thus rebuilds a real set, re-hashing the elements (wrong for elements
        # with a side-effecting __hash__).  Check element constness directly.
        return all(k.vt.is_python_constant() for k in self.set_items)

    def as_python_constant(self) -> Any:
        return {k.vt.as_python_constant() for k in self.set_items}

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L763-L822
        if not self.items:
            return VariableTracker.build(tx, f"{self.python_type_name()}()")
        items = ", ".join(tracked_repr(tx, item.vt) for item in self.set_items)
        return VariableTracker.build(tx, "{" + items + "}")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.foreach([x.vt for x in self.set_items])
        codegen.append_output(create_instruction("BUILD_SET", arg=len(self.set_items)))

    def __contains__(self, vt: VariableTracker) -> bool:
        if not isinstance(vt, VariableTracker):
            raise AssertionError(f"Expected VariableTracker, got {type(vt)}")
        # Use is_hashable as a side-effect-free pre-check.  We can't catch
        # ObservedTypeError from HashableTracker because it modifies
        # tx.exn_vt_stack as a side effect.
        if not is_hashable(vt):
            return False
        key = HashableTracker(vt)
        return key in self.items

    def has_new_items(self) -> bool:
        return self.should_reconstruct_all or any(
            # pyrefly: ignore [bad-argument-type]
            self.is_new_item(self.original_items.get(key.vt), value)
            for key, value in self.items.items()
        )

    def is_new_item(
        self, value: VariableTracker | None, other: VariableTracker
    ) -> bool:
        if value and value.is_realized() and other.is_realized():
            return id(value.realize()) != id(other.realize())
        return id(value) != id(other)

    def unpack_var_sequence(
        self, tx: "InstructionTranslatorBase"
    ) -> list[VariableTracker]:
        return [x.vt for x in self.items]

    def clone(self, **kwargs: Any) -> VariableTracker:
        from torch._dynamo.variables.base import AttributeMutationNew, ValueMutationNew

        if isinstance(
            kwargs.get("mutation_type"), (ValueMutationNew, AttributeMutationNew)
        ):
            kwargs["source"] = None
        return super().clone(**kwargs)

    def is_hashable(self) -> bool:
        return False

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        from ..exc import raise_type_error

        raise_type_error(tx, f"unhashable type: '{self.python_type_name()}'")

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        return VariableTracker.build(tx, hasattr(self.python_type(), name))

    def install_set_contains_guard(
        self, tx: "InstructionTranslatorBase", args: list[VariableTracker]
    ) -> None:
        if not self.source:
            return

        if tx.output.side_effects.is_modified(self):
            return

        contains = args[0] in self
        if args[0].source is None and args[0].is_python_constant():
            guard_fn = (
                type(self).CONTAINS_GUARD if contains else type(self).NOT_CONTAINS_GUARD
            )
            install_guard(
                self.make_guard(
                    functools.partial(
                        guard_fn,
                        key=args[0].as_python_constant(),
                    )
                )
            )

    def _fast_set_method(
        self,
        tx: "InstructionTranslatorBase",
        fn: Any,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        try:
            res = fn(
                *[x.as_python_constant() for x in [self, *args]],
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
        except Exception as exc:
            raise_observed_exception(type(exc), tx, args=list(exc.args))
        return VariableTracker.build(tx, res)

    def _iter_operand_keys(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> "Iterator[HashableTracker]":
        # Lazily yield HashableTracker keys for a set-operation operand, one
        # element at a time so callers that short-circuit (isdisjoint) observe
        # the same generator side effects as CPython. A set or dict operand's
        # keys are reused directly (no re-hashing), mirroring CPython's
        # set_update_internal fast path for set/frozenset/dict operands.
        # HashableTracker raises ObservedTypeError for an unhashable element,
        # matching CPython's hash-at-insert behavior.
        from .dicts import ConstDictVariable

        if isinstance(other, (SetVariable, ConstDictVariable)):
            yield from other.items.keys()
            return

        yield from (HashableTracker(item) for item in lazily_unpack(tx, other))

    def _operand_keys(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> list[HashableTracker]:
        # Eager variant: unpack_iterable takes the fast unpack_var_sequence
        # path for builtin iterables instead of the per-element iterator
        # protocol.
        from .dicts import ConstDictVariable

        if isinstance(other, (SetVariable, ConstDictVariable)):
            return list(other.items.keys())
        return [HashableTracker(x) for x in unpack_iterable(tx, other)]

    def _new_set(self, items: "Iterable[HashableTracker]") -> "SetVariable":
        # Build a fresh set of the same concrete type (set / frozenset /
        # OrderedSet). list() preserves insertion order, which matters for
        # OrderedSet.
        return type(self)(list(items), mutation_type=ValueMutationNew())

    def sq_contains_impl(
        self, tx: "InstructionTranslatorBase", item: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2131-L2149
        if not is_hashable(item):
            # Mirror CPython's set_contains: if hashing fails with TypeError due to
            # an unhashable set, coerce the key to frozenset and retry.
            # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2151-L2159
            if not pyset_check(item):
                raise_type_error(tx, f"unhashable type: '{item.python_type_name()}'")
            # CPython NOTE:
            # Note that 'key' could be a set() or frozenset() object.  Unlike most
            # container types, set allows membership testing with a set key, even
            # though it is not hashable.
            item = FrozensetVariable(item.items)  # type: ignore[missing-attribute]
        self.install_set_contains_guard(tx, [item])
        contains = item in self
        return VariableTracker.build(tx, contains)

    def _try_fast_set_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker | None":
        from ..utils import check_constant_args
        from .dicts import ConstDictVariable

        # Constant, exact-builtin fast path: materialize and call the real
        # method. Set/dict operands take the slow VT path below so their keys
        # are not re-hashed (CPython's do-not-rehash-dict-keys fast path);
        # OrderedSet is excluded because as_python_constant() loses insertion
        # order (it routes through the unordered set_items).
        if (
            not any(isinstance(a, (SetVariable, ConstDictVariable)) for a in args)
            and check_constant_args(args, kwargs)
            and self.python_type() in (set, frozenset)
        ):
            py_type = self.python_type()
            return self._fast_set_method(tx, getattr(py_type, name), args, kwargs)
        return None

    def add(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Convert add to __setitem__ with None value
        tx.output.side_effects.mutation(self)
        self.items[HashableTracker(args[0])] = SetVariable._default_value()
        return ConstantVariable.create(None)

    def pop(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Choose an item at random and pop it
        try:
            result: VariableTracker = self.set_items.pop().vt  # type: ignore[assignment]
        except KeyError as e:
            raise_observed_exception(KeyError, tx, args=list(e.args))
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        self.items.pop(HashableTracker(result))
        return result

    def isdisjoint(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fast = self._try_fast_set_method(tx, "isdisjoint", args, kwargs)
        if fast is not None:
            return fast
        for key in self._iter_operand_keys(tx, args[0]):
            if key in self.items:
                return ConstantVariable.create(False)
        return ConstantVariable.create(True)

    def intersection(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fast = self._try_fast_set_method(tx, "intersection", args, kwargs)
        if fast is not None:
            return fast
        out_items = dict(self.items)
        for other in args:
            other_keys = set(self._operand_keys(tx, other))
            out_items = {k: v for k, v in out_items.items() if k in other_keys}
        return self._new_set(out_items)

    def intersection_update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        kept = dict(self.items)
        for other in args:
            other_keys = set(self._operand_keys(tx, other))
            kept = {k: v for k, v in kept.items() if k in other_keys}
        tx.output.side_effects.mutation(self)
        self.should_reconstruct_all = True
        self.items.clear()
        self.items.update(kept)
        return ConstantVariable.create(None)

    def union(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fast = self._try_fast_set_method(tx, "union", args, kwargs)
        if fast is not None:
            return fast
        out_items = dict(self.items)
        for other in args:
            for key in self._operand_keys(tx, other):
                out_items.setdefault(key, SetVariable._default_value())
        return self._new_set(out_items)

    def difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fast = self._try_fast_set_method(tx, "difference", args, kwargs)
        if fast is not None:
            return fast
        out_items = dict(self.items)
        for other in args:
            for key in self._operand_keys(tx, other):
                out_items.pop(key, None)
        return self._new_set(out_items)

    def difference_update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        tx.output.side_effects.mutation(self)
        self.should_reconstruct_all = True
        for other in args:
            for key in self._operand_keys(tx, other):
                self.items.pop(key, None)
        return ConstantVariable.create(None)

    def symmetric_difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        fast = self._try_fast_set_method(tx, "symmetric_difference", args, kwargs)
        if fast is not None:
            return fast
        other = dict.fromkeys(
            self._operand_keys(tx, args[0]), SetVariable._default_value()
        )
        out_items = {k: v for k, v in self.items.items() if k not in other}
        out_items.update({k: v for k, v in other.items() if k not in self.items})
        return self._new_set(out_items)

    def symmetric_difference_update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        other = dict.fromkeys(
            self._operand_keys(tx, args[0]), SetVariable._default_value()
        )
        new_items = {k: v for k, v in self.items.items() if k not in other}
        new_items.update({k: v for k, v in other.items() if k not in self.items})
        tx.output.side_effects.mutation(self)
        self.should_reconstruct_all = True
        self.items.clear()
        self.items.update(new_items)
        return ConstantVariable.create(None)

    def update(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        if not self.is_mutable():
            return None
        tx.output.side_effects.mutation(self)
        for other in args:
            for key in self._operand_keys(tx, other):
                self.items.setdefault(key, SetVariable._default_value())
        return ConstantVariable.create(None)

    def remove(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if not self.sq_contains_impl(tx, args[0]).as_python_constant():
            raise_observed_exception(KeyError, tx, args=[args[0]])
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        # sq_contains validated/normalized args[0]; a set key was coerced to
        # a frozenset, so pop that same normalized key.
        key = args[0] if is_hashable(args[0]) else FrozensetVariable(args[0].items)  # type: ignore[missing-attribute]
        self.items.pop(HashableTracker(key))
        return ConstantVariable.create(None)

    def discard(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if self.sq_contains_impl(tx, args[0]).as_python_constant():
            self.should_reconstruct_all = True
            tx.output.side_effects.mutation(self)
            # sq_contains validated/normalized args[0]; a set key was coerced
            # to a frozenset, so pop that same normalized key.
            key = (
                args[0] if is_hashable(args[0]) else FrozensetVariable(args[0].items)  # type: ignore[missing-attribute]
            )
            self.items.pop(HashableTracker(key))
        return ConstantVariable.create(None)

    def _ordering_test(self, tx, args, op):
        from .builder import SourcelessBuilder

        other = args[0].realize()
        if not istype(other, SetVariable):
            other = SourcelessBuilder.create(tx, set).call_function(tx, [other], {})
        return SourcelessBuilder.create(tx, op).call_function(tx, [self, other], {})

    def issubset(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self._ordering_test(tx, args, operator.le)

    def issuperset(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self._ordering_test(tx, args, operator.ge)

    def copy(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return set_copy(self)

    def clear(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        self.items.clear()
        return ConstantVariable.create(None)

    tp_methods = {
        "add": Method(add),
        "pop": Method(pop),
        "isdisjoint": Method(isdisjoint),
        "intersection": Method(intersection),
        "intersection_update": Method(intersection_update),
        "union": Method(union),
        "difference": Method(difference),
        "difference_update": Method(difference_update),
        "symmetric_difference": Method(symmetric_difference),
        "symmetric_difference_update": Method(symmetric_difference_update),
        "update": Method(update),
        "remove": Method(remove),
        "discard": Method(discard),
        "issubset": Method(issubset),
        "issuperset": Method(issuperset),
        "copy": Method(copy),
        "clear": Method(clear),
    }

    def getitem_const(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        raise RuntimeError("Illegal to getitem on a set")

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from .builder import SourcelessBuilder

        temp_set_vt = SourcelessBuilder.create(tx, set).call_set(tx, *args, **kwargs)
        tx.output.side_effects.mutation(self)
        self.items.clear()
        self.items.update(temp_set_vt.items)  # type: ignore[attr-defined]
        return ConstantVariable.create(None)

    def tp_iter_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        from .iter import SetIterator

        if self.source and not is_constant_source(self.source):
            tx.output.guard_on_key_order.add(self.source)
        return SetIterator(self.items)

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1318-L1338
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        result = set_copy(self_)
        if self_ is other_:
            return result
        result.items.update(other_.items)  # type: ignore[missing-attribute]
        return result

    def nb_inplace_or_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1340-L1350
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        tx.output.side_effects.mutation(self)
        self.items.update(other.items)  # type: ignore[missing-attribute]
        return self

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L1801-L1812
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        result = set_copy(self_)
        for k in list(other_.items.keys()):  # type: ignore[missing-attribute]
            result.items.pop(k, None)  # type: ignore[missing-attribute]
        return result

    def nb_inplace_subtract_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L1814-L1828
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        tx.output.side_effects.mutation(self)
        for k in list(other.items.keys()):  # type: ignore[missing-attribute]
            self.items.pop(k, None)
        return self

    def nb_and_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1506-L1518 (set_and)
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        return self_.call_method(tx, "intersection", [other_], {})

    def nb_inplace_and_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1520-L1536 (set_iand)
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        self.call_method(tx, "intersection_update", [other], {})
        return self

    def nb_xor_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1984-L1990 (set_xor)
        self_, other_ = (other, self) if reverse else (self, other)

        if not pyanyset_check(self_) or not pyanyset_check(other_):
            return ConstantVariable.create(NotImplemented)

        return self_.call_method(tx, "symmetric_difference", [other_], {})

    def nb_inplace_xor_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L1992-L2004 (set_ixor)
        if not pyanyset_check(other):
            return ConstantVariable.create(NotImplemented)

        self.call_method(tx, "symmetric_difference_update", [other], {})
        return self

    def sq_length_impl(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        return VariableTracker.build(tx, len(self.set_items))

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        """set_richcompare: subset/superset comparisons for all 6 ops.

        https://github.com/python/cpython/blob/e76aa128fe/Objects/setobject.c#L2097
        CPython uses PyAnySet_Check: only accepts set/frozenset (not dict views).
        """
        if not isinstance(other, SetVariable):
            try:
                other_type = other.python_type()
            except NotImplementedError:
                return ConstantVariable.create(NotImplemented)
            if not issubclass(other_type, (set, frozenset)):
                return ConstantVariable.create(NotImplemented)

        # Accessing set_items directly is correct: CPython's set_richcompare
        # operates on the internal C struct (PySet_GET_SIZE, set_next,
        # set_contains_entry) -- it never calls __len__ or __contains__.
        # https://github.com/python/cpython/blob/e76aa128fe/Objects/setobject.c#L2093-L2130
        self_items = self.set_items
        other_items = other.set_items  # type: ignore[attr-defined]
        if op == "__eq__":
            # len check + issubset: same length and subset implies equality.
            if len(self_items) != len(other_items):
                return ConstantVariable.create(False)
            return VariableTracker.build(tx, self_items <= other_items)
        elif op == "__ne__":
            if len(self_items) != len(other_items):
                return ConstantVariable.create(True)
            return VariableTracker.build(tx, not (self_items <= other_items))
        else:
            return VariableTracker.build(
                tx,
                cmp_name_to_op_mapping[op](self_items, other_items),
            )


class OrderedSetClassVariable(VariableTracker):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def as_python_constant(self) -> type[OrderedSet[Any]]:
        return OrderedSet

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__new__":
            if len(args) != 2 or kwargs:
                raise_args_mismatch(
                    tx,
                    name,
                    "OrderedSet.__new__ only accepts one arg"
                    f"{len(args)} args and {len(kwargs)} kwargs",
                )

            return variables.OrderedSetVariable([], mutation_type=ValueMutationNew())

        resolved_fn = getattr(set, name)
        if resolved_fn in set_methods and isinstance(args[0], variables.SetVariable):
            return args[0].call_method(tx, name, args[1:], kwargs)

        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "OrderedSetVariable":
        if len(args) > 1 or kwargs:
            raise_args_mismatch(
                tx,
                "OrderedSet",
                "OrderedSet only accepts one arg"
                f"{len(args)} args and {len(kwargs)} kwargs",
            )

        if len(args) == 0:
            # pyrefly: ignore [implicit-any]
            items = []
        else:
            items = unpack_iterable(tx, args[0])
        return variables.OrderedSetVariable(items, mutation_type=ValueMutationNew())


class OrderedSetVariable(SetVariable):
    _cpython_type = OrderedSet

    def method_flags_type(self) -> type:
        # OrderedSet is pure-Python (no C ml_flags); its named methods mirror
        # set's arities, so derive MethodFlags from set to enforce them.
        return set

    def _get_internal_dict(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # OrderedSet is backed by a dict (self._dict). Expose it so inlined
        # OrderedSet methods (e.g. __contains__ -> `elem in self._dict`) trace
        # natively instead of graph-breaking on an unmodeled attribute.
        from .dicts import ConstDictVariable

        return ConstDictVariable(self.items, mutation_type=ValueMutationNew())  # type: ignore[bad-argument-type]

    tp_members = {"_dict": Member(_get_internal_dict, None)}

    def debug_repr(self) -> str:
        if not self.items:
            return "OrderedSet([])"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "OrderedSet([" + ", ".join(items) + "])"

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        items = ", ".join(tracked_repr(tx, item.vt) for item in self.set_items)
        return VariableTracker.build(tx, f"{self.python_type_name()}([{items}])")

    def as_python_constant(self) -> OrderedSet[Any]:
        return OrderedSet([k.vt.as_python_constant() for k in self.set_items])

    def python_type(self) -> type[OrderedSet[Any]]:
        return OrderedSet

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch.utils._ordered_set", "OrderedSet")
        )
        codegen.foreach([x.vt for x in self.set_items])
        codegen.append_output(create_instruction("BUILD_LIST", arg=len(self.set_items)))
        codegen.extend_output(create_call_function(1, False))

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # OrderedSet does not inherit from Python set, so SetVariable.nb_or_impl
        # won't work due to the PyAnySet_Check
        return super().call_method(tx, "union", [other], {})

    def nb_and_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # OrderedSet does not inherit from Python set, so SetVariable.nb_and_impl
        # won't work due to the PyAnySet_Check
        return super().call_method(tx, "intersection", [other], {})

    def nb_xor_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # OrderedSet does not inherit from Python set, so SetVariable.nb_xor_impl
        # won't work due to the PyAnySet_Check
        return super().call_method(tx, "symmetric_difference", [other], {})

    def nb_subtract_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        self_, other_ = (other, self) if reverse else (self, other)
        return self_.call_method(tx, "difference", [other_], {})

    def nb_inplace_subtract_impl(
        self, tx: "InstructionTranslatorBase", other: VariableTracker
    ) -> VariableTracker:
        tx.output.side_effects.mutation(self)
        self.call_method(tx, "difference_update", [other], {})
        return self


class FrozensetVariable(SetVariable):
    # PyFrozenSet_Type: https://github.com/python/cpython/blob/v3.13.0/Objects/setobject.c#L2526
    _cpython_type = frozenset

    nb_inplace_subtract_impl = None  # type: ignore[bad-override]

    def debug_repr(self) -> str:
        if not self.items:
            return "frozenset()"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "frozenset({" + ", ".join(items) + "})"

    @property
    def set_items(self) -> set["HashableTracker"]:
        return set(self.items.keys())

    def python_type(self) -> type:
        return frozenset

    def as_python_constant(self) -> Any:
        return frozenset({k.vt.as_python_constant() for k in self.set_items})

    def tp_repr_impl(self, tx: "InstructionTranslatorBase") -> "VariableTracker":
        # https://github.com/python/cpython/blob/3.13/Objects/setobject.c#L763-L822
        if not self.items:
            return VariableTracker.build(tx, f"{self.python_type_name()}()")
        items = ", ".join(tracked_repr(tx, item.vt) for item in self.set_items)
        return VariableTracker.build(tx, f"{self.python_type_name()}({{{items}}})")

    def reconstruct(self, codegen: "PyCodegen") -> None:
        codegen.add_push_null(
            lambda: codegen.extend_output(
                [
                    codegen.create_load_global("frozenset"),
                ]
            )
        )
        codegen.foreach([x.vt for x in self.set_items])
        codegen.extend_output(
            [
                create_instruction("BUILD_LIST", arg=len(self.set_items)),
                *create_call_function(1, False),
            ]
        )

    def copy(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if type(self) is FrozensetVariable:
            return self
        return SetVariable.copy(self, tx, args, kwargs)

    def difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        r = SetVariable.difference(self, tx, args, kwargs)
        return FrozensetVariable(r.items)  # type: ignore[attr-defined]

    def intersection(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        r = SetVariable.intersection(self, tx, args, kwargs)
        return FrozensetVariable(r.items)  # type: ignore[attr-defined]

    def symmetric_difference(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        r = SetVariable.symmetric_difference(self, tx, args, kwargs)
        return FrozensetVariable(r.items)  # type: ignore[attr-defined]

    tp_methods = {
        "copy": Method(copy),
        "difference": Method(difference),
        "intersection": Method(intersection),
        "symmetric_difference": Method(symmetric_difference),
    }

    def tp_init_impl(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # frozenset is immutable. Calling __init__ again shouldn't have any effect.
        return ConstantVariable.create(None)

    def is_hashable(self) -> bool:
        return True

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # Overrides SetVariable.hash_impl (which raises TypeError for mutable sets).
        # CPython frozenset_hash: https://github.com/python/cpython/blob/e76aa128fe/Objects/setobject.c#L769
        from .hashable import RawHash
        from .object_protocol import generic_hash_impl

        if self.is_python_constant():
            return hash(self.as_python_constant()), False
        is_fake = False
        raw_hashes = []
        for item in self.set_items:
            h, fake = generic_hash_impl(tx, item.vt)
            is_fake = is_fake or fake
            raw_hashes.append(RawHash(h))
        return hash(frozenset(raw_hashes)), is_fake


class DictKeySetVariable(SetVariable):
    def debug_repr(self) -> str:
        if not self.items:
            return "dict_keys([])"
        else:
            items: list[str] = []
            for k in self.items:
                key_str = _item_debug_repr(k.vt)
                items.append(key_str)
            return "dict_keys([" + ", ".join(items) + "])"

    def install_set_contains_guard(
        self, tx: "InstructionTranslatorBase", args: list[VariableTracker]
    ) -> None:
        # Already EQUALS_MATCH guarded
        pass

    @property
    def set_items(self) -> Any:
        return self.items

    def python_type(self) -> type:
        from ..utils import dict_keys

        return dict_keys

    def as_python_constant(self) -> Any:
        return dict.fromkeys(
            {k.vt.as_python_constant() for k in self.set_items}, None
        ).keys()
<|endoftext|>from inspect import getattr_static
from typing import Any, TYPE_CHECKING, TypeGuard

from torch._guards import Source
from torch.backends.cuda import SDPAParams
from torch.fx.proxy import Proxy

from ..bytecode_transformation import create_call_function
from ..exc import unimplemented
from ..source import AttrSource
from .base import VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

PARAM_NAMES = [
    "query",
    "key",
    "value",
    "attn_mask",
    "dropout",
    "is_causal",
    "enable_gqa",
]


class SDPAParamsVariable(VariableTracker):
    """Represents the c++ params struct for scaled dot product attention.
    This is a read-only container."""

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase", value: Any, source: Source
    ) -> VariableTracker:
        from .torch import TorchInGraphFunctionVariable

        params = [
            VariableTracker.build(tx, getattr(value, p), AttrSource(source, p))
            for p in PARAM_NAMES
        ]
        return TorchInGraphFunctionVariable(SDPAParams).call_function(tx, params, {})

    def __init__(
        self, proxy: Proxy, param_vars: list[VariableTracker], **kwargs: Any
    ) -> None:
        self.proxy = proxy
        self.param_vars = param_vars
        super().__init__(**kwargs)

    def python_type(self) -> type:
        return SDPAParams

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self.source is not None:
            raise AssertionError(
                "SDPAParamsVariable should not have a source during reconstruct"
            )
        if self.param_vars is None:
            raise AssertionError("SDPAParamsVariable.param_vars must not be None")
        codegen.add_push_null(
            lambda: codegen.load_import_from("torch._C", "_SDPAParams")
        )
        codegen.foreach(self.param_vars)
        codegen.extend_output(create_call_function(len(self.param_vars), False))

    def as_proxy(self) -> Proxy:
        return self.proxy

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        import torch._C

        from .builder import wrap_fx_proxy
        from .misc import GetAttrVariable

        try:
            getattr_static(torch._C._SDPAParams, name)
        except AttributeError:
            import torch._dynamo.graph_break_hints as graph_break_hints

            unimplemented(
                gb_type="unsupported torch._C._SDPAParams attribute",
                context=f"name: {name}",
                explanation=f"Unable to fetch attribute {name} from torch._C._SDPAParams.",
                hints=[
                    *graph_break_hints.USER_ERROR,
                ],
            )

        proxy = GetAttrVariable.create_getattr_proxy(self.as_proxy(), name)
        if self.source is not None:
            return wrap_fx_proxy(
                tx=tx, proxy=proxy, source=AttrSource(self.source, name)
            )
        else:
            return wrap_fx_proxy(tx=tx, proxy=proxy)

    @staticmethod
    def is_sdpa_params(value: Any) -> TypeGuard["SDPAParams"]:
        return value is SDPAParams
<|endoftext|>"""
Built-in function and type variable tracking for TorchDynamo's symbolic execution.

This module contains variable tracker classes for Python built-in functions, types,
and operations during graph compilation. It handles symbolic execution of:

- Built-in functions (len, getattr, isinstance, etc.)
- Type constructors (int, float, str, list, dict, etc.)
- Built-in operators and methods
- Special Python constructs (super, hasattr, etc.)

Key classes:
- BuiltinVariable: Tracks built-in functions and handles their execution
- TypeVariable: Manages type constructor calls and type checking
- SuperVariable: Handles super() calls in class hierarchies

These variable trackers ensure that built-in Python operations are correctly
handled during symbolic execution, either by executing them directly when safe
or by creating appropriate graph nodes when needed.
"""

import abc
import ast
import builtins
import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import sys
import types
import typing
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NoReturn, TYPE_CHECKING

import torch
from torch.overrides import BaseTorchFunctionMode
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .. import graph_break_hints, polyfills, variables
from ..exc import (
    handle_observed_exception,
    ObservedAttributeError,
    ObservedTypeError,
    ObservedUserStopIteration,
    raise_observed_exception,
    raise_type_error,
    raise_value_error,
    unimplemented,
    Unsupported,
    UserError,
    UserErrorType,
)
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    GetItemSource,
    GlobalSource,
    is_constant_source,
    LocalSource,
    Source,
    TypeSource,
)
from ..utils import (
    check_constant_args,
    check_numpy_ndarray_args,
    check_unspec_or_constant_args,
    check_unspec_python_args,
    dict_methods,
    extract_fake_example_value,
    get_fake_value,
    is_tensor_getset_descriptor,
    istype,
    numpy_operator_wrapper,
    proxy_args_kwargs,
    raise_args_mismatch,
    str_methods,
    tensortype_to_dtype,
    unpack_iterable,
)
from .base import (
    AsPythonConstantNotImplementedError,
    GetSet,
    Member,
    ValueMutationNew,
    VariableTracker,
)
from .constant import ConstantVariable, FakeIdVariable
from .dicts import (
    ConstDictVariable,
    DictItemsVariable,
    DictKeysVariable,
    DictViewVariable,
    OrderedDictVariable,
)
from .hashable import is_hashable
from .lists import BaseListVariable, ListVariable, TupleIteratorVariable, TupleVariable
from .misc import CellVariable, NullVariable, StringFormatVariable
from .object_protocol import (
    _NO_DEFAULT,
    binary_iop,
    binary_op,
    generic_delitem,
    generic_getattr,
    generic_getitem,
    generic_getiter,
    generic_hash,
    generic_is_true,
    generic_issubclass,
    generic_repr,
    generic_richcompare,
    generic_setitem,
    generic_size,
    generic_str,
    maybe_get_python_type,
    pycallable_check,
    pyiter_check,
    pylong_from_base,
    pynumber_absolute,
    pynumber_add,
    pynumber_float,
    pynumber_index,
    pynumber_inplace_add,
    pynumber_inplace_matrix_multiply,
    pynumber_inplace_multiply,
    pynumber_int,
    pynumber_invert,
    pynumber_matrix_multiply,
    pynumber_multiply,
    pynumber_negative,
    pynumber_positive,
    pynumber_tobase,
    pysequence_check,
    pysequence_contains,
    python_constant_richcompare_impl,
    ternary_iop,
    ternary_op,
    type_implements_mp_length,
    type_implements_sq_length,
    vt_identity_compare,
)
from .sets import FrozensetVariable, SetVariable
from .tensor import (
    FakeItemVariable,
    supported_comparison_ops,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .user_defined import UserDefinedObjectVariable, UserDefinedVariable


if TYPE_CHECKING:
    # Cyclic dependency...
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

log = logging.getLogger(__name__)


IN_PLACE_DESUGARING_MAP = {
    operator.iadd: operator.add,
    operator.isub: operator.sub,
    operator.imul: operator.mul,
    operator.ifloordiv: operator.floordiv,
    operator.itruediv: operator.truediv,
    operator.imod: operator.mod,
    operator.imatmul: operator.imatmul,
    operator.ilshift: operator.lshift,
    operator.irshift: operator.rshift,
    operator.ipow: operator.pow,
    operator.iand: operator.and_,
    operator.ior: operator.or_,
    operator.ixor: operator.xor,
}

_BUILTIN_CONSTANT_FOLDABLE_METHODS: dict[type, frozenset[str]] = {
    int: frozenset({"__new__", "from_bytes"}),
    bool: frozenset({"__new__", "from_bytes"}),
    float: frozenset({"fromhex", "hex"}),
}
if sys.version_info >= (3, 14):
    _BUILTIN_CONSTANT_FOLDABLE_METHODS[complex] = frozenset({"from_number"})


_HandlerCallback = Callable[
    ["InstructionTranslatorBase", typing.Any, typing.Any], VariableTracker | None
]
_TrackersType = type[VariableTracker] | tuple[type[VariableTracker], ...]
_OPERATOR_TO_DUNDER: dict[Callable[..., Any], str] = {
    operator.eq: "__eq__",
    operator.ne: "__ne__",
    operator.lt: "__lt__",
    operator.le: "__le__",
    operator.gt: "__gt__",
    operator.ge: "__ge__",
}


bin_ops = (
    operator.pow,
    operator.mul,
    operator.matmul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.add,
    operator.lt,
    operator.gt,
    operator.ge,
    operator.le,
    operator.ne,
    operator.eq,
    operator.sub,
    operator.ipow,
    operator.imul,
    operator.imatmul,
    operator.ifloordiv,
    operator.itruediv,
    operator.imod,
    operator.iadd,
    operator.isub,
)

bin_int_ops = (
    operator.and_,
    operator.or_,
    operator.xor,
    operator.iand,
    operator.ixor,
    operator.ior,
)

un_int_ops = (operator.invert,)

tensor_and_int_ops = (
    operator.lshift,
    operator.rshift,
    operator.ilshift,
    operator.irshift,
    operator.getitem,
)

un_ops = (
    operator.abs,
    operator.pos,
    operator.neg,
    operator.not_,  # Note: this has a local scalar dense call
    operator.length_hint,
)

_SET_LIKE_OP_SUPPORT: tuple[type[VariableTracker], ...] = (
    DictItemsVariable,
    DictKeysVariable,
    SetVariable,
    UserDefinedObjectVariable,
)

BUILTIN_TO_TENSOR_FN_MAP: dict[Callable[..., Any], Callable[..., Any]] = {}

# These functions represent the r* versions of the above ops
# Basically, if __add__(1, Tensor) is called, it is translated
# to __radd__(Tensor, 1).
# In the builtin var, we check if there is a tensor in the first args position,
# if not, we swap the args and use the r* version of the op.
BUILTIN_TO_TENSOR_RFN_MAP: dict[Callable[..., Any], Callable[..., Any]] = {}

# Sentinel for `inspect.getattr_static` lookups that must distinguish
# "attribute absent" from "attribute is None" (e.g. `__reversed__ = None`
# opt-out).
_MISSING_SENTINEL = object()

_COMPUTED_LAZY_CONSTANT_OPS: frozenset[Callable[..., Any]] = frozenset(
    [
        operator.add,
        operator.sub,
        operator.mul,
    ]
)


def _try_computed_lazy_constant(
    fn: Callable[..., Any], args: list[VariableTracker]
) -> VariableTracker | None:
    """Build a ComputedLazyConstantVariable for fn(*args), or None to fall back."""
    from .lazy import ComputedLazyConstantVariable, LazyConstantVariable

    fn = IN_PLACE_DESUGARING_MAP.get(fn, fn)
    if fn not in _COMPUTED_LAZY_CONSTANT_OPS or len(args) != 2:
        return None
    any_unrealized = False
    for arg in args:
        if (
            isinstance(arg, (LazyConstantVariable, ComputedLazyConstantVariable))
            and not arg.is_realized()
        ):
            any_unrealized = True
        elif not isinstance(arg, ConstantVariable):
            return None
    if not any_unrealized:
        return None
    return ComputedLazyConstantVariable.create(fn, args)


def populate_builtin_to_tensor_fn_map() -> None:
    global BUILTIN_TO_TENSOR_FN_MAP
    if len(BUILTIN_TO_TENSOR_FN_MAP) > 0:
        # Only populate once; after there are elements present no need to
        # repopulate
        return
    most_recent_func: Callable[..., Any] | None = None

    class GetMethodMode(BaseTorchFunctionMode):
        """
        Mode to extract the correct methods from torch function invocations
        (Used to get the correct torch.Tensor methods from builtins)
        """

        def __torch_function__(
            self,
            func: Callable[..., Any],
            types: Any,
            args: Sequence[Any] = (),
            kwargs: dict[str, Any] | None = None,
        ) -> Any:
            kwargs = kwargs or {}
            nonlocal most_recent_func
            most_recent_func = func
            return func(*args, **kwargs)

    inp0 = torch.ones(1)
    inp1 = torch.ones(1)
    inp0_int = torch.ones(1, dtype=torch.int32)
    inp1_int = torch.ones(1, dtype=torch.int32)
    with GetMethodMode():
        setups_and_oplists: list[tuple[Callable[..., Any], Iterable[Any]]] = [
            (lambda o: o(inp0), un_ops),
            (lambda o: o(inp0_int), un_int_ops),
            (lambda o: o(inp0, inp1), bin_ops),
            (lambda o: o(inp0_int, inp1_int), bin_int_ops),
            (lambda o: o(inp0_int, 0), tensor_and_int_ops),
        ]
        for setup_fn, op_list in setups_and_oplists:
            for op in op_list:
                setup_fn(op)
                if most_recent_func is None:
                    raise AssertionError(
                        f"most_recent_func is None after setup for op {op}"
                    )
                BUILTIN_TO_TENSOR_FN_MAP[op] = most_recent_func

        # gather the reverse functions
        rsetups_and_oplists: list[tuple[Callable[..., Any], Iterable[Any]]] = [
            (
                lambda o: o(1, inp1),
                bin_ops,
            ),  # Get r* ops, (ex. __sub__(int, Tensor) -> __rsub__(Tensor, int))
            (lambda o: o(1, inp1_int), bin_int_ops),
            (lambda o: o(0, inp0_int), tensor_and_int_ops),
        ]

        rskips = {operator.matmul, operator.imatmul, operator.getitem}
        for setup_fn, op_list in rsetups_and_oplists:
            for op in op_list:
                if op in rskips:
                    continue
                setup_fn(op)
                if most_recent_func is None:
                    raise AssertionError(
                        f"most_recent_func is None after setup for reverse op {op}"
                    )
                if most_recent_func != BUILTIN_TO_TENSOR_FN_MAP[op]:
                    BUILTIN_TO_TENSOR_RFN_MAP[op] = most_recent_func


class BaseBuiltinVariable(VariableTracker):
    """
    Common base class for all builtin variable trackers (BuiltinVariable,
    DictBuiltinVariable, IterBuiltinVariable, and future specialized builtins).

    Provides shared implementations for guard installation, hasattr tracing,
    and Python-level hashability/equality.

    Specialized subclasses (e.g. DictBuiltinVariable) set `_fn` as a class
    attribute. BuiltinVariable stores the callable on the instance as `self.fn`
    and overrides as_python_constant / reconstruct / tp_getattro_impl accordingly.
    """

    _fn: Any = None

    # Type attribute readers shared by BaseBuiltinVariable and BuiltinVariable.
    # Each reader declines (returns None) when the wrapped callable is not a type,
    # letting getattro_impl fall through to its dynamic getattr handling.
    # CPython classification (Objects/typeobject.c):
    #   __bases__ -> type_getsets[] type_get_bases (getset)
    #   __base__  -> type_members[] {T_OBJECT, offsetof(tp_base), Py_READONLY}
    #   __flags__ -> type_members[] {T_ULONG, offsetof(tp_flags), Py_READONLY}
    def _type_get_bases(
        self: "BaseBuiltinVariable", tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        fn = self.as_python_constant()
        if not isinstance(fn, type):
            return None
        source = self.source and AttrSource(self.source, "__bases__")
        items = [
            VariableTracker.build(tx, b, source and GetItemSource(source, i))
            for i, b in enumerate(fn.__bases__)
        ]
        return variables.TupleVariable(items, source=source)

    def _type_get_base(
        self: "BaseBuiltinVariable", tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        fn = self.as_python_constant()
        if not isinstance(fn, type):
            return None
        source = self.source and AttrSource(self.source, "__base__")
        return VariableTracker.build(tx, fn.__base__, source)

    def _type_get_flags(
        self: "BaseBuiltinVariable", tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        fn = self.as_python_constant()
        if not isinstance(fn, type):
            return None
        source = self.source and AttrSource(self.source, "__flags__")
        return VariableTracker.build(tx, fn.__flags__, source)

    tp_getset = {"__bases__": GetSet(_type_get_bases, None)}
    tp_members = {
        "__base__": Member(_type_get_base, None),
        "__flags__": Member(_type_get_flags, None),
    }

    @classmethod
    def create_with_source(cls, value: Any, source: Source) -> "BaseBuiltinVariable":
        install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        return cls(source=source)

    def as_python_constant(self) -> Any:
        return self._fn

    def reconstruct(self, codegen: "PyCodegen") -> None:
        name = self.as_python_constant().__name__
        if name in codegen.tx.f_globals:
            raise AssertionError("shadowed global")
        codegen.append_output(codegen.create_load_global(name, add=True))

    def tp_getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        # Declarative type-attribute dispatch, mirroring the consultation at the
        # top of VariableTracker.getattro_impl. Inlined here because this
        # override keeps its own GetAttrVariable fallback instead of delegating
        # to super().
        fn = self.as_python_constant()
        source = self.source and AttrSource(self.source, name)
        attr = getattr(fn, name, None)
        return variables.GetAttrVariable(
            self, name, py_type=type(attr) if attr is not None else None, source=source
        )

    def call_obj_hasattr(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> ConstantVariable:
        return VariableTracker.build(tx, hasattr(self.as_python_constant(), name))  # type: ignore[return-value]

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # CPython meth_hash: https://github.com/python/cpython/blob/e76aa128fe/Objects/methodobject.c#L319
        return hash(self.as_python_constant()), False

    def tp_richcompare_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        op: str,
    ) -> VariableTracker:
        return python_constant_richcompare_impl(self, tx, other, op)

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "__str__" and len(args) == 1 and not kwargs:
            arg = args[0]
            if self.as_python_constant() is object:
                return generic_repr(tx, arg)
            if self.as_python_constant() is type:
                if isinstance(arg, variables.UserDefinedClassVariable):
                    return VariableTracker.build(tx, type.__str__(arg.value))
                if arg.is_python_constant() and isinstance(
                    arg.as_python_constant(), type
                ):
                    return VariableTracker.build(
                        tx, type.__str__(arg.as_python_constant())
                    )
            return generic_str(tx, arg)
        if name == "__repr__" and len(args) == 1 and not kwargs:
            arg = args[0]
            if self.as_python_constant() is object and isinstance(
                arg, variables.UserDefinedObjectVariable
            ):
                return VariableTracker.build(tx, object.__repr__(arg.value))
            if self.as_python_constant() is type:
                if isinstance(arg, variables.UserDefinedClassVariable):
                    return VariableTracker.build(tx, type.__repr__(arg.value))
                if arg.is_python_constant() and isinstance(
                    arg.as_python_constant(), type
                ):
                    return VariableTracker.build(
                        tx, type.__repr__(arg.as_python_constant())
                    )
            return generic_repr(tx, arg)
        return super().call_method(tx, name, args, kwargs)


def _uses_custom_classinfo_check(
    type_info: Any,
    *,
    use_instancecheck: bool = True,
    use_subclasscheck: bool = True,
) -> bool:
    if isinstance(type_info, tuple):
        return any(_uses_custom_classinfo_check(item) for item in type_info)
    if isinstance(type_info, types.UnionType):
        return any(
            _uses_custom_classinfo_check(item) for item in typing.get_args(type_info)
        )
    if typing.get_origin(type_info) is typing.Union:
        typing_union_uses_instancecheck = sys.version_info >= (3, 12)
        return any(
            _uses_custom_classinfo_check(
                item,
                use_instancecheck=typing_union_uses_instancecheck,
                use_subclasscheck=True,
            )
            for item in typing.get_args(type_info)
        )
    if isinstance(type_info, type):
        if type_info in tensortype_to_dtype:
            return False

        type_info_meta = type(type_info)
        instancecheck = getattr(type_info_meta, "__instancecheck__", None)
        subclasscheck = getattr(type_info_meta, "__subclasscheck__", None)
        if (
            instancecheck is type.__instancecheck__
            and subclasscheck is type.__subclasscheck__
        ):
            return False

        if issubclass(type_info, torch.Tensor) and (
            type_info_meta.__module__,
            type_info_meta.__qualname__,
        ) in {
            ("torch.nn.parameter", "_ParameterMeta"),
            ("torch.nn.parameter", "_BufferMeta"),
            ("torch.distributed.fsdp._flat_param", "_FlatParameterMeta"),
        }:
            return False

        return (
            use_instancecheck
            and instancecheck is not type.__instancecheck__
            and instancecheck is not abc.ABCMeta.__instancecheck__
        ) or (
            use_subclasscheck
            and subclasscheck is not type.__subclasscheck__
            and subclasscheck is not abc.ABCMeta.__subclasscheck__
        )
    return False


class BuiltinVariable(BaseBuiltinVariable):
    """
    A VariableTracker that represents a built-in value (functions and operators).
    A lot of the code here assumes it will be a function object.

    The BuiltinVariable class wraps Python built-in functions (like len, isinstance, etc.)
    and operators (like +, -, *, etc.) to enable symbolic execution during tracing. This allows
    Dynamo to properly handle these operations when converting Python code to FX graphs while
    maintaining correct semantics and enabling optimizations.
    """

    _SENTINEL = object()
    _nonvar_fields = {
        "fn",
        *VariableTracker._nonvar_fields,
    }

    # __name__ -> type_getsets[] type_get_name (getset). Unlike the type-attribute
    # readers on BaseBuiltinVariable, BuiltinVariable exposes __name__ for any
    # wrapped callable (not just types), so it never declines.
    def _builtin_type_get_name(
        self: "BuiltinVariable", tx: "InstructionTranslatorBase"
    ) -> "VariableTracker | None":
        source = self.source and AttrSource(self.source, "__name__")
        return VariableTracker.build(tx, self.fn.__name__, source)

    tp_getset = {
        "__name__": GetSet(_builtin_type_get_name, None),
    }

    @classmethod
    def create_with_source(cls, value: Any, source: Source) -> "BuiltinVariable":
        install_guard(source.make_guard(GuardBuilder.BUILTIN_MATCH))
        return cls(value, source=source)

    @staticmethod
    @functools.cache
    def _constant_fold_functions() -> set[Callable[..., Any]]:
        fns: set[Callable[..., Any]] = {
            abs,
            all,
            any,
            ascii,
            bin,
            bool,
            callable,
            chr,
            complex,
            divmod,
            float,
            format,
            getattr,
            hex,
            int,
            len,
            max,
            min,
            oct,
            ord,
            pow,
            repr,
            round,
            str,
            str.format,
            sum,
            type,
            operator.abs,
            operator.pos,
            operator.neg,
            operator.not_,
            operator.truth,
            operator.invert,
            operator.pow,
            operator.mul,
            operator.matmul,
            operator.floordiv,
            operator.truediv,
            operator.mod,
            operator.add,
            operator.sub,
            operator.getitem,
            operator.length_hint,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            operator.ipow,
            operator.imul,
            operator.imatmul,
            operator.ifloordiv,
            operator.itruediv,
            operator.imod,
            operator.iadd,
            operator.isub,
            operator.ilshift,
            operator.irshift,
            operator.iand,
            operator.ixor,
            operator.ior,
            operator.index,
        }
        from .tensor import supported_comparison_ops

        fns.update(supported_comparison_ops.values())
        fns.update(x for x in math.__dict__.values() if isinstance(x, type(math.sqrt)))
        return fns

    def can_constant_fold_through(self) -> bool:
        return self.fn in self._constant_fold_functions()

    @staticmethod
    @functools.cache
    def _fx_graph_functions() -> set[Callable[..., Any]]:
        fns = {
            operator.abs,
            operator.pos,
            operator.neg,
            operator.not_,
            operator.invert,
            operator.pow,
            operator.mul,
            operator.matmul,
            operator.floordiv,
            operator.truediv,
            operator.mod,
            operator.add,
            operator.lt,
            operator.gt,
            operator.ge,
            operator.le,
            operator.ne,
            operator.eq,
            operator.sub,
            operator.length_hint,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            operator.ipow,
            operator.imul,
            operator.imatmul,
            operator.ifloordiv,
            operator.itruediv,
            operator.getitem,
            operator.imod,
            operator.iadd,
            operator.isub,
            operator.ilshift,
            operator.irshift,
            operator.iand,
            operator.ixor,
            operator.ior,
        }
        return fns  # type: ignore[return-value]

    @staticmethod
    @functools.cache
    def _binops() -> dict[
        Callable[..., object], tuple[list[str], Callable[..., object]]
    ]:
        # function -> ([forward name, reverse name, in-place name], in-place op)
        fns: dict[Callable[..., object], tuple[list[str], Callable[..., object]]] = {
            # NB: The follow binary operators are not supported for now, since the
            # corresponding magic methods aren't defined on SymInt / SymFloat:
            # operator.matmul
            # divmod
            # operator.and_
            # operator.or_
        }
        return fns

    @staticmethod
    @functools.cache
    def _binop_handlers() -> dict[
        Callable[..., object],
        list[
            tuple[
                tuple[
                    type[VariableTracker],
                    _TrackersType,
                ],
                _HandlerCallback,
            ]
        ],
    ]:
        # Multiple dispatch mechanism defining custom binop behavior for certain type
        # combinations. Handlers are attempted in order, and will be used if the type checks
        # match. They are expected to have the signature:
        # fn(tx, arg0: VariableTracker, arg1: VariableTracker) -> VariableTracker
        from .functions import BaseUserFunctionVariable
        from .nn_module import NNModuleVariable
        from .tensor import supported_const_comparison_ops
        from .torch import BaseTorchVariable

        # Override table contains: op_fn -> [list of handlers]
        op_handlers: dict[Any, list[Any]] = {}
        for (
            op,
            (magic_method_names, in_place_op),
        ) in BuiltinVariable._binops().items():
            op_handlers[op] = []
            op_handlers[in_place_op] = []

            forward_name, reverse_name, inplace_name = magic_method_names

            # User-defined args (highest precedence)
            def user_defined_handler(
                tx: "InstructionTranslatorBase",
                a: VariableTracker,
                b: VariableTracker,
                *,
                forward_name: str = forward_name,
                reverse_name: str = reverse_name,
            ) -> VariableTracker:
                # Manually handle reversing logic if needed (e.g. call __radd__)

                # TODO: If we expand this to handle tensor args, we need to manually
                # handle cases like this:
                #
                # class A(int):
                #     def __radd__(self, other):
                #         print("woof")
                # torch.randn(3) + A(3)
                #
                # In this example, A.__radd__() is not called -> nothing is printed, because
                # Tensor.__add__ only does a subtype test against int, ignoring the subclass.
                # To be fully correct, we should not call A.__radd__() here, and there may be
                # other cases to reason about and add exceptions for.
                if isinstance(a, UserDefinedVariable):
                    return a.call_method(tx, forward_name, [b], {})
                else:
                    return b.call_method(tx, reverse_name, [a], {})

            op_handlers[op].append(
                ((UserDefinedVariable, VariableTracker), user_defined_handler)
            )
            op_handlers[op].append(
                ((VariableTracker, UserDefinedVariable), user_defined_handler)
            )

            def user_defined_inplace_handler(
                tx: "InstructionTranslatorBase",
                a: VariableTracker,
                b: VariableTracker,
                *,
                forward_name: str = inplace_name,
            ) -> VariableTracker:
                return a.call_method(tx, forward_name, [b], {})

            op_handlers[in_place_op].append(
                ((UserDefinedVariable, VariableTracker), user_defined_inplace_handler)
            )
            op_handlers[in_place_op].append(
                ((VariableTracker, UserDefinedVariable), user_defined_inplace_handler)
            )

            # Dynamic shape args
            def dynamic_handler(
                tx: "InstructionTranslatorBase",
                a: VariableTracker,
                b: VariableTracker,
                *,
                fn: Callable[..., Any] = op,
            ) -> VariableTracker:
                from .builder import wrap_fx_proxy

                return wrap_fx_proxy(
                    tx,
                    tx.output.create_proxy(
                        "call_function", fn, *proxy_args_kwargs([a, b], {})
                    ),
                )

            op_handlers[op].append(
                ((SymNodeVariable, VariableTracker), dynamic_handler)
            )
            op_handlers[op].append(
                ((VariableTracker, SymNodeVariable), dynamic_handler)
            )

            # NB: Prefer out-of-place op when calling in-place op to generate valid graph
            op_handlers[in_place_op].append(
                ((SymNodeVariable, VariableTracker), dynamic_handler)
            )
            op_handlers[in_place_op].append(
                ((VariableTracker, SymNodeVariable), dynamic_handler)
            )

        # Special cases - lower precedence but still prefer these over constant folding

        def create_cmp_op_handlers(
            op: Callable[..., Any],
        ) -> list[tuple[tuple[_TrackersType, _TrackersType], _HandlerCallback]]:
            def compare_by_value(
                tx: "InstructionTranslatorBase", a: VariableTracker, b: VariableTracker
            ) -> VariableTracker:
                try:
                    return VariableTracker.build(tx, op(a.value, b.value))  # type: ignore[attr-defined]
                except TypeError as exc:
                    raise_observed_exception(
                        type(exc),
                        tx,
                        args=list(exc.args),
                    )

            result: list[
                tuple[
                    tuple[
                        _TrackersType,
                        _TrackersType,
                    ],
                    _HandlerCallback,
                ]
            ] = [((ConstantVariable, ConstantVariable), compare_by_value)]

            if op in _OPERATOR_TO_DUNDER:
                # For constants, speedup the comparison instead of going
                # through generic_richcompare. Removing this line causes
                # major regression for pr time benchmark - add_loop_eager.
                result = [
                    ((ConstantVariable, ConstantVariable), compare_by_value),
                ]

                op_var = BuiltinVariable(op)
                # Special handling of SymNode variable
                result.extend(
                    [
                        (
                            (SymNodeVariable, VariableTracker),
                            op_var._comparison_with_symnode,
                        ),
                        (
                            (VariableTracker, SymNodeVariable),
                            op_var._comparison_with_symnode,
                        ),
                    ]
                )

                # COMPARE_OP (a == b) dispatches through generic_richcompare,
                # which implements do_richcompare via tp_richcompare_impl slots.
                # See object_protocol.py for details.
                dunder = _OPERATOR_TO_DUNDER[op]

                def handler(
                    tx: "InstructionTranslatorBase",
                    a: VariableTracker,
                    b: VariableTracker,
                ) -> VariableTracker:
                    return generic_richcompare(tx, a, b, dunder)

                result.append(((VariableTracker, VariableTracker), handler))
                return result

            result = [((ConstantVariable, ConstantVariable), compare_by_value)]

            if op in supported_const_comparison_ops.values() and op.__name__.startswith(
                "is_"
            ):
                # Tensor is None, List is not None, etc
                none_result = op(object(), None)

                def never(
                    tx: "InstructionTranslatorBase",
                    a: VariableTracker,
                    b: VariableTracker,
                ) -> VariableTracker:
                    return VariableTracker.build(tx, none_result)

                obj_op_none = never
                none_op_obj = never

                types_that_are_never_none = (
                    TensorVariable,
                    SymNodeVariable,
                    NNModuleVariable,
                    BaseListVariable,
                    UserDefinedVariable,
                    BaseUserFunctionVariable,
                    ConstDictVariable,
                    BaseTorchVariable,
                )
                result.extend(
                    [
                        (
                            (types_that_are_never_none, ConstantVariable),
                            obj_op_none,
                        ),
                        (
                            (ConstantVariable, types_that_are_never_none),
                            none_op_obj,
                        ),
                    ]
                )

                op_var = BuiltinVariable(op)
                result.extend(
                    [
                        (
                            (TensorVariable, VariableTracker),
                            op_var._comparison_with_tensor,
                        ),
                        (
                            (VariableTracker, TensorVariable),
                            op_var._comparison_with_tensor,
                        ),
                        (
                            (SymNodeVariable, VariableTracker),
                            op_var._comparison_with_symnode,
                        ),
                        (
                            (VariableTracker, SymNodeVariable),
                            op_var._comparison_with_symnode,
                        ),
                    ]
                )

                def handle_is(
                    tx: "InstructionTranslatorBase",
                    left: VariableTracker,
                    right: VariableTracker,
                ) -> VariableTracker | None:
                    result = vt_identity_compare(left, right)
                    if result is None:
                        return None
                    is_same = result.as_python_constant()
                    return VariableTracker.build(
                        tx, is_same if op.__name__ == "is_" else not is_same
                    )

                result.append(((VariableTracker, VariableTracker), handle_is))  # type: ignore[arg-type]

            return result

        for op in supported_comparison_ops.values():
            if not callable(op):
                raise AssertionError(f"comparison op {op} is not callable")
            if op in op_handlers:
                raise AssertionError(f"duplicate handler for op {op}")
            op_handlers[op] = create_cmp_op_handlers(op)

        return op_handlers

    @staticmethod
    def _find_binop_handler(
        op: Callable[..., Any], a_type: type[VariableTracker], b_type: type
    ) -> list[_HandlerCallback] | None:
        handlers = BuiltinVariable._binop_handlers().get(op)
        if handlers is None:
            return None

        matches = []
        for (type1, type2), handler in handlers:
            if issubclass(a_type, type1) and issubclass(b_type, type2):
                matches.append(handler)
        return matches

    def can_insert_in_graph(self) -> bool:
        return self.fn in self._fx_graph_functions()

    # Builtins that have been promoted to their own VT classes. Creating a
    # BuiltinVariable for these is a bug; use the specialized class instead.
    MUST_USE_SPECIALIZED: frozenset[Any] = frozenset(
        {dict, getattr, hasattr, iter, list, setattr}
    )

    def __init__(self, fn: Any, **kwargs: Any) -> None:
        if fn in self.MUST_USE_SPECIALIZED:
            raise AssertionError(
                f"Use the specialized VT class for {fn!r}, not BuiltinVariable. "
                f"E.g. DictBuiltinVariable for dict."
            )
        super().__init__(**kwargs)
        self.fn = fn

    def __repr__(self) -> str:
        if self.fn is None:
            name = "None"
        else:
            name = self.fn.__name__

        return f"{self.__class__.__name__}({name})"

    def as_python_constant(self) -> Any:
        return self.fn

    def get_real_python_backed_value(self) -> Any:
        return self.fn

    def nb_or_impl(
        self,
        tx: "InstructionTranslatorBase",
        other: VariableTracker,
        reverse: bool = False,
    ) -> VariableTracker:
        # BuiltinVariable wraps built-in types like list, tuple, dict.
        # type(self.fn).__or__(self.fn, other_val) delegates to CPython's
        # _Py_union_type_or for type unions (e.g. list | tuple).
        # https://github.com/python/cpython/blob/v3.13.0/Objects/typeobject.c#L6028-L6030 (type_as_number.nb_or = _Py_union_type_or)
        # https://github.com/python/cpython/blob/3.13/Objects/unionobject.c#L162 (_Py_union_type_or)
        if not isinstance(self.fn, type):
            return VariableTracker.build(tx, NotImplemented)
        try:
            other_val = other.as_python_constant()
        except NotImplementedError:
            return VariableTracker.build(tx, NotImplemented)
        # pyrefly: ignore[bad-argument-count]
        result = type(self.fn).__or__(self.fn, other_val)
        if result is NotImplemented:
            return VariableTracker.build(tx, NotImplemented)
        return VariableTracker.build(tx, result)

    def as_proxy(self) -> Any:
        DTYPE = {
            bool: torch.bool,
            int: torch.int64,
            float: torch.float64,
        }
        if self.fn in DTYPE:
            return DTYPE[self.fn]
        return super().as_proxy()

    def reconstruct(self, codegen: "PyCodegen") -> None:
        name = self.fn.__name__
        if self.fn.__module__ != "builtins":
            raise AssertionError(f"Expected builtins module, got {self.fn.__module__}")
        if name in codegen.tx.f_globals:
            raise AssertionError("shadowed global")
        codegen.append_output(codegen.create_load_global(name, add=True))

    def constant_args(self, *args: VariableTracker, **kwargs: VariableTracker) -> bool:
        return check_constant_args(args, kwargs)

    def tensor_args(self, *args: VariableTracker) -> bool:
        any_tensor = False
        for arg in args:
            if isinstance(arg, variables.GetAttrVariable):
                return False
            any_tensor = any_tensor or arg.is_tensor()
        return any_tensor

    def tensor_args_type(self, arg_types: list[type]) -> bool:
        any_tensor = False
        for arg_type in arg_types:
            if issubclass(arg_type, variables.GetAttrVariable):
                return False
            any_tensor = any_tensor or issubclass(arg_type, variables.TensorVariable)
        return any_tensor

    def python_and_tensor_constant_only(
        self, *args: VariableTracker, **kwargs: VariableTracker
    ) -> bool:
        tensor_args = []
        non_tensor_args = []
        for i in itertools.chain(args, kwargs.values()):
            if i.is_tensor():
                tensor_args.append(i)
            else:
                non_tensor_args.append(i)
        return all(
            is_constant_source(t.source) if t.source is not None else False
            for t in tensor_args
        ) and self.constant_args(*non_tensor_args)

    @staticmethod
    def unwrap_unspec_args_kwargs(
        args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> tuple[list[Any], dict[str, Any]]:
        return [x.as_python_constant() for x in args], {
            k: v.as_python_constant() for k, v in kwargs.items()
        }

    def has_constant_handler(
        self, args: list[VariableTracker], kwargs: dict[str, VariableTracker]
    ) -> bool:
        return self.can_constant_fold_through() and check_unspec_or_constant_args(
            args, kwargs
        )

    @staticmethod
    def _make_handler(
        fn: Callable[..., Any], arg_types: list[type], has_kwargs: bool
    ) -> Callable[
        [
            "InstructionTranslatorBase",
            list[VariableTracker],
            dict[str, VariableTracker],
        ],
        VariableTracker | None,
    ]:
        from .lazy import (
            ComputedLazyConstantVariable,
            LazyConstantVariable,
            LazyVariableTracker,
        )

        obj = BuiltinVariable(fn)
        handlers: list[_HandlerCallback] = []

        lazy_constant_types = (LazyConstantVariable, ComputedLazyConstantVariable)
        lazy_types = [t for t in arg_types if issubclass(t, LazyVariableTracker)]
        if lazy_types:
            if not all(issubclass(t, lazy_constant_types) for t in lazy_types):
                # Realize non-constant lazy args and re-dispatch.  Any
                # lazy constant args are kept and handled on the
                # second dispatch through the branch below.
                return lambda tx, args, kwargs: obj.call_function(
                    tx,
                    [
                        a.realize()
                        if isinstance(a, LazyVariableTracker)
                        and not isinstance(a, lazy_constant_types)
                        else a
                        for a in args
                    ],
                    kwargs,
                )

            # Only lazy constant types.  Install type guards
            # and resolve the dispatch type.  If the resolved type is
            # ConstantVariable (the common case), delegate to a handler
            # built for ConstantVariable.  Otherwise (e.g. specialize_int=
            # False turned the int into a SymNodeVariable), realize and
            # re-dispatch so the correct handler is used.
            inner_handler = BuiltinVariable._make_handler(
                fn,
                [
                    ConstantVariable if issubclass(t, lazy_constant_types) else t
                    for t in arg_types
                ],
                has_kwargs,
            )

            def lazy_constant_handler(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                if not kwargs:
                    result = _try_computed_lazy_constant(fn, args)
                    if result is not None:
                        return result
                for a in args:
                    if isinstance(a, lazy_constant_types):
                        if a.get_handler_type_for_dispatch() is not ConstantVariable:
                            return obj.call_function(
                                tx,
                                [
                                    v.realize()
                                    if isinstance(v, lazy_constant_types)
                                    else v
                                    for v in args
                                ],
                                kwargs,
                            )
                return inner_handler(tx, args, kwargs)

            return lazy_constant_handler

        if inspect.isclass(fn) and (
            issubclass(fn, BaseException)
            # GeneratorExit doesn't inherit from Exception
            # >>> issubclass(GeneratorExit, Exception)
            # False
            or fn is GeneratorExit
        ):

            def create_exception_class_object(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker:
                if fn is AssertionError and not all(
                    x.is_python_constant() and isinstance(x.as_python_constant(), str)
                    for x in args
                ):
                    unimplemented(
                        gb_type="assert with non-string message",
                        context=str(args),
                        explanation="Dynamo only supports asserts with string messages",
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )

                if fn is StopIteration:
                    return variables.StopIterationVariable(fn, args, kwargs)
                elif fn is AttributeError:
                    return variables.AttributeErrorVariable(fn, args, kwargs)
                elif fn is NameError:
                    return variables.NameErrorVariable(fn, args, kwargs)
                return variables.ExceptionVariable(fn, args, kwargs)

            return create_exception_class_object

        if obj.can_insert_in_graph() and not (
            fn is operator.getitem
            and (
                len(arg_types) != 2
                or not issubclass(arg_types[0], variables.TensorVariable)
            )
        ):
            if obj.tensor_args_type(arg_types):
                return obj._handle_insert_op_in_graph
            elif has_kwargs:
                # need runtime check for kwargs
                handlers.append(obj._handle_insert_op_in_graph)

        # Handle binary ops (e.g. __add__ / __radd__, __iadd__, etc.)
        # NB: Tensor args are handled above and not here
        if len(arg_types) == 2 and not has_kwargs:
            # Try to find a handler for the arg types; otherwise, fall through to constant handler
            binop_handlers = BuiltinVariable._find_binop_handler(fn, *arg_types)
            if not binop_handlers:
                pass
            elif len(binop_handlers) == 1:
                (binop_handler,) = binop_handlers
                handlers.append(lambda tx, args, _: binop_handler(tx, *args))
            else:

                def call_binop_handlers(
                    tx: "InstructionTranslatorBase", args: Any, _: Any
                ) -> Any:
                    # pyrefly: ignore [not-iterable]
                    for fn in binop_handlers:
                        rv = fn(tx, *args)
                        if rv:
                            return rv
                    return None

                handlers.append(call_binop_handlers)

        self_handler = getattr(obj, f"call_{fn.__name__}", None)
        if self_handler:

            def call_self_handler(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                try:
                    # pyrefly: ignore [not-callable]
                    return self_handler(tx, *args, **kwargs)
                except TypeError:
                    # Check if binding is bad. inspect signature bind is expensive.
                    # So check only when handler call fails.
                    try:
                        # pyrefly: ignore [bad-argument-type]
                        inspect.signature(self_handler).bind(tx, *args, **kwargs)
                    except TypeError as e:
                        has_constant_handler = obj.has_constant_handler(args, kwargs)
                        if not has_constant_handler:
                            log.warning(
                                "incorrect arg count %s %s and no constant handler",
                                self_handler,
                                e,
                            )
                            unimplemented(
                                gb_type="invalid call to builtin op handler",
                                context=f"invalid args to {self_handler}: {args} {kwargs}",
                                explanation=f"Encountered TypeError when trying to handle op {fn.__name__}",
                                hints=[*graph_break_hints.DIFFICULT],
                            )
                    else:
                        raise
                except Unsupported as exc:
                    has_constant_handler = obj.has_constant_handler(args, kwargs)
                    if not has_constant_handler:
                        raise
                    # Actually, we will handle this just fine
                    exc.remove_from_stats()
                return None

            handlers.append(call_self_handler)

        if obj.can_constant_fold_through():
            if (
                all(issubclass(x, ConstantVariable) for x in arg_types)
                and not has_kwargs
            ):

                def constant_fold_handler(
                    tx: "InstructionTranslatorBase",
                    args: list[VariableTracker],
                    kwargs: dict[str, VariableTracker],
                ) -> VariableTracker | None:
                    # fast path
                    try:
                        res = fn(
                            *[x.as_python_constant() for x in args],
                        )
                    except Exception as exc:
                        raise_observed_exception(
                            type(exc),
                            tx,
                            args=list(exc.args),
                        )
                    except AsPythonConstantNotImplementedError as exc:
                        unimplemented(
                            gb_type="constant fold exception",
                            context=f"attempted to run function {fn} with arguments {args}",
                            explanation="Encountered exception when attempting to constant fold.",
                            hints=[*graph_break_hints.DYNAMO_BUG],
                            from_exc=exc,
                        )
                    return VariableTracker.build(tx, res)

            else:

                def constant_fold_handler(
                    tx: "InstructionTranslatorBase",
                    args: list[VariableTracker],
                    kwargs: dict[str, VariableTracker],
                ) -> VariableTracker | None:
                    # path with a runtime check
                    if check_unspec_or_constant_args(args, kwargs):
                        try:
                            res = fn(
                                *[x.as_python_constant() for x in args],
                                **{
                                    k: v.as_python_constant() for k, v in kwargs.items()
                                },
                            )
                        except AsPythonConstantNotImplementedError as exc:
                            unimplemented(
                                gb_type="constant fold exception",
                                context=f"attempted to run function {fn} with arguments {args}",
                                explanation="Encountered exception when attempting to constant fold.",
                                hints=[*graph_break_hints.DYNAMO_BUG],
                                from_exc=exc,
                            )
                        except Exception as exc:
                            raise_observed_exception(
                                type(exc),
                                tx,
                                args=list(exc.args),
                            )
                        return VariableTracker.build(tx, res)
                    return None

            handlers.append(constant_fold_handler)

        def call_unimplemented(args: list[VariableTracker]) -> None:
            real_arg_types = [arg.python_type_name() for arg in args]
            unimplemented(
                gb_type="Failed to trace builtin operator",
                context=f"builtin {fn.__name__} {arg_types} {has_kwargs}",
                explanation=f"Dynamo does not know how to trace builtin operator `{fn.__name__}` "
                f"with argument types {real_arg_types} (has_kwargs {has_kwargs})",
                hints=[
                    f"Avoid calling builtin `{fn.__name__}` with argument types {real_arg_types}. "
                    f"Consider using an equivalent alternative function/method to `{fn.__name__}`.",
                    "If you are attempting to call a logging function (e.g. `print`), "
                    "you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.",
                    "Please report an issue to PyTorch.",
                ],
            )

        if len(handlers) == 0:
            return lambda tx, args, kwargs: call_unimplemented(args)
        elif len(handlers) == 1:
            (handler,) = handlers

            def builtin_dispatch(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                rv = handler(tx, args, kwargs)
                if rv:
                    return rv
                call_unimplemented(args)
                return rv

        else:

            def builtin_dispatch(
                tx: "InstructionTranslatorBase",
                args: list[VariableTracker],
                kwargs: dict[str, VariableTracker],
            ) -> VariableTracker | None:
                rv = None
                for fn in handlers:
                    rv = fn(tx, args, kwargs)
                    if rv:
                        return rv
                call_unimplemented(args)
                return rv

        return builtin_dispatch

    @staticmethod
    def _constant_eval_numeric_expr(node: ast.AST) -> bool:
        allowed_nodes = (
            ast.Expression,
            ast.Constant,
            ast.UnaryOp,
            ast.BinOp,
            ast.UAdd,
            ast.USub,
            ast.Invert,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.FloorDiv,
            ast.Mod,
            ast.BitOr,
            ast.BitXor,
            ast.BitAnd,
        )
        return all(
            isinstance(child, allowed_nodes)
            and (
                not isinstance(child, ast.Constant)
                or isinstance(child.value, (bool, int, float, complex))
            )
            for child in ast.walk(node)
        )

    @staticmethod
    def _constant_eval_result(
        tx: "InstructionTranslatorBase", tree: ast.Expression, filename: str
    ) -> VariableTracker | None:
        if any(isinstance(child, ast.Call) for child in ast.walk(tree)):
            return None
        ast.fix_missing_locations(tree)
        try:
            result = ast.literal_eval(tree)
        except ValueError:
            if not BuiltinVariable._constant_eval_numeric_expr(tree):
                return None
            try:
                result = eval(
                    compile(tree, filename, "eval"),
                    {"__builtins__": {}},
                    {},
                )
            except Exception as exc:
                raise_observed_exception(type(exc), tx, args=list(exc.args))
        return VariableTracker.build(tx, result)

    def call_eval(
        self,
        tx: "InstructionTranslatorBase",
        source: VariableTracker,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker | None:
        if args or kwargs:
            return None
        if not source.is_python_constant():
            return None
        source_str = source.as_python_constant()
        if not isinstance(source_str, str):
            return None

        try:
            tree = ast.parse(source_str.strip(), mode="eval")
        except SyntaxError as exc:
            raise_observed_exception(SyntaxError, tx, args=[exc.msg])

        return self._constant_eval_result(tx, tree, "<torch._dynamo.eval>")

    def call_vars(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        if kwargs:
            raise_type_error(tx, "vars() takes no keyword arguments")
        if len(args) == 0:
            return self._call_frame_locals_snapshot(tx)
        if len(args) != 1:
            raise_type_error(tx, f"vars expected at most 1 argument, got {len(args)}")
        # vars(obj) is obj.__dict__ if __dict__ is present else TypeError
        try:
            return args[0].tp_getattro_impl(tx, "__dict__")
        except ObservedAttributeError:
            raise_observed_exception(TypeError, tx)

    def call_locals(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        if len(args) != 0:
            raise_observed_exception(TypeError, tx)
        return self._call_frame_locals_snapshot(tx)

    @staticmethod
    def _call_frame_locals_snapshot(tx: "InstructionTranslatorBase") -> VariableTracker:
        from .builder import VariableBuilder

        frame_local_names = set(tx.f_code.co_varnames) | set(tx.cell_and_freevars())
        cell_and_freevars = set(tx.cell_and_freevars())
        frame_locals = {}
        # symbolic_cellvars registers all of the frame's cells. Cells are listed first so that a colliding fast local of
        # the same name shadows the cell, matching CPython (except for 3.12, which is backwards): the two share a name
        # but not a localsplus slot, and the fast slot wins while it holds a value (an empty one is skipped below,
        # leaving the cell contents visible).
        if sys.version_info[:2] == (3, 12):
            its = (tx.symbolic_locals.items(), tx.symbolic_cellvars.items())
        else:
            its = (tx.symbolic_cellvars.items(), tx.symbolic_locals.items())

        for name, value in itertools.chain(*its):
            if name not in frame_local_names:
                continue
            # Match on CellVariable, not name: a colliding fast local shares a
            # cell's name but is not itself a cell.
            if type.__instancecheck__(CellVariable, value):
                value = tx.output.side_effects.load_cell(value)
            if type.__instancecheck__(NullVariable, value) or isinstance(
                value, variables.DeletedVariable
            ):
                continue
            frame_locals[ConstantVariable.create(name)] = value
        # Include locals pruned from symbolic_locals by the unused-input optimisation.
        # locals()/vars() observes the entire namespace, so build VTs (installing
        # guards) for any pruned entry that still has a runtime value.
        for name in frame_local_names:
            if name in frame_locals or name in cell_and_freevars:
                continue
            if name not in tx.f_locals:
                continue
            vt = VariableBuilder(tx, LocalSource(name))(tx.f_locals[name])
            frame_locals[ConstantVariable.create(name)] = vt
        return ConstDictVariable(
            frame_locals,
            mutation_type=ValueMutationNew(),
        )

    def _handle_insert_op_in_graph(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker | None:
        from .builder import wrap_fx_proxy, wrap_fx_proxy_cls

        if kwargs and not self.tensor_args(*args, *kwargs.values()):
            return None

        # insert handling for torch function here
        from .builder import SourcelessBuilder
        from .torch_function import can_dispatch_torch_function, dispatch_torch_function

        global BUILTIN_TO_TENSOR_RFN_MAP, BUILTIN_TO_TENSOR_FN_MAP
        if can_dispatch_torch_function(tx, args, kwargs):
            # Only remap the fn to tensor methods if we aren't exporting
            # export serde does not handle method descriptors today
            if not tx.export:
                # Ensure the builtin maps are populated before accessing them
                populate_builtin_to_tensor_fn_map()
                # Use sourceless builder, we built the map ourselves
                if not args[0].is_tensor():
                    if self.fn in BUILTIN_TO_TENSOR_RFN_MAP:
                        func = BUILTIN_TO_TENSOR_RFN_MAP[self.fn]
                    else:
                        func = BUILTIN_TO_TENSOR_FN_MAP[self.fn]

                    tmp = args[0]
                    # swap args and call reverse version of func
                    args[0] = args[1]  # type: ignore[index]
                    args[1] = tmp  # type: ignore[index]
                else:
                    func = BUILTIN_TO_TENSOR_FN_MAP[self.fn]
            else:
                func = self.fn

            fn_var = SourcelessBuilder.create(tx, func)

            return dispatch_torch_function(tx, fn_var, args, kwargs)

        fn = self.fn
        try:
            # Constant fold for constant tensor and python constants
            if self.python_and_tensor_constant_only(*args, **kwargs):
                from ..bytecode_transformation import unique_id
                from .functions import invoke_and_store_as_constant

                return invoke_and_store_as_constant(
                    tx, fn, unique_id(fn.__name__), args, kwargs
                )

            if fn in IN_PLACE_DESUGARING_MAP and isinstance(
                args[0], variables.ConstantVariable
            ):
                # In-place operators like += usually mustate tensor
                # values, but in the edge case of immutable values they
                # re-bind the variable.
                #
                # The easiest way to keep the graph consistent in this
                # scenario is to de-sugar eagerly.
                fn = IN_PLACE_DESUGARING_MAP[fn]
                args = [args[0], args[1]]  # type: ignore[assignment]

            if fn is operator.getitem:
                if kwargs:
                    raise_type_error(
                        tx, "_operator.getitem() takes no keyword arguments"
                    )
                if len(args) != 2:
                    raise_type_error(
                        tx, f"getitem expected 2 arguments, got {len(args)}"
                    )
                if isinstance(args[1], SymNodeVariable):
                    # Standard indexing will force specialization due to
                    # __index__.  Rewrite as a regular torch op which will
                    # trace fine
                    fn = torch.select
                    args = [
                        args[0],
                        variables.VariableTracker.build(tx, 0),
                        args[1],
                    ]  # type: ignore[assignment]

            # Interaction between ndarray and tensors:
            #   We prefer the tensor op whenever there are tensors involved
            # NB: Use exact type check here - NumpyNdarrayVariable is a TensorVariable
            # subclass but should NOT trigger the tensor path
            if check_numpy_ndarray_args(args, kwargs) and not any(
                type(arg) is TensorVariable for arg in args
            ):
                proxy = tx.output.create_proxy(
                    "call_function",
                    numpy_operator_wrapper(fn),
                    *proxy_args_kwargs(args, kwargs),
                )

                return wrap_fx_proxy_cls(variables.NumpyNdarrayVariable, tx, proxy)

            if (
                fn in _OPERATOR_TO_DUNDER
                and len(args) == 2
                and any(
                    not isinstance(a, (variables.TensorVariable, SymNodeVariable))
                    for a in args
                )
            ):
                return generic_richcompare(
                    tx, args[0], args[1], _OPERATOR_TO_DUNDER[fn]
                )

            proxy = tx.output.create_proxy(
                "call_function",
                fn,
                *proxy_args_kwargs(args, kwargs),
            )
            if any(isinstance(arg, FakeItemVariable) for arg in args):
                return wrap_fx_proxy_cls(
                    FakeItemVariable,
                    tx,
                    proxy,
                )
            elif check_unspec_python_args(args, kwargs):
                _args, _kwargs = self.unwrap_unspec_args_kwargs(args, kwargs)
                raw_value = fn(*_args, **_kwargs)

                need_unwrap = any(
                    x.need_unwrap
                    for x in itertools.chain(args, kwargs.values())
                    if isinstance(x, variables.UnspecializedPythonVariable)
                )

                return wrap_fx_proxy_cls(
                    UnspecializedPythonVariable,
                    tx,
                    proxy,
                    raw_value=raw_value,
                    need_unwrap=need_unwrap,
                )
            elif all(isinstance(x, SymNodeVariable) for x in args):
                return SymNodeVariable.create(tx, proxy, None)
            else:
                # Work around for vision_maskrcnn due to precision difference
                # specialize the dividend when float divide by tensor
                if fn is operator.truediv and isinstance(
                    args[0], variables.UnspecializedPythonVariable
                ):
                    args = list(args)
                    args[0] = args[0].as_python_constant()
                return wrap_fx_proxy(tx, proxy)

        except NotImplementedError:
            unimplemented(
                gb_type="unimplemented builtin op on tensor arguments",
                context=f"partial tensor op: {self} {args} {kwargs}",
                explanation=f"Dynamo does not know how to trace builtin operator {self.fn} with tensor arguments",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

    call_function_handler_cache: dict[
        tuple[object, ...],
        Callable[
            [
                "InstructionTranslatorBase",
                list[VariableTracker],
                dict[str, VariableTracker],
            ],
            VariableTracker,
        ],
    ] = {}

    def call_function(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if self.fn is object and not args and not kwargs:
            # object() -> a fresh opaque instance, wrapped as ObjectVariable to
            # match how SourcelessBuilder wraps bare `object` instances. Falling
            # through to the constant handler cannot build a VT from object().
            from .builder import SourcelessBuilder

            return SourcelessBuilder.create(tx, object())

        key: tuple[object, ...]
        if kwargs:
            kwargs = {k: v.realize() for k, v in kwargs.items()}
            key = (self.fn, *(type(x) for x in args), True)
        else:
            key = (self.fn, *(type(x) for x in args))

        handler = self.call_function_handler_cache.get(key)
        if not handler:
            self.call_function_handler_cache[key] = handler = self._make_handler(  # type: ignore[assignment]
                self.fn, [type(x) for x in args], bool(kwargs)
            )
        if handler is None:
            raise AssertionError(
                f"No handler found for {self.fn} with args {[type(x) for x in args]}"
            )
        return handler(tx, args, kwargs)  # type: ignore[return-value]

    def call_method(
        self,
        tx: "InstructionTranslatorBase",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if self.fn is object and name == "__setattr__":
            if len(args) != 3:
                raise AssertionError(
                    f"object.__setattr__ expects 3 args, got {len(args)}"
                )
            if len(kwargs) != 0:
                raise AssertionError(
                    f"object.__setattr__ expects no kwargs, got {len(kwargs)}"
                )
            obj, name_var, val = args
            obj = obj.realize()
            if (
                isinstance(obj, UserDefinedObjectVariable)
                and tx.output.side_effects.is_attribute_mutation(obj)
                and name_var.is_python_constant()
            ):
                return obj.method_setattr_standard(tx, name_var, val)

        if name == "__new__":
            # Supported __new__ methods
            if self.fn is object and len(args) == 1:
                if len(kwargs) != 0:
                    raise AssertionError(
                        f"object.__new__ expects no kwargs, got {len(kwargs)}"
                    )
                return tx.output.side_effects.track_new_user_defined_object(
                    self,
                    args[0],
                    args[1:],
                    tx=tx,
                )

            if self.fn is tuple and len(args) == 2 and not kwargs:
                if isinstance(args[0], BuiltinVariable) and args[0].fn is tuple:
                    init_args = unpack_iterable(tx, args[1])
                    return variables.TupleVariable(
                        init_args, mutation_type=ValueMutationNew()
                    )

                return tx.output.side_effects.track_new_user_defined_object(
                    self,
                    args[0],
                    args[1:],
                    tx=tx,
                )

        if name in _BUILTIN_CONSTANT_FOLDABLE_METHODS.get(self.fn, ()):
            if all(a.is_python_constant() for a in args) and all(
                v.is_python_constant() for v in kwargs.values()
            ):
                try:
                    fn = getattr(self.fn, name)
                    res = fn(
                        *(a.as_python_constant() for a in args),
                        **{k: v.as_python_constant() for k, v in kwargs.items()},
                    )
                    return VariableTracker.build(tx, res)
                except Exception as e:
                    raise_observed_exception(
                        type(e),
                        tx,
                        args=list(e.args),
                    )

        if self.fn is object and name == "__init__":
            # object.__init__ is a no-op
            return variables.ConstantVariable.create(None)

        if self.fn in (set, frozenset, list, tuple):
            if isinstance(args[0], variables.UserDefinedObjectVariable):
                if args[0]._base_vt is None:
                    raise AssertionError(
                        "UserDefinedObjectVariable._base_vt must not be None"
                    )
                return args[0]._base_vt.call_method(tx, name, args[1:], kwargs)
            else:
                return args[0].call_method(tx, name, args[1:], kwargs)

        if (
            name in ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__")
            and len(args) == 2
            and not kwargs
            and isinstance(self.fn, type)
            and args[0].is_python_constant()
            and args[1].is_python_constant()
        ):
            # Unbound rich-comparison dunder, e.g. complex.__eq__(1+1j, 2).
            # Invoke only the left type's slot on the constants, mirroring
            # CPython (returns NotImplemented rather than falling back to the
            # reflected operation when the operands are incompatible).
            lval = args[0].as_python_constant()
            if isinstance(lval, self.fn):
                return ConstantVariable.create(
                    getattr(self.fn, name)(lval, args[1].as_python_constant())
                )

        if self.fn is str and len(args) >= 1:
            resolved_fn = getattr(self.fn, name, None)
            if resolved_fn in str_methods:
                # Only delegate to ConstantVariable, not other types that happen to be constants
                if isinstance(args[0], ConstantVariable):
                    return args[0].call_method(tx, name, args[1:], kwargs)

        if self.fn is float and len(args) >= 1:
            # Only delegate to ConstantVariable, not other types that happen to be constants
            if isinstance(args[0], ConstantVariable):
                return VariableTracker.build(
                    tx, getattr(float, name)(args[0].as_python_constant())
                )

        if name == "__len__" and len(args) == 1 and not kwargs:
            # type.__len__(instance) → len(instance)
            # e.g. list.__len__(my_list) → len(my_list)
            return generic_size(tx, args[0])

        if name == "__str__" and len(args) == 1 and not kwargs:
            return super().call_method(tx, name, args, kwargs)

        if name == "__repr__" and len(args) == 1 and not kwargs:
            return super().call_method(tx, name, args, kwargs)

        if name == "__iter__" and len(args) == 1 and not kwargs:
            # type.__iter__(instance) → iter(instance)
            # e.g., tuple.__iter__(my_tuple) → iter(my_tuple)
            # For builtin types called on user-defined subclasses, use the base iterator
            return generic_getiter(tx, args[0])

        if name == "__neg__" and len(args) == 1 and not kwargs:
            # type.__neg__(instance) → neg(instance)
            # e.g., int.__neg__(4) → neg(4)
            return pynumber_negative(tx, args[0])

        if name == "__pos__" and len(args) == 1 and not kwargs:
            # type.__pos__(instance) → pos(instance)
            # e.g., int.__pos__(4) → pos(4)
            return pynumber_positive(tx, args[0])

        if name == "__abs__" and len(args) == 1 and not kwargs:
            # type.__abs__(instance) → abs(instance)
            # e.g., int.__abs__(-4) → abs(-4)
            return pynumber_absolute(tx, args[0])

        if name == "__invert__" and len(args) == 1 and not kwargs:
            # type.__invert__(instance) → ~instance
            # e.g., int.__invert__(4) → ~4
            return pynumber_invert(tx, args[0])

        if name == "__hash__" and len(args) == 1 and not kwargs:
            arg = args[0]
            if (
                isinstance(arg, variables.UserDefinedConstantVariable)
                and arg._base_vt is not None
            ):
                return generic_hash(tx, arg._base_vt)

        return super().call_method(tx, name, args, kwargs)

    def call_int(
        self,
        tx: "InstructionTranslatorBase",
        arg: VariableTracker,
        base: VariableTracker | None = None,
    ) -> VariableTracker | None:
        if base is None:
            return pynumber_int(tx, arg)
        return pylong_from_base(tx, arg, base)

    def call_float(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        return pynumber_float(tx, arg)

    def call_bin(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        return pynumber_tobase(tx, arg, 2)

    def call_oct(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        return pynumber_tobase(tx, arg, 8)

    def call_hex(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        return pynumber_tobase(tx, arg, 16)

    def call_bool(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        # Emulate PyBool_Type.tp_vectorcall which boils down to PyObject_IsTrue.
        return generic_is_true(tx, arg)

    def call_hash(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        return generic_hash(tx, arg)

    def call_repr(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        return generic_repr(tx, arg)

    def call_str(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        return generic_str(tx, arg)

    def call___build_class__(self, tx, *args, **kwargs):
        def fail(args, kwargs) -> NoReturn:
            unimplemented(
                gb_type="Invalid call to __build_class__",
                context=f"Non-constant args to __build_class__: {args} {kwargs}",
                explanation="Cannot trace class definition: the class body function is unsupported or the base class argument are not compile-time constants",
                hints=[*graph_break_hints.SUPPORTABLE],
            )

        if not torch._dynamo.config.enable_trace_load_build_class:
            fail(args, kwargs)

        try:
            if isinstance(args[0], variables.NestedUserFunctionVariable):
                fn = args[0].get_function(allow_sourced_cells=True)
            else:
                fn = args[0].get_function()
        except NotImplementedError:
            fail(args, kwargs)

        if check_constant_args(args[1:], kwargs):
            r = builtins.__build_class__(
                fn,  # type: ignore[possibly-undefined]
                *[a.as_python_constant() for a in args[1:]],
            )
            return VariableTracker.build(tx, r)
        else:
            fail(args, kwargs)

    def _call_min_max(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        key: VariableTracker | None = None,
        default: VariableTracker | None = None,
    ) -> VariableTracker | None:
        name = self.fn.__name__
        if len(args) == 0:
            raise_type_error(tx, f"{name} expected at least 1 argument, got 0")

        # `default` is keyword-only and only valid with a single iterable arg.
        positional = len(args) > 1
        if positional and default is not None:
            raise_type_error(
                tx,
                f"Cannot specify a default for {name}() with multiple positional arguments",
            )

        if key is not None and key.is_constant_none():
            key = None

        items = list(args) if positional else unpack_iterable(tx, args[0])

        if len(items) == 0:
            if default is not None:
                return default
            raise_value_error(tx, f"{name}() iterable argument is empty")

        if key is None:
            return self._call_min_max_seq(tx, items)
        return self._call_min_max_seq_with_key(tx, items, key)

    def _call_min_max_seq(
        self, tx: "InstructionTranslatorBase", items: list[VariableTracker]
    ) -> VariableTracker:
        if len(items) <= 0:
            raise AssertionError("_call_min_max_seq requires at least one item")
        if len(items) == 1:
            return items[0]

        return functools.reduce(functools.partial(self._call_min_max_binary, tx), items)  # type: ignore[arg-type,return-value]

    def _call_min_max_seq_with_key(
        self,
        tx: "InstructionTranslatorBase",
        items: list[VariableTracker],
        key: VariableTracker,
    ) -> VariableTracker:
        # Mirror CPython builtin_min_max: compare on key(item) but return the
        # original element, keeping the first extremal element on ties (only
        # replace on a strict comparison). Like list.sort, comparisons are done
        # through Dynamo and must fold to a compile-time constant, so Tensor and
        # SymInt keys graph break.
        keyvals = [key.call_function(tx, [item], {}) for item in items]
        best_item, best_key = items[0], keyvals[0]
        for item, keyval in zip(items[1:], keyvals[1:]):
            left, right = (best_key, keyval) if self.fn is max else (keyval, best_key)
            cmp = BuiltinVariable(operator.lt).call_function(tx, [left, right], {})
            if not cmp.is_python_constant():
                unimplemented(
                    gb_type="min/max with non-constant key",
                    context=str(keyval),
                    explanation=(
                        f"Cannot compute {self.fn.__name__}() whose key comparison is "
                        f"not a compile-time constant. Key type: {keyval.python_type()}. "
                        f"Most notably, Tensor or SymInt keys are unsupported, but ints work."
                    ),
                    hints=[
                        "Use something else as the key.",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )
            if cmp.as_python_constant():
                best_item, best_key = item, keyval
        return best_item

    def _call_min_max_binary(
        self,
        tx: "InstructionTranslatorBase",
        a: VariableTracker | None,
        b: VariableTracker | None,
    ) -> VariableTracker | None:
        if a is None or b is None:
            # a or b could be none if we reduce and _call_min_max_binary failed
            # to return something
            return None
        if self.tensor_args(a, b):
            if not a.is_tensor():
                a, b = b, a
            if not a.is_tensor():
                raise AssertionError(
                    "Expected at least one tensor argument for min/max"
                )

            # result of an item call is a scalar convert to a tensor
            if isinstance(a, FakeItemVariable):
                a = variables.TorchInGraphFunctionVariable(torch.tensor).call_function(
                    tx, [a], {}
                )

            # Dynamic input does not get resolved, rather, gets stored as call_function
            if isinstance(a, SymNodeVariable) or isinstance(b, SymNodeVariable):
                from .builder import wrap_fx_proxy_cls

                return wrap_fx_proxy_cls(
                    type(a),
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        self.fn,
                        *proxy_args_kwargs([a, b], {}),
                    ),
                )

            # convert min/max to torch ops
            if b.is_python_constant():
                fn: VariableTracker
                if isinstance(a, variables.NumpyNdarrayVariable):
                    import numpy as np

                    fn = variables.NumpyVariable(np.clip)
                else:
                    fn = variables.TorchInGraphFunctionVariable(torch.clamp)
                kwargs = {"min": b} if (self.fn is max) else {"max": b}
                result = fn.call_function(tx, [a], kwargs)
            else:
                if isinstance(a, variables.NumpyNdarrayVariable):
                    import numpy as np

                    np_fn = {max: np.maximum, min: np.minimum}[self.fn]
                    fn = variables.NumpyVariable(np_fn)
                else:
                    torch_fn = {max: torch.maximum, min: torch.minimum}[self.fn]
                    fn = variables.TorchInGraphFunctionVariable(torch_fn)
                result = fn.call_function(tx, [a, b], {})

            # return unspec if both a, b are unspec or const
            if all(
                isinstance(
                    i,
                    (
                        variables.UnspecializedPythonVariable,
                        variables.ConstantVariable,
                    ),
                )
                for i in [a, b]
            ):
                if any(isinstance(val, FakeItemVariable) for val in [a, b]):
                    # type: ignore[arg-type]
                    return variables.FakeItemVariable.from_tensor_variable(result)

                if b.is_python_constant():
                    raw_b = b.as_python_constant()
                else:
                    raw_b = b.raw_value  # type: ignore[attr-defined]
                if self.fn is max:
                    raw_res = max(a.raw_value, raw_b)  # type: ignore[attr-defined]
                else:
                    raw_res = min(a.raw_value, raw_b)  # type: ignore[attr-defined]

                need_unwrap = any(
                    x.need_unwrap
                    for x in [a, b]
                    if isinstance(x, variables.UnspecializedPythonVariable)
                )
                return variables.UnspecializedPythonVariable.from_tensor_variable(
                    result,  # type: ignore[arg-type]
                    raw_res,
                    need_unwrap,
                )
            # otherwise return tensor
            else:
                return result
        elif isinstance(a, SymNodeVariable) or isinstance(b, SymNodeVariable):
            py_fn = torch.sym_max if self.fn is max else torch.sym_min
            proxy = tx.output.create_proxy(
                "call_function", py_fn, *proxy_args_kwargs([a, b], {})
            )
            return SymNodeVariable.create(tx, proxy, None)
        elif isinstance(a, ConstantVariable) and isinstance(b, ConstantVariable):
            value = self.fn(
                a.as_python_constant(),
                b.as_python_constant(),
            )
            return VariableTracker.build(tx, value)
        return None

    call_min = _call_min_max
    call_max = _call_min_max

    def call_abs(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        return pynumber_absolute(tx, arg)

    def call_pos(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        return pynumber_positive(tx, arg)

    def call_index(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        # Specialize SymNodeVariable to a constant first, matching CPython's
        # PyNumber_Index which forces a concrete int.
        return pynumber_index(tx, arg)

    def call_round(
        self,
        tx: "InstructionTranslatorBase",
        arg: VariableTracker,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        from .builder import SourcelessBuilder

        # Call arg.__round__()
        round_method = SourcelessBuilder.create(tx, getattr).call_function(
            tx, [arg, VariableTracker.build(tx, "__round__")], {}
        )
        return round_method.call_function(tx, list(args), kwargs)

    def call_range(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        if kwargs:
            raise_type_error(tx, "range() takes no keyword arguments")
        if len(args) == 0:
            raise_type_error(tx, "range expected at least 1 argument, got 0")
        if len(args) > 3:
            raise_type_error(tx, f"range expected at most 3 arguments, got {len(args)}")
        args = tuple(pynumber_index(tx, arg) for arg in args)
        return variables.RangeVariable(list(args))

    def _dynamic_args(self, *args: VariableTracker, **kwargs: VariableTracker) -> bool:
        return any(isinstance(x, SymNodeVariable) for x in args) or any(
            isinstance(x, SymNodeVariable) for x in kwargs.values()
        )

    def call_slice(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        if not 1 <= len(args) < 4:
            raise_type_error(tx, f"slice expected at least 1 argument, got {len(args)}")
        return variables.SliceVariable(list(args), tx)

    def _dyn_proxy(
        self, tx: "InstructionTranslatorBase", *args: Any, **kwargs: Any
    ) -> VariableTracker:
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_function", self.fn, *proxy_args_kwargs(args, kwargs)
            ),
        )

    def call_tuple(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker | None:
        # ref: https://github.com/python/cpython/blob/main/Objects/abstract.c#L2004-L2078
        if kwargs:
            raise_type_error(
                tx,
                f"{self.fn.__name__}() takes no keyword arguments",
            )
        if len(args) == 0:
            return TupleVariable([], mutation_type=ValueMutationNew())
        elif len(args) > 1:
            raise_type_error(
                tx,
                f"{self.fn.__name__} expected at most 1 argument, got {len(args)}",
            )

        obj = args[0]
        if isinstance(obj, TupleVariable) and obj.python_type() is tuple:
            return obj

        items = unpack_iterable(tx, args[0])
        return TupleVariable(items, mutation_type=ValueMutationNew())

    def call_callable(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker:
        # PyCallable_Check: callable(x) is type(x)->tp_call != NULL. Dispatch on
        # the arg's Python type slot rather than enumerating callable VTs.
        # callable() is the only handler for this builtin, so always return a
        # result (or graph break via maybe_get_python_type) -- never None, which
        # would signal "try another handler" that does not exist.
        return variables.ConstantVariable.create(
            pycallable_check(maybe_get_python_type(arg))
        )

    def call_cast(
        self, _: Any, *args: VariableTracker, **kwargs: VariableTracker
    ) -> VariableTracker | None:
        if len(args) == 2:
            return args[1]

        unimplemented(
            gb_type="bad args to builtin cast()",
            context=f"got args {args} {kwargs}",
            explanation="Dynamo expects exactly 2 args to builtin cast().",
            hints=["Ensure your call to cast() has exactly 2 arguments."],
        )

    def call_dir(
        self, tx: "InstructionTranslatorBase", arg: VariableTracker
    ) -> VariableTracker | None:
        if isinstance(arg, variables.UserDefinedClassVariable):
            return VariableTracker.build(tx, dir(arg.value))
        if isinstance(arg, BuiltinVariable):
            return VariableTracker.build(tx, dir(arg.fn))
        # Enable specialized VTs for constants to work with dir()
        if arg.is_python_constant():
            return VariableTracker.build(tx, dir(arg.as_python_constant()))
        return None

    def call_set(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/main/Objects/setobject.c#L2708-L2735
        # CPython set_init rejects keywords before unpacking positional args, so
        # set(a=1) and set().__init__(a=1) both raise regardless of arg count.
        if kwargs:
            raise_type_error(
                tx,
                "set() takes no keyword arguments",
            )
        if len(args) == 0:
            return variables.SetVariable(set(), mutation_type=ValueMutationNew())
        elif len(args) > 1:
            raise_type_error(
                tx,
                f"set expected at most 1 argument, got {len(args)}",
            )

        s = SetVariable([], mutation_type=ValueMutationNew())
        s.call_method(tx, "update", [args[0]], kwargs)
        return s

    def call_frozenset(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        if kwargs:
            raise_type_error(
                tx,
                "frozenset() takes no keyword arguments",
            )
        if len(args) == 0:
            return variables.FrozensetVariable(set(), mutation_type=ValueMutationNew())
        elif len(args) > 1:
            raise_type_error(
                tx,
                f"frozenset expected at most 1 argument, got {len(args)}",
            )

        if istype(args[0], variables.FrozensetVariable):
            # CPython: frozenset(existing_frozenset) returns the same object.
            return args[0]

        # Reuse existing HashableTracker keys from a set/frozenset/dict operand
        # instead of re-hashing, mirroring CPython's set_update_internal fast
        # path (do-not-rehash-dict-keys).
        if isinstance(args[0], (variables.SetVariable, variables.ConstDictVariable)):
            items = list(args[0].items.keys())
        else:
            items = unpack_iterable(tx, args[0])
        fs = FrozensetVariable(items, mutation_type=ValueMutationNew())
        return fs

    def call_zip(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # ref: https://github.com/python/cpython/blob/v3.13.3/Python/bltinmodule.c#L2822-L2887
        if kwargs:
            if not (len(kwargs) == 1 and "strict" in kwargs):
                raise_args_mismatch(
                    tx,
                    "zip",
                    "1 kwargs (`strict`)",
                    f"{len(kwargs)} kwargs",
                )
        strict = kwargs.pop("strict", ConstantVariable.create(False))
        items = []
        for arg in args:
            items.append(generic_getiter(tx, arg))
        iter_args = TupleVariable(items, mutation_type=ValueMutationNew())
        return variables.ZipVariable(
            iter_args,
            strict=strict.as_python_constant(),
            mutation_type=ValueMutationNew(),
        )

    def call_len(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        if kwargs:
            raise_type_error(tx, "len() takes no keyword arguments")
        if len(args) != 1:
            raise_type_error(
                tx, f"len() takes exactly one argument ({len(args)} given)"
            )
        return generic_size(tx, args[0])

    def call_length_hint(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        # ref: PyObject_LengthHint (Objects/abstract.c): try __len__, then
        # __length_hint__, falling back to the supplied default for either a
        # missing slot or a TypeError raised by the slot.
        if kwargs or not (1 <= len(args) <= 2):
            raise_type_error(
                tx, f"length_hint expected 1 or 2 arguments, got {len(args)}"
            )
        obj = args[0]
        default = args[1] if len(args) == 2 else ConstantVariable.create(0)

        obj_type = maybe_get_python_type(obj)

        if type_implements_sq_length(obj_type) or type_implements_mp_length(obj_type):
            return generic_size(tx, obj)

        if getattr(obj_type, "__length_hint__", None) is None:
            return default
        try:
            return obj.call_method(tx, "__length_hint__", [], {})
        except ObservedTypeError:
            handle_observed_exception(tx)
            return default

    def call_getitem(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        if kwargs:
            raise_type_error(tx, "_operator.getitem() takes no keyword arguments")
        if len(args) != 2:
            raise_type_error(tx, f"getitem expected 2 arguments, got {len(args)}")
        return generic_getitem(tx, args[0], args[1])

    def call_setitem(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        if kwargs:
            raise_type_error(tx, "_operator.setitem() takes no keyword arguments")
        if len(args) != 3:
            raise_type_error(tx, f"setitem expected 3 arguments, got {len(args)}")
        return generic_setitem(tx, args[0], args[1], args[2])

    def call_delitem(
        self,
        tx: "InstructionTranslatorBase",
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        if kwargs:
            raise_type_error(tx, "_operator.delitem() takes no keyword arguments")
        if len(args) != 2:
            raise_type_error(tx, f"delitem expected 2 arguments, got {len(args)}")
        return generic_delitem(tx, args[0], args[1])

    def call_isinstance(
        self,
        tx: "InstructionTranslatorBase",
        arg: VariableTracker,
        isinstance_type_var: VariableTracker,
    ) -> VariableTracker:
        try:
            arg_type = arg.python_type()
        except NotImplementedError:
            unimplemented(
                gb_type="builtin isinstance() cannot determine type of argument",
                context=f"isinstance({arg}, {isinstance_type_var})",
                explanation=f"Dynamo doesn't have a rule to determine the type of argument {arg}",
                hints=[*graph_break_hints.DYNAMO_BUG],
            )
        isinstance_type = isinstance_type_var.as_python_constant()
        # An AsyncCollectiveTensor (ACT) input's tensor-class guard is relaxed
        # (see VariableBuilder.wrap_tensor): the cache entry does not discriminate
        # ACT from the resolved plain Tensor. isinstance() observes the class and
        # is constant-folded below, so reinstall the class guard -- but only when
        # the result would actually differ between the ACT and the resolved
        # Tensor. isinstance(w, ACT) discriminates and must recompile;
        # isinstance(w, torch.Tensor) is True for both and must stay reusable.
        if isinstance(arg, variables.TensorVariable) and arg.source is not None:
            from torch._functorch._aot_autograd.utils import (
                is_async_collective_tensor_type,
            )

            if is_async_collective_tensor_type(arg_type):
                check_types = (
                    isinstance_type
                    if isinstance(isinstance_type, tuple)
                    else (isinstance_type,)
                )
                # Guard if any checked type distinguishes ACT from torch.Tensor,
                # or conservatively if a checked entry is not a plain class.
                if any(
                    not isinstance(ty, type)
                    or issubclass(arg_type, ty) != issubclass(torch.Tensor, ty)
                    for ty in check_types
                ):
                    install_guard(arg.source.make_guard(GuardBuilder.TYPE_MATCH))
        if isinstance(arg, variables.TensorVariable) and arg.dtype is not None:
            if _uses_custom_classinfo_check(isinstance_type):
                unimplemented(
                    gb_type="builtin isinstance() with custom type check on tensor",
                    context=f"isinstance({arg}, {isinstance_type})",
                    explanation="Dynamo cannot soundly trace arbitrary custom "
                    "__instancecheck__ or __subclasscheck__ hooks on tensor "
                    "values because the hook may read external mutable state.",
                    hints=[*graph_break_hints.SUPPORTABLE],
                )

            def _tensor_isinstance(
                tensor_var: VariableTracker, tensor_type: Any
            ) -> bool:
                def check_type(ty: Any) -> bool:
                    if ty not in tensortype_to_dtype:
                        example_val = arg.as_proxy().node.meta["example_value"]
                        if (
                            is_traceable_wrapper_subclass(example_val)
                            and ty is torch.nn.parameter.Parameter
                        ):
                            # N.B: we are calling isinstance directly on the example value.
                            # torch.nn.Parameter has a meta-class that overrides __isinstance__,
                            # the isinstance check here allows us to invoke that logic.
                            return isinstance(example_val, ty)
                        else:
                            return issubclass(arg.python_type(), ty)

                    dtypes = tensortype_to_dtype[ty]
                    # pyrefly: ignore [missing-attribute]
                    return arg.dtype in dtypes

                if type(tensor_type) is tuple:
                    return any(check_type(ty) for ty in tensor_type)
                else:
                    return check_type(tensor_type)

            return VariableTracker.build(tx, _tensor_isinstance(arg, isinstance_type))
        # UserDefinedObject with C extensions can have torch.Tensor attributes,
        # so break graph.
        if isinstance(arg, variables.UserDefinedObjectVariable) and isinstance(
            arg.value, types.MemberDescriptorType
        ):
            unimplemented(
                gb_type="isinstance() called on user defined object with C extensions",
                context=f"isinstance({arg}, {isinstance_type})",
                explanation="User-defined object with C extensions can have torch.Tensor "
                "attributes; intentionally graph breaking.",
                hints=[*graph_break_hints.SUPPORTABLE],
            )
        # handle __instancecheck__ defined in user class
        if (
            isinstance(arg, variables.UserDefinedObjectVariable)
            and "__instancecheck__" in isinstance_type.__class__.__dict__
        ):
            return VariableTracker.build(
                tx,
                isinstance_type.__class__.__instancecheck__(isinstance_type, arg.value),
            )

        if isinstance(arg, variables.UserDefinedExceptionClassVariable):
            # pyrefly: ignore [unbound-name]
            return VariableTracker.build(tx, isinstance(arg_type, isinstance_type))

        isinstance_type_tuple: tuple[type, ...]
        if isinstance(isinstance_type, type) or callable(
            # E.g. isinstance(obj, typing.Sequence)
            getattr(isinstance_type, "__instancecheck__", None)
        ):
            isinstance_type_tuple = (isinstance_type,)
        elif isinstance(isinstance_type, types.UnionType):
            isinstance_type_tuple = typing.get_args(isinstance_type)
        elif isinstance(isinstance_type, tuple) and all(
            isinstance(tp, type) or callable(getattr(tp, "__instancecheck__", None))
            for tp in isinstance_type
        ):
            isinstance_type_tuple = isinstance_type
        else:
            raise_observed_exception(
                TypeError,
                tx,
                args=[
                    "isinstance() arg 2 must be a type, a tuple of types, or a union"
                ],
            )

        try:
            # NB: `isinstance()` does not call `__subclasscheck__` but use `__instancecheck__`.
            # But usually `isinstance(obj, type_info)` and `issubclass(type(obj), type_info)` gives
            # the same result.
            # WARNING: This might run arbitrary user code `__subclasscheck__` and we did not trace
            # through it. This is a limitation of the current implementation.
            # Usually `__subclasscheck__` and `__instancecheck__` can be constant fold through, it
