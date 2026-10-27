                    tuple(self.deserialize_sym_int(val) for val in tensor_meta.strides),  # type: ignore[misc]
                    device=deserialize_device(tensor_meta.device),
                    dtype=_SERIALIZE_TO_TORCH_DTYPE[tensor_meta.dtype],
                    requires_grad=tensor_meta.requires_grad,
                ),
            )

    def deserialize_script_obj_meta(
        self, script_obj_meta: CustomObjArgument
    ) -> ep.CustomObjArgument:
        return ep.CustomObjArgument(
            name=script_obj_meta.name,
            class_fqn=script_obj_meta.class_fqn,
        )

    def deserialize_graph_output(self, output) -> torch.fx.Node | int | None:
        if output.type == "as_tensor":
            return self.serialized_name_to_node[output.as_tensor.name]
        elif output.type == "as_sym_int":
            return self.serialized_name_to_node[output.as_sym_int.as_name]
        elif output.type == "as_sym_bool":
            return self.serialized_name_to_node[output.as_sym_bool.as_name]
        elif output.type == "as_sym_float":
            return self.serialized_name_to_node[output.as_sym_float.as_name]
        elif output.type == "as_int":
            return output.as_int
        elif output.type == "as_float":
            return output.as_float
        elif output.type == "as_bool":
            return output.as_bool
        elif output.type == "as_none":
            return None
        else:
            raise SerializeError(f"Unable to deserialize output node {output}")

    def deserialize_graph(self, serialized_graph: Graph) -> torch.fx.Graph:
        log.debug("\n[deserialize_graph]")

        # Handle the tensor metas.
        for name, tensor_value in serialized_graph.tensor_values.items():
            log.debug("[deserialize_tensor_meta] %s (input): %s", name, tensor_value)
            self.serialized_name_to_meta[name] = (
                lambda v=tensor_value: self.deserialize_tensor_meta(v)
            )

        for name, sym_int_value in serialized_graph.sym_int_values.items():
            log.debug("[deserialize_sym_int] %s (input): %s", name, sym_int_value)
            self.serialized_name_to_meta[name] = (
                lambda v=sym_int_value: self.deserialize_sym_int(v)
            )

        for name, sym_float_value in serialized_graph.sym_float_values.items():
            log.debug("[deserialize_sym_float] %s (input): %s", name, sym_float_value)
            self.serialized_name_to_meta[name] = (
                lambda v=sym_float_value: self.deserialize_sym_float(v)
            )

        for name, sym_bool_value in serialized_graph.sym_bool_values.items():
            log.debug("[deserialize_sym_bool] %s (input): %s", name, sym_bool_value)
            self.serialized_name_to_meta[name] = (
                lambda v=sym_bool_value: self.deserialize_sym_bool(v)
            )

        for name, script_obj_meta in serialized_graph.custom_obj_values.items():
            log.debug("[deserialize_script_obj_meta] %s", script_obj_meta)
            self.serialized_name_to_meta[name] = (
                lambda v=script_obj_meta: self.deserialize_script_obj_meta(v)
            )

        log.debug("\n[deserialize graph nodes]")
        # Inputs: convert to placeholder nodes in FX.
        for i, input_ in enumerate(serialized_graph.inputs):
            log.debug("[deserialize input] %s", input_)
            if input_.type in ("as_tensor", "as_custom_obj"):
                node_name = input_.value.name
                placeholder_node = self.graph.placeholder(node_name)
                # FX might declare a name illegal (e.g. some nn.Modules use "input" as forward() arguments)
                # we will overwrite it
                placeholder_node.name = node_name
                self.sync_fx_node(node_name, placeholder_node)
            elif input_.type == "as_sym_int":
                if input_.value.type == "as_name":
                    node_name = input_.value.as_name
                    placeholder_node = self.graph.placeholder(node_name)
                    # FX might declare a name illegal (e.g. some nn.Modules use "input" as forward() arguments)
                    # we will overwrite it
                    placeholder_node.name = node_name
                    self.sync_fx_node(node_name, placeholder_node)
                else:
                    raise SerializeError(
                        f"Deserializing a constant symint {input_.value} as an input"
                    )
            elif input_.type in (
                "as_int",
                "as_float",
                "as_bool",
                "as_none",
                "as_string",
            ):
                node_name = self.signature.input_specs[i].arg.name or f"arg{i}"
                placeholder_node = self.graph.placeholder(node_name)
                placeholder_node.meta["val"] = self.deserialize_input(input_)
            else:
                raise SerializeError(f"Invalid input type {input_}")

        # Nodes: convert to call_function nodes.
        for serialized_node in serialized_graph.nodes:
            try:
                target = self.deserialize_operator(serialized_node.target)
                self.deserialize_node(serialized_node, target)

            except Exception as e:
                raise SerializeError(
                    f"Failed deserializing node {serialized_node}\n Original exception {traceback.format_exc()}"
                ) from e

        # Outputs: convert to a single `output` node.
        outputs = []
        for output in serialized_graph.outputs:
            log.debug("[deserialize output] %s", output)
            outputs.append(self.deserialize_graph_output(output))

        if serialized_graph.is_single_tensor_return:
            if len(outputs) != 1:
                raise AssertionError(
                    f"expected single output for single_tensor_return, got {len(outputs)}"
                )
            outputs = outputs[0]  # type: ignore[assignment]
        else:
            outputs = tuple(outputs)  # type: ignore[assignment]

        output_node = self.graph.output(outputs)

        if serialized_graph.is_single_tensor_return:
            first_arg = output_node.args[0]
            if not isinstance(first_arg, torch.fx.Node):
                raise AssertionError(
                    f"Expected Node for single tensor return, got {type(first_arg)}"
                )
            output_node.meta["val"] = first_arg.meta["val"]
        else:
            first_arg = output_node.args[0]
            if not isinstance(first_arg, (tuple, list)):
                raise AssertionError(
                    f"Expected tuple/list for multi tensor return, got {type(first_arg)}"
                )
            output_node.meta["val"] = tuple(
                arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg
                for arg in first_arg
            )

        # recompute unbacked bindings
        for node in self.graph.nodes:
            if (val := node.meta.get("val")) is not None and (
                unbacked_bindings := symbolic_shapes._free_unbacked_symbols_with_path(
                    val,
                    (),
                    shape_env=self.shape_env,
                    pending=self.unbacked_symbols,
                    simplify=True,
                )
            ):
                node.meta["unbacked_bindings"] = unbacked_bindings

        if len(self.unbacked_symbols) != 0:
            raise AssertionError(
                f"expected no unbacked symbols, got {len(self.unbacked_symbols)}: {self.unbacked_symbols}"
            )
        return self.graph

    def deserialize_node(self, serialized_node: Node, target: Callable) -> None:
        def _is_single_tensor_return(target) -> bool:
            schema = _get_schema_from_target(target)
            returns = schema.returns
            return len(returns) == 1 and isinstance(
                returns[0].real_type, torch.TensorType
            )

        if (
            target in _SYM_OPS
            or target
            == torch.ops.aten.item.default  # this can produce either SymInt or SymBool
        ):
            # BC: use serialized_node.name if available, otherwise fallback to original logic
            name = (
                serialized_node.name
                if serialized_node.name
                else serialized_node.outputs[0].value.as_name
            )
            args = self.deserialize_sym_op_inputs(serialized_node.inputs)

            fx_node = self.graph.create_node("call_function", target, args, {}, name)
            self.deserialize_sym_op_outputs(serialized_node, fx_node)
        elif (
            target
            is torch._higher_order_ops.triton_kernel_wrap.triton_kernel_wrapper_functional
        ):
            raise SerializeError(
                "deserialize nyi for torch._higher_order_ops.triton_kernel_wrap.triton_kernel_wrapper_functional"
            )
        elif isinstance(target, torch._ops.HigherOrderOperator):
            args, kwargs = self.deserialize_hoo_inputs(serialized_node.inputs)
            metadata = self.deserialize_metadata(serialized_node.metadata)
            for x in (*args, *kwargs.values()):
                if isinstance(x, torch.fx.Node) and x.op == "get_attr":
                    # this means that we have deserialized a graph argument, but
                    # unfortunately the schema for it does not include metadata;
                    # so we reuse the metadata of the HOP call for such arguments
                    x.meta.update(metadata)
            # If a serialized HOP node has a length=1 outputs of type `as_tensor``.
            # There could be two cases:
            # (1) The HOP node returns a single tensor
            # (2) The HOP node returns a tuple containing a single tensor
            # We distinguish (1) and (2) by the `is_single_tensor_return`
            # field in the schema of Node
            # For BC, getattr() will return True if `is_single_tensor_return` doesn't
            # exist. This is because prior to adding `is_single_tensor_return`,
            # only (1) could happen as we handle (2) with type `as_tensors`
            # BC: use serialized_node.name if available, otherwise fallback to original logic
            if serialized_node.name:
                name = serialized_node.name
            else:
                name = (
                    serialized_node.outputs[0].as_tensor.name
                    if len(serialized_node.outputs) == 1
                    and hasattr(serialized_node.outputs[0], "as_tensor")
                    and getattr(serialized_node, "is_hop_single_tensor_return", True)
                    else None
                )
            fx_node = self.graph.create_node(
                "call_function", target, args, kwargs, name
            )
            self.deserialize_outputs(serialized_node, fx_node)
            fx_node.meta.update(metadata)

        elif isinstance(
            target, (torch._ops.OpOverload, *_registered_extension_types())
        ):
            # For convenience: if this node returns a single tensor, name the
            # newly-created node after it. This ensures that these tensor values
            # have names that are consistent with serialized.
            # BC: use serialized_node.name if available, otherwise fallback to original logic
            if serialized_node.name:
                name = serialized_node.name
            else:
                name = (
                    serialized_node.outputs[0].as_tensor.name
                    if _is_single_tensor_return(target)
                    else None  # FX will generate a name for us.
                )

            args, kwargs = self.deserialize_inputs(target, serialized_node)
            fx_node = self.graph.create_node(
                "call_function", target, args, kwargs, name
            )
            self.deserialize_outputs(serialized_node, fx_node)
        elif callable(target):
            # Handle predispatch wrapper functions (vmap, JVP, etc.)
            args, kwargs = self.deserialize_hoo_inputs(serialized_node.inputs)
            name = serialized_node.name if serialized_node.name else None
            fx_node = self.graph.create_node(
                "call_function", target, args, kwargs, name
            )
            self.deserialize_outputs(serialized_node, fx_node)
        else:
            _additional_msg = (
                (
                    f"We failed to resolve {target} to an operator. "
                    + "If it's a custom op/custom triton op, this is usually because the custom op is not registered"
                    + " when deserializing. Please import the custom op to register it before deserializing."
                    + " Otherwise, please file an issue on github."
                )
                if isinstance(target, str)
                else ""
            )
            raise SerializeError(
                _additional_msg
                + f" Unsupported target type for node {serialized_node}: {type(target)}."
            )

        fx_node.meta.update(self.deserialize_metadata(serialized_node.metadata))
        log.debug(
            "[deserialize_node] %s: %s(%s, {%s}) -> %s",
            fx_node.name,
            fx_node.target,
            fx_node.args,
            fx_node.kwargs,
            fx_node.meta.get("val"),
        )

        # handle ShapeEnv asserts
        if target is torch.ops.aten._assert_scalar.default:
            if not isinstance((arg := fx_node.args[0]), bool):
                expr = arg.meta["val"]  # type: ignore[union-attr]
                if isinstance(expr, torch.SymBool):
                    self.shape_env.guard_or_defer_runtime_assert(
                        expr.node.expr, "", fx_node
                    )
        elif target is torch.ops.aten.sym_constrain_range_for_size.default:
            sym = fx_node.args[0].meta["val"]  # type: ignore[union-attr]
            if isinstance(sym, torch.SymInt):
                self.shape_env._constrain_range_for_size(sym.node.expr)

        # handle nn_module_stack; serialization throws away empty dicts
        if (
            fx_node.op not in ["placeholder", "output"]
            and "nn_module_stack" not in fx_node.meta
        ):
            fx_node.meta["nn_module_stack"] = {}

    def deserialize_input_spec(self, i: InputSpec) -> ep.InputSpec:
        log.debug("[deserialize_input_spec] %s", i)
        if i.type == "user_input":
            return ep.InputSpec(
                kind=ep.InputKind.USER_INPUT,
                arg=self.deserialize_argument_spec(i.user_input.arg),
                target=None,
            )
        elif i.type == "parameter":
            return ep.InputSpec(
                kind=ep.InputKind.PARAMETER,
                arg=ep.TensorArgument(name=i.parameter.arg.name),
                target=i.parameter.parameter_name,
            )
        elif i.type == "buffer":
            return ep.InputSpec(
                kind=ep.InputKind.BUFFER,
                arg=ep.TensorArgument(name=i.buffer.arg.name),
                target=i.buffer.buffer_name,
                persistent=i.buffer.persistent,
            )
        elif i.type == "tensor_constant":
            return ep.InputSpec(
                kind=ep.InputKind.CONSTANT_TENSOR,
                arg=ep.TensorArgument(name=i.tensor_constant.arg.name),
                target=i.tensor_constant.tensor_constant_name,
            )
        elif i.type == "custom_obj":
            return ep.InputSpec(
                kind=ep.InputKind.CUSTOM_OBJ,
                arg=ep.CustomObjArgument(
                    name=i.custom_obj.arg.name, class_fqn=i.custom_obj.arg.class_fqn
                ),
                target=i.custom_obj.custom_obj_name,
            )
        elif i.type == "token":
            return ep.InputSpec(
                kind=ep.InputKind.TOKEN,
                arg=ep.TokenArgument(name=i.token.arg.name),
                target=None,
            )
        elif i.type == "constant_input":
            return ep.InputSpec(
                kind=ep.InputKind.USER_INPUT,
                arg=ep.ConstantArgument(
                    name=i.constant_input.name,
                    value=self.deserialize_constant_input(i.constant_input.value),
                ),
                target=None,
            )
        else:
            raise AssertionError(f"Unknown input spec {i}")

    def deserialize_output_spec(self, o: OutputSpec) -> ep.OutputSpec:
        log.debug("[deserialize_output_spec] %s", o)
        if o.type == "user_output":
            return ep.OutputSpec(
                kind=ep.OutputKind.USER_OUTPUT,
                arg=self.deserialize_argument_spec(o.user_output.arg),
                target=None,
            )
        elif o.type == "loss_output":
            return ep.OutputSpec(
                kind=ep.OutputKind.LOSS_OUTPUT,
                arg=ep.TensorArgument(name=o.loss_output.arg.name),
                target=None,
            )
        elif o.type == "buffer_mutation":
            return ep.OutputSpec(
                kind=ep.OutputKind.BUFFER_MUTATION,
                arg=ep.TensorArgument(name=o.buffer_mutation.arg.name),
                target=o.buffer_mutation.buffer_name,
            )
        elif o.type == "parameter_mutation":
            return ep.OutputSpec(
                kind=ep.OutputKind.PARAMETER_MUTATION,
                arg=ep.TensorArgument(name=o.parameter_mutation.arg.name),
                target=o.parameter_mutation.parameter_name,
            )
        elif o.type == "gradient_to_parameter":
            return ep.OutputSpec(
                kind=ep.OutputKind.GRADIENT_TO_PARAMETER,
                arg=ep.TensorArgument(name=o.gradient_to_parameter.arg.name),
                target=o.gradient_to_parameter.parameter_name,
            )
        elif o.type == "gradient_to_user_input":
            return ep.OutputSpec(
                kind=ep.OutputKind.GRADIENT_TO_USER_INPUT,
                arg=ep.TensorArgument(name=o.gradient_to_user_input.arg.name),
                target=o.gradient_to_user_input.user_input_name,
            )
        elif o.type == "user_input_mutation":
            return ep.OutputSpec(
                kind=ep.OutputKind.USER_INPUT_MUTATION,
                arg=ep.TensorArgument(name=o.user_input_mutation.arg.name),
                target=o.user_input_mutation.user_input_name,
            )
        elif o.type == "token":
            return ep.OutputSpec(
                kind=ep.OutputKind.TOKEN,
                arg=ep.TokenArgument(name=o.token.arg.name),
                target=None,
            )
        else:
            raise AssertionError(f"Unknown output spec {o}")

    def deserialize_signature(self, sig: GraphSignature) -> ep.ExportGraphSignature:
        log.debug("\n[deserialize_signature]")
        return ep.ExportGraphSignature(
            input_specs=[self.deserialize_input_spec(i) for i in sig.input_specs],
            output_specs=[self.deserialize_output_spec(o) for o in sig.output_specs],
        )

    def deserialize(
        self,
        serialized_graph_module: GraphModule,
        serialized_state_dict: dict[str, torch.Tensor] | bytes,
        constants: dict[str, Any] | bytes,
        example_inputs: tuple[tuple[torch.Tensor, ...], dict[str, Any]]
        | bytes
        | None = None,
        symbol_name_to_range: dict[str, symbolic_shapes.ValueRanges] | None = None,
    ) -> Result:
        global _CURRENT_DESERIALIZER
        if _CURRENT_DESERIALIZER is not None:
            raise AssertionError("_CURRENT_DESERIALIZER is already set")
        _CURRENT_DESERIALIZER = self
        try:
            log.debug("\n[deserialize]")
            self.shape_env = symbolic_shapes.ShapeEnv(assume_static_by_default=True)
            self.fake_tensor_mode = FakeTensorMode(
                allow_fallback_kernels=False,
                allow_non_fake_inputs=True,
                shape_env=self.shape_env,
            )
            self.sympy_functions = {
                # all torch.utils._sympy.functions should go here
                # TODO(avik): find a better way to keep this collection in sync;
                # e.g.., `exec('from torch.utils._sympy.functions import *', ...)`
                # would work as long as the public API of that module is complete
                "FloorDiv": torch.utils._sympy.functions.FloorDiv,
                "ModularIndexing": torch.utils._sympy.functions.ModularIndexing,
                "Where": torch.utils._sympy.functions.Where,
                "PythonMod": torch.utils._sympy.functions.PythonMod,
                "Mod": torch.utils._sympy.functions.Mod,
                "CleanDiv": torch.utils._sympy.functions.CleanDiv,
                "CeilToInt": torch.utils._sympy.functions.CeilToInt,
                "FloorToInt": torch.utils._sympy.functions.FloorToInt,
                "CeilDiv": torch.utils._sympy.functions.CeilDiv,
                "LShift": torch.utils._sympy.functions.LShift,
                "RShift": torch.utils._sympy.functions.RShift,
                "PowByNatural": torch.utils._sympy.functions.PowByNatural,
                "FloatPow": torch.utils._sympy.functions.FloatPow,
                "FloatTrueDiv": torch.utils._sympy.functions.FloatTrueDiv,
                "IntTrueDiv": torch.utils._sympy.functions.IntTrueDiv,
                "IsNonOverlappingAndDenseIndicator": torch.utils._sympy.functions.IsNonOverlappingAndDenseIndicator,
                "TruncToFloat": torch.utils._sympy.functions.TruncToFloat,
                "TruncToInt": torch.utils._sympy.functions.TruncToInt,
                "RoundToInt": torch.utils._sympy.functions.RoundToInt,
                "RoundDecimal": torch.utils._sympy.functions.RoundDecimal,
                "ToFloat": torch.utils._sympy.functions.ToFloat,
                "Identity": torch.utils._sympy.functions.Identity,
            }
            self.symbol_name_to_symbol: dict[str, sympy.Symbol] = {}
            self.constants = deserialize_torch_artifact(constants)
            self.signature = self.deserialize_signature(
                serialized_graph_module.signature
            )

            # deserialization does analysis with checks on 0/1, so we create fake range constraints and
            # restore the original range constraints afterwards
            self.symbol_name_to_range = {}
            # we also need to bump unbacked sym[float,int] counters in the
            # shape env to accommodate unbacked symbols in the exported program
            self.unbacked_symbols = set()
            count_unbacked_symfloat, count_unbacked_symint = -1, -1
            unbacked_symfloat_prefix, unbacked_symint_prefix = (
                prefix_str[t] for t in [SymT.UNBACKED_FLOAT, SymT.UNBACKED_INT]
            )
            if symbol_name_to_range:
                for k, vr in symbol_name_to_range.items():
                    lower = vr.lower
                    self.symbol_name_to_range[k] = symbolic_shapes.ValueRanges(
                        _int_to_sympy_int(lower, -int_oo), vr.upper
                    )
                    if k.startswith(unbacked_symfloat_prefix):
                        i = int(k[len(unbacked_symfloat_prefix) :])
                        count_unbacked_symfloat = max(count_unbacked_symfloat, i)
                    elif k.startswith(unbacked_symint_prefix):
                        i = int(k[len(unbacked_symint_prefix) :])
                        count_unbacked_symint = max(count_unbacked_symint, i)

            # TODO(pianpwk): if we can clean up unused symbols in range_constraints,
            # then this logic can just be handled with self.unbacked_symbols alone
            for _ in range(count_unbacked_symfloat + 1):
                self.shape_env.unbacked_symfloat_counter += 1
            for _ in range(count_unbacked_symint + 1):
                self.shape_env.unbacked_symint_counter += 1

            if example_inputs is not None and len(example_inputs) > 0:
                self.example_inputs = deserialize_torch_artifact(example_inputs)
            else:
                self.example_inputs = None
            self.deserialize_graph(serialized_graph_module.graph)

            with _enable_graph_inputs_of_type_nn_module(self.example_inputs):
                module_call_graph = self.deserialize_module_call_graph(
                    serialized_graph_module.module_call_graph
                )
            graph_module = ep._create_graph_module_for_export(self.module, self.graph)
            meta = {}
            if custom := serialized_graph_module.metadata.get("custom"):
                meta["custom"] = json.loads(custom)
            if hasattr(serialized_graph_module, "treespec_namedtuple_fields"):
                meta["treespec_namedtuple_fields"] = {}
                for (
                    type_,
                    fields,
                ) in serialized_graph_module.treespec_namedtuple_fields.items():
                    meta["treespec_namedtuple_fields"][type_] = fields.field_names
            graph_module.meta = meta
            return GraphModuleDeserializer.Result(
                graph_module=graph_module,
                signature=self.signature,
                module_call_graph=module_call_graph,
                names_to_symbols=self.symbol_name_to_symbol,
                state_dict=deserialize_torch_artifact(serialized_state_dict),
                constants=self.constants,
                example_inputs=self.example_inputs,
            )
        finally:
            _CURRENT_DESERIALIZER = None

    def sync_fx_node(self, name: str, fx_node: torch.fx.Node):
        if name in self.serialized_name_to_node:
            raise SerializeError(f"Node {name} has already been deserialized before.")
        # overwrite name
        fx_node.name = name
        self.serialized_name_to_node[name] = fx_node
        if "val" in fx_node.meta:
            raise AssertionError(f"fx_node {name} already has 'val' in meta")
        fx_node.meta["val"] = self.serialized_name_to_meta[name]

    def deserialize_sym_op_inputs(self, inputs):
        return tuple(self.deserialize_input(input.arg) for input in inputs)

    def deserialize_inputs(self, target, serialized_node: Node):
        schema_args = _get_schema_from_target(target).arguments
        argument_kinds = {input.name: input.kind for input in serialized_node.inputs}
        actual_args = {
            input.name: self.deserialize_input(input.arg)
            for input in serialized_node.inputs
        }
        args = []
        kwargs: OrderedDict[str, Any] = OrderedDict()
        for schema_arg in schema_args:
            if schema_arg.name in actual_args:
                arg = actual_args[schema_arg.name]
                kind = argument_kinds[schema_arg.name]
                if kind == ArgumentKind.POSITIONAL:
                    args.append(arg)
                    continue
                elif kind == ArgumentKind.KEYWORD and not keyword.iskeyword(
                    schema_arg.name
                ):
                    kwargs[schema_arg.name] = arg
                    continue

            # If there's no ArgumentKind found, fallback to the old cases.
            is_positional = (
                not schema_arg.has_default_value() and not schema_arg.kwarg_only
            )
            if is_positional:
                args.append(actual_args[schema_arg.name])
            elif keyword.iskeyword(schema_arg.name):
                if schema_arg.kwarg_only:
                    raise AssertionError(
                        f"schema_arg {schema_arg.name} should not be kwarg_only"
                    )
                if len(kwargs) > 0:
                    kwargs = OrderedDict()
                    args.extend(list(kwargs.values()))
                args.append(actual_args[schema_arg.name])
            else:
                if schema_arg.name in actual_args:
                    kwargs[schema_arg.name] = actual_args[schema_arg.name]
        return tuple(args), kwargs

    def deserialize_hoo_inputs(self, inputs: list[NamedArgument]):
        """
        For deserializing HOO inputs since HOOs do not have a schema.
        """
        args = []
        kwargs = {}
        for input_ in inputs:
            if input_.kind == ArgumentKind.POSITIONAL or input_.name == "":
                args.append(self.deserialize_input(input_.arg))
            else:
                kwargs[input_.name] = self.deserialize_input(input_.arg)
        return (tuple(args), kwargs)

    def deserialize_input(self, inp: Argument) -> Any:
        value = inp.value
        typ_ = inp.type
        if typ_ == "as_none":
            # None should converted as None, but is encoded as bool in serialized
            # Convert serialized object to torch equivalent
            return None
        elif typ_ == "as_tensor":
            return self.serialized_name_to_node[inp.as_tensor.name]
        elif typ_ == "as_scalar_type":
            return _SERIALIZE_TO_TORCH_DTYPE[inp.as_scalar_type]
        elif typ_ == "as_memory_format":
            return _SERIALIZE_TO_TORCH_MEMORY_FORMAT[inp.as_memory_format]
        elif typ_ == "as_layout":
            return _SERIALIZE_TO_TORCH_LAYOUT[inp.as_layout]
        elif typ_ == "as_graph":
            if not isinstance(value, GraphArgument):
                raise AssertionError(
                    f"expected GraphArgument, got {type(value).__name__}"
                )
            with self.save_graph_module():
                self.deserialize_graph(value.graph)
                submodule = ep._create_graph_module_for_export(self.module, self.graph)
            self.module.register_module(value.name, submodule)
            return self.graph.create_node(
                "get_attr",
                value.name,
                name=value.name,
            )
        elif typ_ == "as_device":
            return deserialize_device(inp.as_device)
        elif typ_ == "as_int":
            return inp.as_int
        elif typ_ == "as_float":
            return inp.as_float
        elif typ_ == "as_bool":
            return inp.as_bool
        elif typ_ == "as_string":
            return inp.as_string
        elif typ_ == "as_complex":
            return complex(inp.as_complex.real, inp.as_complex.imag)
        elif typ_ == "as_sym_int":
            return self.deserialize_sym_argument(inp.as_sym_int)
        elif typ_ == "as_sym_float":
            return self.deserialize_sym_argument(inp.as_sym_float)
        elif typ_ == "as_sym_bool":
            return self.deserialize_sym_argument(inp.as_sym_bool)
        elif isinstance(value, dict):
            if typ_ == "as_string_to_argument":
                # Deserialize dict[str, Argument] recursively
                return {k: self.deserialize_input(v) for k, v in value.items()}
            else:
                raise SerializeError(f"Unknown dict type: {typ_}")
        elif isinstance(value, list):
            if len(value) == 0:
                return []
            elif typ_ == "as_tensors":
                result = [self.serialized_name_to_node[arg.name] for arg in value]
                return result
            elif typ_ in ("as_ints", "as_floats", "as_bools", "as_strings"):
                # convert from serialized.python.types.List to python list
                return list(value)
            elif typ_ == "as_int_lists":
                # Convert list of lists back to list of tuples for Triton grids
                return [tuple(dims) for dims in value]
            elif typ_ == "as_float_lists":
                return [list(floats) for floats in value]
            elif typ_ == "as_nested_tensors":
                # nested list of tensors (List[List[Tensor]])
                return [
                    [self.serialized_name_to_node[arg.name] for arg in inner_list]
                    for inner_list in value
                ]
            elif typ_ in ("as_sym_ints", "as_sym_bools", "as_sym_floats"):
                return [self.deserialize_sym_argument(arg) for arg in value]
            elif typ_ == "as_optional_tensors":

                def deserialize_optional_tensor_args(a):
                    if a.type == "as_none":
                        return None
                    elif a.type == "as_tensor":
                        return self.serialized_name_to_node[a.value.name]
                    else:
                        raise SerializeError(f"Unhandled argument {inp}")

                return list(map(deserialize_optional_tensor_args, value))
            else:
                raise SerializeError(f"Unhandled argument {inp}")
        elif typ_ == "as_custom_obj":
            if inp.as_custom_obj.name in self.serialized_name_to_node:
                # Custom object has been lifted as an input
                return self.serialized_name_to_node[inp.as_custom_obj.name]
            return self.constants[inp.as_custom_obj.name]
        elif typ_ == "as_operator":
            return self.deserialize_operator(inp.as_operator)
        else:
            raise SerializeError(f"Unhandled argument {inp}")

    def deserialize_constant_input(self, inp: ConstantValue) -> Any:
        if inp.type == "as_int":
            return int(inp.as_int)
        elif inp.type == "as_float":
            return float(inp.as_float)
        elif inp.type == "as_string":
            return str(inp.as_string)
        elif inp.type == "as_bool":
            return bool(inp.as_bool)
        elif inp.type == "as_none":
            return None
        else:
            raise SerializeError(f"Unhandled constant argument {inp} to deserialize")

    def deserialize_sym_argument(self, sym_arg):
        if isinstance(sym_arg, SymIntArgument):
            if sym_arg.type == "as_int":
                return sym_arg.as_int
            elif sym_arg.type == "as_name":
                return self.serialized_name_to_node[sym_arg.as_name]
        elif isinstance(sym_arg, SymFloatArgument):
            if sym_arg.type == "as_float":
                return sym_arg.as_float
            elif sym_arg.type == "as_name":
                return self.serialized_name_to_node[sym_arg.as_name]
        elif isinstance(sym_arg, SymBoolArgument):
            if sym_arg.type == "as_bool":
                return sym_arg.as_bool
            elif sym_arg.type == "as_name":
                return self.serialized_name_to_node[sym_arg.as_name]
        raise SerializeError(f"Unknown symbolic argument type: {sym_arg}")

    def deserialize_sym_op_outputs(self, serialized_node: Node, fx_node: torch.fx.Node):
        self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)

    def deserialize_outputs(self, serialized_node: Node, fx_node: torch.fx.Node):
        # Check single value return
        if len(serialized_node.outputs) == 0:
            return

        if (
            len(serialized_node.outputs) == 1
            and "torch.ops.higher_order" in serialized_node.target
            and not getattr(serialized_node, "is_hop_single_tensor_return", True)
            and serialized_node.outputs[0].type != "as_none"
        ):

            def _deserialize_hop_with_single_return(serialized_node, fx_node):
                meta_val: list[Any] = []
                arg = None
                if serialized_node.outputs[0].type == "as_tensor":
                    arg = serialized_node.outputs[0].as_tensor
                elif isinstance(
                    serialized_node.outputs[0].value,
                    (SymIntArgument, SymBoolArgument, SymFloatArgument),
                ):
                    arg = serialized_node.outputs[0].value
                deserialized_metadata = self.deserialize_metadata(
                    serialized_node.metadata
                )
                if arg is None:
                    raise AssertionError("arg should not be None")
                # pyrefly: ignore [bad-argument-type]
                self.generate_getitem(meta_val, fx_node, arg, 0, deserialized_metadata)
                fx_node.meta["val"] = tuple(meta_val)
                self.serialized_name_to_node[fx_node.name] = fx_node
                return

            return _deserialize_hop_with_single_return(serialized_node, fx_node)

        if (
            len(serialized_node.outputs) == 1
            and serialized_node.outputs[0].type == "as_tensor"
        ):
            self.sync_fx_node(serialized_node.outputs[0].as_tensor.name, fx_node)
            return
        elif len(serialized_node.outputs) == 1 and isinstance(
            serialized_node.outputs[0].value,
            (SymIntArgument, SymBoolArgument, SymFloatArgument),
        ):
            self.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node)
            return
        elif (
            len(serialized_node.outputs) == 1
            and serialized_node.outputs[0].type == "as_none"
        ):
            # manually rename the node to a unused name to avoid naming conflicts
            fx_node.meta["val"] = None
            fx_node._rename(f"{self.graph._target_to_str(fx_node.target)}_unused")
            return

        self.deserialize_multiple_outputs(serialized_node, fx_node)

    def generate_getitem(
        self,
        meta_val,
        fx_node: torch.fx.Node,
        arg: TensorArgument | SymIntArgument | SymFloatArgument,
        idx: int,
        deserialized_metadata: dict[str, Any],
    ):
        if isinstance(arg, TensorArgument):
            name = arg.name
        elif isinstance(arg, SymIntArgument):
            name = arg.as_name
        elif isinstance(arg, SymFloatArgument):
            name = arg.as_name
        else:
            raise AssertionError(
                f"generate_getitem got unknown argument type {type(arg)}"
            )
        individual_output = self.graph.create_node(
            "call_function",
            operator.getitem,
            (fx_node, idx),
            name=name,
        )
        self.sync_fx_node(name, individual_output)
        meta_val.append(self.serialized_name_to_meta[name])
        # The derived `getitem` nodes should have the same stacktrace as the
        # original `fx_node`
        individual_output.meta.update(deserialized_metadata)

    def generate_getitems(
        self,
        meta_val,
        fx_node: torch.fx.Node,
        args,
        deserialized_metadata: dict[str, Any],
    ):
        for idx, arg in enumerate(args):
            if isinstance(arg, (TensorArgument, SymIntArgument, SymFloatArgument)):
                self.generate_getitem(
                    meta_val, fx_node, arg, idx, deserialized_metadata
                )
                continue

            if not isinstance(arg, Argument):
                raise AssertionError(f"expected Argument, got {type(arg).__name__}")
            if arg.type in ("as_tensor", "as_sym_int", "as_sym_float"):
                self.generate_getitem(
                    meta_val, fx_node, arg.value, idx, deserialized_metadata
                )
            elif arg.type in (
                "as_tensors",
                "as_sym_ints",
                "as_sym_floats",
                "as_ints",
                "as_floats",
                "as_strings",
                "as_bools",
                "as_sym_bools",
            ):
                list_output = self.graph.create_node(
                    "call_function",
                    operator.getitem,
                    (fx_node, idx),
                )
                meta_val.append([])
                self.generate_getitems(
                    meta_val[-1], list_output, arg.value, deserialized_metadata
                )
                list_output.meta.update(deserialized_metadata)
                list_output.meta["val"] = meta_val[-1]
            elif arg.type == "as_none":
                individual_output = self.graph.create_node(
                    "call_function",
                    operator.getitem,
                    (fx_node, idx),
                    name="as_none",
                )
                meta_val.append(None)
                individual_output.meta["val"] = None
                individual_output.meta.update(deserialized_metadata)
            else:
                raise NotImplementedError(f"Unimplemented node output type: {arg}")

    def deserialize_multiple_outputs(
        self, serialized_node: Node, fx_node: torch.fx.Node
    ) -> None:
        deserialized_metadata = self.deserialize_metadata(serialized_node.metadata)

        # Convert multiple return types to FX format.
        # In FX, each node only returns one value. So in order to represent
        # multiple return values, we have to emit a `getitem` node for each
        # return value.
        # This performs the inverse mapping of the `serialize_outputs` call in
        # serialization, see [NOTE: Multiple outputs]
        meta_val: list[Any] = []
        if len(serialized_node.outputs) == 1:
            if not isinstance(serialized_node.outputs[0].value, list):
                raise AssertionError(
                    f"expected list, got {type(serialized_node.outputs[0].value).__name__}"
                )
            if not isinstance(serialized_node.outputs[0].value[0], TensorArgument):
                raise AssertionError(
                    f"expected TensorArgument, got {type(serialized_node.outputs[0].value[0]).__name__}"
                )
            self.generate_getitems(
                meta_val,
                fx_node,
                serialized_node.outputs[0].as_tensors,
                deserialized_metadata,
            )
        else:
            self.generate_getitems(
                meta_val, fx_node, serialized_node.outputs, deserialized_metadata
            )

        # also update the metaval for `fx_node` to be a list(meta)
        fx_node.meta["val"] = tuple(meta_val)
        self.serialized_name_to_node[fx_node.name] = fx_node

    def deserialize_metadata(self, metadata: dict[str, str]) -> dict[str, Any]:
        ret: dict[str, Any] = {}
        if stack_trace := metadata.get("stack_trace"):
            ret["stack_trace"] = stack_trace

        def deserialize_meta_func(serialized_target: str):
            module = None
            if serialized_target.startswith("torch.nn"):
                module = torch.nn
                serialized_target_names = serialized_target.split(".")[2:]
            elif serialized_target.startswith("torch"):
                module = torch
                serialized_target_names = serialized_target.split(".")[1:]
            else:
                return self.deserialize_operator(serialized_target)

            target = module
            for name in serialized_target_names:
                if not hasattr(target, name):
                    return serialized_target
                else:
                    target = getattr(target, name)
            return target

        if nn_module_stack_str := metadata.get("nn_module_stack"):
            # Originally serialized to "key,orig_path,type_str"
            def import_nn_module_stack(key, path, ty):
                return key, (path, ty)

            # Helper function to split string by commas, accounting for nested parentheses/brackets
            def metadata_split(metadata):
                out = []
                start, n = 0, 0
                a, b = "[(", ")]"
                for end, c in enumerate(metadata):
                    if c in a:
                        n += 1
                    elif c in b:
                        n -= 1
                    elif c == "," and n == 0:
                        out.append(metadata[start:end])
                        start = end + 1
                out.append(metadata[start:])
                if len(out) != 3:
                    raise AssertionError(
                        f"expected metadata_split to return 3 parts, got {len(out)}"
                    )
                return out

            nn_module_stack = dict(
                import_nn_module_stack(*metadata_split(item))
                for item in nn_module_stack_str.split(ST_DELIMITER)
            )
            ret["nn_module_stack"] = nn_module_stack

        if source_fn_st_str := metadata.get("source_fn_stack"):
            # Originally serializes to "fx_node_name,op_str"
            source_fn_st = []
            for source_fn_str in source_fn_st_str.split(ST_DELIMITER):
                name, target_str = source_fn_str.split(",")
                source_fn_st.append((name, deserialize_meta_func(target_str)))
            ret["source_fn_stack"] = source_fn_st

        if torch_fn_str := metadata.get("torch_fn"):
            ret["torch_fn"] = tuple(torch_fn_str.split(ST_DELIMITER))

        if custom_str := metadata.get("custom"):
            ret["custom"] = json.loads(custom_str)

        if from_node_str := metadata.get("from_node"):
            ret["from_node"] = self._deserialize_from_node(json.loads(from_node_str))

        return ret

    def _deserialize_from_node(
        self, from_node_data: list[dict[str, Any]] | None
    ) -> list[NodeSource] | None:
        """
        Deserialize from_node metadata from JSON data.
        """
        if from_node_data is None:
            return None

        if not isinstance(from_node_data, list):
            raise AssertionError(f"expected list, got {type(from_node_data).__name__}")

        return [
            node_source
            for fn_dict in from_node_data
            if (node_source := NodeSource._from_dict(fn_dict)) is not None
        ]

    def deserialize_argument_spec(self, x: Argument) -> ep.ArgumentSpec:
        log.debug("[deserialize_argument_spec] %s", x)
        if x.type == "as_tensor":
            return ep.TensorArgument(name=x.as_tensor.name)
        elif x.type == "as_sym_int":
            return ep.SymIntArgument(name=x.as_sym_int.as_name)
        elif x.type == "as_sym_float":
            return ep.SymFloatArgument(name=x.as_sym_float.as_name)
        elif x.type == "as_custom_obj":
            return ep.ConstantArgument(
                name=x.as_custom_obj.name, value=self.deserialize_input(x)
            )
        else:
            return ep.ConstantArgument(name="", value=self.deserialize_input(x))

    def deserialize_module_call_signature(
        self, module_call_signature: ModuleCallSignature
    ) -> ep.ModuleCallSignature:
        return ep.ModuleCallSignature(
            inputs=[
                self.deserialize_argument_spec(x) for x in module_call_signature.inputs
            ],
            outputs=[
                self.deserialize_argument_spec(x) for x in module_call_signature.outputs
            ],
            in_spec=treespec_loads(module_call_signature.in_spec),
            out_spec=treespec_loads(module_call_signature.out_spec),
            forward_arg_names=(
                names if (names := module_call_signature.forward_arg_names) else None
            ),
        )

    def deserialize_module_call_graph(
        self, module_call_graph: list[ModuleCallEntry]
    ) -> list[ep.ModuleCallEntry]:
        log.debug("\n[deserialize_module_call_graph]")
        return [
            ep.ModuleCallEntry(
                fqn=entry.fqn,
                signature=(
                    self.deserialize_module_call_signature(entry.signature)
                    if entry.signature
                    else None
                ),
            )
            for entry in module_call_graph
        ]


@final
class ExportedProgramDeserializer(metaclass=Final):
    def __init__(self, expected_opset_version: dict[str, int] | None = None):
        self.expected_opset_version: dict[str, int] = {}
        if expected_opset_version:
            self.expected_opset_version.update(expected_opset_version)
        if "aten" not in self.expected_opset_version:
            self.expected_opset_version["aten"] = torch._C._get_max_operator_version()

    def deserialize_range_constraints(
        self,
        symbol_name_to_range: dict[str, symbolic_shapes.ValueRanges],
        symbol_name_to_symbol: dict[str, sympy.Symbol],
    ) -> dict[sympy.Symbol, ValueRanges]:
        log.debug("\n[deserialize_range_constraints]")
        range_constraints = {}
        for k, v in symbol_name_to_range.items():
            if symbol := symbol_name_to_symbol.get(k):
                log.debug("[deserialize_range_constraints] %s -> %s", k, v)
                range_constraints[symbol] = v  # type: ignore[arg-type]
            else:
                log.warning(
                    "Symbol %s did not appear in the graph that was deserialized", k
                )
        return range_constraints

    def deserialize(
        self,
        exported_program: ExportedProgram,
        state_dict: dict[str, torch.Tensor] | bytes,
        constants: dict[str, torch.Tensor] | bytes,
        example_inputs: tuple[tuple[torch.Tensor, ...], dict[str, Any]]
        | bytes
        | None = None,
        *,
        _unsafe_skip_version_check=False,
    ) -> ep.ExportedProgram:
        if not isinstance(exported_program, ExportedProgram):
            raise AssertionError(
                f"expected ExportedProgram, got {type(exported_program).__name__}"
            )
        version = exported_program.schema_version

        # TODO(zhxchen17) blocked on thrift schema refactor
        if version.major != SCHEMA_VERSION[0] and not (
            version.major == 0 and version.minor == 0
        ):
            if not _unsafe_skip_version_check:
                raise SerializeError(
                    f"Serialized schema version {exported_program.schema_version} "
                    f"does not match our current schema version {SCHEMA_VERSION}."
                )

        symbol_name_to_range = {
            k: symbolic_shapes.ValueRanges(
                _int_to_sympy_int(v.min_val, -int_oo),
                _int_to_sympy_int(v.max_val, int_oo),
            )
            for k, v in exported_program.range_constraints.items()
        }
        res = GraphModuleDeserializer().deserialize(
            exported_program.graph_module,
            state_dict,
            constants,
            example_inputs,
            symbol_name_to_range,
        )
        range_constraints = self.deserialize_range_constraints(
            symbol_name_to_range,
            res.names_to_symbols,
        )

        result = ep.ExportedProgram(
            root=res.graph_module,
            graph=res.graph_module.graph,
            graph_signature=res.signature,
            state_dict=res.state_dict,  # type: ignore[arg-type]
            range_constraints=range_constraints,
            module_call_graph=res.module_call_graph,
            example_inputs=res.example_inputs,
            constants=res.constants,
            verifiers=[load_verifier(v) for v in exported_program.verifiers],
        )
        result._guards_code = exported_program.guards_code
        log.debug("\n[deserialize]: %s", result)
        return result


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, bytes):
            return base64.b64encode(obj).decode("utf-8")
        return super().default(obj)


def _dataclass_to_dict(obj):
    if isinstance(obj, _Union):
        return {obj.type: _dataclass_to_dict(obj.value)}
    elif dataclasses.is_dataclass(obj):
        return {
            f.name: _dataclass_to_dict(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    elif isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(_dataclass_to_dict(x) for x in obj)
    elif isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, float):
        if obj == math.inf:
            return "Infinity"
        elif obj == -math.inf:
            return "-Infinity"
        elif math.isnan(obj):
            return "NaN"
        else:
            return obj
    else:
        return obj


def _to_json_bytes(obj: Any) -> bytes:
    return json.dumps(_dataclass_to_dict(obj), cls=EnumEncoder, allow_nan=False).encode(
        "utf-8"
    )


def serialize(
    exported_program: ep.ExportedProgram,
    opset_version: dict[str, int] | None = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
    *,
    serialize_state_dict: bool = True,
    serialize_constants: bool = True,
    serialize_example_inputs: bool = True,
) -> SerializedArtifact:
    with _enable_graph_inputs_of_type_nn_module(exported_program.example_inputs):
        serialized_program = ExportedProgramSerializer(
            opset_version, pickle_protocol
        ).serialize(
            exported_program,
            serialize_state_dict=serialize_state_dict,
            serialize_constants=serialize_constants,
            serialize_example_inputs=serialize_example_inputs,
        )
    if not isinstance(serialized_program.exported_program, ExportedProgram):
        raise AssertionError(
            f"expected ExportedProgram, got {type(serialized_program.exported_program).__name__}"
        )

    json_bytes = _to_json_bytes(serialized_program.exported_program)
    artifact = SerializedArtifact(
        json_bytes,
        serialized_program.state_dict,
        serialized_program.constants,
        serialized_program.example_inputs,
    )
    return artifact


def _resolve_schema_cls(cls):
    if isinstance(cls, str):
        resolved = getattr(schema, cls, None)
        if resolved is not None:
            return resolved
    if isinstance(cls, typing.ForwardRef):
        return _resolve_schema_cls(cls.__forward_arg__)
    return cls


def _dict_to_dataclass(cls, data):
    cls = _resolve_schema_cls(cls)
    if isinstance(cls, str):
        raise AssertionError(f"Unresolved class type: '{cls}'.")
    if typing.get_origin(cls) is Annotated:
        return _dict_to_dataclass(cls.__origin__, data)
    if typing.get_origin(cls) in (typing.Union, types.UnionType) and type(
        None
    ) in typing.get_args(cls):
        if data is None:
            return None
        ty_args = typing.get_args(cls)
        if len(ty_args) != 2:
            raise AssertionError(f"expected 2 type args, got {len(ty_args)}")
        return _dict_to_dataclass(ty_args[0], data)
    elif isinstance(cls, type) and issubclass(cls, _Union):
        if not isinstance(data, dict):
            raise AssertionError(f"expected dict, got {type(data).__name__}")
        if len(data) != 1:
            raise AssertionError(f"expected dict with 1 key, got {len(data)}")
        _type = next(iter(data.keys()))
        _value = next(iter(data.values()))
        if not isinstance(_type, str):
            raise AssertionError(f"expected str key, got {type(_type).__name__}")
        type_hints = typing.get_type_hints(cls, globalns=vars(schema))
        field_type = type_hints[_type]
        # pyrefly: ignore [missing-attribute]
        return cls.create(**{_type: _dict_to_dataclass(field_type, _value)})
    elif dataclasses.is_dataclass(cls):
        fields = {}
        type_hints = typing.get_type_hints(cls, globalns=vars(schema))
        # For forward compatibility consideration, we ignore all the keys
        # that are not showing up in the dataclass definition.
        for f in dataclasses.fields(cls):
            name = f.name
            if name not in data:
                continue
            new_field_obj = _dict_to_dataclass(type_hints[name], data[name])
            fields[name] = new_field_obj
        return cls(**fields)  # type: ignore[operator]
    elif isinstance(data, list):
        if len(data) == 0:
            return data
        d_type = typing.get_args(cls)[0]
        return [_dict_to_dataclass(d_type, d) for d in data]
    elif isinstance(data, dict):
        v_type = typing.get_args(cls)[1]
        return {k: _dict_to_dataclass(v_type, v) for k, v in data.items()}
    elif cls is float:
        return float(data)
    return data


def _bytes_to_dataclass(cls: Any, artifact_bytes: bytes) -> Any:
    artifact_str = artifact_bytes.decode("utf-8")
    artifact_dict = json.loads(artifact_str)
    artifact_dataclass = _dict_to_dataclass(cls, artifact_dict)
    return artifact_dataclass


def deserialize(
    artifact: SerializedArtifact,
    expected_opset_version: dict[str, int] | None = None,
    *,
    _unsafe_skip_version_check=False,
) -> ep.ExportedProgram:
    if not isinstance(artifact.exported_program, bytes):
        raise AssertionError(
            f"expected bytes, got {type(artifact.exported_program).__name__}"
        )
    serialized_exported_program = _bytes_to_dataclass(
        ExportedProgram, artifact.exported_program
    )
    return ExportedProgramDeserializer(expected_opset_version).deserialize(
        serialized_exported_program,
        artifact.state_dict,
        artifact.constants,
        artifact.example_inputs,
        _unsafe_skip_version_check=_unsafe_skip_version_check,
    )


def _canonicalize_graph(
    sorted_inputs, sorted_outputs, graph, constants
) -> tuple[Graph, dict[str, str]]:
    def _get_argument(a: Argument):
        if a.type == "as_none":
            return None
        elif a.type == "as_tensor":
            return a.as_tensor
        elif a.type == "as_tensors":
            return a.as_tensors
        elif a.type == "as_int":
            return None
        elif a.type == "as_ints":
            return None
        elif a.type == "as_float":
            return None
        elif a.type == "as_floats":
            return None
        elif a.type == "as_string":
            return None
        elif a.type == "as_strings":
            return None
        elif a.type == "as_complex":
            return None
        elif a.type == "as_sym_int":
            return a.as_sym_int
        elif a.type == "as_sym_ints":
            return a.as_sym_ints
        elif a.type == "as_sym_float":
            return a.as_sym_float
        elif a.type == "as_sym_floats":
            return a.as_sym_floats
        elif a.type == "as_scalar_type":
            return None
        elif a.type == "as_memory_format":
            return None
        elif a.type == "as_layout":
            return None
        elif a.type == "as_device":
            return None
        elif a.type == "as_bool":
            return None
        elif a.type == "as_bools":
            return None
        elif a.type == "as_sym_bool":
            return a.as_sym_bool
        elif a.type == "as_sym_bools":
            return a.as_sym_bools
        elif a.type == "as_graph":
            return None
        elif a.type == "as_optional_tensors":
            return a.as_optional_tensors
        elif a.type == "as_custom_obj":
            return a.as_custom_obj
        elif a.type == "as_operator":
            return None
        elif a.type == "as_int_lists":
            return None
        elif a.type == "as_float_lists":
            return None
        elif a.type == "as_string_to_argument":
            return None
        elif a.type == "as_nested_tensors":
            return a.as_nested_tensors
        else:
            raise AssertionError(f"Unknown input type to the ExportedProgram: {a}")

    # Stage 1: Reorder named items.
    def for_args(f, a):
        if not isinstance(a, Argument):
            raise AssertionError(f"expected Argument, got {type(a).__name__}")
        pytree.tree_map(f, _get_argument(a))

    def sort_nodes(nodes):
        @dataclass
        class Edges:
            outs: list[int]
            ins: int

        graph_inputs: set[str] = set()
        def_table: dict[str, int] = {}
        edges: dict[int, Edges] = {}
        candidates: list[tuple[str, list[tuple[str, list[int]]], int]] = []
        rank: dict[str, int] = {}
        ret: list[Node] = []

        def get_name(a) -> str | None:
            if a is None:
                return None
            if isinstance(a, TensorArgument):
                return a.name
            elif isinstance(a, (SymIntArgument, SymBoolArgument, SymFloatArgument)):
                if a.type == "as_name":
                    return a.as_name
                elif a.type in ("as_int", "as_bool", "as_float"):
                    return None
                else:
                    raise AssertionError(f"Unknown argument type: {a}")
            elif isinstance(a, OptionalTensorArgument):
                if a.type == "as_tensor":
                    return a.as_tensor.name
                elif a.type == "as_none":
                    return None
                else:
                    raise AssertionError(f"Unknown optional tensor type: {a}")
            elif isinstance(a, CustomObjArgument):
                return a.name
            else:
                raise AssertionError(f"Unknown argument type: {a}")

        for i in sorted_inputs:

            def add_input(a):
                if s := get_name(a):
                    graph_inputs.add(s)

            for_args(add_input, i)

        for idx, node in enumerate(nodes):

            def add_def(a):
                if s := get_name(a):
                    if s in def_table:
                        raise AssertionError(f"symbol {s!r} already in def_table")
                    def_table[s] = idx

            for o in node.outputs:
                for_args(add_def, o)

            edges[idx] = Edges([], 0)

        for idx, user in enumerate(nodes):

            def add_edge(a):
                if s := get_name(a):
                    if s in constants:
                        return
                    if s not in def_table:
                        if s not in graph_inputs:
                            raise AssertionError(
                                f"symbol {s!r} not in def_table or graph_inputs"
                            )
                        return
                    src = def_table[s]
                    edges[src].outs.append(idx)
                    edges[idx].ins += 1

            for i in user.inputs:
                for_args(add_edge, i.arg)

        def add_rank(a):
            if s := get_name(a):
                if s in rank:
                    raise AssertionError(f"symbol {s!r} already in rank")
                rank[s] = len(rank)

        def get_rank(a):
            s = get_name(a)
            if s and s not in constants:
                return rank[s]
            else:
                return -1

        for i in sorted_inputs:
            for_args(add_rank, i)

        def add_candidate(idx: int):
            def get_ranks(i):
                ranks = []
                for_args(lambda x: ranks.append(get_rank(x)), i)
                return ranks

            node = nodes[idx]
            args_rank = [(a.name, get_ranks(a.arg)) for a in node.inputs]
            heapq.heappush(candidates, (node.target, args_rank, idx))

        for idx, e in edges.items():
            if e.ins == 0:
                add_candidate(idx)

        while len(candidates) > 0:
            _, _, idx = heapq.heappop(candidates)
            node = nodes[idx]
            for o in node.outputs:
                for_args(add_rank, o)
            ret.append(node)
            if idx not in edges:
                raise AssertionError(f"idx {idx} not in edges")
            for user in edges[idx].outs:
                e = edges[user]
                if e.ins <= 0:
                    raise AssertionError(f"e.ins should be > 0, got {e.ins}")
                e.ins -= 1
                if e.ins == 0:
                    add_candidate(user)
            edges[idx].outs.clear()

        return ret

    sorted_nodes = sort_nodes(graph.nodes)
    if len(sorted_nodes) != len(graph.nodes):
        raise AssertionError(
            f"expected {len(graph.nodes)} sorted nodes, got {len(sorted_nodes)}"
        )

    # Stage 2: Rename nodes.
    name_table: dict[str, str] = {}

    def rename_def(a):
        def _rename(arg_name, values):
            new_name = f"_{len(name_table)}"
            if arg_name in name_table:
                raise AssertionError(f"arg_name {arg_name!r} already in name_table")
            name_table[arg_name] = new_name
            if arg_name not in values:
                raise AssertionError(f"arg_name {arg_name!r} not in values")
            values[new_name] = values.pop(arg_name)
            return new_name

        if a is None:
            return
        if isinstance(a, TensorArgument):
            a.name = _rename(a.name, graph.tensor_values)
        elif isinstance(a, SymIntArgument):
            if a.type == "as_name":
                a.as_name = _rename(a.as_name, graph.sym_int_values)
        elif isinstance(a, SymFloatArgument):
            if a.type == "as_name":
                a.as_name = _rename(a.as_name, graph.sym_float_values)
        elif isinstance(a, SymBoolArgument):
            if a.type == "as_name":
                a.as_name = _rename(a.as_name, graph.sym_bool_values)
        elif isinstance(a, CustomObjArgument):
            a.name = _rename(a.name, graph.custom_obj_values)
        else:
            raise AssertionError(f"Unknown argument type: {a}")

    def replace_use(a):
        if a is None:
            return
        if isinstance(a, TensorArgument):
            a.name = name_table.get(a.name, a.name)
        elif isinstance(a, (SymIntArgument, SymFloatArgument)):
            if a.type == "as_name":
                a.as_name = name_table.get(a.as_name, a.as_name)
        elif isinstance(a, SymBoolArgument):
            if a.type == "as_name":
                a.as_name = name_table.get(a.as_name, a.as_name)
        elif isinstance(a, OptionalTensorArgument):
            if a.type == "as_tensor":
                a.as_tensor.name = name_table.get(a.as_tensor.name, a.as_tensor.name)
        elif isinstance(a, CustomObjArgument):
            a.name = name_table.get(a.name, a.name)
        else:
            raise AssertionError(f"Unknown argument type: {a}")

    for i in sorted_inputs:
        for_args(rename_def, i)

    for n in sorted_nodes:
        for o in n.outputs:
            for_args(rename_def, o)

    for n in sorted_nodes:
        for i in n.inputs:
            for_args(replace_use, i.arg)

    for o in sorted_outputs:
        for_args(replace_use, o)

    # Stage 3: Remove unstable fields.
    for n in sorted_nodes:
        n.metadata.clear()

    # Stage 4: Aggregate values.
    # pyrefly: ignore [no-matching-overload]
    sorted_tensor_values = dict(
        sorted(graph.tensor_values.items(), key=operator.itemgetter(0))
    )
    # pyrefly: ignore [no-matching-overload]
    sorted_sym_int_values = dict(
        sorted(graph.sym_int_values.items(), key=operator.itemgetter(0))
    )
    # pyrefly: ignore [no-matching-overload]
    sorted_sym_float_values = dict(
        sorted(graph.sym_float_values.items(), key=operator.itemgetter(0))
    )
    # pyrefly: ignore [no-matching-overload]
    sorted_sym_bool_values = dict(
        sorted(graph.sym_bool_values.items(), key=operator.itemgetter(0))
    )
    # pyrefly: ignore [no-matching-overload]
    sorted_custom_obj_values = dict(
        sorted(graph.custom_obj_values.items(), key=operator.itemgetter(0))
    )

    # Stage 5: Recurse in subgraphs.
    counter = 0
    for node in sorted_nodes:
        for i in node.inputs:
            a = i.arg
            if a.type == "as_graph":
                a.as_graph.graph, _ = _canonicalize_graph(
                    a.as_graph.graph.inputs,
                    a.as_graph.graph.outputs,
                    a.as_graph.graph,
                    constants,
                )
                a.as_graph.name = f"_g{counter}"
                counter += 1

    graph = Graph(
        inputs=sorted_inputs,
        outputs=sorted_outputs,
        nodes=sorted_nodes,
        tensor_values=sorted_tensor_values,
        sym_int_values=sorted_sym_int_values,
        sym_float_values=sorted_sym_float_values,
        sym_bool_values=sorted_sym_bool_values,
        is_single_tensor_return=graph.is_single_tensor_return,
        custom_obj_values=sorted_custom_obj_values,
    )
    return graph, name_table


def canonicalize(
    ep: ExportedProgram, constants: set[str] | None = None
) -> ExportedProgram:
    """
    Normalize a serialized ExportedProgram, so that different eager program which
    shares the same semantics can get a single representation on disk.

    This function canonicalizes an ExportedProgram by:

    1. Sorting nodes in topological order.
    2. Rename nodes to have unique names.
    3. Remove unstable fields.
    4. Aggregate the above program fields.
    5. Recurse in subgraphs.

    Args:
        ep (ExportedProgram): The ExportedProgram to canonicalize.
        constants (Optional[set[str]]): Set of constants names

    Returns:
        ExportedProgram: The canonicalized exported program.
    """
    ep = copy.deepcopy(ep)
    # pyrefly: ignore [annotation-mismatch, redefinition]
    constants: set[str] = constants or set()

    opset_version = dict(sorted(ep.opset_version.items(), key=operator.itemgetter(0)))
    range_constraints = dict(
        sorted(ep.range_constraints.items(), key=operator.itemgetter(0))
    )
    guards_code = sorted(ep.guards_code)
    module_call_graph = sorted(ep.graph_module.module_call_graph, key=lambda x: x.fqn)
    signature = ep.graph_module.signature
    graph = ep.graph_module.graph

    if len(graph.inputs) != len(signature.input_specs):
        raise AssertionError(
            f"graph.inputs length {len(graph.inputs)} != signature.input_specs length {len(signature.input_specs)}"
        )
    if len(graph.outputs) != len(signature.output_specs):
        raise AssertionError(
            f"graph.outputs length {len(graph.outputs)} != signature.output_specs length {len(signature.output_specs)}"
        )

    def rank_input(inp) -> tuple[int, str | None, int]:
        idx, (_arg, spec) = inp
        if not isinstance(spec, InputSpec):
            raise AssertionError(f"expected InputSpec, got {type(spec).__name__}")
        if spec.type == "user_input":
            return 5, None, idx
        elif spec.type == "parameter":
            return 1, spec.parameter.parameter_name, idx
        elif spec.type == "buffer":
            return 2, spec.buffer.buffer_name, idx
        elif spec.type == "tensor_constant":
            return 3, spec.tensor_constant.tensor_constant_name, idx
        elif spec.type == "custom_obj":
            return 4, spec.custom_obj.custom_obj_name, idx
        elif spec.type == "token":
            return 0, None, idx
        elif spec.type == "constant_input":
            return 6, spec.constant_input.name, idx
        else:
            raise AssertionError(f"Unknown input type: {spec}")

    def rank_output(out) -> tuple[int, str | None, int]:
        idx, (_arg, spec) = out
        if not isinstance(spec, OutputSpec):
            raise AssertionError(f"expected OutputSpec, got {type(spec).__name__}")
        if spec.type == "user_output":
            return 4, None, idx
        elif spec.type == "loss_output":
            return 4, None, idx
        elif spec.type == "parameter_mutation":
            return 1, spec.parameter_mutation.parameter_name, idx
        elif spec.type == "buffer_mutation":
            return 2, spec.buffer_mutation.buffer_name, idx
        elif spec.type == "gradient_to_parameter":
            return 5, spec.gradient_to_parameter.parameter_name, idx
        elif spec.type == "gradient_to_user_input":
            return 6, None, idx
        elif spec.type == "user_input_mutation":
            return 3, None, idx
        elif spec.type == "token":
            return 0, None, idx
        else:
            raise AssertionError(f"Unknown output type: {spec}")

    sorted_ins = sorted(
        enumerate(zip(graph.inputs, signature.input_specs)), key=rank_input
    )

    if len(sorted_ins) > 0:
        sorted_inputs, input_specs = zip(*(i for idx, i in sorted_ins))  # type: ignore[assignment]
    else:
        sorted_inputs = ()
        input_specs = ()

    sorted_outs = sorted(
        enumerate(zip(graph.outputs, signature.output_specs)), key=rank_output
    )
    sorted_outputs, output_specs = zip(*(i for idx, i in sorted_outs))  # type: ignore[assignment]

    sorted_graph, replace_table = _canonicalize_graph(
        sorted_inputs, sorted_outputs, graph, constants
    )

    def replace_input(spec):
        if not isinstance(spec, InputSpec):
            raise AssertionError(f"expected InputSpec, got {type(spec).__name__}")
        if spec.type == "user_input":
            arg = spec.user_input.arg
            if arg.type == "as_tensor":
                t = arg.as_tensor
                t.name = replace_table[t.name]
            elif arg.type == "as_sym_int":
                s = arg.as_sym_int
                if s.type == "as_name":
                    s.as_name = replace_table[s.as_name]
                elif s.type == "as_int":
                    pass
                else:
                    raise AssertionError(f"Unknown sym_int type: {s}")
            elif arg.type == "as_sym_float":
                f = arg.as_sym_float
                if f.type == "as_name":
                    f.as_name = replace_table[f.as_name]
                elif f.type == "as_float":
                    pass
                else:
                    raise AssertionError(f"Unknown sym_float type: {f}")
            elif arg.type in (
                "as_none",
                "as_bool",
                "as_int",
                "as_float",
                "as_string",
                "as_custom_obj",
            ):
                return
            else:
                raise AssertionError(f"Unknown input type: {arg}")
        elif spec.type == "parameter":
            t = spec.parameter.arg
            t.name = replace_table[t.name]
        elif spec.type == "buffer":
            t = spec.buffer.arg
            t.name = replace_table[t.name]
        elif spec.type == "tensor_constant":
            t = spec.tensor_constant.arg
            t.name = replace_table[t.name]
        elif spec.type == "custom_obj":
            t_custom_obj = spec.custom_obj.arg
            t_custom_obj.name = replace_table[t_custom_obj.name]
            return
        elif spec.type == "token":
            tok = spec.token.arg
            tok.name = replace_table[tok.name]
        elif spec.type == "constant_input":
            return
        else:
            raise AssertionError(f"Unknown input type: {spec}")

    def replace_output(out):
        if not isinstance(spec, OutputSpec):
            raise AssertionError(f"expected OutputSpec, got {type(spec).__name__}")
        if spec.type == "user_output":
            arg = spec.user_output.arg
            if arg.type == "as_tensor":
                t = arg.as_tensor
                t.name = replace_table[t.name]
            elif arg.type == "as_sym_int":
                s = arg.as_sym_int
                if s.type == "as_name":
                    s.as_name = replace_table[s.as_name]
                elif s.type == "as_int":
                    pass
                else:
                    raise AssertionError(f"Unknown sym_int type: {s}")
            elif arg.type == "as_sym_float":
                f = arg.as_sym_float
                if f.type == "as_name":
                    f.as_name = replace_table[f.as_name]
                elif f.type == "as_float":
                    pass
                else:
                    raise AssertionError(f"Unknown sym_float type: {f}")
            elif arg.type in ("as_none", "as_bool", "as_int", "as_float", "as_string"):
                return
            else:
                raise AssertionError(f"Unknown input type: {arg}")
        elif spec.type == "loss_output":
            t = spec.loss_output.arg
            t.name = replace_table[t.name]
        elif spec.type == "buffer_mutation":
            t = spec.buffer_mutation.arg
            t.name = replace_table[t.name]
        elif spec.type == "parameter_mutation":
            t = spec.parameter_mutation.arg
            t.name = replace_table[t.name]
        elif spec.type == "gradient_to_parameter":
            t = spec.gradient_to_parameter.arg
            t.name = replace_table[t.name]
        elif spec.type == "gradient_to_user_input":
            g = spec.gradient_to_user_input
            g.arg.name = replace_table[g.arg.name]
            g.user_input_name = replace_table[g.user_input_name]
        elif spec.type == "user_input_mutation":
            u = spec.user_input_mutation
            u.arg.name = replace_table[u.arg.name]
            u.user_input_name = replace_table[u.user_input_name]
        elif spec.type == "token":
            tok = spec.token.arg
            tok.name = replace_table[tok.name]
        else:
            raise AssertionError(f"Unknown output type: {spec}")

    for spec in input_specs:
        replace_input(spec)

    for spec in output_specs:
        replace_output(spec)

    return ExportedProgram(
        graph_module=GraphModule(
            graph=sorted_graph,
            signature=GraphSignature(
                input_specs=list(input_specs),
                output_specs=list(output_specs),
            ),
            module_call_graph=module_call_graph,
        ),
        opset_version=opset_version,
        range_constraints=range_constraints,
        schema_version=ep.schema_version,
        verifiers=ep.verifiers,
        torch_version=ep.torch_version,
        guards_code=guards_code,
    )


class ExtensionHandler:
    """
    Base class for handling extension operators.
    """

    @classmethod
    def namespace(cls) -> str:
        raise NotImplementedError(f"{cls.__class__} namespace() must be implemented")

    @classmethod
    def to_op_name(cls, op) -> str:
        raise NotImplementedError(f"{cls.__class__} op_name() must be implemented")

    @classmethod
    def from_op_name(cls, name: str):
        raise NotImplementedError(f"{cls.__class__} op_name() must be implemented")

    @classmethod
    def op_schema(cls, op) -> torch.FunctionSchema:
        raise NotImplementedError(f"{cls.__class__} op_schema() must be implemented")


def register_extension(
    op_type: type[Any],
    extension_handler: type[ExtensionHandler],
):
    """Register custom de/serialization method for a node with non-standard type."""
    if not issubclass(extension_handler, ExtensionHandler):
        raise AssertionError(f"Expected ExtensionHandler, got {extension_handler}.")
    if op_type in _serialization_registry:
        raise AssertionError(f"{op_type} is already registered.")
    if not isinstance(op_type, type):
        raise AssertionError(f"op_type must be a type, got {type(op_type).__name__}")
    if op_type.__module__.startswith("torch") or op_type.__module__.startswith(
        "builtins"
    ):
        raise AssertionError(
            f"op_type module {op_type.__module__} should not start with 'torch' or 'builtins'"
        )
    if extension_handler.namespace() in _deserialization_registry:
        raise AssertionError(
            f"namespace {extension_handler.namespace()!r} is already registered"
        )
    _serialization_registry[op_type] = extension_handler
    _deserialization_registry[extension_handler.namespace()] = extension_handler


def _registered_extension_types():
    return tuple(_serialization_registry.keys())


# Registry to store all custom serialization implementations.
# The registry maps an operation to its serialization function (a callable), in their own
# namespace to avoid conflicts.
# Serialization: Op type --> custom handler.
# De-serialization: Namespace --> custom handler.
_serialization_registry: dict[type[Any], type[ExtensionHandler]] = {}
_deserialization_registry: dict[str, type[ExtensionHandler]] = {}
<|endoftext|>import dataclasses
from typing import Any

import torch
from torch._dynamo.exc import UserError, UserErrorType
from torch.export.dynamic_shapes import (
    _check_dynamic_shapes,
    _DerivedDim,
    _DimHint,
    _tree_map_with_path,
    Dim,
)
from torch.utils._pytree import tree_map

from .serialize import _dataclass_to_dict


@dataclasses.dataclass
class RootDim:
    """
    This represents a Dim object.
    """

    min: int
    max: int | None
    derived: list[str]


@dataclasses.dataclass
class DynamicShapesSpec:
    """
    This stores a dynamic_shapes spec for de/serialization.
    """

    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None
    dims: dict[str, RootDim]


def _postprocess_serialized_shapes(
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
    dims: dict[str, dict[str, int | list[str] | None]],
    to_dict: bool | None = False,
) -> DynamicShapesSpec | dict[str, Any]:
    """
    Sorts dims and dumps to dictionary format.
    """
    from torch.utils._sympy.numbers import int_oo

    # pyrefly: ignore [bad-assignment]
    dims = {
        k: RootDim(
            min=v["min"],  # type: ignore[arg-type]
            max=None if v["max"] is int_oo else v["max"],  # type: ignore[arg-type]
            derived=sorted(v["derived"]),  # type: ignore[arg-type]
        )
        for k, v in sorted(dims.items())
    }
    # pyrefly: ignore [bad-argument-type]
    spec = DynamicShapesSpec(dynamic_shapes=dynamic_shapes, dims=dims)
    if to_dict:
        return _dataclass_to_dict(spec)
    else:
        return spec


def _dump_dynamic_shapes(
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
    args: tuple[Any],
    kwargs: dict[str, Any] | None = None,
    to_dict: bool | None = False,
) -> DynamicShapesSpec | dict[str, Any]:
    """
    Utility function for dynamic shapes serialization, serializing a dynamic_shapes spec.
    Returns a DynamicShapesSpec dataclass containing 2 fields, "dynamic_shapes" and "dims".
    Uses args & kwargs to distinguish between tensor-level and dim-level specs (only for Nones).

    dynamic_shapes: A pytree structure mirroring the dynamic_shapes input to export():
        - Each tensor input is represented with a list of values, non-tensor inputs with None.
        - dynamic dimensions (i.e. symbols) in tensors and Dim enums are represented with strings.
        - static dimensions are represented with ints.

    dims: A dictionary mapping each symbol name to the min/max range and derived dim names.

    For example:
    ```
    dx = Dim("dx", min=4, max=16)
    dy = dx + 1

    inputs = (
        [
            torch.randn(4, 4),
            torch.randn(5, 4),
        ],
        torch.randn(4),
        torch.randn(4, 4),
        "hello",
    )
    dynamic_shapes = {
        "a": [
            (dx, 4),
            (dy, 4),
        ],
        "b": (Dim.STATIC,),
        "c": None,
        "d": None,
    }
    out = _dump_dynamic_shapes(dynamic_shapes, inputs, to_dict=True)
    ```
    would generate the following output:
    ```
    {
        "dynamic_shapes": (
            [
                ["dx", 4],
                ["dx + 1", 4],
            ],
            ["_DimHint.STATIC"],
            ["_DimHint.STATIC", "_DimHint.STATIC"],
            None,
        ),
        "dims": {
            "dx": {
                "min": 4,
                "max": 16,
                "derived": ["dx + 1"],
            },
        },
    }
    ```
    """
    dims: dict[str, dict[str, Any]] = {}

    def _standardize_shapes(path, tensor, shape):  # type: ignore[no-untyped-def]
        """
        Helps standardize the dynamic_shapes tree structure we serialize,
        returning lists for each tensor shape, handling tensor-level Nones.
        """
        if not isinstance(tensor, torch.Tensor):
            return None
        if shape is None:
            return [Dim.STATIC] * len(tensor.shape)

        out = []
        if isinstance(shape, dict):
            for i, s in enumerate(tensor.shape):
                out.append(s if shape.get(i) is None else shape.get(i))
        else:
            if not isinstance(shape, (tuple, list)):
                raise AssertionError(f"expected tuple or list, got {type(shape)}")
            for i, s in enumerate(tensor.shape):
                out.append(s if shape[i] is None else shape[i])
        return out

    def _track_dim_from_dims(
        val: None | int | _DimHint | Dim,
    ) -> None | int | str:
        """
        Tracks dims, ranges, derived dims from the standardized dynamic_shapes spec.
        """
        if val is None or isinstance(val, int):  # non-tensor input or static
            return val
        if isinstance(val, _DimHint):  # store enum as string
            return val.__class__.__name__ + "." + val.type.name

        if not isinstance(val, Dim):
            raise AssertionError(f"expected Dim, got {type(val)}")

        # track root dim
        root = val.root if isinstance(val, _DerivedDim) else val  # type: ignore[attr-defined]
        if root.__name__ not in dims:
            dims[root.__name__] = {
                "min": root.min,  # type: ignore[attr-defined,union-attr]
                "max": root.max,  # type: ignore[attr-defined,union-attr]
                "derived": set(),
            }

        # track derived dims
        if isinstance(val, _DerivedDim):
            dims[root.__name__]["derived"].add(val.__name__)

        return val.__name__

    if dynamic_shapes is None:
        return {"dynamic_shapes": None, "dims": {}}

    # convert to tuple of specs, for each arg/kwarg
    kwargs = kwargs or {}
    if isinstance(dynamic_shapes, dict):
        dynamic_shapes = dynamic_shapes.values()  # type: ignore[assignment]
    # pyrefly: ignore [bad-assignment, bad-argument-type]
    dynamic_shapes = tuple(dynamic_shapes)
    combined_args = tuple(args) + tuple(kwargs.values())

    # run same check when we're processing shapes for export - is this too lazy?
    _check_dynamic_shapes(dict(enumerate(combined_args)), dynamic_shapes)  # type: ignore[arg-type]

    tree_shapes = _tree_map_with_path(
        _standardize_shapes, combined_args, dynamic_shapes, tree_name="inputs"
    )
    serialized_shapes = tree_map(_track_dim_from_dims, tree_shapes)
    return _postprocess_serialized_shapes(serialized_shapes, dims, to_dict=to_dict)


def _load_dynamic_shapes(
    spec: DynamicShapesSpec | dict[str, Any],
    from_dict: bool | None = False,
) -> dict[str, Any] | tuple[Any] | list[Any] | None:
    """
    Utility function for dynamic shapes serialization.
    Deserializes a DynamicShapesSpec or corresponding dictionary into a dynamic_shapes input to export().
    """
    import sympy

    from torch.fx.experimental.symbolic_shapes import _is_supported_equivalence

    if from_dict:
        if not isinstance(spec, dict):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"With from_dict=True, expected `spec` to be a dict, got {type(spec)}",
            )
        if sorted(spec.keys()) != ["dims", "dynamic_shapes"]:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "With from_dict=True, expected `spec` to have keys `dims` and `dynamic_shapes`, "
                f"instead found {spec.keys()}",
            )
        dims = {}
        for k, v in spec["dims"].items():
            if not isinstance(k, str):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected `spec['dims']` keys to be strings for symbols, got key {type(k)}",
                )
            if sorted(v.keys()) != ["derived", "max", "min"]:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected `spec['dims']` values to have keys `derived`, `max`, and `min`, "
                    f"instead found {v.keys()}",
                )
            if not isinstance(v["min"], int):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dims in `spec['dims']` to map `min` to an int, got {k}: {v['min']}",
                )
            if not isinstance(v["max"], int) or v["max"] is None:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dims in `spec['dims']` to map `max` to an int or None, got {k}: {v['max']}",
                )
            if not isinstance(v["derived"], list) or any(
                not isinstance(d, str) for d in v["derived"]
            ):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    "Expected dims in `spec['dims']` to map `derived` to a list of derived expressions, "
                    f"got {k}: {v['derived']}",
                )
            dims[k] = RootDim(**v)
        dynamic_shapes = spec["dynamic_shapes"]
    else:
        if not isinstance(spec, DynamicShapesSpec):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"Expected `spec` to be a DynamicShapesSpec, got {type(spec)}",
            )
        dims = spec.dims
        dynamic_shapes = spec.dynamic_shapes

    if dynamic_shapes is None:
        return None

    dim_cache = {}
    for name, info in dims.items():
        symbol = sympy.sympify(name)
        if not isinstance(symbol, sympy.Symbol):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"Expected `spec['dims']` keys to be symbols, got {name}",
            )
        dim_cache[name] = Dim(name, min=info.min, max=info.max)  # cache root dim
        for _expr in info.derived:
            expr = sympy.sympify(_expr)
            if len(expr.free_symbols) != 1 or symbol not in expr.free_symbols:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected derived expressions in to have {name} as the only free symbol, got {expr}",
                )
            if not _is_supported_equivalence(expr):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected derived expressions to be linear expressions, got {expr}",
                )
            modulus, remainder = sympy.polys.polytools.div(expr, symbol)
            ddim = dim_cache[name]
            if modulus != 1:
                ddim = int(modulus) * ddim  # type: ignore[assignment, operator]
            if remainder != 0:
                ddim = ddim + int(remainder)  # type: ignore[assignment, operator]
            dim_cache[_expr] = ddim  # cache derived dims

    def deserialize_shape(
        val: None | int | str,
    ) -> None | int | Dim | _DimHint:
        if val is None or isinstance(val, int):
            return val
        elif val == "_DimHint.AUTO":
            return _DimHint.AUTO()
        elif val == "_DimHint.DYNAMIC":
            return _DimHint.DYNAMIC()
        elif val == "_DimHint.STATIC":
            return _DimHint.STATIC()
        if not isinstance(val, str):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "Expected leaves in `spec['dynamic_shapes']` to be ints, None, Dim.AUTO/STATIC, symbols, "
                f" or derived expressions, got {val}",
            )
        if val not in dim_cache:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "Expected dims in `spec['dynamic_shapes']` to be tracked in `spec['dims']`, "
                f"got {val} which is not in {dims.keys()}",
            )
        return dim_cache[val]  # type: ignore[return-value]

    return tree_map(deserialize_shape, dynamic_shapes)
<|endoftext|>import functools
from collections.abc import Hashable
from dataclasses import dataclass, fields
from typing import Any, TypeVar
from typing_extensions import dataclass_transform


T = TypeVar("T", bound="_Union")


class _UnionTag(str):
    __slots__ = ("_cls",)
    _cls: Hashable

    @staticmethod
    def create(t: str, cls: type["_Union"]) -> "_UnionTag":
        tag = _UnionTag(t)
        if hasattr(tag, "_cls"):
            raise AssertionError("tag already has _cls attribute")
        tag._cls = cls
        return tag

    def __eq__(self, cmp: object) -> bool:
        if not isinstance(cmp, str):
            raise AssertionError(f"expected str, got {type(cmp)}")
        other = str(cmp)
        if other not in _get_field_names(self._cls):
            raise AssertionError(
                f"{other} is not a valid tag for {self._cls}. Available tags: {_get_field_names(self._cls)}"
            )
        return str(self) == other

    def __hash__(self) -> int:
        return hash(str(self))


@functools.cache
def _get_field_names(cls: type["_Union"]) -> set[str]:
    # pyrefly: ignore[bad-argument-type]  # TODO _Union subclasses are dataclasses at runtime via _union_dataclass
    return {f.name for f in fields(cls)}


# If you turn a schema class that inherits from union into a dataclass, please use
# this decorator to configure it. It's safe, faster and allows code sharing.
#
# For example, _union_dataclass customizes the __eq__ method to only check the type
# and value property instead of default implementation of dataclass which goes
# through every field in the dataclass.
@dataclass_transform(eq_default=False)
def _union_dataclass(cls: type[T]) -> type[T]:
    if not issubclass(cls, _Union):
        raise AssertionError(f"{cls} must inherit from {_Union}.")
    return dataclass(repr=False, eq=False)(cls)


class _Union:
    _type: _UnionTag

    @classmethod
    def create(cls: type[T], **kwargs: object) -> T:
        if len(kwargs) != 1:
            raise AssertionError(f"expected exactly 1 kwarg, got {len(kwargs)}")
        obj = cls(**{**{f.name: None for f in fields(cls)}, **kwargs})  # type: ignore[arg-type]
        obj._type = _UnionTag.create(next(iter(kwargs.keys())), cls)
        return obj

    def __post_init__(self) -> None:
        if any(
            f.name in ("type", "_type", "create", "value")
            for f in fields(self)  # type: ignore[arg-type, misc]
        ):
            raise AssertionError(
                "field names 'type', '_type', 'create', 'value' are reserved"
            )

    @property
    def type(self) -> str:
        try:
            return self._type
        except AttributeError as e:
            raise RuntimeError(
                f"Please use {type(self).__name__}.create to instantiate the union type."
            ) from e

    @property
    def value(self) -> Any:
        return getattr(self, self.type)

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if attr is None and name in _get_field_names(type(self)) and name != self.type:
            raise AttributeError(f"Field {name} is not set.")
        return attr

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Union):
            return False
        return self.type == other.type and self.value == other.value

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.type}={getattr(self, self.type)})"
<|endoftext|># NOTE: This is a placeholder for iterating on export serialization schema design.
#       Anything is subject to change and no guarantee is provided at this point.

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Annotated

from torch._export.serde.union import _Union, _union_dataclass


# NOTE: Please update this value if any modifications are made to the schema
SCHEMA_VERSION = (8, 20)
TREESPEC_VERSION = 1


# NOTE: If you updated the schema, please run `scripts/export/update_schema.py`
# to update the auto generated files.
#
# There are also mappings from serialized enum values to c10 enum member names.
# These are used by scripts/export/update_schema.py to generate C++ conversion
# functions for oss_proxy_executor.cpp.
# When adding new enum values, update both the enum AND the mapping.
class ScalarType(IntEnum):
    UNKNOWN = 0
    BYTE = 1
    CHAR = 2
    SHORT = 3
    INT = 4
    LONG = 5
    HALF = 6
    FLOAT = 7
    DOUBLE = 8
    COMPLEXHALF = 9
    COMPLEXFLOAT = 10
    COMPLEXDOUBLE = 11
    BOOL = 12
    BFLOAT16 = 13
    UINT16 = 28
    FLOAT8E4M3FN = 29
    FLOAT8E5M2 = 30
    FLOAT8E4M3FNUZ = 31
    FLOAT8E5M2FNUZ = 32
    FLOAT8E8M0FNU = 33
    UINT32 = 34
    UINT64 = 35


class Layout(IntEnum):
    Unknown = 0
    SparseCoo = 1
    SparseCsr = 2
    SparseCsc = 3
    SparseBsr = 4
    SparseBsc = 5
    _mkldnn = 6
    Strided = 7


class MemoryFormat(IntEnum):
    Unknown = 0
    ContiguousFormat = 1
    ChannelsLast = 2
    ChannelsLast3d = 3
    PreserveFormat = 4


SCALAR_TYPE_TO_C10: dict[int, str] = {
    ScalarType.BYTE: "Byte",
    ScalarType.CHAR: "Char",
    ScalarType.SHORT: "Short",
    ScalarType.INT: "Int",
    ScalarType.LONG: "Long",
    ScalarType.HALF: "Half",
    ScalarType.FLOAT: "Float",
    ScalarType.DOUBLE: "Double",
    ScalarType.COMPLEXHALF: "ComplexHalf",
    ScalarType.COMPLEXFLOAT: "ComplexFloat",
    ScalarType.COMPLEXDOUBLE: "ComplexDouble",
    ScalarType.BOOL: "Bool",
    ScalarType.BFLOAT16: "BFloat16",
    ScalarType.UINT16: "UInt16",
    ScalarType.FLOAT8E4M3FN: "Float8_e4m3fn",
    ScalarType.FLOAT8E5M2: "Float8_e5m2",
    ScalarType.FLOAT8E4M3FNUZ: "Float8_e4m3fnuz",
    ScalarType.FLOAT8E5M2FNUZ: "Float8_e5m2fnuz",
    ScalarType.FLOAT8E8M0FNU: "Float8_e8m0fnu",
    ScalarType.UINT32: "UInt32",
    ScalarType.UINT64: "UInt64",
}

LAYOUT_TO_C10: dict[int, str] = {
    Layout.SparseCoo: "Sparse",
    Layout.SparseCsr: "SparseCsr",
    Layout.SparseCsc: "SparseCsc",
    Layout.SparseBsr: "SparseBsr",
    Layout.SparseBsc: "SparseBsc",
    Layout._mkldnn: "Mkldnn",
    Layout.Strided: "Strided",
}

MEMORY_FORMAT_TO_C10: dict[int, str] = {
    MemoryFormat.ContiguousFormat: "Contiguous",
    MemoryFormat.ChannelsLast: "ChannelsLast",
    MemoryFormat.ChannelsLast3d: "ChannelsLast3d",
    MemoryFormat.PreserveFormat: "Preserve",
}


@dataclass
class Device:
    type: Annotated[str, 10]
    index: Annotated[int | None, 20] = None


@_union_dataclass
class SymExprHint(_Union):
    as_int: Annotated[int, 10]
    as_bool: Annotated[bool, 20]
    as_float: Annotated[float, 30]


# This is for storing the symbolic expressions behind symints/symfloats/symbools
# For example, we can get something like
# SymExpr(expr_str="s0 + s1", hint=SymExprHint(as_int=4)
# if we also have the hint that s0 and s1 are both 2.
@dataclass
class SymExpr:
    expr_str: Annotated[str, 10]
    hint: Annotated[SymExprHint | None, 20] = None


@_union_dataclass
class SymInt(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_int: Annotated[int, 20]


@_union_dataclass
class SymFloat(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_float: Annotated[float, 20]


@_union_dataclass
class SymBool(_Union):
    as_expr: Annotated[SymExpr, 10]
    as_bool: Annotated[bool, 20]


@dataclass
class TensorMeta:
    dtype: Annotated[ScalarType, 10]
    sizes: Annotated[list[SymInt], 20]
    requires_grad: Annotated[bool, 30]
    device: Annotated[Device, 40]
    strides: Annotated[list[SymInt], 50]
    storage_offset: Annotated[SymInt, 60]
    layout: Annotated[Layout, 70]


# In most cases we will use the "as_name" field to store arguments which are
# SymInts.
# The "as_int" field is used in the case where we have a list containing a mix
# of SymInt and ints (ex. [1, s0, ...]). We will serialize this type of list to
# be List[SymIntArgument] and map the SymInts to the "as_name" field, and ints
# to the "as_int" field.
@_union_dataclass
class SymIntArgument(_Union):
    as_name: Annotated[str, 10]
    as_int: Annotated[int, 20]


# In most cases we will use the "as_name" field to store arguments which are
# SymFloats.
# The "as_float" field is used in the case where we have a list containing a mix
# of SymFloat and float (ex. [1.0, s0, ...]). We will serialize this type of list to
# be List[SymFloatArgument] and map the SymFloats to the "as_name" field, and ints
# to the "as_float" field.
@_union_dataclass
class SymFloatArgument(_Union):
    as_name: Annotated[str, 10]
    as_float: Annotated[float, 20]


# In most cases we will use the "as_name" field to store arguments which are
# SymBools.
# The "as_bool" field is used in the case where we have a list containing a mix
# of SymBool and bools (ex. [True, i0, ...]). We will serialize this type of list to
# be List[SymBoolArgument] and map the SymBools to the "as_name" field, and bools
# to the "as_bool" field.
@_union_dataclass
class SymBoolArgument(_Union):
    as_name: Annotated[str, 10]
    as_bool: Annotated[bool, 20]


@dataclass
class TensorArgument:
    name: Annotated[str, 10]


@dataclass
class TokenArgument:
    name: Annotated[str, 10]


# This is use for storing the contents of a list which contain optional tensors
# (Tensor?[], ex. [Tensor, None, ...]), where the list will be serialized to the
# type List[OptionalTensorArgument], with tensor values serialized to the
# "as_tensor" field, and None values serialized to the "as_none" field.
@_union_dataclass
class OptionalTensorArgument(_Union):
    as_tensor: Annotated[TensorArgument, 20]
    as_none: Annotated[bool, 10]


@dataclass
class GraphArgument:
    name: Annotated[str, 10]
    graph: Annotated["Graph", 20]


@dataclass
class CustomObjArgument:
    name: Annotated[str, 10]
    class_fqn: Annotated[str, 20]


@dataclass
class ComplexValue:
    real: Annotated[float, 10]
    imag: Annotated[float, 20]


# This is actually a union type
@_union_dataclass
class Argument(_Union):
    as_none: Annotated[bool, 10]
    as_tensor: Annotated[TensorArgument, 20]
    as_tensors: Annotated[list[TensorArgument], 30]
    as_int: Annotated[int, 50]
    as_ints: Annotated[list[int], 70]
    as_float: Annotated[float, 80]
    as_floats: Annotated[list[float], 90]
    as_string: Annotated[str, 100]
    as_strings: Annotated[list[str], 101]
    as_sym_int: Annotated[SymIntArgument, 110]
    as_sym_ints: Annotated[list[SymIntArgument], 120]
    as_scalar_type: Annotated[ScalarType, 130]
    as_memory_format: Annotated[MemoryFormat, 140]
    as_layout: Annotated[Layout, 150]
    as_device: Annotated[Device, 160]
    as_bool: Annotated[bool, 170]
    as_bools: Annotated[list[bool], 180]
    as_sym_bool: Annotated[SymBoolArgument, 182]
    as_sym_bools: Annotated[list[SymBoolArgument], 184]
    as_graph: Annotated[GraphArgument, 200]
    as_optional_tensors: Annotated[list[OptionalTensorArgument], 190]
    as_custom_obj: Annotated[CustomObjArgument, 210]
    as_operator: Annotated[str, 220]
    as_sym_float: Annotated[SymFloatArgument, 230]
    as_sym_floats: Annotated[list[SymFloatArgument], 240]
    as_optional_tensor: Annotated[OptionalTensorArgument, 250]
    as_complex: Annotated[ComplexValue, 260]
    as_nested_tensors: Annotated[list[list[TensorArgument]], 270]
    as_int_lists: Annotated[list[list[int]], 280]
    as_string_to_argument: Annotated[dict[str, "Argument"], 290]
    as_float_lists: Annotated[list[list[float]], 300]


class ArgumentKind(IntEnum):
    UNKNOWN = 0
    POSITIONAL = 1
    KEYWORD = 2


@dataclass
class NamedArgument:
    # Argument name from the operator schema
    name: Annotated[str, 10]
    arg: Annotated[Argument, 20]
    kind: Annotated[ArgumentKind | None, 30] = None


@dataclass
class Node:
    target: Annotated[str, 10]
    inputs: Annotated[list[NamedArgument], 20]
    outputs: Annotated[list[Argument], 30]
    metadata: Annotated[dict[str, str], 40]
    is_hop_single_tensor_return: Annotated[bool | None, 50] = None
    # For BC, default is None so older serialized models without 'name' can be loaded.
    name: Annotated[str | None, 60] = None


@dataclass
class Graph:
    inputs: Annotated[list[Argument], 10]
    outputs: Annotated[list[Argument], 20]
    nodes: Annotated[list[Node], 30]
    tensor_values: Annotated[dict[str, TensorMeta], 40]
    sym_int_values: Annotated[dict[str, SymInt], 50]
    sym_bool_values: Annotated[dict[str, SymBool], 60]
    # This is for deserializing the submodule graphs from higher order ops
    # (ex. cond, map) where single tensor returns will just return a single
    # tensor, rather than following export schema and returning a singleton
    # list.
    is_single_tensor_return: Annotated[bool, 70] = False
    custom_obj_values: Annotated[dict[str, CustomObjArgument], 80] = field(
        default_factory=dict
    )
    sym_float_values: Annotated[dict[str, SymFloat], 90] = field(default_factory=dict)


@dataclass
class UserInputSpec:
    # Actually, only tensors and SymInts are allowed here
    arg: Annotated[Argument, 10]


@_union_dataclass
class ConstantValue(_Union):
    as_none: Annotated[bool, 10]
    as_int: Annotated[int, 20]
    as_float: Annotated[float, 30]
    as_string: Annotated[str, 40]
    as_bool: Annotated[bool, 50]


@dataclass
class InputToConstantInputSpec:
    name: Annotated[str, 10]
    value: Annotated[ConstantValue, 20]


@dataclass
class InputToParameterSpec:
    arg: Annotated[TensorArgument, 10]
    parameter_name: Annotated[str, 20]


@dataclass
class InputToBufferSpec:
    arg: Annotated[TensorArgument, 10]
    buffer_name: Annotated[str, 20]
    persistent: Annotated[bool, 30]


@dataclass
class InputToTensorConstantSpec:
    arg: Annotated[TensorArgument, 10]
    tensor_constant_name: Annotated[str, 20]


@dataclass
class InputToCustomObjSpec:
    arg: Annotated[CustomObjArgument, 10]
    custom_obj_name: Annotated[str, 20]


@dataclass
class InputTokenSpec:
    arg: Annotated[TokenArgument, 10]


@_union_dataclass
class InputSpec(_Union):
    user_input: Annotated[UserInputSpec, 10]
    parameter: Annotated[InputToParameterSpec, 20]
    buffer: Annotated[InputToBufferSpec, 30]
    tensor_constant: Annotated[InputToTensorConstantSpec, 40]
    custom_obj: Annotated[InputToCustomObjSpec, 50]
    token: Annotated[InputTokenSpec, 70]
    constant_input: Annotated[InputToConstantInputSpec, 60]


@dataclass
class UserOutputSpec:
    arg: Annotated[Argument, 10]


@dataclass
class LossOutputSpec:
    arg: Annotated[TensorArgument, 10]


@dataclass
class BufferMutationSpec:
    arg: Annotated[TensorArgument, 10]
    buffer_name: Annotated[str, 20]


@dataclass
class ParameterMutationSpec:
    arg: Annotated[TensorArgument, 10]
    parameter_name: Annotated[str, 20]


@dataclass
class GradientToParameterSpec:
    arg: Annotated[TensorArgument, 10]
    parameter_name: Annotated[str, 20]


@dataclass
class GradientToUserInputSpec:
    arg: Annotated[TensorArgument, 10]
    user_input_name: Annotated[str, 20]


@dataclass
class UserInputMutationSpec:
    arg: Annotated[TensorArgument, 10]
    user_input_name: Annotated[str, 20]


@dataclass
class OutputTokenSpec:
    arg: Annotated[TokenArgument, 10]


@_union_dataclass
class OutputSpec(_Union):
    user_output: Annotated[UserOutputSpec, 10]
    loss_output: Annotated[LossOutputSpec, 20]
    buffer_mutation: Annotated[BufferMutationSpec, 30]
    gradient_to_parameter: Annotated[GradientToParameterSpec, 40]
    gradient_to_user_input: Annotated[GradientToUserInputSpec, 50]
    user_input_mutation: Annotated[UserInputMutationSpec, 60]
    token: Annotated[OutputTokenSpec, 70]
    parameter_mutation: Annotated[ParameterMutationSpec, 80]


@dataclass
class GraphSignature:
    input_specs: Annotated[list[InputSpec], 10]
    output_specs: Annotated[list[OutputSpec], 20]


@dataclass
class RangeConstraint:
    min_val: Annotated[int | None, 10]
    max_val: Annotated[int | None, 20]


@dataclass
class ModuleCallSignature:
    inputs: Annotated[list[Argument], 10]
    outputs: Annotated[list[Argument], 20]

    # These are serialized by calling pytree.treespec_loads
    # And deserialized by calling pytree.treespec_dumps
    in_spec: Annotated[str, 30]
    out_spec: Annotated[str, 40]

    # This field is used to prettify the graph placeholders
    # after we Ser/Der and retrace
    forward_arg_names: Annotated[list[str] | None, 50] = None


@dataclass
class ModuleCallEntry:
    fqn: Annotated[str, 10]
    signature: Annotated[ModuleCallSignature | None, 30] = None


@dataclass
class NamedTupleDef:
    field_names: Annotated[list[str], 10]


@dataclass
class GraphModule:
    graph: Annotated[Graph, 10]
    signature: Annotated[GraphSignature, 50]
    # This is used for unflattening, by tracking the calling structure of all of
    # the modules in order to unflatten the modules back to the eager calling
    # conventions.
    module_call_graph: Annotated[list[ModuleCallEntry], 60]
    metadata: Annotated[dict[str, str], 40] = field(default_factory=dict)
    # Mapping of namedtuple types to namedtuple field names, used for BC
    treespec_namedtuple_fields: Annotated[dict[str, NamedTupleDef], 70] = field(
        default_factory=dict
    )


# Invariant: Every time a change is made to the schema, one of the versions
#            should be updated.
@dataclass
class SchemaVersion:
    major: Annotated[
        int, 10
    ]  # Major version number is bumped every time a breaking change is made.
    minor: Annotated[
        int, 20
    ]  # Minor version number is bumped when a compatible change is made.


@dataclass
class ExportedProgram:
    graph_module: Annotated[GraphModule, 10]
    # Key is the opset namespace (ex. aten), and value is the version number
    opset_version: Annotated[dict[str, int], 20]
    range_constraints: Annotated[dict[str, RangeConstraint], 30]
    schema_version: Annotated[SchemaVersion, 60]
    verifiers: Annotated[list[str], 70] = field(default_factory=list)
    torch_version: Annotated[str, 80] = "<=2.4"
    guards_code: Annotated[list[str], 90] = field(default_factory=list)


#########################################################################
# Container types for inference tasks, not being used directly for export.
#########################################################################


# The metadata for payload saved in PT2 archive.
# payload includes params, buffers, tensor constants, and custom objects.
@dataclass
class PayloadMeta:
    # the path of the payload in the archive file, e.g. "weight_0"
    path_name: Annotated[str, 10]
    is_param: Annotated[bool, 20]
    # whether the payload is serialized using pickle.
    # Only custom objects and tensor subclasses that are not fake tensors
    # are serialized using pickle.
    use_pickle: Annotated[bool, 30]
    # Custom Objects don't have tensor_meta and will be serialized using pickle
    tensor_meta: Annotated[TensorMeta | None, 40]


# The mapping from payload FQN to its metadata.
@dataclass
class PayloadConfig:
    config: Annotated[dict[str, PayloadMeta], 10]


#
# The structure is used to serialize instances of AOTInductorModel to pass
# them from the publishing pipeline to the predictor.
#
# All new fields should be marked as optional.
#
@dataclass
class AOTInductorModelPickleData:
    # Base name of an associated .so AOTInductor library. Typically looks like:
    # "abc.so".
    library_basename: Annotated[str, 1]

    # AOTInductor engine input names.
    input_names: Annotated[list[str], 2]

    # AOTInductor engine output names.
    output_names: Annotated[list[str], 3]

    # These fields tell whether floating point inputs/outputs should be converted to
    # a certain type. If None, the dtypes that the AOTInductor engine inferred from the sample
    # inputs are used.
    floating_point_input_dtype: Annotated[int | None, 4] = None
    floating_point_output_dtype: Annotated[int | None, 5] = None

    # Whether AOTInductor runtime is for CPU.
    aot_inductor_model_is_cpu: Annotated[bool | None, 6] = None


@dataclass
class ExternKernelNode:
    # name is not the unique identifier of the node
    name: Annotated[str, 10]
    node: Annotated[Node, 20]


@dataclass
class ExternKernelNodes:
    nodes: Annotated[list[ExternKernelNode], 10]
<|endoftext|>import dataclasses
import hashlib
import inspect
import re
import types
import typing
from enum import IntEnum
from typing import Annotated, Any, ForwardRef, Union

from torch._export.serde import schema
from torch._export.serde.union import _Union


class SchemaUpdateError(Exception):
    pass


def _check(x: object, msg: str) -> None:
    if not x:
        raise SchemaUpdateError(msg)


_CPP_TYPE_MAP = {
    str: "std::string",
    int: "int64_t",
    float: "F64",
    bool: "bool",
}

_THRIFT_TYPE_MAP = {
    str: "string",
    int: "i64",
    float: "double",
    bool: "bool",
}


def _staged_schema() -> tuple[dict[str, Any], str, str]:
    yaml_ret: dict[str, Any] = {}
    defs: dict[str, Any] = {}
    cpp_enum_defs: dict[str, str] = {}
    cpp_class_defs: dict[str, str] = {}
    cpp_type_decls: list[str] = []
    cpp_json_defs: list[str] = []
    thrift_enum_defs: list[str] = []
    thrift_type_defs: dict[str, str] = {}

    def _handle_aggregate(
        ty: type[Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        def dump_type(t: Any, level: int) -> tuple[str, str, str]:
            if getattr(t, "__name__", None) in cpp_enum_defs:
                return t.__name__, "int64_t", t.__name__
            elif t in _CPP_TYPE_MAP:
                return (t.__name__, _CPP_TYPE_MAP[t], _THRIFT_TYPE_MAP[t])
            elif isinstance(t, str):
                if t not in defs:
                    raise AssertionError(f"type {t} not in defs")
                if t in cpp_enum_defs:
                    raise AssertionError(f"type {t} unexpectedly in cpp_enum_defs")
                if "[" in t:
                    raise AssertionError(f"type {t} contains '[' which is not allowed")
                return t, f"ForwardRef<{t}>", t
            elif isinstance(t, ForwardRef):
                return (
                    t.__forward_arg__,
                    f"ForwardRef<{t.__forward_arg__}>",
                    t.__forward_arg__,
                )
            elif o := typing.get_origin(t):
                # Lemme know if there's a better way to do this.
                if o is list:
                    yaml_head, cpp_head, thrift_head, thrift_tail = (
                        "List",
                        "std::vector",
                        "list<",
                        ">",
                    )
                elif o is dict:
                    yaml_head, cpp_head, thrift_head, thrift_tail = (
                        "Dict",
                        "std::unordered_map",
                        "map<",
                        ">",
                    )
                elif o is Union or o is types.UnionType:
                    if level != 0:
                        raise AssertionError(
                            f"Optional is only supported at the top level, got level={level}"
                        )
                    args = typing.get_args(t)
                    if len(args) != 2 or args[1] is not type(None):
                        raise AssertionError(
                            f"expected Optional type with 2 args ending in None, got {args}"
                        )
                    yaml_type, cpp_type, thrift_type = dump_type(args[0], level + 1)
                    return (
                        f"Optional[{yaml_type}]",
                        f"std::optional<{cpp_type}>",
                        f"optional {thrift_type}",
                    )
                elif o is Annotated:
                    return dump_type(t.__origin__, level)
                else:
                    raise AssertionError(f"Type {t} is not supported in export schema.")
                yaml_arg_types, cpp_arg_types, thrift_arg_types = zip(
                    *[dump_type(x, level + 1) for x in typing.get_args(t)]
                )
                return (
                    (f"{yaml_head}[{', '.join(yaml_arg_types)}]"),
                    (f"{cpp_head}<{', '.join(cpp_arg_types)}>"),
                    f"{thrift_head}{', '.join(thrift_arg_types)}{thrift_tail}",
                )
            elif isinstance(t, type):
                return (t.__name__, t.__name__, t.__name__)
            else:
                raise AssertionError(f"Type {t} is not supported in export schema.")

        def dump_cpp_value(v: object) -> str:
            if v is None:
                return "std::nullopt"
            elif v is True:
                return "true"
            elif v is False:
                return "false"
            elif v == {}:
                return "{}"
            elif v == []:
                return "{}"
            elif v == ():
                return "{}"
            elif isinstance(v, str):
                return f'"{v}"'
            else:
                raise AssertionError(
                    f"Default value {v} is not supported yet in export schema."
                )

        def dump_field(
            f: dataclasses.Field[Any],
        ) -> tuple[dict[str, Any], str, str | None, str, int]:
            t, cpp_type, thrift_type = dump_type(f.type, 0)
            ret = {"type": t}
            cpp_default: str | None = None
            if typing.get_origin(f.type) is not Annotated:
                raise AssertionError(
                    f"Field {f.name} must be annotated with an integer id."
                )
            # pyrefly: ignore[missing-attribute]  # TODO f.type is an Annotated alias at runtime, not str
            thrift_id = f.type.__metadata__[0]
            if type(thrift_id) is not int:
                raise AssertionError(
                    f"Field {f.name} must be annotated with an integer id, got {type(thrift_id)}"
                )

            value = dataclasses.MISSING
            if f.default is not dataclasses.MISSING:
                value = f.default
            elif f.default_factory is not dataclasses.MISSING:
                value = f.default_factory()

            if value is not dataclasses.MISSING:
                default = str(value)
                ret["default"] = default
                cpp_default = dump_cpp_value(value)

                if t.startswith("Optional[") and value is not None:
                    raise AssertionError(
                        f"Optional field {ty.__name__}.{f.name} must have default value to be None."
                    )

                # Skip emitting "= {}" for non-primitive types whose
                # default-construction already yields the same empty value
                # (containers, strings, classes). Emitting it would trigger
                # readability-redundant-member-init. Keep it for primitives
                # (int64_t, bool, F64) which require explicit value-init.
                if cpp_default == "{}" and cpp_type not in ("int64_t", "bool", "F64"):
                    cpp_default = None
            elif cpp_type in ("int64_t", "bool", "F64"):
                # Value-initialize primitive/enum members (enums map to int64_t)
                # to satisfy cppcoreguidelines-pro-type-member-init. Non-primitive
                # types (std::string, containers, classes) default-construct
                # themselves; emitting "= {}" for those would trigger
                # readability-redundant-member-init.
                cpp_default = "{}"

            return ret, cpp_type, cpp_default, thrift_type, thrift_id

        yaml_ret = {}
        cpp_ret = {}
        thrift_ret = {}
        thrift_ids = set()
        for f in dataclasses.fields(ty):
            yaml_res, cpp_type, cpp_default, thrift_type, thrift_id = dump_field(f)
            yaml_ret[f.name] = yaml_res
            cpp_ret[f.name] = {"cpp_type": cpp_type, "cpp_default": cpp_default}
            thrift_ret[f.name] = {"thrift_type": thrift_type, "thrift_id": thrift_id}
            if thrift_id in thrift_ids:
                raise AssertionError(
                    f"Duplicate thrift id {thrift_id} for field {f.name} in {ty.__name__}."
                )
            thrift_ids.add(thrift_id)
        return yaml_ret, cpp_ret, thrift_ret

    def _handle_int_enum(name: str, ty: type[IntEnum]) -> None:
        yaml_ret[name] = {"kind": "enum", "fields": {x.name: x.value for x in ty}}
        cpp_enum_defs[name] = f"""
enum class {name} {{
{chr(10).join([f"  {x.name} = {x.value}," for x in ty])}
}};

inline std::string_view printEnum(const {name}& e) {{
  switch (e) {{
{chr(10).join([f"    case {name}::{x.name}: return {chr(34)}{x.name}{chr(34)};" for x in ty])}
    default:
      throw std::runtime_error("Unknown enum value");
  }}
}}

inline void parseEnum(std::string_view s, {name}& t) {{
{chr(10).join([f"  if (s == {chr(34)}{x.name}{chr(34)}) {{ t = {name}::{x.name}; return; }}" for x in ty])}
  throw std::runtime_error("Unknown enum value: " + std::string{{s}});
}}
"""
        thrift_enum_defs.append(
            f"""
enum {name} {{
{chr(10).join([f"  {x.name} = {x.value}," for x in ty])}
}}
"""
        )

    def _handle_struct(name: str, ty: type[Any]) -> None:
        fields, cpp_fields, thrift_fields = _handle_aggregate(ty)
        yaml_ret[name] = {"kind": "struct", "fields": fields}
        field_decls = "\n".join(
            f"  {f['cpp_type']} {name}{' = ' + f['cpp_default'] if f['cpp_default'] is not None else ''};"
            for name, f in cpp_fields.items()
        )

        def accessor(name: str, ty: str) -> str:
            type_name = fields[name]["type"]
            if type_name in cpp_enum_defs:
                return f"""
  {type_name} get_{name}() const {{
    return static_cast<{type_name}>({name});
  }}

  void set_{name}({type_name} def) {{
    {name} = static_cast<int64_t>(def);
  }}
"""
            return f"""
  const {ty}& get_{name}() const {{
    return {name};
  }}

  template <typename U>
  void set_{name}(U&& def) {{
    {name} = std::forward<U>(def);
  }}
"""

        to_json_decl = f"void to_json(nlohmann::json& nlohmann_json_j, const {name}& nlohmann_json_t)"
        to_json_def = f"""{{
{chr(10).join([f'  nlohmann_json_j["{name}"] = nlohmann_json_t.{name};' for name in cpp_fields])}
}}
"""
        from_json_decl = f"void from_json(const nlohmann::json& nlohmann_json_j, {name}& nlohmann_json_t)"

        from_json_def = f"""{{
  {name} nlohmann_json_default_obj;
{
            chr(10).join(
                [
                    f'  nlohmann_json_t.{name} = nlohmann_json_j.value("{name}", nlohmann_json_default_obj.{name});'
                    for name in cpp_fields
                ]
            )
        }
}}
"""
        cpp_class_defs[name] = f"""
class {name} {{
 private:
{field_decls}

 public:
{"".join([accessor(name, f["cpp_type"]) for name, f in cpp_fields.items()])}
  friend {to_json_decl};
  friend {from_json_decl};
}};
"""
        cpp_json_defs.append(f"inline {to_json_decl} {to_json_def}")
        cpp_json_defs.append(f"inline {from_json_decl} {from_json_def}")
        cpp_type_decls.append(f"class {name};")

        thrift_type_defs[name] = f"""
struct {name} {{
{chr(10).join(f"  {f['thrift_id']}: {f['thrift_type']} {n};" for n, f in thrift_fields.items())}
}}"""

    def _handle_union(name: str, ty: type[Any]) -> None:
        fields, cpp_fields, thrift_fields = _handle_aggregate(ty)
        yaml_ret[name] = {"kind": "union", "fields": fields}

        def accessor(name: str, ty: str, idx: int) -> str:
            return f"""
  const {ty}& get_{name}() const {{
    return std::get<{idx + 1}>(variant_);
  }}

  template <typename U>
  void set_{name}(U&& def) {{
    variant_.emplace<{idx + 1}>(std::forward<U>(def));
    tag_ = Tag::{name.upper()};
  }}
"""

        to_json_branches = "".join(
            [
                f"""
    if (nlohmann_json_t.tag_ == Tag::{name.upper()}) {{
      nlohmann_json_j["{name}"] = nlohmann_json_t.get_{name}();
      return;
    }}"""
                for idx, (name, f) in enumerate(cpp_fields.items())
            ]
        )
        from_json_branches = "".join(
            [
                f"""
    if (nlohmann_json_j.contains("{name}")) {{
      nlohmann_json_t.variant_.emplace<{idx + 1}>(nlohmann_json_j.at("{name}").template get<{f["cpp_type"]}>());
      nlohmann_json_t.tag_ = Tag::{name.upper()};
      return;
    }}"""
                for idx, (name, f) in enumerate(cpp_fields.items())
            ]
        )

        cpp_class_defs[name] = f"""
class {name} {{
  struct Void {{}};

 public:
  enum class Tag {{
    {", ".join([name.upper() for name in cpp_fields])}
  }};

 private:
  std::variant<Void, {", ".join(f["cpp_type"] for f in cpp_fields.values())}> variant_;
  Tag tag_{{}};

 public:
  Tag tag() const {{
    return tag_;
  }}
{"".join([accessor(name, f["cpp_type"], idx) for idx, (name, f) in enumerate(cpp_fields.items())])}
  friend void to_json(nlohmann::json& nlohmann_json_j, const {name}& nlohmann_json_t) {{
{to_json_branches}
  }}

  friend void from_json(const nlohmann::json& nlohmann_json_j, {name}& nlohmann_json_t) {{
{from_json_branches}
  }}
}};

inline std::string_view printEnum(const {name}::Tag& e) {{
  switch (e) {{
{chr(10).join([f"    case {name}::Tag::{x.upper()}: return {chr(34)}{x.upper()}{chr(34)};" for x in cpp_fields])}
    default:
      throw std::runtime_error("Unknown enum value");
  }}
}}

inline void parseEnum(std::string_view s, {name}::Tag& t) {{
{chr(10).join([f"  if (s == {chr(34)}{x.upper()}{chr(34)}) {{ t = {name}::Tag::{x.upper()}; return; }}" for x in cpp_fields])}
  throw std::runtime_error("Unknown enum value: " + std::string{{s}});
}}

"""
        cpp_type_decls.append(f"class {name};")

        thrift_type_defs[name] = f"""
union {name} {{
{chr(10).join(f"  {f['thrift_id']}: {f['thrift_type']} {n};" for n, f in thrift_fields.items())}
}}"""

    for name in dir(schema):
        if name.startswith("_"):
            continue

        value = getattr(schema, name)

        if hasattr(value, "__module__") and value.__module__ != schema.__name__:
            continue

        defs[name] = value

    class_ordering = {}
    for name, value in defs.items():
        if isinstance(value, type):
            if issubclass(value, IntEnum):
                _handle_int_enum(name, value)
            elif dataclasses.is_dataclass(value):
                class_ordering[name] = inspect.findsource(value)[1]
                if issubclass(value, _Union):
                    _handle_union(name, value)
                else:
                    _handle_struct(name, value)
            else:
                raise AssertionError(f"Unknown schema type {name}: {value}")
        elif isinstance(value, (int, tuple)):
            if name not in ("SCHEMA_VERSION", "TREESPEC_VERSION"):
                raise AssertionError(
                    f"expected SCHEMA_VERSION or TREESPEC_VERSION, got {name}"
                )
        elif isinstance(value, dict):
            # Skip mapping dictionaries used for codegen
            pass
        else:
            raise AssertionError(f"Unknown variable {name}: {value}")

    yaml_ret["SCHEMA_VERSION"] = list(defs["SCHEMA_VERSION"])
    if not all(x > 0 for x in yaml_ret["SCHEMA_VERSION"]):
        raise AssertionError(
            f"all SCHEMA_VERSION values must be > 0, got {yaml_ret['SCHEMA_VERSION']}"
        )
    yaml_ret["TREESPEC_VERSION"] = defs["TREESPEC_VERSION"]
    if yaml_ret["TREESPEC_VERSION"] <= 0:
        raise AssertionError(
            f"TREESPEC_VERSION must be > 0, got {yaml_ret['TREESPEC_VERSION']}"
        )

    cpp_header = f"""
#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef NLOHMANN_JSON_NAMESPACE_BEGIN
#define NLOHMANN_JSON_NAMESPACE_BEGIN namespace nlohmann {{
#endif

#ifndef NLOHMANN_JSON_NAMESPACE_END
#define NLOHMANN_JSON_NAMESPACE_END }}
#endif

// https://github.com/nlohmann/json/pull/2117
NLOHMANN_JSON_NAMESPACE_BEGIN
template <typename T>
struct adl_serializer<std::optional<T>> {{
  static void to_json(json& j, const std::optional<T>& opt) {{
    if (opt == std::nullopt) {{
      j = nullptr;
    }} else {{
      j = *opt; // this will call adl_serializer<T>::to_json which will
                // find the free function to_json in T's namespace!
    }}
  }}

  static void from_json(const json& j, std::optional<T>& opt) {{
    if (j.is_null()) {{
      opt = std::nullopt;
    }} else {{
      opt = j.template get<T>(); // same as above, but with
                                 // adl_serializer<T>::from_json
    }}
  }}
}};
NLOHMANN_JSON_NAMESPACE_END

namespace torch::_export {{

template <typename T>
class ForwardRef {{
  static_assert(!std::is_reference_v<T>, "ForwardRef cannot be a reference type");

 public:
  ForwardRef(): ptr_(std::make_unique<T>()) {{}}
  ForwardRef(ForwardRef<T>&&) noexcept;
  ForwardRef(const ForwardRef<T>& other): ptr_(std::make_unique<T>(*other.ptr_)) {{}}
  ForwardRef<T>& operator=(ForwardRef<T>&&) noexcept;
  ForwardRef<T>& operator=(const ForwardRef<T>& other) {{
    ptr_ = std::make_unique<T>(*other.ptr_);
    return *this;
  }}
  ~ForwardRef();
  const T& operator*() const {{
    return *ptr_;
  }}

  const T* operator->() const {{
    return ptr_.get();
  }}

  void emplace(T&& t) {{
    ptr_ = std::make_unique<T>(std::move(t));
  }}

 private:
  std::unique_ptr<T> ptr_;
}};

template <typename T>
void to_json(nlohmann::json& j, const ForwardRef<T>& p) {{
  j = *p;
}}

template <typename T>
void from_json(const nlohmann::json& j, ForwardRef<T>& p) {{
  p.emplace(j.template get<T>());
}}

class F64 {{
 public:
  double get() const {{
    return value_;
  }}

  void set(double value) {{
    value_ = value;
  }}

 private:
  double value_{{}};
}};

inline void to_json(nlohmann::json& j, const F64& f) {{
  if (std::isinf(f.get())) {{
    j = "Infinity";
  }} else if (std::isinf(-f.get())) {{
    j = "-Infinity";
  }} else if (std::isnan(f.get())) {{
    j = "NaN";
  }} else {{
    j = f.get();
  }}
}}

inline void from_json(const nlohmann::json& j, F64& f) {{
  if (j == "Infinity") {{
    f.set(std::numeric_limits<double>::infinity());
  }} else if (j == "-Infinity") {{
    f.set(-std::numeric_limits<double>::infinity());
  }} else if (j == "NaN") {{
    f.set(std::numeric_limits<double>::quiet_NaN());
  }} else {{
    f.set(j.get<double>());
  }}
}}

{chr(10).join(cpp_type_decls)}
{"".join(cpp_enum_defs.values())}
{"".join(dict(sorted(cpp_class_defs.items(), key=lambda x: class_ordering[x[0]])).values())}
{chr(10).join(cpp_json_defs)}

template <typename T> ForwardRef<T>::ForwardRef(ForwardRef<T>&&) noexcept = default;
template <typename T> ForwardRef<T>& ForwardRef<T>::operator=(ForwardRef<T>&&) noexcept = default;
template <typename T> ForwardRef<T>::~ForwardRef() = default;
}} // namespace torch::_export
"""
    thrift_schema = f"""
namespace py3 torch._export
namespace cpp2 torch._export.schema
{chr(10).join(thrift_enum_defs)}
{chr(10).join(dict(sorted(thrift_type_defs.items(), key=lambda x: class_ordering[x[0]])).values())}
"""
    return yaml_ret, cpp_header, thrift_schema


def _diff_schema(
    dst: dict[str, Any], src: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    additions = {key: src[key] for key in src.keys() - dst.keys()}
    subtractions = {key: dst[key] for key in dst.keys() - src.keys()}

    common_keys = src.keys() & dst.keys()

    versions = {"SCHEMA_VERSION", "TREESPEC_VERSION"}
    common_keys -= versions

    for key in common_keys:
        src_kind = src[key]["kind"]
        src_fields = src[key]["fields"]
        dst_kind = dst[key]["kind"]
        dst_fields = dst[key]["fields"]
        _check(
            src_kind == dst_kind,
            f"Type {key} changed kind from {dst_kind} to {src_kind}",
        )
        if not isinstance(src_fields, dict) or not isinstance(dst_fields, dict):
            raise AssertionError(
                f"expected dict fields, got src={type(src_fields)}, dst={type(dst_fields)}"
            )
        added_fields = {
            key: src_fields[key] for key in src_fields.keys() - dst_fields.keys()
        }
        subtracted_fields = {
            key: dst_fields[key] for key in dst_fields.keys() - src_fields.keys()
        }
        common_fields = src_fields.keys() & dst_fields.keys()

        for field in common_fields:
            src_field = src_fields[field]
            dst_field = dst_fields[field]
            if src_kind == "struct":
                _check(
                    src_field["type"] == dst_field["type"],
                    f"Type of the field {key}.{field} changed from {dst_field['type']} to {src_field['type']}",
                )
                if "default" in src_field and "default" not in dst_field:
                    added_fields[field] = {}
                    added_fields[field]["default"] = src_field["default"]
                if "default" not in src_field and "default" in dst_field:
                    subtracted_fields[field] = {}
                    subtracted_fields[field]["default"] = dst_field["default"]
            elif src_kind == "enum":
                _check(
                    src_field == dst_field,
                    f"Value of the enum field {key}.{field} changed from {dst_field} to {src_field}",
                )
            elif src_kind == "union":
                _check(
                    src_field["type"] == dst_field["type"],
                    f"Type of the field {key}.{field} changed from {dst_field['type']} to {src_field['type']}",
                )
            else:
                raise AssertionError(f"Unknown kind {src_kind}: {key}")
        if len(added_fields) > 0:
            if key in additions:
                raise AssertionError(f"key {key} already in additions")
            additions[key] = {}
            additions[key]["fields"] = added_fields
        if len(subtracted_fields) > 0:
            if key in subtractions:
                raise AssertionError(f"key {key} already in subtractions")
            subtractions[key] = {}
            subtractions[key]["fields"] = subtracted_fields

    return additions, subtractions


def _hash_content(s: str) -> str:
    return hashlib.sha256(s.strip().encode("utf-8")).hexdigest()


def _generate_enum_converters() -> str:
    """Generate C++ converter functions from serialized enum values to c10 enums."""

    def validate_mapping(
        enum_class: type[IntEnum],
        mapping: dict[int, str],
        enum_name: str,
        skip_values: set[int],
    ) -> None:
        """Validate that all enum values have corresponding c10 mappings."""
        for member in enum_class:
            if member.value in skip_values:
                continue
            if member.value not in mapping:
                raise SchemaUpdateError(
                    f"{enum_name}.{member.name} (value={member.value}) is missing "
                    f"from {enum_name.upper()}_TO_C10 mapping in schema.py. "
                    f"Please add the mapping to the c10 enum name."
                )

    # Validate that all enum values have mappings (except UNKNOWN values)
    validate_mapping(
        schema.ScalarType,
        schema.SCALAR_TYPE_TO_C10,
        "ScalarType",
        {schema.ScalarType.UNKNOWN},
    )
    validate_mapping(
        schema.Layout,
        schema.LAYOUT_TO_C10,
        "Layout",
        {schema.Layout.Unknown},
    )
    validate_mapping(
        schema.MemoryFormat,
        schema.MEMORY_FORMAT_TO_C10,
        "MemoryFormat",
        {schema.MemoryFormat.Unknown},
    )

    def generate_converter(
        name: str,
        c10_type: str,
        mapping: dict[int, str],
        max_value: int,
    ) -> str:
        lines: list[str] = []
        for i in range(max_value + 1):
            if i in mapping:
                lines.append(
                    f"      static_cast<int>(c10::{c10_type}::{mapping[i]}), // {i}"
                )
            else:
                lines.append(f"      kInvalid, // {i}")

        return f"""
inline c10::{c10_type} convertSerialized{name}(int serialized_value) {{
  constexpr int kInvalid = -1;
  constexpr int k{name}Map[] = {{
{chr(10).join(lines)}
  }};
  constexpr int kMapSize = sizeof(k{name}Map) / sizeof(k{name}Map[0]);

  TORCH_CHECK(
      serialized_value >= 0 && serialized_value < kMapSize,
      "Serialized {name} value out of range: ",
      serialized_value);
  int result = k{name}Map[serialized_value];
  TORCH_CHECK(
      result != kInvalid,
      "Invalid serialized {name} value: ",
      serialized_value);
  return static_cast<c10::{c10_type}>(result);
}}
"""

    scalar_type_converter = generate_converter(
        "ScalarType",
        "ScalarType",
        schema.SCALAR_TYPE_TO_C10,
        max(schema.SCALAR_TYPE_TO_C10.keys()),
    )
    layout_converter = generate_converter(
        "Layout",
        "Layout",
        schema.LAYOUT_TO_C10,
        max(schema.LAYOUT_TO_C10.keys()),
    )
    memory_format_converter = generate_converter(
        "MemoryFormat",
        "MemoryFormat",
        schema.MEMORY_FORMAT_TO_C10,
        max(schema.MEMORY_FORMAT_TO_C10.keys()),
    )

    return f"""
#pragma once

#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

// Converter functions from serialized enum values (torch._export.serde.schema)
// to c10 enums. The serialized format has different enum values than c10.

namespace torch::aot_inductor {{
{scalar_type_converter}
{layout_converter}
{memory_format_converter}
}} // namespace torch::aot_inductor
"""


@dataclasses.dataclass
class _Commit:
    result: dict[str, Any]
    checksum_next: str
    yaml_path: str
    additions: dict[str, Any]
    subtractions: dict[str, Any]
    base: dict[str, Any]
    checksum_head: str | None
    cpp_header: str
    cpp_header_path: str
    enum_converter_header: str
    enum_converter_header_path: str
    thrift_checksum_head: str | None
    thrift_checksum_real: str | None
    thrift_checksum_next: str
    thrift_schema: str
    thrift_schema_path: str


def update_schema() -> _Commit:
    import importlib.resources

    dst: dict[str, Any]
    # pyrefly: ignore [bad-argument-type]
    if importlib.resources.is_resource(__package__, "schema.yaml"):
        # pyrefly: ignore [bad-argument-type]
        content = importlib.resources.read_text(__package__, "schema.yaml")
        match = re.search("checksum<<([A-Fa-f0-9]{64})>>", content)
        _check(match is not None, "checksum not found in schema.yaml")
        if match is None:
            raise AssertionError("checksum not found in schema.yaml")
        checksum_head = match.group(1)

        thrift_content = importlib.resources.read_text(
            # pyrefly: ignore [bad-argument-type]
            __package__,
            "export_schema.thrift",
        )
        match = re.search("checksum<<([A-Fa-f0-9]{64})>>", thrift_content)
        _check(match is not None, "checksum not found in export_schema.thrift")
        if match is None:
            raise AssertionError("checksum not found in export_schema.thrift")
        thrift_checksum_head = match.group(1)
        thrift_content = thrift_content.splitlines()
        if not thrift_content[0].startswith("// @" + "generated"):
            raise AssertionError(
                f"expected first line to start with '// @generated', got {thrift_content[0]!r}"
            )
        if not thrift_content[1].startswith("// checksum<<"):
            raise AssertionError(
                f"expected second line to start with '// checksum<<', got {thrift_content[1]!r}"
            )
        thrift_checksum_real = _hash_content("\n".join(thrift_content[2:]))

        from yaml import load, Loader

        dst = load(content, Loader=Loader)
        if not isinstance(dst, dict):
            raise AssertionError(f"expected dict from yaml, got {type(dst)}")
    else:
        checksum_head = None
        thrift_checksum_head = None
        thrift_checksum_real = None
        dst = {"SCHEMA_VERSION": None, "TREESPEC_VERSION": None}

    src, cpp_header, thrift_schema = _staged_schema()
    enum_converter_header = _generate_enum_converters()
    additions, subtractions = _diff_schema(dst, src)
    # pyrefly: ignore [missing-attribute]
    yaml_path = __package__.replace(".", "/") + "/schema.yaml"
    # pyrefly: ignore [missing-attribute]
    thrift_schema_path = __package__.replace(".", "/") + "/export_schema.thrift"
    torch_prefix = "torch/"
    if not yaml_path.startswith(torch_prefix):
        raise AssertionError(
            f"yaml_path must start with {torch_prefix}, got {yaml_path}"
        )
    if not thrift_schema_path.startswith(torch_prefix):
        raise AssertionError(
            f"thrift_schema_path must start with {torch_prefix}, got {thrift_schema_path}"
        )

    return _Commit(
        result=src,
        checksum_next=_hash_content(repr(src)),
        yaml_path=yaml_path,
        additions=additions,
        subtractions=subtractions,
        base=dst,
        checksum_head=checksum_head,
        cpp_header=cpp_header,
        cpp_header_path=torch_prefix + "csrc/utils/generated_serialization_types.h",
        enum_converter_header=enum_converter_header,
        enum_converter_header_path=torch_prefix
        + "csrc/inductor/aoti_torch/generated_enum_converters.h",
        thrift_checksum_head=thrift_checksum_head,
        thrift_checksum_real=thrift_checksum_real,
        thrift_checksum_next=_hash_content(thrift_schema),
        thrift_schema=thrift_schema,
        thrift_schema_path=thrift_schema_path,
    )


def check(commit: _Commit, force_unsafe: bool = False) -> tuple[list[int] | None, str]:
    next_version: list[int] | None = None
    reason = ""
    # Step 1: Detect major schema updates.
    if len(commit.additions) > 0:
        for k, v in commit.additions.items():
            if k not in commit.base:
                continue
            kind = commit.result[k]["kind"]
            fields = v["fields"]
            for f, d in fields.items():
                if kind == "struct" and "default" not in d:
                    reason += (
                        f"Field {k}.{f} is added to schema.py without a default value as an incompatible change "
                        + "which requires major version bump.\n"
                    )
                    next_version = [commit.base["SCHEMA_VERSION"][0] + 1, 1]

    if len(commit.subtractions) > 0:
        for k, v in commit.subtractions.items():
            if k not in commit.result:
                continue
            for f in v["fields"]:
                reason = f"Field {k}.{f} is removed from schema.py as an incompatible change which requires major version bump.\n"
            next_version = [commit.base["SCHEMA_VERSION"][0] + 1, 1]

    if force_unsafe:
        reason += "--force-unsafe is used."
        next_version = commit.result["SCHEMA_VERSION"]
    else:
        # Step 2: Detect minor schema updates.
        if next_version is None and len(commit.additions) > 0:
            for k, v in commit.additions.items():
                for f in v["fields"]:
                    reason += (
                        f"Field {k}.{f} is added to schema.py as a compatible change "
                        + "which still requires minor version bump.\n"
                    )
            next_version = [
                commit.base["SCHEMA_VERSION"][0],
                commit.base["SCHEMA_VERSION"][1] + 1,
            ]
        if next_version is None and len(commit.subtractions) > 0:
            for k, v in commit.subtractions.items():
                for f in v["fields"]:
                    reason += (
                        f"Field {k}.{f} is removed from schema.py as a compatible change "
                        + "which still requires minor version bump.\n"
                    )
            next_version = [
                commit.base["SCHEMA_VERSION"][0],
                commit.base["SCHEMA_VERSION"][1] + 1,
            ]

    return next_version, reason
<|endoftext|>
def exportdb_error_message(case_name: str) -> str:
    from .examples import all_examples
    from torch._utils_internal import log_export_usage

    ALL_EXAMPLES = all_examples()
    # Detect whether case_name is really registered in exportdb.
    if case_name in ALL_EXAMPLES:
        url_case_name = case_name.replace("_", "-")
        return f"See {case_name} in exportdb for unsupported case. \
                https://pytorch.org/docs/main/generated/exportdb/index.html#{url_case_name}"
    else:
        log_export_usage(
            event="export.error.casenotregistered",
            message=case_name,
        )
        return f"{case_name} is unsupported."


def get_class_if_classified_error(e: Exception) -> str | None:
    """
    Returns a string case name if the export error e is classified.
    Returns None otherwise.
    """

    from torch._dynamo.exc import TorchRuntimeError, Unsupported, UserError

    ALWAYS_CLASSIFIED = "always_classified"
    DEFAULT_CLASS_SIGIL = "case_name"

    # add error types that should be classified, along with any attribute name
    # whose presence acts like a sigil to further distinguish which errors of
    # that type should be classified. If the attribute name is None, then the
    # error type is always classified.
    _ALLOW_LIST = {
        Unsupported: DEFAULT_CLASS_SIGIL,
        UserError: DEFAULT_CLASS_SIGIL,
        TorchRuntimeError: None,
    }
    if type(e) in _ALLOW_LIST:
        # pyrefly: ignore [bad-index, index-error]
        attr_name = _ALLOW_LIST[type(e)]
        if attr_name is None:
            return ALWAYS_CLASSIFIED
        return getattr(e, attr_name, None)
    return None
<|endoftext|># mypy: allow-untyped-defs
import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from types import ModuleType

import torch

_TAGS: dict[str, dict[str, Any]] = {
    "torch": {
        "cond": {},
        "dynamic-shape": {},
        "escape-hatch": {},
        "map": {},
        "dynamic-value": {},
        "operator": {},
        "mutation": {},
    },
    "python": {
        "assert": {},
        "builtin": {},
        "closure": {},
        "context-manager": {},
        "control-flow": {},
        "data-structure": {},
        "standard-library": {},
        "object-model": {},
    },
}


class SupportLevel(Enum):
    """
    Indicates at what stage the feature
    used in the example is handled in export.
    """

    SUPPORTED = 1
    NOT_SUPPORTED_YET = 0


ArgsType = tuple[Any, ...]


def check_inputs_type(args, kwargs):
    if not isinstance(args, tuple):
        raise ValueError(
            f"Expecting args type to be a tuple, got: {type(args)}"
        )
    if not isinstance(kwargs, dict):
        raise ValueError(
            f"Expecting kwargs type to be a dict, got: {type(kwargs)}"
        )
    for key in kwargs:
        if not isinstance(key, str):
            raise ValueError(
                f"Expecting kwargs keys to be a string, got: {type(key)}"
            )

def _validate_tag(tag: str):
    parts = tag.split(".")
    t = _TAGS
    for part in parts:
        if not set(part) <= set(string.ascii_lowercase + "-"):
            raise AssertionError(f"Tag contains invalid characters: {part}")
        if part in t:
            t = t[part]
        else:
            raise ValueError(f"Tag {tag} is not found in registered tags.")


@dataclass(frozen=True)
class ExportCase:
    example_args: ArgsType
    description: str  # A description of the use case.
    model: torch.nn.Module
    name: str
    example_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_args: ArgsType | None = None  # For testing graph generalization.
    # Tags associated with the use case. (e.g dynamic-shape, escape-hatch)
    tags: set[str] = field(default_factory=set)
    support_level: SupportLevel = SupportLevel.SUPPORTED
    dynamic_shapes: dict[str, Any] | None = None

    def __post_init__(self):
        check_inputs_type(self.example_args, self.example_kwargs)
        if self.extra_args is not None:
            check_inputs_type(self.extra_args, {})

        for tag in self.tags:
            _validate_tag(tag)

        if not isinstance(self.description, str) or len(self.description) == 0:
            raise ValueError(f'Invalid description: "{self.description}"')


_EXAMPLE_CASES: dict[str, ExportCase] = {}
_MODULES: set[ModuleType] = set()
_EXAMPLE_CONFLICT_CASES: dict[str, list[ExportCase]] = {}
_EXAMPLE_REWRITE_CASES: dict[str, list[ExportCase]] = {}


def register_db_case(case: ExportCase) -> None:
    """
    Registers a user provided ExportCase into example bank.
    """
    if case.name in _EXAMPLE_CASES:
        if case.name not in _EXAMPLE_CONFLICT_CASES:
            _EXAMPLE_CONFLICT_CASES[case.name] = [_EXAMPLE_CASES[case.name]]
        _EXAMPLE_CONFLICT_CASES[case.name].append(case)
        return

    _EXAMPLE_CASES[case.name] = case


def to_snake_case(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _make_export_case(m, name, configs):
    if not isinstance(m, torch.nn.Module):
        raise TypeError("Export case class should be a torch.nn.Module.")

    if "description" not in configs:
        # Fallback to docstring if description is missing.
        if m.__doc__ is None:
            raise AssertionError(
                f"Could not find description or docstring for export case: {m}"
            )
        configs = {**configs, "description": m.__doc__}
    # pyrefly: ignore [bad-argument-type]
    return ExportCase(**{**configs, "model": m, "name": name})


def export_case(**kwargs):
    """
    Decorator for registering a user provided case into example bank.
    """

    def wrapper(m):
        configs = kwargs
        module = inspect.getmodule(m)
        if module in _MODULES:
            raise RuntimeError("export_case should only be used once per example file.")

        if module is None:
            raise AssertionError("module must not be None")
        _MODULES.add(module)
        module_name = module.__name__.split(".")[-1]
        case = _make_export_case(m, module_name, configs)
        register_db_case(case)
        return case

    return wrapper


def export_rewrite_case(**kwargs):
    def wrapper(m):
        configs = kwargs

        parent = configs.pop("parent")
        if not isinstance(parent, ExportCase):
            raise AssertionError(f"expected ExportCase, got {type(parent)}")
        key = parent.name
        if key not in _EXAMPLE_REWRITE_CASES:
            _EXAMPLE_REWRITE_CASES[key] = []

        configs["example_args"] = parent.example_args
        case = _make_export_case(m, to_snake_case(m.__name__), configs)
        _EXAMPLE_REWRITE_CASES[key].append(case)
        return case

    return wrapper
<|endoftext|>import os
import sys

import torch._export.db.examples as examples

TEMPLATE = '''import torch

def {case_name}(x):
    """
    """

    return
'''

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise AssertionError(f"expected 2 arguments, got {len(sys.argv)}")
    root_dir = examples.__name__.replace(".", "/")
    if not os.path.exists(root_dir):
        raise AssertionError(f"root_dir does not exist: {root_dir}")
    with open(os.path.join(root_dir, sys.argv[1] + ".py"), "w") as f:
        print("Writing to", f.name, "...")
        f.write(TEMPLATE.format(case_name=sys.argv[1]))
<|endoftext|># mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class MySubModule(torch.nn.Module):
    def foo(self, x):
        return x.cos()

    def forward(self, x):
        return self.foo(x)

class CondBranchClassMethod(torch.nn.Module):
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
      - both branches must take the same args, which must also match the branch args passed to cond.
      - both branches must return a single tensor
      - returned tensor must have the same tensor metadata, e.g. shape and dtype
      - branch function can be free function, nested function, lambda, class methods
      - branch function can not have closure variables
      - no inplace mutations on inputs or global variables


    This example demonstrates using class method in cond().

    NOTE: If the `pred` is tested on a dim with batch size < 2, it will be specialized.
    """

    def __init__(self) -> None:
        super().__init__()
        self.subm = MySubModule()

    def bar(self, x):
        return x.sin()

    def forward(self, x):
        return cond(x.shape[0] <= 2, self.subm.forward, self.bar, [x])

example_args = (torch.randn(3),)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
model = CondBranchClassMethod()
<|endoftext|># mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class CondClosedOverVariable(torch.nn.Module):
    """
    torch.cond() supports branches closed over arbitrary variables.
    """

    def forward(self, pred, x):
        def true_fn(val):
            return x * 2

        def false_fn(val):
            return x - 2

        return cond(pred, true_fn, false_fn, [x + 1])

example_args = (torch.tensor(True), torch.randn(3, 2))
tags = {"torch.cond", "python.closure"}
model = CondClosedOverVariable()
<|endoftext|># mypy: allow-untyped-defs
import torch


class ConstrainAsSizeExample(torch.nn.Module):
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at torch._check APIs.
    """

    def forward(self, x):
        a = x.item()
        torch._check(a >= 0)
        torch._check(a <= 5)
        return torch.zeros((a, 5))


example_args = (torch.tensor(4),)
tags = {
    "torch.dynamic-value",
    "torch.escape-hatch",
}
model = ConstrainAsSizeExample()
<|endoftext|># mypy: allow-untyped-defs
import torch

from torch._export.db.case import SupportLevel
from torch.export import Dim

class DynamicShapeRound(torch.nn.Module):
    """
    Calling round on dynamic shapes is not supported.
    """

    def forward(self, x):
        return x[: round(x.shape[0] / 2)]

x = torch.randn(3, 2)
dim0_x = Dim("dim0_x")
example_args = (x,)
tags = {"torch.dynamic-shape", "python.builtin"}
support_level = SupportLevel.NOT_SUPPORTED_YET
dynamic_shapes = {"x": {0: dim0_x}}
model = DynamicShapeRound()
<|endoftext|># mypy: allow-untyped-defs
import torch


class ModelAttrMutation(torch.nn.Module):
    """
    Attribute mutation raises a warning. Covered in the test_export.py test_detect_leak_strict test.
    """

    def __init__(self) -> None:
        super().__init__()
        self.attr_list = [torch.randn(3, 2), torch.randn(3, 2)]

    def recreate_list(self):
        return [torch.zeros(3, 2), torch.zeros(3, 2)]

    def forward(self, x):
        self.attr_list = self.recreate_list()
        return x.sum() + self.attr_list[0].sum()


example_args = (torch.randn(3, 2),)
tags = {"python.object-model"}
model = ModelAttrMutation()
<|endoftext|># mypy: allow-untyped-defs
import torch

from torch.utils import _pytree as pytree

class PytreeFlatten(torch.nn.Module):
    """
    Pytree from PyTorch can be captured by TorchDynamo.
    """

    def forward(self, x):
        y, _spec = pytree.tree_flatten(x)
        return y[0] + 1

example_args = ({1: torch.randn(3, 2), 2: torch.randn(3, 2)},),
model = PytreeFlatten()
<|endoftext|># mypy: allow-untyped-defs
import functools

import torch

def test_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs) + 1

    return wrapper

class Decorator(torch.nn.Module):
    """
    Decorators calls are inlined into the exported function during tracing.
    """

    @test_decorator
    def forward(self, x, y):
        return x + y

example_args = (torch.randn(3, 2), torch.randn(3, 2))
model = Decorator()
<|endoftext|># mypy: allow-untyped-defs
import torch

class StaticIf(torch.nn.Module):
    """
    `if` statement with static predicate value should be traced through with the
    taken branch.
    """

    def forward(self, x):
        if len(x.shape) == 3:
            return x + torch.ones(1, 1, 1)

        return x

example_args = (torch.randn(3, 2, 2),)
tags = {"python.control-flow"}
model = StaticIf()
<|endoftext|># mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import map

class DynamicShapeMap(torch.nn.Module):
    """
    functorch map() maps a function over the first tensor dimension.
    """

    def forward(self, xs, y):
        def body(x, y):
            return x + y

        return map(body, xs, y)

example_args = (torch.randn(3, 2), torch.randn(2))
tags = {"torch.dynamic-shape", "torch.map"}
model = DynamicShapeMap()
<|endoftext|># mypy: allow-untyped-defs
import contextlib

import torch

class NullContextManager(torch.nn.Module):
    """
    Null context manager in Python will be traced out.
    """

    def forward(self, x):
        """
        Null context manager in Python will be traced out.
        """
        ctx = contextlib.nullcontext()
        with ctx:
            return x.sin() + x.cos()

example_args = (torch.randn(3, 2),)
tags = {"python.context-manager"}
model = NullContextManager()
<|endoftext|># mypy: allow-untyped-defs
import torch

class StaticForLoop(torch.nn.Module):
    """
    A for loop with constant number of iterations should be unrolled in the exported graph.
    """

    def forward(self, x):
        # constant
        ret = [i + x for i in range(10)]
        return ret

example_args = (torch.randn(3, 2),)
tags = {"python.control-flow"}
model = StaticForLoop()
<|endoftext|># mypy: allow-untyped-defs
import torch
import torch._dynamo as torchdynamo


class AssumeConstantResult(torch.nn.Module):
    """
    Applying `assume_constant_result` decorator to burn make non-traceable code as constant.
    """

    @torchdynamo.assume_constant_result
    def get_item(self, y):
        return y.int().item()

    def forward(self, x, y):
        return x[: self.get_item(y)]

example_args = (torch.randn(3, 2), torch.tensor(4))
tags = {"torch.escape-hatch"}
model = AssumeConstantResult()
<|endoftext|># mypy: allow-untyped-defs
import torch

class FnWithKwargs(torch.nn.Module):
    """
    Keyword arguments are not supported at the moment.
    """

    def forward(self, pos0, tuple0, *myargs, mykw0, **mykwargs):
        out = pos0
        for arg in tuple0:
            out = out * arg
        for arg in myargs:
            out = out * arg
        out = out * mykw0
        out = out * mykwargs["input0"] * mykwargs["input1"]
        return out

example_args = (
    torch.randn(4),
    (torch.randn(4), torch.randn(4)),
    *[torch.randn(4), torch.randn(4)]
)
example_kwargs = {
    "mykw0": torch.randn(4),
    "input0": torch.randn(4),
    "input1": torch.randn(4),
}
tags = {"python.data-structure"}
model = FnWithKwargs()
<|endoftext|># mypy: allow-untyped-defs
import torch

class DynamicShapeConstructor(torch.nn.Module):
    """
    Tensor constructors should be captured with dynamic shape inputs rather
    than being baked in with static shape.
    """

    def forward(self, x):
        return torch.zeros(x.shape[0] * 2)

example_args = (torch.randn(3, 2),)
tags = {"torch.dynamic-shape"}
model = DynamicShapeConstructor()
<|endoftext|># mypy: allow-untyped-defs
import torch

from torch.export import Dim

x = torch.randn(3, 2)
dim1_x = Dim("dim1_x")

class ScalarOutput(torch.nn.Module):
    """
    Returning scalar values from the graph is supported, in addition to Tensor
    outputs. Symbolic shapes are captured and rank is specialized.
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x.shape[1] + 1

example_args = (x,)
tags = {"torch.dynamic-shape"}
dynamic_shapes = {"x": {1: dim1_x}}
model = ScalarOutput()
<|endoftext|># mypy: allow-untyped-defs
import torch
from torch._export.db.case import SupportLevel


class TorchSymMin(torch.nn.Module):
    """
    torch.sym_min operator is supported in export.
    """

    def forward(self, x):
        return x.sum() + torch.sym_min(x.size(0), 100)


example_args = (torch.randn(3, 2),)
tags = {"torch.operator"}
support_level = SupportLevel.SUPPORTED
model = TorchSymMin()
<|endoftext|># mypy: allow-untyped-defs
import torch

class NestedFunction(torch.nn.Module):
    """
    Nested functions are traced through. Side effects on global captures
    are not supported though.
    """

    def forward(self, a, b):
        x = a + b
        z = a - b

        def closure(y):
            nonlocal x
            x += 1
            return x * y + z

        return closure(x)

example_args = (torch.randn(3, 2), torch.randn(2))
tags = {"python.closure"}
model = NestedFunction()
<|endoftext|># mypy: allow-untyped-defs
import torch

class MyAutogradFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        return grad_output + 1

class AutogradFunction(torch.nn.Module):
    """
    TorchDynamo does not keep track of backward() on autograd functions. We recommend to
    use `allow_in_graph` to mitigate this problem.
    """

    def forward(self, x):
        return MyAutogradFunction.apply(x)

example_args = (torch.randn(3, 2),)
model = AutogradFunction()
<|endoftext|># mypy: allow-untyped-defs
import torch
from torch._export.db.case import SupportLevel


class OptionalInput(torch.nn.Module):
    """
    Tracing through optional input is not supported yet
    """

    def forward(self, x, y=torch.randn(2, 3)):
        if y is not None:
            return x + y
        return x


example_args = (torch.randn(2, 3),)
tags = {"python.object-model"}
support_level = SupportLevel.SUPPORTED
model = OptionalInput()
<|endoftext|># mypy: allow-untyped-defs
from enum import Enum

import torch

class Animal(Enum):
    COW = "moo"

class SpecializedAttribute(torch.nn.Module):
    """
    Model attributes are specialized.
    """

    def __init__(self) -> None:
        super().__init__()
        self.a = "moo"
        self.b = 4

    def forward(self, x):
        if self.a == Animal.COW.value:
            return x * x + self.b
        else:
            raise ValueError("bad")

example_args = (torch.randn(3, 2),)
model = SpecializedAttribute()
<|endoftext|># mypy: allow-untyped-defs
import torch

class DynamicShapeSlicing(torch.nn.Module):
    """
    Slices with dynamic shape arguments should be captured into the graph
    rather than being baked in.
    """

    def forward(self, x):
        return x[: x.shape[0] - 2, x.shape[1] - 1 :: 2]

example_args = (torch.randn(3, 2),)
tags = {"torch.dynamic-shape"}
model = DynamicShapeSlicing()
<|endoftext|># mypy: allow-untyped-defs
import torch


class TensorSetattr(torch.nn.Module):
    """
    setattr() call onto tensors is not supported.
    """
    def forward(self, x, attr):
        setattr(x, attr, torch.randn(3, 2))
        return x + 4

example_args = (torch.randn(3, 2), "attr")
tags = {"python.builtin"}
model = TensorSetattr()
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
import glob
import inspect
from os.path import basename, dirname, isfile, join

import torch
from torch._export.db.case import (
    _EXAMPLE_CASES,
    _EXAMPLE_CONFLICT_CASES,
    _EXAMPLE_REWRITE_CASES,
    SupportLevel,
    export_case,
    ExportCase,
)


def _collect_examples():
    case_names = glob.glob(join(dirname(__file__), "*.py"))
    case_names = [
        basename(f)[:-3] for f in case_names if isfile(f) and not f.endswith("__init__.py")
    ]

    case_fields = {f.name for f in dataclasses.fields(ExportCase)}
    for case_name in case_names:
        case = __import__(case_name, globals(), locals(), [], 1)
        variables = [name for name in dir(case) if name in case_fields]
        export_case(**{v: getattr(case, v) for v in variables})(case.model)

_collect_examples()

def all_examples():
    return _EXAMPLE_CASES


if len(_EXAMPLE_CONFLICT_CASES) > 0:

    def get_name(case):
        model = case.model
        if isinstance(model, torch.nn.Module):
            model = type(model)
        return model.__name__

    msg = "Error on conflict export case name.\n"
    for case_name, cases in _EXAMPLE_CONFLICT_CASES.items():
        msg += f"Case name {case_name} is associated with multiple cases:\n  "
        msg += f"[{','.join(map(get_name, cases))}]\n"

    raise RuntimeError(msg)


def filter_examples_by_support_level(support_level: SupportLevel):
    return {
        key: val
        for key, val in all_examples().items()
        if val.support_level == support_level
    }


def get_rewrite_cases(case):
    return _EXAMPLE_REWRITE_CASES.get(case.name, [])
<|endoftext|># mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class CondPredicate(torch.nn.Module):
    """
    The conditional statement (aka predicate) passed to cond() must be one of the following:
      - torch.Tensor with a single element
      - boolean expression

    NOTE: If the `pred` is tested on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x):
        pred = x.dim() > 2 and x.shape[2] > 10

        return cond(pred, lambda x: x.cos(), lambda y: y.sin(), [x])

example_args = (torch.randn(6, 4, 3),)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
model = CondPredicate()
<|endoftext|># mypy: allow-untyped-defs
import torch


class UserInputMutation(torch.nn.Module):
    """
    Directly mutate user input in forward
    """

    def forward(self, x):
        x.mul_(2)
        return x.cos()


example_args = (torch.randn(3, 2),)
tags = {"torch.mutation"}
model = UserInputMutation()
<|endoftext|># mypy: allow-untyped-defs
import torch

class A:
    @classmethod
    def func(cls, x):
        return 1 + x

class TypeReflectionMethod(torch.nn.Module):
    """
    type() calls on custom objects followed by attribute accesses are not allowed
    due to its overly dynamic nature.
    """

    def forward(self, x):
        a = A()
        return type(a).func(x)


example_args = (torch.randn(3, 4),)
tags = {"python.builtin"}
model = TypeReflectionMethod()
<|endoftext|># mypy: allow-untyped-defs
import torch

class ClassMethod(torch.nn.Module):
    """
    Class methods are inlined during tracing.
    """

    @classmethod
    def method(cls, x):
        return x + 1

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear(x)
        return self.method(x) * self.__class__.method(x) * type(self).method(x)

example_args = (torch.randn(3, 4),)
model = ClassMethod()
<|endoftext|># mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class CondBranchNonlocalVariables(torch.nn.Module):
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
    - both branches must take the same args, which must also match the branch args passed to cond.
    - both branches must return a single tensor
    - returned tensor must have the same tensor metadata, e.g. shape and dtype
    - branch function can be free function, nested function, lambda, class methods
    - branch function can not have closure variables
    - no inplace mutations on inputs or global variables

    This example demonstrates how to rewrite code to avoid capturing closure variables in branch functions.

    The code below will not work because capturing closure variables is not supported.
    ```
    my_tensor_var = x + 100
    my_primitive_var = 3.14

    def true_fn(y):
        nonlocal my_tensor_var, my_primitive_var
        return y + my_tensor_var + my_primitive_var

    def false_fn(y):
        nonlocal my_tensor_var, my_primitive_var
        return y - my_tensor_var - my_primitive_var

    return cond(x.shape[0] > 5, true_fn, false_fn, [x])
    ```

    NOTE: If the `pred` is tested on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x):
        my_tensor_var = x + 100
        my_primitive_var = 3.14

        def true_fn(x, y, z):
            return x + y + z

        def false_fn(x, y, z):
            return x - y - z

        return cond(
            x.shape[0] > 5,
            true_fn,
            false_fn,
            [x, my_tensor_var, torch.tensor(my_primitive_var)],
        )

example_args = (torch.randn(6),)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
model = CondBranchNonlocalVariables()
<|endoftext|># mypy: allow-untyped-defs
import torch

class Dictionary(torch.nn.Module):
    """
    Dictionary structures are inlined and flattened along tracing.
    """

    def forward(self, x, y):
        elements = {}
        elements["x2"] = x * x
        y = y * elements["x2"]
        return {"y": y}

example_args = (torch.randn(3, 2), torch.tensor(4))
tags = {"python.data-structure"}
model = Dictionary()
<|endoftext|># mypy: allow-untyped-defs

import torch

class ListUnpack(torch.nn.Module):
    """
    Lists are treated as static construct, therefore unpacking should be
    erased after tracing.
    """

    def forward(self, args: list[torch.Tensor]):
        """
        Lists are treated as static construct, therefore unpacking should be
        erased after tracing.
        """
        x, *y = args
        return x + y[0]

example_args = ([torch.randn(3, 2), torch.tensor(4), torch.tensor(5)],)
tags = {"python.control-flow", "python.data-structure"}
model = ListUnpack()
<|endoftext|># mypy: allow-untyped-defs
import torch

class ListContains(torch.nn.Module):
    """
    List containment relation can be checked on a dynamic shape or constants.
    """

    def forward(self, x):
        assert x.size(-1) in [6, 2]  # noqa: S101
        assert x.size(0) not in [4, 5, 6]  # noqa: S101
        assert "monkey" not in ["cow", "pig"]  # noqa: S101
        return x + x

example_args = (torch.randn(3, 2),)
tags = {"torch.dynamic-shape", "python.data-structure", "python.assert"}
model = ListContains()
<|endoftext|># mypy: allow-untyped-defs
import torch

class DynamicShapeAssert(torch.nn.Module):
    """
    A basic usage of python assertion.
    """

    def forward(self, x):
        # assertion with error message
        assert x.shape[0] > 2, f"{x.shape[0]} is greater than 2"  # noqa: S101
        # assertion without error message
        assert x.shape[0] > 1  # noqa: S101
        return x

example_args = (torch.randn(3, 2),)
tags = {"python.assert"}
model = DynamicShapeAssert()
<|endoftext|># mypy: allow-untyped-defs
import torch


class ConstrainAsValueExample(torch.nn.Module):
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at torch._check API.
    """

    def forward(self, x, y):
        a = x.item()
        torch._check(a >= 0)
        torch._check(a <= 5)

        if a < 6:
            return y.sin()
        return y.cos()


example_args = (torch.tensor(4), torch.randn(5, 5))
tags = {
    "torch.dynamic-value",
    "torch.escape-hatch",
}
model = ConstrainAsValueExample()
<|endoftext|># mypy: allow-untyped-defs
import torch

class DynamicShapeIfGuard(torch.nn.Module):
    """
    `if` statement with backed dynamic shape predicate will be specialized into
    one particular branch and generate a guard. However, export will fail if the
    dimension is marked as dynamic shape from higher level API.
    """

    def forward(self, x):
        if x.shape[0] == 3:
            return x.cos()

        return x.sin()

example_args = (torch.randn(3, 2, 2),)
tags = {"torch.dynamic-shape", "python.control-flow"}
model = DynamicShapeIfGuard()
<|endoftext|># mypy: allow-untyped-defs
import torch

class DynamicShapeView(torch.nn.Module):
    """
    Dynamic shapes should be propagated to view arguments instead of being
    baked into the exported graph.
    """

    def forward(self, x):
        new_x_shape = x.size()[:-1] + (2, 5)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

example_args = (torch.randn(10, 10),)
tags = {"torch.dynamic-shape"}
model = DynamicShapeView()
<|endoftext|># mypy: allow-untyped-defs
import torch

from functorch.experimental.control_flow import cond

class CondBranchNestedFunction(torch.nn.Module):
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
      - both branches must take the same args, which must also match the branch args passed to cond.
      - both branches must return a single tensor
      - returned tensor must have the same tensor metadata, e.g. shape and dtype
      - branch function can be free function, nested function, lambda, class methods
      - branch function can not have closure variables
      - no inplace mutations on inputs or global variables

    This example demonstrates using nested function in cond().

    NOTE: If the `pred` is tested on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x):
        def true_fn(x):
            def inner_true_fn(y):
                return x + y

            return inner_true_fn(x)

        def false_fn(x):
            def inner_false_fn(y):
                return x - y

            return inner_false_fn(x)

        return cond(x.shape[0] < 10, true_fn, false_fn, [x])

example_args = (torch.randn(3),)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
model = CondBranchNestedFunction()
<|endoftext|># mypy: allow-untyped-defs
import torch

from torch.export import Dim

x = torch.randn(3, 2)
y = torch.randn(2)
dim0_x = Dim("dim0_x")

class CondOperands(torch.nn.Module):
    """
    The operands passed to cond() must be:
    - a list of tensors
    - match arguments of `true_fn` and `false_fn`

    NOTE: If the `pred` is tested on a dim with batch size < 2, it will be specialized.
    """

    def forward(self, x, y):
        def true_fn(x, y):
            return x + y

        def false_fn(x, y):
            return x - y

        return torch.cond(x.shape[0] > 2, true_fn, false_fn, [x, y])

example_args = (x, y)
tags = {
    "torch.cond",
    "torch.dynamic-shape",
}
extra_inputs = (torch.randn(2, 2), torch.randn(2))
dynamic_shapes = {"x": {0: dim0_x}, "y": None}
model = CondOperands()
<|endoftext|># mypy: allow-untyped-defs
from collections.abc import Callable, Iterable
from math import sqrt
from typing import TypeVar

import torch
from torch import Tensor
from torch._torch_docs import factory_common_args, merge_dicts, parse_kwargs


__all__ = [
    "bartlett",
    "blackman",
    "cosine",
    "exponential",
    "gaussian",
    "general_cosine",
    "general_hamming",
    "hamming",
    "hann",
    "kaiser",
    "nuttall",
]

_T = TypeVar("_T")

window_common_args = merge_dicts(
    parse_kwargs(
        """
    M (int): the length of the window.
        In other words, the number of points of the returned window.
    sym (bool, optional): If `False`, returns a periodic window suitable for use in spectral analysis.
        If `True`, returns a symmetric window suitable for use in filter design. Default: `True`.
"""
    ),
    factory_common_args,
    {
        "normalization": "The window is normalized to 1 (maximum value is 1). However, the 1 doesn't appear if "
        ":attr:`M` is even and :attr:`sym` is `True`.",
    },
)


def _add_docstr(*args: str) -> Callable[[_T], _T]:
    r"""Adds docstrings to a given decorated function.

    Specially useful when the docstrings need string interpolation, e.g., with
    str.format().
    REMARK: Do not use this function if the docstring doesn't need string
    interpolation, just write a conventional docstring.

    Args:
        args (str):
    """

    def decorator(o: _T) -> _T:
        o.__doc__ = "".join(args)
        return o

    return decorator


def _window_function_checks(
    function_name: str, M: int, dtype: torch.dtype, layout: torch.layout
) -> None:
    r"""Performs common checks for all the defined windows.
    This function should be called before computing any window.

    Args:
        function_name (str): name of the window function.
        M (int): length of the window.
        dtype (:class:`torch.dtype`): the desired data type of returned tensor.
        layout (:class:`torch.layout`): the desired layout of returned tensor.
    """
    if M < 0:
        raise ValueError(
            f"{function_name} requires non-negative window length, got M={M}"
        )
    if layout is not torch.strided:
        raise ValueError(
            f"{function_name} is implemented for strided tensors only, got: {layout}"
        )
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(
            f"{function_name} expects float32 or float64 dtypes, got: {dtype}"
        )


@_add_docstr(
    r"""
Computes a window with an exponential waveform.
Also known as Poisson window.

The exponential window is defined as follows:

.. math::
    w_n = \exp{\left(-\frac{|n - c|}{\tau}\right)}

where `c` is the ``center`` of the window.
    """,
    r"""

{normalization}

Args:
    {M}

Keyword args:
    center (float, optional): where the center of the window will be located.
        Default: `M / 2` if `sym` is `False`, else `(M - 1) / 2`.
    tau (float, optional): the decay value.
        Tau is generally associated with a percentage, that means, that the value should
        vary within the interval (0, 100]. If tau is 100, it is considered the uniform window.
        Default: 1.0.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric exponential window of size 10 and with a decay value of 1.0.
    >>> # The center will be at (M - 1) / 2, where M is 10.
    >>> torch.signal.windows.exponential(10)
    tensor([0.0111, 0.0302, 0.0821, 0.2231, 0.6065, 0.6065, 0.2231, 0.0821, 0.0302, 0.0111])

    >>> # Generates a periodic exponential window and decay factor equal to .5
    >>> torch.signal.windows.exponential(10, sym=False,tau=.5)
    tensor([4.5400e-05, 3.3546e-04, 2.4788e-03, 1.8316e-02, 1.3534e-01, 1.0000e+00, 1.3534e-01, 1.8316e-02, 2.4788e-03, 3.3546e-04])
    """.format(**window_common_args),
)
def exponential(
    M: int,
    *,
    center: float | None = None,
    tau: float = 1.0,
    sym: bool = True,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks("exponential", M, dtype, layout)

    if tau <= 0:
        raise ValueError(f"Tau must be positive, got: {tau} instead.")

    if sym and center is not None:
        raise ValueError("Center must be None for symmetric windows")

    if M == 0:
        return torch.empty(
            (0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )

    if center is None:
        center = (M if not sym and M > 1 else M - 1) / 2.0

    constant = 1 / tau

    k = torch.linspace(
        start=-center * constant,
        end=(-center + (M - 1)) * constant,
        steps=M,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )

    return torch.exp(-torch.abs(k))


@_add_docstr(
    r"""
Computes a window with a simple cosine waveform, following the same implementation as SciPy.
This window is also known as the sine window.

The cosine window is defined as follows:

.. math::
    w_n = \sin\left(\frac{\pi (n + 0.5)}{M}\right)

This formula differs from the typical cosine window formula by incorporating a 0.5 term in the numerator,
which shifts the sample positions. This adjustment results in a window that starts and ends with non-zero values.

""",
    r"""

{normalization}

Args:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric cosine window.
    >>> torch.signal.windows.cosine(10)
    tensor([0.1564, 0.4540, 0.7071, 0.8910, 0.9877, 0.9877, 0.8910, 0.7071, 0.4540, 0.1564])

    >>> # Generates a periodic cosine window.
    >>> torch.signal.windows.cosine(10, sym=False)
    tensor([0.1423, 0.4154, 0.6549, 0.8413, 0.9595, 1.0000, 0.9595, 0.8413, 0.6549, 0.4154])
""".format(
        **window_common_args,
    ),
)
def cosine(
    M: int,
    *,
    sym: bool = True,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks("cosine", M, dtype, layout)

    if M == 0:
        return torch.empty(
            (0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )

    start = 0.5
    constant = torch.pi / (M + 1 if not sym and M > 1 else M)

    k = torch.linspace(
        start=start * constant,
        end=(start + (M - 1)) * constant,
        steps=M,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )

    return torch.sin(k)


@_add_docstr(
    r"""
Computes a window with a gaussian waveform.

The gaussian window is defined as follows:

.. math::
    w_n = \exp{\left(-\left(\frac{n}{2\sigma}\right)^2\right)}
    """,
    r"""

{normalization}

Args:
    {M}

Keyword args:
    std (float, optional): the standard deviation of the gaussian. It controls how narrow or wide the window is.
        Default: 1.0.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric gaussian window with a standard deviation of 1.0.
    >>> torch.signal.windows.gaussian(10)
    tensor([4.0065e-05, 2.1875e-03, 4.3937e-02, 3.2465e-01, 8.8250e-01, 8.8250e-01, 3.2465e-01, 4.3937e-02, 2.1875e-03, 4.0065e-05])

    >>> # Generates a periodic gaussian window and standard deviation equal to 0.9.
    >>> torch.signal.windows.gaussian(10, sym=False,std=0.9)
    tensor([1.9858e-07, 5.1365e-05, 3.8659e-03, 8.4658e-02, 5.3941e-01, 1.0000e+00, 5.3941e-01, 8.4658e-02, 3.8659e-03, 5.1365e-05])
""".format(
        **window_common_args,
    ),
)
def gaussian(
    M: int,
    *,
    std: float = 1.0,
    sym: bool = True,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks("gaussian", M, dtype, layout)

    if std <= 0:
        raise ValueError(f"Standard deviation must be positive, got: {std} instead.")

    if M == 0:
        return torch.empty(
            (0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )

    start = -(M if not sym and M > 1 else M - 1) / 2.0

    constant = 1 / (std * sqrt(2))

    k = torch.linspace(
        start=start * constant,
        end=(start + (M - 1)) * constant,
        steps=M,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )

    return torch.exp(-(k**2))  # pyrefly: ignore [unsupported-operation]


@_add_docstr(
    r"""
Computes the Kaiser window.

The Kaiser window is defined as follows:

.. math::
    w_n = I_0 \left( \beta \sqrt{1 - \left( {\frac{n - N/2}{N/2}} \right) ^2 } \right) / I_0( \beta )

where ``I_0`` is the zeroth order modified Bessel function of the first kind (see :func:`torch.special.i0`), and
``N = M - 1 if sym else M``.
    """,
    r"""

{normalization}

Args:
    {M}

Keyword args:
    beta (float, optional): shape parameter for the window. Must be non-negative. Default: 12.0
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Kaiser window with a shape parameter of 12.0.
    >>> torch.signal.windows.kaiser(5)
    tensor([4.0065e-05, 2.1875e-03, 4.3937e-02, 3.2465e-01, 8.8250e-01, 8.8250e-01, 3.2465e-01, 4.3937e-02, 2.1875e-03, 4.0065e-05])
    >>> # Generates a periodic Kaiser window and shape parameter equal to 0.9.
    >>> torch.signal.windows.kaiser(5, sym=False, beta=0.9)
    tensor([1.9858e-07, 5.1365e-05, 3.8659e-03, 8.4658e-02, 5.3941e-01, 1.0000e+00, 5.3941e-01, 8.4658e-02, 3.8659e-03, 5.1365e-05])
""".format(
        **window_common_args,
    ),
)
def kaiser(
    M: int,
    *,
    beta: float = 12.0,
    sym: bool = True,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks("kaiser", M, dtype, layout)

    if beta < 0:
        raise ValueError(f"beta must be non-negative, got: {beta} instead.")

    if M == 0:
        return torch.empty(
            (0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )

    if M == 1:
        return torch.ones(
            (1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )

    # Avoid NaNs by casting `beta` to the appropriate dtype.
    # pyrefly: ignore [bad-assignment]
    beta = torch.tensor(beta, dtype=dtype, device=device)

    start = -beta
    constant = 2.0 * beta / (M if not sym else M - 1)
    end = torch.minimum(
        # pyrefly: ignore [bad-argument-type]
        beta,
        # pyrefly: ignore [bad-argument-type]
        start + (M - 1) * constant,
    )

    k = torch.linspace(
        start=start,
        end=end,
        steps=M,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )

    return torch.i0(torch.sqrt(beta * beta - torch.pow(k, 2))) / torch.i0(
        # pyrefly: ignore [bad-argument-type]
        beta
    )


@_add_docstr(
    r"""
Computes the Hamming window.

The Hamming window is defined as follows:

.. math::
    w_n = \alpha - \beta\ \cos \left( \frac{2 \pi n}{M - 1} \right)
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    alpha (float, optional): The coefficient :math:`\alpha` in the equation above.
    beta (float, optional): The coefficient :math:`\beta` in the equation above.
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Hamming window.
    >>> torch.signal.windows.hamming(10)
    tensor([0.0800, 0.1876, 0.4601, 0.7700, 0.9723, 0.9723, 0.7700, 0.4601, 0.1876, 0.0800])

    >>> # Generates a periodic Hamming window.
    >>> torch.signal.windows.hamming(10, sym=False)
    tensor([0.0800, 0.1679, 0.3979, 0.6821, 0.9121, 1.0000, 0.9121, 0.6821, 0.3979, 0.1679])
""".format(**window_common_args),
)
def hamming(
    M: int,
    *,
    sym: bool = True,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return general_hamming(
        M,
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


@_add_docstr(
    r"""
Computes the Hann window.

The Hann window is defined as follows:

.. math::
    w_n = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{M - 1} \right)\right] =
    \sin^2 \left( \frac{\pi n}{M - 1} \right)
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Hann window.
    >>> torch.signal.windows.hann(10)
    tensor([0.0000, 0.1170, 0.4132, 0.7500, 0.9698, 0.9698, 0.7500, 0.4132, 0.1170, 0.0000])

    >>> # Generates a periodic Hann window.
    >>> torch.signal.windows.hann(10, sym=False)
    tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])
""".format(**window_common_args),
)
def hann(
    M: int,
    *,
    sym: bool = True,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return general_hamming(
        M,
        alpha=0.5,
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


@_add_docstr(
    r"""
Computes the Blackman window.

The Blackman window is defined as follows:

.. math::
    w_n = 0.42 - 0.5 \cos \left( \frac{2 \pi n}{M - 1} \right) + 0.08 \cos \left( \frac{4 \pi n}{M - 1} \right)
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Blackman window.
    >>> torch.signal.windows.blackman(5)
    tensor([-1.4901e-08,  3.4000e-01,  1.0000e+00,  3.4000e-01, -1.4901e-08])

    >>> # Generates a periodic Blackman window.
    >>> torch.signal.windows.blackman(5, sym=False)
    tensor([-1.4901e-08,  2.0077e-01,  8.4923e-01,  8.4923e-01,  2.0077e-01])
""".format(**window_common_args),
)
def blackman(
    M: int,
    *,
    sym: bool = True,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks("blackman", M, dtype, layout)

    return general_cosine(
        M,
        a=[0.42, 0.5, 0.08],
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


@_add_docstr(
    r"""
Computes the Bartlett window.

The Bartlett window is defined as follows:

.. math::
    w_n = 1 - \left| \frac{2n}{M - 1} - 1 \right| = \begin{cases}
        \frac{2n}{M - 1} & \text{if } 0 \leq n \leq \frac{M - 1}{2} \\
        2 - \frac{2n}{M - 1} & \text{if } \frac{M - 1}{2} < n < M \\ \end{cases}
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Bartlett window.
    >>> torch.signal.windows.bartlett(10)
    tensor([0.0000, 0.2222, 0.4444, 0.6667, 0.8889, 0.8889, 0.6667, 0.4444, 0.2222, 0.0000])

    >>> # Generates a periodic Bartlett window.
    >>> torch.signal.windows.bartlett(10, sym=False)
    tensor([0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000, 0.8000, 0.6000, 0.4000, 0.2000])
""".format(**window_common_args),
)
def bartlett(
    M: int,
    *,
    sym: bool = True,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks("bartlett", M, dtype, layout)

    if M == 0:
        return torch.empty(
            (0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )

    if M == 1:
        return torch.ones(
            (1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )

    start = -1
    constant = 2 / (M if not sym else M - 1)

    k = torch.linspace(
        start=start,
        end=start + (M - 1) * constant,
        steps=M,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )

    return 1 - torch.abs(k)


@_add_docstr(
    r"""
Computes the general cosine window.

The general cosine window is defined as follows:

.. math::
    w_n = \sum^{M-1}_{i=0} (-1)^i a_i \cos{ \left( \frac{2 \pi i n}{M - 1}\right)}
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    a (Iterable): the coefficients associated to each of the cosine functions.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric general cosine window with 3 coefficients.
    >>> torch.signal.windows.general_cosine(10, a=[0.46, 0.23, 0.31], sym=True)
    tensor([0.5400, 0.3376, 0.1288, 0.4200, 0.9136, 0.9136, 0.4200, 0.1288, 0.3376, 0.5400])

    >>> # Generates a periodic general cosine window with 2 coefficients.
    >>> torch.signal.windows.general_cosine(10, a=[0.5, 1 - 0.5], sym=False)
    tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])
""".format(**window_common_args),
)
def general_cosine(
    M,
    *,
    a: Iterable,
    sym: bool = True,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    if dtype is None:
        dtype = torch.get_default_dtype()

    _window_function_checks("general_cosine", M, dtype, layout)

    if M == 0:
        return torch.empty(
            (0,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )

    if M == 1:
        return torch.ones(
            (1,), dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )

    if not isinstance(a, Iterable):
        raise TypeError("Coefficients must be a list/tuple")

    if not a:
        raise ValueError("Coefficients cannot be empty")

    constant = 2 * torch.pi / (M if not sym else M - 1)

    k = torch.linspace(
        start=0,
        end=(M - 1) * constant,
        steps=M,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )

    a_i = torch.tensor(
        [(-1) ** i * w for i, w in enumerate(a)],
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
    )
    i = torch.arange(
        a_i.shape[0],
        dtype=a_i.dtype,
        device=a_i.device,
        requires_grad=a_i.requires_grad,
    )
    return (a_i.unsqueeze(-1) * torch.cos(i.unsqueeze(-1) * k)).sum(0)


@_add_docstr(
    r"""
Computes the general Hamming window.

The general Hamming window is defined as follows:

.. math::
    w_n = \alpha - (1 - \alpha) \cos{ \left( \frac{2 \pi n}{M-1} \right)}
    """,
    r"""

{normalization}

Arguments:
    {M}

Keyword args:
    alpha (float, optional): the window coefficient. Default: 0.54.
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

Examples::

    >>> # Generates a symmetric Hamming window with the general Hamming window.
    >>> torch.signal.windows.general_hamming(10, sym=True)
    tensor([0.0800, 0.1876, 0.4601, 0.7700, 0.9723, 0.9723, 0.7700, 0.4601, 0.1876, 0.0800])

    >>> # Generates a periodic Hann window with the general Hamming window.
    >>> torch.signal.windows.general_hamming(10, alpha=0.5, sym=False)
    tensor([0.0000, 0.0955, 0.3455, 0.6545, 0.9045, 1.0000, 0.9045, 0.6545, 0.3455, 0.0955])
""".format(**window_common_args),
)
def general_hamming(
    M,
    *,
    alpha: float = 0.54,
    sym: bool = True,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return general_cosine(
        M,
        a=[alpha, 1.0 - alpha],
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


@_add_docstr(
    r"""
Computes the minimum 4-term Blackman-Harris window according to Nuttall.

.. math::
    w_n = 1 - 0.36358 \cos{(z_n)} + 0.48917 \cos{(2z_n)} - 0.13659 \cos{(3z_n)} + 0.01064 \cos{(4z_n)}

where :math:`z_n = \frac{2 \pi n}{M}`.
    """,
    """

{normalization}

Arguments:
    {M}

Keyword args:
    {sym}
    {dtype}
    {layout}
    {device}
    {requires_grad}

References::

    - A. Nuttall, "Some windows with very good sidelobe behavior,"
      IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 1, pp. 84-91,
      Feb 1981. https://doi.org/10.1109/TASSP.1981.1163506

    - Heinzel G. et al., "Spectrum and spectral density estimation by the Discrete Fourier transform (DFT),
      including a comprehensive list of window functions and some new flat-top windows",
      February 15, 2002 https://holometer.fnal.gov/GH_FFT.pdf

Examples::

    >>> # Generates a symmetric Nuttall window.
    >>> torch.signal.windows.general_hamming(5, sym=True)
    tensor([3.6280e-04, 2.2698e-01, 1.0000e+00, 2.2698e-01, 3.6280e-04])

    >>> # Generates a periodic Nuttall window.
    >>> torch.signal.windows.general_hamming(5, sym=False)
    tensor([3.6280e-04, 1.1052e-01, 7.9826e-01, 7.9826e-01, 1.1052e-01])
""".format(**window_common_args),
)
def nuttall(
    M: int,
    *,
    sym: bool = True,
    dtype: torch.dtype | None = None,
    layout: torch.layout = torch.strided,
    device: torch.device | None = None,
    requires_grad: bool = False,
) -> Tensor:
    return general_cosine(
        M,
        a=[0.3635819, 0.4891775, 0.1365995, 0.0106411],
        sym=sym,
        dtype=dtype,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )
<|endoftext|>import copy
import dataclasses
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from itertools import chain
from typing import Any

from torch.utils._appending_byte_serializer import (
    AppendingByteSerializer,
    BytesReader,
    BytesWriter,
)
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CacheArtifact(ABC):
    """
    Data for each cache artifact that will be serialized and deserialized
    """

    key: str
    content: bytes = dataclasses.field(repr=False)  # Do not display potential binary

    @staticmethod
    def serialize(writer: BytesWriter, cls: "CacheArtifact") -> None:
        writer.write_str(cls.key)
        writer.write_bytes(cls.content)

    @staticmethod
    def deserialize(artifact_type: str, reader: BytesReader) -> "CacheArtifact":
        key = reader.read_str()
        content = reader.read_bytes()
        return CacheArtifactFactory.create(artifact_type, key, content)

    @staticmethod
    def encode(content: Any) -> bytes:
        if not isinstance(content, bytes):
            raise AssertionError(f"Expected bytes, got {type(content)}")
        return content

    @abstractmethod
    def populate_cache(self) -> None:
        pass

    @staticmethod
    def type() -> str:
        """
        Returns the type of the artifact. Must be unique across all CacheArtifact classes.

        CacheArtifactFactory.register will add property method to CacheInfo based on this (def {type}_artifacts)
        that returns all artifacts for specific cache.
        """
        raise RuntimeError("CacheArtifact is an abstract class, please use a subclass")


class CacheArtifactFactory:
    """
    Factory for creating CacheArtifact objects based on their type
    """

    _artifact_types: dict[str, type[CacheArtifact]] = {}

    @classmethod
    def register(cls, artifact_cls: type[CacheArtifact]) -> type[CacheArtifact]:
        artifact_type_key = artifact_cls.type()
        if artifact_cls.type() in cls._artifact_types:
            raise AssertionError(
                f"Artifact of type={artifact_type_key} already registered in mega-cache artifact factory"
            )
        cls._artifact_types[artifact_type_key] = artifact_cls
        setattr(
            CacheInfo,
            f"{artifact_type_key}_artifacts",
            property(lambda self: self.artifacts[artifact_type_key]),
        )
