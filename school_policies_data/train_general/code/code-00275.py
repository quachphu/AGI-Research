    return gm.graph.call_function(int_to_valid_dtype, (val,))


def insert_quantized_node(
    gm: torch.fx.GraphModule,
    val_node: torch.fx.Node,
    scale_node: float | torch.fx.Node,
    zero_point_node: float | torch.fx.Node,
    qmin_node: float | int | torch.fx.Node,
    qmax_node: float | int | torch.fx.Node,
    dtype_node: torch.dtype | torch.fx.Node,
    qscheme: torch.qscheme | None,
) -> torch.fx.Node:
    return gm.graph.call_function(
        quantize_per_tensor,
        (
            val_node,
            scale_node,
            zero_point_node,
            qmin_node,
            qmax_node,
            dtype_node,
        ),
    )


def get_dequantized(
    val: torch.Tensor,
    scale: float | torch.Tensor,
    zero_point: float | torch.Tensor,
    qmin: float | int,
    qmax: float | int,
    dtype: torch.dtype,
    axis: int | None,
    qscheme: torch.qscheme | None,
) -> torch.Tensor:
    if qscheme is torch.per_tensor_affine:
        return dequantize_per_tensor(
            val,
            scale,  # type: ignore[arg-type]
            zero_point,  # type: ignore[arg-type]
            qmin,  # type: ignore[arg-type]
            qmax,  # type: ignore[arg-type]
            dtype,
        )
    elif qscheme is torch.per_channel_affine:
        return dequantize_per_channel(
            val,
            scale,  # type: ignore[arg-type]
            zero_point,  # type: ignore[arg-type]
            axis,  # type: ignore[arg-type]
            qmin,  # type: ignore[arg-type]
            qmax,  # type: ignore[arg-type]
            dtype,
        )
    else:
        raise RuntimeError(f"Unsupported dequantization scheme: {qscheme}")


def insert_dequantized_node(
    gm: torch.fx.GraphModule,
    val_node: torch.fx.Node,
    scale_node: float | torch.fx.Node,
    zero_point_node: float | torch.fx.Node,
    qmin_node: float | int | torch.fx.Node,
    qmax_node: float | int | torch.fx.Node,
    dtype_node: torch.dtype | torch.fx.Node,
    axis_node: int | torch.fx.Node | None,
    qscheme: torch.qscheme | None,
) -> torch.fx.Node:
    if qscheme is torch.per_tensor_affine:
        return gm.graph.call_function(
            dequantize_per_tensor,
            (
                val_node,
                scale_node,
                zero_point_node,
                qmin_node,
                qmax_node,
                dtype_node,
            ),
        )
    elif qscheme is torch.per_channel_affine:
        return gm.graph.call_function(
            dequantize_per_channel,
            (
                val_node,
                scale_node,
                zero_point_node,
                axis_node,
                qmin_node,
                qmax_node,
                dtype_node,
            ),
        )
    else:
        raise RuntimeError(f"Unsupported dequantization scheme: {qscheme}")


def get_qmin_qmax(dtype: torch.dtype) -> tuple[int | float, int | float]:
    return calculate_qmin_qmax(None, None, False, dtype, False)  # type: ignore[arg-type]


def insert_qmin_qmax_node(
    gm: torch.fx.GraphModule, dtype_node: torch.dtype | torch.fx.Node
) -> tuple[torch.fx.Node, torch.fx.Node]:
    q_min_max_node = gm.graph.call_function(
        calculate_qmin_qmax, (None, None, False, dtype_node, False)
    )
    qmin_node = gm.graph.call_function(operator.getitem, (q_min_max_node, 0))
    qmax_node = gm.graph.call_function(operator.getitem, (q_min_max_node, 1))
    return qmin_node, qmax_node


def get_script_object(
    gm: torch.nn.Module, node: torch.fx.Node
) -> torch._C.ScriptObject:
    if not isinstance(node, torch.fx.Node):
        raise AssertionError(f"expected fx.Node, got {type(node).__name__}")
    if node.op != "get_attr":
        raise AssertionError(f"expected get_attr op, got {node.op}")
    attr_name = node.target
    if not isinstance(attr_name, str):
        raise AssertionError(f"expected str target, got {type(attr_name).__name__}")

    mod = gm
    for attr in attr_name.split("."):
        mod = getattr(mod, attr)
    if not isinstance(mod, torch._C.ScriptObject):
        raise AssertionError(f"expected ScriptObject, got {type(mod).__name__}")
    return mod


def insert_weight_and_bias_get_attr_node_from_get_attr_to_scriptobject(
    gm: torch.fx.GraphModule,
    param_node: torch.fx.Node,
) -> tuple[torch.fx.Node, torch.fx.Node | None]:
    """Directly inline tensor from a get_attr fx node."""
    mod = get_script_object(gm, param_node)
    w_qtensor, b_qtensor = mod.unpack()  # type: ignore[attr-defined]
    w_attr_name, b_attr_name = (
        f"dequantized_{param_node.target}_w",
        f"dequantized_{param_node.target}_b",
    )
    return insert_weight_and_bias_get_attr_node(
        gm, w_qtensor, b_qtensor, w_attr_name, b_attr_name
    )


def insert_weight_and_bias_get_attr_node_from_get_attr_to_qtensor(
    gm: torch.fx.GraphModule,
    get_attr_to_weight_node: torch.fx.Node,
    get_attr_to_bias_node: torch.fx.Node | None,
) -> tuple[torch.fx.Node, torch.fx.Node | None]:
    if not isinstance(get_attr_to_weight_node.target, str):
        raise AssertionError(
            f"expected str target, got {type(get_attr_to_weight_node.target).__name__}"
        )
    w_qtensor = getattr(gm, get_attr_to_weight_node.target)
    w_attr_name = f"dequantized_{get_attr_to_weight_node.target}_w"

    if get_attr_to_bias_node is not None:
        if not isinstance(get_attr_to_bias_node.target, str):
            raise AssertionError(
                f"expected str target, got {type(get_attr_to_bias_node.target).__name__}"
            )
        b_qtensor = getattr(gm, get_attr_to_bias_node.target)
        b_attr_name = f"dequantized_{get_attr_to_bias_node.target}_b"
    else:
        b_qtensor, b_attr_name = None, ""

    return insert_weight_and_bias_get_attr_node(
        gm, w_qtensor, b_qtensor, w_attr_name, b_attr_name
    )


def insert_weight_and_bias_get_attr_node(
    gm: torch.fx.GraphModule,
    w_qtensor: torch.Tensor,
    b_qtensor: torch.Tensor | None,
    w_attr_name: str,
    b_attr_name: str,
) -> tuple[torch.fx.Node, torch.fx.Node | None]:
    w_tensor = get_tensor_from_qtensor(w_qtensor)
    _assign_attr(w_tensor, gm, w_attr_name)
    w_tensor_attr = gm.graph.get_attr(w_attr_name)

    if b_qtensor is not None:
        b_tensor = get_tensor_from_qtensor(b_qtensor, dequant=False)
        _assign_attr(b_tensor, gm, b_attr_name)
        b_tensor_attr = gm.graph.get_attr(b_attr_name)
    else:
        b_tensor_attr = None

    return w_tensor_attr, b_tensor_attr


def get_tensor_from_qtensor(
    qtensor: torch.Tensor, dequant: bool = True
) -> torch.Tensor:
    # Manual conversion because qint8 is not used anymore.
    if qtensor.dtype in [torch.qint8, torch.quint8]:
        tensor = qtensor.int_repr()
    else:
        tensor = qtensor

    # Weights need dequantization with scaling and zero_point adjustment, but
    # bias does not need that.
    if dequant:
        qscheme = qtensor.qscheme()
        if qscheme == torch.per_channel_affine:
            scale, zero_point, axis = (
                qtensor.q_per_channel_scales(),
                qtensor.q_per_channel_zero_points(),
                qtensor.q_per_channel_axis(),
            )
        else:
            scale, zero_point, axis = (
                qtensor.q_scale(),  # type: ignore[assignment]
                qtensor.q_zero_point(),  # type: ignore[assignment]
                None,
            )
        dtype = tensor.dtype
        qmin, qmax = get_qmin_qmax(dtype)
        return get_dequantized(
            tensor, scale, zero_point, qmin, qmax, dtype, axis, qscheme
        )
    return tensor


def insert_fused_activation_node(
    gm: torch.fx.GraphModule, opname: str, fx_node: torch.fx.Node
) -> torch.fx.Node:
    if opname in ["conv1d_relu", "conv2d_relu", "linear_relu", "add_relu", "mul_relu"]:
        fx_node = gm.graph.call_function(torch.ops.aten.relu, (fx_node,))
    return fx_node


def _conv1d_op_with_squeeze(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
) -> torch.Tensor:
    # In quantized version, conv1d is emulated using conv2d with squeeze and unsqueeze
    # operations before and after the conv2d operation to match the dimension of weights.
    # Reference: https://github.com/pytorch/pytorch/blob/eca0cb0fbe84bb0a34fa94afe261bceecd52c436/aten/src/ATen/native/quantized/cpu/qconv.cpp#L1827
    s_inp = torch.ops.aten.unsqueeze(inp, 2)
    conv1d_res = torch.ops.aten.conv2d(
        s_inp,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )
    uns_conv1d_res = torch.ops.aten.squeeze(conv1d_res, 2)
    return uns_conv1d_res


def _transform_conv_with_packedparam(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """Conv specific transformation function."""
    if not isinstance(node.target, torch._ops.OpOverload):
        raise AssertionError(f"expected OpOverload, got {type(node.target).__name__}")
    opname = node.target._opname
    scale_node, zero_point_node = node.args[2], node.args[3]

    op_f = (
        torch.ops.aten.conv2d
        if opname in ["conv2d", "conv2d_relu"]
        else _conv1d_op_with_squeeze
    )

    inp_node, param_node = node.args[0], node.args[1]
    if not isinstance(inp_node, torch.fx.Node):
        raise AssertionError(f"expected fx.Node for inp, got {type(inp_node)}")
    if not isinstance(param_node, torch.fx.Node):
        raise AssertionError(f"expected fx.Node for param, got {type(param_node)}")

    if param_node.op == "call_function":
        # Using Conv2dPrepackParam from conv_prepack.
        # We directly skip the packing call and inline weights and bias.
        w_node, b_node = param_node.args[0], param_node.args[1]
        if not isinstance(w_node, torch.fx.Node):
            raise AssertionError(f"expected fx.Node for w, got {type(w_node)}")
        if b_node is not None and not isinstance(b_node, torch.fx.Node):
            raise AssertionError(f"expected fx.Node for b, got {type(b_node)}")
        (
            param_0,
            param_1,
        ) = insert_weight_and_bias_get_attr_node_from_get_attr_to_qtensor(
            gm, w_node, b_node
        )
        op_res_node = gm.graph.call_function(
            op_f, (inp_node, param_0, param_1, *param_node.args[2:])
        )
    else:
        # Using ConvPrepackedParam.
        param = get_script_object(gm, param_node)
        (
            param_0,
            param_1,
        ) = insert_weight_and_bias_get_attr_node_from_get_attr_to_scriptobject(
            gm, param_node
        )  # type: ignore[assignment]
        op_res_node = gm.graph.call_function(
            op_f,
            (
                inp_node,
                param_0,
                param_1,
                param.stride(),  # type: ignore[attr-defined]
                param.padding(),  # type: ignore[attr-defined]
                param.dilation(),  # type: ignore[attr-defined]
                param.groups(),  # type: ignore[attr-defined]
            ),
        )
    return op_res_node, scale_node, zero_point_node


def _transform_linear_with_packedparam(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """Linear specific transformation function."""
    scale_node, zero_point_node = node.args[2], node.args[3]

    inp_node, param_node = node.args[0], node.args[1]
    if not isinstance(inp_node, torch.fx.Node):
        raise AssertionError(f"expected fx.Node for inp, got {type(inp_node)}")
    if not isinstance(param_node, torch.fx.Node):
        raise AssertionError(f"expected fx.Node for param, got {type(param_node)}")

    if param_node.op == "call_function":
        # Using LinearPrepackParam from linear_prepack.
        # We directly skip the packing call and inline weights and bias.
        w_node, b_node = param_node.args[0], param_node.args[1]
        if not isinstance(w_node, torch.fx.Node):
            raise AssertionError(f"expected fx.Node for w, got {type(w_node)}")
        if b_node is not None and not isinstance(b_node, torch.fx.Node):
            raise AssertionError(f"expected fx.Node for b, got {type(b_node)}")
        (
            param_0,
            param_1,
        ) = insert_weight_and_bias_get_attr_node_from_get_attr_to_qtensor(
            gm, w_node, b_node
        )
        op_res_node = gm.graph.call_function(
            torch.ops.aten.linear, (inp_node, param_0, param_1, *param_node.args[2:])
        )
    else:
        # Using LinearPackedParams.
        (
            param_0,
            param_1,
        ) = insert_weight_and_bias_get_attr_node_from_get_attr_to_scriptobject(
            gm, param_node
        )  # type: ignore[assignment]
        op_res_node = gm.graph.call_function(
            torch.ops.aten.linear, (inp_node, param_0, param_1)
        )
    return op_res_node, scale_node, zero_point_node


def _transform_op_where_last_two_arguments_are_scale_and_zero_point(
    gm: torch.fx.GraphModule, node: torch.fx.Node
):
    """
    This transformation function can be used for function where the last two
    parameters are scale and zero point. Additionally, the function's parameters
    do not need any unpacking.
    """
    to_standard_op = {
        "mul": torch.ops.aten.mul,
        "mul_relu": torch.ops.aten.mul,
        "add": torch.ops.aten.add,
        "add_relu": torch.ops.aten.add,
        "softmax": torch.ops.aten.softmax,
        "cat": torch.ops.aten.cat,
        "hardswish": torch.ops.aten.hardswish,
    }

    if not isinstance(node.target, torch._ops.OpOverload):
        raise AssertionError(f"expected OpOverload, got {type(node.target).__name__}")
    opname, args = node.target._opname, node.args
    scale_node, zero_point_node = args[-2], args[-1]
    op_res_node = gm.graph.call_function(to_standard_op[opname], tuple(args[:-2]))
    return op_res_node, scale_node, zero_point_node


def _transform_scalar_arithmetic(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """Transform scalar overload for basic arithmetic."""
    to_standard_op = {
        "mul": torch.ops.aten.mul.Scalar,
        "add": torch.ops.aten.add.Scalar,
    }
    if not isinstance(node.target, torch._ops.OpOverload):
        raise AssertionError(f"expected OpOverload, got {type(node.target).__name__}")
    opname, args = node.target._opname, node.args
    op_res_node = gm.graph.call_function(to_standard_op[opname], args)
    return op_res_node, _SCALE, _ZERO_POINT


def _transform_prepacked_op(gm: torch.fx.GraphModule, node: torch.fx.Node):
    """
    Transformation for functions under prepacked namespace, where they share
    the same handling logic that [...]OpContext contains all parameters.
    """
    if not isinstance(node.target, torch._ops.OpOverload):
        raise AssertionError(f"expected OpOverload, got {type(node.target).__name__}")
    opname, args = node.target._opname, node.args
    op_f = None
    if opname == "conv2d_clamp_run":
        op_f = torch.ops.aten.conv2d
    elif opname == "linear_clamp_run":
        op_f = torch.ops.aten.linear
    else:
        raise RuntimeError(f"Invalid operator {opname}")

    if not isinstance(args[1], torch.fx.Node):
        raise AssertionError(f"expected fx.Node for args[1], got {type(args[1])}")
    so = get_script_object(gm, args[1])

    func_args = []
    func_args += [args[0]]
    func_args += so.unpack()[:2]  # type: ignore[attr-defined]
    if opname == "conv2d_clamp_run":
        func_args += torch.ops.prepacked.unpack_prepacked_sizes_conv2d(so)[2:]

    op_res_node = gm.graph.call_function(op_f, tuple(func_args))
    return op_res_node


def _transform_batch_norm(gm: torch.fx.GraphModule, node: torch.fx.Node):
    args = node.args
    scale_node, zero_point_node = args[-2], args[-1]
    op_res_node = gm.graph.call_function(
        torch.ops.aten.native_batch_norm, (*args[:-3], False, 0.1, args[-3])
    )
    op_res_node = gm.graph.call_function(operator.getitem, (op_res_node, 0))
    return op_res_node, scale_node, zero_point_node


def fx_transform_quantized_op_to_standard_op(
    gm: torch.fx.GraphModule, node: torch.fx.Node
) -> torch.fx.Node:
    global _SCALE, _ZERO_POINT, _INPUT_Q_DTYPE

    if not isinstance(node.target, torch._ops.OpOverload):
        raise AssertionError(f"expected OpOverload, got {type(node.target).__name__}")
    opname, overload = node.target._opname, node.target._overloadname

    key = f"{opname}.{overload}"
    opname_to_transform_f = {
        "conv1d.new": _transform_conv_with_packedparam,
        "conv1d_relu.new": _transform_conv_with_packedparam,
        "conv1d.default": _transform_conv_with_packedparam,
        "conv1d_relu.default": _transform_conv_with_packedparam,
        "conv2d.new": _transform_conv_with_packedparam,
        "conv2d_relu.new": _transform_conv_with_packedparam,
        "conv2d.default": _transform_conv_with_packedparam,
        "conv2d_relu.default": _transform_conv_with_packedparam,
        "linear.default": _transform_linear_with_packedparam,
        "linear_relu.default": _transform_linear_with_packedparam,
        "add.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "add_relu.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "mul.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "mul_relu.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "softmax.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "cat.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "hardswish.default": _transform_op_where_last_two_arguments_are_scale_and_zero_point,
        "batch_norm2d.default": _transform_batch_norm,
        "mul.Scalar": _transform_scalar_arithmetic,
        "add.Scalar": _transform_scalar_arithmetic,
    }

    if f"{key}" not in opname_to_transform_f:
        raise RuntimeError(f"Unsupported quantized op during transformation: {key}")

    op_res_node, scale_node, zero_point_node = opname_to_transform_f[f"{key}"](gm, node)

    # Add fused activation layer.
    op_res_node = insert_fused_activation_node(gm, opname, op_res_node)
    _SCALE, _ZERO_POINT = scale_node, zero_point_node

    if _INPUT_Q_DTYPE is None:
        raise AssertionError("_INPUT_Q_DTYPE should not be None")
    qmin_node, qmax_node = insert_qmin_qmax_node(gm, _INPUT_Q_DTYPE)
    q_fx_node = insert_quantized_node(
        gm,
        op_res_node,
        scale_node,
        zero_point_node,
        qmin_node,
        qmax_node,
        _INPUT_Q_DTYPE,
        torch.per_tensor_affine,
    )
    dq_fx_node = insert_dequantized_node(
        gm,
        q_fx_node,
        scale_node,
        zero_point_node,
        qmin_node,
        qmax_node,
        _INPUT_Q_DTYPE,
        None,
        torch.per_tensor_affine,
    )
    return dq_fx_node


def replace_quantized_ops_with_standard_ops(gm: torch.fx.GraphModule):
    """
    Replace legacy quantized ops (aten.quantize_per_tensor, quantized.conv) with
    PT2 ops (quantize_decomposed.quantize_per_tensor, aten.conv).

    Before:    x || -> aten.q        || -> quantized.conv2d     || -> quantized.linear    || -> aten.dq || -> y

    After:     x || -> qd.q -> qd.dq || -> aten.conv2d -> qd.q -> qd.dq || aten.linear -> qd.q -> qd.dq || -> y

    (qd == quantized_decomposed library, q = quantize, dq = dequantize)
                                          ^
                                          |
                getattr(w), getattr(b) from Conv2dParamPrepack

    During each iteration, the transformation spits out the transformed operator, its quantized output,
    and its dequantized value together. We did this because dequantization need to use the
    scale and zero point parameters from the quantization to recover the approximate original value. After each
    iteration, the new dequantization node will be used as the input to the next node (e.g., dq2 -> linear).

    For operators like conv2d and linear, their weights and bias are packed in a quantized format in the ScriptObject.
    During the transformation, we unpack those objects, get their dequantized tensor, populate those
    as attributes to the module, and use getattr to access them.

    One exception in the transformation is conv_prepack and linear_prepack. Those calls pack
    weight and bias constant tensors into ScriptObject, which are then used by subsequent conv2d or linear calls.
    During transformation, we directly skip transforming conv_prepack or linear_prepack. We check whether ScriptObject to the
    quantized::conv2d or linear is from conv_prepack or linear_prepack. If it is, we then inline those parameters
    to the operator by converting them to a getattr fx.node.

    For prepacked::conv2d_clamp_run and prepacked::linear_clamp_run, we directly convert them to aten.conv2d and aten.linear
    without the need of doing de/quantization.

    Three global variables defined are _INPUT_Q_DTYPE, _SCALE, _ZERO_POINT. _INPUT_Q_DTYPE determines the de/quantization
    data type, which is the same across the entire program, but it only shows up in the very first quantization
    call. _SCALE and _ZERO_POINT are used only when operators do not have those specified. E.g., mul.Scalar.
    """

    global _INPUT_Q_DTYPE

    quantized = False

    last_quantized_node = None
    # pyrefly: ignore [bad-assignment]
    for node in gm.graph.nodes:
        if isinstance(node.target, OpOverload):
            with gm.graph.inserting_before(node):
                namespace, opname = node.target.namespace, node.target._opname
                if namespace == "quantized" and opname not in [
                    "conv_prepack",
                    "linear_prepack",
                ]:
                    quantized = True
                    fx_node = fx_transform_quantized_op_to_standard_op(gm, node)
                    node.replace_all_uses_with(fx_node)
                    last_quantized_node = fx_node
                elif namespace == "prepacked":
                    quantized = True
                    fx_node = _transform_prepacked_op(gm, node)
                    node.replace_all_uses_with(fx_node)
                    last_quantized_node = fx_node
                elif namespace == "aten" and opname == "quantize_per_tensor":
                    inp_node, scale_node, zero_point_node, dtype_node = node.args
                    dtype_node = fx_enum_to_dtype(gm, dtype_node)
                    _INPUT_Q_DTYPE = dtype_node
                    qmin_node, qmax_node = insert_qmin_qmax_node(gm, dtype_node)
                    q_fx_node = insert_quantized_node(
                        gm,
                        inp_node,
                        scale_node,
                        zero_point_node,
                        qmin_node,
                        qmax_node,
                        dtype_node,
                        torch.per_tensor_affine,
                    )
                    dq_fx_node = insert_dequantized_node(
                        gm,
                        q_fx_node,
                        scale_node,
                        zero_point_node,
                        qmin_node,
                        qmax_node,
                        dtype_node,
                        None,
                        torch.per_tensor_affine,
                    )
                    node.replace_all_uses_with(dq_fx_node)
                    last_quantized_node = dq_fx_node
                elif namespace == "aten" and opname == "dequantize":
                    if last_quantized_node is None:
                        raise AssertionError("last_quantized_node should not be None")
                    node.replace_all_uses_with(last_quantized_node)
                else:
                    last_quantized_node = node

    # Post-processing again to remove legacy ScriptObjects and quantized tensors
    # stored as attributes or in the buffer. This is used to clean up the GraphModule
    # to not trigger tracing errors like missing __obj_flatten__ functions.
    def _clean_attr(mod: torch.nn.Module):
        for submod in mod.modules():
            attr_names_to_clean = set()
            for k, v in submod.__dict__.items():
                if isinstance(v, torch.ScriptObject):
                    attr_names_to_clean.add(k)
                if k == "_buffers":
                    buffer_name_to_clean = set()

                    for b_name, b_value in v.items():
                        if isinstance(b_value, torch.Tensor) and b_value.dtype in [
                            torch.qint8,
                            torch.quint8,
                        ]:
                            buffer_name_to_clean.add(b_name)
                    for b_name in buffer_name_to_clean:
                        v.pop(b_name, None)
            for attr_name in attr_names_to_clean:
                delattr(submod, attr_name)

    if quantized:
        """
        TODO: SetAttr + quantized ops will result incorrect program. This flag is used to temporarily
        bypass test cases.

        The deadcode elimination pass is needed to remove legacy quantized ops. Otherwise, retracing
        will throw errors. However, the current way of SetAttr does inplace update to attributes, so
        this pass regard them as dead code and remove them. Below is an example of GraphModule before
        and after the dead code elimination pass.

        class GraphModule(torch.nn.Module):
            def forward(self, x_1):
                # No stacktrace found for following nodes
                data = self.data;  data = None
                data_1 = self.data
                add_tensor = torch.ops.aten.add.Tensor(data_1, x_1, alpha = 1);  data_1 = None
                data_2 = self.data
                copy_ = torch_Tensor_copy_(data_2, add_tensor);  data_2 = add_tensor = copy_ = None
                data_3 = self.data
                add_tensor_1 = torch.ops.aten.add.Tensor(x_1, data_3, alpha = 1);  x_1 = data_3 = None
                return add_tensor_1

        class GraphModule(torch.nn.Module):
            def forward(self, x_1):
                # No stacktrace found for following nodes
                data_3 = self.data
                add_tensor_1 = torch.ops.aten.add.Tensor(x_1, data_3, alpha = 1);  x_1 = data_3 = None
                return add_tensor_1
        """
        gm.graph.eliminate_dead_code()
        _clean_attr(gm)
<|endoftext|>import functools
from collections import defaultdict

import torch
from torch._export.passes._node_metadata_hook import (
    _node_metadata_hook,
    _set_node_metadata_hook,
)
from torch._library.fake_profile import OpProfile, TensorMetadata


def insert_custom_op_guards(gm: torch.fx.GraphModule, ops_to_guard: set[str]) -> None:
    """
    This is used by draft_export to insert guards in front of calls to custom
    operators which have a generated fake kernel.
    """
    for node in gm.graph.nodes:
        if node.op == "call_function" and str(node.target) in ops_to_guard:
            with (
                _set_node_metadata_hook(
                    gm,
                    functools.partial(
                        _node_metadata_hook,
                        metadata={"stack_trace": node.meta.get("stack_trace")},
                    ),
                ),
                gm.graph.inserting_before(node),
            ):
                for arg in (*node.args, *node.kwargs.values()):
                    if isinstance(arg, torch.fx.Node) and isinstance(
                        arg.meta.get("val"), torch.Tensor
                    ):
                        val = arg.meta["val"]
                        gm.graph.call_function(
                            torch.ops.aten._assert_tensor_metadata.default,
                            args=(arg,),
                            kwargs={
                                "dtype": val.dtype,
                                "device": val.device,
                                "layout": val.layout,
                            },
                        )

    gm.recompile()


def get_op_profiles(
    gm: torch.fx.GraphModule, ops_to_guard: set[str]
) -> dict[str, set[OpProfile]]:
    """
    This is used by draft_export to get a list of custom operator profiles so
    that we can generate fake kernels.
    """

    def _get_op_profile(node: torch.fx.Node) -> OpProfile:
        args_profile = tuple(
            TensorMetadata.maybe_from_tensor(arg.meta.get("val"))
            if isinstance(arg, torch.fx.Node)
            else None
            for arg in (*node.args, *node.kwargs.values())
        )

        out_profile = None
        meta = node.meta.get("val")
        if meta is None:
            raise AssertionError("node.meta['val'] must not be None")
        if isinstance(meta, torch.Tensor):
            out_profile = TensorMetadata.maybe_from_tensor(meta)
        elif isinstance(meta, (list, tuple)):
            out_profile = tuple(TensorMetadata.maybe_from_tensor(m) for m in meta)  # type: ignore[assignment]
        if out_profile is None:
            raise AssertionError(
                f"out_profile must not be None for meta type {type(meta)}"
            )

        return OpProfile(args_profile, out_profile)  # type: ignore[arg-type]

    op_profiles: dict[str, set[OpProfile]] = defaultdict(set)

    for node in gm.graph.nodes:
        if node.op == "call_function" and str(node.target) in ops_to_guard:
            op_profiles[str(node.target)].add(_get_op_profile(node))

    return op_profiles
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._higher_order_ops.wrap import wrap_with_autocast

from ..utils import node_inline_, nodes_filter, nodes_first, sequential_split
from .replace_with_hop_pass_util import (
    _replace_with_hop_helper,
    _replace_with_hop_pass_helper,
    _sequential_split_and_maybe_inline_subgraphs_helper,
)


if TYPE_CHECKING:
    from torch.export.graph_signature import ExportGraphSignature


def _is_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    return (
        node
        and node.op == "call_function"
        and node.target
        in [
            torch.amp.autocast_mode._enter_autocast,
            torch.amp.autocast_mode._exit_autocast,
        ]
    )


def _is_enter_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    return (
        node
        and node.op == "call_function"
        and node.target is torch.amp.autocast_mode._enter_autocast
    )


def _is_exit_autocast_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    return (
        node
        and node.op == "call_function"
        and node.target is torch.amp.autocast_mode._exit_autocast
    )


def _is_autocast_sub_mod(node: torch.fx.Node) -> bool:
    """
    Check if the first non-placeholder node is `torch.amp.autocast_mode._enter_autocast`.
    """
    if node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError(f"expected str target, got {type(node.target)}")
        subgm = getattr(node.graph.owning_module, node.target)
        first_non_ph = nodes_first(
            subgm.graph.nodes, lambda node: node.op != "placeholder"
        )
        if (
            first_non_ph
            and first_non_ph.op == "call_function"
            and first_non_ph.target is torch.amp.autocast_mode._enter_autocast
        ):
            # TODO: check if current auto-cast type is the same as the args of
            # _enter_autocast. If so, return False, i.e. do not create a submodule.
            return True
    return False


def _check_valid_autocast_block(
    enter_autocast_node: torch.fx.Node, exit_autocast_node: torch.fx.Node
) -> None:
    if not _is_enter_autocast_node(enter_autocast_node):
        raise AssertionError(
            f"expected enter_autocast node, got {enter_autocast_node.target}"
        )
    if not _is_exit_autocast_node(exit_autocast_node):
        raise AssertionError(
            f"expected exit_autocast node, got {exit_autocast_node.target}"
        )
    if exit_autocast_node.args[0] != enter_autocast_node:
        raise AssertionError(
            "exit_autocast_node.args[0] must match enter_autocast_node"
        )


def _replace_with_hop(node: torch.fx.Node) -> None:
    if node.op != "call_module":
        raise AssertionError(f"expected call_module op, got {node.op}")
    graph: torch.fx.Graph = node.graph
    if graph.owning_module is None:
        raise AssertionError("graph.owning_module must not be None")
    gm: torch.fx.GraphModule = graph.owning_module
    if not isinstance(node.target, str):
        raise AssertionError(f"expected str target, got {type(node.target)}")
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    autocast_nodes = nodes_filter(sub_graph.nodes, _is_autocast_node)
    if len(autocast_nodes) > 0:
        if len(autocast_nodes) <= 1:
            raise AssertionError(
                f"need at least an enter node and an exit node, got {len(autocast_nodes)}"
            )
        enter_autocast_node = autocast_nodes[0]
        exit_autocast_node = autocast_nodes[-1]
        _check_valid_autocast_block(enter_autocast_node, exit_autocast_node)

        _replace_with_hop_helper(node, enter_autocast_node, wrap_with_autocast)
        sub_graph.erase_node(exit_autocast_node)
        sub_graph.erase_node(enter_autocast_node)


def _split_autocast(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    split_autocast creates a new graph module that splits the input graph module into multiple submodules
    based on the `_enter_autocast` and `_exit_autocast` nodes. It doesn't mutate the input graph module.

    Nodes between the **outer-most** `_enter_autocast` and `_exit_autocast(_enter_autocast)` are split
    into a submodule. Nested autocast regions are not split.
    `_enter_autocast` and `_exit_autocast(_enter_autocast)` nodes are in the submodule as well.

    Below is an example of splitting. A, B, C, D, E are blocks of non-autocast nodes in the original graph
    module. Nodes marked with the same number are grouped into the same submodule.
    A               # 0
    enter_autocast  # 1
    B               # 1
    exit_autocast   # 1
    C               # 2
    enter_autocast  # 3
    D               # 3
    exit_autocast   # 3
    E               # 4
    """
    enter_autocast_node_stack: list[torch.fx.Node] = []
    first_node_after_outer_most_exit: bool = False

    def node_call_back(node: torch.fx.Node) -> bool:
        nonlocal enter_autocast_node_stack, first_node_after_outer_most_exit
        increment_id = False
        if first_node_after_outer_most_exit or (
            len(enter_autocast_node_stack) == 0 and _is_enter_autocast_node(node)
        ):
            if len(enter_autocast_node_stack) != 0:
                raise AssertionError(
                    f"expected empty stack, got {len(enter_autocast_node_stack)} items"
                )
            first_node_after_outer_most_exit = False
            increment_id = True
        if _is_enter_autocast_node(node):
            enter_autocast_node_stack.append(node)
        elif _is_exit_autocast_node(node):
            if len(enter_autocast_node_stack) == 0:
                raise AssertionError("enter_autocast_node_stack must not be empty")
            last_enter_autocast_node = enter_autocast_node_stack.pop()
            if node.args[0] != last_enter_autocast_node:
                raise AssertionError("exit node args[0] must match last enter node")
            if len(enter_autocast_node_stack) == 0:
                # next node should be in the next submodule since
                # autocast block ends
                first_node_after_outer_most_exit = True
        return increment_id

    return sequential_split(gm, node_call_back)


def _sequential_split_and_maybe_inline_subgraphs(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Helper function for replace_autocast_with_hop_pass().
    Split the graph module into multiple subgraphs based on the autocast nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    Nodes between `_enter_autocast` and `_exit_autocast(_enter_autocast)` are considered
    as a subgraph.
    """
    need_replacing = any(_is_autocast_node(node) for node in gm.graph.nodes)
    if not need_replacing:
        return gm, graph_signature

    # split_autocast returns a new graph module that could have different output
    # args names. We need to fix the graph signature in `_sequential_split_and_maybe_inline_subgraphs_helper`.
    new_gm = _split_autocast(gm)

    def _maybe_inline_or_replace_with_hop(node: torch.fx.Node) -> None:
        if _is_autocast_sub_mod(node):
            _replace_with_hop(node)
        else:
            if node.op != "call_module":
                raise AssertionError(f"expected call_module op, got {node.op}")
            if not isinstance(node.target, str):
                raise AssertionError(f"expected str target, got {type(node.target)}")
            node_inline_(node)

    return _sequential_split_and_maybe_inline_subgraphs_helper(
        new_gm, graph_signature, _maybe_inline_or_replace_with_hop
    )


def replace_autocast_with_hop_pass(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
    return _replace_with_hop_pass_helper(
        gm,
        graph_signature,
        _sequential_split_and_maybe_inline_subgraphs,
    )
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch
from torch.export.exported_program import ConstantArgument, TensorArgument
from torch.fx.passes.infra.pass_base import PassBase, PassResult


if TYPE_CHECKING:
    from torch.export.exported_program import ModuleCallSignature
    from torch.export.graph_signature import ExportGraphSignature


__all__ = ["CollectTracepointsPass"]


class CollectTracepointsPass(PassBase):
    """
    Performs constant folding and constant propagation.
    """

    def __init__(
        self, specs: dict[str, ModuleCallSignature], sig: ExportGraphSignature
    ) -> None:
        super().__init__()
        self.specs = specs
        self.sig = sig

    def call(self, gm: torch.fx.GraphModule) -> PassResult | None:
        def get_arg_spec(arg) -> TensorArgument | ConstantArgument:
            if isinstance(arg, torch.fx.Node):
                if isinstance(arg.meta.get("val"), torch.Tensor):
                    return TensorArgument(name=arg.name)
                else:
                    raise AssertionError(
                        "Symint input is not implemented yet for submodule call signature."
                    )
            else:
                return ConstantArgument(name="", value=arg)

        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            nn_module_stack = None
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target is torch.ops.higher_order._export_tracepoint:
                    kind = node.kwargs["kind"]
                    if kind == "module_call_outputs":
                        nn_module_stack = node.meta["nn_module_stack"]
                    elif kind == "module_call_inputs":
                        nn_module_stack = None
                    else:
                        raise AssertionError(f"Unknown tracepoint kind: {kind}")
                elif node.meta["nn_module_stack"] == nn_module_stack:
                    node.meta["nn_module_stack"].popitem()
                else:
                    nn_module_stack = None
            nn_module_stack = None
            for node in reversed(module.graph.nodes):
                if node.op != "call_function":
                    continue
                if node.target is torch.ops.higher_order._export_tracepoint:
                    kind = node.kwargs["kind"]
                    if kind == "module_call_inputs":
                        nn_module_stack = node.meta["nn_module_stack"]
                    elif kind == "module_call_outputs":
                        nn_module_stack = None
                    else:
                        raise AssertionError(f"Unknown tracepoint kind: {kind}")
                elif node.meta["nn_module_stack"] == nn_module_stack:
                    node.meta["nn_module_stack"].popitem()
                else:
                    nn_module_stack = None

        def copy_sig(sig) -> ModuleCallSignature:
            from torch.export.exported_program import ModuleCallSignature

            return ModuleCallSignature(
                inputs=[],
                outputs=[],
                in_spec=sig.in_spec,
                out_spec=sig.out_spec,
                forward_arg_names=None,
            )

        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target is torch.ops.higher_order._export_tracepoint:
                    # There's some subtlety worth noting. Here fqn corresponds to
                    # the call name, whereas path corresponds to the module name.
                    # They are not necessarily the same! When a submodule is shared
                    # through different aliases, there are as many _export_tracepoint
                    # markers as there are aliases, since the shared submodule is
                    # wrapped once for each alias.
                    path = node.kwargs["path"]
                    fqn, _ = next(reversed(node.meta["nn_module_stack"].values()))

                    module_key = next(reversed(node.meta["nn_module_stack"]))
                    if "@" in module_key:
                        suffix = module_key.split("@")[-1]
                        path = f"{path}@{suffix}"

                        call_fqn = f"{fqn}@{suffix}"
                        if call_fqn not in self.specs:
                            self.specs[call_fqn] = copy_sig(self.specs[fqn])
                        fqn = call_fqn

                    kind = node.kwargs["kind"]
                    for i, arg in enumerate(node.args):
                        # We only update the signature of the alias used to call
                        # the submodule. Otherwise the signatures of all aliases
                        # would get conflated; the inputs/outputs of every call
                        # would be recorded in every other call as well.
                        if fqn == path:
                            if kind == "module_call_inputs":
                                self.specs[path].inputs.append(get_arg_spec(arg))
                            elif kind == "module_call_outputs":
                                self.specs[path].outputs.append(get_arg_spec(arg))
                            else:
                                raise AssertionError(f"Unknown tracepoint kind: {kind}")
                        if isinstance(arg, torch.fx.Node):
                            for user in node.users:
                                if user.op != "call_function":
                                    raise AssertionError(
                                        f"expected call_function, got {user.op}"
                                    )
                                if user.target is not operator.getitem:
                                    raise AssertionError(
                                        f"expected getitem target, got {user.target}"
                                    )
                                if not isinstance(user.args[1], int):
                                    raise AssertionError(
                                        f"expected int arg, got {type(user.args[1])}"
                                    )
                                if user.args[1] == i:
                                    user.replace_all_uses_with(arg)
                                    self.sig.replace_all_uses(user.name, arg.name)
                                    break
                    users = list(node.users)
                    for user in users:
                        if len(user.users) != 0:
                            raise AssertionError(
                                f"expected no users, got {len(user.users)}"
                            )
                        gm.graph.erase_node(user)
                    gm.graph.erase_node(node)
            return PassResult(gm, True)

        return None
<|endoftext|># mypy: allow-untyped-defs
import contextlib
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.graph_module import GraphModule


_EMPTY_NN_MODULE_STACK_KEY = "_empty_nn_module_stack_from_metadata_hook"


def _node_metadata_hook(
    node: torch.fx.Node,
    metadata: dict[str, Any] | None = None,
    fake_mode: FakeTensorMode | None = None,
) -> None:
    """
    Hook for adding the appropriate metadata to nodes that are created during a
    pass using graph.create_node. An example of how to use it:

    ```
    with _set_node_metadata_hook(gm,
        functools.partial(_node_metadata_hook, metadata={"stack_trace": "file"})
    ):
        pass(gm)
    ```

    This hook should not work for all generic cases -- specifically it assumes
    that nodes being added are only call_function nodes, and copies over the
    first argument node's nn_module_stack.
    """
    # pyrefly: ignore [bad-assignment]
    fake_mode = fake_mode or contextlib.nullcontext()

    if node.op != "call_function" or not callable(node.target):
        raise AssertionError(f"node: {node}, target: {node.target}")

    if (
        isinstance(node.target, torch._ops.OpOverload)
        and len(node.target._schema.returns) == 0
    ):
        node.meta["val"] = None
    else:
        fake_args, fake_kwargs = pytree.tree_map_only(
            torch.fx.Node, lambda arg: arg.meta["val"], (node.args, node.kwargs)
        )
        # pyrefly: ignore [bad-context-manager]
        with fake_mode, enable_python_dispatcher():
            fake_res = node.target(*fake_args, **fake_kwargs)
        node.meta["val"] = fake_res

    if metadata is not None:
        for k, v in metadata.items():
            node.meta[k] = v

    # Copy over metadata from argument nodes
    arg_meta = [
        arg.meta
        for arg in pytree.tree_flatten((node.args, node.kwargs))[0]
        if isinstance(arg, torch.fx.Node)
    ]
    if len(arg_meta) == 0:
        return
    arg_meta = arg_meta[0]

    node.meta["nn_module_stack"] = node.meta.get(
        "nn_module_stack",
        arg_meta.get(
            "nn_module_stack",
            {
                _EMPTY_NN_MODULE_STACK_KEY: (
                    _EMPTY_NN_MODULE_STACK_KEY,
                    _EMPTY_NN_MODULE_STACK_KEY,
                )
            },
        ),
    )

    node.meta["torch_fn"] = node.meta.get(
        "torch_fn",
        (
            f"{node.target.__name__}_0",
            f"{node.target.__class__.__name__}.{node.target.__name__}",
        ),
    )

    node.meta["custom"] = node.meta.get("custom", arg_meta.get("custom", {}))


@contextlib.contextmanager
def _set_node_metadata_hook(gm: torch.fx.GraphModule, f):
    """
    Takes a callable which will be called after we create a new node. The
    callable takes the newly created node as input and returns None.
    """
    if not callable(f):
        raise AssertionError("node_metadata_hook must be a callable.")

    # Add the hook to all submodules
    for m in gm.modules():
        if isinstance(m, GraphModule):
            m._register_create_node_hook(f)
    try:
        yield
    finally:
        # Restore hook for all submodules
        for m in gm.modules():
            if isinstance(m, GraphModule):
                m._unregister_create_node_hook(f)
<|endoftext|># mypy: allow-untyped-defs
import collections
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as pytree


aten = torch.ops.aten

# We would like to split modules into two subgraphs for runtime weight updates to work correctly.
# The use case and more information could be found at:
# https://docs.google.com/document/d/1inZC-8KarJ6gKB7G9egmYLx1V_dKX_apxon0w4zPC0Q/edit?usp=sharing
META_TAG = "MODULE_TYPE"
MODULE_TAG = "_MAIN_MODULE"
CONST_MODULE_TAG = "_CONST_MODULE"


def replace_node_with_constant(gm, node, constant, name=None):
    g = gm.graph

    if name:
        qualname = name
    else:
        if not hasattr(gm, "_frozen_param_count"):
            gm._frozen_param_count = 0
        i = gm._frozen_param_count

        while True:
            qualname = f"_frozen_param{i}"
            if not hasattr(gm, qualname):
                break
            i += 1

        gm._frozen_param_count = i + 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


class ConstantFolder(torch.fx.Interpreter):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        skip_constructors: bool = False,
    ):
        super().__init__(gm)
        self.node_replacements: dict[torch.fx.Node, Any] = {}
        self.replaced_uses: dict[torch.fx.Node, int] = collections.Counter()
        self.unknown_value = object()
        self.skip_constructors: bool = skip_constructors

        # overwrite this to deallocate env values if their only remaining use
        # is the output
        self.user_to_last_uses = self.node_to_last_non_output_use()

    def is_impure(self, node: torch.fx.Node) -> bool:
        if (
            node.target is torch.ops.prims.convert_element_type.default
            and node.args[0].op == "get_attr"  # type: ignore[union-attr]
            and node.args[0].meta["val"].dtype == torch.int8  # type: ignore[union-attr]
            and node.args[1] == torch.bfloat16
        ):
            # For int8_weight -> dq -> bf16_weight
            return True
        if node.target in [
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
            torch.ops.pt2e_quant.dequantize_affine,
        ]:
            # For the pattern fp32_weight -> q -> dq
            # We only folding fp32_weight -> q
            # int8_weight and leave dq in graph to be fused
            return True
        return False

    def node_to_last_non_output_use(self):
        last_non_output_use = collections.defaultdict(list)
        seen_uses = set()
        output_node = next(iter(reversed(self.module.graph.nodes)))  # type: ignore[arg-type, union-attr]

        for node in reversed(self.module.graph.nodes):  # type: ignore[arg-type, union-attr]
            if node.target == "output":
                continue

            def add_use(inp):
                if inp in seen_uses:
                    return

                seen_uses.add(inp)
                last_non_output_use[node].append(inp)

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, add_use, (node.args, node.kwargs))

            # if this node is only used in output, we want to gc it right away
            if len(node.users) == 1 and output_node in node.users:
                last_non_output_use[node].append(node)

        return last_non_output_use

    def run_node(self, node):
        if node.target == "output":
            # because we remove nodes from env on last non output use,
            # re-define them now or we'll get error in interpreter
            def set_env(arg):
                self.env[arg] = self.unknown_value

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, set_env, node.args)
            return super().run_node(node)

        args, kwargs = self.fetch_args_kwargs_from_env(node)
        flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)

        # We need to do this weird thing because in cases where flattened_inputs
        # contains a ScriptObject, equality checking results in a type error if
        # the types are different.
        if any(
            type(self.unknown_value) is type(input_) and self.unknown_value == input_
            for input_ in flattened_inputs
        ):
            return self.unknown_value

        # TODO - fix errors with this
        if (
            node.op == "call_function"
            and node.target is aten._efficientzerotensor.default
        ):
            return self.unknown_value

        # TODO - constant folding triton kernel returns the inputs -- fix this
        if (
            node.op == "call_function"
            and node.name == "triton_kernel_wrapper_functional_proxy"
        ):
            return self.unknown_value

        # skip constructors, since inductor generates optimal code for them already
        # and turning into tensor would result in an additional global memory read
        # TODO - more complicated strategy
        if (
            self.skip_constructors
            and node.op != "get_attr"
            and not any(isinstance(e, torch.Tensor) for e in flattened_inputs)
        ):
            return self.unknown_value

        # All mutations should either be removed or on inputs which we did not make constant
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return self.unknown_value

        out = super().run_node(node)

        if node.op != "get_attr" and isinstance(out, torch.Tensor):
            if out.device.type == "meta":
                return out

            if not self.insertable_tensor_check(out):
                return out

            if self.is_impure(node):
                return self.unknown_value

            self.add_node_replacement(node, out)

            flattened_node_inps = pytree.arg_tree_leaves(*node.args, **node.kwargs)

            for n in flattened_node_inps:
                if not isinstance(n, torch.fx.Node):
                    continue

                self.replaced_uses[n] += 1

            for to_delete in self.user_to_last_uses.get(node, []):
                if self.replaced_uses[to_delete] == len(to_delete.users):
                    self.node_replacements.pop(to_delete, None)

        return out

    def insertable_tensor_check(self, tensor: torch.Tensor) -> bool:
        return True

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor

    def run(self):  # type: ignore[override]
        env = {}
        for n in self.module.graph.find_nodes(op="placeholder"):  # type: ignore[operator, union-attr]
            env[n] = self.unknown_value
        return super().run(initial_env=env)


def constant_fold(
    gm: torch.fx.GraphModule,
    constraint_fn: Callable[[torch.fx.Node], bool] | None = None,
):
    with torch.utils._python_dispatch._disable_current_modes():
        cf = ConstantFolder(gm, skip_constructors=True)
        cf.run()

        for node, constant in cf.node_replacements.items():
            if constraint_fn is not None and not constraint_fn(node):
                continue
            replace_node_with_constant(gm, node, constant)

        erased_params = []
        # Get all attr users by looking up the graph instead from node.users, because in this case
        # _tensor_constant0 and _tensor_constant0_1 are actually referring to the same tensor.

        #     opcode         name                 target            args                         kwargs
        # -------------  -------------------  ----------------  ---------------------------  --------
        # placeholder    arg0_1               arg0              ()                           {}
        # get_attr       _tensor_constant0    state             ()                           {}
        # call_function  add                  aten.add.Tensor   (arg0_1, _tensor_constant0)  {}
        # get_attr       _tensor_constant0_1  state             ()                           {}
        # call_function  add_                 aten.add_.Tensor  (_tensor_constant0_1, 1)     {}
        # output         output               output            ([add],)                     {}

        get_attr_node_users = defaultdict(list)
        for node in gm.graph.nodes:
            if node.op == "get_attr":
                get_attr_node_users[node.target].extend(node.users.keys())
        for node in gm.graph.find_nodes(op="get_attr"):
            if node.op == "get_attr" and len(get_attr_node_users[node.target]) == 0:
                if hasattr(gm, node.target):
                    delattr(gm, node.target)
                erased_params.append(node)
        for node in erased_params:
            gm.graph.erase_node(node)

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()


def constant_graph_tag(gm: torch.fx.GraphModule) -> None:
    with torch.utils._python_dispatch._disable_current_modes():
        cf = ConstantFolder(gm, skip_constructors=True)
        cf.run()

        for node in gm.graph.nodes:
            if (
                node.op == "get_attr"
                or node in cf.node_replacements
                or node in cf.replaced_uses
            ):
                node.meta[META_TAG] = CONST_MODULE_TAG
            else:
                node.meta[META_TAG] = MODULE_TAG


def run_and_get_constant_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Construct a GraphModule which corresponds to the part which could be
    constant folded in provided gm.
    """

    constant_graph_tag(gm)
    # We rewrite the tags, if it's a constant being directly consumed, without
    # any folding opportunity, we keep it in main gm.
    for node in gm.graph.find_nodes(op="get_attr"):
        used_to_fold = False
        for u in node.users:
            if u.meta[META_TAG] == CONST_MODULE_TAG:
                used_to_fold = True
                break
        if not used_to_fold:
            node.meta[META_TAG] = MODULE_TAG

    new_graph = torch.fx.Graph()

    node_remapping: dict[torch.fx.Node, torch.fx.Node] = {}
    output_nodes = []
    for node in gm.graph.nodes:
        if node.meta[META_TAG] == MODULE_TAG:
            continue

        new_node = new_graph.node_copy(node, lambda x: node_remapping[x])
        node_remapping[node] = new_node

        for user in node.users:
            if user.meta[META_TAG] == MODULE_TAG:
                output_nodes.append(new_node)
                break

    new_graph.output(tuple(output_nodes))
    new_graph.lint()
    new_gm = torch.fx.GraphModule(gm, new_graph)

    return new_gm
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch._higher_order_ops.wrap import wrap_with_set_grad_enabled

from ..utils import node_inline_, nodes_filter, nodes_first, nodes_map, sequential_split
from .replace_with_hop_pass_util import (
    _replace_with_hop_helper,
    _replace_with_hop_pass_helper,
    _sequential_split_and_maybe_inline_subgraphs_helper,
)


if TYPE_CHECKING:
    from torch.export.graph_signature import ExportGraphSignature


def _is_set_grad_enabled_node(node: torch.fx.Node) -> torch.fx.Node | bool:
    return (
        node
        and node.op == "call_function"
        and node.target is torch._C._set_grad_enabled
    )


def _is_set_grad_enabled_sub_mod(
    node: torch.fx.Node, omit_if_same_with_ambient: bool = False
) -> bool | torch.Tensor:
    if node.op == "call_module":
        if not isinstance(node.target, str):
            raise AssertionError(f"expected str target, got {type(node.target)}")
        subgm = getattr(node.graph.owning_module, node.target)
        first_non_ph = nodes_first(
            subgm.graph.nodes, lambda node: node.op != "placeholder"
        )
        if (
            first_non_ph
            and first_non_ph.op == "call_function"
            and first_non_ph.target is torch._C._set_grad_enabled
        ):
            return (
                first_non_ph.args[0] != torch.is_grad_enabled()  # type: ignore[bad-return]
                if omit_if_same_with_ambient
                else True
            )
    return False


def _replace_with_hop(node: torch.fx.Node) -> None:
    if node.op != "call_module":
        raise AssertionError(f"expected call_module op, got {node.op}")
    graph: torch.fx.Graph = node.graph
    if graph.owning_module is None:
        raise AssertionError("graph.owning_module must not be None")
    gm: torch.fx.GraphModule = graph.owning_module
    if not isinstance(node.target, str):
        raise AssertionError(f"expected str target, got {type(node.target)}")
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    set_grad_nodes = nodes_filter(sub_graph.nodes, _is_set_grad_enabled_node)
    if len(set_grad_nodes) > 0:
        if len(set_grad_nodes) != 1:
            raise AssertionError(
                f"expected exactly 1 set_grad node, got {len(set_grad_nodes)}"
            )
        set_grad_node = set_grad_nodes[0]
        _replace_with_hop_helper(node, set_grad_node, wrap_with_set_grad_enabled)
        sub_graph.erase_node(set_grad_node)


def _remove_set_grad_and_inline(node: torch.fx.Node) -> None:
    if node.op != "call_module":
        raise AssertionError(f"expected call_module op, got {node.op}")
    graph: torch.fx.Graph = node.graph
    if graph.owning_module is None:
        raise AssertionError("graph.owning_module must not be None")
    gm: torch.fx.GraphModule = graph.owning_module
    if not isinstance(node.target, str):
        raise AssertionError(f"expected str target, got {type(node.target)}")
    sub_gm = getattr(gm, node.target)
    sub_graph = sub_gm.graph
    nodes_map(
        sub_graph.nodes,
        lambda n: sub_graph.erase_node(n) if _is_set_grad_enabled_node(n) else n,
    )
    node_inline_(node)


def _sequential_split_and_maybe_inline_subgraphs(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Helper function for replace_set_grad_with_hop_pass().
    Split the graph module into multiple subgraphs based on the set_grad_enabled nodes.
    For each subgraph, decides whether to construct a HOO subgraph, or inline the calls
    back into the parent graph module.
    """
    need_replacing = any(_is_set_grad_enabled_node(node) for node in gm.graph.nodes)
    if not need_replacing:
        return gm, graph_signature

    # sequential_split returns a new graph module that could have different output
    # args names. We need to fix the graph signature.
    new_gm = sequential_split(gm, _is_set_grad_enabled_node)

    def _maybe_inline_or_replace_with_hop(node: torch.fx.Node):
        if _is_set_grad_enabled_sub_mod(node, omit_if_same_with_ambient=True):
            _replace_with_hop(node)
        else:
            _remove_set_grad_and_inline(node)

    return _sequential_split_and_maybe_inline_subgraphs_helper(
        new_gm, graph_signature, _maybe_inline_or_replace_with_hop
    )


def replace_set_grad_with_hop_pass(
    gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature | None
) -> tuple[torch.fx.GraphModule, ExportGraphSignature | None]:
    """
    Split gm into sub-graph-modules using `sequential_split_and_maybe_inline_subgraphs`, and
    then recursively call itself on each of the submodules.
    """
    return _replace_with_hop_pass_helper(
        gm,
        graph_signature,
        _sequential_split_and_maybe_inline_subgraphs,
    )
<|endoftext|># mypy: allow-untyped-defs
import collections
import logging
from typing import Any

import torch
from torch._export.verifier import SpecViolationError
from torch._guards import detect_fake_mode
from torch._library.fake_class_registry import FakeScriptObject, maybe_to_fake_obj
from torch._library.opaque_object import (
    get_opaque_type_name,
    is_custom_class,
    is_opaque_symbolic_type,
)
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.export.exported_program import (
    ArgumentSpec,
    CustomObjArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    TensorArgument,
)
from torch.fx._symbolic_trace import _ConstantAttributeType
from torch.fx.graph_module import _get_attr


log = logging.getLogger(__name__)


class ConstantAttrMap(collections.abc.MutableMapping):
    """A mapping class that understands how to use module constants (tensors,
    ScriptObjects, FakeScriptObjects, opaque objects) as keys. We store tensors,
    FakeScriptObjects, and opaque objects normally, but ScriptObjects are stored
    by hash, because different torch.ScriptObjects can point to the same
    underlying value (but we guarantee that they will `hash()` to the same value
    if that's the case).
    """

    def __init__(self) -> None:
        # Underlying dict that we use to implement this mapping.
        self._constant_attrs: dict[
            int | torch.Tensor | FakeScriptObject | torch.utils._pytree.TreeSpec,
            list[Any],
        ] = {}
        # Map from the hash(ScriptObject) to the ScriptObject itself. Used for
        # APIs like `__iter__` that should look like they're returning the
        # original ScriptObjects.
        self._script_object_map: dict[int, torch.ScriptObject] = {}

    def __getitem__(self, key: _ConstantAttributeType) -> Any:
        real_key = hash(key) if isinstance(key, torch.ScriptObject) else key
        if not isinstance(
            real_key, (int, torch.Tensor, FakeScriptObject)
        ) and not is_custom_class(type(real_key)):
            raise AssertionError(
                f"expected int, Tensor, FakeScriptObject, or opaque type key, got {type(real_key)}"
            )
        return self._constant_attrs[real_key]

    def __setitem__(self, key: _ConstantAttributeType, value):
        # we shouldn't actually call this, should go to add() instead to handle aliasing
        raise NotImplementedError(
            """Directly setting values for ConstantAttrMap is not supported, please use add(key, value) instead.
The same key can be mapped to multiple values, for handling constant aliasing."""
        )

    def add(self, key: _ConstantAttributeType, value: Any) -> None:
        if isinstance(key, torch.ScriptObject):
            if hash(key) not in self._constant_attrs:
                self._constant_attrs[hash(key)] = []
            self._constant_attrs[hash(key)].append(value)
            self._script_object_map[hash(key)] = key
        elif isinstance(key, (torch.Tensor, FakeScriptObject)) or is_custom_class(
            type(key)
        ):
            if key not in self._constant_attrs:
                self._constant_attrs[key] = []
            self._constant_attrs[key].append(value)
        else:
            raise TypeError(
                f"Expected key to be a tensor or ScriptObject, got {type(key)}"
            )

    def __delitem__(self, key: _ConstantAttributeType):
        real_key = hash(key) if isinstance(key, torch.ScriptObject) else key

        del self._constant_attrs[real_key]

    def __iter__(self):
        for key in self._constant_attrs:
            if isinstance(key, int):
                yield self._script_object_map[key]
            else:
                yield key

    def __len__(self):
        return len(self._constant_attrs)

    def __contains__(self, key: object) -> bool:
        real_key = hash(key) if isinstance(key, torch.ScriptObject) else key
        return real_key in self._constant_attrs


def get_constant_fqn(node: torch.fx.Node, constant_name: str) -> str:
    # The FQN of the constant tensor in the state dict should
    # correspond to the module where the constant tensor was
    # originally used.
    if len(node.meta["nn_module_stack"]) == 0:
        return constant_name
    parent_fqn = list(node.meta["nn_module_stack"].values())[-1][0]
    if len(parent_fqn) > 0:
        return f"{parent_fqn}.{constant_name}"
    else:
        return constant_name


def _get_first_fqn(
    const_attrs: ConstantAttrMap,
    key: _ConstantAttributeType,
) -> Any:
    fqns = const_attrs.get(key)
    return fqns[0] if fqns else None


def _unused_constant(node: torch.fx.Node) -> list[torch.fx.Node] | None:
    """
    If there is a tensor constant created while tracing, here is how the graph
    looks like:

        %_tensor_constant0 : [num_users=1] = get_attr[target=_tensor_constant0]
        %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant0,))
        %detach_ : [num_users=?] = call_function[target=torch.ops.aten.detach_.default](args = (%lift_fresh_copy,))

    To check to see if the tensor constant is being used, we want to traverse to
    the detach node to see if it's actually being used.

    This function returns None if this constant is being used, otherwise it returns the
    lift_fresh and detach node to be removed later.
    """
    if len(node.users) > 1:
        return None

    lift_fresh_node = next(iter(node.users.keys()))
    if not (
        lift_fresh_node.op == "call_function"
        and lift_fresh_node.target
        in (
            torch.ops.aten.lift_fresh.default,
            torch.ops.aten.lift_fresh_copy.default,
        )
    ):
        return None

    if len(lift_fresh_node.users) > 1:
        return None

    # Case 1: lift node is not used anywhere
    if len(lift_fresh_node.users) == 0:
        return [lift_fresh_node, node]

    detach_node = next(iter(lift_fresh_node.users.keys()))
    if not (
        detach_node.op == "call_function"
        and detach_node.target
        in (
            torch.ops.aten.detach_.default,
            torch.ops.aten.detach.default,
        )
    ):
        return None

    if len(detach_node.users) > 0:
        return None
    else:
        # Case 2: Lift node's child is not used anywhere
        return [detach_node, lift_fresh_node, node]


def lift_constants_pass(
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    constant_attrs: ConstantAttrMap,
) -> dict[str, _ConstantAttributeType]:
    """
    Takes a graph module, graph signature, and modifies them inplace to lift any
    constants (tensors or custom classes) as inputs to the graph. Returns a
    dictionary of names to constants.

    Arguments:
        gm (torch.fx.GraphModule): The graph module containing the graph and constants to lift.
        graph_signature (ExportGraphSignature): This graph signature will be
            mutated to add additional CONSTANT_TENSOR and CUSTOM_OBJ inputs.
        constant_attrs (ConstantAttr): A mapping from a constant value to its
            fully-qualified path in `gm`. This is used to maintain consistent
            location of constants between the original module and the exported
            version.

    Returns:
        A dictionary of fqn => constant value.
    """
    all_constants: dict[str, _ConstantAttributeType] = {}

    input_specs = graph_signature.input_specs
    num_custom_obj = sum(
        input_spec.kind == InputKind.CUSTOM_OBJ for input_spec in input_specs
    )
    num_tensor_constants = sum(
        input_spec.kind == InputKind.CONSTANT_TENSOR for input_spec in input_specs
    )

    fake_mode = detect_fake_mode(
        tuple(node.meta["val"] for node in gm.graph.nodes if node.op == "placeholder")
    )

    first_user_input_loc, first_user_input = 0, next(iter(gm.graph.nodes))
    used_target_names = set()

    input_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(input_nodes) != len(input_specs):
        raise AssertionError(
            f"input nodes count {len(input_nodes)} != input specs count {len(input_specs)}"
        )
    for i, (node, input_spec) in enumerate(zip(input_nodes, input_specs)):
        used_target_names.add(input_spec.target)
        if input_spec.kind == InputKind.USER_INPUT:
            first_user_input = node
            first_user_input_loc = i
            break

    lifted_objs = ConstantAttrMap()
    renamed_targets = {}
    for node in list(gm.graph.nodes):
        if node.op == "get_attr":
            if nodes_to_remove := _unused_constant(node):
                # Remove the node if it's not being used
                for node_rm in nodes_to_remove:
                    gm.graph.erase_node(node_rm)
                continue

            constant_val = _get_attr(gm, node.target)
            # These are not hashable and not gonna be lifted
            # so we can skip them earlier
            if isinstance(constant_val, torch.fx.GraphModule):
                continue
            if "LoweredBackendModule" in type(constant_val).__name__:
                continue
            if "AOTInductorRunnerWrapper" in type(constant_val).__name__:
                continue
            if isinstance(constant_val, torch.utils._pytree.TreeSpec):
                continue

            if constant_val in lifted_objs:
                # We already lifted this constant elsewhere. Just rewrite uses
                # of this get_attr to point to the already-existing placeholder
                # node.
                const_placeholder_node = _get_first_fqn(lifted_objs, constant_val)
                node.replace_all_uses_with(const_placeholder_node)
                gm.graph.erase_node(node)
                renamed_targets[node.name] = const_placeholder_node.name
                continue

            # For ScriptObject, Tensor and FakeScriptObject constants:
            # First check if the constant was an attribute on some module by
            # consulting `constant_attrs` map. If it is, use the fqn that keeps
            # its location consistent with the eager module.
            #
            # If it's not in the `constant_attrs` map, that means it's an inline
            # constant (e.g. x + torch.tensor(0)), and thus did not have a
            # specific location in the eager module. In that case, just generate
            # some name and attach it to the module in which it was used.
            if isinstance(
                constant_val, (torch.ScriptObject, FakeScriptObject)
            ) or is_opaque_symbolic_type(type(constant_val)):
                constant_kind = InputKind.CUSTOM_OBJ
                constant_fqn = _get_first_fqn(constant_attrs, constant_val)
                if constant_fqn is not None:
                    constant_name = constant_fqn.replace(".", "_")
                else:
                    constant_name = f"lifted_custom_{num_custom_obj}"
                    constant_fqn = get_constant_fqn(node, constant_name)
                    while constant_fqn in used_target_names:
                        num_custom_obj += 1
                        constant_name = f"lifted_custom_{num_custom_obj}"
                        constant_fqn = get_constant_fqn(node, constant_name)
                    num_custom_obj += 1
            elif isinstance(constant_val, torch.Tensor):
                # Remove the parameterness of constant_val
                if isinstance(constant_val, torch.nn.Parameter):
                    log.debug(
                        "%s created when tracing %s is a parameter. But "
                        "it's not registered with register_parameter(). export will treat it as a constant tensor",
                        str(node.target),
                        str(node.meta.get("stack_trace", "<unknown stack>")),
                    )
                    # We get the real data out of the parameter by disabling the surrounding fake mode.
                    with unset_fake_temporarily():
                        constant_val = constant_val.data
                constant_kind = InputKind.CONSTANT_TENSOR
                constant_fqn = _get_first_fqn(constant_attrs, constant_val)
                if constant_fqn is not None:
                    constant_name = constant_fqn.replace(".", "_")
                else:
                    constant_name = f"lifted_tensor_{num_tensor_constants}"
                    constant_fqn = get_constant_fqn(node, constant_name)
                    while constant_fqn in used_target_names:
                        num_tensor_constants += 1
                        constant_name = f"lifted_tensor_{num_tensor_constants}"
                        constant_fqn = get_constant_fqn(node, constant_name)
                    num_tensor_constants += 1
            else:
                raise SpecViolationError(
                    f"getattr node {node} referencing unsupported type {type(constant_val)}"
                )

            with gm.graph.inserting_before(first_user_input):
                # Insert the constant node before the first user input
                const_placeholder_node = gm.graph.placeholder(constant_name)
                # match target name with its node name in case there is name collision
                # and suffix is added to node name in fx
                const_placeholder_node.target = const_placeholder_node.name

                for k, v in node.meta.items():
                    const_placeholder_node.meta[k] = v

                # Once the FQN has been used, remove nn_module_stack, stack_trace
                const_placeholder_node.meta.pop("nn_module_stack")
                const_placeholder_node.meta.pop("stack_trace", None)

                input_spec_arg: ArgumentSpec
                if isinstance(constant_val, torch.Tensor):
                    if fake_mode is not None:
                        const_placeholder_node.meta["val"] = fake_mode.from_tensor(
                            constant_val, static_shapes=True
                        )
                        const_placeholder_node.meta["val"].constant = constant_val
                    else:
                        const_placeholder_node.meta["val"] = constant_val
                    input_spec_arg = TensorArgument(name=const_placeholder_node.name)
                elif isinstance(constant_val, torch._C.ScriptObject):
                    class_fqn = constant_val._type().qualified_name()  # type: ignore[attr-defined]
                    const_placeholder_node.meta["val"] = CustomObjArgument(
                        constant_fqn, class_fqn
                    )
                    input_spec_arg = CustomObjArgument(
                        name=const_placeholder_node.name, class_fqn=class_fqn
                    )
                elif isinstance(constant_val, FakeScriptObject):
                    class_fqn = constant_val.script_class_name
                    const_placeholder_node.meta["val"] = CustomObjArgument(
                        constant_fqn, class_fqn, constant_val
                    )
                    input_spec_arg = CustomObjArgument(
                        name=const_placeholder_node.name,
                        class_fqn=class_fqn,
                        fake_val=constant_val,
                    )
                elif is_custom_class(type(constant_val)):
                    class_fqn = get_opaque_type_name(type(constant_val))
                    fake_val = (
                        maybe_to_fake_obj(fake_mode, constant_val)
                        if fake_mode
                        else None
                    )
                    const_placeholder_node.meta["val"] = CustomObjArgument(
                        constant_fqn,
                        class_fqn,
                        fake_val,  # pyrefly: ignore[bad-argument-type]
                    )
                    input_spec_arg = CustomObjArgument(
                        name=const_placeholder_node.name,
                        class_fqn=class_fqn,
                        fake_val=fake_val,  # pyrefly: ignore[bad-argument-type]
                    )
                else:
                    raise SpecViolationError(
                        f"tried to lift unsupported type {type(constant_val)} from node {node.format_node()}"
                    )

                lifted_objs.add(constant_val, const_placeholder_node)
                node.replace_all_uses_with(const_placeholder_node)
                gm.graph.erase_node(node)

                renamed_targets[node.name] = const_placeholder_node.name

                # Add the constant as a buffer to the graph signature
                graph_signature.input_specs.insert(
                    first_user_input_loc,
                    InputSpec(
                        kind=constant_kind,
                        arg=input_spec_arg,
                        target=constant_fqn,
                    ),
                )
                if constant_val in constant_attrs:
                    for fqn in constant_attrs[constant_val]:
                        all_constants[fqn] = constant_val
                else:
                    all_constants[constant_fqn] = constant_val
                first_user_input_loc += 1

    for spec in graph_signature.output_specs:
        if spec.arg.name in renamed_targets:
            spec.arg.name = renamed_targets[spec.arg.name]

    return all_constants


def rewrite_script_object_meta(
    gm: torch.fx.GraphModule,
) -> dict[str, _ConstantAttributeType]:
    """When tracing, we produce a graph with FakeScriptObject in the
    meta["val"].

    For now, we rewrite meta["val"] to be a placeholder CustomObjArgument
    """
    constants: dict[
        str,
        _ConstantAttributeType,
    ] = {}
    for node in gm.graph.nodes:
        if "val" not in node.meta:
            continue

        old_meta = node.meta["val"]

        if isinstance(old_meta, torch.ScriptObject):
            class_fqn = old_meta._type().qualified_name()  # type: ignore[attr-defined]
            new_meta = CustomObjArgument(node.name, class_fqn)
            constants[node.name] = old_meta
            node.meta["val"] = new_meta

        elif isinstance(old_meta, FakeScriptObject):
            class_fqn = old_meta.script_class_name  # type: ignore[attr-defined]
            new_meta = CustomObjArgument(node.name, class_fqn, old_meta)
            constants[node.name] = old_meta
            node.meta["val"] = new_meta

    return constants


def _materialize_and_lift_constants(
    gm: torch.fx.GraphModule,
    export_graph_signature: ExportGraphSignature,
    constant_attrs: ConstantAttrMap,
) -> dict[str, _ConstantAttributeType]:
    constants = rewrite_script_object_meta(gm)
    constants.update(lift_constants_pass(gm, export_graph_signature, constant_attrs))
    return constants
<|endoftext|># mypy: allow-untyped-defs
import math
import operator
import traceback
from functools import partial
from typing import NamedTuple, TYPE_CHECKING

import sympy

import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges


if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = ["InputDim"]


class InputDim(NamedTuple):
    input_name: str
    dim: int


def _convert_to_int(val):
    # Convert simple sympy Integers into concrete int
    if val in (sympy.oo, int_oo):
        return math.inf
    if val in (-sympy.oo, -int_oo):
        return -math.inf
    if isinstance(val, sympy.Integer):
        return int(val)
    raise RuntimeError("Export constraints cannot be non-integer expressions")


def _convert_range_to_int(range: ValueRanges):
    if not isinstance(range, ValueRanges):
        raise AssertionError(f"expected ValueRanges, got {type(range)}")
    min_val = _convert_to_int(range.lower)
    max_val = _convert_to_int(range.upper)
    return min_val, max_val


class _AddRuntimeAssertionsForInlineConstraintsPass(PassBase):
    def __init__(
        self,
        range_constraints: dict[sympy.Symbol, ValueRanges],
    ):
        super().__init__()
        self.range_constraints: dict[sympy.Symbol, ValueRanges] = range_constraints
        self._asserts_generated_unbacked_symbols: set[sympy.Symbol] = set()
        self.counter = 0

    def _assert_range_constraint(self, node, lower, upper, assert_msg):
        last_node = node
        if lower > -math.inf:
            last_node = self._insert_assert_async(
                last_node, operator.ge, node, lower, assert_msg
            )

        if upper < math.inf:
            last_node = self._insert_assert_async(
                last_node, operator.le, node, upper, assert_msg
            )

    def _insert_assert_async(self, last_node, op, lower, upper, assert_msg):
        """
        Inserts assert_async call_function nodes in the graph. This function is
        called **during** the interpreter-based pass.
        """
        self.counter += 1
        graph = last_node.graph
        with graph.inserting_after(last_node):
            cmp = graph.call_function(op, (lower, upper), {})
        with graph.inserting_after(cmp):
            cmp_tensor = graph.call_function(
                torch.ops.aten.scalar_tensor.default, (cmp,), {}
            )
        with graph.inserting_after(cmp_tensor):
            assert_async = graph.call_function(
                torch.ops.aten._assert_async.msg,
                (cmp_tensor, assert_msg),
                {},
            )
        return assert_async

    def call(self, graph_module) -> PassResult:
        self.existing_inline_assertions = _get_existing_inline_assertions(
            graph_module, self.range_constraints
        )

        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if "val" not in node.meta:
                    continue

                val = node.meta["val"]
                # In general, we may have to deal the case such as: ret[1].shape[0].
                # We need first find out what symbols require assertion, then we need to follow the path
                # from ret to the symbol, construct the proxies along the way and construct the messages
                # piece-wise at the same time.
                #
                # We use post-order traversal to collect all the proxies callbacks needed, construct
                # the error message callbacks, and at the top-level traversal tree we execute all the callbacks.
                # We need the callbacks because, in order to call the function to create a proxy for shape[0], we
                # need the proxy for shape, which further requires the proxy for ret[1], etc.

                def add_assertions(val):
                    call_backs: list[Callable] = []
                    messages: list[str] = []
                    if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                        symbol = val.node.expr
                        if symbol in self.existing_inline_assertions:
                            return call_backs, messages
                        if isinstance(symbol, sympy.Symbol) and free_unbacked_symbols(
                            symbol
                        ):
                            if symbol in self._asserts_generated_unbacked_symbols:
                                return call_backs, messages
                            # We only care about unbacked symints for these inline
                            # constraints, which are prefixed with 'u'
                            constraint = self.range_constraints[symbol]
                            min_val, max_val = _convert_range_to_int(constraint)
                            assert_msg = f" is outside of inline constraint [{min_val}, {max_val}]."
                            call_backs.append(
                                partial(
                                    self._assert_range_constraint,
                                    lower=min_val,
                                    upper=max_val,
                                )
                            )
                            messages.append(assert_msg)
                            self._asserts_generated_unbacked_symbols.add(symbol)

                    elif isinstance(val, torch.Tensor):
                        for i, sym in enumerate(val.shape):
                            cbs, msgs = add_assertions(sym)
                            for cb, msg in zip(cbs, msgs):

                                def sym_size_cb(node, assert_msg, dim):
                                    with node.graph.inserting_after(node):
                                        dim_node = module.graph.create_size_node(  # type: ignore[union-attr]
                                            node, dim
                                        )
                                    cb(node=dim_node, assert_msg=assert_msg)

                                call_backs.append(partial(sym_size_cb, dim=i))
                                messages.append(f".shape[{i}]" + msg)
                    return call_backs, messages

                callbacks, messages = add_assertions(val)
                for cb, msg in zip(callbacks, messages):
                    cb(node=node, assert_msg=f"{node}" + msg)

            module.recompile()

        # Sometimes this pass would return a wrong graph where we have mismatched
        # node names in signature. Before we fix it, let's just skip it.
        if (
            self.counter == 0
            and type(self) is _AddRuntimeAssertionsForInlineConstraintsPass
        ):
            return PassResult(graph_module, False)

        # Populate the stack trace with dummy vals to respect IR
        for node in graph_module.graph.nodes:
            if not node.meta.get("stack_trace", None) and node.op not in [
                "placeholder",
                "output",
            ]:
                node.meta["stack_trace"] = "".join(traceback.format_stack(limit=1))
        return PassResult(graph_module, True)


def _get_existing_inline_assertions(
    graph_module: torch.fx.GraphModule,
    range_constraints: dict[sympy.Symbol, ValueRanges],
) -> dict[sympy.Symbol, ValueRanges]:
    existing_inline_assertions: dict[sympy.Symbol, ValueRanges] = {}

    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        # Find all the existing inline assertions. They will look something like:
        # %_local_scalar_dense = call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%arg1_1,), kwargs = {})
        # %ge = call_function[target=operator.ge](args = (%_local_scalar_dense, 0), kwargs = {})
        # %_assert_scalar = call_function[target=torch.ops.aten._assert_scalar.default](args = (%scalar_tensor, "..."), kwargs = {})
        for node in module.graph.nodes:
            if node.target != torch.ops.aten._assert_scalar.default:
                continue

            compare_arg = node.args[0]
            if not (
                isinstance(compare_arg, torch.fx.Node)
                and compare_arg.op == "call_function"
                and compare_arg.target in (operator.le, operator.ge)
                and len(compare_arg.args) == 2
            ):
                continue

            compare_op = compare_arg.target
            lhs, rhs = compare_arg.args

            def maybe_get_symint(x):
                if (
                    isinstance(x, torch.fx.Node)
                    and "val" in x.meta
                    and isinstance(x.meta["val"], torch.SymInt)
                ):
                    return x.meta["val"].node.expr
                return x

            lhs = maybe_get_symint(lhs)
            rhs = maybe_get_symint(rhs)

            if compare_op is operator.ge:
                lhs, rhs = rhs, lhs

            if isinstance(lhs, sympy.Symbol) and isinstance(rhs, int):
                symint = lhs
                scalar = rhs
            elif isinstance(rhs, sympy.Symbol) and isinstance(lhs, int):
                symint = rhs
                scalar = lhs
            else:
                continue

            if symint not in range_constraints:
                raise RuntimeError(
                    f"Unable to find symint {symint} in {range_constraints}"
                )

            previous_range = existing_inline_assertions.get(
                symint, ValueRanges(-math.inf, math.inf)
            )

            if symint is lhs:
                bounds = ValueRanges(-math.inf, scalar)
            else:
                bounds = ValueRanges(scalar, math.inf)
            existing_inline_assertions[symint] = previous_range & bounds

    return existing_inline_assertions
<|endoftext|>from typing import Any


NodeMetadataValue = Any


PROTECTED_KEYS: set[str] = {
    "val",
    "stack_trace",
    "nn_module_stack",
    "debug_handle",
    "tensor_meta",
}


class NodeMetadata:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data: dict[str, Any] = data.copy()

    def __getitem__(self, key: str) -> NodeMetadataValue:
        return self.data[key]

    def __setitem__(self, key: str, value: NodeMetadataValue) -> NodeMetadataValue:
        if key in PROTECTED_KEYS:
            raise RuntimeError(f"Could not override node key: {key}")
        self.data[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def copy(self) -> "NodeMetadata":
        return NodeMetadata(self.data.copy())
<|endoftext|># pyre-strict
from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

import torch


_T = TypeVar("_T")


class ProxyValue(Generic[_T]):
    # pyre-ignore
    def __init__(self, data: Iterable[_T], proxy: torch.fx.Proxy | torch.fx.Node):
        # pyre-ignore
        self.data = data
        self.proxy_or_node = proxy

    @property
    def node(self) -> torch.fx.Node:
        if isinstance(self.proxy_or_node, torch.fx.Node):
            return self.proxy_or_node
        if not isinstance(self.proxy_or_node, torch.fx.Proxy):
            raise AssertionError(
                f"expected Node or Proxy, got {type(self.proxy_or_node)}"
            )
        return self.proxy_or_node.node

    @property
    def proxy(self) -> torch.fx.Proxy:
        if not isinstance(self.proxy_or_node, torch.fx.Proxy):
            raise RuntimeError(
                f"ProxyValue doesn't have attached Proxy object. Node: {self.proxy_or_node.format_node()}"
            )
        return self.proxy_or_node

    def to_tensor(self) -> torch.Tensor:
        if not isinstance(self.data, torch.Tensor):
            raise AssertionError(f"expected Tensor, got {type(self.data)}")
        return self.data

    def is_tensor(self) -> bool:
        return isinstance(self.data, torch.Tensor)

    # pyre-ignore
    def __iter__(self) -> Iterator[_T]:
        yield from self.data

    def __bool__(self) -> bool:
        return bool(self.data)
<|endoftext|># mypy: allow-untyped-defs
import base64
import copy
import copyreg
import dataclasses
import heapq
import inspect
import io
import json
import keyword
import logging
import math
import operator
import re
import traceback
import types
import typing
from collections import namedtuple, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any, cast, final, Optional

import sympy

import torch
import torch.export.exported_program as ep
from torch._export.non_strict_utils import _enable_graph_inputs_of_type_nn_module
from torch._export.verifier import load_verifier
from torch._library.opaque_object import get_opaque_type_name, is_custom_class_obj
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx._symbolic_trace import _ConstantAttributeType
from torch.fx.experimental import symbolic_shapes
from torch.fx.traceback import NodeSource
from torch.utils import _pytree as pytree
from torch.utils._pytree import treespec_dumps, treespec_loads
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.symbol import prefix_str, SymT
from torch.utils._sympy.value_ranges import ValueRanges
from torch.utils._traceback import CapturedTraceback
from torch.utils._triton import has_triton

from ..utils import remove_proxy_from_state_dict
from . import schema
from .schema import (  # type: ignore[attr-defined]
    Argument,
    ArgumentKind,
    BufferMutationSpec,
    ComplexValue,
    ConstantValue,
    CustomObjArgument,
    Device,
    ExportedProgram,
    GradientToParameterSpec,
    GradientToUserInputSpec,
    Graph,
    GraphArgument,
    GraphModule,
    GraphSignature,
    InputSpec,
    InputToBufferSpec,
    InputToConstantInputSpec,
    InputToCustomObjSpec,
    InputTokenSpec,
    InputToParameterSpec,
    InputToTensorConstantSpec,
    Layout,
    LossOutputSpec,
    MemoryFormat,
    ModuleCallEntry,
    ModuleCallSignature,
    NamedArgument,
    NamedTupleDef,
    Node,
    OptionalTensorArgument,
    OutputSpec,
    OutputTokenSpec,
    ParameterMutationSpec,
    RangeConstraint,
    ScalarType,
    SCHEMA_VERSION,
    SchemaVersion,
    SymBool,
    SymBoolArgument,
    SymExpr,
    SymExprHint,
    SymFloat,
    SymFloatArgument,
    SymInt,
    SymIntArgument,
    TensorArgument,
    TensorMeta,
    TokenArgument,
    TREESPEC_VERSION,
    UserInputMutationSpec,
    UserInputSpec,
    UserOutputSpec,
)
from .union import _Union


__all__ = [
    "serialize",
    "GraphModuleSerializer",
    "ExportedProgramSerializer",
    "GraphModuleDeserializer",
    "ExportedProgramDeserializer",
]

log = logging.getLogger(__name__)


class SerializeError(RuntimeError):
    pass


def _reverse_map(d: dict[Any, Enum]):
    return {v.value: k for k, v in d.items()}


MetaType = (
    FakeTensor
    | int
    | torch.SymInt
    | float
    | torch.SymFloat
    | bool
    | torch.SymBool
    | ep.CustomObjArgument
)

DEFAULT_PICKLE_PROTOCOL = 2

ST_DELIMITER = ";"

_TORCH_TO_SERIALIZE_DTYPE = {
    torch.uint8: ScalarType.BYTE,
    torch.int8: ScalarType.CHAR,
    torch.uint16: ScalarType.UINT16,
    torch.int16: ScalarType.SHORT,
    torch.int32: ScalarType.INT,
    torch.int64: ScalarType.LONG,
    torch.float16: ScalarType.HALF,
    torch.float32: ScalarType.FLOAT,
    torch.float64: ScalarType.DOUBLE,
    torch.complex32: ScalarType.COMPLEXHALF,
    torch.complex64: ScalarType.COMPLEXFLOAT,
    torch.complex128: ScalarType.COMPLEXDOUBLE,
    torch.bool: ScalarType.BOOL,
    torch.bfloat16: ScalarType.BFLOAT16,
    torch.float8_e4m3fn: ScalarType.FLOAT8E4M3FN,
    torch.float8_e5m2: ScalarType.FLOAT8E5M2,
    torch.float8_e4m3fnuz: ScalarType.FLOAT8E4M3FNUZ,
    torch.float8_e5m2fnuz: ScalarType.FLOAT8E5M2FNUZ,
    torch.float8_e8m0fnu: ScalarType.FLOAT8E8M0FNU,
    torch.uint32: ScalarType.UINT32,
    torch.uint64: ScalarType.UINT64,
}


_SERIALIZE_TO_TORCH_DTYPE = _reverse_map(_TORCH_TO_SERIALIZE_DTYPE)  # type: ignore[arg-type]


_TORCH_TO_SERIALIZE_LAYOUT = {
    torch.sparse_coo: Layout.SparseCoo,
    torch.sparse_csr: Layout.SparseCsr,
    torch.sparse_csc: Layout.SparseCsc,
    torch.sparse_bsr: Layout.SparseBsr,
    torch.sparse_bsc: Layout.SparseBsc,
    torch._mkldnn: Layout._mkldnn,  # type: ignore[attr-defined]
    torch.strided: Layout.Strided,
}


_SERIALIZE_TO_TORCH_LAYOUT = _reverse_map(_TORCH_TO_SERIALIZE_LAYOUT)  # type: ignore[arg-type]


_TORCH_TO_SERIALIZE_MEMORY_FORMAT = {
    torch.contiguous_format: MemoryFormat.ContiguousFormat,
    torch.channels_last: MemoryFormat.ChannelsLast,
    torch.channels_last_3d: MemoryFormat.ChannelsLast3d,
    torch.preserve_format: MemoryFormat.PreserveFormat,
}


_SERIALIZE_TO_TORCH_MEMORY_FORMAT = _reverse_map(_TORCH_TO_SERIALIZE_MEMORY_FORMAT)  # type: ignore[arg-type]

_SYM_OPS = {
    operator.eq,
    operator.ne,
    operator.le,
    operator.ge,
    operator.lt,
    operator.gt,
    operator.neg,
    operator.pos,
    math.trunc,
    torch.sym_not,
    operator.mul,
    operator.add,
    operator.sub,
    operator.floordiv,
    operator.mod,
    operator.pow,
    torch.sym_int,
    torch.sym_float,
    torch.sym_ite,
    torch.sym_max,
    torch.sym_min,
    torch.sym_sqrt,
    operator.truediv,
    operator.and_,
    operator.or_,
    operator.lshift,
    operator.rshift,
}


if any(isinstance(op, torch._ops.OpOverload) for op in _SYM_OPS):
    raise AssertionError("_SYM_OPS should not contain OpOverload instances")


@dataclass
class SerializedArtifact:
    exported_program: bytes
    state_dict: bytes
    constants: bytes
    example_inputs: bytes


@dataclass
class _SerializedProgram:
    exported_program: ExportedProgram
    state_dict: bytes
    constants: bytes
    example_inputs: bytes


class LazyMap(dict):
    """
    Dictionary class for deferred instantiation of node metadata values.
    Purpose is to avoid creation of symbolic-shape tensors before relevant shape guards are parsed.
    """

    def __init__(self):
        self.map = {}
        self.evaluated = set()

    def __setitem__(self, k, v):
        self.map[k] = v

    def __getitem__(self, k):
        out = self.map[k]
        if k in self.evaluated:
            return out
        self.evaluated.add(k)
        self.map[k] = out()
        return self.map[k]

    def __repr__(self):
        return self.map.__repr__()


def deserialize_device(d: Device) -> torch.device:
    if d.index is None:
        return torch.device(type=d.type)  # type: ignore[call-overload]
    return torch.device(type=d.type, index=d.index)


def deserialize_size(sizes: Sequence[SymInt]) -> tuple[int, ...]:
    for sym_int_size in sizes:
        if sym_int_size.type != "as_int":
            raise AssertionError(f"Only as_int is supported, got {sym_int_size.type}")
    return tuple(sym_int_size.as_int for sym_int_size in sizes)


def deserialize_stride(strides: Sequence[SymInt]) -> tuple[int, ...]:
    for sym_int_stride in strides:
        if sym_int_stride.type != "as_int":
            raise AssertionError(f"Only as_int is supported, got {sym_int_stride.type}")
    return tuple(sym_int_stride.as_int for sym_int_stride in strides)


def deserialize_scalar_type(st: ScalarType) -> torch.dtype:
    return _SERIALIZE_TO_TORCH_DTYPE[st]


def deserialize_storage_offset(offset: SymInt) -> int:
    if offset.type != "as_int":
        raise AssertionError(f"Only as_int is supported, got {offset.type}")
    return offset.as_int


def _print_sympy(s: torch.SymInt | torch.SymBool | torch.SymFloat | sympy.Expr):
    if isinstance(s, (torch.SymInt, torch.SymBool, torch.SymFloat)):
        s = s.node.expr
    return sympy.printing.repr.srepr(s)


def serialize_sym_int(s: int | torch.SymInt) -> SymInt:
    if isinstance(s, (torch.SymInt, sympy.Symbol, int)):
        if symbolic_shapes.is_concrete_int(s):
            return SymInt.create(as_int=int(s))
        else:
            if not isinstance(s, (torch.SymInt, sympy.Symbol)):
                raise AssertionError(
                    f"expected SymInt or Symbol, got {type(s).__name__}"
                )
            if s.node.hint is None:
                return SymInt.create(as_expr=SymExpr(_print_sympy(s)))
            else:
                return SymInt.create(
                    as_expr=SymExpr(
                        _print_sympy(s),
                        hint=SymExprHint.create(as_int=s.node.hint),
                    )
                )
    else:
        raise SerializeError(
            f"SymInt should be either symbol or int, got `{s}` of type `{type(s)}`"
        )


def serialize_sym_float(s: float | torch.SymFloat) -> SymFloat:
    if isinstance(s, (torch.SymFloat, sympy.Symbol, float)):
        if symbolic_shapes.is_concrete_float(s):
            return SymFloat.create(as_float=float(s))
        else:
            if not isinstance(s, (torch.SymFloat, sympy.Symbol)):
                raise AssertionError(
                    f"expected SymFloat or Symbol, got {type(s).__name__}"
                )
            if s.node.hint is None:
                return SymFloat.create(as_expr=SymExpr(_print_sympy(s)))
            else:
                return SymFloat.create(
                    as_expr=SymExpr(
                        _print_sympy(s),
                        hint=SymExprHint.create(as_float=s.node.hint),
                    )
                )
    else:
        raise SerializeError(
            f"SymFloat should be either symbol or float, got `{s}` of type `{type(s)}`"
        )


def serialize_sym_bool(s: bool | torch.SymBool) -> SymBool:
    if isinstance(s, (torch.SymBool, bool)):
        if symbolic_shapes.is_concrete_bool(s):
            return SymBool.create(as_bool=bool(s))
        else:
            return SymBool.create(as_expr=SymExpr(expr_str=_print_sympy(s)))
    else:
        raise SerializeError(
            f"SymBool should be either symbol or bool, got `{s}` of type `{type(s)}`"
        )


def serialize_tensor_meta(t: torch.Tensor) -> TensorMeta:
    """
    Extract a TensorMeta describing `t`.
    """
    return TensorMeta(
        dtype=_TORCH_TO_SERIALIZE_DTYPE[t.dtype],
        sizes=[serialize_sym_int(s) for s in t.shape],
        requires_grad=t.requires_grad,
        device=Device(type=t.device.type, index=t.device.index),
        strides=[serialize_sym_int(s) for s in t.stride()],
        storage_offset=serialize_sym_int(t.storage_offset()),
        layout=_TORCH_TO_SERIALIZE_LAYOUT[t.layout],
    )


_CURRENT_DESERIALIZER: Optional["GraphModuleDeserializer"] = None


def _reduce_fake_tensor(fake_tensor: FakeTensor):
    is_parameter = isinstance(fake_tensor, torch.nn.Parameter)
    tensor_meta = serialize_tensor_meta(fake_tensor)
    tensor_meta_bytes = json.dumps(
        _dataclass_to_dict(tensor_meta), cls=EnumEncoder
    ).encode("utf-8")
    return _reconstruct_fake_tensor, (tensor_meta_bytes, is_parameter)


def _reconstruct_fake_tensor(
    serialized_tensor_meta: bytes, is_parameter: bool
) -> FakeTensor:
    # Deserialize the bytes into a TensorMeta
    json_tensor_meta = json.loads(serialized_tensor_meta.decode("utf-8"))
    tensor_meta = _dict_to_dataclass(TensorMeta, json_tensor_meta)
    # Find the current fake mode
    if _CURRENT_DESERIALIZER is None:
        raise AssertionError("Need access to current deserializer state")
    fake_tensor = _CURRENT_DESERIALIZER.deserialize_tensor_meta(tensor_meta)
    if is_parameter:
        fake_tensor = torch.nn.Parameter(fake_tensor)  # type: ignore[assignment]
    # pyrefly: ignore [bad-return]
    return fake_tensor


def serialize_torch_artifact(
    artifact: Any | None, pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL
) -> bytes:
    if artifact is None:
        return b""

    if FakeTensor in copyreg.dispatch_table:
        raise AssertionError("Refusing to stomp on existing FakeTensor reducer")
    try:
        copyreg.pickle(FakeTensor, _reduce_fake_tensor)
        buffer = io.BytesIO()
        # This is a workaround for backend's tensor deserialization problem:
        # unpickleTensor() always create a tensor on the device where it was originally saved
        # This behavior is bad for multi-gpu training, as we wish to directly load the tensor
        # on the designated device.
        # For now, we simply move the tensor to cpu before saving.
        # TODO: this should be fixed by deserialization instead.
        torch.save(artifact, buffer, pickle_protocol=pickle_protocol)
        return buffer.getvalue()
    finally:
        del copyreg.dispatch_table[FakeTensor]


def deserialize_torch_artifact(
    serialized: dict[str, Any] | tuple[Any, ...] | bytes,
):
    if isinstance(serialized, (dict, tuple)):
        return serialized
    if len(serialized) == 0:
        return {}
    buffer = io.BytesIO(serialized)
    buffer.seek(0)
    # weights_only=False as we want to load custom objects here (e.g. ScriptObject)
    try:
        artifact = torch.load(buffer, weights_only=True)
    except Exception as e:
        buffer.seek(0)
        artifact = torch.load(buffer, weights_only=False)
        log.warning(
            "Fallback to weights_only=False succeeded. "
            "Loaded object of type %s after initial failure: %s",
            type(artifact),
            exc_info=e,
        )
    if not isinstance(artifact, (tuple, dict)):
        raise AssertionError(f"expected tuple or dict, got {type(artifact).__name__}")
    return artifact


def _sympy_int_to_int(val: sympy.Expr, adjust: str) -> int | None:
    # Convert simple sympy Integers into concrete int
    if val in (sympy.oo, int_oo):
        return None
    if val in (-sympy.oo, -int_oo):
        return None
    if isinstance(val, sympy.Integer):
        return int(val)

    # TODO: Remove this adjustment when Ed gets rid of fractional ranges
    log.warning(
        "Export constraints cannot be non-integer expressions. Found "
        "type %s, and value %s. We will attempt to %s "
        "this value.",
        type(val),
        val,
        adjust,
    )

    if adjust == "floor":
        return math.floor(val)
    elif adjust == "ceil":
        return math.ceil(val)
    else:
        raise RuntimeError(f"Got invalid adjustment {adjust}")


def _int_to_sympy_int(val: int | None, default) -> sympy.Expr:
    # Convert concrete int into simple sympy Integers
    if val is None:
        return default
    if val in [-int_oo, int_oo]:
        return val
    if val == math.inf:
        return int_oo
    if val == -math.inf:
        return -int_oo
    return sympy.Integer(val)


def _symbol_index(sym: sympy.Symbol, sym_type: SymT):
    return int(str(sym)[len(prefix_str[sym_type]) :])


def serialize_range_constraints(
    range_constraints: dict[sympy.Symbol, ValueRanges],
) -> dict[str, RangeConstraint]:
    return {
        str(k): RangeConstraint(
            _sympy_int_to_int(v.lower, "ceil"),  # type: ignore[arg-type]
            _sympy_int_to_int(v.upper, "floor"),  # type: ignore[arg-type]
        )
        for k, v in range_constraints.items()
    }


def _get_schema_from_target(target):
    if isinstance(target, torch._ops.OpOverload):
        return target._schema
    elif type(target) in _serialization_registry:
        return _serialization_registry[type(target)].op_schema(target)
    raise RuntimeError(f"Cannot find schema for {type(target)}")


@dataclass
class GraphState:
    inputs: list[Argument] = field(default_factory=list)
    outputs: list[Argument] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    tensor_values: dict[str, TensorMeta] = field(default_factory=dict)
    sym_int_values: dict[str, SymInt] = field(default_factory=dict)
    sym_bool_values: dict[str, SymBool] = field(default_factory=dict)
    sym_float_values: dict[str, SymFloat] = field(default_factory=dict)
    is_single_tensor_return: bool = False
    custom_obj_values: dict[str, CustomObjArgument] = field(default_factory=dict)


class Final(type):
    def __new__(metacls, name, bases, classdict):
        for b in bases:
            if isinstance(b, Final):
                raise TypeError(f"type '{b.__name__}' is not an acceptable base type")
        return type.__new__(metacls, name, bases, dict(classdict))


def is_metadata_matched(config, entry_metadata):
    metadata_attrs = ["num_cpu_threads", "num_warps", "num_stages", "num_ctas"]
    for attr in metadata_attrs:
        if hasattr(config, attr) and hasattr(entry_metadata, attr):
            if getattr(config, attr) != getattr(entry_metadata, attr):
                return False
    return True


def get_triton_kernel_and_cache_entry(node: torch.fx.Node):
    if (
        node.target
        is not torch._higher_order_ops.triton_kernel_wrap.triton_kernel_wrapper_functional
    ):
        raise AssertionError(
            f"expected triton_kernel_wrapper_functional, got {node.target}"
        )

    if not has_triton():
        raise AssertionError("triton required to serialize triton kernels")
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    if not isinstance(node.kwargs["kernel_idx"], int):
        raise AssertionError(
            f"expected kernel_idx to be int, got {type(node.kwargs['kernel_idx'])}"
        )
    kernel = torch._higher_order_ops.triton_kernel_wrap.kernel_side_table.get_kernel(
        node.kwargs["kernel_idx"]
    )

    # For Autotuner, we need to look at the underlying JITFunction's cache
    # since the Autotuner itself doesn't have a cache
    is_autotuner = isinstance(kernel, Autotuner)
    # pyrefly: ignore [missing-attribute]
    actual_kernel = kernel.fn if is_autotuner else kernel

    if hasattr(actual_kernel, "device_caches"):
        caches = actual_kernel.device_caches
        if len(caches.keys()) != 1:
            raise AssertionError(
                f"expected exactly 1 device cache, got {len(caches.keys())}"
            )
        cache = next(iter(caches.values()))[0]
    elif hasattr(actual_kernel, "cache"):
        # old path, still used for cpu triton builds
        caches = actual_kernel.cache
        if len(caches.keys()) != 1:
            raise AssertionError(
                f"expected exactly 1 cache key, got {len(caches.keys())}"
            )
        cache = next(iter(caches.values()))
    else:
        raise AssertionError(
            # pyrefly: ignore [missing-attribute]
            f"kernel caches not found for kernel {actual_kernel.__name__}"
        )

    if len(cache.keys()) == 1:
        return actual_kernel, next(iter(cache.values()))

    has_constexprs = (
        isinstance(actual_kernel, JITFunction)
        and hasattr(actual_kernel, "constexprs")
        and len(actual_kernel.constexprs) > 0
    )

    if has_constexprs:
        constexpr_vals = {}
        # pyrefly: ignore [missing-attribute]
        for constexpr_idx in actual_kernel.constexprs:
            # pyrefly: ignore [missing-attribute]
            if constexpr_idx < len(actual_kernel.arg_names):
                # pyrefly: ignore [missing-attribute]
                param_name = actual_kernel.arg_names[constexpr_idx]
                kwargs_dict = node.kwargs.get("kwargs", {})
                if isinstance(kwargs_dict, dict):
                    if param_name in kwargs_dict:
                        constexpr_vals[param_name] = kwargs_dict[param_name]

        expected_values = [
            # pyrefly: ignore [missing-attribute]
            constexpr_vals[actual_kernel.arg_names[idx]]
            # pyrefly: ignore [missing-attribute]
            for idx in actual_kernel.constexprs
            # pyrefly: ignore [missing-attribute]
            if actual_kernel.arg_names[idx] in constexpr_vals
        ]

        # Normalize expected values for comparison with parsed constexpr values.
        # The kernel signature key stores constexprs as strings (e.g., "True", "1.5", "42"),
        # which we parse back to Python types. To ensure proper comparison, we normalize
        # the expected values: booleans, ints, and floats are kept as-is since they can
        # be compared directly with parsed values. Other types (like dtype or string
        # constants) are converted to strings to match the parsed format.
        normalized_expected = []
        for val in expected_values:
            if isinstance(val, (bool, int, float)):
                normalized_expected.append(val)
            else:
                # pyrefly: ignore [bad-argument-type]
                normalized_expected.append(str(val))

        matching_entries = []
        for sig_key, cache_entry in cache.items():
            constexpr_matches = re.findall(r"\('constexpr',\s*([^)]+)\)", sig_key)
            if constexpr_matches:
                # Parse constexpr string values back to Python types for comparison.
                # Booleans are stored as "True"/"False" strings, numbers as their string
                # representation. Values that can't be parsed as numbers are kept as strings
                # (e.g., dtype names like "torch.float32").
                constexpr_values = []
                for match in constexpr_matches:
                    if match in ("True", "False"):
                        constexpr_values.append(match == "True")
                    else:
                        try:
                            constexpr_values.append(float(match))
                        except ValueError:
                            try:
                                constexpr_values.append(int(match))
                            except ValueError:
                                constexpr_values.append(match)

                if constexpr_values == normalized_expected:
                    matching_entries.append((sig_key, cache_entry))
    else:
        matching_entries = list(cache.items())

    if len(matching_entries) == 0:
        raise AssertionError(
            # pyrefly: ignore [missing-attribute]
            f"couldn't find a kernel cache entry with metadata matching the autotuner configs for kernel {actual_kernel.__name__}. "
            f"Available cache keys: {list(cache.keys())}"
        )

    if len(matching_entries) == 1:
        return actual_kernel, matching_entries[0][1]

    if is_autotuner:
        for _sig_key, cache_entry in matching_entries:
            entry_metadata = cache_entry.metadata
            # pyrefly: ignore [missing-attribute]
            for config in kernel.configs:
                if is_metadata_matched(config, entry_metadata):
                    return actual_kernel, cache_entry

        raise AssertionError(
            # pyrefly: ignore [missing-attribute]
            f"Multiple cache entries found for autotuned kernel {actual_kernel.__name__} "
            f"{'with same constexpr values' if has_constexprs else 'with no constexpr'} "
            f"and couldn't disambiguate using configs. "
        )

    raise AssertionError(
        # pyrefly: ignore [missing-attribute]
        f"Multiple cache entries found for non-autotuned kernel {actual_kernel.__name__} "
        f"{'with same constexpr values' if has_constexprs else 'with no constexpr'}. "
        f"This should not happen. Available cache keys: {[key for key, _ in matching_entries]}"
    )


@final
class GraphModuleSerializer(metaclass=Final):
    def __init__(
        self,
        graph_signature: ep.ExportGraphSignature,
        module_call_graph: list[ep.ModuleCallEntry],
    ):
        self.graph_state = GraphState()
        self.graph_signature = graph_signature
        self.module_call_graph = module_call_graph
        self.custom_objs: dict[str, Any] = {}
        self.duplicate_getitem_nodes: dict[str, str] = {}
        self.treespec_namedtuple_fields: dict[str, NamedTupleDef] = {}

    @contextmanager
    def save_graph_state(self):
        saved = self.graph_state
        self.graph_state = GraphState()
        try:
            yield
        finally:
            self.graph_state = saved

    def handle_placeholder(self, node: torch.fx.Node):
        if node.op != "placeholder":
            raise AssertionError(f"expected placeholder op, got {node.op}")
        val = node.meta["val"]
        log.debug("[handle_placeholder] %s: %s", node.name, val)
        if isinstance(val, torch.Tensor):
            graph_input = Argument.create(
                as_tensor=self.serialize_tensor_output(node.name, val)
            )
        elif isinstance(val, torch.SymInt):
            graph_input = Argument.create(
                as_sym_int=self.serialize_sym_int_output(node.name, val)
            )
        elif isinstance(val, torch.SymFloat):
            raise AssertionError("SymFloat graph input is not implemented yet.")
        elif isinstance(val, (int, bool, str, float, type(None))):
            graph_input = self.serialize_input(val)
        elif isinstance(val, ep.CustomObjArgument):
            class_fqn = val.class_fqn
            graph_input = Argument.create(
                as_custom_obj=CustomObjArgument(name=node.name, class_fqn=class_fqn)
            )
            self.graph_state.custom_obj_values[node.name] = (
                self.serialize_script_obj_meta(val)
            )
        else:
            raise AssertionError(f"Unimplemented graph input type: {node.meta['val']}")
        self.graph_state.inputs.append(graph_input)

    def handle_output(self, node: torch.fx.Node):
        if node.op != "output":
            raise AssertionError(f"expected output op, got {node.op}")
        if len(node.args) != 1:
            raise AssertionError(
                f"FX.Node's args should have one arg, got {len(node.args)}"
            )
        node_args = node.args[0]
        log.debug("[handle_output] %s: %s", node.name, node_args)
        if isinstance(node_args, torch.fx.Node):
            # For singleton tensor returns
            self.graph_state.is_single_tensor_return = True
            self.graph_state.outputs = [self.serialize_input(node_args)]
        else:
            if not isinstance(node_args, (tuple, list)):
                raise AssertionError(f"expected tuple or list, got {type(node_args)}")
            self.graph_state.outputs = [self.serialize_input(arg) for arg in node_args]

    def serialize_operator(self, target) -> str:
        if isinstance(target, str):
            return target
        elif target.__module__.startswith("torch._ops"):
            # TODO(zhxchen17) Maybe provide a function name helper in FX.
            # From torch.fx.node._get_qualified_name
            module = target.__module__.replace("torch._ops", "torch.ops")
            return f"{module}.{target.__name__}"
        else:  # TODO(zhxchen17) Don't catch all here.
            return f"{target.__module__}.{target.__name__}"

    def handle_call_function(self, node: torch.fx.Node):
        if node.op != "call_function":
            raise AssertionError(f"expected call_function op, got {node.op}")
        meta_val = node.meta.get("val")
        log.debug(
            "[handle_call_function] %s: %s(%s, {%s}) -> %s",
            node.name,
            node.target,
            node.args,
            node.kwargs,
            meta_val,
        )

        # getitem has been handled in the producer node, skip it here
        if node.target is operator.getitem:
            return

        if node.target in _SYM_OPS or (
            meta_val is not None
            and isinstance(meta_val, (torch.SymInt, torch.SymBool, torch.SymFloat))
        ):
            if len(node.kwargs) != 0:
                raise AssertionError(
                    f"expected no kwargs for sym op, got {len(node.kwargs)}"
                )
            ex_node = Node(
                name=node.name,
                target=self.serialize_operator(node.target),
                inputs=self.serialize_sym_op_inputs(node.target, node.args),
                outputs=[self.serialize_output(node.name, meta_val)],
                metadata=self.serialize_metadata(node),
            )
        elif isinstance(node.target, torch._ops.OpOverload):
            ex_node = Node(
                name=node.name,
                target=self.serialize_operator(node.target),
                inputs=self.serialize_inputs(node.target, node.args, node.kwargs),
                outputs=self.serialize_outputs(node),
                # TODO: create a new tensor_values here, meta might have faketensor info
                metadata=self.serialize_metadata(node),
            )
        elif isinstance(node.target, torch._ops.HigherOrderOperator):

            def _is_hop_single_tensor_return(node) -> bool:
                if not isinstance(node.target, torch._ops.HigherOrderOperator):
                    raise AssertionError(
                        f"expected HigherOrderOperator, got {type(node.target).__name__}"
                    )
                # HOP schema is not always available, so we look at node.meta["val"]
                meta_val = node.meta.get("val", None)
                return meta_val is not None and isinstance(meta_val, torch.Tensor)

            # Special handle serialization for aoti_call_delegate
            if node.target is torch._higher_order_ops.aoti_call_delegate:
                serializable_args = list(node.args)

                # AOTI lowered module is not serializable, serialize the aoti_path instead
                lowered_module_name: str = node.args[0].name  # type: ignore[assignment, no-untyped-def, union-attr]
                if not hasattr(node.graph.owning_module, lowered_module_name):
                    raise AssertionError(
                        f"owning_module does not have attribute {lowered_module_name}"
                    )
                lowered_module = getattr(node.graph.owning_module, lowered_module_name)  # type: ignore[no-untyped-def]
                serializable_args[0] = lowered_module.aoti_path

                # AOTI compiled graph module in node.args[0] is stateful, and will fail the verifier check
                # Skip serializing original_gm as a workaround
                serializable_args[1] = None

                serializable_weight_nodes = []
                if serializable_args[2] is not None and isinstance(
                    serializable_args[2], Iterable
                ):
                    for weight_node in serializable_args[2]:
                        # skip passing custom obj into the weight arg as an hack
                        # The schema of weight input is a list of Tensors.
                        # Downstream runtime is not actively consuming the weighs arg for anything meaningful.
                        if isinstance(weight_node, torch.fx.Node) and isinstance(
                            weight_node.meta.get("val", None), ep.CustomObjArgument
                        ):
                            continue
                        serializable_weight_nodes.append(weight_node)
                    serializable_args[2] = serializable_weight_nodes

                def serialize_tensor_list_output(node):
                    meta_val = node.meta.get("val", None)
                    tensor_args = []
                    for idx, meta in enumerate(meta_val):
                        name = self._output_node_name_at_index(node, idx)
                        tensor_args.append(self.serialize_tensor_output(name, meta))
                    return [Argument.create(as_tensors=tensor_args)]

                ex_node = Node(
                    name=node.name,
                    target=self.serialize_operator(node.target),
                    inputs=self.serialize_hoo_inputs(serializable_args, node.kwargs),
                    outputs=serialize_tensor_list_output(node),
                    metadata=self.serialize_metadata(node),
                    is_hop_single_tensor_return=False,
                )
            elif (
                node.target
                is torch._higher_order_ops.triton_kernel_wrap.triton_kernel_wrapper_functional
            ):
                kernel, kernel_cache_entry = get_triton_kernel_and_cache_entry(node)
                kernel_cache_metadata = kernel_cache_entry.metadata

                meta_val = node.meta["val"]
                if not isinstance(meta_val, dict):
                    raise AssertionError(
                        f"expected meta_val to be dict, got {type(meta_val).__name__}"
                    )

                output_keys = meta_val.keys()
                output_indices = []

                constexpr_keys = {p.name for p in kernel.params if p.is_constexpr}
                found_constexpr = False
                inputs_new = []
                i = 0

                if not isinstance(node.kwargs["kwargs"], dict):
                    raise AssertionError(
                        f"expected kwargs['kwargs'] to be dict, got {type(node.kwargs['kwargs'])}"
                    )
                for k, v in node.kwargs["kwargs"].items():
                    # don't serialize constexpr since they will
                    # be embedded into the binary and don't
                    # need to be passed around as attributes
                    if k in constexpr_keys:
                        found_constexpr = True
                        continue

                    if found_constexpr:
                        raise AssertionError(
                            "non-constexpr args found after constexpr arg(s)"
                        )

                    if k in output_keys:
                        output_indices.append(i)
                    inputs_new.append(
                        NamedArgument(
                            name=k,
                            arg=self.serialize_input(v),
                            kind=ArgumentKind.POSITIONAL,
                        )
                    )
                    i += 1

                if not isinstance(node.kwargs["grid"], list):
                    raise AssertionError(
                        f"expected grid to be list, got {type(node.kwargs['grid'])}"
                    )

                kernel_name_with_hash = (
                    f"{kernel.fn.__name__}_{kernel_cache_metadata.hash}"
                )
                kwargs_new = {
                    "name": kernel_name_with_hash,
                    "grid": node.kwargs["grid"][0],
                    "output_indices": output_indices,
                    "num_warps": kernel_cache_metadata.num_warps,
                }
                if hasattr(kernel_cache_metadata, "num_cpu_threads"):
                    kwargs_new["num_cpu_threads"] = (
                        kernel_cache_metadata.num_cpu_threads
                    )

                if hasattr(kernel_cache_metadata, "shared"):
                    if isinstance(kernel_cache_metadata.shared, bool):
                        kwargs_new["shared_memory_bytes"] = int(
                            kernel_cache_metadata.shared
                        )
                    else:
                        kwargs_new["shared_memory_bytes"] = kernel_cache_metadata.shared

                # MTIA-specific parameters for triton kernel compilation
                if hasattr(kernel_cache_metadata, "tile_width"):
                    kwargs_new["tile_width"] = kernel_cache_metadata.tile_width
                if hasattr(kernel_cache_metadata, "tile_height"):
                    kwargs_new["tile_height"] = kernel_cache_metadata.tile_height
                if hasattr(kernel_cache_metadata, "base_pe"):
                    kwargs_new["base_pe"] = kernel_cache_metadata.base_pe

                # Kernel parameter metadata for MTIA fatbin compilation
                kwargs_new["kernel_param_names"] = [
                    p.name for p in kernel.params if not p.is_constexpr
                ]
                # Use inferred signature types from the compiled kernel's ASTSource
                # when available. The signature is populated at runtime with actual
                # types like "i32", "*fp32" based on the values passed to the kernel
                # (see specialize_impl in jit.py). Fall back to static annotations
                # for architectures that don't rely on precise type information.
                compiled_signature = getattr(
                    getattr(kernel_cache_entry, "src", None), "signature", None
                )
                if compiled_signature is not None:
                    kwargs_new["kernel_param_types"] = [
                        str(compiled_signature.get(p.name, p.annotation))
                        for p in kernel.params
                        if not p.is_constexpr
                    ]
                else:
                    # Default behavior: use static annotations (may be empty)
                    kwargs_new["kernel_param_types"] = [
                        str(p.annotation) for p in kernel.params if not p.is_constexpr
                    ]

                ex_node = Node(
                    name=node.name,
                    target=self.serialize_operator(node.target),
                    inputs=inputs_new
                    + [
                        NamedArgument(
                            name=name,
                            arg=self.serialize_input(a),
                            kind=ArgumentKind.KEYWORD,
                        )
                        for name, a in kwargs_new.items()
                    ],
                    outputs=self.serialize_hoo_outputs(node),
                    metadata=self.serialize_metadata(node),
                    is_hop_single_tensor_return=_is_hop_single_tensor_return(node),
                )
            else:
                ex_node = Node(
                    name=node.name,
                    target=self.serialize_operator(node.target),
                    inputs=self.serialize_hoo_inputs(node.args, node.kwargs),
                    outputs=self.serialize_hoo_outputs(node),
                    metadata=self.serialize_metadata(node),
                    is_hop_single_tensor_return=_is_hop_single_tensor_return(node),
                )
        elif type(node.target) in _serialization_registry:
            # Sanity check for unhandled serialization.
            if type(node.target) not in _serialization_registry:
                raise AssertionError(
                    f"{type(node.target)} is not supported in export serialization."
                )

            handler = _serialization_registry[type(node.target)]
            namespace = handler.namespace()
            op_name = handler.to_op_name(node.target)
            if not isinstance(namespace, str) or not isinstance(op_name, str):
                raise AssertionError(
                    f"expected namespace and op_name to be str, got {type(namespace).__name__} and {type(op_name).__name__}"
                )
            if ":" in namespace or ":" in op_name:
                raise AssertionError(
                    f"namespace and op_name should not contain ':', got {namespace!r} and {op_name!r}"
                )
            ex_node = Node(
                name=node.name,
                target=f"#{namespace}:{op_name}",
                inputs=self.serialize_inputs(node.target, node.args, node.kwargs),
                outputs=self.serialize_outputs(node),
                metadata=self.serialize_metadata(node),
            )
        elif callable(node.target):
            # Handle predispatch wrapper functions (vmap, JVP, etc.) that appear
            # as plain Python function call_function nodes in pre_dispatch graphs.
            ex_node = Node(
                name=node.name,
                target=self.serialize_operator(node.target),
                inputs=self.serialize_hoo_inputs(node.args, node.kwargs),
                outputs=self.serialize_hoo_outputs(node),
                metadata=self.serialize_metadata(node),
            )
        else:
            raise SerializeError(f"Serializing {node.target} is not supported")

        self.graph_state.nodes.append(ex_node)

    def handle_get_attr(self, node):
        log.debug("[handle_get_attr] %s", node.name)

    def _output_node_at_index(self, node, index) -> torch.fx.Node | None:
        user_node = None
        for user in node.users:
            if user.target is not operator.getitem:
                raise AssertionError(f"{user} is not a getitem node")
            if index == user.args[1]:
                if user_node is None:
                    user_node = user
                else:
                    # We want to deduplicate getitem nodes that are trying to
                    # index to the same index
                    self.duplicate_getitem_nodes[user.name] = user_node.name
        return user_node

    def _output_node_name_at_index(self, node, index) -> str:
        user_node = self._output_node_at_index(node, index)
        if user_node is None:
            return f"{node.name}_unused_{index}"
        else:
            return user_node.name

    def serialize_metadata(self, node: torch.fx.Node) -> dict[str, str]:
        ret = {}

        if stack_trace := node.meta.get("stack_trace"):
            ret["stack_trace"] = stack_trace

        if nn_module_stack := node.meta.get("nn_module_stack"):

            def export_nn_module_stack(val):
                if not isinstance(val, tuple) or len(val) != 2:
                    val_len = len(val) if isinstance(val, tuple) else "N/A"
                    raise AssertionError(
                        f"expected tuple of length 2, got {type(val).__name__} of length {val_len}"
                    )
                path, ty = val

                if not isinstance(path, str):
                    raise AssertionError(
                        f"expected path to be str, got {type(path).__name__}"
                    )
                if not isinstance(ty, str):
                    raise AssertionError(
                        f"expected ty to be str, got {type(ty).__name__}"
                    )

                return path + "," + ty

            # Serialize to "key,orig_path,type_str"
            nn_module_list = [
                f"{k},{export_nn_module_stack(v)}" for k, v in nn_module_stack.items()
            ]
            ret["nn_module_stack"] = ST_DELIMITER.join(nn_module_list)

        if source_fn_st := node.meta.get("source_fn_stack"):
            source_fn_list = [
                f"{source_fn[0]},{self.serialize_operator(source_fn[1])}"
                for source_fn in source_fn_st
            ]
            ret["source_fn_stack"] = ST_DELIMITER.join(source_fn_list)

        if torch_fn := node.meta.get("torch_fn"):
            ret["torch_fn"] = ST_DELIMITER.join(list(torch_fn))

        if custom := node.meta.get("custom"):
            try:
                ret["custom"] = json.dumps(custom)
            except Exception as e:
                raise SerializeError(
                    f"Failed to serialize custom metadata for node {node.name} with error {e}"
                ) from e

        if "from_node" in node.meta:
            from_node = node.meta["from_node"]
            # Serialize from_node as JSON since it's a complex nested structure
            ret["from_node"] = json.dumps(self._serialize_from_node(from_node))

        return ret

    def _serialize_from_node(
        self, from_node: list[NodeSource] | None
    ) -> list[dict[str, Any]] | None:
        """
        Serialize from_node metadata from a list of NodeSource objects to a list of dictionaries.
        """
        if from_node is None:
            return None

        return [
            node_source.to_dict()
            for node_source in from_node
            if isinstance(node_source, NodeSource)
        ]

    def serialize_script_obj_meta(
        self, script_obj_meta: ep.CustomObjArgument
    ) -> CustomObjArgument:
        log.debug("[serialize_script_obj_meta] %s", script_obj_meta)
        return CustomObjArgument(
            name=script_obj_meta.name,
            class_fqn=script_obj_meta.class_fqn,
        )

    def serialize_sym_op_inputs(self, op, args) -> list[NamedArgument]:
        if isinstance(op, torch._ops.OpOverload):
            args_names = [arg.name for arg in op._schema.arguments]
        else:
            if op not in _SYM_OPS:
                raise AssertionError(f"op {op} is not in _SYM_OPS")
            args_names = list(inspect.signature(op).parameters.keys())
        serialized_args = []
        for args_name, arg in zip(args_names, args):
            serialized_args.append(
                NamedArgument(
                    name=args_name,
                    arg=self.serialize_input(arg),
                    kind=ArgumentKind.POSITIONAL,
                )
            )
        return serialized_args

    def serialize_inputs(
        self,
        target: Any,  # torch._ops.OpOverload and other custom operator types.
        args,
        kwargs=None,
    ) -> list[NamedArgument]:
        schema = None
        serialized_args = []

        if isinstance(target, torch._higher_order_ops.torchbind.CallTorchBind):
            obj = args[0]
            method = args[1]
            schema = target.schema(obj, method)
        else:
            if not isinstance(
                target, (torch._ops.OpOverload, *_registered_extension_types())
            ):
                raise AssertionError(
                    f"expected OpOverload or registered extension type, got {type(target).__name__}"
                )
            schema = _get_schema_from_target(target)
        if schema is None:
            raise AssertionError("schema should not be None")
        kwargs = kwargs or {}

        for i, schema_arg in enumerate(schema.arguments):
            if schema_arg.name in kwargs:
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(
                            kwargs[schema_arg.name], schema_arg.type
                        ),
                        kind=ArgumentKind.KEYWORD,
                    )
                )
            elif not schema_arg.kwarg_only and i < len(args):
                serialized_args.append(
                    NamedArgument(
                        name=schema_arg.name,
                        arg=self.serialize_input(args[i], schema_arg.type),
                        kind=ArgumentKind.POSITIONAL,
                    )
                )
            else:
                # We intentionally don't serialize the missing arguments
                # with default values
                pass

        return serialized_args

    def serialize_hoo_inputs(self, args, kwargs) -> list[NamedArgument]:
        """
        For serializing HOO inputs since HOOs do not have a schema.
        """
        inputs = [
            NamedArgument(
                name="", arg=self.serialize_input(a), kind=ArgumentKind.POSITIONAL
            )
            for a in args
        ]
        inputs.extend(
            [
                NamedArgument(
                    name=name,
                    arg=self.serialize_input(a),
                    kind=ArgumentKind.KEYWORD,
                )
                for name, a in kwargs.items()
            ]
        )
        return inputs

    def is_inductor_sym_int_arg(self, arg) -> bool:
        # This is a special branch for handling SymInt args in inductor's
        # ExternalFallbackNode.
        # For regular FX graph, SymInt arg should be a fx.Node and should be
        # verified with is_sym_int_arg()
        return type(arg) is int or isinstance(arg, torch.SymInt)

    def is_sym_int_arg(self, arg) -> bool:
        return type(arg) is int or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_int_values
        )

    def is_sym_float_arg(self, arg) -> bool:
        return isinstance(arg, float) or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_float_values
        )

    def is_sym_bool_arg(self, arg) -> bool:
        return isinstance(arg, bool) or (
            isinstance(arg, torch.fx.Node)
            and arg.name in self.graph_state.sym_bool_values
        )

    # should be torch._C.JitType but that annotation is busted
    def serialize_input(self, arg, arg_type: Any | None = None) -> Argument:
        import torch._inductor.ir as inductor_ir

        inductor_tensor_buffers = (
            inductor_ir.Buffer,
            inductor_ir.ReinterpretView,
        )

        if isinstance(arg, torch.fx.Node):
            if arg.op == "get_attr":
                if not isinstance(arg.target, str):
                    raise AssertionError(
                        f"expected arg.target to be str, got {type(arg.target).__name__}"
                    )
                attr = getattr(arg.graph.owning_module, arg.target)

                if isinstance(attr, torch.Tensor):
                    raise SerializeError(
                        "getattr nodes containing tensors should not appear in the graph"
                    )
                elif isinstance(attr, torch.fx.GraphModule):
                    with self.save_graph_state():
                        graph = self.serialize_graph(attr)
                    return Argument.create(
                        as_graph=GraphArgument(name=arg.target, graph=graph)
                    )
                elif type(attr).__name__ == "LoweredBackendModule":
                    # Special handling for executorch_call_delegate HOP
                    # Its first argument is a LoweredBackendModule, for which we
                    # serialize name and backend id of the lowered module
                    module_name = getattr(attr, "module_name", None)
                    backend_id = getattr(attr, "backend_id", None)
                    if module_name is None:
                        raise AssertionError("module_name should not be None")
                    if backend_id is None:
                        raise AssertionError("backend_id should not be None")
                    return Argument.create(as_string=f"{module_name}-{backend_id}")
                else:
                    raise SerializeError(
                        f"Unsupported getattr attribute {arg.target} with type: {type(attr)}"
                    )
            elif self.is_sym_int_arg(arg):
                return Argument.create(
                    as_sym_int=SymIntArgument.create(as_name=arg.name)
                )
            elif self.is_sym_float_arg(arg):
                return Argument.create(
                    as_sym_float=SymFloatArgument.create(as_name=arg.name)
                )
            elif self.is_sym_bool_arg(arg):
                return Argument.create(
                    as_sym_bool=SymBoolArgument.create(as_name=arg.name)
                )
            elif isinstance(arg.meta["val"], ep.CustomObjArgument):
                return Argument.create(
                    as_custom_obj=CustomObjArgument(
                        name=arg.name, class_fqn=arg.meta["val"].class_fqn
                    )
                )
            elif arg.name in self.duplicate_getitem_nodes:
                dedup_name = self.duplicate_getitem_nodes[arg.name]
                return Argument.create(as_tensor=TensorArgument(name=dedup_name))
            else:
                return Argument.create(as_tensor=TensorArgument(name=arg.name))
        elif isinstance(arg, inductor_tensor_buffers):
            # Other branches are for arguments in fx node.
            # This is a special branch for handling buffers (representing tensor arguments)
            # for inductor's ExternalFallbackNode
            # export_extern_kernel_node() is using this function to serialize arguments
            arg_name = arg.get_name()
            if arg_name is None:
                raise AssertionError("Buffer must have valid name")
            return Argument.create(as_tensor=TensorArgument(name=arg_name))
        elif isinstance(arg, inductor_ir.TorchBindObject):
            # This is a special branch for handling TorchBindObject
            # for inductor's ExternalFallbackNode
            # export_extern_kernel_node() is using this function to serialize arguments
            arg_name = arg.get_name()
            if arg_name is None:
                raise AssertionError("Buffer must have valid name")
            arg_val = arg.get_real_obj()
            class_fqn = arg_val._type().qualified_name()
            self.custom_objs[arg_name] = arg_val
            return Argument.create(as_custom_obj=CustomObjArgument(arg_name, class_fqn))
        elif isinstance(arg, torch.SymInt):
            # This is a special branch for handling SymInt args in inductor's
            # ExternalFallbackNode.
            # For regular FX graph, SymInt arg should be a fx.Node with
            # self.is_sym_int_arg(arg) being true
            return Argument.create(as_sym_int=SymIntArgument.create(as_name=str(arg)))
        elif isinstance(arg, torch.SymFloat):
            # This is a special branch for handling SymFloat args in inductor's
            # ExternalFallbackNode.
            # For regular FX graph, SymInt arg should be a fx.Node with
            # self.is_sym_float_arg(arg) being true
            return Argument.create(
                as_sym_float=SymFloatArgument.create(as_name=str(arg))
            )
        elif type(arg) is bool:
            return Argument.create(as_bool=arg)
        elif type(arg) is str:
            return Argument.create(as_string=arg)
        elif type(arg) is int:
            return Argument.create(as_int=arg)
        elif type(arg) is float:
            return Argument.create(as_float=arg)
        elif type(arg) is complex:
            return Argument.create(
                as_complex=ComplexValue(real=arg.real, imag=arg.imag)
            )
        elif arg is None:
            return Argument.create(as_none=True)
        elif isinstance(arg, dict):
            serialized_dict = {}
            for key, value in arg.items():
                if not isinstance(key, str):
                    raise SerializeError(f"Dict keys must be strings, got {type(key)}")
                serialized_dict[key] = self.serialize_input(value)
            return Argument.create(as_string_to_argument=serialized_dict)
        elif isinstance(arg, (list, tuple)):
            if len(arg) == 0:
                if arg_type is not None:
                    if isinstance(arg_type, torch.OptionalType):
                        arg_type = arg_type.getElementType()  # type: ignore[assignment]
                    if not isinstance(arg_type, torch.ListType):
                        raise AssertionError(
                            f"expected ListType, got {type(arg_type).__name__}"
                        )
                    elem_type = arg_type.getElementType()
                    if isinstance(elem_type, torch.OptionalType):
                        elem_type = elem_type.getElementType()

                    if isinstance(elem_type, torch.BoolType):
                        return Argument.create(as_bools=[])
                    elif isinstance(elem_type, torch.IntType):
                        return Argument.create(as_ints=[])
                    elif isinstance(elem_type, torch.FloatType):
                        return Argument.create(as_floats=[])
                    elif isinstance(elem_type, torch.StringType):
                        return Argument.create(as_strings=[])
                    elif isinstance(elem_type, torch.TensorType):
                        return Argument.create(as_tensors=[])
                    elif isinstance(elem_type, torch.ListType):
                        inner_elem_type = elem_type.getElementType()
                        if isinstance(inner_elem_type, torch.IntType):
                            return Argument.create(as_int_lists=[])
                        elif isinstance(inner_elem_type, torch.FloatType):
                            return Argument.create(as_float_lists=[])
                        else:
                            raise SerializeError(
                                f"Empty list with nested type {elem_type} nyi."
                            )
                    else:
                        # I believe empty symint lists default to ints, but
                        # please file an issue if this is not the case
                        raise SerializeError(f"Empty list with type {elem_type} nyi.")
                else:
                    # We could serialize this by default to a tensor list. This
                    # is needed in the HOO case
                    log.warning(
                        "Unsure how to serialize the given empty list, "
                        "as we don't know what is the type of this argument. "
                        "Serializing it as a tensor list by default."
                    )
                    return Argument.create(as_tensors=[])

            if all(type(a) is bool for a in arg):
                return Argument.create(as_bools=list(arg))
            elif all(type(a) is int for a in arg):
                return Argument.create(as_ints=list(arg))
            elif all(type(a) is float for a in arg):
                return Argument.create(as_floats=list(arg))
            elif all(type(a) is str for a in arg):
                return Argument.create(as_strings=list(arg))
            elif all(self.is_inductor_sym_int_arg(a) for a in arg):
                # This is a special branch for handling SymInt args in inductor's
                # ExternalFallbackNode.
                # For regular FX graph, SymInt arg should be a fx.Node
                values = []
                for a in arg:
                    if isinstance(a, torch.SymInt):
                        values.append(SymIntArgument.create(as_name=str(a)))
                    elif type(a) is int:
                        values.append(SymIntArgument.create(as_int=a))
                return Argument.create(as_sym_ints=values)
            elif all(isinstance(a, torch.SymFloat) for a in arg):
                return Argument.create(
                    as_sym_floats=[SymFloatArgument.create(as_name=str(a)) for a in arg]
                )
            elif all(self.is_sym_int_arg(a) for a in arg):
                # list of sym_ints
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymIntArgument.create(as_name=a.name))
                    elif type(a) is int:
                        values.append(SymIntArgument.create(as_int=a))
                return Argument.create(as_sym_ints=values)
            elif all(self.is_sym_float_arg(a) for a in arg):
                # list of sym_float
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymFloatArgument.create(as_name=a.name))
                    elif isinstance(a, float):
                        values.append(SymFloatArgument.create(as_float=a))
                return Argument.create(as_sym_floats=values)
            elif all(self.is_sym_bool_arg(a) for a in arg):
                # list of sym_bools
                values = []
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        values.append(SymBoolArgument.create(as_name=a.name))
                    elif isinstance(a, bool):
                        values.append(SymBoolArgument.create(as_bool=a))
                return Argument.create(as_sym_bools=values)
            elif all(isinstance(a, torch.fx.Node) for a in arg):
                # list of tensors
                arguments = []
                for a in arg:
                    if a.op == "get_attr":
                        raise SerializeError(
                            "getattr nodes containing tensors should not appear in the graph"
                        )
                    arguments.append(TensorArgument(name=a.name))
                return Argument.create(as_tensors=arguments)
            elif all(isinstance(a, (list, tuple)) for a in arg) and all(
                all(isinstance(t, torch.fx.Node) for t in inner) for inner in arg
            ):
                # nested list of tensors (List[List[Tensor]])
                nested_arguments = []
                for inner_list in arg:
                    inner_arguments = []
                    for node in inner_list:
                        if node.op == "get_attr":
                            raise SerializeError(
                                "getattr nodes containing tensors should not appear in the graph"
                            )
                        inner_arguments.append(TensorArgument(name=node.name))
                    nested_arguments.append(inner_arguments)
                return Argument.create(as_nested_tensors=nested_arguments)
            elif all(isinstance(a, (torch.fx.Node, type(None))) for a in arg):
                # list of optional tensors
                def serialize_optional_tensor_args(a):
                    if a is None:
                        return OptionalTensorArgument.create(as_none=True)
                    elif isinstance(a, torch.fx.Node):
                        return OptionalTensorArgument.create(
                            as_tensor=TensorArgument(name=a.name)
                        )
                    else:
                        raise SerializeError(f"Unsupported list/tuple argument: {a}")

                return Argument.create(
                    as_optional_tensors=list(map(serialize_optional_tensor_args, arg))
                )
            elif all(isinstance(a, inductor_tensor_buffers) for a in arg):
                # list of inductor buffers
                return Argument.create(
                    as_tensors=[TensorArgument(name=a.get_name()) for a in arg],
                )
            elif all(
                isinstance(a, (*inductor_tensor_buffers, type(None))) for a in arg
            ):
                # list of inductor buffers as optional tensors
                def serialize_optional_tensor_args(a):
                    if a is None:
                        return OptionalTensorArgument.create(as_none=True)
                    elif isinstance(a, inductor_tensor_buffers):
                        return OptionalTensorArgument.create(
                            as_tensor=TensorArgument(name=a.get_name())
                        )
                    else:
                        raise SerializeError(f"Unsupported list/tuple argument: {a}")

                return Argument.create(
                    as_optional_tensors=list(map(serialize_optional_tensor_args, arg))
                )
            elif all(
                isinstance(a, (list, tuple)) and all(type(x) is int for x in a)
                for a in arg
            ):
                # list of int lists (List[List[int]])
                return Argument.create(as_int_lists=[list(t) for t in arg])
            elif all(
                isinstance(a, (list, tuple)) and all(isinstance(x, float) for x in a)
                for a in arg
            ):
                # list of float lists (List[List[float]])
                return Argument.create(as_float_lists=[list(t) for t in arg])
            else:
                raise SerializeError(
                    f"Unsupported list/tuple argument type: {[type(a) for a in arg]}"
                )
        elif isinstance(arg, torch.dtype):
            return Argument.create(as_scalar_type=_TORCH_TO_SERIALIZE_DTYPE[arg])
        elif isinstance(arg, torch.device):
            return Argument.create(as_device=Device(type=arg.type, index=arg.index))
        elif isinstance(arg, torch.memory_format):
            return Argument.create(
                as_memory_format=_TORCH_TO_SERIALIZE_MEMORY_FORMAT[arg]
            )
        elif isinstance(arg, torch.layout):
            return Argument.create(as_layout=_TORCH_TO_SERIALIZE_LAYOUT[arg])
        elif isinstance(arg, torch._C.ScriptObject):
            if not (
                arg._has_method("__getstate__")  # type: ignore[attr-defined]
                and arg._has_method("__setstate__")  # type: ignore[attr-defined]
            ):
                raise SerializeError(
                    f"Unable to serialize custom class {arg}. Please define "
                    "serialization methods via def_pickle()."
                )
            # Custom objects through torchind are serializable with pickle,
            # through implementing the .def_pickle function.  This should result
            # in the object containing a __getstate__ and __setstate__
            # serialize/deserialize function.
            custom_obj_name = f"_custom_obj_{len(self.custom_objs)}"
            self.custom_objs[custom_obj_name] = arg
            class_fqn = arg._type().qualified_name()  # type: ignore[attr-defined]
            return Argument.create(
                as_custom_obj=CustomObjArgument(custom_obj_name, class_fqn)
            )
        elif is_custom_class_obj(arg):
            custom_obj_name = f"_custom_obj_{len(self.custom_objs)}"
            self.custom_objs[custom_obj_name] = arg
            class_fqn = get_opaque_type_name(type(arg))
            return Argument.create(
                as_custom_obj=CustomObjArgument(custom_obj_name, class_fqn)
            )
        elif isinstance(arg, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
            return Argument.create(as_operator=self.serialize_operator(arg))
        else:
            raise SerializeError(
                f"Unsupported argument type: {type(arg)} with schema arg_type {arg_type}"
            )

    def serialize_tensor_output(self, name, meta_val) -> TensorArgument:
        if name in self.graph_state.tensor_values:
            raise AssertionError(f"name {name!r} already in tensor_values")
        self.graph_state.tensor_values[name] = serialize_tensor_meta(meta_val)
        return TensorArgument(name=name)

    def serialize_sym_int_output(self, name, meta_val) -> SymIntArgument:
        if name in self.graph_state.sym_int_values:
            raise AssertionError(f"name {name!r} already in sym_int_values")
        self.graph_state.sym_int_values[name] = serialize_sym_int(meta_val)
        return SymIntArgument.create(as_name=name)

    def serialize_sym_float_output(self, name, meta_val) -> SymFloatArgument:
        if name in self.graph_state.sym_float_values:
            raise AssertionError(f"name {name!r} already in sym_float_values")
        self.graph_state.sym_float_values[name] = serialize_sym_float(meta_val)
        return SymFloatArgument.create(as_name=name)

    def serialize_sym_bool_output(self, name, meta_val) -> SymBoolArgument:
        if name in self.graph_state.sym_bool_values:
            raise AssertionError(f"name {name!r} already in sym_bool_values")
        self.graph_state.sym_bool_values[name] = serialize_sym_bool(meta_val)
        return SymBoolArgument.create(as_name=name)

    def serialize_input_spec(self, spec: ep.InputSpec) -> InputSpec:
        log.debug("[serialize_input_spec] %s", spec)
        if spec.kind == ep.InputKind.USER_INPUT:
            if isinstance(spec.arg, ep.ConstantArgument):
                if type(spec.arg.value) is int:
                    constant_spec = ConstantValue.create(as_int=spec.arg.value)
                elif type(spec.arg.value) is bool:
                    constant_spec = ConstantValue.create(as_bool=spec.arg.value)
                elif type(spec.arg.value) is str:
                    constant_spec = ConstantValue.create(as_string=spec.arg.value)
                elif type(spec.arg.value) is float:
                    constant_spec = ConstantValue.create(as_float=spec.arg.value)
                elif spec.arg.value is None:
                    constant_spec = ConstantValue.create(as_none=True)
                else:
                    raise SerializeError(
                        f"Unhandled constant input {spec.arg.value} to serialize"
                    )
                return InputSpec.create(
                    constant_input=InputToConstantInputSpec(
                        name=spec.arg.name, value=constant_spec
                    )
                )
            else:
                return InputSpec.create(
                    user_input=UserInputSpec(arg=self.serialize_argument_spec(spec.arg))
                )
        elif spec.kind == ep.InputKind.PARAMETER:
            if spec.target is None:
                raise AssertionError("spec.target should not be None for PARAMETER")
            if not isinstance(spec.arg, ep.TensorArgument):
                raise AssertionError(
                    f"expected TensorArgument, got {type(spec.arg).__name__}"
                )
            return InputSpec.create(
                parameter=InputToParameterSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    parameter_name=spec.target,
                )
            )
        elif spec.kind == ep.InputKind.BUFFER:
            if spec.target is None:
                raise AssertionError("spec.target should not be None for BUFFER")
            if not isinstance(spec.arg, ep.TensorArgument):
                raise AssertionError(
                    f"expected TensorArgument, got {type(spec.arg).__name__}"
                )
            if spec.persistent is None:
                raise AssertionError("spec.persistent should not be None for BUFFER")
            return InputSpec.create(
                buffer=InputToBufferSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    buffer_name=spec.target,
                    persistent=spec.persistent,
                )
            )
        elif spec.kind == ep.InputKind.CONSTANT_TENSOR:
            if spec.target is None:
                raise AssertionError(
                    "spec.target should not be None for CONSTANT_TENSOR"
                )
            if not isinstance(spec.arg, ep.TensorArgument):
                raise AssertionError(
                    f"expected TensorArgument, got {type(spec.arg).__name__}"
                )
            return InputSpec.create(
                tensor_constant=InputToTensorConstantSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    tensor_constant_name=spec.target,
                )
            )
        elif spec.kind == ep.InputKind.CUSTOM_OBJ:
            if spec.target is None:
                raise AssertionError("spec.target should not be None for CUSTOM_OBJ")
            if not isinstance(spec.arg, ep.CustomObjArgument):
                raise AssertionError(
                    f"expected CustomObjArgument, got {type(spec.arg).__name__}"
                )
            return InputSpec.create(
                custom_obj=InputToCustomObjSpec(
                    arg=CustomObjArgument(
                        name=spec.arg.name, class_fqn=spec.arg.class_fqn
                    ),
                    custom_obj_name=spec.target,
                )
            )
        elif spec.kind == ep.InputKind.TOKEN:
            if not isinstance(spec.arg, ep.TokenArgument):
                raise AssertionError(
                    f"expected TokenArgument, got {type(spec.arg).__name__}"
                )
            return InputSpec.create(
                token=InputTokenSpec(
                    arg=TokenArgument(name=spec.arg.name),
                )
            )
        else:
            raise AssertionError(f"Unknown argument kind: {spec}")

    def serialize_output_spec(self, spec: ep.OutputSpec) -> OutputSpec:
        log.debug("[serialize_output_spec] %s", spec)
        if spec.kind == ep.OutputKind.USER_OUTPUT:
            return OutputSpec.create(
                user_output=UserOutputSpec(arg=self.serialize_argument_spec(spec.arg))
            )
        elif spec.kind == ep.OutputKind.LOSS_OUTPUT:
            if not isinstance(spec.arg, ep.TensorArgument):
                raise AssertionError(
                    f"expected TensorArgument, got {type(spec.arg).__name__}"
                )
            return OutputSpec.create(
                loss_output=LossOutputSpec(arg=TensorArgument(name=spec.arg.name))
            )
        elif spec.kind == ep.OutputKind.BUFFER_MUTATION:
            if spec.target is None:
                raise AssertionError(
                    "spec.target should not be None for BUFFER_MUTATION"
                )
            if not isinstance(spec.arg, ep.TensorArgument):
                raise AssertionError(
                    f"expected TensorArgument, got {type(spec.arg).__name__}"
                )
            return OutputSpec.create(
                buffer_mutation=BufferMutationSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    buffer_name=spec.target,
                )
            )
        elif spec.kind == ep.OutputKind.PARAMETER_MUTATION:
            if spec.target is None:
                raise AssertionError(
                    "spec.target should not be None for PARAMETER_MUTATION"
                )
            if not isinstance(spec.arg, ep.TensorArgument):
                raise AssertionError(
                    f"expected TensorArgument, got {type(spec.arg).__name__}"
                )
            return OutputSpec.create(
                parameter_mutation=ParameterMutationSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    parameter_name=spec.target,
                )
            )
        elif spec.kind == ep.OutputKind.GRADIENT_TO_PARAMETER:
            if spec.target is None:
                raise AssertionError(
                    "spec.target should not be None for GRADIENT_TO_PARAMETER"
                )
            if not isinstance(spec.arg, ep.TensorArgument):
                raise AssertionError(
                    f"expected TensorArgument, got {type(spec.arg).__name__}"
                )
            return OutputSpec.create(
                gradient_to_parameter=GradientToParameterSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    parameter_name=spec.target,
                )
            )
        elif spec.kind == ep.OutputKind.GRADIENT_TO_USER_INPUT:
            if spec.target is None:
                raise AssertionError(
                    "spec.target should not be None for GRADIENT_TO_USER_INPUT"
                )
            if not isinstance(spec.arg, ep.TensorArgument):
                raise AssertionError(
                    f"expected TensorArgument, got {type(spec.arg).__name__}"
                )
            return OutputSpec.create(
                gradient_to_user_input=GradientToUserInputSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    user_input_name=spec.target,
                )
            )
        elif spec.kind == ep.OutputKind.USER_INPUT_MUTATION:
            if spec.target is None:
                raise AssertionError(
                    "spec.target should not be None for USER_INPUT_MUTATION"
                )
            if not isinstance(spec.arg, ep.TensorArgument):
                raise AssertionError(
                    f"expected TensorArgument, got {type(spec.arg).__name__}"
                )
            return OutputSpec.create(
                user_input_mutation=UserInputMutationSpec(
                    arg=TensorArgument(name=spec.arg.name),
                    user_input_name=spec.target,
                )
            )
        elif spec.kind == ep.OutputKind.TOKEN:
            if not isinstance(spec.arg, ep.TokenArgument):
                raise AssertionError(
                    f"expected TokenArgument, got {type(spec.arg).__name__}"
                )
            return OutputSpec.create(
                token=OutputTokenSpec(
                    arg=TokenArgument(name=spec.arg.name),
                )
            )
        else:
            raise AssertionError(f"Unknown argument kind: {spec}")

    def serialize_signature(self, sig: ep.ExportGraphSignature) -> GraphSignature:
        log.debug("\n[serialize_signature]")
        return GraphSignature(
            input_specs=[self.serialize_input_spec(s) for s in sig.input_specs],
            output_specs=[self.serialize_output_spec(s) for s in sig.output_specs],
        )

    def serialize_argument_spec(self, x: ep.ArgumentSpec) -> Argument:
        if isinstance(x, ep.TensorArgument):
            return Argument.create(as_tensor=TensorArgument(name=x.name))
        elif isinstance(x, ep.SymIntArgument):
            return Argument.create(as_sym_int=SymIntArgument.create(as_name=x.name))
        elif isinstance(x, ep.SymFloatArgument):
            return Argument.create(as_sym_float=SymFloatArgument.create(as_name=x.name))
        elif isinstance(x, ep.ConstantArgument):
            return self.serialize_input(x.value)
        elif isinstance(x, ep.CustomObjArgument):
            return Argument.create(
                as_custom_obj=CustomObjArgument(name=x.name, class_fqn=x.class_fqn)
            )
        else:
            raise AssertionError("TODO")

    def serialize_treespec(self, treespec: pytree.TreeSpec) -> str:
        # We want to additionally save all the field names of the namedtuples in
        # case users want to check that the treespec types are equivalent
        def store_namedtuple_fields(ts: pytree.TreeSpec) -> None:
            if ts.type is None:
                return
            if ts.type is namedtuple or pytree.is_namedtuple_class(ts.type):
                serialized_type_name = pytree.SUPPORTED_SERIALIZED_TYPES[
                    ts.context
                ].serialized_type_name
                if serialized_type_name in self.treespec_namedtuple_fields:
                    field_names = self.treespec_namedtuple_fields[
                        serialized_type_name
                    ].field_names
                    if field_names != ts.context._fields:
                        raise SerializeError(
                            f"The given TreeSpec's namedtuple type {ts.context} "
                            f"was found to have field names {ts.context._fields} "
                            f"but somehow previously was found to have field names {field_names}."
                        )
                else:
                    self.treespec_namedtuple_fields[serialized_type_name] = (
                        NamedTupleDef(field_names=ts.context._fields)
                    )

            for child in ts.children():
                store_namedtuple_fields(child)

        serialized_treespec = treespec_dumps(treespec, TREESPEC_VERSION)
        store_namedtuple_fields(treespec)
        return serialized_treespec

    def serialize_module_call_signature(
        self, module_call_signature: ep.ModuleCallSignature
    ) -> ModuleCallSignature:
        log.debug("[serialize_module_call_signature] %s", module_call_signature)
        return ModuleCallSignature(
            inputs=[
                self.serialize_argument_spec(x) for x in module_call_signature.inputs
            ],
            outputs=[
                self.serialize_argument_spec(x) for x in module_call_signature.outputs
            ],
            in_spec=self.serialize_treespec(module_call_signature.in_spec),
            out_spec=self.serialize_treespec(module_call_signature.out_spec),
            forward_arg_names=(
                names if (names := module_call_signature.forward_arg_names) else None
            ),
        )

    def serialize_module_call_graph(
        self, module_call_graph: list[ep.ModuleCallEntry]
    ) -> list[ModuleCallEntry]:
        log.debug("\n[serialize_module_call_graph]")
        return [
            ModuleCallEntry(
                fqn=entry.fqn,
                signature=(
                    self.serialize_module_call_signature(entry.signature)
                    if entry.signature
                    else None
                ),
            )
            for entry in module_call_graph
        ]

    def serialize_outputs(self, node: torch.fx.Node) -> list[Argument]:
        """For a given node, return the dataclass representing its output values.

        [NOTE: Multiple outputs] We handle aggregates differently than FX. For
        FX, it looks like:

            x = call_function("multiple_return", ...)
            element0 = call_function(getitem, x, 0)
            foo = call_function("use_output", element0)

        We do not want the intermediate `getitem` call, so our serialized thing looks like:

            element0, element1, element2 = call_function("multiple_return", ...)
            foo = call_function("use_output", element0)

        We want names to be consistent across these two schemes, so that we can
        mostly reuse the names coming from FX. This function computes a mapping from
        the FX representation to our representation, preserving the names.
        """

        def _is_single_tensor_list_return(target: Any) -> bool:
            schema = _get_schema_from_target(target)
            returns = schema.returns

            if len(returns) != 1:
                return False
            return_type = returns[0].real_type
            return isinstance(return_type, torch.ListType) and isinstance(
                return_type.getElementType(), torch.TensorType
            )

        if not (
            node.op == "call_function"
            and isinstance(
                node.target, (torch._ops.OpOverload, *_registered_extension_types())
            )
        ):
            raise AssertionError(
                f"expected call_function with OpOverload or registered extension type, "
                f"got {node.op} with {type(node.target).__name__}"
            )

        schema = _get_schema_from_target(node.target)
        returns = schema.returns

        if len(returns) == 0:
            return []

        meta_val = node.meta["val"]

        # Check single value return
        if _is_single_tensor_list_return(node.target):
            # e.g "-> Tensor[]"
            tensor_args = []
            for idx, meta in enumerate(meta_val):
                name = self._output_node_name_at_index(node, idx)
                tensor_args.append(self.serialize_tensor_output(name, meta))
            return [Argument.create(as_tensors=tensor_args)]
        elif len(returns) == 1:
            return [self.serialize_output(node.name, meta_val)]

        # There are a two possibilities at this point:
        # - This operator returns a tuple of Tensors, e.g. "-> (Tensor, Tensor)"
        # - This operator returns a tuple of mixed of Tensor and Tensors, e.g. "-> (Tensor, Tensor[])"
        #
        # Either way, start by gathering a list of TensorArguments with the correct names.
        # For consistent naming with FX, consult the downstream `getitem` node and
        # make sure our outputs have the same name.

        output_arguments = []
        for idx, (meta, return_schema) in enumerate(zip(meta_val, returns)):
            if meta is None:
                if not isinstance(
                    return_schema.real_type, (torch.OptionalType, torch.TensorType)
                ):
                    raise AssertionError(
                        f"expected OptionalType or TensorType, got {type(return_schema.real_type).__name__}"
                    )
                # When the return type is annotated as Tensor type, the op can also return an
                # undefined Tensor which will be implicitly converted to None in Python.
                output_arguments.append(Argument.create(as_none=True))
            elif isinstance(meta, torch.Tensor):
                if not isinstance(
                    return_schema.real_type, (torch.OptionalType, torch.TensorType)
                ):
                    raise AssertionError(
                        f"expected OptionalType or TensorType, got {type(return_schema.real_type).__name__}"
                    )
                name = self._output_node_name_at_index(node, idx)
                output_arguments.append(self.serialize_output(name, meta))
            elif isinstance(meta, list):
                # for List[Tensor] return type
                if not (
                    isinstance(return_schema.real_type, torch.ListType)
                    and isinstance(
                        return_schema.real_type.getElementType(), torch.TensorType
                    )
                ):
                    raise AssertionError(
                        f"expected ListType with TensorType element, got {type(return_schema.real_type).__name__}"
                    )
                user_node = self._output_node_at_index(node, idx)
                args = []
                for i, m in enumerate(meta):
                    if m is None:
                        continue
                    if user_node is None:
                        name = f"{node.name}_unused_{idx}_{i}"
                    else:
                        name = self._output_node_name_at_index(user_node, i)
                    args.append(self.serialize_tensor_output(name, m))
                output_arguments.append(Argument.create(as_tensors=args))
            elif isinstance(meta, (int, SymInt, float, SymFloat)):
                user_node_name = self._output_node_name_at_index(node, idx)
                output_arguments.append(self.serialize_output(user_node_name, meta))
            else:
                raise ValueError(
                    f"Unhandled output type {type(meta)} from node {node.format_node()}"
                )

        return output_arguments

    def serialize_hoo_outputs(self, node: torch.fx.Node) -> list[Argument]:
        """
        For serializing HOO outputs since HOOs do not have a schema.
        """
        meta_val = node.meta["val"]

        if isinstance(meta_val, tuple):
            outputs = []
            for i, element_meta_val in enumerate(meta_val):
                user_node = self._output_node_at_index(node, i)
                if isinstance(element_meta_val, list):
                    # e.g "-> Tensor[]"
                    tensors = []
                    for j, m in enumerate(element_meta_val):
                        if not isinstance(m, torch.Tensor):
                            raise SerializeError(
                                f"Serialize list output with type {type(m)} nyi"
                            )

                        if user_node is None:
                            name = f"{node.name}_unused_{i}_{j}"
                        else:
                            name = self._output_node_name_at_index(user_node, j)
                        tensors.append(self.serialize_tensor_output(name, m))
                    outputs.append(Argument.create(as_tensors=tensors))

                else:
                    name = (
                        user_node.name
                        if user_node is not None
                        else f"{node.name}_unused_{i}"
                    )

                    outputs.append(self.serialize_output(name, element_meta_val))

            return outputs
        elif isinstance(meta_val, dict):
            tensor_args = []
            # use the dict key as the idx
            for idx, meta in meta_val.items():
                if not isinstance(meta, torch.Tensor):
                    raise SerializeError(
                        f"Serialize list output with type {type(meta)} nyi"
                    )
                name = self._output_node_name_at_index(node, idx)
                tensor_args.append(self.serialize_tensor_output(name, meta))
            return [Argument.create(as_tensors=tensor_args)]
        else:
            return [self.serialize_output(node.name, meta_val)]

    def serialize_output(self, name: str, meta_val: Any) -> Argument:
        # Check single value return
        if meta_val is None:
            return Argument.create(as_none=True)
        if isinstance(meta_val, torch.Tensor):
            # e.g "-> Tensor"
            return Argument.create(
                as_tensor=self.serialize_tensor_output(name, meta_val)
            )
        elif isinstance(meta_val, (bool, torch.SymBool)):
            # e.g "-> SymBool"
            return Argument.create(
                as_sym_bool=self.serialize_sym_bool_output(name, meta_val)
            )
        elif isinstance(meta_val, (int, torch.SymInt)):
            # e.g "-> SymInt"
            if isinstance(meta_val, bool):
                raise AssertionError(
                    "meta_val should not be bool when returning SymInt"
                )
            return Argument.create(
                as_sym_int=self.serialize_sym_int_output(name, meta_val)
            )
        elif isinstance(meta_val, (float, torch.SymFloat)):
            # e.g "-> SymFloat"
            return Argument.create(
                as_sym_float=self.serialize_sym_float_output(name, meta_val)
            )

        # list outputs should've been handled earlier
        raise SerializeError(f"Unable to serialize output {meta_val}")

    def _handle_getitem_users(self, node: torch.fx.Node) -> list[TensorArgument]:
        meta_val = node.meta["val"]

        idx_to_name = {}
        for user in node.users:
            if user.target is not operator.getitem:
                raise AssertionError(f"User node {user} of {node} is incorrect")
            # pyrefly: ignore [unsupported-operation]
            idx_to_name[user.args[1]] = user.name

        for idx, _ in enumerate(meta_val):
            # FX does not emit a getitem node for any outputs that are unused.
            # However, we need a name for them so that the number of outputs will
            # correctly match the schema. Just assign a dummy name.
            if idx not in idx_to_name:
                # pyrefly: ignore [unsupported-operation]
                idx_to_name[idx] = f"{node.name}_unused_{idx}"

        arg_list = []
        for i, element_meta_val in enumerate(meta_val):
            arg_list.append(
                # pyrefly: ignore [bad-index]
                self.serialize_tensor_output(idx_to_name[i], element_meta_val)
            )

        return arg_list

    def serialize_graph(self, graph_module: torch.fx.GraphModule) -> Graph:
        if not isinstance(graph_module, torch.fx.GraphModule):
            raise AssertionError(
                f"expected GraphModule, got {type(graph_module).__name__}"
            )
        log.debug(
            "[serialize_graph]\n\n%s", graph_module.print_readable(print_output=False)
        )

        for node in graph_module.graph.nodes:
            try:
                getattr(self, f"handle_{node.op}")(node)
            except Exception as e:
                raise SerializeError(
                    f"Failed serializing node {node} in graph: {node.format_node()}\n Original exception {traceback.format_exc()}"
                ) from e

        return Graph(
            inputs=self.graph_state.inputs,
            nodes=self.graph_state.nodes,
            tensor_values=self.graph_state.tensor_values,
            sym_int_values=self.graph_state.sym_int_values,
            sym_float_values=self.graph_state.sym_float_values,
            sym_bool_values=self.graph_state.sym_bool_values,
            custom_obj_values=self.graph_state.custom_obj_values,
            outputs=self.graph_state.outputs,
            is_single_tensor_return=self.graph_state.is_single_tensor_return,
        )

    def serialize_graph_module_metadata(self, meta: dict[str, Any]):
        ret = {}
        if custom := meta.get("custom"):
            log.debug("\n[serialize_graph_module_metadata] %s", custom)
            try:
                ret["custom"] = json.dumps(custom)
            except Exception as e:
                raise SerializeError(
                    f"Failed to serialize custom metadata for graph with error {e}"
                ) from e

        return ret

    def serialize(self, graph_module: torch.fx.GraphModule) -> GraphModule:
        log.debug("\n[serialize]")
        graph = self.serialize_graph(graph_module)

        return GraphModule(
            graph=graph,
            signature=self.serialize_signature(self.graph_signature),
            module_call_graph=self.serialize_module_call_graph(self.module_call_graph),
            metadata=self.serialize_graph_module_metadata(graph_module.meta),
            treespec_namedtuple_fields=self.treespec_namedtuple_fields,
        )


@final
class ExportedProgramSerializer(metaclass=Final):
    def __init__(
        self,
        opset_version: dict[str, int] | None = None,
        pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
    ):
        self.opset_version: dict[str, int] = {}
        if opset_version:
            self.opset_version.update(opset_version)
        if "aten" not in self.opset_version:
            self.opset_version["aten"] = torch._C._get_max_operator_version()

        self.pickle_protocol = pickle_protocol

    def serialize(
        self,
        exported_program: ep.ExportedProgram,
        *,
        serialize_state_dict: bool = True,
        serialize_constants: bool = True,
        serialize_example_inputs: bool = True,
    ) -> _SerializedProgram:
        """
        Args:
            exported_program: Exported Program to serialize
        """
        exported_program.validate()

        gm_serializer = GraphModuleSerializer(
            exported_program.graph_signature, exported_program.module_call_graph
        )
        serialized_graph_module = gm_serializer.serialize(exported_program.graph_module)
        serialized_range_constraints = serialize_range_constraints(
            exported_program.range_constraints
        )

        # TODO: Directly serialize exported_program.constants once
        # CustomClassHolders get stored in the ExportedProgram rather than in
        # the graph
        constants: dict[str, Any] = gm_serializer.custom_objs.copy()
        for n, t in exported_program.constants.items():
            if n in constants:
                raise AssertionError(f"constant name {n!r} already exists in constants")
            constants[n] = t

        serialized_ep = ExportedProgram(
            graph_module=serialized_graph_module,
            opset_version=self.opset_version,
            range_constraints=serialized_range_constraints,
            schema_version=SchemaVersion(
                major=SCHEMA_VERSION[0],
                minor=SCHEMA_VERSION[1],
            ),
            verifiers=[v.dialect for v in exported_program.verifiers],
            torch_version=torch.__version__,
            guards_code=exported_program._guards_code,
        )

        # Test canonical form is well defined.
        canonicalize(serialized_ep, set(constants.keys()))

        # Proxy cannot be dumped, so we remove them.
        new_state_dict = remove_proxy_from_state_dict(
            exported_program.state_dict, in_place=False
        )
        serialized_state_dict = b""
        if serialize_state_dict:
            serialized_state_dict = serialize_torch_artifact(
                new_state_dict, self.pickle_protocol
            )

        serialized_constants = b""
        if serialize_constants:
            serialized_constants = serialize_torch_artifact(
                constants, self.pickle_protocol
            )

        serialized_example_inputs = b""
        if serialize_example_inputs:
            serialized_example_inputs = serialize_torch_artifact(
                exported_program.example_inputs, self.pickle_protocol
            )

        return _SerializedProgram(
            serialized_ep,
            serialized_state_dict,
            serialized_constants,
            serialized_example_inputs,
        )


@final
class GraphModuleDeserializer(metaclass=Final):
    @dataclasses.dataclass
    class Result:
        graph_module: torch.fx.GraphModule
        signature: ep.ExportGraphSignature
        module_call_graph: list[ep.ModuleCallEntry]
        names_to_symbols: dict[str, sympy.Symbol]
        state_dict: dict[str, torch.Tensor | torch.nn.Parameter]
        constants: dict[str, _ConstantAttributeType]
        example_inputs: tuple[tuple[torch.Tensor, ...], dict[str, Any]] | None

    def __init__(self) -> None:
        self.serialized_name_to_node: dict[str, torch.fx.Node] = {}
        self.serialized_name_to_meta: LazyMap = LazyMap()  # str -> MetaType
        self.graph = torch.fx.Graph()
        self.module = torch.nn.Module()

    @contextmanager
    def save_graph_module(self) -> Iterator[None]:
        saved = (
            self.graph,
            self.module,
            self.serialized_name_to_node,
            self.serialized_name_to_meta,
            self.unbacked_symbols,
        )
        self.graph = torch.fx.Graph()
        self.module = torch.nn.Module()
        self.serialized_name_to_node = {}
        self.serialized_name_to_meta = LazyMap()
        self.unbacked_symbols: set[sympy.Symbol] = set()
        try:
            yield
        finally:
            (
                self.graph,
                self.module,
                self.serialized_name_to_node,
                self.serialized_name_to_meta,
                self.unbacked_symbols,
            ) = saved

    def deserialize_extension_operator(self, serialized_target: str):
        namespace, op_name = serialized_target.split(":")
        namespace = namespace[1:]  # starting with #
        handler = _deserialization_registry[namespace]
        return handler.from_op_name(op_name)

    def deserialize_operator(self, serialized_target: str):
        if serialized_target.startswith(
            "_operator"
        ):  # TODO(zhxchen17) Follow up on this.
            module = operator
            serialized_target_names = serialized_target.split(".")[1:]
        elif serialized_target.startswith("torch"):
            module = torch  # type: ignore[misc]
            serialized_target_names = serialized_target.split(".")[1:]
        elif serialized_target.startswith("math"):
            module = math  # type: ignore[misc]
            serialized_target_names = serialized_target.split(".")[1:]
        elif serialized_target.startswith("#"):
            return self.deserialize_extension_operator(serialized_target)
        else:  # TODO(zhxchen17) Don't catch all here.
            return serialized_target

        target = module
        for name in serialized_target_names:
            if not hasattr(target, name):
                return serialized_target
            else:
                target = getattr(target, name)
        return target

    def _parse_sym_expr(
        self, expr_str: str, hint: int | bool | float | None = None
    ) -> sympy.Expr:
        """
        Parses and does bottom-up processing of sympy.Expr nodes,
        populating ShapeEnv & caching symbols as needed.
        """

        def _process_sym_expr(
            sym: sympy.Expr, hint: int | bool | float | None = None
        ) -> sympy.Expr:
            if sym.is_Integer or sym.is_Float or sym.is_Boolean:  # base case
                return sym
            else:  # recursive case
                # important to use str(expr) and not _print_sympy(),
                # str(expr) is key for self.symbol_name_to_range
                expr_str = str(sym)
                for arg in sym.args:
                    self._parse_sym_expr(arg)
                # symbol caching
                if expr_str in self.symbol_name_to_symbol:
                    sym = self.symbol_name_to_symbol[expr_str]
                else:
                    self.symbol_name_to_symbol[expr_str] = sym
                    if isinstance(sym, sympy.Symbol) and symbolic_shapes.symbol_is_type(
                        sym, (SymT.UNBACKED_INT, SymT.UNBACKED_FLOAT)
                    ):
                        self.unbacked_symbols.add(sym)
                # hints
                if hint is not None and sym not in self.shape_env.backed_var_to_val:
                    self.shape_env.add_backed_var_to_val(sym, hint)  # type: ignore[arg-type]
                # ValueRanges
                if vr := self.symbol_name_to_range.get(expr_str):
                    self.shape_env.constrain_symbol_range(
                        sym,
                        compiler_min=vr.lower,  # type: ignore[arg-type]
                        compiler_max=vr.upper,  # type: ignore[arg-type]
                    )
                # ShapeEnv meta
                if isinstance(sym, sympy.Symbol):
                    self.shape_env.var_to_stack[sym] = CapturedTraceback.extract(skip=1)
            return sym

        expr = sympy.sympify(
            expr_str,
            locals={**self.sympy_functions, **self.symbol_name_to_symbol},
        )
        return _process_sym_expr(expr, hint)

    def deserialize_sym_int(self, s: SymInt) -> int | torch.SymInt:
        val = s.value
        if s.type == "as_expr":
            if val.hint is None:
                hint = None
            else:
                if val.hint.type != "as_int":
                    raise AssertionError(
                        f"expected hint type 'as_int', got {val.hint.type!r}"
                    )
                hint = val.hint.value

            sym = self._parse_sym_expr(val.expr_str, hint)
            return self.shape_env.create_symintnode(sym, hint=hint)
        elif s.type == "as_int":
            if type(val) is not int:
                raise AssertionError(f"expected int, got {type(val).__name__}")
            return val
        else:
            raise SerializeError(
                f"SymInt has invalid field type {s.type} with value {s.value}"
            )

    def deserialize_sym_float(self, s: SymFloat) -> float | torch.SymFloat:
        val = s.value
        if s.type == "as_expr":
            hint = val.hint.as_float if val.hint else None
            sym = self._parse_sym_expr(val.expr_str, hint)
            return self.shape_env.create_symfloatnode(sym, hint=hint)
        elif s.type == "as_float":
            if not isinstance(val, float):
                raise AssertionError(f"expected float, got {type(val).__name__}")
            return val
        else:
            raise SerializeError(
                f"SymFloat has invalid field type {s.type} with value {s.value}"
            )

    def deserialize_sym_bool(self, s: SymBool) -> bool | torch.SymBool:
        val = s.value
        if s.type == "as_expr":
            expr = self._parse_sym_expr(val.expr_str)
            return self.shape_env.create_symboolnode(expr)
        elif s.type == "as_bool":
            if not isinstance(val, bool):
                raise AssertionError(f"expected bool, got {type(val).__name__}")
            return val
        else:
            raise SerializeError(
                f"SymBool has invalid field type {s.type} with value {s.value}"
            )

    def deserialize_tensor_meta(
        self,
        tensor_meta: TensorMeta,
    ) -> FakeTensor:
        with self.fake_tensor_mode:
            return cast(
                FakeTensor,
                torch.empty_strided(
                    tuple(self.deserialize_sym_int(val) for val in tensor_meta.sizes),  # type: ignore[misc]
