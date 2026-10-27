        inputs[1] = self.add_immediate_int_vector(shape)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.RESHAPE, inputs, outputs)

    def add_flatten(self, node):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))

        _start_ctype, start_dim = self.get_constant_value(node.inputsAt(1), "IntType")
        _end_ctype, end_dim = self.get_constant_value(node.inputsAt(2), "IntType")

        # channels last with channels == 1 or (height & width both 1)
        is_trivial_flatten = len(in_oper.shape) == 4 and (
            in_oper.shape[1] == 1 or (in_oper.shape[2] == 1 and in_oper.shape[3] == 1)
        )
        if in_oper.dim_order != DimOrder.PRESUMED_CONTIGUOUS and not is_trivial_flatten:
            raise Exception(  # noqa: TRY002
                "Currently, flatten is not supported on NHWC tensors unless C=1 or H=W=1"
            )

        if start_dim < 0:
            start_dim += len(in_oper.shape)
        if end_dim < 0:
            end_dim += len(in_oper.shape)

        out_shape = (
            in_oper.shape[:start_dim]
            + (functools.reduce(operator.mul, in_oper.shape[start_dim : end_dim + 1]),)
            + in_oper.shape[end_dim + 1 :]
        )

        if any(dim == 0 for dim in in_oper.shape[start_dim : end_dim + 1]):
            raise Exception(  # noqa: TRY002
                "Flattening flexible dims is not supported yet"
            )
        non_flattened_dims = in_oper.shape[:start_dim] + in_oper.shape[end_dim + 1 :]
        if non_flattened_dims.count(0) > 1:
            raise Exception("Only 1 dim can be flexible")  # noqa: TRY002

        out_oper = in_oper._replace(
            shape=out_shape, dim_order=DimOrder.PRESUMED_CONTIGUOUS
        )
        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)

        for idx, dim in enumerate(out_shape):
            if dim == 0:
                self.forward_operand_shape(out_id, idx, in_id, in_oper.shape.index(0))

        inputs_1 = tuple(dim if dim != 0 else -1 for dim in out_shape)
        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector(inputs_1)

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.RESHAPE, inputs, outputs)

    def add_slice(self, node):
        if node.inputsSize() != 5:
            raise AssertionError(
                f"expected node.inputsSize() == 5, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        _, dim_value = self.get_constant_value(node.inputsAt(1))
        _, start_value = self.get_constant_value(node.inputsAt(2))
        _, stop_value = self.get_constant_value(node.inputsAt(3))
        _, step_value = self.get_constant_value(node.inputsAt(4))

        if start_value is None:
            start_value = 0
        if stop_value is None:
            stop_value = sys.maxsize

        if start_value < 0:
            start_value += in_oper.shape[dim_value]
        elif start_value == sys.maxsize:
            start_value = 0

        if start_value == 0 and stop_value == sys.maxsize:
            self._identity(node)
            return

        if in_oper.shape[dim_value] == 0:
            raise Exception("Unable to slice with flexible shape")  # noqa: TRY002

        if stop_value < 0:
            stop_value += in_oper.shape[dim_value]
        elif stop_value == sys.maxsize:
            stop_value = in_oper.shape[dim_value]

        if start_value >= stop_value:
            raise Exception(  # noqa: TRY002
                "Slice start value should be less than stop value"
            )

        out_len = (stop_value - start_value) // step_value
        out_shape = tuple(
            out_len if i == dim_value else dim for i, dim in enumerate(in_oper.shape)
        )
        out_id = self.add_tensor_operand(
            node.outputsAt(0), in_oper._replace(shape=out_shape)
        )

        # flex inputs
        end_mask = 0
        for idx, dim in enumerate(out_shape):
            if dim == 0:
                self.forward_operand_shape(out_id, idx, in_id, idx)
                end_mask |= 1 << idx

        inputs = [None] * 7
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector(
            [start_value if i == dim_value else 0 for i in range(len(in_oper.shape))]
        )
        inputs[2] = self.add_immediate_int_vector(
            [
                stop_value if i == dim_value else dim
                for i, dim in enumerate(in_oper.shape)
            ]
        )
        inputs[3] = self.add_immediate_int_vector(
            [step_value if i == dim_value else 1 for i in range(len(in_oper.shape))]
        )
        inputs[4] = self.add_immediate_int_scalar(0)  # begin mask
        inputs[5] = self.add_immediate_int_scalar(end_mask)
        inputs[6] = self.add_immediate_int_scalar(0)  # shrink axis mask

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.STRIDED_SLICE, inputs, outputs)

    def add_size(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        _, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        _, value = self.constants[node.inputsAt(1)]
        res = in_oper.shape[value]
        output = node.outputsAt(0)
        self.add_constant_value(output, output.type(), res)

    def add_cat(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        tensors = self.tensor_sequences[node.inputsAt(0)]
        _, dim = self.get_constant_value(node.inputsAt(1), "IntType")

        if len(tensors) <= 0:
            raise AssertionError(f"expected len(tensors) > 0, got {len(tensors)}")
        in_ids = []
        out_oper = None
        out_dim_size = 0
        for inp in tensors:
            in_id, in_oper = self.get_tensor_operand_by_jitval(inp)
            if out_oper is None:
                out_shape = change_element(in_oper.shape, dim, -1)
                out_oper = in_oper._replace(shape=out_shape)
            if in_oper.op_type != out_oper.op_type:
                raise AssertionError(
                    f"in_oper.op_type {in_oper.op_type} != out_oper.op_type {out_oper.op_type}"
                )
            if in_oper.dim_order != out_oper.dim_order:
                raise AssertionError(
                    f"in_oper.dim_order {in_oper.dim_order} != out_oper.dim_order {out_oper.dim_order}"
                )
            if change_element(in_oper.shape, dim, -1) != change_element(
                out_oper.shape, dim, -1
            ):
                raise AssertionError(
                    f"shape mismatch: {change_element(in_oper.shape, dim, -1)} != {change_element(out_oper.shape, dim, -1)}"
                )
            # TODO: Possibly check scale and zero point.
            in_ids.append(in_id)
            # TODO: Possibly support variable-sized inputs.
            out_dim_size += in_oper.shape[dim]

        if out_oper is None:
            raise AssertionError("out_oper must not be None")
        out_oper = out_oper._replace(
            shape=change_element(out_oper.shape, dim, out_dim_size)
        )

        if in_oper.dim_order == DimOrder.CHANNELS_LAST:  # type: ignore[possibly-undefined]
            if len(out_oper.shape) != 4:
                raise AssertionError(
                    f"expected len(out_oper.shape) == 4 for CHANNELS_LAST, got {len(out_oper.shape)}"
                )
            nnapi_dim = [0, 3, 1, 2][dim]
        else:
            nnapi_dim = dim

        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)
        for idx, d in enumerate(out_oper.shape):
            if d == 0:
                if idx == dim:
                    shape = " + ".join(flex_name(ip_id, dim) for ip_id in in_ids)
                    self.compute_operand_shape(out_id, idx, shape)
                else:
                    self.forward_operand_shape(out_id, idx, in_ids[0], idx)

        inputs = in_ids + [self.add_immediate_int_scalar(nnapi_dim)]

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.CONCATENATION, inputs, outputs)

    def add_mean(self, node):
        if node.inputsSize() != 4:
            raise AssertionError(
                f"expected node.inputsSize() == 4, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        dim_ctype, dim = self.get_constant_value(node.inputsAt(1))
        if dim_ctype.kind() != "ListType":
            raise AssertionError(f"expected dim_ctype ListType, got {dim_ctype.kind()}")
        if dim_ctype.getElementType().kind() != "IntType":
            raise AssertionError(
                f"expected dim element type IntType, got {dim_ctype.getElementType().kind()}"
            )
        _, keep_dim = self.get_constant_value(node.inputsAt(2), "BoolType")
        # Expect None for dtype
        self.get_constant_value(node.inputsAt(3), "NoneType")

        if in_oper.dim_order == DimOrder.CHANNELS_LAST:
            if len(in_oper.shape) != 4:
                raise AssertionError(
                    f"expected len(in_oper.shape) == 4 for CHANNELS_LAST, got {len(in_oper.shape)}"
                )
            nnapi_dim = [[0, 3, 1, 2][d] for d in dim]
        else:
            nnapi_dim = dim

        collapsed_dims = set()
        for d in dim:
            if d < 0:
                d += len(in_oper.shape)
            collapsed_dims.add(d)

        if in_oper.dim_order == DimOrder.CHANNELS_LAST and not keep_dim:
            if not collapsed_dims.issuperset({2, 3}):
                raise AssertionError(
                    f"expected collapsed_dims to include {{2, 3}}, got {collapsed_dims}"
                )
            out_dim_order = DimOrder.PRESUMED_CONTIGUOUS
        else:
            out_dim_order = in_oper.dim_order

        out_shape = []
        for i, s in enumerate(in_oper.shape):
            if i not in collapsed_dims:
                out_shape.append(s)
            elif keep_dim:
                out_shape.append(1)

        out_oper = in_oper._replace(shape=out_shape, dim_order=out_dim_order)

        inputs = [None] * 3
        inputs[0] = in_id
        inputs[1] = self.add_immediate_int_vector(nnapi_dim)
        inputs[2] = self.add_immediate_int_scalar(keep_dim)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.MEAN, inputs, outputs)

    def add_quantize(self, node):
        if node.inputsSize() != 4:
            raise AssertionError(
                f"expected node.inputsSize() == 4, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        if in_oper.dim_order != DimOrder.CHANNELS_LAST:
            raise Exception(  # noqa: TRY002
                "Most hardware backends prefer NHWC quantized tensors.  "
                "Try setting `t.nnapi_nhwc = True` on your tensor inputs.  "
            )
        _, scale = self.get_constant_value(node.inputsAt(1), "FloatType")
        _, zero_point = self.get_constant_value(node.inputsAt(2), "IntType")
        _, scalar_type = self.get_constant_value(node.inputsAt(3), "IntType")
        if scalar_type != TorchScalarTypes.QUINT8.value:
            raise Exception(  # noqa: TRY002
                "PyTorch NNAPI export only supports quantized tensors "
                "with the quint8 dtype."
            )
        op_type = NNAPI_OperandCode.TENSOR_QUANT8_ASYMM

        out_oper = in_oper._replace(
            op_type=op_type,
            scale=scale,
            zero_point=zero_point,
        )

        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.QUANTIZE, inputs, outputs)

    def add_dequantize(self, node):
        if node.inputsSize() != 1:
            raise AssertionError(
                f"expected node.inputsSize() == 1, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        out_oper = in_oper._replace(
            op_type=NNAPI_OperandCode.TENSOR_FLOAT32,
            scale=0.0,
            zero_point=0,
        )

        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.DEQUANTIZE, inputs, outputs)

    def add_pointwise_simple_unary_op(self, node, opcode):
        if node.inputsSize() != 1:
            raise AssertionError(
                f"expected node.inputsSize() == 1, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))

        out_oper = in_oper
        if opcode == NNAPI_OperationCode.LOGISTIC:
            # NNAPI docs: For ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, the scale
            # must be 1.f / 256 and the zeroPoint must be 0.
            # https://fburl.com/h52stoog
            if in_oper.op_type == NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
                out_oper = in_oper._replace(zero_point=0, scale=1.0 / 256)

        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)

        for idx, dim in enumerate(in_oper.shape):
            if dim == 0:
                self.forward_operand_shape(out_id, idx, in_id, idx)

        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(opcode, inputs, outputs)

    def _do_add_binary(self, node, opcode, fuse_code, *, qparams=None):
        """Helper for pointwise binary broadcast ops with superfluous extra args."""
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        if node.inputsAt(0).type().kind() != "TensorType":
            raise AssertionError(
                f"expected inputsAt(0) TensorType, got {node.inputsAt(0).type().kind()}"
            )
        if node.inputsAt(1).type().kind() != "TensorType":
            raise AssertionError(
                f"expected inputsAt(1) TensorType, got {node.inputsAt(1).type().kind()}"
            )

        if self.has_operand_for_jitval(node.inputsAt(0)):
            in0_id, in0_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
            in1_id, in1_oper = self.get_tensor_operand_or_constant(
                node.inputsAt(1), in0_oper.dim_order
            )
        elif self.has_operand_for_jitval(node.inputsAt(1)):
            in1_id, in1_oper = self.get_tensor_operand_by_jitval(node.inputsAt(1))
            in0_id, in0_oper = self.get_tensor_operand_or_constant(
                node.inputsAt(0), in1_oper.dim_order
            )
        else:
            raise Exception(  # noqa: TRY002
                f"Can't do a NNAPI binary op: {opcode} on two constants"
            )

        if in0_oper.op_type != in1_oper.op_type:
            raise AssertionError(
                f"in0_oper.op_type {in0_oper.op_type} != in1_oper.op_type {in1_oper.op_type}"
            )
        in0_id, in0_oper, in1_id, in1_oper = self.transpose_for_broadcast(
            in0_id, in0_oper, in1_id, in1_oper
        )
        # NOTE: PyTorch and NNAPI have the same broadcast semantics.
        out_shape = broadcast_shapes(in0_oper.shape, in1_oper.shape)
        out_oper = in0_oper._replace(shape=out_shape)
        if qparams is not None:
            scale, zp = qparams
            out_oper = out_oper._replace(scale=scale, zero_point=zp)

        out_id = self.add_tensor_operand(node.outputsAt(0), out_oper)
        for idx, (d0, d1) in enumerate(zip(in0_oper.shape, in1_oper.shape)):
            if d0 == 1 and d1 == 0:
                self.forward_operand_shape(out_id, idx, in1_id, idx)
            elif d0 == 0 and d1 == 1:
                self.forward_operand_shape(out_id, idx, in0_id, idx)
            elif d0 == 0 and d1 == 0:
                self.flexible_shape_computation_lines.append(
                    f"assert {flex_name(in0_id, idx)} == {flex_name(in1_id, idx)}"
                )
                self.forward_operand_shape(out_id, idx, in0_id, idx)

        inputs = [None] * 3
        inputs[0] = in0_id
        inputs[1] = in1_id
        inputs[2] = self.add_immediate_int_scalar(fuse_code)

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(opcode, inputs, outputs)

    def add_pointwise_simple_binary_broadcast_op(self, node, opcode, fuse_code):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        self._do_add_binary(node, opcode, fuse_code)

    def add_add_sub_op(self, node, opcode, fuse_code):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )

        _, alpha = self.get_constant_value(node.inputsAt(2), "IntType")
        if alpha != 1:
            raise Exception(  # noqa: TRY002
                "NNAPI does not support add/sub with alpha."
            )

        self._do_add_binary(node, opcode, fuse_code)

    def add_qadd(self, node, opcode, fuse_code):
        if node.inputsSize() != 4:
            raise AssertionError(
                f"expected node.inputsSize() == 4, got {node.inputsSize()}"
            )

        _, scale = self.get_constant_value(node.inputsAt(2), "FloatType")
        _, zero_point = self.get_constant_value(node.inputsAt(3), "IntType")

        self._do_add_binary(node, opcode, fuse_code, qparams=(scale, zero_point))

    def add_softmax(self, node):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )
        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))

        _, softmax_dim = self.get_constant_value(node.inputsAt(1), "IntType")

        out_id = self.add_tensor_operand(node.outputsAt(0), in_oper)
        for dim, size in enumerate(in_oper.shape):
            if size == 0:
                self.forward_operand_shape(out_id, dim, in_id, dim)

        inputs = [None] * 3
        inputs[0] = in_id
        inputs[1] = self.add_immediate_float_scalar(
            1.0
        )  # positive scaling factor of exponent, beta
        inputs[2] = self.add_immediate_int_scalar(softmax_dim)

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.SOFTMAX, inputs, outputs)

    def add_hardtanh(self, node):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval_fixed_size(node.inputsAt(0))
        _, min_val = self.get_constant_value(node.inputsAt(1), "FloatType")
        _, max_val = self.get_constant_value(node.inputsAt(2), "FloatType")

        op_map = {
            (-1, 1): NNAPI_OperationCode.RELU1,
            (0, 6): NNAPI_OperationCode.RELU6,  # noqa: E201
        }

        opcode = op_map.get((min_val, max_val))
        if opcode is None:
            raise Exception(  # noqa: TRY002
                "NNAPI only supports hardtanh with args (-1, 1) or (0, 6)."
            )

        inputs = [None] * 1
        inputs[0] = in_id

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), in_oper)

        self.add_operation(opcode, inputs, outputs)

    def add_prelu_op(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        if node.inputsAt(0).type().kind() != "TensorType":
            raise AssertionError(
                f"expected inputsAt(0) TensorType, got {node.inputsAt(0).type().kind()}"
            )
        if node.inputsAt(1).type().kind() != "TensorType":
            raise AssertionError(
                f"expected inputsAt(1) TensorType, got {node.inputsAt(1).type().kind()}"
            )

        in_id, in_oper = self.get_tensor_operand_by_jitval(node.inputsAt(0))
        w_id, w_oper = self.get_tensor_operand_for_weight(node.inputsAt(1))
        if len(w_oper.shape) != 1:
            raise AssertionError(
                f"expected len(w_oper.shape) == 1, got {len(w_oper.shape)}"
            )
        if w_oper.shape[0] <= 0:
            raise AssertionError(f"expected w_oper.shape[0] > 0, got {w_oper.shape[0]}")
        if w_oper.shape[0] > 1:
            if in_oper.use_nchw():
                # TODO: Support this by adding trailing 1 dims.
                raise Exception(  # noqa: TRY002
                    "Per-channel PReLU only supports channels_last right now."
                )

        out_id = self.add_tensor_operand(node.outputsAt(0), in_oper)
        for dim, size in enumerate(in_oper.shape):
            if size > 0:
                pass
            elif dim <= 1:
                raise Exception(  # noqa: TRY002
                    "PReLU requires fixed size for dim 0 and dim 1."
                )
            else:
                self.forward_operand_shape(out_id, dim, in_id, dim)

        inputs = [None] * 2
        inputs[0] = in_id
        inputs[1] = w_id

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.PRELU, inputs, outputs)

    def add_pool2d_node(self, node, opcode):
        if node.inputsSize() != 6:
            raise AssertionError(
                f"expected node.inputsSize() == 6, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        image, kernel, stride, padding, dilation, _ceil_mode = node.inputs()

        stride = stride or kernel

        # TODO: Validate ceil_mode semantics.

        args = self.get_conv_pool_args_2d_from_jit(
            self.get_size_arg(kernel), stride, padding, dilation
        )
        if args.dilation_h != 1 or args.dilation_w != 1:
            raise Exception("NNAPI does not support dilated pooling.")  # noqa: TRY002

        image_id, image_oper = self.get_tensor_operand_by_jitval_fixed_size(image)
        if len(image_oper.shape) != 4:
            raise AssertionError(
                f"expected len(image_oper.shape) == 4, got {len(image_oper.shape)}"
            )

        out_shape = get_conv_pool_shape(
            image_oper.shape, args, image_oper.shape[1], False
        )
        use_nchw = image_oper.use_nchw()

        inputs = [None] * 11
        inputs[0] = image_id
        inputs[1] = self.add_immediate_int_scalar(args.pad_l)
        inputs[2] = self.add_immediate_int_scalar(args.pad_r)
        inputs[3] = self.add_immediate_int_scalar(args.pad_t)
        inputs[4] = self.add_immediate_int_scalar(args.pad_b)
        inputs[5] = self.add_immediate_int_scalar(args.stride_w)
        inputs[6] = self.add_immediate_int_scalar(args.stride_h)
        inputs[7] = self.add_immediate_int_scalar(args.kernel_w)
        inputs[8] = self.add_immediate_int_scalar(args.kernel_h)
        inputs[9] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)
        inputs[10] = self.add_immediate_bool_scalar(use_nchw)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(
            node.outputsAt(0), image_oper._replace(shape=out_shape)
        )

        self.add_operation(opcode, inputs, outputs)

    def add_avg_pool2d(self, node):
        if node.inputsSize() != 7:
            raise AssertionError(
                f"expected node.inputsSize() == 7, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        (
            image,
            kernel,
            stride,
            padding,
            _ceil_mode,
            count_include_pad,
            divisor_override,
        ) = node.inputs()

        _, count_include_pad_value = self.get_constant_value(count_include_pad)
        _, divisor_override_value = self.get_constant_value(divisor_override)
        if not count_include_pad_value or divisor_override_value:
            raise Exception(  # noqa: TRY002
                "NNAPI doesn't support count_include_pad=False or divisor_override"
            )

        args = self.get_conv_pool_args_2d_from_jit(
            self.get_size_arg(kernel), stride, padding
        )

        image_id, image_oper = self.get_tensor_operand_by_jitval(image)
        if len(image_oper.shape) != 4:
            raise AssertionError(
                f"expected len(image_oper.shape) == 4, got {len(image_oper.shape)}"
            )

        out_shape = get_conv_pool_shape(
            image_oper.shape, args, image_oper.shape[1], False
        )
        use_nchw = image_oper.use_nchw()

        inputs = [None] * 11
        inputs[0] = image_id
        inputs[1] = self.add_immediate_int_scalar(args.pad_l)
        inputs[2] = self.add_immediate_int_scalar(args.pad_r)
        inputs[3] = self.add_immediate_int_scalar(args.pad_t)
        inputs[4] = self.add_immediate_int_scalar(args.pad_b)
        inputs[5] = self.add_immediate_int_scalar(args.stride_w)
        inputs[6] = self.add_immediate_int_scalar(args.stride_h)
        inputs[7] = self.add_immediate_int_scalar(args.kernel_w)
        inputs[8] = self.add_immediate_int_scalar(args.kernel_h)
        inputs[9] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)
        inputs[10] = self.add_immediate_bool_scalar(use_nchw)

        outputs = [None] * 1
        out_id = self.add_tensor_operand(
            node.outputsAt(0), image_oper._replace(shape=out_shape)
        )
        self._handle_conv_pool_flexible_input(out_id, image, args, False)
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.AVERAGE_POOL_2D, inputs, outputs)

    def add_adaptive_avg_pool2d(self, node):
        if node.inputsSize() != 2:
            raise AssertionError(
                f"expected node.inputsSize() == 2, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        image_id, image_oper = self.get_tensor_operand_by_jitval_fixed_size(
            node.inputsAt(0)
        )
        if len(image_oper.shape) != 4:
            raise AssertionError(
                f"expected len(image_oper.shape) == 4, got {len(image_oper.shape)}"
            )

        size_ctype, size_arg = self.get_constant_value(node.inputsAt(1))
        if size_ctype.kind() != "ListType":
            raise AssertionError(
                f"expected size_ctype ListType, got {size_ctype.kind()}"
            )
        if size_ctype.getElementType().kind() != "IntType":
            raise AssertionError(
                f"expected size element type IntType, got {size_ctype.getElementType().kind()}"
            )
        if size_arg != [1, 1]:
            raise Exception(  # noqa: TRY002
                "NNAPI only supports adaptive_avg_pool2d with output size (1, 1)."
            )

        out_shape = image_oper.shape[0:2] + tuple(size_arg)
        use_nchw = image_oper.use_nchw()

        inputs = [None] * 11
        inputs[0] = image_id
        inputs[1] = self.add_immediate_int_scalar(0)
        inputs[2] = self.add_immediate_int_scalar(0)
        inputs[3] = self.add_immediate_int_scalar(0)
        inputs[4] = self.add_immediate_int_scalar(0)
        inputs[5] = self.add_immediate_int_scalar(1)
        inputs[6] = self.add_immediate_int_scalar(1)
        inputs[7] = self.add_immediate_int_scalar(image_oper.shape[3])
        inputs[8] = self.add_immediate_int_scalar(image_oper.shape[2])
        inputs[9] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)
        inputs[10] = self.add_immediate_bool_scalar(use_nchw)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(
            node.outputsAt(0), image_oper._replace(shape=out_shape)
        )

        self.add_operation(NNAPI_OperationCode.AVERAGE_POOL_2D, inputs, outputs)

    def add_upsample_nearest2d(self, node):
        if not (node.inputsSize() == 3 or node.inputsSize() == 4):
            raise AssertionError(
                f"expected node.inputsSize() == 3 or 4, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        if node.inputsSize() == 3:
            image, size_jit, scale_jit = node.inputs()
        else:
            image, size_jit, scale_h_jit, scale_w_jit = node.inputs()
        size_ctype, size_arg = self.get_constant_value(size_jit)

        if node.inputsSize() == 3:
            scale_ctype, scale_arg = self.get_constant_value(scale_jit)  # type: ignore[possibly-undefined]
        else:
            scale_h_ctype, scale_h_arg = self.get_constant_value(scale_h_jit)  # type: ignore[possibly-undefined]
            scale_w_ctype, _scale_w_arg = self.get_constant_value(scale_w_jit)  # type: ignore[possibly-undefined]

            # The only way for the 4-argument overload of upsample_nearest2d to
            # have been added to the graph without error is if the scale_h and
            # scale_w arguments are None
            if scale_h_ctype.kind() != "NoneType":
                raise AssertionError(
                    f"expected scale_h_ctype NoneType, got {scale_h_ctype.kind()}"
                )
            if scale_w_ctype.kind() != "NoneType":
                raise AssertionError(
                    f"expected scale_w_ctype NoneType, got {scale_w_ctype.kind()}"
                )

            scale_ctype = scale_h_ctype
            scale_arg = scale_h_arg

        image_id, image_oper = self.get_tensor_operand_by_jitval(image)
        if len(image_oper.shape) != 4:
            raise AssertionError(
                f"expected len(image_oper.shape) == 4, got {len(image_oper.shape)}"
            )

        if size_ctype.kind() != "NoneType" and scale_ctype.kind() != "NoneType":
            raise Exception("Size and scale cannot both be non-None.")  # noqa: TRY002
        elif size_ctype.kind() != "NoneType":
            if size_ctype.kind() != "ListType":
                raise AssertionError(
                    f"expected size_ctype ListType, got {size_ctype.kind()}"
                )
            if size_ctype.getElementType().kind() != "IntType":
                raise AssertionError(
                    f"expected size element type IntType, got {size_ctype.getElementType().kind()}"
                )
            if scale_ctype.kind() != "NoneType":
                raise AssertionError(
                    f"expected scale_ctype NoneType, got {scale_ctype.kind()}"
                )
            if scale_arg is not None:
                raise AssertionError(f"expected scale_arg None, got {scale_arg}")
            if not isinstance(size_arg, list):
                raise AssertionError(
                    f"expected size_arg to be list, got {type(size_arg)}"
                )
            if not size_arg:
                raise AssertionError("expected size_arg to be non-empty")
            if not all(isinstance(val, int) for val in size_arg):
                raise AssertionError("expected all size_arg values to be int")
            if len(size_arg) == 1:
                size_arg = size_arg * 2
            if len(size_arg) != 2:
                raise AssertionError(
                    f"expected len(size_arg) == 2, got {len(size_arg)}"
                )
            out_h = size_arg[0]
            out_w = size_arg[1]
            arg_h = self.add_immediate_int_scalar(out_h)
            arg_w = self.add_immediate_int_scalar(out_w)
        elif scale_ctype.kind() != "NoneType":
            if scale_ctype.kind() != "ListType":
                raise AssertionError(
                    f"expected scale_ctype ListType, got {scale_ctype.kind()}"
                )
            if scale_ctype.getElementType().kind() != "FloatType":
                raise AssertionError(
                    f"expected scale element type FloatType, got {scale_ctype.getElementType().kind()}"
                )
            if size_ctype.kind() != "NoneType":
                raise AssertionError(
                    f"expected size_ctype NoneType, got {size_ctype.kind()}"
                )
            if size_arg is not None:
                raise AssertionError(f"expected size_arg None, got {size_arg}")
            if not isinstance(scale_arg, list):
                raise AssertionError(
                    f"expected scale_arg to be list, got {type(scale_arg)}"
                )
            if not scale_arg:
                raise AssertionError("expected scale_arg to be non-empty")
            if not all(isinstance(val, float) for val in scale_arg):
                raise AssertionError("expected all scale_arg values to be float")
            if len(scale_arg) == 1:
                scale_arg = scale_arg * 2
            if len(scale_arg) != 2:
                raise AssertionError(
                    f"expected len(scale_arg) == 2, got {len(scale_arg)}"
                )
            out_h = int(scale_arg[0] * image_oper.shape[2])
            out_w = int(scale_arg[1] * image_oper.shape[3])
            arg_h = self.add_immediate_float_scalar(scale_arg[0])
            arg_w = self.add_immediate_float_scalar(scale_arg[1])
        else:
            raise Exception("Size and scale cannot both be None.")  # noqa: TRY002

        out_shape = (image_oper.shape[0], image_oper.shape[1], out_h, out_w)
        use_nchw = image_oper.use_nchw()
        out_id = self.add_tensor_operand(
            node.outputsAt(0), image_oper._replace(shape=out_shape)
        )

        if image_oper.shape[0] == 0 or image_oper.shape[1] == 0:
            raise Exception("Flexible batch or channels not supported")  # noqa: TRY002

        # Handle variable input size
        for dim in (2, 3):  # h, w indices
            if image_oper.shape[dim] == 0:
                if size_ctype.kind() != "NoneType":
                    # pyrefly: ignore [unsupported-operation]
                    self.compute_operand_shape(out_id, dim, size_arg[dim - 2])
                elif scale_ctype.kind() != "NoneType":
                    self.compute_operand_shape(
                        out_id,
                        dim,
                        # pyrefly: ignore [unsupported-operation]
                        f"int({scale_arg[dim - 2]} * {flex_name(image_id, dim)})",
                    )
                else:
                    raise Exception(  # noqa: TRY002
                        "Size and scale cannot both be None."
                    )

        inputs = [None] * 4
        inputs[0] = image_id
        inputs[1] = arg_w
        inputs[2] = arg_h
        inputs[3] = self.add_immediate_bool_scalar(use_nchw)

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.RESIZE_NEAREST_NEIGHBOR, inputs, outputs)

    def add_addmm(self, node):
        if node.inputsSize() != 5:
            raise AssertionError(
                f"expected node.inputsSize() == 5, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        jit_bias, jit_input, jit_weight, jit_beta, jit_alpha = node.inputs()

        for jitval in (jit_beta, jit_alpha):
            scale_ctype, scale_value = self.get_constant_value(jitval)
            if scale_ctype.kind() not in ("IntType", "FloatType"):
                raise AssertionError(
                    f"expected scale_ctype IntType or FloatType, got {scale_ctype.kind()}"
                )
            if scale_value != 1:
                raise Exception(  # noqa: TRY002
                    "NNAPI Fully-Connected does not support alpha and beta."
                )

        self.add_addmm_or_linear(node, True, jit_input, jit_weight, jit_bias)

    def add_linear(self, node):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        jit_input, jit_weight, jit_bias = node.inputs()

        self.add_addmm_or_linear(node, False, jit_input, jit_weight, jit_bias)

    def add_addmm_or_linear(
        self, node, transpose_weight, jit_input, jit_weight, jit_bias
    ):
        input_id, input_oper = self.get_tensor_operand_by_jitval(jit_input)
        bias_id, bias_oper = self.get_tensor_operand_for_weight(jit_bias)

        if len(input_oper.shape) != 2:
            raise AssertionError(
                f"expected len(input_oper.shape) == 2, got {len(input_oper.shape)}"
            )
        if len(bias_oper.shape) != 1:
            raise AssertionError(
                f"expected len(bias_oper.shape) == 1, got {len(bias_oper.shape)}"
            )

        # TODO: Transform at load time to share weights with CPU model.
        _, weight_tensor = self.get_constant_value(jit_weight, "TensorType")
        if len(weight_tensor.shape) != 2:
            raise AssertionError(
                f"expected len(weight_tensor.shape) == 2, got {len(weight_tensor.shape)}"
            )
        if transpose_weight:
            nnapi_weight_tensor = weight_tensor.t().contiguous()
        else:
            nnapi_weight_tensor = weight_tensor.contiguous()
        weight_id = self.add_tensor_operand_for_weight(nnapi_weight_tensor)
        weight_oper = self.operands[weight_id]

        out_shape = (input_oper.shape[0], weight_oper.shape[0])
        out_id = self.add_tensor_operand(
            node.outputsAt(0), input_oper._replace(shape=out_shape)
        )

        if input_oper.shape[0] == 0:
            self.forward_operand_shape(out_id, 0, input_id, 0)

        inputs = [None] * 4
        inputs[0] = input_id
        inputs[1] = weight_id
        inputs[2] = bias_id
        inputs[3] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)

        outputs = [None] * 1
        outputs[0] = out_id

        self.add_operation(NNAPI_OperationCode.FULLY_CONNECTED, inputs, outputs)

    def add_qlinear(self, node):
        if node.inputsSize() != 4:
            raise AssertionError(
                f"expected node.inputsSize() == 4, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )
        (
            jit_input,
            jit_packed_weight,
            jit_scale,
            jit_zero_point,
        ) = node.inputs()

        input_id, input_oper = self.get_tensor_operand_by_jitval_fixed_size(jit_input)
        # TODO: Support automatic reshape
        if len(input_oper.shape) != 2:
            raise AssertionError(
                f"expected len(input_oper.shape) == 2, got {len(input_oper.shape)}"
            )

        _, out_scale = self.get_constant_value(jit_scale, "FloatType")
        _, out_zero_point = self.get_constant_value(jit_zero_point, "IntType")
        weight_ctype, packed_weight = self.get_constant_value(jit_packed_weight)
        if weight_ctype.name() != "LinearPackedParamsBase":
            raise AssertionError(
                f"expected weight_ctype LinearPackedParamsBase, got {weight_ctype.name()}"
            )
        raw_weight, raw_bias = packed_weight.__getstate__()[0]
        if raw_bias is None:
            raise AssertionError("raw_bias must not be None")

        if len(raw_weight.shape) != 2:
            raise AssertionError(
                f"expected len(raw_weight.shape) == 2, got {len(raw_weight.shape)}"
            )
        if len(raw_bias.shape) != 1:
            raise AssertionError(
                f"expected len(raw_bias.shape) == 1, got {len(raw_bias.shape)}"
            )
        if raw_bias.shape[0] != raw_weight.shape[0]:
            raise AssertionError(
                f"raw_bias.shape[0] {raw_bias.shape[0]} != raw_weight.shape[0] {raw_weight.shape[0]}"
            )
        if raw_weight.shape[1] != input_oper.shape[1]:
            raise AssertionError(
                f"raw_weight.shape[1] {raw_weight.shape[1]} != input_oper.shape[1] {input_oper.shape[1]}"
            )

        if raw_weight.qscheme() != torch.per_tensor_affine:
            raise AssertionError(
                f"expected raw_weight.qscheme() per_tensor_affine, got {raw_weight.qscheme()}"
            )
        if raw_weight.dtype == torch.quint8:
            unsigned_weight = raw_weight
        else:
            if raw_weight.dtype != torch.qint8:
                raise AssertionError(
                    f"expected raw_weight.dtype qint8, got {raw_weight.dtype}"
                )
            unsigned_weight = torch._make_per_tensor_quantized_tensor(
                (raw_weight.int_repr().int() + 128).to(torch.uint8),
                scale=raw_weight.q_scale(),
                zero_point=raw_weight.q_zero_point() + 128,
            )
        weight_scale = unsigned_weight.q_scale()
        bias_scale = input_oper.scale * weight_scale
        int_bias = torch.quantize_per_tensor(raw_bias, bias_scale, 0, torch.qint32)
        bias_id = self.add_tensor_operand_for_weight(int_bias)

        multiplier = input_oper.scale * weight_scale / out_scale
        if multiplier <= 0:
            raise AssertionError(f"expected multiplier > 0, got {multiplier}")
        if multiplier >= 1:
            raise Exception(  # noqa: TRY002
                "Quantized convolution multiplier is greater than 1.  "
                "This is supported by NNAPI, but not by most hardware backends.  "
                "Try training a model without quantization-aware training.  "
            )

        # TODO: Transform at load time to share weights with CPU model.
        nnapi_weight_tensor = unsigned_weight.contiguous()
        weight_id = self.add_tensor_operand_for_weight(nnapi_weight_tensor)
        weight_oper = self.operands[weight_id]

        out_shape = (input_oper.shape[0], weight_oper.shape[0])
        out_oper = input_oper._replace(
            shape=out_shape,
            scale=out_scale,
            zero_point=out_zero_point,
        )

        inputs = [None] * 4
        inputs[0] = input_id
        inputs[1] = weight_id
        inputs[2] = bias_id
        inputs[3] = self.add_immediate_int_scalar(NNAPI_FuseCode.FUSED_NONE)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(node.outputsAt(0), out_oper)

        self.add_operation(NNAPI_OperationCode.FULLY_CONNECTED, inputs, outputs)

    def get_optional_bias(self, jit_bias, weight_tensor, transpose=False):
        ctype, _value = self.get_constant_value(jit_bias)
        if ctype.kind() == "NoneType":
            bias_idx = 1 if transpose else 0
            nnapi_bias_tensor = torch.zeros(
                weight_tensor.size()[bias_idx], dtype=weight_tensor.dtype
            )
            bias_id = self.add_tensor_operand_for_weight(nnapi_bias_tensor)
            bias_oper = self.operands[bias_id]
            return bias_id, bias_oper
        else:
            return self.get_tensor_operand_for_weight(jit_bias)

    def add_conv2d(self, node):
        if node.inputsSize() != 7:
            raise AssertionError(
                f"expected node.inputsSize() == 7, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        (
            jit_image,
            jit_weight,
            jit_bias,
            jit_stride,
            jit_pad,
            jit_dilation,
            jit_groups,
        ) = node.inputs()

        _, weight_tensor = self.get_constant_value(jit_weight, "TensorType")
        bias_id, _bias_oper = self.get_optional_bias(jit_bias, weight_tensor)
        args = self.get_conv_pool_args_2d_from_jit(
            weight_tensor.shape[2:4], jit_stride, jit_pad, jit_dilation, jit_groups
        )

        return self.add_conv2d_common(
            node.outputsAt(0),
            0.0,
            0,
            jit_image,
            weight_tensor,
            bias_id,
            args,
            False,  # transpose
            NNAPI_FuseCode.FUSED_NONE,
        )

    def add_conv_underscore(self, node):
        if node.inputsSize() != 13:
            raise AssertionError(
                f"expected node.inputsSize() == 13, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        (
            jit_image,
            jit_weight,
            jit_bias,
            jit_stride,
            jit_pad,
            jit_dilation,
            jit_transpose,
            _,
            jit_groups,
            _,
            _,
            _,
            _,
        ) = node.inputs()

        _, weight_tensor = self.get_constant_value(jit_weight, "TensorType")
        _, transpose = self.get_constant_value(jit_transpose)
        bias_id, _bias_oper = self.get_optional_bias(jit_bias, weight_tensor, transpose)
        args = self.get_conv_pool_args_2d_from_jit(
            weight_tensor.shape[2:4], jit_stride, jit_pad, jit_dilation, jit_groups
        )

        return self.add_conv2d_common(
            node.outputsAt(0),
            0.0,
            0,
            jit_image,
            weight_tensor,
            bias_id,
            args,
            transpose,
            NNAPI_FuseCode.FUSED_NONE,
        )

    def add_log_softmax(self, node):
        if node.inputsSize() != 3:
            raise AssertionError(
                f"expected node.inputsSize() == 3, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        jit_input, jit_dim, _jit_half_to_float = node.inputs()
        input_id, input_oper = self.get_tensor_operand_by_jitval_fixed_size(jit_input)
        _, dim = self.get_constant_value(jit_dim, "IntType")

        out_shape = input_oper.shape

        inputs = [None] * 3
        inputs[0] = input_id
        # specifying 1 as the scaling factor for the exponent, beta
        inputs[1] = self.add_immediate_float_scalar(1)
        inputs[2] = self.add_immediate_int_scalar(dim)

        outputs = [None] * 1
        outputs[0] = self.add_tensor_operand(
            node.outputsAt(0), input_oper._replace(shape=out_shape)
        )
        self.add_operation(NNAPI_OperationCode.LOG_SOFTMAX, inputs, outputs)

    def add_qconv2d(self, node, fuse_code, transpose=False):
        if node.inputsSize() != 4:
            raise AssertionError(
                f"expected node.inputsSize() == 4, got {node.inputsSize()}"
            )
        if node.outputsSize() != 1:
            raise AssertionError(
                f"expected node.outputsSize() == 1, got {node.outputsSize()}"
            )

        (
            jit_image,
            jit_packed_weight,
            jit_scale,
            jit_zero_point,
        ) = node.inputs()

        _, out_scale = self.get_constant_value(jit_scale, "FloatType")
        _, out_zero_point = self.get_constant_value(jit_zero_point, "IntType")
        weight_ctype, packed_weight = self.get_constant_value(jit_packed_weight)
        if weight_ctype.name() != "Conv2dPackedParamsBase":
            raise AssertionError(
                f"expected weight_ctype Conv2dPackedParamsBase, got {weight_ctype.name()}"
            )
        (
            pack_version,
            tensors,
            opt_tensors,
        ) = packed_weight.__getstate__()[0]
        if pack_version != "2":
            raise AssertionError(f"expected pack_version '2', got {pack_version!r}")
        packed_config, raw_weight = tensors
        (raw_bias,) = opt_tensors
        if raw_bias is None:
            raise AssertionError("raw_bias must not be None")
        args = self.get_conv_pool_args_2d_from_pack(
            raw_weight.shape[2:4], packed_config
        )

        if raw_weight.qscheme() != torch.per_tensor_affine:
            raise AssertionError(
                f"expected raw_weight.qscheme() per_tensor_affine, got {raw_weight.qscheme()}"
            )
        if raw_weight.dtype == torch.quint8:
            unsigned_weight = raw_weight
        else:
            if raw_weight.dtype != torch.qint8:
                raise AssertionError(
                    f"expected raw_weight.dtype qint8, got {raw_weight.dtype}"
                )
            unsigned_weight = torch._make_per_tensor_quantized_tensor(
                (raw_weight.int_repr().int() + 128).to(torch.uint8),
                scale=raw_weight.q_scale(),
                zero_point=raw_weight.q_zero_point() + 128,
            )
        weight_scale = unsigned_weight.q_scale()
        _, image_oper = self.get_tensor_operand_by_jitval(jit_image)
        bias_scale = image_oper.scale * weight_scale
        int_bias = torch.quantize_per_tensor(raw_bias, bias_scale, 0, torch.qint32)
        bias_id = self.add_tensor_operand_for_weight(int_bias)

        multiplier = image_oper.scale * weight_scale / out_scale
        if multiplier <= 0:
            raise AssertionError(f"expected multiplier > 0, got {multiplier}")
        if multiplier >= 1:
            raise Exception(  # noqa: TRY002
                "Quantized convolution multiplier is greater than 1.  "
                "This is supported by NNAPI, but not by most hardware backends.  "
                "Try training a model without quantization-aware training.  "
            )

        return self.add_conv2d_common(
            node.outputsAt(0),
            out_scale,
            out_zero_point,
            jit_image,
            unsigned_weight,
            bias_id,
            args,
            transpose,
            fuse_code,
        )

    def add_conv2d_common(
        self,
        jit_out,
        out_scale,
        out_zero_point,
        jit_image,
        weight_tensor,
        bias_id,
        args,
        transpose,
        fuse_code,
    ):
        image_id, image_oper = self.get_tensor_operand_by_jitval(jit_image)
        in_c = image_oper.shape[1]

        if args.group == 1:
            # Full convolution
            depthwise = False
            if transpose:
                weight_permutation = (1, 2, 3, 0)
            else:
                weight_permutation = (0, 2, 3, 1)
        elif args.group == in_c:
            # Depthwise convolution
            depthwise = True
            weight_permutation = (1, 2, 3, 0)
        else:
            raise Exception("Group convolution not supported yet.")  # noqa: TRY002

        # TODO: Transform at load time to share weights with CPU model.
        nnapi_weight_tensor = weight_tensor.permute(*weight_permutation).contiguous()
        weight_id = self.add_tensor_operand_for_weight(nnapi_weight_tensor)
        weight_oper = self.operands[weight_id]

        bias_oper = self.operands[bias_id]

        if image_oper.op_type == NNAPI_OperandCode.TENSOR_FLOAT32:
            if weight_oper.op_type != NNAPI_OperandCode.TENSOR_FLOAT32:
                raise AssertionError(
                    f"expected weight_oper TENSOR_FLOAT32, got {weight_oper.op_type}"
                )
            if bias_oper.op_type != NNAPI_OperandCode.TENSOR_FLOAT32:
                raise AssertionError(
                    f"expected bias_oper TENSOR_FLOAT32, got {bias_oper.op_type}"
                )
        elif image_oper.op_type == NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
            if weight_oper.op_type != NNAPI_OperandCode.TENSOR_QUANT8_ASYMM:
                raise AssertionError(
                    f"expected weight_oper TENSOR_QUANT8_ASYMM, got {weight_oper.op_type}"
                )
            if bias_oper.op_type != NNAPI_OperandCode.TENSOR_INT32:
                raise AssertionError(
                    f"expected bias_oper TENSOR_INT32, got {bias_oper.op_type}"
                )
            if not approx_equal(image_oper.scale * weight_oper.scale, bias_oper.scale):
                raise AssertionError(
                    f"scale mismatch: image*weight scale {image_oper.scale * weight_oper.scale} != bias scale {bias_oper.scale}"
                )
            if bias_oper.zero_point != 0:
                raise AssertionError(
                    f"expected bias_oper.zero_point == 0, got {bias_oper.zero_point}"
                )
        else:
            raise Exception(  # noqa: TRY002
                f"Unsupported input type for conv2d: {image_oper.op_type}"
            )

        if len(image_oper.shape) != 4:
            raise AssertionError(
                f"expected len(image_oper.shape) == 4, got {len(image_oper.shape)}"
            )
        if len(weight_oper.shape) != 4:
            raise AssertionError(
                f"expected len(weight_oper.shape) == 4, got {len(weight_oper.shape)}"
            )
        if len(bias_oper.shape) != 1:
            raise AssertionError(
                f"expected len(bias_oper.shape) == 1, got {len(bias_oper.shape)}"
            )

        if depthwise:
            # Depthwise convolution
            one, _kern_h, _kern_w, out_c = weight_oper.shape
            if one != 1:
                raise AssertionError(f"expected weight_oper.shape[0] == 1, got {one}")
            if out_c % in_c != 0:
                raise AssertionError(f"out_c {out_c} must be divisible by in_c {in_c}")
            channel_multiplier = out_c // in_c
            if channel_multiplier != 1:
                raise AssertionError(
                    f"channel_multiplier must be 1, got {channel_multiplier}"
                )
            if out_c != in_c:
                raise AssertionError(f"out_c {out_c} != in_c {in_c}")
        else:
            # Full convolution
            out_c, _kern_h, _kern_w, kern_d = weight_oper.shape
            if kern_d != in_c:
                raise AssertionError(f"kern_d {kern_d} != in_c {in_c}")

        if out_c != bias_oper.shape[0]:
            raise AssertionError(
                f"out_c {out_c} != bias_oper.shape[0] {bias_oper.shape[0]}"
            )

        use_nchw = image_oper.use_nchw()

        if depthwise:
            num_args = 12
            opcode = NNAPI_OperationCode.DEPTHWISE_CONV_2D
        else:
            num_args = 11
            if transpose:
                opcode = NNAPI_OperationCode.TRANSPOSE_CONV_2D
            else:
                opcode = NNAPI_OperationCode.CONV_2D

        inputs = [None] * num_args
        inputs[0] = image_id
        inputs[1] = weight_id
        inputs[2] = bias_id
        inputs[3] = self.add_immediate_int_scalar(args.pad_l)
        inputs[4] = self.add_immediate_int_scalar(args.pad_r)
        inputs[5] = self.add_immediate_int_scalar(args.pad_t)
        inputs[6] = self.add_immediate_int_scalar(args.pad_b)
        inputs[7] = self.add_immediate_int_scalar(args.stride_w)
        inputs[8] = self.add_immediate_int_scalar(args.stride_h)
        if depthwise:
            inputs[9] = self.add_immediate_int_scalar(1)
            inputs[10] = self.add_immediate_int_scalar(fuse_code)
            inputs[11] = self.add_immediate_bool_scalar(use_nchw)
        else:
            inputs[9] = self.add_immediate_int_scalar(fuse_code)
            inputs[10] = self.add_immediate_bool_scalar(use_nchw)

        outputs = [None] * 1
        out_shape = get_conv_pool_shape(image_oper.shape, args, out_c, transpose)
        out_oper = image_oper._replace(
            shape=out_shape,
            scale=out_scale,
            zero_point=out_zero_point,
        )
        out_id = self.add_tensor_operand(jit_out, out_oper)
        self._handle_conv_pool_flexible_input(out_id, jit_image, args, transpose)

        outputs[0] = out_id
        self.add_operation(opcode, inputs, outputs)

    def _handle_conv_pool_flexible_input(self, out_id, jit_image, args, transpose):
        image_id, image_oper = self.get_tensor_operand_by_jitval(jit_image)
        batch, in_ch, in_h, in_w = image_oper.shape

        if batch == 0:
            self.forward_operand_shape(out_id, 0, image_id, 0)
        if in_ch == 0:
            raise Exception("Input channels can't be flexible")  # noqa: TRY002
        # H & W
        if transpose:
            if in_h == 0:
                self.compute_operand_shape(
                    out_id,
                    2,
                    f"({flex_name(image_id, 2)} - 1) * {args.stride_h} + {args.kernel_h} - {args.pad_t} - {args.pad_b}",
                )
            if in_w == 0:
                self.compute_operand_shape(
                    out_id,
                    3,
                    f"({flex_name(image_id, 3)} - 1) * {args.stride_w} + {args.kernel_w} - {args.pad_l} - {args.pad_r}",
                )
        else:
            if in_h == 0:
                self.compute_operand_shape(
                    out_id,
                    2,
                    f"({flex_name(image_id, 2)} - {args.kernel_h} + {args.pad_t} + {args.pad_b}) // {args.stride_h} + 1",
                )
            if in_w == 0:
                self.compute_operand_shape(
                    out_id,
                    3,
                    f"({flex_name(image_id, 3)} - {args.kernel_w} + {args.pad_l} + {args.pad_r}) // {args.stride_w} + 1",
                )


def serialize_model(
    module, inputs, *, config=None, return_shapes=None, use_int16_for_qint16=False
):
    """Convert to NNAPI and serialize torchscript module.

    Parameters:
        module: Torchscript module to convert
        inputs: Tensors used to specify input details for NNAPI
        config (optional): Optional config to attach to module
        return_shapes (optional): Specify shape of outputs if
            your module uses runtime flexible shapes to set output
            buffer size for NNAPI
        use_int16_for_qint16 (optional): Use Pytorch int16 to represent NNAPI qint16 values
    """
    return _NnapiSerializer(config, use_int16_for_qint16).serialize_model(
        module, inputs, return_shapes
    )
<|endoftext|># mypy: allow-untyped-defs
import sys
from contextlib import contextmanager

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


def set_flags(
    _immediate=None,
):
    orig_flags = (torch._C._get_miopen_immediate(),)
    if _immediate is not None:
        torch._C._set_miopen_immediate(_immediate)
    return orig_flags


@contextmanager
def flags(
    immediate=False,
):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(
            immediate,
        )
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.<miopen|mkldnn>.immediate = True


class MiopenModule(PropModule):
    immediate = ContextProp(
        torch._C._get_miopen_immediate, torch._C._set_miopen_immediate
    )


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = MiopenModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
immediate: bool
<|endoftext|>from typing import Optional

import torch


__all__ = [
    "version",
    "is_available",
    "get_max_alg_id",
]

try:
    from torch._C import _cusparselt
except ImportError:
    _cusparselt = None  # type: ignore[assignment]

__cusparselt_version: int | None = None
__MAX_ALG_ID: int | None = None

if _cusparselt is not None:

    def _init() -> bool:
        global __cusparselt_version
        global __MAX_ALG_ID
        if __cusparselt_version is None:
            # pyrefly: ignore [missing-attribute]
            __cusparselt_version = _cusparselt.getVersionInt()
            if __cusparselt_version == 400:
                __MAX_ALG_ID = 4
            elif __cusparselt_version == 502:
                __MAX_ALG_ID = 5
            elif __cusparselt_version == 602:
                __MAX_ALG_ID = 37
        return True

else:

    def _init() -> bool:
        return False


def version() -> int | None:
    """Return the version of cuSPARSELt"""
    if not _init():
        return None
    return __cusparselt_version


def is_available() -> bool:
    r"""Return a bool indicating if cuSPARSELt is currently available."""
    return torch._C._has_cusparselt


def get_max_alg_id() -> int | None:
    if not _init():
        return None
    return __MAX_ALG_ID
<|endoftext|># mypy: allow-untyped-defs
"""
This is a script for launching PyTorch inference on Intel(R) Xeon(R) Scalable Processors with optimal configurations.

Single instance inference, multi-instance inference are enabled.

Note: term "instance" here doesn't refer to a cloud instance. This script is executed as a single process. It invokes
multiple "instances" which are formed from multiple threads for each. "instance" is kind of group of threads in this
context.

Illustrated as below:

::

    +-----------------------------+----------------------+-------+
    |            process          |        thread        | core  |
    +=============================+======================+=======+
    | torch.backends.xeon.run_cpu | instance 0: thread 0 |   0   |
    |                             |             thread 1 |   1   |
    |                             +----------------------+-------+
    |                             | instance 1: thread 0 |   2   |
    |                             |             thread 1 |   3   |
    |                             +----------------------+-------+
    |                             | ...                  |  ...  |
    |                             +----------------------+-------+
    |                             | instance N: thread 0 |   M   |
    |                             |             thread 1 |  M+1  |
    +-----------------------------+----------------------+-------+

To get the peak performance on Intel(R) Xeon(R) Scalable Processors, the script optimizes the configuration of thread and memory
management. For thread management, the script configures thread affinity and the preload of Intel OMP library.
For memory management, it configures NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc).

Environment variables that will be set by this script:

+------------------+-------------------------------------------------------------------------------------------------+
| Environ Variable |                                             Value                                               |
+==================+=================================================================================================+
|    LD_PRELOAD    | Depending on knobs you set, <lib>/libiomp5.so, <lib>/libjemalloc.so, <lib>/libtcmalloc.so might |
|                  | be appended to LD_PRELOAD.                                                                      |
+------------------+-------------------------------------------------------------------------------------------------+
|   KMP_AFFINITY   | If libiomp5.so is preloaded, KMP_AFFINITY could be set to "granularity=fine,compact,1,0".       |
+------------------+-------------------------------------------------------------------------------------------------+
|   KMP_BLOCKTIME  | If libiomp5.so is preloaded, KMP_BLOCKTIME is set to "1".                                       |
+------------------+-------------------------------------------------------------------------------------------------+
|  OMP_NUM_THREADS | value of ncores_per_instance                                                                    |
+------------------+-------------------------------------------------------------------------------------------------+
|    MALLOC_CONF   | If libjemalloc.so is preloaded, MALLOC_CONF will be set to                                      |
|                  | "oversize_threshold:1,background_thread:true,metadata_thp:auto".                                |
+------------------+-------------------------------------------------------------------------------------------------+

*Note*: This script respects environment variables set preliminarily. I.e. If you set the environment variables
mentioned above before running the script, the script will not overwrite the values in the script.

How to use this module:
~~~~~~~~~~~~~~~~~~~~~~~

Single instance inference
-------------------------

1. Run single-instance inference on a single node with all CPU nodes.

::

   python -m torch.backends.xeon.run_cpu --throughput-mode script.py args

2. Run single-instance inference on a single CPU node.

::

   python -m torch.backends.xeon.run_cpu --node-id 1 script.py args

Multi-instance inference
------------------------

1. Multi-instance
   By default this tool runs one process per node. If you want to set the instance numbers and core per instance,
   --ninstances and  --ncores-per-instance should be set.

::

   python -m torch.backends.xeon.run_cpu -- python_script args

   eg: on an Intel(R) Xeon(R) Scalable Processor with 14 instance, 4 cores per instance

::

   python -m torch.backends.xeon.run_cpu --ninstances 14 --ncores-per-instance 4 python_script args

2. Run single-instance inference among multiple instances.
   By default, runs all ninstances. If you want to independently run a single instance among ninstances, specify rank.

   eg: run 0th instance on an Intel(R) Xeon(R) Scalable Processor with 2 instance (i.e., numactl -C 0-27)

::

   python -m torch.backends.xeon.run_cpu --ninstances 2 --rank 0 python_script args

   eg: run 1st instance on an Intel(R) Xeon(R) Scalable Processor with 2 instance (i.e., numactl -C 28-55)

::

   python -m torch.backends.xeon.run_cpu --ninstances 2 --rank 1 python_script args

   eg: run 0th instance on an Intel(R) Xeon(R) Scalable Processor with 2 instance, 2 cores per instance,
   first four cores (i.e., numactl -C 0-1)

::

   python -m torch.backends.xeon.run_cpu --core-list "0, 1, 2, 3" --ninstances 2 --ncores-per-instance 2
   --rank 0 python_script args

3. To look up what optional arguments this module offers:

::

    python -m torch.backends.xeon.run_cpu --help

Memory allocator
----------------

"--enable-tcmalloc" and "--enable-jemalloc" can be used to enable different memory allocator.

"""

import glob
import logging
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
from os.path import expanduser

from torch.distributed.elastic.multiprocessing import (
    DefaultLogsSpecs as _DefaultLogsSpecs,
    start_processes,
    Std,
)


format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=format_str)
logger = logging.getLogger(__name__)


class _CPUinfo:
    """Get CPU information, such as cores list and NUMA information."""

    def __init__(self, test_input=""):
        self.cpuinfo = []
        if platform.system() in ["Windows", "Darwin"]:
            raise RuntimeError(f"{platform.system()} is not supported!!!")
        elif platform.system() == "Linux":
            # Sample output of: `lscpu --parse=CPU,Core,Socket,Node`
            #
            # # The following is the parsable format, which can be fed to other
            # # programs. Each different item in every column has a unique ID
            # # starting from zero.
            # # CPU,Core,Socket,Node
            # 0,0,0,0
            # 1,1,0,0
            # ...
            if test_input == "":
                lscpu_cmd = ["lscpu", "--parse=CPU,Core,Socket,Node"]
                lscpu_info = subprocess.check_output(
                    lscpu_cmd, universal_newlines=True
                ).split("\n")
            else:
                lscpu_info = test_input.split("\n")

            # Get information about  cpu, core, socket and node
            for line in lscpu_info:
                pattern = r"^([\d]+,[\d]+,[\d]+,[\d]?)"
                regex_out = re.search(pattern, line)
                if regex_out:
                    self.cpuinfo.append(regex_out.group(1).strip().split(","))

            # physical cores := core column in lscpu output
            #  logical cores :=  cPU column in lscpu output
            self.node_nums = int(max(line[3] for line in self.cpuinfo)) + 1
            self.node_physical_cores: list[list[int]] = []  # node_id is index
            self.node_logical_cores: list[list[int]] = []  # node_id is index
            self.physical_core_node_map = {}  # physical core to numa node id
            self.logical_core_node_map = {}  # logical core to numa node id

            for node_id in range(self.node_nums):
                cur_node_physical_core = []
                cur_node_logical_core = []
                for cpuinfo in self.cpuinfo:
                    nid = cpuinfo[3] if cpuinfo[3] != "" else "0"
                    if node_id == int(nid):
                        if int(cpuinfo[1]) not in cur_node_physical_core:
                            cur_node_physical_core.append(int(cpuinfo[1]))
                            self.physical_core_node_map[int(cpuinfo[1])] = int(node_id)
                        cur_node_logical_core.append(int(cpuinfo[0]))
                        self.logical_core_node_map[int(cpuinfo[0])] = int(node_id)
                self.node_physical_cores.append(cur_node_physical_core)
                self.node_logical_cores.append(cur_node_logical_core)

    def _physical_core_nums(self):
        overall_core_num = 0
        for phycisal_core_list in self.node_physical_cores:
            overall_core_num += len(phycisal_core_list)
        return overall_core_num

    def _logical_core_nums(self):
        overall_core_num = 0
        for logical_core_list in self.node_logical_cores:
            overall_core_num += len(logical_core_list)
        return overall_core_num

    def get_node_physical_cores(self, node_id):
        if node_id < 0 or node_id > self.node_nums - 1:
            raise ValueError(
                f"Invalid node id: {node_id}. Valid node ids: {list(range(len(self.node_physical_cores)))}"
            )
        return self.node_physical_cores[node_id]

    def get_node_logical_cores(self, node_id):
        if node_id < 0 or node_id > self.node_nums - 1:
            raise ValueError(
                f"Invalid node id: {node_id}. Valid node ids: {list(range(len(self.node_physical_cores)))}"
            )
        return self.node_logical_cores[node_id]

    def get_all_physical_cores(self):
        all_cores = []
        for cores in self.node_physical_cores:
            all_cores.extend(cores)
        return all_cores

    def get_all_logical_cores(self):
        all_cores = []
        for cores in self.node_logical_cores:
            all_cores.extend(cores)
        return all_cores

    def numa_aware_check(self, core_list):
        """
        Check whether all cores in core_list are in the same NUMA node.

        Cross NUMA will reduce performance.
        We strongly advise to not use cores on different nodes.
        """
        cores_numa_map = self.logical_core_node_map
        numa_ids = []
        for core in core_list:
            numa_id = cores_numa_map[core]
            if numa_id not in numa_ids:
                numa_ids.append(numa_id)
        if len(numa_ids) > 1:
            logger.warning(
                "Numa Aware: cores:%s on different NUMA nodes:%s. To avoid \
this behavior, please use --ncores-per-instance knob to make sure number of cores is divisible by --ncores-per-\
instance. Alternatively, please use --skip-cross-node-cores knob.",
                core_list,
                numa_ids,
            )
        if len(numa_ids) == 0:
            raise RuntimeError(
                "invalid number of NUMA nodes; please make sure numa_ids >= 1"
            )
        return numa_ids


class _Launcher:
    r"""Class for launcher."""

    msg_lib_notfound = (
        f"Unable to find the {{0}} library file lib{{1}}.so in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib \
or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or \
{expanduser('~')}/.local/lib/ so the LD_PRELOAD environment variable will not be set."
    )

    def __init__(self) -> None:
        self.cpuinfo = _CPUinfo()

    def add_lib_preload(self, lib_type):
        """Enable TCMalloc/JeMalloc/intel OpenMP."""
        library_paths = []
        if "CONDA_PREFIX" in os.environ:
            library_paths.append(f"{os.environ['CONDA_PREFIX']}/lib")
        if "VIRTUAL_ENV" in os.environ:
            library_paths.append(f"{os.environ['VIRTUAL_ENV']}/lib")

        library_paths += [
            f"{expanduser('~')}/.local/lib",
            "/usr/local/lib",
            "/usr/local/lib64",
            "/usr/lib",
            "/usr/lib64",
        ]

        lib_find = False
        lib_set = False
        for item in os.getenv("LD_PRELOAD", "").split(":"):
            if item.endswith(f"lib{lib_type}.so"):
                lib_set = True
                break
        if not lib_set:
            for lib_path in library_paths:
                # pyrefly: ignore [unbound-name]
                library_file = os.path.join(lib_path, f"lib{lib_type}.so")
                matches = glob.glob(library_file)
                if len(matches) > 0:
                    # pyrefly: ignore [unbound-name]
                    ld_preloads = [f"{matches[0]}", os.getenv("LD_PRELOAD", "")]
                    # pyrefly: ignore [unbound-name]
                    os.environ["LD_PRELOAD"] = os.pathsep.join(
                        # pyrefly: ignore [unbound-name]
                        [p.strip(os.pathsep) for p in ld_preloads if p]
                    )
                    lib_find = True
                    break
        return lib_set or lib_find

    def is_numactl_available(self):
        numactl_available = False
        try:
            cmd = ["numactl", "-C", "0", "-m", "0", "hostname"]
            r = subprocess.run(
                cmd,
                env=os.environ,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if r.returncode == 0:
                numactl_available = True
        except Exception:
            pass
        return numactl_available

    def set_memory_allocator(
        self, enable_tcmalloc=True, enable_jemalloc=False, use_default_allocator=False
    ):
        """
        Enable TCMalloc/JeMalloc with LD_PRELOAD and set configuration for JeMalloc.

        By default, PTMalloc will be used for PyTorch, but TCMalloc and JeMalloc can get better
        memory reuse and reduce page fault to improve performance.
        """
        if enable_tcmalloc and enable_jemalloc:
            raise RuntimeError(
                "Unable to enable TCMalloc and JEMalloc at the same time."
            )

        if enable_tcmalloc:
            find_tc = self.add_lib_preload(lib_type="tcmalloc")
            if not find_tc:
                msg = f'{self.msg_lib_notfound} you can use "conda install -c conda-forge gperftools" to install {{0}}'
                logger.warning(msg.format("TCmalloc", "tcmalloc"))
            else:
                logger.info("Use TCMalloc memory allocator")

        elif enable_jemalloc:
            find_je = self.add_lib_preload(lib_type="jemalloc")
            if not find_je:
                msg = f'{self.msg_lib_notfound} you can use "conda install -c conda-forge jemalloc" to install {{0}}'
                logger.warning(msg.format("Jemalloc", "jemalloc"))
            else:
                logger.info("Use JeMalloc memory allocator")
                self.set_env(
                    "MALLOC_CONF",
                    "oversize_threshold:1,background_thread:true,metadata_thp:auto",
                )

        elif use_default_allocator:
            pass

        else:
            find_tc = self.add_lib_preload(lib_type="tcmalloc")
            if find_tc:
                logger.info("Use TCMalloc memory allocator")
                return
            find_je = self.add_lib_preload(lib_type="jemalloc")
            if find_je:
                logger.info("Use JeMalloc memory allocator")
                return
            logger.warning(
                """Neither TCMalloc nor JeMalloc is found in $CONDA_PREFIX/lib or $VIRTUAL_ENV/lib
                            or /.local/lib/ or /usr/local/lib/ or /usr/local/lib64/ or /usr/lib or /usr/lib64 or
                           %s/.local/lib/ so the LD_PRELOAD environment variable will not be set.
                           This may drop the performance""",
                expanduser("~"),
            )

    def log_env_var(self, env_var_name=""):
        if env_var_name in os.environ:
            logger.info("%s=%s", env_var_name, os.environ[env_var_name])

    def set_env(self, env_name, env_value):
        if not env_value:
            logger.warning("%s is None", env_name)
        if env_name not in os.environ:
            os.environ[env_name] = env_value
        elif os.environ[env_name] != env_value:
            logger.warning(
                "Overriding value with the one set in environment variable: %s. \
Value applied: %s. Value ignored: %s",
                env_name,
                os.environ[env_name],
                env_value,
            )
        self.log_env_var(env_name)

    # set_kmp_affinity is used to control whether to set KMP_AFFINITY or not.
    # In scenario that use all cores on all nodes, including logical cores, setting KMP_AFFINITY disables logical cores.
    # In this case, KMP_AFFINITY should not be set.
    def set_multi_thread_and_allocator(
        self,
        disable_iomp=False,
        set_kmp_affinity=True,
        enable_tcmalloc=True,
        enable_jemalloc=False,
        use_default_allocator=False,
    ):
        """
        Set multi-thread configuration and enable Intel openMP and TCMalloc/JeMalloc.

        By default, GNU openMP and PTMalloc are used in PyTorch. but Intel openMP and TCMalloc/JeMalloc are better alternatives
        to get performance benefit.
        """
        self.set_memory_allocator(
            enable_tcmalloc, enable_jemalloc, use_default_allocator
        )
        if not disable_iomp:
            find_iomp = self.add_lib_preload(lib_type="iomp5")
            if not find_iomp:
                msg = f'{self.msg_lib_notfound} you can use "conda install mkl" to install {{0}}'
                logger.warning(msg.format("iomp", "iomp5"))
            else:
                logger.info("Using Intel OpenMP")
                if set_kmp_affinity:
                    self.set_env("KMP_AFFINITY", "granularity=fine,compact,1,0")
                self.set_env("KMP_BLOCKTIME", "1")
        self.log_env_var("LD_PRELOAD")

    r"""
     Launcher for single instance and multi-instance
     """

    def launch(self, args):
        cores = []
        set_kmp_affinity = True
        enable_taskset = False
        # ncores_per_instance param check and expand for multi-instance
        if not isinstance(args.ncores_per_instance, list):
            args.ncores_per_instance = [args.ncores_per_instance]
        args.ncores_per_instance = [int(x) for x in args.ncores_per_instance]
        if args.ncores_per_instance != [-1]:
            if args.ninstances > 0:
                if len(args.ncores_per_instance) == 1:
                    args.ncores_per_instance = (
                        args.ncores_per_instance * args.ninstances
                    )
                if len(args.ncores_per_instance) != args.ninstances:
                    raise ValueError(
                        'Mismatch for "--ninstances" and "--ncores-per-instance" settings.'
                    )
            elif len(args.ncores_per_instance) > 1:
                args.ninstances = len(args.ncores_per_instance)
        if args.core_list:  # user specify what cores will be used by params
            cores = [int(x) for x in args.core_list.split(",")]
            if args.ncores_per_instance == [-1]:
                raise RuntimeError(
                    'please specify the "--ncores-per-instance" if you have pass the --core-list params'
                )
            if args.ninstances > 0 and sum(args.ncores_per_instance) < len(cores):
                logger.warning(
                    "only first %s cores will be used, \
but you specify %s cores in core_list",
                    sum(args.ncores_per_instance),
                    len(cores),
                )
        else:
            if args.use_logical_core:
                if args.node_id != -1:
                    cores = self.cpuinfo.get_node_logical_cores(args.node_id)
                else:
                    cores = self.cpuinfo.get_all_logical_cores()
                    # When using all cores on all nodes, including logical cores,
                    # setting KMP_AFFINITY disables logical cores. Thus, KMP_AFFINITY should not be set.
                    set_kmp_affinity = False
            else:
                if args.node_id != -1:
                    cores = self.cpuinfo.get_node_physical_cores(args.node_id)
                else:
                    cores = self.cpuinfo.get_all_physical_cores()
            if (
                not args.multi_instance
                and args.ninstances == -1
                and args.ncores_per_instance == [-1]
            ):
                args.ninstances = 1
                args.ncores_per_instance = [len(cores)]
            elif (
                args.multi_instance
                and args.ninstances == -1
                and args.ncores_per_instance == [-1]
            ):
                args.throughput_mode = True
            elif args.ncores_per_instance == [-1] and args.ninstances != -1:
                if args.ninstances > len(cores):
                    raise RuntimeError(
                        f"there are {len(cores)} total cores but you specify {args.ninstances} ninstances; \
please make sure ninstances <= total_cores)"
                    )
                if args.skip_cross_node_cores:
                    logger.warning(
                        "with --skip-cross-node-cores set, the first %d sub-numa node(s) will be used \
to launch the instance(s), each node for 1 instance.",
                        args.ninstances,
                    )
                    if args.ninstances > self.cpuinfo.node_nums:
                        raise ValueError(
                            "--ninstances should not be larger than numa node number \
when --skip-cross-node-cores is set"
                        )
                    cores.clear()
                    args.ncores_per_instance.clear()
                    for i in range(args.ninstances):
                        per_node_cores = self.cpuinfo.get_node_physical_cores(i)
                        cores.extend(per_node_cores)
                        args.ncores_per_instance.append(len(per_node_cores))
                else:
                    logger.warning(
                        "all the CPU cores on the machine will be distributed \
as evenly as possible to each instance. It is possible that some cores in an instance may cross the numa node."
                    )
                    args.ncores_per_instance = [
                        len(cores) // args.ninstances
                    ] * args.ninstances
            elif args.ncores_per_instance != [-1]:
                if args.skip_cross_node_cores:
                    utilized_node_cores = (
                        self.cpuinfo.node_logical_cores
                        if args.use_logical_core
                        else self.cpuinfo.node_physical_cores
                    )
                    ncore_per_node = [len(c) for c in utilized_node_cores]
                    if len(args.ncores_per_instance) > len(ncore_per_node):
                        # too many ncores_per_instance to skip cross-node cores
                        raise ValueError(
                            "there are %s nodes, but %s ncores-per-instance elements and \
--skip-cross-node-cores is specified.",
                            len(ncore_per_node),
                            len(args.ncores_per_instance),
                        )
                    leftover_cores = list()
                    for i, core_num in enumerate(args.ncores_per_instance):
                        leftover_num = core_num - ncore_per_node[i]
                        if leftover_num > 0:
                            # too many ncores_per_instance for a node
                            raise ValueError(
                                "there are %s core(s) in node %s, but specified %s cores for \
this node and --skip-cross-node-cores is specified.",
                                ncore_per_node[i],
                                i,
                                core_num,
                            )
                        elif leftover_num < 0:
                            # Exclude excessive cores from largest core IDs
                            leftover_cores.extend(utilized_node_cores[i][leftover_num:])
                    # used node num < total node num, mark the rest node cores as unused
                    for j in range(i + 1, len(ncore_per_node)):
                        leftover_cores.extend(utilized_node_cores[j][:])
                    for core_rm in leftover_cores:
                        if core_rm in cores:
                            cores.remove(core_rm)
                args.ninstances = len(args.ncores_per_instance)
            else:
                if sum(args.ncores_per_instance) > len(cores):
                    raise RuntimeError(
                        "Please make sure the sum up of ncores_per_instance <= total_cores"
                    )
            if args.latency_mode:
                logger.warning(
                    "--latency-mode is exclusive to --ninstances, --ncores-per-instance, --node-id and \
--use-logical-core. They won't take effect even they are set explicitly."
                )
                core_num_per_instance = 4
                cores = list()
                args.ninstances = 0
                if args.skip_cross_node_cores:
                    # Per-numa core allocation, the remainder cores are not used
                    for node_id, node_core_list in enumerate(
                        self.cpuinfo.node_physical_cores
                    ):
                        node_core_num = len(node_core_list)
                        for i in range(
                            core_num_per_instance,
                            node_core_num + 1,
                            core_num_per_instance,
                        ):
                            args.ninstances += 1
                            cores.extend(node_core_list[i - core_num_per_instance : i])
                        if i < node_core_num:
                            logger.warning(
                                "With --skip-cross-node-cores set, not all the cores on node %d is used \
because the core number on this node is not dividable by %d.",
                                node_id,
                                core_num_per_instance,
                            )
                    args.ncores_per_instance = [core_num_per_instance] * args.ninstances
                else:
                    # Compact allocation of all physical cores, regardless of numa
                    cores = self.cpuinfo.get_all_physical_cores()
                    args.ninstances = len(cores) // core_num_per_instance
                    args.ncores_per_instance = [core_num_per_instance] * args.ninstances
                    remainder = len(cores) % core_num_per_instance
                    if remainder > 0:
                        args.ninstances += 1
                        args.ncores_per_instance.append(remainder)
            if args.throughput_mode:
                logger.warning(
                    "--throughput-mode is exclusive to --ninstances, --ncores-per-instance, --node-id and \
--use-logical-core. They won't take effect even they are set explicitly."
                )
                args.ninstances = self.cpuinfo.node_nums
                cores = self.cpuinfo.get_all_physical_cores()
                args.ncores_per_instance.clear()
                for node_core_list in self.cpuinfo.node_physical_cores:
                    args.ncores_per_instance.append(len(node_core_list))

        if args.ninstances > 1 and args.rank != -1:
            logger.info(
                "assigning %s cores for instance %s",
                args.ncores_per_instance[args.rank],
                args.rank,
            )

        if not args.disable_numactl:
            numactl_available = self.is_numactl_available()
            if not numactl_available:
                if not args.disable_taskset:
                    logger.warning(
                        "Core binding with numactl is not available. Disabling numactl and using taskset instead. \
                    This may affect performance in multi-socket system; please use numactl if memory binding is needed."
                    )
                    args.disable_numactl = True
                    enable_taskset = True
                else:
                    logger.warning(
                        "Core binding with numactl is not available, and --disable_taskset is set. \
                    Please unset --disable_taskset to use taskset instead of numactl."
                    )
                    sys.exit(-1)

        if not args.disable_taskset:
            enable_taskset = True

        self.set_multi_thread_and_allocator(
            args.disable_iomp,
            set_kmp_affinity,
            args.enable_tcmalloc,
            args.enable_jemalloc,
            args.use_default_allocator,
        )
        entrypoint = ""
        launch_args = {}
        launch_envs: dict[int, dict] = {}
        launch_tee = {}
        # check whether is launched from torchrun with --nproc-per-node <num workers>
        local_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        cores = sorted(cores)
        for i in range(args.ninstances):
            cmd = []
            cur_process_cores = ""
            if not args.disable_numactl or enable_taskset:
                if not args.disable_numactl:
                    cmd = ["numactl"]
                elif enable_taskset:
                    cmd = ["taskset"]
                core_index_start = 0
                if (
                    args.rank == -1
                ):  # sequentially assign ncores_per_instance to ninstances
                    for j in range(i):
                        core_index_start += args.ncores_per_instance[j]
                    core_list = cores[
                        core_index_start : core_index_start
                        + args.ncores_per_instance[i]
                    ]
                else:  # assign ncores_per_instance from rank
                    for j in range(args.rank):
                        core_index_start += args.ncores_per_instance[j]
                    core_list = cores[
                        core_index_start : core_index_start
                        + args.ncores_per_instance[args.rank]
                    ]

                core_ranges: list[dict] = []
                if local_size > 1:
                    total_num_cores = len(core_list)
                    cores_per_rank = total_num_cores // local_size
                    if cores_per_rank < 1:
                        raise AssertionError(
                            f"At least one core needs to be assigned to each rank, "
                            f"got {total_num_cores} cores for {local_size} ranks"
                        )
                    core_list = core_list[
                        cores_per_rank * local_rank : cores_per_rank * (local_rank + 1)
                    ]
                for core in core_list:
                    if len(core_ranges) == 0:
                        range_elem = {"start": core, "end": core}
                        core_ranges.append(range_elem)
                    else:
                        if core - core_ranges[-1]["end"] == 1:
                            core_ranges[-1]["end"] = core
                        else:
                            range_elem = {"start": core, "end": core}
                            core_ranges.append(range_elem)
                for r in core_ranges:
                    cur_process_cores = f"{cur_process_cores}{r['start']}-{r['end']},"
                cur_process_cores = cur_process_cores[:-1]
                if not args.disable_numactl:
                    numa_params = f"-C {cur_process_cores} "
                    numa_ids = ",".join(
                        [
                            str(numa_id)
                            for numa_id in self.cpuinfo.numa_aware_check(core_list)
                        ]
                    )
                    numa_params += f"-m {numa_ids}"
                    cmd.extend(numa_params.split())
                elif enable_taskset:
                    taskset_params = f"-c {cur_process_cores} "
                    cmd.extend(taskset_params.split())
            with_python = not args.no_python
            if with_python:
                cmd.append(sys.executable)
                cmd.append("-u")
            if args.module:
                cmd.append("-m")
            cmd.append(args.program)
            cmd.extend(args.program_args)
            cmd_s = " ".join(cmd)
            logger.info(cmd_s)
            if entrypoint == "":
                entrypoint = cmd[0]
            del cmd[0]
            launch_args[i] = tuple(cmd)
            # Use user defined OMP_NUM_THREADS if set explicitly in prior
            # Else set it as ncores-per-instance of the current instance
            predef_omp_thr_num = os.environ.get("OMP_NUM_THREADS")
            launch_envs[i] = (
                {"OMP_NUM_THREADS": str(args.ncores_per_instance[i])}
                if predef_omp_thr_num is None
                else {}
            )
            launch_tee[i] = Std.ALL

            if args.rank != -1:  # launches single instance, rank, only
                break

        ctx = start_processes(
            name=args.log_file_prefix,
            entrypoint=entrypoint,
            args=launch_args,
            envs=launch_envs,
            logs_specs=_DefaultLogsSpecs(log_dir=args.log_path, tee=launch_tee),
        )
        ctx.wait()


def _add_memory_allocator_params(parser):
    group = parser.add_argument_group("Memory Allocator Parameters")
    # allocator control
    group.add_argument(
        "--enable-tcmalloc",
        "--enable_tcmalloc",
        action="store_true",
        default=False,
        help="Enable tcmalloc allocator",
    )
    group.add_argument(
        "--enable-jemalloc",
        "--enable_jemalloc",
        action="store_true",
        default=False,
        help="Enable jemalloc allocator",
    )
    group.add_argument(
        "--use-default-allocator",
        "--use_default_allocator",
        action="store_true",
        default=False,
        help="Use default memory allocator",
    )


def _add_multi_instance_params(parser):
    group = parser.add_argument_group("Multi-instance Parameters")
    # multi-instance control
    group.add_argument(
        "--ncores-per-instance",
        "--ncores_per_instance",
        metavar="\b",
        default=-1,
        type=int,
        nargs="+",
        help="Cores per instance",
    )
    group.add_argument(
        "--ninstances",
        metavar="\b",
        default=-1,
        type=int,
        help="For multi-instance, you should give the cores number you used for per instance.",
    )
    group.add_argument(
        "--skip-cross-node-cores",
        "--skip_cross_node_cores",
        action="store_true",
        default=False,
        help="If specified --ncores-per-instance, skips cross-node cores.",
    )
    group.add_argument(
        "--rank",
        metavar="\b",
        default="-1",
        type=int,
        help="Specify instance index to assign ncores_per_instance for rank; \
otherwise ncores_per_instance will be assigned sequentially to ninstances. Please refer to \
https://github.com/intel/intel-extension-for-pytorch/blob/master/docs/tutorials/performance_tuning/launch_script.md",
    )
    group.add_argument(
        "--latency-mode",
        "--latency_mode",
        action="store_true",
        default=False,
        help="By default 4 core per instance and use all physical cores",
    )
    group.add_argument(
        "--throughput-mode",
        "--throughput_mode",
        action="store_true",
        default=False,
        help="By default one instance per node and use all physical cores",
    )
    group.add_argument(
        "--node-id",
        "--node_id",
        metavar="\b",
        default=-1,
        type=int,
        help="node id for multi-instance, by default all nodes will be used",
    )
    group.add_argument(
        "--use-logical-core",
        "--use_logical_core",
        action="store_true",
        default=False,
        help="Whether only use physical cores",
    )
    group.add_argument(
        "--disable-numactl",
        "--disable_numactl",
        action="store_true",
        default=False,
        help="Disable numactl",
    )
    group.add_argument(
        "--disable-taskset",
        "--disable_taskset",
        action="store_true",
        default=False,
        help="Disable taskset",
    )
    group.add_argument(
        "--core-list",
        "--core_list",
        metavar="\b",
        default=None,
        type=str,
        help='Specify the core list as "core_id, core_id, ....", otherwise, all the cores will be used.',
    )
    group.add_argument(
        "--log-path",
        "--log_path",
        metavar="\b",
        default="",
        type=str,
        help="The log file directory. Default path is "
        ", which means disable logging to files.",
    )
    group.add_argument(
        "--log-file-prefix",
        "--log_file_prefix",
        metavar="\b",
        default="run",
        type=str,
        help="log file prefix",
    )


def _add_kmp_iomp_params(parser):
    group = parser.add_argument_group("IOMP Parameters")
    group.add_argument(
        "--disable-iomp",
        "--disable_iomp",
        action="store_true",
        default=False,
        help="By default, we use Intel OpenMP and libiomp5.so will be add to LD_PRELOAD",
    )


def create_args(parser=None):
    """
    Parse the command line options.

    @retval ArgumentParser
    """
    # pyrefly: ignore [missing-attribute]
    parser.add_argument(
        "--multi-instance",
        "--multi_instance",
        action="store_true",
        default=False,
        help="Enable multi-instance, by default one instance per node",
    )

    # pyrefly: ignore [missing-attribute]
    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script "
        "as a python module, executing with the same behavior as"
        '"python -m".',
    )

    # pyrefly: ignore [missing-attribute]
    parser.add_argument(
        "--no-python",
        "--no_python",
        default=False,
        action="store_true",
        help='Do not prepend the --program script with "python" - just exec '
        "it directly. Useful when the script is not a Python script.",
    )

    _add_memory_allocator_params(parser)
    _add_kmp_iomp_params(parser)

    _add_multi_instance_params(parser)
    # positional
    # pyrefly: ignore [missing-attribute]
    parser.add_argument(
        "program",
        type=str,
        help="The full path to the program/script to be launched. "
        "followed by all the arguments for the script",
    )

    # rest from the training program
    # pyrefly: ignore [missing-attribute]
    parser.add_argument("program_args", nargs=REMAINDER)


def main(args):
    env_before = set(os.environ.keys())
    if platform.system() in ["Windows", "Darwin"]:
        raise RuntimeError(f"{platform.system()} is not supported!!!")

    if args.log_path:
        os.makedirs(args.log_path, exist_ok=True)
    else:
        args.log_path = os.devnull

    if args.latency_mode and args.throughput_mode:
        raise RuntimeError(
            "Either args.latency_mode or args.throughput_mode should be set"
        )

    if not args.no_python and not args.program.endswith(".py"):
        raise RuntimeError(
            'For non Python script, you should use "--no-python" parameter.'
        )

    # Verify LD_PRELOAD
    if "LD_PRELOAD" in os.environ:
        lst_valid = []
        tmp_ldpreload = os.environ["LD_PRELOAD"]
        for item in tmp_ldpreload.split(":"):
            matches = glob.glob(item)
            if len(matches) > 0:
                lst_valid.append(item)
            else:
                logger.warning("%s doesn't exist. Removing it from LD_PRELOAD.", item)
        if len(lst_valid) > 0:
            os.environ["LD_PRELOAD"] = ":".join(lst_valid)
        else:
            os.environ["LD_PRELOAD"] = ""

    launcher = _Launcher()
    launcher.launch(args)
    for x in sorted(set(os.environ.keys()) - env_before):
        logger.debug("%s=%s", x, os.environ[x])


if __name__ == "__main__":
    parser = ArgumentParser(
        description="This is a script for launching PyTorch inference on Intel(R) Xeon(R) Scalable "
        "Processors with optimal configurations. Single instance inference, "
        "multi-instance inference are enable. To get the peak performance on Intel(R) "
        "Xeon(R) Scalable Processors, the script optimizes the configuration "
        "of thread and memory management. For thread management, the script configures thread "
        "affinity and the preload of Intel OMP library. For memory management, it configures "
        "NUMA binding and preload optimized memory allocation library (e.g. tcmalloc, jemalloc) "
        "\n################################# Basic usage ############################# \n"
        "\n 1. single instance\n"
        "\n   >>> python -m torch.backends.xeon.run_cpu python_script args \n"
        "\n2. multi-instance \n"
        "\n   >>> python -m torch.backends.xeon.run_cpu --ninstances xxx "
        "--ncores-per-instance xx python_script args\n"
        "\n############################################################################# \n",
        formatter_class=RawTextHelpFormatter,
    )
    create_args(parser)
    args = parser.parse_args()
    main(args)
<|endoftext|># mypy: allow-untyped-defs
import torch


def is_available():
    r"""Return whether PyTorch is built with MKL support."""
    return torch._C.has_mkl


VERBOSE_OFF = 0
VERBOSE_ON = 1


class verbose:
    """
    On-demand oneMKL verbosing functionality.

    To make it easier to debug performance issues, oneMKL can dump verbose
    messages containing execution information like duration while executing
    the kernel. The verbosing functionality can be invoked via an environment
    variable named `MKL_VERBOSE`. However, this methodology dumps messages in
    all steps. Those are a large amount of verbose messages. Moreover, for
    investigating the performance issues, generally taking verbose messages
    for one single iteration is enough. This on-demand verbosing functionality
    makes it possible to control scope for verbose message dumping. In the
    following example, verbose messages will be dumped out for the second
    inference only.

    .. highlight:: python
    .. code-block:: python

        import torch

        model(data)
        with torch.backends.mkl.verbose(torch.backends.mkl.VERBOSE_ON):
            model(data)

    Args:
        level: Verbose level
            - ``VERBOSE_OFF``: Disable verbosing
            - ``VERBOSE_ON``:  Enable verbosing
    """

    def __init__(self, enable):
        self.enable = enable

    def __enter__(self):
        if self.enable == VERBOSE_OFF:
            return
        st = torch._C._verbose.mkl_set_verbose(self.enable)
        if not st:
            raise AssertionError(
                "Failed to set MKL into verbose mode. Please consider to disable this verbose scope."
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch._C._verbose.mkl_set_verbose(VERBOSE_OFF)
        return False
<|endoftext|>"""
User-facing API for controlling DSL operation overrides.

The torch.backends.python_native module provides control over DSL (Domain Specific Language)
operation overrides defined in torch._native. This allows users to selectively enable or disable
high-performance implementations from various DSLs like Triton and CuteDSL.

The module supports both coarse-grained control (entire DSLs) and fine-grained control
(individual operations or dispatch keys). All control operations support context managers
for temporary state changes.

Example usage::

    import torch.backends.python_native as pn

    # DSL-level control
    pn.triton.enabled = False  # Disable all triton ops
    pn.cutedsl.enabled = True  # Enable all cutedsl ops

    # Individual operation control
    pn.disable_operations("scaled_mm")  # Disable specific op across all DSLs
    pn.enable_operations("scaled_mm")  # Re-enable specific op

    # Context manager support
    with pn.triton.disabled():
        result = some_computation()  # Triton ops disabled here

    # Query capabilities
    print(pn.available_dsls)  # ['triton', 'cutedsl']
    print(pn.get_dsl_operations("triton"))  # Operations for triton
"""

import functools
import sys
import types
from contextlib import contextmanager

from torch.backends import ContextProp, flags_frozen, PropModule


@contextmanager
def _preserve_filter_state():
    """Context manager to save and restore registry filter state."""
    filter_state = _get_filter_state()

    # Save original state
    original_state = (
        set(filter_state._dsl_names),
        set(filter_state._op_symbols),
        set(filter_state._dispatch_keys),
    )

    try:
        yield filter_state
    finally:
        # Restore original state
        filter_state._dsl_names.clear()
        filter_state._op_symbols.clear()
        filter_state._dispatch_keys.clear()

        filter_state._dsl_names.update(original_state[0])
        filter_state._op_symbols.update(original_state[1])
        filter_state._dispatch_keys.update(original_state[2])


def _get_dsl_registry():
    """Lazy import to avoid circular imports."""
    from torch._native.dsl_registry import dsl_registry

    return dsl_registry


def _get_registry_functions():
    """Lazy import of registry functions."""
    from torch._native.registry import (
        _filter_state,
        _graphs,
        deregister_op_overrides,
        reenable_op_overrides,
    )

    return deregister_op_overrides, reenable_op_overrides, _graphs, _filter_state


def _get_filter_state():
    """Direct access to filter state."""
    return _get_registry_functions()[3]


def _get_dsl_module(dsl_name: str):
    """Get the registered DSL module for direct control.

    Uses the DSL registry to dynamically look up DSL modules instead of
    hard-coding the mapping. This makes the function automatically extensible
    for new DSLs without code changes.

    Args:
        dsl_name (str): Name of the DSL to retrieve.

    Returns:
        DSLModuleProtocol: The registered DSL module.

    Raises:
        ValueError: If the DSL is not registered.
    """
    registry = _get_dsl_registry()

    # Use the public API to get the DSL module
    dsl_module = registry.get_dsl_module(dsl_name)
    if dsl_module is not None:
        return dsl_module
    else:
        raise ValueError(
            f"Unknown DSL: {dsl_name}. Available DSLs: {registry.list_all_dsls()}"
        )


class DSLController:
    """Controller for a specific DSL."""

    def __init__(self, dsl_name: str):
        self._dsl_name = dsl_name

    @property
    def name(self) -> str:
        return self._dsl_name

    @property
    def available(self) -> bool:
        """Check if DSL runtime is available."""
        registry = _get_dsl_registry()
        return registry.is_dsl_available(self._dsl_name)

    @property
    def version(self):
        """Get DSL version."""
        registry = _get_dsl_registry()
        return registry.get_dsl_version(self._dsl_name)

    @property
    def backend(self) -> str:
        """Chosen lowering backend for DSLs that have one (e.g. helion)."""
        chosen = getattr(_get_dsl_module(self._dsl_name), "_chosen_backend", None)
        if chosen is None:
            raise AttributeError(f"{self._dsl_name} DSL has no configurable backend")
        return chosen()

    @property
    def enabled(self) -> bool:
        """Check if DSL is currently enabled."""
        filter_state = _get_filter_state()
        return self._dsl_name not in filter_state._dsl_names

    @enabled.setter
    def enabled(self, value: bool):
        """Enable or disable the DSL."""
        if flags_frozen():
            raise RuntimeError(
                f"not allowed to set {self._dsl_name} DSL flags "
                "after disable_global_flags; please use flags() context manager instead"
            )
        if value:
            self.enable()
        else:
            self.disable()

    def disable(self):
        """Disable all operations for this DSL."""
        dsl_module = _get_dsl_module(self._dsl_name)
        dsl_module.deregister_op_overrides()

    def enable(self):
        """Re-enable all operations for this DSL."""
        reenable_op_overrides = _get_registry_functions()[1]
        reenable_op_overrides(enable_dsl_names=self._dsl_name)

    @contextmanager
    def disabled(self):
        """Context manager to temporarily disable DSL."""
        original_state = self.enabled
        try:
            self.disable()
            yield
        finally:
            if original_state:
                self.enable()

    def __repr__(self):
        status = "available" if self.available else "unavailable"
        enabled_status = "enabled" if self.enabled else "disabled"
        return f"DSLController({self._dsl_name}, {status}, {enabled_status})"


class PythonNativeModule(PropModule):
    """Main module for python_native DSL control."""

    def __init__(self, original_module):
        super().__init__(original_module, original_module.__name__)

    @property
    def available_dsls(self) -> list[str]:
        """Get list of available DSLs."""
        registry = _get_dsl_registry()
        result = registry.list_available_dsls()
        return list(result) if not isinstance(result, list) else result

    @property
    def all_dsls(self) -> list[str]:
        """Get list of all registered DSLs."""
        registry = _get_dsl_registry()
        result = registry.list_all_dsls()
        return list(result) if not isinstance(result, list) else result

    def get_dsl_operations(self, dsl_name: str) -> list[str]:
        """Get list of operations registered by a specific DSL.

        Args:
            dsl_name (str): Name of the DSL to query (e.g., 'triton', 'cutedsl').

        Returns:
            list[str]: Sorted list of operation names registered by the DSL.

        Example::

            ops = torch.backends.python_native.get_dsl_operations("triton")
            print(ops)  # ['triton_to_mxfp8_dim0', ...]
        """
        from torch._native.registry import get_dsl_operations

        return get_dsl_operations(dsl_name)

    def disable_operations(self, *op_symbols: str):
        """Disable specific operations across all DSLs.

        Args:
            *op_symbols (str): Names of operations to disable.

        Example::

            # Disable scaled matrix multiply across all DSLs
            torch.backends.python_native.disable_operations("scaled_mm")

            # Disable multiple operations
            torch.backends.python_native.disable_operations(
                "scaled_mm", "flash_attention"
            )
        """
        deregister_op_overrides = _get_registry_functions()[0]
        deregister_op_overrides(disable_op_symbols=list(op_symbols))

    def enable_operations(self, *op_symbols: str):
        """Re-enable specific operations across all DSLs.

        Args:
            *op_symbols (str): Names of operations to re-enable.

        Example::

            # Re-enable previously disabled operations
            torch.backends.python_native.enable_operations(
                "scaled_mm", "flash_attention"
            )
        """
        reenable_op_overrides = _get_registry_functions()[1]
        reenable_op_overrides(enable_op_symbols=list(op_symbols))

    def disable_dispatch_keys(self, *dispatch_keys: str):
        """Disable operations at specific dispatch keys.

        Args:
            *dispatch_keys (str): Dispatch keys to disable (e.g., 'CUDA', 'CPU').

        Example::

            # Disable all native operations on CUDA
            torch.backends.python_native.disable_dispatch_keys("CUDA")
        """
        deregister_op_overrides = _get_registry_functions()[0]
        deregister_op_overrides(disable_dispatch_keys=list(dispatch_keys))

    def enable_dispatch_keys(self, *dispatch_keys: str):
        """Re-enable operations at specific dispatch keys.

        Args:
            *dispatch_keys (str): Dispatch keys to re-enable (e.g., 'CUDA', 'CPU').

        Example::

            # Re-enable native operations on CUDA
            torch.backends.python_native.enable_dispatch_keys("CUDA")
        """
        reenable_op_overrides = _get_registry_functions()[1]
        reenable_op_overrides(enable_dispatch_keys=list(dispatch_keys))

    @contextmanager
    def operations_disabled(self, *op_symbols: str):
        """Context manager to temporarily disable operations.

        Args:
            *op_symbols (str): Names of operations to temporarily disable.

        Example::

            with torch.backends.python_native.operations_disabled("scaled_mm"):
                # scaled_mm is disabled across all DSLs
                result = model(input)
            # scaled_mm is automatically re-enabled here
        """
        filter_state = _get_filter_state()
        previously_disabled_ops = {
            op for op in op_symbols if op in filter_state._op_symbols
        }

        self.disable_operations(*op_symbols)
        try:
            yield
        finally:
            # Only re-enable operations that weren't already disabled
            ops_to_reenable = [
                op for op in op_symbols if op not in previously_disabled_ops
            ]
            if ops_to_reenable:
                self.enable_operations(*ops_to_reenable)

    @functools.lru_cache(maxsize=16)  # noqa: B019
    def _get_dsl_controller(self, name: str) -> "DSLController":
        """Get or create a DSL controller (cached)."""
        return DSLController(name)

    def _get_registry_functions(self):
        """Expose registry functions for testing."""
        return _get_registry_functions()

    def is_operation_disabled(self, op_symbol: str) -> bool:
        """Check if an operation is currently disabled."""
        filter_state = _get_filter_state()
        return op_symbol in filter_state._op_symbols

    def is_dsl_disabled(self, dsl_name: str) -> bool:
        """Check if a DSL is currently disabled."""
        filter_state = _get_filter_state()
        return dsl_name in filter_state._dsl_names

    def __getattr__(self, name: str):
        """Dynamic attribute access for DSL controllers."""
        # Skip dunder attributes to avoid triggering DSL registry lookups
        # during torch initialization. inspect.getmodule() calls
        # hasattr(module, '__file__') which would otherwise cause a circular
        # import through _get_dsl_registry() while torch is still loading.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")

        if name in self.all_dsls:
            return self._get_dsl_controller(name)

        # Expose private functions for testing
        if name == "_get_dsl_module":
            return _get_dsl_module
        if name == "_get_registry_functions":
            return self._get_registry_functions
        if name == "_get_filter_state":
            return _get_filter_state

        raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")

    def __dir__(self):
        """Return available attributes including DSL names."""
        attrs = set(super().__dir__())
        attrs.update(
            {
                "available_dsls",
                "all_dsls",
                "get_dsl_operations",
                "disable_operations",
                "enable_operations",
                "disable_dispatch_keys",
                "enable_dispatch_keys",
                "operations_disabled",
                "is_operation_disabled",
                "is_dsl_disabled",
            }
        )

        # Add DSL names
        try:
            attrs.update(self.all_dsls)
        except Exception:
            # If registry not available yet, skip DSL names
            pass

        return sorted(attrs)


# Replace the current module with our enhanced version
sys.modules[__name__] = PythonNativeModule(sys.modules[__name__])
<|endoftext|># mypy: allow-untyped-defs
import contextlib
from typing import Any
from typing_extensions import deprecated

import torch


__all__ = [
    "is_built",
    "cuFFTPlanCacheAttrContextProp",
    "cuFFTPlanCache",
    "cuFFTPlanCacheManager",
    "cuBLASModule",
    "preferred_linalg_library",
    "preferred_blas_library",
    "cublas_workspace_size",
    "cublaslt_workspace_size",
    "blas_workspace_size",
    "preferred_rocm_fa_library",
    "is_ck_sdpa_available",
    "cufft_plan_cache",
    "matmul",
    "SDPAParams",
    "enable_cudnn_sdp",
    "cudnn_sdp_enabled",
    "enable_flash_sdp",
    "flash_sdp_enabled",
    "enable_mem_efficient_sdp",
    "mem_efficient_sdp_enabled",
    "math_sdp_enabled",
    "enable_math_sdp",
    "allow_fp16_bf16_reduction_math_sdp",
    "fp16_bf16_reduction_math_sdp_allowed",
    "is_flash_attention_available",
    "can_use_flash_attention",
    "can_use_efficient_attention",
    "can_use_cudnn_attention",
    "sdp_kernel",
]


def is_built():
    r"""
    Return whether PyTorch is built with CUDA support.

    Note that this doesn't necessarily mean CUDA is available; just that if this PyTorch
    binary were run on a machine with working CUDA drivers and devices, we would be able to use it.
    """
    return torch._C._has_cuda


class cuFFTPlanCacheAttrContextProp:
    # Like regular ContextProp, but uses the `.device_index` attribute from the
    # calling object as the first argument to the getter and setter.
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter(obj.device_index)

    def __set__(self, obj, val):
        if isinstance(self.setter, str):
            raise RuntimeError(self.setter)
        self.setter(obj.device_index, val)


class cuFFTPlanCache:
    r"""
    Represent a specific plan cache for a specific `device_index`.

    The attributes `size` and `max_size`, and method `clear`, can fetch and/ or
    change properties of the C++ cuFFT plan cache.
    """

    def __init__(self, device_index):
        self.device_index = device_index

    size = cuFFTPlanCacheAttrContextProp(
        torch._cufft_get_plan_cache_size,
        ".size is a read-only property showing the number of plans currently in the "
        "cache. To change the cache capacity, set cufft_plan_cache.max_size.",
    )

    max_size = cuFFTPlanCacheAttrContextProp(
        torch._cufft_get_plan_cache_max_size, torch._cufft_set_plan_cache_max_size
    )

    def clear(self):
        return torch._cufft_clear_plan_cache(self.device_index)


class cuFFTPlanCacheManager:
    r"""
    Represent all cuFFT plan caches, return the cuFFTPlanCache for a given device when indexed.

    Finally, this object, when used directly as a `cuFFTPlanCache` object (e.g.,
    setting the `.max_size`) attribute, the current device's cuFFT plan cache is
    used.
    """

    __initialized = False

    def __init__(self):
        self.caches = []
        self.__initialized = True

    def __getitem__(self, device):
        index = torch.cuda._utils._get_device_index(device)
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"cufft_plan_cache: expected 0 <= device index < {torch.cuda.device_count()}, but got "
                f"device with index {index}"
            )
        if len(self.caches) == 0:
            self.caches.extend(
                cuFFTPlanCache(index) for index in range(torch.cuda.device_count())
            )
        return self.caches[index]

    def __getattr__(self, name):
        return getattr(self[torch.cuda.current_device()], name)

    def __setattr__(self, name, value):
        if self.__initialized:
            return setattr(self[torch.cuda.current_device()], name, value)
        else:
            return super().__setattr__(name, value)


class cuBLASModule:
    @staticmethod
    def _parse_reduction_setting(value: Any, attr_name: str) -> tuple[bool, bool]:
        def _ensure_bool(obj: Any, which: str) -> bool:
            if isinstance(obj, bool):
                return obj
            raise TypeError(
                f"{attr_name} expects a bool for {which}, but got {type(obj)!r}"
            )

        if isinstance(value, bool):
            return value, True
        if isinstance(value, (list, tuple)):
            if not value:
                raise TypeError(f"{attr_name} expects at least one boolean argument")
            if len(value) > 2:
                raise TypeError(f"{attr_name} expects at most two boolean arguments")
            allow_reduced_precision = _ensure_bool(value[0], "allow_reduced_precision")
            if len(value) == 1:
                return allow_reduced_precision, True
            allow_splitk = _ensure_bool(value[1], "allow_splitk")
            return allow_reduced_precision, allow_splitk
        raise TypeError(
            f"{attr_name} expects a bool or a tuple/list of bools, but got {type(value)!r}"
        )

    def __getattr__(self, name):
        if name == "allow_tf32":
            return torch._C._get_cublas_allow_tf32()
        elif name == "prefer_cublaslt_grouped_gemm":
            return torch._C._get_cublaslt_prefer_grouped_gemm()
        elif name == "allow_fp16_reduced_precision_reduction":
            allow_reduced_precision, _ = (
                torch._C._get_cublas_allow_fp16_reduced_precision_reduction()
            )
            return allow_reduced_precision
        elif name == "allow_fp16_reduced_precision_reduction_split_k":
            _, allow_splitk = (
                torch._C._get_cublas_allow_fp16_reduced_precision_reduction()
            )
            return allow_splitk
        elif name == "allow_bf16_reduced_precision_reduction":
            allow_reduced_precision, _ = (
                torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
            )
            return allow_reduced_precision
        elif name == "allow_bf16_reduced_precision_reduction_split_k":
            _, allow_splitk = (
                torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
            )
            return allow_splitk
        elif name == "allow_fp16_accumulation":
            return torch._C._get_cublas_allow_fp16_accumulation()
        elif name == "fp32_precision":
            return torch._C._get_fp32_precision_getter("cuda", "matmul")
        raise AttributeError("Unknown attribute " + name)

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            return torch._C._set_cublas_allow_tf32(value)
        elif name == "prefer_cublaslt_grouped_gemm":
            return torch._C._set_cublaslt_prefer_grouped_gemm(value)
        elif name == "allow_fp16_reduced_precision_reduction":
            allow_reduced_precision, allow_splitk = self._parse_reduction_setting(
                value, "allow_fp16_reduced_precision_reduction"
            )
            return torch._C._set_cublas_allow_fp16_reduced_precision_reduction(
                allow_reduced_precision,
                allow_splitk,
            )
        elif name == "allow_bf16_reduced_precision_reduction":
            allow_reduced_precision, allow_splitk = self._parse_reduction_setting(
                value, "allow_bf16_reduced_precision_reduction"
            )
            return torch._C._set_cublas_allow_bf16_reduced_precision_reduction(
                allow_reduced_precision,
                allow_splitk,
            )
        elif name == "allow_fp16_accumulation":
            return torch._C._set_cublas_allow_fp16_accumulation(value)
        elif name == "fp32_precision":
            return torch._C._set_fp32_precision_setter("cuda", "matmul", value)
        raise AttributeError("Unknown attribute " + name)


_LinalgBackends = {
    "default": torch._C._LinalgBackend.Default,
    "cusolver": torch._C._LinalgBackend.Cusolver,
    "magma": torch._C._LinalgBackend.Magma,
}
_LinalgBackends_str = ", ".join(_LinalgBackends.keys())


def preferred_linalg_library(
    backend: None | str | torch._C._LinalgBackend = None,
) -> torch._C._LinalgBackend:
    r"""
    Override the heuristic PyTorch uses to choose between cuSOLVER and MAGMA for CUDA linear algebra operations.

    .. warning:: This flag is experimental and subject to change.

    When PyTorch runs a CUDA linear algebra operation it often uses the cuSOLVER or MAGMA libraries,
    and if both are available it decides which to use with a heuristic.
    This flag (a :class:`str`) allows overriding those heuristics.

    * If `"cusolver"` is set then cuSOLVER will be used wherever possible.
    * If `"magma"` is set then MAGMA will be used wherever possible.
    * If `"default"` (the default) is set then heuristics will be used to pick between
      cuSOLVER and MAGMA if both are available.
    * When no input is given, this function returns the currently preferred library.
    * User may use the environment variable TORCH_LINALG_PREFER_CUSOLVER=1 to set the preferred library to cuSOLVER
      globally.
      This flag only sets the initial value of the preferred library and the preferred library
      may still be overridden by this function call later in your script.

    Note: When a library is preferred other libraries may still be used if the preferred library
    doesn't implement the operation(s) called.
    This flag may achieve better performance if PyTorch's heuristic library selection is incorrect
    for your application's inputs.

    Currently supported linalg operators:

    * :func:`torch.linalg.inv`
    * :func:`torch.linalg.inv_ex`
    * :func:`torch.linalg.cholesky`
    * :func:`torch.linalg.cholesky_ex`
    * :func:`torch.cholesky_solve`
    * :func:`torch.cholesky_inverse`
    * :func:`torch.linalg.lu_factor`
    * :func:`torch.linalg.lu`
    * :func:`torch.linalg.lu_solve`
    * :func:`torch.linalg.qr`
    * :func:`torch.linalg.eigh`
    * :func:`torch.linalg.eigvals`
    * :func:`torch.linalg.svd`
    * :func:`torch.linalg.svdvals`
    """
    if backend is None:
        pass
    elif isinstance(backend, str):
        if backend not in _LinalgBackends:
            raise RuntimeError(
                f"Unknown input value. Choose from: {_LinalgBackends_str}."
            )
        torch._C._set_linalg_preferred_backend(_LinalgBackends[backend])
    elif isinstance(backend, torch._C._LinalgBackend):
        torch._C._set_linalg_preferred_backend(backend)
    else:
        raise RuntimeError("Unknown input value type.")

    return torch._C._get_linalg_preferred_backend()


_BlasBackends = {
    "default": torch._C._BlasBackend.Default,
    "cublas": torch._C._BlasBackend.Cublas,
    "hipblas": torch._C._BlasBackend.Cublas,  # alias
    "cublaslt": torch._C._BlasBackend.Cublaslt,
    "hipblaslt": torch._C._BlasBackend.Cublaslt,  # alias
    "ck": torch._C._BlasBackend.Ck,
}
_BlasBackends_str = ", ".join(_BlasBackends.keys())


def preferred_blas_library(
    backend: None | str | torch._C._BlasBackend = None,
) -> torch._C._BlasBackend:
    r"""
    Override the library PyTorch uses for BLAS operations. Choose between cuBLAS, cuBLASLt, and CK [ROCm-only].

    .. warning:: This flag is experimental and subject to change.

    When PyTorch runs a CUDA BLAS operation it defaults to cuBLAS even if both cuBLAS and cuBLASLt are available.
    For PyTorch built for ROCm, hipBLAS, hipBLASLt, and CK may offer different performance.
    This flag (a :class:`str`) allows overriding which BLAS library to use.

    * If `"cublas"` is set then cuBLAS will be used wherever possible.
    * If `"cublaslt"` is set then cuBLASLt will be used wherever possible.
    * If `"ck"` is set then CK will be used wherever possible.
    * If `"default"` (the default) is set then heuristics will be used to pick between the other options.
    * When no input is given, this function returns the currently preferred library.
    * User may use the environment variable TORCH_BLAS_PREFER_CUBLASLT=1 to set the preferred library to cuBLASLt
      globally.
      This flag only sets the initial value of the preferred library and the preferred library
      may still be overridden by this function call later in your script.

    Note: When a library is preferred other libraries may still be used if the preferred library
    doesn't implement the operation(s) called.
    This flag may achieve better performance if PyTorch's library selection is incorrect
    for your application's inputs.

    """
    if backend is None:
        pass
    elif isinstance(backend, str):
        if backend not in _BlasBackends:
            raise RuntimeError(
                f"Unknown input value. Choose from: {_BlasBackends_str}."
            )
        torch._C._set_blas_preferred_backend(_BlasBackends[backend])
    elif isinstance(backend, torch._C._BlasBackend):
        torch._C._set_blas_preferred_backend(backend)
    else:
        raise RuntimeError("Unknown input value type.")

    return torch._C._get_blas_preferred_backend()


def cublas_workspace_size(size: None | int = None) -> int:
    r"""Query or set the cuBLAS workspace size in bytes.

    When called with no arguments, returns the current workspace size.
    When called with a size argument, sets the workspace size and returns the new value.
    Setting the workspace size will take precedence over the CUBLAS_WORKSPACE_CONFIG environment variable.
    Changes take effect lazily: only handles used after the change get new workspaces.

    Args:
        size (int, optional): workspace size in bytes. Must be non-negative.

    Returns:
        int: the current (or newly set) workspace size in bytes.
    """
    if size is not None:
        torch._C._cuda_setCublasWorkspaceSize(size)
    return torch._C._cuda_getCublasWorkspaceSize()


def cublaslt_workspace_size(size: None | int = None) -> int:
    r"""Query or set the cuBLASLt workspace size in bytes.

    When called with no arguments, returns the current workspace size.
    When called with a size argument, sets the workspace size and returns the new value.
    Setting the workspace size will take precedence over the CUBLASLT_WORKSPACE_SIZE environment variable.
    Changes take effect lazily: only handles used after the change get new workspaces.

    Args:
        size (int, optional): workspace size in bytes. Must be non-negative.

    Returns:
        int: the current (or newly set) workspace size in bytes.
    """
    if size is not None:
        torch._C._cuda_setCublasLtWorkspaceSize(size)
    return torch._C._cuda_getCublasLtWorkspaceSize()


def blas_workspace_size(
    size: None | int = None,
    backend: None | str | torch._C._BlasBackend = None,
) -> int:
    r"""Query or set the BLAS workspace size for a given backend.

    Convenience wrapper that dispatches to :func:`cublas_workspace_size` or
    :func:`cublaslt_workspace_size` depending on the backend.

    When *backend* is ``None`` the current :func:`preferred_blas_library` is
    used.  ``Default`` is resolved to the platform's default backend (cuBLAS
    on NVIDIA, potentially hipBLASLt on supported ROCm architectures).

    .. note::

       When ``TORCH_CUBLASLT_UNIFIED_WORKSPACE`` is enabled (the default on
       open-source CUDA builds), the cuBLASLt workspace is capped at the
       cuBLAS workspace size and physically reuses the same allocation.
       Setting a large cuBLASLt workspace via this function will therefore
       *not* increase memory beyond the cuBLAS workspace size.

    .. note::

        Setting the workspace size for the cublas backend will take precedence
        over the CUBLAS_WORKSPACE_CONFIG environment variable, and setting the
        workspace size for the cublaslt backend will take precedence over the
        CUBLASLT_WORKSPACE_SIZE environment variable.

    Args:
        size (int, optional): workspace size in bytes.  Must be non-negative.
            When omitted the current size is returned without modification.
        backend (str | torch._C._BlasBackend, optional): which backend's
            workspace to query/set.  Accepts the same strings as
            :func:`preferred_blas_library` (e.g. ``"cublas"``, ``"cublaslt"``).

    Returns:
        int: the current (or newly set) workspace size in bytes.

    Raises:
        RuntimeError: if the resolved backend is CK (no workspace concept).
    """
    if backend is None:
        resolved = preferred_blas_library()
    elif isinstance(backend, str):
        if backend not in _BlasBackends:
            raise RuntimeError(
                f"Unknown backend string. Choose from: {_BlasBackends_str}."
            )
        resolved = _BlasBackends[backend]
    elif isinstance(backend, torch._C._BlasBackend):
        resolved = backend
    else:
        raise RuntimeError("Unknown backend type.")

    if resolved == torch._C._BlasBackend.Default:
        resolved = torch._C._get_blas_default_backend()

    if resolved == torch._C._BlasBackend.Ck:
        raise RuntimeError("CK backend does not use a workspace.")

    if resolved == torch._C._BlasBackend.Cublaslt:
        return cublaslt_workspace_size(size)
    return cublas_workspace_size(size)


_ROCmFABackends = {
    "default": torch._C._ROCmFABackend.Default,
    "aotriton": torch._C._ROCmFABackend.AOTriton,
    "ck": torch._C._ROCmFABackend.Ck,
}
_ROCmFABackends_str = ", ".join(_ROCmFABackends.keys())


from torch._C import _SDPAParams as SDPAParams, _SDPBackend as SDPBackend


def preferred_rocm_fa_library(
    backend: None | str | torch._C._ROCmFABackend = None,
) -> torch._C._ROCmFABackend:
    r"""
    [ROCm-only]
    Override the backend PyTorch uses in ROCm environments for Flash Attention. Choose between AOTriton and CK

    .. warning:: This flag is experimental and subject to change.

    When Flash Attention is enabled and desired, PyTorch defaults to using AOTriton as the backend.
    This flag (a :class:`str`) allows users to override this backend to use composable_kernel

    * If `"default"` is set then the default backend will be used wherever possible. Currently AOTriton.
    * If `"aotriton"` is set then AOTriton will be used wherever possible.
    * If `"ck"` is set then CK will be used wherever possible.
    * When no input is given, this function returns the currently preferred library.
    * User may use the environment variable TORCH_ROCM_FA_PREFER_CK=1 to set the preferred library to CK
      globally.

    Note: When a library is preferred other libraries may still be used if the preferred library
    doesn't implement the operation(s) called.
    This flag may achieve better performance if PyTorch's library selection is incorrect
    for your application's inputs.
    """
    if backend is None:
        pass
    elif isinstance(backend, str):
        if backend not in _ROCmFABackends:
            raise RuntimeError(
                f"Unknown input value. Choose from: {_ROCmFABackends_str}."
            )
        torch._C._set_rocm_fa_preferred_backend(_ROCmFABackends[backend])
    elif isinstance(backend, torch._C._ROCmFABackend):
        torch._C._set_rocm_fa_preferred_backend(backend)
    else:
        raise ValueError(f"Unknown input value. Choose from: {_ROCmFABackends_str}.")

    return torch._C._get_rocm_fa_preferred_backend()


# Set the __module__ attribute
SDPAParams.__module__ = "torch.backends.cuda"
SDPAParams.__name__ = "SDPAParams"


def is_ck_sdpa_available() -> bool:
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether composable_kernel may be used as the backend for
    scaled-dot-product-attention.
    """
    # pyrefly: ignore [missing-attribute]
    return torch._C._is_ck_sdpa_available()


def flash_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether flash scaled dot product attention is enabled or not.
    """
    return torch._C._get_flash_sdp_enabled()


def enable_flash_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables flash scaled dot product attention.
    """
    torch._C._set_sdp_use_flash(enabled)


def mem_efficient_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether memory efficient scaled dot product attention is enabled or not.
    """
    return torch._C._get_mem_efficient_sdp_enabled()


def enable_mem_efficient_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables memory efficient scaled dot product attention.
    """
    torch._C._set_sdp_use_mem_efficient(enabled)


def math_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether math scaled dot product attention is enabled or not.
    """
    return torch._C._get_math_sdp_enabled()


def enable_math_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables math scaled dot product attention.
    """
    torch._C._set_sdp_use_math(enabled)


def allow_fp16_bf16_reduction_math_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables fp16/bf16 reduction in math scaled dot product attention.
    """
    torch._C._set_math_sdp_allow_fp16_bf16_reduction(enabled)


def fp16_bf16_reduction_math_sdp_allowed():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether fp16/bf16 reduction in math scaled dot product attention is enabled or not.
    """
    return torch._C._get_math_sdp_allow_fp16_bf16_reduction()


def is_flash_attention_available() -> bool:
    r"""Check if PyTorch was built with FlashAttention for scaled_dot_product_attention.

    Returns:
        True if FlashAttention is built and available; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    return torch._C._is_flash_attention_available()


def can_use_flash_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if FlashAttention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn debug information as to why FlashAttention could not be run.
            Defaults to False.

    Returns:
        True if FlashAttention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    return torch._C._can_use_flash_attention(params, debug)


def can_use_efficient_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if efficient_attention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn with information as to why efficient_attention could not be run.
            Defaults to False.

    Returns:
        True if efficient_attention can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    return torch._C._can_use_mem_efficient_attention(params, debug)


def can_use_cudnn_attention(params: SDPAParams, debug: bool = False) -> bool:
    r"""Check if cudnn_attention can be utilized in scaled_dot_product_attention.

    Args:
        params: An instance of SDPAParams containing the tensors for query,
                key, value, an optional attention mask, dropout rate, and
                a flag indicating if the attention is causal.
        debug: Whether to logging.warn with information as to why cuDNN attention could not be run.
            Defaults to False.

    Returns:
        True if cuDNN can be used with the given parameters; otherwise, False.

    Note:
        This function is dependent on a CUDA-enabled build of PyTorch. It will return False
        in non-CUDA environments.
    """
    return torch._C._can_use_cudnn_attention(params, debug)


def cudnn_sdp_enabled():
    r"""
    .. warning:: This flag is beta and subject to change.

    Returns whether cuDNN scaled dot product attention is enabled or not.
    """
    return torch._C._get_cudnn_sdp_enabled()


def enable_cudnn_sdp(enabled: bool):
    r"""
    .. warning:: This flag is beta and subject to change.

    Enables or disables cuDNN scaled dot product attention.
    """
    torch._C._set_sdp_use_cudnn(enabled)


@contextlib.contextmanager
@deprecated(
    (
        "`torch.backends.cuda.sdp_kernel()` is deprecated. "
        "In the future, this context manager will be removed. "
        "Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, "
        "with updated signature."
    ),
    category=FutureWarning,
)
def sdp_kernel(
    enable_flash: bool = True,
    enable_math: bool = True,
    enable_mem_efficient: bool = True,
    enable_cudnn: bool = True,
):
    r"""
    .. warning:: This flag is beta and subject to change.

    This context manager can be used to temporarily enable or disable any of the three backends for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored.
    """
    from torch.nn.attention import sdpa_kernel

    backend_list = []
    if enable_flash:
        backend_list.append(SDPBackend.FLASH_ATTENTION)
    if enable_mem_efficient:
        backend_list.append(SDPBackend.EFFICIENT_ATTENTION)
    if enable_math:
        backend_list.append(SDPBackend.MATH)
    if enable_cudnn:
        backend_list.append(SDPBackend.CUDNN_ATTENTION)

    with sdpa_kernel(backend_list) as context:
        try:
            yield context
        finally:
            pass


cufft_plan_cache = cuFFTPlanCacheManager()
matmul = cuBLASModule()
<|endoftext|>from functools import lru_cache as _lru_cache
from typing import Optional, TYPE_CHECKING

import torch
from torch.library import Library as _Library


__all__ = [
    "get_core_count",
    "get_name",
    "is_built",
    "is_available",
    "is_macos13_or_newer",
    "is_macos_or_newer",
]


def is_built() -> bool:
    r"""Return whether PyTorch is built with MPS support.

    Note that this doesn't necessarily mean MPS is available; just that
    if this PyTorch binary were run on a machine with working MPS drivers
    and devices, we would be able to use it.
    """
    return torch._C._has_mps


@_lru_cache
def is_available() -> bool:
    r"""Return a bool indicating if MPS is currently available."""
    return torch._C._mps_is_available()


@_lru_cache
def is_macos_or_newer(major: int, minor: int) -> bool:
    r"""Return a bool indicating whether MPS is running on given MacOS or newer."""
    return torch._C._mps_is_on_macos_or_newer(major, minor)


@_lru_cache
def is_macos13_or_newer(minor: int = 0) -> bool:
    r"""Return a bool indicating whether MPS is running on MacOS 13 or newer."""
    return torch._C._mps_is_on_macos_or_newer(13, minor)


@_lru_cache
def get_name() -> str:
    r"""Return Metal device name"""
    return torch._C._mps_get_name()


@_lru_cache
def get_core_count() -> int:
    r"""Return GPU core count.

    According to the documentation, one core is comprised of 16 Execution Units.
    One execution Unit has 8 ALUs.
    And one ALU can run 24 threads, i.e. one core is capable of executing 3072 threads concurrently.
    """
    return torch._C._mps_get_core_count()
<|endoftext|># mypy: allow-untyped-defs
from contextlib import contextmanager

import torch
from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


__all__ = ["is_available", "flags", "set_flags"]


def is_available():
    r"""Return whether PyTorch is built with NNPACK support."""
    return torch._nnpack_available()


def set_flags(_enabled):
    r"""Set if nnpack is enabled globally"""
    orig_flags = (torch._C._get_nnpack_enabled(),)
    torch._C._set_nnpack_enabled(_enabled)
    return orig_flags


@contextmanager
def flags(enabled=False):
    r"""Context manager for setting if nnpack is enabled globally"""
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(orig_flags[0])
<|endoftext|># mypy: allow-untyped-defs
import sys
import types

import torch


class _XNNPACKEnabled:
    def __get__(self, obj, objtype):
        return torch._C._is_xnnpack_enabled()

    def __set__(self, obj, val):
        raise RuntimeError("Assignment not supported")


class XNNPACKEngine(types.ModuleType):
    def __init__(self, m, name):
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)

    enabled = _XNNPACKEnabled()


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = XNNPACKEngine(sys.modules[__name__], __name__)
<|endoftext|># mypy: allow-untyped-defs
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Optional

import torch
from torch.backends import (
    __allow_nonbracketed_mutation,
    _FP32Precision,
    _get_fp32_precision_getter,
    _set_fp32_precision_setter,
    ContextProp,
    PropModule,
)

from . import rnn


try:
    from torch._C import _cudnn
except ImportError:
    _cudnn = None  # type: ignore[assignment]

# Write:
#
#   torch.backends.cudnn.enabled = False
#
# to globally disable CuDNN/MIOpen

__cudnn_version: int | None = None

if _cudnn is not None:

    def _init():
        global __cudnn_version
        if __cudnn_version is None:
            # pyrefly: ignore [missing-attribute]
            __cudnn_version = _cudnn.getVersionInt()
            # pyrefly: ignore [missing-attribute]
            runtime_version = _cudnn.getRuntimeVersion()
            # pyrefly: ignore [missing-attribute]
            compile_version = _cudnn.getCompileVersion()
            runtime_major, runtime_minor, _ = runtime_version
            compile_major, compile_minor, _ = compile_version
            # Different major versions are always incompatible
            # Starting with cuDNN 7, minor versions are backwards-compatible
            # Not sure about MIOpen (ROCm), so always do a strict check
            if runtime_major != compile_major:
                cudnn_compatible = False
            # pyrefly: ignore [missing-attribute]
            elif runtime_major < 7 or not _cudnn.is_cuda:
                cudnn_compatible = runtime_minor == compile_minor
            else:
                cudnn_compatible = runtime_minor >= compile_minor
            if not cudnn_compatible:
                if os.environ.get("PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK", "0") == "1":
                    return True
                base_error_msg = (
                    f"cuDNN version incompatibility: "
                    f"PyTorch was compiled  against {compile_version} "
                    f"but found runtime version {runtime_version}. "
                    f"PyTorch already comes bundled with cuDNN. "
                    f"One option to resolving this error is to ensure PyTorch "
                    f"can find the bundled cuDNN. "
                )

                if "LD_LIBRARY_PATH" in os.environ:
                    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
                    if any(
                        substring in ld_library_path for substring in ["cuda", "cudnn"]
                    ):
                        raise RuntimeError(
                            f"{base_error_msg}"
                            f"Looks like your LD_LIBRARY_PATH contains incompatible version of cudnn. "
                            f"Please either remove it from the path or install cudnn {compile_version}"
                        )
                    else:
                        raise RuntimeError(
                            f"{base_error_msg}"
                            f"one possibility is that there is a "
                            f"conflicting cuDNN in LD_LIBRARY_PATH."
                        )
                else:
                    raise RuntimeError(base_error_msg)
            # Check if cuDNN version is compatible with available CUDA devices
            if torch.cuda.is_available() and not torch.version.hip:
                min_cc = min(
                    [
                        torch.cuda.get_device_capability(i)
                        for i in range(torch.cuda.device_count())
                    ]
                )
                if __cudnn_version >= 91100 and min_cc < (7, 5):
                    raise RuntimeError(
                        f"cuDNN version {__cudnn_version} is not compatible with devices with SM < 7.5. "
                        f"Please install a version of PyTorch with a compatible cuDNN version. "
                        f"https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix"
                    )

        return True

else:

    def _init():
        return False


def version():
    """Return the version of cuDNN."""
    if not _init():
        return None
    return __cudnn_version


CUDNN_TENSOR_DTYPES = {
    torch.half,
    torch.float,
    torch.double,
}


def is_available():
    r"""Return a bool indicating if CUDNN is currently available."""
    return torch._C._has_cudnn


def is_acceptable(tensor):
    if not torch._C._get_cudnn_enabled():
        return False
    if tensor.device.type != "cuda" or tensor.dtype not in CUDNN_TENSOR_DTYPES:
        return False
    if not is_available():
        warnings.warn(
            "PyTorch was compiled without cuDNN/MIOpen support. To use cuDNN/MIOpen, rebuild "
            "PyTorch making sure the library is visible to the build system.",
            stacklevel=2,
        )
        return False
    if not _init():
        warnings.warn(
            "cuDNN/MIOpen library not found. Check your {libpath}".format(
                libpath={"darwin": "DYLD_LIBRARY_PATH", "win32": "PATH"}.get(
                    sys.platform, "LD_LIBRARY_PATH"
                )
            ),
            stacklevel=2,
        )
        return False
    return True


def set_flags(
    _enabled=None,
    _benchmark=None,
    _benchmark_limit=None,
    _deterministic=None,
    _allow_tf32=None,
    _fp32_precision="none",
    _depthwise_kernel=None,
):
    orig_flags = (
        torch._C._get_cudnn_enabled(),
        torch._C._get_cudnn_benchmark(),
        None if not is_available() else torch._C._cuda_get_cudnn_benchmark_limit(),
        torch._C._get_cudnn_deterministic(),
        torch._C._get_cudnn_allow_tf32(),
        torch._C._get_fp32_precision_getter("cuda", "all"),
        torch._C._get_cudnn_depthwise_kernel(),
    )
    if _enabled is not None:
        torch._C._set_cudnn_enabled(_enabled)
    if _benchmark is not None:
        torch._C._set_cudnn_benchmark(_benchmark)
    if _benchmark_limit is not None and is_available():
        torch._C._cuda_set_cudnn_benchmark_limit(_benchmark_limit)
    if _deterministic is not None:
        torch._C._set_cudnn_deterministic(_deterministic)
    if _allow_tf32 is not None:
        torch._C._set_cudnn_allow_tf32(_allow_tf32)
    if _fp32_precision is not None:
        torch._C._set_fp32_precision_setter("cuda", "all", _fp32_precision)
    if _depthwise_kernel is not None:
        torch._C._set_cudnn_depthwise_kernel(_depthwise_kernel)
    return orig_flags


@contextmanager
def flags(
    enabled=False,
    benchmark=False,
    benchmark_limit=10,
    deterministic=False,
    allow_tf32=True,
    fp32_precision="none",
    depthwise_kernel="auto",
):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(
            enabled,
            benchmark,
            benchmark_limit,
            deterministic,
            allow_tf32,
            fp32_precision,
            depthwise_kernel,
        )
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.<cudnn|mkldnn>.enabled = True


class CudnnModule(PropModule):
    enabled = ContextProp(torch._C._get_cudnn_enabled, torch._C._set_cudnn_enabled)
    deterministic = ContextProp(
        torch._C._get_cudnn_deterministic, torch._C._set_cudnn_deterministic
    )
    benchmark = ContextProp(
        torch._C._get_cudnn_benchmark, torch._C._set_cudnn_benchmark
    )
    benchmark_limit = None
    if is_available():
        benchmark_limit = ContextProp(
            torch._C._cuda_get_cudnn_benchmark_limit,
            torch._C._cuda_set_cudnn_benchmark_limit,
        )
    allow_tf32 = ContextProp(
        torch._C._get_cudnn_allow_tf32, torch._C._set_cudnn_allow_tf32
    )
    conv = _FP32Precision("cuda", "conv")
    fp32_precision = ContextProp(
        _get_fp32_precision_getter("cuda", "all"),
        _set_fp32_precision_setter("cuda", "all"),
    )
    depthwise_kernel = ContextProp(
        torch._C._get_cudnn_depthwise_kernel,
        torch._C._set_cudnn_depthwise_kernel,
    )


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = CudnnModule(sys.modules[__name__], __name__)

# Add type annotation for the replaced module
enabled: bool
deterministic: bool
benchmark: bool
allow_tf32: bool
fp32_precision: str
benchmark_limit: int
depthwise_kernel: str
<|endoftext|># mypy: allow-untyped-defs
import sys

import torch._C
import torch.cuda
from torch.backends import (
    _get_fp32_precision_getter,
    _set_fp32_precision_setter,
    PropModule,
)


try:
    from torch._C import _cudnn
except ImportError:
    # Uses of all the functions below should be guarded by torch.backends.cudnn.is_available(),
    # so it's safe to not emit any checks here.
    _cudnn = None  # type: ignore[assignment]


def get_cudnn_mode(mode):
    if mode == "RNN_RELU":
        # pyrefly: ignore [missing-attribute]
        return int(_cudnn.RNNMode.rnn_relu)
    elif mode == "RNN_TANH":
        # pyrefly: ignore [missing-attribute]
        return int(_cudnn.RNNMode.rnn_tanh)
    elif mode == "LSTM":
        # pyrefly: ignore [missing-attribute]
        return int(_cudnn.RNNMode.lstm)
    elif mode == "GRU":
        # pyrefly: ignore [missing-attribute]
        return int(_cudnn.RNNMode.gru)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# NB: We don't actually need this class anymore (in fact, we could serialize the
# dropout state for even better reproducibility), but it is kept for backwards
# compatibility for old models.
class Unserializable:
    def __init__(self, inner):
        self.inner = inner

    def get(self):
        return self.inner

    def __getstate__(self):
        return "<unserializable>"

    def __setstate__(self, state):
        self.inner = None


# we would like to use ContextProp from backends here but the
# frozen flags appears to be overzealous
class ContextProp:
    def __init__(self, getter, setter):
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        return self.getter()

    def __set__(self, obj, val):
        self.setter(val)


def init_dropout_state(dropout, train, dropout_seed, dropout_state):
    dropout_desc_name = "desc_" + str(torch.cuda.current_device())
    dropout_p = dropout if train else 0
    if (dropout_desc_name not in dropout_state) or (
        dropout_state[dropout_desc_name].get() is None
    ):
        if dropout_p == 0:
            dropout_state[dropout_desc_name] = Unserializable(None)
        else:
            dropout_state[dropout_desc_name] = Unserializable(
                torch._cudnn_init_dropout_state(  # type: ignore[call-arg]
                    dropout_p,
                    train,
                    dropout_seed,
                    # pyrefly: ignore [unexpected-keyword]
                    self_ty=torch.uint8,
                    device=torch.device("cuda"),
                )
            )
    dropout_ts = dropout_state[dropout_desc_name].get()
    return dropout_ts


class CudnnRNNModule(PropModule):
    def __init__(self, m, name):
        super().__init__(m, name)
        self.m.Unserializable = Unserializable
        self.m.get_cudnn_mode = get_cudnn_mode
        self.m.init_dropout_state = init_dropout_state

    @staticmethod
    def init_dropout_state(dropout, train, dropout_seed, dropout_state):
        dropout_desc_name = "desc_" + str(torch.cuda.current_device())
        dropout_p = dropout if train else 0
        if (dropout_desc_name not in dropout_state) or (
            dropout_state[dropout_desc_name].get() is None
        ):
            if dropout_p == 0:
                dropout_state[dropout_desc_name] = Unserializable(None)
            else:
                dropout_state[dropout_desc_name] = Unserializable(
                    torch._cudnn_init_dropout_state(  # type: ignore[call-arg]
                        dropout_p,
                        train,
                        dropout_seed,
                        # pyrefly: ignore [unexpected-keyword]
                        self_ty=torch.uint8,
                        device=torch.device("cuda"),
                    )
                )
        dropout_ts = dropout_state[dropout_desc_name].get()
        return dropout_ts

    fp32_precision = ContextProp(
        _get_fp32_precision_getter("cuda", "rnn"),
        _set_fp32_precision_setter("cuda", "rnn"),
    )


sys.modules[__name__] = CudnnRNNModule(sys.modules[__name__], __name__)
<|endoftext|># Config options to enable/disable C++ kernel for nn.functional.MHA
# and nn.TransformerEncoder
import torch


_is_fastpath_enabled: bool = True


def get_fastpath_enabled() -> bool:
    """Returns whether fast path for TransformerEncoder and MultiHeadAttention
    is enabled, or ``True`` if jit is scripting.

    .. note::
        The fastpath might not be run even if ``get_fastpath_enabled`` returns
        ``True`` unless all conditions on inputs are met.
    """
    if not torch.jit.is_scripting():
        return _is_fastpath_enabled
    return True


def set_fastpath_enabled(value: bool) -> None:
    """Sets whether fast path is enabled"""
    global _is_fastpath_enabled
    _is_fastpath_enabled = value
<|endoftext|># mypy: allow-untyped-defs
import sys
import warnings
from contextlib import contextmanager
from functools import lru_cache as _lru_cache
from typing import Any

from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule


try:
    import opt_einsum as _opt_einsum  # type: ignore[import]
except ImportError:
    _opt_einsum = None


@_lru_cache
def is_available() -> bool:
    r"""Return a bool indicating if opt_einsum is currently available.

    You must install opt-einsum in order for torch to automatically optimize einsum. To
    make opt-einsum available, you can install it along with torch: ``pip install torch[opt-einsum]``
    or by itself: ``pip install opt-einsum``. If the package is installed, torch will import
    it automatically and use it accordingly. Use this function to check whether opt-einsum
    was installed and properly imported by torch.
    """
    return _opt_einsum is not None


def get_opt_einsum() -> Any:
    r"""Return the opt_einsum package if opt_einsum is currently available, else None."""
    return _opt_einsum


def _set_enabled(_enabled: bool) -> None:
    if not is_available() and _enabled:
        raise ValueError(
            f"opt_einsum is not available, so setting `enabled` to {_enabled} will not reap "
            "the benefits of calculating an optimal path for einsum. torch.einsum will "
            "fall back to contracting from left to right. To enable this optimal path "
            "calculation, please install opt-einsum."
        )
    global enabled
    enabled = _enabled


def _get_enabled() -> bool:
    return enabled


def _set_strategy(_strategy: str) -> None:
    if not is_available():
        raise ValueError(
            f"opt_einsum is not available, so setting `strategy` to {_strategy} will not be meaningful. "
            "torch.einsum will bypass path calculation and simply contract from left to right. "
            "Please install opt_einsum or unset `strategy`."
        )
    if not enabled:
        raise ValueError(
            f"opt_einsum is not enabled, so setting a `strategy` to {_strategy} will not be meaningful. "
            "torch.einsum will bypass path calculation and simply contract from left to right. "
            "Please set `enabled` to `True` as well or unset `strategy`."
        )
    if _strategy not in ["auto", "greedy", "optimal"]:
        raise ValueError(
            f"`strategy` must be one of the following: [auto, greedy, optimal] but is {_strategy}"
        )
    global strategy
    strategy = _strategy


def _get_strategy() -> str:
    # pyrefly: ignore [bad-return]
    return strategy


def set_flags(_enabled=None, _strategy=None):
    orig_flags = (enabled, None if not is_available() else strategy)
    if _enabled is not None:
        _set_enabled(_enabled)
    if _strategy is not None:
        _set_strategy(_strategy)
    return orig_flags


@contextmanager
def flags(enabled=None, strategy=None):
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(enabled, strategy)
    try:
        yield
    finally:
        # recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


# The magic here is to allow us to intercept code like this:
#
#   torch.backends.opt_einsum.enabled = True


class OptEinsumModule(PropModule):
    global enabled
    enabled = ContextProp(_get_enabled, _set_enabled)
    global strategy
    strategy = None
    if is_available():
        strategy = ContextProp(_get_strategy, _set_strategy)


# This is the sys.modules replacement trick, see
# https://stackoverflow.com/questions/2447353/getattr-on-a-module/7668273#7668273
sys.modules[__name__] = OptEinsumModule(sys.modules[__name__], __name__)

enabled = bool(is_available())
strategy = "auto" if is_available() else None
<|endoftext|># mypy: allow-untyped-defs
import copy
import logging
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Optional
from warnings import warn


log = logging.getLogger(__name__)

import torch
import torch.cuda
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import _ExperimentalConfig
from torch.autograd import (
    _disable_profiler,
    _enable_profiler,
    _kineto_step,
    _prepare_profiler,
    _ProfilerResult,
    _supported_activities,
    _toggle_collection_dynamic,
    DeviceType,
    kineto_available,
    ProfilerActivity,
    ProfilerConfig,
    ProfilerState,
)
from torch.autograd.profiler_util import (
    _filter_name,
    _filter_stack_entry,
    _rewrite_name,
    EventList,
    FunctionEvent,
    MEMORY_EVENT_NAME,
    MemRecordsAcc,
    OUT_OF_MEMORY_EVENT_NAME,
)
from torch.futures import Future


__all__ = [
    "profile",
    "record_function",
    "emit_itt",
    "emit_nvtx",
    "load_nvprof",
    "EnforceUnique",
    "parse_nvprof_trace",
    "KinetoStepTracker",
    "EventList",
    "FunctionEvent",
    "MemRecordsAcc",
]

try:
    # Available in Python >= 3.2
    from contextlib import ContextDecorator as _ContextDecorator
except ImportError:
    import functools

    class _ContextDecorator:  # type: ignore[no-redef]
        def __enter__(self):
            raise NotImplementedError

        def __exit__(self, exc_type, exc_val, exc_tb):
            raise NotImplementedError

        def __call__(self, func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return wrapped


# global python state - whether profiler is currently enabled
# useful for fast python checks to reduce latency
_is_profiler_enabled: bool = False


def _set_is_profiler_enabled(enable: bool):
    global _is_profiler_enabled
    _is_profiler_enabled = enable


def _run_on_profiler_start():
    _set_is_profiler_enabled(True)


def _run_on_profiler_stop():
    _set_is_profiler_enabled(False)


@dataclass
class _ProfilerStats:
    "Profiler timing and stats used by developers to catch issues/regressions"

    profiling_window_duration_sec: float = 0
    number_of_events: int = 0
    profiler_prepare_call_duration_us: int = 0
    profiler_enable_call_duration_us: int = 0
    profiler_disable_call_duration_us: int = 0
    parse_kineto_call_duration_us: int = 0
    function_events_build_tree_call_duration_us: int = 0


class profile:
    """Context manager that manages autograd profiler state and holds a summary of results.

    .. note::
        This is the backend, most people should use :mod:`torch.profiler` instead.

    Under the hood it just records events of functions being executed in C++ and
    exposes those events to Python. You can wrap any code into it and it will
    only report runtime of PyTorch functions.
    Note: profiler is thread local and is automatically propagated into the async tasks

    Args:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.

        use_device (str, optional): Enables timing of device events.
            Adds approximately 4us of overhead to each tensor operation when use cuda.
            The valid devices options are 'cuda', 'xpu', 'mtia' and 'privateuseone'.

        record_shapes (bool, optional): If shapes recording is set, information
            about input dimensions will be collected. This allows one to see which
            dimensions have been used under the hood and further group by them
            using prof.key_averages(group_by_input_shape=True). Please note that
            shape recording might skew your profiling data. It is recommended to
            use separate runs with and without shape recording to validate the timing.
            Most likely the skew will be negligible for bottom most events (in a case
            of nested function calls). But for higher level functions the total
            self cpu time might be artificially increased because of the shape
            collection.

        with_flops (bool, optional): If with_flops is set, the profiler will estimate
            the FLOPs (floating point operations) value using the operator's input shape.
            This allows one to estimate the hardware performance. Currently,
            this option only works for the matrix multiplication and 2D convolution operators.

        profile_memory (bool, optional): track tensor memory allocation/deallocation.

        with_stack (bool, optional): record source information (file and line number) for the ops.

        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B

            .. deprecated::
                ``with_modules`` is deprecated and will be removed in a future version.
                It only collects data for TorchScript models, which are themselves
                deprecated, and does nothing in eager mode. Use ``with_stack=True``,
                which records ``nn.Module`` events for eager models.

        use_kineto (bool, optional): experimental, enable profiling with Kineto profiler.

        use_cpu (bool, optional): profile CPU events; setting to ``False`` requires
            ``use_kineto=True`` and can be used to lower the overhead for GPU-only profiling.

        experimental_config (_ExperimentalConfig) : A set of experimental options
            used by profiler libraries like Kineto. Note, backward compatibility is not guaranteed.

        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles

        post_processing_timeout_s (float): Optional timeout in seconds for post-processing profiler
            results. In this context, post-processing happens after the profiling itself has finished.
            If specified, event parsing will stop after this duration and return partial results. Useful
            for handling large traces that may take too long to process.


    .. warning::
        Enabling memory profiling or source attribution incurs additional profiler
        overhead

    .. warning::
        This context managers should not be called recursively, i.e. no nested
        instances are allowed

    .. warning::
        Due to some CUDA multiprocessing limitations (see :ref:`multiprocessing-cuda-note`),
        one cannot use the profiler with ``use_device = 'cuda'`` to benchmark
        DataLoaders with ``num_workers > 0``. If you wish to benchmark data loading,
        please use ``use_device = None`` or ``num_workers = 0``.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        >>>     for _ in range(100):  # any normal python code, really!
        >>>         y = x ** 2
        >>>         y.backward()
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total   CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        mul                                  32.048ms         32.048ms         200
        pow                                  27.041ms         27.041ms         200
        PowBackward0                         9.727ms          55.483ms         100
        torch::autograd::AccumulateGrad      9.148ms          9.148ms          100
        torch::autograd::GraphRoot           691.816us        691.816us        100
        -----------------------------------  ---------------  ---------------  ---------------

    """

    def __init__(
        self,
        enabled=True,
        *,
        use_device=None,
        record_shapes=False,
        with_flops=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        use_kineto=False,
        use_cpu=True,
        experimental_config=None,
        acc_events=False,
        custom_trace_id_callback=None,
        post_processing_timeout_s: float | None = None,
        activity_filters: dict[ProfilerActivity, set[str]] | None = None,
    ):
        self.enabled: bool = enabled
        if not self.enabled:
            return
        self.use_device: str | None = use_device
        # TODO Consider changing _function_events into data structure with size cap
        self._function_events: EventList | None = None
        self._old_function_events: EventList | None = None
        # Function event processing is done lazily
        self._needs_processing = False
        self.entered = False
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.record_shapes |= self.with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        self.use_cpu = use_cpu
        self.acc_events = acc_events
        if experimental_config is None:
            experimental_config = _ExperimentalConfig()
        if experimental_config.trace_only and with_stack:
            warn(
                "trace_only=True is incompatible with with_stack=True "
                "(stack traces require event post-processing). "
                "Disabling trace_only."
            )
            experimental_config = copy.copy(experimental_config)
            experimental_config.trace_only = False
        if (
            experimental_config.profiler_metrics
            or experimental_config.profiler_measure_per_kernel
        ):
            warn(
                "profiler_metrics and profiler_measure_per_kernel are deprecated "
                "and ignored. These options will be removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )
        if with_modules:
            warn(
                "with_modules is deprecated and will be removed in a future release. "
                "It only collects data for TorchScript models, which are themselves "
                "deprecated, and does nothing in eager mode. Use with_stack=True, "
                "which records nn.Module events for eager models.",
                FutureWarning,
                stacklevel=2,
            )
        self.experimental_config = experimental_config
        self.kineto_results: _ProfilerResult | None = None
        self.profiling_start_time_ns = 0
        self.profiling_end_time_ns = 0
        self._stats = _ProfilerStats()
        self.custom_trace_id_callback = custom_trace_id_callback
        self.post_processing_timeout_s = post_processing_timeout_s
        self.activity_filters = activity_filters or {}
        self.trace_id = ""
        if not self.use_cpu:
            if not use_kineto:
                raise AssertionError(
                    "Device-only events supported only with Kineto (use_kineto=True)"
                )

        if self.use_device is not None:
            VALID_DEVICE_OPTIONS = ["cuda", "xpu", "mtia", "hpu"]
            if _get_privateuse1_backend_name() != "privateuseone":
                VALID_DEVICE_OPTIONS.append(_get_privateuse1_backend_name())
            if self.use_device not in VALID_DEVICE_OPTIONS:
                warn(
                    f"The {self.use_device} is not a valid device option.", stacklevel=2
                )
                self.use_device = None

            if self.use_device == "cuda" and not torch.cuda.is_available():
                warn("CUDA is not available, disabling CUDA profiling", stacklevel=2)
                self.use_device = None

            if self.use_device == "xpu" and not torch.xpu.is_available():
                warn("XPU is not available, disabling XPU profiling", stacklevel=2)
                self.use_device = None

            if self.use_device == "hpu" and not (
                hasattr(torch, "hpu") and torch.hpu.is_available()
            ):
                warn("HPU is not available, disabling HPU profiling", stacklevel=2)
                self.use_device = None

        self.kineto_activities = set()
        if self.use_cpu:
            self.kineto_activities.add(ProfilerActivity.CPU)

        self.profiler_kind = ProfilerState.KINETO
        if self.use_device == "cuda":
            if not use_kineto or ProfilerActivity.CUDA not in _supported_activities():
                if not self.use_cpu:
                    raise AssertionError("Legacy CUDA profiling requires use_cpu=True")
                self.profiler_kind = ProfilerState.KINETO_GPU_FALLBACK
            else:
                self.kineto_activities.add(ProfilerActivity.CUDA)
        elif self.use_device == "xpu":
            if not (use_kineto and ProfilerActivity.XPU in _supported_activities()):
                raise AssertionError(
                    "Legacy XPU profiling is not supported. Requires use_kineto=True on XPU devices."
                )
            self.kineto_activities.add(ProfilerActivity.XPU)
        elif self.use_device == "mtia":
            if not (use_kineto and ProfilerActivity.MTIA in _supported_activities()):
                raise AssertionError(
                    "Legacy MTIA profiling is not supported. Requires use_kineto=True on MTIA devices."
                )
            self.kineto_activities.add(ProfilerActivity.MTIA)
        elif self.use_device == "hpu":
            if not (use_kineto and ProfilerActivity.HPU in _supported_activities()):
                raise AssertionError(
                    "Legacy HPU profiling is not supported. Requires use_kineto=True on HPU devices."
                )
            self.kineto_activities.add(ProfilerActivity.HPU)
        elif self.use_device is not None and self.use_device != "privateuseone":
            if use_kineto:
                # Native tracing mode: use KINETO_PRIVATEUSE1 with registered IActivityProfiler
                self.profiler_kind = ProfilerState.KINETO_PRIVATEUSE1
                if ProfilerActivity.PrivateUse1 in _supported_activities():
                    self.kineto_activities.add(ProfilerActivity.PrivateUse1)
            else:
                # Marker-only mode: use fallback state
                if not self.use_cpu:
                    raise AssertionError(
                        "Legacy privateuse1 profiling requires use_cpu=True"
                    )
                self.profiler_kind = ProfilerState.KINETO_PRIVATEUSE1_FALLBACK

        if len(self.kineto_activities) == 0:
            raise AssertionError("No activities specified for the profiler")

        if (
            self.post_processing_timeout_s is not None
            and self.post_processing_timeout_s < 0
        ):
            raise ValueError("post_processing_timeout_s must be non-negative")

    def default_trace_id(self):
        # Generate a UUID
        uuid_raw = uuid.uuid4()

        return f"{uuid_raw.int:032X}"

    def create_trace_id(self):
        if self.custom_trace_id_callback:
            return self.custom_trace_id_callback()
        return self.default_trace_id()

    def config(self, create_trace_id=False):
        # only need to generate new trace id upon prepare trace not start trace
        if create_trace_id:
            trace_id = self.create_trace_id()
            self.trace_id = trace_id
        return ProfilerConfig(
            self.profiler_kind,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops,
            self.with_modules,
            self.experimental_config,
            self.trace_id,
        )

    def __enter__(self):
        if not self.enabled:
            return
        if self.entered:
            raise RuntimeError("Profiler context manager is not reentrant")
        self._prepare_trace()
        self._start_trace()
        return self

    def _prepare_trace(self):
        self.entered = True
        t0 = perf_counter_ns()
        _prepare_profiler(
            self.config(create_trace_id=True),
            self.kineto_activities,
            activity_filter=self.activity_filters,
        )
        t1 = perf_counter_ns()
        self._stats.profiler_prepare_call_duration_us = int((t1 - t0) / 1000)

    def _start_trace(self):
        self.entered = True
        _run_on_profiler_start()
        t0 = perf_counter_ns()
        _enable_profiler(self.config(create_trace_id=False), self.kineto_activities)
        t1 = perf_counter_ns()
        self._stats.profiler_enable_call_duration_us = int((t1 - t0) / 1000)
        self.profiling_start_time_ns = t1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        if self.use_device and hasattr(torch, self.use_device):
            device_module = getattr(torch, self.use_device)
            if hasattr(device_module, "synchronize"):
                device_module.synchronize()

        if self._function_events and self.acc_events:
            self._old_function_events = self._function_events
        self._function_events = None
        self._needs_processing = True

        t0 = perf_counter_ns()

        self.kineto_results = _disable_profiler()
        t1 = perf_counter_ns()
        self._stats.profiler_disable_call_duration_us = int((t1 - t0) / 1000)
        self.profiling_end_time_ns = t0

        _run_on_profiler_stop()

        self._stats.profiling_window_duration_sec = (
            (self.profiling_end_time_ns - self.profiling_start_time_ns) * 1.0 / 1e9
        )

        # If we plan to accumulate events we should post process the function events
        # right away to retain the state across multiple start/stop calls
        if self.acc_events and not self.experimental_config.trace_only:
            self._ensure_function_events()
        return False

    def __repr__(self):
        if self._needs_processing:
            self._ensure_function_events()
        if self._function_events is None:
            return "<unfinished torch.autograd.profile>"
        return repr(self._function_events)

    def __str__(self):
        if self._needs_processing:
            self._ensure_function_events()
        if self._function_events is None:
            return "<unfinished torch.autograd.profile>"
        return str(self._function_events)

    def _ensure_function_events(self):
        """Process function events lazily if required"""
        if self.experimental_config.trace_only:
            raise RuntimeError(
                "events() is not available when trace_only=True in "
                "ExperimentalConfig. Use export_chrome_trace() instead."
            )
        if self._function_events is not None:
            return
        self._needs_processing = False

        t0 = perf_counter_ns()
        parsed_results = []
        if self.kineto_results:
            parsed_results = self._parse_kineto_results(
                self.kineto_results, timeout_s=self.post_processing_timeout_s
            )
        t1 = perf_counter_ns()
        self._stats.parse_kineto_call_duration_us = int((t1 - t0) / 1000)

        self._function_events = EventList(
            parsed_results,
            use_device=self.use_device,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        )
        t0 = perf_counter_ns()
        self._function_events._build_tree()
        t1 = perf_counter_ns()
        self._stats.function_events_build_tree_call_duration_us = int((t1 - t0) / 1000)
        self._stats.number_of_events = len(self._function_events)

        if self._old_function_events and self.acc_events:
            for evt in self._old_function_events:
                self._function_events.append(evt)
            self._old_function_events = None

    @property
    def function_events(self):
        if self._function_events is None or self._needs_processing:
            self._ensure_function_events()
        return self._function_events

    def table(
        self,
        sort_by=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        header=None,
        top_level_events_only=False,
    ):
        self._ensure_function_events()
        if self._function_events is None:
            raise AssertionError("Expected profiling results")
        return self._function_events.table(
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            top_level_events_only=top_level_events_only,
        )

    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path, metadata=None, use_python_export=False):
        """
        Exports the collected trace in Chrome JSON format. If kineto is enabled, only
        last cycle in schedule is exported.
        """
        if use_python_export and kineto_available():
            from torch.profiler._chrome_trace_export import (
                export_chrome_trace as _export,
            )

            _export(self.kineto_results, path, metadata)  # type: ignore[union-attr]
        elif kineto_available():
            self.kineto_results.save(path)  # type: ignore[union-attr]
        else:
            self._ensure_function_events()
            return self._function_events.export_chrome_trace(path)  # type: ignore[union-attr]

    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        self._ensure_function_events()
        if self._function_events is None:
            raise AssertionError("Expected profiling results")
        if not self.with_stack:
            raise AssertionError("export_stacks() requires with_stack=True")
        return self._function_events.export_stacks(path, metric)

    def toggle_collection_dynamic(
        self, enabled: bool, activities: Iterable[ProfilerActivity]
    ):
        """
        Toggles the collection of activities for the current profiler instance.
        """
        return _toggle_collection_dynamic(enabled, set(activities))

    def key_averages(
        self,
        group_by_input_shape=False,
        group_by_stack_n=0,
        group_by_overload_name=False,
        include_python_functions=False,
    ):
        self._ensure_function_events()
        if self._function_events is None:
            raise AssertionError("Expected profiling results")
        return self._function_events.key_averages(
            group_by_input_shape,
            group_by_stack_n,
            group_by_overload_name,
            include_python_functions,
        )

    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._ensure_function_events()
        if self._function_events is None:
            raise AssertionError("Expected profiling results")
        return self._function_events.total_average()

    total_average.__doc__ = EventList.total_average.__doc__

    @property
    def self_cpu_time_total(self):
        """Returns total time spent on CPU.

        The total time is a sum of all self times across all the events.
        """
        self._ensure_function_events()
        if self._function_events is None:
            raise AssertionError("Expected profiling results")
        return self._function_events.self_cpu_time_total

    def _parse_kineto_results(
        self, result: _ProfilerResult, timeout_s: float | None = None
    ):
        # result.events() has most of the events - PyTorch op-level and device-level events

        timeout_ns = int(timeout_s * 1e9) if timeout_s is not None else None
        result_events = result.events()
        if timeout_ns is not None and timeout_ns < 0:
            raise ValueError("timeout_s must be non-negative")
        start_time_ns = perf_counter_ns()
        timed_out = False

        def _check_timeout() -> bool:
            """Check if timeout has been exceeded. Returns True if timed out."""
            nonlocal timed_out
            if timeout_ns is not None and not timed_out:
                elapsed_ns = perf_counter_ns() - start_time_ns
                if elapsed_ns >= timeout_ns:
                    timed_out = True
            return timed_out

        trace_start_ns = result.trace_start_ns()
        mem_records = [
            [evt, False] for evt in result_events if evt.name() == MEMORY_EVENT_NAME
        ]
        oom_records = [
            evt for evt in result_events if evt.name() == OUT_OF_MEMORY_EVENT_NAME
        ]
        mem_records_acc = MemRecordsAcc(mem_records)

        def _cpu_memory_usage(mem_record):
            return (
                mem_record.nbytes()
                if mem_record.device_type()
                in [DeviceType.CPU, DeviceType.MKLDNN, DeviceType.IDEEP]
                else 0
            )

        def _device_memory_usage(mem_record):
            return (
                mem_record.nbytes()
                if mem_record.device_type()
                in [
                    DeviceType.CUDA,
                    DeviceType.PrivateUse1,
                    DeviceType.HIP,
                    DeviceType.XPU,
                ]
                else 0
            )

        # Create and return FunctionEvent list, which contains all function events
        # Here 2 function events are created:
        # all_function_events contains all events associated with each kineto event from result
        all_function_events = []
        # frontend_function_events contains the events in aten or torch frontend level,
        # whose correlation id is 0
        frontend_function_events = []
        device_corr_map: dict[int, list[FunctionEvent]] = {}
        max_evt_id = 0
        for kineto_event in result_events:
            if _check_timeout():
                break

            if (
                _filter_name(kineto_event.name())
                or getattr(kineto_event, "is_hidden_event", lambda: False)()
            ):
                continue
            rel_start_ns = kineto_event.start_ns() - trace_start_ns
            rel_end_ns = kineto_event.end_ns() - trace_start_ns

            cpu_memory_usage = 0
            device_memory_usage = 0
            if kineto_event.device_type() == DeviceType.CPU:
                # find the corresponding memory allocation events
                for mem_record in mem_records_acc.in_interval(
                    kineto_event.start_ns(), kineto_event.end_ns()
                ):
                    cpu_memory_usage += _cpu_memory_usage(mem_record[0])
                    device_memory_usage += _device_memory_usage(mem_record[0])
                    mem_record[1] = True

            is_async = kineto_event.is_async() or (
                kineto_event.start_thread_id() != kineto_event.end_thread_id()
            )

            fe = FunctionEvent(
                id=kineto_event.correlation_id(),
                name=_rewrite_name(name=kineto_event.name(), with_wildcard=True),
                overload_name=kineto_event.overload_name(),
                trace_name=_rewrite_name(name=kineto_event.name(), with_wildcard=False),
                thread=kineto_event.start_thread_id(),
                start_us=rel_start_ns / 1000,
                end_us=rel_end_ns / 1000,
                fwd_thread=kineto_event.fwd_thread_id(),
                input_shapes=kineto_event.shapes(),
                concrete_inputs=kineto_event.concrete_inputs(),
                kwinputs=kineto_event.kwinputs(),
                stack=[
                    entry
                    for entry in kineto_event.stack()
                    if _filter_stack_entry(entry)
                ],
                scope=kineto_event.scope(),
                use_device=self.use_device,
                cpu_memory_usage=cpu_memory_usage,
                device_memory_usage=device_memory_usage,
                is_async=is_async,
                sequence_nr=kineto_event.sequence_nr(),
                device_type=kineto_event.device_type(),
                device_index=kineto_event.device_index(),
                device_resource_id=kineto_event.device_resource_id(),
                flops=kineto_event.flops(),
                is_user_annotation=kineto_event.is_user_annotation(),
                is_python_function=kineto_event.is_python_function(),
                activity_type=kineto_event.activity_type(),
                metadata_json=kineto_event.metadata_json(),
                extra_meta=kineto_event.extra_meta() or None,
                typed_metadata=kineto_event.typed_metadata() or None,
                flow_id=kineto_event.flow_id(),
                flow_type=kineto_event.flow_type(),
                flow_start=kineto_event.flow_start(),
                external_id=kineto_event.external_id(),
                linked_correlation_id=kineto_event.linked_correlation_id(),
                structured_input_shapes=kineto_event.structured_input_shapes(),
                structured_input_strides=kineto_event.structured_input_strides(),
                input_dtypes=kineto_event.dtypes(),
                python_id=kineto_event.python_id(),
                python_parent_id=kineto_event.python_parent_id(),
                python_module_id=kineto_event.python_module_id(),
            )
            max_evt_id = max(max_evt_id, fe.id)
            if fe.device_type == DeviceType.CPU and not fe.is_async:
                if self.use_device == _get_privateuse1_backend_name():
                    privateuse1_time = kineto_event.privateuse1_elapsed_us()
                    if privateuse1_time > 0:
                        fe.append_kernel(fe.name, fe.device_index, privateuse1_time)
                        fe.is_legacy = True
                elif self.use_device == "cuda":
                    # Check if we have CUDA time as a fallback
                    cuda_time = kineto_event.cuda_elapsed_us()
                    if cuda_time > 0:
                        fe.append_kernel(fe.name, fe.device_index, cuda_time)
                        fe.is_legacy = True
            all_function_events.append(fe)
            corr_id = kineto_event.linked_correlation_id()
            if corr_id > 0:
                if corr_id not in device_corr_map:
                    device_corr_map[corr_id] = []
                device_corr_map[corr_id].append(fe)
            elif corr_id == 0:
                # Skip OVERHEAD events (profiler-internal host cost):
                # they do no device work and would otherwise inflate reported device time.
                if fe.activity_type != "overhead":
                    frontend_function_events.append(fe)
            else:
                raise RuntimeError(
                    f"Got negative correlation id {corr_id} in profiler post processing"
                )

        # associate device kernels and device runtime (CPU) with CPU events
        for fe in frontend_function_events:
            if (
                fe.device_type == DeviceType.CPU
                and not fe.is_async
                and fe.id in device_corr_map
            ):
                for f_evt in device_corr_map[fe.id]:
                    if f_evt.device_type in [
                        DeviceType.CUDA,
                        DeviceType.PrivateUse1,
                        DeviceType.XPU,
                    ]:
                        fe.append_kernel(
                            f_evt.name,
                            f_evt.device_index,
