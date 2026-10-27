            # extern_semantic_key precedence:
            # 1. Explicit annotation (e.g., FP8 bridge) — highest priority
            # 2. Single-element pre_grad_nodes for externs — derived identity
            # 3. None
            info = _inductor_kernel_extern_info.get(kernel_key)
            if info is not None and info.semantic_key is not None:
                extern_semantic_key = info.semantic_key
            elif info is not None and info.is_extern and len(pre_grad_list) == 1:
                extern_semantic_key = pre_grad_list[0]
            else:
                extern_semantic_key = None

            result[kernel_key] = {
                "stack_traces": stack_traces,
                "post_grad_nodes": post_grad_nodes,
                "pre_grad_nodes": pre_grad_list,
                "extern_semantic_key": extern_semantic_key,
            }

            # Merge shape metadata if available (extern ops only)
            if info is not None and info.shapes:
                result[kernel_key].update(info.shapes)

        return result
    except Exception as e:
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "create_kernel_information_json",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        log.warning("create_kernel_information_json: exception: %s", e)
        return {}


def set_kernel_post_grad_provenance_tracing(
    node_schedule: Sequence[BaseSchedulerNode] | ExternKernel,
    kernel_name: str,
    is_extern: bool = False,
) -> int | None:
    """
    Set the mapping between `kernel_name` and the post_grad nodes in `node_schedule`.

    Returns a unique int debug handler for each call to this function.
    """

    if config.effective_provenance_tracking_level() == 0:
        return None

    try:
        from .codegen.simd_kernel_features import DisableReduction, EnableReduction

        global _inductor_triton_kernel_to_post_grad_node_info
        global _inductor_kernel_stack_trace
        global _inductor_kernel_provenance_debug_handle
        global _inductor_kernel_extern_info

        _inductor_kernel_provenance_debug_handle += 1
        stack_traces: list[str] = []
        kernel_name = f"{kernel_name}:{_inductor_kernel_provenance_debug_handle}"
        if is_extern:
            if not isinstance(node_schedule, ExternKernel):
                raise AssertionError(
                    f"expected ExternKernel, got {type(node_schedule)}"
                )
            extern_info = _inductor_kernel_extern_info.setdefault(
                kernel_name, _KernelExternInfo()
            )
            extern_info.is_extern = True
            curr_node_info = _inductor_triton_kernel_to_post_grad_node_info.setdefault(
                kernel_name, []
            )
            # 'origins' on IR nodes gives what FX IR nodes contributed to any given fused kernel.
            # "origin_node" is more precise and says that the contents of this node corresponds
            # EXACTLY to the output of a particular FX node, but it's not always available
            if node_schedule.origin_node:
                origin_node_name = node_schedule.origin_node.name
                if origin_node_name not in curr_node_info:
                    curr_node_info.append(origin_node_name)
            else:
                curr_node_info.extend(
                    origin.name
                    for origin in node_schedule.origins
                    if origin.name not in curr_node_info
                )
            # Extract extern_semantic_key if stamped by a transform pass (e.g., FP8)
            if node_schedule.origin_node and "extern_semantic_key" in getattr(
                node_schedule.origin_node, "meta", {}
            ):
                extern_info.semantic_key = node_schedule.origin_node.meta[
                    "extern_semantic_key"
                ]
            else:
                for origin in node_schedule.origins:
                    if "extern_semantic_key" in getattr(origin, "meta", {}):
                        extern_info.semantic_key = origin.meta["extern_semantic_key"]
                        break
            # Extract input/output shapes and dtypes for extern ops
            try:
                input_shapes: list[list[int | str]] = []
                input_dtypes: list[str] = []
                for inp in tree_leaves(node_schedule.inputs):
                    if isinstance(inp, ir.IRNode):
                        metadata = _get_tensor_metadata(inp)
                        if metadata is None:
                            continue
                        shape, dtype = metadata
                        input_shapes.append(shape)
                        input_dtypes.append(dtype)

                shape_metadata: dict[str, Any] = {
                    "input_shapes": input_shapes,
                    "input_dtypes": input_dtypes,
                }
                output_metadata = _get_tensor_metadata(node_schedule)
                if output_metadata is not None:
                    output_shape, output_dtype = output_metadata
                    shape_metadata.update(
                        {
                            "output_shape": output_shape,
                            "output_dtype": output_dtype,
                        }
                    )
                extern_info.shapes = shape_metadata
            except Exception as e:
                log.debug("Shape extraction failed for %s: %s", kernel_name, e)
            stack_traces = list(node_schedule.get_stack_traces())
        else:
            if not isinstance(node_schedule, list):
                raise AssertionError(f"expected list, got {type(node_schedule)}")
            stack_traces_set: OrderedSet[str] = OrderedSet()
            for snode in node_schedule:
                if snode not in (EnableReduction, DisableReduction):
                    if snode.node is not None:
                        curr_node_info = (
                            _inductor_triton_kernel_to_post_grad_node_info.setdefault(
                                kernel_name, []
                            )
                        )
                        # pyrefly: ignore [missing-attribute]
                        stack_traces_set.update(snode.node.get_stack_traces())
                        curr_node_info.extend(
                            origin.name
                            # pyrefly: ignore [missing-attribute]
                            for origin in snode.node.origins
                            if origin.name not in curr_node_info
                        )
            stack_traces = list(stack_traces_set)
        _inductor_kernel_stack_trace.setdefault(kernel_name, []).extend(stack_traces)
        return _inductor_kernel_provenance_debug_handle
    except Exception as e:
        # Since this is just debugging, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "set_kernel_post_grad_provenance_tracing",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        return None


def save_args_for_compile_fx_inner(*args: Any, **kwargs: Any) -> None:
    """
    This function is used to save arguments for a compile_fx_inner function call
    to the file system.  Later on one can replay the compile_fx_inner call
    with the saved arguments using load_args_and_run_compile_fx_inner.
    """

    folder = os.path.join(tempfile.gettempdir(), "inductor_saved_args")
    if not os.path.exists(folder):
        os.mkdir(folder)

    def python_value(value: Any) -> int | float | bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value
        # SymPy numbers expose is_integer as a tri-state property.
        if getattr(value, "is_integer", None) is False:
            return float(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return float(value)

    def handle_sym_expr(x: Any, pytype: str) -> SymExprMetadataHolder:
        node = x.node
        shape_env = node.shape_env
        symbol_hints: dict[str, int | float | bool] = {}
        if shape_env is not None:
            for symbol in node.expr.free_symbols:
                hint = shape_env.backed_var_to_val.get(symbol)
                if hint is not None:
                    python_hint = python_value(hint)
                    if python_hint is not None:
                        symbol_hints[str(symbol)] = python_hint
        return SymExprMetadataHolder(
            pytype,
            node.expr,
            python_value(node.hint),
            symbol_hints,
        )

    def handle_tensor(x: Any) -> Any:
        """
        Pickle FakeTensor/SymInt will result in errors like:
        AttributeError: Can't pickle local object 'WeakValueDictionary.__init__.<locals>.remove'

        Convert tensors and symbolic values to metadata. This may also make
        pickle faster.
        """
        if isinstance(x, GraphModule):
            gm = copy.deepcopy(x)
            for node in gm.graph.nodes:
                node.meta = tree_map(handle_tensor, node.meta)
            return gm
        if isinstance(x, torch.Tensor):
            return TensorMetadataHolder(
                tree_map(handle_tensor, _extract_tensor_metadata(x)), x.device
            )
        elif isinstance(x, torch.SymInt):
            return handle_sym_expr(x, "int")
        elif isinstance(x, torch.SymFloat):
            return handle_sym_expr(x, "float")
        elif isinstance(x, torch.SymBool):
            return handle_sym_expr(x, "bool")
        else:
            return x

    args_to_save, kwargs_to_save = tree_map(handle_tensor, (args, kwargs))

    fn_name = "compile_fx_inner"
    path = f"{folder}/{fn_name}_{next(save_args_cnt)}.pkl"
    with open(path, "wb") as f:
        pickle.dump((args_to_save, kwargs_to_save), f)

    if log.isEnabledFor(logging.DEBUG):
        message = f"""
Arguments for a compile_fx_inner call is saved to {path}. To replay the call,
run the following:

from torch._inductor.debug import load_args_and_run_compile_fx_inner
load_args_and_run_compile_fx_inner({path!r})
        """
        # call print rather than log.debug. log.debug will print message
        # prefix for each line which makes the code snippet harder to be
        # copied.
        # Not a big deal since the code is already been guarded by checking
        # the log level.
        print(message)


def load_args_and_run_compile_fx_inner(path: str) -> Any:
    from torch._dynamo.source import ConstantSource
    from torch._inductor.compile_fx import compile_fx_inner
    from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

    with open(path, "rb") as f:
        args, kwargs = pickle.load(f)

    shape_env = ShapeEnv()
    symbol_cache: dict[str, Any] = {}

    def restore_sym_expr(x: SymExprMetadataHolder) -> Any:
        expr = x.expr
        if isinstance(expr, str):
            import sympy

            import torch.utils._sympy.functions as sympy_functions

            expr = sympy.sympify(expr, locals=sympy_functions.__dict__)
        replacements = {}
        for symbol in expr.free_symbols:
            symbol_name = str(symbol)
            if symbol_name not in symbol_cache:
                hint = x.symbol_hints.get(symbol_name)
                if hint is None:
                    if x.hint is not None and expr == symbol:
                        hint = x.hint
                    else:
                        symbol_cache[symbol_name] = shape_env.create_unbacked_symint(
                            ConstantSource(symbol_name)
                        ).node.expr
                        replacements[symbol] = symbol_cache[symbol_name]
                        continue
                symbol_cache[symbol_name] = shape_env.create_symbol(
                    hint,
                    ConstantSource(symbol_name),
                    dynamic_dim=DimDynamic.DYNAMIC,
                    do_not_specialize_zero_one=True,
                )
            replacements[symbol] = symbol_cache[symbol_name]
        expr = expr.xreplace(replacements)
        if x.pytype == "int":
            return shape_env.create_symintnode(expr, hint=x.hint)
        elif x.pytype == "float":
            return shape_env.create_symfloatnode(expr, hint=x.hint)
        elif x.pytype == "bool":
            return shape_env.create_symboolnode(expr)
        else:
            raise AssertionError(f"Unexpected symbolic expression type: {x.pytype}")

    def handle_tensor(x: Any) -> Any:
        if isinstance(x, GraphModule):
            for node in x.graph.nodes:
                node.meta = tree_map(handle_tensor, node.meta)
            return x
        elif isinstance(x, TensorMetadataHolder):
            tensor_metadata = tree_map(handle_tensor, x.tensor_metadata)
            return torch._dynamo.testing.rand_strided(
                tensor_metadata.shape,
                tensor_metadata.stride,
                tensor_metadata.dtype,
                x.device,
            )
        elif isinstance(x, SymExprMetadataHolder):
            return restore_sym_expr(x)
        else:
            return x

    fake_mode = torch._subclasses.FakeTensorMode(
        allow_non_fake_inputs=True, shape_env=shape_env
    )
    with fake_mode, config.patch("save_args", False):
        args, kwargs = tree_map(handle_tensor, (args, kwargs))
        return compile_fx_inner(*args, **kwargs)


def aot_inductor_minifier_wrapper(
    func: Callable[..., str],
    exported_program: torch.export.ExportedProgram,
    *,
    inductor_configs: dict[str, Any],
    package_path: FileLike | None = None,
) -> str:
    from torch._dynamo.debug_utils import AccuracyError
    from torch._dynamo.repro.aoti import dump_to_minify
    from torch._inductor import config
    from torch._inductor.compile_fx import _aoti_flatten_inputs

    use_minifier = config.aot_inductor.dump_aoti_minifier

    gm = exported_program.module(check_guards=False)
    if not isinstance(gm, torch.fx.GraphModule):
        raise AssertionError(f"expected torch.fx.GraphModule, got {type(gm)}")

    args, kwargs = exported_program.example_inputs

    try:
        if use_minifier and config.aot_inductor.repro_level == 3:
            # Always dump the original module in case we have segfaults
            dump_to_minify(
                exported_program,
                "aot_inductor",
                options=inductor_configs,
            )
        if use_minifier and config.aot_inductor.repro_level == 4:
            # Check for accuracy
            # We will first flatten the inputs before compiling and checking for accuracy.
            # This is ok because we will flatten the inputs in the minifier anyway.
            gm_copy = copy.deepcopy(gm)
            example_inputs_copy = copy.deepcopy(exported_program.example_inputs)
            config_copy = copy.deepcopy(inductor_configs)
            flat_example_inputs, config_copy = _aoti_flatten_inputs(
                gm_copy,
                example_inputs_copy[0],
                example_inputs_copy[1],
                options=config_copy,
            )
            tuple_inputs = tuple(flat_example_inputs)
            flattened_ep = torch.export.export(gm_copy, tuple_inputs, strict=False)
            func(
                flattened_ep.module(check_guards=False),
                tuple_inputs,
                inductor_configs=config_copy,
                package_path=package_path,
                load_and_run=True,
                check_accuracy="accuracy",
            )

        return func(
            gm,
            args,
            kwargs,
            inductor_configs=inductor_configs,
            package_path=package_path,
            load_and_run=use_minifier,
        )
    except AccuracyError as e:
        dump_to_minify(
            exported_program,
            "aot_inductor_accuracy",
            command="minify",
            options=inductor_configs,
        )
        log.warning("Accuracy failed")
        raise e
    except Exception as e:
        if use_minifier:
            command = "minify"

            if config.aot_inductor.repro_level == 1:
                command = "run"

            dump_to_minify(
                exported_program,
                "aot_inductor",
                command=command,
                options=inductor_configs,
            )
        raise e
<|endoftext|># mypy: allow-untyped-defs
import functools

import torch
import torch.utils._pytree as pytree
from torch._inductor.kernel.mm_common import mm_args

from . import config, ir
from .codegen.cpp_gemm_template import CppGemmTemplate
from .codegen.cpp_grouped_gemm_template import CppGroupedGemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr
from .ir import TensorBox
from .lowering import (
    add,
    add_needs_realized_inputs,
    aten,
    permute,
    register_lowering,
    to_dtype,
    view,
)
from .select_algorithm import (
    autotune_select_algorithm,
    ChoiceCaller,
    ExternKernelChoice,
)
from .utils import use_aten_gemm_kernels, use_cpp_gemm_template
from .virtualized import ops, OpsValue, V


def create_int8_compensation(
    W_tensor: torch.Tensor,
    packed_weight: ir.TensorBox,
    x_scale: ir.TensorBox,
    x_zp: ir.TensorBox,
    w_scale: ir.TensorBox,
) -> tuple[
    bool,
    ir.TensorBox,
    ir.TensorBox | None,
]:
    x_w_scale: ir.TensorBox | None = None
    use_int8_fast_compensation_path = all(
        isinstance(item, ir.TensorBox)
        and item.get_name() in V.graph.constants
        and hasattr(item.data, "data")
        and isinstance(item.data.data, ir.ConstantBuffer)
        for item in [x_scale, x_zp, w_scale]
    )
    if use_int8_fast_compensation_path:
        x_w_scale_tensor = (
            V.graph.constants[x_scale.get_name()]
            * V.graph.constants[w_scale.get_name()]
        )
        x_w_scale = V.graph.add_tensor_constant(
            x_w_scale_tensor,
            name=packed_weight.get_name() + "_x_w_compens",
        )
        weight_compens_tensor = torch.sum(W_tensor.to(torch.float), dim=0)
        x_zp_tensor = V.graph.constants[x_zp.get_name()]
        weight_compens_tensor = weight_compens_tensor * x_w_scale_tensor * x_zp_tensor
        weight_compens = V.graph.add_tensor_constant(
            weight_compens_tensor,
            name=packed_weight.get_name() + "_BMatrixCompens",
        )
    else:
        weight_compens_tensor = torch.sum(W_tensor.to(torch.float), dim=0)
        weight_compens = V.graph.add_tensor_constant(
            weight_compens_tensor,
            name=packed_weight.get_name() + "_BMatrixCompens",
        )
    return (  # type: ignore[return-type]
        use_int8_fast_compensation_path,
        weight_compens,
        x_w_scale,
    )


def codegen_int8_gemm_template_compensation(
    use_int8_fast_compensation_path: bool,
    input: OpsValue,
    _weight_compo: OpsValue,
    _x_scale: OpsValue | None,
    _x_zp: OpsValue | None,
    _w_scale: OpsValue | None,
    _x_w_scale: OpsValue | None,
) -> OpsValue:
    if use_int8_fast_compensation_path:
        temp = ops.sub(
            ops.mul(
                input,
                _x_w_scale,
            ),
            _weight_compo,
        )
    else:
        temp = ops.mul(
            ops.mul(
                input,
                _x_scale,
            ),
            _w_scale,
        )
        # NOTE: We will apply compensation even if the x_zp is 0 for int8 quantization.
        # That's because when torch.compile is invoked for dynamic quantization,
        # x might coincidentally have such values that x_zp might be zero despite
        # asymmetric quantization.
        # Besides, if x_zp is dummy for int8 x, or if x is statically quantized,
        # we'd still perform that redundant compute to avoid making the code messy
        # because we discovered that redundant computation of compensation did not
        # lead to performance degradation with the input shapes tested.
        temp = ops.sub(
            temp,
            ops.mul(
                ops.mul(
                    ops.mul(
                        _x_scale,
                        _w_scale,
                    ),
                    _x_zp,
                ),
                _weight_compo,
            ),
        )
    return temp


def grouped_gemm_lowering(
    x: TensorBox,
    w: list[TensorBox],
    b: list[TensorBox],
    attr=None,
    scalars=None,
    algorithm=None,
    layout=None,
):
    x_size = x.get_size()
    if len(x_size) > 2:
        # GEMM template needs 2D input, normalize input shape here
        x = view(x, [-1, x_size[-1]])
    num_gemm = len(w)

    if not (config.max_autotune or config.max_autotune_gemm):
        raise AssertionError(
            "grouped_gemm_lowering requires max_autotune or max_autotune_gemm"
        )
    # pyrefly: ignore [bad-assignment]
    b = [bias if bias is None else ir.ExternKernel.realize_input(bias) for bias in b]

    choices: list[ChoiceCaller] = []
    *_, layout, x, _ = mm_args(x, permute(w[0], [1, 0]), layout=layout)

    kwargs = {
        "has_bias": [bias is not None for bias in b],
        "trans_w": True,
        "epilogue_creator": None,
        "act_mapping": dict.fromkeys(range(num_gemm), x),
    }

    input_nodes = [x, *w]
    input_nodes.extend([bias for bias in b if bias is not None])

    CppGroupedGemmTemplate.add_choices(
        choices,
        layout,
        input_nodes,
        **kwargs,  # type: ignore[arg-type]
    )

    if len(choices) == 0:
        raise AssertionError("expected at least one choice for grouped_gemm")
    result, _ = autotune_select_algorithm(
        "grouped_gemm",
        choices,
        input_nodes,
        layout,
    )
    template_buf = result.data.data
    return_bufs = [
        ir.MultiOutput(layout, template_buf, [(list, gemm_idx)])
        for gemm_idx in range(num_gemm)
    ]
    # pyrefly: ignore [bad-argument-type]
    template_buf.layout = ir.MultiOutputLayout(device=input_nodes[0].get_device())
    template_buf.outputs = return_bufs
    return_tensors = [
        ir.TensorBox.create(return_bufs[gemm_idx]) for gemm_idx in range(num_gemm)
    ]
    if len(x_size) > 2:
        for gemm_idx in range(num_gemm):
            return_tensors[gemm_idx] = view(
                return_tensors[gemm_idx],  # type: ignore[arg-type]
                (*x_size[:-1], return_tensors[gemm_idx].get_size()[-1]),
            )
    return return_tensors


grouped_gemm_lowering._inductor_lowering_function = True  # type: ignore[attr-defined]
grouped_gemm_lowering._inductor_lowering_output_metadata_ignores_input_storage = True  # type: ignore[attr-defined]


def _convert_to_0d_constant(
    tensor_box: ir.TensorBox,
    dtype: torch.dtype,
    name_suffix: str = "_0d",
) -> ir.TensorBox:
    """
    Normalize a tensor with all dimensions equal to 1 to a 0D ConstantBuffer.
    """
    if isinstance(tensor_box, ir.TensorBox):
        data = tensor_box.data
        if isinstance(data, ir.StorageBox):
            data = data.data
    else:
        data = tensor_box

    if isinstance(data, ir.ConstantBuffer):
        tensor = V.graph.constants.get(tensor_box.get_name())
        if tensor is not None:
            return V.graph.add_tensor_constant(
                tensor.reshape([]), name=tensor_box.get_name() + name_suffix
            )

    from torch._inductor.lowering import get_constant_value

    const_value = get_constant_value(data)
    if const_value is not None:
        return V.graph.add_tensor_constant(
            torch.tensor(const_value.value, dtype=dtype).reshape([])
        )

    tensor_box.realize()

    from .ir import ExternKernel, GenericView

    if isinstance(tensor_box.data, GenericView):
        tensor_box = ir.TensorBox(ExternKernel.require_contiguous(tensor_box.data))

    result = view(tensor_box, [])
    return result


def register_onednn_fusion_ops():
    if torch._C._has_mkldnn:
        from . import mkldnn_ir

        aten_mkldnn_linear_unary = ExternKernelChoice(
            torch.ops.mkldnn._linear_pointwise,
            "mkldnn::_linear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.LinearUnary.create,
        )
        aten_mkldnn_linear_binary = ExternKernelChoice(
            torch.ops.mkldnn._linear_pointwise.binary,
            "mkldnn::_linear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.LinearBinary.create,
        )
        aten_mkldnn_qlinear_unary = ExternKernelChoice(
            torch.ops.onednn.qlinear_pointwise,
            "onednn::qlinear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.QLinearPointwisePT2E.create,
        )
        aten_mkldnn_qlinear_binary = ExternKernelChoice(
            torch.ops.onednn.qlinear_pointwise.binary,
            "onednn::qlinear_pointwise",
            has_out_variant=False,
            kernel_creator=mkldnn_ir.QLinearPointwiseBinaryPT2E.create,
        )
        cpu_needs_realized_inputs: list[
            torch._ops.OpOverload | torch._ops.OpOverloadPacket
        ] = [
            torch.ops.mkldnn._convolution_pointwise,
            torch.ops.mkldnn._convolution_pointwise_,
            torch.ops.mkldnn._convolution_transpose_pointwise,
            torch.ops.mkldnn._linear_pointwise,
            aten.mkldnn_rnn_layer.default,
            torch.ops.onednn.qconv_pointwise,
        ]

        @register_lowering(torch.ops.mkldnn._convolution_pointwise)
        def convolution_unary(
            x: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            attr,
            scalars,
            algorithm,
        ):
            return TensorBox.create(
                mkldnn_ir.ConvolutionUnary.create(
                    x,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._convolution_pointwise.binary)
        def convolution_binary(
            x: TensorBox,
            other: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            return TensorBox.create(
                mkldnn_ir.ConvolutionBinary.create(
                    x,
                    other,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    binary_attr,
                    binary_alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._convolution_pointwise_.binary)
        def convolution_binary_inplace(
            x: TensorBox,
            other: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            stride,
            dilation,
            groups,
            binary_attr,
            binary_alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            return TensorBox.create(
                mkldnn_ir.ConvolutionBinaryInplace.create(
                    x,
                    other,
                    weight,
                    bias,
                    padding,
                    stride,
                    dilation,
                    groups,
                    binary_attr,
                    binary_alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.mkldnn._linear_pointwise)
        def linear_unary(
            x: TensorBox,
            w: TensorBox,
            b: TensorBox,
            attr,
            scalars,
            algorithm,
            layout=None,
        ):
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            if b is not None:
                b = ir.ExternKernel.realize_input(b)  # type: ignore[assignment]
            choices: list[ChoiceCaller] = []
            if config.max_autotune or config.max_autotune_gemm:
                transposed_w = permute(w, [1, 0])
                *_, layout, x, transposed_w = mm_args(x, transposed_w, layout=layout)
                if use_cpp_gemm_template(layout, x, transposed_w):

                    def epilogue_creator(buf):
                        return create_epilogue_with_attr(
                            buf, attr, scalars=scalars, algorithm=algorithm
                        )

                    kwargs = {
                        "has_bias": b is not None,
                        "trans_w": True,
                        "epilogue_creator": (
                            None if attr == "none" else epilogue_creator
                        ),
                    }
                    if b is not None:
                        kwargs["input_indices"] = [2, 0, 1]  # type: ignore[assignment]
                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, w] if b is None else [x, w, b],
                        **kwargs,  # type: ignore[arg-type]
                    )
            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(attr=attr, scalars=scalars, algorithm=algorithm)
                if b is None:
                    kwargs["B"] = None
                choices.append(
                    aten_mkldnn_linear_unary.bind(
                        [x, w] if b is None else [x, w, b],
                        layout,
                        **kwargs,
                    )
                )
            if w.get_name() not in V.graph.constants:
                raise AssertionError("weight must be a graph constant")
            input_gen_fns = {
                1: lambda x: V.graph.constants[x.get_name()],
            }
            result, _ = autotune_select_algorithm(
                "linear_unary",
                choices,
                [x, w] if b is None else [x, w, b],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

        @register_lowering(torch.ops.mkldnn._linear_pointwise.binary)
        def linear_binary(
            x: TensorBox, y: TensorBox, w: TensorBox, b: TensorBox, attr, layout=None
        ):
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            y_size = y.get_size()
            if len(y_size) > 2:
                y = view(y, [-1, y_size[-1]])
            if b is not None:
                b = ir.ExternKernel.realize_input(b)  # type: ignore[assignment]
            choices: list[ChoiceCaller] = []
            if config.max_autotune or config.max_autotune_gemm:
                transposed_w = permute(w, [1, 0])
                *_, layout, x, transposed_w, y = mm_args(
                    x, transposed_w, y, layout=layout
                )
                if use_cpp_gemm_template(layout, x, transposed_w):

                    def epilogue_creator(buf):
                        return create_epilogue_with_attr(buf, attr, other=y)

                    kwargs = {
                        "has_bias": b is not None,
                        "trans_w": True,
                        "epilogue_creator": epilogue_creator,
                    }

                    # pyrefly: ignore [bad-typed-dict-key, unsupported-operation]
                    kwargs["input_indices"] = [0, 2, 1] if b is None else [3, 0, 2, 1]
                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, y, w] if b is None else [x, y, w, b],
                        **kwargs,  # type: ignore[arg-type]
                    )
            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(attr=attr)
                if b is None:
                    kwargs["B"] = None
                choices.append(
                    aten_mkldnn_linear_binary.bind(
                        [x, y, w] if b is None else [x, y, w, b],
                        layout,
                        **kwargs,
                    )
                )
            if w.get_name() not in V.graph.constants:
                raise AssertionError("weight must be a graph constant")
            input_gen_fns = {
                2: lambda x: V.graph.constants[x.get_name()],
            }
            result, _ = autotune_select_algorithm(
                "linear_binary",
                choices,
                [x, y, w] if b is None else [x, y, w, b],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

        @register_lowering(torch.ops.mkldnn._convolution_transpose_pointwise)
        def convolution_transpose_unary(
            x: TensorBox,
            weight: TensorBox,
            bias: TensorBox,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            attr,
            scalars,
            algorithm,
        ):
            return TensorBox.create(
                mkldnn_ir.ConvolutionTransposeUnary.create(
                    x,
                    weight,
                    bias,
                    padding,
                    output_padding,
                    stride,
                    dilation,
                    groups,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(aten.mkldnn_rnn_layer.default)
        def mkldnn_rnn_layer(
            x: TensorBox,
            w0: TensorBox,
            w1: TensorBox,
            w2: TensorBox,
            w3: TensorBox,
            hx: TensorBox,
            cx: TensorBox,
            reverse: bool,
            batch_sizes: list[int],
            mode: int,
            hidden_size: int,
            num_layers: int,
            has_biases: bool,
            bidirectional: bool,
            batch_first: bool,
            train: bool,
        ):
            return pytree.tree_map(
                TensorBox.create,
                mkldnn_ir.MkldnnRnnLayer.create(
                    x,
                    w0,
                    w1,
                    w2,
                    w3,
                    hx,
                    cx,
                    reverse,
                    batch_sizes,
                    mode,
                    hidden_size,
                    num_layers,
                    has_biases,
                    bidirectional,
                    batch_first,
                    train,
                ),
            )

        @register_lowering(torch.ops.onednn.qconv_pointwise, type_promotion_kind=None)
        def qconvolution_unary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp,
            bias: TensorBox,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            attr,
            scalars,
            algorithm,
        ):
            if not isinstance(x_scale, ir.TensorBox):
                if type(x_scale) is not float:
                    raise AssertionError(
                        f"expected x_scale to be float, got {type(x_scale)}"
                    )
                x_scale = V.graph.add_tensor_constant(
                    torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
                )

            if x_zp is None:
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="x_zp"
                )
            if not isinstance(x_zp, ir.TensorBox):
                if type(x_zp) is not int:
                    raise AssertionError(f"expected x_zp to be int, got {type(x_zp)}")
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
                )

            if w_zp is None:
                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="w_zp"
                )

            return TensorBox.create(
                mkldnn_ir.QConvPointWisePT2E.create(
                    x,
                    x_scale,
                    x_zp,
                    packed_weight,
                    w_scale,
                    w_zp,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    attr,
                    scalars,
                    algorithm,
                )
            )

        @register_lowering(
            torch.ops.onednn.qconv2d_pointwise.binary, type_promotion_kind=None
        )
        @register_lowering(
            torch.ops.onednn.qconv2d_pointwise.binary_tensor, type_promotion_kind=None
        )
        def qconvolution_binary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp,
            accum: TensorBox,
            bias: TensorBox,
            stride,
            padding,
            dilation,
            groups,
            o_inv_scale,
            o_zero_point,
            output_dtype,
            accum_scale,
            accum_zp,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
        ):
            if not isinstance(x_scale, ir.TensorBox):
                if type(x_scale) is not float:
                    raise AssertionError(
                        f"expected x_scale to be float, got {type(x_scale)}"
                    )
                x_scale = V.graph.add_tensor_constant(
                    torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
                )

            if x_zp is None:
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="x_zp"
                )
            if not isinstance(x_zp, ir.TensorBox):
                if type(x_zp) is not int:
                    raise AssertionError(f"expected x_zp to be int, got {type(x_zp)}")
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
                )

            if w_zp is None:
                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="w_zp"
                )

            if (
                binary_attr == "sum"
                and output_dtype in [torch.float32, torch.bfloat16]
                and accum.get_dtype() in [torch.float32, torch.bfloat16]
                and accum.get_dtype() != output_dtype
            ):
                # For int8-mixed-bf16 quantization and inplace add,
                # there is case when accum dtype is float32 but output dtype is bfloat16.
                # Since the accum will be changed in-place with post op sum,
                # we will do accum dtype conversion here.
                accum = to_dtype(accum, output_dtype)
            return TensorBox.create(
                mkldnn_ir.QConvPointWiseBinaryPT2E.create(
                    x,
                    x_scale,  # type: ignore[arg-type]
                    x_zp,  # type: ignore[arg-type]
                    packed_weight,
                    w_scale,
                    w_zp,
                    accum,
                    bias,
                    stride,
                    padding,
                    dilation,
                    groups,
                    o_inv_scale,
                    o_zero_point,
                    output_dtype,
                    accum_scale,
                    accum_zp,
                    binary_attr,
                    alpha,
                    unary_attr,
                    unary_scalars,
                    unary_algorithm,
                )
            )

        @register_lowering(torch.ops.onednn.qlinear_pointwise, type_promotion_kind=None)
        def qlinear_unary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp: TensorBox,
            bias: TensorBox,
            o_scale,
            o_zero_point,
            output_dtype,
            attr,
            scalars,
            algorithm,
            layout=None,
        ):
            if packed_weight.get_dtype() not in [torch.int8, torch.float8_e4m3fn]:
                raise AssertionError(
                    "Only int8 and e4m3fn weights are supported by oneDNN qlinear."
                )
            x_size = x.get_size()
            if len(x_size) > 2:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
            if not isinstance(x_scale, ir.TensorBox):
                if type(x_scale) is not float:
                    raise AssertionError(
                        f"expected x_scale to be float, got {type(x_scale)}"
                    )
                x_scale = V.graph.add_tensor_constant(
                    torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
                )
            else:
                x_scale_size = x_scale.get_size()
                if len(x_scale_size) == 0 or all(dim == 1 for dim in x_scale_size):
                    x_scale = _convert_to_0d_constant(x_scale, torch.float32, "_0d")
                else:
                    x_scale.realize()
                if len(x_scale.get_size()) not in [0, 1]:
                    raise AssertionError("x_scale must be 0D or 1D")

            if x_zp is None:
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="x_zp"
                )
            elif not isinstance(x_zp, ir.TensorBox):
                if type(x_zp) is not int:
                    raise AssertionError(f"expected x_zp to be int, got {type(x_zp)}")
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
                )
            else:
                x_zp_size = x_zp.get_size()
                if len(x_zp_size) == 0 or all(dim == 1 for dim in x_zp_size):
                    # If all outer dims of x_zp are 1, make it a ConstantBuffer with 0D shape
                    # Don't call realize() before _convert_to_0d_constant to preserve ComputedBuffer
                    x_zp = _convert_to_0d_constant(x_zp, torch.int32, "_0d")
                else:
                    x_zp.realize()

            if x_zp.get_numel() != 1:
                raise AssertionError("x_zp is incompatible with oneDNN qlinear")

            # When channels less than 8, w_scale/w_zp is Pointwise instead of ConstantBuffer
            # Refer to
            # https://github.com/pytorch/pytorch/blob/f353d17755ed23b02924c962a86ff99a3405fe10/torch/_inductor/graph.py#L570-L577
            if w_zp is None:
                # If w_zp is None, then it's a dummy tensor created to denote the
                # absence of a zero point, and thus w is int8 symmetrically quantized.
                # Moreover, oneDNN qlinear API doesn't accept None value for zp

                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="w_zp"
                )
            w_scale.realize()
            w_zp.realize()
            if w_zp.get_dtype() != torch.int32 and isinstance(
                ir.InputsKernel.unwrap_storage_for_input(w_zp),
                ir.ConstantBuffer,
            ):
                # W_zp might be a ConstantBuffer with int64, convert it to int32
                w_zp_tensor = V.graph.constants[w_zp.get_name()].to(torch.int32)
                w_zp = V.graph.add_tensor_constant(  # type: ignore[assignment]
                    torch.tensor(w_zp_tensor, dtype=torch.int32), name=w_zp.get_name()
                )

            bias_dtype = None if bias is None else bias.get_dtype()
            choices: list[ChoiceCaller] = []

            if config.max_autotune or config.max_autotune_gemm:
                *_, layout, x, packed_weight = mm_args(
                    x, packed_weight, layout=layout, out_dtype=output_dtype
                )

                if (
                    # GEMM template currently only supports symmetrically quantized weights
                    isinstance(
                        ir.InputsKernel.unwrap_storage_for_input(w_zp),
                        ir.ConstantBuffer,
                    )
                    and torch.equal(
                        torch.zeros_like(V.graph.constants[w_zp.get_name()]),
                        V.graph.constants[w_zp.get_name()],
                    )
                ) and use_cpp_gemm_template(layout, x, packed_weight):
                    W_tensor = V.graph.constants[packed_weight.get_name()].to_dense()

                    (
                        use_int8_fast_compensation_path,
                        weight_compens,
                        x_w_scale,
                    ) = create_int8_compensation(
                        W_tensor,
                        packed_weight,
                        x_scale,
                        x_zp,
                        w_scale,
                    )

                    def epilogue_creator(input_buffer):
                        # Epilogue to convert from s32 to f32 for u8s8f32
                        if output_dtype not in [
                            torch.float32,
                            torch.bfloat16,
                            torch.uint8,
                            torch.int8,
                        ]:
                            raise AssertionError(
                                f"unexpected output_dtype {output_dtype}"
                            )
                        input_loader = input_buffer.make_loader()
                        weight_compens_loader = weight_compens.make_loader()
                        x_w_scale_loader = None
                        if use_int8_fast_compensation_path:
                            if x_w_scale is None:
                                raise AssertionError(
                                    "x_w_scale must be set on the fast compensation path"
                                )
                            x_w_scale_loader = x_w_scale.make_loader()
                        x_scale_loader = x_scale.make_loader()
                        w_scale_loader = w_scale.make_loader()
                        x_zp_loader = x_zp.make_loader()
                        nonlocal bias
                        bias_loader = None
                        if bias is not None:
                            bias_loader = bias.make_loader()

                        def inner_fn(index):
                            nonlocal bias
                            input = input_loader(index)
                            # MicroKernel Output is with int32
                            # cvt to FP32 before doing compensation
                            input = ops.to_dtype(input, torch.float32)
                            weight_compens_index = (index[-1],)

                            _x_scale = None
                            _x_zp = None
                            _w_scale = None
                            if not use_int8_fast_compensation_path:
                                _x_scale = x_scale_loader(())
                                _x_zp = x_zp_loader(())
                                _w_scale = w_scale_loader(weight_compens_index)
                            _weight_compo = weight_compens_loader(weight_compens_index)
                            _x_w_scale = None
                            if use_int8_fast_compensation_path:
                                if x_w_scale_loader is None:
                                    raise AssertionError(
                                        "x_w_scale_loader must be set on the fast compensation path"
                                    )
                                _x_w_scale = x_w_scale_loader(weight_compens_index)
                            # Step 1: Compute s8s8->s32 or u8s8->s32 GEMM & then apply compensation
                            temp = codegen_int8_gemm_template_compensation(
                                use_int8_fast_compensation_path,
                                input,
                                _weight_compo,
                                _x_scale,
                                _x_zp,
                                _w_scale,
                                _x_w_scale,
                            )
                            # Step 2: add Bias if applicable
                            if bias is not None:
                                # pyrefly: ignore [not-callable]
                                _bias = bias_loader(weight_compens_index)
                                nonlocal bias_dtype
                                if bias_dtype not in [torch.float32, torch.bfloat16]:
                                    raise AssertionError(
                                        f"unexpected bias_dtype {bias_dtype}"
                                    )
                                if bias_dtype == torch.bfloat16:
                                    _bias = ops.to_dtype(_bias, torch.float32)
                                temp = ops.add(temp, _bias)

                            return temp

                        output_buf = ir.Pointwise(
                            device=input_buffer.get_device(),
                            dtype=torch.float32,  # Hardcode to FP32 for u8s8f32 & s8s8f32
                            inner_fn=inner_fn,
                            ranges=input_buffer.get_size(),
                        )

                        # Step 3: Doing the unary post op fusion
                        if attr != "none":
                            output_buf = create_epilogue_with_attr(
                                output_buf, attr, scalars=scalars, algorithm=algorithm
                            )

                        # Step 4: Cast output to Target Dtype
                        if output_dtype == torch.bfloat16:
                            output_cast_loader = output_buf.make_loader()

                            def inner_fn_cast_output_to_bf16(index):
                                input = output_cast_loader(index)
                                return ops.to_dtype(input, output_dtype)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device_or_error(),
                                dtype=output_dtype,
                                inner_fn=inner_fn_cast_output_to_bf16,
                                ranges=output_buf.get_size(),
                            )
                        elif output_dtype in [torch.uint8, torch.int8]:
                            from .lowering import _create_constants

                            requant_input_loader = output_buf.make_loader()

                            def inner_fn_requant(index, scale, zero_point):
                                input = requant_input_loader(index)
                                inv_scale, zero_point = _create_constants(
                                    1.0 / scale, zero_point, dtype=torch.float32
                                )
                                val = ops.round(input * inv_scale) + zero_point
                                if output_dtype == torch.uint8:
                                    qmin, qmax = _create_constants(
                                        0, 255, dtype=torch.float32
                                    )
                                else:
                                    qmin, qmax = _create_constants(
                                        -128, 127, dtype=torch.float32
                                    )
                                clamped = ops.minimum(ops.maximum(val, qmin), qmax)
                                return ops.to_dtype(clamped, output_dtype)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device_or_error(),
                                dtype=output_dtype,
                                inner_fn=functools.partial(
                                    inner_fn_requant,
                                    scale=float(o_scale),
                                    zero_point=int(o_zero_point),
                                ),
                                ranges=output_buf.get_size(),
                            )

                        return output_buf

                    if x.get_dtype() not in [torch.uint8, torch.int8]:
                        raise AssertionError(f"unexpected x dtype {x.get_dtype()}")
                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, x_scale, x_zp, packed_weight, w_scale, w_zp]
                        if bias is None
                        else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias],
                        has_bias=bias is not None,
                        epilogue_creator=epilogue_creator,
                        input_indices=[0, 3, 1, 2, 4, 5]
                        if bias is None
                        else [6, 0, 3, 1, 2, 4, 5],
                    )
            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(
                    output_scale=o_scale,
                    output_zero_point=o_zero_point,
                    output_dtype=output_dtype,
                    post_op_name=attr,
                    post_op_args=scalars,
                    post_op_algorithm=algorithm,
                )
                if bias is None:
                    kwargs["bias"] = None
                choices.append(
                    aten_mkldnn_qlinear_unary.bind(
                        (x, x_scale, x_zp, packed_weight, w_scale, w_zp)
                        if bias is None
                        else (x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias),
                        layout,
                        **kwargs,
                    )
                )
            if packed_weight.get_name() not in V.graph.constants:
                raise AssertionError("packed_weight must be a graph constant")
            input_gen_fns = {
                3: lambda x: V.graph.constants[x.get_name()],  # packed weight
                4: lambda x: V.graph.constants[x.get_name()],  # weight scale
                5: lambda x: V.graph.constants[x.get_name()],  # weight zp
                6: lambda x: V.graph.constants[x.get_name()],  # bias
            }
            if isinstance(
                ir.InputsKernel.unwrap_storage_for_input(x_scale),
                ir.ConstantBuffer,
            ):
                # x is statically quantized
                input_gen_fns[1] = lambda x: V.graph.constants[x.get_name()]
            if isinstance(
                ir.InputsKernel.unwrap_storage_for_input(x_zp),
                ir.ConstantBuffer,
            ):
                input_gen_fns[2] = lambda x: V.graph.constants[x.get_name()]

            result, _ = autotune_select_algorithm(
                "qlinear_unary",
                choices,
                [x, x_scale, x_zp, packed_weight, w_scale, w_zp]
                if bias is None
                else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if len(x_size) > 2:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))
            return result

        @register_lowering(
            torch.ops.onednn.qlinear_pointwise.binary, type_promotion_kind=None
        )
        @register_lowering(
            torch.ops.onednn.qlinear_pointwise.binary_tensor, type_promotion_kind=None
        )
        def qlinear_binary(
            x: TensorBox,
            x_scale,
            x_zp,
            packed_weight: TensorBox,
            w_scale: TensorBox,
            w_zp: TensorBox,
            x2: TensorBox,
            bias: TensorBox,
            o_scale,
            o_zero_point,
            output_dtype,
            x2_scale,
            x2_zp,
            binary_attr,
            alpha,
            unary_attr,
            unary_scalars,
            unary_algorithm,
            layout=None,
        ):
            x_size = x.get_size()
            x2_size = x2.get_size()
            if len(x_size) != len(x2_size):
                raise AssertionError(
                    f"expected x and x2 to have equal rank, got {len(x_size)} and {len(x2_size)}"
                )
            if len(x_size) > 2 and binary_attr in ["add", "sum"]:
                # GEMM template needs 2D input, normalize input shape here
                x = view(x, [-1, x_size[-1]])
                x2 = view(x2, [-1, x2_size[-1]])
            if not isinstance(x_scale, ir.TensorBox):
                if type(x_scale) is not float:
                    raise AssertionError(
                        f"expected x_scale to be float, got {type(x_scale)}"
                    )
                x_scale = V.graph.add_tensor_constant(
                    torch.tensor(x_scale, dtype=torch.float32), name="x_scale"
                )
            else:
                x_scale_size = x_scale.get_size()
                if len(x_scale_size) == 0 or all(dim == 1 for dim in x_scale_size):
                    x_scale = _convert_to_0d_constant(x_scale, torch.float32, "_0d")
                else:
                    x_scale.realize()
                if len(x_scale.get_size()) not in [0, 1]:
                    raise AssertionError("x_scale must be 0D or 1D")

            if x_zp is None:
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="x_zp"
                )
            elif not isinstance(x_zp, ir.TensorBox):
                if type(x_zp) is not int:
                    raise AssertionError(f"expected x_zp to be int, got {type(x_zp)}")
                x_zp = V.graph.add_tensor_constant(
                    torch.tensor(x_zp, dtype=torch.int32), name="x_zp"
                )
            else:
                x_zp_size = x_zp.get_size()
                if len(x_zp_size) == 0 or all(dim == 1 for dim in x_zp_size):
                    x_zp = _convert_to_0d_constant(x_zp, torch.int32, "_0d")
                else:
                    x_zp.realize()

            if w_zp is None:
                w_zp = V.graph.add_tensor_constant(
                    torch.tensor(0, dtype=torch.int32), name="w_zp"
                )

            # When channels less than 8, w_scale/w_zp is Pointwise instead of ConstantBuffer
            # Refer to
            # https://github.com/pytorch/pytorch/blob/f353d17755ed23b02924c962a86ff99a3405fe10/torch/_inductor/graph.py#L570-L577
            w_scale.realize()
            w_zp.realize()
            if w_zp.get_dtype() != torch.int32 and isinstance(
                ir.InputsKernel.unwrap_storage_for_input(w_zp),
                ir.ConstantBuffer,
            ):
                w_zp_tensor = V.graph.constants[w_zp.get_name()].to(torch.int32)
                w_zp = V.graph.add_tensor_constant(  # type: ignore[assignment]
                    torch.tensor(w_zp_tensor, dtype=torch.int32), name=w_zp.get_name()
                )
            if binary_attr == "sum":
                if output_dtype in [
                    torch.float32,
                    torch.bfloat16,
                ] and x2.get_dtype() in [torch.float32, torch.bfloat16]:
                    if x2.get_dtype() != output_dtype:
                        # For int8-mixed-bf16 quantization and inplace add,
                        # there is case when accum dtype is float32 but output dtype is bfloat16.
                        # Since the accum will be changed in-place with post op sum,
                        # we will do accum dtype conversion here.
                        x2 = to_dtype(x2, output_dtype)
                else:
                    if x2.get_dtype() != output_dtype:
                        raise AssertionError(
                            "dtype of accum for qlinear post op sum should be the same as output"
                        )
            x2_dtype = x2.get_dtype()
            bias_dtype = bias.get_dtype() if bias is not None else None
            choices: list[ChoiceCaller] = []
            if (config.max_autotune or config.max_autotune_gemm) and binary_attr in [
                "add",
                "sum",
            ]:
                *_, layout, x, packed_weight, x2 = mm_args(
                    x, packed_weight, x2, layout=layout, out_dtype=output_dtype
                )
                if (
                    isinstance(
                        ir.InputsKernel.unwrap_storage_for_input(x_zp),
                        ir.ConstantBuffer,
                    )
                    and len(x_zp.get_layout().size) == 0  # Per tensor quant of act
                    and isinstance(
                        ir.InputsKernel.unwrap_storage_for_input(w_zp),
                        ir.ConstantBuffer,
                    )
                    and torch.equal(
                        torch.zeros_like(V.graph.constants[w_zp.get_name()]),
                        V.graph.constants[w_zp.get_name()],
                    )  # We only compensate MatrixB and assume B_zp is 0 to avoid the compensation of MatrixA
                    and use_cpp_gemm_template(layout, x, packed_weight)
                ):
                    W_tensor = V.graph.constants[packed_weight.get_name()]
                    W_tensor = W_tensor.to_dense()
                    (
                        use_int8_fast_compensation_path,
                        weight_compens,
                        x_w_scale,
                    ) = create_int8_compensation(
                        W_tensor,
                        packed_weight,
                        x_scale,
                        x_zp,
                        w_scale,
                    )

                    def epilogue_creator(input_buffer):
                        # Epilogue to convert from s32 to f32 for u8s8f32
                        if output_dtype not in [
                            torch.float32,
                            torch.bfloat16,
                            torch.uint8,
                            torch.int8,
                        ]:
                            raise AssertionError(
                                f"unexpected output_dtype {output_dtype}"
                            )

                        input_loader = input_buffer.make_loader()
                        x2_loader = x2.make_loader()
                        weight_compens_loader = weight_compens.make_loader()
                        x_w_scale_loader = None
                        if use_int8_fast_compensation_path:
                            if x_w_scale is None:
                                raise AssertionError(
                                    "x_w_scale must be set on the fast compensation path"
                                )
                            x_w_scale_loader = x_w_scale.make_loader()
                        x_scale_loader = x_scale.make_loader()
                        w_scale_loader = w_scale.make_loader()
                        x_zp_loader = x_zp.make_loader()
                        nonlocal bias
                        bias_loader = None
                        if bias is not None:
                            bias_loader = bias.make_loader()

                        def inner_fn(index):
                            nonlocal bias
                            input = input_loader(index)
                            _x2 = x2_loader(index)
                            _x_scale = None
                            _x_zp = None
                            _w_scale = None
                            weight_compens_index = (index[-1],)
                            if not use_int8_fast_compensation_path:
                                _x_scale = x_scale_loader(())
                                _x_zp = x_zp_loader(())
                                _w_scale = w_scale_loader(weight_compens_index)
                            # MicroKernel Output is with int32: cvt to FP32 before doing compensation
                            input = ops.to_dtype(input, torch.float32)
                            _weight_compo = weight_compens_loader(weight_compens_index)
                            _x_w_scale = None
                            if use_int8_fast_compensation_path:
                                if x_w_scale_loader is None:
                                    raise AssertionError(
                                        "x_w_scale_loader must be set on the fast compensation path"
                                    )
                                _x_w_scale = x_w_scale_loader(weight_compens_index)
                            # Step 1: Doing compensation to cvt fp32
                            temp = codegen_int8_gemm_template_compensation(
                                use_int8_fast_compensation_path,
                                input,
                                _weight_compo,
                                _x_scale,
                                _x_zp,
                                _w_scale,
                                _x_w_scale,
                            )
                            # Step 2: add Bias if applicable
                            if bias is not None:
                                # pyrefly: ignore [not-callable]
                                _bias = bias_loader(weight_compens_index)
                                nonlocal bias_dtype
                                if bias_dtype not in [torch.float32, torch.bfloat16]:
                                    raise AssertionError(
                                        f"unexpected bias_dtype {bias_dtype}"
                                    )
                                if bias_dtype == torch.bfloat16:
                                    _bias = ops.to_dtype(_bias, torch.float32)
                                temp = ops.add(temp, _bias)

                            # Step 3: Binary add
                            nonlocal x2_dtype
                            if x2_dtype not in [torch.float32, torch.bfloat16]:
                                raise AssertionError(f"unexpected x2_dtype {x2_dtype}")
                            if x2_dtype == torch.bfloat16:
                                _x2 = ops.to_dtype(_x2, torch.float32)
                            temp = ops.add(temp, _x2)

                            return temp

                        output_buf = ir.Pointwise(
                            device=input_buffer.get_device(),
                            dtype=torch.float32,  # Hardcode to FP32 for u8s8f32
                            inner_fn=inner_fn,
                            ranges=input_buffer.get_size(),
                        )

                        # Step 4: Unary post op if has
                        if unary_attr != "none":
                            output_buf = create_epilogue_with_attr(
                                output_buf,
                                unary_attr,
                                scalars=unary_scalars,
                                algorithm=unary_algorithm,
                            )

                        # Step 5: Cast output to Target Dtype
                        if output_dtype == torch.bfloat16:
                            output_cast_loader = output_buf.make_loader()

                            def inner_fn_cast_output_to_bf16(index):
                                input = output_cast_loader(index)
                                return ops.to_dtype(input, output_dtype)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device_or_error(),
                                dtype=output_dtype,
                                inner_fn=inner_fn_cast_output_to_bf16,
                                ranges=output_buf.get_size(),
                            )
                        elif output_dtype in [torch.uint8, torch.int8]:
                            from .lowering import _create_constants

                            requant_input_loader = output_buf.make_loader()

                            def inner_fn_requant(index, scale, zero_point):
                                input = requant_input_loader(index)
                                inv_scale, zero_point = _create_constants(
                                    1.0 / scale, zero_point, dtype=torch.float32
                                )
                                val = ops.round(input * inv_scale) + zero_point
                                if output_dtype == torch.uint8:
                                    qmin, qmax = _create_constants(
                                        0, 255, dtype=torch.float32
                                    )
                                else:
                                    qmin, qmax = _create_constants(
                                        -128, 127, dtype=torch.float32
                                    )
                                clamped = ops.minimum(ops.maximum(val, qmin), qmax)
                                return ops.to_dtype(clamped, torch.uint8)

                            output_buf = ir.Pointwise(
                                device=output_buf.get_device_or_error(),
                                dtype=torch.uint8,
                                inner_fn=functools.partial(
                                    inner_fn_requant,
                                    scale=float(o_scale),
                                    zero_point=int(o_zero_point),
                                ),
                                ranges=output_buf.get_size(),
                            )

                        return output_buf

                    CppGemmTemplate.add_choices(
                        choices,
                        layout,
                        [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2]
                        if bias is None
                        else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2, bias],
                        has_bias=bias is not None,
                        epilogue_creator=epilogue_creator,
                        # Reorder bias and x2
                        input_indices=[0, 3, 1, 2, 4, 5, 6]
                        if bias is None
                        else [7, 0, 3, 1, 2, 4, 5, 6],
                    )

            if len(choices) == 0 or use_aten_gemm_kernels():
                kwargs = dict(
                    output_scale=o_scale,
                    output_zero_point=o_zero_point,
                    output_dtype=output_dtype,
                    other_scale=x2_scale,
                    other_zp=x2_zp,
                    binary_post_op=binary_attr,
                    binary_alpha=alpha,
                    unary_post_op=unary_attr,
                    unary_post_op_args=unary_scalars,
                    unary_post_op_algorithm=unary_algorithm,
                )
                if bias is None:
                    kwargs["bias"] = None
                choices.append(
                    aten_mkldnn_qlinear_binary.bind(
                        (x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2)
                        if bias is None
                        else (x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2, bias),
                        layout,
                        **kwargs,
                    )
                )
            if packed_weight.get_name() not in V.graph.constants:
                raise AssertionError("packed_weight must be a graph constant")
            input_gen_fns = {
                3: lambda x: V.graph.constants[x.get_name()],
                4: lambda x: V.graph.constants[x.get_name()],
                5: lambda x: V.graph.constants[x.get_name()],
            }
            if bias is not None:
                input_gen_fns[7] = lambda x: V.graph.constants[x.get_name()]  # For bias
            result, _ = autotune_select_algorithm(
                "qlinear_binary",
                choices,
                [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2]
                if bias is None
                else [x, x_scale, x_zp, packed_weight, w_scale, w_zp, x2, bias],
                layout,
                input_gen_fns=input_gen_fns,
            )
            if (
                isinstance(result.data.data, ir.CppTemplateBuffer)
                and binary_attr == "sum"
                and result.data.data.layout == x2.get_layout()
            ):
                # In this case, since x2 is inplace updated when binary_attr is "sum"
                # we update the layout of result to view of x2
                result = ir.TensorBox.create(
                    ir.CppTemplateBuffer(
                        layout=ir.NonOwningLayout(
                            ir.ReinterpretView(data=x2, layout=x2.get_layout())
                        ),
                        inputs=result.data.data.inputs,  # type: ignore[arg-type]
                        make_kernel_render=result.data.data.make_kernel_render,  # type: ignore[arg-type]
                        template=result.data.data.template,
                        choice=result.data.data.choice,
                    )
                )
            if len(x_size) > 2 and binary_attr in ["add", "sum"]:
                result = view(result, (*x_size[:-1], result.get_size()[-1]))  # type: ignore[arg-type]
            return result

        if torch._C.has_mkl:
            aten_mkl_linear = ExternKernelChoice(
                torch.ops.mkl._mkl_linear,
                "mkl::_mkl_linear",
                has_out_variant=False,
                kernel_creator=mkldnn_ir.MKLPackedLinear.create,
            )
            cpu_needs_realized_inputs.append(torch.ops.mkl._mkl_linear)

            @register_lowering(torch.ops.mkl._mkl_linear)
            def mkl_packed_linear(
                x: TensorBox,
                packed_w: TensorBox,
                orig_w: TensorBox,
                b: TensorBox | None,
                batch_size,
                *,
                layout=None,
            ):
                choices: list[ChoiceCaller] = []
                if config.max_autotune or config.max_autotune_gemm:
                    transposed_w = permute(orig_w, [1, 0])
                    *_, layout, x, transposed_w = mm_args(
                        x, transposed_w, layout=layout
                    )
                    if use_cpp_gemm_template(layout, x, transposed_w):
                        CppGemmTemplate.add_choices(
                            choices,
                            layout,
                            [x, packed_w, orig_w],
                            trans_w=True,
                            input_indices=[0, 2],
                        )

                if len(choices) == 0 or use_aten_gemm_kernels():
                    choices.append(
                        aten_mkl_linear.bind(
                            (x, packed_w, orig_w), layout, B=None, batch_size=batch_size
                        )
                    )

                if packed_w.get_name() not in V.graph.constants:
                    raise AssertionError("packed_w must be a graph constant")
                if orig_w.get_name() not in V.graph.constants:
                    raise AssertionError("orig_w must be a graph constant")
                # packed_w is a mkldnn tensor which we can't generate directly
                # so we use the weights from the original tensor in autotune.
                input_gen_fns = {
                    1: lambda x: V.graph.constants[x.get_name()],
                    2: lambda x: V.graph.constants[x.get_name()],
                }
                result: TensorBox  # annotation on separate line since tuple unpacking doesn't support inline annotation
                result, _ = autotune_select_algorithm(
                    "packed_linear",
                    choices,
                    [x, packed_w, orig_w],
                    layout,
                    input_gen_fns=input_gen_fns,
                )
                if b is not None:
                    result = add(result, b)
                return result

        add_needs_realized_inputs(cpu_needs_realized_inputs)


register_onednn_fusion_ops()
<|endoftext|>import os
import sys

from torch.utils._config_module import install_config_module


# Whether to use c10d._time_estimator for collectives runtime estimations.
runtime_estimations_use_nccl_lib_estimations: bool = False

# Config to enable sync of runtime estimations across distributed ranks,
# To prevent passes using this runtime estimations to make different
# decisions on different distributed ranks.
runtime_estimations_align_across_all_distributed_ranks: bool = False

reorder_iterative_debug_memory_recompute: bool = False
reorder_iterative_debug_limit_to_reorder: int | None = (
    None
    if (env_str := os.getenv("PYTORCH_REORDER_COLLECTIVES_LIMIT")) is None
    else int(env_str)
)
sink_waits_iterative_debug_limit_to_sink: int | None = (
    # pyrefly: ignore[unbound-name]
    None if (env_str := os.getenv("PYTORCH_SINK_WAITS_LIMIT")) is None else int(env_str)
)


# Should be used with config.runtime_estimations_mms_benchmark = True
reorder_iterative_use_runtime_estimations: bool = False
sink_iterative_use_runtime_estimations: bool = False

# Broadcast runtime estimations doing real Collective operation between all ranks.
# If non-deterministic runtime estimations are used this must be used to make
# all ranks to do identical decisions and prevent global Collectives reordering,
# (that will result in NCCL hangs)
reorder_for_compute_comm_overlap_broadcast_runtime_estimations: bool = False

# Block of Ratios to workaround imperfection of current runtime estimations
# for collectives and compute for different scenarios.
# Multiplier of collectives estimated durations
reorder_sink_runtime_estimations_comm_mult: float = 2.0
# Multiplier of compute estimated durations
reorder_sink_runtime_estimations_non_comm_mult: float = 1.0
# The reordering will stop to reorder
# when overlap_comp >= (1 + extra_overlap_ratio) * comm_time
# Allows to configure more aggressive overlap
reorder_iterative_extra_comm_comp_overlap: float = 0.5
# The sink waits reordering will stop to reorder
# when overlap_comp >= (1 + extra_overlap_ratio) * comm_time
# Allows to configure more aggressive sink waits
sink_iterative_extra_comm_comp_overlap: float = 0.5

# Allow reorder iterative pass to increase peak memory
# up to peak_memory_before_pass * (1 + budget)
reorder_iterative_peak_memory_budget: float = 0.2
# Allow sink waits iterative pass to increase peak memory
# up to peak_memory_before_pass * (1 + budget)
sink_iterative_peak_memory_budget: float = 0.2

# Experimental unsafe configuration that allows changing relative collectives order.
# Must be used with runtime_estimations_align_across_all_distributed_ranks = True
reorder_iterative_unsafe_collectives_reorder: bool = True
sink_waits_iterative_unsafe_collectives_reorder: bool = True

# Allow group and move other collectives during reordering
reorder_iterative_group_with_collectives: bool = False
sink_waits_iterative_swap_with_collectives: bool = False

reorder_sink_verbose_logging: bool = False

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
<|endoftext|>"""Utils for Meta Triton autoWS."""

from __future__ import annotations

import functools


@functools.cache
def has_meta_ws() -> bool:
    """Whether Meta Triton autoWS is available."""
    try:
        from triton import knobs
    except ImportError:
        return False
    return hasattr(getattr(knobs, "nvidia", None), "use_meta_ws")


def meta_ws_enabled() -> bool:
    """Whether Meta Triton autoWS is available and enabled via its knob."""
    if not has_meta_ws():
        return False
    from triton import knobs

    return bool(knobs.nvidia.use_meta_ws)
<|endoftext|>import argparse
import datetime
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

import torch
from torch.autograd import DeviceType
from torch.utils._ordered_set import OrderedSet

from .runtime.benchmarking import benchmarker
from .runtime.runtime_utils import create_bandwidth_info_str, get_num_bytes


class BenchmarkCallableType(Protocol):
    def __call__(self, times: int, repeat: int) -> float: ...


_kernel_category_choices = [
    "foreach",
    "persistent_reduction",
    "pointwise",
    "reduction",
    "split_scan",
    "template",
]


def get_kernel_category_by_source_code(src_code: str) -> str:
    """
    Similar to get_kernel_category but use the source code. Call this API
    if we have not compile the src_code to module yet.
    """
    choices = [
        ch for ch in _kernel_category_choices if f"@triton_heuristics.{ch}" in src_code
    ]
    if len(choices) == 1:
        return choices[0]
    else:
        return "unknown"


def get_kernel_category(kernel_mod: ModuleType) -> str:
    """
    Given the module defining a triton kernel, return the category of the kernel.
    Category can be one of:
    - pointwise
    - reduction
    - persistent_reduction

    Currently we simply decide the category depending on what decorator is imported
    by the kernel.
    """
    choices = [ch for ch in _kernel_category_choices if ch in kernel_mod.__dict__]
    if len(choices) == 1:
        return choices[0]
    else:
        return "unknown"


def get_triton_kernel(mod: ModuleType):  # type: ignore[no-untyped-def]
    from torch._inductor.runtime.triton_heuristics import CachingAutotuner

    cand_list = [
        v
        for k, v in mod.__dict__.items()
        if k.startswith("triton_") and isinstance(v, CachingAutotuner)
    ]
    if len(cand_list) != 1:
        raise AssertionError(
            f"expected exactly one triton kernel, got {len(cand_list)}"
        )
    return cand_list[0]


def benchmark_all_kernels(
    benchmark_name: str, benchmark_all_configs: dict[Any, Any] | None
) -> None:
    """
    An experimental API used only when config.benchmark_kernel is true.

    Run the kernel benchmarks for all the kernels cached in PyCodeCache.
    Used in the compiled modules.

    Put this method here rather than codegen it for convenience since its implementation
    does not change based on different graph modules being compiled.
    """
    from torch._inductor.codecache import PyCodeCache

    nfound = 0
    for kernel_mod in PyCodeCache.modules:
        kernel_key = kernel_mod.key
        if not hasattr(kernel_mod, "get_args") or not hasattr(kernel_mod, "call"):
            continue

        triton_kernel = get_triton_kernel(kernel_mod)
        device_type = triton_kernel.device_props.type
        kernel_category = get_kernel_category(kernel_mod)
        args = kernel_mod.get_args()
        num_in_out_ptrs = len(
            [
                arg_name
                for arg_name in triton_kernel.fn.arg_names
                if arg_name.startswith("in_out_ptr")
            ]
        )
        num_gb = triton_kernel.inductor_meta.get("kernel_num_gb", None)
        if num_gb is None:
            num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9

        def get_info_str(
            ms: float,
            n_regs: Any | None,
            n_spills: Any | None,
            shared: Any | None,
            prefix: str = "",
        ) -> str:
            if not any(x is None for x in [n_regs, n_spills, shared]):
                kernel_detail_str = (
                    f"  {n_regs:3} regs  {n_spills:3} spills  {shared:8} shared mem"
                )
            else:
                kernel_detail_str = ""

            gb_per_s = num_gb / (ms / 1e3)
            return create_bandwidth_info_str(
                ms, num_gb, gb_per_s, prefix=prefix, suffix=kernel_detail_str
            )

        kernel_desc = (
            f"{benchmark_name:20} {kernel_category[:3].upper()} {kernel_key[:10]}"
        )
        if benchmark_all_configs:
            if not hasattr(kernel_mod, "benchmark_all_configs"):
                raise AssertionError("kernel_mod is missing benchmark_all_configs")
            bench_result = kernel_mod.benchmark_all_configs(args)
            print(kernel_desc)
            for launcher, ms in bench_result.items():
                print(
                    f"  {get_info_str(ms, launcher.n_regs, launcher.n_spills, launcher.shared)} @ {launcher.config}"
                )
        else:
            ms = benchmarker.benchmark(
                lambda: kernel_mod.call(args),
                device=device_type,
                rep=40,
            )
            if len(triton_kernel.launchers) != 1:
                raise AssertionError("Autotuner should have selected the best config")
            launcher = triton_kernel.launchers[0]
            print(
                get_info_str(
                    ms,
                    launcher.n_regs,
                    launcher.n_spills,
                    launcher.shared,
                    prefix=f"{kernel_desc} ",
                )
            )

        nfound += 1
    if nfound == 0:
        print(
            "No kernel with benchmark functionality found. Make sure you run inductor with config.benchmark_kernel being True"
        )


@dataclass
class ProfileEvent:
    category: str
    key: str
    self_device_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float


def parse_profile_event_list(
    benchmark_name: str,
    event_list: torch.autograd.profiler_util.EventList,
    wall_time_ms: float,
    nruns: int,
    device_name: str,
) -> None:
    """
    Parse and generate a report for an event_list.
    """

    def get_self_device_time(
        ev: torch.autograd.profiler_util.EventList,
    ) -> float:
        """
        ev.self_device_time_total is in microsecond. Convert to millisecond.
        """
        return ev.self_device_time_total / 1000 / nruns  # type: ignore[attr-defined]

    all_events: dict[str, list[ProfileEvent]] = defaultdict(list)

    def add_event(
        ev: torch.autograd.profiler_util.EventList,
        category: str,
    ) -> None:
        profile_ev = ProfileEvent(
            category=category,
            key=ev.key,  # type: ignore[attr-defined]
            self_device_time_ms=get_self_device_time(ev),
            count=ev.count / nruns,  # type: ignore[operator] # average across all runs
        )
        all_events[category].append(profile_ev)

    for ev in event_list:
        if ev.is_legacy:
            raise AssertionError("Don't support the legacy profiler")
        if ev.device_type == DeviceType.CPU:
            # ignore the event on CPU side
            continue

        category = "unknown"
        if ev.key.startswith("triton_"):
            if ev.key.startswith("triton_poi"):
                category = "triton_pointwise"
            elif ev.key.startswith("triton_red"):
                category = "triton_reduction"
            elif ev.key.startswith("triton_per"):
                category = "triton_persistent_reduction"
            else:
                category = "triton_unknown"

        add_event(ev, category)

    def report_category(category: str, profile_events: list[ProfileEvent]) -> float:
        if not device_name:
            return 0.0

        from tabulate import tabulate

        profile_events.sort(key=lambda ev: ev.self_device_time_ms, reverse=True)

        rows = []
        total_time = 0.0
        print(f"\n  == {category} category kernels == ")
        for ev in profile_events:
            total_time += ev.self_device_time_ms
            percent = f"{ev.self_device_time_ms / wall_time_ms * 100:.2f}%"
            rows.append([ev.key[:120], ev.self_device_time_ms, ev.count, percent])
        rows.append(
            ["Total", total_time, "", f"{total_time / wall_time_ms * 100:.2f}%"]
        )
        print(
            tabulate(
                rows,
                headers=[
                    "Kernel",
                    f"Self {device_name.upper()} TIME (ms)",
                    "Count",
                    "Percent",
                ],
            )
        )
        return total_time

    def report() -> None:
        category_list = [
            "triton_pointwise",
            "triton_reduction",
            "triton_persistent_reduction",
            "triton_unknown",
            "unknown",
        ]
        if not OrderedSet(all_events.keys()).issubset(OrderedSet(category_list)):
            raise AssertionError(f"{list(all_events.keys())}")

        per_category_wall_time = {}
        total_device_ms = 0.0
        for category in category_list:
            if category in all_events:
                _time = report_category(category, all_events[category])
                per_category_wall_time[category] = _time
                total_device_ms += _time

        device_busy_percent = f"{total_device_ms / wall_time_ms * 100:.2f}%"
        if device_name:
            print(
                f"\nPercent of time when {device_name.upper()} is busy: {device_busy_percent}"
            )
        else:
            print("No device detected")

        print(f"Total wall time {wall_time_ms:.3f} ms")

        # output such a line so we can gather such line from all compiled modules from all
        # benchmarks and tabulate it!
        # Columns: benchmark_name, pointwise_percent, reduction_percent, persistent_reduction_percent,
        #   unknown_category_percent, device_busy_percent, wall_time_ms
        tabulate_line = f"Output for tabulate: {benchmark_name}"
        for category in category_list:
            percent = (
                f"{per_category_wall_time.get(category, 0.0) / wall_time_ms * 100:.2f}%"
            )
            tabulate_line += f", {percent}"
        tabulate_line += f", {device_busy_percent}, {wall_time_ms:.3f}ms"

        print(tabulate_line)

    report()


PROFILE_DIR = tempfile.gettempdir()
PROFILE_PATH = f"{PROFILE_DIR}/compiled_module_profile.json"


def perf_profile(
    wall_time_ms: float,
    times: int,
    repeat: int,
    benchmark_name: str,
    benchmark_compiled_module_fn: BenchmarkCallableType,
) -> None:
    with torch.profiler.profile(record_shapes=True) as p:
        benchmark_compiled_module_fn(times=times, repeat=repeat)

    path = PROFILE_PATH
    p.export_chrome_trace(path)
    print(f"Profiling result for a compiled module of benchmark {benchmark_name}:")
    print(f"Chrome trace for the profile is written to {path}")
    event_list = p.key_averages(group_by_input_shape=True)
    print(event_list.table(sort_by="self_device_time_total", row_limit=10))
    parse_profile_event_list(
        benchmark_name, event_list, wall_time_ms, times * repeat, p.use_device or ""
    )


def ncu_analyzer(
    benchmark_name: str,
    benchmark_compiled_module_fn: BenchmarkCallableType,
    args: argparse.Namespace,
) -> None:
    import inspect
    import os
    import subprocess

    kernel_regex = args.ncu_kernel_regex
    metrics = args.ncu_metrics

    module_file = inspect.getfile(benchmark_compiled_module_fn)
    module_dir = os.path.dirname(module_file)
    module_name = os.path.splitext(os.path.basename(module_file))[0]

    ncu_dir = tempfile.gettempdir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ncu_output = os.path.join(ncu_dir, f"ncu_output_{timestamp}.ncu-rep")
    python_cmd = (
        f"""import sys; sys.path.insert(0, '{module_dir}'); """
        f"""from {module_name} import benchmark_compiled_module; """
        """benchmark_compiled_module(times=1, repeat=1)"""
    )

    ncu_cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--replay-mode",
        "kernel",
        "--kernel-name-base",
        "function",
        "--print-units",
        "base",
        "--import-source",
        "yes",
        "--force-overwrite",
        "--export",
        ncu_output,
    ]

    if kernel_regex:
        ncu_cmd.extend(["--kernel-name", f"regex:{kernel_regex}"])

    if metrics:
        ncu_cmd.extend(["--metrics", metrics])
    else:
        ncu_cmd.extend(["--set", "full"])

    ncu_cmd.extend(
        [
            "python",
            "-c",
            python_cmd,
        ]
    )

    try:
        subprocess.run(ncu_cmd, check=True)
        print(f"\nNCU profiling results for benchmark {benchmark_name}:")
        print(f"NCU report has been written to {ncu_output}")

    except subprocess.CalledProcessError as e:
        print(f"NCU profiling failed with error: {e}")
        return


def collect_memory_snapshot(
    benchmark_compiled_module_fn: BenchmarkCallableType,
) -> None:
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is not available")

    torch.cuda.memory._record_memory_history(max_entries=100000)
    benchmark_compiled_module_fn(times=10, repeat=1)  # run 10 times
    snapshot_path = f"{tempfile.gettempdir()}/memory_snapshot.pickle"
    torch.cuda.memory._dump_snapshot(snapshot_path)
    torch.cuda.memory._record_memory_history(enabled=None)
    print(f"The collect memory snapshot has been written to {snapshot_path}")


# With AOTAutograd cache, we directly call the compiled module. So prevent
# Dynamo from reentering
@torch.compiler.disable  # type: ignore[misc]
def compiled_module_main(
    benchmark_name: str, benchmark_compiled_module_fn: BenchmarkCallableType
) -> None:
    """
    This is the function called in __main__ block of a compiled module.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-kernels",
        "-k",
        action="store_true",
        help="Whether to benchmark each individual kernels",
    )
    parser.add_argument(
        "--benchmark-all-configs",
        "-c",
        action="store_true",
        help="Whether to benchmark each individual config for a kernel",
    )
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Whether to profile the compiled module",
    )
    parser.add_argument(
        "--cuda-memory-snapshot",
        action="store_true",
        help="""
            Whether to collect CUDA memory snapshot. Refer to
            "https://pytorch.org/blog/understanding-gpu-memory-1/
            for details about how to visualize the collected snapshot
        """,
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Whether to run ncu analysis",
    )
    parser.add_argument(
        "--ncu-kernel-regex",
        type=str,
        default=None,
        help=(
            "Filter kernels profiled by NCU using a regex (e.g., '^triton_.*'). "
            "Maps to '--kernel-name regex:<regex>'. "
            "If None, NCU will profile all kernels."
        ),
    )
    parser.add_argument(
        "--ncu-metrics",
        type=str,
        default=None,
        help=(
            "Comma-separated list of NCU metrics to collect (e.g., 'dram__bytes.sum.per_second'). "
            "If None, NCU will use '--set full'."
        ),
    )
    parser.add_argument(
        "--times",
        type=int,
        default=10,
        help="Number of times to run each benchmark iteration",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=10,
        help="Number of repetitions of each benchmark run",
    )

    args = parser.parse_args()

    if args.benchmark_kernels:
        benchmark_all_kernels(benchmark_name, args.benchmark_all_configs)
    else:
        times = args.times
        repeat = args.repeat

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        wall_time_ms = benchmark_compiled_module_fn(times=times, repeat=repeat) * 1000

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated()
            print(f"Peak GPU memory usage {peak_mem / 1e6:.3f} MB")

        if torch.cuda.is_available() and args.cuda_memory_snapshot:
            collect_memory_snapshot(benchmark_compiled_module_fn)

        if args.profile:
            perf_profile(
                wall_time_ms,
                times,
                repeat,
                benchmark_name,
                benchmark_compiled_module_fn,
            )
        if args.ncu:
            ncu_analyzer(
                benchmark_name,
                benchmark_compiled_module_fn,
                args=args,
            )
<|endoftext|>import contextlib
import os

from torch._dynamo.test_case import (
    run_tests as dynamo_run_tests,
    TestCase as DynamoTestCase,
)
from torch._functorch import config as functorch_config
from torch._inductor import config
from torch._inductor.utils import fresh_cache


def run_tests(needs: str | tuple[str, ...] = ()) -> None:
    dynamo_run_tests(needs)


class TestCase(DynamoTestCase):
    """
    A base TestCase for inductor tests. Enables FX graph caching and isolates
    the cache directory for each test.
    """

    def setUp(self) -> None:
        super().setUp()
        self._inductor_test_stack = contextlib.ExitStack()
        self._inductor_test_stack.enter_context(
            functorch_config.patch(
                {
                    "enable_autograd_cache": True,
                }
            )
        )

        if (
            "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ
            and "TORCHINDUCTOR_FX_GRAPH_CACHE_DEFAULT" not in os.environ
        ):
            self._inductor_test_stack.enter_context(
                config.patch({"fx_graph_cache": True})
            )

        if (
            os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1"
            and os.environ.get("TORCH_COMPILE_DEBUG") != "1"
        ):
            self._inductor_test_stack.enter_context(fresh_cache())

    def tearDown(self) -> None:
        super().tearDown()
        self._inductor_test_stack.close()
<|endoftext|>"""
ROCm Multi-Architecture Support Utilities
Compile LLVM IR to multi-arch bundles that HIP can load automatically.
"""

import logging
import os
import re
import subprocess

import torch
from torch.utils.cpp_extension import _join_rocm_home, ROCM_HOME


log = logging.getLogger(__name__)


def get_rocm_compiler() -> str:
    """
    Get path to ROCm's clang compiler.
    Uses PyTorch's ROCM_HOME detection.

    Returns:
        Path to clang compiler

    Raises:
        RuntimeError: If ROCm is not found
    """
    if ROCM_HOME is None:
        raise RuntimeError(
            "ROCm installation not found. "
            "PyTorch was not built with ROCm support or ROCM_HOME is not set."
        )

    # ROCm's clang is at <ROCM_HOME>/llvm/bin/clang
    clang_path = _join_rocm_home("llvm", "bin", "clang")

    if not os.path.exists(clang_path):
        raise RuntimeError(
            f"ROCm clang not found at {clang_path}. ROCM_HOME is set to {ROCM_HOME}"
        )

    return clang_path


def get_rocm_bundler() -> str:
    """
    Get path to clang-offload-bundler.
    Uses PyTorch's ROCM_HOME detection.

    Returns:
        Path to bundler

    Raises:
        RuntimeError: If bundler is not found
    """
    if ROCM_HOME is None:
        raise RuntimeError(
            "ROCm installation not found. "
            "PyTorch was not built with ROCm support or ROCM_HOME is not set."
        )

    # Bundler is at <ROCM_HOME>/llvm/bin/clang-offload-bundler
    bundler_path = _join_rocm_home("llvm", "bin", "clang-offload-bundler")

    if not os.path.exists(bundler_path):
        raise RuntimeError(
            f"clang-offload-bundler not found at {bundler_path}. "
            f"ROCM_HOME is set to {ROCM_HOME}"
        )

    return bundler_path


def get_rocm_target_archs() -> list[str]:
    env_archs = os.environ.get("PYTORCH_ROCM_ARCH", "").strip()
    if env_archs:
        archs = [arch.strip() for arch in env_archs.replace(";", ",").split(",")]
        archs = [arch for arch in archs if arch]
        if archs:
            # Ensure current device arch is included
            if torch.cuda.is_available():
                for dev_idx in range(torch.cuda.device_count()):
                    current_arch = torch.cuda.get_device_properties(
                        dev_idx
                    ).gcnArchName.split(":")[0]
                    if current_arch not in archs:
                        archs.append(current_arch)
            return archs

    try:
        from torch._inductor import config

        if hasattr(config, "rocm") and hasattr(config.rocm, "target_archs"):
            archs = config.rocm.target_archs
            if archs:
                return archs

    except Exception:
        pass

    return torch.cuda.get_arch_list()


def _sanitize_llvm_ir_for_rocm(llvm_ir_path: str) -> str:
    """
    Sanitize LLVM IR to be compatible with ROCm's clang.

    Triton's LLVM (upstream) may emit attributes and metadata that ROCm's
    older clang does not yet support. Only strips attributes confirmed to
    cause parse errors — preserves all others to maintain correct codegen.

    Currently strips:
        - nocreateundeforpoison: function attribute (upstream LLVM, not in ROCm)
        - dwarfAddressSpace: debug metadata field (upstream LLVM, not in ROCm)

    Returns:
        Path to sanitized .ll file, or original path if no changes needed.
    """
    with open(llvm_ir_path) as f:
        content = f.read()

    sanitized = content
    sanitized = re.sub(r"\bnocreateundeforpoison\b\s*", "", sanitized)
    sanitized = re.sub(r",\s*dwarfAddressSpace:\s*\d+", "", sanitized)

    if sanitized == content:
        return llvm_ir_path

    sanitized_path = llvm_ir_path + ".sanitized.ll"
    with open(sanitized_path, "w") as f:
        f.write(sanitized)

    log.debug(
        "Sanitized LLVM IR for ROCm clang compatibility: %s -> %s",
        llvm_ir_path,
        sanitized_path,
    )
    return sanitized_path


def compile_llvm_ir_to_code_object(
    llvm_ir_path: str, output_path: str, target_arch: str
) -> bool:
    """
    Compile unbundled LLVM IR to a single-arch code object.

    Args:
        llvm_ir_path: Path to .ll file
        output_path: Where to write .hsaco file
        target_arch: Target architecture (e.g., 'gfx90a')

    Returns:
        True if successful
    """
    if not os.path.exists(llvm_ir_path):
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        clang = get_rocm_compiler()
    except RuntimeError:
        return False

    # Sanitize LLVM IR to remove attributes unsupported by ROCm's clang
    llvm_ir_path = _sanitize_llvm_ir_for_rocm(llvm_ir_path)

    # Using clang and not hipcc since we are not compiling source code
    # Instead we use the LLVM IR (.ll) provided by triton
    cmd = [
        clang,
        "-target",
        "amdgcn-amd-amdhsa",
        f"-mcpu={target_arch}",
        llvm_ir_path,
        "-o",
        output_path,
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        if not os.path.exists(output_path):
            return False

        return True

    except subprocess.CalledProcessError:
        return False


def create_multiarch_bundle(code_objects: dict, output_bundle_path: str) -> bool:
    """
    Bundle multiple architecture code objects into a single multi-arch bundle.

    Uses clang-offload-bundler to create a fat binary that HIP runtime can load.
    The runtime automatically selects the correct architecture at load time.

    Args:
        code_objects: Dict mapping architecture to code object path
        output_bundle_path: Path for output bundle

    Returns:
        True if successful
    """
    if not code_objects:
        return False

    os.makedirs(os.path.dirname(output_bundle_path), exist_ok=True)

    try:
        bundler = get_rocm_bundler()
    except RuntimeError:
        return False

    # Build targets and inputs lists for clang-offload-bundler
    targets = ["host-x86_64-unknown-linux-gnu"]

    # We include a dummy host entry to satisfy the bundler format
    inputs = ["/dev/null"]

    for arch, path in sorted(code_objects.items()):
        if not os.path.exists(path):
            continue
        # hipv4 = HIP version 4 code object format
        # amdgcn-amd-amdhsa = target triple for ROCm/HSA runtime
        # arch = specific GPU (gfx90a, gfx942, etc.)
        targets.append(f"hipv4-amdgcn-amd-amdhsa--{arch}")
        inputs.append(path)

    if len(inputs) == 1:  # Only host, no device code
        return False

    cmd = [
        bundler,
        "--type=o",
        # CRITICAL: HIP runtime expects 4096-byte alignment for loading bundles
        # Without this, hipModuleLoadData gives segmentation fault
        "-bundle-align=4096",  # CRITICAL: Required by HIP runtime!
        f"--targets={','.join(targets)}",
    ]

    for input_file in inputs:
        cmd.append(f"--input={input_file}")

    cmd.append(f"--output={output_bundle_path}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        if not os.path.exists(output_bundle_path):
            return False

        return True

    except subprocess.CalledProcessError:
        return False


def compile_multiarch_bundle_from_llvm_ir(
    llvm_ir_path: str, output_bundle_path: str, target_archs: list[str] | None = None
) -> bool:
    """
    Complete workflow: LLVM IR → multiple code objects → bundle.

    This is the main entry point for multi-arch compilation.

    Args:
        llvm_ir_path: Path to .ll file
        output_bundle_path: Where to write bundle
        target_archs: Optional list of architectures

    Returns:
        True if successful
    """
    if target_archs is None:
        # Get architectures from environment variable or config
        target_archs = get_rocm_target_archs()

    # Step 1: Compile LLVM IR to code object for each architecture
    code_objects = {}
    temp_dir = os.path.dirname(output_bundle_path)
    kernel_name = os.path.splitext(os.path.basename(llvm_ir_path))[0]

    for arch in target_archs:
        # Create temporary single-architecture code object
        # Format: kernel_name_gfx90a.co, kernel_name_gfx942.co, etc.
        co_path = os.path.join(temp_dir, f"{kernel_name}_{arch}.co")

        # Compile with clang backend: LLVM IR → GPU machine code
        if compile_llvm_ir_to_code_object(llvm_ir_path, co_path, arch):
            code_objects[arch] = co_path

    if not code_objects:
        return False

    # Step 2: Bundle all code objects together
    # Uses clang-offload-bundler to create fat binary
    success = create_multiarch_bundle(code_objects, output_bundle_path)

    # Step 3: Clean up temporary single-arch code objects
    # The bundle contains all the code, so intermediates are no longer needed
    for co_path in code_objects.values():
        try:
            os.remove(co_path)
        except Exception:
            pass

    return success
<|endoftext|># mypy: allow-untyped-defs
import ast
import gzip
import json
import logging
import os
import re
import tempfile
from bisect import bisect_left
from collections import defaultdict
from typing import Any

import torch._inductor.config as inductor_config
from torch.utils._ordered_set import OrderedSet


__all__ = ["inductor_trace_handler"]

log = logging.getLogger(__name__)

# https://github.com/pytorch/kineto/blob/a054a4be0db117c579a21747debf19c863631f26/libkineto/src/output_json.cpp#L559
# Kernel and CUDA runtime API events are connected by ac2g; forward and backward
# ATen ops are connected by fwdbwd.
_profile_flow_types = OrderedSet(["fwdbwd", "ac2g"])


class _EventItem:
    def __init__(self, event):
        self.e = event
        self.start, self.end = self._interval_of()

    def _interval_of(self):
        start = self.e["ts"]
        duration = self.e.get("dur", 0)
        return start, start + duration


def _find_events_covered_in(
    events: list[dict[str, Any]],
    top_level_events: list[dict[str, Any]],
):
    """Find events covered in top level events.

    Returns a dict mapping top-level event uid to covered event uids.
    """
    top_level_by_tid = defaultdict(list)
    for event in top_level_events:
        top_level_by_tid[event["tid"]].append(_EventItem(event))
    for tid in top_level_by_tid:
        top_level_by_tid[tid] = sorted(top_level_by_tid[tid], key=lambda x: x.start)
    starts_by_tid = {
        tid: [event.start for event in event_items]
        for tid, event_items in top_level_by_tid.items()
    }
    top_level_id_to_events = defaultdict(OrderedSet)

    for event in events:
        if event.get("cat") not in {
            "cuda_runtime",
            "cuLaunchKernel",
            "cpu_op",
            "cuda_driver",
            "gpu_memset",
            "python_function",
        }:
            continue
        if "CallFrom" in event.get("args", {}):
            continue
        tid = event.get("tid")
        if tid not in top_level_by_tid:
            continue
        item = _EventItem(event)
        event_items = top_level_by_tid[tid]
        idx = bisect_left(starts_by_tid[tid], item.start) - 1
        if idx < 0 or idx >= len(event_items):
            continue
        top_level_item = event_items[idx]
        if top_level_item.start <= item.start and item.end <= top_level_item.end:
            top_level_id_to_events[top_level_item.e["uid"]].add(event["uid"])
    return top_level_id_to_events


class _InductorTraceProcessor:
    def _assign_uniq_id_to_event(self, trace):
        if trace.get("uid_assigned"):
            return {event["uid"]: event for event in trace["traceEvents"]}
        uid_2_events = {}
        for uid, event in enumerate(trace["traceEvents"]):
            event["uid"] = uid
            uid_2_events[uid] = event
        trace["uid_assigned"] = True
        return uid_2_events

    def _build_flow_mapping(self, trace, flow_events):
        """Build src/dst event mappings from profiler flow events."""
        if not flow_events:
            return {}, {}

        flow_pair: dict[int, list[int | None]] = {}
        prev_event = None
        for event in trace["traceEvents"]:
            if (
                event.get("name") not in _profile_flow_types
                and event.get("cat") not in _profile_flow_types
            ):
                prev_event = event
                continue
            if event.get("ph") == "s":
                if prev_event is None:
                    continue
                pair = flow_pair.setdefault(int(event["id"]), [None, None])
                pair[0] = prev_event["uid"]
            elif event.get("ph") == "f":
                if prev_event is None:
                    continue
                pair = flow_pair.setdefault(int(event["id"]), [None, None])
                pair[1] = prev_event["uid"]
            prev_event = event

        src2dst = {}
        dst2src = {}
        for src, dst in flow_pair.values():
            if src is None or dst is None:
                continue
            src2dst[src] = dst
            dst2src[dst] = src
        return src2dst, dst2src

    def _maybe_triton_call(self, kernel_name):
        return kernel_name.startswith("triton_")

    def _has_kernel_info(self, compile_info, graph_key, kernel_name):
        graph_info = compile_info.get(graph_key, {})
        if kernel_name in graph_info:
            return True
        kernel_prefix = kernel_name + ":"
        return any(name.startswith(kernel_prefix) for name in graph_info)

    def _maybe_extern_call(self, event, compile_info, fwd_key, bw_keys, need_bw):
        kernel_name = event["name"]
        return not self._has_kernel_info(compile_info, fwd_key, kernel_name) and (
            not need_bw
            or all(
                not self._has_kernel_info(compile_info, bw_key, kernel_name)
                for bw_key in bw_keys
            )
        )

    def add_to_chrome_trace(self, origin_trace):
        """Add Inductor kernel stack information to a Chrome trace."""
        from torch._inductor.debug import get_kernel_information_jsons

        compile_linenos = get_kernel_information_jsons()
        if not compile_linenos:
            return origin_trace

        trace = origin_trace
        uid_2_events = self._assign_uniq_id_to_event(trace)

        real_events = []
        flow_events = []
        compiled_events = []
        extern_events = []
        for event in trace["traceEvents"]:
            name = event.get("name", "")
            cat = event.get("cat", "")
            if name in _profile_flow_types or cat in _profile_flow_types:
                flow_events.append(event)
                continue

            real_events.append(event)
            if cat == "cpu_op" and (
                name == "CompiledFunctionBackward" or "Torch-Compiled Region" in name
            ):
                compiled_events.append(event)
            if (
                name
                and "extern_kernels" in name
                and cat
                in (
                    "cpu_op",
                    "python_function",
                )
            ):
                extern_events.append(event)

        compiled_events = sorted(compiled_events, key=lambda x: _EventItem(x).start)
        compiled_event_items = [_EventItem(event) for event in compiled_events]
        compiled_event_starts = [item.start for item in compiled_event_items]
        ops_in_compile_region = _find_events_covered_in(real_events, compiled_events)
        ops_in_extern_region = _find_events_covered_in(real_events, extern_events)
        src2dst, dst2src = self._build_flow_mapping(trace, flow_events)
        kernel_uids_by_external_id = defaultdict(list)
        for event in real_events:
            if event.get("cat") != "kernel":
                continue
            external_id = event.get("args", {}).get("External id")
            if external_id is not None:
                kernel_uids_by_external_id[external_id].append(event["uid"])

        def _related_compile_region(compile_event):
            src_event = uid_2_events[dst2src[compile_event["uid"]]]
            src_item = _EventItem(src_event)
            idx = bisect_left(compiled_event_starts, src_item.start) - 1
            # Compiled regions are expected to be nearly nested around their
            # source events, so the nearest preceding region usually matches.
            while idx >= 0:
                region_item = compiled_event_items[idx]
                if (
                    src_item.start > region_item.start
                    and src_item.end < region_item.end
                ):
                    return region_item.e
                idx -= 1
            raise ValueError(f"Cannot find compile region for {compile_event}")

        def _parse_compile_region_key(key):
            try:
                parsed_key = ast.literal_eval(key)
            except (SyntaxError, ValueError):
                return None
            if (
                not isinstance(parsed_key, tuple)
                or len(parsed_key) != 2
                or not isinstance(parsed_key[0], str)
                or not isinstance(parsed_key[1], bool)
            ):
                return None
            return parsed_key

        def _backward_graph_keys(compile_info, compile_name):
            bw_graph_key = str((compile_name, True))
            keys = [bw_graph_key]
            if bw_graph_key in compile_info:
                return keys

            prefix = compile_name + "_"
            for key, info in compile_info.items():
                if not info:
                    continue
                parsed_key = _parse_compile_region_key(key)
                if parsed_key is None:
                    continue
                graph_name, is_backward = parsed_key
                if is_backward and graph_name.startswith(prefix):
                    keys.append(key)
            return keys

        def _stack_from_kernel_info(kernel_info):
            if isinstance(kernel_info, dict):
                return kernel_info.get("stack_traces")
            return kernel_info

        def _stack_for_kernel(compile_info, graph_key, kernel_name):
            graph_info = compile_info.get(graph_key, {})
            kernel_info = graph_info.get(kernel_name)
            if kernel_info is None:
                kernel_prefix = kernel_name + ":"
                for name, info in graph_info.items():
                    if name.startswith(kernel_prefix):
                        kernel_info = info
                        break
            return _stack_from_kernel_info(kernel_info)

        def _single_stack_for_graph(compile_info, graph_key):
            stacks = [
                stack
                for stack in (
                    _stack_from_kernel_info(kernel_info)
                    for kernel_info in compile_info.get(graph_key, {}).values()
                )
                if stack is not None
            ]
            if len(stacks) == 1:
                return stacks[0]
            return None

        def _assign_stack(event, compile_info, fwd_key, bw_keys, kernel_name):
            for bw_key in bw_keys:
                stack = _stack_for_kernel(compile_info, bw_key, kernel_name)
                if stack is not None:
                    event.setdefault("args", {})["stack"] = stack
                    return True
            stack = _stack_for_kernel(compile_info, fwd_key, kernel_name)
            if stack is not None:
                event.setdefault("args", {})["stack"] = stack
                return True
            return False

        def _assign_single_stack(event, compile_info, fwd_key, bw_keys):
            for bw_key in bw_keys:
                stack = _single_stack_for_graph(compile_info, bw_key)
                if stack is not None:
                    event.setdefault("args", {})["stack"] = stack
                    return True
            stack = _single_stack_for_graph(compile_info, fwd_key)
            if stack is not None:
                event.setdefault("args", {})["stack"] = stack
                return True
            return False

        def _kernel_events_for_op(op_id):
            if op_id in src2dst:
                return [uid_2_events[src2dst[op_id]]]
            external_id = uid_2_events[op_id].get("args", {}).get("External id")
            if external_id is None:
                return []
            return [
                uid_2_events[kernel_uid]
                for kernel_uid in kernel_uids_by_external_id.get(external_id, [])
            ]

        def _extern_kernel_name(name):
            match = re.search(r"\b(extern\w+)\s*,", name)
            return match.group(1) if match else name

        for compile_func_id, scoped_ops in ops_in_compile_region.items():
            compile_event = uid_2_events[compile_func_id]
            compile_name = compile_event["name"]
            need_bw = "Backward" in compile_name

            if need_bw:
                try:
                    compile_name = _related_compile_region(compile_event)["name"]
                except ValueError:
                    continue

            fw_graph_key = str((compile_name, False))
            bw_graph_keys = _backward_graph_keys(compile_linenos, compile_name)
            if fw_graph_key not in compile_linenos and all(
                bw_graph_key not in compile_linenos for bw_graph_key in bw_graph_keys
            ):
                continue

            for op_id in scoped_ops:
                kernel_events = _kernel_events_for_op(op_id)
                if not kernel_events:
                    continue
                for kernel_event in kernel_events:
                    assigned_stack = False
                    warn_if_unrecognized = False
                    if self._maybe_extern_call(
                        kernel_event,
                        compile_linenos,
                        fw_graph_key,
                        bw_graph_keys,
                        need_bw,
                    ):
                        for extern_call, related_ops in ops_in_extern_region.items():
                            if op_id in related_ops:
                                extern_event = uid_2_events[extern_call]
                                assigned_stack = _assign_stack(
                                    kernel_event,
                                    compile_linenos,
                                    fw_graph_key,
                                    bw_graph_keys,
                                    _extern_kernel_name(extern_event.get("name", "")),
                                )
                                if assigned_stack:
                                    break
                        if not assigned_stack:
                            assigned_stack = _assign_single_stack(
                                kernel_event,
                                compile_linenos,
                                fw_graph_key,
                                bw_graph_keys,
                            )
                    elif not self._maybe_triton_call(kernel_event.get("name", "")):
                        warn_if_unrecognized = True
                        assigned_stack = _assign_single_stack(
                            kernel_event,
                            compile_linenos,
                            fw_graph_key,
                            bw_graph_keys,
                        )
                    else:
                        assigned_stack = _assign_stack(
                            kernel_event,
                            compile_linenos,
                            fw_graph_key,
                            bw_graph_keys,
                            kernel_event.get("name", ""),
                        )
                    if not assigned_stack and warn_if_unrecognized:
                        log.warning(
                            "Kernel %s cannot be recognized as a custom kernel or triton kernel. "
                            "Please try profile with stack=True.",
                            kernel_event.get("name", ""),
                        )
        return trace


def _add_inductor_provenance(path):
    from torch._inductor.debug import get_kernel_information_jsons

    try:
        # Kineto owns the initial serialization path.  Re-read the exported
        # trace here so timeline provenance stays decoupled from Kineto's
        # JSON writer and remains a Python-only post-processing step.
        with open(path) as f:
            trace = json.load(f)

        num_events = len(trace.get("traceEvents", []))
        max_events = inductor_config.trace.provenance_tracking_max_events
        if max_events > 0 and num_events > max_events:
            log.warning(
                "Skipping provenance tracking: trace has %d events "
                "(exceeds limit of %d). Set TORCH_COMPILE_DEBUG_MAX_EVENTS=0 "
                "to disable this protection or increase the limit.",
                num_events,
                max_events,
            )
            return

        if not inductor_config.triton.unique_kernel_names:
            log.warning(
                "Profiling trace does not contain Triton kernel stack traces "
                "because TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=0."
            )
        if inductor_config.cpp_wrapper:
            log.warning(
                "Profiling trace does not contain compiled kernel stack traces "
                "because cpp_wrapper is enabled."
            )
            return
        try:
            log.info("Add stack trace to compiled kernel.")
            trace = _InductorTraceProcessor().add_to_chrome_trace(trace)
            with open(path, "w") as f:
                json.dump(trace, f, indent=1)
        except MemoryError:
            log.error(
                "MemoryError during add_to_chrome_trace. "
                "Try increasing TORCH_COMPILE_DEBUG_MAX_EVENTS or disable provenance tracking."
            )
            raise
        except Exception:
            log.exception("Failed to add stack trace to compiled kernel")
    finally:
        get_kernel_information_jsons().clear()


def inductor_trace_handler(
    dir_name: str,
    worker_name: str | None = None,
    use_gzip: bool = False,
    use_python_export: bool = False,
):
    """
    Outputs tracing files with Inductor provenance to dir_name.

    Provenance tracking must also be enabled with TORCH_COMPILE_DEBUG_EXTEND=1
    or trace.provenance_tracking_to_timeline=True.
    """
    import socket
    import time

    from torch.profiler import kineto_available

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception as e:
                raise RuntimeError("Can't create directory: " + dir_name) from e
        if not worker_name:
            worker_name = f"{socket.gethostname()}_{os.getpid()}"
        file_name = f"{worker_name}.{time.time_ns()}.pt.trace.json"
        if use_gzip:
            file_name = file_name + ".gz"
        path = os.path.join(dir_name, file_name)
        provenance_enabled = (
            kineto_available() and inductor_config.trace.provenance_tracking_to_timeline
        )

        if getattr(prof, "_use_cupti_monitor", False):
            observer = getattr(prof, "_cupti_profiler_observer", None)
            has_monitor_export = (
                observer is not None
                and observer.available
                and getattr(prof, "_monitor_window_id", None) is not None
            )
            prof.export_chrome_trace(
                path,
                use_python_export=use_python_export,
            )
            if provenance_enabled and has_monitor_export:
                from torch._inductor.debug import get_kernel_information_jsons

                get_kernel_information_jsons().clear()
            return

        if use_gzip:
            with tempfile.NamedTemporaryFile("w+b", suffix=".json") as fp:
                prof.export_chrome_trace(
                    fp.name,
                    use_python_export=use_python_export,
                )
                if provenance_enabled:
                    _add_inductor_provenance(fp.name)
                with open(fp.name, "rb") as fin, gzip.open(path, "wb") as fout:
                    fout.writelines(fin)
        else:
            prof.export_chrome_trace(
                path,
                use_python_export=use_python_export,
            )
            if provenance_enabled:
                _add_inductor_provenance(path)

    return handler_fn
<|endoftext|>import argparse
import logging
import os

from torch._inductor.autotune_process import TuningProcess
from torch._inductor.compile_worker.utils import _async_compile_initializer


log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=int)
    parser.add_argument("--read-fd", type=int)
    parser.add_argument("--write-fd", type=int)
    args = parser.parse_args()
    read_pipe = os.fdopen(args.read_fd, "rb")
    write_pipe = os.fdopen(args.write_fd, "wb")

    try:
        # Ensures the subprocess exits if the parent crashes:
        _async_compile_initializer(args.parent)
        TuningProcess.process_main(read_pipe, write_pipe)
    except Exception:
        log.exception("Uncaught exception in autotune subprocess")
    finally:
        read_pipe.close()
        write_pipe.close()


if __name__ == "__main__":
    main()
<|endoftext|>from __future__ import annotations

import collections
import dataclasses
import heapq
import logging
from typing import TYPE_CHECKING, TypedDict

import torch
from torch._environment import is_fbcode
from torch._utils_internal import signpost_event
from torch.utils._ordered_set import OrderedSet

from . import config
from .ir import MultiOutputLayout, NoneLayout
from .utils import get_dtype_size, is_nonfreeable_buffers
from .virtualized import V


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .dependencies import Dep
    from .scheduler import BaseSchedulerNode, SchedulerBuffer

from .dependencies import WeakDep


torch_log = logging.getLogger(__name__)


@dataclasses.dataclass
class PeakMemoryResult:
    order: list[BaseSchedulerNode]
    peak_memory: int
    method: str


@dataclasses.dataclass
class MemoryPlanningInfoForBuffer:
    size_alloc: int = 0
    size_free: int = 0
    # succ_nodes used for buffer lifetime/freeing (excludes is_fake WeakDeps)
    succ_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )
    # succ_nodes used for node ordering (includes is_fake WeakDeps)
    succ_nodes_for_ordering: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )

    def __post_init__(self) -> None:
        torch._check(
            len(self.succ_nodes) <= len(self.succ_nodes_for_ordering),
            lambda: f"succ_nodes must be a subset of succ_nodes_for_ordering. "
            f"len(succ_nodes)={len(self.succ_nodes)}, len(succ_nodes_for_ordering)={len(self.succ_nodes_for_ordering)}",
        )


@dataclasses.dataclass
class MemoryPlanningInfoForNode:
    index: int = 0
    size: int = 0
    pred_buffers: OrderedSet[SchedulerBuffer | FreeableInputBuffer] = dataclasses.field(
        default_factory=OrderedSet
    )
    pred_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )
    succ_nodes: OrderedSet[BaseSchedulerNode] = dataclasses.field(
        default_factory=OrderedSet
    )


@dataclasses.dataclass
class FreeableInputBuffer:
    name: str
    mpi_buffer: MemoryPlanningInfoForBuffer = dataclasses.field(
        default_factory=MemoryPlanningInfoForBuffer
    )

    def get_name(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)


def get_freeable_input_buf(
    nodes: list[BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
) -> dict[str, FreeableInputBuffer]:
    """
    Create and keep track of all input buffers that can be freed during the program

    Returns:
        A dictionary containing all freeable input buffers, keyed by their names.
    """

    def _dep_size_hint(dep: Dep) -> int:
        return V.graph.get_dep_size_hint(dep)

    # get freeable input buffers' successor nodes for memory lifetime (excludes is_fake WeakDeps)
    # and for ordering (includes all deps)
    dep_name_to_succ_nodes: dict[str, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    dep_name_to_succ_nodes_for_ordering: dict[str, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    dep_name_to_size: dict[str, int] = dict()

    for node in nodes:
        for dep in node.read_writes.reads:
            if dep.name in graph_inputs:
                if not is_nonfreeable_buffers(dep):
                    # All deps contribute to ordering, but fake weak deps do not contribute to
                    # memory liveness
                    dep_name_to_succ_nodes_for_ordering[dep.name].add(node)
                    dep_name_to_size[dep.name] = _dep_size_hint(dep)
                    if not (isinstance(dep, WeakDep) and dep.is_fake):
                        dep_name_to_succ_nodes[dep.name].add(node)

    # create FreeableInputBuffer objects and add them to the returned dictionary
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer] = dict()
    for dep_name in dep_name_to_succ_nodes_for_ordering:
        name_to_freeable_input_buf[dep_name] = FreeableInputBuffer(
            dep_name,
            MemoryPlanningInfoForBuffer(
                size_free=dep_name_to_size[dep_name],
                succ_nodes=dep_name_to_succ_nodes[dep_name],
                succ_nodes_for_ordering=dep_name_to_succ_nodes_for_ordering[dep_name],
            ),
        )
    return name_to_freeable_input_buf


def compute_size_for_scheduler_buffer(
    name_to_buf: dict[str, SchedulerBuffer],
) -> dict[str, tuple[int, int]]:
    """
    Compute the size of each scheduler buffer, including (1) memory allocated when
    it is created and (2) memory deallocated when it is freed.

    We specially handle the case of MultiOutputLayout.
    Consider the following case:
        buf0 = some_ops_with_multi_outputs(...)
        buf1 = buf0[0] # assume 10 bytes
        buf2 = buf0[1] # assume 20 bytes
    In such cases,
        buf0: at creation, 30 bytes allocated, when deleted, 0 bytes freed
        buf1: at creation, 0 bytes allocated, when deleted, 10 bytes freed
        buf2: at creation, 0 bytes allocated, when deleted, 20 bytes freed

    When an operation mutates a buffer in-place, the scheduler creates a new buffer name
    to track the "before" and "after" states, even though they share the same memory.

    The mutated buffer represents a rename with zero allocation and deallocation cost.
    During dependency tracking, we transfer dependencies from the mutated name back to
    the original buffer, ensuring the original memory is only freed when all aliases
    are done.

    This handles cases where a buffer has multiple non-overlapping aliases - rather than
    trying to assign free costs to individual aliases, we forward all alias dependencies
    to the original buffer.

    Consider:
        buf0 = op0()
        buf1 = mutation_op_(buf0)
        del buf0
        ...
        op(buf1)
        del buf1

    The only memory events are the creation prior to op0, and the deletion following buf1.

    Returns:
        A dictionary mapping a scheduler buffer to a tuple of (size_alloc, size_free).
    """
    from .ir import MultiOutput
    from .scheduler import OutputNode

    sched_buf_to_size: dict[str, tuple[int, int]] = dict()

    def _compute_and_update_buf_size(
        sched_buf: SchedulerBuffer, user_of_MultiOutputLayout: bool = False
    ) -> int:
        if sched_buf.get_name() in V.graph.scheduler.mutation_real_name:
            sched_buf_to_size[sched_buf.get_name()] = (0, 0)
            return 0
        elif isinstance(sched_buf.node.layout, NoneLayout):
            sched_buf_to_size[sched_buf.get_name()] = (0, 0)
            return 0
        elif isinstance(sched_buf.node.layout, MultiOutputLayout):
            size_alloc = 0
            for user in sched_buf.users:
                if isinstance(user.node, OutputNode):
                    continue
                for buf in user.node.get_outputs():
                    if isinstance(buf.node, MultiOutput):
                        size_alloc += _compute_and_update_buf_size(buf, True)
            sched_buf_to_size[sched_buf.get_name()] = (
                0 if user_of_MultiOutputLayout else size_alloc,
                0,
            )
            return size_alloc
        else:
            buf_size = V.graph.sizevars.optimization_hint(
                sched_buf.node.get_numel(), fallback=0
            ) * get_dtype_size(sched_buf.node.get_dtype())
            sched_buf_to_size[sched_buf.get_name()] = (
                0 if user_of_MultiOutputLayout else buf_size,
                buf_size,
            )
            return buf_size

    for sched_buf in name_to_buf.values():
        # skip if sched_buf is already processed as a user of another SchedulerBuffer
        # whose layout is of the type MultiOutputLayout
        if sched_buf.get_name() not in sched_buf_to_size:
            _compute_and_update_buf_size(sched_buf)

    return sched_buf_to_size


def assign_memory_planning_info_for_scheduler_buffers(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
) -> None:
    """
    For each SchedulerBuffer, assign its size info and successor nodes.
    A buffer's successor nodes determines when a buffer can be freed.
    """
    # get buffer sizes
    sched_buf_to_size = compute_size_for_scheduler_buffer(name_to_buf)

    # get buffer's successor nodes for memory lifetime (excludes is_fake WeakDeps)
    # and for ordering (includes all deps)
    dep_name_to_succ_nodes: dict[str, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    dep_name_to_succ_nodes_for_ordering: dict[str, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    for node in nodes:
        for dep in node.unmet_dependencies:
            # All deps contribute to ordering, but fake weak deps do not contribute to
            # memory liveness
            dep_name_to_succ_nodes_for_ordering[dep.name].add(node)
            if not (isinstance(dep, WeakDep) and dep.is_fake):
                dep_name_to_succ_nodes[dep.name].add(node)

    # iterate in reverse, so dependencies are picked up transitively.
    for mutating_buf_name, real_buf_name in reversed(
        V.graph.scheduler.mutation_real_name.items()
    ):
        dep_name_to_succ_nodes[real_buf_name] |= dep_name_to_succ_nodes[
            mutating_buf_name
        ]
        dep_name_to_succ_nodes_for_ordering[real_buf_name] |= (
            dep_name_to_succ_nodes_for_ordering[mutating_buf_name]
        )

    # populate the MemoryPlanningInfoForBuffer attribute to each scheduler buffer
    # note: there are scheduler buffers not in dep_name_to_succ_nodes (e.g., graph outputs)
    for buf_name in name_to_buf:
        name_to_buf[buf_name].mpi_buffer = MemoryPlanningInfoForBuffer(
            size_alloc=sched_buf_to_size[buf_name][0],
            size_free=sched_buf_to_size[buf_name][1],
            succ_nodes=dep_name_to_succ_nodes[buf_name],
            succ_nodes_for_ordering=dep_name_to_succ_nodes_for_ordering[buf_name],
        )


def assign_memory_planning_info_for_scheduler_nodes(
    nodes: list[BaseSchedulerNode],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
) -> None:
    """
    Assign to each scheduler node its predecessor and successor nodes.
    """

    node_to_pred_nodes: dict[BaseSchedulerNode, OrderedSet[BaseSchedulerNode]] = (
        collections.defaultdict(OrderedSet)
    )
    node_to_succ_nodes: dict[BaseSchedulerNode, OrderedSet[BaseSchedulerNode]] = {}
    node_to_pred_buffers: dict[
        BaseSchedulerNode, OrderedSet[SchedulerBuffer | FreeableInputBuffer]
    ] = collections.defaultdict(OrderedSet)

    # collect all predecessors using existing successor mappings
    for node in nodes:
        succ_nodes = OrderedSet(
            succ_node
            for buffer in node.get_outputs()
            for succ_node in buffer.mpi_buffer.succ_nodes_for_ordering
        )
        node_to_succ_nodes[node] = succ_nodes

        # For each successor, add current node as its predecessor
        for succ_node in succ_nodes:
            node_to_pred_nodes[succ_node].add(node)

        # For each output buffer, add it as predecessor to its successor nodes
        # Use succ_nodes (not succ_nodes_for_ordering) since pred_buffers is used
        # for memory lifetime tracking, not ordering
        for buffer in node.get_outputs():
            for succ_node in buffer.mpi_buffer.succ_nodes:
                node_to_pred_buffers[succ_node].add(buffer)

    for freeable_buffer in name_to_freeable_input_buf.values():
        for succ_node in freeable_buffer.mpi_buffer.succ_nodes:
            node_to_pred_buffers[succ_node].add(freeable_buffer)

    # Second pass: assign memory planning info using completed predecessor mappings
    for index, node in enumerate(nodes):
        size_alloc = sum(buffer.mpi_buffer.size_alloc for buffer in node.get_outputs())
        succ_nodes = node_to_succ_nodes[node]
        pred_nodes = node_to_pred_nodes[node]

        # make sure we do not make node a successor or predecessor of itself
        succ_nodes.discard(node)
        pred_nodes.discard(node)

        node.mpi_node = MemoryPlanningInfoForNode(
            index=index,
            size=size_alloc,
            pred_buffers=node_to_pred_buffers[node],
            pred_nodes=node_to_pred_nodes[node],
            succ_nodes=succ_nodes,
        )


# map each scheduler buffer to its size, start step, and end step
@dataclasses.dataclass
class BufferInfo:
    buffer: SchedulerBuffer | FreeableInputBuffer
    size_alloc: int
    size_free: int
    start_step: int
    end_step: int


def compute_memory_timeline(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[
    list[BufferInfo],
    dict[BaseSchedulerNode, int],
    dict[FreeableInputBuffer | SchedulerBuffer, BaseSchedulerNode],
]:
    """
    Compute buffer allocation and deallocation sizes and map their
    lifetime to the node schedule
    """

    # get the execution step of each node, this will be used to determine
    # the end_step of buffers
    node_to_step: dict[BaseSchedulerNode, int] = {
        node: step for step, node in enumerate(nodes)
    }

    # get buffers' size and liveliness information
    buf_info_list: list[BufferInfo] = []
    buf_to_snode_last_use: dict[
        FreeableInputBuffer | SchedulerBuffer, BaseSchedulerNode
    ] = {}

    def _get_end_step_and_snode(
        buf: FreeableInputBuffer | SchedulerBuffer,
    ) -> tuple[int, BaseSchedulerNode | None]:
        max_step: int = -1
        max_step_snode: BaseSchedulerNode | None = None
        succ_nodes = buf.mpi_buffer.succ_nodes
        if succ_nodes:
            for succ_node in succ_nodes:
                step = node_to_step[succ_node]
                if step > max_step:
                    max_step = step
                    max_step_snode = succ_node
            if max_step_snode is None:
                raise AssertionError("expected max_step_snode to be set")
        return max_step, max_step_snode

    # 1. for freeable input buffers
    for buf_name, input_buf in name_to_freeable_input_buf.items():
        end_step = -1
        if buf_name not in graph_outputs:
            end_step, end_step_snode = _get_end_step_and_snode(input_buf)
            if end_step_snode is None:
                raise AssertionError("expected end_step_snode to be set")
            buf_to_snode_last_use[input_buf] = end_step_snode

        buf_info_list.append(
            BufferInfo(
                input_buf,
                input_buf.mpi_buffer.size_free,
                input_buf.mpi_buffer.size_free,
                0,
                end_step,
            )
        )

    # 2. for scheduler buffers
    for step, node in enumerate(nodes):
        for sched_buf in node.get_outputs():
            # note: it is possible for a non-graph-output sched_buf to have no succ_nodes and
            # to be only used by its defining op (e.g., due to fusion when all consumers of
            # the buffer are fused with its defining op). In such cases, end_step is step.
            buf_name = sched_buf.get_name()
            end_step = -1
            if buf_name not in graph_outputs:
                end_step, end_step_snode = _get_end_step_and_snode(sched_buf)
                if end_step == -1:
                    end_step = step
                    buf_to_snode_last_use[sched_buf] = node
                else:
                    if end_step_snode is None:
                        raise AssertionError("expected end_step_snode to be set")
                    buf_to_snode_last_use[sched_buf] = end_step_snode

            buf_info_list.append(
                BufferInfo(
                    sched_buf,
                    sched_buf.mpi_buffer.size_alloc,
                    sched_buf.mpi_buffer.size_free,
                    step,
                    end_step,
                )
            )

    return buf_info_list, node_to_step, buf_to_snode_last_use


def peak_memory_from_buf_info_list(
    buf_info_list: list[BufferInfo], num_steps: int
) -> tuple[int, list[int]]:
    """Compute peak and per-step live memory from buffer lifetimes.

    Skip the free for graph-output buffers (`end_step == -1`); otherwise
    `delta[0] -= size_free` would cancel the alloc.
    """
    delta = [0] * (num_steps + 1)
    for bi in buf_info_list:
        delta[bi.start_step] += bi.size_alloc
        if bi.end_step != -1:
            delta[bi.end_step + 1] -= bi.size_free

    max_memory = 0
    cur_memory = 0
    memories_at_nodes = [0] * (num_steps + 1)
    for t in range(num_steps + 1):
        cur_memory += delta[t]
        memories_at_nodes[t] = cur_memory
        if cur_memory > max_memory:
            max_memory = cur_memory
    return max_memory, memories_at_nodes


def live_memory_before_steps_from_buf_info_list(
    buf_info_list: list[BufferInfo], num_steps: int
) -> list[int]:
    """Compute live bytes before each scheduler step runs."""
    delta = [0] * (num_steps + 1)
    cur_memory = 0
    for bi in buf_info_list:
        if isinstance(bi.buffer, FreeableInputBuffer):
            cur_memory += bi.size_alloc
        else:
            delta[bi.start_step + 1] += bi.size_alloc

        if bi.end_step != -1:
            delta[bi.end_step + 1] -= bi.size_free

    live_before = [0] * (num_steps + 1)
    live_before[0] = cur_memory
    for t in range(1, num_steps + 1):
        cur_memory += delta[t]
        live_before[t] = cur_memory
    return live_before


def estimate_peak_memory(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[int, list[int]]:
    """
    Given a list of nodes in their execution order, estimate the peak memory, by
    keeping track of the liveliness of SchedulerBuffers and FreeableInputBuffers.

    Returns:
        int: peak memory
        List[int]: memory usage at each node (or each step).
    """
    buf_info_list, _, _ = compute_memory_timeline(
        nodes, name_to_freeable_input_buf, graph_outputs
    )
    return peak_memory_from_buf_info_list(buf_info_list, len(nodes))


def estimate_region_peak_memory(
    nodes_in_window: Iterable[BaseSchedulerNode],
    *,
    region_start: int,
    region_end: int,
    step_of: Callable[[BaseSchedulerNode], int],
    graph_outputs: OrderedSet[str],
    cur_memory: int = 0,
) -> int:
    """Peak memory inside `[region_start, region_end]` for the
    hypothetical post-reorder schedule.

    Walks `nodes_in_window` (post-rewrite, in proposed order). For
    each node: alloc = sum of `size_alloc` over its outputs; free =
    sum of `size_free` over `pred_buffers` whose proposed last
    consumer is this node. Then accumulates per step starting from
    `cur_memory` (live bytes at the window boundary) and returns
    the maximum live bytes.
    """
    R = region_end - region_start + 1
    region = [SNodeMemory(0, 0) for _ in range(R)]

    for node in nodes_in_window:
        s = step_of(node)
        slot = s - region_start
        if not (0 <= slot < R):
            raise AssertionError(f"expected 0 <= slot < {R}, got {slot}")

        for buf in node.get_outputs():
            bi = buf.mpi_buffer
            region[slot].size_alloc += bi.size_alloc
            name = buf.get_name()
            if name in graph_outputs:
                continue
            succ_steps = [step_of(n) for n in bi.succ_nodes]
            if not succ_steps:
                region[slot].size_free += bi.size_free

        for pb in node.mpi_node.pred_buffers:
            name = pb.get_name()
            if name in graph_outputs:
                continue
            succ_steps = [step_of(n) for n in pb.mpi_buffer.succ_nodes]
            if not succ_steps:
                raise AssertionError("expected non-empty succ_steps")
            if max(succ_steps) == s:
                region[slot].size_free += pb.mpi_buffer.size_free

    cur = cur_memory
    peak = cur
    for af in region:
        cur += af.size_alloc
        if cur > peak:
            peak = cur
        cur -= af.size_free
    return peak


@dataclasses.dataclass
class SNodeMemory:
    size_alloc: int
    size_free: int


def estimate_peak_memory_allocfree(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    graph_outputs: OrderedSet[str],
) -> tuple[
    int,
    list[tuple[int, int]],
    dict[BaseSchedulerNode, SNodeMemory],
    dict[FreeableInputBuffer | SchedulerBuffer, BaseSchedulerNode],
]:
    """
    Alternative version of estimate_peak_memory, that respects the fact,
    that every SchedulerNode has multiple phases:
    1. alloc ( outputs )
    2. run_kernel
    3. dealloc last_use buffers
    estimate_peak_memory collapses memory into one value: size_alloc - size_free
    While peak memory happens after alloc.

    Duplicating the code to not migrate all callsites at once,
    In future usages of estimate_peak_memory will migrate to this version.
    """

    buf_info_list, _, buf_to_snode_last_use = compute_memory_timeline(
        nodes, name_to_freeable_input_buf, graph_outputs
    )

    # incremental memory changes at each step
    step_idx_allocfree = [SNodeMemory(0, 0) for _ in range(len(nodes))]

    # for each buffer, update memory when created and when freed
    for buf_info in buf_info_list:
        step_idx_allocfree[buf_info.start_step].size_alloc += buf_info.size_alloc
        if buf_info.end_step != -1:
            step_idx_allocfree[buf_info.end_step].size_free += buf_info.size_free

    snodes_allocfree = {}
    for i, node in enumerate(nodes):
        snodes_allocfree[node] = step_idx_allocfree[i]

    max_memory = 0
    cur_memory = 0
    snodes_curr_memory = []
    for t in range(len(nodes)):
        alloc = step_idx_allocfree[t].size_alloc
        free = step_idx_allocfree[t].size_free
        cur_memory += alloc
        post_alloc = cur_memory
        max_memory = max(max_memory, cur_memory)
        cur_memory -= free
        post_free = cur_memory
        snodes_curr_memory.append((post_alloc, post_free))

    return (
        max_memory,
        snodes_curr_memory,
        snodes_allocfree,
        buf_to_snode_last_use,
    )


def topological_sort_lpmf(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    name_to_buf: dict[str, SchedulerBuffer],
    graph_outputs: OrderedSet[str],
) -> list[BaseSchedulerNode]:
    """
    A bfs-based greedy topological order. LPMF stands for "Least Peak Memory First".

    The idea is from this paper:
    Buffer memory optimization for video codec application modeled in Simulink
    https://www.cs.york.ac.uk/rts/docs/DAC-1964-2006/PAPERS/2006/DAC06/PDFFILES/P0689.PDF

    The algorithm maintains the max memory so far.
    At every iteration, for each scheduleable node, it computes:
        - how much memory needs to be allocated for the output buffers of this node;
        - how much memory can be freed as a result of executing this node.
    This gives us two values for each node:
        (1) mem1: memory during the execution of the node;
        (2) mem2: memory after executing the node, after some input buffers are freed.
    The greedy approach select as follows:
        (i) if there are nodes whose mem1 values are below the max memory so far,
            then pick the node with the lowest mem2 value;
        (ii) otherwise, pick the one with the lowest mem1 value.
    """

    class NodeInfo(TypedDict):
        indegree: int
        memory_to_free: int

    class BufferInfo(TypedDict):
        outdegree: int

    node_info: dict[BaseSchedulerNode, NodeInfo] = dict()
    buf_info: dict[SchedulerBuffer | FreeableInputBuffer, BufferInfo] = dict()

    # compute nodes' number of unmet dependencies (for schedulability)
    # initialize the list of nodes ready to be scheduled
    nodes_to_schedule: OrderedSet[BaseSchedulerNode] = OrderedSet()
    for node in nodes:
        node_info[node] = {
            "indegree": len(node.mpi_node.pred_nodes),
            "memory_to_free": 0,
        }
        if node_info[node]["indegree"] == 0:
            nodes_to_schedule.add(node)

    # compute buffers' number of unmet successors (used to decide when to free)
    for buf in list(name_to_buf.values()) + list(name_to_freeable_input_buf.values()):
        buf_info[buf] = {
            "outdegree": len(buf.mpi_buffer.succ_nodes)
            + (1 if buf.get_name() in graph_outputs else 0)
        }

    # initialize memory estimations
    live_memory = sum(
        input_buf.mpi_buffer.size_free
        for input_buf in name_to_freeable_input_buf.values()
    )

    # this is the total output memory, which is a lower bound for peak memory
    # we do not include the memory of non freeable input buffers
    output_memory = 0
    for buf_name in graph_outputs:
        if buf_name in name_to_buf:
            output_memory += name_to_buf[buf_name].mpi_buffer.size_free
        elif buf_name in name_to_freeable_input_buf:
            output_memory += name_to_freeable_input_buf[buf_name].mpi_buffer.size_free
    max_memory = max(live_memory, output_memory)
    memory_gap = max_memory - live_memory

    # compute the amount of memory that is allocated when a node is scheduled
    # and the amount of memory that can be freed when a node is scheduled
    for node in nodes:
        # 1. if a buffer read by this node is last used by this node
        for buf in node.mpi_node.pred_buffers:
            if buf_info[buf]["outdegree"] == 1:
                node_info[node]["memory_to_free"] += buf.mpi_buffer.size_free
        # 2. if a buffer written by this node is used internally and not used later
        for buf in node.get_outputs():
            if buf_info[buf]["outdegree"] == 0:
                node_info[node]["memory_to_free"] += buf.mpi_buffer.size_free

    # schedule nodes one at a time
    schedule: list[BaseSchedulerNode] = []
    size_threshold = config.size_threshold_for_succ_based_strategy
    num_iters: int = 0
    while num_iters < len(nodes) and nodes_to_schedule:
        # select a node to schedule:
        if (
            size_threshold > 0
            and min(node.mpi_node.size for node in nodes_to_schedule) > size_threshold
        ):
            selected_node = min(
                nodes_to_schedule,
                key=lambda node: min(
                    (
                        succ_node.mpi_node.index
                        for succ_node in node.mpi_node.succ_nodes
                    ),
                    default=len(nodes),
                ),
            )
        else:
            selected_node = min(
                nodes_to_schedule,
                key=lambda node: (
                    node.mpi_node.size if node.mpi_node.size > memory_gap else 0,
                    node.mpi_node.size - node_info[node]["memory_to_free"],
                    node.mpi_node.index,
                ),
            )
        nodes_to_schedule.remove(selected_node)
        schedule.append(selected_node)
        num_iters += 1

        # update memory usage
        live_memory += selected_node.mpi_node.size
        max_memory = max(max_memory, live_memory)
        live_memory -= node_info[selected_node]["memory_to_free"]
        memory_gap = max_memory - live_memory

        # update successor nodes and nodes_to_schedule
        for succ_node in selected_node.mpi_node.succ_nodes:
            if node_info[succ_node]["indegree"] <= 0:
                raise AssertionError(
                    f"expected positive indegree, got {node_info[succ_node]['indegree']}"
                )
            node_info[succ_node]["indegree"] -= 1
            if node_info[succ_node]["indegree"] == 0:
                nodes_to_schedule.add(succ_node)

        # update predecessor nodes
        for buf in selected_node.mpi_node.pred_buffers:
            if buf_info[buf]["outdegree"] <= 0:
                raise AssertionError(
                    f"expected positive outdegree, got {buf_info[buf]['outdegree']}"
                )
            buf_info[buf]["outdegree"] -= 1
            if buf_info[buf]["outdegree"] == 1:
                for succ_node in buf.mpi_buffer.succ_nodes:
                    node_info[succ_node]["memory_to_free"] += buf.mpi_buffer.size_free

    if num_iters > len(nodes):
        raise RuntimeError("Failed to schedule, while loop ran too long for lpmf")

    return schedule


def topological_sort_bfs(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    A BFS topological sort that selects nodes whose dependencies are executed the
    earliest. This follows a FIFO idea. Specifically, at every iteration, for each node
    that is schedulable, we gather the order in which its predecessor nodes are executed,
    and this sorted list of execution orders of predecessor nodes defines the priority.
    We select the node whose predecessors nodes are executed the earliest. The FIFO
    idea aims to reduce the liveness duration of buffers created.
    """

    class NodeInfo(TypedDict):
        indegree: int
        order: int

    node_info: dict[BaseSchedulerNode, NodeInfo] = dict()

    @dataclasses.dataclass
    class NodeWithPriority:
        priority: list[int]
        node: BaseSchedulerNode

        def __lt__(self, other: NodeWithPriority) -> bool:
            if self.priority == other.priority:
                return self.node.mpi_node.index < other.node.mpi_node.index
            return self.priority < other.priority

    def _node_priority(node: BaseSchedulerNode) -> list[int]:
        # priority is the order in which predecessor nodes are executed
        if node_info[node]["indegree"] != 0:
            raise AssertionError(
                f"expected zero indegree, got {node_info[node]['indegree']}"
            )
        exec_orders = sorted(
            OrderedSet(
                node_info[pred_node]["order"] for pred_node in node.mpi_node.pred_nodes
            )
        )
        return exec_orders

    # compute nodes' number of unmet dependencies (for schedulability)
    # initialize the list of nodes ready to be scheduled
    nodes_to_schedule: list[NodeWithPriority] = []
    for node in nodes:
        node_info[node] = {"indegree": len(node.mpi_node.pred_nodes), "order": -1}
        if node_info[node]["indegree"] == 0:
            heapq.heappush(
                nodes_to_schedule, NodeWithPriority(_node_priority(node), node)
            )

    # schedule nodes one at a time
    schedule: list[BaseSchedulerNode] = []
    num_iters: int = 0
    while num_iters < len(nodes) and nodes_to_schedule:
        # select a node to schedule
        selected_node = heapq.heappop(nodes_to_schedule).node
        node_info[selected_node]["order"] = len(schedule)
        schedule.append(selected_node)
        num_iters += 1

        # update successor nodes and nodes_to_schedule
        for succ_node in selected_node.mpi_node.succ_nodes:
            if node_info[succ_node]["indegree"] <= 0:
                raise AssertionError(
                    f"expected positive indegree, got {node_info[succ_node]['indegree']}"
                )
            node_info[succ_node]["indegree"] -= 1
            if node_info[succ_node]["indegree"] == 0:
                heapq.heappush(
                    nodes_to_schedule,
                    NodeWithPriority(_node_priority(succ_node), succ_node),
                )

    if num_iters > len(nodes):
        raise RuntimeError("Failed to schedule, while loop ran too long for bfs")

    return schedule


def topological_sort_dfs(nodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    This is a DFS topological sort. The setup is similar to `topological_sort_schedule`
    in scheduler.py. The difference is the order nodes are visited in the outer loop.
    In `topological_sort_schedule`, nodes are visited in their original order.
    In this function, nodes are visited based on their priority -- for each node, we
    compute the total memory of all buffers it reads from or writes to, and we visit
    the nodes in ascending order of this priority.
    """
    seen: OrderedSet[BaseSchedulerNode] = OrderedSet()
    name_to_node: dict[str, BaseSchedulerNode] = dict()
    result: list[BaseSchedulerNode] = []
    size_with_reads: dict[BaseSchedulerNode, int] = dict()

    def visit(n: BaseSchedulerNode) -> None:
        if n not in seen:
            seen.add(n)
            dep_nodes = [
                name_to_node[dep.name]
                for dep in n.unmet_dependencies
                if dep.name in name_to_node
            ]
            for node in sorted(
                dep_nodes, key=lambda n: (size_with_reads[n], n.mpi_node.index)
            ):
                visit(node)
            result.append(n)

    for node in nodes:
        for name in node.get_buffer_names():
            name_to_node[name] = node

    for node in nodes:
        size_with_reads[node] = node.mpi_node.size + sum(
            pred_buf.mpi_buffer.size_free for pred_buf in node.mpi_node.pred_buffers
        )
    for node in sorted(nodes, key=lambda n: (size_with_reads[n], n.mpi_node.index)):
        visit(node)

    return result


def validate_graph_acyclic(nodes: list[BaseSchedulerNode]) -> None:
    """
    Validate that the graph is acyclic by checking predecessor relationships.

    Raises:
        RuntimeError: If a cycle is detected in the graph
    """
    # DFS coloring scheme for cycle detection:
    # WHITE (0): Node has not been visited yet
    # GRAY (1): Node is currently being processed (in the recursion stack)
    # BLACK (2): Node has been completely processed (finished exploring all its predecessors)
    # A back edge (cycle) is detected when we encounter a GRAY node during DFS traversal
    WHITE, GRAY, BLACK = 0, 1, 2
    color = dict.fromkeys(nodes, WHITE)
    path: list[BaseSchedulerNode] = []  # Track current DFS path

    def dfs_visit(node: BaseSchedulerNode) -> None:
        if color[node] == BLACK:
            return

        if color[node] == GRAY:
            path.append(node)
            path_info = " -> ".join([node.get_name() for node in path])

            raise RuntimeError(
                f"Cycle detected in memory planning graph"
                f"Path containing cycle (i -> j: j is a dependency of i): {path_info} "
                f"This indicates invalid dependency relationships in the scheduler graph"
            )

        color[node] = GRAY
        path.append(node)

        for pred_node in node.mpi_node.pred_nodes:
            if pred_node == node:
                raise AssertionError("expected pred_node != node (self-loop)")
            dfs_visit(pred_node)

        path.pop()
        color[node] = BLACK

    # Start DFS from all unvisited nodes
    for node in nodes:
        if color[node] == WHITE:
            dfs_visit(node)


def validate_unique_buffer_names(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
) -> None:
    """
    Validate that for each node's output buffer, the name_to_buf mapping is correct.
    For each output buffer buf, we should have name_to_buf[buf.get_name()] == buf.
    Also validate that no buffer names overlap with freeable input buffer names.

    Raises:
        RuntimeError: If buffer name mapping is incorrect or names overlap
    """
    for node in nodes:
        for buf in node.get_outputs():
            buf_name = buf.get_name()

            # Check if buffer name exists in the mapping
            if buf_name not in name_to_buf:
                raise RuntimeError(
                    f"{buf_name} from {node.get_name()} is not found in name_to_buf mapping."
                    f" This indicates a missing buffer mapping."
                )

            # Check if the mapping points to the correct buffer object
            if name_to_buf[buf_name] != buf:
                raise RuntimeError(
                    f"Buffer name mapping is incorrect for '{buf_name}'."
                    f"Expected name_to_buf['{buf_name}'] to be {buf.debug_str()}"
                    f"but got {name_to_buf[buf_name].debug_str()}"
                    f"This indicates some buffers share the same name"
                )

            # Check if buffer name conflicts with freeable input buffer names
            if buf_name in name_to_freeable_input_buf:
                raise RuntimeError(
                    f"Buffer name conflict detected: '{buf_name}' from node {node.get_name()} "
                    f"is also used as a freeable input buffer name. "
                )


def prepare_planning_info(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
) -> tuple[int, dict[str, FreeableInputBuffer]]:
    """
    Prepare planning info. As nodes are scheduled one at a time, these help
    keep track of when a buffer can be freed, and when a node can be scheduled

    Returns:
        int: peak memory estimation
        dict[str, FreeableInputBuffer]: name to freeable input buffer
    """
    name_to_freeable_input_buf = get_freeable_input_buf(nodes, graph_inputs)
    assign_memory_planning_info_for_scheduler_buffers(nodes, name_to_buf)
    assign_memory_planning_info_for_scheduler_nodes(
        nodes, name_to_fused_node, name_to_buf, name_to_freeable_input_buf
    )

    # the default
    estimated_peak_memory, _ = estimate_peak_memory(
        nodes, name_to_freeable_input_buf, graph_outputs
    )

    return estimated_peak_memory, name_to_freeable_input_buf


def reorder_for_peak_memory(
    nodes: list[BaseSchedulerNode],
    name_to_buf: dict[str, SchedulerBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
    methods: list[Callable[..., list[BaseSchedulerNode]]] = [  # noqa: B006
        topological_sort_lpmf,
        topological_sort_bfs,
        topological_sort_dfs,
    ],
) -> list[BaseSchedulerNode]:
    """
    Try a few heuristics based topological sort algorithms, and pick the one whose
    resulting topological order has the lowest peak memory estimation.
    """

    torch_log.info("Reordering for peak memory -- %d nodes", len(nodes))

    estimated_peak_memory, name_to_freeable_input_buf = prepare_planning_info(
        nodes,
        name_to_buf,
        name_to_fused_node,
        graph_inputs,
        graph_outputs,
    )

    # export graph for simulator if needed
    if config.reorder_for_peak_memory_debug:
        export_graph_for_simulator(
            nodes,
            name_to_freeable_input_buf,
            name_to_fused_node,
            graph_inputs,
            graph_outputs,
        )

    # Validate planning info before proceeding with reordering
    try:
        validate_graph_acyclic(nodes)
        validate_unique_buffer_names(nodes, name_to_buf, name_to_freeable_input_buf)
    except RuntimeError:
        torch_log.exception("Memory planning validation failed")
        if not is_fbcode():  # TODO: remove after ensuring OSS side is safe
            raise

    # keep track of the peak memory estimates of different methods
    peak_memory_diff_methods: list[PeakMemoryResult] = []
    peak_memory_diff_methods.append(
        PeakMemoryResult(nodes, estimated_peak_memory, "baseline")
    )
    torch_log.info("Baseline peak memory: %d", estimated_peak_memory)

    # other methods
    for method in methods:
        try:
            if method is topological_sort_lpmf:
                order = method(
                    nodes, name_to_freeable_input_buf, name_to_buf, graph_outputs
                )
            else:
                order = method(nodes)
            if len(order) != len(nodes):
                raise AssertionError(
                    f"expected order length {len(nodes)}, got {len(order)}"
                )
            peak_memory, _ = estimate_peak_memory(
                order, name_to_freeable_input_buf, graph_outputs
            )
            peak_memory_diff_methods.append(
                PeakMemoryResult(order, peak_memory, method.__name__)
            )
            torch_log.info("%s peak memory: %d", method.__name__, peak_memory)
        except Exception:
            torch_log.exception("Failed to reorder for %s", method.__name__)
            if not is_fbcode():  # TODO: remove after ensuring OSS side is safe
                raise

    signpost_event(
        category="inductor",
        name="memory",
        parameters={
            "orm": {elem.method: elem.peak_memory for elem in peak_memory_diff_methods},
        },
    )

    # get the optimal one
    best_result = min(peak_memory_diff_methods, key=lambda x: x.peak_memory)

    return best_result.order


def export_graph_for_simulator(
    nodes: list[BaseSchedulerNode],
    name_to_freeable_input_buf: dict[str, FreeableInputBuffer],
    name_to_fused_node: dict[str, BaseSchedulerNode],
    graph_inputs: OrderedSet[str],
    graph_outputs: OrderedSet[str],
) -> None:
    """
    This is for debugging purposes. It will dump a json file that records graph information.
    The graph can then be used in a simulator: https://fburl.com/code/3l3d3qi4
    """

    class ORMBuffer(TypedDict):
        name: str
        size_alloc: int
        size_free: int
        size: int  # for backward compatibility
        is_input: bool
        is_output: bool
        deps: list[str]
        unmet_deps: list[str]

    class ORMNode(TypedDict):
        name: str
        buffer_names: list[str]

    class ORMGraph(TypedDict):
        nodes: list[ORMNode]
        buffers: list[ORMBuffer]

    orm_buffers: list[ORMBuffer] = []
    orm_nodes: list[ORMNode] = []

    # get orm buffers for freeable input buffers
    for buf_name, input_buf in name_to_freeable_input_buf.items():
        orm_buf_input_buffer: ORMBuffer = {
            "name": buf_name,
            "size_alloc": input_buf.mpi_buffer.size_free,
            "size_free": input_buf.mpi_buffer.size_free,
            "size": input_buf.mpi_buffer.size_free,
            "is_input": True,
            "is_output": buf_name in graph_outputs,
            "deps": [],
            "unmet_deps": [],
        }
        orm_buffers.append(orm_buf_input_buffer)

    # get orm buffers for scheduler buffers
    name_to_buf: dict[str, SchedulerBuffer] = {
        buf.get_name(): buf for node in nodes for buf in node.get_outputs()
    }  # need to reassign due to probably node pruning
    for buf_name, sched_buf in name_to_buf.items():
        if sched_buf.defining_op is None:
            continue
        deps = [
            pred_buf.get_name()
            for pred_buf in name_to_fused_node[
                sched_buf.defining_op.get_name()
            ].mpi_node.pred_buffers
        ]
        orm_buf_scheduler_buffer: ORMBuffer = {
            "name": buf_name,
            "size_alloc": sched_buf.mpi_buffer.size_alloc,
            "size_free": sched_buf.mpi_buffer.size_free,
            "size": sched_buf.mpi_buffer.size_free,
            "is_input": False,
            "is_output": buf_name in graph_outputs,
            "deps": deps,
            "unmet_deps": [
                buf_name for buf_name in deps if buf_name not in graph_inputs
            ],
        }
        orm_buffers.append(orm_buf_scheduler_buffer)

    # get orm nodes
    for node in nodes:
        orm_node: ORMNode = {
            "name": node.get_name(),
            "buffer_names": list(node.get_buffer_names()),
        }
        orm_nodes.append(orm_node)

    # create the graph object
    g: ORMGraph = {
        "nodes": orm_nodes,
        "buffers": orm_buffers,
    }

    # dump the graph
    import json
    import os

    import torch
    from functorch.compile import get_graph_being_compiled

    name = os.path.splitext(get_graph_being_compiled())[0] + "_fused"

    g_str = json.dumps(g, indent=2)

    torch._logging.trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": name,
            "encoding": "string",
        },
        payload_fn=lambda: g_str,
    )
<|endoftext|>from __future__ import annotations

import os
import tempfile
import textwrap
from functools import lru_cache
from typing import Any, TYPE_CHECKING

from torch._dynamo.exc import BackendCompilerFailed, ShortenTraceback


if TYPE_CHECKING:
    import types

    from torch.cuda import _CudaDeviceProperties

if os.environ.get("TORCHINDUCTOR_WRITE_MISSING_OPS") == "1":

    @lru_cache(None)
    def _record_missing_op(target: Any) -> None:
        with open(f"{tempfile.gettempdir()}/missing_ops.txt", "a") as fd:
            fd.write(str(target) + "\n")

else:

    def _record_missing_op(target: Any) -> None:  # type: ignore[misc]
        pass


class OperatorIssue(RuntimeError):
    @staticmethod
    def operator_str(target: Any, args: list[Any], kwargs: dict[str, Any]) -> str:
        lines = [f"target: {target}"] + [
            f"args[{i}]: {arg}" for i, arg in enumerate(args)
        ]
        if kwargs:
            lines.append(f"kwargs: {kwargs}")
        return textwrap.indent("\n".join(lines), "  ")


class MissingOperatorWithoutDecomp(OperatorIssue):
    def __init__(self, target: Any, args: list[Any], kwargs: dict[str, Any]) -> None:
        _record_missing_op(target)
        super().__init__(f"missing lowering\n{self.operator_str(target, args, kwargs)}")


class MissingOperatorWithDecomp(OperatorIssue):
    def __init__(self, target: Any, args: list[Any], kwargs: dict[str, Any]) -> None:
        _record_missing_op(target)
        super().__init__(
            f"missing decomposition\n{self.operator_str(target, args, kwargs)}"
            + textwrap.dedent(
                f"""

                There is a decomposition available for {target} in
                torch._decomp.get_decompositions().  Please add this operator to the
                `decompositions` list in torch._inductor.decomposition
                """
            )
        )


class LoweringException(OperatorIssue):
    def __init__(
        self,
        exc: Exception,
        target: Any,
        args: list[Any],
        kwargs: dict[str, Any],
        stack_trace: str | None = None,
    ) -> None:
        msg = f"{type(exc).__name__}: {exc}\n{self.operator_str(target, args, kwargs)}"
        if stack_trace:
            msg += f"{msg}\nFound from : \n {stack_trace}"
        super().__init__(msg)


class SubgraphLoweringException(RuntimeError):
    pass


class InvalidCxxCompiler(RuntimeError):
    def __init__(self, compiler: str | None = None) -> None:
        from . import config

        if compiler is None:
            msg = (
                f"No working C++ compiler found in {config.__name__}.cpp.cxx: "
                f"{config.cpp.cxx}"
            )
        else:
            msg = f"Compiler: {compiler} is not found."
        super().__init__(msg)


class CppWrapperCodegenError(RuntimeError):
    def __init__(self, msg: str) -> None:
        super().__init__(f"C++ wrapper codegen error: {msg}")


class CppCompileError(RuntimeError):
    def __init__(self, cmd: list[str], output: str) -> None:
        if isinstance(output, bytes):
            output = output.decode("utf-8")

        self.cmd = cmd
        self.output = output

        super().__init__(
            textwrap.dedent(
                """
                    C++ compile error

                    Command:
                    {cmd}

                    Output:
                    {output}
                """
            )
            .strip()
            .format(cmd=" ".join(cmd), output=output)
        )

    def __reduce__(self) -> tuple[type, tuple[list[str], str]]:
        return (self.__class__, (self.cmd, self.output))


class CUDACompileError(CppCompileError):
    pass


class XPUCompileError(CppCompileError):
    pass


class TritonMissing(ShortenTraceback):
    def __init__(self, first_useful_frame: types.FrameType | None) -> None:
        super().__init__(
            "Cannot find a working triton installation. "
            "Either the package is not installed or it is too old. "
            "More information on installing Triton can be found at: https://github.com/triton-lang/triton",
            first_useful_frame=first_useful_frame,
        )


class GPUTooOldForTriton(ShortenTraceback):
    def __init__(
        self,
        # pyrefly: ignore [not-a-type]
        device_props: _CudaDeviceProperties,
        first_useful_frame: types.FrameType | None,
    ) -> None:
        super().__init__(
            f"Found {device_props.name} which is too old to be supported by the triton GPU compiler, "
            "which is used as the backend. Triton only supports devices of CUDA Capability >= 7.0, "
            f"but your device is of CUDA capability {device_props.major}.{device_props.minor}",
            first_useful_frame=first_useful_frame,
        )


class InductorError(BackendCompilerFailed):
    backend_name = "inductor"

    def __init__(
        self,
        inner_exception: Exception,
        first_useful_frame: types.FrameType | None,
    ) -> None:
        self.inner_exception = inner_exception
        ShortenTraceback.__init__(
            self,
            f"{type(inner_exception).__name__}: {inner_exception}",
            first_useful_frame=first_useful_frame,
        )
<|endoftext|>import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, TYPE_CHECKING, TypeAlias

import torch.fx.graph


if TYPE_CHECKING:
    from torch._functorch.partitioners import NodeInfo
    from torch._inductor.scheduler import BaseSchedulerNode


class CustomPassBase(ABC):
    """
    Implement this interface for custom passes:

    The uuid() method enables inductor to cache compiled graphs when your custom
    passes are applied. This method can return any identifier as long as it uniquely
    identifies your implementation (and can be pickled). The caching logic includes this
    identifier in its key calculation, i.e., any new value will effectively invalidate
    existing entries. We expect custom passes would typically depend purely on the
    textual representation of the implementation. In that case, we recommend using the
    'get_hash_for_files' helper below to compute a unique hash from the contents of a
    static list of source files, i.e., the source(s) containing the custom pass
    implementation. That approach ensures that any change to the implementation will
    mean a new uuid.

    ** IMPORTANT ** If your custom pass's behavior depends on some external state, then
    you'll need to implement something more complicated (or disable caching).

    EXAMPLE:

    class MyCustomGraphPass(CustomGraphPass):
        def __call__(self, graph: torch.fx.graph.Graph) -> None:
            # my custom graph optimization pass
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def uuid(self) -> Any | None:
        """
        Return an ID to uniquely identify your custom pass implementation. Return None
        to skip inductor code caching entirely.
        """


class CustomGraphPass(CustomPassBase):
    """
    Implement this interface for custom Graph passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.

    EXAMPLE:

    class MyCustomGraphPass(CustomGraphPass):
        def __call__(self, graph: torch.fx.graph.Graph) -> None:
            # my custom graph optimization pass
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph) -> None:
        """
        Implementation of the custom pass.
        """


class CustomGraphModulePass(CustomPassBase):
    """
    Implement this interface for custom passes that operate on GraphModule:

    1) The __call__() method contains the implementation of the custom pass.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.
    """

    @abstractmethod
    def __call__(self, gm: torch.fx.GraphModule) -> None:
        """
        Implementation of the custom pass.
        """


class CustomSchedulerPass(CustomPassBase):
    """
    Implement this interface for custom Scheduler passes:

    1) The __call__() method contains the implementation of the custom pass.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.
    """

    @abstractmethod
    def __call__(self, nodes: list["BaseSchedulerNode"]) -> list["BaseSchedulerNode"]:
        """
        Implementation of the custom pass.
        """


class CustomInferenceAwareGraphPass(CustomGraphPass):
    """
    Implement this interface for custom inference aware Graph passes.
    """

    @abstractmethod
    def __call__(self, graph: torch.fx.graph.Graph, is_inference: bool) -> None:
        """
        Implementation of the custom pass.
        """


CustomGraphPassCallable: TypeAlias = (
    CustomGraphPass | Callable[[torch.fx.graph.Graph], None]
)
CustomGraphPassType: TypeAlias = (
    CustomGraphPassCallable
    | list[CustomGraphPassCallable]
    | tuple[CustomGraphPassCallable, ...]
    | None
)

CustomSchedulerPassCallable: TypeAlias = (
    CustomSchedulerPass
    | Callable[[list["BaseSchedulerNode"]], list["BaseSchedulerNode"]]
)


def get_custom_graph_passes(
    custom_pass: CustomGraphPassType,
) -> tuple[CustomGraphPassCallable, ...]:
    if custom_pass is None:
        return ()
    if isinstance(custom_pass, (list, tuple)):
        return tuple(custom_pass)
    return (custom_pass,)


@lru_cache(1)
def get_hash_for_files(paths: tuple[str, ...], extra: str = "") -> bytes:
    """
    Helper to compute a unique string by hashing the contents of a list of files.
    """
    hasher = hashlib.sha256()
    hasher.update(extra.encode("utf-8"))
    for path in paths:
        with open(path, "rb") as f:
            hasher.update(f.read())
    return hasher.digest()


class CustomPartitionerFn(CustomPassBase):
    """
    Implement this interface for custom partitioner:

    1) The __call__() method contains the implementation of the custom partitioner.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.

    EXAMPLE:

    from torch._inductor.custom_graph_pass import get_hash_for_files

    class MyCustomPartitionerFn(CustomPartitionerFn):
        def __call__(
            self,
            gm: torch.fx.GraphModule,
            joint_inputs: Sequence[object],
            **kwargs: Any
        ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
            # my custom partitioner implementation
            #     ...

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))

    """

    @abstractmethod
    def __call__(
        self, gm: torch.fx.GraphModule, joint_inputs: Sequence[object], **kwargs: Any
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        """
        Implementation of the custom partitioner.
        """

    @abstractmethod
    def uuid(self) -> Any | None:
        """
        Return an ID to uniquely identify your custom partitioner implementation.
        Return None to skip inductor code caching entirely.
        """


CustomPartitionerFnType: TypeAlias = CustomPartitionerFn | None


class CustomRuntimeEstimator(CustomPassBase):
    """
    Implement this interface for custom runtime estimators:

    1) The __call__() method contains the implementation of the runtime estimation.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.

    EXAMPLE:

    class MyCustomRuntimeEstimator(CustomRuntimeEstimator):
        def __call__(self, node: fx.Node) -> float:
            # my custom runtime estimation logic
            return estimated_runtime

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))
    """

    @abstractmethod
    def __call__(self, node: "torch.fx.Node") -> float:
        """
        Implementation of the custom runtime estimator.

        Args:
            node: An fx.Node object whose runtime is to be estimated.

        Returns:
            float: The estimated runtime for the node.
        """


class CustomKnapsackSolver(CustomPassBase):
    """
    Implement this interface for custom knapsack solvers:

    1) The __call__() method contains the implementation of the knapsack solver.

    2) See CustomPassBase.uuid() docstring for implementing the uuid() method.

    EXAMPLE:

    class MyCustomKnapsackSolver(CustomKnapsackSolver):
        def __call__(
            self,
            memory: list[float],
            joint_graph: fx.Graph,
            max_memory: float,
            node_info: NodeInfo,
            all_recomputable_banned_nodes: list[fx.Node],
        ) -> tuple[list[int], list[int]]:
            # my custom knapsack solver logic
            return saved_node_idx, recomp_node_idx

        def uuid(self) -> Optional[Any]:
            return get_hash_for_files((__file__,))
    """

    @abstractmethod
    def __call__(
        self,
        memory: list[float],
        joint_graph: "torch.fx.Graph",
        max_memory: float,
        node_info: "NodeInfo",
        all_recomputable_banned_nodes: list["torch.fx.Node"],
    ) -> tuple[list[int], list[int]]:
        """
        Implementation of the custom knapsack solver.
        """
<|endoftext|># mypy: allow-untyped-defs
# pyre-strict
from __future__ import annotations

import heapq
import importlib
import itertools
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
from torch._logging import trace_structured
from torch.utils._ordered_set import OrderedSet

from . import config, config_comms, ir
from .dependencies import WeakDep


if TYPE_CHECKING:
    from .ir import IRNode, Operation

from .memory import (
    estimate_peak_memory_allocfree,
    FreeableInputBuffer,
    get_freeable_input_buf,
    SNodeMemory,
)
from .utils import contains_collective, contains_wait, is_fallback_op, is_wait
from .virtualized import V


log = logging.getLogger(__name__)
overlap_log = torch._logging.getArtifactLogger(__name__, "overlap")

if TYPE_CHECKING:
    from torch._inductor.scheduler import BaseSchedulerNode


def align_runtime_estimations_across_all_distributed_ranks(
    snodes: list[BaseSchedulerNode],
):
    runtime_estimations = [snode.get_estimated_runtime() for snode in snodes]

    import torch.distributed as dist
    from torch.distributed.distributed_c10d import _get_default_group

    world_size = dist.get_world_size()
    pg = _get_default_group()
    gathered_runtime_estimations: list[list[float]] = [[] for _ in range(world_size)]
    dist.all_gather_object(
        gathered_runtime_estimations,
        runtime_estimations,
        pg,
    )

    lengths = OrderedSet([len(e) for e in gathered_runtime_estimations])
    if len(lengths) != 1:
        log.warning(
            "Different ranks have different numbers of scheduler nodes (%s), "
            "skipping runtime estimation alignment",
            [len(e) for e in gathered_runtime_estimations],
        )
        for idx, snode in enumerate(snodes):
            snode.override_estimated_runtime = runtime_estimations[idx]
        return

    median_runtime_estimations = torch.median(
        torch.tensor(gathered_runtime_estimations), dim=0
    ).values.tolist()

    for idx, snode in enumerate(snodes):
        snode.override_estimated_runtime = median_runtime_estimations[idx]


def sink_waits(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Greedily schedules waits as late as possible.
    """
    return _schedule_for_comm(
        snodes, raise_comms=False, sink_waits=True, reorder_for_overlap=False
    )


def raise_comms(snodes: list[BaseSchedulerNode]) -> list[BaseSchedulerNode]:
    """
    Greedily schedules comms as early as possible.
    """
    return _schedule_for_comm(
        snodes, raise_comms=True, sink_waits=False, reorder_for_overlap=False
    )


def reorder_compute_for_overlap(
    snodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """
    This achieves the following overall scheduling procedure:
        Step 1: Given that we've currently scheduled comm N, we now schedule all compute nodes
            that are required for comm N + 1 but do not depend on comm N, to run at the same time with comm N.
        Step 2: If all those compute nodes are sufficient to overlap comm N, we're done.
            Otherwise, we now need to look elsewhere to find compute that overlaps with comm N.
            We prioritize compute nodes that are needed sooner.
        Step 3: We schedule the compute nodes dependent on comm N and required for comm N + 1.
        Step 4: We schedule comm N + 1.
        Repeat this for subsequent comm nodes.
    """
    return _schedule_for_comm(
        snodes, raise_comms=True, sink_waits=True, reorder_for_overlap=True
    )


def reorder_communication_preserving_peak_memory(
    snodes: list[BaseSchedulerNode],
) -> list[BaseSchedulerNode]:
    """
    Reorders communication ops relative to computation ops to improve communication-compute overlapping and hide comm
    latency.  Stops moving a particular op if it reaches a point that would have increased the peak memory footprint.

    Currently, follows these heuristics (subject to change or tune):
    - never reorders collectives relative to one another, for SPMD safety
    - has an option for per-collective prefetch limit, but does not enable it by default
    - limits the total number of reorder steps to some factor of the graph size to prevent worst-case quadratic
      performance

    Prerequisite: sink_comms_and_waits - ensure comm and wait nodes are scheduled as late as possible, respecting data
    dependencies.  That allows reorder_communication_preserving_peak_memory to take a best case peak-memory snapshot,
    and then monotonically improve latency by moving collectives backward in time.

    Peak memory impact is computed in an iterative fashion.  First, memory use at each timestep is computed, and global
    peak memory is computed as a max over timesteps.  Then, when swapping any two adjacent nodes, only the curr-memory
    for the earlier of the nodes after the swap is affected.  This enables checking step by step whether a swap is
    peak-memory-safe, and bailing out if not.  Example:

    0   n0      C0
    1   n1      C0 + Allocs(n1) - Frees(n1)
    2   n2      C0 + Allocs(n1) - Frees(n1) + Allocs(n2) - Frees(n2)

    0   n0      C0
    1   n2      C0 + Allocs(n2) - Frees(n2)    <-- After moving n2 to Time 1, only time1 memory changes
    2   n1      C0 + Allocs(n2) - Frees(n2) + Allocs(n1) - Frees(n1)

    """
    reordered_snodes, node_stats = (
        _reorder_communication_preserving_peak_memory_internal(snodes)
    )

    return reordered_snodes


@dataclass
class ReorderInfo:
    """
    Debug info describing how an individual snode was reordered
    """

    limiting_factor: str = "None"
    moves: int = 0
    grouped: int = 0
    grouped_info: str = ""
    comm_time: float = -1.0
    comp_time: float = -1.0
    initial_exposed: float = -1.0
    final_exposed: float = -1.0
    overlap_info: str = "None"

    @property
    def improvement(self):
        return self.initial_exposed - self.final_exposed


def is_gemm_like(node: IRNode | Operation | None) -> bool:
    if node is None:
        return False

    if is_fallback_op(
        node,  # type: ignore[arg-type]
        torch.ops.aten._scaled_dot_product_flash_attention.default,
    ):
        return True

    if (
        python_kernel_name := getattr(node, "python_kernel_name", None)
    ) and "extern_kernels" in python_kernel_name:
        return True
    return False


def contains_gemm_like(snode: BaseSchedulerNode) -> bool:
    from torch._inductor.scheduler import GroupedSchedulerNode

    if isinstance(snode, GroupedSchedulerNode):
        return any(contains_gemm_like(x) for x in snode.snodes)
    else:
        return is_gemm_like(snode.node)


def _temp_group_visit_leaves(snode: BaseSchedulerNode, fn):
    from torch._inductor.scheduler import GroupedSchedulerNode

    if isinstance(snode, GroupedSchedulerNode) and snode.temp_grouping:
        for _snode in snode.snodes:
            fn(_snode)
    else:
        fn(snode)


def _group_name(snode, with_bufs=False) -> str:
    ret = ""
    for n in snode.snodes:
        if ret:
            ret += "_"
        ret += n.get_name()
        if with_bufs:
            ret += f"{list(snode.get_buffer_names())}"
    return ret


def _is_fake_dep(d):
    return isinstance(d, WeakDep) and d.is_fake


def _group_names(gns: list[BaseSchedulerNode]) -> str:
    return "~".join([gn.get_name() for gn in gns])


def _initialize_memory_tracking(snodes, graph_inputs=None, graph_outputs=None):
    """Initialize memory tracking data structures"""
    if graph_inputs is None:
        graph_inputs = OrderedSet(V.graph.graph_inputs.keys())
    if graph_outputs is None:
        graph_outputs = OrderedSet(V.graph.get_output_names())

    name_to_freeable_input_buf = get_freeable_input_buf(snodes, graph_inputs)
    peak_memory, snodes_curr_memory, snodes_allocfree, buf_to_snode_last_use = (
        estimate_peak_memory_allocfree(
            snodes, name_to_freeable_input_buf, graph_outputs
        )
    )
    _curr_memory = dict(zip(snodes, snodes_curr_memory))
    # pyrefly: ignore [unsupported-operation]
    _curr_memory[None] = (0, 0)

    # Build candidate buffer map for optimization
    candidate_buffer_map = _build_candidate_buffer_map(buf_to_snode_last_use)

    return (
        peak_memory,
        _curr_memory,
        snodes_allocfree,
        buf_to_snode_last_use,
        name_to_freeable_input_buf,
        candidate_buffer_map,
    )


def _initialize_double_linked_list(
    snodes: list[BaseSchedulerNode],
) -> tuple[
    dict[BaseSchedulerNode, BaseSchedulerNode | None],
    dict[BaseSchedulerNode, BaseSchedulerNode | None],
    BaseSchedulerNode,
]:
    """Create double-linked list structure from snodes"""
    _prev = {}
    _next = {}
    for i, snode in enumerate(snodes):
        _prev[snode] = snodes[i - 1] if i > 0 else None
        _next[snode] = snodes[i + 1] if i < len(snodes) - 1 else None
    _head = snodes[0]
    return _prev, _next, _head


def _build_candidate_buffer_map(
    buf_to_snode_last_use: dict,
) -> dict[BaseSchedulerNode, OrderedSet]:
    """
    Build inverted index: node -> set of buffers where node appears in successors.

    This optimization reduces buffer iteration from O(total_buffers) to O(buffers_per_node).
    Since buffer successors are immutable during reordering, this map doesn't need updates.

    Returns:
        dict mapping each node to the set of buffers that have this node in their successors
    """
    node_to_candidate_bufs: dict[BaseSchedulerNode, OrderedSet] = defaultdict(
        OrderedSet
    )

    for buf in buf_to_snode_last_use:
        # Add to every successor node's buffer set
        for succ_node in buf.mpi_buffer.succ_nodes:
            node_to_candidate_bufs[succ_node].add(buf)

    return dict(node_to_candidate_bufs)


def _precompute_node_output_sets(
    snodes: list[BaseSchedulerNode],
) -> dict[BaseSchedulerNode, OrderedSet[str]]:
    """
    Pre-compute output name sets for all nodes.

    This optimization avoids creating OrderedSet objects repeatedly during
    exposed time calculations.

    Returns:
        dict mapping each node to a set of its output names
    """
    return {
        snode: OrderedSet(o.get_name() for o in snode.get_outputs()) for snode in snodes
    }


def _op_runtime_estimate_mult(snode):
    # Apply multipliers for faster experimentation.
    # TODO(ivankobzarev): Remove after confirmation that runtime estimations are correct.
    if contains_collective(snode):
        return config_comms.reorder_sink_runtime_estimations_comm_mult

    return config_comms.reorder_sink_runtime_estimations_non_comm_mult


def is_async_collective(snode):
    """
    Filtering out ops that contain Collective and Wait inside and considered as Collectives.
    See contains_collective function.
    If the op contains Wait inside - consider as Synchronous compute.
    """
    if python_kernel_name := getattr(snode.node, "python_kernel_name", None):
        if "torch.ops._dtensor.shard_dim_alltoall.default" in python_kernel_name:
            return False

    return True


def contains_async_collective(snode):
    return contains_collective(snode, is_async_collective)


def _group_nodes_from_linked_list(
    head: BaseSchedulerNode | None,
    tail: BaseSchedulerNode | None,
    next_dict: dict[BaseSchedulerNode, BaseSchedulerNode | None],
) -> list[BaseSchedulerNode]:
    """
    Traverse doubly-linked list from head to tail and return nodes as a list.

    Args:
        head: Starting node of the segment
        tail: Ending node of the segment (inclusive)
        next_dict: Dictionary mapping each node to its next node

    Returns:
        List of nodes from head to tail (inclusive)
    """
