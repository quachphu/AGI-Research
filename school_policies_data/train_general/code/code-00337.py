                raise RuntimeError(f"Invalid impl group index: {idx}")

            group = impl_groups[idx]
            merged_kwargs = _merge_config_and_runtime_kwargs(
                group.impl_kwargs, runtime_kwargs
            )

            @torch._dynamo.dont_skip_tracing
            def group_fn(*ops):
                return group.impl_func(*ops, **merged_kwargs)

            if idx == num_impl_groups - 1:
                return group_fn

            next_fn = build_nested_cond(idx + 1)

            @torch._dynamo.dont_skip_tracing
            def cond_wrapper(*ops, _ranges=group.ranges):
                return torch.cond(
                    pred=build_range_predicate(_ranges),
                    true_fn=group_fn,
                    false_fn=next_fn,
                    operands=ops,
                )

            return cond_wrapper

        return build_nested_cond(0)(*fake_tensors)

    with V.fake_mode:
        fake_inputs = tuple(ir_node_to_tensor(inp) for inp in tensor_inputs)
        decomposition_table = select_decomp_table()
        shape_env = V.fake_mode.shape_env

        log.info("Tracing torch.cond dispatch with symbolic shapes...")

        try:
            context = (
                shape_env.error_on_new_guards
                if shape_env is not None
                else contextlib.nullcontext
            )
            with context():
                dispatch_gm = make_fx(
                    dispatch_fn,
                    decomposition_table=decomposition_table,
                    tracing_mode="symbolic",
                )(*fake_inputs)

            log.info("Successfully traced torch.cond dispatch")
            log.info("Traced graph:\n%s", dispatch_gm.graph)

        except (_ShapeEnvGuardError, AssertionError) as e:
            is_guard_error = isinstance(e, _ShapeEnvGuardError) or (
                isinstance(e, AssertionError)
                and "Guard attempted while ShapeEnv guards are frozen" in str(e)
            )
            if not is_guard_error:
                raise
            log.info("Dispatch function adds guards, skipping custom op lowering")
            counters["inductor"]["custom_op_decomp_guard_skips"] += 1
            return None

        except Exception:
            log.exception("make_fx tracing FAILED")
            raise

    ops_before = len(V.graph.operations)
    result = inline_subgraph_to_ir_nodes(dispatch_gm, tensor_inputs, f"{name}_dispatch")

    # Apply config_patches from all impl groups to inlined operations
    # TODO - consider conflicting patches
    merged_patches: dict[str, Any] = {}
    for group in impl_groups:
        merged_patches.update(group.config_patches)
    if merged_patches:
        _apply_config_patches_recursive(V.graph.operations[ops_before:], merged_patches)

    log.info(
        "Successfully created torch.cond dispatch for %d impl groups", len(impl_groups)
    )

    validate_ir(result)
    return result


def _create_autotuning_lowering(
    processed_configs: list[CustomOpConfig],
    default_impl: Callable[..., Any],
    name: str,
    op_overload: torch._ops.OpOverload,
    input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None,
    range_upper_bound: int,
    is_range_based: bool = False,
    config_generator: Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    | None = None,
    dispatch_on: tuple[str, int] | None = None,
    split_points: list[int] | None = None,
    min_speedup_threshold: float = 1.0,
    benchmark_with_cudagraphs: bool = False,
) -> Callable[..., Any]:
    """Create the lowering function for autotuning."""
    if not is_range_based:
        # Standard autotuning path
        @functools.wraps(op_overload)
        def standard_lowering_wrapper(*args: Any, **kwargs: Any) -> Any:
            tensor_inputs, runtime_kwargs = _extract_tensor_inputs(
                args, kwargs, op_overload
            )
            return _standard_lowering_fn(
                processed_configs=processed_configs,
                default_impl=default_impl,
                name=name,
                op_overload=op_overload,
                input_gen_fns=input_gen_fns,
                tensor_inputs=tensor_inputs,
                runtime_kwargs=runtime_kwargs,
                config_generator=config_generator,
                min_speedup_threshold=min_speedup_threshold,
                benchmark_with_cudagraphs=benchmark_with_cudagraphs,
            )

        return standard_lowering_wrapper

    # Range-based autotuning path
    # pyrefly: ignore [not-iterable]
    tensor_name, dim_index = dispatch_on
    # pyrefly: ignore [bad-argument-type]
    ranges = _create_ranges_from_split_points(split_points)

    @functools.wraps(op_overload)
    def range_based_lowering_wrapper(*args: Any, **kwargs: Any) -> Any:
        tensor_inputs, runtime_kwargs = _extract_tensor_inputs(
            args, kwargs, op_overload
        )
        return _range_based_lowering_fn(
            processed_configs=processed_configs,
            default_impl=default_impl,
            name=name,
            op_overload=op_overload,
            input_gen_fns=input_gen_fns,
            tensor_name=tensor_name,
            dim_index=dim_index,
            ranges=ranges,
            tensor_inputs=tensor_inputs,
            runtime_kwargs=runtime_kwargs,
            range_upper_bound=range_upper_bound,
            config_generator=config_generator,
            min_speedup_threshold=min_speedup_threshold,
            benchmark_with_cudagraphs=benchmark_with_cudagraphs,
        )

    return range_based_lowering_wrapper


def register_custom_op_autotuning(
    custom_op: torch._library.custom_ops.CustomOpDef | torch._ops.OpOverload,
    configs: list[CustomOpConfig] | list[Callable[..., Any]] | None = None,
    config_generator: Callable[[dict[str, torch.Tensor]], list[CustomOpConfig]]
    | None = None,
    name: str | None = None,
    input_gen_fns: dict[str, Callable[[torch.Tensor], torch.Tensor]] | None = None,
    dispatch_on: DispatchOnConfig | None = None,
    split_points: list[int] | None = None,
    min_speedup_threshold: float = 1.0,
    benchmark_with_cudagraphs: bool = False,
) -> None:
    """Register custom op for autotuning with custom_op configs where each config
    specifies a decomposition implementation function with its parameter values.
    It also supports Range-based autotuning to benchmark per range and generate
    runtime dispatch.

    Args:
        custom_op: Custom operation (CustomOpDef from @torch.library.custom_op) or
                   OpOverload (e.g., torch.ops.aten.mm.default)
        configs: List of CustomOpConfig objects for static inputs. Mutually exclusive with config_generator.
        config_generator: Dynamic config generator function that takes a dict mapping
                          parameter names to fake tensors, and returns list[CustomOpConfig]
                          based on input tensor properties. Mutually exclusive with configs.
        name: Operation name (default: "{op_name}_autotuned")
        input_gen_fns: Custom input generators for benchmarking
        dispatch_on: Dict for range-based dispatch with keys:
            - 'tensor_name': Name of tensor parameter to dispatch on
            - 'dim': Dimension index to check size
            - 'unbounded_size' (optional): Benchmark size for the unbounded (last) range, such
                as [2048, inf] -> [2048, unbounded_size]. Set based on your expected workload size.
                Default is DEFAULT_RANGE_UPPER_BOUND=65536.
        split_points: List of range endpoints in ascending order for range-based autotuning
        min_speedup_threshold: Only pick a non-fallback choice if it beats the fallback
            by at least this ratio. Default is 1.0 (any speedup wins). Set to e.g. 1.1
            to require 10% speedup over fallback.
        benchmark_with_cudagraphs: If True, benchmark the fallback kernel using CUDA graph
            capture and replay for fair comparison with compiled kernels. Default is False.

    The default/fallback implementation is automatically derived:
    - For CustomOpDef: Uses the decorated function
    - For OpOverload: Traces the op call, which falls through to normal inductor lowering

    Examples:
        # Static configs
        @torch.library.custom_op("mylib::attention", mutates_args=())
        def my_attention(query, key, value, head_dim=32):
            ...

        register_custom_op_autotuning(
            my_attention,
            configs=[
                CustomOpConfig(attention_impl, head_dim=32, method='chunked'),
                CustomOpConfig(attention_impl, head_dim=64, method='tiled'),
                CustomOpConfig(head_dim=128),  # No decomposition specified, use default
            ],
            input_gen_fns={
                "query": lambda fake: torch.randn_like(fake, device='cuda'),
                "key": lambda fake: torch.randn_like(fake, device='cuda'),
                "value": lambda fake: torch.randn_like(fake, device='cuda'),
            },
        )

        # Dynamic config generation based on input tensor properties
        def generate_k_split_configs(fake_tensors: dict[str, torch.Tensor]) -> list[CustomOpConfig]:
            # Access tensor shapes, dtypes, devices, etc.
            m, k = fake_tensors["mat1"].shape
            _, n = fake_tensors["mat2"].shape
            k_splits = ... # compute possible k splits based on tensor properties
            return [CustomOpConfig(k_splits=k) for k in k_splits]

        register_custom_op_autotuning(
            matmul_decomposeK_op,
            config_generator=generate_k_split_configs,
            input_gen_fns={...},
        )

    Range-based Example:
        register_custom_op_autotuning(
            my_op,
            configs=[CustomOpConfig(impl1), CustomOpConfig(impl2), CustomOpConfig(impl3)],
            dispatch_on={
                # Dispatch based on x.shape[1]
                "tensor_name": "x",
                "dim": 1,
                # Optional Benchmark size used for the unbounded range [2049, inf].
                # Since inf is not a concrete value, we use range_upper_bound as the benchmark size.
                # Default value is 65536 (DEFAULT_RANGE_UPPER_BOUND) if not provided.
                "range_upper_bound": 8192,
            },
            split_points=[512, 2048],  # Creates ranges: [1,512], [513,2048], [2049, 8192]
        )
    """
    from torch._library.custom_ops import CustomOpDef

    # Handle both CustomOpDef and OpOverload - derive impl_fn automatically
    # Both cases call through op_overload so fake handlers are used during tracing
    if isinstance(custom_op, CustomOpDef):
        op_overload = custom_op._opoverload
    elif isinstance(custom_op, torch._ops.OpOverload):
        op_overload = custom_op
    else:
        raise TypeError(
            f"custom_op must be a CustomOpDef or OpOverload, got {type(custom_op)}."
        )

    # impl_fn calls through op_overload so fake handlers are used during tracing
    def impl_fn(*args, **kwargs):
        return op_overload(*args, **kwargs)

    # Validate configs and config_generator are mutually exclusive
    if configs is not None and config_generator is not None:
        raise ValueError(
            "Cannot specify both 'configs' and 'config_generator'. "
            "Use 'config_generator' for shape-dependent configs."
        )

    if configs is None and config_generator is None:
        raise ValueError("Must specify either 'configs' or 'config_generator'")

    # Process and validate static configs at registration time
    static_configs = None
    if configs is not None:
        if not isinstance(configs, (list, tuple)):
            raise TypeError(f"configs must be a list or tuple, got {type(configs)}")

        static_configs = []
        for cfg in configs:
            if isinstance(cfg, CustomOpConfig):
                static_configs.append(cfg)
            else:
                raise TypeError(
                    f"Each config must be a CustomOpConfig object, got {type(cfg)}"
                )

        if not static_configs:
            raise ValueError("At least one config must be provided")

    if name is None:
        name = f"{op_overload._name}_autotuned"

    # Validate range-based parameters
    is_range_based = dispatch_on is not None or split_points is not None
    dispatch_on_tuple: tuple[str, int] | None = None
    range_upper_bound = DEFAULT_RANGE_UPPER_BOUND
    if is_range_based:
        if dispatch_on is None or split_points is None:
            raise ValueError(
                "Both dispatch_on and split_points must be specified for range-based autotuning"
            )
        if not isinstance(dispatch_on, dict):
            raise ValueError(
                "dispatch_on must be a dict with 'tensor_name' and 'dim' keys, "
                f"e.g., {{'tensor_name': 'x', 'dim': 1}}. Got: {type(dispatch_on)}"
            )
        if "tensor_name" not in dispatch_on or "dim" not in dispatch_on:
            raise ValueError(
                "dispatch_on must contain 'tensor_name' and 'dim' keys, "
                f"e.g., {{'tensor_name': 'x', 'dim': 1}}. Got keys: {list(dispatch_on.keys())}"
            )
        if not isinstance(dispatch_on["tensor_name"], str):
            raise ValueError(
                f"dispatch_on['tensor_name'] must be a string (tensor parameter name), "
                f"got {type(dispatch_on['tensor_name'])}"
            )
        if not isinstance(dispatch_on["dim"], int):
            raise ValueError(
                f"dispatch_on['dim'] must be an integer (dimension index), "
                f"got {type(dispatch_on['dim'])}"
            )
        dispatch_on_tuple = (dispatch_on["tensor_name"], dispatch_on["dim"])
        range_upper_bound = dispatch_on.get(
            "range_upper_bound", DEFAULT_RANGE_UPPER_BOUND
        )
        if not isinstance(range_upper_bound, int) or range_upper_bound <= 0:
            raise ValueError(
                f"dispatch_on['range_upper_bound'] must be a positive integer, "
                f"got {range_upper_bound}"
            )
        if not isinstance(split_points, list) or len(split_points) == 0:
            raise ValueError("split_points must be a non-empty list of integers")
        if sorted(split_points) != split_points:
            raise ValueError("split_points must be sorted in ascending order")

    # Create and register the lowering function
    lowering_fn = _create_autotuning_lowering(
        # pyrefly: ignore [bad-argument-type]
        processed_configs=static_configs,
        default_impl=impl_fn,
        name=name,
        op_overload=op_overload,
        input_gen_fns=input_gen_fns,
        is_range_based=is_range_based,
        config_generator=config_generator,
        dispatch_on=dispatch_on_tuple,
        split_points=split_points,
        range_upper_bound=range_upper_bound,
        min_speedup_threshold=min_speedup_threshold,
        benchmark_with_cudagraphs=benchmark_with_cudagraphs,
    )

    # Register in user_lowerings which takes priority over built-in lowerings
    # The dispatch in graph.py checks user_lowerings first with recursion guard
    user_lowerings[op_overload] = lowering_fn
<|endoftext|># mypy: allow-untyped-defs
import logging
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch._inductor.select_algorithm import realize_inputs, SymbolicGridFn
from torch._inductor.utils import get_current_backend, sympy_product
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

from .. import config
from ..codegen.wrapper import PythonWrapperCodegen
from ..ir import _IntLike, Layout, TensorBox
from ..utils import load_template, triton_type


log = logging.getLogger(__name__)


@SymbolicGridFn
def mm_grid(m, n, meta, *, cdiv):
    """
    The CUDA grid size for matmul triton templates.
    """
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), 1, 1)


@SymbolicGridFn
def persistent_mm_grid(M: int, N: int, meta: dict[str, Any], *, cdiv, min):
    """Defines the grid for persistent kernels."""
    return (
        min(meta["NUM_SMS"], cdiv(M, meta["BLOCK_M"]) * cdiv(N, meta["BLOCK_N"])),
        1,
        1,
    )


@SymbolicGridFn
def persistent_grouped_mm_grid(*args):
    meta = args[-1]
    return (meta["NUM_SMS"], 1, 1)


def acc_type(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return "tl.float32"
    if dtype.is_floating_point and dtype.itemsize == 1:  # fp8 dtypes
        return "tl.float32"
    return triton_type(dtype)


def mm_args(
    mat1,
    mat2,
    *others,
    layout=None,
    out_dtype=None,
    use_4x2_dim=False,
    mat2_transposed=False,
):
    """
    Common arg processing for mm,bmm,addmm,etc
    """
    mat1, mat2 = realize_inputs(mat1, mat2)
    *b1, m, k1 = mat1.get_size()
    if mat2_transposed:
        *b2, n, k2 = mat2.get_size()
    else:
        *b2, k2, n = mat2.get_size()
    b = [V.graph.sizevars.check_equals_and_simplify(a, b) for a, b in zip(b1, b2)]
    if use_4x2_dim:
        k2 = k2 * 2
    k = V.graph.sizevars.check_equals_and_simplify(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()

        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            [*b, m, n],
        )
    else:
        if out_dtype is not None:
            raise AssertionError("out_dtype is ignored if layout is specified.")
    from ..lowering import expand

    others = [realize_inputs(expand(x, layout.size)) for x in others]

    return [m, n, k, layout, mat1, mat2, *others]


def addmm_epilogue(dtype, alpha, beta):
    def epilogue(acc, bias):
        if alpha != 1:
            acc = V.ops.mul(acc, V.ops.constant(alpha, dtype))
        if beta != 1:
            bias = V.ops.mul(bias, V.ops.constant(beta, dtype))
        return V.ops.add(acc, bias)

    return epilogue


def scale_mm_epilogue():
    """
    Create an epilogue function that applies scaling to matrix multiplication result
    using the given scale factors.

    Args:
        dtype: The data type of the output
        scale_a: Scale factor for matrix A
        scale_b: Scale factor for matrix B

    Returns:
        Epilogue function that takes the accumulator and applies scaling
    """

    def epilogue(acc, inv_a_scale, inv_b_scale, bias=None):
        # The epilogue function receives the accumulator (result of mat1 @ mat2)
        # and applies the scaling factors
        # In the original scaled_mm, we use inverse scales, so we multiply by them
        mul_scales = V.ops.mul(inv_a_scale, inv_b_scale)
        mul_acc = V.ops.mul(acc, mul_scales)
        if bias is not None:
            return V.ops.add(mul_acc, bias)
        else:
            return mul_acc

    return epilogue


def use_native_matmul(mat1, mat2):
    if not config.triton.native_matmul:
        return False

    # If tma matmul is on, don't do native matmul
    if (
        config.triton.enable_persistent_tma_matmul
        and torch.utils._triton.has_triton_tma_device()
    ):
        raise AssertionError("native matmul doesn't support tma codegen yet")

    # Currently only enable native matmul for default indexing
    # TODO : support block ptr
    if config.triton.use_block_ptr:
        raise AssertionError("native matmul doesn't support block_ptr codegen yet")

    # Currently only enable native matmul for triton on GPU.
    device_type = mat1.get_device().type
    if not (
        device_type in ("cuda", "xpu") and get_current_backend(device_type) == "triton"
    ):
        return False

    # Currently, tl.dot only supports following dtypes
    triton_supported_dtype = [
        torch.int8,
        torch.uint8,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    if mat1.dtype not in triton_supported_dtype:
        return False
    if mat2.dtype not in triton_supported_dtype:
        return False

    # (..., M, K) @ (..., K, N)
    m, k, n = mat1.get_size()[-2], mat1.get_size()[-1], mat2.get_size()[-1]

    # If the shape has unbacked symbols, don't do native matmul.
    # This is related to the behavior of statically_known_multiple_of on unbacked symints.
    # Since statically_known_multiple_of just returns False for unbacked symbols
    # due to the expensive cost, codegen fails when there is a unbacked symbol.
    # In particular, it fails at _split_iteration_ranges in codegen/simd.py.
    # See this : https://github.com/pytorch/pytorch/pull/131649
    if any(map(has_free_unbacked_symbols, [m, k, n])):
        return False

    # Consider the shape (m,k,n) > 1
    # TODO : support when size = 1
    if (
        V.graph.sizevars.statically_known_leq(m, 1)
        or V.graph.sizevars.statically_known_leq(k, 1)
        or V.graph.sizevars.statically_known_leq(n, 1)
    ):
        return False

    return True


def _use_small_mm_pointwise(m, k, n, layout) -> bool:
    """Check if mm should be lowered to pointwise ops for small K and N.

    For very small inner dimensions (K < 5 and N < 5) with M >= 64,
    cuBLAS launch overhead dominates and a fused pointwise kernel is
    faster (1.2-4.8x on H200).  M >= 64 excludes tiny matrices where
    pointwise kernel launch overhead negates the benefit.  K,N < 5
    avoids shapes where Triton codegen quality is unstable (K=5 regresses
    at large M due to reduction-dimension alignment).  Disabled under
    max_autotune to preserve template selection.
    See https://github.com/pytorch/pytorch/issues/186348
    """
    if config.max_autotune or config.max_autotune_gemm:
        return False
    if layout.device.type in ("cpu", "mps"):
        return False
    skt = V.graph.sizevars.statically_known_true
    return skt(m >= 64) and skt(k < 5) and skt(n < 5)


def _is_static_problem(layout: Layout) -> tuple[bool, bool]:
    """
    Check if input tensors and output layout have static shapes and non-zero sizes.

    Args:
        layout: Output layout object with a 'size' attribute.

    Returns:
        Tuple[bool, bool]: (is_static, is_nonzero)
            is_static: True if all shapes are statically known
            is_nonzero: True if all dimensions are non-zero
    """
    static_shape = True
    static_size = PythonWrapperCodegen.statically_known_list_of_ints_or_none(
        layout.size
    )
    if static_size is None:
        nonzero = True
        for s in layout.size:
            sz = PythonWrapperCodegen.statically_known_int_or_none(s)
            if sz is not None and sz == 0:
                nonzero = False
                break
        return False, nonzero
    numel = 1
    for dim in static_size:
        numel *= dim
    nonzero = numel > 0
    return static_shape, nonzero


def check_supported_striding(mat_a: TensorBox, mat_b: TensorBox) -> None:
    def is_row_major(stride: Sequence[_IntLike]) -> bool:
        return stride[-1] == 1

    def is_col_major(stride: Sequence[_IntLike]) -> bool:
        return stride[-2] == 1

    def has_zero_dim(size: Sequence[_IntLike]) -> bool:
        return bool(size[0] == 0 or size[1] == 0)

    # Check mat_a (self) stride requirements
    torch._check(
        is_row_major(mat_a.get_stride()) or has_zero_dim(mat_a.get_size()),
        lambda: f"mat_a must be row_major, got stride {mat_a.get_stride()}",
    )

    # Check mat_b stride requirements
    torch._check(
        is_col_major(mat_b.get_stride()) or has_zero_dim(mat_b.get_size()),
        lambda: f"mat_b must be col_major, got stride {mat_b.get_stride()}",
    )


def is_batch_stride_largest_or_zero(mat1, mat2, layout) -> bool:
    """
    Checking if the batch stride is the largest in the stride.
    """
    sizes = [mat1.get_size(), mat2.get_size(), layout.size]
    strides = [mat1.get_stride(), mat2.get_stride(), layout.stride]
    for size, stride in zip(sizes, strides):
        if not (len(size) == len(stride) == 3):
            raise AssertionError("Expect 3D tensors")
        if stride[0] != 0 and stride[0] != sympy_product(size[1:]):
            return False

    return True


_KERNEL_TEMPLATE_DIR = Path(__file__).parent / "templates"
load_kernel_template = partial(load_template, template_dir=_KERNEL_TEMPLATE_DIR)
<|endoftext|># mypy: allow-untyped-defs
import logging
from typing import TYPE_CHECKING

import torch
from torch._dynamo.utils import counters
from torch._inductor.codegen.rocm.ck_universal_gemm_template import CKGemmTemplate
from torch._inductor.kernel.mm_common import load_kernel_template

from .. import config as inductor_config, ir, lowering as L
from ..kernel_inputs import MMKernelInputs
from ..lowering import lowerings, make_pointwise, make_reduction, transform_args
from ..runtime.runtime_utils import get_max_y_grid
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    SymbolicGridFn,
    TritonTemplate,
)
from ..utils import (
    _use_cutlass_for_op,
    use_aten_gemm_kernels,
    use_ck_gemm_template,
    use_cpp_bmm_template,
    use_cutlass_template,
    use_nv_universal_gemm_template,
    use_triton_template,
)
from ..virtualized import ops, V
from .mm_common import (
    _is_static_problem,
    is_batch_stride_largest_or_zero,
    mm_args,
    use_native_matmul,
)


if TYPE_CHECKING:
    from ..ir import ChoiceCaller
    from ..select_algorithm import KernelTemplate

log = logging.getLogger(__name__)
aten = torch.ops.aten


@SymbolicGridFn
def bmm_grid(b, m, n, meta, *, cdiv, max):
    tiles = cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"])
    # Split batch across grid_y and grid_z to avoid exceeding CUDA grid_y limit.
    # When b <= max_y_grid, grid_z = 1 and behavior is identical to the original.
    max_y_grid = get_max_y_grid()
    grid_z = max(cdiv(b, max_y_grid), 1)
    grid_y = cdiv(b, grid_z)
    return (tiles, grid_y, grid_z)


# We define each template kernel in a separate file which is the name of the input to load_kernel_template
# (e.g. triton_bmm for templates/triton_bmm.py.jinja).
# If you are adding a new template, please follow that pattern and add a new file with your implementation in the templates folder.
bmm_template = TritonTemplate(
    name="bmm",
    grid=bmm_grid,
    source=load_kernel_template("triton_bmm"),
    cache_codegen_enabled_for_template=True,
)

aten_bmm = ExternKernelChoice(torch.bmm, "at::bmm_out", op_overload=aten.bmm.out)
aten_bmm_dtype = ExternKernelChoice(
    torch.bmm,
    "at::_bmm_out_dtype_xpu" if torch.xpu._is_compiled() else "at::_bmm_out_dtype_cuda",
    name="bmm_dtype",
    op_overload=aten.bmm.dtype_out,
)
aten_baddbmm = ExternKernelChoice(
    torch.baddbmm, "at::baddbmm_out", op_overload=aten.baddbmm.out
)

# This path targets vmapped dot products and similar tiny vector contractions,
# where extern bmm launch overhead and lost fusion dominate. Keep the threshold
# conservative so reductions proven to be larger than the threshold continue
# through the normal bmm machinery.
_BMM_DOT_K_DECOMPOSE_THRESHOLD = 32
_BMM_DOT_DECOMPOSE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


@L.register_lowering(aten.bmm)
def tuned_bmm(mat1, mat2, out_dtype=None, *, layout=None):
    """
    Lowering for autotuning aten.bmm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    sizevars = V.graph.sizevars
    dtype = mat1.get_dtype()
    device_type = mat1.get_device().type

    def dim_is_one_or_hint(dim):
        # The mul+sum decomposition is valid for any M/N. The size-1 hint is
        # only a profitability signal, so avoid specializing dynamic dims here.
        return sizevars.optimization_hint(dim, fallback=2) == 1

    def dim_is_not_known_gt(dim, threshold):
        # Do not use optimization_hint() for K: the same symbolic FX graph can
        # be code-cache reused across different concrete K values.
        return not sizevars.statically_known_gt(dim, threshold)

    if (
        out_dtype is None
        and device_type in ("cuda", "xpu")
        and device_type == mat2.get_device().type
        and dtype == mat2.get_dtype()
        and dtype in _BMM_DOT_DECOMPOSE_DTYPES
        and dim_is_one_or_hint(mat1.get_size()[1])
        and dim_is_one_or_hint(mat2.get_size()[2])
        and dim_is_not_known_gt(mat1.get_size()[2], _BMM_DOT_K_DECOMPOSE_THRESHOLD)
    ):
        # Preserve dot-shaped bmm as pointwise/reduction IR so surrounding
        # operations can fuse instead of dispatching a tiny extern bmm.
        mat1 = L.unsqueeze(mat1, -1)
        mat2 = L.unsqueeze(mat2, 1)
        return L.sum_(L.mul(mat1, mat2), axis=2)

    if all(x.get_device().type == "cpu" for x in [mat1, mat2]):
        # decompose to small ops when memory bound
        if mat1.get_size()[1] == 1 or mat2.get_size()[2] == 1:
            mat1 = L.unsqueeze(mat1, -1)
            mat2 = L.unsqueeze(mat2, 1)
            return L.sum_(L.mul(mat1, mat2), axis=2)

        def is_valid_to_require_contiguous(t):
            if not ir.is_storage_and_layout(t):
                return True
            _, layout = ir.as_storage_and_layout(t, freeze=False)
            return isinstance(layout, ir.FlexibleLayout)

        def is_preferred_layout_as_bmm_input(sizes, strides):
            # contiguous on one of the last two dims
            return (
                strides[-1] == 1 and (sizes[-2] == 1 or strides[-2] >= sizes[-1])
            ) or (strides[-2] == 1 and (sizes[-1] == 1 or strides[-1] >= sizes[-2]))

        # Make the input of bmm contiguous
        # if it is not contiguous on either of the last two dims,
        # because bmm cpu implementation would do contiguous() if not.
        # This is to avoid additional copies in bmm.
        def may_require_contiguous(t, meta_t):
            sizes = meta_t.meta["val"].size()
            strides = meta_t.meta["val"].stride()
            if not is_preferred_layout_as_bmm_input(sizes, strides):
                t = ir.ExternKernel.require_contiguous(t)
            return t

        if is_valid_to_require_contiguous(mat1):
            meta_mat1 = V.graph.current_node.args[0]
            mat1 = may_require_contiguous(mat1, meta_mat1)
        if is_valid_to_require_contiguous(mat2):
            meta_mat2 = V.graph.current_node.args[1]
            mat2 = may_require_contiguous(mat2, meta_mat2)

    if use_native_matmul(mat1, mat2):
        mat1 = lowerings[aten.unsqueeze](mat1, -1)
        mat2 = lowerings[aten.unsqueeze](mat2, 1)
        args, kwargs = transform_args(
            args=[mat1, mat2],
            kwargs={},
            broadcast=True,
            type_promotion_kind=None,
            convert_input_to_bool=False,
        )  # Handles broadcasting the arguments

        if inductor_config.triton.codegen_upcast_to_fp32 and mat1.dtype in [
            torch.float16,
            torch.bfloat16,
        ]:

            def _to_dtype(x):
                return ops.to_dtype(x, mat1.dtype, use_compute_types=False)

            args = [make_pointwise(_to_dtype)(x) for x in args]

        mul_pointwise = make_pointwise(ops.dot)(*args)
        dot_reduction = make_reduction("dot")(mul_pointwise, 2)

        return dot_reduction

    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=out_dtype
    )
    name = "bmm"

    # Create MMKernelInputs for BMM at the top
    kernel_inputs = MMKernelInputs([mat1, mat2], out_dtype=out_dtype)

    # below is for getting an overview logging info of inductor mms
    batch_size = mat1.get_size()[0]  # Extract batch dimension
    counters["aten_mm_info"][f"aten.bmm_{batch_size}_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.bmm: batch=%s, m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        batch_size,
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    aten_handler: ExternKernelChoice = aten_bmm
    aten_extra_kwargs = {}
    if out_dtype:
        if mat1.get_device().type not in ("cuda", "xpu"):
            raise AssertionError("out_dtype is only supported for CUDA or XPU")
        aten_handler = aten_bmm_dtype
        aten_extra_kwargs = {"out_dtype": out_dtype}

    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    kwarg_overrides = {}

    if use_aten_gemm_kernels():
        templates_to_use.append(aten_handler)
        kwarg_overrides[aten_handler.uid] = aten_extra_kwargs

    if use_triton_template(layout, check_max_autotune=False):
        templates_to_use.append(bmm_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            templates_to_use,
            name,
            kwarg_overrides=kwarg_overrides,
        )
    )
    _, is_nonzero = _is_static_problem(layout)
    batch_stride_largest_or_zero = is_batch_stride_largest_or_zero(mat1, mat2, layout)
    if (
        batch_stride_largest_or_zero
        and is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        from ..codegen.cutlass.gemm_template import CUTLASS3xGemmTemplate

        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, kernel_inputs.nodes()
        )  # type: ignore[arg-type]

    if use_cpp_bmm_template(layout, mat1, mat2):
        from ..codegen.cpp_bmm_template import CppBmmTemplate

        CppBmmTemplate.add_choices(
            choices,
            layout,
            kernel_inputs.nodes(),
        )

    if use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())

    if is_nonzero and use_nv_universal_gemm_template(layout, m, n, k, mat1, mat2):
        from ..codegen.nv_universal_gemm import add_nv_universal_gemm_choices

        add_nv_universal_gemm_choices(choices, layout, kernel_inputs)

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


@L.register_lowering(aten.baddbmm)
def tuned_baddbmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    """
    Lowering for autotuning aten.mm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    if use_native_matmul(mat1, mat2):
        if beta == 0:
            arg1 = 0
        else:
            arg1 = lowerings[aten.mul](beta, inp)

        if alpha == 0:
            arg2 = 0
        else:
            arg2 = lowerings[aten.mul](alpha, lowerings[aten.bmm](mat1, mat2))

        return lowerings[aten.add](arg1, arg2)

    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2, inp = mm_args(mat1, mat2, inp, layout=layout)

    # Create MMKernelInputs for BadDBMM at the top
    kernel_inputs = MMKernelInputs(
        [inp, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )

    # below is for getting an overview logging info of inductor mms
    batch_size = mat1.get_size()[0]
    counters["aten_mm_info"][f"aten.baddbmm_{batch_size}_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.baddbmm: batch_size=%s, m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, inp=%s, output_layout=%s",
        batch_size,
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        inp.get_dtype(),
        layout,
    )
    name = "baddbmm"
    # options to tune from
    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    if use_aten_gemm_kernels():
        templates_to_use.append(aten_baddbmm)

    if use_triton_template(layout, check_max_autotune=False):
        templates_to_use.append(bmm_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, name)
    )

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node
<|endoftext|># mypy: allow-untyped-defs
"""Shared CuTeDSL emission primitives for GEMM epilogues."""

from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import (
    CuteDSLCSEVariable,
    CuteDSLOpOverrides,
)
from torch._inductor.virtualized import V
from torch.utils._sympy.value_ranges import ValueRanges


def gemm_epilogue_op_scope(cute: Any) -> dict[str, Any]:
    import operator

    import cutlass
    from cutlass._mlir.dialects import math as mlir_math

    def sigmoid(x: Any) -> Any:
        return 1.0 / (1.0 + cute.math.exp(-x))

    return {
        "cutlass": cutlass,
        "operator": operator,
        "mlir_math": mlir_math,
        "cute": cute,
        "erf": cute.math.erf,
        "exp": cute.math.exp,
        "gelu": lambda x: 0.5 * x * (1.0 + cute.math.erf(x * 0.7071067811865476)),
        "relu": lambda x: cute.math.max(x, cute.full_like(x, 0.0)),
        "sigmoid": sigmoid,
        "silu": lambda x: x * sigmoid(x),
        "tanh": cute.math.tanh,
    }


class GemmEpilogueCuteDSLBody:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def writeline(self, line: str) -> None:
        self.lines.append(line)


class GemmEpilogueCuteDSLCSE:
    def __init__(self) -> None:
        self.index = 0

    def generate(self, body, expr, *, bounds=None, dtype=None, shape=None):
        name = f"tmp{self.index}"
        self.index += 1
        body.writeline(f"{name} = {expr}")
        return CuteDSLCSEVariable(
            name,
            ValueRanges.unknown() if bounds is None else bounds,
            dtype=dtype,
            shape=shape,
        )


class GemmEpilogueCuteDSLKernel:
    def __init__(self) -> None:
        self.body = GemmEpilogueCuteDSLBody()
        self.cse = GemmEpilogueCuteDSLCSE()


class GemmEpilogueCuteDSLOpOverrides(CuteDSLOpOverrides):
    """Normalize ATen spellings before emitting CuTeDSL expressions."""

    @staticmethod
    def add(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.add(a, rhs)

    @staticmethod
    def sub(a: Any, b: Any, *, alpha: Any = 1) -> Any:
        rhs = b if alpha == 1 else CuteDSLOpOverrides.mul(b, alpha)
        return CuteDSLOpOverrides.sub(a, rhs)

    @staticmethod
    def _to_copy(x: Any, *, dtype: torch.dtype, **kwargs: Any) -> Any:
        unsupported_kwargs = {
            key: value
            for key, value in kwargs.items()
            if value not in (None, False, torch.preserve_format)
        }
        if unsupported_kwargs:
            raise NotImplementedError(
                f"unsupported GEMM epilogue _to_copy kwargs: {unsupported_kwargs}"
            )
        return CuteDSLOpOverrides.to_dtype(x, dtype)

    @staticmethod
    def where(condition: Any, a: Any, b: Any) -> Any:
        """Preserve all-scalar conditionals as scalar epilogue expressions."""
        if any(
            CuteDSLOpOverrides._is_tensor_like(value) for value in (condition, a, b)
        ):
            return CuteDSLOpOverrides.where(condition, a, b)
        result_expr = (
            f"({CuteDSLOpOverrides._as_expr(a)} if "
            f"{CuteDSLOpOverrides._as_expr(condition)} else "
            f"{CuteDSLOpOverrides._as_expr(b)})"
        )
        cse_vars = tuple(
            CuteDSLOpOverrides._get_cse_var(value) for value in (a, b, condition)
        )
        if all(value is None for value in cse_vars):
            return result_expr
        dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(a, b, condition)
        result = V.kernel.cse.generate(
            V.kernel.body,
            result_expr,
            bounds=bounds,
            dtype=dtype if dtype is not None else torch.int32,
        )
        result.is_scalar_expr = True
        return result

    @staticmethod
    def clamp(x: Any, min: Any = None, max: Any = None) -> Any:
        result = x
        if min is not None:
            result = CuteDSLOpOverrides.maximum(result, min)
        if max is not None:
            result = CuteDSLOpOverrides.minimum(result, max)
        return result

    @staticmethod
    def clamp_min(x: Any, min: Any) -> Any:
        return CuteDSLOpOverrides.maximum(x, min)

    @staticmethod
    def clamp_max(x: Any, max: Any) -> Any:
        return CuteDSLOpOverrides.minimum(x, max)

    @staticmethod
    def convert_element_type(x: Any, dtype: torch.dtype) -> Any:
        return CuteDSLOpOverrides.to_dtype(x, dtype)
<|endoftext|># mypy: allow-untyped-defs
"""Symbolic shape helpers shared by GEMM epilogue frontends."""

from collections.abc import Sequence
from typing import Any

import sympy

import torch
from torch._inductor.virtualized import V
from torch.fx.experimental.symbolic_shapes import (
    guard_int,
    has_guarding_hint,
    statically_known_true as fx_statically_known_true,
)


def normalize_shape(shape: Any) -> Any:
    """Canonicalize sequence-like shapes to tuples."""
    return tuple(shape) if isinstance(shape, (list, tuple, torch.Size)) else shape


def guarded_int(value: Any) -> int | None:
    """Return an integer after guarding backed symbolic values."""
    if isinstance(value, torch.fx.Node):
        value = value.meta.get("val")
    if isinstance(value, torch.SymInt):
        if not has_guarding_hint(value):
            return None
        # TODO: Defer speculative guards if they become a meaningful source of
        # recompilation.
        return guard_int(value)
    return value if isinstance(value, int) else None


def statically_known(expr: Any) -> bool:
    """Return whether a symbolic predicate is known true without adding guards."""
    if isinstance(expr, bool):
        return expr
    if isinstance(expr, sympy.Basic):
        return V.graph.sizevars.statically_known_true(expr)
    return fx_statically_known_true(expr)


def statically_known_equal(lhs: Any, rhs: Any) -> bool:
    """Return whether symbolic shape values are known equal without adding guards."""
    return statically_known(lhs == rhs)


def statically_known_shape_equal(
    actual_shape: Sequence[Any], expected_shape: Sequence[Any]
) -> bool:
    """Compare possibly symbolic shape tuples without adding guards."""
    return len(actual_shape) == len(expected_shape) and all(
        statically_known_equal(actual, expected)
        for actual, expected in zip(actual_shape, expected_shape)
    )
<|endoftext|># mypy: allow-untyped-defs
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

import torch
from torch._inductor.codegen.rocm.ck_conv_template import CKGroupedConvFwdTemplate

from .. import config, ir
from ..lowering import (
    add_layout_constraint,
    constrain_to_fx_strides,
    fallback_handler,
    lowerings as L,
    register_lowering,
)
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    SymbolicGridFn,
    TritonTemplate,
)
from ..utils import (
    is_ones,
    is_zeros,
    pad_listlike,
    sympy_product,
    use_ck_conv_template,
    use_triton_template,
)
from ..virtualized import V
from .mm_common import load_kernel_template


if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..ir import TensorBox

log = logging.getLogger(__name__)


aten = torch.ops.aten


@SymbolicGridFn
def conv2d_grid(n, c, h, w, meta, *, cdiv):
    return (
        cdiv(n * h * w, meta["BLOCK_M"]),
        cdiv(c, meta["BLOCK_N"]),
        meta["GROUPS"],
    )


@SymbolicGridFn
def conv3d_grid(n, c, d, h, w, meta, *, cdiv):
    return (
        cdiv(n * d * h * w, meta["BLOCK_M"]),
        cdiv(c, meta["BLOCK_N"]),
        meta["GROUPS"],
    )


# =============================================================================
# Depthwise conv1d (groups == in_channels == out_channels)
# Uses direct element-wise multiply-accumulate instead of implicit GEMM.
# Channels-last (NLC) layout with 3D tiling: BLOCK_N x BLOCK_L x BLOCK_C.
# =============================================================================


@SymbolicGridFn
def depthwise_conv1d_grid(n, c, l, meta, *, cdiv):
    return (
        cdiv(n, meta["BLOCK_N"]),
        cdiv(l, meta["BLOCK_L"]),
        cdiv(c, meta["BLOCK_C"]),
    )


depthwise_conv1d_template = TritonTemplate(
    name="depthwise_conv1d",
    grid=depthwise_conv1d_grid,
    source=load_kernel_template("triton_depthwise_conv"),
    cache_codegen_enabled_for_template=True,
)

LOOP_BODY_2D = """
        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)
"""

"""
This is a relatively simple conv implementation that can likely be
improved.  Many alternate conv versions can be found here:
https://github.com/pytorch/torchdynamo/pull/971
"""
conv2d_template = TritonTemplate(
    name="convolution2d",
    grid=conv2d_grid,
    source=r"""
{{def_kernel("X", "W")}}
    # Tensor dimensions
    BATCH = {{size("X", 0)}}
    IN_C = {{size("X", 1)}}
    IN_H = {{size("X", 2)}}
    IN_W = {{size("X", 3)}}
    OUT_C = {{size(None, 1)}}
    OUT_H = {{size(None, 2)}}
    OUT_W = {{size(None, 3)}}

    # Strides:
    stride_xn = {{stride("X", 0)}}
    stride_xc = {{stride("X", 1)}}
    stride_xh = {{stride("X", 2)}}
    stride_xw = {{stride("X", 3)}}
    stride_wc_out = {{stride("W", 0)}}
    stride_wc_in = {{stride("W", 1)}}
    stride_wh = {{stride("W", 2)}}
    stride_ww = {{stride("W", 3)}}

    nhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = nhw % OUT_W
    nh = nhw // OUT_W
    idx_y_h = nh % OUT_H
    idx_n = nh // OUT_H
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)

{% if GROUPS == 1 %}
    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C
{% else %}
    group = tl.program_id(2).to(INDEX_DTYPE)
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
{% endif %}

    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

{% if UNROLL %}
{% for i in range(KERNEL_H) %}
{% for j in range(KERNEL_W) %}
    i = {{i}}
    j = {{j}}
    for k in range(0, GROUP_IN_C, BLOCK_K):
        """
    + LOOP_BODY_2D
    + """
{% endfor %}
{% endfor %}
{% else %}
    # Could be simplified, but slightly slower:
    # for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        i = ij // KERNEL_W
        j = ij % KERNEL_W
        """
    + LOOP_BODY_2D
    + """
{% endif %}

    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    {{store_output(("idx_n", "idx_c", "idx_h", "idx_w"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
""",
)

LOOP_BODY_3D = """
        idx_x_d = d - PADDING_D + idx_y_d * STRIDE_D
        idx_x_h = i - PADDING_H + idx_y_h * STRIDE_H
        idx_x_w = j - PADDING_W + idx_y_w * STRIDE_W
        idx_x_c = tl.arange(0, BLOCK_K) + k

        x_ptrs = x_base + (
            (idx_x_d * stride_xd)[:, None]
            + (idx_x_h * stride_xh)[:, None]
            + (idx_x_w * stride_xw)[:, None]
            + (idx_x_c * stride_xc)[None, :]
        )
        mask_x = (
            (idx_n < BATCH)[:, None]
            & (idx_x_d >= 0)[:, None]
            & (idx_x_d < IN_D)[:, None]
            & (idx_x_h >= 0)[:, None]
            & (idx_x_h < IN_H)[:, None]
            & (idx_x_w >= 0)[:, None]
            & (idx_x_w < IN_W)[:, None]
            & (idx_x_c < GROUP_IN_C)[None, :]
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)

        w_ptrs = w_base + (
            (idx_x_c * stride_wc_in)[:, None] +
            (d * stride_wd) + (i * stride_wh) + (j * stride_ww)
        )
        mask_w = (idx_x_c[:, None] < GROUP_IN_C) & (idx_y_c[None, :] < GROUP_OUT_C)
        matrix_w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc += tl.dot(matrix_x, matrix_w, allow_tf32=ALLOW_TF32)
"""

conv3d_template = TritonTemplate(
    name="convolution3d",
    grid=conv3d_grid,
    source=r"""
{{def_kernel("X", "W")}}
    # Tensor dimensions
    BATCH = {{size("X", 0)}}
    IN_C = {{size("X", 1)}}
    IN_D = {{size("X", 2)}}
    IN_H = {{size("X", 3)}}
    IN_W = {{size("X", 4)}}
    OUT_C = {{size(None, 1)}}
    OUT_D = {{size(None, 2)}}
    OUT_H = {{size(None, 3)}}
    OUT_W = {{size(None, 4)}}

    # Strides:
    stride_xn = {{stride("X", 0)}}
    stride_xc = {{stride("X", 1)}}
    stride_xd = {{stride("X", 2)}}
    stride_xh = {{stride("X", 3)}}
    stride_xw = {{stride("X", 4)}}
    stride_wc_out = {{stride("W", 0)}}
    stride_wc_in = {{stride("W", 1)}}
    stride_wd = {{stride("W", 2)}}
    stride_wh = {{stride("W", 3)}}
    stride_ww = {{stride("W", 4)}}

    ndhw = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M + tl.arange(0, BLOCK_M)
    idx_y_w = ndhw % OUT_W
    ndh = ndhw // OUT_W
    idx_y_h = ndh % OUT_H
    nd = ndh // OUT_H
    idx_y_d = nd % OUT_D
    idx_n = nd // OUT_D
    idx_y_c = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)

{% if GROUPS == 1 %}
    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C
{% else %}
    group = tl.program_id(2).to(INDEX_DTYPE)
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
{% endif %}

    x_base = X + (group * stride_xc * GROUP_IN_C + idx_n * stride_xn)[:, None]
    w_base = (
        W + (group * stride_wc_out * GROUP_OUT_C + idx_y_c * stride_wc_out)[None, :]
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

{% if UNROLL %}
{% for d in range(KERNEL_D) %}
{% for i in range(KERNEL_H) %}
{% for j in range(KERNEL_W) %}
    d = {{d}}
    i = {{i}}
    j = {{j}}
    for k in range(0, GROUP_IN_C, BLOCK_K):
        """
    + LOOP_BODY_3D
    + """
{% endfor %}
{% endfor %}
{% endfor %}
{% else %}
    # Could be simplified, but slightly slower:
    # for d in range(KERNEL_D):
    #   for i in range(KERNEL_H):
    #     for j in range(KERNEL_W):
    #         for k in range(0, GROUP_IN_C, BLOCK_K):
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K
    for dijk in range(KERNEL_D * KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k = (dijk % BLOCK_K_COUNT) * BLOCK_K
        dij = dijk // BLOCK_K_COUNT
        j = dij % KERNEL_W
        di = dij // KERNEL_W
        i = di % KERNEL_H
        d = di // KERNEL_H
        """
    + LOOP_BODY_3D
    + """
{% endif %}

    mask = (
        (idx_n < BATCH)[:, None]
        & (idx_y_d < OUT_D)[:, None]
        & (idx_y_h < OUT_H)[:, None]
        & (idx_y_w < OUT_W)[:, None]
        & (idx_y_c < GROUP_OUT_C)[None, :]
    )
    idx_n = idx_n[:, None]
    idx_c = idx_y_c[None, :] + group * GROUP_OUT_C
    idx_d = idx_y_d[:, None]
    idx_h = idx_y_h[:, None]
    idx_w = idx_y_w[:, None]

    # inductor generates a suffix
    {{store_output(("idx_n", "idx_c", "idx_d", "idx_h", "idx_w"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
""",
)

aten_convolution = ExternKernelChoice(
    torch.convolution,
    "at::convolution",
    has_out_variant=False,
    op_overload=aten.convolution.default,
)


def conv1x1_via_mm(x, w, *, out):
    w = torch.squeeze(torch.squeeze(w, -1), -1)
    return torch.matmul(
        x.permute(0, 2, 3, 1), w.permute(1, 0), out=out.permute(0, 2, 3, 1)
    )


aten_conv1x1_via_mm = ExternKernelChoice(conv1x1_via_mm, None)


class ConvLayoutParams(TypedDict):
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    transposed: bool
    output_padding: tuple[int, ...]
    groups: int


def conv_layout(
    x: TensorBox,
    weight: TensorBox,
    bias: TensorBox | None,
    stride: Sequence[int],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    transposed: bool,
    output_padding: tuple[int, ...],
    groups: int,
) -> ir.Layout:
    """Determine output layout for a convolution"""
    # We use guard_int_seq rather than size_hints because the output shape
    # depends on these values — if they ever contained symbols, size_hints
    # would silently substitute a hint that could be wrong, producing an
    # incorrect layout. guard_int_seq will install a proper guard instead.
    # Note: stride and padding are already guarded via guard_int_seq in
    # convolution() above, but we guard all four here so conv_layout is
    # self-contained and doesn't rely on callers.
    guard = V.graph.sizevars.guard_int_seq
    with V.graph.fake_mode:
        output = torch.ops.aten.convolution(
            ir.ir_node_to_tensor(x),
            ir.ir_node_to_tensor(weight),
            ir.ir_node_to_tensor(bias),
            guard(stride),
            guard(padding),
            guard(dilation),
            transposed,
            guard(output_padding),
            groups,
        )
        sizes = ir.convert_shape_to_inductor(output.size())
        stride = ir.convert_shape_to_inductor(output.stride())  # type: ignore[assignment]

    return ir.FixedLayout(
        x.get_device_or_error(),
        x.get_dtype(),
        sizes,
        stride,
    )


def channels_last_order(rank):
    order = list(reversed(range(rank)))
    order.insert(1, order.pop(-1))
    return order


def convert_1x1_conv_to_mm(x, weight, bias):
    # special case for 1x1 convolution, which is actually just a matmul
    rank = len(weight.get_size())
    for _ in range(rank - 2):
        weight = L[aten.squeeze](weight, dim=-1)
    weight = L[aten.permute](weight, [1, 0])

    x = ir.ExternKernel.require_stride_order(x, channels_last_order(rank))
    x_permute = list(range(rank))
    x_permute.append(x_permute.pop(1))
    x = L[aten.permute](x, x_permute)
    *sizes, in_chan = x.get_size()
    x = L[aten.reshape](x, [sympy_product(sizes), in_chan])
    if bias is None:
        result = L[aten.mm](x, weight)
    else:
        result = L[aten.addmm](bias, x, weight)
    result = L[aten.reshape](result, [*sizes, -1])
    result_permute = list(range(rank))
    result_permute.insert(1, result_permute.pop(-1))
    return L[aten.permute](result, result_permute)


@register_lowering(aten.convolution)
def convolution(
    x: TensorBox,
    weight: TensorBox,
    bias: TensorBox | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
):
    """Lower aten.convolution using Inductor convolution kernels or fallbacks."""
    stride = tuple(stride)
    padding = tuple(padding)
    dilation = tuple(dilation)
    output_padding = tuple(output_padding)
    if not isinstance(groups, int):
        groups = V.graph.sizevars.guard_int(groups)
    if not isinstance(groups, int):
        raise AssertionError(f"Expected int for groups, got {type(groups)}")

    # Need use hint for triton template since the template does not
    # work with a dynamic shape.
    #
    # Dilation must also be guarded for transposed convolutions because the
    # backward-input template substitutes DILATION_H/W as tl.constexpr.
    # For non-transposed convolutions the forward template gates on
    # dilation==1, so guarding is unnecessary.
    stride = tuple(V.graph.sizevars.guard_int_seq(stride))
    padding = tuple(V.graph.sizevars.guard_int_seq(padding))
    if transposed:
        dilation = tuple(V.graph.sizevars.guard_int_seq(dilation))

    kwargs: ConvLayoutParams = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "transposed": transposed,
        "output_padding": output_padding,
        "groups": groups,
    }

    device_type = ir.get_device_type(x)

    if len(x.get_size()) == len(weight.get_size()) - 1:
        # add batch dimension to simplify rest of function
        return L[aten.squeeze](
            convolution(L[aten.expand](x, [1, *x.get_size()]), weight, bias, **kwargs),
            dim=0,
        )

    out_chan, in_chan, *kernel_shape = V.graph.sizevars.guard_int_seq(weight.get_size())

    # Always convert conv1D to 2D for Intel GPU.
    # Only conv2D can be converted to channel last layout,
    # which have much better performance.
    if len(x.get_size()) == 3 and len(kernel_shape) == 1 and device_type == "xpu":
        kwargs.update(
            {
                "stride": (1,) + stride,
                "padding": (0,) + padding,
                "dilation": (1,) + dilation,
                "output_padding": (0,) + output_padding,
            }
        )
        # (N, C, L) -> (N, C, 1, L)
        x = L[aten.unsqueeze](x, dim=2)
        weight = L[aten.unsqueeze](weight, dim=2)

        return L[aten.squeeze](
            convolution(x, weight, bias, **kwargs),
            dim=2,
        )

    ndim = len(kernel_shape)
    stride = pad_listlike(stride, ndim)
    padding = pad_listlike(padding, ndim)
    dilation = pad_listlike(dilation, ndim)
    output_padding = pad_listlike(output_padding, ndim)

    def channels_last_conv():
        if V.graph.layout_opt and ndim == 2:
            return True

        layout = conv_layout(x, weight, None, **kwargs)
        # TODO: This does not guard on the stride order decision,
        # shall we use optimization_hint to handle unbacked?
        req_stride_order = ir.get_stride_order(
            V.graph.sizevars.guarding_hints_or_throw(layout.stride)
        )
        return req_stride_order == ir.NHWC_STRIDE_ORDER

    autotuning_gemm = config.max_autotune or config.max_autotune_gemm

    if (
        (config.conv_1x1_as_mm or (autotuning_gemm and channels_last_conv()))
        and is_ones(kernel_shape)
        and is_ones(stride)
        and is_zeros(padding)
        and is_ones(dilation)
        and not transposed
        and is_zeros(output_padding)
        and groups == 1
        and V.graph.sizevars.statically_known_gt(sympy_product(x.get_size()), 0)
    ):
        return convert_1x1_conv_to_mm(x, weight, bias)

    if bias is not None and device_type != "cpu":
        # peel off the bias, cudnn is slower with it
        result = convolution(x, weight, None, **kwargs)
        if V.graph.sizevars.statically_known_equals(result.get_size()[1], 0):
            # we should not add bias when the output channel is 0
            return result
        return L[aten.add](
            result, L[aten.view](bias, [result.get_size()[1]] + ndim * [1])
        )

    x.realize()
    weight.realize()

    # ndim can be 1 for convolution in models such as demucs
    # TODO: check if it's beneficial to convert Conv1d to Conv2d and then
    # apply channels last.
    if V.graph.layout_opt and ndim == 2:
        V.graph.num_channels_last_conv += 1
        x = ir.ExternKernel.require_channels_last(x)  # type: ignore[assignment]
        # TODO maybe we can convert weights to channels last just once before
        # running the model.
        weight = ir.ExternKernel.require_channels_last(weight)  # type: ignore[assignment]
        layout = conv_layout(x, weight, None, **kwargs)
    else:
        layout = conv_layout(x, weight, None, **kwargs)
        # TODO: This does not guard on the stride order decision,
        # shall we use optimization_hint to handle unbacked?
        req_stride_order = ir.get_stride_order(
            V.graph.sizevars.guarding_hints_or_throw(layout.stride)
        )
        x = ir.ExternKernel.require_stride_order(x, req_stride_order)  # type: ignore[assignment]
        weight = ir.ExternKernel.require_stride_order(weight, req_stride_order)  # type: ignore[assignment]

    ordered_kwargs_for_cpp_kernel = [
        "stride",
        "padding",
        "dilation",
        "transposed",
        "output_padding",
        "groups",
    ]
    if bias is None:
        args = [x, weight]
        kwargs["bias"] = None  # type: ignore[typeddict-unknown-key]
        ordered_kwargs_for_cpp_kernel.insert(0, "bias")
    else:
        bias = ir.ExternKernel.realize_input(bias)  # type: ignore[assignment]
        if bias is None:
            raise AssertionError("bias must not be None after realize_input")
        args = [x, weight, bias]
        bias.freeze_layout()
        V.graph.sizevars.guard_int_seq(bias.get_size())

    choices = []
    if torch._inductor.utils._use_conv_autotune_backend("ATEN"):
        choices = [
            aten_convolution.bind(
                args,
                layout,
                ordered_kwargs_for_cpp_kernel,
                **kwargs,
            )
        ]

    if (
        torch._inductor.utils._use_conv_autotune_backend("TRITON")
        and use_triton_template(layout)
        # templates only support these:
        and is_ones(dilation)
        and not transposed
        and is_zeros(output_padding)
        # there are some odd models where this check fails (e.g. shufflenet_v2_x1_0)
        and V.graph.sizevars.statically_known_equals(in_chan * groups, x.get_size()[1])  # type: ignore[arg-type]
    ):
        if (
            is_ones(kernel_shape)
            and is_ones(stride)
            and is_zeros(padding)
            and groups == 1
        ):
            choices.append(aten_conv1x1_via_mm.bind(args, layout))

        is_depthwise = groups > 1 and in_chan == 1 and out_chan == groups
        if is_depthwise and ndim == 1:
            depthwise_configs = V.choices.get_depthwise_conv_configs(device_type)
            for cfg in depthwise_configs:
                depthwise_conv1d_template.maybe_append_choice(
                    choices,
                    input_nodes=(x, weight),
                    layout=layout,
                    KERNEL_SIZE=kernel_shape[0],
                    CONV_STRIDE=stride[0],
                    PADDING=padding[0],
                    num_stages=cfg.num_stages,
                    num_warps=cfg.num_warps,
                    **cfg.kwargs,
                )

        conv_configs = V.choices.get_conv_configs(device_type)

        dtype_size = x.get_dtype().itemsize
        for cfg in conv_configs(
            sympy_product([x.get_size()[0], *x.get_size()[2:]]),
            out_chan,
            in_chan,
            dtype_size=dtype_size,
        ):
            unroll = is_ones(kernel_shape)
            # The non-unrolled loop in these templates triggers triton#1254
            # with 8 warps, producing incorrect results for non-1x1 kernels.
            num_warps = cfg.num_warps if unroll else min(cfg.num_warps, 4)

            if ndim == 2:
                conv2d_template.maybe_append_choice(
                    choices,
                    input_nodes=(x, weight),
                    layout=layout,
                    KERNEL_H=kernel_shape[0],
                    KERNEL_W=kernel_shape[1],
                    STRIDE_H=stride[0],
                    STRIDE_W=stride[1],
                    PADDING_H=padding[0],
                    PADDING_W=padding[1],
                    GROUPS=groups,
                    # TODO(jansel): try unroll for bigger kernels once fixed:
                    #               https://github.com/triton-lang/triton/issues/1254
                    UNROLL=unroll,
                    ALLOW_TF32=torch.backends.cudnn.fp32_precision == "tf32",
                    num_stages=cfg.num_stages,
                    num_warps=num_warps,
                    **cfg.kwargs,
                )
            elif ndim == 3:
                conv3d_template.maybe_append_choice(
                    choices,
                    input_nodes=(x, weight),
                    layout=layout,
                    KERNEL_D=kernel_shape[0],
                    KERNEL_H=kernel_shape[1],
                    KERNEL_W=kernel_shape[2],
                    STRIDE_D=stride[0],
                    STRIDE_H=stride[1],
                    STRIDE_W=stride[2],
                    PADDING_D=padding[0],
                    PADDING_H=padding[1],
                    PADDING_W=padding[2],
                    GROUPS=groups,
                    # TODO(jansel): try unroll for bigger kernels once fixed:
                    #               https://github.com/triton-lang/triton/issues/1254
                    UNROLL=unroll,
                    ALLOW_TF32=torch.backends.cudnn.fp32_precision == "tf32",
                    num_stages=cfg.num_stages,
                    num_warps=num_warps,
                    **cfg.kwargs,
                )
    # The Triton conv2d backward-input kernels miscompile on CUDA (wrong results
    # on sm90, ptxas illegal memory access on sm100) with no perf win, so keep
    # them on ROCm and fall back to ATEN on CUDA. Same rationale as the
    # convolution_backward lowering. See
    # https://github.com/pytorch/pytorch/issues/187081.
    disable_triton_conv_bwd = device_type == "cuda" and not torch.version.hip
    if (
        torch._inductor.utils._use_conv_autotune_backend("TRITON")
        and use_triton_template(layout)
        and transposed
        and ndim == 2
        and not disable_triton_conv_bwd
    ):
        # ConvTranspose2d is mathematically identical to conv_backward_input:
        # the input plays the role of grad_output and the same weight layout
        # is used. Reuse the backward-input Triton template.
        conv_configs = V.choices.get_conv_configs(device_type)
        dtype_size = x.get_dtype().itemsize
        for cfg in conv_configs(
            sympy_product([layout.size[0], layout.size[2], layout.size[3]]),
            out_chan,
            in_chan,
            dtype_size=dtype_size,
        ):
            conv2d_bwd_input_template.maybe_append_choice(
                choices,
                input_nodes=(x, weight),
                layout=layout,
                KERNEL_H=kernel_shape[0],
                KERNEL_W=kernel_shape[1],
                PADDING_H=padding[0],
                PADDING_W=padding[1],
                STRIDE_H=stride[0],
                STRIDE_W=stride[1],
                DILATION_H=dilation[0],
                DILATION_W=dilation[1],
                GROUPS=groups,
                ALLOW_TF32=torch.backends.cudnn.fp32_precision == "tf32",
                num_stages=cfg.num_stages,
                num_warps=cfg.num_warps,
                **cfg.kwargs,
            )

    if use_ck_conv_template(layout):
        CKGroupedConvFwdTemplate.add_ck_conv_choices(
            choices,
            layout,
            input_nodes=(x, weight) + ((bias,) if bias is not None else tuple()),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            n_spatial_dimensions=ndim,
        )

    if not choices and config.max_autotune_conv_backends.strip():
        choices.append(
            aten_convolution.bind(
                args,
                layout,
                ordered_kwargs_for_cpp_kernel,
                **kwargs,
            )
        )

    node, _ = autotune_select_algorithm("convolution", choices, args, layout)
    return node


@register_lowering(aten._convolution)
def _convolution(
    x,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    benchmark,
    deterministic,
    cudnn_enabled,
    allow_tf32,
):
    return convolution(
        x, weight, bias, stride, padding, dilation, transposed, output_padding, groups
    )


def constrain_conv_to_fx_strides(fx_node, *args, **kwargs):
    if fx_node.target is not torch.ops.aten.convolution.default:
        raise AssertionError(f"Expected aten.convolution.default, got {fx_node.target}")
    if V.graph.layout_opt:
        return args, kwargs
    else:
        return constrain_to_fx_strides(fx_node, *args, **kwargs)


add_layout_constraint(aten.convolution, constrain_conv_to_fx_strides)


def conv_bwd_input_layout(
    grad_out: TensorBox,
    input: TensorBox,
    weight: TensorBox,
    stride: Sequence[int],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    transposed: bool,
    output_padding: tuple[int, ...],
    groups: int,
) -> ir.Layout:
    guard = V.graph.sizevars.guard_int_seq
    with V.graph.fake_mode:
        go = ir.ir_node_to_tensor(grad_out)
        x = ir.ir_node_to_tensor(input)
        w = ir.ir_node_to_tensor(weight)

        dx, _, _ = torch.ops.aten.convolution_backward(
            go,
            x,
            w,
            None,  # bias_sizes
            guard(stride),
            guard(padding),
            guard(dilation),
            transposed,
            guard(output_padding),
            groups,
            (True, False, False),
        )
        sizes = ir.convert_shape_to_inductor(dx.size())
        stride_ = ir.convert_shape_to_inductor(dx.stride())

    return ir.FixedLayout(
        input.get_device_or_error(),
        input.get_dtype(),
        sizes,
        stride_,
    )


def conv_bwd_weight_layout(
    grad_out: TensorBox,
    input: TensorBox,
    weight: TensorBox,
    stride: Sequence[int],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    transposed: bool,
    output_padding: tuple[int, ...],
    groups: int,
) -> ir.Layout:
    guard = V.graph.sizevars.guard_int_seq
    with V.graph.fake_mode:
        go = ir.ir_node_to_tensor(grad_out)
        x = ir.ir_node_to_tensor(input)
        w = ir.ir_node_to_tensor(weight)

        _, dw, _ = torch.ops.aten.convolution_backward(
            go,
            x,
            w,
            None,  # bias_sizes
            guard(stride),
            guard(padding),
            guard(dilation),
            transposed,
            guard(output_padding),
            groups,
            (False, True, False),
        )
        sizes = ir.convert_shape_to_inductor(dw.size())
        stride_ = ir.convert_shape_to_inductor(dw.stride())

    return ir.FixedLayout(
        weight.get_device_or_error(),
        weight.get_dtype(),
        sizes,
        stride_,
    )


def call_aten_dw(
    x_t,
    go_t,
    *,
    w_shape,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    out,
):
    if x_t.is_contiguous(memory_format=torch.channels_last):
        memory_fmt = torch.channels_last
    else:
        memory_fmt = torch.contiguous_format

    dummy_weight = torch.empty(
        w_shape, dtype=out.dtype, device=x_t.device, memory_format=memory_fmt
    )

    torch.ops.aten.convolution_backward.out(
        out1=None,
        out2=out,
        out3=None,
        grad_output=go_t,
        input=x_t,
        weight=dummy_weight,
        bias_sizes=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=transposed,
        output_padding=output_padding,
        groups=groups,
        output_mask=(False, True, False),
    )
    return out


ext_kn_aten_dw = ExternKernelChoice(call_aten_dw, None)


def call_aten_dx(
    go_t,
    w_t,
    *,
    x_shape,
    stride,
    padding,
    dilation,
    transposed,
    output_padding,
    groups,
    out,
):
    if go_t.is_contiguous(memory_format=torch.channels_last):
        memory_fmt = torch.channels_last
    else:
        memory_fmt = torch.contiguous_format

    dummy_input = torch.empty(
        x_shape, dtype=out.dtype, device=go_t.device, memory_format=memory_fmt
    )

    torch.ops.aten.convolution_backward.out(
        out1=out,
        out2=None,
        out3=None,
        grad_output=go_t,
        input=dummy_input,
        weight=w_t,
        bias_sizes=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        transposed=transposed,
        output_padding=output_padding,
        groups=groups,
        output_mask=(True, False, False),
    )
    return out


ext_kn_aten_dx = ExternKernelChoice(call_aten_dx, None)


@SymbolicGridFn
def conv2d_bwd_weight_grid(cout, cin, kh, kw, meta, *, cdiv):
    g = meta["GROUPS"]
    return (
        cdiv(cin * kh * kw, meta["BLOCK_M"]),
        cdiv(cout // g, meta["BLOCK_N"]),
        g,
    )


conv2d_bwd_weight_template = TritonTemplate(
    name="convolution2d_bwd_weight",
    grid=conv2d_bwd_weight_grid,
    source=load_kernel_template("triton_conv2d_bwd_weight"),
)


@SymbolicGridFn
def conv2d_bwd_input_grid(n, cin, h, w, meta, *, cdiv):
    g = meta["GROUPS"]
    return (
        cdiv(n * h * w, meta["BLOCK_M"]),
        cdiv(cin // g, meta["BLOCK_N"]),
        g,
    )


conv2d_bwd_input_template = TritonTemplate(
    name="convolution2d_bwd_input",
    grid=conv2d_bwd_input_grid,
    source=load_kernel_template("triton_conv2d_bwd_input"),
)


aten_convolution_backward_fallback = fallback_handler(aten.convolution_backward.default)


@register_lowering(aten.convolution_backward.default)
def convolution_backward_lowering(
    grad_out: TensorBox,
    input: TensorBox,
    weight: TensorBox,
    bias_sizes: Sequence[int] | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
    output_mask: Sequence[bool],
):
    """
    Lowering function for backward convolution operator.

    TRITON kernels are only registered for supported configurations (currently 2D convolutions).
    For unsupported dimensions or configurations, the choices list remains empty,
    triggering an automatic fallback to the ATen reference implementation.
    This ensures correctness for all cases while enabling TRITON optimizations only where implemented.
    """
    stride = tuple(stride)
    padding = tuple(padding)
    dilation = tuple(dilation)
    output_padding = tuple(output_padding)
    if not isinstance(groups, int):
        groups = V.graph.sizevars.guard_int(groups)

    out_chan, in_chan, *kernel_shape = V.graph.sizevars.guard_int_seq(weight.get_size())

    # The Triton bwd templates substitute DILATION_H/W into the generated
    # kernel source, so they must be concrete Python ints. The fwd template
    # gates on is_ones(dilation) and hard-codes dilation=1, so it can skip
    # this guard.
    stride = tuple(V.graph.sizevars.guard_int_seq(stride))
    padding = tuple(V.graph.sizevars.guard_int_seq(padding))
    dilation = tuple(V.graph.sizevars.guard_int_seq(dilation))

    input.realize()
    weight.realize()
    grad_out.realize()

    kwargs: ConvLayoutParams = {
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "transposed": transposed,
        "output_padding": output_padding,
        "groups": groups,
    }

    ndim = len(kernel_shape)
    stride = pad_listlike(stride, ndim)
    padding = pad_listlike(padding, ndim)
    dilation = pad_listlike(dilation, ndim)
    output_padding = pad_listlike(output_padding, ndim)

    device_type = ir.get_device_type(input)

    # The Triton conv2d backward kernels hit a ptxas miscompile (illegal memory
    # access) on NVIDIA sm100 and show no perf win on CUDA, so keep them on ROCm
    # only and fall back to ATEN. See
    # https://github.com/pytorch/pytorch/issues/187081.
    disable_triton_conv_bwd = device_type == "cuda" and not torch.version.hip

    conv_configs = V.choices.get_conv_configs(device_type)
    dtype_size = input.get_dtype().itemsize

    has_triton_dw_choices = False
    dw = None
    choices_dw = []
    args_w = []
    layout_dw = conv_bwd_weight_layout(grad_out, input, weight, **kwargs)
    if output_mask[1]:
        if V.graph.layout_opt and ndim == 2:
            V.graph.num_channels_last_conv += 1
            input = ir.ExternKernel.require_channels_last(input)  # type: ignore[assignment]
            grad_out = ir.ExternKernel.require_channels_last(grad_out)  # type: ignore[assignment]
            layout_dw = conv_bwd_weight_layout(grad_out, input, weight, **kwargs)
        else:
            guard = V.graph.sizevars.guard_int_seq
            stride_order = ir.get_stride_order(guard(layout_dw.stride))
            input = ir.ExternKernel.require_stride_order(input, stride_order)  # type: ignore[assignment]
            grad_out = ir.ExternKernel.require_stride_order(grad_out, stride_order)  # type: ignore[assignment]

        args_w = [input, grad_out]

        if (
            not disable_triton_conv_bwd
            and torch._inductor.utils._use_conv_bwd_weight_autotune_backend("TRITON")
            and use_triton_template(layout_dw)
            and not transposed
            and is_zeros(output_padding)
        ):
            for cfg in conv_configs(
                sympy_product([input.get_size()[0], *input.get_size()[2:]]),
                out_chan,
                in_chan,
                dtype_size=dtype_size,
            ):
                if ndim == 2:
                    has_triton_dw_choices = True
                    conv2d_bwd_weight_template.maybe_append_choice(
                        choices_dw,
                        input_nodes=(input, grad_out),
                        layout=layout_dw,
                        KERNEL_H=kernel_shape[0],
                        KERNEL_W=kernel_shape[1],
                        PADDING_H=padding[0],
                        PADDING_W=padding[1],
                        STRIDE_H=stride[0],
                        STRIDE_W=stride[1],
                        DILATION_H=dilation[0],
                        DILATION_W=dilation[1],
                        GROUPS=groups,
                        ALLOW_TF32=torch.backends.cudnn.allow_tf32,
                        num_stages=cfg.num_stages,
                        num_warps=cfg.num_warps,
                        **cfg.kwargs,
                    )

                # TODO: backward weight 3D

    has_triton_dx_choices = False
    dx = None
    choices_dx = []
    args_x = []
    layout_dx = conv_bwd_input_layout(grad_out, input, weight, **kwargs)
    if output_mask[0]:
        if V.graph.layout_opt and ndim == 2:
            V.graph.num_channels_last_conv += 1
            grad_out = ir.ExternKernel.require_channels_last(grad_out)  # type: ignore[assignment]
            weight = ir.ExternKernel.require_channels_last(weight)  # type: ignore[assignment]
            layout_dx = conv_bwd_input_layout(grad_out, input, weight, **kwargs)
        else:
            guard = V.graph.sizevars.guard_int_seq
            stride_order = ir.get_stride_order(guard(layout_dx.stride))
            grad_out = ir.ExternKernel.require_stride_order(grad_out, stride_order)  # type: ignore[assignment]
            weight = ir.ExternKernel.require_stride_order(weight, stride_order)  # type: ignore[assignment]

        args_x = [grad_out, weight]

        if (
            not disable_triton_conv_bwd
            and torch._inductor.utils._use_conv_bwd_input_autotune_backend("TRITON")
            and use_triton_template(layout_dx)
            and not transposed
            and is_zeros(output_padding)
        ):
            # TODO: Use the autotune configuration specific to backward convolution.
            for cfg in conv_configs(
                sympy_product([input.get_size()[0], *input.get_size()[2:]]),
                out_chan,
                in_chan,
                dtype_size=dtype_size,
            ):
                if ndim == 2:
                    has_triton_dx_choices = True
                    conv2d_bwd_input_template.maybe_append_choice(
                        choices_dx,
                        input_nodes=(grad_out, weight),
                        layout=layout_dx,
                        KERNEL_H=kernel_shape[0],
                        KERNEL_W=kernel_shape[1],
                        PADDING_H=padding[0],
                        PADDING_W=padding[1],
                        STRIDE_H=stride[0],
                        STRIDE_W=stride[1],
                        DILATION_H=dilation[0],
                        DILATION_W=dilation[1],
                        GROUPS=groups,
                        ALLOW_TF32=torch.backends.cudnn.allow_tf32,
                        num_stages=cfg.num_stages,
                        num_warps=cfg.num_warps,
                        **cfg.kwargs,
                    )

                # TODO: backward input 3D

    # Fallback when no TRITON choices available, i.e., ndim != 2, backend config = ATEN,...
    if not has_triton_dx_choices and not has_triton_dw_choices:
        return aten_convolution_backward_fallback(
            grad_out,
            input,
            weight,
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )

    if output_mask[1]:
        if (
            torch._inductor.utils._use_conv_bwd_weight_autotune_backend("ATEN")
            or not has_triton_dw_choices
        ):
            choices_dw.append(
                ext_kn_aten_dw.bind(
                    input_nodes=args_w,
                    layout=layout_dw,
                    ordered_kwargs_for_cpp_kernel=[
                        "w_shape",
                        "stride",
                        "padding",
                        "dilation",
                        "transposed",
                        "output_padding",
                        "groups",
                    ],
                    w_shape=weight.get_size(),
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    transposed=transposed,
                    output_padding=output_padding,
                    groups=groups,
                )
            )

        # TODO: use_ck_conv_template for bwd conv

        dw, _ = autotune_select_algorithm(
            "convolution_bwd_weight", choices_dw, args_w, layout_dw
        )

    if output_mask[0]:
        if (
            torch._inductor.utils._use_conv_bwd_input_autotune_backend("ATEN")
            or not has_triton_dx_choices
        ):
            choices_dx.append(
                ext_kn_aten_dx.bind(
                    input_nodes=args_x,
                    layout=layout_dx,
                    ordered_kwargs_for_cpp_kernel=[
                        "x_shape",
                        "stride",
                        "padding",
                        "dilation",
                        "transposed",
                        "output_padding",
                        "groups",
                    ],
                    x_shape=input.get_size(),
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    transposed=transposed,
                    output_padding=output_padding,
                    groups=groups,
                )
            )

        # TODO: use_ck_conv_template for bwd conv

        dx, _ = autotune_select_algorithm(
            "convolution_bwd_input", choices_dx, args_x, layout_dx
        )

    db = None
    if output_mask[2] and bias_sizes is not None:
        db = L[aten.sum](grad_out, axis=[0] + list(range(2, ndim + 2)))

    return (dx, dw, db)


def constrain_conv_bwd_to_fx_strides(fx_node, *args, **kwargs):
    if fx_node.target != torch.ops.aten.convolution_backward.default:
        raise AssertionError(
            f"expected convolution_backward target, got {fx_node.target}"
        )
    if V.graph.layout_opt:
        return args, kwargs
    else:
        return constrain_to_fx_strides(fx_node, *args, **kwargs)


add_layout_constraint(aten.convolution_backward, constrain_conv_bwd_to_fx_strides)
<|endoftext|># mypy: allow-untyped-defs
import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
from torch._inductor.heuristics.template.cutedsl import get_groupgemm_configs
from torch._inductor.runtime.triton_compat import tl
from torch._inductor.virtualized import V
from torch.utils._triton import has_triton

from ..ir import ChoiceCaller, Layout, TensorBox
from ..lowering import register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
    TritonTemplate,
)
from ..utils import (
    get_gpu_shared_memory,
    get_num_sms,
    has_free_symbols,
    use_aten_gemm_kernels,
    use_blackwell_cutedsl_grouped_mm,
    use_nv_universal_gemm_template,
    use_triton_template,
)
from .mm_common import (
    _is_static_problem,
    check_supported_striding,
    load_kernel_template,
    persistent_grouped_mm_grid,
)


log = logging.getLogger(__name__)
aten = torch.ops.aten


@dataclass
class Config:
    kwargs: dict[str, int]
    num_stages: int
    num_warps: int


_NV_CONFIGS = [
    Config(
        {
            "BLOCK_M": block_size_m,
            "BLOCK_N": block_size_n,
            "BLOCK_K": block_size_k,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [16, 32, 64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
]


def grouped_mm_configs():
    return _NV_CONFIGS


def early_config_prune(g, m, dtsize, configs, named_args):
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, num_consumer_groups = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["BLOCK_K"],
            config.num_stages,
            config.num_warps,
            getattr(config, "num_consumer_groups", 0),
        )

        # 1. Prune NV configs depending on g and m.
        if not has_free_symbols((g, m)):
            a_is_2d, b_is_2d = named_args["A_IS_2D"], named_args["B_IS_2D"]
            m_avg = m // g if a_is_2d and not b_is_2d else m
            if m_avg <= 16:
                if BLOCK_M > 32:
                    continue
            elif m_avg <= 32:
                if BLOCK_M > 64:
                    continue
            elif m_avg <= 64:
                if BLOCK_M <= 16:
                    continue
            else:
                if BLOCK_M <= 32:
                    continue

        # 2. make sure we have enough smem
        max_shared_memory = get_gpu_shared_memory()

        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        use_warp_specialization = num_consumer_groups >= 1

        # 3. make sure we can partition for ws
        if use_warp_specialization:
            if num_warps != 4:
                continue

            # "tritongpu-warp-spec-data-partition"
            m_slice = BLOCK_M // num_consumer_groups
            n_slice = BLOCK_N // num_consumer_groups
            if m_slice < 64 and n_slice < 256:
                continue

        pruned_configs.append(config)

    return pruned_configs


triton_grouped_mm_template = TritonTemplate(
    name="grouped_mm",
    grid=persistent_grouped_mm_grid,
    source=load_kernel_template("triton_mm_grouped"),
)

triton_scaled_grouped_mm_template = TritonTemplate(
    name="scaled_grouped_mm",
    grid=persistent_grouped_mm_grid,
    source=load_kernel_template("triton_mm_grouped"),
)

cutedsl_grouped_mm_template = CuteDSLTemplate(
    name="grouped_gemm_cutedsl",
    source=load_kernel_template("cutedsl_mm_grouped"),
)


def has_grouped_mm_triton_support() -> bool:
    if not torch.cuda.is_available():
        return False
    if torch.version.hip:
        # The grouped GEMM Triton template is supported on ROCm too. ATen
        # remains a separate autotune choice when fallback kernels are enabled.
        return True
    return torch.cuda.get_device_capability() >= (9, 0)


def _rocm_gcn_arch() -> str:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).gcnArchName.split(":", 1)[0]


def has_rocm_fp8_hardware_support() -> bool:
    if not torch.version.hip:
        return False

    # Keep this in sync with torch.testing._internal.common_cuda.PLATFORM_SUPPORTS_FP8;
    # this is the production-side equivalent used to gate Triton FP8 lowering.
    arch = _rocm_gcn_arch()
    rocm_version = tuple(int(v) for v in torch.version.hip.split(".")[:2])
    if arch.startswith("gfx94"):
        return True
    if arch.startswith("gfx120") and rocm_version >= (6, 3):
        return True
    if arch.startswith("gfx95") and rocm_version >= (6, 5):
        return True
    return False


def has_scaled_grouped_mm_triton_support(mat_a: TensorBox, mat_b: TensorBox) -> bool:
    if not torch.version.hip:
        return True
    if not has_rocm_fp8_hardware_support():
        return False

    arch = _rocm_gcn_arch()
    # Match ATen's ROCm rowwise scaled grouped GEMM contract: gfx94 uses the
    # FNUZ FP8 encoding, while newer FP8-capable arches use OCP FP8.
    expected_dtype = (
        torch.float8_e4m3fnuz if arch.startswith("gfx94") else torch.float8_e4m3fn
    )
    return mat_a.get_dtype() == expected_dtype and mat_b.get_dtype() == expected_dtype


def grouped_mm_args(
    mat1: TensorBox,
    mat2: TensorBox,
    offs: TensorBox | None,
    layout=None,
    out_dtype=None,
):
    mat1, mat2 = realize_inputs(mat1, mat2)
    if offs is not None:
        realize_inputs(offs)
    mat1_size = mat1.get_size()
    mat2_size = mat2.get_size()

    m1dim, m2dim = len(mat1_size), len(mat2_size)

    if m1dim not in (2, 3):
        raise AssertionError(f"Expected 2D or 3D mat1, got {m1dim}D")
    if m2dim not in (2, 3):
        raise AssertionError(f"Expected 2D or 3D mat2, got {m2dim}D")

    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()
        alignment = 16 // out_dtype.itemsize

        if m1dim == 2:
            if m2dim == 2:
                if offs is None:
                    raise AssertionError("offs must be provided for 2D x 2D grouped mm")
                out_size = [offs.get_size()[0], mat1_size[0], mat2_size[1]]
            else:
                out_size = [mat1_size[0], mat2_size[-1]]
        else:
            if m2dim == 2:
                out_size = [mat1_size[1], mat2_size[1]]
            else:
                out_size = [mat1_size[0], mat1_size[1], mat2_size[-1]]
        # Match the ATen extern output layout: CUDA pads grouped GEMM outputs for
        # TMA alignment, while ROCm's ATen path returns contiguous tensors.
        # TODO: Revisit whether 16-byte alignment would be beneficial for gfx1250.
        if torch.version.hip:
            if len(out_size) == 2:
                out_stride = [out_size[1], 1]
            else:
                out_stride = [out_size[1] * out_size[2], out_size[2], 1]
        else:
            size_padded = (out_size[-1] + alignment - 1) // alignment * alignment
            if len(out_size) == 2:
                out_stride = [size_padded, 1]
            else:
                out_stride = [out_size[1] * size_padded, size_padded, 1]

        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            out_size,
            out_stride,
        )
    else:
        if out_dtype is not None:
            raise AssertionError("out_dtype is ignored if layout is specified.")

    return (mat1_size, mat2_size, layout, mat1, mat2, offs)


aten__grouped_mm = ExternKernelChoice(
    torch._grouped_mm,
    "at::_grouped_mm",
    op_overload=aten._grouped_mm.default,
    has_out_variant=False,
)


aten__scaled_grouped_mm = ExternKernelChoice(
    torch._scaled_grouped_mm,
    "at::_scaled_grouped_mm",
    op_overload=aten._scaled_grouped_mm.default,
    has_out_variant=False,
)


def can_use_triton_kernel(
    mat_a: TensorBox,
    mat_b: TensorBox,
    offs: TensorBox | None,
    bias: TensorBox | None,
    scale_result: TensorBox | None,
) -> bool:
    if not has_grouped_mm_triton_support():
        return False
    if not has_triton():
        return False

    # The _grouped_mm()/_scaled_grouped_mm() operator do not support
    # bias nor scale_result yet.
    if bias is not None:
        return False
    if scale_result is not None:
        return False

    if len(mat_a.get_size()) == 2 or len(mat_b.get_size()) == 2:
        return offs is not None
    else:
        return offs is None


def create_offsets(offs_box, m1_is_2d, m2_is_2d, m, n, k, alignment):
    if m1_is_2d:
        if m2_is_2d:
            end = k
        else:
            end = m
    else:
        if m2_is_2d:
            end = n
        else:
            return None

    end_hint = V.graph.sizevars.optimization_hint(end)
    noffs_hint = V.graph.sizevars.optimization_hint(offs_box.get_size()[0])
    offs = torch.arange(1, noffs_hint + 1, dtype=torch.float32) * (
        end_hint / noffs_hint
    )
    offs[:-1] = (offs[:-1] / alignment).round() * alignment
    offs[-1] = end_hint
    return offs.to(dtype=offs_box.get_dtype(), device=offs_box.get_device())


def _tuned_grouped_mm_common(
    operator_name: str,
    algorithm_name: str,
    extern_kernel_choice: ExternKernelChoice,
    kernel_template: TritonTemplate,
    mat_a: TensorBox,
    mat_b: TensorBox,
    scale_a: TensorBox | None = None,
    scale_b: TensorBox | None = None,
    offs: TensorBox | None = None,
    bias: TensorBox | None = None,
    scale_result: TensorBox | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool | None = None,
    layout: Layout | None = None,
) -> TensorBox:
    if (scale_a is None) != (scale_b is None):
        raise AssertionError(
            "scale_a and scale_b must both be None or both be provided"
        )
    if scale_result is not None and scale_a is None:
        raise AssertionError("scale_result requires scale_a and scale_b")

    m1_size, m2_size, layout, mat_a, mat_b, offs = grouped_mm_args(
        mat_a, mat_b, offs, layout=layout, out_dtype=out_dtype
    )
    counters["aten_mm_info"][operator_name] += 1
    log_message = f"Tuned {operator_name}: mat1_shape=%s, mat2_shape=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s"
    log.info(
        log_message,
        m1_size,
        m2_size,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )

    if scale_a is not None and scale_b is not None:
        check_supported_striding(mat_a, mat_b)

    # workaround for Inductor not supporting optional tensor input arguments
    input_nodes: list[Any] = [mat_a, mat_b]
    if scale_a is not None:
        input_nodes.append(realize_inputs(scale_a))
    if scale_b is not None:
        input_nodes.append(realize_inputs(scale_b))
    if offs is not None:
        input_nodes.append(realize_inputs(offs))

    if use_fast_accum is None:
        aten_choice = extern_kernel_choice.bind(
            input_nodes,
            layout,
            out_dtype=out_dtype,
        )
    else:
        aten_choice = extern_kernel_choice.bind(
            input_nodes,
            layout,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
    if use_fast_accum is None:
        use_fast_accum = False

    choices: list[ChoiceCaller] = []
    if use_aten_gemm_kernels():
        choices.append(aten_choice)

    _, is_nonzero = _is_static_problem(layout)

    # Checking only for the equality of corresponding dims of
    # multiplicands here, relying on meta function checks for
    # everything else.
    if len(m1_size) == 2:
        if len(m2_size) == 2:
            m, k1 = m1_size
            k2, n = m2_size

            g = offs.get_size()[0]
            k = V.graph.sizevars.check_equals(k1, k2)
            a_is_2d, b_is_2d = True, True
        else:
            g1 = offs.layout.size[0]
            m, k1 = m1_size
            g2, k2, n = m2_size
            g = V.graph.sizevars.check_equals_and_simplify(g1, g2)
            k = V.graph.sizevars.check_equals(k1, k2)
            a_is_2d, b_is_2d = True, False
    else:
        if len(m2_size) == 2:
            g1 = offs.layout.size[0]
            g2, m, k1 = m1_size
            k2, n = m2_size
            g = V.graph.sizevars.check_equals_and_simplify(g1, g2)
            k = V.graph.sizevars.check_equals(k1, k2)
            a_is_2d, b_is_2d = False, True
        else:
            g1, m, k1 = m1_size
            g2, k2, n = m2_size
            g = V.graph.sizevars.check_equals_and_simplify(g1, g2)
            k = V.graph.sizevars.check_equals(k1, k2)
            a_is_2d, b_is_2d = False, False

    scaled = scale_a is not None

    if (
        is_nonzero
        and use_triton_template(layout)
        and can_use_triton_kernel(mat_a, mat_b, offs, bias, scale_result)
        and (not scaled or has_scaled_grouped_mm_triton_support(mat_a, mat_b))
    ):
        a_is_k_major = mat_a.get_stride()[-1] == 1
        b_is_k_major = mat_b.get_stride()[-2] == 1

        triton_has_make_tensor_descriptor = hasattr(tl, "make_tensor_descriptor")
        triton_has_experimental_make_tensor_descriptor = hasattr(
            tl, "_experimental_make_tensor_descriptor"
        )
        use_tma_load = (
            triton_has_make_tensor_descriptor
            or triton_has_experimental_make_tensor_descriptor
        )
        kwargs = {
            "SCALED": scaled,
            "A_IS_2D": a_is_2d,
            "B_IS_2D": b_is_2d,
            "A_IS_K_MAJOR": a_is_k_major,
            "B_IS_K_MAJOR": b_is_k_major,
            "USE_FAST_ACCUM": use_fast_accum,
            "NUM_SMS": get_num_sms(),
            "USE_TMA_LOAD": use_tma_load,
            "USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR": triton_has_experimental_make_tensor_descriptor,
        }

        for config in early_config_prune(
            g, m, mat_a.dtype.itemsize, grouped_mm_configs(), kwargs
        ):
            kernel_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=layout,
                num_stages=config.num_stages,
                num_warps=config.num_warps,
                **kwargs,
                **config.kwargs,
            )

    if use_blackwell_cutedsl_grouped_mm(
        mat_a, mat_b, layout, a_is_2d, b_is_2d, offs, bias, scale_result
    ):
        for config in get_groupgemm_configs():
            kwargs = dict(
                ACC_DTYPE="cutlass.Float32",
            )

            cutedsl_grouped_mm_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=layout,
                **kwargs,
                **asdict(config),
            )

    if (
        is_nonzero
        and a_is_2d
        and not b_is_2d
        and offs is not None
        and use_nv_universal_gemm_template(layout, m, n, k, mat_a, mat_b, offs, g)
    ):
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm import (
            add_nv_universal_grouped_gemm_choices,
        )

        add_nv_universal_grouped_gemm_choices(
            choices,
            layout,
            input_nodes,
            accumulator_type=torch.float32,
        )

    input_gen_fns = {}
    if offs is not None:
        input_offs_idx = 2 if scale_a is None else 4
        alignment = 16 // mat_a.dtype.itemsize
        input_gen_fns[input_offs_idx] = lambda x: create_offsets(
            x, a_is_2d, b_is_2d, m, n, k, alignment
        )
    node, _ = autotune_select_algorithm(
        algorithm_name, choices, input_nodes, layout, input_gen_fns=input_gen_fns
    )
    return node


@register_lowering(aten._grouped_mm.default, type_promotion_kind=None)
def tuned_grouped_mm(
    mat_a: TensorBox,
    mat_b: TensorBox,
    offs: TensorBox | None = None,
    bias: TensorBox | None = None,
    out_dtype: torch.dtype | None = None,
    layout: Layout | None = None,
) -> TensorBox:
    """Auto-tuning for _grouped_mm() operator."""

    return _tuned_grouped_mm_common(
        "aten._grouped_mm.default",
        "grouped_mm",
        aten__grouped_mm,
        triton_grouped_mm_template,
        mat_a,
        mat_b,
        None,
        None,
        offs,
        bias,
        None,
        out_dtype,
        None,
        layout,
    )


@register_lowering(aten._scaled_grouped_mm.default, type_promotion_kind=None)
def tuned_scaled_grouped_mm(
    mat_a: TensorBox,
    mat_b: TensorBox,
    scale_a: TensorBox,
    scale_b: TensorBox,
    offs: TensorBox | None = None,
    bias: TensorBox | None = None,
    scale_result: TensorBox | None = None,
    out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
    layout: Layout | None = None,
) -> TensorBox:
    """Auto-tuning for _scaled_grouped_mm() operator."""

    # matching _scaled_grouped_mm_cuda Blas.cpp implementation
    out_dtype = out_dtype or torch.bfloat16

    return _tuned_grouped_mm_common(
        "aten._scaled_grouped_mm.default",
        "scaled_grouped_mm",
        aten__scaled_grouped_mm,
        triton_scaled_grouped_mm_template,
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        offs,
        bias,
        scale_result,
        out_dtype,
        use_fast_accum,
        layout,
    )
<|endoftext|># mypy: allow-untyped-defs

import logging
from typing import TYPE_CHECKING

import torch

from .. import config as inductor_config
from ..kernel_inputs import MMKernelInputs
from ..lowering import lowerings
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    TritonTemplate,
)
from ..utils import use_aten_gemm_kernels, use_triton_template
from ..virtualized import V
from .mm_common import mm_args, mm_grid


if TYPE_CHECKING:
    from torch._inductor.ir import ChoiceCaller
    from torch._inductor.select_algorithm import KernelTemplate

log = logging.getLogger(__name__)

aten = torch.ops.aten

aten_mm_plus_mm = ExternKernelChoice(
    torch.ops.inductor._mm_plus_mm, "torch::inductor::_mm_plus_mm"
)

mm_plus_mm_template = TritonTemplate(
    name="mm_plus_mm",
    grid=mm_grid,
    debug=False,
    source=r"""
{{def_kernel("A", "B", "C", "D")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K1 = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    # K2 = {{size("C", 1)}}
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}
    stride_cm = {{stride("C", 0)}}
    stride_ck = {{stride("C", 1)}}
    stride_dk = {{stride("D", 0)}}
    stride_dn = {{stride("D", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if (((stride_am == 1 and stride_ak == M) or (stride_am == K1 and stride_ak == 1))
        and ((stride_cm == 1 and stride_ck == M) or (stride_cm == K1 and stride_ck == 1))):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M

    if (((stride_bk == 1 and stride_bn == K1) or (stride_bk == N and stride_bn == 1))
        and ((stride_dk == 1 and stride_dn == K1) or (stride_dk == N and stride_dn == 1))):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    C = C + (ram[:, None] * stride_cm + rk[None, :] * stride_ck)
    D = D + (rk[:, None] * stride_dk + rbn[None, :] * stride_dn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k1 in range(K1, 0, -BLOCK_K):
        # First matmul with A @ B
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k1, other=0.)
            b = tl.load(B, mask=rk[:, None] < k1, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    for k2 in range(K1, 0, -BLOCK_K):

        # Second matmul with C @ D
        if EVEN_K:
            c = tl.load(C)
            d = tl.load(D)
        else:
            c = tl.load(C, mask=rk[None, :] < k2, other=0.)
            d = tl.load(D, mask=rk[:, None] < k2, other=0.)
        acc += tl.dot(c, d, allow_tf32=ALLOW_TF32)
        C += BLOCK_K * stride_ck
        D += BLOCK_K * stride_dk


    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
""",
    cache_codegen_enabled_for_template=True,
)


def tuned_mm_plus_mm(mat1, mat2, mat3, mat4, *, layout=None):
    """
    Computes mm(mat1, mat2) + mm(mat3, mat4)
    """
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m1, n1, k1, layout1, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    m2, n2, _, layout2, mat3, mat4 = mm_args(mat3, mat4, layout=layout)

    # Optimization is optional, because we can always just not do the fusion
    if (
        m1 * n1 == 0
        or m2 * n2 == 0
        or not V.graph.sizevars.statically_known_list_equals(
            mat1.get_size(), mat3.get_size()
        )
        or not V.graph.sizevars.statically_known_list_equals(
            mat2.get_size(), mat4.get_size()
        )
        or inductor_config.triton.native_matmul
    ):
        # TODO(jansel): support different K values when this is fixed:
        # https://github.com/triton-lang/triton/issues/967
        return lowerings[aten.add](
            lowerings[aten.mm](mat1, mat2), lowerings[aten.mm](mat3, mat4)
        )

    # Create MMKernelInputs for MM Plus MM (matrices are at indices 0, 1 for first pair)
    # Note: This is a special case with 4 matrices, but we use the first pair for M, N, K extraction
    kernel_inputs = MMKernelInputs([mat1, mat2, mat3, mat4], mat1_idx=0, mat2_idx=1)

    if layout1 != layout2:
        raise AssertionError("layout1 and layout2 must be equal")
    # options to tune from
    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    if use_aten_gemm_kernels():
        templates_to_use.append(aten_mm_plus_mm)

    if use_triton_template(layout1, check_max_autotune=False):
        templates_to_use.append(mm_plus_mm_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, "mm_plus_mm")
    )

    node, _ = autotune_select_algorithm(
        "mm_plus_mm", choices, kernel_inputs.nodes(), layout1
    )
    return node
<|endoftext|># mypy: allow-untyped-defs
"""Lower Inductor GEMM epilogue loop IR to CuTeDSL source."""

from collections.abc import Sequence
from typing import Any

import torch
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLCSEVariable
from torch._inductor.ir import ComputedBuffer
from torch._inductor.virtualized import V
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.value_ranges import ValueRanges

from .gemm_epilogue_codegen import (
    GemmEpilogueCuteDSLKernel,
    GemmEpilogueCuteDSLOpOverrides,
)
from .loop_ir_epilogue_lowering import GemmEpilogueIRAnalysis, GemmEpilogueIRExpression


class LoopIRCuteDSLCodegen:
    """Lower scheduler loop-body expressions to a CuTeDSL fragment function."""

    def __init__(self, accumulator: str, removed_buffers: OrderedSet[str]) -> None:
        self.accumulator = accumulator
        self.removed_buffers = removed_buffers
        self.kernel = GemmEpilogueCuteDSLKernel()
        self.reads: OrderedSet[str] = OrderedSet()
        self.input_values: dict[str, CuteDSLCSEVariable] = {}
        self.accumulator_value = CuteDSLCSEVariable(
            "accum", ValueRanges.unknown(), dtype=torch.float32, shape=(1,)
        )

    def load(self, name: str, stored: Any) -> Any:
        if stored is not None:
            return self.lower(stored)
        if name == self.accumulator:
            return self.accumulator_value
        self.reads.add(name)
        if name not in self.input_values:
            dtype = V.graph.get_dtype(name)
            self.input_values[name] = CuteDSLCSEVariable(
                name, ValueRanges.unknown(), dtype=dtype, shape=(1,)
            )
        return self.input_values[name]

    def lower(self, value: Any) -> Any:
        if not isinstance(value, GemmEpilogueIRExpression):
            if isinstance(value, tuple):
                return tuple(self.lower(item) for item in value)
            return value
        if value.op == "load":
            name, _, stored = value.args
            return self.load(name, stored)
        if value.op == "identity":
            return self.lower(value.args[0])
        if value.op in ("reduction", "to_dtype_bitcast"):
            raise NotImplementedError(
                f"CuTeDSL GEMM epilogue does not support {value.op}"
            )
        lowered_args = tuple(self.lower(arg) for arg in value.args)
        lowered_kwargs = {key: self.lower(arg) for key, arg in value.kwargs}
        op = getattr(GemmEpilogueCuteDSLOpOverrides, value.op, None)
        if op is None:
            raise NotImplementedError(
                f"CuTeDSL GEMM epilogue op is not implemented: {value.op}"
            )
        return op(*lowered_args, **lowered_kwargs)

    def render(
        self, buffers: Sequence[ComputedBuffer], fn_name: str
    ) -> tuple[list[str], list[str], dict[str, str], str]:
        analysis = GemmEpilogueIRAnalysis.from_buffers(buffers)
        outputs: list[tuple[str, Any]] = []
        if self.accumulator not in self.removed_buffers:
            outputs.append((self.accumulator, self.accumulator_value))
        for name, store in analysis.stores.items():
            if name not in self.removed_buffers:
                outputs.append((name, self.lower(store.value)))
        if not outputs:
            raise NotImplementedError("CuTeDSL GEMM epilogue has no outputs")

        renames = {name: name for name in self.reads}
        result_names: list[str] = []
        for index, (buffer_name, value) in enumerate(outputs):
            result_name = "D" if index == len(outputs) - 1 else f"output{index}"
            self.kernel.body.writeline(f"{result_name} = {value}")
            renames[result_name] = buffer_name
            result_names.append(result_name)

        params = ", ".join(("accum", *self.reads))
        body = "\n".join(f"    {line}" for line in self.kernel.body.lines)
        source = (
            f"def {fn_name}({params}):\n{body}\n    return {', '.join(result_names)}"
        )
        return list(self.reads), [name for name, _ in outputs], renames, source

    @classmethod
    def from_buffers(
        cls,
        accumulator: str,
        buffers: Sequence[ComputedBuffer],
        removed_buffers: OrderedSet[str],
        fn_name: str,
    ) -> tuple[list[str], list[str], dict[str, str], str]:
        codegen = cls(accumulator, removed_buffers)
        with (
            V.set_kernel_handler(codegen.kernel),
            V.set_ops_handler(GemmEpilogueCuteDSLOpOverrides()),
        ):
            return codegen.render(buffers, fn_name)
<|endoftext|># mypy: allow-untyped-defs
"""Shared FX analysis and output planning for grouped GEMM epilogues."""

import dataclasses
from collections.abc import Sequence
from typing import Any

import torch
from torch._inductor.kernel.flex_gemm.constraints import (
    FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR,
    local_reduce_compressed_shape,
    LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR,
    LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR,
    LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_FRAGMENT_WIDTH,
    LOCAL_REDUCE_GROUPED_RESHAPE_ERROR,
    LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR,
    LOCAL_REDUCE_MATCH_NODE_ERROR,
    LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR,
    LOCAL_REDUCE_MIXED_MATCH_ERROR,
    LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR,
    LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR,
    LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR,
    local_reduce_unsupported_tensorssa_error,
    validate_local_reduce_feed_main_capability,
    validate_local_reduce_tensorssa_group_size,
)
from torch._inductor.kernel.flex_gemm.quack_reductions import (
    is_shape_preserving_pointwise_node,
    tensor_meta_shape,
)
from torch._inductor.kernel.gemm_epilogue import (
    GemmEpilogueGraph,
    GemmReductionGeometry,
    GemmReductionPlan,
    iter_fx_node_inputs,
    NormalizedGetItem,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedSqueeze,
    NormalizedUnsupportedReduction,
    NormalizedView,
)
from torch._inductor.kernel.gemm_epilogue_utils import (
    guarded_int,
    normalize_shape,
    statically_known_equal,
    statically_known_shape_equal,
)
from torch.utils._ordered_set import OrderedSet


def _is_inferred_reshape_dim(value: Any) -> bool:
    """Return whether a reshape dimension is the literal inferred-size marker."""
    return isinstance(value, int) and value == -1


def _kept_dim_matches_source(kept_size: Any, source_size: Any) -> bool:
    return _is_inferred_reshape_dim(kept_size) or statically_known_equal(
        kept_size, source_size
    )


def _guard_grouped_reshape_group(
    shape: tuple[Any, ...], source_shape: tuple[Any, ...]
) -> tuple[Any, ...]:
    """Specialize a backed group dimension used to recognize a grouped reshape."""
    if len(shape) != 3:
        return shape
    for group_index, kept_index in ((-1, 0), (-2, -1)):
        if not _kept_dim_matches_source(shape[kept_index], source_shape[kept_index]):
            continue
        group_value = shape[group_index]
        symbolic = (
            group_value.meta.get("val")
            if isinstance(group_value, torch.fx.Node)
            else group_value
        )
        if not isinstance(symbolic, torch.SymInt):
            continue
        group = guarded_int(group_value)
        if group is not None:
            result = list(shape)
            result[group_index] = group
            return tuple(result)
    return shape


def _syntactic_grouped_tensor_layout(
    shape: tuple[Any, ...],
) -> GemmReductionGeometry | None:
    """Match grouped-reshape syntax before validating source geometry."""
    if len(shape) not in (3, 4):
        return None
    if (
        isinstance(shape[-1], int)
        and shape[-1] > 0
        and _is_inferred_reshape_dim(shape[-2])
    ):
        return GemmReductionGeometry(group=shape[-1], axis=1)
    if (
        _is_inferred_reshape_dim(shape[-3])
        and isinstance(shape[-2], int)
        and shape[-2] > 0
    ):
        return GemmReductionGeometry(group=shape[-2], axis=0)
    return None


def _group_count_matches_selected_dim(
    group_count: Any,
    selected_size: Any,
    group: int,
    kept_size: Any,
) -> bool:
    """Match a group count, allowing -1 to infer the selected source dimension."""
    if _is_inferred_reshape_dim(group_count):
        return True
    return statically_known_equal(group_count * group, selected_size) or (
        not _is_inferred_reshape_dim(kept_size)
        and statically_known_equal(group_count, selected_size // group)
    )


def _grouped_layout_matches_source_shape(
    shape: tuple[Any, ...],
    source_shape: tuple[Any, ...],
    layout: GemmReductionGeometry,
) -> bool:
    """Require a 2-D GEMM output reshape to split exactly M or N."""
    if len(shape) != 3:
        return False

    m, n = source_shape
    match layout.axis, shape:
        case 1, (kept_m, group_count, group) if group == layout.group:
            return _kept_dim_matches_source(
                kept_m, m
            ) and _group_count_matches_selected_dim(group_count, n, group, kept_m)
        case 0, (group_count, group, kept_n) if group == layout.group:
            return _kept_dim_matches_source(
                kept_n, n
            ) and _group_count_matches_selected_dim(group_count, m, group, kept_n)
        case _:
            return False


def grouped_tensor_layout(
    shape: Any, source_shape: Any | None = None
) -> GemmReductionGeometry | None:
    """Recognize grouped M/N geometry, specializing backed group dimensions."""
    shape = normalize_shape(shape)
    if not isinstance(shape, tuple):
        return None
    if len(shape) == 1 and isinstance(shape[0], (list, tuple, torch.Size)):
        shape = normalize_shape(shape[0])
    if source_shape is not None:
        source_shape = normalize_shape(source_shape)
        if isinstance(source_shape, tuple) and len(source_shape) == 2:
            shape = _guard_grouped_reshape_group(shape, source_shape)
            candidates = []
            match shape:
                case (*_, int(group)) if group > 0:
                    candidates.append(GemmReductionGeometry(group=group, axis=1))
            match shape:
                case (*_, int(group), _) if group > 0:
                    candidates.append(GemmReductionGeometry(group=group, axis=0))
            for layout in candidates:
                if _grouped_layout_matches_source_shape(shape, source_shape, layout):
                    return layout
            if _syntactic_grouped_tensor_layout(shape) is not None:
                raise NotImplementedError(LOCAL_REDUCE_GROUPED_RESHAPE_ERROR)
            return None
    return _syntactic_grouped_tensor_layout(shape)


FEED_MAIN_BINARY_FUNCTIONS = frozenset(
    (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.add.Scalar,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.mul.Scalar,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.sub.Scalar,
    )
)


@dataclasses.dataclass(frozen=True)
class GemmLocalReduceMatch:
    """Describe a supported grouped local-reduction value found in the FX graph.

    Attributes:
        value_node: FX node that produces the matched local-reduction value.
        geometry: Group size and GEMM output axis reduced by the value.
    """

    value_node: torch.fx.Node
    geometry: GemmReductionGeometry
    reduction_node: torch.fx.Node | None = None
    reduction_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.value_node, torch.fx.Node):
            raise RuntimeError(LOCAL_REDUCE_MATCH_NODE_ERROR)
        if self.reduction_node is None:
            object.__setattr__(self, "reduction_node", self.value_node)

    def to_plan(
        self,
        *,
        store: "GemmLocalReduceStore | None",
        feeds_main: bool,
    ) -> "GemmOutputLocalReducePlan":
        """Bind this matched value to its output consumers."""
        return GemmOutputLocalReducePlan(self, store=store, feeds_main=feeds_main)

    @classmethod
    def common(
        cls,
        matches: list["GemmLocalReduceMatch"],
        mixed_match_error: str,
    ) -> "GemmLocalReduceMatch | None":
        """Return the common match when all values use one reduction geometry."""
        if not matches:
            return None
        match = matches[0]
        if any(item.geometry != match.geometry for item in matches):
            raise NotImplementedError(mixed_match_error)
        return match

    @classmethod
    def common_value(
        cls,
        matches: list["GemmLocalReduceMatch"],
        mixed_match_error: str,
    ) -> "GemmLocalReduceMatch | None":
        """Return the common match when all consumers use one physical value."""
        match = cls.common(matches, mixed_match_error)
        if match is None:
            return None
        if any(item.value_node is not match.value_node for item in matches):
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        return match


@dataclasses.dataclass(frozen=True)
class GemmLocalReduceStore:
    """Describe where a compressed local reduction appears in graph outputs.

    Attributes:
        node: FX node returned as the compressed local-reduction output.
        aux_index: Position of that node among the graph's auxiliary outputs.
    """

    node: torch.fx.Node
    aux_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.node, torch.fx.Node) or self.aux_index < 0:
            raise RuntimeError(LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR)


@dataclasses.dataclass(frozen=True)
class GemmOutputLocalReducePlan:
    """Bind a matched local reduction to store and/or main-output consumers.

    Attributes:
        match: Supported local-reduction value identified during FX analysis.
        store: Compressed auxiliary output receiving the value, when requested.
        feeds_main: Whether the reduced value is also consumed by the main output.
    """

    match: GemmLocalReduceMatch
    store: GemmLocalReduceStore | None = None
    feeds_main: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.match, GemmLocalReduceMatch) or (
            self.store is None and not self.feeds_main
        ):
            raise RuntimeError(LOCAL_REDUCE_OUTPUT_PLAN_NODE_ERROR)

    @property
    def needs_physical_callbacks(self) -> bool:
        return self.match.geometry.needs_physical_callbacks


@dataclasses.dataclass(frozen=True)
class GemmOutputPlan:
    """Classify the values returned by a FlexGEMM body.

    Attributes:
        output: FX node returned as the main GEMM result.
        aux_outputs: Same-shape auxiliary FX outputs returned after the main result.
        local_reduce: Compressed or feed-main local-reduction output behavior.
    """

    output: torch.fx.Node
    aux_outputs: tuple[torch.fx.Node, ...] = ()
    local_reduce: GemmOutputLocalReducePlan | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output, torch.fx.Node) or not all(
            isinstance(aux_output, torch.fx.Node) for aux_output in self.aux_outputs
        ):
            raise RuntimeError(FLEX_GEMM_OUTPUT_PLAN_NODE_ERROR)

    @property
    def reduction_plan(self) -> GemmReductionPlan | None:
        """Finalize FX ownership metadata into the shared reduction contract."""
        local_reduce = self.local_reduce
        if local_reduce is None:
            return None
        match = local_reduce.match
        reduction_type = match.reduction_type
        if reduction_type is None:
            return None
        reduction_output = (
            local_reduce.store.node.name if local_reduce.store is not None else None
        )
        return GemmReductionPlan(
            reduction_output,
            match.geometry.group,
            match.geometry.axis,
            reduction_type,
            "identity",
            self.output.name,
            feeds_main=local_reduce.feeds_main,
            feed_output=self.output.name if local_reduce.feeds_main else None,
        )


@dataclasses.dataclass
class GemmLocalReduceAnalysis:
    """Collect grouped TensorSSA layouts and supported local-reduction matches.

    ``from_graph_module`` visits the FX graph in topological order. See
    ``GemmReductionGeometry`` for the grouped layout attached to reshape and
    pointwise nodes, and ``GemmLocalReduceMatch`` for each supported reduced
    value found from those layouts.

    Attributes:
        graph: Dependency index used by recursive feed-main matching.
        grouped_tensors: FX nodes whose values carry a grouped TensorSSA layout.
        matches: FX values matched to a supported grouped local reduction.
    """

    graph: GemmEpilogueGraph
    grouped_tensors: dict[torch.fx.Node, GemmReductionGeometry] = dataclasses.field(
        default_factory=dict
    )
    matches: dict[torch.fx.Node, GemmLocalReduceMatch] = dataclasses.field(
        default_factory=dict
    )

    @classmethod
    def from_graph_module(
        cls, graph_module: torch.fx.GraphModule
    ) -> "GemmLocalReduceAnalysis":
        """Build shared dependency and reduction state in one topological pass."""
        nodes = tuple(graph_module.graph.nodes)
        analysis = cls(GemmEpilogueGraph.from_nodes(nodes))
        for node in nodes:
            if node.op == "output":
                break
            analysis.visit_node(node)
        return analysis

    def visit_node(self, node: torch.fx.Node) -> None:
        """Record grouped layouts and local-reduction matches for one FX node."""
        if node.op != "call_function":
            return
        normalized = self.graph.normalized_nodes.get(node)
        if isinstance(normalized, NormalizedView):
            if self.propagate_local_reduce_match(node, normalized.source):
                return
            if self.bind_grouped_layout(node, normalized.shape, normalized.source):
                return
        elif isinstance(normalized, NormalizedReduction):
            if self.bind_grouped_reduction(
                node,
                normalized.source,
                normalized.dim,
                normalized.dtype,
                reduction_type=normalized.reduction_type,
            ):
                return
        elif isinstance(normalized, NormalizedPrepareSoftmax):
            if self.bind_grouped_reduction(
                node,
                normalized.source,
                normalized.dim,
                raise_invalid_dims=False,
            ):
                return
        elif isinstance(normalized, NormalizedUnsupportedReduction):
            if normalized.source in self.grouped_tensors:
                raise local_reduce_unsupported_tensorssa_error(normalized.target)
        elif isinstance(normalized, (NormalizedSqueeze, NormalizedGetItem)):
            if self.propagate_local_reduce_match(node, normalized.source):
                return
        if is_shape_preserving_pointwise_node(node):
            self.propagate_pointwise_match(node, LOCAL_REDUCE_MIXED_MATCH_ERROR)

    def bind_grouped_layout(self, node: torch.fx.Node, shape: Any, source: Any) -> bool:
        """Attach a grouped TensorSSA layout introduced by a reshape."""
        source_shape = (
            tensor_meta_shape(source) if isinstance(source, torch.fx.Node) else None
        )
        grouped_layout = grouped_tensor_layout(shape, source_shape)
        if grouped_layout is None or not isinstance(source, torch.fx.Node):
            return False
        self.grouped_tensors[node] = grouped_layout
        return True

    def propagate_local_reduce_match(self, node: torch.fx.Node, source: Any) -> bool:
        """Copy a matched local-reduction value through an FX wrapper."""
        if not isinstance(source, torch.fx.Node):
            return False
        match = self.matches.get(source)
        if match is None:
            return False
        self.matches[node] = match
        return True

    def bind_grouped_reduction(
        self,
        node: torch.fx.Node,
        input_node: Any,
        dim: Any,
        dtype: Any = None,
        *,
        reduction_type: str | None = None,
        raise_invalid_dims: bool = True,
    ) -> bool:
        """Match and record a reduction over a grouped TensorSSA layout."""
        if not isinstance(input_node, torch.fx.Node):
            return False
        layout = self.grouped_tensors.get(input_node)
        if layout is None:
            return False
        if dtype is not None:
            raise NotImplementedError(LOCAL_REDUCE_EXPLICIT_DTYPE_ERROR)
        validate_local_reduce_tensorssa_group_size(layout.axis, layout.group)
        if not layout.matches_reduction_dim(dim):
            if not raise_invalid_dims:
                return False
            raise NotImplementedError(LOCAL_REDUCE_INNERMOST_GROUPED_DIM_ERROR)
        self.matches[node] = GemmLocalReduceMatch(
            node,
            layout,
            reduction_node=node,
            reduction_type=reduction_type,
        )
        return True

    def has_physical_grouped_input(self, value: Any) -> bool:
        """Return whether a value depends on a grouped layout needing callbacks."""
        active_geometries = OrderedSet(
            match.geometry for match in self.matches.values()
        )
        physical_grouped_nodes = OrderedSet(
            node
            for node, layout in self.grouped_tensors.items()
            if layout.needs_physical_callbacks and layout in active_geometries
        )
        return any(
            node in physical_grouped_nodes
            or any(
                dependency in physical_grouped_nodes
                for dependency in self.graph.dependencies.get(node, ())
            )
            for node in iter_fx_node_inputs(value)
        )

    def propagate_pointwise_match(
        self, node: torch.fx.Node, mixed_match_error: str
    ) -> bool:
        """Propagate grouped layouts and local-reduction matches through pointwise ops."""
        grouped_layouts = [
            self.grouped_tensors[arg]
            for arg in iter_fx_node_inputs((node.args, node.kwargs))
            if arg in self.grouped_tensors
        ]
        if grouped_layouts:
            grouped_layout = grouped_layouts[0]
            if any(layout != grouped_layout for layout in grouped_layouts):
                raise NotImplementedError(LOCAL_REDUCE_MIXED_GROUPED_LAYOUT_ERROR)
            self.grouped_tensors[node] = grouped_layout
        match = GemmLocalReduceMatch.common(
            [
                self.matches[arg]
                for arg in iter_fx_node_inputs((node.args, node.kwargs))
                if arg in self.matches
            ],
            mixed_match_error,
        )
        if match is None:
            return False
        self.matches[node] = dataclasses.replace(match, value_node=node)
        return True

    def match_feed_value(
        self,
        value: Any,
        grouped_source: torch.fx.Node,
        layout: GemmReductionGeometry,
    ) -> GemmLocalReduceMatch | None:
        """Find the grouped reduction that produces a broadcast value."""
        if not isinstance(value, torch.fx.Node):
            return None
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            if normalized.source is not grouped_source:
                if self.graph.depends_on(normalized.source, grouped_source):
                    raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            if (
                normalized.dtype is not None
                or not normalized.keepdim
                or not layout.matches_reduction_dim(normalized.dim)
            ):
                raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
            return GemmLocalReduceMatch(
                value,
                layout,
                reduction_node=value,
                reduction_type=normalized.reduction_type,
            )
        if not is_shape_preserving_pointwise_node(value):
            return None
        matches = [
            match
            for arg in iter_fx_node_inputs((value.args, value.kwargs))
            if (match := self.match_feed_value(arg, grouped_source, layout)) is not None
        ]
        return GemmLocalReduceMatch.common_value(
            matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
        )

    def validate_hidden_feed_main_reduction_input(
        self,
        input_node: Any,
        grouped_source: torch.fx.Node,
    ) -> None:
        """Reject reduction inputs that would need another physical feed-main value."""
        if input_node is grouped_source:
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)
        if not isinstance(input_node, torch.fx.Node):
            return
        if self.graph.depends_on(input_node, grouped_source):
            raise NotImplementedError(LOCAL_REDUCE_SOURCE_EXPRESSION_ERROR)
        if self.has_physical_grouped_input(input_node):
            raise NotImplementedError(LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR)

    def validate_feed_main_source_reductions(
        self,
        value: Any,
        grouped_source: torch.fx.Node,
        selected_reduction: torch.fx.Node,
        seen: OrderedSet[torch.fx.Node] | None = None,
    ) -> None:
        """Reject hidden physical reductions outside the selected feed-main value."""
        if not isinstance(value, torch.fx.Node):
            for arg in iter_fx_node_inputs(value):
                self.validate_feed_main_source_reductions(
                    arg, grouped_source, selected_reduction, seen
                )
            return
        if value is selected_reduction:
            return
        if seen is None:
            seen = OrderedSet()
        if value in seen:
            return
        seen.add(value)
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            self.validate_hidden_feed_main_reduction_input(
                normalized.source, grouped_source
            )
        for arg in iter_fx_node_inputs((value.args, value.kwargs)):
            self.validate_feed_main_source_reductions(
                arg, grouped_source, selected_reduction, seen
            )

    def validate_feed_main_source_match(
        self,
        source: torch.fx.Node,
        match: GemmLocalReduceMatch | None,
    ) -> GemmLocalReduceMatch | None:
        """Preserve the one-physical-value ABI across recursive source matching."""
        if match is None:
            return None
        normalized = self.graph.normalized_nodes.get(match.value_node)
        if isinstance(normalized, NormalizedReduction):
            self.validate_feed_main_source_reductions(
                source, normalized.source, match.value_node
            )
        return match

    @staticmethod
    def feed_main_binary_candidates(
        source: torch.fx.Node,
    ) -> tuple[tuple[Any, Any], ...]:
        """Return operand orderings for supported binary feed-main expressions."""
        if (
            len(source.args) < 2
            or source.op != "call_function"
            or source.target not in FEED_MAIN_BINARY_FUNCTIONS
        ):
            return ()
        lhs, rhs = source.args[:2]
        return ((lhs, rhs), (rhs, lhs))

    def feed_main_grouped_reduction(
        self,
        value: Any,
        grouped_source: torch.fx.Node,
        layout: GemmReductionGeometry,
    ) -> bool:
        """Return whether a candidate contains a grouped feed-main reduction."""
        if not isinstance(value, torch.fx.Node):
            return False
        normalized = self.graph.normalized_nodes.get(value)
        if isinstance(normalized, NormalizedReduction):
            return (
                normalized.dtype is None
                and bool(normalized.keepdim)
                and layout.matches_reduction_dim(normalized.dim)
                and (
                    normalized.source is grouped_source
                    or self.graph.depends_on(normalized.source, grouped_source)
                )
            )
        if not is_shape_preserving_pointwise_node(value):
            return False
        return any(
            self.feed_main_grouped_reduction(arg, grouped_source, layout)
            for arg in iter_fx_node_inputs((value.args, value.kwargs))
        )

    def match_feed_main_candidate(
        self,
        grouped_source: Any,
        value: Any,
        output_meta: Any,
    ) -> GemmLocalReduceMatch | None:
        """Match one grouped-source and reduced-value operand ordering."""
        if not isinstance(grouped_source, torch.fx.Node) or not isinstance(
            value, torch.fx.Node
        ):
            return None
        normalized = self.graph.normalized_nodes.get(grouped_source)
        if not isinstance(normalized, NormalizedView):
            return None
        source_node = normalized.source
        layout = self.grouped_tensors.get(grouped_source)
        if layout is None:
            return None
        if layout.axis != 0:
            if not self.feed_main_grouped_reduction(value, grouped_source, layout):
                return None
            if layout.group <= LOCAL_REDUCE_FRAGMENT_WIDTH:
                # Intentional fallthrough: axis-1 feeds within one TensorSSA
                # fragment lower as plain generated TensorSSA without a feed plan.
                return None
            raise NotImplementedError(LOCAL_REDUCE_FEED_MAIN_AXIS1_FRAGMENT_ERROR)
        validate_local_reduce_feed_main_capability(layout.axis, layout.group)
        source_meta = source_node.meta.get("val")
        if (
            output_meta is not None
            and source_meta is not None
            and not statically_known_shape_equal(output_meta.shape, source_meta.shape)
        ):
            return None
        return self.match_feed_value(value, grouped_source, layout)

    def match_feed_main_source(
        self,
        source: torch.fx.Node,
        output_meta: Any,
    ) -> GemmLocalReduceMatch | None:
        """Find one physical feed-main value inside a pointwise expression."""
        matches = [
            match
            for grouped_source, value in self.feed_main_binary_candidates(source)
            if (
                match := self.match_feed_main_candidate(
                    grouped_source, value, output_meta
                )
            )
            is not None
        ]
        if matches:
            return self.validate_feed_main_source_match(
                source,
                GemmLocalReduceMatch.common_value(
                    matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
                ),
            )
        if not is_shape_preserving_pointwise_node(source):
            return None
        matches = [
            match
            for arg in iter_fx_node_inputs((source.args, source.kwargs))
            if isinstance(arg, torch.fx.Node)
            if (match := self.match_feed_main_source(arg, output_meta)) is not None
        ]
        return self.validate_feed_main_source_match(
            source,
            GemmLocalReduceMatch.common_value(
                matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
            ),
        )

    def feed_main_plan(
        self,
        output: torch.fx.Node,
    ) -> GemmLocalReduceMatch | None:
        """Match feed-main reductions through trailing pointwise nodes."""
        normalized = self.graph.normalized_nodes.get(output)
        if isinstance(normalized, NormalizedView):
            return self.match_feed_main_source(
                normalized.source, output.meta.get("val")
            )
        if not is_shape_preserving_pointwise_node(output):
            return None
        matches = [
            match
            for arg in iter_fx_node_inputs((output.args, output.kwargs))
            if isinstance(arg, torch.fx.Node)
            if (match := self.feed_main_plan(arg)) is not None
        ]
        return self.validate_feed_main_source_match(
            output,
            GemmLocalReduceMatch.common_value(
                matches, LOCAL_REDUCE_ONE_PHYSICAL_VALUE_ERROR
            ),
        )

    def common_feed_main_match(
        self,
        candidates: tuple[Any, ...],
    ) -> GemmLocalReduceMatch | None:
        """Find the physical reduction value shared by feed-main consumers."""
        matches = [
            match
            for candidate in candidates
            if isinstance(candidate, torch.fx.Node)
            if (match := self.feed_main_plan(candidate)) is not None
        ]
        return GemmLocalReduceMatch.common_value(
            matches, LOCAL_REDUCE_FEED_MAIN_MIXED_MATCH_ERROR
        )

    def common_reduction_dependency_match(
        self, outputs: Sequence[torch.fx.Node]
    ) -> GemmLocalReduceMatch | None:
        """Return one physical reduction transitively consumed by the outputs."""
        reachable = OrderedSet(outputs)
        for output in outputs:
            reachable.update(self.graph.dependencies.get(output, ()))
        matches = [self.matches[node] for node in reachable if node in self.matches]
        if not matches:
            return None
        reduction_node = matches[0].reduction_node
        if reduction_node is None or any(
            match.reduction_node is not reduction_node for match in matches
        ):
            return None
        return matches[0]

    def compressed_aux_plan(
        self,
        output: Any,
        aux: torch.fx.Node,
        aux_index: int,
    ) -> GemmOutputLocalReducePlan | None:
        """Plan a matched local reduction returned in compressed output shape."""
        match = self.matches.get(aux)
        output_meta = (
            output.meta.get("val") if isinstance(output, torch.fx.Node) else None
        )
        aux_meta = aux.meta.get("val")
        if match is None or aux_meta is None or output_meta is None:
            return None
        expected_aux_shape = local_reduce_compressed_shape(
            output_meta.shape, match.geometry.group, match.geometry.axis
        )
        if not statically_known_shape_equal(expected_aux_shape, aux_meta.shape):
            return None
        return match.to_plan(
            store=GemmLocalReduceStore(aux, aux_index), feeds_main=False
        )

    def feed_main_output_plan(
        self,
        output: torch.fx.Node,
        aux_outputs: tuple[torch.fx.Node, ...] = (),
        *,
        allow_dependency_match: bool = False,
    ) -> GemmOutputPlan | None:
        """Plan one physical reduction value consumed by the main output."""
        match = self.common_feed_main_match((output, *aux_outputs))
        if match is None and allow_dependency_match:
            match = self.common_reduction_dependency_match((output, *aux_outputs))
        if match is None:
            return None
        return GemmOutputPlan(
            output,
            aux_outputs,
            match.to_plan(store=None, feeds_main=True),
        )
<|endoftext|># mypy: allow-untyped-defs
import functools
import logging
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch._inductor.autoheuristic.autoheuristic import AutoHeuristicSelectAlgorithm
from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    context_add_strides,
    context_add_using_tf32,
    mm_operations,
)
from torch._inductor.codegen.cpp_gemm_template import CppGemmTemplate
from torch._inductor.remote_gemm_autotune_cache import gen_best_config
from torch._inductor.virtualized import ops, V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.functional import ScalingType  # type: ignore[attr-defined]
from torch.torch_version import TorchVersion
from torch.utils._ordered_set import OrderedSet

from .. import config as inductor_config, distributed_autotune, lowering as L
from ..codegen.cutlass.gemm_template import CUTLASS2xGemmTemplate, CUTLASS3xGemmTemplate
from ..codegen.rocm.ck_tile_universal_gemm_template import CKTileGemmTemplate
from ..codegen.rocm.ck_universal_gemm_template import CKGemmTemplate
from ..codegen.subgraph import SubgraphChoiceCaller, SubgraphTemplate
from ..ir import Buffer, ChoiceCaller, is_triton, Layout
from ..kernel_inputs import MMKernelInputs
from ..lowering import (
    fallback_handler,
    lowerings,
    make_pointwise,
    make_reduction,
    register_lowering,
    transform_args,
)
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    KernelTemplate,
    realize_inputs,
    TritonTemplate,
)
from ..utils import (
    _use_cutlass_for_op,
    ceildiv,
    use_aten_gemm_kernels,
    use_ck_gemm_template,
    use_ck_tile_gemm_template,
    use_cpp_gemm_template,
    use_cutlass_template,
    use_decompose_k_choice,
    use_nv_universal_gemm_template,
    use_triton_blackwell_tma_template,
    use_triton_scaling_template,
    use_triton_template,
    use_triton_tma_template,
)
from .mm_common import (
    _is_static_problem,
    _use_small_mm_pointwise,
    load_kernel_template,
    mm_args,
    mm_grid,
    persistent_mm_grid,
    use_native_matmul,
)


try:
    import triton

    triton_version = TorchVersion(triton.__version__)
    has_triton = True
except ImportError:
    triton_version = TorchVersion("0.0.0")
    has_triton = False

log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims

# We define each template kernel in a separate file which is the name of the input to load_kernel_template
# (e.g. triton_mm for templates/triton_mm.py.jinja).
# If you are adding a new template, please follow that pattern and add a new file with your implementation in the templates folder.
mm_template = TritonTemplate(
    name="mm",
    grid=mm_grid,
    source=load_kernel_template("triton_mm")
    if (torch.version.hip is None) or triton_version >= "3.3.0"
    # FIXME: To get around rocm failures.
    # The only difference between the two templates is M >= BLOCK_M and N >= BLOCK_N checking.
    # See more details in https://github.com/pytorch/pytorch/pull/146293
    else load_kernel_template("triton_mm_rocm"),
    cache_codegen_enabled_for_template=True,
    prologue_loads_all_inputs=True,
)

persistent_tma_mm_template = TritonTemplate(
    name="mm_persistent_tma",
    grid=persistent_mm_grid,
    source=load_kernel_template("triton_persistent_tma_mm"),
)

# Non-TMA Triton template for persistent MM
# used on AMD
persistent_mm_template = TritonTemplate(
    name="mm_persistent",
    grid=persistent_mm_grid,
    source=load_kernel_template("triton_persistent_mm"),
)


scaled_mm_device_tma_epilogue_scaling_template = TritonTemplate(
    name="scaled_mm_device_tma_epilogue_scaling",
    grid=persistent_mm_grid,
    source=load_kernel_template("triton_epilogue_scaled_mm"),
)


scaled_mm_device_tma_main_loop_scaling_template = TritonTemplate(
    name="scaled_mm_device_tma_main_loop_scaling",
    grid=persistent_mm_grid,
    source=load_kernel_template("triton_main_loop_scaled_mm"),
)

blackwell_ws_persistent_device_tma_mm_template = TritonTemplate(
    name="blackwell_ws_persistent_device_tma",
    grid=persistent_mm_grid,
    source=load_kernel_template("triton_blackwell_ws_persistent_device_tma_mm"),
)


# prevent duplication registration of extern functions
@functools.cache
def lazy_register_extern_choice(fn):
    return ExternKernelChoice(fn)


aten_mm = ExternKernelChoice(torch.mm, "at::mm_out", op_overload=aten.mm.out)
aten_mm_dtype = ExternKernelChoice(
    torch.mm,
    "at::mm_dtype_out",
    name="mm_dtype",
    op_overload=aten.mm.dtype_out,
)

aten_addmm = ExternKernelChoice(
    torch.addmm, "at::addmm_out", op_overload=aten.addmm.out
)

aten__int_mm = ExternKernelChoice(
    torch._int_mm, "at::_int_mm_out", op_overload=aten._int_mm.out
)

aten__sparse_semi_structured_mm = ExternKernelChoice(
    torch._sparse_semi_structured_mm,
    "at::_sparse_semi_structured_mm",
    has_out_variant=False,
    op_overload=aten._sparse_semi_structured_mm.default,
)

aten__fp8_mm = ExternKernelChoice(
    torch._scaled_mm, "at::_scaled_mm_out", op_overload=aten._scaled_mm.out
)


def _is_int8_mat(mat):
    return mat.get_dtype() in (torch.int8, torch.uint8)


def bias_addmm(inp, mat1, mat2, *, out=None, alpha=1, beta=1):
    """
    Giving torch.addmm a 1D tensor calls a different (faster) cublasLt
    kernel under the hood.  There are a few shapes where this is slower,
    but they are rare.
    """
    if (inp.stride(0) == 0 and inp.size(0) != 0) or inp.size(0) == 1:
        return torch.addmm(inp[0], mat1, mat2, out=out, alpha=alpha, beta=beta)
    return torch.addmm(inp, mat1, mat2, out=out, alpha=alpha, beta=beta)


def check_supported_striding(mat_a, mat_b) -> None:
    def is_row_major(stride) -> bool:
        return V.graph.sizevars.statically_known_equals(stride[1], 1)

    def is_col_major(stride) -> bool:
        return V.graph.sizevars.statically_known_equals(stride[0], 1)

    def has_zero_dim(size) -> bool:
        return bool(
            V.graph.sizevars.statically_known_equals(size[0], 0)
            or V.graph.sizevars.statically_known_equals(size[1], 0)
        )

    # Check mat_a (self) stride requirements
    torch._check(
        is_row_major(mat_a.get_stride()) or has_zero_dim(mat_a.get_size()),
        lambda: f"mat_a must be row_major, got stride {mat_a.get_stride()}",
    )

    # Check mat_b stride requirements
    torch._check(
        is_col_major(mat_b.get_stride()) or has_zero_dim(mat_b.get_size()),
        lambda: f"mat_b must be col_major, got stride {mat_b.get_stride()}",
    )


aten_bias_addmm = ExternKernelChoice(bias_addmm, None)


def _check_addmm_input_metadata(inp, mat1, mat2) -> None:
    torch._check(
        inp.get_dtype() == mat1.get_dtype() and inp.get_dtype() == mat2.get_dtype(),
        lambda: "input dtypes must be the same",
    )
    torch._check(
        inp.get_device() == mat1.get_device() and inp.get_device() == mat2.get_device(),
        lambda: "all inputs must be on the same device",
    )


def decomposeK(a, b, k_splits):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    k_parts = k // k_splits
    B = k_splits
    a_reshaped = torch.permute(a.reshape(m, B, k_parts), (1, 0, 2))
    b_reshaped = b.reshape(B, k_parts, n)
    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    reduced_buf = torch.sum(result, 0)
    return reduced_buf.to(a.dtype)


class DecomposeKSugraphTemplate(SubgraphTemplate):
    def __init__(self):
        super().__init__(
            name="decompose_k",
        )

    def generate(  # type: ignore[override]
        self,
        input_nodes: list[Buffer],
        layout: Layout,
        k_split: int,
    ) -> SubgraphChoiceCaller:
        from torch._dispatch.python import enable_python_dispatcher

        from ..decomposition import select_decomp_table

        name = f"decompose_k_mm_{k_split}_split"
        description = f"{k_split=}"

        with enable_python_dispatcher():
            decompositions = select_decomp_table()
            fn = make_fx(
                functools.partial(decomposeK, k_splits=k_split),
                decompositions,
            )

            return super().generate(
                name=name,
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=fn,
                description=description,
            )


decompose_k_subgraph_template = DecomposeKSugraphTemplate()


class ContiguousTemplate(SubgraphTemplate):
    def __init__(self, name: str, description: str, fn: Any):
        self.name = name
        self.description = description
        self.fn = fn
        super().__init__(
            name=name,
        )

    def generate(  # type: ignore[override]
        self,
        input_nodes: list[Buffer],
        layout: Layout,
    ) -> SubgraphChoiceCaller:
        from torch._dispatch.python import enable_python_dispatcher

        from ..decomposition import select_decomp_table

        with enable_python_dispatcher():
            decompositions = select_decomp_table()
            fn = make_fx(
                self.fn,
                decompositions,
            )

            return super().generate(
                name=self.name,
                input_nodes=input_nodes,
                layout=layout,
                make_fx_graph=fn,
                description=self.description,
            )


def contiguous_mm(a, b):
    return torch.mm(a, b.contiguous())


def contiguous_addmm(inp, a, b):
    return torch.addmm(inp, a, b.contiguous())


mm_contiguous_subgraph_template = ContiguousTemplate(
    "contiguous_mm", "contiguous mm", contiguous_mm
)
addmm_contiguous_subgraph_template = ContiguousTemplate(
    "contiguous_addmm", "contiguous addmm", contiguous_addmm
)


@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, out_dtype=None, *, layout=None):
    """
    Lowering for autotuning aten.mm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    if out_dtype is not None:
        input_dtype = mat1.get_dtype()
        torch._check(
            mat2.get_dtype() == input_dtype,
            lambda: "input dtypes must be the same",
        )
        torch._check(
            mat1.get_device().type in ("cuda", "xpu"),
            lambda: "out_dtype is only supported for CUDA or XPU",
        )
        torch._check(
            out_dtype == input_dtype
            or (
                out_dtype == torch.float32
                and input_dtype in (torch.float16, torch.bfloat16)
            ),
            lambda: "out_dtype must be the same as input dtype or fp32 for fp16/bf16 inputs",
        )

    # Lower matmul-related operations (e.g., torch.matmul / torch.bmm / torch.addmm)
    # into native matmul IR using `ops.dot`. When we see a matmul pattern
    # (C[y, x] = A[y, r] * B[r, x]), the core idea is to emulate a broadcasted
    # multiply followed by a sum.
    #
    # For example, given `C = torch.matmul(A, B)`, this can be rewritten as:
    #
    #     Prod = A.unsqueeze(-1) * B.unsqueeze(0)
    #     C = Prod.sum(dim=1)
    #
    # Instead of explicitly using `ops.mul` and `ops.reduction("sum")`, we lower
    # these into `ops.dot` (pointwise) and `ops.reduction("dot")`. These IR nodes
    # are semantically equivalent to the `ops.mul` + `ops.reduction("sum")`
    # combination, but are lowered to `tl.dot` during the code generation phase.
    if use_native_matmul(mat1, mat2):
        mat1 = lowerings[aten.unsqueeze](mat1, -1)
        mat2 = lowerings[aten.unsqueeze](mat2, 0)
        args, kwargs = transform_args(
            args=[mat1, mat2],
            kwargs={},
            broadcast=True,
            type_promotion_kind=None,
            convert_input_to_bool=False,
        )  # Handles broadcasting the arguments

        if inductor_config.triton.codegen_upcast_to_fp32 and mat1.dtype in [
            torch.float16,
            torch.bfloat16,
        ]:

            def _to_dtype(x):
                return ops.to_dtype(x, mat1.dtype, use_compute_types=False)

            args = [make_pointwise(_to_dtype)(x) for x in args]
        mul_pointwise = make_pointwise(ops.dot)(*args)
        dot_reduction = make_reduction("dot")(mul_pointwise, 1)

        return dot_reduction

    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=out_dtype
    )

    if out_dtype is None and _use_small_mm_pointwise(m, k, n, layout):
        counters["inductor"]["decompose_mm_pointwise"] += 1
        mat1 = L.unsqueeze(mat1, -1)
        mat2 = L.unsqueeze(mat2, 0)
        return L.sum_(L.mul(mat1, mat2), axis=1)

    static_shape, is_nonzero = _is_static_problem(layout)
    name = "mm"

    # Create MMKernelInputs for standard MM at the top
    kernel_inputs = MMKernelInputs([mat1, mat2], out_dtype=out_dtype)

    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten.mm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.mm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    choices: list[ChoiceCaller] = []
    static_shape, is_nonzero = _is_static_problem(layout)

    aten_handler: ExternKernelChoice = aten_mm
    aten_extra_kwargs: dict[str, Any] = {}
    if out_dtype is not None:
        aten_handler = aten_mm_dtype
        aten_extra_kwargs = {"out_dtype": out_dtype}

    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    kwarg_overrides: dict[str, dict[str, Any]] = {}
    if use_aten_gemm_kernels():
        templates_to_use.append(aten_handler)
        if aten_extra_kwargs:
            kwarg_overrides[aten_handler.uid] = aten_extra_kwargs

    if (
        out_dtype is None
        and is_nonzero
        and use_triton_template(layout, check_max_autotune=True)
    ):
        if use_decompose_k_choice(m, n, k):
            templates_to_use.append(decompose_k_subgraph_template)
        # Triton Templates typically perform very poorly for large K.
        # Its highly unlikely that if we want to use decompose_k, then
        # Triton will ever win.
        #
        # To be conservative we increase this threshold for N/M by 2.
        is_exhaustive = inductor_config.max_autotune_gemm_search_space == "exhaustive"
        if is_exhaustive or not use_decompose_k_choice(m, n, k, threshold_multiple=2):
            templates_to_use.append(mm_template)

            if use_triton_blackwell_tma_template(
                mat1, mat2, output_layout=layout, add_guards=True
            ):
                templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
            elif use_triton_tma_template(
                mat1, mat2, output_layout=layout, add_guards=True
            ):
                if torch.version.hip is None:
                    templates_to_use.append(persistent_tma_mm_template)
                else:
                    templates_to_use.append(persistent_mm_template)

        templates_to_use.append(mm_contiguous_subgraph_template)

    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            templates_to_use,
            "mm",
            kwarg_overrides=kwarg_overrides,
        )
    )

    if (
        out_dtype is None
        and is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op("mm")
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, kernel_inputs.nodes()
        )

    if out_dtype is None and is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())
    if out_dtype is None and is_nonzero and use_ck_tile_gemm_template(layout, m, n, k):
        CKTileGemmTemplate.add_choices(choices, layout, kernel_inputs.nodes())

    if (
        out_dtype is None
        and is_nonzero
        and use_nv_universal_gemm_template(layout, m, n, k, mat1, mat2)
    ):
        from ..codegen.nv_universal_gemm import add_nv_universal_gemm_choices

        add_nv_universal_gemm_choices(choices, layout, kernel_inputs)

    if out_dtype is None and use_cpp_gemm_template(layout, mat1, mat2):
        CppGemmTemplate.add_choices(
            choices,
            layout,
            kernel_inputs.nodes(),
        )

    input_nodes = [mat1, mat2]
    if (
        out_dtype is None
        and is_nonzero
        and use_triton_template(layout)
        and torch._inductor.config.run_autoheuristic(name)
        and is_triton(mat1)
    ):
        always_included = []
        if use_aten_gemm_kernels():
            always_included.append("extern_mm")
        num_choices_before_extra_configs = len(choices)
        choices.extend(
            V.choices.get_template_configs(
                # TODO(coconutruben): remove once we deprecate ah
                # mm-extra is a hack to keep the ah functionality alive
                # while we transition to the unified kwargs retrieval
                kernel_inputs,
                [mm_template],
                "mm-ah",
            )
        )

        # using AutoHeuristic for ranking
        ah_choices = mm_autoheuristic(
            mat1,
            mat2,
            m,
            n,
            k,
            choices,
            name,
            input_nodes,
            mm_operations(),
            None,
            top_k=10,
            always_included=always_included,
        )
        if not torch._inductor.config.collect_autoheuristic(name):
            # if we are collecting data, we do not want to modify choices
            if ah_choices is not None and len(ah_choices) > 0:
                # the order in which autoheuristic returns choices is not the same as
                # as the order of choices, which affects things like epilogue fusion.
                # once epilogue fusion benchmarks choices in sorted order, I think we can
                # just use the order returned by autoheuristic
                choices = [choice for choice in choices if choice in ah_choices]
            else:
                choices = choices[:num_choices_before_extra_configs]

    if out_dtype is None:
        for k in inductor_config.external_matmul:
            choices.append(
                lazy_register_extern_choice(k).bind(kernel_inputs.nodes(), layout)
            )

    best_config_future = None
    if out_dtype is None and torch._inductor.config.remote_gemm_autotune_cache:
        # Purposely not awaiting the future here - this kicks off the best config lookup at lowering time
        # The future will be awaited at scheduling time in select_algorithm.py
        best_config_future = gen_best_config(mat1, mat2)

    if box := distributed_autotune.maybe_autotune_remote(
        name, choices, kernel_inputs.nodes(), layout
    ):
        return box

    node, _ = autotune_select_algorithm(
        name,
        choices,
        kernel_inputs.nodes(),
        layout,
        best_config_future=best_config_future,
    )
    return node


@register_lowering(aten._int_mm, type_promotion_kind=None)
def tuned_int_mm(mat1, mat2, *, layout=None):
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=torch.int32
    )
    name = "int_mm"
    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten._int_mm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten._int_mm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    static_shape, is_nonzero = _is_static_problem(layout)
    use_cutlass = static_shape and is_nonzero and use_cutlass_template(layout, m, n, k)
    choices: list[ChoiceCaller] = []

    # Create MMKernelInputs for Int MM
    kernel_inputs = MMKernelInputs([mat1, mat2], out_dtype=torch.int32)

    # Collect all templates for unified call
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    if use_aten_gemm_kernels():
        templates_to_use.append(aten__int_mm)

    if is_nonzero and use_triton_template(
        layout, enable_int32=True, check_max_autotune=False
    ):
        templates_to_use.append(mm_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, name)
    )

    if use_cutlass and _use_cutlass_for_op(name):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, kernel_inputs.nodes(), fuseable=True, non_fuseable=True
        )

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


@register_lowering(aten.addmm, type_promotion_kind=None)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    """
    Lowering for autotuning aten.addmm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    if beta == 0 and mat1.get_device().type == "cuda":
        _check_addmm_input_metadata(inp, mat1, mat2)
        if alpha == 0:
            _, _, _, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
            return lowerings[aten.full](
                layout.size,
                0,
                dtype=layout.dtype,
                device=layout.device,
            )
        if layout is not None:
            result = lowerings[aten.mm](mat1, mat2, layout=layout)
        else:
            result = lowerings[aten.mm](mat1, mat2)
        if alpha != 1:
            result = lowerings[aten.mul](alpha, result)
        return result

    if use_native_matmul(mat1, mat2):
        if beta == 0:
            _check_addmm_input_metadata(inp, mat1, mat2)
            arg1 = 0
        else:
            arg1 = lowerings[aten.mul](beta, inp)

        if alpha == 0:
            arg2 = 0
        else:
            arg2 = lowerings[aten.mul](alpha, lowerings[aten.mm](mat1, mat2))

        return lowerings[aten.add](arg1, arg2)

    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    inp = realize_inputs(inp)
    static_shape, is_nonzero = _is_static_problem(layout)
    name = "addmm"

    # Create MMKernelInputs for AddMM at the top
    kernel_inputs = MMKernelInputs(
        [inp_expanded, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )
    kernel_inputs_aten = MMKernelInputs(
        [inp, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )

    choices: list[ChoiceCaller] = []

    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten.addmm_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.addmm: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )
    if (not is_nonzero) or (
        not (inductor_config.max_autotune or inductor_config.max_autotune_gemm)
    ):
        choices.extend(
            V.choices.get_template_configs(
                kernel_inputs_aten,
                [aten_addmm],
                name,
            )
        )
        node, _ = autotune_select_algorithm(
            name, choices, kernel_inputs.nodes(), layout
        )
        return node

    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []

    if use_aten_gemm_kernels():
        aten_templates: list[ExternKernelChoice | KernelTemplate] = [aten_addmm]
        if (
            inp.get_stride()[0] == 0
            and len(inp.get_size()) == 2
            and inductor_config.triton.autotune_cublasLt
            and not V.graph.cpp_wrapper  # bias_addmm only has a Python implementation
        ):
            aten_templates.append(aten_bias_addmm)

        # On ROCm, ATen choices use original bias input; non-ROCm keeps unified inputs.
        choices.extend(
            V.choices.get_template_configs(kernel_inputs_aten, aten_templates, name)
        )

    if is_nonzero and use_triton_template(layout, check_max_autotune=False):
        templates_to_use.append(mm_template)

        if use_triton_blackwell_tma_template(
            mat1, mat2, output_layout=layout, add_guards=True
        ):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
        elif use_triton_tma_template(mat1, mat2, output_layout=layout, add_guards=True):
            if torch.version.hip is None:
                templates_to_use.append(persistent_tma_mm_template)
            else:
                templates_to_use.append(persistent_mm_template)

        # Manually call get_template_configs as use 1-D bias if possible
        choices.extend(
            V.choices.get_template_configs(
                kernel_inputs_aten, [addmm_contiguous_subgraph_template], name
            )
        )
    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, name)
    )

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            # reorder here because CUTLASS expects (x, w, bias) but torch
            # is bias, x, w
            kernel_inputs.nodes(reorder=[1, 2, 0]),
            alpha=alpha,
            beta=beta,
            input_reorder=[2, 0, 1],
        )

    if is_nonzero and use_nv_universal_gemm_template(layout, m, n, k, mat1, mat2):
        from ..codegen.nv_universal_gemm import add_nv_universal_addmm_choices

        # inp is the original (un-expanded) bias, so a 1D row vector routes to
        # the row-broadcast epilogue impl.
        add_nv_universal_addmm_choices(
            choices, layout, kernel_inputs, inp, alpha=alpha, beta=beta
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(
            choices,
            layout,
            # reorder here because CK expects (x, w, bias) but torch
            # is bias, x, w
            kernel_inputs.nodes(reorder=[1, 2, 0]),
            alpha=alpha,
            beta=beta,
            input_reorder=[2, 0, 1],
        )

    if use_cpp_gemm_template(layout, mat1, mat2):
        CppGemmTemplate.add_choices(
            choices,
            layout,
            kernel_inputs.nodes(),
            alpha=alpha,
            beta=beta,
            has_bias=True,
        )

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


@register_lowering(aten._sparse_semi_structured_mm, type_promotion_kind=None)
def tuned_sparse_semi_structured_mm(
    mat1, mat1_meta, mat2, *, out_dtype=None, layout=None
):
    from torch._inductor.select_algorithm import realize_inputs

    # TODO(coconturuben): support V.choices.get_mm_configs for sparse_semi_structured_mm
    mat1, mat1_meta, mat2 = realize_inputs(mat1, mat1_meta, mat2)
    m1, k1 = mat1.get_size()
    m2, _ = mat1_meta.get_size()
    k2, n = mat2.get_size()
    m = V.graph.sizevars.check_equals_and_simplify(m1, m2)
    k = V.graph.sizevars.check_equals_and_simplify(2 * k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        layout = FixedLayout(
            mat2.get_device(),
            out_dtype if out_dtype else mat2.get_dtype(),
            [m, n],
            [n, 1],
        )
    else:
        if out_dtype is not None:
            raise AssertionError("out_dtype is ignored if layout is specified.")

    choices = (
        [
            aten__sparse_semi_structured_mm.bind(
                (mat1, mat1_meta, mat2), layout, out_dtype=out_dtype
            )
        ]
        if use_aten_gemm_kernels()
        else []
    )

    if (
        m * n != 0
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op("sparse_semi_structured_mm")
    ):
        CUTLASS2xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2, mat1_meta], fuseable=True, non_fuseable=True
        )

    node, _ = autotune_select_algorithm(
        "sparse_semi_structured_mm", choices, (mat1, mat1_meta, mat2), layout
    )
    return node


scaling_pairs = [
    (ScalingType.TensorWise, ScalingType.TensorWise),
    (ScalingType.RowWise, ScalingType.RowWise),
    (ScalingType.BlockWise1x128, ScalingType.BlockWise128x128),
    (ScalingType.BlockWise1x128, ScalingType.BlockWise1x128),
    (ScalingType.BlockWise128x128, ScalingType.BlockWise1x128),
]


epilogue_scaling_types = [ScalingType.TensorWise, ScalingType.RowWise]
main_loop_scaling_types = [ScalingType.BlockWise1x128, ScalingType.BlockWise128x128]


def _is_tensorwise_scaling(sz: Any) -> bool:
    return (len(sz) == 0) or all(
        V.graph.sizevars.statically_known_equals(d, 1) for d in sz
    )


def _is_rowwise_scaling(sz: Any, transpose: bool) -> bool:
    idx = 0 if transpose else -1
    return V.graph.sizevars.statically_known_equals(sz[idx], 1)


def _is_blockwise1xTILESIZE_scaling(
    sz: Any, tensor_sz: Any, tile_size: int, transpose: bool
) -> bool:
    lhs = 1 if transpose else 0
    rhs = 0 if transpose else 1
    return V.graph.sizevars.statically_known_equals(
        sz[lhs], tensor_sz[lhs]
    ) and V.graph.sizevars.statically_known_equals(
        sz[rhs], ceildiv(tensor_sz[rhs], tile_size)
    )


def _is_blockwise128x128_scaling(sz: Any, tensor_sz: Any) -> bool:
    return V.graph.sizevars.statically_known_equals(
        sz[0], ceildiv(tensor_sz[0], 128)
    ) and V.graph.sizevars.statically_known_equals(sz[1], ceildiv(tensor_sz[1], 128))


def is_desired_scaling(
    t: Any,
    scale_size: torch.Tensor,
    scaling_type: ScalingType,
    transpose: bool = False,
) -> bool:
    match scaling_type:
        case ScalingType.TensorWise:
            return _is_tensorwise_scaling(scale_size)
        case ScalingType.RowWise:
            return _is_rowwise_scaling(scale_size, transpose)
        case ScalingType.BlockWise1x128:
            return _is_blockwise1xTILESIZE_scaling(
                scale_size, t.get_size(), 128, transpose
            )
        case ScalingType.BlockWise128x128:
            return _is_blockwise128x128_scaling(scale_size, t.get_size())
        case _:
            raise AssertionError(f"Unsupported scaling type {scaling_type}")


def get_tile_size(scale_option) -> int:
    match scale_option:
        case ScalingType.BlockWise128x128:
            return 128
        case ScalingType.BlockWise1x128:
            return 128
        case _:
            raise AssertionError(
                f"Unsupported scaling type {scale_option} in get_tile_size"
            )


def get_scaling_options(
    mat_a: Any,
    mat_b: Any,
    scale_a_size: torch.Tensor,
    scale_b_size: torch.Tensor,
) -> tuple[ScalingType, ScalingType]:
    for scale_option_a, scale_option_b in scaling_pairs:
        if is_desired_scaling(
            mat_a, scale_a_size, scale_option_a
        ) and is_desired_scaling(mat_b, scale_b_size, scale_option_b, transpose=True):
            return scale_option_a, scale_option_b

    raise AssertionError(
        f"Inductor Triton does not support scale_a.shape = {scale_a_size}, scale_b.shape = {scale_b_size}"
    )  # verify that shapes are supported by at least one existing pairing


# Inductor has no template or extern choice that understands swizzled scale
# layouts for _scaled_mm_v2 yet; defer those to the eager op. add_to_fallback_set
# is False because this handler is invoked manually from the lowering below, not
# registered as the op's global fallback.
scaled_mm_v2_fallback = fallback_handler(
    aten._scaled_mm_v2.default, add_to_fallback_set=False
)


@register_lowering(aten._scaled_mm_v2.default, type_promotion_kind=None)
def tuned_scaled_mm_v2(
    mat_a,
    mat_b,
    scale_a: list[Any],
    recipe_a: list[int],
    swizzle_a: list[int],
    scale_b: list[Any],
    recipe_b: list[int],
    swizzle_b: list[int],
    bias=None,
    out_dtype=None,
    contraction_dim=None,
    use_fast_accum=False,
    layout=None,
):
    """
    Performs an optimized matrix multiplication where scaling factors are
    applied to the inputs per the supplied recipes, and optionally swizzled.

    This is the _scaled_mm_v2 API, which takes scale recipes (ScalingType) and
    swizzle patterns alongside the scale tensors, and supports multi-level
    scaling via lists.
    """

    # Inductor only has Triton/extern lowerings for single-level, fp32-scaled,
    # non-swizzled _scaled_mm_v2 with the "supported" recipes (TensorWise,
    # RowWise, and DeepSeek BlockWise1x128/128x128). Everything else has no
    # template or extern choice here, so defer to the eager _scaled_mm_v2 op:
    #   - swizzled scale layouts (e.g. CUDA/Blackwell MXFP8/NVFP4)
    #   - the blockwise MX/NVFP4 recipes BlockWise1x32/1x16 (also how XPU
    #     expresses MX/NVFP4, with NO_SWIZZLE)
    #   - multi-level scales (two-level NVFP4)
    #   - any non-fp32 block scale
    # The eager op is called directly so it keeps its native v2 scale_b
    # convention, unlike the v1 aten__fp8_mm choice used on the supported path.
    def check_supported_recipe(recipe: list[int]) -> bool:
        disallowed = OrderedSet([ScalingType.BlockWise1x16, ScalingType.BlockWise1x32])
        return all(ScalingType(r) not in disallowed for r in recipe)

    is_single_level_scale = len(scale_a) == 1 and len(scale_b) == 1
    supported_recipe = check_supported_recipe(recipe_a) and check_supported_recipe(
        recipe_b
    )
    if (
        any(s != 0 for s in swizzle_a)
        or any(s != 0 for s in swizzle_b)
        or not supported_recipe
        or not is_single_level_scale
        or scale_a[0].dtype != torch.float32
    ):
        # contraction_dim is a non-optional int[] in the schema (default []);
        # this lowering defaults it to None, so coerce before the eager call.
        fallback_contraction_dim = [] if contraction_dim is None else contraction_dim
        return scaled_mm_v2_fallback(
            mat_a,
            mat_b,
            scale_a,
            recipe_a,
            swizzle_a,
            scale_b,
            recipe_b,
            swizzle_b,
            bias,
            out_dtype,
            fallback_contraction_dim,
            use_fast_accum,
        )
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat_a, mat_b = mm_args(
        mat_a, mat_b, layout=layout, out_dtype=out_dtype
    )
    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten._scaled_mm_v2.default_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten._scaled_mm_v2.default: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )
    name = "scaled_mm"
    check_supported_striding(mat_a, mat_b)

    scale_a_real, scale_b_real = realize_inputs(scale_a[0], scale_b[0])

    input_nodes: list[Any]

    if not bias:
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real]
    else:
        bias_real = realize_inputs(bias)
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real, bias_real]

    # Create MMKernelInputs for Scaled MM (matrices are at indices 0, 1)
    kernel_inputs = MMKernelInputs(
        input_nodes, mat1_idx=0, mat2_idx=1, out_dtype=out_dtype
    )

    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    kwarg_overrides = {}

    if use_aten_gemm_kernels():
        templates_to_use.append(aten__fp8_mm)
        kwarg_overrides[aten__fp8_mm.uid] = dict(
            out_dtype=out_dtype, use_fast_accum=use_fast_accum
        )

    _, is_nonzero = _is_static_problem(layout)

    if (
        # We don't have triton lowerings for the MX variants yet
        is_single_level_scale
        and supported_recipe
        and scale_a[0].dtype == torch.float32
        and is_nonzero
        and use_triton_template(layout, enable_float8=True, check_max_autotune=False)
    ):
        overriders = dict(USE_FAST_ACCUM=use_fast_accum)

        # Note: No NVFP4 support at this point - can ignore swizzling, and take only the
        #       first scale types passed.
        scale_option_a, scale_option_b = (
            ScalingType(recipe_a[0]),
            ScalingType(recipe_b[0]),
        )

        # TODO (paulzhan): There is no template that exists for bias and TMA
        # Don't run tma template currently if bias exist
        if (
            use_triton_tma_template(mat_a, mat_b, output_layout=layout, add_guards=True)
            and not bias
        ):
            overriders["SCALE_RECIPE_A"] = scale_option_a.value
            overriders["SCALE_RECIPE_B"] = scale_option_b.value

            if use_triton_scaling_template(
                scale_option_a, scale_option_b, epilogue_scaling_types
            ):
                templates_to_use.append(scaled_mm_device_tma_epilogue_scaling_template)
                kwarg_overrides[scaled_mm_device_tma_epilogue_scaling_template.uid] = (
                    overriders
                )
            elif use_triton_scaling_template(
                scale_option_a, scale_option_b, main_loop_scaling_types
            ):
                overriders["TILE_SIZE_A"] = get_tile_size(scale_option_a)
                overriders["TILE_SIZE_B"] = get_tile_size(scale_option_b)

                templates_to_use.append(scaled_mm_device_tma_main_loop_scaling_template)
                kwarg_overrides[scaled_mm_device_tma_main_loop_scaling_template.uid] = (
                    overriders
                )
            else:
                raise AssertionError(
                    "Inductor Triton does not support scaling options that are present "
                    + "in both epilogue scaling and main loop scaling"
                )

        if (
            use_triton_blackwell_tma_template(
                mat_a, mat_b, output_layout=layout, add_guards=True
            )
            and not bias
        ):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
            kwarg_overrides[blackwell_ws_persistent_device_tma_mm_template.uid] = (
                overriders
            )

        if use_triton_scaling_template(
            scale_option_a, scale_option_b, epilogue_scaling_types
        ):
            templates_to_use.append(mm_template)
            kwarg_overrides[mm_template.uid] = overriders

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            templates_to_use,
            name,
            kwarg_overrides=kwarg_overrides,
        )
    )

    # NVGEMM get_kernels() will return empty if the scaling mode/dtype is unsupported
    if is_nonzero and use_nv_universal_gemm_template(layout, m, n, k, mat_a, mat_b):
        from ..codegen.nv_universal_gemm import add_nv_universal_scaled_gemm_choices

        add_nv_universal_scaled_gemm_choices(
            choices,
            layout,
            input_nodes,
            kernel_inputs=kernel_inputs,
        )

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            kernel_inputs.nodes(),  # type: ignore[arg-type]
            use_fast_accum=use_fast_accum,  # type: ignore[arg-type]
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


@register_lowering(aten._scaled_mm.default, type_promotion_kind=None)  # type: ignore[misc]
def tuned_scaled_mm(
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
    layout=None,
):
    """
    Performs an optimized matrix multiplication where scaling factors are applied
    to the inputs and/or output.

    Args:
        mat1 (Tensor): First input matrix
        mat2 (Tensor): Second input matrix
        scale1 (Tensor): Scale factor applied to mat1 (supports broadcasting)
        scale2 (Tensor): Scale factor applied to mat2 (supports broadcasting)
        bias (Tensor, optional): Optional bias tensor to add to the result
        layout: Layout hint for optimization

    Returns:
        Tensor: The result of the scaled matrix multiplication
    """
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat_a, mat_b = mm_args(
        mat_a, mat_b, layout=layout, out_dtype=out_dtype
    )
    # below is for getting an overview logging info of inductor mms
    counters["aten_mm_info"][f"aten._scaled_mm.default_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten._scaled_mm.default: m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        m,
        n,
        k,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )
    name = "scaled_mm"
    check_supported_striding(mat_a, mat_b)

    scale_a_real, scale_b_real = realize_inputs(scale_a, scale_b)

    input_nodes: list[Any]

    if not bias:
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real]
    else:
        bias_real = realize_inputs(bias)
        input_nodes = [mat_a, mat_b, scale_a_real, scale_b_real, bias_real]

    # Create MMKernelInputs for Scaled MM (matrices are at indices 0, 1)
    kernel_inputs = MMKernelInputs(
        input_nodes, mat1_idx=0, mat2_idx=1, out_dtype=out_dtype
    )

    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    kwarg_overrides = {}

    if use_aten_gemm_kernels():
        templates_to_use.append(aten__fp8_mm)
        kwarg_overrides[aten__fp8_mm.uid] = dict(
            out_dtype=out_dtype, use_fast_accum=use_fast_accum
        )

    _, is_nonzero = _is_static_problem(layout)

    if (
        # We don't have triton lowerings for the MX variants yet
        scale_a.dtype == torch.float32
        and is_nonzero
        and use_triton_template(layout, enable_float8=True, check_max_autotune=False)
    ):
        overriders = dict(USE_FAST_ACCUM=use_fast_accum)

        scale_a_size, scale_b_size = scale_a_real.shape, scale_b_real.shape

        scale_option_a, scale_option_b = get_scaling_options(
            mat_a, mat_b, scale_a_size, scale_b_size
        )

        # TODO (paulzhan): There is no template that exists for bias and TMA
        # Don't run tma template currently if bias exist
        if (
            use_triton_tma_template(mat_a, mat_b, output_layout=layout, add_guards=True)
            and not bias
        ):
            overriders["SCALE_RECIPE_A"] = scale_option_a.value
            overriders["SCALE_RECIPE_B"] = scale_option_b.value

            if use_triton_scaling_template(
                scale_option_a, scale_option_b, epilogue_scaling_types
            ):
                templates_to_use.append(scaled_mm_device_tma_epilogue_scaling_template)
                kwarg_overrides[scaled_mm_device_tma_epilogue_scaling_template.uid] = (
                    overriders
                )
            elif use_triton_scaling_template(
                scale_option_a, scale_option_b, main_loop_scaling_types
            ):
                overriders["TILE_SIZE_A"] = get_tile_size(scale_option_a)
                overriders["TILE_SIZE_B"] = get_tile_size(scale_option_b)

                templates_to_use.append(scaled_mm_device_tma_main_loop_scaling_template)
                kwarg_overrides[scaled_mm_device_tma_main_loop_scaling_template.uid] = (
                    overriders
                )
            else:
                raise AssertionError(
                    "Inductor Triton does not support scaling options that are present "
                    + "in both epilogue scaling and main loop scaling"
                )

        if (
            use_triton_blackwell_tma_template(
                mat_a, mat_b, output_layout=layout, add_guards=True
            )
            and not bias
        ):
            templates_to_use.append(blackwell_ws_persistent_device_tma_mm_template)
            kwarg_overrides[blackwell_ws_persistent_device_tma_mm_template.uid] = (
                overriders
            )

        if use_triton_scaling_template(
            scale_option_a, scale_option_b, epilogue_scaling_types
        ):
            templates_to_use.append(mm_template)
            kwarg_overrides[mm_template.uid] = overriders

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            templates_to_use,
            name,
            kwarg_overrides=kwarg_overrides,
        )
    )

    # NVGEMM get_kernels() will return empty if the scaling mode/dtype is unsupported
    if is_nonzero and use_nv_universal_gemm_template(layout, m, n, k, mat_a, mat_b):
        from ..codegen.nv_universal_gemm import add_nv_universal_scaled_gemm_choices

        add_nv_universal_scaled_gemm_choices(
            choices,
            layout,
            input_nodes,
            kernel_inputs=kernel_inputs,
        )

    # Early return for MX variants
    if scale_a.dtype != torch.float32:
        node, _ = autotune_select_algorithm(name, choices, input_nodes, layout)
        return node

    if (
        is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices,
            layout,
            kernel_inputs.nodes(),  # type: ignore[arg-type]
            use_fast_accum=use_fast_accum,  # type: ignore[arg-type]
        )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


@functools.cache
def _is_sm7x_or_older_gpu(index: int | None) -> bool:
    props = torch.cuda.get_device_properties(index or 0)
    return props.major <= 7


def dims_are_int(dims):
    return all(isinstance(dim, int) for dim in dims)


def mm_autoheuristic(
    mat1,
    mat2,
    m,
    n,
    k,
    choices,
    name,
    input_nodes,
    ops,
    precondition,
    top_k: int | None = None,
    always_included=None,
):
    m, n, k = get_size_hints(mat1, mat2, m, n, k)
    if not dims_are_int([m, n, k]):
        return None
    mat1_stride, mat2_stride = get_size_hints_strides(mat1, mat2)

    def get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride):
        context = AHContext()
        context.add_feature("m", m)
        context.add_feature("k", k)
        context.add_feature("n", n)
        context.add_feature("mat1_dtype", mat1.layout.dtype, is_categorical=True)
        context.add_feature("mat2_dtype", mat2.layout.dtype, is_categorical=True)
        context_add_strides(context, "mat1", mat1_stride)
        context_add_strides(context, "mat2", mat2_stride)
        context.add_feature(
            "mat1_iscontig", mat1.layout.is_contiguous(), is_categorical=True
        )
        context.add_feature(
            "mat2_iscontig", mat2.layout.is_contiguous(), is_categorical=True
        )
        if name == "mm":
            context_add_using_tf32(context, mat1.layout.dtype)
        return context

    def fallback():
        return None

    context = get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride)
    autoheuristic = AutoHeuristicSelectAlgorithm(
        fallback=fallback,
        choices=choices,
        input_nodes=input_nodes,
        context=context,
        name=name,
        augment_context=ops,
        precondition=precondition,
    )

    if top_k is not None:
        # TODO: is there a cleaner way to ensure aten.mm is always included?
        return autoheuristic.get_top_k_choices_caller(
            top_k, always_included=always_included
        )

    return autoheuristic.get_choice_caller()


def get_size_hints(mat1, mat2, m, n, k):
    if not isinstance(m, int) or not isinstance(k, int):
        (m, k) = V.graph.sizevars.optimization_hints(mat1.get_size())

    if not isinstance(n, int) or not isinstance(k, int):
        (k, n) = V.graph.sizevars.optimization_hints(mat2.get_size())
    return m, n, k


def get_size_hints_strides(mat1, mat2):
    mat1_stride = mat1.layout.stride
    mat2_stride = mat2.layout.stride
    strides = [mat1_stride, mat2_stride]
    strides_hints = []
    for stride in strides:
        if not isinstance(stride, int):
            stride = V.graph.sizevars.optimization_hints(stride)
        strides_hints.append(stride)
    return strides_hints[0], strides_hints[1]
<|endoftext|>"""Format and emit opt-in FlexGEMM compilation diagnostics.

The ``flex_gemm`` logger follows lowering from the captured FX body through
semantic analysis, buffer planning, generated CuTeDSL, and kernel selection.
Each phase is also available as a structured trace artifact.
"""

import logging
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.kernel.gemm_epilogue import (
    NormalizedGetItem,
    NormalizedPrepareSoftmax,
    NormalizedReduction,
    NormalizedSelect,
    NormalizedSplit,
    NormalizedSqueeze,
    NormalizedUnsupportedReduction,
    NormalizedView,
)
from torch._logging import LazyString, trace_structured


if TYPE_CHECKING:
    from torch._inductor import ir
    from torch._inductor.heuristics.template.flex_gemm import GemmConfigKey
    from torch._inductor.kernel.flex_gemm.fx_cutedsl_codegen import (
        FlexGemmEpilogueAnalysis,
    )


flex_gemm_log = logging.getLogger(__name__)


def log_flex_gemm_artifact(
    name: str,
    payload_fn: Callable[[], str],
    *,
    lowering_name: str | None = None,
    verbose: bool = False,
) -> None:
    """Emit one lazily rendered local and structured FlexGEMM phase."""
    heading = "FLEXGEMM LOWERING"
    if lowering_name is not None:
        heading += f" [{lowering_name}]"
    flex_gemm_log.log(
        logging.DEBUG if verbose else logging.INFO,
        "%s\n ===== %s =====\n%s",
        heading,
        name.replace("_", " ").upper(),
        LazyString(payload_fn),
    )
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": f"flex_gemm_{name}",
            "encoding": "string",
            "verbosity": "verbose" if verbose else "concise",
            **({} if lowering_name is None else {"lowering_name": lowering_name}),
        },
        payload_fn=payload_fn,
    )


def _append_items(lines: list[str], label: str, items: Iterable[str]) -> None:
    """Append an indented report section, including an explicit empty marker."""
    values = tuple(items)
    lines.append(f"{label}:")
    lines.extend(f"  {value}" for value in values or ("(none)",))


def _format_ir_tensor(name: str, node: "ir.IRNode") -> str:
    """Format the tensor contract visible to Inductor lowering."""
    stride = node.maybe_get_stride()
    return (
        f"{name}: shape={tuple(node.get_size())}, "
        f"stride={'unrealized' if stride is None else tuple(stride)}, "
        f"dtype={node.get_dtype()}, device={node.get_device_or_error()}"
    )


def format_flex_gemm_problem(
    graph_module: torch.fx.GraphModule,
    gemm_op: torch._ops.OpOverload,
    gemm_inputs: Sequence[tuple[str, "ir.IRNode"]],
    captures: Sequence[tuple[str, "ir.IRNode"]],
    *,
    alpha: float,
    beta: float,
    tuned: bool,
    fast_math: bool,
    explicit_config: dict[str, Any] | None,
) -> str:
    """Render the inputs and captured body entering FlexGEMM analysis."""
    lines = [
        f"gemm_op: {gemm_op}",
        "kernel_options:",
        "  backend: QUACK",
        f"  tuned: {tuned}",
        f"  fast_math: {fast_math}",
        f"  config: {explicit_config!r}",
        f"alpha: {alpha}",
        f"beta: {beta}",
    ]
    _append_items(
        lines,
        "gemm_inputs",
        (_format_ir_tensor(name, node) for name, node in gemm_inputs),
    )
    _append_items(
        lines,
        "captures",
        (_format_ir_tensor(name, node) for name, node in captures),
    )
    lines.extend(
        (
            "body:",
            graph_module.print_readable(
                print_output=False,
                include_stride=True,
                include_device=True,
            ).strip(),
        )
    )
    return "\n".join(lines)


def _format_fx_tensor(node: torch.fx.Node) -> str:
    """Format an FX node using its captured fake-tensor metadata."""
    meta = node.meta.get("val")
    if not isinstance(meta, torch.Tensor):
        return node.name
    return (
        f"{node.name}: shape={tuple(meta.shape)}, stride={tuple(meta.stride())}, "
        f"dtype={meta.dtype}"
    )


def _format_geometry(geometry: Any) -> str:
    """Format grouped GEMM geometry in logical M/N terms."""
    axis = "M" if geometry.axis == 0 else "N"
    return f"axis={axis}, group={geometry.group}"


def _format_output_contraction(contraction: Any | None) -> str:
    """Format the logical contraction applied to the main output."""
    if contraction is None:
        return "none"
    layout = "chunked" if contraction.chunked else "interleaved"
    return f"N-axis, group={contraction.group}, layout={layout}"


def _format_normalized_dataflow(node: torch.fx.Node, normalized: Any) -> str:
    """Render one normalized FX operation as compact dataflow."""
    match normalized:
        case NormalizedView(shape=shape):
            operation = f"view(shape={shape})"
        case NormalizedReduction(
            dim=dim,
            keepdim=keepdim,
            reduction_type=reduction_type,
        ):
            operation = f"{reduction_type}(dim={dim}, keepdim={keepdim})"
        case NormalizedPrepareSoftmax(dim=dim):
            operation = f"prepare_softmax(dim={dim})"
        case NormalizedSqueeze():
            operation = "squeeze"
        case NormalizedGetItem(index=index):
            operation = f"getitem(index={index})"
        case NormalizedSplit(split_size=split_size, dim=dim):
            operation = f"split(size={split_size}, dim={dim})"
        case NormalizedSelect(dim=dim, index=index):
            operation = f"select(dim={dim}, index={index})"
        case NormalizedUnsupportedReduction():
            operation = f"unsupported_reduction({node.target})"
        case _:
            return repr(normalized)
    return f"{normalized.source.name} -> {operation}"


def format_flex_gemm_analysis(analysis: "FlexGemmEpilogueAnalysis") -> str:
    """Render the semantic decisions a FlexGEMM developer acts on first."""
    outputs = analysis.outputs
    lines = [
        "outputs:",
        f"  main: {_format_fx_tensor(outputs.output)}",
        f"  output_contraction: {_format_output_contraction(outputs.output_contraction)}",
    ]
    if outputs.aux_outputs:
        lines.append("  auxiliary:")
        lines.extend(
            f"    {_format_fx_tensor(output)}" for output in outputs.aux_outputs
        )
    else:
        lines.append("  auxiliary: (none)")

    lines.append("")
    if outputs.local_reduce is None:
        lines.append("local_reduction: none")
    else:
        local_reduce = outputs.local_reduce
        store = local_reduce.store
