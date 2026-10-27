                is_static_int=True,
            )

        expr = self.kernel.rename_indexing(expr)
        is_lane_uniform, is_contiguous, contiguous_width = self._analyze_index_fragment(
            expr
        )
        result = self.kernel.cse.newvar(dtype=torch_dtype)
        self.kernel.body.writeline(
            f"{result} = ssa_to_fragment({self.kernel.kexpr(expr)}, {cute_dtype})"
        )
        return CuteDSLIndexFragment(
            str(result),
            is_static_int=False,
            is_lane_uniform=is_lane_uniform,
            is_contiguous=is_contiguous,
            contiguous_width=contiguous_width,
        )

    def _analyze_index_fragment(
        self, expr: sympy.Expr
    ) -> tuple[bool, bool, int | None]:
        if self.vector_load_config is None:
            return False, False, None
        if self.vector_load_config.vec_size <= 1:
            return True, False, None

        contiguous_ok = True
        for sym in expr.free_symbols:
            info = self._index_symbol_info.get(sym)
            if info is not None:
                if info.expr is None:
                    # If we dont know the semantics of the index, we cannot determine lane behavior.
                    # So we default to safe/pessimistic behavior: per-lane gather, no uniformity, no contiguity.
                    return False, False, None
                contiguous_ok &= info.contiguous_ok

        semantic_expr = self._semantic_index_expr(expr)
        if self._has_unresolved_index_symbols(semantic_expr):
            # ditto -> we dont know so choose safe/pessimistic behavior
            return False, False, None

        lane_info = classify_lane_expr(
            semantic_expr,
            self.vector_load_config.index,
            max_width=self.vector_load_config.vec_size,
            uniform_symbols=self._lane_uniform_symbols(),
        )
        return (
            lane_info.is_uniform,
            lane_info.is_contiguous and contiguous_ok,
            lane_info.contiguous_width if contiguous_ok else None,
        )

    def _has_unresolved_index_symbols(self, semantic_expr: sympy.Expr) -> bool:
        """Return whether the expression references opaque computed indices.

        Lane analysis treats unknown non-lane symbols as lane-uniform scalars,
        which is correct for sizes and fixed inputs but silently broadcasts
        lane 0's gather when the symbol is actually per-lane runtime data
        (#188878). Recovered semantics substitute away generated CSE names, so
        a surviving symbol in the reserved ``tmpN`` namespace means recovery
        failed and the load must stay a per-lane gather.
        """
        prefix = self.kernel.cse.name_prefix
        lane_var = self.vector_load_config.index if self.vector_load_config else None
        return any(
            sym is not lane_var
            and sym.name.startswith(prefix)
            and sym.name[len(prefix) :].isdigit()
            for sym in semantic_expr.free_symbols
        )

    @staticmethod
    def _static_integer_expr(expr: object) -> int | sympy.Integer | None:
        if isinstance(expr, int | sympy.Integer):
            return expr
        if isinstance(expr, sympy.Symbol) and expr.name.lstrip("-").isdigit():
            return int(expr.name)
        if isinstance(expr, sympy.Expr):
            expr = V.graph.sizevars.simplify(expr)
            if expr.is_Integer:
                return expr
        return None

    @staticmethod
    def _wrap_static_index(
        index: int | sympy.Integer, dim_size: sympy.Expr
    ) -> int | sympy.Expr:
        if int(index) < 0:
            return V.graph.sizevars.simplify(sympy.Integer(index) + dim_size)
        return index

    def _semantic_index_expr(self, expr: sympy.Expr) -> sympy.Expr:
        """Recover the original index meaning from generated CSE names."""
        replacements: dict[sympy.Symbol, sympy.Expr] = {
            symbol: info.expr
            for symbol, info in self._index_symbol_info.items()
            if info.expr is not None
        }
        for var in self.kernel.cse._cache.values():
            if isinstance(var, CuteDSLCSEVariable) and var.index_expr is not None:
                replacements[sympy_index_symbol(var.name)] = var.index_expr
        return V.graph.sizevars.simplify(expr.xreplace(replacements))

    def _lane_uniform_symbols(self) -> OrderedSet[sympy.Symbol]:
        """Return generated CSE source symbols that are uniform across lanes."""
        vector_index_name = (
            None
            if self.vector_load_config is None
            else self.vector_load_config.index.name
        )
        return OrderedSet(
            [
                sympy_index_symbol(source_expr)
                for source_expr in self.kernel.cse._cache
                if isinstance(source_expr, str) and source_expr != vector_index_name
            ]
        )

    def indirect_indexing(self, index_var: str, size, check, wrap_neg=True):
        """Convert index variable to symbolic form."""
        if isinstance(index_var, int | sympy.Integer):
            if wrap_neg and index_var < 0:
                return V.graph.sizevars.simplify(sympy.Integer(index_var) + size)
            return sympy.Integer(index_var)
        if (
            wrap_neg
            and isinstance(index_var, CuteDSLCSEVariable)
            and index_var.index_expr is not None
            and not V.graph.sizevars.statically_known_geq(index_var.index_expr, 0)
        ):
            wrapped = self.kernel.cse.newvar(dtype=index_var.dtype)
            # Dynamic sizes render as Int64 aux_scalars; cast to the index dtype so
            # both cute.where branches agree (per-dim size, so narrowing is safe).
            index_cute_dtype = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(
                index_var.dtype, "cutlass.Int32"
            )
            size_expr = f"{index_cute_dtype}({self.kernel.kexpr(size)})"
            self.kernel.body.writeline(
                f"{wrapped} = cute.where("
                f"{index_var} < 0, {index_var} + {size_expr}, {index_var})"
            )
            symbol = sympy_index_symbol(str(wrapped))
            # Preserve the pre-wrap semantics so lane analysis sees the true
            # lane structure; the wrap remaps negative lanes individually, so
            # contiguity of the pre-wrap expression does not survive it.
            self._index_symbol_info[symbol] = IndexSymbolInfo(
                index_var.index_expr, contiguous_ok=False
            )
            return symbol
        source_name = None
        for expr, var in self.kernel.cse._cache.items():
            if var is index_var:
                source_name = expr if isinstance(expr, str) else None
                break
        symbol = sympy_index_symbol(source_name or str(index_var))
        index_expr = (
            index_var.index_expr if isinstance(index_var, CuteDSLCSEVariable) else None
        )
        self._index_symbol_info[symbol] = IndexSymbolInfo(index_expr)
        return symbol

    # pyrefly: ignore [bad-override]
    def store(
        self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None
    ) -> str:
        raise NotImplementedError(
            "Store operations not supported - CuteDSL limited to read-only operations"
        )

    def _add_kernel_input(self, name: str):
        """Add name as input to kernel and return input ref."""
        # Get the remapped name that will be used in the kernel
        remapped_name = self.kernel.args.input(name)
        # Track the remapped name for later collection
        if remapped_name not in self.tensor_buffers:
            self.tensor_buffers.append(remapped_name)
        return remapped_name

    def _process_indexing(self, index):
        """Process and rename indexing, adding symbols as kernel inputs."""
        renamed = self.kernel.rename_indexing(index)
        return self.kernel.kexpr(renamed)

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        try:
            return getattr(self._inner, name)(*args, **kwargs)
        except NotImplementedError as e:
            bar = "=" * 80
            msg = textwrap.dedent(f"""
                {bar}
                UNSUPPORTED CUTEDSL OPERATION: '{name}'
                {bar}
                This operation is not yet implemented in Inductor.

                Please open an issue at: https://github.com/pytorch/pytorch/issues
                with the following information:

                Operation: {name}
                Args: {args!r}
                Kwargs: {kwargs!r}

                Title your issue: [CuteDSL] Missing operation: {name}
                {bar}
            """).strip()
            raise NotImplementedError(msg) from e
<|endoftext|># mypy: allow-untyped-defs
"""
CuteDSL-specific operation overrides for pointwise operations.

This module provides CuteDSL implementations of common operations used in
template kernels, particularly for flex attention modifications.
"""

import dataclasses
import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import sympy

import torch
from torch._inductor.codegen.common import CSEVariable, OpOverrides
from torch._inductor.ops_handler import ReductionType
from torch._inductor.utils import get_bounds_index_expr
from torch._inductor.virtualized import OpsValue, V
from torch.utils._sympy.functions import Max, Min
from torch.utils._sympy.value_ranges import ValueRanges


class CuteDSLCSEVariable(CSEVariable):
    def __init__(
        self,
        name: str,
        bounds: ValueRanges[sympy.Expr],
        dtype: torch.dtype | None = None,
        shape=None,
        *,
        is_scalar_expr: bool = False,
        index_expr: sympy.Expr | None = None,
    ):
        super().__init__(name, bounds, dtype, shape)
        self.is_scalar_expr = is_scalar_expr
        self.index_expr = index_expr


CuteDSLArg = CSEVariable | str | bool | float | int


_CUTEDSL_FAST_MATH: ContextVar[bool] = ContextVar("cutedsl_fast_math", default=False)


@contextmanager
def use_cutedsl_fast_math(enabled: bool) -> Iterator[None]:
    """Scope whether CuTeDSL math operations request fast-math lowering."""
    token = _CUTEDSL_FAST_MATH.set(enabled)
    try:
        yield
    finally:
        _CUTEDSL_FAST_MATH.reset(token)


@dataclasses.dataclass(frozen=True)
class TensorSSAReduction:
    """Describe the CuTeDSL lowering for one TensorSSA value reduction."""

    cute_op: str
    init_val: str
    combine_expr: str


TENSORSSA_REDUCTIONS: dict[str, TensorSSAReduction] = {
    "sum": TensorSSAReduction("cute.ReductionOp.ADD", "0.0", "lhs + rhs"),
    "prod": TensorSSAReduction("cute.ReductionOp.MUL", "1.0", "lhs * rhs"),
    "max": TensorSSAReduction(
        "cute.ReductionOp.MAX", 'float("-inf")', "cutlass.max(lhs, rhs)"
    ),
    "min": TensorSSAReduction(
        "cute.ReductionOp.MIN", 'float("inf")', "cutlass.min(lhs, rhs)"
    ),
}


def tensorssa_reduction(reduction_type: ReductionType) -> TensorSSAReduction:
    """Return the TensorSSA descriptor for reductions representable by CuTe."""
    reduction = TENSORSSA_REDUCTIONS.get(reduction_type)
    if reduction is None:
        raise NotImplementedError(
            f"{reduction_type} does not map to a CuTe TensorSSA reduction"
        )
    return reduction


def upcast_compute_type(dtype: torch.dtype) -> torch.dtype:
    """Maybe upcast [b]float16 to float32"""
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


class CuteDSLOpOverrides(OpOverrides):
    """
    CuteDSL-specific operation overrides that generate code using CuteDSL syntax.

    CuteDSL TensorSSA objects have built-in operator overloads (__add__, __mul__, etc.)
    and math functions (cute.math.exp, cute.math.sqrt, etc.)
    """

    TORCH_TO_CUTE_DTYPE = {
        torch.float16: "cutlass.Float16",
        torch.bfloat16: "cutlass.BFloat16",
        torch.float32: "cutlass.Float32",
        torch.float64: "cutlass.Float64",
        torch.int8: "cutlass.Int8",
        torch.int16: "cutlass.Int16",
        torch.int32: "cutlass.Int32",
        torch.int64: "cutlass.Int64",
        torch.uint8: "cutlass.Uint8",
        torch.uint16: "cutlass.Uint16",
        torch.bool: "cutlass.Boolean",
        torch.float8_e4m3fn: "cutlass.Float8E4M3FN",
        torch.float8_e8m0fnu: "cutlass.Float8E8M0FNU",
        torch.float8_e5m2: "cutlass.Float8E5M2",
    }

    # Math constants
    LOG2_E = 1.4426950408889634  # 1/ln(2) for converting natural exp to base-2 exp

    @staticmethod
    def _get_cse_var(arg: CuteDSLArg) -> CSEVariable | None:
        """Extract CSEVariable from arg if it's a tensor (either direct or wrapped in OpsValue)."""
        if isinstance(arg, CSEVariable):
            return arg
        if isinstance(arg, OpsValue) and isinstance(arg.value, CSEVariable):
            return arg.value
        return None

    @staticmethod
    def _as_expr(arg: CuteDSLArg) -> str:
        cse_var = CuteDSLOpOverrides._get_cse_var(arg)
        if cse_var is not None:
            return str(cse_var)
        return str(arg)

    @staticmethod
    def _node_tensor_flags() -> tuple[bool, bool] | None:
        node = V.current_node
        if not isinstance(node, torch.fx.Node) or len(node.args) < 2:
            return None

        def _is_tensor(raw_arg: object) -> bool:
            if hasattr(raw_arg, "meta"):
                val = raw_arg.meta.get("val")
                return isinstance(val, torch.Tensor) and val.ndim > 0
            return False

        return _is_tensor(node.args[0]), _is_tensor(node.args[1])

    @staticmethod
    def _is_scalar_expr(arg: CuteDSLArg) -> bool:
        cse_var = CuteDSLOpOverrides._get_cse_var(arg)
        return isinstance(cse_var, CuteDSLCSEVariable) and cse_var.is_scalar_expr

    @staticmethod
    def _is_tensor_like(arg: CuteDSLArg) -> bool:
        cse_var = CuteDSLOpOverrides._get_cse_var(arg)
        return cse_var is not None and not CuteDSLOpOverrides._is_scalar_expr(arg)

    @staticmethod
    def _index_expr(arg: CuteDSLArg) -> sympy.Expr | None:
        cse_var = CuteDSLOpOverrides._get_cse_var(arg)
        if isinstance(cse_var, CuteDSLCSEVariable):
            return cse_var.index_expr
        if isinstance(arg, (int, sympy.Integer)):
            return sympy.Integer(arg)
        if isinstance(arg, str) and arg.lstrip("-").isdigit():
            return sympy.Integer(int(arg))
        return None

    @staticmethod
    def _ensure_tensor_ssa(
        arg: CuteDSLArg,
        template_tensor: CuteDSLArg,
        *,
        is_tensor: bool,
        dtype: torch.dtype | None = None,
    ) -> str:
        """
        Convert scalar arguments to TensorSSA using cute.full_like if needed.

        Args:
            arg: The argument to check (CSEVariable for tensors, str for scalars, or OpsValue wrapper)
            template_tensor: A tensor argument to use as template for full_like

        Returns:
            String representation suitable for CuteDSL operations
        """
        arg_expr = CuteDSLOpOverrides._as_expr(arg)
        if is_tensor:
            return arg_expr
        template_expr = CuteDSLOpOverrides._as_expr(template_tensor)
        if dtype is None:
            return f"cute.full_like({template_expr}, {arg_expr})"
        cute_dtype = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE[dtype]
        return f"cute.full_like({template_expr}, {arg_expr}, {cute_dtype})"

    @staticmethod
    def _extract_dtype_and_bounds(
        *args: CuteDSLArg,
    ) -> tuple[torch.dtype | None, ValueRanges[sympy.Expr]]:
        """Extract dtype and bounds from CSEVariable arguments (including OpsValue wrappers)."""
        for arg in args:
            cse_var = CuteDSLOpOverrides._get_cse_var(arg)
            if cse_var is not None:
                return cse_var.dtype, cse_var.bounds
        return None, ValueRanges.unknown()

    @staticmethod
    def _apply_binary_op(
        a: CuteDSLArg,
        b: CuteDSLArg,
        op_format: str,
        index_expr_fn=None,
    ) -> CuteDSLArg:
        """
        Apply a binary operation with automatic scalar-to-tensor conversion.

        CuteDSL requires both operands to be TensorSSA objects for tensor operations.
        This helper automatically converts scalar arguments to TensorSSA using
        cute.full_like when at least one argument is a tensor (CSEVariable or OpsValue).

        Args:
            a: First operand (CSEVariable for tensors, str for scalars, or OpsValue wrapper)
            b: Second operand (CSEVariable for tensors, str for scalars, or OpsValue wrapper)
            op_format: Format string with {a} and {b} placeholders for the operation

        Returns:
            CSEVariable if at least one operand is a tensor, otherwise string
        """
        a_cse = CuteDSLOpOverrides._get_cse_var(a)
        b_cse = CuteDSLOpOverrides._get_cse_var(b)

        node_flags = CuteDSLOpOverrides._node_tensor_flags()
        if node_flags is not None:
            a_is_tensor, b_is_tensor = node_flags
            a_is_tensor &= not CuteDSLOpOverrides._is_scalar_expr(a)
            b_is_tensor &= not CuteDSLOpOverrides._is_scalar_expr(b)
        else:
            a_is_tensor = CuteDSLOpOverrides._is_tensor_like(a)
            b_is_tensor = CuteDSLOpOverrides._is_tensor_like(b)

        tensor_arg = a if a_is_tensor else (b if b_is_tensor else None)
        if tensor_arg is None:
            tensor_arg = (a_cse if CuteDSLOpOverrides._is_tensor_like(a) else None) or (
                b_cse if CuteDSLOpOverrides._is_tensor_like(b) else None
            )

        if tensor_arg is not None:
            if a_cse is None and b_cse is None:
                return op_format.format(
                    a=CuteDSLOpOverrides._as_expr(a),
                    b=CuteDSLOpOverrides._as_expr(b),
                )

            a_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(
                a, tensor_arg, is_tensor=a_is_tensor
            )
            b_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(
                b, tensor_arg, is_tensor=b_is_tensor
            )
            result_expr = op_format.format(a=a_ssa, b=b_ssa)

            dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(a, b)
            expected = CuteDSLOpOverrides._expected_tensor_val()
            if dtype is None:
                dtype = expected.dtype if expected is not None else torch.int32
            if a_cse is not None:
                shape = a_cse.shape
            elif b_cse is not None:
                shape = b_cse.shape
            else:
                shape = tuple(expected.size()) if expected is not None else None

            result = V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=bounds, dtype=dtype, shape=shape
            )
            if index_expr_fn is not None and isinstance(result, CuteDSLCSEVariable):
                a_expr = CuteDSLOpOverrides._index_expr(a)
                b_expr = CuteDSLOpOverrides._index_expr(b)
                if a_expr is not None and b_expr is not None:
                    result.index_expr = V.graph.sizevars.simplify(
                        index_expr_fn(a_expr, b_expr)
                    )
            return result

        result_expr = op_format.format(
            a=CuteDSLOpOverrides._as_expr(a),
            b=CuteDSLOpOverrides._as_expr(b),
        )
        if a_cse is None and b_cse is None:
            return result_expr

        dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(a, b)
        result = V.kernel.cse.generate(
            V.kernel.body,
            result_expr,
            bounds=bounds,
            dtype=dtype if dtype is not None else torch.int32,
        )
        result.is_scalar_expr = True
        if index_expr_fn is not None and isinstance(result, CuteDSLCSEVariable):
            a_expr = CuteDSLOpOverrides._index_expr(a)
            b_expr = CuteDSLOpOverrides._index_expr(b)
            if a_expr is not None and b_expr is not None:
                result.index_expr = V.graph.sizevars.simplify(
                    index_expr_fn(a_expr, b_expr)
                )
        return result

    @staticmethod
    def _expected_tensor_val() -> torch.Tensor | None:
        """Return the fake-tensor value from the current FX node's metadata, if any."""
        node = V.current_node
        if not isinstance(node, torch.fx.Node):
            return None
        val = node.meta.get("val")
        return val if isinstance(val, torch.Tensor) else None

    @staticmethod
    def _cast_expr(expr: str, dtype: torch.dtype) -> str:
        cute_type = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(dtype)
        if cute_type is None:
            return expr
        return f"{cute_type}({expr})"

    @staticmethod
    def _math_call(name: str, *args: str) -> str:
        fast_math = ", fastmath=True" if _CUTEDSL_FAST_MATH.get() else ""
        return f"cute.math.{name}({', '.join(args)}{fast_math})"

    @staticmethod
    def _apply_unary_op(
        x: CuteDSLArg, op_format: str, index_expr_fn=None
    ) -> CuteDSLArg:
        """
        Apply a unary operation, returning CSEVariable if input is a tensor.

        Args:
            x: Input operand (CSEVariable for tensors, str for scalars, or OpsValue wrapper)
            op_format: Format string with {x} placeholder for the operation
            index_expr_fn: Optional sympy builder to propagate index semantics
                through the op for lane analysis

        Returns:
            CSEVariable if input is a tensor, otherwise string
        """
        cse_var = CuteDSLOpOverrides._get_cse_var(x)
        if cse_var is not None:
            result_expr = op_format.format(x=str(cse_var))
            result = V.kernel.cse.generate(
                V.kernel.body,
                result_expr,
                bounds=cse_var.bounds,
                dtype=cse_var.dtype,
                shape=cse_var.shape,
            )
            if index_expr_fn is not None and isinstance(result, CuteDSLCSEVariable):
                x_expr = CuteDSLOpOverrides._index_expr(x)
                if x_expr is not None:
                    result.index_expr = V.graph.sizevars.simplify(index_expr_fn(x_expr))
            if CuteDSLOpOverrides._is_scalar_expr(x):
                result.is_scalar_expr = True
            return result

        return op_format.format(x=x)

    @staticmethod
    def _apply_math_unary_op(x: CuteDSLArg, name: str) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_unary_op(
            x, CuteDSLOpOverrides._math_call(name, "{x}")
        )

    @staticmethod
    def constant(value: bool | float | int, dtype: torch.dtype) -> str:
        """Generate CuteDSL constant representation."""
        if value == float("-inf"):
            return "float('-inf')"
        elif value == float("inf"):
            return "float('inf')"
        elif math.isnan(value):
            return "float('nan')"
        return repr(value)

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> CuteDSLArg:
        if isinstance(expr, (int, sympy.Integer)):
            return CuteDSLOpOverrides.constant(int(expr), dtype)

        idx_str = V.kernel.kexpr(V.kernel.rename_indexing(expr))
        result = V.kernel.cse.generate(
            V.kernel.body,
            idx_str,
            bounds=get_bounds_index_expr(expr),
            dtype=dtype,
        )
        result.is_scalar_expr = True
        if isinstance(result, CuteDSLCSEVariable):
            result.index_expr = V.graph.sizevars.simplify(expr)
        return result

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> CuteDSLArg:
        # CuteDSL index_expr already emits the requested dtype, so value_expr
        # has the same lowering here.
        return CuteDSLOpOverrides.index_expr(expr, dtype)

    @staticmethod
    def add(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(
            a, b, "({a} + {b})", lambda a_expr, b_expr: a_expr + b_expr
        )

    @staticmethod
    def mul(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(
            a, b, "({a} * {b})", lambda a_expr, b_expr: a_expr * b_expr
        )

    @staticmethod
    def sub(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(
            a, b, "({a} - {b})", lambda a_expr, b_expr: a_expr - b_expr
        )

    @staticmethod
    def truediv(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} / {b})")

    @staticmethod
    def floordiv(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(
            a, b, "({a} // {b})", lambda a_expr, b_expr: a_expr // b_expr
        )

    @staticmethod
    def mod(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        # Inductor ops.mod is C-style; SymPy/Python % is Python-style for
        # negative values, so don't attach index_expr metadata here.
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} % {b})")

    @staticmethod
    def remainder(a, b):
        return CuteDSLOpOverrides._apply_binary_op(
            a, b, "({a} % {b})", lambda a_expr, b_expr: a_expr % b_expr
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp(x: CuteDSLArg) -> CuteDSLArg:
        """Exponential using CuteDSL cute.math.exp2 with log2(e) scaling."""
        if CuteDSLOpOverrides._get_cse_var(x) is None:
            x = CuteDSLOpOverrides._cast_expr(str(x), torch.float32)
        return CuteDSLOpOverrides._apply_unary_op(
            x,
            CuteDSLOpOverrides._math_call(
                "exp2", f"{{x}} * {CuteDSLOpOverrides.LOG2_E}"
            ),
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def exp2(x: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_math_unary_op(x, "exp2")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sqrt(x: CuteDSLArg) -> CuteDSLArg:
        """Square root using CuteDSL cute.math.sqrt function."""
        return CuteDSLOpOverrides._apply_math_unary_op(x, "sqrt")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def rsqrt(x: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_math_unary_op(x, "rsqrt")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log(x: CuteDSLArg) -> CuteDSLArg:
        """Natural logarithm using CuteDSL cute.math.log function."""
        return CuteDSLOpOverrides._apply_math_unary_op(x, "log")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log2(x: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_math_unary_op(x, "log2")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def log10(x: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_math_unary_op(x, "log10")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def cos(x: CuteDSLArg) -> CuteDSLArg:
        """Cosine using CuteDSL cute.math.cos function."""
        return CuteDSLOpOverrides._apply_math_unary_op(x, "cos")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sin(x: CuteDSLArg) -> CuteDSLArg:
        """Sine using CuteDSL cute.math.sin function."""
        return CuteDSLOpOverrides._apply_math_unary_op(x, "sin")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def tan(x: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_math_unary_op(x, "tan")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def acos(x: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_math_unary_op(x, "acos")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def asin(x: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_math_unary_op(x, "asin")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan(x: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_math_unary_op(x, "atan")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan2(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(
            a,
            b,
            CuteDSLOpOverrides._math_call("atan2", "{a}", "{b}"),
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def erf(x: CuteDSLArg) -> CuteDSLArg:
        """Error function using CuteDSL cute.math.erf function."""
        return CuteDSLOpOverrides._apply_math_unary_op(x, "erf")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sigmoid(x: CuteDSLArg) -> CuteDSLArg:
        """Sigmoid with fp32 compute and cast-back to expected output dtype."""
        x_cse = CuteDSLOpOverrides._get_cse_var(x)
        if x_cse is not None:
            x_fp32: CuteDSLArg = CuteDSLOpOverrides.to_dtype(
                x_cse,
                torch.float32,
                use_compute_types=False,
            )
        else:
            x_fp32 = CuteDSLOpOverrides._cast_expr(str(x), torch.float32)

        exp = CuteDSLOpOverrides._math_call(
            "exp2", f"-{{x}} * {CuteDSLOpOverrides.LOG2_E}"
        )
        result = CuteDSLOpOverrides._apply_unary_op(
            x_fp32,
            f"(1.0 / (1.0 + {exp}))",
        )

        expected = CuteDSLOpOverrides._expected_tensor_val()
        expected_dtype = expected.dtype if expected is not None else None
        if expected_dtype is not None and expected_dtype != torch.float32:
            result_cse = CuteDSLOpOverrides._get_cse_var(result)
            if result_cse is not None:
                return CuteDSLOpOverrides.to_dtype(
                    result_cse,
                    expected_dtype,
                    use_compute_types=False,
                )
            return CuteDSLOpOverrides._cast_expr(str(result), expected_dtype)
        return result

    @staticmethod
    def _minmax(a: CuteDSLArg, b: CuteDSLArg, *, op: str) -> CuteDSLArg:
        index_expr_fn = Max if op == ">" else Min
        if CuteDSLOpOverrides._is_tensor_like(a) or CuteDSLOpOverrides._is_tensor_like(
            b
        ):
            return CuteDSLOpOverrides._apply_binary_op(
                a,
                b,
                f"cute.where(({{a}}) {op} ({{b}}), {{a}}, {{b}})",
                index_expr_fn,
            )

        lhs = CuteDSLOpOverrides._as_expr(a)
        rhs = CuteDSLOpOverrides._as_expr(b)
        expected = CuteDSLOpOverrides._expected_tensor_val()
        if expected is not None:
            lhs = CuteDSLOpOverrides._cast_expr(lhs, expected.dtype)
            rhs = CuteDSLOpOverrides._cast_expr(rhs, expected.dtype)
        result_expr = f"({lhs} if {lhs} {op} {rhs} else {rhs})"
        a_cse = CuteDSLOpOverrides._get_cse_var(a)
        b_cse = CuteDSLOpOverrides._get_cse_var(b)
        if a_cse is None and b_cse is None:
            return result_expr
        dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(a, b)
        result = V.kernel.cse.generate(
            V.kernel.body,
            result_expr,
            bounds=bounds,
            dtype=dtype if dtype is not None else torch.int32,
        )
        if isinstance(result, CuteDSLCSEVariable):
            a_expr = CuteDSLOpOverrides._index_expr(a)
            b_expr = CuteDSLOpOverrides._index_expr(b)
            if a_expr is not None and b_expr is not None:
                result.index_expr = V.graph.sizevars.simplify(
                    index_expr_fn(a_expr, b_expr)
                )
        result.is_scalar_expr = True
        return result

    @staticmethod
    # pyrefly: ignore [bad-override]
    def maximum(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._minmax(a, b, op=">")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def minimum(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._minmax(a, b, op="<")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def where(
        condition: CuteDSLArg,
        a: CuteDSLArg,
        b: CuteDSLArg,
    ) -> CuteDSLArg:
        """Conditional selection - handles CSEVariable, OpsValue, and string inputs."""
        a_cse = CuteDSLOpOverrides._get_cse_var(a)
        b_cse = CuteDSLOpOverrides._get_cse_var(b)
        cond_cse = CuteDSLOpOverrides._get_cse_var(condition)
        a_is_tensor = CuteDSLOpOverrides._is_tensor_like(a)
        b_is_tensor = CuteDSLOpOverrides._is_tensor_like(b)
        cond_is_tensor = CuteDSLOpOverrides._is_tensor_like(condition)
        tensor_arg = (
            (a_cse if a_is_tensor else None)
            or (b_cse if b_is_tensor else None)
            or (cond_cse if cond_is_tensor else None)
        )

        if tensor_arg is not None:
            a_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(
                a, tensor_arg, is_tensor=a_is_tensor
            )
            b_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(
                b, tensor_arg, is_tensor=b_is_tensor
            )
            cond_ssa = CuteDSLOpOverrides._ensure_tensor_ssa(
                condition, tensor_arg, is_tensor=cond_is_tensor, dtype=torch.bool
            )
            result_expr = f"cute.where({cond_ssa}, {a_ssa}, {b_ssa})"

            dtype, bounds = CuteDSLOpOverrides._extract_dtype_and_bounds(
                a, b, condition
            )

            return V.kernel.cse.generate(
                V.kernel.body, result_expr, bounds=bounds, dtype=dtype
            )

        result_expr = (
            f"cute.where({CuteDSLOpOverrides._as_expr(condition)}, "
            f"{CuteDSLOpOverrides._as_expr(a)}, {CuteDSLOpOverrides._as_expr(b)})"
        )
        if a_cse is None and b_cse is None and cond_cse is None:
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
    def pow(a: CuteDSLArg, b: CuteDSLArg):
        return CuteDSLOpOverrides._apply_binary_op(a, b, "({a} ** {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def floor(x: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_math_unary_op(x, "floor")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def abs(x: CuteDSLArg) -> CuteDSLArg:
        """Absolute value using CuteDSL cute.math.abs function."""
        if isinstance(x, CSEVariable):
            x_dtype = x.dtype
        elif isinstance(x, OpsValue) and isinstance(x.value, CSEVariable):
            x_dtype = x.value.dtype
        else:
            x_dtype = torch.float32

        abs_op = (
            "mlir_math.absf"
            if x_dtype in (torch.float16, torch.bfloat16, torch.float32)
            else "mlir_math.absi"
        )
        if CuteDSLOpOverrides._is_tensor_like(x):
            return CuteDSLOpOverrides._apply_unary_op(
                x,
                f"cute.TensorSSA({abs_op}({{x}}), {{x}}.shape, {{x}}.dtype)",
                index_expr_fn=sympy.Abs,
            )
        return CuteDSLOpOverrides._apply_unary_op(
            x, f"{abs_op}({{x}})", index_expr_fn=sympy.Abs
        )

    @staticmethod
    def neg(x: CuteDSLArg) -> CuteDSLArg:
        """Negation for both TensorSSA and scalar-like expressions."""
        if CuteDSLOpOverrides._is_tensor_like(x):
            return CuteDSLOpOverrides._apply_unary_op(
                x,
                "cute.TensorSSA(-{x}, {x}.shape, {x}.dtype)",
                index_expr_fn=lambda expr: -expr,
            )
        return CuteDSLOpOverrides._apply_unary_op(
            x, "(-{x})", index_expr_fn=lambda expr: -expr
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def inline_asm_elementwise(
        *inputs: CuteDSLArg,
        asm: str,
        constraints: str | None = None,
        dtype: torch.dtype = torch.float32,
        is_pure: bool = True,
        pack: int = 1,
        input_dtypes: tuple[torch.dtype, ...] | None = None,
        scalar_sources: tuple[bool, ...] | None = None,
    ) -> CuteDSLArg:
        """Emit an inline PTX block elementwise over a fragment.

        The requested dtype controls the logical result and eventual storage.
        E8M0 results may stay decoded as Float32 while fused consumers use them.
        """
        if constraints is None:
            raise NotImplementedError(
                "CuteDSL inline asm requires an explicit constraint list"
            )
        result_type = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(dtype)
        if result_type is None:
            raise NotImplementedError(
                f"CuteDSL inline asm result dtype not supported: {dtype}"
            )

        constraint_parts = [part.strip() for part in constraints.split(",")]
        input_constraints = [
            part.lstrip("=&") for part in constraint_parts if not part.startswith("=")
        ]

        cast_inputs = list(inputs)
        if input_dtypes is not None:
            for source_index, (value, input_dtype) in enumerate(
                zip(inputs, input_dtypes)
            ):
                source_constraints = input_constraints[
                    source_index * pack : (source_index + 1) * pack
                ]
                cse_var = CuteDSLOpOverrides._get_cse_var(value)
                if (
                    source_constraints
                    and all(constraint == "h" for constraint in source_constraints)
                    and input_dtype in (torch.float16, torch.bfloat16)
                    and (cse_var is None or cse_var.dtype != input_dtype)
                ):
                    if cse_var is None:
                        cast_inputs[source_index] = CuteDSLOpOverrides._cast_expr(
                            CuteDSLOpOverrides._as_expr(value), input_dtype
                        )
                    else:
                        cast_inputs[source_index] = CuteDSLOpOverrides.to_dtype(
                            value, input_dtype, use_compute_types=False
                        )

        operands = ", ".join(
            str(CuteDSLOpOverrides._as_expr(value)) for value in cast_inputs
        )
        if scalar_sources is None:
            scalar_sources = (False,) * len(inputs)
        elif len(scalar_sources) != len(inputs):
            raise ValueError(
                f"CuteDSL inline asm received {len(scalar_sources)} scalar-source "
                f"flags for {len(inputs)} inputs"
            )
        scalar_sources = tuple(
            source_is_scalar or CuteDSLOpOverrides._is_scalar_expr(value)
            for source_is_scalar, value in zip(scalar_sources, cast_inputs)
        )
        expr = (
            f"inline_asm_elementwise_intrinsic({operands}, asm={asm!r}, "
            f"constraints={constraints!r}, result_type={result_type}, "
            f"is_pure={is_pure!r}, pack={pack!r}, "
            f"scalar_sources={scalar_sources!r})"
        )
        compute_dtype = torch.float32 if dtype == torch.float8_e8m0fnu else dtype
        if not any(
            CuteDSLOpOverrides._get_cse_var(value) is not None for value in inputs
        ):
            return expr
        return V.kernel.cse.generate(
            V.kernel.body,
            expr,
            dtype=compute_dtype,
        )

    @staticmethod
    def to_dtype(
        x: CuteDSLArg, dtype: torch.dtype, src_dtype=None, use_compute_types=True
    ) -> CuteDSLArg:
        """Type conversion using CuteDSL TensorSSA.to(Type[Numeric]).

        Maps torch dtypes to cutlass.cute.typing numeric types and emits
        `{x}.to(cute.typing.<Type>)`.

        Raises NotImplementedError for unsigned integer and unsupported dtypes.
        """
        if use_compute_types:
            dtype = upcast_compute_type(dtype)

        cute_type = CuteDSLOpOverrides.TORCH_TO_CUTE_DTYPE.get(dtype)
        if cute_type is None:
            raise NotImplementedError(
                f"CuteDSL dtype cast not implemented for torch dtype: {dtype}"
            )

        if isinstance(x, CSEVariable):
            result_expr = f"{str(x)}.to({cute_type})"
            result = V.kernel.cse.generate(
                V.kernel.body,
                result_expr,
                bounds=x.bounds,
                dtype=dtype,
                shape=x.shape,
            )
            if CuteDSLOpOverrides._is_scalar_expr(x):
                result.is_scalar_expr = True
            if (
                dtype in (torch.int32, torch.int64)
                and isinstance(result, CuteDSLCSEVariable)
                and isinstance(x, CuteDSLCSEVariable)
            ):
                result.index_expr = x.index_expr
            return result

        return f"{x}.to({cute_type})"

    @staticmethod
    def tanh(x0: CuteDSLArg) -> CuteDSLArg:
        """Hyperbolic tangent using CuteDSL cute.math.tanh function."""
        return CuteDSLOpOverrides._apply_math_unary_op(x0, "tanh")

    # Logical operations
    @staticmethod
    def logical_and(x0: CuteDSLArg, x1: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x0, x1, "({a} & {b})")

    @staticmethod
    def logical_or(x0: CuteDSLArg, x1: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x0, x1, "({a} | {b})")

    @staticmethod
    def logical_xor(x0: CuteDSLArg, x1: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x0, x1, "({a} ^ {b})")

    # Bitwise operations (override parent class to properly CSE)
    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_and(x: CuteDSLArg, y: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x, y, "({a} & {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_or(x: CuteDSLArg, y: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x, y, "({a} | {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_xor(x: CuteDSLArg, y: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x, y, "({a} ^ {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_not(x: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_unary_op(x, "(~{x})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_left_shift(x: CuteDSLArg, y: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x, y, "({a} << {b})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def bitwise_right_shift(x: CuteDSLArg, y: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(x, y, "({a} >> {b})")

    @staticmethod
    def logical_not(a):
        """Logical NOT."""
        return CuteDSLOpOverrides._apply_unary_op(a, "({x} == 0)")

    # Comparison operations
    @staticmethod
    def eq(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.eq({a}, {b})")

    @staticmethod
    def ne(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.ne({a}, {b})")

    @staticmethod
    def lt(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.lt({a}, {b})")

    @staticmethod
    def le(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.le({a}, {b})")

    @staticmethod
    def gt(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.gt({a}, {b})")

    @staticmethod
    def ge(a: CuteDSLArg, b: CuteDSLArg) -> CuteDSLArg:
        return CuteDSLOpOverrides._apply_binary_op(a, b, "operator.ge({a}, {b})")
<|endoftext|># mypy: allow-untyped-defs

import dataclasses
from collections.abc import Callable
from typing import ClassVar

import sympy


@dataclasses.dataclass(frozen=True, slots=True)
class CuteDSLAuxScalarBindings:
    """Render symbolic shape captures through FA4's runtime aux_scalars tuple.

    Inductor represents captured dynamic ints/floats as SymPy expressions in the
    other-buffer list, while tensor captures remain TensorBox-backed aux_tensors.
    CuTe kernels receive those scalar values as ordinary kernel arguments, wrap
    them for FA4, and rewrite matching symbols in generated score_mod/mask_mod
    expressions to tuple lookups.
    """

    symbols: tuple[sympy.Symbol, ...] = ()
    tuple_name: ClassVar[str] = "aux_scalars"

    def symbol_codes(self) -> dict[sympy.Symbol, str]:
        """Render symbols as tuple lookups for CuTe expressions."""
        return {
            symbol: f"{self.tuple_name}[{index}]"
            for index, symbol in enumerate(self.symbols)
        }

    def symbol_codes_with_renames(
        self, rename: Callable[[sympy.Symbol], sympy.Expr]
    ) -> dict[sympy.Symbol, str]:
        """Include Inductor-renamed symbols used in generated kernel signatures."""
        codes = self.symbol_codes()
        for symbol, code in list(codes.items()):
            renamed = rename(symbol)
            if isinstance(renamed, sympy.Symbol):
                codes[renamed] = code
        return codes

    def tuple_expr(
        self,
        rename: Callable[[sympy.Symbol], sympy.Expr],
        print_expr: Callable[[sympy.Expr], str],
    ) -> str:
        """Render runtime scalar values with FA4-compatible scalar wrapper types."""
        if not self.symbols:
            return "None"
        scalar_values = []
        for symbol in self.symbols:
            scalar_type = "cutlass.Int64" if symbol.is_integer else "cutlass.Float64"
            scalar_values.append(f"{scalar_type}({print_expr(rename(symbol))})")
        if len(scalar_values) == 1:
            return f"({scalar_values[0]},)"
        return f"({', '.join(scalar_values)})"
<|endoftext|># mypy: allow-untyped-defs
"""
Utility functions for NVIDIA Universal GEMM.
"""

import copyreg
from typing import Any

from torch.nn.functional import ScalingType, SwizzleType


def _rebuild_scaling_type(name: str) -> Any:
    return getattr(ScalingType, name)


def _rebuild_swizzle_type(name: str) -> Any:
    return getattr(SwizzleType, name)


def _register_enum_pickling() -> None:
    """Make torch.nn.functional ScalingType/SwizzleType picklable.

    They are pybind11 C++ enums whose class __qualname__ is `_ScalingType` /
    `_SwizzleType`, but torch.nn.functional only binds the public names
    `ScalingType` / `SwizzleType`. Pickle's default reduction reconstructs by
    the private qualname and fails, which blocks shipping a scaled-GEMM
    benchmark request to the autotuning subprocess pool. Reduce by `.name`
    string via the public alias instead (no class object in the pickle).
    """
    copyreg.pickle(
        type(ScalingType.BlockWise1x16),
        lambda v: (_rebuild_scaling_type, (v.name,)),
    )
    copyreg.pickle(
        type(SwizzleType.NO_SWIZZLE),
        lambda v: (_rebuild_swizzle_type, (v.name,)),
    )


_register_enum_pickling()


def _rebuild_scaled_operand_constraints(
    quantized: Any, scale: Any, mode: Any, swizzle: Any
) -> Any:
    from cutlass.operators.metadata import ScaledOperandConstraints

    return ScaledOperandConstraints(quantized, scale, mode, swizzle)


def _register_scaled_operand_constraints_pickling() -> None:
    """Avoid CUTLASS's recursive ``__getattr__`` during unpickling."""
    try:
        from cutlass.operators.metadata import ScaledOperandConstraints
    except ImportError:
        return

    copyreg.pickle(
        ScaledOperandConstraints,
        lambda value: (
            _rebuild_scaled_operand_constraints,
            (value.quantized, value.scale, value.mode, value.swizzle),
        ),
    )


_register_scaled_operand_constraints_pickling()


def to_cutlass_scale_mode(
    scale_type: Any, swizzle_type: Any
) -> tuple[Any | None, Any | None]:
    """
    Map PyTorch ScalingType/SwizzleType to cutlass.operators ScaleMode/ScaleSwizzleMode.

    Args:
        scale_type: ScalingType from torch.nn.functional
        swizzle_type: SwizzleType from torch.nn.functional

    Returns:
        Tuple of (ScaleMode, ScaleSwizzleMode) from cutlass.operators,
        or (None, None) if the types are not supported.

    The returned enum objects can be used directly with cutlass.operators, or their
    .name attribute can be used for codegen (e.g., scale_mode.name -> "Blockwise1x32").

    NOTE:
        Currently on Blackwell (SM100), NVGEMM only supports MXFP8 scaling modes.
        Update this mapping when additional scaling modes are added.
    """
    from cutlass.operators import ScaleMode, ScaleSwizzleMode

    scale_mode_map = {
        ScalingType.BlockWise1x32: ScaleMode.Blockwise1x32,
        ScalingType.BlockWise1x16: ScaleMode.Blockwise1x16,
    }
    swizzle_mode_map = {
        SwizzleType.SWIZZLE_32_4_4: ScaleSwizzleMode.Swizzle32x4x4,
        SwizzleType.NO_SWIZZLE: ScaleSwizzleMode.SwizzleNone,
    }
    return scale_mode_map.get(scale_type), swizzle_mode_map.get(swizzle_type)


# NOTE: cutlass.torch.dtype() doesn't support Float4E2M1FN (raises TypeError),
# so we maintain our own mapping that includes FP4.
_CUTLASS_TO_TORCH_DTYPE: dict | None = None


def cutlass_dtype_to_torch(cutlass_dtype: Any) -> "Any | None":
    """Map a cutlass dtype to the corresponding torch dtype."""
    import torch

    global _CUTLASS_TO_TORCH_DTYPE
    if _CUTLASS_TO_TORCH_DTYPE is None:
        import cutlass

        _CUTLASS_TO_TORCH_DTYPE = {
            cutlass.Float4E2M1FN: torch.float4_e2m1fn_x2,
            cutlass.Float8E4M3FN: torch.float8_e4m3fn,
            cutlass.Float8E5M2: torch.float8_e5m2,
            cutlass.Float8E8M0FNU: torch.float8_e8m0fnu,
            cutlass.BFloat16: torch.bfloat16,
            cutlass.Float16: torch.float16,
            cutlass.Float32: torch.float32,
        }
    return _CUTLASS_TO_TORCH_DTYPE.get(cutlass_dtype)
<|endoftext|># mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM scheduling for PyTorch Inductor.
"""

import dataclasses
import hashlib
import logging
from collections.abc import Sequence
from typing import Any, cast

from torch._inductor.utils import (
    get_fused_kernel_name,
    get_kernel_metadata,
    Placeholder,
)
from torch.utils._ordered_set import OrderedSet

from ... import config
from ...codecache import code_hash, get_path
from ...ir import (
    Buffer,
    ComputedBuffer,
    Layout,
    MultiTemplateBuffer,
    NVUniversalGemmBuffer,
    Pointwise,
    Reduction,
)
from ...kernel.gemm_epilogue import GemmReductionConfig, GemmReductionPlan
from ...kernel.loop_ir_cutedsl_codegen import LoopIRCuteDSLCodegen
from ...kernel.loop_ir_epilogue_lowering import (
    centered_mean_consumer_type_unrolled_ir,
    GemmEpilogueIRAnalysis,
    grouped_reduction_ir,
    single_source_affine_ir,
)
from ...scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    FusedSchedulerNode,
    SchedulerNode,
)
from ...virtualized import V
from ..common import BackendFeature, IndentedBuffer
from ..cutlass.python_evt import _ACCUMULATOR_ARG_NAME, CutlassEVTCodegen
from .nv_universal_gemm import NVUniversalGemmCaller


log = logging.getLogger(__name__)

MAIN_SUFFIX = "main"
_BENCHMARK_KERNEL_PREFIX = "nv_gemm_"
EPILOGUE_FN_NAME = "_epilogue_fn"


@dataclasses.dataclass(frozen=True)
class NVGemmEpiloguePlan:
    nodes: tuple[BaseSchedulerNode, ...]
    reductions: tuple[GemmReductionConfig, ...]
    reduction_nodes: tuple[BaseSchedulerNode, ...]
    candidate_reduction: GemmReductionConfig | None = None
    feed_main: GemmReductionConfig | None = None

    @property
    def evt_nodes(self) -> list[BaseSchedulerNode]:
        owned = OrderedSet(self.reduction_nodes)
        return [node for node in self.nodes if node not in owned]


class NVUniversalGemmScheduling(BaseScheduling):
    """
    Scheduling implementation for NVIDIA Universal GEMM kernels.

    This class is intended to be used in combination with other schedulers,
    and delegated to by CUDACombinedScheduling.
    """

    @classmethod
    def get_backend_features(cls, device) -> OrderedSet[BackendFeature]:
        return OrderedSet()

    @staticmethod
    def _is_nvgemm_ir_buffer(ir_node: Any) -> bool:
        """Return True if `ir_node` is an NVGEMM buffer or an MTB resolving to one.

        Honors finalize_as_*_caller's swap (a MultiTemplateBuffer whose render
        kind is "triton" is no longer NVGEMM, even if the autotune winner was).
        Falls back to the autotune winner only when no swap has happened.
        """
        if isinstance(ir_node, NVUniversalGemmBuffer):
            return True
        if not isinstance(ir_node, MultiTemplateBuffer):
            return False
        if ir_node._render_kind == "triton":
            return False
        if ir_node._render_kind == "nvgemm":
            return True
        # Fast path: avoid forcing autotune just to answer this query.
        if not any(isinstance(c, NVUniversalGemmCaller) for c in ir_node._choices):
            return False
        try:
            min_choice, _ = ir_node.get_min_choice()
            return isinstance(min_choice, NVUniversalGemmCaller)
        except (RuntimeError, ValueError):
            return False

    @staticmethod
    def is_nv_universal_gemm_template(node: BaseSchedulerNode) -> bool:
        """Check if a node is an NVGEMM template SchedulerNode."""
        if not isinstance(node, SchedulerNode):
            return False
        return NVUniversalGemmScheduling._is_nvgemm_ir_buffer(node.node)

    @staticmethod
    def _best_nvgemm_choice(
        ir_node: MultiTemplateBuffer,
        require_epilogue_fusion: bool = False,
    ) -> NVUniversalGemmCaller:
        """Find the best NVUniversalGemmCaller from an MTB's choice timings."""
        choice_timings = ir_node.choice_timings()
        best: NVUniversalGemmCaller | None = None
        best_time = float("inf")
        for choice, timing in choice_timings.items():
            if not isinstance(choice, NVUniversalGemmCaller):
                continue
            if require_epilogue_fusion and not choice.supports_epilogue_fusion:
                continue
            if best is None or timing < best_time:
                best_time = timing
                best = choice
        if best is None:
            kind = "EFC kernel" if require_epilogue_fusion else "NVUniversalGemmCaller"
            raise RuntimeError(f"No {kind} found in choices")
        return best

    @staticmethod
    def get_nv_gemm_buffer_from_node(
        node: BaseSchedulerNode, require_epilogue_fusion: bool = False
    ) -> NVUniversalGemmBuffer:
        """Extract NVUniversalGemmBuffer from node (direct or via MultiTemplateBuffer)."""
        if not isinstance(node, SchedulerNode):
            raise AssertionError(f"expected SchedulerNode, got {type(node)}")
        ir_node = node.node

        if isinstance(ir_node, NVUniversalGemmBuffer):
            return ir_node
        elif isinstance(ir_node, MultiTemplateBuffer):
            # Honor an explicit swap/finalize -- the fusion benchmark loop swaps
            # in each EFC choice one at a time and must not re-select from timings.
            if isinstance(ir_node._render_caller, NVUniversalGemmCaller) and (
                not require_epilogue_fusion
                or ir_node._render_caller.supports_epilogue_fusion
            ):
                selected_choice = ir_node._render_caller
            elif require_epilogue_fusion:
                selected_choice = NVUniversalGemmScheduling._best_nvgemm_choice(
                    ir_node, require_epilogue_fusion=True
                )
            else:
                min_choice, _ = ir_node.get_min_choice()
                if isinstance(min_choice, NVUniversalGemmCaller):
                    selected_choice = min_choice
                else:
                    selected_choice = NVUniversalGemmScheduling._best_nvgemm_choice(
                        ir_node
                    )
            tensor_box = selected_choice.output_node()
            # pyrefly: ignore [missing-attribute]
            return cast(NVUniversalGemmBuffer, tensor_box.data.data)

        raise TypeError(
            f"Expected NVUniversalGemmBuffer or MultiTemplateBuffer, got {type(ir_node).__name__}"
        )

    @staticmethod
    def is_nv_universal_gemm_fused_template(node: BaseSchedulerNode) -> bool:
        """Check if a node is a fused NVIDIA Universal GEMM template."""
        if not isinstance(node, FusedSchedulerNode):
            return False
        return NVUniversalGemmScheduling._is_nvgemm_ir_buffer(node.get_template_node())

    def can_fuse_vertical(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if self.is_nv_universal_gemm_template(node1):
            return self._can_fuse_epilogue_impl(
                cast(SchedulerNode, node1),
                [],
                node2,
            )
        elif self.is_nv_universal_gemm_fused_template(node1):
            fnode1 = cast(FusedSchedulerNode, node1)
            template_snode = next(
                (n for n in fnode1.snodes if self.is_nv_universal_gemm_template(n)),
                None,
            )
            if template_snode is None:
                return False
            return self._can_fuse_epilogue_impl(
                cast(SchedulerNode, template_snode),
                self._unwrap_epilogue_nodes(fnode1),
                node2,
            )
        return False

    def _unwrap_epilogue_nodes(
        self, fused_node: FusedSchedulerNode
    ) -> list[BaseSchedulerNode]:
        """Extract epilogue nodes from a fused node."""
        epilogue_nodes = []
        for node in fused_node.snodes:
            if not self.is_nv_universal_gemm_template(node):
                epilogue_nodes.append(node)
        return epilogue_nodes

    @classmethod
    def _grouped_reduce_config(
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> GemmReductionConfig | None:
        nodes = scheduler_node.get_nodes()
        if len(nodes) not in (1, 2):
            return None
        buffers = [snode.node for snode in nodes]
        if not all(isinstance(buffer, ComputedBuffer) for buffer in buffers):
            return None
        buffers = [cast(ComputedBuffer, buffer) for buffer in buffers]
        node = buffers[0]
        access_node = scheduler_node
        output_name = node.get_name()
        if len(node.data.ranges) != 2 or len(gemm_node.get_size()) != 2:
            return None
        try:
            m, n = (V.graph.sizevars.optimization_hint(v) for v in gemm_node.get_size())
            out_m, out_n = (
                V.graph.sizevars.optimization_hint(v) for v in node.data.ranges
            )
        except Exception:
            return None

        if isinstance(node.data, Reduction):
            reduction = node.data
            if len(reduction.reduction_ranges) != 1:
                return None
            group = V.graph.sizevars.optimization_hint(reduction.reduction_ranges[0])
            if m == out_m and n % group == 0 and out_n == n // group:
                axis = 1
                expected_strides = [n, group, 1]
                max_group = n
            elif n == out_n and m % group == 0 and out_m == m // group:
                axis = 0
                expected_strides = [group * n, 1, n]
                max_group = 16
            else:
                return None
        elif isinstance(node.data, Pointwise):
            if n != out_n or out_m <= 0 or m % out_m != 0:
                return None
            group = m // out_m
            axis = 0
            max_group = 16
            expected_strides = [group * n, 1]
        else:
            return None
        if group <= 1 or group > max_group:
            return None

        store = GemmEpilogueIRAnalysis.store_from_buffer(node)
        classified = (
            grouped_reduction_ir(
                store,
                gemm_node.get_name(),
                group,
                gemm_node.get_dtype(),
            )
            if store is not None
            else None
        )
        if classified is None:
            return None
        reduction_type, source_type = classified
        if reduction_type not in (
            "sum",
            "mean",
            "prod",
            "max",
            "min",
        ) or source_type not in ("identity", "square"):
            return None
        if (
            isinstance(node.data, Reduction)
            and node.data.reduction_type != reduction_type
        ):
            return None
        if len(buffers) == 2:
            finalizer = buffers[1]
            finalizer_store = GemmEpilogueIRAnalysis.store_from_buffer(finalizer)
            finalizer_reads = list(nodes[1].read_writes.reads)
            if (
                reduction_type != "sum"
                or not isinstance(node.data, Reduction)
                or not isinstance(finalizer.data, Pointwise)
                or not finalizer_reads
                or any(read.name != node.get_name() for read in finalizer_reads)
                or not V.graph.sizevars.statically_known_list_equals(
                    node.get_size(), finalizer.get_size()
                )
                or finalizer_store is None
                or single_source_affine_ir(finalizer_store, node.get_name())
                != (1.0 / group, 0.0)
            ):
                return None
            reduction_type = "mean"
            access_node = nodes[0]
            output_name = finalizer.get_name()

        reads = list(access_node.read_writes.reads)
        if not reads or any(read.name != gemm_node.get_name() for read in reads):
            return None
        range_vars = access_node.read_writes.range_vars
        if range_vars is None:
            return None
        if not range_vars:
            return GemmReductionConfig(
                output_name, group, axis, reduction_type, source_type
            )
        if isinstance(node.data, Reduction):
            if len(reads) != 1:
                return None
            strides = V.graph.sizevars.stride_vars(reads[0].index, range_vars)
            if list(strides) != expected_strides:
                return None
        else:
            if len(reads) != group:
                return None
            expected_base = group * n * range_vars[0] + range_vars[1]
            offsets = []
            for read in reads:
                strides = V.graph.sizevars.stride_vars(read.index, range_vars)
                if list(strides) != expected_strides:
                    return None
                offsets.append(V.graph.sizevars.simplify(read.index - expected_base))
            expected_offsets = OrderedSet(offset * n for offset in range(group))
            if OrderedSet(offsets) != expected_offsets:
                return None
        return GemmReductionConfig(
            output_name, group, axis, reduction_type, source_type
        )

    @classmethod
    def _grouped_reduce_feeds_main_config(
        cls, gemm_node: Buffer, scheduler_node: BaseSchedulerNode
    ) -> GemmReductionConfig | None:
        nodes = scheduler_node.get_nodes()
        if len(nodes) != 1 or not isinstance(nodes[0].node, ComputedBuffer):
            return None
        node = cast(ComputedBuffer, nodes[0].node)
        if not isinstance(node.data, Pointwise):
            return None
        reads = list(scheduler_node.read_writes.reads)
        range_vars = scheduler_node.read_writes.range_vars
        if range_vars is None or len(reads) <= 2:
            return None
        try:
            _, n = map(V.graph.sizevars.optimization_hint, gemm_node.get_size())
        except Exception:
            return None
        group = len(reads) - 1
        if group <= 1 or group > 4:
            return None
        store = GemmEpilogueIRAnalysis.store_from_buffer(node)
        if (
            store is None
            or centered_mean_consumer_type_unrolled_ir(
                store, gemm_node.get_name(), group
            )
            != "mean_linear:1:-1:0"
        ):
            return None
        if any(read.name != gemm_node.get_name() for read in reads):
            return None
        grouped_base = reads[1].index
        if any(
            V.graph.sizevars.simplify(read.index - grouped_base) != i * n
            for i, read in enumerate(reads[1:])
        ):
            return None
        return GemmReductionConfig(node.get_name(), group, 0, "mean", "identity")

    @classmethod
    def _epilogue_plan(
        cls,
        gemm_node: Buffer,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        candidate: BaseSchedulerNode | None = None,
    ) -> NVGemmEpiloguePlan:
        reductions: list[GemmReductionConfig] = []
        reduction_nodes: list[BaseSchedulerNode] = []
        candidate_reduction = None
        for node in epilogue_nodes:
            config = cls._grouped_reduce_config(gemm_node, node)
            if config is None:
                continue
            reductions.append(config)
            reduction_nodes.extend(node.get_nodes())
            if node is candidate:
                candidate_reduction = config
        nodes = tuple(
            scheduler_node
            for node in epilogue_nodes
            for scheduler_node in node.get_nodes()
        )
        feed_main = (
            cls._grouped_reduce_feeds_main_config(gemm_node, epilogue_nodes[0])
            if len(epilogue_nodes) == 1
            else None
        )
        return NVGemmEpiloguePlan(
            nodes,
            tuple(reductions),
            tuple(reduction_nodes),
            candidate_reduction,
            feed_main,
        )

    def _can_fuse_epilogue_impl(
        self,
        gemm_template_node: SchedulerNode,
        existing_epilogue_nodes: list[BaseSchedulerNode],
        node_to_fuse: BaseSchedulerNode,
    ) -> bool:
        from .nv_universal_gemm import GemmVariant

        if not config.epilogue_fusion:
            return False

        ir_node = gemm_template_node.node
        if not isinstance(ir_node, (NVUniversalGemmBuffer, MultiTemplateBuffer)):
            return False

        if isinstance(ir_node, NVUniversalGemmBuffer):
            if ir_node.variant not in (GemmVariant.GEMM, GemmVariant.SCALED_GEMM):
                log.debug(
                    "NVGEMM epilogue fusion: not supported for %s variant",
                    ir_node.variant.op_name,
                )
                return False
            if not ir_node.supports_epilogue_fusion:
                log.debug(
                    "NVGEMM epilogue fusion: kernel %s does not support epilogue fusion",
                    ir_node.kernel_metadata.get("kernel_name", "unknown"),
                )
                return False
        elif isinstance(ir_node, MultiTemplateBuffer):
            # Use _choices, not choice_timings() — the latter forces autotune sync.
            has_efc_choice = False
            for choice in ir_node._choices:
                if not (
                    isinstance(choice, NVUniversalGemmCaller)
                    and choice.supports_epilogue_fusion
                ):
                    continue
                has_efc_choice = True
                if choice.variant not in (
                    GemmVariant.GEMM,
                    GemmVariant.SCALED_GEMM,
                ):
                    log.debug(
                        "NVGEMM epilogue fusion: MultiTemplateBuffer has unsupported EFC choices"
                    )
                    return False
            if not has_efc_choice:
                log.debug("NVGEMM epilogue fusion: no EFC kernel available in choices")
                return False

        plan = self._epilogue_plan(
            ir_node, (*existing_epilogue_nodes, node_to_fuse), node_to_fuse
        )
        reduction_nodes = OrderedSet(plan.reduction_nodes)
        local_reduce = plan.candidate_reduction
        feed_main = plan.feed_main
        variants = (
            (ir_node.variant,)
            if isinstance(ir_node, NVUniversalGemmBuffer)
            else tuple(
                choice.variant
                for choice in ir_node._choices
                if isinstance(choice, NVUniversalGemmCaller)
                and choice.supports_epilogue_fusion
            )
        )
        scaled_epilogue = bool(variants) and all(
            variant == GemmVariant.SCALED_GEMM for variant in variants
        )
        if local_reduce is not None:
            if not scaled_epilogue:
                return False

        for s_node in plan.nodes:
            node = s_node.node
            if not isinstance(node, ComputedBuffer):
                log.debug("NVGEMM epilogue fusion: %s is not a ComputedBuffer", node)
                return False
            if (
                feed_main is None
                and not isinstance(node.data, Pointwise)
                and s_node not in reduction_nodes
            ):
                log.debug("NVGEMM epilogue fusion: %s is not a Pointwise op", node)
                return False

            if (
                feed_main is None
                and s_node not in reduction_nodes
                and not V.graph.sizevars.statically_known_list_equals(
                    node.get_size(), ir_node.get_size()
                )
            ):
                log.debug(
                    "NVGEMM epilogue fusion: size mismatch %s vs %s",
                    node.get_size(),
                    ir_node.get_size(),
                )
                return False
        # cutlass.operators' EVT supports matrix, row, and column loads here.
        # Reject unresolved inputs and shapes outside those broadcast patterns;
        # the trial EVT trace below remains the final capability check.
        gemm_size = ir_node.get_size()
        name_to_buf = V.graph.name_to_buffer | V.graph.graph_inputs
        internal_names = OrderedSet([ir_node.get_name()]) | OrderedSet(
            [s_node.get_name() for s_node in plan.nodes]
        )
        epilogue_inputs: OrderedSet[str] = OrderedSet()
        for s_node in plan.nodes:
            for rd in s_node.read_writes.reads:
                if rd.name in internal_names:
                    continue
                epilogue_inputs.add(rd.name)
                read_buf = name_to_buf.get(rd.name)
                if read_buf is None:
                    log.debug(
                        "NVGEMM epilogue fusion: read %s not in name_to_buffer/graph_inputs, refusing to fuse",
                        rd.name,
                    )
                    return False
                read_size = read_buf.get_size()
                if not read_size or len(read_size) > len(gemm_size):
                    log.debug(
                        "NVGEMM epilogue fusion: read buffer %s has unsupported rank",
                        rd.name,
                    )
                    return False
                padded_size = [1] * (len(gemm_size) - len(read_size)) + list(read_size)
                supported_shapes = (
                    gemm_size,
                    [1, gemm_size[1]],
                    [gemm_size[0], 1],
                )
                if not any(
                    V.graph.sizevars.statically_known_list_equals(padded_size, shape)
                    for shape in supported_shapes
                ):
                    log.debug(
                        "NVGEMM epilogue fusion: read buffer %s size %s is not broadcastable to GEMM size %s",
                        rd.name,
                        read_size,
                        gemm_size,
                    )
                    return False
        if scaled_epilogue and len(epilogue_inputs) > 4:
            log.debug(
                "NVGEMM scaled epilogue has %d captured tensors; at most four are supported",
                len(epilogue_inputs),
            )
            return False

        if not existing_epilogue_nodes:
            reads = OrderedSet(rd.name for rd in node_to_fuse.read_writes.reads)
            if ir_node.get_name() not in reads:
                log.debug(
                    "NVGEMM epilogue fusion: first epilogue node doesn't read from GEMM output"
                )
                return False

        if node_to_fuse.has_aliasing_or_mutation():
            log.debug("NVGEMM epilogue fusion: node has aliasing or mutation")
            return False
        elif node_to_fuse.is_reduction() and local_reduce is None:
            log.debug("NVGEMM epilogue fusion: reductions not supported")
            return False

        all_epilogue_nodes = list(plan.nodes)
        fused_buffer_names = OrderedSet(
            n.get_name() for n in [gemm_template_node, *all_epilogue_nodes]
        )
        preserve_gemm_output = (
            not V.graph.scheduler.can_buffer_be_removed_through_fusion(
                ir_node.get_name(), fused_buffer_names
            )
        )
        # Multi-store epilogues wire each output to its own destination tensor.
        trial_removed_buffers = V.graph.removed_buffers.copy()
        if not preserve_gemm_output:
            trial_removed_buffers.add(ir_node.get_name())
        try:
            evt_nodes = plan.evt_nodes
            trial_reads: list[str] = []
            trial_writes: list[str] = []
            if feed_main is not None:
                evt_nodes = []
            if evt_nodes:
                trial_reads, trial_writes, _, _ = (
                    CutlassEVTCodegen.ir_to_evt_python_code(
                        ir_node.get_name(),
                        evt_nodes,
                        trial_removed_buffers,
                    )
                )
            if scaled_epilogue:
                if len(trial_reads) > 4 or len(trial_writes) > 4:
                    return False
                for read_name in trial_reads:
                    read_buf = name_to_buf.get(read_name)
                    if read_buf is None:
                        log.debug(
                            "NVGEMM scaled EVT input %s cannot be resolved",
                            read_name,
                        )
                        return False
        except (NotImplementedError, AssertionError) as e:
            log.debug("NVGEMM epilogue fusion: trial EVT codegen failed: %s", e)
            return False

        return True

    def can_fuse_horizontal(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        # NVIDIA Universal GEMM templates don't support horizontal fusion yet
        return False

    def can_fuse_reduction_epilogue(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        template = node1.get_template_node()
        return isinstance(template, Buffer) and (
            self._grouped_reduce_config(template, node2) is not None
            or self._grouped_reduce_feeds_main_config(template, node2) is not None
        )

    def define_kernel(
        self, src_code: str, node_schedule, precompile_metadata=None
    ) -> str:
        """
        Define a NVIDIA Universal GEMM kernel by writing source code and generating wrapper.

        Based on CuteDSLScheduling.define_kernel.
        """
        wrapper = V.graph.wrapper_code

        # Use the string as the key for caching
        if src_code in wrapper.src_to_kernel:
            return wrapper.src_to_kernel[src_code]

        fused_name = (
            get_fused_kernel_name(node_schedule, config.triton.descriptive_names)
            if config.triton.descriptive_names
            else ""
        )

        kernel_hash = hashlib.sha256(src_code.encode("utf-8")).hexdigest()[:8]
        if fused_name == "fused":
            kernel_name = f"nv_universal_gemm_{kernel_hash}"
        else:
            kernel_name = f"nv_universal_gemm_{fused_name}_{kernel_hash}"

        wrapper.src_to_kernel[src_code] = kernel_name

        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_name)

        _, _, kernel_path = get_path(code_hash(src_code), "py")

        compile_wrapper = IndentedBuffer()
        compile_wrapper.writeline(
            f"async_compile.nv_universal_gemm({kernel_name!r}, r'''"
        )
        compile_wrapper.splice(src_code, strip=True)
        if precompile_metadata is not None:
            compile_wrapper.writeline(
                f"''', precompile_metadata={precompile_metadata!r})"
            )
        else:
            compile_wrapper.writeline("''')")

        metadata_comment = f"# kernel path: {kernel_path}"
        origins, detailed_origins = get_kernel_metadata(node_schedule, wrapper)
        metadata_comment += "\n" + origins + "\n" + detailed_origins
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), metadata_comment)

        return kernel_name

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
        *,
        only_gen_src_code: bool = False,
    ) -> str | None:
        """
        Codegen a NVIDIA Universal GEMM template with optional epilogue fusion.

        If `only_gen_src_code=True` the src code will be returned instead of being
        codegenned into the wrapper (used for benchmarking).
        """
        log.debug(
            "NVGEMM codegen_template: template_node=%s, epilogue_nodes=%s, prologue_nodes=%s",
            template_node,
            [n.get_name() for n in epilogue_nodes] if epilogue_nodes else [],
            [n.get_name() for n in prologue_nodes] if prologue_nodes else [],
        )
        if not self.is_nv_universal_gemm_template(template_node):
            raise AssertionError(
                "Template node passed to NVUniversalGemmScheduling.codegen_template must be a "
                "SchedulerNode that wraps a NVUniversalGemmBuffer or MultiTemplateBuffer with NVGEMM choice"
            )
        if prologue_nodes:
            raise AssertionError(
                "NVIDIA Universal GEMM doesn't support prologue fusion yet"
            )

        template_node = cast(SchedulerNode, template_node)

        original_ir_node = template_node.node
        if not isinstance(original_ir_node, Buffer):
            raise AssertionError(f"expected Buffer, got {type(original_ir_node)}")
        original_buffer_name = original_ir_node.get_name()

        ctb: NVUniversalGemmBuffer = self.get_nv_gemm_buffer_from_node(
            template_node, require_epilogue_fusion=bool(epilogue_nodes)
        )

        epilogue_fn_code: str | None = None
        epilogue_reads: list[str] = []
        epilogue_writes: list[str] = []
        epilogue_var_renames: dict[str, Any] = {}
        local_reduce: GemmReductionPlan | None = None

        if epilogue_nodes:
            scheduler = V.graph.scheduler
            try:
                plan = self._epilogue_plan(original_ir_node, epilogue_nodes)
                if len(plan.reductions) > 1:
                    raise NotImplementedError(
                        "NVGEMM supports one grouped local reduction"
                    )
                if plan.reductions:
                    reduction = plan.reductions[0]
                    local_reduce = GemmReductionPlan(
                        reduction_output=reduction.output_name,
                        group=reduction.group,
                        axis=reduction.axis,
                        reduction_type=reduction.reduction_type,
                        source_type=reduction.source_type,
                        primary_output=original_buffer_name,
                    )
                feed_main = plan.feed_main
                if feed_main is not None:
                    local_reduce = GemmReductionPlan(
                        reduction_output=None,
                        group=feed_main.group,
                        axis=feed_main.axis,
                        reduction_type=feed_main.reduction_type,
                        source_type=feed_main.source_type,
                        primary_output=feed_main.output_name,
                        feeds_main=True,
                    )
                    epilogue_fn_code = (
                        f"def {EPILOGUE_FN_NAME}(accum):\n    D = accum\n    return D"
                    )
                    epilogue_writes = [feed_main.output_name]
                    epilogue_var_renames = {
                        _ACCUMULATOR_ARG_NAME: original_buffer_name,
                        "D": feed_main.output_name,
                    }
                evt_nodes = [] if feed_main is not None else plan.evt_nodes
                fused_buffer_names: OrderedSet[str] = OrderedSet(
                    n.get_name() for n in epilogue_nodes
                )
                fused_buffer_names.add(original_buffer_name)
                removed_buffers_with_gemm = V.graph.removed_buffers.copy()
                if scheduler.can_buffer_be_removed_through_fusion(
                    original_buffer_name, fused_buffer_names
                ):
                    removed_buffers_with_gemm.add(original_buffer_name)

                if evt_nodes:
                    evt_buffers = [
                        node.node
                        for node in evt_nodes
                        if isinstance(node.node, ComputedBuffer)
                    ]
                    try:
                        reads, writes, var_renames, evt_code = (
                            LoopIRCuteDSLCodegen.from_buffers(
                                original_buffer_name,
                                evt_buffers,
                                removed_buffers_with_gemm,
                                EPILOGUE_FN_NAME,
                            )
                        )
                    except NotImplementedError:
                        reads, writes, var_renames, evt_code = (
                            CutlassEVTCodegen.ir_to_evt_python_code(
                                original_buffer_name,
                                list(evt_nodes),
                                removed_buffers_with_gemm,
                                fn_name=EPILOGUE_FN_NAME,
                                as_standalone_function=True,
                            )
                        )
                    epilogue_fn_code = evt_code
                    epilogue_reads = reads
                    epilogue_writes = writes
                    epilogue_var_renames = var_renames

                if not only_gen_src_code:
                    write_bufs = OrderedSet(epilogue_writes)
                    if local_reduce is not None:
                        write_bufs.update(local_reduce.auxiliary_outputs)
                    for node in epilogue_nodes:
                        node_name = node.get_name()
                        if node_name in write_bufs:
                            continue
                        if scheduler.can_buffer_be_removed_through_fusion(
                            node_name, fused_buffer_names
                        ):
                            V.graph.removed_buffers.add(node_name)
                    if (
                        original_buffer_name not in write_bufs
                        and scheduler.can_buffer_be_removed_through_fusion(
                            original_buffer_name, fused_buffer_names
                        )
                    ):
                        V.graph.removed_buffers.add(original_buffer_name)
                    for node in epilogue_nodes:
                        node.mark_run()

                log.debug(
                    "NVGEMM epilogue fusion: %d nodes, reads=%s, writes=%s, local_reduce=%s",
                    len(epilogue_nodes),
                    epilogue_reads,
                    epilogue_writes,
                    local_reduce,
                )
            except (NotImplementedError, AssertionError) as e:
                log.warning("NVGEMM epilogue codegen failed unexpectedly: %s", e)
                raise

        if ctb.make_kernel_render is None:
            raise AssertionError("expected ctb.make_kernel_render to be not None")
        kernel, render = ctb.make_kernel_render(
            ctb,
            epilogue_fn_code=epilogue_fn_code,
            epilogue_reads=epilogue_reads,
            epilogue_writes=epilogue_writes,
            epilogue_var_renames=epilogue_var_renames,
            local_reduce=local_reduce,
        )

        if not only_gen_src_code:
            template_node.mark_run()

        src_code = render()

        if only_gen_src_code:
            return src_code

        # Precompile only base (non-EFC) kernels. EFC kernels produce
        # closure-wrapped artifacts that can't be serialized to disk cache.
        if epilogue_nodes:
            precompile_metadata = None
        else:
            precompile_metadata = self._build_precompile_metadata(kernel, ctb)

        with V.set_kernel_handler(kernel):
            node_schedule: list[BaseSchedulerNode] = [template_node]
            if epilogue_nodes:
                node_schedule.extend(epilogue_nodes)
            kernel_name = self.define_kernel(
                src_code, node_schedule, precompile_metadata
            )

        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove
        self.free_buffers_in_scheduler()
        return None

    def _build_precompile_metadata(self, kernel, ctb):
        """Extract shapes and dtypes from kernel inputs/output for subprocess precompilation.

        Only called for base (non-epilogue) kernels. Returns None if shapes are
        symbolic (dynamic shapes), in which case the subprocess will skip
        precompilation and the kernel compiles lazily on first call.
        """
        if not hasattr(kernel, "_template_input_args"):
            return None

        precompile_shapes = {}
        precompile_strides = {}
        precompile_dtypes = {}

        try:
            for param_name, input_node in kernel._template_input_args:
                size = input_node.get_size()
                precompile_shapes[param_name] = [int(s) for s in size]
                stride = input_node.get_stride()
                precompile_strides[param_name] = [int(s) for s in stride]
                precompile_dtypes[param_name] = str(
                    input_node.get_dtype()
                ).removeprefix("torch.")

            out_layout = cast(Layout, ctb.layout)
            precompile_shapes["output"] = [int(s) for s in out_layout.size]
            precompile_strides["output"] = [int(s) for s in out_layout.stride]
            precompile_dtypes["output"] = str(out_layout.dtype).removeprefix("torch.")
        except (TypeError, RuntimeError, ValueError):
            log.debug(
                "Skipping NV Universal GEMM precompile metadata: symbolic sizes "
                "cannot be resolved to concrete values"
            )
            return None

        device = ctb.layout.device
        device_index = device.index if device.index is not None else 0

        import torch

        device_capability = None
        if torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability(device_index)

        max_active_clusters = None
        kernel_name = ctb.kernel_metadata.get("kernel_name")
        try:
            if kernel_name and torch.cuda.is_available():
                from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
                    get_kernel_by_name,
                )

                k = get_kernel_by_name(kernel_name)
                if k is not None and hasattr(k, "impl"):
                    from cutlass.operators.providers.cutedsl.integration_utils.mma import (
                        get_max_active_clusters,
                    )

                    max_active_clusters = get_max_active_clusters(
                        k.impl.cluster_shape_mn
                    )
        except Exception:
            log.debug(
                "Failed to resolve max_active_clusters for precompile", exc_info=True
            )

        return {
            "precompile_shapes": precompile_shapes,
            "precompile_strides": precompile_strides,
            "precompile_dtypes": precompile_dtypes,
            "device_index": device_index,
            "device_capability": device_capability,
            "max_active_clusters": max_active_clusters,
        }

    def generate_kernel_code_from_nodes(
        self,
        nodes: Sequence[BaseSchedulerNode],
        benchmark_kernel: bool = False,
        hint_override: int | None = None,
    ) -> str:
        prologue, template, epilogue = nodes[0].get_prologue_template_epilogue(
            list(nodes)
        )

        epilogue_reads: list[str] = []
        output_bufs: list[str] = []
        if epilogue:
            template_sn = cast(SchedulerNode, template)
            if not isinstance(template_sn.node, Buffer):
                raise AssertionError(f"expected Buffer, got {type(template_sn.node)}")
            original_buffer_name = template_sn.node.get_name()
            plan = self._epilogue_plan(template_sn.node, epilogue)
            evt_nodes = [] if plan.feed_main is not None else plan.evt_nodes
            removed_buffers_with_gemm = V.graph.removed_buffers.copy()
            if not plan.reductions:
                removed_buffers_with_gemm.add(original_buffer_name)
            try:
                if evt_nodes:
                    reads, writes, var_renames, _ = (
                        CutlassEVTCodegen.ir_to_evt_python_code(
                            original_buffer_name,
                            evt_nodes,
                            removed_buffers_with_gemm,
                        )
                    )
                    epilogue_reads = reads
                    d_buf = var_renames.get("D")
                    output_bufs = ([d_buf] if d_buf else []) + [
                        w for w in writes if w != d_buf
                    ]
                if plan.reductions:
                    output_bufs = [
                        original_buffer_name,
                        *output_bufs,
                        plan.reductions[0].output_name,
                    ]
                elif plan.feed_main is not None:
                    output_bufs = [plan.feed_main.output_name]
            except (NotImplementedError, AssertionError) as e:
                log.warning("NVGEMM benchmark epilogue codegen failed: %s", e)

        with config.patch("benchmark_kernel", benchmark_kernel):
            src_code = self.codegen_template(
                template,
                epilogue,
                prologue,
                only_gen_src_code=True,
            )

        if src_code is None:
            raise AssertionError("expected src_code to be not None")
        src_code = src_code.replace(
            str(Placeholder.KERNEL_NAME), _BENCHMARK_KERNEL_PREFIX
        )

        if benchmark_kernel:
            src_code = self._add_benchmark_helpers(
                src_code, template, epilogue, epilogue_reads, output_bufs
            )

        return src_code

    def _add_benchmark_helpers(
        self,
        src_code: str,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        epilogue_reads: list[str],
        output_bufs: list[str] | None = None,
    ) -> str:
        template_node = cast(SchedulerNode, template_node)
        ctb: NVUniversalGemmBuffer = self.get_nv_gemm_buffer_from_node(
            template_node, require_epilogue_fusion=bool(epilogue_nodes)
        )

        input_nodes = cast(list[Buffer], ctb.inputs)
        # Output store layouts in out_ptr order. A multi-store epilogue has one
        # per graph output; otherwise a single output (the fused final node, or
        # the plain GEMM layout).
        output_layouts: list[Layout] = []
        if output_bufs:
            for b in output_bufs:
                buf = V.graph.get_buffer(b)
                # pyrefly: ignore [missing-attribute]
                output_layouts.append(cast(Layout, buf.get_layout()))
        elif epilogue_nodes:
            final_node = cast(SchedulerNode, epilogue_nodes[-1])
            # pyrefly: ignore [missing-attribute]
            output_layouts.append(cast(Layout, final_node.node.get_layout()))
        else:
            output_layouts.append(cast(Layout, ctb.layout))

        args_code = IndentedBuffer()
        args_code.writeline("")
        args_code.writeline("is_nvgemm = True")
        args_code.writeline("")
        args_code.writeline("def get_args():")
        with args_code.indent():
            args_code.writeline("import torch")
            args_code.writeline("from torch._dynamo.testing import rand_strided")
            args_code.writeline("args = []")

            for inp in input_nodes:
                size = V.graph.sizevars.optimization_hints(inp.get_size())
                stride = V.graph.sizevars.optimization_hints(inp.get_stride())
                dtype = inp.get_dtype()
                device = inp.get_device()
                args_code.writeline(
                    f"args.append(rand_strided({size}, {stride}, device='{device}', dtype={dtype}))"
                )

            for ol in output_layouts:
                out_size = V.graph.sizevars.optimization_hints(ol.size)
                out_stride = V.graph.sizevars.optimization_hints(ol.stride)
                args_code.writeline(
                    f"args.append(rand_strided({out_size}, {out_stride}, device='{ol.device}', dtype={ol.dtype}))"
                )

            for read_name in epilogue_reads:
                buf = V.graph.get_buffer(read_name)
                size = V.graph.sizevars.optimization_hints(buf.get_size())
                stride = V.graph.sizevars.optimization_hints(buf.get_stride())
                dtype = buf.get_dtype()
                device = buf.get_device()
                args_code.writeline(
                    f"args.append(rand_strided({size}, {stride}, device='{device}', dtype={dtype}))"
                )

            if ctb.workspace_size > 0:
                args_code.writeline(
                    f"args.append(torch.empty({ctb.workspace_size}, "
                    f"device='{output_layouts[0].device}', dtype=torch.int8))"
                )

            args_code.writeline("return args")

        args_code.writeline("")
        args_code.writeline("def call(args):")
        with args_code.indent():
            args_code.writeline("import torch")
            num_inputs = len(input_nodes)
            n_fixed = num_inputs + len(output_layouts)
            param_list = [f"args[{i}]" for i in range(n_fixed)]

            for j in range(len(epilogue_reads)):
                param_list.append(f"args[{n_fixed + j}]")

            if ctb.workspace_size > 0:
                param_list.append(f"args[{n_fixed + len(epilogue_reads)}]")

            params_str = ", ".join(param_list)
            args_code.writeline("stream = torch.cuda.current_stream().cuda_stream")
            bench_fn_name = f"{_BENCHMARK_KERNEL_PREFIX}_{MAIN_SUFFIX}"
            args_code.writeline(f"{bench_fn_name}({params_str}, stream=stream)")

        return src_code + args_code.getvalue()
<|endoftext|># mypy: allow-untyped-defs
"""
Global kernel cache for NVIDIA Universal GEMM.

This module provides a lazy-initialized cache for cutlass.operators kernels,
avoiding expensive manifest scans on every kernel lookup.

The first call to get_kernel_by_name() loads all kernels from cutlass.operators
(~10 seconds) and builds a name->kernel dict. Subsequent calls use the
dict for O(1) lookup (~0.1 μs).
"""

import dataclasses
import functools
import logging
import threading
from collections.abc import Callable
from typing import Any, Literal

import torch


log = logging.getLogger(__name__)


@functools.cache
def _device_target(cc: int) -> Any:
    """The CUDA arch target for this device (e.g. cc=100 -> TargetSm '100a').

    Used to reject kernels whose `supported_targets` don't cover this device.
    `designed_for_min_cc` alone is insufficient: an arch-conditional sm90 kernel
    reports min_cc=90 (which is <= a 100 device) yet only lists a cc=90 target
    and fails to compile on sm100 ("expects arch sm_90a, but got sm_100a").
    """
    from cutlass.operators.arch import TargetSm

    return TargetSm.ensure(f"{cc}a")


def _epilogue_args_signature(epilogue_args: Any) -> tuple:
    """Extract a hashable signature of epilogue args for cache keying.

    Two callers with the same `(efc_kernel_name, epilogue_source)` but
    different aux-tensor specs (dtype, shape, stride) would otherwise share
    a kernel object whose internal JIT state is mutated by each call to
    `kernel.compile(args)` — a silent miscompile, since the compiled
    artifact's launch closure reads the kernel's CURRENT JIT at launch time
    rather than the one from when the artifact was built.
    """
    if epilogue_args is None:
        return ()
    tensors = getattr(epilogue_args, "tensors", None)
    if not tensors:
        return ()
    sig: list[tuple] = []
    for name, val in tensors.items():
        if torch.is_tensor(val):
            sig.append(
                (name, "tensor", val.dtype, tuple(val.shape), tuple(val.stride()))
            )
        else:
            sig.append((name, type(val).__name__))
    return tuple(sig)


# Reentrant: get_efc_kernel_with_epilogue holds this lock and may call
# get_kernel_by_name -> _ensure_caches, which re-acquires it on the same thread.
_cache_lock = threading.RLock()

# Full kernel manifest: kernel_name -> kernel object. Built lazily (~14s, ~290K
# operators) only as a LAST-RESORT fallback -- the common paths (choice
# generation and subprocess precompile) resolve operators via cheap
# args-filtered queries and cache them in _ops_by_name.
_kernel_by_name_cache: dict[str, Any] | None = None


def _operand_dtype_str(operand: Any) -> str | None:
    """Best-effort cutlass dtype name for a kernel operand or args operand."""
    dtype = getattr(operand, "dtype", None)
    if dtype is None:
        tensor = getattr(operand, "tensor", None)
        dtype = getattr(tensor, "dtype", None)
    return None if dtype is None else str(dtype)


def _build_kernel_cache() -> dict[str, Any]:
    """Build the full kernel name -> kernel manifest (fallback path only)."""
    import cutlass.operators

    log.debug("Building NVGEMM kernel cache (this may take a few seconds)...")

    try:
        from torch._inductor.kernel.vendored_templates.cutedsl import (  # noqa: F401
            wrappers,
        )
    except ImportError:
        log.debug("Vendored kernel wrappers not available")

    all_kernels = cutlass.operators.get_operators()
    cache = {k.metadata.operator_name: k for k in all_kernels}
    log.debug("NVGEMM kernel manifest built: %d kernels", len(cache))
    return cache


def _ensure_caches() -> None:
    global _kernel_by_name_cache
    if _kernel_by_name_cache is None:
        with _cache_lock:
            if _kernel_by_name_cache is None:
                _kernel_by_name_cache = _build_kernel_cache()


def _get_kernel_cache() -> dict[str, Any]:
    """Return the fallback manifest, building it lazily if needed.

    Snapshot to local frame: a concurrent clear_cache() rebinding the global to
    None cannot turn the caller's subsequent read into AttributeError.
    """
    _ensure_caches()
    cache = _kernel_by_name_cache
    if cache is None:
        raise AssertionError("expected cache to be not None")
    return cache


def _operand_sig(operand: Any) -> tuple | None:
    """Hashable (dtype, shape, stride, scale-sig, mode, swizzle) signature."""
    if operand is None:
        return None
    w = getattr(operand, "tensor", operand)
    dtype = _operand_dtype_str(operand)
    shape = getattr(w, "shape", None)
    stride = getattr(w, "stride", None)
    if dtype is None or shape is None or stride is None:
        return None
    scale = getattr(operand, "scale", None)
    scale_sig = _operand_sig(scale) if scale is not None else None
    # ScaledOperand.mode/.swizzle are distinct from the scale tensor's layout
    # and are not folded into it, yet supports(args) depends on them -- two
    # scaled GEMMs with identical operand/scale layouts but different scale or
    # swizzle modes must not collide. Read from __dict__ (only ScaledOperand
    # sets these) so a dense operand's torch.Tensor.mode method isn't captured.
    d = getattr(operand, "__dict__", {})
    mode, swizzle = d.get("mode"), d.get("swizzle")
    # mode/swizzle are cutlass.operators enums (ScaleMode/ScaleSwizzleMode) that
    # are unhashable, so key on their .name to keep the sig usable as a dict key.
    mode = getattr(mode, "name", mode)
    swizzle = getattr(swizzle, "name", swizzle)
    return (dtype, tuple(shape), tuple(stride), scale_sig, mode, swizzle)


def _partition_sig(args: Any) -> tuple | None:
    """Signature capturing everything `supports(args)` depends on, or None if
    it can't be fully determined (falls back to a non-memoized scan)."""
    a = _operand_sig(getattr(args, "A", None))
    b = _operand_sig(getattr(args, "B", None))
    o = _operand_sig(getattr(args, "out", None))
    if a is None or b is None or o is None:
        return None
    return (a, b, o, getattr(args, "accumulator_type", None))


# Per-process cache of instantiated operators, keyed by operator_name. Populated
# incrementally by the args-filtered queries below (choice generation and the
# subprocess precompile fast path). One args query returns the whole set of
# operators compatible with a shape (hundreds); we index all of them, so any
# later lookup -- another config of that shape, another shape sharing those
# templates, or get_kernel_by_name for a chosen
# kernel -- resolves in O(1) without rebuilding the ~14s manifest. Operator
# names embed the arch (e.g. "..._sm100_..."), so the name alone is unambiguous
# within a process. Reusing an operator across different args is safe for the
# same reason the manifest always has been: an operator is a config template
# (its args-specific state lives in the artifact returned by compile(args)), and
# every (name, args) pairing routed here was validated by supports(args).
_ops_by_name: dict[str, Any] = {}

# Memoizes partition results. `partition_compatible_kernels` is called once per
# GEMM node during choice generation, but FLUX has hundreds of nodes sharing a
# handful of shapes (e.g. 114 nodes are all 4608x3072x3072); memoizing collapses
# that to one args query per unique shape.
_partition_cache: dict[tuple, list[list[Any]]] = {}


def _args_query_candidates(args: Any, cc: int, efc_only: bool) -> list[Any]:
    """Compatible operators via an args-filtered get_operators query (~0.05s).

    cutlass prunes by supports(args) and the device target internally. Only
    valid for the dense GEMM path: for that path this returns the identical set
    as the full-manifest scan (verified), avoiding the ~14s manifest build.
    """
    import cutlass.operators

    metadata_filter = (
        (lambda md: "EFC" in md.operator_class.__name__) if efc_only else None
    )
    return cutlass.operators.get_operators(
        args=args, target_sm=f"{cc}a", metadata_filter=metadata_filter
    )


def _filter_supported(kernels: Any, args: Any, cc: int) -> list[Any]:
    """Keep the kernels that support `args` on this device.

    Prunes by (A, B) operand dtype first (a cheap string compare) so the
    expensive supports() runs only on the ~hundreds that could possibly match --
    the ~290K manifest spans all dtypes, and a GEMM only matches its own.
    """
    device_target = _device_target(cc)
    dtype_a = _operand_dtype_str(getattr(args, "A", None))
    dtype_b = _operand_dtype_str(getattr(args, "B", None))
    out = []
    for kernel in kernels:
        if kernel.designed_for_min_cc > cc:
            continue
        operands = kernel.metadata.operands
        if (
            _operand_dtype_str(getattr(operands, "A", None)) != dtype_a
            or _operand_dtype_str(getattr(operands, "B", None)) != dtype_b
        ):
            continue
        if not device_target.supports_operators_from(kernel.metadata.supported_targets):
            continue
        if kernel.supports(args).error is not None:
            continue
        out.append(kernel)
    return out


def _manifest_candidates(args: Any, cc: int, efc_only: bool) -> list[Any]:
    """Compatible operators via a full-manifest scan (last-resort fallback)."""
    kernels = _get_kernel_cache().values()
    if efc_only:
        kernels = (
            kernel
            for kernel in kernels
            if "EFC" in kernel.metadata.operator_class.__name__
        )
    return _filter_supported(kernels, args, cc)


def _blockscaled_provider_classes() -> list[Any]:
    """The operator classes that produce block-scaled (fp4/fp8) kernels.

    Both the cutlass built-in and the vendored provider matter: for NVFP4 the
    vendored kernels supply most of the working configs (60 of 96), so missing
    either under-generates and breaks scaled autotuning.
    """
    classes: list[Any] = []
    try:
        from cutlass.operators.providers.cutedsl.gemm.sm100_dense_blockscaled_static_persistent import (
            PersistentDenseBlockScaledGemmOperator,
        )

        classes.append(PersistentDenseBlockScaledGemmOperator)
    except ImportError:
        pass
    try:
        from torch._inductor.kernel.vendored_templates.cutedsl.wrappers.dense_blockscaled_gemm_kernel import (
            VendoredDenseBlockScaledGemmKernel,
        )

        classes.append(VendoredDenseBlockScaledGemmKernel)
    except ImportError:
        pass
    return classes


@functools.cache
def _blockscaled_operators() -> tuple:
    """Generate the architecture-neutral block-scaled operator pool."""
    ops: list[Any] = []
    for cls in _blockscaled_provider_classes():
        ops.extend(cls.generate_operators(lambda md: True, args=None))
    return tuple(ops)


def _scaled_operand_type_signature(args: Any) -> tuple:
    def operand_signature(operand: Any) -> tuple:
        scale = getattr(operand, "scale", None)
        return (
            _operand_dtype_str(operand),
            _operand_dtype_str(scale),
            str(getattr(operand, "mode", None)),
            str(getattr(operand, "swizzle", None)),
        )

    return (
        operand_signature(args.A),
        operand_signature(args.B),
        _operand_dtype_str(args.out),
        str(getattr(args, "accumulator_type", None)),
    )


def _scaled_metadata_type_signature(metadata: Any) -> tuple:
    operands = metadata.operands

    def operand_signature(operand: Any) -> tuple:
        scale = getattr(operand, "scale", None)
        return (
            _operand_dtype_str(operand),
            _operand_dtype_str(scale),
            str(getattr(operand, "mode", None)),
            str(getattr(operand, "swizzle", None)),
        )

    return (
        operand_signature(operands.A),
        operand_signature(operands.B),
        _operand_dtype_str(operands.out),
        str(getattr(operands, "accumulator_type", None)),
    )


@functools.cache
def _blockscaled_manifest(cc: int, type_signature: tuple):
    """Build a block-scaled manifest for one operand type recipe.

    The provider set is generated without a concrete target to preserve its
    complete design space. The manifest applies shape and target filtering when
    candidates are requested.
    """
    import cutlass.operators

    manifest = cutlass.operators.Manifest()
    manifest.add_operators(
        [
            op
            for op in _blockscaled_operators()
            if _scaled_metadata_type_signature(op.metadata) == type_signature
        ]
    )
    return manifest


def _scaled_candidates(args: Any, cc: int, efc_only: bool) -> list[Any]:
    """Compatible operators for a scaled GEMM via direct block-scaled enumeration.

    get_operators(args=...) derives operand configs from the args and
    under-generates for scaled (e.g. ~36 of 96 valid fp4 kernels), so we scan the
    block-scaled sub-provider's full design space instead -- complete, and ~50x
    cheaper than the manifest. Falls back to the manifest if the provider is
    unavailable or nothing matches (e.g. a future non-block-scaled scaled dtype).
    """
    from torch._inductor.kernel.vendored_templates.cutedsl.wrappers.dense_blockscaled_gemm_kernel import (
        VendoredDenseBlockScaledGemmEFC,
        VendoredDenseBlockScaledGemmKernel,
    )

    manifest = _blockscaled_manifest(cc, _scaled_operand_type_signature(args))
    if manifest.operators:
        out = manifest.filter_operators(
            args=args,
            target_sm=f"{cc}a",
        )
        if out:
            efc = []
            for op in out:
                if op.metadata.operator_class is not VendoredDenseBlockScaledGemmKernel:
                    continue
                metadata = dataclasses.replace(
                    op.metadata,
                    operator_name=op.metadata.operator_name.replace(
                        "VendoredDenseBlockScaledGemmKernel",
                        "VendoredDenseBlockScaledGemmEFC",
                        1,
                    ),
                    operator_class=VendoredDenseBlockScaledGemmEFC,
                )
                efc.append(VendoredDenseBlockScaledGemmEFC(metadata))
            if efc:
                return efc if efc_only else [*out, *efc]
            if not efc_only:
                return out
    return _manifest_candidates(args, cc, efc_only)


def partition_compatible_kernels(
    args: Any,
    cc: int,
    classifier: Callable[[Any], int],
    num_buckets: int,
    efc_only: bool = False,
    candidate_source: Literal["args", "scaled", "manifest"] = "manifest",
    classifier_key: str | None = None,
) -> list[list[Any]]:
    """Partition the operators compatible with `args` into N buckets.

    `classifier(metadata)` returns a bucket index in [0, num_buckets-1] or -1 to
    drop the kernel. `candidate_source` selects how the compatible set is found,
    all of which avoid the ~14s full manifest except "manifest":
      - "args"     dense GEMM: cheap args-filtered get_operators query
      - "scaled"   scaled GEMM: direct block-scaled sub-provider enumeration
      - "manifest" fallback (e.g. grouped GEMM): full-manifest scan
    `efc_only` restricts to epilogue-fusion-capable kernels (the addmm/bias
    path). Results are memoized per (shape, cc, num_buckets, efc_only, source).
    """
    sig = _partition_sig(args)
    cache_key = (
        (sig, cc, num_buckets, efc_only, candidate_source, classifier_key)
        if sig is not None and classifier_key is not None
        else None
    )
    if cache_key is not None:
        cached = _partition_cache.get(cache_key)
        if cached is not None:
            return [list(b) for b in cached]

    if candidate_source == "args":
        candidates = _args_query_candidates(args, cc, efc_only)
    elif candidate_source == "scaled":
        candidates = _scaled_candidates(args, cc, efc_only)
    elif candidate_source == "manifest":
        candidates = _manifest_candidates(args, cc, efc_only)
    else:
        raise ValueError(f"Unknown NVGEMM candidate source: {candidate_source}")

    buckets: list[list[Any]] = [[] for _ in range(num_buckets)]
    for kernel in candidates:
        # Share every compatible operator so a later get_kernel_by_name (e.g. the
        # scheduling max_active_clusters lookup for the chosen kernel) hits
        # without building the manifest.
        _ops_by_name.setdefault(kernel.metadata.operator_name, kernel)
        bucket = classifier(kernel.metadata)
        if bucket < 0:
            continue
        buckets[bucket].append(kernel)
    log.debug(
        "Partitioned %d compatible operators into buckets %s",
        len(candidates),
        [len(b) for b in buckets],
    )
    if cache_key is not None:
        _partition_cache[cache_key] = [list(b) for b in buckets]
    return buckets


def get_kernel_by_name(kernel_name: str) -> Any:
    """Get a kernel by name.

    Checks the incrementally-populated operator cache first (filled by the
    args-filtered queries during choice generation / precompile), falling back
    to the full manifest only if the name was never resolved via args.
    """
    kernel = _ops_by_name.get(kernel_name)
    if kernel is not None:
        return kernel
    return _get_kernel_cache().get(kernel_name)


def get_kernel_by_name_via_args(kernel_name: str, args: Any, cc: int) -> Any:
    """Fast single-kernel lookup via an args-filtered get_operators query.

    Passing concrete `args` (a cutlass RuntimeArguments) plus the device target
    prunes the operator space to the ~hundreds compatible with this exact GEMM
    in ~0.05s, versus ~14s to enumerate/construct the full ~294K-kernel manifest
    (`get_kernel_by_name`). Used by subprocess precompile workers, which need
    only the single named kernel and would otherwise each rebuild the whole
    manifest.

    A miss runs one args query and indexes every operator it returns into
    _ops_by_name, so the ~50ms cost is paid once per distinct operator (not once
    per lookup) and is amortized across all configs and shapes a persistent
    worker handles. Returns None if no operator matches the name (caller then
    falls back to the full manifest).
    """
    import cutlass.operators

    cached = _ops_by_name.get(kernel_name)
    if cached is not None:
        return cached

    ops = cutlass.operators.get_operators(args=args, target_sm=f"{cc}a")
    for op in ops:
        # setdefault: keep an already-cached (possibly already-compiled) object
        # rather than replacing it with a fresh instance from this query.
        _ops_by_name.setdefault(op.metadata.operator_name, op)
    if (
        kernel_name not in _ops_by_name
        and "VendoredDenseBlockScaledGemmEFC" in kernel_name
    ):
        from cutlass.operators.arguments import GemmArguments

        base_args = GemmArguments(
            args.A,
            args.B,
            args.out,
            accumulator_type=args.accumulator_type,
        )
        scaled_ops = _scaled_candidates(base_args, cc, efc_only=False)
        log.debug(
            "Scaled by-name fallback found %d candidates for %s",
            len(scaled_ops),
            kernel_name,
        )
        for op in scaled_ops:
            _ops_by_name.setdefault(op.metadata.operator_name, op)
    return _ops_by_name.get(kernel_name)


def ensure_cache_initialized() -> None:
    """Ensure the kernel cache is initialized."""
    _get_kernel_cache()


_efc_epilogue_cache: dict[tuple[str, str, tuple], Any] = {}


def clear_cache() -> None:
    """Clear all kernel caches."""
    global _kernel_by_name_cache, _efc_epilogue_cache
    with _cache_lock:
        _kernel_by_name_cache = None
        _efc_epilogue_cache = {}
        _partition_cache.clear()
        _ops_by_name.clear()
        _blockscaled_operators.cache_clear()
        _blockscaled_manifest.cache_clear()


class _NVGEMMCacheWrapper:
    def cache_clear(self) -> None:
        clear_cache()


from torch._inductor.utils import clear_on_fresh_cache


clear_on_fresh_cache(_NVGEMMCacheWrapper())


def _meta_epilogue_metadata(epilogue_args: Any) -> Any:
    """Build the cached kernel's epilogue metadata from meta-device tensors.

    The EFC kernel is cached in ``_efc_epilogue_cache`` and reused across every
    call; the real output/aux tensors are supplied per call via the runtime
    ``GemmArguments.epilogue``, so the cached kernel needs only the epilogue's
    shape/dtype/layout (its IR). Building it from the real ``EpilogueArguments``
    would make it retain those tensors forever -- ``EpilogueMetadata`` shares the
    arguments' ``tensors`` (wrapping the output ``D`` and aux inputs) and traced
    functor -- keeping the (often intermediate) output alive: a memory leak, and
    it breaks CUDA graph capture (a live graph-pool allocation that is not a
    tracked graph output). Re-trace the epilogue with meta tensors (carrying
    dtype/layout but no device storage; the tracer reads only their metadata) so
    the cached kernel holds nothing real.
    """
    from cutlass.operators.arguments.epilogue import EpilogueArguments
    from cutlass.operators.metadata import EpilogueMetadata
    from cutlass.operators.utils.tensor import TensorWrapper

    from .nv_universal_gemm_utils import cutlass_dtype_to_torch

    def _meta_like(t: torch.Tensor) -> torch.Tensor:
        return torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device="meta")

    fake_kwargs = {}
    for name, val in epilogue_args.tensors.items():
        # Build a storage-free meta tensor with the same shape/stride/dtype.
        # A compile-time-only TensorWrapper (the subprocess-precompile path) has
        # no runtime tensor -- its `.runtime_tensor` property raises ValueError --
        # so read `_runtime_tensor` directly and fall back to the wrapper's own
        # shape/stride and its CUTLASS dtype.
        if isinstance(val, torch.Tensor):
            fake_kwargs[name] = _meta_like(val)
        elif isinstance(val, TensorWrapper) and isinstance(
            val._runtime_tensor, torch.Tensor
        ):
            fake_kwargs[name] = _meta_like(val._runtime_tensor)
        elif (
            isinstance(val, TensorWrapper)
            and (dtype := cutlass_dtype_to_torch(val.dtype)) is not None
        ):
            fake_kwargs[name] = torch.empty_strided(
                tuple(val.shape), tuple(val.stride), dtype=dtype, device="meta"
            )
        else:
            fake_kwargs[name] = val
    fake_args = EpilogueArguments(epilogue_args.epilogue_fn, **fake_kwargs)
    accum = epilogue_args.traced_epilogue.example_inputs["accum"]
    fake_args.trace(accum.shape, accum.element)
    return EpilogueMetadata.from_args(fake_args)


def get_efc_kernel_with_epilogue(
    efc_kernel_name: str,
    epilogue_args: Any,
    epilogue_source: str = "",
    base_kernel: Any | None = None,
) -> Any:
    """Get (or create and cache) an EFC kernel bound to a specific epilogue.

    epilogue_source is preferred over inspect.getsource — generated functions
    produce unstable source strings that can't be hashed reliably.

    base_kernel: pre-resolved base EFC kernel (e.g. from the args-filtered fast
    lookup in a subprocess worker). When None, falls back to the full manifest.
    """
    if not epilogue_source:
        epilogue_source = str(epilogue_args) if epilogue_args is not None else ""

    cache_key = (
        efc_kernel_name,
        epilogue_source,
        _epilogue_args_signature(epilogue_args),
    )

    with _cache_lock:
        if cache_key in _efc_epilogue_cache:
            log.debug("EFC kernel with epilogue found in cache: %s", efc_kernel_name)
            return _efc_epilogue_cache[cache_key]

        if base_kernel is None:
            # Prefer the args-resolved operator cache (populated by choice
            # generation); only build the full manifest if the name was never
            # seen via args.
            base_kernel = get_kernel_by_name(efc_kernel_name)
        if base_kernel is None:
            log.debug("Base EFC kernel not found: %s", efc_kernel_name)
            return None

        from cutlass.operators.metadata import OperatorMetadata

        epilogue_metadata = _meta_epilogue_metadata(epilogue_args)

        base_metadata = base_kernel.metadata
        new_metadata = OperatorMetadata(
            operands=base_metadata.operands,
            design=base_metadata.design,
            operator_name=base_metadata.operator_name,
            operator_class=base_metadata.operator_class,
            supported_targets=base_metadata.supported_targets,
            epilogue=epilogue_metadata,
        )

        kernel_class = base_metadata.operator_class
        new_kernel = kernel_class(new_metadata)

        _efc_epilogue_cache[cache_key] = new_kernel
        log.debug("Created and cached EFC kernel with epilogue: %s", efc_kernel_name)

        return new_kernel
<|endoftext|># mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM (NVGEMM) backend for PyTorch Inductor.

This module provides integration with the cutlass.operators library to enable
high-performance GEMM kernels for NVIDIA GPUs.
"""

import itertools
import re
from enum import auto, Enum
from typing import Any, Literal

import torch
from torch._inductor import config
from torch._inductor.autotune_process import (
    BenchmarkRequest,
    GPUDeviceBenchmarkMixin,
    TensorMeta,
)
from torch._inductor.codegen.cuda.cuda_env import get_cuda_arch
from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
    _compile_nvgemm,
    _create_gemm_arguments,
    _create_gemm_cache_key,
    _current_target_sm,
    _get_scaled_gemm_modes,
    _make_disk_config_key,
    _nvgemm_bias_add_epilogue,
    _rewrap_efc_compiled_obj,
    _unwrap_efc_compiled_obj,
)
from torch._inductor.heuristics.template.nv_universal_gemm import get_nvgemm_heuristics
from torch._inductor.ir import (
    Buffer,
    ChoiceCaller,
    FixedLayout,
    Layout,
    PermuteView,
    TensorBox,
)
from torch._inductor.kernel_inputs import MMKernelInputs
from torch._inductor.utils import ensure_nv_universal_gemm_available
from torch._inductor.virtualized import V
from torch._logging import getArtifactLogger


log = getArtifactLogger(__name__, "output_code")


class GemmVariant(Enum):
    """
    Enum for different GEMM operation types supported by NVIDIA Universal GEMM.
    """

    GEMM = auto()

    GROUPED_GEMM = auto()

    SCALED_GEMM = auto()

    @property
    def op_name(self) -> str:
        """Return the operation name for logging and naming."""
        if self == GemmVariant.GROUPED_GEMM:
            return "nv_universal_grouped_gemm"
        if self == GemmVariant.SCALED_GEMM:
            return "nv_universal_scaled_gemm"
        return "nv_universal_gemm"


class NVUniversalGemmBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    """Benchmark request for NVIDIA Universal GEMM kernels."""

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: TensorMeta | list[TensorMeta],
        output_tensor_meta: TensorMeta | list[TensorMeta],
        kernel,  # cutlass.operators.Operator object
        accumulator_type: torch.dtype,
        variant: GemmVariant,
        workspace_size: int = 0,
        scale_type_a: Any | None = None,
        scale_type_b: Any | None = None,
        swizzle_type_a: Any | None = None,
        swizzle_type_b: Any | None = None,
        swap_ab: bool = False,
        has_bias_epilogue: bool = False,
    ) -> None:
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, ())
        self.kernel = kernel
        self.accumulator_type = accumulator_type
        self.has_bias_epilogue = has_bias_epilogue
        self._workspace: torch.Tensor | None = None
        self._disk_fn_cache: dict = {}
        self.workspace_size = workspace_size
        self.variant = variant
        self.scale_type_a = scale_type_a
        self.scale_type_b = scale_type_b
        self.swizzle_type_a = swizzle_type_a
        self.swizzle_type_b = swizzle_type_b
        self.swap_ab = swap_ab

    def benchmark(
        self,
        *input_tensors: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> float:
        """Benchmark the NVIDIA Universal GEMM kernel.

        Override the base class to always create tensors from input_tensor_meta.
        This is necessary because input_nodes may be ReinterpretViews that share
        the same underlying buffer name. The autotuning framework deduplicates
        inputs by name (in AlgorithmSelectorCache.get_inputs()), resulting in
        fewer tensors than expected. By always creating from input_tensor_meta,
        we ensure each input gets its own tensor with the correct size/stride/offset
        from the view's layout.

        """
        from torch._inductor.runtime.benchmarking import benchmarker

        input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
        if out is None:
            out = self.output_tensor_meta.to_tensor()

        fn = self.make_run_fn(*input_tensors, out=out)
        try:
            if self.benchmark_with_cudagraphs:
                res = benchmarker.benchmark_gpu_with_cuda_graph(fn)
            else:
                res = self.do_bench(fn, *input_tensors, out=out)
        finally:
            self.cleanup_run_fn()
        return res

    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor):
        """Create a function to run the NVIDIA Universal GEMM kernel."""
        from cutlass.operators.artifact import CompiledArtifact

        from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set
        from torch._inductor.utils import _ensure_fp4_dtype_registered

        _ensure_fp4_dtype_registered()

        if self.has_bias_epilogue:
            # Benchmark the real fused kernel (GEMM + bias-add epilogue). The
            # bias is the last input; the rest are the GEMM operands.
            return self._make_bias_run_fn(input_tensors, out)

        # swap_ab: transpose operands and write into a transposed view of `out`
        # (zero-copy). See _nvgemm_run for the full explanation.
        if self.swap_ab and len(input_tensors) >= 2:
            a, b = input_tensors[0], input_tensors[1]
            if len(input_tensors) >= 4:
                sa, sb = input_tensors[2], input_tensors[3]
                input_tensors = (b.t(), a.t(), sb, sa) + input_tensors[4:]
            else:
                input_tensors = (b.t(), a.t()) + input_tensors[2:]
            out = out.t()

        helper_kwargs: dict[str, Any] = {}
        if self.variant == GemmVariant.SCALED_GEMM:
            scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b = (
                _get_scaled_gemm_modes(
                    self.scale_type_a,
                    self.swizzle_type_a,
                    self.scale_type_b,
                    self.swizzle_type_b,
                )
            )
            helper_kwargs = {
                "scale_mode_a": scale_mode_a,
                "swizzle_mode_a": swizzle_mode_a,
                "scale_mode_b": scale_mode_b,
                "swizzle_mode_b": swizzle_mode_b,
            }

        cache_key = _create_gemm_cache_key(input_tensors, out)
        dev_idx = input_tensors[0].device.index or 0
        kernel_name = self.kernel.metadata.operator_name
        disk_config_key = _make_disk_config_key(
            kernel_name,
            self.variant.name,
            self.accumulator_type,
            self.scale_type_a,
            self.scale_type_b,
            self.swizzle_type_a,
            self.swizzle_type_b,
        )

        def disk_fallback(kernel):
            compiled_fn = disk_cache_get(
                self._disk_fn_cache,
                kernel_name,
                disk_config_key,
                cache_key,
                dev_idx,
            )
            if compiled_fn is not None:
                compiled_fn = _rewrap_efc_compiled_obj(compiled_fn, kernel)
            if compiled_fn is not None:
                return CompiledArtifact(
                    compiled_fn, kernel, _current_target_sm(dev_idx)
                )
            return None

        artifact, args, kernel, was_compiled = _compile_nvgemm(
            self.variant.name,
            input_tensors,
            out,
            self.accumulator_type,
            kernel_obj=self.kernel,
            args_kwargs=helper_kwargs,
            fallback_fn=disk_fallback,
        )

        if was_compiled:
            obj = _unwrap_efc_compiled_obj(artifact.compiled_obj)
            disk_cache_set(
                self._disk_fn_cache,
                kernel_name,
                disk_config_key,
                cache_key,
                obj,
                dev_idx,
            )

        # Allocate workspace if needed
        if self.workspace_size > 0:
            self._workspace = torch.empty(
                self.workspace_size, device=out.device, dtype=torch.int8
            )
        else:
            self._workspace = None

        workspace = self._workspace

        def run_kernel():
            stream = torch.cuda.current_stream()
            kernel.run(
                args,
                artifact,
                stream=stream,
                workspace=workspace,
                assume_supported_args=True,
            )

        return run_kernel

    def _make_bias_run_fn(self, input_tensors, out):
        """Build a run fn for the fused GEMM + bias-add epilogue (addmm).

        Mirrors make_run_fn's disk caching but with the bias epilogue: the
        unwrap/rewrap machinery serializes the EFC artifact, so subprocess
        precompile (which writes the same key) hands off to the benchmark
        instead of recompiling serially.
        """
        from cutlass.operators.arguments import EpilogueArguments
        from cutlass.operators.artifact import CompiledArtifact

        from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set

        *gemm_tensors, bias = input_tensors
        gemm_tensors = tuple(gemm_tensors)
        epilogue_args = EpilogueArguments(_nvgemm_bias_add_epilogue, bias=bias, D=out)

        kernel_name = self.kernel.metadata.operator_name
        cache_key = _create_gemm_cache_key(
            gemm_tensors, out, has_epilogue=True, aux_tensors=(bias,)
        )
        dev_idx = gemm_tensors[0].device.index or 0
        disk_config_key = _make_disk_config_key(
            kernel_name,
            self.variant.name,
            self.accumulator_type,
            self.scale_type_a,
            self.scale_type_b,
            self.swizzle_type_a,
            self.swizzle_type_b,
        )

        def disk_fallback(kernel):
            compiled_fn = disk_cache_get(
                self._disk_fn_cache, kernel_name, disk_config_key, cache_key, dev_idx
            )
            if compiled_fn is not None:
                compiled_fn = _rewrap_efc_compiled_obj(
                    compiled_fn, kernel, epilogue_args=epilogue_args
                )
            if compiled_fn is not None:
                return CompiledArtifact(
                    compiled_fn, kernel, _current_target_sm(dev_idx)
                )
            return None

        artifact, args, kernel, was_compiled = _compile_nvgemm(
            self.variant.name,
            gemm_tensors,
            out,
            self.accumulator_type,
            kernel_name=kernel_name,
            epilogue_args=epilogue_args,
            epilogue_source="nvgemm_addmm_bias_v1",
            fallback_fn=disk_fallback,
            base_kernel=self.kernel,
        )

        if was_compiled:
            disk_cache_set(
                self._disk_fn_cache,
                kernel_name,
                disk_config_key,
                cache_key,
                _unwrap_efc_compiled_obj(artifact.compiled_obj),
                dev_idx,
            )

        if self.workspace_size > 0:
            self._workspace = torch.empty(
                self.workspace_size, device=out.device, dtype=torch.int8
            )
        else:
            self._workspace = None
        workspace = self._workspace

        def run_kernel():
            kernel.run(
                args,
                artifact,
                stream=torch.cuda.current_stream(),
                workspace=workspace,
                assume_supported_args=True,
            )

        return run_kernel

    def precompile(self):
        input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
        out = self.output_tensor_meta.to_tensor()
        self.make_run_fn(*input_tensors, out=out)
        self.cleanup_run_fn()

    def cleanup_run_fn(self) -> None:
        self._workspace = None


class NVUniversalGemmCaller(ChoiceCaller):
    """
    ChoiceCaller for NVIDIA Universal GEMM kernels.

    Wraps a cutlass.operators kernel and integrates with Inductor's autotuning.
    """

    index_counter = itertools.count()

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        kernel,  # cutlass.operators.Operator object
        accumulator_type: torch.dtype,
        variant: GemmVariant,
        workspace_size: int = 0,
        scale_type_a: Any | None = None,
        scale_type_b: Any | None = None,
        swizzle_type_a: Any | None = None,
        swizzle_type_b: Any | None = None,
        supports_epilogue_fusion: bool = False,
        swap_ab: bool = False,
        kernel_layout: Layout | None = None,
        bias_node: Buffer | None = None,
    ) -> None:
        super().__init__(
            name=name,
            input_nodes=input_nodes,
            layout=layout,
            description=f"{variant.op_name} {kernel.metadata.operator_name}"
            + (" swap_ab" if swap_ab else "")
            + (" bias" if bias_node is not None else ""),
        )
        self.kernel = kernel
        self.accumulator_type = accumulator_type
        self.workspace_size = workspace_size
        self.variant = variant
        self.scale_type_a = scale_type_a
        self.scale_type_b = scale_type_b
        self.swizzle_type_a = swizzle_type_a
        self.swizzle_type_b = swizzle_type_b
        self.supports_epilogue_fusion = supports_epilogue_fusion
        self.swap_ab = swap_ab
        self.bias_node = bias_node
        self._kernel_layout = kernel_layout or layout
        self._cached_output_node: TensorBox | None = None

        output_buffer = Buffer(name=f"{variant.op_name}_out", layout=layout)

        self.bmreq = NVUniversalGemmBenchmarkRequest(
            kernel_name=name,
            input_tensor_meta=TensorMeta.from_irnodes(input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(output_buffer),
            kernel=kernel,
            accumulator_type=accumulator_type,
            workspace_size=workspace_size,
            variant=variant,
            scale_type_a=scale_type_a,
            scale_type_b=scale_type_b,
            swizzle_type_a=swizzle_type_a,
            swizzle_type_b=swizzle_type_b,
            swap_ab=swap_ab,
            has_bias_epilogue=bias_node is not None,
        )

    def __str__(self) -> str:
        return f"NVUniversalGemmCaller({self.kernel.metadata.operator_name})"

    def precompile(self):
        self.bmreq.precompile()

    def benchmark(self, *args, out) -> float:
        self.bmreq.benchmark_with_cudagraphs = self._benchmark_with_cudagraphs
        return self.bmreq.benchmark(*args, out=out)

    def output_node(self) -> TensorBox:
        from torch._inductor.ir import NVUniversalGemmBuffer

        # Without memoization, each call registers a new buffer (via
        # TemplateBuffer.__init__ → V.graph.register_buffer), leaking orphan
        # buffers into the graph's name tables during EFC benchmarking.
        #
        # But a buffer registered during a benchmark sub-context can have its
        # name freed when that context rolls back `V.graph.buffers` (names are
        # `buf{len(buffers)}`, so a freed slot is reused). A stale cached buffer
        # then collides with a fresh buffer that reused its name, corrupting
        # finalize_multi_template_buffers. Drop the cache entry in that case and
        # rebuild a properly-registered buffer.
        if self._cached_output_node is not None:
            # pyrefly: ignore [missing-attribute]
            cached_name = self._cached_output_node.data.data.get_name()
            if cached_name in V.graph.name_to_buffer:
                return self._cached_output_node
            self._cached_output_node = None

        # For swap_ab, use the original (M, N) layout so the scheduler and
        # downstream IR see the expected shape. The runtime handles the
        # transpose internally via a temp buffer in _nvgemm_run.
        buffer = NVUniversalGemmBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            kernel=self.kernel,
            accumulator_type=self.accumulator_type,
            workspace_size=self.workspace_size,
            variant=self.variant,
            scale_type_a=self.scale_type_a,
            scale_type_b=self.scale_type_b,
            swizzle_type_a=self.swizzle_type_a,
            swizzle_type_b=self.swizzle_type_b,
            supports_epilogue_fusion=self.supports_epilogue_fusion,
            swap_ab=self.swap_ab,
            bias_node=self.bias_node,
        )
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]

        self._cached_output_node = TensorBox.create(buffer)
        return self._cached_output_node

    def call_name(self) -> str:
        return self.name

    def to_callable(self):
        return self.bmreq.make_run_fn

    def hash_key(self) -> str:
        # `select_algorithm` uses this as a precompile dedup key. Two callers
        # wrapping the same physical kernel name but with different accumulator/
        # scale/swizzle types produce distinct compiled artifacts; collapsing
        # them here would silently drop the second from autotuning.
        return "_".join(
            str(x)
            for x in (
                self.variant.op_name,
                self.kernel.metadata.operator_name,
                self.accumulator_type,
                self.scale_type_a,
                self.scale_type_b,
                self.swizzle_type_a,
                self.swizzle_type_b,
                "swap_ab" if self.swap_ab else "",
            )
        )

    def info_dict(self) -> dict[str, Any]:
        info = {
            "name": self.name,
            "backend": self.variant.op_name,
            "kernel_name": self.kernel.metadata.operator_name,
        }
        if self.swap_ab:
            info["swap_ab"] = True
        return info

    def get_make_kernel_render(self):
        from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_kernel import (
            NVUniversalGemmKernel,
        )
        from torch._inductor.utils import Placeholder

        kernel_metadata = {
            "kernel_name": self.kernel.metadata.operator_name,
            "min_cc": self.kernel.designed_for_min_cc,
        }
        accumulator_type = self.accumulator_type
        workspace_size = self.workspace_size
        variant = self.variant
        scale_type_a = self.scale_type_a
        scale_type_b = self.scale_type_b
        swizzle_type_a = self.swizzle_type_a
        swizzle_type_b = self.swizzle_type_b
        input_nodes = self.input_nodes
        swap_ab = self.swap_ab
        has_bias = self.bias_node is not None

        def make_kernel_render(
            out_node,
            hint_override=None,
            epilogue_fn_code=None,
            epilogue_reads=None,
            epilogue_writes=None,
            epilogue_var_renames=None,
            local_reduce=None,
        ):
            from torch._inductor.ir import StorageBox, TensorBox

            processed_inputs = []
            for inp in input_nodes:
                if isinstance(inp, TensorBox):
                    inp = inp.data
                if isinstance(inp, StorageBox):
                    inp = inp.data
                processed_inputs.append(inp)

            # A baked bias is the last input, consumed by the epilogue rather
            # than as a GEMM operand.
            bias_node = None
            if has_bias:
                bias_node = processed_inputs[-1]
                processed_inputs = processed_inputs[:-1]

            kernel_name = str(Placeholder.KERNEL_NAME)

            render_kernel = NVUniversalGemmKernel(
                kernel_name=kernel_name,
                # pyrefly: ignore [bad-argument-type]
                input_nodes=processed_inputs,
                output_node=out_node,
                kernel_metadata=kernel_metadata,
                accumulator_type=accumulator_type,
                workspace_size=workspace_size,
                variant=variant,
                scale_type_a=scale_type_a,
                scale_type_b=scale_type_b,
                swizzle_type_a=swizzle_type_a,
                swizzle_type_b=swizzle_type_b,
                epilogue_fn_code=epilogue_fn_code,
                epilogue_reads=epilogue_reads,
                epilogue_writes=epilogue_writes,
                epilogue_var_renames=epilogue_var_renames,
                local_reduce=local_reduce,
                swap_ab=swap_ab,
                # pyrefly: ignore [bad-argument-type]
                bias_node=bias_node,
            )

            def render():
                return render_kernel.render()

            return render_kernel, render

        return make_kernel_render


def _create_dummy_tensor_from_layout(
    layout: Layout, dtype_override: torch.dtype | None = None
) -> torch.Tensor | None:
    """
    Create a FakeTensor from a Layout for kernel filtering.

    Uses Layout.get_example() which creates FakeTensors within V.fake_mode,
    avoiding real CUDA memory allocation. cutlass.operators only needs shape/stride/dtype
    metadata for its supports() checks.
    """
    try:
        result = layout.get_example()
        if dtype_override is not None and result.dtype != dtype_override:
            result = result.view(dtype_override)
        return result
    except Exception:
        # Broad: layout.get_example()/torch.empty_strided under fake mode can
        # raise a variety of unexpected errors (TypeError, AssertionError, etc.)
        # depending on stride/symint state. Failing to materialize a dummy
        # should never abort autotune — just skip this candidate.
        return None


_TILE_RE = re.compile(r"tile(\d+)x\d+x\d+")


def _include_efc_kernels_only(metadata) -> bool:
    """Filter to include only EFC (Epilogue Fusion Compatible) kernels.

    Excludes tile_M=64 EFC kernels: cutlass.operators has a broadcast bug in the
    epilogue thread operation for aux-tensor inputs with tile_M=64, and we
    don't yet know at autotune time whether fusion will consume aux tensors.
    Non-EFC kernels still cover tile_M=64, so plain GEMM autotune is unaffected.

    Strictly requires the kernel name to encode tile dims; if cutlass.operators ever
    changes the naming scheme, this raises rather than silently letting the
    broken tile_M=64 kernels through.
    """
    if "EFC" not in metadata.operator_class.__name__:
        return False
    match = _TILE_RE.search(metadata.operator_name)
    if match is None:
        raise RuntimeError(
            f"NVGEMM EFC kernel name does not match expected tile pattern "
            f"'tileMxNxK': {metadata.operator_name}. The tile_M=64 broadcast "
            f"workaround in _include_efc_kernels_only depends on this naming "
            f"convention; update the regex or move to metadata-based filtering."
        )
    return int(match.group(1)) >= 128


def _exclude_efc_kernels(metadata) -> bool:
    """Filter to exclude EFC kernels (for non-epilogue cases)."""
    return "EFC" not in metadata.operator_class.__name__


def _add_nv_gemm_choices_impl(
    choices: list[ChoiceCaller],
    layout: Layout,
    input_nodes: list[Buffer],
    variant: GemmVariant,
    accumulator_type: torch.dtype,
    mm_inputs: MMKernelInputs | None = None,
    scale_type_a: Any | None = None,
    scale_type_b: Any | None = None,
    swizzle_type_a: Any | None = None,
    swizzle_type_b: Any | None = None,
    swap_ab: bool = False,
    kernel_layout: Layout | None = None,
    bias_node: Buffer | None = None,
) -> None:
    """
    Unified implementation for adding NVIDIA Universal GEMM choices.

    Args:
        choices: List to append ChoiceCaller objects to
        layout: Output layout
        input_nodes: GEMM operand nodes (A, B)
        variant: The GEMM variant (determines behavior)
        accumulator_type: Accumulator dtype
        mm_inputs: Optional MMKernelInputs for heuristics
        scale_type_a: ScalingType for A (required for SCALED_GEMM)
        scale_type_b: ScalingType for B (required for SCALED_GEMM)
        swizzle_type_a: SwizzleType for A (required for SCALED_GEMM)
        swizzle_type_b: SwizzleType for B (required for SCALED_GEMM)
        bias_node: Optional addmm bias, consumed as a fixed bias-add epilogue.
            Requires EFC kernels; only these are considered when set.
    """
    from torch._inductor.utils import _ensure_fp4_dtype_registered

    _ensure_fp4_dtype_registered()

    from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
        partition_compatible_kernels,
    )

    # Create dummy tensors for cutlass.operators's supports() checks.
    # Pass node dtype to handle FP4 ReinterpretView (uint8 storage viewed as float4_e2m1fn_x2).
    dummy_tensors = [
        _create_dummy_tensor_from_layout(
            node.get_layout(), dtype_override=node.get_dtype()
        )
        for node in input_nodes
    ]

    if swap_ab and len(dummy_tensors) >= 2:
        # swap_ab: transpose mat_a/mat_b dummies (and swap scales when scaled).
        # Original: dummy[0]=mat_a(M,K) row, dummy[1]=mat_b(K,N) col
        # Swap: new_A=mat_b.t()=(N,K) row, new_B=mat_a.t()=(K,M) col
        d_a, d_b = dummy_tensors[0], dummy_tensors[1]
        if d_a is not None and d_b is not None:
            if len(dummy_tensors) >= 4:
                # scaled: new_scale_a=scale_b, new_scale_b=scale_a
                d_sa, d_sb = dummy_tensors[2], dummy_tensors[3]
                dummy_tensors = [d_b.t(), d_a.t(), d_sb, d_sa] + dummy_tensors[4:]
            else:
                dummy_tensors = [d_b.t(), d_a.t()] + dummy_tensors[2:]

    effective_layout = kernel_layout if kernel_layout is not None else layout
    out_tensor = _create_dummy_tensor_from_layout(effective_layout)

    if any(t is None for t in dummy_tensors) or out_tensor is None:
        log.debug("Failed to create dummy tensors for %s", variant.op_name)
        return

    helper_kwargs: dict[str, Any] = {}
    if variant == GemmVariant.SCALED_GEMM:
        try:
            scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b = (
                _get_scaled_gemm_modes(
                    scale_type_a,
                    swizzle_type_a,
                    scale_type_b,
                    swizzle_type_b,
                )
            )
        except NotImplementedError:
            return
        helper_kwargs = {
            "scale_mode_a": scale_mode_a,
            "swizzle_mode_a": swizzle_mode_a,
            "scale_mode_b": scale_mode_b,
            "swizzle_mode_b": swizzle_mode_b,
        }

    args = _create_gemm_arguments(
        variant.name,
        tuple(dummy_tensors),
        out_tensor,
        accumulator_type,
        **helper_kwargs,
    )

    cc = get_cuda_arch()
    if cc is None:
        log.debug("Failed to get CUDA arch")
        return
    cc_int = int(cc)

    # Single-pass partition over the ~390K-entry kernel cache. The two-pass
    # form below called `kernel.supports(args)` once per bucket -- i.e. twice
    # per non-EFC-class kernel -- across the full cache.
    def _classify(metadata) -> int:
        if _include_efc_kernels_only(metadata):
            return 1  # efc bucket (with tile_M >= 128)
        if _exclude_efc_kernels(metadata):
            return 0  # non-efc bucket
        # NOTE: tile_M < 128 EFC kernels are dropped due to a cutlass.operators
        # broadcast bug. Tracking: https://github.com/pytorch/pytorch/issues/181901
        return -1

    # A baked bias-add requires an epilogue-fusion-capable (EFC) kernel, so
    # scan only EFC kernels -- skips supports() on the far larger non-EFC set.
    # The args-filtered fast query is complete only for dense GEMM; scaled uses
    # direct block-scaled sub-provider enumeration (get_operators under-generates
    # operands from scaled args); anything else falls back to the full manifest.
    candidate_source: Literal["args", "scaled", "manifest"]
    if variant == GemmVariant.GEMM:
        candidate_source = "args"
    elif variant == GemmVariant.SCALED_GEMM:
        candidate_source = "scaled"
    else:
        candidate_source = "manifest"
    non_efc_kernels, efc_kernels = partition_compatible_kernels(
        args,
        cc_int,
        _classify,
        num_buckets=2,
        efc_only=bias_node is not None,
        candidate_source=candidate_source,
        classifier_key="nvgemm_efc_partition_v1",
    )
    if not config.epilogue_fusion or swap_ab:
        efc_kernels = []
    if bias_node is not None:
        non_efc_kernels = []
    if not non_efc_kernels and not efc_kernels:
        log.debug("No compatible %s kernels found", variant.op_name)
        return

    max_configs = config.nvgemm_max_profiling_configs or max(
        len(non_efc_kernels), len(efc_kernels)
    )
    if variant in (GemmVariant.GEMM, GemmVariant.SCALED_GEMM) and mm_inputs is not None:
        heuristics = get_nvgemm_heuristics()
        non_efc_kernels = heuristics.filter_kernels(
            non_efc_kernels, mm_inputs, max_configs, accumulator_type
        )
        efc_kernels = heuristics.filter_kernels(
            efc_kernels, mm_inputs, max_configs, accumulator_type
        )
    else:
        # TODO(nikhilap): Enable heuristics for grouped GEMM
        # when nvMatmulHeuristics adds support
        non_efc_kernels = non_efc_kernels[:max_configs]
        efc_kernels = efc_kernels[:max_configs]

    all_kernels = [(kernel, False) for kernel in non_efc_kernels] + [
        (kernel, True) for kernel in efc_kernels
    ]

    # When a bias is baked in, it is appended as the last caller input so the
    # scheduler tracks it as a read; the caller/kernel split it back off.
    caller_input_nodes = (
        [*input_nodes, bias_node] if bias_node is not None else input_nodes
    )

    num_added = 0
    for kernel, supports_epilogue_fusion in all_kernels:
        name = f"{variant.op_name}_{next(NVUniversalGemmCaller.index_counter)}"
        workspace_size = kernel.get_workspace_size(args).size_bytes
        try:
            caller = NVUniversalGemmCaller(
                name=name,
                input_nodes=caller_input_nodes,
                layout=layout,
                kernel=kernel,
                accumulator_type=accumulator_type,
                workspace_size=workspace_size,
                variant=variant,
                scale_type_a=scale_type_a,
                scale_type_b=scale_type_b,
                swizzle_type_a=swizzle_type_a,
                swizzle_type_b=swizzle_type_b,
                supports_epilogue_fusion=supports_epilogue_fusion,
                swap_ab=swap_ab,
                kernel_layout=kernel_layout,
                bias_node=bias_node,
            )
            choices.append(caller)
            num_added += 1
        except Exception:
            # Broad: caller construction touches cutlass.operators / fake-mode tensors
            # which can raise types other than RuntimeError/ValueError. A single
            # bad choice should never abort the rest of autotune choice population.
            log.debug("Failed to create %s choice", variant.op_name, exc_info=True)

    log.debug("Added %d %s choices", num_added, variant.op_name)


def add_nv_universal_gemm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    inputs: MMKernelInputs,
    accumulator_type: torch.dtype | None = None,
) -> None:
    """
    Add NVIDIA Universal GEMM kernels to the autotune choices.

    Thin wrapper around _add_nv_gemm_choices_impl for regular GEMM.
    """
    if not ensure_nv_universal_gemm_available():
        log.debug(
            "cutlass.operators not available, skipping NVIDIA Universal GEMM choices"
        )
        return

    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=inputs.nodes(),
        variant=GemmVariant.GEMM,
        accumulator_type=accumulator_type or torch.float32,
        mm_inputs=inputs,
    )

    # swap_ab: swap A/B so the large N goes on the M-axis, improving tile
    # utilization for small-M / tall-skinny shapes (M << N). The kernel computes
    # (N, M) and the runtime writes it into a transposed view of the (M, N)
    # output. Mirrors the scaled path (see add_nv_universal_scaled_gemm_choices)
    # but without scale operands. Only valid for 2D mm: this entry point also
    # serves BMM (layout [B, M, N]), where the (M, N) / 2D-PermuteView swap logic
    # would be wrong, so skip it there.
    if not config.nvgemm_swap_ab or len(layout.size) != 2:
        return
    m, n = layout.size[0], layout.size[1]
    swap_kernel_layout = FixedLayout(layout.device, layout.dtype, [n, m])
    # Rank swap configs with the heuristic on the SWAPPED (N, M, K) problem:
    # new mat1 = mat_b.t() (N, K) row, new mat2 = mat_a.t() (K, M) col. Without
    # this, the swap variant would fall back to an unranked prefix and miss the
    # small-tile configs that make swap_ab win.
    mat_a, mat_b = inputs.mat1mat2()
    swap_inputs = MMKernelInputs(
        [PermuteView.create(mat_b, [1, 0]), PermuteView.create(mat_a, [1, 0])]
    )
    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=inputs.nodes(),
        variant=GemmVariant.GEMM,
        accumulator_type=accumulator_type or torch.float32,
        mm_inputs=swap_inputs,
        swap_ab=True,
        kernel_layout=swap_kernel_layout,
    )


def add_nv_universal_addmm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    inputs: MMKernelInputs,
    bias_node: Buffer,
    alpha: float | int = 1,
    beta: float | int = 1,
    accumulator_type: torch.dtype | None = None,
) -> None:
    """Add NVGEMM addmm choices: ``bias + mat1 @ mat2`` via an EFC bias-add
    epilogue.

    Only the plain ``alpha == beta == 1`` case is handled; other scalings fall
    back to the remaining backends. ``bias_node`` should be the original
    (un-expanded) bias so a 1D row vector routes to the row-broadcast epilogue
    impl -- passing the (M, N) broadcast *view* is rejected by cutlass.operators.
    """
    if not ensure_nv_universal_gemm_available():
        return
    # Bias-add requires an epilogue-fusion-capable (EFC) kernel.
    if not config.epilogue_fusion:
        return
    if alpha != 1 or beta != 1:
        return

    # Only broadcastable row-vector or full biases are supported. The bias last
    # dim must match N; higher-rank or column-only biases are left to fallbacks.
    n = layout.size[-1]
    bias_size = bias_node.get_size()
    # Only rank-1 (row vector) or rank-2 (full) biases are supported. The
    # epilogue indexes bias_size[-1], so a rank-0 scalar bias would IndexError;
    # reject it (and higher-rank biases) so they fall through to other backends.
    if not 1 <= len(bias_size) <= 2:
        return
    if not V.graph.sizevars.statically_known_equals(bias_size[-1], n):
        return

    mat1, mat2 = inputs.mat1mat2()
    gemm_inputs = MMKernelInputs([mat1, mat2])
    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=[mat1, mat2],
        variant=GemmVariant.GEMM,
        accumulator_type=accumulator_type or torch.float32,
        mm_inputs=gemm_inputs,
        bias_node=bias_node,
    )


def add_nv_universal_grouped_gemm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    input_nodes: list[Buffer],
    accumulator_type: torch.dtype | None = None,
) -> None:
    """
    Add NVIDIA Universal Grouped GEMM kernels to the autotune choices.

    Thin wrapper around _add_nv_gemm_choices_impl for grouped GEMM.

    For grouped GEMM (contiguous offset variant):
    - A is (TotalM, K) with problems stacked along M
    - B is (G, K, N) where B[i] is the weight for problem i
    - offsets is (G,) marking where each problem ends in A
    - Output is (TotalM, N)
    """
    if not ensure_nv_universal_gemm_available():
        log.debug(
            "cutlass.operators not available, skipping NVIDIA Universal Grouped GEMM choices"
        )
        return

    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=input_nodes,
        variant=GemmVariant.GROUPED_GEMM,
        accumulator_type=accumulator_type or torch.float32,
    )


def add_nv_universal_scaled_gemm_choices(
    choices: list[ChoiceCaller],
    layout: Layout,
    input_nodes: list[Buffer],
    accumulator_type: torch.dtype | None = None,
    kernel_inputs: MMKernelInputs | None = None,
) -> None:
    """
    Add NVIDIA Universal Scaled GEMM (FP8) kernels to the autotune choices.

    The scaling type is inferred from the input shapes/dtypes.
    If the scaling mode is unsupported by NVGEMM, this function returns without
    adding any choices.
    """
    if not ensure_nv_universal_gemm_available():
        return

    from torch._inductor.utils import infer_scale_swizzle_ir

    if len(input_nodes) < 4:
        return

    mat_a, mat_b, scale_a, scale_b = input_nodes[:4]

    scale_type_a, swizzle_type_a = infer_scale_swizzle_ir(mat_a, scale_a)
    scale_type_b, swizzle_type_b = infer_scale_swizzle_ir(
        mat_b, scale_b, transpose=True
    )

    if scale_type_a is None or scale_type_b is None:
        return

    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=input_nodes,
        variant=GemmVariant.SCALED_GEMM,
        accumulator_type=accumulator_type or torch.float32,
        mm_inputs=kernel_inputs,
        scale_type_a=scale_type_a,
        scale_type_b=scale_type_b,
        swizzle_type_a=swizzle_type_a,
        swizzle_type_b=swizzle_type_b,
    )

    # swap_ab: see add_nv_universal_gemm_choices for the rationale (swap A/B so
    # the large N lands on the well-tiled M-axis for small-M shapes).
    if not config.nvgemm_swap_ab:
        return

    # In the IR, mat_a=(M, K/2) row-major, mat_b=(K/2, N) column-major (already .t()).
    # swap_ab computes: un_transpose(mat_b) @ transpose(mat_a) = (N,K/2)@(K/2,M) = (N,M).
    # Scales: new A uses scale_b (weight scale), new B uses scale_a (activation scale).
    # The un-transposed mat_b is (N, K/2) row-major — same as the original weight.
    # The transposed mat_a is (K/2, M) column-major.
    # Both have the same layout pattern as the original (row-major A, column-major B).

    # Scale inference: new A is the original weight (N, K/2), no transpose.
    # For infer_scale_swizzle_ir, we need to pass the un-transposed shape.
    # mat_b.t() gives (N, K/2) — but we use transpose=True on mat_b itself
    # since mat_b is (K/2, N), so transpose=True flips it to (N, K/2).
    swap_scale_type_a, swap_swizzle_type_a = infer_scale_swizzle_ir(
        mat_b, scale_b, transpose=True
    )
    # New B is the original activation (M, K/2), used transposed as (K/2, M).
    # infer_scale_swizzle_ir with transpose=True on mat_a flips (M, K/2) to (K/2, M).
    swap_scale_type_b, swap_swizzle_type_b = infer_scale_swizzle_ir(
        mat_a, scale_a, transpose=True
    )

    if swap_scale_type_a is None or swap_scale_type_b is None:
        return

    # Kernel output shape is (N, M) — the transpose of the original (M, N)
    m, n = layout.size[0], layout.size[1]
    swap_kernel_layout = FixedLayout(layout.device, layout.dtype, [n, m])

    # Skip heuristic filtering for swap_ab: mm_inputs has original (M, N, K) but
    # the swapped kernel sees (N, M, K). Let the benchmark pick the best kernel.
    _add_nv_gemm_choices_impl(
        choices=choices,
        layout=layout,
        input_nodes=input_nodes,
        variant=GemmVariant.SCALED_GEMM,
        accumulator_type=accumulator_type or torch.float32,
        mm_inputs=None,
        scale_type_a=swap_scale_type_a,
        scale_type_b=swap_scale_type_b,
        swizzle_type_a=swap_swizzle_type_a,
        swizzle_type_b=swap_swizzle_type_b,
        swap_ab=True,
        kernel_layout=swap_kernel_layout,
    )
<|endoftext|># mypy: allow-untyped-defs
"""
NVIDIA Universal GEMM kernel code generation.

This module generates Python code that calls cutlass.operators to execute GEMM operations.
The runtime helpers (_nvgemm_run, _nvgemm_precompile, etc.) are imported by the
generated wrapper at runtime, keeping the generated code thin.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import re
import threading
from typing import Any, TYPE_CHECKING

from torch._inductor.codegen.common import (
    IndentedBuffer,
    Kernel,
    WorkspaceArg,
    WorkspaceZeroMode,
)
from torch._inductor.codegen.cutedsl.cutedsl_op_overrides import CuteDSLOpOverrides
from torch._inductor.codegen.cutlass.python_evt import _ACCUMULATOR_ARG_NAME
from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm_utils import (
    to_cutlass_scale_mode,
)
from torch._inductor.ir import (
    BaseView,
    Buffer,
    ExternKernel,
    MutableBox,
    ReinterpretView,
)
from torch._inductor.virtualized import V


if TYPE_CHECKING:
    from torch._inductor.codegen.nv_universal_gemm.nv_universal_gemm import GemmVariant
    from torch._inductor.kernel.gemm_epilogue import GemmReductionPlan


log = logging.getLogger(__name__)


def _current_target_sm(dev_idx: int):
    """TargetSm for the current CUDA device, used to stamp disk-cached artifacts."""
    from cutlass.operators.arch import TargetSm

    import torch

    cc = torch.cuda.get_device_capability(dev_idx)
    return TargetSm(cc=cc[0] * 10 + cc[1])


def _make_disk_config_key(
    kernel_name: str,
    variant_name: str,
    accumulator_type: Any,
    scale_type_a: Any | None = None,
    scale_type_b: Any | None = None,
    swizzle_type_a: Any | None = None,
    swizzle_type_b: Any | None = None,
) -> tuple:
    return (
        kernel_name,
        variant_name,
        str(accumulator_type),
        str(scale_type_a),
        str(scale_type_b),
        str(swizzle_type_a),
        str(swizzle_type_b),
    )


def _compile_nvgemm(
    variant_name,
    input_tensors,
    out,
    accumulator_type,
    *,
    kernel_obj=None,
    kernel_name: str | None = None,
    args_kwargs=None,
    epilogue_args=None,
    epilogue_source="",
    fallback_fn=None,
    cc: int | None = None,
    base_kernel=None,
):
    """Compile an NVGEMM artifact, trying a fallback (disk cache) first.

    Thread safety is handled at the dispatch layer: autotuning precompile
    runs in subprocess workers (process-isolated), runtime is
    single-threaded per graph execution.

    kernel_obj: pre-resolved kernel (skips _lookup_gemm_kernel).
    kernel_name: kernel name for _lookup_gemm_kernel.
    base_kernel: pre-reconstructed operator (from metadata passed by the main
        process); lets _lookup_gemm_kernel skip operator discovery entirely.
    args_kwargs: extra kwargs forwarded to _create_gemm_arguments.
    fallback_fn: callable(kernel) -> artifact | None, called before
        compiling (for disk cache lookup).

    Returns (artifact, args, kernel, was_compiled).
    """
    was_compiled = False

    args = _create_gemm_arguments(
        variant_name,
        input_tensors,
        out,
        accumulator_type,
        epilogue=epilogue_args,
        **(args_kwargs or {}),
    )
    if kernel_obj is not None:
        kernel = kernel_obj
    else:
        kernel = _lookup_gemm_kernel(
            kernel_name,  # pyrefly: ignore[bad-argument-type]
            epilogue_args=epilogue_args,
            epilogue_source=epilogue_source,
            args=args if cc is not None else None,
            cc=cc,
            base_kernel=base_kernel,
        )

    artifact = None
    if fallback_fn is not None:
        artifact = fallback_fn(kernel)
    if artifact is None:
        artifact = kernel.compile(args)
        was_compiled = True

    return artifact, args, kernel, was_compiled


class CUDAContextMetadata:
    """CUDA device metadata that subprocess workers can't query on their own.

    Subprocess compile workers don't have CUDA initialized (and can't due to
    fork restrictions). This bundles the values the main process must provide.
    """

    __slots__ = ("max_active_clusters", "device_capability")

    def __init__(
        self,
        max_active_clusters: int | None,
        device_capability: tuple[int, int],
    ):
        self.max_active_clusters = max_active_clusters
        self.device_capability = device_capability

    @staticmethod
    def from_kernel(kernel, device) -> CUDAContextMetadata:
        """Build from a cutlass.operators Kernel and torch device in the main process."""
        import torch

        mac = None
        try:
            cluster_shape = getattr(
                getattr(kernel, "impl", None), "cluster_shape_mn", None
            )
            if cluster_shape is not None:
                from cutlass.operators.providers.cutedsl.integration_utils.mma import (
                    get_max_active_clusters,
                )

                mac = get_max_active_clusters(cluster_shape)
        except Exception:
            log.warning("Failed to query max_active_clusters", exc_info=True)
        cc = torch.cuda.get_device_capability(device)
        return CUDAContextMetadata(mac, cc)


def _patch_max_active_clusters(max_active_clusters):
    """Patch get_max_active_clusters in all CuTeDSL modules.

    Returns a list of (module, original_fn) pairs for restoration.
    Subprocess workers don't have CUDA initialized, so they can't query
    the driver; this patches in the pre-computed value from the main process.
    """
    patched = []

    def mac_fn(*a, **kw):
        return max_active_clusters

    for mod_name in _MAX_ACTIVE_CLUSTERS_MODULES:
        try:
            m = importlib.import_module(mod_name)
        except ImportError:
            continue
        if hasattr(m, "get_max_active_clusters"):
            patched.append((m, m.get_max_active_clusters))
            m.get_max_active_clusters = mac_fn  # pyrefly: ignore [missing-attribute]
    return patched


def _restore_max_active_clusters(patched):
    """Undo _patch_max_active_clusters."""
    for m, orig in patched:
        m.get_max_active_clusters = orig  # pyrefly: ignore [missing-attribute]


def _worker_nvgemm_autotuning_precompile(
    kernel_name,
    variant_name,
    accumulator_type,
    input_tensor_meta,
    output_tensor_meta,
    extra_env,
    cuda_ctx,
    scale_type_a=None,
    scale_type_b=None,
    swizzle_type_a=None,
    swizzle_type_b=None,
    has_bias_epilogue=False,
    swap_ab=False,
    metadata=None,
):
    """Subprocess worker: compile one NVGEMM kernel and save to disk cache.

    Called from AsyncCompile.nvgemm_precompile() in a subprocess worker
    so CuTeDSL compilation is process-isolated (no thread-safety issues).
    Returns (None, elapsed_us) for compatibility with the precompile callback.
    """
    import time

    # Share the persistent-worker env/cache handling used by PyCodeCache workers.
    from torch._inductor.runtime.compile_tasks import (
        _apply_subprocess_env_and_clear_caches,
    )

    _apply_subprocess_env_and_clear_caches(extra_env)

    start_ns = time.time_ns()

    import torch
    from torch._inductor.runtime.cutedsl_cache import disk_cache_set
    from torch._inductor.utils import _ensure_fp4_dtype_registered
    from torch._subclasses.fake_tensor import FakeTensorMode

    _ensure_fp4_dtype_registered()

    with FakeTensorMode():
        input_tensors = tuple(
            torch.empty_strided(m.sizes, m.strides, device=m.device, dtype=m.dtype)
            for m in input_tensor_meta
        )
        out = torch.empty_strided(
            output_tensor_meta.sizes,
            output_tensor_meta.strides,
            device=output_tensor_meta.device,
            dtype=output_tensor_meta.dtype,
        )

    # swap_ab: the kernel was selected for the transposed (N, M) problem, so the
    # worker must swap operands here too -- both to resolve the kernel via the
    # args-filtered fast lookup and to key the disk cache the same way the
    # benchmark's make_run_fn does (it swaps before compiling). Mirrors the
    # runtime swap in _nvgemm_run. swap_ab and a bias epilogue never co-occur.
    if swap_ab and len(input_tensors) >= 2:
        a, b = input_tensors[0], input_tensors[1]
        if len(input_tensors) >= 4:
            sa, sb = input_tensors[2], input_tensors[3]
            input_tensors = (b.t(), a.t(), sb, sa) + input_tensors[4:]
        else:
            input_tensors = (b.t(), a.t()) + input_tensors[2:]
        out = out.t()

    helper_kwargs: dict[str, Any] = {}
    if variant_name == "SCALED_GEMM":
        scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b = (
            _get_scaled_gemm_modes(
                scale_type_a, swizzle_type_a, scale_type_b, swizzle_type_b
            )
        )
        helper_kwargs = {
            "scale_mode_a": scale_mode_a,
            "swizzle_mode_a": swizzle_mode_a,
            "scale_mode_b": scale_mode_b,
            "swizzle_mode_b": swizzle_mode_b,
        }

    # For an addmm bias choice the last input is the bias, consumed by a
    # bias-add epilogue; the rest are the GEMM operands. Building the epilogue
    # (and keying the cache with it) here lets the parallel precompile hand off
    # to the benchmark, which reads the same key.
    epilogue_args = None
    epilogue_source = ""
    aux_tensors: tuple = ()
    if has_bias_epilogue:
        from cutlass.operators.arguments import EpilogueArguments

        *gemm_list, bias = input_tensors
        input_tensors = tuple(gemm_list)
        epilogue_args = EpilogueArguments(_nvgemm_bias_add_epilogue, bias=bias, D=out)
        epilogue_source = "nvgemm_addmm_bias_v1"
        aux_tensors = (bias,)

    cache_key = _create_gemm_cache_key(
        input_tensors, out, has_epilogue=has_bias_epilogue, aux_tensors=aux_tensors
    )
    dev_idx = input_tensors[0].device.index or 0
    disk_config_key = _make_disk_config_key(
        kernel_name,
        variant_name,
        accumulator_type,
        scale_type_a,
        scale_type_b,
        swizzle_type_a,
        swizzle_type_b,
    )
    disk_fn_cache: dict = {}

    # cc for the args-filtered fast kernel lookup: workers can't query the CUDA
    # driver, so use the bundled device capability instead of rebuilding the
    # full ~294K-kernel manifest just to resolve one kernel by name.
    major, minor = cuda_ctx.device_capability
    worker_cc = major * 10 + minor

    patched = _patch_max_active_clusters(cuda_ctx.max_active_clusters)
    try:
        # Best path: the main process passed the chosen operator's metadata, so
        # reconstruct exactly that one operator (~us) instead of enumerating the
        # operator space. Done inside the patched region since construction may
        # query max_active_clusters, which the worker can't get from the driver.
        base_kernel = None
        if metadata is not None:
            try:
                base_kernel = metadata.operator_class(metadata)
            except Exception:
                log.warning(
                    "Failed to reconstruct NVGEMM operator %s from metadata; "
                    "falling back to argument-based lookup",
                    kernel_name,
                    exc_info=True,
                )

        artifact, _, _, was_compiled = _compile_nvgemm(
            variant_name,
            input_tensors,
            out,
            accumulator_type,
            kernel_name=kernel_name,
            args_kwargs=helper_kwargs,
            epilogue_args=epilogue_args,
            epilogue_source=epilogue_source,
            cc=worker_cc,
            base_kernel=base_kernel,
        )

        if was_compiled:
            obj = _unwrap_efc_compiled_obj(artifact.compiled_obj)
            disk_cache_set(
                disk_fn_cache,
                kernel_name,
                disk_config_key,
                cache_key,
                obj,
                dev_idx,
                device_capability=cuda_ctx.device_capability,
            )
    finally:
        _restore_max_active_clusters(patched)

    elapsed_us = (time.time_ns() - start_ns) // 1000
    return None, elapsed_us


# ── Runtime helpers (imported by generated wrapper code at runtime) ───────────


def _get_scaled_gemm_modes(
    scale_type_a: Any | None,
    swizzle_type_a: Any | None,
    scale_type_b: Any | None,
    swizzle_type_b: Any | None,
) -> tuple[Any, Any, Any, Any]:
    scale_mode_a, swizzle_mode_a = to_cutlass_scale_mode(scale_type_a, swizzle_type_a)
    scale_mode_b, swizzle_mode_b = to_cutlass_scale_mode(scale_type_b, swizzle_type_b)
    if any(
        mode is None
        for mode in (scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b)
    ):
        raise NotImplementedError("Unsupported scale/swizzle mode for scaled GEMM")
    return scale_mode_a, swizzle_mode_a, scale_mode_b, swizzle_mode_b


def _create_gemm_arguments(
    variant_name: str,
    input_tensors,
    out,
    accumulator_type: Any,
    *,
    scale_mode_a: Any | None = None,
    swizzle_mode_a: Any | None = None,
    scale_mode_b: Any | None = None,
    swizzle_mode_b: Any | None = None,
    epilogue: Any | None = None,
    local_reduce_out: Any | None = None,
    local_reduce_group: int = 0,
    local_reduce_axis: int = 1,
    local_reduce_type: str = "sum",
    local_reduce_source: str = "identity",
    local_reduce_feeds_main: bool = False,
):
    import cutlass.operators

    has_local_reduce = local_reduce_out is not None or local_reduce_feeds_main
    if has_local_reduce and variant_name != "SCALED_GEMM":
        raise NotImplementedError(
            "NVGEMM local reductions currently require scaled GEMM"
        )
    if has_local_reduce and (
        local_reduce_axis not in (0, 1) or local_reduce_group <= 1
    ):
        raise NotImplementedError(
            "NVGEMM local reductions require an M- or N-axis group greater than 1"
        )

    def with_local_reduce(args):
        if not has_local_reduce:
            return args
        from cutlass.operators.utils.tensor import TensorWrapper

        args.local_reduce_out = (
            TensorWrapper(local_reduce_out) if local_reduce_out is not None else None
        )
        args.local_reduce_group = local_reduce_group
        args.local_reduce_axis = local_reduce_axis
        args.local_reduce_type = local_reduce_type
        args.local_reduce_source = local_reduce_source
        args.local_reduce_feeds_main = local_reduce_feeds_main
        return args

    if epilogue is not None and variant_name == "GROUPED_GEMM":
        raise NotImplementedError(
            "Epilogue fusion is not yet supported for grouped GEMM"
        )

    match variant_name:
        case "GROUPED_GEMM":
            a, b, offsets = input_tensors
            return cutlass.operators.arguments.GroupedGemmArguments(
                a,
                b,
                out,
                accumulator_type=accumulator_type,
                offsets=offsets,
            )

        case "SCALED_GEMM":
            from cutlass.operators.arguments import ScaledOperand

            a, b, scale_a, scale_b = input_tensors
            scaled_a = ScaledOperand(a, scale_a, scale_mode_a, swizzle_mode_a)
            scaled_b = ScaledOperand(b, scale_b, scale_mode_b, swizzle_mode_b)
            kwargs: dict[str, Any] = {"accumulator_type": accumulator_type}
            if epilogue is not None:
                kwargs["epilogue"] = epilogue
            return with_local_reduce(
                cutlass.operators.arguments.GemmArguments(
                    scaled_a,
                    scaled_b,
                    out,
                    **kwargs,
                )
            )

        case "GEMM":
            a, b = input_tensors
            kwargs: dict[str, Any] = {"accumulator_type": accumulator_type}
            if epilogue is not None:
                kwargs["epilogue"] = epilogue
            return with_local_reduce(
                cutlass.operators.arguments.GemmArguments(a, b, out, **kwargs)
            )

        case _:
            raise NotImplementedError(f"Unsupported NVGEMM variant: {variant_name}")


def _lookup_gemm_kernel(
    kernel_name: str,
    *,
    epilogue_args: Any | None = None,
    epilogue_source: str = "",
    args: Any | None = None,
    cc: int | None = None,
    base_kernel: Any | None = None,
):
    from torch._inductor.codegen.nv_universal_gemm.kernel_cache import (
        get_efc_kernel_with_epilogue,
        get_kernel_by_name,
        get_kernel_by_name_via_args,
    )

    # Best path (subprocess precompile): the caller reconstructed the exact
    # operator from metadata passed by the main process (operator_class(metadata),
    # ~us), so we skip operator discovery entirely -- no args-filtered query and
    # no full manifest.
    # Fast path (base_kernel absent): resolve the single named kernel via an
    # args-filtered query (~0.05s) instead of building the full manifest (~14s).
    fast = args is not None and cc is not None

    if epilogue_args is None:
        kernel = base_kernel
        # Fast path can miss when the named kernel doesn't support these exact
        # args (e.g. a swap_ab kernel selected for the transposed problem while
        # the worker holds the original operands); fall back to the manifest.
        if kernel is None and fast:
            kernel = get_kernel_by_name_via_args(kernel_name, args, cc)
        if kernel is None:
            kernel = get_kernel_by_name(kernel_name)
        if kernel is None:
            raise RuntimeError(f"Could not find kernel: {kernel_name}")
        return kernel

    if base_kernel is None and fast:
        base_kernel = get_kernel_by_name_via_args(kernel_name, args, cc)
    kernel = get_efc_kernel_with_epilogue(
        kernel_name,
        epilogue_args,
        epilogue_source=epilogue_source,
        base_kernel=base_kernel,
    )
    if kernel is None:
        raise RuntimeError(f"Could not find EFC kernel: {kernel_name}")
    return kernel


def _tensor_sig(t):
    return (t.shape, t.stride(), t.dtype)


def _create_gemm_cache_key(
    input_tensors,
    out,
    *,
    has_epilogue: bool = False,
    aux_tensors: tuple = (),
    epilogue_specialization: tuple = (),
):
    cache_key = tuple(s for t in input_tensors for s in _tensor_sig(t))
    cache_key = (*cache_key, *_tensor_sig(out))

    if has_epilogue:
        aux_sig = tuple(_tensor_sig(t) for t in aux_tensors)
        return (*cache_key, "epilogue", aux_sig, epilogue_specialization)
    return cache_key


def _unwrap_efc_compiled_obj(compiled_obj):
    """Extract the inner JIT function from an EFC closure for disk serialization."""
    if hasattr(compiled_obj, "__closure__") and compiled_obj.__closure__:
        inner = compiled_obj.__closure__[0].cell_contents
        if hasattr(inner, "export_to_c"):
            return inner
    return compiled_obj


def _init_efc_jit(kernel, epilogue_args):
    """Lightweight EFC JIT init from epilogue args, without a full kernel.compile().

    EFC.compile() calls analyze_epilogue_with_arguments() then creates the JIT
    object. We replicate just those two steps so that disk-cached artifacts can
    be rewrapped without recompiling the CuTe DSL kernel.
    """
    from cutlass.operators.providers.cutedsl.evt.common_efc import EFC
    from cutlass.operators.utils.tensor import TensorWrapper

    efc = kernel.impl.efc
    params = [
        e.compile_time_tensor if isinstance(e, TensorWrapper) else e
        for e in epilogue_args.parameters
    ]
    efc.analyze_epilogue_with_arguments(params)
    efc.jit = EFC.JIT(efc, efc.epilogue_parameter_names)


def _rewrap_efc_compiled_obj(compiled_fn, kernel, epilogue_args=None):
    """Reconstruct the EFC wrapped_launch closure from a loaded JIT function.

    When epilogue_args is provided and efc.jit is missing, performs a
    lightweight initialization (analyze + JIT creation) so that
    disk-cached EFC artifacts can be reused without a full recompile.
    Falls back to returning None (triggering recompile) only when
    epilogue_args is not available.
    """
    if not hasattr(kernel, "impl") or not hasattr(kernel.impl, "efc"):
        return compiled_fn
    if not hasattr(kernel.impl.efc, "jit"):
        if epilogue_args is not None:
            _init_efc_jit(kernel, epilogue_args)
        else:
            return None

    from cutlass.operators.providers.cutedsl.gemm.sm100_static_persistent_efc import (
        Operand as KernelOperand,
        TensorWrapper,
    )

    def wrapped_launch(a_tensor, b_tensor, stream, *supplemental_args):
        runtime_args = [
            e.runtime_tensor
            if isinstance(e, TensorWrapper)
            else (e.tensor.runtime_tensor if isinstance(e, KernelOperand) else e)
            for e in supplemental_args
        ]
        # A disk-reloaded artifact is the raw kernel module func: it takes the
        # epilogue tensors unpacked (a, b, stream, aux..., out), unlike the
        # freshly-JIT-compiled closure which expects pack_arguments(...). All
        # callers of _rewrap_efc_compiled_obj load from disk, so pass unpacked.
        return compiled_fn(a_tensor, b_tensor, stream, *runtime_args)

    return wrapped_launch


def _update_reuse_args_tensors(
    variant_name, args, input_tensors, out, epilogue_args, args_kwargs=None
) -> bool:
    """Redirect the cached args at this call's runtime tensors.

    A cache hit on ``mem_key`` guarantees identical shapes/strides/dtypes, so
    only the data pointers change; mutating ``TensorWrapper._runtime_tensor`` in
    place is sufficient and lets us skip the (expensive) _create_gemm_arguments
    rebuild. Returns False for GROUPED_GEMM (unsupported), signalling the caller
    to use the rebuild path.
    """
    if variant_name == "GROUPED_GEMM":
        return False
    if variant_name == "SCALED_GEMM":
        a, b, scale_a, scale_b = input_tensors
        args.A.quantized.tensor._runtime_tensor = a
        args.A.scale.tensor._runtime_tensor = scale_a
        args.B.quantized.tensor._runtime_tensor = b
        args.B.scale.tensor._runtime_tensor = scale_b
        args.out.tensor._runtime_tensor = out
    else:
        # GEMM: dense mm and the EFC bias/epilogue addmm path.
        a, b = input_tensors
        args.A.tensor._runtime_tensor = a
        args.B.tensor._runtime_tensor = b
        args.out.tensor._runtime_tensor = out
    epilogue = getattr(args, "epilogue", None)
    if epilogue is not None:
        for name, wrapper in epilogue.tensors.items():
            val = epilogue_args.tensors[name]
            wrapper._runtime_tensor = getattr(val, "runtime_tensor", val)
    local_reduce = getattr(args, "local_reduce_out", None)
    if local_reduce is not None:
        if args_kwargs is None:
            raise AssertionError("expected args_kwargs to be not None")
        local_reduce._runtime_tensor = args_kwargs["local_reduce_out"]
    return True


def _clear_reuse_args_tensors(variant_name, args, epilogue_args):
    """Drop the runtime tensors held by cached args after a launch.

    kernel.run enqueues the launch synchronously and reads the pointers during
    the call, and the caller keeps the tensors alive, so clearing our cached
    references is safe. Required for cudagraph capture, which fails if a cached
    object retains a tensor from the graph pool between iterations.
    """
    if variant_name == "SCALED_GEMM":
        args.A.quantized.tensor._runtime_tensor = None
        args.A.scale.tensor._runtime_tensor = None
        args.B.quantized.tensor._runtime_tensor = None
        args.B.scale.tensor._runtime_tensor = None
        args.out.tensor._runtime_tensor = None
    else:
        args.A.tensor._runtime_tensor = None
        args.B.tensor._runtime_tensor = None
        args.out.tensor._runtime_tensor = None
    epilogue = getattr(args, "epilogue", None)
    if epilogue is not None:
        for wrapper in epilogue.tensors.values():
            wrapper._runtime_tensor = None
        # kernel.run traces the epilogue on first launch, and the trace retains
        # the real output/aux tensors in example_inputs (used only for tracing,
        # not the launch). Replace them with meta tensors so the cached args
        # hold no real storage -- otherwise an intermediate output stays live
        # and CUDA graph capture fails. Mirrors _meta_epilogue_metadata.
        import torch

        te = getattr(epilogue, "traced_epilogue", None)
        example_inputs = getattr(te, "example_inputs", None) if te else None
        if example_inputs:
            for name, val in example_inputs.items():
                if isinstance(val, torch.Tensor) and val.device.type != "meta":
                    example_inputs[name] = torch.empty_strided(
                        val.shape, val.stride(), dtype=val.dtype, device="meta"
                    )
    local_reduce = getattr(args, "local_reduce_out", None)
    if local_reduce is not None:
        local_reduce._runtime_tensor = None


def _nvgemm_run(
    variant_name: str,
    kernel_name: str,
    input_tensors: tuple,
    out,
    accumulator_type,
    compiled_cache: dict,
    disk_fn_cache: dict,
    module_path: str,
    disk_config_key: tuple,
    *,
    stream=None,
    workspace=None,
    variant_kwargs: dict | None = None,
    epilogue_args=None,
    epilogue_source: str = "",
    has_epilogue: bool = False,
    aux_tensors: tuple = (),
    swap_ab: bool = False,
):
    if swap_ab and len(input_tensors) >= 2:
        a, b = input_tensors[0], input_tensors[1]
        if len(input_tensors) >= 4:
            sa, sb = input_tensors[2], input_tensors[3]
            input_tensors = (b.t(), a.t(), sb, sa) + input_tensors[4:]
        else:
            input_tensors = (b.t(), a.t()) + input_tensors[2:]
        # Swapped GEMM computes (N, M) = out.t(); write it zero-copy into a
        # transposed (column-major) view of the real (M, N) output -- no temp
        # buffer / copy (the kernel handles column-major C).
        out = out.t()
    from cutlass.operators.artifact import CompiledArtifact

    from torch._inductor.runtime.cutedsl_cache import disk_cache_get, disk_cache_set

    epilogue_specialization = tuple(
        (key, variant_kwargs[key])
        for key in ("local_reduce_type", "local_reduce_source")
        if variant_kwargs is not None and key in variant_kwargs
    )
    cache_key = _create_gemm_cache_key(
        input_tensors,
        out,
        has_epilogue=has_epilogue,
        aux_tensors=aux_tensors,
        epilogue_specialization=epilogue_specialization,
    )
    dev_idx = input_tensors[0].device.index or 0
    mem_key = (cache_key, dev_idx)

    if mem_key in compiled_cache:
        artifact = compiled_cache[mem_key]
        # Fast path: reuse the previously-built args, redirecting only the
        # runtime tensor pointers. Skips _create_gemm_arguments, whose cost
        # (get_type_hints + deepcopy + full epilogue/layout reprocessing)
        # depends solely on shapes/layout -- fixed for a given mem_key.
        reuse = getattr(artifact, "_nvgemm_reuse", None)
        if reuse is not None:
            args, kernel, reuse_lock = reuse
            # The cached args are shared mutable state. Serialize the
            # redirect->launch->clear so concurrent launches on the same cached
            # artifact (e.g. two graph executions sharing this kernel) don't
            # stomp each other's runtime tensor pointers.
            with reuse_lock:
                if _update_reuse_args_tensors(
                    variant_name,
                    args,
                    input_tensors,
                    out,
                    epilogue_args,
                    variant_kwargs,
                ):
                    try:
                        kernel.run(
                            args,
                            artifact,
                            stream=stream,
                            workspace=workspace,
                            assume_supported_args=True,
                        )
                    finally:
                        _clear_reuse_args_tensors(variant_name, args, epilogue_args)
                    return
        # No reusable args (e.g. artifact loaded from disk by a subprocess
        # precompile): build them once, then attach below for future calls.
        args = _create_gemm_arguments(
            variant_name,
            input_tensors,
            out,
            accumulator_type,
            epilogue=epilogue_args,
            **(variant_kwargs or {}),
        )
        kernel = artifact.operator_obj
    else:

        def disk_fallback(kernel):
            compiled_fn = disk_cache_get(
                disk_fn_cache,
                module_path,
                disk_config_key,
                cache_key,
                dev_idx,
            )
            if compiled_fn is not None:
                compiled_fn = _rewrap_efc_compiled_obj(
                    compiled_fn,
                    kernel,
                    epilogue_args=epilogue_args,
                )
            if compiled_fn is not None:
                return CompiledArtifact(
                    compiled_fn, kernel, _current_target_sm(dev_idx)
                )
            return None

        artifact, args, kernel, was_compiled = _compile_nvgemm(
            variant_name,
            input_tensors,
            out,
            accumulator_type,
            kernel_name=kernel_name,
            args_kwargs=variant_kwargs,
            epilogue_args=epilogue_args,
            epilogue_source=epilogue_source,
            fallback_fn=disk_fallback,
            cc=_current_target_sm(dev_idx).cc,
        )

        if was_compiled:
            disk_cache_set(
                disk_fn_cache,
                module_path,
                disk_config_key,
                cache_key,
                _unwrap_efc_compiled_obj(artifact.compiled_obj),
                dev_idx,
            )

        compiled_cache[mem_key] = artifact

    # Attach the built args for reuse on subsequent calls (skipped for
    # GROUPED_GEMM, where _update_reuse_args_tensors returns False). This runs
    # for both the disk-loaded rebuild branch and the fresh-compile branch.
    reuse_supported = _update_reuse_args_tensors(
        variant_name,
        args,
        input_tensors,
        out,
        epilogue_args,
        variant_kwargs,
    )
    if reuse_supported:
        # Publish and launch under the lock so a concurrent reuse-path caller
        # that picks up these args blocks until this launch has cleared them.
        reuse_lock = threading.Lock()
        with reuse_lock:
            artifact._nvgemm_reuse = (args, kernel, reuse_lock)
            try:
                kernel.run(
                    args,
                    artifact,
                    stream=stream,
                    workspace=workspace,
                    assume_supported_args=True,
                )
            finally:
                _clear_reuse_args_tensors(variant_name, args, epilogue_args)
    else:
        kernel.run(
            args,
            artifact,
            stream=stream,
            workspace=workspace,
            assume_supported_args=True,
        )


# Patch both the canonical definition (integration_utils.mma) for callers that
# reference it module-qualified, and each gemm module's namespace for callers
# that imported the bare name. _patch_max_active_clusters skips any without it.
_MAX_ACTIVE_CLUSTERS_MODULES = [
    "cutlass.operators.providers.cutedsl.integration_utils.mma",
    "cutlass.operators.providers.cutedsl.gemm.sm100_persistent",
    "cutlass.operators.providers.cutedsl.gemm.sm100_persistent_preferred_cluster",
    "cutlass.operators.providers.cutedsl.gemm.sm100_static_persistent_efc",
    "cutlass.operators.providers.cutedsl.gemm.sm100_dense_blockscaled_static_persistent",
    "cutlass.operators.providers.cutedsl.gemm.sm100_contiguous_offset_2d3d_dense_gemm",
    "cutlass.operators.providers.cutedsl.gemm.sm100_mixed_input",
    "cutlass.operators.providers.cutedsl.gemm.sm90_static_persistent",
]


def _nvgemm_precompile(
    precompile_shapes: dict,
    precompile_strides: dict,
    precompile_dtypes: dict,
    device_index: int = 0,
    device_capability: tuple | None = None,
    *,
    variant_name: str,
    kernel_name: str,
    accumulator_type,
    compiled_cache: dict,
    disk_fn_cache: dict,
    module_path: str,
    disk_config_key: tuple,
    input_param_names: list[str],
    variant_kwargs: dict | None = None,
    max_active_clusters: int | None = None,
):
    """Precompile an NVGEMM kernel in a subprocess for parallel compilation.

    Precompiles only the base (non-EFC) kernel. EFC kernels produce
    closure-wrapped artifacts that can't be serialized to disk cache.
    """
    import torch
    from torch._inductor.runtime.cutedsl_cache import disk_cache_set
    from torch._subclasses.fake_tensor import FakeTensorMode

    if max_active_clusters is None:
        return

    # cutlass.operators queries device occupancy while compiling. In async-compile
    # workers forked from a CUDA-initialized parent, skip precompile.
    if torch.cuda._is_in_bad_fork():
        return

    device = f"cuda:{device_index}"
    with FakeTensorMode():
        tensors = {}
        for name in [*input_param_names, "output"]:
            tensors[name] = torch.empty_strided(
                tuple(precompile_shapes[name]),
                tuple(precompile_strides[name]),
                device=device,
                dtype=getattr(torch, precompile_dtypes[name]),
            )

    input_tensors = tuple(tensors[n] for n in input_param_names)
    out = tensors["output"]

    patched = _patch_max_active_clusters(max_active_clusters)
    try:
        cache_key = _create_gemm_cache_key(input_tensors, out)
        mem_key = (cache_key, device_index)
        if mem_key not in compiled_cache:
            artifact, _, _, _ = _compile_nvgemm(
                variant_name,
                input_tensors,
                out,
                accumulator_type,
                kernel_name=kernel_name,
                args_kwargs=variant_kwargs,
            )
            disk_cache_set(
                disk_fn_cache,
                module_path,
                disk_config_key,
                cache_key,
                artifact.compiled_obj,
                device_index,
                device_capability=device_capability,
            )
            compiled_cache[mem_key] = artifact
    finally:
        _restore_max_active_clusters(patched)


# ── Kernel wrapper (returned by async_compile.nv_universal_gemm) ─────────────


class NVUniversalGemmKernelWrapper:
    """Wrapper to provide .run() interface for NVIDIA Universal GEMM kernels."""

    def __init__(self, kernel_fn, kernel_path: str | None = None):
        self.kernel_fn = kernel_fn
        self.kernel_path = kernel_path

    def run(self, *args, stream=None, **kwargs):
        """Execute the NVIDIA Universal GEMM kernel."""
        return self.kernel_fn(*args, stream=stream, **kwargs)


# ── Kernel codegen class ─────────────────────────────────────────────────────


# Module-level bias-add epilogue for benchmark tracing. Must be a real
# (introspectable) function -- EpilogueArguments parses its source via
# inspect.getsource -- and must contain NO string literals (the AST tracer
# treats any constant as an immediate). The param name is deliberately not
# ``C`` (see _build_bias_epilogue) so a 1D bias routes to the row-broadcast impl.
def _nvgemm_bias_add_epilogue(accum, bias):
    D = accum + bias
    return D


def _build_bias_epilogue(
    bias_name: str, out_name: str
) -> tuple[str, list[str], list[str], dict[str, str]]:
    """Build the epilogue fields for an addmm bias-add (``accum + bias``).

    Uses the bias buffer name as the epilogue-fn parameter, matching
    CutlassEVTCodegen's convention. This deliberately avoids the name ``C``:
    cutlass.operators' LoadSrcImpl claims any epilogue param named ``C`` whose
    shape equals the output (ignoring stride), which shadows the row/column
    broadcast impls and silently mis-reads a broadcast (1D) bias.
    """
    epilogue_fn_code = (
        f"def _epilogue_fn(accum, {bias_name}):\n"
        f"    D = accum + {bias_name}\n"
        f"    return D"
    )
    epilogue_reads = [bias_name]
    epilogue_writes: list[str] = []
    epilogue_var_renames = {bias_name: bias_name, "D": out_name}
    return epilogue_fn_code, epilogue_reads, epilogue_writes, epilogue_var_renames


def _compose_bias_into_epilogue(
    epilogue_fn_code: str,
    epilogue_reads: list[str],
    epilogue_var_renames: dict[str, Any],
    bias_name: str,
) -> tuple[str, list[str], dict[str, Any]]:
    """Inject an addmm bias-add into a scheduler-fused epilogue.

    The EFC kernel's ``accum`` is the raw ``A @ B``; a fused pointwise epilogue
    written by CutlassEVTCodegen assumes its ``accum`` is the addmm output
    (``A @ B + bias``). We keep ``accum`` as the raw accumulator (required by
    the cutlass epilogue tracer) and rewrite the fused body to read a new
    ``biased = accum + bias`` value instead, adding ``bias`` as an aux input.
    """
    lines = epilogue_fn_code.splitlines()
    def_line = lines[0]
    body = lines[1:]
    # Add bias as the 2nd parameter (after the required `accum`), unless the
    # fused epilogue already reads the same buffer -- then it is already a
    # parameter and re-adding it would emit `def fn(accum, b, b)` (SyntaxError).
    if bias_name not in epilogue_reads:
        def_line = def_line.replace("(accum", f"(accum, {bias_name}", 1)
    # Rewrite fused-body references to the accumulator to the biased value.
    rewritten = [re.sub(r"\baccum\b", "biased", bl) for bl in body]
    composed = "\n".join([def_line, f"    biased = accum + {bias_name}", *rewritten])
    reads = list(epilogue_reads)
    if bias_name not in reads:
        reads.append(bias_name)
    renames = dict(epilogue_var_renames)
    renames[bias_name] = bias_name
    return composed, reads, renames


class NVUniversalGemmKernel(Kernel):
    """
    Kernel implementation for NVIDIA Universal GEMM.

    Generates a thin Python wrapper that delegates to runtime helpers
    (_nvgemm_run, _nvgemm_precompile) for the actual GEMM execution and
    compilation logic.
    """

    def __init__(
        self,
        kernel_name: str,
        input_nodes: list[Buffer],
        output_node: Buffer,
        kernel_metadata: dict[str, Any],
        accumulator_type: Any,
        variant: GemmVariant,
        workspace_size: int = 0,
        scale_type_a: Any | None = None,
        scale_type_b: Any | None = None,
        swizzle_type_a: Any | None = None,
