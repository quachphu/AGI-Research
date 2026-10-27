        if expr.has(sympy.I):
            raise ValueError(
                f"_maybe_realize_expr received a complex expression: {expr}. "
                "Tensor dimensions cannot be complex numbers."
            )
        if expr in (int_oo, sympy.oo):
            return sys.maxsize
        if expr in (-int_oo, -sympy.oo):
            return -sys.maxsize
        if nan_fallback is not None and (expr is sympy.nan or expr.has(sympy.nan)):
            return nan_fallback

    return None


def _guarding_hint_or_throw_base(
    shape_env: ShapeEnv,
    expr: sympy.Expr | sympy.Basic | int | bool,
    precomputed_replacements: dict[sympy.Expr, sympy.Symbol],
) -> int | bool:
    """
    Return a concrete integer hint for an expression that is safe to use for guarding.

    This function evaluates the expression using only backed-symbols hints. Unlike
    _optimization_hint_base(), this function does NOT use heuristics or fallback values
    for unbacked symbols.

    Use this when you need a hint value that will be used for a guarding decision.

    Args:
        shape_env: The ShapeEnv instance.
        expr: A sympy expression or integer to evaluate.
        precomputed_replacements: Precomputed replacements for PRECOMPUTED_SIZE symbols.

    Returns:
        The concrete integer value of the expression based on backed symbol hints.

    Raises:
        GuardOnDataDependentSymNode: If the expression contains unbacked symbols
        (data-dependent values) that cannot be resolved to concrete values.

    See Also:
        _optimization_hint_base: For cases where fallback/heuristic values are acceptable
            for unbacked symbols.
    """
    from torch.fx.experimental.symbolic_shapes import (
        has_free_unbacked_symbols,
        symbol_is_type,
        SymT,
    )

    # sympy.expand() doesn't work with boolean expressions like Or/And
    if isinstance(expr, sympy.Expr):
        expr = sympy.expand(expr).xreplace(shape_env.replacements)
    else:
        expr = sympy.sympify(expr).xreplace(shape_env.replacements)

    if isinstance(expr, sympy.Expr):
        expr = expr.expand(identity=True)

    result = _maybe_realize_expr(expr, None)
    if result is not None:
        return result

    if not isinstance(expr, sympy.Basic):
        raise RuntimeError("isinstance(expr, sympy.Basic)", expr, type(expr))

    if any(symbol_is_type(s, SymT.PRECOMPUTED_SIZE) for s in expr.free_symbols):  # type: ignore[attr-defined]
        expr = _sympy_subs(expr, precomputed_replacements)

    # TODO do we need sympy_subs, or just xreplace
    expr = _sympy_subs(expr, shape_env.backed_var_to_val)
    if isinstance(expr, sympy.Expr):
        expr = expr.expand(identity=True)

    if has_free_unbacked_symbols(expr):
        # Note: we could do better here and call
        # _maybe_evaluate_static(orig_expr, compute_hint=True)
        # but is it worth the overhead? probably not.
        raise shape_env._make_data_dependent_error(expr, expr)

    result = _maybe_realize_expr(expr, None)
    if result is None:
        raise RuntimeError("unexpected None!", expr)
    return result


def _get_unbacked_replacements(shape_env: ShapeEnv) -> dict[sympy.Expr, sympy.Expr]:
    """Builds a mapping from unbacked expressions to canonical equivalents
    using a union-find algorithm over deferred runtime asserts.
    Used by optimization_hint to resolve unbacked symbols to consistent values."""
    from collections import defaultdict

    from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols
    from torch.utils._ordered_set import OrderedSet

    if shape_env._unbacked_replacements is not None:
        return shape_env._unbacked_replacements

    class CanonicalExprFinder:
        """
        A disjoint-set/union-find data structure that can return the
        "canonical" expression for a group of equivalent expressions.
        - The canonical expression must come from the input eq_graph.
        - The heuristics used to choose a leader determines which
        expression becomes the canonical expression.
        """

        def __init__(self, eq_graph: dict[sympy.Expr, OrderedSet[sympy.Expr]]) -> None:
            self.eq_graph = eq_graph
            self.expressions = list(eq_graph.keys())
            self.reverse_expressions = {
                expr: i for i, expr in enumerate(self.expressions)
            }
            self.leader = list(range(len(self.expressions)))
            self.size = [1] * len(self.expressions)
            self._build_canonical_expr_mapping()

        def _build_canonical_expr_mapping(self) -> None:
            for expr, edges in self.eq_graph.items():
                for adj in edges:
                    self.union_expr(expr, adj)

        def union_expr(self, a: sympy.Expr, b: sympy.Expr) -> bool:
            return self.union(self.reverse_expressions[a], self.reverse_expressions[b])

        def union(self, a: int, b: int) -> bool:
            rootA = self.find(a)
            rootB = self.find(b)
            if rootA == rootB:
                return False
            leader, other = self.choose_leader(rootA, rootB)
            self.leader[other] = leader
            self.size[leader] += self.size[other]
            return True

        def find_expr(self, expr: sympy.Expr) -> sympy.Expr:
            parent = self.find(self.reverse_expressions[expr])
            return self.expressions[parent]

        def find(self, x: int) -> int:
            if self.leader[x] != x:
                self.leader[x] = self.find(self.leader[x])
            return self.leader[x]

        def choose_leader(self, a: int, b: int) -> tuple[int, int]:
            """
            The leader will become the canonical expression.
            Returns a (leader, follower) tuple.

            Heuristics:
            1. Backed expression or constants preferred over unbacked expr
            2. Simpler sub-expr when one contains the other
            3. Higher frequency across equalities from deferred runtime assertions
            4. Size of the set
            5. Fallback to sympy.Basic.compare
            """

            def _choose(x: int, y: int) -> bool:
                lhs, rhs = self.expressions[x], self.expressions[y]

                any_unbacked_lhs = has_free_unbacked_symbols(lhs)
                any_unbacked_rhs = has_free_unbacked_symbols(rhs)
                if any_unbacked_lhs != any_unbacked_rhs:
                    return bool(any_unbacked_rhs)

                if lhs.has(rhs):
                    return False
                elif rhs.has(lhs):
                    return True

                degrees_lhs = len(self.eq_graph[lhs])
                degrees_rhs = len(self.eq_graph[rhs])
                if degrees_lhs != degrees_rhs:
                    return degrees_lhs > degrees_rhs

                if self.size[x] != self.size[y]:
                    return self.size[x] > self.size[y]

                return lhs.compare(rhs) == -1

            if _choose(a, b):
                return a, b
            return b, a

    # Build an undirected graph using ShapeEnv's deferred runtime assertions.
    shape_env._equality_graph = defaultdict(OrderedSet)
    for assertions in shape_env.deferred_runtime_asserts.values():
        for assertion in assertions:
            if not isinstance(assertion.expr, sympy.Equality):
                continue
            lhs = sympy.sympify(assertion.expr.lhs)
            rhs = sympy.sympify(assertion.expr.rhs)
            shape_env._equality_graph[lhs].add(rhs)
            shape_env._equality_graph[rhs].add(lhs)

    uf = CanonicalExprFinder(shape_env._equality_graph)

    shape_env._unbacked_replacements = {}
    for expr in shape_env._equality_graph:
        canonical_expr = uf.find_expr(expr)
        if expr != canonical_expr:
            shape_env._unbacked_replacements[expr] = canonical_expr

    return shape_env._unbacked_replacements


@lru_cache(maxsize=1024)
def _sub_unbacked_exprs(shape_env: ShapeEnv, expr: sympy.Expr) -> sympy.Expr:
    """Substitute unbacked expressions with canonical equivalents.
    Used by optimization_hint to maximize consistency when hinting unbacked symbols.
    """
    replacements = _get_unbacked_replacements(shape_env)
    symbol_replacements = {
        k: v for k, v in replacements.items() if isinstance(k, sympy.Symbol)
    }
    expr_replacements = {
        k: v for k, v in replacements.items() if not isinstance(k, sympy.Symbol)
    }

    # consider making this threshold configurable
    sub_cnt_limit = 30
    sub_cnt = 0
    while sub_cnt < sub_cnt_limit:
        if expr_replacements:
            free_symbol_count = len(expr.free_symbols)
            if (
                free_symbol_count * len(expr_replacements)
                <= SYMPY_SUBS_MAX_SYMBOL_REPLACEMENT_PRODUCT
            ):
                new_expr = expr.subs(replacements)
            else:
                new_expr = _sympy_subs(expr, symbol_replacements)
                log.debug(
                    "Skipping %d expensive unbacked expression substitutions "
                    "over %d free symbols in optimization_hint",
                    len(expr_replacements),
                    free_symbol_count,
                )
        else:
            new_expr = _sympy_subs(expr, symbol_replacements)
        if new_expr == expr:
            break
        if len(new_expr.free_symbols) <= SYMPY_FACTOR_MAX_FREE_SYMBOLS:
            expr = sympy.factor(new_expr)
        else:
            expr = new_expr
        sub_cnt += 1
    else:
        log.warning("Substitution limit (%d) reached w/ %s", sub_cnt_limit, expr)

    expr = _sympy_subs(expr, shape_env.backed_var_to_val)
    expr = _sympy_subs(expr, shape_env.var_to_hint_override)
    return expr


def _hint_bounds_from_runtime_asserts(
    shape_env: ShapeEnv, s: sympy.Symbol
) -> tuple[int | None, int | None]:
    """Concrete ``(lower, upper)`` bounds for an unbacked symbol derived from the
    runtime asserts, with backed symbols replaced by their hint values.

    Relational invariants such as ``u0 <= s44`` live only in
    ``deferred_runtime_asserts`` -- ``var_to_range`` keeps the loose
    ``[0, unbacked_symint_fallback]``. Feeding these bounds into the size hint keeps
    it consistent with the graph: e.g. a reducing slice ``x[u0:]`` sized ``s44 - u0``
    cannot hint negative, because ``u0`` is capped at ``s44``'s hint instead of the
    generic fallback. Only linear single-symbol bounds (``c*s + k <= 0``) with
    concrete integer coefficients are used, so this never tightens past what the
    asserts prove; anything else is skipped.
    """
    lower: int | None = None
    upper: int | None = None
    backed = shape_env.backed_var_to_val
    # Read only the asserts keyed to this symbol (bounds on an unbacked symbol
    # only ever live there); guards constrain backed symbols and would just be a
    # wasted scan.
    try:
        for ra in shape_env.deferred_runtime_asserts.get(s, ()):
            ax = ra.expr
            if not isinstance(ax, sympy.Basic):
                continue
            # Substitute backed hints, so e.g. ``u0 <= s44`` becomes ``u0 <= 16``.
            ax = ax.xreplace(backed)
            if s not in ax.free_symbols:
                continue
            # Normalize the relation to ``g < 0`` (strict) or ``g <= 0``.
            if isinstance(ax, sympy.LessThan):
                g, strict = ax.lhs - ax.rhs, False
            elif isinstance(ax, sympy.StrictLessThan):
                g, strict = ax.lhs - ax.rhs, True
            elif isinstance(ax, sympy.GreaterThan):
                g, strict = ax.rhs - ax.lhs, False
            elif isinstance(ax, sympy.StrictGreaterThan):
                g, strict = ax.rhs - ax.lhs, True
            else:
                continue
            # Require g == c*s + k with concrete integer c, k (no other free syms).
            c = g.coeff(s, 1)
            k = g.coeff(s, 0)
            if g - (c * s + k) != 0 or not (c.is_Integer and k.is_Integer):
                continue
            c, k = int(c), int(k)
            if c == 0:
                continue
            thr = sympy.Rational(-k, c)  # c*s + k (<|<=) 0  ->  s (<|<=) thr, c>0
            if c > 0:
                bnd = int(sympy.ceiling(thr) - 1) if strict else int(sympy.floor(thr))
                upper = bnd if upper is None else min(upper, bnd)
            else:  # dividing by c<0 flips the inequality: s (>|>=) thr
                bnd = int(sympy.floor(thr) + 1) if strict else int(sympy.ceiling(thr))
                lower = bnd if lower is None else max(lower, bnd)
    except Exception:
        pass
    return lower, upper


def _optimization_hint_base(
    shape_env: ShapeEnv,
    expr: sympy.Expr | int,
    precomputed_replacements: dict[sympy.Expr, sympy.Symbol],
    fallback: int | None = None,
) -> int:
    """
    Return a concrete integer hint for an expression using heuristics.

    This function should be used for non-guarding based optimizations.
    It will hint unbacked symbols using user provided optimization hints.
    If not provided, fallback will be used along with some heuristics
    that try to maximize consistency with the shape environment.

    Args:
        shape_env: The ShapeEnv instance.
        expr: A sympy expression or integer to evaluate.
        precomputed_replacements: Precomputed replacements for PRECOMPUTED_SIZE symbols.
        fallback: Fallback value for unbacked symbols. If None, reads from config.

    Returns:
        A concrete integer hint for the expression.
    """
    from torch.fx.experimental.symbolic_shapes import (
        has_free_unbacked_symbols,
        symbol_is_type,
        SymT,
    )

    # Read config at call time to respect runtime patches (e.g., in tests)
    if fallback is None:
        from torch._inductor.config import unbacked_symint_fallback

        fallback = unbacked_symint_fallback

    # to have expanded (Identity free) expr stored in original
    if isinstance(expr, sympy.Expr):
        expr = expr.expand(identity=True)

    original = expr
    # sympy.expand() doesn't work with boolean expressions like Or/And
    if isinstance(expr, sympy.Expr):
        expr = expr.xreplace(shape_env.replacements)
    else:
        expr = sympy.sympify(expr).xreplace(shape_env.replacements)

    result = _maybe_realize_expr(expr, fallback)
    if result is not None:
        return result

    if isinstance(expr, sympy.Expr):
        expr = expr.expand(identity=True)

    # Replace backed symbols with their hints, leaving unbacked symbols alone.
    result = _maybe_realize_expr(expr, None)
    if result is not None:
        return result

    if not isinstance(expr, sympy.Expr):
        raise RuntimeError("isinstance(expr, sympy.Expr)", expr)

    if any(symbol_is_type(s, SymT.PRECOMPUTED_SIZE) for s in expr.free_symbols):  # type: ignore[attr-defined]
        expr = _sympy_subs(expr, precomputed_replacements)

    expr = _sympy_subs(expr, shape_env.backed_var_to_val)
    if isinstance(expr, sympy.Expr):
        expr = expr.expand(identity=True)

    result = _maybe_realize_expr(expr, fallback)
    if result is not None:
        return result

    expr = _sympy_subs(expr, shape_env.var_to_hint_override)

    result = _maybe_realize_expr(expr, fallback)
    if result is not None:
        return result

    # If unbacked symbols remain, try to substitute them using heuristics
    # that maximize consistency with the shape environment.
    if has_free_unbacked_symbols(expr):
        # Make sure to substitute with the factored version
        # e.g. 10*(s0 + u0) instead of 10*s0 + 10*u0
        if (
            isinstance(original, sympy.Expr)
            and len(original.free_symbols) <= SYMPY_FACTOR_MAX_FREE_SYMBOLS
        ):
            original = sympy.factor(original)
        expr = _sub_unbacked_exprs(shape_env, original)

    # For multiple expressions that depend on an unbacked symint,
    # we want to compute them consistently for a size hint we have chosen.
    # So, recursively compute expressions via size hints of contained symbols.
    # For example: u1 * u2 - 10 ==> fallback * fallback - 10

    if not isinstance(expr, sympy.Expr):
        raise RuntimeError(f"Expected sympy Expr, got {type(expr)}: {expr}")
    free_symbols = expr.free_symbols

    # Constrain fallback per-symbol based on var_to_range bounds, then tighten
    # further with invariants that live only in the runtime asserts (e.g.
    # u0 <= s44). Without the latter, an unbacked symbol keeps its generic
    # fallback even when the graph proves it is bounded by a backed dim, so a
    # size expression like s44 - u0 can hint negative and overflow downstream
    # allocations (autotuning example tensors, buffer sizing).
    size_dict = {}
    for s in free_symbols:
        lower: int | None = None
        upper: int | None = None
        vr = shape_env.var_to_range.get(s, None)
        if vr is not None:
            if isinstance(vr.lower, (int, sympy.Integer)):
                lower = int(vr.lower)
            if isinstance(vr.upper, (int, sympy.Integer)):
                upper = int(vr.upper)
        ra_lower, ra_upper = _hint_bounds_from_runtime_asserts(shape_env, s)
        if ra_lower is not None:
            lower = ra_lower if lower is None else max(lower, ra_lower)
        if ra_upper is not None:
            upper = ra_upper if upper is None else min(upper, ra_upper)
        sym_fallback = fallback
        if lower is not None:
            sym_fallback = max(sym_fallback, lower)
        if upper is not None:
            sym_fallback = min(sym_fallback, upper)
        size_dict[s] = sym_fallback

    try:
        final_result = _sympy_subs(expr, size_dict)
    except ZeroDivisionError:
        # Expressions like ModularIndexing(x, u1, 4) crash during substitution
        # when u1 is replaced with 0, because sympy eagerly evaluates
        # (x // 0) % 4.  This can happen when an unbacked symbol with
        # var_to_range lower=0 is used as a divisor (e.g. from
        # _dynamic_reshape_indexer) and the fallback also maps to 0.
        # Return fallback in that case.
        return fallback if fallback is not None else 0

    final_result = _maybe_realize_expr(final_result, fallback)
    if final_result is None:
        raise RuntimeError(f"Failed to realize expression to int: {expr}")

    return final_result
<|endoftext|>import inspect
from typing import Any

import torch
import torch.fx
from torch._jit_internal import boolean_dispatched
from torch.fx import Transformer
from torch.fx.graph_module import GraphModule
from torch.fx.node import Argument, Target
from torch.fx.operator_schemas import _torchscript_type_to_python_type
from torch.fx.proxy import Proxy


__all__ = ["AnnotateTypesWithSchema"]


class AnnotateTypesWithSchema(Transformer):
    """
    Use Python function signatures to annotate types for `Nodes` within an FX graph.
    This pulls out Python function signatures for:

        1. Standard `torch.nn` Module calls
        2. `torch.nn.functional` calls
        3. Attribute fetches via `get_attr`

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = AnnotateTypesWithSchema(traced).transform()

    """

    def __init__(
        self,
        module: GraphModule,
        annotate_functionals: bool = True,
        annotate_modules: bool = True,
        annotate_get_attrs: bool = True,
    ) -> None:
        super().__init__(module)
        self.annotate_functionals = annotate_functionals
        self.annotate_modules = annotate_modules
        self.annotate_get_attrs = annotate_get_attrs

    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        python_ret_type = None
        if self.annotate_functionals and target.__module__ == "torch.nn.functional":
            target_for_analysis = target
            if target in boolean_dispatched:
                # HACK: `boolean_dispatch` as used in `torch.nn.functional` makes it so that we have
                # a 2-way dispatch based on a boolean value. Here we check that the `true` and `false`
                # branches of the dispatch have exactly the same signature. If they do, use the `true`
                # branch signature for analysis. Otherwise, leave this un-normalized
                if isinstance(target, str):
                    raise AssertionError("target should not be a string here")
                dispatched = boolean_dispatched[target]
                if_true, if_false = dispatched["if_true"], dispatched["if_false"]
                # TODO: can we emit the union of these? What are the implications on TorchScript
                # compilation?
                if (
                    inspect.signature(if_true).return_annotation
                    != inspect.signature(if_false).return_annotation
                ):
                    return super().call_function(target, args, kwargs)
                target_for_analysis = if_true

            python_ret_type = self._extract_python_return_type(target_for_analysis)

        return_proxy = super().call_function(target, args, kwargs)
        return_proxy.node.type = (
            return_proxy.node.type if return_proxy.node.type else python_ret_type
        )
        return return_proxy

    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        python_ret_type = None
        if not isinstance(target, str):
            raise AssertionError(f"Expected str target, got {type(target)}")
        submod = self.fetch_attr(target)
        if self.annotate_modules and hasattr(submod.__class__, "__name__"):
            classname = submod.__class__.__name__
            if getattr(torch.nn, classname, None) == submod.__class__:
                python_ret_type = self._extract_python_return_type(submod.forward)
        return_proxy = super().call_module(target, args, kwargs)
        return_proxy.node.type = (
            return_proxy.node.type if return_proxy.node.type else python_ret_type
        )
        return return_proxy

    def get_attr(
        self,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Any],
    ) -> Proxy:
        attr_proxy = super().get_attr(target, args, kwargs)

        if self.annotate_get_attrs:
            module_itr = self.module
            if not isinstance(target, str):
                raise AssertionError(f"Expected str target, got {type(target)}")
            atoms = target.split(".")
            for i, atom in enumerate(atoms):
                if not hasattr(module_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonextent target {'.'.join(atoms[:i])}!"
                    )
                module_itr = getattr(module_itr, atom)

            maybe_inferred_ts_type = torch._C._jit_try_infer_type(module_itr)
            if maybe_inferred_ts_type.success():
                python_type = _torchscript_type_to_python_type(
                    maybe_inferred_ts_type.type()
                )
                attr_proxy.node.type = (
                    python_type if not attr_proxy.node.type else attr_proxy.node.type
                )

        return attr_proxy

    def _extract_python_return_type(self, target: Target) -> Any | None:
        """
        Given a Python call target, try to extract the Python return annotation
        if it is available, otherwise return None

        Args:

            target (Callable): Python callable to get return annotation for

        Returns:

            Optional[Any]: Return annotation from the `target`, or None if it was
                not available.
        """
        if not callable(target):
            raise AssertionError(f"Expected callable target, got {type(target)}")
        try:
            sig = inspect.signature(target)
        except (ValueError, TypeError):
            return None

        return (
            sig.return_annotation
            if sig.return_annotation is not inspect.Signature.empty
            else None
        )
<|endoftext|># Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextvars
import functools
import inspect
import logging
import operator
import threading
import typing
import typing_extensions
from collections import defaultdict, OrderedDict
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import _GeneratorContextManager, contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
from typing import (
    Any,
    cast,
    Concatenate,
    Literal,
    overload,
    Protocol,
    TYPE_CHECKING,
    TypeAlias,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec, Self, TypeVarTuple, Unpack
from weakref import WeakKeyDictionary

import torch
import torch._ops
import torch.fx as fx
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch import SymBool, SymInt, Tensor
from torch._custom_class_base import CustomClassBase
from torch._dispatch.python import enable_python_dispatcher
from torch._library.fake_class_registry import FakeScriptObject
from torch._library.opaque_object import (
    get_reconstruct_fn,
    is_custom_class_obj,
    is_opaque_constant_type,
    is_opaque_symbolic_type,
    should_hoist,
)
from torch._logging import trace_structured
from torch._ops import HigherOrderOperator, OpOverload
from torch._subclasses.fake_impls import fast_detach
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    get_plain_tensors,
    is_fake,
    is_fake_tensor,
    maybe_get_fake_mode,
    unset_fake_temporarily,
)
from torch._subclasses.functional_tensor import FunctionalTensor
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx import GraphModule, Proxy, Tracer
from torch.fx.experimental.dynamic_spec import (
    _coerce_to_shapes_spec,
    IntVar,
    LeafSpec,
    ParamsSpec,
    ShapesSpec,
    TensorSpec,
)
from torch.fx.graph_module import _assign_attr
from torch.fx.node import (
    _side_effectful_need_to_be_preserved_pre_dispatch,
    Argument,
    Target,
)
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.nn import Module
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import (
    _disable_infra_mode,
    _push_mode,
    _unset_infra_mode,
    autograd_would_have_decomposed,
    TorchDispatchMode,
    TraceableWrapperSubclass,
)
from torch.utils._stats import count
from torch.utils._thunk import Thunk
from torch.utils.weak import _WeakHashRef, WeakIdKeyDictionary, WeakTensorKeyDictionary

from ._backward_state import BackwardState
from .sym_node import SymNode


if TYPE_CHECKING:
    import types
    from collections.abc import MutableMapping

    import sympy

    from torch._guards import Source
    from torch._higher_order_ops.utils import FunctionalizeCtxWrapper
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.types import BoolLikeType, FloatLikeType, IntLikeType

__all__ = [
    "PythonKeyTracer",
    "dispatch_trace",
    "make_fx",
    "DecompositionInterpreter",
    "selective_decompose",
    "py_sym_types",
    "get_innermost_proxy_mode",
    "get_proxy_mode",
    "handle_sym_dispatch",
    "maybe_enable_thunkify",
    "maybe_disable_thunkify",
]

_ProxyTracer = Union["PythonKeyTracer", "_GraphAppendingTracerEx"]
_TracingMode: TypeAlias = Literal["real", "fake", "symbolic"]

_AnyScriptObject = (torch.ScriptObject, FakeScriptObject)
_AnyScriptObjectType = torch.ScriptObject | FakeScriptObject

aten = torch.ops.aten
prim = torch.ops.prim

log = logging.getLogger(__name__)
not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")

CONSTANT_NUMEL_LIMIT = 1

T = TypeVar("T")
U = TypeVar("U")
_P = ParamSpec("_P")
R = TypeVar("R")
_Ts = TypeVarTuple("_Ts")

# We currently convert all SymInt to proxies before we use them.
# This could plausibly be handled at the Dynamo level.
pytree.register_pytree_node(
    torch.Size,
    lambda xs: (list(xs), None),
    lambda xs, _: tuple(xs),
    # pyrefly: ignore [bad-argument-type]
    flatten_with_keys_fn=lambda xs: (
        [(pytree.SequenceKey(i), x) for i, x in enumerate(xs)],
        None,
    ),
    serialized_type_name="torch.Size",
)
# Ideally unflattening should not lose info, but we unflatten
# torch.Size to tuple (see above). This is necessary because the
# torch.Size constructor only accepts ints whereas our infra often
# transforms them to non-ints, e.g. symint proxies. Anyway, losing
# such info can cause pytree mapping or spec matching to fail, so
# work around this problem using the following dict as needed.
_pytree_subclasses_that_lose_info = {torch.Size: tuple}


def fake_signature(fn: Callable[_P, R], nargs: int) -> Callable[_P, R]:
    """FX gets confused by varargs, de-confuse it"""
    argnames = ",".join(f"arg{i}" for i in range(nargs))
    return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})


@contextmanager
def decompose(
    decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None,
) -> Generator[Mapping[OpOverload, Callable[..., Any]], None, None]:
    mode = get_proxy_mode()
    if mode is None:
        raise AssertionError("Expected proxy mode to be set")
    with mode.enable_decompositions(decomposition_table) as table:
        yield table


# ensure we cannot collide with other properties
proxy_slot = object()


class _NoDefault:
    pass


no_default = _NoDefault()

from torch.types import py_sym_types, PySymType


class _HasMeta(Protocol):
    meta: dict[str, PySymType]


def is_sym_node(node: _HasMeta) -> bool:
    if not hasattr(node, "meta"):
        raise AssertionError("All nodes traced with proxy_tensor should have meta")
    return "val" in node.meta and isinstance(node.meta["val"], py_sym_types)


class _DisableUpdateTensorTracker(threading.local):
    value: bool = False


_disable_update_tensor_tracker_tls = _DisableUpdateTensorTracker()


_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT: dict[int, torch.fx.Node] = {}


def _is_proxy_tensor_update_tensor_tracker_disabled() -> bool:
    """
    Returns current state of disabling update tensor tracker.
    """
    return _disable_update_tensor_tracker_tls.value


@contextmanager
def _proxy_tensor_disable_update_tensor_tracker() -> Generator[None, None, None]:
    """
    NOTE "Do not clobber inplace ops"
    By default tensor_tracker is updated every time.
    This leads to chaining every operation by the FakeTensor.
    For example for mutable ops if we have several consecutive mutable operations:

    def f(x, y, z):
        x.copy_(y)
        x.copy_(z)
        return x

    Default graph result:
    def f_graph(x, y, z)
        x_1 = x.copy_(y)
        x_2 = x_1.copy_(z)
        return x_2

    This chaining simplifies the fx passes and helps to prevent the reordering.
    But in some cases, we want those nodes to be disconnected.
    E.g. in case of splitting joint graph into forward and backward.
    If first inplace op happened in forward, second in backward,
    we want them after split to be properly placed.

    Enabling this context manager for copy_ will result in:
    def f_graph_2(x, y, z):
        x_1 = x.copy_(y)
        x_2 = x.copy_(z)
        return x

    Results of copy_ x1 and x2 will have empty users in the graph.
    The reason why this behavior is not enabled for all inplace ops is that
    some fx passes (e.g. fx quantization) rely on chaining inplace ops like add_
    in their fusions passes.
    We could revisit enabling this logic for all inplace ops in future.
    """
    orig_value = _disable_update_tensor_tracker_tls.value
    _disable_update_tensor_tracker_tls.value = True
    try:
        yield
    finally:
        _disable_update_tensor_tracker_tls.value = orig_value


@overload
def set_proxy_slot(obj: Tensor, tracer: _ProxyTracer, proxy: _ProxyTensor) -> None: ...


@overload
def set_proxy_slot(
    obj: _AnyScriptObjectType | CustomClassBase, tracer: _ProxyTracer, proxy: Proxy
) -> None: ...


@overload
def set_proxy_slot(
    obj: PySymType, tracer: _ProxyTracer, proxy: _PySymProxyType
) -> None: ...


def set_proxy_slot(
    obj: PySymType | _AnyScriptObjectType | Tensor | CustomClassBase,
    tracer: _ProxyTracer,
    proxy: object,
) -> None:
    log.debug("set_proxy_slot %s (%s) %s", obj, id(obj), proxy)
    if isinstance(obj, Tensor):
        # We DO want to clobber proxies whenever we run an inplace operation
        # on a tensor, and it affects the metadata on the proxy.
        if not isinstance(proxy, _ProxyTensor):
            raise AssertionError(f"Expected _ProxyTensor, got {type(proxy)}")
        # see NOTE [Do not clobber inplace ops]
        if (
            obj not in tracer.tensor_tracker
            or not _is_proxy_tensor_update_tensor_tracker_disabled()
        ):
            tracer.tensor_tracker[obj] = proxy
    elif isinstance(obj, (_AnyScriptObject)) or is_custom_class_obj(obj):
        if not isinstance(proxy, Proxy):
            raise AssertionError(f"Expected Proxy, got {type(proxy)}")
        # ScriptObject (actual C++ torchbind) uses _WeakHashRef-keyed tracker
        # because the same C++ IValue can produce different Python wrappers.
        # FakeScriptObject/CustomClassBase uses WeakIdRef-keyed tracker because
        # value-equal objects (e.g. primal vs tangent) must be tracked separately.
        if isinstance(obj, torch.ScriptObject):
            tracer.script_object_tracker[obj] = proxy
        else:
            # NB: Never clobber a pre-existing proxy for the same
            # underlying real object.  Multiple FakeScriptObject wrappers
            # can share the same real_obj (e.g. primal vs tangent
            # placeholders during joint graph tracing).  We always keep the
            # first proxy registered, with the same rationale as the
            # symnode_tracker first-one-wins policy below: primals are
            # registered first, so this avoids spurious tangent dependencies
            # in forward outputs (which would break the partitioner).
            real_obj = None
            if isinstance(obj, FakeScriptObject):
                try:
                    real_obj = object.__getattribute__(obj, "real_obj")
                except AttributeError:
                    pass

            if real_obj is not None:
                existing = tracer._opaque_real_obj_proxy.get(id(real_obj))
                if existing is not None:
                    tracer.opaque_tracker[obj] = existing
                else:
                    tracer.opaque_tracker[obj] = proxy
                    tracer._opaque_real_obj_proxy[id(real_obj)] = proxy
            else:
                tracer.opaque_tracker[obj] = proxy
    else:
        # NB: Never clobber pre-existing proxy.  Although the proxies
        # are in principle equivalent, when we do graph partitioning
        # we need there not to be spurious dependencies on tangent inputs.
        # This works because primals get their SymInts set first, and
        # THEN later we allocate tangent inputs.  Make sure if a SymInt
        # is derivable from a primal that we use that.
        if not isinstance(obj, py_sym_types):
            raise AssertionError(f"Expected py_sym_types, got {type(obj)}")
        if obj not in tracer.symnode_tracker:
            proxy = typing.cast(_PySymProxyType, proxy)
            tracer.symnode_tracker[obj] = proxy

            # WAR: python test/dynamo/test_subclasses.py
            # TestNestedTensor.test_basic_autograd
            #
            # AOTAutograd doesn't pass the "outer sizes" as an actual argument
            # to make_fx, but it is made use of internally in AOTAutograd's
            # call to tensor unflatten.  Because the outer sizes isn't passed
            # as an argument, it is therefore untracked.  However, it turns
            # out you luck out, because *Dynamo* will manually add the outer
            # sizes as an argument so you can fix up the proxy'ness.
            #
            # This is probably fixed in
            # https://github.com/pytorch/pytorch/pull/125941/
            import sympy

            if isinstance(obj.node.expr, sympy.Symbol):
                tracer.sympy_expr_tracker[obj.node.expr] = _SympyExprTrackerValue(
                    proxy, obj
                )


def has_proxy_slot(obj: Tensor, tracer: _ProxyTracer) -> bool:
    if not isinstance(obj, (Tensor, SymNode)):
        raise AssertionError(f"Expected Tensor or SymNode, got {type(obj)}")

    return bool(get_proxy_slot(obj, tracer, False, lambda _: True))


_PySymProxyType: TypeAlias = Thunk[Proxy]
_OpaqueObjectProxyType: TypeAlias = Proxy


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
) -> _ProxyTensor: ...


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
    default: U,
) -> _ProxyTensor | U: ...


@overload
def get_proxy_slot(
    obj: Tensor,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_ProxyTensor], R],
) -> R | U: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
) -> Proxy: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
    default: U,
) -> Proxy | U: ...


@overload
def get_proxy_slot(
    obj: _AnyScriptObjectType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[Proxy], R],
) -> R | U: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
) -> _PySymProxyType: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
    default: T,
) -> T | _PySymProxyType: ...


@overload
def get_proxy_slot(
    obj: CustomClassBase,
    tracer: _ProxyTracer,
    default: T,
) -> T | _OpaqueObjectProxyType: ...


@overload
def get_proxy_slot(
    obj: PySymType,
    tracer: _ProxyTracer,
    default: U,
    transform: Callable[[_PySymProxyType], R],
) -> R | U: ...


# the default argument is what to return if the slot is not set.
# the transform argument is handy if you need to extract a subfield from
# the successfully looked up result (but NOT the default.)
def get_proxy_slot(
    obj: Tensor | _AnyScriptObjectType | PySymType | CustomClassBase,
    tracer: _ProxyTracer,
    default: object = no_default,
    transform: Callable[..., Any] = lambda x: x,
) -> object:
    tracker: Any
    if isinstance(obj, Tensor):
        tracker = tracer.tensor_tracker
    elif isinstance(obj, _AnyScriptObject) or is_custom_class_obj(obj):
        if isinstance(obj, torch.ScriptObject):
            tracker = tracer.script_object_tracker
        else:
            tracker = tracer.opaque_tracker
    else:
        if not isinstance(obj, py_sym_types):
            raise AssertionError(f"Expected py_sym_types, got {type(obj)}")
        tracker = tracer.symnode_tracker

    # pyrefly: ignore [index-error]
    # pyrefly: ignore [no-matching-overload, bad-argument-type]
    value = tracker.get(obj)

    if value is None and isinstance(obj, py_sym_types):
        if obj.node.is_symbolic():
            # Last ditch - we found a SymInt (SymBool, etc) we don't know
            # about.
            if (tmp := tracer.sympy_expr_tracker.get(obj.node.expr)) is not None:
                value = tmp.proxy

            else:
                # Attempt to build it from first principles.
                _build_proxy_for_sym_expr(tracer, obj.node.expr, obj)
                # pyrefly: ignore [bad-argument-type, no-matching-overload]
                value = tracker.get(obj)

    if value is None and isinstance(obj, FakeScriptObject):
        # A new FakeScriptObject wrapping the same real_obj may have been
        # created (e.g. output flattening in unwrap_tensor_subclasses calls
        # maybe_to_fake_obj which always mints a fresh wrapper).  Fall back
        # to the real-object dedup map that set_proxy_slot maintains.
        value = tracer._opaque_real_obj_proxy.get(id(obj.real_obj))

    if value is None:
        # We don't know this value - return the default.
        if isinstance(default, _NoDefault):
            raise RuntimeError(
                f"{obj} ({type(obj)}, {id(obj)})is not tracked with proxy for {tracer}"
            )
        return default

    res = transform(value)
    return res


# Recursively traverses traceable wrapper subclasses,
# returnining an (unordered) list of Proxy objects that are tracked
# for all inner tensors, given the current extant proxy mode.
# Returns an empty list if no proxy mode is active.
def _get_proxies(t: torch.Tensor | TraceableWrapperSubclass) -> list[Proxy]:
    proxies = []
    mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
    if mode is None:
        return proxies
    if not isinstance(mode, ProxyTorchDispatchMode):
        raise AssertionError(f"Expected ProxyTorchDispatchMode, got {type(mode)}")
    tracer = mode.tracer
    for t_inner in get_plain_tensors(t, out=[]):
        if isinstance(t_inner, FunctionalTensor):
            t_inner = torch._from_functional_tensor(t_inner.elem)
        if not isinstance(t_inner, torch.Tensor):
            continue
        proxy_tensor = get_proxy_slot(t_inner, tracer)
        if proxy_tensor is not None:
            proxies.append(proxy_tensor.proxy)
    return proxies


# sympy Max/Min are n-ary (they flatten, e.g. Max(a, Max(b, c)) -> Max(a, b, c)),
# but torch.sym_max/sym_min are binary. Reduce so _build_proxy_for_sym_expr can
# rebuild flattened expressions as nested binary ops. These are module-level
# (not lambdas) so they have a qualified name and survive FX codegen/pickling
# when used as a graph node target.
def _nary_sym_max(*args: Any) -> Any:
    return functools.reduce(torch.sym_max, args)


def _nary_sym_min(*args: Any) -> Any:
    return functools.reduce(torch.sym_min, args)


@functools.cache
def _sympy_handlers() -> dict[type[sympy.Expr], Callable[..., Any]]:
    """
    Returns a dict mapping sympy types to Python callables
    (e.g. ``sympy.Mul`` -> ``operator.mul``, ``sympy.Add`` -> ``torch.sym_sum``).
    """
    import sympy

    import torch.utils._sympy.interp

    handlers = {}
    for k, v in torch.utils._sympy.interp.handlers().items():
        op = getattr(operator, v, None)
        if op is not None:
            handlers[k] = op
        # sympy Max/Min have no operator.* equivalent. Map them to n-ary-safe
        # wrappers over torch.sym_max/sym_min. Without this,
        # _build_proxy_for_sym_expr cannot rebuild expressions like Max(1, u2)
        # that escape a disable_proxy_modes_tracing region (e.g. DTensor shard
        # propagation size inference).
        elif v == "maximum":
            handlers[k] = _nary_sym_max
        elif v == "minimum":
            handlers[k] = _nary_sym_min
        # sympy.Pow / PowByNatural map to the interp name "pow_by_natural",
        # which has no operator.* equivalent. sympy canonicalizes x * x into
        # Pow(x, 2), so without this _build_proxy_for_sym_expr cannot rebuild
        # any repeated-symbol product (e.g. a reduction numel s0 * s1**2 when
        # two equal dims duck-share a symbol).
        elif v == "pow_by_natural":
            handlers[k] = operator.pow

    # sympy.Add is n-ary (e.g. Add(a, b, c)) but operator.add is binary.
    # torch.sym_sum handles n-ary integer addition and accepts both
    # sym_sum([a, b, c]) and sym_sum(a, b, c).
    handlers[sympy.Add] = torch.sym_sum
    return handlers


def _build_proxy_for_sym_expr(
    tracer: _ProxyTracer, expr: sympy.Expr, out: PySymType | None = None
) -> IntLikeType | FloatLikeType | BoolLikeType | None:
    """
    Decompose `expr` and look for the pieces as inputs. If `out` is provided
    then that will be the resulting SymNode (and `out.expr` must be the same as
    `expr`).

    This function is used when the ProxyTorchDispatchMode sees a SymNode
    that it hasn't seen before to try to associate it with traced inputs.

    How can this happen?

    First thing to remember is that although sympy.Exprs are interned (so
    `sympy.Expr("s3*s4")` will always have the same `id` and will always compare
    equal) SymNode does not (so doing `SymNode("s3")*SymNode("s4")` twice in a
    row will give two unique SymNodes).

    - On way for this to happen is if we turn off tracing to compute an
      intermediate value and then USE that value with tracing turned on - for
      example if we turn off tracing to do some FakeTensor propagation to
      compute a size (dtensor does this) but then turn tracing back on and use
      that computed size.

    - Another way is if we compute a size in one graph and stash it somewhere
      hidden (such as in some meta-data) and later use it in a different graph
      (dtensor does this too). Since the size was computed in the first graph
      and it's not an official input to the second graph it's not tracked
      properly. This is often going to show up as it usually works in fullgraph
      but a graph break causes a failure.

    To handle this we decompose the sympy.Expr and look for the pieces as
    inputs. But there are problems with this approach:

    - We lose operation provenance: We end up figuring out where to get the
      inputs - but those may not actually be correct. If we have "s1" coming in
      from both tensor1 and tensor2 and we pick the wrong one we could end up
      keeping a tensor alive longer than intended.

    - There's no guarantee that those values are inputs to the graph: If we have
      "s1*s2" computed in a graph #1 and used in graph #2 there's no guarantee
      that the input that holds "s1" is actually an input on graph #2.

    - The decomposition isn't guaranteed to be the same: Sympy can "simplify"
      expressions so it's possible that our inputs are "s1*s2" and "s3" but we
      decompose it into "s1" and "s2*s3" - which wouldn't be found.

    Other ways we could handle this:

    - Don't: Just require that all inputs are tracked properly. This is the
      "correct" solution but harder because you need to track down each
      potential problem one by one and fix them. And when it fails it's a lot of
      work to figure out both why it's failing and the right way to fix it. This
      is complicated by the fact that a stashed value could be incorrect but
      work fine until we happen to get an graph break in the wrong place - so it
      may be a while before the bug is found. (Maybe we need a "dynamo abuse
      mode" where we run tests with as many graph breaks inserted as possible?)

    - Track SymNode ops separately from proxy tracing: Right now SymNode
      operations are tracked as part of the proxy tracing - so when we disable
      proxy tracing we also disable SymNode tracing. But we don't have to do
      that - we could instead always have SymNodes track where they came from
      and just use that when needed. This solves the problem of tracing being
      temporarily turned off but doesn't help if an input isn't present after a
      graph break.

    - Better decomposition: Right now the decomposition is pretty simple. We do
      have a sat-solver available to us so we could theoretically do a better
      job figuring out a "correct" decomposition. But that still relies on
      having the inputs available at all - which isn't a guarantee.
    """

    if (value := tracer.sympy_expr_tracker.get(expr)) is not None:
        if out:
            raise AssertionError(
                "out should be empty when expr is in sympy_expr_tracker"
            )
        return value.value

    if isinstance(expr, (int, float, bool)):
        return expr
    if expr.is_Integer:
        return int(expr)
    if expr.is_Float:
        return float(expr)

    args = []
    for arg in expr.args:
        if (arg_value := _build_proxy_for_sym_expr(tracer, arg)) is None:
            return None
        args.append(arg_value)

    func: OpOverload | None = _sympy_handlers().get(expr.func)  # type: ignore[assignment]
    if not func:
        return None

    if out is None:
        out = func(*args)
    else:
        _sym_register(tracer, func, tuple(args), out)
    return out


def snapshot_fake(val: Tensor, include_real: bool = False) -> Tensor | None:
    # val.detach() will also eventually call fast_detach(),
    # but this saves us a full trip into __torch_dispatch__
    # (snapshot_fake is called a lot)
    if is_fake_tensor(val):
        return fast_detach(maybe_get_fake_mode(val), val, include_real)
    else:
        return val.detach()


_ExtractValType: TypeAlias = (
    None
    | PySymType
    | CustomClassBase
    | _AnyScriptObjectType
    | BackwardState
    | list["_ExtractValType"]
    | tuple["_ExtractValType", ...]
    | dict[str, "_ExtractValType"]
    | Tensor
    | int
    | float
    | bool
)


def extract_val(val: _ExtractValType, include_real: bool = False) -> _ExtractValType:
    if is_fake(val):
        return snapshot_fake(val, include_real=include_real)
    elif isinstance(val, py_sym_types):
        return val
    elif isinstance(val, (_AnyScriptObject, CustomClassBase)):
        return val
    elif isinstance(val, BackwardState):
        return val
    elif is_custom_class_obj(val):
        return val
    elif isinstance(val, (list, tuple)):
        return val.__class__([extract_val(x) for x in val])
    elif isinstance(val, dict):
        return {k: extract_val(v) for k, v in val.items()}
    elif isinstance(val, Tensor):
        if not val.is_sparse:
            # NB: Kinda hacky, but we should try to get val as the metadata
            # everywhere
            # TODO: This doesn't properly track storages.  A more robust
            # approach would be to maintain a per-trace FakeTensorMode and
            # from_real_tensor to create fake values (don't forget to
            # snapshot_fake)
            from torch._guards import detect_fake_mode

            fake_tensor_mode = detect_fake_mode(val)
            if not fake_tensor_mode:
                fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=True)
            with fake_tensor_mode:
                return torch.empty_strided(
                    val.shape, val.stride(), device=val.device, dtype=val.dtype
                )
        else:
            return None
    elif isinstance(val, (int, float, bool)):
        return val
    elif val is None:
        return None

    typing_extensions.assert_never(val)


@contextmanager
def _enable_thunkify(
    tracer: _ProxyTracer, *, enable: bool = True
) -> Generator[None, None, None]:
    """
    Enable thunkification inside the context manager.  Thunkification prevents
    SymNode computation from directly being traced into an FX graph; instead,
    the compute is only added to the graph if it is actually used.  This helps
    us track SymNode compute when it is computed (since we need /something/
    to put in the tracker) even if it is unlikely to be used.
    """
    old = tracer.enable_thunkify
    tracer.enable_thunkify = enable
    try:
        yield
    finally:
        tracer.enable_thunkify = old


@contextmanager
def maybe_disable_thunkify() -> Generator[None, None, None]:
    """Within a context, disable thunkification.  See :func:`maybe_enable_thunkify`
    for more details.  This is helpful if you have a wrapper function which
    you want to enable thunkification on, but in some segment on the inside (say,
    the original user function), you want to disable thunkification as you know
    it is not needed there.
    """
    proxy_mode = get_proxy_mode()
    if proxy_mode is not None:
        with _enable_thunkify(proxy_mode.tracer, enable=False):
            yield
    else:
        yield


@contextmanager
def maybe_enable_thunkify() -> Generator[None, None, None]:
    """Within this context manager, if you are doing make_fx tracing, we will thunkify
    all SymNode compute and avoid tracing it into the graph unless it is actually needed.
    You should prefer to avoid using this as much as possible, as lazy evaluation of
    SymNode tracing can lead to long chains of thunks which will stack overflow
    if you evaluate them.  However, this is currently sometimes necessary as there
    are buggy parts of PT2 which will fail with "s0 is not tracked with proxy" error
    due to insufficient tracing of SymNode computation.
    """
    proxy_mode = get_proxy_mode()
    if proxy_mode is not None:
        with _enable_thunkify(proxy_mode.tracer):
            yield
    else:
        yield


# Note [invariants for node meta 'val']
# What invariants do we have for the 'val' set on the FX node?  It has accurate
# metadata... but only for metadata that exists "below" all other subsystems
# (most notably autograd, but also vmap, functorch transforms, etc).  This means
# you can get the dtype, shape, stride, storage, but you CANNOT get requires_grad,
# grad_fn, _base (_base actually may be set due to recursive call to
# ADInplaceOrView, but you shouldn't rely on it.)
def set_meta(proxy: Proxy, val: _ExtractValType) -> Proxy:
    proxy.node.meta["val"] = extract_val(
        val, include_real=(proxy.node.op == "placeholder")
    )

    with _enable_thunkify(proxy.tracer):  # type: ignore[arg-type]
        # Best effort tensor_meta setting; prefer using val!
        if is_fake(val):
            proxy.node.meta["tensor_meta"] = _extract_tensor_metadata(val)
        elif isinstance(val, Tensor) and not val.is_sparse:
            proxy.node.meta["tensor_meta"] = _extract_tensor_metadata(val)
    return proxy


def thunkify(
    tracer: _ProxyTracer, f: Callable[_P, R], *args: _P.args, **kwargs: _P.kwargs
) -> Thunk[R]:
    """
    Delays computation of f until it's called again
    Also caches the result
    """
    if tracer.enable_thunkify:
        return Thunk(functools.partial(f, *args, **kwargs))
    else:
        r = f(*args, **kwargs)
        return Thunk(lambda: r)


def track_tensor(
    tensor: Tensor, proxy: Proxy, *, constant: Tensor | None, tracer: _ProxyTracer
) -> None:
    def try_set_proxy_slot(
        outer_s: IntLikeType,
        proxy_callable: Callable[Concatenate[PySymType, _P], Proxy],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> None:
        if not callable(proxy_callable):
            raise AssertionError("proxy_callable must be callable")
        if isinstance(outer_s, SymInt):
            with _enable_thunkify(tracer):
                set_proxy_slot(
                    outer_s,
                    tracer,
                    thunkify(tracer, proxy_callable, outer_s, *args, **kwargs),
                )

    # The basic idea is that we need to associate each tensor/SymInt
    # with a Proxy.  How do we setup this association?  We just store
    # the proxy on the proxy slot of the object, keyed on the tracer
    # (so that if we have multiple tracers at the same time, they
    # don't clobber each other.)
    for i, s in enumerate(tensor.shape):
        try_set_proxy_slot(
            s,
            lambda x, i: set_meta(
                tracer.create_proxy(
                    "call_function", torch.ops.aten.sym_size.int, (proxy, i), {}
                ),
                x,
            ),
            i,
        )

    if not is_sparse_any(tensor):
        for i, s in enumerate(tensor.stride()):
            try_set_proxy_slot(
                s,
                lambda x, i: set_meta(
                    tracer.create_proxy(
                        "call_function", torch.ops.aten.sym_stride.int, (proxy, i), {}
                    ),
                    x,
                ),
                i,
            )

    try_set_proxy_slot(
        tensor.numel(),
        lambda x: set_meta(
            tracer.create_proxy(
                "call_function", torch.ops.aten.sym_numel.default, (proxy,), {}
            ),
            x,
        ),
    )
    if not is_sparse_any(tensor):
        try_set_proxy_slot(
            tensor.storage_offset(),
            lambda x: set_meta(
                tracer.create_proxy(
                    "call_function",
                    torch.ops.aten.sym_storage_offset.default,
                    (proxy,),
                    {},
                ),
                x,
            ),
        )
    set_proxy_slot(tensor, tracer, _ProxyTensor(proxy, constant))


_NestedProxys = Union[  # noqa: UP007
    Proxy, Sequence["_NestedProxys"], Mapping[object, "_NestedProxys"]
]
_NestedTensors = Union[  # noqa: UP007
    Tensor, Sequence["_NestedTensors"], Mapping[object, "_NestedTensors"]
]


def track_tensor_tree(
    inner_res: T,
    proxy_res: _NestedProxys,
    *,
    constant: _NestedTensors | None,
    tracer: _ProxyTracer,
) -> T:
    # NB: We call set_unbacked_bindings only on the *topmost* call to
    # track_tensor_tree, not recursive calls.  This is because there must
    # be only ONE unbacked_binding proxy call, and it should be the one
    # where all of the unbacked SymInts actually first come into existence.
    # If you call this again on the inner proxies for the tuple projections,
    # you will have multiple unbacked_bindings for the same symbol, but
    # they're not going to show up anywhere.
    #
    # I was briefly deceived into setting unbacked bindings recursively when
    # working on https://github.com/pytorch/pytorch/pull/133585 because I
    # observed that some extra unbacked bindings were needed to handle some
    # higher order operator code.  But actually it looks like this was
    # just an unrelated bug that needed to be fixed separately.
    _set_unbacked_bindings(inner_res, proxy_res)

    def wrap_with_proxy(
        e: object, proxy: _NestedProxys, constant: _NestedTensors | None
    ) -> None:
        if isinstance(e, Tensor):
            if not isinstance(proxy, Proxy):
                raise AssertionError(f"Expected Proxy, got {type(proxy)}")
            if not (constant is None or isinstance(constant, Tensor)):
                raise AssertionError(f"Expected None or Tensor, got {type(constant)}")
            track_tensor(e, proxy, tracer=tracer, constant=constant)
            set_meta(proxy, e)
        elif isinstance(e, py_sym_types):
            if not isinstance(proxy, Proxy):
                raise AssertionError(f"Expected Proxy, got {type(proxy)}")
            # NB: eagerly set meta here, so that the numbering is in order
            set_meta(proxy, e)
            set_proxy_slot(e, tracer, thunkify(tracer, lambda: proxy))
        elif isinstance(e, _AnyScriptObject) or is_custom_class_obj(e):
            if not isinstance(proxy, Proxy):
                raise AssertionError(f"Expected Proxy, got {type(proxy)}")
            # Non-hoisted opaque value types should be baked as constants
            # in the graph, not tracked as proxy references. This matches
            # dynamo's behavior where non-hoisted values are not graph inputs.
            if (
                is_opaque_constant_type(type(e))  # pyrefly: ignore[bad-argument-type]
                and not should_hoist(type(e))
            ):
                set_meta(proxy, e)
                return
            set_proxy_slot(e, tracer, proxy)
            set_meta(proxy, e)
        elif isinstance(e, (tuple, list)):
            # example use case: allreduce_ returns ([tensor], work)
            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)

            def get_constant(
                c: _NestedTensors | None, idx: int
            ) -> _NestedTensors | None:
                if c is None:
                    return None
                else:
                    if not isinstance(c, (list, tuple)):
                        raise AssertionError(f"Expected list or tuple, got {type(c)}")
                    # pyrefly: ignore [bad-return]
                    return c[idx]

            for idx, ee in enumerate(e):
                # Use an indexer here - if proxy is a List then it will unwrap
                # it. If it's a Proxy then it will proxy the getelem.
                wrap_with_proxy(ee, proxy[idx], get_constant(constant, idx))  # type: ignore[index]

        elif isinstance(e, dict):
            # example use case: triton_kernel_wrapper takes arguments as kwargs

            # In theory we could support const-prop when proxy-tensor-tracing
            # operators that returns dicts of tensors, but we have no use case
            # for it today (since the only op we currently trace that can
            # return a dict is triton_kernel_wrapper_functional/mutation,
            # which does not participate in const-prop)
            if constant is not None:
                raise AssertionError("Expected constant to be None")

            if isinstance(proxy, fx.Proxy):
                set_meta(proxy, e)

            for key, val in e.items():
                wrap_with_proxy(val, proxy[key], None)  # type: ignore[index]

        elif isinstance(e, BackwardState):
            if not isinstance(proxy, Proxy):
                raise AssertionError(f"Expected Proxy, got {type(proxy)}")
            set_meta(proxy, e)
            e.proxy = proxy
        else:
            # intentionally pass on unsupported primitives
            pass

    wrap_with_proxy(inner_res, proxy_res, constant)

    return inner_res


@dataclass
class _ProxyTensor:
    proxy: Proxy
    constant: Tensor | None


def fetch_sym_proxy(
    tracer: _ProxyTracer,
) -> Callable[[PySymType], bool | int | float | Proxy]:
    def inner(e: PySymType) -> int | bool | float | Proxy:
        n = e.node
        if n.constant is not None:
            return n.constant
        if e.node.expr.is_number:
            if isinstance(e, SymBool):
                return bool(e.node.expr)
            elif isinstance(e, SymInt):
                return int(e.node.expr)
            return float(e.node.expr)
        else:
            if not isinstance(e, py_sym_types):
                raise AssertionError(f"Expected py_sym_types, got {type(e)}")
            # NB: we REQUIRE all symints to be tracked
            return get_proxy_slot(e, tracer).force()

    return inner


@overload
def fetch_object_proxy(tracer: _ProxyTracer, t: Tensor) -> _ProxyTensor | Tensor: ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: _AnyScriptObjectType
) -> Proxy | _AnyScriptObjectType: ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: PySymType
) -> _PySymProxyType | PySymType: ...


@overload
def fetch_object_proxy(
    tracer: _ProxyTracer, t: CustomClassBase
) -> _OpaqueObjectProxyType | PySymType: ...


def fetch_object_proxy(
    tracer: _ProxyTracer,
    t: Tensor | _AnyScriptObjectType | PySymType | CustomClassBase,
) -> object:
    return get_proxy_slot(t, tracer, t)


HANDLED_TYPES = (Tensor, torch.nn.Parameter, FakeTensor)


def _maybe_record_pointwise_barrier(
    func: object, proxy_mode: ProxyTorchDispatchMode
) -> None:
    """
    Records operators whose tensor outputs or inputs are fp16/bf16 so downstream pointwise code can
    emulate eager's rounding behavior when emulate_precision_casts is enabled.
    """
    if proxy_mode.decomp_layers or not proxy_mode.emulate_precision_casts:
        return

    if not isinstance(func, torch._ops.OpOverload):
        return

    last_node = next(iter(reversed(proxy_mode.tracer.graph.nodes)))
    t = last_node.meta.get("val")
    low_pr_fp = (torch.bfloat16, torch.float16)

    output_low_precision = isinstance(t, torch.Tensor) and t.dtype in low_pr_fp

    if not output_low_precision:
        for input_node in last_node.all_input_nodes:
            val = input_node.meta.get("val") if hasattr(input_node, "meta") else None
            if isinstance(val, torch.Tensor) and val.dtype in low_pr_fp:
                output_low_precision = True
                break

    if not output_low_precision:
        return

    last_node.meta["low_precision_pointwise_barrier"] = True


def _fetch_proxies_and_all_constant_flag(
    flat_args_kwargs: list[object] | tuple[object, ...], tracer: _ProxyTracer
) -> tuple[list[object], tuple[object, ...], bool]:
    """
    Given flat arguments, fetch the proxies and whether they are all constants.
    This is later used in proxy_call or when someone is trying to stitch together
    graph node in tf or td modes.
    """
    f_flat_args_kwargs = [
        (
            fetch_object_proxy(tracer, x)
            if isinstance(x, (Tensor, _AnyScriptObject)) or is_custom_class_obj(x)
            else x
        )
        for x in flat_args_kwargs
    ]

    # If there are SymInts, we also should not consider this constant.
    # However, fake tensor handling of SymInts is sufficiently broken that
    # I couldn't write a test for this case
    all_constant = (
        not any(
            t.constant is None
            for t in f_flat_args_kwargs
            if isinstance(t, _ProxyTensor)
        )
        # TODO: maybe constant SymInts should also be allowed?  Not sure if
        # this can happen
        and not any(isinstance(x, py_sym_types) for x in flat_args_kwargs)
    )

    proxy_flat_args_kwargs = [
        e.proxy if isinstance(e, _ProxyTensor) else e for e in f_flat_args_kwargs
    ]

    proxy_flat_args_kwargs = [
        (fetch_sym_proxy(tracer)(e) if isinstance(e, py_sym_types) else e)
        for e in proxy_flat_args_kwargs
    ]

    return f_flat_args_kwargs, tuple(proxy_flat_args_kwargs), all_constant


def _coor_enabled() -> bool:
    """True when compile-on-one-rank (CooR) device-as-parameter rewriting should run.

    Reads the cross-cutting flag from ``torch.compiler.config`` (its canonical home),
    honoring ``config.patch``.
    """
    from torch.compiler import config as compiler_config

    return compiler_config.compile_on_one_rank


def _coor_current_accelerator() -> torch.device | None:
    """The current accelerator as an indexed device (e.g. cuda:0), or None if there is no
    accelerator. Used to classify device operands under compile-on-one-rank."""
    # check_available matters: current_accelerator() reports the accelerator the build
    # supports, so on a CUDA-enabled build with no visible GPUs it returns cuda and the
    # index lookup below then raises "No CUDA GPUs are available". Such a machine has no
    # accelerator for our purposes, and a cpu-only graph must still compile there.
    acc = torch.accelerator.current_accelerator(check_available=True)
    if acc is None:
        return None
    return torch.device(acc.type, torch.accelerator.current_device_index())


def _coor_current_device() -> torch.device:
    """The current accelerator as a fully-specialized indexed device (e.g. cuda:0).

    Runtime target of the compile-on-one-rank device node: each rank resolves it to its
    own current device (cuda:0 on rank 0, cuda:3 on rank 3). Carries the concrete index
    (unlike type-only ``current_accelerator()``) so consuming ops -- and inductor -- get a
    fully specified device.
    """
    cur = _coor_current_accelerator()
    if cur is None:
        raise RuntimeError("compile-on-one-rank requires an accelerator")
    return cur


# Registered as an op (not a bare function) so it is a serializable call_function target
# with a stable identity for precompile/export and for consumers to match. It reads only
# torch.accelerator, so it lives in core fx with no torch.distributed coupling.
torch.library.define(
    "coor::current_device",
    "() -> Device",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.impl("coor::current_device", "CompositeExplicitAutograd")
def _coor_current_device_impl() -> torch.device:
    return _coor_current_device()


@torch.library.register_fake("coor::current_device")
def _coor_current_device_fake() -> torch.device:
    return _coor_current_device()


def _coor_check_current_accelerator(
    device: torch.device, cur: torch.device | None
) -> None:
    """Raise if ``device`` is an accelerator other than ``cur`` (``cpu``/``meta`` pass).

    A different accelerator type or a non-current index is rank-specific and cannot be made
    device-agnostic for compile-on-one-rank, so the graph is refused rather than left
    silently non-portable. Callers supply ``cur`` from their own per-scope cache of the
    current accelerator (the tracer while tracing, the wrapper during codegen).
    """
    if device.type in ("cpu", "meta"):
        return
    if cur is None or device.type != cur.type:
        raise RuntimeError(
            f"device_as_parameter: the graph references {device}, which is not the "
            f"current accelerator ({cur}); the traced graph cannot be made "
            f"device-agnostic for compile-on-one-rank."
        )
    if device.index is not None and device.index != cur.index:
        raise RuntimeError(
            f"device_as_parameter: the graph references {device}, whose index differs "
            f"from the current accelerator ({cur}); the traced graph cannot be made "
            f"device-agnostic for compile-on-one-rank."
        )


def _current_device_edge(tracer: _ProxyTracer, device: torch.device) -> Proxy:
    """Return a per-graph ``coor::current_device`` proxy for CooR, cached once per graph.

    Replaces a baked accelerator device operand (``cuda:0`` or bare ``cuda``) with a
    reference to a single node that, at runtime, yields each rank's specialized device
    (cuda:0 on rank 0, cuda:3 on rank 3). The node *call* is rank-agnostic; only its
    runtime return and ``node.meta["val"]`` vary, so ``.code`` matches across ranks (a
    graph fingerprint must therefore ignore the per-tensor devices in ``meta``).

    NOTE: a consumer that lowers this graph (e.g. inductor) cannot represent a
    device-returning op, so it must strip this node before lowering.
    """
    cached = tracer._current_device_node
    if cached is not None:
        return cached
    edge = tracer.create_proxy(
        "call_function", torch.ops.coor.current_device.default, (), {}
    )
    edge.node.meta["val"] = _coor_current_device()
    tracer._current_device_node = edge
    return edge


def _coor_traced_current_accelerator(tracer: _ProxyTracer) -> torch.device | None:
    """The current accelerator for this trace, read once (a cudaGetDevice) and cached on the
    tracer beside the current_device node, so the input check and the per-op device-operand
    rewrite together pay a single lookup per trace."""
    if tracer._coor_current_accelerator is None:
        tracer._coor_current_accelerator = _coor_current_accelerator()
    return tracer._coor_current_accelerator


def _coor_match_current_accelerator(tracer: _ProxyTracer, device: torch.device) -> bool:
    """Whether ``device`` should be replaced by the current-device node under CooR.

    True for the current accelerator (bare ``cuda`` or its current index); False for portable
    ``cpu``/``meta``; raises for a rank-specific device.
    """
    if device.type in ("cpu", "meta"):
        return False
    _coor_check_current_accelerator(device, _coor_traced_current_accelerator(tracer))
    return True


def proxy_call(
    proxy_mode: ProxyTorchDispatchMode,
    func: OpOverload,
    pre_dispatch: bool,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    unrecognized_types: list[type] = []
    flat_args_kwargs, spec = pytree.tree_flatten((args, kwargs))

    def can_handle_tensor(x: Tensor) -> bool:
        r = type(x) in HANDLED_TYPES or has_proxy_slot(x, proxy_mode.tracer)
        if proxy_mode._allow_fake_constant:
            r = r or type(x) is torch._subclasses.FakeTensor
        if not r:
            unrecognized_types.append(type(x))
        return r

    # If there are any tensor subclasses, we need to handle those tensor subclasses first
    # TODO: we could use types to test this
    if not all(can_handle_tensor(x) for x in flat_args_kwargs if isinstance(x, Tensor)):
        not_implemented_log.debug(
            "ProxyTensorMode tensors without proxy had unrecognized subclasses: %s",
            unrecognized_types,
        )
        return NotImplemented

    r = maybe_handle_decomp(proxy_mode, func, args, kwargs)
    if r is not NotImplemented:
        _maybe_record_pointwise_barrier(func, proxy_mode)
        return r

    # For pre-autograd tracing, we do not want to run CompositeImplicit decomps.
    if (
        not pre_dispatch
        and func
        not in [
            torch.ops.aten.size.default,
            torch.ops.aten.stride.default,
            torch.ops.aten.storage_offset.default,
        ]
        and autograd_would_have_decomposed(func, flat_args_kwargs)
    ):
        with proxy_mode:
            r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r

    if func is torch.ops.aten.is_nonzero.default:
        with proxy_mode:
            torch._check(
                args[0].numel() == 1,  # type: ignore[attr-defined]
                lambda: "Boolean value of Tensor with more than one value is ambiguous",
            )
            return (args[0] != 0).item()  # type: ignore[attr-defined]

    tracer = proxy_mode.tracer
    f_flat_args_kwargs, proxy_flat_args_kwargs, all_constant = (
        _fetch_proxies_and_all_constant_flag(flat_args_kwargs, tracer)
    )

    if torch.Tag.data_dependent_output in func.tags:
        # Check if all of the Tensor inputs are constants
        if all_constant:
            const_flat_args_kwargs = [
                t.constant if isinstance(t, _ProxyTensor) else t
                for t in f_flat_args_kwargs
            ]
            const_args, const_kwargs = pytree.tree_unflatten(
                const_flat_args_kwargs, spec
            )
            with unset_fake_temporarily():
                return func(*const_args, **const_kwargs)
        # If any of the Tensor inputs are "real" (not FakeTensor), we may
        # incorrectly burn in constants by allowing this access.  Raise
        # an error in this case
        if proxy_mode._error_on_data_dependent_ops and pytree.tree_all_only(
            Tensor, lambda t: not is_fake(t), (args, kwargs)
        ):
            raise RuntimeError(
                f"It appears that you're trying to get value out of a tracing tensor with {func} - erroring out! "
                "It's likely that this is caused by data-dependent control flow or similar.  "
                "It may be possible to trace this with dynamic shapes; try setting tracing_mode='symbolic' "
                "in your make_fx call."
            )

    # [device-as-parameter] Under compile-on-one-rank, replace a baked accelerator device
    # operand matching the current accelerator with a reference to a single current_device()
    # node, so the traced graph follows each rank's current device instead of the baked
    # device. Only the operand changes; the fake op below still runs with the real device.
    if _coor_enabled() and any(
        isinstance(e, torch.device) for e in proxy_flat_args_kwargs
    ):
        proxy_flat_args_kwargs = [
            _current_device_edge(tracer, e)
            if isinstance(e, torch.device)
            and _coor_match_current_accelerator(tracer, e)
            else e
            for e in proxy_flat_args_kwargs
        ]

    proxy_args, proxy_kwargs = pytree.tree_unflatten(proxy_flat_args_kwargs, spec)

    # When we trace through a torch.tensor invocation, you never actually
    # see a torch.ops.aten.tensor call. Instead, the way this function is
    # implemented internally is that we allocate a plain tensor (this is
    # *guaranteed* to be a plain tensor, we disable all modes when doing
    # so), and then call at::lift_fresh on it (to give modes a chance to do
    # their stuff).  Furthermore, the tensor argument to lift_fresh is guaranteed
    # to be freshly allocated, so we want lift_fresh to be a no-op (directly
    # returning the input argument).
    #
    # Here is the basic problem: when we trace this sequence of executions
    # into an FX graph, what happens to this call sequence?  Traditionally,
    # tensor constants get interned as buffers on the FX GraphModule.  But
    # this is dangerous.  Consider:
    #
    #       x = torch.tensor(1)
    #       x.add_(2)
    #
    # Naively, this traces into:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh(t)
    #       x.add_(2)
    #
    # If lift_fresh returns t directly, the subsequent add_ call will
    # modify the tensor constant. Really, the problem is we've violated
    # the invariant the argument to lift is fresh.  So what we should
    # preserve the invariant by replacing lift_fresh with lift_fresh_copy:
    #
    #       t = self._tensor_constant0  # initialized to torch.tensor(1)
    #       x = torch.ops.aten.lift_fresh_copy(t)
    #       x.add_(2)
    #
    # This is what the overload modification does.
    if func is torch.ops.aten.lift_fresh.default:
        func = torch.ops.aten.lift_fresh_copy.default

    proxy_out = proxy_mode.tracer.create_proxy(
        "call_function",
        func,
        proxy_args,
        proxy_kwargs,
        name=proxy_mode.tracer.graph._target_to_str(func.overloadpacket.__name__),
    )

    with _enable_thunkify(proxy_mode.tracer):
        out = func(*args, **kwargs)

    # In some circumstances, we will be tracing in a situation where a tensor
    # is *statically* known to be a constant (currently, this only happens if
    # you run torch.tensor; deterministic factory functions like torch.arange
    # don't get this treatment).  When the tensor in question is small, it's
    # helpful to due constant propagation in case we call item() (in which
    # case we can return the constant value that is known, rather than give
    # an error.)  The logic here tests if constant propagation is possible
    # (because all of the inputs are constant).  If so, we disable fake tensor
    # mode (if it is on) and do true compute on the constant.
    #
    # It's worth highlighting that we're making a policy decision here.
    # There is a potential that the tensor is actually quite large, and we
    # don't actually want to run the compute.  The tensor being quite large
    # is one of the reasons why factory functions don't get this treatment
    # (since they can be quite large; if a parameter is initialized to a
    # constant value it will be!)  Similarly, there is also a potential
    # to run an operator that blows up the size of a small tensor; we don't
    # protect against this case, but we could force, e.g., only single
    # element constant computation by testing the numel of the result before
    # propagating const-ness.  Similarly, we don't require the constant to
    # live on CPU, but we could.
    any_constant = any(
        t.constant is not None
        for t in f_flat_args_kwargs
        if isinstance(t, _ProxyTensor)
    )

    constant = None

    def tensor_numel_in_limit(t: Tensor) -> bool:
        return t.numel() <= CONSTANT_NUMEL_LIMIT

    # If this is a lift, the input tensor is guaranteed to be a
    # constant, so we keep a copy of the original argument along so
    # we can query it if we're asked to item() it at some later point
    if (
        func is torch.ops.aten.lift_fresh_copy.default
        and out.numel() <= CONSTANT_NUMEL_LIMIT
    ):
        with unset_fake_temporarily():
            if not isinstance(args[0], (Proxy, Tensor)):
                raise AssertionError(f"Expected Proxy or Tensor, got {type(args[0])}")
            constant = args[0].clone()
    elif (
        torch.Tag.nondeterministic_seeded not in func.tags
        and all_constant
        and any_constant
        and pytree.tree_all_only(Tensor, tensor_numel_in_limit, out)
    ):
        # NB: do NOT include factories as constants
        with unset_fake_temporarily():
            const_flat_args_kwargs = [
                t.constant if isinstance(t, _ProxyTensor) else t
                for t in f_flat_args_kwargs
            ]
            const_args, const_kwargs = pytree.tree_unflatten(
                const_flat_args_kwargs, spec
            )
            constant = func(*const_args, **const_kwargs)
    else:
        constant = None

    track_tensor_tree(
        out,
        proxy_out,
        # pyrefly: ignore[bad-argument-type]
        constant=constant,
        tracer=tracer,
    )
    _maybe_record_pointwise_barrier(func, proxy_mode)
    return out


class _SymNodeDict:
    """
    Wrapper around a dictionary that will hash SymInts with their nodes
    """

    def __init__(self) -> None:
        self.sym_node_dict: dict[PySymType, _PySymProxyType] = {}

    def __setitem__(self, key: PySymType, value: _PySymProxyType) -> None:
        # pyrefly: ignore [unsupported-operation]
        self.sym_node_dict[key.node] = value

    def __getitem__(self, key: PySymType) -> _PySymProxyType:
        # pyrefly: ignore [bad-index]
        return self.sym_node_dict[key.node]

    def __contains__(self, key: PySymType) -> bool:
        return key.node in self.sym_node_dict

    def get(
        self, key: PySymType, default: _PySymProxyType | None = None
    ) -> _PySymProxyType:
        # dict.get()'s annotation doesn't accept `None` when the value type
        # isn't Optional.
        return self.sym_node_dict.get(key.node, default)  # type: ignore[arg-type, return-value]

    def __iter__(self) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.sym_node_dict)


@dataclass
class _SympyExprTrackerValue:
    proxy: _PySymProxyType
    value: PySymType


def _init_proxy_trackers(tracer: PythonKeyTracer | _GraphAppendingTracerEx) -> None:
    """Initialize the tracker dictionaries shared by PythonKeyTracer and _GraphAppendingTracerEx."""
    tracer.tensor_tracker = WeakTensorKeyDictionary()
    tracer.symnode_tracker = _SymNodeDict()
    tracer.script_object_tracker = WeakIdKeyDictionary(dict=None, ref_type=_WeakHashRef)
    tracer.opaque_tracker = WeakIdKeyDictionary()
    tracer._opaque_real_obj_proxy = {}
    tracer.sympy_expr_tracker = {}
    tracer._current_device_node = None
    tracer._coor_current_accelerator = None
    # Stores the torch function that was called during tracing
    tracer.torch_fn_metadata = None
    # Stores the counts for every torch function called. This is to help
    # distinguish between different calls to the same torch function.
    tracer.torch_fn_counts = {}
    tracer.enable_thunkify = False


class PythonKeyTracer(Tracer):
    # ScriptObject uses _WeakHashRef because the same C++ IValue can produce
    # different Python wrapper objects, so Python id() won't match.
    script_object_tracker: MutableMapping[torch.ScriptObject, Proxy]
    # FakeScriptObject/CustomClassBase uses WeakIdRef because distinct objects that
    # are value-equal (e.g. primal vs tangent opaques) must be tracked separately.
    opaque_tracker: MutableMapping[FakeScriptObject | CustomClassBase, Proxy]
    # Maps id(real_obj) -> proxy for opaque FSOs, so that multiple FSO wrappers
    # of the same real object (e.g. primal vs tangent) resolve to one proxy.
    _opaque_real_obj_proxy: dict[int, Proxy]
    symnode_tracker: _SymNodeDict
    sympy_expr_tracker: dict[sympy.Symbol, _SympyExprTrackerValue]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    # [device-as-parameter] the single current_device() node for this graph (CooR)
    _current_device_node: Proxy | None
    # [device-as-parameter] current accelerator for this graph, read once (None = not yet)
    _coor_current_accelerator: torch.device | None
    torch_fn_metadata: OpOverload | None
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool = False

    def __init__(self) -> None:
        super().__init__(autowrap_modules=())  # type: ignore[arg-type]
        _init_proxy_trackers(self)

    # In general, we don't want to make modules leaves. In principle, users of
    # this tracer might want to override this in order to turn a couple specific
    # modules into leaves in the traced graph.
    def call_module(
        self,
        m: Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return forward(*args, **kwargs)

    # We don't want to turn getattr calls into proxies. So we just return the actual value.
    def getattr(
        self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]
    ) -> object:
        return attr_val

    def create_arg(self, a: object) -> fx.node.Node:
        if isinstance(a, torch.nn.Parameter):
            for n, p in self.root.named_parameters():
                if a is p:
                    return self.create_node("get_attr", n, (), {})

            qualname = self.get_fresh_qualname("_param_constant")
            setattr(self.root, qualname, a)

            return self.create_node("get_attr", qualname, (), {})
        elif isinstance(a, py_sym_types):
            if a.node.constant is None:
                raise AssertionError("a.node.constant should not be None")
            # pyrefly: ignore [bad-return]
            return a.constant

        # Try reconstructing untracked opaque reference types from existing
        # graph inputs (e.g. derive a DeviceMesh submesh from its root mesh).
        if isinstance(a, (FakeScriptObject, CustomClassBase)):
            node = self._try_reconstruct_opaque(a)
            if node is not None:
                return node

        return super().create_arg(a)  # type: ignore[return-value]

    def _try_reconstruct_opaque(
        self, a: FakeScriptObject | CustomClassBase
    ) -> fx.node.Node | None:
        """Try to reconstruct an opaque object from existing graph inputs.

        When make_fx encounters an untracked opaque reference type (e.g. a
        DeviceMesh submesh captured by a backward closure), this method checks
        if the type has a registered reconstruct_fn that can derive the object
        from inputs already in the graph.  Returns an FX Node on success,
        None on failure (falls back to get_attr constant).
        """
        real_obj: CustomClassBase = a.real_obj if isinstance(a, FakeScriptObject) else a

        if not is_opaque_symbolic_type(type(real_obj)):
            return None

        reconstruct_fn = get_reconstruct_fn(type(real_obj))
        if reconstruct_fn is None:
            return None

        def get_tracked_proxy(obj: CustomClassBase) -> Proxy | None:
            proxy = self._opaque_real_obj_proxy.get(id(obj))
            if proxy is not None:
                return proxy
            proxy = get_proxy_slot(obj, self, None)
            if proxy is not None:
                return proxy
            # The object may be identity-different but equal to a tracked
            # FSO's real_obj. This happens because maybe_to_fake_obj creates
            # a new DeviceMesh wrapper, so the FSO's real_obj is a distinct
            # object from the one held by submesh._root_mesh. Equality
            # comparison is needed, not identity.
            for tracked_obj, p in self.opaque_tracker.items():
                if not isinstance(tracked_obj, FakeScriptObject):
                    continue
                if tracked_obj.real_obj == obj:
                    return p
            return None

        result = reconstruct_fn(real_obj, get_tracked_proxy, self)
        if result is None:
            return None

        # The reconstruct_fn dispatches through a custom op with a Proxy
        # argument, which goes through Proxy.__torch_function__.  That path
        # creates a graph node but does NOT populate meta["val"] (unlike the
        # ProxyTorchDispatchMode.__torch_dispatch__ path which calls
        # track_tensor_tree).  Set it here so downstream consumers (e.g. the
        # min-cut partitioner) can classify the node correctly.
        if "val" not in result.node.meta:
            set_meta(result, a)

        # Also register for the *input* object so dedup works for re-encounters.
        set_proxy_slot(a, self, result)
        if id(real_obj) not in self._opaque_real_obj_proxy:
            self._opaque_real_obj_proxy[id(real_obj)] = result

        return result.node

    @overload
    def unwrap_proxy(self, e: Tensor) -> Proxy | Tensor: ...

    @overload
    def unwrap_proxy(self, e: PySymType) -> Proxy | PySymType: ...

    @overload
    def unwrap_proxy(self, e: _AnyScriptObjectType) -> Proxy | _AnyScriptObjectType: ...

    def unwrap_proxy(self, e: T) -> object:
        if isinstance(e, Tensor):
            return get_proxy_slot(e, self, e, lambda x: x.proxy)  # type: ignore[attr-defined]
        elif isinstance(e, py_sym_types):
            return get_proxy_slot(e, self, e, lambda e: e.force())
        elif isinstance(e, _AnyScriptObject) or is_custom_class_obj(e):
            return get_proxy_slot(e, self, e)
        else:
            return e

    def create_node(
        self,
        kind: str,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Argument],
        name: str | None = None,
        type_expr: Any | None = None,
    ) -> torch.fx.Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)  # type: ignore[arg-type]

        if node.op in ["placeholder", "output"] and "stack_trace" in node.meta:
            del node.meta["stack_trace"]

        if kind == "get_attr":
            if not isinstance(target, str):
                raise AssertionError(f"Expected str, got {type(target)}")
            attr = getattr(self.root, target)
            if isinstance(attr, torch.Tensor):
                with disable_proxy_modes_tracing():
                    node.meta["val"] = extract_val(attr)

        def map_fn(v: Any) -> _ExtractValType | None:
            if not isinstance(v, torch.fx.Node) or "val" not in v.meta:
                return None
            val = v.meta["val"]
            # other subclasses like FunctionalTensor error on `extract_val`
            # "Attempting to use FunctionalTensor on its own." just store FakeTensors for now
            if isinstance(val, torch.Tensor) and not is_fake_tensor(val):
                return None
            return extract_val(v.meta["val"])

        if _should_save_eager_input_vals(target, (args, kwargs)):
            # NOTE "eager_input_vals"
            # We save the original (args, kwargs) FakeTensor values for nodes
            # that have exact stride requirements. This is useful downstream.
            # We use this information inside Inductor to ensure that inputs to
            # stride-sensitive operators have the correct strides.
            arg_inp, kwarg_inp = torch.fx.node.map_aggregate((args, kwargs), map_fn)  # type: ignore[misc, arg-type]
            node.meta["eager_input_vals"] = (arg_inp, kwarg_inp)

        return node


def _should_save_eager_input_vals(
    target: Any,
    args_kwargs: tuple[tuple[Argument, ...], dict[str, Argument]] | None = None,
) -> bool:
    from torch._higher_order_ops.invoke_subgraph import InvokeSubgraphHOP

    if not callable(target):
        return False
    if isinstance(
        target,
        (
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperFunctional,
            torch._higher_order_ops.triton_kernel_wrap.TritonKernelWrapperMutation,
            InvokeSubgraphHOP,
        ),
    ):
        return True
    if args_kwargs is not None and (
        target is torch.ops.higher_order.auto_functionalized
        or target is torch.ops.higher_order.auto_functionalized_v2
    ):
        args = args_kwargs[0]
        if not isinstance(
            args[0], (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
        ):
            raise AssertionError(
                f"Expected OpOverload or HigherOrderOperator, got {type(args[0])}"
            )
        return _should_save_eager_input_vals(args[0], None)
    if target is torch.ops.higher_order.with_effects:
        # TODO: inductor lowering for with_effects needs to be updated to propagate
        # the arg_kwarg_vals
        return False
    if isinstance(target, torch._ops.HigherOrderOperator):
        if pytree.tree_any(_should_save_eager_input_vals, args_kwargs):
            raise RuntimeError(
                f"NYI: The HOP {target} has an input that is an OpOverload that "
                f"needs exact strides. We probably need special logic to "
                f"propagate the FakeTensor vals. Please file an issue."
            )
    if isinstance(target, torch._ops.OpOverload):
        from torch._library.utils import get_layout_constraint_tag

        return get_layout_constraint_tag(target) == torch._C.Tag.needs_exact_strides
    return False


def _make_temp_remove_mode_context_manager(
    mode_ty: type[TorchFunctionMode],
) -> Callable[[], _GeneratorContextManager[TorchFunctionMode | None]]:
    @contextmanager
    def context_manager_fn() -> Generator[TorchFunctionMode | None, None, None]:
        from torch.overrides import _len_torch_function_stack, _pop_mode, _push_mode

        temp_elements = []
        removed_mode = None

        while _len_torch_function_stack() > 0:
            mode = _pop_mode()
            if isinstance(mode, mode_ty):
                removed_mode = mode
                break
            else:
                temp_elements.append(mode)

        for mode in reversed(temp_elements):
            _push_mode(mode)

        try:
            yield removed_mode

        finally:
            if removed_mode is not None:
                count = len(temp_elements)
                while count > 0:
                    mode = _pop_mode()
                    count -= 1

                temp_elements.append(removed_mode)

                for mode in reversed(temp_elements):
                    _push_mode(mode)

    return context_manager_fn


@torch._disable_dynamo
def dispatch_trace(
    root: Module | Callable[..., Any],
    tracer: Tracer,
    concrete_args: tuple[object, ...] | None = None,
) -> GraphModule:
    graph = tracer.trace(root, concrete_args)  # type: ignore[arg-type]

    # NB: be careful not to DCE .item() calls
    def impure_pred(n: fx.Node) -> bool:
        from .symbolic_shapes import is_accessor_node

        # Always defer to the built-in notion of impure
        if n.is_impure():
            return True

        # Accessors always OK to DCE
        if is_accessor_node(n):
            return False

        # If the operator in question takes SymInt args to SymInt output,
        # we assume it's pure and OK to DCE
        if (
            isinstance(n.meta.get("val"), py_sym_types)
            and
            # NB: constant args ok
            all(
                isinstance(a.meta.get("val"), py_sym_types)
                for a in n.args
                if isinstance(a, fx.Node)
            )
        ):
            return False

        # No idea, just assume it's not OK
        return True

    graph.eliminate_dead_code(impure_pred)
    from torch._inductor.fx_passes.dedupe_symint_uses import dedupe_symints

    dedupe_symints(graph)
    name = root.__class__.__name__ if isinstance(root, Module) else root.__name__
    return fx._lazy_graph_module._make_graph_module(tracer.root, graph, name)


def wrap_key(
    f: Callable[[Unpack[_Ts]], R],
    tensors: tuple[Unpack[_Ts]],
    tracer: _ProxyTracer,
    pre_dispatch: bool,
) -> Callable[_P, R]:
    flat_tensors, _tensors_spec = pytree.tree_flatten(tensors)

    @functools.wraps(f)
    def wrapped(*proxies: _P.args, **_unused: _P.kwargs) -> R:
        nonlocal tensors

        flat_proxies, _proxies_spec = pytree.tree_flatten(proxies)
        if len(flat_proxies) != len(flat_tensors):
            raise AssertionError(
                f"Expected same length: {len(flat_proxies)} vs {len(flat_tensors)}"
            )
        with disable_proxy_modes_tracing() as m:
            if not isinstance(m, ProxyTorchDispatchMode):
                raise AssertionError(f"Expected ProxyTorchDispatchMode, got {type(m)}")
            track_tensor_tree(flat_tensors, flat_proxies, constant=None, tracer=tracer)
            # `wrap_key` is the make_fx input-placeholder path. Keep scalar
            # metadata here instead of teaching recursive output tracking to
            # treat arbitrary primitive leaves as proxy-trackable values.
            for input_value, proxy in zip(flat_tensors, flat_proxies):
                if isinstance(input_value, (int, float, bool)) and isinstance(
                    proxy, fx.Proxy
                ):
                    set_meta(proxy, input_value)

        # [device-as-parameter] Under compile-on-one-rank a graph input on a non-current
        # accelerator is rank-specific and can't be made device-agnostic (the wrapper
        # resolves a single current device at runtime, so the whole graph must live on it).
        # Inductor's pattern-matcher templates are traced with CooR off, so this only sees
        # real user graphs, not internal fixed-device template traces.
        if _coor_enabled():
            cur = _coor_traced_current_accelerator(tracer)
            for input_value in flat_tensors:
                if isinstance(input_value, torch.Tensor):
                    _coor_check_current_accelerator(input_value.device, cur)

        if getattr(tracer, "proxy_module_inputs", False):
            tensors = [  # type: ignore[assignment, var-annotated]
                p if isinstance(t, torch.nn.Module) else t
                for t, p in zip(tensors, proxies)  # type: ignore[arg-type]
            ]

        def get_tensor_proxy_slot(t: Tensor) -> Tensor | Proxy:
            return get_proxy_slot(t, tracer, t, lambda x: x.proxy)  # type: ignore[attr-defined]

        out = f(*tensors)  # type:ignore[call-arg]
        out = pytree.tree_map_only(Tensor, get_tensor_proxy_slot, out)
        out = pytree.tree_map_only(
            _AnyScriptObject, lambda t: get_proxy_slot(t, tracer, t, lambda x: x), out
        )
        out = pytree.tree_map_only(
            CustomClassBase, lambda t: get_proxy_slot(t, tracer, t, lambda x: x), out
        )

        def get_sym_proxy_slot(t: PySymType) -> Proxy:
            return get_proxy_slot(t, tracer).force()

        out = pytree.tree_map_only(py_sym_types, get_sym_proxy_slot, out)
        return out

    return wrapped


# TODO: Make downstream users of this work with OperatorBase
ORIGINAL_ATEN: contextvars.ContextVar[object | None] = contextvars.ContextVar(
    "ORIGINAL_ATEN", default=None
)


@contextmanager
def set_original_aten_op(
    func: OpOverload | torch._ops.HigherOrderOperator,
) -> Generator[None, None, None]:
    if ORIGINAL_ATEN.get() is None and fx_traceback.has_preserved_node_meta():
        token = ORIGINAL_ATEN.set(func)
        fx_traceback.current_meta["original_aten"] = func
        try:
            yield
        finally:
            ORIGINAL_ATEN.reset(token)
            fx_traceback.current_meta["original_aten"] = None
    else:
        yield


class TorchFunctionMetadataMode(TorchFunctionMode):
    def __init__(self, tracer: _ProxyTracer) -> None:
        self.tracer = tracer

    def __torch_function__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        # pyrefly: ignore [bad-assignment]
        self.tracer.torch_fn_metadata = func
        self.tracer.torch_fn_counts[func] = self.tracer.torch_fn_counts.get(func, 0) + 1
        if (
            func is torch.ops.aten.size.default
            and len(args) == 1
            and isinstance(args[0], Tensor)
            and any(isinstance(s, py_sym_types) for s in args[0].size())
        ):
            return torch.ops.aten.sym_size.default(*args, **kwargs)
        return func(*args, **kwargs)


_temp_remove_metadata_torch_function_mode = _make_temp_remove_mode_context_manager(
    TorchFunctionMetadataMode
)


# This mode is **only** used for pre_dispatch tracing.
# In particular, we need to make sure that autograd/autocast API's
# that do not desugar into dispatcher operators stay in the graph.
class PreDispatchTorchFunctionMode(TorchFunctionMode):
    def __init__(self, tracer: _ProxyTracer) -> None:
        self.tracer = tracer
        # The input to torch.amp.autocast_mode._exit_autocast graph node should be the
        # enter_autocast node. So we have to save the enter autocast node here, and assign it
        # to the exit_autocast call_function node.
        self.enter_autocast_nodes: list[torch.fx.Node] = []

    def __torch_function__(
        self,
        func: OpOverload | Callable[..., Any],
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        if func in _side_effectful_need_to_be_preserved_pre_dispatch:
            # It's for passing the export verifier which needs to verify the meta['val']
            # TODO(tmanlaibaatar): we should systematically couple it with export verifier,
            # instead of hardcoding it here.
            # T203648563
            if func is torch.amp.autocast_mode._exit_autocast:
                enter_node = self.enter_autocast_nodes.pop()
                args = (enter_node,)
            node = self.tracer.create_node("call_function", func, args, {})  # type: ignore[arg-type]
            if func is torch.amp.autocast_mode._enter_autocast:
                self.enter_autocast_nodes.append(node)
            if func in [
                torch._C._set_grad_enabled,
                torch.amp.autocast_mode._enter_autocast,
                torch.amp.autocast_mode._exit_autocast,
            ]:
                node.meta["val"] = None
            # For autocast, the python APIs run so we don't have to run them again
            # here.
            if func is torch._C._set_grad_enabled:
                # pyrefly: ignore [bad-argument-type]
                func(*args, **kwargs)
            return node
        # We need more complicated handling here because the inputs
        # to these functions are sometimes tensors or symints where
        # we need to fetch the proxies properly.
        if func in [
            torch._functorch.predispatch._add_batch_dim,
            torch._functorch.predispatch._remove_batch_dim,
            torch._functorch.predispatch._vmap_increment_nesting,
            torch._functorch.predispatch._vmap_decrement_nesting,
            torch._functorch.vmap.lazy_load_decompositions,
            torch._functorch.predispatch._make_dual,
            torch._functorch.predispatch._unpack_dual,
            torch._functorch.predispatch._jvp_increment_nesting,
            torch._functorch.predispatch._jvp_decrement_nesting,
            torch._functorch.predispatch._unwrap_for_grad,
            torch._functorch.predispatch._enter_dual_level,
            torch._functorch.predispatch._exit_dual_level,
        ]:
            _, proxies, _ = _fetch_proxies_and_all_constant_flag(args, self.tracer)
            out_proxy = self.tracer.create_proxy(
                "call_function",
                func,
                proxies,
                kwargs,
            )
            res = func(*args, **kwargs)
            # When JVP transforms are active, snapshot_fake calls detach
            # which goes through C++ functorch dispatch keys, potentially
            # re-wrapping the result as a grad tracking tensor.
            # Temporarily disable functorch transforms during tracking to
            # prevent this corruption of meta["val"].
            if func in _jvp_predispatch_functions:
                with torch._C._DisableFuncTorch():
                    track_tensor_tree(res, out_proxy, constant=None, tracer=self.tracer)
            else:
                track_tensor_tree(res, out_proxy, constant=None, tracer=self.tracer)
            return res
        return func(*args, **kwargs)


_temp_remove_pre_dispatch_torch_function_mode = _make_temp_remove_mode_context_manager(
    PreDispatchTorchFunctionMode
)


# JVP predispatch functions need special handling during tracing:
# _DisableFuncTorch prevents grad tracking tensor corruption in snapshot_fake.
_jvp_predispatch_functions = frozenset(
    {
        torch._functorch.predispatch._make_dual,
        torch._functorch.predispatch._unpack_dual,
        torch._functorch.predispatch._jvp_increment_nesting,
        torch._functorch.predispatch._jvp_decrement_nesting,
        torch._functorch.predispatch._unwrap_for_grad,
        torch._functorch.predispatch._enter_dual_level,
        torch._functorch.predispatch._exit_dual_level,
    }
)

# These JVP predispatch wrappers manage transform nesting/level state
# and must survive FX dead code elimination even with zero output users.
# Registered here rather than in predispatch.py to avoid circular imports
# (predispatch.py loads before torch.fx during torch.autograd init).
from torch.fx.node import _side_effectful_functions


_side_effectful_functions.update(
    {
        torch._functorch.predispatch._jvp_increment_nesting,
        torch._functorch.predispatch._jvp_decrement_nesting,
        torch._functorch.predispatch._enter_dual_level,
        torch._functorch.predispatch._exit_dual_level,
    }
)


class ProxyTorchDispatchMode(TorchDispatchMode):
    # Ensure this is read-only; this exists only for legacy reasons
    @property
    def enable_tracing(self) -> bool:
        return True

    def __init__(
        self,
        tracer: _ProxyTracer,
        tracing_mode: _TracingMode,
        pre_dispatch: bool = False,
        _allow_fake_constant: bool = False,
        _error_on_data_dependent_ops: bool = True,
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None = None,
    ) -> None:
        dk = torch._C.DispatchKey.PreDispatch if pre_dispatch else None
        super().__init__(dk)
        self.tracer = tracer
        self.tracing_mode = tracing_mode
        self.pre_dispatch = pre_dispatch
        self._allow_fake_constant = _allow_fake_constant
        self._error_on_data_dependent_ops = _error_on_data_dependent_ops
        # Indicates to our torch_dispatch dispatching infra that
        # this is an "infra" mode with lower dispatching precedence.
        self._mode_key = torch._C._TorchDispatchModeKey.PROXY
        # Every time we enter a mode, we maintain a stack telling us what the previous
        # ProxyTorchDispatchMode state was (if there was any).
        # This lets us properly reset the state on exit.
        self.enter_stack: list[ProxyTorchDispatchMode | None] = []
        self.decomposition_table: Mapping[OpOverload, Callable[..., Any]] = (
            decomposition_table or {}
        )
        self._decomposition_table_stack: list[
            Mapping[OpOverload, Callable[..., Any]]
        ] = []
        self.decomp_layers: int = 0
        # See invoke_subgraph
        self._invoke_subgraph_names: set[str] = set()
        self._invoke_subgraph_cache: dict[
            torch.fx.GraphModule | FunctionalizeCtxWrapper, str
        ] = {}
        from torch._inductor import config

        self.emulate_precision_casts: bool = config.emulate_precision_casts

    @count
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        with set_original_aten_op(func):
            kwargs = kwargs or {}

            if func == prim.device.default:
                return func(*args, **kwargs)

            return proxy_call(self, func, self.pre_dispatch, args, kwargs)

    def __enter__(self) -> Self:
        # Stash and store the previous proxy mode (there may or may not be one)
        maybe_prev_proxy_mode = _unset_infra_mode(torch._C._TorchDispatchModeKey.PROXY)
        self.enter_stack.append(maybe_prev_proxy_mode)
        return super().__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        b = super().__exit__(exc_type, exc_value, traceback)

        # Re-enable the previous proxy mode, if there was one.
        mb_previous_proxy_mode = self.enter_stack.pop()
        if mb_previous_proxy_mode is not None:
            _push_mode(mb_previous_proxy_mode)

        return b

    @classmethod
    def is_infra_mode(cls) -> bool:
        return True

    @contextmanager
    def enable_decompositions(
        self,
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None,
    ) -> Generator[Mapping[OpOverload, Callable[..., Any]], None, None]:
        table = decomposition_table or {}
        self._decomposition_table_stack.append(self.decomposition_table)
        self.decomposition_table = table
        try:
            yield table
        finally:
            self.decomposition_table = self._decomposition_table_stack.pop()

    def __sym_dispatch__(
        self,
        func: OpOverload,
        types: tuple[torch._C._TensorMeta, ...],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        # Peephole optimize multiply by one
        # NB: be careful not to trigger guards here!
        if func is operator.mul:
            if isinstance(args[1], int) and args[1] == 1:
                return args[0]
            elif isinstance(args[0], int) and args[0] == 1:
                return args[1]

        # For speed, we assume there are no nested data structures
        # (otherwise we could use tree_map)
        # We also assume there are no keyword arguments.
        if kwargs:
            raise AssertionError("Expected no keyword arguments")
        out = func(*args, **kwargs)
        _sym_register(self.tracer, func, args, out)
        return out


def _sym_register(
    tracer: _ProxyTracer, func: OpOverload, args: tuple[object, ...], out: object
) -> None:
    # If func returned a constant, we don't need to trace; we have
    # determined that the result is constant (no matter if the inputs
    # were symbolic) and it is no longer necessary to trace the
    # computation.  This could occur if func triggered some guards.
    if isinstance(out, py_sym_types):
        p_out_thunk = thunkify(
            tracer, _compute_proxy, tracer, func=func, args=args, out=out
        )
        set_proxy_slot(out, tracer, p_out_thunk)


def _compute_proxy(
    tracer: _ProxyTracer, func: OpOverload, args: tuple[object, ...], out: PySymType
) -> Proxy:
    # Handle torch.sym_sum
    n_args: tuple[object, ...]
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        n_args = (
            tuple(
                (
                    get_proxy_slot(a, tracer).force().node
                    if isinstance(a, py_sym_types)
                    else a
                )
                for a in args[0]
            ),
        )
    else:
        n_args = tuple(
            (
                get_proxy_slot(a, tracer).force().node
                if isinstance(a, py_sym_types)
                else a
            )
            for a in args
        )

    # func doesn't have a __torch_function__ that Proxy can interpose, so
    # we gotta do it manually
    n_out = tracer.create_node("call_function", func, n_args, {})  # type: ignore[arg-type]
    p_out = fx.Proxy(n_out, tracer)
    set_meta(p_out, out)
    return p_out


class _GraphAppendingTracerEx(fx.proxy.GraphAppendingTracer):
    script_object_tracker: MutableMapping[torch.ScriptObject, Proxy]
    opaque_tracker: MutableMapping[FakeScriptObject | CustomClassBase, Proxy]
    # Maps id(real_obj) -> proxy for opaque FSOs, so that multiple FSO wrappers
    # of the same real object (e.g. primal vs tangent) resolve to one proxy.
    _opaque_real_obj_proxy: dict[int, Proxy]
    symnode_tracker: _SymNodeDict
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    sympy_expr_tracker: dict[sympy.Symbol, _SympyExprTrackerValue]
    # [device-as-parameter] the single current_device() node for this graph (CooR)
    _current_device_node: Proxy | None
    # [device-as-parameter] current accelerator for this graph, read once (None = not yet)
    _coor_current_accelerator: torch.device | None
    torch_fn_metadata: OpOverload | None
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool = False

    def __init__(self, graph: fx.graph.Graph) -> None:
        super().__init__(graph)
        _init_proxy_trackers(self)


# TODO: I'm not sure what the point of this class is; you can just
# make_fx through a regular Interpreter
class DecompositionInterpreter(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        new_graph: fx.Graph,
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(module, **kwargs)  # type: ignore[arg-type]
        self.new_graph = new_graph
        self.tracer = _GraphAppendingTracerEx(self.new_graph)
        # Blegh
        self.decomposition_table = decomposition_table or {}
        self.mode = ProxyTorchDispatchMode(
            self.tracer,
            tracing_mode="real",
            decomposition_table=self.decomposition_table,
        )

    # pyrefly: ignore [bad-override]
    def placeholder(
        self,
        target: str,  # type: ignore[override]
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        out = super().placeholder(target, args, kwargs)  # type: ignore[arg-type]
        proxy = fx.Proxy(self.new_graph.placeholder(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        # Primitive scalar placeholders are real graph inputs. Record their
        # value on the placeholder itself without making recursive output
        # tracking treat all primitive leaves as proxy-trackable values.
        if isinstance(out, (int, float, bool)):
            set_meta(proxy, out)
        # TODO handle case where the first character of target is '*'
        return out

    # pyrefly: ignore [bad-override]
    def get_attr(
        self,
        target: str,  # type: ignore[override]
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        out = super().get_attr(target, args, kwargs)  # type: ignore[arg-type]
        proxy = fx.Proxy(self.new_graph.get_attr(target), self.tracer)
        track_tensor_tree(out, proxy, constant=None, tracer=self.tracer)
        return out

    # call_function, call_method, call_module get traced automatically by the outer mode.

    # pyrefly: ignore [bad-override]
    def output(
        self,
        target: str,  # type: ignore[override]
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        out = super().output(target, args, kwargs)  # type: ignore[arg-type]

        def get_proxy_node(x: _ProxyTensor) -> fx.node.Node:
            return x.proxy.node

        def unwrap(e: Tensor) -> Tensor | fx.Node:
            return get_proxy_slot(e, self.tracer, e, get_proxy_node)

        self.new_graph.output(pytree.tree_map(unwrap, out))
        return out

    def run(self, *args: object, **kwargs: object) -> object:
        # Should enter the mode at least once for being able to restore it later
        # See: https://github.com/pytorch/pytorch/pull/82549#discussion_r934782025
        with self.mode:
            return super().run(*args, **kwargs)  # type: ignore[arg-type]


class _SelectiveDecomposeInterpreter(fx.Interpreter):
    def __init__(
        self,
        module: fx.GraphModule,
        should_decompose: Callable[[fx.Node], bool],
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None,
        **kwargs: object,
    ) -> None:
        """
        For all nodes in `module`, selectively decompose if is `should_decompose`,
        following the given `decomposition_table`.
        """
        super().__init__(module, **kwargs)  # type: ignore[arg-type]
        self.should_decompose = should_decompose
        self.decomposition_table = decomposition_table

    @staticmethod
    def recursive_wrap(
        gm: fx.GraphModule,
        should_decompose: Callable[[fx.Node], bool],
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None,
        **kwargs: object,
    ) -> _SelectiveDecomposeInterpreter:
        """
        Recursively wrap gm and its sub graph modules. Specifically, HOP takes
        sub graph module as args. We may not want to decompose all nodes within
        these sub graph modules. So we also need to wrap these sub graph modules.
        As a result:
        - if should_decompose(hop) is True, we decompose all nodes within the hop.
        - if should_decompose(hop) is False, we check each node within the hop
            and decide whether decompose or not.
        """
        for node in gm.graph.nodes:
            if node.op == "call_function" and isinstance(
                node.target, HigherOrderOperator
            ):
                new_args = []
                for arg in node.args:
                    if isinstance(arg, fx.GraphModule):
                        new_arg = _SelectiveDecomposeInterpreter.recursive_wrap(
                            arg, should_decompose, decomposition_table, **kwargs
                        )
                    else:
                        new_arg = arg
                    new_args.append(new_arg)
                node.args = tuple(new_args)

        return _SelectiveDecomposeInterpreter(
            gm, should_decompose, decomposition_table, **kwargs
        )

    def run_node(self, n: fx.Node) -> Any:
        if self.should_decompose(n):
            with decompose(self.decomposition_table):
                result = super().run_node(n)
        else:
            result = super().run_node(n)
        return result


def selective_decompose(
    joint_gm: fx.GraphModule,
    *args: object,
    decomposition: Mapping[OpOverload, Callable[..., Any]] | None,
    should_decompose: Callable[..., bool],
    trace_joint_graph: bool,
) -> fx.GraphModule:
    """Retrace a joint graph module and selectively apply decomposition."""

    if trace_joint_graph:
        # the arg name, primals and tangents, are important.
        # make_fx keeps the name in the traced graph and partitioner later relies
        # on the name to partition joint graph correctly.
        def wrap_fn(primals: list[Any], tangents: list[Any]) -> Any:
            return _SelectiveDecomposeInterpreter.recursive_wrap(
                joint_gm, should_decompose, decomposition
            ).run(*args)
    else:

        def wrap_fn(*args: Any) -> Any:
            return _SelectiveDecomposeInterpreter.recursive_wrap(
                joint_gm, should_decompose, decomposition
            ).run(*args)

    return make_fx(wrap_fn, decomposition_table={})(*args)


def wrapper_and_args_for_make_fx(
    func: Callable[..., R], args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[Callable[[list[object]], R], list[object]]:
    # make_fx doesn't support kwargs, so we need to do this flattening
    # and then unflatten the args before calling func
    flat_args, spec = pytree.tree_flatten((args, kwargs))

    def wrapped(flat_args: list[object]) -> R:
        fn_args, fn_kwargs = pytree.tree_unflatten(flat_args, spec)
        return func(*fn_args, **fn_kwargs)

    return wrapped, flat_args


@contextmanager
def disable_autocast_cache() -> Generator[None, None, None]:
    old_value = torch.is_autocast_cache_enabled()
    torch.set_autocast_cache_enabled(False)
    try:
        yield
    finally:
        torch.set_autocast_cache_enabled(old_value)


class _ModuleNotInstalledAsSubmoduleError(NameError):
    pass


# Base class for inline _ModuleStackTracer.__init__.AttrProxy
class _AttrProxy:
    def reset_proxy_mapping(self, base: Module, path: str) -> None:
        pass


class _ModuleStackTracer(PythonKeyTracer):
    r"""Customized version of PythonKeyTracer that retains module stack
    information in node.meta["nn_module_stack"].

    FX symbolic trace actually does this already, but it relies on `self.root`
    being the actual module being traced. Since make_fx traces a lambda of our
    creation, things don't work properly.

    So for this version we hold onto a reference to the original module
    (scope_root) and use that to match the path. Also when we see,
            A
           / \
          B   C
           \ /
            D
    we want to record the path as A.B.D by recording only one path.
    See Note [Preserving the nn module stack metadata during export non-strict mode]  # noqa: W605
    """

    def __init__(self, scope_root: GraphModule) -> None:
        super().__init__()
        self.record_stack_traces = not fx.config.do_not_emit_stack_traces
        self._record_forward_stack_traces_only = True
        self.scope_root = scope_root
        self.enable_attr_proxy = False
        self.submodule_paths = {}
        for name, m in self.scope_root.named_modules(remove_duplicate=False):
            if m in self.submodule_paths:
                log.info(
                    "Shared module found between %s and %s, AttrProxy is enabled.",
                    self.submodule_paths[m],
                    name,
                )
                self.enable_attr_proxy = True
            else:
                self.submodule_paths[m] = name

        self.proxy_paths: WeakKeyDictionary[_AttrProxy, str] = WeakKeyDictionary()
        self.attr_proxy_map: WeakKeyDictionary[Module, _AttrProxy] = WeakKeyDictionary()
        self.proxy_modules: WeakKeyDictionary[_AttrProxy, Module] = WeakKeyDictionary()
        self.counter = 0

        self.module_id_cache = defaultdict(list)
        for name, mod in self.scope_root.named_modules(remove_duplicate=False):
            self.module_id_cache[id(mod)].append(name)

        # Build a wrapper around _AttrProxy to provide the tracer. We can't
        # store it on _AttrProxy itself because we mimic the underlying class
        # (including its attributes).
        tracer = self

        class AttrProxy(_AttrProxy):
            def __init__(self, base: Module | _AttrProxy, path: str) -> None:
                if isinstance(base, _AttrProxy):
                    base = base.get_base()  # type: ignore[attr-defined]

                if not isinstance(base, Module):
                    raise AssertionError(f"Expected Module, got {type(base)}")
                # Class is modified to be a subclass of torch.nn.Module
                # Warning: We blow away our own attributes here to mimic the base class
                # - so don't expect `self.x` to do anything useful.
                # pyrefly: ignore [no-matching-overload]

                self.__class__ = type(
                    base.__class__.__name__,
                    (self.__class__, base.__class__),
                    {},
                )
                self.__dict__ = base.__dict__
                self.__class__.__module__ = base.__class__.__module__
                self.__class__.__qualname__ = base.__class__.__qualname__

                # This overwrites any existing paths if `base` is an AttrProxy
                tracer.proxy_paths[self] = path
                tracer.proxy_modules[self] = base

            def __getattr__(self, name: str) -> AttrProxy:
                if not isinstance(self, Module):
                    raise AssertionError(f"Expected Module, got {type(self)}")
                # Calling into torch.nn.Module.__getattr__ with super(),
                # That __getattr__ is patched to be module_getattr_wrapper in _symbolic_trace.py.
                # which then calls into _ModuleStackTracer.getattr
                attr_val = super().__getattr__(name)  # type: ignore[misc]
                if not isinstance(attr_val, Module):
                    return attr_val

                return AttrProxy(attr_val, tracer.proxy_paths[self] + "." + name)

            def get_base(self) -> Module:
                return tracer.proxy_modules[self]

            def __getitem__(self, idx: int | slice) -> AttrProxy:
                if isinstance(idx, slice):
                    if isinstance(self, torch.nn.Sequential):
                        # Copied from nn/modules/container.py
                        res = torch.nn.Sequential(
                            OrderedDict(list(self._modules.items())[idx])
                        )

                        return AttrProxy(res, f"{tracer.proxy_paths[self]}.{idx}")
                    elif isinstance(self, torch.nn.ModuleList):
                        # Copied from nn/modules/container.py
                        res = torch.nn.ModuleList(list(self._modules.values())[idx])

                        return AttrProxy(res, f"{tracer.proxy_paths[self]}.{idx}")

                return super().__getitem__(idx)  # type: ignore[misc]

            @property
            def _modules(self) -> dict[str, AttrProxy]:
                if "_modules" not in self.__dict__:
                    raise AssertionError("_modules not in __dict__")
                submodules = self.__dict__["_modules"]
                if not isinstance(submodules, dict):
                    raise AssertionError(f"Expected dict, got {type(submodules)}")
                # pyrefly: ignore [bad-return]
                return {
                    key: (
                        AttrProxy(value, tracer.proxy_paths[self] + "." + str(key))  # type: ignore[misc]
                        if value is not None
                        else value
                    )
                    for key, value in submodules.items()
                }

        self.proxy_type = AttrProxy

    def path_of_module(self, mod: Module) -> str:
        """
        Use tracked access path during tracing instead of the default BFS behavior.
        Still use all the possible module paths to verify the result.
        """
        if mod is self.scope_root:
            return ""

        if isinstance(mod, _AttrProxy):
            return self.proxy_paths[mod]

        try:
            return Tracer.path_of_module(self, mod)
        except NameError as e:
            raise _ModuleNotInstalledAsSubmoduleError from e

    def getattr(
        self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]
    ) -> object:
        if (
            not isinstance(attr_val, Module)
            or isinstance(attr_val, fx.GraphModule)
            or not self.enable_attr_proxy
        ):
            return super().getattr(attr, attr_val, parameter_proxy_cache)
        if isinstance(attr_val, _AttrProxy):
            return attr_val

        # See NOTE [caching AttrProxy].
        if attr_val not in self.attr_proxy_map:
            self.attr_proxy_map[attr_val] = self.proxy_type(attr_val, attr)
        else:
            self.attr_proxy_map[attr_val].reset_proxy_mapping(attr_val, attr)
        return self.attr_proxy_map[attr_val]

    def trace(  # type: ignore[override]
        self, root: Module | Callable[..., Any], concrete_args: dict[str, object] | None
    ) -> fx.Graph:
        res = super().trace(root, concrete_args)

        # NOTE [export non-strict fake tensor leak detection]
        # In non-strict export, we don't have dynamo's side effect
        # tracking logic which makes some cases hard to detect.
        # In general, our detecting strategy is:
        #  (1) We instrument fake tensor creation to log all the fake tensors created during export.
        #  (2) We dump the proxy to fake tensor map from make_fx tracer (_FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT))
        #  (3) Filter out fake tensors that are logged during (1):
        #      (1) Associated with TrackedFake (input tracking thing in symbolic_shapes)
        #      (2) Associated with gm.meta
        #  (4) Do ID match with the proxies

        global _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT
        _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT.clear()

        for key, val in self.tensor_tracker.items():
            _FAKE_TENSOR_ID_TO_PROXY_MAP_FOR_EXPORT[id(key)] = val.proxy.node

        # Since we are making _AttrProxy mimic the original
        # submodule, when someone registers a module directly
        # to the tracer while tracing, the proxy object gets registered
        # first. So we need to replace the proxy modules with the real ones
        # This can happen during HOO tracing
        proxy_module_names_to_be_replaced: list[tuple[str, _AttrProxy]] = []
        for name, module in self.root.named_modules():
            if module in self.proxy_modules:
                proxy_module_names_to_be_replaced.append((name, module))

        def _delete_proxy_attr(obj: Module, target: str) -> bool:
            # Copied from fx/graph_module.py
            # Customized it for proxy type
            atoms = target.split(".")
            path, target_submod = atoms[:-1], atoms[-1]
            if not isinstance(obj, Module):
                raise AssertionError(f"Expected Module, got {type(obj)}")
            mod = obj

            # Get the parent module
            # pyrefly: ignore [bad-assignment]
            for item in path:
                if not hasattr(mod, item):
                    return False

                mod = getattr(mod, item)

                if not isinstance(mod, (_AttrProxy, Module)):
                    return False

            if not hasattr(mod, target_submod):
                return False

            # At least the leaf module should be proxy type.
            if not isinstance(getattr(mod, target_submod), _AttrProxy):
                return False

            delattr(mod, target_submod)
            return True

        for proxy_module_name, proxy_module in proxy_module_names_to_be_replaced:
            _delete_proxy_attr(self.root, proxy_module_name)
            actual_module = self.proxy_modules[proxy_module]
            _assign_attr(actual_module, self.root, proxy_module_name)

        return res

    def call_module(
        self,
        m: Module,
        forward: Callable[..., Any],
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        """PythonKeyTracer overrides call_module to avoid the scope handling,
        but we actually want it.
        """
        from torch._dynamo import OptimizedModule

        # FIXME (tmanlaibaatar)
        # When we call torch.compile inside HOO, we will end up
        # invoking a module that is not registered on the root. For
        # now, we just inline them. But once we start supporting
        # mark_strict in export, we do need to properly handle this.
        # Right now, it doesn't matter because current non-strict
        # use cases don't need to work with HOO.
        if isinstance(m, (OptimizedModule, GraphModule)):
            return forward(*args, **kwargs)

        try:
            return Tracer.call_module(self, m, forward, args, kwargs)
        except _ModuleNotInstalledAsSubmoduleError:
            log.debug(
                "Unable to find the path of the module %s. "
                "This might be because the module was not properly registered "
                "as a submodule, which is not good practice. We will trace "
                "through the module without recording stack information.",
                m,
            )
            return forward(*args, **kwargs)

    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        return False

    def create_node(self, *args: object, **kwargs: object) -> fx.node.Node:
        """
        Create node and add on metadata.
        Add nn_module_stack here instead of TracerBase,
        since calls to make_fx() might not want to record module stack metadata.
        Add torch_fn by looking at torch_fn_metadata and torch_fn_counts.
        Add stack_trace by filtering out forward() stack frames.
        """
        node = super().create_node(*args, **kwargs)  # type: ignore[arg-type]

        # nn_module_stack
        if node.op not in ["placeholder", "output"]:
            if node.meta.get("nn_module_stack") is None:
                node.meta["nn_module_stack"] = self.module_stack.copy()
            # convert nn_module_stack from Dict[key, (FQN, class)] -> Dict[str, Tuple[str, str]]
            for key, (fqn, mod_cls) in node.meta["nn_module_stack"].items():
                if isinstance(mod_cls, type):
                    node.meta["nn_module_stack"][key] = (
                        fqn,
                        mod_cls.__module__ + "." + mod_cls.__qualname__,
                    )

        # torch_fn
        if (
            node.op == "call_function"
            and (torch_fn := self.torch_fn_metadata) is not None
            and "torch_fn" not in node.meta
        ):
            node.meta["torch_fn"] = (
                # pyrefly: ignore[missing-attribute,bad-index]
                f"{torch_fn.__name__}_{self.torch_fn_counts[torch_fn]}",
                # pyrefly: ignore[missing-attribute]
                f"{torch_fn.__class__.__name__}.{torch_fn.__name__}",
            )

        return node


class _MakefxTracer:
    def __init__(
        self,
        decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None,
        tracing_mode: _TracingMode,
        _allow_non_fake_inputs: bool,
        pre_dispatch: bool,
        record_module_stack: bool,
        _allow_fake_constant: bool,
        _error_on_data_dependent_ops: bool,
        record_stack_traces: bool = False,
        parent_tracer: _MakefxTracer | None = None,
        proxy_module_inputs: bool = False,
        _disable_torch_fn_metadata_mode: bool = False,
        dynamic_shapes: ShapesSpec | None = None,
    ) -> None:
        # Configurations that are used to initialize the context managers and their states.
        # Should not modify them during tracing.
        self.decomposition_table: dict[OpOverload, Callable[..., Any]] = dict(
            decomposition_table or {}
        )
        self.decomposition_table.setdefault(
            torch.ops.aten.sym_numel.default, torch._decomp.decompositions.sym_numel
        )
        # Only inject the default detach decomp when the caller passed no table
        # at all. An explicit table, even empty, opts out and preserves exact
        # detach semantics for pre-autograd export and compile paths.
        if decomposition_table is None and not pre_dispatch:
            self.decomposition_table.setdefault(
                torch.ops.aten.detach.default,
                torch._decomp.decompositions.nop_decomposition,
            )
        self.tracing_mode: _TracingMode = tracing_mode
        self._allow_non_fake_inputs: bool = _allow_non_fake_inputs
        self.pre_dispatch: bool = pre_dispatch
        self.record_module_stack: bool = record_module_stack
        self._allow_fake_constant: bool = _allow_fake_constant
        self._error_on_data_dependent_ops: bool = _error_on_data_dependent_ops

        # All context managers and their states should be initialized before tracing based on the inputs
        # and configurations. After tracing, their states should be cleaned except for shape_env.
        # Remember to specify how to initialize it from user inputs and from parent tracer whenever
        # adding new modes in _MakefxTracer.
        self.fake_tensor_mode: FakeTensorMode | None = None
        self.proxy_mode: nullcontext[None] | ProxyTorchDispatchMode = nullcontext()
        self.proxy_function_mode: nullcontext[None] | PreDispatchTorchFunctionMode = (
            nullcontext()
        )
        self.fx_tracer: PythonKeyTracer | None = None
        self.python_dispatcher_mode: nullcontext[None] | Any = nullcontext()
        self.torch_fn_metadata_mode: nullcontext[None] | TorchFunctionMetadataMode = (
            nullcontext()
        )
        self.record_stack_traces = record_stack_traces
        self.parent_tracer: _MakefxTracer | None = parent_tracer
        self.proxy_module_inputs = proxy_module_inputs
        self._disable_torch_fn_metadata_mode = _disable_torch_fn_metadata_mode
        # Declares which inputs are dynamic. See
        # ``torch.fx.experimental.dynamic_spec`` for the spec API.
        #
        # Only set on the top-level tracer; sub-tracers default to None.
        # Higher-order ops (``cond``, ``while_loop``, ``scan``, ...) spawn
        # sub-tracers via ``_make_sub_tracer`` to produce a separate
        # subgraph for each branch/body.

        # ``dynamic_shapes`` is pre-normalized by ``make_fx()``; assign as-is.
        self.dynamic_shapes: ShapesSpec | None = dynamic_shapes
        # ``dynamic_shapes`` requires ``tracing_mode="fake"`` so the spec
        # is the *sole* source of dynamism:
        #   - ``"real"``: no fakification → spec would be silently
        #     ignored (footgun).
        #   - ``"symbolic"``: every dim becomes a backed symbol by
        #     default.
        if self.dynamic_shapes is not None and self.tracing_mode != "fake":
            raise ValueError(
                f"make_fx(dynamic_shapes=...) requires tracing_mode='fake' "
                f"(got tracing_mode={self.tracing_mode!r}). 'real' would "
                f"silently ignore the spec; 'symbolic' conflicts with the "
                f"spec's unbacked-by-default semantics."
            )

    def _checkpoint_modes(self) -> list[Any]:
        return [
            self.fake_tensor_mode,
            self.proxy_mode,
            self.proxy_function_mode,
            self.fx_tracer,
            self.python_dispatcher_mode,
            self.torch_fn_metadata_mode,
        ]

    def _restore_modes(
        self,
        prev_fake_tensor_mode: FakeTensorMode | None,
        prev_proxy_mode: nullcontext[None] | ProxyTorchDispatchMode,
        prev_proxy_function_mode: nullcontext[None] | PreDispatchTorchFunctionMode,
        prev_fx_tracer: PythonKeyTracer | None,
        prev_python_dispatcher_mode: nullcontext[None] | Any,
        prev_torch_fn_metadata_mode: nullcontext[None] | TorchFunctionMetadataMode,
    ) -> None:
        self.fake_tensor_mode = prev_fake_tensor_mode
        self.proxy_mode = prev_proxy_mode
        self.proxy_function_mode = prev_proxy_function_mode
        self.fx_tracer = prev_fx_tracer
        self.python_dispatcher_mode = prev_python_dispatcher_mode
        self.torch_fn_metadata_mode = prev_torch_fn_metadata_mode

    @contextmanager
    def _init_modes_from_inputs(
        self, f: Callable[..., Any], args: tuple[object, ...]
    ) -> Generator[None, None, None]:
        prev_modes = self._checkpoint_modes()
        try:
            # Avoid importing sympy at a module level
            from .symbolic_shapes import ShapeEnv

            if hasattr(f, "_orig_mod") and self.record_module_stack:
                scope_root = f._orig_mod
                # _ModuleStackTracer always try to preserve stack trace
                # in forward functions
                self.fx_tracer = _ModuleStackTracer(scope_root)
            else:
                self.fx_tracer = PythonKeyTracer()
                self.fx_tracer.record_stack_traces = (
                    self.record_stack_traces and not fx.config.do_not_emit_stack_traces
                )
                if self.fx_tracer.record_stack_traces:
                    self.fx_tracer._record_forward_stack_traces_only = True

            if self.tracing_mode == "fake":
                import torch._dynamo

                fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
                if fake_tensor_mode is None:
                    import torch._functorch.config as _config

                    with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
                        fake_tensor_mode = FakeTensorMode(
                            allow_fallback_kernels=True,
                            allow_non_fake_inputs=self._allow_non_fake_inputs,
                            shape_env=ShapeEnv(),
                            static_shapes=True,
                        )
                # dynamic_shapes wires unbacked symbols into the shape env, so
                # it requires one. The fake mode we create above always has it;
                # a detected ambient fake mode might not, and mutating a
                # borrowed mode is unsafe, so reject it with a clear error.
                if (
                    self.dynamic_shapes is not None
                    and fake_tensor_mode.shape_env is None
                ):
                    raise ValueError(
                        "make_fx(dynamic_shapes=...) requires the active "
                        "FakeTensorMode to have a shape_env, but the detected "
                        "fake mode has none."
                    )
                self.fake_tensor_mode = fake_tensor_mode
            elif self.tracing_mode == "symbolic":
                import torch._dynamo

                fake_tensor_mode = torch._dynamo.utils.detect_fake_mode(args)
                if fake_tensor_mode is None:
                    shape_env = ShapeEnv()
                    import torch._functorch.config as _config

                    with _config.patch(fake_tensor_allow_unsafe_data_ptr_access=False):
                        fake_tensor_mode = FakeTensorMode(
                            allow_fallback_kernels=False,
                            allow_non_fake_inputs=self._allow_non_fake_inputs,
                            shape_env=shape_env,
                        )
                if fake_tensor_mode.shape_env is None:
                    raise AssertionError(
                        "shape_env should be set if tracing with 'symbolic'"
                    )
                self.fake_tensor_mode = fake_tensor_mode
            else:
                if not self.tracing_mode == "real":
                    raise AssertionError(
                        f"Unexpected tracing type: {self.tracing_mode}"
                    )

            self._construct_modes_with_fx_tracer(self.fx_tracer)
            yield
        finally:
            self._restore_modes(*prev_modes)

    def _construct_modes_with_fx_tracer(self, fx_tracer: _ProxyTracer) -> None:
        self.proxy_mode = ProxyTorchDispatchMode(
            fx_tracer,
            self.tracing_mode,
            decomposition_table=self.decomposition_table,
            pre_dispatch=self.pre_dispatch,
            _allow_fake_constant=self._allow_fake_constant,
            _error_on_data_dependent_ops=self._error_on_data_dependent_ops,
        )

        if self.pre_dispatch:
            self.proxy_function_mode = PreDispatchTorchFunctionMode(fx_tracer)

        # pre-autograd tracing uses per-dispatch-key modes,
        # which requires the python dispatcher
        if self.tracing_mode == "symbolic" or self.pre_dispatch:
            self.python_dispatcher_mode = enable_python_dispatcher()

        if not self._disable_torch_fn_metadata_mode:
            self.torch_fn_metadata_mode = TorchFunctionMetadataMode(fx_tracer)
        fx_tracer.proxy_module_inputs = self.proxy_module_inputs  # type: ignore[union-attr]

    @contextmanager
    def _init_modes_from_parent(
        self, parent_tracer: _MakefxTracer
    ) -> Generator[None, None, None]:
        # By default, subtracer creates new modes based on parent tracer's config.
        # However, there are cases where we want to share the same modes with parent tracer
        # For example, fake_tensor_mode, we want the example value's fake_mode of parent graph and subgraphs to be the same.
        prev_modes = self._checkpoint_modes()
        try:
            self.fake_tensor_mode = parent_tracer.fake_tensor_mode

            def _create_sub_fx_tracer(parent_tracer: _ProxyTracer) -> PythonKeyTracer:
                if type(parent_tracer) is PythonKeyTracer:
                    return PythonKeyTracer()
                elif type(parent_tracer) is _ModuleStackTracer:
                    return _ModuleStackTracer(parent_tracer.scope_root)
                else:
                    raise RuntimeError(
                        f"Unexpected tracer type: {type(parent_tracer)}."
                    )

            if parent_tracer.fx_tracer is None:
                raise AssertionError("parent_tracer.fx_tracer should not be None")
            self.fx_tracer = _create_sub_fx_tracer(parent_tracer.fx_tracer)
            self._construct_modes_with_fx_tracer(self.fx_tracer)
            yield
        finally:
            self._restore_modes(*prev_modes)

    def _convert_args_to_fake(self, f: Callable[..., Any], args: T) -> T:
        """Fakify ``args``. If ``self.dynamic_shapes`` is set, create
        tensors with dynamic dims accordingly.
        """
        if self.tracing_mode == "real":
            return args

        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental._spec_binding import _bind_spec_to_args

        # NB: these spec-wiring helpers live in symbolic_shapes, which does
        # ``import sympy`` at module scope. Import them lazily here (once per
        # trace) — a module-level import would pull sympy into ``import torch``
        # and break test_not_import_sympy.
        from torch.fx.experimental.symbolic_shapes import (
            _finalize_spec_wiring,
            _symbolic_context_from_shapes_spec,
            _wire_spec_assumptions,
            _wire_spec_slot,
            _wire_tensor_spec_dims,
        )

        def fakify_leaf(x: object, source: Source, leaf_spec: LeafSpec) -> object:
            """Fakify a single flat input leaf (closes over the helpers above)."""
            if self.fake_tensor_mode is None:
                raise AssertionError("fake_tensor_mode should not be None")

            if isinstance(x, Tensor):
                if leaf_spec is None:
                    return self.fake_tensor_mode.from_tensor(x, source=source)
                # leaf_spec is guaranteed to be a TensorSpec by _walk_spec.
                tensor_spec = cast(TensorSpec, leaf_spec)
                ctx = _symbolic_context_from_shapes_spec(
                    x, source, tensor_spec, None, {}
                )
                fake_x = self.fake_tensor_mode.from_tensor(
                    x, static_shapes=False, source=source, symbolic_context=ctx
                )
                _wire_tensor_spec_dims(tensor_spec, fake_x)
                return fake_x

            # NB: don't match on bools.
            if type(x) is int:
                if isinstance(leaf_spec, (IntVar, SymInt)):
                    shape_env = self.fake_tensor_mode.shape_env
                    if shape_env is None:
                        raise AssertionError("shape_env should be set for spec inputs")
                    sym_node = shape_env.create_unbacked_symint(source=source)
                    _wire_spec_slot(leaf_spec, sym_node)
                    return sym_node
                if self.tracing_mode == "symbolic":
                    shape_env = self.fake_tensor_mode.shape_env
                    if shape_env is None:
                        raise AssertionError(
                            "shape_env should be set if tracing with 'symbolic'"
                        )
                    return shape_env.create_symintnode(
                        shape_env.create_symbol(x, source, positive=None),
                        hint=x,
                        source=source,
                    )
                # Otherwise: an int not declared in the spec stays static.

            if isinstance(x, torch.ScriptObject) or is_custom_class_obj(x):
                if is_opaque_constant_type(
                    type(x)  # pyrefly: ignore[bad-argument-type]
                ):
                    return x
                return torch._library.fake_class_registry.maybe_to_fake_obj(
                    self.fake_tensor_mode, x
                )

            if isinstance(x, FakeScriptObject):
                raise AssertionError(
                    f"ScriptObject {x} has been fakified. Cannot wrap_fake it again."
                )
            return x

        # Pass an empty ShapesSpec when no user spec was supplied so the
        # binding path is uniform (yields all-None leaf_specs aligned to the
        # flat layout). Both fake_tensor_mode and shape_env are set for tracing.
        shape_env: ShapeEnv = self.fake_tensor_mode.shape_env  # type: ignore[assignment]
        user_spec = self.dynamic_shapes
        # Wire assumptions BEFORE processing inputs so derived / assumption
        # checks can drain as inputs bind.
        if user_spec is not None and user_spec._assumptions:
            _wire_spec_assumptions(shape_env, user_spec)

        leaf_specs, flat_args, args_pytree_spec = _bind_spec_to_args(
            f,
            args,
            {},
            user_spec or ShapesSpec(),
        )

        # ``shape_env`` is None for plain fake tracing with no symbolic shapes;
        # only the spec path allocates unbacked symbols needing the guard.
        unbacked_ctx = (
            shape_env.ignore_fresh_unbacked_symbols()
            if shape_env is not None
            else nullcontext()
        )
        with unbacked_ctx:
            fake_leaves = [
                fakify_leaf(x, ConstantSource(f"input{i}"), leaf_specs[i])
                for i, x in enumerate(flat_args)
            ]

        # Verify every spec assumption / derived check that survived input
        # processing has been emitted.
        if user_spec is not None:
            _finalize_spec_wiring(shape_env)

        # args_pytree_spec wraps (args, {}); unflatten and take the args.
        return pytree.tree_unflatten(fake_leaves, args_pytree_spec)[0]  # type: ignore[return-value]

    def _trace_inner(self, f: Callable[..., Any], *args: object) -> GraphModule:
        # TODO: We need to explicitly import torch._dynamo before calling dispatch_trace,
        # because dispatch_trace will introduce the lazy import of torch._dynamo,
        # and some contexts set before calling dispatch_trace will cause problems with the import of torch._dynamo,
        # such as some torch API(torch.ones and so on) in populate_builtin_to_tensor_fn_map() will be affected
        # by the context set before dispatch_trace.
        import torch._dynamo

        phs = pytree.tree_map(lambda _: torch.fx._symbolic_trace.PH, args)

        args = self._convert_args_to_fake(f, args)

        # FX doesn't support varargs, so we gotta fake up a wrapper
        # TODO: Would be nice to fix this at the source...
        func: Callable[..., Any] = f
        if (
            not hasattr(inspect.unwrap(f), "__code__")
            or inspect.unwrap(f).__code__.co_flags & inspect.CO_VARARGS
        ):
            func = fake_signature(f, len(phs))
        # We disable the autocast cache as the autocast cache causes type conversions on parameters to
        # check a cache, which introduces untracked tensors into the graph
        #
        # We also disable tracing by any other tensor proxy-based tracers except the current. The
        # purpose of `make_fx` is to produce graphmodules as a side effect; its internal execution is
        # thus irrelevant to any external functional trace.
        proxy_mode: ProxyTorchDispatchMode = typing.cast(
            ProxyTorchDispatchMode, self.proxy_mode
        )
        with ExitStack() as stack:
            if self.fake_tensor_mode:
                stack.enter_context(self.fake_tensor_mode)
            stack.enter_context(self.python_dispatcher_mode)
            stack.enter_context(self.proxy_function_mode)
            stack.enter_context(self.torch_fn_metadata_mode)
            stack.enter_context(proxy_mode)
            stack.enter_context(disable_autocast_cache())
            stack.enter_context(_set_make_fx_tracer(self))

            # Under compile-on-one-rank, redirect legacy in-place c10d
            # collectives to functional collectives so the ProcessGroup flows
            # into the graph as a (serializable) op argument instead of being
            # baked in as a torchbind constant by the in-place op.
            if (
                torch.compiler.config.compile_on_one_rank
                and torch.distributed.is_available()
            ):
                from torch.distributed._functional_collectives import (
                    _LegacyToFunctionalCollectiveMode,
                )

                stack.enter_context(_LegacyToFunctionalCollectiveMode())

            if self.fx_tracer is None:
                raise AssertionError("fx_tracer should not be None")
            try:
                t = dispatch_trace(
                    wrap_key(func, args, self.fx_tracer, self.pre_dispatch),
                    tracer=self.fx_tracer,
                    concrete_args=tuple(phs),
                )
            except Exception:
                trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "make_fx_fail_partial",
                        "encoding": "string",
                    },
                    payload_fn=lambda: self.fx_tracer.graph.python_code(  # type: ignore[union-attr]
                        root_module="self",
                        verbose=True,
                        include_stride=True,
                        include_device=True,
                    ).src,
                )
                raise

        if (
            # TODO: consider running this unconditionally so data-dependent ops
            # (e.g. ``.item()``, ``nonzero``) get runtime asserts in plain
            # make_fx traces too, not only when an explicit ``dynamic_shapes``
            # spec or a HOP subgraph is involved.
            (self.is_hop_subgraph_tracer() or self.dynamic_shapes is not None)
            and (fake_mode := torch._guards.detect_fake_mode(args))
            and fake_mode.shape_env is not None
        ):
            from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts

            if self.dynamic_shapes is not None:
                # Attach ``unbacked_bindings`` to input placeholders so the
                # runtime-assert pass can resolve deferred asserts keyed by
                # spec-introduced unbacked symbols.
                from torch.export._trace import _add_input_unbacked_bindings

                _add_input_unbacked_bindings(t)
            insert_deferred_runtime_asserts(t, fake_mode.shape_env, "reenter_make_fx")
            t.recompile()
        # TODO: kind of a bad way to do it, should maybe figure out a better way
        if self.tracing_mode == "symbolic":
            if self.fake_tensor_mode is None:
                raise AssertionError("fake_tensor_mode should not be None")
            t.shape_env = self.fake_tensor_mode.shape_env  # type: ignore[assignment]
        return t

    def trace(self, f: Callable[..., Any], *args: object) -> fx.GraphModule:
        with self._init_modes_from_inputs(f, args):
            return self._trace_inner(f, *args)

    def is_hop_subgraph_tracer(self) -> bool:
        return self.parent_tracer is not None

    def _make_sub_tracer(
        self,
        decomp_table: Mapping[OpOverload, Callable[..., Any]] | None = None,
        tracing_mode: _TracingMode = "real",
    ) -> _MakefxTracer:
        return _MakefxTracer(
            decomp_table if decomp_table is not None else self.decomposition_table,
            tracing_mode,
            self._allow_non_fake_inputs,
            self.pre_dispatch,
            self.record_module_stack,
            self._allow_fake_constant,
            self._error_on_data_dependent_ops,
            parent_tracer=self,
        )

    def trace_subgraph(self, f: Callable[..., Any], *args: object) -> GraphModule:
        sub_tracer = self._make_sub_tracer()
        with sub_tracer._init_modes_from_parent(self):
            return sub_tracer._trace_inner(f, *args)

    def trace_subgraph_custom_decomp(
        self,
        f: Callable[..., Any],
        decomp_table: Mapping[OpOverload, Callable[..., Any]],
        *args: object,
    ) -> GraphModule:
        if not isinstance(decomp_table, Mapping):
            raise AssertionError(f"Expected Mapping, got {type(decomp_table)}")
        sub_tracer = self._make_sub_tracer(decomp_table=decomp_table)
        with sub_tracer._init_modes_from_parent(self):
            return sub_tracer._trace_inner(f, *args)


_CURRENT_MAKE_FX_TRACER: _MakefxTracer | None = None


@contextmanager
def _set_make_fx_tracer(tracer: _MakefxTracer) -> Generator[None, None, None]:
    global _CURRENT_MAKE_FX_TRACER
    prev_tracer = _CURRENT_MAKE_FX_TRACER
    try:
        _CURRENT_MAKE_FX_TRACER = tracer
        yield
    finally:
        _CURRENT_MAKE_FX_TRACER = prev_tracer


def make_fx(
    f: Callable[..., Any],
    decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None = None,
    tracing_mode: _TracingMode = "real",
    _allow_non_fake_inputs: bool = False,
    *,
    pre_dispatch: bool = False,
    record_module_stack: bool = False,
    _allow_fake_constant: bool = False,
    _error_on_data_dependent_ops: bool = True,
    record_stack_traces: bool = False,
    proxy_module_inputs: bool = False,
    _disable_torch_fn_metadata_mode: bool = False,
    dynamic_shapes: ShapesSpec | ParamsSpec | dict[str, Any] | None = None,
) -> Callable[..., GraphModule]:
    """
    Given a function f, return a new function which when executed with valid
    arguments to f, returns an FX GraphModule representing the set of operations that
    were executed during the course of execution.

    If record_stack_traces is True, the stack trace will be preserved on node.meta["stack_trace"]

    By default, post-dispatch traces without an explicit decomposition_table
    rewrite detach to alias. Re-differentiating the returned graph does not
    preserve those detach calls as autograd gradient cuts; pass an explicit
    decomposition_table or use pre_dispatch=True if exact detach nodes are needed.

    ``tracing_mode``:
        - ``"real"``: no fakification, traces with real tensors.
        - ``"fake"``: tensors become FakeTensors. By default all dims are
          static (concrete); if ``dynamic_shapes`` is set, the spec
          controls which dims become unbacked symbolic sizes — everything
          not declared by the spec stays static.
        - ``"symbolic"``: tensors become FakeTensors and **every dim becomes a
          backed symbolic size**. NOTE: using backed symbols this way is generally
          **not recommended and not sound** — backed symbols come with assumptions
          and can be constrained by guards silently. Kept for backward compatibility.
          Prefer ``"fake"`` + ``dynamic_shapes`` to declare dynamism explicitly with
          unbacked symbols.
    """

    if tracing_mode not in ["real", "fake", "symbolic"]:
        raise AssertionError(
            f"tracing_mode must be real/fake/symbolic, got {tracing_mode}"
        )

    # If ``f`` carries an ``@dynamic_spec(...)`` decorator, the attached
    # ``ShapesSpec`` is used as ``dynamic_shapes``. Passing both raises.
    from torch.fx.experimental.dynamic_spec import _resolve_dynamic_shapes

    dynamic_shapes = _resolve_dynamic_shapes(f, dynamic_shapes)

    from torch._inductor import config

    make_fx_tracer = _MakefxTracer(
        decomposition_table,
        tracing_mode,
        _allow_non_fake_inputs,
        pre_dispatch,
        record_module_stack,
        _allow_fake_constant,
        _error_on_data_dependent_ops,
        record_stack_traces=record_stack_traces
        or config.effective_provenance_tracking_level() == 1,
        proxy_module_inputs=proxy_module_inputs,
        _disable_torch_fn_metadata_mode=_disable_torch_fn_metadata_mode,
        dynamic_shapes=_coerce_to_shapes_spec(dynamic_shapes),
    )

    @functools.wraps(f)
    def wrapped(*args: object) -> GraphModule:
        return make_fx_tracer.trace(f, *args)

    return wrapped


def get_torch_dispatch_modes() -> list[TorchDispatchMode]:
    return torch.utils._python_dispatch._get_current_dispatch_mode_stack()


# TODO: this is a legacy name, there is only ever one proxy mode as it's an
# infra mode
def get_innermost_proxy_mode() -> ProxyTorchDispatchMode | None:
    return get_proxy_mode()


def get_proxy_mode() -> ProxyTorchDispatchMode | None:
    """
    Current the currently active proxy tracing mode, or None if
    we are not currently tracing.  This includes pre-dispatch proxy
    tracing.
    """
    pre_dispatch_mode = torch._ops._get_dispatch_mode_pre_dispatch(
        torch._C._TorchDispatchModeKey.PROXY
    )
    mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.PROXY)
    if not (pre_dispatch_mode is None or mode is None):
        raise AssertionError(f"pre_dispatch_mode={pre_dispatch_mode}, mode={mode}")
    return typing.cast("ProxyTorchDispatchMode | None", pre_dispatch_mode or mode)


def handle_sym_dispatch(
    func: Callable[_P, R],
    args: _P.args,  # type: ignore[valid-type]  # not allowed to use _P.args here
    kwargs: _P.kwargs,  # type: ignore[valid-type]  # not allowed to use _P.kwargs here
) -> R:
    """
    Call into the currently active proxy tracing mode to do a
    SymInt/SymFloat/SymBool dispatch trace on a function that operates on
    these arguments.
    """
    mode = get_proxy_mode()
    if not mode:
        raise AssertionError("Expected proxy mode to be set")
    # Have to do it manually, because we're not doing the normal torch
    # dispatch machinery which disables it for us
    with disable_proxy_modes_tracing():
        # TODO: properly compute types
        types: list[type] = []
        return mode.__sym_dispatch__(func, types, args, kwargs)  # type: ignore[arg-type, return-value]


@contextmanager
def disable_proxy_modes_tracing() -> Generator[ProxyTorchDispatchMode, None, None]:
    return _disable_infra_mode(torch._C._TorchDispatchModeKey.PROXY)


def maybe_handle_decomp(
    proxy_mode: ProxyTorchDispatchMode,
    op: OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    from torch._inductor.compiler_bisector import CompilerBisector

    decomp_table = proxy_mode.decomposition_table
    if op in decomp_table:
        if CompilerBisector.disable_subsystem(
            "aot_eager_decomp_partition", "decomposition", lambda: repr(op)
        ):
            return NotImplemented

        with proxy_mode:
            proxy_mode.decomp_layers += 1
            out = decomp_table[op](*args, **kwargs)
            proxy_mode.decomp_layers -= 1
            return out

    return NotImplemented


def get_isolated_graphmodule(
    func: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    tracing_mode: _TracingMode = "real",
    decomposition_table: Mapping[OpOverload, Callable[..., Any]] | None = None,
) -> GraphModule:
    """A helper function used to get the GraphModule for the given func.

    It's expected to be used in the ProxyTensor tracing context.
    It detaches the args and kwargs from the current tracer so that the trace of
    the current graph module can be created without any side-effects.
    """
    wrapped, all_args = wrapper_and_args_for_make_fx(func, args, kwargs)

    with disable_proxy_modes_tracing():
        gm = make_fx(
            wrapped, decomposition_table=decomposition_table, tracing_mode=tracing_mode
        )(all_args)
    return gm


def _set_unbacked_bindings(out: object, out_proxy: _NestedProxys) -> None:
    """A helper function for setting up unbacked_bindings on the destination FX graph."""
    from .symbolic_shapes import compute_unbacked_bindings

    # Can't use detect_fake_mode here,
    #
    # python test/distributed/_tensor/test_dtensor_compile.py -k
    # test_tp_compile_fullgraph_is_seq_parallel_False
    #
    # will fail.  Very strange, it probably isn't right for them to be using
    # two fake modes there...
    fake_mode = typing.cast(
        "FakeTensorMode | None",
        torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE),
    )
    if fake_mode and fake_mode.shape_env:
        if symbol_to_path := compute_unbacked_bindings(fake_mode.shape_env, out):
            # `symbol_to_path` is keyed by the fresh unbacked symbol; each path
            # is relative to the *whole* output pytree (`out`). Route each
            # binding to the proxy that owns it. If `out_proxy` is already a
            # Proxy, it owns the whole output and keeps the original path.
            bindings_by_proxy: dict[Proxy, dict[sympy.Symbol, pytree.KeyPath]] = {}
            for symbol, path in symbol_to_path.items():
                proxy: _NestedProxys = out_proxy
                local_path = path
                # `out` and `out_proxy` are structurally congruent, so the
                # leading container keys index identically into both. Consume
                # them off `out_proxy` until we reach the owning Proxy leaf; the
                # remaining suffix (e.g. `.size()[0]`, or an InnerTensorKey for
                # a wrapper subclass) is the tensor-local accessor that the
                # per-node consumer resolves against that node's own value.
                while not isinstance(proxy, Proxy) and local_path:
                    try:
                        proxy = local_path[0].get(proxy)
                    except (IndexError, KeyError, AttributeError) as e:
                        raise AssertionError(
                            f"unbacked_bindings path {pytree.keystr(path)} does "
                            f"not index into the proxy pytree {out_proxy}"
                        ) from e
                    local_path = local_path[1:]

                if not isinstance(proxy, Proxy):
                    raise AssertionError(
                        f"Expected Proxy at binding path {pytree.keystr(path)}, "
                        f"got {type(proxy)}"
                    )

                bindings_by_proxy.setdefault(proxy, {})[symbol] = local_path

            # Merge rather than overwrite: multiple symbols (e.g. two unbacked
            # dims on one tensor) can route to the same node.
            for proxy, bindings in bindings_by_proxy.items():
                proxy.node.meta.setdefault("unbacked_bindings", {}).update(bindings)
<|endoftext|>import builtins
import functools
import warnings
from collections.abc import Callable
from typing import Any, TypeVar

import torch
import torch.fx
from torch.fx.node import Node
from torch.fx.proxy import Proxy


_C = TypeVar("_C", bound=Callable[..., Any])

__all__ = [
    "embedding_override",
    "functional_relu_override",
    "gen_constructor_wrapper",
    "manual_meta_overrides",
    "MetaAttribute",
    "MetaDeviceAttribute",
    "MetaProxy",
    "MetaTracer",
    "nn_layernorm_override",
    "proxys_to_metas",
    "symbolic_trace",
    "torch_abs_override",
    "torch_nn_relu_override",
    "torch_relu_override",
    "torch_where_override",
]


def embedding_override(self: torch.nn.Embedding, input: torch.Tensor) -> torch.Tensor:
    return torch.empty(*input.shape, self.weight.shape[-1], device="meta")


def nn_layernorm_override(
    self: torch.nn.LayerNorm, input: torch.Tensor
) -> torch.Tensor:
    return input


def torch_relu_override(x: torch.Tensor) -> torch.Tensor:
    return x


def torch_nn_relu_override(self: torch.nn.ReLU, x: torch.Tensor) -> torch.Tensor:
    return x


def functional_relu_override(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        raise AssertionError(
            "don't support inplace functional.relu for metatensor analysis"
        )
    return x


def torch_where_override(
    condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    # torch.where returns the broadcasted tensor of condition, x, and y,
    # so hack it by using addition
    return condition.to(device="meta") + x.to(device="meta") + y.to(device="meta")


def torch_abs_override(
    input: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    if out is not None:
        raise AssertionError("Don't support in-place abs for MetaTensor analysis")
    return input


manual_meta_overrides: dict[Callable[..., Any], Callable[..., Any]] = {
    torch.nn.Embedding: embedding_override,
    torch.nn.LayerNorm: nn_layernorm_override,
    torch.relu: torch_relu_override,
    torch.nn.functional.relu: functional_relu_override,
    torch.nn.ReLU: torch_nn_relu_override,
    torch.where: torch_where_override,
    torch.abs: torch_abs_override,
}


def gen_constructor_wrapper(
    target: _C,
) -> tuple[Callable[..., Any], _C]:
    @functools.wraps(target)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        proxy = None

        def check_has_proxy(v: Any) -> None:
            if isinstance(v, torch.fx.Proxy):
                nonlocal proxy
                proxy = v

        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        if proxy is not None:
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        else:
            return target(*args, **kwargs)

    return wrapper, target


class MetaProxy(torch.fx.Proxy):
    def install_tensor_meta(self, tensor_meta: torch.Tensor) -> None:
        self._tensor_meta = tensor_meta

    def size(self, dim: int | None = None) -> Any:
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.size(*[dim] if dim else [])
        return self.tracer.create_proxy(
            "call_method", "size", (self, dim) if dim else (self,), {}
        )

    def dim(self) -> Any:
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.dim()
        return self.tracer.create_proxy("call_method", "dim", (self,), {})

    @property
    def shape(self) -> Any:
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.shape
        return self.tracer.create_proxy(
            "call_function", builtins.getattr, (self, "shape"), {}
        )

    @property
    def dtype(self) -> Any:
        if hasattr(self, "_tensor_meta") and self._tensor_meta is not None:
            return self._tensor_meta.dtype
        return self.tracer.create_proxy(
            "call_function", builtins.getattr, (self, "dtype"), {}
        )

    @property
    def device(self) -> "MetaDeviceAttribute":
        # Hack so we can track when devices are used. During meta-tensor propagation,
        # replace these values with a constant 'meta'
        return MetaDeviceAttribute(self, "device")

    def __getattr__(self, k: str) -> Any:
        if k == "_tensor_meta":
            return self.__getattribute__(k)
        # note: not added to the graph yet, if this is a method call
        # we peephole optimize to the method invocation
        return MetaAttribute(self, k)


class MetaAttribute(MetaProxy):
    def __init__(self, root: MetaProxy, attr: str) -> None:
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node: Node | None = None

    @property
    def node(self):  # type: ignore[override]
        # the node for attributes is added lazily, since most will just be method calls
        # which do not rely on the getitem call
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}
            ).node
        return self._node

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.tracer.create_proxy(
            "call_method", self.attr, (self.root,) + args, kwargs
        )


class MetaDeviceAttribute(MetaAttribute):
    pass


def proxys_to_metas(v: Any) -> Any:
    if isinstance(v, MetaDeviceAttribute):
        return "meta"
    if isinstance(v, torch.fx.Proxy):
        if not isinstance(v, MetaProxy):
            raise AssertionError(f"Expected MetaProxy but got {type(v)}")
        if not hasattr(v, "_tensor_meta"):
            raise AssertionError("MetaProxy does not have an associated meta")
        return v._tensor_meta
    return v


class MetaTracer(torch.fx.Tracer):
    allow_insert_stateless_mods: bool = True

    _TORCH_METHODS_TO_PATCH = ["arange", "zeros", "ones", "full_like", "eye"]

    def create_proxy(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        name: str | None = None,
        type_expr: Any = None,
        proxy_factory_fn: Callable[[Node], Proxy] | None = None,
    ) -> MetaProxy:
        rv = super().create_proxy(
            kind,
            target,
            args,
            kwargs,
            name,
            type_expr,
            # pyrefly: ignore [bad-argument-type]
            proxy_factory_fn,
        )

        if kind == "placeholder" and target in self.meta_args:
            rv.install_tensor_meta(self.meta_args[target])
            return rv  # pyrefly: ignore [bad-return]

        if target in self.orig_fns:
            # NOTE: tensor constructors in PyTorch define the `device` argument as
            # *kwargs-only*. That is why this works. If you add methods to
            # _TORCH_METHODS_TO_PATCH that do not define `device` as kwarg-only,
            # this will break and you will likely see issues where we cannot infer
            # the size of the output.
            if "device" in kwargs:
                kwargs["device"] = "meta"

        try:
            args_metas = torch.fx.node.map_aggregate(args, proxys_to_metas)
            kwargs_metas = torch.fx.node.map_aggregate(kwargs, proxys_to_metas)

            if kind == "call_function":
                # pyrefly: ignore [no-matching-overload]
                meta_target = manual_meta_overrides.get(target, target)

                meta_out = meta_target(*args_metas, **kwargs_metas)
            elif kind == "call_method":
                meta_target = getattr(args_metas[0], target)  # type: ignore[index]
                meta_out = meta_target(*args_metas[1:], **kwargs_metas)  # type: ignore[index]
            elif kind == "call_module":
                if not hasattr(self, "orig_forward"):
                    raise AssertionError("orig_forward not set for call_module")
                self._disable_module_getattr = True
                try:
                    # pyrefly: ignore [bad-argument-type]
                    mod = self.root.get_submodule(target)
                    mod_type = type(mod)
                    if mod_type in manual_meta_overrides:
                        meta_out = manual_meta_overrides[mod_type](
                            mod, *args_metas, **kwargs_metas
                        )  # type: ignore[misc, arg-type]
                    else:
                        meta_out = self.orig_forward(*args_metas, **kwargs_metas)
                finally:
                    self._disable_module_getattr = False
            elif kind == "get_attr":
                self._disable_module_getattr = True
                try:
                    attr_itr = self.root
                    atoms = target.split(".")  # pyrefly: ignore [missing-attribute]
                    for atom in atoms:
                        attr_itr = getattr(attr_itr, atom)
                    if not isinstance(attr_itr, torch.Tensor):
                        raise AssertionError(f"Expected Tensor, got {type(attr_itr)}")
                    meta_out = attr_itr.to(device="meta")
                finally:
                    self._disable_module_getattr = False
            else:
                return rv  # pyrefly: ignore [bad-return]

            # TODO
            if not isinstance(rv, torch.fx.Proxy):
                raise AssertionError("Don't support composite output yet")
            rv.install_tensor_meta(meta_out)
        except Exception as e:
            warnings.warn(f"Could not compute metadata for {kind} target {target}: {e}")

        return rv  # pyrefly: ignore [bad-return]

    def getattr(
        self, attr: str, attr_val: Any, parameter_proxy_cache: dict[str, Proxy]
    ) -> Any:
        if getattr(self, "_disable_module_getattr", False):
            return attr_val
        else:
            return super().getattr(attr, attr_val, parameter_proxy_cache)

    def call_module(
        self,
        m: torch.nn.Module,
        forward: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        self.orig_forward = forward
        return super().call_module(m, forward, args, kwargs)

    def _insert_module_as_submodule(self, mod: torch.nn.Module) -> str:
        """
        Helper method which tries to insert a module that was not declared as submodule.
        """
        idx = 0
        mod_name = mod.__class__.__name__.lower()
        path = f"{mod_name}_{idx}"
        while hasattr(self.root, path):
            path = f"{mod_name}_{idx}"
            idx += 1

        self.root.add_module(path, mod)
        return path

    def path_of_module(self, mod: torch.nn.Module) -> str:
        try:
            return super().path_of_module(mod)
        except NameError:
            if (
                self.allow_insert_stateless_mods
                and len(list(mod.parameters())) == 0
                and len(list(mod.buffers())) == 0
            ):
                path = self._insert_module_as_submodule(mod)
                self.prev_module = path
                return path
            raise

    def proxy(self, node: torch.fx.Node) -> MetaProxy:
        return MetaProxy(node, self)

    def trace(self, root, meta_args: dict[str, torch.Tensor], concrete_args=None):  # type: ignore[override]
        if not isinstance(meta_args, dict):
            raise AssertionError(f"Expected dict for meta_args, got {type(meta_args)}")
        self.meta_args = meta_args

        self.patched_torch_methods = {
            target: gen_constructor_wrapper(getattr(torch, target))
            for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        try:
            graph = super().trace(root, concrete_args)
            graph._tracer_extras = {"meta_args": meta_args}
            return graph
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)


def symbolic_trace(
    root: torch.nn.Module | Callable[..., Any],
    meta_args: dict[str, torch.Tensor] | None = None,
    concrete_args: dict[str, Any] | None = None,
) -> torch.fx.GraphModule:
    tracer = MetaTracer()
    graph = tracer.trace(root, meta_args, concrete_args)  # type: ignore[arg-type]
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    gm = torch.fx.GraphModule(tracer.root, graph, name)
    return gm
<|endoftext|>import itertools
import operator

import torch
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.node import Node
from torch.fx.passes.tools_common import legalize_graph


def split_result_tensors(
    result: torch.Tensor, inputs: list[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """
    A free function for use in the merge_matmul graph transformation below that
    splits the output from a merged matmul into the individual results for each
    input tensor.

    Arguments:
        result: The merged matmul result tensor.
        inputs: The list of inputs that were merged into one for the matmul.

    Returns:
        List of matmul results for each input tensor.
    """
    # When fx tracer is running, x.shape[0] will be torch.fx.Attribute but we
    # need an int even when tracing
    if isinstance(result, torch.fx.Proxy):
        splits = [0] * len(inputs)
    else:
        splits = [x.shape[0] for x in inputs]

    return torch.split(result, splits)


def may_depend_on(a: Node, b: Node, search_depth: int = 6) -> bool:
    """
    Determine if one node depends on another in a torch.fx.Graph.

    Arguments:
        a: The node that may have a dependency on b.
        b: The node that a may have a dependency on.
        search_depth: In the case of an indirect dependency, this function
                        searches up to this many nodes away in search of a
                        data dependency. If none is found, the function
                        makes the conservative assumption that there is a
                        dependency.

    Returns:
        True if a may depend on b, False if it definitely does not.
    """
    # Equivalence is defined as dependence.
    if a == b:
        return True

    # If a has no inputs, it cannot depend on b.
    if len(a.all_input_nodes) == 0:
        return False

    # If the search depth has been exhausted and no conclusion has been
    # reached, assume that there is a data dependency.
    if search_depth == 0:
        return True

    # Recursively check all inputs of a.
    for inp in a.all_input_nodes:
        if may_depend_on(inp, b, search_depth - 1):
            return True

    return False


def are_nodes_independent(nodes: list[Node]) -> bool:
    """
    Check if all of the given nodes are pairwise-data independent.

    Arguments:
        nodes: The nodes to check for data dependencies.

    Returns:
        True if any pair in nodes has a data dependency.
    """
    # For each pair in nodes:
    for i, j in itertools.combinations(nodes, 2):
        if may_depend_on(i, j) or may_depend_on(j, i):
            return False

    return True


def merge_matmul(in_mod: torch.nn.Module) -> torch.fx.GraphModule:
    """
    A graph transformation that merges matrix multiplication operations that share the same right-hand
    side operand into one large matrix multiplication.

    ::

                   ____      _________        _________
          ----    |    |    |         |     M|  A * C  |
        M| A  |  T| B  | * K|    C    | =    |---------|
          ---- ,  |    |    |         |     T|  B * C  |
           K       ----      ---------        ---------
                    K            R                R
    """
    gm = symbolic_trace(in_mod)

    rhs_users: dict[Node, list[Node]] = {}
    lhs_users: dict[Node, list[Node]] = {}

    # Populate rhs_users and lhs_users - maps from LHS/RHS matrix multiply operands to
    # the matmul of which they are the LHS/RHS.
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is not torch.matmul:
            continue

        lhs, rhs = node.args

        # TODO: Properly handle aliasing caused by get_attr. For now,
        # use the attribute name as the operand if the node is a
        # get_attr.
        lhs = lhs.target if lhs.op == "get_attr" else lhs
        rhs = rhs.target if rhs.op == "get_attr" else rhs

        lhs_users.setdefault(lhs, []).append(node)
        rhs_users.setdefault(rhs, []).append(node)

    for rhs, mms in rhs_users.items():
        # There must be at least two matmuls for a merge to make sense.
        if len(mms) < 2:
            continue

        # All matmuls must not depend on each other directly or indirectly
        # in order for the merge to be possible.
        if not are_nodes_independent(mms):
            continue

        lhs_vals = [mm.args[0] for mm in mms]

        # Merge the matmul.
        # Collect a list of LHS operands and the single RHS operand.
        lhs = [gm.graph.get_attr(l) if isinstance(l, str) else l for l in lhs_vals]
        rhs = gm.graph.get_attr(rhs) if isinstance(rhs, str) else rhs

        # Concatenate all the LHS operands.
        merge_mm_cat = gm.graph.call_function(torch.cat, (lhs,), {})

        # Multiply the concatenated LHS operands with the one RHS. This will produce
        # the same results as all the individual matmuls involving rhs in the original graph,
        # but they will all be concatenated together.
        merge_mm = gm.graph.call_function(
            torch.matmul,
            (
                merge_mm_cat,
                rhs,
            ),
            {},
        )

        # Split the result of the merged matmul using the shapes of the LHS operands
        # to ascertain how large each chunk should be.
        merge_mm_split = gm.graph.call_function(
            split_result_tensors, (merge_mm, lhs), {}
        )
        merge_mm_res = [
            gm.graph.call_function(operator.getitem, (merge_mm_split, out), {})
            for out in range(len(lhs))
        ]

        # Replace all uses of the original, unmerged matmuls with the equivalent split chunk from the merged matmul.
        for old, new in zip(mms, merge_mm_res):
            old.replace_all_uses_with(new)
            gm.graph.erase_node(old)

        # All of the new nodes created above were inserted at the end, so we need to sort
        # the nodes topologically to make sure all definitions precede uses.
        legalize_graph(gm)

    gm.recompile()
    gm.graph.lint()
    return gm
<|endoftext|>import re
import types
from collections.abc import Callable
from typing import Any

import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module


__all__ = [
    "FoldedGraphModule",
    "get_unique_attr_name_in_module",
    "split_const_subgraphs",
]


class FoldedGraphModule(torch.fx.GraphModule):
    """
    FoldedGraphModule is a GraphModule which also contains another
    `const_subgraph_module` representing a subgraph which has all const attr
    inputs and which can be run once before running the main standard
    `graph`. The `const_output_names` are the ordered list names of attrs which
    represent what each respective output from the const_subgraph should be set
    on which attrs.
    """

    def __init__(
        self,
        root: torch.nn.Module,
        graph: torch.fx.Graph,
        const_subgraph: torch.fx.Graph | None = None,
        fx_const_folded_attrs_name: str | None = None,
        device_for_folded_attrs: str = "cuda",
    ) -> None:
        super().__init__(root, graph)
        self.const_subgraph_module = (
            None
            if const_subgraph is None
            else torch.fx.GraphModule(root, const_subgraph)
        )
        self.has_folding_been_run = False
        self.fx_const_folded_attrs_name = fx_const_folded_attrs_name
        self.device_for_folded_attrs = device_for_folded_attrs

    def __call__(self, *args: object, **kwargs: object) -> Any:
        if not self.has_folding_been_run:
            self.run_folding()
        return super().__call__(*args)

    def run_folding(self) -> None:
        # If there's no const subgraph module or attr output names to use, return
        # early as there is no const folding to perform.
        if (
            self.const_subgraph_module is None
            or self.fx_const_folded_attrs_name is None
        ):
            return

        if self.has_folding_been_run:
            raise AssertionError("Folding has already been run")
        self.has_folding_been_run = True

        # Actually run const folding subgraph. Note that single attr const fold
        # subgraphs output a single Tensor while multiple outputs are returned as
        # Tuple[Tensor,].
        folded_attrs = self.const_subgraph_module()

        def _create_param(i: torch.Tensor | int) -> torch.nn.Parameter:
            return torch.nn.Parameter(
                i.detach().clone()
                if not isinstance(i, int)
                else torch.Tensor([i]).to(device=self.device_for_folded_attrs),
                requires_grad=i.requires_grad if isinstance(i, torch.Tensor) else False,
            )

        params = (
            torch.nn.ParameterList([_create_param(i) for i in folded_attrs])
            if isinstance(folded_attrs, tuple)
            else _create_param(folded_attrs)
        )
        setattr(self, self.fx_const_folded_attrs_name, params)


def _inline_module(
    gm: torch.fx.GraphModule,
    inline_mod_name: str,
    run_dce: bool = True,
    is_impure_node: Callable[[torch.fx.Node], bool] | None = None,
) -> dict[torch.fx.Node, torch.fx.Node]:
    """
    Given `gm` and some graph module which is called with target name `inline_mod_name`,
    this helper will inline all of the nodes from that called graph module into `gm`.

    Returns a mapping from subgraph nodes to the newly created/mapped nodes in gm.
    """
    # Fetch the inner graph module that we want to inline inside `gm`.
    inline_mod = dict(gm.named_modules())[inline_mod_name]
    if not isinstance(inline_mod, torch.fx.GraphModule):
        raise AssertionError(f"Expected GraphModule, got {type(inline_mod)}")
    call_mod_node_to_replace = None
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target == inline_mod_name:
            call_mod_node_to_replace = node
            break
    if call_mod_node_to_replace is None:
        raise AssertionError(f"Could not find call_module node for {inline_mod_name}")

    # Now actually do the swap. Note that we have to keep track of new nodes that are
    # copied into `gm` -- we do this via replacement_mapping.
    call_mod_args = call_mod_node_to_replace.args
    call_mod_kwargs = call_mod_node_to_replace.kwargs

    replacement_mapping: dict[torch.fx.Node, torch.fx.Node] = {}
    ph_count = 0

    def replacement_fn(node: torch.fx.Node) -> torch.fx.Node:
        new_node = replacement_mapping[node]
        new_node.meta = node.meta.copy()
        return new_node

    for inline_node in inline_mod.graph.nodes:
        if inline_node.op == "placeholder":
            replacement_mapping[inline_node] = (
                call_mod_kwargs[inline_node.name]
                if inline_node.name in call_mod_kwargs
                else call_mod_args[ph_count]
            )

            ph_count += 1
            continue

        if inline_node.op == "output":
            outputs = inline_node.args[0]
            output_replacements = map_arg(outputs, replacement_fn)

            # If output is a tuple, we need to handle getitem users specially.
            # Capture users before replace_all_uses_with modifies them.
            getitem_users: list[torch.fx.Node] = []
            if isinstance(output_replacements, (list, tuple)):
                import operator

                getitem_users = [
                    user
                    for user in call_mod_node_to_replace.users
                    if user.op == "call_function"
                    and user.target is operator.getitem
                    and isinstance(user.args[1], int)
                ]

            call_mod_node_to_replace.replace_all_uses_with(output_replacements)

            # Inline getitem nodes that now index into the tuple literal
            for user in getitem_users:
                idx = user.args[1]
                if not isinstance(idx, int):
                    raise AssertionError(f"Expected int index, got {type(idx)}")
                user.replace_all_uses_with(output_replacements[idx])
                gm.graph.erase_node(user)
                replacement_mapping[user] = output_replacements[idx]

            continue

        with gm.graph.inserting_before(call_mod_node_to_replace):
            new_node = gm.graph.node_copy(inline_node, replacement_fn)
        replacement_mapping[inline_node] = new_node

    # Explicitly remove the module that was just inlined,
    # this module may contain impure ops so cannot be dead code eliminated,
    # this module is unneeded as it's just inlined back to main graph.
    gm.graph.erase_node(call_mod_node_to_replace)
    if run_dce:
        gm.graph.eliminate_dead_code(is_impure_node=is_impure_node)

    return replacement_mapping


def get_unique_attr_name_in_module(mod_traced: torch.fx.GraphModule, name: str) -> str:
    """
    Make sure the name is unique (in a module) and can represents an attr.
    """
    # Delete all characters that are illegal in a Python identifier.
    name = re.sub("[^0-9a-zA-Z_]+", "_", name)
    if name[0].isdigit():
        name = f"_{name}"
    # Now make sure it is in fact unique to the module by incrementing suffix value.
    while hasattr(mod_traced, name):
        match = re.match(r"(.*)_(\d+)$", name)
        if match is None:
            name = name + "_1"
        else:
            base, num = match.group(1, 2)
            name = f"{base}_{int(num) + 1}"

    return name


def split_const_subgraphs(
    module: torch.nn.Module | torch.fx.GraphModule,
    skip_folding_node_fn: Callable[[torch.fx.Node], bool] | None = None,
    device_for_folded_attrs: str = "cpu",
    is_impure_node: Callable[[torch.fx.Node], bool] | None = None,
) -> FoldedGraphModule:
    """
    Looks through `module` for any nodes that have all constant attribute inputs
    and separates them out into their own constant subgraph, and returns a
    FoldedGraphModule which runs that constant subgraph on the first run to set
    attributes on the module prior to running the non-constant portion of the
    graph.

    `skip_folding_node_fn`, if provided, may be invoked on nodes owned by nested
    `call_module` subgraphs, not just top-level nodes: a `call_module` node is
    folded atomically, so it is skipped if any node inside its subgraph is
    skipped. Predicates must therefore be node-local; one that needs to resolve a
    node's `target` to a submodule must use `node.graph.owning_module` rather than
    a captured top-level module.

    `is_impure_node`, if provided, is forwarded to `eliminate_dead_code` so DCE
    preserves nodes the caller considers impure beyond the default
    `Node.is_impure()` check (e.g. out-variant ops that write a pre-allocated
    buffer via an `out=` kwarg not declared mutable in their schema).
    """

    import sympy

    if not isinstance(module, torch.fx.GraphModule):
        mod_traced = torch.fx.symbolic_trace(module)
    else:
        mod_traced = module

    def _subgraph_has_impure_ops(module: torch.fx.GraphModule) -> bool:
        """
        Return True if a GraphModule type subgraph contains any impure op, else False.
        """
        if not isinstance(module, torch.fx.GraphModule):
            raise AssertionError(
                "caller should only pass GraphModule to subgraph_has_impure_ops check"
            )
        for node in module.graph.nodes:
            if node.op == "call_function" and node.is_impure():
                return True
            if (
                node.op == "call_module"
                # pyrefly: ignore [not-callable]
                and (submodule := module.get_submodule(node.target))
                and isinstance(submodule, torch.fx.GraphModule)
            ):
                return _subgraph_has_impure_ops(submodule)
        return False

    def _subgraph_has_skipped_node(
        module: torch.fx.GraphModule,
        skip_fn: Callable[[torch.fx.Node], bool],
    ) -> bool:
        """
        Return True if a GraphModule type subgraph contains any node that
        `skip_fn` says to skip, recursing into nested submodules.
        """
        for node in module.graph.nodes:
            if node.op in {"placeholder", "output"}:
                continue
            if skip_fn(node):
                return True
            if (
                node.op == "call_module"
                # pyrefly: ignore [not-callable]
                and (submodule := module.get_submodule(node.target))
                and isinstance(submodule, torch.fx.GraphModule)
                and _subgraph_has_skipped_node(submodule, skip_fn)
            ):
                return True
        return False

    # Build up a list of const_nodes, defined as nodes that are themselves
    # get_attrs, or have all get_attr or other constant node inputs.
    const_nodes: set[torch.fx.Node] = set()
    found_const_folding = False
    for node in mod_traced.graph.nodes:
        # Skip over placeholders/outputs because they can't be const folded and
        # we don't want to add tags to them.
        if node.op in {"placeholder", "output"}:
            continue

        # If the node itself is constant, or all of its inputs are constant,
        # then tag it as constant.
        if node.op == "get_attr":
            # Only fold get_attr nodes that resolve to tensors (parameters,
            # buffers). Module references and unresolvable paths must be
            # excluded -- split_module handles get_attr nodes outside the
            # normal partition callback, so tagging alone is not enough.
            try:
                parts = node.target.split(".")
                obj = mod_traced
                for part in parts:
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            if isinstance(obj, (torch.nn.Module, types.ModuleType)):
                continue
        elif not set(node.all_input_nodes).issubset(const_nodes):
            continue

        # If provided skip folding function says to skip, then skip. Also skip a
        # call_module node whose subgraph contains a node that should be skipped,
        # since a call_module node can only be folded as a whole.
        if skip_folding_node_fn is not None:
            if skip_folding_node_fn(node):
                continue
            if (
                node.op == "call_module"
                # pyrefly: ignore [not-callable]
                and (target_mod := mod_traced.get_submodule(node.target))
                and isinstance(target_mod, torch.fx.GraphModule)
                and _subgraph_has_skipped_node(target_mod, skip_folding_node_fn)
            ):
                continue

        # Skip folding side-effectful functions
        if node.is_impure():
            continue

        # Skip folding nodes that have symbolic fill_value
        if isinstance(node.kwargs.get("fill_value", None), sympy.Expr):
            continue

        # Skip folding submodules that have impure ops
        if (
            node.op == "call_module"
            # pyrefly: ignore [not-callable]
            and (target_mod := mod_traced.get_submodule(node.target))
            and isinstance(target_mod, torch.fx.GraphModule)
            and _subgraph_has_impure_ops(target_mod)
        ):
            continue

        # Must be a constant foldable node at this point.
        const_nodes.add(node)
        if node.op != "get_attr":
            found_const_folding = True

    # If we did not find any const folding then return early without a const fold subgraph.
    if not found_const_folding:
        return FoldedGraphModule(mod_traced, mod_traced.graph)

    # Partition the module into two: submod_0 for constant folding subgraph, and
    # submod_1 for the rest.
    def mod_partition(node: torch.fx.Node) -> int:
        return 0 if node in const_nodes else 1

    split = split_module(mod_traced, module, mod_partition)

