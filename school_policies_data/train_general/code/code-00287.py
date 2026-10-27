        if visited_fns is None:
            visited_fns = set()

        fn_id = id(fn)
        if fn_id in visited_fns:
            return []
        if depth > MAX_RECURSION_DEPTH:
            logger.debug(
                "reached max recursion depth (%s) in find_triton_kernels",
                MAX_RECURSION_DEPTH,
            )
            return []

        visited_fns.add(fn_id)

        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError):
            return []  # Source code not available

        from torch._inductor.utils import IndentedBuffer

        buffer = IndentedBuffer()
        buffer.splice(source, strip=True)
        tree = ast.parse(buffer.getrawvalue())

        # Visitor to collect function calls, assignments, and triton kernels
        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.triton_kernels: list[Any] = []
                # track local variable assignments: var_name -> list of RHS expressions
                self.assignments: dict[str, list[ast.expr]] = {}
                # track function calls
                self.called_functions: list[str] = []
                # track return statement expressions
                self.return_exprs: list[ast.expr] = []

            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.assignments.setdefault(target.id, []).append(node.value)
                self.generic_visit(node)

            def visit_Return(self, node: ast.Return) -> None:
                if node.value is not None:
                    self.return_exprs.append(node.value)
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                triton_func_names = ("capture_triton", "wrap_triton")
                if isinstance(node.func, ast.Attribute):
                    attr = node.func
                    if isinstance(attr.value, ast.Attribute):
                        if (
                            isinstance(attr.value.value, ast.Name)
                            and attr.value.value.id == "torch"
                            and attr.value.attr == "_library"
                            and attr.attr in triton_func_names
                        ):
                            if node.args and isinstance(node.args[0], ast.Name):
                                self.triton_kernels.append(node.args[0].id)
                        elif (
                            isinstance(attr.value.value, ast.Attribute)
                            and isinstance(attr.value.value.value, ast.Name)
                            and attr.value.value.value.id == "torch"
                            and attr.value.value.attr == "ops"
                        ):
                            self.called_functions.append(
                                f"{attr.value.attr}::{attr.attr}"
                            )
                # Catch capture_triton, wrap_triton that's been
                # imported directly
                elif isinstance(node.func, ast.Name):
                    if node.func.id in triton_func_names:
                        if node.args and isinstance(node.args[0], ast.Name):
                            self.triton_kernels.append(node.args[0].id)
                    else:
                        # track regular function calls for recursive analysis
                        self.called_functions.append(node.func.id)

                self.generic_visit(node)

        collector = Visitor()
        collector.visit(tree)

        def extract_names_from_expr(expr: ast.expr) -> list[str]:
            """Extract all Name references from an AST expression."""
            names: list[str] = []

            class NameExtractor(ast.NodeVisitor):
                def visit_Name(self, node: ast.Name) -> None:
                    names.append(node.id)

                def visit_Call(self, node: ast.Call) -> None:
                    # for function calls, visit the function and all args
                    self.generic_visit(node)

            NameExtractor().visit(expr)
            return names

        def resolve_to_kernel(obj: object) -> object | None:
            """Check if obj is a triton kernel or wrapper and return the kernel."""
            if isinstance(obj, (JITFunction, Autotuner)):
                return obj
            # handle wrappers that have a .fn attribute pointing to JITFunction
            if callable(obj) and hasattr(obj, "fn"):
                inner = obj.fn
                if isinstance(inner, JITFunction):
                    return inner
            return None

        def build_namespace(func_obj: object) -> dict[str, Any]:
            """Build a combined namespace from a function's globals and closures."""
            # unwrap decorated fns (e.g., @lru_cache)
            if callable(func_obj):
                try:
                    func_obj = inspect.unwrap(func_obj)
                except ValueError:
                    pass
            if not callable(func_obj) or not hasattr(func_obj, "__code__"):
                return {}
            func_closure_vars = inspect.getclosurevars(func_obj)
            namespace: dict[str, Any] = {}
            namespace.update(func_closure_vars.builtins)
            namespace.update(func_closure_vars.globals)
            namespace.update(func_closure_vars.nonlocals)
            if hasattr(func_obj, "__globals__"):
                namespace.update(func_obj.__globals__)
            return namespace

        all_names = build_namespace(fn)

        def resolve_names_to_kernels(
            names: list[str],
            namespace: dict[str, Any],
            assignments: dict[str, list[ast.expr]] | None = None,
            visited: set[str] | None = None,
        ) -> list[object]:
            """
            Resolve a list of names to triton kernels using the given namespace.
            """
            if visited is None:
                visited = set()

            results: list[object] = []
            for name in names:
                if name in visited:
                    continue
                visited.add(name)

                if name in namespace:
                    obj = namespace[name]
                    kernel = resolve_to_kernel(obj)
                    if kernel is not None:
                        results.append(kernel)
                        continue
                    # recurse into callable objects (factory fn's),
                    # unwrapping decorators if applicable
                    if callable(obj):
                        try:
                            unwrapped = inspect.unwrap(obj)
                        except ValueError:
                            unwrapped = obj
                        if hasattr(unwrapped, "__code__"):
                            nested = find_triton_kernels(
                                unwrapped, visited_fns, depth + 1
                            )
                            if nested:
                                results.extend(nested)
                                continue
                    logger.debug("failed to resolve %s to a triton kernel", name)
                elif assignments is not None and name in assignments:
                    # trace through local assignments
                    for rhs_expr in assignments[name]:
                        referenced = extract_names_from_expr(rhs_expr)
                        traced = resolve_names_to_kernels(
                            referenced, namespace, assignments, visited
                        )
                        results.extend(traced)
                else:
                    logger.debug("%s not found in namespace or assignments", name)

            return results

        # resolve kernel names, tracing through local variables if needed
        resolved: list[object] = []
        seen_ids: set[int] = set()

        names_to_resolve: list[str] = list(collector.triton_kernels)
        for expr in collector.return_exprs:
            names_to_resolve.extend(extract_names_from_expr(expr))

        for name in names_to_resolve:
            traced_objects = resolve_names_to_kernels(
                [name], all_names, collector.assignments
            )
            for obj in traced_objects:
                obj_id = id(obj)
                if obj_id not in seen_ids:
                    seen_ids.add(obj_id)
                    resolved.append(obj)

        for func_name in collector.called_functions:
            func_obj = all_names.get(func_name)

            if func_obj is None:
                from torch._library.custom_ops import OPDEFS

                if func_name in OPDEFS:
                    func_obj = OPDEFS[func_name]._abstract_fn

            # skip if not a callable or if it's a triton kernel itself
            if func_obj is None or not callable(func_obj):
                continue

            # skip built-in functions and C extensions (they can't contain triton kernels)
            if not hasattr(func_obj, "__code__"):
                continue

            try:
                nested_kernels = find_triton_kernels(func_obj, visited_fns, depth + 1)
                for kernel in nested_kernels:
                    kernel_id = id(kernel)
                    if kernel_id not in seen_ids:
                        seen_ids.add(kernel_id)
                        resolved.append(kernel)
            except Exception:
                logger.debug(
                    "failed to analyze called function %s", func_name, exc_info=True
                )

        return resolved

    return find_triton_kernels(fn)


@exposed_in("torch.library")
def triton_op(
    name: str,
    fn: Callable | None = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    schema: str | None = None,
) -> Callable:
    """Create a custom operator whose implementation is backed by 1+ triton kernels.

    This is a more structured way of using triton kernels with PyTorch.
    Prefer using triton kernels with no ``torch.library`` custom operator wrappers
    (like :func:`torch.library.custom_op`, :func:`torch.library.triton_op`) because
    that is simpler;
    only use :func:`torch.library.custom_op`/:func:`torch.library.triton_op` if you
    want to create an operator that behaves like PyTorch built-in operators.
    For example, you may use a ``torch.library`` wrapper API to define the
    behavior of the triton kernel when passed a tensor subclass or under
    a TorchDispatchMode.

    Use :func:`torch.library.triton_op` instead of :func:`torch.library.custom_op`
    when the implementation
    consists of 1+ triton kernels. :func:`torch.library.custom_op` treats
    custom operators as opaque (:func:`torch.compile` and
    :func:`torch.export.export` will never trace into them), but ``triton_op``
    makes the implementation visible to these subsystems, allowing them
    to optimize the triton kernel(s).

    Note that ``fn`` must only consist of calls to PyTorch-understood
    operators and triton kernels. Any triton kernels called inside ``fn``
    must be wrapped in a call to :func:`torch.library.wrap_triton`.

    Args:
        name (str): A name for the custom op that looks like "{namespace}::{name}",
            e.g. "mylib::my_linear". The name is used as the op's stable identifier
            in PyTorch subsystems (e.g. torch.export, FX graphs).
            To avoid name collisions, please use your project name as the namespace;
            e.g. all custom ops in pytorch/fbgemm use "fbgemm" as the namespace.
        mutates_args (Iterable[str] or "unknown"): The names of args that the function mutates.
            This MUST be accurate, otherwise, the behavior is undefined. If "unknown",
            it pessimistically assumes that all inputs to the operator are being mutated.
        schema (str | None): A schema string for the operator. If None
            (recommended) we'll infer a schema for the operator from its type
            annotations. We recommend letting us infer a schema unless you
            have a specific reason not to.
            Example: "(Tensor x, int y) -> (Tensor, Tensor)".

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> from torch.library import triton_op, wrap_triton
        >>>
        >>> import triton
        >>> from triton import language as tl
        >>>
        >>> @triton.jit
        >>> def add_kernel(
        >>>     in_ptr0,
        >>>     in_ptr1,
        >>>     out_ptr,
        >>>     n_elements,
        >>>     BLOCK_SIZE: "tl.constexpr",
        >>> ):
        >>>     pid = tl.program_id(axis=0)
        >>>     block_start = pid * BLOCK_SIZE
        >>>     offsets = block_start + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     x = tl.load(in_ptr0 + offsets, mask=mask)
        >>>     y = tl.load(in_ptr1 + offsets, mask=mask)
        >>>     output = x + y
        >>>     tl.store(out_ptr + offsets, output, mask=mask)
        >>>
        >>> @triton_op("mylib::add", mutates_args={})
        >>> def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        >>>     output = torch.empty_like(x)
        >>>     n_elements = output.numel()
        >>>
        >>>     def grid(meta):
        >>>         return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        >>>
        >>>     # NB: we need to wrap the triton kernel in a call to wrap_triton
        >>>     wrap_triton(add_kernel)[grid](x, y, output, n_elements, 16)
        >>>     return output
        >>>
        >>> @torch.compile
        >>> def f(x, y):
        >>>     return add(x, y)
        >>>
        >>> x = torch.randn(3, device="cuda")
        >>> y = torch.randn(3, device="cuda")
        >>>
        >>> z = f(x, y)
        >>> assert torch.allclose(z, x + y)

    """

    def dec(fn: Callable[..., object]) -> CustomOpDef:
        def backend_fn(*args, **kwargs):  # type: ignore[no-untyped-def]
            # Optimization: we're passing regular Tensors into the triton kernel, so
            # no need to go through HOP dispatch
            with set_wrap_triton_enabled(False):
                return fn(*args, **kwargs)

        result = custom_op(
            name,
            backend_fn,
            mutates_args=mutates_args,
            schema=(
                schema
                if schema is not None
                else infer_schema(fn, mutates_args=mutates_args)
            ),
        )
        from .._subclasses.functional_tensor import FunctionalTensorMode

        # We require that the user pass us a function that is make_fx traceable,
        # so we can just register it as the Fake/meta kernel.
        result.register_fake(fn)

        # We decompose the operator when FunctionalTensorMode is active.
        # The goal is to decompose the operator in AOTDispatcher.
        # - With torch.compile, this means that the backend (usually Inductor)
        #   can see a call to the triton kernel(s) and so it can directly optimize
        #   them by inlining them into the lowering process.
        def functional_decomp(  # type: ignore[no-untyped-def]
            mode, op, types, args, kwargs
        ):
            # NOTE [Export custom triton op]
            # For torch.export (strict and non-strict), we don't do functional decomposition.
            # Instead, we preserve the custom triton ops as custom ops. This is because we want
            # the exported program to be high-level and serializable. If we decompose
            # the custom op to a functional hop and make it a node in exported program,
            # we need to figure out ways of serializing the hop and its arguments, which can be triton.jited
            # functions and triton dtypes. This is undesirable because:
            # - it can be tedious to maintain a layer that serializes the jited function (e.g. with a string) and dtypes.
            # - exported program will contain the implementation detail (e.g. triton source code) for a specific
            #   backend (GPU), which is probably at a wrong level of abstraction.
            # - changes to triton or the serialization logic for triton arguments can be BC breaking
            #
            # In the short term, we expect users to have a separate aot_compile stage that compiles the exported program
            # into a Cubin file on the same machine that users call export, which does autotuning and removes triton
            # dependency and serve the model with Cubin. This guarantees that triton changes won't break BC.
            # In the long term, we may export multiple cubins for the triton op directly
            from torch.export._trace import custom_triton_ops_decomposition_disabled

            if custom_triton_ops_decomposition_disabled():
                return mode.__torch_dispatch__(op, types, args, kwargs)
            else:
                # TODO: https://github.com/pytorch/pytorch/issues/160333
                # We should deduplicate the unrecognized_types logic.
                import torch._subclasses

                unrecognized_types = [
                    t
                    for t in types
                    if not issubclass(t, torch._subclasses.FakeTensor)
                    and t
                    not in [
                        torch.Tensor,
                        torch._subclasses.functional_tensor.FunctionalTensor,
                    ]
                ]

                if unrecognized_types:
                    return NotImplemented
                with mode:
                    return fn(*args, **kwargs)

        triton_kernels = get_inner_triton_kernels(fn)
        triton_ops_to_kernels[name] = triton_kernels
        result.register_torch_dispatch(FunctionalTensorMode, functional_decomp)
        return result

    if fn is None:
        return dec
    else:
        return dec(fn)


wrap_triton_enabled = threading.local()
wrap_triton_enabled_default = True


@contextlib.contextmanager
def set_wrap_triton_enabled(enabled: bool) -> Generator[None, None, None]:
    """If triton kernels annotated with @wrap_triton should dispatch via HOP
    or go straight to the triton kernel execution.

    We have this switch because eager-mode performance of HOP dispatch is slow
    enough to matter (~1ms) and we know that wrap_triton isn't necessary in
    some situations (eager-mode with regular Tensors)
    """
    try:
        prev = is_wrap_triton_enabled()
        wrap_triton_enabled.value = enabled
        yield
    finally:
        wrap_triton_enabled.value = prev


def is_wrap_triton_enabled() -> bool:
    return getattr(wrap_triton_enabled, "value", wrap_triton_enabled_default)


def _is_interpreted_triton_kernel(triton_kernel: Callable) -> bool:
    try:
        from triton.runtime.interpreter import InterpretedFunction
    except ImportError:
        return False

    return isinstance(triton_kernel, InterpretedFunction)


def _has_interpreted_triton_autotuner(triton_kernel: Callable) -> bool:
    try:
        from triton.runtime.autotuner import Autotuner, Heuristics
        from triton.runtime.interpreter import InterpretedFunction
    except ImportError:
        return False

    seen: set[int] = set()
    kernel: Any = triton_kernel
    while isinstance(kernel, (Autotuner, Heuristics)):
        kernel_id = id(kernel)
        if kernel_id in seen:
            return False
        seen.add(kernel_id)
        kernel = kernel.fn
    return isinstance(kernel, InterpretedFunction)


def capture_triton(triton_kernel: Callable, /) -> Any:
    """This API has been renamed to wrap_triton"""
    return wrap_triton(triton_kernel)


@exposed_in("torch.library")
def wrap_triton(triton_kernel: Callable, /) -> Any:
    """Allows capture of a triton kernel into a graph via make_fx or
    non-strict ``torch.export``.

    These technologies perform Dispatcher-based tracing (via
    ``__torch_dispatch__``) and cannot see calls to raw triton kernels.
    The ``wrap_triton`` API wraps a triton kernel into a callable that
    can actually be traced into a graph.

    Please use this API together with :func:`torch.library.triton_op`.

    Examples:

        >>> # xdoctest: +SKIP
        >>> import torch
        >>> import triton
        >>> from triton import language as tl
        >>> from torch.fx.experimental.proxy_tensor import make_fx
        >>> from torch.library import wrap_triton
        >>>
        >>> @triton.jit
        >>> def add_kernel(
        >>>     in_ptr0,
        >>>     in_ptr1,
        >>>     out_ptr,
        >>>     n_elements,
        >>>     BLOCK_SIZE: "tl.constexpr",
        >>> ):
        >>>     pid = tl.program_id(axis=0)
        >>>     block_start = pid * BLOCK_SIZE
        >>>     offsets = block_start + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     x = tl.load(in_ptr0 + offsets, mask=mask)
        >>>     y = tl.load(in_ptr1 + offsets, mask=mask)
        >>>     output = x + y
        >>>     tl.store(out_ptr + offsets, output, mask=mask)
        >>>
        >>> def add(x, y):
        >>>     output = torch.empty_like(x)
        >>>     n_elements = output.numel()
        >>>
        >>>     def grid_fn(meta):
        >>>         return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        >>>
        >>>     wrap_triton(add_kernel)[grid_fn](x, y, output, n_elements, 16)
        >>>     return output
        >>>
        >>> x = torch.randn(3, device="cuda")
        >>> y = torch.randn(3, device="cuda")
        >>> gm = make_fx(add)(x, y)
        >>> print(gm.code)
        >>> # def forward(self, x_1, y_1):
        >>> #     empty_like = torch.ops.aten.empty_like.default(x_1, pin_memory = False)
        >>> #     triton_kernel_wrapper_mutation_proxy = triton_kernel_wrapper_mutation(
        >>> #         kernel_idx = 0, constant_args_idx = 0,
        >>> #         grid = [(1, 1, 1)], kwargs = {
        >>> #             'in_ptr0': x_1, 'in_ptr1': y_1, 'out_ptr': empty_like,
        >>> #             'n_elements': 3, 'BLOCK_SIZE': 16
        >>> #         })
        >>> #     return empty_like

    """
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    from torch._higher_order_ops.triton_kernel_wrap import TraceableTritonKernelWrapper

    if isinstance(triton_kernel, JITFunction):
        if not is_wrap_triton_enabled():
            return triton_kernel
        return TraceableTritonKernelWrapper(triton_kernel, None, None)

    if isinstance(triton_kernel, Autotuner):
        # TRITON_INTERPRET=1 can make @triton.autotune wrap an
        # InterpretedFunction. It is eager-launchable but lacks the JIT
        # metadata the traceable wrapper needs.
        if _has_interpreted_triton_autotuner(triton_kernel):
            return triton_kernel
        if not is_wrap_triton_enabled():
            return triton_kernel
        return TraceableTritonKernelWrapper(triton_kernel, None, None)

    # TRITON_INTERPRET=1 makes @triton.jit return InterpretedFunction.
    if _is_interpreted_triton_kernel(triton_kernel):
        return triton_kernel

    raise RuntimeError(
        "wrap_triton only works on functions annotated with triton.jit or triton.autotune"
    )
<|endoftext|>from __future__ import annotations

import logging

import torch


log = logging.getLogger(__name__)


# Manual registry for ops whose out variant is not discoverable via
# to_out_variant() (e.g. flat _out naming instead of .out overload).
_manual_out_variant_registry: dict[torch._ops.OpOverload, torch._ops.OpOverload] = {}


def register_out_variant(
    functional_op: torch._ops.OpOverload,
    out_op: torch._ops.OpOverload,
) -> None:
    """Register a functional op -> out variant mapping."""
    _manual_out_variant_registry[functional_op] = out_op


def lookup_manual_out_variant(
    op: torch._ops.OpOverload,
) -> torch._ops.OpOverload | None:
    """Return the manually registered out variant for op, or None."""
    return _manual_out_variant_registry.get(op)


def _is_functional(schema: torch._C.FunctionSchema) -> bool:
    """
    A schema is functional if no argument is written to and the name doesn't
    end with '_'.
    """
    op_name = schema.name.split("::")[-1]
    if op_name.endswith("_"):
        return False
    return not any(arg.is_write for arg in schema.arguments)


def _is_mutable_arg(arg: torch._C.Argument) -> bool:
    return arg.alias_info is not None and arg.alias_info.is_write


def _signatures_match(
    schema_a: torch._C.FunctionSchema,
    schema_b: torch._C.FunctionSchema,
) -> bool:
    """Compare two schemas by their non-mutable arguments (name, type, default value)."""
    non_mutable_args_a = [arg for arg in schema_a.arguments if not _is_mutable_arg(arg)]
    non_mutable_args_b = [arg for arg in schema_b.arguments if not _is_mutable_arg(arg)]
    if len(non_mutable_args_a) != len(non_mutable_args_b):
        return False
    for a, b in zip(non_mutable_args_a, non_mutable_args_b):
        if a.name != b.name:
            return False
        if str(a.type) != str(b.type):
            return False
        if a.default_value != b.default_value:
            return False
    return True


def _has_valid_out_variant_returns(
    schema: torch._C.FunctionSchema,
    mutable_args: list[torch._C.Argument],
) -> bool:
    """Out variant must return either nothing or the mutable args themselves."""
    if len(schema.returns) == 0:
        return True

    if len(schema.returns) != len(mutable_args):
        return False

    # Each return must alias exactly one mutable arg, in order
    for ret, arg in zip(schema.returns, mutable_args):
        if ret.alias_info is None or arg.alias_info is None:
            return False
        if ret.alias_info.before_set != arg.alias_info.before_set:
            return False
    return True


def get_out_arg_names(out_op: torch._ops.OpOverload) -> list[str]:
    """Get the names of out arguments for an out variant op."""
    schema = out_op._schema
    return [arg.name for arg in schema.arguments if _is_mutable_arg(arg)]


def to_out_variant(op: torch._ops.OpOverload) -> torch._ops.OpOverload | None:
    """
    Given a functional operator overload, return its corresponding out variant.
    """
    schema = op._schema

    if not _is_functional(schema):
        raise RuntimeError(
            f"Failed to find out variant for op '{op}' as its schema is not functional. \n"
            f"  {schema}"
        )

    # Get the op packet to access all overloads
    namespace = op.namespace
    op_name = schema.name.split("::")[1]
    torch_packet = getattr(getattr(torch.ops, namespace), op_name)

    # Search through all overloads for matching out variant
    for candidate in torch_packet.op_overloads():
        # pyrefly: ignore [missing-attribute]
        if torch.Tag.out not in candidate.tags:
            continue

        candidate_schema = candidate._schema

        if not _signatures_match(schema, candidate_schema):
            continue

        # We assume that all mutable args are used for out
        mutable_args = [
            arg for arg in candidate_schema.arguments if _is_mutable_arg(arg)
        ]
        if len(mutable_args) != len(schema.returns):
            continue

        if not _has_valid_out_variant_returns(candidate_schema, mutable_args):
            raise RuntimeError(
                f"Out variant {candidate} has invalid returns. "
                f"Expected either no returns or returns that alias the mutable args, "
                f"got: {candidate_schema}"
            )

        return candidate

    return None


def check_out_variant(
    functional_op: torch._ops.OpOverload, expected_out_op: torch._ops.OpOverload
) -> None:
    """
    Checks that to_out_variant returns the expected out variant for a functional op.
    Raises AssertionError if the out variant is not valid.
    """
    out_op = to_out_variant(functional_op)
    if out_op is None:
        tagged_info = _get_out_variants_info(functional_op)
        raise AssertionError(
            f"We did not find an out variant for {functional_op}. Some common mistakes include:\n"
            "  1. The out variant is missing the torch.Tag.out tag.\n"
            "  2. The out variant is not an overload of the original op (e.g., 'op.out' or 'op.overload_out') \n"
            "  3. The out variant's input arguments does not match the functional op's signature (excluding the mutable args).\n"
            "  4. The original operator is not functional.\n"
            f"Overloads tagged with out:\n"
            f"{tagged_info or '  (none)'}"
        )
    if out_op != expected_out_op:
        raise AssertionError(
            f"to_out_variant({functional_op}) returned {out_op}, "
            f"but expected {expected_out_op}. "
            f"The out variant name does not match the functional op."
        )


def _get_out_variants_info(functional_op) -> str:
    """Collect information about overloads tagged with out for debugging."""
    namespace = functional_op.namespace
    op_name = functional_op._schema.name.split("::")[1]
    torch_packet = getattr(getattr(torch.ops, namespace), op_name)

    overloads_info: list[str] = []
    for overload_name in torch_packet.overloads():
        candidate = getattr(torch_packet, overload_name)
        # pyrefly: ignore [missing-attribute]
        if torch.Tag.out in candidate.tags:
            overloads_info.append(f"  - {overload_name}: {candidate._schema}")

    return "\n".join(overloads_info)
<|endoftext|>from collections.abc import Callable
from typing import Any

from torch.utils._ordered_set import OrderedSet

from .effects import EffectHolder
from .fake_impl import FakeImplHolder
from .utils import RegistrationHandle


__all__ = [
    "SimpleLibraryRegistry",
    "SimpleOperatorEntry",
    "singleton",
    "SymmMemArgsHolder",
]


class SimpleLibraryRegistry:
    """Registry for the "simple" torch.library APIs

    The "simple" torch.library APIs are a higher-level API on top of the
    raw PyTorch DispatchKey registration APIs that includes:
    - fake impl

    Registrations for these APIs do not go into the PyTorch dispatcher's
    table because they may not directly involve a DispatchKey. For example,
    the fake impl is a Python function that gets invoked by FakeTensor.
    Instead, we manage them here.

    SimpleLibraryRegistry is a mapping from a fully qualified operator name
    (including the overload) to SimpleOperatorEntry.
    """

    def __init__(self) -> None:
        self._data: dict[str, SimpleOperatorEntry] = {}

    def find(self, qualname: str) -> "SimpleOperatorEntry":
        res = self._data.get(qualname, None)
        if res is None:
            self._data[qualname] = res = SimpleOperatorEntry(qualname)
        return res

    def get(self, qualname: str) -> "SimpleOperatorEntry | None":
        return self._data.get(qualname, None)


singleton: SimpleLibraryRegistry = SimpleLibraryRegistry()


class SimpleOperatorEntry:
    """This is 1:1 to an operator overload.

    The fields of SimpleOperatorEntry are Holders where kernels can be
    registered to.
    """

    def __init__(self, qualname: str) -> None:
        self.qualname: str = qualname
        self.fake_impl: FakeImplHolder = FakeImplHolder(qualname)
        self.torch_dispatch_rules: GenericTorchDispatchRuleHolder = (
            GenericTorchDispatchRuleHolder(qualname)
        )

        self.effect: EffectHolder = EffectHolder(qualname)
        self.symm_mem_args: SymmMemArgsHolder = SymmMemArgsHolder(qualname)

    # For compatibility reasons. We can delete this soon.
    @property
    def abstract_impl(self) -> FakeImplHolder:
        return self.fake_impl


class GenericTorchDispatchRuleHolder:
    def __init__(self, qualname: str) -> None:
        self._data: dict[type, Callable[..., Any]] = {}
        self.qualname: str = qualname

    def register(
        self, torch_dispatch_class: type, func: Callable[..., Any]
    ) -> RegistrationHandle:
        if self.find(torch_dispatch_class):
            raise RuntimeError(
                f"{torch_dispatch_class} already has a `__torch_dispatch__` rule registered for {self.qualname}"
            )
        self._data[torch_dispatch_class] = func

        def deregister() -> None:
            del self._data[torch_dispatch_class]

        return RegistrationHandle(deregister)

    def find(self, torch_dispatch_class: type) -> Callable[..., Any] | None:
        return self._data.get(torch_dispatch_class, None)


class SymmMemArgsHolder:
    """Tracks which arguments of an operator require symmetric memory allocation.

    Used by Inductor during lowering to automatically realize tensors as
    symmetric memory buffers.
    """

    def __init__(self, qualname: str) -> None:
        self._symm_mem_args: OrderedSet[str] | None = None
        self.qualname: str = qualname

    def register(self, arg_names: list[str]) -> RegistrationHandle:
        if not arg_names:
            raise ValueError(
                f"Cannot register empty arg_names list for {self.qualname}"
            )

        if self._symm_mem_args is not None:
            import logging

            log = logging.getLogger(__name__)
            log.warning(
                "Overwriting symm_mem arg registration for %s. Old: %s, New: %s",
                self.qualname,
                self._symm_mem_args,
                arg_names,
            )

        self._symm_mem_args = OrderedSet(arg_names)

        def deregister() -> None:
            self._symm_mem_args = None

        return RegistrationHandle(deregister)

    def get(self) -> OrderedSet[str] | None:
        return self._symm_mem_args

    def is_registered(self) -> bool:
        return self._symm_mem_args is not None

    def is_symm_mem_arg(self, arg_name: str) -> bool:
        return self._symm_mem_args is not None and arg_name in self._symm_mem_args


def find_torch_dispatch_rule(
    op: Any, torch_dispatch_class: type
) -> Callable[..., Any] | None:
    return singleton.find(op.__qualname__).torch_dispatch_rules.find(
        torch_dispatch_class
    )
<|endoftext|>"""
Note [Opaque Objects]

Opaque objects are the way we allow custom operators to accept a user-defined
"black box" object as an input.

There are two kinds of opaque types: CONSTANT type and SYMBOLIC type.
The distinction determines how torch.compile handles the object.

SYMBOLIC TYPES (default):

Symbolic-typed opaque objects represent mutable stateful objects and are
treated as black boxes. In torch.compile, since torch.compile cannot optimize
the anything (including tensors) within the object, the object must be an
input to the graph.

You can register a custom class as being a symbolic-typed opaque object class
through `register_custom_class(MyClass, typ="symbolic")`.

CONSTANT TYPES:

Constant-typed opaque objects represent constant values.
In torch.compile, the graph specializes on the object like how other constants
are. Therefore there are a couple of methods on the class that must be
implemented before registering it as a constant-typed opaque object class:
  - __eq__: torch.compile will create guards based on the equality of this
  object, meaning that a recompilation will happen if __eq__ returns False.
  - __hash__: This must be implemented for Fake Tensor caching
  - __fx_repr__: This must be implemented to provide an evaluable representation
    for FX graph codegen. It should return a tuple of (repr_string, dict[str, type])
    where repr_string can reconstruct the object and the dict maps names used in
    repr_string to their corresponding types.

You can register a custom class as being a constant-typed opaque object class
through `register_custom_class(MyClass, typ="constant")`.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, NewType, TYPE_CHECKING, TypeAlias
from typing_extensions import TypeIs
from weakref import WeakKeyDictionary

import torch
from torch._custom_class_base import CustomClassBase, CustomClassBaseMeta


if TYPE_CHECKING:
    from torch.fx import Proxy
    from torch.fx.experimental.proxy_tensor import PythonKeyTracer

from .fake_class_registry import register_fake_class


log = logging.getLogger(__name__)


class MemberType(Enum):
    """
    Defines how a member (attribute/property/method) of an opaque object is handled
    during torch.compile tracing.
    """

    # Reads/calls the member at trace time with the real object and bakes the result as a constant
    USE_REAL = "use_real"
    # Inlines/traces the member
    INLINED = "inlined"


@register_fake_class("aten::OpaqueObject")
class FakeOpaqueObject:
    def __init__(self) -> None:
        pass

    @classmethod
    def __obj_unflatten__(cls, flattened_ctx: dict[str, Any]) -> None:
        raise RuntimeError(
            "FakeOpaqueObject should not be created through __obj_unflatten__ "
            "and should be special handled. Please file an issue to Github."
        )


OpaqueTypeStr = "__torch__.torch.classes.aten.OpaqueObject"

OpaqueType = NewType("OpaqueType", torch._C.ScriptObject)

# Type for reconstruct_fn: called by PythonKeyTracer.create_arg when make_fx
# encounters an untracked opaque reference (e.g. a backward closure capture).
# Should derive the object from existing graph inputs or return None to fall
# back to get_attr.  Args: (obj, get_tracked_proxy, tracer).
ReconstructFn: TypeAlias = Callable[
    [CustomClassBase, Callable[[CustomClassBase], "Proxy | None"], "PythonKeyTracer"],
    "Proxy | None",
]


@dataclass
class _OpaqueTypeInfo:
    class_name: str
    opaque_typ: Literal["symbolic", "constant"]
    guard_fn: Callable[
        [Any], list[Any]
    ]  # Callable that takes the object and returns list of values to guard on
    members: dict[str, MemberType]  # Maps member name to how it should be handled
    hoist: bool
    reconstruct_fn: ReconstructFn | None


# Mapping of type -> (string name, symbolic/constant type)
_OPAQUE_TYPES: WeakKeyDictionary[Any, _OpaqueTypeInfo] = WeakKeyDictionary()
# Mapping of class_name -> (type, symbolic/constant type)
_OPAQUE_TYPES_BY_NAME: dict[str, _OpaqueTypeInfo] = {}


def _resolve_opaque_type_info(cls: Any) -> _OpaqueTypeInfo | None:
    if cls in _OPAQUE_TYPES:
        return _OPAQUE_TYPES[cls]
    if not isinstance(cls, type):
        return None

    # Allow subclasses too
    for parent in cls.__mro__[1:]:
        if parent in _OPAQUE_TYPES:
            return _OPAQUE_TYPES[parent]
    return None


def get_opaque_type_name(cls: Any) -> str:
    """
    Gets the registered opaque type name for a given class.

    Args:
        cls (type): The class to get the type name for.

    Returns:
        str: The registered type name for the class.

    Raises:
        ValueError: If the class is not registered as an opaque type.
    """
    info = _resolve_opaque_type_info(cls)
    if info is None:
        raise ValueError(
            f"Class {cls} is not registered as an opaque type. "
            f"Call register_custom_class({cls.__name__}) first."
        )
    return info.class_name


def register_custom_class(
    cls: Any,
    *,
    typ: str,
    hoist=False,
    guard_fn: Any = None,
    members: dict[str, MemberType] | None = None,
    reconstruct_fn: ReconstructFn | None = None,
) -> None:
    """
    Registers the given type as an opaque type which allows this to be consumed
    by a custom operator.

    The type name will be automatically generated from the class's fully
    qualified name (ex. my_module.MyClass).

    Args:
        cls (type): The class to register as an opaque type.
        typ (str): Either "symbolic" or "constant". See Note [Opaque Objects] for
            more details.
        hoist (bool): Only applies to constant types. A hoist=True constant type
            object is lifted as an input to the torch.compile'd graph, instead
            of being a constant baked into the graph. This is useful to
            improve compilation times in hierarchical compilation
            (e.g., change your custom ops to use hoisted strings to avoid
            baking the string into the Dynamo/AOTAutograd/FX graphs).
            This flag does nothing for symbolic types.
        guard_fn (callable | None): A function that takes an instance of the opaque
            object and returns a list of values to guard on. These values will be compared
            for equality on each function call, triggering recompilation if they change.
            Only applicable for symbolic types.
            Example: lambda obj: [obj.x, obj.y]
        members (dict[str, MemberType] | None): Dictionary mapping member names
            (attributes, properties, or methods) to their MemberType, which controls
            how they are handled during torch.compile tracing:
            - MemberType.USE_REAL: Evaluates with the real object at compile time and
              bakes the result as a constant
            - MemberType.INLINED: Inlines the method call into the trace
    """
    import torch.utils._pytree as pytree

    # Prevent registration of built-in types (int, str, list, dict, etc.) and torch.Tensor
    if cls.__module__ == "builtins" or cls is torch.Tensor:
        raise ValueError(
            f"Unable to register built-in type {cls} as an opaque type. "
            "Please wrap it in a custom class and register the custom class as opaque."
        )

    if cls in pytree.SUPPORTED_NODES:
        raise ValueError(
            f"{cls} cannot be registered as an opaque object as it has been "
            "registered as a pytree. Opaque objects must be pytree leaves."
        )

    if typ == "reference":
        log.warning("typ='reference' is deprecated, use typ='symbolic' instead")
        typ = "symbolic"

    if typ == "value":
        log.warning("typ='value' is deprecated, use typ='constant' instead")
        typ = "constant"

    # Constant types store the real object directly during tracing (no
    # FakeScriptObject wrapper), so they don't need CustomClassBaseMeta.
    if typ != "constant" and not isinstance(cls, CustomClassBaseMeta):
        raise TypeError(
            f"Custom class {cls} must subclass torch._custom_class_base.CustomClassBase "
            "or 'metaclass=torch._custom_class_base.CustomClassBaseMeta'. "
            "This is required so that FakeScriptObject can be registered "
            "as a virtual subclass, allowing isinstance() checks to work "
            "during torch.compile tracing. "
        )

    if typ not in ["symbolic", "constant"]:
        raise AssertionError(
            f"Custom class type must be either 'symbolic' or 'constant', got {typ!r}"
        )

    if typ == "constant":
        # Enums use identity-based equality (singletons), which is fine for guarding.
        if not issubclass(cls, Enum) and cls.__eq__ is object.__eq__:  # type: ignore[comparison-overlap]
            raise TypeError(
                f"Value-type opaque object of type {cls} is "
                "expected to have a non-default `__eq__` "
                "implementation as we will use this in torch.compile "
                "to guard on the equality of objects."
            )

        # Class with a custom `__eq__` without `__hash__` won't inherit the default
        # `__hash__` from object; see https://stackoverflow.com/a/1608907.
        if cls.__hash__ is None:  # type: ignore[comparison-overlap]
            raise TypeError(
                f"Value-type opaque object of type {cls} is "
                "expected to have a non-default `__hash__` "
                "implementation as we will use this in torch.compile "
                "for FakeTensor caching."
            )

        # Enums are special-cased in get_opaque_obj_repr.
        if not issubclass(cls, Enum) and not hasattr(cls, "__fx_repr__"):
            raise TypeError(
                f"Value-type opaque object of type {cls} is "
                "expected to have a `__fx_repr__` method "
                "implementation as we will use this to reconstruct "
                "the object in the FX codegen. __fx_repr__ should return "
                "a tuple of (repr_string, dict[str, type])."
            )

        if guard_fn is not None:
            raise TypeError(
                "No need to specify `guard_fn` for "
                f"value-type opaque class {cls} as it will be guarded based "
                "on `__eq__`."
            )

    # Generate a fully qualified name by combining module and qualname
    name = f"{cls.__module__}.{cls.__qualname__}"

    type_info = _OpaqueTypeInfo(
        name, typ, guard_fn, members or {}, hoist, reconstruct_fn
    )
    _OPAQUE_TYPES[cls] = type_info
    _OPAQUE_TYPES_BY_NAME[name] = type_info

    torch._C._register_opaque_type(name)


def register_opaque_type(
    cls: Any,
    *,
    typ: str,
    hoist=False,
    guard_fn: Any = None,
    members: dict[str, MemberType] | None = None,
    reconstruct_fn: ReconstructFn | None = None,
) -> None:
    log.warning("register_opaque_type is deprecated, use register_custom_class instead")
    register_custom_class(
        cls,
        typ=typ,
        hoist=hoist,
        guard_fn=guard_fn,
        members=members,
        reconstruct_fn=reconstruct_fn,
    )


# Enums are always opaque constant types.
register_custom_class(Enum, typ="constant")


def is_custom_class_obj(value: object) -> TypeIs[OpaqueType]:
    if is_custom_class(type(value)):
        return True
    from torch._library.fake_class_registry import FakeScriptObject

    if isinstance(value, FakeScriptObject):
        return is_custom_class(type(value.real_obj))
    return False


def is_opaque_value(value: object) -> TypeIs[OpaqueType]:
    log.warning("is_opaque_value is deprecated, use is_custom_class_obj instead")
    return is_custom_class_obj(value)


def should_hoist(cls: Any) -> bool:
    info = _resolve_opaque_type_info(cls)
    if info is None:
        return False
    return info.hoist


def get_reconstruct_fn(cls: type[CustomClassBase]) -> ReconstructFn | None:
    info = _resolve_opaque_type_info(cls)
    if info is None:
        return None
    return info.reconstruct_fn


def has_members(cls: Any) -> bool:
    info = _resolve_opaque_type_info(cls)
    if info is None:
        return False
    return len(info.members) > 0


def is_custom_class(cls: type[Any] | str) -> bool:
    """
    Checks if the given type is a registered custom class.
    Also returns True for subclasses of registered custom classes.
    """
    if isinstance(cls, str):
        return torch._C._is_opaque_type_registered(cls)

    if not isinstance(cls, type):
        log.warning("Passed invalid type `%s` to is_custom_class, returning False", cls)
        return False

    info = _resolve_opaque_type_info(cls)
    if info is None:
        return False

    return torch._C._is_opaque_type_registered(info.class_name)


def is_opaque_type(cls: type[Any] | str) -> bool:
    log.warning("is_opaque_type is deprecated, use is_custom_class instead")
    return is_custom_class(cls)


def is_opaque_constant_type(cls: type[Any] | str) -> bool:
    """
    Checks if the given type is an opaque **constant** type.
    See Note [Opaque Objects] for more information.
    """
    if not is_custom_class(cls):
        return False

    if isinstance(cls, str):
        return _OPAQUE_TYPES_BY_NAME[cls].opaque_typ == "constant"

    info = _resolve_opaque_type_info(cls)
    if info is None:
        return False
    return info.opaque_typ == "constant"


def is_opaque_value_type(cls: type[Any] | str) -> bool:
    log.warning(
        "is_opaque_value_type is deprecated, use is_opaque_constant_type instead"
    )
    return is_opaque_constant_type(cls)


def is_opaque_symbolic_type(cls: Any) -> bool:
    """
    Checks if the given type is an opaque **symbolic** type.
    See Note [Opaque Objects] for more information.
    """
    if not is_custom_class(cls):
        return False

    if isinstance(cls, str):
        return _OPAQUE_TYPES_BY_NAME[cls].opaque_typ == "symbolic"

    info = _resolve_opaque_type_info(cls)
    if info is None:
        return False
    return info.opaque_typ == "symbolic"


def is_opaque_reference_type(cls: Any) -> bool:
    log.warning(
        "is_opaque_reference_type is deprecated, use is_opaque_symbolic_type instead"
    )
    return is_opaque_symbolic_type(cls)


def get_opaque_obj_repr(obj: Any) -> tuple[str, dict[str, type]]:
    """
    Get the FX-evaluable repr for an opaque object and collect required globals.

    Objects must implement __fx_repr__() which should return:
        (repr_string, dict_mapping_name_to_type)

    where repr_string is an evaluable string representation and
    dict_mapping_name_to_type maps the names used in repr_string to their types.

    For example, if repr_string is "Foo(bar=Bar(1))", the dict should be:
        {"Foo": Foo, "Bar": Bar}
    """

    # Enums are special cased
    if isinstance(obj, Enum):
        cls = type(obj)
        return f"{cls.__name__}.{obj.name}", {cls.__name__: cls}

    if not hasattr(obj, "__fx_repr__"):
        raise TypeError(
            f"Value-type opaque object of type {obj} is "
            "expected to have a `__fx_repr__` method "
            "implementation as we will use this to reconstruct "
            "the object in the FX codegen. __fx_repr__ should return "
            "a tuple of (repr_string, dict[str, type])."
        )

    repr_str, globals_dict = obj.__fx_repr__()

    if not isinstance(repr_str, str):
        raise TypeError(
            f"__fx_repr__ for {type(obj).__name__} must return a string as the "
            f"first element, got {type(repr_str).__name__}"
        )

    if not isinstance(globals_dict, dict):
        raise TypeError(
            f"__fx_repr__ for {type(obj).__name__} must return a dict as the "
            f"second element, got {type(globals_dict).__name__}"
        )

    return repr_str, globals_dict


def get_opaque_obj_info(cls: Any) -> _OpaqueTypeInfo | None:
    if not is_custom_class(cls):
        return None

    if isinstance(cls, str):
        return _OPAQUE_TYPES_BY_NAME[cls]

    return _resolve_opaque_type_info(cls)


def get_member_type(cls: Any, member_name: str) -> MemberType | None:
    """
    Get the MemberType for a specific member of an opaque object class.

    Args:
        cls: The opaque object class (or its string name)
        member_name: The name of the member to query

    Returns:
        MemberType if the member is registered, None otherwise
    """
    info = get_opaque_obj_info(cls)
    if info is None:
        return None
    return info.members.get(member_name)
<|endoftext|># mypy: allow-untyped-defs
import contextlib
import functools
import logging
from collections.abc import Callable
from typing_extensions import deprecated

import torch
from torch._library.utils import Kernel, RegistrationHandle


log = logging.getLogger(__name__)


class FakeImplHolder:
    """A holder where one can register a fake impl to."""

    def __init__(self, qualname: str):
        self.qualname: str = qualname
        # kernels stores all registered fake kernels, ordered by registration
        # time ascendingly (newer registration after older registration). If an
        # operator library gets loaded that overrides an existing fake kernel,
        # both kernels will be in the list, but the newest one will be the one
        # that is run. If the library is unloaded, we will remove the kernel
        # from this list.
        self.kernels: list[Kernel] = []

    @property
    def kernel(self):
        if len(self.kernels) == 0:
            return None
        return self.kernels[-1]

    @kernel.setter
    def kernel(self, value):
        raise RuntimeError("Unable to directly set kernel.")

    def register(
        self, func: Callable, source: str, lib, *, allow_override=True
    ) -> RegistrationHandle:
        """Register a fake impl.

        Returns a RegistrationHandle that one can use to de-register this
        fake impl.
        """

        if not allow_override:
            if self.kernel is not None:
                raise RuntimeError(
                    f"register_fake(...): the operator {self.qualname} "
                    f"already has a fake impl registered at "
                    f"{self.kernel.source}."
                )
            if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta"):
                raise RuntimeError(
                    f"register_fake(...): the operator {self.qualname} "
                    f"already has a DispatchKey::Meta implementation via a "
                    f"pre-existing torch.library or TORCH_LIBRARY registration. "
                    f"Please either remove that registration or don't call "
                    f"register_fake."
                )

            if torch._C._dispatch_has_kernel_for_dispatch_key(
                self.qualname, "CompositeImplicitAutograd"
            ):
                raise RuntimeError(
                    f"register_fake(...): the operator {self.qualname} "
                    f"already has an implementation for this device type via a "
                    f"pre-existing registration to "
                    f"DispatchKey::CompositeImplicitAutograd."
                    f"CompositeImplicitAutograd operators do not need a fake "
                    f"impl; "
                    f"instead, the operator will decompose into its constituents "
                    f"and those "
                    f"can have fake impls defined on them."
                )

        # Store the kernel in this holder
        kernel = Kernel(func, source)
        self.kernels.append(kernel)

        def deregister_fake_kernel():
            self.kernels.remove(kernel)

        meta_kernel = construct_meta_kernel(self.qualname, self)
        try:
            lib.impl(self.qualname, meta_kernel, "Meta", allow_override=allow_override)
        except Exception:
            log.info(
                "Failed to register fake_impl '%s':",
                self.qualname,
            )
            self.kernels.remove(kernel)
            raise

        handle = RegistrationHandle(deregister_fake_kernel)
        return handle


def construct_meta_kernel(qualname: str, fake_impl_holder: FakeImplHolder) -> Callable:
    if fake_impl_holder.kernel is None:
        raise AssertionError("fake_impl_holder.kernel must not be None")

    @functools.wraps(fake_impl_holder.kernel.func)
    def meta_kernel(*args, **kwargs):
        if fake_impl_holder.kernel is None:
            raise AssertionError("fake_impl_holder.kernel must not be None")
        source = fake_impl_holder.kernel.source

        def error_on_ctx():
            raise RuntimeError(
                f"{qualname} ({source}): You're trying to run this operator "
                f"with meta Tensors (as opposed to FakeTensors), but this "
                f"operator may return an output Tensor with data-dependent shape. Meta "
                f"Tensors don't support operators with outputs that have data-dependent shapes "
                f"but FakeTensors do. "
                f"If your operator does not return an output with data-dependent shape, "
                f"make sure the FakeTensor and/or meta kernel does not call "
                f"torch.library.get_ctx(). Otherwise, please use FakeTensors."
            )

        with set_ctx_getter(error_on_ctx):
            return fake_impl_holder.kernel(*args, **kwargs)

    return meta_kernel


def get_none():
    return None


global_ctx_getter: Callable = get_none


@contextlib.contextmanager
def set_ctx_getter(ctx_getter):
    global global_ctx_getter
    prev = global_ctx_getter
    try:
        global_ctx_getter = ctx_getter
        yield
    finally:
        global_ctx_getter = prev


class FakeImplCtx:
    """
    Context object for writing fake implementations for custom operators.
    """

    def __init__(self, _fake_mode, _op):
        self._fake_mode = _fake_mode
        self._shape_env = _fake_mode.shape_env
        self._op = _op

    @deprecated(
        "`create_unbacked_symint` is deprecated, please use `new_dynamic_size` instead",
        category=FutureWarning,
    )
    def create_unbacked_symint(self, *, min=2, max=None) -> torch.SymInt:
        return self.new_dynamic_size(min=min, max=max)

    def new_dynamic_size(self, *, min=0, max=None) -> torch.SymInt:
        """Constructs a new symint (symbolic int) representing a data-dependent value.

        This is useful for writing the fake implementation (which is necessary
        for torch.compile) for a CustomOp where an output Tensor has a size
        that depends on the data of the input Tensors.

        Args:
            min (int): A statically known inclusive lower bound for this symint. Default: 0
            max (Optional[int]): A statically known inclusive upper bound for this
                symint. Default: None

        .. warning:

            It is important that the ``min`` and ``max`` (if not None) values are set
            correctly, otherwise, there will be undefined behavior under
            torch.compile. The default value of ``min`` is 2 due to torch.compile
            specializing on 0/1 sizes.

            You must also verify that your implementation on concrete Tensors
            (e.g. CPU/CUDA) only returns Tensors where the size that corresponds
            to the symint also has respects these constraint.
            The easiest way to do this is to add an assertion in the CPU/CUDA/etc
            implementation that the size follows these bounds.

        Example::

            >>> # An operator with data-dependent output shape
            >>> lib = torch.library.Library("mymodule", "FRAGMENT")
            >>> lib.define("mymodule::custom_nonzero(Tensor x) -> Tensor")
            >>>
            >>> @torch.library.register_fake("mymodule::custom_nonzero")
            >>> def _(x):
            >>>     # Number of nonzero-elements is data-dependent.
            >>>     # Since we cannot peek at the data in a fake impl,
            >>>     # we use the ctx object to construct a new symint that
            >>>     # represents the data-dependent size.
            >>>     ctx = torch.library.get_ctx()
            >>>     nnz = ctx.new_dynamic_size()
            >>>     shape = [nnz, x.dim()]
            >>>     result = x.new_empty(shape, dtype=torch.int64)
            >>>     return result
            >>>
            >>> @torch.library.impl(lib, "custom_nonzero", "CPU")
            >>> def _(x):
            >>>     x_np = x.numpy()
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     return torch.tensor(res, device=x.device)

        """
        if (
            self._shape_env is None
            or not self._shape_env.allow_dynamic_output_shape_ops
        ):
            raise torch._subclasses.fake_tensor.DynamicOutputShapeException(self._op)

        if isinstance(min, torch.SymInt) or isinstance(max, torch.SymInt):
            raise ValueError(
                f"ctx.new_dynamic_size(min={min}, max={max}): expected "
                f"min and max to be statically known ints but got SymInt. "
                f"This is not supported."
            )

        if min < 0:
            raise ValueError(
                f"ctx.new_dynamic_size(min={min}, ...): expected min to be "
                f"greater than or equal to 0: this API can only create "
                f"non-negative sizes."
            )

        return allocate_size(self._shape_env, min, max)


def allocate_size(shape_env, min_val=0, max_val=None):
    result = shape_env.create_unbacked_symint()
    torch.fx.experimental.symbolic_shapes._constrain_range_for_size(
        result, min=min_val, max=max_val
    )
    return result
<|endoftext|># mypy: allow-untyped-defs
import collections
import inspect
import typing
from types import GenericAlias

import torch
from torch import device, dtype, Tensor, types
from torch.utils._exposed_in import exposed_in

from .opaque_object import (
    _resolve_opaque_type_info,
    is_custom_class,
    is_opaque_symbolic_type,
)


# This is used as a negative test for
# test_custom_ops.py::TestTypeConversion::test_type_eval.
_TestTensor = torch.Tensor


@exposed_in("torch.library")
def infer_schema(
    prototype_function: typing.Callable,
    /,
    *,
    mutates_args,
    op_name: str | None = None,
    tags: torch.Tag | typing.Sequence[torch.Tag] | None = (),
) -> str:
    r"""Parses the schema of a given function with type hints. The schema is inferred from the
    function's type hints, and can be used to define a new operator.

    We make the following assumptions:

    * None of the outputs alias any of the inputs or each other.
    * | String type annotations "device, dtype, Tensor, types" without library specification are
      | assumed to be torch.*. Similarly, string type annotations "Optional, List, Sequence, Union"
      | without library specification are assumed to be typing.*.
    * | Only the args listed in ``mutates_args`` are being mutated. If ``mutates_args`` is "unknown",
      | it assumes that all inputs to the operator are being mutates.

    Callers (e.g. the custom ops API) are responsible for checking these assumptions.

    Args:
        prototype_function: The function from which to infer a schema for from its type annotations.
        op_name (Optional[str]): The name of the operator in the schema. If ``name`` is None, then the
            name is not included in the inferred schema. Note that the input schema to
            ``torch.library.Library.define`` requires an operator name.
        mutates_args ("unknown" | Iterable[str]): The arguments that are mutated in the function.
        tags (Tag | Sequence[Tag] | None): one or more tags to apply to the
            inferred schema. Use ``torch.Tag.inplace`` or ``torch.Tag.out`` to
            infer the conventional aliasing for those operator kinds.

    Returns:
        The inferred schema.

    Example:
        >>> def foo_impl(x: torch.Tensor) -> torch.Tensor:
        >>>     return x.sin()
        >>>
        >>> infer_schema(foo_impl, op_name="foo", mutates_args={})
        foo(Tensor x) -> Tensor
        >>>
        >>> infer_schema(foo_impl, mutates_args={})
        (Tensor x) -> Tensor
    """
    UNKNOWN_MUTATES = "unknown"
    pf_globals = prototype_function.__globals__
    pf_locals = None
    # TODO: Once our minimum version is py3.10+ pass `eval_str=True` to
    # inspect.signature() and we no longer need to deal with stringified
    # annotations below.
    sig = inspect.signature(prototype_function)
    if tags is None:
        tags = ()
    elif isinstance(tags, torch.Tag):
        tags = (tags,)
    is_inplace = torch.Tag.inplace in tags
    is_out = torch.Tag.out in tags
    if is_inplace and is_out:
        raise AssertionError(
            "torch.Tag.inplace and torch.Tag.out are mutually exclusive"
        )
    if type(mutates_args) is not str:
        mutates_args = tuple(mutates_args)

    def error_fn(what):
        raise ValueError(f"infer_schema(func): {what} Got func with signature {sig})")

    def convert_type_string(annotation_type: str):
        try:
            return eval(annotation_type, pf_globals, pf_locals)
        except NameError as e:
            error_fn(
                f"Unsupported type annotation {annotation_type}. It is not a type. "
                f"({e}). "
                f"If you are using 'from __future__ import annotations', note that "
                f"annotations are evaluated lazily as strings; make sure all types "
                f"referenced in annotations are importable at module scope, not only "
                f"inside a local function."
            )
        except Exception as e:
            error_fn(
                f"Unsupported type annotation {annotation_type}. It is not a type. "
                f"({type(e).__name__}: {e})"
            )

    def unstringify_types(
        tys: tuple[type[object] | str, ...],
    ) -> tuple[tuple[typing.Any, ...], bool]:
        res = []
        changed = False
        for ty in tys:
            ty, ty_changed = unstringify_type(ty)
            res.append(ty)
            changed |= ty_changed
        if changed:
            return tuple(res), True
        else:
            return tys, False  # type: ignore[return-value]

    def unstringify_type(ty: type[object] | str) -> tuple[typing.Any, bool]:
        # Dig through a generic type and if it contains a stringified type
        # convert that to a real type. The second return value indicates if the
        # type contained a string or not.
        if isinstance(ty, str):
            return convert_type_string(ty), True
        elif origin := typing.get_origin(ty):
            args, args_changed = unstringify_types(typing.get_args(ty))
            if args_changed:
                return GenericAlias(origin, args), True

        return ty, False

    params = []
    seen_args = set()
    saw_kwarg_only_arg = False
    first_positional_arg_name = None
    first_positional_arg_schema_type = None
    first_positional_arg_alias = None
    out_arg_aliases = []
    out_arg_names = []
    for idx, (name, param) in enumerate(sig.parameters.items()):
        if not supported_param(param):
            error_fn("We do not support positional-only args, varargs, or varkwargs.")

        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            # The first time we see a kwarg-only arg, add "*" to the schema.
            if not saw_kwarg_only_arg:
                params.append("*")
                saw_kwarg_only_arg = True

        if param.annotation is inspect.Parameter.empty:
            error_fn(f"Parameter {name} must have a type annotation.")

        # The annotation might be converted to a string by annotation,
        # we convert it to the actual type.
        annotation_type, _ = unstringify_type(param.annotation)

        schema_type = None
        if annotation_type not in SUPPORTED_PARAM_TYPES:
            if is_custom_class(annotation_type):
                schema_type = _resolve_opaque_type_info(annotation_type).class_name  # type: ignore[union-attr]
            elif annotation_type == torch._C.ScriptObject:
                error_fn(
                    f"Parameter {name}'s type cannot be inferred from the schema "
                    "as it is a ScriptObject. Please manually specify the schema "
                    "using the `schema=` kwarg with the actual type of the ScriptObject."
                )
            elif (
                hasattr(annotation_type, "__origin__")
                and annotation_type.__origin__ is tuple
            ):
                list_type = tuple_to_list(annotation_type)
                example_type_str = "\n\n"
                # Only suggest the list type if this type is supported.
                if list_type in SUPPORTED_PARAM_TYPES:
                    example_type_str = f"For example, {list_type}.\n\n"
                error_fn(
                    f"Parameter {name} has unsupported type {param.annotation}. "
                    f"We do not support Tuple inputs in schema. As a workaround, please try to use List instead. "
                    f"{example_type_str}"
                    f"The valid types are: {SUPPORTED_PARAM_TYPES.keys()}."
                )
            else:
                error_fn(
                    f"Parameter {name} has unsupported type {param.annotation}. "
                    f"The valid types are: {SUPPORTED_PARAM_TYPES.keys()}."
                )
        else:
            schema_type = SUPPORTED_PARAM_TYPES[annotation_type]

        if schema_type is None:
            raise AssertionError(f"schema_type is None for param {name}")

        if (
            param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            and first_positional_arg_name is None
        ):
            first_positional_arg_name = name
            first_positional_arg_schema_type = schema_type
            first_positional_arg_alias = f"a{idx}"

        is_mutated = False
        if type(mutates_args) is str:
            if mutates_args != UNKNOWN_MUTATES:
                raise ValueError(
                    "mutates_args must either be a sequence of the names of "
                    "the arguments that are mutated or the string 'unknown'. "
                )
            if schema_type.startswith("Tensor"):
                is_mutated = True
                schema_type = f"Tensor(a{idx}!){schema_type[len('Tensor') :]}"
        elif name in mutates_args:
            is_mutated = True
            if not schema_type.startswith("Tensor"):
                error_fn(
                    f"Parameter {name} is in mutable_args but only Tensors or collections of Tensors can be mutated"
                )
            schema_type = f"Tensor(a{idx}!){schema_type[len('Tensor') :]}"
        if is_out and is_mutated:
            if param.kind != inspect.Parameter.KEYWORD_ONLY:
                error_fn("torch.Tag.out requires mutable arguments to be keyword-only.")
            if schema_type != f"Tensor(a{idx}!)":
                error_fn(
                    "torch.Tag.out only supports mutable keyword-only Tensor arguments."
                )
            out_arg_aliases.append(f"a{idx}")
            out_arg_names.append(name)
        seen_args.add(name)
        if param.default is inspect.Parameter.empty:
            # pyrefly: ignore [bad-argument-type]
            params.append(f"{schema_type} {name}")
        else:
            default_repr = None
            if param.default is None or isinstance(param.default, (int, float, bool)):
                default_repr = str(param.default)
            elif isinstance(param.default, (str, torch.device)):
                default_repr = f'"{param.default}"'
            elif isinstance(param.default, torch.dtype):
                dtype_repr = str(param.default)
                torch_dot = "torch."
                if not dtype_repr.startswith(torch_dot):
                    raise AssertionError(
                        f"dtype repr {dtype_repr!r} must start with 'torch.'"
                    )
                default_repr = dtype_repr[len(torch_dot) :]
            else:
                error_fn(
                    f"Parameter {name} has an unsupported default value type {type(param.default)}. "
                    f"Please file an issue on GitHub so we can prioritize this."
                )
            # pyrefly: ignore [bad-argument-type]
            params.append(f"{schema_type} {name}={default_repr}")
    if mutates_args != UNKNOWN_MUTATES:
        mutates_args_not_seen = set(mutates_args) - seen_args
        if len(mutates_args_not_seen) > 0:
            error_fn(
                f"{mutates_args_not_seen} in mutates_args were not found in "
                f"the custom op's signature. "
                f"mutates_args should contain the names of all args that the "
                f"custom op mutates, or just the string 'unknown' if you don't know."
            )
    if is_inplace:
        if first_positional_arg_name is None:
            error_fn("torch.Tag.inplace requires a positional Tensor argument.")
        if first_positional_arg_schema_type != "Tensor":
            error_fn(
                "torch.Tag.inplace requires the first positional argument to be a Tensor."
            )
        if type(mutates_args) is str or set(mutates_args) != {
            first_positional_arg_name
        }:
            error_fn(
                "torch.Tag.inplace requires mutates_args to contain exactly "
                f"the first positional argument, got {mutates_args}."
            )
    if is_out and len(out_arg_aliases) == 0:
        error_fn(
            "torch.Tag.out requires at least one mutable keyword-only Tensor argument."
        )
    return_annotation, _ = unstringify_type(sig.return_annotation)
    if is_out:
        ret = _infer_out_return_schema(
            return_annotation, out_arg_aliases, out_arg_names, error_fn
        )
    else:
        ret = parse_return(return_annotation, error_fn)
    if is_inplace:
        if ret != "Tensor":
            error_fn(
                "torch.Tag.inplace requires the return annotation to be torch.Tensor."
            )
        ret = f"Tensor({first_positional_arg_alias}!)"
    if op_name is not None:
        return f"{op_name}({', '.join(params)}) -> {ret}"
    return f"({', '.join(params)}) -> {ret}"


def derived_types(
    base_type: type | typing._SpecialForm,
    cpp_type: str,
    list_base: bool,
    optional_base_list: bool,
    optional_list_base: bool,
):
    result: list[tuple[type | typing._SpecialForm | GenericAlias, str]] = [
        (base_type, cpp_type),
        # pyrefly: ignore [not-a-type]
        (typing.Optional[base_type], f"{cpp_type}?"),  # noqa: UP045
    ]

    def derived_seq_types(typ: type | typing._SpecialForm):
        return (
            typing.Sequence[typ],  # type: ignore[valid-type]
            typing.List[typ],  # type: ignore[valid-type]  # noqa: UP006
            GenericAlias(collections.abc.Sequence, (typ,)),
            GenericAlias(list, (typ,)),
        )

    if list_base:
        result.extend(
            (seq_typ, f"{cpp_type}[]") for seq_typ in derived_seq_types(base_type)
        )
    if optional_base_list:
        result.extend(
            (seq_typ, f"{cpp_type}?[]")
            # pyrefly: ignore [not-a-type]
            for seq_typ in derived_seq_types(typing.Optional[base_type])  # noqa: UP045
        )
    if optional_list_base:
        result.extend(
            (typing.Optional[seq_typ], f"{cpp_type}[]?")  # noqa: UP045
            for seq_typ in derived_seq_types(base_type)
        )
    return result


def get_supported_param_types():
    data: list[tuple[type | typing._SpecialForm, str, bool, bool, bool]] = [
        # (python type, schema type, type[] variant, type?[] variant, type[]? variant
        (Tensor, "Tensor", True, True, False),
        (int, "SymInt", True, False, True),
        (float, "float", True, False, True),
        (bool, "bool", True, False, True),
        (str, "str", False, False, False),
        (types.Number, "Scalar", True, False, False),
        (dtype, "ScalarType", False, False, False),
        (device, "Device", False, False, False),
    ]

    if torch.distributed.is_available():
        from torch.distributed.distributed_c10d import GroupName

        data.append((typing.cast(type, GroupName), "str", False, False, False))

    result = []
    for line in data:
        result.extend(derived_types(*line))
    return dict(result)


SUPPORTED_RETURN_TYPES = {
    Tensor: "Tensor",
    typing.List[Tensor]: "Tensor[]",  # noqa: UP006
    list[Tensor]: "Tensor[]",
    int: "SymInt",
    float: "float",
    bool: "bool",
    types.Number: "Scalar",
}


def parse_return(annotation, error_fn):
    if annotation is None:
        return "()"

    if annotation is inspect.Parameter.empty:
        error_fn("No return type annotation was provided. Please add one.")

    origin = typing.get_origin(annotation)
    if origin is not tuple:
        if annotation not in SUPPORTED_RETURN_TYPES:
            if is_opaque_symbolic_type(annotation):
                return _resolve_opaque_type_info(annotation).class_name  # type: ignore[union-attr]
            error_fn(
                f"Return has unsupported type {annotation}. "
                f"The valid types are: {SUPPORTED_RETURN_TYPES}."
            )

        return SUPPORTED_RETURN_TYPES[annotation]

    args = typing.get_args(annotation)
    for arg in args:
        if arg not in SUPPORTED_RETURN_TYPES and not is_opaque_symbolic_type(arg):
            error_fn(
                f"Return has unsupported type {annotation}. "
                f"The valid types are: {SUPPORTED_RETURN_TYPES}."
            )

    def _return_type_str(arg):
        if ty := SUPPORTED_RETURN_TYPES.get(arg):
            return ty
        return _resolve_opaque_type_info(arg).class_name  # type: ignore[union-attr]

    output_ty = ", ".join(_return_type_str(arg) for arg in args)

    # use (()) to represent tuple with single element
    if len(args) == 1:
        output_ty = "(" + output_ty + ")"
    return "(" + output_ty + ")"


def _infer_out_return_schema(
    return_annotation,
    out_arg_aliases,
    out_arg_names,
    error_fn,
):
    if len(out_arg_aliases) == 1:
        return_annotation_matches_out_args = return_annotation is Tensor
    else:
        return_annotation_matches_out_args = typing.get_origin(
            return_annotation
        ) is tuple and typing.get_args(return_annotation) == (Tensor,) * len(
            out_arg_aliases
        )

    if not return_annotation_matches_out_args:
        error_fn(
            "torch.Tag.out requires the return annotation to match the "
            f"mutable keyword-only Tensor arguments {tuple(out_arg_names)}."
        )

    aliased_returns = [f"Tensor({out_arg_alias}!)" for out_arg_alias in out_arg_aliases]
    if len(aliased_returns) == 1:
        return aliased_returns[0]
    return f"({', '.join(aliased_returns)})"


SUPPORTED_PARAM_TYPES = get_supported_param_types()


def supported_param(param: inspect.Parameter) -> bool:
    return param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def tuple_to_list(tuple_type: type[tuple]) -> type[list]:
    """
    Convert `tuple_type` into a list type with the same type arguments. Assumes that `tuple_type` is typing.Tuple type.
    """
    type_args = typing.get_args(tuple_type)
    if (
        tuple_type is typing.Tuple  # noqa: UP006
        or tuple_type is tuple
        or not type_args
    ):
        # Handle the case of an empty tuple type
        return list
    elif len(type_args) == 1:
        # General case: create a List with the same type arguments
        return list[type_args[0]]  # type: ignore[valid-type]
    elif len(type_args) == 2 and type_args[1] is Ellipsis:
        return list[type_args[0]]  # type: ignore[valid-type]
    else:
        return list[typing.Union[tuple(type_args)]]  # type: ignore[misc, return-value]  # noqa: UP007
<|endoftext|>import contextlib
import io
import logging
import os
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch._library.custom_ops import _maybe_get_opdef
from torch.types import FileLike


log = logging.getLogger(__name__)


class MissingOpProfile(RuntimeError):
    """
    This is raised when we don't have an operator profile available for the
    given inputs.
    """


@dataclass(frozen=True)
class TensorMetadata:
    rank: int
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout

    @staticmethod
    def maybe_from_tensor(t: Any) -> Optional["TensorMetadata"]:
        if not isinstance(t, torch.Tensor):
            return None
        return TensorMetadata(t.dim(), t.dtype, t.device, t.layout)


@dataclass(frozen=True)
class OpProfile:
    args_profile: tuple[TensorMetadata | None]
    out_profile: TensorMetadata | tuple[TensorMetadata]


def _generate_fake_kernel(op_name: str, op_profile: set[OpProfile]) -> Callable:
    def _match_args(args_profile: tuple[TensorMetadata | None], args: Any) -> bool:
        return all(
            TensorMetadata.maybe_from_tensor(arg) == args_profile[i]
            for i, arg in enumerate(args)
        )

    def _generate_res(
        out_profile: TensorMetadata | tuple[TensorMetadata],
    ) -> torch.Tensor | list[torch.Tensor]:
        ctx = torch.library.get_ctx()

        def _generate_tensor_out(t: TensorMetadata) -> torch.Tensor:
            fake_shape = [ctx.new_dynamic_size() for _ in range(t.rank)]
            fake_strides = [-1] * t.rank
            expected = 1
            fake_stride = expected
            # pyrefly: ignore [bad-assignment]
            for i in range(t.rank):
                fake_strides[i] = fake_stride  # type: ignore[assignment]
                fake_stride = fake_stride * fake_shape[i]  # type: ignore[assignment]

            return torch.empty_strided(
                fake_shape,
                fake_strides,
                device=t.device,
                dtype=t.dtype,
                layout=t.layout,
            )

        if isinstance(out_profile, TensorMetadata):
            return _generate_tensor_out(out_profile)
        else:
            return [_generate_tensor_out(t) for t in out_profile]

    def _fake_kernel(*args, **kwargs):  # type: ignore[no-untyped-def]
        for profile in op_profile:
            if _match_args(profile.args_profile, (*args, *kwargs.values())):
                return _generate_res(profile.out_profile)

        raise MissingOpProfile(
            f"No fake kernel was found for {op_name}, and although we have "
            "previously registered some profiles to generate a fake kernel, "
            f"no profiles match the given inputs: {args, kwargs}."
        )

    return _fake_kernel


@contextlib.contextmanager
def unsafe_generate_fake_kernels(op_profiles: dict[str, set[OpProfile]]) -> Generator:
    """
    Registers a fake kernel based on the given operator profiles. This fake
    kernel registration will override any existing fake kernel registrations.

    The input is a dictionary mapping operator names to a set of operator
    profiles, which we will use to generate fake kernels. The operator profiles
    are a record of the input and output tensor metadata. Based on this
    information we will match a given input to the recorded profile, and return
    an output with the same metadata as in the recorded profile. If a profile
    doesn't exist then an exception will be thrown.

    The fake kernel generation is considered unsafe because it relies on the
    rigid, pre-defined operator profiles that do not account for potential
    variations in output behavior. Specifically, the generated kernels assume a
    fixed relationship between input and output ranks. However, in reality, it's
    possible that data-dependent operations may produce outputs of different
    ranks even when given inputs of the same rank. The generated fake kernels
    are inflexible and unable to accommodate these nuances, making them
    potentially unsafe.

    Args:
        op_profiles (dict[str, set[OpProfile]]): A dictionary mapping operator
            name to a set of operator profiles from which we will generate fake
            kernels.

    Examples:

        >>> # Example: Registering an op-profile from draft-export
        >>> import torch
        >>> from torch.export._draft_export import draft_export
        >>>
        >>> @torch.library.custom_op("mylib::foo", mutates_args=())
        >>> def foo(x: Tensor, y: Tensor) -> Tensor:
        >>>     return x + y
        >>>
        >>> class M(torch.nn.Module):
        >>>     def forward(self, a, b):
        >>>         res = torch.ops.mylib.foo(a, b)  # no fake impl
        >>>         return res
        >>>
        >>> ep = draft_export(M(), (torch.ones(3, 4), torch.ones(3, 4))
        >>>
        >>> with torch._library.fake_profile.unsafe_generate_fake_kernels(ep._report.op_profiles):
        >>>     decomp = ep.run_decompositions()

    """

    libs: list[torch.library.Library] = []
    # Stores old fake impls from custom ops declared through @custom_op
    old_fake_impls: dict[str, Callable] = {}
    for op_name, profiles in op_profiles.items():
        log.warning(
            "Registering fake profile for %s. This will override any existing "
            "fake kernel registration.",
            op_name,
        )

        op_name_split = op_name.split(".")
        namespace, op_name_str = op_name_split[0], op_name_split[1]
        op_str = f"{namespace}::{op_name_str}"

        fake_kernel = _generate_fake_kernel(op_str, profiles)

        if opdef := _maybe_get_opdef(op_str):
            # If the op is a CustomOpDef, save the existing abstract_fn so that
            # we can restore it after this contextmanager
            if opdef._abstract_fn is not None:
                old_fake_impls[op_str] = opdef._abstract_fn
            opdef.register_fake(fake_kernel)

        else:
            # Create a new library so that we can register a new fake impl.
            # These libraries will then be destroyed after the contextmanager,
            # which will automatically restore the previously registered fake
            # impls.
            newlib = torch.library.Library(namespace, "FRAGMENT")
            torch.library.register_fake(
                op_str, fake_kernel, lib=newlib, allow_override=True
            )
            libs.append(newlib)

    try:
        yield libs
    finally:
        # Destroying the libraries will automatically restore the previously
        # registered fake impls
        for lib in libs:
            lib._destroy()

        # Restore abstract_fns for CustomOpDefs
        for op_str, old_fake in old_fake_impls.items():
            opdef = _maybe_get_opdef(op_str)
            if opdef is None:
                raise AssertionError(f"opdef for {op_str} must not be None")
            opdef.register_fake(old_fake)


def get_torch_version() -> str:
    version = torch.__version__.split(".")
    return f"{int(version[0])}.{int(version[1])}"


def generate_yaml_from_profiles(op_profiles: dict[str, set[OpProfile]]) -> str:
    """
    Generates a yaml string from the given operator profiles which can be saved
    to a file. The yaml string can be loaded back into an operator profile
    structure using `read_profiles_from_yaml`.
    """

    import yaml

    from torch._export.serde.serialize import (
        _TORCH_TO_SERIALIZE_DTYPE,
        _TORCH_TO_SERIALIZE_LAYOUT,
    )

    def serialize_tensor_metadata(t: TensorMetadata) -> dict:
        return {
            "rank": t.rank,
            "dtype": _TORCH_TO_SERIALIZE_DTYPE[t.dtype].value,
            "device": str(t.device),
            "layout": _TORCH_TO_SERIALIZE_LAYOUT[t.layout].value,
        }

    def serialize_op_profile(op: OpProfile) -> dict:
        return {
            "args_profile": [
                serialize_tensor_metadata(arg)
                for arg in op.args_profile
                if arg is not None
            ],
            "out_profile": (
                serialize_tensor_metadata(op.out_profile)
                if isinstance(op.out_profile, TensorMetadata)
                else [serialize_tensor_metadata(out) for out in op.out_profile]
            ),
        }

    serialized_data = {
        operator: [serialize_op_profile(profile) for profile in profiles]
        for operator, profiles in op_profiles.items()
    }
    return yaml.dump(
        {"torch_version": get_torch_version(), "operators": serialized_data},
        sort_keys=False,
    )


def save_op_profiles(op_profiles: dict[str, set[OpProfile]], f: FileLike) -> None:
    """
    Serializes the given operator profiles into a yaml format and saves it to
    the given file. The operator profile can be loaded back using `load_op_profiles`.
    """
    yaml_str = generate_yaml_from_profiles(op_profiles)

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

        with open(f, "w") as file:
            file.write(yaml_str)

    elif isinstance(f, io.BytesIO):
        f.write(yaml_str.encode("utf-8"))

    else:
        raise ValueError(f"Invalid type of file {f}")


def read_profiles_from_yaml(yaml_str: str) -> dict[str, set[OpProfile]]:
    """
    Reads the yaml saved by `save_op_profiles` and returns the operator profiles.
    """

    import yaml

    from torch._export.serde.serialize import (
        _SERIALIZE_TO_TORCH_DTYPE,
        _SERIALIZE_TO_TORCH_LAYOUT,
    )

    def deserialize_tensor_metadata(data: dict) -> TensorMetadata:
        return TensorMetadata(
            rank=data["rank"],
            dtype=_SERIALIZE_TO_TORCH_DTYPE[data["dtype"]],
            device=torch.device(data["device"]),
            layout=_SERIALIZE_TO_TORCH_LAYOUT[data["layout"]],
        )

    def deserialize_op_profile(data: dict) -> OpProfile:
        args_profile = tuple(
            deserialize_tensor_metadata(arg) for arg in data["args_profile"]
        )
        out_profile_data = data["out_profile"]
        out_profile: tuple[TensorMetadata] | TensorMetadata = (
            tuple(deserialize_tensor_metadata(out) for out in out_profile_data)  # type: ignore[assignment]
            if isinstance(out_profile_data, list)
            else deserialize_tensor_metadata(out_profile_data)
        )
        return OpProfile(args_profile=args_profile, out_profile=out_profile)  # type: ignore[arg-type]

    loaded_data = yaml.safe_load(yaml_str)
    loaded_torch_version = loaded_data["torch_version"]

    if loaded_torch_version != get_torch_version():
        raise RuntimeError(
            "Unable to load outdated profile. It was saved with torch version: "
            f"{loaded_torch_version} but the current torch version is: {get_torch_version()}"
        )

    operators_data = loaded_data["operators"]
    return {
        operator: {deserialize_op_profile(profile) for profile in profiles}
        for operator, profiles in operators_data.items()
    }


def load_op_profiles(f: FileLike) -> dict[str, set[OpProfile]]:
    """
    Loads the saved operator profiles from `save_op_profiles`.
    """
    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

        with open(f) as file:
            yaml_str = file.read()

    elif isinstance(f, io.BytesIO):
        yaml_str = f.read().decode("utf-8")

    else:
        raise ValueError(f"Invalid type of file {f}")

    return read_profiles_from_yaml(yaml_str)
<|endoftext|># pyre-strict

r"""This package adds support for device memory management implemented in MTIA."""

from typing import Any

import torch

from . import Device, is_initialized
from ._utils import _get_device_index


def memory_stats(device: Device = None) -> dict[str, Any]:
    r"""Return a dictionary of MTIA memory allocator statistics for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return {}
    return torch._C._mtia_memoryStats(_get_device_index(device, optional=True))


def max_memory_allocated(device: Device = None) -> int:
    r"""Return the maximum memory allocated in bytes for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return 0
    return memory_stats(device).get("dram", 0).get("peak_bytes", 0)


def memory_allocated(device: Device = None) -> int:
    r"""Return the current MTIA memory occupied by tensors in bytes for a given device.

    Args:
        device (torch.device or int or str, optional): selected device. Returns
            statistic for the current device, given by :func:`~torch.mtia.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    if not is_initialized():
        return 0
    return memory_stats(device).get("dram", 0).get("allocated_bytes", 0)


def reset_peak_memory_stats(device: Device = None) -> None:
    r"""Reset the peak memory stats for a given device.


    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    if not is_initialized():
        return
    torch._C._mtia_resetPeakMemoryStats(_get_device_index(device, optional=True))


__all__ = [
    "memory_stats",
    "max_memory_allocated",
    "memory_allocated",
    "reset_peak_memory_stats",
]
<|endoftext|># pylint: disable=useless-parent-delegation
from __future__ import annotations

from typing_extensions import Self

import torch


_POOL_HANDLE = tuple[int, int]


def graph_pool_handle() -> _POOL_HANDLE:
    """
    Return an opaque token representing the id of a graph memory pool.
    """
    # pyrefly: ignore [missing-attribute]
    return torch._C._mtia_graphPoolHandle()


class MTIAGraph(torch._C._MTIAGraph):
    """
    Wrapper around a MTIA graph.
    """

    def __new__(cls, keep_graph: bool = False) -> Self:
        return super().__new__(cls, keep_graph)

    def capture_begin(self, pool: _POOL_HANDLE) -> None:
        """
        Begin capturing a MTIA graph.
        """
        super().capture_begin(pool)

    def capture_end(self) -> None:
        """
        End the capture of a MTIA graph.
        """
        super().capture_end()

    def instantiate(self) -> None:
        """
        Instantiate the captured MTIA graph.
        """
        super().instantiate()

    def replay(self) -> None:
        """
        Replay the captured MTIA graph.
        """
        super().replay()

    def reset(self) -> None:
        """
        Destroy the captured graph and reset the states.
        """
        super().reset()

    def pool(self) -> _POOL_HANDLE:
        """
        Return an opaque token representing the id of this graph's memory pool
        """
        return super().pool()


class graph:
    default_capture_stream: torch.mtia.Stream | None = None

    def __init__(
        self,
        mtia_graph: MTIAGraph,
        pool: _POOL_HANDLE | None = None,
        stream: torch.mtia.Stream | None = None,
    ):
        if self.__class__.default_capture_stream is None:
            self.__class__.default_capture_stream = torch.mtia.current_stream()

        self.pool: tuple[()] | tuple[_POOL_HANDLE] = () if pool is None else (pool,)
        self.capture_stream = (
            stream if stream is not None else self.__class__.default_capture_stream
        )
        if self.capture_stream is None:
            raise AssertionError("capture_stream must not be None")
        self.stream_ctx = torch.mtia.stream(self.capture_stream)
        self.mtia_graph = mtia_graph

    def __enter__(self) -> None:
        torch.mtia.synchronize()
        torch.mtia.empty_cache()

        self.stream_ctx.__enter__()

        pool_arg = self.pool[0] if self.pool else (0, 0)
        self.mtia_graph.capture_begin(pool_arg)

    def __exit__(self, *args: object) -> None:
        self.mtia_graph.capture_end()
        self.stream_ctx.__exit__(*args)


__all__ = [
    "MTIAGraph",
    "graph",
    "graph_pool_handle",
]
<|endoftext|># mypy: allow-untyped-defs
r"""
This package enables an interface for accessing MTIA backend in python
"""

import threading
import traceback
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch._environment import is_fbcode, is_prod
from torch._utils import _LazySeedTracker
from torch.types import Device

from ._utils import _get_device_index


# torch.mtia.Event/Stream is alias of torch.Event/Stream
Event = torch.Event
Stream = torch.Stream

# Default generators are initialized inside _mtia_init
default_generators: tuple[torch._C.Generator, ...] = ()  # type: ignore[assignment]


_initialized = False
_queued_calls: list[
    tuple[Callable[[], None], list[str]]
] = []  # don't invoke these until initialization occurs
_tls = threading.local()
_initialization_lock = threading.Lock()
_lazy_seed_tracker = _LazySeedTracker()


if hasattr(torch._C, "_mtia_exchangeDevice"):
    _exchange_device = torch._C._mtia_exchangeDevice
else:

    def _exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without MTIA support")


if hasattr(torch._C, "_mtia_maybeExchangeDevice"):
    _maybe_exchange_device = torch._C._mtia_maybeExchangeDevice
else:

    def _maybe_exchange_device(device: int) -> int:
        if device < 0:
            return -1
        raise RuntimeError("PyTorch was compiled without MTIA support")


def init():
    _lazy_init()


def is_initialized():
    r"""Return whether PyTorch's MTIA state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs):
    with _initialization_lock:
        if is_initialized():
            return callable()
        else:
            global _lazy_seed_tracker
            if kwargs.get("seed_all", False):
                _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
            elif kwargs.get("seed", False):
                _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
            else:
                _queued_calls.append((callable, traceback.format_stack()))


def _is_in_bad_fork() -> bool:
    return torch._C._mtia_isInBadFork()


def _lazy_init() -> None:
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # We be double-checking locking, boys! This is OK because
        # the above test was GIL protected anyway. The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize MTIA in forked subprocess. To use MTIA with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        if not _is_compiled():
            raise AssertionError(
                "Torch not compiled with MTIA enabled. "
                "Ensure you have `import mtia.host_runtime.torch_mtia.dynamic_library` in your python "
                "src file and include `//mtia/host_runtime/torch_mtia:torch_mtia` as "
                "your target dependency!"
            )

        # Install the C++ resource manager to enable Buck resource lookup from Python.
        # This must be called before _mtia_init() which may access Buck resources.
        if is_fbcode() and is_prod():
            try:
                from libfb.py.cxx_resources import cxx_resource_manager

                cxx_resource_manager.install()
            except ModuleNotFoundError:
                # cxx_resource_manager is not available in all build configurations
                pass

        torch._C._mtia_init()
        # Some of the queued calls may reentrantly call _lazy_init();
        # we need to just return without initializing in that case.
        # However, we must not let any *other* threads in!
        _tls.is_initializing = True

        _queued_calls.extend(calls for calls in _lazy_seed_tracker.get_calls() if calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"MTIA call failed lazily at initialization with error: {str(e)}\n\n"
                        f"MTIA call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise DeferredMtiaCallError(msg) from e
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


class DeferredMtiaCallError(Exception):
    pass


def _is_compiled() -> bool:
    r"""Return true if compiled with MTIA support."""
    return torch._C._mtia_isBuilt()


def is_available() -> bool:
    r"""Return true if MTIA device is available"""
    if not _is_compiled():
        return False
    # MTIA has to init devices first to know if there is any devices available.
    return device_count() > 0


def synchronize(device: Device = None) -> None:
    r"""Waits for all jobs in all streams on a MTIA device to complete."""
    with torch.mtia.device(device):
        return torch._C._mtia_deviceSynchronize()


def device_count() -> int:
    r"""Return the number of MTIA devices available."""
    return torch._C._mtia_getDeviceCount()


def current_device() -> int:
    r"""Return the index of a currently selected device."""
    # pyrefly: ignore [missing-attribute]
    return torch._C._mtia_getDevice()


def current_stream(device: Device = None) -> Stream:
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~torch.mtia.current_device`, if :attr:`device` is ``None``
            (default).
    """
    device_idx = _get_device_index(device, optional=True)
    return torch._C._mtia_getCurrentStream(device_idx)


def default_stream(device: Device = None) -> Stream:
    r"""Return the default :class:`Stream` for a given device.

    Args:
        device (torch.device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~torch.mtia.current_device`, if :attr:`device` is ``None``
            (default).
    """
    return torch._C._mtia_getDefaultStream(_get_device_index(device, optional=True))


def record_memory_history(
    enabled: str | None = "all", stacks: str = "python", max_entries: int = 0
) -> None:
    r"""Enable/Disable the memory profiler on MTIA allocator

    Args:
        enabled (all or state, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).

        stacks ("python" or "cpp", optional). Select the stack trace to record.

        max_entries (int, optional). Maximum number of entries to record.
    """
    if not is_initialized():
        return
    torch._C._mtia_recordMemoryHistory(enabled, stacks, max_entries)


def snapshot() -> dict[str, Any]:
    r"""Return a dictionary of MTIA memory allocator history"""

    return torch._C._mtia_memorySnapshot()


def attach_out_of_memory_observer(
    observer: Callable[[int, int, int, int], None],
) -> None:
    r"""Attach an out-of-memory observer to MTIA memory allocator"""
    torch._C._mtia_attachOutOfMemoryObserver(observer)


def is_bf16_supported(including_emulation: bool = True):
    r"""Return a bool indicating if the current MTIA device supports dtype bfloat16."""
    return True


def get_device_capability(device: Device = None) -> tuple[int, int]:
    r"""Return capability of a given device as a tuple of (major version, minor version).

    Args:
        device (torch.device or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    return torch._C._mtia_getDeviceCapability(_get_device_index(device, optional=True))


def empty_cache() -> None:
    r"""Empty the MTIA device cache."""
    return torch._C._mtia_emptyCache()


def set_stream(stream: Stream):
    r"""Set the current stream. This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    torch._C._mtia_setCurrentStream(stream)


def set_device(device: Device) -> None:
    r"""Set the current device.

    Args:
        device (torch.device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        # pyrefly: ignore [missing-attribute]
        torch._C._mtia_setDevice(device)


def get_device_properties(device: Device = None) -> dict[str, Any]:
    r"""Return a dictionary of MTIA device properties

    Args:
        device (torch.device or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
    return torch._C._mtia_getDeviceProperties(_get_device_index(device, optional=True))


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (torch.device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        # pyrefly: ignore [missing-attribute]
        self.prev_idx = torch._C._mtia_maybeExchangeDevice(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # pyrefly: ignore [missing-attribute]
        self.idx = torch._C._mtia_maybeExchangeDevice(self.prev_idx)
        return False


class StreamContext:
    r"""Context-manager that selects a given stream.

    All MTIA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Stream | None

    def __init__(self, stream: Stream | None):
        self.cur_stream = None
        self.stream = stream
        self.idx = _get_device_index(None, True)
        if not torch.jit.is_scripting():
            if self.idx is None:
                self.idx = -1  # pyrefly: ignore [bad-assignment]

        self.src_prev_stream = (
            None if not torch.jit.is_scripting() else torch.mtia.default_stream(None)
        )
        self.dst_prev_stream = (
            None if not torch.jit.is_scripting() else torch.mtia.default_stream(None)
        )

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or MTIA device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = torch.mtia.current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = torch.mtia.current_stream(cur_stream.device)
        torch.mtia.set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no MTIA device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            torch.mtia.set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        torch.mtia.set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def _set_stream_by_id(stream_id, device_index, device_type):
    r"""set stream specified by the stream id, device index and
        device type

    Args: stream_id (int): stream id in stream pool
          device_index (int): device index in topo
          device_type (int): enum device type
    """
    torch._C._mtia_setStream(stream_id, device_index, device_type)


def stream(stream: Stream | None) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: In eager mode stream is of type Stream class while in JIT it doesn't support torch.mtia.stream
    """
    return StreamContext(stream)


def get_rng_state(device: Device = "mtia") -> Tensor:
    r"""Returns the random number generator state of the specified MTIA device as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'mtia'`` (i.e., ``torch.device('mtia')``, the current mtia device).

    .. warning::
        This function eagerly initializes MTIA.
    """
    _lazy_init()
    idx = _get_device_index(device, optional=True)
    if idx is None:
        idx = current_device()
    default_generator = default_generators[idx]
    return default_generator.get_state()


def get_rng_state_all() -> list[Tensor]:
    r"""Returns a list of ByteTensor representing the random number states of all devices."""
    results = [get_rng_state(i) for i in range(device_count())]
    return results


def set_rng_state(new_state: Tensor, device: Device = "mtia") -> None:
    r"""Sets the random number generator state of the specified MTIA device.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'mtia'`` (i.e., ``torch.device('mtia')``, the current mtia device).
    """
    if not is_initialized():
        with torch._C._DisableFuncTorch():
            # Clone the state because the callback will be triggered
            # later when MTIA is lazy initialized.
            new_state = new_state.clone(memory_format=torch.contiguous_format)

    idx = _get_device_index(device, optional=True)
    if idx is None:
        idx = current_device()

    def cb():
        default_generator = default_generators[idx]
        default_generator.set_state(new_state)

    _lazy_call(cb)


def set_rng_state_all(new_states: list[Tensor]) -> None:
    r"""Sets the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
    for i, state in enumerate(new_states):
        set_rng_state(state, i)


def manual_seed(seed: int) -> None:
    r"""Sets the seed for generating random numbers for the current MTIA device.
    It's safe to call this function if MTIA is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    seed = int(seed)

    def cb():
        idx = current_device()
        default_generator = default_generators[idx]
        default_generator.manual_seed(seed)

    _lazy_call(cb, seed=True)


def manual_seed_all(seed: int) -> None:
    r"""Sets the seed for generating random numbers on all MTIA devices.
    It's safe to call this function if MTIA is not available; in that case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)

    def cb():
        for i in range(device_count()):
            default_generator = default_generators[i]
            default_generator.manual_seed(seed)

    _lazy_call(cb, seed_all=True)


def seed() -> int:
    r"""Sets the seed for generating random numbers to a random number for the current MTIA device.
    It's safe to call this function if MTIA is not available; in that case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """

    def cb():
        idx = current_device()
        default_generator = default_generators[idx]
        return default_generator.seed()

    return _lazy_call(cb, seed=True)


def seed_all() -> int:
    r"""Sets the seed for generating random numbers to a random number on all MTIA devices.

    It's safe to call this function if MTIA is not available; in that case, it is silently ignored.
    """

    def cb():
        random_seed = 0
        seeded = False
        for i in range(device_count()):
            default_generator = default_generators[i]
            if not seeded:
                random_seed = default_generator.seed()
                seeded = True
            else:
                default_generator.manual_seed(random_seed)
        return random_seed

    return _lazy_call(cb, seed_all=True)


def initial_seed() -> int:
    r"""Returns the current random seed of the current MTIA device.

    .. warning::
        This function eagerly initializes MTIA.
    """
    _lazy_init()
    idx = current_device()
    return default_generators[idx].initial_seed()


from .memory import *  # noqa: F403
from .mtia_graph import *  # noqa: F403


__all__ = [
    "init",
    "is_available",
    "is_initialized",
    "synchronize",
    "device_count",
    "current_device",
    "current_stream",
    "default_stream",
    "memory_stats",
    "max_memory_allocated",
    "memory_allocated",
    "reset_peak_memory_stats",
    "get_device_capability",
    "get_device_properties",
    "record_memory_history",
    "snapshot",
    "attach_out_of_memory_observer",
    "empty_cache",
    "set_device",
    "set_stream",
    "stream",
    "device",
    "set_rng_state",
    "get_rng_state",
    "set_rng_state_all",
    "get_rng_state_all",
    "manual_seed",
    "manual_seed_all",
    "seed",
    "seed_all",
    "initial_seed",
    "is_bf16_supported",
    "MTIAGraph",
    "graph",
    "graph_pool_handle",
]
<|endoftext|>from typing import Any

import torch


def _get_device_index(
    device: Any, optional: bool = False, allow_cpu: bool = False
) -> int:
    r"""Get the device index from :attr:`device`, which can be a torch.device object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a MTIA device. Note that for a MTIA device without a specified index,
    i.e., ``torch.device('mtia')``, this will return the current default MTIA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default MTIA
    device if :attr:`optional` is ``True``.
    """

    if device is None and optional:
        # If device is None (frequent), then we can short-circuit the logic
        return torch._C._mtia_getDevice()
    if isinstance(device, int):
        return device
    if not torch.jit.is_scripting():
        if isinstance(device, torch.mtia.device):
            return device.idx
    if isinstance(device, str):
        device = torch.device(device)
    device_idx: int | None = None
    if isinstance(device, torch.device):
        if not allow_cpu and device.type == "cpu":
            raise ValueError(f"Expected a non cpu device, but got: {device}")
        if device.type not in ["mtia", "cpu"]:
            raise ValueError(f"Expected a mtia or cpu device, but got: {device}")
        device_idx = -1 if device.type == "cpu" else device.index
    if device_idx is None:
        if optional:
            device_idx = torch._C._mtia_getDevice()
        else:
            raise ValueError(
                f"Expected a torch.device with a specified index or an integer, but got: {device}"
            )
    return device_idx
<|endoftext|># mypy: allow-untyped-defs
"""
State dict utilities for torch.export.

This module provides utilities for restoring state dicts to traced modules,
ensuring that FQNs (Fully Qualified Names) match the original module structure.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.fx


def _get_underlying_module(
    module_or_method: torch.nn.Module | Callable[..., Any],
) -> torch.nn.Module:
    """Extract the underlying nn.Module from either a module or a bound method.

    Args:
        module_or_method: Either an nn.Module or a bound method of an nn.Module.

    Returns:
        The underlying nn.Module.

    Raises:
        TypeError: If module_or_method is neither an nn.Module nor a bound method.
    """
    if isinstance(module_or_method, torch.nn.Module):
        return module_or_method
    # Handle bound methods (e.g., module.method)
    if (
        mod_self := getattr(module_or_method, "__self__", None)
    ) is not None and isinstance(mod_self, torch.nn.Module):
        return mod_self
    raise TypeError(
        f"Expected nn.Module or bound method of nn.Module, got {type(module_or_method)}"
    )


def _clear_traced_params_buffers(
    traced_module: torch.fx.GraphModule, const_keys: Sequence[str]
) -> None:
    """Remove all parameters and buffers from traced module before restoring.

    For constants (parameters/buffers that don't need FQN mapping), this function
    removes them from the _buffers dict and re-assigns them as direct attributes.
    This ensures constants don't show up as buffers in the state dict.

    Args:
        traced_module: The traced GraphModule to clean up.
        const_keys: List of keys that represent constants to be cleared.
    """
    for key in const_keys:
        if key not in traced_module._buffers:
            raise AssertionError(f"Key {key} not found in traced_module._buffers")
        # We don't want constants to show up as a buffer in the state dict.
        # Instead they should just be a direct attribute.
        buffer = traced_module._buffers[key]
        del traced_module._buffers[key]
        # Note: setattr will register the value per nn.Module rules:
        # - If it's a Tensor, it'll be re-registered as a buffer (ends up back in _buffers).
        # - Otherwise, it becomes a plain attribute (not part of state_dict).
        setattr(traced_module, key, buffer)


def _restore_state_dict(
    original_module: torch.nn.Module | Callable[..., Any],
    traced_module: torch.fx.GraphModule,
) -> None:
    """
    Restores the state dict of the traced module to match the original module exactly.

    This function ensures that:
    1. Parameters and buffers in the traced module use the same FQNs (Fully Qualified Names)
       as the original module.
    2. The ordering of parameters/buffers matches the original module.
    3. Graph nodes referencing the old names are updated to use the correct FQNs.

    This is useful after using functional tracing APIs (like dynamo_graph_capture_for_export)
    that may flatten parameter/buffer names.

    Args:
        original_module: The original nn.Module (or a bound method of one) that was traced.
        traced_module: The traced fx.GraphModule whose state dict needs to be restored.

    Example::

        import torch
        from torch._dynamo.functional_export import _dynamo_graph_capture_for_export
        from torch.export import _restore_state_dict


        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.layer(x)


        model = Model()
        gm = _dynamo_graph_capture_for_export(model)(torch.randn(1, 10))

        # Before: gm may have flattened names like "p_layer_weight"
        # After: gm will have proper FQNs like "layer.weight"
        _restore_state_dict(model, gm)
    """
    # Extract the underlying module if a bound method was passed
    module = _get_underlying_module(original_module)

    # Build ID-based lookups for traced module params/buffers
    # Collect all data first to avoid modifying during iteration
    traced_params: dict[int, tuple[str, torch.nn.Parameter]] = {}
    for name, param in traced_module.named_parameters(remove_duplicate=False):
        traced_params[id(param)] = (name, param)

    traced_buffers: dict[int, tuple[str, torch.Tensor]] = {}
    for name, buffer in traced_module.named_buffers(remove_duplicate=False):
        traced_buffers[id(buffer)] = (name, buffer)

    # Collect original module's parameters and buffers upfront to avoid
    # issues with shared tensor objects during iteration
    orig_params_list: list[tuple[str, torch.nn.Parameter]] = list(
        module.named_parameters(remove_duplicate=False)
    )
    orig_buffers_list: list[tuple[str, torch.Tensor]] = list(
        module.named_buffers(remove_duplicate=False)
    )

    # Build mapping from old names to new names for graph node updates
    name_mapping: dict[str, str] = {}

    # Track which traced names we've processed
    processed_traced_names: set[str] = set()

    # Restore parameters in the order they appear in original module
    for orig_name, orig_param in orig_params_list:
        if id(orig_param) in traced_params:
            # This param exists in traced module - restore it with original FQN
            traced_name, traced_param = traced_params[id(orig_param)]
            processed_traced_names.add(traced_name)
            if traced_name != orig_name:
                # Only reassign if the name is different
                torch.fx.graph_module._assign_attr(
                    traced_param, traced_module, orig_name
                )
                torch.fx.graph_module._del_attr(traced_module, traced_name)
                name_mapping[traced_name] = orig_name
        else:
            # This param doesn't exist in traced module - add it
            torch.fx.graph_module._assign_attr(orig_param, traced_module, orig_name)

    # Restore buffers in the order they appear in original module
    for orig_name, orig_buffer in orig_buffers_list:
        if id(orig_buffer) in traced_buffers:
            # This buffer exists in traced module - restore it with original FQN
            traced_name, traced_buffer = traced_buffers[id(orig_buffer)]
            processed_traced_names.add(traced_name)
            if traced_name != orig_name:
                # Only reassign if the name is different
                torch.fx.graph_module._assign_attr(
                    orig_buffer, traced_module, orig_name
                )
                torch.fx.graph_module._del_attr(traced_module, traced_name)
                name_mapping[traced_name] = orig_name
        else:
            # This buffer doesn't exist in traced module - add it
            torch.fx.graph_module._assign_attr(orig_buffer, traced_module, orig_name)

    param_names = [v[0] for v in traced_params.values()]
    buffer_names = [v[0] for v in traced_buffers.values()]
    # Constants are traced params/buffers that weren't matched to any original param/buffer
    const_keys = list(
        set(param_names + buffer_names).difference(processed_traced_names)
    )

    _clear_traced_params_buffers(traced_module, const_keys)

    # Update get_attr nodes in the graph to use the correct FQNs
    for node in traced_module.graph.nodes:
        if node.op == "get_attr" and node.target in name_mapping:
            node.target = name_mapping[node.target]

    traced_module.recompile()
<|endoftext|># mypy: allow-untyped-defs
import dataclasses
import inspect
import logging
import sys
from collections import defaultdict
from collections.abc import Callable
from enum import auto, Enum
from typing import Any, TYPE_CHECKING, Union

import torch
from torch.utils._pytree import (
    _get_node_type,
    BUILTIN_TYPES,
    KeyPath,
    keystr,
    MappingKey,
    SequenceKey,
    SUPPORTED_NODES,
    tree_iter,
    tree_map,
    tree_map_with_path,
    tree_structure,
    TreeSpec,
)

from .exported_program import ExportedProgram


if TYPE_CHECKING:
    from sympy import Symbol

    from torch._guards import Source
    from torch.fx.experimental.symbolic_shapes import ShapeEnv, StrictMinMaxConstraint

__all__ = [
    "Constraint",
    "Dim",
    "dims",
    "refine_dynamic_shapes_from_suggested_fixes",
    "AdditionalInputs",
]


log = logging.getLogger(__name__)


class _DimHintType(Enum):
    """
    Enum for dynamic shape hints.
    - AUTO means automatic inference of shape (static or dynamic).
    - STATIC means static shape (always specialized).
    - DYNAMIC means dynamic, will error out if specialized.
    """

    AUTO = auto()
    STATIC = auto()
    DYNAMIC = auto()


@dataclasses.dataclass
class _DimHint:
    """
    Internal class for dynamic shape hints.
    - min and max are optional.
    - _factory is for UX only, below example:
        auto_hint = _DimHint.AUTO()  # _factory=True
        bounded_hint = auto_hint(min=10, max=100)  # Returns new instance with _factory=False
        bounded_hint(min=5, max=50)  # Will fail, non-factory instance cannot be called
    """

    type: _DimHintType
    min: int | None = None
    max: int | None = None
    _factory: bool | None = True

    @staticmethod
    def AUTO():
        return _DimHint(_DimHintType.AUTO)

    @staticmethod
    def DYNAMIC():
        return _DimHint(_DimHintType.DYNAMIC)

    @staticmethod
    def STATIC():
        return _DimHint(_DimHintType.STATIC)

    def __call__(self, min=None, max=None) -> "_DimHint":
        if not self._factory:
            raise TypeError(f"'{type(self)}' object is not callable")
        if min is not None and min < 0:
            raise AssertionError(f"min must be non-negative, got {min}")
        if max is not None and max < 0:
            raise AssertionError(f"max must be non-negative, got {max}")
        if min is not None and max is not None and min > max:
            raise AssertionError(f"min must be <= max, got min={min}, max={max}")
        return _DimHint(self.type, min=min, max=max, _factory=False)

    def __repr__(self):
        parts = [self.type.name]
        if self.min is not None:
            parts.append(f"min={self.min}")
        if self.max is not None:
            parts.append(f"max={self.max}")
        return f"DimHint({', '.join(parts)})"


class Dim:
    """
    The ``Dim`` class allows users to specify dynamism in their exported
    programs. By marking a dimension with a ``Dim``, the compiler associates the
    dimension with a symbolic integer containing a dynamic range.

    The API can be used in 2 ways: Dim hints (i.e. automatic dynamic shapes:
    ``Dim.AUTO``, ``Dim.DYNAMIC``, ``Dim.STATIC``), or named Dims (i.e.
    ``Dim("name", min=1, max=2)``).

    Dim hints provide the lowest barrier to exportability, with the user only
    needing to specify if a dimension is dynamic, static, or left for the
    compiler to decide (``Dim.AUTO``). The export process will automatically
    infer the remaining constraints on min/max ranges and relationships between
    dimensions.

    Example::

        class Foo(nn.Module):
            def forward(self, x, y):
                assert x.shape[0] == 4
                assert y.shape[0] >= 16
                return x @ y


        x = torch.randn(4, 8)
        y = torch.randn(8, 16)
        dynamic_shapes = {
            "x": {0: Dim.AUTO, 1: Dim.AUTO},
            "y": {0: Dim.AUTO, 1: Dim.AUTO},
        }
        ep = torch.export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)

    Here, export would raise an exception if we replaced all uses of ``Dim.AUTO`` with ``Dim.DYNAMIC``,
    as ``x.shape[0]`` is constrained to be static by the model.

    More complex relations between dimensions may also be codegened as runtime assertion nodes by the compiler,
    e.g. ``(x.shape[0] + y.shape[1]) % 4 == 0``, to be raised if runtime inputs do not satisfy such constraints.

    You may also specify min-max bounds for Dim hints, e.g. ``Dim.AUTO(min=16, max=32)``, ``Dim.DYNAMIC(max=64)``,
    with the compiler inferring the remaining constraints within the ranges. An exception will be raised if
    the valid range is entirely outside the user-specified range.

    Named Dims provide a stricter way of specifying dynamism, where exceptions are raised if the compiler
    infers constraints that do not match the user specification. For example, exporting the previous
    model, the user would need the following ``dynamic_shapes`` argument::

        s0 = Dim("s0")
        s1 = Dim("s1", min=16)
        dynamic_shapes = {
            "x": {0: 4, 1: s0},
            "y": {0: s0, 1: s1},
        }
        ep = torch.export(Foo(), (x, y), dynamic_shapes=dynamic_shapes)

    Named Dims also allow specification of relationships between dimensions, up
    to univariate linear relations.  For example, the following indicates one
    dimension is a multiple of another plus 4::

        s0 = Dim("s0")
        s1 = 3 * s0 + 4

    """

    AUTO = _DimHint.AUTO()
    DYNAMIC = _DimHint.DYNAMIC()
    STATIC = _DimHint.STATIC()

    def __init__(self, name: str, *, min: int | None = None, max: int | None = None):
        from torch.utils._sympy.numbers import int_oo

        _min = 0 if min is None else min
        _max = int_oo if max is None else max
        if not (_max > _min):
            raise AssertionError(
                f"Cannot create Dim with inconsistent min={min}, max={max}"
            )
        if not name.isidentifier():
            raise AssertionError(f"Dim name must be a valid identifier, got {name}")
        self.__name__ = name
        self.min = _min
        self.max = _max

    def __add__(self, other) -> "Dim":
        # e.g., dim + 1
        if type(other) is not int:
            raise NotImplementedError(
                f"Attempted to add {other} to {self.__name__}, where an integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return self._derive(lambda x: x + other)

    def __radd__(self, other) -> "Dim":
        return self + other

    def __sub__(self, other) -> "Dim":
        # e.g., dim - 1
        if type(other) is not int:
            raise NotImplementedError(
                f"Attempted to subtract {other} from {self.__name__}, where an integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return self._derive(lambda x: x - other)

    def __rsub__(self, other) -> "Dim":
        raise NotImplementedError(
            f"Attempted to negate {self.__name__}. "
            "(Only increasing linear operations with integer coefficients are supported.)"
        )

    def __mul__(self, other) -> "Dim":
        # e.g., dim * 2
        if type(other) is not int or other <= 0:
            raise NotImplementedError(
                f"Attempted to multiply {other} with {self.__name__}, where a positive integer was expected. "
                "(Only increasing linear operations with integer coefficients are supported.)"
            )
        return self._derive(lambda x: x * other)

    def __rmul__(self, other) -> "Dim":
        return self * other

    def _derived_name(self, fn) -> str:
        from sympy import sympify

        return str(fn(sympify(self.__name__)))

    def _derive(self, fn) -> "Dim":
        return _DerivedDim(self._derived_name(fn), self, fn)

    @staticmethod
    def _readable(name: str, min_: int, max_: int) -> str:
        from torch.utils._sympy.numbers import int_oo

        if min_ == 2:
            min_ = None  # type: ignore[assignment]
        if max_ == int_oo:
            max_ = None  # type: ignore[assignment]
        if min_ is None and max_ is None:
            return f"Dim('{name}')"
        if min_ is None:
            return f"Dim('{name}', max={max_})"
        if max_ is None:
            return f"Dim('{name}', min={min_})"
        return f"Dim('{name}', min={min_}, max={max_})"

    def __repr__(self):
        return Dim._readable(self.__name__, self.min, self.max)


_Dim = Dim  # TODO(pianpwk): remove after it's no longer internally breaking


class _StaticDim(Dim):
    """
    Class for static :func:`Dim` types.

    This class is only for setting and checking static dim constraints,
    and the user should never interact with it.
    """

    def __init__(self, value: int):
        self.__name__ = str(value)
        self.value = value

    @property
    def min(self):  # type: ignore[override]
        return self.value  # type: ignore[attr-defined]

    @property
    def max(self):  # type: ignore[override]
        return self.value  # type: ignore[attr-defined]


class _DerivedDim(Dim):
    """
    Class for derived :func:`Dim` types.

    Currently we only support increasing linear expressions with integer coefficients.
    In other words, a derived Dim can always be written in the form Ax + B, where
    x is a regular Dim (i.e., non-derived Dim), A and B are integers, and A is positive.
    (In particular, the latter ensures that x < y => Ax + B < Ay + B.)
    These restrictions on the form of derived Dims makes the metatheory simpler: e.g.,
    it simplifies computing ranges for derived Dims, solving for underlying regular Dims,
    deciding equalities between derived Dims, and so on.

    The function lambda x: Ax + B is expressed by `fn`, where x is a normal Dim, `root`.
    The range of a derived Dim is computed by mapping `fn` over the range of its `root`.
    """

    def __init__(self, name: str, root: Dim, fn: Callable):
        self.__name__ = name
        self.root = root
        self.fn = fn

    @property
    def min(self):  # type: ignore[override]
        # assume that self.fn is an increasing function
        # TODO(avik): use sympy value range analysis instead?
        from sympy import Integer

        from torch.utils._sympy.numbers import int_oo

        if self.root.min is -int_oo:  # type: ignore[attr-defined]
            return -int_oo  # fn not needed cuz increasing

        _min_symint = self.fn(Integer(self.root.min))  # type: ignore[attr-defined]
        root = self.root  # type: ignore[attr-defined]
        if _min_symint < 0:
            raise AssertionError(
                f"Expected derived min value of {self.__name__} to be >= 0. "
                f"Please specify an appropriate min value for {root.__name__} "
                f"(currently {root.min})."
            )
        return int(_min_symint)

    @property
    def max(self):  # type: ignore[override]
        # assume that self.fn is an increasing function
        # TODO(avik): use sympy value range analysis instead?
        from sympy import Integer

        from torch.utils._sympy.numbers import int_oo

        if self.root.max is int_oo:  # type: ignore[attr-defined]
            return int_oo  # fn not needed cuz increasing

        _max_symint = self.fn(Integer(self.root.max))  # type: ignore[attr-defined]
        root = self.root  # type: ignore[attr-defined]
        if _max_symint > sys.maxsize - 1:
            raise AssertionError(
                f"Expected derived max value of {self.__name__} to be <= {sys.maxsize - 1}. "
                f"Please specify an appropriate max value for {root.__name__} "
                f"(currently {root.max})."
            )
        return int(_max_symint)

    def _derive(self, fn):
        # We support nesting, e.g., 2*dim + 1.
        # This is implemented by composing operations on the same root.
        # As a consequence, roots are always regular Dims (i.e., not derived Dims).
        return _DerivedDim(
            self._derived_name(fn),
            self.root,
            lambda x: fn(self.fn(x)),
        )

    def __repr__(self):
        return self.__name__


def dims(
    *names: str, min: int | None = None, max: int | None = None
) -> tuple[Dim, ...]:
    """
    Util to create multiple :func:`Dim` types.

    Returns:
        A tuple of :func:`Dim` types.
    """
    return tuple(Dim(name, min=min, max=max) for name in names)  # type: ignore[misc]


@dataclasses.dataclass
class _ConstraintTarget:
    """
    This represents input tensor dimensions.
    """

    t_id: int
    dim: int


@dataclasses.dataclass
class _Constraint(_ConstraintTarget):
    """
    This represents a Dim describing a constraint target.

    `name` is the name of the Dim.
    `constraint_range` contains the min/max bounds of the Dim.
    """

    name: str
    constraint_range: "StrictMinMaxConstraint"

    def _clone_with_range(self, lower=0, upper=None):
        # Import sympy locally
        from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
        from torch.utils._sympy.numbers import int_oo
        from torch.utils._sympy.value_ranges import ValueRanges

        if upper is None:
            upper = int_oo

        constraint_range = StrictMinMaxConstraint(
            vr=self.constraint_range.vr & ValueRanges(lower=lower, upper=upper),
            warn_only=False,
        )
        return _Constraint(
            self.t_id,
            self.dim,
            self.name,
            constraint_range,
        )

    def __ge__(self, lower):
        return self._clone_with_range(lower=lower)

    def __gt__(self, lower):
        return self._clone_with_range(lower=lower + 1)

    def __le__(self, upper):
        return self._clone_with_range(upper=upper)

    def __lt__(self, upper):
        return self._clone_with_range(upper=upper - 1)

    def __bool__(self):
        # NOTE(avik): We do not support compound expressions like a <= x <= b.
        # This is because Python implicitly desugars them into bool(a <= x) and bool(x <= b),
        # and moreover, enforces that any overload of __bool__ must return True or False.
        # FWIW, sympy also raises TypeError in this case.
        raise TypeError(
            "Cannot determine truth value of _Constraint. "
            "If you are trying to combine _Constraint's with logical connectives, "
            "you can specify them separately instead."
        )

    @property
    def serializable_spec(self):
        # We need a serialization compatible format of the constraint so that it
        # can be savedin the graph module w/o breaking the module serialization.
        # The saved constraints will be used directly for the post-exporting pass
        # that converts constraints to runtime assertion. The saved constraints
        # will not be saved in the serialized module.
        # TODO: A better way is needed. Currently we use 't_id' to map the constraint,
        # which is not reliable
        return {
            "t_id": self.t_id,
            "dim": self.dim,
            "min": self.constraint_range.vr.lower,
            "max": self.constraint_range.vr.upper,
        }


@dataclasses.dataclass
class _PhantomRoot:
    """
    This represents the root of a derived Dim where the root does not directly
    specify the shape of any input dimension, but the derived Dim does.

    e.g., the input shapes 2*dim and dim + 1 are related via a "phantom" dim.

    The fields `name`, `constraint_range`, and `val` carried by a phantom root
    help create a symbol for it. Any derived dims with this phantom root are
    backed by expressions over this symbol.
    """

    name: str
    constraint_range: "StrictMinMaxConstraint"
    val: int


@dataclasses.dataclass
class _DerivedConstraint(_ConstraintTarget):
    """
    This represents a derived Dim, whose root is either a regular constraint target
    (which directly specifies the shape of some input dimension) or a phantom root
    (which does so indirectly).

    It can be thought of as a subclass of `_Constraint`, except that it does not
    support <, <=, >, >= operations.
    """

    name: str
    constraint_range: "StrictMinMaxConstraint"
    root: _ConstraintTarget | _PhantomRoot
    fn: Callable

    @property
    def serializable_spec(self):
        # same as _Constraint.serializable_spec
        return {
            "t_id": self.t_id,
            "dim": self.dim,
            "min": self.constraint_range.vr.lower,
            "max": self.constraint_range.vr.upper,
        }


@dataclasses.dataclass
class _RelaxedConstraint(_ConstraintTarget):
    """
    This represents a dim marked with Dim.AUTO/DYNAMIC (i.e. mark_dynamic() or maybe_mark_dynamic()),
    which leaves relations & min/max ranges for inference, instead of requiring explicit specification.
    The intention is for constraint violations to not be raised if produce_guards() finds equalities or
    relations between a _RelaxedConstraint and another type of _Constraint.
    """

    @property
    def serializable_spec(self):
        return {
            "t_id": self.t_id,
            "dim": self.dim,
        }


Constraint = _Constraint | _DerivedConstraint | _RelaxedConstraint


@dataclasses.dataclass
class _IntWrapper:
    """
    Dummy wrapper class to wrap around integer inputs so that when we parse the
    dynamic_shapes structure, we can mark if any of the integers were marked as
    dynamic.
    """

    val: int
    # Disallow specifying dynamism
    dynamism: _DimHint | int | None = dataclasses.field(init=False, default=None)


def _process_equalities(
    constraint: Constraint,
    get_sources: Callable[[int, int], list["Source"]],
    shape_env: "ShapeEnv",
    names: dict[str, tuple[int, int]],
    source_pairs: list[tuple["Source", "Source"]],
    derived_equalities: list[tuple["Source", Union["Source", "Symbol"], Callable]],
    phantom_symbols: dict[str, "Symbol"],
    relaxed_sources: set["Source"],
):
    """
    Updates `source_pairs`, `derived_equalities`, and `phantom_symbols` (which become
    fields of `EqualityConstraint`) based on a given input `constraint`.
    """

    sources = get_sources(constraint.t_id, constraint.dim)
    if not sources:  # empty sources due to unused shapes
        return

    source, *other_sources = sources
    # When t.size()[dim] maps to src0, src1, ..., srcN, we add
    # constraints that make src0 "equal" to src1, ..., srcN.
    source_pairs.extend((source, other_source) for other_source in other_sources)
    if isinstance(constraint, _Constraint):
        if constraint.name in names:
            shared_t_id, shared_dim = names[constraint.name]
            other_sources = get_sources(shared_t_id, shared_dim)
            source_pairs.extend(
                (source, other_source) for other_source in other_sources
            )
        else:
            names[constraint.name] = (constraint.t_id, constraint.dim)
    elif isinstance(constraint, _DerivedConstraint):
        # branch based on the root of the _DerivedConstraint
        if not isinstance(constraint.root, _PhantomRoot):
            # either root points to an input source
            root = get_sources(constraint.root.t_id, constraint.root.dim)[0]
        else:
            # or root points to a phantom symbol
            if constraint.root.name in phantom_symbols:
                root = phantom_symbols[constraint.root.name]
            else:
                # create a phantom symbol in the shape env based on the _PhantomRoot
                root = shape_env.create_symbol(
                    val=constraint.root.val,
                    source=torch._dynamo.source.ConstantSource(constraint.root.name),
                    dynamic_dim=torch.fx.experimental.symbolic_shapes.DimDynamic.DYNAMIC,
                    constraint_dim=constraint.root.constraint_range,
                )
                phantom_symbols[constraint.root.name] = root

        fn = constraint.fn
        # A derived equality (source, root, fn) informally corresponds to source = fn(root).
        # Here source describes an input and root might describe another input or a phantom symbol.
        derived_equalities.append((source, root, fn))
    elif isinstance(constraint, _RelaxedConstraint):
        relaxed_sources.add(source)


def _tree_map_with_path(
    func: Callable[..., Any],
    tree: Any,
    *dynamic_shapes: Any,
    tree_name: str | None = None,
) -> Any:
    """
    Customized tree_map for mapping pytrees to dynamic_shapes.

    For built-in types (e.g., standard collections) this behaves exactly like tree_map.

    OTOH for a user-defined class C registered with pytree, we cannot assume that a C
    containing tensors can be mapped to a C containing dynamic shapes (i.e., C may not
    be a polymorphic container). In that case we use the flattened form of C instead.
    Thus a C(**tensors) that flattens to (**tensors) will map to (**dynamic_shapes).

    Args:
        func: function to apply to each (int, float, str, bool, None, torch.Tensor)
        tree: input pytree
        dynamic_shapes: zero or more (typically one) dynamic_shapes to match

    Returns:
        output pytree mapping func to each (int, float, str, bool, None, torch.Tensor)
    """

    def is_leaf(t):
        # BUILTIN_TYPES is a subset of SUPPORTED_NODES, the latter being all types
        # registered with pytree. Types *not* in BUILTIN_TYPES include primitive types
        # (int, float, str, bool, None, torch.Tensor), which are not in SUPPORTED_NODES,
        # as well as user-defined classes registered with pytree, which are.
        return _get_node_type(t) not in BUILTIN_TYPES

    def f(path, t, *dynamic_shapes):
        typ = _get_node_type(t)
        # typ is not in BUILTIN_TYPES
        if typ in SUPPORTED_NODES:
            # thus typ is a user-defined class registered with pytree,
            # in which case flatten and recurse
            return tree_map_with_path(
                f,
                SUPPORTED_NODES[typ].flatten_fn(t)[0],
                *dynamic_shapes,
                is_leaf=is_leaf,
            )
        else:
            return func(path, t, *dynamic_shapes)

    try:
        return tree_map_with_path(f, tree, *dynamic_shapes, is_leaf=is_leaf)
    except ValueError as e:
        if "mismatch" in e.args[0]:
            # When PyTree finds a structural mismatch between tree and dynamic_shapes,
            # the error message is unfortunately quite horrible. Let's fix that.
            if not dynamic_shapes:
                raise AssertionError(
                    "Cannot be a mismatch if there is no dynamic_shapes"
                ) from None
            if not tree_name:
                raise AssertionError(
                    "Must provide a tree_name when there might be a mismatch"
                ) from None

            def _key(type_, context, i):
                # derive a PyTree key given the type, context, and child # of a TreeSpec
                if type_ is dict:
                    return MappingKey(context[i])
                if type_ in (list, tuple):
                    if context is not None:
                        raise AssertionError(
                            f"expected context to be None for type {type_}, got {context}"
                        )
                    return SequenceKey(i)
                raise AssertionError(f"Did not expect type {type_}")

            def raise_mismatch_error(msg):
                from torch._dynamo.exc import UserError, UserErrorType

                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Detected mismatch between the structure of `{tree_name}` and `dynamic_shapes`: {msg}",
                    case_name="dynamic_shapes_validation",
                )

            def _compare(
                treespec: TreeSpec, other_treespec: TreeSpec, path: KeyPath
            ) -> None:
                # raise an error at the point where tree and dynamic_shapes differ,
                # including the path to that point and the reason for the difference
                rendered_path = keystr(path)
                if treespec.is_leaf():
                    return
                if other_treespec.is_leaf():
                    raise_mismatch_error(
                        f"`{tree_name}{rendered_path}` is a {treespec.type}, "
                        f"but `dynamic_shapes{rendered_path}` is not"
                    )
                if treespec.type != other_treespec.type:
                    raise_mismatch_error(
                        f"`{tree_name}{rendered_path}` is a {treespec.type}, "
                        f"but `dynamic_shapes{rendered_path}` is a {other_treespec.type}"
                    )
                if treespec.num_children != other_treespec.num_children:
                    raise_mismatch_error(
                        f"`{tree_name}{rendered_path}` has {treespec.num_children} elements, "
                        f"but `dynamic_shapes{rendered_path}` has {other_treespec.num_children} elements"
                    )
                if treespec.type is dict:
                    # context, children could be out of order
                    if set(treespec.context) != set(other_treespec.context):
                        raise_mismatch_error(
                            f"`{tree_name}{rendered_path}` has keys {treespec.context}, "
                            f"but `dynamic_shapes{rendered_path}` has keys {other_treespec.context}"
                        )
                    _remap = dict(
                        zip(other_treespec.context, other_treespec.children())
                    )
                    other_children = [_remap[k] for k in treespec.context]
                else:
                    other_children = other_treespec.children()
                for i, (child, other_child) in enumerate(
                    zip(treespec.children(), other_children)
                ):
                    _compare(
                        child,
                        other_child,
                        path + (_key(treespec.type, treespec.context, i),),
                    )

            treespec = tree_structure(tree, is_leaf=is_leaf)
            for other_tree in dynamic_shapes:
                other_treespec = tree_structure(other_tree, is_leaf)
                _compare(treespec, other_treespec, ())
        raise


_DynamicShapesSpec = dict[str, Any] | tuple[Any, ...] | list[Any]


def _signature(
    f: ExportedProgram | torch.nn.Module | Callable[..., Any],
) -> inspect.Signature:
    if isinstance(f, ExportedProgram):
        f = f.module()
    return (
        inspect.signature(f.forward)
        if isinstance(f, torch.nn.Module)
        else inspect.signature(f)
    )


def _combine_args_from_signature(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    kwargs = kwargs if kwargs is not None else {}
    return signature.bind(*args, **kwargs).arguments


def _combine_args(
    f: ExportedProgram | torch.nn.Module | Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    # combine args and kwargs following the signature of f, as it happens
    # in the body of f when called with *args, **kwargs
    return _combine_args_from_signature(_signature(f), args, kwargs)


_MISSING = object()


def _var_keyword_param_name(signature: inspect.Signature) -> str | None:
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return name
    return None


def _variadic_kwargs_info(
    f: ExportedProgram | torch.nn.Module | Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> tuple[inspect.Signature, dict[str, Any], str | None, dict[str, Any]]:
    signature = _signature(f)
    combined_args = _combine_args_from_signature(signature, args, kwargs)
    var_keyword_name = _var_keyword_param_name(signature)
    if var_keyword_name is None or var_keyword_name not in combined_args:
        return signature, combined_args, var_keyword_name, {}

    var_kwargs = combined_args[var_keyword_name]
    if not isinstance(var_kwargs, dict):
        return signature, combined_args, var_keyword_name, {}
    return signature, combined_args, var_keyword_name, var_kwargs


def _colliding_variadic_kwarg_names(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    var_kwargs: dict[str, Any],
) -> list[str]:
    positional_names = set()
    arg_i = 0
    for param in signature.parameters.values():
        if arg_i == len(args):
            break
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            positional_names.update(
                f"{param.name}_{i}" for i in range(len(args) - arg_i)
            )
            break
        if param.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            positional_names.add(param.name)
            arg_i += 1
        elif param.kind == inspect.Parameter.KEYWORD_ONLY:
            break
    return sorted(set(var_kwargs) & positional_names)


def _normalize_dynamic_shapes(
    dynamic_shapes: _DynamicShapesSpec | None,
    f: ExportedProgram | torch.nn.Module | Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
) -> _DynamicShapesSpec | None:
    """
    Normalize call-like dynamic shape specs for **kwargs to the signature-shaped
    structure used by the rest of dynamic shapes processing.
    """
    if not dynamic_shapes:
        return dynamic_shapes

    signature, combined_args, var_keyword_name, var_kwargs = _variadic_kwargs_info(
        f, args, kwargs
    )
    if var_keyword_name is None or not var_kwargs:
        return dynamic_shapes

    from torch._dynamo.exc import UserError, UserErrorType

    collisions = _colliding_variadic_kwarg_names(signature, args, var_kwargs)
    if collisions:
        raise UserError(
            UserErrorType.INVALID_INPUT,
            "Cannot represent dynamic shapes for variadic keyword argument(s) "
            f"{collisions} because they collide with another input name used "
            "for positional inputs.",
            case_name="dynamic_shapes_validation",
        )

    if not isinstance(dynamic_shapes, dict):
        return dynamic_shapes

    result = dict(dynamic_shapes)
    nested = result.get(var_keyword_name, _MISSING)
    nested_shapes = dict(nested) if isinstance(nested, dict) else nested
    fixed_arg_names = set(combined_args) - {var_keyword_name}

    for key in var_kwargs:
        has_nested_shape = isinstance(nested_shapes, dict) and key in nested_shapes
        if key in fixed_arg_names or key == var_keyword_name:
            if key in result and not has_nested_shape:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Cannot represent dynamic shape for variadic keyword argument "
                    f"'{key}' as a top-level key because it collides with another "
                    f"input name. Specify it under the variadic keyword parameter "
                    f"instead, e.g. `dynamic_shapes['{var_keyword_name}']['{key}']`.",
                    case_name="dynamic_shapes_validation",
                )
            continue

        if key in result:
            if nested is not _MISSING and not isinstance(nested_shapes, dict):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Cannot merge dynamic shape for variadic keyword argument "
                    f"'{key}' into `dynamic_shapes['{var_keyword_name}']` because "
                    "that entry is not a dict.",
                    case_name="dynamic_shapes_validation",
                )
            if has_nested_shape:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Found dynamic shape for variadic keyword argument '{key}' "
                    f"both as a top-level key and under "
                    f"`dynamic_shapes['{var_keyword_name}']`.",
                    case_name="dynamic_shapes_validation",
                )
            if not isinstance(nested_shapes, dict):
                nested_shapes = {}
            nested_shapes[key] = result.pop(key)

    if isinstance(nested_shapes, dict):
        result[var_keyword_name] = nested_shapes
    return result


def _combine_args_for_tracing(
    f: ExportedProgram | torch.nn.Module | Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    dynamic_shapes: Any,
) -> tuple[dict[Any, Any], Any]:
    if not dynamic_shapes:
        return _combine_args(f, args, kwargs), dynamic_shapes

    kwargs = kwargs if kwargs is not None else {}
    signature, combined_args, var_keyword_name, var_kwargs = _variadic_kwargs_info(
        f, args, kwargs
    )
    if isinstance(dynamic_shapes, (tuple, list)):
        dynamic_shapes_by_name = dict(zip(combined_args, dynamic_shapes))
    else:
        dynamic_shapes_by_name = dynamic_shapes

    if not isinstance(dynamic_shapes_by_name, dict):
        return combined_args, dynamic_shapes

    traced_args = []
    traced_dynamic_shapes = []
    traced_names = []
    arg_i = 0
    for param in signature.parameters.values():
        if arg_i == len(args):
            break
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            var_args = combined_args[param.name]
            var_arg_dynamic_shapes = dynamic_shapes_by_name[param.name]
            for i, arg in enumerate(var_args):
                traced_args.append(arg)
                traced_dynamic_shapes.append(var_arg_dynamic_shapes[i])
                traced_names.append(f"{param.name}_{i}")
            arg_i = len(args)
            break
        if param.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            traced_args.append(combined_args[param.name])
            traced_dynamic_shapes.append(dynamic_shapes_by_name[param.name])
            traced_names.append(param.name)
            arg_i += 1

    for key, arg in kwargs.items():
        traced_args.append(arg)
        traced_names.append(key)
        if var_keyword_name is not None and key in var_kwargs:
            traced_dynamic_shapes.append(dynamic_shapes_by_name[var_keyword_name][key])
        else:
            traced_dynamic_shapes.append(dynamic_shapes_by_name[key])

    if isinstance(dynamic_shapes, dict) and len(set(traced_names)) == len(traced_names):
        return dict(zip(traced_names, traced_args)), dict(
            zip(traced_names, traced_dynamic_shapes)
        )
    return dict(enumerate(traced_args)), tuple(traced_dynamic_shapes)


class ShapesCollection:
    """
    Builder for dynamic_shapes.
    Used to assign dynamic shape specifications to tensors that appear in inputs.

    Note: this produces the ``Dim``-based ``dynamic_shapes`` format only; it does
    not (yet) emit the structured ``ShapesSpec`` / ``ParamsSpec`` API.

    This is useful particularly when :func:`args` is a nested input structure, and it's
    easier to index the input tensors, than to replicate the structure of :func:`args` in
    the :func:`dynamic_shapes` specification.

    Example::

        args = {"x": tensor_x, "others": [tensor_y, tensor_z]}

        dim = torch.export.Dim(...)
        dynamic_shapes = torch.export.ShapesCollection()
        dynamic_shapes[tensor_x] = (dim, dim + 1, 8)
        dynamic_shapes[tensor_y] = {0: dim * 2}
        # This is equivalent to the following (now auto-generated):
        # dynamic_shapes = {"x": (dim, dim + 1, 8), "others": [{0: dim * 2}, None]}

        torch.export(..., args, dynamic_shapes=dynamic_shapes)

    To specify dynamism for integers, we need to first wrap the integers using
    _IntWrapper so that we have a "unique identification tag" for each integer.

    Example::

        args = {"x": tensor_x, "others": [int_x, int_y]}
        # Wrap all ints with _IntWrapper
        mapped_args = pytree.tree_map_only(int, lambda a: _IntWrapper(a), args)

        dynamic_shapes = torch.export.ShapesCollection()
        dynamic_shapes[tensor_x] = (dim, dim + 1, 8)
        dynamic_shapes[mapped_args["others"][0]] = Dim.DYNAMIC

        # This is equivalent to the following (now auto-generated):
        # dynamic_shapes = {"x": (dim, dim + 1, 8), "others": [Dim.DYNAMIC, None]}

        torch.export(..., args, dynamic_shapes=dynamic_shapes)
    """

    def __init__(self):
        self._shapes = {}

    def __setitem__(self, t, shape):
        if not isinstance(t, (torch.Tensor, _IntWrapper)):
            raise AssertionError(
                f"Cannot assign shape to non-tensor or non-_IntWrapper type {type(t)}"
            )

        # TODO(avik): check that shape is indeed a Shape

        t_id = id(t)
        if t_id in self._shapes:
            _shape = self._shapes[t_id]
            if shape != _shape:
                raise AssertionError(
                    f"Shapes assigned to input do not match: expected {_shape}, got {shape}"
                )
        else:
            self._shapes[id(t)] = shape

    def __getitem__(self, t):
        t_id = id(t)
        if t_id not in self._shapes:
            self._shapes[t_id] = {}
        return self._shapes[t_id]

    def __len__(self):
        return len(self._shapes)

    def dynamic_shapes(self, m, args, kwargs=None):
        """
        Generates the :func:`dynamic_shapes` pytree structure according to :func:`args` and :func:`kwargs`.
        """

        t_ids = set()

        def find_shape(path, t):
            t_id = id(t)
            if t_id in self._shapes:
                t_ids.add(t_id)
                return self._shapes[t_id]
            else:
                return None

        combined_args = _combine_args(m, args, kwargs)
        dynamic_shapes = _tree_map_with_path(find_shape, combined_args)
        if any(t_id not in t_ids for t_id in self._shapes):
            raise ValueError(
                "Some tensors that were assigned shapes were not found in args. "
                "Maybe such tensors were copied when passing them as args? "
                "Maybe such tensors are contained in classes that were not registered with pytree?"
            )
        return dynamic_shapes


class AdditionalInputs:
    """
    Infers dynamic_shapes based on additional inputs.

    Note: this produces the ``Dim``-based ``dynamic_shapes`` format only; it does
    not (yet) emit the structured ``ShapesSpec`` / ``ParamsSpec`` API.

    This is useful particularly for deployment engineers who, on the one hand, may
    have access to ample testing or profiling data that can provide a fair sense of
    representative inputs for a model, but on the other hand, may not know enough
    about the model to guess which input shapes should be dynamic.

    Input shapes that are different than the original are considered dynamic; conversely,
    those that are the same as the original are considered static. Moreover, we verify
    that the additional inputs are valid for the exported program. This guarantees that
    tracing with them instead of the original would have generated the same graph.

    Example::

        args0, kwargs0 = ...  # example inputs for export

        # other representative inputs that the exported program will run on
        dynamic_shapes = torch.export.AdditionalInputs()
        dynamic_shapes.add(args1, kwargs1)
        ...
        dynamic_shapes.add(argsN, kwargsN)

        torch.export(..., args0, kwargs0, dynamic_shapes=dynamic_shapes)
    """

    def __init__(self):
        self._examples = []

    def add(self, args, kwargs=None):
        """
        Additional input :func:`args` and :func:`kwargs`.
        """

        if type(args) is not tuple:
            raise AssertionError(f"Representative args {args} must be a tuple")
        if kwargs is not None and type(kwargs) is not dict:
            raise AssertionError(
                f"Representative kwargs {kwargs} must be None or a dict"
            )
        self._examples.append((args, kwargs))

    def dynamic_shapes(self, m, args, kwargs=None):
        """
        Infers a :func:`dynamic_shapes` pytree structure by merging shapes of the
        original input :func:`args` and :func:`kwargs` and of each additional input
        args and kwargs.
        """

        dynamic_shapes, *other_dynamic_shapes = [
            _tree_map_with_path(
                lambda path, t: tuple(t.shape) if isinstance(t, torch.Tensor) else t,
                _combine_args(m, args, kwargs),
            )
            for args, kwargs in [(args, kwargs), *self._examples]
        ]

        def _mark_dynamism(v, *other_vs):
            if not all(type(v) is type(other) for other in other_vs):
                raise ValueError(
                    "The following inputs were found to have differing types, "
                    f"so they cannot be marked as dynamic: {(v,) + other_vs}."
                )

            if isinstance(v, int) and not isinstance(v, bool):
                if all(other_v == v for other_v in other_vs):
                    return None
                else:
                    return Dim.DYNAMIC
            else:
                if not all(other_v == v for other_v in other_vs):
                    raise ValueError(
                        "The following inputs were found to have differing values, "
                        f"but they cannot be marked as dynamic: {(v,) + other_vs}."
                    )
                return None

        return tree_map(
            _mark_dynamism,
            dynamic_shapes,
            *other_dynamic_shapes,
            is_leaf=lambda i: type(i) is int,
        )

    def verify(self, ep):
        """
        Verifies that an exported program is valid for each additional input.
        """

        epm = ep.module()
        for args, kwargs in self._examples:
            torch.export._unlift._check_input_constraints_for_module(
                epm, args, kwargs or {}
            )


def _warn_on_None_dynamic_shape_dimension():
    msg = (
        "Using None as a dynamic shape dimension is deprecated. "
        "Please use Dim.STATIC instead"
    )
    # TODO(avik): raise an error in the future
    log.warning(msg)


def _check_dynamic_shapes(
    combined_args: dict[str, Any],
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
):
    """
    Checks the dynamic_shapes specification for correctness,
    using combined args + kwargs as reference for inputs structure.
    """
    from torch._dynamo.exc import UserError, UserErrorType

    if dynamic_shapes is None or len(dynamic_shapes) == 0:
        return
    if isinstance(dynamic_shapes, (tuple, list)):
        combined_args = type(dynamic_shapes)(combined_args.values())  # type: ignore[assignment, misc]

    bounds: dict[str, tuple[int, int]] = {}

    def check_same_bounds(dim):
        if dim.__name__ in bounds:
            min_, max_ = bounds[dim.__name__]
            if dim.min != min_ or dim.max != max_:
                this_ = Dim._readable(dim.__name__, min_, max_)
                that_ = Dim._readable(dim.__name__, dim.min, dim.max)
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Found different definitions {this_} and {that_} "
                    f"for the same symbolic dimension {dim}!",
                )
        else:
            bounds[dim.__name__] = (dim.min, dim.max)

    def check_symbols(path, tensor, shape):
        if isinstance(shape, dict):
            for i, dim in shape.items():
                if isinstance(dim, Dim):
                    check_same_bounds(dim)
                elif dim is None:
                    _warn_on_None_dynamic_shape_dimension()
                elif not (isinstance(dim, (int, _DimHint))):
                    raise UserError(
                        UserErrorType.INVALID_INPUT,
                        f"Unexpected dimension mapped to index {i} in input tensor shape {shape} "
                        f"specified at `dynamic_shapes{keystr(path)}` "
                        f"(expected None, an int, a Dim, Dim.AUTO, Dim.STATIC, or Dim.DYNAMIC, "
                        f" but got {dim!r} instead)",
                        case_name="dynamic_shapes_validation",
                    )
        elif isinstance(shape, (tuple, list)):
            if len(shape) != len(tensor.shape):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dynamic shape spec {shape} specified at `dynamic_shapes{keystr(path)}` "
                    f"to have the same length as the actual tensor shape {tensor.shape} "
                    f"(expected {len(tensor.shape)}, but got {len(shape)} instead)",
                    case_name="dynamic_shapes_validation",
                )
            for i, dim in enumerate(shape):
                if isinstance(dim, Dim):
                    check_same_bounds(dim)
                elif dim is None:
                    _warn_on_None_dynamic_shape_dimension()
                elif not (isinstance(dim, (int, _DimHint))):
                    raise UserError(
                        UserErrorType.INVALID_INPUT,
                        f"Unexpected dimension #{i} in input tensor shape {shape} "
                        f"specified at `dynamic_shapes{keystr(path)}` "
                        f"(expected None, an int, a Dim, Dim.AUTO, Dim.STATIC, or Dim.DYNAMIC, "
                        f"but got {dim!r} instead)",
                        case_name="dynamic_shapes_validation",
                    )
        elif shape is not None:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"Unexpected input tensor shape {shape} specified at `dynamic_shapes{keystr(path)}` "
                f"(expected either a list/tuple of dimensions, or a dict mapping indices to dimensions,"
                f" where each dimension is an int, a Dim, Dim.AUTO, Dim.STATIC, or Dim.DYNAMIC)",
                case_name="dynamic_shapes_validation",
            )

    if not isinstance(dynamic_shapes, (dict, tuple, list)):
        raise AssertionError(
            f"expected dynamic_shapes to be dict, tuple, or list, got {type(dynamic_shapes)}"
        )
    if isinstance(dynamic_shapes, dict):
        got_keys = list(dynamic_shapes.keys())
        expected_arg_names = list(combined_args.keys())
        if sorted(got_keys) != sorted(expected_arg_names):
            msg = (
                f"When `dynamic_shapes` is specified as a dict, its top-level keys "
                f"must be the arg names {expected_arg_names} of `inputs`, but "
                f"here they are {got_keys}. "
            )
            if (
                len(combined_args) == 1
                and expected_arg_names[0] not in got_keys
                and isinstance(combined_args[expected_arg_names[0]], dict)
            ):
                msg += (
                    "Since here `inputs` is a list/tuple enclosing a single dict, "
                    "maybe you just forgot to enclose `dynamic_shapes` in a list/tuple?"
                )
            else:
                msg += (
                    "Alternatively, you could also ignore arg names entirely "
                    "and specify `dynamic_shapes` as a list/tuple matching `inputs`."
                )
            raise UserError(
                UserErrorType.INVALID_INPUT, msg, case_name="dynamic_shapes_validation"
            )

    def check_shape(path, t, dynamic_shape):
        if isinstance(t, torch.Tensor):
            check_symbols(path, t, dynamic_shape)
        elif isinstance(t, _IntWrapper):
            if isinstance(dynamic_shape, _Dim):
                raise ValueError(
                    "Unable to specify input integers as dynamic through named "
                    "Dims. Please use Dim.AUTO/DYNAMIC instead."
                )
            if dynamic_shape is not None and not isinstance(
                dynamic_shape, (int, _DimHint)
            ):
                raise AssertionError(
                    f"expected dynamic_shape to be None, int, or _DimHint for _IntWrapper, got {type(dynamic_shape)}"
                )
        else:
            if dynamic_shape is not None:
                rendered_path = keystr(path)
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Cannot associate shape {dynamic_shape} specified at `dynamic_shapes{rendered_path}` "
                    f"to non-tensor type {type(t)} at `inputs{rendered_path}` (expected None)",
                    case_name="dynamic_shapes_validation",
                )

    _tree_map_with_path(check_shape, combined_args, dynamic_shapes, tree_name="inputs")


def _process_dynamic_shapes(
    combined_args: dict[str, Any],
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
) -> list[Constraint]:
    """
    Reads the dynamic_shapes specification and produces a list of constraints.
    """
    from torch._dynamo.exc import UserError, UserErrorType

    if dynamic_shapes is None or len(dynamic_shapes) == 0:
        # we run with dynamic by default, so no need to produce constraints
        return []
    if isinstance(dynamic_shapes, (tuple, list)):
        combined_args = type(dynamic_shapes)(combined_args.values())  # type: ignore[assignment, misc]

    # map of Dim names representing input shape dimensions to constraints on them
    symbols: dict[str, list[Constraint]] = defaultdict(list)
    # track roots that do not directly represent input shape dimensions
    phantom_roots: dict[str, _PhantomRoot] = {}
    derived_constraints_with_phantom_root: list[_DerivedConstraint] = []
    # list of constraints to return
    constraints: list[Constraint] = []

    def to_constraint(dim, tensor, i):
        import sympy

        from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
        from torch.utils._sympy.solve import try_solve
        from torch.utils._sympy.value_ranges import ValueRanges

        def root_value():
            # given tensor.shape[i] is the value of dim = fn(root),
            # find the value of root
            symbol = sympy.Symbol(dim.root.__name__, integer=True)
            expr = dim.fn(symbol)
            solution = try_solve(sympy.Eq(expr, tensor.shape[i]), symbol)
            if solution is not None:
                return int(solution[1])
            else:
                raise UserError(
                    UserErrorType.CONSTRAINT_VIOLATION,
                    f"Expected shape[{i}] = {tensor.shape[i]} of input Tensor to be "
                    f"of the form {expr}, where {symbol} is an integer",
                )

        if isinstance(dim, _DerivedDim):
            # generate a _DerivedConstraint where the root is:
            # - either a _ConstraintTarget (if dim.root directly describes an input shape)
            # - or a _PhantomRoot (otherwise)
            dim_root = dim.root  # type: ignore[attr-defined]
            if dim_root.__name__ in symbols:
                # root represents an input shape dimension
                root_constraint = symbols[dim_root.__name__][0]
                root = _ConstraintTarget(
                    root_constraint.t_id,
                    root_constraint.dim,
                )
            elif dim_root.__name__ not in phantom_roots:
                # create a phantom root
                root = _PhantomRoot(  # type: ignore[assignment]
                    name=dim_root.__name__,
                    constraint_range=StrictMinMaxConstraint(
                        vr=ValueRanges(lower=dim_root.min, upper=dim_root.max),
                        warn_only=False,
                    ),
                    val=root_value(),
                )
                phantom_roots[dim_root.__name__] = root  # type: ignore[assignment]
            else:
                root = phantom_roots[dim_root.__name__]  # type: ignore[assignment]
            constraint = _DerivedConstraint(
                id(tensor),
                i,
                dim.__name__,
                StrictMinMaxConstraint(
                    vr=ValueRanges(lower=dim.min, upper=dim.max),
                    warn_only=False,
                ),
                root,
                dim.fn,  # type: ignore[attr-defined]
            )
            if isinstance(root, _PhantomRoot):
                # NOTE(avik): since we have not processed all inputs yet, we may replace this
                # with a root that does represent an input shape dimension later (see below)
                derived_constraints_with_phantom_root.append(constraint)
        elif isinstance(dim, _StaticDim):
            constraint = _Constraint(  # type: ignore[assignment]
                id(tensor),
                i,
                dim.__name__,
                StrictMinMaxConstraint(
                    vr=ValueRanges(lower=dim.value, upper=dim.value),  # type: ignore[attr-defined]
                    warn_only=False,
                ),
            )
        else:
            if not isinstance(dim, Dim):
                raise AssertionError(f"expected dim to be Dim, got {type(dim)}")
            constraint = _Constraint(  # type: ignore[assignment]
                id(tensor),
                i,
                dim.__name__,
                StrictMinMaxConstraint(
                    vr=ValueRanges(lower=dim.min, upper=dim.max),  # type: ignore[attr-defined]
                    warn_only=False,
                ),
            )
        return constraint

    def _parse_tensor_dim(tensor, idx, dim) -> None:
        def _create_static_dim(tensor, i, value):
            return _StaticDim(value)

        if isinstance(dim, (int, Dim)):
            if isinstance(dim, int):
                dim = _create_static_dim(tensor, idx, dim)
            constraint = to_constraint(dim, tensor, idx)
            symbols[dim.__name__].append(constraint)
        elif isinstance(dim, _DimHint):
            if dim.type == _DimHintType.AUTO:
                torch._dynamo.maybe_mark_dynamic(tensor, idx)
            elif dim.type == _DimHintType.STATIC:
                torch._dynamo.mark_static(tensor, idx)
            elif dim.type == _DimHintType.DYNAMIC:
                torch._dynamo.mark_dynamic(tensor, idx)
            constraints.append(_RelaxedConstraint(id(tensor), idx))
        elif dim is None:
            torch._dynamo.mark_static(tensor, idx)

    def update_symbols(path, tensor, shape):
        # clean out decorators from user side, or previous export call
        # we also delete these attributes in non_strict_utils.py/make_constraints()
        tensor._dynamo_weak_dynamic_indices = set()
        tensor._dynamo_dynamic_indices = set()
        tensor._dynamo_dynamic_range = set()
        tensor._dynamo_static_indices = set()
        tensor._dynamo_unbacked_indices = set()

        if isinstance(shape, dict):
            for i, dim in shape.items():
                _parse_tensor_dim(tensor, i, dim)
        elif isinstance(shape, (tuple, list)):
            for i, dim in enumerate(shape):
                _parse_tensor_dim(tensor, i, dim)
        elif shape is None:
            for i in range(tensor.dim()):
                _parse_tensor_dim(tensor, i, None)

    def assoc_shape(path, t, dynamic_shape):
        if isinstance(t, torch.Tensor):
            update_symbols(path, t, dynamic_shape)
        elif isinstance(t, _IntWrapper):
            # If tensor dimensions are marked as dynamic, the tensors themselves
            # get marked using mark_dynamic. However since we can't mark
            # integers as dynamic, we first wrap integers in this class, and
            # then set the `dim` field of the class with the dynamic shapes dim
            # to mark the integer as dynamic.
            t.dynamism = dynamic_shape

    _tree_map_with_path(assoc_shape, combined_args, dynamic_shapes, tree_name="inputs")

    for derived_constraint_with_phantom_root in derived_constraints_with_phantom_root:
        phantom_root_name = derived_constraint_with_phantom_root.root.name  # type: ignore[union-attr]
        if phantom_root_name in symbols:
            # We found an input shape dimension corresponding to this name, so we
            # do not need a phantom symbol for it after all.
            # NOTE(avik): Overall we want to maintain the invariant that roots that
            # are phantom symbols are really "phantom," i.e., they cannot be represented
            # by any input source. This is important when we are deciding derived equalities,
            # since we can focus our attention exclusively on input sources: deciding
            # derived equalities involving phantom symbols are, in comparison, trivial.
            derived_constraint_with_phantom_root.root = symbols[phantom_root_name][0]

    for dynamic_dims in symbols.values():
        constraints.extend(dynamic_dims)

    return constraints


def _get_dim_name_mapping(
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
):
    name_to_dim = {}
    for dim in tree_iter(dynamic_shapes, is_leaf=lambda x: isinstance(x, Dim)):
        if dim is None:
            # NOTE: this must denote a non-Tensor or automatic at this point.
            continue
        if isinstance(dim, int):
            continue
        elif isinstance(dim, Dim):
            name_to_dim[dim.__name__] = dim
            if isinstance(dim, _DerivedDim):
                name_to_dim[dim.root.__name__] = dim.root  # type: ignore[attr-defined]
        else:
            if not isinstance(dim, _DimHint):
                raise AssertionError(f"expected dim to be _DimHint, got {type(dim)}")
    return name_to_dim


def refine_dynamic_shapes_from_suggested_fixes(
    msg: str,
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any],
) -> dict[str, Any] | tuple[Any] | list[Any]:
    """
    When exporting with :func:`dynamic_shapes`, export may fail with a ConstraintViolation error if the specification
    doesn't match the constraints inferred from tracing the model. The error message may provide suggested fixes -
    changes that can be made to :func:`dynamic_shapes` to export successfully.

    Example ConstraintViolation error message::

        Suggested fixes:

            dim = Dim('dim', min=3, max=6)  # this just refines the dim's range
            dim = 4  # this specializes to a constant
            dy = dx + 1  # dy was specified as an independent dim, but is actually tied to dx with this relation

    This is a helper function that takes the ConstraintViolation error message and the original :func:`dynamic_shapes` spec,
    and returns a new :func:`dynamic_shapes` spec that incorporates the suggested fixes.

    Example usage::

        try:
            ep = export(mod, args, dynamic_shapes=dynamic_shapes)
        except torch._dynamo.exc.UserError as exc:
            new_shapes = refine_dynamic_shapes_from_suggested_fixes(
                exc.msg, dynamic_shapes
            )
            ep = export(mod, args, dynamic_shapes=new_shapes)

    """

    import re

    import sympy

    from torch._dynamo.exc import UserError, UserErrorType
    from torch.fx.experimental.symbolic_shapes import _is_supported_equivalence

    try:
        shape_fixes_msg = msg.split("Suggested fixes:")[1].strip()
    except Exception as exc:
        raise UserError(
            UserErrorType.INVALID_INPUT,
            "Suggested fixes not found in error message given to refine_dynamic_shapes_from_suggested_fixes()",
        ) from exc

    # build shape_fixes dictionary
    shape_fixes = {}
    for fix in shape_fixes_msg.split("\n"):
        fix = fix.strip()
        if match := re.match(r"(.*) = Dim\('(.*)'.*\)", fix):
            name = match.group(1)
            _min, _max = None, None
            if match_min := re.match(r".* = Dim\('.*', min\=([0-9]+).*\)", fix):
                _min = int(match_min.group(1))
            if match_max := re.match(r".* = Dim\('.*'.*max\=([0-9]+)\)", fix):
                _max = int(match_max.group(1))
            shape_fixes[name] = Dim(name, min=_min, max=_max)
        else:
            name, expr = fix.split(" = ")
            expr = sympy.sympify(expr)
            if isinstance(expr, sympy.Number):
                # static, integer
                shape_fixes[name] = int(expr)  # type: ignore[assignment]
            else:
                # relation or derived dim
                shape_fixes[name] = expr

    name_to_dim = _get_dim_name_mapping(dynamic_shapes)

    # track derived dim roots
    roots: set[str] = set()
    for k, c in shape_fixes.items():
        if not isinstance(c, (int, Dim, _DerivedDim, sympy.Expr)):
            raise AssertionError(
                f"expected shape_fixes[{k!r}] to be int, Dim, _DerivedDim, or sympy.Expr, got {type(c)}"
            )
        if isinstance(c, sympy.Expr):  # check dim/derived dim expression
            if not _is_supported_equivalence(c):
                raise AssertionError(f"sympy.Expr {c} is not a supported equivalence")
            shape_fixes[k] = c
            roots.add(str(next(iter(c.free_symbols))))
        if isinstance(c, _DerivedDim):
            roots.add(c.root.__name__)  # type: ignore[attr-defined]

    # check keys are existing dims or new roots
    for k in shape_fixes:
        if k not in name_to_dim and k not in roots:
            raise AssertionError(
                f"shape_fixes key {k!r} not found in name_to_dim or roots"
            )

    # cache so we don't produce multiple derived dim objects
    derived_dim_cache: dict[str, _DerivedDim] = {}

    def apply_fixes(path, dim, dummy):
        if dim is None or isinstance(dim, int):  # not dynamic
            return dim
        elif dim.__name__ in shape_fixes:  # directly fix
            fix = shape_fixes[dim.__name__]
            if isinstance(fix, sympy.Expr):  # now derived or related
                if str(fix) in derived_dim_cache:
                    return derived_dim_cache[str(fix)]
                else:
                    symbol = next(iter(fix.free_symbols))
                    # try to locate symbol
                    if symbol.name in shape_fixes:
                        root = shape_fixes[symbol.name]
                    else:
                        if symbol.name not in name_to_dim:
                            raise AssertionError(
                                f"symbol.name {symbol.name!r} not found in name_to_dim"
                            )
                        root = name_to_dim[symbol.name]
                    # figure out value of fix
                    modulus, remainder = sympy.polys.polytools.div(fix, symbol)
                    dim = root
                    if modulus != 1:
                        dim = int(modulus) * dim
                    if remainder != 0:
                        dim = dim + int(remainder)
                    # pyrefly: ignore [unsupported-operation]
                    derived_dim_cache[str(fix)] = dim
                    return dim
            else:
                return fix
        elif isinstance(dim, _DerivedDim) and dim.root.__name__ in shape_fixes:  # type: ignore[attr-defined]
            if dim.__name__ in derived_dim_cache:
                return derived_dim_cache[dim.__name__]
            else:  # evaluate new derived value based on root
                _dim = dim.fn(shape_fixes[dim.root.__name__])  # type: ignore[attr-defined]
                derived_dim_cache[dim.__name__] = _dim
                return _dim
        return dim  # unchanged dim

    return _tree_map_with_path(apply_fixes, dynamic_shapes, dynamic_shapes)
<|endoftext|># mypy: allow-untyped-defs
import importlib

import torch


lib = torch.library.Library("export", "FRAGMENT")

lib.define(
    "access_subclass_inner_tensor(Tensor src_subclass_tensor, str attr) -> Tensor"
)


@torch.library.impl(lib, "access_subclass_inner_tensor", "Autograd")
# When running under torch.inference_mode(), we seem to skip Autograd key
# so we should desugar this op as soon as we start tracing to post-dispatch.
@torch.library.impl(lib, "access_subclass_inner_tensor", "Python")
def _access_subclass_inner_tensor(
    src_subclass_tensor: torch.Tensor, attr: str
) -> torch.Tensor:
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    if not is_traceable_wrapper_subclass(src_subclass_tensor):
        raise AssertionError(
            f"Expected src_subclass_tensor to be a traceable wrapper subclass, "
            f"but got {type(src_subclass_tensor)}"
        )
    val = getattr(src_subclass_tensor, attr, None)
    if val is None or not isinstance(val, torch.Tensor):
        raise RuntimeError(
            f"Attribute {attr} is not a tensor or doesn't exist in {src_subclass_tensor}"
        )
    return val


def _call_custom_autograd_function_in_pre_dispatch(function_cls_name, *args, **kwargs):
    """
    Import a custom autograd function by string name and call it. This is pretty bad
    because:
    1) There is no schema

    Ideally we should automatically wrap custom autograd functions with a custom op, but
    that is too much work because we need to schematize custom autograd functions. For now,
    we just hackily put it in the IR.
    """
    # Parse module and class name
    module_name, class_name = function_cls_name.rsplit(".", 1)

    # Import the module and get the class
    module = importlib.import_module(module_name)
    function_cls = getattr(module, class_name)
    if not hasattr(function_cls, "apply"):
        raise AssertionError(
            f"Expected function class {function_cls_name} to have 'apply' method"
        )
    return function_cls.apply(*args, **kwargs)
<|endoftext|>import torch


class _WrapperModule(torch.nn.Module):
    def __init__(self, f):  # type: ignore[no-untyped-def]
        super().__init__()
        self.f = f

    def forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.f(*args, **kwargs)
<|endoftext|># Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    auto_functionalized_v2,
)
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized
from torch.export import ExportedProgram
from torch.fx import Graph


def remove_self_clone(graph: Graph) -> None:
    for node in graph.nodes:
        if node.target is torch.ops.aten.copy_.default and node.args[0] == node.args[1]:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)


def unsafe_remove_auto_functionalized_pass(
    ep: ExportedProgram,
) -> ExportedProgram:
    """
    This pass removes an instances of the higher order op 'auto_functionalized',
    and modifies the calling EP inplace to have the original mutator op.
    This pass doesn't perform safety checks to make sure that this inplace mutation is safe.
    """

    with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
        for module in ep.graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in ep.graph.nodes:
                if (
                    node.op == "call_function" and node.target is auto_functionalized
                ) or (
                    node.op == "call_function" and node.target is auto_functionalized_v2
                ):
                    func = node.args[0]
                    if not isinstance(func, torch._ops.OpOverload):
                        raise AssertionError(
                            f"Expected func to be an OpOverload, but got {type(func)}"
                        )
                    # re-inplace everything
                    node.meta["only_clone_these_tensors"] = []
            decompose_auto_functionalized(ep.graph)
            remove_self_clone(ep.graph)
            ep.graph.eliminate_dead_code()

    return ep
<|endoftext|>from collections.abc import Callable
from typing import Any

from torch.utils._pytree import Context, TreeSpec


def reorder_kwargs(user_kwargs: dict[str, Any], spec: TreeSpec) -> dict[str, Any]:
    """Reorder user-provided kwargs to match the order in `spec`. `spec` is
    expected to be the in_spec of an exported program, i.e. the spec that
    results from flattening `(args, kwargs)`.

    We need this to provide consistent input ordering, such so that users can
    pass in foo(a=a, b=b) OR foo(b=b, a=a) and receive the same result.
    """
    # Make sure that the spec is actually shaped like (args, kwargs)
    if spec.type is not tuple:
        raise AssertionError(f"Expected spec type to be tuple, but got {spec.type}")
    if spec.num_children != 2:
        raise AssertionError(
            f"Expected spec to have 2 children, but got {spec.num_children}"
        )
    kwargs_spec = spec.child(1)
    if kwargs_spec.type is not dict:
        raise AssertionError(
            f"Expected kwargs_spec type to be dict, but got {kwargs_spec.type}"
        )

    if set(user_kwargs) != set(kwargs_spec.context):
        raise ValueError(
            f"Ran into a kwarg keyword mismatch: "
            f"Got the following keywords {list(user_kwargs)} but expected {kwargs_spec.context}"
        )

    reordered_kwargs = {}
    for kw in kwargs_spec.context:
        reordered_kwargs[kw] = user_kwargs[kw]

    return reordered_kwargs


def is_equivalent(
    spec1: TreeSpec,
    spec2: TreeSpec,
    equivalence_fn: Callable[[type | None, Context, type | None, Context], bool],
) -> bool:
    """Customizable equivalence check for two TreeSpecs.

    Arguments:
        spec1: The first TreeSpec to compare
        spec2: The second TreeSpec to compare
        equivalence_fn: A function to determine the equivalence of two
            TreeSpecs by examining their types and contexts. It will be called like:

                equivalence_fn(spec1.type, spec1.context, spec2.type, spec2.context)

            This function will be applied recursively to all children.

    Returns:
        True if the two TreeSpecs are equivalent, False otherwise.
    """
    if not equivalence_fn(spec1.type, spec1.context, spec2.type, spec2.context):
        return False

    # Recurse on children
    if spec1.num_children != spec2.num_children:
        return False

    for child_spec1, child_spec2 in zip(spec1.children(), spec2.children()):
        if not is_equivalent(child_spec1, child_spec2, equivalence_fn):
            return False

    return True
<|endoftext|># mypy: allow-untyped-defs
import copy
import inspect
import math
import warnings
from collections.abc import Sequence
from itertools import chain
from typing import Any

import sympy

import torch
import torch.utils._pytree as pytree
from torch._export.non_strict_utils import (
    _enter_enable_graph_inputs_of_type_nn_module,
    _exit_enable_graph_inputs_of_type_nn_module,
    _get_graph_inputs_of_type_nn_module,
)
from torch._export.passes.add_runtime_assertions_for_constraints_pass import (
    _convert_range_to_int,
)
from torch._export.utils import _check_input_constraints_for_graph
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx.experimental.proxy_tensor import _pytree_subclasses_that_lose_info
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo
from torch.fx.traceback import NodeSource, NodeSourceAction
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import ValueRanges

from ._remove_effect_tokens_pass import _remove_effect_tokens
from ._tree_utils import reorder_kwargs
from .exported_program import (
    ExportedProgram,
    ExportGraphSignature,
    InputKind,
    OutputKind,
)


def eq_spec(self: pytree.TreeSpec, other: pytree.TreeSpec) -> bool:
    """
    Refinement of TreeSpec.__eq__ where, e.g., torch.Size(...) matches tuple(...).
    See _pytree_subclasses_that_lose_info in proxy_tensor.py for more details.
    """

    def _normalize_type(t):
        return str(_pytree_subclasses_that_lose_info.get(t, t))

    def _match_normalized_structure(a, b):
        if a is b:
            return True
        if _normalize_type(a.type) != _normalize_type(b.type):
            return False
        if a.type is dict and b.type is dict:
            # in the case of dict, the context is list of keys and we allow the keys to be in any order
            if set(a.context) != set(b.context):
                return False
        elif a.context != b.context:
            return False
        if a.num_children != b.num_children:
            return False
        return all(
            _match_normalized_structure(a, b)
            for a, b in zip(a.children(), b.children())
        )

    return _match_normalized_structure(self, other)


def _check_inputs_match(args, kwargs, in_spec: pytree.TreeSpec) -> list:
    reordered_kwargs = reorder_kwargs(kwargs, in_spec)
    flat_args_with_path, received_spec = pytree.tree_flatten_with_path(
        (args, reordered_kwargs)
    )

    if not eq_spec(received_spec, in_spec):
        raise ValueError(
            "Trying to flatten user inputs with exported input tree spec: \n"
            f"{in_spec}\n"
            "but actually got inputs with tree spec of: \n"
            f"{received_spec}.\n"
            "Please check that the inputs have the same number and type of "
            "args and kwargs as the ones you used when tracing."
        )

    return flat_args_with_path


def _force_ep_signature_match(ep_guards_code: list[str], input_paths):
    # TODO (tmanlaibaatar)
    # This is band-aid solution to export new tracer replacing
    # shape env sources to flat_args. The real fix should be replacing
    # shape env sources to original user sources but this is quite
    # involved because you need to carefully construct new sources using
    # dynamo and replace all instances of it inside shape env. But it is
    # lot easier to manipulate after we turn them into strings and only
    # time we use these guards is during retracing or running exported program,
    # so it is probably ok to have "not useful" guards on ep for now.
    name_mapping = {}
    for idx, path in enumerate(input_paths):
        name_mapping[f"L['flat_args'][{idx}]"] = f"L{pytree.keystr(path)}"

    new_guards_code = []
    for guard in ep_guards_code:
        for old_name, new_name in name_mapping.items():
            guard = guard.replace(old_name, new_name)
        new_guards_code.append(guard)

    return new_guards_code


def _force_gm_signature_match(ep_guards_code: list[str], signature):
    """
    The signature of the originally exported module may not match
    the signature of the unlifted graph module extracted from the
    exported program. The guards code extracted from the exported
    program is based on the former, but the generated guards fn is
    based on the latter; thus we need to reconcile any such diff.
    """

    import re

    # Handle case where signatures may differ in var args.
    orig_arg_names = set()
    for g in ep_guards_code:
        # match substrings of the form L['<name>'][<number>]
        orig_arg_names.update(re.findall(r"L\[\'([^\']+)\'\]\[([0-9]+)\]", g))

    sig_arg_names = set()
    for n in signature.parameters:
        # match substrings of the form <name>_<number>
        sig_arg_names.update(re.findall(r"(.+)_([0-9]+)", n))

    # replace L['<name>'][<number>] with L['<name>_<number>']
    new_guards_code = ep_guards_code
    for match in orig_arg_names:
        if match in sig_arg_names:
            base, idx = match
            new_guards_code = [
                g.replace(f"L['{base}'][{idx}]", f"L['{base}_{idx}']")
